use std::collections::HashMap;
use std::ops::Range;
use std::sync::mpsc::channel;
use crate::basis_io::{spheric_gto_value_serial, spheric_gto_1st_value_serial};
use crate::scf_io::SCF;
use crate::scf_io::SCFType;
use crate::{geom_io,dft,molecule_io, basis_io, utilities};
use crate::dft::Grids as dftgrids;
use crate::molecule_io::Molecule;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rayon::prelude::{IntoParallelRefMutIterator, IntoParallelRefIterator, IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
use rest_tensors::{MatrixFull, RIFull, ERIFull};
use tensors::external_libs::matr_copy_from_ri;
use tensors::{TensorSlice, TensorSliceMut};
use std::cmp::Ordering;
//use rand::distributions::normal::StandardNormal;
use dft::gen_grids;
use crate::utilities::balancing;
use crate::geom_io::get_mass_charge;
mod lib;
use tensors::matrix_blas_lapack::{omp_get_num_threads_wrapper,omp_set_num_threads_wrapper,_dgemm_full,_qrpinv};
use itertools::Itertools;
use num_traits::ToPrimitive;
use std::ffi::{c_char,c_double,c_int};
use std::process::Command;

/* Given grid points and center of each clusters, classify points to nearest cluster centers.
Input:
Rgrid, c_mu(original point)
output: 
    ind_R, dist_R */
pub fn cvt_classification(rgrids: &Vec<[f64;3]>, lambda_r: &Vec<f64>, c_mu: &Vec<[f64;3]>) -> (Vec<usize>,Vec<f64>){
    //consider par, tested
    let num_grids = lambda_r.len();  
    let mut ind_r:Vec<usize> = vec![0; num_grids];
    let mut dist_r:Vec<f64> = vec![0.0;num_grids];
    let num_mu = c_mu.len();

    let (sender , receiver) = channel();
    rgrids.par_iter().enumerate().for_each_with(sender , |s, (i,rgrids_i)|{
        let mut r_c_mu:Vec<f64> = vec![0.0; num_mu];
        for j in 0..c_mu.len(){
            r_c_mu[j] = rgrids_i.iter().zip(c_mu[j].iter()).fold(0.0,|r,(ac,gc)| {r + (ac-gc).powf(2.0)}).sqrt();
        }
        let ind = index_of_min(&mut r_c_mu);
        let dist = r_c_mu[ind];
        s.send((i,ind, dist)).unwrap();
    });
    receiver.into_iter().for_each(|(i, ind, dist)|{
        ind_r[i] = ind;
        dist_r[i] =dist;
    });

    (ind_r, dist_r)
}

/*  Given classification for grid points, regenerate cluster centers by taking mean value.
    Input: Rgrid, ind_R(from CVT_classification), n_mu, lambda_R, c_mu_old
    Output: c_mu_new(after update)
    Update formula:
        If no points corresponds to it: stay there.
        If some points corresponds to it: update normally */
pub fn cvt_update_cmu(rgrids: &Vec<[f64;3]>, lambda_r: &Vec<f64>, c_mu_old: &Vec<[f64;3]>, ind_r: &Vec<usize>) -> Vec<[f64;3]>{
    let num_mu = c_mu_old.len();
    let num_grids = lambda_r.len(); 
    let mut c_mu_save = c_mu_old.clone();

    //Added together
    let mut c_mu_tmp:Vec<[f64; 3]> = vec![[0.0;3];num_mu];
    let mut weight_sum: Vec<f64> = vec![0.0; num_mu];

    let (sender, receiver) = channel();
    rgrids.par_iter().zip(lambda_r.par_iter()).zip(ind_r.par_iter())
        .for_each_with(sender, |s,((rgrids_i,lambda_i),ind_r_i)| {
        let mut c_mu_tmp = [0.0_f64;3];
        c_mu_tmp.iter_mut().zip(rgrids_i.iter()).for_each(|(x,y)| {
            *x += lambda_i * y;
        });
        s.send((c_mu_tmp, lambda_i, ind_r_i)).unwrap()

    });
    receiver.into_iter().for_each(|(c_mu_tmp_i, lambda_i, ind_r_i)| {
        c_mu_tmp[*ind_r_i].iter_mut().zip(c_mu_tmp_i.iter()).for_each(|(x,y)| {*x += y});
        weight_sum[*ind_r_i] += lambda_i;
    });

    //Renew c_mu
    let non_zero_ind = weight_sum.iter()
    .enumerate()
    .filter(|(_, &r)| r >= 1e-8)
    .map(|(index, _)| index)
    .collect::<Vec<_>>();//1e-8 could be adjusted.
    for i in 0..non_zero_ind.len(){
        c_mu_save[non_zero_ind[i]].iter_mut().zip(c_mu_tmp[non_zero_ind[i]].iter()).for_each(|(x,y)|{
            *x = y / weight_sum[non_zero_ind[i]];
        })
    }
    c_mu_save
}

/* find points in Rgrid minimizing 2-norm to c_mu[i]
ind_sorted: 归属于哪一个cluster */
pub fn cvt_find_corresponding_point(rgrids: &Vec<[f64;3]>, lambda_r: &Vec<f64>, c_mu: &Vec<[f64;3]>, ind_r: &Vec<usize>) -> Vec<usize> {
    
    let num_mu = c_mu.len();
    let num_grids = lambda_r.len(); 
    let mut ind_mu: Vec<usize> = vec![0; num_mu];;

    //find nearest center
    let mut min_dist = vec![1.0e10; num_mu];
  
    //find those whose clusters have been changed
    for i in 0..num_grids{
        let mut value = rgrids[i].iter().zip(c_mu[ind_r[i]].iter()).fold(0.0,|r,(ac,gc)| {r + (ac-gc).powf(2.0)}).sqrt();
        if value < min_dist[ind_r[i]]{
            ind_mu[ind_r[i]] = i ;
            min_dist[ind_r[i]] = value;
        }
    }
    let need_search_full_rgrid = ind_mu.iter().enumerate().filter(|(_, &r)| r == 0).map(|(index,_)|index).collect::<Vec<_>>();
    for i in need_search_full_rgrid{
        let mut dist_to_all_grids: Vec<f64> = vec![0.0; num_grids];
        dist_to_all_grids.iter_mut().zip(rgrids).for_each(|(x,y)|{
            *x = y.iter().zip(c_mu[i].iter()).fold(0.0,|r,(ac,gc)| {r + (ac-gc).powf(2.0)}).sqrt();
        });
        ind_mu[i] = index_of_min(&mut dist_to_all_grids);   
    }


    ind_mu
}

pub fn cvt_isdf_v2(rgrids_old: &Vec<[f64;3]>, lambda_r_old: &Vec<f64>, n_mu: usize) ->  (Vec<[f64;3]>, Vec<f64>){
    // 筛选权重在阈值之上的格点并存为新格点rgrids
    let threshold = 1.0e-8;
    let effective_ind = lambda_r_old.iter()
    .enumerate()
    .filter(|(_, &r)| r >= threshold)
    .map(|(index, _)| index)
    .collect::<Vec<_>>();
    if n_mu >= effective_ind.len() {
        println!("There are {} effective index. N_mu is {}.", effective_ind.len(), &n_mu);
        panic!("n_mu is smaller than effective_ind!");
    }

    let ngrids = effective_ind.len();
    let mut lambda_r = vec![0.0; effective_ind.len()];
    lambda_r.iter_mut().zip(effective_ind.iter()).for_each(|(new,index_new)|{
        *new = lambda_r_old[*index_new];
    });
    let mut rgrids = vec![[0.0;3]; effective_ind.len()];
    rgrids.iter_mut().zip(effective_ind.iter()).for_each(|(new,index_new)|{
        new.iter_mut().zip(rgrids_old[*index_new].iter()).for_each(|(a,b)|{
            *a = *b;
        }) 
    });
    
    // 按高斯分布生成随机点
    /* let mut rng = rand::thread_rng();
    let mut c_mu: Vec<[f64;3]> = vec![[0.0;3];n_mu];
    c_mu.iter_mut().for_each(|[x,y,z]|{
        let StandardNormal(a) = rand::random();
        *x =a;
        
        let StandardNormal(b) = rand::random();
        *y =b;
        let StandardNormal(c) = rand::random();
        *z =c;
    }); */

    //类manual_seed
    let seed: [usize; 64] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64];
    let seed_bytes = std::array::from_fn::<u8, 32, _>(|i| seed[i] as u8);
    let mut rng: StdRng = SeedableRng::from_seed(seed_bytes);
    let mut c_mu: Vec<[f64;3]> = vec![[0.0;3];n_mu];
    for i in 0..n_mu {
        let mut random_number: u32 = rng.random_range(0u32..lambda_r.len() as u32);
        c_mu[i] = rgrids[random_number as usize]
        
    }

    // get generators
    let max_iter = 300;
    let mut class_result = (vec![0usize; n_mu],vec![0.0; n_mu]);
    let mut ind_r = vec![0usize; n_mu];
    let mut dist_r = vec![0.0; n_mu];
    let mut count = 0;
    let mut dist = 0.0;
    let mut c_mu_new = vec![[0.0;3];n_mu];
    let result = loop{
        class_result = cvt_classification(&rgrids, &lambda_r, &c_mu);
        ind_r = class_result.0;
        dist_r = class_result.1;
        let mut init_dist_r = 0.0;
        dist_r.iter().zip(lambda_r.iter()).for_each(|(d, w)|{
            init_dist_r += w * d * d;
        });
        dist = init_dist_r.sqrt();
        
        let mut c_mu_new = cvt_update_cmu(&rgrids, &lambda_r, &c_mu, &ind_r);
        let mut sum = 0.0;
        
        if count == 0 {
        }
        if count == max_iter - 1 {
            c_mu_new.iter().zip(c_mu.iter()).for_each(|([x,y,z],[a,b,c])|{
                sum += ((x-a) * (x-a) + (y-b) * (y-b) + (z-c) * (z-c)).sqrt();
            });
            c_mu = c_mu_new;
            let ind_mu = cvt_find_corresponding_point(&rgrids, &lambda_r, &c_mu, &ind_r);
            break (ind_mu, sum);
        } 

        c_mu_new.iter().zip(c_mu.iter()).for_each(|([x,y,z],[a,b,c])|{
            sum += ((x-a).powf(2.0) + (y-b).powf(2.0) + (z-c).powf(2.0)).sqrt();
        });
        
        if sum <= 1e-6{
            c_mu = c_mu_new;
            let ind_mu = cvt_find_corresponding_point(&rgrids, &lambda_r, &c_mu, &ind_r);
            println!("Random points converged after {} iterations.", &count-1);
            println!("Dist: {:?}", &dist);
            break (ind_mu, sum);
        }else {
            c_mu = c_mu_new;
            count += 1;     
        }
    };

    let mut inter_points = vec![[0.0;3]; n_mu];
    let mut inter_weights = vec![0.0; n_mu];
    inter_points.iter_mut().zip(inter_weights.iter_mut().zip(result.0.iter())).for_each(|(p, (w,i))|{
        p.iter_mut().zip(rgrids[*i].iter()).for_each(|(x,y)|{*x = *y});
        *w = lambda_r[*i];

    });     
    (inter_points, inter_weights)
    
} 


pub fn prod_states_gw (phi: &MatrixFull<f64>, psi: &MatrixFull<f64>) -> MatrixFull<f64>{
    //Generate P: n_1n_2 \times n_r, P_{ij}(r) = \phi_i(r)\psi_j(r)
    //psi, phi: two wfn matrix
    // (nmu,nao,nao) 
    // (k, j, i)
    //(0,0,0),   (1,0,0),   (2,0,0) ...   (nmu,0,0)
    //(0,1,0),   (1,1,0),   (2,1,0) ...   (nmu,1,0)
    // ....
    //(0,nao,0), (1,nao,0), (2,nao,0) ... (nmu,nao,0)
    //
    // (0,0,i), (1,0,i), (2,0,i) ... (nmu,0,i)
    if phi.size[0] != psi.size[0]{
        panic!("Wrong inputs: row dimensions of Phi and Psi do not match!\n");
    }
    
    let mut prod = MatrixFull::new([phi.size[0], phi.size[1]*psi.size[1]],0.0);
    for i in 0..phi.size[1]{
        for j in 0..psi.size[1]{
            for k in 0..phi.size[0]{
                prod.data[k+phi.size[0]*(j+i*psi.size[1])] = phi.data[k+i*phi.size[0]] * psi.data[k+j*phi.size[0]];
            }
        }
    }
    prod
}

pub fn fourcenter_after_isdf(k_mu: usize, mol: &Molecule, grids: &dft::Grids) -> MatrixFull<f64> {
    let nao = mol.num_basis;
    let nri = mol.num_auxbas;

    let lambda_r = &grids.weights;
    let ngrids = lambda_r.len();
    let rgrids = &grids.coordinates;
    let mut phi = tabulated_ao(&mol, &rgrids);

    let n_mu = k_mu * nri;
    let mut lambda_r_for_isdf = vec![0.0; ngrids];
    lambda_r_for_isdf.iter_mut().zip(lambda_r.iter()).for_each(|(x,y)|{
        *x = y.abs();
    });
    let mut lambda_phi = MatrixFull::new([nao,ngrids],0.0);
    lambda_phi.iter_columns_full_mut().zip(lambda_r.iter().zip(phi.iter_columns_full()))
    .for_each(|(x, (weight,aos))|{
        x.iter_mut().zip(aos.iter()).for_each(|(y,ao)|{
            *y = weight * ao;
        });
    });

    let (ip, ip_weights) = cvt_isdf_v2(&rgrids, &lambda_r_for_isdf, n_mu);
    let mut varphi = tabulated_ao(&mol, &ip);
    let mut lambda_varphi = MatrixFull::new([nao,n_mu],0.0);
    lambda_varphi.iter_columns_full_mut().zip(ip_weights.iter().zip(varphi.iter_columns_full()))
    .for_each(|(x, (weight,aos))|{
        x.iter_mut().zip(aos.iter()).for_each(|(y,ao)|{
            *y = weight * ao;
        });
    });

    //C1 = (lambda_phi.T \cdot lambda_varphi) \times (phi.T \cdot varphi.T)  \times: Hadmard
    //C2 = (lambda_varphi.T \cdot lambda_varphi) \times (varphi.T \cdot varphi)
    /* let mut c11 = MatrixFull::new([ngrids, ind_mu.len()], 0.0);
    c11.lapack_dgemm(&mut lambda_phi, &mut lambda_varphi, 'T', 'N', 1.0, 0.0);
    let mut c12 = MatrixFull::new([ngrids, ind_mu.len()], 0.0);
    c12.lapack_dgemm(&mut phi, &mut varphi, 'T', 'N', 1.0, 0.0);
    let mut c1 = MatrixFull::new([ngrids, ind_mu.len()], 0.0);
    c1.iter_columns_full_mut().zip(c11.iter_columns_full()).zip(c12.iter_columns_full())
        .for_each(|((x,a),b)|{
            x.iter_mut().zip(a.iter()).zip(b.iter()).for_each(|((x1,a1),b1)|{
                *x1 = a1 *b1;
            });
        }); */

    let mut c21 = MatrixFull::new([n_mu, n_mu], 0.0);
    let mut lambda_varphi_mid = lambda_varphi.clone();
    c21.lapack_dgemm(&mut lambda_varphi, &mut lambda_varphi_mid, 'T', 'N', 1.0, 0.0);
    let mut c22 = MatrixFull::new([n_mu, n_mu], 0.0);
    let mut varphi_mid = varphi.clone();
    c22.lapack_dgemm(&mut varphi, &mut varphi_mid, 'T', 'N', 1.0, 0.0);
    let mut c2 = MatrixFull::new([n_mu, n_mu], 0.0);
    c2.iter_columns_full_mut().zip(c21.iter_columns_full()).zip(c22.iter_columns_full())
        .for_each(|((x,a),b)|{
            x.iter_mut().zip(a.iter()).zip(b.iter()).for_each(|((x1,a1),b1)|{
                *x1 = a1 * b1;
            });
        });
    //<Z|V|P><P|V|P><P|V|Z>
    let mut cint_data = mol.initialize_cint(true);
    let n_basis_shell = mol.cint_bas.len();
    let n_auxbas_shell = mol.cint_aux_bas.len();
    let mut ri3fn = RIFull::new([nao,nao,nri],0.0);
    cint_data.cint2c2e_optimizer_rust();
    let mut ri_v_ri = MatrixFull::new([nri,nri],0.0);
    for l in 0..n_auxbas_shell {
        let basis_start_l = mol.cint_aux_fdqc[l][0];
        let basis_len_l = mol.cint_aux_fdqc[l][1];
        let gl  = l + n_basis_shell;
        for k in 0..n_auxbas_shell {
            let basis_start_k = mol.cint_aux_fdqc[k][0];
            let basis_len_k = mol.cint_aux_fdqc[k][1];
            let gk  = k + n_basis_shell;
            let buf = cint_data.cint_2c2e(gk as i32, gl as i32);
            
            let mut tmp_slices = ri_v_ri.iter_submatrix_mut(
                basis_start_k..basis_start_k+basis_len_k,
                basis_start_l..basis_start_l+basis_len_l);
            tmp_slices.zip(buf.iter()).for_each(|value| {*value.0 = *value.1});

        }
    }
    cint_data.cint3c2e_optimizer_rust();
    for k in 0..n_auxbas_shell {
        let basis_start_k = mol.cint_aux_fdqc[k][0];
        let basis_len_k = mol.cint_aux_fdqc[k][1];
        let gk  = k + n_basis_shell;
        for j in 0..n_basis_shell {
            let basis_start_j = mol.cint_fdqc[j][0];
            let basis_len_j = mol.cint_fdqc[j][1];
            // can be optimized with "for i in 0..(j+1)"
            for i in 0..n_basis_shell {
                let basis_start_i = mol.cint_fdqc[i][0];
                let basis_len_i = mol.cint_fdqc[i][1];
                let buf = RIFull::from_vec([basis_len_i, basis_len_j,basis_len_k], 
                    cint_data.cint_3c2e(i as i32, j as i32, gk as i32)).unwrap();
                ri3fn.copy_from_ri(
                    basis_start_i..basis_start_i+basis_len_i,
                    basis_start_j..basis_start_j+basis_len_j,
                    basis_start_k..basis_start_k+basis_len_k,
                    & buf, 
                    0..basis_len_i, 
                    0..basis_len_j, 
                    0..basis_len_k);
            }
        }
    }
        cint_data.final_c2r();

    let mut ri_v_ao_t = MatrixFull::from_vec([nao*nao, nri],ri3fn.data).unwrap();
    let mut c = prod_states_gw(&lambda_varphi.transpose(), &varphi.transpose());
    let mut tmp1 = MatrixFull::new([nri, n_mu],0.0);

    tmp1.lapack_dgemm(&mut ri_v_ao_t, &mut c, 'T', 'T', 1.0, 0.0);
    let mut tmp0 = MatrixFull::new([nri,n_mu],0.0);
    let mut inv_cctrans = c2.pinv(1.0e-12);
    tmp0.lapack_dgemm(&mut tmp1, &mut inv_cctrans, 'N', 'N', 1.0, 0.0);
    println!("prepared for dgesv");

    let mut tmp01 = tmp0.clone();
    let mut tmp = ri_v_ri.lapack_dgesv(&mut tmp01, nri as i32);
    let mut kernel_part = MatrixFull::new([n_mu,n_mu], 0.0);
    kernel_part.lapack_dgemm(&mut tmp0, &mut tmp, 'T', 'N', 1.0, 0.0);
    //&kernel_part.formated_output_e(5, "full");
    // generate result
    let mut mid = MatrixFull::new([nao*nao,n_mu],0.0);
    mid.lapack_dgemm(&mut c, &mut kernel_part, 'T', 'N', 1.0, 0.0);
    let mut fourcenter_after_isdf = MatrixFull::new([nao*nao,nao*nao],0.0);
    fourcenter_after_isdf.lapack_dgemm(&mut mid, &mut c, 'N', 'N', 1.0, 0.0);

    fourcenter_after_isdf
}


pub fn error_isdf(k_mu: Range<usize>, scf_data: &SCF) -> (Vec<usize>,Vec<f64>, Vec<f64>){
    //calculate the error of eri generated by isdf
    let length = k_mu.len();
    let mut k_mu_list = vec![0 as usize; length];
    let mut abs_error_isdf = vec![0.0; length];
    let mut rel_error_isdf = vec![0.0; length];
    let mol = &scf_data.mol;
    let grids = scf_data.grids.as_ref().unwrap();
    // int2e
    let mut cint_data = mol.initialize_cint(true);
        let nbas = mol.num_basis;
        let mut mat_full = 
            ERIFull::new([nbas,nbas,nbas,nbas],0.0);
        let nbas_shell = mol.cint_bas.len();
        cint_data.cint2e_optimizer_rust();
        for l in 0..nbas_shell {
            let bas_start_l = mol.cint_fdqc[l][0];
            let bas_len_l = mol.cint_fdqc[l][1];
            for k in 0..(l+1) {
                let bas_start_k = mol.cint_fdqc[k][0];
                let bas_len_k = mol.cint_fdqc[k][1];
                for j in 0..nbas_shell {
                    let bas_start_j = mol.cint_fdqc[j][0];
                    let bas_len_j = mol.cint_fdqc[j][1];
                    for i in 0..j+1 {
                        let bas_start_i = mol.cint_fdqc[i][0];
                        let bas_len_i = mol.cint_fdqc[i][1];
                        let buf = cint_data.cint_ijkl_by_shell(i as i32, j as i32, k as i32, l as i32);
                        mat_full.chrunk_copy([bas_start_i..bas_start_i+bas_len_i,
                                              bas_start_j..bas_start_j+bas_len_j,
                                              bas_start_k..bas_start_k+bas_len_k,
                                              bas_start_l..bas_start_l+bas_len_l,
                                              ], buf.clone());
                        // copy the "upper" part to the lower part
                        if i<j {
                            mat_full.chrunk_copy_transpose_ij([
                                bas_start_i..bas_start_i+bas_len_i,
                                bas_start_j..bas_start_j+bas_len_j,
                                bas_start_k..bas_start_k+bas_len_k,
                                bas_start_l..bas_start_l+bas_len_l,
                                ], buf);
                        }
                    };
                }
            }
        }
        cint_data.final_c2r();
        // to copy the upper part of the (k,l) pair to the lower block
        for k in 0..nbas {
            for l in 0..k {
                let from_slice =  mat_full.get4d_slice([0,0,l,k], mat_full.indicing[2]).unwrap().to_vec();
                let mut to_slice = mat_full.get4d_slice_mut([0,0,k,l], mat_full.indicing[2]).unwrap();
                to_slice.iter_mut().zip(from_slice.iter()).for_each(|(t,f)|*t = *f);
            }
        }

        let mut fourcenter_before_isdf = MatrixFull::from_vec([nbas*nbas, nbas*nbas],mat_full.data).unwrap();
        let mut value = 0.0;
        fourcenter_before_isdf.iter().for_each(|x|{
            value += x * x;
        });
        let mut k = 0usize;
        for i in k_mu{
            let mut fourcenter_after_isdf = fourcenter_after_isdf(i, &mol, &grids);
            let mut abs_error = 0.0;
            fourcenter_after_isdf.data.iter().zip(fourcenter_before_isdf.data.iter()).for_each(|(after,before)|{
                abs_error += (after - before) * (after - before);
            });
            abs_error_isdf[k] = abs_error.sqrt();
            rel_error_isdf[k] = abs_error_isdf[k]/(value.sqrt());
            k_mu_list[k] = i;
            k += 1;
        }

    (k_mu_list, abs_error_isdf, rel_error_isdf)
}

pub fn index_of_min(nets: &mut Vec<f64>) -> usize{
    let index_of_min: Option<usize> = nets
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(index, _)| index);
    index_of_min.unwrap()
}

pub fn tabulated_ao (mol: &Molecule, rand_p: &Vec<[f64; 3]>) -> MatrixFull<f64>{
    let default_omp_num_threads = mol.ctrl.num_threads.unwrap();

    let num_grids = rand_p.len();
    let num_basis = mol.num_basis;
    let mut ao = MatrixFull::new([num_basis,num_grids],0.0);

    let par_tasks = utilities::balancing(num_grids, rayon::current_num_threads());
    let (sender, receiver) = channel();
    par_tasks.par_iter().for_each_with(sender, |s, range_grids| {
        omp_set_num_threads_wrapper(1);

        let loc_num_grids = range_grids.len();
        let mut loc_ao = MatrixFull::new([num_basis, loc_num_grids],0.0);
        mol.basis4elem.iter().zip(mol.geom.position.iter_columns_full()).for_each(|(elem, geom)| {
            let ind_glb_bas = elem.global_index.0;
            let loc_num_bas = elem.global_index.1;
            let start = ind_glb_bas;
            let end = start + loc_num_bas;
            let mut tmp_geom = [0.0;3];
            tmp_geom.iter_mut().zip(geom.iter()).for_each(|value| {*value.0 = *value.1});
            let tab_den = spheric_gto_value_serial(&rand_p[range_grids.clone()], &tmp_geom, elem);

            loc_ao.copy_from_matr(start..end, 0..loc_num_grids, &tab_den, 0..loc_num_bas, 0..loc_num_grids);

        });
        s.send((loc_ao, range_grids)).unwrap()
    });
    receiver.into_iter().for_each(|(loc_ao, range_grids)| {
        let loc_num_grids = range_grids.len();
        ao.copy_from_matr(0..num_basis, range_grids.clone(), &loc_ao, 0..num_basis, 0..loc_num_grids);
    });
    omp_set_num_threads_wrapper(default_omp_num_threads as usize);
    ao // [num_basis,num_grids]
}

pub fn prepare_for_ri_isdf(k_mu: usize, mol: &Molecule, grids: &dft::Grids) -> RIFull<f64> {
    let nao = mol.num_basis;
    let nri = mol.num_auxbas;
    let lambda_r = &grids.weights;
    let ngrids = lambda_r.len();
    let rgrids = &grids.coordinates;

    let mut phi = tabulated_ao(&mol, &rgrids);
    let n_mu = k_mu * nri;
    let mut lambda_r_for_isdf = vec![0.0; ngrids];
    lambda_r_for_isdf.iter_mut().zip(lambda_r.iter()).for_each(|(x,y)|{
        *x = y.abs();
    });
     
    let (ip, ip_weights) = cvt_isdf_v2(&rgrids, &lambda_r_for_isdf, n_mu);
    let mut varphi = tabulated_ao(&mol, &ip);
    let mut lambda_varphi = MatrixFull::new([nao,n_mu],0.0);
    lambda_varphi.iter_columns_full_mut().zip(ip_weights.iter().zip(varphi.iter_columns_full()))
    .for_each(|(x, (weight,aos))|{
        x.iter_mut().zip(aos.iter()).for_each(|(y,ao)|{
            *y = weight * ao;
        });
    });

    //C2 = (lambda_varphi.T \cdot lambda_varphi) \times (varphi.T \cdot varphi)
    // c2就是S_{KL}，见公式17
    let mut c21 = MatrixFull::new([n_mu, n_mu], 0.0);
    let mut lambda_varphi_mid = lambda_varphi.clone();
    c21.lapack_dgemm(&mut lambda_varphi, &mut lambda_varphi_mid, 'T', 'N', 1.0, 0.0);
    let mut c22 = MatrixFull::new([n_mu, n_mu], 0.0);
    let mut varphi_mid = varphi.clone();
    c22.lapack_dgemm(&mut varphi, &mut varphi_mid, 'T', 'N', 1.0, 0.0);
    let mut c2 = MatrixFull::new([n_mu, n_mu], 0.0);
    c2.iter_columns_full_mut().zip(c21.iter_columns_full()).zip(c22.iter_columns_full())
        .for_each(|((x,a),b)|{
            x.iter_mut().zip(a.iter()).zip(b.iter()).for_each(|((x1,a1),b1)|{
                *x1 = a1 * b1;
            });
        });

    //<Z|V|P><P|V|P><P|V|Z>
    let mut cint_data = mol.initialize_cint(true);
    let n_basis_shell = mol.cint_bas.len();
    let n_auxbas_shell = mol.cint_aux_bas.len();
    cint_data.cint2c2e_optimizer_rust();
    let mut ri3fn = RIFull::new([nao,nao,nri],0.0);
    let mut ri_v_ri = MatrixFull::new([nri,nri],0.0);
    for l in 0..n_auxbas_shell {
        let basis_start_l = mol.cint_aux_fdqc[l][0];
        let basis_len_l = mol.cint_aux_fdqc[l][1];
        let gl  = l + n_basis_shell;
        for k in 0..n_auxbas_shell {
            let basis_start_k = mol.cint_aux_fdqc[k][0];
            let basis_len_k = mol.cint_aux_fdqc[k][1];
            let gk  = k + n_basis_shell;
            let buf = cint_data.cint_2c2e(gk as i32, gl as i32);
            
            let mut tmp_slices = ri_v_ri.iter_submatrix_mut(
                basis_start_k..basis_start_k+basis_len_k,
                basis_start_l..basis_start_l+basis_len_l);
            tmp_slices.zip(buf.iter()).for_each(|value| {*value.0 = *value.1});

        }
    }
    cint_data.cint3c2e_optimizer_rust();
    for k in 0..n_auxbas_shell {
        let basis_start_k = mol.cint_aux_fdqc[k][0];
        let basis_len_k = mol.cint_aux_fdqc[k][1];
        let gk  = k + n_basis_shell;
        for j in 0..n_basis_shell {
            let basis_start_j = mol.cint_fdqc[j][0];
            let basis_len_j = mol.cint_fdqc[j][1];
            // can be optimized with "for i in 0..(j+1)"
            for i in 0..n_basis_shell {
                let basis_start_i = mol.cint_fdqc[i][0];
                let basis_len_i = mol.cint_fdqc[i][1];
                let buf = RIFull::from_vec([basis_len_i, basis_len_j,basis_len_k], 
                    cint_data.cint_3c2e(i as i32, j as i32, gk as i32)).unwrap();
                ri3fn.copy_from_ri(
                    basis_start_i..basis_start_i+basis_len_i,
                    basis_start_j..basis_start_j+basis_len_j,
                    basis_start_k..basis_start_k+basis_len_k,
                    & buf, 
                    0..basis_len_i, 
                    0..basis_len_j, 
                    0..basis_len_k);
            }
        }
    }
    cint_data.final_c2r();

    // ri_v_ao_t是公式22里面的三中心积分(P|\mu\nu)
    // ri_v_ri是公式22里面的(Q|P)
    let mut ri_v_ao_t = MatrixFull::from_vec([nao*nao, nri],ri3fn.data).unwrap();

    // c就是C_{\mu\nu}^{L}*\lambda(r_L)， 见公式22,24
    let mut c = prod_states_gw(&lambda_varphi.transpose(), &varphi.transpose());
    omp_set_num_threads_wrapper(mol.ctrl.num_threads.unwrap());
    //let omp_num = omp_get_num_threads_wrapper();
    let mut tmp1 = MatrixFull::new([nri, n_mu],0.0);

    tmp1.lapack_dgemm(&mut ri_v_ao_t, &mut c, 'T', 'T', 1.0, 0.0);

    let mut tmp0 = MatrixFull::new([nri,n_mu],0.0);
    let mut inv_cctrans = c2.pinv(1.0e-12);
    //tmp0是公式22里面除去(Q|P)^{-1/2}部分的矩阵
    tmp0.lapack_dgemm(&mut tmp1, &mut inv_cctrans, 'N', 'N', 1.0, 0.0);  
    let mut tmp01 = tmp0.clone();
    let mut tmp = ri_v_ri.lapack_dgesv(&mut tmp01, nri as i32);

    // kernel_part就是M_{KL}，见公式23 
    let mut kernel_part = MatrixFull::new([n_mu,n_mu], 0.0);
    kernel_part.lapack_dgemm(&mut tmp0, &mut tmp, 'T', 'N', 1.0, 0.0);
    // generate result
    // aux_v就是M_{KL}^{1/2}
    let mut aux_v = kernel_part.lapack_power(0.5, 1.0E-8).unwrap();
    let mut c3 = vec![0.0; nao*nao*n_mu];
    for i in 0..n_mu{
        for j in 0..nao{
            for k in 0..nao{
                c3[i*nao*nao + j*nao + k]= c.data[k*n_mu*nao + j*n_mu + i];
            }
        }
    }
    let mut ri3fn = RIFull::from_vec([nao,nao,n_mu], c3).unwrap();
    let mut tmp_ovlp_matr = MatrixFull::new([nao,n_mu],0.0);
    let mut aux_ovlp_matr = MatrixFull::new([nao,n_mu],0.0);
    let n_basis = nao;
    let n_auxbas = n_mu;
    let size = [nao,n_mu];
    for j in 0..nao {
        matr_copy_from_ri(&ri3fn.data, &ri3fn.size,0..n_basis, 0..n_auxbas, j, 1,
            &mut tmp_ovlp_matr.data, &size, 0..n_basis, 0..n_auxbas);

        aux_ovlp_matr.to_matrixfullslicemut().lapack_dgemm(
            &tmp_ovlp_matr.to_matrixfullslice(), &aux_v.to_matrixfullslice(), 
            'N', 'N', 1.0,0.0);

        ri3fn.copy_from_matr(0..n_basis, 0..n_auxbas, j, 1, 
            &aux_ovlp_matr, 0..n_basis, 0..n_auxbas)
    }

    ri3fn
}

pub fn gen_atom_grids_for_isdf(mol:&Molecule) -> Vec<dftgrids>{
    let radial_precision = mol.ctrl.radial_precision;
    let min_num_angular_points: usize = mol.ctrl.min_num_angular_points;
    let max_num_angular_points: usize = mol.ctrl.max_num_angular_points;
    let hardness: usize = mol.ctrl.hardness;
    let pruning: String = mol.ctrl.pruning.clone();
    let grid_gen_level: usize = mol.ctrl.grid_gen_level;
    let rad_grid_method: String = mol.ctrl.rad_grid_method.clone();

        // obtain system-dependent parameters
    let mass_charge = get_mass_charge(&mol.geom.elem);
    let proton_charges: Vec<i32> = mass_charge.iter().map(|value| value.1 as i32).collect();
    let center_coordinates_bohr = mol.geom.to_numgrid_io();
        //mol.fdqc_bas[0].
    let mut alpha_max: Vec<f64> = vec![];
    let mut alpha_min: Vec<HashMap<usize,f64>> = vec![];
    mol.basis4elem.iter().for_each(|value| {
        let (tmp_alpha_min, tmp_alpha_max) = value.to_numgrid_io();
        alpha_max.push(tmp_alpha_max);
        alpha_min.push(tmp_alpha_min);
    });

    let mut num_points: usize = 0;
    let mut coordinates: Vec<[f64;3]> =vec![];
    let mut weights:Vec<f64> = vec![];
    let mut atom_grids: Vec<dftgrids> = vec![];

    alpha_min.iter().zip(alpha_max.iter()).enumerate().for_each(|(center_index,value)| {
        let (rs_atom, ws_atom) = gen_grids::atom_grid(
            value.0.clone(), 
            value.1.clone(), 
            radial_precision, 
            min_num_angular_points, 
            max_num_angular_points, 
            proton_charges.clone(), 
            center_index, 
            center_coordinates_bohr.clone(), 
            hardness,
            pruning.clone(),
            rad_grid_method.clone(),
            grid_gen_level,
        );

        let mut coor = vec![[0.0;3]; rs_atom.len()];
        coor.iter_mut().zip(rs_atom.iter()).for_each(|(x,y)|{
            x[0] = y.0;
            x[1] = y.1;
            x[2] = y.2;
        });

        let threshold = 1.0e-15;
        let effective_ind = &ws_atom.iter()
        .enumerate()
        .filter(|(_, &r)| r.abs() >= threshold)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
        println!("effective index: {}", effective_ind.len());
        let ngrids = effective_ind.len();
        let mut lambda_r = vec![0.0; effective_ind.len()];
        lambda_r.iter_mut().zip(effective_ind.iter()).for_each(|(new,index_new)|{
            *new = ws_atom[*index_new];
        });

        let mut rgrids = vec![[0.0;3]; effective_ind.len()];
        rgrids.iter_mut().zip(effective_ind.iter()).for_each(|(new,index_new)|{
            new.iter_mut().zip(coor[*index_new].iter()).for_each(|(a,b)|{
                *a = *b;
            }) 
        });

        let parallel_balancing = balancing(coordinates.len(), rayon::current_num_threads());
        let ao_grid = dftgrids {
            weights: lambda_r,
            coordinates: rgrids,
            ao: None,
            aop: None, 
            parallel_balancing,
            non0tab: None,
            ao_cutoff: 0.0,
            ao_compressed: None,
            aop_compressed: None,
        };
        atom_grids.push(ao_grid)
            
    });
        
        atom_grids
}

pub fn atom_isdf(rgrids_old: &Vec<[f64;3]>, lambda_r_old: &Vec<f64>, n_mu: usize) ->   (Vec<[f64;3]>, Vec<f64>){
    let mut lambda_r_abs = vec![0.0; lambda_r_old.len()];
    lambda_r_abs.iter_mut().zip(lambda_r_old.iter()).for_each(|(abs,old)|{
        *abs = old.abs();
    });

    let threshold = 1.0e-8;
    let effective_ind = lambda_r_abs.iter()
    .enumerate()
    .filter(|(_, &r)| r >= threshold)
    .map(|(index, _)| index)
    .collect::<Vec<_>>();
    if n_mu >= effective_ind.len() {
        println!("There are {} effective index. N_mu is {}.", effective_ind.len(), &n_mu);
        panic!("n_mu is smaller than effective_ind!");
    }

    let ngrids = effective_ind.len();
    let mut lambda_r = vec![0.0; effective_ind.len()];
    lambda_r.iter_mut().zip(effective_ind.iter()).for_each(|(new,index_new)|{
        *new = lambda_r_abs[*index_new];
    });

    let mut lambda_r_org = vec![0.0; effective_ind.len()];
    lambda_r_org.iter_mut().zip(effective_ind.iter()).for_each(|(new,index_new)|{
        *new = lambda_r_old[*index_new];
    });

    let mut rgrids = vec![[0.0;3]; effective_ind.len()];
    rgrids.iter_mut().zip(effective_ind.iter()).for_each(|(new,index_new)|{
        new.iter_mut().zip(rgrids_old[*index_new].iter()).for_each(|(a,b)|{
            *a = *b;
        }) 
    });

    // 随机从格点中选取c_mu
    let mut c_mu: Vec<[f64;3]> = vec![[0.0;3];n_mu];
    for i in 0..n_mu{
        let mut rng = rand::rng();
        let mut random_number = rng.random_range(0..lambda_r.len());
        c_mu[i] = rgrids[random_number];
    }
    
    // get generators
    let max_iter = 300;
    let mut class_result = (vec![0usize; n_mu],vec![0.0; n_mu]);
    let mut ind_r = vec![0usize; n_mu];
    let mut dist_r = vec![0.0; n_mu];
    let mut count = 0;
    let mut dist = 0.0;
    let mut c_mu_new = vec![[0.0;3];n_mu];
    let result = loop{
        class_result = cvt_classification(&rgrids, &lambda_r, &c_mu);
        ind_r = class_result.0;
        dist_r = class_result.1;
        let mut init_dist_r = 0.0;
        dist_r.iter().zip(lambda_r.iter()).for_each(|(d, w)|{
            init_dist_r += w * d * d;
        });
        dist = init_dist_r.sqrt();

        let mut c_mu_new = cvt_update_cmu(&rgrids, &lambda_r, &c_mu, &ind_r);
        let mut sum = 0.0;
        
        if count == max_iter - 1 {
            c_mu_new.iter().zip(c_mu.iter()).for_each(|([x,y,z],[a,b,c])|{
                sum += ((x-a) * (x-a) + (y-b) * (y-b) + (z-c) * (z-c)).sqrt();
            });
            c_mu = c_mu_new;
            let ind_mu = cvt_find_corresponding_point(&rgrids, &lambda_r, &c_mu, &ind_r);
            break (ind_mu, sum);
        } 

        c_mu_new.iter().zip(c_mu.iter()).for_each(|([x,y,z],[a,b,c])|{
            sum += ((x-a).powf(2.0) + (y-b).powf(2.0) + (z-c).powf(2.0)).sqrt();
        });
        
        if sum <= 1e-6{
            c_mu = c_mu_new;
            let ind_mu = cvt_find_corresponding_point(&rgrids, &lambda_r, &c_mu, &ind_r);
            println!("Random points converged after {} iterations.", &count-1);
            println!("Dist: {:?}", &dist);
            break (ind_mu, sum);
        }else {
            c_mu = c_mu_new;
            count += 1;     
        }
    };

    let mut inter_points = vec![[0.0;3]; n_mu];
    let mut inter_weights = vec![0.0; n_mu];
    inter_points.iter_mut().zip(inter_weights.iter_mut().zip(result.0.iter())).for_each(|(p, (w,i))|{
        p.iter_mut().zip(rgrids[*i].iter()).for_each(|(x,y)|{*x = *y});
        *w = lambda_r_org[*i];

    });     
    (inter_points, inter_weights)
}

pub fn cvt_isdf_v3(mol: &Molecule, k_mu: usize) -> (Vec<[f64;3]>, Vec<f64>){
    //atom-used
    let atom_grids = gen_atom_grids_for_isdf(&mol);
    let natm = atom_grids.len();
    let n_mu_tot = natm * k_mu * mol.num_auxbas;
    let n_mu = k_mu * mol.num_auxbas;
    let mut ip = vec![[0.0;3];n_mu_tot];
    let mut weight = vec![0.0; n_mu_tot];
    for i in 0..natm{
        let (atom_ip, atom_weight) = atom_isdf(&atom_grids[i].coordinates, &atom_grids[i].weights , n_mu);
        for j in 0 .. n_mu{
            ip[i*n_mu + j] = atom_ip[j];
            weight[i*n_mu + j] = atom_weight[j];
        }
    }

    (ip, weight)

}

pub fn prepare_for_ri_isdf_atoms(k_mu: usize, mol: &Molecule, grids: &dft::Grids) -> RIFull<f64> {
    let nao = mol.num_basis;
    let nri = mol.num_auxbas;
    let threshold = 1.0e-20;
    let effective_ind = &grids.weights.iter()
    .enumerate()
    .filter(|(_, &r)| r.abs() >= threshold)
    .map(|(index, _)| index)
    .collect::<Vec<_>>();
    
    let ngrids = effective_ind.len();
    let mut lambda_r = vec![0.0; effective_ind.len()];
    lambda_r.iter_mut().zip(effective_ind.iter()).for_each(|(new,index_new)|{
        *new = grids.weights[*index_new];
    });
    let mut rgrids = vec![[0.0;3]; effective_ind.len()];
    rgrids.iter_mut().zip(effective_ind.iter()).for_each(|(new,index_new)|{
        new.iter_mut().zip(grids.coordinates[*index_new].iter()).for_each(|(a,b)|{
            *a = *b;
        }) 
    });

    let mut phi = tabulated_ao(&mol, &rgrids);

    let n_mu = k_mu * nri * mol.geom.elem.len();
    let mut lambda_r_for_isdf = vec![0.0; ngrids];

    let (ip, ip_weights) = cvt_isdf_v3(&mol, k_mu);
    let mut varphi = tabulated_ao(&mol, &ip);
    let mut lambda_varphi = MatrixFull::new([nao,n_mu],0.0);
    lambda_varphi.iter_columns_full_mut().zip(ip_weights.iter().zip(varphi.iter_columns_full()))
    .for_each(|(x, (weight,aos))|{
        x.iter_mut().zip(aos.iter()).for_each(|(y,ao)|{
            *y = weight * ao;
        });
    });

    // c2就是S_{KL}，见公式17
    let mut c21 = MatrixFull::new([n_mu, n_mu], 0.0);
    let mut lambda_varphi_mid = lambda_varphi.clone();
    c21.lapack_dgemm(&mut lambda_varphi, &mut lambda_varphi_mid, 'T', 'N', 1.0, 0.0);
    let mut c22 = MatrixFull::new([n_mu, n_mu], 0.0);
    let mut varphi_mid = varphi.clone();
    c22.lapack_dgemm(&mut varphi, &mut varphi_mid, 'T', 'N', 1.0, 0.0);
    let mut c2 = MatrixFull::new([n_mu, n_mu], 0.0);
    c2.iter_columns_full_mut().zip(c21.iter_columns_full()).zip(c22.iter_columns_full())
        .for_each(|((x,a),b)|{
            x.iter_mut().zip(a.iter()).zip(b.iter()).for_each(|((x1,a1),b1)|{
                *x1 = a1 * b1;
            });
        });
    //<Z|V|P><P|V|P><P|V|Z>
    let mut cint_data = mol.initialize_cint(true);
    let n_basis_shell = mol.cint_bas.len();
    let n_auxbas_shell = mol.cint_aux_bas.len();
    cint_data.cint2c2e_optimizer_rust();
    let mut ri3fn = RIFull::new([nao,nao,nri],0.0);
    let mut ri_v_ri = MatrixFull::new([nri,nri],0.0);
    for l in 0..n_auxbas_shell {
        let basis_start_l = mol.cint_aux_fdqc[l][0];
        let basis_len_l = mol.cint_aux_fdqc[l][1];
        let gl  = l + n_basis_shell;
        for k in 0..n_auxbas_shell {
            let basis_start_k = mol.cint_aux_fdqc[k][0];
            let basis_len_k = mol.cint_aux_fdqc[k][1];
            let gk  = k + n_basis_shell;
            let buf = cint_data.cint_2c2e(gk as i32, gl as i32);
            
            let mut tmp_slices = ri_v_ri.iter_submatrix_mut(
                basis_start_k..basis_start_k+basis_len_k,
                basis_start_l..basis_start_l+basis_len_l);
            tmp_slices.zip(buf.iter()).for_each(|value| {*value.0 = *value.1});

        }
    }
    cint_data.cint3c2e_optimizer_rust();
    for k in 0..n_auxbas_shell {
        let basis_start_k = mol.cint_aux_fdqc[k][0];
        let basis_len_k = mol.cint_aux_fdqc[k][1];
        let gk  = k + n_basis_shell;
        for j in 0..n_basis_shell {
            let basis_start_j = mol.cint_fdqc[j][0];
            let basis_len_j = mol.cint_fdqc[j][1];
            // can be optimized with "for i in 0..(j+1)"
            for i in 0..n_basis_shell {
                let basis_start_i = mol.cint_fdqc[i][0];
                let basis_len_i = mol.cint_fdqc[i][1];
                let buf = RIFull::from_vec([basis_len_i, basis_len_j,basis_len_k], 
                    cint_data.cint_3c2e(i as i32, j as i32, gk as i32)).unwrap();
                ri3fn.copy_from_ri(
                    basis_start_i..basis_start_i+basis_len_i,
                    basis_start_j..basis_start_j+basis_len_j,
                    basis_start_k..basis_start_k+basis_len_k,
                    & buf, 
                    0..basis_len_i, 
                    0..basis_len_j, 
                    0..basis_len_k);
            }
        }
    }
    cint_data.final_c2r();

    // ri_v_ao_t是公式22里面的三中心积分(P|\mu\nu)
    // ri_v_ri是公式22里面的(Q|P)
    let mut ri_v_ao_t = MatrixFull::from_vec([nao*nao, nri],ri3fn.data).unwrap();
    
    // c就是C_{\mu\nu}^{L}*\lambda(r_L)， 见公式22,24
    let mut c = prod_states_gw(&lambda_varphi.transpose(), &varphi.transpose());
    println!("C prepared");
    let mut tmp1 = MatrixFull::new([nri, n_mu],0.0);

    tmp1.lapack_dgemm(&mut ri_v_ao_t, &mut c, 'T', 'T', 1.0, 0.0);

    let mut tmp0 = MatrixFull::new([nri,n_mu],0.0);
    let mut inv_cctrans = c2.pinv(1.0e-12);

    //tmp0是公式22里面除去(Q|P)^{-1/2}部分的矩阵
    tmp0.lapack_dgemm(&mut tmp1, &mut inv_cctrans, 'N', 'N', 1.0, 0.0);
    println!("prepared for dgesv");
    println!("dgesv finished");
    let mut tmp01 = tmp0.clone();
    let mut tmp = ri_v_ri.lapack_dgesv(&mut tmp01, nri as i32);
    // kernel_part就是M_{KL}，见公式23 
    let mut kernel_part = MatrixFull::new([n_mu,n_mu], 0.0);
    kernel_part.lapack_dgemm(&mut tmp0, &mut tmp, 'T', 'N', 1.0, 0.0);
    // generate result
    // aux_v就是M_{KL}^{1/2}
    let mut aux_v = kernel_part.lapack_power(0.5, 1.0E-8).unwrap();
    // c3就是\omega_\lambda(r_L)\omega_\sigma(r_L)， 见公式6
    let mut c3 = vec![0.0; nao*nao*n_mu];
    for i in 0..n_mu{
        for j in 0..nao{
            for k in 0..nao{
                c3[i*nao*nao + j*nao + k]= c.data[k*n_mu*nao + j*n_mu + i];
            }
        }
    }
    let mut ri3fn = RIFull::from_vec([nao,nao,n_mu], c3).unwrap();
    let mut tmp_ovlp_matr = MatrixFull::new([nao,n_mu],0.0);
    let mut aux_ovlp_matr = MatrixFull::new([nao,n_mu],0.0);
    let n_basis = nao;
    let n_auxbas = n_mu;
    let size = [nao,n_mu];
    for j in 0..nao {
        matr_copy_from_ri(&ri3fn.data, &ri3fn.size,0..n_basis, 0..n_auxbas, j, 1,
            &mut tmp_ovlp_matr.data, &size, 0..n_basis, 0..n_auxbas);

        aux_ovlp_matr.to_matrixfullslicemut().lapack_dgemm(
            &tmp_ovlp_matr.to_matrixfullslice(), &aux_v.to_matrixfullslice(), 
            'N', 'N', 1.0,0.0);

        ri3fn.copy_from_matr(0..n_basis, 0..n_auxbas, j, 1, 
            &aux_ovlp_matr, 0..n_basis, 0..n_auxbas)
    }

    ri3fn
}

pub fn init_by_rho(grids: &mut dftgrids, dm: &Vec<MatrixFull<f64>>, spin_channel: usize, n_mu: usize) -> Vec<[f64;3]> {
    let mut dm = dm.clone();
    let rho = grids.prepare_tabulated_density(&mut dm, spin_channel);
    let ngrids = rho.size[0];
    let mut new_rho = vec![0.0; ngrids];
    if spin_channel == 2 {
        for i in 0..ngrids{
            new_rho[i] = rho.data[i] + rho.data[ngrids + i];
        }
    }else{
        new_rho = rho.data.clone();
    }
    let mut varrho = vec![0.0; ngrids];
    varrho.iter_mut().zip(new_rho.iter()).zip(grids.weights.iter()) 
        .for_each(|((x_w,x),w)|{
            *x_w = *x * w
        });
    let mut indices: Vec<usize> = (0..varrho.len()).collect();

    varrho.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let new_order_indices: Vec<usize> = varrho.iter()
        .map(|n| indices[varrho.iter().position(|&x| (x - *n).abs() < 1e-10).unwrap()])
        .collect();

    let mut new_grids = vec![[0.0;3]; n_mu];
    //let mut new_weights = vec![0.0; n_mu];
    for i in 0..n_mu{
        new_grids[i] = grids.coordinates[new_order_indices[i]];

    }

    new_grids

}

pub fn prepare_m_isdf(k_mu: usize, mol: &Molecule, grids: &dft::Grids) -> (MatrixFull<f64>, MatrixFull<f64>) {
    //used for isdf_k 
    let nao = mol.num_basis;
    let nri = mol.num_auxbas;

    let lambda_r = &grids.weights;
    let ngrids = lambda_r.len();
    let rgrids = &grids.coordinates;
    let mut phi = tabulated_ao(&mol, &rgrids);

    let n_mu = k_mu * nri;
    let mut lambda_r_for_isdf = vec![0.0; ngrids];
    lambda_r_for_isdf.iter_mut().zip(lambda_r.iter()).for_each(|(x,y)|{
        *x = y.abs();
    });
    let mut lambda_phi = MatrixFull::new([nao,ngrids],0.0);
    lambda_phi.iter_columns_full_mut().zip(lambda_r.iter().zip(phi.iter_columns_full()))
    .for_each(|(x, (weight,aos))|{
        x.iter_mut().zip(aos.iter()).for_each(|(y,ao)|{
            *y = weight * ao;
        });
    });

    let (ip, ip_weights) = cvt_isdf_v2(&rgrids, &lambda_r_for_isdf, n_mu);
    let mut varphi = tabulated_ao(&mol, &ip);
    let mut lambda_varphi = MatrixFull::new([nao,n_mu],0.0);
    lambda_varphi.iter_columns_full_mut().zip(ip_weights.iter().zip(varphi.iter_columns_full()))
    .for_each(|(x, (weight,aos))|{
        x.iter_mut().zip(aos.iter()).for_each(|(y,ao)|{
            *y = weight * ao;
        });
    });

    let mut c21 = MatrixFull::new([n_mu, n_mu], 0.0);
    let mut lambda_varphi_mid = lambda_varphi.clone();
    c21.lapack_dgemm(&mut lambda_varphi, &mut lambda_varphi_mid, 'T', 'N', 1.0, 0.0);
    let mut c22 = MatrixFull::new([n_mu, n_mu], 0.0);
    let mut varphi_mid = varphi.clone();
    c22.lapack_dgemm(&mut varphi, &mut varphi_mid, 'T', 'N', 1.0, 0.0);
    let mut c2 = MatrixFull::new([n_mu, n_mu], 0.0);
    c2.iter_columns_full_mut().zip(c21.iter_columns_full()).zip(c22.iter_columns_full())
        .for_each(|((x,a),b)|{
            x.iter_mut().zip(a.iter()).zip(b.iter()).for_each(|((x1,a1),b1)|{
                *x1 = a1 * b1;
            });
        });

    let mut cint_data = mol.initialize_cint(true);
    let n_basis_shell = mol.cint_bas.len();
    let n_auxbas_shell = mol.cint_aux_bas.len();
    let mut ri3fn = RIFull::new([nao,nao,nri],0.0);
    cint_data.cint2c2e_optimizer_rust();
    let mut ri_v_ri = MatrixFull::new([nri,nri],0.0);
    for l in 0..n_auxbas_shell {
        let basis_start_l = mol.cint_aux_fdqc[l][0];
        let basis_len_l = mol.cint_aux_fdqc[l][1];
        let gl  = l + n_basis_shell;
        for k in 0..n_auxbas_shell {
            let basis_start_k = mol.cint_aux_fdqc[k][0];
            let basis_len_k = mol.cint_aux_fdqc[k][1];
            let gk  = k + n_basis_shell;
            let buf = cint_data.cint_2c2e(gk as i32, gl as i32);
            
            let mut tmp_slices = ri_v_ri.iter_submatrix_mut(
                basis_start_k..basis_start_k+basis_len_k,
                basis_start_l..basis_start_l+basis_len_l);
            tmp_slices.zip(buf.iter()).for_each(|value| {*value.0 = *value.1});

        }
    }
    cint_data.cint3c2e_optimizer_rust();
    for k in 0..n_auxbas_shell {
        let basis_start_k = mol.cint_aux_fdqc[k][0];
        let basis_len_k = mol.cint_aux_fdqc[k][1];
        let gk  = k + n_basis_shell;
        for j in 0..n_basis_shell {
            let basis_start_j = mol.cint_fdqc[j][0];
            let basis_len_j = mol.cint_fdqc[j][1];
            // can be optimized with "for i in 0..(j+1)"
            for i in 0..n_basis_shell {
                let basis_start_i = mol.cint_fdqc[i][0];
                let basis_len_i = mol.cint_fdqc[i][1];
                let buf = RIFull::from_vec([basis_len_i, basis_len_j,basis_len_k], 
                    cint_data.cint_3c2e(i as i32, j as i32, gk as i32)).unwrap();
                ri3fn.copy_from_ri(
                    basis_start_i..basis_start_i+basis_len_i,
                    basis_start_j..basis_start_j+basis_len_j,
                    basis_start_k..basis_start_k+basis_len_k,
                    & buf, 
                    0..basis_len_i, 
                    0..basis_len_j, 
                    0..basis_len_k);
            }
        }
    }
        cint_data.final_c2r();

    let mut ri_v_ao_t = MatrixFull::from_vec([nao*nao, nri],ri3fn.data).unwrap();

    let mut c = prod_states_gw(&lambda_varphi.transpose(), &varphi.transpose());
    let mut tmp1 = MatrixFull::new([nri, n_mu],0.0);
    tmp1.lapack_dgemm(&mut ri_v_ao_t, &mut c, 'T', 'T', 1.0, 0.0);

    let mut tmp0 = MatrixFull::new([nri,n_mu],0.0);

    //=====================test_pinv_time==========================
    let mut time_mark = utilities::TimeRecords::new();
    time_mark.new_item("test pinv", "pseudo inverse of c2");
    time_mark.count_start("test pinv");
    let mut inv_cctrans = c2.pinv(1.0e-12);
    time_mark.count("test pinv");
    time_mark.report_all();
    //=============================================================

    tmp0.lapack_dgemm(&mut tmp1, &mut inv_cctrans, 'N', 'N', 1.0, 0.0);

    let mut tmp01 = tmp0.clone();
    let mut tmp = ri_v_ri.lapack_dgesv(&mut tmp01, nri as i32);
    let mut kernel_part = MatrixFull::new([n_mu,n_mu], 0.0);
    kernel_part.lapack_dgemm(&mut tmp0, &mut tmp, 'T', 'N', 1.0, 0.0);
    for i in 0..n_mu{
        for j in 0..n_mu{
            kernel_part[[i,j]] *= ip_weights[i] * ip_weights[j]
        }
    }
    (varphi, kernel_part)
}

pub fn find_current_python() -> Option<String> {
    println!("=========================================================");
    println!("Check Python interpreter");
    let python_commands = ["python", "python3"];
    for cmd in &python_commands {
        match Command::new(cmd).arg("--version").output() {
            Ok(output) => {
                let version = String::from_utf8_lossy(&output.stdout);
                println!("Find {}: {}", cmd, version.trim());
                println!("=========================================================");
                return Some(cmd.to_string());
            },
            Err(_) => {}
        }
    }
    println!("Could not find python or python3!");
    None
}

// \alpha = 1.0, 2.0, 3.0
// \Omega = ZC^T(CC^T)^{-1}
pub fn prepare_m_isdf_weight(k_mu: usize, mol: &Molecule, grids: &dft::Grids, occupation: &[Vec<f64>; 2], scftype: SCFType) -> (MatrixFull<f64>, MatrixFull<f64>) {
    let nao = mol.num_basis;
    let nri = mol.num_auxbas;

    let rgrids = &grids.coordinates;
    let ngrids = rgrids.len();
    let mut phi = tabulated_ao(&mol, &rgrids);

    //prepare weight function
    let mut lambda_r = vec![0.0; ngrids];
    let alpha = 1.0;
    for i in 0..ngrids{
        let mut sum = 1.0;
        let scaling_factor = match scftype {
            SCFType::RHF => 2.0,
            _ => 1.0,
        };

        //===========================weight_function================================
        //\omega(r_j) = (\sum_1^{N_e}|\phi_i|^\alpha)+(\sum_1^{N}|\phi_i|^\alpha)
        if scaling_factor == 2.0{
            phi.iter_column(i).zip(occupation[0].iter()).for_each(|(ao, occ)|{
                sum += occ * ao.powf(alpha) + scaling_factor * ao.powf(alpha);
            });
        }else{
            phi.iter_column(i).zip(occupation[0].iter()).for_each(|(ao, occ)|{
                sum += occ * ao.powf(alpha) + scaling_factor * ao.powf(alpha);
            });
            phi.iter_column(i).zip(occupation[1].iter()).for_each(|(ao, occ)|{
                sum += occ * ao.powf(alpha) + scaling_factor * ao.powf(alpha);
            });
        }

        //====================================================================

        lambda_r[i] = sum;

    }


    let n_mu = k_mu * nao;
    let mut lambda_r_for_isdf = vec![0.0; ngrids];
    lambda_r_for_isdf.iter_mut().zip(lambda_r.iter()).for_each(|(x,y)|{
        *x = y.abs();
    });
    let mut lambda_phi = phi.clone();

    let (ip, ip_weights) = cvt_isdf_v2(&rgrids, &lambda_r_for_isdf, n_mu);
    let mut varphi = tabulated_ao(&mol, &ip);
    let mut lambda_varphi = MatrixFull::new([nao,n_mu],0.0);
    lambda_varphi.iter_columns_full_mut().zip(ip_weights.iter().zip(varphi.iter_columns_full()))
    .for_each(|(x, (weight,aos))|{
        x.iter_mut().zip(aos.iter()).for_each(|(y,ao)|{
            *y = weight * ao;
        });
    });

    let mut c21 = MatrixFull::new([n_mu, n_mu], 0.0);
    let mut lambda_varphi_mid = lambda_varphi.clone();
    c21.lapack_dgemm(&mut lambda_varphi, &mut lambda_varphi_mid, 'T', 'N', 1.0, 0.0);
    let mut c22 = MatrixFull::new([n_mu, n_mu], 0.0);
    let mut varphi_mid = varphi.clone();
    c22.lapack_dgemm(&mut varphi, &mut varphi_mid, 'T', 'N', 1.0, 0.0);
    let mut c2 = MatrixFull::new([n_mu, n_mu], 0.0);
    c2.iter_columns_full_mut().zip(c21.iter_columns_full()).zip(c22.iter_columns_full())
        .for_each(|((x,a),b)|{
            x.iter_mut().zip(a.iter()).zip(b.iter()).for_each(|((x1,a1),b1)|{
                *x1 = a1 * b1;
            });
        });

    let mut cint_data = mol.initialize_cint(true);
    let n_basis_shell = mol.cint_bas.len();
    let n_auxbas_shell = mol.cint_aux_bas.len();
    let mut ri3fn = RIFull::new([nao,nao,nri],0.0);
    cint_data.cint2c2e_optimizer_rust();
    let mut ri_v_ri = MatrixFull::new([nri,nri],0.0);
    for l in 0..n_auxbas_shell {
        let basis_start_l = mol.cint_aux_fdqc[l][0];
        let basis_len_l = mol.cint_aux_fdqc[l][1];
        let gl  = l + n_basis_shell;
        for k in 0..n_auxbas_shell {
            let basis_start_k = mol.cint_aux_fdqc[k][0];
            let basis_len_k = mol.cint_aux_fdqc[k][1];
            let gk  = k + n_basis_shell;
            let buf = cint_data.cint_2c2e(gk as i32, gl as i32);
            
            let mut tmp_slices = ri_v_ri.iter_submatrix_mut(
                basis_start_k..basis_start_k+basis_len_k,
                basis_start_l..basis_start_l+basis_len_l);
            tmp_slices.zip(buf.iter()).for_each(|value| {*value.0 = *value.1});

        }
    }
    cint_data.cint3c2e_optimizer_rust();
    for k in 0..n_auxbas_shell {
        let basis_start_k = mol.cint_aux_fdqc[k][0];
        let basis_len_k = mol.cint_aux_fdqc[k][1];
        let gk  = k + n_basis_shell;
        for j in 0..n_basis_shell {
            let basis_start_j = mol.cint_fdqc[j][0];
            let basis_len_j = mol.cint_fdqc[j][1];
            for i in 0..n_basis_shell {
                let basis_start_i = mol.cint_fdqc[i][0];
                let basis_len_i = mol.cint_fdqc[i][1];
                let buf = RIFull::from_vec([basis_len_i, basis_len_j,basis_len_k], 
                    cint_data.cint_3c2e(i as i32, j as i32, gk as i32)).unwrap();
                ri3fn.copy_from_ri(
                    basis_start_i..basis_start_i+basis_len_i,
                    basis_start_j..basis_start_j+basis_len_j,
                    basis_start_k..basis_start_k+basis_len_k,
                    & buf, 
                    0..basis_len_i, 
                    0..basis_len_j, 
                    0..basis_len_k);
            }
        }
    }
        cint_data.final_c2r();

    let mut ri_v_ao_t = MatrixFull::from_vec([nao*nao, nri],ri3fn.data).unwrap();

    let mut c = prod_states_gw(&lambda_varphi.transpose(), &varphi.transpose());
    let mut tmp1 = MatrixFull::new([nri, n_mu],0.0);
    tmp1.lapack_dgemm(&mut ri_v_ao_t, &mut c, 'T', 'T', 1.0, 0.0);

    let mut tmp0 = MatrixFull::new([nri,n_mu],0.0);

    //=====================test_pinv_time==========================
    let mut time_mark = utilities::TimeRecords::new();
    time_mark.new_item("test pinv", "pseudo inverse of c2");
    time_mark.count_start("test pinv");
    let mut inv_cctrans = c2.pinv(1.0e-12);
    time_mark.count("test pinv");
    time_mark.report_all();
    //=============================================================

    tmp0.lapack_dgemm(&mut tmp1, &mut inv_cctrans, 'N', 'N', 1.0, 0.0);

    let mut tmp01 = tmp0.clone();
    let mut tmp = ri_v_ri.lapack_dgesv(&mut tmp01, nri as i32);
    let mut kernel_part = MatrixFull::new([n_mu,n_mu], 0.0);
    kernel_part.lapack_dgemm(&mut tmp0, &mut tmp, 'T', 'N', 1.0, 0.0);
    (varphi, kernel_part)
}

pub fn ip_from_dm_dense(tab_ao: &MatrixFull<f64>, coordinates: &Vec<[f64;3]>, lambda_r: &Vec<f64>, mol: &Molecule, dm:&Vec<MatrixFull<f64>>, k_mu: usize, batch_size: Option<usize>) -> (Vec<[f64;3]>, Vec<f64>, usize){
    /// for dense tabulated_ao, which is saved as MatrixFull<f64>
    let spin_channel = mol.spin_channel;
    let mut varrho = tabulated_density_batch(coordinates, &tab_ao, dm, spin_channel, batch_size);
    let n_mu = k_mu * mol.num_auxbas;
    let ngrids_ori = coordinates.len();

    if spin_channel == 2 {
        for i in 0..ngrids_ori{
            varrho.data[i] = varrho.data[i] + varrho.data[ngrids_ori + i];
        }
        varrho.data.truncate(ngrids_ori);
        varrho.size = [ngrids_ori,1];
    }

    let threshold = 1.0e-3 / ngrids_ori.to_f64().unwrap();
    let index: Vec<usize> = varrho.data.iter().enumerate()
    .filter(|&(_, &value)| value >= threshold)
    .sorted_by(|a, b| b.1.partial_cmp(&a.1).unwrap())
    .map(|(index, _)| index)
    .collect();
    drop(varrho);

    let index_len = index.len();
    let interval = (index_len-1) / (n_mu - 1);
    let mut ip = vec![[0.0;3]; n_mu];
    let mut weight = vec![0.0; n_mu];

    for i in 0..n_mu{
        let i_grid = index[i * interval];
        ip[i] = coordinates[i_grid];
        weight[i] = lambda_r[i_grid]
    }

    (ip, weight, n_mu)
}

pub fn ip_from_dm_compressed(grids: &dftgrids, mol: &Molecule, dm: &Vec<MatrixFull<f64>>, k_mu: usize) -> (Vec<[f64; 3]>, Vec<f64>, usize) {
    /// for compressed tabulated_ao, which is saved as CompressedAO
    let ao_c = match &grids.ao_compressed {
        Some(c) => c,
        None => panic!("This function should be used if AO is compressed. This must be a bug!"),
    };
    let spin_channel = dm.len();
    let n_mu = k_mu * mol.num_auxbas;

    let mut density_list: Vec<(usize, f64)> = Vec::new();
    for ibatch in 0..ao_c.batches.len() {
        let grid_range = ao_c.batch_grid_ranges[ibatch].clone();
        let rho_batch = grids.prepare_tabulated_density_compressed(dm,spin_channel,grid_range.clone());
        for g_local in 0..rho_batch.size[0] {
            let total_rho = if spin_channel == 1 {
                rho_batch[[g_local, 0]]
            } else {
                rho_batch[[g_local, 0]] + rho_batch[[g_local, 1]]
            };
            density_list.push((grid_range.start + g_local, total_rho));
        }
    }

    density_list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let total = density_list.len();
    let mut coords = Vec::with_capacity(n_mu);
    let mut weights = Vec::with_capacity(n_mu);

    for i in 0..n_mu {
        let idx = if n_mu == 1 {
            0
        } else {
            ((i as f64 * (total - 1) as f64) / (n_mu - 1) as f64).round() as usize
        };
        let global_idx = density_list[idx].0;
        coords.push(grids.coordinates[global_idx]);
        weights.push(grids.weights[global_idx]);
    }

    (coords, weights, n_mu)
}

pub fn tabulated_density(coordinates: &Vec<[f64;3]>, ao:&MatrixFull<f64>, dm: &Vec<MatrixFull<f64>>, spin_channel: usize) -> MatrixFull<f64> {
    let num_grids = coordinates.len();
    let mut cur_rho = MatrixFull::new([num_grids,spin_channel],0.0);
    for i_spin in 0..spin_channel {
        let dt0 = utilities::init_timing();
        let dm_s = dm.get(i_spin).unwrap();
        let mut wao = MatrixFull::new(ao.size.clone(),0.0);
        _dgemm_full(dm_s,'N',ao,'N',&mut wao, 1.0, 0.0);
        let dt1 = utilities::timing(&dt0, Some("Evalute weighted ao (wao)"));
        ao.par_iter_columns_full().zip(wao.par_iter_columns_full()).map(|(ao_r,wao_r)| (ao_r,wao_r))
        .zip(cur_rho.par_iter_column_mut(i_spin))
        .for_each(|((ao_r,wao_r),cur_rho_s)| {
            *cur_rho_s = wao_r.iter().zip(ao_r.iter()).fold(0.0, |acc, (a,b)| {
                acc + a*b
            })
        });
        let dt2 = utilities::timing(&dt1, Some("Contracting ao*wao"));
        
    };
    cur_rho
}

pub fn tabulated_density_batch(coordinates: &Vec<[f64;3]>, ao: &MatrixFull<f64>, dm: &Vec<MatrixFull<f64>>, spin_channel: usize, batch_size: Option<usize>) -> MatrixFull<f64> {
    // apply tabulated_density with batch
    // i.e. generate wao and rho by batch
    if batch_size.is_none() {
        // use origin version
        tabulated_density(coordinates, ao, dm, spin_channel)
    } else {
        let block_size = batch_size.unwrap();
        let num_grids = coordinates.len();
        let mut cur_rho = MatrixFull::new([num_grids, spin_channel], 0.0);
        let nao = ao.size[0];

        let trans_n = b'N' as c_char;
        let alpha: f64 = 1.0;
        let beta: f64 = 0.0;
        let m = nao as i32;
        let k = nao as i32;
        let lda = nao as i32;
        let ldb = nao as i32;
        let ldc = nao as i32;

        for i_spin in 0..spin_channel {
            let dm_s = &dm[i_spin];
            let dm_vec = &dm_s.data;

            let mut start = 0;
            while start < num_grids {
                let end = (start + block_size).min(num_grids);
                let block_len = end - start;
                let n = block_len as i32;

                let ao_chunk = &ao.data[start * nao..end * nao];

                let mut wao_chunk: Vec<f64> = vec![0.0; nao * block_len];

                dgemm_ffi(
                    dm_vec,
                    &ao_chunk,
                    &mut wao_chunk,
                    &trans_n,
                    &trans_n,
                    &m,
                    &n,
                    &k,
                    &alpha,
                    &lda,
                    &ldb,
                    &beta,
                    &ldc,
                );

                let rho_col = &mut cur_rho.data[i_spin * num_grids..(i_spin + 1) * num_grids];
                let rho_slice = &mut rho_col[start..end];

                let ao_cols: Vec<&[f64]> = (0..block_len)
                    .map(|j| &ao_chunk[j * nao..(j + 1) * nao])
                    .collect();
                let wao_cols: Vec<&[f64]> = (0..block_len)
                    .map(|j| &wao_chunk[j * nao..(j + 1) * nao])
                    .collect();

                rho_slice
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(idx, rho_val)| {
                        let ao_col = ao_cols[idx];
                        let wao_col = wao_cols[idx];
                        *rho_val = wao_col
                            .iter()
                            .zip(ao_col.iter())
                            .map(|(w, a)| w * a)
                            .sum();
                    });

                start = end;
            }
        }
        cur_rho
    }
}

// CURRENT VERSION
pub fn prepare_m_isdf_dm_v2(k_mu: usize, mol: &mut Molecule, grids: &mut Option<dft::Grids>, dm:&Vec<MatrixFull<f64>>, batch_size: Option<usize>) -> (MatrixFull<f64>, MatrixFull<f64>) {
    let nao = mol.num_basis;
    let nri = mol.num_auxbas;

    let (ip, ip_weights, n_mu) = if let Some(grids_ref) = grids.as_ref() {
        ip_from_dm_dense(
            &grids_ref.ao.as_ref().unwrap(), 
            &grids_ref.coordinates, 
            &grids_ref.weights, 
            &mol, 
            &dm, 
            k_mu,
            batch_size)
    } else {
        panic!("This is a bug.");
        (vec![], vec![], 0)
    };
    // delete ao and generate them (and aop if needed) after isdf procedure
    let dftgrids = grids.as_mut().unwrap();
    dftgrids.ao = None;
    dftgrids.aop = None;
    dftgrids.ao_compressed = None;
    dftgrids.aop_compressed = None;
    // check, and test
    assert!(grids.as_ref().unwrap().ao.is_none());
    assert!(grids.as_ref().unwrap().aop.is_none());
    assert!(grids.as_ref().unwrap().ao_compressed.is_none());
    assert!(grids.as_ref().unwrap().aop_compressed.is_none());

    let mut varphi = tabulated_ao(&mol, &ip);
    let mut lambda_varphi = MatrixFull::new([nao,n_mu],0.0);
    lambda_varphi.iter_columns_full_mut().zip(ip_weights.iter().zip(varphi.iter_columns_full()))
    .for_each(|(x, (weight,aos))|{
        x.iter_mut().zip(aos.iter()).for_each(|(y,ao)|{
            *y = weight * ao;
        });
    });
    // format S and S^-1 first
    // use dgemm_ffi to aviod clone
    // work1 for lambda_varphi.T @ lambda_varphi and s
    // work2 for varphi.T @ varphi
    // necessary parameters
    let transn = b'N' as c_char;
    let transt = b'T' as c_char;
    let nip = n_mu.clone() as i32;
    let nbasis = nao.clone() as i32;
    let mut work1 = vec![0.0;n_mu*n_mu];
    let mut work2 = vec![0.0;n_mu*n_mu];
    dgemm_ffi(&lambda_varphi.data, &lambda_varphi.data, &mut work1, &transt, &transn, &nip, &nip, &nbasis, &1.0, &nbasis, &nbasis, &0.0, &nip);
    dgemm_ffi(&varphi.data, &varphi.data, &mut work2, &transt, &transn, &nip, &nip, &nbasis, &1.0, &nbasis, &nbasis, &0.0, &nip);
    work1.par_iter_mut().zip(work2.par_iter()).for_each(|(dst,src)| {
        *dst *= *src;
    });
    // let all elements in work2 be 0.0
    work2.fill(0.0);
    let mut work1 = MatrixFull::from_vec([n_mu, n_mu], work1).unwrap();

    //=====================test_pinv_time==========================
    let mut time_mark = utilities::TimeRecords::new();
    time_mark.new_item("pinv", "pseudo inverse of S");
    time_mark.count_start("pinv");
    /// DGESDD
//    println!("Apply pinv_sdd");
//    let mut inv_cctrans = pinv_sdd(&mut mat_s, 1.0e-12);
//    println!("pinv_sdd done");
    /// DGESDD
//    println!("Apply pinv_sdd_i64");
//    drop(work2);
//    let mut inv_cctrans = pinv_sdd_i64(&mut work1, 1.0e-12);
//    println!("pinv_sdd_i64 done");
    /// DGESVD
//    println!("Apply svd pinv");
//    drop(work2);
//    let mut inv_cctrans = work1.pinv(1.0e-12);
//    println!("svd pinv done");
    /// DGEQP3
//    println!("Apply qr pinv");
    let mut inv_cctrans = _qrpinv(&mut work1, work2, Some(1e-10)).unwrap();
//    println!("qr pinv done");
    time_mark.count("pinv");
    //=============================================================

    time_mark.new_item("RI", "necessary RI calculation");
    time_mark.count_start("RI");
    let mut cint_data = mol.initialize_cint(true);
    let n_basis_shell = mol.cint_bas.len();
    let n_auxbas_shell = mol.cint_aux_bas.len();

    let mut ri_v_ri = mol.int_ij_aux_columb();
    let mut mat_a = MatrixFull::new([n_mu, nri], 0.0);
    let mut tmp_a = MatrixFull::new([nao, n_mu], 0.0);
    cint_data.cint3c2e_optimizer_rust();
    for k in 0..n_auxbas_shell {
        let basis_start_k = mol.cint_aux_fdqc[k][0];
        let basis_len_k = mol.cint_aux_fdqc[k][1];
        let mut tmp_part = RIFull::new([nao, nao, basis_len_k], 0.0);
        let gk  = k + n_basis_shell;

        for j in 0..n_basis_shell {
            let basis_start_j = mol.cint_fdqc[j][0];
            let basis_len_j = mol.cint_fdqc[j][1];
            for i in 0..n_basis_shell {
                let basis_start_i = mol.cint_fdqc[i][0];
                let basis_len_i = mol.cint_fdqc[i][1];
                let buf = RIFull::from_vec([basis_len_i, basis_len_j,basis_len_k], 
                    cint_data.cint_3c2e(i as i32, j as i32, gk as i32)).unwrap();
                tmp_part.copy_from_ri(
                    basis_start_i..basis_start_i+basis_len_i,
                    basis_start_j..basis_start_j+basis_len_j,
                    0..basis_len_k,
                    & buf, 
                    0..basis_len_i, 
                    0..basis_len_j, 
                    0..basis_len_k);
            }
        }

        mat_a.iter_columns_mut(basis_start_k..basis_start_k+basis_len_k).zip(tmp_part.iter_auxbas(0..basis_len_k).unwrap()).for_each(|(aux_k,part)|{
            // change to dgemm_ffi to aviod clone
            let transt = b'T' as c_char;
            let transn = b'N' as c_char;
            let n1 = nao.clone() as i32;
            let n2 = n_mu.clone() as i32;
            dgemm_ffi(part, &lambda_varphi.data, &mut tmp_a.data, &transt, &transn, &n1, &n2, &n1, &1.0, &n1, &n1, &0.0, &n1);
            aux_k.par_iter_mut().zip(tmp_a.par_iter_columns_full()).zip(varphi.par_iter_columns_full()).for_each(|((x,t_i),phi_i)|{
                t_i.iter().zip(phi_i.iter()).for_each(|(t_ij, phi_ij)|{
                    *x += *t_ij * *phi_ij
                });
            });
        });

    }
    
    cint_data.final_c2r(); 
    time_mark.count("RI");
    time_mark.report_all();

    let mut tmp0 = MatrixFull::new([nri,n_mu],0.0);
    tmp0.lapack_dgemm(&mut mat_a, &mut inv_cctrans, 'T', 'N', 1.0, 0.0);
    drop(mat_a);
    drop(inv_cctrans);
    let mut tmp01 = tmp0.clone();
    let mut tmp = ri_v_ri.lapack_dgesv(&mut tmp01, nri as i32);
    let mut kernel_part = MatrixFull::new([n_mu,n_mu], 0.0);
    kernel_part.lapack_dgemm(&mut tmp0, &mut tmp, 'T', 'N', 1.0, 0.0);

    kernel_part.data.par_chunks_mut(n_mu)
        .enumerate()
        .for_each(|(j, col)| {
            let wj = ip_weights[j];
            for i in 0..n_mu {
                col[i] *= ip_weights[i] * wj;
            }
        });

    // generate ao and aop again here (if needed)
    // copy from prepare_density_grids
    if mol.xc_data.is_dfa_scf() || mol.ctrl.initial_guess == "vsap" {
        if let Some(dftgrids) = grids {
            dftgrids.ao_cutoff = mol.ctrl.ao_cutoff;
            if dftgrids.ao_cutoff > 0.0 {
                // sparse path: two-pass batch scan → compressed directly, no dense allocation
                dftgrids.prepare_tabulated_ao_sparse(&mol);
            } else {
                // dense path: allocate full AO/AOP, then optionally build non0tab + compressed
                dftgrids.prepare_tabulated_ao(&mol);
                dftgrids.build_non0tab(&mol);
                dftgrids.build_compressed_storage();
            }
            if mol.ctrl.drop_dense_ao && dftgrids.ao_compressed.is_some() {
                if mol.ctrl.print_level >= 1 {
                    let (dense_bytes, comp_bytes, _) = dftgrids.memory_footprint();
                    let ratio = if dense_bytes > 0 { comp_bytes as f64 / dense_bytes as f64 * 100.0 } else { 0.0 };
                    println!(" [non0tab] dropping dense ao/aop, dense={:.1}GB, compressed={:.1}GB ({:.1}%)",
                        dense_bytes as f64 / 1e9, comp_bytes as f64 / 1e9, ratio);
                }
                dftgrids.ao = None;
                dftgrids.aop = None;
            }
        }
    }

    estimate_memory_for_isdf_new(mol);
    (varphi, kernel_part)
}

pub fn call_isdf_k_generator(mol: &Molecule) -> usize {
    let project_root = std::env!("CARGO_MANIFEST_DIR");
    let file_path = std::path::Path::new(project_root).parent().unwrap().join(file!());
    let current_dir = file_path.parent().unwrap();
    let wrapper_path = current_dir.join("set_k_auto.py");
    let python_command = find_current_python().unwrap();
    let output = Command::new(python_command.as_str())
        .arg(wrapper_path.to_str().unwrap())
        .arg(current_dir.to_str().unwrap())
        .arg(&mol.ctrl.ctrl_file.as_str())
        .output().unwrap();

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("Python script failed: {}", stderr);
    };

    let stdout_str = str::from_utf8(&output.stdout).unwrap();
    let result: usize = stdout_str.trim().parse().unwrap();
    result
}

pub fn set_isdf_k(mol: &mut Molecule) {
    if mol.ctrl.isdf_k.is_some() {
        return
    } else {
        mol.ctrl.isdf_k = Some(call_isdf_k_generator(&mol));
        println!("k_IP in ISDF has been set to {} by empirical rules", mol.ctrl.isdf_k.as_ref().unwrap());
    }
}

pub fn estimate_memory_for_isdf_new(mol: &mut Molecule) {
    use std::cmp::max;
    // estimate memory costs for isdf
    // and then set ctrl.max_memory for ri-j-on-the-fly in MB
    assert!(&mol.ctrl.use_isdf);
    // some essential parameters
    let num_basis = mol.num_basis;
    let isdf_k = mol.ctrl.isdf_k.unwrap();
    let num_aux_basis = mol.num_auxbas;
    let nip = isdf_k*num_aux_basis;
    // It is a rough estimate!
    let nocc = (mol.num_elec[0] / 2.0).floor() as usize;
    let nvir = num_basis - nocc;
    let mut est_mem = 0.0;
    // for HF and hybrid DFT
    est_mem = (((num_basis*num_basis + num_basis*nip + nip*nip) * 8) as f64) / 1024.0 / 1024.0;
    mol.ctrl.max_memory_backup = mol.ctrl.max_memory.clone();
    mol.ctrl.max_memory = Some(est_mem);
}

// a simple wrapper for dgemm
pub fn dgemm_ffi(mat_a: &[f64], mat_b: &[f64], mat_c: &mut [f64], transa: &c_char, transb: &c_char, m: &i32, n: &i32, k: &i32, alpha: &f64, lda: &i32, ldb: &i32, beta: &f64, ldc: &i32) {
    unsafe {
        dgemm_(
            transa as *const c_char,
            transb as *const c_char,
            m as *const i32,
            n as *const i32,
            k as *const i32,
            alpha as *const f64,
            mat_a.as_ptr(),
            lda as *const i32,
            mat_b.as_ptr(),
            ldb as *const i32,
            beta as *const f64,
            mat_c.as_mut_ptr(),
            ldc as *const i32,
        );
    }
}

// use ffi to perform pinv_sdd
pub fn pinv_sdd_i64(mat: &mut MatrixFull<f64>, threshold: f64) -> MatrixFull<f64> {
    let jobz = b'A' as c_char;
    let m = mat.size[0];
    //m = n = lda = ldu = ldvt
    let mut s = vec![0.0; m];
    let mut u = vec![0.0; (m * m)];
    //use jobz = 'O', the mat itself will be u
    let mut vt = vec![0.0;(m * m)];
    let lwork = 5 * m * m + 10 * m;
    let mut work = vec![0.0; lwork];
    let mut iwork = vec![0; (8 * m)];
    let mut info = 0;
    let m = m as i32;
    let lwork = lwork as i64;
    //u -> mat, vt -> vt, sigma -> s
    unsafe {
        dgesdd_(
            &jobz as *const c_char,
            &m as *const i32,
            &m as *const i32,
            mat.data.as_mut_ptr(),
            &m as *const i32,
            s.as_mut_ptr(),
            u.as_mut_ptr(),
            &m as *const i32,
            vt.as_mut_ptr(),
            &m as *const i32,
            work.as_mut_ptr(),
            &lwork as *const i64,
            iwork.as_mut_ptr(),
            &mut info as *mut i32,
            );
    };
    if info != 0 {
        panic!("Lapack dgesdd failed");
    };
    //filter s and get s_inv
    let cutoff = threshold * s[0];
    s.par_iter_mut().for_each(|x| {
        if *x < cutoff {
            *x = 0.0;
        } else {
            *x = x.recip();
        }
    });
    //Note u_mat = mat.data
    let mut u_mat = MatrixFull::from_vec([m as usize,m as usize], u).unwrap();
    for j in 0..(m as usize) {
        u_mat.iter_column_mut(j).for_each(|x| {
            *x *= s[j];
        });
    };
    let mut inv_s = vec![0.0;(m as usize) * (m as usize)];
    let transa = b'T' as c_char;
    let transb = b'T' as c_char;
    let alpha = 1.0;
    let beta = 0.0;
    unsafe {
        dgemm_(
            &transa as *const c_char,
            &transb as *const c_char,
            &m as *const i32,
            &m as *const i32,
            &m as *const i32,
            &alpha as *const f64,
            vt.as_ptr(),
            &m as *const i32,
            u_mat.data.as_ptr(),
            &m as *const i32,
            &beta as *const f64,
            inv_s.as_mut_ptr(),
            &m as *const i32,
            );
    };
    let inv_s = MatrixFull::from_vec([m as usize,m as usize],inv_s).unwrap();
    inv_s
}

#[link(name="lapack")]
extern "C" {
    fn dgesdd_(
        jobz: *const c_char,
        m: *const i32,
        n: *const i32,
        a: *mut c_double,
        lda: *const i32,
        s: *mut c_double,
        u: *mut c_double,
        ldu: *const i32,
        vt: *mut c_double,
        ldvt: *const i32,
        work: *mut c_double,
        lwork: *const i64,
        iwork: *mut c_int,
        info: *mut i32,
        );
}

#[link(name="lapack")]
extern "C" {
    fn dgemm_(
        transa: *const c_char,
        transb: *const c_char,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const f64,
        a: *const c_double,
        lda: *const i32,
        b: *const c_double,
        ldb: *const i32,
        beta: *const f64,
        c: *mut c_double,
        ldc: *const i32,
        );
}