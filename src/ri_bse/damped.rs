use crate::scf_io::{SCF, SCFType};
use crate::mpi_io::MPIOperator;
use std::path::Path;
use rest_libcint::rest_libcint_wrapper::int1e_r;
use tensors::{MathMatrix, MatrixFull, RIFull,MatrixFullSlice};
use crate::constants::{ANG, AU2DEBYE, SPECIES_INFO};
use itertools::Itertools;
use crate::ri_gw::get_occupation_parameters;
use std::f64::consts::SQRT_2;
use rest_tensors::matrix::matrix_blas_lapack::{_dgeev, _dgemv,_dgemm_full,_dsolve};
use crate::ri_bse;
use crate::ri_bse::davidson_solver::{dot_product,num_product,vector_scaled_add};
use crate::ctrl_io::quasiparticle_methods::QuasiParticle;
use rayon::prelude::*;
use std::time::Instant;
use std::sync::atomic::{AtomicPtr, Ordering};
use crate::basis_io::{self, spheric_gto_value_serial};
use crate::molecule_io::Molecule;
use std::fs::OpenOptions;
use std::io::Write;


/// 计算所有AO在给定空间格点上的波函数值
/// 返回 [N_grid, N_AO] 矩阵，行是格点，列是AO index（与SCF.eigenvectors的AO顺序一致）
pub fn eval_ao_on_grids(
    mol: &Molecule,
    grids: &[[f64; 3]]
) -> MatrixFull<f64> {
    let num_basis = mol.num_basis;
    let n_grid = grids.len();

    // 首先计算 [num_basis, n_grid]，然后转置
    let mut ao = MatrixFull::new([num_basis, n_grid], 0.0);

    mol.basis4elem.iter().zip(mol.geom.position.iter_columns_full()).for_each(|(elem, geom)| {
        let start = elem.global_index.0;
        let num_bas_elem = elem.global_index.1;
        let mut tmp_geom = [0.0; 3];
        tmp_geom.iter_mut().zip(geom.iter()).for_each(|(to, from)| { *to = *from; });

        // tab_den: [num_bas_elem, n_grid]
        let tab_den = spheric_gto_value_serial(grids, &tmp_geom, elem);

        // 拷贝到 ao[start..start+num_bas_elem, 0..n_grid]
        ao.copy_from_matr(start..start + num_bas_elem, 0..n_grid, &tab_den, 0..num_bas_elem, 0..n_grid);
    });

    // 转置为 [n_grid, num_basis]
    ao
}

pub fn obtain_mu_ia_z(eigenvectors:&MatrixFull<f64>,ao_dip:&RIFull<f64>,i:usize,a:usize)->f64{
    let num_state=eigenvectors.size[0];
    let ao_dip_tmp = ao_dip.get_reducing_matrix(2).unwrap();
    let mu_ia:f64=(0..num_state).cartesian_product(0..num_state).fold(0.0,|acc,(m,n)|{
        let idx = m * ao_dip_tmp.size[0] + n;
        acc+eigenvectors[[m,i]]*eigenvectors[[n,a]]*ao_dip_tmp.data[idx]});
    mu_ia
}
pub fn compute_mu_z_vec(scf_data:&SCF)->Vec<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let ao_dip=ri_bse::dipoles::obtain_ao_dips(scf_data,None);
    let mut mu_z_vec=vec![0.0;occ_size*vir_size];
    let eigenvectors=scf_data.eigenvectors[0].clone();
    //eigenvectors.formated_output(1000,"full");
    (0..occ_size).cartesian_product(0..vir_size).for_each(|(i,a)|{
        let mu_ia=obtain_mu_ia_z(&eigenvectors,&ao_dip,i,a+occ_size);
        mu_z_vec[i+a*occ_size]=mu_ia;
    });
    //dipole_matrix.formated_output(1000,"full");
    mu_z_vec
}
pub fn prepare_p0_r_i(mu_z_vec:&Vec<f64>,gwqp:&Vec<f64>,omega:f64,gamma:f64,occ_size:usize,vir_size:usize)->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>){
    let mut p0rp=vec![0.0;occ_size*vir_size];
    let mut p0rm=vec![0.0;occ_size*vir_size];
    let mut p0ip=vec![0.0;occ_size*vir_size];
    let mut p0im=vec![0.0;occ_size*vir_size];
    (0..occ_size).cartesian_product(0..vir_size).for_each(|(i,a)|{
        let mu_ia_z=mu_z_vec[i+a*occ_size];
        let energy_gap=gwqp[a+occ_size]-gwqp[i];
        // Non-interacting response for GMRES convention (omega -> omega - i*gamma):
        //   A_4c_nonint * z_nonint = b_4c
        //   z0_nonint = P*(D-omega)/Delta   (rp component)
        //   z2_nonint = -gamma*P/Delta      (ip component)
        //   z1_nonint = P*(D+omega)/Delta_plus  (rm component)
        //   z3_nonint = gamma*P/Delta_plus      (im component)
        p0rp[i+a*occ_size]=(energy_gap-omega)/((omega-energy_gap).powf(2.0)+gamma.powf(2.0))*mu_ia_z;
        p0rm[i+a*occ_size]=(omega+energy_gap)/((omega+energy_gap).powf(2.0)+gamma.powf(2.0))*mu_ia_z;
        p0ip[i+a*occ_size]=-gamma/((omega-energy_gap).powf(2.0)+gamma.powf(2.0))*mu_ia_z;
        p0im[i+a*occ_size]= gamma/((omega+energy_gap).powf(2.0)+gamma.powf(2.0))*mu_ia_z;
    });
    (p0rp,p0rm,p0ip,p0im)
}
pub fn a_block_matvec(scf_data:&SCF,ri_vv:&MatrixFull<f64>,ri_ov:&MatrixFull<f64>,ri_oo_tilde:&MatrixFull<f64>,z_vec:&Vec<f64>,qp_ctrl:&QuasiParticle)->Vec<f64>{
    let mut result=ri_bse::matvec::w_contribution_a_block_dgemm(scf_data,ri_vv,z_vec,ri_oo_tilde,&qp_ctrl);
    result=ri_bse::matvec::coulomb_contribution(ri_ov,z_vec).iter().zip(result.iter()).map(|(v_i,z_i)|v_i+z_i).collect();
    result
}
pub fn b_block_matvec(scf_data:&SCF,ri_ov_a:&MatrixFull<f64>,ri_ov_b:&MatrixFull<f64>,ri_ov_tilde:&MatrixFull<f64>,z_vec:&Vec<f64>)->Vec<f64>{
    let mut result=ri_bse::matvec::w_contribution_b_block_dgemm(scf_data,ri_ov_b,z_vec,ri_ov_tilde);
    result=ri_bse::matvec::coulomb_contribution(ri_ov_a,z_vec).iter().zip(result.iter()).map(|(v_i,z_i)|v_i+z_i).collect();
    result
}
pub fn pairvec_matvec(scf_data:&SCF,ri_vv:&MatrixFull<f64>,ri_ov:&MatrixFull<f64>,ri_oo_tilde:&MatrixFull<f64>,
    ri_ov_w:&MatrixFull<f64>,ri_ov_tilde:&MatrixFull<f64>,p_vec:&Vec<f64>,m_vec:&Vec<f64>,
    qp_ctrl:&QuasiParticle)->(Vec<f64>,Vec<f64>){
    let ap=a_block_matvec(scf_data,ri_vv,ri_ov,ri_oo_tilde,p_vec,qp_ctrl);
    let am=a_block_matvec(scf_data,ri_vv,ri_ov,ri_oo_tilde,m_vec,qp_ctrl);
    let bp=b_block_matvec(scf_data,ri_ov,ri_ov_w,ri_ov_tilde,p_vec);
    let bm=b_block_matvec(scf_data,ri_ov,ri_ov_w,ri_ov_tilde,m_vec);
    (ap.iter().enumerate().map(|(k,apk)|apk+bm[k]).collect(),bp.iter().enumerate().map(|(k,bpk)|bpk+am[k]).collect())
}
pub fn update_w_vecs<F1>(rp:&Vec<f64>,rm:&Vec<f64>,ip:&Vec<f64>,im:&Vec<f64>,pair_matvec:F1,
    gwqp:&Vec<f64>,omega:f64,gamma:f64,occ_size:usize,vir_size:usize)->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)
    where F1:Fn(&(Vec<f64>,Vec<f64>))->(Vec<f64>,Vec<f64>){
    let preal=(rp.clone(),rm.clone());
    let pimag=(ip.clone(),im.clone());
    let (krp,krm)=pair_matvec(&preal);
    let (kip,kim)=pair_matvec(&pimag);
    let mut wrp=vec![0.0;occ_size*vir_size];
    let mut wrm=vec![0.0;occ_size*vir_size];
    let mut wip=vec![0.0;occ_size*vir_size];
    let mut wim=vec![0.0;occ_size*vir_size];
    (0..occ_size).cartesian_product(0..vir_size).for_each(|(i,a)|{
        let energy_gap=gwqp[a+occ_size]-gwqp[i];
        let prefactor_p:f64=1.0/((omega-energy_gap).powf(2.0)+gamma.powf(2.0));
        let prefactor_m:f64=1.0/((omega+energy_gap).powf(2.0)+gamma.powf(2.0));
        // T(z) = A_4c_diag^{-1} * Interaction  (exact equivalence with GMRES)
        // wrp/ip: sign flip needed (rp/ip block)
        wrp[i+a*occ_size]=-prefactor_p*((omega-energy_gap)*krp[i+a*occ_size]-gamma*kip[i+a*occ_size]);
        wip[i+a*occ_size]=-prefactor_p*((omega-energy_gap)*kip[i+a*occ_size]+gamma*krp[i+a*occ_size]);
        // wrm: no sign flip (rm block)
        wrm[i+a*occ_size]= prefactor_m*((omega+energy_gap)*krm[i+a*occ_size]-gamma*kim[i+a*occ_size]);
        // wim: sign of (omega+energy_gap)*kim term flipped relative to original
        wim[i+a*occ_size]= prefactor_m*((omega+energy_gap)*kim[i+a*occ_size]+gamma*krm[i+a*occ_size]);
        // Diagonal correction (cancels D contribution, forces T(z_ni)=0)
        wrp[i+a*occ_size] += energy_gap * prefactor_p * ((omega-energy_gap)*rp[i+a*occ_size] - gamma*ip[i+a*occ_size]);
        wrm[i+a*occ_size] -= energy_gap * prefactor_m * ((omega+energy_gap)*rm[i+a*occ_size] - gamma*im[i+a*occ_size]);
        wip[i+a*occ_size] += energy_gap * prefactor_p * ((omega-energy_gap)*ip[i+a*occ_size] + gamma*rp[i+a*occ_size]);
        wim[i+a*occ_size] -= energy_gap * prefactor_m * ((omega+energy_gap)*im[i+a*occ_size] + gamma*rm[i+a*occ_size]);
    });
    (wrp,wrm,wip,wim)
} 
pub fn fourvec_dot_product(fourvec_1:&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>),fourvec_2:&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>))->f64{
    dot_product(&fourvec_1.0,&fourvec_2.0)+dot_product(&fourvec_1.1,&fourvec_2.1)+dot_product(&fourvec_1.2,&fourvec_2.2)+dot_product(&fourvec_1.3,&fourvec_2.3)
}
pub fn fourvec_scaled_add(fourvec_1:&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>),scale1:f64,fourvec_2:&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>),scale2:f64)->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>){
    (vector_scaled_add(&fourvec_1.0,scale1,&fourvec_2.0,scale2),vector_scaled_add(&fourvec_1.1,scale1,&fourvec_2.1,scale2),vector_scaled_add(&fourvec_1.2,scale1,&fourvec_2.2,scale2),vector_scaled_add(&fourvec_1.3,scale1,&fourvec_2.3,scale2))
}
pub fn poples_numerical_trick<F1,F2>(p0:&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>),update_w:F1,precond:F2,tol:f64,max_iterations:usize)->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)
    where F1:Fn(&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>))->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>),
          F2:Fn(&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>))->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>){
    const EPS: f64 = 1e-14;
    let p0p0sqrt=fourvec_dot_product(p0,p0).sqrt();
    let u0=(num_product(&p0.0,1.0/p0p0sqrt),num_product(&p0.1,1.0/p0p0sqrt),num_product(&p0.2,1.0/p0p0sqrt),num_product(&p0.3,1.0/p0p0sqrt));
    let occvir=p0.0.len();
    // Initialize subspace with u0 and w0 = T(u0)
    let w0 = update_w(&u0);
    let mut u_vecs:Vec<(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)>=vec![u0.clone()];
    let mut w_vecs:Vec<(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)>=vec![w0];
    let mut iter_count=0;
    loop{
        let m = u_vecs.len(); // current subspace size
        // Build Q[i][k] = u_i^T * w_k + δ_ik, q[i] = u_i^T * p0
        let mut q_mat = MatrixFull::new([m, m], 0.0);
        let mut q_vec = vec![0.0; m];
        for i in 0..m {
            q_vec[i] = fourvec_dot_product(&u_vecs[i], p0);
            for k in 0..m {
                let mut val = fourvec_dot_product(&u_vecs[i], &w_vecs[k]);
                if i == k { val += 1.0; }
                q_mat[[i, k]] = val;
            }
        }
        // Solve Q * t = q
        let coeff = match _dsolve(&q_mat, &q_vec) {
            Some(c) => c,
            None => {
                eprintln!("Pople solver: subspace solve failed at iteration {}", iter_count);
                break;
            }
        };
        // Form full-space solution: z = Σ t_k * u_k
        let mut z_p_result = (vec![0.0;occvir],vec![0.0;occvir],vec![0.0;occvir],vec![0.0;occvir]);
        for k in 0..m {
            z_p_result = fourvec_scaled_add(&z_p_result, 1.0, &u_vecs[k], coeff[k]);
        }
        // Residual: r = p0 - z - T(z)  (fresh T(z), avoids numerical drift in stored w_vecs)
        let t_of_z = update_w(&z_p_result);
        let mut r = p0.clone();
        r = fourvec_scaled_add(&r, 1.0, &z_p_result, -1.0);
        r = fourvec_scaled_add(&r, 1.0, &t_of_z, -1.0);
        let res_norm = fourvec_dot_product(&r, &r).sqrt();
        if res_norm < tol {
            println!("Pople Solver converged at iteration {} with residual {:.2e}", iter_count, res_norm);
            break;
        }
        iter_count += 1;
        println!("Pople iteration {}, subspace size {}, residual = {:.2e}, threshold = {}",
                 iter_count, m, res_norm, tol);
        if iter_count >= max_iterations {
            eprintln!("Warning: Pople solver max iterations ({}) reached, res={:.2e}", max_iterations, res_norm);
            break;
        }
        // JD correction: u_new = ε * M^{-1}*z - M^{-1}*r
        // ε = (z^T * M^{-1} * r) / (z^T * M^{-1} * z)
        let m_inv_z = precond(&z_p_result);
        let m_inv_r = precond(&r);
        let zt_m_inv_r = fourvec_dot_product(&z_p_result, &m_inv_r);
        let zt_m_inv_z = fourvec_dot_product(&z_p_result, &m_inv_z);
        let epsilon = if zt_m_inv_z.abs() > EPS { zt_m_inv_r / zt_m_inv_z } else { 0.0 };
        let mut u_new = fourvec_scaled_add(&m_inv_z, epsilon, &m_inv_r, -1.0);
        // MGS reorthogonalization × 2
        for _reortho in 0..2 {
            for existing_u in &u_vecs {
                let u_norm2 = fourvec_dot_product(existing_u, existing_u);
                if u_norm2 > EPS {
                    let dot_val = fourvec_dot_product(&u_new, existing_u);
                    u_new = fourvec_scaled_add(&u_new, 1.0, existing_u, -dot_val / u_norm2);
                }
            }
        }
        let u_new_norm = fourvec_dot_product(&u_new, &u_new).sqrt();
        if u_new_norm < EPS {
            println!("Pople solver: new direction norm too small ({:.2e}), stopping", u_new_norm);
            break;
        }
        let inv_unorm = 1.0 / u_new_norm;
        let v_new = (num_product(&u_new.0, inv_unorm),num_product(&u_new.1, inv_unorm),
                     num_product(&u_new.2, inv_unorm),num_product(&u_new.3, inv_unorm));
        let w_new = update_w(&v_new);
        u_vecs.push(v_new);
        w_vecs.push(w_new);
    }
    // Reconstruct final solution from best subspace coefficients
    let m = u_vecs.len();
    let mut q_mat = MatrixFull::new([m, m], 0.0);
    let mut q_vec = vec![0.0; m];
    for i in 0..m {
        q_vec[i] = fourvec_dot_product(&u_vecs[i], p0);
        for k in 0..m {
            let mut val = fourvec_dot_product(&u_vecs[i], &w_vecs[k]);
            if i == k { val += 1.0; }
            q_mat[[i, k]] = val;
        }
    }
    let mut p_result = (vec![0.0;occvir],vec![0.0;occvir],vec![0.0;occvir],vec![0.0;occvir]);
    if let Some(coeff) = _dsolve(&q_mat, &q_vec) {
        for k in 0..m {
            p_result = fourvec_scaled_add(&p_result, 1.0, &u_vecs[k], coeff[k]);
        }
    }
    p_result
}
pub fn solve_for_coeff(u_vecs:&Vec<(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)>,w_vecs:&Vec<(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)>,p0:&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>))->Vec<f64>{
    let n=u_vecs.len();
    let mut matrix=MatrixFull::new([n,n],0.0);
    (0..n).cartesian_product(0..n).for_each(|(i,k)|{
        let wkui=fourvec_dot_product(&u_vecs[i],&w_vecs[k]);
        matrix[[i,k]]=if i==k{wkui+1.0}else {wkui};
    });
    let mut b=vec![0.0;n];
    (0..n).for_each(|i|{
        b[i]=fourvec_dot_product(&u_vecs[i],p0);
    });
    _dsolve(&matrix,&b).unwrap()
}
pub fn calculate_residue(u_vecs:&Vec<(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)>,w_vecs:&Vec<(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)>,
    p0:&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>),coeff:&Vec<f64>)->f64{
    let mut residue=p0.clone();
    // Fix 4: Correct sign - residue = p0 - Σ c_k(u_k + w_k)
    (0..u_vecs.len()).for_each(|k|{
        residue=fourvec_scaled_add(&u_vecs[k],-coeff[k],&residue,1.0);
        residue=fourvec_scaled_add(&w_vecs[k],-coeff[k],&residue,1.0);
    });
    fourvec_dot_product(&residue,&residue).sqrt()
}
pub fn damped_bse(scf_data:&SCF)->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>){
    let qp_ctrl = scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    match qp_ctrl.damped_bse_solver.as_str() {
        "pople" => damped_bse_pople(scf_data),
        "gmres" => damped_bse_gmres(scf_data),
        "klopper" => damped_bse_klopper(scf_data),
        "dense" => damped_bse_dense(scf_data),
        other => panic!("Invalid damped_bse_solver: \"{}\". Expected \"pople\", \"gmres\", \"dense\", or \"klopper\".", other),
    }
}

pub fn damped_bse_pople(scf_data:&SCF)->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>){
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let inverse_dielectric=ri_bse::construct_inverse_dielectric(scf_data,&scf_data.eigenvalues[0]);
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    println!("Now begins damped BSE calculation (Pople). Parameters:\nOcc Size={},Vir Size={},External Field Freq={},Lifetime Gamma={}",occ_size,vir_size,qp_ctrl.external_field_freq,qp_ctrl.lifetime_gamma);
    let mu_z_vec=compute_mu_z_vec(scf_data);
    let ri_oo=ri_bse::get_submatrix(scf_data,'O','O','N');
    println!("num_auxbas={}",ri_oo.size[0]);
    let num_auxbas=ri_oo.size[0];
    let mut ri_oo_tilde:MatrixFull<f64>=MatrixFull::new(ri_oo.size,0.0);
    _dgemm_full(&inverse_dielectric,'N',&ri_oo,'N',&mut ri_oo_tilde,1.0,0.0);
    drop(ri_oo);
    ri_oo_tilde.reshape([num_auxbas*occ_size,occ_size]);
    ri_oo_tilde=ri_oo_tilde.transpose_and_drop();
    ri_oo_tilde.reshape([occ_size*num_auxbas,occ_size]);
    let ri_ov=ri_bse::get_submatrix(scf_data,'O','V','N');
    let mut ri_ov_w=ri_ov.clone();
    ri_ov_w.reshape([num_auxbas*occ_size,vir_size]);
    let mut ri_vv=ri_bse::get_submatrix(scf_data,'V','V','N');
    ri_vv.reshape([num_auxbas*vir_size,vir_size]);
    let mut ri_ov_tilde:MatrixFull<f64>=MatrixFull::new(ri_ov.size,0.0);
    _dgemm_full(&inverse_dielectric,'N',&ri_ov,'N',&mut ri_ov_tilde,1.0,0.0);
    drop(inverse_dielectric);
    ri_ov_tilde.reshape([num_auxbas*occ_size,vir_size]);
    let external_field_freq=qp_ctrl.external_field_freq;
    let lifetime_gamma=qp_ctrl.lifetime_gamma;
    let p0=prepare_p0_r_i(&mu_z_vec,&scf_data.gwqp.0,external_field_freq,lifetime_gamma,occ_size,vir_size);
    let wrapped_pair_matvec = |pairvec:&(Vec<f64>,Vec<f64>)| -> (Vec<f64>,Vec<f64>) {
        pairvec_matvec(scf_data,&ri_vv,&ri_ov,&ri_oo_tilde,
            &ri_ov_w,&ri_ov_tilde,&pairvec.0,&pairvec.1,
            &qp_ctrl)
    };
    let wrapped_update_w=|u:&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)|->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>){
        update_w_vecs(&u.0,&u.1,&u.2,&u.3,|x|wrapped_pair_matvec(&x),
            &scf_data.gwqp.0,external_field_freq,lifetime_gamma,occ_size,vir_size)
    };
    // Diagonal preconditioner: M = diag(A_4c)
    let energy_diag=ri_bse::construct_energy_diag_for_a(&scf_data.gwqp.0,occ_size,vir_size);
    let diag_13:Vec<f64>=energy_diag.iter().map(|d|d-external_field_freq).collect();
    let diag_24:Vec<f64>=energy_diag.iter().map(|d|d+external_field_freq).collect();
    let precond=|z:&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)|->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>){
        let apply_inv = |v:&Vec<f64>,d:&Vec<f64>|->Vec<f64>{
            v.iter().zip(d.iter()).map(|(vi,di)|{
                if di.abs()>1e-8 {vi/di} else {0.0}
            }).collect()
        };
        (apply_inv(&z.0,&diag_13),apply_inv(&z.1,&diag_24),
         apply_inv(&z.2,&diag_13),apply_inv(&z.3,&diag_24))
    };
    let result=poples_numerical_trick(&p0,|z|wrapped_update_w(&z),|z|precond(&z),qp_ctrl.damped_bse_tol,qp_ctrl.damped_bse_max_iter);
    let density_real:Vec<f64>=(0..occ_size*vir_size).map(|ia|result.0[ia]-result.1[ia]).collect();
    let polarized_density_matrix=MatrixFull::from_vec([occ_size,vir_size],density_real).unwrap();
    export_density(scf_data,&polarized_density_matrix,&qp_ctrl);
    result
}
fn export_density(scf_data:&SCF,polarized_density_matrix:&MatrixFull<f64>,qp_ctrl:&QuasiParticle){
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let occ_eigenvecs=MatrixFull::from_vec([num_state,occ_size],scf_data.eigenvectors[0].data[0..occ_size*num_state].to_vec().clone()).unwrap();
    let vir_eigenvecs=MatrixFull::from_vec([num_state,vir_size],scf_data.eigenvectors[0].data[occ_size*num_state..].to_vec().clone()).unwrap();
    let mut first_prod=MatrixFull::new([num_state,vir_size],0.0);
    let mut ao_polarized_density_matrix=MatrixFull::new([num_state,num_state],0.0);
    _dgemm_full(&occ_eigenvecs,'N',polarized_density_matrix,'N',&mut first_prod,1.0,0.0);
    _dgemm_full(&first_prod,'N',&vir_eigenvecs,'T',&mut ao_polarized_density_matrix,1.0,0.0);
    let mut file = OpenOptions::new().write(true).create(true).truncate(true).open("AO_Polarized_Density_Matrix.txt").expect("open failure");
    writeln!(file, "AO Basis Polarized Density Matrix:\nSize:{:?}\nData(Column Major){:#?}",ao_polarized_density_matrix.size,ao_polarized_density_matrix.data).expect("write failure");
    let grids = &qp_ctrl.damped_bse_grids;
    let grid_val=eval_ao_on_grids(&scf_data.mol,&grids);
    //grid_val.formated_output(1000,"full");
    let mut first_prod=MatrixFull::new([num_state,grids.len()],0.0);
    _dgemm_full(&ao_polarized_density_matrix,'N',&grid_val,'N',&mut first_prod,1.0,0.0);
    let grid_data:Vec<f64>=first_prod.iter_columns_full().zip(grid_val.iter_columns_full()).map(|(fp,gval)|{
        let fp_vec=fp.to_vec();
        let gval_vec=gval.to_vec();
        fp_vec.iter().zip(gval_vec.iter()).fold(0.0,|acc,(x,y)|acc+x*y)
    }).collect();

    // Write to file with clear formatting (replace mode)
    let mut file = OpenOptions::new().write(true).create(true).truncate(true).open("Polarized_Density_Grids.txt").expect("open failure");

    // Header
    writeln!(file, "{}", "=".repeat(70)).expect("write failure");
    writeln!(file, "POLARIZED DENSITY ON SPATIAL GRIDS (damped BSE result)").expect("write failure");
    writeln!(file, "{}", "=".repeat(70)).expect("write failure");

    // Molecular geometry information
    writeln!(file, "\n{}", "-".repeat(70)).expect("write failure");
    writeln!(file, "MOLECULAR GEOMETRY").expect("write failure");
    writeln!(file, "{}", "-".repeat(70)).expect("write failure");
    writeln!(file, "{}", "=".repeat(70)).expect("write failure");
    writeln!(file, "{}", "=".repeat(70)).expect("write failure");

    let elem = &scf_data.mol.geom.elem;
    let position = scf_data.mol.geom.position.clone();
    for (i, e) in elem.iter().enumerate() {
        let pos = position.iter_columns_full().nth(i).unwrap();
        writeln!(file, "{:>2}  {:.8e}  {:.8e}  {:.8e}", e, pos[0], pos[1], pos[2]).expect("write failure");
    }

    // Grid sampling information
    writeln!(file, "\n{}", "-".repeat(70)).expect("write failure");
    writeln!(file, "GRID SAMPLING INFORMATION").expect("write failure");
    writeln!(file, "{}", "-".repeat(70)).expect("write failure");
    writeln!(file, "{}", "=".repeat(70)).expect("write failure");
    writeln!(file, "X start: {:.8e}  X end: {:.8e}  X points: {:>6}", qp_ctrl.damped_bse_x_start, qp_ctrl.damped_bse_x_end, qp_ctrl.damped_bse_x_points).expect("write failure");
    writeln!(file, "Y start: {:.8e}  Y end: {:.8e}  Y points: {:>6}", qp_ctrl.damped_bse_y_start, qp_ctrl.damped_bse_y_end, qp_ctrl.damped_bse_y_points).expect("write failure");
    writeln!(file, "Z start: {:.8e}  Z end: {:.8e}  Z points: {:>6}", qp_ctrl.damped_bse_z_start, qp_ctrl.damped_bse_z_end, qp_ctrl.damped_bse_z_points).expect("write failure");
    let x_step = if qp_ctrl.damped_bse_x_points > 1 { (qp_ctrl.damped_bse_x_end - qp_ctrl.damped_bse_x_start) / (qp_ctrl.damped_bse_x_points - 1) as f64 } else { 0.0 };
    let y_step = if qp_ctrl.damped_bse_y_points > 1 { (qp_ctrl.damped_bse_y_end - qp_ctrl.damped_bse_y_start) / (qp_ctrl.damped_bse_y_points - 1) as f64 } else { 0.0 };
    let z_step = if qp_ctrl.damped_bse_z_points > 1 { (qp_ctrl.damped_bse_z_end - qp_ctrl.damped_bse_z_start) / (qp_ctrl.damped_bse_z_points - 1) as f64 } else { 0.0 };
    writeln!(file, "X step: {:.8e}  Y step: {:.8e}  Z step: {:.8e}", x_step, y_step, z_step).expect("write failure");
    writeln!(file, "Total grid points: {}", grids.len()).expect("write failure");

    // Grid data
    writeln!(file, "\n{}", "-".repeat(70)).expect("write failure");
    writeln!(file, "GRID DENSITY DATA").expect("write failure");
    writeln!(file, "{}", "-".repeat(70)).expect("write failure");
    writeln!(file, "(OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z)").expect("write failure");
    writeln!(file, "{}", "=".repeat(70)).expect("write failure");
    writeln!(file, "{:>10}  {:>20}  {:>20}  {:>20}  {:>20}", "Index", "X", "Y", "Z", "Density").expect("write failure");
    writeln!(file, "{}", "-".repeat(70)).expect("write failure");
    for (i, g) in grids.iter().enumerate() {
        let x: f64 = g[0];
        let y: f64 = g[1];
        let z: f64 = g[2];
        writeln!(file, "{:>10}  {:.8e}  {:.8e}  {:.8e}  {:.12e}", i, x, y, z, grid_data[i]).expect("write failure");
    }

    writeln!(file, "\n{}", "=".repeat(70)).expect("write failure");
    writeln!(file, "END OF FILE").expect("write failure");
    writeln!(file, "{}", "=".repeat(70)).expect("write failure");
}
pub fn damped_bse_dense(scf_data:&SCF)->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>){
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let inverse_dielectric=ri_bse::construct_inverse_dielectric(scf_data,&scf_data.eigenvalues[0]);
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    println!("Now begins damped BSE calculation. Parameters:\nOcc Size={},Vir Size={},External Field Freq={},Lifetime Gamma={}",occ_size,vir_size,qp_ctrl.external_field_freq,qp_ctrl.lifetime_gamma);
    let mu_z_vec=compute_mu_z_vec(scf_data);
    let a=ri_bse::construct_submat_a(scf_data,&inverse_dielectric,&scf_data.gwqp.0, 'R');
    let b=ri_bse::construct_submat_b(scf_data,'R',&inverse_dielectric);
    let mut linear_equation_lhs=MatrixFull::new([occ_size*vir_size*4,occ_size*vir_size*4],0.0);
    let mut linear_equation_rhs=vec![0.0;4*occ_size*vir_size];
    let start=Instant::now();
    for i in 0..occ_size*vir_size {
        for j in 0..vir_size*occ_size{
            linear_equation_lhs[[i,j]]=a[[i,j]];
            linear_equation_lhs[[occ_size*vir_size+i,j]]=b[[i,j]];
            linear_equation_lhs[[i,occ_size*vir_size+j]]=b[[i,j]];
            linear_equation_lhs[[occ_size*vir_size+i,occ_size*vir_size+j]]=a[[i,j]];

            linear_equation_lhs[[i+occ_size*vir_size*2,j+occ_size*vir_size*2]]=a[[i,j]];
            linear_equation_lhs[[occ_size*vir_size*3+i,j+occ_size*vir_size*2]]=b[[i,j]];
            linear_equation_lhs[[i+occ_size*vir_size*2,occ_size*vir_size*3+j]]=b[[i,j]];
            linear_equation_lhs[[occ_size*vir_size*3+i,occ_size*vir_size*3+j]]=a[[i,j]];
        }
    }
    let external_field_freq=qp_ctrl.external_field_freq;
    let lifetime_gamma=qp_ctrl.lifetime_gamma;
    for i in 0..occ_size{
        for a in 0..vir_size{
            linear_equation_lhs[[i+a*occ_size,i+a*occ_size]]-=external_field_freq;
            linear_equation_lhs[[i+a*occ_size+occ_size*vir_size,i+a*occ_size+occ_size*vir_size]]+=external_field_freq;
            linear_equation_lhs[[i+a*occ_size+2*occ_size*vir_size,i+a*occ_size+2*occ_size*vir_size]]-=external_field_freq;
            linear_equation_lhs[[i+a*occ_size+3*occ_size*vir_size,i+a*occ_size+3*occ_size*vir_size]]+=external_field_freq;

            linear_equation_lhs[[i+a*occ_size+2*occ_size*vir_size,i+a*occ_size]]+=lifetime_gamma;
            linear_equation_lhs[[i+a*occ_size+3*occ_size*vir_size,i+a*occ_size+occ_size*vir_size]]-=lifetime_gamma;
            linear_equation_lhs[[i+a*occ_size,i+a*occ_size+2*occ_size*vir_size]]-=lifetime_gamma;
            linear_equation_lhs[[i+a*occ_size+occ_size*vir_size,i+a*occ_size+3*occ_size*vir_size]]+=lifetime_gamma;

            linear_equation_rhs[i+a*occ_size]=mu_z_vec[i+a*occ_size];
            linear_equation_rhs[i+a*occ_size+occ_size*vir_size]=mu_z_vec[i+a*occ_size];
        }
    }
    let preptime=start.elapsed();
    println!("Preparation of dense linear problem took {:?}",preptime);
    let result=_dsolve(&linear_equation_lhs,&linear_equation_rhs).unwrap();
    println!("Solving dense linear problem took {:?}",start.elapsed()-preptime);
    let solution=(result[0..occ_size*vir_size].to_vec(),result[occ_size*vir_size..2*occ_size*vir_size].to_vec(),result[2*occ_size*vir_size..3*occ_size*vir_size].to_vec(),result[3*occ_size*vir_size..4*occ_size*vir_size].to_vec());
    let density_real:Vec<f64>=(0..occ_size*vir_size).map(|ia|solution.0[ia]+solution.1[ia]).collect();
    let polarized_density_matrix=MatrixFull::from_vec([occ_size,vir_size],density_real).unwrap();
    export_density(scf_data,&polarized_density_matrix,&qp_ctrl);
    let mut file = OpenOptions::new().write(true).create(true).truncate(true).open("Polarized_Density_Matrix.txt").expect("open failure");
    writeln!(file, "Polarized Density Matrix(X):\n{:#?}\n(Y):{:#?}", solution.0,solution.1).expect("write failure");
    solution
}
pub fn damped_bse_gmres(scf_data:&SCF)->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>){
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let inverse_dielectric=ri_bse::construct_inverse_dielectric(scf_data,&scf_data.eigenvalues[0]);
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    println!("Now begins damped BSE calculation. Parameters:\nOcc Size={},Vir Size={},External Field Freq={},Lifetime Gamma={}",occ_size,vir_size,qp_ctrl.external_field_freq,qp_ctrl.lifetime_gamma);
    let mu_z_vec=compute_mu_z_vec(scf_data);
    let ri_oo=ri_bse::get_submatrix(scf_data,'O','O','N');
    println!("num_auxbas={}",ri_oo.size[0]);
    let num_auxbas=ri_oo.size[0];
    let mut ri_oo_tilde:MatrixFull<f64>=MatrixFull::new(ri_oo.size,0.0);
    _dgemm_full(&inverse_dielectric,'N',&ri_oo,'N',&mut ri_oo_tilde,1.0,0.0);
    drop(ri_oo);
    ri_oo_tilde.reshape([num_auxbas*occ_size,occ_size]);
    ri_oo_tilde=ri_oo_tilde.transpose_and_drop();
    ri_oo_tilde.reshape([occ_size*num_auxbas,occ_size]);
    let ri_ov=ri_bse::get_submatrix(scf_data,'O','V','N');
    let mut ri_ov_w=ri_ov.clone();
    ri_ov_w.reshape([num_auxbas*occ_size,vir_size]);
    let mut ri_vv=ri_bse::get_submatrix(scf_data,'V','V','N');
    ri_vv.reshape([num_auxbas*vir_size,vir_size]);
    let mut ri_ov_tilde:MatrixFull<f64>=MatrixFull::new(ri_ov.size,0.0);
    _dgemm_full(&inverse_dielectric,'N',&ri_ov,'N',&mut ri_ov_tilde,1.0,0.0);
    drop(inverse_dielectric);
    ri_ov_tilde.reshape([num_auxbas*occ_size,vir_size]);
    let external_field_freq=qp_ctrl.external_field_freq;
    let lifetime_gamma=qp_ctrl.lifetime_gamma;
    let casida_a_matvec=|z:&Vec<f64>|{
        ri_bse::matvec::a_block_matvec(scf_data,&qp_ctrl,&ri_vv,&ri_ov,&ri_oo_tilde,z)
    };
    let casida_b_matvec=|z:&Vec<f64>|{
        ri_bse::matvec::b_block_matvec(scf_data,&qp_ctrl,&ri_ov,&ri_ov_w,&ri_ov_tilde,z)
    };
    let gmres_matvec=|z:&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)|{
        let mut vec1=casida_a_matvec(&z.0);
        vec1=vector_scaled_add(&vec1,1.0,&z.0,-external_field_freq);
        vec1=vector_scaled_add(&vec1,1.0,&casida_b_matvec(&z.1),1.0);
        vec1=vector_scaled_add(&vec1,1.0,&z.2,-lifetime_gamma);
        let mut vec2=casida_b_matvec(&z.0);
        vec2=vector_scaled_add(&vec2,1.0,&casida_a_matvec(&z.1),1.0);
        vec2=vector_scaled_add(&vec2,1.0,&z.1,external_field_freq);
        vec2=vector_scaled_add(&vec2,1.0,&z.3,lifetime_gamma);
        let mut vec3=casida_a_matvec(&z.2);
        vec3=vector_scaled_add(&vec3,1.0,&z.0,lifetime_gamma);
        vec3=vector_scaled_add(&vec3,1.0,&z.2,-external_field_freq);
        vec3=vector_scaled_add(&vec3,1.0,&casida_b_matvec(&z.3),1.0);
        let mut vec4=casida_b_matvec(&z.2);
        vec4=vector_scaled_add(&vec4,1.0,&z.1,-lifetime_gamma);
        vec4=vector_scaled_add(&vec4,1.0,&casida_a_matvec(&z.3),1.0);
        vec4=vector_scaled_add(&vec4,1.0,&z.3,external_field_freq);
        (vec1,vec2,vec3,vec4)
    };
    let energy_diag=ri_bse::construct_energy_diag_for_a(&scf_data.gwqp.0,occ_size,vir_size);
    let diag_13:Vec<f64>=energy_diag.iter().map(|d|d-external_field_freq).collect();
    let diag_24:Vec<f64>=energy_diag.iter().map(|d|d+external_field_freq).collect();
    let precond=|z:&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)|->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>){
        let apply_inv = |v:&Vec<f64>,d:&Vec<f64>|->Vec<f64>{
            v.iter().zip(d.iter()).map(|(vi,di)|{
                if di.abs()>1e-8 {vi/di} else {0.0}
            }).collect()
        };
        (apply_inv(&z.0,&diag_13),apply_inv(&z.1,&diag_24),
         apply_inv(&z.2,&diag_13),apply_inv(&z.3,&diag_24))
    };
    let precond_trivial=|z:&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)|->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>){
        z.clone()
    };
    let mu_z_vec=compute_mu_z_vec(scf_data);
    let rhs_p=(mu_z_vec.clone(),mu_z_vec.clone(),vec![0.0;occ_size*vir_size],vec![0.0;occ_size*vir_size]);
    let solution=fourvec_gmres(&gmres_matvec,&precond,&rhs_p,qp_ctrl.damped_bse_tol,qp_ctrl.damped_bse_max_iter);
    let density_real:Vec<f64>=(0..occ_size*vir_size).map(|ia|solution.0[ia]+solution.1[ia]).collect();
    let polarized_density_matrix=MatrixFull::from_vec([occ_size,vir_size],density_real).unwrap();
    export_density(scf_data,&polarized_density_matrix,&qp_ctrl);
    let mut file = OpenOptions::new().write(true).create(true).truncate(true).open("Polarized_Density_Matrix.txt").expect("open failure");
    writeln!(file, "Polarized Density Matrix(X):\n{:#?}\n(Y):{:#?}", solution.0,solution.1).expect("write failure");
    solution
}

/// Klopper subspace solver: implements the iterative subspace method from
/// Section 2.2 of Kehry et al., Mol. Phys. 118, e1755064 (2020).
///
/// Solves the 4-component non-Hermitian linear system:
///   A_4c * z = b_4c     where A_4c = gmres_matvec, b_4c = rhs
///
/// Algorithm:
/// 1. Start with initial guess z_0 = p0 (non-interacting response, Eqns 14-15)
/// 2. Normalize to get v_0, compute w_0 = A * v_0
/// 3. At each iteration, project the system onto the subspace:
///    Q[i][k] = v_i^T * w_k = v_i^T * A * v_k,   q[i] = v_i^T * b
/// 4. Solve Q * t = q, form full-space solution: z = \Sigma t_k * v_k
/// 5. Compute residual: r = A*z - b = \Sigma t_k * w_k - b
/// 6. Generate new direction via Jacobi-Davidson correction (Eqns 21-23):
///    u = e * M^{-1}*z - M^{-1}*r,  e = (z^T M^{-1} r) / (z^T M^{-1} z)
/// 7. Orthogonalize and normalize to get v_{k+1}, compute w_{k+1}
/// 8. Repeat until convergence
fn klopper_subspace_solver(
    matvec: impl Fn(&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)) -> (Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>),
    precond: impl Fn(&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)) -> (Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>),
    rhs: &(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>),
    p0: &(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>),
    tol: f64,
    max_subspace: usize,
) -> (Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>) {
    const EPS: f64 = 1e-14;

    let n = rhs.0.len();

    // Normalize initial guess to get v[0]
    let p0_norm = fourvec_dot_product(p0, p0).sqrt();
    if p0_norm < EPS {
        eprintln!("Warning: initial guess norm is too small ({})", p0_norm);
        return (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    }
    let inv_norm = 1.0 / p0_norm;
    let mut v_vecs: Vec<(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)> = vec![(
        num_product(&p0.0, inv_norm),
        num_product(&p0.1, inv_norm),
        num_product(&p0.2, inv_norm),
        num_product(&p0.3, inv_norm),
    )];

    // Compute w[0] = A * v[0]
    let w0 = matvec(&v_vecs[0]);
    let mut w_vecs: Vec<(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)> = vec![w0];

    for iter in 0..max_subspace {
        let m = iter + 1; // current subspace size

        // Build subspace matrix: Q[i][k] = v[i]^T * w[k] = v[i]^T * A * v[k]
        // Build projected RHS: q[i] = v[i]^T * rhs
        let mut q_mat = MatrixFull::new([m, m], 0.0);
        let mut q_vec = vec![0.0; m];
        for i in 0..m {
            q_vec[i] = fourvec_dot_product(&v_vecs[i], rhs);
            for k in 0..m {
                q_mat[[i, k]] = fourvec_dot_product(&v_vecs[i], &w_vecs[k]);
            }
        }

        // Solve Q * t = q (dense subspace solve)
        let t = match _dsolve(&q_mat, &q_vec) {
            Some(t) => t,
            None => {
                eprintln!("Klopper solver: subspace solve failed at iteration {}", iter);
                break;
            }
        };

        // Form full-space solution: z = \Sigma t_k * v[k]
        let mut z = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for k in 0..m {
            z = fourvec_scaled_add(&z, 1.0, &v_vecs[k], t[k]);
        }

        // Compute residual: r = A*z - rhs = \Sigma t_k * w[k] - rhs
        let mut r = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for k in 0..m {
            r = fourvec_scaled_add(&r, 1.0, &w_vecs[k], t[k]);
        }
        r = fourvec_scaled_add(&r, 1.0, rhs, -1.0);

        let res_norm = fourvec_dot_product(&r, &r).sqrt();
        println!("Klopper Solver: Iteration {}, Subspace Size {}, Residual = {:.2e}, Threshold = {:.2e}",
                 iter, m, res_norm, tol);

        if res_norm < tol {
            println!("Klopper Solver converged at iteration {} with residual {:.2e}", iter, res_norm);
            return z;
        }

        // Generate new trial vector using Jacobi-Davidson correction (paper Eqns 21-23)
        // u = e * M^{-1}*z - M^{-1}*r
        // e = (z^T * M^{-1} * r) / (z^T * M^{-1} * z)
        let m_inv_z = precond(&z);
        let m_inv_r = precond(&r);

        let zt_m_inv_r = fourvec_dot_product(&z, &m_inv_r);
        let zt_m_inv_z = fourvec_dot_product(&z, &m_inv_z);

        let epsilon = if zt_m_inv_z.abs() > EPS {
            zt_m_inv_r / zt_m_inv_z
        } else {
            0.0
        };

        // u = e * M^{-1}*z - M^{-1}*r
        let mut u = fourvec_scaled_add(&m_inv_z, epsilon, &m_inv_r, -1.0);

        // Modified Gram-Schmidt (MGS) with iterated reorthogonalization
        // for numerical stability. Each projection uses the progressively
        // updated u, and reorthogonalization removes residual components.
        for _reortho in 0..2 {
            for existing_v in &v_vecs {
                let v_norm2 = fourvec_dot_product(existing_v, existing_v);
                if v_norm2 < EPS { continue; }
                let dot_val = fourvec_dot_product(&u, existing_v);
                u = fourvec_scaled_add(&u, 1.0, existing_v, -dot_val / v_norm2);
            }
        }

        // Normalize u to get v[m] = u_{new}
        let u_norm = fourvec_dot_product(&u, &u).sqrt();
        if u_norm < EPS {
            println!("Klopper Solver: new direction norm too small ({}), stopping", u_norm);
            return z;
        }
        let inv_u = 1.0 / u_norm;
        let v_new = (
            num_product(&u.0, inv_u),
            num_product(&u.1, inv_u),
            num_product(&u.2, inv_u),
            num_product(&u.3, inv_u),
        );

        // Compute w_new = A * v_new
        let w_new = matvec(&v_new);

        // Expand subspace
        v_vecs.push(v_new);
        w_vecs.push(w_new);
    }

    eprintln!("Warning: Klopper Solver reached maximum subspace size ({}) without convergence", max_subspace);

    // Return best approximation from final subspace
    let m = v_vecs.len();
    let mut q_mat = MatrixFull::new([m, m], 0.0);
    let mut q_vec = vec![0.0; m];
    for i in 0..m {
        q_vec[i] = fourvec_dot_product(&v_vecs[i], rhs);
        for k in 0..m {
            q_mat[[i, k]] = fourvec_dot_product(&v_vecs[i], &w_vecs[k]);
        }
    }
    if let Some(t) = _dsolve(&q_mat, &q_vec) {
        let mut z = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for k in 0..m {
            z = fourvec_scaled_add(&z, 1.0, &v_vecs[k], t[k]);
        }
        return z;
    }
    (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n])
}

pub fn damped_bse_klopper(scf_data: &SCF) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (start_mo, num_state, occ_size, vir_size, homo, lumo) = get_occupation_parameters(scf_data, 'N');
    let inverse_dielectric = ri_bse::construct_inverse_dielectric(scf_data, &scf_data.eigenvalues[0]);
    let qp_ctrl = scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    println!("Now begins damped BSE calculation (Klopper Subspace Solver). Parameters:\nOcc Size={}, Vir Size={}, External Field Freq={}, Lifetime Gamma={}",
             occ_size, vir_size, qp_ctrl.external_field_freq, qp_ctrl.lifetime_gamma);
    let mu_z_vec = compute_mu_z_vec(scf_data);
    let ri_oo = ri_bse::get_submatrix(scf_data, 'O', 'O', 'N');
    println!("num_auxbas={}", ri_oo.size[0]);
    let num_auxbas = ri_oo.size[0];
    let mut ri_oo_tilde: MatrixFull<f64> = MatrixFull::new(ri_oo.size, 0.0);
    _dgemm_full(&inverse_dielectric, 'N', &ri_oo, 'N', &mut ri_oo_tilde, 1.0, 0.0);
    drop(ri_oo);
    ri_oo_tilde.reshape([num_auxbas * occ_size, occ_size]);
    ri_oo_tilde = ri_oo_tilde.transpose_and_drop();
    ri_oo_tilde.reshape([occ_size * num_auxbas, occ_size]);
    let ri_ov = ri_bse::get_submatrix(scf_data, 'O', 'V', 'N');
    let mut ri_ov_w = ri_ov.clone();
    ri_ov_w.reshape([num_auxbas * occ_size, vir_size]);
    let mut ri_vv = ri_bse::get_submatrix(scf_data, 'V', 'V', 'N');
    ri_vv.reshape([num_auxbas * vir_size, vir_size]);
    let mut ri_ov_tilde: MatrixFull<f64> = MatrixFull::new(ri_ov.size, 0.0);
    _dgemm_full(&inverse_dielectric, 'N', &ri_ov, 'N', &mut ri_ov_tilde, 1.0, 0.0);
    drop(inverse_dielectric);
    ri_ov_tilde.reshape([num_auxbas * occ_size, vir_size]);

    let external_field_freq = qp_ctrl.external_field_freq;
    let lifetime_gamma = qp_ctrl.lifetime_gamma;

    // Matrix-vector product: same 4-component operator as GMRES
    let casida_a_matvec = |z: &Vec<f64>| {
        ri_bse::matvec::a_block_matvec(scf_data, &qp_ctrl, &ri_vv, &ri_ov, &ri_oo_tilde, z)
    };
    let casida_b_matvec = |z: &Vec<f64>| {
        ri_bse::matvec::b_block_matvec(scf_data, &qp_ctrl, &ri_ov, &ri_ov_w, &ri_ov_tilde, z)
    };
    let gmres_matvec = |z: &(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)| {
        let mut vec1 = casida_a_matvec(&z.0);
        vec1 = vector_scaled_add(&vec1, 1.0, &z.0, -external_field_freq);
        vec1 = vector_scaled_add(&vec1, 1.0, &casida_b_matvec(&z.1), 1.0);
        vec1 = vector_scaled_add(&vec1, 1.0, &z.2, -lifetime_gamma);
        let mut vec2 = casida_b_matvec(&z.0);
        vec2 = vector_scaled_add(&vec2, 1.0, &casida_a_matvec(&z.1), 1.0);
        vec2 = vector_scaled_add(&vec2, 1.0, &z.1, external_field_freq);
        vec2 = vector_scaled_add(&vec2, 1.0, &z.3, lifetime_gamma);
        let mut vec3 = casida_a_matvec(&z.2);
        vec3 = vector_scaled_add(&vec3, 1.0, &z.0, lifetime_gamma);
        vec3 = vector_scaled_add(&vec3, 1.0, &z.2, -external_field_freq);
        vec3 = vector_scaled_add(&vec3, 1.0, &casida_b_matvec(&z.3), 1.0);
        let mut vec4 = casida_b_matvec(&z.2);
        vec4 = vector_scaled_add(&vec4, 1.0, &z.1, -lifetime_gamma);
        vec4 = vector_scaled_add(&vec4, 1.0, &casida_a_matvec(&z.3), 1.0);
        vec4 = vector_scaled_add(&vec4, 1.0, &z.3, external_field_freq);
        (vec1, vec2, vec3, vec4)
    };

    // Diagonal preconditioner: M = diag(A_4c)
    let energy_diag = ri_bse::construct_energy_diag_for_a(&scf_data.gwqp.0, occ_size, vir_size);
    let diag_13: Vec<f64> = energy_diag.iter().map(|d| d - external_field_freq).collect();
    let diag_24: Vec<f64> = energy_diag.iter().map(|d| d + external_field_freq).collect();
    let precond = |z: &(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)| -> (Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>) {
        let apply_inv = |v: &Vec<f64>, d: &Vec<f64>| -> Vec<f64> {
            v.iter().zip(d.iter()).map(|(vi, di)| {
                if di.abs() > 1e-8 { vi / di } else { 0.0 }
            }).collect()
        };
        (apply_inv(&z.0, &diag_13), apply_inv(&z.1, &diag_24),
         apply_inv(&z.2, &diag_13), apply_inv(&z.3, &diag_24))
    };

    // RHS: dipole integrals in 4-component form
    let rhs = (mu_z_vec.clone(), mu_z_vec.clone(),
               vec![0.0; occ_size * vir_size], vec![0.0; occ_size * vir_size]);

    // Initial guess: non-interacting response (paper Eqns 14-15)
    let p0 = prepare_p0_r_i(&mu_z_vec, &scf_data.gwqp.0,
                             external_field_freq, lifetime_gamma,
                             occ_size, vir_size);

    // Run the subspace solver
    let solution = klopper_subspace_solver(&gmres_matvec, &precond, &rhs, &p0,
                                           qp_ctrl.damped_bse_tol, qp_ctrl.damped_bse_max_iter);

    let density_real: Vec<f64> = (0..occ_size * vir_size).map(|ia| solution.0[ia] + solution.1[ia]).collect();
    let polarized_density_matrix = MatrixFull::from_vec([occ_size, vir_size], density_real).unwrap();
    export_density(scf_data, &polarized_density_matrix, &qp_ctrl);
    solution
}

fn fourvec_gmres(
    matvec:impl Fn(&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>))->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>),
    precond:impl Fn(&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>))->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>),
    rhs:&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>),
    tol: f64,
    max_total_iter: usize)
    ->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>){
    const KRYLOV_DIM: usize = 30;
    let max_restarts = if max_total_iter > KRYLOV_DIM { max_total_iter / KRYLOV_DIM } else { 1 };
    const EPS: f64 = 1e-14;

    let n = rhs.0.len();
    // initial guess: x = 0
    let mut x = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut iter_count=0;
    for _restart in 0..max_restarts {
        // r = b - A*x
        let ax = matvec(&x);
        let r = fourvec_scaled_add(rhs, 1.0, &ax, -1.0);
        let beta = fourvec_dot_product(&r, &r).sqrt();

        if beta < tol {
            break;
        }

        // v[0] = r / beta
        let inv_beta = 1.0 / beta;
        let mut v: Vec<(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)> = vec![(
            num_product(&r.0, inv_beta),
            num_product(&r.1, inv_beta),
            num_product(&r.2, inv_beta),
            num_product(&r.3, inv_beta),
        )];
        // z[i] = M^{-1} * v[i] for solution reconstruction
        let mut z: Vec<(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)> = Vec::new();

        // Hessenberg matrix: (KRYLOV_DIM+1) rows x KRYLOV_DIM cols, stored as h[row][col]
        let mut h = vec![vec![0.0; KRYLOV_DIM]; KRYLOV_DIM + 1];
        let mut c = vec![0.0; KRYLOV_DIM];
        let mut s = vec![0.0; KRYLOV_DIM];
        let mut g = vec![0.0; KRYLOV_DIM + 1];
        g[0] = beta;

        let mut actual_dim = KRYLOV_DIM;
        let mut converged = false;

        for j in 0..KRYLOV_DIM {
            // Right-preconditioned GMRES:
            //   z_j = M^{-1} * v_j   (apply preconditioner)
            //   w_j = A * z_j         (matrix-vector product on preconditioned vector)
            let zj = precond(&v[j]);
            let w = matvec(&zj);
            z.push(zj);

            // Arnoldi: orthogonalize w against v[0..j]
            let mut w_orth = w.clone();
            for i in 0..=j {
                h[i][j] = fourvec_dot_product(&w_orth, &v[i]);
                w_orth = fourvec_scaled_add(&w_orth, 1.0, &v[i], -h[i][j]);
            }

            h[j+1][j] = fourvec_dot_product(&w_orth, &w_orth).sqrt();

            if h[j+1][j] < EPS {
                // happy breakdown
                actual_dim = j + 1;
                converged = true;
                break;
            }

            // v_{j+1} = w_orth / h_{j+1,j}
            let inv_h = 1.0 / h[j+1][j];
            v.push((
                num_product(&w_orth.0, inv_h),
                num_product(&w_orth.1, inv_h),
                num_product(&w_orth.2, inv_h),
                num_product(&w_orth.3, inv_h),
            ));

            // Apply previous Givens rotations to column j of H
            for i in 0..j {
                let h_ij = h[i][j];
                let h_i1j = h[i+1][j];
                h[i][j]   =  c[i] * h_ij + s[i] * h_i1j;
                h[i+1][j] = -s[i] * h_ij + c[i] * h_i1j;
            }

            // Compute new Givens rotation to eliminate h[j+1][j]
            let r = (h[j][j].powi(2) + h[j+1][j].powi(2)).sqrt();
            c[j] = h[j][j] / r;
            s[j] = h[j+1][j] / r;
            h[j][j] = r;
            h[j+1][j] = 0.0;

            // Apply rotation to g
            let g_j = g[j];
            g[j]   =  c[j] * g_j + s[j] * g[j+1];  // g[j+1] is 0 here
            g[j+1] = -s[j] * g_j + c[j] * g[j+1];  // = -s[j] * g_j

            let residual = g[j+1].abs();
            iter_count+=1;
            println!("GMRES Iteration {}, Residue={}, Converging Threshold is {}",iter_count,residual,tol);
            if residual < tol {
                converged = true;
                actual_dim = j + 1;
                break;
            }
        }

        // Back-substitution: solve R * y = g[0..m]
        let m = actual_dim;
        let mut y = vec![0.0; m];
        for i in (0..m).rev() {
            y[i] = g[i];
            for k in i+1..m {
                y[i] -= h[i][k] * y[k];
            }
            y[i] /= h[i][i];
        }

        // Update solution: x = x + sum_{i=0}^{m-1} y[i] * z[i]
        // (z[i] = M^{-1} * v[i] for right-preconditioned GMRES)
        for i in 0..m {
            x = fourvec_scaled_add(&x, 1.0, &z[i], y[i]);
        }

        if converged {
            break;
        }
    }

    x
}