use crate::scf_io::{SCF, SCFType};
use tensors::{MathMatrix, MatrixFull, RIFull,MatrixFullSlice};
use crate::ri_gw::get_occupation_parameters;
use itertools::Itertools;
use rest_tensors::matrix::matrix_blas_lapack::{_dgeev, _dgemv,_dgemm_full};
use crate::ri_bse;
use crate::ri_bse::{davidson_solver,sbse};
use crate::ctrl_io::quasiparticle_methods::QuasiParticle;
use rayon::prelude::*;
use std::time::Instant;
use std::sync::atomic::{AtomicPtr, Ordering};

pub fn diagonal_elements_contribution(scf_data:&SCF,vec:&Vec<f64>)->Vec<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let qp=scf_data.gwqp.0.clone();
    let mut prod_vec=vec![0.0;occ_size*vir_size];
    (0..occ_size).cartesian_product(0..vir_size).for_each(|(i,a)|{prod_vec[i+a*occ_size]=(qp[a+occ_size]-qp[i])*vec[i+a*occ_size]});
    prod_vec
}
pub fn coulomb_contribution(ri_matrix:&MatrixFull<f64>,vec:&Vec<f64>)->Vec<f64>{
    let mut inter_result=vec![0.0;ri_matrix.size[0]];
    _dgemv(ri_matrix,vec , &mut inter_result, 'N', 1.0, 0.0, 1, 1);
    let mut result=vec![0.0;ri_matrix.size[1]];
    _dgemv(ri_matrix,&inter_result , &mut result, 'T', 1.0, 0.0, 1, 1);
    result
}
pub fn w_contribution(scf_data:&SCF,z_vec:&Vec<f64>,ri_oo_tilde:&MatrixFull<f64>)->Vec<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let num_auxbas=ri_oo_tilde.size[0];
    let ri_vv=ri_bse::get_submatrix(scf_data,'V','V','N');
    let mut result=vec![0.0;occ_size*vir_size];
    let mut t_tensor:MatrixFull<f64>=MatrixFull::new([num_auxbas,occ_size*vir_size],0.0);
    (0..occ_size).cartesian_product(0..vir_size).cartesian_product(0..num_auxbas).for_each(|((j,a),q)|{
        t_tensor[[q,j+a*occ_size]]=(0..vir_size).fold(0.0,|acc,b|acc+(z_vec[j+b*occ_size]*ri_vv[[q,a*vir_size+b]]))
    });
    (0..occ_size).cartesian_product(0..vir_size).for_each(|(i,a)|{
        result[i+a*occ_size]=(0..occ_size).cartesian_product(0..num_auxbas).fold(0.0,|acc,(j,q)|acc+(ri_oo_tilde[[q,i+j*occ_size]]*t_tensor[[q,j+a*occ_size]]))
    });
    result
}
pub fn w_contribution_rayon_a_block(scf_data:&SCF,ri_vv:&MatrixFull<f64>,z_vec:&Vec<f64>,ri_oo_tilde:&MatrixFull<f64>)->Vec<f64>{
    let start=Instant::now();
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let duration1=start.elapsed();
    println!("W第一个操作耗时: {:?}", duration1);
    let num_auxbas=ri_oo_tilde.size[0];
    // 使用并行化计算 t_tensor - 使用 AtomicPtr 解决所有权问题
    let mut t_tensor_data = vec![0.0; num_auxbas * occ_size * vir_size];
    let t_tensor_ptr = AtomicPtr::new(t_tensor_data.as_mut_ptr());
    //println!("occ_size={},vir_size={},ri_vv size={},{}",occ_size,vir_size,ri_vv.size[0],ri_vv.size[1]);
    // 并行化外层循环：对 j 进行并行化
    (0..occ_size).into_par_iter().for_each(|j| {
        let z_vec_ref = &z_vec;
        let ptr = t_tensor_ptr.load(Ordering::Relaxed);
        
        for a in 0..vir_size {
            for q in 0..num_auxbas {
                let mut sum = 0.0;
                for b in 0..vir_size {
                    sum += z_vec_ref[j + b * occ_size] * ri_vv[[q, a * vir_size + b]];
                }
                // 使用与串行版本相同的索引方式：q + (j + a * occ_size) * num_auxbas
                let index = q + (j + a * occ_size) * num_auxbas;
                unsafe {
                    *ptr.add(index) = sum;
                }
            }
        }
    });
    let duration2=start.elapsed();
    println!("W第二个操作耗时: {:?}", duration2-duration1);
    let t_tensor = MatrixFull::from_vec([num_auxbas, occ_size * vir_size], t_tensor_data).unwrap();
    // 并行化第二部分：计算最终结果
    let result: Vec<f64> = (0..vir_size).into_par_iter().flat_map(|a| {
        let ri_oo_tilde_ref = &ri_oo_tilde;
        let t_tensor_ref = &t_tensor;
        (0..occ_size).into_par_iter().map(move |i| {
            (0..occ_size).fold(0.0, |acc_j, j| {
                (0..num_auxbas).fold(acc_j, |acc_q, q| {
                    acc_q + ri_oo_tilde_ref[[q, i + j * occ_size]] * t_tensor_ref[[q, j + a * occ_size]]
                })
            })
        })
    }).collect();
    let duration3=start.elapsed();
    println!("W第三个操作耗时: {:?}", duration3-duration2);
    result
}
pub fn w_contribution_a_block_dgemm(scf_data:&SCF,ri_vv:&MatrixFull<f64>,z_vec:&Vec<f64>,ri_oo_tilde:&MatrixFull<f64>,qp_ctrl:&QuasiParticle)->Vec<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let num_auxbas=ri_vv.size[0]/ri_vv.size[1];
    let mut t_tensor=MatrixFull::new([num_auxbas*vir_size,occ_size],0.0);
    let z_mat=MatrixFull::from_vec([occ_size,vir_size],z_vec.clone()).unwrap();
    _dgemm_full(ri_vv,'N',&z_mat,'T',&mut t_tensor,1.0,0.0);
    t_tensor=t_tensor.transpose_and_drop();
    t_tensor.reshape([num_auxbas*occ_size,vir_size]);
    let mut result_tensor=MatrixFull::new([occ_size,vir_size],0.0);
    _dgemm_full(ri_oo_tilde,'T',&t_tensor,'N',&mut result_tensor,1.0,0.0);
    result_tensor.data.iter().map(|x|x*qp_ctrl.bse_exchange_rescaling).collect()
}
pub fn w_contribution_b_block_dgemm(scf_data:&SCF,ri_ov:&MatrixFull<f64>,z_vec:&Vec<f64>,ri_ov_tilde:&MatrixFull<f64>)->Vec<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let num_auxbas=ri_ov.size[0]/occ_size;
    let mut t_tensor=MatrixFull::new([num_auxbas*occ_size,occ_size],0.0);
    let z_mat=MatrixFull::from_vec([occ_size,vir_size],z_vec.clone()).unwrap();
    _dgemm_full(ri_ov,'N',&z_mat,'T',&mut t_tensor,1.0,0.0);
    let mut result = vec![0.0; t_tensor.data.len()];
    // 并行处理每个目标块
    let reshape_t_time=Instant::now();
    result.par_chunks_exact_mut(num_auxbas).enumerate().for_each(|(new_idx, target_chunk)| {
        // 计算对应的原块索引
        let n2 = new_idx / occ_size;  // 新的行索引（原列索引）
        let n1 = new_idx % occ_size;  // 新的列索引（原行索引）
        let orig_idx = n1 * occ_size + n2;  // 原块索引
        
        // 复制数据
        let source_start = orig_idx * num_auxbas;
        let source_end = source_start + num_auxbas;
        target_chunk.copy_from_slice(&t_tensor.data[source_start..source_end]);
    });
    t_tensor=MatrixFull::from_vec([num_auxbas*occ_size,occ_size],result).unwrap();
    let mut result_tensor=MatrixFull::new([occ_size,vir_size],0.0);
    _dgemm_full(&t_tensor,'T',ri_ov_tilde,'N',&mut result_tensor,1.0,0.0);
    result_tensor.data
}
pub fn w_contribution_rayon_b_block(scf_data:&SCF,ri_ov:&MatrixFull<f64>,z_vec:&Vec<f64>,ri_ov_tilde:&MatrixFull<f64>)->Vec<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let num_auxbas=ri_ov_tilde.size[0];
    // 使用并行化计算 t_tensor - 使用 AtomicPtr 解决所有权问题
    let mut t_tensor_data = vec![0.0; num_auxbas * occ_size * occ_size];
    let t_tensor_ptr = AtomicPtr::new(t_tensor_data.as_mut_ptr());
    
    // 并行化外层循环：对 j 进行并行化
    (0..occ_size).into_par_iter().for_each(|j| {
        let z_vec_ref = &z_vec;
        let ptr = t_tensor_ptr.load(Ordering::Relaxed);
        
        for i in 0..occ_size {
            for q in 0..num_auxbas {
                let mut sum = 0.0;
                for b in 0..vir_size {
                    sum += z_vec_ref[j + b * occ_size] * ri_ov[[q, b*occ_size+i]];
                }
                // 使用与串行版本相同的索引方式：q + (j + a * occ_size) * num_auxbas
                let index = q + (j + i * occ_size) * num_auxbas;
                unsafe {
                    *ptr.add(index) = sum;
                }
            }
        }
    });
    let t_tensor = MatrixFull::from_vec([num_auxbas, occ_size * occ_size], t_tensor_data).unwrap();
    // 并行化第二部分：计算最终结果
    let result: Vec<f64> = (0..vir_size).into_par_iter().flat_map(|a| {
        let ri_ov_tilde_ref = &ri_ov_tilde;
        let t_tensor_ref = &t_tensor;
        (0..occ_size).into_par_iter().map(move |i| {
            (0..occ_size).fold(0.0, |acc_j, j| {
                (0..num_auxbas).fold(acc_j, |acc_q, q| {
                    acc_q + ri_ov_tilde_ref[[q, j + a * occ_size]] * t_tensor_ref[[q, j + i * occ_size]]
                })
            })
        })
    }).collect();
    result
}
pub fn test_v_w_contribution(scf_data:&SCF){
    test_v_w_contribution_v01(scf_data);
}
pub fn test_v_w_contribution_v01(scf_data:&SCF){
    let ri_ov=ri_bse::get_submatrix(scf_data,'O','V','N');
    let ri_vv=ri_bse::get_submatrix(scf_data,'V','V','N');
    let ri_oo=ri_bse::get_submatrix(scf_data,'O','O','N');
    let num_auxbas=ri_ov.size[0];
    let mut ri_oo_tilde:MatrixFull<f64>=MatrixFull::new(ri_oo.size,0.0);
    let ks_energies:Vec<f64>=scf_data.eigenvalues[0].clone();
    let inverse_dielectric=ri_bse::construct_inverse_dielectric(scf_data,&ks_energies);
    _dgemm_full(&inverse_dielectric,'N',&ri_oo,'N',&mut ri_oo_tilde,1.0,0.0);
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let mut v=ri_bse::construct_coulomb(&ri_ov,&ri_ov);
    let raw_w=ri_bse::construct_raw_w(&ri_oo,&ri_vv,&inverse_dielectric);
    let w=ri_bse::reorganize_w(raw_w, 'A', occ_size, vir_size);
    let mut z_vec=vec![0.0;occ_size*vir_size];
    z_vec[1]=1.0;
    let mut az=vec![0.0;occ_size*vir_size];
    _dgemv(&v,&z_vec , &mut az, 'N', 1.0, 0.0, 1, 1);
    println!("Vz Exact:{}",az[3]);
    az=vec![0.0;occ_size*vir_size];
    _dgemv(&w,&z_vec , &mut az, 'N', 1.0, 0.0, 1, 1);
    println!("Wz Exact:{},{}",az[0],az[3]);
    az=coulomb_contribution(&ri_ov,&z_vec);
    println!("Vz Implicit:{},{}",az[0],az[3]);
    let start1=Instant::now();
    az=w_contribution(scf_data,&z_vec,&ri_oo_tilde);
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let serial_time=start1.elapsed();
    println!("Serial W contribution time={:?}",serial_time);
    println!("Wz Implicit:{},{}",az[0],az[3]);
    let dgemm_time=Instant::now();
    let mut ri_oo_tilde_reshape=ri_oo_tilde.clone();
    ri_oo_tilde_reshape.reshape([num_auxbas*occ_size,occ_size]);
    ri_oo_tilde_reshape=ri_oo_tilde_reshape.transpose_and_drop();
    ri_oo_tilde_reshape.reshape([occ_size*num_auxbas,occ_size]);
    let mut ri_vv_reshape=ri_vv.clone();
    ri_vv_reshape.reshape([num_auxbas*vir_size,vir_size]);
    let prep_time=dgemm_time.elapsed();
    println!("DGEMM Prep Time={:?}",prep_time);
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let az=w_contribution_a_block_dgemm(scf_data,&ri_vv_reshape,&z_vec,&ri_oo_tilde_reshape,&qp_ctrl);
    let calc_time=start1.elapsed();
    println!("DGEMM W contribution time={:?}",calc_time-prep_time);
    println!("Wz DGEMM:{},{}",az[0],az[3]);
    //let start2=Instant::now();
    //az=w_contribution_rayon_a_block(scf_data,&z_vec,&ri_oo_tilde);
    //let rayon_time=start2.elapsed();
    //println!("Rayon W contribution time={:?}",rayon_time);
    //println!("Wz Implicit Rayon:{},{}",az[0],az[3]);
    let raw_w_b=ri_bse::construct_raw_w(&ri_ov,&ri_ov,&inverse_dielectric);
    let mut w_b=ri_bse::reorganize_w(raw_w_b, 'B', occ_size, vir_size);
    let mut ri_ov_tilde:MatrixFull<f64>=MatrixFull::new(ri_ov.size,0.0);
    _dgemm_full(&inverse_dielectric,'N',&ri_ov,'N',&mut ri_ov_tilde,1.0,0.0);
    let bblock_timing=Instant::now();
    let bz=w_contribution_rayon_b_block(scf_data,&ri_ov,&z_vec,&ri_ov_tilde);
    println!("Wz_B Implicit:{},{}",bz[0],bz[2]);
    let rayon_time=bblock_timing.elapsed();
    println!("B Block Rayon Time={:?}",rayon_time);
    //println!("RI OV TILDE=");
    //ri_ov_tilde.formated_output(1000,"full");
    //println!("RI OV=");
    //ri_ov.formated_output(1000,"full");
    //println!("DATA={:#?}",z_vec);
    let restart_begin=Instant::now();
    ri_ov_tilde.reshape([num_auxbas*occ_size,vir_size]);
    //println!("RI OV TILDE RESHAPED=");
    //ri_ov_tilde.formated_output(1000,"full");
    println!("Reshape Time={:?}",restart_begin.elapsed());
    let mut bz=vec![0.0;occ_size*vir_size];
    let mut ri_ov_reshape=ri_ov.clone();
    ri_ov_reshape.reshape([num_auxbas*occ_size,vir_size]);
    _dgemv(&w_b,&z_vec , &mut bz, 'N', 1.0, 0.0, 1, 1);
    println!("Wz_B Exact:{:?}",bz);
    let bblock_dgemm_timing=Instant::now();
    let mut bz=w_contribution_b_block_dgemm(scf_data,&ri_ov_reshape,&z_vec,&ri_ov_tilde);
    println!("B Block DGEMM Time={:?}",bblock_dgemm_timing.elapsed());
    println!("Wz_B DGEMM:{:?}",bz);
    /*let auxbas_dir=scf_data.mol.ctrl.auxbas_path.clone();
    let elements=scf_data.mol.geom.elem.clone();
    let relevant_indices=sbse::obtain_relevant_indices(&elements,&auxbas_dir,0);
    
    let ri_oo=ri_bse::get_submatrix(scf_data,'O','O','N');
    let mut ri_vv_new=ri_bse::get_submatrix(scf_data,'V','V','N');
    println!("Full NumAuxBas={}\nRelevant Indices={:?}",ri_ov.size[0],relevant_indices);
    let mut ri_oo_tilde_small=sbse::obtain_ri_with_reduced_ang_momentum(&ri_oo,&relevant_indices);
    println!("RI-OO-Tilde Size={},{}, where occ_size={}",ri_oo_tilde_small.size[0],ri_oo_tilde_small.size[1],occ_size);
    let reduced_num_auxbas=ri_oo_tilde_small.size[0];
    println!("Reduced NumAuxBas={}",reduced_num_auxbas);
    ri_oo_tilde_small.reshape([reduced_num_auxbas*occ_size,occ_size]);
    ri_oo_tilde_small=ri_oo_tilde_small.transpose_and_drop();
    ri_oo_tilde_small.reshape([occ_size*reduced_num_auxbas,occ_size]);
    let mut ri_vv_small=sbse::obtain_ri_with_reduced_ang_momentum(&ri_vv_new,&relevant_indices);
    ri_vv_small.reshape([reduced_num_auxbas*vir_size,vir_size]);
    let ratio=sbse::power_iterative_norm(
        |z|w_contribution_a_block_dgemm(scf_data,&ri_vv_reshape,&z,&ri_oo_tilde_reshape,&qp_ctrl),
        |z|w_contribution_a_block_dgemm(scf_data,&ri_vv_small,&z,&ri_oo_tilde_small,&qp_ctrl),
        occ_size*vir_size);
    */



    /*let paired_vec=davidson_solver::PairedVector{x:vec![0.3;occ_size*vir_size],y:vec![0.3;occ_size*vir_size]};
    let paired_product=paired_vec.pair_matvec({|z|a_block_matvec(scf_data,&ri_oo_tilde,&z)},{|z|b_block_matvec(scf_data,&ri_ov_tilde,&z)});
    let full_product={let mut x_vec=paired_product.x;let y_vec=paired_product.y;x_vec.extend(&y_vec);x_vec};
    println!("Full matvec Implicit:{},{}",full_product[0],full_product[occ_size*vir_size+3]);
    let xlet=if scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap().bse_spin=="triplet"{'T'}else{'S'};
    let bse_hamiltonian=ri_bse::construct_full_bse_hamitonian(scf_data, xlet,&inverse_dielectric,&scf_data.gwqp.0.clone());
    let full_vec=vec![0.3;occ_size*vir_size*2];
    let mut result=vec![0.0;occ_size*vir_size*2];
    _dgemv(&bse_hamiltonian,&full_vec , &mut result, 'N', 1.0, 0.0, 1, 1);
    println!("Full matvec explicit{},{}",result[0],result[occ_size*vir_size+3])*/
}
pub fn a_block_matvec(scf_data:&SCF,qp_ctrl:&QuasiParticle,ri_vv:&MatrixFull<f64>,ri_ov:&MatrixFull<f64>,ri_oo_tilde:&MatrixFull<f64>,z_vec:&Vec<f64>)->Vec<f64>{
    let start=Instant::now();
    let xlet=if qp_ctrl.bse_spin=="triplet"{'T'}else if qp_ctrl.bse_spin=="singlet"{'S'}else{'R'};
    let mut result=diagonal_elements_contribution(scf_data,z_vec);
    let duration1=start.elapsed();
    if scf_data.mol.ctrl.print_level>1{
        println!("对角元操作耗时: {:?}", duration1);   
    }
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    result=w_contribution_a_block_dgemm(scf_data,ri_vv,z_vec,ri_oo_tilde,&qp_ctrl).iter().zip(result.iter()).map(|(w_i,z_i)|-w_i+z_i).collect();
    let duration2=start.elapsed();
    if scf_data.mol.ctrl.print_level>1{
        println!("W操作耗时: {:?}", duration2-duration1); 
    }
    if xlet=='S'{
        result=coulomb_contribution(ri_ov,z_vec).iter().zip(result.iter()).map(|(v_i,z_i)|2.0*v_i+z_i).collect();
        let duration3=start.elapsed();
        if scf_data.mol.ctrl.print_level>1{
            println!("库仑操作耗时: {:?}", duration3-duration2);
        }
    }
    if xlet=='R'{
        result=coulomb_contribution(ri_ov,z_vec).iter().zip(result.iter()).map(|(v_i,z_i)|v_i+z_i).collect();
        let duration3=start.elapsed();
        if scf_data.mol.ctrl.print_level>1{
            println!("库仑操作耗时: {:?}", duration3-duration2);
        }
    }
    result
}
pub fn b_block_matvec(scf_data:&SCF,qp_ctrl:&QuasiParticle,ri_ov_a:&MatrixFull<f64>,ri_ov_b:&MatrixFull<f64>,ri_ov_tilde:&MatrixFull<f64>,z_vec:&Vec<f64>)->Vec<f64>{
    let xlet=if qp_ctrl.bse_spin=="triplet"{'T'}else if qp_ctrl.bse_spin=="singlet"{'S'}else{'R'};
    let mut result=vec![0.0;z_vec.len()];
    let mut ri_ov_tilde_old=ri_ov_tilde.clone();
    ri_ov_tilde_old.reshape(ri_ov_a.size);
    result=w_contribution_b_block_dgemm(scf_data,ri_ov_b,z_vec,ri_ov_tilde).iter().zip(result.iter()).map(|(w_i,z_i)|-w_i+z_i).collect();
    if xlet=='S'{
        result=coulomb_contribution(ri_ov_a,z_vec).iter().zip(result.iter()).map(|(v_i,z_i)|2.0*v_i+z_i).collect();
    }
    if xlet=='R'{
        result=coulomb_contribution(ri_ov_a,z_vec).iter().zip(result.iter()).map(|(v_i,z_i)|v_i+z_i).collect();
    }
    result
}
pub fn sbse_matvec(scf_data:&SCF,mo_coeff:&MatrixFull<f64>,w_ao_basis:&MatrixFull<f64>,occ_size:usize,vir_size:usize,ri_ov:&MatrixFull<f64>,z_vec:&Vec<f64>)->Vec<f64>{
    let xlet=if scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap().bse_spin=="triplet"{'T'}else{'S'};
    let mut result=diagonal_elements_contribution(scf_data,z_vec);
    result=sbse::sbse_matvec_w_contribution(w_ao_basis,mo_coeff,occ_size,vir_size,z_vec).iter().zip(result.iter()).map(|(w_i,z_i)|-w_i+z_i).collect();
    if xlet=='S'{
        result=coulomb_contribution(ri_ov,z_vec).iter().zip(result.iter()).map(|(v_i,z_i)|2.0*v_i+z_i).collect();
    }
    result
}
pub fn test_v_w_contribution_v02(scf_data:&SCF){
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    //ri_bse::prepare_ri3mo(scf_data,'N');
    let (ri3ao, mut basbas2baspair, mut baspar2basbas) =  if let Some((riao,basbas2baspair, baspar2basbas))=&scf_data.rimatr {
        (riao,basbas2baspair, baspar2basbas)
    } else {
        panic!("rimatr should be initialized in the preparation of riao");
    };
    let ri3ao=ri3ao.transpose();
    let mut epsilon:Vec<f64>=scf_data.eigenvalues[0].clone();
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let energy_diag=ri_bse::construct_energy_diag_for_a(&scf_data.gwqp.0,occ_size,vir_size);
    let inverse_dielectric=ri_bse::construct_inverse_dielectric(scf_data,&epsilon);
    let w_ao=sbse::w_ao_basis(&inverse_dielectric,epsilon.len(),ri3ao.clone());
    let mo_coeff=scf_data.eigenvectors[0].clone();
    let ri_ov=ri_bse::get_submatrix(scf_data,'O','V','N');
    let vector1=vec![1.0;occ_size*vir_size];
    let mut vector2=vector1.clone();
    vector2[3]=3.0;
    vector2[5]=10.0;
    let vec1_p_vec2=vector1.iter().zip(vector2.iter()).map(|(v1,v2)|v1+v2).collect();
    let sbse_matvec=|z|sbse_matvec(scf_data,&mo_coeff,&w_ao,occ_size,vir_size,&ri_ov,&z);
    let av1=sbse_matvec(vector1);
    let av2=sbse_matvec(vector2);
    let av1pv2=sbse_matvec(vec1_p_vec2);
    let av1pav2:Vec<f64>=av1.iter().zip(av2.iter()).map(|(v1,v2)|v1+v2).collect();
    println!("A(v1+v2)={:#?}",av1pv2);
    println!("Av1+Av2={:#?}",av1pav2);
}