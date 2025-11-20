use crate::scf_io::{SCF, SCFType};
use tensors::{MathMatrix, MatrixFull, RIFull,MatrixFullSlice};
use crate::ri_gw::get_occupation_parameters;
use itertools::Itertools;
use rest_tensors::matrix::matrix_blas_lapack::{_dgeev, _dgemv,_dgemm_full};
use crate::ri_bse;
use crate::ri_bse::davidson_solver;
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
pub fn coulomb_contribution(scf_data:&SCF,vec:&Vec<f64>)->Vec<f64>{
    let ri_matrix=ri_bse::get_submatrix(scf_data,'O','V','N');
    let mut inter_result=vec![0.0;ri_matrix.size[0]];
    _dgemv(&ri_matrix,vec , &mut inter_result, 'N', 1.0, 0.0, 1, 1);
    let mut result=vec![0.0;ri_matrix.size[1]];
    _dgemv(&ri_matrix,&inter_result , &mut result, 'T', 1.0, 0.0, 1, 1);
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
pub fn w_contribution_rayon_a_block(scf_data:&SCF,z_vec:&Vec<f64>,ri_oo_tilde:&MatrixFull<f64>)->Vec<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let num_auxbas=ri_oo_tilde.size[0];
    let ri_vv=ri_bse::get_submatrix(scf_data,'V','V','N');
    // 使用并行化计算 t_tensor - 使用 AtomicPtr 解决所有权问题
    let mut t_tensor_data = vec![0.0; num_auxbas * occ_size * vir_size];
    let t_tensor_ptr = AtomicPtr::new(t_tensor_data.as_mut_ptr());
    
    // 并行化外层循环：对 j 进行并行化
    (0..occ_size).into_par_iter().for_each(|j| {
        let ri_vv_ref = &ri_vv;
        let z_vec_ref = &z_vec;
        let ptr = t_tensor_ptr.load(Ordering::Relaxed);
        
        for a in 0..vir_size {
            for q in 0..num_auxbas {
                let mut sum = 0.0;
                for b in 0..vir_size {
                    sum += z_vec_ref[j + b * occ_size] * ri_vv_ref[[q, a * vir_size + b]];
                }
                // 使用与串行版本相同的索引方式：q + (j + a * occ_size) * num_auxbas
                let index = q + (j + a * occ_size) * num_auxbas;
                unsafe {
                    *ptr.add(index) = sum;
                }
            }
        }
    });
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
    result
}
pub fn w_contribution_rayon_b_block(scf_data:&SCF,z_vec:&Vec<f64>,ri_ov_tilde:&MatrixFull<f64>)->Vec<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let num_auxbas=ri_ov_tilde.size[0];
    let ri_ov=ri_bse::get_submatrix(scf_data,'O','V','N');
    // 使用并行化计算 t_tensor - 使用 AtomicPtr 解决所有权问题
    let mut t_tensor_data = vec![0.0; num_auxbas * occ_size * occ_size];
    let t_tensor_ptr = AtomicPtr::new(t_tensor_data.as_mut_ptr());
    
    // 并行化外层循环：对 j 进行并行化
    (0..occ_size).into_par_iter().for_each(|j| {
        let ri_ov_ref = &ri_ov;
        let z_vec_ref = &z_vec;
        let ptr = t_tensor_ptr.load(Ordering::Relaxed);
        
        for i in 0..occ_size {
            for q in 0..num_auxbas {
                let mut sum = 0.0;
                for b in 0..vir_size {
                    sum += z_vec_ref[j + b * occ_size] * ri_ov_ref[[q, b*occ_size+i]];
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
    let ri_ov=ri_bse::get_submatrix(scf_data,'O','V','N');
    let ri_vv=ri_bse::get_submatrix(scf_data,'V','V','N');
    let ri_oo=ri_bse::get_submatrix(scf_data,'O','O','N');
    let mut ri_oo_tilde:MatrixFull<f64>=MatrixFull::new(ri_oo.size,0.0);
    let ks_energies:Vec<f64>=scf_data.eigenvalues[0].clone();
    let inverse_dielectric=ri_bse::construct_inverse_dielectric(scf_data,&ks_energies);
    _dgemm_full(&inverse_dielectric,'N',&ri_oo,'N',&mut ri_oo_tilde,1.0,0.0);
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let mut v=ri_bse::construct_coulomb(&ri_ov,&ri_ov);
    let raw_w=ri_bse::construct_raw_w(&ri_oo,&ri_vv,&inverse_dielectric);
    let w=ri_bse::reorganize_w(raw_w, 'A', occ_size, vir_size);
    let z_vec=vec![0.3;occ_size*vir_size];
    let mut az=vec![0.0;occ_size*vir_size];
    _dgemv(&v,&z_vec , &mut az, 'N', 1.0, 0.0, 1, 1);
    println!("Vz Exact:{}",az[0]);
    az=vec![0.0;occ_size*vir_size];
    _dgemv(&w,&z_vec , &mut az, 'N', 1.0, 0.0, 1, 1);
    println!("Wz Exact:{}",az[0]);
    az=coulomb_contribution(scf_data,&z_vec);
    println!("Vz Implicit:{}",az[0]);
    let start1=Instant::now();
    az=w_contribution(scf_data,&z_vec,&ri_oo_tilde);
    let serial_time=start1.elapsed();
    println!("Serial W contribution time={:?}",serial_time);
    println!("Wz Implicit:{}",az[3]);
    let start2=Instant::now();
    az=w_contribution_rayon_a_block(scf_data,&z_vec,&ri_oo_tilde);
    let rayon_time=start2.elapsed();
    println!("Rayon W contribution time={:?}",rayon_time);
    println!("Wz Implicit Rayon:{}",az[3]);
    let raw_w_b=ri_bse::construct_raw_w(&ri_ov,&ri_ov,&inverse_dielectric);
    let mut w_b=ri_bse::reorganize_w(raw_w_b, 'B', occ_size, vir_size);
    let mut ri_ov_tilde:MatrixFull<f64>=MatrixFull::new(ri_ov.size,0.0);
    _dgemm_full(&inverse_dielectric,'N',&ri_ov,'N',&mut ri_ov_tilde,1.0,0.0);
    let mut bz=vec![0.0;occ_size*vir_size];
    _dgemv(&w_b,&z_vec , &mut bz, 'N', 1.0, 0.0, 1, 1);
    println!("Wz_B Exact:{},{}",bz[0],bz[2]);
    bz=w_contribution_rayon_b_block(scf_data,&z_vec,&ri_ov_tilde);
    println!("Wz_B Implicit:{},{}",bz[0],bz[2]);
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
pub fn a_block_matvec(scf_data:&SCF,ri_oo_tilde:&MatrixFull<f64>,z_vec:&Vec<f64>)->Vec<f64>{
    let xlet=if scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap().bse_spin=="triplet"{'T'}else{'S'};
    let mut result=diagonal_elements_contribution(scf_data,z_vec);
    result=w_contribution_rayon_a_block(scf_data,z_vec,ri_oo_tilde).iter().zip(result.iter()).map(|(w_i,z_i)|-w_i+z_i).collect();
    if xlet=='S'{
        result=coulomb_contribution(scf_data,z_vec).iter().zip(result.iter()).map(|(v_i,z_i)|2.0*v_i+z_i).collect();
    }
    result
}
pub fn b_block_matvec(scf_data:&SCF,ri_ov_tilde:&MatrixFull<f64>,z_vec:&Vec<f64>)->Vec<f64>{
    let xlet=if scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap().bse_spin=="triplet"{'T'}else{'S'};
    let mut result=vec![0.0;z_vec.len()];
    result=w_contribution_rayon_b_block(scf_data,z_vec,ri_ov_tilde).iter().zip(result.iter()).map(|(w_i,z_i)|-w_i+z_i).collect();
    if xlet=='S'{
        result=coulomb_contribution(scf_data,z_vec).iter().zip(result.iter()).map(|(v_i,z_i)|2.0*v_i+z_i).collect();
    }
    result
}