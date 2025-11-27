use crate::scf_io::{SCF, SCFType};
use tensors::{MathMatrix, MatrixFull, RIFull,MatrixFullSlice};
use crate::ri_gw::get_occupation_parameters;
use itertools::Itertools;
use rest_tensors::matrix::matrix_blas_lapack::{_dgeev, _dgemv,_dgemm,_dgemm_full};
use crate::ri_bse;
use crate::ri_bse::davidson_solver;
use rayon::prelude::*;
use std::time::Instant;
use std::sync::atomic::{AtomicPtr, Ordering};
use regex::Regex;
use std::fs;
use std::path::Path;

pub fn w_ao_basis(inverse_dielectric:&MatrixFull<f64>,num_state:usize,ri3ao:MatrixFull<f64>)->MatrixFull<f64>{
    println!("size of ri3ao:{},{}",ri3ao.size[0],ri3ao.size[1]);
    println!("number of orbitals:{}",num_state);
    let iterator=ri3ao.iter_columns_full().enumerate();
    let num_auxbas=inverse_dielectric.size[0];
    let mut w_c=MatrixFull::new([num_state,num_state],0.0);
    let mut first_product=vec![0.0;num_auxbas];
    let mut row=0;
    let mut column=0;
    for (a,vec) in iterator{
        row=if column==0{0}else{a-(column*(column+1)/2)};
        _dgemv(inverse_dielectric,vec,&mut first_product,'N', 1.0, 0.0, 1, 1);
        w_c[[row,column]]=first_product.iter().zip(vec.iter()).map(|(a,b)|a*b).sum();
        w_c[[column,row]]=w_c[[row,column]];
        if row==column{
            column+=1;
        }
    }
    //println!("W matrix in AO basis:");
    //w_c.formated_output(1000,"full");
    w_c
}
pub fn sbse_matvec_w_contribution(w_ao_basis:&MatrixFull<f64>,mo_coeff:&MatrixFull<f64>,occ_size:usize,vir_size:usize,z:&Vec<f64>)->Vec<f64>{
    let nao=w_ao_basis.size[0];
    let z_matrix=MatrixFull::from_vec([occ_size,vir_size],z.clone()).unwrap();
    //println!("MatVec Debug:z_vector:{:#?},z matrix:",z);
    //z_matrix.formated_output(1000,"full");
    let mut c_nub_z_jb=MatrixFull::new([nao,occ_size],0.0);
    _dgemm(mo_coeff,((0..nao),(occ_size..occ_size+vir_size)),'N',&z_matrix,((0..occ_size),(0..vir_size)),'T',&mut c_nub_z_jb,((0..nao),(0..occ_size)),1.0,0.0);
    let mut k_mu_nu=MatrixFull::new([nao,nao],0.0);
    _dgemm(mo_coeff,((0..nao),(0..occ_size)),'N',&c_nub_z_jb,((0..nao),(0..occ_size)),'T',&mut k_mu_nu,((0..nao),(0..nao)),1.0,0.0);
    let wk_data=k_mu_nu.data.iter().zip(w_ao_basis.data.iter()).map(|(k,w)|k*w).collect();
    let wk=MatrixFull::from_vec([nao,nao],wk_data).unwrap();
    let mut c_nu_a_wk_mu_nu=MatrixFull::new([nao,vir_size],0.0);
    _dgemm(&wk,((0..nao),(0..nao)),'N',mo_coeff,((0..nao),(0..vir_size)),'N',&mut c_nu_a_wk_mu_nu,((0..nao),(0..vir_size)),1.0,0.0);
    let mut w_z_matrix=MatrixFull::new([occ_size,vir_size],0.0);
    _dgemm(mo_coeff,((0..nao),(0..occ_size)),'T',&c_nu_a_wk_mu_nu,((0..nao),(0..vir_size)),'N',&mut w_z_matrix,((0..occ_size),(0..vir_size)),1.0,0.0);
    let result=w_z_matrix.data;
    //println!("MatVec:{:#?} into: {:#?}",z,result);
    result
}
pub fn count_angular_momentum_regex<P>(filename: P,angular_momentum:usize) -> Result<usize, Box<dyn std::error::Error>>
where
    P: AsRef<Path>,
{
    let content = fs::read_to_string(filename)?;
    
    // 创建正则表达式模式，注意要匹配确切的缩进
    let pattern = format!(r#"\[\s*{}\s*\]"#,angular_momentum);
    let re = Regex::new(&pattern)?;
    
    Ok(re.find_iter(&content).count())
}
pub fn count_all_ao<P>(filename: P) -> Result<usize, Box<dyn std::error::Error>>
where
    P: AsRef<Path>,
{
    let content = fs::read_to_string(filename)?;
    
    // 创建正则表达式模式，注意要匹配确切的缩进
    let pattern = r#"angular_momentum"#;
    let re = Regex::new(&pattern)?;
    
    Ok(re.find_iter(&content).count())
}
pub fn obtain_relevant_indices(elements:&Vec<String>,auxbas_dir:&String,max_angular_momentum:usize)->Vec<usize>{
    let mut indices=vec![1;0];
    let mut starting_index=0;
    elements.iter().for_each(|elem|{
        let tot_num=count_all_ao(format!("{}/{}.json",auxbas_dir,elem));
        let mut basis_funcs_pushed=0;
        (0..max_angular_momentum).for_each(|angular_momentum|basis_funcs_pushed+=count_angular_momentum_regex(format!("{}/{}.json",auxbas_dir,elem),angular_momentum));
        (0..basis_funcs_pushed).for_each(|n|indices.push(starting_index+n));
        starting_index+=tot_num;
    });
    indices
}