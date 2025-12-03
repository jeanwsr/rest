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
use crate::ctrl_io::quasiparticle_methods::QuasiParticle;

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
pub fn count_all_ang_momentum_under_value<P>(filename: P,angular_momentum:usize)-> usize
where
    P: AsRef<Path>+Clone,
{
    let mut count=0;
    (0..angular_momentum+1).for_each(|j|count+=count_angular_momentum_regex(filename.clone(),j).unwrap());
    count
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
        let tot_num=count_all_ao(format!("{}/{}.json",auxbas_dir,elem)).unwrap();
        println!("Now Atom={}",elem);
        let basis_funcs_pushed=count_all_ang_momentum_under_value(format!("{}/{}.json",auxbas_dir,elem),max_angular_momentum);
        (0..basis_funcs_pushed).for_each(|n|{indices.push(starting_index+n);println!("Pushed index={}",starting_index+n)});
        starting_index+=tot_num;
    });
    indices
}
pub fn obtain_ri_with_reduced_ang_momentum(ri_matr:&MatrixFull<f64>,indices:&Vec<usize>)->MatrixFull<f64>{
    let mut now_index=0;
    let ri_matr_t=ri_matr.transpose();
    let mut reduced_ri_matr=MatrixFull::new([ri_matr.size[1],0],0.0);
    ri_matr_t.iter_columns_full().enumerate().for_each(|(n,ri_n)|{
        if now_index<indices.len(){if n==indices[now_index]{
            reduced_ri_matr.push_column(ri_n);
            now_index+=1;
        }}
    });
    reduced_ri_matr.transpose()
}
pub fn power_iterative_norm<F1,F2>(matvec1:F1,matvec2:F2,occ_vir:usize)->f64 where F1:Fn(&Vec<f64>)->Vec<f64>,F2:Fn(&Vec<f64>)->Vec<f64>{
    println!("start power method procedure for finding norm");
    let mut vector=vec![1.0;occ_vir];
    let mut avec=matvec1(&vector);
    let mut atavec=matvec1(&avec);
    let mut norm=atavec.iter().fold(0.0,|acc,x|acc+x.powf(2.0)).powf(0.5);
    vector=atavec.iter().map(|x|x/norm).collect();
    avec=matvec1(&vector);
    atavec=matvec1(&avec);
    let mut estimate=vector.iter().zip(atavec.iter()).fold(0.0,|acc,(x,y)|acc+x*y).powf(0.5);
    let mut new_estimate=0.0;
    let mut converge=false;
    let mut count=0;
    loop{
        norm=atavec.iter().fold(0.0,|acc,x|acc+x.powf(2.0)).powf(0.5);
        vector=atavec.iter().map(|x|x/norm).collect();
        new_estimate=vector.iter().zip(atavec.iter()).fold(0.0,|acc,(x,y)|acc+x*y).powf(0.5);
        println!("Residue currently={}",new_estimate-estimate);
        if (new_estimate-estimate).abs()<0.0001{
            break
        }
        count+=1;
        estimate=new_estimate;
        avec=matvec1(&vector);
        atavec=matvec1(&avec);
    }
    let norm1=estimate;
    println!("Norm 1={} obtained after {} power iterations",norm1,count);
    let mut count=0;
    avec=matvec2(&vector);
    atavec=matvec2(&avec);
    loop{
        norm=atavec.iter().fold(0.0,|acc,x|acc+x.powf(2.0)).powf(0.5);
        vector=atavec.iter().map(|x|x/norm).collect();
        new_estimate=vector.iter().zip(atavec.iter()).fold(0.0,|acc,(x,y)|acc+x*y).powf(0.5);
        println!("Residue currently={}",new_estimate-estimate);
        if (new_estimate-estimate).abs()<0.0001{
            break
        }
        count+=1;
        estimate=new_estimate;
        avec=matvec2(&vector);
        atavec=matvec2(&avec);
    }
    let norm2=estimate;
    println!("Norm 2={} obtained after {} power iterations",norm2,count);
    norm1/norm2
}