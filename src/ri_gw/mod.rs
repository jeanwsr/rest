//use std::simd::num;
use crate::constants::PI;
use itertools::Itertools;
use std::ops::Range;
use crate::utilities;
use crate::scf_io::SCF;
//use std::slice::Iter::<'_, f64>;
use std::cmp;
use std::fs::OpenOptions;
use std::io::Write;
use rayon::iter::ParallelBridge;
use rayon::result;
use reqwest::blocking::Response;
use std::time::Instant;
use std::sync::{Arc, Mutex};
use rest_tensors::{RIFull};
use rayon::prelude::ParallelSliceMut;
use tensors::{matrix_blas_lapack::{_dinverse,_dsyev}, ri, MathMatrix, MatrixFull};
//use rest::molecule_io::Molecule;
use rest_tensors::matrix::matrix_blas_lapack::{_dgees,_dgemm_full,_dgemv};
use crate::tensors::matrix_blas_lapack::omp_get_num_threads_wrapper;
use rest_tensors::MatrixUpper;
use crate::ri_bse;
use crate::ri_rpa;
use crate::molecule_io;
use std::{fs, io};
use crate::scf_io;
use crate::dft;
use crate::dft::DFA4REST;
use std::sync::mpsc::channel;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
pub mod renormalized_singles;
pub mod scgw;
pub mod display;
pub mod fourier_self_energy;
pub mod qsgw;
use crate::mpi_io::MPIOperator;

#[cfg(target_os = "linux")]
use libc::seccomp_notif;

pub fn gw_main(scf_data:&mut SCF,vxc_nn:&Vec<f64>,mpi_operator:&Option<MPIOperator>){
    let printlevel=scf_data.mol.ctrl.print_level.clone();
    show_energy_levels(scf_data);
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(&scf_data,'Y');
    let mut rs_particles:Vec<f64>=Vec::new();
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let renormalized_singles=qp_ctrl.renormalized_singles;
    if renormalized_singles==true{
        println!("Starts renormalized singles calculations!");
        let w_rs=qp_ctrl.w_rs;
        //println!("Now starting to compute rs_particles");
        rs_particles=renormalized_singles::renormalized_singles_diagonalization(scf_data,w_rs,mpi_operator);
        if printlevel>0{
            println!("Renormalized Singles particles:{:#?}",rs_particles);
        }
        scf_data.renormalized_singles_particles=rs_particles
    }
    initialize_qp_g_w(scf_data);
    let gw_scheme=qp_ctrl.gw_scheme.clone();
    let scgw=qp_ctrl.scgw.clone();
    let quasiparticle_energies:Vec<f64>=
        if scgw=="qsgw" {
            println!("You are doing QSGW (quasiparticle self-consistent GW) calculations");
            qsgw::qsgw_loop(scf_data, vxc_nn, mpi_operator)
        } else if scgw=="g0w0"&&renormalized_singles==false{
            println!("You are doing G0W0 calculations of entire energy spectrum");
            scgw::g0w0(scf_data,20,&vxc_nn,true)
        }else if scgw=="evgw"{
            println!("You are doing evGW calculations of entire energy spectrum");
            scgw::evgw(scf_data,20,&vxc_nn,qp_ctrl.evgw_rounds)
        }else if scgw=="g0w0"&&renormalized_singles==true{
            scgw::g0w0(scf_data,20,&vxc_nn,true)
        }else{panic!("invalid expression for scgw!")};
    if gw_scheme !="no gw"{
        println!("One round of GW by {} scheme has finished.",gw_scheme);
    }
    if qp_ctrl.save_qp_path.len()>0{
        let save_path=qp_ctrl.save_qp_path.clone();
        let mut file = OpenOptions::new().append(true).create(true).open(save_path);
        scf_data.gwqp.0.iter().for_each(|qp|{
            writeln!(file.as_ref().expect("write failure"), "{}",qp);
        });
    }
}
pub fn initialize_qp_g_w(scf_data:&mut SCF){
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    if qp_ctrl.renormalized_singles==true{
        let rsp=scf_data.renormalized_singles_particles.clone();
        if qp_ctrl.w_rs==true{
            scf_data.gwqp=(rsp.clone(),rsp);
        }else{
            let eigenenergies=scf_data.eigenvalues[0].clone();
            scf_data.gwqp=(rsp,eigenenergies);
        }
    }else{
        let eigenenergies=scf_data.eigenvalues[0].clone();
        scf_data.gwqp=(eigenenergies.clone(),eigenenergies);
    }
}
pub fn show_energy_levels(scf_data:&SCF){
    let num_state = scf_data.mol.num_state;
    let mut homo = 0_usize;
    let mut lumo = num_state;
    let start_mo = scf_data.mol.start_mo;
    for i_spin in 0..scf_data.mol.spin_channel {

        let i_homo = scf_data.homo.get(i_spin).unwrap().clone();
        let i_lumo = scf_data.lumo.get(i_spin).unwrap().clone();

        homo = homo.max(i_homo);
        lumo = lumo.min(i_lumo);
    }
    let occ_size=homo-start_mo+1;
    let vir_size=num_state-lumo;
    let eigenvalues=scf_data.eigenvalues[0].clone();
    for i_occ in 0..occ_size{
        if scf_data.mol.ctrl.print_level>1{
            println!("{}th occupied orbital, energy={}",i_occ+1,eigenvalues[start_mo+i_occ])   
        }
    }
    for i_vir in 0..vir_size{
        if scf_data.mol.ctrl.print_level>1{
            println!("{}th virtual orbital, energy={}",i_vir+1,eigenvalues[lumo+i_vir])
        }
    }
    let eigenvalues=scf_data.eigenvalues[0].clone();
    if scf_data.mol.ctrl.print_level>1{
        println!("homo:{},lumo:{},occ_size:{},vir_size:{},start_mo:{},num_state:{}",homo,lumo,occ_size,vir_size,start_mo,num_state);
    }
    
}

pub fn gw_calculations(scf_data:&mut SCF,num_freq:usize,vxc_nn:&Vec<f64>,cancel_dfa_xc:bool)->Vec<f64>{
    let mut ri_mat:MatrixFull<f64>=ri_bse::get_submatrix(scf_data,'F','F','Y');
    let mut ri_ov:MatrixFull<f64>=ri_bse::get_submatrix(scf_data,'O','V','Y');
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let v_matrix=v_matrix(&scf_data,&ri_mat);
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'Y');
    let (start_mo,num_state_cutoff,occ_size,vir_size_cutoff,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let hatree=27.2113863;
    let spin_channel=scf_data.mol.ctrl.spin_channel;
    let mut quasiparticle_energies_g:Vec<f64>=Vec::new();
    let mut quasiparticle_energies_w:Vec<f64>=Vec::new();
    let start:usize=0;
    let end:usize=6*occ_size;
    if scf_data.mol.ctrl.print_level>2{
        println!("ri_mat full:");
        ri_mat.formated_output(1000,"full");
    }
    let mut quasiparticle_energies_g=scf_data.gwqp.0.clone();
    let quasiparticle_energies_w=scf_data.gwqp.1.clone();

    // Route to low-rank accelerated version if enabled
    if qp_ctrl.use_low_rank_contour {
        println!("Low-rank contour deformation acceleration ENABLED.");
        println!("  real-axis grid points: {}", qp_ctrl.nomega_chi_real);
        println!("  eigenvalue tolerance:  {}", qp_ctrl.low_rank_tolerance);
        return gw_calculations_lowrank(
            scf_data, num_freq, vxc_nn, cancel_dfa_xc,
            qp_ctrl.nomega_chi_real, qp_ctrl.low_rank_tolerance,
        );
    }

    //display_and_save_quasiparticles(scf_data,&quasiparticle_energies_g,0);
    let w_c_at_freqs=generate_w_c(scf_data,&ri_ov,&ri_mat,&quasiparticle_energies_g,&quasiparticle_energies_w,num_state,occ_size,vir_size,num_freq);

    // 检查是否启用自能校正
    let use_fourier = qp_ctrl.fourier_self_energy;
    let use_hermite = qp_ctrl.hermite_self_energy;

    if use_fourier && use_hermite {
        panic!("Cannot enable both Fourier self-energy and Hermite self-energy simultaneously!");
    }

    // 第一轮：正常GW计算（不添加自能校正）
    let mut save_energies_no_se:Vec<f64>=vec![0.0;num_state_cutoff];
    save_energies_no_se=(0..num_state_cutoff).map(|n|{
        let consts=if cancel_dfa_xc==true{
            let mut exchange=0.0;
            for i in 0..homo+1{
                exchange-=v_matrix[[n,i]];
            }
            scf_data.eigenvalues[0][n]+exchange-vxc_nn[n]
        }else{quasiparticle_energies_g[n]};
        let side=if n>=occ_size{1.0}else{-1.0};
        let printlevel=scf_data.mol.ctrl.print_level.clone();
        let mut real_qp=0.0;
        let mut have_crossing=true;
        let qp_eq_func=|omega: f64|{
            quasiparticle_equation(omega,n,consts,&ri_ov,&ri_mat,&quasiparticle_energies_g,&quasiparticle_energies_w,occ_size,vir_size,num_state,&w_c_at_freqs)
        };
        (have_crossing,real_qp)=linear_interpolation_solver(qp_eq_func,scf_data.eigenvalues[0][n],side,21,0.1);
        println!("for n={}, first round (no self-energy correction): qp energy={}",n,real_qp);
        real_qp
    }).collect::<Vec<f64>>().clone();

    // 如果不需要自能校正，直接返回第一轮结果
    if !use_fourier && !use_hermite {
        quasiparticle_energies_g=save_energies_no_se.clone();
        scf_data.gwqp.0=quasiparticle_energies_g.clone();
        display::full_quasiparticles(&quasiparticle_energies_g,occ_size);
        return quasiparticle_energies_g;
    }

    // 第二轮：添加自能校正
    let mut save_energies:Vec<f64>=vec![0.0;num_state_cutoff];

    if use_fourier {
        let (powers, t, sin_coeff, cos_coeff) = fourier_self_energy::define_fourier_series(&qp_ctrl);
        println!("Fourier Self Energy enabled for second round");

        save_energies=(0..num_state_cutoff).map(|n|{
            let origin = save_energies_no_se[n];
            let consts=if cancel_dfa_xc==true{
                let mut exchange=0.0;
                for i in 0..homo+1{
                    exchange-=v_matrix[[n,i]];
                }
                scf_data.eigenvalues[0][n]+exchange-vxc_nn[n]
            }else{quasiparticle_energies_g[n]};
            let side=if n>=occ_size{1.0}else{-1.0};
            let printlevel=scf_data.mol.ctrl.print_level.clone();

            let qp_eq_func_with_fse=|omega: f64|{
                quasiparticle_equation(omega,n,consts,&ri_ov,&ri_mat,&quasiparticle_energies_g,&quasiparticle_energies_w,occ_size,vir_size,num_state,&w_c_at_freqs)
                    + fourier_self_energy::fourier_series(&sin_coeff, &cos_coeff, powers, t, omega - origin)
            };

            let (have_crossing,mut real_qp)=linear_interpolation_solver(qp_eq_func_with_fse,save_energies_no_se[n],side,qp_ctrl.gw_search_grid,qp_ctrl.gw_span_energy);
            println!("for n={}, second round (with FSE): qp energy={}",n,real_qp);
            real_qp
        }).collect::<Vec<f64>>().clone();
    } else if use_hermite {
        let hermite_coeff = fourier_self_energy::define_hermite_series(&qp_ctrl);
        println!("Hermite Self Energy enabled for second round");

        save_energies=(0..num_state_cutoff).map(|n|{
            let origin = save_energies_no_se[n];
            let consts=if cancel_dfa_xc==true{
                let mut exchange=0.0;
                for i in 0..homo+1{
                    exchange-=v_matrix[[n,i]];
                }
                scf_data.eigenvalues[0][n]+exchange-vxc_nn[n]
            }else{quasiparticle_energies_g[n]};
            let side=if n>=occ_size{1.0}else{-1.0};
            let printlevel=scf_data.mol.ctrl.print_level.clone();

            let qp_eq_func_with_hse=|omega: f64|{
                quasiparticle_equation(omega,n,consts,&ri_ov,&ri_mat,&quasiparticle_energies_g,&quasiparticle_energies_w,occ_size,vir_size,num_state,&w_c_at_freqs)
                    + fourier_self_energy::sigma_hermite(origin, omega, &hermite_coeff)
            };

            let (have_crossing,mut real_qp)=linear_interpolation_solver(qp_eq_func_with_hse,save_energies_no_se[n],side,qp_ctrl.gw_search_grid,qp_ctrl.gw_span_energy);
            println!("for n={}, second round (with HSE): qp energy={}",n,real_qp);
            real_qp
        }).collect::<Vec<f64>>().clone();
    }

    quasiparticle_energies_g=save_energies.clone();
    scf_data.gwqp.0=quasiparticle_energies_g.clone();
    display::full_quasiparticles(&quasiparticle_energies_g,occ_size);
    quasiparticle_energies_g
} 
pub fn vxc_ao2mo(scf_data:&SCF)->Vec<f64>{
    let eigenvecs=scf_data.eigenvectors[0].clone();
    let vxc_ao=scf_data.generate_vxc_rayon(1.0).2[0].to_matrixfull().unwrap().clone();
    let dimensions=vxc_ao.size[0];
    let mut vxc_nn=vec![0.0;dimensions];
    let mut element=0.0;
    eigenvecs.iter_columns_full().map(|vec|{
        let ev=vec.to_vec();
        let mut fv=vec![0.0;dimensions];
        _dgemv(&vxc_ao,&ev,&mut fv,'N', 1.0, 0.0, 1, 1);
        ev.iter().zip(fv.iter()).fold(0.0,|acc,(ev_i,fv_i)|acc+(ev_i*fv_i))
    }).collect()
    //No need to worry about cutoff
}
pub fn vxc_ao2mo_rayon(scf_data:&SCF)->MatrixFull<f64>{
    let eigenvecs=scf_data.eigenvectors.clone();
    let vxc_ao=scf_data.generate_vxc_rayon(1.0).2[0].to_matrixfull().unwrap().clone();
    let dimensions=vxc_ao.size[0];
    let mut vxc_mo:MatrixFull<f64>=MatrixFull::new([dimensions,dimensions],0.0);
    vxc_mo.data.par_iter_mut().enumerate().for_each(|(index, element)| {
        let i = index / dimensions;
        let j = index % dimensions;
        println!("now computing:{}-{}",i,j);
        // 计算element的值并赋值
        for mu in 0..dimensions{
            for nu in 0..dimensions{
                *element+=(eigenvecs[0][[mu,i]]*eigenvecs[0][[nu,j]]*vxc_ao[[mu,nu]]);
            }
        }
    });
    vxc_mo
    //Cutoff will help, but not necessary
}
pub fn v_mn_matrix_element(scf_data:&SCF,m:usize,n:usize,ri_mat:&MatrixFull<f64>)->f64{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'Y');
    let mut v_mn_matrix_element=0.0;
    for i in 0..(ri_mat.size[0]){
        v_mn_matrix_element+=ri_mat[[i as usize,m*num_state+n]].powf(2.0);
    }
    v_mn_matrix_element
}
pub fn v_matrix(
    scf_data: &SCF,
    ri_mat: &MatrixFull<f64>
) -> MatrixFull<f64> {
    let (start_mo, num_state, occ_size, vir_size, homo, lumo) = 
        get_occupation_parameters(scf_data, 'Y');
    
    let mut v_matrix = MatrixFull::new([num_state, num_state], 0.0);
    let n_rows = ri_mat.size[0];
    
    // 预计算列索引，减少重复计算
    let mut col_indices = Vec::with_capacity(num_state * num_state);
    for i in 0..num_state {
        for j in 0..num_state {
            col_indices.push(i * num_state + j);
        }
    }
    // 并行化外层循环
    v_matrix.data.par_chunks_mut(num_state)
        .enumerate()
        .for_each(|(i, row)| {
            for j in 0..num_state {
                let col_index = col_indices[i * num_state + j];
                if col_index < ri_mat.size[1] {
                    let mut sum = 0.0;
                    let start = col_index * n_rows;
                    
                    // 使用迭代器和fold
                    sum = ri_mat.data[start..start + n_rows]
                        .iter()
                        .fold(0.0, |acc, &x| acc + x * x);
                    
                    row[j] = sum;
                }
            }
        });
    
    v_matrix
}
pub fn v_matrix_symmetric_optimized(
    scf_data: &SCF,
    ri_mat: &MatrixFull<f64>
) -> MatrixFull<f64> {
    let (_, num_state, _, _, _, _) = get_occupation_parameters(scf_data, 'Y');
    let n_rows = ri_mat.size[0];
    
    // 创建结果矩阵
    let mut v_matrix = MatrixFull::new([num_state, num_state], 0.0);
    
    // 预先计算所有列的点积
    let column_sums: Vec<f64> = (0..num_state * num_state)
        .into_par_iter()
        .map(|col_idx| {
            let start = col_idx * n_rows;
            let end = start + n_rows;
            ri_mat.data[start..end]
                .iter()
                .fold(0.0, |acc, &x| acc + x * x)
        })
        .collect();
    
    // 将列和重塑为矩阵形式以便并行访问
    let column_sums_matrix: Vec<Vec<f64>> = column_sums
        .chunks(num_state)
        .map(|chunk| chunk.to_vec())
        .collect();
    
    // 使用行并行填充上三角
    v_matrix.data.par_chunks_mut(num_state)
        .enumerate()
        .for_each(|(i, row)| {
            // 获取当前行的列和
            let row_sums = &column_sums_matrix[i];
            
            // 填充当前行的上三角部分
            for j in i..num_state {
                row[j] = row_sums[j];
            }
        });
    
    // 串行填充下三角（对称性）
    for i in 0..num_state {
        for j in 0..i {
            v_matrix[[i, j]] = v_matrix[[j, i]];
        }
    }
    
    v_matrix
}
pub fn v_matrix_old(scf_data:&SCF,ri_mat:&MatrixFull<f64>)->MatrixFull<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'Y');
    let mut v_matrix:MatrixFull<f64>=MatrixFull::new([num_state,num_state],0.0);
    for i in 0..num_state{
        for j in 0..num_state{
            v_matrix[[i,j]]=v_mn_matrix_element(scf_data,i,j,&ri_mat);
        }
    }
    v_matrix
    //No need to worry about cutoff
}
pub fn generate_w_c(scf_data:&SCF,ri_ov:&MatrixFull<f64>,ri_full:&MatrixFull<f64>,quasiparticle_energies_g:&Vec<f64>,quasiparticle_energies_w:&Vec<f64>,num_state:usize,occ_size:usize,vir_size:usize,num_freq:usize)->Vec<(f64,f64,MatrixFull<f64>)>{
    omp_get_num_threads_wrapper();
    let freq_grid_type = scf_data.mol.ctrl.freq_grid_type;
    let max_freq = scf_data.mol.ctrl.freq_cut_off;
    let mut sp = format!("The frequency integration is tabulated by {:3} grids using", num_freq);
    let (mut omega_1,weight) = if freq_grid_type==0 {
        sp = format!("{} the modified Gauss-Legendre grids",sp);
        ri_rpa::trans_gauss_legendre_grids(1.0, num_freq)
    } else if freq_grid_type==1 {
        sp = format!("{} the standard Gauss-Legendre grids",sp);
        ri_rpa::gauss_legendre_grids([0.0,max_freq], num_freq)
    } else if freq_grid_type== 2 {
        sp = format!("{} the logarithmic grids",sp);
        ri_rpa::logarithmic_grid([0.0,max_freq], num_freq)
    } else {
        sp = format!("{} the modified Gauss-Legendre grids",sp);
        ri_rpa::trans_gauss_legendre_grids(1.0, num_freq)
    };
    if scf_data.mol.ctrl.print_level>1 {
        println!("{}", sp);
    }
    let (sender,receiver) = channel();
    rayon::prelude::IndexedParallelIterator::zip(omega_1.par_iter(), weight.par_iter()).for_each_with(sender, |s, (omega_1, weight)| {
        let start=Instant::now();
        let response=response_matrix(quasiparticle_energies_w,occ_size,vir_size,ri_ov,*omega_1,'I');
        let inverse_dielectric=inverse_dielectric_matrix(&response,'I');
        let w_c=w_c_matrix(&inverse_dielectric,num_state,ri_full);
        println!("Evaluation of W_c for omega={} has finished. This step took {:?}",omega_1,start.elapsed());
        s.send((*omega_1,*weight,w_c)).expect("unsuccessful collection of w_c")
    });
    let w_c_at_freqs:Vec<(f64,f64,MatrixFull<f64>)>=receiver.into_iter().collect();
    omp_get_num_threads_wrapper();
    w_c_at_freqs
}
pub fn generate_w_c_serial(scf_data:&SCF,ri_ov:&MatrixFull<f64>,ri_full:&MatrixFull<f64>,quasiparticle_energies_g:&Vec<f64>,quasiparticle_energies_w:&Vec<f64>,num_state:usize,occ_size:usize,vir_size:usize,num_freq:usize)->Vec<(f64,f64,MatrixFull<f64>)>{
    omp_get_num_threads_wrapper();
    let freq_grid_type = scf_data.mol.ctrl.freq_grid_type;
    let max_freq = scf_data.mol.ctrl.freq_cut_off;
    let mut sp = format!("The frequency integration is tabulated by {:3} grids using", num_freq);
    let (mut omega_1,weight) = if freq_grid_type==0 {
        sp = format!("{} the modified Gauss-Legendre grids",sp);
        ri_rpa::trans_gauss_legendre_grids(1.0, num_freq)
    } else if freq_grid_type==1 {
        sp = format!("{} the standard Gauss-Legendre grids",sp);
        ri_rpa::gauss_legendre_grids([0.0,max_freq], num_freq)
    } else if freq_grid_type== 2 {
        sp = format!("{} the logarithmic grids",sp);
        ri_rpa::logarithmic_grid([0.0,max_freq], num_freq)
    } else {
        sp = format!("{} the modified Gauss-Legendre grids",sp);
        ri_rpa::trans_gauss_legendre_grids(1.0, num_freq)
    };
    if scf_data.mol.ctrl.print_level>1 {
        println!("{}", sp);
    }
    omega_1.iter().zip(weight.iter()).map(|(omega_1, weight)| {
        let start=Instant::now();
        let response=response_matrix(quasiparticle_energies_w,occ_size,vir_size,ri_ov,*omega_1,'I');
        let inverse_dielectric=inverse_dielectric_matrix(&response,'I');
        let w_c=w_c_matrix(&inverse_dielectric,num_state,ri_full);
        println!("Evaluation of W_c for omega={} has finished. This step took {:?}",omega_1,start.elapsed());
        (*omega_1,*weight,w_c)
    }).collect()
}
pub fn calculate_imag(w_c_at_freqs:&Vec<(f64,f64,MatrixFull<f64>)>,num_state:usize,n:usize,omega:f64,quasiparticle_energies_g:&Vec<f64>,quasiparticle_energies_w:&Vec<f64>)->f64{
    omp_get_num_threads_wrapper();
    let (sender,receiver)=channel();
    //println!("0th freq:  calculating imag: w_c size:{},{}; quasiparticles_g size:{}",w_c_at_freqs[0].2.size[0],w_c_at_freqs[0].2.size[1],quasiparticle_energies_g.len());
    //println!("10th freq: calculating imag: w_c size:{},{}; quasiparticles_g size:{}",w_c_at_freqs[10].2.size[0],w_c_at_freqs[10].2.size[1],quasiparticle_energies_g.len());
    w_c_at_freqs.par_iter().for_each_with(sender, |s, (omega_1,weight,w_c)|{
        let imag_n_omega=(quasiparticle_energies_g.iter().enumerate().map(|(m,qp_m)|{
            let rpod=omega-qp_m;
            let gfc=rpod/(rpod.powf(2.0)+omega_1.powf(2.0));
            2.0*gfc*(w_c[[n,m]])/(2.0*PI)
        }).sum::<f64>())*weight;
        s.send(imag_n_omega).expect("unsuccessful collection of imag_n_omega")
    });
    let imag_n=receiver.into_iter().sum();
    omp_get_num_threads_wrapper();
    imag_n
    //No need to worry about cutoff
}
pub fn get_occupation_parameters(scf_data:&SCF,response_or_not:char)->(usize,usize,usize,usize,usize,usize){
    let cutoff=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap().bse_cutoff_energy;
    let mut num_state = scf_data.mol.num_state;
    let mut homo = 0_usize;
    let mut lumo = num_state;
    let start_mo = scf_data.mol.start_mo;
    for i_spin in 0..scf_data.mol.spin_channel {
        let i_homo = scf_data.homo.get(i_spin).unwrap().clone();
        let i_lumo = scf_data.lumo.get(i_spin).unwrap().clone();
        homo = homo.max(i_homo);
        lumo = lumo.min(i_lumo);
    }
    let occ_size=homo-start_mo+1;
    let mut vir_size=0;
    if response_or_not=='Y'{
        vir_size=num_state-occ_size;
    }
    
    else if response_or_not=='N'{
        num_state=scf_data.eigenvalues[0].clone().iter().filter(|x|**x<cutoff).count();
        vir_size=num_state-occ_size;
    }
    (start_mo,num_state,occ_size,vir_size,homo,lumo)
}
pub fn display_and_save_quasiparticles(scf_data:&mut SCF,quasiparticle_energies_g:&Vec<f64>,round:usize){
    for i in 0..quasiparticle_energies_g.len(){
        println!("quasiparticles:{:#?}",quasiparticle_energies_g[i]);
        //scf_data.eigenvalues[0][i]=quasiparticle_energies[i];
    }
}
pub fn response_matrix_legacy(quasiparticle_energies_w:&Vec<f64>,occ_size:usize,vir_size:usize,ri_ov:&MatrixFull<f64>,omega:f64,part:char)->MatrixFull<f64>{
    let mut diag=vec![0.0;occ_size*vir_size];
    let num_auxbas=ri_ov.size[1];
    let mut response=MatrixFull::new([num_auxbas,0],0.0);
    (0..occ_size*vir_size).for_each(|n|{
        let i=n%occ_size;
        let a=occ_size+n/occ_size;
        let energy_gap=quasiparticle_energies_w[a]-quasiparticle_energies_w[i];
        diag[n]=if part=='I'{-2.0*energy_gap/(energy_gap.powf(2.0)+omega.powf(2.0))}
            else{-2.0*energy_gap/(energy_gap.powf(2.0)-omega.powf(2.0))};
    });
    let mut chi_as_vecs:Vec<(usize,Vec<f64>)>=ri_ov.iter_columns_full().enumerate().par_bridge().map(|(p,ri_p)|{
        let mut work_vec=vec![0.0;occ_size*vir_size];
        (0..occ_size*vir_size).for_each(|ia|work_vec[ia]=ri_p[ia]*diag[ia]);
        let mut chi_p=vec![0.0;num_auxbas];
        _dgemv(ri_ov,&work_vec,&mut chi_p,'T', 1.0, 0.0, 1, 1);
        (p,chi_p.clone())
    }).collect();
    chi_as_vecs.sort_by_key(|(i, _)| *i);
    for chi_p in chi_as_vecs {
        response.push_column(&chi_p.1);
    }
    response.self_multiple(2.0);
    //println!("a response has been collected, its size is:{},{}",response.size[0],response.size[1]);
    response
}
pub fn response_matrix(quasiparticle_energies_w:&Vec<f64>,occ_size:usize,vir_size:usize,ri_ov:&MatrixFull<f64>,omega:f64,part:char)->MatrixFull<f64>{
    let num_auxbas=ri_ov.size[0];
    let mut ri_to_be_processed=MatrixFull::new([num_auxbas,0],0.0);
    let mut ri_as_vecs:Vec<(usize,Vec<f64>)>=ri_ov.iter_columns_full().enumerate().par_bridge().map(|(n,ri_n)|{
        let i=n%occ_size;
        let a=occ_size+n/occ_size;
        let energy_gap=quasiparticle_energies_w[a]-quasiparticle_energies_w[i];
        let mut multiply_number=if part=='I'{-2.0*energy_gap/(energy_gap.powf(2.0)+omega.powf(2.0))}
            else{-2.0*energy_gap/(energy_gap.powf(2.0)-omega.powf(2.0))};
        let mut ri_n_vec=ri_n.to_vec();
        ri_n_vec.par_iter_mut().for_each(|x| *x *= multiply_number);
        (n,ri_n_vec)
    }).collect();
    ri_as_vecs.sort_by_key(|(i, _)| *i);
    for ri_vec in ri_as_vecs {
        ri_to_be_processed.push_column(&ri_vec.1);
    }
    let mut response:MatrixFull<f64>=MatrixFull::new([num_auxbas,num_auxbas],0.0);
    _dgemm_full(ri_ov,'N',&ri_to_be_processed,'T',&mut response,1.0,0.0); 
    response.self_multiple(2.0);
    //println!("a response has been collected, its size is:{},{}",response.size[0],response.size[1]);
    response
}
pub fn inverse_dielectric_matrix(response:&MatrixFull<f64>,part:char)->MatrixFull<f64>{
    let mut dielectric:MatrixFull<f64>=response.clone();
    dielectric.self_multiple(-1.0);
    let num_auxbas=response.size[0];
    dielectric+=ri_bse::identity_matrix(num_auxbas);
    let mut inverse_dielectric=_dinverse(&dielectric).expect("unsuccessful _dinverse");
    if part=='C'||part=='I'{
        inverse_dielectric-=ri_bse::identity_matrix(num_auxbas);
    }
    inverse_dielectric
}
pub fn w_c_matrix(inverse_dielectric:&MatrixFull<f64>,num_state:usize,ri_full:&MatrixFull<f64>)->MatrixFull<f64>{
    let iterator=ri_full.iter_columns((0..(num_state*num_state))).enumerate();
    let num_auxbas=inverse_dielectric.size[0];
    let mut w_c=MatrixFull::new([num_state,num_state],0.0);
    let mut first_product=vec![0.0;num_auxbas];
    let mut check_matrix=MatrixFull::new([num_state,num_state],0.0);
    for (a,vec) in iterator{
        let n=a/num_state;
        let m=a%num_state;
        if check_matrix[[m,n]]<0.5{
            _dgemv(inverse_dielectric,vec,&mut first_product,'N', 1.0, 0.0, 1, 1);
            w_c[[m,n]]=first_product.iter().zip(vec.iter()).map(|(a,b)|a*b).sum();
            w_c[[n,m]]=w_c[[m,n]];
            check_matrix[[m,n]]=1.0;
            check_matrix[[n,m]]=1.0;
        }
    }
    w_c
}
pub fn contour_rayon(omega:f64,n:usize,quasiparticle_energies_g:&Vec<f64>,quasiparticle_energies_w:&Vec<f64>,occ_size:usize,vir_size:usize,num_state:usize,ri_ov:&MatrixFull<f64>,ri_full:&MatrixFull<f64>)->f64{
    let fermi_energy=(quasiparticle_energies_g[occ_size-1]+quasiparticle_energies_g[occ_size])/2.0;
    let sign=if omega>fermi_energy{1}else{-1};
    let num_auxbas=ri_ov.size[0];
    if sign==1{
        (0..vir_size).into_par_iter().map(|a|{
            let mut residue=0.0;
            if quasiparticle_energies_g[occ_size+a]<omega{
                let gap=omega-quasiparticle_energies_g[occ_size+a];
                let response=response_matrix(quasiparticle_energies_w,occ_size,vir_size,ri_ov,gap,'C');
                let inverse_dielectric=inverse_dielectric_matrix(&response,'C');
                let vec:Vec<f64>=ri_full.iter_column(n+(occ_size+a)*num_state).copied().collect::<Vec<f64>>();
                let mut first_product=vec![0.0;num_auxbas];
                _dgemv(&inverse_dielectric, &vec, &mut first_product, 'N', 1.0, 0.0, 1, 1);
                residue=first_product.iter().zip(vec.iter()).map(|(a,b)|a*b).sum::<f64>();
            }
            residue*=(sign as f64);
            residue
        }).sum()
    }else{
        (0..occ_size).into_par_iter().map(|i|{
            let mut residue=0.0;
            if quasiparticle_energies_g[i]>omega{
                let gap=quasiparticle_energies_g[i]-omega;
                let response=response_matrix(quasiparticle_energies_w,occ_size,vir_size,ri_ov,gap,'C');
                let inverse_dielectric=inverse_dielectric_matrix(&response,'C');
                let vec:Vec<f64>=ri_full.iter_column(n+i*num_state).copied().collect::<Vec<f64>>();
                let mut first_product=vec![0.0;num_auxbas];
                _dgemv(&inverse_dielectric, &vec, &mut first_product, 'N', 1.0, 0.0, 1, 1);
                residue=first_product.iter().zip(vec.iter()).map(|(a,b)|a*b).sum::<f64>();
            }
            residue*=(sign as f64);
            residue
        }).sum()
    }
}
pub fn newton_solver<F>(mut f:F,n:usize,consts:f64,ri_ov:&MatrixFull<f64>,ri_full:&MatrixFull<f64>,quasiparticle_energies_g:&Vec<f64>,quasiparticle_energies_w:&Vec<f64>,occ_size:usize,vir_size:usize,num_state:usize,w_c_at_freqs:&Vec<(f64,f64,MatrixFull<f64>)>,starting_point:f64,tol:f64,max_iter:usize,side:f64,printlevel:usize)->f64 where F:Fn(f64,usize,f64,&MatrixFull<f64>,&MatrixFull<f64>,&Vec<f64>,&Vec<f64>,usize,usize,usize,&Vec<(f64,f64,MatrixFull<f64>)>)->f64,{
    let h =0.000001;
    let delta=0.02;
    let mut x_curr=starting_point+side*delta;
    let mut y_curr=f(x_curr,n,consts,ri_ov,ri_full,quasiparticle_energies_g,quasiparticle_energies_w,occ_size,vir_size,num_state,w_c_at_freqs);
    let mut y_plus=f(x_curr+h,n,consts,ri_ov,ri_full,quasiparticle_energies_g,quasiparticle_energies_w,occ_size,vir_size,num_state,w_c_at_freqs);
    let mut y_minus=f(x_curr-h,n,consts,ri_ov,ri_full,quasiparticle_energies_g,quasiparticle_energies_w,occ_size,vir_size,num_state,w_c_at_freqs);
    let mut converge=0;
    let mut iter_times=0;
    loop{
        let derivative=(y_plus-y_minus)/(2.0*h);
        let shift=-y_curr/derivative;
        if printlevel>1{
            println!("newton now:n={},x_curr={},y_curr={},derivative={},shift={}",n,x_curr,y_curr,derivative,shift);
        }
        x_curr=x_curr+shift;
        y_curr=f(x_curr,n,consts,ri_ov,ri_full,quasiparticle_energies_g,quasiparticle_energies_w,occ_size,vir_size,num_state,w_c_at_freqs);
        if shift.abs()<tol{
            converge+=1;
            if printlevel>1{
                println!("convergence: x_curr={},shift={}, y_curr={}",x_curr,shift,y_curr);
            }
        }
        y_plus=f(x_curr+h,n,consts,ri_ov,ri_full,quasiparticle_energies_g,quasiparticle_energies_w,occ_size,vir_size,num_state,w_c_at_freqs);
        y_minus=f(x_curr-h,n,consts,ri_ov,ri_full,quasiparticle_energies_g,quasiparticle_energies_w,occ_size,vir_size,num_state,w_c_at_freqs);
        iter_times+=1;
        if converge==1 || iter_times==max_iter{
            break
        }
    }
    if iter_times==max_iter{
        println!("warning!!! newton solver did not converge for orbital {}!",n);
    }
    x_curr
}
pub fn single_newton_step<F>(mut f:F,starting_point:f64,side:f64,span_energy:f64)->f64 where F:Fn(f64)->f64{
    let h=0.000001;
    let delta=0.02;
    let mut x_curr=starting_point+side*delta;
    let y_curr=f(x_curr);
    let y_minus=f(x_curr-h);
    let y_plus=f(x_curr+h);
    let derivative=(y_plus-y_minus)/(2.0*h);
    let shift=-y_curr/derivative;
    x_curr+shift
}
pub fn linear_interpolation_solver<F>(mut f:F,starting_point:f64,side:f64,grid_freqs:usize,span_energy:f64)->(bool,f64) where F:Fn(f64)->f64{
    let h=0.00000001;
    let delta=0.02;
    let mut x_curr=starting_point+side*delta;
    let y_curr=f(x_curr);
    let y_minus=f(x_curr-h);
    let y_plus=f(x_curr+h);
    let derivative=(y_plus-y_minus)/(2.0*h);
    let shift=-y_curr/derivative;
    x_curr=x_curr+shift;
    let energy_step=span_energy*2.0/((grid_freqs-1)as f64);
    let grid_results:Vec<(f64,f64)>=(0..grid_freqs).map(|i|{
            let x=x_curr-span_energy+((i as f64)*energy_step);
            let fx=f(x);
            //println!("omega={}, qp_eq={}",x,fx);
            (x,fx)
    }).collect();
    let mut xing=vec![0.0;0];
    let mut answer=0.0;
    for i in (0..grid_freqs-1){
        if grid_results[i].1*grid_results[i+1].1<0.0{
            let slope=(grid_results[i+1].1-grid_results[i].1)/(grid_results[i+1].0-grid_results[i].0);
            xing.push(grid_results[i].0-(grid_results[i].1/slope));
        }
    }
    let mut have_crossing=true;
    // Always compute fallback (static Sigma_c approximation at starting point)
    let fallback = starting_point + f(starting_point);
    //println!("---------------\nfound {} crossings",xing.len());
    if xing.len()>1{
        let mut spectral_weights:Vec<(f64,f64)>=xing.iter().map(|e|{
            let y_minus=f(e-h);
            let y_plus=f(e+h);
            let derivative=(y_plus-y_minus)/(2.0*h)+1.0;
            let spectral_weight=(1.0-derivative).powf(-1.0);
            println!("crossing at {}, spectral value={}",e,spectral_weight);
            (spectral_weight,*e)
        }).collect();
        spectral_weights.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        answer=spectral_weights[0].1;
        println!("  Graphical solution: E_qp = {:.6} Ha,  static-coupling fallback: E_qp = {:.6} Ha", answer, fallback);
    }else if xing.len()==1{
        answer=xing[0];
        let y_minus=f(answer-h);
        let y_plus=f(answer+h);
        let derivative=(y_plus-y_minus)/(2.0*h)+1.0;
        let spectral_weight=(1.0-derivative).powf(-1.0);
        println!("crossing at {}, derivative={}, spectral value={}",answer,derivative,spectral_weight);
        println!("  Graphical solution: E_qp = {:.6} Ha,  static-coupling fallback: E_qp = {:.6} Ha", answer, fallback);
    }else{
        have_crossing=false;
        // Fallback: use static approximation Sigma_c(E0) at the starting point.
        // Matches MOLGW's behavior in find_qp_energy_graphical (line 331-334):
        //   E_qp = E0 + Sigma_c(E0) + (Sigma_x - Vxc)
        //   = starting_point + f(starting_point)
        println!("  No graphical solution in scan range. Using static-coupling fallback: E_qp = {:.6} Ha", fallback);
        answer = fallback;
    }
    (have_crossing,answer)
}
pub fn quasiparticle_equation(omega:f64,n:usize,consts:f64,ri_ov:&MatrixFull<f64>,ri_full:&MatrixFull<f64>,quasiparticle_energies_g:&Vec<f64>,quasiparticle_energies_w:&Vec<f64>,occ_size:usize,vir_size:usize,num_state:usize,w_c_at_freqs:&Vec<(f64,f64,MatrixFull<f64>)>)->f64{
    let contour=contour_rayon(omega,n,quasiparticle_energies_g,quasiparticle_energies_w,occ_size,vir_size,num_state,ri_ov,ri_full);
    let imag=calculate_imag(&w_c_at_freqs,num_state,n,omega,quasiparticle_energies_g,quasiparticle_energies_w);
    //println!("quasiparticle equation residue now:={}",consts+contour-imag-omega);
    consts+contour-imag-omega
}

pub fn spectrum_test(scf_data:&SCF,num_freq:usize){
    let eigenenergies:Vec<f64>=scf_data.eigenvalues[0].clone();
    let mut quasiparticle_energies:Vec<f64>=Vec::new();
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'Y');
    for n in 0..num_state{
        quasiparticle_energies.push(eigenenergies[n]);
    }
    let mut ri_full:MatrixFull<f64>=ri_bse::get_submatrix(scf_data,'F','F','Y');
    let mut ri_ov:MatrixFull<f64>=ri_bse::get_submatrix(scf_data,'O','V','Y');
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let start_freq:f64=qp_ctrl.spectrum_test_start;
    let end_freq:f64=qp_ctrl.spectrum_test_end;
    let step:f64=qp_ctrl.spectrum_test_step;
    let steps:usize=((end_freq-start_freq)/step).ceil() as usize;
    let w_c_at_freqs=generate_w_c(scf_data,&ri_ov,&ri_full,&quasiparticle_energies,&quasiparticle_energies,num_state,occ_size,vir_size,num_freq);
    for n in (homo..homo+2){
        println!("Now is the spectrum of orbital #{}",n);
        (0..steps+1).into_par_iter().for_each(|w|{
            let freq=start_freq+(w as f64)*step;
            let contour=contour_rayon(freq,n,&quasiparticle_energies,&quasiparticle_energies,occ_size,vir_size,num_state,&ri_ov,&ri_full);
            let imag=calculate_imag(&w_c_at_freqs,num_state,n,freq,&quasiparticle_energies,&quasiparticle_energies);
            let sigma=contour-imag;
            println!("{},{}",freq,sigma);
        });
    }
}
pub fn x_alpha_gw(scf_data:&mut SCF)->Vec<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'Y');
    let ks_energies:Vec<f64>=scf_data.eigenvalues[0].clone();
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let xc_name = scf_data.mol.ctrl.xc.to_lowercase();
    let exchange_under_dfa=get_pure_x_or_c_of_xc(scf_data,&xc_name,'X');
    let x_alpha=qp_ctrl.x_alpha;
    let ri_mat:MatrixFull<f64>=ri_bse::get_submatrix(scf_data,'F','F','Y');
    let v_matrix=v_matrix(&scf_data,&ri_mat);
    ks_energies.into_iter().enumerate().map(|(n,e_n)|{
        let mut exchange=0.0;
        for i in 0..homo+1{
            exchange-=v_matrix[[n,i]];
        }
        e_n+x_alpha*(exchange-exchange_under_dfa[n])
    }).collect()
}
pub fn linearized_gw(scf_data:&mut SCF,num_freq:usize,vxc_nn:&Vec<f64>,cancel_dfa_xc:bool)->Vec<f64>{
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let delta=qp_ctrl.gw_linearize_shift;
    let ri_mat:MatrixFull<f64>=ri_bse::get_submatrix(scf_data,'F','F','Y');
    let v_matrix=v_matrix(&scf_data,&ri_mat);
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'Y');
    let mut num_state_qp_range=num_state;
    if qp_ctrl.scgw=="g0w0"{
        let (start_mo_n,num_state_n,occ_size_n,vir_size_n,homo_n,lumo_n)=get_occupation_parameters(scf_data,'N');
        num_state_qp_range=num_state_n;
    }
    let eigenenergies:Vec<f64>=scf_data.eigenvalues[0].clone();
    let ri_mat:MatrixFull<f64>=ri_bse::get_submatrix(scf_data,'F','F','Y');
    let mut quasiparticle_energies_g:Vec<f64>=scf_data.gwqp.0.clone();
    let mut quasiparticle_energies_w:Vec<f64>=scf_data.gwqp.1.clone();
    let mut ri_ov:MatrixFull<f64>=ri_bse::get_submatrix(scf_data,'O','V','Y');

    // Route to low-rank accelerated version if enabled
    if qp_ctrl.use_low_rank_contour {
        println!("Low-rank contour deformation acceleration ENABLED (linearized GW).");
        println!("  real-axis grid points: {}", qp_ctrl.nomega_chi_real);
        println!("  eigenvalue tolerance:  {}", qp_ctrl.low_rank_tolerance);
        return linearized_gw_lowrank(
            scf_data, num_freq, vxc_nn, cancel_dfa_xc,
            qp_ctrl.nomega_chi_real, qp_ctrl.low_rank_tolerance,
        );
    }

    //display_and_save_quasiparticles(scf_data,&quasiparticle_energies_g,0);
    let w_c_at_freqs=generate_w_c(scf_data,&ri_ov,&ri_mat,&quasiparticle_energies_g,&quasiparticle_energies_w,num_state,occ_size,vir_size,num_freq);
    let h=qp_ctrl.gw_linearize_derivative_h;

    // 检查是否启用自能校正
    let use_fourier = qp_ctrl.fourier_self_energy;
    let use_hermite = qp_ctrl.hermite_self_energy;

    if use_fourier && use_hermite {
        panic!("Cannot enable both Fourier self-energy and Hermite self-energy simultaneously!");
    }

    // 第一轮：正常线性化GW计算（不添加自能校正）
    let mut save_energies_no_se:Vec<f64>=vec![0.0;num_state];
    save_energies_no_se=(0..num_state_qp_range).map(|n|{
        let side=if n>=occ_size{1.0}else{-1.0};
        let omega_shifted=scf_data.eigenvalues[0][n]-delta*side;
        let contour=contour_rayon(omega_shifted,n,&quasiparticle_energies_g,&quasiparticle_energies_w,occ_size,vir_size,num_state,&ri_ov,&ri_mat);
        let imag_n=calculate_imag(&w_c_at_freqs,num_state,n,omega_shifted,&quasiparticle_energies_g,&quasiparticle_energies_w);

        let contour_plus_h=contour_rayon(omega_shifted+h,n,&quasiparticle_energies_g,&quasiparticle_energies_w,occ_size,vir_size,num_state,&ri_ov,&ri_mat);
        let contour_minus_h=contour_rayon(omega_shifted-h,n,&quasiparticle_energies_g,&quasiparticle_energies_w,occ_size,vir_size,num_state,&ri_ov,&ri_mat);
        let imag_plus_h=calculate_imag(&w_c_at_freqs,num_state,n,omega_shifted+h,&quasiparticle_energies_g,&quasiparticle_energies_w);
        let imag_minus_h=calculate_imag(&w_c_at_freqs,num_state,n,omega_shifted-h,&quasiparticle_energies_g,&quasiparticle_energies_w);
        let self_energy_plus_h=contour_plus_h-imag_plus_h;
        let self_energy_minus_h=contour_minus_h-imag_minus_h;
        let derivative=(self_energy_plus_h-self_energy_minus_h)/(2.0*h);
        if scf_data.mol.ctrl.print_level>1{
            println!("derivative at n={}: {}",n,derivative);
        }
        let spectral_weight=(1.0-derivative).powf(-1.0);
        println!("spectral weight at n={}: {}",n,spectral_weight);
        let real_qp=if cancel_dfa_xc==true{
            let mut exchange=0.0;
            for i in 0..homo+1{
                exchange-=v_matrix[[n,i]];
            }
            omega_shifted+(spectral_weight*(contour-imag_n+exchange-vxc_nn[n]))
        }else{omega_shifted+(spectral_weight*(contour-imag_n))};
        println!("for n={}, first round (no self-energy correction): linearized gw yields qp energy={}",n,real_qp);
        real_qp
    }).collect::<Vec<f64>>().clone();

    // 如果不需要自能校正，直接返回第一轮结果
    if !use_fourier && !use_hermite {
        quasiparticle_energies_g=save_energies_no_se.clone();
        display::full_quasiparticles(&quasiparticle_energies_g,occ_size);
        return quasiparticle_energies_g;
    }

    // 第二轮：添加自能校正
    let mut save_energies:Vec<f64>=vec![0.0;num_state];

    if use_fourier {
        let (powers, t, sin_coeff, cos_coeff) = fourier_self_energy::define_fourier_series(&qp_ctrl);
        println!("Fourier Self Energy enabled for second round (linearized GW)");

        save_energies=(0..num_state_qp_range).map(|n|{
            let origin = save_energies_no_se[n];
            let side=if n>=occ_size{1.0}else{-1.0};
            let omega_shifted=scf_data.eigenvalues[0][n]-delta*side;

            // 计算带FSE的自能及其导数
            let sigma_with_fse = |omega: f64| -> f64 {
                let contour=contour_rayon(omega,n,&quasiparticle_energies_g,&quasiparticle_energies_w,occ_size,vir_size,num_state,&ri_ov,&ri_mat);
                let imag=calculate_imag(&w_c_at_freqs,num_state,n,omega,&quasiparticle_energies_g,&quasiparticle_energies_w);
                let fse=fourier_self_energy::fourier_series(&sin_coeff, &cos_coeff, powers, t, omega - origin);
                contour - imag + fse
            };

            let self_energy_at_omega=sigma_with_fse(omega_shifted);
            let self_energy_plus_h=sigma_with_fse(omega_shifted+h);
            let self_energy_minus_h=sigma_with_fse(omega_shifted-h);
            let derivative=(self_energy_plus_h-self_energy_minus_h)/(2.0*h);

            if scf_data.mol.ctrl.print_level>1{
                println!("derivative at n={} (with FSE): {}",n,derivative);
            }
            let spectral_weight=(1.0-derivative).powf(-1.0);
            println!("spectral weight at n={} (with FSE): {}",n,spectral_weight);

            let real_qp=if cancel_dfa_xc==true{
                let mut exchange=0.0;
                for i in 0..homo+1{
                    exchange-=v_matrix[[n,i]];
                }
                omega_shifted+(spectral_weight*(self_energy_at_omega+exchange-vxc_nn[n]))
            }else{omega_shifted+(spectral_weight*self_energy_at_omega)};
            println!("for n={}, second round (with FSE): linearized gw yields qp energy={}",n,real_qp);
            real_qp
        }).collect::<Vec<f64>>().clone();
    } else if use_hermite {
        let hermite_coeff = fourier_self_energy::define_hermite_series(&qp_ctrl);
        println!("Hermite Self Energy enabled for second round (linearized GW)");

        save_energies=(0..num_state_qp_range).map(|n|{
            let origin = save_energies_no_se[n];
            let side=if n>=occ_size{1.0}else{-1.0};
            let omega_shifted=scf_data.eigenvalues[0][n]-delta*side;

            // 计算带HSE的自能及其导数
            let sigma_with_hse = |omega: f64| -> f64 {
                let contour=contour_rayon(omega,n,&quasiparticle_energies_g,&quasiparticle_energies_w,occ_size,vir_size,num_state,&ri_ov,&ri_mat);
                let imag=calculate_imag(&w_c_at_freqs,num_state,n,omega,&quasiparticle_energies_g,&quasiparticle_energies_w);
                let hse=fourier_self_energy::sigma_hermite(origin, omega, &hermite_coeff);
                contour - imag + hse
            };

            let self_energy_at_omega=sigma_with_hse(omega_shifted);
            let self_energy_plus_h=sigma_with_hse(omega_shifted+h);
            let self_energy_minus_h=sigma_with_hse(omega_shifted-h);
            let derivative=(self_energy_plus_h-self_energy_minus_h)/(2.0*h);

            if scf_data.mol.ctrl.print_level>1{
                println!("derivative at n={} (with HSE): {}",n,derivative);
            }
            let spectral_weight=(1.0-derivative).powf(-1.0);
            println!("spectral weight at n={} (with HSE): {}",n,spectral_weight);

            let real_qp=if cancel_dfa_xc==true{
                let mut exchange=0.0;
                for i in 0..homo+1{
                    exchange-=v_matrix[[n,i]];
                }
                omega_shifted+(spectral_weight*(self_energy_at_omega+exchange-vxc_nn[n]))
            }else{omega_shifted+(spectral_weight*self_energy_at_omega)};
            println!("for n={}, second round (with HSE): linearized gw yields qp energy={}",n,real_qp);
            real_qp
        }).collect::<Vec<f64>>().clone();
    }

    quasiparticle_energies_g=save_energies.clone();
    display::full_quasiparticles(&quasiparticle_energies_g,occ_size);
    quasiparticle_energies_g
}
pub fn get_pure_x_or_c_of_xc(scf_data:&mut SCF,name:&str,choice:char)->Vec<f64>{
    let eigenvecs=scf_data.eigenvectors.clone();
    let x_or_c=if choice=='X'{1}else if choice=='C'{2}else{panic!("invalid choice for pure x/c!")};
    let dfa_compnt_save=scf_data.mol.xc_data.dfa_compnt_scf.clone();
    let dfa_paramr_save=scf_data.mol.xc_data.dfa_paramr_scf.clone();
    let mut new_paramr=vec![0.0,0.0,0.0];
    new_paramr[x_or_c]=1.0;
    scf_data.mol.xc_data.dfa_compnt_scf=DFA4REST::libxc_code_fdqc(name).to_vec();
    scf_data.mol.xc_data.dfa_paramr_scf=new_paramr;
    let v_x_or_c_ao=scf_data.generate_vxc_rayon(1.0).2[0].to_matrixfull().unwrap().clone();
    let dimensions=v_x_or_c_ao.size[0];
    let mut v_x_or_c_nn=vec![0.0;dimensions];
    let mut element=0.0;
    for i in (0..dimensions){
        element=0.0;
        for mu in 0..dimensions{
            for nu in 0..dimensions{
                element+=(eigenvecs[0][[mu,i]]*eigenvecs[0][[nu,i]]*v_x_or_c_ao[[mu,nu]]);
            }
        }
        v_x_or_c_nn[i]=element;
    }
    scf_data.mol.xc_data.dfa_compnt_scf=dfa_compnt_save;
    scf_data.mol.xc_data.dfa_paramr_scf=dfa_paramr_save;
    v_x_or_c_nn
}
pub fn get_homo_vx_or_vc(scf_data:&mut SCF,name:&str,choice:char)->f64{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'Y');
    let eigenvecs=scf_data.eigenvectors.clone();
    let x_or_c=if choice=='X'{1}else if choice=='C'{2}else{panic!("invalid choice for pure x/c!")};
    //println!("About to save DFA4REST");
    let dfa_compnt_save=scf_data.mol.xc_data.dfa_compnt_scf.clone();
    let dfa_paramr_save=scf_data.mol.xc_data.dfa_paramr_scf.clone();
    //println!("About to obtain xc_code and change DFA4REST");
    scf_data.mol.xc_data.dfa_compnt_scf=vec![DFA4REST::libxc_code_fdqc(name)[x_or_c]];
    //println!("xc_code has been obtained");
    scf_data.mol.xc_data.dfa_paramr_scf=vec![1.0];
    //println!("About to generate vxc");
    let v_x_or_c_ao=scf_data.generate_vxc_rayon(1.0).2[0].to_matrixfull().unwrap().clone();
    //println!("vxc has been generated");
    let dimensions=v_x_or_c_ao.size[0];
    let mut element=0.0;
    for mu in 0..dimensions{
        for nu in 0..dimensions{
            element+=(eigenvecs[0][[mu,homo]]*eigenvecs[0][[nu,homo]]*v_x_or_c_ao[[mu,nu]]);
        }
    }
    //println!("About to recover DFA4REST");
    scf_data.mol.xc_data.dfa_compnt_scf=dfa_compnt_save;
    scf_data.mol.xc_data.dfa_paramr_scf=dfa_paramr_save;
    //println!("DFA4REST has been recovered");
    element
}
pub fn get_homo_lumo_qp_only(scf_data:&mut SCF,num_freq:usize,vxc_nn:&Vec<f64>,mpi_operator:&Option<MPIOperator>){
    let printlevel=scf_data.mol.ctrl.print_level.clone();
    let ri_mat:MatrixFull<f64>=ri_bse::get_submatrix(scf_data,'F','F','Y');
    let v_matrix=v_matrix(&scf_data,&ri_mat);
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'Y');
    let mut rs_particles:Vec<f64>=Vec::new();
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let renormalized_singles=qp_ctrl.renormalized_singles;
    if renormalized_singles==true{
        println!("Starts renormalized singles calculations!");
        let w_rs=qp_ctrl.w_rs;
        //println!("Now starting to compute rs_particles");
        rs_particles=renormalized_singles::renormalized_singles_diagonalization(scf_data,w_rs,mpi_operator);
        if printlevel>0{
            println!("rs_particles:{:?}",rs_particles);
        }
        scf_data.renormalized_singles_particles=rs_particles
    }
    let eigenenergies:Vec<f64>=scf_data.eigenvalues[0].clone();
    let mut quasiparticle_energies_g:Vec<f64>=Vec::new();
    let mut quasiparticle_energies_w:Vec<f64>=Vec::new();
    if scf_data.mol.ctrl.print_level>2{
        println!("ri_mat full:");
        ri_mat.formated_output(1000,"full");
    }
    
    if qp_ctrl.renormalized_singles==true{
        quasiparticle_energies_g=scf_data.renormalized_singles_particles.clone();
        if qp_ctrl.w_rs==true{
            quasiparticle_energies_w=scf_data.renormalized_singles_particles.clone();
        }else{
            for n in 0..num_state{
                quasiparticle_energies_w.push(eigenenergies[n]);
            }
        }
    }else{
        for n in 0..num_state{
            quasiparticle_energies_g.push(eigenenergies[n]);
            quasiparticle_energies_w.push(eigenenergies[n]);
        }
    }
    let mut ri_ov:MatrixFull<f64>=ri_bse::get_submatrix(scf_data,'O','V','Y');

    // Route to low-rank accelerated version if enabled
    if qp_ctrl.use_low_rank_contour {
        println!("Low-rank contour deformation acceleration ENABLED (homo_lumo_qp).");
        println!("  real-axis grid points: {}", qp_ctrl.nomega_chi_real);
        println!("  eigenvalue tolerance:  {}", qp_ctrl.low_rank_tolerance);
        return get_homo_lumo_qp_only_lowrank(
            scf_data, num_freq, vxc_nn,
            qp_ctrl.nomega_chi_real, qp_ctrl.low_rank_tolerance,
        );
    }

    let w_c_at_freqs=generate_w_c(scf_data,&ri_ov,&ri_mat,&quasiparticle_energies_g,&quasiparticle_energies_w,num_state,occ_size,vir_size,num_freq);
    let n=homo;
    let mut exchange=0.0;
    for i in 0..homo+1{
        exchange-=v_matrix[[n,i]];
    }
    let consts=eigenenergies[n]+exchange-vxc_nn[n];
    if scf_data.mol.ctrl.print_level>1{
        println!("for n={},exchange={}",n,exchange);
    }
    let side=if n>=occ_size{1.0}else{-1.0};
    let printlevel=scf_data.mol.ctrl.print_level;
    let homo_qp=newton_solver(quasiparticle_equation,n,consts,&ri_ov,&ri_mat,&quasiparticle_energies_g,&quasiparticle_energies_w,occ_size,vir_size,num_state,&w_c_at_freqs,eigenenergies[n],0.00001,50,side,printlevel);
    //println!("for n={}, quasiparticle equation yields:qp energy={}",n,homo_qp);
    let save_path=qp_ctrl.save_qp_path.clone();
    println!("The QP energy of HOMO obtained by GWA is {}",homo_qp);
    let n=lumo;
    let mut exchange=0.0;
    for i in 0..homo+1{
        exchange-=v_matrix[[n,i]];
    }
    let consts=eigenenergies[n]+exchange-vxc_nn[n];
    if scf_data.mol.ctrl.print_level>1{
        println!("for n={},exchange={}",n,exchange);   
    }
    let side=if n>=occ_size{1.0}else{-1.0};
    let printlevel=scf_data.mol.ctrl.print_level;
    let lumo_qp=newton_solver(quasiparticle_equation,n,consts,&ri_ov,&ri_mat,&quasiparticle_energies_g,&quasiparticle_energies_w,occ_size,vir_size,num_state,&w_c_at_freqs,eigenenergies[n],0.00001,50,side,printlevel);
    let save_path=qp_ctrl.save_qp_path.clone();
    if qp_ctrl.save_gw_homo_lumo_qp==true{
        let mut file = OpenOptions::new().append(true).create(true).open(save_path);
        writeln!(file.expect("write failure"), "{},{}",homo_qp,lumo_qp);
    }
    println!("The QP energy of LUMO obtained by GWA is {}",lumo_qp);
}
pub fn obtain_vx_vc_terms(scf_data:&mut SCF){
    println!("starts obtaining HOMO vx and vc terms from various DFAs");
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'Y');
    let mut file = OpenOptions::new().append(true).create(true).open("ks_homo.txt");
    writeln!(file.expect("write failure"), "{}", scf_data.eigenvalues[0][occ_size-1]);
    let mut file = OpenOptions::new().append(true).create(true).open("lda_x_slater.txt");
    writeln!(file.expect("write failure"), "{}", get_homo_vx_or_vc(scf_data,"lda_x_slater",'X'));
    let mut file = OpenOptions::new().append(true).create(true).open("gga_x_b88.txt");
    writeln!(file.expect("write failure"), "{}", get_homo_vx_or_vc(scf_data,"gga_x_b88",'X'));
    let mut file = OpenOptions::new().append(true).create(true).open("gga_x_pbe.txt");
    writeln!(file.expect("write failure"), "{}", get_homo_vx_or_vc(scf_data,"gga_x_pbe",'X'));
    let mut file = OpenOptions::new().append(true).create(true).open("gga_x_xpbe.txt");
    writeln!(file.expect("write failure"), "{}", get_homo_vx_or_vc(scf_data,"gga_x_xpbe",'X'));
    let mut file = OpenOptions::new().append(true).create(true).open("lda_c_vwn.txt");
    writeln!(file.expect("write failure"), "{}", get_homo_vx_or_vc(scf_data,"lda_c_vwn",'C'));
    let mut file = OpenOptions::new().append(true).create(true).open("lda_c_vwn_rpa.txt");
    writeln!(file.expect("write failure"), "{}", get_homo_vx_or_vc(scf_data,"lda_c_vwn_rpa",'C'));
    let mut file = OpenOptions::new().append(true).create(true).open("gga_c_lyp.txt");
    writeln!(file.expect("write failure"), "{}", get_homo_vx_or_vc(scf_data,"gga_c_lyp",'C'));
    let mut file = OpenOptions::new().append(true).create(true).open("gga_c_pbe.txt");
    writeln!(file.expect("write failure"), "{}", get_homo_vx_or_vc(scf_data,"gga_c_pbe",'C'));
    let mut file = OpenOptions::new().append(true).create(true).open("gga_c_xpbe.txt");
    writeln!(file.expect("write failure"), "{}", get_homo_vx_or_vc(scf_data,"gga_c_xpbe",'C'));
}
/// Low-rank version of get_homo_lumo_qp_only
fn get_homo_lumo_qp_only_lowrank(
    scf_data: &mut SCF,
    num_freq: usize,
    vxc_nn: &Vec<f64>,
    nomega_chi_real: usize,
    low_rank_tolerance: f64,
) {
    let printlevel = scf_data.mol.ctrl.print_level.clone();
    let ri_mat: MatrixFull<f64> = ri_bse::get_submatrix(scf_data, 'F', 'F', 'Y');
    let v_matrix = v_matrix(&scf_data, &ri_mat);
    let (start_mo, num_state, occ_size, vir_size, homo, lumo) =
        get_occupation_parameters(scf_data, 'Y');
    let qp_ctrl = scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let eigenenergies: Vec<f64> = scf_data.eigenvalues[0].clone();
    let quasiparticle_energies_g: Vec<f64> = eigenenergies.clone();
    let quasiparticle_energies_w: Vec<f64> = eigenenergies.clone();

    let ri_ov: MatrixFull<f64> = ri_bse::get_submatrix(scf_data, 'O', 'V', 'Y');

    // Step A: Generate W_c on imaginary axis
    let w_c_at_freqs = generate_w_c(
        scf_data, &ri_ov, &ri_mat,
        &quasiparticle_energies_g, &quasiparticle_energies_w,
        num_state, occ_size, vir_size, num_freq,
    );

    // Step B: Precompute low-rank on real axis
    // Only scan states that we're computing GW for (HOMO and LUMO)
    // to avoid core orbitals inflating de_max
    let nsemin = homo;
    let nsemax = lumo;
    let nomega_sigma = qp_ctrl.nomega_sigma;
    let step_sigma = qp_ctrl.step_sigma;
    let grid_type = if qp_ctrl.low_rank_grid_type == "quadratic" { 1 } else { 0 };
    let pl = scf_data.mol.ctrl.print_level;
    let real_axis_vchiv = generate_real_axis_vchiv(
        &quasiparticle_energies_g,
        &quasiparticle_energies_w,
        occ_size,
        vir_size,
        num_state,
        &ri_ov,
        &ri_mat,
        nomega_chi_real,
        nsemin,
        nsemax,
        nomega_sigma,
        step_sigma,
        low_rank_tolerance,
        grid_type,
        pl,
    );

    // HOMO
    let n = homo;
    let mut exchange = 0.0;
    for i in 0..homo + 1 {
        exchange -= v_matrix[[n, i]];
    }
    let consts = eigenenergies[n] + exchange - vxc_nn[n];
    if printlevel > 1 {
        println!("for n={},exchange={}", n, exchange);
    }
    let side = if n >= occ_size { 1.0 } else { -1.0 };
    let homo_qp = newton_solver_lowrank(
        n, consts, &ri_mat,
        &quasiparticle_energies_g, &quasiparticle_energies_w,
        occ_size, vir_size, num_state,
        &w_c_at_freqs, &real_axis_vchiv,
        eigenenergies[n], 0.00001, 50, side, printlevel,
    );
    println!("The QP energy of HOMO obtained by GWA (low-rank) is {}", homo_qp);

    // LUMO
    let n = lumo;
    let mut exchange = 0.0;
    for i in 0..homo + 1 {
        exchange -= v_matrix[[n, i]];
    }
    let consts = eigenenergies[n] + exchange - vxc_nn[n];
    if printlevel > 1 {
        println!("for n={},exchange={}", n, exchange);
    }
    let side = if n >= occ_size { 1.0 } else { -1.0 };
    let lumo_qp = newton_solver_lowrank(
        n, consts, &ri_mat,
        &quasiparticle_energies_g, &quasiparticle_energies_w,
        occ_size, vir_size, num_state,
        &w_c_at_freqs, &real_axis_vchiv,
        eigenenergies[n], 0.00001, 50, side, printlevel,
    );

    let save_path = qp_ctrl.save_qp_path.clone();
    if qp_ctrl.save_gw_homo_lumo_qp == true {
        let mut file = OpenOptions::new().append(true).create(true).open(save_path);
        writeln!(file.expect("write failure"), "{},{}", homo_qp, lumo_qp);
    }
    println!("The QP energy of LUMO obtained by GWA (low-rank) is {}", lumo_qp);
}

pub fn read_floats(path: &str) -> io::Result<Vec<f64>> {
    fs::read_to_string(path)?
        .lines()
        .map(|s| s.parse().map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e)))
        .collect()
}

//=========================================================================
// Low-rank contour deformation acceleration
//
// Implements the MOLGW-style approach: precompute sqrt(v)*chi*sqrt(v)
// on a real-axis linear grid, diagonalize to obtain low-rank representations,
// then use nearest-neighbor interpolation for residue evaluations.
//
// This avoids the O(N_aux^3) matrix build+inversion inside the Newton solver.
//=========================================================================

/// Low-rank representation of sqrt(v) * chi(omega) * sqrt(v) at a real frequency
pub struct LowRankVChiV {
    pub omega: f64,
    pub eigvec: MatrixFull<f64>,   // eigenvectors: (nauxil, n_keep)
    pub eigval: Vec<f64>,          // eigenvalues: lambda_v = lambda0_v / (1 - lambda0_v)
    pub n_keep: usize,
}

/// Collection of low-rank sqrt(v)*chi*sqrt(v) on a real-axis frequency grid
pub struct RealAxisVChiV {
    pub grid: Vec<LowRankVChiV>,
    pub omega_min: f64,
    pub omega_max: f64,
    pub n_points: usize,
    pub grid_type: usize,    // 0 = linear, 1 = power-law (quadratic, denser near zero)
}

impl LowRankVChiV {
    /// Interpolate to use this LowRankVChiV (for nearest-neighbor, just return self)
    pub fn n_keep(&self) -> usize {
        self.n_keep
    }
}

/// Build sqrt(v)*chi0(omega)*sqrt(v) at a single real frequency omega,
/// diagonalize it, and return the low-rank representation of sqrt(v)*chi*sqrt(v).
///
/// chi0(I,J) = 2 * sum_{ia} (I|ia) * (J|ia) * 2*(f_i-f_a)*de / (omega^2 - de^2)
///
/// After diagonalization: chi0 * e_v = lambda0_v * e_v
/// RPA Dyson equation: lambda_v = lambda0_v / (1 - lambda0_v)
///
/// Only eigenvalues with |lambda_v| > tolerance are retained.
pub fn low_rank_vchi_vsqrt(
    quasiparticle_energies_w: &Vec<f64>,
    occ_size: usize,
    vir_size: usize,
    ri_ov: &MatrixFull<f64>,    // 3-center integrals (I | i a), shape (nauxil, occ_size*vir_size)
    omega: f64,                  // real frequency
    tolerance: f64,              // eigenvalue cutoff (e.g., 1e-3)
) -> LowRankVChiV {
    let n_aux = ri_ov.size[0];
    let n_trans = occ_size * vir_size;

    // Step 1: Build ri_weighted(I, ia) = ri_ov(I, ia) * sqrt(|factor|)
    // where factor = 2 * docc * de / (omega^2 - de^2)
    // Note: For real omega, omega^2 - de^2 could be negative → factor sign handled properly
    let zero_threshold = 1e-12_f64;
    let mut ri_weighted = MatrixFull::new([n_aux, n_trans], 0.0);
    for ia in 0..n_trans {
        let i = ia % occ_size;
        let a = occ_size + ia / occ_size;
        let de = quasiparticle_energies_w[a] - quasiparticle_energies_w[i];
        let docc = 2.0;  // spin-restricted, occupation difference
        let denom = omega * omega - de * de;
        if denom.abs() < zero_threshold {
            continue;
        }
        let factor = 2.0 * docc * de / denom;
        let factor_scaled = if factor.abs() < zero_threshold {
            0.0
        } else {
            factor.abs().sqrt() * if factor > 0.0 { 1.0 } else { -1.0 }
        };
        for aux in 0..n_aux {
            ri_weighted[[aux, ia]] = ri_ov[[aux, ia]] * factor_scaled;
        }
    }

    // Step 2: chi0 = ri_weighted * ri_weighted^T  (n_aux × n_aux)
    // Note: The absolute and sign handling above means we effectively build
    // chi0 = sum_{ia} factor * (I|ia) * (J|ia)
    // via DGEMM with ri_weighted (which carries sqrt(|factor|) * sign)
    // However, DGEMM gives us ri_weighted * ri_weighted^T = sum (sqrt|f|*sign * I) * (sqrt|f|*sign * J)
    // = sum f * (I|ia) * (J|ia). Wait, no. DGEMM of A * A^T gives sum_k A_ik * A_jk.
    // If A(I,ia) = ri_ov(I,ia) * sqrt(|factor|) * sign, then
    // sum_ia A(I,ia)*A(J,ia) = sum_ia ri_ov(I,ia)*ri_ov(J,ia) * |factor| * sign^2
    // = sum_ia factor * ri_ov(I,ia) * ri_ov(J,ia) -- correct!
    // Wait, sign^2 = 1 always. So we lose the sign information.
    // We need a different approach. Let me reconsider.

    // Actually, the chi0 matrix is:
    // chi0(I,J) = sum_{ia} 2*docc*de/(omega^2 - de^2) * (I|ia) * (J|ia)
    //
    // If omega < de_min, then omega^2 - de^2 < 0 for all ia, so factor < 0.
    // But for residue corrections on the real axis, omega = |de| where de is the pole energy,
    // so omega < de for some transitions and omega > de for others.
    // The factor can be positive or negative.
    //
    // To handle this with DGEMM, we split:
    // ri_pos: (I,ia) * sqrt(|factor|) for factor > 0
    // ri_neg: (I,ia) * sqrt(|factor|) for factor < 0
    // chi0 = ri_pos * ri_pos^T - ri_neg * ri_neg^T

    // Actually, a simpler approach: build chi0 directly as described in MOLGW.
    // Let's just fill eri3_t1 and eri3_t2 separately, where eri3_t1 carries the factor
    // and eri3_t2 is the bare integral. Then: chi0 = eri3_t1 * eri3_t2^T

    // Let me redo this more carefully, following MOLGW exactly:
    // eri3_t1(:, ia) = (I|ia) * factor
    // eri3_t2(:, ia) = (I|ia)
    // chi0 = eri3_t1 * eri3_t2^T

    let mut eri3_t1 = MatrixFull::new([n_aux, n_trans], 0.0);
    let mut eri3_t2 = MatrixFull::new([n_aux, n_trans], 0.0);
    for ia in 0..n_trans {
        let i = ia % occ_size;
        let a = occ_size + ia / occ_size;
        let de = quasiparticle_energies_w[a] - quasiparticle_energies_w[i];
        let docc = 2.0;
        let denom = omega * omega - de * de;
        let factor = if denom.abs() < zero_threshold {
            0.0
        } else {
            2.0 * docc * de / denom
        };
        for aux in 0..n_aux {
            eri3_t1[[aux, ia]] = ri_ov[[aux, ia]] * factor;
            eri3_t2[[aux, ia]] = ri_ov[[aux, ia]];
        }
    }

    let mut chi0 = MatrixFull::new([n_aux, n_aux], 0.0);
    _dgemm_full(&eri3_t1, 'N', &eri3_t2, 'T', &mut chi0, 1.0, 0.0);

    // Step 3: Diagonalize chi0
    let (eigvecs_opt, eigvals_raw, _info) = _dsyev(&chi0, 'V');
    let eigvecs = eigvecs_opt.expect("low_rank_vchi_vsqrt: dsyev failed");
    // _dsyev returns eigenvalues in ascending order, but we want descending by magnitude
    // Collect and sort
    let mut pairs: Vec<(f64, Vec<f64>)> = eigvals_raw.iter().enumerate().map(|(v, &lam0)| {
        let lam = lam0 / (1.0 - lam0);
        let vec: Vec<f64> = (0..n_aux).map(|r| eigvecs[[r, v]]).collect();
        (lam, vec)
    }).collect();

    // Sort by absolute eigenvalue descending
    pairs.sort_by(|a, b| b.0.abs().partial_cmp(&a.0.abs()).unwrap_or(std::cmp::Ordering::Equal));

    // Step 4: Keep only non-negligible eigenvalues
    let mut keep_idx: Vec<usize> = Vec::new();
    for (idx, (lam, _)) in pairs.iter().enumerate() {
        if lam.abs() > tolerance {
            keep_idx.push(idx);
        }
    }
    let n_keep = keep_idx.len();

    // Build eigvec and eigval arrays
    let mut eigvec_mat = MatrixFull::new([n_aux, n_keep], 0.0);
    let mut eigval_vec = Vec::with_capacity(n_keep);
    for (j, &idx) in keep_idx.iter().enumerate() {
        eigval_vec.push(pairs[idx].0);
        for aux in 0..n_aux {
            eigvec_mat[[aux, j]] = pairs[idx].1[aux];
        }
    }

    if n_keep < n_aux {
        //println!("  omega={:.6}: kept {} eigenvalues out of {} (tolerance={})", omega, n_keep, n_aux, tolerance);
    }

    LowRankVChiV {
        omega,
        eigvec: eigvec_mat,
        eigval: eigval_vec,
        n_keep,
    }
}

/// Precompute low-rank sqrt(v)*chi*sqrt(v) on a real-axis grid [0, de_max].
///
/// First scans all (mstate, pstate, omega_sigma) combinations to find the maximum
/// real frequency needed for residue corrections (de_max).
/// Then builds the low-rank representation at nomega_chi_real grid points.
///
/// grid_type:
///   0 = linear (equally-spaced)
///   1 = quadratic (power-law, denser near zero; recommended for core-level GW)
///        omega_i = de_max * (i/(N-1))^2
pub fn generate_real_axis_vchiv(
    quasiparticle_energies_g: &Vec<f64>,
    quasiparticle_energies_w: &Vec<f64>,
    occ_size: usize,
    vir_size: usize,
    num_state: usize,
    ri_ov: &MatrixFull<f64>,
    ri_full: &MatrixFull<f64>,
    nomega_chi_real: usize,
    nsemin: usize,
    nsemax: usize,
    nomega_sigma: usize,
    step_sigma: f64,
    tolerance: f64,
    grid_type: usize,
    print_level: usize,
) -> RealAxisVChiV {
    if print_level > 2 {
        println!("[DEBUG generate_real_axis_vchiv] === Real-axis frequency grid setup ===");
        println!("[DEBUG] nomega_chi_real={} nsemin={} nsemax={} nomega_sigma={} step_sigma={}",
                 nomega_chi_real, nsemin, nsemax, nomega_sigma, step_sigma);
        println!("[DEBUG] Total sigma scan range per state: [{:.6}, {:.6}] Ha",
                 -(nomega_sigma as f64) * step_sigma, (nomega_sigma as f64) * step_sigma);
    }

    // Step 1: Find de_max
    let eta = 1e-6_f64;
    let mut de_max = 0.0_f64;

    for mstate in nsemin..=nsemax {
        let energy0 = quasiparticle_energies_g[mstate];
        if print_level > 2 {
            println!("[DEBUG] Scanning mstate={} energy0={:.6} Ha", mstate, energy0);
        }
        let mut state_de_max = 0.0_f64;
        for iomega_sigma in -(nomega_sigma as isize)..=(nomega_sigma as isize) {
            let omega = energy0 + (iomega_sigma as f64) * step_sigma;

            // Occupied state poles: eps_p > omega, de = eps_p - omega
            for p in 0..occ_size {
                let de = quasiparticle_energies_g[p] - omega;
                if de > eta {
                    de_max = de_max.max(de);
                    state_de_max = state_de_max.max(de);
                }
            }

            // Empty state poles: omega > eps_a, de = omega - eps_a
            for a in occ_size..num_state {
                let de = omega - quasiparticle_energies_g[a];
                if de > eta {
                    de_max = de_max.max(de);
                    state_de_max = state_de_max.max(de);
                }
            }
        }
        if print_level > 2 {
            println!("[DEBUG]   de_max contribution from mstate={}: {:.6} Ha", mstate, state_de_max);
        }
    }

    // Add a small margin
    de_max = de_max * 1.05 + 0.1;

    let grid_label = if grid_type == 1 { "quadratic (power-law)" } else { "linear" };
    println!("Low-rank contour: Maximum real frequency needed for v*chi*v = {:.6} Ha = {:.6} eV",
             de_max, de_max * 27.2113863);
    println!("Low-rank contour: Computing sqrt(v)*chi*sqrt(v) at {} real-axis grid points ({})",
             nomega_chi_real, grid_label);
    if print_level > 2 {
        println!("[DEBUG] Grid formula: omega_i = de_max * i/(N-1)  (i=0..{})", nomega_chi_real-1);
        println!("[DEBUG] de_max (after 1.05* + 0.1) = {:.10} Ha", de_max);
    }

    // Step 2: Build low-rank representation at each grid point
    let mut grid: Vec<LowRankVChiV> = Vec::with_capacity(nomega_chi_real);
    for i_omega in 0..nomega_chi_real {
        let omega_real = if nomega_chi_real > 1 {
            let t = (i_omega as f64) / ((nomega_chi_real - 1) as f64);
            if grid_type == 1 {
                // Power-law grid: omega = de_max * t^2, denser near zero
                de_max * t * t
            } else {
                // Linear grid
                de_max * t
            }
        } else {
            de_max
        };
        if print_level > 2 {
            println!("[DEBUG] Real-axis freq {} / {} : omega = {:.10} Ha = {:.6} eV  (t={:.6})",
                     i_omega + 1, nomega_chi_real, omega_real, omega_real * 27.2113863,
                     (i_omega as f64) / ((nomega_chi_real - 1) as f64));
        }
        let lr = low_rank_vchi_vsqrt(
            quasiparticle_energies_w, occ_size, vir_size, ri_ov, omega_real, tolerance,
        );
        println!("    Non-negligible eigenvalues: {} / {}", lr.n_keep, ri_ov.size[0]);
        grid.push(lr);
    }

    RealAxisVChiV {
        grid,
        omega_min: 0.0,
        omega_max: de_max,
        n_points: nomega_chi_real,
        grid_type: grid_type,
    }
}

/// Nearest-neighbor interpolation to obtain the low-rank vchi_v at |de|
/// Uses binary search for non-uniform grids, or direct index for uniform grids.
pub fn interpolate_vchiv_nearest<'a>(
    real_axis: &'a RealAxisVChiV,
    de_abs: f64,
) -> &'a LowRankVChiV {
    let n = real_axis.grid.len();
    if n <= 1 {
        return &real_axis.grid[0];
    }

    // Clamp to grid range
    if de_abs >= real_axis.grid[n - 1].omega {
        return &real_axis.grid[n - 1];
    }
    if de_abs <= real_axis.grid[0].omega {
        return &real_axis.grid[0];
    }

    // Binary search for nearest neighbor
    let mut lo = 0;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if real_axis.grid[mid].omega < de_abs {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    // Choose the closer of grid[lo] and grid[hi]
    if (de_abs - real_axis.grid[lo].omega) < (real_axis.grid[hi].omega - de_abs) {
        &real_axis.grid[lo]
    } else {
        &real_axis.grid[hi]
    }
}

/// Compute a single residue contribution using the low-rank representation:
///
///   residue = sum_v [ sum_I ri_vec[I] * eigvec[I, v] ]^2 * eigval[v]
///
/// This corresponds to: (I|n,p)^T * [sum_v e_v * lambda_v * e_v^T] * (I|n,p)
pub fn compute_single_residue_lowrank(lr: &LowRankVChiV, ri_vec: &[f64]) -> f64 {
    if lr.n_keep == 0 {
        return 0.0;
    }

    let n_aux = lr.eigvec.size[0];
    let n_keep = lr.n_keep;
    let mut residue = 0.0_f64;

    for v in 0..n_keep {
        let mut tmp = 0.0_f64;
        for aux in 0..n_aux {
            tmp += ri_vec[aux] * lr.eigvec[[aux, v]];
        }
        residue += tmp * tmp * lr.eigval[v];
    }

    residue
}

/// Fast version of contour_rayon that uses pre-computed low-rank real-axis v*chi*v
/// instead of building and inverting the full response matrix at each residue pole.
///
/// The mathematical decomposition:
///   Sigma_c = Sigma_c^imag (from generate_w_c) + Sigma_c^residue (from this function)
///
/// For occupied state poles (omega < epsilon_F):
///   Sigma_c^res_occ = - SUM_{p} [ sum_v (tmp_v^p)^2 * lambda_v(|de|) ]  where de = eps_p - omega
///
/// For empty state poles (omega > epsilon_F):
///   Sigma_c^res_vir = + SUM_{a} [ sum_v (tmp_v^a)^2 * lambda_v(|de|) ]  where de = omega - eps_a
pub fn contour_rayon_lowrank(
    omega: f64,
    n: usize,
    quasiparticle_energies_g: &Vec<f64>,
    quasiparticle_energies_w: &Vec<f64>,
    occ_size: usize,
    vir_size: usize,
    num_state: usize,
    ri_full: &MatrixFull<f64>,
    real_axis_vchiv: &RealAxisVChiV,
    print_level: usize,
) -> f64 {
    let fermi_energy = (quasiparticle_energies_g[occ_size - 1] + quasiparticle_energies_g[occ_size]) / 2.0;
    let sign = if omega > fermi_energy { 1.0_f64 } else { -1.0_f64 };
    let eta_pole = 1e-6_f64;   // MOLGW uses eta=1e-6 for pole protection

    if sign == 1.0 {
        // Empty state residue (omega > epsilon_F): sign is +1
        // Contributions from virtual poles where de = omega - eps_a > -eta
        // MOLGW: if( de < -eta ) cycle; factor = MERGE(0.5, 1.0, ABS(de) < eta)
        (0..vir_size).into_par_iter().map(|a| {
            let a_global = occ_size + a;
            let mut residue = 0.0_f64;
            let de = omega - quasiparticle_energies_g[a_global];
            if de >= -eta_pole {
                let lr = interpolate_vchiv_nearest(real_axis_vchiv, de);
                let col_idx = n + a_global * num_state;
                let ri_vec: Vec<f64> = ri_full.iter_column(col_idx).copied().collect();
                let pole_factor = if de.abs() < eta_pole { 0.5 } else { 1.0 };
                residue = compute_single_residue_lowrank(lr, &ri_vec) * pole_factor;
                if print_level > 2 {
                    println!("[DEBUG individual residue] virtual a={} eps_a={:.6} de={:.6}  factor={:.2}  residue_contrib={:.10}",
                             a_global, quasiparticle_energies_g[a_global], de, pole_factor, residue * sign);
                }
            }
            residue * sign
        }).sum()
    } else {
        // Occupied state residue (omega < epsilon_F): sign is -1
        // Contributions from occupied poles where de = eps_i - omega > -eta
        // MOLGW: if( de < -eta ) cycle; factor = MERGE(0.5, 1.0, ABS(de) < eta)
        //
        // In MOLGW m_gw_selfenergy_grid.f90:498-499:
        //   sigmagw(...) = sigmagw(...) - SUM(tmp^2 * eigval) * factor
        //
        // And the equivalent REST contour_rayon occupies:
        //   residue = +dot(inv_dielectric * vec, vec)  [line 610]
        //   residue *= sign  [line 612]
        //   where sign=-1 for occupied
        //
        // So: Sigma_res_occ *= -1 => -SUM_i residue_per_pole
        //
        // Now with low-rank, the residue_per_pole is:
        //   compute_single_residue_lowrank(lr, ri_vec) = sum_v tmp_v^2 * lambda_v
        //
        // And the full residue = - SUM_i compute_single_residue_lowrank(...) * sign
        // = - SUM_i r_lowrank * (-1) = +SUM_i r_lowrank
        //
        // Wait, but in contour_rayon the full function returns sum over all parallels:
        //   residue *= (sign as f64); residue
        // Then .sum() sums these up.
        //
        // So for occupied (sign=-1):
        //   if eps_i > omega:
        //     residue = dot(inv_di, vec) * (-1)
        //   Sum_i residue (all negative or zero)
        //
        // The total contour = SUM_i [residue at each pole] (already has sign applied)
        //
        // In the low-rank version, we compute residue = compute_single_residue_lowrank(lr, ri_vec)
        // which returns a positive value when eigenvalues are nonzero.
        // Then we multiply by sign:
        //   for occupied (sign=-1): residue * (-1) = -residue
        //   for empty (sign=+1):    residue * (+1) = +residue
        //
        // But wait, signing: in MOLGW, for occupied poles (first quadrant):
        //   sigmagw -= SUM(tmp^2 * eigval)   [line 499]
        // This gives a negative contribution.
        //
        // In REST's contour_rayon original:
        //   residue = dot(inverse_dielectric * vec, vec)   -- this is positive
        //   residue *= sign   -- sign=-1 for occupied → negative
        //   return residue    -- negative
        //   Sum_i residue
        //
        // So REST returns -SUM_i[dots for poles]. That matches MOLGW.
        //
        // For low-rank, the "dot product" equivalent is compute_single_residue_lowrank which
        // returns positive. Then:
        //   residue = compute_single_residue_lowrank(lr, &ri_vec) * sign
        // For occupied: positive * (-1) = negative  ✓
        //
        // But wait, MOLGW's occupied residue is:
        //   sigmagw -= SUM( tmp^2 * lambda )    (negative contribution)
        // REST's occupied residue:
        //   residue = dot(W_c * vec, vec) * (-1)   (negative per pole)
        //
        // The low-rank representation gives us v*chi*v, which is equivalent to W_c.
        // compute_single_residue_lowrank(lr, &ri_vec) = ri_vec^T * (v*chi*v) * ri_vec
        // This is exactly the "dot(inverse_dielectric * vec, vec)" from REST's original code.
        //
        // So the correct formulation is:
        //   let r = compute_single_residue_lowrank(lr, &ri_vec);
        //   residue = r * sign;

        (0..occ_size).into_par_iter().map(|i| {
            let mut residue = 0.0_f64;
            let de = quasiparticle_energies_g[i] - omega;
            if de >= -eta_pole {
                let lr = interpolate_vchiv_nearest(real_axis_vchiv, de);
                let col_idx = n + i * num_state;
                let ri_vec: Vec<f64> = ri_full.iter_column(col_idx).copied().collect();
                let pole_factor = if de.abs() < eta_pole { 0.5 } else { 1.0 };
                residue = compute_single_residue_lowrank(lr, &ri_vec) * pole_factor;
                if print_level > 2 {
                    println!("[DEBUG individual residue] occupied i={} eps_i={:.6} de={:.6}  factor={:.2}  residue_contrib={:.10}",
                             i, quasiparticle_energies_g[i], de, pole_factor, residue * sign);
                }
            }
            residue * sign
        }).sum()
    }
}

/// Low-rank version of quasiparticle_equation: uses pre-computed imaginary-axis W_c
/// and pre-computed real-axis low-rank v*chi*v for contour (residue) contributions.
///
/// This is equivalent to the original `quasiparticle_equation` but uses
/// `contour_rayon_lowrank` instead of `contour_rayon`.
pub fn quasiparticle_equation_lowrank(
    omega: f64,
    n: usize,
    consts: f64,
    ri_full: &MatrixFull<f64>,
    quasiparticle_energies_g: &Vec<f64>,
    quasiparticle_energies_w: &Vec<f64>,
    occ_size: usize,
    vir_size: usize,
    num_state: usize,
    w_c_at_freqs: &Vec<(f64, f64, MatrixFull<f64>)>,
    real_axis_vchiv: &RealAxisVChiV,
    print_level: usize,
) -> f64 {
    let contour = contour_rayon_lowrank(
        omega, n, quasiparticle_energies_g, quasiparticle_energies_w,
        occ_size, vir_size, num_state, ri_full, real_axis_vchiv, print_level,
    );
    let imag = calculate_imag(
        w_c_at_freqs, num_state, n, omega,
        quasiparticle_energies_g, quasiparticle_energies_w,
    );
    if print_level > 2 {
        println!("[DEBUG Sigma_c breakdown] n={} omega={:.10} Ha  contour(residue)={:.10} Ha  imag_axis={:.10} Ha  Sigma_c={:.10} Ha",
                 n, omega, contour, imag, contour - imag);
    }
    consts + contour - imag - omega
}

/// Low-rank version of the Newton solver for QP equation.
/// Uses pre-computed low-rank real-axis v*chi*v for residue evaluations.
pub fn newton_solver_lowrank(
    n: usize,
    consts: f64,
    ri_full: &MatrixFull<f64>,
    quasiparticle_energies_g: &Vec<f64>,
    quasiparticle_energies_w: &Vec<f64>,
    occ_size: usize,
    vir_size: usize,
    num_state: usize,
    w_c_at_freqs: &Vec<(f64, f64, MatrixFull<f64>)>,
    real_axis_vchiv: &RealAxisVChiV,
    starting_point: f64,
    tol: f64,
    max_iter: usize,
    side: f64,
    printlevel: usize,
) -> f64 {
    let h = 0.000001;
    let delta = 0.02;
    let mut x_curr = starting_point + side * delta;

    let qp_eq = |omega: f64| {
        quasiparticle_equation_lowrank(
            omega, n, consts, ri_full,
            quasiparticle_energies_g, quasiparticle_energies_w,
            occ_size, vir_size, num_state, w_c_at_freqs, real_axis_vchiv, 0,
        )
    };

    let mut y_curr = qp_eq(x_curr);
    let mut y_plus = qp_eq(x_curr + h);
    let mut y_minus = qp_eq(x_curr - h);
    let mut converge = 0;
    let mut iter_times = 0;

    loop {
        let derivative = (y_plus - y_minus) / (2.0 * h);
        let shift = -y_curr / derivative;
        if printlevel > 1 {
            println!(
                "newton (lowrank) now: n={}, x_curr={}, y_curr={}, derivative={}, shift={}",
                n, x_curr, y_curr, derivative, shift
            );
        }
        x_curr += shift;
        y_curr = qp_eq(x_curr);
        if shift.abs() < tol {
            converge += 1;
            if printlevel > 1 {
                println!(
                    "convergence (lowrank): x_curr={}, shift={}, y_curr={}",
                    x_curr, shift, y_curr
                );
            }
        }
        y_plus = qp_eq(x_curr + h);
        y_minus = qp_eq(x_curr - h);
        iter_times += 1;
        if converge == 1 || iter_times == max_iter {
            break;
        }
    }
    if iter_times == max_iter {
        println!(
            "warning!!! newton solver (lowrank) did not converge for orbital {}!",
            n
        );
    }
    x_curr
}

/// Low-rank version of linear_interpolation_solver for QP equation.
pub fn linear_interpolation_solver_lowrank(
    f: impl Fn(f64) -> f64,
    starting_point: f64,
    side: f64,
    grid_freqs: usize,
    span_energy: f64,
) -> (bool, f64) {
    // For the linear interpolation solver, we wrap the low-rank QP equation
    // as a closure and delegate to the existing linear_interpolation_solver
    // Since linear_interpolation_solver takes Fn(f64)->f64, we can directly use it.
    // But the original linear_interpolation_solver doesn't take a generic closure...
    // Let me check the original signature.
    // The original is: pub fn linear_interpolation_solver<F>(mut f:F,...) where F:Fn(f64)->f64
    // So yes, it already takes a closure. But we need to import/use it.
    // Since we're in the same module, we just call it directly.

    linear_interpolation_solver(f, starting_point, side, grid_freqs, span_energy)
}

/// Full GW calculation using low-rank accelerated contour deformation.
/// This is the low-rank equivalent of `gw_calculations`.
pub fn gw_calculations_lowrank(
    scf_data: &mut SCF,
    num_freq: usize,
    vxc_nn: &Vec<f64>,
    cancel_dfa_xc: bool,
    nomega_chi_real: usize,
    low_rank_tolerance: f64,
) -> Vec<f64> {
    let mut ri_mat: MatrixFull<f64> = ri_bse::get_submatrix(scf_data, 'F', 'F', 'Y');
    let mut ri_ov: MatrixFull<f64> = ri_bse::get_submatrix(scf_data, 'O', 'V', 'Y');
    let qp_ctrl = scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let v_matrix = v_matrix(&scf_data, &ri_mat);
    let (start_mo, num_state, occ_size, vir_size, homo, lumo) =
        get_occupation_parameters(scf_data, 'Y');
    let (_start_mo, num_state_cutoff, _occ_size, _vir_size_cutoff, _homo, _lumo) =
        get_occupation_parameters(scf_data, 'N');

    let quasiparticle_energies_g = scf_data.gwqp.0.clone();
    let quasiparticle_energies_w = scf_data.gwqp.1.clone();

    // Step A: Compute W_c on imaginary axis (same as original)
    println!("Low-rank contour: Generating W_c on imaginary axis grid...");
    let w_c_at_freqs = generate_w_c(
        scf_data, &ri_ov, &ri_mat,
        &quasiparticle_energies_g, &quasiparticle_energies_w,
        num_state, occ_size, vir_size, num_freq,
    );

    // Step B: Precompute low-rank sqrt(v)*chi*sqrt(v) on real axis
    println!("Low-rank contour: Precomputing real-axis sqrt(v)*chi*sqrt(v)...");
    let nsemin = 0;
    let nsemax = num_state_cutoff - 1;
    let nomega_sigma = qp_ctrl.nomega_sigma;
    let step_sigma = qp_ctrl.step_sigma;
    let grid_type = if qp_ctrl.low_rank_grid_type == "quadratic" { 1 } else { 0 };
    let pl = scf_data.mol.ctrl.print_level;
    let real_axis_vchiv = generate_real_axis_vchiv(
        &quasiparticle_energies_g,
        &quasiparticle_energies_w,
        occ_size,
        vir_size,
        num_state,
        &ri_ov,
        &ri_mat,
        nomega_chi_real,
        nsemin,
        nsemax,
        nomega_sigma,
        step_sigma,
        low_rank_tolerance,
        grid_type,
        pl,
    );

    // Check whether self-energy correction is enabled
    let use_fourier = qp_ctrl.fourier_self_energy;
    let use_hermite = qp_ctrl.hermite_self_energy;

    if use_fourier && use_hermite {
        panic!("Cannot enable both Fourier self-energy and Hermite self-energy simultaneously!");
    }

    // First round: normal GW without self-energy correction
    let mut save_energies_no_se: Vec<f64> = vec![0.0; num_state_cutoff];

    save_energies_no_se = (0..num_state_cutoff)
        .map(|n| {
            let consts = if cancel_dfa_xc == true {
                let mut exchange = 0.0;
                for i in 0..homo + 1 {
                    exchange -= v_matrix[[n, i]];
                }
                scf_data.eigenvalues[0][n] + exchange - vxc_nn[n]
            } else {
                quasiparticle_energies_g[n]
            };
            let side = if n >= occ_size { 1.0 } else { -1.0 };

            let qp_eq_func = |omega: f64| {
                quasiparticle_equation_lowrank(
                    omega, n, consts, &ri_mat,
                    &quasiparticle_energies_g, &quasiparticle_energies_w,
                    occ_size, vir_size, num_state,
                    &w_c_at_freqs, &real_axis_vchiv, 0,
                )
            };
            let (_have_crossing, real_qp) =
                linear_interpolation_solver(qp_eq_func, scf_data.eigenvalues[0][n], side, 21, 0.1);
            println!(
                "for n={}, first round (low-rank, no SE correction): qp energy={}",
                n, real_qp
            );
            real_qp
        })
        .collect::<Vec<f64>>()
        .clone();

    if !use_fourier && !use_hermite {
        let result = save_energies_no_se.clone();
        scf_data.gwqp.0 = result.clone();
        display::full_quasiparticles(&result, occ_size);
        return result;
    }

    // Second round: with self-energy correction
    let mut save_energies: Vec<f64> = vec![0.0; num_state_cutoff];

    if use_fourier {
        let (powers, t, sin_coeff, cos_coeff) =
            fourier_self_energy::define_fourier_series(&qp_ctrl);
        println!("Fourier Self Energy enabled for second round (low-rank)");

        save_energies = (0..num_state_cutoff)
            .map(|n| {
                let origin = save_energies_no_se[n];
                let consts = if cancel_dfa_xc == true {
                    let mut exchange = 0.0;
                    for i in 0..homo + 1 {
                        exchange -= v_matrix[[n, i]];
                    }
                    scf_data.eigenvalues[0][n] + exchange - vxc_nn[n]
                } else {
                    quasiparticle_energies_g[n]
                };
                let side = if n >= occ_size { 1.0 } else { -1.0 };

                let qp_eq_func_with_fse = |omega: f64| {
                    quasiparticle_equation_lowrank(
                        omega, n, consts, &ri_mat,
                        &quasiparticle_energies_g, &quasiparticle_energies_w,
                        occ_size, vir_size, num_state,
                        &w_c_at_freqs, &real_axis_vchiv, 0,
                    ) + fourier_self_energy::fourier_series(
                        &sin_coeff, &cos_coeff, powers, t, omega - origin,
                    )
                };

                let (_have_crossing, real_qp) = linear_interpolation_solver(
                    qp_eq_func_with_fse,
                    save_energies_no_se[n],
                    side,
                    qp_ctrl.gw_search_grid,
                    qp_ctrl.gw_span_energy,
                );
                println!(
                    "for n={}, second round (low-rank, with FSE): qp energy={}",
                    n, real_qp
                );
                real_qp
            })
            .collect::<Vec<f64>>()
            .clone();
    } else if use_hermite {
        let hermite_coeff =
            fourier_self_energy::define_hermite_series(&qp_ctrl);
        println!("Hermite Self Energy enabled for second round (low-rank)");

        save_energies = (0..num_state_cutoff)
            .map(|n| {
                let origin = save_energies_no_se[n];
                let consts = if cancel_dfa_xc == true {
                    let mut exchange = 0.0;
                    for i in 0..homo + 1 {
                        exchange -= v_matrix[[n, i]];
                    }
                    scf_data.eigenvalues[0][n] + exchange - vxc_nn[n]
                } else {
                    quasiparticle_energies_g[n]
                };
                let side = if n >= occ_size { 1.0 } else { -1.0 };

                let qp_eq_func_with_hse = |omega: f64| {
                    quasiparticle_equation_lowrank(
                        omega, n, consts, &ri_mat,
                        &quasiparticle_energies_g, &quasiparticle_energies_w,
                        occ_size, vir_size, num_state,
                        &w_c_at_freqs, &real_axis_vchiv, 0,
                    ) + fourier_self_energy::sigma_hermite(origin, omega, &hermite_coeff)
                };

                let (_have_crossing, real_qp) = linear_interpolation_solver(
                    qp_eq_func_with_hse,
                    save_energies_no_se[n],
                    side,
                    qp_ctrl.gw_search_grid,
                    qp_ctrl.gw_span_energy,
                );
                println!(
                    "for n={}, second round (low-rank, with HSE): qp energy={}",
                    n, real_qp
                );
                real_qp
            })
            .collect::<Vec<f64>>()
            .clone();
    }

    let result = save_energies.clone();
    scf_data.gwqp.0 = result.clone();
    display::full_quasiparticles(&result, occ_size);
    result
}

/// Low-rank accelerated version of linearized_gw.
///
/// Computes GW QP energies using linearization of the self-energy,
/// with contour deformation accelerated by pre-computed low-rank
/// sqrt(v)*chi*sqrt(v) on the real axis.
pub fn linearized_gw_lowrank(
    scf_data: &mut SCF,
    num_freq: usize,
    vxc_nn: &Vec<f64>,
    cancel_dfa_xc: bool,
    nomega_chi_real: usize,
    low_rank_tolerance: f64,
) -> Vec<f64> {
    let qp_ctrl = scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let delta = qp_ctrl.gw_linearize_shift;
    let ri_mat: MatrixFull<f64> = ri_bse::get_submatrix(scf_data, 'F', 'F', 'Y');
    let v_matrix = v_matrix(&scf_data, &ri_mat);
    let (start_mo, num_state, occ_size, vir_size, homo, lumo) =
        get_occupation_parameters(scf_data, 'Y');
    let mut num_state_qp_range = num_state;
    if qp_ctrl.scgw == "g0w0" {
        let (_start_mo_n, num_state_n, _occ_size_n, _vir_size_n, _homo_n, _lumo_n) =
            get_occupation_parameters(scf_data, 'N');
        num_state_qp_range = num_state_n;
    }
    let eigenenergies: Vec<f64> = scf_data.eigenvalues[0].clone();
    let ri_mat_inner: MatrixFull<f64> = ri_bse::get_submatrix(scf_data, 'F', 'F', 'Y');
    let quasiparticle_energies_g: Vec<f64> = scf_data.gwqp.0.clone();
    let quasiparticle_energies_w: Vec<f64> = scf_data.gwqp.1.clone();
    let mut ri_ov: MatrixFull<f64> = ri_bse::get_submatrix(scf_data, 'O', 'V', 'Y');

    // Generate W_c on imaginary axis (same as original)
    let w_c_at_freqs = generate_w_c(
        scf_data, &ri_ov, &ri_mat_inner,
        &quasiparticle_energies_g, &quasiparticle_energies_w,
        num_state, occ_size, vir_size, num_freq,
    );

    // Precompute low-rank on real axis
    let nsemin = 0;
    let nsemax = num_state_qp_range - 1;
    let nomega_sigma = qp_ctrl.nomega_sigma;
    let step_sigma = qp_ctrl.step_sigma;
    let grid_type = if qp_ctrl.low_rank_grid_type == "quadratic" { 1 } else { 0 };
    let pl = scf_data.mol.ctrl.print_level;
    let real_axis_vchiv = generate_real_axis_vchiv(
        &quasiparticle_energies_g,
        &quasiparticle_energies_w,
        occ_size,
        vir_size,
        num_state,
        &ri_ov,
        &ri_mat_inner,
        nomega_chi_real,
        nsemin,
        nsemax,
        nomega_sigma,
        step_sigma,
        low_rank_tolerance,
        grid_type,
        pl,
    );

    let h = qp_ctrl.gw_linearize_derivative_h;
    let use_fourier = qp_ctrl.fourier_self_energy;
    let use_hermite = qp_ctrl.hermite_self_energy;

    if use_fourier && use_hermite {
        panic!("Cannot enable both Fourier self-energy and Hermite self-energy simultaneously!");
    }

    // First round: normal linearized GW without self-energy correction
    let save_energies_no_se: Vec<f64> = (0..num_state_qp_range)
        .map(|n| {
            let side = if n >= occ_size { 1.0 } else { -1.0 };
            let omega_shifted = scf_data.eigenvalues[0][n] - delta * side;

            let qp_eq = |omega: f64| {
                quasiparticle_equation_lowrank(
                    omega, n, 0.0, &ri_mat_inner,
                    &quasiparticle_energies_g, &quasiparticle_energies_w,
                    occ_size, vir_size, num_state,
                    &w_c_at_freqs, &real_axis_vchiv, 0,
                )
            };

            let imag_n = calculate_imag(
                &w_c_at_freqs, num_state, n, omega_shifted,
                &quasiparticle_energies_g, &quasiparticle_energies_w,
            );
            let contour = contour_rayon_lowrank(
                omega_shifted, n,
                &quasiparticle_energies_g, &quasiparticle_energies_w,
                occ_size, vir_size, num_state,
                &ri_mat_inner, &real_axis_vchiv, 0,
            );

            let imag_plus_h = calculate_imag(
                &w_c_at_freqs, num_state, n, omega_shifted + h,
                &quasiparticle_energies_g, &quasiparticle_energies_w,
            );
            let imag_minus_h = calculate_imag(
                &w_c_at_freqs, num_state, n, omega_shifted - h,
                &quasiparticle_energies_g, &quasiparticle_energies_w,
            );
            let contour_plus_h = contour_rayon_lowrank(
                omega_shifted + h, n,
                &quasiparticle_energies_g, &quasiparticle_energies_w,
                occ_size, vir_size, num_state,
                &ri_mat_inner, &real_axis_vchiv, 0,
            );
            let contour_minus_h = contour_rayon_lowrank(
                omega_shifted - h, n,
                &quasiparticle_energies_g, &quasiparticle_energies_w,
                occ_size, vir_size, num_state,
                &ri_mat_inner, &real_axis_vchiv, 0,
            );

            let self_energy_plus_h = contour_plus_h - imag_plus_h;
            let self_energy_minus_h = contour_minus_h - imag_minus_h;
            let derivative = (self_energy_plus_h - self_energy_minus_h) / (2.0 * h);
            if scf_data.mol.ctrl.print_level > 1 {
                println!("derivative at n={}: {}", n, derivative);
            }
            let spectral_weight = (1.0 - derivative).powf(-1.0);
            println!("spectral weight at n={}: {}", n, spectral_weight);
            let real_qp = if cancel_dfa_xc == true {
                let mut exchange = 0.0;
                for i in 0..homo + 1 {
                    exchange -= v_matrix[[n, i]];
                }
                omega_shifted
                    + (spectral_weight * (contour - imag_n + exchange - vxc_nn[n]))
            } else {
                omega_shifted + (spectral_weight * (contour - imag_n))
            };
            println!(
                "for n={}, first round (low-rank, no SE corr.): linearized gw yields qp energy={}",
                n, real_qp
            );
            real_qp
        })
        .collect::<Vec<f64>>();

    if !use_fourier && !use_hermite {
        let result = save_energies_no_se.clone();
        display::full_quasiparticles(&result, occ_size);
        return result;
    }

    // Second round: add self-energy correction
    if use_fourier {
        let (powers, t, sin_coeff, cos_coeff) =
            fourier_self_energy::define_fourier_series(&qp_ctrl);
        println!("Fourier Self Energy enabled for second round (low-rank, linearized GW)");

        let save_energies: Vec<f64> = (0..num_state_qp_range)
            .map(|n| {
                let origin = save_energies_no_se[n];
                let side = if n >= occ_size { 1.0 } else { -1.0 };
                let omega_shifted = scf_data.eigenvalues[0][n] - delta * side;

                let sigma_with_fse = |omega: f64| -> f64 {
                    let contour = contour_rayon_lowrank(
                        omega, n,
                        &quasiparticle_energies_g, &quasiparticle_energies_w,
                        occ_size, vir_size, num_state,
                        &ri_mat_inner, &real_axis_vchiv, 0,
                    );
                    let imag = calculate_imag(
                        &w_c_at_freqs, num_state, n, omega,
                        &quasiparticle_energies_g, &quasiparticle_energies_w,
                    );
                    let fse = fourier_self_energy::fourier_series(
                        &sin_coeff, &cos_coeff, powers, t, omega - origin,
                    );
                    contour - imag + fse
                };

                let self_energy_at_omega = sigma_with_fse(omega_shifted);
                let self_energy_plus_h = sigma_with_fse(omega_shifted + h);
                let self_energy_minus_h = sigma_with_fse(omega_shifted - h);
                let derivative = (self_energy_plus_h - self_energy_minus_h) / (2.0 * h);

                if scf_data.mol.ctrl.print_level > 1 {
                    println!("derivative at n={} (with FSE): {}", n, derivative);
                }
                let spectral_weight = (1.0 - derivative).powf(-1.0);
                println!("spectral weight at n={} (with FSE): {}", n, spectral_weight);

                let real_qp = if cancel_dfa_xc == true {
                    let mut exchange = 0.0;
                    for i in 0..homo + 1 {
                        exchange -= v_matrix[[n, i]];
                    }
                    omega_shifted
                        + (spectral_weight
                            * (self_energy_at_omega + exchange - vxc_nn[n]))
                } else {
                    omega_shifted + (spectral_weight * self_energy_at_omega)
                };
                println!(
                    "for n={}, second round (low-rank, with FSE): linearized gw yields qp energy={}",
                    n, real_qp
                );
                real_qp
            })
            .collect::<Vec<f64>>();
        display::full_quasiparticles(&save_energies, occ_size);
        return save_energies;
    } else if use_hermite {
        let hermite_coeff =
            fourier_self_energy::define_hermite_series(&qp_ctrl);
        println!("Hermite Self Energy enabled for second round (low-rank, linearized GW)");

        let save_energies: Vec<f64> = (0..num_state_qp_range)
            .map(|n| {
                let origin = save_energies_no_se[n];
                let side = if n >= occ_size { 1.0 } else { -1.0 };
                let omega_shifted = scf_data.eigenvalues[0][n] - delta * side;

                let sigma_with_hse = |omega: f64| -> f64 {
                    let contour = contour_rayon_lowrank(
                        omega, n,
                        &quasiparticle_energies_g, &quasiparticle_energies_w,
                        occ_size, vir_size, num_state,
                        &ri_mat_inner, &real_axis_vchiv, 0,
                    );
                    let imag = calculate_imag(
                        &w_c_at_freqs, num_state, n, omega,
                        &quasiparticle_energies_g, &quasiparticle_energies_w,
                    );
                    let hse = fourier_self_energy::sigma_hermite(origin, omega, &hermite_coeff);
                    contour - imag + hse
                };

                let self_energy_at_omega = sigma_with_hse(omega_shifted);
                let self_energy_plus_h = sigma_with_hse(omega_shifted + h);
                let self_energy_minus_h = sigma_with_hse(omega_shifted - h);
                let derivative = (self_energy_plus_h - self_energy_minus_h) / (2.0 * h);

                if scf_data.mol.ctrl.print_level > 1 {
                    println!("derivative at n={} (with HSE): {}", n, derivative);
                }
                let spectral_weight = (1.0 - derivative).powf(-1.0);
                println!("spectral weight at n={} (with HSE): {}", n, spectral_weight);

                let real_qp = if cancel_dfa_xc == true {
                    let mut exchange = 0.0;
                    for i in 0..homo + 1 {
                        exchange -= v_matrix[[n, i]];
                    }
                    omega_shifted
                        + (spectral_weight
                            * (self_energy_at_omega + exchange - vxc_nn[n]))
                } else {
                    omega_shifted + (spectral_weight * self_energy_at_omega)
                };
                println!(
                    "for n={}, second round (low-rank, with HSE): linearized gw yields qp energy={}",
                    n, real_qp
                );
                real_qp
            })
            .collect::<Vec<f64>>();
        display::full_quasiparticles(&save_energies, occ_size);
        return save_energies;
    }
    // Should not reach here
    panic!("Invalid self-energy correction configuration in linearized_gw_lowrank");
}