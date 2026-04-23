//use std::simd::num;
use crate::constants::PI;
use itertools::Itertools;
use std::ops::Range;
use crate::utilities;
use crate::scf_io::SCF;
//use std::slice::Iter::<'_, f64>;
use rayon::result;
use reqwest::blocking::Response;
use rest_tensors::{RIFull};
use tensors::{matrix_blas_lapack::{_dinverse,_dsyev}, ri, MathMatrix, MatrixFull};
//use rest::molecule_io::Molecule;
use rest_tensors::matrix::matrix_blas_lapack::{_dgees,_dgemm_full,_dgemv};
use rest_tensors::MatrixUpper;
use crate::ri_bse;
use crate::ri_rpa;
use crate::molecule_io;
use crate::scf_io;
use crate::dft;
use crate::dft::DFA4REST;
use std::sync::mpsc::channel;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use crate::ri_gw;
use std::time::Instant;
use std::cmp;

#[cfg(target_os = "linux")]
use libc::seccomp_notif;

pub fn g0w0(scf_data:&mut SCF,num_freq:usize,vxc_nn:&Vec<f64>,cancel_dfa_xc:bool)->Vec<f64>{
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let gw_scheme=qp_ctrl.gw_scheme.clone();
    if gw_scheme=="qp equation"{
        ri_gw::gw_calculations(scf_data,20,&vxc_nn,cancel_dfa_xc)
    }else if gw_scheme=="linearize"{
        ri_gw::linearized_gw(scf_data,20,&vxc_nn,cancel_dfa_xc)
    }else if gw_scheme=="x alpha"{
        ri_gw::x_alpha_gw(scf_data)
    }else if gw_scheme=="extrapolated"{
        gw_near_fermi_surface(scf_data,20,&vxc_nn,qp_ctrl.threshold)
    }
    else if gw_scheme=="no gw"{
        Vec::new()
    }else {panic!("invalid expression for gw scheme!")}
}
pub fn evgw(scf_data:&mut SCF,num_freq:usize,vxc_nn:&Vec<f64>,iter_rounds:usize)->Vec<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=ri_gw::get_occupation_parameters(scf_data,'Y');
    for i in 0..iter_rounds{
        println!("Now is round #{} of evGW calculation. There will be {} rounds in total.",i+1,iter_rounds);
        //let cancel_dfa_xc=if i==0 && scf_data.mol.ctrl.renormalized_singles==false{true}else{false};
        let cancel_dfa_xc=true;
        let previous_qp=scf_data.gwqp.0.clone();
        let qp=g0w0(scf_data,num_freq,vxc_nn,cancel_dfa_xc);
        scf_data.gwqp.0=qp.clone();
        scf_data.gwqp.1=qp.clone();
    }
    scf_data.gwqp.0.clone()
}
pub fn single_orbital_gw(scf_data:&mut SCF,v_matrix:&MatrixFull<f64>,ri_ov:&MatrixFull<f64>,ri_mat:&MatrixFull<f64>,w_c_at_freqs:&Vec<(f64,f64,MatrixFull<f64>)>,n:usize,num_freq:usize,vxc_nn:f64)->f64{
    let start=Instant::now();
    let mut exchange=0.0;
    let gwqp_g=scf_data.gwqp.0.clone();
    let gwqp_w=scf_data.gwqp.1.clone();
    let e_ks_n=scf_data.eigenvalues[0][n];
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=ri_gw::get_occupation_parameters(&scf_data,'Y');
    for i in 0..homo+1{
        exchange-=v_matrix[[n,i]];
    }
    let hybrid_param=scf_data.mol.xc_data.dfa_hybrid_scf;
    println!("Exchange={},V_xc={}",exchange,vxc_nn+hybrid_param*exchange);
    let side=if n>=occ_size{1.0}else{-1.0};
    let consts=scf_data.eigenvalues[0][n]+exchange*(1.0-hybrid_param)-vxc_nn;
    //let imag=ri_gw::calculate_imag(&w_c_at_freqs,num_state,n,0.0,&gwqp_g,&gwqp_w);
    //let contour=ri_gw::contour_rayon(0.0,n,&gwqp_g,&gwqp_w,occ_size,vir_size,num_state,ri_ov,ri_mat);
    //println!("Static Self Energy(Correlation part) Sigma_c(omega=0)={}(imag={},contour={})",contour-imag,imag,contour);
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();

    // 第一轮：正常GW计算（不添加Fourier自能）
    let qp_eq_func_no_fse = |omega: f64| {
        ri_gw::quasiparticle_equation(omega, n, consts, ri_ov, ri_mat, &gwqp_g, &gwqp_w, occ_size, vir_size, num_state, w_c_at_freqs)
    };

    let rootfinder = qp_ctrl.gw_rootfinder.clone();
    let qp_energy_no_fse;

    if rootfinder == "newton".to_string() {
        println!("Orbital #{} (first round, no Fourier self-energy):", n);
        qp_energy_no_fse = ri_gw::newton_solver(
            ri_gw::quasiparticle_equation, n, consts, ri_ov, ri_mat,
            &gwqp_g, &gwqp_w, occ_size, vir_size, num_state, w_c_at_freqs,
            e_ks_n, 0.00001, 50, side, scf_data.mol.ctrl.print_level
        );
        println!("First round QP energy (no FSE): {}", qp_energy_no_fse);
    } else if rootfinder == "interpolation".to_string() {
        println!("Orbital #{} (first round, no Fourier self-energy):", n);
        let (have_crossing, mut qp_energy) = ri_gw::linear_interpolation_solver(
            qp_eq_func_no_fse, e_ks_n, side, qp_ctrl.gw_search_grid, qp_ctrl.gw_span_energy
        );
        if !have_crossing {
            println!("No graphical crossings found for n={}, using Newton solver instead.", n);
            qp_energy = ri_gw::newton_solver(
                ri_gw::quasiparticle_equation, n, consts, ri_ov, ri_mat,
                &gwqp_g, &gwqp_w, occ_size, vir_size, num_state, w_c_at_freqs,
                e_ks_n, 0.00001, 50, side, scf_data.mol.ctrl.print_level
            );
        }
        qp_energy_no_fse = qp_energy;
        println!("First round QP energy (no FSE): {}", qp_energy_no_fse);
    } else {
        panic!("Invalid choice of GW rootfinder!");
    }

    // 检查是否启用了自能校正
    let use_fourier = qp_ctrl.fourier_self_energy;
    let use_hermite = qp_ctrl.hermite_self_energy;

    // 确保不会同时启用Fourier和Hermite自能
    if use_fourier && use_hermite {
        panic!("Cannot enable both Fourier self-energy and Hermite self-energy simultaneously!");
    }

    // 如果不需要任何自能校正，直接返回第一轮结果
    if !use_fourier && !use_hermite {
        println!("GW Evaluation of orbital #{} took {:?}", n, start.elapsed());
        if scf_data.mol.ctrl.print_level > 1 {
            println!("Shift = {}", qp_energy_no_fse - e_ks_n);
        }
        return qp_energy_no_fse;
    }

    // 第二轮：在正常GW自能基础上添加自能校正，以第一轮QP能量为origin
    let origin = qp_energy_no_fse;

    let final_qp_energy;

    if use_fourier {
        // Fourier自能处理
        let (powers, t, sin_coeff, cos_coeff) = ri_gw::fourier_self_energy::define_fourier_series(&qp_ctrl);
        println!("Fourier Self Energy Defined for second round:\nOrigin={}, Powers={}, t={},\nsin_coeff={:#?}\ncos_coeff={:#?}",
                 origin, powers, t, sin_coeff, cos_coeff);

        let qp_eq_func_with_se = |omega: f64| {
            ri_gw::quasiparticle_equation(omega, n, consts, ri_ov, ri_mat, &gwqp_g, &gwqp_w, occ_size, vir_size, num_state, w_c_at_freqs)
                + ri_gw::fourier_self_energy::fourier_series(&sin_coeff, &cos_coeff, powers, t, omega - origin)
        };

        final_qp_energy = solve_with_self_energy(
            n, qp_eq_func_with_se, qp_energy_no_fse, side, &qp_ctrl, rootfinder,
            consts, ri_ov, ri_mat, &gwqp_g, &gwqp_w, occ_size, vir_size, num_state, w_c_at_freqs,
            scf_data.mol.ctrl.print_level, "Fourier"
        );
    } else if use_hermite {
        // Hermite自能处理
        let hermite_coeff = ri_gw::fourier_self_energy::define_hermite_series(&qp_ctrl);
        println!("Hermite Self Energy Defined for second round:\nOrigin={}, Number of coefficients={},\nhermite_coeff={:#?}",
                 origin, hermite_coeff.len(), hermite_coeff);

        let qp_eq_func_with_se = |omega: f64| {
            ri_gw::quasiparticle_equation(omega, n, consts, ri_ov, ri_mat, &gwqp_g, &gwqp_w, occ_size, vir_size, num_state, w_c_at_freqs)
                + ri_gw::fourier_self_energy::sigma_hermite(origin, omega, &hermite_coeff)
        };

        final_qp_energy = solve_with_self_energy(
            n, qp_eq_func_with_se, qp_energy_no_fse, side, &qp_ctrl, rootfinder,
            consts, ri_ov, ri_mat, &gwqp_g, &gwqp_w, occ_size, vir_size, num_state, w_c_at_freqs,
            scf_data.mol.ctrl.print_level, "Hermite"
        );
    } else {
        // 不应该到达这里，因为前面已经检查过
        panic!("No self-energy correction enabled, but reached second round!");
    }

    let self_energy_type = if use_fourier { "FSE" } else { "HSE" };
    println!("Final QP energy (with {}): {}", self_energy_type, final_qp_energy);
    println!("GW Evaluation of orbital #{} took {:?}", n, start.elapsed());
    if scf_data.mol.ctrl.print_level > 1 {
        println!("Total shift = {}", final_qp_energy - e_ks_n);
        println!("{} contribution = {}", self_energy_type, final_qp_energy - qp_energy_no_fse);
    }

    final_qp_energy
}

/// Helper function to solve QP equation with self-energy correction
fn solve_with_self_energy<F>(
    n: usize,
    qp_eq_func: F,
    initial_guess: f64,
    side: f64,
    qp_ctrl: &crate::ctrl_io::quasiparticle_methods::QuasiParticle,
    rootfinder: String,
    consts: f64,
    ri_ov: &tensors::MatrixFull<f64>,
    ri_mat: &tensors::MatrixFull<f64>,
    gwqp_g: &Vec<f64>,
    gwqp_w: &Vec<f64>,
    occ_size: usize,
    vir_size: usize,
    num_state: usize,
    w_c_at_freqs: &Vec<(f64, f64, tensors::MatrixFull<f64>)>,
    print_level: usize,
    self_energy_type: &str,
) -> f64
where
    F: Fn(f64) -> f64 + Clone,
{
    if rootfinder == "newton".to_string() {
        println!("Orbital #{} (second round, with {} self-energy):", n, self_energy_type);
        // 对于牛顿法，需要创建一个包装函数来匹配newton_solver的签名
        let wrapped_f = |omega: f64, n_local: usize, consts_local: f64, ri_ov_local: &tensors::MatrixFull<f64>, ri_full_local: &tensors::MatrixFull<f64>,
                         quasiparticle_energies_g_local: &Vec<f64>, quasiparticle_energies_w_local: &Vec<f64>,
                         occ_size_local: usize, vir_size_local: usize, num_state_local: usize,
                         w_c_at_freqs_local: &Vec<(f64, f64, tensors::MatrixFull<f64>)>| -> f64 {
            // 忽略传入的参数，使用闭包捕获的变量
            qp_eq_func(omega)
        };

        crate::ri_gw::newton_solver(
            wrapped_f, n, consts, ri_ov, ri_mat,
            gwqp_g, gwqp_w, occ_size, vir_size, num_state, w_c_at_freqs,
            initial_guess, 0.00001, 50, side, print_level
        )
    } else if rootfinder == "interpolation".to_string() {
        println!("Orbital #{} (second round, with {} self-energy):", n, self_energy_type);
        // 克隆qp_eq_func以便在linear_interpolation_solver中使用，保留原始版本供后续使用
        let qp_eq_func_cloned = qp_eq_func.clone();
        let (have_crossing, mut qp_energy) = crate::ri_gw::linear_interpolation_solver(
            qp_eq_func_cloned, initial_guess, side, qp_ctrl.gw_search_grid, qp_ctrl.gw_span_energy
        );
        if !have_crossing {
            println!("No graphical crossings found for n={} (with {}), using Newton solver instead.", n, self_energy_type);
            let wrapped_f = |omega: f64, n_local: usize, consts_local: f64, ri_ov_local: &tensors::MatrixFull<f64>, ri_full_local: &tensors::MatrixFull<f64>,
                             quasiparticle_energies_g_local: &Vec<f64>, quasiparticle_energies_w_local: &Vec<f64>,
                             occ_size_local: usize, vir_size_local: usize, num_state_local: usize,
                             w_c_at_freqs_local: &Vec<(f64, f64, tensors::MatrixFull<f64>)>| -> f64 {
                qp_eq_func(omega)
            };

            qp_energy = crate::ri_gw::newton_solver(
                wrapped_f, n, consts, ri_ov, ri_mat,
                gwqp_g, gwqp_w, occ_size, vir_size, num_state, w_c_at_freqs,
                initial_guess, 0.00001, 50, side, print_level
            );
        }
        qp_energy
    } else {
        panic!("Invalid choice of GW rootfinder!");
    }
}

pub fn gw_near_fermi_surface(scf_data:&mut SCF,num_freq:usize,vxc_nn:&Vec<f64>,threshold:f64)->Vec<f64>{
    //println!("Check that gwqp is RS or not: GWQP_G-E_KS[4]={},GWQP_W-E_KS[4]={},GWQP_G-E_KS[4]={}",scf_data.gwqp.0[4]-scf_data.eigenvalues[0][4],scf_data.gwqp.1[4]-scf_data.eigenvalues[0][4],scf_data.gwqp.0[4]-scf_data.renormalized_singles_particles[4]);
    let mut ri_ov:MatrixFull<f64>=ri_bse::get_submatrix(scf_data,'O','V','Y');
    println!("RI-OV Shape={:?}",ri_ov.size);
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let mut ri_mat:MatrixFull<f64>=ri_bse::get_submatrix(scf_data,'F','F','Y');
    // if qp_ctrl.simplified_bse==true{
    //     let ang_momentum=if qp_ctrl.simplified_bse==true{cmp::min(qp_ctrl.bse_max_ang_momentum,6)}else{6};
    //     let elements=scf_data.mol.geom.elem.clone();
    //     let relevant_indices=ri_bse::sbse::obtain_relevant_indices(scf_data,&elements,ang_momentum);
    //     ri_ov=ri_bse::sbse::obtain_ri_with_reduced_ang_momentum(&ri_ov,&relevant_indices);
    //     ri_mat=ri_bse::sbse::obtain_ri_with_reduced_ang_momentum(&ri_mat,&relevant_indices);
    // }
    let start=Instant::now();
    let v_matrix=ri_gw::v_matrix(&scf_data,&ri_mat);
    let time1=start.elapsed();
    println!("V Matrix Constructed. This step took {:?}",time1);
    let v_matrix=ri_gw::v_matrix_old(&scf_data,&ri_mat);
    println!("As comparison, previous version of this step took {:?}",start.elapsed()-time1);
    let ks_energies=scf_data.eigenvalues[0].clone();
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=ri_gw::get_occupation_parameters(&scf_data,'Y');
    let w_c_at_freqs=if qp_ctrl.gw_imag_rayon{
        ri_gw::generate_w_c(scf_data,&ri_ov,&ri_mat,&scf_data.gwqp.0,&scf_data.gwqp.1,num_state,occ_size,vir_size,num_freq)
    }else{
        ri_gw::generate_w_c_serial(scf_data,&ri_ov,&ri_mat,&scf_data.gwqp.0,&scf_data.gwqp.1,num_state,occ_size,vir_size,num_freq)
    };
    let e_homo=ks_energies[occ_size-1];
    let e_lumo=ks_energies[occ_size];
    let calc_orbs_indices:Vec<usize>=ks_energies.into_iter().enumerate().filter(|(n,e_n)|*e_n>e_homo-threshold && *e_n<e_lumo+threshold).map(|(n,e_n)|n).collect();
    println!("calculated orbital indices:{:?}",calc_orbs_indices);
    let calc_orbs:Vec<(usize,f64)>=calc_orbs_indices.iter().map(|&n|(n,single_orbital_gw(scf_data,&v_matrix,&ri_ov,&ri_mat,&w_c_at_freqs,n,num_freq,vxc_nn[n]))).collect();
    let occ_shift=calc_orbs[0].1-scf_data.eigenvalues[0][calc_orbs[0].0];
    let vir_shift=calc_orbs[calc_orbs.len()-1].1-scf_data.eigenvalues[0][calc_orbs[calc_orbs.len()-1].0];
    let mut gwqp:Vec<f64>=Vec::new();
    if scf_data.mol.ctrl.print_level>1{
        println!("low extrapolations:{}",calc_orbs[0].0);
        println!("calculated orbitals:{}",calc_orbs.len());
        println!("high extrapolations:{}",num_state-1-calc_orbs[calc_orbs.len()-1].0);
        println!("Occ Shift={}, Vir Shift={}",occ_shift,vir_shift);
    }
    for i in 0 .. calc_orbs[0].0{
        gwqp.push(scf_data.eigenvalues[0][i]+occ_shift)
    }
    for i in 0..calc_orbs.len(){
        gwqp.push(calc_orbs[i].1)
    }
    for i in calc_orbs[calc_orbs.len()-1].0+1 .. num_state{
        gwqp.push(scf_data.eigenvalues[0][i]+vir_shift)
    }
    //println!("extrapolated GW Results:{:#?}",gwqp);
    ri_gw::display::extrapolation_quasiparticles(&gwqp,occ_size,&calc_orbs_indices,threshold);
    scf_data.gwqp=(gwqp.clone(),gwqp.clone());
    return gwqp
}
pub fn prepare_gwqp(scf_data:&mut SCF){
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=ri_gw::get_occupation_parameters(&scf_data,'Y');
    let mut quasiparticle_energies_g:Vec<f64> = vec![];
    let mut quasiparticle_energies_w:Vec<f64> = vec![];
    let eigenenergies:Vec<f64>=scf_data.eigenvalues[0].clone();
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
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
    scf_data.gwqp=(quasiparticle_energies_g,quasiparticle_energies_w)
}