//use std::simd::num;
use crate::constants::PI;
use crate::ctrl_io::quasiparticle_methods::GwVariant;
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
use crate::ri_gw::ac::{ImaginaryAxisSample, PadeApproximant};
use num_complex::Complex64;
use std::time::Instant;
use std::cmp;

#[cfg(target_os = "linux")]
use libc::seccomp_notif;

pub fn g0w0(
    scf_data: &mut SCF,
    num_freq: usize,
    vxc_nn: &Vec<f64>,
    cancel_dfa_xc: bool,
) -> Vec<f64> {
    let qp_ctrl = scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let gw_scheme = qp_ctrl.gw_scheme.clone();

    if gw_scheme == "qp equation" {
        ri_gw::gw_calculations(scf_data, 20, &vxc_nn, cancel_dfa_xc)
    } else if gw_scheme == "linearize" {
        ri_gw::linearized_gw(scf_data, 20, &vxc_nn, cancel_dfa_xc)
    } else if gw_scheme == "x alpha" {
        ri_gw::x_alpha_gw(scf_data)
    } else if gw_scheme == "extrapolated" {
        match qp_ctrl.gw_variant {
            GwVariant::Cd => {
                gw_near_fermi_surface(scf_data, 20, &vxc_nn, qp_ctrl.gw_extrapolate_occ_threshold, qp_ctrl.gw_extrapolate_vir_threshold)
            }
            GwVariant::Ac => {
                gw_near_fermi_surface_ac(scf_data, num_freq, &vxc_nn, qp_ctrl.gw_extrapolate_occ_threshold)
            }
        }
    } else if gw_scheme == "no gw" {
        Vec::new()
    } else {
        panic!("invalid expression for gw scheme!")
    }
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
pub fn single_orbital_gw_ac(
    scf_data: &mut SCF,
    v_matrix: &MatrixFull<f64>,
    ri_ov: &MatrixFull<f64>,
    ri_row_n: &MatrixFull<f64>,
    w_c_at_freqs: &Vec<(f64, f64, MatrixFull<f64>)>,
    n: usize,
    _num_freq: usize,
    vxc_nn: f64,
) -> f64 {
    let start = Instant::now();
    let gwqp_g = scf_data.gwqp.0.clone();
    let gwqp_w = scf_data.gwqp.1.clone();
    let e_ks_n = scf_data.eigenvalues[0][n];
    let (_start_mo, num_state, occ_size, vir_size, homo, _lumo) =
        ri_gw::get_occupation_parameters(&scf_data, 'Y');

    let mut exchange = 0.0;
    for i in 0..homo + 1 {
        exchange -= v_matrix[[n, i]];
    }
    let hybrid_param = scf_data.mol.xc_data.dfa_hybrid_scf;
    let consts = scf_data.eigenvalues[0][n] + exchange * (1.0 - hybrid_param) - vxc_nn;
    let side = if n >= occ_size { 1.0 } else { -1.0 };

    let qp_ctrl = scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();

    let ac_num_samples = qp_ctrl.ac_num_samples;
    let ac_omega_max = qp_ctrl.ac_omega_max;
    let ac_eta = qp_ctrl.ac_eta;
    let ef = (gwqp_g[occ_size - 1] + gwqp_g[occ_size]) / 2.0;

    if ac_num_samples < 2 {
        panic!(
            "ac_num_samples must be >= 2 (got {})",
            ac_num_samples
        );
    }
    if ac_omega_max < 0.0 {
        panic!(
            "ac_omega_max must be > 0 (got {})",
            ac_omega_max
        );
    }

    // AC imaginary-axis sampling grid.
    // Use PySCF-style frequency selection via get_ac_idx.
    // Select ac_num_samples indices from the num_freq quadrature points,
    // skipping ω=0, with step_ratio=2/3 (PySCF default).
    let ac_step_ratio = 2.0 / 3.0;
    let ac_indices = ri_gw::ac::get_ac_idx(_num_freq, ac_num_samples, ac_step_ratio);
    let ac_freqs: Vec<f64> = ac_indices
        .iter()
        .map(|&idx| {
            // idx ranges from 1 to num_freq (1-based), mapping to quad_freqs[idx-1]
            let idx0 = idx.saturating_sub(1).min(w_c_at_freqs.len().saturating_sub(1));
            w_c_at_freqs[idx0].0
        })
        .collect();

    let samples: Vec<ImaginaryAxisSample> = ac_freqs
        .iter()
        .map(|&lambda| {
            let sigma_c = ri_gw::calculate_sigma_c_imag_freq(w_c_at_freqs, n, lambda, ef, &gwqp_g);
            ImaginaryAxisSample::new_shifted(lambda, ef, sigma_c)
        })
        .collect();

    let pade = match PadeApproximant::from_imaginary_axis(&samples) {
        Ok(p) => p,
        Err(e) => {
            panic!(
                "Pade construction failed for orbital {}: {}",
                n, e
            );
        }
    };

    let qp_eq_func = |omega: f64| -> f64 {
        match pade.evaluate_retarded(omega, ac_eta) {
            Ok(sigma_c) => consts + sigma_c.re - omega,
            Err(_) => f64::NAN,
        }
    };

    let rootfinder = qp_ctrl.gw_rootfinder.clone();
    let qp_energy_no_fse;

    if e_ks_n.abs() > qp_ctrl.gw_switch_fallback_threshold {
        // Static approximation: Σ_c evaluated at the KS energy.
        let sigma_static = match pade.evaluate_retarded(e_ks_n, ac_eta) {
            Ok(s) => s.re,
            Err(e) => panic!(
                "Pade evaluation failed for orbital {} at static fallback: {}",
                n, e
            ),
        };
        qp_energy_no_fse = consts + sigma_static;
        println!(
            "Orbital #{} (AC): |E_KS|={:.6} > gw_switch_fallback_threshold={:.6}, using static fallback QP energy={:.6}",
            n,
            e_ks_n.abs(),
            qp_ctrl.gw_switch_fallback_threshold,
            qp_energy_no_fse
        );
    } else if rootfinder == "newton".to_string() {
        println!("Orbital #{} (AC):", n);
        let wrapped_f = |omega: f64,
                          _n: usize,
                          _consts: f64,
                          _ri_ov: &MatrixFull<f64>,
                          _ri_row_n: &MatrixFull<f64>,
                          _qpg: &Vec<f64>,
                          _qpw: &Vec<f64>,
                          _occ: usize,
                          _vir: usize,
                          _ns: usize,
                          _wcf: &Vec<(f64, f64, MatrixFull<f64>)>|
         -> f64 { qp_eq_func(omega) };

        qp_energy_no_fse = ri_gw::newton_solver(
            wrapped_f,
            n,
            consts,
            ri_ov,
            ri_row_n,
            &gwqp_g,
            &gwqp_w,
            occ_size,
            vir_size,
            num_state,
            w_c_at_freqs,
            e_ks_n,
            0.00001,
            50,
            side,
            scf_data.mol.ctrl.print_level,
        );
        println!("QP energy (AC): {}", qp_energy_no_fse);
    } else if rootfinder == "interpolation".to_string() {
        println!("Orbital #{} (AC):", n);
        let (have_crossing, mut qp_energy) = ri_gw::linear_interpolation_solver(
            qp_eq_func,
            e_ks_n,
            side,
            qp_ctrl.gw_search_grid,
            qp_ctrl.gw_span_energy,
        );
        if !have_crossing {
            println!(
                "No graphical crossings found for n={} (AC), using Newton solver instead.",
                n
            );
            let wrapped_f = |omega: f64,
                              _n: usize,
                              _consts: f64,
                              _ri_ov: &MatrixFull<f64>,
                              _ri_row_n: &MatrixFull<f64>,
                              _qpg: &Vec<f64>,
                              _qpw: &Vec<f64>,
                              _occ: usize,
                              _vir: usize,
                              _ns: usize,
                              _wcf: &Vec<(f64, f64, MatrixFull<f64>)>|
             -> f64 { qp_eq_func(omega) };

            qp_energy = ri_gw::newton_solver(
                wrapped_f,
                n,
                consts,
                ri_ov,
                ri_row_n,
                &gwqp_g,
                &gwqp_w,
                occ_size,
                vir_size,
                num_state,
                w_c_at_freqs,
                e_ks_n,
                0.00001,
                50,
                side,
                scf_data.mol.ctrl.print_level,
            );
        }
        qp_energy_no_fse = qp_energy;
        println!("QP energy (AC): {}", qp_energy_no_fse);
    } else {
        panic!("Invalid choice of GW rootfinder!");
    }

    if !qp_energy_no_fse.is_finite() {
        panic!(
            "AC-GW failed to produce a finite QP energy for orbital {}",
            n
        );
    }

    if qp_ctrl.fourier_self_energy || qp_ctrl.hermite_self_energy {
        println!(
            "GW(AC) orbital #{}: Fourier/Hermite self-energy correction is not supported in AC path; using bare AC result.",
            n
        );
    }

    println!("GW of orbital #{} (AC) took {:?}", n, start.elapsed());
    if scf_data.mol.ctrl.print_level > 1 {
        println!("Shift = {}", qp_energy_no_fse - e_ks_n);
    }
    qp_energy_no_fse
}
pub fn single_orbital_gw(scf_data:&mut SCF,v_matrix:&MatrixFull<f64>,ri_ov:&MatrixFull<f64>,ri_row_n:&MatrixFull<f64>,w_c_at_freqs:&Vec<(f64,f64,MatrixFull<f64>)>,n:usize,num_freq:usize,vxc_nn:f64)->f64{
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
    //let contour=ri_gw::contour_rayon(0.0,n,&gwqp_g,&gwqp_w,occ_size,vir_size,num_state,ri_ov,ri_row_n);
    //println!("Static Self Energy(Correlation part) Sigma_c(omega=0)={}(imag={},contour={})",contour-imag,imag,contour);
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let cdgw_eta = qp_ctrl.cdgw_eta;
    let cdgw_res_tol = qp_ctrl.cdgw_res_tol;
    //let self_energy_static=ri_gw::contour_rayon(scf_data.eigenvalues[0][n],n,&gwqp_g, &gwqp_w, occ_size, vir_size, num_state,ri_ov,ri_row_n)-ri_gw::calculate_imag(w_c_at_freqs,num_state,n,scf_data.eigenvalues[0][n],&gwqp_g, &gwqp_w);
    // 第一轮：正常GW计算（不添加Fourier自能）
    let qp_eq_func_no_fse = |omega: f64| {
        ri_gw::quasiparticle_equation(omega, n, consts, ri_ov, ri_row_n, &gwqp_g, &gwqp_w, occ_size, vir_size, num_state, w_c_at_freqs, cdgw_res_tol, cdgw_eta)
    };
    //println!("Static Approximation Yields:{},Sigma_c[E_KS]={},consts={},E_ks={}",qp_eq_func_no_fse(scf_data.eigenvalues[0][n])+scf_data.eigenvalues[0][n],self_energy_static,consts,scf_data.eigenvalues[0][n]);
    let rootfinder = qp_ctrl.gw_rootfinder.clone();
    let qp_energy_no_fse;

    if e_ks_n.abs() > qp_ctrl.gw_switch_fallback_threshold {
        qp_energy_no_fse = qp_eq_func_no_fse(e_ks_n) + e_ks_n;
        println!("Orbital #{}: |E_KS|={:.6} > gw_switch_fallback_threshold={:.6}, using static fallback QP energy={:.6}",
                 n, e_ks_n.abs(), qp_ctrl.gw_switch_fallback_threshold, qp_energy_no_fse);
    } else if rootfinder == "newton".to_string() {
        println!("Orbital #{} (first round, no Fourier self-energy):", n);
        qp_energy_no_fse = ri_gw::newton_solver(
            |om, nn, cc, ov, rn, qpg, qpw, os, vs, ns, wcf| ri_gw::quasiparticle_equation(om, nn, cc, ov, rn, qpg, qpw, os, vs, ns, wcf, cdgw_res_tol, cdgw_eta),
            n, consts, ri_ov, ri_row_n,
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
                |om, nn, cc, ov, rn, qpg, qpw, os, vs, ns, wcf| ri_gw::quasiparticle_equation(om, nn, cc, ov, rn, qpg, qpw, os, vs, ns, wcf, cdgw_res_tol, cdgw_eta),
                n, consts, ri_ov, ri_row_n,
                &gwqp_g, &gwqp_w, occ_size, vir_size, num_state, w_c_at_freqs,
                e_ks_n, 0.00001, 50, side, scf_data.mol.ctrl.print_level
            );
        }
        qp_energy_no_fse = qp_energy;
        println!("First round QP energy (no FSE): {}", qp_energy_no_fse);
        let self_energy_final=ri_gw::contour_rayon(qp_energy_no_fse,n,&gwqp_g, &gwqp_w, occ_size, vir_size, num_state,ri_ov,ri_row_n,cdgw_res_tol,cdgw_eta)-ri_gw::calculate_imag(w_c_at_freqs,num_state,n,qp_energy_no_fse,&gwqp_g, &gwqp_w);
        println!("Sigma_c[E_QP]={}",self_energy_final);
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
            ri_gw::quasiparticle_equation(omega, n, consts, ri_ov, ri_row_n, &gwqp_g, &gwqp_w, occ_size, vir_size, num_state, w_c_at_freqs, cdgw_res_tol, cdgw_eta)
                + ri_gw::fourier_self_energy::fourier_series(&sin_coeff, &cos_coeff, powers, t, omega - origin)
        };

        final_qp_energy = solve_with_self_energy(
            n, qp_eq_func_with_se, qp_energy_no_fse, side, &qp_ctrl, rootfinder,
            consts, ri_ov, ri_row_n, &gwqp_g, &gwqp_w, occ_size, vir_size, num_state, w_c_at_freqs,
            scf_data.mol.ctrl.print_level, "Fourier"
        );
    } else if use_hermite {
        // Hermite自能处理
        let hermite_coeff = ri_gw::fourier_self_energy::define_hermite_series(&qp_ctrl);
        println!("Hermite Self Energy Defined for second round:\nOrigin={}, Number of coefficients={},\nhermite_coeff={:#?}",
                 origin, hermite_coeff.len(), hermite_coeff);

        let qp_eq_func_with_se = |omega: f64| {
            ri_gw::quasiparticle_equation(omega, n, consts, ri_ov, ri_row_n, &gwqp_g, &gwqp_w, occ_size, vir_size, num_state, w_c_at_freqs, cdgw_res_tol, cdgw_eta)
                + ri_gw::fourier_self_energy::sigma_hermite(origin, omega, &hermite_coeff)
        };

        final_qp_energy = solve_with_self_energy(
            n, qp_eq_func_with_se, qp_energy_no_fse, side, &qp_ctrl, rootfinder,
            consts, ri_ov, ri_row_n, &gwqp_g, &gwqp_w, occ_size, vir_size, num_state, w_c_at_freqs,
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
    ri_row_n: &tensors::MatrixFull<f64>,
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
            wrapped_f, n, consts, ri_ov, ri_row_n,
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
                wrapped_f, n, consts, ri_ov, ri_row_n,
                gwqp_g, gwqp_w, occ_size, vir_size, num_state, w_c_at_freqs,
                initial_guess, 0.00001, 50, side, print_level
            );
        }
        qp_energy
    } else {
        panic!("Invalid choice of GW rootfinder!");
    }
}

pub fn gw_near_fermi_surface_ac(
    scf_data: &mut SCF,
    num_freq: usize,
    vxc_nn: &Vec<f64>,
    threshold: f64,
) -> Vec<f64> {
    let mut ri_ov: MatrixFull<f64> = ri_bse::get_submatrix(scf_data, 'O', 'V', 'Y');
    println!("RI-OV Shape={:?}", ri_ov.size);
    let qp_ctrl = scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let start = Instant::now();
    let v_matrix = ri_gw::v_matrix_from_scf(scf_data);
    let time1 = start.elapsed();
    println!("V Matrix Constructed. This step took {:?}", time1);
    let ks_energies = scf_data.eigenvalues[0].clone();
    let (start_mo, num_state, occ_size, vir_size, homo, lumo) =
        ri_gw::get_occupation_parameters(&scf_data, 'Y');

    // AC does not use the low-rank contour or real-axis v*chi*v path.
    // The low-rank machinery (generate_w_c_lowrank, real_axis_vchiv) is
    // designed for CD and relies on residue-pole decomposition on the
    // real axis. The AC approach bypasses the real axis entirely by
    // continuing from the imaginary axis via Padé. Therefore we always
    // use the full W_c matrix on the imaginary axis (the v1 path).
    let w_c_at_freqs: Vec<(f64, f64, MatrixFull<f64>)> = if qp_ctrl.gw_imag_rayon {
        ri_gw::generate_w_c(
            scf_data,
            &ri_ov,
            &scf_data.gwqp.0,
            &scf_data.gwqp.1,
            num_state,
            occ_size,
            vir_size,
            num_freq,
        )
    } else {
        ri_gw::generate_w_c_serial(
            scf_data,
            &ri_ov,
            &scf_data.gwqp.0,
            &scf_data.gwqp.1,
            num_state,
            occ_size,
            vir_size,
            num_freq,
        )
    };

    let e_homo = ks_energies[occ_size - 1];
    let e_lumo = ks_energies[occ_size];
    let calc_orbs_indices: Vec<usize> = ks_energies
        .into_iter()
        .enumerate()
        .filter(|(n, e_n)| *e_n > e_homo - threshold && *e_n < e_lumo + threshold)
        .map(|(n, _e_n)| n)
        .collect();
    println!("calculated orbital indices:{:?}", calc_orbs_indices);

    let calc_orbs: Vec<(usize, f64)> = calc_orbs_indices
        .iter()
        .map(|&n| {
            let ri_row_n = ri_gw::compute_ri3mo_row(scf_data, n);
            (
                n,
                single_orbital_gw_ac(
                    scf_data,
                    &v_matrix,
                    &ri_ov,
                    &ri_row_n,
                    &w_c_at_freqs,
                    n,
                    num_freq,
                    vxc_nn[n],
                ),
            )
        })
        .collect();

    let occ_shift = calc_orbs[0].1 - scf_data.eigenvalues[0][calc_orbs[0].0];
    let vir_shift =
        calc_orbs[calc_orbs.len() - 1].1 - scf_data.eigenvalues[0][calc_orbs[calc_orbs.len() - 1].0];
    let mut gwqp: Vec<f64> = Vec::new();
    if scf_data.mol.ctrl.print_level > 1 {
        println!("low extrapolations:{}", calc_orbs[0].0);
        println!("calculated orbitals:{}", calc_orbs.len());
        println!(
            "high extrapolations:{}",
            num_state - 1 - calc_orbs[calc_orbs.len() - 1].0
        );
        println!("Occ Shift={}, Vir Shift={}", occ_shift, vir_shift);
    }
    for i in 0..calc_orbs[0].0 {
        gwqp.push(scf_data.eigenvalues[0][i] + occ_shift)
    }
    for i in 0..calc_orbs.len() {
        gwqp.push(calc_orbs[i].1)
    }
    for i in calc_orbs[calc_orbs.len() - 1].0 + 1..num_state {
        gwqp.push(scf_data.eigenvalues[0][i] + vir_shift)
    }
    ri_gw::display::extrapolation_quasiparticles(&gwqp, occ_size, &calc_orbs_indices, threshold, threshold);
    scf_data.gwqp = (gwqp.clone(), gwqp.clone());
    gwqp
}
pub fn gw_near_fermi_surface(scf_data:&mut SCF,num_freq:usize,vxc_nn:&Vec<f64>,occ_threshold:f64,vir_threshold:f64)->Vec<f64>{
    let mut ri_ov:MatrixFull<f64>=ri_bse::get_submatrix(scf_data,'O','V','Y');
    println!("RI-OV Shape={:?}",ri_ov.size);
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let start=Instant::now();
    let v_matrix=ri_gw::v_matrix_from_scf(scf_data);
    let time1=start.elapsed();
    println!("V Matrix Constructed. This step took {:?}",time1);
    let ks_energies=scf_data.eigenvalues[0].clone();
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=ri_gw::get_occupation_parameters(&scf_data,'Y');

    // REST_GW_IMAG_LR=0 forces v1 path (full W_c matrix on imag axis) for A/B
    // verification against the v2 imaginary-axis low-rank path.
    let imag_lr_enabled = std::env::var("REST_GW_IMAG_LR")
        .map(|v| v != "0")
        .unwrap_or(true);
    let need_full_wc = !qp_ctrl.use_low_rank_contour || !imag_lr_enabled;

    // When use_low_rank_contour is enabled (and v2 path is active), build the
    // imaginary-axis low-rank representation (w_c_lr) instead of the full W_c
    // matrix (w_c_at_freqs).
    let w_c_at_freqs: Vec<(f64, f64, MatrixFull<f64>)> = if need_full_wc {
        if qp_ctrl.gw_imag_rayon {
            ri_gw::generate_w_c(scf_data,&ri_ov,&scf_data.gwqp.0,&scf_data.gwqp.1,num_state,occ_size,vir_size,num_freq)
        } else {
            ri_gw::generate_w_c_serial(scf_data,&ri_ov,&scf_data.gwqp.0,&scf_data.gwqp.1,num_state,occ_size,vir_size,num_freq)
        }
    } else {
        Vec::new() // placeholder; v2 path uses w_c_lr instead
    };

    let w_c_lr: Option<Vec<(f64, f64, ri_gw::LowRankVChiV)>> = if qp_ctrl.use_low_rank_contour && imag_lr_enabled {
        println!("Low-rank contour (v2): Generating imaginary-axis low-rank sqrt(v)*chi*sqrt(v)...");
        Some(ri_gw::generate_w_c_lowrank(
            scf_data, &ri_ov, &scf_data.gwqp.1,
            occ_size, vir_size, num_freq, qp_ctrl.low_rank_tolerance,
        ))
    } else {
        None
    };

    // Precompute low-rank real-axis v*chi*v if enabled
    let real_axis_vchiv: Option<ri_gw::RealAxisVChiV> = if qp_ctrl.use_low_rank_contour {
        println!("Low-rank contour: Precomputing real-axis sqrt(v)*chi*sqrt(v)...");
        let gwqp_g = scf_data.gwqp.0.clone();
        let gwqp_w = scf_data.gwqp.1.clone();
        let (start_mo_2,num_state_2,occ_size_2,vir_size_2,homo_2,lumo_2)=ri_gw::get_occupation_parameters(&scf_data,'Y');
        let nsemin_demax = (occ_size_2.saturating_sub(1))
            .saturating_sub(qp_ctrl.selfenergy_state_range)
            .max(0);
        let nsemax_demax = (occ_size_2 + qp_ctrl.selfenergy_state_range)
            .min(num_state_2.saturating_sub(1));
        let grid_type = if qp_ctrl.low_rank_grid_type == "quadratic" { 1 } else { 0 };
        let pl = scf_data.mol.ctrl.print_level;
        Some(ri_gw::generate_real_axis_vchiv(
            &gwqp_g, &gwqp_w, occ_size_2, vir_size_2, num_state_2,
            &ri_ov,
            qp_ctrl.nomega_chi_real,
            nsemin_demax, nsemax_demax,
            qp_ctrl.nomega_sigma,
            qp_ctrl.step_sigma,
            qp_ctrl.low_rank_tolerance,
            grid_type,
            qp_ctrl.omega_chi_max,
            pl,
            qp_ctrl.cdgw_eta,
            qp_ctrl.cdgw_res_tol,
        ))
    } else {
        None
    };

    let e_homo=ks_energies[occ_size-1];
    let e_lumo=ks_energies[occ_size];
    let calc_orbs_indices:Vec<usize>=ks_energies.into_iter().enumerate().filter(|(n,e_n)|*e_n>e_homo-occ_threshold && *e_n<e_lumo+vir_threshold).map(|(n,e_n)|n).collect();
    println!("calculated orbital indices:{:?}",calc_orbs_indices);
    let calc_orbs:Vec<(usize,f64)>=calc_orbs_indices.iter().map(|&n|{
        let ri_row_n=ri_gw::compute_ri3mo_row(scf_data,n);
        if let (Some(ref wlr), Some(ref ra)) = (&w_c_lr, &real_axis_vchiv) {
            // v2 path: imaginary-axis low-rank with pre-computed wc_rows.
            let wc_rows = ri_gw::precompute_wc_rows_lowrank(wlr, &ri_row_n, num_state);
            (n, single_orbital_gw_lowrank_v2(scf_data, &v_matrix, &ri_ov, &ri_row_n, &wc_rows, ra, n, num_freq, vxc_nn[n]))
        } else if let Some(ref ra) = real_axis_vchiv {
            // v1 path: full W_c on imag axis + real-axis low-rank.
            (n, single_orbital_gw_lowrank(scf_data, &v_matrix, &ri_ov, &ri_row_n, &w_c_at_freqs, ra, n, num_freq, vxc_nn[n]))
        } else {
            (n, single_orbital_gw(scf_data, &v_matrix, &ri_ov, &ri_row_n, &w_c_at_freqs, n, num_freq, vxc_nn[n]))
        }
    }).collect();
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
    ri_gw::display::extrapolation_quasiparticles(&gwqp,occ_size,&calc_orbs_indices,occ_threshold,vir_threshold);
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

/// Low-rank version of single_orbital_gw for core/valence GW calculations.
/// Uses pre-computed real-axis low-rank v*chi*v for contour (residue) contributions.
fn single_orbital_gw_lowrank(
    scf_data: &mut SCF,
    v_matrix: &MatrixFull<f64>,
    ri_ov: &MatrixFull<f64>,
    ri_row_n: &MatrixFull<f64>,
    w_c_at_freqs: &Vec<(f64, f64, MatrixFull<f64>)>,
    real_axis_vchiv: &ri_gw::RealAxisVChiV,
    n: usize,
    num_freq: usize,
    vxc_nn: f64,
) -> f64 {
    let start = Instant::now();
    let gwqp_g = scf_data.gwqp.0.clone();
    let gwqp_w = scf_data.gwqp.1.clone();
    let e_ks_n = scf_data.eigenvalues[0][n];
    let (start_mo, num_state, occ_size, vir_size, homo, lumo) = ri_gw::get_occupation_parameters(&scf_data, 'Y');
    let mut exchange = 0.0;
    for i in 0..homo + 1 {
        exchange -= v_matrix[[n, i]];
    }
    let hybrid_param = scf_data.mol.xc_data.dfa_hybrid_scf;
    let side = if n >= occ_size { 1.0 } else { -1.0 };
    let consts = scf_data.eigenvalues[0][n] + exchange * (1.0 - hybrid_param) - vxc_nn;

    let qp_ctrl = scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let cdgw_eta = qp_ctrl.cdgw_eta;
    let rootfinder = qp_ctrl.gw_rootfinder.clone();
    let qp_energy_no_fse;

    if e_ks_n.abs() > qp_ctrl.gw_switch_fallback_threshold {
        let qp_eq_func = |omega: f64| {
            ri_gw::quasiparticle_equation_lowrank(
                omega, n, consts, ri_row_n, &gwqp_g, &gwqp_w,
                occ_size, vir_size, num_state,
                w_c_at_freqs, real_axis_vchiv, 0, cdgw_eta,
            )
        };
        qp_energy_no_fse = qp_eq_func(e_ks_n) + e_ks_n;
        println!("Orbital #{} (low-rank): |E_KS|={:.6} > gw_switch_fallback_threshold={:.6}, using static fallback QP energy={:.6}",
                 n, e_ks_n.abs(), qp_ctrl.gw_switch_fallback_threshold, qp_energy_no_fse);
    } else if rootfinder == "newton".to_string() {
        println!("Orbital #{} (low-rank, no FSE):", n);
        let qp_start = gwqp_g[n];
        qp_energy_no_fse = ri_gw::newton_solver_lowrank(
            n, consts, ri_row_n, &gwqp_g, &gwqp_w, occ_size, vir_size, num_state,
            w_c_at_freqs, real_axis_vchiv,
            qp_start, 0.00001, 50, side, scf_data.mol.ctrl.print_level, cdgw_eta,
        );
        println!("QP energy (low-rank, no FSE): {}", qp_energy_no_fse);
    } else if rootfinder == "interpolation".to_string() {
        // Debug print at starting point (once per state)
        // Use gwqp_g[n] as omega to ensure self-pole (de = qp_g[n] - omega = 0) is captured
        // The solver's scan uses e_ks_n as starting_point, but the self-pole FIXED POINT
        // is at omega = qp_energy (gwqp_g[n]), not the KS energy.
        let debug_omega = gwqp_g[n];
        let pl = scf_data.mol.ctrl.print_level;
        ri_gw::quasiparticle_equation_lowrank(
            debug_omega, n, consts, ri_row_n, &gwqp_g, &gwqp_w,
            occ_size, vir_size, num_state,
            w_c_at_freqs, real_axis_vchiv, pl, cdgw_eta,
        );
        let qp_eq_func = |omega: f64| {
            ri_gw::quasiparticle_equation_lowrank(
                omega, n, consts, ri_row_n, &gwqp_g, &gwqp_w,
                occ_size, vir_size, num_state,
                w_c_at_freqs, real_axis_vchiv, 0, cdgw_eta,
            )
        };
        // Use gwqp_g[n] as starting_point so the self-pole (de=0) at omega = qp_energy
        // is captured by fallback and solver scan. Without this, the half-pole is missed
        // when renormalized_singles or evGW shifts gwqp_g[n] away from e_ks_n.
        let qp_start = gwqp_g[n];
        let (have_crossing, mut qp_energy) = ri_gw::linear_interpolation_solver(
            qp_eq_func, qp_start, side, qp_ctrl.gw_search_grid, qp_ctrl.gw_span_energy,
        );
        // let static_result=qp_eq_func(scf_data.eigenvalues[0][n])+scf_data.eigenvalues[0][n];
        // println!("Static Approximation yields: {}",static_result);
        if !have_crossing {
            println!("No graphical crossings for n={}, using Newton solver.", n);
            qp_energy = ri_gw::newton_solver_lowrank(
                n, consts, ri_row_n, &gwqp_g, &gwqp_w, occ_size, vir_size, num_state,
                w_c_at_freqs, real_axis_vchiv,
                qp_start, 0.00001, 50, side, scf_data.mol.ctrl.print_level, cdgw_eta,
            );
        }
        qp_energy_no_fse = qp_energy;
        println!("QP energy (low-rank, no FSE): {}", qp_energy_no_fse);
    } else {
        panic!("Invalid choice of GW rootfinder!");
    }

    if !qp_ctrl.fourier_self_energy && !qp_ctrl.hermite_self_energy {
        println!("GW of orbital #{} (low-rank) took {:?}", n, start.elapsed());
        return qp_energy_no_fse;
    }

    // Second round with self-energy correction
    let origin = qp_energy_no_fse;
    let final_qp;

    if qp_ctrl.fourier_self_energy {
        let (powers, t, sin_coeff, cos_coeff) = ri_gw::fourier_self_energy::define_fourier_series(&qp_ctrl);
        let qp_eq_func = |omega: f64| {
            ri_gw::quasiparticle_equation_lowrank(
                omega, n, consts, ri_row_n, &gwqp_g, &gwqp_w,
                occ_size, vir_size, num_state,                 w_c_at_freqs, real_axis_vchiv, 0, cdgw_eta,
            ) + ri_gw::fourier_self_energy::fourier_series(&sin_coeff, &cos_coeff, powers, t, omega - origin)
        };
        let (have_crossing, qp) = ri_gw::linear_interpolation_solver(
            qp_eq_func, qp_energy_no_fse, side, qp_ctrl.gw_search_grid, qp_ctrl.gw_span_energy,
        );
        final_qp = if !have_crossing {
            ri_gw::newton_solver_lowrank(
                n, consts, ri_row_n, &gwqp_g, &gwqp_w, occ_size, vir_size, num_state,
                w_c_at_freqs, real_axis_vchiv,
                qp_energy_no_fse, 0.00001, 50, side, scf_data.mol.ctrl.print_level, cdgw_eta,
            )
        } else {
            qp
        };
    } else if qp_ctrl.hermite_self_energy {
        let hermite_coeff = ri_gw::fourier_self_energy::define_hermite_series(&qp_ctrl);
        let qp_eq_func = |omega: f64| {
            ri_gw::quasiparticle_equation_lowrank(
                omega, n, consts, ri_row_n, &gwqp_g, &gwqp_w,
                occ_size, vir_size, num_state,                 w_c_at_freqs, real_axis_vchiv, 0, cdgw_eta,
            ) + ri_gw::fourier_self_energy::sigma_hermite(origin, omega, &hermite_coeff)
        };
        let (have_crossing, qp) = ri_gw::linear_interpolation_solver(
            qp_eq_func, qp_energy_no_fse, side, qp_ctrl.gw_search_grid, qp_ctrl.gw_span_energy,
        );
        final_qp = if !have_crossing {
            ri_gw::newton_solver_lowrank(
                n, consts, ri_row_n, &gwqp_g, &gwqp_w, occ_size, vir_size, num_state,
                w_c_at_freqs, real_axis_vchiv,
                qp_energy_no_fse, 0.00001, 50, side, scf_data.mol.ctrl.print_level, cdgw_eta,
            )
        } else {
            qp
        };
    } else {
        panic!("No self-energy correction enabled but reached second round!");
    }

    println!("GW of orbital #{} (low-rank) took {:?}, QP={}", n, start.elapsed(), final_qp);
    final_qp
}

/// v2 low-rank version of single_orbital_gw: uses pre-computed imaginary-axis
/// wc_rows (from generate_w_c_lowrank + precompute_wc_rows_lowrank) in addition
/// to the pre-computed real-axis low-rank v*chi*v. Identical math to
/// single_orbital_gw_lowrank; only the imaginary-axis integration path changes
/// (calculate_imag → calculate_imag_from_rows, quasiparticle_equation_lowrank
/// → _v2, newton_solver_lowrank → _v2).
fn single_orbital_gw_lowrank_v2(
    scf_data: &mut SCF,
    v_matrix: &MatrixFull<f64>,
    ri_ov: &MatrixFull<f64>,
    ri_row_n: &MatrixFull<f64>,
    wc_rows: &Vec<(f64, f64, Vec<f64>)>,
    real_axis_vchiv: &ri_gw::RealAxisVChiV,
    n: usize,
    num_freq: usize,
    vxc_nn: f64,
) -> f64 {
    let start = Instant::now();
    let gwqp_g = scf_data.gwqp.0.clone();
    let gwqp_w = scf_data.gwqp.1.clone();
    let e_ks_n = scf_data.eigenvalues[0][n];
    let (start_mo, num_state, occ_size, vir_size, homo, lumo) = ri_gw::get_occupation_parameters(&scf_data, 'Y');
    let mut exchange = 0.0;
    for i in 0..homo + 1 {
        exchange -= v_matrix[[n, i]];
    }
    let hybrid_param = scf_data.mol.xc_data.dfa_hybrid_scf;
    let side = if n >= occ_size { 1.0 } else { -1.0 };
    let consts = scf_data.eigenvalues[0][n] + exchange * (1.0 - hybrid_param) - vxc_nn;

    let qp_ctrl = scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let cdgw_eta = qp_ctrl.cdgw_eta;
    let rootfinder = qp_ctrl.gw_rootfinder.clone();
    let qp_energy_no_fse;

    if e_ks_n.abs() > qp_ctrl.gw_switch_fallback_threshold {
        let qp_eq_func = |omega: f64| {
            ri_gw::quasiparticle_equation_lowrank_v2(
                omega, n, consts, ri_row_n, &gwqp_g, &gwqp_w,
                occ_size, vir_size, num_state,
                wc_rows, real_axis_vchiv, 0, cdgw_eta,
            )
        };
        qp_energy_no_fse = qp_eq_func(e_ks_n) + e_ks_n;
        println!("Orbital #{} (low-rank v2): |E_KS|={:.6} > gw_switch_fallback_threshold={:.6}, using static fallback QP energy={:.6}",
                 n, e_ks_n.abs(), qp_ctrl.gw_switch_fallback_threshold, qp_energy_no_fse);
    } else if rootfinder == "newton".to_string() {
        println!("Orbital #{} (low-rank v2, no FSE):", n);
        let qp_start = gwqp_g[n];
        qp_energy_no_fse = ri_gw::newton_solver_lowrank_v2(
            n, consts, ri_row_n, &gwqp_g, &gwqp_w, occ_size, vir_size, num_state,
            wc_rows, real_axis_vchiv,
            qp_start, 0.00001, 50, side, scf_data.mol.ctrl.print_level, cdgw_eta,
        );
        println!("QP energy (low-rank v2, no FSE): {}", qp_energy_no_fse);
    } else if rootfinder == "interpolation".to_string() {
        let debug_omega = gwqp_g[n];
        let pl = scf_data.mol.ctrl.print_level;
        ri_gw::quasiparticle_equation_lowrank_v2(
            debug_omega, n, consts, ri_row_n, &gwqp_g, &gwqp_w,
            occ_size, vir_size, num_state,
            wc_rows, real_axis_vchiv, pl, cdgw_eta,
        );
        let qp_eq_func = |omega: f64| {
            ri_gw::quasiparticle_equation_lowrank_v2(
                omega, n, consts, ri_row_n, &gwqp_g, &gwqp_w,
                occ_size, vir_size, num_state,
                wc_rows, real_axis_vchiv, 0, cdgw_eta,
            )
        };
        let qp_start = gwqp_g[n];
        let (have_crossing, mut qp_energy) = ri_gw::linear_interpolation_solver(
            qp_eq_func, qp_start, side, qp_ctrl.gw_search_grid, qp_ctrl.gw_span_energy,
        );
        if !have_crossing {
            println!("No graphical crossings for n={}, using Newton solver.", n);
            qp_energy = ri_gw::newton_solver_lowrank_v2(
                n, consts, ri_row_n, &gwqp_g, &gwqp_w, occ_size, vir_size, num_state,
                wc_rows, real_axis_vchiv,
                qp_start, 0.00001, 50, side, scf_data.mol.ctrl.print_level, cdgw_eta,
            );
        }
        qp_energy_no_fse = qp_energy;
        println!("QP energy (low-rank v2, no FSE): {}", qp_energy_no_fse);
    } else {
        panic!("Invalid choice of GW rootfinder!");
    }

    if !qp_ctrl.fourier_self_energy && !qp_ctrl.hermite_self_energy {
        println!("GW of orbital #{} (low-rank v2) took {:?}", n, start.elapsed());
        return qp_energy_no_fse;
    }

    // Second round with self-energy correction
    let origin = qp_energy_no_fse;
    let final_qp;

    if qp_ctrl.fourier_self_energy {
        let (powers, t, sin_coeff, cos_coeff) = ri_gw::fourier_self_energy::define_fourier_series(&qp_ctrl);
        let qp_eq_func = |omega: f64| {
            ri_gw::quasiparticle_equation_lowrank_v2(
                omega, n, consts, ri_row_n, &gwqp_g, &gwqp_w,
                occ_size, vir_size, num_state, wc_rows, real_axis_vchiv, 0, cdgw_eta,
            ) + ri_gw::fourier_self_energy::fourier_series(&sin_coeff, &cos_coeff, powers, t, omega - origin)
        };
        let (have_crossing, qp) = ri_gw::linear_interpolation_solver(
            qp_eq_func, qp_energy_no_fse, side, qp_ctrl.gw_search_grid, qp_ctrl.gw_span_energy,
        );
        final_qp = if !have_crossing {
            ri_gw::newton_solver_lowrank_v2(
                n, consts, ri_row_n, &gwqp_g, &gwqp_w, occ_size, vir_size, num_state,
                wc_rows, real_axis_vchiv,
                qp_energy_no_fse, 0.00001, 50, side, scf_data.mol.ctrl.print_level, cdgw_eta,
            )
        } else {
            qp
        };
    } else if qp_ctrl.hermite_self_energy {
        let hermite_coeff = ri_gw::fourier_self_energy::define_hermite_series(&qp_ctrl);
        let qp_eq_func = |omega: f64| {
            ri_gw::quasiparticle_equation_lowrank_v2(
                omega, n, consts, ri_row_n, &gwqp_g, &gwqp_w,
                occ_size, vir_size, num_state, wc_rows, real_axis_vchiv, 0, cdgw_eta,
            ) + ri_gw::fourier_self_energy::sigma_hermite(origin, omega, &hermite_coeff)
        };
        let (have_crossing, qp) = ri_gw::linear_interpolation_solver(
            qp_eq_func, qp_energy_no_fse, side, qp_ctrl.gw_search_grid, qp_ctrl.gw_span_energy,
        );
        final_qp = if !have_crossing {
            ri_gw::newton_solver_lowrank_v2(
                n, consts, ri_row_n, &gwqp_g, &gwqp_w, occ_size, vir_size, num_state,
                wc_rows, real_axis_vchiv,
                qp_energy_no_fse, 0.00001, 50, side, scf_data.mol.ctrl.print_level, cdgw_eta,
            )
        } else {
            qp
        };
    } else {
        panic!("No self-energy correction enabled but reached second round!");
    }

    println!("GW of orbital #{} (low-rank v2) took {:?}, QP={}", n, start.elapsed(), final_qp);
    final_qp
}
