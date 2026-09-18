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
pub fn g0w0_spin(
    scf_data: &mut SCF,
    num_freq: usize,
    vxc_nn: &[Vec<f64>;2],
    cancel_dfa_xc: bool,
) -> [Vec<f64>;2] {
    let qp_ctrl = scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let gw_scheme = qp_ctrl.gw_scheme.clone();
    if gw_scheme == "extrapolated" {
        match qp_ctrl.gw_variant {
            GwVariant::Cd => gw_near_fermi_surface_spin(
                scf_data,
                num_freq,
                vxc_nn,
                qp_ctrl.gw_extrapolate_occ_threshold,
                qp_ctrl.gw_extrapolate_vir_threshold,
                cancel_dfa_xc,
            ),
            _ => panic!("g0w0_spin: only CD variant is currently supported in the unrestricted path"),
        }
    } else {
        panic!("g0w0_spin: only gw_scheme=extrapolated is currently supported in the unrestricted path");
    }
}

pub fn single_orbital_gw_spin(
    scf_data: &SCF,
    v_matrix: &MatrixFull<f64>,
    ri_ov: &[MatrixFull<f64>;2],
    ri_row_n: &MatrixFull<f64>,
    w_c_at_freqs: &Vec<ri_gw::WcAtFreq>,
    n: usize,
    spin: usize,
    vxc_nn: f64,
) -> f64 {
    let qp_ctrl = scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let gwqp_g_all = scf_data.gwqp_spin.0.clone();
    let gwqp_w_all = scf_data.gwqp_spin.1.clone();
    let gwqp_g = &gwqp_g_all[spin];
    let ops = ri_gw::get_occ_params_per_spin(scf_data, 'Y');
    let op = ops[spin];
    let e_ks_n = scf_data.eigenvalues[spin][n];
    let mut exchange = 0.0;
    for i_local in 0..op.occ_size {
        exchange -= v_matrix[[n, op.start_mo + i_local]];
    }
    let hybrid_param = scf_data.mol.xc_data.dfa_hybrid_scf;
    let side = if n > op.homo { 1.0 } else { -1.0 };
    let consts = scf_data.eigenvalues[spin][n] + exchange * (1.0 - hybrid_param) - vxc_nn;
    let cdgw_eta = qp_ctrl.cdgw_eta;
    let cdgw_res_tol = qp_ctrl.cdgw_res_tol;
    if qp_ctrl.fourier_self_energy || qp_ctrl.hermite_self_energy {
        panic!("Unrestricted GW with Fourier/Hermite self-energy is not implemented yet");
    }

    let qp_eq_func = |omega: f64| {
        ri_gw::quasiparticle_equation_spin(
            scf_data, omega, n, spin, consts,
            gwqp_g, &gwqp_w_all, &ops, ri_ov, ri_row_n,
            w_c_at_freqs, cdgw_res_tol, cdgw_eta,
        )
    };

    if qp_ctrl.gw_rootfinder == "newton" {
        let mut x = ri_gw::single_newton_step(&qp_eq_func, e_ks_n, side, 0.2);
        for _ in 0..20 {
            let xnew = ri_gw::single_newton_step(&qp_eq_func, x, side, 0.2);
            if (xnew - x).abs() < 1.0e-8 { x = xnew; break; }
            x = xnew;
        }
        println!("Spin {} orbital #{}: QP energy = {}", spin, n, x);
        x
    } else {
        let (have_crossing, mut qp) = ri_gw::linear_interpolation_solver(
            qp_eq_func, e_ks_n, side, qp_ctrl.gw_search_grid, qp_ctrl.gw_span_energy,
        );
        if !have_crossing {
            qp = ri_gw::single_newton_step(&qp_eq_func, e_ks_n, side, 0.2);
        }
        println!("Spin {} orbital #{}: QP energy = {}", spin, n, qp);
        qp
    }
}

fn single_orbital_gw_lowrank_spin(
    scf_data: &SCF,
    v_matrix: &MatrixFull<f64>,
    ri_row_n: &MatrixFull<f64>,
    wc_rows: &Vec<(f64, f64, Vec<f64>)>,
    real_axis_vchiv: &ri_gw::RealAxisVChiV,
    n: usize,
    spin: usize,
    vxc_nn: f64,
) -> f64 {
    let qp_ctrl = scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let ops = ri_gw::get_occ_params_per_spin(scf_data, 'Y');
    let op = ops[spin];
    let gwqp_g = &scf_data.gwqp_spin.0[spin];
    let e_ks_n = scf_data.eigenvalues[spin][n];
    let mut exchange = 0.0;
    for i_local in 0..op.occ_size {
        exchange -= v_matrix[[n, op.start_mo + i_local]];
    }
    let hybrid_param = scf_data.mol.xc_data.dfa_hybrid_scf;
    let side = if n > op.homo { 1.0 } else { -1.0 };
    let consts = scf_data.eigenvalues[spin][n] + exchange * (1.0 - hybrid_param) - vxc_nn;
    let res_tol = qp_ctrl.cdgw_res_tol;

    let qp_eq_func = |omega: f64| {
        ri_gw::quasiparticle_equation_lowrank_spin_v2(
            scf_data,
            omega,
            n,
            spin,
            consts,
            gwqp_g,
            &ops,
            ri_row_n,
            wc_rows,
            real_axis_vchiv,
            res_tol,
        )
    };

    // A small Newton solver that mirrors `newton_solver_lowrank_v2` but uses
    // the spin-resolved low-rank QP equation. This is more robust for deep/high
    // orbitals than a fixed number of single Newton steps.
    let h = 1.0e-6;
    let delta = 0.02;
    let mut solve_newton = |start: f64| -> f64 {
        let mut x = start + side * delta;
        let mut y_curr = qp_eq_func(x);
        let mut y_plus = qp_eq_func(x + h);
        let mut y_minus = qp_eq_func(x - h);
        let mut converge = 0;
        for _iter in 0..50 {
            let derivative = (y_plus - y_minus) / (2.0 * h);
            let shift = -y_curr / derivative;
            x += shift;
            y_curr = qp_eq_func(x);
            if shift.abs() < 1.0e-8 {
                converge += 1;
            }
            y_plus = qp_eq_func(x + h);
            y_minus = qp_eq_func(x - h);
            if converge == 1 {
                break;
            }
        }
        if converge == 0 {
            println!("warning!!! low-rank spin Newton did not converge for spin {} orbital {}", spin, n);
        }
        x
    };

    let qp = if qp_ctrl.gw_rootfinder == "newton" {
        solve_newton(e_ks_n)
    } else {
        let (have_crossing, mut qp) = ri_gw::linear_interpolation_solver(
            &qp_eq_func,
            e_ks_n,
            side,
            qp_ctrl.gw_search_grid,
            qp_ctrl.gw_span_energy,
        );
        if !have_crossing {
            qp = solve_newton(e_ks_n);
        }
        qp
    };
    println!("Spin {} orbital #{}: QP energy (low-rank) = {}", spin, n, qp);
    qp
}


pub fn gw_near_fermi_surface_spin(
    scf_data: &mut SCF,
    num_freq: usize,
    vxc_nn: &[Vec<f64>;2],
    occ_threshold: f64,
    vir_threshold: f64,
    _cancel_dfa_xc: bool,
) -> [Vec<f64>;2] {
    let qp_ctrl = scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let nspin = scf_data.mol.spin_channel;
    if qp_ctrl.use_low_rank_contour {
        return gw_near_fermi_surface_spin_lowrank(
            scf_data,
            num_freq,
            vxc_nn,
            occ_threshold,
            vir_threshold,
        );
    }
    let ops = ri_gw::get_occ_params_per_spin(scf_data, 'Y');
    let ri_ov0 = ri_bse::get_submatrix_spin(scf_data, 'O', 'V', 'Y', 0);
    let ri_ov1 = if nspin == 2 { ri_bse::get_submatrix_spin(scf_data, 'O', 'V', 'Y', 1) } else { ri_ov0.clone() };
    let ri_ov = [ri_ov0, ri_ov1];
    let gwqp_w_all = scf_data.gwqp_spin.1.clone();
    let w_c_at_freqs = ri_gw::generate_w_c_spin(scf_data, &ri_ov, &gwqp_w_all, &ops, num_freq);

    let mut out = [vec![], vec![]];
    for spin in 0..nspin {
        let op = ops[spin];
        let v_matrix = ri_gw::v_matrix_from_scf_spin(scf_data, spin);
        let ks_energies = scf_data.eigenvalues[spin].clone();
        let e_homo = ks_energies[op.homo];
        let e_lumo = ks_energies[op.lumo];
        let calc_orbs_indices: Vec<usize> = ks_energies.iter().enumerate()
            .filter(|(_, e_n)| **e_n > e_homo - occ_threshold && **e_n < e_lumo + vir_threshold)
            .map(|(n, _)| n).collect();
        let calc_orbs: Vec<(usize, f64)> = calc_orbs_indices.iter().map(|&n| {
            let ri_row_n = ri_gw::compute_ri3mo_row_spin(scf_data, n, spin);
            (n, single_orbital_gw_spin(scf_data, &v_matrix, &ri_ov, &ri_row_n, &w_c_at_freqs, n, spin, vxc_nn[spin][n]))
        }).collect();

        let occ_shift = calc_orbs[0].1 - scf_data.eigenvalues[spin][calc_orbs[0].0];
        let vir_shift = calc_orbs[calc_orbs.len() - 1].1 - scf_data.eigenvalues[spin][calc_orbs[calc_orbs.len() - 1].0];
        let mut gwqp: Vec<f64> = Vec::new();
        for i in 0..calc_orbs[0].0 {
            gwqp.push(scf_data.eigenvalues[spin][i] + occ_shift)
        }
        for (_, e) in calc_orbs.iter() { gwqp.push(*e); }
        for i in calc_orbs[calc_orbs.len() - 1].0 + 1..op.num_state {
            gwqp.push(scf_data.eigenvalues[spin][i] + vir_shift)
        }
        out[spin] = gwqp;
    }
    for spin in 0..nspin {
        scf_data.gwqp_spin.0[spin] = out[spin].clone();
        scf_data.gwqp_spin.1[spin] = out[spin].clone();
    }
    if nspin == 1 {
        scf_data.gwqp_spin.0[1] = scf_data.gwqp_spin.0[0].clone();
        scf_data.gwqp_spin.1[1] = scf_data.gwqp_spin.1[0].clone();
        out[1] = out[0].clone();
    }
    out
}

fn gw_near_fermi_surface_spin_lowrank(
    scf_data: &mut SCF,
    num_freq: usize,
    vxc_nn: &[Vec<f64>;2],
    occ_threshold: f64,
    vir_threshold: f64,
) -> [Vec<f64>;2] {
    let qp_ctrl = scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let nspin = scf_data.mol.spin_channel;
    let ops = ri_gw::get_occ_params_per_spin(scf_data, 'Y');
    let ri_ov0 = ri_bse::get_submatrix_spin(scf_data, 'O', 'V', 'Y', 0);
    let ri_ov1 = if nspin == 2 { ri_bse::get_submatrix_spin(scf_data, 'O', 'V', 'Y', 1) } else { ri_ov0.clone() };
    let ri_ov = [ri_ov0, ri_ov1];
    let gwqp_g_all = scf_data.gwqp_spin.0.clone();
    let gwqp_w_all = scf_data.gwqp_spin.1.clone();

    println!("Unrestricted low-rank contour GW is enabled.");
    println!("Low-rank contour (unrestricted): Generating imaginary-axis total sqrt(v)*chi*sqrt(v)...");
    let w_c_lr = ri_gw::generate_w_c_lowrank_spin(
        scf_data,
        &gwqp_w_all,
        &ops,
        &ri_ov,
        num_freq,
        qp_ctrl.low_rank_tolerance,
    );

    println!("Low-rank contour (unrestricted): Precomputing real-axis total sqrt(v)*chi*sqrt(v)...");
    let mut nsemin = [0usize; 2];
    let mut nsemax = [0usize; 2];
    for s in 0..nspin {
        let op = ops[s];
        let start = op.start_mo;
        let range = qp_ctrl.selfenergy_state_range;
        nsemin[s] = start.saturating_add(op.homo.saturating_sub(start).saturating_sub(range));
        nsemax[s] = op.homo.saturating_add(range).min(op.num_state.saturating_sub(1));
    }
    let grid_type = if qp_ctrl.low_rank_grid_type == "quadratic" { 1 } else { 0 };
    let mut real_axis_vchiv = ri_gw::generate_real_axis_vchiv_spin(
        &gwqp_g_all,
        &gwqp_w_all,
        &ops,
        &ri_ov,
        qp_ctrl.nomega_chi_real,
        nsemin,
        nsemax,
        qp_ctrl.nomega_sigma,
        qp_ctrl.step_sigma,
        qp_ctrl.low_rank_tolerance,
        grid_type,
        qp_ctrl.omega_chi_max,
        scf_data.mol.ctrl.print_level,
        qp_ctrl.cdgw_eta,
        qp_ctrl.cdgw_res_tol,
    );
    real_axis_vchiv.interp = if qp_ctrl.low_rank_interp.eq_ignore_ascii_case("nearest") { 0 } else { 1 };

    let mut out = [vec![], vec![]];
    for spin in 0..nspin {
        let op = ops[spin];
        let v_matrix = ri_gw::v_matrix_from_scf_spin(scf_data, spin);
        let ks_energies = scf_data.eigenvalues[spin].clone();
        let e_homo = ks_energies[op.homo];
        let e_lumo = ks_energies[op.lumo];
        let calc_orbs_indices: Vec<usize> = ks_energies.iter().enumerate()
            .filter(|(_, e_n)| **e_n > e_homo - occ_threshold && **e_n < e_lumo + vir_threshold)
            .map(|(n, _)| n).collect();

        let calc_orbs: Vec<(usize, f64)> = calc_orbs_indices.iter().map(|&n| {
            let ri_row_n = ri_gw::compute_ri3mo_row_spin(scf_data, n, spin);
            let wc_rows = ri_gw::precompute_wc_rows_lowrank(&w_c_lr, &ri_row_n, op.num_state);
            (
                n,
                single_orbital_gw_lowrank_spin(
                    scf_data,
                    &v_matrix,
                    &ri_row_n,
                    &wc_rows,
                    &real_axis_vchiv,
                    n,
                    spin,
                    vxc_nn[spin][n],
                ),
            )
        }).collect();

        let occ_shift = calc_orbs[0].1 - scf_data.eigenvalues[spin][calc_orbs[0].0];
        let vir_shift = calc_orbs[calc_orbs.len() - 1].1 - scf_data.eigenvalues[spin][calc_orbs[calc_orbs.len() - 1].0];
        let mut gwqp: Vec<f64> = Vec::new();
        for i in 0..calc_orbs[0].0 {
            gwqp.push(scf_data.eigenvalues[spin][i] + occ_shift)
        }
        for (_, e) in calc_orbs.iter() {
            gwqp.push(*e);
        }
        for i in calc_orbs[calc_orbs.len() - 1].0 + 1..op.num_state {
            gwqp.push(scf_data.eigenvalues[spin][i] + vir_shift)
        }
        out[spin] = gwqp;
    }

    for spin in 0..nspin {
        scf_data.gwqp_spin.0[spin] = out[spin].clone();
        scf_data.gwqp_spin.1[spin] = out[spin].clone();
    }
    if nspin == 1 {
        scf_data.gwqp_spin.0[1] = scf_data.gwqp_spin.0[0].clone();
        scf_data.gwqp_spin.1[1] = scf_data.gwqp_spin.1[0].clone();
        out[1] = out[0].clone();
    }
    out
}


/// Dense linear solve with partial pivoting for the (m+1)x(m+1) Pulay system.
/// Returns `None` when the matrix is numerically singular.
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let mut piv = col;
        for r in (col + 1)..n {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1.0e-300 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in (col + 1)..n {
            let f = a[r][col] / a[col][col];
            if f != 0.0 {
                for c in col..n {
                    a[r][c] -= f * a[col][c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    let mut x = vec![0.0_f64; n];
    for r in (0..n).rev() {
        let mut s = b[r];
        for c in (r + 1)..n {
            s -= a[r][c] * x[c];
        }
        x[r] = s / a[r][r];
    }
    Some(x)
}

/// Pulay/DIIS (a.k.a. Anderson) extrapolation of the evGW quasiparticle energy
/// vector.
///
/// This mirrors what `pyscf.gw.evgw` does: one full GW pass produces a new
/// energy vector `x_out` from the vector `x_in` used to build G (and W); the
/// plain fixed-point iteration `x_in <- x_out` is then replaced by
///
/// 1. an optional *damped* step `x_mix = (1-a) x_in + a x_out`, and
/// 2. a Pulay extrapolation `x_next = sum_i c_i x_mix^(i)` that minimises
///    `|sum_i c_i r^(i)|` subject to `sum_i c_i = 1`, with residuals
///    `r^(i) = x_mix^(i) - x_in^(i)`.
///
/// A plain undamped and unextrapolated iteration is recovered with
/// `damping = 1.0` and DIIS disabled.
struct EvgwDiis {
    space: usize,
    min_history: usize,
    history: Vec<Vec<f64>>,
    errors: Vec<Vec<f64>>,
    /// Reject a DIIS step only for *sanity* reasons (non-finite, or an
    /// excursion larger than `max_excursion`).  A bracket-restricted variant
    /// is available through `evgw_diis_safeguard = true`.
    safeguard: bool,
    max_excursion: f64,
}

impl EvgwDiis {
    fn new(space: usize, min_history: usize, safeguard: bool) -> Self {
        EvgwDiis {
            space: space.max(2),
            min_history: min_history.max(2),
            history: Vec::new(),
            errors: Vec::new(),
            safeguard,
            max_excursion: 1.0,
        }
    }

    fn update(
        &mut self,
        x_in: &[f64],
        x_out: &[f64],
        damping: f64,
        max_step: f64,
        use_diis: bool,
    ) -> Vec<f64> {
        let n = x_out.len();
        let alpha = damping.clamp(1.0e-3, 1.0);
        // Damped (linear-mixed) step, optionally with a per-orbital cap on the
        // move.  MolGW's GnWn update is `E + Z (F(E) - E)` with Z clamped to
        // [0, 1]; `max_step` is the same idea expressed as a hard bound on the
        // per-round quasiparticle-energy change (it also protects against a
        // Newton solve landing on a spurious distant root).
        let x_mix: Vec<f64> = (0..n)
            .map(|i| {
                let mut d = alpha * (x_out[i] - x_in[i]);
                if max_step > 0.0 && d.abs() > max_step {
                    d = max_step * d.signum();
                }
                x_in[i] + d
            })
            .collect();
        if !use_diis {
            return x_mix;
        }
        let err: Vec<f64> = (0..n).map(|i| x_mix[i] - x_in[i]).collect();
        self.history.push(x_mix.clone());
        self.errors.push(err);
        while self.history.len() > self.space {
            self.history.remove(0);
            self.errors.remove(0);
        }
        let m = self.history.len();
        if m < self.min_history {
            return x_mix;
        }
        // Assemble the Pulay matrix.
        let dim = m + 1;
        let mut a = vec![vec![0.0_f64; dim]; dim];
        let mut max_diag = 0.0_f64;
        for i in 0..m {
            for j in 0..m {
                let mut s = 0.0;
                for k in 0..n {
                    s += self.errors[i][k] * self.errors[j][k];
                }
                a[i][j] = s;
            }
            max_diag = max_diag.max(a[i][i].abs());
        }
        // Tikhonov regularisation: orbitals outside the explicitly computed
        // window are extrapolated from a *common* energy shift, so their
        // residual vectors are exactly collinear and the Pulay matrix is rank
        // deficient (or badly conditioned) without a small diagonal shift.
        let eps = 1.0e-8 * max_diag + 1.0e-300;
        for i in 0..m {
            a[i][i] += eps;
        }
        for i in 0..m {
            a[i][m] = -1.0;
            a[m][i] = -1.0;
        }
        a[m][m] = 0.0;
        let mut rhs = vec![0.0_f64; dim];
        rhs[m] = -1.0;
        let coeffs = match solve_linear_system(a, rhs) {
            Some(c) => c,
            None => return x_mix,
        };
        let mut x_next = vec![0.0_f64; n];
        for i in 0..m {
            let ci = coeffs[i];
            for k in 0..n {
                x_next[k] += ci * self.history[i][k];
            }
        }
        // Sanity guard.  The default accepts any finite extrapolation that does
        // not run away from the damped step by more than `max_excursion`
        // (Anderson extrapolation is *meant* to overshoot).  With
        // `safeguard = true` a stricter bracket check is used instead: the
        // extrapolation may not leave the interval spanned by `x_in` and
        // `x_out` by more than half of its width plus 1e-3 Ha.
        let mut reject = false;
        for k in 0..n {
            if !x_next[k].is_finite() {
                reject = true;
                break;
            }
            if self.safeguard {
                let lo = x_in[k].min(x_out[k]);
                let hi = x_in[k].max(x_out[k]);
                let margin = 0.5 * (hi - lo).abs() + 1.0e-3;
                if (x_next[k] - hi).max(lo - x_next[k]).max(0.0) > margin {
                    reject = true;
                    break;
                }
            } else if (x_next[k] - x_mix[k]).abs() > self.max_excursion {
                reject = true;
                break;
            }
        }
        if reject {
            return x_mix;
        }
        x_next
    }
}

/// Eigenvalue-self-consistent GW (evGW).
///
/// Each round is one complete GW pass (`g0w0`) in which the Green's function
/// (and, as in MolGW's `GnWn`, the screened interaction) is rebuilt from the
/// current quasiparticle energies.  The plain fixed-point iteration
///
///     E^(k+1) = F(E^(k)),
///
/// where `F` is the map defined by one GW pass, is **not** a contraction for
/// typical molecules: the Jacobian of `F` has eigenvalues that approach (and
/// can exceed) unit modulus, so the raw iteration settles into a limit cycle
/// instead of converging.  Both reference implementations therefore accelerate
/// the outer loop:
///
/// * **MolGW** (`m_selfenergy_tools.f90`, `find_qp_energy_linearization`)
///   updates with `E_new = E_in + Z (F(E_in) - E_in)` where
///   `Z = 1 / (1 - dSigma/domega)` is clamped to `[0, 1]`.  In the diagonal
///   approximation this is Newton's method for the quasiparticle equation and
///   removes the fixed-point self-interaction entirely; it is a *damped*
///   update with a state-dependent step.
/// * **PySCF** (`pyscf/gw/evgw.py`) solves the quasiparticle equation exactly
///   for every orbital and then applies Pulay/DIIS extrapolation
///   (`pyscf.lib.diis.DIIS`) to the whole `mo_energy` vector.
///
/// REST already solves the quasiparticle equation exactly per orbital, so the
/// missing ingredient is the outer accelerator.  This routine therefore offers
/// the same two mechanisms through `evgw_damping` (scalar analogue of MolGW's
/// Z-scaling) and `evgw_diis` (PySCF's DIIS), plus a convergence test with
/// early exit so that "did evGW converge?" is answerable from the log.
pub fn evgw(scf_data:&mut SCF,num_freq:usize,vxc_nn:&Vec<f64>,iter_rounds:usize)->Vec<f64>{
    let (_start_mo,_num_state,occ_size,_vir_size,_homo,_lumo)=ri_gw::get_occupation_parameters(scf_data,'Y');
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let damping=qp_ctrl.evgw_damping.clamp(1.0e-3,1.0);
    let update_kind=evgw_update_kind(&qp_ctrl);
    let solver=match update_kind{
        EvgwUpdate::ZStep=>String::from("z_update"),
        EvgwUpdate::PlainEval=>String::from("molgw"),
        EvgwUpdate::Root=>String::from("diis"),
    };
    // The accelerators are alternatives for the same fixed-point problem and
    // must not be stacked: with `z_update` / `molgw` each GW pass already forms
    // the next quasiparticle energy directly, so DIIS is switched off.
    let use_diis=qp_ctrl.evgw_diis && update_kind==EvgwUpdate::Root;
    let diis_space=qp_ctrl.evgw_diis_space;
    let diis_start=qp_ctrl.evgw_diis_start;
    let conv_tol=qp_ctrl.evgw_conv_tol;
    let stop_on_conv=qp_ctrl.evgw_stop_on_convergence;
    let report=qp_ctrl.evgw_report;
    let diis_safeguard=qp_ctrl.evgw_diis_safeguard;
    let max_step_cap=qp_ctrl.evgw_max_step;

    if qp_ctrl.scgw=="evgw" && report{
        println!("evGW outer-loop settings: solver={}, damping(alpha)={:.4}, DIIS={} (space={}, start={}, safeguard={}), conv_tol={:.3e} Ha, stop_on_convergence={}, max_step={}",
                 solver,damping,use_diis,diis_space.max(2),diis_start.max(2),diis_safeguard,conv_tol,stop_on_conv,
                 if max_step_cap>0.0{format!("{:.3e} Ha",max_step_cap)}else{String::from("unlimited")});
    }

    let mut diis=EvgwDiis::new(diis_space,diis_start,diis_safeguard);
    let mut converged=false;
    let mut last_residual=f64::INFINITY;
    let mut last_dg=f64::INFINITY;
    let nstate=scf_data.gwqp.0.len();

    for i in 0..iter_rounds{
        println!("Now is round #{} of evGW calculation. There will be {} rounds in total.",i+1,iter_rounds);
        let cancel_dfa_xc=true;
        // Energy vector fed into this round (G and W both built from it, as in
        // MolGW's GnWn / PySCF's EVGW with W0 = False).
        let qp_in=scf_data.gwqp.0.clone();
        scf_data.gwqp.0=qp_in.clone();
        scf_data.gwqp.1=qp_in.clone();
        // One full GW pass: exact quasiparticle-equation solve per orbital.
        let qp_out=g0w0(scf_data,num_freq,vxc_nn,cancel_dfa_xc);

        // Convergence measures.
        let mut max_de=0.0_f64;
        let mut max_at=0usize;
        let mut dg=0.0_f64;
        for n in 0..nstate.min(qp_out.len()){
            let d=(qp_out[n]-qp_in[n]).abs();
            if d>max_de{max_de=d;max_at=n;}
            if qp_in[n].abs()>1.0e-12 && qp_out[n].abs()>1.0e-12{
                dg+=(1.0/qp_out[n]-1.0/qp_in[n]).abs();
            }
        }
        dg/=(nstate*nstate) as f64;
        last_residual=max_de;
        last_dg=dg;

        let qp_next=diis.update(&qp_in,&qp_out,damping,max_step_cap,use_diis);
        let mut max_step=0.0_f64;
        for n in 0..nstate.min(qp_next.len()){
            let d=(qp_next[n]-qp_in[n]).abs();
            if d>max_step{max_step=d;}
        }

        if report{
            let homo_qp=qp_out.get(occ_size-1).copied().unwrap_or(f64::NAN);
            let lumo_qp=qp_out.get(occ_size).copied().unwrap_or(f64::NAN);
            println!("evGW round {} summary: max|dE_qp|={:.4e} Ha (orbital #{}), |dG|={:.4e}, max|step|={:.4e} Ha",
                     i+1,max_de,max_at,dg,max_step);
            println!("evGW round {}: HOMO(#{} QP)={:.8} Ha, LUMO(#{} QP)={:.8} Ha, gap={:.8} Ha",
                     i+1,occ_size-1,homo_qp,occ_size,lumo_qp,lumo_qp-homo_qp);
        }

        scf_data.gwqp.0=qp_next.clone();
        scf_data.gwqp.1=qp_next.clone();

        if max_de<conv_tol{
            converged=true;
            if report{
                println!("evGW converged after {} round(s): max|dE_qp|={:.4e} Ha < conv_tol={:.4e} Ha",
                         i+1,max_de,conv_tol);
            }
            if stop_on_conv{
                break;
            }
        }
    }
    if !converged && iter_rounds>0{
        println!("WARNING: evGW did not reach conv_tol={:.4e} Ha within {} round(s); last max|dE_qp|={:.4e} Ha, |dG|={:.4e}.",
                 conv_tol,iter_rounds,last_residual,last_dg);
        println!("         Consider reducing evgw_damping, keeping evgw_diis = true (solver=\"diis\"; for solver=\"z_update\" DIIS does not apply), or increasing evgw_rounds.");
    }
    let final_qp=scf_data.gwqp.0.clone();
    if report && iter_rounds>0{
        ri_gw::display::full_quasiparticles(&final_qp,occ_size);
    }
    final_qp
}
/// Step size (Ha) of the central finite difference used to obtain the
/// quasiparticle-equation derivative for the MolGW-style `z_update` solver.
const Z_UPDATE_DERIVATIVE_H: f64 = 1.0e-5;

/// One MolGW-style Z-damped quasiparticle step.
///
/// MolGW's eigenvalue-self-consistent `GnWn` loop does not iterate the
/// quasiparticle equation to its root; it takes a single *linearized* Newton
/// step per GW pass (`find_qp_energy_linearization` in
/// `m_selfenergy_tools.f90`):
///
///     E_out = E_in + Z * (E_KS - V_xc + Sigma_c(E_in) - E_in),
///     Z     = 1 / (1 - dSigma_c/domega),   clamped to [0, 1].
///
/// `qp_eq(omega)` here is `consts + Sigma_c(omega) - omega`, so
/// `E_out = E_in + Z * qp_eq(E_in)` and `d(qp_eq)/domega = Sigma_c' - 1`,
/// hence `Z = -1 / (d qp_eq / d omega)`.  Clamping Z to `[0, 1]` guarantees
/// that a weak self-energy pole sitting close to `E_in` (which would give a
/// tiny or sign-changing `dSigma/domega`) cannot produce an overshoot; this is
/// exactly the `MIN(MAX(zz, 0), 1)` clamp in MolGW.
///
/// Returns `(E_out, Z, qp_eq(E_in))`.
fn one_shot_step<F: Fn(f64) -> f64>(
    qp_eq: &F,
    e_in: f64,
    h: f64,
    force_z: Option<f64>,
) -> (f64, f64, f64, f64) {
    let g0 = qp_eq(e_in);
    if !g0.is_finite() {
        return (e_in, 0.0, g0, f64::NAN);
    }
    if let Some(z0) = force_z {
        let z = z0.clamp(0.0, 1.0);
        return (e_in + z * g0, z, g0, f64::NAN);
    }
    let dq = central_difference(qp_eq, e_in, h);
    let mut z = if dq.is_finite() && dq.abs() > 1.0e-12 { -1.0 / dq } else { 0.0 };
    if !z.is_finite() {
        z = 0.0;
    }
    z = z.clamp(0.0, 1.0);
    (e_in + z * g0, z, g0, dq)
}

/// Central finite difference of the quasiparticle equation at `x`.
fn central_difference<F: Fn(f64) -> f64>(f: &F, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * h)
}

/// How one evGW round forms the next quasiparticle energy for each orbital.
#[derive(Clone, Copy, PartialEq, Eq)]
enum EvgwUpdate {
    /// Historical REST behaviour: solve the quasiparticle equation to its root
    /// (Newton / interpolation rootfinder).
    Root,
    /// Z-damped linearized Newton step, `Z = 1/(1 - dSigma/domega)` clamped to
    /// `[0, 1]`.
    ZStep,
    /// MolGW's `GnWn` (EVSC) rule: a *single* evaluation
    /// `E_new = consts + Sigma_c(E_in)` at the previous quasiparticle energy,
    /// with no root solve and no damping.
    PlainEval,
}

fn evgw_update_kind(qp_ctrl: &crate::ctrl_io::quasiparticle_methods::QuasiParticle) -> EvgwUpdate {
    if qp_ctrl.scgw != "evgw" {
        return EvgwUpdate::Root;
    }
    match qp_ctrl.evgw_solver.to_lowercase().as_str() {
        "z_update" | "z-update" | "zupdate" => EvgwUpdate::ZStep,
        "molgw" | "molgw_update" | "plain_eval" | "eval" => EvgwUpdate::PlainEval,
        _ => EvgwUpdate::Root,
    }
}

fn mode_label(update: EvgwUpdate) -> &'static str {
    match update {
        EvgwUpdate::PlainEval => "molgw_update",
        _ => "z_update",
    }
}

fn forced_z(update: EvgwUpdate) -> Option<f64> {
    match update {
        EvgwUpdate::PlainEval => Some(1.0),
        _ => None,
    }
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

    if evgw_update_kind(&qp_ctrl) != EvgwUpdate::Root {
        // MolGW-style Z-damped update (see z_update_step).
        let e_in = gwqp_g[n];
        let kind = evgw_update_kind(&qp_ctrl);
        let (e_out, z, g0, dq) = one_shot_step(&qp_eq_func, e_in, qp_ctrl.evgw_z_step, forced_z(kind));
        if scf_data.mol.ctrl.print_level > 1 && kind == EvgwUpdate::ZStep {
            println!("  [z_update diag] d(Sigma-omega)/domega at h=1e-5: {:.4e}, h=1e-3: {:.4e}, h=1e-2: {:.4e}, h=5e-2: {:.4e}",
                     central_difference(&qp_eq_func, e_in, 1.0e-5),
                     central_difference(&qp_eq_func, e_in, 1.0e-3),
                     central_difference(&qp_eq_func, e_in, 1.0e-2),
                     central_difference(&qp_eq_func, e_in, 5.0e-2));
        }
        println!("Orbital #{} (AC, {}): E_in={:.8} Z={:.6} qp_eq(E_in)={:.8} dq={:.6e} -> E_out={:.8}",
                 n, mode_label(kind), e_in, z, g0, dq, e_out);
        qp_energy_no_fse = e_out;
    } else if e_ks_n.abs() > qp_ctrl.gw_switch_fallback_threshold {
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

    if evgw_update_kind(&qp_ctrl) != EvgwUpdate::Root {
        // MolGW-style Z-damped update (see z_update_step).
        let e_in = gwqp_g[n];
        let kind = evgw_update_kind(&qp_ctrl);
        let (e_out, z, g0, dq) = one_shot_step(&qp_eq_func_no_fse, e_in, qp_ctrl.evgw_z_step, forced_z(kind));
        if scf_data.mol.ctrl.print_level > 1 && kind == EvgwUpdate::ZStep {
            println!("  [z_update diag] d(Sigma-omega)/domega at h=1e-5: {:.4e}, h=1e-3: {:.4e}, h=1e-2: {:.4e}, h=5e-2: {:.4e}",
                     central_difference(&qp_eq_func_no_fse, e_in, 1.0e-5),
                     central_difference(&qp_eq_func_no_fse, e_in, 1.0e-3),
                     central_difference(&qp_eq_func_no_fse, e_in, 1.0e-2),
                     central_difference(&qp_eq_func_no_fse, e_in, 5.0e-2));
        }
        println!("Orbital #{} ({}): E_in={:.8} Z={:.6} qp_eq(E_in)={:.8} dq={:.6e} -> E_out={:.8}",
                 n, mode_label(kind), e_in, z, g0, dq, e_out);
        qp_energy_no_fse = e_out;
    } else if e_ks_n.abs() > qp_ctrl.gw_switch_fallback_threshold {
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
        Some({
            let mut ra = ri_gw::generate_real_axis_vchiv(
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
            );
            ra.interp = if qp_ctrl.low_rank_interp.eq_ignore_ascii_case("nearest") { 0 } else { 1 };
            ra
        })
    } else {
        None
    };

    let e_homo=ks_energies[occ_size-1];
    let e_lumo=ks_energies[occ_size];
    // [DBG] KS-eigenvalue diagnostics (electron count / homo / window), only at print_level >= 2
    if scf_data.mol.ctrl.print_level >= 2 {
        println!("[DBG KS] occ_size={} start_mo={} num_state={} e_homo={:.12e} e_lumo={:.12e}", occ_size, start_mo, num_state, e_homo, e_lumo);
        println!("[DBG KS] num_elec={:?} scf_homo={:?} scf_lumo={:?}", scf_data.mol.num_elec, scf_data.homo, scf_data.lumo);
        for dbg_n in 0..25usize.min(ks_energies.len()) {
            println!("[DBG KS] n={} e={:.12e}", dbg_n, ks_energies[dbg_n]);
        }
    }
    let calc_orbs_indices:Vec<usize>=ks_energies.into_iter().enumerate().filter(|(n,e_n)|*e_n>e_homo-occ_threshold && *e_n<e_lumo+vir_threshold).map(|(n,e_n)|n).collect();
    println!("calculated orbital indices:{:?}",calc_orbs_indices);
    // v1/v2 paths use only ri_row_n (per-orbital) + pre-computed low-rank data;
    // the full naux×nov ri_ov is not read in the orbital loop. Drop it (free the
    // N³ resident) before entering the loop. Only the v0 path (contour_rayon
    // builds response matrices from ri_ov) keeps it alive.
    let calc_orbs:Vec<(usize,f64)>=if let Some(ref ra) = real_axis_vchiv {
        drop(ri_ov);
        calc_orbs_indices.iter().map(|&n|{
            let ri_row_n=ri_gw::compute_ri3mo_row(scf_data,n);
            if let Some(ref wlr) = w_c_lr {
                // v2 path: imaginary-axis low-rank with pre-computed wc_rows.
                let wc_rows = ri_gw::precompute_wc_rows_lowrank(wlr, &ri_row_n, num_state);
                (n, single_orbital_gw_lowrank_v2(scf_data, &v_matrix, &ri_row_n, &wc_rows, ra, n, num_freq, vxc_nn[n]))
            } else {
                // v1 path: full W_c on imag axis + real-axis low-rank.
                (n, single_orbital_gw_lowrank(scf_data, &v_matrix, &ri_row_n, &w_c_at_freqs, ra, n, num_freq, vxc_nn[n]))
            }
        }).collect()
    } else {
        calc_orbs_indices.iter().map(|&n|{
            let ri_row_n=ri_gw::compute_ri3mo_row(scf_data,n);
            (n, single_orbital_gw(scf_data, &v_matrix, &ri_ov, &ri_row_n, &w_c_at_freqs, n, num_freq, vxc_nn[n]))
        }).collect()
    };
    let occ_shift=calc_orbs[0].1-scf_data.eigenvalues[0][calc_orbs[0].0];
    let vir_shift=calc_orbs[calc_orbs.len()-1].1-scf_data.eigenvalues[0][calc_orbs[calc_orbs.len()-1].0];
    let mut gwqp:Vec<f64>=Vec::new();
    if scf_data.mol.ctrl.print_level>1{
        println!("low extrapolations:{}",calc_orbs[0].0);
        println!("calculated orbitals:{}",calc_orbs.len());
        println!("high extrapolations:{}",num_state-1-calc_orbs[calc_orbs.len()-1].0);
        println!("Occ Shift={}, Vir Shift={}",occ_shift,vir_shift);
    }
    // States outside the explicitly computed window.
    //
    // Default (REST historical behaviour): extrapolate them with the rigid
    // shift of the lowest/highest computed orbital.
    //
    // `evgw_freeze_outside` selects MolGW's behaviour instead: those states
    // keep their *initial* (Kohn-Sham) energies.  MolGW does this implicitly --
    // `find_qp_energy_linearization` initialises `energy_qp_z = energy0` and
    // only overwrites `nsemin..nsemax`, so the states outside the self-energy
    // window are written back unchanged and never enter the self-consistent
    // loop.  Extrapolating them instead (REST) makes their rigid shift a slow
    // feedback channel into G and W.
    let freeze_outside = qp_ctrl.evgw_freeze_outside && qp_ctrl.scgw == "evgw";
    let baseline = &scf_data.eigenvalues[0];
    for i in 0 .. calc_orbs[0].0{
        gwqp.push(if freeze_outside {baseline[i]} else {baseline[i]+occ_shift})
    }
    for i in 0..calc_orbs.len(){
        gwqp.push(calc_orbs[i].1)
    }
    for i in calc_orbs[calc_orbs.len()-1].0+1 .. num_state{
        gwqp.push(if freeze_outside {baseline[i]} else {baseline[i]+vir_shift})
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
    let cdgw_res_tol = qp_ctrl.cdgw_res_tol;
    let rootfinder = qp_ctrl.gw_rootfinder.clone();
    let qp_energy_no_fse;

    if evgw_update_kind(&qp_ctrl) != EvgwUpdate::Root {
        // MolGW-style Z-damped update (see z_update_step).
        let qp_eq_func = |omega: f64| {
            ri_gw::quasiparticle_equation_lowrank(
                omega, n, consts, ri_row_n, &gwqp_g, &gwqp_w,
                occ_size, vir_size, num_state,
                w_c_at_freqs, real_axis_vchiv, 0, cdgw_res_tol,
            )
        };
        let e_in = gwqp_g[n];
        let kind = evgw_update_kind(&qp_ctrl);
        let (e_out, z, g0, dq) = one_shot_step(&qp_eq_func, e_in, qp_ctrl.evgw_z_step, forced_z(kind));
        if scf_data.mol.ctrl.print_level > 1 && kind == EvgwUpdate::ZStep {
            println!("  [z_update diag] d(Sigma-omega)/domega at h=1e-5: {:.4e}, h=1e-3: {:.4e}, h=1e-2: {:.4e}, h=5e-2: {:.4e}",
                     central_difference(&qp_eq_func, e_in, 1.0e-5),
                     central_difference(&qp_eq_func, e_in, 1.0e-3),
                     central_difference(&qp_eq_func, e_in, 1.0e-2),
                     central_difference(&qp_eq_func, e_in, 5.0e-2));
        }
        println!("Orbital #{} (low-rank, {}): E_in={:.8} Z={:.6} qp_eq(E_in)={:.8} dq={:.6e} -> E_out={:.8}",
                 n, mode_label(kind), e_in, z, g0, dq, e_out);
        qp_energy_no_fse = e_out;
    } else if e_ks_n.abs() > qp_ctrl.gw_switch_fallback_threshold {
        let qp_eq_func = |omega: f64| {
            ri_gw::quasiparticle_equation_lowrank(
                omega, n, consts, ri_row_n, &gwqp_g, &gwqp_w,
                occ_size, vir_size, num_state,
                w_c_at_freqs, real_axis_vchiv, 0, cdgw_res_tol,
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
            qp_start, 0.00001, 50, side, scf_data.mol.ctrl.print_level, cdgw_res_tol,
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
            w_c_at_freqs, real_axis_vchiv, pl, cdgw_res_tol,
        );
        let qp_eq_func = |omega: f64| {
            ri_gw::quasiparticle_equation_lowrank(
                omega, n, consts, ri_row_n, &gwqp_g, &gwqp_w,
                occ_size, vir_size, num_state,
                w_c_at_freqs, real_axis_vchiv, 0, cdgw_res_tol,
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
                qp_start, 0.00001, 50, side, scf_data.mol.ctrl.print_level, cdgw_res_tol,
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
                occ_size, vir_size, num_state,                 w_c_at_freqs, real_axis_vchiv, 0, cdgw_res_tol,
            ) + ri_gw::fourier_self_energy::fourier_series(&sin_coeff, &cos_coeff, powers, t, omega - origin)
        };
        let (have_crossing, qp) = ri_gw::linear_interpolation_solver(
            qp_eq_func, qp_energy_no_fse, side, qp_ctrl.gw_search_grid, qp_ctrl.gw_span_energy,
        );
        final_qp = if !have_crossing {
            ri_gw::newton_solver_lowrank(
                n, consts, ri_row_n, &gwqp_g, &gwqp_w, occ_size, vir_size, num_state,
                w_c_at_freqs, real_axis_vchiv,
                qp_energy_no_fse, 0.00001, 50, side, scf_data.mol.ctrl.print_level, cdgw_res_tol,
            )
        } else {
            qp
        };
    } else if qp_ctrl.hermite_self_energy {
        let hermite_coeff = ri_gw::fourier_self_energy::define_hermite_series(&qp_ctrl);
        let qp_eq_func = |omega: f64| {
            ri_gw::quasiparticle_equation_lowrank(
                omega, n, consts, ri_row_n, &gwqp_g, &gwqp_w,
                occ_size, vir_size, num_state,                 w_c_at_freqs, real_axis_vchiv, 0, cdgw_res_tol,
            ) + ri_gw::fourier_self_energy::sigma_hermite(origin, omega, &hermite_coeff)
        };
        let (have_crossing, qp) = ri_gw::linear_interpolation_solver(
            qp_eq_func, qp_energy_no_fse, side, qp_ctrl.gw_search_grid, qp_ctrl.gw_span_energy,
        );
        final_qp = if !have_crossing {
            ri_gw::newton_solver_lowrank(
                n, consts, ri_row_n, &gwqp_g, &gwqp_w, occ_size, vir_size, num_state,
                w_c_at_freqs, real_axis_vchiv,
                qp_energy_no_fse, 0.00001, 50, side, scf_data.mol.ctrl.print_level, cdgw_res_tol,
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
    let cdgw_res_tol = qp_ctrl.cdgw_res_tol;
    let rootfinder = qp_ctrl.gw_rootfinder.clone();
    let qp_energy_no_fse;

    if evgw_update_kind(&qp_ctrl) != EvgwUpdate::Root {
        // MolGW-style Z-damped update (see z_update_step).
        let qp_eq_func = |omega: f64| {
            ri_gw::quasiparticle_equation_lowrank_v2(
                omega, n, consts, ri_row_n, &gwqp_g, &gwqp_w,
                occ_size, vir_size, num_state,
                wc_rows, real_axis_vchiv, 0, cdgw_res_tol,
            )
        };
        let e_in = gwqp_g[n];
        let kind = evgw_update_kind(&qp_ctrl);
        let (e_out, z, g0, dq) = one_shot_step(&qp_eq_func, e_in, qp_ctrl.evgw_z_step, forced_z(kind));
        if scf_data.mol.ctrl.print_level > 1 && kind == EvgwUpdate::ZStep {
            println!("  [z_update diag] d(Sigma-omega)/domega at h=1e-5: {:.4e}, h=1e-3: {:.4e}, h=1e-2: {:.4e}, h=5e-2: {:.4e}",
                     central_difference(&qp_eq_func, e_in, 1.0e-5),
                     central_difference(&qp_eq_func, e_in, 1.0e-3),
                     central_difference(&qp_eq_func, e_in, 1.0e-2),
                     central_difference(&qp_eq_func, e_in, 5.0e-2));
        }
        println!("Orbital #{} (low-rank v2, {}): E_in={:.8} Z={:.6} qp_eq(E_in)={:.8} dq={:.6e} -> E_out={:.8}",
                 n, mode_label(kind), e_in, z, g0, dq, e_out);
        qp_energy_no_fse = e_out;
    } else if e_ks_n.abs() > qp_ctrl.gw_switch_fallback_threshold {
        let qp_eq_func = |omega: f64| {
            ri_gw::quasiparticle_equation_lowrank_v2(
                omega, n, consts, ri_row_n, &gwqp_g, &gwqp_w,
                occ_size, vir_size, num_state,
                wc_rows, real_axis_vchiv, 0, cdgw_res_tol,
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
            qp_start, 0.00001, 50, side, scf_data.mol.ctrl.print_level, cdgw_res_tol,
        );
        println!("QP energy (low-rank v2, no FSE): {}", qp_energy_no_fse);
    } else if rootfinder == "interpolation".to_string() {
        let debug_omega = gwqp_g[n];
        let pl = scf_data.mol.ctrl.print_level;
        ri_gw::quasiparticle_equation_lowrank_v2(
            debug_omega, n, consts, ri_row_n, &gwqp_g, &gwqp_w,
            occ_size, vir_size, num_state,
            wc_rows, real_axis_vchiv, pl, cdgw_res_tol,
        );
        let qp_eq_func = |omega: f64| {
            ri_gw::quasiparticle_equation_lowrank_v2(
                omega, n, consts, ri_row_n, &gwqp_g, &gwqp_w,
                occ_size, vir_size, num_state,
                wc_rows, real_axis_vchiv, 0, cdgw_res_tol,
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
                qp_start, 0.00001, 50, side, scf_data.mol.ctrl.print_level, cdgw_res_tol,
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
                occ_size, vir_size, num_state, wc_rows, real_axis_vchiv, 0, cdgw_res_tol,
            ) + ri_gw::fourier_self_energy::fourier_series(&sin_coeff, &cos_coeff, powers, t, omega - origin)
        };
        let (have_crossing, qp) = ri_gw::linear_interpolation_solver(
            qp_eq_func, qp_energy_no_fse, side, qp_ctrl.gw_search_grid, qp_ctrl.gw_span_energy,
        );
        final_qp = if !have_crossing {
            ri_gw::newton_solver_lowrank_v2(
                n, consts, ri_row_n, &gwqp_g, &gwqp_w, occ_size, vir_size, num_state,
                wc_rows, real_axis_vchiv,
                qp_energy_no_fse, 0.00001, 50, side, scf_data.mol.ctrl.print_level, cdgw_res_tol,
            )
        } else {
            qp
        };
    } else if qp_ctrl.hermite_self_energy {
        let hermite_coeff = ri_gw::fourier_self_energy::define_hermite_series(&qp_ctrl);
        let qp_eq_func = |omega: f64| {
            ri_gw::quasiparticle_equation_lowrank_v2(
                omega, n, consts, ri_row_n, &gwqp_g, &gwqp_w,
                occ_size, vir_size, num_state, wc_rows, real_axis_vchiv, 0, cdgw_res_tol,
            ) + ri_gw::fourier_self_energy::sigma_hermite(origin, omega, &hermite_coeff)
        };
        let (have_crossing, qp) = ri_gw::linear_interpolation_solver(
            qp_eq_func, qp_energy_no_fse, side, qp_ctrl.gw_search_grid, qp_ctrl.gw_span_energy,
        );
        final_qp = if !have_crossing {
            ri_gw::newton_solver_lowrank_v2(
                n, consts, ri_row_n, &gwqp_g, &gwqp_w, occ_size, vir_size, num_state,
                wc_rows, real_axis_vchiv,
                qp_energy_no_fse, 0.00001, 50, side, scf_data.mol.ctrl.print_level, cdgw_res_tol,
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
