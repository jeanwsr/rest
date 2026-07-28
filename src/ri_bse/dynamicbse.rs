// ============================================================================
// dynamicbse.rs — NonLinear FEAST (NLFEAST) for BSE using the new
// IA-pair projection matvec scheme.
//
// Mirrors nonlinbse.rs in algorithm structure but replaces the
// epsilon^{-1} dressing approach with the direct IA-pair summation
// method for the screened-exchange (W) contribution.
//
// Key differences from nonlinbse.rs:
//   1. Contour nodes: no response calc, no inverse dielectric, no dressing
//   2. Real-axis residual: no on-the-fly response/inverse/dressing
//   3. Matvec: uses dynamicbse_matvec functions instead of nonlinbse_matvec
//   4. Pre-computation: only RI transposes and Delta array
// ============================================================================

use num::Complex;
use rand::Rng;
use std::time::Instant;
use tensors::{MathMatrix, MatrixFull};
use crate::constants::EV;
use rest_tensors::matrix::matrix_blas_lapack::{
    _dgemm_full, _dgemm_scaled, _dgeev,
};

use crate::ctrl_io::quasiparticle_methods::QuasiParticle;
use crate::ri_gw::get_occupation_parameters;
use crate::scf_io::SCF;

use super::dynamicbse_matvec;
use super::nonlinbse::{BlockDiagPrecond, NLFeastResult, gmres, qr_orthonormalise};

// ============================================================================
// Data structures (simplified — no dressed RI matrices)
// ============================================================================

/// Quadrature node data for the contour integral.
pub struct ContourNodeDataSimple {
    pub z_re: f64,
    pub z_im: f64,
    pub w_re: f64,    // quadrature weight (real part)
    pub w_im: f64,    // quadrature weight (imaginary part)
}

// ============================================================================
// Contour preparation (simplified — just quadrature)
// ============================================================================

/// Pre-compute contour quadrature nodes.
///
/// For each quadrature node z_j on the circle (centre c, radius r):
///   z_j = c + r·exp(i·θ_j),   θ_j = 2π·j/n_quad
///   w_j = (z_j − c) / n_quad  (trapezoidal rule on the circle)
///
/// No response/inverse/dressing computation needed.
pub fn prepare_contour_nodes_simple(
    centre: f64,
    radius: f64,
    n_quad: usize,
) -> Vec<ContourNodeDataSimple> {
    let mut nodes = Vec::with_capacity(n_quad);
    for j in 0..n_quad {
        let theta = 2.0 * std::f64::consts::PI * j as f64 / n_quad as f64;
        let (ct, st) = (theta.cos(), theta.sin());
        let z_re = centre + radius * ct;
        let z_im = radius * st;

        // Quadrature weight: w = (z - c) / n_quad
        let w_re = radius * ct / n_quad as f64;
        let w_im = radius * st / n_quad as f64;

        nodes.push(ContourNodeDataSimple { z_re, z_im, w_re, w_im });
    }
    nodes
}

// ============================================================================
// Real-axis matvec T_real(λ)·x (real eigenvalue estimate)
// ============================================================================

/// Evaluate T_real(λ)·x for a real λ and real vector x.
///
///   T(λ) = ((A+B)(A-B) − λ²I)
///
/// Uses the new IA-pair summation matvec directly — no response/dielectric.
pub fn compute_real_matvec_new(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    occ_size: usize,
    vir_size: usize,
    ri_oo_T: &MatrixFull<f64>,
    ri_vv_T: &MatrixFull<f64>,
    ri_IA_T: &MatrixFull<f64>,
    delta_IA: &[f64],
    eps_occ: &[f64],
    eps_vir: &[f64],
    lambda: f64,
    x: &[f64],
    block_size: usize,
) -> Vec<f64> {
    dynamicbse_matvec::composite_matvec_new_real(
        scf_data, qp_ctrl,
        ri_oo_T, ri_vv_T, ri_IA_T, delta_IA, eps_occ, eps_vir,
        lambda, x, block_size,
    )
}

/// TDA real-axis matvec: T(λ)·x = A(λ)·x − λ·x
pub fn compute_real_tda_matvec_new(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    occ_size: usize,
    vir_size: usize,
    ri_oo_T: &MatrixFull<f64>,
    ri_vv_T: &MatrixFull<f64>,
    ri_IA_T: &MatrixFull<f64>,
    delta_IA: &[f64],
    eps_occ: &[f64],
    eps_vir: &[f64],
    lambda: f64,
    x: &[f64],
    block_size: usize,
) -> Vec<f64> {
    let a_x = dynamicbse_matvec::a_block_matvec_new(
        scf_data, qp_ctrl,
        ri_oo_T, ri_vv_T, ri_IA_T, delta_IA, eps_occ, eps_vir,
        lambda, x, block_size,
    );
    a_x.iter().zip(x.iter()).map(|(a, &xi)| a - lambda * xi).collect()
}

// ============================================================================
// Build contour shifted matvec closure (new scheme)
// ============================================================================

fn make_shifted_matvec_new<'a>(
    scf_data: &'a SCF,
    qp_ctrl: &'a QuasiParticle,
    n: usize,
    ri_oo_T: &'a MatrixFull<f64>,
    ri_vv_T: &'a MatrixFull<f64>,
    ri_IA_T: &'a MatrixFull<f64>,
    delta_IA: &'a [f64],
    eps_occ: &'a [f64],
    eps_vir: &'a [f64],
    node: &'a ContourNodeDataSimple,
    block_size: usize,
) -> impl Fn(&[f64]) -> Vec<f64> + 'a {
    let z = Complex::new(node.z_re, node.z_im);
    move |v: &[f64]| -> Vec<f64> {
        let vr = &v[0..n];
        let vi = &v[n..2 * n];
        let (t_re, t_im) = dynamicbse_matvec::composite_matvec_new_complex(
            scf_data, qp_ctrl,
            ri_oo_T, ri_vv_T, ri_IA_T, delta_IA, eps_occ, eps_vir,
            z, vr, vi, block_size,
        );
        let mut out = vec![0.0; 2 * n];
        for i in 0..n {
            out[i]     = t_re[i];
            out[n + i] = t_im[i];
        }
        out
    }
}

fn make_shifted_tda_matvec_new<'a>(
    scf_data: &'a SCF,
    qp_ctrl: &'a QuasiParticle,
    n: usize,
    ri_oo_T: &'a MatrixFull<f64>,
    ri_vv_T: &'a MatrixFull<f64>,
    ri_IA_T: &'a MatrixFull<f64>,
    delta_IA: &'a [f64],
    eps_occ: &'a [f64],
    eps_vir: &'a [f64],
    node: &'a ContourNodeDataSimple,
    block_size: usize,
) -> impl Fn(&[f64]) -> Vec<f64> + 'a {
    let z = Complex::new(node.z_re, node.z_im);
    move |v: &[f64]| -> Vec<f64> {
        let vr = &v[0..n];
        let vi = &v[n..2 * n];
        let (a_re, a_im) = dynamicbse_matvec::a_block_matvec_new_complex(
            scf_data, qp_ctrl,
            ri_oo_T, ri_vv_T, ri_IA_T, delta_IA, eps_occ, eps_vir,
            z, vr, vi, block_size,
        );
        // T(z)·(vr+i·vi) = A(z)·(vr+i·vi) − z·(vr+i·vi)
        let mut out = vec![0.0; 2 * n];
        for i in 0..n {
            out[i]     = a_re[i] - node.z_re * vr[i] + node.z_im * vi[i];
            out[n + i] = a_im[i] - node.z_re * vi[i] - node.z_im * vr[i];
        }
        out
    }
}

// ============================================================================
// Projected solver: solve Q^T T(λ) Q y = 0 in the current subspace
// ============================================================================

fn solve_projected_bse_new(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    occ_size: usize,
    vir_size: usize,
    n: usize,
    ri_oo_T: &MatrixFull<f64>,
    ri_vv_T: &MatrixFull<f64>,
    ri_IA_T: &MatrixFull<f64>,
    delta_IA: &[f64],
    eps_occ: &[f64],
    eps_vir: &[f64],
    q: &MatrixFull<f64>,
    lambda_init: &[f64],
    block_size: usize,
) -> (Vec<Complex<f64>>, MatrixFull<f64>) {
    let m0 = q.size[1];
    assert_eq!(lambda_init.len(), m0, "lambda_init must have m0 entries");

    // Apply (A+B)(A-B) at λ_j to each column q_j
    // Uses the new matvec with ω = λ_j (real)
    let mut aq = MatrixFull::new([n, m0], 0.0);
    for j in 0..m0 {
        let lambda = lambda_init[j];
        let qj: Vec<f64> = (0..n).map(|i| q[[i, j]]).collect();

        // (A+B)(A-B) at λ_j applied to q_j  — INLINE the computation
        // to avoid the -λ_j²·x shift that composite_matvec_new_real adds.
        //
        // Unlike the old code (which achieves this by passing ω=0 to a
        // composite matvec whose dressed RI matrices were already prepared
        // at λ_j), the new scheme's W(ω) is controlled by the same ω
        // parameter that also controls the ω²·x shift.  So we need to
        // apply A and B blocks directly and skip the shift.
        let a_x = dynamicbse_matvec::a_block_matvec_new(
            scf_data, qp_ctrl,
            ri_oo_T, ri_vv_T, ri_IA_T, delta_IA, eps_occ, eps_vir,
            lambda, &qj, block_size,
        );
        let b_x = dynamicbse_matvec::b_block_matvec_new(
            scf_data, qp_ctrl, ri_IA_T, delta_IA, eps_occ, eps_vir,
            lambda, &qj, block_size,
        );
        let t: Vec<f64> = a_x.iter().zip(b_x.iter()).map(|(a, b)| a - b).collect();
        let a_t = dynamicbse_matvec::a_block_matvec_new(
            scf_data, qp_ctrl,
            ri_oo_T, ri_vv_T, ri_IA_T, delta_IA, eps_occ, eps_vir,
            lambda, &t, block_size,
        );
        let b_t = dynamicbse_matvec::b_block_matvec_new(
            scf_data, qp_ctrl, ri_IA_T, delta_IA, eps_occ, eps_vir,
            lambda, &t, block_size,
        );
        let result: Vec<f64> = a_t.iter().zip(b_t.iter()).map(|(a, b)| a + b).collect();
        for i in 0..n {
            aq[[i, j]] = result[i];
        }
    }

    // M_Q = Q^T · AQ  (m₀ × m₀, non-symmetric in general)
    let mq = _dgemm_scaled(q, 'T', &aq, 'N', 1.0);

    // Solve M_Q · y = μ · y  via dgeev
    let (_vr_dummy, wr, wi, _vl, vr, info) = _dgeev(&mq, 'N', 'V');
    if info != 0 {
        eprintln!("Warning: dgeev in solve_projected_bse_new returned info={}", info);
    }

    // λ = √μ
    let mut lambdas = Vec::with_capacity(m0);
    let mut ritz = MatrixFull::new([n, m0], 0.0);
    for j in 0..m0 {
        let mu = Complex::new(wr[j], wi[j]);
        let sqrt_mu = mu.sqrt();
        let lam = Complex::new(sqrt_mu.re.max(0.0), 0.0);
        lambdas.push(lam);

        // Ritz vector = Q · yⱼ
        for i in 0..n {
            let mut val = 0.0;
            for k in 0..m0 {
                val += q[[i, k]] * vr[[k, j]];
            }
            ritz[[i, j]] = val;
        }
    }

    (lambdas, ritz)
}

fn solve_projected_tda_new(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    occ_size: usize,
    vir_size: usize,
    n: usize,
    ri_oo_T: &MatrixFull<f64>,
    ri_vv_T: &MatrixFull<f64>,
    ri_IA_T: &MatrixFull<f64>,
    delta_IA: &[f64],
    eps_occ: &[f64],
    eps_vir: &[f64],
    q: &MatrixFull<f64>,
    lambda_init: &[f64],
    block_size: usize,
) -> (Vec<Complex<f64>>, MatrixFull<f64>) {
    let m0 = q.size[1];
    assert_eq!(lambda_init.len(), m0);

    // Apply A(λ_j) to each column q_j
    let mut aq = MatrixFull::new([n, m0], 0.0);
    for j in 0..m0 {
        let lambda = lambda_init[j];
        let qj: Vec<f64> = (0..n).map(|i| q[[i, j]]).collect();

        let result = dynamicbse_matvec::a_block_matvec_new(
            scf_data, qp_ctrl,
            ri_oo_T, ri_vv_T, ri_IA_T, delta_IA, eps_occ, eps_vir,
            lambda, &qj, block_size,
        );
        for i in 0..n {
            aq[[i, j]] = result[i];
        }
    }

    // M_Q = Q^T · AQ
    let mq = _dgemm_scaled(q, 'T', &aq, 'N', 1.0);

    // Solve M_Q · y = λ · y
    let (_vr_dummy, wr, wi, _vl, vr, info) = _dgeev(&mq, 'N', 'V');
    if info != 0 {
        eprintln!("Warning: dgeev in solve_projected_tda_new returned info={}", info);
    }

    let mut lambdas = Vec::with_capacity(m0);
    let mut ritz = MatrixFull::new([n, m0], 0.0);
    for j in 0..m0 {
        let lam = Complex::new(wr[j].max(0.0), 0.0);
        lambdas.push(lam);
        for i in 0..n {
            let mut val = 0.0;
            for k in 0..m0 {
                val += q[[i, k]] * vr[[k, j]];
            }
            ritz[[i, j]] = val;
        }
    }

    (lambdas, ritz)
}

// ============================================================================
// Build final result from interior eigenpairs
// ============================================================================

fn build_final_result_dynamic(
    n: usize,
    lambda_cur: &[f64],
    x_mat: &MatrixFull<f64>,
    residuals: &[f64],
    inside_flags: &[bool],
    centre: f64,
    radius: f64,
) -> NLFeastResult {
    let mut pairs: Vec<(f64, f64, usize)> = Vec::new();
    for (j, &inside) in inside_flags.iter().enumerate() {
        if !inside { continue; }
        let dist = (lambda_cur[j] - centre).abs();
        if dist <= radius * 1.05 {
            pairs.push((dist, lambda_cur[j], j));
        }
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let n_found = pairs.len();
    let mut eigvals = Vec::with_capacity(n_found);
    let mut eigvecs = MatrixFull::new([n, n_found], 0.0);
    let mut final_res = Vec::with_capacity(n_found);
    for (k, (_, lam, orig_j)) in pairs.iter().enumerate() {
        eigvals.push(*lam);
        for i in 0..n { eigvecs[[i, k]] = x_mat[[i, *orig_j]]; }
        final_res.push(residuals[*orig_j]);
    }
    NLFeastResult {
        eigenvalues: eigvals,
        eigenvectors: eigvecs,
        n_found,
        iterations: 0,
        residuals: final_res,
    }
}

// ============================================================================
// Main NLFEAST solver (new scheme)
// ============================================================================

/// Run the NLFEAST solver using the new IA-pair projection matvec.
///
/// Mirrors nonlinbse::nlfeast_bse but uses the new matvec scheme:
///   - No epsilon^{-1} pre-computation for contour nodes
///   - No on-the-fly response/inverse/dielectric for real-axis
///   - All W contributions via IA-pair summation
///
/// Arguments:
///   ri_oo_T: [no², naux] — transposed RI OO matrix
///   ri_vv_T: [nv², naux] — transposed RI VV matrix
///   ri_IA_T: [nIA, naux] — transposed RI OV (IA pairs) matrix
///   delta_IA: [nIA] — ε_vir[A] - ε_occ[I]
///   eps_occ: [no], eps_vir: [nv] — orbital energies
///   contour_nodes: pre-computed quadrature nodes (just geometry, no dressing)
///   block_size: IA batch size (auto-selected)
pub fn dynamic_bse_solve(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    occ_size: usize,
    vir_size: usize,
    ri_oo_T: &MatrixFull<f64>,
    ri_vv_T: &MatrixFull<f64>,
    ri_IA_T: &MatrixFull<f64>,
    delta_IA: &[f64],
    eps_occ: &[f64],
    eps_vir: &[f64],
    contour_nodes: &[ContourNodeDataSimple],
    centre: f64,
    radius: f64,
    m0: usize,
    n_quad: usize,
    max_iter: usize,
    tol: f64,
    gmres_restart: usize,
    gmres_max_it: usize,
    gmres_tol: f64,
    tda: bool,
    block_size: usize,
) -> NLFeastResult {
    let n = occ_size * vir_size;
    let centre_cplx = Complex::new(centre, 0.0);

    // ── QP energy gaps for preconditioner (always GW QP energies) ──
    let energy_diag: Vec<f64> = {
        let energies = eps_occ.iter().chain(eps_vir.iter()).cloned().collect::<Vec<_>>();
        let mut d = Vec::with_capacity(n);
        for a in 0..vir_size {
            for i in 0..occ_size {
                d.push(energies[occ_size + a] - energies[i]);
            }
        }
        d
    };

    // ── Initial subspace ──
    let mut rng = rand::rng();
    let mut q_mat = MatrixFull::new([n, m0], 0.0);
    let step = if m0 > 1 { (2.0 * radius) / (m0 - 1) as f64 } else { 0.0 };
    let half_step = step * 0.5;
    let a_width = half_step * half_step;
    for j in 0..m0 {
        let e_k = (centre - radius) + j as f64 * step;
        for i in 0..n {
            let de = energy_diag[i] - e_k;
            let weight = (-de * de / a_width.max(1e-30)).exp();
            let sign = if rng.random::<f64>() > 0.5 { 1.0 } else { -1.0 };
            q_mat[[i, j]] = sign * weight;
        }
    }
    q_mat = qr_orthonormalise(&q_mat);

    let mut n_iter = 0;
    let mut prev_lambda: Vec<f64> = Vec::new();
    let mut last_lambda_real: Vec<f64> = Vec::new();
    let mut last_x_mat = MatrixFull::new([n, 0], 0.0);
    let mut last_residuals: Vec<f64> = Vec::new();
    let mut lambda_init: Vec<f64> = {
        let mut d_with_dist: Vec<(f64, &f64)> = energy_diag.iter()
            .map(|d| ((d - centre).abs(), d)).collect();
        d_with_dist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        d_with_dist.iter().take(m0).map(|(_, &d)| d).collect()
    };
    if lambda_init.len() < m0 {
        lambda_init.resize(m0, centre);
    }

    for it in 0..max_iter {
        n_iter = it + 1;
        let m0_eff = q_mat.size[1];
        if m0_eff == 0 { break; }
        lambda_init.truncate(m0_eff);

        // ── Step 1: Solve projected NLEVP ──
        let (all_lambda, all_x) = if tda {
            solve_projected_tda_new(
                scf_data, qp_ctrl, occ_size, vir_size, n,
                ri_oo_T, ri_vv_T, ri_IA_T, delta_IA, eps_occ, eps_vir,
                &q_mat, &lambda_init, block_size,
            )
        } else {
            solve_projected_bse_new(
                scf_data, qp_ctrl, occ_size, vir_size, n,
                ri_oo_T, ri_vv_T, ri_IA_T, delta_IA, eps_occ, eps_vir,
                &q_mat, &lambda_init, block_size,
            )
        };
        if it == 0 {
            eprint!("  DynamicBSE initial eigenvalue estimates (first 8):");
            for j in 0..8.min(all_lambda.len()) {
                eprint!(" {:.4}", all_lambda[j].re);
            }
            eprintln!();
        }

        // ── Step 2: Select m0_eff eigenvalues closest to contour centre ──
        let n_all = all_lambda.len();
        let mut with_dist: Vec<(f64, usize)> = (0..n_all)
            .map(|idx| ((all_lambda[idx] - centre_cplx).norm(), idx))
            .collect();
        with_dist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let lambda_cur: Vec<Complex<f64>> = with_dist[..m0_eff.min(n_all)]
            .iter().map(|(_, idx)| all_lambda[*idx]).collect();
        let mut x_mat = MatrixFull::new([n, lambda_cur.len()], 0.0);
        for (pj, (_, idx)) in with_dist[..lambda_cur.len()].iter().enumerate() {
            for i in 0..n { x_mat[[i, pj]] = all_x[[i, *idx]]; }
        }
        let m0_eff = lambda_cur.len();

        // ── Step 3: Convergence check ──
        let mut n_inside = 0;
        let mut n_inside_converged = 0;
        let mut max_resid_inside = 0.0_f64;
        let mut residuals = Vec::with_capacity(m0_eff);
        let mut inside_flags = vec![false; m0_eff];

        for j in 0..m0_eff {
            let lam = lambda_cur[j];
            let lam_real = lam.re;
            let xj: Vec<f64> = (0..n).map(|i| x_mat[[i, j]]).collect();
            let txj = if tda {
                compute_real_tda_matvec_new(
                    scf_data, qp_ctrl, occ_size, vir_size,
                    ri_oo_T, ri_vv_T, ri_IA_T, delta_IA, eps_occ, eps_vir,
                    lam_real, &xj, block_size,
                )
            } else {
                compute_real_matvec_new(
                    scf_data, qp_ctrl, occ_size, vir_size,
                    ri_oo_T, ri_vv_T, ri_IA_T, delta_IA, eps_occ, eps_vir,
                    lam_real, &xj, block_size,
                )
            };
            let res: f64 = txj.iter().map(|&v| v * v).sum::<f64>().sqrt();
            residuals.push(res);

            if (lam - centre_cplx).norm() <= radius {
                inside_flags[j] = true;
                n_inside += 1;
                if res <= tol { n_inside_converged += 1; }
                if res > max_resid_inside { max_resid_inside = res; }
            }
        }

        eprintln!(
            "DynamicBSE iter {}: max ‖T(λ)x‖_inside = {:.2e}, inside {}/{}",
            n_iter, max_resid_inside, n_inside_converged, n_inside
        );

        if n_inside > 0 && n_inside_converged == n_inside {
            eprintln!("  → All interior eigenvalues converged.");
            let mut result = build_final_result_dynamic(
                n, &lambda_cur.iter().map(|l| l.re).collect::<Vec<_>>(),
                &x_mat, &residuals, &inside_flags, centre, radius,
            );
            result.iterations = n_iter;
            return result;
        }

        // Stagnation detection
        let lambda_real: Vec<f64> = lambda_cur.iter().map(|l| l.re).collect();
        let improved = if prev_lambda.len() == lambda_real.len() {
            let mut d = 0.0;
            for j in 0..lambda_real.len() { d += (lambda_real[j] - prev_lambda[j]).abs(); }
            d / lambda_real.len() as f64
        } else { 1.0 };
        if n_iter > 3 && improved < 1e-12 && n_inside > 0 {
            eprintln!("  → Eigenvalues stabilised (Δλ ≈ {:.2e}). Returning interior results.", improved);
            let mut result = build_final_result_dynamic(
                n, &lambda_real, &x_mat, &residuals, &inside_flags, centre, radius,
            );
            result.iterations = n_iter;
            return result;
        }
        last_lambda_real = lambda_real.clone();
        last_x_mat = x_mat.clone();
        last_residuals = residuals.clone();

        lambda_init = lambda_real.clone();
        prev_lambda = lambda_real;

        // ── Step 4: Contour integration ──
        // Compute T(Λ)·X on the real axis
        let mut tx_mat = MatrixFull::new([n, m0_eff], 0.0);
        for j in 0..m0_eff {
            let lam = lambda_cur[j].re;
            let xj: Vec<f64> = (0..n).map(|i| x_mat[[i, j]]).collect();
            let txj = if tda {
                compute_real_tda_matvec_new(
                    scf_data, qp_ctrl, occ_size, vir_size,
                    ri_oo_T, ri_vv_T, ri_IA_T, delta_IA, eps_occ, eps_vir,
                    lam, &xj, block_size,
                )
            } else {
                compute_real_matvec_new(
                    scf_data, qp_ctrl, occ_size, vir_size,
                    ri_oo_T, ri_vv_T, ri_IA_T, delta_IA, eps_occ, eps_vir,
                    lam, &xj, block_size,
                )
            };
            for i in 0..n { tx_mat[[i, j]] = txj[i]; }
        }

        let mut q_new = MatrixFull::new([n, m0_eff], 0.0);

        for node in contour_nodes {
            let c_matvec: Box<dyn Fn(&[f64]) -> Vec<f64>> = if tda {
                Box::new(make_shifted_tda_matvec_new(
                    scf_data, qp_ctrl, n,
                    ri_oo_T, ri_vv_T, ri_IA_T, delta_IA, eps_occ, eps_vir,
                    node, block_size,
                ))
            } else {
                Box::new(make_shifted_matvec_new(
                    scf_data, qp_ctrl, n,
                    ri_oo_T, ri_vv_T, ri_IA_T, delta_IA, eps_occ, eps_vir,
                    node, block_size,
                ))
            };

            // Build 2×2 block-diagonal preconditioner
            let z_re = node.z_re;
            let z_im = node.z_im;
            let z_re2 = z_re * z_re - z_im * z_im;
            let z_im2 = 2.0 * z_re * z_im;
            let mut re_part = Vec::with_capacity(n);
            let mut im_part = Vec::with_capacity(n);
            for d in &energy_diag {
                if tda {
                    re_part.push(d - z_re);
                    im_part.push(-z_im);
                } else {
                    let d2 = d * d;
                    re_part.push(d2 - z_re2);
                    im_part.push(-z_im2);
                }
            }
            let gmres_precond = BlockDiagPrecond::from_re_im(&re_part, &im_part);

            // Solve T(z_j)·U_j = T(Λ)·X for each column
            let mut u_re = MatrixFull::new([n, m0_eff], 0.0);
            let mut u_im = MatrixFull::new([n, m0_eff], 0.0);

            for col in 0..m0_eff {
                let mut b_2n = vec![0.0; 2 * n];
                for i in 0..n { b_2n[i] = tx_mat[[i, col]]; }
                let x_2n = gmres(
                    &c_matvec, &b_2n,
                    gmres_restart, gmres_max_it, gmres_tol,
                    col == 0, Some(&gmres_precond),
                );
                for i in 0..n {
                    u_re[[i, col]] = x_2n[i];
                    u_im[[i, col]] = x_2n[n + i];
                }
            }

            // Accumulate: Q_new += w_j · (X − U_j) · (z_j I − Λ)⁻¹
            for col in 0..m0_eff {
                let lam = lambda_cur[col];
                let dz_re = node.z_re - lam.re;
                let dz_im = node.z_im - lam.im;
                let denom = dz_re * dz_re + dz_im * dz_im;
                if denom < 1e-30 { continue; }
                let inv_re = dz_re / denom;
                let inv_im = -dz_im / denom;

                let f_re = node.w_re * inv_re - node.w_im * inv_im;
                let f_im = node.w_re * inv_im + node.w_im * inv_re;

                for i in 0..n {
                    let dx_re = x_mat[[i, col]] - u_re[[i, col]];
                    let dx_im = -u_im[[i, col]];
                    q_new[[i, col]] += f_re * dx_re - f_im * dx_im;
                }
            }
        }

        // ── Step 5: QR orthogonalise Q_new ──
        q_mat = qr_orthonormalise(&q_new);
        if q_mat.size[1] < m0_eff / 2 {
            eprintln!("  Warning: subspace collapsed, stopping.");
            break;
        }
    }

    // Fallback
    if !last_lambda_real.is_empty() {
        let mut last_inside_flags = vec![false; last_lambda_real.len()];
        for (j, &lam) in last_lambda_real.iter().enumerate() {
            if (lam - centre).abs() <= radius {
                last_inside_flags[j] = true;
            }
        }
        let mut result = build_final_result_dynamic(
            n, &last_lambda_real, &last_x_mat, &last_residuals,
            &last_inside_flags, centre, radius,
        );
        result.iterations = n_iter;
        return result;
    }
    NLFeastResult {
        eigenvalues: vec![],
        eigenvectors: MatrixFull::new([n, 0], 0.0),
        n_found: 0,
        iterations: n_iter,
        residuals: vec![],
    }
}

// ============================================================================
// Top-level entry point
// ============================================================================

/// Top-level entry point for the new dynamic BSE calculation.
pub fn dynamic_bse_main(scf_data: &SCF, qp_ctrl: &QuasiParticle) {
    let start = Instant::now();
    let (start_mo, num_state, occ_size, vir_size, homo, lumo) =
        get_occupation_parameters(scf_data, 'N');
    let n = occ_size * vir_size;

    // ── Orbital energies ──
    let quasiparticle_energies = scf_data.gwqp.0.clone();

    // eps_occ/eps_vir for diagonal and denominator ε_i/ε_a → always GW QP
    let eps_occ: Vec<f64> = quasiparticle_energies[0..occ_size].to_vec();
    let eps_vir: Vec<f64> = quasiparticle_energies[occ_size..occ_size + vir_size].to_vec();

    // IA-pair energy gaps Δ_p = ε_I − ε_A — controlled by bse_qp_polarization
    // (analogous to whether the polarization uses KS or QP energies)
    let epsilon_ia = if qp_ctrl.bse_qp_polarization {
        quasiparticle_energies.clone()
    } else {
        scf_data.eigenvalues[0].clone()
    };
    let eps_occ_ia: Vec<f64> = epsilon_ia[0..occ_size].to_vec();
    let eps_vir_ia: Vec<f64> = epsilon_ia[occ_size..occ_size + vir_size].to_vec();

    let num_auxbas = crate::ri_bse::get_submatrix(scf_data, 'O', 'V', 'N').size[0];
    println!("=============================================");
    println!("  Dynamic BSE (new IA-pair projection scheme)");
    println!("=============================================");
    println!("  occ_size = {}, vir_size = {}, n = {}", occ_size, vir_size, n);
    println!("  num_auxbas = {}", num_auxbas);
    if qp_ctrl.bse_tda {
        println!("  Method: TDA  (A(ω) − ωI)");
    } else {
        println!("  Method: non-TDA  ((A+B)(A-B) − ω²I)");
    }

    // ── Fetch RI integrals ──
    let ri_ov = crate::ri_bse::get_submatrix(scf_data, 'O', 'V', 'N');
    let ri_oo = crate::ri_bse::get_submatrix(scf_data, 'O', 'O', 'N');
    let ri_vv = crate::ri_bse::get_submatrix(scf_data, 'V', 'V', 'N');

    // ── Pre-compute transposed RI matrices ──
    let ri_oo_T = ri_oo.transpose_and_drop();
    let ri_vv_T = ri_vv.transpose_and_drop();
    let ri_IA_T = ri_ov.transpose_and_drop();
    let nIA = occ_size * vir_size;

    // ── Pre-compute Delta_IA ──
    let delta_IA = dynamicbse_matvec::compute_delta_ia(&eps_occ_ia, &eps_vir_ia, occ_size, vir_size);
    if qp_ctrl.bse_qp_polarization {
        println!("  IA-pair gaps: GW quasiparticle energies");
    } else {
        println!("  IA-pair gaps: KS eigenvalues");
    }

    // ── Auto-select block size ──
    let block_size = dynamicbse_matvec::auto_block_size(occ_size, vir_size, nIA);
    println!("  IA block_size = {} (nIA = {})", block_size, nIA);

    // ── Pre-compute contour nodes (simple) ──
    let centre = qp_ctrl.nlfeast_centre;
    let radius = qp_ctrl.nlfeast_radius;
    let n_quad = qp_ctrl.nlfeast_n_quad;
    let m0 = qp_ctrl.nlfeast_m0;

    println!("  Contour: centre={:.6}, radius={:.6}", centre, radius);
    println!("  n_quad={}, m0={}", n_quad, m0);

    super::matvec_trace::set_enabled(qp_ctrl.export_matvec_count);

    let contour_nodes = prepare_contour_nodes_simple(centre, radius, n_quad);
    let prep_time = start.elapsed();
    println!("  Contour preparation time: {:?} (no dressing needed)", prep_time);

    // ── Run Dynamic BSE ──
    let result = dynamic_bse_solve(
        scf_data, qp_ctrl,
        occ_size, vir_size,
        &ri_oo_T, &ri_vv_T, &ri_IA_T,
        &delta_IA, &eps_occ, &eps_vir,
        &contour_nodes,
        centre, radius, m0, n_quad,
        qp_ctrl.nlfeast_max_iter,
        qp_ctrl.nlfeast_tol,
        qp_ctrl.nlfeast_gmres_restart,
        qp_ctrl.nlfeast_gmres_max_it,
        qp_ctrl.nlfeast_gmres_tol,
        qp_ctrl.bse_tda,
        block_size,
    );
    let total_time = start.elapsed();
    println!("  Dynamic BSE total time: {:?}", total_time);
    println!("  Outer iterations: {}", result.iterations);

    // ── Print results ──
    if result.n_found == 0 {
        println!("  No eigenvalues found inside the contour.");
        return;
    }

    println!("\n  Found {} excitation(s) inside [{:.4}, {:.4}]:\n",
        result.n_found, centre - radius, centre + radius);
    println!("  #       Excitation energy (eV)    ‖T(λ)x‖");
    println!("  ───     ─────────────────────    ──────────");
    for k in 0..result.n_found {
        let lam_ev = result.eigenvalues[k] * EV;
        println!("  {:>3}     {:>12.6} eV             {:>9.2e}",
            k, lam_ev, result.residuals[k]);
    }

    if scf_data.mol.ctrl.print_level > 1 {
        for k in 0..result.n_found {
            let xv: Vec<f64> = (0..n).map(|i| result.eigenvectors[[i, k]]).collect();
            println!("\n  Excitation #{}: λ = {:.6} Ha = {:.6} eV",
                k, result.eigenvalues[k], result.eigenvalues[k] * EV);
            super::leading_components(&xv, occ_size, vir_size);
        }
    }

    if qp_ctrl.save_bse_excitations {
        let line = result.eigenvalues.iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        use std::fs::OpenOptions;
        use std::io::Write;
        let mut file = OpenOptions::new()
            .append(true).create(true)
            .open("dynamic_bse_excitations.txt")
            .expect("Failed to open dynamic_bse_excitations.txt");
        writeln!(file, "{}", line).expect("Failed to write excitations");
    }

    if qp_ctrl.save_first_excitation && result.n_found > 0 {
        let save_path = qp_ctrl.save_first_excitation_path.clone();
        use std::fs::OpenOptions;
        use std::io::Write;
        let mut file = OpenOptions::new()
            .append(true).create(true)
            .open(save_path)
            .expect("Failed to open save path");
        writeln!(file, "{}", result.eigenvalues[0]).expect("Failed to write");
    }
}
