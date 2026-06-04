// ============================================================================
// NonLinear FEAST (NLFEAST) for BSE
//
// Solves the nonlinear eigenvalue problem:
//   T(λ)·x = 0,  T(λ) = ((A+B)(A-B) − λ²I)
//
// where A, B are the BSE blocks whose screened-exchange (W) contributions
// depend on λ through the frequency-dependent inverse dielectric matrix.
//
// Algorithm follows Li & Polizzi (2025) / Gavin et al. (2018).
//
// Key design:
//   1. Pre-compute per-quadrature-node dressed RI matrices
//      (response + inverse dielectric + dress) so that the complex
//      contour-integral matvec is fast (pure BLAS, no on-the-fly inversion).
//   2. For the real-axis residual, compute response + inverse dielectric
//      + dress on the fly for each eigenvalue estimate.
//   3. All complex linear algebra is real-embedded (2N real GMRES).
// ============================================================================

use num::Complex;
use rand::Rng;
use rayon::prelude::*;
use std::time::Instant;
use tensors::{MathMatrix, MatrixFull};
use rest_tensors::matrix::matrix_blas_lapack::{
    _dgemm_full, _dgemm_scaled, _dgeev,
};

use crate::ctrl_io::quasiparticle_methods::QuasiParticle;
use crate::ri_gw::get_occupation_parameters;
use crate::scf_io::SCF;

use super::nonlinbse_matvec;

// ============================================================================
// Data structures
// ============================================================================

/// Dressed RI matrices for one quadrature node on the complex contour.
pub struct ContourNodeData {
    pub z_re: f64,
    pub z_im: f64,
    pub w_re: f64,  // quadrature weight (real part)
    pub w_im: f64,  // quadrature weight (imag part)
    // A-block dressed RI (re/im split)
    pub ri_oo_tilde_re: MatrixFull<f64>,
    pub ri_oo_tilde_im: MatrixFull<f64>,
    // B-block dressed RI (re/im split)
    pub ri_ov_tilde_re: MatrixFull<f64>,
    pub ri_ov_tilde_im: MatrixFull<f64>,
}

/// Result of the NLFEAST solver.
pub struct NLFeastResult {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: MatrixFull<f64>,
    pub n_found: usize,
    pub iterations: usize,
    pub residuals: Vec<f64>,
}

// ============================================================================
// 2×2 block-diagonal preconditioner for 2N real-embedded systems
// ============================================================================

/// 2×2 block-diagonal preconditioner for the 2N real-embedded system.
///
/// For each complex index i, the 2×2 block is:
///   block_i = [re  -im]    representing complex value λ = re + i·im
///             [im   re]
///
/// Application: v_out = block_i⁻¹ · v_in  for each block i.
pub struct BlockDiagPrecond {
    pub blocks: Vec<[f64; 4]>,
}

impl BlockDiagPrecond {
    /// Create from real and imaginary parts: block_i = [[re[i], -im[i]], [im[i], re[i]]]
    pub fn from_re_im(re_part: &[f64], im_part: &[f64]) -> Self {
        let n = re_part.len();
        let mut blocks = Vec::with_capacity(n);
        for i in 0..n {
            blocks.push([re_part[i], -im_part[i], im_part[i], re_part[i]]);
        }
        BlockDiagPrecond { blocks }
    }

    /// Apply the inverse: for each block i, solve block_i · out = v_in
    pub fn apply_inverse(&self, v: &[f64]) -> Vec<f64> {
        let n = self.blocks.len();
        let mut out = v.to_vec();
        for i in 0..n {
            let [a, b, c, d] = self.blocks[i];
            let det = a * d - b * c;
            let inv_det = 1.0 / f64::max(det.abs(), 1e-30);
            let r = v[i];
            let s = v[n + i];
            out[i]     = ( d * r - b * s) * inv_det;
            out[n + i] = (-c * r + a * s) * inv_det;
        }
        out
    }

    /// 2-norm of preconditioned vector.
    pub fn norm(&self, v: &[f64]) -> f64 {
        let n = self.blocks.len();
        let mut sum = 0.0;
        for i in 0..n {
            let [a, b, c, d] = self.blocks[i];
            let det = a * d - b * c;
            let inv_det = 1.0 / f64::max(det.abs(), 1e-30);
            let r = v[i];
            let s = v[n + i];
            let pr = ( d * r - b * s) * inv_det;
            let ps = (-c * r + a * s) * inv_det;
            sum += pr * pr + ps * ps;
        }
        sum.sqrt()
    }
}

// ============================================================================
// GMRES Solver (real-embedded, for complex linear systems)
// ============================================================================

pub fn givens_rotation(a: f64, b: f64) -> (f64, f64) {
    if b.abs() < 1e-30 {
        (1.0, 0.0)
    } else if a.abs() > b.abs() {
        let tau = b / a;
        let c = 1.0 / (1.0 + tau * tau).sqrt();
        (c, c * tau)
    } else {
        let tau = a / b;
        let s = 1.0 / (1.0 + tau * tau).sqrt();
        (s * tau, s)
    }
}

pub fn gmres(
    a_mul: impl Fn(&[f64]) -> Vec<f64>,
    b: &[f64],
    restart: usize,
    max_it: usize,
    tol: f64,
    verb: bool,
    precondition: Option<&BlockDiagPrecond>,
) -> Vec<f64> {
    let n = b.len();
    let b_nrm: f64 = if let Some(prec) = precondition {
        prec.norm(b)
    } else {
        b.iter().map(|x| x * x).sum::<f64>().sqrt()
    };
    let abs_tol = tol * b_nrm.max(1e-30);
    let mut x = vec![0.0; n];
    let mut r_norm = b_nrm;
    let mut total_mv = 0_usize;
    let mut n_rest = 0_usize;

    loop {
        if r_norm <= abs_tol || total_mv >= max_it { break; }
        let ax = a_mul(&x);
        total_mv += 1;
        let r_raw: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
        // Apply preconditioner to residual
        let r = if let Some(prec) = precondition {
            prec.apply_inverse(&r_raw)
        } else {
            r_raw
        };
        let beta = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if beta <= abs_tol { r_norm = beta; break; }

        let mut v: Vec<Vec<f64>> = Vec::with_capacity(restart + 1);
        v.push(r.iter().map(|x| x / beta).collect());

        let mut h = vec![0.0f64; (restart + 1) * restart];
        let mut rhs = vec![0.0f64; restart + 1];
        rhs[0] = beta;
        let mut giv_c = vec![0.0f64; restart];
        let mut giv_s = vec![0.0f64; restart];
        let mut inner_k = 0;
        let mut done = false;

        for j in 0..restart {
            if total_mv >= max_it { break; }
            let w0 = a_mul(&v[j]);
            total_mv += 1;
            // Apply preconditioner to Arnoldi vector
            let w0_prec = if let Some(prec) = precondition {
                prec.apply_inverse(&w0)
            } else {
                w0
            };

            let mut w = w0_prec;
            for i in 0..=j {
                let mut dot = 0.0;
                for ii in 0..n { dot += w[ii] * v[i][ii]; }
                h[i * restart + j] = dot;
                for ii in 0..n { w[ii] -= dot * v[i][ii]; }
            }
            let mut nrm = 0.0;
            for ii in 0..n { nrm += w[ii] * w[ii]; }
            nrm = nrm.sqrt();
            if nrm < 1e-15 { h[(j + 1) * restart + j] = nrm; inner_k = j + 1; done = true; break; }
            h[(j + 1) * restart + j] = nrm;
            v.push(w.iter().map(|x| x / nrm).collect());

            for i in 0..j {
                let ci = giv_c[i]; let si = giv_s[i];
                let hij = h[i * restart + j]; let hip1 = h[(i + 1) * restart + j];
                h[i * restart + j] = ci * hij + si * hip1;
                h[(i + 1) * restart + j] = -si * hij + ci * hip1;
            }
            let (cj, sj) = givens_rotation(h[j * restart + j], h[(j + 1) * restart + j]);
            giv_c[j] = cj; giv_s[j] = sj;
            let hjj = h[j * restart + j]; let hjp1 = h[(j + 1) * restart + j];
            h[j * restart + j] = cj * hjj + sj * hjp1;
            h[(j + 1) * restart + j] = 0.0;
            let old = rhs[j]; rhs[j] = cj * old; rhs[j + 1] = -sj * old;
            inner_k = j + 1;
            if rhs[j + 1].abs() <= abs_tol { done = true; break; }
        }

        if inner_k > 0 {
            let mut y = vec![0.0; inner_k];
            for i in (0..inner_k).rev() {
                let mut s = rhs[i];
                for jj in (i + 1)..inner_k { s -= h[i * restart + jj] * y[jj]; }
                y[i] = s / h[i * restart + i];
            }
            for i in 0..n {
                let mut dx = 0.0;
                for jj in 0..inner_k { dx += v[jj][i] * y[jj]; }
                x[i] += dx;
            }
        }
        if done {
            r_norm = rhs[inner_k].abs();
            if verb { eprintln!("  GMRES converged: {} mv, {} rest(s), rel.res {:.2e}", total_mv, n_rest + 1, r_norm / b_nrm); }
            break;
        }
        let ax = a_mul(&x); total_mv += 1;
        let r_restart: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
        let r_restart_prec = if let Some(prec) = precondition {
            prec.apply_inverse(&r_restart)
        } else {
            r_restart
        };
        r_norm = r_restart_prec.iter().map(|v| v * v).sum::<f64>().sqrt();
        n_rest += 1;
        if verb && n_rest % 10 == 0 { eprintln!("  GMRES restart {}: {} mv, rel.res {:.2e}", n_rest, total_mv, r_norm / b_nrm); }
    }
    if verb && r_norm > abs_tol { eprintln!("  WARNING: GMRES not converged after {} mv, rel.res {:.2e}", total_mv, r_norm / b_nrm); }
    x
}

// ============================================================================
// Build the 2N real-embedded shifted matvec from pre-computed contour-node data
// ============================================================================

/// Given pre-computed contour node data, build a 2N×2N real matvec closure
/// for solving T_cplx(z_j)·U = RHS via real embedding:
///
///   M₂·[vr; vi] = [Re(T(z_j)·vr) − Im(T(z_j)·vi);
///                   Im(T(z_j)·vr) + Re(T(z_j)·vi)]
///
/// Each M₂ matvec calls `composite_matvec_complex` twice (vr, vi).
fn make_shifted_matvec<'a>(
    scf_data: &'a SCF,
    qp_ctrl: &'a QuasiParticle,
    n: usize,
    ri_vv: &'a MatrixFull<f64>,
    ri_ov_a: &'a MatrixFull<f64>,
    ri_ov_b: &'a MatrixFull<f64>,
    node: &'a ContourNodeData,
) -> impl Fn(&[f64]) -> Vec<f64> + 'a {
    let z = Complex::new(node.z_re, node.z_im);
    move |v: &[f64]| -> Vec<f64> {
        let vr = &v[0..n];
        let vi = &v[n..2 * n];
        // T(z)·(vr + i·vi) as a single complex matvec
        let (t_re, t_im) = nonlinbse_matvec::composite_matvec_complex(
            scf_data, qp_ctrl, ri_vv, ri_ov_a, ri_ov_b,
            &node.ri_oo_tilde_re, &node.ri_oo_tilde_im,
            &node.ri_ov_tilde_re, &node.ri_ov_tilde_im,
            vr, vi, z,
        );
        // Real embedding: [Re(T(z)·(vr+i·vi)); Im(T(z)·(vr+i·vi))]
        let mut out = vec![0.0; 2 * n];
        for i in 0..n {
            out[i]     = t_re[i];
            out[n + i] = t_im[i];
        }
        out
    }
}

// ============================================================================
// QR orthogonalisation (modified Gram–Schmidt)
// ============================================================================

pub fn qr_orthonormalise(a: &MatrixFull<f64>) -> MatrixFull<f64> {
    let n = a.size[0];
    let k = a.size[1];
    let mut q = a.clone();

    for j in 0..k {
        for i in 0..j {
            let mut dot = 0.0;
            for ii in 0..n { dot += q[[ii, j]] * q[[ii, i]]; }
            for ii in 0..n { q[[ii, j]] -= dot * q[[ii, i]]; }
        }
        let mut nrm = 0.0;
        for ii in 0..n { nrm += q[[ii, j]] * q[[ii, j]]; }
        nrm = nrm.sqrt();
        if nrm > 1e-14 { for ii in 0..n { q[[ii, j]] /= nrm; } }
        else { for ii in 0..n { q[[ii, j]] = 0.0; } }
    }

    let mut valid = Vec::new();
    for j in 0..k {
        let mut nrm = 0.0;
        for i in 0..n { nrm += q[[i, j]] * q[[i, j]]; }
        if nrm.sqrt() > 1e-14 { valid.push(j); }
    }
    if valid.len() == k { return q; }
    let nk = valid.len();
    let mut qt = MatrixFull::new([n, nk], 0.0);
    for (pj, &j) in valid.iter().enumerate() {
        for i in 0..n { qt[[i, pj]] = q[[i, j]]; }
    }
    qt
}

// ============================================================================
// Contour initialisation: pre-compute per-node dressed RI matrices
// ============================================================================

/// Pre-compute all contour-node data.
///
/// For each quadrature node z_j on the circle (centre c, radius r):
///   1. response_matrix_complex(z_j)          → complex response
///   2. inverse_dielectric_matrix_complex()   → (re, im) pair
///   3. ri_oo_tilde_re/im = ϵ⁻¹_re/im · ri_oo, same reshape as in mod.rs
///   4. ri_ov_tilde_re/im = ϵ⁻¹_re/im · ri_ov, same reshape as in mod.rs
pub fn prepare_contour_nodes(
    quasiparticle_energies: &Vec<f64>,
    occ_size: usize,
    vir_size: usize,
    ri_ov_response: &MatrixFull<f64>,
    ri_oo: &MatrixFull<f64>,
    ri_ov: &MatrixFull<f64>,
    centre: f64,
    radius: f64,
    n_quad: usize,
) -> Vec<ContourNodeData> {
    let num_auxbas = ri_ov_response.size[0];

    // Trapezoidal rule on the circle z(θ) = c + r·exp(iθ)
    // Weight at node: w_j = (z_j − c) / n_quad
    let mut nodes = Vec::with_capacity(n_quad);

    for j in 0..n_quad {
        let theta = 2.0 * std::f64::consts::PI * j as f64 / n_quad as f64;
        let (ct, st) = (theta.cos(), theta.sin());
        let z_re = centre + radius * ct;
        let z_im = radius * st;

        // 1. Complex response at z_j
        let response = nonlinbse_matvec::response_matrix_complex(
            quasiparticle_energies, occ_size, vir_size, ri_ov_response,
            Complex::new(z_re, z_im),
        );

        // 2. Split inverse dielectric into real and imaginary parts
        let (inv_re, inv_im) = nonlinbse_matvec::inverse_dielectric_matrix_complex(&response);

        // 3. Dress RI_oo (A block)
        let mut ri_oo_tilde_re = MatrixFull::new(ri_oo.size, 0.0);
        let mut ri_oo_tilde_im = MatrixFull::new(ri_oo.size, 0.0);
        _dgemm_full(&inv_re, 'N', ri_oo, 'N', &mut ri_oo_tilde_re, 1.0, 0.0);
        _dgemm_full(&inv_im, 'N', ri_oo, 'N', &mut ri_oo_tilde_im, 1.0, 0.0);
        // Reshape as in mod.rs for A-block matvec
        ri_oo_tilde_re.reshape([num_auxbas * occ_size, occ_size]);
        ri_oo_tilde_re = ri_oo_tilde_re.transpose_and_drop();
        ri_oo_tilde_re.reshape([occ_size * num_auxbas, occ_size]);
        ri_oo_tilde_im.reshape([num_auxbas * occ_size, occ_size]);
        ri_oo_tilde_im = ri_oo_tilde_im.transpose_and_drop();
        ri_oo_tilde_im.reshape([occ_size * num_auxbas, occ_size]);

        // 4. Dress RI_ov (B block)
        let mut ri_ov_tilde_re = MatrixFull::new(ri_ov.size, 0.0);
        let mut ri_ov_tilde_im = MatrixFull::new(ri_ov.size, 0.0);
        _dgemm_full(&inv_re, 'N', ri_ov, 'N', &mut ri_ov_tilde_re, 1.0, 0.0);
        _dgemm_full(&inv_im, 'N', ri_ov, 'N', &mut ri_ov_tilde_im, 1.0, 0.0);
        // Reshape as in mod.rs for B-block matvec
        ri_ov_tilde_re.reshape([num_auxbas * occ_size, vir_size]);
        ri_ov_tilde_im.reshape([num_auxbas * occ_size, vir_size]);

        // Quadrature weight: w = (z - c) / n_quad
        let w_re = radius * ct / n_quad as f64;
        let w_im = radius * st / n_quad as f64;

        nodes.push(ContourNodeData {
            z_re, z_im, w_re, w_im,
            ri_oo_tilde_re, ri_oo_tilde_im,
            ri_ov_tilde_re, ri_ov_tilde_im,
        });
    }

    nodes
}

// ============================================================================
// On-the-fly real-axis matvec  T_real(λ)·x
// ============================================================================

/// Evaluate T_real(λ)·x for a real λ and real vector x.
///
///   T(λ) = ((A+B)(A-B) − λ²I)
///
/// This is the **real-axis** operator used for residual computation.
/// For each λ, we compute the response matrix, invert the dielectric,
/// dress the RI matrices, and then apply the composite matvec — all
/// on the fly.
pub fn compute_real_matvec(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    quasiparticle_energies: &Vec<f64>,
    occ_size: usize,
    vir_size: usize,
    ri_ov_response: &MatrixFull<f64>,
    ri_ov: &MatrixFull<f64>,
    ri_oo: &MatrixFull<f64>,
    ri_vv: &MatrixFull<f64>,
    lambda: f64,
    x: &[f64],
) -> Vec<f64> {
    let num_auxbas = ri_ov_response.size[0];

    // 1. Real response at λ
    let response = nonlinbse_matvec::response_matrix_real(
        quasiparticle_energies, occ_size, vir_size, ri_ov_response, lambda,
    );

    // 2. Real inverse dielectric
    let inv = nonlinbse_matvec::inverse_dielectric_matrix_real(&response);

    // 3. Dress RI_oo (A block) — real
    let mut ri_oo_tilde = MatrixFull::new(ri_oo.size, 0.0);
    _dgemm_full(&inv, 'N', ri_oo, 'N', &mut ri_oo_tilde, 1.0, 0.0);
    ri_oo_tilde.reshape([num_auxbas * occ_size, occ_size]);
    ri_oo_tilde = ri_oo_tilde.transpose_and_drop();
    ri_oo_tilde.reshape([occ_size * num_auxbas, occ_size]);

    // 4. Dress RI_ov (B block) — real
    let mut ri_ov_tilde = MatrixFull::new(ri_ov.size, 0.0);
    _dgemm_full(&inv, 'N', ri_ov, 'N', &mut ri_ov_tilde, 1.0, 0.0);
    ri_ov_tilde.reshape([num_auxbas * occ_size, vir_size]);

    // 5. RI_ov_b = reshaped ri_ov (for B-block matvec)
    let mut ri_ov_b = ri_ov.clone();
    ri_ov_b.reshape([num_auxbas * occ_size, vir_size]);

    // 6. Apply real composite matvec
    let result = nonlinbse_matvec::composite_matvec_real(
        scf_data, qp_ctrl,
        ri_vv, ri_ov, &ri_ov_b,
        &ri_oo_tilde, &ri_ov_tilde,
        x, lambda,
    );

    result
}

// ============================================================================
// Projected NLEVP solver for the current subspace Q
// ============================================================================

/// Solve the projected problem Q^T T(λ) Q y = 0 in the current subspace.
///
/// For T(λ) = ((A+B)(A-B) − λ²I) and Q orthonormal, this becomes:
///   M_Q · y = λ² · y    where M_Q = Q^T · (A+B)(A-B) · Q
///
/// KEY:  Uses the eigenvalue estimate λ_j for each column j to compute
/// the frequency-dependent screened exchange W(λ_j).  This makes M_Q
/// non-symmetric (different λ per column), handled by dgeev.
///
/// `lambda_init` — current eigenvalue estimates, one per column of Q.
fn solve_projected_bse(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    quasiparticle_energies: &Vec<f64>,
    occ_size: usize,
    vir_size: usize,
    n: usize,
    ri_vv: &MatrixFull<f64>,
    ri_ov: &MatrixFull<f64>,
    ri_ov_response: &MatrixFull<f64>,
    ri_oo: &MatrixFull<f64>,
    q: &MatrixFull<f64>,
    lambda_init: &[f64],
) -> (Vec<Complex<f64>>, MatrixFull<f64>) {
    let m0 = q.size[1];
    let num_auxbas = ri_ov_response.size[0];
    assert_eq!(lambda_init.len(), m0, "lambda_init must have m0 entries");

    // ── Apply (A+B)(A-B) at λ_j to each column q_j ──
    let mut aq = MatrixFull::new([n, m0], 0.0);
    for j in 0..m0 {
        let lambda = lambda_init[j];
        let qj: Vec<f64> = (0..n).map(|i| q[[i, j]]).collect();

        // Response and inverse dielectric at the eigenvalue estimate λ_j
        let response = nonlinbse_matvec::response_matrix_real(
            quasiparticle_energies, occ_size, vir_size, ri_ov_response, lambda,
        );
        let inv = nonlinbse_matvec::inverse_dielectric_matrix_real(&response);

        // Dress RI_oo at λ_j — same reshape as mod.rs
        let mut ri_oo_tilde = MatrixFull::new(ri_oo.size, 0.0);
        _dgemm_full(&inv, 'N', ri_oo, 'N', &mut ri_oo_tilde, 1.0, 0.0);
        ri_oo_tilde.reshape([num_auxbas * occ_size, occ_size]);
        ri_oo_tilde = ri_oo_tilde.transpose_and_drop();
        ri_oo_tilde.reshape([occ_size * num_auxbas, occ_size]);

        // Dress RI_ov at λ_j
        let mut ri_ov_tilde = MatrixFull::new(ri_ov.size, 0.0);
        _dgemm_full(&inv, 'N', ri_ov, 'N', &mut ri_ov_tilde, 1.0, 0.0);
        ri_ov_tilde.reshape([num_auxbas * occ_size, vir_size]);

        // RI_ov_b for B block
        let mut ri_ov_b = ri_ov.clone();
        ri_ov_b.reshape([num_auxbas * occ_size, vir_size]);

        // (A+B)(A-B) at λ_j applied to q_j  (use ω=0 — the -λ²I term
        // is NOT part of the subspace projection M_Q = Q^T·(A+B)(A-B)·Q)
        let result = nonlinbse_matvec::composite_matvec_real(
            scf_data, qp_ctrl, ri_vv, ri_ov, &ri_ov_b,
            &ri_oo_tilde, &ri_ov_tilde, &qj, 0.0,
        );
        for i in 0..n {
            aq[[i, j]] = result[i];
        }
    }

    // M_Q = Q^T · AQ  (m₀ × m₀, non-symmetric in general)
    let mq = _dgemm_scaled(q, 'T', &aq, 'N', 1.0);

    // Solve M_Q · y = μ · y  via dgeev
    let (_vr_dummy, wr, wi, _vl, vr, info) = _dgeev(&mq, 'N', 'V');
    if info != 0 {
        eprintln!("Warning: dgeev in solve_projected_bse returned info={}", info);
    }

    // λ = √μ, keep real part only (physical excitation energies)
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

// ============================================================================
// Build final result from interior eigenpairs
// ============================================================================

fn build_final_result(
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
// NLFEAST Main Algorithm
// ============================================================================

/// Run the NLFEAST solver for the BSE nonlinear eigenvalue problem.
///
/// The operator is T(λ) = ((A+B)(A-B) − λ²I) where A and B are the BSE
/// blocks whose W (screened-exchange) part depends on λ through the
/// frequency-dependent inverse dielectric matrix.
///
/// # Arguments
///
/// * `scf_data`  — SCF data (energies, occupation, etc.)
/// * `qp_ctrl`   — Quasiparticle control parameters
/// * `quasiparticle_energies` — QP energies (used for response matrix)
/// * `occ_size`, `vir_size` — occupied / virtual orbital counts
/// * `ri_ov_response` — RI OV integrals for response (original shape [num_auxbas, occ*vir])
/// * `ri_ov` — RI OV integrals for B block (original shape [num_auxbas, occ*vir])
/// * `ri_oo` — RI OO integrals (original shape [num_auxbas, occ²])
/// * `ri_vv_reshaped` — RI VV reshaped to [num_auxbas*vir, vir]
/// * `contour_nodes` — pre-computed per-quadrature-node data
/// * `centre`, `radius` — search contour
/// * `m0` — initial subspace dimension (> expected eigenvalues inside)
/// * `n_quad` — number of quadrature nodes
/// * `max_iter` — max outer iterations
/// * `tol` — convergence tolerance on ‖T(λ)·x‖
/// * `gmres_restart`, `gmres_max_it`, `gmres_tol` — GMRES parameters
pub fn nlfeast_bse(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    quasiparticle_energies: &Vec<f64>,
    occ_size: usize,
    vir_size: usize,
    ri_ov_response: &MatrixFull<f64>,
    ri_ov: &MatrixFull<f64>,
    ri_oo: &MatrixFull<f64>,
    ri_vv_reshaped: &MatrixFull<f64>,
    contour_nodes: &[ContourNodeData],
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
) -> NLFeastResult {
    let n = occ_size * vir_size;
    let num_auxbas = ri_ov_response.size[0];
    let centre_cplx = Complex::new(centre, 0.0);

    // ── RI_ov reshaped for B-block Coulomb contribution ──
    let mut ri_ov_b = ri_ov.clone();
    ri_ov_b.reshape([num_auxbas * occ_size, vir_size]);

    // ── Build QP energy gaps for initial guess and preconditioner ──
    let energy_diag: Vec<f64> = {
        let energies: Vec<f64> = if qp_ctrl.bse_qp_polarization {
            quasiparticle_energies.clone()
        } else {
            scf_data.eigenvalues[0].clone()
        };
        let mut d = Vec::with_capacity(n);
        for a in 0..vir_size {
            for i in 0..occ_size {
                d.push(energies[occ_size + a] - energies[i]);
            }
        }
        d
    };

    // ── Initial subspace: Gaussian-weighted by QP energy gaps ──
    let mut rng = rand::thread_rng();
    let mut q_mat = MatrixFull::new([n, m0], 0.0);
    let step = if m0 > 1 { (2.0 * radius) / (m0 - 1) as f64 } else { 0.0 };
    let half_step = step * 0.5;
    let a_width = half_step * half_step;
    for j in 0..m0 {
        let e_k = (centre - radius) + j as f64 * step;
        for i in 0..n {
            let de = energy_diag[i] - e_k;
            let weight = (-de * de / a_width.max(1e-30)).exp();
            let sign = if rng.gen::<f64>() > 0.5 { 1.0 } else { -1.0 };
            q_mat[[i, j]] = sign * weight;
        }
    }
    q_mat = qr_orthonormalise(&q_mat);

    let mut n_iter = 0;
    let mut prev_lambda: Vec<f64> = Vec::new();
    let mut last_lambda_real: Vec<f64> = Vec::new();
    let mut last_x_mat = MatrixFull::new([n, 0], 0.0);
    let mut last_residuals: Vec<f64> = Vec::new();
    // lambda_init for solve_projected: QP gaps (1st iter), then previous Lambda
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
        // Truncate lambda_init to match compressed subspace size
        lambda_init.truncate(m0_eff);

        // ---- Step 1: Solve projected NLEVP at current lambda_init ----
        let (all_lambda, all_x) = if tda {
            solve_projected_tda(
                scf_data, qp_ctrl, quasiparticle_energies,
                occ_size, vir_size, n,
                ri_vv_reshaped, ri_ov, ri_ov_response, ri_oo,
                &q_mat, &lambda_init,
            )
        } else {
            solve_projected_bse(
            scf_data, qp_ctrl, quasiparticle_energies,
            occ_size, vir_size, n,
            ri_vv_reshaped, ri_ov, ri_ov_response, ri_oo,
            &q_mat, &lambda_init,
            )
        };
        if it == 0 {
            eprint!("  Initial eigenvalue estimates (first 8):");
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

        // ── Step 3: Convergence check (real-axis residuals) ──
        let mut n_inside = 0;
        let mut n_inside_converged = 0;
        let mut max_resid_inside = 0.0_f64;
        let mut residuals = Vec::with_capacity(m0_eff);
        let mut inside_flags = vec![false; m0_eff];

        for j in 0..m0_eff {
            let lam = lambda_cur[j];
            let lam_real = lam.re;
            let xj: Vec<f64> = (0..n).map(|i| x_mat[[i, j]]).collect();
            // On-the-fly real-axis matvec
            let txj = if tda {
                compute_real_tda_matvec(
                    scf_data, qp_ctrl, quasiparticle_energies,
                    occ_size, vir_size,
                    ri_ov_response, ri_oo, ri_vv_reshaped, ri_ov,
                    lam_real, &xj,
                )
            } else {
                compute_real_matvec(
                    scf_data, qp_ctrl, quasiparticle_energies,
                    occ_size, vir_size,
                    ri_ov_response, ri_ov, ri_oo, ri_vv_reshaped,
                    lam_real, &xj,
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
            "NLFEAST iter {}: max ‖T(λ)x‖_inside = {:.2e}, inside {}/{}",
            n_iter, max_resid_inside, n_inside_converged, n_inside
        );

        if n_inside > 0 && n_inside_converged == n_inside {
            eprintln!("  → All {} interior eigenvalues converged.", n_inside);
            let mut result = build_final_result(
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
            let mut result = build_final_result(
                n, &lambda_real, &x_mat, &residuals, &inside_flags, centre, radius,
            );
            result.iterations = n_iter;
            return result;
        }
        // Save best results for fallback
        last_lambda_real = lambda_real.clone();
        last_x_mat = x_mat.clone();
        last_residuals = residuals.clone();

        // Update lambda_init for the next solve_projected iteration
        lambda_init = lambda_real.clone();

        prev_lambda = lambda_real;

        // ── Step 4: Contour integration to update subspace ──
        //   Q_new = Σ_j w_j · (X − T(z_j)⁻¹·T(X,Λ)) · (z_j I − Λ)⁻¹
        //
        // We first compute the block residual T(Λ)·X (on the real axis),
        // then solve T(z_j)·U_j = T(Λ)·X for each node, and accumulate.

        // Compute T(Λ)·X on the real axis
        let mut tx_mat = MatrixFull::new([n, m0_eff], 0.0);
        for j in 0..m0_eff {
            let lam = lambda_cur[j].re;
            let xj: Vec<f64> = (0..n).map(|i| x_mat[[i, j]]).collect();
            let txj = if tda {
                compute_real_tda_matvec(
                    scf_data, qp_ctrl, quasiparticle_energies,
                    occ_size, vir_size,
                    ri_ov_response, ri_oo, ri_vv_reshaped, ri_ov,
                    lam, &xj,
                )
            } else {
                compute_real_matvec(
                    scf_data, qp_ctrl, quasiparticle_energies,
                    occ_size, vir_size,
                    ri_ov_response, ri_ov, ri_oo, ri_vv_reshaped,
                    lam, &xj,
                )
            };
            for i in 0..n { tx_mat[[i, j]] = txj[i]; }
        }

        let mut q_new = MatrixFull::new([n, m0_eff], 0.0);

        for node in contour_nodes {
            let c_matvec: Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync> = if tda {
                Box::new(make_shifted_tda_matvec(
                    scf_data, qp_ctrl, n,
                    ri_vv_reshaped, ri_ov, node,
                ))
            } else {
                Box::new(make_shifted_matvec(
                    scf_data, qp_ctrl, n,
                    ri_vv_reshaped, ri_ov, &ri_ov_b,
                    node,
                ))
            };

            // Build 2×2 block-diagonal preconditioner
            //   re = D² − (z_re² − z_im²)
            //   im = −2·z_re·z_im
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

            // Solve T(z_j)·U_j = T(Λ)·X  for each column
            let mut u_re = MatrixFull::new([n, m0_eff], 0.0);
            let mut u_im = MatrixFull::new([n, m0_eff], 0.0);

            for col in 0..m0_eff {
                // Build the 2N RHS: [tx_col; 0]  (RHS is always real)
                let mut b_2n = vec![0.0; 2 * n];
                for i in 0..n { b_2n[i] = tx_mat[[i, col]]; }
                let x_2n = gmres(&c_matvec, &b_2n, gmres_restart, gmres_max_it, gmres_tol, col == 0, Some(&gmres_precond));
                for i in 0..n {
                    u_re[[i, col]] = x_2n[i];
                    u_im[[i, col]] = x_2n[n + i];
                }
            }

            // Accumulate: Q_new += w_j · (X − U_j) · (z_j I − Λ)⁻¹
            for col in 0..m0_eff {
                let lam = lambda_cur[col];
                // Denominator: (z_j - λ)  ∈ ℂ
                let dz_re = node.z_re - lam.re;
                let dz_im = node.z_im - lam.im;
                let denom = dz_re * dz_re + dz_im * dz_im;
                if denom < 1e-30 { continue; }
                let inv_re = dz_re / denom;
                let inv_im = -dz_im / denom;

                // Combined factor: f = w_j · (z_j - λ)⁻¹  ∈ ℂ
                let f_re = node.w_re * inv_re - node.w_im * inv_im;
                let f_im = node.w_re * inv_im + node.w_im * inv_re;

                for i in 0..n {
                    // (X − U_j)  → X is real, U_j is complex
                    let dx_re = x_mat[[i, col]] - u_re[[i, col]];
                    let dx_im = -u_im[[i, col]];
                    // Apply f = f_re + i·f_im
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

    // Fallback: return best available interior results from last iteration
    if !last_lambda_real.is_empty() {
        let mut last_inside_flags = vec![false; last_lambda_real.len()];
        for (j, &lam) in last_lambda_real.iter().enumerate() {
            if (lam - centre).abs() <= radius {
                last_inside_flags[j] = true;
            }
        }
        let mut result = build_final_result(
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
// TDA-specific helpers (single-block A(ω))
// ============================================================================

/// TDA:  T(λ)·x = A(λ)·x − λ·x  (real-axis residual)
fn compute_real_tda_matvec(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    quasiparticle_energies: &Vec<f64>,
    occ_size: usize,
    vir_size: usize,
    ri_ov_response: &MatrixFull<f64>,
    ri_oo: &MatrixFull<f64>,
    ri_vv: &MatrixFull<f64>,
    ri_ov: &MatrixFull<f64>,
    lambda: f64,
    x: &[f64],
) -> Vec<f64> {
    let num_auxbas = ri_ov_response.size[0];
    // Response and inverse dielectric at λ
    let response = nonlinbse_matvec::response_matrix_real(
        quasiparticle_energies, occ_size, vir_size, ri_ov_response, lambda,
    );
    let inv = nonlinbse_matvec::inverse_dielectric_matrix_real(&response);

    // Dress RI_oo at λ — same reshape as mod.rs
    let mut ri_oo_tilde = MatrixFull::new(ri_oo.size, 0.0);
    _dgemm_full(&inv, 'N', ri_oo, 'N', &mut ri_oo_tilde, 1.0, 0.0);
    ri_oo_tilde.reshape([num_auxbas * occ_size, occ_size]);
    ri_oo_tilde = ri_oo_tilde.transpose_and_drop();
    ri_oo_tilde.reshape([occ_size * num_auxbas, occ_size]);

    // A(λ)·x via matvec.rs a_block_matvec
    let a_x = super::matvec::a_block_matvec(
        scf_data, qp_ctrl, ri_vv, ri_ov, &ri_oo_tilde, &x.to_vec(),
    );

    // T(λ)·x = A(λ)·x − λ·x
    a_x.iter().zip(x.iter()).map(|(a, &xi)| a - lambda * xi).collect()
}

/// TDA shifted matvec: T(z)·(vr+i·vi) = A(z)·(vr+i·vi) − z·(vr+i·vi)
fn make_shifted_tda_matvec<'a>(
    scf_data: &'a SCF,
    qp_ctrl: &'a QuasiParticle,
    n: usize,
    ri_vv: &'a MatrixFull<f64>,
    ri_ov: &'a MatrixFull<f64>,
    node: &'a ContourNodeData,
) -> impl Fn(&[f64]) -> Vec<f64> + 'a {
    let z = Complex::new(node.z_re, node.z_im);
    move |v: &[f64]| -> Vec<f64> {
        let vr = &v[0..n];
        let vi = &v[n..2 * n];
        // A(z)·(vr + i·vi) as a single complex matvec
        let (a_re, a_im) = nonlinbse_matvec::a_block_matvec_complex(
            scf_data, qp_ctrl, ri_vv, ri_ov,
            &node.ri_oo_tilde_re, &node.ri_oo_tilde_im,
            vr, vi,
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

/// TDA projected solve: M_Q = Q^T · A(0) · Q,  M_Q · y = λ · y
fn solve_projected_tda(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    quasiparticle_energies: &Vec<f64>,
    occ_size: usize,
    vir_size: usize,
    n: usize,
    ri_vv: &MatrixFull<f64>,
    ri_ov: &MatrixFull<f64>,
    ri_ov_response: &MatrixFull<f64>,
    ri_oo: &MatrixFull<f64>,
    q: &MatrixFull<f64>,
    lambda_init: &[f64],
) -> (Vec<Complex<f64>>, MatrixFull<f64>) {
    let m0 = q.size[1];
    let num_auxbas = ri_ov_response.size[0];
    assert_eq!(lambda_init.len(), m0, "lambda_init must have m0 entries");

    // ── Apply A(λ_j) to each column q_j ──
    let mut aq = MatrixFull::new([n, m0], 0.0);
    for j in 0..m0 {
        let lambda = lambda_init[j];
        let qj: Vec<f64> = (0..n).map(|i| q[[i, j]]).collect();

        // Response and inverse dielectric at λ_j
        let response = nonlinbse_matvec::response_matrix_real(
            quasiparticle_energies, occ_size, vir_size, ri_ov_response, lambda,
        );
        let inv = nonlinbse_matvec::inverse_dielectric_matrix_real(&response);

        // Dress RI_oo at λ_j
        let mut ri_oo_tilde = MatrixFull::new(ri_oo.size, 0.0);
        _dgemm_full(&inv, 'N', ri_oo, 'N', &mut ri_oo_tilde, 1.0, 0.0);
        ri_oo_tilde.reshape([num_auxbas * occ_size, occ_size]);
        ri_oo_tilde = ri_oo_tilde.transpose_and_drop();
        ri_oo_tilde.reshape([occ_size * num_auxbas, occ_size]);

        // A(λ_j)·q_j via matvec.rs a_block_matvec
        let result = super::matvec::a_block_matvec(
            scf_data, qp_ctrl, ri_vv, ri_ov, &ri_oo_tilde, &qj,
        );
        for i in 0..n {
            aq[[i, j]] = result[i];
        }
    }

    // M_Q = Q^T · AQ
    let mq = _dgemm_scaled(q, 'T', &aq, 'N', 1.0);

    // Solve M_Q · y = λ · y  via dgeev
    let (_vr_dummy, wr, wi, _vl, vr, info) = _dgeev(&mq, 'N', 'V');
    if info != 0 {
        eprintln!("Warning: dgeev in solve_projected_tda returned info={}", info);
    }

    // λ = eigenvalues of M_Q (real part only)
    let mut lambdas = Vec::with_capacity(m0);
    let mut ritz = MatrixFull::new([n, m0], 0.0);
    for j in 0..m0 {
        let lam = Complex::new(wr[j].max(0.0), 0.0);
        lambdas.push(lam);
        // Ritz vector = Q · y_j
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
// Top-level entry point: prepare data and run NLFEAST for BSE
// ============================================================================

/// Top-level entry point for NLFEAST BSE calculation.
pub fn nlfeast_bse_main(scf_data: &SCF, qp_ctrl: &QuasiParticle) {
    let start = Instant::now();
    let (start_mo, num_state, occ_size, vir_size, homo, lumo) =
        get_occupation_parameters(scf_data, 'N');
    let n = occ_size * vir_size;
    let quasiparticle_energies = scf_data.gwqp.0.clone();

    let ks_energies: Vec<f64> = scf_data.eigenvalues[0].clone();
    let mut epsilon = ks_energies.clone();
    if qp_ctrl.bse_qp_polarization {
        epsilon = quasiparticle_energies.clone();
    }

    let num_auxbas = super::get_submatrix(scf_data, 'O', 'V', 'N').size[0];
    println!("========================================");
    println!("  NLFEAST BSE Calculation");
    println!("========================================");
    println!("  occ_size = {}, vir_size = {}, n = {}", occ_size, vir_size, n);
    println!("  num_auxbas = {}", num_auxbas);
    if qp_ctrl.bse_tda {
        println!("  Method: TDA  (A(ω) − ωI)");
    } else {
        println!("  Method: non-TDA  ((A+B)(A-B) − ω²I)");
    }

    // ── Fetch and reshape RI integrals ──
    let ri_ov_response = super::get_submatrix(scf_data, 'O', 'V', 'Y');
    let ri_oo = super::get_submatrix(scf_data, 'O', 'O', 'N');
    let mut ri_vv = super::get_submatrix(scf_data, 'V', 'V', 'N');
    ri_vv.reshape([num_auxbas * vir_size, vir_size]);
    let ri_ov = super::get_submatrix(scf_data, 'O', 'V', 'N');

    // ── Pre-compute contour nodes ──
    let centre = qp_ctrl.nlfeast_centre;
    let radius = qp_ctrl.nlfeast_radius;
    let n_quad = qp_ctrl.nlfeast_n_quad;
    let m0 = qp_ctrl.nlfeast_m0;

    println!("  Contour: centre={:.6}, radius={:.6}", centre, radius);
    println!("  n_quad={}, m0={}", n_quad, m0);

    super::matvec_trace::set_enabled(qp_ctrl.export_matvec_count);

    let contour_nodes = prepare_contour_nodes(
        &quasiparticle_energies, occ_size, vir_size,
        &ri_ov_response, &ri_oo, &ri_ov,
        centre, radius, n_quad,
    );
    let prep_time = start.elapsed();
    println!("  Contour preparation time: {:?}", prep_time);

    // ── Run NLFEAST ──
    let result = nlfeast_bse(
        scf_data, qp_ctrl,
        &quasiparticle_energies,
        occ_size, vir_size,
        &ri_ov_response, &ri_ov, &ri_oo, &ri_vv,
        &contour_nodes,
        centre, radius, m0, n_quad,
        qp_ctrl.nlfeast_max_iter,
        qp_ctrl.nlfeast_tol,
        qp_ctrl.nlfeast_gmres_restart,
        qp_ctrl.nlfeast_gmres_max_it,
        qp_ctrl.nlfeast_gmres_tol,
        qp_ctrl.bse_tda,
    );
    let total_time = start.elapsed();
    println!("  NLFEAST total time: {:?}", total_time);
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
        let lam_ev = result.eigenvalues[k] * 27.2114;
        println!("  {:>3}     {:>12.6} eV             {:>9.2e}",
            k, lam_ev, result.residuals[k]);
    }

    if scf_data.mol.ctrl.print_level > 1 {
        for k in 0..result.n_found {
            let xv: Vec<f64> = (0..n).map(|i| result.eigenvectors[[i, k]]).collect();
            println!("\n  Excitation #{}: λ = {:.6} Ha = {:.6} eV",
                k, result.eigenvalues[k], result.eigenvalues[k] * 27.2114);
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
            .open("nlfeast_excitations.txt")
            .expect("Failed to open nlfeast_excitations.txt");
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

// ============================================================================
// NLFEAST v3 — reference-correct algorithm from FEAST_Workspace
//
// Algorithm per iteration:
//   1. Projected solve: for each column j: A(λ_init[j])·Q[:,j]
//      → M_Q = Q^T·AQ → dgeev → λ_cur, Ritz vectors X
//   2. Select λ closest to contour centre, truncate to m0
//   3. Real-axis residual: tx_mat = T(λ_cur)·X
//   4. Convergence check
//   5. Contour integration:
//      GMRES: T(z_j)·U_j = tx_mat   (RHS = current residual)
//      Q_new += w_j · (X − U_j) · (z_j − λ_cur)⁻¹
//   6. QR(Q_new) → new Q
//   7. λ_init = λ_cur
// ============================================================================
/*
use super::dynamicbse_matvec::{
    dynamical_a_block_matvec_real,
    dynamical_a_block_matvec_complex,
    dynamical_composite_matvec_real,
    dynamical_composite_matvec_complex,
};

pub fn nlfeast_dynamical_bse_v3(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    quasiparticle_energies: &Vec<f64>,
    occ_size: usize,
    vir_size: usize,
    ri_ov_response: &MatrixFull<f64>,
    ri_ov: &MatrixFull<f64>,
    ri_oo: &MatrixFull<f64>,
    ri_vv_reshaped: &MatrixFull<f64>,
    vbar: &MatrixFull<f64>,
    rpa_omega: &[f64],
    contour_nodes: &[ContourNodeData],
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
) -> NLFeastResult {
    let n = occ_size * vir_size;
    let num_auxbas = ri_ov_response.size[0];
    let centre_cplx = Complex::new(centre, 0.0);

    let xlet = if qp_ctrl.bse_spin == "singlet" { 'S' }
               else if qp_ctrl.bse_spin == "triplet" { 'T' }
               else { 'R' };
    let addition = if xlet == 'S' { 2.0 } else if xlet == 'R' { 1.0 } else { 0.0 };

    let mut ri_ov_b = ri_ov.clone();
    ri_ov_b.reshape([num_auxbas * occ_size, vir_size]);
    let ri_ov_b_ref = &ri_ov_b;

    // ── Spin factor for Coulomb ──
    // A = D − W_dyn + (exchange_rescaling + addition)·V
    // With exchange_rescaling = 0, addition = spin_factor
    let er = 0.0_f64;
    let ad = addition;

    // ── QP energy gaps (preconditioner + initial guesses) ──
    let energy_diag: Vec<f64> = {
        let energies: Vec<f64> = if qp_ctrl.bse_qp_polarization {
            quasiparticle_energies.clone()
        } else { scf_data.eigenvalues[0].clone() };
        let mut d = Vec::with_capacity(n);
        for a in 0..vir_size { for i in 0..occ_size { d.push(energies[occ_size + a] - energies[i]); } }
        d
    };

    // ── Initial subspace Q: random orthonormal ──
    let mut rng = rand::thread_rng();
    let n_init = m0.min(n);
    let mut q = MatrixFull::new([n, n_init], 0.0);
    for j in 0..n_init { for i in 0..n { q[[i, j]] = rng.gen::<f64>() * 2.0 - 1.0; } }
    q = qr_orthonormalise(&q);
    let mut m0_eff = q.size[1];

    // ── Λ(0) = initial estimates from QP gaps ──
    let mut lambda: Vec<f64> = {
        let mut d: Vec<(f64, &f64)> = energy_diag.iter().map(|d| ((d - centre).abs(), d)).collect();
        d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        d.iter().take(m0_eff).map(|(_, &d)| d).collect()
    };
    while lambda.len() < m0_eff { lambda.push(centre); }
    let mut prev_lambda: Vec<f64> = Vec::new();
    let mut residuals = vec![0.0_f64; m0_eff];
    let mut inside_flags = vec![false; m0_eff];

    // ── Main NLFEAST loop ──
    for it in 0..max_iter {
        // ── Step 1: Projected solve ──
        // For each column j of Q, compute A(λ_init[j])·Q[:,j]
        // M_Q = Q^T · AQ → dgeev → eigenvalues + Ritz vectors
        let mq_size = q.size[1];
        let mut aq = MatrixFull::new([n, mq_size], 0.0);
        for j in 0..mq_size {
            let qj: Vec<f64> = (0..n).map(|i| q[[i, j]]).collect();
            let lam = lambda[j.min(lambda.len().saturating_sub(1))];
            let result = if tda {
                dynamical_a_block_matvec_real(
                    ri_oo, ri_vv_reshaped, ri_ov,
                    vbar, rpa_omega, quasiparticle_energies,
                    occ_size, vir_size, er, ad, lam, &qj)
            } else {
                let a_x = dynamical_a_block_matvec_real(
                    ri_oo, ri_vv_reshaped, ri_ov,
                    vbar, rpa_omega, quasiparticle_energies,
                    occ_size, vir_size, er, ad, lam, &qj);
                let b_x = super::matvec::b_block_matvec(
                    scf_data, qp_ctrl, ri_ov, &ri_ov_b,
                    &MatrixFull::new([num_auxbas * occ_size, vir_size], 0.0),
                    &qj.to_vec());
                let t: Vec<f64> = a_x.iter().zip(b_x.iter()).map(|(a,b)| a-b).collect();
                let a_t = dynamical_a_block_matvec_real(
                    ri_oo, ri_vv_reshaped, ri_ov,
                    vbar, rpa_omega, quasiparticle_energies,
                    occ_size, vir_size, er, ad, lam, &t);
                let b_t = super::matvec::b_block_matvec(
                    scf_data, qp_ctrl, ri_ov, &ri_ov_b,
                    &MatrixFull::new([num_auxbas * occ_size, vir_size], 0.0),
                    &t.to_vec());
                a_t.iter().zip(b_t.iter()).map(|(a,b)| a+b).collect()
            };
            for i in 0..n { aq[[i, j]] = result[i]; }
        }

        // Solve M_Q · y = µ · y  (TDA: µ=λ; non-TDA: λ=√µ)
        let mq = _dgemm_scaled(&q, 'T', &aq, 'N', 1.0);
        let (_vr_dummy, wr, wi, _vl, vr, info) = _dgeev(&mq, 'N', 'V');
        if info != 0 { eprintln!("  Warning: dgeev in projected solve info={}", info); }

        let mut all_lambda = Vec::with_capacity(mq_size);
        let mut all_x = MatrixFull::new([n, mq_size], 0.0);
        for j in 0..mq_size {
            let lam = if tda { wr[j] } else {
                let mu_re = wr[j]; let mu_im = wi[j];
                (mu_re * mu_re + mu_im * mu_im).powf(0.25)  // √|µ|
            };
            all_lambda.push(Complex::new(lam, 0.0));
            for i in 0..n {
                let mut val = 0.0;
                for k in 0..mq_size { val += q[[i, k]] * vr[[k, j]]; }
                all_x[[i, j]] = val;
            }
        }

        // ── Step 2: Select m0_eff eigenvalues closest to centre ──
        let mut with_dist: Vec<(f64, usize)> = (0..mq_size)
            .map(|idx| ((all_lambda[idx] - centre_cplx).norm(), idx)).collect();
        with_dist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let n_sel = with_dist.len().min(m0);
        let mut lambda_cur = Vec::with_capacity(n_sel);
        let mut x_mat = MatrixFull::new([n, n_sel], 0.0);
        for (pj, (_, idx)) in with_dist[..n_sel].iter().enumerate() {
            lambda_cur.push(all_lambda[*idx].re);
            for i in 0..n { x_mat[[i, pj]] = all_x[[i, *idx]]; }
        }
        let m0_eff = n_sel;

        if it == 0 {
            eprint!("  v3 initial λ:");
            for j in 0..m0_eff.min(8) { eprint!(" {:.4}", lambda_cur[j]); }
            eprintln!();
        }

        // ── Step 3: Compute residuals T(λ)·x on the real axis ──
        let mut max_res = 0.0_f64;
        let mut n_inside = 0;
        let mut n_inside_converged = 0;
        let mut residuals = vec![0.0_f64; m0_eff];
        let mut inside_flags = vec![false; m0_eff];
        let mut tx_mat = MatrixFull::new([n, m0_eff], 0.0);

        for j in 0..m0_eff {
            let lam = lambda_cur[j];
            let xj: Vec<f64> = (0..n).map(|i| x_mat[[i, j]]).collect();
            let txj = if tda {
                let a_x = dynamical_a_block_matvec_real(
                    ri_oo, ri_vv_reshaped, ri_ov,
                    vbar, rpa_omega, quasiparticle_energies,
                    occ_size, vir_size, er, ad, lam, &xj);
                a_x.iter().zip(xj.iter()).map(|(a, &xi)| a - lam * xi).collect()
            } else {
                dynamical_composite_matvec_real(
                    scf_data, qp_ctrl,
                    ri_oo, ri_vv_reshaped, ri_ov, &ri_ov_b,
                    &MatrixFull::new([num_auxbas * occ_size, vir_size], 0.0),
                    vbar, rpa_omega, quasiparticle_energies,
                    occ_size, vir_size, er, ad, lam, &xj)
            };
            let res: f64 = txj.iter().map(|v| v*v).sum::<f64>().sqrt();
            residuals[j] = res;
            if res > max_res { max_res = res; }
            let inside = (Complex::new(lam, 0.0) - centre_cplx).norm() <= radius;
            inside_flags[j] = inside;
            if inside { n_inside += 1; if res <= tol { n_inside_converged += 1; } }
            for i in 0..n { tx_mat[[i, j]] = txj[i]; }
        }

        eprintln!("  v3[{}]: max_res={:.2e} inside {}/{}  λ≈{:.4}..{:.4}",
            it+1, max_res, n_inside_converged, n_inside,
            lambda_cur.iter().cloned().fold(1e30_f64, f64::min),
            lambda_cur.iter().cloned().fold(-1e30_f64, f64::max));

        // ── Step 4: Convergence check ──
        if n_inside > 0 && n_inside_converged == n_inside {
            eprintln!("  → All interior eigenvalues converged.");
            let mut result = build_final_result(n, &lambda_cur, &x_mat, &residuals, &inside_flags, centre, radius);
            result.iterations = it + 1;
            return result;
        }

        // Stagnation
        let improved = if prev_lambda.len() == lambda_cur.len() {
            let mut d = 0.0;
            for j in 0..lambda_cur.len() { d += (lambda_cur[j] - prev_lambda[j]).abs(); }
            d / lambda_cur.len() as f64
        } else { 1.0 };
        prev_lambda = lambda_cur.clone();
        if it > 3 && improved < 1e-12 && n_inside > 0 {
            eprintln!("  → Eigenvalues stabilised (Δλ≈{:.2e}).", improved);
            let mut result = build_final_result(n, &lambda_cur, &x_mat, &residuals, &inside_flags, centre, radius);
            result.iterations = it + 1;
            return result;
        }

        // ── Step 5: Contour integration ──
        let q_new_contribs: Vec<MatrixFull<f64>> = contour_nodes
            .par_iter()
            .map(|node| {
                let z = Complex::new(node.z_re, node.z_im);
                let c_matvec: Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync> = if tda {
                    use super::dynamicbse_matvec::dynamical_a_block_matvec_complex;
                    let zr = node.z_re; let zi = node.z_im;
                    let rr = ri_oo; let rv = ri_vv_reshaped; let ro = ri_ov;
                    let vb = vbar; let rp = rpa_omega; let qp = quasiparticle_energies;
                    let (oc, vs, er2, ad2) = (occ_size, vir_size, er, ad);
                    Box::new(move |v: &[f64]| -> Vec<f64> {
                        let n2 = oc * vs;
                        let vr = &v[0..n2]; let vi = &v[n2..2*n2];
                        let (a_re, a_im) = dynamical_a_block_matvec_complex(
                            rr, rv, ro, vb, rp, qp, oc, vs, er2, ad2,
                            Complex::new(zr, zi), vr, vi);
                        let mut out = vec![0.0; 2*n2];
                        for i in 0..n2 { out[i] = a_re[i] - zr*vr[i] + zi*vi[i]; out[n2+i] = a_im[i] - zr*vi[i] - zi*vr[i]; }
                        out
                    })
                } else {
                    Box::new(move |v: &[f64]| -> Vec<f64> {
                        let n2 = occ_size * vir_size;
                        let vr = &v[0..n2]; let vi = &v[n2..2*n2];
                        let (t_re, t_im) = dynamical_composite_matvec_complex(
                            scf_data, qp_ctrl,
                            ri_oo, ri_vv_reshaped, ri_ov, ri_ov_b_ref,
                            &node.ri_ov_tilde_re, &node.ri_ov_tilde_im,
                            vbar, rpa_omega, quasiparticle_energies,
                            occ_size, vir_size, er, ad, vr, vi, z);
                        let mut out = vec![0.0; 2*n2];
                        for i in 0..n2 { out[i] = t_re[i]; out[n2+i] = t_im[i]; }
                        out
                    })
                };

                // Preconditioner
                let mut re_part = Vec::with_capacity(n);
                let mut im_part = Vec::with_capacity(n);
                for d in &energy_diag {
                    if tda { re_part.push(d - z.re); im_part.push(-z.im); }
                    else { let d2 = d*d; re_part.push(d2 - z.re*z.re + z.im*z.im); im_part.push(-2.0*z.re*z.im); }
                }
                let precond = BlockDiagPrecond::from_re_im(&re_part, &im_part);

                // Solve T(z_j)·U_j = T(Λ)·X for each column
                let mut u_re = vec![0.0_f64; n * m0_eff];
                let mut u_im = vec![0.0_f64; n * m0_eff];
                for col in 0..m0_eff {
                    let mut b_2n = vec![0.0; 2 * n];
                    for i in 0..n { b_2n[i] = tx_mat[[i, col]]; }
                    let x_2n = gmres(&c_matvec, &b_2n, gmres_restart, gmres_max_it,
                                     gmres_tol, col == 0, Some(&precond));
                    for i in 0..n { u_re[col*n+i] = x_2n[i]; u_im[col*n+i] = x_2n[n+i]; }
                }

                // Accumulate: Q_new += w_j · (X − U_j) · (z_j I − Λ)⁻¹
                let mut q_node = MatrixFull::new([n, m0_eff], 0.0);
                for col in 0..m0_eff {
                    let lam = lambda_cur[col];
                    let dz_re = z.re - lam;
                    let denom = dz_re*dz_re + z.im*z.im;
                    if denom < 1e-30 { continue; }
                    let inv_re = dz_re / denom;
                    let inv_im = -z.im / denom;
                    let f_re = node.w_re * inv_re - node.w_im * inv_im;
                    let f_im = node.w_re * inv_im + node.w_im * inv_re;
                    for i in 0..n {
                        let dx_re = x_mat[[i, col]] - u_re[col*n+i];
                        let dx_im = -u_im[col*n+i];
                        q_node[[i, col]] += f_re * dx_re - f_im * dx_im;
                    }
                }
                q_node
            })
            .collect();

        let mut q_new = MatrixFull::new([n, m0_eff], 0.0);
        for contrib in &q_new_contribs {
            for j in 0..m0_eff { for i in 0..n { q_new[[i, j]] += contrib[[i, j]]; } }
        }

        // ── Step 6: QR → new subspace Q ──
        q = qr_orthonormalise(&q_new);
        if q.size[1] < m0_eff / 2 {
            eprintln!("  Warning: subspace collapsed, stopping.");
            break;
        }

        // ── Step 7: λ_init = λ_cur for next iteration ──
        lambda = lambda_cur;
        lambda.truncate(q.size[1]);  // match subspace after QR
    }

    eprintln!("  v3: max iterations reached, returning best results.");
    let mut result = build_final_result(n, &lambda, &q, &residuals, &inside_flags, centre, radius);
    result.iterations = max_iter;
    result
}

pub fn nlfeast_dynamical_bse_main(scf_data: &SCF, qp_ctrl: &QuasiParticle) {
    use super::dynamicbse_matvec::precompute_rpa_data;

    let start = Instant::now();
    let (_start_mo, _num_state, occ_size, vir_size, _homo, _lumo) =
        get_occupation_parameters(scf_data, 'N');
    let n = occ_size * vir_size;
    let quasiparticle_energies = scf_data.gwqp.0.clone();
    let num_auxbas = super::get_submatrix(scf_data, 'O', 'V', 'N').size[0];

    println!("==============================================");
    println!("  NLFEAST v3 — reference-correct algorithm");
    println!("==============================================");
    println!("  occ={} vir={} n={} naux={}", occ_size, vir_size, n, num_auxbas);
    if qp_ctrl.bse_tda { println!("  Method: TDA"); }
    else { println!("  Method: non-TDA (static B-block)"); }

    let ri_ov_response = super::get_submatrix(scf_data, 'O', 'V', 'Y');
    let ri_oo = super::get_submatrix(scf_data, 'O', 'O', 'N');
    let mut ri_vv = super::get_submatrix(scf_data, 'V', 'V', 'N');
    ri_vv.reshape([num_auxbas * vir_size, vir_size]);
    let ri_ov = super::get_submatrix(scf_data, 'O', 'V', 'N');

    eprintln!("  Precomputing RPA data...");
    let rpa_data = precompute_rpa_data(
        &quasiparticle_energies, occ_size, vir_size, &ri_ov_response,
    );
    eprintln!("    n_rpa = {}", rpa_data.rpa_omega.len());

    let centre = qp_ctrl.nlfeast_centre;
    let radius = qp_ctrl.nlfeast_radius;
    let n_quad = qp_ctrl.nlfeast_n_quad;
    let m0 = qp_ctrl.nlfeast_m0;

    let contour_nodes = prepare_contour_nodes(
        &quasiparticle_energies, occ_size, vir_size,
        &ri_ov_response, &ri_oo, &ri_ov,
        centre, radius, n_quad,
    );

    let result = nlfeast_dynamical_bse_v3(
        scf_data, qp_ctrl, &quasiparticle_energies,
        occ_size, vir_size,
        &ri_ov_response, &ri_ov, &ri_oo, &ri_vv,
        &rpa_data.vbar, &rpa_data.rpa_omega,
        &contour_nodes,
        centre, radius, m0, n_quad,
        qp_ctrl.nlfeast_max_iter,
        qp_ctrl.nlfeast_tol,
        qp_ctrl.nlfeast_gmres_restart,
        qp_ctrl.nlfeast_gmres_max_it,
        qp_ctrl.nlfeast_gmres_tol,
        qp_ctrl.bse_tda,
    );

    let total_time = start.elapsed();
    println!("  NLFEASTv3 total time: {:?}", total_time);
    println!("  Outer iterations: {}", result.iterations);
    if result.n_found == 0 {
        println!("  No eigenvalues found inside the contour.");
        return;
    }
    println!("\n  Found {} excitation(s) inside [{:.4}, {:.4}]:\n",
        result.n_found, centre - radius, centre + radius);
    println!("  #       Excitation energy (eV)    ‖T(λ)x‖");
    println!("  ───     ─────────────────────    ──────────");
    for k in 0..result.n_found {
        let lam_ev = result.eigenvalues[k] * 27.2114;
        println!("  {:>3}     {:>12.6} eV             {:>9.2e}", k, lam_ev, result.residuals[k]);
    }
}
*/