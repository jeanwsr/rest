// ============================================================================
// FEAST Solver for Generalized Eigenvalue Problem A x = λ B x
//
// Context-independent implementation of the basic linear FEAST algorithm
// (Polizzi 2009) to find all eigenvalues of (A, B) within a specified interval
// [λ_min, λ_max].
//
// The solver only accesses A and B through matrix-vector product closures
// `a_mul(x)` and `b_mul(x)`, satisfying:
//   a_mul(x) = A * x   where A is real symmetric
//   b_mul(x) = B * x   where B is symmetric positive definite
//
// Linear systems (z*B - A) * X = B*Y (with complex shift z = α + iβ)
// are solved by embedding as a 2N-dimensional real system and using
// GMRES.  The 2N × 2N matrix is assembled implicitly through the a_mul /
// b_mul closures.
//
// The module also exposes the standalone preconditioned CG solver used by
// callers that need to apply an implicit inverse metric (e.g. (A+B)^{-1} in
// the non-TDA FEAST transformation).
// ============================================================================
use crate::tensors::MathMatrix;
use rest_tensors::matrix::matrix_blas_lapack::{
    _dgemm_scaled, _dpotrf, _dsyevd, _dinverse, _dgeev,
};
use rest_tensors::{BasicMatrix, MatrixFull};
use rand::Rng;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Gauss-Legendre quadrature node/weight generation for arbitrary n
// ---------------------------------------------------------------------------

/// Compute n-point Gauss-Legendre quadrature nodes and weights on [-1, 1].
///
/// Uses the Newton-Raphson method on Legendre polynomials to find roots.
/// Returns Vec<(node, weight)>.
fn gauss_legendre_nodes(n: usize) -> Vec<(f64, f64)> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![(0.0, 2.0)];
    }

    let eps = 1e-15;
    let np1 = n as f64;
    let mut points = Vec::with_capacity(n);

    for i in 0..n {
        // Initial guess: symmetric cosine approximation
        let z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();

        // Newton-Raphson to find the root
        let mut z_curr = z;
        loop {
            // Compute P_n(z_curr) via recurrence
            let mut p_prev = 1.0; // P_0
            let mut p_curr = z_curr; // P_1
            for k in 1..n {
                let kf = k as f64;
                let p_next = ((2.0 * kf + 1.0) * z_curr * p_curr - kf * p_prev) / (kf + 1.0);
                p_prev = p_curr;
                p_curr = p_next;
            }
            // P'_n = n * (x * P_n - P_{n-1}) / (x² - 1)
            let pp = np1 * (z_curr * p_curr - p_prev) / (z_curr * z_curr - 1.0);
            let delta = p_curr / pp;
            z_curr -= delta;
            if delta.abs() <= eps {
                break;
            }
        }

        // Final node
        let node = z_curr;

        // Recompute P_n and P_{n-1} for the converged node
        let mut p_prev = 1.0;
        let mut p_curr = node;
        for k in 1..n {
            let kf = k as f64;
            let p_next = ((2.0 * kf + 1.0) * node * p_curr - kf * p_prev) / (kf + 1.0);
            p_prev = p_curr;
            p_curr = p_next;
        }
        let pp = np1 * (node * p_curr - p_prev) / (node * node - 1.0);
        // weight = 2 / ((1 - x²) · [P'_n(x)]²)
        let weight = 2.0 / ((1.0 - node * node) * pp * pp);
        points.push((node, weight));
    }
    points
}

// ============================================================================
// Section 1: CG Solver — standard Conjugate Gradient
// ============================================================================

/// Solve A * x = b using preconditioned CG, where A is an SPD matrix
/// accessed through the closure `a_mul`.  Returns the solution vector x.
///
/// If `precond_diag` is provided, a diagonal (Jacobi) preconditioner
/// `M = diag(precond_diag)` is applied to the residual.  The diagonal
/// entries must be positive for the preconditioner to be SPD; non-positive
/// entries fall back to 1.0 (no preconditioning for that component).
pub fn cg(
    a_mul: impl Fn(&Vec<f64>) -> Vec<f64>,
    b: &Vec<f64>,
    max_iter: usize,
    tol: f64,
    precond_diag: Option<&Vec<f64>>,
) -> Vec<f64> {
    let n = b.len();

    // Precompute the inverse diagonal once per solve.
    let inv_diag: Option<Vec<f64>> = precond_diag.map(|diag| {
        diag.iter()
            .map(|&d| if d > 1e-30 { 1.0 / d } else { 1.0 })
            .collect()
    });

    // Apply M^{-1} to a residual vector.
    let precond = |r: &Vec<f64>| -> Vec<f64> {
        match &inv_diag {
            Some(inv) => r.iter().zip(inv).map(|(&ri, &idi)| ri * idi).collect(),
            None => r.clone(),
        }
    };

    let mut x = vec![0.0; n];
    let ax = a_mul(&x);
    let mut r: Vec<f64> = b.iter().zip(&ax).map(|(bi, axi)| bi - axi).collect();
    let z = precond(&r);
    let mut p = z;
    let mut r_dot_z: f64 = r.iter().zip(&p).map(|(ri, zi)| ri * zi).sum();

    for _iter in 0..max_iter {
        let ap = a_mul(&p);
        let p_dot_ap: f64 = p.iter().zip(&ap).map(|(pi, api)| pi * api).sum();
        if p_dot_ap.abs() < 1e-30 || r_dot_z.abs() < 1e-30 {
            break;
        }
        let alpha = r_dot_z / p_dot_ap;
        for i in 0..n {
            x[i] += alpha * p[i];
        }
        let r_new: Vec<f64> = r.iter().zip(&ap).map(|(ri, api)| ri - alpha * api).collect();
        let residual: f64 = r_new.iter().map(|ri| ri * ri).sum::<f64>().sqrt();
        if residual < tol {
            break;
        }
        let z_new = precond(&r_new);
        let r_new_dot_z_new: f64 = r_new.iter().zip(&z_new).map(|(ri, zi)| ri * zi).sum();
        let beta = r_new_dot_z_new / r_dot_z;
        for i in 0..n {
            p[i] = z_new[i] + beta * p[i];
        }
        r = r_new;
        r_dot_z = r_new_dot_z_new;
    }
    x
}

// ============================================================================
// Section 2: GMRES-based iterative solver for complex-shifted system
//            (z·B - A) · X = B·Y
//
// The system matrix M = (α + iβ)·B - A is complex.  We embed it as a
// 2N×2N real system using the real embedding:
//
//   M₂ · [Xr; Xi] = [Br; 0]
//
//   where  M₂ = [ α·B−A   −β·B ]
//               [  β·B    α·B−A ]
//
// M₂ is NOT symmetric (due to the off-diagonal sign asymmetry), so we
// use GMRES (Generalized Minimal RESidual) to solve the 2N×2N system
// column by column, requiring only implicit matvec via a_mul / b_mul.
//
// The M₂ matvec for a 2N-vector [vr; vi] is:
//   top:    α·B(vr) − A(vr) − β·B(vi)
//   bottom: β·B(vr) + α·B(vi) − A(vi)
//
// Per matvec: 2 A-calls + 2 B-calls = 4 closure calls.
// ============================================================================

/// Compute Givens rotation (c, s) to zero out the second component:
/// [c  s] [a] = [r]
/// [-s c] [b]   [0]
fn givens_rotation(a: f64, b: f64) -> (f64, f64) {
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

/// Solve A·x = b using restarted GMRES(m) where A is accessed via `a_mul`.
///
/// GMRES works for non-symmetric matrices by building an orthonormal basis
/// of the Krylov subspace via the Arnoldi process with modified Gram-Schmidt.
///
/// Parameters:
///   a_mul       – matvec closure (A is n×n, non-symmetric)
///   b           – RHS vector (length n)
///   restart     – restart dimension (Krylov subspace size)
///   max_iter    – maximum total number of matvec calls
///   tol         – relative residual tolerance ‖r‖/‖b‖ < tol
///   verbose     – if true, print convergence info at each restart
///   precondition – optional left-diagonal preconditioner (length n)
///                  Solves M^{-1} A x = M^{-1} b with M = diag(precondition)
fn gmres(
    a_mul: impl Fn(&Vec<f64>) -> Vec<f64>,
    b: &Vec<f64>,
    restart: usize,
    max_iter: usize,
    tol: f64,
    verbose: bool,
    precondition: Option<&GmresPrecond>,
) -> (Vec<f64>, usize) {
    let n = b.len();

    // Use unpreconditioned ‖b‖₂ for tolerance — independent of preconditioner choice
    let b_norm: f64 = b.iter().map(|bi| bi * bi).sum::<f64>().sqrt();
    let tol_abs = tol * b_norm.max(1e-30);

    // Helper: apply block-diagonal preconditioner to a vector
    let apply_prec = |w: &Vec<f64>, prec: &GmresPrecond| -> Vec<f64> {
        prec.apply_inverse(w)
    };

    let mut x = vec![0.0; n];
    let mut r_norm: f64 = b_norm;
    let mut total_matvecs = 0_usize;
    let mut n_restarts = 0_usize;

    // ── Outer restart loop ──
    loop {
        if r_norm <= tol_abs { break; }
        if total_matvecs >= max_iter { break; }

        // Compute r = precondition .* (b - A(x))
        let ax = a_mul(&x);
        total_matvecs += 1;
        let r_raw: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
        let r: Vec<f64> = if let Some(prec) = precondition {
            apply_prec(&r_raw, prec)
        } else {
            r_raw
        };
        let beta = r.iter().map(|ri| ri * ri).sum::<f64>().sqrt();

        if beta <= tol_abs {
            r_norm = beta;
            break;
        }

        // First Krylov vector: v0 = r / beta
        let mut v: Vec<Vec<f64>> = Vec::with_capacity(restart + 1);
        v.push(r.iter().map(|ri| ri / beta).collect());

        // Hessenberg matrix: (restart+1) × restart, flattened row-major
        let mut h = vec![0.0f64; (restart + 1) * restart];
        // RHS for least-squares: c[0] = beta
        let mut c = vec![0.0f64; restart + 1];
        c[0] = beta;
        let mut givens_c = vec![0.0f64; restart];
        let mut givens_s = vec![0.0f64; restart];

        let mut inner_k = 0;

        for j in 0..restart {
            if total_matvecs >= max_iter { break; }

            // Arnoldi: w = precondition .* (A · vj)
            let w_raw = a_mul(&v[j]);
            total_matvecs += 1;
            let w = if let Some(prec) = precondition {
                apply_prec(&w_raw, prec)
            } else {
                w_raw
            };

            // Modified Gram-Schmidt
            let mut w_ortho = w;
            for i in 0..=j {
                let dot: f64 = w_ortho.iter().zip(&v[i]).map(|(a, b)| a * b).sum();
                h[i * restart + j] = dot;
                for ii in 0..n { w_ortho[ii] -= dot * v[i][ii]; }
            }

            let h_norm: f64 = w_ortho.iter().map(|wi| wi * wi).sum::<f64>().sqrt();

            if h_norm < 1e-15 {
                // Happy breakdown
                h[(j + 1) * restart + j] = h_norm;
                inner_k = j + 1;
                break;
            }

            h[(j + 1) * restart + j] = h_norm;
            v.push(w_ortho.iter().map(|wi| wi / h_norm).collect());

            // Apply previous Givens rotations to column j
            for i in 0..j {
                let ci = givens_c[i];
                let si = givens_s[i];
                let hij = h[i * restart + j];
                let hip1_j = h[(i + 1) * restart + j];
                h[i * restart + j]       =  ci * hij + si * hip1_j;
                h[(i + 1) * restart + j] = -si * hij + ci * hip1_j;
            }

            // Compute Givens rotation to zero h[j+1][j]
            let (cj, sj) = givens_rotation(h[j * restart + j], h[(j + 1) * restart + j]);
            givens_c[j] = cj;
            givens_s[j] = sj;

            // Apply to H
            let hjj = h[j * restart + j];
            let hjp1_j = h[(j + 1) * restart + j];
            h[j * restart + j]       =  cj * hjj + sj * hjp1_j;
            h[(j + 1) * restart + j] =  0.0;

            // Apply to c
            let cj_old = c[j];
            c[j]     =  cj * cj_old;
            c[j + 1] = -sj * cj_old;

            inner_k = j + 1;

            // Exit Arnoldi early when the preconditioned residual estimate
            // is small — saves matvecs.  True convergence is always verified
            // against the unpreconditioned residual at the end of the restart.
            if c[j + 1].abs() <= tol_abs {
                break;
            }
        }

        // ── Solve H_k · y = c_k (upper triangular) ──
        if inner_k > 0 {
            let mut y = vec![0.0f64; inner_k];
            for i in (0..inner_k).rev() {
                let mut sum = c[i];
                for jj in (i + 1)..inner_k {
                    sum -= h[i * restart + jj] * y[jj];
                }
                y[i] = sum / h[i * restart + i];
            }

            // Update x = x + V_k · y
            for i in 0..n {
                let mut dx = 0.0;
                for jj in 0..inner_k {
                    dx += v[jj][i] * y[jj];
                }
                x[i] += dx;
            }
        }

        // Compute true (unpreconditioned) residual after each restart
        let ax = a_mul(&x);
        total_matvecs += 1;
        let r_unprec: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
        r_norm = r_unprec.iter().map(|ri| ri * ri).sum::<f64>().sqrt();
        n_restarts += 1;
        if r_norm <= tol_abs {
            if verbose {
                eprintln!("  GMRES converged: {} matvecs, {} restart(s), residual {:.2e}",
                    total_matvecs, n_restarts, r_norm / b_norm);
            }
            break;
        }
        if verbose && n_restarts % 10 == 0 {
            eprintln!("  GMRES restart {}: {} matvecs, rel.res = {:.2e}",
                n_restarts, total_matvecs, r_norm / b_norm);
        }
    }

    if r_norm > tol_abs {
        panic!("GMRES Did Not Converge for FEAST Grid, Consider Enlarging m_expected");
    }

    (x, total_matvecs)
}

/// 2×2 block-diagonal preconditioner for the 2N real-embedded system.
///
/// For each original complex index i (mapping to real components at indices
/// i and N+i in the 2N vector), the block is stored as [a, b, c, d] in
/// row-major order representing:
///
///   block_i = [a  b]
///             [c  d]
///
/// Application: v_out = block_i⁻¹ · v_in  (solved for each i).
pub struct BlockDiagPrecond {
    pub blocks: Vec<[f64; 4]>,
}

impl BlockDiagPrecond {
    /// Create from two arrays: re_part[i] = Re(λ_i), im_part[i] = Im(λ_i).
    /// The correct 2×2 block for complex value λ = re + i·im in the
    /// real embedding is [[re, -im], [im, re]].
    pub fn from_re_im(re_part: &[f64], im_part: &[f64]) -> Self {
        let n = re_part.len();
        let mut blocks = Vec::with_capacity(n);
        for i in 0..n {
            blocks.push([re_part[i], -im_part[i], im_part[i], re_part[i]]);
        }
        BlockDiagPrecond { blocks }
    }

    /// Apply the inverse of this block-diagonal matrix to a vector v
    /// of length 2*n (first n = real parts, next n = imag parts).
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

    /// Compute the 2-norm of the preconditioned vector (for residual checks).
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
// Preconditioner for GMRES: inner GMRES with s-only basis as implicit inverse
// ============================================================================

/// Inner GMRES preconditioner: uses a cheaper (s-only) RI basis matvec
/// to implicitly approximate M₂⁻¹ via a few steps of inner GMRES.
pub struct InnerGmresPrecond {
    /// s-only A·x closure (wrapped for type erasure, Send+Sync for Rayon)
    pub precond_a: Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync>,
    /// s-only B·x closure
    pub precond_b: Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync>,
    /// Real part of complex shift z for the current quadrature point
    pub z_re: f64,
    /// Imaginary part of complex shift z
    pub z_im: f64,
    /// Inner GMRES restart dimension
    pub inner_restart: usize,
    /// Inner GMRES max iterations
    pub inner_max_iter: usize,
    /// Inner GMRES tolerance
    pub inner_tol: f64,
    /// Diagonal matrix D (length n) used for inner GMRES diagonal precond
    pub diag: Vec<f64>,
}

impl InnerGmresPrecond {
    /// Apply the inner GMRES preconditioner: solve M₂(s-only)·y = v approximately.
    pub fn apply_inverse(&self, v: &[f64]) -> Vec<f64> {
        let n2 = v.len(); // 2*n
        let n = n2 / 2;
        let qp_z_re = self.z_re;
        let qp_z_im = self.z_im;

        // Build s-only M₂ matvec
        let m2_matvec = |w: &Vec<f64>| -> Vec<f64> {
            let wr: Vec<f64> = w[0..n].to_vec();
            let wi: Vec<f64> = w[n..n2].to_vec();
            let b_wr = (self.precond_b)(&wr);
            let b_wi = (self.precond_b)(&wi);
            let a_wr = (self.precond_a)(&wr);
            let a_wi = (self.precond_a)(&wi);
            let mut result = vec![0.0; n2];
            for i in 0..n {
                result[i]    = qp_z_re * b_wr[i] - a_wr[i] - qp_z_im * b_wi[i];
                result[n + i] = qp_z_im * b_wr[i] + qp_z_re * b_wi[i] - a_wi[i];
            }
            result
        };

        // Build block-diag precond for the inner GMRES
        let mut re_part = vec![0.0; n];
        let mut im_part = vec![0.0; n];
        for i in 0..n {
            re_part[i] = qp_z_re - self.diag[i];
            im_part[i] = -qp_z_im;
        }
        let inner_prec = GmresPrecond::Diag(BlockDiagPrecond::from_re_im(&re_part, &im_part));

        // Inner GMRES solve
        let (y, inner_matvecs) = gmres(
            &m2_matvec,
            &v.to_vec(),
            self.inner_restart,
            self.inner_max_iter,
            self.inner_tol,
            false, // no verbose for inner GMRES
            Some(&inner_prec),
        );

        y
    }

    /// Compute 2-norm of preconditioned vector (for outer GMRES residual check).
    pub fn norm(&self, v: &[f64]) -> f64 {
        // Use the same inner GMRES: ‖P⁻¹·v‖ ≈ ‖solve(M₂_s, v)‖
        let pv = self.apply_inverse(v);
        pv.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

/// Enum encapsulating preconditioner variants for GMRES.
pub enum GmresPrecond {
    Diag(BlockDiagPrecond),
    InnerGmres(InnerGmresPrecond),
}

impl GmresPrecond {
    pub fn apply_inverse(&self, v: &[f64]) -> Vec<f64> {
        match self {
            GmresPrecond::Diag(p) => p.apply_inverse(v),
            GmresPrecond::InnerGmres(p) => p.apply_inverse(v),
        }
    }

    pub fn norm(&self, v: &[f64]) -> f64 {
        match self {
            GmresPrecond::Diag(p) => p.norm(v),
            GmresPrecond::InnerGmres(p) => p.norm(v),
        }
    }
}

// ============================================================================
// Section 3: FEAST main algorithm
// ============================================================================

/// Solve the generalized eigenvalue problem A x = λ B x.
///
/// The solver uses only closures for matrix-vector products; no explicit
/// matrix storage is required.
///
/// Parameters:
///   n          – dimension of the matrices
///   a_mul, b_mul – closures implementing A·x and B·x
///   λ_min, λ_max – search interval
///   m_expected – expected number of eigenvalues inside [λ_min, λ_max]
///   max_feast_iter – maximum outer FEAST iterations
///   tol_feast  – trace convergence tolerance
pub fn feast<F, G>(
    n: usize,
    a_mul: &F,
    b_mul: &G,
    gmres_a_mul: Option<&(dyn Fn(&Vec<f64>) -> Vec<f64> + Sync)>,
    gmres_b_mul: Option<&(dyn Fn(&Vec<f64>) -> Vec<f64> + Sync)>,
    λ_min: f64,
    λ_max: f64,
    m_expected: usize,
    max_feast_iter: usize,
    tol_feast: f64,
    gmres_restart: usize,
    gmres_max_iter: usize,
    gmres_tol: f64,
    diag_a: Option<&Vec<f64>>,
    init_guess_type: &str,
    init_diag: Option<&Vec<f64>>,
    gaussian_width_factor: f64,
    use_contour_rayon: bool,
    custom_init_vectors: Option<&Vec<Vec<f64>>>,
    precond_a_mul: Option<Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync>>,
    precond_b_mul: Option<Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync>>,
    precond_type: &str,
    precond_diag: Option<&Vec<f64>>,
    inner_gmres_tol: f64,
    inner_gmres_restart: usize,
    inner_gmres_max_iter: usize,
) -> Vec<(f64, Vec<f64>)>
where
    F: Fn(&Vec<f64>) -> Vec<f64> + Sync,
    G: Fn(&Vec<f64>) -> Vec<f64> + Sync,
{
    // ---- Step 0: parameters ------------------------------------------------
    let c = (λ_max + λ_min) / 2.0; // centre of the contour
    let r = (λ_max - λ_min) / 2.0; // radius
    let mut m0 = m_expected; // subspace size (shrinks after compression)
    let mut subspace_ever_compressed = false;

    // Resolve GMRES matvec closures: use specified or fall back to a_mul/b_mul
    let default_a: &(dyn Fn(&Vec<f64>) -> Vec<f64> + Sync) = a_mul;
    let default_b: &(dyn Fn(&Vec<f64>) -> Vec<f64> + Sync) = b_mul;
    let gmres_a: &(dyn Fn(&Vec<f64>) -> Vec<f64> + Sync) = gmres_a_mul.unwrap_or(default_a);
    let gmres_b: &(dyn Fn(&Vec<f64>) -> Vec<f64> + Sync) = gmres_b_mul.unwrap_or(default_b);

    let scale = λ_min.abs().max(λ_max.abs()).max(1.0);

    // Pre-compute the quadrature contributions per node
    struct QuadData {
        _θ: f64,
        z_re: f64, // Re(z) = centre + r·cos(θ)
        z_im: f64, // Im(z) = r·sin(θ)
        w: f64, // weight contribution factor
        cosθ: f64,
        sinθ: f64,
    }
    let gl_points = gauss_legendre_nodes(8);
    let quad: Vec<QuadData> = gl_points
        .iter()
        .map(|&(x_e, w_e)| {
            let θ = -std::f64::consts::PI * (x_e - 1.0) / 2.0; // θ ∈ [0, π]
            let cosθ = θ.cos();
            let sinθ = θ.sin();
            QuadData {
                _θ: θ,
                z_re: c + r * cosθ,
                z_im: r * sinθ,
                // Contribution factor Q += -0.5 * w_e * Re(r·exp(iθ) · Q_temp)
                //                    = -0.5 * w_e * (r·cosθ · Re(Q_temp) - r·sinθ · Im(Q_temp))
                w: -0.5 * w_e,
                cosθ,
                sinθ,
            }
        })
        .collect();

    // ---- Step 1: initial subspace ---------------------------------------------
    let mut rng = rand::rng();
    let mut y = MatrixFull::new([n, m0], 0.0);

    if let Some(custom_vecs) = custom_init_vectors {
        // Use provided eigenvectors + random padding as initial guess
        let n_provided = custom_vecs.len();
        let n_use = std::cmp::min(n_provided, m0);
        for j in 0..n_use {
            for i in 0..n {
                y[[i, j]] = custom_vecs[j][i];
            }
        }
        // Pad remaining columns with random vectors
        for j in n_use..m0 {
            for i in 0..n {
                y[[i, j]] = rng.random::<f64>() * 2.0 - 1.0;
            }
        }
    } else if init_guess_type == "gaussian" {
        // Gaussian-weighted initial guess strategy:
        //   1. Extend the search window by 50% on each side → sampling window
        //   2. Divide sampling window into m0-1 equal segments → m0 equally spaced E_k
        //   3. For each E_k, build weight vector w_j = exp(-(D_j - E_k)² / a)
        //      where a = (half spacing)², then normalize and apply random signs.
        let l = λ_max - λ_min;
        let samp_min = λ_min - 0.5 * l;
        let samp_max = λ_max + 0.5 * l;
        let step = if m0 > 1 { (samp_max - samp_min) / (m0 - 1) as f64 } else { 0.0 };
        let half_step = step * gaussian_width_factor;
        let a = half_step * half_step; // Gaussian width parameter

        if let Some(diag) = init_diag {
            for j in 0..m0 {
                let e_k = samp_min + j as f64 * step;
                // Compute unnormalized Gaussian weights
                let raw_w: Vec<f64> = if a > 1e-30 {
                    diag.iter()
                        .map(|&d| {
                            let de = d - e_k;
                            (-de * de / a).exp()
                        })
                        .collect()
                } else {
                    // a is effectively zero: all weight at the sampling energy
                    diag.iter()
                        .map(|&d| {
                            let de = (d - e_k).abs();
                            if de < 1e-12 { 1.0 } else { 0.0 }
                        })
                        .collect()
                };
                // Normalize
                let norm: f64 = raw_w.iter().map(|&wi| wi * wi).sum::<f64>().sqrt();
                if norm > 1e-30 {
                    for i in 0..n {
                        let sign = if rng.random::<f64>() > 0.5 { 1.0 } else { -1.0 };
                        y[[i, j]] = sign * raw_w[i] / norm;
                    }
                } else {
                    // Fallback to random if all weights vanish
                    for i in 0..n {
                        y[[i, j]] = rng.random::<f64>() * 2.0 - 1.0;
                    }
                }
            }
        } else {
            // Fallback to random if no diag available
            for j in 0..m0 {
                for i in 0..n {
                    y[[i, j]] = rng.random::<f64>() * 2.0 - 1.0;
                }
            }
        }
    } else {
        // Default random strategy
        for j in 0..m0 {
            for i in 0..n {
                y[[i, j]] = rng.random::<f64>() * 2.0 - 1.0; // uniform in [-1, 1]
            }
        }
    }

    let mut trace_old = 0.0;

    // ---- Step 2: FEAST outer loop ------------------------------------------
    for iter in 0..max_feast_iter {
        let t_iter = Instant::now();
        println!("=== FEAST iteration {} start ===", iter);

        // ---- Pre-compute B * Y (needed for all quadrature nodes) ------------
        let mut by = MatrixFull::new([n, m0], 0.0);
        for j in 0..m0 {
            let yj: Vec<f64> = (0..n).map(|i| y[[i, j]]).collect();
            let byj = gmres_b(&yj);
            for i in 0..n {
                by[[i, j]] = byj[i];
            }
        }

        // ── Step 2a: contour integration over Gauss-Legendre nodes ──
        // Two modes controlled by `use_contour_rayon`:
        //   true  — parallel work-stealing: all (qp, col) pairs flattened into
        //           a single work queue.  Rayon distributes them dynamically,
        //           so finished threads pick up remaining work automatically.
        //   false — serial: outer loop over qp, inner loop over columns.
        //           DGEMM inside each GMRES can still use OpenMP threads.
        let t_gmres = Instant::now();
        println!("    [Phase GMRES] Starting contour integration ({} quadrature points × {} columns = {} total solves, rayon={}) ...",
                 quad.len(), m0, quad.len() * m0, use_contour_rayon);

        // Pre-build preconditioners for all quadrature points.
        // Supports two types:
        //   "diagonal"    — 2×2 block-diagonal (default for Round 1)
        //   "inner_gmres" — inner GMRES with s-only basis as implicit inverse
        let preconds: Vec<Option<GmresPrecond>> = if precond_type == "inner_gmres"
            && precond_a_mul.is_some() && precond_b_mul.is_some()
            && precond_diag.is_some()
        {
            let pa = precond_a_mul.as_ref().unwrap();
            let pb = precond_b_mul.as_ref().unwrap();
            let pd = precond_diag.unwrap();
            let inner_restart = inner_gmres_restart;     // configurable restart
            let inner_maxit = inner_gmres_max_iter;      // configurable max iterations
            let inner_tol = inner_gmres_tol; // configurable inner GMRES tolerance
            quad.iter().map(move |qp| {
                let inner = InnerGmresPrecond {
                    precond_a: Arc::clone(&pa),
                    precond_b: Arc::clone(&pb),
                    z_re: qp.z_re,
                    z_im: qp.z_im,
                    inner_restart,
                    inner_max_iter: inner_maxit,
                    inner_tol: inner_tol,
                    diag: pd.clone(),
                };
                Some(GmresPrecond::InnerGmres(inner))
            }).collect()
        } else {
            quad.iter().map(|qp| {
                diag_a.map(|d| {
                    let mut re_part = vec![0.0; n];
                    let mut im_part = vec![0.0; n];
                    for i in 0..n {
                        re_part[i] = qp.z_re - d[i];
                        im_part[i] = -qp.z_im;
                    }
                    GmresPrecond::Diag(BlockDiagPrecond::from_re_im(&re_part, &im_part))
                })
            }).collect()
        };

        // ── Helper closure: solve one (qp, col) and return its contribution ──
        // Returns a Vec<f64> of length n (the scaled contour contribution).
        let solve_one_col = |qp_idx: usize, col: usize, verbose: bool| -> (Vec<f64>, usize) {
            let qp = &quad[qp_idx];

            // Build M₂ matvec: [α·B−A, −β·B;  β·B,  α·B−A]
            let m2_matvec = |v: &Vec<f64>| -> Vec<f64> {
                let vr: Vec<f64> = v[0..n].to_vec();
                let vi: Vec<f64> = v[n..2 * n].to_vec();
                let b_vr = gmres_b(&vr);
                let b_vi = gmres_b(&vi);
                let a_vr = gmres_a(&vr);
                let a_vi = gmres_a(&vi);
                let mut result = vec![0.0; 2 * n];
                for i in 0..n {
                    result[i]       = qp.z_re * b_vr[i] - a_vr[i] - qp.z_im * b_vi[i];
                    result[n + i]   = qp.z_im * b_vr[i] + qp.z_re * b_vi[i] - a_vi[i];
                }
                result
            };

            // RHS: column `col` of the pre-computed B·Y matrix
            let br: Vec<f64> = (0..n).map(|i| by[[i, col]]).collect();
            let mut b_2n = vec![0.0; 2 * n];
            b_2n[..n].copy_from_slice(&br);

            // Solve via restarted GMRES
            let (x_2n, matvec_cnt) = gmres(
                &m2_matvec, &b_2n,
                gmres_restart, gmres_max_iter, gmres_tol,
                verbose,
                preconds[qp_idx].as_ref(),
            );


            // Contribution = w · (r·cosθ · x_re  −  r·sinθ · x_im)
            let fac_re = qp.w * r * qp.cosθ;
            let fac_im = qp.w * r * qp.sinθ;
            let mut contrib = vec![0.0; n];
            for i in 0..n {
                contrib[i] = fac_re * x_2n[i] - fac_im * x_2n[n + i];
            }
            (contrib, matvec_cnt)
        };

        let q_acc: Vec<Vec<f64>> = if use_contour_rayon {
            // ── Parallel work-stealing path ──
            let q_accum: Vec<Mutex<Vec<f64>>> = (0..m0).map(|_| Mutex::new(vec![0.0; n])).collect();
            let col_counts: Vec<Mutex<usize>> = (0..m0).map(|_| Mutex::new(0usize)).collect();
            let work_items: Vec<(usize, usize)> = (0..quad.len())
                .flat_map(|q| (0..m0).map(move |j| (q, j)))
                .collect();
            let total_solves = work_items.len();
            let solves_done = AtomicUsize::new(0);

            work_items.par_iter().for_each(|&(qp_idx, col)| {
                let verbose = true;
                let (contrib, matvecs) = solve_one_col(qp_idx, col, verbose);

                let done = solves_done.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 100 == 0 {
                    println!("  GMRES progress: {}/{} columns solved", done, total_solves);
                }

                let mut col_lock = q_accum[col].lock().unwrap();
                for i in 0..n {
                    col_lock[i] += contrib[i];
                }
                let mut cnt_lock = col_counts[col].lock().unwrap();
                *cnt_lock += matvecs;
            });

            // Print GMRES convergence statistics
            let counts: Vec<usize> = col_counts.into_iter()
                .map(|m| m.into_inner().unwrap())
                .collect();
            let max_mv = *counts.iter().max().unwrap_or(&0);
            let avg_mv = counts.iter().sum::<usize>() as f64 / counts.len().max(1) as f64;
            println!("    [Phase GMRES] Column GMRES matvecs: max={}, avg={:.1}", max_mv, avg_mv);

            // Unwrap all Mutexes into plain Vecs
            q_accum.into_iter()
                .map(|m| m.into_inner().unwrap())
                .collect()
        } else {
            // ── Serial path: quadrature points one by one ──
            let mut q_acc: Vec<Vec<f64>> = (0..m0).map(|_| vec![0.0; n]).collect();
            let mut col_counts: Vec<usize> = vec![0usize; m0];
            let mut total_done = 0usize;
            let total_solves = quad.len() * m0;

            for qp_idx in 0..quad.len() {
                println!("    Quadrature point {}/{} ...", qp_idx + 1, quad.len());
                for col in 0..m0 {
                    let verbose = true;
                    let (contrib, matvecs) = solve_one_col(qp_idx, col, verbose);
                    for i in 0..n {
                        q_acc[col][i] += contrib[i];
                    }
                    col_counts[col] += matvecs;
                    total_done += 1;
                    if total_done % 100 == 0 {
                        println!("  GMRES progress: {}/{} columns solved", total_done, total_solves);
                    }
                }
            }
            let max_mv = *col_counts.iter().max().unwrap_or(&0);
            let avg_mv = col_counts.iter().sum::<usize>() as f64 / col_counts.len().max(1) as f64;
            println!("    [Phase GMRES] Column GMRES matvecs: max={}, avg={:.1}", max_mv, avg_mv);
            q_acc
        };

        // ── Build Q matrix from per-column accumulators ──
        let mut q = MatrixFull::new([n, m0], 0.0);
        for j in 0..m0 {
            for i in 0..n {
                q[[i, j]] = q_acc[j][i];
            }
        }

        println!("    [Phase GMRES] All quadrature GMRES solves done [{:?}]", t_gmres.elapsed());

        // ── Step 2a': Subspace orthogonalization and compression ──
        // Purify Q by removing near-linear-dependent directions:
        //   1. Compute Gram matrix S = Q^T·Q
        //   2. Eigendecompose S = V·Λ·V^T
        //   3. Keep eigenvectors with eigenvalue > 1e-4
        //   4. Form orthonormal Q_tilde = Q·V_selected·diag(1/√λ)
        let t_a = Instant::now();
        let qtq = _dgemm_scaled(&q, 'T', &q, 'N', 1.0);
        let (v_opt, eigval_qtq, _) = _dsyevd(&qtq, 'V');
        let v = v_opt.unwrap();
        let threshold = 1e-6;
        let m_eff = std::cmp::max(1, eigval_qtq.iter().filter(|&&val| val > threshold).count());
        if m_eff < m0 {
            let offset = m0 - m_eff;
            let mut q_tilde = MatrixFull::new([n, m_eff], 0.0);
            for j in 0..m_eff {
                let inv_sqrt = 1.0 / eigval_qtq[offset + j].sqrt();
                for i in 0..n {
                    let mut s = 0.0;
                    for k in 0..m0 {
                        s += q[[i, k]] * v[[k, offset + j]];
                    }
                    q_tilde[[i, j]] = s* inv_sqrt;
                }
            }
            q = q_tilde;
            println!("FEAST iter {}: subspace compressed from {} to {} [{:?}]", iter, m0, m_eff, t_a.elapsed());
            m0 = m_eff;
            subspace_ever_compressed = true;
        } else if !subspace_ever_compressed {
            panic!("No Rank Deficiency in first FEAST iteration. Consider Enlarging m_expected");
        }

        // ── Step 2b: Rayleigh–Ritz — form reduced matrices ──
        //   A_Q = Q^T · A · Q      (m_eff × m_eff)
        //   B_Q = Q^T · B · Q      (m_eff × m_eff)
        //
        // Compute A·Q and B·Q one column at a time using closures.
        let t_b = Instant::now();
        println!("    [Phase B] Rayleigh–Ritz: computing A·Q and B·Q (m_eff={}) ...", m_eff);
        let mut aq = MatrixFull::new([n, m_eff], 0.0);
        let mut bq = MatrixFull::new([n, m_eff], 0.0);
        for j in 0..m_eff {
            let qj: Vec<f64> = (0..n).map(|i| q[[i, j]]).collect();
            let aqj = a_mul(&qj);
            let bqj = b_mul(&qj);
            for i in 0..n {
                aq[[i, j]] = aqj[i];
                bq[[i, j]] = bqj[i];
            }
        }
        let mut a_q = _dgemm_scaled(&q, 'T', &aq, 'N', 1.0); // Q^T · A·Q
        let mut b_q = _dgemm_scaled(&q, 'T', &bq, 'N', 1.0); // Q^T · B·Q
        println!("    [Phase B] A·Q and B·Q done [{:?}], solving reduced EVP (m_eff={}) ...", t_b.elapsed(), m_eff);
        // let (_, wr_aq, _, _, _, _) = _dgeev(&a_q, 'N', 'N');
        // let min_aq = wr_aq.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        // println!("Minimum eigenvalue of A_Q (subspace) matrix: {}", min_aq);
        // let (_, wr_bq, _, _, _, _) = _dgeev(&b_q, 'N', 'N');
        // let min_bq = wr_bq.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        // println!("Minimum eigenvalue of B_Q (subspace) matrix: {}", min_bq);


        // ── Step 2c: Rayleigh-Ritz — solve reduced eigenvalue problem ──
        // B_Q should be SPD for the standard Cholesky-based reduction path.
        // If not SPD, fall back to solving B_Q^{-1}·A_Q via dgeev.

        // let aq_p_bq=MatrixFull::add(&a_q,&b_q).unwrap();
        // let mut bq_clone=b_q.clone();
        // bq_clone.self_multiple(-1.0);
        // let aq_m_bq=MatrixFull::add(&a_q,&bq_clone).unwrap();
        // let (_, wr_apb, _, _, _, _) = _dgeev(&aq_p_bq, 'N', 'N');
        // let min_apb = wr_apb.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        // println!("Minimum eigenvalue of A_Q + B_Q (subspace) matrix: {}", min_apb);
        // let (_, wr_amb, _, _, _, _) = _dgeev(&aq_m_bq, 'N', 'N');
        // let min_amb = wr_amb.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        // println!("Minimum eigenvalue of A_Q - B_Q (subspace) matrix: {}", min_amb);
        let cholesky_ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            _dpotrf(&mut b_q, 'L');
        }));

        let (lambda, x_new) = if cholesky_ok.is_ok() {
            // ── Cholesky path: H = L⁻¹ · A_Q · L⁻ᵀ, then dsyevd ──

            // Extract lower-triangular L from B_Q (after dpotrf, L is in lower triangle)
            let mut l_mat = MatrixFull::new([m_eff, m_eff], 0.0);
            for j in 0..m_eff {
                for i in j..m_eff {
                    l_mat[[i, j]] = b_q[[i, j]];
                }
            }
            // Invert L
            let l_inv = match _dinverse(&l_mat) {
                Some(m) => m,
                None => {
                    eprintln!("Warning: could not invert L.  Using identity fallback.");
                    let mut eye = MatrixFull::new([m_eff, m_eff], 0.0);
                    for i in 0..m_eff {
                        eye[[i, i]] = 1.0;
                    }
                    eye
                }
            };
            let l_inv_t = l_inv.transpose();

            // H = L⁻¹ · A_Q · L⁻ᵀ
            let tmp = _dgemm_scaled(&l_inv, 'N', &a_q, 'N', 1.0);
            let h = _dgemm_scaled(&tmp, 'N', &l_inv_t, 'N', 1.0);

            // Standard symmetric eigenvalue problem: H · Ψ = Ψ · diag(Λ)
            let (psi_opt, lambda, _) = _dsyevd(&h, 'V');
            let psi = psi_opt.unwrap();

            // Φ = L⁻ᵀ · Ψ
            let phi = _dgemm_scaled(&l_inv_t, 'N', &psi, 'N', 1.0);

            // Ritz vectors: X_new = Q · Φ
            let x_new = _dgemm_scaled(&q, 'N', &phi, 'N', 1.0);

            (lambda, x_new)

        } else {
            // ── B_Q not SPD → fallback: solve B_Q^{-1}·A_Q via dgeev ──
            eprintln!(
                "Warning: B_Q not SPD at iteration {}.  Using dgeev fallback.",
                iter
            );

            // Attempt to invert B_Q; if singular, fall through to dsyevd on A_Q
            let b_q_inv_opt = _dinverse(&b_q);
            let (lambda, x_new) = if let Some(b_q_inv) = b_q_inv_opt {
                // B_Q invertible: form B_Q^{-1}·A_Q and solve via dgeev
                let b_inv_a = _dgemm_scaled(&b_q_inv, 'N', &a_q, 'N', 1.0);
                let (_, wr, wi, _, vr, info) = _dgeev(&b_inv_a, 'N', 'V');

                if info != 0 {
                    eprintln!("Warning: dgeev returned info={} at iteration {}.", info, iter);
                }

                // Select real eigenvalues (BSE should give real excitation energies)
                let mut eigen_pairs: Vec<(usize, f64)> = (0..m_eff)
                    .filter(|&j| wi[j].abs() < 1.0e-10)
                    .map(|j| (j, wr[j]))
                    .collect();
                eigen_pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                let mut lambda: Vec<f64> = eigen_pairs.iter().map(|&(_, val)| val).collect();
                let m_selected = eigen_pairs.len();

                // Build phi from selected right eigenvectors
                let mut phi = MatrixFull::new([m_eff, m_selected], 0.0);
                for (col, &(j, _)) in eigen_pairs.iter().enumerate() {
                    for i in 0..m_eff {
                        phi[[i, col]] = vr[[i, j]];
                    }
                }

                // Ritz vectors: X_new = Q · Φ
                let mut x_new = _dgemm_scaled(&q, 'N', &phi, 'N', 1.0);

                // Pad lambda and x_new back to m_eff columns (for subspace consistency)
                if m_selected < m_eff {
                    lambda.resize(m_eff, 0.0);
                    let mut x_new_padded = MatrixFull::new([n, m_eff], 0.0);
                    for j in 0..m_selected {
                        for i in 0..n {
                            x_new_padded[[i, j]] = x_new[[i, j]];
                        }
                    }
                    x_new = x_new_padded;
                }
                (lambda, x_new)
            } else {
                // B_Q is singular → use A_Q directly (identity metric)
                eprintln!("Warning: B_Q is singular.  Using A_Q directly via dsyevd.");
                let (psi_opt, lambda, _) = _dsyevd(&a_q, 'V');
                let psi = psi_opt.unwrap();
                let x_new = _dgemm_scaled(&q, 'N', &psi, 'N', 1.0);
                (lambda, x_new)
            };
            (lambda, x_new)
        };

        // ── Step 2f: select eigenvalues inside the interval ──
        let inside: Vec<usize> = lambda
            .iter()
            .enumerate()
            .filter(|(_, &val)| val >= λ_min - 1e-12 && val <= λ_max + 1e-12)
            .map(|(idx, _)| idx)
            .collect();
        let m_inside = inside.len();

        // ── Step 2g: convergence check via trace ──
        let trace_new: f64 = inside.iter().map(|&j| lambda[j]).sum();
        if iter > 0 {
            let δ_trace = (trace_new - trace_old).abs() / scale;
            println!("  FEAST iter {}: trace change δ = {:.2e} (tol = {:.2e})", iter, δ_trace, tol_feast);
            if δ_trace <= tol_feast {
                // Build final result
                return extract_eigenpairs(
                    &lambda, &x_new, n, &inside, &λ_min, &λ_max,
                );
            }
        }
        trace_old = trace_new;

        // ── Step 2h: prepare Y for next iteration: Y = B · X_new ──
        let t_y = Instant::now();
        println!("    [Phase Y] Preparing Y for next iteration (m0={}) ...", m0);
        for j in 0..m0 {
            let xj: Vec<f64> = (0..n).map(|i| x_new[[i, j]]).collect();
            let yj = b_mul(&xj);
            for i in 0..n {
                y[[i, j]] = yj[i];
            }
        }
        println!("    [Phase Y] Y preparation done [{:?}]", t_y.elapsed());
        println!("=== FEAST iteration {} done, total [{:?}] ===", iter, t_iter.elapsed());
    }

    // ---- Not converged within max iterations; return best available result ----
    // We re-run RQ to get the final values (simplified: just return empty)
    vec![]
}

/// Helper: extract eigenpairs inside the interval and sort ascending.
fn extract_eigenpairs(
    lambda: &Vec<f64>,
    x_new: &MatrixFull<f64>,
    n: usize,
    inside: &[usize],
    _λ_min: &f64,
    _λ_max: &f64,
) -> Vec<(f64, Vec<f64>)> {
    let m_inside = inside.len();
    if m_inside == 0 {
        return vec![];
    }

    // Collect (index, value) pairs and sort by eigenvalue
    let mut pairs: Vec<(usize, f64)> = inside.iter().map(|&j| (j, lambda[j])).collect();
    pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut result = Vec::with_capacity(m_inside);
    for &(j, val) in &pairs {
        let eigvec: Vec<f64> = (0..n).map(|i| x_new[[i, j]]).collect();
        result.push((val, eigvec));
    }
    result
}
