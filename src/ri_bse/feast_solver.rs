// ============================================================================
// FEAST Solver for Generalized Eigenvalue Problem A x = λ B x
//
// Implements the basic linear FEAST algorithm (Polizzi 2009) to find all
// eigenvalues of (A, B) within a specified interval [λ_min, λ_max].
//
// The solver only accesses A and B through matrix-vector product closures
// `a_mul(x)` and `b_mul(x)`, satisfying:
//   a_mul(x) = A * x   where A is real symmetric
//   b_mul(x) = B * x   where B is symmetric positive definite
//
// Linear systems (z*B - A) * X = B*Y (with complex shift z = α + iβ)
// are solved by embedding as a 2N-dimensional real system and using
// LAPACK's dgesv (LU factorization).  The 2N × 2N matrix is assembled
// via O(N) calls to the a_mul / b_mul closures on unit vectors.
// ============================================================================
use crate::tensors::MathMatrix;
use crate::scf_io::SCF;
use rest_tensors::matrix::matrix_blas_lapack::{
    _dgemm_scaled,_dgemm_full, _dpotrf, _dsyevd, _dinverse, _dgeev,
};
use rest_tensors::{BasicMatrix, MatrixFull};
use rand::Rng;
use rayon::prelude::*;
use crate::ri_bse::{get_submatrix,construct_inverse_dielectric,construct_energy_diag_for_a};
use crate::ri_gw::{get_occupation_parameters};
use crate::ri_bse::matvec;
use crate::ctrl_io::quasiparticle_methods::QuasiParticle;
use std::time::Instant;
use std::io::Write;
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global log file for inner GMRES preconditioner statistics.
static INNER_GMRES_LOG: OnceLock<Mutex<std::fs::File>> = OnceLock::new();

fn inner_gmres_log() -> &'static Mutex<std::fs::File> {
    INNER_GMRES_LOG.get_or_init(|| {
        Mutex::new(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("inner-gmres.log")
                .expect("Failed to open inner-gmres.log"),
        )
    })
}

/// Write a line to the inner GMRES log file (thread-safe).
fn log_inner(msg: &str) {
    if let Ok(mut f) = inner_gmres_log().lock() {
        let _ = writeln!(f, "{}", msg);
    }
}

// ---------------------------------------------------------------------------
// 8-point Gauss-Legendre quadrature nodes and weights (from Appendix A)
// ---------------------------------------------------------------------------
const GL_POINTS: [(f64, f64); 8] = [
    (-0.960289856497536, 0.101228536290376),
    (-0.796666477413626, 0.222381034453374),
    (-0.525532409916328, 0.313706645877887),
    (-0.183434642495649, 0.362683783378361),
    ( 0.183434642495649, 0.362683783378361),
    ( 0.525532409916328, 0.313706645877887),
    ( 0.796666477413626, 0.222381034453374),
    ( 0.960289856497536, 0.101228536290376),
];

// ============================================================================
// Section 1: CG Solver — standard Conjugate Gradient
// ============================================================================

/// Solve A * x = b using CG, where A is an SPD matrix accessed
/// through the closure `a_mul`.  Returns the solution vector x.
pub fn cg(
    a_mul: impl Fn(&Vec<f64>) -> Vec<f64>,
    b: &Vec<f64>,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    let ax = a_mul(&x);
    let mut r: Vec<f64> = b.iter().zip(&ax).map(|(bi, axi)| bi - axi).collect();
    let mut p = r.clone();
    let mut r_dot_r: f64 = r.iter().map(|ri| ri * ri).sum();

    for iter in 0..max_iter {
        let ap = a_mul(&p);
        let p_dot_ap: f64 = p.iter().zip(&ap).map(|(pi, api)| pi * api).sum();
        if p_dot_ap.abs() < 1e-30 {
            break;
        }
        let alpha = r_dot_r / p_dot_ap;
        for i in 0..n {
            x[i] += alpha * p[i];
        }
        let r_new: Vec<f64> = r.iter().zip(&ap).map(|(ri, api)| ri - alpha * api).collect();
        let residual: f64 = r_new.iter().map(|ri| ri * ri).sum::<f64>().sqrt();
        if residual < tol {
            break;
        }
        let r_new_dot_r_new: f64 = r_new.iter().map(|ri| ri * ri).sum();
        let beta = r_new_dot_r_new / r_dot_r;
        for i in 0..n {
            p[i] = r_new[i] + beta * p[i];
        }
        r = r_new;
        r_dot_r = r_new_dot_r_new;
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

        log_inner(&format!("INNER {}", inner_matvecs));
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
    log_inner(&format!("=== FEAST start m0={} λ_min={} λ_max={} ===", m0, λ_min, λ_max));

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
    let quad: Vec<QuadData> = GL_POINTS
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
    let mut rng = rand::thread_rng();
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
                y[[i, j]] = rng.gen::<f64>() * 2.0 - 1.0;
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
                        let sign = if rng.gen::<f64>() > 0.5 { 1.0 } else { -1.0 };
                        y[[i, j]] = sign * raw_w[i] / norm;
                    }
                } else {
                    // Fallback to random if all weights vanish
                    for i in 0..n {
                        y[[i, j]] = rng.gen::<f64>() * 2.0 - 1.0;
                    }
                }
            }
        } else {
            // Fallback to random if no diag available
            for j in 0..m0 {
                for i in 0..n {
                    y[[i, j]] = rng.gen::<f64>() * 2.0 - 1.0;
                }
            }
        }
    } else {
        // Default random strategy
        for j in 0..m0 {
            for i in 0..n {
                y[[i, j]] = rng.gen::<f64>() * 2.0 - 1.0; // uniform in [-1, 1]
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

            log_inner(&format!("OUTER qp={} col={} {}", qp_idx, col, matvec_cnt));

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

// ============================================================================
// Section 4: BSE-specific FEAST solver entry points
// ============================================================================

/// Build s-only matvec closures and diag for use as inner GMRES preconditioner.
/// Must be called while `ri3fn_bse`/`rimatr_bse` are still populated (before clearing).
/// Returns (a_mul, b_mul, diag) where a_mul/b_mul are `Box<dyn Fn>` closures
/// and diag is the diagonal of A (energy gaps) used for inner GMRES.
fn build_s_only_precond_data(
    scf_data: &mut SCF,
    qp_ctrl: &QuasiParticle,
    quasiparticle_energies: &Vec<f64>,
    occ_size: usize,
    vir_size: usize,
    bse_tda: bool,
) -> (
    Option<Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync>>,
    Option<Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync>>,
    Option<Vec<f64>>,
) {
    let ks_energies: Vec<f64> = scf_data.eigenvalues[0].clone();
    let epsilon: Vec<f64> = if qp_ctrl.bse_qp_polarization {
        quasiparticle_energies.clone()
    } else {
        ks_energies.clone()
    };
    let inverse_dielectric = construct_inverse_dielectric(scf_data, &epsilon);

    // Build QP energy gaps diagonal
    let energies: Vec<f64> = if qp_ctrl.bse_qp_polarization {
        scf_data.gwqp.0.clone()
    } else {
        scf_data.eigenvalues[0].clone()
    };
    let diag_a: Vec<f64> = construct_energy_diag_for_a(&energies, occ_size, vir_size);

    let qp_ctrl_c = qp_ctrl.clone();

    if bse_tda {
        // ── TDA: a_mul = A_matvec(s-only), b_mul = identity ──
        let num_auxbas = inverse_dielectric.size[0];
        let ri_ov = get_submatrix(scf_data, 'O', 'V', 'N');
        let mut ri_vv = get_submatrix(scf_data, 'V', 'V', 'N');
        let ri_oo = get_submatrix(scf_data, 'O', 'O', 'N');

        let mut ri_oo_tilde = MatrixFull::new(ri_oo.size, 0.0);
        _dgemm_full(&inverse_dielectric, 'N', &ri_oo, 'N', &mut ri_oo_tilde, 1.0, 0.0);
        ri_oo_tilde.reshape([num_auxbas * occ_size, occ_size]);
        ri_oo_tilde = ri_oo_tilde.transpose_and_drop();
        ri_oo_tilde.reshape([occ_size * num_auxbas, occ_size]);
        ri_vv.reshape([num_auxbas * vir_size, vir_size]);

        // Capture all needed data by value for Send + Sync safety
        let occ_v = occ_size;
        let vir_v = vir_size;
        let qpc = qp_ctrl_c;
        let energies_s = energies.clone();

        let a_mul: Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync> = Arc::new(
            move |z_vec: &Vec<f64>| -> Vec<f64> {
                let xlet = if qpc.bse_spin == "triplet" { 'T' }
                           else if qpc.bse_spin == "singlet" { 'S' } else { 'R' };
                // Diagonal contribution
                let mut result = matvec::diagonal_contribution_standalone(
                    z_vec, &energies_s, occ_v, vir_v);
                // W contribution (s-only RI)
                let w = matvec::w_contribution_a_block_dgemm_standalone(
                    &ri_vv, z_vec, &ri_oo_tilde, &qpc, occ_v, vir_v);
                result = w.iter().zip(result.iter()).map(|(w_i, z_i)| -w_i + z_i).collect();
                // Coulomb contribution (full RI for accuracy)
                if xlet == 'S' {
                    let v = matvec::coulomb_contribution(&ri_ov, z_vec);
                    result = v.iter().zip(result.iter()).map(|(v_i, z_i)| 2.0 * v_i + z_i).collect();
                } else if xlet == 'R' {
                    let v = matvec::coulomb_contribution(&ri_ov, z_vec);
                    result = v.iter().zip(result.iter()).map(|(v_i, z_i)| v_i + z_i).collect();
                }
                result
            },
        );
        let b_mul: Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync> =
            Arc::new(|z_vec: &Vec<f64>| z_vec.clone());

        (Some(a_mul), Some(b_mul), Some(diag_a))
    } else {
        // non-TDA: for now fall back to diagonal (inner GMRES precond not yet implemented for non-TDA)
        println!("Warning: inner_gmres preconditioner not yet implemented for non-TDA; using diagonal instead.");
        (None, None, None)
    }
}

/// Filter eigenvalues and eigenvectors to only those within [emin, emax].
fn filter_eigenpairs(
    eigenvals: Vec<f64>,
    eigenvecs: Vec<Vec<f64>>,
    emin: f64,
    emax: f64,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut fil_vals = Vec::new();
    let mut fil_vecs = Vec::new();
    for (e, v) in eigenvals.into_iter().zip(eigenvecs.into_iter()) {
        if e >= emin - 1e-12 && e <= emax + 1e-12 {
            fil_vals.push(e);
            fil_vecs.push(v);
        }
    }
    (fil_vals, fil_vecs)
}

/// Print Rayleigh-Ritz eigenpairs using the same format as the BSE excitation output.
fn print_ritz_eigenpairs(
    scf_data: &SCF,
    ritz_eigenvalues: &Vec<f64>,
    ritz_eigenvectors: &Vec<Vec<f64>>,
    label: &str,
    occ_size: usize,
    vir_size: usize,
) {
    let dipole_matrix = super::dipoles::compute_dipole_matrix(scf_data);
    let k = ritz_eigenvalues.len();
    println!("Rayleigh-Ritz {}: {} excitations within the window:", label, k);
    for (n, (e, vec)) in ritz_eigenvalues.iter().zip(ritz_eigenvectors.iter()).enumerate() {
        let v = super::dipoles::normalize(vec, true);
        let dipole_square = super::dipoles::transition_dipole_square(&dipole_matrix, &v, true);
        println!("#{} Excitation energy={}, norm={:.6}", n, e,
                 vec.iter().map(|x| x*x).sum::<f64>().sqrt());
        println!("Transition Dipole Square:{}; Oscillator Strength:{}",
                 dipole_square, dipole_square * e * 2.0 / 3.0);
        super::leading_components(&v, occ_size, vir_size);
    }
}

/// Rayleigh-Ritz refinement: project Round 1 eigenvectors onto the full A matrix
/// (all angular momenta) and diagonalize the projected subspace to get optimal
/// linear combinations for warm-starting Round 2 FEAST.
/// Returns (Ritz_eigenvalues, Ritz_eigenvectors).
fn rayleigh_ritz_refine(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    quasiparticle_energies: &Vec<f64>,
    occ_size: usize,
    vir_size: usize,
    eigvecs: &Vec<Vec<f64>>,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let k = eigvecs.len();
    if k == 0 {
        return (Vec::new(), Vec::new());
    }

    let ks_energies: Vec<f64> = scf_data.eigenvalues[0].clone();
    let epsilon: Vec<f64> = if qp_ctrl.bse_qp_polarization {
        quasiparticle_energies.clone()
    } else {
        ks_energies.clone()
    };
    let inverse_dielectric = construct_inverse_dielectric(scf_data, &epsilon);

    // Build full A-matrix matvec closure (same pattern as feast_solve_bse_tda)
    let num_auxbas = inverse_dielectric.size[0];
    let ri_ov = get_submatrix(scf_data, 'O', 'V', 'N');
    let mut ri_vv = get_submatrix(scf_data, 'V', 'V', 'N');
    let ri_oo = get_submatrix(scf_data, 'O', 'O', 'N');

    let mut ri_oo_tilde = MatrixFull::new(ri_oo.size, 0.0);
    _dgemm_full(&inverse_dielectric, 'N', &ri_oo, 'N', &mut ri_oo_tilde, 1.0, 0.0);
    ri_oo_tilde.reshape([num_auxbas * occ_size, occ_size]);
    ri_oo_tilde = ri_oo_tilde.transpose_and_drop();
    ri_oo_tilde.reshape([occ_size * num_auxbas, occ_size]);
    ri_vv.reshape([num_auxbas * vir_size, vir_size]);

    let a_matvec = |z: &Vec<f64>| -> Vec<f64> {
        matvec::a_block_matvec(scf_data, qp_ctrl, &ri_vv, &ri_ov, &ri_oo_tilde, z)
    };

    // Normalize input eigenvectors to unit L2 norm
    let mut eigvecs_norm: Vec<Vec<f64>> = Vec::with_capacity(k);
    for v in eigvecs.iter() {
        let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            eigvecs_norm.push(v.iter().map(|&x| x / norm).collect());
        } else {
            eigvecs_norm.push(v.clone());
        }
    }

    // Compute A * v_i for each normalized eigenvector
    let av: Vec<Vec<f64>> = eigvecs_norm.iter().map(|v| a_matvec(v)).collect();

    // Check for NaN/Inf in A*v results
    let mut av_bad = false;
    let mut av_bad_col = 0;
    let mut av_bad_kind = "";
    for (j, v) in av.iter().enumerate() {
        for &x in v.iter() {
            if x.is_nan() { av_bad = true; av_bad_kind = "NaN"; av_bad_col = j; break; }
            if x.is_infinite() { av_bad = true; av_bad_kind = "Inf"; av_bad_col = j; break; }
        }
        if av_bad { break; }
    }
    if av_bad {
        eprintln!("Warning: Rayleigh-Ritz A*vector contains {} in column {} — \
                   using original eigenvectors.", av_bad_kind, av_bad_col);
        return (Vec::new(), eigvecs.clone());
    }

    // Pack eigenvectors and A*v into MatrixFull for BLAS-based projection.
    // Use _dgemm_scaled (same path that FEAST subspace diagonalization uses
    // successfully) instead of manual dot products.
    let n = occ_size * vir_size;
    let mut v_mat = MatrixFull::new([n, k], 0.0);
    let mut av_mat = MatrixFull::new([n, k], 0.0);
    for j in 0..k {
        for i in 0..n {
            v_mat[[i, j]] = eigvecs_norm[j][i];
            av_mat[[i, j]] = av[j][i];
        }
    }

    // H_proj = V^T · (A·V)   via BLAS
    let h_proj = _dgemm_scaled(&v_mat, 'T', &av_mat, 'N', 1.0);

    // Diagnostic
    let mut h_min = f64::INFINITY; let mut h_max = f64::NEG_INFINITY;
    for i in 0..k {
        for j in 0..k {
            let hv = h_proj[[i, j]];
            if hv.is_finite() { h_min = h_min.min(hv); h_max = h_max.max(hv); }
        }
    }
    println!("Rayleigh-Ritz: H_proj range=[{:.3e}, {:.3e}] (k={})", h_min, h_max, k);

    let (ritz_eigenvalues, psi_opt) = if qp_ctrl.bse_tda {
        // TDA: eigenvectors from FEAST are orthonormal (B=I) → standard EVP.
        // Use _dgeev (QR algorithm) instead of _dsyevd (divide-and-conquer)
        // because dsyevd can fail to converge for certain eigenvalue distributions.
        let (_, wr, wi, _, vr, info) = _dgeev(&h_proj, 'N', 'V');
        if info != 0 {
            eprintln!("Warning: Rayleigh-Ritz dgeev failed with info={}, using original eigenvectors", info);
            return (Vec::new(), eigvecs.clone());
        }
        // Select real eigenvalues and sort ascending.
        // h_proj is symmetric so all eigenvalues should be real; the filter
        // is a safety net against numerical noise in dgeev.
        let mut eigen_pairs: Vec<(usize, f64)> = (0..k)
            .filter(|&j| wi[j].abs() < 1e-10)
            .map(|j| (j, wr[j]))
            .collect();
        if eigen_pairs.is_empty() {
            eprintln!("Warning: Rayleigh-Ritz dgeev returned no real eigenvalues, using original eigenvectors");
            return (Vec::new(), eigvecs.clone());
        }
        eigen_pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let evals: Vec<f64> = eigen_pairs.iter().map(|&(_, v)| v).collect();
        let m_sel = eigen_pairs.len();
        let mut psi_mat = MatrixFull::new([k, m_sel], 0.0);
        for (col, &(orig_j, _)) in eigen_pairs.iter().enumerate() {
            for row in 0..k {
                psi_mat[[row, col]] = vr[[row, orig_j]];
            }
        }
        (evals, Some(psi_mat))
    } else {
        // Non-TDA: vectors are 2X components, not orthonormal → generalized EVP
        let mut s_proj = _dgemm_scaled(&v_mat, 'T', &v_mat, 'N', 1.0);
        let mut s_min = f64::INFINITY; let mut s_max = f64::NEG_INFINITY;
        for i in 0..k { for j in 0..k {
            let sv = s_proj[[i, j]];
            if sv.is_finite() { s_min = s_min.min(sv); s_max = s_max.max(sv); }
        }}
        println!("Rayleigh-Ritz: S_proj range=[{:.3e}, {:.3e}] (k={})", s_min, s_max, k);
        solve_generalized_eigenproblem(&h_proj, &mut s_proj, k)
    };
    let psi = match psi_opt {
        Some(p) => p,
        None => {
            eprintln!("Warning: Rayleigh-Ritz generalized EVP failed, using original eigenvectors");
            return (Vec::new(), eigvecs.clone());
        }
    };

    // Transform: ritz_vec[i] = Σ_j psi[j,i] * eigvecs_norm[j], then L2-normalize.
    // _dgeev eigenvectors are not unit-norm, so explicit normalization is needed.
    let mut ritz_vecs: Vec<Vec<f64>> = Vec::with_capacity(ritz_eigenvalues.len());
    for i in 0..ritz_eigenvalues.len() {
        let mut new_v = vec![0.0; n];
        for j in 0..k {
            let coeff = psi[[j, i]];
            for idx in 0..n {
                new_v[idx] += coeff * eigvecs_norm[j][idx];
            }
        }
        let norm: f64 = new_v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in new_v.iter_mut() { *x /= norm; }
        }
        ritz_vecs.push(new_v);
    }

    (ritz_eigenvalues, ritz_vecs)
}

/// Solve generalized eigenvalue problem H·C = S·C·E via Cholesky of S.
/// Only called for non-TDA where basis vectors are not orthonormal.
/// Returns (eigenvalues, Some(eigenvectors)) on success, or (empty, None) on failure.
fn solve_generalized_eigenproblem(
    h: &MatrixFull<f64>,
    s: &mut MatrixFull<f64>,
    k: usize,
) -> (Vec<f64>, Option<MatrixFull<f64>>) {
    // Cholesky S = L·L^T
    let cholesky_ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        _dpotrf(s, 'L');
    }));

    if cholesky_ok.is_err() {
        // S not SPD — fall back to dsyevd on H
        let (psi_opt, evals, info) = _dsyevd(h, 'V');
        if info != 0 {
            eprintln!("Warning: Rayleigh-Ritz dsyevd fallback failed with info={}", info);
            return (Vec::new(), None);
        }
        return (evals, Some(psi_opt.unwrap()));
    }

    // Build lower-triangular L from factorized S
    let mut l_mat = MatrixFull::new([k, k], 0.0);
    for col in 0..k {
        for row in col..k {
            l_mat[[row, col]] = s[[row, col]];
        }
    }
    let l_inv = match _dinverse(&l_mat) {
        Some(m) => m,
        None => {
            let (psi_opt, evals, info) = _dsyevd(h, 'V');
            if info != 0 {
                eprintln!("Warning: Rayleigh-Ritz dsyevd fallback failed with info={}", info);
                return (Vec::new(), None);
            }
            return (evals, Some(psi_opt.unwrap()));
        }
    };
    let l_inv_t = l_inv.transpose();

    // H_trans = L⁻¹ · H · L⁻ᵀ, then standard EVP
    let tmp = _dgemm_scaled(&l_inv, 'N', h, 'N', 1.0);
    let h_trans = _dgemm_scaled(&tmp, 'N', &l_inv_t, 'N', 1.0);

    let (psi_trans_opt, evals, info) = _dsyevd(&h_trans, 'V');
    if info != 0 {
        eprintln!("Warning: Rayleigh-Ritz dsyevd on transformed H failed with info={}", info);
        // Last resort: try dsyevd on original H
        let (psi_opt, evals2, info2) = _dsyevd(h, 'V');
        if info2 != 0 {
            eprintln!("Warning: Rayleigh-Ritz dsyevd last-resort failed with info={}", info2);
            return (Vec::new(), None);
        }
        return (evals2, Some(psi_opt.unwrap()));
    }
    let psi_trans = psi_trans_opt.unwrap();

    // Transform back: C = L⁻ᵀ · Ψ_trans
    let psi = _dgemm_scaled(&l_inv_t, 'N', &psi_trans, 'N', 1.0);
    (evals, Some(psi))
}

/// FEAST solver for singlet BSE excitations (handles both TDA and non-TDA).
/// When `bse_feast_renormalized_doubles` is true, uses s-only FEAST plus
/// Rayleigh-Ritz refinement with full integrals — no second FEAST round.
pub fn feast_solve_bse_singlet(scf_data:&mut SCF)->Vec<(f64,Vec<f64>)>{
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    // For TDA, FEAST solves A*x = omega*x (eigenvalue = omega directly).
    // For non-TDA, FEAST solves for omega^2, so the search range must be squared.
    let (eigenrange_min, eigenrange_max) = if qp_ctrl.bse_tda {
        (qp_ctrl.bse_eigenrange_min, qp_ctrl.bse_eigenrange_max)
    } else {
        (qp_ctrl.bse_eigenrange_min * qp_ctrl.bse_eigenrange_min,
         qp_ctrl.bse_eigenrange_max * qp_ctrl.bse_eigenrange_max)
    };
    let m_expected=qp_ctrl.bse_m_expected;
    let max_feast_iter=qp_ctrl.bse_max_feast_iter;
    let tol_feast=qp_ctrl.bse_tol_feast;
    let quasiparticle_energies=scf_data.gwqp.0.clone();
    let mut qp_ctrl_singlet=qp_ctrl.clone();
    qp_ctrl_singlet.bse_spin=String::from("singlet");

    println!("[DEBUG] feast_solve_bse_singlet: bse_feast_renormalized_doubles={}", qp_ctrl.bse_feast_renormalized_doubles);
    if qp_ctrl.bse_feast_renormalized_doubles {
        let ew = qp_ctrl.bse_renormalized_doubles_extra_width;
        let (r1_min, r1_max) = if qp_ctrl.bse_tda {
            (qp_ctrl.bse_eigenrange_min - ew, qp_ctrl.bse_eigenrange_max + ew)
        } else {
            let orig_min = (qp_ctrl.bse_eigenrange_min - ew).max(0.0);
            let orig_max = qp_ctrl.bse_eigenrange_max + ew;
            (orig_min * orig_min, orig_max * orig_max)
        };
        println!("--- Round 1: BSE-specific RI integrals (s-only) for singlet, range=[{:.6},{:.6}] ---",
                 r1_min, r1_max);
        let round1 = feast_solve_bse_spin(scf_data,&qp_ctrl_singlet,&quasiparticle_energies,
                             r1_min, r1_max, m_expected, max_feast_iter, tol_feast, None,
                             None, None, "diagonal", None, 0.0001, 0, 0);
        let n_found = round1.len();
        println!("Round 1 found {} eigenpairs within the window", n_found);
        let eigvecs: Vec<Vec<f64>> = round1.into_iter().map(|(_, v)| v).collect();

        let (_, _, occ_size, vir_size, _, _) = get_occupation_parameters(scf_data, 'N');

        // Clear BSE-specific integrals → fallback to regular RI for Rayleigh-Ritz
        scf_data.ri3fn_bse = None;
        scf_data.rimatr_bse = None;

        // Rayleigh-Ritz refinement with full integrals
        let (ritz_vals, ritz_vecs) = rayleigh_ritz_refine(scf_data, &qp_ctrl_singlet, &quasiparticle_energies,
                                           occ_size, vir_size, &eigvecs);
        let emin = qp_ctrl.bse_eigenrange_min;
        let emax = qp_ctrl.bse_eigenrange_max;
        let (fil_vals, fil_vecs) = filter_eigenpairs(ritz_vals, ritz_vecs, emin, emax);
        print_ritz_eigenpairs(scf_data, &fil_vals, &fil_vecs, "singlet", occ_size, vir_size);
        fil_vals.into_iter().zip(fil_vecs.into_iter()).collect()
    } else {
        feast_solve_bse_spin(scf_data,&qp_ctrl_singlet,&quasiparticle_energies,
                             eigenrange_min,eigenrange_max,m_expected,max_feast_iter,tol_feast,None,
                             None, None, "diagonal", None, 0.0001, 0, 0)
    }
}

/// FEAST solver for triplet BSE excitations (handles both TDA and non-TDA).
/// When `bse_feast_renormalized_doubles` is true, uses s-only FEAST plus
/// Rayleigh-Ritz refinement with full integrals — no second FEAST round.
pub fn feast_solve_bse_triplet(scf_data:&mut SCF)->Vec<(f64,Vec<f64>)>{
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    // For TDA, FEAST solves A*x = omega*x (eigenvalue = omega directly).
    // For non-TDA, FEAST solves for omega^2, so the search range must be squared.
    let (eigenrange_min, eigenrange_max) = if qp_ctrl.bse_tda {
        (qp_ctrl.bse_eigenrange_min, qp_ctrl.bse_eigenrange_max)
    } else {
        (qp_ctrl.bse_eigenrange_min * qp_ctrl.bse_eigenrange_min,
         qp_ctrl.bse_eigenrange_max * qp_ctrl.bse_eigenrange_max)
    };
    let m_expected=qp_ctrl.bse_m_expected;
    let max_feast_iter=qp_ctrl.bse_max_feast_iter;
    let tol_feast=qp_ctrl.bse_tol_feast;
    let quasiparticle_energies=scf_data.gwqp.0.clone();
    let mut qp_ctrl_triplet=qp_ctrl.clone();
    qp_ctrl_triplet.bse_spin=String::from("triplet");

    if qp_ctrl.bse_feast_renormalized_doubles {
        let ew = qp_ctrl.bse_renormalized_doubles_extra_width;
        let (r1_min, r1_max) = if qp_ctrl.bse_tda {
            (qp_ctrl.bse_eigenrange_min - ew, qp_ctrl.bse_eigenrange_max + ew)
        } else {
            let orig_min = (qp_ctrl.bse_eigenrange_min - ew).max(0.0);
            let orig_max = qp_ctrl.bse_eigenrange_max + ew;
            (orig_min * orig_min, orig_max * orig_max)
        };
        println!("--- Round 1: BSE-specific RI integrals (s-only) for triplet, range=[{:.6},{:.6}] ---",
                 r1_min, r1_max);
        let round1 = feast_solve_bse_spin(scf_data,&qp_ctrl_triplet,&quasiparticle_energies,
                             r1_min, r1_max, m_expected, max_feast_iter, tol_feast, None,
                             None, None, "diagonal", None, 0.0001, 0, 0);
        let n_found = round1.len();
        println!("Round 1 found {} eigenpairs within the window", n_found);
        let eigvecs: Vec<Vec<f64>> = round1.into_iter().map(|(_, v)| v).collect();

        let (_, _, occ_size, vir_size, _, _) = get_occupation_parameters(scf_data, 'N');

        // Clear BSE-specific integrals → fallback to regular RI for Rayleigh-Ritz
        scf_data.ri3fn_bse = None;
        scf_data.rimatr_bse = None;

        // Rayleigh-Ritz refinement with full integrals
        let (ritz_vals, ritz_vecs) = rayleigh_ritz_refine(scf_data, &qp_ctrl_triplet, &quasiparticle_energies,
                                           occ_size, vir_size, &eigvecs);
        let (ritz_vals, ritz_vecs) = rayleigh_ritz_refine(scf_data, &qp_ctrl_triplet, &quasiparticle_energies,
                                           occ_size, vir_size, &eigvecs);
        let emin = qp_ctrl.bse_eigenrange_min;
        let emax = qp_ctrl.bse_eigenrange_max;
        let (fil_vals, fil_vecs) = filter_eigenpairs(ritz_vals, ritz_vecs, emin, emax);
        print_ritz_eigenpairs(scf_data, &fil_vals, &fil_vecs, "triplet", occ_size, vir_size);
        fil_vals.into_iter().zip(fil_vecs.into_iter()).collect()
    } else {
        feast_solve_bse_spin(scf_data,&qp_ctrl_triplet,&quasiparticle_energies,
                             eigenrange_min,eigenrange_max,m_expected,max_feast_iter,tol_feast,None,
                             None, None, "diagonal", None, 0.0001, 0, 0)
    }
}

/// Internal helper: solve for a specific spin (set in qp_ctrl.bse_spin).
fn feast_solve_bse_spin(
    scf_data:&SCF,
    qp_ctrl:&QuasiParticle,
    quasiparticle_energies:&Vec<f64>,
    eigenrange_min:f64,
    eigenrange_max:f64,
    m_expected:usize,
    max_feast_iter:usize,
    tol_feast:f64,
    custom_init_vectors: Option<&Vec<Vec<f64>>>,
    precond_a_mul: Option<Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync>>,
    precond_b_mul: Option<Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync>>,
    precond_type: &str,
    precond_diag: Option<&Vec<f64>>,
    inner_gmres_tol: f64,
    inner_gmres_restart: usize,
    inner_gmres_max_iter: usize,
)->Vec<(f64,Vec<f64>)>{
    let (_,_,occ_size,vir_size,_,_)=get_occupation_parameters(scf_data,'N');
    let ks_energies:Vec<f64>=scf_data.eigenvalues[0].clone();
    let mut epsilon=ks_energies.clone();
    if qp_ctrl.bse_qp_polarization==true{
        epsilon=quasiparticle_energies.clone();
    }
    let inverse_dielectric=construct_inverse_dielectric(scf_data,&epsilon);

    if qp_ctrl.bse_tda==true{
        feast_solve_bse_tda(scf_data,qp_ctrl,&inverse_dielectric,occ_size,vir_size,
                            eigenrange_min,eigenrange_max,m_expected,max_feast_iter,tol_feast,
                            custom_init_vectors,
                            precond_a_mul, precond_b_mul, precond_type, precond_diag, inner_gmres_tol, inner_gmres_restart, inner_gmres_max_iter)
    }else{
        feast_solve_bse_nontda(scf_data,qp_ctrl,&inverse_dielectric,occ_size,vir_size,
                               eigenrange_min,eigenrange_max,m_expected,max_feast_iter,tol_feast,
                               custom_init_vectors,
                               precond_a_mul, precond_b_mul, precond_type, precond_diag, inner_gmres_tol, inner_gmres_restart, inner_gmres_max_iter)
    }
}

/// TDA branch for a single spin.
fn feast_solve_bse_tda(
    scf_data:&SCF,
    qp_ctrl:&QuasiParticle,
    inverse_dielectric:&MatrixFull<f64>,
    occ_size:usize,
    vir_size:usize,
    eigenrange_min:f64,
    eigenrange_max:f64,
    m_expected:usize,
    max_feast_iter:usize,
    tol_feast:f64,
    custom_init_vectors: Option<&Vec<Vec<f64>>>,
    precond_a_mul: Option<Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync>>,
    precond_b_mul: Option<Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync>>,
    precond_type: &str,
    precond_diag: Option<&Vec<f64>>,
    inner_gmres_tol: f64,
    inner_gmres_restart: usize,
    inner_gmres_max_iter: usize,
)->Vec<(f64,Vec<f64>)>{
    let num_auxbas=inverse_dielectric.size[0];
    let ri_ov=get_submatrix(scf_data,'O','V','N');
    let mut ri_vv=get_submatrix(scf_data,'V','V','N');
    let ri_oo=get_submatrix(scf_data,'O','O','N');

    let mut ri_oo_tilde:MatrixFull<f64>=MatrixFull::new(ri_oo.size,0.0);
    _dgemm_full(inverse_dielectric,'N',&ri_oo,'N',&mut ri_oo_tilde,1.0,0.0);
    ri_oo_tilde.reshape([num_auxbas*occ_size,occ_size]);
    ri_oo_tilde=ri_oo_tilde.transpose_and_drop();
    ri_oo_tilde.reshape([occ_size*num_auxbas,occ_size]);
    ri_vv.reshape([num_auxbas*vir_size,vir_size]);

    let feast_a_matvec=|z:&Vec<f64>|{
        matvec::a_block_matvec(scf_data,qp_ctrl,&ri_vv,&ri_ov,&ri_oo_tilde,z)
    };
    let feast_b_matvec=|z:&Vec<f64>|{
        z.clone()
    };

    // ── GMRES diagonal preconditioner for TDA ──
    // The transformed GMRES matrix is (z·I − A), diagonal ≈ z − D_j
    // where D_j = ε_a − ε_i are the quasi-particle energy gaps.
    let diag_a: Vec<f64> = {
        let energies: Vec<f64> = if qp_ctrl.bse_qp_polarization {
            scf_data.gwqp.0.clone()
        } else {
            scf_data.eigenvalues[0].clone()
        };
        construct_energy_diag_for_a(&energies, occ_size, vir_size)
    };

    let gmres_restart = qp_ctrl.bse_feast_gmres_restart;
    let gmres_max_iter = qp_ctrl.bse_feast_gmres_max_iter;
    let gmres_tol = qp_ctrl.bse_feast_cg_tol;
    feast(occ_size*vir_size,&feast_a_matvec,&feast_b_matvec,None,None,
          eigenrange_min,eigenrange_max,m_expected,max_feast_iter,tol_feast,
          gmres_restart,gmres_max_iter,gmres_tol,
          Some(&diag_a),
          &qp_ctrl.bse_feast_init_guess_type, Some(&diag_a),
          qp_ctrl.bse_feast_gaussian_width_factor,
          qp_ctrl.bse_feast_contour_rayon,
          custom_init_vectors,
          precond_a_mul, precond_b_mul, precond_type, precond_diag, inner_gmres_tol, inner_gmres_restart, inner_gmres_max_iter)
}

/// Non-TDA branch for a single spin.
fn feast_solve_bse_nontda(
    scf_data:&SCF,
    qp_ctrl:&QuasiParticle,
    inverse_dielectric:&MatrixFull<f64>,
    occ_size:usize,
    vir_size:usize,
    eigenrange_min:f64,
    eigenrange_max:f64,
    m_expected:usize,
    max_feast_iter:usize,
    tol_feast:f64,
    custom_init_vectors: Option<&Vec<Vec<f64>>>,
    precond_a_mul: Option<Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync>>,
    precond_b_mul: Option<Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync>>,
    precond_type: &str,
    precond_diag: Option<&Vec<f64>>,
    inner_gmres_tol: f64,
    inner_gmres_restart: usize,
    inner_gmres_max_iter: usize,
)->Vec<(f64,Vec<f64>)>{
    let num_auxbas=inverse_dielectric.size[0];
    let ri_oo=get_submatrix(scf_data,'O','O','N');

    let mut ri_oo_tilde:MatrixFull<f64>=MatrixFull::new(ri_oo.size,0.0);
    _dgemm_full(inverse_dielectric,'N',&ri_oo,'N',&mut ri_oo_tilde,1.0,0.0);
    drop(ri_oo);
    ri_oo_tilde.reshape([num_auxbas*occ_size,occ_size]);
    ri_oo_tilde=ri_oo_tilde.transpose_and_drop();
    ri_oo_tilde.reshape([occ_size*num_auxbas,occ_size]);

    let ri_ov=get_submatrix(scf_data,'O','V','N');
    let mut ri_vv=get_submatrix(scf_data,'V','V','N');
    ri_vv.reshape([num_auxbas*vir_size,vir_size]);

    let mut ri_ov_tilde:MatrixFull<f64>=MatrixFull::new(ri_ov.size,0.0);
    _dgemm_full(inverse_dielectric,'N',&ri_ov,'N',&mut ri_ov_tilde,1.0,0.0);
    ri_ov_tilde.reshape([num_auxbas*occ_size,vir_size]);

    let mut ri_ov_b=ri_ov.clone();
    ri_ov_b.reshape([num_auxbas*occ_size,vir_size]);
    let feast_a_matvec=|z:&Vec<f64>|->Vec<f64>{
        let a = matvec::a_block_matvec(scf_data,qp_ctrl,&ri_vv,&ri_ov,&ri_oo_tilde,z);
        let b = matvec::b_block_matvec(scf_data,qp_ctrl,&ri_ov,&ri_ov_b,&ri_ov_tilde,z);
        a.into_iter().zip(b.into_iter()).map(|(a,b)|a-b).collect()
    };
    let feast_b_matvec=|z:&Vec<f64>|->Vec<f64>{
        let apb_matvec=|p:&Vec<f64>|->Vec<f64>{
            let a = matvec::a_block_matvec(scf_data,qp_ctrl,&ri_vv,&ri_ov,&ri_oo_tilde,p);
            let b = matvec::b_block_matvec(scf_data,qp_ctrl,&ri_ov,&ri_ov_b,&ri_ov_tilde,p);
            a.into_iter().zip(b.into_iter()).map(|(a,b)|a+b).collect()
        };
        cg(&apb_matvec,z,qp_ctrl.bse_feast_cg_max_iter,qp_ctrl.bse_feast_cg_tol)
    };

    let gmres_restart = qp_ctrl.bse_feast_gmres_restart;
    let gmres_max_iter = qp_ctrl.bse_feast_gmres_max_iter;
    let gmres_tol = qp_ctrl.bse_feast_cg_tol;
    let gmres_a_mul=|z:&Vec<f64>|->Vec<f64>{
        let apb_q=feast_a_matvec(z);
        let a = matvec::a_block_matvec(scf_data,qp_ctrl,&ri_vv,&ri_ov,&ri_oo_tilde,&apb_q);
        let b = matvec::b_block_matvec(scf_data,qp_ctrl,&ri_ov,&ri_ov_b,&ri_ov_tilde,&apb_q);
        a.into_iter().zip(b.into_iter()).map(|(a,b)|a+b).collect()
    };
    let gmres_b_mul=|z:&Vec<f64>|{
        z.clone()
    };

    // ── GMRES diagonal preconditioner for non-TDA ──
    // The transformed GMRES matrix is (z·I − (A−B)(A+B)).
    // Approximating A_diag ≈ D_j (QP energy gaps) and B_diag ≈ 0 gives
    // (A−B)(A+B)_diag ≈ D_j².  So M₂_diag ≈ z − D_j².
    let energies: Vec<f64> = if qp_ctrl.bse_qp_polarization {
        scf_data.gwqp.0.clone()
    } else {
        scf_data.eigenvalues[0].clone()
    };
    let diag = construct_energy_diag_for_a(&energies, occ_size, vir_size);
    let diag_sq: Vec<f64> = diag.iter().map(|&d| d * d).collect();

    let eigenpairs_xpy=feast(occ_size*vir_size,&feast_a_matvec,&feast_b_matvec,Some(&gmres_a_mul),Some(&gmres_b_mul),
                             eigenrange_min,eigenrange_max,m_expected,max_feast_iter,tol_feast,
                             gmres_restart,gmres_max_iter,gmres_tol,
                             Some(&diag_sq),
                             &qp_ctrl.bse_feast_init_guess_type, Some(&diag),
                             qp_ctrl.bse_feast_gaussian_width_factor,
                             qp_ctrl.bse_feast_contour_rayon,
                             custom_init_vectors,
                             precond_a_mul, precond_b_mul, precond_type, precond_diag, inner_gmres_tol, inner_gmres_restart, inner_gmres_max_iter);
    eigenpairs_xpy.iter().map(|(omega2,xpy)|{
        let xmy=feast_a_matvec(xpy);
        (omega2.sqrt(),xmy.iter().zip(xpy.iter()).map(|(xmy_k,xpy_k)|(xmy_k/omega2.sqrt())+xpy_k).collect::<Vec<_>>())
    }).collect()
}

/// Solve both singlet and triplet BSE excitations using FEAST.
/// When `bse_feast_renormalized_doubles` is true, runs two rounds:
/// Round 1 with BSE-specific RI integrals (both spins), Round 2 with
/// regular integrals warm-started by Round 1 eigenvectors.
pub fn feast_solve_bse(scf_data:&mut SCF)->(Vec<(f64,Vec<f64>)>,Vec<(f64,Vec<f64>)>){
    let start=Instant::now();
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let eigenrange_min_orig = qp_ctrl.bse_eigenrange_min;
    let eigenrange_max_orig = qp_ctrl.bse_eigenrange_max;
    let eigenrange_min = eigenrange_min_orig * eigenrange_min_orig;
    let eigenrange_max = eigenrange_max_orig * eigenrange_max_orig;
    let m_expected=qp_ctrl.bse_m_expected;
    let max_feast_iter=qp_ctrl.bse_max_feast_iter;
    let tol_feast=qp_ctrl.bse_tol_feast;
    let quasiparticle_energies=scf_data.gwqp.0.clone();

    let (_,_,occ_size,vir_size,_,_)=get_occupation_parameters(scf_data,'N');
    let ks_energies:Vec<f64>=scf_data.eigenvalues[0].clone();
    let mut epsilon=ks_energies.clone();
    if qp_ctrl.bse_qp_polarization==true{
        epsilon=quasiparticle_energies.clone();
    }
    let inverse_dielectric=construct_inverse_dielectric(scf_data,&epsilon);

    if !qp_ctrl.bse_feast_renormalized_doubles {
        // ---- Original single-round flow ----
        let mut eigenpairs_singlet:Vec<(f64,Vec<f64>)>=Vec::new();
        let mut eigenpairs_triplet:Vec<(f64,Vec<f64>)>=Vec::new();

        if qp_ctrl.bse_tda==true{
            let mut qp_ctrl_s=qp_ctrl.clone();
            qp_ctrl_s.bse_spin=String::from("singlet");
            eigenpairs_singlet=feast_solve_bse_tda(
                scf_data,&qp_ctrl_s,&inverse_dielectric,occ_size,vir_size,
                eigenrange_min_orig,eigenrange_max_orig,m_expected,max_feast_iter,tol_feast,None,
                None, None, "diagonal", None, 0.0001, 0, 0);
            let one_feast_time=start.elapsed();
            println!("Singlets (FEAST) calculation took {:?}",one_feast_time);

            let mut qp_ctrl_t=qp_ctrl.clone();
            qp_ctrl_t.bse_spin=String::from("triplet");
            eigenpairs_triplet=feast_solve_bse_tda(
                scf_data,&qp_ctrl_t,&inverse_dielectric,occ_size,vir_size,
                eigenrange_min_orig,eigenrange_max_orig,m_expected,max_feast_iter,tol_feast,None,
                None, None, "diagonal", None, 0.0001, 0, 0);
            println!("Triplets (FEAST) calculation took {:?}",start.elapsed()-one_feast_time);
        }else{
            let mut qp_ctrl_s=qp_ctrl.clone();
            qp_ctrl_s.bse_spin=String::from("singlet");
            eigenpairs_singlet=feast_solve_bse_nontda(
                scf_data,&qp_ctrl_s,&inverse_dielectric,occ_size,vir_size,
                eigenrange_min,eigenrange_max,m_expected,max_feast_iter,tol_feast,None,
                None, None, "diagonal", None, 0.0001, 0, 0);
            let one_feast_time=start.elapsed();
            println!("Singlets (FEAST) calculation took {:?}",one_feast_time);

            let mut qp_ctrl_t=qp_ctrl.clone();
            qp_ctrl_t.bse_spin=String::from("triplet");
            eigenpairs_triplet=feast_solve_bse_nontda(
                scf_data,&qp_ctrl_t,&inverse_dielectric,occ_size,vir_size,
                eigenrange_min,eigenrange_max,m_expected,max_feast_iter,tol_feast,None,
                None, None, "diagonal", None, 0.0001, 0, 0);
            println!("Triplets (FEAST) calculation took {:?}",start.elapsed()-one_feast_time);
        }

        return (eigenpairs_singlet,eigenpairs_triplet);
    }

    // ---- Renormalized doubles flow ----
    // Round 1: s-only FEAST → Rayleigh-Ritz with full integrals → output result
    let ew = qp_ctrl.bse_renormalized_doubles_extra_width;
    let (r1_min_tda, r1_max_tda) = (eigenrange_min_orig - ew, eigenrange_max_orig + ew);
    let r1_min_nontda = (eigenrange_min_orig - ew).max(0.0);
    let r1_min_nontda_sq = r1_min_nontda * r1_min_nontda;
    let r1_max_nontda_sq = (eigenrange_max_orig + ew) * (eigenrange_max_orig + ew);
    println!("--- Round 1: BSE-specific RI integrals (s-only) for both spins, original range widened by {} ---", ew);

    let round1_singlet: Vec<(f64,Vec<f64>)>;
    let round1_triplet: Vec<(f64,Vec<f64>)>;

    if qp_ctrl.bse_tda==true{
        let mut qp_ctrl_s=qp_ctrl.clone();
        qp_ctrl_s.bse_spin=String::from("singlet");
        round1_singlet=feast_solve_bse_tda(
            scf_data,&qp_ctrl_s,&inverse_dielectric,occ_size,vir_size,
            r1_min_tda,r1_max_tda,m_expected,max_feast_iter,tol_feast,None,
            None, None, "diagonal", None, 0.0001, 0, 0);
        let one_feast_time=start.elapsed();
        println!("Round 1 Singlets took {:?}",one_feast_time);

        let mut qp_ctrl_t=qp_ctrl.clone();
        qp_ctrl_t.bse_spin=String::from("triplet");
        round1_triplet=feast_solve_bse_tda(
            scf_data,&qp_ctrl_t,&inverse_dielectric,occ_size,vir_size,
            r1_min_tda,r1_max_tda,m_expected,max_feast_iter,tol_feast,None,
            None, None, "diagonal", None, 0.0001, 0, 0);
        println!("Round 1 Triplets took {:?}",start.elapsed()-one_feast_time);
    }else{
        let mut qp_ctrl_s=qp_ctrl.clone();
        qp_ctrl_s.bse_spin=String::from("singlet");
        round1_singlet=feast_solve_bse_nontda(
            scf_data,&qp_ctrl_s,&inverse_dielectric,occ_size,vir_size,
            r1_min_nontda_sq,r1_max_nontda_sq,m_expected,max_feast_iter,tol_feast,None,
            None, None, "diagonal", None, 0.0001, 0, 0);
        let one_feast_time=start.elapsed();
        println!("Round 1 Singlets took {:?}",one_feast_time);

        let mut qp_ctrl_t=qp_ctrl.clone();
        qp_ctrl_t.bse_spin=String::from("triplet");
        round1_triplet=feast_solve_bse_nontda(
            scf_data,&qp_ctrl_t,&inverse_dielectric,occ_size,vir_size,
            r1_min_nontda_sq,r1_max_nontda_sq,m_expected,max_feast_iter,tol_feast,None,
            None, None, "diagonal", None, 0.0001, 0, 0);
        println!("Round 1 Triplets took {:?}",start.elapsed()-one_feast_time);
    }

    let n_s = round1_singlet.len();
    let n_t = round1_triplet.len();
    println!("Round 1 found {} singlet + {} triplet eigenpairs", n_s, n_t);

    let eigvecs_s: Vec<Vec<f64>> = round1_singlet.into_iter().map(|(_, v)| v).collect();
    let eigvecs_t: Vec<Vec<f64>> = round1_triplet.into_iter().map(|(_, v)| v).collect();

    // Clear BSE integrals → fallback to regular RI for Rayleigh-Ritz
    scf_data.ri3fn_bse = None;
    scf_data.rimatr_bse = None;

    // Rayleigh-Ritz refinement with full integrals
    let mut qp_ctrl_rr_s = qp_ctrl.clone();
    qp_ctrl_rr_s.bse_spin = String::from("singlet");
    let (ritz_vals_s, ritz_vecs_s) = rayleigh_ritz_refine(scf_data, &qp_ctrl_rr_s, &quasiparticle_energies,
                                         occ_size, vir_size, &eigvecs_s);
    let (fil_vals_s, fil_vecs_s) = filter_eigenpairs(ritz_vals_s, ritz_vecs_s,
                                                      eigenrange_min_orig, eigenrange_max_orig);
    print_ritz_eigenpairs(scf_data, &fil_vals_s, &fil_vecs_s, "singlet", occ_size, vir_size);
    let mut qp_ctrl_rr_t = qp_ctrl.clone();
    qp_ctrl_rr_t.bse_spin = String::from("triplet");
    let (ritz_vals_t, ritz_vecs_t) = rayleigh_ritz_refine(scf_data, &qp_ctrl_rr_t, &quasiparticle_energies,
                                         occ_size, vir_size, &eigvecs_t);
    let (fil_vals_t, fil_vecs_t) = filter_eigenpairs(ritz_vals_t, ritz_vecs_t,
                                                      eigenrange_min_orig, eigenrange_max_orig);
    print_ritz_eigenpairs(scf_data, &fil_vals_t, &fil_vecs_t, "triplet", occ_size, vir_size);

    let eigenpairs_singlet = fil_vals_s.into_iter().zip(fil_vecs_s.into_iter()).collect();
    let eigenpairs_triplet = fil_vals_t.into_iter().zip(fil_vecs_t.into_iter()).collect();
    (eigenpairs_singlet,eigenpairs_triplet)
}
pub fn test_preconditioner_working(scf_data: &mut SCF) {
    println!("\n========== Preconditioner Validation Test ==========");

    let (_, _, occ_size, vir_size, _, _) = get_occupation_parameters(scf_data, 'N');
    let n = occ_size * vir_size;
    println!("System dimension: n={} (occ={}, vir={})", n, occ_size, vir_size);

    let qp_ctrl = scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let c = (qp_ctrl.bse_eigenrange_min + qp_ctrl.bse_eigenrange_max) / 2.0;
    let radius = (qp_ctrl.bse_eigenrange_max - qp_ctrl.bse_eigenrange_min) / 2.0;

    // QP energies for diagonals
    let energies: Vec<f64> = if qp_ctrl.bse_qp_polarization {
        scf_data.gwqp.0.clone()
    } else {
        scf_data.eigenvalues[0].clone()
    };
    let diag_a: Vec<f64> = construct_energy_diag_for_a(&energies, occ_size, vir_size);

    let ks_energies: Vec<f64> = scf_data.eigenvalues[0].clone();
    let epsilon: Vec<f64> = if qp_ctrl.bse_qp_polarization {
        scf_data.gwqp.0.clone()
    } else {
        ks_energies.clone()
    };

    // ---- Build s-only matvec and precond data (BSE integrals currently active) ----
    println!("Building s-only matvec (BSE-specific auxbas)...");
    let inverse_dielectric_s = construct_inverse_dielectric(scf_data, &epsilon);
    let num_aux_s = scf_data.num_auxbas_bse.unwrap_or(scf_data.mol.num_auxbas);
    println!("  s-only num_auxbas: {}", num_aux_s);

    let ri_ov_s = get_submatrix(scf_data, 'O', 'V', 'N');
    let mut ri_vv_s = get_submatrix(scf_data, 'V', 'V', 'N');
    let ri_oo_s = get_submatrix(scf_data, 'O', 'O', 'N');
    let mut ri_oo_tilde_s = MatrixFull::new(ri_oo_s.size, 0.0);
    _dgemm_full(&inverse_dielectric_s, 'N', &ri_oo_s, 'N', &mut ri_oo_tilde_s, 1.0, 0.0);
    ri_oo_tilde_s.reshape([num_aux_s * occ_size, occ_size]);
    ri_oo_tilde_s = ri_oo_tilde_s.transpose_and_drop();
    ri_oo_tilde_s.reshape([occ_size * num_aux_s, occ_size]);
    ri_vv_s.reshape([num_aux_s * vir_size, vir_size]);

    let occ_v = occ_size;
    let vir_v = vir_size;
    let energies_s = energies.clone();
    let qpc_s = qp_ctrl.clone();
    let a_mul_s: Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync> = Arc::new(
        move |z_vec: &Vec<f64>| -> Vec<f64> {
            let xlet = if qpc_s.bse_spin == "triplet" { 'T' }
                       else if qpc_s.bse_spin == "singlet" { 'S' } else { 'R' };
            let mut result = matvec::diagonal_contribution_standalone(z_vec, &energies_s, occ_v, vir_v);
            let w = matvec::w_contribution_a_block_dgemm_standalone(
                &ri_vv_s, z_vec, &ri_oo_tilde_s, &qpc_s, occ_v, vir_v);
            result = w.iter().zip(result.iter()).map(|(wi, zi)| -wi + zi).collect();
            if xlet == 'S' {
                let v = matvec::coulomb_contribution(&ri_ov_s, z_vec);
                result = v.iter().zip(result.iter()).map(|(vi, zi)| 2.0 * vi + zi).collect();
            } else if xlet == 'R' {
                let v = matvec::coulomb_contribution(&ri_ov_s, z_vec);
                result = v.iter().zip(result.iter()).map(|(vi, zi)| vi + zi).collect();
            }
            result
        },
    );
    let b_mul_s: Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync> =
        Arc::new(|z_vec: &Vec<f64>| z_vec.clone());

    // ---- Switch to regular auxbas for full matvec ----
    let tmp_bse_fn = scf_data.ri3fn_bse.take();
    let tmp_bse_rm = scf_data.rimatr_bse.take();

    println!("Building full matvec (regular auxbas)...");
    let inverse_dielectric_full = construct_inverse_dielectric(scf_data, &epsilon);
    let num_aux_full = scf_data.mol.num_auxbas;
    println!("  full num_auxbas: {}", num_aux_full);

    let ri_ov_f = get_submatrix(scf_data, 'O', 'V', 'N');
    let mut ri_vv_f = get_submatrix(scf_data, 'V', 'V', 'N');
    let ri_oo_f = get_submatrix(scf_data, 'O', 'O', 'N');
    let mut ri_oo_tilde_f = MatrixFull::new(ri_oo_f.size, 0.0);
    _dgemm_full(&inverse_dielectric_full, 'N', &ri_oo_f, 'N', &mut ri_oo_tilde_f, 1.0, 0.0);
    ri_oo_tilde_f.reshape([num_aux_full * occ_size, occ_size]);
    ri_oo_tilde_f = ri_oo_tilde_f.transpose_and_drop();
    ri_oo_tilde_f.reshape([occ_size * num_aux_full, occ_size]);
    ri_vv_f.reshape([num_aux_full * vir_size, vir_size]);

    let qpc_f = qp_ctrl.clone();
    let a_mul_full: Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync> = Arc::new(
        move |z_vec: &Vec<f64>| -> Vec<f64> {
            let xlet = if qpc_f.bse_spin == "triplet" { 'T' }
                       else if qpc_f.bse_spin == "singlet" { 'S' } else { 'R' };
            let mut result = matvec::diagonal_contribution_standalone(z_vec, &energies, occ_v, vir_v);
            let w = matvec::w_contribution_a_block_dgemm_standalone(
                &ri_vv_f, z_vec, &ri_oo_tilde_f, &qpc_f, occ_v, vir_v);
            result = w.iter().zip(result.iter()).map(|(wi, zi)| -wi + zi).collect();
            if xlet == 'S' {
                let v = matvec::coulomb_contribution(&ri_ov_f, z_vec);
                result = v.iter().zip(result.iter()).map(|(vi, zi)| 2.0 * vi + zi).collect();
            } else if xlet == 'R' {
                let v = matvec::coulomb_contribution(&ri_ov_f, z_vec);
                result = v.iter().zip(result.iter()).map(|(vi, zi)| vi + zi).collect();
            }
            result
        },
    );

    let gmres_restart = qp_ctrl.bse_feast_gmres_restart.min(1000);
    let gmres_max_iter = qp_ctrl.bse_feast_gmres_max_iter.min(1000);
    let gmres_tol = 1e-7;

    let mut rng = rand::thread_rng();
    println!("\n--- Looping over 8 quadrature points (c={:.6}, r={:.6}) ---", c, radius);

    let mut all_ok = true;
    let mut total_mv_diag = 0usize;
    let mut total_mv_ig = 0usize;
    let mut total_t_diag = std::time::Duration::ZERO;
    let mut total_t_ig = std::time::Duration::ZERO;

    for qp_idx in 0..GL_POINTS.len() {
        let (x_e, _w_e) = GL_POINTS[qp_idx];
        let θ = -std::f64::consts::PI * (x_e - 1.0) / 2.0;
        let z_re = c + radius * θ.cos();
        let z_im = radius * θ.sin();

        println!("  qp={}: z = {:.6} + i·{:.6}", qp_idx, z_re, z_im);

        // Random RHS
        let mut rhs = vec![0.0; 2 * n];
        for i in 0..2 * n {
            rhs[i] = rng.gen::<f64>() * 2.0 - 1.0;
        }

        // Build full M₂ matvec for this quadrature point
        let a_full = a_mul_full.clone();
        let m2_full = move |v: &Vec<f64>| -> Vec<f64> {
            let vr: Vec<f64> = v[0..n].to_vec();
            let vi: Vec<f64> = v[n..2 * n].to_vec();
            let a_vr = a_full(&vr);
            let a_vi = a_full(&vi);
            let mut result = vec![0.0; 2 * n];
            for i in 0..n {
                result[i]     = z_re * vr[i] - a_vr[i] - z_im * vi[i];
                result[n + i] = z_im * vr[i] + z_re * vi[i] - a_vi[i];
            }
            result
        };

        // Diagonal preconditioner
        let mut re_part = vec![0.0; n];
        let mut im_part = vec![0.0; n];
        for i in 0..n {
            re_part[i] = z_re - diag_a[i];
            im_part[i] = -z_im;
        }
        let diag_prec = GmresPrecond::Diag(BlockDiagPrecond::from_re_im(&re_part, &im_part));

        // Inner GMRES preconditioner
        let inner_prec = InnerGmresPrecond {
            precond_a: a_mul_s.clone(),
            precond_b: b_mul_s.clone(),
            z_re,
            z_im,
            inner_restart: qp_ctrl.bse_feast_inner_gmres_restart,
            inner_max_iter: qp_ctrl.bse_feast_inner_gmres_max_iter,
            inner_tol: qp_ctrl.bse_feast_inner_gmres_tol,
            diag: diag_a.clone(),
        };
        let igmres_prec = GmresPrecond::InnerGmres(inner_prec);

        // Solve with diagonal preconditioner
        let t0 = Instant::now();
        let (sol_diag, mv_diag) = gmres(
            &m2_full, &rhs, gmres_restart, gmres_max_iter, gmres_tol, false, Some(&diag_prec),
        );
        let t_diag = t0.elapsed();

        // Solve with inner-GMRES preconditioner
        let t0 = Instant::now();
        let (sol_ig, mv_ig) = gmres(
            &m2_full, &rhs, gmres_restart, gmres_max_iter, gmres_tol, false, Some(&igmres_prec),
        );
        let t_ig = t0.elapsed();

        // Verify solutions agree
        let diff: f64 = sol_diag.iter().zip(sol_ig.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>().sqrt();
        let norm_rhs: f64 = rhs.iter().map(|x| x * x).sum::<f64>().sqrt();
        let rel_diff = diff / norm_rhs.max(1e-30);
        let ok = rel_diff < 1e-6;

        println!("    diag: {:>5} matvecs, {:>8.2?}  |  igmres: {:>5} matvecs, {:>8.2?}  |  rel_diff={:.2e}  {}",
                 mv_diag, t_diag, mv_ig, t_ig, rel_diff,
                 if ok { "✓" } else { "✗" });

        if !ok { all_ok = false; }
        total_mv_diag += mv_diag;
        total_mv_ig += mv_ig;
        total_t_diag += t_diag;
        total_t_ig += t_ig;
    }

    // ── Summary ──
    println!("\n--- Summary over 8 quadrature points ---");
    println!("  Precond type      | Total GMRES matvecs | Total wall time");
    println!("  -------------------+---------------------+----------------");
    println!("  diagonal           | {:>19} | {:>13.2?}", total_mv_diag, total_t_diag);
    println!("  inner-GMRES        | {:>19} | {:>13.2?}", total_mv_ig, total_t_ig);
    if total_mv_diag > 0 {
        println!("  Speedup (outer iters): {:.2}x", total_mv_diag as f64 / total_mv_ig as f64);
    }
    if total_t_diag.as_secs_f64() > 0.0 {
        println!("  Speedup (wall time):   {:.2}x", total_t_diag.as_secs_f64() / total_t_ig.as_secs_f64().max(1e-9));
    }
    if all_ok {
        println!("  ✓ All 8 quadrature points: solutions AGREE");
    } else {
        println!("  ✗ Some points DISAGREE!");
    }

    // Restore BSE integrals
    scf_data.ri3fn_bse = tmp_bse_fn;
    scf_data.rimatr_bse = tmp_bse_rm;

    println!("========== Preconditioner Validation Done ==========\n");
}