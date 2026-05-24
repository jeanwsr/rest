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
use crate::ri_bse::{get_submatrix,construct_inverse_dielectric,construct_energy_diag_for_a};
use crate::ri_gw::{get_occupation_parameters};
use crate::ri_bse::matvec;
use crate::ctrl_io::quasiparticle_methods::QuasiParticle;
use std::time::Instant;

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
    precondition: Option<&BlockDiagPrecond>,
) -> Vec<f64> {
    let n = b.len();

    // Preconditioned RHS norm for tolerance
    let b_norm: f64 = if let Some(prec) = precondition {
        prec.norm(b)
    } else {
        b.iter().map(|bi| bi * bi).sum::<f64>().sqrt()
    };
    let tol_abs = tol * b_norm.max(1e-30);

    // Helper: apply block-diagonal preconditioner to a vector
    let apply_prec = |w: &Vec<f64>, prec: &BlockDiagPrecond| -> Vec<f64> {
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
        let mut converged = false;

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
                converged = true;
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

            // Check convergence
            if c[j + 1].abs() <= tol_abs {
                converged = true;
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

        if !converged {
            let ax = a_mul(&x);
            total_matvecs += 1;
            let r_restart: Vec<f64> = if let Some(prec) = precondition {
                apply_prec(
                    &(0..n).map(|i| b[i] - ax[i]).collect(),
                    prec,
                )
            } else {
                (0..n).map(|i| b[i] - ax[i]).collect()
            };
            r_norm = r_restart.iter().map(|ri| ri * ri).sum::<f64>().sqrt();
            n_restarts += 1;
            if verbose && n_restarts % 10 == 0 {
                eprintln!("  GMRES restart {}: {} matvecs, rel.res = {:.2e}",
                    n_restarts, total_matvecs, r_norm / b_norm);
            }
        } else {
            r_norm = c[inner_k].abs();
            if verbose {
                eprintln!("  GMRES converged: {} matvecs, {} restart(s), residual {:.2e}",
                    total_matvecs, n_restarts + 1, r_norm / b_norm);
            }
            break;
        }
    }

    if r_norm > tol_abs && verbose {
        eprintln!("  WARNING: GMRES did not converge after {} matvecs, rel.res = {:.2e}",
            total_matvecs, r_norm / b_norm);
    }

    x
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

/// Solve (z·B − A) · X = RHS  for all columns of RHS using GMRES.
///
/// The complex system (α+iβ)·B − A is embedded as a 2N×2N real system
/// and solved column-by-column with restarted GMRES.  Only a_mul / b_mul
/// closures are needed — no explicit matrix is constructed.
///
/// rhs_matrix is n×m0 (real).  Returns the complex solution as a pair
/// of real matrices (X_re, X_im), each n×m0.
///
/// diag_a (optional): diagonal of the Ã operator in the transformed system
///   (z·I − Ã)·X = Y.  For TDA: diag_a = D (QP energy gaps).
///   For non-TDA: diag_a = D² (QP energy gaps squared).
fn solve_complex_iterative(
    a_mul: &impl Fn(&Vec<f64>) -> Vec<f64>,
    b_mul: &impl Fn(&Vec<f64>) -> Vec<f64>,
    gmres_a_mul: &dyn Fn(&Vec<f64>) -> Vec<f64>,
    gmres_b_mul: &dyn Fn(&Vec<f64>) -> Vec<f64>,
    n: usize,
    alpha: f64,
    beta: f64,
    rhs_matrix: &MatrixFull<f64>,
    gmres_restart: usize,
    gmres_max_iter: usize,
    gmres_tol: f64,
    diag_a: Option<&Vec<f64>>,
) -> (MatrixFull<f64>, MatrixFull<f64>) {
    let m0 = rhs_matrix.size()[1];

    // Build M₂ block-diagonal preconditioner
    // The 2×2 block for each complex index i is:
    //   [α − Ã[i]    −β]
    //   [  β       α − Ã[i]]
    // We store this as [re, -im, im, re] = [α-d[i], -β, β, α-d[i]]
    let precond = diag_a.map(|d| {
        let mut re_part = vec![0.0; n];
        let mut im_part = vec![0.0; n];
        for i in 0..n {
            re_part[i] = alpha - d[i];
            im_part[i] = -beta;  // −β from the upper-right block
        }
        BlockDiagPrecond::from_re_im(&re_part, &im_part)
    });

    let mut x_re = MatrixFull::new([n, m0], 0.0);
    let mut x_im = MatrixFull::new([n, m0], 0.0);

    for j in 0..m0 {
        // Verbose only for first RHS column to avoid noise
        let verbose = j == 0;

        // Extract RHS column br = B·Y_[:, j]
        let br: Vec<f64> = (0..n).map(|i| rhs_matrix[[i, j]]).collect();

        // ── Build 2N RHS: [br; 0] ──
        let mut b_2n = vec![0.0; 2 * n];
        for i in 0..n {
            b_2n[i] = br[i];
        }

        // ── Build M₂ matvec closure (uses gmres_a/b_mul for the linear solver) ──
        // M₂[vr; vi] = [α·B(vr) − A(vr) − β·B(vi);
        //               β·B(vr) + α·B(vi) − A(vi)]
        let m2_matvec = |v: &Vec<f64>| -> Vec<f64> {
            let vr: Vec<f64> = v[0..n].to_vec();
            let vi: Vec<f64> = v[n..2 * n].to_vec();

            let b_vr = gmres_b_mul(&vr);
            let b_vi = gmres_b_mul(&vi);
            let a_vr = gmres_a_mul(&vr);
            let a_vi = gmres_a_mul(&vi);

            let mut result = vec![0.0; 2 * n];
            for i in 0..n {
                result[i]       = alpha * b_vr[i] - a_vr[i] - beta * b_vi[i];
                result[n + i]   = beta * b_vr[i] + alpha * b_vi[i] - a_vi[i];
            }
            result
        };

        // ── Solve M₂ · x = b using GMRES ──
        let x_2n = gmres(m2_matvec, &b_2n, gmres_restart, gmres_max_iter,
                         gmres_tol, verbose, precond.as_ref());

        // ── Split into real and imaginary parts ──
        for i in 0..n {
            x_re[[i, j]] = x_2n[i];
            x_im[[i, j]] = x_2n[n + i];
        }
    }

    (x_re, x_im)
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
pub fn feast(
    n: usize,
    a_mul: &impl Fn(&Vec<f64>) -> Vec<f64>,
    b_mul: &impl Fn(&Vec<f64>) -> Vec<f64>,
    gmres_a_mul: Option<&(dyn Fn(&Vec<f64>) -> Vec<f64>)>,
    gmres_b_mul: Option<&(dyn Fn(&Vec<f64>) -> Vec<f64>)>,
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
) -> Vec<(f64, Vec<f64>)> {
    // ---- Step 0: parameters ------------------------------------------------
    let c = (λ_max + λ_min) / 2.0; // centre of the contour
    let r = (λ_max - λ_min) / 2.0; // radius
    let m0 = std::cmp::max(2 * m_expected, m_expected + 10); // subspace size

    // Resolve GMRES matvec closures: use specified or fall back to a_mul/b_mul
    let default_a: &dyn Fn(&Vec<f64>) -> Vec<f64> = a_mul;
    let default_b: &dyn Fn(&Vec<f64>) -> Vec<f64> = b_mul;
    let gmres_a: &dyn Fn(&Vec<f64>) -> Vec<f64> = gmres_a_mul.unwrap_or(default_a);
    let gmres_b: &dyn Fn(&Vec<f64>) -> Vec<f64> = gmres_b_mul.unwrap_or(default_b);

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
    if init_guess_type == "gaussian" {
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
        let mut q = MatrixFull::new([n, m0], 0.0);

        for qp in &quad {
            // Solve (Ze·B − A) · Q_temp = B·Y   for all M0 columns at once
            // using the iterative GMRES-based 2N solver.
            let (qt_re, qt_im) = solve_complex_iterative(
                a_mul, b_mul, gmres_a, gmres_b, n, qp.z_re, qp.z_im, &by,
                gmres_restart, gmres_max_iter, gmres_tol, diag_a,
            );

            // Accumulate:  Q += w · (r·cosθ · qt_re  −  r·sinθ · qt_im)
            // where w = −0.5 · w_e   (already stored in qp.w)
            let fac_re = qp.w * r * qp.cosθ;
            let fac_im = qp.w * r * qp.sinθ;
            for j in 0..m0 {
                for i in 0..n {
                    q[[i, j]] += fac_re * qt_re[[i, j]] - fac_im * qt_im[[i, j]];
                }
            }
        }

        // ── Step 2a': Subspace orthogonalization and compression ──
        // Purify Q by removing near-linear-dependent directions:
        //   1. Compute Gram matrix S = Q^T·Q
        //   2. Eigendecompose S = V·Λ·V^T
        //   3. Keep eigenvectors with eigenvalue > 1e-4
        //   4. Form orthonormal Q_tilde = Q·V_selected·diag(1/√λ)
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
            println!("FEAST iter {}: subspace compressed from {} to {}", iter, m0, m_eff);
        }

        // ── Step 2b: Rayleigh–Ritz — form reduced matrices ──
        //   A_Q = Q^T · A · Q      (m_eff × m_eff)
        //   B_Q = Q^T · B · Q      (m_eff × m_eff)
        //
        // Compute A·Q and B·Q one column at a time using closures.
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

        // ── Pad subspace results back to m0 columns for next iteration ──
        let (lambda, x_new) = if m_eff < m0 {
            let mut lambda_padded = lambda;
            lambda_padded.resize(m0, 0.0);
            let mut x_new_padded = MatrixFull::new([n, m0], 0.0);
            for j in 0..m_eff {
                for i in 0..n {
                    x_new_padded[[i, j]] = x_new[[i, j]];
                }
            }
            (lambda_padded, x_new_padded)
        } else {
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
        for j in 0..m0 {
            let xj: Vec<f64> = (0..n).map(|i| x_new[[i, j]]).collect();
            let yj = b_mul(&xj);
            for i in 0..n {
                y[[i, j]] = yj[i];
            }
        }
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

/// FEAST solver for singlet BSE excitations (handles both TDA and non-TDA).
pub fn feast_solve_bse_singlet(scf_data:&SCF)->Vec<(f64,Vec<f64>)>{
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let eigenrange_min=qp_ctrl.bse_eigenrange_min*qp_ctrl.bse_eigenrange_min;
    let eigenrange_max=qp_ctrl.bse_eigenrange_max*qp_ctrl.bse_eigenrange_max;
    let m_expected=qp_ctrl.bse_m_expected;
    let max_feast_iter=qp_ctrl.bse_max_feast_iter;
    let tol_feast=qp_ctrl.bse_tol_feast;
    let quasiparticle_energies=scf_data.gwqp.0.clone();
    let mut qp_ctrl_singlet=qp_ctrl.clone();
    qp_ctrl_singlet.bse_spin=String::from("singlet");
    feast_solve_bse_spin(scf_data,&qp_ctrl_singlet,&quasiparticle_energies,
                         eigenrange_min,eigenrange_max,m_expected,max_feast_iter,tol_feast)
}

/// FEAST solver for triplet BSE excitations (handles both TDA and non-TDA).
pub fn feast_solve_bse_triplet(scf_data:&SCF)->Vec<(f64,Vec<f64>)>{
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let eigenrange_min=qp_ctrl.bse_eigenrange_min*qp_ctrl.bse_eigenrange_min;
    let eigenrange_max=qp_ctrl.bse_eigenrange_max*qp_ctrl.bse_eigenrange_max;
    let m_expected=qp_ctrl.bse_m_expected;
    let max_feast_iter=qp_ctrl.bse_max_feast_iter;
    let tol_feast=qp_ctrl.bse_tol_feast;
    let quasiparticle_energies=scf_data.gwqp.0.clone();
    let mut qp_ctrl_triplet=qp_ctrl.clone();
    qp_ctrl_triplet.bse_spin=String::from("triplet");
    feast_solve_bse_spin(scf_data,&qp_ctrl_triplet,&quasiparticle_energies,
                         eigenrange_min,eigenrange_max,m_expected,max_feast_iter,tol_feast)
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
                            eigenrange_min,eigenrange_max,m_expected,max_feast_iter,tol_feast)
    }else{
        feast_solve_bse_nontda(scf_data,qp_ctrl,&inverse_dielectric,occ_size,vir_size,
                               eigenrange_min,eigenrange_max,m_expected,max_feast_iter,tol_feast)
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
          qp_ctrl.bse_feast_gaussian_width_factor)
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
                             qp_ctrl.bse_feast_gaussian_width_factor);
    eigenpairs_xpy.iter().map(|(omega2,xpy)|{
        let xmy=feast_a_matvec(xpy);
        (omega2.sqrt(),xmy.iter().zip(xpy.iter()).map(|(xmy_k,xpy_k)|(xmy_k/omega2.sqrt())+xpy_k).collect::<Vec<_>>())
    }).collect()
}

/// Solve both singlet and triplet BSE excitations using FEAST.
pub fn feast_solve_bse(scf_data:&SCF)->(Vec<(f64,Vec<f64>)>,Vec<(f64,Vec<f64>)>){
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

    let mut eigenpairs_singlet:Vec<(f64,Vec<f64>)>=Vec::new();
    let mut eigenpairs_triplet:Vec<(f64,Vec<f64>)>=Vec::new();

    if qp_ctrl.bse_tda==true{
        // ---- TDA branch: singlet ----
        let mut qp_ctrl_s=qp_ctrl.clone();
        qp_ctrl_s.bse_spin=String::from("singlet");
        eigenpairs_singlet=feast_solve_bse_tda(
            scf_data,&qp_ctrl_s,&inverse_dielectric,occ_size,vir_size,
            eigenrange_min_orig,eigenrange_max_orig,m_expected,max_feast_iter,tol_feast);
        let one_feast_time=start.elapsed();
        println!("Singlets (FEAST) calculation took {:?}",one_feast_time);

        // ---- TDA branch: triplet ----
        let mut qp_ctrl_t=qp_ctrl.clone();
        qp_ctrl_t.bse_spin=String::from("triplet");
        eigenpairs_triplet=feast_solve_bse_tda(
            scf_data,&qp_ctrl_t,&inverse_dielectric,occ_size,vir_size,
            eigenrange_min_orig,eigenrange_max_orig,m_expected,max_feast_iter,tol_feast);
        println!("Triplets (FEAST) calculation took {:?}",start.elapsed()-one_feast_time);
    }else{
        // ---- non-TDA branch: singlet ----
        let mut qp_ctrl_s=qp_ctrl.clone();
        qp_ctrl_s.bse_spin=String::from("singlet");
        eigenpairs_singlet=feast_solve_bse_nontda(
            scf_data,&qp_ctrl_s,&inverse_dielectric,occ_size,vir_size,
            eigenrange_min,eigenrange_max,m_expected,max_feast_iter,tol_feast);
        let one_feast_time=start.elapsed();
        println!("Singlets (FEAST) calculation took {:?}",one_feast_time);

        // ---- non-TDA branch: triplet ----
        let mut qp_ctrl_t=qp_ctrl.clone();
        qp_ctrl_t.bse_spin=String::from("triplet");
        eigenpairs_triplet=feast_solve_bse_nontda(
            scf_data,&qp_ctrl_t,&inverse_dielectric,occ_size,vir_size,
            eigenrange_min,eigenrange_max,m_expected,max_feast_iter,tol_feast);
        println!("Triplets (FEAST) calculation took {:?}",start.elapsed()-one_feast_time);
    }

    (eigenpairs_singlet,eigenpairs_triplet)
}