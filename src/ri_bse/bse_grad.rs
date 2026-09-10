//! Analytic nuclear gradients of static BSE excitation energies built on the
//! low-rank CD-G0W0 quasiparticle energies of [`crate::ri_gw::gw_grad`].
//!
//! Port of the validated PySCF implementation
//! `pyscf/gw/bse_grad_optimized.py` to REST.  The reference definition is
//!
//!     d Omega_S / dR = sum_p c_p^(S) dE_p/dR + d Omega_kernel^(S)/dR
//!
//! where `c_p` are the BSE amplitude weights (X^2 + Y^2 over the occupied and
//! virtual blocks) times the quasiparticle renormalisation factors Z_p, and
//! the kernel derivative is the derivative of the fixed-amplitude BSE
//! interaction kernel, pulled back onto (q, J, screening energies).
//!
//! * The static screening is `W(0)` in the raw RI representation:
//!   `D0 = J - Q0`, `Q0 = sum_ia q_ia (-4/de_ia) q_ia^T` (REST's real-axis
//!   response at omega = 0, matching `ri_bse::construct_inverse_dielectric`).
//! * `A = diag(gap) + kappa*v_bare - W_A`, `B = kappa*v_bare - W_B` with
//!   `kappa = 2` (singlet) / `0` (triplet), matching REST's
//!   `construct_submat_a/b` and PySCF's conventions.
//! * `ScreeningEnergy::Reference` builds the screening from the mean-field
//!   orbital energies (REST `bse_qp_polarization = false`, the default);
//!   `ScreeningEnergy::Qp` chains the screening-energy covector through the
//!   relaxed QP gradients (PySCF `screening_energy='qp'`).
//! * Dense and Davidson solvers produce the same (Omega, X, Y) and share the
//!   gradient assembly.  Amplitudes are metric-normalised
//!   (`X^T X - Y^T Y = 1`) so that
//!   `d Omega = X^T dA X + X^T dB Y + Y^T dB X + Y^T dA Y`.

use crate::ri_gw::gw_grad::{GwGradEngine, QpCache};
use crate::scf_io::SCF;
use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;
use rest_tensors::matrix::matrix_blas_lapack::_dsolve;
use rest_tensors::matrix::matrix_blas_lapack::_dsyevd;
use tensors::{MathMatrix, MatrixFull};
use tensors::matrix::BasicMatrix;

use crate::ri_cphf::CPHFSolverPySCF;
use crate::solvers::davidson::{lr_davidson_solver, DavidsonConfig, generate_initial_guess};

use std::time::Instant;

/// step-level timing of the gradient assembly (REST_GRAD_TIMING=1)
pub(crate) fn grad_timing() -> bool {
    std::env::var("REST_GRAD_TIMING").map_or(false, |v| v != "0")
}

/// Screening-energy convention for the static W(0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScreeningEnergy {
    /// screening from the mean-field orbital energies (REST default)
    Reference,
    /// screening from the QP energies (PySCF default); the screening-energy
    /// covector is chained through the relaxed QP gradients
    Qp,
}

/// BSE solver selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BseSolver {
    Dense,
    Davidson,
}

// ---------------------------------------------------------------------------
// small dense helpers (flat col-major buffers, first index fastest)
// ---------------------------------------------------------------------------

fn to_mat(v: &[f64], r: usize, c: usize) -> MatrixFull<f64> {
    MatrixFull::from_vec([r, c], v.to_vec()).unwrap()
}

/// Permute the columns of a flat col-major `[n1, naux*n2]` matrix whose
/// column index is `p + x*naux` into the equivalent `[n1, n2*naux]` matrix
/// with column index `x + p*n2`, so that the per-aux blocks are contiguous.
fn permute_aux_slowest(v: &[f64], n1: usize, n2: usize, naux: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n1 * n2 * naux];
    for p in 0..naux {
        for x in 0..n2 {
            let src = (p + x * naux) * n1;
            let dst = (x + p * n2) * n1;
            out[dst..dst + n1].copy_from_slice(&v[src..src + n1]);
        }
    }
    out
}

fn gemm_nn(a: &[f64], ar: usize, ac: usize, b: &[f64], br: usize, bc: usize) -> Vec<f64> {
    let am = to_mat(a, ar, ac);
    let bm = to_mat(b, br, bc);
    let mut out = MatrixFull::new([ar, bc], 0.0);
    _dgemm_full(&am, 'N', &bm, 'N', &mut out, 1.0, 0.0);
    out.data
}

fn gemm_nt(a: &[f64], ar: usize, ac: usize, b: &[f64], br: usize, bc: usize) -> Vec<f64> {
    let am = to_mat(a, ar, ac);
    let bm = to_mat(b, br, bc);
    let mut out = MatrixFull::new([ac, bc], 0.0);
    _dgemm_full(&am, 'T', &bm, 'N', &mut out, 1.0, 0.0);
    out.data
}

fn solve_columns(a: &MatrixFull<f64>, rhs: &[f64], naux: usize, ncol: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; naux * ncol];
    for c in 0..ncol {
        let sol = _dsolve(a, &rhs[c * naux..(c + 1) * naux])
            .expect("bse_grad: singular linear system");
        out[c * naux..(c + 1) * naux].copy_from_slice(&sol);
    }
    out
}

// ---------------------------------------------------------------------------
// static screening
// ---------------------------------------------------------------------------

/// Static screening data `W(0)` with the solved blocks (raw RI representation).
///
/// The BSE transition space may be restricted to the lowest `nvir` virtual
/// orbitals (`bse_cutoff_energy` approximation): the stored `qov`, `qvv`,
/// `v_ia`, `u_ab` and `u_ia` blocks are in the *active* layout (virtual index
/// counted inside the active window), while `gap_full`/`qov_full` keep the
/// full-space response entering `D0 = J - Q0` (its derivative chains over all
/// occupied-virtual pairs, not only the active ones).
pub struct BseScreening {
    pub nocc: usize,
    /// active virtuals in the BSE transition space (<= nvir_full)
    pub nvir: usize,
    /// all virtual orbitals of the mean-field reference
    pub nvir_full: usize,
    pub naux: usize,
    /// active screening gaps de_ia = eps_a - eps_i (nov, ia = i + a*nocc)
    pub gap: Vec<f64>,
    /// full-space screening gaps [naux-independent, nocc*nvir_full]
    pub gap_full: Vec<f64>,
    /// active q_ia [naux, nov]
    pub qov: Vec<f64>,
    /// full-space q_ia [naux, nocc*nvir_full] (D0 response sum)
    pub qov_full: Vec<f64>,
    /// q_ij [naux, nocc^2]
    pub qoo: Vec<f64>,
    /// active q_ab [naux, nvir^2]
    pub qvv: Vec<f64>,
    /// active v_ia = J^-1 q_ia [naux, nov]
    pub v_ia: Vec<f64>,
    /// u_ij = D0^-1 q_ij [naux, nocc^2]
    pub u_ij: Vec<f64>,
    /// active u_ab = D0^-1 q_ab [naux, nvir^2]
    pub u_ab: Vec<f64>,
    /// active u_ia = D0^-1 q_ia [naux, nov]
    pub u_ia: Vec<f64>,
}

impl BseScreening {
    /// Build the static screening from raw ingredients (used by `build` and
    /// by the synthetic-perturbation diagnostics).
    pub fn build_from_parts(
        q: &[f64],
        j_mat: &MatrixFull<f64>,
        nocc: usize,
        nmo: usize,
        eps_screen: &[f64],
    ) -> BseScreening {
        Self::build_from_parts_cutoff(q, j_mat, nocc, nmo, nmo - nocc, eps_screen)
    }

    /// Build the static screening with the BSE transition space restricted to
    /// the lowest `nvir_act` virtual orbitals.  The static RPA response `Q0`
    /// (and hence `D0` and its derivative chain) still runs over *all*
    /// occupied-virtual pairs built from `eps_screen`; only the solved
    /// W(0) blocks handed to the BSE operator are restricted.
    pub fn build_from_parts_cutoff(
        q: &[f64],
        j_mat: &MatrixFull<f64>,
        nocc: usize,
        nmo: usize,
        nvir_act: usize,
        eps_screen: &[f64],
    ) -> BseScreening {
        let naux = j_mat.size()[0];
        let nvir_full = nmo - nocc;
        assert!(nvir_act >= 1 && nvir_act <= nvir_full, "bse_grad: invalid active virtual count");
        let pair = |p: usize, q_mo: usize| p + q_mo * nmo;

        let nov_full = nocc * nvir_full;
        let nov = nocc * nvir_act;
        let mut qov_full = vec![0.0f64; naux * nov_full];
        for a in nocc..nmo {
            for i in 0..nocc {
                let src = pair(i, a) * naux;
                let dst = (i + (a - nocc) * nocc) * naux;
                qov_full[dst..dst + naux].copy_from_slice(&q[src..src + naux]);
            }
        }
        let mut qov = vec![0.0f64; naux * nov];
        qov.copy_from_slice(&qov_full[..naux * nov]);
        let mut qoo = vec![0.0f64; naux * nocc * nocc];
        for j in 0..nocc {
            for i in 0..nocc {
                let src = pair(i, j) * naux;
                let dst = (i + j * nocc) * naux;
                qoo[dst..dst + naux].copy_from_slice(&q[src..src + naux]);
            }
        }
        let mut qvv = vec![0.0f64; naux * nvir_act * nvir_act];
        for b in 0..nvir_act {
            for a in 0..nvir_act {
                let src = pair(nocc + a, nocc + b) * naux;
                let dst = (a + b * nvir_act) * naux;
                qvv[dst..dst + naux].copy_from_slice(&q[src..src + naux]);
            }
        }
        let mut gap_full = vec![0.0f64; nov_full];
        let mut gap = vec![0.0f64; nov];
        for a in nocc..nmo {
            for i in 0..nocc {
                let g = eps_screen[a] - eps_screen[i];
                gap_full[i + (a - nocc) * nocc] = g;
                if a - nocc < nvir_act {
                    gap[i + (a - nocc) * nocc] = g;
                }
            }
        }
        if gap_full.iter().any(|&g| g <= 0.0) {
            panic!("bse_grad: static screening requires positive occupied-virtual gaps");
        }

        // D0 = J - Q0 with Q0 = sum_ia q_ia (-4/de_ia) q_ia^T over ALL pairs
        // (Q0 negative definite), so D0 = J + sum_ia q_ia (4/de_ia) q_ia^T.
        // The scaled form Qs = qov_full * sqrt(4/de) turns this into a
        // single BLAS-3 product D0 = J + Qs Qs^T.
        let mut qs = vec![0.0f64; naux * nov_full];
        for col in 0..nov_full {
            let s = (4.0 / gap_full[col]).sqrt();
            for p in 0..naux {
                qs[p + col * naux] = qov_full[p + col * naux] * s;
            }
        }
        let mut d0m = to_mat(&j_mat.data, naux, naux);
        let qsm = to_mat(&qs, naux, nov_full);
        _dgemm_full(&qsm, 'N', &qsm, 'T', &mut d0m, 1.0, 1.0);
        let v_ia = solve_columns(j_mat, &qov, naux, nov);
        let u_ij = solve_columns(&d0m, &qoo, naux, nocc * nocc);
        let u_ab = solve_columns(&d0m, &qvv, naux, nvir_act * nvir_act);
        let u_ia = solve_columns(&d0m, &qov, naux, nov);
        BseScreening {
            nocc,
            nvir: nvir_act,
            nvir_full,
            naux,
            gap,
            gap_full,
            qov,
            qov_full,
            qoo,
            qvv,
            v_ia,
            u_ij,
            u_ab,
            u_ia,
        }
    }

    /// Build the static screening from the GW engine's raw-RI data with the
    /// given screening orbital energies.
    pub fn build(gw: &GwGradEngine, eps_screen: &[f64]) -> BseScreening {
        Self::build_from_parts(&gw.q, &gw.j_mat, gw.nocc, gw.nmo, eps_screen)
    }
}

// ---------------------------------------------------------------------------
// dense BSE matrices and solvers (amplitude flat index = i + a*nocc, i fastest)
// ---------------------------------------------------------------------------

/// Dense static BSE matrices:
/// `A = diag(gap_qp) + kappa*v_bare - W_A`, `B = kappa*v_bare - W_B`.
pub fn build_bse_matrices(
    screen: &BseScreening,
    qp_energies: &[f64],
    kappa: f64,
) -> (Vec<f64>, Vec<f64>) {
    let nocc = screen.nocc;
    let nvir = screen.nvir;
    let dim = nocc * nvir;

    // bare e-h exchange v_bare[ia, jb] = (ia|jb) = q_ia^T J^-1 q_jb.
    // In the raw RI representation the Coulomb metric is explicit
    // (v_ia = J^-1 q_ia); the transformed REST representation
    // (q_tilde^T q_tilde) is algebraically identical.
    let mut v_bare = vec![0.0f64; dim * dim];
    for ia in 0..dim {
        let i = ia % nocc;
        let av = ia / nocc;
        let c1 = i + av * nocc;
        for jb in 0..dim {
            let j = jb % nocc;
            let bv = jb / nocc;
            let c2 = j + bv * nocc;
            let mut acc = 0.0;
            for p in 0..screen.naux {
                acc += screen.qov[p + c1 * screen.naux] * screen.v_ia[p + c2 * screen.naux];
            }
            v_bare[ia + jb * dim] = acc;
        }
    }
    // w_a[i+j*nocc, a+b*nvir] = q_oo^T D0^-1 q_ab
    let w_a = gemm_nt(
        &screen.qoo,
        screen.naux,
        nocc * nocc,
        &screen.u_ab,
        screen.naux,
        nvir * nvir,
    );
    // w_b_raw[c1, c2] = q_ia[c1]^T D0^-1 u_ia[c2]; W_B[ia, jb] = w_b_raw[i + b*nocc, j + a*nocc]
    let w_b_raw = gemm_nt(&screen.qov, screen.naux, dim, &screen.u_ia, screen.naux, dim);

    let mut a_mat = vec![0.0f64; dim * dim];
    let mut b_mat = vec![0.0f64; dim * dim];
    for ia in 0..dim {
        let i = ia % nocc;
        let av = ia / nocc;
        for jb in 0..dim {
            let j = jb % nocc;
            let bv = jb / nocc;
            let vb = v_bare[ia + jb * dim];
            let wa = w_a[(i + j * nocc) + (av + bv * nvir) * (nocc * nocc)];
            let wb = w_b_raw[(i + bv * nocc) + (j + av * nocc) * dim];
            // diagonal QP gap term (PySCF: A = np.diag(gap) + kappa*v - wA)
            let gap_term = if ia == jb {
                qp_energies[nocc + av] - qp_energies[i]
            } else {
                0.0
            };
            a_mat[ia + jb * dim] = gap_term + kappa * vb - wa;
            b_mat[ia + jb * dim] = kappa * vb - wb;
        }
    }
    (a_mat, b_mat)
}

fn metric_norm(x: &[f64], y: &[f64], dim: usize) -> f64 {
    let mut m = 0.0f64;
    for r in 0..dim {
        m += x[r] * x[r] - y[r] * y[r];
    }
    m.sqrt()
}

/// Solve the full (non-TDA) BSE eigenproblem densely through the
/// nonsymmetric squared product `(A-B)(A+B) u = omega^2 u`.
/// Returns the `nroots` lowest metric-normalised roots.
pub fn solve_bse_dense(
    a_mat: &[f64],
    b_mat: &[f64],
    dim: usize,
    nroots: usize,
) -> Vec<(f64, Vec<f64>, Vec<f64>)> {
    let n2 = dim * dim;
    let apb: Vec<f64> = (0..n2).map(|k| a_mat[k] + b_mat[k]).collect();
    let amb: Vec<f64> = (0..n2).map(|k| a_mat[k] - b_mat[k]).collect();
    // squared problem matrix P = (A-B)(A+B); its eigenvectors are U = X+Y
    let p_mat = gemm_nn(&amb, dim, dim, &apb, dim, dim);
    let pm = to_mat(&p_mat, dim, dim);
    let (_tm, wr, wi, _vl, vr, _info) =
        rest_tensors::matrix::matrix_blas_lapack::_dgeev(&pm, 'N', 'V');

    // collect positive-real roots sorted by omega
    let mut cands: Vec<(f64, usize)> = Vec::new();
    for k in 0..dim {
        if wi[k].abs() < 1.0e-10 && wr[k] > 1.0e-10 {
            cands.push((wr[k].sqrt(), k));
        }
    }
    cands.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    if cands.len() < nroots {
        panic!("bse_grad: found only {} real positive roots (need {})", cands.len(), nroots);
    }

    let mut out = Vec::with_capacity(nroots);
    for k in 0..nroots {
        let (omega, idx) = cands[k];
        let mut u: Vec<f64> = (0..dim).map(|r| vr[r + idx * dim]).collect();
        // inverse-iteration refinement (dgeev vectors can be inaccurate for
        // clustered eigenvalues)
        {
            let shift = omega * omega * (1.0 + 1.0e-8);
            let mut pm = p_mat.clone();
            for r in 0..dim {
                pm[r + r * dim] -= shift;
            }
            let pmm = to_mat(&pm, dim, dim);
            let mut z = u.clone();
            for _ in 0..3 {
                let sol = _dsolve(&pmm, &z).expect("bse_grad: inverse iteration failed");
                let zn = sol.iter().map(|v| v * v).sum::<f64>().sqrt();
                for r in 0..dim {
                    z[r] = sol[r] / zn;
                }
            }
            let dot: f64 = z.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
            if dot < 0.0 {
                for r in 0..dim {
                    z[r] = -z[r];
                }
            }
            u = z;
        }
        // (A+B) U = omega V  ->  V = (A+B) U / omega
        let v_vec = gemm_nn(&apb, dim, dim, &u, dim, 1);
        let mut x = vec![0.0f64; dim];
        let mut yy = vec![0.0f64; dim];
        let mut uv = 0.0f64;
        for r in 0..dim {
            let vv = v_vec[r] / omega;
            x[r] = 0.5 * (u[r] + vv);
            yy[r] = 0.5 * (u[r] - vv);
            uv += u[r] * vv;
        }
        // metric normalisation X^T X - Y^T Y = U^T V = 1
        let nrm = uv.sqrt();
        for r in 0..dim {
            x[r] /= nrm;
            yy[r] /= nrm;
        }
        out.push((omega, x, yy));
    }
    out
}

fn identity(dim: usize) -> Vec<f64> {
    let mut e = vec![0.0f64; dim * dim];
    for i in 0..dim {
        e[i + i * dim] = 1.0;
    }
    e
}

/// Solve the BSE with REST's Davidson solver (Casida form).
///
/// Following the standard Davidson practice, the solver is asked for a few
/// more roots than requested and only the lowest `nroots` are kept; each
/// returned root is verified against the original BSE equations.
pub fn solve_bse_davidson(
    a_mat: &[f64],
    b_mat: &[f64],
    dim: usize,
    nroots: usize,
    config: &DavidsonConfig,
) -> Vec<(f64, Vec<f64>, Vec<f64>)> {
    let nrun = (nroots + 2).min(dim);
    let a_cl = a_mat.clone();
    let b_cl = b_mat.clone();
    let mut a_mv = |z: &Vec<f64>| -> Vec<f64> { gemm_nn(&a_cl, dim, dim, z, dim, 1) };
    let mut b_mv = |z: &Vec<f64>| -> Vec<f64> { gemm_nn(&b_cl, dim, dim, z, dim, 1) };
    let diag: Vec<f64> = (0..dim).map(|k| a_mat[k + k * dim]).collect();
    let guess = generate_initial_guess(&diag, nrun);
    let mut pairs = lr_davidson_solver(&mut a_mv, &mut b_mv, nrun, &diag, guess, config);
    pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    assert!(
        pairs.len() >= nroots,
        "bse_grad: Davidson returned only {} of {} requested roots (dim={}); \
         increase max_subspace/max_iter or use the dense solver",
        pairs.len(),
        nroots,
        dim
    );

    pairs
        .into_iter()
        .take(nroots)
        .map(|(omega, vec)| {
            let mut x = vec[..dim].to_vec();
            let mut y = vec[dim..2 * dim].to_vec();
            let nrm = metric_norm(&x, &y, dim);
            for r in 0..dim {
                x[r] /= nrm;
                y[r] /= nrm;
            }
            // verify the original equations: A X + B Y = omega X, B X + A Y = -omega Y
            let (r1, r2) = bse_residual(a_mat, b_mat, dim, omega, &x, &y);
            let scale = 1.0f64.max(omega.abs());
            let res = (r1.iter().map(|v| v * v).sum::<f64>().sqrt()
                + r2.iter().map(|v| v * v).sum::<f64>().sqrt())
                / scale;
            assert!(
                res < 1.0e-6,
                "bse_grad: Davidson root omega={:.10} failed the BSE residual check ({:.3e}); \
                 increase eigen_tol/max_subspace or use the dense solver",
                omega, res
            );
            (omega, x, y)
        })
        .collect()
}

/// Residuals of the original non-Hermitian BSE equations for one root.
fn bse_residual(
    a_mat: &[f64],
    b_mat: &[f64],
    dim: usize,
    omega: f64,
    x: &[f64],
    y: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let ax = gemm_nn(a_mat, dim, dim, x, dim, 1);
    let by = gemm_nn(b_mat, dim, dim, y, dim, 1);
    let bx = gemm_nn(b_mat, dim, dim, x, dim, 1);
    let ay = gemm_nn(a_mat, dim, dim, y, dim, 1);
    let r1 = (0..dim)
        .map(|r| ax[r] + by[r] - omega * x[r])
        .collect::<Vec<_>>();
    let r2 = (0..dim)
        .map(|r| bx[r] + ay[r] + omega * y[r])
        .collect::<Vec<_>>();
    (r1, r2)
}

// ---------------------------------------------------------------------------
// kernel VJP and amplitude weights
// ---------------------------------------------------------------------------

/// Amplitude weights c_p = d Omega / d E_p at fixed amplitudes (nmo).
///
/// Amplitude layout throughout: flat = i + a*nocc (i fastest), identical to
/// the A/B matrix convention.
pub fn qp_gap_weights(x: &[f64], y: &[f64], nocc: usize, nvir: usize) -> Vec<f64> {
    let mut w = vec![0.0f64; nocc + nvir];
    for i in 0..nocc {
        let mut s = 0.0f64;
        for a in 0..nvir {
            let k = i + a * nocc;
            s += x[k] * x[k] + y[k] * y[k];
        }
        w[i] = -s;
    }
    for a in 0..nvir {
        let mut s = 0.0f64;
        for i in 0..nocc {
            let k = i + a * nocc;
            s += x[k] * x[k] + y[k] * y[k];
        }
        w[nocc + a] = s;
    }
    w
}

/// Individual pullback pieces of one BSE state (for validation; `kernel_vjp`
/// returns their sum).  Layouts as in `kernel_vjp`.
pub struct KernelVjpPieces {
    /// bare-exchange q covector (symmetrised full-pair form)
    pub qbar_ex: Vec<f64>,
    /// bare-exchange J covector `-kappa vz vz^T` (symmetric)
    pub jbar_ex: Vec<f64>,
    /// direct-A q covector: `-(foo oo-block + fvv vv-block)` (full-pair)
    pub qbar_oo: Vec<f64>,
    pub qbar_vv: Vec<f64>,
    /// direct-B q covector `-2 fov` (symmetrised full-pair form)
    pub qbar_ov_direct: Vec<f64>,
    /// direct-A D0 covector (symmetric); contracts with dJ and, through the
    /// screening chain, with dq and deps
    pub polar_a: Vec<f64>,
    /// direct-B D0 covector (symmetric)
    pub polar_b: Vec<f64>,
    /// screening-chain q covector from the direct-A Dbar
    pub qbar_bov_a: Vec<f64>,
    /// screening-chain q covector from the direct-B Dbar
    pub qbar_bov_b: Vec<f64>,
    /// screening-energy covector from the direct-A Dbar
    pub ebar_a: Vec<f64>,
    /// screening-energy covector from the direct-B Dbar
    pub ebar_b: Vec<f64>,
}

/// Screening chain `Qbar -> (dq, deps)` for one Dbar contribution.
/// The D0 dependence runs over ALL occupied-virtual pairs (`qov_full`,
/// `gap_full`), not only the active transition space:
/// `q_times = -Dbar @ q_ia`, `bar_ov = -8 q_times / gap` (half on each of the
/// (i,a)/(a,i) entries), `bar_gap = 4 sum_P q q_times / gap^2` with
/// `dOmega/deps_i -= bar_gap`, `dOmega/deps_a += bar_gap`.
fn chain_from_dbar(
    screen: &BseScreening,
    dbar: &[f64],
    q_pair_layout: impl Fn(usize, usize) -> usize,
    nmo: usize,
) -> (Vec<f64>, Vec<f64>) {
    let nocc = screen.nocc;
    let naux = screen.naux;
    let nov = nocc * screen.nvir_full;
    let mut qbar_bov = vec![0.0f64; naux * nmo * nmo];
    let mut ebar = vec![0.0f64; nmo];
    // q_times[p + c*naux] = -sum_q dbar[p + q*naux] qov_full[q + c*naux]
    let dbar_m = to_mat(dbar, naux, naux);
    let qov_m = to_mat(&screen.qov_full, naux, nov);
    let mut q_times = MatrixFull::new([naux, nov], 0.0);
    _dgemm_full(&dbar_m, 'N', &qov_m, 'N', &mut q_times, -1.0, 0.0);
    for c in 0..nov {
        let i = c % nocc;
        let a = nocc + c / nocc;
        let inv_gap = 1.0 / screen.gap_full[c];
        let col = q_pair_layout(i, a) * naux;
        let ctrans = q_pair_layout(a, i) * naux;
        let mut bar_gap = 0.0f64;
        for p in 0..naux {
            let bov = -8.0 * q_times.data[p + c * naux] * inv_gap;
            qbar_bov[col + p] += 0.5 * bov;
            qbar_bov[ctrans + p] += 0.5 * bov;
            bar_gap += 4.0 * screen.qov_full[p + c * naux] * q_times.data[p + c * naux] * inv_gap * inv_gap;
        }
        ebar[i] -= bar_gap;
        ebar[a] += bar_gap;
    }
    (qbar_bov, ebar)
}

/// Pull one BSE state back onto (q, J, screening energies), returning the
/// individual pieces (see [`KernelVjpPieces`]).
pub fn kernel_vjp_pieces(
    screen: &BseScreening,
    x: &[f64],
    y: &[f64],
    kappa: f64,
    q_pair_layout: impl Fn(usize, usize) -> usize,
    nmo: usize,
) -> KernelVjpPieces {
    let nocc = screen.nocc;
    let nvir = screen.nvir;
    let naux = screen.naux;
    let nov = nocc * nvir;
    let dim = nocc * nvir;

    let mut qbar_ex = vec![0.0f64; naux * nmo * nmo];
    let mut jbar_ex = vec![0.0f64; naux * naux];
    let mut timers = [0.0f64; 6];

    // ---- bare exchange (rank-one in X+Y) ----
    let t0 = Instant::now();
    let mut z = vec![0.0f64; dim];
    for k in 0..dim {
        z[k] = x[k] + y[k];
    }
    let mut vz = vec![0.0f64; naux];
    for p in 0..naux {
        let mut acc = 0.0;
        for c in 0..nov {
            acc += screen.v_ia[p + c * naux] * z[c];
        }
        vz[p] = acc;
    }
    for c in 0..nov {
        let i = c % nocc;
        let a = nocc + c / nocc;
        let col = q_pair_layout(i, a) * naux;
        let ctrans = q_pair_layout(a, i) * naux;
        for p in 0..naux {
            // the full symmetric dq counts both entries, so half each
            qbar_ex[col + p] += kappa * vz[p] * z[c];
            qbar_ex[ctrans + p] += kappa * vz[p] * z[c];
        }
    }
    for p in 0..naux {
        for q in 0..naux {
            jbar_ex[p + q * naux] = -kappa * vz[p] * vz[q];
        }
    }
    timers[0] = t0.elapsed().as_secs_f64();

    // ---- direct A and B blocks (BLAS-3) ----
    // Intermediate contracted tensors are stored aux-slowest:
    // `foo[p*nocc^2 + (i + j*nocc)]`, `fvv[p*nvir^2 + (a + b*nvir)]`,
    // `fov[p*nov + (i + a*nocc)]`.
    let x_m = to_mat(x, nocc, nvir);
    let y_m = to_mat(y, nocc, nvir);
    let u_ab_view = to_mat(&screen.u_ab, naux * nvir, nvir);
    let u_ij_block = to_mat(&screen.u_ij, naux * nocc, nocc);
    let u_ij_view = to_mat(&screen.u_ij, naux, nocc * nocc);
    let u_ia_view = to_mat(&screen.u_ia, naux, nov);
    let mut qbar_oo = vec![0.0f64; naux * nmo * nmo];
    let mut qbar_vv = vec![0.0f64; naux * nmo * nmo];
    let mut polar_a = vec![0.0f64; naux * naux];
    for amplitude in [&x_m, &y_m] {
        // foo[p, i, j] = sum_ab amplitude[i,a] u_ab[p,a,b] amplitude[j,b]:
        //   t1[(p + a*naux), j] = sum_b u_ab[(p,a), b] amplitude[j, b]
        // rows of t1 are permuted aux-slowest (contiguous per-P blocks, the
        // permuted buffer is only naux*nvir*nocc) and then
        //   foo_p = amplitude @ t1_p                             (per P)
        let t0 = Instant::now();
        let mut t1 = MatrixFull::new([naux * nvir, nocc], 0.0);
        _dgemm_full(&u_ab_view, 'N', amplitude, 'T', &mut t1, 1.0, 0.0);
        let mut t1_pm = vec![0.0f64; naux * nvir * nocc];
        for p in 0..naux {
            for a in 0..nvir {
                let r = p + a * naux;
                let dst = (p * nvir + a) * nocc;
                for j in 0..nocc {
                    t1_pm[dst + j] = t1.data[r + j * naux * nvir];
                }
            }
        }
        let mut foo = vec![0.0f64; naux * nocc * nocc];
        {
            // the permuted block is naturally [nocc, nvir] col-major with
            // sl[j, a] = t1_p[a, j], so contract with 'T'
            let mut sl = MatrixFull::new([nocc, nvir], 0.0);
            let mut fp = MatrixFull::new([nocc, nocc], 0.0);
            for p in 0..naux {
                sl.data.copy_from_slice(&t1_pm[p * nvir * nocc..(p + 1) * nvir * nocc]);
                _dgemm_full(amplitude, 'N', &sl, 'T', &mut fp, 1.0, 0.0);
                foo[p * nocc * nocc..(p + 1) * nocc * nocc].copy_from_slice(&fp.data);
            }
        }
        for p in 0..naux {
            for ij in 0..nocc * nocc {
                let i = ij % nocc;
                let j = ij / nocc;
                let col = q_pair_layout(i, j) * naux;
                let ctrans = q_pair_layout(j, i) * naux;
                qbar_oo[col + p] -= 0.5 * foo[p * nocc * nocc + ij];
                qbar_oo[ctrans + p] -= 0.5 * foo[p * nocc * nocc + ij];
            }
        }
        timers[1] += t0.elapsed().as_secs_f64();
        // fvv[p, a, b] = sum_ij amplitude[i,a] u_ij[p,i,j] amplitude[j,b]:
        //   t2[(p + i*naux), b] = sum_j u_ij[(p,i), j] amplitude[j, b]
        // rows permuted aux-slowest, then fvv_p = amplitude^T t2_p (per P)
        let t0 = Instant::now();
        let mut t2 = MatrixFull::new([naux * nocc, nvir], 0.0);
        _dgemm_full(&u_ij_block, 'N', amplitude, 'N', &mut t2, 1.0, 0.0);
        let mut t2_pm = vec![0.0f64; naux * nocc * nvir];
        for p in 0..naux {
            for i in 0..nocc {
                let r = p + i * naux;
                let dst = (p * nocc + i) * nvir;
                for b in 0..nvir {
                    t2_pm[dst + b] = t2.data[r + b * naux * nocc];
                }
            }
        }
        let mut fvv = vec![0.0f64; naux * nvir * nvir];
        {
            // the permuted block is naturally [nvir, nocc] col-major with
            // t2p[b, i] = t2_p[i, b]
            let mut t2p = MatrixFull::new([nvir, nocc], 0.0);
            let mut fvp = MatrixFull::new([nvir, nvir], 0.0);
            for p in 0..naux {
                t2p.data.copy_from_slice(&t2_pm[p * nocc * nvir..(p + 1) * nocc * nvir]);
                _dgemm_full(amplitude, 'T', &t2p, 'T', &mut fvp, 1.0, 0.0);
                fvv[p * nvir * nvir..(p + 1) * nvir * nvir].copy_from_slice(&fvp.data);
            }
        }
        for p in 0..naux {
            for ab in 0..nvir * nvir {
                let a = nocc + ab % nvir;
                let b = nocc + ab / nvir;
                let col = q_pair_layout(a, b) * naux;
                let ctrans = q_pair_layout(b, a) * naux;
                qbar_vv[col + p] -= 0.5 * fvv[p * nvir * nvir + ab];
                qbar_vv[ctrans + p] -= 0.5 * fvv[p * nvir * nvir + ab];
            }
        }
        timers[2] += t0.elapsed().as_secs_f64();
        // Dbar_A[p,q] = sum_ij u_ij[p,ij] foo[q,ij] (one gemm per amplitude)
        let t0 = Instant::now();
        let foo_mat = to_mat(&foo, nocc * nocc, naux);
        let mut pol = MatrixFull::new([naux, naux], 0.0);
        _dgemm_full(&u_ij_view, 'N', &foo_mat, 'N', &mut pol, 1.0, 0.0);
        for (d, s) in polar_a.iter_mut().zip(pol.data.iter()) {
            *d += s;
        }
        timers[4] += t0.elapsed().as_secs_f64();
    }

    // ---- direct B: fov[p,i,a] = sum_jk (X[i,k] u_ia[p,k,j] Y[j,a] + Y X);
    // qbar_ov -= 2 fov and Dbar += u_ia @ fov^T (PySCF direct-B block) ----
    let t0 = Instant::now();
    let mut qbar_ov_direct = vec![0.0f64; naux * nmo * nmo];
    //   s1[i, (p + j*naux)] = sum_k X[i,k] u_ia[(p,j), k];  s2 likewise with Y
    // columns permuted aux-slowest (contiguous per-P blocks), then
    //   fov_p = s1_p Y + s2_p X                                      (per P)
    let u2_view = to_mat(&screen.u_ia, naux * nocc, nvir);
    let mut s1 = MatrixFull::new([nocc, naux * nocc], 0.0);
    _dgemm_full(&x_m, 'N', &u2_view, 'T', &mut s1, 1.0, 0.0);
    let mut s2 = MatrixFull::new([nocc, naux * nocc], 0.0);
    _dgemm_full(&y_m, 'N', &u2_view, 'T', &mut s2, 1.0, 0.0);
    let s1_pm = permute_aux_slowest(&s1.data, nocc, nocc, naux);
    let s2_pm = permute_aux_slowest(&s2.data, nocc, nocc, naux);
    let mut fov = vec![0.0f64; naux * nov];
    {
        let mut s1p = MatrixFull::new([nocc, nocc], 0.0);
        let mut s2p = MatrixFull::new([nocc, nocc], 0.0);
        let mut fvp = MatrixFull::new([nocc, nvir], 0.0);
        let mut tmp = MatrixFull::new([nocc, nvir], 0.0);
        for p in 0..naux {
            s1p.data.copy_from_slice(&s1_pm[p * nocc * nocc..(p + 1) * nocc * nocc]);
            s2p.data.copy_from_slice(&s2_pm[p * nocc * nocc..(p + 1) * nocc * nocc]);
            _dgemm_full(&s1p, 'N', &y_m, 'N', &mut fvp, 1.0, 0.0);
            _dgemm_full(&s2p, 'N', &x_m, 'N', &mut tmp, 1.0, 0.0);
            for (d, s) in fvp.data.iter_mut().zip(tmp.data.iter()) {
                *d += s;
            }
            fov[p * nov..(p + 1) * nov].copy_from_slice(&fvp.data);
        }
    }
    for c in 0..nov {
        let i = c % nocc;
        let a = nocc + c / nocc;
        let col = q_pair_layout(i, a) * naux;
        let ctrans = q_pair_layout(a, i) * naux;
        for p in 0..naux {
            qbar_ov_direct[col + p] += -fov[p * nov + c];
            qbar_ov_direct[ctrans + p] += -fov[p * nov + c];
        }
    }
    timers[3] = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    // Dbar_B[p,q] = sum_c u_ia[p,c] fov[q,c] (one gemm)
    let fov_mat = to_mat(&fov, nov, naux);
    let mut pol = MatrixFull::new([naux, naux], 0.0);
    _dgemm_full(&u_ia_view, 'N', &fov_mat, 'N', &mut pol, 1.0, 0.0);
    let mut polar_b = pol.data;
    for p in 0..naux {
        for q in 0..naux {
            let sm = 0.5 * (polar_a[p + q * naux] + polar_a[q + p * naux]);
            polar_a[p + q * naux] = sm;
            polar_a[q + p * naux] = sm;
            let sm = 0.5 * (polar_b[p + q * naux] + polar_b[q + p * naux]);
            polar_b[p + q * naux] = sm;
            polar_b[q + p * naux] = sm;
        }
    }
    timers[4] += t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let (qbar_bov_a, ebar_a) = chain_from_dbar(screen, &polar_a, &q_pair_layout, nmo);
    let (qbar_bov_b, ebar_b) = chain_from_dbar(screen, &polar_b, &q_pair_layout, nmo);
    timers[5] = t0.elapsed().as_secs_f64();
    if grad_timing() {
        eprintln!(
            "[bse_grad timing] kernel_vjp pieces: ex {:8.3}s  oo {:8.3}s  vv {:8.3}s  ov {:8.3}s  polar {:8.3}s  chain {:8.3}s",
            timers[0], timers[1], timers[2], timers[3], timers[4], timers[5]
        );
    }

    KernelVjpPieces {
        qbar_ex,
        jbar_ex,
        qbar_oo,
        qbar_vv,
        qbar_ov_direct,
        polar_a,
        polar_b,
        qbar_bov_a,
        qbar_bov_b,
        ebar_a,
        ebar_b,
    }
}

/// Pull one BSE state back onto (q, J, screening energies).
///
/// Returns `(qbar [naux*nmo*nmo symmetric-pair layout], jbar [naux^2],
/// ebar [nmo])` with
/// `d Omega_kernel = sum qbar:dq + sum jbar:dJ + sum ebar[p] deps[p]`.
pub fn kernel_vjp(
    screen: &BseScreening,
    x: &[f64],
    y: &[f64],
    kappa: f64,
    q_pair_layout: impl Fn(usize, usize) -> usize,
    nmo: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let naux = screen.naux;
    let pcs = kernel_vjp_pieces(screen, x, y, kappa, q_pair_layout, nmo);
    let mut qbar = pcs.qbar_ex;
    for src in [
        &pcs.qbar_oo,
        &pcs.qbar_vv,
        &pcs.qbar_ov_direct,
        &pcs.qbar_bov_a,
        &pcs.qbar_bov_b,
    ] {
        for (d, s) in qbar.iter_mut().zip(src.iter()) {
            *d += s;
        }
    }
    let mut jbar = vec![0.0f64; naux * naux];
    for k in 0..jbar.len() {
        jbar[k] = pcs.jbar_ex[k] + pcs.polar_a[k] + pcs.polar_b[k];
    }
    let mut ebar = vec![0.0f64; nmo];
    for p in 0..nmo {
        ebar[p] = pcs.ebar_a[p] + pcs.ebar_b[p];
    }
    (qbar, jbar, ebar)
}

// ---------------------------------------------------------------------------
// engine
// ---------------------------------------------------------------------------

/// One solved BSE root.
pub struct BseRoot {
    pub omega: f64,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

/// The static-BSE analytic-gradient engine for one geometry.
pub struct BseGradEngine<'a> {
    pub gw: &'a GwGradEngine<'a>,
    pub screen: BseScreening,
    pub kappa: f64,
    pub screening_energy: ScreeningEnergy,
    pub qp_caches: Vec<QpCache>,
    pub roots: Vec<BseRoot>,
}

impl<'a> BseGradEngine<'a> {
    /// Build the engine: QP caches for every orbital and the static screening.
    pub fn new(
        scf: &'a SCF,
        gw: &'a GwGradEngine<'a>,
        multi: char,
        screening_energy: ScreeningEnergy,
    ) -> Self {
        Self::new_with_cutoff(scf, gw, multi, screening_energy, f64::INFINITY)
    }

    /// Build the engine with the BSE transition space restricted to virtual
    /// orbitals whose *reference* orbital energy lies below `vir_cutoff`
    /// (REST's `bse_cutoff_energy` approximation; pass `f64::INFINITY` for
    /// the full space).  The screening response itself always runs over the
    /// full occupied-virtual space.
    ///
    /// QP caches are only solved for orbitals that carry a finite gradient
    /// weight: the occupied orbitals and the active virtuals.  Inactive
    /// virtuals get empty dummy caches, which `qp_pullback` skips.
    pub fn new_with_cutoff(
        scf: &'a SCF,
        gw: &'a GwGradEngine<'a>,
        multi: char,
        screening_energy: ScreeningEnergy,
        vir_cutoff: f64,
    ) -> Self {
        let nvir_full = gw.nmo - gw.nocc;
        let nvir_act = if vir_cutoff.is_infinite() {
            nvir_full
        } else {
            let eps_ref = &scf.eigenvalues[0][..gw.nmo];
            let nvir_act = (gw.nocc..gw.nmo).filter(|&a| eps_ref[a] < vir_cutoff).count();
            assert!(nvir_act >= 1, "bse_grad: virtual cutoff leaves no active virtual orbitals");
            nvir_act
        };
        let qp_caches: Vec<QpCache> = (0..gw.nmo)
            .map(|n| {
                let needed = n < gw.nocc + nvir_act
                    || matches!(screening_energy, ScreeningEnergy::Qp);
                if needed {
                    gw.build_target_cache(n)
                } else {
                    QpCache {
                        target: n,
                        omega: 0.0,
                        z_factor: 0.0,
                        w_imag: Vec::new(),
                        y_imag: Vec::new(),
                        residues: Vec::new(),
                    }
                }
            })
            .collect();
        let eps_screen: Vec<f64> = match screening_energy {
            ScreeningEnergy::Qp => qp_caches.iter().map(|c| c.omega).collect(),
            ScreeningEnergy::Reference => scf.eigenvalues[0][..gw.nmo].to_vec(),
        };
        let screen = BseScreening::build_from_parts_cutoff(
            &gw.q, &gw.j_mat, gw.nocc, gw.nmo, nvir_act, &eps_screen,
        );
        let kappa = match multi {
            's' | 'S' => 2.0,
            't' | 'T' => 0.0,
            _ => panic!("bse_grad: multi must be 's' or 't'"),
        };
        BseGradEngine {
            gw,
            screen,
            kappa,
            screening_energy,
            qp_caches,
            roots: Vec::new(),
        }
    }

    /// Solve for the lowest `nroots` excitation energies.
    pub fn solve(&mut self, nroots: usize, solver: BseSolver, config: &DavidsonConfig) {
        let qp_energies: Vec<f64> = self.qp_caches.iter().map(|c| c.omega).collect();
        let (a_mat, b_mat) = build_bse_matrices(&self.screen, &qp_energies, self.kappa);
        let dim = self.screen.nocc * self.screen.nvir;
        self.roots = match solver {
            BseSolver::Dense => solve_bse_dense(&a_mat, &b_mat, dim, nroots),
            BseSolver::Davidson => {
                solve_bse_davidson(&a_mat, &b_mat, dim, nroots, config)
            }
        }
        .into_iter()
        .map(|(omega, x, y)| BseRoot { omega, x, y })
        .collect();
    }

    /// Excitation energies of the solved roots.
    pub fn excitation_energies(&self) -> Vec<f64> {
        self.roots.iter().map(|r| r.omega).collect()
    }

    /// Analytic gradients of all solved roots.
    ///
    /// Returns `(grads[k][atm*3 + comp], omegas)`.
    pub fn analytic_gradients(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let nk = self.roots.len();
        let naux = self.gw.naux;
        let nmo = self.gw.nmo;
        let nocc = self.gw.nocc;
        let nvir = self.screen.nvir;
        let pair = |p: usize, q: usize| p + q * nmo;

        let mut weights: Vec<Vec<f64>> = Vec::with_capacity(nk);
        let mut ebar_direct: Vec<Vec<f64>> = Vec::with_capacity(nk);
        let mut bq_k: Vec<Vec<f64>> = Vec::with_capacity(nk);
        let mut bj_k: Vec<Vec<f64>> = Vec::with_capacity(nk);
        let t_kernel = Instant::now();
        for root in &self.roots {
            // amplitude weights cover the occupied orbitals and the active
            // virtuals; inactive virtuals carry zero QP weight
            let w_act = qp_gap_weights(&root.x, &root.y, nocc, nvir);
            let mut w = vec![0.0f64; nmo];
            w[..nocc + nvir].copy_from_slice(&w_act);
            let (qbar, jbar, ebar) =
                kernel_vjp(&self.screen, &root.x, &root.y, self.kappa, pair, nmo);
            match self.screening_energy {
                ScreeningEnergy::Qp => {
                    for p in 0..nmo {
                        w[p] += ebar[p];
                    }
                }
                ScreeningEnergy::Reference => {
                    ebar_direct.push(ebar);
                }
            }
            weights.push(w);
            bq_k.push(qbar);
            bj_k.push(jbar);
        }
        if ebar_direct.len() < nk {
            let pad = vec![0.0f64; nmo];
            while ebar_direct.len() < nk {
                ebar_direct.push(pad.clone());
            }
        }

        let caches: Vec<&QpCache> = self.qp_caches.iter().collect();
        let t_kernel = t_kernel.elapsed().as_secs_f64();
        let t_qp = Instant::now();
        let (bq, bj, be, bb) = self.gw.qp_pullback(&caches, &weights);
        let t_qp = t_qp.elapsed().as_secs_f64();

        // analdrv CP-HF: one block-Krylov solve over all (atm, comp)
        // perturbations
        let t_cp = Instant::now();
        let responses = self.gw.canonical_response_batch();
        let t_cp = t_cp.elapsed().as_secs_f64();
        let t_asm = Instant::now();
        let mut t_qx = 0.0f64;
        let zero_q = vec![0.0f64; naux * nmo * nmo];
        let zero_j = vec![0.0f64; naux * naux];
        let mut grads = vec![vec![0.0f64; self.gw.natm * 3]; nk];
        for atm in 0..self.gw.natm {
            for comp in 0..3 {
                let di = self.gw.raw.d_i_atom(atm, comp);
                let dj = self.gw.raw.d_j_atom(atm, comp);
                let (u, eps1, b_x) = responses[atm * 3 + comp].clone();
                let t0 = Instant::now();
                let qx = self.gw.qx_from_u(&di, &u);
                t_qx += t0.elapsed().as_secs_f64();
                for k in 0..nk {
                    // QP part (with Z factors) + kernel part (direct covector)
                    let qp = self.gw.contract_perturbation(
                        &bq[k..k + 1],
                        &bj[k..k + 1],
                        &be[k..k + 1],
                        &bb[k..k + 1],
                        &qx, &dj, &eps1, &b_x,
                    )[0];
                    let kernel = bq_k[k].iter().zip(qx.iter()).map(|(a, b)| a * b).sum::<f64>()
                        + bj_k[k].iter().zip(dj.iter()).map(|(a, b)| a * b).sum::<f64>()
                        + ebar_direct[k].iter().zip(eps1.iter()).map(|(a, b)| a * b).sum::<f64>();
                    grads[k][atm * 3 + comp] = qp + kernel;
                }
            }
        }
        let omegas = self.roots.iter().map(|r| r.omega).collect();
        if grad_timing() {
            eprintln!(
                "[bse_grad timing] kernel_vjp {:8.2}s  qp_pullback {:8.2}s  cphf+assembly(cphf) {:8.2}s  qx_from_u {:8.2}s  contraction {:8.2}s",
                t_kernel,
                t_qp,
                t_cp,
                t_qx,
                t_asm.elapsed().as_secs_f64() - t_qx
            );
        }
        (grads, omegas)
    }
}

/// Test helper: W_A[i+j*nocc, a+b*nvir] = q_oo^T D0^-1 q_ab.
pub fn build_w_a_test(screen: &BseScreening, _qp: &[f64]) -> Vec<f64> {
    let a = to_mat(&screen.qoo, screen.naux, screen.nocc * screen.nocc);
    let b = to_mat(&screen.u_ab, screen.naux, screen.nvir * screen.nvir);
    let mut out = MatrixFull::new([screen.nocc * screen.nocc, screen.nvir * screen.nvir], 0.0);
    _dgemm_full(&a, 'T', &b, 'N', &mut out, 1.0, 0.0);
    out.data
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    fn rnd(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0
    }

    /// naive scalar reference of the pre-BLAS kernel_vjp_pieces
    fn naive_pieces(
        screen: &BseScreening,
        x: &[f64],
        y: &[f64],
        kappa: f64,
        pair: fn(usize, usize) -> usize,
        nmo: usize,
    ) -> KernelVjpPieces {
        let nocc = screen.nocc;
        let nvir = screen.nvir;
        let naux = screen.naux;
        let nov = nocc * nvir;
        let dim = nocc * nvir;
        let mut qbar_ex = vec![0.0f64; naux * nmo * nmo];
        let mut jbar_ex = vec![0.0f64; naux * naux];
        let mut z = vec![0.0f64; dim];
        for k in 0..dim {
            z[k] = x[k] + y[k];
        }
        let mut vz = vec![0.0f64; naux];
        for p in 0..naux {
            let mut acc = 0.0;
            for c in 0..nov {
                acc += screen.v_ia[p + c * naux] * z[c];
            }
            vz[p] = acc;
        }
        for c in 0..nov {
            let i = c % nocc;
            let a = nocc + c / nocc;
            let col = pair(i, a) * naux;
            let ctrans = pair(a, i) * naux;
            for p in 0..naux {
                qbar_ex[col + p] += kappa * vz[p] * z[c];
                qbar_ex[ctrans + p] += kappa * vz[p] * z[c];
            }
        }
        for p in 0..naux {
            for q in 0..naux {
                jbar_ex[p + q * naux] = -kappa * vz[p] * vz[q];
            }
        }
        let mut qbar_oo = vec![0.0f64; naux * nmo * nmo];
        let mut qbar_vv = vec![0.0f64; naux * nmo * nmo];
        let mut polar_a = vec![0.0f64; naux * naux];
        for amplitude in [x, y] {
            let mut foo = vec![0.0f64; naux * nocc * nocc];
            for p in 0..naux {
                for j in 0..nocc {
                    for i in 0..nocc {
                        let mut acc = 0.0;
                        for a in 0..nvir {
                            for b in 0..nvir {
                                acc += amplitude[i + a * nocc]
                                    * screen.u_ab[p + (a + b * nvir) * naux]
                                    * amplitude[j + b * nocc];
                            }
                        }
                        foo[p + (i + j * nocc) * naux] = acc;
                    }
                }
            }
            for p in 0..naux {
                for ij in 0..nocc * nocc {
                    let i = ij % nocc;
                    let j = ij / nocc;
                    let col = pair(i, j) * naux;
                    let ctrans = pair(j, i) * naux;
                    qbar_oo[col + p] -= 0.5 * foo[p + ij * naux];
                    qbar_oo[ctrans + p] -= 0.5 * foo[p + ij * naux];
                }
            }
            let mut fvv = vec![0.0f64; naux * nvir * nvir];
            for p in 0..naux {
                for b in 0..nvir {
                    for a in 0..nvir {
                        let mut acc = 0.0;
                        for j in 0..nocc {
                            for i in 0..nocc {
                                acc += amplitude[i + a * nocc]
                                    * screen.u_ij[p + (i + j * nocc) * naux]
                                    * amplitude[j + b * nocc];
                            }
                        }
                        fvv[p + (a + b * nvir) * naux] = acc;
                    }
                }
            }
            for p in 0..naux {
                for ab in 0..nvir * nvir {
                    let a = nocc + ab % nvir;
                    let b = nocc + ab / nvir;
                    let col = pair(a, b) * naux;
                    let ctrans = pair(b, a) * naux;
                    qbar_vv[col + p] -= 0.5 * fvv[p + ab * naux];
                    qbar_vv[ctrans + p] -= 0.5 * fvv[p + ab * naux];
                }
            }
            for p in 0..naux {
                for q in 0..naux {
                    let mut acc = 0.0;
                    for ij in 0..nocc * nocc {
                        acc += screen.u_ij[p + ij * naux] * foo[q + ij * naux];
                    }
                    polar_a[p + q * naux] += acc;
                }
            }
        }
        let mut qbar_ov_direct = vec![0.0f64; naux * nmo * nmo];
        let mut polar_b = vec![0.0f64; naux * naux];
        let mut fov = vec![0.0f64; naux * nov];
        for p in 0..naux {
            for a in 0..nvir {
                for i in 0..nocc {
                    let mut acc = 0.0;
                    for j in 0..nocc {
                        for k in 0..nvir {
                            let u_kj = screen.u_ia[p + (j + k * nocc) * naux];
                            acc += x[i + k * nocc] * u_kj * y[j + a * nocc]
                                + y[i + k * nocc] * u_kj * x[j + a * nocc];
                        }
                    }
                    fov[p + (i + a * nocc) * naux] = acc;
                }
            }
        }
        for c in 0..nov {
            let i = c % nocc;
            let a = nocc + c / nocc;
            let col = pair(i, a) * naux;
            let ctrans = pair(a, i) * naux;
            for p in 0..naux {
                qbar_ov_direct[col + p] += -fov[p + c * naux];
                qbar_ov_direct[ctrans + p] += -fov[p + c * naux];
            }
        }
        for p in 0..naux {
            for q in 0..naux {
                let mut acc = 0.0;
                for c in 0..nov {
                    acc += screen.u_ia[p + c * naux] * fov[q + c * naux];
                }
                polar_b[p + q * naux] += acc;
            }
        }
        for p in 0..naux {
            for q in 0..naux {
                let sm = 0.5 * (polar_a[p + q * naux] + polar_a[q + p * naux]);
                polar_a[p + q * naux] = sm;
                polar_a[q + p * naux] = sm;
                let sm = 0.5 * (polar_b[p + q * naux] + polar_b[q + p * naux]);
                polar_b[p + q * naux] = sm;
                polar_b[q + p * naux] = sm;
            }
        }
        let (qbar_bov_a, ebar_a) = chain_from_dbar(screen, &polar_a, pair, nmo);
        let (qbar_bov_b, ebar_b) = chain_from_dbar(screen, &polar_b, pair, nmo);
        KernelVjpPieces {
            qbar_ex,
            jbar_ex,
            qbar_oo,
            qbar_vv,
            qbar_ov_direct,
            polar_a,
            polar_b,
            qbar_bov_a,
            qbar_bov_b,
            ebar_a,
            ebar_b,
        }
    }

    fn pair_layout(p: usize, q: usize) -> usize {
        p + q * 8
    }

    #[test]
    fn kernel_vjp_matches_naive_reference() {
        let mut seed = 7u64;
        let nocc = 3usize;
        let nvir = 4usize;
        let nvir_full = 5usize;
        let naux = 6usize;
        let nmo = 8usize;
        let nov = nocc * nvir;
        let nov_full = nocc * nvir_full;
        let screen = BseScreening {
            nocc,
            nvir,
            nvir_full,
            naux,
            gap: (0..nov).map(|_| 0.5 + rnd(&mut seed)).collect(),
            gap_full: (0..nov_full).map(|_| 0.4 + rnd(&mut seed)).collect(),
            qov: (0..naux * nov).map(|_| 0.2 * rnd(&mut seed)).collect(),
            qov_full: (0..naux * nov_full).map(|_| 0.2 * rnd(&mut seed)).collect(),
            qoo: (0..naux * nocc * nocc).map(|_| 0.2 * rnd(&mut seed)).collect(),
            qvv: (0..naux * nvir * nvir).map(|_| 0.2 * rnd(&mut seed)).collect(),
            v_ia: (0..naux * nov).map(|_| 0.2 * rnd(&mut seed)).collect(),
            u_ij: (0..naux * nocc * nocc).map(|_| 0.2 * rnd(&mut seed)).collect(),
            u_ab: (0..naux * nvir * nvir).map(|_| 0.2 * rnd(&mut seed)).collect(),
            u_ia: (0..naux * nov).map(|_| 0.2 * rnd(&mut seed)).collect(),
        };
        let x: Vec<f64> = (0..nov).map(|_| rnd(&mut seed)).collect();
        let y: Vec<f64> = (0..nov).map(|_| rnd(&mut seed)).collect();
        let kappa = 2.0;
        let reference = naive_pieces(&screen, &x, &y, kappa, pair_layout, nmo);
        let new = kernel_vjp_pieces(&screen, &x, &y, kappa, pair_layout, nmo);
        let cmp = |name: &str, a: &[f64], b: &[f64]| {
            let e = a.iter().zip(b.iter()).map(|(u, v)| (u - v).abs()).fold(0.0, f64::max);
            assert!(e < 1.0e-12, "{name} mismatch {e}");
        };
        cmp("qbar_ex", &reference.qbar_ex, &new.qbar_ex);
        cmp("jbar_ex", &reference.jbar_ex, &new.jbar_ex);
        cmp("qbar_oo", &reference.qbar_oo, &new.qbar_oo);
        cmp("qbar_vv", &reference.qbar_vv, &new.qbar_vv);
        cmp("qbar_ov_direct", &reference.qbar_ov_direct, &new.qbar_ov_direct);
        cmp("polar_a", &reference.polar_a, &new.polar_a);
        cmp("polar_b", &reference.polar_b, &new.polar_b);
        cmp("qbar_bov_a", &reference.qbar_bov_a, &new.qbar_bov_a);
        cmp("qbar_bov_b", &reference.qbar_bov_b, &new.qbar_bov_b);
        cmp("ebar_a", &reference.ebar_a, &new.ebar_a);
        cmp("ebar_b", &reference.ebar_b, &new.ebar_b);
    }
}
