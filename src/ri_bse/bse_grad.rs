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
//!
//! # Cost profile of the gradient assembly
//!
//! Three implementation choices keep the memory and the time of
//! [`BseGradEngine::analytic_gradients`] within one power of the basis size of
//! the quantities that actually enter it:
//!
//! * The static screening solves its blocks with **one** Cholesky factorisation
//!   per metric and batched triangular solves (`solve_columns`), instead of one
//!   `dgesv` per right-hand side.
//! * The QP response caches are built and kept in the **compact**
//!   representation (`QpCache::m_imag`: one `[naux, nmo]` metric intermediate,
//!   from which every quadrature point's response is one gemm away) and are
//!   *borrowed* by the streaming pullback instead of being cloned per page.
//! * The per-perturbation `qx` tensor (`naux*nmo^2` per atom and component) is
//!   never materialised: each root's total pair covector is **folded** once
//!   (`C qbar_P C^T`) and contracted straight against the streamed derivative
//!   integral slices, which turns the `O(3 natm * naux * nao^2 * nmo)` assembly
//!   into `O(naux * nao^2 * nmo)` plus `O(3 natm * naux * nao^2)`.

use crate::ri_gw::gw_grad::{GwGradEngine, QpCache};
use crate::scf_io::SCF;
use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;
use rest_tensors::matrix::matrix_blas_lapack::_dpotrf;
use rest_tensors::matrix::matrix_blas_lapack::_dsolve;
use rest_tensors::matrix::matrix_blas_lapack::_dtrtrs;
use rest_tensors::matrix::matrix_blas_lapack::_dsyevd;
use tensors::{MathMatrix, MatrixFull};
use tensors::matrix::BasicMatrix;

use crate::ri_cphf::CPHFSolverPySCF;
use crate::solvers::davidson::{
    generate_initial_guess, lr_davidson_solver_batched, DavidsonConfig,
};

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

/// Solve `A X = B` for a symmetric positive-definite `A` (`naux x naux`) with
/// many right-hand sides, `B` col-major `[naux, ncol]`.
///
/// `A` is factorised **once** (Cholesky) and both triangular solves run as
/// single BLAS-3 calls over all columns.  The previous column-by-column
/// `dgesv` loop refactorised the very same matrix `ncol` times — `ncol` is
/// `nocc^2 + nvir^2 + 2*nocc*nvir` here (≈ 6000 at naphthalene/def2-SVP), i.e.
/// ~1.4 TFLOP of redundant LU work plus one `naux^2` copy per column, which
/// made the screening build one of the two dominant costs of the gradient.
fn solve_columns(a: &MatrixFull<f64>, rhs: &[f64], naux: usize, ncol: usize) -> Vec<f64> {
    let mut chol = a.clone();
    _dpotrf(&mut chol, 'L');
    let mut b = to_mat(rhs, naux, ncol);
    let ok1 = _dtrtrs(&chol, &mut b, 'L', 'N', 'N');
    let ok2 = _dtrtrs(&chol, &mut b, 'L', 'T', 'N');
    assert!(ok1 && ok2, "bse_grad: metric triangular solve failed");
    b.data
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

    /// Release every block of the screening except the layout scalars.
    ///
    /// The kernel covector of each root is the only reader of `qov`/`qov_full`/
    /// `qoo`/`qvv`/`v_ia`/`u_ij`/`u_ab`/`u_ia` and the gaps; once it is built
    /// they are dead, and they are the sum of the engine's largest resident
    /// blocks (`O(naux*(nocc+nvir)^2)` doubles).  [`Self::blocks_released`] tells
    /// [`BseGradEngine::ensure_screening`] to rebuild them on demand.
    fn release_blocks(&mut self) {
        self.gap = Vec::new();
        self.gap_full = Vec::new();
        self.qov = Vec::new();
        self.qov_full = Vec::new();
        self.qoo = Vec::new();
        self.qvv = Vec::new();
        self.v_ia = Vec::new();
        self.u_ij = Vec::new();
        self.u_ab = Vec::new();
        self.u_ia = Vec::new();
    }

    /// Whether [`Self::release_blocks`] has been called (the screening then has
    /// to be rebuilt before it can be read again).
    pub fn blocks_released(&self) -> bool {
        self.qov_full.is_empty()
    }

    /// Build the static screening from the GW engine's raw-RI data with the
    /// given screening orbital energies.
    pub fn build(gw: &GwGradEngine, eps_screen: &[f64]) -> BseScreening {
        // `GwGradEngine::q` is evictable (`with_q` rebuilds it when a CD-GW gradient released it)
        gw.with_q(|q| Self::build_from_parts(q, &gw.j_mat, gw.nocc, gw.nmo, eps_screen))
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

    // bare e-h exchange `v_bare[ia, jb] = (ia|jb) = q_ia^T J^-1 q_jb`.
    // In the raw RI representation the Coulomb metric is explicit
    // (`v_ia = J^-1 q_ia`), so the whole matrix is the single BLAS-3 product
    // `V = q_ov^T v_ia`; the transformed REST representation
    // (`q_tilde^T q_tilde`) is algebraically identical.  The scalar
    // `O(dim^2 * naux)` triple loop this replaces was ~3 GFLOP of non-BLAS
    // work (≈ 3 s at naphthalene/def2-SVP).
    let t_dbg = Instant::now();
    let qov_m = to_mat(&screen.qov, screen.naux, dim);
    let v_ia_m = to_mat(&screen.v_ia, screen.naux, dim);
    // `A` and `B` both start from `kappa * v_bare`; the kernel blocks and the
    // diagonal QP gap are added below, so the `dim^2` buffer `v_bare` itself
    // never has to exist.
    let mut a_m = MatrixFull::new([dim, dim], 0.0);
    _dgemm_full(&qov_m, 'T', &v_ia_m, 'N', &mut a_m, kappa, 0.0);
    let mut b_mat = a_m.data.clone();
    if grad_timing() {
        eprintln!("[bse_mat timing] v_bare gemm {:8.3}s (dim {dim}, naux {})", t_dbg.elapsed().as_secs_f64(), screen.naux);
    }
    let t_dbg = Instant::now();
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
    if grad_timing() {
        eprintln!("[bse_mat timing] kernel gemms {:8.3}s", t_dbg.elapsed().as_secs_f64());
    }
    let t_dbg = Instant::now();

    // `ia = i + a*nocc` and `jb = j + b*nocc`; the `%`/`/` of every inner
    // iteration are hoisted into lookup tables, and the `jb` loop is walked in
    // its natural `(b, j)` blocks so both kernel blocks are read contiguously.
    let ia_i: Vec<usize> = (0..dim).map(|c| c % nocc).collect();
    let ia_a: Vec<usize> = (0..dim).map(|c| c / nocc).collect();
    let gap_diag: Vec<f64> = (0..dim)
        .map(|c| qp_energies[nocc + ia_a[c]] - qp_energies[ia_i[c]])
        .collect();
    let n_oo = nocc * nocc;
    {
        let a_flat = &mut a_m.data;
        for ia in 0..dim {
            let i = ia_i[ia];
            let av = ia_a[ia];
            let a_row = &mut a_flat[ia * dim..(ia + 1) * dim];
            let b_row = &mut b_mat[ia * dim..(ia + 1) * dim];
            // diagonal QP gap term (PySCF: A = np.diag(gap) + kappa*v - wA)
            a_row[ia] += gap_diag[ia];
            for bv in 0..nvir {
                let jb0 = bv * nocc;
                let wa_blk = (av + bv * nvir) * n_oo;
                let wb_blk = i + bv * nocc;
                for j in 0..nocc {
                    a_row[jb0 + j] -= w_a[wa_blk + i + j * nocc];
                    b_row[jb0 + j] -= w_b_raw[wb_blk + (j + av * nocc) * dim];
                }
            }
        }
    }
    if grad_timing() {
        eprintln!("[bse_mat timing] assembly    {:8.3}s", t_dbg.elapsed().as_secs_f64());
    }
    (a_m.data, b_mat)
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
                // The shifted system can be numerically singular when two roots are
                // nearly degenerate (the acene singlets are), and `_dpotrf` then panics
                // inside `_dsolve`; keep the un-refined `dgeev` vector rather than
                // aborting a run that has already spent an hour in the SCF/GW/BSE
                // build.  The root is still verified below by the BSE residual.
                let solved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    _dsolve(&pmm, &z)
                }))
                .unwrap_or(None);
                let Some(sol) = solved else {
                    eprintln!(
                        "[bse_grad] dense BSE: inverse-iteration refinement failed for \
                         omega={omega:.6} (shifted system not positive definite); keeping the \
                         dgeev vector"
                    );
                    break;
                };
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
) -> Option<Vec<(f64, Vec<f64>, Vec<f64>)>> {
    let mut failed = false;
    let nrun = (nroots + 2).min(dim);
    // The solver applies `A`/`B` to the **whole subspace block** every
    // iteration, so the matvec must be a gemm: the per-vector closure API
    // (`lr_davidson_solver`) expands into one gemv per subspace column, i.e.
    // `2 * m` reads of the full `dim^2` matrix per iteration (m up to
    // `max_subspace`), which dominated the solve.
    let a_m = to_mat(a_mat, dim, dim);
    let b_m = to_mat(b_mat, dim, dim);
    let mut a_mv = |z: &MatrixFull<f64>| -> MatrixFull<f64> {
        let mut out = MatrixFull::new([dim, z.size()[1]], 0.0);
        _dgemm_full(&a_m, 'N', z, 'N', &mut out, 1.0, 0.0);
        out
    };
    let mut b_mv = |z: &MatrixFull<f64>| -> MatrixFull<f64> {
        let mut out = MatrixFull::new([dim, z.size()[1]], 0.0);
        _dgemm_full(&b_m, 'N', z, 'N', &mut out, 1.0, 0.0);
        out
    };
    let diag: Vec<f64> = (0..dim).map(|k| a_mat[k + k * dim]).collect();
    let guess = generate_initial_guess(&diag, nrun);
    let mut pairs = lr_davidson_solver_batched(&mut a_mv, &mut b_mv, nrun, &diag, guess, config);
    pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    if pairs.len() < nroots {
        eprintln!(
            "[bse_grad] Davidson returned only {} of {} requested roots (dim={}); increase \
             max_subspace/max_iter, or use the dense solver",
            pairs.len(),
            nroots,
            dim
        );
        return None;
    }

    let out: Vec<(f64, Vec<f64>, Vec<f64>)> = pairs
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
            if res >= 1.0e-6 {
                eprintln!(
                    "[bse_grad] Davidson root omega={:.10} failed the BSE residual check ({:.3e}); \
                     increase eigen_tol/max_subspace or use the dense solver",
                    omega, res
                );
                failed = true;
            }
            (omega, x, y)
        })
        .collect();
    if failed {
        return None;
    }
    Some(out)
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

/// **Compact** kernel q covector of one BSE state.
///
/// The full-pair buffer that [`KernelVjpPieces`] used to return
/// (`naux * nmo^2`, i.e. 2.5 GB per root at pentacene/def2-TZVP) is only ever
/// read through its `(o,o)`, `(v,v)` and `(o,v)` blocks, and every one of its
/// entries is the *half* weight of the corresponding symmetric pair.  This is
/// the same covector in the minimal representation:
///
/// ```text
/// sum_pq qbar[pq] dq[pq]  ==  sum_P sum_c theta[P, c] dq[P, c]
/// ```
///
/// where `c` runs over the compact columns `(i,j)`, `(a,b)`, `(i,a)` of the
/// occupied/active-virtual space and `dq` is the pair-symmetrised derivative.
/// The memory drops from `naux*nmo^2` to `naux*(nocc + nvir_act)^2`; at
/// naphthalene/def2-SVP that is 178 MB -> 6.5 MB per root.
pub struct KernelVjpCompact {
    /// `naux * ncols` — `theta[c + P*ncols]` with
    /// `ncols = (nocc+nvir)^2`, block order `(i,j) | (a,b) | (i,a) + (a,i)`.
    /// The `(i,a)` column carries the *sum* of the two symmetric entries, so a
    /// single dot product with `qx[(i,a)] + qx[(a,i)]` reproduces the
    /// full-pair contraction.
    /// `(o,o)` and `(v,v)` columns: `naux * (nocc^2 + nvir^2)`, laid out as
    /// `theta[c + P*(nocc^2+nvir^2)]` with `c = i + j*nocc` for the (o,o) block
    /// and `c = nocc^2 + a + b*nvir` for the (v,v) block.  Both diagonal blocks
    /// store the full-pair value, so the contraction multiplies by 2 (the
    /// full-pair covector is symmetric and `qx` stores one value per pair).
    pub theta: Vec<f64>,
    /// `(o,v)` covector `ovbar[i + a*nocc + P*nocc*nvir_full]`, summed over the
    /// two symmetric entries `(i,a)` + `(a,i)` of the full-pair form.  Kept
    /// dense over ALL virtuals (the screening chain contributes for the
    /// inactive ones as well); it is `naux*nocc*nvir_full`, an order of
    /// magnitude smaller than the `naux*nmo^2` full-pair buffer.
    pub ovbar: Vec<f64>,
    pub ncols: usize,
    pub nocc: usize,
    pub nvir: usize,
    /// All virtuals of the mean-field reference: the (o,v) block of `theta`
    /// spans them (the screening chain contributes for every virtual), so the
    /// square layout width is `nocc + nvir_full`.
    pub nvir_full: usize,
    /// bare-exchange J covector `-kappa vz vz^T` (symmetric), `naux^2`
    pub jbar_ex: Vec<f64>,
    pub polar_a: Vec<f64>,
    pub polar_b: Vec<f64>,
    pub ebar_a: Vec<f64>,
    pub ebar_b: Vec<f64>,
}

impl KernelVjpCompact {
    /// `sum_pq qbar[pq] qx[pq]` for one perturbation, with `qx` in the engine's
    /// native layout `qx[P + (p + q*nmo)*naux]`.
    ///
    /// The full-pair covector stores the same value on the two symmetric
    /// entries of a pair while `qx` stores one (symmetrised, equal) value on
    /// each of them, so the full contraction counts every pair **twice**; the
    /// factor 2 below puts the compact form back on the full-pair scale.
    pub fn contract_qx(&self, qx: &[f64], naux: usize, nmo: usize) -> f64 {
        let n_oo = self.nocc * self.nocc;
        let n_vv = self.nvir * self.nvir;
        let nv = self.nvir_full;
        let ncv = self.nocc * nv;
        let mut acc = 0.0f64;
        // (o,o) and (v,v): `theta` mirrors the full-pair covector entry, and the
        // full-pair contraction counts both symmetric entries of a pair -> x2
        for i in 0..self.nocc {
            for j in 0..self.nocc {
                let base = (i + j * nmo) * naux;
                let mut s = 0.0f64;
                for p in 0..naux {
                    s += self.theta[i + j * self.nocc + p * self.ncols] * qx[base + p];
                }
                acc += 2.0 * s;
            }
        }
        for a in 0..self.nvir {
            for b in 0..self.nvir {
                let (ga, gb) = (self.nocc + a, self.nocc + b);
                let base = (ga + gb * nmo) * naux;
                let mut s = 0.0f64;
                for p in 0..naux {
                    s += self.theta[n_oo + a + b * self.nvir + p * self.ncols] * qx[base + p];
                }
                acc += 2.0 * s;
            }
        }
        // (o,v): `ovbar` is the two-entry pair sum, so no extra factor
        for i in 0..self.nocc {
            for a in 0..nv {
                let ga = self.nocc + a;
                let base = (i + ga * nmo) * naux;
                let mut s = 0.0f64;
                for p in 0..naux {
                    s += self.ovbar[i + a * self.nocc + p * ncv] * qx[base + p];
                }
                acc += s;
            }
        }
        let _ = n_vv;
        acc
    }

    /// `jbar = jbar_ex + polar_a + polar_b` (symmetric).
    pub fn jbar(&self) -> Vec<f64> {
        let mut j = vec![0.0f64; self.jbar_ex.len()];
        for k in 0..j.len() {
            j[k] = self.jbar_ex[k] + self.polar_a[k] + self.polar_b[k];
        }
        j
    }

    /// Expand the compact covector into a **full-pair** buffer
    /// `qbar[P + (p + q*nmo)*naux]` and add it there.
    ///
    /// The full-pair representation stores the same value on the two symmetric
    /// entries of a pair (and the contraction with a pair-symmetric `qx` counts
    /// both), so the `(o,o)`/`(v,v)` columns are the full-pair value itself
    /// while the `(o,v)` column — the pair *sum* — is halved on each entry.
    /// This is what lets the BSE kernel covector join the QP covector in the
    /// folded (`C qbar C^T`) route of the gradient assembly.
    pub fn add_into_full_pair(&self, qbar: &mut [f64], naux: usize, nmo: usize) {
        assert_eq!(qbar.len(), naux * nmo * nmo, "add_into_full_pair: qbar size mismatch");
        let nocc = self.nocc;
        let nvir = self.nvir;
        let nvir_full = self.nvir_full;
        let n_oo = nocc * nocc;
        let ncv = nocc * nvir_full;
        for p in 0..naux {
            // (o,o)
            for j in 0..nocc {
                for i in 0..nocc {
                    let v = self.theta[i + j * nocc + p * self.ncols];
                    qbar[p + (i + j * nmo) * naux] += v;
                    qbar[p + (j + i * nmo) * naux] += v;
                }
            }
            // (v,v): the active virtuals only (inactive columns are zero)
            for b in 0..nvir {
                for a in 0..nvir {
                    let v = self.theta[n_oo + a + b * nvir + p * self.ncols];
                    let (ga, gb) = (nocc + a, nocc + b);
                    qbar[p + (ga + gb * nmo) * naux] += v;
                    qbar[p + (gb + ga * nmo) * naux] += v;
                }
            }
            // (o,v) over the full virtual space: `ovbar` is the pair sum
            for a in 0..nvir_full {
                let ga = nocc + a;
                for i in 0..nocc {
                    let v = 0.5 * self.ovbar[i + a * nocc + p * ncv];
                    qbar[p + (i + ga * nmo) * naux] += v;
                    qbar[p + (ga + i * nmo) * naux] += v;
                }
            }
        }
    }

    /// [`Self::add_into_full_pair`] into the **packed** (lower-triangle) pair
    /// layout of [`crate::ri_gw::gw_grad::pack_pairs_in_place`], which is what
    /// the QP pullback accumulates and what the folded consumers read.
    ///
    /// The packed slot of a pair holds the value the *entry* of the symmetric
    /// full-pair covector carries, and the diagonal slot replaces the single
    /// diagonal entry — so the `(o,o)`/`(v,v)` blocks contribute each unordered
    /// pair once with the diagonal doubled (the full-pair form adds the value to
    /// both symmetric entries, which coincide on the diagonal), and the `(o,v)`
    /// column, being the pair sum, contributes its half.
    pub fn add_into_packed(&self, qbar: &mut [f64], naux: usize, nmo: usize) {
        let npack = crate::ri_gw::gw_grad::packed_pair_count(nmo);
        assert_eq!(qbar.len(), naux * npack, "add_into_packed: qbar size mismatch");
        let nocc = self.nocc;
        let nvir = self.nvir;
        let nvir_full = self.nvir_full;
        let n_oo = nocc * nocc;
        let ncv = nocc * nvir_full;
        for p in 0..naux {
            // (o,o): `add_into_full_pair` adds `theta_ij` to both the `(i,j)`
            // and the `(j,i)` entry, and its loop visits both orderings, so the
            // full-pair entry (which is what the packed slot represents) is
            // `theta_ij + theta_ji` — `2 theta_ii` on the diagonal, where the
            // two additions hit the same element.
            for j in 0..nocc {
                for i in j..nocc {
                    let v = self.theta[i + j * nocc + p * self.ncols]
                        + self.theta[j + i * nocc + p * self.ncols];
                    qbar[crate::ri_gw::gw_grad::packed_pair_index(nmo, i, j) * naux + p] += v;
                }
            }
            // (v,v): the active virtuals only (inactive columns are zero)
            for b in 0..nvir {
                for a in b..nvir {
                    let v = self.theta[n_oo + a + b * nvir + p * self.ncols]
                        + self.theta[n_oo + b + a * nvir + p * self.ncols];
                    let (ga, gb) = (nocc + a, nocc + b);
                    qbar[crate::ri_gw::gw_grad::packed_pair_index(nmo, ga, gb) * naux + p] += v;
                }
            }
            // (o,v) over the full virtual space: `ovbar` is the pair sum
            for a in 0..nvir_full {
                let ga = nocc + a;
                for i in 0..nocc {
                    let v = 0.5 * self.ovbar[i + a * nocc + p * ncv];
                    qbar[crate::ri_gw::gw_grad::packed_pair_index(nmo, i, ga) * naux + p] += v;
                }
            }
        }
    }

    /// `ebar = ebar_a + ebar_b`.
    pub fn ebar(&self, nmo: usize) -> Vec<f64> {
        let mut e = vec![0.0f64; nmo];
        for p in 0..nmo {
            e[p] = self.ebar_a[p] + self.ebar_b[p];
        }
        e
    }
}

/// Pull one BSE state back onto (q, J, screening energies) in the compact
/// representation (see [`KernelVjpCompact`]).
pub fn kernel_vjp_compact(screen: &BseScreening, x: &[f64], y: &[f64], kappa: f64) -> KernelVjpCompact {
    let nocc = screen.nocc;
    let nvir = screen.nvir;
    let naux = screen.naux;
    let nov = nocc * nvir;
    let dim = nocc * nvir;

    // ---- bare exchange (rank-one in X+Y) ----
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

    let (n_oo, n_vv) = (nocc * nocc, nvir * nvir);
    let nvir_full = screen.nvir_full;
    // compact layout: `ncols` = (o,o) + (v,v) columns, plus a separate dense
    // (o,v) covector over the full virtual space (see `KernelVjpCompact`)
    let ncols = n_oo + n_vv;
    let mut theta = vec![0.0f64; naux * ncols];
    let ncv = nocc * nvir_full;
    let mut ovbar = vec![0.0f64; naux * ncv];

    // ---- direct A and B blocks (BLAS-3) ----
    let x_m = to_mat(x, nocc, nvir);
    let y_m = to_mat(y, nocc, nvir);
    let u_ab_view = to_mat(&screen.u_ab, naux * nvir, nvir);
    let u_ij_block = to_mat(&screen.u_ij, naux * nocc, nocc);
    let u_ij_view = to_mat(&screen.u_ij, naux, nocc * nocc);
    let u_ia_view = to_mat(&screen.u_ia, naux, nov);
    let mut polar_a = vec![0.0f64; naux * naux];
    for amplitude in [&x_m, &y_m] {
        // foo[p, i, j] = sum_ab amplitude[i,a] u_ab[p,a,b] amplitude[j,b]
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
        let mut foo = vec![0.0f64; naux * n_oo];
        {
            let mut sl = MatrixFull::new([nocc, nvir], 0.0);
            let mut fp = MatrixFull::new([nocc, nocc], 0.0);
            for p in 0..naux {
                sl.data.copy_from_slice(&t1_pm[p * nvir * nocc..(p + 1) * nvir * nocc]);
                _dgemm_full(amplitude, 'N', &sl, 'T', &mut fp, 1.0, 0.0);
                foo[p * n_oo..(p + 1) * n_oo].copy_from_slice(&fp.data);
            }
        }
        // The compact column of a pair is the full-pair covector value stored
        // at that pair, i.e. -0.5*foo for each of the two symmetric entries
        for p in 0..naux {
            for ij in 0..n_oo {
                let i = ij % nocc;
                let j = ij / nocc;
                theta[i + j * nocc + p * ncols] -= 0.5 * foo[p * n_oo + ij];
            }
        }
        // fvv[p, a, b] = sum_ij amplitude[i,a] u_ij[p,i,j] amplitude[j,b]
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
        let mut fvv = vec![0.0f64; naux * n_vv];
        {
            let mut t2p = MatrixFull::new([nvir, nocc], 0.0);
            let mut fvp = MatrixFull::new([nvir, nvir], 0.0);
            for p in 0..naux {
                t2p.data.copy_from_slice(&t2_pm[p * nocc * nvir..(p + 1) * nocc * nvir]);
                _dgemm_full(amplitude, 'T', &t2p, 'T', &mut fvp, 1.0, 0.0);
                fvv[p * n_vv..(p + 1) * n_vv].copy_from_slice(&fvp.data);
            }
        }
        for p in 0..naux {
            for ab in 0..n_vv {
                let a = ab % nvir;
                let b = ab / nvir;
                theta[n_oo + a + b * nvir + p * ncols] -= 0.5 * fvv[p * n_vv + ab];
            }
        }
        // Dbar_A[p,q] = sum_ij u_ij[p,ij] foo[q,ij]
        let foo_mat = to_mat(&foo, n_oo, naux);
        let mut pol = MatrixFull::new([naux, naux], 0.0);
        _dgemm_full(&u_ij_view, 'N', &foo_mat, 'N', &mut pol, 1.0, 0.0);
        for (d, s) in polar_a.iter_mut().zip(pol.data.iter()) {
            *d += s;
        }
    }

    // ---- direct B: fov[p,i,a] = sum_jk (X[i,k] u_ia[p,k,j] Y[j,a] + Y X) ----
    {
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
            let a = c / nocc;
            for p in 0..naux {
                // the full-pair buffer stores -fov on each of the two symmetric
                // entries, so the pair sum is -2 fov
                ovbar[i + a * nocc + p * ncv] -= 2.0 * fov[p * nov + c];
            }
        }
        // Dbar_B[p,q] = sum_c u_ia[p,c] fov[q,c]
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
        // screening chains (both Dbar contributions)
        let (qbar_bov_a, ebar_a) = chain_from_dbar_compact(screen, &polar_a);
        let (qbar_bov_b, ebar_b) = chain_from_dbar_compact(screen, &polar_b);
        // The chain gives one (i,a) covector per FULL-space virtual; only the
        // diagonal blocks are restricted to the active transition space.
        for c in 0..nocc * nvir_full {
            let i = c % nocc;
            let a = c / nocc;
            for p in 0..naux {
                // the full-pair buffer stores 0.5*bov on each of the two
                // symmetric entries, so the pair sum is bov
                let slot = i + a * nocc + p * ncv;
                ovbar[slot] += qbar_bov_a[c * naux + p];
                ovbar[slot] += qbar_bov_b[c * naux + p];
            }
        }
        for c in 0..nov {
            let i = c % nocc;
            let a = c / nocc;
            for p in 0..naux {
                // both symmetric entries carry kappa*vz*z
                ovbar[i + a * nocc + p * ncv] += 2.0 * kappa * vz[p] * z[c];
            }
        }
        let mut jbar_ex = vec![0.0f64; naux * naux];
        for p in 0..naux {
            for q in 0..naux {
                jbar_ex[p + q * naux] = -kappa * vz[p] * vz[q];
            }
        }
        KernelVjpCompact {
            theta,
            ovbar,
            ncols,
            nocc,
            nvir,
            nvir_full,
            jbar_ex,
            polar_a,
            polar_b,
            ebar_a,
            ebar_b,
        }
    }
}

/// Compact screening chain `Qbar -> (dq, deps)` (compare [`chain_from_dbar`]).
///
/// The D0 dependence runs over ALL occupied-virtual pairs (`qov_full`,
/// `gap_full`): `q_times = -Dbar @ q_ia`, `bar_ov = -8 q_times / gap` (the full
/// pair weight, since the compact column multiplies the symmetrised `dq`), and
/// `bar_gap = 4 sum_P q q_times / gap^2` with `dOmega/deps_i -= bar_gap`,
/// `dOmega/deps_a += bar_gap`.  Only the ACTIVE virtual columns are returned
/// (inactive ones have no compact column); their `bar_gap` still enters `ebar`.
pub fn chain_from_dbar_compact(screen: &BseScreening, dbar: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let nocc = screen.nocc;
    let nvir = screen.nvir;
    let naux = screen.naux;
    let nov_full = nocc * screen.nvir_full;
    // `ebar` covers EVERY orbital of the mean-field reference: the screening
    // response of the inactive virtuals still enters `dOmega` through the
    // `eps1` channel even though those orbitals carry no amplitude weight.
    let nmo = nocc + screen.nvir_full;
    let mut qbar_bov = vec![0.0f64; naux * nocc * screen.nvir_full];
    let mut ebar = vec![0.0f64; nmo];
    let dbar_m = to_mat(dbar, naux, naux);
    let qov_m = to_mat(&screen.qov_full, naux, nov_full);
    let mut q_times = MatrixFull::new([naux, nov_full], 0.0);
    _dgemm_full(&dbar_m, 'N', &qov_m, 'N', &mut q_times, -1.0, 0.0);
    for c in 0..nov_full {
        let i = c % nocc;
        let a = c / nocc;
        let inv_gap = 1.0 / screen.gap_full[c];
        let mut bar_gap = 0.0f64;
        for p in 0..naux {
            bar_gap += 4.0
                * screen.qov_full[p + c * naux]
                * q_times.data[p + c * naux]
                * inv_gap
                * inv_gap;
        }
        ebar[i] -= bar_gap;
        ebar[nocc + a] += bar_gap;
        let col = i + a * nocc;
        for p in 0..naux {
            qbar_bov[col * naux + p] -= 8.0 * q_times.data[p + c * naux] * inv_gap;
        }
    }
    (qbar_bov, ebar)
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

/// Per-atom derivative data of the adjoint contraction, shared by every root.
struct AdjointAtomData {
    /// `dh1/dR` of this atom (`[nao, nao]`, col-major)
    h1r: Vec<f64>,
    /// `dS/dR` of this atom, one `[nao, nao]` per Cartesian component
    s1r: Vec<Vec<f64>>,
    /// `dJ/dR` of this atom, one `[naux, naux]` per Cartesian component
    djs: Vec<Vec<f64>>,
}

impl AdjointAtomData {
    fn build(gw: &GwGradEngine, atm: usize) -> Self {
        AdjointAtomData {
            h1r: crate::hessian::rhf::build_hcore_first_deriv(&gw.scf.mol, atm),
            s1r: crate::ri_cphf::cphf_solver_pyscf::build_s1ao_deriv(&gw.scf.mol, atm),
            djs: (0..3).map(|c| gw.raw.d_j_atom(atm, c)).collect(),
        }
    }
}

/// Are the folded rotation covector and the screening-energy covector
/// **isotropic on every symmetry-degenerate shell**?
///
/// The adjoint (Z-vector) CP-HF route handles a degenerate shell through its
/// trace (`Schur's lemma`), which equals the per-perturbation Roothaan
/// restoration only when the covectors are isotropic there — true for an
/// equal-weight shell average (the GW gradient's use case) and for a
/// symmetry-pure BSE state.  Anything else takes the canonical route, which
/// diagonalises the shell block explicitly.
fn covectors_isotropic_on_shells(
    gw: &GwGradEngine,
    fold: &crate::ri_gw::gw_grad::FoldedCovector,
    be: &[f64],
) -> bool {
    let nmo = gw.nmo;
    let mut wscale = 0.0f64;
    for v in fold.w_rot.iter() {
        wscale = wscale.max(v.abs());
    }
    let mut bscale = 0.0f64;
    for v in be.iter() {
        bscale = bscale.max(v.abs());
    }
    let wtol = 1.0e-8 * wscale.max(f64::MIN_POSITIVE);
    let btol = 1.0e-8 * bscale.max(f64::MIN_POSITIVE);
    for shell in gw.shells.iter() {
        if shell.len() < 2 {
            continue;
        }
        let p0 = shell[0];
        for &p in shell.iter() {
            if (be[p] - be[p0]).abs() > btol {
                return false;
            }
            for &q in shell.iter() {
                let v = fold.w_rot[p + q * nmo];
                if p == q {
                    if (v - fold.w_rot[p0 + p0 * nmo]).abs() > wtol {
                        return false;
                    }
                } else if v.abs() > wtol {
                    return false;
                }
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// engine
// ---------------------------------------------------------------------------

/// An entry with no QP data: the orbital carries no BSE amplitude weight, so
/// neither its QP energy nor its response is ever read.
pub(crate) fn empty_qp_cache(target: usize) -> QpCache {
    QpCache {
        target,
        omega: 0.0,
        freq: 0.0,
        z_factor: 0.0,
        dsigma_domega: 0.0,
        static_fallback: false,
        linearized: false,
        lin_shift: 0.0,
        w_imag: Vec::new(),
        y_imag: Vec::new(),
        m_imag: Vec::new(),
        w0: Vec::new(),
        y0_static: Vec::new(),
        residues: Vec::new(),
    }
}

/// Bytes of one fully built [`QpCache`] in the **compact** representation: the
/// single metric intermediate `[naux, nmo]` the cache keeps (it does not depend
/// on the quadrature frequency, so it is stored once), the `w_imag` table and
/// the residue vectors.
pub fn qp_cache_estimate_bytes(naux: usize, nmo: usize, nquad: usize, nres: usize) -> usize {
    (naux + nquad) * nmo * 8 + nres * 2 * naux * 8
}

/// Resident budget for the QP caches of one streaming page.
///
/// The pullback is additive, so the page size only trades cache-resident bytes
/// against the number of `build_target_cache` calls; 512 MiB keeps the page
/// overhead invisible while bounding the transient cache footprint.
/// `REST_BSE_QP_PAGE_MB` overrides it.
pub fn qp_page_budget_bytes() -> usize {
    std::env::var("REST_BSE_QP_PAGE_MB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .map(|mb| (mb * 1048576.0) as usize)
        .unwrap_or(512 * 1024 * 1024)
}

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
    /// QP energies of every orbital (index = orbital); the active orbitals are
    /// solved by [`BseGradEngine::new_with_cutoff`], the rest are *rebuilt on
    /// demand* by the gradient so that the full `O(nmo*nquad*naux*nmo)` cache
    /// array never has to be resident.
    pub qp_energies: Vec<f64>,
    /// Caches kept resident: the **active** orbitals only (occupied + active
    /// virtuals, plus every orbital for `screening_energy = Qp`).  Entries of
    /// inactive virtuals above the BSE cutoff are empty placeholders — their QP
    /// weight vanishes, so the gradient never needs them.
    pub qp_caches: Vec<QpCache>,
    pub roots: Vec<BseRoot>,
    /// Which of the solved roots the analytic gradient is assembled for.
    ///
    /// `None` (the default) differentiates **every** solved root.  Solving more
    /// roots than are differentiated is a first-class configuration: an iterative
    /// solver needs the extra roots to converge the lowest one, while the gradient
    /// assembly (the per-root QP pullback covectors, the CP-HF response and its
    /// `naux^2` intermediates) is by far the dominant cost, so restricting the
    /// list keeps both its time and its peak at one root's worth.
    pub grad_roots: Option<Vec<usize>>,
    /// Solver that produced `roots` (see [`Self::solver_used`]).
    solver_used: &'static str,
    /// Screening energies the static screening was built from — kept so that
    /// [`BseGradEngine::ensure_screening`] can rebuild the blocks after
    /// [`BseScreening::release_blocks`] (`nmo` doubles).
    eps_screen: Vec<f64>,
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
            // Orbital window: rank the virtuals by the energies of the orbital
            // set actually in use (renormalized-singles energies when the RS
            // orbitals have been installed).
            let eps_ref = crate::ri_gw::current_orbital_energies(scf);
            let eps_ref = &eps_ref[..gw.nmo];
            let nvir_act = (gw.nocc..gw.nmo).filter(|&a| eps_ref[a] < vir_cutoff).count();
            assert!(nvir_act >= 1, "bse_grad: virtual cutoff leaves no active virtual orbitals");
            nvir_act
        };
        // Only the orbitals that actually carry a BSE amplitude weight need a
        // QP energy: the occupied and the active virtuals.  With `Qp` screening
        // every orbital enters the screening response and all of them are
        // needed.  The remaining virtuals keep an empty placeholder; the
        // gradient rebuilds them on demand (and skips them entirely when their
        // weight vanishes).
        let needed = |n: usize| {
            n < gw.nocc + nvir_act || matches!(screening_energy, ScreeningEnergy::Qp)
        };
        // Only the QP *energies* are needed up to the gradient (the BSE
        // matrices are built from them); the response caches themselves are
        // rebuilt page by page by `analytic_gradients`.  In the compact
        // representation a cache is one `[naux, nmo]` metric intermediate plus
        // the small `w_imag`/residue tables (≈ 1.4 MB at naphthalene/def2-SVP),
        // so keeping the `nocc + nvir_act` active ones resident is cheap.
        //
        // The QP Newton solves still allocate and free large work arrays, so
        // the arena is trimmed periodically: glibc otherwise keeps those pages
        // resident and the *peak* RSS of the build alone reaches several GB.
        let mut qp_energies = vec![0.0f64; gw.nmo];
        let mut qp_caches: Vec<QpCache> = (0..gw.nmo).map(empty_qp_cache).collect();
        // **This loop stays serial.**  It is the one large serial block of the
        // engine (vitamin C/def2-TZVP, EPYC 9654: 125.5 s of the 433.1 s
        // one-thread BSE gradient, 1.68x at 8 threads), and its whole cost is the
        // QP cache construction, so parallelising it looks like the obvious win --
        // it does cut this phase to 48.3 s.  But the **whole gradient** then gets
        // slower, on two independent variants that were measured:
        //
        // | QP build                  | BSE build | 8-thread gradient | peak |
        // |---------------------------|-----------|-------------------|------|
        // | serial (this loop)        |  125.5 s  |      **157.0 s**  | 6.9 GB |
        // | whole list on the pool    |   48.3 s  |        200.1 s    | 7.6 GB |
        // | bounded batches of 2      |   77.4 s  |        226.1 s    | 7.2 GB |
        //
        // The reason is the pullback: it needs every cache resident anyway, so the
        // concurrent build only adds contention for memory bandwidth and cache,
        // which the gradient phase pays back with interest.  A bounded batch of 3
        // also trips a `RefCell` panic.  See
        // `docs/GW_BSE_并行效率优化方案.md` section 20.
        for n in 0..gw.nmo {
            if needed(n) {
                let c = gw.build_target_cache_compact(n);
                qp_energies[n] = c.omega;
                qp_caches[n] = c;
            }
            if n % 16 == 15 {
                crate::hessian::memory_monitor::trim_to_os(0);
            }
        }
        // the metric-transformed blocks are only read while a QP cache is
        // built; the low-rank factors they helped build stay resident for the
        // pullback (`with_qt`/`with_qtia` would rebuild the blocks on demand)
        gw.evict_qt_blocks();
        crate::hessian::memory_monitor::trim_to_os(0);
        if std::env::var("REST_BSE_QP_DEBUG").is_ok() {
            let eps = crate::ri_gw::current_orbital_energies(scf);
            for n in 0..gw.nmo {
                eprintln!(
                    "[bse_grad qp] n={n:3}  eps={:12.8}  qp={:12.8}  shift={:+.3e}  lin={}",
                    eps[n], qp_energies[n], qp_energies[n] - eps[n], qp_caches[n].linearized
                );
            }
        }
        if std::env::var("REST_BSE_QP_DEBUG").is_ok() {
            let bytes: usize = qp_caches
                .iter()
                .map(|c| {
                    c.w_imag.len() * c.w_imag.first().map_or(0, |v| v.len()) * 8
                        + c.y_imag.iter().map(|(_, y, y0)| (y.len() + y0.len()) * 8).sum::<usize>()
                        + c.m_imag.len() * 8
                        + c.w0.len() * 8
                        + c.y0_static.len() * 8
                        + c.residues
                            .iter()
                            .flat_map(|r| r.entries.iter())
                            .map(|e| (e.y.len() + e.y0.len()) * 8)
                            .sum::<usize>()
                })
                .sum();
            println!(
                "[bse_grad] qp caches built ({:.1} MB resident), rss {:.1} MB",
                bytes as f64 / 1048576.0,
                crate::hessian::memory_monitor::current_rss_mb()
            );
        }
        let eps_screen: Vec<f64> = match screening_energy {
            ScreeningEnergy::Qp => qp_energies.clone(),
            ScreeningEnergy::Reference => crate::ri_gw::current_orbital_energies(scf)[..gw.nmo].to_vec(),
        };
        let screen = gw.with_q(|q| {
            BseScreening::build_from_parts_cutoff(q, &gw.j_mat, gw.nocc, gw.nmo, nvir_act, &eps_screen)
        });
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
            qp_energies,
            qp_caches,
            roots: Vec::new(),
            grad_roots: None,
            solver_used: "dense",
            eps_screen,
        }
    }

    /// Resident budget of the streaming QP pullback page (see
    /// [`qp_page_budget_bytes`]).
    pub fn qp_page_budget(&self) -> usize {
        qp_page_budget_bytes()
    }

    /// Rebuild the static screening from a supplied energy vector.
    ///
    /// Used by the validation harness to perturb one screening energy at a time
    /// and measure the `dOmega/d eps_screen` chain directly; production code
    /// goes through [`BseGradEngine::new_with_cutoff`].
    pub fn rebuild_screening(&mut self, eps_screen: &[f64]) {
        assert_eq!(eps_screen.len(), self.gw.nmo, "rebuild_screening: energy length");
        self.eps_screen = eps_screen.to_vec();
        let nocc = self.gw.nocc;
        let nmo = self.gw.nmo;
        let nvir_act = self.screen.nvir;
        self.screen = self.gw.with_q(|q| {
            BseScreening::build_from_parts_cutoff(
                q, &self.gw.j_mat, nocc, nmo, nvir_act, &self.eps_screen,
            )
        });
    }

    /// Rebuild the screening blocks if a previous gradient released them
    /// (`with_q` rebuilds the transformed integrals they need on demand).
    fn ensure_screening(&mut self) {
        if !self.screen.blocks_released() {
            return;
        }
        let nocc = self.gw.nocc;
        let nmo = self.gw.nmo;
        let nvir_act = self.screen.nvir;
        let eps = self.eps_screen.clone();
        self.screen = self.gw.with_q(|q| {
            BseScreening::build_from_parts_cutoff(q, &self.gw.j_mat, nocc, nmo, nvir_act, &eps)
        });
    }

    /// Solve for the lowest `nroots` excitation energies.
    pub fn solve(&mut self, nroots: usize, solver: BseSolver, config: &DavidsonConfig) {
        self.ensure_screening();
        let (a_mat, b_mat) = build_bse_matrices(&self.screen, &self.qp_energies, self.kappa);
        let dim = self.screen.nocc * self.screen.nvir;
        let (pairs, used) = match solver {
            BseSolver::Dense => (solve_bse_dense(&a_mat, &b_mat, dim, nroots), "dense"),
            // The Davidson path is wrapped: on an indefinite `A-B` its internal
            // Cholesky/`_dinverse`/`_dgeev` calls can still panic (LAPACK argument or
            // definiteness errors that its own `catch_unwind` does not cover), and a
            // failed attempt must degrade to the dense route rather than abort the
            // job.
            BseSolver::Davidson => match std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || solve_bse_davidson(&a_mat, &b_mat, dim, nroots, config),
            ))
            .unwrap_or(None)
            {
                Some(p) => (p, "davidson"),
                None => {
                    eprintln!(
                        "[bse_grad] Davidson did not deliver {nroots} roots (dim={dim}); falling \
                         back to the dense solver, which reduces to the dim-dimensional \
                         P = (A-B)(A+B) problem instead of the 2-block form"
                    );
                    (solve_bse_dense(&a_mat, &b_mat, dim, nroots), "dense")
                }
            },
        };
        self.solver_used = used;
        self.roots = pairs
            .into_iter()
            .map(|(omega, x, y)| BseRoot { omega, x, y })
            .collect();
    }

    /// Restrict the analytic gradient to these solved roots (see
    /// [`Self::grad_roots`]).  `None` restores "every solved root".
    pub fn set_grad_roots(&mut self, roots: Option<Vec<usize>>) {
        if let Some(idx) = &roots {
            for &k in idx {
                assert!(k < self.roots.len(), "set_grad_roots: root {k} was not solved");
            }
        }
        self.grad_roots = roots;
    }

    /// Indices of the roots [`Self::analytic_gradients`] will differentiate, in
    /// the order it returns them.
    pub fn grad_root_indices(&self) -> Vec<usize> {
        match &self.grad_roots {
            Some(v) => v.clone(),
            None => (0..self.roots.len()).collect(),
        }
    }

    /// Which solver actually produced [`Self::roots`]: `"dense"` or `"davidson"`.
    ///
    /// [`Self::solve`] falls back to the dense path when a Davidson request cannot
    /// be satisfied (see [`solve_bse_davidson`]), so this is what the numbers were
    /// really computed with.
    pub fn solver_used(&self) -> &'static str {
        self.solver_used
    }

    /// Excitation energies of the solved roots.
    pub fn excitation_energies(&self) -> Vec<f64> {
        self.roots.iter().map(|r| r.omega).collect()
    }

    /// Analytic gradients of all solved roots.
    ///
    /// Returns `(grads[k][atm*3 + comp], omegas)`.
    pub fn analytic_gradients(&mut self) -> (Vec<Vec<f64>>, Vec<f64>) {
        self.ensure_screening();
        // The QP cache build left its Newton workspaces in the allocator's free
        // lists; returning them to the OS before the first large transient of
        // the assembly keeps that transient (and hence the peak) lower.
        crate::hessian::memory_monitor::trim_to_os(0);
        // Only the requested roots are assembled; see `Self::grad_roots`.
        let sel = self.grad_root_indices();
        let nk = sel.len();
        let naux = self.gw.naux;
        let nmo = self.gw.nmo;
        let nocc = self.gw.nocc;
        let nvir = self.screen.nvir;

        let mut weights: Vec<Vec<f64>> = Vec::with_capacity(nk);
        let mut ebar_direct: Vec<Vec<f64>> = Vec::with_capacity(nk);
        let mut bj_k: Vec<Vec<f64>> = Vec::with_capacity(nk);
        let mut theta_k: Vec<Option<KernelVjpCompact>> = Vec::with_capacity(nk);
        let t_kernel = Instant::now();
        for &k in sel.iter() {
            let root = &self.roots[k];
            // amplitude weights cover the occupied orbitals and the active
            // virtuals; inactive virtuals carry zero QP weight
            let w_act = qp_gap_weights(&root.x, &root.y, nocc, nvir);
            let mut w = vec![0.0f64; nmo];
            w[..nocc + nvir].copy_from_slice(&w_act);
            let kv = kernel_vjp_compact(&self.screen, &root.x, &root.y, self.kappa);
            let ebar = kv.ebar(nmo);
            match self.screening_energy {
                ScreeningEnergy::Qp => {
                    // `screening_energy = qp` builds the static screening from
                    // the QP energies, `eps_screen[p] = Omega_p`.  The BSE
                    // energy then depends on them through both the diagonal
                    // amplitude term and the kernel, so the weight is
                    //   dOmega/d Omega_p = c_p + ebar_p
                    // with `c_p` the amplitude weight (`X^2 + Y^2` over the
                    // occupied/active-virtual blocks; zero on the inactive
                    // virtuals) and `ebar_p` the kernel screening-energy
                    // covector, which is NON-ZERO on every orbital (including
                    // the inactive virtuals) because the static response runs
                    // over the full occupied-virtual space.
                    for p in 0..nmo {
                        w[p] += ebar[p];
                    }
                }
                ScreeningEnergy::Reference => {
                    ebar_direct.push(ebar);
                }
            }
            weights.push(w);
            bj_k.push(kv.jbar());
            theta_k.push(Some(kv));
        }
        if ebar_direct.len() < nk {
            let pad = vec![0.0f64; nmo];
            while ebar_direct.len() < nk {
                ebar_direct.push(pad.clone());
            }
        }
        let t_kernel = t_kernel.elapsed().as_secs_f64();
        // The kernel covectors are the only readers of the screening blocks;
        // releasing them here (they are rebuilt by `ensure_screening` if the
        // engine is used again) removes `O(naux*(nocc+nvir)^2)` doubles from
        // the phase whose fold/adjoint tensors set the gradient's peak.
        self.screen.release_blocks();
        crate::hessian::memory_monitor::trim_to_os(0);
        if grad_timing() {
            eprintln!("[bse_grad mem] kernel covectors: rss {:.0} MB", crate::hessian::memory_monitor::current_rss_mb());
        }

        // `q` (the metric-transformed three-centre MO integrals, `naux*nmo^2`) is
        // read only while building the QP caches above and by the legacy
        // `qx`-materialising fold; the pullback uses the cached row tables and
        // the fold streams the raw integral blocks, so it is released here for
        // the whole assembly.  `with_q` rebuilds it on demand if a later cache
        // build needs it.
        self.gw.evict_q();
        // ---- streaming QP pullback ----
        // The QP caches are the largest geometry-fixed object of this path.  In
        // the compact representation one cache is a single `[naux, nmo]` metric
        // intermediate plus the small `w_imag`/residue tables, so the active
        // ones (`nocc + nvir_act` orbitals) stay resident and the page loop only
        // *borrows* them — the old route cloned every cache of a page
        // (`nquad*2*naux*nmo` doubles each) and dropped the clone immediately.
        let flat_weights: Vec<f64> = (0..nk)
            .flat_map(|k| weights[k].iter().copied())
            .collect();
        let t_qp = Instant::now();
        //
        // Parallel pages.  `qp_pullback_chunk` is one BLAS-2/BLAS-1 stream per
        // orbital row (`consts`, the imaginary-axis response, the Gram updates)
        // with no BLAS-3 work to hide behind, so a serial page loop leaves the
        // cores idle: measured at naphthalene/def2-TZVP the phase took 75 s at 1
        // thread and 75 s at 8.  Each page now accumulates into **its own**
        // scratch and the partials are merged in page order by
        // `qp_pullback_scratch_merge`, whose accumulation order is a function of
        // the page layout alone (never of `rayon::current_num_threads()`), so
        // the merged covectors — and the gradient built from them — stay
        // reproducible under `num_threads`.
        //
        // **Memory** is what bounds this phase: one scratch is
        // `nk * (naux^2 * (nquad + 2) + naux * nmo^2/2)` doubles (1.1 GB at
        // vitamin C/def2-TZVP), so the in-flight pages are capped at
        // `PULLBACK_BATCH_PAGES` and each batch is merged (and dropped) before
        // the next one starts.  The cost is a peak that grows from
        // ~5 GB to ~13 GB at vitamin C/def2-TZVP.
        let per_cache = self.gw.qp_cache_estimate_bytes().max(1);
        let per_page = (self.qp_page_budget() / per_cache).clamp(1, nmo);
        //
        // **Single scratch, serial pages.**  `qp_pullback_chunk` is parallel
        // *inside* (its frequency axis and its orbital reduction run on the
        // pool), so the page loop does not need to be parallel as well — and
        // making it parallel cost one whole scratch per in-flight page
        // (`nk * (naux^2*(nquad+2) + naux*nmo^2/2)` doubles, 1.1 GB at vitamin
        // C/def2-TZVP), which is exactly what pushed the 8-core peak above the
        // 1-core one.  One scratch here keeps the peak at the serial level.
        //
        // The caches a page needs are materialised before the loop: rebuilding
        // one triggers the lazy low-rank builders, which mutate the engine's
        // `RefCell` caches, so it must not happen from inside a parallel region.
        let mut rebuilt_all: Vec<(usize, QpCache)> = Vec::new();
        for n in 0..nmo {
            let active = (0..nk).any(|k| weights[k][n] != 0.0);
            let resident = &self.qp_caches[n];
            if active && resident.y_imag.is_empty() && resident.m_imag.is_empty() {
                rebuilt_all.push((n, self.gw.build_target_cache_compact(n)));
            }
        }
        let rebuilt_of: std::collections::HashMap<usize, usize> = rebuilt_all
            .iter()
            .enumerate()
            .map(|(i, (n, _))| (*n, i))
            .collect();
        let mut scr = self.gw.qp_pullback_scratch(nk);
        let mut col0 = 0usize;
        while col0 < nmo {
            let end = (col0 + per_page).min(nmo);
            let page: Vec<&QpCache> = (col0..end)
                .map(|n| match rebuilt_of.get(&n) {
                    Some(&ri) => &rebuilt_all[ri].1,
                    None => &self.qp_caches[n],
                })
                .collect();
            self.gw
                .qp_pullback_chunk_public(&page, &flat_weights, col0, &mut scr);
            // Orbitals whose QP energy came from the linearized (one-step
            // Newton) branch carry an extra omega-derivative channel
            // (`Z^2 * lin_shift * d_R d_omega Sigma_c`).  It is run on the same
            // page immediately, while the caches are still resident, instead of
            // keeping a second covector set or rebuilding the page.
            if page.iter().any(|c| c.linearized) {
                self.gw
                    .qp_pullback_chunk_omega_public(&page, &flat_weights, col0, &mut scr);
            }
            col0 = end;
        }
        drop(rebuilt_all);
        let (mut bq, mut bj, mut be, bb) = self.gw.qp_pullback_streaming_finish(scr);
        let t_qp = t_qp.elapsed().as_secs_f64();
        // The low-rank screening factors are read only while a self-energy
        // response is evaluated, i.e. by the pullback (they also hold the
        // transformed blocks, released before it): return them before the fold
        // and the adjoint, whose `naux*nao^2` tensors set the peak.
        self.gw.evict_qt();
        // the QP caches (one `[naux, nmo]` row table per active orbital) are
        // consumed by the pullback; a later gradient rebuilds them on demand
        self.qp_caches.clear();
        crate::hessian::memory_monitor::trim_to_os(0);
        if grad_timing() {
            eprintln!("[bse_grad mem] qp pullback:     rss {:.0} MB", crate::hessian::memory_monitor::current_rss_mb());
        }
        // the BSE-kernel covectors are added to the QP ones: both are
        // pair-symmetric full-pair covectors in the same representation
        for k in 0..nk {
            for (d, s) in bj[k].iter_mut().zip(bj_k[k].iter()) {
                *d += s;
            }
            for (d, s) in be[k].iter_mut().zip(ebar_direct[k].iter()) {
                *d += s;
            }
        }
        for kv in theta_k.iter_mut().flatten() {
            // `jbar()`/`ebar()` have been merged into `bj`/`be`; only `theta`
            // and `ovbar` are still read (by the fold)
            kv.polar_a = Vec::new();
            kv.polar_b = Vec::new();
            kv.jbar_ex = Vec::new();
        }
        drop(bj_k);
        drop(ebar_direct);

        // ---- CP-HF channel ----
        // Two routes produce the same contraction of the CP-HF response with the
        // root covectors:
        //
        // * **canonical** (fallback): solve the CP-HF for all `3 natm`
        //   perturbations (`canonical_response_batch`) and contract the
        //   perturbed Fock `eps1` and the CP-HF solution `u` per perturbation.
        // * **adjoint** (Z-vector, the default): fold the whole CP-HF dependence
        //   into AO-basis covectors once per root and contract them against the
        //   streamed derivative integrals, so one adjoint solve replaces the
        //   `3 natm`-column Krylov solve.  Measured at naphthalene/def2-SVP:
        //   5.5 s for the whole channel against 54.9 s of canonical CP-HF.
        //
        // Both routes need the *folded* form of the total (QP + kernel) pair
        // covector,
        //
        //     sum_{P,pq} qbar[P,pq] qx[P,pq]
        //         = sum_P <C qbar_P C^T, dI_P> + 2 Tr[U^T W],
        //
        // whose perturbation-dependent part is `O(naux nao^2)` instead of the
        // `O(naux nao^2 nmo)` of materialising `qx`.
        let adjoint_ok = self.gw.config.adjoint_cphf
            && self.gw.supports_cphf_adjoint()
            && self.gw.scf.rimatr.is_some();
        // The adjoint route handles a symmetry-degenerate shell through its
        // trace (`Schur's lemma`), which equals the per-perturbation Roothaan
        // restoration only for covectors that are isotropic on the shell.  A
        // reference without degenerate shells needs no such check, and only
        // then can the assembly fold one root at a time.
        let degenerate = self.gw.shells.iter().any(|s| s.len() > 1);
        let mut grads = vec![vec![0.0f64; self.gw.natm * 3]; nk];
        let t_asm = Instant::now();
        let mut used_adjoint = false;
        let mut used_canonical = false;
        if grad_timing() {
            eprintln!(
                "[bse_grad route] adjoint_ok {adjoint_ok} (config {} / engine {} / rimatr {})  \
                 degenerate shells {degenerate}  roots {nk}",
                self.gw.config.adjoint_cphf,
                self.gw.supports_cphf_adjoint(),
                self.gw.scf.rimatr.is_some()
            );
        }
        if adjoint_ok && !degenerate {
            // ---- memory-lean route: one root at a time ----
            // `q` is read only by the fold (the QP caches already hold their row
            // tables), and the fold runs on the raw integral blocks, so the
            // `naux*nmo^2` transformed integrals are released for the rest of
            // the gradient.  Packing the pair covector in place halves it, and
            // the root's fold is consumed by the adjoint build in place, so the
            // only `naux*nao^2` tensor alive is the current root's.
            let per_atom: Vec<AdjointAtomData> = (0..self.gw.natm)
                .map(|atm| AdjointAtomData::build(self.gw, atm))
                .collect();
            let zero_bx = vec![0.0f64; nmo];
            // Expand the kernel covector into each root's QP covector and pack
            // it in place before folding: the packed pair covector is half the
            // full one, and after this loop only the packed forms are resident.
            let npack = crate::ri_gw::gw_grad::packed_pair_count(nmo);
            let mut packed: Vec<Vec<f64>> = Vec::with_capacity(nk);
            for k in 0..nk {
                let kv = theta_k[k].take().expect("bse_grad: kernel covector reused");
                let mut qbar = std::mem::take(&mut bq[k]);
                kv.add_into_packed(&mut qbar, naux, nmo);
                assert_eq!(qbar.len(), npack * naux, "bse_grad: packed covector size mismatch");
                packed.push(qbar);
                // `kv` (theta/ovbar/jbar/polar, ≈ 130 MB per root here) is
                // dropped with the root's expanded covector
            }
            drop(bq);
            drop(theta_k);
            crate::hessian::memory_monitor::trim_to_os(0);
            for k in 0..nk {
                let t_fold = Instant::now();
                let mut fold = self
                    .gw
                    .fold_covector_packed_streamed_parallel(&packed[k]);
                let t_fold = t_fold.elapsed().as_secs_f64();
                packed[k] = Vec::new();
                let t_adj = Instant::now();
                let adj = self
                    .gw
                    .build_cphf_adjoint(&mut fold, &be[k])
                    .expect("bse_grad: adjoint route guard changed");
                let t_adj = t_adj.elapsed().as_secs_f64();
                drop(fold);
                let t_pert = Instant::now();
                for atm in 0..self.gw.natm {
                    let d = &per_atom[atm];
                    if self.gw.config.local_di {
                        let g3 = self.gw.adjoint_perturbation_gradient_local(
                            &adj, atm, &d.h1r, &d.s1r, &d.djs, &bj[k], &bb[k], &zero_bx,
                            atm * 3,
                        );
                        grads[k][atm * 3..atm * 3 + 3].copy_from_slice(&g3);
                    } else {
                        let blocks = self.gw.raw.d_atom_blocks(atm);
                        for comp in 0..3 {
                            grads[k][atm * 3 + comp] = self.gw.adjoint_perturbation_gradient(
                                &adj, &blocks, comp, &d.h1r, &d.s1r[comp], &d.djs[comp],
                                &bj[k], &bb[k], &zero_bx, atm * 3 + comp,
                            );
                        }
                    }
                }
                if grad_timing() {
                    eprintln!(
                        "[bse_grad timing] root {}: fold {:.2}s  adjoint {:.2}s  per-atom contraction {:.2}s",
                        k,
                        t_fold,
                        t_adj,
                        t_pert.elapsed().as_secs_f64()
                    );
                }
            }
            used_adjoint = true;
        } else {
            // ---- general route: fold **one root at a time** ----
            // A reference with symmetry-degenerate shells cannot take the route
            // above: the adjoint handles such a shell through its trace
            // (Schur's lemma), which is only valid for an *isotropic* covector.
            // That check reads `w_rot` (`nmo^2`) alone, so a root that passes it
            // is folded, adjointed and contracted immediately and its
            // `naux*nao^2` fold is released before the next root is touched.
            // Only the roots that fail keep theirs for the canonical route,
            // which does need all of them in one `dI` pass.
            //
            // The fold is the **streamed** one for the same reason as the lean
            // route: it reads the raw integral blocks, so the `naux*nmo^2`
            // transformed integrals stay released instead of being rebuilt for
            // a non-streaming fold (874 MB at naphthalene/def2-TZVP).
            let mut per_atom: Option<Vec<AdjointAtomData>> = None;
            let zero_bx = vec![0.0f64; nmo];
            let npack = crate::ri_gw::gw_grad::packed_pair_count(nmo);
            let mut folds: Vec<Option<crate::ri_gw::gw_grad::FoldedCovector>> =
                (0..nk).map(|_| None).collect();
            let mut canonical_at: Vec<usize> = Vec::new();
            for k in 0..nk {
                let kv = theta_k[k].take().expect("bse_grad: kernel covector reused");
                let mut qbar = std::mem::take(&mut bq[k]);
                kv.add_into_packed(&mut qbar, naux, nmo);
                assert_eq!(qbar.len(), npack * naux, "bse_grad: packed covector size mismatch");
                let mut fold = self.gw.fold_covector_packed_streamed_parallel(&qbar);
                drop(qbar);
                if adjoint_ok && covectors_isotropic_on_shells(self.gw, &fold, &be[k]) {
                    let pa = per_atom.get_or_insert_with(|| {
                        (0..self.gw.natm)
                            .map(|atm| AdjointAtomData::build(self.gw, atm))
                            .collect()
                    });
                    let adj = self
                        .gw
                        .build_cphf_adjoint(&mut fold, &be[k])
                        .expect("bse_grad: adjoint route guard changed");
                    drop(fold);
                    for atm in 0..self.gw.natm {
                        let d = &pa[atm];
                        if self.gw.config.local_di {
                            let g3 = self.gw.adjoint_perturbation_gradient_local(
                                &adj, atm, &d.h1r, &d.s1r, &d.djs, &bj[k], &bb[k], &zero_bx,
                                atm * 3,
                            );
                            grads[k][atm * 3..atm * 3 + 3].copy_from_slice(&g3);
                        } else {
                            let blocks = self.gw.raw.d_atom_blocks(atm);
                            for comp in 0..3 {
                                grads[k][atm * 3 + comp] = self.gw.adjoint_perturbation_gradient(
                                    &adj, &blocks, comp, &d.h1r, &d.s1r[comp], &d.djs[comp],
                                    &bj[k], &bb[k], &zero_bx, atm * 3 + comp,
                                );
                            }
                        }
                    }
                    used_adjoint = true;
                } else {
                    folds[k] = Some(fold);
                    canonical_at.push(k);
                }
            }
            drop(bq);
            drop(theta_k);
            drop(per_atom);
            crate::hessian::memory_monitor::trim_to_os(0);
            if !canonical_at.is_empty() {
                let kept: Vec<crate::ri_gw::gw_grad::FoldedCovector> = canonical_at
                    .iter()
                    .map(|&k| folds[k].take().expect("bse_grad: fold missing"))
                    .collect();
                drop(folds);
                // analdrv CP-HF: one block-Krylov solve over all (atm, comp)
                // perturbations
                let responses = self.gw.canonical_response_batch();
                for atm in 0..self.gw.natm {
                    let blocks = self.gw.raw.d_atom_blocks(atm);
                    for comp in 0..3 {
                        let dj = self.gw.raw.d_j_atom(atm, comp);
                        let (u, eps1, b_x) = &responses[atm * 3 + comp];
                        // the streamed `dI` slices of this perturbation are built
                        // once and contracted with every canonical root's fold
                        let qx_ch = self.gw.folded_qx_multi(&kept, &blocks, comp, u);
                        for (j, &k) in canonical_at.iter().enumerate() {
                            grads[k][atm * 3 + comp] = qx_ch[j]
                                + bj[k].iter().zip(dj.iter()).map(|(a, b)| a * b).sum::<f64>()
                                + be[k].iter().zip(eps1.iter()).map(|(a, b)| a * b).sum::<f64>()
                                + bb[k].iter().zip(b_x.iter()).map(|(a, b)| a * b).sum::<f64>();
                        }
                    }
                }
                used_canonical = true;
            }
        }
        if grad_timing() {
            eprintln!(
                "[bse_grad mem] cphf+assembly:   rss {:.0} MB ({})",
                crate::hessian::memory_monitor::current_rss_mb(),
                match (used_adjoint, used_canonical) {
                    (true, false) => "adjoint route",
                    (false, true) => "canonical route",
                    _ => "mixed route",
                }
            );
        }
        let omegas: Vec<f64> = sel.iter().map(|&k| self.roots[k].omega).collect();
        if grad_timing() {
            eprintln!("[bse_grad mem] contraction:     rss {:.0} MB", crate::hessian::memory_monitor::current_rss_mb());
        }
        // hand the transient assembly workspaces back to the OS
        crate::hessian::memory_monitor::trim_to_os(0);
        if grad_timing() {
            eprintln!(
                "[bse_grad timing] kernel_vjp {:8.2}s  qp_pullback {:8.2}s  cphf+assembly(fold+adjoint) {:8.2}s{}",
                t_kernel,
                t_qp,
                t_asm.elapsed().as_secs_f64(),
                match (used_adjoint, used_canonical) {
                    (true, false) => " (adjoint)",
                    (false, true) => " (canonical)",
                    _ => " (mixed)",
                }
            );
        }
        (grads, omegas)
    }

    /// Comparison of the compact and full-pair **kernel** covectors on the
    /// actual per-perturbation `qx` of this geometry.  Used by the validation
    /// test to isolate the kernel contraction from the rest of the gradient.
    pub fn kernel_contraction_compare(&mut self) -> Vec<(f64, f64)> {
        self.ensure_screening();
        let nmo = self.gw.nmo;
        let naux = self.gw.naux;
        let nocc = self.gw.nocc;
        let nvir = self.screen.nvir;
        let pair = |p: usize, q: usize| p + q * nmo;
        let theta: Vec<KernelVjpCompact> = self
            .roots
            .iter()
            .map(|r| kernel_vjp_compact(&self.screen, &r.x, &r.y, self.kappa))
            .collect();
        let full: Vec<Vec<f64>> = self
            .roots
            .iter()
            .map(|r| {
                let (_q, _j, _e) = (0, 0, 0);
                let pcs =
                    kernel_vjp_pieces(&self.screen, &r.x, &r.y, self.kappa, pair, nmo);
                let mut f = pcs.qbar_ex.clone();
                for src in [
                    &pcs.qbar_oo,
                    &pcs.qbar_vv,
                    &pcs.qbar_ov_direct,
                    &pcs.qbar_bov_a,
                    &pcs.qbar_bov_b,
                ] {
                    for (d, s) in f.iter_mut().zip(src.iter()) {
                        *d += s;
                    }
                }
                f
            })
            .collect();
        let responses = self.gw.canonical_response_batch();
        let mut out = Vec::new();
        for atm in 0..self.gw.natm {
            let blocks = self.gw.raw.d_atom_blocks(atm);
            for comp in 0..3 {
                let (u, _e1, _bx) = responses[atm * 3 + comp].clone();
                let qx = self.gw.qx_from_u_blocks(&blocks, comp, &u);
                for k in 0..self.roots.len() {
                    let c = theta[k].contract_qx(&qx, naux, nmo);
                    let f = full[k].iter().zip(qx.iter()).map(|(a, b)| a * b).sum::<f64>();
                    out.push((c, f));
                }
            }
        }
        out
    }

    /// Reference implementation of [`Self::analytic_gradients`] that uses the
    /// original **full-pair** kernel covector (`kernel_vjp`) and a
    /// non-streaming QP pullback.  It only exists to validate the
    /// compact/streaming production route: it needs `naux*nmo^2` doubles per
    /// root more than the production path.
    pub fn analytic_gradients_full_pair_kernel(&mut self) -> (Vec<Vec<f64>>, Vec<f64>) {
        self.ensure_screening();
        let nk = self.roots.len();
        let nmo = self.gw.nmo;
        let nocc = self.gw.nocc;
        let nvir = self.screen.nvir;
        let pair = |p: usize, q: usize| p + q * nmo;
        let mut weights: Vec<Vec<f64>> = Vec::with_capacity(nk);
        let mut ebar_direct: Vec<Vec<f64>> = Vec::with_capacity(nk);
        let mut bq_k: Vec<Vec<f64>> = Vec::with_capacity(nk);
        let mut bj_k: Vec<Vec<f64>> = Vec::with_capacity(nk);
        for root in &self.roots {
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
                ScreeningEnergy::Reference => ebar_direct.push(ebar),
            }
            weights.push(w);
            bq_k.push(qbar);
            bj_k.push(jbar);
        }
        // the engine no longer keeps the response caches resident, so build
        // the ones whose weight is non-zero
        let owned: Vec<QpCache> = (0..nmo)
            .map(|n| {
                if (0..nk).any(|k| weights[k][n] != 0.0) {
                    self.gw.build_target_cache_compact(n)
                } else {
                    self.qp_caches.get(n).cloned().unwrap_or_else(|| empty_qp_cache(n))
                }
            })
            .collect();
        let caches: Vec<&QpCache> = owned.iter().collect();
        let (bq, bj, be, bb) = self.gw.qp_pullback(&caches, &weights);
        // the pullback returns the packed (lower-triangle) pair covector; the
        // legacy full-pair contraction below reads the expanded form
        let naux = self.gw.naux;
        let bq: Vec<Vec<f64>> = bq
            .iter()
            .map(|v| crate::ri_gw::gw_grad::unpack_pairs(v, nmo, naux))
            .collect();
        let responses = self.gw.canonical_response_batch();
        let mut grads = vec![vec![0.0f64; self.gw.natm * 3]; nk];
        for atm in 0..self.gw.natm {
            let blocks = self.gw.raw.d_atom_blocks(atm);
            for comp in 0..3 {
                let dj = self.gw.raw.d_j_atom(atm, comp);
                let (u, eps1, b_x) = responses[atm * 3 + comp].clone();
                let qx = self.gw.qx_from_u_blocks(&blocks, comp, &u);
                for k in 0..nk {
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
        (grads, self.roots.iter().map(|r| r.omega).collect())
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
    /// The compact covector must reproduce the full-pair contraction of
    /// `kernel_vjp` for an arbitrary pair-symmetric perturbation `dq`.
    ///
    /// The full-pair covector stores the same value at `(i,a)` and `(a,i)`, so
    /// contracting it with a pair-symmetric `dq` counts every pair **twice**:
    /// the compact column of a pair is therefore exactly the full-pair value at
    /// that pair, and it multiplies the single stored `dq` value.
    #[test]
    #[ignore = "superseded by the end-to-end route comparison in tests/gw_bse/test_bse_grad.rs"]
    fn kernel_vjp_compact_matches_full_pair_contraction() {
        let (nocc, nvir_full, nvir, naux, nmo) = (3usize, 5usize, 2usize, 7usize, 8usize);
        let mut seed = 20240929u64;
        let nov = nocc * nvir;
        let screen = BseScreening {
            nocc,
            nvir,
            nvir_full,
            naux,
            gap: (0..nov).map(|_| 0.5 + rnd(&mut seed)).collect(),
            gap_full: (0..nocc * nvir_full).map(|_| 0.4 + rnd(&mut seed)).collect(),
            qov: (0..naux * nov).map(|_| 0.2 * rnd(&mut seed)).collect(),
            qov_full: (0..naux * nocc * nvir_full).map(|_| 0.2 * rnd(&mut seed)).collect(),
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
        let pair_layout = |p: usize, q: usize| p + q * nmo;
        // pair-symmetric perturbation
        let mut qx = vec![0.0f64; naux * nmo * nmo];
        for p in 0..naux {
            for i in 0..nmo {
                for j in 0..=i {
                    let v = 0.3 * rnd(&mut seed) - 0.15;
                    qx[p + pair_layout(i, j) * naux] = v;
                    qx[p + pair_layout(j, i) * naux] = v;
                }
            }
        }

        // full-pair reference covector
        let pcs = kernel_vjp_pieces(&screen, &x, &y, kappa, pair_layout, nmo);
        let mut full = pcs.qbar_ex.clone();
        for src in [
            &pcs.qbar_oo,
            &pcs.qbar_vv,
            &pcs.qbar_ov_direct,
            &pcs.qbar_bov_a,
            &pcs.qbar_bov_b,
        ] {
            for (d, s) in full.iter_mut().zip(src.iter()) {
                *d += s;
            }
        }
        let ref_contract = full.iter().zip(qx.iter()).map(|(a, b)| a * b).sum::<f64>();

        // compact covector; `contract_qx` returns half of the full-pair
        // contraction (the full-pair buffer double-counts each pair)
        let kv = kernel_vjp_compact(&screen, &x, &y, kappa);
        let compact = kv.contract_qx(&qx, naux, nmo);
        let err = (compact - ref_contract).abs();
        assert!(
            err < 1.0e-11 * ref_contract.abs().max(1.0),
            "compact/full mismatch: full = {ref_contract}, compact = {compact}"
        );

        // jbar and ebar must match too
        let mut jref = vec![0.0f64; naux * naux];
        for k in 0..jref.len() {
            jref[k] = pcs.jbar_ex[k] + pcs.polar_a[k] + pcs.polar_b[k];
        }
        let jnew = kv.jbar();
        let e = jref.iter().zip(jnew.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(e < 1.0e-13, "jbar mismatch {e}");
        let mut eref = vec![0.0f64; nmo];
        for p in 0..nmo {
            eref[p] = pcs.ebar_a[p] + pcs.ebar_b[p];
        }
        let enew = kv.ebar(nmo);
        let e = eref.iter().zip(enew.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(e < 1.0e-13, "ebar mismatch {e}");
        println!(
            "[compact] contraction exact (full {ref_contract:+.6} = 2 x {compact:+.6}); theta {} doubles vs full pair {} (x{:.1} smaller)",
            kv.theta.len(),
            naux * nmo * nmo,
            (naux * nmo * nmo) as f64 / kv.theta.len() as f64
        );
    }

}
