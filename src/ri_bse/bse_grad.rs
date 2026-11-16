//! Analytic nuclear gradients of static BSE excitation energies built on the
//! low-rank CD-G0W0 quasiparticle energies of [`crate::ri_gw::gw_grad`].
//!
//! Port of the validated PySCF implementation
//! `pyscf/gw/bse_grad_optimized.py` to REST.  The reference definition is
//!
//! ```text
//!     d Omega_S / dR = sum_p c_p^(S) dE_p/dR + d Omega_kernel^(S)/dR
//! ```
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

use crate::ri_gw::gw_grad::{CdQpCache, GwCdGradEngine, GwGradEngine, QpCache};
use crate::utilities::linalg::{gemm_nn, gemm_nt, to_mat};
use crate::scf_io::SCF;
use rayon::prelude::*;
use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;
use rest_tensors::matrix::matrix_blas_lapack::_dpotrf;
use rest_tensors::matrix::matrix_blas_lapack::_dsolve;
use rest_tensors::matrix::matrix_blas_lapack::_dtrtrs;
use rest_tensors::matrix::matrix_blas_lapack::_dsyevd;
use tensors::{MathMatrix, MatrixFull};
use tensors::matrix::BasicMatrix;

use crate::ri_bse::matvec;
use crate::ri_cphf::CPHFSolverPySCF;
use crate::solvers::davidson::{
    generate_initial_guess, lr_davidson_solver, tda_davidson_solver, DavidsonConfig,
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

/// Columns per task of a column-split gemm.
///
/// A fixed constant, not a function of the pool: the split is a scheduling decision, and the
/// result must not depend on it.
const GEMM_COL_BLK: usize = 256;

/// `out = alpha * a * b`, with `b` a flat col-major `[k, n]` buffer and the `n` **columns**
/// split across the pool.  Returns the col-major `[m, n]` result.
///
/// The three dominant BLAS-3 products of the BSE kernel VJP — `Dbar · q_ov`
/// ([`chain_from_dbar_compact`]), `u_ij · foo` and `u_ia · fov` ([`kernel_vjp_compact`]) — all
/// share one shape: a fixed left operand times a tall right-hand side, with **disjoint columns**.
/// Splitting the free index therefore needs no reduction and no partial accumulator, and the
/// contracted dimension `k` is never split, so every output element is produced by the same
/// expression as in the serial product.
///
/// This is not a micro-optimisation.  REST pins the process-wide BLAS to one thread (so that
/// OpenBLAS never oversubscribes the rayon pool), so before this a single `dgemm` of the form
/// `(naux x naux) · (naux x n_ov_full)` — `n_ov_full` is `nocc * nvir_full`, tens of thousands of
/// columns — ran on **one core of a 40-core node**.  Measured on polyene n=2 / def2-TZVPP on
/// XPK2, the `kernel_vjp + theta_k` phase took 8.59 s at one thread and **54.34 s at 40 threads**,
/// the single largest contributor to the gradient becoming *slower* with more cores.
fn gemm_nn_cols_par(a: &MatrixFull<f64>, b: &[f64], k: usize, n: usize, alpha: f64) -> Vec<f64> {
    let m = a.size()[0];
    debug_assert_eq!(a.size()[1], k, "gemm_nn_cols_par: inner dimensions disagree");
    let mut out = vec![0.0f64; m * n];
    if n == 0 || m == 0 {
        return out;
    }
    out.par_chunks_mut(GEMM_COL_BLK * m)
        .enumerate()
        .for_each(|(bi, dst)| {
            let c0 = bi * GEMM_COL_BLK;
            let nc = (n - c0).min(GEMM_COL_BLK);
            let bm = to_mat(&b[c0 * k..(c0 + nc) * k], k, nc);
            let mut om = MatrixFull::new([m, nc], 0.0);
            _dgemm_full(a, 'N', &bm, 'N', &mut om, alpha, 0.0);
            dst.copy_from_slice(&om.data);
        });
    out
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
/// The BSE **transition space** may be restricted to an active window (the
/// `bse_cutoff_energy` / `gradient_bse_nvir` approximation): the stored `qov`,
/// `qvv`, `v_ia`, `u_ab` and `u_ia` blocks are in the *active* layout (indices
/// counted inside the active window).
///
/// The **screening response** `D0 = J - Q0` is a different space: it runs over
/// every occupied-virtual pair of the untruncated mean-field reference
/// (`qov_full`/`gap_full`, `screen_nocc * nvir_full` columns), exactly like the
/// BSE spectrum path.  Truncating the transition space must not change `W`, nor
/// its derivative chain.
pub struct BseScreening {
    /// **active** occupied orbitals (`occ_start .. occ_start + nocc`).  Deep
    /// core orbitals below the `bse_cutoff_energy` window are not in the
    /// transition space, so every occupied block here is compacted over this
    /// window while the orbital index of active occupied `i` is
    /// `occ_start + i`.
    pub nocc: usize,
    /// first active occupied orbital (0 unless the window cuts the core)
    pub occ_start: usize,
    /// first active virtual orbital (the mean-field `nocc`, i.e. the HOMO+1);
    /// active virtual `a` is orbital `vir_start + a`
    pub vir_start: usize,
    /// active virtuals in the BSE transition space (<= nvir_full)
    pub nvir: usize,
    /// all virtual orbitals of the mean-field reference
    pub nvir_full: usize,
    /// first occupied orbital of the **screening response** (`mol.start_mo`, i.e.
    /// 0 unless a frozen core is excluded from the correlation treatment)
    pub screen_occ_start: usize,
    /// occupied orbitals entering the screening response.  This is the **full**
    /// occupied manifold of the mean-field reference, NOT the active transition
    /// window: the RPA response (and hence `D0`, `W` and its derivative chain) is
    /// built from every occupied-virtual pair of the untruncated space, exactly
    /// as the BSE spectrum path does (`ri_bse::construct_inverse_dielectric` uses
    /// `get_occupation_parameters(..., 'Y')`).  A truncated transition space only
    /// selects *which rows/columns of A/B are kept*; it must not change the
    /// screening.
    pub screen_nocc: usize,
    pub naux: usize,
    /// active screening gaps de_ia = eps_a - eps_i (nov, ia = i + a*nocc)
    pub gap: Vec<f64>,
    /// screening gaps over the untruncated space [screen_nocc*nvir_full]
    pub gap_full: Vec<f64>,
    /// active q_ia [naux, nov]
    pub qov: Vec<f64>,
    /// screening q_ia over the untruncated space [naux, screen_nocc*nvir_full]
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
        Self::build_from_parts_cutoff(q, j_mat, 0, nocc, nocc, nmo, nmo - nocc, eps_screen, 0)
    }

    /// Build the static screening with the BSE transition space restricted to
    /// the occupied orbitals `occ_start .. occ_start + nocc` and to the
    /// `nvir_act` lowest virtual orbitals starting at `vir_start` (the
    /// mean-field `nocc`, i.e. the HOMO+1).
    ///
    /// The transition space (`occ_start`, `nocc`, `nvir_act`) only decides which
    /// blocks are kept.  The static RPA response `Q0` -- hence `D0`, `W` and the
    /// whole `D0` derivative chain -- runs over **every** occupied-virtual pair of
    /// the untruncated mean-field space: occupied `screen_occ_start .. vir_start`
    /// times all virtuals `vir_start .. nmo`.  The `bse_cutoff_energy` window is a
    /// transition-space truncation and must not change the screening (the BSE
    /// spectrum path builds the same full-space response).
    pub fn build_from_parts_cutoff(
        q: &[f64],
        j_mat: &MatrixFull<f64>,
        occ_start: usize,
        nocc: usize,
        vir_start: usize,
        nmo: usize,
        nvir_act: usize,
        eps_screen: &[f64],
        screen_occ_start: usize,
    ) -> BseScreening {
        let naux = j_mat.size()[0];
        let nvir_full = nmo - vir_start;
        assert!(nocc >= 1 && occ_start + nocc <= vir_start, "bse_grad: invalid active occupied window");
        assert!(nvir_act >= 1 && nvir_act <= nvir_full, "bse_grad: invalid active virtual count");
        let pair = |p: usize, q_mo: usize| p + q_mo * nmo;

        // screening (response) space: every occupied orbital of the reference
        let screen_nocc = vir_start - screen_occ_start;
        assert!(screen_nocc >= nocc && occ_start >= screen_occ_start,
            "bse_grad: the screening response must span the whole occupied manifold");
        let screen_off = occ_start - screen_occ_start;
        let nov_full = screen_nocc * nvir_full;
        let nov = nocc * nvir_act;
        let mut qov_full = vec![0.0f64; naux * nov_full];
        for a in vir_start..nmo {
            for i in 0..screen_nocc {
                let src = pair(screen_occ_start + i, a) * naux;
                let dst = (i + (a - vir_start) * screen_nocc) * naux;
                qov_full[dst..dst + naux].copy_from_slice(&q[src..src + naux]);
            }
        }
        // active transition block (column `i + a*nocc`, `i` counted inside the window)
        let mut qov = vec![0.0f64; naux * nov];
        for a in 0..nvir_act {
            for i in 0..nocc {
                let src = ((screen_off + i) + a * screen_nocc) * naux;
                let dst = (i + a * nocc) * naux;
                qov[dst..dst + naux].copy_from_slice(&qov_full[src..src + naux]);
            }
        }
        let mut qoo = vec![0.0f64; naux * nocc * nocc];
        for j in 0..nocc {
            for i in 0..nocc {
                let src = pair(occ_start + i, occ_start + j) * naux;
                let dst = (i + j * nocc) * naux;
                qoo[dst..dst + naux].copy_from_slice(&q[src..src + naux]);
            }
        }
        let mut qvv = vec![0.0f64; naux * nvir_act * nvir_act];
        for b in 0..nvir_act {
            for a in 0..nvir_act {
                let src = pair(vir_start + a, vir_start + b) * naux;
                let dst = (a + b * nvir_act) * naux;
                qvv[dst..dst + naux].copy_from_slice(&q[src..src + naux]);
            }
        }
        let mut gap_full = vec![0.0f64; nov_full];
        let mut gap = vec![0.0f64; nov];
        for a in vir_start..nmo {
            for i in 0..screen_nocc {
                let g = eps_screen[a] - eps_screen[screen_occ_start + i];
                gap_full[i + (a - vir_start) * screen_nocc] = g;
            }
        }
        for a in 0..nvir_act {
            for i in 0..nocc {
                gap[i + a * nocc] = gap_full[(screen_off + i) + a * screen_nocc];
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
            occ_start,
            vir_start,
            nvir: nvir_act,
            nvir_full,
            screen_occ_start,
            screen_nocc,
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
        .map(|c| {
            qp_energies[screen.vir_start + ia_a[c]] - qp_energies[screen.occ_start + ia_i[c]]
        })
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

// ---------------------------------------------------------------------------
// matrix-free BSE solvers (the ones the BSE spectrum path uses)
// ---------------------------------------------------------------------------

/// The RI blocks consumed by [`crate::ri_bse::matvec`], in **exactly** the
/// layout the BSE spectrum path builds them in (`ri_bse::bse_both_spins`).
///
/// The gradient used to carry its own Davidson implementation and its own
/// matvec, expressed in the screening layout of [`BseScreening`].  That was a
/// second operator with its own algebra; the solvers below apply the *same*
/// operator the spectrum path applies, so a gradient card and a spectrum card
/// with the same keywords solve the same eigenvalue problem by construction.
struct BseMatvecTensors {
    /// `[naux*nvir, nvir]` virtual-virtual block (`q_vv`, reshaped)
    ri_vv: MatrixFull<f64>,
    /// `[naux, nocc*nvir]` occupied-virtual block (`q_ov`)
    ri_ov: MatrixFull<f64>,
    /// `[nocc*naux, nocc]` `D^-1 q_oo`, transposed into the A-block layout
    ri_oo_tilde: MatrixFull<f64>,
    /// `[naux*nocc, nvir]` `q_ov` reshaped for the B-block contraction
    ri_ov_b: MatrixFull<f64>,
    /// `[naux*nocc, nvir]` `D^-1 q_ov` reshaped for the B-block contraction
    ri_ov_tilde: MatrixFull<f64>,
    /// compact orbital energies `[nocc | nvir]` of the transition window
    energies: Vec<f64>,
}

impl BseMatvecTensors {
    fn bytes(&self) -> usize {
        8 * (self.ri_vv.data.len()
            + self.ri_ov.data.len()
            + self.ri_oo_tilde.data.len()
            + self.ri_ov_b.data.len()
            + self.ri_ov_tilde.data.len()
            + self.energies.len())
    }
}

/// Build the spectrum path's RI blocks for **this gradient's** transition
/// window and screening energies.
///
/// Returns `None` (and the caller keeps the dense solver) when the window the
/// spectrum path would build from the card's `bse_cutoff_energy` /
/// `bse_active_occ_start` / `bse_active_vir_end` keywords is not the window this
/// gradient actually differentiates — for example when the card pins a fixed
/// *count* of virtuals through `gradient_bse_nvir`.  Solving a different
/// operator than the one the pullback differentiates would silently produce
/// wrong gradients, so the mismatch is a fallback, never a silent substitution.
fn build_matvec_tensors(
    scf: &SCF,
    screen: &BseScreening,
    qp_energies: &[f64],
    eps_screen: &[f64],
) -> Option<BseMatvecTensors> {
    let (occ_start, nocc, vir_start, nvir) =
        (screen.occ_start, screen.nocc, screen.vir_start, screen.nvir);
    let nvir_full = screen.nvir_full;
    let naux = screen.naux;

    // The scripted (non-gradient) BSE path applies `bse_exchange_rescaling` to
    // the kernel blocks; the dense gradient matrices do not.  They coincide at
    // the default value, otherwise the two operators differ and this route must
    // not be taken.
    let rescaling = scf.mol.ctrl.quasiparticle_methods.as_ref()
        .map(|qp| qp.bse_exchange_rescaling).unwrap_or(1.0);
    if (rescaling - 1.0).abs() > 1.0e-12 {
        eprintln!(
            "[bse_grad] matrix-free BSE Davidson: bse_exchange_rescaling = {rescaling} is applied \
             by the spectrum path but not by the gradient's dense BSE matrices; this card keeps \
             the dense solver"
        );
        return None;
    }

    // Every block is sliced over **this gradient's** window, so the operator
    // below is the one the pullback differentiates no matter how the card's
    // `bse_cutoff_energy` / `gradient_bse_nvir` keywords compare.
    let occ_range = occ_start..vir_start;
    let vir_range = vir_start..vir_start + nvir;
    let vir_full_range = vir_start..vir_start + nvir_full;

    // `D0^-1` over the gradient's response pair space: the active occupied
    // window times *all* virtuals, dressed with the gradient's screening
    // energies (`response_matrix` takes a compact `[occ | vir]` energy vector).
    // The response space is the **untruncated** occupied manifold times all
    // virtuals (`screen.screen_nocc`), not the active transition window: that is
    // how `BseScreening` builds `D0` and how the spectrum path builds it.
    let screen_occ_start = screen.screen_occ_start;
    let screen_nocc = screen.screen_nocc;
    let mut eps_compact = Vec::with_capacity(screen_nocc + nvir_full);
    eps_compact.extend_from_slice(&eps_screen[screen_occ_start..screen_occ_start + screen_nocc]);
    eps_compact.extend_from_slice(&eps_screen[vir_start..vir_start + nvir_full]);
    let ri_ov_full = crate::ri_bse::get_submatrix_window(
        scf, 'O', 'V', screen_occ_start..vir_start, vir_full_range.clone(), 'Y');
    let response = crate::ri_gw::response_matrix(
        &eps_compact, screen_nocc, nvir_full, &ri_ov_full, 0.0, 'R', 0.0,
    );
    drop(ri_ov_full);
    let inverse_dielectric = crate::ri_gw::inverse_dielectric_matrix(response, 'R');
    assert_eq!(
        inverse_dielectric.size[0], naux,
        "bse_grad: the inverse dielectric of the response space has {} auxiliary functions, \
         the screening was built with a different auxiliary space",
        inverse_dielectric.size[0]
    );

    let ri_ov = crate::ri_bse::get_submatrix_window(
        scf, 'O', 'V', occ_range.clone(), vir_range.clone(), 'N');
    let ri_oo = crate::ri_bse::get_submatrix_window(
        scf, 'O', 'O', occ_range.clone(), occ_range, 'N');
    let mut ri_vv = crate::ri_bse::get_submatrix_window(
        scf, 'V', 'V', vir_range.clone(), vir_range.clone(), 'N');
    let mut ri_oo_tilde = MatrixFull::new(ri_oo.size, 0.0);
    _dgemm_full(&inverse_dielectric, 'N', &ri_oo, 'N', &mut ri_oo_tilde, 1.0, 0.0);
    drop(ri_oo);
    ri_oo_tilde.reshape([naux * nocc, nocc]);
    ri_oo_tilde = ri_oo_tilde.transpose_and_drop();
    ri_oo_tilde.reshape([nocc * naux, nocc]);
    ri_vv.reshape([naux * nvir, nvir]);
    let mut ri_ov_tilde = MatrixFull::new(ri_ov.size, 0.0);
    _dgemm_full(&inverse_dielectric, 'N', &ri_ov, 'N', &mut ri_ov_tilde, 1.0, 0.0);
    drop(inverse_dielectric);
    ri_ov_tilde.reshape([naux * nocc, nvir]);
    let mut ri_ov_b = ri_ov.clone();
    ri_ov_b.reshape([naux * nocc, nvir]);
    assert_eq!(
        (ri_ov.data.len(), ri_vv.data.len(), ri_oo_tilde.data.len()),
        (naux * nocc * nvir, naux * nvir * nvir, naux * nocc * nocc),
        "bse_grad: unexpected RI block sizes for the matrix-free BSE solver"
    );

    // The A-block diagonal is the QP gap of the transition window, i.e. the
    // energies the *gradient* differentiates (not the screening energies).
    let mut energies = Vec::with_capacity(nocc + nvir);
    energies.extend_from_slice(&qp_energies[occ_start..occ_start + nocc]);
    energies.extend_from_slice(&qp_energies[vir_start..vir_start + nvir]);

    let t = BseMatvecTensors {
        ri_vv,
        ri_ov,
        ri_oo_tilde,
        ri_ov_b,
        ri_ov_tilde,
        energies,
    };
    if grad_timing() {
        eprintln!(
            "[bse_mat timing] matrix-free RI blocks {:.1} MB",
            t.bytes() as f64 / 1048576.0
        );
    }
    Some(t)
}

/// `true` when the caller asked for the dense solver even where the matrix-free
/// one is available (`REST_BSE_FORCE_DENSE=1`).  Validation switch: running one
/// card twice must give the same `BSE_ROOT<n>` gradient.
fn force_dense_solver() -> bool {
    std::env::var("REST_BSE_FORCE_DENSE").map_or(false, |v| v != "0")
}

/// `true` when `REST_BSE_SOLVER_CHECK=1`: after a matrix-free solve, solve the
/// same card densely and print how far the two operators and their roots are
/// apart.  This is the equivalence proof between "the gradient's Davidson" and
/// the dense route; it costs one extra `dim^2` diagonalisation.
fn solver_check_requested() -> bool {
    std::env::var("REST_BSE_SOLVER_CHECK").map_or(false, |v| v != "0")
}

/// Compare a matrix-free solve with the dense one on the same card
/// (see [`solver_check_requested`]).
fn report_solver_check(
    screen: &BseScreening,
    qp_energies: &[f64],
    kappa: f64,
    roots: &[(f64, Vec<f64>, Vec<f64>)],
    matvec_a: &dyn Fn(&Vec<f64>) -> Vec<f64>,
    matvec_b: Option<&dyn Fn(&Vec<f64>) -> Vec<f64>>,
) {
    let dim = screen.nocc * screen.nvir;
    let (a_mat, b_mat) = build_bse_matrices(screen, qp_energies, kappa);
    // 1. operator identity on a deterministic pseudo-random vector
    let mut z = vec![0.0f64; dim];
    let mut state = 0x2545f4914f6cdd1du64;
    for v in z.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
    }
    let az = gemm_nn(&a_mat, dim, dim, &z, dim, 1);
    let mz = matvec_a(&z);
    let num = az.iter().zip(mz.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
    let den = az.iter().map(|a| a.abs()).fold(0.0f64, f64::max).max(1.0e-300);
    eprintln!(
        "[bse_grad check] |A_dense z - a_block_matvec_window(z)| / |A_dense z| = {:.3e}",
        num / den
    );
    if let Some(bmv) = matvec_b {
        let bz = gemm_nn(&b_mat, dim, dim, &z, dim, 1);
        let mzb = bmv(&z);
        let num = bz.iter().zip(mzb.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        let den = bz.iter().map(|a| a.abs()).fold(0.0f64, f64::max).max(1.0e-300);
        eprintln!(
            "[bse_grad check] |B_dense z - b_block_matvec_window(z)| / |B_dense z| = {:.3e}",
            num / den
        );
    }
    // 1b. the lowest eigenvalues of the dense A itself: a truncated static kernel
    // can make the TDA Hamiltonian indefinite, and that is what the positive-root
    // filter of both routes is for.
    {
        let (_, vals, _) = _dsyevd(&to_mat(&a_mat, dim, dim), 'N');
        let n_nonpos = vals.iter().filter(|v| **v <= 1.0e-10).count();
        let n_show = vals.len().min(6);
        eprintln!(
            "[bse_grad check] lowest eigenvalues of A: {}  ({} non-positive of {dim})",
            vals[..n_show].iter().map(|v| format!("{v:+.6}")).collect::<Vec<_>>().join(" "),
            n_nonpos
        );
    }
    // 2. root-by-root comparison with the dense solver.  Both equations are
    // checked for both routes: a vector that satisfies only one of them is not a
    // BSE eigenpair (the dense route's inverse-iteration refinement can drift to
    // a neighbouring eigenvector of the squared problem when two roots are
    // nearly degenerate).
    let dense = if matvec_b.is_some() {
        solve_bse_dense(&a_mat, &b_mat, dim, roots.len().max(1))
    } else {
        solve_bse_tda_dense(&a_mat, dim, roots.len())
    };
    let residuals = |omega: f64, x: &[f64], y: &[f64]| -> (f64, f64) {
        let ax = gemm_nn(&a_mat, dim, dim, x, dim, 1);
        let by = gemm_nn(&b_mat, dim, dim, y, dim, 1);
        let bx = gemm_nn(&b_mat, dim, dim, x, dim, 1);
        let ay = gemm_nn(&a_mat, dim, dim, y, dim, 1);
        let r1 = (0..dim).map(|r| (ax[r] + by[r] - omega * x[r]).abs()).fold(0.0f64, f64::max);
        let r2 = (0..dim).map(|r| (bx[r] + ay[r] + omega * y[r]).abs()).fold(0.0f64, f64::max);
        (r1, r2)
    };
    for (k, (omega, x, y)) in roots.iter().enumerate() {
        let (r1, r2) = residuals(*omega, x, y);
        let (od, dw, ov, dr1, dr2) = match dense.get(k) {
            Some((od, xd, yd)) => {
                let (p, q) = residuals(*od, xd, yd);
                let ov = (0..dim).map(|r| x[r] * xd[r] - y[r] * yd[r]).sum::<f64>();
                (*od, (od - omega).abs(), ov, p, q)
            }
            None => (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        };
        eprintln!(
            "[bse_grad check] root {k}: omega_dav = {omega:.10}  omega_dense = {od:.10}  \
             |d_omega| = {dw:.3e}  metric overlap = {ov:+.6}"
        );
        eprintln!(
            "[bse_grad check]          residual A: dav {r1:.2e} dense {dr1:.2e}   \
             residual B: dav {r2:.2e} dense {dr2:.2e}"
        );
    }
}
/// Residual of the matrix-free BSE equations for one root,
/// `(|r1| + |r2|) / max(1, omega)` with
/// `r1 = A X + B Y - omega X` and `r2 = B X + A Y + omega Y`
/// (`r2` is empty for a TDA root, where `B = Y = 0`).
fn matvec_residual(
    omega: f64,
    x: &[f64],
    y: &[f64],
    matvec_a: impl Fn(&Vec<f64>) -> Vec<f64>,
    matvec_b: Option<&dyn Fn(&Vec<f64>) -> Vec<f64>>,
) -> f64 {
    let dim = x.len();
    let xv = x.to_vec();
    let ax = matvec_a(&xv);
    let mut r1: Vec<f64> = ax.iter().zip(x).map(|(v, xi)| v - omega * xi).collect();
    let mut r2: Vec<f64> = Vec::new();
    if let Some(b) = matvec_b {
        let yv = y.to_vec();
        let by = b(&yv);
        let bx = b(&xv);
        let ay = matvec_a(&yv);
        for k in 0..dim {
            r1[k] += by[k];
        }
        r2 = (0..dim).map(|k| bx[k] + ay[k] + omega * y[k]).collect();
    }
    let n1 = r1.iter().map(|v| v * v).sum::<f64>().sqrt();
    let n2 = r2.iter().map(|v| v * v).sum::<f64>().sqrt();
    (n1 + n2) / 1.0f64.max(omega.abs())
}

/// Solve the **TDA** BSE problem with the matrix-free Davidson solver of the
/// BSE spectrum path ([`crate::ri_bse::matvec::a_block_matvec_window`] +
/// [`crate::solvers::davidson::tda_davidson_solver`]).
///
/// `y` is identically zero, as the TDA pullback expects.  Returns `None` when
/// the tensor build is not applicable or the solver does not deliver `nroots`
/// converged roots, so the caller can fall back to the dense route.
pub fn solve_bse_tda_davidson(
    scf: &SCF,
    screen: &BseScreening,
    qp_energies: &[f64],
    eps_screen: &[f64],
    kappa: f64,
    nroots: usize,
    config: &DavidsonConfig,
) -> Option<Vec<(f64, Vec<f64>, Vec<f64>)>> {
    let t = build_matvec_tensors(scf, screen, qp_energies, eps_screen)?;
    let (nocc, nvir) = (screen.nocc, screen.nvir);
    let dim = nocc * nvir;
    let qp_ctrl = scf.mol.ctrl.quasiparticle_methods.clone()
        .expect("bse_grad: the BSE solver needs the quasiparticle control block");
    let spin = if kappa == 0.0 { 'T' } else { 'S' };
    let energy_diag = crate::ri_bse::construct_energy_diag_for_a_window(
        &qp_energies.to_vec(), screen.occ_start, nocc, screen.vir_start, nvir,
    );
    let mut a = |z: &Vec<f64>| {
        matvec::a_block_matvec_window(
            &qp_ctrl, &t.energies, nocc, nvir, spin, &t.ri_vv, &t.ri_ov, &t.ri_oo_tilde, z,
        )
    };
    // A truncated TDA Hamiltonian can be **indefinite** (the screened kernel
    // `-W` of a long alkene makes `A` negative at strong coupling), and the
    // Davidson returns the algebraically lowest Ritz values, so a negative
    // eigenvalue would be reported as "root 0" -- the dense route drops them
    // (`_dsyevd` eigenvalues `> 1e-10`).  Ask for as many extra roots as there
    // are non-positive ones and keep the lowest **positive** roots, exactly what
    // `solve_bse_tda_dense` does.
    let mut nreq = nroots;
    let mut pairs;
    let mut n_dropped = 0usize;
    loop {
        let guess = generate_initial_guess(&energy_diag, nreq);
        let mut p = tda_davidson_solver(&mut a, nreq, &energy_diag, guess, config);
        p.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
        if p.len() != nreq {
            eprintln!(
                "[bse_grad] matrix-free TDA Davidson returned {} of {nreq} roots (dim={dim})",
                p.len()
            );
            return None;
        }
        let before = p.len();
        p.retain(|(omega, _)| *omega > 1.0e-10);
        n_dropped = before - p.len();
        if p.len() >= nroots {
            pairs = p;
            break;
        }
        if nreq >= dim {
            eprintln!(
                "[bse_grad] matrix-free TDA Davidson found only {} positive roots of {nroots} \
                 requested (dim={dim}); is the BSE kernel stable?",
                p.len()
            );
            return None;
        }
        nreq = (nreq + n_dropped.max(1) + 2).min(dim);
    }
    if n_dropped > 0 {
        println!(
            "[bse_grad] TDA Davidson: {n_dropped} non-positive eigenvalue(s) of A discarded \
             (indefinite truncated kernel); the lowest {nroots} positive roots are kept"
        );
    }
    pairs.truncate(nroots);
    // Reject unconverged Ritz pairs: the pullback would differentiate a vector
    // that is not the eigenvector the gradient formula assumes.  The bar is the
    // solver's own convergence test (`|r| <= sqrt(tol)`), so this only catches a
    // `max_iter` exit, not a root the solver legitimately converged.
    let bar = config.tol.sqrt().max(1.0e-9);
    for (omega, x) in &pairs {
        let res = matvec_residual(*omega, x, &vec![0.0f64; dim], &a, None);
        if !(res.is_finite() && res < bar) {
            eprintln!(
                "[bse_grad] matrix-free TDA Davidson root omega={omega:.10} failed the residual \
                 check ({res:.3e}); increase davidson_max_iter / davidson_converge_threshold"
            );
            return None;
        }
    }
    let out: Vec<(f64, Vec<f64>, Vec<f64>)> =
        pairs.into_iter().map(|(o, x)| (o, x, vec![0.0f64; dim])).collect();
    if solver_check_requested() {
        report_solver_check(screen, qp_energies, kappa, &out, &a, None);
    }
    Some(out)
}

/// Solve the **full** (non-TDA) BSE problem with REST's Casida Davidson solver
/// ([`crate::solvers::davidson::lr_davidson_solver`]) applied to the matrix-free
/// A/B blocks of the spectrum path.
///
/// The returned `(omega, X, Y)` are metric-normalised (`X^T X - Y^T Y = 1`),
/// which is the convention the gradient pullback uses.
pub fn solve_bse_lr_davidson(
    scf: &SCF,
    screen: &BseScreening,
    qp_energies: &[f64],
    eps_screen: &[f64],
    kappa: f64,
    nroots: usize,
    config: &DavidsonConfig,
) -> Option<Vec<(f64, Vec<f64>, Vec<f64>)>> {
    let t = build_matvec_tensors(scf, screen, qp_energies, eps_screen)?;
    let (nocc, nvir) = (screen.nocc, screen.nvir);
    let dim = nocc * nvir;
    let qp_ctrl = scf.mol.ctrl.quasiparticle_methods.clone()
        .expect("bse_grad: the BSE solver needs the quasiparticle control block");
    let spin = if kappa == 0.0 { 'T' } else { 'S' };
    let energy_diag = crate::ri_bse::construct_energy_diag_for_a_window(
        &qp_energies.to_vec(), screen.occ_start, nocc, screen.vir_start, nvir,
    );
    let guess = generate_initial_guess(&energy_diag, nroots);
    let mut a = |z: &Vec<f64>| {
        matvec::a_block_matvec_window(
            &qp_ctrl, &t.energies, nocc, nvir, spin, &t.ri_vv, &t.ri_ov, &t.ri_oo_tilde, z,
        )
    };
    let mut b = |z: &Vec<f64>| {
        matvec::b_block_matvec_window(
            nocc, nvir, spin, &t.ri_ov, &t.ri_ov_b, &t.ri_ov_tilde, z,
        )
    };
    let mut pairs = lr_davidson_solver(&mut a, &mut b, nroots, &energy_diag, guess, config);
    pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    if pairs.len() < nroots {
        eprintln!(
            "[bse_grad] matrix-free Casida Davidson returned {} of {nroots} roots (dim={dim})",
            pairs.len()
        );
        return None;
    }
    let mut out = Vec::with_capacity(nroots);
    for (omega, vec) in pairs.into_iter().take(nroots) {
        if vec.len() < 2 * dim {
            return None;
        }
        let mut x = vec[..dim].to_vec();
        let mut y = vec[dim..2 * dim].to_vec();
        let nrm = metric_norm(&x, &y, dim);
        if !(nrm.is_finite() && nrm > 1.0e-12) {
            eprintln!(
                "[bse_grad] matrix-free Casida Davidson root omega={omega:.10} has a non-positive \
                 metric norm ({nrm:.3e})"
            );
            return None;
        }
        for r in 0..dim {
            x[r] /= nrm;
            y[r] /= nrm;
        }
        let res = matvec_residual(omega, &x, &y, &a, Some(&b));
        if !(res.is_finite() && res < config.tol.sqrt().max(1.0e-9)) {
            eprintln!(
                "[bse_grad] matrix-free Casida Davidson root omega={omega:.10} failed the residual \
                 check ({res:.3e}); increase davidson_max_iter / davidson_converge_threshold"
            );
            return None;
        }
        out.push((omega, x, y));
    }
    if solver_check_requested() {
        report_solver_check(screen, qp_energies, kappa, &out, &a, Some(&b));
    }
    Some(out)
}

/// Solve the **TDA** (Hermitian) BSE problem densely.
///
/// With `A` symmetric the excitation energies are simply the eigenvalues of `A`
/// and the amplitudes are orthonormal, so this route needs no `A-B` Cholesky, no
/// `omega = sqrt(lambda)` and no metric normalisation `sqrt(U^T V)`.  That is
/// exactly what keeps it well defined where the full (non-TDA) problem has an
/// indefinite `A-B` -- the case that stalls the DA1/PBE0 relaxation and that the
/// Davidson solver answers with a per-iteration TDA fallback whose derivative
/// the non-TDA pullback cannot represent.  The returned `y` is identically zero,
/// which is what the downstream pullback expects for a TDA root.
pub fn solve_bse_tda_dense(
    a_mat: &[f64],
    dim: usize,
    nroots: usize,
) -> Vec<(f64, Vec<f64>, Vec<f64>)> {
    // `A` is symmetric by construction; symmetrise defensively against the
    // round-off asymmetry of the `v_bare` and kernel GEMMs.
    let mut m = to_mat(a_mat, dim, dim);
    for i in 0..dim {
        for j in (i + 1)..dim {
            let v = 0.5 * (m.data[i + j * dim] + m.data[j + i * dim]);
            m.data[i + j * dim] = v;
            m.data[j + i * dim] = v;
        }
    }
    let (vecs, vals, _info) = rest_tensors::matrix::matrix_blas_lapack::_dsyevd(&m, 'V');
    // The third tuple element of `_dsyevd` is the matrix dimension, not the
    // LAPACK `info`; validate the eigenvalues instead of trusting it.
    let vecs = vecs.expect("bse_grad: dsyevd returned no eigenvectors");
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "bse_grad: the TDA problem produced non-finite eigenvalues"
    );
    assert!(
        vals.windows(2).all(|w| w[0] <= w[1] + 1.0e-10),
        "bse_grad: dsyevd returned unsorted TDA eigenvalues"
    );
    let mut cands: Vec<usize> = (0..dim).filter(|&k| vals[k] > 1.0e-10).collect();
    cands.sort_by(|&a, &b| vals[a].partial_cmp(&vals[b]).unwrap_or(std::cmp::Ordering::Equal));
    assert!(
        cands.len() >= nroots,
        "bse_grad: the TDA problem has only {} positive roots (need {nroots})",
        cands.len()
    );
    cands
        .into_iter()
        .take(nroots)
        .map(|k| {
            let omega = vals[k];
            let mut u = vec![0.0f64; dim];
            let mut nrm = 0.0f64;
            for r in 0..dim {
                let v = vecs.data[r + k * dim];
                u[r] = v;
                nrm += v * v;
            }
            let inv = 1.0 / nrm.sqrt().max(1.0e-300);
            for v in u.iter_mut() {
                *v *= inv;
            }
            (omega, u, vec![0.0f64; dim])
        })
        .collect()
}

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
        if !(uv > 0.0) {
            eprintln!(
                "[bse_grad] dense BSE: root omega={omega:.6} has a non-positive metric norm \
                 U^T V = {uv:+.6e}; the amplitudes and their gradient are then NaN"
            );
        }
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
    /// first active occupied orbital and first active virtual orbital: the
    /// compact block indices are `i`/`a`, the *orbital* indices they stand for
    /// are `occ_start + i` and `vir_start + a` (see [`BseScreening`]).
    pub occ_start: usize,
    pub vir_start: usize,
    pub nvir: usize,
    /// All virtuals of the mean-field reference: the (o,v) block of `theta`
    /// spans them (the screening chain contributes for every virtual), so the
    /// square layout width is `nocc + nvir_full`.
    pub nvir_full: usize,
    /// Screening-response covector of the **core** occupied orbitals, i.e. the
    /// orbitals below the active window that the untruncated RPA response still
    /// contains (`occ_start - screen_occ_start` of them): layout
    /// `ovbar_core[i + a*ncore + P*ncore*nvir_full]` for the orbital pair
    /// `(screen_occ_start+i, vir_start+a)`.  Empty when the window starts at the
    /// first occupied orbital.
    pub ovbar_core: Vec<f64>,
    /// occupied orbitals in `ovbar_core` and the orbital its row `0` is
    /// (see [`BseScreening::screen_occ_start`]).
    pub ncore: usize,
    pub screen_occ_start: usize,
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
        let (os, vs) = (self.occ_start, self.vir_start);
        let mut acc = 0.0f64;
        // (o,o) and (v,v): `theta` mirrors the full-pair covector entry, and the
        // full-pair contraction counts both symmetric entries of a pair -> x2
        for i in 0..self.nocc {
            for j in 0..self.nocc {
                let base = ((os + i) + (os + j) * nmo) * naux;
                let mut s = 0.0f64;
                for p in 0..naux {
                    s += self.theta[i + j * self.nocc + p * self.ncols] * qx[base + p];
                }
                acc += 2.0 * s;
            }
        }
        for a in 0..self.nvir {
            for b in 0..self.nvir {
                let (ga, gb) = (vs + a, vs + b);
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
                let ga = vs + a;
                let base = ((os + i) + ga * nmo) * naux;
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
        let (os, vs) = (self.occ_start, self.vir_start);
        let n_oo = nocc * nocc;
        let ncv = nocc * nvir_full;
        let ncv_core = self.ncore * nvir_full;
        for p in 0..naux {
            // (o,o)
            for j in 0..nocc {
                for i in 0..nocc {
                    let v = self.theta[i + j * nocc + p * self.ncols];
                    qbar[p + ((os + i) + (os + j) * nmo) * naux] += v;
                    qbar[p + ((os + j) + (os + i) * nmo) * naux] += v;
                }
            }
            // (v,v): the active virtuals only (inactive columns are zero)
            for b in 0..nvir {
                for a in 0..nvir {
                    let v = self.theta[n_oo + a + b * nvir + p * self.ncols];
                    let (ga, gb) = (vs + a, vs + b);
                    qbar[p + (ga + gb * nmo) * naux] += v;
                    qbar[p + (gb + ga * nmo) * naux] += v;
                }
            }
            // (o,v) over the full virtual space: `ovbar` is the pair sum
            for a in 0..nvir_full {
                let ga = vs + a;
                for i in 0..nocc {
                    let v = 0.5 * self.ovbar[i + a * nocc + p * ncv];
                    qbar[p + ((os + i) + ga * nmo) * naux] += v;
                    qbar[p + (ga + (os + i) * nmo) * naux] += v;
                }
                // core occupied orbitals of the screening response
                for i in 0..self.ncore {
                    let gi = self.screen_occ_start + i;
                    let v = 0.5 * self.ovbar_core[i + a * self.ncore + p * ncv_core];
                    qbar[p + (gi + ga * nmo) * naux] += v;
                    qbar[p + (ga + gi * nmo) * naux] += v;
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
        let nocc = self.nocc;
        let nvir = self.nvir;
        // `qbar` may be stored banded (`min(p, q) < nrow`): every block written below lives in
        // the band `nrow = vir_start + nvir` (the largest orbital written is `vir_start+nvir-1`),
        // and the packed index of a banded covector is the same as in the full layout.
        let npack = crate::ri_gw::gw_grad::banded_pair_count(nmo, self.vir_start + nvir);
        assert!(qbar.len() >= naux * npack, "add_into_packed: qbar too small");
        let nvir_full = self.nvir_full;
        let (os, vs) = (self.occ_start, self.vir_start);
        let n_oo = nocc * nocc;
        let ncv = nocc * nvir_full;
        let ncv_core = self.ncore * nvir_full;
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
                    qbar[crate::ri_gw::gw_grad::packed_pair_index(nmo, os + i, os + j) * naux + p] += v;
                }
            }
            // (v,v): the active virtuals only (inactive columns are zero)
            for b in 0..nvir {
                for a in b..nvir {
                    let v = self.theta[n_oo + a + b * nvir + p * self.ncols]
                        + self.theta[n_oo + b + a * nvir + p * self.ncols];
                    let (ga, gb) = (vs + a, vs + b);
                    qbar[crate::ri_gw::gw_grad::packed_pair_index(nmo, ga, gb) * naux + p] += v;
                }
            }
            // (o,v) over the full virtual space: `ovbar` is the pair sum
            for a in 0..nvir_full {
                let ga = vs + a;
                for i in 0..nocc {
                    let v = 0.5 * self.ovbar[i + a * nocc + p * ncv];
                    qbar[crate::ri_gw::gw_grad::packed_pair_index(nmo, os + i, ga) * naux + p] += v;
                }
                // core occupied orbitals of the screening response
                for i in 0..self.ncore {
                    let gi = self.screen_occ_start + i;
                    let v = 0.5 * self.ovbar_core[i + a * self.ncore + p * ncv_core];
                    qbar[crate::ri_gw::gw_grad::packed_pair_index(nmo, gi, ga) * naux + p] += v;
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
    // TDA roots carry `y = 0` identically.  Every `y` leg of the RPA pullback
    // below is then exactly zero, so skip it instead of multiplying zeros: the
    // A-block loop runs once instead of twice and the whole direct-B block
    // (whose `fov` is a *cross* term `X^T u Y`) disappears.  Non-TDA roots are
    // untouched (`has_y` is true for them).
    let has_y = y.iter().any(|&v| v != 0.0);

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
    let ncore = screen.occ_start - screen.screen_occ_start;
    let mut ovbar_core = vec![0.0f64; naux * ncore * nvir_full];

    // ---- direct A and B blocks (BLAS-3) ----
    let x_m = to_mat(x, nocc, nvir);
    let y_m = to_mat(y, nocc, nvir);
    let u_ab_view = to_mat(&screen.u_ab, naux * nvir, nvir);
    let u_ij_block = to_mat(&screen.u_ij, naux * nocc, nocc);
    let u_ij_view = to_mat(&screen.u_ij, naux, nocc * nocc);
    let u_ia_view = to_mat(&screen.u_ia, naux, nov);
    let mut polar_a = vec![0.0f64; naux * naux];
    let amplitudes: [&MatrixFull<f64>; 2] = [&x_m, &y_m];
    for amplitude in amplitudes.iter().take(if has_y { 2 } else { 1 }) {
        let amplitude = *amplitude;
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
    // A pure cross term in (x, y): identically zero for a TDA root (`y = 0`).
    // Skipping it there needs the block restructured (it *is* the cross term and
    // its screening chain); the current TDA saving is the `y` pass of the
    // A-block above.
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
        // diagonal blocks are restricted to the active transition space.  Its
        // occupied index runs over the **screening** space (`screen_nocc`, which
        // includes the core orbitals below the active window), so every column is
        // routed either into the active block (`ovbar`) or into the core block
        // (`ovbar_core`).
        let snocc = screen.screen_nocc;
        let ncv_core = ncore * nvir_full;
        for c in 0..snocc * nvir_full {
            let i_screen = c % snocc;
            let a = c / snocc;
            let is_core = i_screen < ncore;
            for p in 0..naux {
                // the full-pair buffer stores 0.5*bov on each of the two
                // symmetric entries, so the pair sum is bov
                let v = qbar_bov_a[c * naux + p] + qbar_bov_b[c * naux + p];
                if is_core {
                    ovbar_core[i_screen + a * ncore + p * ncv_core] += v;
                } else {
                    ovbar[i_screen - ncore + a * nocc + p * ncv] += v;
                }
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
            occ_start: screen.occ_start,
            vir_start: screen.vir_start,
            nvir,
            nvir_full,
            ovbar_core,
            ncore,
            screen_occ_start: screen.screen_occ_start,
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
    let nocc = screen.screen_nocc;
    let nvir = screen.nvir;
    let naux = screen.naux;
    let nov_full = nocc * screen.nvir_full;
    // `ebar` covers EVERY orbital of the mean-field reference: the screening
    // response of the inactive virtuals still enters `dOmega` through the
    // `eps1` channel even though those orbitals carry no amplitude weight.
    let nmo = screen.vir_start + screen.nvir_full;
    let (os, vs) = (screen.screen_occ_start, screen.vir_start);
    let mut qbar_bov = vec![0.0f64; naux * nocc * screen.nvir_full];
    let mut ebar = vec![0.0f64; nmo];
    let dbar_m = to_mat(dbar, naux, naux);
    // `q_times = -Dbar @ q_ov`, column-split across the pool (`q_ov_full` has
    // `nocc * nvir_full` columns, so the serial product was one core's worth of BLAS-3).
    let q_times = gemm_nn_cols_par(&dbar_m, &screen.qov_full, naux, nov_full, -1.0);
    for c in 0..nov_full {
        let i = c % nocc;
        let a = c / nocc;
        let inv_gap = 1.0 / screen.gap_full[c];
        let mut bar_gap = 0.0f64;
        for p in 0..naux {
            bar_gap += 4.0
                * screen.qov_full[p + c * naux]
                * q_times[p + c * naux]
                * inv_gap
                * inv_gap;
        }
        ebar[os + i] -= bar_gap;
        ebar[vs + a] += bar_gap;
        let col = i + a * nocc;
        for p in 0..naux {
            qbar_bov[col * naux + p] -= 8.0 * q_times[p + c * naux] * inv_gap;
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
    let nocc = screen.screen_nocc;
    let naux = screen.naux;
    let nov = nocc * screen.nvir_full;
    let mut qbar_bov = vec![0.0f64; naux * nmo * nmo];
    let mut ebar = vec![0.0f64; nmo];
    // q_times[p + c*naux] = -sum_q dbar[p + q*naux] qov_full[q + c*naux]
    let dbar_m = to_mat(dbar, naux, naux);
    // column-split across the pool, as in `chain_from_dbar_compact`
    let q_times = gemm_nn_cols_par(&dbar_m, &screen.qov_full, naux, nov, -1.0);
    for c in 0..nov {
        let i = screen.screen_occ_start + c % nocc;
        let a = screen.vir_start + c / nocc;
        let inv_gap = 1.0 / screen.gap_full[c];
        let col = q_pair_layout(i, a) * naux;
        let ctrans = q_pair_layout(a, i) * naux;
        let mut bar_gap = 0.0f64;
        for p in 0..naux {
            let bov = -8.0 * q_times[p + c * naux] * inv_gap;
            qbar_bov[col + p] += 0.5 * bov;
            qbar_bov[ctrans + p] += 0.5 * bov;
            bar_gap += 4.0 * screen.qov_full[p + c * naux] * q_times[p + c * naux] * inv_gap * inv_gap;
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
    // compact block indices -> orbital indices (see `BseScreening`)
    let (os, vs) = (screen.occ_start, screen.vir_start);
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
        let i = os + c % nocc;
        let a = vs + c / nocc;
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
                let i = os + ij % nocc;
                let j = os + ij / nocc;
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
                let a = vs + ab % nvir;
                let b = vs + ab / nvir;
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
        let a = vs + c / nocc;
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
        ac: crate::ri_gw::gw_grad::AcQpData::default(),
        wac: None,
    }
}

/// Placeholder for an inactive full-CD QP row; never read by the pullback.
pub(crate) fn empty_cd_qp_cache(target: usize) -> CdQpCache {
    CdQpCache {
        target,
        omega: 0.0,
        freq: 0.0,
        z_factor: 1.0,
        dsigma_domega: 0.0,
        static_fallback: false,
        linearized: false,
        lin_shift: 0.0,
        w_imag: Vec::new(),
        y_imag: Vec::new(),
        w0: Vec::new(),
        y0_static: Vec::new(),
        y0_metric: Vec::new(),
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

/// Linear-response **stability** of the BSE reference state, i.e. the spectra of
/// the two symmetric blocks of the BSE Hamiltonian `[[A, B], [B, A]]`.
///
/// Both solvers depend on their definiteness, silently:
///
/// * the Davidson route factorises `A - B` with a Cholesky (the Casida form is
///   only defined for a positive-definite `A - B`), and
/// * the dense route forms `P = (A - B)(A + B)` and inverse-iterates through
///   `P - omega^2`.
///
/// With `A = diag(gap_qp) + kappa*v_bare - W_A` and `B = kappa*v_bare - W_B`
/// the resonant difference is
///
/// ```text
/// A - B = diag(gap_qp) - (W_A - W_B),
/// ```
///
/// so when the screened direct kernel overwhelms the quasiparticle gap -- which
/// is what the longer linear acenes do -- `A - B` acquires a negative
/// eigenvalue, the reference state is triplet-unstable and *neither* route is
/// valid.  The smallest eigenvalue of `A - B` is therefore the quantity to
/// watch before committing hours to a gradient, and [`BseGradEngine::stability`]
/// computes it from the same matrices `solve` would build.
pub struct BseStability {
    /// Dimension of the transition space (`nocc * nvir`).
    pub dim: usize,
    /// Smallest quasiparticle gap on the diagonal of `A` (Ha).
    pub gap_min: f64,
    /// Eigenvalues of `A - B`, ascending (Ha of the BSE Hamiltonian; `A - B` is
    /// the block whose smallest eigenvalue must stay positive).
    pub amb: Vec<f64>,
    /// Eigenvalues of `A + B`, ascending.
    pub apb: Vec<f64>,
}

impl BseStability {
    /// Smallest eigenvalue of `A - B`.
    pub fn amb_min(&self) -> f64 {
        self.amb.first().copied().unwrap_or(f64::NAN)
    }
    /// Smallest eigenvalue of `A + B`.
    pub fn apb_min(&self) -> f64 {
        self.apb.first().copied().unwrap_or(f64::NAN)
    }
    /// Number of negative eigenvalues of `A - B`.
    pub fn amb_negative(&self) -> usize {
        self.amb.iter().filter(|w| **w < 0.0).count()
    }
    /// Whether both blocks are positive definite, i.e. whether the reference
    /// state is stable and both BSE solvers are applicable.
    pub fn stable(&self) -> bool {
        self.amb_min() > 0.0 && self.apb_min() > 0.0
    }
}

/// The static-BSE analytic-gradient engine for one geometry.
pub struct BseGradEngine<'a> {
    pub gw: &'a GwGradEngine<'a>,
    /// Full-CD QP provider when `gw_variant = "cd"`/`"fscd"`; `None` for AC/WAC.
    pub cd: Option<&'a GwCdGradEngine<'a>>,
    /// Full-CD QP rows for the active orbitals (empty for AC/WAC).
    pub cd_qp_caches: Vec<CdQpCache>,
    pub screen: BseScreening,
    pub kappa: f64,
    /// Solve (and differentiate) the **TDA** problem instead of the full
    /// particle-hole RPA one.  `A` is symmetric, so the roots are ordinary
    /// eigenvalues and the amplitudes orthonormal (`y = 0`); the energy path's
    /// `bse_tda` keyword selects the same approximation on its side, so the two
    /// stages stay on one functional.
    pub tda: bool,
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
    /// Energy-degenerate BSE root manifolds (indices into [`Self::roots`]),
    /// detected by [`Self::finalize_degenerate_manifolds`] right after a solve.
    ///
    /// Inside a manifold the individual eigenvectors are only defined up to a
    /// rotation, so the Hellmann-Feynman contraction of each member is
    /// basis-dependent.  The members are metric-orthonormalised at detection
    /// time and [`Self::analytic_gradients`] replaces their gradients by the
    /// manifold average — the derivative of the mean manifold energy, which is
    /// the basis-invariant (and physically meaningful) quantity as long as the
    /// degeneracy is not split by the displacement.
    pub degenerate_groups: Vec<Vec<usize>>,
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
        Self::new_with_window(
            scf, gw, None, multi, screening_energy, f64::INFINITY, f64::NEG_INFINITY,
        )
    }

    /// [`Self::new_with_window`] with the full occupied manifold and a virtual
    /// cutoff only (the historical signature).
    pub fn new_with_cutoff(
        scf: &'a SCF,
        gw: &'a GwGradEngine<'a>,
        multi: char,
        screening_energy: ScreeningEnergy,
        vir_cutoff: f64,
    ) -> Self {
        Self::new_with_window(scf, gw, None, multi, screening_energy, vir_cutoff, f64::NEG_INFINITY)
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
    pub fn new_with_window(
        scf: &'a SCF,
        gw: &'a GwGradEngine<'a>,
        cd: Option<&'a GwCdGradEngine<'a>>,
        multi: char,
        screening_energy: ScreeningEnergy,
        vir_cutoff: f64,
        occ_cutoff: f64,
    ) -> Self {
        let nvir_full = gw.nmo - gw.nocc;
        // Orbital window: rank both manifolds by the energies of the orbital
        // set actually in use (renormalized-singles energies when the RS
        // orbitals have been installed).  `vir_cutoff` bounds the virtuals from
        // above (`e_LUMO + x`), `occ_cutoff` bounds the occupied orbitals from
        // below (`e_HOMO - x`); the pair of them is the `bse_cutoff_energy`
        // window that `ri_gw::bse_active_bounds` implements for the
        // non-gradient BSE path.
        let (occ_start, nocc_act, nvir_act) = if vir_cutoff.is_infinite()
            && occ_cutoff.is_infinite()
        {
            (0, gw.nocc, nvir_full)
        } else {
            let eps_ref = crate::ri_gw::current_orbital_energies(scf);
            let eps_ref = &eps_ref[..gw.nmo];
            let vir_start = gw.nocc;
            let nvir_act = if vir_cutoff.is_infinite() {
                nvir_full
            } else {
                (vir_start..gw.nmo).filter(|&a| eps_ref[a] < vir_cutoff).count()
            };
            assert!(nvir_act >= 1, "bse_grad: virtual cutoff leaves no active virtual orbitals");
            // The HOMO itself always stays in the window: dropping it would
            // leave no occupied orbital to excite from.
            let occ_start = if occ_cutoff.is_infinite() {
                0
            } else {
                (0..gw.nocc)
                    .find(|&i| eps_ref[i] >= occ_cutoff)
                    .unwrap_or(gw.nocc - 1)
                    .min(gw.nocc - 1)
            };
            (occ_start, gw.nocc - occ_start, nvir_act)
        };
        let vir_start = gw.nocc;
        if occ_start != 0 || nocc_act != gw.nocc {
            println!(
                "    [bse window] active occupied #{}..#{} ({} of {}), active virtuals #{}..#{} \
                 ({} of {}), dim {}",
                occ_start, vir_start - 1, nocc_act, gw.nocc,
                vir_start, vir_start + nvir_act - 1, nvir_act, nvir_full,
                nocc_act * nvir_act
            );
        }
        // Only the orbitals that actually carry a BSE amplitude weight need a
        // QP energy: the occupied and the active virtuals.  With `Qp` screening
        // every orbital enters the screening response and all of them are
        // needed.  The remaining virtuals keep an empty placeholder; the
        // gradient rebuilds them on demand (and skips them entirely when their
        // weight vanishes).
        let needed = |n: usize| {
            (n >= occ_start && n < vir_start + nvir_act)
                || matches!(screening_energy, ScreeningEnergy::Qp)
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
        let mut cd_qp_caches: Vec<CdQpCache> = if cd.is_some() {
            (0..gw.nmo).map(empty_cd_qp_cache).collect()
        } else {
            Vec::new()
        };
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
        // BLAS threads for the serial build below.  `build_target_cache_compact`
        // is pure BLAS/LAPACK: per orbital it factorises one `naux x naux` chi0
        // per frequency node (20 nodes, `naux = 1234` at azobenzene/def2-TZVPP),
        // so the whole loop is thousands of dense `dsyevd` calls with no rayon
        // parallelism to fall back on.  The *process-wide* OpenBLAS thread count
        // is what decides how fast they run, and because this library is the
        // pthread build (`libopenblasp`, verified via /proc/<pid>/maps) the
        // `omp_set_num_threads(1)` guards elsewhere in REST do **not** reach it --
        // only `openblas_set_num_threads` does, which
        // `omp_set_num_threads_global_wrapper` sets together with the OpenMP
        // one.  A launcher that exports `OPENBLAS_NUM_THREADS=1` (the historical
        // advice, to keep BLAS from oversubscribing the rayon regions) would
        // otherwise pin this loop to a single core: measured 120% CPU of 1600%
        // available, and an excited-state gradient that takes hours instead of
        // minutes.  Restore whatever the caller had on the way out so the
        // phase-local policy of the surrounding code is preserved.
        let blas_threads_saved =
            crate::utilities::rstsr_util::blas_get_num_threads();
        let blas_threads_want = scf
            .mol
            .ctrl
            .num_threads
            .unwrap_or_else(|| rayon::current_num_threads())
            .max(1);
        rest_tensors::matrix_blas_lapack::omp_set_num_threads_global_wrapper(blas_threads_want);
        let mut n_qp_built = 0usize;
        for n in 0..gw.nmo {
            if needed(n) {
                if let Some(cd_engine) = cd {
                    let c = cd_engine.build_target_cache(n);
                    qp_energies[n] = c.omega;
                    cd_qp_caches[n] = c;
                } else {
                    let c = gw.build_target_cache_compact(n);
                    qp_energies[n] = c.omega;
                    qp_caches[n] = c;
                }
                n_qp_built += 1;
            }
            if n % 16 == 15 {
                crate::hessian::memory_monitor::trim_to_os(0);
            }
        }
        // Cost evidence for the `bse_cutoff_energy` window: the QP (GW) solve is
        // the expensive part of the engine, and only the orbitals that carry a
        // BSE amplitude weight need one.  Without the window this is `nmo`.
        println!(
            "[bse_grad] GW quasiparticle solves: {n_qp_built} of {} orbitals \
             (active {nocc_act} occupied + {nvir_act} virtual){}",
            gw.nmo,
            if occ_start == 0 && nocc_act == gw.nocc && vir_start + nvir_act == gw.nmo {
                ""
            } else {
                " -- deep/inactive orbitals skipped"
            }
        );
        rest_tensors::matrix_blas_lapack::omp_set_num_threads_global_wrapper(blas_threads_saved);
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
                    eps[n], qp_energies[n], qp_energies[n] - eps[n],
                    if cd.is_some() { cd_qp_caches[n].linearized } else { qp_caches[n].linearized }
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
            BseScreening::build_from_parts_cutoff(
                q, &gw.j_mat, occ_start, nocc_act, vir_start, gw.nmo, nvir_act, &eps_screen,
                scf.mol.start_mo,
            )
        });
        let kappa = match multi {
            's' | 'S' => 2.0,
            't' | 'T' => 0.0,
            _ => panic!("bse_grad: multi must be 's' or 't'"),
        };
        // Same keyword the energy path uses, so both stages solve one functional.
        let tda = scf
            .mol
            .ctrl
            .quasiparticle_methods
            .as_ref()
            .map(|qp| qp.bse_tda)
            .unwrap_or(false);
        if tda {
            println!(
                "GW-BSE gradient: TDA (Hermitian A only) -- no A-B Cholesky, y = 0"
            );
        }
        BseGradEngine {
            gw,
            cd,
            cd_qp_caches,
            screen,
            kappa,
            tda,
            screening_energy,
            qp_energies,
            qp_caches,
            roots: Vec::new(),
            grad_roots: None,
            degenerate_groups: Vec::new(),
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
        let nmo = self.gw.nmo;
        let (occ_start, nocc) = (self.screen.occ_start, self.screen.nocc);
        let (vir_start, nvir_act) = (self.screen.vir_start, self.screen.nvir);
        self.screen = self.gw.with_q(|q| {
            BseScreening::build_from_parts_cutoff(
                q, &self.gw.j_mat, occ_start, nocc, vir_start, nmo, nvir_act, &self.eps_screen,
                self.gw.scf.mol.start_mo,
            )
        });
    }

    /// Rebuild the screening blocks if a previous gradient released them
    /// (`with_q` rebuilds the transformed integrals they need on demand).
    fn ensure_screening(&mut self) {
        if !self.screen.blocks_released() {
            return;
        }
        let nmo = self.gw.nmo;
        let (occ_start, nocc) = (self.screen.occ_start, self.screen.nocc);
        let (vir_start, nvir_act) = (self.screen.vir_start, self.screen.nvir);
        let eps = self.eps_screen.clone();
        self.screen = self.gw.with_q(|q| {
            BseScreening::build_from_parts_cutoff(
                q, &self.gw.j_mat, occ_start, nocc, vir_start, nmo, nvir_act, &eps,
                self.gw.scf.mol.start_mo,
            )
        });
    }

    /// Solve for the lowest `nroots` excitation energies.
    pub fn solve(&mut self, nroots: usize, solver: BseSolver, config: &DavidsonConfig) {
        self.ensure_screening();
        let dim = self.screen.nocc * self.screen.nvir;
        let dense_only = force_dense_solver();
        if matches!(solver, BseSolver::Davidson) && dim > 128 && !dense_only {
            // REST's own matrix-free BSE Davidson: the *same* `a_block_matvec` /
            // `b_block_matvec` and `tda_davidson_solver` / `lr_davidson_solver`
            // that the BSE spectrum path uses, applied to the RI blocks of this
            // gradient's transition window (see [`build_matvec_tensors`]).  A
            // TDA card solves the Hermitian `A` (`y = 0`, no `A-B`
            // factorisation); a full card solves the Casida form.  Both are
            // wrapped: the Casida route factorises `A-B` with a Cholesky, which
            // is indefinite on some references, and a failed attempt must
            // degrade to the dense route instead of aborting the job.
            let solved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if self.tda {
                    solve_bse_tda_davidson(
                        self.gw.scf, &self.screen, &self.qp_energies, &self.eps_screen,
                        self.kappa, nroots, config,
                    )
                } else {
                    solve_bse_lr_davidson(
                        self.gw.scf, &self.screen, &self.qp_energies, &self.eps_screen,
                        self.kappa, nroots, config,
                    )
                }
            }))
            .unwrap_or(None);
            if let Some(pairs) = solved {
                self.solver_used = if self.tda { "tda-davidson" } else { "davidson" };
                self.roots = pairs
                    .into_iter()
                    .map(|(omega, x, y)| BseRoot { omega, x, y })
                    .collect();
                self.finalize_degenerate_manifolds();
                return;
            }
            eprintln!(
                "[bse_grad] matrix-free BSE Davidson did not deliver {nroots} roots (dim={dim}); \
                 using the dense solver"
            );
        }
        let (a_mat, b_mat) = build_bse_matrices(&self.screen, &self.qp_energies, self.kappa);
        if self.tda {
            self.solver_used = "tda-dense";
            self.roots = solve_bse_tda_dense(&a_mat, dim, nroots)
                .into_iter()
                .map(|(omega, x, y)| BseRoot { omega, x, y })
                .collect();
            self.finalize_degenerate_manifolds();
            return;
        }
        self.solver_used = "dense";
        self.roots = solve_bse_dense(&a_mat, &b_mat, dim, nroots)
            .into_iter()
            .map(|(omega, x, y)| BseRoot { omega, x, y })
            .collect();
        self.finalize_degenerate_manifolds();
    }

    /// Detect energy-degenerate BSE root manifolds, metric-orthonormalise their
    /// members and remember the groups for the gradient averaging.
    ///
    /// The tolerance is `REST_BSE_DEGEN_TOL` (Hartree, default `1e-6`).  This
    /// is the BSE-root analogue of the symmetry-degenerate *orbital* shells
    /// handled in the GW gradient: without it, two roots with the same Omega
    /// receive different gradients because the solver's basis inside the
    /// manifold is arbitrary.
    fn finalize_degenerate_manifolds(&mut self) {
        self.degenerate_groups.clear();
        let tol = std::env::var("REST_BSE_DEGEN_TOL")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(1.0e-6)
            .max(0.0);
        if tol == 0.0 || self.roots.len() < 2 {
            return;
        }
        let n = self.roots.len();
        let mut i = 0usize;
        while i < n {
            let e0 = self.roots[i].omega;
            let mut j = i + 1;
            while j < n && (self.roots[j].omega - e0).abs() <= tol {
                j += 1;
            }
            let g = j - i;
            if g > 1 {
                self.orthonormalise_manifold(i, g);
                let split = self.roots[j - 1].omega - self.roots[i].omega;
                println!(
                    "[bse_grad] degenerate manifold detected: roots {}..{}, split={:.3e} Ha, projected (metric-orthonormalised, gradient averaged)",
                    i,
                    j - 1,
                    split
                );
                self.degenerate_groups.push((i..j).collect());
            }
            i = j;
        }
    }

    /// Loewdin-orthonormalise `roots[start..start+g]` in the BSE metric
    /// `S_ij = X_i.X_j - Y_i.Y_j` (the metric of the Casida problem).
    fn orthonormalise_manifold(&mut self, start: usize, g: usize) {
        let dim = self.roots[start].x.len();
        let mut s = vec![0.0f64; g * g];
        for a in 0..g {
            for b in 0..g {
                let ra = &self.roots[start + a];
                let rb = &self.roots[start + b];
                s[a + b * g] = ra.x.iter().zip(rb.x.iter()).map(|(p, q)| p * q).sum::<f64>()
                    - ra.y.iter().zip(rb.y.iter()).map(|(p, q)| p * q).sum::<f64>();
            }
        }
        // symmetrise (the metric is symmetric up to round-off)
        for a in 0..g {
            for b in (a + 1)..g {
                let m = 0.5 * (s[a + b * g] + s[b + a * g]);
                s[a + b * g] = m;
                s[b + a * g] = m;
            }
        }
        let (vecs, vals, _) = _dsyevd(&to_mat(&s, g, g), 'V');
        let vecs = vecs.expect("bse_grad: manifold metric eigenvectors");
        if vals.iter().any(|&v| v <= 1.0e-12) {
            println!(
                "[bse_grad] degenerate manifold {:?}: metric is singular, keeping the solver basis",
                (start..start + g).collect::<Vec<_>>()
            );
            return;
        }
        // T = V diag(1/sqrt(v)) V^T   (symmetric inverse square root)
        let mut t = vec![0.0f64; g * g];
        for a in 0..g {
            for b in 0..g {
                let mut acc = 0.0f64;
                for k in 0..g {
                    acc += vecs.data[a + k * g] * vecs.data[b + k * g] / vals[k].sqrt();
                }
                t[a + b * g] = acc;
            }
        }
        let xs: Vec<Vec<f64>> = (0..g).map(|a| self.roots[start + a].x.clone()).collect();
        let ys: Vec<Vec<f64>> = (0..g).map(|a| self.roots[start + a].y.clone()).collect();
        for a in 0..g {
            for r in 0..dim {
                let mut x = 0.0f64;
                let mut y = 0.0f64;
                for b in 0..g {
                    x += t[a + b * g] * xs[b][r];
                    y += t[a + b * g] * ys[b][r];
                }
                self.roots[start + a].x[r] = x;
                self.roots[start + a].y[r] = y;
            }
        }
    }

    /// Replace the gradient of every member of a detected degenerate manifold
    /// by the manifold average (see [`Self::degenerate_groups`]).
    ///
    /// `grads[p]` is the gradient of `sel[p]`; a manifold is averaged only when
    /// **all** its members are differentiated, because the average over a
    /// subset is not the derivative of a manifold-invariant quantity.
    fn average_degenerate_gradients(&self, grads: &mut [Vec<f64>], sel: &[usize]) {
        for group in &self.degenerate_groups {
            let pos: Vec<usize> = group
                .iter()
                .filter_map(|k| sel.iter().position(|s| s == k))
                .collect();
            if pos.len() < 2 {
                continue;
            }
            if pos.len() != group.len() {
                println!(
                    "[bse_grad] warning: degenerate manifold {:?} is only partially differentiated                      ({:?}); the individual gradients are basis dependent and are left as computed",
                    group, pos
                );
                continue;
            }
            let ncomp = grads[pos[0]].len();
            let mut mean = vec![0.0f64; ncomp];
            for &p in &pos {
                for c in 0..ncomp {
                    mean[c] += grads[p][c];
                }
            }
            let inv = 1.0 / pos.len() as f64;
            for c in 0..ncomp {
                mean[c] *= inv;
            }
            for &p in &pos {
                grads[p].copy_from_slice(&mean);
            }
            println!(
                "[bse_grad] degenerate manifold {:?}: gradients replaced by the manifold average",
                group
            );
        }
    }

    /// Linear-response stability of the reference state (see [`BseStability`]).
    ///
    /// Builds the same two dense blocks [`Self::solve`] would build and
    /// diagonalises them, and touches nothing else of the engine: it is the
    /// cheap pre-flight that says whether a Davidson/dense solve of this system
    /// is legitimate *before* the gradient spends hours on it.
    pub fn stability(&mut self) -> BseStability {
        self.ensure_screening();
        let (a_mat, b_mat) = build_bse_matrices(&self.screen, &self.qp_energies, self.kappa);
        let dim = self.screen.nocc * self.screen.nvir;
        let block = |sign: f64| -> MatrixFull<f64> {
            MatrixFull::from_vec(
                [dim, dim],
                (0..dim * dim).map(|k| a_mat[k] + sign * b_mat[k]).collect(),
            )
            .expect("bse_grad: A +- B block shape")
        };
        let (_, amb, _) = _dsyevd(&block(-1.0), 'N');
        let (_, apb, _) = _dsyevd(&block(1.0), 'N');
        let gap_min = (0..dim)
            .map(|k| a_mat[k + k * dim])
            .fold(f64::INFINITY, f64::min);
        BseStability {
            dim,
            gap_min,
            amb,
            apb,
        }
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
        let nocc_act = self.screen.nocc;
        let (occ_start, vir_start) = (self.screen.occ_start, self.screen.vir_start);
        let nvir = self.screen.nvir;

        if std::env::var("REST_GWGRAD_MEMDBG").is_ok() {
            let g = &self.gw;
            let d = |n: usize| n as f64 * 8.0 / 1024.0 / 1024.0;
            println!(
                "[meminv] dims: nao {} naux {} nmo {} nocc {} nvir_act {} natm {} nk {} | \
                 raw.i3 {:.0} MB | q {:.0}  qt {:.0}  qia {:.0}  qtia {:.0}  q_pairmajor {:.0} | \
                 j_jchol_smetric {:.0} | imag quad nodes {} | one pair covector naux*nmo(nmo+1)/2 {:.0}",
                g.nao, g.naux, g.nmo, g.nocc, self.screen.nvir, g.natm, nk,
                d(g.raw.i3_len()), d(g.q.borrow().len()), d(g.qt.borrow().len()),
                d(g.qia.len()), d(g.qtia.borrow().len()), d(g.q_pairmajor.len()),
                d(g.j_mat.data.len() + g.j_chol.data.len() + g.s_metric.len()),
                g.quad.len(),
                d(g.naux * g.nmo * (g.nmo + 1) / 2),
            );
        }
        let mut weights: Vec<Vec<f64>> = Vec::with_capacity(nk);
        let mut ebar_direct: Vec<Vec<f64>> = Vec::with_capacity(nk);
        let mut bj_k: Vec<Vec<f64>> = Vec::with_capacity(nk);
        let mut theta_k: Vec<Option<KernelVjpCompact>> = Vec::with_capacity(nk);
        let t_kernel = Instant::now();
        for &k in sel.iter() {
            let root = &self.roots[k];
            // amplitude weights cover the occupied orbitals and the active
            // virtuals; inactive virtuals carry zero QP weight
            let w_act = qp_gap_weights(&root.x, &root.y, nocc_act, nvir);
            let mut w = vec![0.0f64; nmo];
            // `w_act` is compacted over the active window: occupied `i` is
            // orbital `occ_start + i`, active virtual `a` is `vir_start + a`.
            w[occ_start..occ_start + nocc_act].copy_from_slice(&w_act[..nocc_act]);
            w[vir_start..vir_start + nvir].copy_from_slice(&w_act[nocc_act..]);
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
        crate::ri_gw::gw_grad::mem_probe("bse: after kernel_vjp + theta_k");
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
        crate::ri_gw::gw_grad::mem_probe("bse: before qp pullback");
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
        let t_qp = Instant::now();
        let (mut bq, mut bj, mut be, bb, nrow_pair) = if let Some(cd_engine) = self.cd {
            // Full-CD QP pullback: no low-rank page loop.  The active weights
            // are compacted to the orbitals whose full-CD cache was built; the
            // returned pair covector is dense and is packed into the same
            // lower-triangle layout the fold/adjoint code consumes.
            let active: Vec<usize> = (0..nmo)
                .filter(|&n| (0..nk).any(|k| weights[k][n] != 0.0))
                .collect();
            let refs: Vec<&CdQpCache> = active.iter().map(|&n| &self.cd_qp_caches[n]).collect();
            let w_sub: Vec<Vec<f64>> = weights
                .iter()
                .map(|w| active.iter().map(|&n| w[n]).collect())
                .collect();
            let (mut bq, bj, be, bb) = cd_engine.cd_qp_pullback(&refs, &w_sub);
            for v in &mut bq {
                crate::ri_gw::gw_grad::pack_pairs_in_place(v, nmo, naux);
            }
            self.cd_qp_caches.clear();
            (bq, bj, be, bb, nmo)
        } else {
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
            // Pair-covector band: the largest orbital that carries a QP weight, + 1.  Every writer
            // of `bq` addresses pairs whose smaller index is one of those rows, so the rest of the
            // packed pair space is identically zero and need not be stored.
            let nrow_pair = weights
                .iter()
                .map(|w| w.iter().rposition(|&v| v != 0.0).map_or(0, |i| i + 1))
                .max()
                .unwrap_or(nmo)
                .max(1)
                // The packed pair covector must also cover the complete active
                // window even when every amplitude weight happens to vanish at a
                // particular (near-degenerate, single-configuration) geometry.
                // Without this floor `add_into_packed` can address pairs outside
                // the banded buffer.
                .max(occ_start + nocc_act)
                .max(vir_start + nvir)
                .min(nmo);
            let mut scr = self.gw.qp_pullback_scratch_banded(nk, nrow_pair);
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
            (bq, bj, be, bb, nrow_pair)
        };
        crate::ri_gw::gw_grad::mem_probe("bse: after qp pullback (bq/bj/be/bb live)");
        let t_qp = t_qp.elapsed().as_secs_f64();
        // The low-rank screening factors are read only while a self-energy
        // response is evaluated, i.e. by the pullback (they also hold the
        // transformed blocks, released before it): return them before the fold
        // and the adjoint, whose `naux*nao^2` tensors set the peak.
        self.gw.evict_qt();
        // the low-rank QP caches are consumed by the pullback; a later
        // gradient rebuilds them on demand
        if self.cd.is_none() {
            self.qp_caches.clear();
        }
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
        crate::ri_gw::gw_grad::mem_probe("bse: after kernel covector merge");

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
            crate::ri_gw::gw_grad::mem_probe("bse: enter memory-lean adjoint route");
            let per_atom: Vec<AdjointAtomData> = (0..self.gw.natm)
                .map(|atm| AdjointAtomData::build(self.gw, atm))
                .collect();
            crate::ri_gw::gw_grad::mem_probe("bse: after per-atom derivative blocks");
            // Explicit v_xc grid derivative of a DFT reference: one sweep,
            // shared by every root and perturbation (`None` for HF, whose
            // `b_x` is identically zero).  `b_x`'s U-dependent parts live in
            // the adjoint right-hand side (`build_cphf_adjoint`).
            let vxc_atoms = self.gw.vxc_grid_derivative_atoms();
            crate::ri_gw::gw_grad::mem_probe("bse: after v_xc grid-derivative sweep");
            // Expand the kernel covector into each root's QP covector and pack
            // it in place before folding: the packed pair covector is half the
            // full one, and after this loop only the packed forms are resident.
            let npack = crate::ri_gw::gw_grad::banded_pair_count(nmo, nrow_pair);
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
            crate::ri_gw::gw_grad::mem_probe("bse: after packing + trim");
            for k in 0..nk {
                let t_fold = Instant::now();
                let mut fold = self
                    .gw
                    .fold_covector_packed_streamed_parallel_banded(&packed[k], nrow_pair);
                let t_fold = t_fold.elapsed().as_secs_f64();
                crate::ri_gw::gw_grad::mem_probe("bse: after fold_covector (packed live)");
                packed[k] = Vec::new();
                let t_adj = Instant::now();
                let adj = self
                    .gw
                    .build_cphf_adjoint(&mut fold, &be[k], Some(&bb[k]))
                    .expect("bse_grad: adjoint route guard changed");
                let t_adj = t_adj.elapsed().as_secs_f64();
                crate::ri_gw::gw_grad::mem_probe("bse: after build_cphf_adjoint");
                drop(fold);
                let t_pert = Instant::now();
                for atm in 0..self.gw.natm {
                    let d = &per_atom[atm];
                    let bx3: Vec<Vec<f64>> = match vxc_atoms.as_ref() {
                        Some(a) => (0..3).map(|c| self.gw.vxc_diag_bx(a, atm, c)).collect(),
                        None => vec![vec![0.0f64; nmo]; 3],
                    };
                    if self.gw.config.local_di {
                        let g3 = self.gw.adjoint_perturbation_gradient_local(
                            &adj, atm, &d.h1r, &d.s1r, &d.djs, &bj[k], &bb[k], &bx3,
                            vxc_atoms.as_ref().map(|a| &a[atm]), atm * 3,
                        );
                        grads[k][atm * 3..atm * 3 + 3].copy_from_slice(&g3);
                    } else {
                        let blocks = self.gw.raw.d_atom_blocks(atm);
                        for comp in 0..3 {
                            grads[k][atm * 3 + comp] = self.gw.adjoint_perturbation_gradient(
                                &adj, &blocks, comp, &d.h1r, &d.s1r[comp], &d.djs[comp],
                                &bj[k], &bb[k], &bx3[comp],
                                vxc_atoms.as_ref().map(|a| &a[atm]), atm * 3 + comp,
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
            let vxc_atoms = self.gw.vxc_grid_derivative_atoms();
            let npack = crate::ri_gw::gw_grad::banded_pair_count(nmo, nrow_pair);
            let mut folds: Vec<Option<crate::ri_gw::gw_grad::FoldedCovector>> =
                (0..nk).map(|_| None).collect();
            let mut canonical_at: Vec<usize> = Vec::new();
            for k in 0..nk {
                let kv = theta_k[k].take().expect("bse_grad: kernel covector reused");
                let mut qbar = std::mem::take(&mut bq[k]);
                kv.add_into_packed(&mut qbar, naux, nmo);
                assert_eq!(qbar.len(), npack * naux, "bse_grad: packed covector size mismatch");
                let mut fold = self
                    .gw
                    .fold_covector_packed_streamed_parallel_banded(&qbar, nrow_pair);
                drop(qbar);
                if adjoint_ok && covectors_isotropic_on_shells(self.gw, &fold, &be[k]) {
                    let pa = per_atom.get_or_insert_with(|| {
                        (0..self.gw.natm)
                            .map(|atm| AdjointAtomData::build(self.gw, atm))
                            .collect()
                    });
                    let adj = self
                        .gw
                        .build_cphf_adjoint(&mut fold, &be[k], Some(&bb[k]))
                        .expect("bse_grad: adjoint route guard changed");
                    drop(fold);
                    for atm in 0..self.gw.natm {
                        let d = &pa[atm];
                        let bx3: Vec<Vec<f64>> = match vxc_atoms.as_ref() {
                            Some(a) => (0..3).map(|c| self.gw.vxc_diag_bx(a, atm, c)).collect(),
                            None => vec![vec![0.0f64; nmo]; 3],
                        };
                        if self.gw.config.local_di {
                            let g3 = self.gw.adjoint_perturbation_gradient_local(
                                &adj, atm, &d.h1r, &d.s1r, &d.djs, &bj[k], &bb[k], &bx3,
                                vxc_atoms.as_ref().map(|a| &a[atm]), atm * 3,
                            );
                            grads[k][atm * 3..atm * 3 + 3].copy_from_slice(&g3);
                        } else {
                            let blocks = self.gw.raw.d_atom_blocks(atm);
                            for comp in 0..3 {
                                grads[k][atm * 3 + comp] = self.gw.adjoint_perturbation_gradient(
                                    &adj, &blocks, comp, &d.h1r, &d.s1r[comp], &d.djs[comp],
                                    &bj[k], &bb[k], &bx3[comp],
                                    vxc_atoms.as_ref().map(|a| &a[atm]), atm * 3 + comp,
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
        crate::ri_gw::gw_grad::mem_probe("bse: after per-direction assembly");
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
        self.average_degenerate_gradients(&mut grads, &sel);
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
        let nocc_act = self.screen.nocc;
        let (occ_start, vir_start) = (self.screen.occ_start, self.screen.vir_start);
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

    /// Expand one root's compact kernel covector into the full-pair layout and
    /// compare it **entry by entry** with the explicit full-pair route, so a
    /// mis-mapped block index can be located (the scalar contraction of
    /// [`Self::kernel_contraction_compare`] only shows *that* something is off).
    /// Returns `(compact, full, naux, nmo)`.
    pub fn kernel_covector_compare(&mut self, k: usize) -> (Vec<f64>, Vec<f64>, usize, usize) {
        self.ensure_screening();
        let nmo = self.gw.nmo;
        let naux = self.gw.naux;
        let pair = |p: usize, q: usize| p + q * nmo;
        let r = &self.roots[k];
        let kv = kernel_vjp_compact(&self.screen, &r.x, &r.y, self.kappa);
        let mut compact = vec![0.0f64; naux * nmo * nmo];
        kv.add_into_full_pair(&mut compact, naux, nmo);
        let pcs = kernel_vjp_pieces(&self.screen, &r.x, &r.y, self.kappa, pair, nmo);
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
        (compact, full, naux, nmo)
    }

    /// Per-piece value of one root's kernel covector at one orbital pair
    /// (`naux`-summed? no: at every `naux` row), used to localise an index
    /// mismatch between the compact and the full-pair route.
    pub fn kernel_pieces_at(&mut self, k: usize, p_orb: usize, q_orb: usize) -> Vec<(&'static str, f64, f64)> {
        self.ensure_screening();
        let nmo = self.gw.nmo;
        let naux = self.gw.naux;
        let pair = |p: usize, q: usize| p + q * nmo;
        let r = &self.roots[k];
        let kv = kernel_vjp_compact(&self.screen, &r.x, &r.y, self.kappa);
        let mut compact = vec![0.0f64; naux * nmo * nmo];
        kv.add_into_full_pair(&mut compact, naux, nmo);
        let pcs = kernel_vjp_pieces(&self.screen, &r.x, &r.y, self.kappa, pair, nmo);
        let base = pair(p_orb, q_orb) * naux;
        let sum = |v: &[f64]| (0..naux).map(|a| v[base + a]).sum::<f64>();
        vec![
            ("ex", sum(&pcs.qbar_ex), sum(&pcs.qbar_ex)),
            ("oo", sum(&pcs.qbar_oo), sum(&pcs.qbar_oo)),
            ("vv", sum(&pcs.qbar_vv), sum(&pcs.qbar_vv)),
            ("ov_direct", sum(&pcs.qbar_ov_direct), sum(&pcs.qbar_ov_direct)),
            ("bov_a", sum(&pcs.qbar_bov_a), sum(&pcs.qbar_bov_a)),
            ("bov_b", sum(&pcs.qbar_bov_b), sum(&pcs.qbar_bov_b)),
            ("COMPACT", sum(&compact), sum(&compact)),
        ]
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
        let nocc_act = self.screen.nocc;
        let (occ_start, vir_start) = (self.screen.occ_start, self.screen.vir_start);
        let nvir = self.screen.nvir;
        let pair = |p: usize, q: usize| p + q * nmo;
        let mut weights: Vec<Vec<f64>> = Vec::with_capacity(nk);
        let mut ebar_direct: Vec<Vec<f64>> = Vec::with_capacity(nk);
        let mut bq_k: Vec<Vec<f64>> = Vec::with_capacity(nk);
        let mut bj_k: Vec<Vec<f64>> = Vec::with_capacity(nk);
        for root in &self.roots {
            let w_act = qp_gap_weights(&root.x, &root.y, nocc_act, nvir);
            let mut w = vec![0.0f64; nmo];
            // `w_act` is compacted over the active window: occupied `i` is
            // orbital `occ_start + i`, active virtual `a` is `vir_start + a`.
            w[occ_start..occ_start + nocc_act].copy_from_slice(&w_act[..nocc_act]);
            w[vir_start..vir_start + nvir].copy_from_slice(&w_act[nocc_act..]);
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
        let sel: Vec<usize> = (0..nk).collect();
        self.average_degenerate_gradients(&mut grads, &sel);
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
        let (os, vs) = (screen.occ_start, screen.vir_start);
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
            let i = os + c % nocc;
            let a = vs + c / nocc;
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
                    let a = vs + ab % nvir;
                    let b = vs + ab / nvir;
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
            let i = os + c % nocc;
            let a = vs + c / nocc;
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
            occ_start: 0,
            vir_start: nocc,
            nvir,
            nvir_full,
            screen_occ_start: 0,
            screen_nocc: nocc,
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
            occ_start: 0,
            vir_start: nocc,
            nvir,
            nvir_full,
            screen_occ_start: 0,
            screen_nocc: nocc,
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
