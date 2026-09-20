//! AO-basis ("memory-efficient") BSE matvec.
//!
//! Selected with `bse_matvec_style = "ao"` in
//! `[quasiparticle_methods]`; the default `"fast"` keeps the historical
//! implementation in [`super::matvec`] completely untouched.
//!
//! The authoritative derivation is `docs/bse_matvec_fast_derivation.md`; every
//! convention used here is locked by an element-wise test in
//! `tests/test_bse_matvec_fast.rs`.
//!
//! # Notation
//!
//! `N = n_bas`, `M = n_aux`, `O`/`V` occupied/virtual counts;
//! `start_mo`/`homo`/`lumo` are *absolute* MO indices and `O = homo - start_mo + 1`.
//! MO coefficients `X` are `[N, n_mo]` column-major (`X[mu + A*N]`), the
//! transition vector is column-major `[O, V]` (`z[i + a*O]`).
//!
//! `rimatr_bse` stores the upper triangle of `J^q[mu,nu] = (mu nu | q)` with row
//! `hi` at `[hi(hi+1)/2, hi(hi+1)/2 + hi]`, i.e. the entry `(hi, lo)` with
//! `hi >= lo` sits at `hi*(hi+1)/2 + lo` (`Molecule::prepare_baspair_map`).
//! `J^q` is symmetric.
//!
//! # The identity the module rests on
//!
//! With `D` the inverse dielectric matrix and
//!
//! ```text
//! T[q,mu,nu] = sum_r D[q,r] J^r[mu,nu]                       (screened AO 3-centre)
//! ```
//!
//! every MO-basis RI tensor the historical matvec uses is a **fold** of `T`:
//!
//! ```text
//! A_q[i,j] = ri_oo_tilde[q, j + i*O] = sum_{mu,nu} T[q,mu,nu] X[mu,start_mo+i] X[nu,start_mo+j]
//! B_q[a,b] = ri_vv      [q, a + b*V] = sum_{mu,nu} T[q,mu,nu] X[mu,lumo+a]     X[nu,lumo+b]
//! Z_q[i,a] = ri_ov_tilde[q, i + a*O] = sum_{mu,nu} T[q,mu,nu] X[mu,start_mo+i] X[nu,lumo+a]
//! R_q[i,a] = ri_ov      [q, i + a*O] = sum_{mu,nu} J^q[mu,nu] X[mu,start_mo+i] X[nu,lumo+a]
//! ```
//!
//! The last one is **unscreened**: the direct term of the BSE kernel uses
//! `ri_ov`, not `ri_ov_tilde`.  That is the most easily misread point of the
//! historical implementation.
//!
//! # Matvec recipes (restricted reference)
//!
//! ```text
//! (A z)[i,a] = (eps[lumo+a] - eps[start_mo+i]) z[i,a]
//!              - f * sum_q sum_{j,b} A_q[i,j] B_q[a,b] z[j + b*O]
//!              + c * V2[i,a]
//! (B z)[i,a] = -     sum_q sum_{j,b} Z_q[j,a] R_q[b,i] z[j + b*O]
//!              + c * V2[i,a]
//! V2[i,a]    = sum_q s_q R_q[i,a],   s_q = sum_{j,b} R_q[j,b] z[j + b*O]
//! ```
//!
//! `f = bse_exchange_rescaling`; `c = 2` (singlet) / `1` (RPA) / `0` (triplet).
//! Two historical asymmetries are reproduced deliberately: the B block does
//! **not** apply `bse_exchange_rescaling`, and the triplet channel omits the
//! direct term altogether.
//!
//! # Implementation
//!
//! Everything is organised around one primitive.  For a per-auxiliary tensor
//! `M_q[i, a] = M[q][i][a]` and two MO blocks `Xp` (leg over `i`) and `Xq` (leg
//! over `a`), define
//!
//! ```text
//! fold_q(M, Xp, Xq)[q, A, B] = sum_{i,a} M_q[i,a] Xp[i,A] Xq[B,a]
//! ```
//!
//! This is a BLAS-3 contraction over the two AO legs of `M_q` (`[N,N] x [N,O]`
//! then `[N,N] x [N,V]`), i.e. exactly the historical `ao2mo` cost, but it keeps
//! the auxiliary index `q` open, so the dielectric contraction can be applied
//! afterwards with a **single** auxiliary-index contraction of `O(M^2 * n)` data.
//! The three folds the matvec needs are
//!
//! ```text
//! wA = fold_q(T, Xo, Xv)   applied to (j,b) -> (i,a)      (A block exchange)
//! wB = fold_q(T, Xv, Xo)   applied to (j,b) -> (i,a)      (B block exchange)
//! r  = fold_q(J, Xo, Xv)                                  (direct term, UNscreened)
//! ```
//!
//! each `M * n_o * n_v` in size.  No `[O V, O V]` matrix, no MO-basis `ri3mo`
//! tensor and no `[M, N, N]` object is kept after the build.
//!
//! ## Cost
//!
//! Build: `O(M N^2 (O + V))` for the folds plus `O(M^2 n_o n_v)` for the
//! dielectric contractions (one auxiliary-major pass each).
//! Per matvec: `O(M O V)` per term.  Persistent storage `O(M O V)`.

use rayon::prelude::*;
use rest_tensors::matrix::matrix_blas_lapack::omp_set_num_threads_wrapper;
use rest_tensors::matrix::MatrixFull;
use std::os::raw::c_int;
use std::time::Instant;

// --------------------------------------------------------------- BLAS access --
//
// The workspace already links a BLAS (`rest_tensors` calls `dgemm_` itself), so
// the CBLAS entry point is available without adding a dependency.  We call it
// directly instead of going through `rest_tensors::_dgemm_full` / `_dgemm`
// because both derive the leading dimension from something other than the
// operand we pass:
//
//   * `_dgemm_full` uses the *logical* row count (wrong for reshaped operands),
//   * `_dgemm` routes through `general_dgemm_f`, which takes LDA from the C
//     sub-block and LDB from the C sub-block's column count.
//
// Every operand below is a contiguous column-major block whose leading
// dimension we know exactly, so we pass it explicitly.
extern "C" {
    fn cblas_dgemm(
        layout: c_int,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f64,
        a: *const f64,
        lda: c_int,
        b: *const f64,
        ldb: c_int,
        beta: f64,
        c: *mut f64,
        ldc: c_int,
    );
}
const CBLAS_COL_MAJOR: c_int = 102;
const CBLAS_NO_TRANS: c_int = 111;
const CBLAS_TRANS: c_int = 112;

/// `C[m,n] = alpha * A[m,k] * B[n,k]^T + beta * C[m,n]`, all column-major.
///
/// `lda`, `ldb`, `ldc` are the true leading dimensions of the parents the slices
/// belong to, so a per-`q` slab of an `[n1, n2*M]` parent can be passed directly.
#[inline]
#[allow(clippy::too_many_arguments)]
fn dgemm_nt(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    debug_assert!(
        a.len() >= (k - 1) * lda + m,
        "A slice too short: {} < {}",
        a.len(),
        (k - 1) * lda + m
    );
    debug_assert!(b.len() >= (k - 1) * ldb + n, "B slice too short");
    debug_assert!(c.len() >= (n - 1) * ldc + m, "C slice too short");
    unsafe {
        cblas_dgemm(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            m as c_int,
            n as c_int,
            k as c_int,
            alpha,
            a.as_ptr(),
            lda as c_int,
            b.as_ptr(),
            ldb as c_int,
            beta,
            c.as_mut_ptr(),
            ldc as c_int,
        );
    }
}

/// `C[m,n] = alpha * A[m,k] * B[k,n] + beta * C[m,n]`, all column-major.
/// Used for the auxiliary-index contraction `T = D * J` (`A = D`).
#[inline]
#[allow(clippy::too_many_arguments)]
fn dgemm_nn(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    unsafe {
        cblas_dgemm(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as c_int,
            n as c_int,
            k as c_int,
            alpha,
            a.as_ptr(),
            lda as c_int,
            b.as_ptr(),
            ldb as c_int,
            beta,
            c.as_mut_ptr(),
            ldc as c_int,
        );
    }
}

use crate::ctrl_io::quasiparticle_methods::QuasiParticle;
use crate::ri_gw::{get_occ_params_per_spin, get_occupation_parameters};
use crate::scf_io::SCF;

/// The AO-basis BSE matvec context.
pub struct FastBseContext {
    n_bas: usize,
    n_aux: usize,
    /// `J[q][mu][nu]` (`[M, N, N]`, `nu` fastest) -- the **unscreened** AO tensor.
    ///
    /// **`None` in production, and no `[M, N, N]` array exists at any point of
    /// the build**: the fold pass streams the packed `rimatr_bse` one block of
    /// columns at a time.  This field is only filled by
    /// `build_keeping_ao_tensor`, for the element-wise diagnostic hooks.
    jdiag: Option<Vec<f64>>,
    /// Inverse dielectric matrix, `[M, M]` column-major — needed by the B block,
    /// whose dielectric contraction cannot be pre-folded (it multiplies a
    /// `z`-dependent intermediate).
    d: Vec<f64>,
    /// Per-AO-pair screening mask over the packed triangle (`false` = the pair is
    /// negligible for **every** auxiliary function and is skipped everywhere).
    ///
    /// `None` when no screening was requested.
    pair_keep: Option<Vec<bool>>,
    spins: Vec<SpinChannel>,
    exchange_rescaling: f64,
    /// Seconds spent in [`FastBseContext::build`].
    pub build_seconds: f64,
    /// Bytes of persistent storage.
    pub bytes: usize,
    /// OpenMP/BLAS thread count configured for the run (`ctrl.num_threads`).
    ///
    /// The kernels set the BLAS to **one** thread while the Rayon `q`-loop runs
    /// (otherwise every `dgemm` spawns its own OpenMP team on top of the pool and
    /// the two oversubscribe); this is the value they restore afterwards, using
    /// the same convention as `molecule_io`.
    blas_threads: usize,
    /// Diagnostic counter.  Atomic so that the context is `Send + Sync` and can
    /// be shared (behind an `Arc`) by the FEAST preconditioner closures.
    matvec_count: std::sync::atomic::AtomicUsize,
}

/// One spin channel: the MO window plus the folds the matvec consumes.
struct SpinChannel {
    n_o: usize,
    n_v: usize,
    n_state: usize,
    start_mo: usize,
    lumo: usize,
    /// `X[mu, A]`, `[N, n_mo]` column-major.
    x: Vec<f64>,
    /// `Bq[q][a][b] = sum_{mu,la} J[q,mu,la] Xv[mu,a] Xv[la,b]` — the **un**screened
    /// virtual x virtual block.
    ///
    /// Stored as `M` contiguous **column-major** `[V, V]` slabs
    /// (`bq[q*V*V + b*V + a]`), i.e. exactly the operand BLAS reads when the
    /// slab is passed as `A[V,V]` with `lda = V`.  The parent is an
    /// `[V, V*M]` matrix and slab `q` is its column block `q*V..(q+1)*V`.
    bq: Vec<f64>,
    /// `R[q][i][b] = ri_ov[q, i + b*O]` — the **un**screened occupied x virtual fold.
    ///
    /// Column-major `[O, V]` slabs (`r[q*V*O + b*O + i] == r[q][i][b]`), parent
    /// `[O, V*M]`.  `direct_term` walks it with unit stride; `exchange_b` hands
    /// the slab straight to BLAS.
    r: Vec<f64>,
    /// `Aq[q][i][j] = sum_{mu,nu} T[q,mu,nu] Xo[mu,i] Xo[nu,j]` — the SCREENED
    /// occupied x occupied block (`ri_oo_tilde`).
    ///
    /// Column-major `[O, O]` slabs (`aq[q*O*O + j*O + i]`), parent `[O, O*M]`.
    aq: Vec<f64>,
    /// `Z[q][j][a] = sum_{mu,nu} T[q,mu,nu] Xv[mu,a] Xo[nu,j]` — the SCREENED
    /// occupied x virtual block (`ri_ov_tilde`), `zt[q*O*V + j*V + a]`.
    ///
    /// Column-major `[V, O]` slabs when the two labels are read as `(a, j)`,
    /// parent `[V, O*M]`; BLAS consumes it with the transpose flag.
    zt: Vec<f64>,
}

#[inline]
fn x_at(c: &SpinChannel, n_bas: usize, mu: usize, mo: usize) -> f64 {
    c.x[mu + mo * n_bas]
}

/// One blocked instance of the module's only fold primitive.
///
/// For each auxiliary function `q` of the block,
///
/// ```text
/// m[q, i, a] = m_block[q*ni*na + a*ni + i]        (column-major, `i` fastest)
/// out[q, A, B] = sum_{i,a} m[q,i,a] Xp[i,A] Xq[a,B]
/// ```
///
/// with `Xp` `[ldp, n_a_out]`, `Xq` `[ldq, n_b_out]`; the result is written into
/// the caller's slice of the *global* fold array `[M, n_a_out, n_b_out]` (second
/// index fastest), so no per-block result buffer is allocated.
///
/// Serial on purpose: the caller parallelises over blocks.
///
/// Independent leading dimensions on the two operands matter as soon as the fold
/// is not square -- for the `[N, N]` AO slab they coincide, for the `[O, V]` fold
/// of it they do not.
#[allow(clippy::too_many_arguments)]
fn fold_q_block(
    m: &[f64],
    xp: &[f64],
    xq: &[f64],
    ldp: usize,
    ldq: usize,
    n_aux: usize,
    ni: usize,
    na: usize,
    n_a_out: usize,
    n_b_out: usize,
    out: &mut [f64],
) {
    debug_assert_eq!(m.len(), n_aux * ni * na);
    debug_assert_eq!(out.len(), n_aux * n_a_out * n_b_out);
    for (q, oq) in out.chunks_mut(n_a_out * n_b_out).enumerate() {
        let mq = &m[q * ni * na..(q + 1) * ni * na];
        // step 1: tmp[bb, i] = sum_a xq[a + bb*ldq] * m[q, i, a]
        let mut tmp = vec![0.0f64; n_b_out * ni];
        for a in 0..na {
            let ma = &mq[a * ni..(a + 1) * ni];
            for bb in 0..n_b_out {
                let xv = xq[a + bb * ldq];
                if xv == 0.0 {
                    continue;
                }
                let trow = &mut tmp[bb * ni..(bb + 1) * ni];
                for (d, src) in trow.iter_mut().zip(ma.iter()) {
                    *d += src * xv;
                }
            }
        }
        // step 2: out[aa, bb] = sum_i xp[i + aa*ldp] * tmp[bb, i]
        for aa in 0..n_a_out {
            let orow = &mut oq[aa * n_b_out..(aa + 1) * n_b_out];
            for i in 0..ni {
                let xi = xp[i + aa * ldp];
                if xi == 0.0 {
                    continue;
                }
                for bb in 0..n_b_out {
                    orow[bb] += tmp[bb * ni + i] * xi;
                }
            }
        }
    }
}

/// One blocked instance of the square fold
/// `out[q,i,j] = sum_{mu,nu} m[q,mu,nu] X[mu,i] X[nu,j]`, result **column-major**
/// in `(i,j)` (`out[q*n*n + j*n + i]`), i.e. the operand layout BLAS reads.
/// The fold is symmetric in `(i,j)`, so writing it this way costs nothing.
fn fold2_sym_block(t: &[f64], x: &[f64], n_bas: usize, n_aux: usize, n: usize, out: &mut [f64]) {
    debug_assert_eq!(t.len(), n_aux * n_bas * n_bas);
    debug_assert_eq!(out.len(), n_aux * n * n);
    for (q, oq) in out.chunks_mut(n * n).enumerate() {
        let tq = &t[q * n_bas * n_bas..(q + 1) * n_bas * n_bas];
        // step 1: p[mu, j] = sum_nu t[q,mu,nu] X[nu,j]
        let mut p = vec![0.0f64; n_bas * n];
        for nu in 0..n_bas {
            for j in 0..n {
                let xj = x[nu + j * n_bas];
                if xj == 0.0 {
                    continue;
                }
                for mu in 0..n_bas {
                    p[mu + j * n_bas] += tq[mu * n_bas + nu] * xj;
                }
            }
        }
        // step 2: out[i, j] = sum_mu X[mu,i] p[mu,j], written column-major.
        // Nesting `j -> mu -> i` makes the innermost loop write `oq[j*n + i]`
        // with unit stride and read `x[mu + i*n_bas]` with stride `n_bas`.
        for j in 0..n {
            let ocol = &mut oq[j * n..(j + 1) * n];
            for mu in 0..n_bas {
                let pj = p[mu + j * n_bas];
                if pj == 0.0 {
                    continue;
                }
                let xcol = &x[mu..];
                for i in 0..n {
                    ocol[i] += xcol[i * n_bas] * pj;
                }
            }
        }
    }
}

/// Unpack `this` columns of the packed RI tensor into `this` contiguous
/// `[n_bas, n_bas]` slabs (`nu` fastest), honouring the screening mask.
///
/// `rdata` is the column-major `[n_packed, n_aux]` parent, so column `q` is a
/// *contiguous* `n_packed`-long run -- the streamed read is sequential.
fn unpack_block(
    slab: &mut [f64],
    rdata: &[f64],
    n_packed: usize,
    n_bas: usize,
    q0: usize,
    this: usize,
    keep: &Option<Vec<bool>>,
) {
    debug_assert_eq!(slab.len(), this * n_bas * n_bas);
    for qq in 0..this {
        let col = (q0 + qq) * n_packed;
        let sl = &mut slab[qq * n_bas * n_bas..(qq + 1) * n_bas * n_bas];
        for hi in 0..n_bas {
            for lo in 0..=hi {
                let k = hi * (hi + 1) / 2 + lo;
                if keep.as_ref().map_or(true, |m| m[k]) {
                    let v = rdata[col + k];
                    sl[hi * n_bas + lo] = v;
                    sl[lo * n_bas + hi] = v;
                }
            }
        }
    }
}

impl FastBseContext {
    /// Build the context.
    ///
    /// `inverse_dielectric` must be the same matrix the historical path uses
    /// (`ri_bse::construct_inverse_dielectric`).
    pub fn build(scf_data: &SCF, inverse_dielectric: &MatrixFull<f64>) -> Option<Self> {
        Self::build_with_options(scf_data, inverse_dielectric, 0.0, false)
    }

    /// Same as [`Self::build`], but keeps the `[M, N, N]` screened AO tensor `T`
    /// alive so the element-wise diagnostic hooks (`t_at`, `aq_at`, `ztilde_at`,
    /// `tq_at`) can be used.  **Do not use this for production runs**: `T` is
    /// `M*N*N` doubles and is never touched by the matvec.
    pub fn build_keeping_ao_tensor(
        scf_data: &SCF,
        inverse_dielectric: &MatrixFull<f64>,
    ) -> Option<Self> {
        Self::build_with_options(scf_data, inverse_dielectric, 0.0, true)
    }

    /// Build with RI pre-screening.
    ///
    /// `screening_tol` (atomic units) is applied to the entries of the
    /// RI-fitted three-centre tensor `rimatr_bse`:
    ///
    /// ```text
    ///   keep(q, mu, nu)  <=>  max_q |rimatr_bse[(mu,nu), q]| > screening_tol
    /// ```
    ///
    /// i.e. an AO pair is dropped from **all** folds when its largest fitted
    /// three-centre integral over the whole auxiliary set is below the threshold.
    /// Because `rimatr_bse` is the metric-fitted tensor, its entries are the
    /// RI expansion coefficients of `(mu nu |` in an orthonormalised auxiliary
    /// basis, so an absolute threshold is directly meaningful: dropping all
    /// coefficients below `tol` changes every fitted integral by at most
    /// `sqrt(M) * tol` by Cauchy-Schwarz.
    ///
    /// `tol = 0.0` disables screening (bit-for-bit the unscreened result).
    ///
    /// The `[M, N, N]` AO tensor is **not** retained (see
    /// [`Self::build_keeping_ao_tensor`] for the diagnostic variant).
    pub fn build_with_screening(
        scf_data: &SCF,
        inverse_dielectric: &MatrixFull<f64>,
        screening_tol: f64,
    ) -> Option<Self> {
        Self::build_with_options(scf_data, inverse_dielectric, screening_tol, false)
    }

    /// Build with explicit control over screening and over whether the
    /// `[M, N, N]` AO tensor is retained.
    pub fn build_with_options(
        scf_data: &SCF,
        inverse_dielectric: &MatrixFull<f64>,
        screening_tol: f64,
        keep_ao_tensor: bool,
    ) -> Option<Self> {
        let start = Instant::now();
        let n_bas = scf_data.mol.num_basis;
        let n_aux = inverse_dielectric.size[0];
        let qp_ctrl = scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap_or_default();
        let n_spin = scf_data.mol.spin_channel.max(1);

        // The packed AO RI tensor.  `rimatr_bse` is the *BSE-specific* auxiliary
        // basis (built only when `[quasiparticle_methods] bse_auxbas_path` is
        // set); when it is absent every consumer in REST -- including the
        // `ri_ov` used to build the inverse dielectric -- falls back to the
        // regular `rimatr` of `auxbas_path`, so we fall back the same way.  That
        // keeps the AO matvec available *without* paying for a second
        // `[N(N+1)/2, M]` tensor.
        let (rimatr, _, _) = match scf_data.rimatr_bse.as_ref().or(scf_data.rimatr.as_ref()) {
            Some(v) => v,
            None => {
                println!(
                    "bse_matvec_style = \"ao\" needs the packed AO-basis RI tensor \
                     (rimatr_bse, or the regular rimatr when no BSE auxiliary basis is set): \
                     set use_ri_symm = true. Keeping the MO-basis matvec."
                );
                return None;
            }
        };
        let n_packed = n_bas * (n_bas + 1) / 2;
        if rimatr.size[0] != n_packed || rimatr.size[1] != n_aux {
            println!(
                "bse_matvec_style = \"ao\": rimatr_bse has shape [{}, {}], expected \
                 [{}, {}]; keeping the MO-basis matvec.",
                rimatr.size[0], rimatr.size[1], n_packed, n_aux
            );
            return None;
        }

        // ---- RI pre-screening mask over the packed AO pairs ----
        //
        // A pair is kept when *any* auxiliary function has a fitted integral
        // above the threshold.  This is a per-pair (not per-triple) decision, so
        // the contraction loops keep their simple unit-stride form.
        let pair_keep: Option<Vec<bool>> = if screening_tol >= 0.0 {
            let mut keep = vec![false; n_packed];
            // `mk` accumulates max_q |R| for every pair
            let mut mk = vec![0.0f64; n_packed];
            for q in 0..n_aux {
                let packed = &rimatr.data[q * n_packed..(q + 1) * n_packed];
                for k in 0..n_packed {
                    let v = packed[k].abs();
                    if v > mk[k] {
                        mk[k] = v;
                    }
                }
            }
            let mut kept = 0usize;
            for k in 0..n_packed {
                if mk[k] > screening_tol {
                    keep[k] = true;
                    kept += 1;
                }
            }
            println!(
                "BSE matvec AO pre-screening: kept {}/{} AO pairs ({:.2}%) at tol = {:.1e}",
                kept,
                n_packed,
                100.0 * kept as f64 / n_packed as f64,
                screening_tol
            );
            Some(keep)
        } else {
            None
        };

        // ---- streaming fold pass over the *packed* RI tensor ----------------
        //
        // `rimatr_bse` is a column-major `[n_packed, n_aux]` matrix: column `q`
        // holds the packed upper triangle of `J^q` and is contiguous.  We stream
        // one block of columns at a time into a small `[nb, N, N]` scratch and
        // fold immediately, so **the `[M, N, N]` AO tensor is never
        // materialised** -- neither the unscreened `J` nor the screened `T`
        // (the previous version held both, `2 * M*N*N` doubles).
        //
        // The dielectric contraction is applied *after* the fold, to the folds
        // themselves.  The fold is linear in `J^q`, so
        //
        //   sum_{mu,nu} T[q,mu,nu] X X  =  sum_r D[q,r] ( sum_{mu,nu} J^r[mu,nu] X X )
        //
        // i.e. `aq = D * aqQ`, `zt = D * ztQ` with `aqQ`/`ztQ` the **unscreened**
        // folds.  That replaces an `O(M^2 N^2)` streaming loop (which walked the
        // whole `[M,N,N]` array once per `q`, ~2.6 TB of traffic at
        // benzene/cc-pVQZ) by an `O(M^2 (O^2+OV))` BLAS-3 contraction.
        let rdata = &rimatr.data;
        let nb_max = (4usize << 20) / (n_bas * n_bas * 8).max(1); // ~4 MB of slab
        let nb = nb_max.clamp(1, 64);
        let n_blocks = (n_aux + nb - 1) / nb;

        // Diagnostic-only copy of the whole unscreened AO tensor (never built in
        // production): `jdiag[q*N*N + mu*N + nu]`.
        let jdiag: Option<Vec<f64>> = if keep_ao_tensor {
            let mut jbuf = vec![0.0f64; n_aux * n_bas * n_bas];
            jbuf.par_chunks_mut(n_bas * n_bas).enumerate().for_each(|(q, jq)| {
                let col = q * n_packed;
                match &pair_keep {
                    Some(keep) => {
                        for hi in 0..n_bas {
                            for lo in 0..=hi {
                                let k = hi * (hi + 1) / 2 + lo;
                                if keep[k] {
                                    let v = rdata[col + k];
                                    jq[hi * n_bas + lo] = v;
                                    jq[lo * n_bas + hi] = v;
                                }
                            }
                        }
                    }
                    None => {
                        for hi in 0..n_bas {
                            for lo in 0..=hi {
                                let k = hi * (hi + 1) / 2 + lo;
                                let v = rdata[col + k];
                                jq[hi * n_bas + lo] = v;
                                jq[lo * n_bas + hi] = v;
                            }
                        }
                    }
                }
            });
            Some(jbuf)
        } else {
            None
        };

        // ---- per-spin folds ----
        let occ = get_occ_params_per_spin(scf_data, 'N');
        let mut spins = Vec::with_capacity(n_spin);
        for ispin in 0..n_spin {
            let (start_mo, n_state, n_o, n_v, homo, lumo) = if n_spin == 1 {
                get_occupation_parameters(scf_data, 'N')
            } else {
                let o = occ[ispin];
                (o.start_mo, o.num_state, o.occ_size, o.vir_size, o.homo, o.lumo)
            };
            let eig = &scf_data.eigenvectors[ispin];
            if eig.size[0] != n_bas
                || n_o == 0
                || n_v == 0
                || homo + 1 - start_mo != n_o
                || eig.size[1] < lumo + n_v
            {
                println!(
                    "bse_matvec_style = \"ao\": inconsistent orbital window for spin \
                     {} (n_bas={}, n_mo={}, start_mo={}, homo={}, lumo={}, n_o={}, n_v={}); keeping \
                     the MO-basis matvec.",
                    ispin, eig.size[0], eig.size[1], start_mo, homo, lumo, n_o, n_v
                );
                return None;
            }
            let x = eig.data.clone();
            let xo = &x[start_mo * n_bas..(homo + 1) * n_bas];
            let xv = &x[lumo * n_bas..(lumo + n_v) * n_bas];

            // the four **unscreened** folds of `J`, each built by streaming the
            // packed RI tensor in blocks of `nb` auxiliary functions
            let mut bq = vec![0.0f64; n_aux * n_v * n_v];
            let mut r = vec![0.0f64; n_aux * n_v * n_o];
            let mut aqq = vec![0.0f64; n_aux * n_o * n_o];
            let mut ztq = vec![0.0f64; n_aux * n_o * n_v];

            bq.par_chunks_mut(nb * n_v * n_v)
                .enumerate()
                .for_each_init(
                    || vec![0.0f64; nb * n_bas * n_bas],
                    |slab, (blk, ob)| {
                        let q0 = blk * nb;
                        let this = (n_aux - q0).min(nb);
                        let slab = &mut slab[..this * n_bas * n_bas];
                        unpack_block(slab, rdata, n_packed, n_bas, q0, this, &pair_keep);
                        fold2_sym_block(slab, xv, n_bas, this, n_v, ob);
                    },
                );
            r.par_chunks_mut(nb * n_v * n_o)
                .enumerate()
                .for_each_init(
                    || vec![0.0f64; nb * n_bas * n_bas],
                    |slab, (blk, ob)| {
                        let q0 = blk * nb;
                        let this = (n_aux - q0).min(nb);
                        let slab = &mut slab[..this * n_bas * n_bas];
                        unpack_block(slab, rdata, n_packed, n_bas, q0, this, &pair_keep);
                        // `(Xv, Xo)` order so that the occupied index `i` is the
                        // fastest one: `r[q*V*O + b*O + i] = ri_ov[q, i + b*O]`.
                        fold_q_block(
                            slab, xv, xo, n_bas, n_bas, this, n_bas, n_bas, n_v, n_o, ob,
                        );
                    },
                );
            aqq.par_chunks_mut(nb * n_o * n_o)
                .enumerate()
                .for_each_init(
                    || vec![0.0f64; nb * n_bas * n_bas],
                    |slab, (blk, ob)| {
                        let q0 = blk * nb;
                        let this = (n_aux - q0).min(nb);
                        let slab = &mut slab[..this * n_bas * n_bas];
                        unpack_block(slab, rdata, n_packed, n_bas, q0, this, &pair_keep);
                        fold2_sym_block(slab, xo, n_bas, this, n_o, ob);
                    },
                );
            ztq.par_chunks_mut(nb * n_o * n_v)
                .enumerate()
                .for_each_init(
                    || vec![0.0f64; nb * n_bas * n_bas],
                    |slab, (blk, ob)| {
                        let q0 = blk * nb;
                        let this = (n_aux - q0).min(nb);
                        let slab = &mut slab[..this * n_bas * n_bas];
                        unpack_block(slab, rdata, n_packed, n_bas, q0, this, &pair_keep);
                        fold_q_block(
                            slab, xo, xv, n_bas, n_bas, this, n_bas, n_bas, n_o, n_v, ob,
                        );
                    },
                );

            // ---- dielectric contraction, applied to the folds (BLAS-3) ----
            //
            //   aq[q,x] = sum_r D[q,r] aqQ[r,x]      (x runs over the O*O pairs)
            //
            // The folds are stored as `M` contiguous slabs, i.e. `[M, n_x]` with
            // `x` **fastest** -- which is the *column-major* `[n_x, M]` matrix
            // `X[x,q] = aqQ[q,x]` (address `x + q*n_x`).  So the contraction is one
            // `dgemm_nt`:  `aq_cm = aqQ_cm * D^T`,  `(n_x, M) = (n_x, M) x (M, M)`.
            let dmat = &inverse_dielectric.data;
            let nx_o = n_o * n_o;
            let mut aq = vec![0.0f64; n_aux * nx_o];
            dgemm_nt(nx_o, n_aux, n_aux, 1.0, &aqq, nx_o, dmat, n_aux, 0.0, &mut aq, nx_o);
            drop(aqq);
            let nx_z = n_o * n_v;
            let mut zt = vec![0.0f64; n_aux * nx_z];
            dgemm_nt(nx_z, n_aux, n_aux, 1.0, &ztq, nx_z, dmat, n_aux, 0.0, &mut zt, nx_z);
            drop(ztq);

            spins.push(SpinChannel {
                n_o,
                n_v,
                n_state,
                start_mo,
                lumo,
                x,
                bq,
                r,
                aq,
                zt,
            });
        }

        let fold_bytes: usize = spins
            .iter()
            .map(|c| (c.bq.len() + c.r.len() + c.aq.len() + c.zt.len()) * 8)
            .sum::<usize>();
        let t_bytes = jdiag.as_ref().map_or(0, |j| j.len() * 8);
        let bytes: usize = fold_bytes + t_bytes + inverse_dielectric.data.len() * 8;
        let build_seconds = start.elapsed().as_secs_f64();
        println!(
            "BSE matvec (memory-efficient): {} spin channel(s), {:.3} MB of AO folds{} (no ri3mo, \
             no ri_vv/ri_oo/ri_ov_tilde, no [OV,OV] matrices), built in {:.3} s",
            n_spin,
            bytes as f64 / 1024.0f64.powi(2),
            if keep_ao_tensor {
                " + the [M,N,N] AO tensor (diagnostic build)"
            } else {
                "; the [M,N,N] AO tensor is released after the build"
            },
            build_seconds
        );

        Some(Self {
            n_bas,
            n_aux,
            jdiag,
            d: inverse_dielectric.data.clone(),
            pair_keep,
            spins,
            exchange_rescaling: qp_ctrl.bse_exchange_rescaling,
            build_seconds,
            bytes,
            blas_threads: scf_data.mol.ctrl.num_threads.unwrap_or(1).max(1),
            matvec_count: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    pub fn n_spin(&self) -> usize {
        self.spins.len()
    }

    pub fn dims(&self, ispin: usize) -> (usize, usize) {
        (self.spins[ispin].n_o, self.spins[ispin].n_v)
    }

    pub fn matvec_count(&self) -> usize {
        self.matvec_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    // --------------------------------------------------------------- matvecs --

    /// A-block matvec, restricted reference.
    pub fn a_block_matvec(&self, scf_data: &SCF, qp_ctrl: &QuasiParticle, z: &[f64]) -> Vec<f64> {
        self.matvec_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let c = &self.spins[0];
        let (o, v) = (c.n_o, c.n_v);
        let nov = o * v;
        assert_eq!(z.len(), nov);
        let qp = &scf_data.gwqp.0;

        let mut result = vec![0.0f64; nov];
        for a in 0..v {
            let ea = qp[c.lumo + a];
            for i in 0..o {
                result[i + a * o] = (ea - qp[c.start_mo + i]) * z[i + a * o];
            }
        }

        let wa = self.exchange_a(0, z);
        let f = self.exchange_rescaling;
        for k in 0..nov {
            result[k] -= f * wa[k];
        }
        match spin_letter(qp_ctrl) {
            'S' => {
                let d = self.direct_term(0, z);
                for k in 0..nov {
                    result[k] += 2.0 * d[k];
                }
            }
            'R' => {
                let d = self.direct_term(0, z);
                for k in 0..nov {
                    result[k] += d[k];
                }
            }
            _ => {}
        }
        result
    }

    /// B-block matvec, restricted reference.  Reproduces the historical
    /// asymmetries: no `bse_exchange_rescaling`, and no direct term for triplets.
    pub fn b_block_matvec(&self, _scf_data: &SCF, qp_ctrl: &QuasiParticle, z: &[f64]) -> Vec<f64> {
        self.matvec_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let c = &self.spins[0];
        let (o, v) = (c.n_o, c.n_v);
        let nov = o * v;
        assert_eq!(z.len(), nov);

        let wb = self.exchange_b(0, z);
        let mut result = vec![0.0f64; nov];
        for k in 0..nov {
            result[k] = -wb[k];
        }
        match spin_letter(qp_ctrl) {
            'S' => {
                let d = self.direct_term(0, z);
                for k in 0..nov {
                    result[k] += 2.0 * d[k];
                }
            }
            'R' => {
                let d = self.direct_term(0, z);
                for k in 0..nov {
                    result[k] += d[k];
                }
            }
            _ => {}
        }
        result
    }

    /// `tq[q,j,a] = sum_{nu,mu} T[q,mu,nu] Xo[nu,j] Xv[mu,a]`, layout
    /// `[M, O, V]` with `a` fastest.
    ///
    /// This single fold is the backbone of both exchange terms: `T_q` is symmetric
    /// in its two AO labels, so one and the same object serves as
    ///
    /// ```text
    ///   B_q[a,b] = sum_{bb} tq[.,b,bb] z[...]      (screened virtual x virtual)
    ///   Z_q[a,j] = sum_{aa} Xv[.,aa] tq[.,j,aa]    (screened virtual x occupied)
    /// ```
    fn fold_tq(&self, ispin: usize) -> Vec<f64> {
        let c = &self.spins[ispin];
        let (o, v, m, nb) = (c.n_o, c.n_v, self.n_aux, self.n_bas);
        let x = &c.x;
        let xo = |j: usize, nu: usize| x[nu + (c.start_mo + j) * nb];
        let xv = |a: usize, mu: usize| x[mu + (c.lumo + a) * nb];
        let mut tq = vec![0.0f64; m * o * v];
        for q in 0..m {
            let oq = &mut tq[q * o * v..(q + 1) * o * v];
            for j in 0..o {
                for a in 0..v {
                    let mut s = 0.0f64;
                    for nu in 0..nb {
                        let xn = xo(j, nu);
                        if xn == 0.0 {
                            continue;
                        }
                        for mu in 0..nb {
                            s += self.t_at(q, mu, nu) * xn * xv(a, mu);
                        }
                    }
                    oq[j * v + a] = s;
                }
            }
        }
        tq
    }

    /// `W_A[i,a] = sum_q sum_{j,b} A_q[i,j] B_q[a,b] z[j + b*O]`.
    ///
    /// Both blocks are folds of the *same* `T_q` with the two MO legs assigned
    /// independently, so this factorises as
    ///
    /// ```text
    ///   tq[q,j,bb]  = sum_{nu,mu} T[q,mu,nu] Xo[nu,j]   Xv[mu,bb]
    ///   Bq[q,a,bb]  = sum_{aa} Xv[aa?] tq[q,bb,aa]      -- see `bq`
    ///   zB[q,j,a]   = sum_bb Bq[q,a,bb] z[j + bb*O]
    ///   A_q[i,j]    = sum_{mu,nu} T[q,mu,nu] Xo[mu,i] Xo[nu,j]
    ///   W_A[i,a]    = sum_q sum_j A_q[i,j] zB[q,j,a]
    /// ```
    pub fn exchange_a(&self, ispin: usize, z: &[f64]) -> Vec<f64> {
        let c = &self.spins[ispin];
        let (o, v, m) = (c.n_o, c.n_v, self.n_aux);
        let nov = o * v;
        assert_eq!(z.len(), nov);

        // Per auxiliary function `q`, two BLAS-3 calls over the *contiguous
        // column-major slabs* of `bq` / `aq`:
        //
        //   tz[V,O]   =  B_q[V,V] * z[O,V]^T       lda = V, ldb = O, ldc = V
        //   acc[O,V] +=  A_q[O,O] * tz[V,O]^T      lda = O, ldb = V, ldc = O
        //
        // i.e. exactly `W_A[i,a] = sum_q sum_j A_q[i,j] (sum_b B_q[a,b] z[j+bO])`,
        // with `tz`/`acc` in column-major pair order.
        //
        // Parallel over `q` (embarrassingly so) with **one BLAS thread per Rayon
        // worker**: letting every BLAS call spawn its own OpenMP team on top of
        // the Rayon pool oversubscribes badly -- measured on the same shape,
        // 4-way Rayon + 1 BLAS thread reaches 294 GFLOP/s while serial-`q` +
        // 4 BLAS threads only reaches 79.
        //
        // The two scratch buffers live in the Rayon `fold` accumulator, so they
        // are allocated once per job instead of once per `q`.
        let w = (0..m)
            .into_par_iter()
            .fold(
                || {
                    omp_set_num_threads_wrapper(1);
                    (vec![0.0f64; nov], vec![0.0f64; v * o])
                },
                |(mut acc, mut tz), q| {
                    let bqq = &c.bq[q * v * v..(q + 1) * v * v];
                    let aqq = &c.aq[q * o * o..(q + 1) * o * o];
                    dgemm_nt(v, o, v, 1.0, bqq, v, z, o, 0.0, &mut tz, v);
                    dgemm_nt(o, v, o, 1.0, aqq, o, &tz, v, 1.0, &mut acc, o);
                    (acc, tz)
                },
            )
            .map(|(acc, _tz)| acc)
            .reduce(
                || vec![0.0f64; nov],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b.iter()) {
                        *x += y;
                    }
                    a
                },
            );
        // Rayon may let the *calling* thread run part of the job, so restore the
        // configured BLAS thread count for it (`molecule_io` uses the same
        // set-1-then-restore convention around its parallel regions).
        omp_set_num_threads_wrapper(self.blas_threads);
        w
    }

    /// `W_B[i,a] = sum_q sum_j sum_b Z_q[j,a] R_q[i,b] z[j + b*O]`.
    ///
    /// `Z_q[j,a] = ri_ov_tilde[q, j + a*O]` is the **screened** occupied x virtual
    /// fold (precomputed as `zt`), `R_q[i,b] = ri_ov[q, i + b*O]` the
    /// **unscreened** one (precomputed as `r`).  Both are `z`-independent; the
    /// `z`-dependent quantity is the inner projection
    ///
    /// ```text
    ///   T2[q,j,i] = sum_b R_q[i,b] z[j + b*O]      -- the summed virtual index
    ///   W_B[i,a]  = sum_q sum_j Z_q[j,a] T2[q,j,i] -- the summed occupied index
    /// ```
    ///
    /// Note that the pair `(j, b)` is *shared* between `z` and the two folds in
    /// different slots: `z`'s occupied index `j` is contracted with `Z_q`, and
    /// `z`'s virtual index `b` is contracted with `R_q`.  This is what the
    /// historical `w_contribution_b_block_dgemm` does (verified against it
    /// element-wise in `tests/test_bse_matvec_vs_historical.rs`); the *free*
    /// indices `i` and `a` sit on `R_q` and `Z_q` respectively.
    pub fn exchange_b(&self, ispin: usize, z: &[f64]) -> Vec<f64> {
        let c = &self.spins[ispin];
        let (o, v, m) = (c.n_o, c.n_v, self.n_aux);
        let nov = o * v;
        assert_eq!(z.len(), nov);

        // Per auxiliary function `q`, two BLAS-3 calls:
        //
        //   T2[O,O]   =  R_q[O,V] * z[O,V]^T       lda = O, ldb = O, ldc = O
        //   acc[O,V] +=  T2[O,O] * Zc[V,O]^T       lda = O, ldb = V, ldc = O
        //
        // `Zc[a,j] = Z_q[j,a]`: the `zt` slab *is* a column-major `[V, O]` matrix
        // (`zt[q*O*V + j*V + a]`), so it is passed by slice with the transpose
        // flag.  `T2[i,j] = sum_b R_q[i,b] z[j+bO]` is what the historical
        // `w_contribution_b_block_dgemm` contracts against `Z_q`.
        let w = (0..m)
            .into_par_iter()
            .fold(
                || {
                    omp_set_num_threads_wrapper(1);
                    (vec![0.0f64; nov], vec![0.0f64; o * o])
                },
                |(mut acc, mut t2), q| {
                    let rq = &c.r[q * v * o..(q + 1) * v * o];
                    let ztq = &c.zt[q * o * v..(q + 1) * o * v];
                    dgemm_nt(o, o, v, 1.0, rq, o, z, o, 0.0, &mut t2, o);
                    dgemm_nt(o, v, o, 1.0, &t2, o, ztq, v, 1.0, &mut acc, o);
                    (acc, t2)
                },
            )
            .map(|(acc, _t2)| acc)
            .reduce(
                || vec![0.0f64; nov],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b.iter()) {
                        *x += y;
                    }
                    a
                },
            );
        omp_set_num_threads_wrapper(self.blas_threads);
        w
    }

    pub fn direct_term(&self, ispin: usize, z: &[f64]) -> Vec<f64> {
        let c = &self.spins[ispin];
        let (o, v, m) = (c.n_o, c.n_v, self.n_aux);
        let nov = o * v;
        assert_eq!(z.len(), nov);
        // Storage convention of `r` (set by the fold in `build`):
        //   r[q*V*O + b*O + i] = R_q[i,b] = ri_ov[q, i + b*O]
        // i.e. the occupied index is the fastest one.
        let mut result = vec![0.0f64; nov];
        for q in 0..m {
            let rq = &c.r[q * v * o..(q + 1) * v * o];
            let mut s = 0.0f64;
            for a in 0..v {
                let ra = &rq[a * o..(a + 1) * o];
                let za = &z[a * o..(a + 1) * o];
                for j in 0..o {
                    s += ra[j] * za[j];
                }
            }
            if s == 0.0 {
                continue;
            }
            for a in 0..v {
                let ra = &rq[a * o..(a + 1) * o];
                let res = &mut result[a * o..(a + 1) * o];
                for i in 0..o {
                    res[i] += s * ra[i];
                }
            }
        }
        result
    }

    // ------------------------------------------------------------ test hooks --

    pub fn bq_at(&self, ispin: usize, q: usize, bb: usize, bp: usize) -> f64 {
        let c = &self.spins[ispin];
        c.bq[q * c.n_v * c.n_v + bp * c.n_v + bb]
    }
    /// Raw storage of the screened occupied x occupied fold.
    pub fn aq_raw(&self, ispin: usize, q: usize, x: usize, y: usize) -> f64 {
        let c = &self.spins[ispin];
        c.aq[q * c.n_o * c.n_o + y * c.n_o + x]
    }
    /// Raw storage of the screened occupied x virtual fold: `zt[q, j*V + a]`.
    pub fn zt_raw(&self, ispin: usize, q: usize, j: usize, a: usize) -> f64 {
        let c = &self.spins[ispin];
        c.zt[(q * c.n_o + j) * c.n_v + a]
    }
    /// `aq[q,i,j]` — element-wise test hook.
    pub fn aq_at(&self, ispin: usize, q: usize, i: usize, j: usize) -> f64 {
        let c = &self.spins[ispin];
        let nb = self.n_bas;
        let x = &c.x;
        let xo = |ii: usize, mu: usize| x[mu + (c.start_mo + ii) * nb];
        let mut s = 0.0f64;
        for nu in 0..nb {
            let xn = xo(j, nu);
            if xn == 0.0 { continue; }
            for mu in 0..nb {
                s += self.t_at(q, mu, nu) * xo(i, mu) * xn;
            }
        }
        s
    }
    /// `tq[q,j,a]` — element-wise test hook.
    pub fn tq_at(&self, ispin: usize, q: usize, j: usize, a: usize) -> f64 {
        let c = &self.spins[ispin];
        self.fold_tq(ispin)[(q * c.n_o + j) * c.n_v + a]
    }
    /// `Z_q[j,a]` (screened) — element-wise test hook.
    pub fn ztilde_at(&self, ispin: usize, q: usize, j: usize, a: usize) -> f64 {
        let c = &self.spins[ispin];
        let nb = self.n_bas;
        let x = &c.x;
        let xo = |jj: usize, nu: usize| x[nu + (c.start_mo + jj) * nb];
        let xv = |aa: usize, mu: usize| x[mu + (c.lumo + aa) * nb];
        let mut s = 0.0f64;
        for nu in 0..nb {
            let xn = xo(j, nu);
            if xn == 0.0 { continue; }
            for mu in 0..nb {
                s += self.t_at(q, mu, nu) * xv(a, mu) * xn;
            }
        }
        s
    }
    /// Raw storage of the direct-term fold: `r[q, b*O + i]` (diagnostic).
    pub fn r_raw(&self, ispin: usize, q: usize, i: usize, a: usize) -> f64 {
        let c = &self.spins[ispin];
        c.r[(q * c.n_v + a) * c.n_o + i]
    }
    /// `R_q[i,b] = ri_ov[q, i + b*O]` (unscreened) — element-wise test hook.
    pub fn rov_at(&self, ispin: usize, q: usize, b: usize, i: usize) -> f64 {
        let c = &self.spins[ispin];
        c.r[(q * c.n_v + b) * c.n_o + i]
    }
    /// `J[q,mu,nu]` -- the raw unscreened AO tensor.
    ///
    /// Only available on a context built with
    /// [`FastBseContext::build_keeping_ao_tensor`].
    #[inline]
    pub fn j_at(&self, q: usize, mu: usize, nu: usize) -> f64 {
        self.jdiag.as_ref().expect(
            "the [M,N,N] AO tensor is streamed and not retained in this build; use \
             FastBseContext::build_keeping_ao_tensor for element-wise diagnostics",
        )[(q * self.n_bas + mu) * self.n_bas + nu]
    }
    /// `T[q,mu,nu] = sum_r D[q,r] J^r[mu,nu]` (diagnostic builds only; the
    /// production build never forms `T` at all).
    #[inline]
    pub fn t_at(&self, q: usize, mu: usize, nu: usize) -> f64 {
        (0..self.n_aux)
            .map(|r| self.d[q + r * self.n_aux] * self.j_at(r, mu, nu))
            .sum()
    }
    /// `true` when the `[M, N, N]` AO tensor is resident (diagnostic builds).
    pub fn has_ao_tensor(&self) -> bool {
        self.jdiag.is_some()
    }
    /// Bytes held by the folds only (i.e. what the matvec actually needs).
    pub fn fold_bytes(&self) -> usize {
        self.spins
            .iter()
            .map(|c| (c.bq.len() + c.r.len() + c.aq.len() + c.zt.len()) * 8)
            .sum()
    }
    pub fn r_at(&self, ispin: usize, q: usize, i: usize, a: usize) -> f64 {
        let c = &self.spins[ispin];
        c.r[(q * c.n_v + a) * c.n_o + i]
    }
}

fn spin_letter(qp_ctrl: &QuasiParticle) -> char {
    if qp_ctrl.bse_spin == "triplet" {
        'T'
    } else if qp_ctrl.bse_spin == "singlet" {
        'S'
    } else {
        'R'
    }
}
