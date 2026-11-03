//! Zero-copy, streaming AO-basis tensor engine for GW (`gw_tensor_style = "ao"`).
//!
//! Design and proofs: `docs/gw_ao_lean_derivation.md`.  The contract is:
//!
//! * the **only** large array involved is `SCF::rimatr` (the packed AO
//!   three-centre tensor the SCF already holds) — it is *borrowed*, never
//!   copied, and never renamed into a second buffer;
//! * **no MO-basis RI tensor is ever materialised**, in particular no `ri_ov`
//!   (`[n_aux, O*V]`) and no `[n_aux, n_mo, n_mo]`;
//! * every routine works in blocks whose size is chosen by the caller, so the
//!   transient cost of a GW pass is `O(n_aux^2)` plus what the caller asks for.
//!
//! # Layout facts this module rests on
//!
//! `MatrixFull` is column-major, so `SCF::rimatr` of logical shape
//! `[n_packed, n_aux]` stores `rimatr[[k, q]]` at `data[q * n_packed + k]`:
//! **the auxiliary index is the slow one**, and a fixed-`q` triangle is a
//! contiguous slice.  `n_packed = N(N+1)/2`, and the packed index of the pair
//! `(hi, lo)` with `hi >= lo` is `hi(hi+1)/2 + lo`.
//!
//! # The folding identity
//!
//! ```text
//!   (Q | n, m) = sum_{mu,nu} J_Q[mu,nu] X[mu,n] X[nu,m]
//!              = sum_nu ( sum_mu J_Q[mu,nu] X[mu,n] ) X[nu,m]
//! ```
//!
//! * [`fold_block`] evaluates it for a **row block**, with one `dsymm` and one
//!   `dgemm` per auxiliary function (BLAS is efficient because the requested
//!   column count is large).
//! * [`GwAoPlan`]-level `ChiBuf::add_orbital` evaluates all
//!   `(Q | start_mo+i, lumo+a)` for **one occupied orbital** with a
//!   packed-triangle sweep plus a single `dgemm`.  That is what makes the RPA
//!   response buildable without `ri_ov`.
//!
//! # Response matrix without `ri_ov`
//!
//! ```text
//!   chi(omega) = sum_i u^(i) diag(f^(i)(omega)) u^(i)T ,   u^(i) in R^{M x V}
//! ```
//!
//! one BLAS-3 rank-`V` update per occupied orbital.  The flop count
//! `M^2 * O * V` is *identical* to the historical `chi = ri_ov diag(f) ri_ov^T`;
//! the only new work is folding `u^(i)`, at `M * O * (N^2/2 + N*V)`.

use rayon::prelude::*;
use rest_tensors::matrix::matrix_blas_lapack::{_dgemm, _dgemm_full, _dsymm};
use rest_tensors::MatrixFull;

use crate::scf_io::SCF;

/// Shape and orbital-window bookkeeping.
#[derive(Clone, Debug)]
pub struct GwAoDims {
    pub n_bas: usize,
    pub n_aux: usize,
    pub n_packed: usize,
    pub start_mo: usize,
    pub num_state: usize,
    pub n_mo: usize,
    pub homo: usize,
    pub lumo: usize,
    pub n_o: usize,
    pub n_v: usize,
}

/// Kept packed indices per row of the triangle (`None` = every pair kept).
struct Screen {
    row_start: Vec<usize>,
    row_items: Vec<usize>,
}

/// The whole state of the AO route.
///
/// It holds **no RI tensor**: only dimensions, the optional pre-screening index
/// and a few counters.  Everything else is read out of the `SCF` at call time.
pub struct GwAoPlan {
    pub dims: GwAoDims,
    pub screening_tol: f64,
    pub kept_pairs: usize,
    pub build_seconds: f64,
    /// Bookkeeping bytes only — the AO tensor itself is **not** owned here.
    pub bytes: usize,
    screen: Option<Screen>,
}

#[inline(always)]
fn pack(hi: usize, lo: usize) -> usize {
    hi * (hi + 1) / 2 + lo
}

/// Row-block size of the `(Q|n m)` folds (`REST_GW_AO_BLOCK`).
fn row_block() -> usize {
    std::env::var("REST_GW_AO_BLOCK")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|b| *b > 0)
        .unwrap_or(8)
}

impl GwAoPlan {
    /// Build the plan without pre-screening.
    pub fn build(scf_data: &SCF) -> Option<Self> {
        Self::build_with_screening(scf_data, 0.0)
    }

    /// Build the plan from `SCF::rimatr` / `SCF::eigenvectors`.
    ///
    /// `screening_tol` drops AO pairs whose largest coefficient over the whole
    /// auxiliary set, **weighted by the MO coefficients of the window**, is
    /// below the threshold:
    ///
    /// ```text
    ///   keep(mu,nu)  <=>  max_Q |J_Q[mu,nu]| * w_mu * w_nu > tol
    ///   w_mu = max_{m in window} |X[mu,m]|
    /// ```
    ///
    /// Since `(Q|n m) = sum_{mu,nu} J_Q[mu,nu] X[mu,n] X[nu,m]`, that product is
    /// exactly the bound on the perturbation of any fitted three-centre
    /// integral caused by dropping the pair.  `0.0` keeps every pair.
    pub fn build_with_screening(scf_data: &SCF, screening_tol: f64) -> Option<Self> {
        let t0 = std::time::Instant::now();
        let n_bas = scf_data.mol.num_basis;

        // `ri_bse::get_submatrix('O','V')` switches to `rimatr_bse` when a BSE
        // auxiliary basis is loaded, while every other GW tensor comes from
        // `rimatr`; the two are different tensors, so the AO route declines.
        if scf_data.rimatr_bse.is_some() || scf_data.ri3fn_bse.is_some() {
            println!(
                "gw_tensor_style = \"ao\": a BSE-specific auxiliary basis is loaded, so the \
                 historical GW code mixes `rimatr_bse` (ri_ov) with `rimatr` (the rows). \
                 Keeping the MO-basis GW tensors."
            );
            return None;
        }
        if scf_data.ri3fn.is_some() {
            println!(
                "gw_tensor_style = \"ao\": `use_ri_symm = false` (full `ri3fn` storage) is not \
                 supported by the AO route; keeping the MO-basis GW tensors."
            );
            return None;
        }
        let rimatr = match &scf_data.rimatr {
            Some((r, _, _)) => r,
            None => {
                println!(
                    "gw_tensor_style = \"ao\" needs the packed AO-basis RI tensor \
                     (`SCF::rimatr`): set `use_ri_symm = true` and provide an auxiliary basis. \
                     Keeping the MO-basis GW tensors."
                );
                return None;
            }
        };
        let n_aux = rimatr.size[1];
        let n_packed = n_bas * (n_bas + 1) / 2;
        if rimatr.size[0] != n_packed || rimatr.data.len() != n_packed * n_aux {
            println!(
                "gw_tensor_style = \"ao\": rimatr has shape [{}, {}] with {} values, expected a \
                 dense [{}, {}]; keeping the MO-basis GW tensors.",
                rimatr.size[0],
                rimatr.size[1],
                rimatr.data.len(),
                n_packed,
                n_aux
            );
            return None;
        }

        let (start_mo, num_state, n_o, n_v, homo, lumo) =
            crate::ri_gw::get_occupation_parameters(scf_data, 'Y');
        let n_mo = num_state - start_mo;
        let eig = &scf_data.eigenvectors[0];
        if eig.size[0] != n_bas || n_o == 0 || n_v == 0 || eig.size[1] < num_state {
            println!(
                "gw_tensor_style = \"ao\": inconsistent orbital window (n_bas={}, n_mo={}, \
                 n_o={}, n_v={}); keeping the MO-basis GW tensors.",
                n_bas, n_mo, n_o, n_v
            );
            return None;
        }

        // ---- optional pre-screening index (at most n_packed usizes) ---------
        let (screen, kept_pairs) = if screening_tol > 0.0 {
            let mut w = vec![0.0f64; n_bas];
            for m in start_mo..num_state {
                for mu in 0..n_bas {
                    let a = eig.data[mu + m * n_bas].abs();
                    if a > w[mu] {
                        w[mu] = a;
                    }
                }
            }
            let mut mx = vec![0.0f64; n_packed];
            for q in 0..n_aux {
                let col = &rimatr.data[q * n_packed..(q + 1) * n_packed];
                for hi in 0..n_bas {
                    let whi = w[hi];
                    if whi == 0.0 {
                        continue;
                    }
                    let base = pack(hi, 0);
                    for lo in 0..=hi {
                        let a = col[base + lo].abs() * whi * w[lo];
                        if a > mx[base + lo] {
                            mx[base + lo] = a;
                        }
                    }
                }
            }
            let mut row_start = Vec::with_capacity(n_bas + 1);
            let mut row_items = Vec::with_capacity(n_packed);
            for hi in 0..n_bas {
                row_start.push(row_items.len());
                let base = pack(hi, 0);
                for lo in 0..=hi {
                    if mx[base + lo] > screening_tol {
                        row_items.push(lo);
                    }
                }
            }
            row_start.push(row_items.len());
            let kept = row_items.len();
            println!(
                "GW tensors (ao): kept {}/{} AO pairs ({:.2}%) at tol = {:.1e} (weighted by \
                 max|X| over the MO window)",
                kept,
                n_packed,
                100.0 * kept as f64 / n_packed as f64,
                screening_tol
            );
            let bytes = row_items.len() * 8 + row_start.len() * 8;
            (Some(Screen { row_start, row_items }), kept)
        } else {
            (None, n_packed)
        };

        let bytes = kept_pairs * 8;
        let dims = GwAoDims {
            n_bas,
            n_aux,
            n_packed,
            start_mo,
            num_state,
            n_mo,
            homo,
            lumo,
            n_o,
            n_v,
        };
        let build_seconds = t0.elapsed().as_secs_f64();
        println!(
            "GW tensors (ao): n_bas={}, n_aux={}, n_mo={} ({} occ, {} vir); plan built in \
             {:.3} s — rimatr is read in place, no RI tensor is copied or stored",
            n_bas, n_aux, n_mo, n_o, n_v, build_seconds
        );
        Some(Self {
            dims,
            screening_tol,
            kept_pairs,
            build_seconds,
            bytes,
            screen,
        })
    }

    /// `true` when this plan still describes `scf_data`.
    pub fn matches(&self, scf_data: &SCF) -> bool {
        let d = &self.dims;
        let (start_mo, num_state, n_o, n_v, _, _) =
            crate::ri_gw::get_occupation_parameters(scf_data, 'Y');
        d.n_bas == scf_data.mol.num_basis
            && d.start_mo == start_mo
            && d.num_state == num_state
            && d.n_o == n_o
            && d.n_v == n_v
            && scf_data.rimatr.is_some()
    }

    /// True when pre-screening is active.
    pub fn screening_active(&self) -> bool {
        self.screen.is_some()
    }
}

// ---------------------------------------------------------------- kernels --

/// `out[q * n_rows*n_cols + r*row_stride + c*col_stride]
///      = sum_{mu,nu} J_Q[mu,nu] X[mu, rows[r]] X[nu, cols0+c]`.
///
/// Reads `SCF::rimatr` directly.  Per worker it allocates
/// `[n_bas,n_bas] + [n_bas,n_rows] + [n_bas,n_cols] + [n_rows,n_cols]`, i.e.
/// `O(N^2)`, and only for the duration of the call.
pub fn fold_block(
    scf_data: &SCF,
    plan: &GwAoPlan,
    rows: &[usize],
    cols0: usize,
    n_cols: usize,
    row_stride: usize,
    col_stride: usize,
    out: &mut [f64],
) {
    if rows.is_empty() || n_cols == 0 {
        return;
    }
    let d = &plan.dims;
    let (n_bas, n_aux, n_packed) = (d.n_bas, d.n_aux, d.n_packed);
    let n_rows = rows.len();
    let block = n_rows * n_cols;
    debug_assert_eq!(out.len(), n_aux * block);

    let jpacked = &scf_data.rimatr.as_ref().expect("rimatr").0.data;
    let x = &scf_data.eigenvectors[0].data;

    type Buffers = (
        MatrixFull<f64>, // sym: J_Q, upper triangle
        MatrixFull<f64>, // xsel: [n_bas, n_rows]
        MatrixFull<f64>, // xc:   [n_bas, n_cols]
        MatrixFull<f64>, // tmp:  [n_bas, n_rows]
        MatrixFull<f64>, // g:    [n_rows, n_cols]
    );
    let make = || -> Buffers {
        let mut xsel = MatrixFull::new([n_bas, n_rows], 0.0);
        for (r, &mo) in rows.iter().enumerate() {
            xsel.data[r * n_bas..(r + 1) * n_bas]
                .copy_from_slice(&x[mo * n_bas..(mo + 1) * n_bas]);
        }
        let mut xc = MatrixFull::new([n_bas, n_cols], 0.0);
        for c in 0..n_cols {
            let src = (cols0 + c) * n_bas;
            xc.data[c * n_bas..(c + 1) * n_bas].copy_from_slice(&x[src..src + n_bas]);
        }
        (
            MatrixFull::new([n_bas, n_bas], 0.0),
            xsel,
            xc,
            MatrixFull::new([n_bas, n_rows], 0.0),
            MatrixFull::new([n_rows, n_cols], 0.0),
        )
    };

    out.par_chunks_mut(block)
        .enumerate()
        .for_each_init(make, |buf, (q, oq)| {
            let (sym, xsel, xc, tmp, g) = buf;
            let jq = &jpacked[q * n_packed..(q + 1) * n_packed];

            match &plan.screen {
                None => {
                    for hi in 0..n_bas {
                        let base = hi * (hi + 1) / 2;
                        let col = &mut sym.data[hi * n_bas..(hi + 1) * n_bas];
                        col[..=hi].copy_from_slice(&jq[base..base + hi + 1]);
                    }
                }
                Some(s) => {
                    for hi in 0..n_bas {
                        let base = hi * (hi + 1) / 2;
                        for &lo in &s.row_items[s.row_start[hi]..s.row_start[hi + 1]] {
                            sym.data[lo + hi * n_bas] = jq[base + lo];
                        }
                    }
                }
            }

            // T = J_Q * X_sel                          (dsymm,  n_bas^2 * n_rows)
            _dsymm(sym, xsel, tmp, 'L', 'U', 1.0, 0.0);
            // (Q|r,c) = X_sel^T * T                    (dgemm,  n_rows*n_bas*n_cols)
            _dgemm(
                tmp,
                (0..n_bas, 0..n_rows),
                'T',
                xc,
                (0..n_bas, 0..n_cols),
                'N',
                g,
                (0..n_rows, 0..n_cols),
                1.0,
                0.0,
            );
            for r in 0..n_rows {
                let obase = r * row_stride;
                for c in 0..n_cols {
                    oq[obase + c * col_stride] = g[[r, c]];
                }
            }
        });
}

/// `[n_aux, n_rows * n_cols]` as a `MatrixFull` (column-major, BLAS-ready).
pub fn mo_block_matfull(
    scf_data: &SCF,
    plan: &GwAoPlan,
    rows: &[usize],
    cols0: usize,
    n_cols: usize,
) -> MatrixFull<f64> {
    let n_aux = plan.dims.n_aux;
    let ncol = rows.len() * n_cols;
    let mut raw = vec![0.0f64; n_aux * ncol];
    fold_block(scf_data, plan, rows, cols0, n_cols, n_cols, 1, &mut raw);
    let mut out = vec![0.0f64; n_aux * ncol];
    out.par_chunks_mut(n_aux).enumerate().for_each(|(col, dst)| {
        for (q, v) in dst.iter_mut().enumerate() {
            *v = raw[q * ncol + col];
        }
    });
    MatrixFull::from_vec([n_aux, ncol], out).unwrap()
}

/// One `ri3mo` row: `MatrixFull [n_aux, n_cols]`, column `c` = `(Q | mo, cols0+c)`.
pub fn ri_row(scf_data: &SCF, plan: &GwAoPlan, mo: usize, cols0: usize, n_cols: usize) -> MatrixFull<f64> {
    mo_block_matfull(scf_data, plan, &[mo], cols0, n_cols)
}

/// `[n_aux, O*V]` in the historical `ri_ov` layout (column `i + a*n_o`).
///
/// Only used by the validation tests — the production AO path never calls it,
/// which is the point of this module.
pub fn ri_ov_materialised(scf_data: &SCF, plan: &GwAoPlan) -> MatrixFull<f64> {
    let d = &plan.dims;
    let rows: Vec<usize> = (d.start_mo..d.start_mo + d.n_o).collect();
    let n_aux = d.n_aux;
    let ncol = d.n_o * d.n_v;
    let mut raw = vec![0.0f64; n_aux * ncol];
    fold_block(scf_data, plan, &rows, d.lumo, d.n_v, 1, d.n_o, &mut raw);
    let mut out = vec![0.0f64; n_aux * ncol];
    out.par_chunks_mut(n_aux).enumerate().for_each(|(col, dst)| {
        for (q, v) in dst.iter_mut().enumerate() {
            *v = raw[q * ncol + col];
        }
    });
    MatrixFull::from_vec([n_aux, ncol], out).unwrap()
}

// ---------------------------------------------- response matrix, no ri_ov --

/// How many occupied orbitals are folded in one sweep of the AO triangle
/// (`REST_GW_AO_SWEEP_BATCH`), subject to a memory cap
/// (`REST_GW_AO_SWEEP_MEM_MB`, default 32 MB) on `batch * n_bas * n_aux`.
fn sweep_batch(n_bas: usize, n_aux: usize, n_o: usize) -> usize {
    let cap_mb: f64 = std::env::var("REST_GW_AO_SWEEP_MEM_MB")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|b| *b > 0.0)
        .unwrap_or(32.0);
    let per_orb = (n_bas * n_aux) as f64 * 8.0 / 1048576.0;
    let by_mem = (cap_mb / per_orb.max(1e-9)).floor() as usize;
    let want = std::env::var("REST_GW_AO_SWEEP_BATCH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|b| *b > 0)
        .unwrap_or(8);
    want.min(by_mem).clamp(1, n_o.max(1))
}

/// Fold `(Q | start_mo+chunk[i], lumo+a)` for a **batch** of occupied orbitals,
/// sharing one read of the AO triangle between them.
///
/// Per auxiliary function:
///
/// ```text
///   unpack J_Q into the upper triangle of a dense [n_bas,n_bas] buffer
///   T_Q = J_Q * X_batch                     (one dsymm, n = batch)
///   scatter T_Q into tb[i][:, Q]
/// ```
///
/// then one `dgemm` per orbital contracts `tb[i]` with the virtual block of the
/// MO coefficients.  Using `dsymm` rather than a scalar triangular sweep per
/// orbital is what keeps the fold at BLAS speed; `batch` amortises the AO
/// triangle read.
///
/// Returns `[n_aux, batch * n_v]` with the orbital index outermost: column
/// `i*n_v + a` is `u^(chunk[i])[., a]`.
fn fold_occ_batch(scf_data: &SCF, plan: &GwAoPlan, chunk: &[usize]) -> MatrixFull<f64> {
    let d = &plan.dims;
    let (n_bas, n_aux, n_packed) = (d.n_bas, d.n_aux, d.n_packed);
    let (n_v, nb) = (d.n_v, chunk.len());
    let jpacked = &scf_data.rimatr.as_ref().expect("rimatr").0.data;
    let x = &scf_data.eigenvectors[0].data;

    let mut sym = MatrixFull::new([n_bas, n_bas], 0.0);
    let mut xb = MatrixFull::new([n_bas, nb], 0.0);
    for (bi, &i) in chunk.iter().enumerate() {
        let src = (d.start_mo + i) * n_bas;
        xb.data[bi * n_bas..(bi + 1) * n_bas].copy_from_slice(&x[src..src + n_bas]);
    }
    let mut tq = MatrixFull::new([n_bas, nb], 0.0);
    let mut tb: Vec<MatrixFull<f64>> =
        (0..nb).map(|_| MatrixFull::new([n_bas, n_aux], 0.0)).collect();

    for q in 0..n_aux {
        let jq = &jpacked[q * n_packed..(q + 1) * n_packed];
        match &plan.screen {
            None => {
                for hi in 0..n_bas {
                    let base = hi * (hi + 1) / 2;
                    let col = &mut sym.data[hi * n_bas..(hi + 1) * n_bas];
                    col[..=hi].copy_from_slice(&jq[base..base + hi + 1]);
                }
            }
            Some(sc) => {
                for hi in 0..n_bas {
                    let base = hi * (hi + 1) / 2;
                    for &lo in &sc.row_items[sc.row_start[hi]..sc.row_start[hi + 1]] {
                        sym.data[lo + hi * n_bas] = jq[base + lo];
                    }
                }
            }
        }
        // T_Q = J_Q * X_batch   (dsymm, n_bas^2 * batch)
        _dsymm(&sym, &xb, &mut tq, 'L', 'U', 1.0, 0.0);
        for bi in 0..nb {
            tb[bi].data[q * n_bas..(q + 1) * n_bas]
                .copy_from_slice(&tq.data[bi * n_bas..(bi + 1) * n_bas]);
        }
    }

    // u^(i)[., a] = tb[i]^T * X[:, lumo..lumo+n_v]
    let mut ucat = MatrixFull::new([n_aux, nb * n_v], 0.0);
    for bi in 0..nb {
        let mut u = MatrixFull::new([n_aux, n_v], 0.0);
        _dgemm(
            &tb[bi],
            (0..n_bas, 0..n_aux),
            'T',
            &scf_data.eigenvectors[0],
            (0..n_bas, d.lumo..d.lumo + n_v),
            'N',
            &mut u,
            (0..n_aux, 0..n_v),
            1.0,
            0.0,
        );
        ucat.data[bi * n_v * n_aux..(bi + 1) * n_v * n_aux].copy_from_slice(&u.data);
    }
    ucat
}

/// Frequency factor of the RPA response, exactly as `response_matrix_per_spin`.
#[inline]
fn response_factor(de: f64, omega: f64, part: char, eta: f64) -> f64 {
    if part == 'I' {
        -2.0 * de / (de * de + omega * omega)
    } else {
        let de2 = de * de;
        let om2 = omega * omega;
        let eta2 = eta * eta;
        let num = de2 - om2 + eta2;
        let den = (de2 - om2).powi(2) + 2.0 * eta2 * (de2 + om2) + eta2 * eta2;
        if den.abs() < 1e-30 {
            0.0
        } else {
            -2.0 * de * num / den
        }
    }
}

/// `u[Q, a] = (Q | start_mo+i, lumo+a)`, `[n_aux, n_v]`, built by the batched
/// streaming fold the response matrix uses.  Exposed for validation.
pub fn occ_vir_block(scf_data: &SCF, plan: &GwAoPlan, i_local: usize) -> MatrixFull<f64> {
    let d = &plan.dims;
    let ucat = fold_occ_batch(scf_data, plan, &[i_local]);
    let mut u = MatrixFull::new([d.n_aux, d.n_v], 0.0);
    u.data.copy_from_slice(&ucat.data);
    u
}

/// RPA response matrix, one spin channel (historical `response_matrix_per_spin`).
///
/// `chi(omega)[Q,Q'] = sum_{i,a} f_ia (Q|ia)(Q'|ia)`, built as
/// `sum_i u^(i) diag(f^(i)) u^(i)T` with `u^(i)` folded from `rimatr` on the
/// fly.  Parallel over the occupied orbitals, one `[n_aux,n_aux]` accumulator
/// per worker; `ri_ov` is never formed.
pub fn response_matrix_per_spin(
    scf_data: &SCF,
    plan: &GwAoPlan,
    qp_w: &[f64],
    omega: f64,
    part: char,
    eta: f64,
) -> MatrixFull<f64> {
    let d = &plan.dims;
    let (n_aux, n_bas, n_o, n_v) = (d.n_aux, d.n_bas, d.n_o, d.n_v);
    // One accumulator per chunk; the chunk count is bounded by the batch size,
    // so the `[n_aux, n_aux]` working set stays at `n_threads * n_aux^2`.
    let nb = sweep_batch(n_bas, n_aux, n_o);
    let idx: Vec<usize> = (0..n_o).collect();
    idx.par_chunks(nb)
        .map(|chunk| {
            // u^(i) for the whole batch, sharing one read of the AO triangle
            let ucat = fold_occ_batch(scf_data, plan, chunk);
            // chi += U diag(f) U^T for the batch in ONE BLAS-3 call
            // (k = batch * n_v, so the GEMM is as efficient as the MO route's).
            let mut uscat = MatrixFull::new([n_aux, chunk.len() * n_v], 0.0);
            for (bi, &i) in chunk.iter().enumerate() {
                for a in 0..n_v {
                    let f = response_factor(qp_w[n_o + a] - qp_w[i], omega, part, eta);
                    let col = bi * n_v + a;
                    let src = &ucat.data[col * n_aux..(col + 1) * n_aux];
                    let dst = &mut uscat.data[col * n_aux..(col + 1) * n_aux];
                    for (dd, ss) in dst.iter_mut().zip(src.iter()) {
                        *dd = f * ss;
                    }
                }
            }
            let mut chi = MatrixFull::new([n_aux, n_aux], 0.0);
            _dgemm_full(&uscat, 'N', &ucat, 'T', &mut chi, 1.0, 0.0);
            chi
        })
        .reduce(
            || MatrixFull::new([n_aux, n_aux], 0.0),
            |mut a, b| {
                for (x, y) in a.data.iter_mut().zip(b.data.iter()) {
                    *x += y;
                }
                a
            },
        )
}

/// Restricted wrapper: one spatial channel times two physical spins
/// (historical `response_matrix`).
pub fn response_matrix(
    scf_data: &SCF,
    plan: &GwAoPlan,
    qp_w: &[f64],
    omega: f64,
    part: char,
    eta: f64,
) -> MatrixFull<f64> {
    let mut r = response_matrix_per_spin(scf_data, plan, qp_w, omega, part, eta);
    for v in r.data.iter_mut() {
        *v *= 2.0;
    }
    r
}

// -------------------------------------------------------------- V and W_c --

/// `V[n,m] = sum_Q (Q|n m)^2` over the MO window.
pub fn v_matrix(scf_data: &SCF, plan: &GwAoPlan) -> MatrixFull<f64> {
    let d = &plan.dims;
    let nmo = d.n_mo;
    let mut v = MatrixFull::new([d.num_state, d.num_state], 0.0);
    let block = row_block().min(nmo).max(1);
    let mut r0 = 0usize;
    while r0 < nmo {
        let r1 = (r0 + block).min(nmo);
        let rows: Vec<usize> = (d.start_mo + r0..d.start_mo + r1).collect();
        let n_cols = nmo - r0; // (Q|n m) = (Q|m n): only m >= n is needed
        let g = mo_block_matfull(scf_data, plan, &rows, d.start_mo + r0, n_cols);
        for ln in 0..(r1 - r0) {
            let gn = r0 + ln;
            for lc in 0..n_cols {
                let gm = r0 + lc;
                let s = (ln * n_cols + lc) * d.n_aux;
                let sum: f64 = g.data[s..s + d.n_aux].iter().map(|x| x * x).sum();
                v[[gn, gm]] = sum;
                v[[gm, gn]] = sum;
            }
        }
        r0 = r1;
    }
    v
}

/// `W_c[n,m] = sum_Q (Q|n m) * (eps^-1 (Q|n m))_Q` for every entry of
/// `inverse_dielectrics`, written into `out` in the same order.
///
/// The fold `(Q|n m)` is frequency independent, so for a given row block it is
/// evaluated **once** and reused for every matrix in the slice.  The caller
/// controls the peak by choosing how many frequencies to hand over per call.
pub fn w_c_matrices(
    scf_data: &SCF,
    plan: &GwAoPlan,
    inverse_dielectrics: &[MatrixFull<f64>],
    out: &mut [MatrixFull<f64>],
) {
    let d = &plan.dims;
    let nmo = d.n_mo;
    let n_aux = d.n_aux;
    assert_eq!(out.len(), inverse_dielectrics.len());
    if out.is_empty() {
        return;
    }
    for inv in inverse_dielectrics {
        assert_eq!(
            inv.size,
            [n_aux, n_aux],
            "w_c_matrices: inverse dielectric has shape {:?}, expected [{}, {}]",
            inv.size,
            n_aux,
            n_aux
        );
    }
    let block = row_block().min(nmo).max(1);
    let mut r0 = 0usize;
    while r0 < nmo {
        let r1 = (r0 + block).min(nmo);
        let bn = r1 - r0;
        let n_cols = nmo - r0;
        let rows: Vec<usize> = (d.start_mo + r0..d.start_mo + r1).collect();
        let g = mo_block_matfull(scf_data, plan, &rows, d.start_mo + r0, n_cols);

        out.par_iter_mut()
            .zip(inverse_dielectrics.par_iter())
            .for_each_init(
                || MatrixFull::new([n_aux, bn * n_cols], 0.0),
                |tmp, (w_c, inv)| {
                    _dgemm_full(inv, 'N', &g, 'N', tmp, 1.0, 0.0);
                    for ln in 0..bn {
                        let gn = r0 + ln;
                        for lc in 0..n_cols {
                            let gm = r0 + lc;
                            let s = (ln * n_cols + lc) * n_aux;
                            let dot: f64 = g.data[s..s + n_aux]
                                .iter()
                                .zip(tmp.data[s..s + n_aux].iter())
                                .map(|(a, b)| a * b)
                                .sum();
                            w_c[[gn, gm]] = dot;
                            w_c[[gm, gn]] = dot;
                        }
                    }
                },
            );
        r0 = r1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_index_matches_triangle_rows() {
        for hi in 0..8 {
            assert_eq!(pack(hi, 0), hi * (hi + 1) / 2);
            for lo in 0..=hi {
                assert_eq!(pack(hi, lo), hi * (hi + 1) / 2 + lo);
            }
        }
    }
}
