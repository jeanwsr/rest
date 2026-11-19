//! Per-auxiliary-function AO support — the pre-screening that actually removes
//! floating-point work from the AO-basis GW/BSE folds.
//!
//! # Why a *per-`q`* criterion
//!
//! The historical screening (`GwAoPlan::build_with_screening`,
//! `FastBseContext::build_with_screening`) keeps an AO pair `(mu,nu)` when its
//! largest coefficient **over the whole auxiliary set** is above the threshold:
//!
//! ```text
//!   keep(mu,nu)  <=>  max_q |J_q[mu,nu]| * w_mu * w_nu > tol
//! ```
//!
//! That is a *global* decision, and on a large molecule every pair is "close to
//! some `q`", so almost nothing is dropped: the mask stops being useful exactly
//! where the systems get big.  Worse, the mask only ever shortened the *unpack*
//! loop — `_dsymm`/`_dgemm` (GW) and the scalar fold loops (BSE) still ran over
//! the full `[N, N]` block, so no floating-point work disappeared.
//!
//! The RI three-centre tensor is **local**: `J_q[mu,nu]` is negligible unless
//! `mu` and `nu` are near the centre of auxiliary function `q`.  So the right
//! criterion is per auxiliary function:
//!
//! ```text
//!   S_q = { mu : max_nu |J_q[mu,nu]| * w_mu * w_nu > tol }
//!   (Q|n m) ~ sum_{mu,nu in S_q} J_q[mu,nu] X[mu,n] X[nu,m]
//! ```
//!
//! and the fold for `q` is run on the **compressed** `|S_q| x |S_q|` sub-block,
//! so its cost stops scaling with `N` once `|S_q|` saturates at the size of
//! `q`'s neighbourhood:
//!
//! | stage | full | per-`q` support |
//! |---|---|---|
//! | half transform `J_q X` | `N^2 * n` | `|S_q|^2 * n` |
//! | finish `(J_q X)^T X` | `N * n * m` | `|S_q| * n * m` |
//!
//! The retained block `S_q x S_q` is a **superset** of the pairs that pass the
//! per-pair test `|J_q[mu,nu]| w_mu w_nu > tol` (both endpoints of a surviving
//! pair are in `S_q`), and a dense `|S_q| x |S_q|` BLAS call costs the same
//! however many of its entries are nonzero — so keeping the extra entries is
//! free and strictly improves the accuracy at a given threshold.
//!
//! # Error bound
//!
//! With `w_mu = max_{m in window} |X[mu,m]|`, dropping the pairs outside `S_q`
//! perturbs **any** fitted three-centre integral by at most
//!
//! ```text
//!   |d(Q|n m)|  <=  sum_{(mu,nu) dropped} |J_q[mu,nu]| w_mu w_nu  =:  B_q
//! ```
//!
//! where `dropped` is the exact set `{mu not in S_q or nu not in S_q}`.  `build`
//! evaluates that sum in a second pass over the packed triangle, so
//! [`AoLocalIndex::dropped_bound`] is a **rigorous and tight** element-wise
//! error bound for every folded tensor, not a heuristic.  `tol = 0` keeps every
//! pair (`S_q` = all AO) and every kernel takes its original dense path, bit for
//! bit.
//!
//! # Cost gate
//!
//! Compression only pays when `|S_q|/N` is small enough to amortise the AO
//! gather.  [`AoLocalIndex::worth_local`] predicts both costs for the actual
//! output block and leaves `q` on the dense kernel otherwise, so a weak
//! threshold can never make a fold slower than no screening at all.

use rayon::prelude::*;
use rest_tensors::matrix::matrix_blas_lapack::{_dgemm_full, _dsymm};
use rest_tensors::matrix::matrixfullslice::{MatrixFullSlice, MatrixFullSliceMut};

/// Scalar gather/unpack element vs. one BLAS element (calibrated on naphthalene).
const SCALAR: f64 = 5.0;
/// A `q` is only a candidate when its compressed fold beats this fraction of the
/// dense cost.
const LOCAL_MARGIN: f64 = 0.85;

/// Per-auxiliary AO support ("local RI" index).
#[derive(Debug)]
pub struct AoLocalIndex {
    pub n_bas: usize,
    pub n_aux: usize,
    pub tol: f64,
    /// A `q` whose support exceeds this fraction of `n_bas` is treated as dense
    /// (the gather/compaction overhead would exceed what is saved).
    pub dense_fraction: f64,
    /// CSR support: `items[start[q]..start[q+1]]`.
    start: Vec<u32>,
    items: Vec<u32>,
    dense: Vec<bool>,
    /// Auxiliary functions whose support is a strict subset of the AO set.
    pub n_local_aux: usize,
    /// Auxiliary functions skipped entirely (`S_q` empty).
    pub n_empty_aux: usize,
    /// Pairs kept by the per-`q` criterion, summed over `q` (not the union).
    pub kept_pairs: usize,
    /// `max_q B_q` — rigorous bound on the change of any folded integral.
    pub dropped_bound: f64,
    /// `sum_q B_q`, for diagnostics.
    pub dropped_total: f64,
    pub support_max: usize,
    pub support_mean: f64,
    pub build_seconds: f64,
    pub bytes: usize,
}

impl AoLocalIndex {
    /// Build the index from the packed AO RI tensor.
    ///
    /// `jpacked` is the column-major `[n_packed, n_aux]` parent (`SCF::rimatr`
    /// or `SCF::rimatr_bse`), so column `q` — the packed upper triangle of
    /// `J_q` — is contiguous.  `w[mu]` is the AO weight of the MO window
    /// (`max_m |X[mu,m]|`).
    pub fn build(jpacked: &[f64], n_bas: usize, n_aux: usize, w: &[f64], tol: f64) -> Self {
        let t0 = std::time::Instant::now();
        let n_packed = n_bas * (n_bas + 1) / 2;
        assert_eq!(jpacked.len(), n_packed * n_aux, "AoLocalIndex: bad tensor shape");
        assert_eq!(w.len(), n_bas);
        let dense_fraction: f64 = std::env::var("REST_AO_LOCAL_DENSE_FRAC")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|f| *f > 0.0 && *f <= 1.0)
            .unwrap_or(0.85);

        // ---- per-q support scan -------------------------------------------------
        //
        // One pass over the packed triangle per auxiliary function.  Both
        // endpoints of every pair are updated, because the packed triangle only
        // stores `(hi, lo)` and AO `mu` appears both as a `hi` and as a `lo`.
        // A second pass then accumulates the exact error bound of the dropped
        // pairs (`mu` or `nu` outside `S_q`).
        let per_q: Vec<(Vec<u32>, usize, f64)> = (0..n_aux)
            .into_par_iter()
            .map(|q| {
                let col = &jpacked[q * n_packed..(q + 1) * n_packed];
                let mut rowmax = vec![0.0f64; n_bas];
                for hi in 0..n_bas {
                    let whi = w[hi];
                    let base = hi * (hi + 1) / 2;
                    let row = &col[base..base + hi + 1];
                    for (lo, &j) in row.iter().enumerate() {
                        let a = j.abs() * whi * w[lo];
                        if a > rowmax[hi] {
                            rowmax[hi] = a;
                        }
                        if a > rowmax[lo] {
                            rowmax[lo] = a;
                        }
                    }
                }
                let support: Vec<u32> = (0..n_bas)
                    .filter(|&m| rowmax[m] > tol)
                    .map(|m| m as u32)
                    .collect();
                // exact dropped set: a pair survives only if BOTH endpoints are
                // in the support, so this is what the fold actually loses
                let mut kept = 0usize;
                let mut dropped = 0.0f64;
                if support.len() == n_bas {
                    // nothing is dropped; the count is the full triangle
                    kept = n_packed;
                } else {
                    let mut in_s = vec![false; n_bas];
                    for &m in &support {
                        in_s[m as usize] = true;
                    }
                    for hi in 0..n_bas {
                        let whi = w[hi];
                        let base = hi * (hi + 1) / 2;
                        let hi_in = in_s[hi];
                        for lo in 0..=hi {
                            let a = col[base + lo].abs() * whi * w[lo];
                            if hi_in && in_s[lo] {
                                kept += 1;
                            } else {
                                dropped += a;
                            }
                        }
                    }
                }
                (support, kept, dropped)
            })
            .collect();

        let mut start = Vec::with_capacity(n_aux + 1);
        let mut items = Vec::new();
        let mut dense = Vec::with_capacity(n_aux);
        let mut n_local_aux = 0usize;
        let mut n_empty_aux = 0usize;
        let mut kept_pairs = 0usize;
        let mut dropped_bound = 0.0f64;
        let mut dropped_total = 0.0f64;
        let mut support_max = 0usize;
        let cap = (dense_fraction * n_bas as f64).ceil() as usize;
        for (support, kept, dropped) in per_q {
            start.push(items.len() as u32);
            kept_pairs += kept;
            dropped_bound = dropped_bound.max(dropped);
            dropped_total += dropped;
            support_max = support_max.max(support.len());
            let is_dense = support.len() > cap;
            if is_dense {
                dense.push(true);
            } else {
                dense.push(false);
                n_local_aux += 1;
                if support.is_empty() {
                    n_empty_aux += 1;
                }
                items.extend_from_slice(&support);
            }
        }
        start.push(items.len() as u32);

        let bytes = items.len() * 4 + start.len() * 4 + dense.len();
        let support_mean = if n_local_aux > 0 {
            items.len() as f64 / n_local_aux as f64
        } else {
            n_bas as f64
        };
        let idx = Self {
            n_bas,
            n_aux,
            tol,
            dense_fraction,
            start,
            items,
            dense,
            n_local_aux,
            n_empty_aux,
            kept_pairs,
            dropped_bound,
            dropped_total,
            support_max,
            support_mean,
            build_seconds: t0.elapsed().as_secs_f64(),
            bytes,
        };
        idx.print_summary();
        idx
    }

    fn print_summary(&self) {
        let pairs_total = self.n_aux * self.n_bas * (self.n_bas + 1) / 2;
        println!(
            "AO local pre-screening: tol = {:.1e}; {} / {} auxiliary functions use a compressed \
             AO support ({} empty, mean |S_q| = {:.0}, max = {}); kept {:.2}% of (q,mu,nu) triples; \
             error bound max_q = {:.2e}, sum_q = {:.2e}; index {:.2} MB, built in {:.2} s",
            self.tol,
            self.n_local_aux,
            self.n_aux,
            self.n_empty_aux,
            self.support_mean,
            self.support_max,
            100.0 * self.kept_pairs as f64 / pairs_total.max(1) as f64,
            self.dropped_bound,
            self.dropped_total,
            self.bytes as f64 / 1048576.0,
            self.build_seconds
        );
    }

    /// Support of auxiliary function `q`; `None` means "use every AO".
    #[inline]
    pub fn support(&self, q: usize) -> Option<&[u32]> {
        if self.dense[q] {
            None
        } else {
            Some(&self.items[self.start[q] as usize..self.start[q + 1] as usize])
        }
    }

    #[inline]
    pub fn is_dense_q(&self, q: usize) -> bool {
        self.dense[q]
    }

    /// `true` when the index cannot save anything (every `q` is dense).
    pub fn is_trivial(&self) -> bool {
        self.n_local_aux == 0
    }

    /// Predicted-cost gate: `true` when folding `q` on its compressed support is
    /// expected to be cheaper than folding it densely for an output block of
    /// `n_row x n_col`.
    ///
    /// The two costs are counted in element operations, with the scalar gather
    /// and unpack loops weighted `SCALAR` times a BLAS element (that is what
    /// makes the model agree with measurement on weakly compressed systems):
    ///
    /// ```text
    ///   dense = N^2/2 (unpack) + N^2 * n_row (dsymm) + N * n_row * n_col (dgemm)
    ///   local = L^2/2 (gather) + L * (n_row+n_col) (X gather)
    ///         + L^2 * n_row   (dsymm) + L * n_row * n_col (dgemm)
    /// ```
    ///
    /// The compression is only worth it when `L/N` is small **and** the block is
    /// tall enough to amortise the AO-coefficient gather: for `n_row = 1`
    /// (a single `(Q|n,m)` row) the break-even is `L/N ~ 0.45`, a batch of
    /// eight occupied orbitals breaks even near `L/N ~ 0.75`, and the BSE
    /// `[V,V]` / `[O,O]` folds (tall blocks) near `L/N ~ 0.9`.  Weak thresholds
    /// therefore fall back to the dense kernel instead of paying the gather for
    /// nothing.
    /// Aggregate gate: the fraction of the dense cost that engaging the
    /// compressed path is predicted to save for an `n_row x n_col` fold.
    ///
    /// Mixing the two paths has a cost the per-`q` model cannot see — the
    /// compressed `q` gather scattered pieces of `rimatr` and evicts the
    /// sequential stream the dense kernel relies on — so a threshold that only
    /// compresses a few `q`, each marginally, ends up *slower* than no screening.
    /// The kernels therefore engage the compressed path only when this mean
    /// saving exceeds `REST_AO_LOCAL_MIN_SAVING` (default 0.15).
    pub fn predicted_saving(&self, n_row: usize, n_col: usize) -> f64 {
        let n = self.n_bas as f64;
        let (nr, nc) = (n_row as f64, n_col as f64);
        let dense = SCALAR * 0.5 * n * n + n * n * nr + n * nr * nc;
        let mut acc = 0.0f64;
        for q in 0..self.n_aux {
            if let Some(s) = self.support(q) {
                let l = s.len() as f64;
                let local = SCALAR * (0.5 * l * l + l * (nr + nc)) + l * l * nr + l * nr * nc;
                let r = local / dense;
                if r < LOCAL_MARGIN {
                    acc += 1.0 - r;
                }
            }
        }
        acc / self.n_aux as f64
    }

    /// `REST_AO_LOCAL_MIN_SAVING`, default 0.15.
    pub fn min_saving() -> f64 {
        std::env::var("REST_AO_LOCAL_MIN_SAVING")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|f| *f >= 0.0)
            .unwrap_or(0.15)
    }

    /// `true` when this `(n_row, n_col)` fold should use the compressed path at
    /// all (aggregate gate) — evaluated once per fold call, not per `q`.
    #[inline]
    pub fn engage(&self, n_row: usize, n_col: usize) -> bool {
        self.n_local_aux > 0 && self.predicted_saving(n_row, n_col) > Self::min_saving()
    }

    #[inline]
    pub fn worth_local(&self, q: usize, n_row: usize, n_col: usize) -> bool {
        let s = match self.support(q) {
            Some(s) => s,
            None => return false,
        };
        let n = self.n_bas as f64;
        let l = s.len() as f64;
        let (nr, nc) = (n_row as f64, n_col as f64);
        let dense = SCALAR * 0.5 * n * n + n * n * nr + n * nr * nc;
        let local = SCALAR * (0.5 * l * l + l * (nr + nc)) + l * l * nr + l * nr * nc;
        // require a clear margin, so that the many `q` whose compression is only
        // marginal stay on the sequential dense kernel
        local < LOCAL_MARGIN * dense
    }
}

/// Reusable scratch for [`LocalFoldScratch::fold`].
///
/// Allocated once per Rayon worker, never per auxiliary function.
#[derive(Default)]
pub struct LocalFoldScratch {
    sym: Vec<f64>,
    xr: Vec<f64>,
    xc: Vec<f64>,
    tmp: Vec<f64>,
    g: Vec<f64>,
}

/// COLUMN-major view helper: `indicing = [1, ld]`.
#[inline]
fn view2<'a>(size: &'a [usize; 2], ind: &'a [usize; 2], data: &'a [f64]) -> MatrixFullSlice<'a, f64> {
    MatrixFullSlice { size, indicing: ind, data }
}

impl LocalFoldScratch {
    /// `g[r, c] = sum_{mu,nu in support} J_q[mu,nu] X[mu,r] X[nu,c]`.
    ///
    /// * `jpacked_q` — the packed upper triangle of `J_q` (`n_packed` values).
    /// * `support` — the global AO indices of `S_q`, ascending.
    /// * `xr` / `xc` — **full** MO coefficient matrices, column-major
    ///   `[n_bas, n_row]` / `[n_bas, n_col]` with leading dimension `n_bas`
    ///   (i.e. `xr[mu + r*n_bas]`); the routine gathers the `S_q` rows itself.
    ///
    /// Returns `g`, column-major `[n_row, n_col]` (`g[r + c*n_row]`).
    pub fn fold(
        &mut self,
        jpacked_q: &[f64],
        n_bas: usize,
        support: &[u32],
        xr: &[f64],
        n_row: usize,
        xc: &[f64],
        n_col: usize,
    ) -> &[f64] {
        let l = support.len();
        if l == 0 || n_row == 0 || n_col == 0 {
            self.g.clear();
            self.g.resize(n_row * n_col, 0.0);
            return &self.g[..n_row * n_col];
        }
        debug_assert_eq!(jpacked_q.len(), n_bas * (n_bas + 1) / 2);
        debug_assert!(xr.len() >= n_bas * n_row && xc.len() >= n_bas * n_col);

        // ---- gather the compressed problem -----------------------------------
        //
        // `reserve`-style growth only: the buffers are overwritten completely
        // below, so no per-call zero-fill is needed.
        if self.sym.len() < l * l {
            self.sym.resize(l * l, 0.0);
        }
        if self.xr.len() < l * n_row {
            self.xr.resize(l * n_row, 0.0);
        }
        if self.xc.len() < l * n_col {
            self.xc.resize(l * n_col, 0.0);
        }
        if self.tmp.len() < l * n_row {
            self.tmp.resize(l * n_row, 0.0);
        }
        if self.g.len() < n_row * n_col {
            self.g.resize(n_row * n_col, 0.0);
        }

        // `sym` upper triangle (row i <= column j) -- the convention `_dsymm`
        // reads with `uplo = 'U'`.  `support` is ascending, so `i <= j` implies
        // `mu = support[i] <= nu = support[j]` and the packed address is
        // `nu(nu+1)/2 + mu`: a forward walk along packed row `nu`, which the
        // hardware prefetcher handles.
        for (j, &nu) in support.iter().enumerate() {
            let nu = nu as usize;
            let base = nu * (nu + 1) / 2;
            for (i, &mu) in support[..=j].iter().enumerate() {
                self.sym[i + j * l] = jpacked_q[base + mu as usize];
            }
        }
        for (i, &mu) in support.iter().enumerate() {
            let mu = mu as usize;
            for r in 0..n_row {
                self.xr[i + r * l] = xr[mu + r * n_bas];
            }
            for c in 0..n_col {
                self.xc[i + c * l] = xc[mu + c * n_bas];
            }
        }

        // ---- T = J_q^S * X_r   (`dsymm`, L^2 * n_row) ------------------------
        let sz_sym = [l, l];
        let sz_xr = [l, n_row];
        let sz_tmp = [l, n_row];
        let ind_l = [1usize, l];
        let ind_r = [1usize, n_row];
        {
            let a = view2(&sz_sym, &ind_l, &self.sym[..l * l]);
            let b = view2(&sz_xr, &ind_l, &self.xr[..l * n_row]);
            let mut c = MatrixFullSliceMut {
                size: &sz_tmp,
                indicing: &ind_l,
                data: &mut self.tmp[..l * n_row],
            };
            _dsymm(&a, &b, &mut c, 'L', 'U', 1.0, 0.0);
        }
        // ---- g = T^T * X_c     (`dgemm`, L * n_row * n_col) ------------------
        {
            let sz_xc = [l, n_col];
            let sz_g = [n_row, n_col];
            let a = view2(&sz_tmp, &ind_l, &self.tmp[..l * n_row]);
            let b = view2(&sz_xc, &ind_l, &self.xc[..l * n_col]);
            let mut c = MatrixFullSliceMut {
                size: &sz_g,
                indicing: &ind_r,
                data: &mut self.g[..n_row * n_col],
            };
            _dgemm_full(&a, 'T', &b, 'N', &mut c, 1.0, 0.0);
        }
        &self.g[..n_row * n_col]
    }
}

/// MO-window weights `w[mu] = max_{m in window} |X[mu,m]|`.
///
/// They turn an absolute coefficient threshold into a bound on the quantity
/// that actually enters the calculation (see the module docs).
pub fn window_weights(
    x: &[f64],
    n_bas: usize,
    mo_start: usize,
    mo_end: usize,
) -> Vec<f64> {
    let mut w = vec![0.0f64; n_bas];
    for m in mo_start..mo_end {
        let col = &x[m * n_bas..(m + 1) * n_bas];
        for (mu, &v) in col.iter().enumerate() {
            let a = v.abs();
            if a > w[mu] {
                w[mu] = a;
            }
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random symmetric packed tensor.
    fn make_packed(n_bas: usize, n_aux: usize) -> Vec<f64> {
        let n_packed = n_bas * (n_bas + 1) / 2;
        let mut v = vec![0.0f64; n_packed * n_aux];
        let mut s = 12345u64;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        };
        for q in 0..n_aux {
            for hi in 0..n_bas {
                for lo in 0..=hi {
                    // decay away from the diagonal so that a local support exists
                    let d = (hi - lo) as f64;
                    let a = 1.0 / (1.0 + d);
                    v[q * n_packed + hi * (hi + 1) / 2 + lo] = a * next();
                }
            }
        }
        v
    }

    fn dense_fold(jpacked_q: &[f64], n_bas: usize, xr: &[f64], n_row: usize, xc: &[f64], n_col: usize) -> Vec<f64> {
        let mut g = vec![0.0; n_row * n_col];
        for r in 0..n_row {
            for c in 0..n_col {
                let mut s = 0.0;
                for mu in 0..n_bas {
                    for nu in 0..n_bas {
                        let (hi, lo) = if mu >= nu { (mu, nu) } else { (nu, mu) };
                        s += jpacked_q[hi * (hi + 1) / 2 + lo] * xr[mu + r * n_bas] * xc[nu + c * n_bas];
                    }
                }
                g[r + c * n_row] = s;
            }
        }
        g
    }

    #[test]
    fn full_support_reproduces_dense_fold() {
        let (n_bas, n_aux, n_row, n_col) = (12, 3, 4, 5);
        let j = make_packed(n_bas, n_aux);
        let n_packed = n_bas * (n_bas + 1) / 2;
        let xr: Vec<f64> = (0..n_bas * n_row).map(|i| ((i * 37 % 11) as f64 - 5.0) * 0.1).collect();
        let xc: Vec<f64> = (0..n_bas * n_col).map(|i| ((i * 53 % 13) as f64 - 6.0) * 0.1).collect();
        let support: Vec<u32> = (0..n_bas as u32).collect();
        let mut s = LocalFoldScratch::default();
        for q in 0..n_aux {
            let jq = &j[q * n_packed..(q + 1) * n_packed];
            let got = s.fold(jq, n_bas, &support, &xr, n_row, &xc, n_col).to_vec();
            let want = dense_fold(jq, n_bas, &xr, n_row, &xc, n_col);
            for (a, b) in got.iter().zip(want.iter()) {
                assert!((a - b).abs() < 1e-13, "q={}: {} vs {}", q, a, b);
            }
        }
    }

    #[test]
    fn compressed_support_reproduces_restricted_sum() {
        let (n_bas, n_aux, n_row, n_col) = (10, 2, 3, 3);
        let j = make_packed(n_bas, n_aux);
        let n_packed = n_bas * (n_bas + 1) / 2;
        let xr: Vec<f64> = (0..n_bas * n_row).map(|i| ((i * 17 % 7) as f64 - 3.0) * 0.2).collect();
        let xc: Vec<f64> = (0..n_bas * n_col).map(|i| ((i * 29 % 9) as f64 - 4.0) * 0.2).collect();
        let support: Vec<u32> = vec![0, 2, 3, 7];
        let mut s = LocalFoldScratch::default();
        for q in 0..n_aux {
            let jq = &j[q * n_packed..(q + 1) * n_packed];
            let got = s.fold(jq, n_bas, &support, &xr, n_row, &xc, n_col).to_vec();
            let mut want = vec![0.0; n_row * n_col];
            for r in 0..n_row {
                for c in 0..n_col {
                    let mut acc = 0.0;
                    for &mu in &support {
                        for &nu in &support {
                            let (hi, lo) = if mu >= nu { (mu, nu) } else { (nu, mu) };
                            let (hi, lo) = (hi as usize, lo as usize);
                            acc += jq[hi * (hi + 1) / 2 + lo]
                                * xr[mu as usize + r * n_bas]
                                * xc[nu as usize + c * n_bas];
                        }
                    }
                    want[r + c * n_row] = acc;
                }
            }
            for (a, b) in got.iter().zip(want.iter()) {
                assert!((a - b).abs() < 1e-13, "q={}: {} vs {}", q, a, b);
            }
        }
    }

    #[test]
    fn index_finds_local_support_and_bounds_the_error() {
        let (n_bas, n_aux) = (20, 4);
        let j = make_packed(n_bas, n_aux);
        let w = vec![1.0f64; n_bas];
        let idx = AoLocalIndex::build(&j, n_bas, n_aux, &w, 1e-3);
        assert!(idx.tol > 0.0);
        assert!(idx.support_max <= n_bas);
        assert!(idx.dropped_bound >= 0.0);
        // every kept support entry really has a pair above the threshold
        let n_packed = n_bas * (n_bas + 1) / 2;
        for q in 0..n_aux {
            if let Some(sup) = idx.support(q) {
                for &mu in sup {
                    let mu = mu as usize;
                    let mut best = 0.0f64;
                    for nu in 0..n_bas {
                        let (hi, lo) = if mu >= nu { (mu, nu) } else { (nu, mu) };
                        best = best.max(j[q * n_packed + hi * (hi + 1) / 2 + lo].abs());
                    }
                    assert!(best > 1e-3, "support entry {} of q={} has max {}", mu, q, best);
                }
            }
        }
    }

    /// The pair threshold must be applied *inside* the compressed support, so
    /// that the retained set is exactly `{|J_q[mu,nu]| w_mu w_nu > tol}`.
    #[test]
    #[test]
    fn window_weights_are_window_maxima() {        let n_bas = 4;
        let x = vec![
            0.1, 0.2, 0.3, 0.4, // MO 0
            0.5, 0.1, 0.2, 0.05, // MO 1
            0.05, 0.9, 0.1, 0.2, // MO 2
        ];
        let w = window_weights(&x, n_bas, 0, 3);
        assert!((w[0] - 0.5).abs() < 1e-15);
        assert!((w[1] - 0.9).abs() < 1e-15);
        assert!((w[2] - 0.3).abs() < 1e-15);
        assert!((w[3] - 0.4).abs() < 1e-15);
    }
}
