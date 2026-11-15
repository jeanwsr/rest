//! Integral-level Schwartz screening for full-tensor three-centre producers.
//!
//! Generalizes the per-`(shell pair, aux shell)` evaluation pattern of
//! [`RIJSchwartzEngine`](super::pure_schwartz_rij::RIJSchwartzEngine) to the
//! producers that store the whole three-centre tensor: the output of
//! [`ScreenedC3Engine::evaluate_s2ij`] / [`ScreenedC3Engine::evaluate_s1_ip1`]
//! has **exactly the shape and memory layout** `CInt::integrate_cross` would
//! produce for `mols = [mol, mol, aux]` and the same `shls_slice`, with the
//! blocks that fail the screen left at zero. Downstream contractions are
//! therefore untouched; the zeros are the truncation.
//!
//! Screening uses the same exact bounds as the RI-J Schwartz engine, in log
//! space (see the `pure_schwartz_rij` module docs):
//!
//! - pair bound `q_cond(i,j) = ½ ln max_{μ∈i, ν∈j} |(μν|μν)|` (libcint
//!   `int2e` diagonals, contraction coefficients included);
//! - aux-shell bound `q_aux(psh) = ½ ln max_{P∈psh} (P|P)` (libcint `int2c2e`
//!   diagonals);
//! - a `(pair, aux shell)` triple is kept when `q_cond + q_aux ≥ ln(threshold)`,
//!   which bounds every raw `(μν|P)` entry of the triple by `threshold`
//!   (Schwarz inequality).
//!
//! For the derivative integrals (`int3c2e_ip1`, `int3c2e_ip2`) the entries are
//! amplified by the GTO derivative (≈ `sqrt(2α)` of the tightest primitive,
//! plus angular factors); the threshold is tightened by [`DERIV_AMP`] so a
//! dropped derivative entry stays bounded by `threshold`. For the plain
//! `int3c2e` tensor that feeds the metric transformation, the transformation
//! can mix the dropped entries; the threshold is tightened by [`RIMATR_AMP`].
//! Both constants are validated empirically (see the gradient/energy
//! deviations in the PR notes).
//!
//! Consumers:
//! - the RI tensor build (`generate_rimatr_bare_on_cint_prescreened`,
//!   `int3c2e` / `s2ij`);
//! - the analytical-gradient derivative batches (`grad::rhf`,
//!   `int3c2e_ip1` / `s1` and `int3c2e_ip2` / `s2ij`).

use core::ffi::c_int;

use super::prelude_dev::*;
use super::pure_schwartz_rij::{PairList, aux_q_bounds};

/// Derivative amplification margin: a dropped `(∂μν|P)` entry of a screened
/// triple is bounded by `threshold × DERIV_AMP` in value terms, so the screen
/// keeps a triple when its value bound reaches `threshold / DERIV_AMP`.
/// `1e3` covers `sqrt(2α)`-type prefactors of the GTO first derivative for
/// primitive exponents up to ≈ 5·10⁵ (6-31G core shells reach ≈ 3·10³).
pub const DERIV_AMP: f64 = 1.0e3;

/// Compact row addressing of a pre-screened AO-pair space for the three-centre
/// producers.
///
/// `full2row[p]` is the compact row of the full packed pair `p`, or
/// [`CompactRowMap::NOT_KEPT`] when the pre-screen dropped that pair. A
/// producer that takes one ([`ScreenedC3Engine::evaluate_s2ij`] with
/// `rows = Some(..)`) skips the shell pairs without a single kept row and
/// scatters the kept ones directly into the compact row space, so the full
/// `nao(nao+1)/2 × naux` tensor is never materialized: the output allocation
/// and every downstream stage (metric solve, row-max sweep) only ever see
/// `len()` rows.
///
/// The compact layout matches what [`super::compact_rimatr_rows`] produces on
/// the full tensor — one row per kept pair, in the ascending order of `keep`
/// — so the values of the kept rows are identical to the evaluate-then-compact
/// path, and a caller that wants the full-space tensor back can zero-pad.
#[derive(Clone, Debug)]
pub struct CompactRowMap {
    /// full packed pair index -> compact row, or [`CompactRowMap::NOT_KEPT`]
    full2row: Vec<u32>,
    /// Number of kept (stored) rows.
    nkept: usize,
}

impl CompactRowMap {
    /// Sentinel of [`CompactRowMap::full2row`] for a pair that is not kept.
    pub const NOT_KEPT: u32 = u32::MAX;

    /// Build from the ascending full packed pair indices of the pre-screen.
    pub fn from_keep(keep: &[usize], npair: usize) -> Self {
        let mut full2row = vec![Self::NOT_KEPT; npair];
        for (slot, &p) in keep.iter().enumerate() {
            debug_assert!(p < npair, "kept pair {} out of range {}", p, npair);
            full2row[p] = slot as u32;
        }
        Self { full2row, nkept: keep.len() }
    }

    /// Number of kept (stored) rows.
    pub fn len(&self) -> usize {
        self.nkept
    }

    /// Whether the map keeps nothing.
    pub fn is_empty(&self) -> bool {
        self.nkept == 0
    }
}

/// Metric-transformation margin for the plain `int3c2e` tensor: dropped raw
/// entries are bounded by `threshold / RIMATR_AMP`, and the column mixing of
/// the 2c-2e metric solve amplifies them back below `threshold` for the
/// well-conditioned universal fitting bases.
pub const RIMATR_AMP: f64 = 1.0e2;

/// Per-geometry Schwartz screening engine for full-tensor three-centre
/// evaluation on one (mol, aux) pair. Cheap to build (shell-pair `int2e`
/// diagonals + aux `int2c2e` diagonals, parallel) and shared by every
/// `evaluate_*` call of the geometry.
pub struct ScreenedC3Engine {
    /// mol+aux concatenation; 3c shell triples index it as
    /// `[ish, jsh, nbas_mol + psh]` (the same construction
    /// `CInt::integrate_cross` performs internally).
    merged: CInt,
    /// Static shell-pair list with exact Schwarz bounds.
    pub pairs: PairList,
    /// Aux-shell Schwarz bounds (½ ln convention), indexed by aux shell.
    pub q_aux: Vec<f64>,
    /// aux-basis ao_loc (len nbas_aux+1).
    aux_ao_loc: Vec<usize>,
    nbas_mol: usize,
    nao: usize,
    /// libcint optimizers of the merged basis, built once per integrator and
    /// reused across every batch of the geometry (the unscreened
    /// `integrate_cross` driver rebuilds them on each call).
    opt_3c2e: std::sync::OnceLock<CIntOptimizer>,
    opt_ip1: std::sync::OnceLock<CIntOptimizer>,
    opt_ip2: std::sync::OnceLock<CIntOptimizer>,
}

impl ScreenedC3Engine {
    /// Build the engine: shell-pair Schwarz bounds, aux-shell bounds and the
    /// mol+aux concatenation, all independent of the threshold. The neglect
    /// threshold itself is passed per `evaluate_*` call in log form
    /// (`(threshold / DERIV_AMP).ln()` for derivatives,
    /// `(threshold / RIMATR_AMP).ln()` for the metric-transformed rimatr).
    pub fn build(mol: &CInt, aux: &CInt, overlap_tol2: f64) -> Self {
        let pairs = PairList::build(mol, overlap_tol2);
        let q_aux = aux_q_bounds(aux);
        let mol_ao_loc = mol.ao_loc();
        let aux_ao_loc = aux.ao_loc();
        let merged = mol + aux;

        Self {
            merged,
            pairs,
            q_aux,
            nao: *mol_ao_loc.last().unwrap(),
            nbas_mol: mol.nbas(),
            aux_ao_loc,
            opt_3c2e: std::sync::OnceLock::new(),
            opt_ip1: std::sync::OnceLock::new(),
            opt_ip2: std::sync::OnceLock::new(),
        }
    }

    /// Cached optimizer of one integrator on the merged basis.
    fn optimizer_for(&self, intor: &str) -> &CIntOptimizer {
        let lock = match intor {
            "int3c2e" => &self.opt_3c2e,
            "int3c2e_ip1" => &self.opt_ip1,
            "int3c2e_ip2" => &self.opt_ip2,
            other => panic!("ScreenedC3Engine: unsupported integrator '{}'", other),
        };
        lock.get_or_init(|| self.merged.get_optimizer(&*CInt::get_integrator(intor)))
    }

    /// Output (data, shape) of the `s2ij` three-centre tensors
    /// (`int3c2e` → `[npair, nk]`, `int3c2e_ip2` → `[npair, nk, 3]`), matching
    /// `CInt::integrate_cross(intor, [mol, mol, aux], "s2ij",
    /// [[0,nbas],[0,nbas],[aux_range]])` element for element on the kept
    /// triples and zero on the screened ones.
    ///
    /// With `rows = Some(map)` the output lives on the compact row space of the
    /// pre-screen instead (`[nkept, nk]` / `[nkept, nk, 3]`): shell pairs
    /// without a single kept AO pair are not evaluated at all, and the kept
    /// ones scatter element-wise to their compact offsets. The kept rows carry
    /// the same values as the full tensor, in the order of `keep`.
    pub fn evaluate_s2ij(
        &self,
        intor: &str,
        aux_range: [usize; 2],
        log_thresh: f64,
        rows: Option<&CompactRowMap>,
    ) -> (Vec<f64>, Vec<usize>) {
        let integrator = CInt::get_integrator(intor);
        let opt = self.optimizer_for(intor);
        let cache_size = self.merged.max_cache_size(&*integrator, &[]);
        let n_comp = integrator.n_comp();

        let nk = self.aux_ao_loc[aux_range[1]] - self.aux_ao_loc[aux_range[0]];
        let npair = self.nao * (self.nao + 1) / 2;
        if let Some(map) = rows {
            debug_assert_eq!(map.full2row.len(), npair, "row map does not match the packed pair space");
        }
        // row count of the output: the compact space when a map is given
        let nrow = rows.map_or(npair, |m| m.len());

        // task list: (pair task, aux shell) with bound >= threshold, minus the
        // shell pairs the pre-screen dropped entirely
        let tasks: Vec<(usize, usize)> = self.kept_tasks(aux_range, log_thresh, rows);

        let mut shape = vec![nrow, nk];
        if n_comp > 1 {
            shape.push(n_comp);
        }
        let mut out = vec![0.0_f64; nrow * nk * n_comp];

        let ao_loc = &self.pairs.ao_loc;
        let max_dm = ao_loc.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(1);
        let max_da = self.aux_ao_loc.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(1);
        let buf_size = max_dm * max_dm * max_da * n_comp;

        // Tasks write disjoint (pair rows × aux shell) blocks of `out`; the
        // shared mutable output goes through an atomic raw pointer, the same
        // pattern as the RI-J Schwartz engine's phases.
        let out_len = out.len();
        let out_ptr = std::sync::atomic::AtomicPtr::new(out.as_mut_ptr());

        // Reorder by aux shell before the parallel dispatch. `kept_tasks`
        // enumerates pair-major, so the heavy pairs (large q_cond, kept on
        // nearly every aux shell) contribute long runs of expensive blocks
        // that a static split lumps onto one worker — measured on ts2 as a
        // ~10 s single-thread tail while the other workers idle on the rayon
        // latch (xc_valley_report.md, appendix 4). Interleaving by aux shell
        // spreads the heavy pairs across workers; the block writes are
        // disjoint, so the result is bit-for-bit unchanged.
        let mut tasks = tasks;
        tasks.sort_unstable_by_key(|&(_, psh)| psh);

        tasks.into_par_iter().with_min_len(4).for_each_init(
            || (vec![0.0_f64; buf_size], vec![0.0_f64; cache_size]),
            |(buf, cache), (task, psh)| {
                let pair = self.pairs.pairs[task];
                // fill convention: smaller shell is the bra (μ ≤ ν of the packed pair)
                let (shl_s, shl_l) = (pair.jsh, pair.ish);
                let di = ao_loc[shl_s + 1] - ao_loc[shl_s];
                let dj = ao_loc[shl_l + 1] - ao_loc[shl_l];
                let dk = self.aux_ao_loc[psh + 1] - self.aux_ao_loc[psh];
                let loc_s = ao_loc[shl_s];
                let loc_l = ao_loc[shl_l];
                let k0 = self.aux_ao_loc[psh] - self.aux_ao_loc[aux_range[0]];

                let shls = [
                    shl_s as c_int,
                    shl_l as c_int,
                    (self.nbas_mol + psh) as c_int,
                ];
                // SAFETY: buf holds di·dj·dk·n_comp (≤ buf_size) and cache the
                // optimizer's own max_cache_size; integral_block fills exactly
                // the (shl_s shl_l | psh) block.
                unsafe {
                    self.merged
                        .integral_block(&*integrator, buf, &shls, &[], Some(&opt), cache);
                }

                // SAFETY: every (pair, aux shell) task owns a disjoint slab of
                // `out` (packed pair rows × this aux shell's columns), so the
                // parallel writes never alias.
                let out = unsafe {
                    std::slice::from_raw_parts_mut(
                        out_ptr.load(std::sync::atomic::Ordering::Relaxed),
                        out_len,
                    )
                };

                // scatter, replicating rest_libcint's copy_f_4d_s2ij: packed
                // index p(μ,ν) = ν(ν+1)/2 + μ (μ ≤ ν), then κ, then component
                let offdiag = shl_s != shl_l;
                for c in 0..n_comp {
                    for k in 0..dk {
                        // column offset of this (aux ao, component) within `out`
                        let col = nrow * (k0 + k + nk * c);
                        for j in 0..dj {
                            let nu = loc_l + j;
                            let bbase = di * (j + dj * (k + dk * c));
                            let nrun = if offdiag { di } else { j + 1 };
                            match rows {
                                // full row space: the run μ ∈ [loc_s, loc_s+nrun)
                                // is contiguous both in `out` and in the packed
                                // pair index, so one copy suffices
                                None => {
                                    let base = loc_s + nu * (nu + 1) / 2 + col;
                                    out[base..base + nrun]
                                        .copy_from_slice(&buf[bbase..bbase + nrun]);
                                }
                                // compact row space: each element goes through
                                // the map (a kept run is contiguous in the map's
                                // ascending order, but runs interleave kept and
                                // dropped pairs in general)
                                Some(map) => {
                                    let mut p = loc_s + nu * (nu + 1) / 2;
                                    let mut bb = bbase;
                                    for _ in 0..nrun {
                                        let row = map.full2row[p];
                                        if row != CompactRowMap::NOT_KEPT {
                                            out[row as usize + col] = buf[bb];
                                        }
                                        p += 1;
                                        bb += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            },
        );

        (out, shape)
    }

    /// Whether any AO pair the (task's) shell pair scatters to is kept by the
    /// compact row map. Runs over the same (μ, ν) runs the scatter fills:
    /// `ν` over the larger shell, `μ ≤ ν` truncated inside the smaller shell,
    /// `μ` over the full smaller shell across shells.
    fn pair_has_kept_row(&self, task: usize, map: &CompactRowMap) -> bool {
        let pair = self.pairs.pairs[task];
        let (shl_s, shl_l) = (pair.jsh, pair.ish);
        let ao_loc = &self.pairs.ao_loc;
        let (i0, i1) = (ao_loc[shl_s], ao_loc[shl_s + 1]);
        let (j0, j1) = (ao_loc[shl_l], ao_loc[shl_l + 1]);
        let offdiag = shl_s != shl_l;
        for (j, nu) in (j0..j1).enumerate() {
            let nrun = if offdiag { i1 - i0 } else { j + 1 };
            let p0 = nu * (nu + 1) / 2 + i0;
            if map.full2row[p0..p0 + nrun]
                .iter()
                .any(|&r| r != CompactRowMap::NOT_KEPT)
            {
                return true;
            }
        }
        false
    }

    /// (pair task, aux shell) triples whose bound `q_cond + q_aux` reaches
    /// `log_thresh`, minus the shell pairs the compact row map drops entirely.
    /// The pair list is sorted by descending `q_cond`, so a partition would do
    /// for one aux shell; over a *range* of aux shells the flat filter is the
    /// exact union.
    fn kept_tasks(
        &self,
        aux_range: [usize; 2],
        log_thresh: f64,
        rows: Option<&CompactRowMap>,
    ) -> Vec<(usize, usize)> {
        self.pairs
            .pairs
            .iter()
            .enumerate()
            .filter(|&(task, _)| rows.map_or(true, |m| self.pair_has_kept_row(task, m)))
            .flat_map(|(task, pair)| {
                let cutoff = log_thresh - pair.q_cond;
                (aux_range[0]..aux_range[1])
                    .filter(move |&psh| self.q_aux[psh] >= cutoff)
                    .map(move |psh| (task, psh))
            })
            .collect()
    }

    /// Kept fraction of the (pair, aux shell) triples of a range — for the
    /// screening report of callers. Reports the Schwartz level only (no row
    /// map), so the fraction is comparable across configurations.
    pub fn kept_fraction(&self, aux_range: [usize; 2], log_thresh: f64) -> f64 {
        let n_pairs = self.pairs.pairs.len();
        let total = n_pairs * (aux_range[1] - aux_range[0]);
        if total == 0 {
            return 1.0;
        }
        self.kept_tasks(aux_range, log_thresh, None).len() as f64 / total as f64
    }

    /// Output (data, shape) of the `s1` derivative tensor `int3c2e_ip1` →
    /// `[nao, nao, nk, 3]`, matching
    /// `CInt::integrate_cross("int3c2e_ip1", [mol, mol, aux], "s1",
    /// [[0,nbas],[0,nbas],[aux_range]])`. Both (i,j) and (j,i) shell-pair
    /// orders are evaluated (the derivative acts on the bra center, so the
    /// mirrored block is *not* the transpose), exactly like the unscreened
    /// driver does.
    pub fn evaluate_s1_ip1(
        &self,
        aux_range: [usize; 2],
        log_thresh: f64,
    ) -> (Vec<f64>, Vec<usize>) {
        let integrator = CInt::get_integrator("int3c2e_ip1");
        let opt = self.optimizer_for("int3c2e_ip1");
        let cache_size = self.merged.max_cache_size(&*integrator, &[]);
        let n_comp = integrator.n_comp();
        debug_assert_eq!(n_comp, 3);

        let nk = self.aux_ao_loc[aux_range[1]] - self.aux_ao_loc[aux_range[0]];
        let nao = self.nao;

        let tasks: Vec<(usize, usize, bool)> = self
            .kept_tasks(aux_range, log_thresh, None)
            .into_iter()
            .flat_map(|(task, psh)| [(task, psh, true), (task, psh, false)])
            .collect();

        let mut out = vec![0.0_f64; nao * nao * nk * n_comp];
        let shape = vec![nao, nao, nk, n_comp];

        let ao_loc = &self.pairs.ao_loc;
        let max_dm = ao_loc.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(1);
        let max_da = self.aux_ao_loc.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(1);
        let buf_size = max_dm * max_dm * max_da * n_comp;

        let out_len = out.len();
        let out_ptr = std::sync::atomic::AtomicPtr::new(out.as_mut_ptr());

        // same interleaving as `evaluate_s2ij`: spread heavy pairs across workers
        let mut tasks = tasks;
        tasks.sort_unstable_by_key(|&(_, psh, _)| psh);

        tasks.into_par_iter().with_min_len(4).for_each_init(
            || (vec![0.0_f64; buf_size], vec![0.0_f64; cache_size]),
            |(buf, cache), (task, psh, forward)| {
                let pair = self.pairs.pairs[task];
                // forward: bra = smaller shell (μ of the s1 tensor rows);
                // mirrored: bra = larger shell (the driver evaluates both)
                let (shl_mu, shl_nu) = if forward { (pair.jsh, pair.ish) } else { (pair.ish, pair.jsh) };
                let dmu = ao_loc[shl_mu + 1] - ao_loc[shl_mu];
                let dnu = ao_loc[shl_nu + 1] - ao_loc[shl_nu];
                let dk = self.aux_ao_loc[psh + 1] - self.aux_ao_loc[psh];
                let loc_mu = ao_loc[shl_mu];
                let loc_nu = ao_loc[shl_nu];
                let k0 = self.aux_ao_loc[psh] - self.aux_ao_loc[aux_range[0]];

                let shls = [
                    shl_mu as c_int,
                    shl_nu as c_int,
                    (self.nbas_mol + psh) as c_int,
                ];
                // SAFETY: as in evaluate_s2ij.
                unsafe {
                    self.merged
                        .integral_block(&*integrator, buf, &shls, &[], Some(&opt), cache);
                }

                // SAFETY: disjoint (μ shell rows, ν shell columns, aux shell)
                // slab per task, as in evaluate_s2ij.
                let out = unsafe {
                    std::slice::from_raw_parts_mut(
                        out_ptr.load(std::sync::atomic::Ordering::Relaxed),
                        out_len,
                    )
                };

                // scatter, replicating copy_f_4d_s1: element (μ, ν, κ, c) at
                // μ + nao·(ν + nao·(κ + nk·c)) in column-major
                for c in 0..n_comp {
                    for k in 0..dk {
                        for j in 0..dnu {
                            let base =
                                loc_mu + nao * (loc_nu + j + nao * (k0 + k + nk * c));
                            let bbase = dmu * (j + dnu * (k + dk * c));
                            out[base..base + dmu].copy_from_slice(&buf[bbase..bbase + dmu]);
                        }
                    }
                }
            },
        );

        (out, shape)
    }
}
