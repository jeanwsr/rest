//! Pure implementation of the Schwartz-screened RI-J algorithm (`ri-schwartz`).
//!
//! Integral-direct "standard RI" (Scheme 3 of Neese, J. Comput. Chem. 24, 1740 (2003)) with
//! per-shell-pair Schwarz screening: every `(μν|P)` block is produced by libcint
//! (`CInt::integral_block` on a mol+aux basis concatenation) and contracted the moment it exists,
//! so no 3c-2e tensor is ever stored. Per SCF iteration and density matrix set `k`:
//!
//! ```text
//! g_k[P]   = Σ_pairs f · Σ_{μν∈pair} dm_k[μ,ν] · (μν|P)  (phase 1, parallel over aux shells)
//! d_k      = V⁻¹ g_k                                      (2c-2e metric solve)
//! J_k[μ,ν] = Σ_pairs Σ_P d_k[P] · (μν|P)                  (phase 2, parallel over pairs)
//! ```
//!
//! with `f = 2 − δ_ij` for off-diagonal shell pairs (the density is symmetric, so the mirrored
//! triangle enters through this factor).
//!
//! Screening works in log space with exact Schwarz bounds taken from libcint itself:
//!
//! - pair bound `q_cond(i,j) = ½ ln max_{μ∈i, ν∈j} |(μν|μν)|`, from the diagonal of `int2e` blocks
//!   `(i j | i j)`;
//! - aux-shell bound `q_aux(psh) = ½ ln max_{P∈psh} (P|P)`, from the diagonal of `int2c2e` blocks;
//! - dynamic bounds of the strongest matrix of the batch: `dm_cond = ln max_k max_{μν∈pair}
//!   |dm_k[μ,ν]|` in phase 1, and `ln max_k max_{P∈psh} |d_k[P]|` in phase 2.
//!
//! Phase 1 skips a (pair, aux shell) combination when `q_cond + dm_cond + q_aux < ln(threshold)`,
//! phase 2 when `q_cond + q_aux + ln|d| < ln(threshold)`. Both loops visit their inner axis in
//! descending-bound order, so a `partition_point` prefix is exactly the set that passes; screening
//! on the strongest matrix of the batch makes each prefix the union of the per-matrix prefixes,
//! so batched outputs match individual builds to round-off, and outputs are deterministic for a
//! fixed argument set.
//!
//! # Block layout
//!
//! `integral_block` fills the raw libcint buffer Fortran-ordered `[di, dj, np]`, i.e., linear
//! index `μ + di·ν + di·dj·P` — μ fastest, the aux function P slowest. [`DmPack`] packs each
//! pair's density blocks in exactly that order, so every per-P slab of the buffer dots against a
//! contiguous packed block; phase 2 accumulates the buffer P-slab-wise into a packed J block per
//! matrix and scatters it to both triangles of J at the end.
//!
//! # Shell Indices and Density packing
//!
//! Relation of shell indices and basis functions (if spherical), for example:
//!
//! ```log
//! angular  s  p        s  d              p
//! μ (u  ) [0  1  2  3  4  5  6  7  8  9 10 11 12   ], nao  = 13
//! i (ish) [0  1        2  3              4         ], nbas = 5
//! ao_loc  [0  1        4  5             10       13], ao_loc.len() = nbas+1 = 6
//! di      [1  3        1  5              3         ]
//! ```

use core::ffi::c_int;
use std::sync::atomic::{AtomicPtr, Ordering};

use super::decompose::{J2CDecompOption, J2CDecompose};
use super::prelude_dev::*;
use super::pure_decompose::get_j2c_decomp;

/// Sentinel for negligible blocks (≈ ln 1e-304), used as the bound of shell pairs whose Schwarz
/// diagonal vanishes.
pub const NEGLIGIBLE: f64 = -700.0;

/// Shell pair indices and Schwarz screen value of the original basis.
#[derive(Clone, Copy, Debug)]
pub struct ShellPair {
    /// Shell index i, ish ∈ [0, nbas).
    pub ish: usize,
    /// Shell index j, usually ish >= jsh, jsh ∈ [0, nbas).
    pub jsh: usize,
    /// ½ ln max_{μ∈i, ν∈j} |(μν|μν)|, i.e., ln of the Schwarz ERI integral bound.
    pub q_cond: f64,
}

/// Triangle (ish >= jsh) shell-pair list of a basis, sorted by descending `q_cond` so the
/// per-task early-exit prefix cuts of the engine are tight.
pub struct PairList {
    /// Shell pairs (ish, jsh) sorted by descending q_cond (max len = nbas*(nbas+1)/2).
    pub pairs: Vec<ShellPair>,
    /// mol-basis ao_loc (len nbas+1).
    pub ao_loc: Vec<usize>,
    /// Cumulative sum of di·dj over pairs (len pairs+1).
    pub pair_loc: Vec<usize>,
}

impl PairList {
    /// Number of shells of the original basis, deduced from `ao_loc`.
    pub fn nbas(&self) -> usize {
        self.ao_loc.len() - 1
    }

    /// Σ di·dj over pairs, deduced from the last prefix sum of `pair_loc`.
    pub fn total_dij(&self) -> usize {
        *self.pair_loc.last().unwrap()
    }

    /// Build the shell-pair list of the mol basis with exact Schwarz bounds.
    ///
    /// A cheap static pre-mask (overlap of each shell pair's most-diffuse primitives) removes
    /// distant pairs before the exact sweep; the survivors get `q_cond` from the diagonal of
    /// libcint `int2e` blocks `(i j | i j)`, which is exact — contraction coefficients included
    /// by the integrals themselves.
    pub fn build(mol: &CInt, overlap_tol2: f64) -> Self {
        let nbas = mol.nbas();
        let ao_loc = mol.ao_loc();

        // 1. candidate shell pairs (ish >= jsh): drop pairs whose most-diffuse primitives are too
        //    far apart to overlap
        let diffuse: Vec<(f64, [f64; 3])> = (0..nbas).map(|sh| diffuse_shell(mol, sh)).collect();
        let mut cand: Vec<(usize, usize)> = Vec::with_capacity(nbas * (nbas + 1) / 2);
        for ish in 0..nbas {
            for jsh in 0..=ish {
                let (a, ra) = diffuse[ish];
                let (b, rb) = diffuse[jsh];
                if s_overlap(a, ra, b, rb) > overlap_tol2 {
                    cand.push((ish, jsh));
                }
            }
        }

        // 2. exact Schwarz diagonal (μν|μν) from libcint int2e blocks
        let integrator = CInt::get_integrator("int2e");
        let opt = mol.get_optimizer(&*integrator);
        let cache_size = mol.max_cache_size(&*integrator, &[]);
        let max_d = max_cgto(&ao_loc);
        let buf_size = max_d * max_d * max_d * max_d;

        let q_cond: Vec<f64> = cand
            .par_iter()
            .map_init(
                || (vec![0.0f64; buf_size], vec![0.0f64; cache_size]),
                |(buf, cache), &(ish, jsh)| pair_q_cond(mol, &*integrator, &opt, &ao_loc, ish, jsh, buf, cache),
            )
            .collect();

        // 3. sort descending by q_cond (contribution order); q_cond is never NaN (the 1e-300 guard
        //    in pair_q_cond), so the total-order comparison never panics
        let mut order: Vec<usize> = (0..cand.len()).collect();
        order.sort_unstable_by(|&a, &b| q_cond[b].total_cmp(&q_cond[a]));
        let pairs: Vec<ShellPair> =
            order.iter().map(|&c| ShellPair { ish: cand[c].0, jsh: cand[c].1, q_cond: q_cond[c] }).collect();
        let mut pair_loc = Vec::with_capacity(pairs.len() + 1);
        pair_loc.push(0);
        let mut total = 0usize;
        for pair in &pairs {
            let di = ao_loc[pair.ish + 1] - ao_loc[pair.ish];
            let dj = ao_loc[pair.jsh + 1] - ao_loc[pair.jsh];
            total += di * dj;
            pair_loc.push(total);
        }
        PairList { pairs, ao_loc, pair_loc }
    }
}

/// Per-iteration density packing of phase 1: the densities packed by shell pairs, plus the
/// dynamic (dm-dependent) screening bounds and order.
///
/// `pblock` holds each pair's di·dj density block in the μ-fastest order of the "# Block layout"
/// section — the same layout as the per-P slabs of libcint's Fortran-ordered `[di, dj, np]` 3c-2e
/// buffer, so the per-P contraction is one contiguous dot — one full set per input matrix,
/// matrix-major: matrix k's block for pair `task` sits at `k·total_dij + pair_loc[task]`.
///
/// `qd_order` lists `(q_cond + dm_cond, pair task)` descending by the combined dynamic bound
/// `q_cond + ln max_k max_{μν∈pair} |dm_k[μ,ν]|` (1e-300-guarded) of the strongest matrix of the
/// batch, which makes the engine's descending-order `partition_point` prefix cut exact for the
/// whole batch at once. Pairs whose dynamic bound cannot reach even the strongest aux shell sort
/// to the tail; keeping them in the order preserves stable indices.
///
/// Everything in here depends on the densities, so it is rebuilt every iteration; only the static
/// [`PairList`] survives across iterations.
pub struct DmPack {
    /// Density blocks packed μ-fastest, one full set per input matrix (matrix-major), laid out by
    /// [`PairList::pair_loc`].
    pub pblock: Vec<f64>,
    /// `(q_cond + dm_cond, pair task)`, descending by the first element.
    pub qd_order: Vec<(f64, usize)>,
}

impl DmPack {
    /// Pack the f-contiguous (nao, nao, nset) density stack `dms_raw` (element (u, v, k) at
    /// `u + nao·v + k·nao²`) against the shell pairs of `pairs`.
    pub fn build(pairs: &PairList, dms_raw: &[f64], nao: usize, nset: usize) -> Self {
        let total_dij = pairs.total_dij();
        let ao_loc = &pairs.ao_loc;
        let npairs = pairs.pairs.len();
        let nao2 = nao * nao;
        let mut pblock = vec![0.0f64; nset * total_dij];
        let mut dm_cond = vec![0.0f64; npairs];
        {
            let pb_ptr = AtomicPtr::new(pblock.as_mut_ptr());
            let cond_ptr = AtomicPtr::new(dm_cond.as_mut_ptr());
            (0..npairs).into_par_iter().for_each(|task| {
                let pb_ptr = pb_ptr.load(Ordering::Relaxed);
                let cond_ptr = cond_ptr.load(Ordering::Relaxed);
                let pair = pairs.pairs[task];
                let (i0, di) = (ao_loc[pair.ish], ao_loc[pair.ish + 1] - ao_loc[pair.ish]);
                let (j0, dj) = (ao_loc[pair.jsh], ao_loc[pair.jsh + 1] - ao_loc[pair.jsh]);
                let off = pairs.pair_loc[task];
                let mut vmax = 0.0f64;
                for k in 0..nset {
                    let mut m = 0;
                    for v in 0..dj {
                        for u in 0..di {
                            // packed order: μ fastest, m = μ + di·ν
                            let val = dms_raw[k * nao2 + (i0 + u) + nao * (j0 + v)];
                            // SAFETY: task writes exactly [k·total_dij + off, + di·dj) of pblock
                            // — disjoint across tasks and matrices by pair_loc — and the single
                            // dm_cond slot task itself owns.
                            unsafe {
                                *pb_ptr.add(k * total_dij + off + m) = val;
                            }
                            vmax = vmax.max(val.abs());
                            m += 1;
                        }
                    }
                }
                unsafe {
                    *cond_ptr.add(task) = (vmax + 1e-300).ln();
                }
            });
        }
        let mut qd_order: Vec<(f64, usize)> =
            pairs.pairs.iter().zip(&dm_cond).enumerate().map(|(task, (pair, &dc))| (pair.q_cond + dc, task)).collect();
        qd_order.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        DmPack { pblock, qd_order }
    }
}

/// Per-geometry data of the Schwartz-screened RI-J algorithm: static shell-pair list, aux-shell
/// bounds, the 3c-2e machinery of the mol+aux concatenation, and the decomposed 2c-2e Coulomb
/// metric. Built once per geometry by [`RiJSchwartzEngine::build`] and reused by every
/// [`get_vj_ri_schwartz`] call of the SCF iterations.
pub struct RiJSchwartzEngine {
    /// mol+aux concatenation: 3c-2e shell triples index it as `[ish, jsh, nbas_mol + psh]` (the
    /// same construction `CInt::integrate_cross` performs internally).
    merged: CInt,
    /// Static shell-pair list with exact Schwarz bounds.
    pub pairs: PairList,
    /// Aux Schwarz bounds, one per aux shell.
    pub q_aux: Vec<f64>,
    /// Aux shell indices, descending by q_aux.
    pub aux_order: Vec<usize>,
    /// Aux-basis ao_loc (len nbas_aux+1).
    aux_ao_loc: Vec<usize>,
    /// libcint 3c-2e optimizer built from `merged`; the integrator itself is re-created per call.
    opt: CIntOptimizer,
    /// Per-worker buffer/cache sizes for `integral_block`.
    buf_size: usize,
    cache_size: usize,
    /// ln of the integral neglect threshold (Hartree).
    pub log_thresh: f64,
    pub nao: usize,
    pub naux: usize,
    /// Decomposed 2c-2e Coulomb metric of the aux basis.
    pub j2c_decomp: J2CDecompose,
    nbas_mol: usize,
}

impl RiJSchwartzEngine {
    /// Build the per-geometry data of the Schwartz-screened RI-J algorithm.
    ///
    /// - `mol`: [`CInt`]; `aux`: [`CInt`]: molecule and auxiliary basis objects.
    /// - `threshold`: integral neglect threshold (Hartree); the log is taken internally.
    /// - `overlap_tol2`: threshold of the static overlap pre-mask of the pair-list build.
    /// - `j2c_decomp_option`: policy of the 2c-2e Coulomb metric decomposition, same as the incore
    ///   RI algorithms (see [`J2CDecompOption`]).
    pub fn build(mol: CInt, aux: CInt, threshold: f64, overlap_tol2: f64, j2c_decomp_option: J2CDecompOption) -> Self {
        let nbas_mol = mol.nbas();
        let nbas_aux = aux.nbas();
        let aux_ao_loc = aux.ao_loc();

        // 1. static pair list with exact Schwarz bounds (int2e diagonals)
        let pairs = PairList::build(&mol, overlap_tol2);

        // 2. aux Schwarz bounds (int2c2e diagonals) + descending contribution order
        let q_aux = aux_q_bounds(&aux);
        let mut aux_order: Vec<usize> = (0..nbas_aux).collect();
        aux_order.sort_unstable_by(|&a, &b| q_aux[b].total_cmp(&q_aux[a]));

        // 3. mol+aux concatenation and its 3c-2e machinery
        let merged = &mol + &aux;
        let integrator = CInt::get_integrator("int3c2e");
        let opt = merged.get_optimizer(&*integrator);
        let cache_size = merged.max_cache_size(&*integrator, &[]);
        let buf_size = max_cgto(&pairs.ao_loc) * max_cgto(&pairs.ao_loc) * max_cgto(&aux_ao_loc);

        // 4. decomposed 2c-2e Coulomb metric, applied in every J evaluation
        let device = DeviceBLAS::default();
        let j2c_decomp = get_j2c_decomp(&aux, &device, j2c_decomp_option);

        RiJSchwartzEngine {
            merged,
            pairs,
            q_aux,
            aux_order,
            aux_ao_loc,
            opt,
            buf_size,
            cache_size,
            log_thresh: threshold.ln(),
            nao: mol.nao(),
            naux: aux.nao(),
            j2c_decomp,
            nbas_mol,
        }
    }
}

/// Generate Coulomb (J) matrices using the RI method with per-shell-pair Schwartz screening.
///
/// This function is low-level implementation, using RSTSR tensors as input and output. For
/// high-level interface (using rest_tensors as input and output), please refer to
/// [`generate_vj_ri_schwartz`](super::schwartz_rij::generate_vj_ri_schwartz).
///
/// The two-phase evaluation is the same RI fitting as
/// [`get_vj_ri_direct`](super::pure_direct::get_vj_ri_direct), with the 3c-2e blocks screened
/// pair-by-pair instead of evaluated in aux-shell batches.
///
/// # Parameters
///
/// - `engine`: `&`[`RiJSchwartzEngine`]
///
///   - Per-geometry data built by [`RiJSchwartzEngine::build`]; can be shared across SCF iterations
///     and threads.
///
/// - `dms`: [`TsrView<f64>`]
///
///   - Density matrices in shape (nao, nao, nset), stored in f-contiguous order.
///   - We will check `ndim == 3` and the shape against the engine. Please expand dimension if
///     necessary, especially for RHF case where `nset = 1`.
///   - This matrix is assumed to be symmetric. We will not perform symmetry check or symmetrize
///     operation.
///
/// # Returns
///
/// - [`Tsr<f64>`]
///
///   - Coulomb (J) matrices in shape (nao, nao, nset), stored in f-contiguous order.
///   - J matrices are symmetric by definition in real arithmetic.
pub fn get_vj_ri_schwartz(engine: &RiJSchwartzEngine, dms: TsrView<f64>) -> Tsr<f64> {
    let nao = engine.nao;
    let naux = engine.naux;
    assert_eq!(dms.ndim(), 3, "DM must have 3 dimensions");
    let nset = dms.shape()[2];
    assert_eq!(dms.shape(), &[nao, nao, nset], "Density matrices must have shape (nao, nao, nset)");
    assert!(dms.f_contig(), "Density matrices must be stored in f-contiguous order");
    let device = dms.device().clone();

    // element (u, v, k) of the f-contiguous stack sits at u + nao·v + k·nao² of the raw storage
    let dms_raw: &[f64] = &dms.raw()[dms.offset()..dms.offset() + nao * nao * nset];

    // -- (eq.1) -- //
    // phase 1: g_k[P] = Σ_pairs f · Σ_{μν} dm_k[μν] (μν|P), parallel over aux shells
    let g = phase1(engine, dms_raw, nset, nao);

    // -- (eq.2) -- //
    // solve the fitting equations d_k = V⁻¹ g_k with the decomposed 2c-2e metric
    let g_tsr = rt::asarray((g, [naux, nset].f(), &device));
    let d_tsr = solve_by_metric(g_tsr.view(), &engine.j2c_decomp);
    let d_raw: &[f64] = &d_tsr.raw()[..naux * nset];

    // -- (eq.3) -- //
    // phase 2: J_k[μν] = Σ_pairs Σ_P d_k[P] (μν|P), parallel over pairs
    let js = phase2(engine, d_raw, nset);

    rt::asarray((js, [nao, nao, nset].f(), &device))
}

/// First half of a Coulomb build: densities → auxiliary vectors `g`, one (naux,) vector per
/// matrix of the batch. Each (pair, aux shell) block is evaluated once for the whole batch and
/// contracted with every matrix; parallel over aux shells, each task owning a disjoint slice of
/// every g.
fn phase1(engine: &RiJSchwartzEngine, dms_raw: &[f64], nset: usize, nao: usize) -> Vec<f64> {
    let RiJSchwartzEngine {
        merged,
        pairs,
        q_aux,
        aux_order,
        aux_ao_loc,
        opt,
        buf_size,
        cache_size,
        log_thresh,
        naux,
        nbas_mol,
        ..
    } = engine;
    let (buf_size, cache_size, naux, nbas_mol, log_thresh) = (*buf_size, *cache_size, *naux, *nbas_mol, *log_thresh);
    let integrator = CInt::get_integrator("int3c2e");

    let pack = DmPack::build(pairs, dms_raw, nao, nset);

    let ao_loc = &pairs.ao_loc;
    let total_dij = pairs.total_dij();

    let mut g = vec![0.0f64; nset * naux];
    let g_ptr = AtomicPtr::new(g.as_mut_ptr());
    aux_order.par_iter().for_each_init(
        || (vec![0.0f64; buf_size], vec![0.0f64; cache_size]),
        |(buf, cache), &psh| {
            let np = aux_ao_loc[psh + 1] - aux_ao_loc[psh];
            let cutoff = log_thresh - q_aux[psh];
            let prefix = pack.qd_order.partition_point(|&(q, _)| q >= cutoff);
            if prefix == 0 {
                return;
            }
            let psh_off = (nbas_mol + psh) as c_int;
            let mut g_psh = vec![0.0f64; nset * np];
            for &(_, task) in &pack.qd_order[..prefix] {
                let pair = pairs.pairs[task];
                let di = ao_loc[pair.ish + 1] - ao_loc[pair.ish];
                let dj = ao_loc[pair.jsh + 1] - ao_loc[pair.jsh];
                let nd = di * dj;
                let shls = [pair.ish as c_int, pair.jsh as c_int, psh_off];
                // SAFETY: buf holds di·dj·np (≤ buf_size, the product of the three largest shell
                // widths) and cache the optimizer's own max_cache_size; integral_block fills
                // exactly the (ish jsh | psh) block.
                unsafe {
                    merged.integral_block(&*integrator, buf, &shls, &[], Some(opt), cache);
                }
                // one block evaluation, contracted with every matrix: each P slab of the buffer
                // dots against the matrix's packed density block
                let f = if pair.ish != pair.jsh { 2.0 } else { 1.0 };
                let off = pairs.pair_loc[task];
                for (k, gk) in g_psh.chunks_exact_mut(np).enumerate() {
                    let blk = &pack.pblock[k * total_dij + off..][..nd];
                    for (p, gp) in gk.iter_mut().enumerate() {
                        let slab = &buf[p * nd..][..nd];
                        *gp += f * slab.iter().zip(blk).map(|(&a, &b)| a * b).sum::<f64>();
                    }
                }
            }
            let p0 = aux_ao_loc[psh];
            for (k, gk) in g_psh.chunks_exact(np).enumerate() {
                // SAFETY: aux shell psh owns [p0, p0+np) of the k-th matrix slice of g —
                // disjoint across tasks and matrices.
                unsafe {
                    std::ptr::copy_nonoverlapping(gk.as_ptr(), g_ptr.load(Ordering::Relaxed).add(k * naux + p0), np);
                }
            }
        },
    );
    g
}

/// Second half of a Coulomb build: fit coefficients `d` → Coulomb matrices, one (nao, nao)
/// f-contiguous matrix per set of the batch. Parallel over shell pairs, each task owning the
/// disjoint (both-triangle) block of every J its unique pair covers.
fn phase2(engine: &RiJSchwartzEngine, d_raw: &[f64], nset: usize) -> Vec<f64> {
    let RiJSchwartzEngine {
        merged,
        pairs,
        q_aux,
        aux_ao_loc,
        opt,
        buf_size,
        cache_size,
        log_thresh,
        nao,
        naux,
        nbas_mol,
        ..
    } = engine;
    let (buf_size, cache_size, log_thresh, nao, naux, nbas_mol) =
        (*buf_size, *cache_size, *log_thresh, *nao, *naux, *nbas_mol);
    let integrator = CInt::get_integrator("int3c2e");
    let nao2 = nao * nao;

    // per-aux-shell dynamic bounds: q_aux + ln max_k max_{P∈psh} |d_k[P]|
    let nauxbas = aux_ao_loc.len() - 1;
    let qdk: Vec<f64> = (0..nauxbas)
        .into_par_iter()
        .map(|psh| {
            let (p0, p1) = (aux_ao_loc[psh], aux_ao_loc[psh + 1]);
            let mut vmax = 0.0f64;
            for k in 0..nset {
                let m = d_raw[k * naux + p0..k * naux + p1].iter().fold(0.0f64, |a, &v| a.max(v.abs()));
                vmax = vmax.max(m);
            }
            q_aux[psh] + (vmax + 1e-300).ln()
        })
        .collect();
    let mut aux_order2: Vec<usize> = (0..nauxbas).collect();
    aux_order2.sort_unstable_by(|&a, &b| qdk[b].total_cmp(&qdk[a]));
    let qdk_sorted: Vec<f64> = aux_order2.iter().map(|&psh| qdk[psh]).collect();

    // largest pair block: di·dj ≤ max_cgto²; one packed J block per matrix
    let ao_loc = &pairs.ao_loc;
    let npairs = pairs.pairs.len();
    let max_nd = max_cgto(ao_loc) * max_cgto(ao_loc);

    let mut js = vec![0.0f64; nset * nao2];
    let js_ptr = AtomicPtr::new(js.as_mut_ptr());
    (0..npairs).into_par_iter().for_each_init(
        || (vec![0.0f64; buf_size], vec![0.0f64; cache_size], vec![0.0f64; nset * max_nd]),
        |(buf, cache, jblk), task| {
            let pair = pairs.pairs[task];
            let cutoff = log_thresh - pair.q_cond;
            let prefix = qdk_sorted.partition_point(|&q| q >= cutoff);
            if prefix == 0 {
                return;
            }
            let (i0, di) = (ao_loc[pair.ish], ao_loc[pair.ish + 1] - ao_loc[pair.ish]);
            let (j0, dj) = (ao_loc[pair.jsh], ao_loc[pair.jsh + 1] - ao_loc[pair.jsh]);
            let nd = di * dj;
            for jb in jblk.chunks_exact_mut(max_nd) {
                jb[..nd].fill(0.0);
            }
            for &psh in &aux_order2[..prefix] {
                let (p0, np) = (aux_ao_loc[psh], aux_ao_loc[psh + 1] - aux_ao_loc[psh]);
                let psh_off = (nbas_mol + psh) as c_int;
                let shls = [pair.ish as c_int, pair.jsh as c_int, psh_off];
                // SAFETY: as in phase 1.
                unsafe {
                    merged.integral_block(&*integrator, buf, &shls, &[], Some(opt), cache);
                }
                // one block evaluation, accumulated into every J block
                for k in 0..nset {
                    let jb = &mut jblk[k * max_nd..][..nd];
                    for p in 0..np {
                        let dp = d_raw[k * naux + p0 + p];
                        if dp == 0.0 {
                            continue;
                        }
                        let slab = &buf[p * nd..][..nd];
                        for (v, &s) in jb.iter_mut().zip(slab) {
                            *v += dp * s;
                        }
                    }
                }
            }
            // scatter to both triangles of every J (blocks are disjoint across unique pairs);
            // for ish == jsh the block covers both triangles of itself
            let js_ptr = js_ptr.load(Ordering::Relaxed);
            for k in 0..nset {
                let blk = &jblk[k * max_nd..][..nd];
                let mut m = 0;
                for v in 0..dj {
                    for u in 0..di {
                        let val = blk[m];
                        m += 1;
                        // SAFETY: the unique pair task owns the (i0..i0+di)×(j0..j0+dj) block
                        // and its mirror of the k-th J matrix — disjoint across tasks.
                        unsafe {
                            *js_ptr.add(k * nao2 + (i0 + u) + nao * (j0 + v)) = val;
                            if pair.ish != pair.jsh {
                                *js_ptr.add(k * nao2 + (j0 + v) + nao * (i0 + u)) = val;
                            }
                        }
                    }
                }
            }
        },
    );
    js
}

/// Solve the fitting equations `d = V⁻¹ g` with the decomposed 2c-2e Coulomb metric, for the
/// (naux, nset) right-hand-side batch `g`.
fn solve_by_metric(g: TsrView<f64>, j2c_decomp: &J2CDecompose) -> Tsr<f64> {
    match j2c_decomp {
        J2CDecompose::Cd { j2c_l, uplo, .. } => match uplo {
            Upper => {
                // V = LᵀL with the upper-triangular factor L: forward then backward substitution
                let y = rt::linalg::solve_triangular((j2c_l.t(), g, Lower));
                rt::linalg::solve_triangular((j2c_l.view(), y, Upper))
            },
            Lower => {
                // V = L·Lᵀ with the lower-triangular factor L
                let y = rt::linalg::solve_triangular((j2c_l.view(), g, Lower));
                rt::linalg::solve_triangular((j2c_l.t(), y, Upper))
            },
        },
        J2CDecompose::Eig { j2c_l_inv, .. } => {
            // V⁻¹ = V^-1/2 · V^-1/2 with the symmetric -1/2 power
            let y = j2c_l_inv.view() % g;
            j2c_l_inv.view() % &y
        },
    }
}

/// ln-scale Schwarz bound of one shell pair: ½ ln max_{μ∈i, ν∈j} |(μν|μν)| from the diagonal of
/// the raw int2e buffer.
///
/// # Safety (caller-side)
///
/// `buf` must hold `(di*dj)^2` and `cache` the optimizer's requirement for this
/// molecule/integrator — both are sized by the caller before this runs.
#[allow(clippy::too_many_arguments)]
fn pair_q_cond(
    mol: &CInt,
    integrator: &dyn Integrator,
    opt: &CIntOptimizer,
    ao_loc: &[usize],
    ish: usize,
    jsh: usize,
    buf: &mut [f64],
    cache: &mut [f64],
) -> f64 {
    let di = ao_loc[ish + 1] - ao_loc[ish];
    let dj = ao_loc[jsh + 1] - ao_loc[jsh];
    let shls = [ish as c_int, jsh as c_int, ish as c_int, jsh as c_int];
    // SAFETY: invariants documented above; libcint fills exactly the (i j | i j) block of
    // (di*dj)^2 doubles.
    unsafe { mol.integral_block(integrator, buf, &shls, &[], Some(opt), cache) };
    // raw buffer is Fortran [di, dj, di, dj]; its diagonal elements (μ,ν,μ,ν) sit at
    // (μ + di·ν)·(1 + di·dj)
    let step = 1 + di * dj;
    let mut vmax = 0.0f64;
    for idx in 0..di * dj {
        let v = buf[idx * step].abs();
        if v > vmax {
            vmax = v;
        }
    }
    if vmax <= 1e-300 {
        NEGLIGIBLE
    } else {
        0.5 * vmax.ln()
    }
}

/// Aux-shell Schwarz bounds: `q_aux(psh) = ½ ln max_{P∈psh} (P|P)`, from int2c2e diagonals.
pub fn aux_q_bounds(aux: &CInt) -> Vec<f64> {
    let integrator = CInt::get_integrator("int2c2e");
    let opt = aux.get_optimizer(&*integrator);
    let cache_size = aux.max_cache_size(&*integrator, &[]);
    let ao_loc = aux.ao_loc();
    let max_d = max_cgto(&ao_loc);
    let buf_size = max_d * max_d;

    (0..aux.nbas())
        .into_par_iter()
        .map_init(
            || (vec![0.0f64; buf_size], vec![0.0f64; cache_size]),
            |(buf, cache), psh| {
                let np = ao_loc[psh + 1] - ao_loc[psh];
                let shls = [psh as c_int, psh as c_int];
                // SAFETY: buf holds np², cache the optimizer's requirement.
                unsafe {
                    aux.integral_block(&*integrator, buf, &shls, &[], Some(&opt), cache);
                }
                // Fortran [np, np]: diagonal at P·(1 + np)
                let mut vmax = 0.0f64;
                for p in 0..np {
                    let v = buf[p * (1 + np)].abs();
                    if v > vmax {
                        vmax = v;
                    }
                }
                if vmax <= 1e-300 {
                    NEGLIGIBLE
                } else {
                    0.5 * vmax.ln()
                }
            },
        )
        .collect()
}

/// Largest function count of any shell in the basis.
pub fn max_cgto(ao_loc: &[usize]) -> usize {
    ao_loc.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(1)
}

/// Most-diffuse primitive exponent of a shell and its center: the pair of most-diffuse
/// primitives has the largest spatial reach, so their overlap is the conservative static
/// representative of the whole shell pair.
fn diffuse_shell(mol: &CInt, sh: usize) -> (f64, [f64; 3]) {
    let a = mol.bas_exp(sh).iter().copied().fold(f64::INFINITY, f64::min);
    (a, mol.bas_coord(sh))
}

/// Normalized s-type overlap of two primitives (static mask).
fn s_overlap(a: f64, ra: [f64; 3], b: f64, rb: [f64; 3]) -> f64 {
    use core::f64::consts::PI;
    let (dx, dy, dz) = (ra[0] - rb[0], ra[1] - rb[1], ra[2] - rb[2]);
    let r2 = dx * dx + dy * dy + dz * dz;
    let ab = a * b;
    let apb = a + b;
    let nab = (2.0 * a / PI).powf(0.75) * (2.0 * b / PI).powf(0.75);
    nab * (PI / apb).powf(1.5) * (-ab / apb * r2).exp()
}
