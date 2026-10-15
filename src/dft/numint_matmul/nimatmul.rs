// Note on linkage:
//
// This algorithm implementation strictly requires linkage of openblas.
// If mkl is also linked and precedence is given to mkl, the algorithm will be very slow due to
// threading conflict. Use `patchelf` to remove mkl linkage from the binary if necessary.

use super::prelude::*;

/// Numerical integration driver using matrix-multiplication.
///
/// Holds molecular coordinates, grid weights, and caches AO integral evaluations
/// to avoid recomputation across multiple density/XCPot evaluations.
#[derive(Clone, Debug)]
pub struct NIMatmul<'a> {
    pub cint: CInt,
    pub coords: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    /// Atom index of the center each grid point was generated on (`usize::MAX` for grids
    /// attached to no atom, e.g. external grids); the same convention as `Grids::atm_idx`.
    ///
    /// The Becke grid-shift terms consume this attribution; see
    /// `crate::dft::gen_grids::becke_partitioning_deriv` for the ByAtom scheme that requires
    /// it to be non-decreasing (atom-grouped grids).
    pub atm_idx: Vec<usize>,
    /// Pre-partition (radial × angular) quadrature weight of each grid point, before the Becke
    /// partitioning factor; the same convention as `Grids::quadrature_weights`.
    pub quadrature_weights: Vec<f64>,

    /// Cache for computed AO values, keyed by derivative order (e.g., "deriv0", "deriv1", etc.).
    ///
    /// This cache is not designed to be modified by API caller in usual cases.
    pub cache_tensor: HashMap<String, TsrCow<'a, f64>>,

    /// Budget, in bytes, of whole-grid AO tables kept in [`Self::cache_tensor`] across calls.
    ///
    /// Evaluating a whole-grid AO table is a pure time/memory trade: the same data can be
    /// re-evaluated with libcint on demand.  Without a budget, one evaluation makes the table
    /// resident for the rest of the object's life, which is exactly what turns a one-shot
    /// preparation (the CP-KS `vxc`/`fxc`, a fock build) into a multi-GB resident allocation.
    /// Tables larger than the remaining budget are therefore served from the single transient
    /// slot [`Self::transient_ao`] instead of being retained.
    ///
    /// `usize::MAX` = no budget (retain everything, the historical behaviour), `0` = retain
    /// nothing.  Set by `REST_NIM_AO_CACHE_MB` (negative = no budget); default 256 MiB.
    pub ao_cache_budget_bytes: usize,

    /// Bytes currently retained in [`Self::cache_tensor`] under the `ao_deriv*` keys.
    pub ao_cached_bytes: usize,

    /// Single-slot AO table for requests that do not fit [`Self::ao_cache_budget_bytes`].
    ///
    /// It is replaced by the next such request and released by [`Self::drop_ao_cache`], so at
    /// most one over-budget whole-grid AO is alive per object, and none survives a phase
    /// boundary.  The stored derivative order is the one the table was evaluated at, so
    /// repeated requests for the same (or a lower) derivative still hit.
    transient_ao: Option<(usize, Tsr)>,

    /// Number of grid points to process in one chunk.
    ///
    /// Relations of size: full-grid > batch > chunk > per-grid = 1.
    ///
    /// This value is better set around KC of micro-kernel (256-512 for usual x86 server).
    /// Default to be 1536 (3-6 KCs).
    pub nchunk: usize,

    /// Number of grid points to process in one batch.
    ///
    /// Relations of size: full-grid > batch > chunk > per-grid = 1.
    ///
    /// Full grid requires `[ngrids, nao, ncomp]` AO tensor, which can be too large to fit in memory
    /// for big systems. That's why we need to split the grid into batches.
    ///
    /// This value is better set to a proper size, not exceeding available memory, and be multiple
    /// of `nchunk` for better performance.  It defaults to [`NIM_BATCH_CHUNKS`] chunks.
    ///
    /// **It is deliberately independent of `rayon::current_num_threads()`.**  It used to be
    /// `nchunk * current_num_threads()`, which made the *decomposition itself* a function of the
    /// pool size, with three measured consequences on a 40-core node: the batched consumers
    /// materialise one block's AO table at a time (`nbatch * nao * ncomp` doubles), so the
    /// transient slot — and with it the peak RSS — grew linearly with the core count; the block
    /// boundaries moved with the pool, so the SCF energy of one binary differed in its last
    /// digits between a 16- and a 40-thread run (the accumulation order is a function of the
    /// blocks); and the number of blocks a serial `grid_blocks` loop walks shrank as threads were
    /// added, i.e. the per-block overhead grew exactly when it was least affordable.  The
    /// decomposition of a fixed problem must not depend on how many workers happen to be
    /// available.
    pub nbatch: usize,
}

/// Reorder the grid arrays into atom-grouped order: `atm_idx` non-decreasing, atoms in ascending
/// order, grids of one atom consecutive (a stable counting sort on `atm_idx`).
///
/// This is the order the ByAtom attribution of the Becke grid-shift requires.  Grids attached to
/// no atom (`atm_idx >= natm`, e.g. external grids) keep their relative order at the tail.  The
/// values are only permuted, never changed.
///
/// If `atm_idx` is already non-decreasing — the common case, grids as generated atom-by-atom are
/// exactly in this order — the arrays are returned unchanged, at no extra cost.
pub fn regroup_grids_by_atom(
    coordinates: Vec<[f64; 3]>,
    weights: Vec<f64>,
    atm_idx: Vec<usize>,
    quadrature_weights: Vec<f64>,
    natm: usize,
) -> (Vec<[f64; 3]>, Vec<f64>, Vec<usize>, Vec<f64>) {
    // the common case: grids as generated (atom by atom) are already grouped
    if !atm_idx.windows(2).any(|w| w[1] < w[0]) {
        return (coordinates, weights, atm_idx, quadrature_weights);
    }
    let ngrids = coordinates.len();
    // counting sort with natm + 1 buckets; the last bucket collects unattributed grids
    let mut counts = vec![0usize; natm + 1];
    for &a in &atm_idx {
        counts[a.min(natm)] += 1;
    }
    let mut cursor = vec![0usize; natm + 2];
    for i in 0..=natm {
        cursor[i + 1] = cursor[i] + counts[i];
    }
    let mut write = cursor.clone();
    let mut coordinates_new = vec![[0.0f64; 3]; ngrids];
    let mut weights_new = vec![0.0f64; ngrids];
    let mut atm_idx_new = vec![0usize; ngrids];
    let mut quadrature_weights_new = vec![0.0f64; ngrids];
    for g in 0..ngrids {
        let bucket = atm_idx[g].min(natm);
        let p = write[bucket];
        write[bucket] += 1;
        coordinates_new[p] = coordinates[g];
        weights_new[p] = weights[g];
        atm_idx_new[p] = atm_idx[g];
        quadrature_weights_new[p] = quadrature_weights[g];
    }
    (coordinates_new, weights_new, atm_idx_new, quadrature_weights_new)
}

/// Grid points per batch, in units of [`NIMatmul::nchunk`].
///
/// Eight chunks is the value the batched consumers were tuned and validated at; it keeps the
/// batch at 12288 points, which is what an 8-thread pool used to get.  The pool size must not
/// scale it — see [`NIMatmul::nbatch`].  `REST_NIM_BATCH_CHUNKS` overrides it, which is how the
/// memory/overhead trade-off is tuned without a rebuild.
const NIM_BATCH_CHUNKS: usize = 8;

/// Number of `nchunk`s in one batch, from `REST_NIM_BATCH_CHUNKS` (default
/// [`NIM_BATCH_CHUNKS`], clamped to at least 1).
fn batch_chunks_from_env() -> usize {
    std::env::var("REST_NIM_BATCH_CHUNKS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(NIM_BATCH_CHUNKS)
        .max(1)
}

/// Retained-AO budget of a new [`NIMatmul`], from `REST_NIM_AO_CACHE_MB`.
///
/// A negative value means "no budget" (the historical behaviour: every evaluated AO table is
/// retained), `0` means "retain nothing", and the default is 256 MiB — large enough to keep the
/// tables of small systems resident, small enough that a whole SCF-grid table (GBs at
/// def2-TZVP/polyene scale) is never retained by side effect.
fn ao_cache_budget_from_env() -> usize {
    match std::env::var("REST_NIM_AO_CACHE_MB").ok().and_then(|v| v.trim().parse::<f64>().ok()) {
        Some(mb) if mb < 0.0 => usize::MAX,
        Some(mb) => (mb * 1.0e6).max(0.0) as usize,
        None => 256 * 1024 * 1024,
    }
}

/// `REST_NIM_MEMDBG` gates the AO-cache accounting lines.
fn nim_memdbg() -> bool {
    std::env::var("REST_NIM_MEMDBG").is_ok()
}

/// Dense byte count of an AO table.
fn ao_table_bytes(t: &Tsr) -> usize {
    t.shape().iter().product::<usize>() * std::mem::size_of::<f64>()
}

impl<'a> NIMatmul<'a> {
    /// Creates a new instance with the given integral engine, grid coordinates/weights and the
    /// per-grid atom attribution and pre-partition quadrature weights (see the struct fields).
    pub fn new(
        cint: &CInt,
        coords: &[[f64; 3]],
        weights: &[f64],
        atm_idx: &[usize],
        quadrature_weights: &[f64],
    ) -> Self {
        assert!(coords.len() == weights.len(), "Number of coordinates must match number of weights");
        assert!(coords.len() == atm_idx.len(), "Number of coordinates must match length of atm_idx");
        assert!(
            coords.len() == quadrature_weights.len(),
            "Number of coordinates must match number of quadrature weights"
        );
        let nchunk = 1536;
        // Thread-independent: see `nbatch` and `NIM_BATCH_CHUNKS`.
        let nbatch = nchunk * batch_chunks_from_env();
        let out = Self {
            cint: cint.clone(),
            coords: coords.to_vec(),
            weights: weights.to_vec(),
            atm_idx: atm_idx.to_vec(),
            quadrature_weights: quadrature_weights.to_vec(),
            cache_tensor: HashMap::new(),
            ao_cache_budget_bytes: ao_cache_budget_from_env(),
            ao_cached_bytes: 0,
            transient_ao: None,
            nchunk,
            nbatch,
        };
        if nim_memdbg() {
            let ngrids = out.coords.len();
            let nao = out.cint.nao();
            let per_table = ngrids * nao * std::mem::size_of::<f64>();
            // One block's AO table is what a batched consumer holds at a time; report it so a
            // run's log shows the decomposition is independent of the pool size.
            let block_mb = |ncomp: usize| {
                (out.nbatch * nao * ncomp * std::mem::size_of::<f64>()) as f64 / 1024.0 / 1024.0
            };
            println!(
                "[nim] new object: ngrids {:8} | nao {:5} | nchunk {} nbatch {} ({} blocks) | block AO table {:.1}/{:.1}/{:.1} MB (deriv 0/2/3) | whole-grid {:7.1} MB per deriv component | budget {} MB",
                ngrids,
                nao,
                out.nchunk,
                out.nbatch,
                ngrids.div_ceil(out.nbatch.max(1)),
                block_mb(1),
                block_mb(AO_DERIV_DIM[2]),
                block_mb(AO_DERIV_DIM[3]),
                per_table as f64 / 1024.0 / 1024.0,
                if out.ao_cache_budget_bytes == usize::MAX {
                    "unlimited".to_string()
                } else {
                    format!("{:.0}", out.ao_cache_budget_bytes as f64 / 1024.0 / 1024.0)
                },
            );
        }
        out
    }

    /// Clone everything, except the cached tensors.
    pub fn duplicate(&self) -> Self {
        Self {
            cint: self.cint.clone(),
            coords: self.coords.clone(),
            weights: self.weights.clone(),
            atm_idx: self.atm_idx.clone(),
            quadrature_weights: self.quadrature_weights.clone(),
            cache_tensor: HashMap::new(),
            ao_cache_budget_bytes: self.ao_cache_budget_bytes,
            ao_cached_bytes: 0,
            transient_ao: None,
            nchunk: self.nchunk,
            nbatch: self.nbatch,
        }
    }

    /// Release every whole-grid AO table this object retains (retained cache and the transient
    /// slot), and report how many bytes that freed when `REST_NIM_MEMDBG` is set.
    ///
    /// The AO tables are a pure cache: they are recomputed by [`Self::get_cached_ao`] on demand,
    /// so calling this never changes a result.  It is meant for phase boundaries — after a
    /// one-shot preparation has consumed its AO, before a long solve starts — where keeping the
    /// table would only inflate the peak.
    ///
    /// The integrator's own workspace (grids, coordinates, weights) is untouched.
    pub fn drop_ao_cache(&mut self) -> usize {
        let mut freed = 0usize;
        let keys: Vec<String> = self.cache_tensor.keys().cloned().collect();
        for key in keys {
            let is_ao = key.strip_prefix("ao_deriv").and_then(|s| s.parse::<usize>().ok()).is_some();
            if is_ao {
                if let Some(t) = self.cache_tensor.remove(&key) {
                    let v = t.view();
                    freed += v.shape().iter().product::<usize>() * std::mem::size_of::<f64>();
                }
            }
        }
        if let Some((_, t)) = self.transient_ao.take() {
            freed += ao_table_bytes(&t);
        }
        self.ao_cached_bytes = 0;
        if nim_memdbg() {
            println!("[nim] drop_ao_cache: freed {:.1} MB", freed as f64 / 1024.0 / 1024.0);
        }
        freed
    }

    /// Release only the transient slot (the over-budget table), keeping the retained cache.
    ///
    /// Used by one-shot consumers (a fock build) that are the last user of a table which was too
    /// large to retain: the table cannot serve anyone else, so leaving it resident would only
    /// inflate the peak until the next over-budget evaluation replaces it.
    pub fn drop_transient_ao(&mut self) -> usize {
        match self.transient_ao.take() {
            Some((_, t)) => {
                let bytes = ao_table_bytes(&t);
                if nim_memdbg() {
                    println!("[nim] drop_transient_ao: freed {:.1} MB", bytes as f64 / 1024.0 / 1024.0);
                }
                bytes
            },
            None => 0,
        }
    }

    /// Whether a whole-grid AO table of derivative order `deriv` would be *retained* under the
    /// current budget (rather than served from the transient slot).
    ///
    /// Consumers that would otherwise materialize the whole grid use this to decide between the
    /// plain call (the table becomes a reusable cache — the historical behaviour) and a blocked
    /// loop (the table never exists as a whole).
    pub fn ao_table_would_retain(&self, deriv: usize) -> bool {
        assert!(deriv < AO_DERIV_DIM.len(), "Derivative order {deriv} is too high");
        let bytes = self.coords.len() * self.cint.nao() * AO_DERIV_DIM[deriv] * std::mem::size_of::<f64>();
        self.ao_cache_budget_bytes == usize::MAX
            || bytes <= self.ao_cache_budget_bytes.saturating_sub(self.ao_cached_bytes)
    }

    /// Grid blocks used by the memory-bounded consumers: `(start, end)` ranges in ascending
    /// `start` order, of at most `nbatch` points each.
    ///
    /// `nbatch` is the same size the batched Krylov response path already uses
    /// (`split_batch` + one `get_cached_ao` per batch), so blocking a preparation this way
    /// reproduces the AO-evaluation pattern of the iterative path exactly, and the accumulation
    /// order stays deterministic.
    pub fn grid_blocks(&self, ngrids: usize) -> Vec<(usize, usize)> {
        let nbatch = self.nbatch.max(1);
        (0..ngrids).step_by(nbatch).map(|s| (s, (s + nbatch).min(ngrids))).collect()
    }

    /// Creates a new instance for a subset of the grid points, slicing from `start` to `end`
    /// (exclusive).
    ///
    /// Cached AO tensors for the full grid are sliced accordingly and cached for the batch.
    ///
    /// This is useful for processing the grid in batches to reduce memory usage.
    pub fn split_batch(&self, start: usize, end: usize) -> NIMatmul<'_> {
        assert!(start < end && end <= self.coords.len(), "Invalid batch range");
        let mut new = self.duplicate();
        new.coords = self.coords[start..end].to_vec();
        new.weights = self.weights[start..end].to_vec();
        new.atm_idx = self.atm_idx[start..end].to_vec();
        new.quadrature_weights = self.quadrature_weights[start..end].to_vec();
        // if AO cached for the full grid exists, slice and cache the AO for the batch
        let mut cached_tensors = HashMap::new();
        for keys in self.cache_tensor.keys() {
            if keys.strip_prefix("ao_deriv").and_then(|s| s.parse::<usize>().ok()).is_some() {
                let ao_full = self.cache_tensor.get(keys).unwrap();
                let ao_batch = ao_full.i(start..end);
                cached_tensors.insert(keys.clone(), ao_batch.into_cow());
            }
        }
        new.cache_tensor = cached_tensors;
        new
    }

    /// Evaluates AO integrals for the given derivative order and returns as a tensor.
    ///
    /// The returned tensor has shape `[ngrids, nao, ncomp]` where `ncomp = AO_DERIV_DIM[deriv]`.
    pub fn prepare_ao(&self, deriv: usize) -> Tsr {
        let eval_name = format!("deriv{}", deriv);
        let (out, shape) = self.cint.eval_gto(&eval_name, &self.coords).into();
        let device = DeviceBLAS::default();
        rt::asarray((out, shape, &device))
    }

    /// Returns cached AO values for the given derivative order, computing and caching if needed.
    ///
    /// When a higher derivative order has already been cached, the required subset is returned
    /// directly without recomputation.
    ///
    /// A table is only *retained* (kept for later calls) when it fits
    /// [`Self::ao_cache_budget_bytes`]; otherwise it lives in the single transient slot for as
    /// long as the next over-budget request, and the following call for the same derivative
    /// still hits it.  `REST_NIM_AO_CACHE_MB` (negative = unlimited) selects the budget.
    pub fn get_cached_ao(&mut self, deriv: usize) -> TsrView<'_> {
        assert!(
            deriv < AO_DERIV_DIM.len(),
            "Derivative order {deriv} is too high, max supported is {}",
            AO_DERIV_DIM.len() - 1
        );
        // determine the maximum ao deriv that have already been computed and cached
        let filter_closure = |k: &String| k.strip_prefix("ao_deriv").and_then(|s| s.parse::<usize>().ok());
        let max_cached_deriv = self.cache_tensor.keys().filter_map(filter_closure).max();
        // if the requested deriv is already cached, return it
        if let Some(max_deriv) = max_cached_deriv {
            if max_deriv >= deriv {
                let cache_key = format!("ao_deriv{}", max_deriv);
                return self.cache_tensor.get(&cache_key).unwrap().i((.., .., ..AO_DERIV_DIM[deriv]));
            }
        }
        // same, for the transient slot (a table that did not fit the budget)
        if let Some((max_deriv, _)) = self.transient_ao.as_ref() {
            if *max_deriv >= deriv {
                return self.transient_ao.as_ref().unwrap().1.i((.., .., ..AO_DERIV_DIM[deriv]));
            }
        }

        // otherwise, compute the requested ao deriv
        let ao = self.prepare_ao(deriv);
        let bytes = ao_table_bytes(&ao);
        let budget = self.ao_cache_budget_bytes;
        let retain = budget == usize::MAX || bytes <= budget.saturating_sub(self.ao_cached_bytes);
        if nim_memdbg() {
            println!(
                "[nim] get_cached_ao deriv{} : {:?} = {:.1} MB -> {} (budget {:.0} MB, retained {:.0} MB)",
                deriv,
                ao.shape(),
                bytes as f64 / 1024.0 / 1024.0,
                if retain { "retained" } else { "transient" },
                if budget == usize::MAX {
                    -1.0
                } else {
                    budget as f64 / 1024.0 / 1024.0
                },
                self.ao_cached_bytes as f64 / 1024.0 / 1024.0,
            );
        }
        if retain {
            let key = format!("ao_deriv{}", deriv);
            self.cache_tensor.insert(key.clone(), ao.into_cow());
            self.ao_cached_bytes += bytes;
            return self.cache_tensor.get(&key).unwrap().view();
        }
        self.transient_ao = Some((deriv, ao));
        self.transient_ao.as_ref().unwrap().1.view()
    }

    /// Evaluates density from density matrices.
    ///
    /// # Parameters
    ///
    /// - `dm_list` : density matrices, each of shape `[nao, nao]`; one per set
    /// - `den_type` : which density components to compute
    ///
    /// # Returns
    ///
    /// Density tensor of shape `[ngrids, nvar, nset]`.
    pub fn make_rho_from_dm(&mut self, dm_list: &[TsrView], den_type: XCDenType) -> Tsr {
        let nchunk = self.nchunk;
        let ao = self.get_cached_ao(den_type.num_ao_deriv());

        let ngrids = ao.shape()[0];
        let nao = ao.shape()[1];
        for dm in dm_list {
            ni_check_shape!(dm.ndim(), 2, "Each density matrix must be 2-dim");
            ni_check_shape!(dm.shape()[0..2], [nao, nao], "Density matrix must match AO dimension");
        }
        let nset = dm_list.len();

        let out_shape = [ngrids, den_type.num_nvar(), nset];
        let device = ao.device().clone();
        let mut out = rt::zeros((out_shape.f(), &device));
        get_rho_from_dm_with_output(ao, dm_list, den_type, out.view_mut(), nchunk);
        out
    }

    /// Evaluates density from orbital coefficients where bra and ket are the same.
    ///
    /// # Parameters
    ///
    /// - `bra_list` : orbital coefficient matrices, each of shape `[nao, nocc_i]`
    /// - `den_type` : which density components to compute
    ///
    /// # Returns
    ///
    /// Density tensor of shape `[ngrids, nvar, nset]`.
    pub fn make_rho_from_homogeneous_braket(&mut self, bra_list: &[TsrView], den_type: XCDenType) -> Tsr {
        let nchunk = self.nchunk;
        let ao = self.get_cached_ao(den_type.num_ao_deriv());

        let ngrids = ao.shape()[0];
        let nao = ao.shape()[1];
        for bra in bra_list {
            ni_check_shape!(bra.ndim(), 2, "Each braket must be 2-dim");
            ni_check_shape!(bra.shape()[0], nao, "Bra's first dimension must match AO dimension");
        }
        let nset = bra_list.len();
        let out_shape = [ngrids, den_type.num_nvar(), nset];
        let device = ao.device().clone();
        let mut out = rt::zeros((out_shape.f(), &device));
        get_rho_from_homogeneous_braket_with_output(ao, bra_list, den_type, out.view_mut(), nchunk);
        out
    }

    /// Evaluates density from one shared bra and multiple kets.
    ///
    /// # Parameters
    ///
    /// - `bra` : shared orbital coefficient matrix, shape `[nao, nocc]`
    /// - `ket_list` : orbital coefficient matrices, each of shape `[nao, nocc]`
    /// - `den_type` : which density components to compute
    ///
    /// # Returns
    ///
    /// Density tensor of shape `[ngrids, nvar, nset]`.
    pub fn make_rho_from_one_bra_mult_ket(&mut self, bra: TsrView, ket_list: &[TsrView], den_type: XCDenType) -> Tsr {
        let nchunk = self.nchunk;
        let ao = self.get_cached_ao(den_type.num_ao_deriv());

        let ngrids = ao.shape()[0];
        let nao = ao.shape()[1];
        ni_check_shape!(bra.ndim(), 2, "Bra must be 2-dim");
        ni_check_shape!(bra.shape()[0], nao, "Bra first dimension must match AO dimension");
        let nocc = bra.shape()[1];
        for ket in ket_list {
            ni_check_shape!(ket.ndim(), 2, "Each ket must be 2-dim");
            ni_check_shape!(ket.shape()[0], nao, "Ket first dimension must match AO dimension");
            ni_check_shape!(ket.shape()[1], nocc, "Ket second dimension must match bra");
        }
        let nset = ket_list.len();

        let out_shape = [ngrids, den_type.num_nvar(), nset];
        let device = ao.device().clone();
        let mut out = rt::zeros((out_shape.f(), &device));
        get_rho_from_one_bra_mult_ket_with_output(ao, bra, ket_list, den_type, out.view_mut(), nchunk);
        out
    }

    /// Evaluates density from multiple bra-ket pairs.
    ///
    /// # Parameters
    ///
    /// - `bra_list` : orbital coefficient matrices for bra, each of shape `[nao, nocc_i]`
    /// - `ket_list` : orbital coefficient matrices for ket, each of shape `[nao, nocc_i]`
    /// - `den_type` : which density components to compute
    ///
    /// # Returns
    ///
    /// Density tensor of shape `[ngrids, nvar, nset]`.
    pub fn make_rho_from_mult_bra_mult_ket(
        &mut self,
        bra_list: &[TsrView],
        ket_list: &[TsrView],
        den_type: XCDenType,
    ) -> Tsr {
        let nchunk = self.nchunk;
        let ao = self.get_cached_ao(den_type.num_ao_deriv());

        let ngrids = ao.shape()[0];
        let nao = ao.shape()[1];
        for (bra, ket) in bra_list.iter().zip(ket_list.iter()) {
            ni_check_shape!(bra.ndim(), 2, "Each bra must be 2-dim");
            ni_check_shape!(ket.ndim(), 2, "Each ket must be 2-dim");
            ni_check_shape!(nao, bra.shape()[0], "Bra first dimension must match AO dimension");
            ni_check_shape!(nao, ket.shape()[0], "Ket first dimension must match AO dimension");
            ni_check_shape!(bra.shape()[1], ket.shape()[1], "Bra and ket occupation must match");
        }
        let nset = bra_list.len();

        let out_shape = [ngrids, den_type.num_nvar(), nset];
        let device = ao.device().clone();
        let mut out = rt::zeros((out_shape.f(), &device));
        get_rho_from_mult_bra_mult_ket_with_output(ao, bra_list, ket_list, den_type, out.view_mut(), nchunk);
        out
    }

    /// Evaluates XC potential (1st order) with vxc_eff.
    ///
    /// # Parameters
    ///
    /// - `vxc_eff` : effective XC potential, shape `[ngrids, nvar]` for RKS, `[ngrids, nvar, 2]`
    ///   for UKS
    /// - `den_type` : which density components to compute
    /// - `spin` : spin polarization or not
    ///
    /// # Returns
    ///
    /// XC potential, shape `[nao, nao]` for RKS, `[nao, nao, 2]` for UKS.
    pub fn make_vxc_pot_with_eff(&mut self, vxc_eff: TsrView, den_type: XCDenType, spin: XCSpin) -> Tsr {
        let nchunk = self.nchunk;
        let weights_data = self.weights.clone();
        let ao = self.get_cached_ao(den_type.num_ao_deriv());
        let nao = ao.shape()[1];
        let device = ao.device().clone();
        let weights_tsr = rt::asarray((weights_data.clone(), [weights_data.len()], &device));

        match spin {
            XCSpin::Unpolarized => {
                let mut out = rt::zeros(([nao, nao], &device));
                rks_vxc_pot_with_eff_with_output(den_type, vxc_eff, ao, weights_tsr.view(), out.view_mut(), nchunk);
                out
            },
            XCSpin::Polarized => {
                let mut out = rt::zeros(([nao, nao, 2], &device));
                uks_vxc_pot_with_eff_with_output(den_type, vxc_eff, ao, weights_tsr.view(), out.view_mut(), nchunk);
                out
            },
        }
    }

    /// Evaluates XC potential (2nd order) with fxc_eff.
    ///
    /// # Parameters
    ///
    /// - `fxc_eff` : effective XC kernel, shape `[ngrids, nvar, nvar]` for RKS, `[ngrids, nvar, 2,
    ///   nvar, 2]` for UKS
    /// - `rho1` : first-order density response, shape `[ngrids, nvar, nset]` for RKS, `[ngrids,
    ///   nvar, 2, nset]` for UKS
    /// - `den_type` : which density components to compute
    /// - `spin` : spin polarization or not
    ///
    /// # Returns
    ///
    /// Second-order XC potential, shape `[nao, nao, nset]` for RKS, `[nao, nao, 2, nset]` for UKS.
    pub fn make_fxc_pot_with_eff(&mut self, fxc_eff: TsrView, rho1: TsrView, den_type: XCDenType, spin: XCSpin) -> Tsr {
        let nchunk = self.nchunk;
        let weights_data = self.weights.clone();
        let ao = self.get_cached_ao(den_type.num_ao_deriv());
        let nao = ao.shape()[1];
        let device = ao.device().clone();
        let weights_tsr = rt::asarray((weights_data.clone(), [weights_data.len()], &device));

        match spin {
            XCSpin::Unpolarized => {
                let nset = rho1.shape()[2];
                let mut out = rt::zeros(([nao, nao, nset], &device));
                rks_fxc_pot_with_eff_with_output(
                    den_type,
                    fxc_eff,
                    rho1,
                    ao,
                    weights_tsr.view(),
                    out.view_mut(),
                    nchunk,
                );
                out
            },
            XCSpin::Polarized => {
                let nset = rho1.shape()[3];
                let mut out = rt::zeros(([nao, nao, 2, nset], &device));
                uks_fxc_pot_with_eff_with_output(
                    den_type,
                    fxc_eff,
                    rho1,
                    ao,
                    weights_tsr.view(),
                    out.view_mut(),
                    nchunk,
                );
                out
            },
        }
    }

    /// Evaluates XC potential (2nd order) with fxc_eff, applying bra transformation.
    ///
    /// This function only works for RKS (spin-unpolarized case).
    ///
    /// Bra is usually the occupied orbital coefficient, which can lower the computational cost.
    ///
    /// # Parameters
    ///
    /// - `fxc_eff` : effective XC kernel, shape `[ngrids, nvar, nvar]`
    /// - `rho1` : first-order density response, shape `[ngrids, nvar, nset]`
    /// - `bra` : bra orbital coefficients, shape `[nao, nocc]`
    /// - `den_type` : which density components to compute
    ///
    /// # Returns
    ///
    /// Bra-transformed XC potential, shape `[nao, nocc, nset]`.
    pub fn make_rks_fxc_pot_with_eff_bra_trans(
        &mut self,
        fxc_eff: TsrView,
        rho1: TsrView,
        bra: TsrView,
        den_type: XCDenType,
    ) -> Tsr {
        let nchunk = self.nchunk;
        let weights_data = self.weights.clone();
        let ao = self.get_cached_ao(den_type.num_ao_deriv());
        let nao = ao.shape()[1];
        let device = ao.device().clone();
        let weights_tsr = rt::asarray((weights_data.clone(), [weights_data.len()], &device));

        let nset = rho1.shape()[2];
        let nocc = bra.shape()[1];
        let mut out = rt::zeros(([nao, nocc, nset], &device));
        rks_fxc_pot_with_eff_bra_trans_with_output(
            den_type,
            fxc_eff,
            rho1,
            ao,
            weights_tsr.view(),
            bra,
            out.view_mut(),
            nchunk,
        );
        out
    }

    /// Evaluates XC potential (2nd order) with fxc_eff, applying bra transformation.
    ///
    /// This function only works for UKS (spin-polarized case).
    ///
    /// Bra is usually the occupied orbital coefficient, which can lower the computational cost.
    ///
    /// # Parameters
    ///
    /// - `fxc_eff` : effective XC kernel, shape `[ngrids, nvar, 2, nvar, 2]`
    /// - `rho1` : first-order density response, shape `[ngrids, nvar, 2, nset]`
    /// - `bra` : bra orbital coefficients, first shape `[nao, nocc_alpha]`, second shape `[nao,
    ///   nocc_beta]`
    /// - `den_type` : which density components to compute
    pub fn make_uks_fxc_pot_with_eff_bra_trans(
        &mut self,
        fxc_eff: TsrView,
        rho1: TsrView,
        bra: &[TsrView; 2],
        den_type: XCDenType,
    ) -> [Tsr; 2] {
        let nchunk = self.nchunk;
        let weights_data = self.weights.clone();
        let ao = self.get_cached_ao(den_type.num_ao_deriv());
        let nao = ao.shape()[1];
        let device = ao.device().clone();
        let weights_tsr = rt::asarray((weights_data.clone(), [weights_data.len()], &device));

        let nset = rho1.shape()[3];
        let nocc_alpha = bra[0].shape()[1];
        let nocc_beta = bra[1].shape()[1];
        let mut out_alpha = rt::zeros(([nao, nocc_alpha, nset], &device));
        let mut out_beta = rt::zeros(([nao, nocc_beta, nset], &device));
        uks_fxc_pot_with_eff_bra_trans_with_output(
            den_type,
            fxc_eff,
            rho1,
            ao,
            weights_tsr.view(),
            bra,
            &mut [out_alpha.view_mut(), out_beta.view_mut()],
            nchunk,
        );
        [out_alpha, out_beta]
    }

    /// Evaluates XC potential (3rd order) with kxc_eff.
    ///
    /// # Parameters
    ///
    /// - `kxc_eff` : effective XC kernel, shape `[ngrids, nvar, nvar, nvar]` for RKS, `[ngrids,
    ///   nvar, 2, nvar, 2, nvar, 2]` for UKS
    /// - `rho1` : first-order density response, shape `[ngrids, nvar, nset1]` for RKS, `[ngrids,
    ///   nvar, 2, nset1]` for UKS
    /// - `rho2` : second-order density response, shape `[ngrids, nvar, nset2]` for RKS, `[ngrids,
    ///   nvar, 2, nset2]` for UKS
    /// - `den_type` : which density components to compute
    /// - `spin` : spin polarization or not
    ///
    /// # Returns
    ///
    /// Third-order XC potential, shape `[nao, nao, nset1, nset2]` for RKS,
    /// `[nao, nao, 2, nset1, nset2]` for UKS.
    pub fn make_kxc_pot_with_eff(
        &mut self,
        kxc_eff: TsrView,
        rho1: TsrView,
        rho2: TsrView,
        den_type: XCDenType,
        spin: XCSpin,
    ) -> Tsr {
        let nchunk = self.nchunk;
        let weights_data = self.weights.clone();
        let ao = self.get_cached_ao(den_type.num_ao_deriv());
        let nao = ao.shape()[1];
        let device = ao.device().clone();
        let weights_tsr = rt::asarray((weights_data.clone(), [weights_data.len()], &device));

        match spin {
            XCSpin::Unpolarized => {
                let nset1 = rho1.shape()[2];
                let nset2 = rho2.shape()[2];
                let mut out = rt::zeros(([nao, nao, nset1, nset2], &device));
                rks_kxc_pot_with_eff_with_output(
                    den_type,
                    kxc_eff,
                    rho1,
                    rho2,
                    ao,
                    weights_tsr.view(),
                    out.view_mut(),
                    nchunk,
                );
                out
            },
            XCSpin::Polarized => {
                let nset1 = rho1.shape()[3];
                let nset2 = rho2.shape()[3];
                let mut out = rt::zeros(([nao, nao, 2, nset1, nset2], &device));
                uks_kxc_pot_with_eff_with_output(
                    den_type,
                    kxc_eff,
                    rho1,
                    rho2,
                    ao,
                    weights_tsr.view(),
                    out.view_mut(),
                    nchunk,
                );
                out
            },
        }
    }
}
