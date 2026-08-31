//! Standalone Becke partitioning implementation for DFT numerical integration.
//!
//! This file is intended not to interfere to other parts of the codebase.
//!
//! The file is organized in three tiers, mirrored by the `#region` markers below: `Becke*`
//! (main-thread interface: public types, entry points, molecular tables, and the per-call
//! context), `Task*` (parallel-thread interface: batch task splitting and per-worker buffers),
//! and `Lane*` (grid-iteration interface: lane gathering, scratch, evaluation, and write-back);
//! cross-cutting utilities (switch functions, FpSimd helpers, misc) come last.
//!
//! This code tries to somehow optimize with SIMD. However, this requires manually setting
//! `target-cpu` or something similar when cargo build (only release is not enough). The efficiency
//! is not tested currently, but theoretically should not be too pitiful I believe.

use itertools::Itertools;
use rest_libcint::gto::deriv_util::FpSimd;
use rayon::prelude::*;
use std::sync::Mutex;

use super::bragg;
use super::parameters;

#[allow(non_camel_case_types)]
type f64simd = FpSimd<f64, SIMDD>;

const SIMDD: usize = 8;
const INVTOL: f64 = 1e-14;
/// Side length (in f64simd elements) of the L1-resident accumulator tiles of
/// the deriv-2 rank-1 cross-term matrix; see [`eval_lane_ddw`].
const CROSS_CB: usize = 12;

/* #region Becke*: main-thread interface (public API, molecular tables, context) */

/// Output of Becke partitioning function.
///
/// - `w`: Partition weights for each grid point. Shape `(ngrids, )`.
/// - `dw`: 1st derivative of partition weights for each grid point. Shape `(natm, 3, ngrids)` in
///   row-major or `(ngrids, 3, natm)` in column-major. We do not provide shape information here,
///   since it can be derived from length of variable `w`.
/// - `ddw`: 2nd derivative of partition weights for each grid point. Shape `(natm, 3, natm, 3,
///   ngrids)` in row-major or `(ngrids, 3, natm, 3, natm)` in column-major.
/// - `c`: Contracted weights for each grid point. Shape `(nset, )`.
/// - `dc`: 1st derivative of contracted weights for each grid point. Shape `(natm, 3, nset)` in
///   row-major or `(nset, 3, natm)` in column-major.
/// - `ddc`: 2nd derivative of contracted weights for each grid point. Shape `(natm, 3, natm, 3,
///   nset)` in row-major or `(nset, 3, natm, 3, natm)` in column-major.
#[derive(Debug, Clone, Default)]
pub struct BeckePartitionOutput {
    pub w: Option<Vec<f64>>,
    pub dw: Option<Vec<f64>>,
    pub ddw: Option<Vec<f64>>,
    pub c: Option<Vec<f64>>,
    pub dc: Option<Vec<f64>>,
    pub ddc: Option<Vec<f64>>,
}

/// Arguments for Becke partitioning function.
///
/// For usual usage, just default should work. You can contract the weights manually outside becke
/// partition function.
///
/// For advanced usage, user may want not only to compute the partition weights, but also to
/// contract them (especially 2nd derivative that generate a large $O(N^3)$ tensor of `ddw`, and the
/// user may want to avoid it). For this case, the `contract` field will provide a slice, of shape
/// `(nset, ngrids)` in row-major or `(ngrids, nset)` in column-major, to contract the weights. For
/// the output shape, see [`BeckePartitionOutput`].
///
/// The user may want to specify if they want to output `w`, `dw`, `ddw`. However,
/// - If deriv level is 0, `dw` and `ddw` will still be None, even `output_dw/ddw` is true. Same
///   applies to deriv level 1.
/// - If `contract_w/dw/ddw` is provided, `c/dc/ddc` will be computed with the given deriv level by
///   default. But if it is not provided, `c/dc/ddc` will be None, even `output_c/dc/ddc` is true.
#[derive(Debug)]
pub struct BeckePartitionArg<'a> {
    pub output_w: bool,
    pub output_dw: bool,
    pub output_ddw: bool,
    pub contract_w: Option<&'a [f64]>,
    pub contract_dw: Option<&'a [f64]>,
    pub contract_ddw: Option<&'a [f64]>,
}

impl<'a> Default for BeckePartitionArg<'a> {
    fn default() -> Self {
        Self {
            output_w: true,
            output_dw: true,
            output_ddw: true,
            contract_w: None,
            contract_dw: None,
            contract_ddw: None,
        }
    }
}

impl<'a> BeckePartitionArg<'a> {
    /// Set the contraction weights for the Becke partitioning.
    pub fn set_contract(mut self, deriv: usize, contract: &'a [f64]) -> Self {
        match deriv {
            0 => self.contract_w = Some(contract),
            1 => self.contract_dw = Some(contract),
            2 => self.contract_ddw = Some(contract),
            _ => panic!("Unsupported derivative order: {}", deriv),
        }
        self
    }

    /// Set the output flags for the Becke partitioning.
    pub fn set_output_weights(mut self, deriv: usize, value: bool) -> Self {
        match deriv {
            0 => self.output_w = value,
            1 => self.output_dw = value,
            2 => self.output_ddw = value,
            _ => panic!("Unsupported derivative order: {}", deriv),
        }
        self
    }
}

/// Attribution of the grid points to their generating atoms (the atoms whose Lebedev
/// angular grids produced them).
///
/// The two variants also select the evaluation strategy: [`Self::ByGrid`] attributes
/// each grid point on its own (lane-wise masks), while [`Self::ByAtom`] guarantees that
/// each atom's grids form one contiguous interval, so every batch carries a single
/// definite generating atom and the partition selection is index-based.
#[derive(Debug, Clone, Copy)]
pub enum AtmIndices<'a> {
    /// Per-grid generating atom index, length `ngrids`.  Usually values satisfy
    /// `0 <= value < natm`; values `>= natm` are treated as padding and select no
    /// partition weight.  Grid points of different atoms may be interleaved freely.
    ByGrid(&'a [usize]),
    /// Cumulative grid boundaries per atom, length `natm + 1`: atom `A` owns the grids
    /// `[indices[A], indices[A + 1])` (the grid must be grouped by atom).  Batches
    /// never cross an atom boundary.
    ByAtom(&'a [usize]),
}

/// Check that a per-grid atom-index array is monotonically non-decreasing, and convert
/// it to the cumulative per-atom grid boundaries of [`AtmIndices::ByAtom`].
///
/// Grids grouped by their generating atom (all grids of atom `A` consecutive, atoms in
/// ascending order) are exactly what the ByAtom attribution requires; an array that is
/// not so grouped (e.g. round-robin permuted grids) yields `None`.
///
/// - Atoms that generated no grids keep their slot: their interval
///   `[split[A], split[A + 1])` is empty (`split[A] == split[A + 1]`).
/// - Values `>= natm` are valid (padding, e.g. `usize::MAX` for grids associated with
///   no atom); by monotonicity they can only form a trailing segment, which is attached
///   to no atom: the boundaries end at `split[natm]`, and the trailing grids beyond are
///   left out of the split (discarded by the ByAtom way of use).  Without such padding,
///   `split[natm] == atm_idx.len()`.
pub fn try_atm_quad_split(atm_idx: &[usize], natm: usize) -> Option<Vec<usize>> {
    // monotonic (non-decreasing) check: grids must be grouped by generating atom
    if atm_idx.windows(2).any(|w| w[1] < w[0]) {
        return None;
    }
    // the boundary after atom A is the first grid whose value exceeds A; scan once,
    // so atoms absent from atm_idx (value already beyond A) get an empty interval
    let mut split = Vec::with_capacity(natm + 1);
    split.push(0);
    let mut pos = 0;
    for A in 0..natm {
        while pos < atm_idx.len() && atm_idx[pos] <= A {
            pos += 1;
        }
        split.push(pos);
    }
    Some(split)
}

/// Molecular tables of the Becke partitioning: the data that depends only on the
/// molecule (atomic positions, radii adjustment factors, interatomic distances and
/// their 1st derivatives), precomputed once and shared immutably by any number of
/// [`becke_partition_with_tables`] calls.
///
/// Chunk-level drivers call the partition function per chunk; building these tables
/// inside the per-call context would re-transpose the adjustment factor and
/// re-compute the (rayon-parallel) distance tables for every chunk.  Hoisting them
/// here keeps the per-chunk context free of molecular precomputation.
#[derive(Debug, Clone)]
pub struct BeckeMolTables {
    /// Atomic coordinates, length `natm`.
    pub atm_coords: Vec<[f64; 3]>,
    /// Row-major adjustment factors (usually adjusted by atomic radii):
    /// `adjustment_factor[A][B]` reads entry `(A, B)` of the `(natm, natm)` matrix.
    /// cf Becke 1988 eqs (A2, A5).
    pub adjustment_factor: Vec<Vec<f64>>,
    /// Interatomic distances `[A][B]`; the diagonal is `INFINITY`.
    pub atm_dist: Vec<Vec<f64>>,
    /// 1st derivative of the interatomic distances `[A][B][t]` (computed for
    /// `deriv >= 1` only).
    pub dR_atm_dist: Option<Vec<Vec<[f64; 3]>>>,
}

impl BeckeMolTables {
    /// Validate the row-major adjustment factor and precompute the interatomic
    /// distances (and their 1st derivatives for `deriv >= 1`).
    pub fn new(atm_coords: &[[f64; 3]], adjustment_factor: &[Vec<f64>], deriv: usize) -> Self {
        let natm = atm_coords.len();
        assert!(adjustment_factor.len() == natm, "adjustment_factor must have natm rows");
        assert!(adjustment_factor.iter().all(|r| r.len() == natm), "adjustment_factor rows must have length natm");

        // generate atm_dist before iteration to grid coordinates
        // since it is not bottleneck, we duplicate the calculation for A > B.
        let atm_dist: Vec<f64> = (0..natm * natm)
            .into_par_iter()
            .map(|idx| {
                let (A, B) = (idx / natm, idx % natm);
                match A == B {
                    true => f64::INFINITY,
                    false => dist3_naive(&atm_coords[A], &atm_coords[B]),
                }
            })
            .collect();
        let atm_dist: Vec<Vec<f64>> = atm_dist.chunks_exact(natm).map(|r| r.to_vec()).collect_vec();

        // compute derivative of atm_dist if deriv >= 1
        let dR_atm_dist: Option<Vec<Vec<[f64; 3]>>> = (deriv >= 1).then(|| {
            let flat: Vec<[f64; 3]> = (0..natm * natm)
                .into_par_iter()
                .map(|idx| {
                    let (A, B) = (idx / natm, idx % natm);
                    (0..3).map(|t| (atm_coords[A][t] - atm_coords[B][t]) / atm_dist[A][B]).collect_array().unwrap()
                })
                .collect();
            flat.chunks_exact(natm).map(|r| r.to_vec()).collect_vec()
        });

        Self { atm_coords: atm_coords.to_vec(), adjustment_factor: adjustment_factor.to_vec(), atm_dist, dR_atm_dist }
    }
}

/// Generate the row-major adjustment-factor matrix `a_ab` from Bragg atomic radii,
/// the input of [`BeckeMolTables::new`] / [`becke_partition`].
///
/// For each atom pair `(A, B)` with distinct radii (Becke 1988 eq. (A5)),
/// `u_ab = (r_a + r_b) / (r_b - r_a)` and `a_ab = u_ab / (u_ab^2 - 1)`, clamped to
/// `[-0.5, 0.5]`; pairs of equal radii (within [`parameters::SMALL`]) get 0, so that
/// `nu_ab = mu_ab`.  The matrix is anti-symmetric with a zero diagonal, the same
/// convention the switch-function passes read `adjustment_factor[A][B]` for `B < A`.
pub fn gen_adjustment_factor(proton_charges: &[i32]) -> Vec<Vec<f64>> {
    let natm = proton_charges.len();
    let radii: Vec<f64> = proton_charges.iter().map(|&c| bragg::get_bragg_angstrom(c)).collect();
    let mut adjustment_factor = vec![vec![0.0; natm]; natm];
    for A in 0..natm {
        for B in 0..A {
            if (radii[A] - radii[B]).abs() > parameters::SMALL {
                let u_ab = (radii[A] + radii[B]) / (radii[B] - radii[A]);
                let a_ab = u_ab / (u_ab * u_ab - 1.0);
                adjustment_factor[A][B] = a_ab.min(0.5).max(-0.5);
                adjustment_factor[B][A] = -adjustment_factor[A][B];
            }
        }
    }
    adjustment_factor
}

/// Becke partitioning implementation for DFT numerical integration, preparing the
/// molecular tables ([`BeckeMolTables`]) on the fly.  For repeated calls on the same
/// molecule (e.g. chunk-level drivers), build the tables once and use
/// [`becke_partition_with_tables`] instead.
///
/// # Arguments
///
/// - `grid_coords`: A slice of 3D coordinates representing the grid points. Length `ngrids`.
/// - `atm_coords`: A slice of 3D coordinates representing the atomic positions. Length `natm`.
/// - `atm_indices`: Attribution of each grid point to its generating atom, see [`AtmIndices`]:
///   - [`AtmIndices::ByGrid`]: length `ngrids`; usually values should not exceed `natm - 1`, but
///     will treat as padding atom if exceeds.
///   - [`AtmIndices::ByAtom`]: length `natm + 1`; atom `A` owns the grid interval `[indices[A],
///     indices[A + 1])`.
/// - `quadrature_weights`: A slice of original quadrature weights (generated by Lebedev angular
///   quadrature) for each grid point. Length `ngrids`.
/// - `adjustment_factor`: Adjustment factors (usually adjusted by atomic radii), as `natm`
///   row-major rows of length `natm`: `adjustment_factor[A][B]` reads entry `(A, B)`. cf Becke 1988
///   eqs (A2, A5).
///   - **Note**: this pairwise matrix is anti-symmetric. If your table is flattened in column-major
///     order, transpose it to rows first.
/// - `hardness`: Cutoff hardness of screen function. cf Becke 1988 eq (20) and FIG 1. Most commonly
///   used value is 3, and we have manually dispatched the case of 3 in code implementation.
/// - `nbatch`: Batch size for parallel processing. Must be a multiple of `SIMDD` (8 for AVX-512).
/// - `deriv`: Derivative order.
/// - `deriv_arg`: Optional arguments for derivative output and contraction. If None, default values
///   will be used, which outputs all weights and derivatives without contraction.
///
/// # Reference
///
/// A multicenter numerical integration scheme for polyatomic molecules
/// A. D. Becke
/// J. Chem. Phys. 88, 2547 (1988), doi: 10.1063/1.454033
#[allow(clippy::too_many_arguments)]
pub fn becke_partition<'a>(
    grid_coords: &[[f64; 3]],
    atm_coords: &[[f64; 3]],
    atm_indices: AtmIndices<'a>,
    quadrature_weights: &[f64],
    adjustment_factor: &[Vec<f64>],
    hardness: usize,
    nbatch: usize,
    deriv: usize,
    deriv_arg: Option<BeckePartitionArg<'a>>,
) -> BeckePartitionOutput {
    let tables = BeckeMolTables::new(atm_coords, adjustment_factor, deriv);
    becke_partition_with_tables(
        &tables,
        grid_coords,
        atm_indices,
        quadrature_weights,
        hardness,
        nbatch,
        deriv,
        deriv_arg,
    )
}

/// [`becke_partition`] on precomputed molecular tables, skipping the per-call
/// molecular precomputation (see [`BeckeMolTables`]).
#[allow(clippy::too_many_arguments)]
pub fn becke_partition_with_tables<'a>(
    tables: &BeckeMolTables,
    grid_coords: &[[f64; 3]],
    atm_indices: AtmIndices<'a>,
    quadrature_weights: &[f64],
    hardness: usize,
    nbatch: usize,
    deriv: usize,
    deriv_arg: Option<BeckePartitionArg<'a>>,
) -> BeckePartitionOutput {
    let ctx = BeckePartitionContext::new(
        tables,
        grid_coords,
        atm_indices,
        quadrature_weights,
        hardness,
        nbatch,
        deriv,
        deriv_arg,
    );

    // split the grid into parallel tasks.  ByGrid: uniform chunks of `nbatch`
    // grids (attribution may vary within a chunk).  ByAtom: chunks never cross
    // an atom's grid interval, so each task carries one definite generating
    // atom.
    let tasks: Vec<BatchTask<'_>> = match ctx.atm_indices {
        AtmIndices::ByGrid(attribution) => (0..ctx.ngrids.div_ceil(nbatch))
            .map(|itask| {
                let g0 = itask * nbatch;
                BatchTask::ByGrid { attribution, g0, g1: (g0 + nbatch).min(ctx.ngrids) }
            })
            .collect(),
        AtmIndices::ByAtom(split) => {
            let mut tasks = Vec::new();
            for A in 0..ctx.natm {
                let mut g0 = split[A];
                let end = split[A + 1];
                while g0 < end {
                    let g1 = (g0 + nbatch).min(end);
                    tasks.push(BatchTask::ByAtom { atm: A, g0, g1 });
                    g0 = g1;
                }
            }
            tasks
        },
    };

    // par-iter over grid coordinates in batches.  Each task accumulates its
    // contraction partials privately (see [`TaskBuffers`]) and takes this lock
    // once for the reduction; w/dw/ddw write disjoint grid ranges and need no
    // lock.  `for_each_init` allocates the task buffers (including the large
    // per-lane scratch) once per worker thread, not once per task.
    let contract_guard = Mutex::new(());
    tasks.into_par_iter().for_each_init(
        || TaskBuffers::new(&ctx),
        |buffers, task| {
            let (g0, g1) = task.range();
            let lanes = gather_lane_batch(grid_coords, quadrature_weights, &task, ctx.natm);
            for ilane in 0..lanes.coords.len() {
                process_lane(&ctx, &lanes, ilane, buffers, g0, g1);
            }
            // reduce() also resets the contraction partials, since the worker's
            // next task reuses the same buffers
            buffers.contraction.reduce(&ctx, &contract_guard);
        },
    );

    ctx.into_output()
}

/// Per-call context of [`becke_partition_with_tables`]: validated dimensions
/// and dispatch flags, the borrowed molecular tables, the contraction sets
/// (per-set grid slices), and the output buffers.
struct BeckePartitionContext<'a> {
    // dimensions and dispatch flags
    ngrids: usize,
    natm: usize,
    hardness: usize,
    deriv: usize,
    /// whether the deriv2 machinery runs at all: `deriv >= 2` AND some 2nd-order output (the full
    /// `ddw` tensor or its contraction `ddc`) is actually requested.  When false the entire deriv2
    /// machinery is skipped even if `deriv == 2`, since neither `ddw` nor `ddc` would be consumed.
    do_deriv2: bool,
    // molecular data (precomputed, shared across chunk-level calls)
    tables: &'a BeckeMolTables,
    /// grid attribution scheme (see [`AtmIndices`]).
    atm_indices: AtmIndices<'a>,
    // contraction sets (per-set grid slices, shape `(nset, ngrids)` row-major)
    contract_w: Option<Vec<&'a [f64]>>,
    contract_dw: Option<Vec<&'a [f64]>>,
    contract_ddw: Option<Vec<&'a [f64]>>,
    nset_w: Option<usize>,
    nset_dw: Option<usize>,
    nset_ddw: Option<usize>,
    // output buffers
    output: BeckePartitionOutput,
}

impl<'a> BeckePartitionContext<'a> {
    /// Validate arguments, allocate output buffers, and split contraction sets.  The
    /// molecular data is borrowed from the precomputed [`BeckeMolTables`].
    #[allow(clippy::too_many_arguments)]
    fn new(
        tables: &'a BeckeMolTables,
        grid_coords: &[[f64; 3]],
        atm_indices: AtmIndices<'a>,
        quadrature_weights: &[f64],
        hardness: usize,
        nbatch: usize,
        deriv: usize,
        deriv_arg: Option<BeckePartitionArg<'a>>,
    ) -> Self {
        // dimensions
        let ngrids = grid_coords.len();
        let natm = tables.atm_coords.len();
        assert!(nbatch % SIMDD == 0, "nbatch must be a multiple of {SIMDD}");
        assert!(quadrature_weights.len() == ngrids, "quadrature_weights must have length ngrids");
        assert!(
            deriv < 1 || tables.dR_atm_dist.is_some(),
            "BeckeMolTables was built with deriv 0, but deriv >= 1 is requested"
        );
        match atm_indices {
            AtmIndices::ByGrid(v) => {
                assert!(v.len() == ngrids, "ByGrid atm_indices must have length ngrids");
            },
            AtmIndices::ByAtom(v) => {
                assert!(v.len() == natm + 1, "ByAtom atm_indices must have length natm + 1");
                assert!(v[0] == 0, "ByAtom atm_indices must start with 0");
                assert!(v[natm] == ngrids, "ByAtom atm_indices must end with ngrids");
                // intervals must stay within [0, ngrids] and be non-decreasing,
                // hence cover [0, ngrids) without gaps
                assert!(v.iter().all(|&x| x <= ngrids), "ByAtom atm_indices must not exceed ngrids");
                assert!(v.windows(2).all(|w| w[0] <= w[1]), "ByAtom atm_indices must be non-decreasing");
            },
        }

        let deriv_arg = deriv_arg.unwrap_or_default();
        assert!(deriv <= 2, "deriv must be 0, 1, or 2 at current time");

        // check if contraction is requested, and split the contraction weights
        // into per-set grid slices (shape `(nset, ngrids)` row-major)
        let contract_w = deriv_arg.contract_w.as_ref().map(|c| {
            assert!(c.len() % ngrids == 0, "contract_w length must be a multiple of ngrids");
            c.chunks_exact(ngrids).collect_vec()
        });
        let contract_dw = deriv_arg.contract_dw.as_ref().map(|c| {
            assert!(c.len() % ngrids == 0, "contract_dw length must be a multiple of ngrids");
            c.chunks_exact(ngrids).collect_vec()
        });
        let contract_ddw = deriv_arg.contract_ddw.as_ref().map(|c| {
            assert!(c.len() % ngrids == 0, "contract_ddw length must be a multiple of ngrids");
            c.chunks_exact(ngrids).collect_vec()
        });
        let nset_w = contract_w.as_ref().map(|c| c.len());
        let nset_dw = (deriv >= 1 && contract_dw.is_some()).then(|| contract_dw.as_ref().unwrap().len());
        let nset_ddw = (deriv >= 2 && contract_ddw.is_some()).then(|| contract_ddw.as_ref().unwrap().len());
        // whether any 2nd-order output (the full `ddw` tensor or its
        // contraction) is actually requested
        let need_ddw = deriv_arg.output_ddw || contract_ddw.is_some();

        let output = BeckePartitionOutput {
            w: deriv_arg.output_w.then(|| vec![0.0; ngrids]),
            dw: (deriv_arg.output_dw && deriv >= 1).then(|| vec![0.0; natm * 3 * ngrids]),
            ddw: (deriv_arg.output_ddw && deriv >= 2).then(|| vec![0.0; natm * 3 * natm * 3 * ngrids]),
            c: nset_w.map(|nset| vec![0.0; nset]),
            dc: nset_dw.map(|nset| vec![0.0; natm * 3 * nset]),
            ddc: nset_ddw.map(|nset| vec![0.0; natm * 3 * natm * 3 * nset]),
        };

        Self {
            ngrids,
            natm,
            hardness,
            deriv,
            do_deriv2: deriv >= 2 && need_ddw,
            tables,
            atm_indices,
            contract_w,
            contract_dw,
            contract_ddw,
            nset_w,
            nset_dw,
            nset_ddw,
            output,
        }
    }

    fn into_output(self) -> BeckePartitionOutput {
        self.output
    }
}

/* #endregion */

/* #region Task*: parallel-thread interface (batch tasks, per-worker buffers) */

/// One parallel task: a grid range `[g0, g1)` plus its attribution to generating atoms.
enum BatchTask<'a> {
    /// ByGrid scheme: per-grid attribution read from the full `ngrids`-length slice.
    ByGrid { attribution: &'a [usize], g0: usize, g1: usize },
    /// ByAtom scheme: the whole range lies inside atom `atm`'s grid interval.
    ByAtom { atm: usize, g0: usize, g1: usize },
}

impl BatchTask<'_> {
    fn range(&self) -> (usize, usize) {
        match self {
            BatchTask::ByGrid { g0, g1, .. } | BatchTask::ByAtom { g0, g1, .. } => (*g0, *g1),
        }
    }
}

/// Contraction partials of one task.
///
/// `c`/`dc`/`ddc` hold this task's contraction partials (sums over the task's
/// grid range), reduced into the shared output by [`Self::reduce`] under the
/// task mutex — this is what lets the caller obtain a contracted `ddc` without
/// materializing the full `O(natm^2 * ngrids)` `ddw` tensor.
///
/// `cw_lanes_dw`/`cw_lanes_ddw` are per-lane scratch (one SIMD register per
/// contraction set), refilled each lane so the dc/ddc contraction allocates
/// nothing inside the lane loop.
struct TaskContraction {
    /// `[iset]`: contraction partial of `w`.
    c: Option<Vec<f64>>,
    /// flat `[A, t, iset]`: contraction partial of `dc`.
    dc: Option<Vec<f64>>,
    /// flat `[A, t, B, s, iset]`: contraction partial of `ddc`.
    ddc: Option<Vec<f64>>,
    /// `[iset]`: this lane's `contract_dw` weights.
    cw_lanes_dw: Vec<f64simd>,
    /// `[iset]`: this lane's `contract_ddw` weights.
    cw_lanes_ddw: Vec<f64simd>,
}

impl TaskContraction {
    fn new(ctx: &BeckePartitionContext<'_>) -> Self {
        let natm = ctx.natm;
        Self {
            c: ctx.nset_w.map(|n| vec![0.0; n]),
            dc: ctx.nset_dw.map(|n| vec![0.0; natm * 3 * n]),
            ddc: ctx.nset_ddw.map(|n| vec![0.0; natm * 3 * natm * 3 * n]),
            cw_lanes_dw: ctx.nset_dw.map_or_else(Vec::new, |n| vec![simd_val(0.0); n]),
            cw_lanes_ddw: ctx.nset_ddw.map_or_else(Vec::new, |n| vec![simd_val(0.0); n]),
        }
    }

    /// Add this task's contraction partials into the shared output buffers,
    /// under the task mutex (c/dc/ddc are grid sums written by every task), and
    /// reset them: the buffers are owned by the worker thread (see
    /// `for_each_init`) and reused by its next task, so the partials must not
    /// survive past the reduction.
    fn reduce(&mut self, ctx: &BeckePartitionContext<'_>, guard: &Mutex<()>) {
        if self.c.is_none() && self.dc.is_none() && self.ddc.is_none() {
            return;
        }
        let _guard = guard.lock().unwrap();
        for (buf, partial) in
            [(&ctx.output.c, &mut self.c), (&ctx.output.dc, &mut self.dc), (&ctx.output.ddc, &mut self.ddc)]
        {
            if let (Some(buf), Some(partial)) = (buf, partial) {
                // SAFETY: exclusive access granted by the task mutex
                let slc = unsafe { cast_mut_slice(buf) };
                for (o, v) in slc.iter_mut().zip(partial.iter_mut()) {
                    *o += *v;
                    *v = 0.0;
                }
            }
        }
    }
}

/// Per-worker-thread buffers: the task's contraction partials plus the
/// flattened per-lane scratch (see [`LaneScratch`]), allocated once by
/// `for_each_init` and reused across all tasks handled by the worker.
struct TaskBuffers {
    contraction: TaskContraction,
    lane: LaneScratch,
}

impl TaskBuffers {
    fn new(ctx: &BeckePartitionContext<'_>) -> Self {
        Self { contraction: TaskContraction::new(ctx), lane: LaneScratch::new(ctx.natm, ctx.deriv, ctx.do_deriv2) }
    }
}

/* #endregion */

/* #region Lane*: grid-iteration interface (gather, scratch, evaluation, write-back) */

// --- gather: batch -> SIMD lanes --- //

/// One task's grid batch gathered into SIMD lanes.
struct LaneBatch {
    /// shape `[nlane][t]`; padding lanes (past the batch end) are zero-filled.
    coords: Vec<[f64simd; 3]>,
    /// shape `[nlane]`; padding lanes zero-filled.
    wquad: Vec<f64simd>,
    /// attribution of the batch's lanes to their generating atoms.
    attr: LaneAttribution,
}

/// Attribution of a batch's lanes to their generating atoms.
enum LaneAttribution {
    /// per-lane generating atom (ByGrid); padding lanes carry `natm`
    ByGrid(Vec<[usize; SIMDD]>),
    /// single generating atom for every lane of the batch (ByAtom)
    ByAtom(usize),
}

/// Per-lane view of [`LaneAttribution`], passed into the evaluation functions.
#[derive(Clone, Copy)]
enum LaneAttrib {
    ByGrid([usize; SIMDD]),
    ByAtom(usize),
}

impl LaneAttribution {
    fn lane(&self, ilane: usize) -> LaneAttrib {
        match self {
            LaneAttribution::ByGrid(v) => LaneAttrib::ByGrid(v[ilane]),
            LaneAttribution::ByAtom(a) => LaneAttrib::ByAtom(*a),
        }
    }
}

impl LaneAttrib {
    /// Per-lane select of `value` on lanes whose generating atom is `A`, `fallback` on the
    /// others: the lane-wise mask (ByGrid) or definite-atom (ByAtom) selection.  Used inside
    /// the pair loop where `A`/`B` iterate over atoms.
    #[inline]
    fn select(self, A: usize, value: f64simd, fallback: f64simd) -> f64simd {
        match self {
            LaneAttrib::ByGrid(atm_idx) => value.mask_select(atm_idx.map(|a| a == A), fallback),
            LaneAttrib::ByAtom(atm_g) => {
                if A == atm_g {
                    value
                } else {
                    fallback
                }
            },
        }
    }
}

/// Gather one task's grid batch into SIMD lanes.
///
/// A ByGrid task reads the per-grid indices, marking lanes past the batch end
/// with the padding atom index `natm` (out of range, selects no partition
/// weight).  A ByAtom task attributes every lane to its single atom; its
/// padding lanes keep the zero-filled coordinates/quadrature weights and their
/// lane values are never read back.
fn gather_lane_batch(
    grid_coords: &[[f64; 3]],
    quadrature_weights: &[f64],
    task: &BatchTask<'_>,
    natm: usize,
) -> LaneBatch {
    let (g0, g1) = task.range();
    let nlane = (g1 - g0).div_ceil(SIMDD);
    let mut coords = vec![[simd_val(0.0); 3]; nlane];
    let mut wquad = vec![simd_val(0.0); nlane];

    match task {
        BatchTask::ByGrid { attribution, .. } => {
            let mut atm_idx = vec![[0; SIMDD]; nlane];
            for lane in 0..nlane {
                let glane = g0 + lane * SIMDD;
                for g in 0..SIMDD {
                    if glane + g < g1 {
                        for t in 0..3 {
                            coords[lane][t][g] = grid_coords[glane + g][t];
                        }
                        wquad[lane][g] = quadrature_weights[glane + g];
                        atm_idx[lane][g] = attribution[glane + g];
                    } else {
                        // padding atom index set to natm (out of range)
                        atm_idx[lane][g] = natm;
                    }
                }
            }
            LaneBatch { coords, wquad, attr: LaneAttribution::ByGrid(atm_idx) }
        },
        BatchTask::ByAtom { atm, .. } => {
            for lane in 0..nlane {
                let glane = g0 + lane * SIMDD;
                for g in 0..SIMDD {
                    if glane + g < g1 {
                        for t in 0..3 {
                            coords[lane][t][g] = grid_coords[glane + g][t];
                        }
                        wquad[lane][g] = quadrature_weights[glane + g];
                    }
                }
            }
            LaneBatch { coords, wquad, attr: LaneAttribution::ByAtom(*atm) }
        },
    }
}

// --- per-lane scratch and orchestration --- //

/// Flattened per-lane scratch, reused across lanes and tasks.
///
/// Field docs give logical dimensions (`[A, t]`, `[B, t, s]`, ...); every
/// buffer is one flat `Vec<f64simd>` in row-major order of the listed
/// indices, so `[A, t]` reads at `A * 3 + t` and `[A, t, B, s]` at
/// `(A * 3 + t) * (3 * natm) + (B * 3 + s)`.  The evaluation functions
/// abbreviate the flattened (atom, cartesian) row/column pair as
/// `r = 3A + t` / `c = 3B + s`.  Only the pair-accumulated diagonal
/// entries/blocks are re-zeroed per lane; every other element is fully
/// assigned/overwritten by its unique writer (see [`eval_switch_pair_pass`]
/// and [`eval_lane_ddw`]).
///
/// Each buffer is allocated only for the machinery that actually runs: the
/// 1st-order row buffers when `deriv >= 1`, the deriv-2 tensors when the
/// context's `do_deriv2` is set; the rest stay empty and are never touched
/// (their readers are gated on the same flags in `process_lane`).
struct LaneScratch {
    /// `[A]`: switch-product numerators `P_A` (overwritten, initialized to 1).
    P: Vec<f64simd>,
    /// `[A]`: grid-atom distances (overwritten).
    dist: Vec<f64simd>,
    /// `[A, t]`: `d|A - g| / dR_A[t]` unit vectors (overwritten).
    dR_dist: Vec<f64simd>,
    /// `[M, t, s]`: projection matrices `Proj(r_M)/|r_M|` (overwritten; deriv 2 only).
    PrM: Vec<f64simd>,
    /// `[A, t]`: `dZ / dR_A[t]` (pair-accumulated, re-zeroed).
    dR_Z: Vec<f64simd>,
    /// `[A, t]`: `dPg / dR_A[t]` (pair-accumulated, re-zeroed).
    dR_Pg: Vec<f64simd>,
    /// `[M, C, t]`: `dlog P_M / dR_C[t]` (deriv 2 only; off-diagonal `C` rows
    /// assigned by their unique pair, diagonal re-zeroed).
    dR_log_P: Vec<f64simd>,
    /// `[A, t, B, s]`: `d2Z / (dR_A[t] dR_B[s])` (deriv 2 only; off-diagonal
    /// entries assigned by their unique pair, diagonal entries pair-accumulated
    /// and re-zeroed).
    ddR_Z: Vec<f64simd>,
    /// same layout as `ddR_Z` for `d2Pg` (deriv 2 only).
    ddR_Pg: Vec<f64simd>,
    /// `[A, t]`: quotient-rule 1st derivative of `q` (overwritten; deriv 2 only).
    dq: Vec<f64simd>,
    /// `[A, t]`: gathered `dlog P_{A_g} / dR_A[t]` row (overwritten; deriv 2 only).
    dlog_Ag: Vec<f64simd>,
    /// `[A, t, B, s]`: the finalized `ddw` (fully overwritten; deriv 2 only).
    ddw: Vec<f64simd>,
    /// `[B, t, s]`: `sum_A ddw[A, t, B, s]` (accumulated, re-zeroed; deriv 2 only).
    fullA: Vec<f64simd>,
    /// `[A, t, s]`: `sum_B ddw[A, t, B, s]` (per-row assignment; deriv 2 only).
    fullB: Vec<f64simd>,
    /// `[A, t]`: the translation-invariance-fixed `dw` (overwritten).
    dw: Vec<f64simd>,
}

impl LaneScratch {
    fn new(natm: usize, deriv: usize, do_deriv2: bool) -> Self {
        let do_deriv1 = deriv >= 1;
        // `n` zeroed registers when the machinery flagged by `on` runs, an empty
        // (allocation-free) vector otherwise
        let z = |n: usize, on: bool| if on { vec![simd_val(0.0); n] } else { Vec::new() };
        Self {
            P: z(natm, true),
            dist: z(natm, true),
            dR_dist: z(3 * natm, do_deriv1),
            PrM: z(9 * natm, do_deriv2),
            dR_Z: z(3 * natm, do_deriv1),
            dR_Pg: z(3 * natm, do_deriv1),
            dR_log_P: z(natm * 3 * natm, do_deriv2),
            ddR_Z: z(natm * natm * 9, do_deriv2),
            ddR_Pg: z(natm * natm * 9, do_deriv2),
            dq: z(3 * natm, do_deriv2),
            dlog_Ag: z(3 * natm, do_deriv2),
            ddw: z(3 * natm * 3 * natm, do_deriv2),
            fullA: z(9 * natm, do_deriv2),
            fullB: z(9 * natm, do_deriv2),
            dw: z(3 * natm, do_deriv1),
        }
    }
}

/// Evaluate one SIMD lane at all requested derivative levels, writing the results into the output
/// buffers and accumulating the contraction partials.
fn process_lane(
    ctx: &BeckePartitionContext<'_>,
    lanes: &LaneBatch,
    ilane: usize,
    buffers: &mut TaskBuffers,
    g0: usize,
    g1: usize,
) {
    let coords = &lanes.coords[ilane];
    let wquad = lanes.wquad[ilane];
    let attr = lanes.attr.lane(ilane);

    let g_start = g0 + ilane * SIMDD;
    let g_end = (g_start + SIMDD).min(g1);

    // split the per-worker buffers into the (disjoint) contraction partials and
    // lane scratch so the stores can borrow one while the lane eval holds the other
    let TaskBuffers { contraction, lane } = buffers;

    // --- deriv 0 --- //

    let part = eval_partition(ctx, coords, wquad, attr, lane);
    store_lane_w(ctx, contraction, part.w, g_start, g_end);

    // --- deriv 1 --- //

    if ctx.deriv >= 1 {
        eval_switch_pair_pass(ctx, coords, attr, lane);
        eval_lane_dw(ctx, wquad, attr, &part, lane);
        store_lane_dw(ctx, contraction, &lane.dw, g_start, g_end);

        // --- deriv 2 --- //

        if ctx.do_deriv2 {
            eval_lane_ddw(ctx, wquad, attr, &part, lane);
            store_lane_ddw(ctx, contraction, &lane.ddw, g_start, g_end);
        }
    }
}

// --- per-lane evaluation --- //

/// Deriv-0 lane intermediates from [`eval_partition`], also consumed by the deriv-1/2
/// passes.  The per-atom arrays `P`/`dist` live in the lane scratch (see
/// [`LaneScratch`]); only the lane registers are returned.
struct LanePartition {
    /// partition weight numerator selected by the generating atom, and the normalizing sum.
    Pg: f64simd,
    Z: f64simd,
    /// quadrature-weighted partition weight `wquad * Pg / Z`.
    w: f64simd,
}

/// Deriv-0 lane evaluation: grid-atom distances, 1st pass of the switch function (without
/// derivative), and the normalized partition weight.
fn eval_partition(
    ctx: &BeckePartitionContext<'_>,
    coords: &[f64simd; 3],
    wquad: f64simd,
    attr: LaneAttrib,
    scr: &mut LaneScratch,
) -> LanePartition {
    let natm = ctx.natm;

    // evaluate grid distance to atom
    for A in 0..natm {
        scr.dist[A] = dist3_hybrid(coords, &ctx.tables.atm_coords[A]);
    }

    // 1st pass of switch function (without derivative)
    scr.P.fill(simd_val(1.0));
    for A in 0..natm {
        for B in 0..A {
            let a_factor = ctx.tables.adjustment_factor[A][B];
            let mu = (scr.dist[A] - scr.dist[B]) / ctx.tables.atm_dist[A][B];
            let f3 = match ctx.hardness {
                3 => switch_f3(mu, a_factor),
                _ => switch_f_hardness(mu, a_factor, ctx.hardness),
            };
            scr.P[A] *= simd_val(0.5) * (simd_val(1.0) - f3);
            scr.P[B] *= simd_val(0.5) * (simd_val(1.0) + f3);
        }
    }

    // compute partition function and weights
    let mut Z = simd_val(0.0);
    for A in 0..natm {
        Z += scr.P[A];
    }
    // partition numerator: the generating atom's P (a definite atom for ByAtom; a
    // per-lane gather over the per-grid indices for ByGrid)
    let Pg = match attr {
        // one scalar assignment per lane (padding lanes keep 0) instead of natm
        // mask_select passes over the whole register
        LaneAttrib::ByGrid(atm_idx) => {
            let mut Pg = simd_val(0.0);
            for g in 0..SIMDD {
                let A = atm_idx[g];
                if A < natm {
                    Pg[g] = scr.P[A][g];
                }
            }
            Pg
        },
        LaneAttrib::ByAtom(atm_g) => scr.P[atm_g],
    };
    let partition = Pg / Z;
    let w = wquad * partition;

    LanePartition { Pg, Z, w }
}

/// 2nd pass of the switch function (with 1st derivative), over all atom pairs,
/// accumulating the flattened derivative tensors of [`LaneScratch`].
///
/// Variable `P` is required to be generated in the 1st pass (see [`eval_partition`]), so two passes
/// cannot merge for first derivative.  For deriv 2, the per-pair 2nd-order (L2) contributions are
/// accumulated in the same pair loop.
///
/// Zero-init discipline: the `[M][M]`/`[A][A]`/`[B][B]` diagonal entries/blocks of
/// `dR_log_P`/`ddR_Z`/`ddR_Pg` receive `+=` contributions from several pairs and are
/// re-zeroed per lane (O(natm) elements); every off-diagonal entry/block is touched
/// by exactly one pair `(A, B)` and is plain-assigned, so the large O(natm^2)
/// buffers never need a zero fill.
fn eval_switch_pair_pass(
    ctx: &BeckePartitionContext<'_>,
    coords: &[f64simd; 3],
    attr: LaneAttrib,
    scr: &mut LaneScratch,
) {
    let natm = ctx.natm;
    let n3 = 3 * natm;
    let do_deriv2 = ctx.do_deriv2;
    let dR_atm_dist = ctx.tables.dR_atm_dist.as_ref().unwrap();

    // evaluate derivative of grid distance to atom (flat `[A, t]`)
    for A in 0..natm {
        for t in 0..3 {
            scr.dR_dist[A * 3 + t] = (-coords[t] + ctx.tables.atm_coords[A][t]) / scr.dist[A];
        }
    }

    // 1st-order accumulators start from zero each lane
    scr.dR_Z.fill(simd_val(0.0));
    scr.dR_Pg.fill(simd_val(0.0));

    // 2nd-order intermediates (only materialized for deriv >= 2): re-zero the
    // pair-accumulated diagonals, and precompute the per-atom projection
    // matrices PrM[M] = Proj(r_M)/|r_M| flat as `[M, t, s]` (they depend
    // only on the atom, not the pair partner).
    if do_deriv2 {
        for M in 0..natm {
            let c0 = M * n3 + 3 * M;
            scr.dR_log_P[c0..c0 + 3].fill(simd_val(0.0));
            for t in 0..3 {
                let r0 = (M * 3 + t) * n3 + M * 3;
                scr.ddR_Z[r0..r0 + 3].fill(simd_val(0.0));
                scr.ddR_Pg[r0..r0 + 3].fill(simd_val(0.0));
            }
            let inv_d = simd_val(1.0) / scr.dist[M];
            for t in 0..3 {
                for s in 0..3 {
                    let delta = if t == s { simd_val(1.0) } else { simd_val(0.0) };
                    scr.PrM[M * 9 + t * 3 + s] = (delta - scr.dR_dist[M * 3 + t] * scr.dR_dist[M * 3 + s]) * inv_d;
                }
            }
        }
    }

    let P = &scr.P;
    let dist = &scr.dist;

    for A in 0..natm {
        for B in 0..A {
            let a_factor = ctx.tables.adjustment_factor[A][B];
            let inv_atm_dist_AB = 1.0 / ctx.tables.atm_dist[A][B];
            let mu = (dist[A] - dist[B]) * inv_atm_dist_AB;
            // switch value + 1st nu-deriv always; the 2nd nu-deriv (f3pp) is only
            // needed for deriv >= 2, so the deriv == 1 path uses the cheaper
            // 1st-order-only switch and avoids computing f3''.
            let (f3, df3, ddf3): (f64simd, f64simd, Option<f64simd>) = if do_deriv2 {
                let (f3, df3, ddf3) = match ctx.hardness {
                    3 => switch_d2nu_f3(mu, a_factor),
                    _ => switch_d2nu_f_hardness(mu, a_factor, ctx.hardness),
                };
                (f3, df3, Some(ddf3))
            } else {
                let (f3, df3) = match ctx.hardness {
                    3 => switch_dnu_f3(mu, a_factor),
                    _ => switch_dnu_f_hardness(mu, a_factor, ctx.hardness),
                };
                (f3, df3, None)
            };
            let sA = simd_val(0.5) * (simd_val(1.0) - f3);
            let sB = simd_val(0.5) * (simd_val(1.0) + f3);
            let dmu_nu = simd_val(1.0) - simd_val(2.0) * mu * a_factor;
            let dmu_sA = -simd_val(0.5) * df3 * dmu_nu;
            let dmu_sB = simd_val(0.5) * df3 * dmu_nu;
            let sA_safe = sA.max_compare(INVTOL);
            let sB_safe = sB.max_compare(INVTOL);
            let dmu_log_sA = dmu_sA / sA_safe;
            let dmu_log_sB = dmu_sB / sB_safe;

            let common_Z = P[A] * dmu_log_sA + P[B] * dmu_log_sB;
            // only the pair member that generated the grid point contributes to the Pg
            // numerator (lane-wise mask for ByGrid; definite atom for ByAtom)
            let common_Pg =
                attr.select(A, P[A] * dmu_log_sA, simd_val(0.0)) + attr.select(B, P[B] * dmu_log_sB, simd_val(0.0));

            let mut dR_mu_roleA = [simd_val(0.0); 3];
            let mut dR_mu_roleB = [simd_val(0.0); 3];
            let dR_atm_dist_AB = dR_atm_dist[A][B];
            let (ia, ib) = (A * 3, B * 3);
            for t in 0..3 {
                dR_mu_roleA[t] = (scr.dR_dist[ia + t] - mu * dR_atm_dist_AB[t]) * inv_atm_dist_AB;
                dR_mu_roleB[t] = (-scr.dR_dist[ib + t] + mu * dR_atm_dist_AB[t]) * inv_atm_dist_AB;
                scr.dR_Z[ia + t] += common_Z * dR_mu_roleA[t];
                scr.dR_Z[ib + t] += common_Z * dR_mu_roleB[t];
                scr.dR_Pg[ia + t] += common_Pg * dR_mu_roleA[t];
                scr.dR_Pg[ib + t] += common_Pg * dR_mu_roleB[t];
            }

            // --- deriv 2 (per-pair L2 accumulation) --- //

            if do_deriv2 {
                let ddf3 = ddf3.unwrap();

                // 2nd mu-derivatives of s(mu); ddmu_sB = -ddmu_sA (s_BA = 1 - s_AB)
                let ddmu_nu = -simd_val(2.0) * a_factor; // nu'' = -2a
                let ddmu_sA = -simd_val(0.5) * ddf3 * dmu_nu * dmu_nu - simd_val(0.5) * df3 * ddmu_nu;
                let ddmu_sB = -ddmu_sA;
                // ddmu_log_s = s''/s - (s'/s)^2
                let ddmu_log_sA = ddmu_sA / sA_safe - dmu_log_sA * dmu_log_sA;
                let ddmu_log_sB = ddmu_sB / sB_safe - dmu_log_sB * dmu_log_sB;

                // 2nd role derivatives of mu_{AB} (3 blocks) via quotient rule
                //   d2(f/g) = [f_xy g - (f_x g_y + g_x f_y) - f g_xy]/g^2 + 2 f g_x g_y/g^3
                // rA, rB = unit vec (R_atom - r_g)/|r|; Uvec = unit vec (R_A - R_B)/|R_AB|;
                // PrA = Proj(rA)/|r_A|, PrB = Proj(rB)/|r_B|, PU = Proj(Uvec)/|R_AB|.
                let mut rA = [simd_val(0.0); 3];
                let mut rB = [simd_val(0.0); 3];
                let mut PrA = [[simd_val(0.0); 3]; 3];
                let mut PrB = [[simd_val(0.0); 3]; 3];
                for t in 0..3 {
                    rA[t] = scr.dR_dist[ia + t];
                    rB[t] = scr.dR_dist[ib + t];
                    for s in 0..3 {
                        PrA[t][s] = scr.PrM[A * 9 + t * 3 + s];
                        PrB[t][s] = scr.PrM[B * 9 + t * 3 + s];
                    }
                }
                let Uvec = [
                    f64simd::splat(dR_atm_dist_AB[0]),
                    f64simd::splat(dR_atm_dist_AB[1]),
                    f64simd::splat(dR_atm_dist_AB[2]),
                ];
                // PrA/PrB are per-atom (precomputed in PrM above); only PU is per-pair.
                let mut PU = [[simd_val(0.0); 3]; 3];
                for t in 0..3 {
                    for s in 0..3 {
                        let delta = if t == s { simd_val(1.0) } else { simd_val(0.0) };
                        PU[t][s] = (delta - Uvec[t] * Uvec[s]) * inv_atm_dist_AB;
                    }
                }
                let g_ab = ctx.tables.atm_dist[A][B];
                let f_ab = dist[A] - dist[B]; // f = |r_A| - |r_B| (= mu * g_ab)
                let inv_g2 = inv_atm_dist_AB * inv_atm_dist_AB;
                let inv_g3 = inv_g2 * inv_atm_dist_AB;
                let zero_ts = [[simd_val(0.0); 3]; 3];
                let nrB = neg3(rB);
                let nUv = neg3(Uvec);
                let nPrB = neg33(PrB);
                let nPU = neg33(PU);
                // d2mu(fX, fY, fXY, gX, gY, gXY) -> [[f64simd;3];3] over (t, s)
                let d2mu = |fX: &[f64simd; 3],
                            fY: &[f64simd; 3],
                            fXY: &[[f64simd; 3]; 3],
                            gX: &[f64simd; 3],
                            gY: &[f64simd; 3],
                            gXY: &[[f64simd; 3]; 3]|
                 -> [[f64simd; 3]; 3] {
                    let mut out = [[simd_val(0.0); 3]; 3];
                    for t in 0..3 {
                        for s in 0..3 {
                            let ofg = fX[t] * gY[s] + gX[t] * fY[s];
                            let ogg = gX[t] * gY[s];
                            out[t][s] = (fXY[t][s] * g_ab - ofg - f_ab * gXY[t][s]) * inv_g2
                                + simd_val(2.0) * f_ab * ogg * inv_g3;
                        }
                    }
                    out
                };
                let ddR_mu_roleAA = d2mu(&rA, &rA, &PrA, &Uvec, &Uvec, &PU);
                let ddR_mu_roleAB = d2mu(&rA, &nrB, &zero_ts, &Uvec, &nUv, &nPU);
                let ddR_mu_roleBB = d2mu(&nrB, &nrB, &nPrB, &nUv, &nUv, &PU);
                // role BA = role AB transposed in (t, s)
                let mut ddR_mu_roleBA = [[simd_val(0.0); 3]; 3];
                for t in 0..3 {
                    for s in 0..3 {
                        ddR_mu_roleBA[t][s] = ddR_mu_roleAB[s][t];
                    }
                }

                // accumulate dR_log_P (flat `[M, C, t]`); convention
                // dmu_log_sA = +nat[A,B], dmu_log_sB = -nat[B,A]; all contributions
                // below are plus.  Off-diagonal entries have exactly one writer
                // (this pair), so they are assigned, not accumulated.
                {
                    // rows M = A and M = B of dR_log_P (B < A, so row B lies in the
                    // part before row A; both slices are disjoint by construction)
                    let (head, tail) = scr.dR_log_P.split_at_mut(A * n3);
                    let rowA = &mut tail[..n3];
                    let rowB = &mut head[B * n3..(B + 1) * n3];
                    for t in 0..3 {
                        rowA[A * 3 + t] += dmu_log_sA * dR_mu_roleA[t]; // M=A, role A (diag)
                        rowA[B * 3 + t] = dmu_log_sA * dR_mu_roleB[t]; // M=A, role B (off)
                        rowB[A * 3 + t] = dmu_log_sB * dR_mu_roleA[t]; // M=B, role B (off)
                        rowB[B * 3 + t] += dmu_log_sB * dR_mu_roleB[t]; // M=B, role A (diag)
                    }
                }

                // L2 (2nd log-deriv) into ddR_Z and ddR_Pg (flat `[A, t, B, s]`).
                // The w*(d_a mu)(d_b mu) term uses the FIRST role derivatives
                // dR_mu_roleA/B (NOT the unit vectors rA/rB).  The 4 role outer
                // products are formed once per (t,s) and reused for both ddR_Z
                // and ddR_Pg.  Per (t, s) the four blocks write the rows 3A+t /
                // 3B+t at columns 3A+s / 3B+s; the row groups of A and B are
                // disjoint (B < A).
                let common_dd = P[A] * ddmu_log_sA + P[B] * ddmu_log_sB;
                let coef_A = attr.select(A, P[A], simd_val(0.0));
                let coef_B = attr.select(B, P[B], simd_val(0.0));
                let c1_Pg = coef_A * dmu_log_sA + coef_B * dmu_log_sB;
                let cdd_Pg = coef_A * ddmu_log_sA + coef_B * ddmu_log_sB;
                for (buf, coef1, coefdd) in [(&mut scr.ddR_Z, common_Z, common_dd), (&mut scr.ddR_Pg, c1_Pg, cdd_Pg)] {
                    let (head, tail) = buf.split_at_mut(A * 3 * n3);
                    let grpA = &mut tail[..3 * n3]; // rows 3A .. 3A+3
                    let grpB = &mut head[B * 3 * n3..(B * 3 + 3) * n3]; // rows 3B .. 3B+3
                    for t in 0..3 {
                        let rowA = &mut grpA[t * n3..(t + 1) * n3];
                        let rowB = &mut grpB[t * n3..(t + 1) * n3];
                        for s in 0..3 {
                            let ooAA = dR_mu_roleA[t] * dR_mu_roleA[s];
                            let ooAB = dR_mu_roleA[t] * dR_mu_roleB[s];
                            let ooBA = dR_mu_roleB[t] * dR_mu_roleA[s];
                            let ooBB = dR_mu_roleB[t] * dR_mu_roleB[s];
                            // diagonal entries accumulate, off-diagonal entries
                            // are written exactly once by this pair
                            rowA[A * 3 + s] += coefdd * ooAA + coef1 * ddR_mu_roleAA[t][s];
                            rowA[B * 3 + s] = coefdd * ooAB + coef1 * ddR_mu_roleAB[t][s];
                            rowB[A * 3 + s] = coefdd * ooBA + coef1 * ddR_mu_roleBA[t][s];
                            rowB[B * 3 + s] += coefdd * ooBB + coef1 * ddR_mu_roleBB[t][s];
                        }
                    }
                }
            }
        }
    }
}

/// Deriv-1 finalize: quotient rule for `dw` followed by the
/// translation-invariance fix, writing the flat `dw` row of [`LaneScratch`].
fn eval_lane_dw(
    ctx: &BeckePartitionContext<'_>,
    wquad: f64simd,
    attr: LaneAttrib,
    part: &LanePartition,
    scr: &mut LaneScratch,
) {
    let natm = ctx.natm;

    // fill derivatives (flat `[A, t]`)
    let inv_Z = simd_val(1.0) / part.Z;
    for A in 0..natm {
        for t in 0..3 {
            let r = A * 3 + t;
            scr.dw[r] = wquad * inv_Z * (scr.dR_Pg[r] - part.Pg * inv_Z * scr.dR_Z[r]);
        }
    }

    // apply translation invariance
    let mut dw_neg_sum = [simd_val(0.0); 3];
    match attr {
        // the gathered row entry of lane g is scr.dw[atm_g * 3 + t][g] itself, so the
        // fix reduces to a per-lane in-place addition (padding lanes skipped)
        LaneAttrib::ByGrid(atm_idx) => {
            for A in 0..natm {
                for t in 0..3 {
                    dw_neg_sum[t] -= scr.dw[A * 3 + t];
                }
            }
            for g in 0..SIMDD {
                let atm_g = atm_idx[g];
                if atm_g < natm {
                    for t in 0..3 {
                        scr.dw[atm_g * 3 + t][g] += dw_neg_sum[t][g];
                    }
                }
            }
        },
        LaneAttrib::ByAtom(atm_g) => {
            // the whole lane belongs to atm_g: the gathered row is simply
            // dw[atm_g] itself, and the per-lane atom check degenerates to one
            // uniform row update
            for A in 0..natm {
                for t in 0..3 {
                    dw_neg_sum[t] -= scr.dw[A * 3 + t];
                }
            }
            for t in 0..3 {
                scr.dw[atm_g * 3 + t] += dw_neg_sum[t];
            }
        },
    }
}

/// Deriv-2 finalize: cross terms, quotient rule, and translation-invariance
/// fix for `ddw`.
///
/// The cross terms need the fully accumulated `dR_log_P`, so they cannot be
/// folded into the pair loop.  The symmetric rank-1 form
/// `C[r][c] = sum_M P[M] dlog[M][r] dlog[M][c]` is evaluated first, tile by
/// tile over `CROSS_CB x CROSS_CB` blocks with the `M` loop innermost: each
/// tile keeps a `CROSS_CB x CROSS_CB` accumulator (L1-resident) and streams
/// only the two `dR_log_P` row slices it needs, so `dR_log_P` is re-read
/// `3 * natm / CROSS_CB` times instead of once per output row.  Symmetry
/// (`C[r][c] == C[c][r]` bitwise, both summed over `M` in the same order)
/// halves the tile count; only the upper-triangle tiles are evaluated and
/// mirrored on store.  `C` is materialized in the `ddw` buffer itself; the
/// quotient sweep then transforms each row in place.
///
/// The quotient sweep also accumulates the translation-invariance axis sums
/// (fullA = sum_A, fullB = sum_B, fullAB = sum_{A,B}) in the same pass, so
/// `ddR_Z`/`ddR_Pg` are consumed read-only and read exactly once, and the flat
/// `ddw` rows are fully overwritten (no zero fill).
fn eval_lane_ddw(
    ctx: &BeckePartitionContext<'_>,
    wquad: f64simd,
    attr: LaneAttrib,
    part: &LanePartition,
    scr: &mut LaneScratch,
) {
    let natm = ctx.natm;
    let n3 = 3 * natm;
    let P = &scr.P;
    let dR_log_P = &scr.dR_log_P[..];
    let ddR_Z = &scr.ddR_Z[..];
    let ddR_Pg = &scr.ddR_Pg[..];

    // gather M = A_g for the ddR_Pg cross term (dlog_Ag[r], P_Ag) first
    let P_Ag = match attr {
        // per-lane gather of the generating atom's row: stream each row M =
        // atm_idx[g] of dR_log_P into lane g (padding lanes keep 0 from the fill)
        LaneAttrib::ByGrid(atm_idx) => {
            scr.dlog_Ag.fill(simd_val(0.0));
            let mut P_Ag = simd_val(0.0);
            for g in 0..SIMDD {
                let M = atm_idx[g];
                if M >= natm {
                    continue;
                }
                let row = &dR_log_P[M * n3..(M + 1) * n3];
                for r in 0..n3 {
                    scr.dlog_Ag[r][g] = row[r][g];
                }
                P_Ag[g] = P[M][g];
            }
            P_Ag
        },
        // definite generating atom: direct index instead of the per-lane gathers
        LaneAttrib::ByAtom(atm_g) => {
            let row = &dR_log_P[atm_g * n3..(atm_g + 1) * n3];
            scr.dlog_Ag.copy_from_slice(row);
            P[atm_g]
        },
    };

    // quotient rule for ddw (r_g fixed): q = Pg / Z,
    //   d2q = (ddR_Pg - (dq_B)(dZ_A) - q ddR_Z) / Z - (dq_A)(dZ_B) / Z
    let inv_Z = simd_val(1.0) / part.Z;
    let q = part.Pg * inv_Z;
    let dq = &mut scr.dq;
    let dR_Z = &scr.dR_Z;
    let dR_Pg = &scr.dR_Pg;
    for r in 0..n3 {
        dq[r] = (dR_Pg[r] - q * dR_Z[r]) * inv_Z;
    }

    // rank-1 cross-term matrix C, tiled into the ddw buffer (upper-triangle
    // tiles, mirrored on store); see the function doc for the blocking
    let ddw = &mut scr.ddw[..];
    for rb in (0..n3).step_by(CROSS_CB) {
        let rlen = CROSS_CB.min(n3 - rb);
        for cb in (rb..n3).step_by(CROSS_CB) {
            let clen = CROSS_CB.min(n3 - cb);
            let mut acc = [[simd_val(0.0); CROSS_CB]; CROSS_CB];
            for M in 0..natm {
                let pM = P[M];
                let row = &dR_log_P[M * n3..M * n3 + n3];
                for i in 0..rlen {
                    let xri = pM * row[rb + i];
                    let arri = &mut acc[i];
                    for j in 0..clen {
                        arri[j] += xri * row[cb + j];
                    }
                }
            }
            for i in 0..rlen {
                for j in 0..clen {
                    let v = acc[i][j];
                    ddw[(rb + i) * n3 + cb + j] = v;
                    ddw[(cb + j) * n3 + rb + i] = v;
                }
            }
        }
    }

    // quotient sweep over the flat rows r = 3A + t: ddw_partial[r][c] =
    // wquad * d2q, transforming the C row in place; the translation-invariance
    // axis sums are accumulated in the same sweep (fullB per row in fs
    // registers, so only fullA needs a zero fill).
    scr.fullA.fill(simd_val(0.0));
    let mut fullAB = [[simd_val(0.0); 3]; 3]; // [t][s] = sum_A sum_B ddw
    let dlog_Ag = &scr.dlog_Ag[..];
    for A in 0..natm {
        for t in 0..3 {
            let r = A * 3 + t;
            // finalize the row; cross terms:
            //   ddZ  = ddR_Z  + C[r][c]           (C still stored in ddw row r)
            //   ddPg = ddR_Pg + P_Ag (dlog_Ag r)(dlog_Ag c)
            let dA_r = dlog_Ag[r];
            let dq_r = dq[r];
            let dZ_r = dR_Z[r];
            let ts = t * 3;
            let mut fs = [simd_val(0.0); 3];
            let ddw_row = &mut ddw[r * n3..(r + 1) * n3];
            let ddz_row = &ddR_Z[r * n3..(r + 1) * n3];
            let ddp_row = &ddR_Pg[r * n3..(r + 1) * n3];
            for B in 0..natm {
                for s in 0..3 {
                    let c = B * 3 + s;
                    let ddZ = ddz_row[c] + ddw_row[c];
                    let ddPg = ddp_row[c] + P_Ag * dA_r * dlog_Ag[c];
                    let d2q = (ddPg - dq[c] * dZ_r - q * ddZ) * inv_Z - dq_r * dR_Z[c] * inv_Z;
                    let v = wquad * d2q;
                    ddw_row[c] = v;
                    scr.fullA[B * 9 + ts + s] += v;
                    fs[s] += v;
                }
            }
            // fs[s] = sum_B v is exactly the fullB block of (A, t)
            scr.fullB[A * 9 + ts..A * 9 + ts + 3].copy_from_slice(&fs);
            for s in 0..3 {
                fullAB[t][s] += fs[s];
            }
        }
    }

    // translation-invariance fix (mirrors the deriv1 pattern): the quotient-rule
    // partial value is already correct for A,B != A_g; only the A=A_g row and
    // B=A_g column need the fix.  The axis sums are accumulated in f64simd and the
    // per-lane `for g` only writes back that row/column (minimal scalar work).
    //   row : ddw[A_g, t, B, s]   = -sum_{A'!=A_g} = -fullA[B,t,s] + ddw_partial[A_g,B,t,s]
    //   col : ddw[A, t, A_g, s]   = -sum_{B'!=A_g} = -fullB[A,t,s] + ddw_partial[A,A_g,t,s]
    //   corner: ddw[A_g,t,A_g,s] =  sum_{A'!=A_g,B'!=A_g}
    //                            = fullAB - fullB[A_g] - fullA[A_g] + ddw_partial[A_g,A_g]
    let ddw = &mut scr.ddw[..];
    let fullA = &scr.fullA[..];
    let fullB = &scr.fullB[..];
    match attr {
        LaneAttrib::ByGrid(atm_idx) => {
            for g in 0..SIMDD {
                let atm_g = atm_idx[g];
                if atm_g >= natm {
                    continue;
                }
                // row A = atm_g: B == atm_g is the corner, otherwise the row value
                //   (reads ddw[atm_g][B][t][s][g] = ddw_partial, still unmodified here)
                for B in 0..natm {
                    for t in 0..3 {
                        for s in 0..3 {
                            let idx = (atm_g * 3 + t) * n3 + B * 3 + s;
                            ddw[idx][g] = if B == atm_g {
                                fullAB[t][s][g] - fullB[atm_g * 9 + t * 3 + s][g] - fullA[atm_g * 9 + t * 3 + s][g]
                                    + ddw[(atm_g * 3 + t) * n3 + atm_g * 3 + s][g]
                            } else {
                                -fullA[B * 9 + t * 3 + s][g] + ddw[idx][g]
                            };
                        }
                    }
                }
                // column B = atm_g (A != atm_g; the A = atm_g corner is already written
                //   above, and ddw[A][atm_g] is still ddw_partial since the row only
                //   touched A = atm_g)
                for A in 0..natm {
                    if A == atm_g {
                        continue;
                    }
                    for t in 0..3 {
                        for s in 0..3 {
                            ddw[(A * 3 + t) * n3 + atm_g * 3 + s][g] -= fullB[A * 9 + t * 3 + s][g];
                        }
                    }
                }
            }
        },
        LaneAttrib::ByAtom(atm_g) => {
            // same fix, but atm_g is definite and uniform over the lanes, so the
            // row/column formulas apply to whole SIMD registers
            for B in 0..natm {
                for t in 0..3 {
                    for s in 0..3 {
                        let idx = (atm_g * 3 + t) * n3 + B * 3 + s;
                        ddw[idx] = if B == atm_g {
                            fullAB[t][s] - fullB[atm_g * 9 + t * 3 + s] - fullA[atm_g * 9 + t * 3 + s]
                                + ddw[(atm_g * 3 + t) * n3 + atm_g * 3 + s]
                        } else {
                            -fullA[B * 9 + t * 3 + s] + ddw[idx]
                        };
                    }
                }
            }
            for A in 0..natm {
                if A == atm_g {
                    continue;
                }
                for t in 0..3 {
                    for s in 0..3 {
                        ddw[(A * 3 + t) * n3 + atm_g * 3 + s] -= fullB[A * 9 + t * 3 + s];
                    }
                }
            }
        },
    }
}

// --- lane write-back and contraction --- //

/// Write one lane of `w` to the output buffer and accumulate the `c`
/// contraction partial.
///
/// - `w` : the lane's partition weight register.
/// - `g_start`/`g_end` : the lane's grid range within `[0, ngrids)`.
fn store_lane_w(
    ctx: &BeckePartitionContext<'_>,
    contraction: &mut TaskContraction,
    w: f64simd,
    g_start: usize,
    g_end: usize,
) {
    let nlane_g = g_end - g_start;
    if let Some(w_buf) = ctx.output.w.as_ref() {
        // SAFETY: tasks own disjoint grid ranges
        let wslc = unsafe { cast_mut_slice(&w_buf[g_start..g_end]) };
        wslc[..nlane_g].copy_from_slice(&w.0[..nlane_g]);
    }

    // contract w -> c:  c[iset] += sum_g contract_w[iset, g] * w[g]
    if let Some(cw) = ctx.contract_w.as_ref() {
        let cp = contraction.c.as_mut().unwrap();
        for iset in 0..ctx.nset_w.unwrap() {
            let cw_lane = load_simd_pad(&cw[iset][g_start..g_end]);
            cp[iset] += sum_lanes(w * cw_lane, nlane_g);
        }
    }
}

/// Write one lane of `dw` to the output buffer and accumulate the `dc`
/// contraction partial.
///
/// - `dw` : the lane's `dw` registers, flattened `[A, t]`.
fn store_lane_dw(
    ctx: &BeckePartitionContext<'_>,
    contraction: &mut TaskContraction,
    dw: &[f64simd],
    g_start: usize,
    g_end: usize,
) {
    let natm = ctx.natm;
    let ngrids = ctx.ngrids;
    let nlane_g = g_end - g_start;
    if let Some(dw_buf) = ctx.output.dw.as_ref() {
        // SAFETY: tasks own disjoint grid ranges
        let dweights = unsafe { cast_mut_slice(dw_buf) };
        for A in 0..natm {
            for t in 0..3 {
                let base = A * 3 * ngrids + t * ngrids + g_start;
                dweights[base..base + nlane_g].copy_from_slice(&dw[A * 3 + t].0[..nlane_g]);
            }
        }
    }

    // contract dw -> dc:  dc[A, t, iset] += sum_g contract_dw[iset, g] * dw[A, t, g]
    if let Some(cdw) = ctx.contract_dw.as_ref() {
        let dcp = contraction.dc.as_mut().unwrap();
        let nset = ctx.nset_dw.unwrap();
        // load this lane's contraction weights once per set, reuse across (A, t)
        for iset in 0..nset {
            contraction.cw_lanes_dw[iset] = load_simd_pad(&cdw[iset][g_start..g_end]);
        }
        for A in 0..natm {
            for t in 0..3 {
                let dwv = dw[A * 3 + t];
                for iset in 0..nset {
                    dcp[(A * 3 + t) * nset + iset] += sum_lanes(dwv * contraction.cw_lanes_dw[iset], nlane_g);
                }
            }
        }
    }
}

/// Write one lane of `ddw` to the output buffer and accumulate the `ddc`
/// contraction partial.
///
/// - `ddw` : the lane's `ddw` registers, flattened `[A, t, B, s]` (per-row length `3 * natm`, one
///   row per `[A, t]`).  The flat output/contraction index is C-order for `[A, t, B, s, (g|iset)]`.
fn store_lane_ddw(
    ctx: &BeckePartitionContext<'_>,
    contraction: &mut TaskContraction,
    ddw: &[f64simd],
    g_start: usize,
    g_end: usize,
) {
    let natm = ctx.natm;
    let n3 = 3 * natm;
    let ngrids = ctx.ngrids;
    let nlane_g = g_end - g_start;

    if let Some(ddw_buf) = ctx.output.ddw.as_ref() {
        // SAFETY: tasks own disjoint grid ranges
        let ddweights = unsafe { cast_mut_slice(ddw_buf) };
        for A in 0..natm {
            for t in 0..3 {
                for B in 0..natm {
                    for s in 0..3 {
                        let base = ((A * 3 + t) * natm + B) * (3 * ngrids) + s * ngrids + g_start;
                        ddweights[base..base + nlane_g]
                            .copy_from_slice(&ddw[(A * 3 + t) * n3 + B * 3 + s].0[..nlane_g]);
                    }
                }
            }
        }
    }

    // contract ddw -> ddc:
    //   ddc[A, t, B, s, iset] += sum_g contract_ddw[iset, g] * ddw[A, t, B, s, g]
    if let Some(cddw) = ctx.contract_ddw.as_ref() {
        let ddcp = contraction.ddc.as_mut().unwrap();
        let nset = ctx.nset_ddw.unwrap();
        // load this lane's contraction weights once per set, reuse across (A, t, B, s)
        for iset in 0..nset {
            contraction.cw_lanes_ddw[iset] = load_simd_pad(&cddw[iset][g_start..g_end]);
        }
        for A in 0..natm {
            for t in 0..3 {
                for B in 0..natm {
                    for s in 0..3 {
                        let ddwv = ddw[(A * 3 + t) * n3 + B * 3 + s];
                        for iset in 0..nset {
                            ddcp[((A * 3 + t) * natm + B) * 3 * nset + s * nset + iset] +=
                                sum_lanes(ddwv * contraction.cw_lanes_ddw[iset], nlane_g);
                        }
                    }
                }
            }
        }
    }
}

/* #endregion */

/* #region switch function utilities */

fn switch_f3(mu: f64simd, a_factor: f64) -> f64simd {
    let nu = mu + (simd_val(1.0) - mu * mu) * a_factor; // eq (A2)
    let f1 = (simd_val(1.5) - simd_val(0.5) * nu * nu) * nu; // eq (19)
    let f2 = (simd_val(1.5) - simd_val(0.5) * f1 * f1) * f1; // eq (19)
    let f3 = (simd_val(1.5) - simd_val(0.5) * f2 * f2) * f2; // eq (19)
    f3
}

fn switch_f_hardness(mu: f64simd, a_factor: f64, hardness: usize) -> f64simd {
    let nu = mu + (simd_val(1.0) - mu * mu) * a_factor; // eq (A2)
    let mut f = nu;
    for _ in 0..hardness {
        f = (simd_val(1.5) - simd_val(0.5) * f * f) * f; // eq (19)
    }
    f
}

fn switch_dnu_f3(mu: f64simd, a_factor: f64) -> (f64simd, f64simd) {
    let nu = mu + (simd_val(1.0) - mu * mu) * a_factor; // eq (A2)
    let f1 = (simd_val(1.5) - simd_val(0.5) * nu * nu) * nu; // eq (19)
    let f2 = (simd_val(1.5) - simd_val(0.5) * f1 * f1) * f1; // eq (19)
    let f3 = (simd_val(1.5) - simd_val(0.5) * f2 * f2) * f2; // eq (19)

    let df1 = simd_val(1.5) * (simd_val(1.0) - nu * nu);
    let df2 = simd_val(1.5) * (simd_val(1.0) - f1 * f1) * df1;
    let df3 = simd_val(1.5) * (simd_val(1.0) - f2 * f2) * df2;
    (f3, df3)
}

fn switch_dnu_f_hardness(mu: f64simd, a_factor: f64, hardness: usize) -> (f64simd, f64simd) {
    let nu = mu + (simd_val(1.0) - mu * mu) * a_factor; // eq (A2)
    let mut f = nu;
    let mut df = simd_val(1.0);
    for _ in 0..hardness {
        df = simd_val(1.5) * (simd_val(1.0) - f * f) * df;
        f = (simd_val(1.5) - simd_val(0.5) * f * f) * f; // eq (19)
    }
    (f, df)
}

/// Switch function `f3(nu)` together with its 1st and 2nd derivatives wrt `nu`, where
/// `nu = mu + a(1 - mu^2)` and `f3 = p∘p∘p(nu)`, `p(x) = 3/2 x − 1/2 x^3` (hardness = 3).
///
/// With `g_i = p'(f_{i-1})` (`f_0 = nu`), `p'(x) = 3/2(1 − x^2)`, `p''(x) = −3x`:
/// `f3'(nu) = g2 g1 g0`, `f3''(nu) = −3 [f2 (g1 g0)^2 + f1 g2 g0^2 + nu g2 g1]`.
fn switch_d2nu_f3(mu: f64simd, a_factor: f64) -> (f64simd, f64simd, f64simd) {
    let nu = mu + (simd_val(1.0) - mu * mu) * a_factor; // eq (A2)
    let f1 = (simd_val(1.5) - simd_val(0.5) * nu * nu) * nu; // eq (19)
    let f2 = (simd_val(1.5) - simd_val(0.5) * f1 * f1) * f1; // eq (19)
    let f3 = (simd_val(1.5) - simd_val(0.5) * f2 * f2) * f2; // eq (19)

    let g0 = simd_val(1.5) * (simd_val(1.0) - nu * nu);
    let g1 = simd_val(1.5) * (simd_val(1.0) - f1 * f1);
    let g2 = simd_val(1.5) * (simd_val(1.0) - f2 * f2);
    let f3p = g2 * g1 * g0;
    let f3pp = -simd_val(3.0) * (f2 * (g1 * g0) * (g1 * g0) + f1 * g2 * g0 * g0 + nu * g2 * g1);
    (f3, f3p, f3pp)
}

/// Arbitrary-hardness variant of [`switch_d2nu_f3`]: returns `f, f'(nu), f''(nu)` for
/// `f = p^hardness(nu)`. Loop recurrence (compute `ddf` first with old `f, df`, then `df`,
/// then `f`), `g = p'(f) = 3/2(1 − f^2)`, `p''(x) = −3x`: `ddf = −3 f df^2 + g ddf`.
fn switch_d2nu_f_hardness(mu: f64simd, a_factor: f64, hardness: usize) -> (f64simd, f64simd, f64simd) {
    let nu = mu + (simd_val(1.0) - mu * mu) * a_factor; // eq (A2)
    let mut f = nu;
    let mut df = simd_val(1.0);
    let mut ddf = simd_val(0.0);
    for _ in 0..hardness {
        let g = simd_val(1.5) * (simd_val(1.0) - f * f);
        ddf = -simd_val(3.0) * f * df * df + g * ddf; // f'' recurrence (old f, df, ddf)
        df = g * df; // f'  recurrence (old df)
        f = (simd_val(1.5) - simd_val(0.5) * f * f) * f; // f = p(f) (old f)
    }
    (f, df, ddf)
}

/* #endregion */

/* #region FpSimd enhancement */

/// Create a SIMD register with all lanes set to `x`.
const fn simd_val(x: f64) -> f64simd {
    f64simd::splat(x)
}

/// Lane-wise negation of a length-3 SIMD vector.
fn neg3(v: [f64simd; 3]) -> [f64simd; 3] {
    [-v[0], -v[1], -v[2]]
}

/// Lane-wise negation of a 3x3 SIMD matrix (over `(t, s)`).
fn neg33(m: [[f64simd; 3]; 3]) -> [[f64simd; 3]; 3] {
    [[-m[0][0], -m[0][1], -m[0][2]], [-m[1][0], -m[1][1], -m[1][2]], [-m[2][0], -m[2][1], -m[2][2]]]
}

/// Load up to `SIMDD` elements from `slc` into a SIMD register, zero-padding
/// the remaining lanes.
#[inline(always)]
fn load_simd_pad(slc: &[f64]) -> f64simd {
    let mut s = simd_val(0.0);
    for i in 0..slc.len() {
        s[i] = slc[i];
    }
    s
}

/// Horizontal sum of the first `n` lanes of a SIMD register (`n <= SIMDD`).
#[inline(always)]
fn sum_lanes(s: f64simd, n: usize) -> f64 {
    unsafe { std::hint::assert_unchecked(n <= SIMDD) };
    let mut acc = 0.0;
    for i in 0..n {
        acc += s[i];
    }
    acc
}

/// Lane-wise helpers on [`FpSimd`] used by the partition evaluation.
trait FpSimdEnhanceAPI<T> {
    /// Select `self` on lanes with `mask == true`, `other` on the rest.
    fn mask_select(self, mask: [bool; SIMDD], other: Self) -> Self;
    /// Lane-wise maximum with the scalar `val` (the `INVTOL` floor).
    fn max_compare(self, val: T) -> Self
    where
        T: PartialOrd;
}

impl<T: Copy> FpSimdEnhanceAPI<T> for FpSimd<T> {
    fn mask_select(self, mask: [bool; SIMDD], other: Self) -> Self {
        let mut result = self;
        for i in 0..SIMDD {
            match mask[i] {
                true => result[i] = self[i],
                false => result[i] = other[i],
            }
        }
        result
    }

    fn max_compare(mut self, val: T) -> Self
    where
        T: PartialOrd,
    {
        for i in 0..SIMDD {
            self[i] = if self[i] > val { self[i] } else { val };
        }
        self
    }
}

/* #endregion */

/* #region other utilities */

fn dist3_naive(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn dist3_hybrid(a: &[f64simd; 3], b: &[f64; 3]) -> f64simd {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).map(f64::sqrt)
}

/// Derive a `&mut [T]` from `&[T]` — undefined behaviour on aliased use.
///
/// Used only on the output buffers, whose writers are externally synchronized:
/// disjoint grid ranges for `w`/`dw`/`ddw`, the task mutex for `c`/`dc`/`ddc`.
#[allow(clippy::mut_from_ref)]
unsafe fn cast_mut_slice<T>(slc: &[T]) -> &mut [T] {
    let len = slc.len();
    let ptr = slc.as_ptr() as *mut T;
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

/* #endregion */

#[cfg(test)]
mod tests {
    use super::try_atm_quad_split;

    #[test]
    fn test_try_atm_quad_split() {
        // every atom owns a non-empty interval
        assert_eq!(try_atm_quad_split(&[0, 0, 1, 2], 3).unwrap(), vec![0, 2, 3, 4]);
        // atom 1 generated no grids: empty interval keeps its slot
        assert_eq!(try_atm_quad_split(&[0, 0, 2, 2], 3).unwrap(), vec![0, 2, 2, 4]);
        // out-of-range padding values (only possible at the tail) are left out
        assert_eq!(try_atm_quad_split(&[0, 0, 1, usize::MAX], 2).unwrap(), vec![0, 2, 3]);
        // interleaved grids are not grouped by atom
        assert!(try_atm_quad_split(&[0, 1, 0], 2).is_none());
    }
}
