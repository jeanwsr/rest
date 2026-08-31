// RKS Hessian XC terms with the Becke grid-shift; see also pyhessref/nimatmul/rks.py
//
// The quadrature grid is glued to the atoms that generated it, so the weight
// factor of the XC energy carries nuclear-coordinate derivatives.  On top of
// the grid-fixed terms this module adds the seven skeleton terms `de_becke_*`
// and the three f1ao corrections `vmat_becke_*`, restoring translational
// invariance of `de_xc_skeleton` and `vmat_deriv1_grid`.
//
// The 2nd Becke derivative `ddw` is never materialized: the only consumer
// (`de_becke_full_2`) contracts `ddw` with `exc * rho[0]` over the grid, which
// is exactly the `contract_ddw` channel of `becke_partition` with `nset = 1`.
//
// The grid-shift terms are optional (`grid_shift` argument): with it off, only
// the grid-fixed terms are evaluated, and `de_xc_skeleton` /
// `vmat_deriv1_grid` degenerate to their grid-fixed values.
//
// # Index and shape conventions
//
// - Index letters: `g` grid point; `u`/`v` AO basis; `x`/`y` rho component; `t` Cartesian direction
//   of the Hessian row atom `A`; `s` direction of the column atom `B`.
// - On-grid tensors are grid-leading: `rho [ngrids, nvar]`, `fxc [ngrids, nvar, nvar]`, `ao
//   [ngrids, nao, ncomp]`.
// - Hessian-like tensors `[3, 3, natm, natm]` (t, s, A, B); skeleton-Fock-like tensors `[nao, nao,
//   3, natm]` (u, v, t, A).
// - `nvar` (number of rho components): RHO 1, SIGMA 4, TAU 5 (see [`XCDenType::num_nvar`]).
// - AO derivative components (constants below): value `O` 0; gradient `X..Z` 1..3; 2nd derivatives
//   `XX..ZZ` 4..9 (symmetric pairs, see `IDX_AO_DERIV2`); 3rd derivatives `XXX..ZZZ` 10..19.

use super::prelude::*;
use crate::analdrv::prelude::*;
use crate::dft::gen_grids::becke_partitioning_deriv::{
    becke_partition_with_tables, gen_adjustment_factor, try_atm_quad_split, AtmIndices, BeckeMolTables,
    BeckePartitionArg,
};

use std::sync::atomic::{AtomicUsize, Ordering};

use XCDenType::*;

/// Becke switch-function hardness assumed by the grid-shift terms: the value the grid weights
/// were generated with (grid generation fixes 3).  Not exposed at this API level; the underlying
/// `becke_partition` accepts any hardness should this ever need to change.
pub const BECKE_HARDNESS: usize = 3;

/* #region const dimensions/indices definition */

const O: usize = 0;
const X: usize = 1;
const Y: usize = 2;
const Z: usize = 3;
const XX: usize = 4;
const XY: usize = 5;
const XZ: usize = 6;
const YX: usize = 5;
const YY: usize = 7;
const YZ: usize = 8;
const ZX: usize = 6;
const ZY: usize = 8;
const ZZ: usize = 9;
const XXX: usize = 10;
const XXY: usize = 11;
const XXZ: usize = 12;
const XYY: usize = 13;
const XYZ: usize = 14;
const XZZ: usize = 15;
const YYY: usize = 16;
const YYZ: usize = 17;
const YZZ: usize = 18;
const ZZZ: usize = 19;

const IDX_AO_DERIV2: [[usize; 3]; 3] = [[XX, XY, XZ], [XY, YY, YZ], [XZ, YZ, ZZ]];

/// AO derivative order of the `ao` tensor for the Hessian evaluation.  RHO
/// needs 2 (`get_drho` uses gradient channels, `make_dao_vxc_diag` the 2nd
/// derivatives); SIGMA/TAU need 3 (`make_dao_vxc_diag` uses 3rd derivatives).
///
/// # Returns
///
/// - `deriv` : 2 for RHO, 3 for SIGMA/TAU.  The `ao` tensor then carries `AO_DERIV_DIM[deriv]`
///   components (10 resp. 20).
pub const fn get_hess_ao_deriv(xc_type: XCDenType) -> usize {
    match xc_type {
        RHO => 2,
        SIGMA => 3,
        TAU => 3,
        LAPL => unimplemented!(),
    }
}

/// Number of leading AO channels contracted with the density matrix into
/// `ao_dm0`: the value only for RHO, value + gradient for SIGMA/TAU.
///
/// # Returns
///
/// - `ncomp` : 1 for RHO, 4 for SIGMA/TAU (same as [`XCDenType::num_ao_comp`] for the implemented
///   families).
pub fn get_hess_ncomp_ao_dm0(xc_type: XCDenType) -> usize {
    match xc_type {
        LAPL => unimplemented!(),
        _ => xc_type.num_ao_comp(),
    }
}

/// Sentinel "no atom" value of the per-chunk attribution: chunks of grids that
/// belong to no generating atom (trailing grids beyond `atm_quad_split[natm]`)
/// carry it.  The grid-shift terms need atom attribution, so such chunks are
/// only legal with `grid_shift = false`.
pub const NO_ATM: usize = usize::MAX;

/* #endregion */

/* #region macro for indexing last dimension */

macro_rules! index {
    ($tsr: ident, $($idx:expr),*) => {
        $tsr.i((Ellipsis, $($idx),*))
    };
}

macro_rules! index_mut {
    ($tsr: ident, $($idx:expr),*) => {
        (*&mut $tsr.i_mut((Ellipsis, $($idx),*)))
    };
}

/* #endregion */

/* #region grid splitting helpers */

/// Becke grid-attribution boundaries for a grid range (chunk) that holds only atom
/// `atm_idx`'s grids: atoms before `atm_idx` get the empty interval `[0, 0)`, atom
/// `atm_idx` owns `[0, n)`, and atoms after own the empty `[n, n)`.
pub fn by_atom_chunk(natm: usize, atm_idx: usize, n: usize) -> Vec<usize> {
    let mut v = vec![0; natm + 1];
    for x in v.iter_mut().skip(atm_idx + 1) {
        *x = n;
    }
    v
}

/// Split the atom-grouped grid into pieces of at most `nsplit` grids, respecting
/// atom boundaries, so every piece carries one definite generating atom.  The
/// number of atoms is `atm_quad_split.len() - 1`.
///
/// Grids beyond `atm_quad_split[natm]` (no generating atom, e.g. external grids)
/// form trailing pieces attributed to [`NO_ATM`]; they carry only the grid-fixed
/// terms, so they are legal only with `grid_shift = false`.
///
/// The driver uses this at chunk granularity (`nsplit = nchunk`): the returned
/// `(atm_idx, start, end)` triples are the parallel work units of a single flat
/// chunk-level par_iter.
pub fn quad_split_by_atom(atm_quad_split: &[usize], ngrids: usize, nsplit: usize) -> Vec<(usize, usize, usize)> {
    assert!(!atm_quad_split.is_empty(), "atm_quad_split must have length natm + 1");
    let natm = atm_quad_split.len() - 1;
    let mut pieces = Vec::new();
    for A in 0..natm {
        let mut start = atm_quad_split[A];
        let end = atm_quad_split[A + 1];
        while start < end {
            let next_end = (start + nsplit).min(end);
            pieces.push((A, start, next_end));
            start = next_end;
        }
    }
    // trailing grids attached to no atom, attributed to the sentinel
    let mut start = atm_quad_split[natm];
    while start < ngrids {
        let next_end = (start + nsplit).min(ngrids);
        pieces.push((NO_ATM, start, next_end));
        start = next_end;
    }
    pieces
}

/* #endregion */

/* #region basic pure functions of skeleton hessian evaluation */

/// On-grid density with 1st/2nd functional derivatives and the per-particle XC
/// energy density.
///
/// The per-particle energy density `exc` (order 0) is needed by the `cddw`
/// contraction of `de_becke_full_2`; callers that do not need it simply ignore
/// the output.
///
/// # Parameters
///
/// - `xc_func_list` : list of `(scale, functional)` pairs.  The overall family is the strictest one
///   across the list; contributions of looser families are added into their leading `nvar_i` slice.
/// - `ao` : shape `[ngrids, nao, ncomp]` (g, u, component).  AO values and derivatives; only each
///   family's leading channels are read.
/// - `ao_dm0` : shape `[ngrids, nao, ncomp_ao_dm0]`.  Leading AO channels contracted with the
///   density matrix.
///
/// # Returns
///
/// - `rho` : shape `[ngrids, nvar]` (g, x).  On-grid density components.
/// - `exc` : shape `[ngrids]`.  Per-particle XC energy density.
/// - `vxc` : shape `[ngrids, nvar]`.  1st functional derivative.
/// - `fxc` : shape `[ngrids, nvar, nvar]`.  2nd functional derivative.
pub fn get_rho_exc_vxc_fxc(
    xc_func_list: &[(f64, LibXCFunctional)],
    ao: TsrView,
    ao_dm0: TsrView,
) -> (Tsr, Tsr, Tsr, Tsr) {
    assert!(!xc_func_list.is_empty(), "xc_func_list must not be empty");
    let xc_type = xc_func_list
        .iter()
        .map(|(_, f)| determine_den_type(f))
        .max_by_key(|t| t.num_nvar())
        .expect("xc_func_list must not be empty");
    let nvar = xc_type.num_nvar();
    let ngrids = ao.shape()[0];
    let device = ao.device().clone();

    let mut rho = rt::zeros(([ngrids, nvar], &device));
    index_mut!(rho, 0) += rt::vecdot(index!(ao, 0), index!(ao_dm0, O), 1);
    if matches!(xc_type, SIGMA | TAU) {
        index_mut!(rho, X) += 2 * rt::vecdot(index!(ao, X), index!(ao_dm0, O), 1);
        index_mut!(rho, Y) += 2 * rt::vecdot(index!(ao, Y), index!(ao_dm0, O), 1);
        index_mut!(rho, Z) += 2 * rt::vecdot(index!(ao, Z), index!(ao_dm0, O), 1);
    }
    if matches!(xc_type, TAU) {
        index_mut!(rho, 4) += 0.5
            * (rt::vecdot(index!(ao, X), index!(ao_dm0, X), 1)
                + rt::vecdot(index!(ao, Y), index!(ao_dm0, Y), 1)
                + rt::vecdot(index!(ao, Z), index!(ao_dm0, Z), 1))
    }

    let mut exc = rt::zeros(([ngrids], &device));
    let mut vxc = rt::zeros(([ngrids, nvar], &device));
    let mut fxc = rt::zeros(([ngrids, nvar, nvar], &device));
    for (scale, xc_func) in xc_func_list {
        let xc_type_i = determine_den_type(xc_func);
        let nvar_i = xc_type_i.num_nvar();
        let rho_i = rho.i((.., ..nvar_i));
        let xc_eff = libxc_eval_eff(xc_func, rho_i, 2, false);
        let [e_i, vxc_i, fxc_i] = xc_eff.into_iter().collect_array().unwrap();
        exc += *scale * e_i.into_shape([ngrids]);
        *&mut vxc.i_mut((.., ..nvar_i)) += *scale * vxc_i;
        *&mut fxc.i_mut((.., ..nvar_i, ..nvar_i)) += *scale * fxc_i;
    }

    (rho, exc, vxc, fxc)
}

/// Evaluate the (spin-unpolarized) XC potential `vxc` and kernel `fxc` from a given ground-state
/// `rho`, by summing each sub-functional's `libxc_eval_eff` (deriv = 2) contribution.
///
/// `rho` : shape `[ngrids, nvar]`, where `nvar` is the strictest density-variable count across the
/// functional list. Extracted from the rho formation so that callers which already have `rho`
/// (e.g. obtained directly from occupied orbitals via a bra-ket contraction) can skip the
/// `ao_dm0`-based density formation.
pub fn eval_vxc_fxc_from_rho(xc_func_list: &[(f64, LibXCFunctional)], rho: TsrView) -> (Tsr, Tsr) {
    assert!(!xc_func_list.is_empty(), "xc_func_list must not be empty");
    let xc_type = xc_func_list
        .iter()
        .map(|(_, f)| determine_den_type(f))
        .max_by_key(|t| t.num_nvar())
        .expect("xc_func_list must not be empty");
    let nvar = xc_type.num_nvar();
    let ngrids = rho.shape()[0];
    let device = rho.device().clone();

    let mut vxc = rt::zeros(([ngrids, nvar], &device));
    let mut fxc = rt::zeros(([ngrids, nvar, nvar], &device));
    for (scale, xc_func) in xc_func_list {
        let xc_type_i = determine_den_type(xc_func);
        let nvar_i = xc_type_i.num_nvar();
        // each sub-functional consumes only the leading `nvar_i` rho components.
        let rho_i = rho.i((.., ..nvar_i));
        let xc_eff = libxc_eval_eff(xc_func, rho_i, 2, false);
        let [_, vxc_i, fxc_i] = xc_eff.into_iter().collect_array().unwrap();
        // accumulate into the leading slice of the (possibly larger) global tensors.
        *&mut vxc.i_mut((.., ..nvar_i)) += *scale * vxc_i;
        *&mut fxc.i_mut((.., ..nvar_i, ..nvar_i)) += *scale * fxc_i;
    }
    (vxc, fxc)
}

/// Lean evaluation of only `vxc` and `fxc` on a given grid, for use as the CP-KS `cpks_vxc` /
/// `cpks_fxc` when a dedicated (coarser) CP-KS grid is attached.
///
/// Compared to [`make_hessian_setup_becke`], this:
/// - skips all skeleton-Hessian intermediates (`de_fxc`, `de_vxc_diag`, `de_vxc_off`, `vmat_ip`,
///   `vmat_deriv1`);
/// - evaluates AO at the minimum derivative order needed to form the density
///   (`xc_type.num_ao_deriv()`, i.e. 0 for LDA / 1 for GGA, MGGA) instead of the Hessian derivative
///   order (`get_hess_ao_deriv`, 2 / 3);
/// - forms the ground-state density `rho` directly from the occupied MO coefficients via a bra-ket
///   contraction ([`NIMatmul::make_rho_from_homogeneous_braket`]) instead of building the full
///   `[nao, nao]` `dm0` matrix, exploiting the low rank of the occupied space (consistent with the
///   CP-KS response path in [`get_rks_response_bra`]).
///
/// The CP-KS grid is assumed coarse enough that the full-grid AO tensor fits in memory; no batching
/// is performed.
pub fn make_cpks_vxc_fxc(
    xc_func_list: &[(f64, LibXCFunctional)],
    ni: &mut NIMatmul,
    mo_coeff: TsrView,
    mo_occ: TsrView,
) -> (Tsr, Tsr) {
    let xc_type = determine_den_type_from_list(&xc_func_list.iter().map(|(_, f)| f).collect_vec());
    // bake the occupation into the occupied coefficients (sqrt so the bra-ket square reproduces the
    // occ-weighted density for every density component: rho, sigma, tau are all bilinear in phi).
    let occidx = mo_occ.view().greater(0).into_vec();
    let mocc = mo_coeff.bool_select(-1, &occidx);
    let occ = mo_occ.bool_select(-1, &occidx);
    let occ_sqrt = occ.mapv(f64::sqrt);
    let mocc_2 = &mocc * occ_sqrt.i((None, ..));
    // rho : [ngrids, nvar, 1] (single set) -> [ngrids, nvar]
    let rho = ni.make_rho_from_homogeneous_braket(&[mocc_2.view()], xc_type);
    eval_vxc_fxc_from_rho(xc_func_list, rho.i((.., .., 0)))
}

/// 1st-order skeleton derivative of the rho components with respect to nuclear
/// coordinates.
///
/// The skeleton derivative counts only the basis functions following the
/// nucleus they are centred on, density matrix held fixed: for each atom `A`
/// and direction `t` the derivative acts on bra indices inside `A`'s AO slice.
/// Symmetric components (rho + gradient) carry a factor 2 from bra/ket
/// symmetry; the tau channel does not.  The bra contraction is accumulated
/// with a leading minus, so `drho` summed over `A` equals `-d rho / dr` (the
/// negated spatial derivative); the drivers negate the atom-sum back into
/// `prho`.
///
/// # Parameters
///
/// - `xc_type` : density family; selects which components contribute.
/// - `ao` : shape `[ngrids, nao, ncomp]`; reads up to the 2nd-order channels.
/// - `ao_dm0` : shape `[ngrids, nao, ncomp_ao_dm0]`.
/// - `aoslices` : shape `[natm, 4]`; per-atom `[shl0, shl1, p0, p1]` AO slices.
///
/// # Returns
///
/// - `drho` : shape `[ngrids, nvar, 3, natm]` (g, x, t, A).
pub fn get_drho(xc_type: XCDenType, ao: TsrView, ao_dm0: TsrView, aoslices: &[[usize; 4]]) -> Tsr {
    let ngrids = ao.shape()[0];
    let nvar = xc_type.num_nvar();
    let natm = aoslices.len();
    let device = ao.device().clone();

    let mut drho = rt::zeros(([ngrids, nvar, 3, natm], &device));

    // components: [rho_var, t_direction, cbra, cket]
    let mut components = vec![(0, 0, X, O), (0, 1, Y, O), (0, 2, Z, O)];
    if matches!(xc_type, SIGMA | TAU) {
        let sigma_bra2_ket0 = [
            [(1, 0, XX, O), (2, 0, XY, O), (3, 0, XZ, O)],
            [(1, 1, YX, O), (2, 1, YY, O), (3, 1, YZ, O)],
            [(1, 2, ZX, O), (2, 2, ZY, O), (3, 2, ZZ, O)],
        ];
        components.extend(sigma_bra2_ket0.concat());
        let sigma_bra1_ket1 = [
            [(1, 0, X, X), (2, 0, X, Y), (3, 0, X, Z)],
            [(1, 1, Y, X), (2, 1, Y, Y), (3, 1, Y, Z)],
            [(1, 2, Z, X), (2, 2, Z, Y), (3, 2, Z, Z)],
        ];
        components.extend(sigma_bra1_ket1.concat());
    }
    if matches!(xc_type, TAU) {
        let tau_bra2_ket1 = [
            [(4, 0, XX, X), (4, 0, XY, Y), (4, 0, XZ, Z)],
            [(4, 1, YX, X), (4, 1, YY, Y), (4, 1, YZ, Z)],
            [(4, 2, ZX, X), (4, 2, ZY, Y), (4, 2, ZZ, Z)],
        ];
        components.extend(tau_bra2_ket1.concat());
    }

    for (A, &[_, _, p0, p1]) in aoslices.iter().enumerate() {
        let slc = rt::slice!(p0, p1);
        for &(v, t, cbra, cket) in &components {
            *&mut drho.i_mut((.., v, t, A)) -= rt::vecdot(ao.i((.., slc, cbra)), ao_dm0.i((.., slc, cket)), 1);
        }
    }

    match xc_type {
        RHO => *&mut drho.i_mut((.., 0..1)) *= 2.0,
        SIGMA | TAU => *&mut drho.i_mut((.., 0..4)) *= 2.0,
        LAPL => unimplemented!(),
    }
    drho
}

/// fxc contribution to the XC skeleton Hessian:
/// `einsum("gxy, gxtA, gysB -> tsAB", wf, drho, drho)`.
///
/// # Parameters
///
/// - `wf` : shape `[ngrids, nvar, nvar]` (g, x, y).  Grid-weighted fxc kernel.
/// - `drho` : shape `[ngrids, nvar, 3, natm]` (output of [`get_drho`]).
///
/// # Returns
///
/// - `de_fxc` : shape `[3, 3, natm, natm]` (t, s, A, B).
pub fn get_de_fxc(wf: TsrView, drho: TsrView) -> Tsr {
    // gxy, gxtA, gysB -> tsAB

    let [ngrids, nvar, _, natm] = drho.shape().iter().cloned().collect_array().unwrap();

    let tmp1 = rt::vecdot(wf.i((.., .., .., None, None)), drho.i((.., .., None, .., ..)), 1);
    let tmp1 = tmp1.reshape([ngrids * nvar, natm * 3]);
    let drho = drho.reshape([ngrids * nvar, natm * 3]);
    let tmp2 = tmp1.t() % drho;

    tmp2.reshape([3, natm, 3, natm]).transpose([0, 2, 1, 3]).into_contig(ColMajor)
}

/// AO-resolved diagonal vxc kernel, the builder part of [`get_de_vxc_diag`].
/// Both the same-atom Hessian block `de_vxc_diag` and the grid-shift part
/// [`get_de_becke_vxc_parts`] contract this same kernel, so it is built once
/// per chunk and shared.
///
/// # Parameters
///
/// - `xc_type` : density family.
/// - `ao` : shape `[ngrids, nao, ncomp]`; reads the 2nd-order channels, and the 3rd-order ones for
///   SIGMA/TAU.
/// - `ao_dm0` : shape `[ngrids, nao, ncomp_ao_dm0]`.
/// - `wv` : shape `[ngrids, nvar]`.  Grid-weighted vxc.
///
/// # Returns
///
/// - `dao_vxc_diag` : shape `[nao, 6]` (u, pair); the 6 components are the symmetric Cartesian
///   pairs (xx, xy, xz, yy, yz, zz).
pub fn make_dao_vxc_diag(xc_type: XCDenType, ao: TsrView, ao_dm0: TsrView, wv: TsrView) -> Tsr {
    const TRIPLE_SIGMA_DIAG: [[usize; 3]; 6] =
        [[XXX, XXY, XXZ], [XXY, XYY, XYZ], [XXZ, XYZ, XZZ], [XYY, YYY, YYZ], [XYZ, YYZ, YZZ], [XZZ, YZZ, ZZZ]];
    const TRIPLE_TAU_DIAG: [[usize; 6]; 3] =
        [[XXX, XXY, XXZ, XYY, XYZ, XZZ], [XXY, XYY, XYZ, YYY, YYZ, YZZ], [XXZ, XYZ, XZZ, YYZ, YZZ, ZZZ]];

    let nao = ao.shape()[1];
    let device = ao.device().clone();

    let mut dao_vxc_diag: Tsr = rt::zeros(([nao, 6], &device));

    // contribution 1: lda/gga ao deriv 2
    let mut aow = index!(ao_dm0, O) * index!(wv, 0);
    if matches!(xc_type, SIGMA | TAU) {
        aow += index!(ao_dm0, X) * index!(wv, X);
        aow += index!(ao_dm0, Y) * index!(wv, Y);
        aow += index!(ao_dm0, Z) * index!(wv, Z);
    }
    for (idx_ts, its) in [XX, XY, XZ, YY, YZ, ZZ].into_iter().enumerate() {
        index_mut!(dao_vxc_diag, idx_ts) += 2 * rt::vecdot(index!(ao, its), &aow, 0);
    }

    // contribution 2: gga ao deriv 3
    if matches!(xc_type, SIGMA | TAU) {
        for (idx_ts, &[i3x, i3y, i3z]) in TRIPLE_SIGMA_DIAG.iter().enumerate() {
            let aow =
                index!(ao, i3x) * index!(wv, X) + index!(ao, i3y) * index!(wv, Y) + index!(ao, i3z) * index!(wv, Z);
            index_mut!(dao_vxc_diag, idx_ts) += 2 * rt::vecdot(&aow, index!(ao_dm0, O), 0);
        }
    }

    // contribution 3: tau ao deriv 3
    if matches!(xc_type, TAU) {
        for (r, &idx_tri) in TRIPLE_TAU_DIAG.iter().enumerate() {
            let aow = index!(ao_dm0, r + 1) * index!(wv, 4);
            for (idx_ts, &i3) in idx_tri.iter().enumerate() {
                index_mut!(dao_vxc_diag, idx_ts) += rt::vecdot(index!(ao, i3), &aow, 0);
            }
        }
    }

    dao_vxc_diag
}

/// Same-atom (A == B) block of the XC skeleton Hessian, the reduction part of
/// [`make_dao_vxc_diag`]: sums the kernel over each atom's AO slice and
/// expands the 6 symmetric pairs into a dense (3, 3) block.
///
/// # Parameters
///
/// - `dao_vxc_diag` : shape `[nao, 6]` (output of [`make_dao_vxc_diag`]).
/// - `aoslices` : shape `[natm, 4]`.
///
/// # Returns
///
/// - `de_vxc_diag` : shape `[3, 3, natm, natm]`; only the `A == B` diagonal blocks are non-zero.
pub fn get_de_vxc_diag(dao_vxc_diag: TsrView, aoslices: &[[usize; 4]]) -> Tsr {
    let natm = aoslices.len();
    let device = dao_vxc_diag.device().clone();

    let mut de_vxc_diag = rt::zeros(([6, natm, natm], &device));
    for (A, &[_, _, p0, p1]) in aoslices.iter().enumerate() {
        let slc = rt::slice!(p0, p1);
        de_vxc_diag.i_mut((.., A, A)).assign(dao_vxc_diag.i(slc).sum_axes(0));
    }
    de_vxc_diag.index_select(0, [0, 1, 2, 1, 3, 4, 2, 4, 5]).into_shape([3, 3, natm, natm])
}

/// AO-resolved two-index vxc kernel, the builder part of [`get_de_vxc_off`];
/// also contracted by the grid-shift part [`get_de_becke_vxc_parts`].
///
/// Note the axis order: the AO indices lead and the direction pair trails,
/// keeping each (t, s) block contiguous in column-major storage.
///
/// # Parameters
///
/// - `xc_type` : density family.
/// - `ao` : shape `[ngrids, nao, ncomp]`; reads channels up to the 2nd-order ones.
/// - `wv` : shape `[ngrids, nvar]`.  Grid-weighted vxc.
///
/// # Returns
///
/// - `dao_vxc_off` : shape `[nao, nao, 3, 3]` (u, v, t, s), symmetrised under `[t, s, u, v] -> [s,
///   t, v, u]`.
pub fn make_dao_vxc_off(xc_type: XCDenType, ao: TsrView, wv: TsrView) -> Tsr {
    let nao = ao.shape()[1];
    let device = ao.device().clone();

    let mut dao_vxc_off: Tsr = rt::zeros(([nao, nao, 3, 3], &device));

    if matches!(xc_type, RHO) {
        for t in 0..3 {
            let aowv = index!(wv, 0) * index!(ao, t + 1);
            for s in 0..3 {
                index_mut!(dao_vxc_off, t, s).matmul_from(aowv.t(), index!(ao, s + 1), 1.0, 1.0);
            }
        }
    }

    if matches!(xc_type, SIGMA | TAU) {
        for t in 0..3 {
            let mut aowv: Tsr = 0.5 * index!(wv, 0) * index!(ao, t + 1);
            for r in 0..3 {
                aowv += index!(wv, r + 1) * index!(ao, IDX_AO_DERIV2[t][r]);
            }
            for s in 0..3 {
                index_mut!(dao_vxc_off, t, s).matmul_from(aowv.t(), index!(ao, s + 1), 2.0, 1.0);
            }
        }
    }

    if matches!(xc_type, TAU) {
        let mut dao_vxc_tau: Tsr = rt::zeros(([nao, nao, 3, 3], &device));
        for r in 0..3 {
            for t in 0..3 {
                let aowv: Tsr = 0.5 * index!(wv, 4) * index!(ao, IDX_AO_DERIV2[t][r]);
                for s in 0..t + 1 {
                    index_mut!(dao_vxc_tau, t, s).matmul_from(aowv.t(), index!(ao, IDX_AO_DERIV2[s][r]), 1.0, 1.0);
                }
            }
        }

        for t in 0..3 {
            for s in 0..t + 1 {
                index_mut!(dao_vxc_off, t, s) += &index!(dao_vxc_tau, t, s);
            }
            for s in 0..t {
                index_mut!(dao_vxc_off, s, t) += &index!(dao_vxc_tau, t, s).t();
            }
        }
    }

    // symmetrised under [t, s, mu, nu] -> [s, t, nu, mu]
    &dao_vxc_off + dao_vxc_off.transpose([1, 0, 3, 2])
}

/// Two-atom block of the XC skeleton Hessian, the reduction part of
/// [`make_dao_vxc_off`]: contracts the kernel with the matching `dm0` AO
/// slices per (A, B) block.  Both `A == B` and `A != B` entries are populated —
/// the diag/off decomposition is by integral kernel, not by atom index.
///
/// # Parameters
///
/// - `dao_vxc_off` : shape `[nao, nao, 3, 3]` (u, v, t, s), output of [`make_dao_vxc_off`].
/// - `dm0` : shape `[nao, nao]`.  Density matrix in AO basis.
/// - `aoslices` : shape `[natm, 4]`.
///
/// # Returns
///
/// - `de_vxc_off` : shape `[3, 3, natm, natm]`.
pub fn get_de_vxc_off(dao_vxc_off: TsrView, dm0: TsrView, aoslices: &[[usize; 4]]) -> Tsr {
    let natm = aoslices.len();
    let device = dao_vxc_off.device().clone();

    let mut de_vxc_off = rt::zeros(([3, 3, natm, natm], &device));
    for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
        let slcA = rt::slice!(p0A, p1A);
        for (B, &[_, _, p0B, p1B]) in aoslices.iter().enumerate() {
            let slcB = rt::slice!(p0B, p1B);
            let contrib = rt::vecdot(dao_vxc_off.i((slcA, slcB)), dm0.i((slcA, slcB)), ([0, 1], [0, 1]));
            de_vxc_off.i_mut((.., .., A, B)).assign(&contrib);
            de_vxc_off.i_mut((.., .., B, A)).assign(contrib.t());
        }
    }

    de_vxc_off
}

/// Gradient-level Vxc matrix shared across all atoms: the AO-space object
/// whose bra-side AO slice per atom yields the on-atom contribution that
/// [`get_vmat_vxc`] adds to the per-atom skeleton Fock derivative.
///
/// # Parameters
///
/// - `xc_type` : density family.
/// - `ao` : shape `[ngrids, nao, ncomp]`; reads up to the 2nd-order channels.
/// - `wv` : shape `[ngrids, nvar]`.  Grid-weighted vxc.
///
/// # Returns
///
/// - `vmat_ip` : shape `[nao, nao, 3]` (u, v, t), indexed by the Cartesian direction of the bra
///   derivative.  Not symmetrised in AO indices — the symmetrisation happens per atom slice in
///   [`get_vmat_vxc`].
pub fn get_vmat_ip(xc_type: XCDenType, ao: TsrView, wv: TsrView) -> Tsr {
    let nao = ao.shape()[1];
    let device = ao.device().clone();

    let mut vmat_ip = rt::zeros(([nao, nao, 3], &device));

    if matches!(xc_type, RHO) {
        // bra-on-A and ket-on-A halves are identical for LDA
        // (both equal 0.5 * wv[0] * ao[t+1]^T @ ao[O]); folded into a single contraction.
        let aow: Tsr = index!(wv, 0) * index!(ao, O);
        for t in 0..3 {
            index_mut!(vmat_ip, t).matmul_from(&index!(ao, t + 1).t(), &aow, 1.0, 1.0);
        }
        return vmat_ip;
    }

    assert!(matches!(xc_type, SIGMA | TAU));

    // GGA + MGGA share the same SIGMA structure
    let mut aow: Tsr = 0.5 * index!(wv, 0) * index!(ao, O);
    for r in 0..3 {
        aow += index!(wv, r + 1) * index!(ao, r + 1);
    }
    for t in 0..3 {
        index_mut!(vmat_ip, t).matmul_from(&index!(ao, t + 1).t(), &aow, 1.0, 1.0);
    }

    for t in 0..3 {
        let mut aow_d: Tsr = 0.5 * index!(wv, 0) * index!(ao, t + 1);
        for r in 0..3 {
            aow_d += index!(wv, r + 1) * index!(ao, IDX_AO_DERIV2[t][r]);
        }
        index_mut!(vmat_ip, t).matmul_from(&aow_d.t(), &index!(ao, O), 1.0, 1.0);
    }

    // MGGA tau channel
    if matches!(xc_type, TAU) {
        for r in 0..3 {
            let aow: Tsr = 0.5 * index!(wv, 4) * index!(ao, r + 1);
            for t in 0..3 {
                index_mut!(vmat_ip, t).matmul_from(&index!(ao, IDX_AO_DERIV2[t][r]).t(), &aow, 1.0, 1.0);
            }
        }
    }

    vmat_ip
}

/// vxc (ipip, basis-derivative) contribution to the per-atom skeleton
/// derivative of the Vxc Fock matrix: the slice of the gradient-level
/// `vmat_ip` that lives on each atom `A`'s bra rows.  Spin-diagonal, so UKS
/// reuses it per spin channel.
///
/// # Parameters
///
/// - `vmat_ip` : shape `[nao, nao, 3]`, output of [`get_vmat_ip`].
/// - `aoslices` : shape `[natm, 4]`.
///
/// # Returns
///
/// - `vmat_vxc` : shape `[nao, nao, 3, natm]`, assembled across the AO axes (bra + ket).
pub fn get_vmat_vxc(vmat_ip: TsrView, aoslices: &[[usize; 4]]) -> Tsr {
    let natm = aoslices.len();
    let nao = vmat_ip.shape()[0];

    let mut vmat_vxc: Tsr = rt::zeros(([nao, nao, 3, natm], vmat_ip.device()));

    for (A, &[_, _, p0, p1]) in aoslices.iter().enumerate() {
        let slc = rt::slice!(p0, p1);
        *&mut vmat_vxc.i_mut((slc, .., .., A)) -= vmat_ip.i((slc, .., ..));
    }

    &vmat_vxc + vmat_vxc.swapaxes(0, 1)
}

/// fxc contribution to the per-atom skeleton derivative of the Vxc Fock
/// matrix: the fxc kernel folded against the skeleton density derivative
/// `drho[A]`, one [`xc_fock_stack`] call per atom row.  The genuinely
/// spin-coupled piece for UKS, so the UKS counterpart is kept separate.
///
/// # Parameters
///
/// - `xc_type` : density family.
/// - `ao` : shape `[ngrids, nao, ncomp]`; reads the leading `num_ao_comp` channels (1 for RHO, 4
///   for SIGMA/TAU), through [`xc_fock_stack`].
/// - `drho` : shape `[ngrids, nvar, 3, natm]`, output of [`get_drho`].
/// - `wf` : shape `[ngrids, nvar, nvar]`.  Grid-weighted fxc.
///
/// # Returns
///
/// - `vmat_fxc` : shape `[nao, nao, 3, natm]`, assembled across the AO axes.
pub fn get_vmat_fxc(xc_type: XCDenType, ao: TsrView, drho: TsrView, wf: TsrView) -> Tsr {
    let natm = drho.shape()[3];
    let nao = ao.shape()[1];

    let mut vmat_fxc: Tsr = rt::zeros(([nao, nao, 3, natm], ao.device()));

    for A in 0..natm {
        // field [g, x, t] = sum_y wf[g, x, y] drho[g, y, t, A], passed unscaled: the AO-axis
        // symmetrisation of `xc_fock_stack` restores the rho factor to 1 and the tau factor
        // to 0.5 (the tau pair products enter the symmetrisation only once)
        let field = rt::vecdot(&wf, drho.i((.., None, .., .., A)), 2);
        *&mut vmat_fxc.i_mut((.., .., .., A)) += &xc_fock_stack(xc_type, ao.view(), field.view());
    }

    vmat_fxc
}

/// Per-atom skeleton derivative of the Vxc Fock matrix, the sum of its vxc
/// (ipip) and fxc contributions.  Each part is assembled independently
/// (bra + ket) and the two are summed; the split is exact up to
/// floating-point order.
pub fn get_vmat_deriv1(
    xc_type: XCDenType,
    ao: TsrView,
    drho: TsrView,
    wf: TsrView,
    vmat_ip: TsrView,
    aoslices: &[[usize; 4]],
) -> Tsr {
    let vmat_fxc = get_vmat_fxc(xc_type, ao, drho, wf);
    let vmat_vxc = get_vmat_vxc(vmat_ip, aoslices);
    &vmat_fxc + &vmat_vxc
}

/* #endregion */

/* #region becke grid-shift parts: skeleton hessian */

/// `de_becke_atom_1`: `einsum("g, txg, xyg, Bsyg -> Bts", w, prho, fxc, drho)`,
/// evaluated on the chunk's grids only, with the free direction axes `s`/`t`
/// interchanged (equivalent under the driver's symmetrisation).
///
/// # Parameters
///
/// - `w` : shape `[ngrids]`.  Grid weights of the chunk.
/// - `prho` : shape `[ngrids, nvar, 3]` (g, x, t).  Spatial density derivative `d rho / dr` (= the
///   negated atom-sum of `drho`).
/// - `fxc` : shape `[ngrids, nvar, nvar]`.
/// - `drho` : shape `[ngrids, nvar, 3, natm]`.
///
/// # Returns
///
/// - `de_becke_atom_1` : shape `[3, 3, natm]` (t, s, A): the chunk atom's contribution, ordered for
///   a direct scatter into the last (B) axis of the `[3, 3, natm, natm]` accumulator; the (A, t)
///   <-> (B, s) symmetrisation applied by the driver after the accumulation restores the row
///   semantics.
pub fn get_de_becke_atom_1(w: TsrView, prho: TsrView, fxc: TsrView, drho: TsrView) -> Tsr {
    // fxc_drho [g, x, t, A] = sum_y fxc[g, x, y] drho[g, y, t, A]
    let fxc_drho = rt::vecdot(fxc.i((.., .., .., None, None)), drho.i((.., None, .., .., ..)), 2);
    // fold in the chunk grid weights
    let fxc_drho = fxc_drho * w.i((.., None, None, None));
    // de_becke_atom_1 [t, s, A] = sum_{g, x} fxc_drho[g, x, t, A] prho[g, x, s]
    rt::vecdot(fxc_drho.i((.., .., .., None, ..)), prho.i((.., .., None, .., None)), ([0, 1], [0, 1]))
}

/// `de_becke_atom_2`: `einsum("Bsg, xg, txg -> Bts", dw, vxc, prho)`,
/// evaluated with the free direction axes `s`/`t` interchanged (equivalent
/// under the driver's symmetrisation).
///
/// # Parameters
///
/// - `dw` : shape `[ngrids, 3, natm]` (g, t, A).  Grid-first Becke `dw`: the Fortran-order wrap of
///   the C-order `[A, t, g]` becke output buffer (see [`make_hessian_setup_chunk_becke`]).
/// - `vxc` : shape `[ngrids, nvar]`.
/// - `prho` : shape `[ngrids, nvar, 3]` (g, x, t).  Spatial density derivative `d rho / dr`.
///
/// # Returns
///
/// - `de_becke_atom_2` : shape `[3, 3, natm]` (t, s, A), for the last (B) axis scatter of the `[3,
///   3, natm, natm]` accumulator (see [`get_de_becke_atom_1`]).
pub fn get_de_becke_atom_2(dw: TsrView, vxc: TsrView, prho: TsrView) -> Tsr {
    // vdw2 [g, x, t, A] = vxc[g, x] dw[g, t, A]
    let vdw2 = dw.i((.., None, .., ..)) * vxc.i((.., .., None, None));
    // de_becke_atom_2 [t, s, A] = sum_{g, x} vdw2[g, x, t, A] prho[g, x, s]
    rt::vecdot(vdw2.i((.., .., .., None, ..)), prho.i((.., .., None, .., None)), ([0, 1], [0, 1]))
}

/// `de_becke_atom_3`: `einsum("g, xyg, syg, txg -> ts", w, fxc, prho, prho)`,
/// evaluated on the chunk's grids only — fills the `[atm_idx, atm_idx]`
/// diagonal block of the `[3, 3, natm, natm]` accumulator.
///
/// # Parameters
///
/// - `w` : shape `[ngrids]`.  Grid weights of the chunk.
/// - `prho` : shape `[ngrids, nvar, 3]` (g, x, t).  Spatial density derivative `d rho / dr`.
/// - `fxc` : shape `[ngrids, nvar, nvar]`.
///
/// # Returns
///
/// - `de_becke_atom_3` : shape `[3, 3]` (t, s).
pub fn get_de_becke_atom_3(w: TsrView, prho: TsrView, fxc: TsrView) -> Tsr {
    // fp [g, x, t] = sum_y fxc[g, x, y] prho[g, y, t]
    let fp = rt::vecdot(fxc.i((.., .., .., None)), prho.i((.., None, .., ..)), 2);
    // wprho [g, x, s]
    let wprho = &prho * w.i((.., None, None));
    // de_becke_atom_3 [t, s] = sum_{g, x} fp[g, x, t] wprho[g, x, s]
    rt::vecdot(fp.i((.., .., None, ..)), wprho.i((.., .., .., None)), ([0, 1], [0, 1]))
}

/// Contract a per-grid-atom skeleton-Vxc kernel into the chunk atom's Hessian
/// column (with the free direction axes interchanged): the full-AO sum enters
/// the `A == B` block, the per-atom AO-slice sums enter every `B` column of
/// the row atom.  The 0.5 factor of the `de_becke_vxc_*` grid-shift terms
/// lives here: only the `A == B` full-AO sum carries it — the driver's
/// symmetrisation doubles it back to 1 — while the slice sums are unscaled.
///
/// # Parameters
///
/// - `pvxc` : shape `[nao, 3, 3]` (u, t, s).  The (s, t)-interchanged per-grid-atom kernel, AO axis
///   leading (contiguous AO runs in column-major storage).
/// - `atm_idx` : atom that generated the chunk's grids.
/// - `aoslices` : shape `[natm, 4]`.
///
/// # Returns
///
/// - `de_pvxc` : shape `[3, 3, natm]` (t, s, B), for the last (B) axis scatter of the `[3, 3, natm,
///   natm]` accumulator; the (A, t) <-> (B, s) symmetrisation applied by the driver after the
///   accumulation restores the row semantics.
pub fn contract_pvxc(pvxc: TsrView, atm_idx: usize, aoslices: &[[usize; 4]]) -> Tsr {
    let natm = aoslices.len();
    let mut row: Tsr = rt::zeros(([3, 3, natm], pvxc.device()));

    *&mut row.i_mut((.., .., atm_idx)) += 0.5 * &pvxc.sum_axes(0);
    for (B, &[_, _, p0, p1]) in aoslices.iter().enumerate() {
        let slc = rt::slice!(p0, p1);
        *&mut row.i_mut((.., .., B)) -= pvxc.i((slc, .., ..)).sum_axes(0);
    }

    row
}

/// `de_becke_vxc_diag` / `de_becke_vxc_off`: the basis form of the
/// `de_becke_vxc` grid-shift terms, contracting the per-chunk `dao_vxc_*`
/// kernels (shared with `de_vxc_*`) with the density, which avoids building
/// the 2nd-order skeleton density derivatives altogether.
///
/// # Parameters
///
/// - `dao_vxc_diag` : shape `[nao, 6]`, output of [`make_dao_vxc_diag`].
/// - `dao_vxc_off` : shape `[nao, nao, 3, 3]` (u, v, t, s), output of [`make_dao_vxc_off`].
/// - `dm0` : shape `[nao, nao]`.
/// - `atm_idx` : atom that generated the chunk's grids.
/// - `aoslices` : shape `[natm, 4]`.
///
/// # Returns
///
/// - `de_becke_vxc_diag` : shape `[3, 3, natm]`, from `dao_vxc_diag` expanded to dense (3, 3)
///   pairs; the 0.5 factor of the grid-shift terms lives in [`contract_pvxc`].
/// - `de_becke_vxc_off` : shape `[3, 3, natm]`, from `dao_vxc_off` contracted with `dm0` on its
///   leading AO axis.
///
/// Both for the last (B) axis scatter (see [`contract_pvxc`]).
pub fn get_de_becke_vxc_parts(
    dao_vxc_diag: TsrView,
    dao_vxc_off: TsrView,
    dm0: TsrView,
    atm_idx: usize,
    aoslices: &[[usize; 4]],
) -> (Tsr, Tsr) {
    let nao = dao_vxc_diag.shape()[0];

    // pvxc_diag [nao, 3, 3] = dao_vxc_diag[IDX_PAIR_TS]; the symmetric
    // pairs make it invariant under (t, s), so it doubles as the interchanged
    // kernel
    const IDX_PAIR_TS: [usize; 9] = [0, 1, 2, 1, 3, 4, 2, 4, 5];
    let pvxc_diag: Tsr = dao_vxc_diag.index_select(1, IDX_PAIR_TS).into_shape([nao, 3, 3]);

    // pvxc_off[u, t, s] = sum_v dao_vxc_off[u, v, t, s] dm0[u, v]
    let pvxc_off: Tsr = rt::vecdot(dao_vxc_off, dm0, 1);
    (contract_pvxc(pvxc_diag.view(), atm_idx, aoslices), contract_pvxc(pvxc_off.view(), atm_idx, aoslices))
}

/* #endregion */

/* #region becke grid-shift parts: f1ao (CP-KS RHS) */

/// Symmetric XC-style Fock matrices from a stack of grid-weighted effective
/// fields: the standard on-grid Vxc build, batched over the trailing field
/// axis.
///
/// The field is not restricted to the vxc itself: feeding the grid-weighted
/// vxc produces the Vxc Fock matrix (`vmat_becke_dw` with the Becke `dw` rows
/// as the weights), while feeding an fxc kernel folded with a density
/// derivative produces first skeleton derivatives of the Fock matrix
/// (`vmat_fxc`, `vmat_becke_fxc`).  The build is linear in `wv`, and the
/// 0.5 factors of the Vxc build live here, not with the callers: the rho
/// channel enters both the bra and the ket AO factor (the AO-axis
/// symmetrisation supplies the ket half), and the tau channel enters the
/// kinetic-density pair products only once (those products are already
/// AO-symmetric).
///
/// # Parameters
///
/// - `xc_type` : density family.
/// - `ao` : shape `[ngrids, nao, ncomp]`; reads the leading `num_ao_comp` channels (1 for RHO, 4
///   for SIGMA/TAU).
/// - `wv` : shape `[ngrids, nvar, k]` (g, x, field).  Stack of grid-weighted effective fields,
///   weights already folded in and unscaled.  `k` is typically the 3 Cartesian directions of one
///   atom row; batching them keeps the per-field `[nao, ngrids] x [ngrids, nao]` GEMMs down to one
///   wide GEMM per row without a `[ngrids, nao, 3 * natm]`-sized intermediate.
///
/// # Returns
///
/// - `fock` : shape `[nao, nao, k]`.  Symmetric XC-style Fock matrices, one per field.
pub fn xc_fock_stack(xc_type: XCDenType, ao: TsrView, wv: TsrView) -> Tsr {
    let ngrids = ao.shape()[0];
    let nao = ao.shape()[1];
    let nk = wv.shape()[2];

    let mut wv: Tsr = wv.into_contig(ColMajor);
    *&mut wv.i_mut((.., O, ..)) *= 0.5;

    // aow [g, u, k] = sum_{c < nc} wv[:, c, :] ao[:, :, c]; nc = value + gradient AO channels
    // (rho components excluding tau, which is contracted separately below)
    let nc = xc_type.num_ao_comp();
    let aow = rt::vecdot(ao.i((.., .., ..nc)), wv.i((.., None, ..nc)), 2);

    // one wide GEMM over the k fields; x (v, u, k) = sum_g ao[g, v] aow[g, u, k]
    let x = (index!(ao, O).t() % aow.into_shape([ngrids, nao * nk])).into_shape([nao, nao, nk]);
    // AO-axis symmetrisation of the value/gradient part: fock[u, v, k] = x[v, u, k] + x[u, v, k]
    let mut fock: Tsr = &x + x.swapaxes(0, 1);

    if matches!(xc_type, TAU) {
        *&mut wv.i_mut((.., 4, ..)) *= 0.5;
        for j in 1..4 {
            let aow_j = ao.i((.., .., None, j)) * wv.i((.., None, 4, ..));
            let x_j = (index!(ao, j).t() % aow_j.into_shape([ngrids, nao * nk])).into_shape([nao, nao, nk]);
            fock += &x_j.swapaxes(0, 1);
        }
    }

    fock
}

/// f1ao-level Becke grid-shift parts of the skeleton Vxc Fock derivative: the
/// increment `vmat_becke_dw + vmat_becke_vxc + vmat_becke_fxc` that restores
/// translational invariance of `vmat_deriv1` (the DFT part of the CP-KS
/// right-hand side f1ao).
///
/// - `vmat_becke_dw` (weight part): [`xc_fock_stack`] per atom row, built with the Becke `dw[g, t,
///   A]` slices as the weight field; every grid of the chunk contributes to every atom's row.
/// - `vmat_becke_vxc` (functional part, Vxc): the chunk's `vmat_ip` symmetrised in AO — the chunk
///   holds one atom's grids, so `vmat_ip` already is the per-grid-atom kernel.
/// - `vmat_becke_fxc` (functional part, fxc): the fxc kernel folded with the spatial density
///   derivative `prho`, contracted as an [`xc_fock_stack`] on the chunk weights.
///
/// # Parameters
///
/// - `xc_type` : density family.
/// - `ao` : shape `[ngrids, nao, ncomp]`; reads the leading `num_ao_comp` channels (1 for RHO, 4
///   for SIGMA/TAU), through [`xc_fock_stack`].
/// - `vxc` : shape `[ngrids, nvar]`.
/// - `fxc` : shape `[ngrids, nvar, nvar]`.
/// - `prho` : shape `[ngrids, nvar, 3]` (g, x, t).  Spatial density derivative `d rho / dr`.
/// - `w` : shape `[ngrids]`.  Grid weights of the chunk.
/// - `dw` : shape `[ngrids, 3, natm]` (g, t, A).  Grid-first Becke `dw` (see
///   [`get_de_becke_atom_2`]).
/// - `vmat_ip` : shape `[nao, nao, 3]`, output of [`get_vmat_ip`].
///
/// # Returns
///
/// - `vmat_becke_dw` : shape `[nao, nao, 3, natm]`, filled on all rows.
/// - `vmat_becke_vxc` : shape `[nao, nao, 3]` — the chunk atom's row, scattered into the `[nao,
///   nao, 3, natm]` accumulator by the driver.
/// - `vmat_becke_fxc` : shape `[nao, nao, 3]`, scattered like `vmat_becke_vxc`.
#[allow(clippy::too_many_arguments)]
pub fn get_vmat_becke_parts(
    xc_type: XCDenType,
    ao: TsrView,
    vxc: TsrView,
    fxc: TsrView,
    prho: TsrView,
    w: TsrView,
    dw: TsrView,
    vmat_ip: TsrView,
) -> (Tsr, Tsr, Tsr) {
    let nao = ao.shape()[1];
    let natm = dw.shape()[2];
    let device = ao.device().clone();

    // dw part: XC-style Fock with the becke dw[·, ·, A] rows as weights (all rows)
    let mut vmat_becke_dw = rt::zeros(([nao, nao, 3, natm], &device));
    for A in 0..natm {
        // wv [g, x, t] = vxc[g, x] dw[g, t, A]
        let wv = vxc.i((.., .., None)) * dw.i((.., None, .., A));
        *&mut vmat_becke_dw.i_mut((.., .., .., A)) += &xc_fock_stack(xc_type, ao.view(), wv.view());
    }

    // vxc part: chunk's vmat_ip symmetrised in AO
    let vmat_becke_vxc = &vmat_ip + vmat_ip.swapaxes(0, 1);

    // fxc part: fxc folded with prho, contracted on the chunk weights
    // fxc_prho [g, x, t] = sum_y fxc[g, x, y] prho[g, y, t]
    let fxc_prho = rt::vecdot(fxc, prho.i((.., None, ..)), 2);
    let wv: Tsr = fxc_prho * w.i((.., None, None));
    let vmat_becke_fxc = xc_fock_stack(xc_type, ao.view(), wv.view());

    (vmat_becke_dw, vmat_becke_vxc, vmat_becke_fxc)
}

/* #endregion */

/* #region per-chunk evaluation */

/// Per-chunk evaluation of all skeleton ingredients with the grid-shift.  The
/// chunk must hold grids of the single atom `atm_idx` (ByAtom attribution) and
/// computes its own AO integrals through `ni.get_cached_ao`.
///
/// With `grid_shift = false` only the grid-fixed terms are evaluated; a chunk
/// attributed to [`NO_ATM`] (grids of no generating atom) is then legal, and
/// contributes only through the grid-fixed terms.
///
/// # Parameters
///
/// - `mol` : molecule (AO slices, dimensions).
/// - `xc_func_list` : list of `(scale, functional)` pairs.
/// - `ni` : numerical-integration driver restricted to the chunk's grids.
/// - `dm0` : shape `[nao, nao]`.  Reference density matrix in AO basis.
/// - `atm_idx` : atom that generated the chunk's grids.
/// - `tables` : precomputed molecular tables of the Becke partition (shared across chunks); must
///   be `Some` when `grid_shift` is set.
/// - `hardness` : Becke switch-function hardness.
/// - `grid_shift` : evaluate the Becke grid-shift terms.
///
/// # Returns
///
/// Map from key to the chunk's contribution.  Full-grid keys accumulate
/// across chunks by a plain sum; grid-atom keys carry only the chunk atom's
/// contribution and are scattered by [`make_hessian_setup_becke`]:
///
/// - Sum: `fxc [ngrids, nvar, nvar]` (disjoint grid ranges); `de_vxc_diag`, `de_vxc_off`, `de_fxc`,
///   `de_becke_full_1/2` `[3, 3, natm, natm]`; `vmat_ip [nao, nao, 3]`; `vmat_fxc`, `vmat_vxc`,
///   `vmat_deriv1`, `vmat_becke_dw` `[nao, nao, 3, natm]`.
/// - Scatter into column `B = atm_idx` (direction axes interchanged): `de_becke_atom_1/2`,
///   `de_becke_vxc_diag/off` `[3, 3, natm]`; `vmat_becke_vxc/fxc` `[nao, nao, 3]`.
/// - Scatter into the `[atm_idx, atm_idx]` diagonal block: `de_becke_atom_3` `[3, 3]`.
///
/// The `de_becke_*` / `vmat_becke_*` keys are present only with `grid_shift`.
#[allow(clippy::too_many_arguments)]
pub fn make_hessian_setup_chunk_becke(
    mol: &CInt,
    xc_func_list: &[(f64, LibXCFunctional)],
    ni: &mut NIMatmul,
    dm0: TsrView,
    atm_idx: usize,
    tables: Option<&BeckeMolTables>,
    hardness: usize,
    grid_shift: bool,
) -> HashMap<&'static str, Tsr> {
    assert!(atm_idx != NO_ATM || !grid_shift, "the grid-shift terms require atom-attributed grids (ByAtom)");
    let natm = mol.natm();
    let device = dm0.device().clone();
    let aoslices = mol.aoslice_by_atom();
    let xc_type = determine_den_type_from_list(&xc_func_list.iter().map(|(_, f)| f).collect_vec());

    // owned copies of the chunk's grid data; `ni` stays borrowed by the AO cache
    let grid_coords = ni.coords.clone();
    let quadrature_weights = ni.quadrature_weights.clone();
    let weights_data = ni.weights.clone();
    let ngrids = weights_data.len();

    // --- ao, rho, exc, vxc, fxc --- //

    let ao = ni.get_cached_ao(get_hess_ao_deriv(xc_type));
    let ncomp_ao_dm0 = get_hess_ncomp_ao_dm0(xc_type);
    let ao_dm0 = index!(ao, ..ncomp_ao_dm0) % &dm0;
    let (rho, exc, vxc, fxc) = get_rho_exc_vxc_fxc(xc_func_list, ao.view(), ao_dm0.view());

    let weights = rt::asarray((weights_data, &device));
    let wv = &weights * &vxc;
    let wf = &weights * &fxc;

    // --- drho, prho --- //

    let drho = get_drho(xc_type, ao.view(), ao_dm0.view(), &aoslices);
    // prho [ngrids, nvar, 3] = d rho / dr = -(drho summed over atoms)
    let prho = -drho.sum_axes(3);

    // --- without-becke parts --- //

    let de_fxc = get_de_fxc(wf.view(), drho.view());
    let dao_vxc_diag = make_dao_vxc_diag(xc_type, ao.view(), ao_dm0.view(), wv.view());
    let de_vxc_diag = get_de_vxc_diag(dao_vxc_diag.view(), &aoslices);
    let dao_vxc_off = make_dao_vxc_off(xc_type, ao.view(), wv.view());
    let de_vxc_off = get_de_vxc_off(dao_vxc_off.view(), dm0.view(), &aoslices);

    let vmat_ip = get_vmat_ip(xc_type, ao.view(), wv.view());
    let vmat_fxc = get_vmat_fxc(xc_type, ao.view(), drho.view(), wf.view());
    let vmat_vxc = get_vmat_vxc(vmat_ip.view(), &aoslices);
    // per-atom skeleton Vxc Fock derivative; both parts are already assembled
    // across the AO axes
    let vmat_deriv1 = &vmat_fxc + &vmat_vxc;

    let mut becke_entries: Vec<(&'static str, Tsr)> = Vec::new();
    if grid_shift {
        // --- becke partition: dw in full, ddw only through the cddw contraction --- //

        // cddw (nset = 1): the only ddw consumer is de_becke_full_2 = ddw . (exc * rho0)
        let cddw = (&exc * rho.i((.., 0))).into_vec();

        let tables = tables.expect("BeckeMolTables must be given when grid_shift is set");
        let boundaries = by_atom_chunk(natm, atm_idx, ngrids);
        let deriv_arg = BeckePartitionArg {
            output_w: false,
            output_dw: true,
            output_ddw: false,
            contract_w: None,
            contract_dw: None,
            contract_ddw: Some(&cddw),
        };
        let becke_result = becke_partition_with_tables(
            tables,
            &grid_coords,
            AtmIndices::ByAtom(&boundaries),
            &quadrature_weights,
            hardness,
            1024,
            2,
            Some(deriv_arg),
        );
        // dw flat is C-order [A, t, g] == Fortran-order [g, t, A]
        let dw = rt::asarray((becke_result.dw.unwrap(), [ngrids, 3, natm].f(), &device));

        // --- becke grid-shift parts --- //

        // de_becke_full_1: einsum("Atg, xg, Bsxg -> ABts", dw, vxc, drho);
        // [t, s, A, B] = sum_g dw[g, t, A] vxc_drho[g, s, B], where
        // vxc_drho [g, s, B] = sum_x vxc[g, x] drho[g, x, s, B]
        let de_becke_full_1 = {
            let vxc_drho = rt::vecdot(drho.view(), vxc.view(), 1);
            rt::vecdot(dw.i((.., .., None, .., None)), vxc_drho.i((.., None, .., None, ..)), 0)
        };

        // de_becke_full_2: einsum("AtBsg, g, g -> ABts", ddw, exc, rho[0]) via the cddw
        // contraction above (nset = 1), naturally symmetric;
        // ddc flat is C-order [A, t, B, s, iset] == Fortran-order [iset, s, B, t, A]
        let de_becke_full_2 = rt::asarray((becke_result.ddc.unwrap(), [3, natm, 3, natm].f(), &device))
            .transpose([2, 0, 3, 1])
            .into_contig(ColMajor);

        // grid-atom parts: compact tensors for the chunk atom's row (resp.
        // diagonal block); the scatter into `[3, 3, natm, natm]` is done by
        // `make_hessian_setup_becke`
        let de_becke_atom_1 = get_de_becke_atom_1(weights.view(), prho.view(), fxc.view(), drho.view());
        let de_becke_atom_2 = get_de_becke_atom_2(dw.view(), vxc.view(), prho.view());
        let de_becke_atom_3 = get_de_becke_atom_3(weights.view(), prho.view(), fxc.view());

        let (de_becke_vxc_diag, de_becke_vxc_off) =
            get_de_becke_vxc_parts(dao_vxc_diag.view(), dao_vxc_off.view(), dm0.view(), atm_idx, &aoslices);

        let (vmat_becke_dw, vmat_becke_vxc, vmat_becke_fxc) = get_vmat_becke_parts(
            xc_type,
            ao.view(),
            vxc.view(),
            fxc.view(),
            prho.view(),
            weights.view(),
            dw.view(),
            vmat_ip.view(),
        );

        becke_entries.extend([
            ("de_becke_full_1", de_becke_full_1),
            ("de_becke_full_2", de_becke_full_2),
            ("de_becke_atom_1", de_becke_atom_1),
            ("de_becke_atom_2", de_becke_atom_2),
            ("de_becke_atom_3", de_becke_atom_3),
            ("de_becke_vxc_diag", de_becke_vxc_diag),
            ("de_becke_vxc_off", de_becke_vxc_off),
            ("vmat_becke_dw", vmat_becke_dw),
            ("vmat_becke_vxc", vmat_becke_vxc),
            ("vmat_becke_fxc", vmat_becke_fxc),
        ]);
    }

    let mut result = HashMap::from([
        ("fxc", fxc),
        ("de_vxc_diag", de_vxc_diag),
        ("de_vxc_off", de_vxc_off),
        ("de_fxc", de_fxc),
        ("vmat_ip", vmat_ip),
        ("vmat_fxc", vmat_fxc),
        ("vmat_vxc", vmat_vxc),
        ("vmat_deriv1", vmat_deriv1),
    ]);
    result.extend(becke_entries);
    result
}

/* #endregion */

/* #region parallel driver */

/// `x + x.transpose(1, 0, 3, 2)` on a `[3, 3, natm, natm]` (tsAB) tensor: the
/// `(A, t) <-> (B, s)` symmetrisation.
fn symmetrize_ts_ab(x: Tsr) -> Tsr {
    &x + x.transpose([1, 0, 3, 2])
}

/// Parallel driver for all DFT skeleton ingredients with the grid-shift.
///
/// The atom-grouped grid is parallelized at chunk level only:
/// [`quad_split_by_atom`] at `nchunk` granularity produces `(atm_idx, start,
/// end)` work units that never cross an atom boundary, and each unit evaluates
/// [`make_hessian_setup_chunk_becke`] (its own AO integrals included) inside
/// one flat par_iter.  Atom grid sizes differ, so atom-sized work units would
/// not fill the thread pool; `nchunk`-sized pieces (as far as atom boundaries
/// allow) do.
///
/// With `grid_shift = false` the Becke terms are skipped entirely:
/// `de_xc_skeleton` equals `de_xc_skeleton_no_becke` and `vmat_deriv1_grid`
/// equals `vmat_deriv1`.
///
/// The Becke parameters are fixed by grid generation and not exposed at this
/// level: the hardness is [`BECKE_HARDNESS`] and the radii-adjustment table is
/// derived from the Bragg radii of the nuclear charges ([`gen_adjustment_factor`]).
///
/// # Parameters
///
/// - `mol` : molecule.
/// - `xc_func_list` : list of `(scale, functional)` pairs.
/// - `ni` : numerical-integration driver over the full grid; its `atm_idx` must be atom-grouped
///   (non-decreasing; see [`super::nimatmul::regroup_grids_by_atom`]) and, with `grid_shift`,
///   attribute every grid to an atom.
/// - `dm0` : shape `[nao, nao]`.  Reference density matrix in AO basis.
/// - `grid_shift` : evaluate the Becke grid-shift terms; requires full atom attribution of the
///   grids.
/// - `atm_list` : must be `None` or the full atom list.
/// - `verbose` : print per-chunk progress.
///
/// # Returns
///
/// - `result` : all keys of [`make_hessian_setup_chunk_becke`] accumulated over chunks (full-grid
///   keys summed, grid-atom keys scattered into the chunk atom's column of the last (B) axis; the
///   interchanged-direction chunks need no transposes), plus the assemblies:
///   `de_xc_skeleton_no_becke [3, 3, natm, natm]` = `de_vxc_diag + de_vxc_off + de_fxc`;
///   `de_xc_skeleton [3, 3, natm, natm]` with all `de_becke_*` grid-shift parts added
///   (translationally invariant); `vmat_deriv1_grid [nao, nao, 3, natm]` = `vmat_deriv1 +
///   vmat_becke_dw + vmat_becke_vxc + vmat_becke_fxc` (translationally invariant).  The keys
///   `de_becke_full_1/atom_1/atom_2/vxc_diag/vxc_off` are symmetrised under `(A, t) <-> (B, s)`;
///   `de_becke_full_2` is naturally symmetric.  With `grid_shift = false` the `de_becke_*` /
///   `vmat_becke_*` keys are absent.
/// - `timing` : wall-time progress entries.
pub fn make_hessian_setup_becke(
    mol: &CInt,
    xc_func_list: &[(f64, LibXCFunctional)],
    ni: &mut NIMatmul,
    dm0: TsrView,
    grid_shift: bool,
    atm_list: Option<&[usize]>,
    verbose: bool,
) -> (HashMap<&'static str, Tsr>, IndexMap<&'static str, f64>) {
    assert!(!xc_func_list.is_empty(), "xc_func_list must not be empty");
    assert!(
        atm_list.is_none() || atm_list.unwrap().len() == mol.natm(),
        "the DFT skeleton hessian currently requires the full atom list"
    );
    let natm = mol.natm();
    let nao = mol.nao();
    let ngrids = ni.weights.len();
    let nchunk = ni.nchunk;
    let xc_type = determine_den_type_from_list(&xc_func_list.iter().map(|(_, f)| f).collect_vec());
    let nvar = xc_type.num_nvar();
    let device = dm0.device().clone();

    // per-atom grid boundaries, deduced from the (atom-grouped) attribution
    let atm_quad_split = try_atm_quad_split(&ni.atm_idx, natm)
        .expect("the hessian driver requires atom-grouped grids (NIMatmul::atm_idx non-decreasing)");
    if grid_shift {
        assert!(
            atm_quad_split[natm] == ngrids,
            "the grid-shift terms require full atom attribution of the grids; \
             disable grid_shift_deriv or use standard generated grids"
        );
    }

    // molecular tables of the Becke partition, built once and shared by all
    // chunks; the radii table derives from the Bragg radii of the nuclear
    // charges (ECP-reduced charges would pick different radii, but grid
    // generation and the grid-shift must agree, which they do for plain
    // nuclear charges)
    let hardness = BECKE_HARDNESS;
    let tables = grid_shift.then(|| {
        let proton_charges: Vec<i32> = mol.atom_charges().iter().map(|&c| c as i32).collect_vec();
        let adjustment_factor = gen_adjustment_factor(&proton_charges);
        BeckeMolTables::new(&mol.atom_coords(), &adjustment_factor, 2)
    });

    let chunks = quad_split_by_atom(&atm_quad_split, ngrids, nchunk);
    let nchunks = chunks.len();

    let fxc_full: Tsr = rt::zeros(([ngrids, nvar, nvar], &device));
    let de_fxc: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_vxc_diag: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_vxc_off: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let vmat_ip: Tsr = rt::zeros(([nao, nao, 3], &device));
    let vmat_fxc: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_vxc: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_deriv1: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let de_becke_full_1: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_becke_full_2: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_becke_atom_1: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_becke_atom_2: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_becke_atom_3: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_becke_vxc_diag: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_becke_vxc_off: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let vmat_becke_dw: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_becke_vxc: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_becke_fxc: Tsr = rt::zeros(([nao, nao, 3, natm], &device));

    let timing = Arc::new(Mutex::new(IndexMap::from([("total", 0.0)])));
    let time_total = std::time::Instant::now();

    // serializes the `+=` reductions below
    let guard = Mutex::new(());

    // single-level parallelization by chunk; each task owns its `ni_chunk`
    // and its AO evaluation
    let ni = &*ni;
    let progress = AtomicUsize::new(0);

    chunks.into_par_iter().for_each(|(atm_idx, start, end)| {
        let mut ni_chunk = ni.split_batch(start, end);
        let result_chunk = make_hessian_setup_chunk_becke(
            mol,
            xc_func_list,
            &mut ni_chunk,
            dm0.view(),
            atm_idx,
            tables.as_ref(),
            hardness,
            grid_shift,
        );
        // fxc: disjoint grid ranges
        unsafe {
            let fxc_slc = fxc_full.i(start..end);
            let mut fxc_slc = fxc_slc.force_mut();
            fxc_slc.assign(&result_chunk["fxc"]);
        }
        // sum the full-grid keys, scatter the grid-atom keys into the chunk
        // atom's column of the last (B) axis
        unsafe {
            let _lock = guard.lock().unwrap();
            *&mut de_fxc.force_mut() += &result_chunk["de_fxc"];
            *&mut de_vxc_diag.force_mut() += &result_chunk["de_vxc_diag"];
            *&mut de_vxc_off.force_mut() += &result_chunk["de_vxc_off"];
            *&mut vmat_ip.force_mut() += &result_chunk["vmat_ip"];
            *&mut vmat_fxc.force_mut() += &result_chunk["vmat_fxc"];
            *&mut vmat_vxc.force_mut() += &result_chunk["vmat_vxc"];
            *&mut vmat_deriv1.force_mut() += &result_chunk["vmat_deriv1"];
            if grid_shift {
                *&mut de_becke_full_1.force_mut() += &result_chunk["de_becke_full_1"];
                *&mut de_becke_full_2.force_mut() += &result_chunk["de_becke_full_2"];
                *&mut vmat_becke_dw.force_mut() += &result_chunk["vmat_becke_dw"];
                *&mut de_becke_atom_1.i((Ellipsis, atm_idx)).force_mut() += &result_chunk["de_becke_atom_1"];
                *&mut de_becke_atom_2.i((Ellipsis, atm_idx)).force_mut() += &result_chunk["de_becke_atom_2"];
                *&mut de_becke_atom_3.i((Ellipsis, atm_idx, atm_idx)).force_mut() += &result_chunk["de_becke_atom_3"];
                *&mut de_becke_vxc_diag.i((Ellipsis, atm_idx)).force_mut() += &result_chunk["de_becke_vxc_diag"];
                *&mut de_becke_vxc_off.i((Ellipsis, atm_idx)).force_mut() += &result_chunk["de_becke_vxc_off"];
                *&mut vmat_becke_vxc.i((Ellipsis, atm_idx)).force_mut() += &result_chunk["vmat_becke_vxc"];
                *&mut vmat_becke_fxc.i((Ellipsis, atm_idx)).force_mut() += &result_chunk["vmat_becke_fxc"];
            }
        }
        let ichunk = progress.fetch_add(1, Ordering::Relaxed);
        timing.lock().unwrap().insert("total", time_total.elapsed().as_secs_f64());
        if verbose {
            println!(
                "In make_hessian_setup_becke, chunk {}/{} (atom {atm_idx}): grids {start}..{end}",
                ichunk + 1,
                nchunks,
            );
            println!("  Elapsed time from start (Wall time): {:.4} sec", timing.lock().unwrap()["total"]);
        }
    });

    // final assemblies
    let de_xc_skeleton_no_becke = &de_vxc_diag + &de_vxc_off + &de_fxc;

    let (de_xc_skeleton, vmat_deriv1_grid, becke_entries) = if grid_shift {
        // (A, t) <-> (B, s) symmetrisation for the becke parts; de_becke_full_2
        // is naturally symmetric
        let de_becke_full_1 = symmetrize_ts_ab(de_becke_full_1);
        let de_becke_atom_1 = symmetrize_ts_ab(de_becke_atom_1);
        let de_becke_atom_2 = symmetrize_ts_ab(de_becke_atom_2);
        let de_becke_vxc_diag = symmetrize_ts_ab(de_becke_vxc_diag);
        let de_becke_vxc_off = symmetrize_ts_ab(de_becke_vxc_off);

        let de_xc_skeleton = &de_xc_skeleton_no_becke
            + &de_becke_full_1
            + &de_becke_full_2
            + &de_becke_atom_1
            + &de_becke_atom_2
            + &de_becke_atom_3
            + &de_becke_vxc_diag
            + &de_becke_vxc_off;
        let vmat_deriv1_grid = &vmat_deriv1 + &vmat_becke_dw + &vmat_becke_vxc + &vmat_becke_fxc;
        let becke_entries = vec![
            ("de_becke_full_1", de_becke_full_1),
            ("de_becke_full_2", de_becke_full_2),
            ("de_becke_atom_1", de_becke_atom_1),
            ("de_becke_atom_2", de_becke_atom_2),
            ("de_becke_atom_3", de_becke_atom_3),
            ("de_becke_vxc_diag", de_becke_vxc_diag),
            ("de_becke_vxc_off", de_becke_vxc_off),
            ("vmat_becke_dw", vmat_becke_dw),
            ("vmat_becke_vxc", vmat_becke_vxc),
            ("vmat_becke_fxc", vmat_becke_fxc),
        ];
        (de_xc_skeleton, vmat_deriv1_grid, becke_entries)
    } else {
        (de_xc_skeleton_no_becke.clone(), vmat_deriv1.clone(), Vec::new())
    };

    let mut result = HashMap::from([
        ("fxc", fxc_full),
        ("de_vxc_diag", de_vxc_diag),
        ("de_vxc_off", de_vxc_off),
        ("de_fxc", de_fxc),
        ("vmat_ip", vmat_ip),
        ("vmat_fxc", vmat_fxc),
        ("vmat_vxc", vmat_vxc),
        ("vmat_deriv1", vmat_deriv1),
        ("de_xc_skeleton_no_becke", de_xc_skeleton_no_becke),
        ("de_xc_skeleton", de_xc_skeleton),
        ("vmat_deriv1_grid", vmat_deriv1_grid),
    ]);
    result.extend(becke_entries);

    let timing = timing.lock().unwrap().clone();
    (result, timing)
}

/* #endregion */

/* #region response */

pub fn get_rks_response_bra(
    ni: &mut NIMatmul,
    den_type: XCDenType,
    fxc_eff: TsrView,
    mo1_bra: TsrView,
    mocc: TsrView,
) -> (Tsr, IndexMap<&'static str, f64>) {
    let nao = mo1_bra.shape()[0];
    let nocc = mo1_bra.shape()[1];
    let mo1_bra_shape = mo1_bra.shape().to_vec();
    let mo1_bra = mo1_bra.reshape((nao, nocc, -1));
    let mo1_bra_list = mo1_bra.axes_iter(-1).collect_vec();

    let mut timing = IndexMap::new();

    let mut tic = |label: &'static str, t0: std::time::Instant| {
        let elapsed = t0.elapsed().as_secs_f64();
        timing.insert(label, elapsed);
    };

    let t0 = std::time::Instant::now();
    ni.get_cached_ao(den_type.num_ao_deriv());
    tic("ao", t0);

    let t0 = std::time::Instant::now();
    let rho1 = ni.make_rho_from_one_bra_mult_ket(mocc.view(), &mo1_bra_list, den_type);
    tic("rho1", t0);

    let t0 = std::time::Instant::now();
    let resp = ni.make_rks_fxc_pot_with_eff_bra_trans(fxc_eff, rho1.view(), mocc.view(), den_type);
    tic("resp", t0);

    // The 4.0 times is a trick of closed-shell coefficient
    let resp = 4.0 * resp.into_shape(mo1_bra_shape);
    (resp, timing)
}

pub fn get_rks_response_bra_batched(
    ni: &mut NIMatmul,
    den_type: XCDenType,
    fxc_eff: TsrView,
    mo1_bra: TsrView,
    mocc: TsrView,
    verbose: bool,
) -> (Tsr, IndexMap<&'static str, f64>) {
    let ngrids = ni.weights.len();
    let nbatch = ni.nbatch;
    let mo1_bra_shape = mo1_bra.shape().to_vec();
    let device = mo1_bra.device().clone();
    let mut resp = rt::zeros((mo1_bra_shape, &device));
    let mut timing = IndexMap::from([("ao", 0.0), ("rho1", 0.0), ("resp", 0.0), ("total", 0.0)]);

    let t0 = std::time::Instant::now();
    for start in (0..ngrids).step_by(nbatch) {
        let end = (start + nbatch).min(ngrids);
        let mut ni_batch = ni.split_batch(start, end);
        let (resp_batch, timing_batch) =
            get_rks_response_bra(&mut ni_batch, den_type, fxc_eff.i(start..end), mo1_bra.view(), mocc.view());
        resp += resp_batch;
        for (key, value) in timing_batch {
            *timing.get_mut(key).unwrap() += value;
        }
        let duration = t0.elapsed().as_secs_f64();
        timing.insert("total", duration);
        if verbose {
            println!("In get_rks_response_bra_batched, Batch {start}..{end}");
            println!("  Elapsed time from start (Wall time): {:.4} sec", duration);
        }
    }

    if verbose {
        println!("Finished get_rks_response_bra_batched");
        println!("  Total elapsed time (Wall time): {:.4} sec", timing["total"]);
        println!("  Timing breakdown (Wall time):");
        for (key, value) in timing.iter() {
            if *key != "total" {
                println!("  {key:>20}: {value:.4} sec");
            }
        }
    }

    (resp, timing)
}

/* #endregion */

/* #region legacy grid-fixed batched driver (temporary) */

// The grid-fixed batched driver predates the grid-shift chunk driver above.
// It is kept temporarily as the numerical reference of the transition (the
// tests compare its output against the chunk driver at `grid_shift = false`)
// and is scheduled for removal once the grid-shift driver is confirmed.

pub fn make_hessian_setup(
    mol: &CInt,
    xc_func_list: &[(f64, LibXCFunctional)],
    ni: &mut NIMatmul,
    dm0: TsrView,
    atm_list: Option<&[usize]>,
) -> (HashMap<&'static str, Tsr>, IndexMap<&'static str, f64>) {
    assert!(!xc_func_list.is_empty(), "xc_func_list must not be empty");
    let atm_list = atm_list.map_or_else(|| (0..mol.natm()).collect_vec(), |lst| lst.to_vec());
    // ao slices indexed by `atm_list`
    let aoslices_full = mol.aoslice_by_atom();
    let aoslices = atm_list.iter().map(|&iatm| aoslices_full[iatm]).collect_vec();
    // overall xc_type is the strictest (max nvar) one across the functionals;
    let xc_type = determine_den_type_from_list(&xc_func_list.iter().map(|(_, f)| f).collect_vec());

    let device = dm0.device().clone();
    let weights = rt::asarray((ni.weights.clone(), &device));

    let mut timing = IndexMap::new();

    let mut tic = |label: &'static str, t0: std::time::Instant| {
        let elapsed = t0.elapsed().as_secs_f64();
        timing.insert(label, elapsed);
    };

    // --- ao, rho, vxc, fxc --- //

    // ao      [ngrids, nao, ncomp]
    // ao_dm0  [ngrids, nao, ncomp_ao_dm0]
    // rho     [ngrids, nvar]
    // vxc     [ngrids, nvar]
    // fxc     [ngrids, nvar, nvar]

    let t0 = std::time::Instant::now();
    let ao = ni.get_cached_ao(get_hess_ao_deriv(xc_type));
    tic("ao", t0);

    let t0 = std::time::Instant::now();
    let ncomp_ao_dm0 = get_hess_ncomp_ao_dm0(xc_type);
    let ao_dm0 = index!(ao, ..ncomp_ao_dm0) % &dm0;
    tic("ao_dm0", t0);

    let t0 = std::time::Instant::now();
    let (rho, _exc, vxc, fxc) = get_rho_exc_vxc_fxc(xc_func_list, ao.view(), ao_dm0.view());
    let wv = &weights * &vxc;
    let wf = &weights * &fxc;
    tic("rho, vxc, fxc", t0);

    // --- drho --- //

    // drho    [ngrids, nvar, 3, natm]
    let t0 = std::time::Instant::now();
    let drho = get_drho(xc_type, ao.view(), ao_dm0.view(), &aoslices);
    tic("drho", t0);

    // --- de_fxc --- //

    // de_fxc  [3, 3, natm, natm]
    let t0 = std::time::Instant::now();
    let de_fxc = get_de_fxc(wf.view(), drho.view());
    tic("de_fxc", t0);

    // --- de_vxc_diag --- //

    // de_vxc_diag [3, 3, natm, natm]
    let t0 = std::time::Instant::now();
    let dao_vxc_diag = make_dao_vxc_diag(xc_type, ao.view(), ao_dm0.view(), wv.view());
    let de_vxc_diag = get_de_vxc_diag(dao_vxc_diag.view(), &aoslices);
    tic("de_vxc_diag", t0);

    // --- de_vxc_off --- //

    // de_vxc_off [3, 3, natm, natm]
    let t0 = std::time::Instant::now();
    let dao_vxc_off = make_dao_vxc_off(xc_type, ao.view(), wv.view());
    let de_vxc_off = get_de_vxc_off(dao_vxc_off.view(), dm0.view(), &aoslices);
    tic("de_vxc_off", t0);

    // --- vmat_ip --- //

    // vmat_ip [nao, nao, 3]
    let t0 = std::time::Instant::now();
    let vmat_ip = get_vmat_ip(xc_type, ao.view(), wv.view());
    tic("vmat_ip", t0);

    // --- vmat_deriv1 --- //

    // vmat_deriv1 [nao, nao, 3, natm]
    let t0 = std::time::Instant::now();
    let vmat_deriv1 = get_vmat_deriv1(xc_type, ao.view(), drho.view(), wf.view(), vmat_ip.view(), &aoslices);
    tic("vmat_deriv1", t0);

    let result = HashMap::from([
        ("rho", rho),
        ("vxc", vxc),
        ("fxc", fxc),
        ("de_fxc", de_fxc),
        ("de_vxc_diag", de_vxc_diag),
        ("de_vxc_off", de_vxc_off),
        ("vmat_ip", vmat_ip),
        ("vmat_deriv1", vmat_deriv1),
    ]);
    (result, timing)
}

pub fn make_hessian_setup_batched(
    mol: &CInt,
    xc_func_list: &[(f64, LibXCFunctional)],
    ni: &mut NIMatmul,
    dm0: TsrView,
    atm_list: Option<&[usize]>,
    verbose: bool,
) -> (HashMap<&'static str, Tsr>, IndexMap<&'static str, f64>) {
    // batch for grids
    // - except for rho, vxc, fxc; other tensors can be added (reduced)
    // - rho, vxc, fxc requires concation

    // outer iter: batch by nbatch (limit memory usage)
    // inner iter: batch by nchunk (for parallel)

    let ngrids = ni.weights.len();
    let nbatch = ni.nbatch;
    let nchunk = ni.nchunk;
    let device = dm0.device().clone();
    let xc_type = determine_den_type_from_list(&xc_func_list.iter().map(|(_, f)| f).collect_vec());
    let nvar = xc_type.num_nvar();
    let deriv_level = get_hess_ao_deriv(xc_type);
    let natm = atm_list.map_or_else(|| mol.natm(), |lst| lst.len());
    let nao = mol.nao();

    let rho: Tsr = rt::zeros(([ngrids, nvar], &device));
    let vxc: Tsr = rt::zeros(([ngrids, nvar], &device));
    let fxc: Tsr = rt::zeros(([ngrids, nvar, nvar], &device));
    let de_fxc: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_vxc_diag: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_vxc_off: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let vmat_ip: Tsr = rt::zeros(([nao, nao, 3], &device));
    let vmat_deriv1: Tsr = rt::zeros(([nao, nao, 3, natm], &device));

    let timing = Arc::new(Mutex::new(IndexMap::from([
        ("ao", 0.0),
        ("ao_dm0", 0.0),
        ("rho, vxc, fxc", 0.0),
        ("drho", 0.0),
        ("de_fxc", 0.0),
        ("de_vxc_diag", 0.0),
        ("de_vxc_off", 0.0),
        ("vmat_ip", 0.0),
        ("vmat_deriv1", 0.0),
        ("total", 0.0),
    ])));
    let time_total = std::time::Instant::now();

    // atomic guard to avoid racing write
    let guard = Mutex::new(());

    for start_batch in (0..ngrids).step_by(nbatch) {
        let end_batch = (start_batch + nbatch).min(ngrids);

        // handle AO integral at batch level
        let t0 = std::time::Instant::now();
        let mut ni_batch = ni.split_batch(start_batch, end_batch);
        ni_batch.get_cached_ao(deriv_level);
        {
            let mut timing = timing.lock().unwrap();
            timing["ao"] += t0.elapsed().as_secs_f64();
        }

        // other parts can be parallelized by chunk, with atomic guard for reduction
        (start_batch..end_batch).into_par_iter().step_by(nchunk).for_each(|start| {
            let end = (start + nchunk).min(end_batch);
            let mut ni_chunk = ni_batch.split_batch(start - start_batch, end - start_batch);
            let (result_chunk, timing_chunk) =
                make_hessian_setup(mol, xc_func_list, &mut ni_chunk, dm0.view(), atm_list);
            // fill rho, vxc, fxc
            // this is assumed to be not racing, so no guard at here
            unsafe {
                let rho_slc = rho.i(start..end);
                let vxc_slc = vxc.i(start..end);
                let fxc_slc = fxc.i(start..end);
                let mut rho_slc = rho_slc.force_mut();
                let mut vxc_slc = vxc_slc.force_mut();
                let mut fxc_slc = fxc_slc.force_mut();
                rho_slc.assign(&result_chunk["rho"]);
                vxc_slc.assign(&result_chunk["vxc"]);
                fxc_slc.assign(&result_chunk["fxc"]);
            }
            // add up other tensors
            unsafe {
                let lock = guard.lock().unwrap();
                *&mut de_fxc.force_mut() += &result_chunk["de_fxc"];
                *&mut de_vxc_diag.force_mut() += &result_chunk["de_vxc_diag"];
                *&mut de_vxc_off.force_mut() += &result_chunk["de_vxc_off"];
                *&mut vmat_ip.force_mut() += &result_chunk["vmat_ip"];
                *&mut vmat_deriv1.force_mut() += &result_chunk["vmat_deriv1"];
                drop(lock);
            }
            // add up timing
            {
                let mut timing = timing.lock().unwrap();
                for (key, value) in timing_chunk {
                    *timing.get_mut(key).unwrap() += value;
                }
            }
        });

        // fill total time outside parallel loop
        {
            let mut timing = timing.lock().unwrap();
            timing.insert("total", time_total.elapsed().as_secs_f64());
        }

        // verbose print of timing and batch info
        if verbose {
            let timing = timing.lock().unwrap();
            println!("In make_hessian_setup_batched, Batch {start_batch}..{end_batch}");
            println!("  Elapsed time from start (Wall time): {:.4} sec", timing["total"]);
        }
    }

    let timing = timing.lock().unwrap();
    if verbose {
        println!("Finished make_hessian_setup_batched");
        println!("  Total elapsed time (Wall time): {:.4} sec", timing["total"]);
        println!("  Timing breakdown (CPU time for others, Wall time for `ao`):");
        for (key, value) in timing.iter() {
            if *key != "total" {
                println!("  {key:>20}: {value:.4} sec");
            }
        }
    }

    let result = HashMap::from([
        ("rho", rho),
        ("vxc", vxc),
        ("fxc", fxc),
        ("de_fxc", de_fxc),
        ("de_vxc_diag", de_vxc_diag),
        ("de_vxc_off", de_vxc_off),
        ("vmat_ip", vmat_ip),
        ("vmat_deriv1", vmat_deriv1),
    ]);

    (result, timing.clone())
}

/* #endregion */

/* #region final implementation of RKS Hessian */

/// RKS Hessian XC component with the Becke grid-shift: [`RHessElecInteractAPI`]
/// with `make_skeleton_hess` returning the translationally invariant
/// `de_xc_skeleton` and `get_deriv1_ao` the translationally invariant
/// `vmat_deriv1_grid` (their grid-fixed values when `grid_shift` is off).
///
/// The grid data of `ni` carries the atom attribution (`NIMatmul::atm_idx`,
/// `NIMatmul::quadrature_weights`); grids must be atom-grouped (the ByAtom
/// attribution scheme of `becke_partitioning_deriv`).  The Becke parameters
/// (hardness, radii-adjustment table) are fixed by grid generation and not
/// exposed here.
pub struct RHessKSNIMatmul<'a> {
    /// Molecule.
    pub mol: CInt,
    /// List of `(scale, functional)` pairs of the XC functional.
    pub xc_func_list: Vec<(f64, LibXCFunctional)>,
    /// Numerical-integration driver over the (atom-grouped) Hessian grid.
    pub ni: NIMatmul<'a>,
    /// Optional separate grid for the CP-KS response; `None` reuses `ni`.
    pub ni_cpks: Option<NIMatmul<'a>>,
    /// Evaluate the Becke grid-shift terms; requires full atom attribution of
    /// the grids.
    pub grid_shift: bool,
    /// Print per-chunk progress of the Hessian setup.
    pub verbose: bool,
    /// Intermediates of the Hessian setup: all keys of
    /// [`make_hessian_setup_becke`] (with `fxc` renamed to `cpks_fxc` unless a
    /// CP-KS-specific grid is given), plus `mo_coeff [nao, nmo]`, `mo_occ
    /// [nmo]` from [`RHessElecInteractAPI::make_response_preparation`].
    pub intmd: HashMap<String, Tsr>,
}

impl<'a> RHessKSNIMatmul<'a> {
    /// Create a new RKS Hessian object.
    ///
    /// # Parameters
    ///
    /// - `mol` : molecule.
    /// - `xc_func_list` : list of `(scale, functional)` pairs.
    /// - `ni` : numerical-integration driver over the atom-grouped grid.
    /// - `grid_shift` : evaluate the Becke grid-shift terms.
    /// - `verbose` : print progress.
    pub fn new(
        mol: &CInt,
        xc_func_list: Vec<(f64, LibXCFunctional)>,
        ni: NIMatmul<'a>,
        grid_shift: bool,
        verbose: bool,
    ) -> Self {
        Self { mol: mol.clone(), xc_func_list, ni, ni_cpks: None, grid_shift, verbose, intmd: HashMap::new() }
    }

    /// Attach a dedicated CP-KS numerical-integration grid (`ni_cpks`).
    ///
    /// When set, the CP-KS response (`get_response_bra`) is evaluated on this grid instead of the
    /// skeleton grid, and `cpks_vxc` / `cpks_fxc` are recomputed on it during
    /// [`make_response_preparation`](Self::make_response_preparation).
    pub fn set_ni_cpks(mut self, ni_cpks: NIMatmul<'a>) -> Self {
        self.ni_cpks = Some(ni_cpks);
        self
    }

    /// Perform the Hessian setup for RKS calculations.
    ///
    /// `fxc` is stored as `cpks_fxc` (unless a CP-KS-specific grid is given)
    /// for the response; `de_xc_skeleton` and `vmat_deriv1_grid` are the main
    /// results.
    pub fn make_hessian_setup(&mut self, mo_coeff: TsrView, mo_occ: TsrView, atm_list: Option<&[usize]>) {
        // run RKS hessian setup
        let dm0 = get_dm0_restricted(mo_coeff, mo_occ);
        let (result, _timing) = make_hessian_setup_becke(
            &self.mol,
            &self.xc_func_list,
            &mut self.ni,
            dm0.view(),
            self.grid_shift,
            atm_list,
            self.verbose,
        );

        // handling intermediates and results
        for (key, val) in result.into_iter() {
            if key == "fxc" {
                // fxc storage is actually for cp-ks.
                // If `ni_cpks` is not specified, then we can use the fxc from the hessian setup
                // for cp-ks as well.
                if self.ni_cpks.is_none() {
                    self.intmd.insert("cpks_fxc".to_string(), val);
                }
            } else {
                self.intmd.insert(key.to_string(), val);
            }
        }
    }

    /// Check if the Hessian setup is done by verifying the presence of the
    /// "de_xc_skeleton" key in the intermediate results.
    pub fn is_hessian_setup_done(&self) -> bool {
        self.intmd.contains_key("de_xc_skeleton")
    }
}

impl<'a> HessUtilAPI for RHessKSNIMatmul<'a> {}

impl<'a> RHessElecInteractAPI for RHessKSNIMatmul<'a> {
    fn make_skeleton_hess(&mut self, mo_coeff: TsrView, mo_occ: TsrView, atm_list: Option<&[usize]>) -> Tsr {
        if !self.is_hessian_setup_done() {
            self.make_hessian_setup(mo_coeff, mo_occ, atm_list);
        }
        self.intmd["de_xc_skeleton"].to_owned()
    }

    fn get_deriv1_ao(&mut self, mo_coeff: TsrView, mo_occ: TsrView, atm_list: Option<&[usize]>) -> Tsr {
        if !self.is_hessian_setup_done() {
            self.make_hessian_setup(mo_coeff, mo_occ, atm_list);
        }
        self.intmd["vmat_deriv1_grid"].to_owned()
    }

    fn make_response_preparation(&mut self, mo_coeff: TsrView, mo_occ: TsrView) {
        self.intmd.insert("mo_coeff".to_string(), mo_coeff.into_contig(ColMajor));
        self.intmd.insert("mo_occ".to_string(), mo_occ.into_contig(ColMajor));

        // When a dedicated CP-KS grid is set, `cpks_vxc` / `cpks_fxc` were NOT stored during
        // `make_hessian_setup` (the skeleton grid's vxc/fxc live on a different grid and must not
        // be reused). Recompute them here on the CP-KS grid from the ground-state density, using
        // the lean [`make_cpks_vxc_fxc`] (no skeleton intermediates, minimal AO derivative order,
        // density formed from occupied MOs via a bra-ket contraction rather than a full dm0).
        if let Some(ni_cpks) = self.ni_cpks.as_mut() {
            let mo_coeff = self.intmd["mo_coeff"].view();
            let mo_occ = self.intmd["mo_occ"].view();
            let (vxc, fxc) = make_cpks_vxc_fxc(&self.xc_func_list, ni_cpks, mo_coeff, mo_occ);
            self.intmd.insert("cpks_vxc".to_string(), vxc);
            self.intmd.insert("cpks_fxc".to_string(), fxc);
        }
    }

    fn get_response_bra(&mut self, bra: TsrView) -> Tsr {
        let ni_cpks = self.ni_cpks.as_mut().unwrap_or(&mut self.ni);
        let mo_coeff = self.intmd.get("mo_coeff").unwrap();
        let mo_occ = self.intmd.get("mo_occ").unwrap();
        let fxc_eff = self.intmd.get("cpks_fxc").unwrap();
        let occidx = mo_occ.view().greater(0).into_vec();
        let mocc = mo_coeff.bool_select(-1, &occidx);

        let (resp, _timing) = get_rks_response_bra_batched(
            ni_cpks,
            determine_den_type_from_list(&self.xc_func_list.iter().map(|(_, f)| f).collect_vec()),
            fxc_eff.view(),
            bra,
            mocc.view(),
            self.verbose,
        );
        resp
    }
}

/* #endregion */
