// see also pyhessref/nimatmul/rks.py

use super::prelude::*;
use crate::analdrv::prelude::*;

use XCDenType::*;

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

pub const fn get_hess_ao_deriv(xc_type: XCDenType) -> usize {
    match xc_type {
        RHO => 2,
        SIGMA => 3,
        TAU => 3,
        LAPL => unimplemented!(),
    }
}

pub const fn get_hess_ncomp_ao_dm0(xc_type: XCDenType) -> usize {
    match xc_type {
        RHO => 1,
        SIGMA => 4,
        TAU => 4,
        LAPL => unimplemented!(),
    }
}

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

/* #region basic pure functions of skeleton hessian evaluation */

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
    let (rho, vxc, fxc) = get_rho_vxc_fxc(xc_func_list, ao.view(), ao_dm0.view());
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
    let de_vxc_diag = get_de_vxc_diag(xc_type, ao.view(), ao_dm0.view(), wv.view(), &aoslices);
    tic("de_vxc_diag", t0);

    // --- de_vxc_off --- //

    // de_vxc_off [3, 3, natm, natm]
    let t0 = std::time::Instant::now();
    let de_vxc_off = get_de_vxc_off(xc_type, ao.view(), dm0.view(), wv.view(), &aoslices);
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

pub fn get_rho_vxc_fxc(xc_func_list: &[(f64, LibXCFunctional)], ao: TsrView, ao_dm0: TsrView) -> (Tsr, Tsr, Tsr) {
    // see also pyhessref/nimatmul/rks.py

    assert!(!xc_func_list.is_empty(), "xc_func_list must not be empty");
    // overall xc_type is the strictest one across the functionals; partial
    // contributions from looser families are added into the leading slice.
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

    let (vxc, fxc) = eval_vxc_fxc_from_rho(xc_func_list, rho.view());
    (rho, vxc, fxc)
}

/// Evaluate the (spin-unpolarized) XC potential `vxc` and kernel `fxc` from a given ground-state
/// `rho`, by summing each sub-functional's `libxc_eval_eff` (deriv = 2) contribution.
///
/// `rho` : shape `[ngrids, nvar]`, where `nvar` is the strictest density-variable count across the
/// functional list. Extracted from [`get_rho_vxc_fxc`] so that callers which already have `rho`
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
/// Compared to [`make_hessian_setup_batched`], this:
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

pub fn get_drho(xc_type: XCDenType, ao: TsrView, ao_dm0: TsrView, aoslices: &[[usize; 4]]) -> Tsr {
    // direct transformation of `pyhessref/nimatmul/rks.py`, function `_make_drho`

    let ngrids = ao.shape()[0];
    let nvar = xc_type.num_nvar();
    let natm = aoslices.len();
    let device = ao.device().clone();

    let mut drho = rt::zeros(([ngrids, nvar, 3, natm], &device));

    // components: [rho_var, t_direction, cbra, cket]
    // tuple that contribute to each rho component
    // for symmetric components, result is multiplied by 2 at the end

    // RHO part
    let mut components = vec![(0, 0, X, O), (0, 1, Y, O), (0, 2, Z, O)];
    // SIGMA part
    if matches!(xc_type, SIGMA | TAU) {
        // bra deriv2, ket 0
        let sigma_bra2_ket0 = [
            [(1, 0, XX, O), (2, 0, XY, O), (3, 0, XZ, O)],
            [(1, 1, YX, O), (2, 1, YY, O), (3, 1, YZ, O)],
            [(1, 2, ZX, O), (2, 2, ZY, O), (3, 2, ZZ, O)],
        ];
        components.extend(sigma_bra2_ket0.concat());
        // bra deriv1, ket deriv1
        let sigma_bra1_ket1 = [
            [(1, 0, X, X), (2, 0, X, Y), (3, 0, X, Z)],
            [(1, 1, Y, X), (2, 1, Y, Y), (3, 1, Y, Z)],
            [(1, 2, Z, X), (2, 2, Z, Y), (3, 2, Z, Z)],
        ];
        components.extend(sigma_bra1_ket1.concat());
    }
    // TAU part
    if matches!(xc_type, TAU) {
        // bra deriv2, ket deriv1
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

    // RHO and SIGMA part multiply factor 2
    // TAU part does not multiply factor 2 because of the 0.5 factor in rho_tau
    match xc_type {
        RHO => *&mut drho.i_mut((.., 0..1)) *= 2.0,
        SIGMA | TAU => *&mut drho.i_mut((.., 0..4)) *= 2.0,
        LAPL => unimplemented!(),
    }
    drho
}

pub fn get_de_fxc(wf: TsrView, drho: TsrView) -> Tsr {
    // wf , drho, drho -> de_fxc
    // gxy, gxtA, gysB -> tsAB

    let [ngrids, nvar, _, natm] = drho.shape().iter().cloned().collect_array().unwrap();

    // wf    * drho  -> tmp
    // gxy.. * gx.tA -> gytA
    let tmp1 = rt::vecdot(wf.i((.., .., .., None, None)), drho.i((.., .., None, .., ..)), 1);

    // tmp1  * drho -> tmp2
    // gytA  * gysB -> tAsB
    let tmp1 = tmp1.reshape([ngrids * nvar, natm * 3]);
    let drho = drho.reshape([ngrids * nvar, natm * 3]);
    let tmp2 = tmp1.t() % drho;

    // transpose tmp2 to get de_fxc
    // tAsB -> tsAB
    tmp2.reshape([3, natm, 3, natm]).transpose([0, 2, 1, 3]).into_contig(ColMajor)
}

pub fn get_de_vxc_diag(xc_type: XCDenType, ao: TsrView, ao_dm0: TsrView, wv: TsrView, aoslices: &[[usize; 4]]) -> Tsr {
    const TRIPLE_SIGMA_DIAG: [[usize; 3]; 6] =
        [[XXX, XXY, XXZ], [XXY, XYY, XYZ], [XXZ, XYZ, XZZ], [XYY, YYY, YYZ], [XYZ, YYZ, YZZ], [XZZ, YZZ, ZZZ]];
    const TRIPLE_TAU_DIAG: [[usize; 6]; 3] =
        [[XXX, XXY, XXZ, XYY, XYZ, XZZ], [XXY, XYY, XYZ, YYY, YYZ, YZZ], [XXZ, XYZ, XZZ, YYZ, YZZ, ZZZ]];

    let natm = aoslices.len();
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

    // reduce ao-wise contributions to atom-wise contributions
    let mut de_vxc_diag = rt::zeros(([6, natm, natm], &device));
    for (A, &[_, _, p0, p1]) in aoslices.iter().enumerate() {
        let slc = rt::slice!(p0, p1);
        de_vxc_diag.i_mut((.., A, A)).assign(dao_vxc_diag.i(slc).sum_axes(0));
    }
    // symmetrize and expand de_vxc_diag from [6, natm, natm] to [3, 3, natm, natm]
    de_vxc_diag.index_select(0, [0, 1, 2, 1, 3, 4, 2, 4, 5]).into_shape([3, 3, natm, natm])
}

pub fn get_de_vxc_off(xc_type: XCDenType, ao: TsrView, dm0: TsrView, wv: TsrView, aoslices: &[[usize; 4]]) -> Tsr {
    let natm = aoslices.len();
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

    // add transposition
    let dao_vxc_off = &dao_vxc_off + dao_vxc_off.transpose([1, 0, 3, 2]);

    // reduce ao-wise contributions to atom-wise contributions
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

pub fn get_vmat_ip(xc_type: XCDenType, ao: TsrView, wv: TsrView) -> Tsr {
    // direct transformation of `pyhessref/nimatmul/rks.py`, function `_vmat_ip`

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

    // GGA + MGGA share the same SIGMA structure
    if matches!(xc_type, SIGMA | TAU) {
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

pub fn get_vmat_deriv1(
    xc_type: XCDenType,
    ao: TsrView,
    drho: TsrView,
    wf: TsrView,
    vmat_ip: TsrView,
    aoslices: &[[usize; 4]],
) -> Tsr {
    let natm = aoslices.len();
    let nao = ao.shape()[1];

    let mut vmat_deriv1: Tsr = rt::zeros(([nao, nao, 3, natm], ao.device()));

    for (A, &[_, _, p0, p1]) in aoslices.iter().enumerate() {
        let slc = rt::slice!(p0, p1);

        if matches!(xc_type, RHO) {
            // wf_rho: [ngrids, 3]
            let wf_rho: Tsr = 0.5 * index!(wf, O, O) * drho.i((.., O, .., A));
            for t in 0..3 {
                let aow = index!(wf_rho, t) * index!(ao, O);
                index_mut!(vmat_deriv1, t, A).matmul_from(aow.t(), index!(ao, O), 1.0, 1.0);
            }
        }

        if matches!(xc_type, SIGMA | TAU) {
            // wf_rho = np.einsum("gxy, gxt -> gyt", wf, drho[A])
            // wf_rho[:, 0] *= 0.5, wf_rho[:, 4] *= 0.25
            let mut wf_rho = rt::vecdot(&wf, drho.i((.., .., None, .., A)), 1);
            *&mut wf_rho.i_mut((.., 0)) *= 0.5;
            if matches!(xc_type, TAU) {
                *&mut wf_rho.i_mut((.., 4)) *= 0.25;
            }
            for t in 0..3 {
                let aow = rt::vecdot(wf_rho.i((.., None, ..4, t)), ao.i((.., .., ..4)), 2);
                index_mut!(vmat_deriv1, t, A).matmul_from(aow.t(), index!(ao, O), 1.0, 1.0);
            }

            if matches!(xc_type, TAU) {
                for r in [X, Y, Z] {
                    for t in 0..3 {
                        let aow = wf_rho.i((.., 4, t)) * index!(ao, r);
                        index_mut!(vmat_deriv1, t, A).matmul_from(aow.t(), index!(ao, r), 1.0, 1.0);
                    }
                }
            }
        }

        *&mut vmat_deriv1.i_mut((slc, .., .., A)) -= vmat_ip.i((slc, .., ..));
    }

    &vmat_deriv1 + vmat_deriv1.swapaxes(0, 1)
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

/* #endregion */

/* #region parallel/batch wrapper */

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

/* #region final implementation of RKS Hessian */

pub struct RHessKSNIMatmul<'a> {
    pub mol: CInt,
    pub xc_func_list: Vec<(f64, LibXCFunctional)>,
    pub ni: NIMatmul<'a>,
    pub ni_cpks: Option<NIMatmul<'a>>,
    pub verbose: bool,
    pub intmd: HashMap<String, Tsr>,
    pub result: HashMap<String, Tsr>,
}

impl<'a> RHessKSNIMatmul<'a> {
    pub fn new(mol: &CInt, xc_func_list: Vec<(f64, LibXCFunctional)>, ni: NIMatmul<'a>, verbose: bool) -> Self {
        Self {
            mol: mol.clone(),
            xc_func_list,
            ni,
            ni_cpks: None,
            verbose,
            intmd: HashMap::new(),
            result: HashMap::new(),
        }
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
    /// Some intermediates:
    /// - `vxc`, `fxc`: These will be renamed to `cpks_vxc` and `cpks_fxc` if CP-KS specific grid
    ///   (`ni_cpks`) is not specified by user. If CP-KS specific grid is specified, then `cpks_vxc`
    ///   and `cpks_fxc` will be computed after in trait implementation
    ///   [`RHessElecInteractAPI::make_response_preparation`].
    /// - `rho`: discarded.
    /// - `de_fxc`, `de_vxc_diag`, `de_vxc_off`, `vmat_ip`, `vmat_deriv1`: These are the main
    ///   results of the Hessian setup and will be stored in `self.intmd` for later use in the
    ///   Hessian calculation.
    pub fn make_hessian_setup(&mut self, mo_coeff: TsrView, mo_occ: TsrView, atm_list: Option<&[usize]>) {
        // run RKS hessian setup
        let dm0 = get_dm0_restricted(mo_coeff, mo_occ);
        let (result, _timing) =
            make_hessian_setup_batched(&self.mol, &self.xc_func_list, &mut self.ni, dm0.view(), atm_list, self.verbose);

        // handling intermediates and results
        for (key, val) in result.into_iter() {
            if key == "vxc" || key == "fxc" {
                // vxc, fxc storation is actually for cp-ks.
                // If `ni_cpks` is not specified, then we can use the vxc/fxc from the hessian setup for
                // cp-ks as well.
                if self.ni_cpks.is_none() {
                    let key_to_store = format!("cpks_{key}");
                    self.intmd.insert(key_to_store, val);
                }
            } else if ["rho"].contains(&key) {
                // some keys to be discarded
                continue;
            } else {
                self.intmd.insert(key.to_string(), val);
            }
        }
    }

    /// Check if the Hessian setup is done by verifying the presence of the "de_fxc" key in the
    /// intermediate results.
    pub fn is_hessian_setup_done(&self) -> bool {
        self.intmd.contains_key("de_fxc")
    }
}

impl<'a> HessUtilAPI for RHessKSNIMatmul<'a> {}

impl<'a> RHessElecInteractAPI for RHessKSNIMatmul<'a> {
    fn make_skeleton_hess(&mut self, mo_coeff: TsrView, mo_occ: TsrView, atm_list: Option<&[usize]>) -> Tsr {
        if !self.is_hessian_setup_done() {
            self.make_hessian_setup(mo_coeff, mo_occ, atm_list);
        }
        &self.intmd["de_fxc"] + &self.intmd["de_vxc_diag"] + &self.intmd["de_vxc_off"]
    }

    fn get_deriv1_ao(&mut self, mo_coeff: TsrView, mo_occ: TsrView, atm_list: Option<&[usize]>) -> Tsr {
        if !self.is_hessian_setup_done() {
            self.make_hessian_setup(mo_coeff, mo_occ, atm_list);
        }
        self.intmd["vmat_deriv1"].to_owned()
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
