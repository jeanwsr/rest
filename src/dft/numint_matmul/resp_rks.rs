//! Response (Fock and response matrix) implementation for the RKS numint-matmul XC component.
//!
//! The response object [`RRespKSNIMatmul`] is a standalone object for the fock/response
//! functionality of the RKS numerical-integration XC contribution, independent of the hessian
//! machinery of `RHessKSNIMatmul` (which owns its own grid, built from the same grid data).
//!
//! The two grids are separated by purpose: the fock path (`get_fock_rdm`/`get_fock_coeff`) and
//! the rdm-form response (`get_response_rdm`, a Lagrangian-type evaluation of the fxc kernel
//! upon a full density, as needed by the generalized-Fock machinery of post-SCF methods) always
//! evaluate on the common (large) grid `ni`, while the response path
//! (`make_response_preparation`/`get_response_bra`) uses the small response grid `ni_resp` when
//! one is attached. The returned fock is the DFT numerical-integration XC contribution only,
//! never the full SCF fock.
//!
//! Main-thread only (not `Send`/`Sync`).

use super::prelude::*;
use crate::analdrv::prelude::*;

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
/// Compared to [`make_hessian_setup_becke`](super::hess_rks::make_hessian_setup_becke), this:
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

/// Response (fock/response matrix) object for the RKS numint-matmul XC contribution.
pub struct RRespKSNIMatmul<'a> {
    /// List of `(scale, functional)` pairs of the XC functional.
    pub xc_func_list: Vec<(f64, LibXCFunctional)>,
    /// Common (large) numerical-integration grid, used by the fock path.
    pub ni: NIMatmul<'a>,
    /// Optional separate small grid for the response path; `None` reuses `ni`.
    pub ni_resp: Option<NIMatmul<'a>>,
    /// Print progress of the response evaluation.
    pub verbose: bool,
    /// Response intermediates: `mo_coeff [nao, nmo]` and `mo_occ [nmo]` from
    /// [`RRespAPI::make_response_preparation`], `cpks_vxc [ngrids, nvar]` and
    /// `cpks_fxc [ngrids, nvar, nvar]` on the selected response grid, and `fxc_common_grid
    /// [ngrids, nvar, nvar]` on the common grid.
    pub intmd: HashMap<String, Tsr>,
}

impl<'a> RRespKSNIMatmul<'a> {
    /// Create a new RKS response object.
    ///
    /// # Parameters
    ///
    /// - `xc_func_list` : list of `(scale, functional)` pairs.
    /// - `ni` : numerical-integration driver over the common (large) grid.
    /// - `verbose` : print progress.
    pub fn new(xc_func_list: Vec<(f64, LibXCFunctional)>, ni: NIMatmul<'a>, verbose: bool) -> Self {
        Self { xc_func_list, ni, ni_resp: None, verbose, intmd: HashMap::new() }
    }

    /// Attach a dedicated small numerical-integration grid (`ni_resp`) for the response path.
    ///
    /// When set, the response (`get_response_bra`) is evaluated on this grid instead of the common
    /// (large) grid, and `cpks_vxc` / `cpks_fxc` are computed on it during
    /// [`make_response_preparation`](RRespAPI::make_response_preparation).
    pub fn set_ni_resp(mut self, ni_resp: NIMatmul<'a>) -> Self {
        self.ni_resp = Some(ni_resp);
        self
    }
}

impl<'a> AnalDrvBaseAPI for RRespKSNIMatmul<'a> {}

impl<'a> RRespAPI for RRespKSNIMatmul<'a> {
    /// Fock (the XC numint contribution only) from density matrix, on the common grid `ni`.
    fn get_fock_rdm(&mut self, rdm: TsrView) -> Tsr {
        assert_eq!(rdm.ndim(), 2, "rdm must have 2 dimensions");
        let den_type = determine_den_type_from_list(&self.xc_func_list.iter().map(|(_, f)| f).collect_vec());
        let rho = self.ni.make_rho_from_dm(&[rdm], den_type);
        let (vxc, _fxc) = eval_vxc_fxc_from_rho(&self.xc_func_list, rho.i((.., .., 0)));
        self.ni.make_vxc_pot_with_eff(vxc.view(), den_type, XCSpin::Unpolarized)
    }

    // note: `get_fock_coeff` keeps the trait default (dm route via `get_dm0_restricted`).

    /// Response matrix from (symmetrizable) full density matrices: `4 * fxc-response` on the
    /// common grid. The `rdm` carries a leading `[nao, nao]` square followed by an arbitrary
    /// (possibly empty) set of trailing dimensions; the output has the same shape.
    ///
    /// This is the rdm-form entry of the restricted response kernel, consistent with the
    /// `4 J - 2 K` convention of [`RRespRIJK`](crate::ri_jk::resp_r::RRespRIJK), and matches
    /// pyscf's orbital-hessian `vind` (ground-state `_gen_rhf_response`) contracted with `4`.
    /// It is the form required by the generalized-Fock Lagrangian term `A_{ai, pq} D_{pq}` of
    /// post-SCF methods; accordingly, the fxc kernel is built from the ground-state density on
    /// the common (large) grid `ni` — not on the dedicated response grid — mirroring the
    /// Lagrangian evaluation on the SCF grids in pyscf-forge.
    fn get_response_rdm(&mut self, rdm: TsrView) -> Tsr {
        assert!(rdm.ndim() >= 2, "rdm must have at least 2 dimensions");
        let rdm_shape = rdm.shape().to_vec();
        let nao = rdm_shape[0];
        assert_eq!(nao, rdm_shape[1], "the first two dimensions of rdm must be equal");
        let nset: usize = rdm_shape[2..].iter().product();
        let den_type = determine_den_type_from_list(&self.xc_func_list.iter().map(|(_, f)| f).collect_vec());

        if !self.intmd.contains_key("fxc_common_grid") {
            // panic with "no entry" if preparation is skipped
            let mo_coeff = self.intmd["mo_coeff"].view();
            let mo_occ = self.intmd["mo_occ"].view();
            let (_, fxc) = make_cpks_vxc_fxc(&self.xc_func_list, &mut self.ni, mo_coeff, mo_occ);
            self.intmd.insert("fxc_common_grid".to_string(), fxc);
        }
        // NOTE: unlike `cpks_fxc`, the cached `fxc_common_grid` is NOT re-validated against the
        // orbitals on later calls. This is safe under the analdrv driver contract (the orbitals
        // are fixed for the lifetime of a driver object), but any future caller that
        // re-prepares with different orbitals must invalidate this entry as well.
        let fxc = self.intmd["fxc_common_grid"].view();

        // flatten the trailing dimensions into one set dimension; assume each set of the rdm can
        // be symmetrized — the fxc kernel only sees the symmetric part anyway
        let rdm_sym = ((&rdm + &rdm.swapaxes(0, 1)) * 0.5).into_shape((nao, nao, nset));
        let dm_views: Vec<TsrView> = (0..nset).map(|s| rdm_sym.i((.., .., s))).collect();
        let rho1 = self.ni.make_rho_from_dm(&dm_views, den_type);
        let resp = self.ni.make_fxc_pot_with_eff(fxc, rho1.view(), den_type, XCSpin::Unpolarized);
        // 4.0 times is a trick of closed-shell coefficient; restore the input's trailing shape
        (resp * 4.0_f64).into_shape(rdm_shape)
    }

    /// Cached on first call for fixed inputs: the CP-KS `vxc`/`fxc` evaluation is skipped when
    /// this object was already prepared with the same orbitals (e.g. repeated calls from
    /// multi-order property evaluations directly reuse the stored results).
    fn make_response_preparation(&mut self, mo_coeff: TsrView, mo_occ: TsrView) {
        // cache check (before the orbital refresh): whether the expensive evaluation below was
        // already performed with the same orbitals (`mo_coeff`/`mo_occ` are always inserted
        // together with `cpks_vxc`/`cpks_fxc`, so the keys coexist)
        let already_prepared = self.intmd.contains_key("cpks_fxc")
            && is_same_tensor(self.intmd["mo_coeff"].view(), mo_coeff.view())
            && is_same_tensor(self.intmd["mo_occ"].view(), mo_occ.view());
        self.intmd.insert("mo_coeff".to_string(), mo_coeff.into_contig(ColMajor));
        self.intmd.insert("mo_occ".to_string(), mo_occ.into_contig(ColMajor));
        if already_prepared {
            return;
        }

        // Compute `cpks_vxc` / `cpks_fxc` on the selected response grid (the small `ni_resp` when
        // attached, else the common grid) from the ground-state density, using the lean
        // [`make_cpks_vxc_fxc`] (no skeleton intermediates, minimal AO derivative order, density
        // formed from occupied MOs via a bra-ket contraction rather than a full dm0).
        let ni_resp = self.ni_resp.as_mut().unwrap_or(&mut self.ni);
        let mo_coeff = self.intmd["mo_coeff"].view();
        let mo_occ = self.intmd["mo_occ"].view();
        let (vxc, fxc) = make_cpks_vxc_fxc(&self.xc_func_list, ni_resp, mo_coeff, mo_occ);
        self.intmd.insert("cpks_vxc".to_string(), vxc);
        self.intmd.insert("cpks_fxc".to_string(), fxc);
    }

    fn get_response_bra(&mut self, bra: TsrView) -> Tsr {
        let ni_resp = self.ni_resp.as_mut().unwrap_or(&mut self.ni);
        let mo_coeff = self.intmd.get("mo_coeff").unwrap();
        let mo_occ = self.intmd.get("mo_occ").unwrap();
        let fxc_eff = self.intmd.get("cpks_fxc").unwrap();
        let occidx = mo_occ.view().greater(0).into_vec();
        let mocc = mo_coeff.bool_select(-1, &occidx);

        let (resp, _timing) = get_rks_response_bra_batched(
            ni_resp,
            determine_den_type_from_list(&self.xc_func_list.iter().map(|(_, f)| f).collect_vec()),
            fxc_eff.view(),
            bra,
            mocc.view(),
            self.verbose,
        );
        resp
    }
}
