//! Response (Fock and response matrix) implementation for the UKS numint-matmul XC component.
//!
//! The response object [`URespKSNIMatmul`] is a standalone object for the fock/response
//! functionality of the UKS numerical-integration XC contribution, independent of the hessian
//! machinery of `UHessKSNIMatmul` (which owns its own grid, built from the same grid data).
//!
//! The two grids are separated by purpose: the fock path (`get_fock_rdm`/`get_fock_coeff`) always
//! evaluates on the common (large) grid `ni`, while the response path
//! (`make_response_preparation`/`get_response_bra`) uses the small response grid `ni_resp` when
//! one is attached. The returned fock is the DFT numerical-integration XC contribution only, never
//! the full SCF fock.
//!
//! Main-thread only (not `Send`/`Sync`).

use super::prelude::*;
use crate::analdrv::prelude::*;

#[allow(non_upper_case_globals)]
const α: usize = 0;
#[allow(non_upper_case_globals)]
const β: usize = 1;

/// Evaluate the spin-polarized XC potential `vxc` and kernel `fxc` from a given ground-state
/// `rho`, by summing each sub-functional's spin-polarized `libxc_eval_eff` (deriv = 2) contribution.
///
/// `rho` : shape `[ngrids, nvar, 2]`. Extracted from the rho formation so that callers which
/// already have `rho` (e.g. obtained directly from occupied spin-orbitals via a bra-ket
/// contraction) can skip the `ao_dm0`-based density formation.
pub fn eval_vxc_fxc_uks_from_rho(xc_func_list: &[(f64, LibXCFunctional)], rho: TsrView) -> (Tsr, Tsr) {
    assert!(!xc_func_list.is_empty(), "xc_func_list must not be empty");
    let xc_type = xc_func_list
        .iter()
        .map(|(_, f)| determine_den_type(f))
        .max_by_key(|t| t.num_nvar())
        .expect("xc_func_list must not be empty");
    let nvar = xc_type.num_nvar();
    let ngrids = rho.shape()[0];
    let device = rho.device().clone();

    let mut vxc = rt::zeros(([ngrids, nvar, 2], &device));
    let mut fxc = rt::zeros(([ngrids, nvar, 2, nvar, 2], &device));
    for (scale, xc_func) in xc_func_list {
        let xc_type_i = determine_den_type(xc_func);
        let nvar_i = xc_type_i.num_nvar();
        let rho_i = rho.i((.., ..nvar_i, ..));
        let xc_eff = libxc_eval_eff(xc_func, rho_i, 2, false);
        let [_, vxc_i, fxc_i] = xc_eff.into_iter().collect_array().unwrap();
        *&mut vxc.i_mut((.., ..nvar_i, ..)) += *scale * vxc_i;
        *&mut fxc.i_mut((.., ..nvar_i, .., ..nvar_i, ..)) += *scale * fxc_i;
    }
    (vxc, fxc)
}

/// Lean evaluation of only `vxc` and `fxc` on a given grid, for use as the CP-KS `cpks_vxc` /
/// `cpks_fxc` when a dedicated (coarser) CP-KS grid is attached (UKS counterpart of
/// [`make_cpks_vxc_fxc`](super::resp_rks::make_cpks_vxc_fxc)).
///
/// Skips all skeleton-Hessian intermediates, evaluates AO at the minimum derivative order needed
/// to form the spin densities (`xc_type.num_ao_deriv()`), and forms each spin density `rhoσ`
/// directly from the occupied spin-orbital coefficients via a bra-ket contraction
/// ([`NIMatmul::make_rho_from_homogeneous_braket`]) instead of building the full `[nao, nao]` spin
/// density matrices (consistent with the CP-KS response path in [`get_uks_response_bra`]).
pub fn make_cpks_vxc_fxc_uks(
    xc_func_list: &[(f64, LibXCFunctional)],
    ni: &mut NIMatmul,
    mo_coeff: &[TsrView; 2],
    mo_occ: &[TsrView; 2],
) -> (Tsr, Tsr) {
    let xc_type = determine_den_type_from_list(&xc_func_list.iter().map(|(_, f)| f).collect_vec());

    // For each spin, bake the occupation into the occupied spin coefficients (sqrt so the bra-ket
    // square reproduces the occ-weighted spin density for every component) and form rhoσ directly
    // via a bra-ket contraction. For UKS the spin occupation is 1, so this is a no-op in the common
    // case but stays correct for fractional occupation.
    let occidx_α = mo_occ[α].view().greater(0).into_vec();
    let mocc_α = mo_coeff[α].bool_select(-1, &occidx_α);
    let occ_α = mo_occ[α].bool_select(-1, &occidx_α);
    let occ_sqrt_α = occ_α.mapv(f64::sqrt);
    let mocc_2_α = &mocc_α * occ_sqrt_α.i((None, ..));

    let occidx_β = mo_occ[β].view().greater(0).into_vec();
    let mocc_β = mo_coeff[β].bool_select(-1, &occidx_β);
    let occ_β = mo_occ[β].bool_select(-1, &occidx_β);
    let occ_sqrt_β = occ_β.mapv(f64::sqrt);
    let mocc_2_β = &mocc_β * occ_sqrt_β.i((None, ..));

    // rho : [ngrids, nvar, 2]
    let rho = ni.make_rho_from_homogeneous_braket(&[mocc_2_α.view(), mocc_2_β.view()], xc_type);
    eval_vxc_fxc_uks_from_rho(xc_func_list, rho.view())
}

/* #region response */

pub fn get_uks_response_bra(
    ni: &mut NIMatmul,
    den_type: XCDenType,
    fxc_eff: TsrView,
    bra: &[TsrView; 2],
    mocc: &[TsrView; 2],
) -> ([Tsr; 2], IndexMap<&'static str, f64>) {
    let nao = bra[α].shape()[0];
    let nocc_α = bra[α].shape()[1];
    let nocc_β = bra[β].shape()[1];
    let bra_α_shape = bra[α].shape().to_vec();
    let bra_β_shape = bra[β].shape().to_vec();
    let bra_α = bra[α].reshape((nao, nocc_α, -1));
    let bra_β = bra[β].reshape((nao, nocc_β, -1));
    let nset = bra_α.shape()[2];

    let mut timing = IndexMap::new();
    let mut tic = |label: &'static str, t0: std::time::Instant| {
        let elapsed = t0.elapsed().as_secs_f64();
        timing.insert(label, elapsed);
    };

    let t0 = std::time::Instant::now();
    ni.get_cached_ao(den_type.num_ao_deriv());
    tic("ao", t0);

    // Compute per-spin rho1
    let t0 = std::time::Instant::now();
    let bra_α_list = bra_α.axes_iter(-1).collect_vec();
    let bra_β_list = bra_β.axes_iter(-1).collect_vec();
    let rho1α = ni.make_rho_from_one_bra_mult_ket(mocc[α].view(), &bra_α_list, den_type);
    let rho1β = ni.make_rho_from_one_bra_mult_ket(mocc[β].view(), &bra_β_list, den_type);
    // Stack into [ngrids, nvar, 2, nset]
    let ngrids = rho1α.shape()[0];
    let nvar = den_type.num_nvar();
    let device = rho1α.device().clone();
    let mut rho1 = rt::zeros(([ngrids, nvar, 2, nset], &device));
    rho1.i_mut((.., .., α, ..)).assign(&rho1α);
    rho1.i_mut((.., .., β, ..)).assign(&rho1β);
    tic("rho1", t0);

    // Compute UKS fxc bra-trans response
    let t0 = std::time::Instant::now();
    let resp = ni.make_uks_fxc_pot_with_eff_bra_trans(fxc_eff, rho1.view(), mocc, den_type);
    tic("resp", t0);

    // UKS CPHF factor: 2.0 (hermitian symmetry only, no spin degeneracy)
    let [resp_α, resp_β] = resp;
    let resp_α = 2.0 * resp_α.into_shape(bra_α_shape);
    let resp_β = 2.0 * resp_β.into_shape(bra_β_shape);
    ([resp_α, resp_β], timing)
}

pub fn get_uks_response_bra_batched(
    ni: &mut NIMatmul,
    den_type: XCDenType,
    fxc_eff: TsrView,
    bra: &[TsrView; 2],
    mocc: &[TsrView; 2],
    verbose: bool,
) -> ([Tsr; 2], IndexMap<&'static str, f64>) {
    let ngrids = ni.weights.len();
    let nbatch = ni.nbatch;
    let bra_α_shape = bra[α].shape().to_vec();
    let bra_β_shape = bra[β].shape().to_vec();
    let device = bra[α].device().clone();
    let mut resp_α = rt::zeros((bra_α_shape, &device));
    let mut resp_β = rt::zeros((bra_β_shape, &device));
    let mut timing = IndexMap::from([("ao", 0.0), ("rho1", 0.0), ("resp", 0.0), ("total", 0.0)]);

    let t0 = std::time::Instant::now();
    for start in (0..ngrids).step_by(nbatch) {
        let end = (start + nbatch).min(ngrids);
        let mut ni_batch = ni.split_batch(start, end);
        let ([resp_batch_α, resp_batch_β], timing_batch) =
            get_uks_response_bra(&mut ni_batch, den_type, fxc_eff.i(start..end), bra, mocc);
        resp_α += resp_batch_α;
        resp_β += resp_batch_β;
        for (key, value) in timing_batch {
            *timing.get_mut(key).unwrap() += value;
        }
        let duration = t0.elapsed().as_secs_f64();
        timing.insert("total", duration);
        if verbose {
            println!("In get_uks_response_bra_batched, Batch {start}..{end}");
            println!("  Elapsed time from start (Wall time): {:.4} sec", duration);
        }
    }

    if verbose {
        println!("Finished get_uks_response_bra_batched");
        println!("  Total elapsed time (Wall time): {:.4} sec", timing["total"]);
        println!("  Timing breakdown:");
        for (key, value) in timing.iter() {
            if *key != "total" {
                println!("  {key:>20}: {value:.4} sec");
            }
        }
    }

    ([resp_α, resp_β], timing)
}

/* #endregion */

/// Response (fock/response matrix) object for the UKS numint-matmul XC contribution.
pub struct URespKSNIMatmul<'a> {
    /// List of `(scale, functional)` pairs of the XC functional (spin-polarized).
    pub xc_func_list: Vec<(f64, LibXCFunctional)>,
    /// Common (large) numerical-integration grid, used by the fock path.
    pub ni: NIMatmul<'a>,
    /// Optional separate small grid for the response path; `None` reuses `ni`.
    pub ni_resp: Option<NIMatmul<'a>>,
    /// Print progress of the response evaluation.
    pub verbose: bool,
    /// Response intermediates: `mo_coeff_0/1 [nao, nmo_s]` and `mo_occ_0/1 [nmo_s]` from
    /// [`URespAPI::make_response_preparation`], `cpks_vxc [ngrids, nvar, 2]` and `cpks_fxc
    /// [ngrids, nvar, 2, nvar, 2]` on the selected response grid.
    pub intmd: HashMap<String, Tsr>,
}

impl<'a> URespKSNIMatmul<'a> {
    /// Create a new UKS response object.
    ///
    /// # Parameters
    ///
    /// - `xc_func_list` : list of `(scale, functional)` pairs (spin-polarized functionals).
    /// - `ni` : numerical-integration driver over the common (large) grid.
    /// - `verbose` : print progress.
    pub fn new(xc_func_list: Vec<(f64, LibXCFunctional)>, ni: NIMatmul<'a>, verbose: bool) -> Self {
        Self { xc_func_list, ni, ni_resp: None, verbose, intmd: HashMap::new() }
    }

    /// Attach a dedicated small numerical-integration grid (`ni_resp`) for the response path.
    ///
    /// When set, the response (`get_response_bra`) is evaluated on this grid instead of the common
    /// (large) grid, and `cpks_vxc` / `cpks_fxc` are computed on it during
    /// [`make_response_preparation`](URespAPI::make_response_preparation).
    pub fn set_ni_resp(mut self, ni_resp: NIMatmul<'a>) -> Self {
        self.ni_resp = Some(ni_resp);
        self
    }
}

impl<'a> AnalDrvBaseAPI for URespKSNIMatmul<'a> {}

impl<'a> URespAPI for URespKSNIMatmul<'a> {
    /// Fock (the XC numint contribution only) per spin from spin density matrices, on the common
    /// grid `ni`.
    fn get_fock_rdm(&mut self, rdm: &[TsrView; 2]) -> [Tsr; 2] {
        for rdm_s in rdm {
            assert_eq!(rdm_s.ndim(), 2, "rdm must have 2 dimensions");
        }
        let den_type = determine_den_type_from_list(&self.xc_func_list.iter().map(|(_, f)| f).collect_vec());
        // rho : [ngrids, nvar, 2] (the spin is the set axis)
        let rho = self.ni.make_rho_from_dm(&[rdm[α].view(), rdm[β].view()], den_type);
        let (vxc, _fxc) = eval_vxc_fxc_uks_from_rho(&self.xc_func_list, rho.view());
        let pot = self.ni.make_vxc_pot_with_eff(vxc.view(), den_type, XCSpin::Polarized);
        [pot.i((.., .., α)).to_owned(), pot.i((.., .., β)).to_owned()]
    }

    // note: `get_fock_coeff` keeps the trait default (dm route via `get_dm0_restricted` per spin).

    /// rdm-form entry of the unrestricted response kernel; reserved for the future ugfock
    /// machinery (no consumer yet), mirroring the restricted side before RI-PT2.
    fn get_response_rdm(&mut self, _rdm: &[TsrView; 2]) -> [Tsr; 2] {
        unimplemented!("get_response_rdm is not implemented for URespKSNIMatmul; reserved for the future ugfock machinery.")
    }

    /// Cached on first call for fixed inputs: the CP-KS `vxc`/`fxc` evaluation is skipped when
    /// this object was already prepared with the same orbitals (e.g. repeated calls from
    /// multi-order property evaluations directly reuse the stored results).
    fn make_response_preparation(&mut self, mo_coeff: &[TsrView; 2], mo_occ: &[TsrView; 2]) {
        // cache check (before the orbital refresh): whether the expensive evaluation below was
        // already performed with the same orbitals (`mo_coeff`/`mo_occ` are always inserted
        // together with `cpks_vxc`/`cpks_fxc`, so the keys coexist)
        let already_prepared = self.intmd.contains_key("cpks_fxc")
            && is_same_tensor(self.intmd["mo_coeff_0"].view(), mo_coeff[α].view())
            && is_same_tensor(self.intmd["mo_coeff_1"].view(), mo_coeff[β].view())
            && is_same_tensor(self.intmd["mo_occ_0"].view(), mo_occ[α].view())
            && is_same_tensor(self.intmd["mo_occ_1"].view(), mo_occ[β].view());
        self.intmd.insert("mo_coeff_0".to_string(), mo_coeff[α].view().into_contig(ColMajor));
        self.intmd.insert("mo_coeff_1".to_string(), mo_coeff[β].view().into_contig(ColMajor));
        self.intmd.insert("mo_occ_0".to_string(), mo_occ[α].view().into_contig(ColMajor));
        self.intmd.insert("mo_occ_1".to_string(), mo_occ[β].view().into_contig(ColMajor));
        if already_prepared {
            return;
        }

        // Compute `cpks_vxc` / `cpks_fxc` on the selected response grid (the small `ni_resp` when
        // attached, else the common grid) from the ground-state spin densities, using the lean
        // [`make_cpks_vxc_fxc_uks`] (no skeleton intermediates, minimal AO derivative order, spin
        // densities formed from occupied spin-orbitals via a bra-ket contraction rather than full
        // spin dm0 matrices).
        let ni_resp = self.ni_resp.as_mut().unwrap_or(&mut self.ni);
        let mo_coeff_α = self.intmd["mo_coeff_0"].view();
        let mo_coeff_β = self.intmd["mo_coeff_1"].view();
        let mo_occ_α = self.intmd["mo_occ_0"].view();
        let mo_occ_β = self.intmd["mo_occ_1"].view();
        let (vxc, fxc) =
            make_cpks_vxc_fxc_uks(&self.xc_func_list, ni_resp, &[mo_coeff_α, mo_coeff_β], &[mo_occ_α, mo_occ_β]);
        self.intmd.insert("cpks_vxc".to_string(), vxc);
        self.intmd.insert("cpks_fxc".to_string(), fxc);
    }

    fn get_response_bra(&mut self, bra: &[TsrView; 2]) -> [Tsr; 2] {
        let ni_resp = self.ni_resp.as_mut().unwrap_or(&mut self.ni);
        let mo_coeff_α = self.intmd["mo_coeff_0"].view();
        let mo_coeff_β = self.intmd["mo_coeff_1"].view();
        let mo_occ_α = self.intmd["mo_occ_0"].view();
        let mo_occ_β = self.intmd["mo_occ_1"].view();
        let fxc_eff = self.intmd["cpks_fxc"].view();

        let occidx_α = mo_occ_α.view().greater(0).into_vec();
        let occidx_β = mo_occ_β.view().greater(0).into_vec();
        let mocc_α = mo_coeff_α.bool_select(-1, &occidx_α);
        let mocc_β = mo_coeff_β.bool_select(-1, &occidx_β);

        let den_type = determine_den_type_from_list(&self.xc_func_list.iter().map(|(_, f)| f).collect_vec());

        let ([resp_α, resp_β], _timing) = get_uks_response_bra_batched(
            ni_resp,
            den_type,
            fxc_eff.view(),
            bra,
            &[mocc_α.view(), mocc_β.view()],
            self.verbose,
        );
        [resp_α, resp_β]
    }
}
