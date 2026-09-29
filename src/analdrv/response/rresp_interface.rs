//! Response (fock/response) objects for restricted SCF, built from SCF data.
//!
//! This is the response-side counterpart of the hessian's
//! [`rscf_hess_interface`](crate::analdrv::hessian::rscf_interface::rscf_hess_interface): it
//! assembles the `RRespAPI` objects for all electron-interaction contributions of a converged
//! restricted SCF. The hessian interface calls it and feeds the returned objects to `RHessSCF`; a
//! future standalone response/property driver can call it directly, without any hessian
//! machinery.
//!
//! The precision of the response contractions is selected per call through the `prec` flag of
//! [`RRespAPI`]: the high-precision (`prec = true`) resource is the SCF-grade one — the SCF
//! `rimatr` for RI-JK, and the SCF grid for DFT (in its native round-robin atom-interleaved
//! order, since the fock path is a plain quadrature sum, so no atom-regrouping is applied) —
//! while the low-precision (`prec = false`) resource of the CP-SCF machinery is the dedicated
//! (usually coarser) response grid / the `resp_auxbas_path` basis when attached. The
//! hessian-side skeleton grid policy (including the MGGA level bump) lives in the hessian
//! interface; no grid identity or grid data is shared between the hessian and response
//! subsystems.

use crate::analdrv::config::AnalDrvRespCfg;
use crate::analdrv::prelude::*;
use crate::ri_jk::resp_auxbas;
use crate::dft::numint_matmul::nimatmul::NIMatmul;
use crate::dft::numint_matmul::resp_rks::RRespKSNIMatmul;
use crate::dft::Grids;
use crate::ri_jk::resp_r::RRespRIJK;
use crate::ri_jk::util::get_cint_mol;
use crate::SCF;

use libxc::prelude::*;

/// J/K factors of the RI-JK electronic-interaction contributions of the (restricted) SCF.
///
/// Returns `(factor_j, factor_k, rsh)`. For range-separated hybrids, `rsh = Some((omega,
/// factor_k_sr))` carries the range-separation parameter and the short-range exchange factor of
/// the correction object, following the same decomposition as the SCF Fock assembly:
/// `K = alpha * K_full + factor_k_sr * K_erfc` with `factor_k_sr = hyb - alpha`.
pub fn scf_jk_factors(scf_data: &SCF) -> (f64, f64, Option<(f64, f64)>) {
    let is_hf = scf_data.mol.xc_data.dfa_compnt_scf.is_empty();
    let factor_j = 1.0;
    let factor_k = match is_hf {
        true => 1.0,
        false => scf_data.mol.xc_data.rsh_alpha().unwrap_or(scf_data.mol.xc_data.dfa_hybrid_scf),
    };
    let rsh = scf_data.mol.xc_data.omega().map(|omega| {
        let alpha = scf_data.mol.xc_data.rsh_alpha().unwrap();
        let hyb = scf_data.mol.xc_data.dfa_hybrid_scf;
        (omega, hyb - alpha)
    });
    (factor_j, factor_k, rsh)
}

/// List of `(scale, functional)` pairs of the SCF XC functional, in the given libxc spin
/// treatment.
pub fn scf_xc_func_list_with_spin(scf_data: &SCF, spin: LibXCSpin) -> Vec<(f64, LibXCFunctional)> {
    let xc_code = &scf_data.mol.xc_data.dfa_compnt_scf;
    let xc_params = &scf_data.mol.xc_data.dfa_paramr_scf;
    xc_code
        .iter()
        .zip(xc_params.iter())
        .map(|(&code, &param)| (param, LibXCFunctional::from_number(code as _, spin)))
        .collect_vec()
}

/// List of `(scale, functional)` pairs of the SCF XC functional (spin-unpolarized).
pub fn scf_xc_func_list(scf_data: &SCF) -> Vec<(f64, LibXCFunctional)> {
    scf_xc_func_list_with_spin(scf_data, LibXCSpin::Unpolarized)
}

/// The response (fock/response) objects of all electron-interaction contributions of a restricted
/// SCF, as an owned list of `RRespAPI` trait objects.
///
/// The type is itself a composite `RRespAPI`: every trait method fans out to the elements, with
/// the matrix-valued outputs (`get_fock_rdm`/`get_fock_coeff`/`get_response_rdm`/
/// `get_response_bra`) summed over the contributions. Drivers therefore hold `&mut RRespSCF` as
/// their single response object.
///
/// On top of the trait, the type carries the CP-SCF machinery shared by the drivers
/// ([`Self::response_mo`]/[`Self::response_dimless_cpscf`]/[`Self::solve_dimless_cpscf`]): the
/// solver settings are captured at build time, and the orbital state is stored by
/// [`Self::make_cpscf_preparation`].
///
/// The list is open-ended: further contributions can be appended to [`Self::resp_list`] without
/// changing this type. Should a driver require the concrete type of an entry, manual downcast
/// through utilities on `AnalDrvBaseAPI` is the escape hatch.
pub struct RRespSCF<'a> {
    /// Response objects of all electron-interaction contributions: for examples, RI-JK, RSH
    /// exchange, DFT XC NIMatmul.
    pub resp_list: Vec<Box<dyn RRespAPI + 'a>>,
    /// CP-SCF solver settings, captured at build time by [`rscf_resp_interface`].
    pub resp_cfg: AnalDrvRespCfg,
    /// Orbital state stored by [`Self::make_cpscf_preparation`]; the CP-SCF machinery panics
    /// until it is set.
    cpscf_state: Option<RCpscfState>,
}

/// Orbital state cached by [`RRespSCF::make_cpscf_preparation`] for the inherent CP-SCF
/// machinery, and reused by drivers upon it (e.g. the Z-vector solve of
/// [`crate::analdrv::response::rgfock_interface::solve_z_vector`]).
pub struct RCpscfState {
    /// Molecular orbital coefficients, shape `[nao, nmo]`.
    pub mo_coeff: Tsr,
    /// Level-shifted orbital-energy differences `e_a - e_i + shift`, shape `[nvir, nocc]`.
    pub e_ai_shift: Tsr,
    /// Number of occupied orbitals.
    pub nocc: usize,
    /// Total number of molecular orbitals.
    pub nmo: usize,
}

impl<'a> AnalDrvBaseAPI for RRespSCF<'a> {}

impl<'a> RRespAPI for RRespSCF<'a> {
    fn get_fock_rdm(&mut self, rdm: TsrView, prec: bool) -> Tsr {
        let mut fock = rt::zeros_like(&rdm);
        for resp_obj in self.resp_list.iter_mut() {
            fock += resp_obj.get_fock_rdm(rdm.view(), prec);
        }
        fock
    }

    fn get_fock_coeff(&mut self, mo_coeff: TsrView, mo_occ: TsrView, prec: bool) -> Tsr {
        // delegated per element (not through the default rdm route), so that elements overriding
        // this function for efficiency keep their advantage
        let mut fock = None;
        for resp_obj in self.resp_list.iter_mut() {
            let fock_obj = resp_obj.get_fock_coeff(mo_coeff.view(), mo_occ.view(), prec);
            fock = Some(match fock {
                Some(fock) => fock + fock_obj,
                None => fock_obj,
            });
        }
        fock.expect("RRespSCF must hold at least one response object")
    }

    fn make_response_preparation(&mut self, mo_coeff: TsrView, mo_occ: TsrView, prec: bool) {
        for resp_obj in self.resp_list.iter_mut() {
            resp_obj.make_response_preparation(mo_coeff.view(), mo_occ.view(), prec);
        }
    }

    fn get_response_rdm(&mut self, rdm: TsrView, prec: bool) -> Tsr {
        let mut resp = rt::zeros_like(&rdm);
        for resp_obj in self.resp_list.iter_mut() {
            resp += resp_obj.get_response_rdm(rdm.view(), prec);
        }
        resp
    }

    fn get_response_bra(&mut self, bra: TsrView, prec: bool) -> Tsr {
        let mut resp = rt::zeros_like(&bra);
        for resp_obj in self.resp_list.iter_mut() {
            resp += resp_obj.get_response_bra(bra.view(), prec);
        }
        resp
    }
}

impl<'a> RRespSCF<'a> {
    /// Store the orbital state and prepare all response objects for the CP-SCF calculation.
    ///
    /// This wraps the trait-level [`RRespAPI::make_response_preparation`] fan-out, and
    /// additionally caches the orbital coefficients and the level-shifted orbital-energy
    /// differences, so that [`Self::response_mo`], [`Self::response_dimless_cpscf`] and
    /// [`Self::solve_dimless_cpscf`] can be called with the perturbation only.
    ///
    /// # Parameters
    ///
    /// - `mo_coeff` : shape `[nao, nmo]`. Molecular orbital coefficients.
    /// - `mo_occ` : shape `[nmo]`. Molecular orbital occupation numbers.
    /// - `mo_energy` : shape `[nmo]`. Molecular orbital energies.
    pub fn make_cpscf_preparation(&mut self, mo_coeff: TsrView, mo_occ: TsrView, mo_energy: TsrView) {
        // the CP-SCF core runs the response machinery in low precision (`prec = false`): the
        // dedicated response grid / `resp_auxbas_path` basis when attached
        for resp_obj in self.resp_list.iter_mut() {
            resp_obj.make_response_preparation(mo_coeff.view(), mo_occ.view(), false);
        }

        let occidx = mo_occ.view().greater(0).into_vec();
        let viridx = occidx.iter().map(|&x| !x).collect_vec();
        let eocc = mo_energy.bool_select(-1, &occidx);
        let evir = mo_energy.bool_select(-1, &viridx);
        let e_ai = evir.i((.., None)) - eocc.i((None, ..));
        let e_ai_shift = &e_ai + self.resp_cfg.level_shift;

        self.cpscf_state = Some(RCpscfState {
            mo_coeff: mo_coeff.to_owned(),
            e_ai_shift,
            nocc: occidx.iter().filter(|&&x| x).count(),
            nmo: mo_occ.shape()[0],
        });
    }

    /// The CP-SCF orbital state cached by [`Self::make_cpscf_preparation`], for reuse by drivers
    /// building on the CP-SCF machinery (e.g. Z-vector solves). Panics before the preparation.
    ///
    /// Note the inherent CP-SCF methods of this type access the state as a direct field, keeping
    /// the borrow disjoint from `resp_list`; external callers use this accessor.
    pub fn cpscf_state(&self) -> &RCpscfState {
        self.cpscf_state
            .as_ref()
            .expect("Call `RRespSCF::make_cpscf_preparation` before the CP-SCF machinery.")
    }

    /// Compute the response in MO space to a perturbation in MO space.
    ///
    /// Half-transforms the perturbation to the AO bra, contracts the response of all
    /// contributions, and transforms back to MO space.
    /// Call [`Self::make_cpscf_preparation`] before this function to make sure the data is ready.
    ///
    /// # Parameters
    ///
    /// - `mo1` : shape `[nmo, nocc, ...]`. The perturbation in MO space.
    ///
    /// # Returns
    ///
    /// - `resp` : shape `[nmo, nocc, ...]`. The response in MO space.
    pub fn response_mo(&mut self, mo1: TsrView) -> Tsr {
        let state = self
            .cpscf_state
            .as_ref()
            .expect("Call `RRespSCF::make_cpscf_preparation` before the CP-SCF machinery.");
        let mo_coeff = state.mo_coeff.view();
        let ubra = &mo_coeff % &mo1;
        let mut resp = rt::zeros_like(&mo1);
        for resp_obj in self.resp_list.iter_mut() {
            // low precision: the CP-SCF core contracts the response on the dedicated
            // low-precision resource when attached (this includes the last-iteration W-operator
            // assembly of the hessian driver)
            resp += mo_coeff.t() % resp_obj.get_response_bra(ubra.view(), false);
        }
        resp
    }

    /// Compute the response in the dimensionless form used by the CP-SCF solve.
    ///
    /// Compared to [`Self::response_mo`], this additionally handles
    /// - the level shift in denominator
    /// - the zeroing of occupied-part response (we use `mo1[occ, occ]` part for evaluating
    ///   `resp[vir, occ]`, but we actually only want to solve the `mo1[vir, occ]` part and freeze
    ///   `mo1[occ, occ]` part to always be 0.5 times of ovlp_deriv1).
    /// Call [`Self::make_cpscf_preparation`] before this function to make sure the data is ready.
    ///
    /// # Parameters
    ///
    /// - `mo1` : shape `[nmo, nocc, ...]`. The perturbation in MO space.
    ///
    /// # Returns
    ///
    /// - `resp` : shape `[nmo, nocc, ...]`. The dimensionless response in MO space.
    pub fn response_dimless_cpscf(&mut self, mo1: TsrView) -> Tsr {
        let mut resp = self.response_mo(mo1.view());

        let state = self
            .cpscf_state
            .as_ref()
            .expect("Call `RRespSCF::make_cpscf_preparation` before the CP-SCF machinery.");
        let level_shift = self.resp_cfg.level_shift;
        let so = rt::slice!(0, state.nocc);
        let sv = rt::slice!(state.nocc, state.nmo);

        // handle dimensionless denominator and force handle virtual-part only
        if level_shift != 0.0 {
            resp -= level_shift * mo1;
        }
        *&mut resp.i_mut(sv) /= &state.e_ai_shift;
        resp.i_mut(so).fill(0.0);
        resp
    }

    /// Solve the dimensionless CP-SCF equation using a Krylov solver.
    ///
    /// This solves `U + resp(U) = rhs`. Note difference of standard CP-SCF equation as mentioned
    /// in the hessian driver.
    /// Call [`Self::make_cpscf_preparation`] before this function to make sure the data is ready.
    ///
    /// # Parameters
    ///
    /// - `rhs` : shape `[nmo, nocc, ...]`. Dimensionless right-hand side.
    ///
    /// # Returns
    ///
    /// - `mo1` : shape `[nmo, nocc, ...]`. Perturbation in MO space that solves the dimensionless
    ///   CP-SCF equation.
    pub fn solve_dimless_cpscf(&mut self, rhs: TsrView) -> Tsr {
        let rhs_shape = rhs.shape().to_vec();
        let nmo = rhs.shape()[0];
        let nocc = rhs.shape()[1];
        let rhs = rhs.reshape((nmo * nocc, -1));

        let tol = self.resp_cfg.tol;
        let max_cycle = self.resp_cfg.max_cycle;
        let max_space = self.resp_cfg.max_space;
        let lindep = self.resp_cfg.lindep;
        let tol_inflation = self.resp_cfg.tol_inflation;

        let response_cpscf_flattened = |x: TsrView| -> Tsr {
            let x = x.reshape((nmo, nocc, -1));
            let y = self.response_dimless_cpscf(x.view());
            y.into_shape((nmo * nocc, -1))
        };
        let mo1 = krylov_block(
            response_cpscf_flattened,
            rhs.view(),
            None,
            tol,
            max_cycle,
            max_space,
            lindep,
            tol_inflation,
        );
        mo1.into_shape(rhs_shape)
    }
}

/// The DFT response grids: the common (fock-path) grid and the optional dedicated response grid.
///
/// Shared by the restricted and unrestricted response interfaces (the grid policy is
/// spin-independent). Returns `(ni, ni_resp)`:
///
/// - the common grid is the SCF grid as is; the fock path is a plain quadrature sum, so it does
///   not require the atom-grouped (ByAtom) ordering;
/// - the response grid is dedicated (built at `config.resp.grid_level`, by default the coarser
///   `grid_gen_level.max(3) - 2`) when that level differs from the SCF grid generation level,
///   and `None` when they coincide (the response then evaluates and caches on the common grid).
pub fn resp_grids<'a>(scf_data: &'a SCF, config: &AnalDrvConfig) -> (NIMatmul<'a>, Option<NIMatmul<'a>>) {
    let mol = get_cint_mol(&scf_data.mol);

    let grids = scf_data.grids.as_ref().unwrap();
    let ni = NIMatmul::new(&mol, &grids.coordinates, &grids.weights, &grids.atm_idx, &grids.quadrature_weights);

    let grid_gen_level = scf_data.mol.ctrl.grid_gen_level;
    let grid_resp_level = config.resp.grid_level.unwrap_or(grid_gen_level.max(3) - 2);
    let ni_resp = if grid_resp_level == grid_gen_level {
        None
    } else {
        let cpscf_grid = Grids::build_with_level(&scf_data.mol, grid_resp_level);
        let ni_resp = NIMatmul::new(
            &mol,
            &cpscf_grid.coordinates,
            &cpscf_grid.weights,
            &cpscf_grid.atm_idx,
            &cpscf_grid.quadrature_weights,
        );
        Some(ni_resp)
    };
    (ni, ni_resp)
}

/// Build the response (fock/response) objects for a converged restricted SCF.
pub fn rscf_resp_interface<'a>(scf_data: &'a SCF, config: &AnalDrvConfig) -> RRespSCF<'a> {
    let mut resp_list: Vec<Box<dyn RRespAPI + 'a>> = Vec::new();

    // --- RI-JK --- //

    let (factor_j, factor_k, rsh) = scf_jk_factors(scf_data);

    // decomposed ERIs of the RI-JK response object, per precision: the fock path always borrows
    // the SCF rimatr (zero copy); the low-precision response path attaches the freshly built one
    // of `resp_auxbas_path` when set (owned), and reuses the SCF rimatr otherwise
    {
        let (cderi, cderi_resp) = resp_auxbas::cderi_pair(scf_data, config);
        let resp_obj = match cderi_resp {
            Some(cderi_resp) => RRespRIJK::new_with_cderi(factor_j, factor_k, cderi).set_cderi_resp(cderi_resp),
            None => RRespRIJK::new_with_cderi(factor_j, factor_k, cderi),
        };
        resp_list.push(Box::new(resp_obj));
    }

    // The short-range exchange correction (range-separated hybrids) is a separate response
    // object reusing the full-range implementation; it evaluates on the short-range `rimatr_sr`
    // ERI, with no Coulomb part (factor_j = 0).
    if let Some((_omega, factor_k_sr)) = rsh {
        let (cderi_sr, cderi_resp_sr) = resp_auxbas::cderi_pair_sr(scf_data, config);
        let resp_obj = match cderi_resp_sr {
            Some(cderi_resp_sr) => {
                RRespRIJK::new_with_cderi(0.0, factor_k_sr, cderi_sr).set_cderi_resp(cderi_resp_sr)
            },
            None => RRespRIJK::new_with_cderi(0.0, factor_k_sr, cderi_sr),
        };
        resp_list.push(Box::new(resp_obj));
    }

    // --- DFT --- //

    let is_hf = scf_data.mol.xc_data.dfa_compnt_scf.is_empty();
    if !is_hf {
        let xc_func_list = scf_xc_func_list(scf_data);
        let verbose = scf_data.mol.ctrl.print_level > 2;

        let (ni, ni_resp) = resp_grids(scf_data, config);
        let resp_obj = match ni_resp {
            Some(ni_resp) => RRespKSNIMatmul::new(xc_func_list, ni, verbose).set_ni_resp(ni_resp),
            None => RRespKSNIMatmul::new(xc_func_list, ni, verbose),
        };
        resp_list.push(Box::new(resp_obj));
    }

    RRespSCF { resp_list, resp_cfg: config.resp.clone(), cpscf_state: None }
}
