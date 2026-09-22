//! Response (fock/response) objects for unrestricted SCF, built from SCF data.
//!
//! This is the unrestricted counterpart of [`rscf_resp_interface`](crate::analdrv::response::
//! rresp_interface::rscf_resp_interface): it assembles the `URespAPI` objects for all
//! electron-interaction contributions of a converged unrestricted SCF. The hessian interface calls
//! it and feeds the returned objects to `UHessSCF`; a future standalone unrestricted
//! response/property driver can call it directly, without any hessian machinery.
//!
//! The DFT XC response object evaluates the fock path directly on the SCF grid (in its native
//! round-robin atom-interleaved order — the fock path is a plain quadrature sum, so no
//! atom-regrouping is applied), and the response path on a dedicated (usually coarser) cpscf
//! grid. The hessian-side skeleton grid policy (including the MGGA level bump) lives in the
//! hessian interface; no grid identity or grid data is shared between the hessian and response
//! subsystems.

use crate::analdrv::config::AnalDrvRespCfg;
use crate::analdrv::prelude::*;
use crate::analdrv::response::rresp_interface::scf_jk_factors;
use crate::dft::numint_matmul::nimatmul::NIMatmul;
use crate::dft::numint_matmul::resp_uks::URespKSNIMatmul;
use crate::dft::Grids;
use crate::ri_jk::resp_u::URespRIJK;
use crate::ri_jk::util::get_cint_mol;
use crate::SCF;

use libxc::prelude::*;

/// List of `(scale, functional)` pairs of the SCF XC functional (spin-polarized).
///
/// Spin-polarized sibling of [`scf_xc_func_list`](crate::analdrv::response::rresp_interface::
/// scf_xc_func_list): the functionals are created with [`LibXCSpin::Polarized`], as required by
/// the UKS rho/vxc/fxc spin-block layout.
pub fn scf_xc_func_list_uks(scf_data: &SCF) -> Vec<(f64, LibXCFunctional)> {
    let xc_code = &scf_data.mol.xc_data.dfa_compnt_scf;
    let xc_params = &scf_data.mol.xc_data.dfa_paramr_scf;
    xc_code
        .iter()
        .zip(xc_params.iter())
        .map(|(&code, &param)| (param, LibXCFunctional::from_number(code as _, LibXCSpin::Polarized)))
        .collect_vec()
}

/// The response (fock/response) objects of all electron-interaction contributions of an
/// unrestricted SCF, as an owned list of `URespAPI` trait objects.
///
/// The type is itself a composite `URespAPI`: every trait method fans out to the elements, with
/// the matrix-valued outputs (`get_fock_rdm`/`get_fock_coeff`/`get_response_rdm`/
/// `get_response_bra`) summed over the contributions per spin. Drivers therefore hold `&mut
/// URespSCF` as their single response object.
///
/// On top of the trait, the type carries the CP-SCF machinery shared by the drivers
/// ([`Self::response_mo`]/[`Self::response_dimless_cpscf`]/[`Self::solve_dimless_cpscf`]): the
/// solver settings are captured at build time, and the orbital state is stored by
/// [`Self::make_cpscf_preparation`]. The CP-SCF solve is genuinely **coupled** across spins: the
/// α and β unknowns are flattened into one block-Krylov space (`nmo_α·nocc_α + nmo_β·nocc_β`),
/// and every response contraction sees both spins in a single call.
///
/// The list is open-ended: further contributions can be appended to [`Self::resp_list`] without
/// changing this type. Should a driver require the concrete type of an entry, manual downcast
/// through utilities on `AnalDrvBaseAPI` is the escape hatch.
pub struct URespSCF<'a> {
    /// Response objects of all electron-interaction contributions: for examples, RI-JK, RSH
    /// exchange, DFT XC NIMatmul.
    pub resp_list: Vec<Box<dyn URespAPI + 'a>>,
    /// CP-SCF solver settings, captured at build time by [`uscf_resp_interface`].
    pub resp_cfg: AnalDrvRespCfg,
    /// Orbital state stored by [`Self::make_cpscf_preparation`]; the CP-SCF machinery panics
    /// until it is set.
    cpscf_state: Option<UCpscfState>,
}

/// Orbital state cached by [`URespSCF::make_cpscf_preparation`] for the inherent CP-SCF
/// machinery, per spin (index 0 = α, index 1 = β).
pub struct UCpscfState {
    /// Molecular orbital coefficients, shape `[nao, nmo_s]` per spin.
    pub mo_coeff: [Tsr; 2],
    /// Level-shifted orbital-energy differences `e_a - e_i + shift`, shape `[nvir_s, nocc_s]`
    /// per spin.
    pub e_ai_shift: [Tsr; 2],
    /// Number of occupied orbitals per spin.
    pub nocc: [usize; 2],
    /// Total number of molecular orbitals per spin.
    pub nmo: [usize; 2],
}

impl<'a> AnalDrvBaseAPI for URespSCF<'a> {}

impl<'a> URespAPI for URespSCF<'a> {
    fn get_fock_rdm(&mut self, rdm: &[TsrView; 2]) -> [Tsr; 2] {
        let mut fock = [rt::zeros_like(&rdm[0]), rt::zeros_like(&rdm[1])];
        for resp_obj in self.resp_list.iter_mut() {
            let fock_obj = resp_obj.get_fock_rdm(&[rdm[0].view(), rdm[1].view()]);
            fock[0] += &fock_obj[0];
            fock[1] += &fock_obj[1];
        }
        fock
    }

    fn get_fock_coeff(&mut self, mo_coeff: &[TsrView; 2], mo_occ: &[TsrView; 2]) -> [Tsr; 2] {
        // delegated per element (not through the default rdm route), so that elements overriding
        // this function for efficiency keep their advantage
        let mut fock: Option<[Tsr; 2]> = None;
        for resp_obj in self.resp_list.iter_mut() {
            let fock_obj = resp_obj
                .get_fock_coeff(&[mo_coeff[0].view(), mo_coeff[1].view()], &[mo_occ[0].view(), mo_occ[1].view()]);
            fock = Some(match fock {
                Some([f0, f1]) => [f0 + &fock_obj[0], f1 + &fock_obj[1]],
                None => fock_obj,
            });
        }
        fock.expect("URespSCF must hold at least one response object")
    }

    fn make_response_preparation(&mut self, mo_coeff: &[TsrView; 2], mo_occ: &[TsrView; 2]) {
        for resp_obj in self.resp_list.iter_mut() {
            resp_obj.make_response_preparation(&[mo_coeff[0].view(), mo_coeff[1].view()], &[
                mo_occ[0].view(),
                mo_occ[1].view(),
            ]);
        }
    }

    fn get_response_rdm(&mut self, rdm: &[TsrView; 2]) -> [Tsr; 2] {
        let mut resp = [rt::zeros_like(&rdm[0]), rt::zeros_like(&rdm[1])];
        for resp_obj in self.resp_list.iter_mut() {
            let resp_obj_rdm = resp_obj.get_response_rdm(&[rdm[0].view(), rdm[1].view()]);
            resp[0] += &resp_obj_rdm[0];
            resp[1] += &resp_obj_rdm[1];
        }
        resp
    }

    fn get_response_bra(&mut self, bra: &[TsrView; 2]) -> [Tsr; 2] {
        let mut resp = [rt::zeros_like(&bra[0]), rt::zeros_like(&bra[1])];
        for resp_obj in self.resp_list.iter_mut() {
            let resp_obj_bra = resp_obj.get_response_bra(&[bra[0].view(), bra[1].view()]);
            resp[0] += &resp_obj_bra[0];
            resp[1] += &resp_obj_bra[1];
        }
        resp
    }
}

impl<'a> URespSCF<'a> {
    /// Store the orbital state and prepare all response objects for the CP-SCF calculation.
    ///
    /// This wraps the trait-level [`URespAPI::make_response_preparation`] fan-out, and
    /// additionally caches the per-spin orbital coefficients and the level-shifted
    /// orbital-energy differences, so that [`Self::response_mo`], [`Self::response_dimless_cpscf`]
    /// and [`Self::solve_dimless_cpscf`] can be called with the perturbation only.
    ///
    /// # Parameters
    ///
    /// - `mo_coeff` : shape `[nao, nmo_s]` per spin. Molecular orbital coefficients.
    /// - `mo_occ` : shape `[nmo_s]` per spin. Molecular orbital occupation numbers.
    /// - `mo_energy` : shape `[nmo_s]` per spin. Molecular orbital energies.
    pub fn make_cpscf_preparation(&mut self, mo_coeff: &[TsrView; 2], mo_occ: &[TsrView; 2], mo_energy: &[TsrView; 2]) {
        let [α, β] = [0, 1];
        for resp_obj in self.resp_list.iter_mut() {
            resp_obj.make_response_preparation(&[mo_coeff[α].view(), mo_coeff[β].view()], &[
                mo_occ[α].view(),
                mo_occ[β].view(),
            ]);
        }

        let occidx = [mo_occ[α].view().greater(0).into_vec(), mo_occ[β].view().greater(0).into_vec()];
        let viridx = [occidx[α].iter().map(|&x| !x).collect_vec(), occidx[β].iter().map(|&x| !x).collect_vec()];
        let eocc = [mo_energy[α].bool_select(-1, &occidx[α]), mo_energy[β].bool_select(-1, &occidx[β])];
        let evir = [mo_energy[α].bool_select(-1, &viridx[α]), mo_energy[β].bool_select(-1, &viridx[β])];
        let e_ai = [evir[α].i((.., None)) - eocc[α].i((None, ..)), evir[β].i((.., None)) - eocc[β].i((None, ..))];
        let e_ai_shift = [&e_ai[α] + self.resp_cfg.level_shift, &e_ai[β] + self.resp_cfg.level_shift];

        self.cpscf_state = Some(UCpscfState {
            mo_coeff: [mo_coeff[α].to_owned(), mo_coeff[β].to_owned()],
            e_ai_shift,
            nocc: [occidx[α].iter().filter(|&&x| x).count(), occidx[β].iter().filter(|&&x| x).count()],
            nmo: [mo_occ[α].shape()[0], mo_occ[β].shape()[0]],
        });
    }

    /// The CP-SCF orbital state cached by [`Self::make_cpscf_preparation`], for reuse by drivers
    /// building on the CP-SCF machinery. Panics before the preparation.
    ///
    /// Note the inherent CP-SCF methods of this type access the state as a direct field, keeping
    /// the borrow disjoint from `resp_list`; external callers use this accessor.
    pub fn cpscf_state(&self) -> &UCpscfState {
        self.cpscf_state.as_ref().expect("Call `URespSCF::make_cpscf_preparation` before the CP-SCF machinery.")
    }

    /// Compute the response in MO space to a perturbation in MO space.
    ///
    /// Half-transforms the per-spin perturbation to the AO bra, contracts the response of all
    /// contributions (each contraction sees both spins at once — the Coulomb and XC kernels
    /// couple them), and transforms back to MO space.
    /// Call [`Self::make_cpscf_preparation`] before this function to make sure the data is ready.
    ///
    /// # Parameters
    ///
    /// - `mo1` : shape `[nmo_s, nocc_s, ...]` per spin. The perturbation in MO space.
    ///
    /// # Returns
    ///
    /// - `resp` : shape `[nmo_s, nocc_s, ...]` per spin. The response in MO space.
    pub fn response_mo(&mut self, mo1: &[TsrView; 2]) -> [Tsr; 2] {
        let [α, β] = [0, 1];
        let state =
            self.cpscf_state.as_ref().expect("Call `URespSCF::make_cpscf_preparation` before the CP-SCF machinery.");
        let mo_coeff = [state.mo_coeff[α].view(), state.mo_coeff[β].view()];
        let ubra_α = &mo_coeff[α] % &mo1[α];
        let ubra_β = &mo_coeff[β] % &mo1[β];
        let mut resp_α = rt::zeros_like(&ubra_α);
        let mut resp_β = rt::zeros_like(&ubra_β);

        // the get_response_bra fan-out is inlined (not the composite trait method), keeping the
        // `cpscf_state` borrow disjoint from the `resp_list` iteration
        for resp_obj in self.resp_list.iter_mut() {
            let el_resp = resp_obj.get_response_bra(&[ubra_α.view(), ubra_β.view()]);
            resp_α += mo_coeff[α].t() % &el_resp[α];
            resp_β += mo_coeff[β].t() % &el_resp[β];
        }
        [resp_α, resp_β]
    }

    /// Compute the response in the dimensionless form used by the CP-SCF solve.
    ///
    /// Compared to [`Self::response_mo`], this additionally handles per spin
    /// - the level shift in denominator
    /// - the zeroing of occupied-part response (we use `mo1[occ, occ]` part for evaluating
    ///   `resp[vir, occ]`, but we actually only want to solve the `mo1[vir, occ]` part and freeze
    ///   `mo1[occ, occ]` part to always be 0.5 times of ovlp_deriv1).
    /// Call [`Self::make_cpscf_preparation`] before this function to make sure the data is ready.
    ///
    /// # Parameters
    ///
    /// - `mo1` : shape `[nmo_s, nocc_s, ...]` per spin. The perturbation in MO space.
    ///
    /// # Returns
    ///
    /// - `resp` : shape `[nmo_s, nocc_s, ...]` per spin. The dimensionless response in MO space.
    pub fn response_dimless_cpscf(&mut self, mo1: &[TsrView; 2]) -> [Tsr; 2] {
        let [α, β] = [0, 1];
        let mut resp = self.response_mo(mo1);

        let state =
            self.cpscf_state.as_ref().expect("Call `URespSCF::make_cpscf_preparation` before the CP-SCF machinery.");
        let level_shift = self.resp_cfg.level_shift;
        let so = [rt::slice!(0, state.nocc[α]), rt::slice!(0, state.nocc[β])];
        let sv = [rt::slice!(state.nocc[α], state.nmo[α]), rt::slice!(state.nocc[β], state.nmo[β])];

        // handle dimensionless denominator and force handle virtual-part only
        if level_shift != 0.0 {
            resp[α] -= level_shift * &mo1[α];
            resp[β] -= level_shift * &mo1[β];
        }
        *&mut resp[α].i_mut(sv[α]) /= &state.e_ai_shift[α];
        *&mut resp[β].i_mut(sv[β]) /= &state.e_ai_shift[β];
        resp[α].i_mut(so[α]).fill(0.0);
        resp[β].i_mut(so[β]).fill(0.0);
        resp
    }

    /// Solve the dimensionless CP-SCF equation using a Krylov solver.
    ///
    /// This solves `U + resp(U) = rhs` for both spins **coupled**: the α and β unknowns are
    /// flattened into one block-Krylov space (`nmo_α·nocc_α + nmo_β·nocc_β` per property), so
    /// the spin-coupled response (total-density Coulomb, spin-block XC kernel) is applied
    /// exactly. Note difference of standard CP-SCF equation as mentioned in the hessian driver.
    /// Call [`Self::make_cpscf_preparation`] before this function to make sure the data is ready.
    ///
    /// # Parameters
    ///
    /// - `rhs` : shape `[nmo_s, nocc_s, ...]` per spin. Dimensionless right-hand side.
    ///
    /// # Returns
    ///
    /// - `mo1` : shape `[nmo_s, nocc_s, ...]` per spin. Perturbation in MO space that solves the
    ///   dimensionless CP-SCF equation.
    pub fn solve_dimless_cpscf(&mut self, rhs: &[TsrView; 2]) -> [Tsr; 2] {
        let [α, β] = [0, 1];
        let rhs_shape = [rhs[α].shape().to_vec(), rhs[β].shape().to_vec()];
        let nmo = [rhs[α].shape()[0], rhs[β].shape()[0]];
        let nocc = [rhs[α].shape()[1], rhs[β].shape()[1]];
        let rhs = [rhs[α].reshape((nmo[α], nocc[α], -1)), rhs[β].reshape((nmo[β], nocc[β], -1))];
        let device = rhs[α].device().clone();

        let tol = self.resp_cfg.tol;
        let max_cycle = self.resp_cfg.max_cycle;
        let max_space = self.resp_cfg.max_space;
        let lindep = self.resp_cfg.lindep;
        let tol_inflation = self.resp_cfg.tol_inflation;

        let pack_flattened = |x: &[TsrView; 2]| -> Tsr {
            // original: [nmo_α, nocc_α, nprop] and [nmo_β, nocc_β, nprop]
            // target: [nmo_α * nocc_α + nmo_β * nocc_β, nprop]
            assert_eq!(x[α].ndim(), 3, "Expected x[α] to have shape [nmo_α, nocc_α, nprop]");
            assert_eq!(x[β].ndim(), 3, "Expected x[β] to have shape [nmo_β, nocc_β, nprop]");
            let nprop = x[α].shape()[2];
            let mut x_flattened = rt::zeros(([nmo[α] * nocc[α] + nmo[β] * nocc[β], nprop], &device));
            for A in 0..nprop {
                x_flattened.i_mut((..nmo[α] * nocc[α], A)).assign(x[α].i((.., .., A)).reshape(-1));
                x_flattened.i_mut((nmo[α] * nocc[α].., A)).assign(x[β].i((.., .., A)).reshape(-1));
            }
            x_flattened
        };

        let unpack_flattened = |x: TsrView| -> [Tsr; 2] {
            // original: [nmo_α * nocc_α + nmo_β * nocc_β, nprop]
            // target: [nmo_α, nocc_α, nprop] and [nmo_β, nocc_β, nprop]
            assert_eq!(x.ndim(), 2, "Expected x to have shape [nmo_α * nocc_α + nmo_β * nocc_β, nprop]");
            let nprop = x.shape()[1];
            let idx_split = nmo[α] * nocc[α];
            let mut x_α = rt::zeros(([nmo[α], nocc[α], nprop], &device));
            let mut x_β = rt::zeros(([nmo[β], nocc[β], nprop], &device));
            for A in 0..nprop {
                x_α.i_mut((.., .., A)).assign(x.i((..idx_split, A)).reshape((nmo[α], nocc[α])));
                x_β.i_mut((.., .., A)).assign(x.i((idx_split.., A)).reshape((nmo[β], nocc[β])));
            }
            [x_α, x_β]
        };

        let response_cpscf_flattened = |x: TsrView| -> Tsr {
            // split x by spin and reshape to original shape
            let [x_α, x_β] = unpack_flattened(x);
            // compute response by usual means
            let resp = self.response_dimless_cpscf(&[x_α.view(), x_β.view()]);
            // flatten resp to shape (nmo*nocc, nprop)
            let resp_view = resp.iter().map(|r| r.view()).collect_array().unwrap();
            pack_flattened(&resp_view)
        };

        let rhs_view = rhs.iter().map(|r| r.view()).collect_array().unwrap();
        let rhs_packed = pack_flattened(&rhs_view);
        let mo1_flattened = krylov_block(
            response_cpscf_flattened,
            rhs_packed.view(),
            None,
            tol,
            max_cycle,
            max_space,
            lindep,
            tol_inflation,
        );
        let [mo1_α, mo1_β] = unpack_flattened(mo1_flattened.view());
        let mo1_α = mo1_α.into_shape(rhs_shape[α].to_vec());
        let mo1_β = mo1_β.into_shape(rhs_shape[β].to_vec());
        [mo1_α, mo1_β]
    }
}

/// Build the response (fock/response) objects for a converged unrestricted SCF.
pub fn uscf_resp_interface<'a>(scf_data: &'a SCF, config: &AnalDrvConfig) -> URespSCF<'a> {
    let device = DeviceBLAS::default();
    let mut resp_list: Vec<Box<dyn URespAPI + 'a>> = Vec::new();

    // --- RI-JK --- //

    let (factor_j, factor_k, rsh) = scf_jk_factors(scf_data);

    {
        let (rimatr, _, _) = scf_data.rimatr.as_ref().expect(
            "This implementation requires cholesky decomposed ERI (or rimatr) to be available and stored in memory.",
        );
        let cderi = rimatr.to_rstsr_view(&device).into_cow();
        resp_list.push(Box::new(URespRIJK::new_with_cderi(factor_j, factor_k, cderi)));
    }

    // The short-range exchange correction (range-separated hybrids) is a separate response
    // object reusing the full-range implementation; it evaluates on the short-range `rimatr_sr`
    // ERI, with no Coulomb part (factor_j = 0).
    if let Some((_omega, factor_k_sr)) = rsh {
        let (rimatr_sr, _, _) = scf_data.rimatr_sr.as_ref().expect(
            "The range-separated response requires the short-range ERI (rimatr_sr) to be built and stored in memory.",
        );
        let cderi_sr = rimatr_sr.to_rstsr_view(&device).into_cow();
        resp_list.push(Box::new(URespRIJK::new_with_cderi(0.0, factor_k_sr, cderi_sr)));
    }

    // --- DFT --- //

    let is_hf = scf_data.mol.xc_data.dfa_compnt_scf.is_empty();
    if !is_hf {
        let mol = get_cint_mol(&scf_data.mol);
        let xc_func_list = scf_xc_func_list_uks(scf_data);
        let verbose = scf_data.mol.ctrl.print_level > 2;

        // common grid (fock path): the SCF grid as is; the fock path is a plain quadrature sum,
        // so it does not require the atom-grouped (ByAtom) ordering.
        let grids = scf_data.grids.as_ref().unwrap();
        let ni = NIMatmul::new(&mol, &grids.coordinates, &grids.weights, &grids.atm_idx, &grids.quadrature_weights);

        // response grid: when it coincides with the SCF grid, leave `ni_resp = None`; the
        // response then evaluates (and caches) on the common grid. Otherwise build a dedicated
        // (usually coarser) grid.
        let grid_gen_level = scf_data.mol.ctrl.grid_gen_level;
        let grid_resp_level = config.resp.grid_level.unwrap_or(grid_gen_level.max(3) - 2);
        let resp_obj = if grid_resp_level == grid_gen_level {
            URespKSNIMatmul::new(xc_func_list, ni, verbose)
        } else {
            let cpscf_grid = Grids::build_with_level(&scf_data.mol, grid_resp_level);
            let ni_resp = NIMatmul::new(
                &mol,
                &cpscf_grid.coordinates,
                &cpscf_grid.weights,
                &cpscf_grid.atm_idx,
                &cpscf_grid.quadrature_weights,
            );
            URespKSNIMatmul::new(xc_func_list, ni, verbose).set_ni_resp(ni_resp)
        };
        resp_list.push(Box::new(resp_obj));
    }

    URespSCF { resp_list, resp_cfg: config.resp.clone(), cpscf_state: None }
}
