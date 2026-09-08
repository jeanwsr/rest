//! Response (fock/response) objects for restricted SCF, built from SCF data.
//!
//! This is the response-side counterpart of the hessian's
//! [`rscf_hess_interface`](crate::analdrv::hessian::rscf_interface::rscf_hess_interface): it
//! assembles the `RRespAPI` objects for all electron-interaction contributions of a converged
//! restricted SCF. The hessian interface calls it and feeds the returned list to `RHessSCF`; a
//! future standalone response/property driver can call it directly, without any hessian
//! machinery.
//!
//! The common (large) DFT grid data is generated (and regrouped) once here; the response object
//! and the returned hessian-side `NIMatmul` are built from the same grid data as independent
//! instances, so no grid identity is shared between the hessian and response subsystems.

use crate::analdrv::prelude::*;
use crate::dft::numint_matmul::nimatmul::{regroup_grids_by_atom, NIMatmul};
use crate::dft::numint_matmul::resp_rks::RRespKSNIMatmul;
use crate::dft::xceff::prelude::{determine_den_type_from_list, XCDenType};
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

/// List of `(scale, functional)` pairs of the SCF XC functional (spin-unpolarized).
pub fn scf_xc_func_list(scf_data: &SCF) -> Vec<(f64, LibXCFunctional)> {
    let xc_code = &scf_data.mol.xc_data.dfa_compnt_scf;
    let xc_params = &scf_data.mol.xc_data.dfa_paramr_scf;
    xc_code
        .iter()
        .zip(xc_params.iter())
        .map(|(&code, &param)| (param, LibXCFunctional::from_number(code as _, LibXCSpin::Unpolarized)))
        .collect_vec()
}

/// The response (fock/response) objects of all electron-interaction contributions of a restricted
/// SCF, held by name (not as trait objects) so that drivers can borrow them as `&mut dyn RRespAPI`
/// while the concrete objects stay alive.
pub struct RRespSCFList<'a> {
    /// RI-JK full-range contribution.
    pub rijk: RRespRIJK<'a>,
    /// RI-JK short-range exchange correction (range-separated hybrids only).
    pub rijk_sr: Option<RRespRIJK<'a>>,
    /// DFT XC numerical-integration contribution (absent for pure HF).
    pub nimatmul: Option<RRespKSNIMatmul<'a>>,
}

impl<'a> RRespSCFList<'a> {
    /// Borrow all response objects as `RRespAPI` trait objects.
    pub fn iter_mut(&mut self) -> Vec<&mut (dyn RRespAPI + 'a)> {
        let mut list: Vec<&mut (dyn RRespAPI + 'a)> = vec![&mut self.rijk];
        if let Some(resp_obj) = self.rijk_sr.as_mut() {
            list.push(resp_obj);
        }
        if let Some(resp_obj) = self.nimatmul.as_mut() {
            list.push(resp_obj);
        }
        list
    }
}

/// Build the response (fock/response) objects for a converged restricted SCF.
///
/// # Returns
///
/// - `resp_objs` : the response objects of all electron-interaction contributions.
/// - `ni_hess` : a `NIMatmul` over the common (large) DFT grid for the hessian skeleton setup,
///   `None` for pure HF; built from the same grid data as the response object's grid, as an
///   independent instance.
pub fn rscf_resp_interface<'a>(
    scf_data: &'a SCF,
    config: &AnalDrvConfig,
) -> (RRespSCFList<'a>, Option<NIMatmul<'a>>) {
    let device = DeviceBLAS::default();

    // --- RI-JK --- //

    let (factor_j, factor_k, rsh) = scf_jk_factors(scf_data);

    let rijk = {
        let (rimatr, _, _) = scf_data.rimatr.as_ref().expect(
            "This implementation requires cholesky decomposed ERI (or rimatr) to be available and stored in memory.",
        );
        let cderi = rimatr.to_rstsr_view(&device).into_cow();
        RRespRIJK::new_with_cderi(factor_j, factor_k, cderi)
    };

    // The short-range exchange correction (range-separated hybrids) is a separate response
    // object reusing the full-range implementation; it evaluates on the short-range `rimatr_sr`
    // ERI, with no Coulomb part (factor_j = 0).
    let rijk_sr = rsh.map(|(_omega, factor_k_sr)| {
        let (rimatr_sr, _, _) = scf_data
            .rimatr_sr
            .as_ref()
            .expect("The range-separated response requires the short-range ERI (rimatr_sr) to be built and stored in memory.");
        let cderi_sr = rimatr_sr.to_rstsr_view(&device).into_cow();
        RRespRIJK::new_with_cderi(0.0, factor_k_sr, cderi_sr)
    });

    // --- DFT --- //

    let is_hf = scf_data.mol.xc_data.dfa_compnt_scf.is_empty();
    let mut nimatmul = None;
    let mut ni_hess: Option<NIMatmul<'a>> = None;
    if !is_hf {
        let mol = get_cint_mol(&scf_data.mol);
        let xc_func_list = scf_xc_func_list(scf_data);
        let verbose = scf_data.mol.ctrl.print_level > 2;

        // Determine common-grid (skeleton-level) / cpscf grid levels.
        // - common grid: the SCF DFT grid; only MGGA (TAU) without the grid-shift adds 2 levels
        //   (the grid-shift terms restore the grid-related accuracy the finer grid compensated).
        // - cpscf:     grid_gen_level.max(3) - 2 (coarser, for the iterative CP-SCF response).
        let xc_type = determine_den_type_from_list(&xc_func_list.iter().map(|(_, f)| f).collect_vec());
        let is_mgga = matches!(xc_type, XCDenType::TAU);
        let grid_gen_level = scf_data.mol.ctrl.grid_gen_level;
        let grid_shift = config.nucgrad.grid_shift_deriv;
        let sk_level = config.nucgrad.grid_level_skeleton.unwrap_or(if is_mgga && !grid_shift {
            grid_gen_level + 2
        } else {
            grid_gen_level
        });
        let cpscf_level = config.cpscf.grid_level.unwrap_or(grid_gen_level.max(3) - 2);

        // common grid: reuse the SCF grid when the level matches, else regenerate.  Either
        // way, regroup to atom-grouped order (non-decreasing atm_idx): the SCF grid is
        // round-robin permuted for load balancing, while the Becke grid-shift attribution
        // requires the ByAtom grouping.  The regrouping only permutes, never changes values.
        //
        // The grid data is built once; the response object's grid and the returned hessian-side
        // grid are two independent `NIMatmul` instances over the same data.
        let (coordinates, weights, atm_idx, quadrature_weights) = if sk_level == grid_gen_level {
            let grids = scf_data.grids.as_ref().unwrap();
            (
                grids.coordinates.clone(),
                grids.weights.clone(),
                grids.atm_idx.clone(),
                grids.quadrature_weights.clone(),
            )
        } else {
            let sk_grid = Grids::build_with_level(&scf_data.mol, sk_level);
            (sk_grid.coordinates, sk_grid.weights, sk_grid.atm_idx, sk_grid.quadrature_weights)
        };
        let (coordinates, weights, atm_idx, quadrature_weights) =
            regroup_grids_by_atom(coordinates, weights, atm_idx, quadrature_weights, mol.natm());
        let ni = NIMatmul::new(&mol, &coordinates, &weights, &atm_idx, &quadrature_weights);
        let ni_hess_obj = NIMatmul::new(&mol, &coordinates, &weights, &atm_idx, &quadrature_weights);

        // response grid: when it coincides with the common grid, leave `ni_resp = None`; the
        // response then evaluates (and caches) on the common grid. Otherwise build a dedicated
        // (coarser) grid.
        let resp_nimatmul_obj = if cpscf_level == sk_level {
            RRespKSNIMatmul::new(xc_func_list, ni, verbose)
        } else {
            let cpscf_grid = Grids::build_with_level(&scf_data.mol, cpscf_level);
            let ni_resp = NIMatmul::new(
                &mol,
                &cpscf_grid.coordinates,
                &cpscf_grid.weights,
                &cpscf_grid.atm_idx,
                &cpscf_grid.quadrature_weights,
            );
            RRespKSNIMatmul::new(xc_func_list, ni, verbose).set_ni_resp(ni_resp)
        };
        nimatmul = Some(resp_nimatmul_obj);

        ni_hess = Some(ni_hess_obj);
    }

    (RRespSCFList { rijk, rijk_sr, nimatmul }, ni_hess)
}
