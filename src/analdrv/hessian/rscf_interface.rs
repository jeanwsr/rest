use crate::analdrv::prelude::*;
use super::print_hessian_and_timing;
use crate::analdrv::response::rresp_interface::{scf_jk_factors, scf_xc_func_list};
use crate::analdrv::vibration::vib::*;
use crate::analdrv::vibration::vib_interface::*;
use crate::dft::gen_grids::RadiiAdjust;
use crate::dft::numint_matmul::nimatmul::{regroup_grids_by_atom, NIMatmul};
use crate::dft::Grids;
use crate::dft::xceff::prelude::{determine_den_type_from_list, XCDenType};
use crate::dftd::hess::HessDFTD;
use crate::ri_jk::util::{get_cint_aux, get_cint_mol};
use crate::SCF;

/// Restricted SCF hessian: assembles the driver (overlap/nuclear/core/electronic parts plus
/// the response object), evaluates the hessian and performs the vibrational analysis.
///
/// `resp_obj` is the SCF response object (composite [`RRespSCF`], RI-JK plus XC parts) that
/// provides the CP-SCF solve for the relaxed density. It is not built here: initialize it
/// beforehand with [`crate::analdrv::response::rresp_interface::rscf_resp_interface`] on the
/// same `scf_data`. The driver mutates it (response preparation and cached intermediates), and
/// a shared object can be reused by further tasks afterwards.
pub fn rscf_hess_interface<'a>(
    scf_data: &'a SCF,
    cfg: &AnalDrvNucgradCfg,
    resp_obj: &mut RRespSCF<'a>,
) -> (Vec<f64>, VibInfo, Option<GauThermoInfo>) {
    let device = DeviceBLAS::default();

    // --- basic preparation --- //
    let mo_coeff = {
        let mo_coeff = &scf_data.eigenvectors[0];
        rt::asarray((&mo_coeff.data, mo_coeff.size, &device)).into_contig(ColMajor)
    };
    let mo_occ = {
        let mo_occ = &scf_data.occupation[0];
        rt::asarray((mo_occ, [mo_occ.len()], &device)).into_contig(ColMajor)
    };
    let mo_energy = {
        let mo_energy = &scf_data.eigenvalues[0];
        rt::asarray((mo_energy, [mo_energy.len()], &device)).into_contig(ColMajor)
    };

    let mol_obj = &scf_data.mol;
    let mol = get_cint_mol(mol_obj);
    let aux = get_cint_aux(mol_obj);

    let mut hess_ovlp_obj = RHessOvlp::new(&mol, &device);
    let mut hess_nuc_repl_obj = HessNucRepl::new(&mol, &device);
    let mut hess_hcore_obj = RHessHcore::new(&mol, &device);

    let mut hess_nuc_list: Vec<&mut dyn HessNucAPI> = vec![&mut hess_nuc_repl_obj];

    // --- empirical dispersion (DFTD3/DFTD4) --- //

    // The dispersion energy is independent of the density matrix (nuclear-like term). Its
    // Hessian is evaluated numerically from the analytic dispersion gradient, and is only
    // added if empirical dispersion is specified in the input.
    let mut hess_dftd_obj = HessDFTD::new(mol_obj, cfg.dftd_hess_step);
    if let Some(ref mut hess_dftd_obj) = hess_dftd_obj {
        hess_nuc_list.push(hess_dftd_obj);
    }
    let hess_hcore_list: Vec<&mut dyn RHessCoreAPI> = vec![&mut hess_hcore_obj];
    let mut hess_el_list: Vec<&mut dyn RHessElecInteractAPI> = Vec::new();

    // --- RI-JK --- //

    use crate::ri_jk::hess_r::RHessRIJK;

    let (scale_j, scale_k, rsh) = scf_jk_factors(scf_data);
    let j2c_decomp_option = &scf_data.mol.ctrl.j2c_decomp;
    let j2c_decomp = crate::ri_jk::get_j2c_decomp(&aux, &device, *j2c_decomp_option);

    let mut hess_rijk_obj = if let Some((rimatr, _, _)) = &scf_data.rimatr {
        let cderi = rimatr.to_rstsr_view(&device).into_cow();
        RHessRIJK::new_with_cderi(&mol, &aux, scale_j, scale_k, cderi, j2c_decomp)
    } else {
        panic!(
            "This implementation requires cholesky decomposed ERI (or rimatr) to be available and stored in memory."
        );
    };

    hess_el_list.push(&mut hess_rijk_obj);

    // --- RI-JK short-range part (range-separated hybrids) --- //

    // The short-range exchange correction -(alpha - hyb) * K_erfc is evaluated by a second
    // RI-JK Hessian object reusing the same implementation as the full-range one: the object
    // carries mol/aux clones holding the (negative) omega, so all its derivative integrals
    // are short-range (erfc kernel), and the cderi is the incore `rimatr_sr` built in the SCF.
    // There is no short-range Coulomb contribution (factor_j = 0).
    let mut hess_rijk_sr_obj = rsh.map(|(omega, factor_k_sr)| {
        let (mut mol_sr, mut aux_sr) = (mol.clone(), aux.clone());
        // negative omega in libcint's convention evaluates the short-range erfc kernel
        mol_sr.set_omega(-omega);
        aux_sr.set_omega(-omega);
        if let Some((rimatr_sr, _, _)) = &scf_data.rimatr_sr {
            let cderi_sr = rimatr_sr.to_rstsr_view(&device).into_cow();
            let j2c_decomp_sr = crate::ri_jk::get_j2c_decomp(&aux_sr, &device, *j2c_decomp_option);
            RHessRIJK::new_with_cderi(&mol_sr, &aux_sr, 0.0, factor_k_sr, cderi_sr, j2c_decomp_sr)
        } else {
            panic!(
                "The range-separated Hessian requires the short-range ERI (rimatr_sr) to be built and stored in memory."
            )
        }
    });
    if let Some(ref mut hess_rijk_sr_obj) = hess_rijk_sr_obj {
        hess_el_list.push(hess_rijk_sr_obj);
    }

    // --- DFT --- //

    // The skeleton-level grid of the XC hessian object is built by `scf_skeleton_nimatmul`
    // (SCF grid reused or regenerated per the skeleton level policy); no grid data is shared
    // with the response objects.
    let mut hess_nimatmul_obj = scf_skeleton_nimatmul(scf_data, cfg).map(|ni| {
        use crate::dft::numint_matmul::hess_rks::RHessKSNIMatmul;

        let verbose = scf_data.mol.ctrl.print_level > 2;
        let grid_shift = cfg.grid_shift_deriv;
        RHessKSNIMatmul::new(&mol, scf_xc_func_list(scf_data), ni, grid_shift, verbose)
    });
    if let Some(ref mut hess_nimatmul_obj) = hess_nimatmul_obj {
        hess_el_list.push(hess_nimatmul_obj);
    }

    // --- run hessian --- //

    // The driver and its borrowed lists live in an inner scope, so that the borrows of the
    // hessian/response objects provably end before the objects themselves are dropped.
    let de_hess = {
        let mut hess_scf = RHessSCF::new(
            mo_coeff,
            mo_occ,
            mo_energy,
            &mut hess_ovlp_obj,
            hess_nuc_list,
            hess_hcore_list,
            hess_el_list,
            resp_obj,
            cfg,
        );

        let de_hess = hess_scf.make_hess();

        print_hessian_and_timing(&de_hess, &hess_scf.timing, scf_data.mol.ctrl.print_level);

        de_hess
    };

    // --- perform vibrational analysis --- //

    vibration_analysis_interface(scf_data, cfg, de_hess.view())
}

/// The skeleton-level `NIMatmul` for the DFT XC hessian contribution, `None` for pure HF.
///
/// Shared by the unrestricted hessian interface: the grid policy is spin-independent (the
/// density-type determination depends only on the functional families, and the grid data itself
/// carries no spin treatment).
///
/// The skeleton grid level is the SCF grid level, raised by 2 for MGGA (TAU) functionals when
/// the grid-shift derivative terms are off (the grid-shift terms restore the grid-related
/// accuracy the finer grid compensated). The grid reuses the SCF grid when the level matches,
/// else is regenerated; either way it is regrouped to atom-grouped order (non-decreasing
/// `atm_idx`): the SCF grid is round-robin permuted for load balancing, while the Becke
/// grid-shift attribution requires the ByAtom grouping. The regrouping only permutes, never
/// changes values.
pub(crate) fn scf_skeleton_nimatmul<'a>(scf_data: &'a SCF, cfg: &AnalDrvNucgradCfg) -> Option<NIMatmul<'a>> {
    if scf_data.mol.xc_data.dfa_compnt_scf.is_empty() {
        return None;
    }
    let mol = get_cint_mol(&scf_data.mol);

    let xc_func_list = scf_xc_func_list(scf_data);
    let xc_type = determine_den_type_from_list(&xc_func_list.iter().map(|(_, f)| f).collect_vec());
    let is_mgga = matches!(xc_type, XCDenType::TAU);
    let grid_gen_level = scf_data.mol.ctrl.grid_gen_level;
    let grid_shift = cfg.grid_shift_deriv;
    let sk_level = cfg.grid_level_skeleton.unwrap_or(if is_mgga && !grid_shift {
        grid_gen_level + 2
    } else {
        grid_gen_level
    });

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
    // the Becke grid-shift terms of the hessian rebuild the partition from the atomic radii, so
    // the driver must know the radii adjustment scheme the grid was generated with
    let radii_adjust = RadiiAdjust::from_str(&scf_data.mol.ctrl.radii_adjust);
    Some(NIMatmul::new(&mol, &coordinates, &weights, &atm_idx, &quadrature_weights).with_radii_adjust(radii_adjust))
}
