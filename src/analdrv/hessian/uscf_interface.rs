use crate::analdrv::prelude::*;
use crate::analdrv::vibration::vib::*;
use crate::analdrv::vibration::vib_interface::*;
use crate::dftd::hess::HessDFTD;
use crate::ri_jk::util::{get_cint_aux, get_cint_mol};
use crate::SCF;

pub fn uscf_hess_interface(scf_data: &SCF, config: &AnalDrvConfig) -> (Vec<f64>, VibInfo, Option<GauThermoInfo>) {
    let device = DeviceBLAS::default();

    // --- basic preparation --- //
    let mo_coeff = {
        let mo_coeff_0 = &scf_data.eigenvectors[0];
        let mo_coeff_1 = &scf_data.eigenvectors[1];
        let mo_coeff_0 = rt::asarray((&mo_coeff_0.data, mo_coeff_0.size, &device)).into_contig(ColMajor);
        let mo_coeff_1 = rt::asarray((&mo_coeff_1.data, mo_coeff_1.size, &device)).into_contig(ColMajor);
        [mo_coeff_0, mo_coeff_1]
    };
    let mo_occ = {
        let mo_occ_0 = &scf_data.occupation[0];
        let mo_occ_1 = &scf_data.occupation[1];
        let mo_occ_0 = rt::asarray((mo_occ_0, [mo_occ_0.len()], &device)).into_contig(ColMajor);
        let mo_occ_1 = rt::asarray((mo_occ_1, [mo_occ_1.len()], &device)).into_contig(ColMajor);
        [mo_occ_0, mo_occ_1]
    };
    let mo_energy = {
        let mo_energy_0 = &scf_data.eigenvalues[0];
        let mo_energy_1 = &scf_data.eigenvalues[1];
        let mo_energy_0 = rt::asarray((mo_energy_0, [mo_energy_0.len()], &device)).into_contig(ColMajor);
        let mo_energy_1 = rt::asarray((mo_energy_1, [mo_energy_1.len()], &device)).into_contig(ColMajor);
        [mo_energy_0, mo_energy_1]
    };

    let mol_obj = &scf_data.mol;
    let mol = get_cint_mol(mol_obj);
    let aux = get_cint_aux(mol_obj);

    let mut hess_ovlp_obj = UHessOvlp::new(&mol, &device);
    let mut hess_nuc_repl_obj = HessNucRepl::new(&mol, &device);
    let mut hess_hcore_obj = UHessHcore::new(&mol, &device);

    let mut hess_nuc_list: Vec<&mut dyn HessNucAPI> = vec![&mut hess_nuc_repl_obj];

    // --- empirical dispersion (DFTD3/DFTD4) --- //

    // The dispersion energy is independent of the density matrix (nuclear-like term). Its
    // Hessian is evaluated numerically from the analytic dispersion gradient, and is only
    // added if empirical dispersion is specified in the input.
    let mut hess_dftd_obj = HessDFTD::new(mol_obj, config.nucgrad.dftd_hess_step);
    if let Some(ref mut hess_dftd_obj) = hess_dftd_obj {
        hess_nuc_list.push(hess_dftd_obj);
    }
    let hess_hcore_list: Vec<&mut dyn UHessCoreAPI> = vec![&mut hess_hcore_obj];
    let mut hess_el_list: Vec<&mut dyn UHessElecInteractAPI> = Vec::new();

    // --- RI-JK --- //

    use crate::ri_jk::hess_u::UHessRIJK;

    let is_hf = scf_data.mol.xc_data.dfa_compnt_scf.is_empty();
    let scale_j = 1.0;
    // For range-separated hybrids the exchange is decomposed as (same as the SCF Fock assembly)
    //   K = alpha * K_full - (alpha - hyb) * K_erfc,
    // so the full-range object carries the long-range coefficient `alpha`, and the short-range
    // correction is a separate object evaluated below.
    let scale_k = match is_hf {
        true => 1.0,
        false => scf_data.mol.xc_data.rsh_alpha().unwrap_or(scf_data.mol.xc_data.dfa_hybrid_scf),
    };
    let j2c_decomp_option = &scf_data.mol.ctrl.j2c_decomp;
    let j2c_decomp = crate::ri_jk::get_j2c_decomp(&aux, &device, *j2c_decomp_option);

    let mut hess_rijk_obj = if let Some((rimatr, _, _)) = &scf_data.rimatr {
        let cderi = rimatr.to_rstsr_view(&device).into_cow();
        UHessRIJK::new_with_cderi(&mol, &aux, scale_j, scale_k, cderi, j2c_decomp)
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
    let mut hess_rijk_sr_obj = scf_data.mol.xc_data.omega().map(|omega| {
        let alpha = scf_data.mol.xc_data.rsh_alpha().unwrap();
        let hyb = scf_data.mol.xc_data.dfa_hybrid_scf;
        let (mut mol_sr, mut aux_sr) = (mol.clone(), aux.clone());
        // negative omega in libcint's convention evaluates the short-range erfc kernel
        mol_sr.set_omega(-omega);
        aux_sr.set_omega(-omega);
        if let Some((rimatr_sr, _, _)) = &scf_data.rimatr_sr {
            let cderi_sr = rimatr_sr.to_rstsr_view(&device).into_cow();
            let j2c_decomp_sr = crate::ri_jk::get_j2c_decomp(&aux_sr, &device, *j2c_decomp_option);
            UHessRIJK::new_with_cderi(&mol_sr, &aux_sr, 0.0, hyb - alpha, cderi_sr, j2c_decomp_sr)
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

    let mut hess_nimatmul_obj = (!is_hf).then(|| {
        use crate::dft::numint_matmul::hess_uks::UHessKSNIMatmul;
        use crate::dft::numint_matmul::nimatmul::NIMatmul;
        use crate::dft::xceff::prelude::{determine_den_type_from_list, XCDenType};
        use crate::dft::Grids;
        use libxc::prelude::*;

        let xc_func_list = {
            let xc_code = &scf_data.mol.xc_data.dfa_compnt_scf;
            let xc_params = &scf_data.mol.xc_data.dfa_paramr_scf;
            xc_code
                .iter()
                .zip(xc_params.iter())
                .map(|(&code, &param)| (param, LibXCFunctional::from_number(code as _, LibXCSpin::Polarized)))
                .collect_vec()
        };
        let verbose = scf_data.mol.ctrl.print_level > 2;

        // Determine skeleton / cpscf grid levels.
        // - skeleton: the SCF DFT grid; only MGGA (TAU) without the grid-shift adds 2 levels
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

        // skeleton grid: reuse the SCF grid when the level matches, else regenerate.  Either
        // way, regroup to atom-grouped order (non-decreasing atm_idx): the SCF grid is
        // round-robin permuted for load balancing, while the Becke grid-shift attribution
        // requires the ByAtom grouping.  The regrouping only permutes, never changes values.
        let ni = {
            use crate::dft::numint_matmul::nimatmul::regroup_grids_by_atom;

            let (coordinates, weights, atm_idx, quadrature_weights) = if sk_level == grid_gen_level {
                let grids = scf_data.grids.as_ref().unwrap();
                (
                    grids.coordinates.clone(),
                    grids.weights.clone(),
                    grids.atm_idx.clone(),
                    grids.quadrature_weights.clone(),
                )
            } else {
                let sk_grid = Grids::build_with_level(mol_obj, sk_level);
                (sk_grid.coordinates, sk_grid.weights, sk_grid.atm_idx, sk_grid.quadrature_weights)
            };
            let (coordinates, weights, atm_idx, quadrature_weights) =
                regroup_grids_by_atom(coordinates, weights, atm_idx, quadrature_weights, mol.natm());
            NIMatmul::new(&mol, &coordinates, &weights, &atm_idx, &quadrature_weights)
        };

        // cpscf grid: when it coincides with the skeleton grid, leave `ni_cpks = None` so the
        // skeleton's vxc/fxc are reused; otherwise build a dedicated (coarser) grid.
        let hess_nimatmul_obj = if cpscf_level == sk_level {
            UHessKSNIMatmul::new(&mol, xc_func_list, ni, grid_shift, verbose)
        } else {
            let cpscf_grid = Grids::build_with_level(mol_obj, cpscf_level);
            let ni_cpks = NIMatmul::new(
                &mol,
                &cpscf_grid.coordinates,
                &cpscf_grid.weights,
                &cpscf_grid.atm_idx,
                &cpscf_grid.quadrature_weights,
            );
            UHessKSNIMatmul::new(&mol, xc_func_list, ni, grid_shift, verbose).set_ni_cpks(ni_cpks)
        };
        hess_nimatmul_obj
    });
    if let Some(ref mut hess_nimatmul_obj) = hess_nimatmul_obj {
        hess_el_list.push(hess_nimatmul_obj);
    }

    // --- run hessian --- //

    let mut hess_scf = UHessSCF::new(
        mo_coeff,
        mo_occ,
        mo_energy,
        &mut hess_ovlp_obj,
        hess_nuc_list,
        hess_hcore_list,
        hess_el_list,
        config,
    );

    let de_hess = hess_scf.make_hess();

    if scf_data.mol.ctrl.print_level >= 2 {
        println!("=== HESSIAN ===");
        println!("Print hessian in [tA, sB] format (component xyz first, atom then)");
        println!("");
        // print hessian matrix [t, s, A, B] -> [tA, sB]
        let natm = de_hess.shape()[3];
        let hess_mat = de_hess.transpose([0, 2, 1, 3]).into_shape((3 * natm, 3 * natm));
        // print 6 columns at a time, with index header
        for j in (0..3 * natm).step_by(6) {
            let j_end = (j + 6).min(3 * natm);
            let col_header = " ".repeat(6) + &(j..j_end).map(|i| format!("{:>12}", i)).collect::<String>();
            println!("{}", col_header);
            for i in 0..3 * natm {
                let row_str =
                    format!("{i:>4}  ") + &(j..j_end).map(|j| format!("{:12.6}", hess_mat[[i, j]])).collect::<String>();
                println!("{}", row_str);
            }
            println!("");
        }
    }

    // print timing information
    if scf_data.mol.ctrl.print_level >= 2 {
        println!("Timing in Hessian calculation:");
        for (key, value) in hess_scf.timing.iter() {
            println!("    {:60}: {:10.6} seconds", key, value);
        }
    }

    // --- perform vibrational analysis --- //

    vibration_analysis_interface(scf_data, config, de_hess.view())
}
