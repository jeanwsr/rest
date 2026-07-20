use super::prelude::*;
use crate::analdrv::vib::*;
use crate::analdrv::vib_interface::*;
use crate::ri_jk::util::{get_cint_aux, get_cint_mol};
use crate::SCF;

pub fn rscf_hess_interface(scf_data: &SCF, config: &AnalDrvConfig) -> (Vec<f64>, VibInfo, Option<GauThermoInfo>) {
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

    let hess_nuc_list: Vec<&mut dyn HessNucAPI> = vec![&mut hess_nuc_repl_obj];
    let hess_hcore_list: Vec<&mut dyn RHessCoreAPI> = vec![&mut hess_hcore_obj];
    let mut hess_el_list: Vec<&mut dyn RHessElecInteractAPI> = Vec::new();

    // --- RI-JK --- //

    use crate::ri_jk::hess_r::RHessRIJK;

    let is_hf = scf_data.mol.xc_data.dfa_compnt_scf.is_empty();
    let scale_j = 1.0;
    let scale_k = match is_hf {
        true => 1.0,
        false => scf_data.mol.xc_data.dfa_hybrid_scf,
    };
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

    // --- DFT --- //

    let mut hess_nimatmul_obj = (!is_hf).then(|| {
        use crate::dft::numint_matmul::hess_rks::RHessKSNIMatmul;
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
                .map(|(&code, &param)| (param, LibXCFunctional::from_number(code as _, LibXCSpin::Unpolarized)))
                .collect_vec()
        };
        let verbose = scf_data.mol.ctrl.print_level >= 2;

        // Determine skeleton / cphf grid levels.
        // - skeleton: LDA/GGA use the SCF DFT grid; MGGA (TAU) adds 2 levels.
        // - cphf:     grid_gen_level.max(3) - 2 (coarser, for the iterative CP-KS response).
        let xc_type = determine_den_type_from_list(&xc_func_list.iter().map(|(_, f)| f).collect_vec());
        let is_mgga = matches!(xc_type, XCDenType::TAU);
        let grid_gen_level = scf_data.mol.ctrl.grid_gen_level;
        let sk_level = config.grid_level_skeleton.unwrap_or(if is_mgga { grid_gen_level + 2 } else { grid_gen_level });
        let cphf_level = config.grid_level_cphf.unwrap_or(grid_gen_level.max(3) - 2);

        // skeleton grid: reuse the SCF grid when the level matches, else regenerate.
        let ni = if sk_level == grid_gen_level {
            let grid_coords = &scf_data.grids.as_ref().unwrap().coordinates;
            let grid_weights = &scf_data.grids.as_ref().unwrap().weights;
            NIMatmul::new(&mol, grid_coords, grid_weights)
        } else {
            let sk_grid = Grids::build_with_level(mol_obj, sk_level);
            NIMatmul::new(&mol, &sk_grid.coordinates, &sk_grid.weights)
        };

        // cphf grid: when it coincides with the skeleton grid, leave `ni_cpks = None` so the
        // skeleton's vxc/fxc are reused; otherwise build a dedicated (coarser) grid.
        let hess_nimatmul_obj = if cphf_level == sk_level {
            RHessKSNIMatmul::new(&mol, xc_func_list, ni, verbose)
        } else {
            let cphf_grid = Grids::build_with_level(mol_obj, cphf_level);
            let ni_cpks = NIMatmul::new(&mol, &cphf_grid.coordinates, &cphf_grid.weights);
            RHessKSNIMatmul::new(&mol, xc_func_list, ni, verbose).set_ni_cpks(ni_cpks)
        };
        hess_nimatmul_obj
    });
    if let Some(ref mut hess_nimatmul_obj) = hess_nimatmul_obj {
        hess_el_list.push(hess_nimatmul_obj);
    }

    // --- run hessian --- //

    let mut hess_scf = RHessSCF::new(
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
