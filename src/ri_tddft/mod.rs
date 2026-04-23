use std::time::Instant;
use tensors::{MatrixFull, MathMatrix};
use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;

use crate::scf_io::SCF;
use crate::ctrl_io::tddft_parameters::TDDFTParameters;
use crate::ctrl_io::quasiparticle_methods::QuasiParticle;
use crate::ri_gw::get_occupation_parameters;
use crate::ri_bse;
use crate::ri_bse::davidson_solver;

pub mod fxc_kernel;
pub mod matvec;

/// Build a QuasiParticle struct from TDDFTParameters for reusing the Davidson solver
fn tddft_to_qp(p: &TDDFTParameters) -> QuasiParticle {
    let mut qp = QuasiParticle::default();
    qp.davidson_target_excitations = p.nroots;
    qp.davidson_converge_threshold = p.davidson_tol;
    qp.davidson_max_iter = p.davidson_max_iter;
    qp.davidson_maximum_subspace_size = p.davidson_max_subspace * p.nroots;
    qp.davidson_restart_dimensions = p.nroots + 2;
    qp.davidson_add_dimensions = p.nroots + 2;
    qp
}

pub fn tddft_main(scf_data: &mut SCF) {
    let start = Instant::now();
    // Clone tddft_ctrl early to avoid holding an immutable borrow while we
    // temporarily mutate scf_data.mol.ctrl.quasiparticle_methods below.
    let tddft_ctrl = scf_data.mol.ctrl.tddft.clone()
        .expect("TDDFT parameters must be set");

    println!("=== LR-TDDFT Calculation ===");

    // Temporarily inject a default QuasiParticle so that get_occupation_parameters
    // (which reads quasiparticle_methods) does not panic when no GW block is present.
    let had_qp = scf_data.mol.ctrl.quasiparticle_methods.is_some();
    if !had_qp {
        scf_data.mol.ctrl.quasiparticle_methods = Some(QuasiParticle::default());
    }

    let (start_mo, num_state, occ_size, vir_size, homo, lumo) =
        get_occupation_parameters(scf_data, 'N');

    // Release the temporary QuasiParticle if we injected it.
    println!("occ_size={}, vir_size={}, homo={}, lumo={}", occ_size, vir_size, homo, lumo);

    // Prepare fxc kernel data
    let tddft_data = fxc_kernel::prepare_tddft_data(scf_data);

    // Prepare RI integrals
    let ri_ov = ri_bse::get_submatrix(scf_data, 'O', 'V', 'N');
    let num_auxbas = ri_ov.size[0];

    let mut ri_oo = ri_bse::get_submatrix(scf_data, 'O', 'O', 'N');
    ri_oo.reshape([num_auxbas * occ_size, occ_size]);
    ri_oo = ri_oo.transpose_and_drop();
    ri_oo.reshape([occ_size * num_auxbas, occ_size]);

    let mut ri_vv = ri_bse::get_submatrix(scf_data, 'V', 'V', 'N');
    ri_vv.reshape([num_auxbas * vir_size, vir_size]);

    let mut ri_ov_b = ri_ov.clone();
    ri_ov_b.reshape([num_auxbas * occ_size, vir_size]);

    let eigenvalues = scf_data.eigenvalues[0].clone();
    let energy_diag = ri_bse::construct_energy_diag_for_a(&eigenvalues, occ_size, vir_size);

    let nroots = tddft_ctrl.nroots;
    let qp_ctrl = tddft_to_qp(&tddft_ctrl);
    let initial_guess = davidson_solver::generate_initial_guess(&energy_diag, nroots);

    let prep_time = start.elapsed();
    println!("TDDFT Preparation Time: {:?}", prep_time);

    let is_singlet = tddft_ctrl.tddft_spin != "triplet";
    let spin_label = if is_singlet { "Singlet" } else { "Triplet" };
    let use_tda = tddft_ctrl.tddft_method == "tda";

    if use_tda {
        println!("TDA-TDDFT {} Calculation with Davidson solver", spin_label);
        let a_matvec = |z: &Vec<f64>| -> Vec<f64> {
            matvec::tddft_a_matvec(
                &tddft_data, &ri_ov, &ri_oo, &ri_vv,
                &eigenvalues, is_singlet, z,
            )
        };
        let excitations = davidson_solver::tda_davidson_solver(
            scf_data.mol.ctrl.print_level,
            a_matvec,
            nroots,
            &energy_diag,
            initial_guess,
            &qp_ctrl,
        );
        println!("TDA-TDDFT {} Excitation Energies:", spin_label);
        for (n, (e, _vec)) in excitations.iter().enumerate().take(nroots) {
            println!("  #{}: {:.6} Ha = {:.4} eV", n + 1, e, e * 27.211386);
        }
        println!("Davidson Solver took {:?}", start.elapsed() - prep_time);
    } else {
        println!("Full LR-TDDFT {} Calculation with Davidson solver", spin_label);
        let a_matvec = |z: &Vec<f64>| -> Vec<f64> {
            matvec::tddft_a_matvec(
                &tddft_data, &ri_ov, &ri_oo, &ri_vv,
                &eigenvalues, is_singlet, z,
            )
        };
        let b_matvec = |z: &Vec<f64>| -> Vec<f64> {
            let n = occ_size * vir_size;
            let mut result = vec![0.0; n];

            let fxc_z = matvec::fxc_matvec(&tddft_data, is_singlet, z);
            for i in 0..n {
                result[i] += fxc_z[i];
            }

            if is_singlet {
                let v_z = ri_bse::matvec::coulomb_contribution(&ri_ov, z);
                for i in 0..n {
                    result[i] += 2.0 * v_z[i];
                }
            }

            if tddft_data.alpha_hybrid.abs() > 1.0e-10 {
                let kb_z = matvec::exchange_b_matvec(
                    &ri_ov, &ri_ov_b, occ_size, vir_size, z,
                );
                for i in 0..n {
                    result[i] -= tddft_data.alpha_hybrid * kb_z[i];
                }
            }
            result
        };
        let excitations = davidson_solver::lr_davidson_solver(
            scf_data.mol.ctrl.print_level,
            a_matvec,
            b_matvec,
            nroots,
            &energy_diag,
            initial_guess,
        );
        println!("Full LR-TDDFT {} Excitation Energies:", spin_label);
        for (n, (e, _vec)) in excitations.iter().enumerate().take(nroots) {
            println!("  #{}: {:.6} Ha = {:.4} eV", n + 1, e, e * 27.211386);
        }
        println!("LR Davidson Solver took {:?}", start.elapsed() - prep_time);
    }
    if !had_qp {
        scf_data.mol.ctrl.quasiparticle_methods = None;
    }
}
