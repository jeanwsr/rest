use crate::scf_io::{scf, SCFType, SCF};
use crate::scf_io;
use crate::utilities::TimeRecords;
use crate::mpi_io::MPIOperator;
use crate::main_driver::{collect_total_energy, performance_essential_calculations};
use crate::post_scf_analysis::save_chkfile;


pub fn apply_yamaguchi_spin_correction(scf_data: &mut SCF, time_mark: &mut TimeRecords, mpi_operator: &Option<MPIOperator>) {
    // See Yamaguchi, K et al. Chem. Phys. Lett. 1988, 149, 537–542
    println!("==========================================");
    println!("Now apply the Yamaguchi spin correction.");
    println!("==========================================");

    let scf_energy_singlet = scf_data.scf_energy;
    let tot_energy_singlet = collect_total_energy(scf_data);

    let [square_spin_singlet, _] = scf_io::evaluate_spin_angular_momentum(&scf_data.density_matrix, &scf_data.ovlp, scf_data.mol.spin_channel, &scf_data.mol.num_elec);

    if scf_data.mol.ctrl.spin == 1.0 && square_spin_singlet >= 1e-3 {
        if scf_data.mol.ctrl.restart {
            save_chkfile(&scf_data);  // save singlet wavefunction
        }
        time_mark.new_item("spin_correction", "the whole job");
        time_mark.count_start("spin_correction");
        scf_data.mol.ctrl.guess_mix = false;
        scf_data.mol.ctrl.force_state_occupation = vec![];
        //scf_data.mol.ctrl.chkfile.push_str("_triplet");
        scf_data.mol.ctrl.restart = false;
        scf_data.mol.num_elec[1] += 1.0;
        scf_data.mol.num_elec[2] -= 1.0;
        scf_data.mol.ctrl.initial_guess = String::from("inherit");
        match scf_data.mol.ctrl.yamaguchi_triplet_type.as_deref() {
            Some("r") | Some("ro") => {
                println!("Applying ROHF method (with level shift) for triplet energy calculation.");
                scf_data.mol.ctrl.level_shift = Some(0.5);
                scf_data.scftype = SCFType::ROHF;
            },
            Some("u") | None => {
                println!("Applying UHF method for triplet energy calculation.");
                scf_data.scftype = SCFType::UHF;
            },
            Some(other) => {
                panic!("Unknown Yamaguchi spin correction type: '{}'. Expected 'u' or 'r'('ro').", other);
            }
        }
        println!("Computing the triplet energy...");

                
        scf_io::initialize_scf(scf_data, &mpi_operator);
        performance_essential_calculations(scf_data, time_mark, &mpi_operator);

        let scf_energy_triplet = scf_data.scf_energy;
        let tot_energy_triplet = collect_total_energy(scf_data);
        let [square_spin_triplet, _] = scf_io::evaluate_spin_angular_momentum(&scf_data.density_matrix, &scf_data.ovlp, scf_data.mol.spin_channel, &scf_data.mol.num_elec);

        // main formula
        let factor = square_spin_singlet / (square_spin_triplet - square_spin_singlet);
        let scf_energy_corrected = scf_energy_singlet - (scf_energy_triplet - scf_energy_singlet) * factor;
        let tot_energy_corrected = tot_energy_singlet - (tot_energy_triplet - tot_energy_singlet) * factor;
                
        println!("----------------------------------------------------------------------");
        println!("Report for Yamaguchi spin correction: ");
        println!("Open-shell singlet: scf_energy = {:18.10} Ha, tot_energy = {:18.10} Ha, <s^2> = {:6.3}.", scf_energy_singlet, tot_energy_singlet, square_spin_singlet);
        println!("Triplet: scf_energy = {:18.10} Ha, tot_energy = {:18.10} Ha, <s^2> = {:6.3}.", scf_energy_triplet, tot_energy_triplet, square_spin_triplet);
        println!("Corrected: scf_energy = {:18.10} Ha, tot_energy = {:18.10} Ha.", scf_energy_corrected, tot_energy_corrected);
        println!("----------------------------------------------------------------------");

        scf_data.energies.insert("yamaguchi_scf_corrected".to_string(), vec![scf_energy_corrected]);
        scf_data.energies.insert("yamaguchi_tot_corrected".to_string(), vec![tot_energy_corrected]);   
        time_mark.count("spin_correction");
    } else {
        println!("Yamaguchi spin correction skipped: either spin is not 1.0 or contamination is negligible.");
    }
}