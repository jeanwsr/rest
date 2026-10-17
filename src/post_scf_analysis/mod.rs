#![warn(unused_imports)]
pub mod rand_wf_real_space;
pub mod cube_build;
pub mod hirshfeld;
pub mod molden_build;
pub mod mulliken;
pub mod strong_correlation_correction;
pub mod rrs_pbc;
pub mod spin_correction;

use rest_libcint::prelude::rest_libcint_wrapper::int1e_r;
use tensors::{MathMatrix, RIFull};

use crate::constants::{BOHR, AU2DEBYE};
use crate::grad::{formated_force, formated_force_ev, numerical_force};
use crate::grad::traits::GradAPI;
use crate::mpi_io::MPIOperator;
use crate::ri_pt2::sbge2::{close_shell_sbge2_rayon, open_shell_sbge2_rayon};
use crate::ri_rpa::scsrpa::{evaluate_osrpa_correlation_rayon};
use crate::ri_rpa::{evaluate_rpa_correlation_rayon};
use crate::ri_gw;
use crate::ri_bse;
use crate::scf_io::{SCF, SCFType};
use crate::scf_io::print::print_force_for_ghost_point_charges;
use crate::ri_pt2::{close_shell_pt2_rayon, open_shell_pt2_rayon};
use crate::utilities::TimeRecords;

use self::molden_build::{gen_molden};
use self::strong_correlation_correction::scc15_for_rxdh7;
pub use crate::fileop::chkfile::{save_chkfile, save_hamiltonian, save_overlap, save_geometry};

pub fn post_scf_output(scf_data: &SCF, mpi_operator: &Option<MPIOperator>) {
    scf_data.mol.ctrl.outputs.iter().for_each(|output_type| {
        if output_type.eq("fchk") {
            if let Some(mpi_op) = &mpi_operator {
                if mpi_op.rank == 0 {scf_data.save_fchk_of_gaussian();}
            } else {
                scf_data.save_fchk_of_gaussian();
            }
        } else if output_type.eq("fciqmc_dump") {
            if let Some(mpi_op) = &mpi_operator {
                if mpi_op.rank == 0 {fciqmc_dump(&scf_data)}
            } else {
                fciqmc_dump(&scf_data);
            }
        } else if output_type.eq("wfn_in_real_space") {
            if let Some(mpi_op) = &mpi_operator {
                panic!("The MPI version is not yet implemented for generating the wavefunction in real space");
            } else {
                let np = 100;
                let slater_determinant = rand_wf_real_space::slater_determinant(&scf_data, np);
                let output = serde_json::to_string(&slater_determinant).unwrap();
                let mut file = std::fs::File::create("./wf_in_real_space.txt").unwrap();
                std::io::Write::write(&mut file, output.as_bytes());
            }
        } else if output_type.eq("cube_orb") {
            println!("Now generating the cube files for given orbitals");
            if let Some(mpi_op) = &mpi_operator {
                panic!("The MPI version is not yet implemented for generating orbital cube files");
            } else {
                cube_build::get_cube_orb(&scf_data);
            }
        } else if output_type.eq("tabulated_exc") {
            println!("Now tabulating e_[xc] to each grid points");
            if let Some(mpi_op) = &mpi_operator {
                panic!("The MPI version is not yet implemented for tabulating e_[xc] to each grid points");
            } else {
                if let Some(grids) = &scf_data.grids {
                    let dm = &scf_data.density_matrix;
                    let mo = &scf_data.eigenvectors;
                    let occ = &scf_data.occupation;
                    scf_data.mol.xc_data.post_tabulated_exc(grids, dm, mo, occ);
                } else {
                    panic!("The grids are not yet initialized");
                }
            }
        } else if output_type.eq("molden") {
            if let Some(mpi_op) = &mpi_operator {
                panic!("The MPI version is not yet implemented for generating molden file");
            } else {
                molden_build::gen_molden(&scf_data);
            }
        } else if output_type.eq("hamiltonian") {
            if let Some(mpi_op) = &mpi_operator {
                if mpi_op.rank == 0 {save_hamiltonian(&scf_data);}
            } else {
                save_hamiltonian(&scf_data);
            }
        } else if output_type.eq("geometry") {
            if let Some(mpi_op) = &mpi_operator {
                if mpi_op.rank == 0 {
                    save_geometry(&scf_data);
                    scf_data.mol.geom.to_xyz("geometry.xyz".to_string());
                }
            } else {
                save_geometry(&scf_data);
                scf_data.mol.geom.to_xyz("geometry.xyz".to_string());
            }
        } else if output_type.eq("overlap") {
            if let Some(mpi_op) = &mpi_operator {
                if mpi_op.rank == 0 {save_overlap(&scf_data);}
            } else {
               save_overlap(&scf_data);
            }
        } else if output_type.eq("multiwfn") {
            if let Some(mpi_op) = &mpi_operator {
                panic!("The MPI version is not yet implemented for generating multiwfn file");
            } else {
                gen_molden(&scf_data);
            }
        } else if output_type.eq("deeph") {
            if let Some(mpi_op) = &mpi_operator {
                if mpi_op.rank == 0 {
                    save_hamiltonian(&scf_data);
                    save_geometry(&scf_data);
                    scf_data.mol.geom.to_xyz("geometry.xyz".to_string());
                }
            } else {
                save_hamiltonian(&scf_data);
                save_geometry(&scf_data);
                scf_data.mol.geom.to_xyz("geometry.xyz".to_string());
            }
        } else if output_type.eq("dipole") {
            if let Some(mpi_op) = &mpi_operator {
                if mpi_op.rank == 0 {
                    let dp = evaluate_dipole_moment(scf_data, None);
                    println!("Dipole Moment in DEBYE: {:16.8}, {:16.8}, {:16.8}", dp[0], dp[1], dp[2]);
                }
            } else {
                let dp = evaluate_dipole_moment(scf_data, None);
                println!("Dipole Moment in DEBYE: {:16.8}, {:16.8}, {:16.8}", dp[0], dp[1], dp[2]);
            }
        } else if output_type.eq("hirshfeld") || output_type.eq("hirshfeld_charge") {
            if mpi_operator.is_some() {
                panic!("The MPI version is not yet implemented for Hirshfeld charge analysis");
            }
            hirshfeld::print_hirshfeld_analysis(scf_data);
        } else if output_type.eq("num_force") {
            let displace = match scf_data.mol.geom.unit {
                crate::geom_io::GeomUnit::Angstrom => scf_data.mol.ctrl.nforce_displacement/BOHR,
                crate::geom_io::GeomUnit::Bohr => scf_data.mol.ctrl.nforce_displacement,
            };
            let (energy, num_force) = numerical_force(scf_data, displace, mpi_operator);
            if let Some(mpi_op) = &mpi_operator {
                if mpi_op.rank == 0 {
                    println!("Total atomic forces [a.u.]: ");
                    println!("{}", formated_force(&num_force, &scf_data.mol.geom.elem));
                    println!("Total atomic forces [ev/ang]: ");
                    println!("{}", formated_force_ev(&num_force, &scf_data.mol.geom.elem));
                }
            } else {
                println!("Total atomic forces [a.u.]: ");
                println!("{}", formated_force(&num_force, &scf_data.mol.geom.elem));
                println!("Total atomic forces [ev/ang]: ");
                println!("{}", formated_force_ev(&num_force, &scf_data.mol.geom.elem));
            }
        } else if output_type.eq("force_for_ghost_point_charges") {
            print_force_for_ghost_point_charges(&scf_data)
        }
    });
}


pub fn print_out_dfa(scf_data: &SCF) {
    let dfa = crate::dft::DFA4REST::new_xc(scf_data.mol.spin_channel, scf_data.mol.ctrl.print_level);
    let post_xc_energy = if let Some(grids) = &scf_data.grids {
        dfa.post_xc_exc(&scf_data.mol.ctrl.post_xc, grids, &scf_data.density_matrix, &scf_data.eigenvectors, &scf_data.occupation)
    } else {
        vec![[0.0,0.0]]
    };
    post_xc_energy.iter().zip(scf_data.mol.ctrl.post_xc.iter()).for_each(|(energy, name)| {
        println!("{:<16}: {:16.8} Ha", name, energy[0]+energy[1]);
    });
}

pub fn print_out_xc_potentials(scf_data: &SCF) {
    let dfa = crate::dft::DFA4REST::new_xc(scf_data.mol.spin_channel, scf_data.mol.ctrl.print_level);
    let post_xc_energy = if let Some(grids) = &scf_data.grids {
        dfa.post_xc_exc(&scf_data.mol.ctrl.post_xc, grids, &scf_data.density_matrix, &scf_data.eigenvectors, &scf_data.occupation)
    } else {
        vec![[0.0,0.0]]
    };
    post_xc_energy.iter().zip(scf_data.mol.ctrl.post_xc.iter()).for_each(|(energy, name)| {
        println!("{:<16}: {:16.8} Ha", name, energy[0]+energy[1]);
    });
}

pub fn post_ai_correction(scf_data: &mut SCF, mpi_operator: &Option<MPIOperator>) -> Option<Vec<f64>> {
    let xc_method = &scf_data.mol.ctrl.xc.to_lowercase();
    let post_ai_corr = &scf_data.mol.ctrl.post_ai_correction.to_lowercase();
    let mut scc = 0.0;
    if post_ai_corr.eq("scc15") && xc_method.eq("r-xdh7") {
        scc = scc15_for_rxdh7(scf_data, mpi_operator);
        let total_energy = scf_data.energies.get("xdh_energy").unwrap()[0];
        if scf_data.mol.ctrl.print_level>0 {
            println!("E(R-xDH7-SCC15): {:16.8} Ha", total_energy + scc);
        }
        //scf_data.energies.insert("scc23".to_string(), vec![scc]);
        return Some(vec![scc])
    };
    None
}

/// NOTE: only support symmetric RI-V tensors
pub fn post_scf_correlation(scf_data: &mut SCF) {

    // Phase-0 guard (P0-4): every post-correlation branch below (PT2/SBGE2/RPA/SCSRPA)
    // evaluates its correlation with serial kernels that assume a full (non-distributed)
    // RI3MO tensor. Under MPI the RI3MO tensor is aux-distributed and these kernels would
    // read out of bounds. Error out at the entrance instead of deep inside the kernels.
    if scf_data.mol.mpi_data.is_some() {
        panic!(
            "post_correlation is not yet supported under MPI: the PT2/SBGE2/RPA/SCSRPA \
             post-correlation kernels assume a full RI3MO tensor, which is aux-distributed \
             under MPI. Please run post-correlation tasks without MPI for now."
        );
    }

    let mut timerecords = TimeRecords::new();
    let spin_channel = scf_data.mol.spin_channel;
    let dfa_family_pos = if let Some(tmp_dfa) = &scf_data.mol.xc_data.dfa_family_pos {
        tmp_dfa.clone()
    } else {crate::dft::DFAFamily::Unknown};

    if let None = scf_data.ri3mo {
        let (occ_range, vir_range) = crate::scf_io::determine_ri3mo_size_for_pt2_and_rpa(&scf_data);
        if scf_data.mol.ctrl.print_level>1 {
            //println!("generate RI3MO only for occ_range:{:?}, vir_range:{:?}", &occ_range, &vir_range);
            let spin_orb_indices = split_indices_by_spin_occ(&scf_data.occupation, 0.5);
            let (alpha_occ, alpha_vir) = &spin_orb_indices[0];
            println!("Occupied orbitals (alpha): {}", format_indices(alpha_occ));
            println!("Virtual orbitals (alpha): {}", format_indices(alpha_vir));
            if matches!(scf_data.scftype, SCFType::UHF | SCFType::ROHF) {
                let (beta_occ,  beta_vir)  = &spin_orb_indices[1];
                println!("Occupied orbitals (beta): {}", format_indices(beta_occ));
                println!("Virtual orbitals (beta): {}", format_indices(beta_vir));
            }
        };
        scf_data.generate_ri3mo_rayon(vir_range, occ_range);
    }

    let mut post_corr: Vec<(crate::dft::DFAFamily, [f64;3])> = vec![];
    scf_data.mol.ctrl.post_correlation.iter().filter(|corr| *corr!=&dfa_family_pos).for_each(|corr| {
        match corr {
            crate::dft::DFAFamily::PT2 => {
                timerecords.new_item("PT2", "the PT2 calculation");
                timerecords.count_start("PT2");
                println!("Evaluating the PT2 correlation");
                let mut energy_post = if spin_channel == 1 {
                    close_shell_pt2_rayon(&scf_data).unwrap()
                } else {
                    open_shell_pt2_rayon(&scf_data).unwrap()
                };
                let os_factor = scf_data.mol.ctrl.ri_pt2.os_factor.unwrap_or(1.0);
                let ss_factor = scf_data.mol.ctrl.ri_pt2.ss_factor.unwrap_or(1.0);
                if scf_data.mol.ctrl.print_level > 1 && (os_factor != 1.0 || ss_factor != 1.0) {
                    println!("PT2 scaling factors: OS: {:16.8}, SS: {:16.8}", os_factor, ss_factor);
                }
                let sos_energy = os_factor * energy_post[1];
                let sss_energy = ss_factor * energy_post[2];
                energy_post[0] = sos_energy + sss_energy;
                post_corr.push((crate::dft::DFAFamily::PT2, energy_post));
                timerecords.count("PT2");
            },
            crate::dft::DFAFamily::SBGE2 => {
                timerecords.new_item("sBGE2", "the sBGE2 calculation");
                timerecords.count_start("sBGE2");
                println!("Evaluating the sBGE2 correlation");
                let energy_post = if spin_channel == 1 {
                    close_shell_sbge2_rayon(&scf_data).unwrap()
                } else {
                    //[0.0,0.0,0.0]
                    open_shell_sbge2_rayon(&scf_data).unwrap()
                };
                post_corr.push((crate::dft::DFAFamily::SBGE2, energy_post));
                timerecords.count("sBGE2");

            },
            crate::dft::DFAFamily::RPA => {
                timerecords.new_item("dRPA", "the dRPA calculation");
                timerecords.count_start("dRPA");
                println!("Evaluating the dRPA correlation");
                let energy_post = evaluate_rpa_correlation_rayon(&scf_data).unwrap();
                post_corr.push((crate::dft::DFAFamily::RPA, [energy_post, 0.0,0.0]));
                timerecords.count("dRPA");
            },
            crate::dft::DFAFamily::SCSRPA => {
                timerecords.new_item("SCSRPA", "the scsRPA calculation");
                timerecords.count_start("SCSRPA");
                println!("Evaluating the scsRPA correlation");
                let energy_post = evaluate_osrpa_correlation_rayon(&scf_data).unwrap();
                post_corr.push((crate::dft::DFAFamily::SCSRPA, energy_post));
                timerecords.count("SCSRPA");
            }
            _ => {println!("Unknown post-scf correlation methods")}
        }
    });

    if scf_data.mol.ctrl.print_level>0 {
        println!("----------------------------------------------------------------------");
        println!("{:16}: {:>16}, {:>16}, {:>16}","Methods","Total Corr", "OS Corr", "SS Corr");
        println!("----------------------------------------------------------------------");

        post_corr.iter().for_each(|(name,energy)| {
            println!("{:16}: {:16.8}, {:16.8}, {:16.8}", name.to_name(), energy[0], energy[1], energy[2]);
        });
        println!("----------------------------------------------------------------------");
        if scf_data.mol.ctrl.print_level>1 {timerecords.report_all()};
    }
}

/// Has a checkpoint been restored whose GW/evGW work is already complete?
fn checkpoint_has_finished_gw(scf_data: &SCF) -> bool {
    matches!(
        scf_data.gw_checkpoint_state.as_ref().map(|state| state.stage),
        Some(crate::fileop::gw_checkpoint::GwCheckpointStage::GwFinished)
    )
}

/// Log the fact that a restored checkpoint lets us skip the GW/evGW work.
fn announce_gw_skipped(scf_data: &SCF, output_type: &str) {
    println!("--------------------------------------------------------------------------------");
    println!(
        "[gw_checkpoint] 'resume_from_checkpoint = true': the GW/evGW calculation is SKIPPED -- \
         {} quasiparticle energies were restored from the checkpoint.",
        scf_data.gwqp.0.len()
    );
    if output_type.eq("gw") {
        println!("[gw_checkpoint] the restored spectrum is reported below; nothing is recomputed.");
    } else {
        println!(
            "[gw_checkpoint] '{}' continues directly with the BSE/response step.",
            output_type
        );
    }
    println!("--------------------------------------------------------------------------------");
}

/// Compute the grid-derived `v_xc` of the GW/BSE gradient engine and then release the SCF
/// grid's tabulated AO tables.
///
/// The tabulated tables (`grids.ao`/`aop`/`ao_compressed`/`aop_compressed`/`non0tab`) exist
/// for the SCF's own XC evaluation.  In the whole GW/BSE analytic-gradient path they have a
/// **single** reader — `GwGradEngine::precompute_vxc` (`ri_gw`/`ri_bse` never touch
/// `scf.grids`, and the CP-KS response, the f_xc kernel and `d v_xc/dR` all evaluate their
/// own AO from `grids.coordinates`/`weights`).  Once `precompute_vxc` has returned they are
/// dead weight for the rest of the process, and at def2-TZVPP/polyene scale they are
/// ~1-3.5 GB; releasing them here is therefore free, and the release lands *before* the
/// gradient's peak phase (which is well after the engine build).
///
/// This is the *only* place the gradient driver gives them up; the escape hatch
/// `REST_KEEP_GRID_AO=1` restores the historical behaviour for an A/B comparison.
fn precompute_vxc_and_release_grid_ao(scf_data: &mut SCF) -> ri_gw::gw_grad::GwVxc {
    let vxc = ri_gw::gw_grad::GwGradEngine::precompute_vxc(scf_data);
    if std::env::var("REST_KEEP_GRID_AO").is_ok() {
        return vxc;
    }
    // The only post-SCF consumer outside this gradient is the TDDFT-eigenvalue route
    // (`num_int::prepare_fxc_data`), which runs *after* the QP section in `main_driver` and
    // reads the compressed tables directly instead of re-evaluating.  A card that asks for
    // both keeps the tables (the historical behaviour).
    if scf_data.mol.ctrl.tddft.is_some() {
        return vxc;
    }
    if let Some(grids) = scf_data.grids.as_mut() {
        let freed = grids.release_tabulated_ao();
        if freed > 0 {
            crate::hessian::memory_monitor::trim_to_os(0);
            if scf_data.mol.ctrl.print_level >= 1 {
                println!(
                    " [qp] released {:.0} MB of tabulated DFT-grid AO (the GW/BSE gradient evaluates its own AO)",
                    freed as f64 / 1.0e6
                );
            }
        }
    }
    vxc
}

pub fn quasiparticle_methods(scf_data:&mut SCF,mpi_operator:&Option<MPIOperator>) -> crate::ri_bse::BseOutput {
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let output_type=qp_ctrl.gw_or_bse.clone();
    // `"cvs bse"` is the documented spelling; accept the underscore/dash
    // variants as a convenience.  Matching is case-insensitive.
    let normalized_output_type = output_type.trim().to_lowercase();
    let is_cvs_bse = matches!(
        normalized_output_type.as_str(),
        "cvs bse" | "cvs_bse" | "cvs-bse"
    );
    let mut bse_output = crate::ri_bse::BseOutput {
        first_excitation: None,
        excitation_energies: Vec::new(),
    };
    // A restored checkpoint may already contain the GW/evGW work.  In that case
    // we must not redo it -- and must not overwrite the checkpoint either.
    let gw_already_done = checkpoint_has_finished_gw(scf_data);
    if let Some(state) = scf_data.gw_checkpoint_state.as_ref() {
        if state.stage == crate::fileop::gw_checkpoint::GwCheckpointStage::EvgwRound
            && !state.scgw_mode.eq_ignore_ascii_case(&qp_ctrl.scgw)
        {
            panic!(
                "[gw_checkpoint] the checkpoint '{}' was written in the middle of a {} run \
                 (round {} of {}), but the current input asks for scgw = \"{}\".\n\
                 evGW and evGW0 store different W vectors, so an interrupted archive must be \
                 resumed with the mode that wrote it.  Fix the input so that scgw = \"{}\" \
                 (and evgw_rounds > {}), or set 'resume_from_checkpoint = false' to start over.",
                crate::fileop::gw_checkpoint::checkpoint_path(scf_data),
                state.scgw_mode,
                state.rounds_done,
                state.total_rounds,
                qp_ctrl.scgw,
                state.scgw_mode,
                state.rounds_done,
            );
        }
    }
    if gw_already_done {
        announce_gw_skipped(scf_data, &output_type);
    }
    if output_type.eq("gw"){
        if gw_already_done {
            // Nothing to recompute: just report the restored quasiparticle
            // spectrum, exactly as the freshly computed one would have been.
            let (_,_,occ_size,_,_,_)=ri_gw::get_occupation_parameters(scf_data,'Y');
            ri_gw::display::full_quasiparticles(&scf_data.gwqp.0,occ_size);
        } else {
            let vxc_nn=ri_gw::vxc_ao2mo(scf_data);
            let vxc_nn_spin: Option<[Vec<f64>;2]> = if scf_data.mol.spin_channel==2 {
                Some(ri_gw::vxc_ao2mo_spin(scf_data))
            } else { None };
            let xc_data=scf_data.mol.xc_data.clone();
            println!("Current XC data:");
            println!("dfa_compnt_scf={:?}",xc_data.dfa_compnt_scf);
            println!("dfa_paramr_scf={:?}",xc_data.dfa_paramr_scf);
            println!("dfa_hybrid_scf={}",xc_data.dfa_hybrid_scf);
            if qp_ctrl.homo_lumo_gw_qp==true{
                ri_gw::get_homo_lumo_qp_only(scf_data,20,&vxc_nn,mpi_operator);
            }else if qp_ctrl.self_energy_spectrum_test==true{
                ri_gw::spectrum_test(scf_data,20);
            }else if qp_ctrl.obtain_vx_vc_terms==true{
                ri_gw::obtain_vx_vc_terms(scf_data);
            }else if scf_data.mol.spin_channel==2{
                ri_gw::gw_main_spin(scf_data,&vxc_nn_spin.unwrap(),mpi_operator);
            }else{
                ri_gw::gw_main(scf_data,&vxc_nn,mpi_operator);
                if scf_data.mol.ctrl.print_level>1{
                    ri_bse::matvec::test_v_w_contribution(scf_data);
                }
            }
            // The GW pass has finished: archive the state so that a later run
            // can resume without redoing it.
            crate::fileop::gw_checkpoint::save_gw_finished_checkpoint(scf_data);
        }
    }else if output_type.eq("bse"){
        // BSE-specific integrals must be prepared AFTER GW calculation,
        // otherwise get_submatrix would pick them up during GW and return
        // wrong-dimensional RI matrices (BSE aux basis instead of full).
        if gw_already_done {
            // restored from the checkpoint: GW is not recomputed
        }else if qp_ctrl.gw_scheme=="parse from file"{
            let parse_qp_path=qp_ctrl.parse_qp_path.clone();
            let qp=ri_gw::read_floats(&parse_qp_path).expect("Failure when reading from GW QP energies file!");
            scf_data.gwqp.0=qp.clone();
            if scf_data.mol.spin_channel==2{
                scf_data.gwqp_spin.0[0]=qp.clone();
                scf_data.gwqp_spin.0[1]=qp;
                scf_data.gwqp_spin.1=scf_data.gwqp_spin.0.clone();
            }
        }else if scf_data.mol.spin_channel==2{
            let vxc_nn_spin=ri_gw::vxc_ao2mo_spin(scf_data);
            ri_gw::gw_main_spin(scf_data,&vxc_nn_spin,mpi_operator);
        }else{
            let vxc_nn=ri_gw::vxc_ao2mo(scf_data);
            ri_gw::gw_main(scf_data,&vxc_nn,mpi_operator);
        }
        // Hook: GW is complete, BSE/response has not started yet.
        if !gw_already_done {
            crate::fileop::gw_checkpoint::save_gw_finished_checkpoint(scf_data);
        }
        scf_data.prepare_bse_integrals(mpi_operator);
        bse_output = ri_bse::bse_main(scf_data);
    }else if is_cvs_bse {
        // Core-valence separated BSE.  The GW stage is the ordinary one, except
        // that the extrapolated-window thresholds are widened so the core
        // quasiparticle energies are computed explicitly rather than rigidly
        // shifted from the valence window.
        if gw_already_done {
            // restored from the checkpoint: GW is not recomputed
        } else {
            ri_bse::cvs::prepare_cvs_gw(scf_data);
            // Re-read the (possibly widened) control block.
            let qp_ctrl_cvs = scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
            if qp_ctrl_cvs.gw_scheme == "parse from file" {
                let parse_qp_path = qp_ctrl_cvs.parse_qp_path.clone();
                scf_data.gwqp.0 = ri_gw::read_floats(&parse_qp_path)
                    .expect("Failure when reading from GW QP energies file!");
            } else {
                let vxc_nn = ri_gw::vxc_ao2mo(scf_data);
                ri_gw::gw_main(scf_data, &vxc_nn, mpi_operator);
            }
        }
        if !gw_already_done {
            crate::fileop::gw_checkpoint::save_gw_finished_checkpoint(scf_data);
        }
        scf_data.prepare_bse_integrals(mpi_operator);
        bse_output = ri_bse::cvs::cvs_bse_main(scf_data);
    }else if output_type.eq("response_bse"){
        if gw_already_done {
            // restored from the checkpoint: GW is not recomputed
        }else if qp_ctrl.gw_scheme=="parse from file"{
            let parse_qp_path=qp_ctrl.parse_qp_path.clone();
            scf_data.gwqp.0=ri_gw::read_floats(&parse_qp_path).expect("Failure when reading from GW QP energies file!");
        }else{
            let vxc_nn=ri_gw::vxc_ao2mo(scf_data);
            ri_gw::gw_main(scf_data,&vxc_nn,mpi_operator);
        }
        if !gw_already_done {
            crate::fileop::gw_checkpoint::save_gw_finished_checkpoint(scf_data);
        }
        let p_induced=ri_bse::response::response_bse(scf_data);
        // println!("Induced Density Matrix:");
        // println!("P Real (Plus Half):\n{:#?}",p_induced.0);
        // println!("P Real (Minus Half):\n{:#?}",p_induced.1);
        // println!("P Imaginary (Plus Half):\n{:#?}",p_induced.2);
        // println!("P Imaginary (Minus Half):\n{:#?}",p_induced.3);
    }else if output_type.eq("nonlinear_bse"){
        // NLFEAST requires GW quasiparticle energies for diagonal elements
        if gw_already_done {
            // restored from the checkpoint: GW is not recomputed
        }else if qp_ctrl.gw_scheme=="parse from file"{
            let parse_qp_path=qp_ctrl.parse_qp_path.clone();
            scf_data.gwqp.0=ri_gw::read_floats(&parse_qp_path).expect("Failure when reading from GW QP energies file!");
        }else{
            let vxc_nn=ri_gw::vxc_ao2mo(scf_data);
            ri_gw::gw_main(scf_data,&vxc_nn,mpi_operator);
        }
        if !gw_already_done {
            crate::fileop::gw_checkpoint::save_gw_finished_checkpoint(scf_data);
        }
        ri_bse::nonlinbse::nlfeast_bse_main(scf_data, &qp_ctrl);
    }else if output_type.eq("dynamic_bse"){
        // Dynamical BSE: number-conserving RPA-pair kernel (Sangalli et al.,
        // J. Chem. Phys. 134, 034115 (2011)) solved as a nonlinear eigenvalue
        // problem with the NLFEAST kernel.  Requires GW quasiparticle energies.
        if gw_already_done {
            // restored from the checkpoint: GW is not recomputed
        }else if qp_ctrl.gw_scheme=="parse from file"{
            let parse_qp_path=qp_ctrl.parse_qp_path.clone();
            scf_data.gwqp.0=ri_gw::read_floats(&parse_qp_path).expect("Failure when reading from GW QP energies file!");
        }else{
            let vxc_nn=ri_gw::vxc_ao2mo(scf_data);
            ri_gw::gw_main(scf_data,&vxc_nn,mpi_operator);
        }
        if !gw_already_done {
            crate::fileop::gw_checkpoint::save_gw_finished_checkpoint(scf_data);
        }
        ri_bse::dynamicbse::dynamic_bse_main(scf_data, &qp_ctrl);
    }else if output_type.eq("gw_gradient") || output_type.eq("bse_gradient"){
        // Analytic nuclear gradients of the GW quasiparticle energy or of the
        // lowest BSE excitation energies.  Both are evaluated through the
        // library engines that the numerical tests use; this branch is the
        // input-card entry point for them.
        if scf_data.mol.spin_channel==2 {
            panic!("GW-BSE analytic gradients currently support only spin-restricted (closed-shell) references.");
        }
        if qp_ctrl.gw_scheme=="parse from file"{
            let parse_qp_path=qp_ctrl.parse_qp_path.clone();
            scf_data.gwqp.0=ri_gw::read_floats(&parse_qp_path)
                .expect("Failure when reading from GW QP energies file!");
        }
        if scf_data.mol.ctrl.print_level>0 {
            println!("GW-BSE analytic nuclear gradient invoked (gw_or_bse = \"{output_type}\").");
        }
        let mut cfg = ri_gw::gw_grad::GwGradConfig::from_scf(scf_data);
        cfg.z_min = qp_ctrl.gradient_z_min;
        cfg.force_static_fallback = qp_ctrl.gradient_force_fallback
            .split(',')
            .filter_map(|t| t.trim().parse::<usize>().ok())
            .collect();
        if cfg.z_min > 0.0 && (cfg.z_min - 0.3).abs() > 0.0 {
            println!("GW-BSE gradient: linearized (one-step Newton) branch used when Z < {:.3}", cfg.z_min);
        }
        if !cfg.force_static_fallback.is_empty() {
            println!(
                "GW-BSE gradient: {} orbital(s) pinned to the static on-shell fallback: {:?}",
                cfg.force_static_fallback.len(), cfg.force_static_fallback
            );
        }
        cfg.smooth_pole = qp_ctrl.gradient_smooth_pole;
        if cfg.smooth_pole {
            println!("GW-BSE gradient: C^1 residue pole-factor switch (no 0/0.5/1 jump)");
        }
        cfg.exact_residue_z = qp_ctrl.gradient_exact_residue;
        if cfg.exact_residue_z {
            println!("GW-BSE gradient: continuous (exact-frequency) residue evaluation, no real-axis grid");
        }
        cfg.graphical_rescue = qp_ctrl.gradient_graphical_rescue;
        if cfg.graphical_rescue {
            println!("GW-BSE gradient: graphical (linear-interpolation) rescue of a stalled Newton step enabled");
        }
        cfg.force_linearized = qp_ctrl.gradient_force_linearized
            .split(',')
            .filter_map(|t| t.trim().parse::<usize>().ok())
            .collect();
        if !cfg.force_linearized.is_empty() {
            println!(
                "GW-BSE gradient: {} orbital(s) pinned to the linearized (one-step Newton) branch: {:?}",
                cfg.force_linearized.len(), cfg.force_linearized
            );
        }
        if output_type.eq("gw_gradient"){
            let target = if qp_ctrl.gradient_orbital < 0 {
                scf_data.homo[0]
            } else {
                qp_ctrl.gradient_orbital as usize
            };
            assert!(target < scf_data.mol.num_state, "gradient_orbital {target} out of range");
            let use_cd = qp_ctrl.gradient_engine.eq_ignore_ascii_case("cd");
            let vxc = precompute_vxc_and_release_grid_ao(scf_data);
            let (grad, omega, z) = if use_cd {
                let engine = ri_gw::gw_grad::GwCdGradEngine::new_with_vxc(scf_data, cfg, vxc);
                engine.analytic_gradient(target)
            } else {
                let engine = ri_gw::gw_grad::GwGradEngine::new_with_vxc(scf_data, cfg, vxc);
                engine.analytic_gradient(target)
            };
            println!(
                "GW quasiparticle energy of orbital {target}: {omega:.12} Ha  (Z = {z:.6}, engine = \"{}\")",
                if use_cd {"cd"} else {"lr"}
            );
            report_analytic_gradient(scf_data, "GW", &grad);
            report_state_and_total_gradient(scf_data, "GW", &grad.clone());
        }else{
            let multi = if qp_ctrl.bse_spin.eq_ignore_ascii_case("triplet") {'t'} else {'s'};
            let screen = if qp_ctrl.gradient_screening.eq_ignore_ascii_case("qp") {
                ri_bse::bse_grad::ScreeningEnergy::Qp
            } else {
                ri_bse::bse_grad::ScreeningEnergy::Reference
            };
            // How many BSE roots to solve.  The *default* is the lowest
            // excited state alone (`gradient_state = 0`), i.e. exactly the roots
            // needed up to the requested state: the analytic gradient cost is
            // dominated by the transition-space construction, which grows with
            // the number of solved roots, and the extra roots are never
            // differentiated unless they are asked for.  Set
            // `gradient_nroots` in the input card to solve (and report) more.
            // Stage 1 of the two-stage workflow (`bse_roots_only`): solve
            // `bse_roots_n` roots, dump their fingerprints and stop before the
            // expensive per-root gradient assembly.
            let nroots = if qp_ctrl.bse_roots_only {
                qp_ctrl.bse_roots_n
            } else if qp_ctrl.gradient_nroots == 0 {
                qp_ctrl.gradient_state + 1
            } else {
                qp_ctrl.gradient_nroots
            };
            assert!(
                nroots > qp_ctrl.gradient_state,
                "gradient_nroots ({nroots}) must exceed gradient_state ({})",
                qp_ctrl.gradient_state
            );
            let vxc = precompute_vxc_and_release_grid_ao(scf_data);
            let (grads, omegas) = {
                let gw = ri_gw::gw_grad::GwGradEngine::new_with_vxc(scf_data, cfg, vxc);
                // `gradient_bse_window = 0` keeps the full virtual space, the
                // historical behaviour; a positive value restricts the
                // transition space to `e_F + window` (the same convention as the
                // timing driver's `PAPER_WINDOW`).
                let cutoff = if qp_ctrl.gradient_bse_nvir > 0 {
                    // Fixed *count*: the cutoff sits halfway between the n-th and
                    // the (n+1)-th virtual at this geometry, so the transition
                    // space keeps the same dimension at every point of a
                    // relaxation path.  An energy window lets orbitals drift in
                    // and out (observed `dim 816 -> 864 -> 1008` on
                    // azobenzene/def2-TZVPP), which makes `E_GS + Omega`
                    // discontinuous at those crossings.
                    let e = &scf_data.eigenvalues[0];
                    let last = scf_data.homo[0] + qp_ctrl.gradient_bse_nvir;
                    if last + 1 < e.len() {
                        0.5 * (e[last] + e[last + 1])
                    } else {
                        f64::INFINITY
                    }
                } else if qp_ctrl.gradient_bse_window > 0.0 {
                    let e = &scf_data.eigenvalues[0];
                    let nocc = scf_data.homo[0] + 1;
                    0.5 * (e[nocc - 1] + e[nocc]) + qp_ctrl.gradient_bse_window
                } else {
                    f64::INFINITY
                };
                let mut bse = ri_bse::bse_grad::BseGradEngine::new_with_cutoff(
                    scf_data, &gw, multi, screen, cutoff);
                println!(
                    "BSE transition space: {} occupied x {} of {} virtuals, spin channel '{multi}'",
                    bse.screen.nocc, bse.screen.nvir, bse.screen.nvir_full
                );
                // Archive the quasiparticle spectrum this gradient is built on,
                // so that a later card (e.g. stage 2 of the two-stage
                // excited-state workflow) can restore the SCF *and* the GW pass
                // from the chkfile instead of redoing them.  No-op unless
                // `save_gw_checkpoint = true` and `chkfile` is set; the archive
                // is validated against molecule/basis/geometry on reload, so a
                // mismatched geometry is a hard error rather than silent wrong
                // physics.
                crate::fileop::gw_checkpoint::save_gw_finished_checkpoint(scf_data);
                // Pick the solver by the size of the transition space, exactly
                // as the timing driver does: the dense route materialises the
                // full `dim x dim` matrix (plus its eigenvectors and LAPACK
                // workspace), so `dim` in the thousands costs gigabytes before
                // the gradient even starts.  The card path used to force
                // `Dense`, which is what made an unrestricted transition space
                // (see `gradient_bse_window`) run out of memory.
                let dim = bse.screen.nocc * bse.screen.nvir;
                let solver = if dim <= 128 {
                    ri_bse::bse_grad::BseSolver::Dense
                } else {
                    ri_bse::bse_grad::BseSolver::Davidson
                };
                let mut dcfg = crate::solvers::davidson::DavidsonConfig::default();
                dcfg.max_subspace = 200;
                dcfg.max_iter = 400;
                bse.solve(nroots, solver, &dcfg);
                for (k, om) in bse.excitation_energies().iter().enumerate() {
                    println!("  BSE root {k}: {:.12} Ha = {:.6} eV", om, om * 27.211386245988);
                }
                // Composition of every solved root: the dominant `i -> a`
                // transitions with their `x^2 - y^2` weights and the oscillator
                // strength (dark vs bright).  This is what an external
                // geometry-optimisation driver needs in order to follow a
                // *state* (by composition) instead of a root *index* when two
                // states approach each other, and it makes a run auditable after
                // the fact -- the amplitudes are otherwise never printed.
                report_root_composition(scf_data, &bse);
                if !qp_ctrl.gradient_amplitude_path.is_empty() {
                    save_bse_amplitudes(&bse, &qp_ctrl.gradient_amplitude_path);
                }
                if qp_ctrl.bse_roots_only {
                    // Stage 1: write the fingerprints and return -- no gradient.
                    save_bse_roots(scf_data, &bse, &qp_ctrl.bse_roots_path);
                    println!(
                        "bse_roots_only: {} root(s) dumped to {} -- no gradient assembled. \
                         Use a second card with gradient_state = <index> to differentiate one.",
                        bse.roots.len(),
                        qp_ctrl.bse_roots_path
                    );
                    return bse_output;
                }
                bse.analytic_gradients()
            };
            let state = qp_ctrl.gradient_state;
            println!(
                "BSE analytic gradient requested for root {state} (omega = {:.12} Ha = {:.6} eV)",
                omegas[state], omegas[state] * 27.211386245988
            );
            report_analytic_gradient(scf_data, &format!("BSE_ROOT{state}"), &grads[state]);
            report_state_and_total_gradient(
                scf_data,
                &format!("BSE_ROOT{state}"),
                &grads[state].clone(),
            );
        }
    }else{
        print!("Warning: You entered an invalid quasiparticle method. No quasiparticle methods Were triggered.")
    }
    bse_output
}

/// Print the transition composition of every solved BSE root.
///
/// One line per root: the excitation energy, the oscillator strength and the
/// five largest `(i -> a)` contributions with their `x^2 - y^2` weights,
/// labelled relative to the HOMO/LUMO (`H`, `H-1`, ..., `L`, `L+1`, ...).
/// Amplitude flat index is `i + a*nocc` (occupied index fastest), so the
/// orbital pair is `(i, nocc + a)`.
fn report_root_composition(scf_data:&SCF, bse:&ri_bse::bse_grad::BseGradEngine) {
    let (nocc, nvir) = (bse.screen.nocc, bse.screen.nvir);
    if nocc == 0 || nvir == 0 { return; }
    let dim = nocc * nvir;
    // The active virtuals are the lowest `nvir`, so the active block of the
    // `[3, nocc*nvir_full]` dipole matrix is its leading `dim` columns.
    let mu_full = ri_bse::dipoles::compute_dipole_matrix(scf_data);
    let mut mu_act = tensors::MatrixFull::new([3, dim], 0.0);
    for d in 0..3 {
        for k in 0..dim {
            if k < mu_full.size[1] {
                mu_act[[d, k]] = mu_full[[d, k]];
            }
        }
    }
    println!("BSE root composition (weights x^2 - y^2, |mu|^2 in a.u.):");
    for (k, r) in bse.roots.iter().enumerate() {
        if r.x.len() < dim || r.y.len() < dim { continue; }
        // A failed solve (Cholesky of `A - B` can fail for a hybrid reference at
        // a bad geometry) leaves NaN amplitudes; `partial_cmp` would then panic
        // in the sort below and abort the whole REST run.  Report it instead.
        if !r.x.iter().chain(r.y.iter()).all(|v| v.is_finite()) {
            println!(
                "  root {k}: NON-FINITE amplitudes (omega = {}) -- BSE solve unusable, no composition",
                r.omega
            );
            continue;
        }
        let mut w: Vec<(f64, usize, usize)> = (0..dim)
            .map(|idx| (r.x[idx].powi(2) - r.y[idx].powi(2), idx % nocc, idx / nocc))
            .collect();
        w.sort_by(|p, q| q.0.abs().total_cmp(&p.0.abs()));
        let mut xy = r.x.clone();
        xy.extend_from_slice(&r.y);
        let f = ri_bse::dipoles::transition_dipole_square(&mu_act, &xy, false);
        let top: Vec<String> = w.iter().take(5).map(|(wi, i, a)| {
            let li = if *i == nocc - 1 { "H".to_string() } else { format!("H-{}", nocc - 1 - i) };
            let la = if *a == 0 { "L".to_string() } else { format!("L+{a}") };
            format!("{li}->{la}:{wi:+.4}")
        }).collect();
        println!(
            "  root {k}: Omega = {:.9} Ha ({:.4} eV)  |mu|^2 = {:.5}  {}",
            r.omega, r.omega * 27.211386245988, f, top.join("  ")
        );
    }
}

/// Write the BSE amplitudes of every solved root to `path`, one block per root:
///
/// ```text
/// # root <k> omega <Ha> nocc <n> nvir <m>
/// <2*nocc*nvir numbers: X then Y, flat index i + a*nocc>
/// ```
///
/// An external optimisation driver uses these to follow a state by amplitude
/// overlap from one geometry to the next; they also make the run auditable
/// (nothing else records which state was actually differentiated).
fn save_bse_amplitudes(bse:&ri_bse::bse_grad::BseGradEngine, path:&str) {
    use std::io::Write;
    let (nocc, nvir) = (bse.screen.nocc, bse.screen.nvir);
    let mut out = String::new();
    for (k, r) in bse.roots.iter().enumerate() {
        out.push_str(&format!(
            "# root {k} omega {:.12} nocc {nocc} nvir {nvir}\n",
            r.omega
        ));
        for v in r.x.iter().chain(r.y.iter()) {
            out.push_str(&format!("{v:.16e} "));
        }
        out.push('\n');
    }
    match std::fs::File::create(path) {
        Ok(mut f) => {
            if f.write_all(out.as_bytes()).is_ok() {
                println!("BSE amplitudes of {} root(s) written to {path}", bse.roots.len());
            }
        }
        Err(e) => println!("WARNING: cannot write BSE amplitudes to {path}: {e}"),
    }
}

/// Transition dipole components `mu_d = 2 sum_ia (X_ia + Y_ia) mu^d_ia`.
///
/// The same contraction as [`ri_bse::dipoles::transition_dipole_square`], which
/// returns only its square; the *direction* is what identifies a state by
/// symmetry.
fn dipole_components(dipole_matrix:&tensors::MatrixFull<f64>, xy:&[f64]) -> [f64;3] {
    let dim = dipole_matrix.size[1];
    let mut mu = [0.0f64;3];
    for d in 0..3 {
        let mut s = 0.0;
        for ia in 0..dim {
            s += (xy[ia] + xy[dim + ia]) * dipole_matrix[[d, ia]];
        }
        mu[d] = 2.0 * s;
    }
    mu
}

/// Stage-1 root dump (`bse_roots_only`), written as JSON.
///
/// Every root is described by quantities that are **invariant under a change of
/// root order or of the orbital labelling inside the active space**: the
/// excitation energy, the oscillator strength and the transition-dipole
/// direction (symmetry-protected: a dark `n -> pi*` state has `|mu| ~ 0` while a
/// bright `pi -> pi*` state has a large in-plane component), the count of roots
/// below it, and the five largest `(i -> a)` weights.  An external driver picks
/// the index from these and hands it to a second card via `gradient_state`.
fn save_bse_roots(scf_data:&SCF, bse:&ri_bse::bse_grad::BseGradEngine, path:&str) {
    use std::io::Write;
    let (nocc, nvir) = (bse.screen.nocc, bse.screen.nvir);
    let dim = nocc * nvir;
    let mu_full = ri_bse::dipoles::compute_dipole_matrix(scf_data);
    let mut mu_act = tensors::MatrixFull::new([3, dim], 0.0);
    for d in 0..3 {
        for k in 0..dim {
            if k < mu_full.size[1] {
                mu_act[[d, k]] = mu_full[[d, k]];
            }
        }
    }
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"nocc\": {nocc},\n  \"nvir\": {nvir},\n  \"dim\": {dim},\n"));
    out.push_str(&format!("  \"nroots\": {},\n  \"roots\": [\n", bse.roots.len()));
    for (k, r) in bse.roots.iter().enumerate() {
        let finite = r.x.len() >= dim && r.y.len() >= dim
            && r.x.iter().chain(r.y.iter()).all(|v| v.is_finite());
        if !finite {
            out.push_str(&format!("    {{\"index\": {k}, \"omega_ha\": {:.12}, \"failed\": true}}", r.omega));
        } else {
            let mut xy = r.x.clone();
            xy.extend_from_slice(&r.y);
            let mu = dipole_components(&mu_act, &xy);
            let mu2 = mu.iter().map(|x| x * x).sum::<f64>();
            let mut w: Vec<(f64, usize, usize)> = (0..dim)
                .map(|idx| (r.x[idx].powi(2) - r.y[idx].powi(2), idx % nocc, idx / nocc))
                .collect();
            w.sort_by(|p, q| q.0.abs().total_cmp(&p.0.abs()));
            let top: Vec<String> = w.iter().take(5).map(|(wi, i, a)| {
                format!(
                    "{{\"i_from_homo\": {}, \"a_from_lumo\": {}, \"weight\": {:.6}}}",
                    nocc - 1 - i, a, wi
                )
            }).collect();
            out.push_str(&format!(
                "    {{\"index\": {k}, \"omega_ha\": {:.12}, \"omega_ev\": {:.6}, \
                 \"mu2\": {:.6}, \"mu\": [{:.6}, {:.6}, {:.6}], \"dim\": {dim}, \"top\": [{}]}}",
                r.omega, r.omega * 27.211386245988, mu2, mu[0], mu[1], mu[2], top.join(", ")
            ));
        }
        out.push_str(if k + 1 == bse.roots.len() { "\n" } else { ",\n" });
    }
    out.push_str("  ]\n}\n");
    match std::fs::File::create(path) {
        Ok(mut f) => match f.write_all(out.as_bytes()) {
            Ok(_) => println!("BSE root fingerprints written to {path}"),
            Err(e) => println!("WARNING: cannot write {path}: {e}"),
        },
        Err(e) => println!("WARNING: cannot create {path}: {e}"),
    }
}

/// Print an analytic Cartesian gradient (flat index `3*atom + comp`, in
/// Hartree/Bohr) with the standard force formatter and archive it in
/// `scf_data.energies` under `<label>_gradient`.
fn report_analytic_gradient(scf_data:&mut SCF, label:&str, grad:&[f64]) {
    let natm = scf_data.mol.geom.elem.len();
    assert_eq!(grad.len(), natm*3, "{label} gradient: unexpected length {} (expected {})", grad.len(), natm*3);
    let mut m = tensors::MatrixFull::new([3, natm], 0.0);
    for a in 0..natm {
        for c in 0..3 {
            m.data[c + a*3] = grad[a*3 + c];
        }
    }
    println!("------ {label} analytic gradient [a.u.] ------");
    println!("{}", formated_force(&m, &scf_data.mol.geom.elem));
    println!("---------------------------------------------");
    scf_data.energies.insert(format!("{}_gradient", label.to_lowercase()), grad.to_vec());
}

/// Analytic nuclear gradient of the *reference* state, flat `3*atom + comp`.
///
/// This is the ground-state (mean-field) gradient that has to be added to a
/// GW quasiparticle or BSE excitation-energy gradient: those engines
/// differentiate only their own energy (`E_QP` / `Omega_S`), while the total
/// energy of the state is `E_GS(R) + E_QP(R)` or `E_GS(R) + Omega_S(R)`.
///
/// A Kohn-Sham reference needs the `calc_rks` path (it adds the xc
/// contribution to the skeleton derivative) and a Hartree-Fock one `calc`;
/// using the latter on a DFT reference leaves the xc term out entirely.
fn reference_gradient(scf_data:&SCF) -> Vec<f64> {
    let mut gs = crate::grad::rhf::RIRHFGradient::new(scf_data, &None);
    if scf_data.mol.xc_data.dfa_compnt_scf.is_empty() {
        gs.calc();
    } else {
        gs.calc_rks();
    }
    let m = gs.get_gradient();
    let natm = scf_data.mol.geom.elem.len();
    let mut g = vec![0.0f64; natm*3];
    for a in 0..natm {
        for c in 0..3 {
            g[a*3 + c] = m.data[c + a*3];
        }
    }
    g
}

/// Report the ground-state gradient and the *total* gradient of an electronic
/// state whose own gradient is `state_grad` (`E_GS + E_state`), so that the
/// caller can see both the engine's output and the quantity that defines the
/// potential-energy surface the state actually lives on.
fn report_state_and_total_gradient(
    scf_data:&mut SCF,
    state_label:&str,
    state_grad:&[f64],
) -> Vec<f64> {
    let gs = reference_gradient(scf_data);
    report_analytic_gradient(scf_data, "GS", &gs);
    let total: Vec<f64> = gs.iter().zip(state_grad).map(|(a,b)| a + b).collect();
    println!(
        "Total gradient of {state_label} = d(E_GS)/dR + d({state_label})/dR \
         (the state's own gradient alone is not the PES gradient)."
    );
    report_analytic_gradient(scf_data, &format!("{state_label}_TOTAL"), &total);
    total
}

fn fciqmc_dump(scf_data: &SCF) {
    if let Some(ri3fn) = &scf_data.ri3fn {
        // prepare RI-V three-center coefficients for HF orbitals
        for i_spin in 0..scf_data.mol.spin_channel {
            let ri3mo = ri3fn.ao2mo_v01(&scf_data.eigenvectors[i_spin]).unwrap();
            for i in 0.. scf_data.mol.num_state {
                let ri3mo_i = ri3mo.get_reducing_matrix(i).unwrap();
                for j in 0.. scf_data.mol.num_state {
                    let ri3mo_ij = ri3mo_i.get_slice_x(j);
                    for k in 0.. scf_data.mol.num_state {
                        let ri3mo_k = ri3mo.get_reducing_matrix(k).unwrap();
                        for l in 0.. scf_data.mol.num_state {
                            let ri3mo_kl = ri3mo_k.get_slice_x(l);
                            let ijkl = ri3mo_ij.iter().zip(ri3mo_kl.iter())
                                .fold(0.0, |acc, (val1, val2)| acc + val1*val2);
                            if ijkl.abs() > 1.0E-8 {
                                println! ("{:16.8} {:5} {:5} {:5} {:5}",ijkl, i,j,k,l);
                            }
                        }
                    }
                }
            }
        }
    }
}


//pub fn test_hdf5_string() {
//    let file = hdf5::File::create("test_string").unwrap();
//    let geom = file.group("geom").unwrap_or(file.create_group("geom").unwrap());
//    let dd = vec!["string1", "string2"];
//    let builder = geom.new_dataset_builder();
//    builder.with_data(&ndarray::arr1(&dd)).create("elem");
//    file.close();
//}

// evaluate the dipole moment based on the converged density matrix (dm): scf_data.density_matrix
// the dipole moment of the nuclear part (nucl_dip) is given by scf_data.mol.geom.evaluate_dipole_moment()
// the dipole moment of the atomic orbitals (ao_dip) is given by scf_data.mol.int_ij_matrixuppers()
// the dipole moment of the electronic part (el_dip) is given ('ij,ji', ao_dip[x], dm)
pub fn evaluate_dipole_moment(scf_data: &SCF, orig: Option<[f64;3]>) -> [f64;3] {

    let (mut tot_dip, mass_tot) = scf_data.mol.geom.evaluate_dipole_moment(None);

    let mut dm = scf_data.density_matrix[0].clone();
    if scf_data.mol.spin_channel == 2 {
        dm.self_add(&scf_data.density_matrix[1]);
    }

    let mut cint_data = scf_data.mol.initialize_cint(false);

    let p_orig = cint_data.get_common_origin();

    let r_orig: [f64;3] = if let Some(u_orig) = orig {
        u_orig.try_into().unwrap()
    } else {
        p_orig.clone()
    };
    cint_data.set_common_origin(r_orig);

    let (out, out_shape)= cint_data.integral_s1::<int1e_r>(None);
    //let mut out_shape_1 = [0;3];
    //out_shape_1.iter_mut().zip(out_shape.iter()).for_each(|(out_shape_1, &out_shape)| {*out_shape_1 = out_shape});

    let ao_dip = RIFull::from_vec(out_shape.try_into().unwrap(), out).unwrap();

    let mut el_dip = [0.0;3];
    for i in 0..3 {
        let ao_dip_tmp = ao_dip.get_reducing_matrix(i).unwrap();
        el_dip[i] = dm.iter_columns_full().zip(ao_dip_tmp.iter_columns_full()).fold(0.0,|acc_c,(dm_col, ao_dip_col)| {
            let acc_r = dm_col.iter().zip(ao_dip_col.iter()).fold(0.0, |acc_r, (dm_val, ao_dip_val)| {acc_r + dm_val*ao_dip_val});
            acc_c + acc_r
        });

    }

    cint_data.set_common_origin(p_orig);

    //nucl_dip.iter().zip(el_dip.iter()).map(|(nucl, el)| (*nucl - *el)*AU2DEBYE).collect::<Vec<f64>>()

    tot_dip.iter_mut().zip(el_dip.iter()).for_each(|(nucl, el)| *nucl = (*nucl - *el)*AU2DEBYE);

    tot_dip

}

fn split_indices_by_occ(
    spin_occ: &[f64],
    occ_threshold: f64,
) -> (Vec<usize>, Vec<usize>) {
    let mut occ_idx = Vec::new();
    let mut vir_idx = Vec::new();

    for (i, &n_occ) in spin_occ.iter().enumerate() {
        if n_occ > occ_threshold {
            occ_idx.push(i);
        } else {
            vir_idx.push(i);
        }
    }

    (occ_idx, vir_idx)
}

pub fn split_indices_by_spin_occ(
    occupations: &[Vec<f64>], // occupations[0]=alpha, occupations[1]=beta
    occ_threshold: f64,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    occupations
        .iter()
        .map(|spin_occ| split_indices_by_occ(spin_occ, occ_threshold))
        .collect()
}

fn compress_indices_to_ranges(indices: &[usize]) -> Vec<(usize, usize)> {
    if indices.is_empty() {
        return Vec::new();
    }

    let mut ranges = Vec::new();

    let mut start = indices[0];
    let mut prev  = indices[0];

    for &idx in indices.iter().skip(1) {
        if idx == prev + 1 {
            // still contiguous
            prev = idx;
        } else {
            // end current range
            ranges.push((start, prev));
            start = idx;
            prev  = idx;
        }
    }

    // push last range
    ranges.push((start, prev));

    ranges
}

fn format_ranges(ranges: &[(usize, usize)]) -> String {
    ranges
        .iter()
        .map(|(start, end)| {
            if start == end {
                format!("{}", start)
            } else {
                format!("{}-{}", start, end)
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

pub fn format_indices(indices: &[usize]) -> String {
    let ranges = compress_indices_to_ranges(indices);
    format_ranges(&ranges)
}
