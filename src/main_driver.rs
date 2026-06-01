#![allow(unused)]
extern crate rest_tensors as tensors;
//extern crate rest_libxc as libxc;
extern crate chrono as time;
extern crate hdf5_metno as hdf5;
use std::{f64, fs::File, io::{Write,Read}};
use std::io;
use std::path::PathBuf;
use crate::basis_io::ecp::ghost_effective_potential_matrix;
use crate::external_field::num_dipole::numerical_dipole;
use crate::geom_io::{GeomUnit, get_mass_charge};
use num_traits::Pow;
use pyo3::prelude::*;
use autocxx::prelude::*;
use crate::ctrl_io::JobType;
use crate::constants::{ANG, AU2DEBYE};
use crate::scf_io::{scf_without_build, SCFType, SCF};
use tensors::{MathMatrix, MatrixFull};
use tensors::matrix_blas_lapack::_dsyevd;
use crate::{utilities, ri_pt2, ri_rpa, dft, scf_io, post_scf_analysis};

//use rayon;
use crate::constants::EV;
use crate::grad::{formated_force, numerical_force};
use crate::initial_guess::enxc::{effective_nxc_matrix, effective_nxc_tensors};
//use crate::grad::rhf::Gradient;
use crate::initial_guess::sap::*;

use anyhow;
//use crate::isdf::error_isdf;
use crate::dft::DFA4REST;
use crate::post_scf_analysis::mulliken::mulliken_pop;
use crate::post_scf_analysis::spin_correction::apply_yamaguchi_spin_correction;

//use crate::post_scf_analysis::{post_scf_correlation, print_out_dfa, save_chkfile};
use crate::scf_io::{initialize_scf, scf};
use time::{DateTime,Local};
use crate::molecule_io::Molecule;
//use crate::isdf::error_isdf;
//use crate::dft::DFA4REST;
use crate::post_scf_analysis::{post_scf_correlation, print_out_dfa, save_chkfile, rand_wf_real_space, cube_build, molden_build, post_ai_correction,quasiparticle_methods,rrs_pbc};
use liblbfgs::{lbfgs,Progress};
use crate::mpi_io::{MPIOperator,MPIData};
use std::collections::HashMap;

//use crate::mpi_io::initialization;


pub fn main_driver() -> anyhow::Result<()> {

                                  
    //let mut time_mark = utilities::TimeRecords::new();
    //time_mark.new_item("Overall", "the whole job");
    //time_mark.new_item("SCF", "the scf procedure");




    // VERY IMPORTANCE: introduce mpi_operator:
    let (mpi_operator , mut mpi_data)= MPIData::initialization();

    let ctrl_file = utilities::parse_input().value_of("input_file").unwrap_or("ctrl.in").to_string();
    if ! PathBuf::from(ctrl_file.clone()).is_file() {
        panic!("Input file ({:}) does not exist", ctrl_file);
    }
    let mut mol = Molecule::build(ctrl_file, mpi_data)?;
    if mol.ctrl.print_level>0 {println!("Molecule_name: {}", &mol.geom.name)};
    if mol.ctrl.print_level>=2 {
        println!("{}", mol.ctrl.formated_output_in_toml());
    }
    let mut time_mark = initialize_time_record(&mol);
    time_mark.count_start("Overall");

    if mol.ctrl.deep_pot {
        //let mut scf_data = scf_io::SCF::build(&mut mol);
        let mut effective_hamiltonian = mol.int_ij_matrixupper(String::from("hcore"));
        //effective_hamiltonian.formated_output(5, "full");
        let effective_nxc = effective_nxc_matrix(&mut mol);
        effective_nxc.formated_output(5, "full");
        effective_hamiltonian.data.iter_mut().zip(effective_nxc.data.iter()).for_each(|(to,from)| {*to += from});

        let mut ecp = mol.int_ij_matrixupper(String::from("ecp"));
        //println!("ecp: {:?}", ecp);
        ecp.iter_mut().zip(effective_nxc.iter()).for_each(|(to, from)| {*to -= from});

        ecp.formated_output(5, "full");
        let acc_error = ecp.iter().fold(0.0, |acc, x| {acc + x.abs()});
        println!("acc_error: {}", acc_error);
        
        return Ok(())
    }

    if mol.ctrl.bench_eps {
        let ecp = mol.int_ij_matrixupper(String::from("ecp"));
        let enxc = effective_nxc_matrix(&mut mol);
        let gep = ghost_effective_potential_matrix(
            &mol.cint_env, &mol.cint_atm, &mol.cint_bas, &mol.cint_type, mol.num_basis, 
            &mol.geom.ghost_ep_path, &mol.geom.ghost_ep_pos);

        let d12 = ecp.data.iter().zip(enxc.data.iter()).fold(0.0, |acc, dt| {acc + (dt.0 -dt.1).powf(2.0)});
        let d13 = ecp.data.iter().zip(gep.data.iter()).fold(0.0, |acc, dt| {acc + (dt.0 -dt.1).powf(2.0)});
        let num_data = ecp.data.len() as f64;
        println!("Compare between ECP, ENXC and GEP with the matrix sizes of");
        println!(" {:?}, {:?}, and {:?}, respectively", ecp.size(), enxc.size(), gep.size());
        println!("RMSDs between (ECP, ENXC) and (ECP, GEP): ({:16.8}, {:16.8})", 
            (d12/num_data).powf(0.5), (d13/num_data).powf(0.5)
        );

        return Ok(())
    }
    //check rrs-pbc
    let mut rrs_pbc_index_map = HashMap::new();
    let mut unit_cell_elem = vec![];
    if mol.geom.rrs_pbc {
        rrs_pbc::rrs_pbc_check(&mol);
        (rrs_pbc_index_map, unit_cell_elem) = rrs_pbc::rrs_pbc_match(&mol);
        if mol.ctrl.print_level >= 1 {
            println!("RRS-PBC check result:");
            for key in rrs_pbc_index_map.keys() {
                println!("For step path = {:?}, found target index = {:?}",key,rrs_pbc_index_map.get(key).unwrap());
            };
        };
    }



    // initialize the time record
    // initialize the SCF procedure
    time_mark.count_start("SCF");
    let mut scf_data = scf_io::SCF::build(mol,&mpi_operator);
    time_mark.count("SCF");
    // perform the SCF and post SCF evaluation for the specified xc method
    performance_essential_calculations(&mut scf_data, &mut time_mark, &mpi_operator);

    let spin_correction_scheme: Option<String> = scf_data.mol.ctrl.spin_correction_scheme.clone();
    match spin_correction_scheme.as_deref() {
        Some("yamaguchi") => {
            apply_yamaguchi_spin_correction(&mut scf_data, &mut time_mark, &mpi_operator);
        },
        None => {
        },
        Some(other) => {
            println!(
                "Warning: Unrecognized spin correction scheme '{}'.\nOnly 'yamaguchi' is currently supported. Spin correction will be skipped.",
                other
            );            
        }
    }


    let jobtype = scf_data.mol.ctrl.job_type.clone();
    match jobtype {
        JobType::Force => {
            eval_force(&mut scf_data, &mut time_mark, &mpi_operator);
        },
        JobType::NumDipole => {
            time_mark.count_start("numerical dipole");
            if scf_data.mol.ctrl.print_level>0 {
                println!("Numerical dipole calculation invoked");
            }

            let displace = scf_data.mol.ctrl.ndipole_displacement;
            let dp_au = numerical_dipole(&scf_data, displace);
            let dp = dp_au.iter().map(|x| *x * AU2DEBYE).collect::<Vec<f64>>();
            println!("Dipole Moment in A.U. : {:16.8}, {:16.8}, {:16.8}", dp_au[0], dp_au[1], dp_au[2]);
            println!("Dipole Moment in DEBYE: {:16.8}, {:16.8}, {:16.8}", dp[0], dp[1], dp[2]);
        },
        JobType::GeomOpt => {
            //let opt_engine = scf_data.mol.ctrl.opt_engine.clone().unwrap_or("lbfgs".to_string());
            let opt_engine = scf_data.mol.ctrl.opt_engine.clone().expect("Optimization engine is not specified");
            if opt_engine == "lbfgs" {
                println!("LBFGS geometry optimization invoked");
                time_mark.count_start("geom_opt");
                if scf_data.mol.ctrl.print_level>0 {
                    println!("Geometry optimization invoked");
                }
                let displace = scf_data.mol.ctrl.nforce_displacement/ANG;

                let mut position = scf_data.mol.geom.position.iter().map(|x| *x).collect::<Vec<f64>>();
                lbfgs().minimize(
                    &mut position, 
                    |x: &[f64], gx: &mut [f64]| {
                        scf_data.mol.geom.geom_update(x, GeomUnit::Bohr);
                        if scf_data.mol.ctrl.print_level>0 {
                            println!("Input geometry in this round is:");
                            println!("{}", scf_data.mol.geom.formated_geometry());
                        }
                        //scf_data.mol.ctrl.initial_guess = String::from("inherit");
                        //initialize_scf(&mut scf_data, &mpi_operator);
                        //performance_essential_calculations(&mut scf_data, &mut time_mark, &mpi_operator);
                        //let (energy, nforce) = numerical_force(&scf_data, displace, &mpi_operator);
                        
                        let coords = MatrixFull::from_vec([3, x.len()/3], x.to_vec()).unwrap();
                        let (energy, force) = eval_force_with_position(&mut scf_data, &mut time_mark, &mpi_operator, &coords);

                        gx.iter_mut().zip(force.iter()).for_each(|(to, from)| {*to = *from});

                        if scf_data.mol.ctrl.print_level>0 {
                            println!("Output force in this round [a.u.] is:");
                            println!("{}", formated_force(&force, &scf_data.mol.geom.elem));
                        }

                        Ok(energy)
                    },
                    |prgr| {
                        println!("Iteration {}, Evaluation: {}", &prgr.niter, &prgr.neval);
                        println!(" xnorm = {}, gnorm = {}, step = {}",
                            &prgr.xnorm, &prgr.gnorm, &prgr.step
                        );
                        false
                    },
                );
                println!("Geometry after relaxation [Ang]:");
                println!("{}", scf_data.mol.geom.formated_geometry());
                time_mark.count("geom_opt");

                time_mark.report("geom_opt");
            } else if opt_engine == "geometric_pyo3" {
                #[cfg(feature = "geometric-pyo3")]
                {
                    //println!("Geometric geometry optimization invoked");
                    time_mark.count_start("geom_opt");
                    if scf_data.mol.ctrl.print_level>0 {
                        println!("Geometry optimization invoked using the optimization engine of geometric_pyo3");
                    }
                    geometric_pyo3_impl::optimize_geometric_pyo3(&mut scf_data, &mut time_mark);
                    println!("Geometry after relaxation [Ang]:");
                    println!("{}", scf_data.mol.geom.formated_geometry());
                    time_mark.count("geom_opt");
                    time_mark.report("geom_opt");
                }
                #[cfg(not(feature = "geometric-pyo3"))]
                panic!("Geometric-Pyo3 feature is not enabled. Please enable it in Cargo.toml.");
            } else {
                panic!("Invalid optimization engine: {}", opt_engine);
            }
        },
        // UNVERIFIED NORMAL MODES CALCULATION
        // JobType::NormalModes => {
        //     eval_normal_modes(&mut scf_data, &mut time_mark, &mpi_operator);
        // },
        // ------------
        _ => {}
    }

    if scf_data.mol.ctrl.has_chkfile {
        if let Some(mp_op) = &mpi_operator {
            if mp_op.rank == 0 {
                println!("Rank 0: now save the converged SCF results");
                save_chkfile(&scf_data)
            }
        } else {
            println!("now save the converged SCF results");
            save_chkfile(&scf_data)
        }
    };

    if scf_data.mol.ctrl.check_stab {
        time_mark.new_item("Stability", "the scf stability check");
        time_mark.count_start("Stability");

        scf_data.stability();

        time_mark.count("Stability");
    }

    //====================================
    // Now for post-xc calculations
    //====================================
    if scf_data.mol.ctrl.post_xc.len()>=1 {
        print_out_dfa(&scf_data);
    }

    //====================================
    // Now for post-SCF analysis
    //====================================
    if scf_data.mol.ctrl.print_level > 0 {
        let mulliken = mulliken_pop(&scf_data);
        println!("Mulliken population analysis:");
        let elem_tot = scf_data.mol.geom.elem.clone().into_iter().chain(scf_data.mol.geom.ghost_bs_elem.clone().into_iter()).collect::<Vec<_>>();
        let ghost_atm_start = scf_data.mol.geom.elem.len();
        for (i, (pop, atom)) in mulliken.iter().zip(elem_tot.iter()).enumerate() {
            if i < ghost_atm_start {
                println!("{:3}-{:3}: {:10.6}", i, atom, pop)
            } else {
                println!("{:3}-{:3}: {:10.6}, Ghost Atom", i, atom, pop)
            };
        }
    }
    
    post_scf_analysis::post_scf_output(&scf_data, &mpi_operator);

    //====================================
    // Now for post-correlation calculations
    //====================================
    if scf_data.mol.ctrl.post_correlation.len()>=1 {
        post_scf_correlation(&mut scf_data);
    }

    //===================================
    // Now for rrs-pbc calculations
    //===================================
    if scf_data.mol.geom.rrs_pbc {
        time_mark.count_start("RRS-PBC");
        rrs_pbc::rrs_pbc_output(&scf_data,unit_cell_elem,rrs_pbc_index_map);
        time_mark.count("RRS-PBC");
    }    
    if let Some(qp_ctrl)=scf_data.mol.ctrl.quasiparticle_methods.clone(){
        print!("Now starts quasiparticle method computation!\n");
        quasiparticle_methods(&mut scf_data,&mpi_operator);
    }

    //===================================
    // Now for TDDFT calculations
    //===================================
    if let Some(tddft_ctrl) = &scf_data.mol.ctrl.tddft {
        if !tddft_ctrl.damped_tddft {
            time_mark.new_item("TDDFT", "the TDDFT eigenvalue calculation");
            time_mark.count_start("TDDFT");
            if let Err(e) = crate::ri_tddft::tddft_main(&mut scf_data) {
                eprintln!("Error in TDDFT calculation: {}", e);
            }
            time_mark.count("TDDFT");
        }
    }

    //===================================
    // Now for damped TDDFT calculations
    //===================================
    if let Some(tddft_ctrl) = &scf_data.mol.ctrl.tddft {
        if tddft_ctrl.damped_tddft {
            time_mark.new_item("DampedTDDFT", "the damped TDDFT calculation");
            time_mark.count_start("DampedTDDFT");
            if let Err(e) = crate::ri_tddft::damped_tddft(&mut scf_data) {
                eprintln!("Error in damped TDDFT calculation: {}", e);
            }
            time_mark.count("DampedTDDFT");
        }
    }

    //===================================
    // Now for CP-HF calculations
    //===================================
    if let Some(cphf_ctrl) = &scf_data.mol.ctrl.cphf {
        let label = if cphf_ctrl.solver == "dense" { "dense" } else { "krylov" };
        println!("\n=== CP-HF Calculation (solver={}) ===", label);
        time_mark.new_item("CPHF", &format!("the CP-HF {} solver test", label));
        time_mark.count_start("CPHF");
        if let Err(e) = crate::ri_cphf::test_cphf_dense(&scf_data) {
            eprintln!("Error in CP-HF calculation: {}", e);
        }
        time_mark.count("CPHF");
    }

    time_mark.count("Overall");

    if scf_data.mol.ctrl.print_level > 0 {
        println!("");
        println!("====================================================");
        println!("              REST: Mission accomplished");
        println!("====================================================");
        output_result(&scf_data);
        time_mark.report_all();
    }

    //if let Some(mpi_op) = &mpi_operator {
    //    // I would like to finish the mpi world
    //}


    Ok(())
}

pub fn output_result(scf_data: &scf_io::SCF) {
    
    //--------------------------
    // 0. Solvent energy
    //--------------------------
    if scf_data.mol.use_solvent{
        println!("The solvent energy    : {:18.10} Ha", scf_data.energies.get("solvent_energy").unwrap()[0]);
    }
    let xc_name = scf_data.mol.ctrl.xc.to_lowercase();

    //===========================================================
    // 1. Print SCF energy (Yamaguchi-corrected overrides SCF)
    //===========================================================
    if let Some(v) = scf_data.energies.get("yamaguchi_scf_corrected") {
        println!("The SCF energy        : {:18.10} Ha", v[0]);
    } else {
        if scf_data.mol.ctrl.smear.is_some() {
            println!("The SCF energy (E)    : {:18.10} Ha", scf_data.scf_energy);
            let sigma = scf_data.mol.ctrl.smear_sigma.unwrap_or(0.0);
            let s = scf_data.smearing_entropy;
            println!("Free energy  (E-TS)  : {:18.10} Ha", scf_data.scf_energy - sigma * s);
            println!("Zero-temp energy (E0): {:18.10} Ha", scf_data.scf_energy - 0.5 * sigma * s);
        } else {
            println!("The SCF energy        : {:18.10} Ha", scf_data.scf_energy);
        }
    }


    //===========================================================
    // 2. Helper: fetch corrected or normal total energy
    //===========================================================
    let yamaguchi_tot = scf_data.energies.get("yamaguchi_tot_corrected").map(|v| v[0]);


    //===========================================================
    // 3. Print energies corresponding to the functional family
    //===========================================================
    //--------------------------
    // RPA
    //--------------------------
    if scf_data.mol.xc_data.is_rpa() {
        if let Some(e) = yamaguchi_tot {
            println!("The RPA energy        : {:18.10} Ha", e);
        } else {
            let total = scf_data.energies.get("rpa_energy").unwrap()[0];
            println!("The RPA energy        : {:18.10} Ha", total);
        }
        return;
    }
    //--------------------------
    // DH / ZRPS / SCSRPA
    //--------------------------
    if scf_data.mol.xc_data.is_fifth_dfa() {
        if let Some(e) = yamaguchi_tot {
            println!("The (R)-xDH energy    : {:18.10} Ha", e);
        } else {
            let total = scf_data.energies.get("xdh_energy").unwrap()[0];
            //let post_ai_correction = scf_data.mol.ctrl.post_ai_correction.to_lowercase();
            //let ai_correction = if xc_name.eq("r-xdh7") && post_ai_correction.eq("scc15") {
            //    let ai_correction = scf_data.energies.get("ai_correction").unwrap()[0];
            //    println!("AI Correction         : {:18.10} Ha", ai_correction);
            //    ai_correction
            //} else {
            //    0.0
            //};
            let ai_corr = scf_data.energies.get("ai_correction").map(|v| v[0]).unwrap_or(0.0);
            println!("The (R)-xDH energy    : {:18.10} Ha", total + ai_corr);
        }
        return;
    }



}

/// Perform key SCF and post-SCF calculations
/// Return the total energy of the specfied xc method
/// Assume the initialization of SCF is ready
pub fn performance_essential_calculations(scf_data: &mut SCF, time_mark: &mut utilities::TimeRecords, mpi_operator: &Option<MPIOperator>) -> f64 {

    let mut total_energy = 0.0;

    //=================================================================
    // Now evaluate the SCF energy for the given method
    //=================================================================
    time_mark.count_start("SCF");
    scf_without_build(scf_data, mpi_operator);
    //println!("debug time mark SCF turn off");
    time_mark.count("SCF");

    //==================================================================
    // Now evaluate the advanced correction energy for the given method
    //==================================================================
    //let mut time_mark = utilities::TimeRecords::new();
    if let Some(dft_method) = &scf_data.mol.xc_data.dfa_family_pos {
        match dft_method {
            dft::DFAFamily::PT2 | dft::DFAFamily::SBGE2 => {
                //time_mark.new_item("PT2", "the PT2 evaluation");
                time_mark.count_start("PT2");
                ri_pt2::xdh_calculations(scf_data, mpi_operator);
                time_mark.count("PT2");
            },
            dft::DFAFamily::RPA => {
                //time_mark.new_item("RPA", "the RPA evaluation");
                time_mark.count_start("RPA");
                ri_rpa::rpa_calculations(scf_data, mpi_operator);
                time_mark.count("RPA");
            }
            dft::DFAFamily::SCSRPA => {
                //time_mark.new_item("SCS-RPA", "the SCS-RPA evaluation");
                time_mark.count_start("SCS-RPA");
                ri_pt2::xdh_calculations(scf_data, mpi_operator);
                time_mark.count("SCS-RPA");
            }
            _ => {}
        }
    }
    //====================================
    // Now for post ai correction
    //====================================
    if let Some(scc) = post_ai_correction(scf_data, mpi_operator) {
        scf_data.energies.insert("ai_correction".to_string(), scc);
    }

    collect_total_energy(scf_data)

}

pub fn collect_total_energy(scf_data: &SCF) -> f64 {
    //====================================
    // Determine the total energy
    //====================================
    let mut total_energy = scf_data.scf_energy;
    
    // let xc_name = scf_data.mol.ctrl.xc.to_lowercase();


    // total_energy = match xc_name.as_str() {
    //     "mp2" => scf_data.energies.get("xdh_energy").unwrap()[0],
    //     "scs-mp2" => scf_data.energies.get("xdh_energy").unwrap()[0],
    //     "b2plyp" => scf_data.energies.get("xdh_energy").unwrap()[0],
    //     "b2gpplyp" => scf_data.energies.get("xdh_energy").unwrap()[0],
    //     "pbe-qidh" => scf_data.energies.get("xdh_energy").unwrap()[0],
    //     "pbe0dh" => scf_data.energies.get("xdh_energy").unwrap()[0],
    //     "dsdpbep86-nodisp" => scf_data.energies.get("xdh_energy").unwrap()[0],
    //     "dsdpbep86" => scf_data.energies.get("xdh_energy").unwrap()[0],
    //     "dsdpbeb95" => scf_data.energies.get("xdh_energy").unwrap()[0],
    //     "dsdblyp" => scf_data.energies.get("xdh_energy").unwrap()[0],
    //     "xyg3" => scf_data.energies.get("xdh_energy").unwrap()[0],
    //     "xygjos" => scf_data.energies.get("xdh_energy").unwrap()[0],
    //     "xyg7" => scf_data.energies.get("xdh_energy").unwrap()[0],
    //     "xyg2" => scf_data.energies.get("xdh_energy").unwrap()[0],
    //     "r-xyg3" => scf_data.energies.get("xdh_energy").unwrap()[0],
    //     "r-xygjos" => scf_data.energies.get("xdh_energy").unwrap()[0],
    //     "r-xyg7" => scf_data.energies.get("xdh_energy").unwrap()[0],
    //     "r-xyg2" => scf_data.energies.get("xdh_energy").unwrap()[0],
    //     "xdh-pbe0" => scf_data.energies.get("xdh_energy").unwrap()[0],
    //     "r-xdh7" => scf_data.energies.get("xdh_energy").unwrap()[0],
    //     "zrps" => scf_data.energies.get("xdh_energy").unwrap()[0],
    //     "scsrpa" => scf_data.energies.get("xdh_energy").unwrap()[0],
    //     "rpa@pbe" => scf_data.energies.get("rpa_energy").unwrap()[0],
    //     _ => scf_data.scf_energy,
    // };
    if scf_data.mol.xc_data.is_rpa() {
        total_energy = scf_data.energies.get("rpa_energy").unwrap()[0];
    } else if scf_data.mol.xc_data.is_fifth_dfa() {
        total_energy = scf_data.energies.get("xdh_energy").unwrap()[0];
    } else {
        total_energy = scf_data.scf_energy;
    }

    if let Some(post_ai_correction) = scf_data.energies.get("ai_correction") {
        total_energy += post_ai_correction[0]
    };

    total_energy

}


fn initialize_time_record(mol: &Molecule) -> utilities::TimeRecords {
    let mut time_mark = utilities::TimeRecords::new();
    time_mark.new_item("Overall", "the whole job");
    time_mark.new_item("SCF", "the scf procedure");
    let jobtype = mol.ctrl.job_type.clone();
    match jobtype {
        JobType::GeomOpt => {
            time_mark.new_item("geom_opt", "geometry optimization");
        },
        JobType::Force => {
            time_mark.new_item("force", "force calculation");
        },
        _ => {}
    };
    if let Some(dft_method) = &mol.xc_data.dfa_family_pos {
        match dft_method {
            dft::DFAFamily::PT2 | dft::DFAFamily::SBGE2 => {
                time_mark.new_item("PT2", "the PT2 evaluation");
            },
            dft::DFAFamily::RPA => {
                time_mark.new_item("RPA", "the RPA evaluation");
            }
            dft::DFAFamily::SCSRPA => {
                time_mark.new_item("SCS-RPA", "the SCS-RPA evaluation");
            }
            _ => {}
        }
    }
    if mol.geom.rrs_pbc {
        time_mark.new_item("RRS-PBC", "the RRS-PBC calculation");
    }

    time_mark

}

/* #region force and geomopt utilities */


fn eval_force(scf_data: &mut SCF, time_mark: &mut utilities::TimeRecords, mpi_operator: &Option<MPIOperator>) -> (f64, MatrixFull<f64>) {
    // this is a temporary workaround for the force evaluation
    // currently, this framework could not work for post-scf,
    // especially that currently there's no class that represents post-scf computation

    time_mark.count_start("force");
    if scf_data.mol.ctrl.print_level>0 {
        println!("Force calculation invoked");
    }

    let (energy, gradient) = if scf_data.mol.ctrl.numerical_force {
        if scf_data.mol.ctrl.print_level > 1 {
            println!("Gradient evaluation using numerical differentiation");
        }
        let displace = scf_data.mol.ctrl.nforce_displacement / ANG;
        let (energy, nforce) = numerical_force(&scf_data, displace, &mpi_operator);
        println!("------ Output gradient [a.u.] ------");
        println!("{}", formated_force(&nforce, &scf_data.mol.geom.elem));
        println!("------------------------------------");
        (energy, nforce)
    } else {
        // current available analytical gradients methods:
        // 1) analytical RHF, UHF force
        // 2) dftd force
        // 
        // disallow post-scf calculations for force
        if scf_data.mol.xc_data.is_fifth_dfa() {
            panic!("Analytic Gradient calculation is currently not available for post-SCF methods.");
        }

        if scf_data.mol.ctrl.print_level > 1 {
            println!("Gradient evaluation using Analytical differentiation");
        }

        let is_hf = scf_data.mol.xc_data.dfa_compnt_scf.is_empty();

        // Please note that this is only a temporary workaround implemented gradients.
        // Totally refactor the following code if necessary if other types of gradients to be implemented.
        use crate::grad::traits::GradAPI;

        // we will collect the gradient data from different components into a list,
        // and then summarize the total gradient at the end
        // list of (gradient name, gradient data)
        let mut grad_data_list: Vec<(String, Box<dyn GradAPI>)> = vec![];

        // 1. self-consistent gradient data
        let grad_data_scf: Box<dyn crate::grad::traits::GradAPI> = {
            if !scf_data.mol.ctrl.spin_polarization {
                let mut grad_data_scf = crate::grad::rhf::RIRHFGradient::new(&scf_data);

                if is_hf {
                    grad_data_scf.calc();
                } else {
                    grad_data_scf.calc_rks();
                }

                Box::new(grad_data_scf)
            } else {
                let mut grad_data_scf = crate::grad::uhf::RIUHFGradient::new(&scf_data);

                if is_hf {
                    grad_data_scf.calc();
                } else {
                    grad_data_scf.calc_uks();
                }
                
                Box::new(grad_data_scf)
            }
        };
        grad_data_list.push(("SCF".into(), grad_data_scf));

        // 2. dftd gradient data
        //    we will force to evaluate dftd gradient, since dftd3 is not bottleneck for small to medium molecules
        use crate::grad::dftd::DFTDGrad;
        let mut grad_data_dftd = DFTDGrad::new(&scf_data);
        grad_data_dftd.make_grad();
        // only append the dftd gradient if dftd really exists
        if grad_data_dftd.result.is_some() {
            grad_data_list.push(("DFTD".into(), Box::new(grad_data_dftd)));
        }

        // summarize the gradient contributions by different components
        let natm = scf_data.mol.geom.elem.len();
        let mut gradient = MatrixFull::new([3, natm], 0.0);
        for (grad_name, grad_data) in grad_data_list.iter() {
            let grad_contrib = grad_data.get_gradient();
            gradient.data.iter_mut().zip(grad_contrib.data.iter()).for_each(|(to, from)| {*to += from});
            if scf_data.mol.ctrl.print_level > 1 {
                println!("Gradient contribution from {:} [a.u.]:", grad_name);
                println!("{}", formated_force(&grad_contrib, &scf_data.mol.geom.elem));
            }
        }

        println!("------ Output gradient [a.u.] ------");
        println!("{}", formated_force(&gradient, &scf_data.mol.geom.elem));
        println!("------------------------------------");

        (scf_data.scf_energy, gradient)
    };

    time_mark.count("force");
    time_mark.report("force");

    (energy, gradient)
}

fn eval_force_with_position(scf_data: &mut SCF, time_mark: &mut utilities::TimeRecords, mpi_operator: &Option<MPIOperator>, position: &MatrixFull<f64>) -> (f64, MatrixFull<f64>) {
    scf_data.mol.geom.geom_update(&position.data(), GeomUnit::Bohr);
    if scf_data.mol.ctrl.print_level>0 {
        println!("Input geometry in this round is:");
        println!("{}", scf_data.mol.geom.formated_geometry());
    }
    scf_data.mol.ctrl.initial_guess = String::from("inherit");
    initialize_scf(scf_data, mpi_operator);
    performance_essential_calculations(scf_data, time_mark, mpi_operator);
    let (energy, gradient) = eval_force(scf_data, time_mark, mpi_operator);
    return (energy, gradient);
}

// UNVERIFIED NORMAL MODES CALCULATION
// fn eval_normal_modes(
//     scf_data: &mut SCF,
//     time_mark: &mut utilities::TimeRecords,
//     mpi_operator: &Option<MPIOperator>,
// ) {
//     if scf_data.mol.xc_data.is_fifth_dfa() {
//         panic!("Normal modes calculation is currently not available for post-SCF methods.");
//     }

//     let num_atoms = scf_data.mol.geom.nfree;
//     let dim = num_atoms * 3;
//     let displace_ang = scf_data.mol.ctrl.nhessian_displacement;
//     let displace = displace_ang / ANG; // convert Angstrom to Bohr

//     if scf_data.mol.ctrl.print_level > 0 {
//         println!("");
//         println!("=========================================================");
//         println!("      Vibrational Normal Modes (Frequency) Calculation");
//         println!("=========================================================");
//         println!("Number of atoms:               {}", num_atoms);
//         println!("Hessian dimension:             {}", dim);
//         println!("Finite-difference displacement: {:.6} Bohr ({:.6} Ang)", displace, displace_ang);
//         println!("Number of displaced SCF jobs:  {}", dim * 2);
//         println!("---------------------------------------------------------");
//     }

//     // Build Hessian via central finite difference of analytical gradients
//     let mut hessian = MatrixFull::new([dim, dim], 0.0);

//     if scf_data.mol.ctrl.print_level > 0 {
//         print!("  Hessian finite difference progress: ");
//         io::stdout().flush().unwrap();
//     }

//     for atm_idx in 0..num_atoms {
//         for xyz in 0..3 {
//             let col_idx = atm_idx * 3 + xyz;

//             // +δ displacement
//             let mut vec_plus = vec![0.0; 3];
//             vec_plus[xyz] = displace;
//             let mut scf_plus = scf_data.clone();
//             scf_plus.mol.geom.geom_shift(atm_idx, vec_plus);
//             scf_plus.mol.ctrl.print_level = 0;
//             scf_plus.mol.ctrl.initial_guess = String::from("inherit");
//             initialize_scf(&mut scf_plus, mpi_operator);
//             let _ = performance_essential_calculations(&mut scf_plus, time_mark, mpi_operator);
//             let (_, force_plus) = eval_force(&mut scf_plus, time_mark, mpi_operator);

//             // -δ displacement
//             let mut vec_minus = vec![0.0; 3];
//             vec_minus[xyz] = -displace;
//             let mut scf_minus = scf_data.clone();
//             scf_minus.mol.geom.geom_shift(atm_idx, vec_minus);
//             scf_minus.mol.ctrl.print_level = 0;
//             scf_minus.mol.ctrl.initial_guess = String::from("inherit");
//             initialize_scf(&mut scf_minus, mpi_operator);
//             let _ = performance_essential_calculations(&mut scf_minus, time_mark, mpi_operator);
//             let (_, force_minus) = eval_force(&mut scf_minus, time_mark, mpi_operator);

//             // Hessian column = (F(+) - F(-)) / (2δ)
//             // force_plus/minus are [3, num_atoms], we flatten to 3*num_atoms vector
//             for a in 0..num_atoms {
//                 for d in 0..3 {
//                     let row_idx = a * 3 + d;
//                     hessian[(row_idx, col_idx)] = (force_plus[(d, a)] - force_minus[(d, a)]) / (2.0 * displace);
//                 }
//             }

//             if scf_data.mol.ctrl.print_level > 0 {
//                 print!(".");
//                 io::stdout().flush().unwrap();
//             }
//         }
//     }

//     if scf_data.mol.ctrl.print_level > 0 {
//         println!(" done");
//     }

//     // Symmetrize Hessian: H = (H + H^T) / 2
//     for i in 0..dim {
//         for j in 0..i {
//             let avg = 0.5 * (hessian[(i, j)] + hessian[(j, i)]);
//             hessian[(i, j)] = avg;
//             hessian[(j, i)] = avg;
//         }
//     }

//     // Get atomic masses in amu
//     let mass_charge = get_mass_charge(&scf_data.mol.geom.elem);

//     // Mass-weight Hessian: H̃_ij = H_ij / sqrt(m_i * m_j)
//     let mut mw_hessian = MatrixFull::new([dim, dim], 0.0);
//     for i in 0..dim {
//         let atom_i = i / 3;
//         let mass_i = mass_charge[atom_i].0; // mass in amu
//         for j in 0..=i {
//             let atom_j = j / 3;
//             let mass_j = mass_charge[atom_j].0;
//             let mw_val = hessian[(i, j)] / (mass_i * mass_j).sqrt();
//             mw_hessian[(i, j)] = mw_val;
//             mw_hessian[(j, i)] = mw_val;
//         }
//     }

//     // Diagonalize mass-weighted Hessian
//     let (eigenvectors, eigenvalues, info) = _dsyevd(&mw_hessian, 'V');
//     if info != 0 {
//         panic!("DSYEVD (diagonalization of mass-weighted Hessian) failed with info = {}", info);
//     }

//     let eigvec = eigenvectors.unwrap();

//     // Conversion factor: sqrt(λ / (amu)) to cm⁻¹
//     // λ is eigenvalue of mass-weighted Hessian in Hartree/(Bohr²·amu)
//     // ν̃ (cm⁻¹) = sign(λ) × sqrt(|λ|) × 5140.487
//     const FREQ_CONV: f64 = 5140.487;

//     // Print frequencies sorted by mode number (already sorted by DSYEVD)
//     if scf_data.mol.ctrl.print_level > 0 {
//         println!("");
//         println!("=========================================================");
//         println!("      Vibrational Frequencies");
//         println!("=========================================================");
//         println!("{:>7} {:>22} {:>22}", "Mode", "Eigenvalue", "Freq (cm⁻¹)");
//         println!("---------------------------------------------------------");

//         for i in 0..dim {
//             let lambda = eigenvalues[i];
//             let freq_sign = if lambda < 0.0 { -1.0 } else { 1.0 };
//             let freq = freq_sign * lambda.abs().sqrt() * FREQ_CONV;
//             let lambda_str = if lambda >= 0.0 {
//                 format!("{:22.8}", lambda)
//             } else {
//                 format!("{:22.8}", lambda)
//             };
//             println!("{:7} {:>22} {:>22.4}", i + 1, lambda_str, freq);
//         }
//         println!("=========================================================");

//         // Identify near-zero modes (translations + rotations)
//         println!("");
//         println!("Near-zero modes (translations/rotations):");
//         let threshold = 100.0; // cm⁻¹
//         for i in 0..dim {
//             let lambda = eigenvalues[i];
//             let freq_sign = if lambda < 0.0 { -1.0 } else { 1.0 };
//             let freq = freq_sign * lambda.abs().sqrt() * FREQ_CONV;
//             if freq.abs() < threshold {
//                 println!("  Mode {:4}: {:12.4} cm⁻¹", i + 1, freq);
//             }
//         }
//     }

//     // Print Cartesian normal mode displacements at higher print levels
//     if scf_data.mol.ctrl.print_level > 1 {
//         println!("");
//         println!("Cartesian normal mode displacements:");
//         for i in 0..dim {
//             let lambda = eigenvalues[i];
//             let freq_sign = if lambda < 0.0 { -1.0 } else { 1.0 };
//             let freq = freq_sign * lambda.abs().sqrt() * FREQ_CONV;
//             println!("");
//             println!("Mode {:4} (ω = {:12.4} cm⁻¹):", i + 1, freq);
//             println!("{:>6} {:>14} {:>14} {:>14}", "Atom", "dX", "dY", "dZ");
//             for a in 0..num_atoms {
//                 let dx = eigvec[(a * 3,     i)];
//                 let dy = eigvec[(a * 3 + 1, i)];
//                 let dz = eigvec[(a * 3 + 2, i)];
//                 println!("{:>6} {:14.8} {:14.8} {:14.8}",
//                     scf_data.mol.geom.elem[a], dx, dy, dz);
//             }
//         }
//     }
// }
//------------------

#[cfg(feature = "geometric-pyo3")]
mod geometric_pyo3_impl {
    use super::*;
    use geometric_pyo3::prelude::*;
    use geometric_pyo3::engine::molecule_build_topology;
    use pyo3::prelude::*;

    pub(crate) struct GeometricOptDriver<'a> {
        scf_data: &'a mut SCF,
        time_mark: &'a mut utilities::TimeRecords,
    }

    impl GeomDriverAPI for GeometricOptDriver<'_> {
        fn calc_new(&mut self, coords: &[f64], _dirname: &str) -> GradOutput {
            let coords = coords.to_vec();
            let coords = MatrixFull::from_vec([3, coords.len()/3], coords).unwrap();
            let mpi_operator = None;
            let (scf_data, time_mark) = (&mut self.scf_data, &mut self.time_mark);
            let (energy, gradient) = eval_force_with_position(scf_data, time_mark, &mpi_operator, &coords);
            //gradient.formated_output(3, "full");
            //gradient *= -1.0;
            let gradient = gradient.data();
            return GradOutput {
                energy,
                gradient,
            }
        }
    }

    pub(crate) fn optimize_geometric_pyo3(scf_data: &mut SCF, time_mark: &mut utilities::TimeRecords) -> PyResult<(f64, MatrixFull<f64>)> {
        pyo3::prepare_freethreaded_python();
        
        let elem = scf_data.mol.geom.elem.iter().map(|x| x.as_str()).collect::<Vec<&str>>();
        const BOHR: f64 = crate::constants::BOHR;
        let xyz = scf_data.mol.geom.position.data.iter().map(|x| x * BOHR).collect::<Vec<f64>>();
        //let xyz = scf_data.mol.geom.position.iter().map(|x| *x).collect::<Vec<f64>>();
        let xyzs = vec![xyz];
        let molecule = init_pyo3_molecule(&elem, &xyzs).unwrap();
        molecule_build_topology(&molecule, None).unwrap();
        
        //let optimizer_params = r#"
        //    convergence_energy   = 1.0e-6  # Eh
        //    convergence_grms     = 3.0e-4  # Eh/Bohr
        //    convergence_gmax     = 4.5e-4  # Eh/Bohr
        //    convergence_drms     = 1.2e-3  # Angstrom
        //    convergence_dmax     = 1.8e-3  # Angstrom
        //"#;
        //let mut params = tomlstr2py(optimizer_params)?;

        let mut params = if let Some(params) = &scf_data.mol.ctrl.geometric_pyo3 {
            let params = toml2py(&params.to_toml())?;
            params
        } else {
            panic!("For geometric_pyo3, you must specify the parameters in the control file.")
        };


        let input = None;

        let pyo3_engine_cls = get_pyo3_engine_cls()?;
        let geometric_opt_driver = GeometricOptDriver {
            scf_data,
            time_mark,
        };
        let driver: PyGeomDriver = geometric_opt_driver.into();

        
        let (last_energy, last_coords) = Python::with_gil(|py| -> PyResult<(f64, Vec<f64>)> {
            let custom_engine = pyo3_engine_cls.call1(py, (molecule,))?;
            custom_engine.call_method1(py, "set_driver", (driver,))?;

            let res = run_optimization(custom_engine, &params, input)?;

            let last_energy = res
                .getattr(py, "qm_energies")?
                .call_method1(py, "__getitem__", (-1,))?
                .extract::<f64>(py)?;

            let last_coords = res
                .getattr(py, "xyzs")?
                .call_method1(py, "__getitem__", (-1,))?
                .call_method0(py, "flatten")?
                .call_method0(py, "tolist")?
                .extract::<Vec<f64>>(py)?;

            Ok((last_energy, last_coords))
        })?;

        let last_coords = MatrixFull::from_vec([3, last_coords.len()/3], last_coords).unwrap();

        return Ok((last_energy, last_coords));
    }
}

/* #endregion */
