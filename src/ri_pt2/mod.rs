use std::num;
use std::ops::Range;
use std::sync::Arc;
use std::sync::mpsc::channel;
#[cfg(feature = "mpi")]
use mpi::collective::SystemOperation;
#[cfg(feature = "mpi")]
use crate::mpi_io::{mpi_allreduce, mpi_broadcast, mpi_broadcast_vector};
#[cfg(feature = "mpi")]
use mpi::traits::*;
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, IntoParallelRefIterator};
use rayon::slice::ParallelSlice;
use rest_tensors::{TensorOpt,RIFull, MatrixFull, MatrixFullSlice};
use rest_tensors::matrix_blas_lapack::{_dgemm_nn,_dgemm_tn};
use sbge2::{close_shell_sbge2_rayon_mpi, open_shell_sbge2_rayon_mpi};
use serde::{Deserialize, Serialize};
use tensors::BasicMatrix;
use tensors::matrix_blas_lapack::{_dsymm, _dgemm};

#[cfg(feature = "mpi")]
use crate::ri_pt2::pt2_25d::{initialize_metadata, initialize_metadata_linear, check_memory_25d, swap_ownership, local_computation_close_shell_batch, local_computation_ss_batch, local_computation_os_batch};

use crate::ri_pt2::sbge2::{close_shell_sbge2_rayon,open_shell_sbge2_rayon};
use crate::ri_rpa::scsrpa::{evaluate_osrpa_correlation_rayon, evaluate_osrpa_correlation_rayon_mpi};
use crate::scf_io::{determine_ri3mo_size_for_pt2_and_rpa, scf};
use crate::molecule_io::Molecule;
use crate::scf_io::{SCF, SCFType};
use crate::utilities::{TimeRecords, self};
use crate::mpi_io::MPIOperator;
#[cfg(feature = "mpi")]
use crate::mpi_io::{self, mpi_reduce};
use crate::post_scf_analysis::{split_indices_by_spin_occ, format_indices};

use tensors::matrix_blas_lapack::{omp_get_num_threads_wrapper,omp_set_num_threads_wrapper};

use crate::utilities::memory_batch::{detect_available_memory_mb};

#[cfg(feature = "mpi")]
pub mod pt2_25d;

pub mod sbge2;
pub mod sbge2_25d;
pub mod pure_pt2_pair_eng;
pub mod pt2_pair_eng;

pub mod pure_pt2_r_elecderiv;
pub mod rgfock_pt2;

#[derive(Clone)]
pub struct PT2 {
    pub pt2_type: usize,
    pub pt2_param: [f64;2],
    pub pt2_energy: [f64;2]
}

impl PT2 {
    pub fn new(mol: &Molecule) -> PT2 {
        PT2 {
            pt2_type: 0,
            pt2_param:[1.0,1.0],
            pt2_energy:[0.0,0.0]
        }
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PT2FPMode {
    #[serde(alias = "FP64", alias = "fp64", alias = "f64")]
    FP64,
    #[default]
    #[serde(alias = "FP32", alias = "fp32", alias = "f32")]
    FP32,
}

pub fn xdh_calculations(scf_data: &mut SCF, mpi_operator: &Option<MPIOperator>) -> anyhow::Result<f64> {
    let postscf_method = match scf_data.mol.xc_data.dfa_family_pos.clone().unwrap() {
        crate::dft::DFAFamily::PT2 => "PT2".to_string(),
        crate::dft::DFAFamily::SBGE2 => "SBGE2".to_string(),
        crate::dft::DFAFamily::RPA => "RPA".to_string(),
        crate::dft::DFAFamily::SCSRPA => "SCSRPA".to_string(),
        _ => panic!("Error: No post-scf calculation is needed"),
    };
    if scf_data.mol.ctrl.print_level>0 {
        println!("==========================================");
        println!("Now evaluate the {:>6} correlation energy", postscf_method);
        println!("==========================================");
    }

    let mut pt2_c = [
        // total pt2 correlation energy;
        0.0_f64, 
        // opposite-spin pt2 correlation energy;
        0.0_f64,
        // same-spin pt2 correlation energy;
        0.0_f64
    ];

    let mut timerecords = TimeRecords::new();

    timerecords.new_item("xc_energy", "for x_hf, xc_scf, xc_xdh");
    timerecords.new_item("c_r5dft", "for advanced correlations");
    timerecords.new_item("ao2mo", "for the generation of RI3MO");

    timerecords.count_start("xc_energy");
    let x_energy = scf_data.evaluate_exact_exchange_ri_v(mpi_operator);
    let xc_energy_scf = scf_data.evaluate_xc_energy(0, mpi_operator);
    let xc_energy_xdh = scf_data.evaluate_xc_energy(1, mpi_operator);
    scf_data.energies.insert(String::from("x_hf"), vec![x_energy]);
    timerecords.count("xc_energy");

    // If post_ai_correction (e.g. SCC15) will run after xdh_calculations,
    // pre-compute the PBE exchange energy it needs, since we are about to
    // free the DFT grids.  SCC15 uses x_pbe to compute dxpbe = (x_pbe - x_hf)/x_hf.
    if !scf_data.mol.ctrl.post_ai_correction.to_lowercase().eq("none") {
        if let Some(grids) = &scf_data.grids {
            let dfa = crate::dft::DFA4REST::new_xc(scf_data.mol.spin_channel, scf_data.mol.ctrl.print_level);
            let post_xc_energy = dfa.post_xc_exc(
                &vec![String::from("gga_x_pbe")], grids,
                &scf_data.density_matrix, &scf_data.eigenvectors, &scf_data.occupation,
            );
            let mut x_pbe = post_xc_energy[0][0] + post_xc_energy[0][1];
            // MPI: the DFT grid is rank-distributed, so post_xc_exc returns only the
            // local contribution — sum over all ranks to recover the full value.
            #[cfg(feature = "mpi")]
            if let Some(mpi_op) = mpi_operator {
                use mpi::collective::SystemOperation;
                use mpi::traits::*;
                let mut x_pbe_global = 0.0f64;
                mpi_op.world.any_process().all_reduce_into(&x_pbe, &mut x_pbe_global, &SystemOperation::sum());
                x_pbe = x_pbe_global;
            }
            scf_data.energies.insert(String::from("x_pbe"), vec![x_pbe]);
        }
    }

    // Free DFT grid data before entering PT2/post-SCF correlation.
    // The grids (ao, aop, compressed variants) can consume 40-100+ GB for
    // large systems.  After the XC energy evaluation (and the optional PBE
    // pre-computation above), grids are no longer needed.
    scf_data.grids = None;

    let dfa_family_pos = scf_data.mol.xc_data.dfa_family_pos.clone().unwrap();

    let use_new_driver = scf_data.mol.ctrl.ri_pt2.new_driver
        && mpi_operator.is_none()
        && dfa_family_pos == crate::dft::DFAFamily::PT2
        && matches!(scf_data.scftype, SCFType::RHF | SCFType::UHF);
    
    // Streaming PT2 is enabled when all of:
    //   - ri_pt2.streaming = true
    //   - rimatr is materialized (use_ri_symm = true, not isdf)
    //   - PT2 family (not SBGE2/SCSRPA, which use their own drivers)
    //   - new_driver is not also requested (precedence: new_driver > streaming)
    //   - MPI not active (single-node only for now)
    let use_streaming = !use_new_driver
        && scf_data.mol.ctrl.ri_pt2.streaming
        && mpi_operator.is_none()
        && scf_data.mol.ctrl.use_ri_symm
        && scf_data.rimatr.is_some()
        && dfa_family_pos == crate::dft::DFAFamily::PT2;

    if use_new_driver {
        // we have already checked dfa_family_pos = PT2
        let spin_orb_indices = split_indices_by_spin_occ(&scf_data.occupation, 0.5);
        let spin_channel = scf_data.mol.spin_channel;
        // frozen-core approximation: the occupied space starts at mol.start_mo,
        // so orbitals below idx_core must be excluded from the PT2 pair sums
        let idx_core = scf_data.mol.start_mo;
        let occ_filtered: Vec<Vec<usize>> = (0..spin_channel)
            .map(|i_spin| {
                spin_orb_indices[i_spin]
                    .0
                    .iter()
                    .copied()
                    .filter(|&i| i >= idx_core)
                    .collect()
            })
            .collect();
        let vir_filtered: Vec<Vec<usize>> = (0..spin_channel)
            .map(|i_spin| spin_orb_indices[i_spin].1.clone())
            .collect();
        let mut occidx: [Option<&[usize]>; 2] = [None, None];
        let mut viridx: [Option<&[usize]>; 2] = [None, None];
        for i_spin in 0..spin_channel {
            occidx[i_spin] = Some(occ_filtered[i_spin].as_slice());
            viridx[i_spin] = Some(vir_filtered[i_spin].as_slice());
        }
        
        let pt2_fp_mode = scf_data.mol.ctrl.ri_pt2.fp_mode;
        pt2_c = match scf_data.scftype {
            SCFType::RHF => match pt2_fp_mode {
                PT2FPMode::FP64 => pt2_pair_eng::evaluate_ript2_eng::<f64>(scf_data, &mut timerecords, occidx, viridx),
                PT2FPMode::FP32 => pt2_pair_eng::evaluate_ript2_eng::<f32>(scf_data, &mut timerecords, occidx, viridx),
            },
            SCFType::UHF => match pt2_fp_mode {
                PT2FPMode::FP64 => pt2_pair_eng::evaluate_riupt2_eng::<f64>(scf_data, &mut timerecords, occidx, viridx),
                PT2FPMode::FP32 => pt2_pair_eng::evaluate_riupt2_eng::<f32>(scf_data, &mut timerecords, occidx, viridx),
            },
            SCFType::ROHF => unreachable!("currently not implemented, and should not go here due to `use_new_driver` condition"),
        };
    } else if use_streaming {
        // Streaming PT2: process occ orbitals in blocks, never materialize full ri3mo.
        // See `*_pt2_rayon_streaming` docstrings for memory and FLOPs analysis.
        // ROHF needs semi-canonical orbitals/eigenvalues/Fock; mirror what
        // generate_ri3mo_rayon does at scf_io/mod.rs:3245-3247.
        if let SCFType::ROHF = scf_data.scftype {
            scf_data.semi_diagonalize_hamiltonian();
        }
        let block_size = scf_data.mol.ctrl.ri_pt2.stream_block_size;
        timerecords.count_start("ao2mo");
        // (ao2mo is interleaved with PT2 contraction inside the streaming drivers;
        // we keep the timer label for parity with the legacy driver.)
        timerecords.count("ao2mo");
        timerecords.count_start("c_r5dft");
        pt2_c = match scf_data.scftype {
            SCFType::RHF => close_shell_pt2_rayon_streaming(scf_data, block_size).unwrap(),
            SCFType::UHF => open_shell_pt2_rayon_streaming(scf_data, block_size).unwrap(),
            SCFType::ROHF => restricted_open_shell_pt2_rayon_streaming(scf_data, block_size).unwrap(),
        };
        timerecords.count("c_r5dft");
    } else if scf_data.mol.ctrl.use_ri_symm {
        timerecords.count_start("ao2mo");
        crate::scf_io::generate_ri3mo_rayon_for_pt2_and_rpa(scf_data);
        timerecords.count("ao2mo");
        timerecords.count_start("c_r5dft");

        #[cfg(feature = "mpi")]
        let use_25d = check_conditions_25d(&scf_data, mpi_operator, &dfa_family_pos);
        #[cfg(not(feature = "mpi"))]
        let use_25d = false;

        #[cfg(feature = "mpi")]
        if use_25d {
            pt2_c = match scf_data.scftype {
                SCFType::RHF => match  dfa_family_pos {
                    crate::dft::DFAFamily::PT2 => close_shell_pt2_rayon_mpi_25d(&scf_data,mpi_operator).unwrap(),
                    crate::dft::DFAFamily::SBGE2 => crate::ri_pt2::sbge2_25d::close_shell_sbge2_rayon_mpi_25d(&scf_data, mpi_operator).unwrap(),
                    crate::dft::DFAFamily::SCSRPA => crate::ri_rpa::scsrpa_25d::close_shell_osrpa_rayon_mpi_25d(scf_data, mpi_operator).unwrap(),
                    _ => [0.0,0.0,0.0]
                },
                SCFType::UHF => match  dfa_family_pos {
                    crate::dft::DFAFamily::PT2 => open_shell_pt2_rayon_mpi_25d(&scf_data, mpi_operator).unwrap(),
                    crate::dft::DFAFamily::SBGE2 => crate::ri_pt2::sbge2_25d::open_shell_sbge2_rayon_mpi_25d(&scf_data, mpi_operator).unwrap(),
                    crate::dft::DFAFamily::SCSRPA => crate::ri_rpa::scsrpa_25d::open_shell_osrpa_rayon_mpi_25d(scf_data, mpi_operator).unwrap(),
                    _ => [0.0,0.0,0.0]
                },
                SCFType::ROHF => match dfa_family_pos {
                    crate::dft::DFAFamily::PT2 => restricted_open_shell_pt2_rayon_mpi_25d(&scf_data, mpi_operator).unwrap(),
                    crate::dft::DFAFamily::SBGE2 => crate::ri_pt2::sbge2_25d::open_shell_sbge2_rayon_mpi_25d(&scf_data, mpi_operator).unwrap(),
                    crate::dft::DFAFamily::SCSRPA => crate::ri_rpa::scsrpa_25d::open_shell_osrpa_rayon_mpi_25d(scf_data, mpi_operator).unwrap(),
                    _ => [0.0,0.0,0.0]
                }
            };
        }
        if !use_25d {
            pt2_c = match scf_data.scftype {
                SCFType::RHF => match  dfa_family_pos {
                    crate::dft::DFAFamily::PT2 => close_shell_pt2_rayon_mpi(&scf_data,mpi_operator).unwrap(),
                    crate::dft::DFAFamily::SBGE2 => close_shell_sbge2_rayon_mpi(scf_data,mpi_operator).unwrap(),
                    crate::dft::DFAFamily::SCSRPA => evaluate_osrpa_correlation_rayon_mpi(scf_data, mpi_operator).unwrap(),
                    _ => [0.0,0.0,0.0]
                },
                SCFType::UHF => match  dfa_family_pos {
                    crate::dft::DFAFamily::PT2 => open_shell_pt2_rayon_mpi(&scf_data, mpi_operator).unwrap(),
                    crate::dft::DFAFamily::SBGE2 => open_shell_sbge2_rayon_mpi(scf_data, mpi_operator).unwrap(),
                    crate::dft::DFAFamily::SCSRPA => evaluate_osrpa_correlation_rayon_mpi(scf_data, mpi_operator).unwrap(),
                    _ => [0.0,0.0,0.0]
                },
                SCFType::ROHF => match dfa_family_pos {
                    crate::dft::DFAFamily::PT2 => restricted_open_shell_pt2_rayon_mpi(&scf_data, mpi_operator).unwrap(),
                    crate::dft::DFAFamily::SBGE2 => open_shell_sbge2_rayon_mpi(scf_data, mpi_operator).unwrap(),
                    crate::dft::DFAFamily::SCSRPA => evaluate_osrpa_correlation_rayon_mpi(scf_data, mpi_operator).unwrap(),
                    _ => [0.0,0.0,0.0]
                }
            };
        }

        timerecords.count("c_r5dft");
    } else {
        // Phase-0 guard (P0-1): the primitive path below implements ONLY the PT2 kernel.
        // For the SCSRPA/RPA families it would silently compute an MP2-like correlation
        // energy and store it under the "scsrpa"/"rpa" label — a wrong total energy with
        // no warning. Hard-error instead of silent wrong physics.
        match dfa_family_pos {
            crate::dft::DFAFamily::SCSRPA | crate::dft::DFAFamily::RPA => panic!(
                "xc = `{}` (post-SCF family {:?}): the non-RI-symm path (use_ri_symm = false) \
                 only implements the PT2 kernel and would silently return a wrong correlation \
                 energy for this family. Please set use_ri_symm = true.",
                scf_data.mol.ctrl.xc, dfa_family_pos
            ),
            _ => {}
        }
        pt2_c = if scf_data.mol.spin_channel == 1 {
            close_shell_pt2(&scf_data).unwrap()
        } else {
            open_shell_pt2(&scf_data).unwrap()
        };
    };

    match dfa_family_pos {
        crate::dft::DFAFamily::PT2 => scf_data.energies.insert(String::from("pt2"), pt2_c.to_vec()),
        crate::dft::DFAFamily::SBGE2 => scf_data.energies.insert(String::from("sbge2"), pt2_c.to_vec()),
        crate::dft::DFAFamily::SCSRPA => scf_data.energies.insert(String::from("scsrpa"), pt2_c.to_vec()),
        crate::dft::DFAFamily::RPA => scf_data.energies.insert(String::from("rpa"), pt2_c.to_vec()),
        _ => scf_data.energies.insert(String::from("unknown"), pt2_c.to_vec())
    };

    if scf_data.mol.ctrl.xc.eq(&"scsrpa") {
        let pt2_c_old = pt2_c.clone();
        pt2_c[2] = pt2_c[0] - pt2_c[1];
    };
    
    //println!("{:?}",&pt2_c);
    let hy_coeffi_scf = scf_data.mol.xc_data.dfa_hybrid_scf;
    let hy_coeffi_xdh = if let Some(coeff) = scf_data.mol.xc_data.dfa_hybrid_pos {coeff} else {0.0};
    let hy_coeffi_pt2 = if let Some(coeff) = &scf_data.mol.xc_data.dfa_paramr_adv {coeff.clone()} else {vec![0.0,0.0]};
    let xdh_pt2_energy: f64 = pt2_c[1..3].iter().zip(hy_coeffi_pt2.iter()).map(|(e,c)| e*c).sum();
    if scf_data.mol.ctrl.print_level>2 {
        println!("Exc_scf: ({:?},{:?}),Exc_pos: ({:?},{:?})",xc_energy_scf,hy_coeffi_scf,xc_energy_xdh,hy_coeffi_xdh);
    }
    let total_energy = scf_data.scf_energy +
                            x_energy * (hy_coeffi_xdh-hy_coeffi_scf) +
                            xc_energy_xdh-xc_energy_scf +
                            xdh_pt2_energy;

    if scf_data.mol.ctrl.print_level>0 {
        println!("----------------------------------------------------------------------");
        println!("{:16}: {:>16}, {:>16}, {:>16}", "Methods", "Total Corr", "OS Corr", "SS Corr");
        println!("----------------------------------------------------------------------");
        println!("{:16}: {:16.8}, {:16.8}, {:16.8}", 
            dfa_family_pos.to_name(), pt2_c[0], pt2_c[1], pt2_c[2]);
        println!("----------------------------------------------------------------------");
        println!("Fifth-rung correlation energy : {:20.12} Ha", xdh_pt2_energy);
        println!("E[{:5}]=: {:16.8} Ha, Ex[HF]: {:16.8} Ha, Ec[{:5}]: {:16.8} Ha", 
            scf_data.mol.ctrl.xc.to_uppercase(), 
            total_energy, 
            x_energy, 
            postscf_method,
            xdh_pt2_energy
        );
        println!("Exc[KS-DFA]: {:16.8} Ha", xc_energy_xdh);
    }

    scf_data.energies.insert(String::from("xdh_energy"), vec![total_energy]);

    if scf_data.mol.ctrl.print_level>1 {timerecords.report_all()};

    Ok(total_energy)
}


/// ==========================================================================================
///    `E_c[PT2]`=\sum_{i<j}^{occ}\sum_{a<b}^{vir}         |(ia||jb)|^2
///                                                x --------------------------
///                                                    e_i+e_j-e_a-e_b
///  For each electron-pair correlation e_{ij}:
///    e_{ij} = \sum_{a<b}^{vir}          |(ia||jb)|^2
///                              x --------------------------
///                                 e_i+e_j-e_a-e_b
///           = \sum_{a<b}^{vir}      |(ia|jb)-(ib|ja)|^2
///                              x --------------------------
///                                 e_i+e_j-e_a-e_b
///  Then:
///   `E_c[PT2]`=\sum{i<j}^{occ}e_{ij}
/// ==========================================================================================
fn close_shell_pt2(scf_data: &SCF) -> anyhow::Result<[f64;3]> {
    if let Some(riao)=&scf_data.ri3fn {
        let mut e_mp2_ss = 0.0_f64;
        let mut e_mp2_os = 0.0_f64;
        let eigenvector = scf_data.eigenvectors.get(0).unwrap();
        let eigenvalues = scf_data.eigenvalues.get(0).unwrap();

        let homo = scf_data.homo.get(0).unwrap().clone();
        let lumo = scf_data.lumo.get(0).unwrap().clone();
        let num_basis = eigenvector.size.get(0).unwrap().clone();
        let num_state = eigenvector.size.get(1).unwrap().clone();
        let start_mo: usize = scf_data.mol.start_mo;
        //let num_occu = homo + 1;
        //let num_occu = lumo;
        //let num_occu = scf_data.mol.num_elec.get(i_spin + 1).unwrap().clone() as usize;
        let num_occu = if scf_data.mol.num_elec[0] <= 1.0e-6 {0} else {homo + 1};
        //let num_virt = num_state - num_occu;
        //println!("{:?},{:?},{:?},{:?}",homo,lumo,num_state, num_basis);
        //for i in 0..homo {
        //    for j in i..homo {
        //    }
        //}
        let mut tmp_record = TimeRecords::new();
        tmp_record.new_item("rimo", "the generation of three-center RI tensor for MO");
        tmp_record.count_start("rimo");
        let mut rimo = riao.ao2mo(eigenvector).unwrap();
        tmp_record.count("rimo");

        tmp_record.new_item("dgemm", "prepare four-center integrals from RI-MO");
        tmp_record.new_item("get2d", "get the ERI values");
        for i_state in start_mo..num_occu {
            let i_state_eigen = eigenvalues.get(i_state).unwrap();
            for j_state in i_state..num_occu {

                let mut e_mp2_term_ss = 0.0_f64;
                let mut e_mp2_term_os = 0.0_f64;

                let j_state_eigen = eigenvalues.get(j_state).unwrap();
                let ij_state_eigen = i_state_eigen + j_state_eigen;

                tmp_record.count_start("dgemm");
                let ri_i = rimo.get_reducing_matrix(i_state).unwrap();
                let ri_j = rimo.get_reducing_matrix(j_state).unwrap();
                //if (i_state == start_mo && j_state == i_state) {println!("debug {:?}", &ri_i.get_slice_x(lumo))};
                let eri_virt = _dgemm_tn(&ri_i,&ri_j);
                tmp_record.count("dgemm");

                for i_virt in lumo..num_state {
                    let i_virt_eigen = eigenvalues.get(i_virt).unwrap();
                    for j_virt in lumo..num_state {

                        let j_virt_eigen = eigenvalues.get(j_virt).unwrap();
                        let ij_virt_eigen = i_virt_eigen + j_virt_eigen;

                        let mut double_gap = ij_virt_eigen - ij_state_eigen;
                        if double_gap.abs()<=1.0E-6 {
                            println!("Warning: too close to degeneracy");
                            double_gap = 1.0e-6;
                        };

                        //tmp_record.count_start("get2d");
                        let e_mp2_a = eri_virt.get2d([i_virt,j_virt]).unwrap();
                        let e_mp2_b = eri_virt.get2d([j_virt,i_virt]).unwrap();
                        //tmp_record.count("get2d");
                        e_mp2_term_ss += (e_mp2_a - e_mp2_b) * e_mp2_a / double_gap;
                        e_mp2_term_os += e_mp2_a * e_mp2_a / double_gap;

                    }
                }
                if i_state != j_state {
                    e_mp2_term_ss *= 2.0;
                    e_mp2_term_os *= 2.0;
                }
                //println!("{:?},{:?},{:?}, {:?}",i_state, j_state,e_mp2_term_os,e_mp2_term_ss);
                e_mp2_ss -= e_mp2_term_ss;
                e_mp2_os -= e_mp2_term_os;
            }
        }
        //tmp_record.report_all();
        return(Ok([e_mp2_ss+e_mp2_os,e_mp2_os,e_mp2_ss]));
    } else {
        panic!("ri3fn should be initialized for RI-PT2 calculations")
    };

}

fn open_shell_pt2(scf_data: &SCF) -> anyhow::Result<[f64;3]> {
    if let Some(riao)=&scf_data.ri3fn {
        let start_mo: usize = scf_data.mol.start_mo;
        let mut e_mp2_ss = 0.0_f64;
        let mut e_mp2_os = 0.0_f64;
        let num_basis = scf_data.mol.num_basis;
        let num_state = scf_data.mol.num_state;
        let spin_channel = scf_data.mol.spin_channel;
        let i_spin_pair: [(usize,usize);3] = [(0,0),(0,1),(1,1)];
        for (i_spin_1,i_spin_2) in i_spin_pair {
            if i_spin_1 == i_spin_2 {

                let i_spin = i_spin_1;
                let eigenvector = scf_data.eigenvectors.get(i_spin).unwrap();
                let eigenvalues = scf_data.eigenvalues.get(i_spin).unwrap();

                let homo = scf_data.homo.get(i_spin).unwrap().clone();
                let lumo = scf_data.lumo.get(i_spin).unwrap().clone();
                //let num_occu = homo + 1;
                //let num_occu = lumo;
                //let num_occu = scf_data.mol.num_elec.get(i_spin + 1).unwrap().clone() as usize;
                let num_occu = if scf_data.mol.num_elec[i_spin+1] <= 1.0e-6 {0} else {homo + 1};

                let mut rimo = riao.ao2mo(eigenvector).unwrap();

                for i_state in start_mo..num_occu {
                    let i_state_eigen = eigenvalues.get(i_state).unwrap();
                    for j_state in i_state+1..num_occu {

                        let mut e_mp2_term_ss = 0.0_f64;

                        let j_state_eigen = eigenvalues.get(j_state).unwrap();
                        let ij_state_eigen = i_state_eigen + j_state_eigen;
                        let ri_i = rimo.get_reducing_matrix(i_state).unwrap();
                        let ri_j = rimo.get_reducing_matrix(j_state).unwrap();
                        let eri_virt = _dgemm_tn(&ri_i,&ri_j);

                        for i_virt in lumo..num_state {
                            let i_virt_eigen = eigenvalues.get(i_virt).unwrap();
                            for j_virt in i_virt+1..num_state {
                                let j_virt_eigen = eigenvalues.get(j_virt).unwrap();
                                let ij_virt_eigen = i_virt_eigen + j_virt_eigen;

                                let mut double_gap = ij_virt_eigen - ij_state_eigen;
                                if double_gap.abs()<=10E-6 {
                                    println!("Warning: too close to degeneracy")
                                };

                                let e_mp2_a = eri_virt.get2d([i_virt,j_virt]).unwrap();
                                let e_mp2_b = eri_virt.get2d([j_virt,i_virt]).unwrap();
                                e_mp2_term_ss += (e_mp2_a - e_mp2_b).powf(2.0) / double_gap;
                                //e_mp2_term_os += e_mp2_a * e_mp2_a / double_gap;

                            }
                        }
                        e_mp2_ss -= e_mp2_term_ss;
                    }
                }
            } else {
                let eigenvector_1 = scf_data.eigenvectors.get(i_spin_1).unwrap();
                let eigenvalues_1 = scf_data.eigenvalues.get(i_spin_1).unwrap();
                let homo_1 = scf_data.homo.get(i_spin_1).unwrap().clone();
                let lumo_1 = scf_data.lumo.get(i_spin_1).unwrap().clone();
                //let num_occu_1 = homo_1 + 1;
                //let num_occu_1 = lumo_1;
                let num_occu_1 = if scf_data.mol.num_elec[i_spin_1+1] <= 1.0e-6 {0} else {homo_1 + 1};
                let mut rimo_1 = riao.ao2mo_v01(eigenvector_1).unwrap();

                let eigenvector_2 = scf_data.eigenvectors.get(i_spin_2).unwrap();
                let eigenvalues_2 = scf_data.eigenvalues.get(i_spin_2).unwrap();
                let homo_2 = scf_data.homo.get(i_spin_2).unwrap().clone();
                let lumo_2 = scf_data.lumo.get(i_spin_2).unwrap().clone();
                //let num_occu_2 = homo_2 + 1;
                //let num_occu_2 = lumo_2 + 1;
                let num_occu_2 = if scf_data.mol.num_elec[i_spin_2+1] <= 1.0e-6 {0} else {homo_2 + 1};
                let mut rimo_2 = riao.ao2mo_v01(eigenvector_2).unwrap();
                for i_state in start_mo..num_occu_1 {
                    let i_state_eigen = eigenvalues_1.get(i_state).unwrap();
                    let ri_i = rimo_1.get_reducing_matrix(i_state).unwrap();
                    for j_state in start_mo..num_occu_2 {

                        let mut e_mp2_term_os = 0.0_f64;

                        let j_state_eigen = eigenvalues_2.get(j_state).unwrap();
                        let ri_j = rimo_2.get_reducing_matrix(j_state).unwrap();

                        let ij_state_eigen = i_state_eigen + j_state_eigen;
                        let eri_virt = _dgemm_tn(&ri_i,&ri_j);

                        for i_virt in lumo_1..num_state {
                            let i_virt_eigen = eigenvalues_1.get(i_virt).unwrap();
                            for j_virt in lumo_2..num_state {
                                let j_virt_eigen = eigenvalues_2.get(j_virt).unwrap();
                                let ij_virt_eigen = i_virt_eigen + j_virt_eigen;

                                let mut double_gap = ij_virt_eigen - ij_state_eigen;
                                if double_gap.abs()<=1.0E-6 {
                                    println!("Warning: too close to degeneracy");
                                    double_gap = 1.0e-6;
                                };

                                let e_mp2_a = eri_virt.get2d([i_virt,j_virt]).unwrap();
                                e_mp2_term_os += e_mp2_a * e_mp2_a / double_gap;

                            }
                        }
                        e_mp2_os -= e_mp2_term_os;
                    }
                }
            }
        }
        return(Ok([e_mp2_ss+e_mp2_os,e_mp2_os,e_mp2_ss]));
    } else {
        panic!("ri3fn should be initialized for RI-PT2 calculations")
    };

}

pub fn close_shell_pt2_rayon(scf_data: &SCF) -> anyhow::Result<[f64;3]> {
    //let mut tmp_record = TimeRecords::new();

    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = scf_data.mol.ctrl.num_threads.unwrap();

    let mut e_mp2_ss = 0.0_f64;
    let mut e_mp2_os = 0.0_f64;

    if let Some(ri3mo_vec) = &scf_data.ri3mo {
        let (rimo, vir_range, occ_range) = &ri3mo_vec[0];

        let eigenvector = scf_data.eigenvectors.get(0).unwrap();
        let eigenvalues = scf_data.eigenvalues.get(0).unwrap();
        let occupation = scf_data.occupation.get(0).unwrap();

        let homo = scf_data.homo.get(0).unwrap().clone();
        let lumo = scf_data.lumo.get(0).unwrap().clone();
        let num_basis = eigenvector.size.get(0).unwrap().clone();
        let num_auxbas = rimo.size[0];
        let num_state = eigenvector.size.get(1).unwrap().clone();
        let start_mo: usize = scf_data.mol.start_mo;
        //let num_occu = homo + 1;
        //let num_occu = lumo;
        let num_occu = if scf_data.mol.num_elec[0] <= 1.0e-6 {0} else {homo + 1};
        //tmp_record.new_item("dgemm", "prepare four-center integrals from RI-MO");
        //tmp_record.new_item("get2d", "get the ERI values");
        let mut elec_pair: Vec<[usize;2]> = vec![];
        for i_state in start_mo..num_occu {
            for j_state in i_state..num_occu {
                elec_pair.push([i_state,j_state])
            }
        };
        let (sender, receiver) = channel();
        elec_pair.par_iter().for_each_with(sender,|s,i_pair| {
            omp_set_num_threads_wrapper(1);
            let mut e_mp2_term_ss = 0.0_f64;
            let mut e_mp2_term_os = 0.0_f64;

            let i_state = i_pair[0];
            let j_state = i_pair[1];
            let i_state_eigen = eigenvalues.get(i_state).unwrap();
            let j_state_eigen = eigenvalues.get(j_state).unwrap();
            let ij_state_eigen = i_state_eigen + j_state_eigen;
            let i_state_occ = occupation.get(i_state).unwrap()/2.0;
            let j_state_occ = occupation.get(j_state).unwrap()/2.0;

            // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
            // the indices in rimo are shifted

            if i_state_occ.abs() > 1.0e-6 && j_state_occ.abs() > 1.0e-6 {
                let i_loc_state = i_state-occ_range.start;
                let j_loc_state = j_state-occ_range.start;
                let ri_i = rimo.get_reducing_matrix(i_loc_state).unwrap();
                let ri_j = rimo.get_reducing_matrix(j_loc_state).unwrap();
                let mut eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                _dgemm(
                    &ri_i, (0..num_auxbas,0..vir_range.len()), 'T', 
                    &ri_j,(0..num_auxbas,0..vir_range.len()) , 'N', 
                    &mut eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                    1.0,0.0);

                for i_virt in lumo..num_state {
                    let i_virt_eigen = eigenvalues.get(i_virt).unwrap();
                    for j_virt in lumo..num_state {
                        let i_virt_occ = occupation.get(i_virt).unwrap()/2.0;
                        let j_virt_occ = occupation.get(j_virt).unwrap()/2.0;
                        if (1.0-i_virt_occ).abs() > 1.0e-6 && (1.0-j_virt_occ).abs() > 1.0e-6 {
                            let j_virt_eigen = eigenvalues.get(j_virt).unwrap();
                            let ij_virt_eigen = i_virt_eigen + j_virt_eigen;

                            let mut double_gap = (ij_virt_eigen - ij_state_eigen);
                            if double_gap.abs()<=1.0E-6 {
                                println!("Warning: too close to degeneracy");
                                double_gap = 1.0e-6;
                            };
                            double_gap /= (i_state_occ*j_state_occ*(1.0-i_virt_occ)*(1.0-j_virt_occ));

                            // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                            // the indices in rimo are shifted
                            let i_loc_virt = i_virt-vir_range.start;
                            let j_loc_virt = j_virt-vir_range.start;

                            let e_mp2_a = eri_virt.get2d([i_loc_virt,j_loc_virt]).unwrap();
                            let e_mp2_b = eri_virt.get2d([j_loc_virt,i_loc_virt]).unwrap();
                            e_mp2_term_ss += (e_mp2_a - e_mp2_b) * e_mp2_a / double_gap;
                            e_mp2_term_os += e_mp2_a * e_mp2_a / double_gap;
                        }
                    }
                }
            }
            if i_state != j_state {
                e_mp2_term_ss *= 2.0;
                e_mp2_term_os *= 2.0;
            }

            s.send((e_mp2_term_os, e_mp2_term_ss)).unwrap()

        });
        receiver.into_iter().for_each(|(e_mp2_term_os,e_mp2_term_ss)| {
            e_mp2_os -= e_mp2_term_os;
            e_mp2_ss -= e_mp2_term_ss;
        });
    } else {
        panic!("RI3MO should be initialized before the PT2 calculations")
    };

    // reuse the default omp_num_threads setting
    omp_set_num_threads_wrapper(default_omp_num_threads);
    //tmp_record.report_all();
    Ok([e_mp2_ss+e_mp2_os,e_mp2_os,e_mp2_ss])

}


pub fn open_shell_pt2_rayon(scf_data: &SCF) -> anyhow::Result<[f64;3]> {
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = scf_data.mol.ctrl.num_threads.unwrap();

    let mut e_mp2_ss = 0.0_f64;
    let mut e_mp2_os = 0.0_f64;

    if let Some(ri3mo_vec) = &scf_data.ri3mo {

        let start_mo: usize = scf_data.mol.start_mo;
        let num_basis = scf_data.mol.num_basis;
        let num_state = scf_data.mol.num_state;
        let num_auxbas = scf_data.mol.num_auxbas;
        let spin_channel = scf_data.mol.spin_channel;
        let i_spin_pair: [(usize,usize);3] = [(0,0),(0,1),(1,1)];

        for (i_spin_1,i_spin_2) in i_spin_pair {
            if i_spin_1 == i_spin_2 {

                let i_spin = i_spin_1;
                let eigenvector = scf_data.eigenvectors.get(i_spin).unwrap();
                let eigenvalues = scf_data.eigenvalues.get(i_spin).unwrap();
                let occupation = scf_data.occupation.get(i_spin).unwrap();

                let homo = scf_data.homo.get(i_spin).unwrap().clone();
                let lumo = scf_data.lumo.get(i_spin).unwrap().clone();
                //let num_occu = homo + 1;
                //let num_occu = lumo;
                let num_occu = if scf_data.mol.num_elec[i_spin + 1] <= 1.0e-6 {0} else {homo + 1};

                let (rimo, vir_range, occ_range) = &ri3mo_vec[i_spin];

                //let mut rimo = riao.ao2mo(eigenvector).unwrap();
                let mut elec_pair: Vec<[usize;2]> = vec![];
                for i_state in start_mo..num_occu {
                    for j_state in i_state..num_occu {
                        elec_pair.push([i_state,j_state])
                    }
                };
                let (sender, receiver) = channel();
                elec_pair.par_iter().for_each_with(sender,|s,i_pair| {
                    omp_set_num_threads_wrapper(1);

                    let mut e_mp2_term_ss = 0.0_f64;

                    let i_state = i_pair[0];
                    let j_state = i_pair[1];
                    let i_state_eigen = eigenvalues.get(i_state).unwrap();
                    let j_state_eigen = eigenvalues.get(j_state).unwrap();
                    let ij_state_eigen = i_state_eigen + j_state_eigen;
                    let i_state_occ = occupation.get(i_state).unwrap();
                    let j_state_occ = occupation.get(j_state).unwrap();

                    if i_state_occ.abs() > 1.0e-6 && j_state_occ.abs() > 1.0e-6 {
                        // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                        // the indices in rimo are shifted
                        let i_loc_state = i_state-occ_range.start;
                        let j_loc_state = j_state-occ_range.start;
                        let ri_i = rimo.get_reducing_matrix(i_loc_state).unwrap();
                        let ri_j = rimo.get_reducing_matrix(j_loc_state).unwrap();
                        let mut eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                        _dgemm(
                            &ri_i, (0..num_auxbas,0..vir_range.len()), 'T', 
                            &ri_j,(0..num_auxbas,0..vir_range.len()) , 'N', 
                            &mut eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                            1.0,0.0);
                        //// ==== DEBUG IGOR ====
                        //if i_state == 1 && j_state== 2 {
                        //    eri_virt.formated_output(5, "full");
                        //}
                        //// ==== DEBUG IGOR ====
                        for i_virt in lumo..num_state {
                            let i_virt_eigen = eigenvalues[i_virt];
                            for j_virt in i_virt+1..num_state {
                                let j_virt_eigen = eigenvalues[j_virt];
                                let ij_virt_eigen = i_virt_eigen + j_virt_eigen;
                                let i_virt_occ = occupation.get(i_virt).unwrap();
                                let j_virt_occ = occupation.get(j_virt).unwrap();

                                if (1.0-i_virt_occ).abs() > 1.0e-6 && (1.0-j_virt_occ).abs() > 1.0e-6 {
                                    let mut double_gap = ij_virt_eigen - ij_state_eigen;
                                    if double_gap.abs()<=1.0E-6 {
                                        println!("Warning: too close to degeneracy");
                                        double_gap = 1.0e-6;
                                    };
                                    double_gap /= (i_state_occ*j_state_occ*(1.0-i_virt_occ)*(1.0-j_virt_occ));

                                    // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                                    // the indices in rimo are shifted
                                    let i_loc_virt = i_virt-vir_range.start;
                                    let j_loc_virt = j_virt-vir_range.start;
                                    let e_mp2_a = eri_virt.get2d([i_loc_virt,j_loc_virt]).unwrap();
                                    let e_mp2_b = eri_virt.get2d([j_loc_virt,i_loc_virt]).unwrap();
                                    e_mp2_term_ss += (e_mp2_a - e_mp2_b).powf(2.0) / double_gap;
                                }
                            }
                        }
                    }
                    s.send(e_mp2_term_ss).unwrap()
                });

                e_mp2_ss -= receiver.into_iter().sum::<f64>();

            } else {
                let eigenvector_1 = scf_data.eigenvectors.get(i_spin_1).unwrap();
                let eigenvalues_1 = scf_data.eigenvalues.get(i_spin_1).unwrap();
                let occupation_1 = scf_data.occupation.get(i_spin_1).unwrap();
                let homo_1 = scf_data.homo.get(i_spin_1).unwrap().clone();
                let lumo_1 = scf_data.lumo.get(i_spin_1).unwrap().clone();
                //let num_occu_1 = homo_1 + 1;
                //let num_occu_1 = lumo_1;
                let num_occu_1 = if scf_data.mol.num_elec[i_spin_1 + 1] <= 1.0e-6 {0} else {homo_1 + 1};
                let (rimo_1, vir_range, occ_range) = &ri3mo_vec[i_spin_1];

                let eigenvector_2 = scf_data.eigenvectors.get(i_spin_2).unwrap();
                let eigenvalues_2 = scf_data.eigenvalues.get(i_spin_2).unwrap();
                let occupation_2 = scf_data.occupation.get(i_spin_2).unwrap();
                let homo_2 = scf_data.homo.get(i_spin_2).unwrap().clone();
                let lumo_2 = scf_data.lumo.get(i_spin_2).unwrap().clone();
                //let num_occu_2 = homo_2 + 1;
                //let num_occu_2 = lumo_2;
                let num_occu_2 = if scf_data.mol.num_elec[i_spin_2 + 1] <= 1.0e-6 {0} else {homo_2 + 1};
                let (rimo_2, _, _) = &ri3mo_vec[i_spin_2];


                // prepare the elec_pair for the rayon parallelization
                let mut elec_pair: Vec<[usize;2]> = vec![];
                for i_state in start_mo..num_occu_1 {
                    for j_state in start_mo..num_occu_2 {
                        elec_pair.push([i_state,j_state])
                    }
                };
                let (sender, receiver) = channel();
                elec_pair.par_iter().for_each_with(sender,|s,i_pair| {
                    omp_set_num_threads_wrapper(1);
                    let mut e_mp2_term_os = 0.0_f64;
                    let i_state = i_pair[0];
                    let j_state = i_pair[1];
                    let i_state_eigen = eigenvalues_1.get(i_state).unwrap();
                    let j_state_eigen = eigenvalues_2.get(j_state).unwrap();
                    let ij_state_eigen = i_state_eigen + j_state_eigen;
                    let i_state_occ = occupation_1.get(i_state).unwrap();
                    let j_state_occ = occupation_2.get(j_state).unwrap();

                    if i_state_occ.abs() > 1.0e-6 && j_state_occ.abs() > 1.0e-6 {
                        // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                        // the indices in rimo are shifted
                        let i_loc_state = i_state-occ_range.start;
                        let j_loc_state = j_state-occ_range.start;
                        let ri_i = rimo_1.get_reducing_matrix(i_loc_state).unwrap();
                        let ri_j = rimo_2.get_reducing_matrix(j_loc_state).unwrap();
                        let mut eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                        _dgemm(
                            &ri_i, (0..num_auxbas,0..vir_range.len()), 'T', 
                            &ri_j,(0..num_auxbas,0..vir_range.len()) , 'N', 
                            &mut eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                            1.0,0.0);
                        //// ==== DEBUG IGOR ====
                        //if i_state == 0 && j_state== 0 {
                        //    eri_virt.formated_output(5, "full");
                        //}
                        // ==== DEBUG IGOR ====
                        for i_virt in lumo_1..num_state {
                            let i_virt_eigen = eigenvalues_1.get(i_virt).unwrap();
                            let i_virt_occ = occupation_1.get(i_virt).unwrap();
                            for j_virt in lumo_2..num_state {
                                let j_virt_eigen = eigenvalues_2.get(j_virt).unwrap();
                                let ij_virt_eigen = i_virt_eigen + j_virt_eigen;
                                let j_virt_occ = occupation_2.get(j_virt).unwrap();

                                //let mut test_value = 0.0; //DEBUG IGOR
                                if (1.0-i_virt_occ).abs() > 1.0e-6 && (1.0-j_virt_occ).abs() > 1.0e-6 {
                                    let mut double_gap = ij_virt_eigen - ij_state_eigen;
                                    if double_gap.abs()<=1.0E-6 {
                                        println!("Warning: too close to degeneracy")
                                    };
                                    double_gap /= (i_state_occ*j_state_occ*(1.0-i_virt_occ)*(1.0-j_virt_occ));

                                    // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                                    // the indices in rimo are shifted
                                    let i_loc_virt = i_virt-vir_range.start;
                                    let j_loc_virt = j_virt-vir_range.start;
                                    let e_mp2_a = eri_virt.get2d([i_loc_virt,j_loc_virt]).unwrap();
                                    e_mp2_term_os += e_mp2_a * e_mp2_a / double_gap;
                                    //test_value = e_mp2_a * e_mp2_a / double_gap; //DEBUG IGOR
                                }
                                //println!("debug e_mp2_term: {:16.8}, index: {} {}", test_value, i_virt, j_virt);
                            }
                        }
                    }
                    s.send(e_mp2_term_os).unwrap()
                });

                e_mp2_os -= receiver.into_iter().sum::<f64>();
            }
        }
    } else {
        panic!("RI3MO should be initialized before the PT2 calculations")
    };
    // reuse the default omp_num_threads setting
    omp_set_num_threads_wrapper(default_omp_num_threads);

    Ok([e_mp2_ss+e_mp2_os,e_mp2_os,e_mp2_ss])

}


pub fn open_shell_pt2_rayon_mpi(scf_data: &SCF, mpi_operator: &Option<MPIOperator>) -> anyhow::Result<[f64;3]> {

    #[cfg(feature = "mpi")]
    {
    let print_level = scf_data.mol.ctrl.print_level;

    if let (Some(mpi_op), Some(mpi_ix)) = (&mpi_operator, &scf_data.mol.mpi_data) {

        let num_threads = if let Some(nt) = scf_data.mol.ctrl.num_threads {nt} else {1};
        omp_set_num_threads_wrapper(num_threads);

        let mut e_mp2_ss = 0.0_f64;
        let mut e_mp2_os = 0.0_f64;

        let my_rank = mpi_ix.rank;
        let size = mpi_ix.size;
        let ran_auxbas_loc = if let Some(loc_auxbas) = &mpi_ix.auxbas {
            loc_auxbas[my_rank].clone()
        } else {
            panic!("Memory distrubtion should be initalized for the auxiliary basis sets before post-SCF calculations")
        };
        let num_auxbas_loc = ran_auxbas_loc.len();

        if let Some(ri3mo_vec) = &scf_data.ri3mo {

            let start_mo: usize = scf_data.mol.start_mo;
            let num_basis = scf_data.mol.num_basis;
            let num_state = scf_data.mol.num_state;
            //let num_auxbas = scf_data.mol.num_auxbas;
            let spin_channel = scf_data.mol.spin_channel;
            let i_spin_pair: [(usize,usize);3] = [(0,0),(0,1),(1,1)];
            

            for (i_spin_1,i_spin_2) in i_spin_pair {
                if i_spin_1 == i_spin_2 {

                    let i_spin = i_spin_1;
                    let eigenvector = scf_data.eigenvectors.get(i_spin).unwrap();
                    let eigenvalues = scf_data.eigenvalues.get(i_spin).unwrap();
                    let occupation = scf_data.occupation.get(i_spin).unwrap();

                    let homo = scf_data.homo.get(i_spin).unwrap().clone();
                    let lumo = scf_data.lumo.get(i_spin).unwrap().clone();
                    //let num_occu = homo + 1;
                    //let num_occu = lumo;
                    let num_occu = if scf_data.mol.num_elec[i_spin + 1] <= 1.0e-6 {0} else {homo + 1};

                    let (rimo, vir_range, occ_range) = &ri3mo_vec[i_spin];
                    let mut eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                    let mut loc_eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                    let virt_ss_pair = mpi_ix.distribution_same_spin_virtual_orbital_pair(lumo, num_state);

                    for i_state in start_mo..num_occu {
                        for j_state in i_state..num_occu {
                            let i_state_eigen = eigenvalues.get(i_state).unwrap();
                            let j_state_eigen = eigenvalues.get(j_state).unwrap();
                            let ij_state_eigen = i_state_eigen + j_state_eigen;
                            let i_state_occ = occupation.get(i_state).unwrap();
                            let j_state_occ = occupation.get(j_state).unwrap();

                            if i_state_occ.abs() > 1.0e-6 && j_state_occ.abs() > 1.0e-6 {
                                // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                                // the indices in rimo are shifted
                                let i_loc_state = i_state-occ_range.start;
                                let j_loc_state = j_state-occ_range.start;
                                let ri_i = rimo.get_reducing_matrix(i_loc_state).unwrap();
                                let ri_j = rimo.get_reducing_matrix(j_loc_state).unwrap();
                                //if print_level >= 2 {
                                //    println!("Debug: enter the preparation of eri_virt");
                                //}
                                //let mut eri_virt = {
                                    //let mut loc_eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                                _dgemm(
                                    &ri_i, (0..num_auxbas_loc,0..vir_range.len()), 'T', 
                                    &ri_j,(0..num_auxbas_loc,0..vir_range.len()) , 'N', 
                                    &mut loc_eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                                    1.0,0.0);
                                mpi_allreduce(&mpi_op.world, loc_eri_virt.data_ref().unwrap(), eri_virt.data_ref_mut().unwrap(), &SystemOperation::sum());
                                //let mut eri_virt = mpi_reduce(&mpi_op.world, &loc_eri_virt.data_ref().unwrap(), 0, &SystemOperation::sum());
                                //mpi_broadcast_vector(&mpi_op.world, &mut eri_virt, 0);
                                //MatrixFull::from_vec([vir_range.len(), vir_range.len()], eri_virt).unwrap()
                                //};
                                //if print_level >= 2 {
                                //    println!("Debug: leave the preparation of eri_virt");
                                //}
                                //// ==== DEBUG IGOR ====
                                //if i_state == 1 && j_state== 2 && my_rank == 0 {
                                //    eri_virt.formated_output(5, "full");
                                //}
                                //// ==== DEBUG IGOR ====

                                let (sender, receiver) = channel();
                                virt_ss_pair.par_iter().for_each_with(sender,|s,i_pair| {
                                    let mut e_mp2_term_ss = 0.0_f64;
                                    let i_virt = i_pair[0];
                                    let j_virt = i_pair[1];
                                    let i_virt_eigen = eigenvalues[i_virt];
                                    let j_virt_eigen = eigenvalues[j_virt];
                                    let ij_virt_eigen = i_virt_eigen + j_virt_eigen;
                                    let i_virt_occ = occupation.get(i_virt).unwrap();
                                    let j_virt_occ = occupation.get(j_virt).unwrap();

                                    if (1.0-i_virt_occ).abs() > 1.0e-6 && (1.0-j_virt_occ).abs() > 1.0e-6 {
                                        let mut double_gap = ij_virt_eigen - ij_state_eigen;
                                        if double_gap.abs()<=1.0E-6 {
                                            println!("Warning: too close to degeneracy");
                                            double_gap = 1.0e-6;
                                        };
                                        double_gap /= (i_state_occ*j_state_occ*(1.0-i_virt_occ)*(1.0-j_virt_occ));

                                        // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                                        // the indices in rimo are shifted
                                        let i_loc_virt = i_virt-vir_range.start;
                                        let j_loc_virt = j_virt-vir_range.start;
                                        let e_mp2_a = eri_virt.get2d([i_loc_virt,j_loc_virt]).unwrap();
                                        let e_mp2_b = eri_virt.get2d([j_loc_virt,i_loc_virt]).unwrap();
                                        e_mp2_term_ss += (e_mp2_a - e_mp2_b).powf(2.0) / double_gap;
                                    }
                                    s.send(e_mp2_term_ss).unwrap()
                                });
                                e_mp2_ss -= receiver.into_iter().sum::<f64>();
                                if print_level >= 2 {
                                    println!("Debug: ({},{}) with the same spin ({}) finishes ", i_state,j_state, i_spin_1);
                                }
                            }
                        }
                    }


                } else {
                    let eigenvector_1 = scf_data.eigenvectors.get(i_spin_1).unwrap();
                    let eigenvalues_1 = scf_data.eigenvalues.get(i_spin_1).unwrap();
                    let occupation_1 = scf_data.occupation.get(i_spin_1).unwrap();
                    let homo_1 = scf_data.homo.get(i_spin_1).unwrap().clone();
                    let lumo_1 = scf_data.lumo.get(i_spin_1).unwrap().clone();
                    //let num_occu_1 = homo_1 + 1;
                    //let num_occu_1 = lumo_1;
                    let num_occu_1 = if scf_data.mol.num_elec[i_spin_1 + 1] <= 1.0e-6 {0} else {homo_1 + 1};
                    let (rimo_1, vir_range, occ_range) = &ri3mo_vec[i_spin_1];

                    let eigenvector_2 = scf_data.eigenvectors.get(i_spin_2).unwrap();
                    let eigenvalues_2 = scf_data.eigenvalues.get(i_spin_2).unwrap();
                    let occupation_2 = scf_data.occupation.get(i_spin_2).unwrap();
                    let homo_2 = scf_data.homo.get(i_spin_2).unwrap().clone();
                    let lumo_2 = scf_data.lumo.get(i_spin_2).unwrap().clone();
                    //let num_occu_2 = homo_2 + 1;
                    //let num_occu_2 = lumo_2;
                    let num_occu_2 = if scf_data.mol.num_elec[i_spin_2 + 1] <= 1.0e-6 {0} else {homo_2 + 1};
                    let (rimo_2, _, _) = &ri3mo_vec[i_spin_2];

                    let mut eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                    let mut loc_eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                    let virt_os_pair = mpi_ix.distribution_opposite_spin_virtual_orbital_pair(lumo_1, lumo_2, num_state, scf_data.mol.ctrl.ri_pt2.mpi_mode);


                    // prepare the elec_pair for the rayon parallelization
                    //let mut elec_pair: Vec<[usize;2]> = vec![];
                    //for i_state in start_mo..num_occu_1 {
                    //    for j_state in start_mo..num_occu_2 {
                    //        elec_pair.push([i_state,j_state])
                    //    }
                    //};
                    //let (sender, receiver) = channel();
                    //elec_pair.par_iter().for_each_with(sender,|s,i_pair| {
                    for i_state in start_mo..num_occu_1 {
                        for j_state in start_mo..num_occu_2 {
                            //let i_state = i_pair[0];
                            //let j_state = i_pair[1];
                            let i_state_eigen = eigenvalues_1.get(i_state).unwrap();
                            let j_state_eigen = eigenvalues_2.get(j_state).unwrap();
                            let ij_state_eigen = i_state_eigen + j_state_eigen;
                            let i_state_occ = occupation_1.get(i_state).unwrap();
                            let j_state_occ = occupation_2.get(j_state).unwrap();

                            if i_state_occ.abs() > 1.0e-6 && j_state_occ.abs() > 1.0e-6 {
                                // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                                // the indices in rimo are shifted
                                let i_loc_state = i_state-occ_range.start;
                                let j_loc_state = j_state-occ_range.start;
                                let ri_i = rimo_1.get_reducing_matrix(i_loc_state).unwrap();
                                let ri_j = rimo_2.get_reducing_matrix(j_loc_state).unwrap();
                                //let mut eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                                //_dgemm(
                                //    &ri_i, (0..num_auxbas,0..vir_range.len()), 'T', 
                                //    &ri_j,(0..num_auxbas,0..vir_range.len()) , 'N', 
                                //    &mut eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                                //    1.0,0.0);
                                //if print_level >= 2 {
                                //    println!("Debug: enter the preparation of eri_virt");
                                //}
                                //let mut eri_virt = {
                                //    let mut loc_eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                                //    _dgemm(
                                //        &ri_i, (0..num_auxbas_loc,0..vir_range.len()), 'T', 
                                //        &ri_j,(0..num_auxbas_loc,0..vir_range.len()) , 'N', 
                                //        &mut loc_eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                                //        1.0,0.0);
                                //    let mut eri_virt = mpi_reduce(&mpi_op.world, &loc_eri_virt.data_ref().unwrap(), 0, &SystemOperation::sum());
                                //    mpi_broadcast(&mpi_op.world, &mut eri_virt, 0);
                                //    MatrixFull::from_vec([vir_range.len(), vir_range.len()], eri_virt).unwrap()
                                //};
                                //if print_level >= 2 {
                                //    println!("Debug: leave the preparation of eri_virt");
                                //}
                                _dgemm(
                                    &ri_i, (0..num_auxbas_loc,0..vir_range.len()), 'T', 
                                    &ri_j,(0..num_auxbas_loc,0..vir_range.len()) , 'N', 
                                    &mut loc_eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                                    1.0,0.0);
                                mpi_allreduce(&mpi_op.world, loc_eri_virt.data_ref().unwrap(), eri_virt.data_ref_mut().unwrap(), &SystemOperation::sum());
                                //// ==== DEBUG IGOR ====
                                //if i_state == 0 && j_state== 0 && my_rank == 0 {
                                //    println!("my rank = {}", my_rank);
                                //    eri_virt.formated_output(5, "full");
                                //}
                                //if i_state == 0 && j_state== 0 && my_rank == 1 {
                                //    println!("my rank = {}", my_rank);
                                //    eri_virt.formated_output(5, "full");
                                //}
                                //// ==== DEBUG IGOR ====
                                let (sender, receiver) = channel();
                                virt_os_pair.par_iter().for_each_with(sender,|s,i_pair| {
                                    let mut e_mp2_term_os = 0.0_f64;
                                    let i_virt = i_pair[0];
                                    let j_virt = i_pair[1];
                                    let i_virt_eigen = eigenvalues_1.get(i_virt).unwrap();
                                    let i_virt_occ = occupation_1.get(i_virt).unwrap();
                                    let j_virt_eigen = eigenvalues_2.get(j_virt).unwrap();
                                    let ij_virt_eigen = i_virt_eigen + j_virt_eigen;
                                    let j_virt_occ = occupation_2.get(j_virt).unwrap();

                                    if (1.0-i_virt_occ).abs() > 1.0e-6 && (1.0-j_virt_occ).abs() > 1.0e-6 {
                                        let mut double_gap = ij_virt_eigen - ij_state_eigen;
                                        if double_gap.abs()<=1.0E-6 {
                                            println!("Warning: too close to degeneracy")
                                        };
                                        double_gap /= (i_state_occ*j_state_occ*(1.0-i_virt_occ)*(1.0-j_virt_occ));

                                        // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                                        // the indices in rimo are shifted
                                        let i_loc_virt = i_virt-vir_range.start;
                                        let j_loc_virt = j_virt-vir_range.start;
                                        let e_mp2_a = eri_virt.get2d([i_loc_virt,j_loc_virt]).unwrap();
                                        e_mp2_term_os += e_mp2_a * e_mp2_a / double_gap;
                                    } //println!("debug e_mp2_term: {:16.8}, index: {:?}", e_mp2_term_os, i_pair);
                                    s.send(e_mp2_term_os).unwrap()
                                    
                                });
                                e_mp2_os -= receiver.into_iter().sum::<f64>();
                                if print_level >= 2 {
                                    println!("Debug: ({},{}) with the opposite spin ({},{}) finishes ", i_state,j_state, i_spin_1, i_spin_2);
                                }
                            }
                        }
                    };
                }
            }
        } else {
            panic!("RI3MO should be initialized before the PT2 calculations")
        };
        // reuse the default omp_num_threads setting
        //utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

        //// sum up the ss and os contribution from the mpi tasks.
        let mut e_mp2_ss = mpi_reduce(&mpi_op.world, &mut [e_mp2_ss], 0, &SystemOperation::sum())[0];
        mpi_broadcast(&mpi_op.world, &mut e_mp2_ss, 0);
        let mut e_mp2_os = mpi_reduce(&mpi_op.world, &mut [e_mp2_os], 0, &SystemOperation::sum())[0];
        mpi_broadcast(&mpi_op.world, &mut e_mp2_os, 0);
        Ok([e_mp2_ss+e_mp2_os,e_mp2_os,e_mp2_ss])
    } else {
        open_shell_pt2_rayon(scf_data)
    }
    }
    #[cfg(not(feature = "mpi"))]
    { open_shell_pt2_rayon(scf_data) }

}

pub fn close_shell_pt2_rayon_mpi(scf_data: &SCF, mpi_operator: &Option<MPIOperator>) -> anyhow::Result<[f64;3]> {
    #[cfg(feature = "mpi")]
    if let (Some(mpi_op), Some(mpi_ix)) = (&mpi_operator, &scf_data.mol.mpi_data) {
        // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
        // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
        //let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
        let num_threads = if let Some(num_threads) = scf_data.mol.ctrl.num_threads {num_threads} else {1};
        omp_set_num_threads_wrapper(num_threads);
        let mut e_mp2_ss = 0.0_f64;
        let mut e_mp2_os = 0.0_f64;

        let my_rank = mpi_ix.rank;
        let size = mpi_ix.size;
        let ran_auxbas_loc = if let Some(loc_auxbas) = &mpi_ix.auxbas {
            loc_auxbas[my_rank].clone()
        } else {
            panic!("Memory distrubtion should be initalized for the auxiliary basis sets before post-SCF calculations")
        };

        let num_auxbas_loc = ran_auxbas_loc.len();

        if let Some(ri3mo_vec) = &scf_data.ri3mo {

            let (rimo, vir_range, occ_range) = &ri3mo_vec[0];

            let mut eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
            let mut loc_eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);

            let eigenvector = scf_data.eigenvectors.get(0).unwrap();
            let eigenvalues = scf_data.eigenvalues.get(0).unwrap();
            let occupation = scf_data.occupation.get(0).unwrap();

            let homo = scf_data.homo.get(0).unwrap().clone();
            let lumo = scf_data.lumo.get(0).unwrap().clone();
            let num_basis = eigenvector.size.get(0).unwrap().clone();
            //let num_auxbas = rimo.size[0];
            let num_state = eigenvector.size.get(1).unwrap().clone();
            let start_mo: usize = scf_data.mol.start_mo;
            //let num_occu = homo + 1;
            //let num_occu = lumo;
            let num_occu = if scf_data.mol.num_elec[0] <= 1.0e-6 {0} else {homo + 1};
            //let (sender, receiver) = channel();
            //elec_pair.par_iter().for_each_with(sender,|s,i_pair| {

            let virt_os_pair = mpi_ix.distribution_opposite_spin_virtual_orbital_pair(lumo, lumo, num_state, scf_data.mol.ctrl.ri_pt2.mpi_mode);

            for i_state in start_mo..num_occu {
                for j_state in i_state..num_occu {

                    let i_state_eigen = eigenvalues.get(i_state).unwrap();
                    let j_state_eigen = eigenvalues.get(j_state).unwrap();
                    let ij_state_eigen = i_state_eigen + j_state_eigen;
                    let i_state_occ = occupation.get(i_state).unwrap()/2.0;
                    let j_state_occ = occupation.get(j_state).unwrap()/2.0;

                    // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                    // the indices in rimo are shifted

                    if i_state_occ.abs() > 1.0e-6 && j_state_occ.abs() > 1.0e-6 {
                        let i_loc_state = i_state-occ_range.start;
                        let j_loc_state = j_state-occ_range.start;
                        let ri_i = rimo.get_reducing_matrix(i_loc_state).unwrap();
                        let ri_j = rimo.get_reducing_matrix(j_loc_state).unwrap();
                        //let mut eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                        //_dgemm(
                        //    &ri_i, (0..num_auxbas_loc,0..vir_range.len()), 'T', 
                        //    &ri_j,(0..num_auxbas_loc,0..vir_range.len()) , 'N', 
                        //    &mut eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                        //    1.0,0.0);
                        //let mut eri_virt = {
                        //    let mut loc_eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                        //    _dgemm(
                        //        &ri_i, (0..num_auxbas_loc,0..vir_range.len()), 'T', 
                        //        &ri_j,(0..num_auxbas_loc,0..vir_range.len()) , 'N', 
                        //        &mut loc_eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                        //        1.0,0.0);
                        //    let mut eri_virt = mpi_reduce(&mpi_op.world, &loc_eri_virt.data_ref().unwrap(), 0, &SystemOperation::sum());
                        //    mpi_broadcast_vector(&mpi_op.world, &mut eri_virt, 0);
                        //    MatrixFull::from_vec([vir_range.len(), vir_range.len()], eri_virt).unwrap()
                        //};
                        _dgemm(
                            &ri_i, (0..num_auxbas_loc,0..vir_range.len()), 'T', 
                            &ri_j,(0..num_auxbas_loc,0..vir_range.len()) , 'N', 
                            &mut loc_eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                            1.0,0.0);
                        mpi_allreduce(&mpi_op.world, loc_eri_virt.data_ref().unwrap(), eri_virt.data_ref_mut().unwrap(), &SystemOperation::sum());
                        let (sender, receiver) = channel();
                        virt_os_pair.par_iter().for_each_with(sender,|s,i_pair| {
                            let mut e_mp2_term_ss = 0.0_f64;
                            let mut e_mp2_term_os = 0.0_f64;
                            let i_virt = i_pair[0];
                            let j_virt = i_pair[1];
                            let i_virt_eigen = eigenvalues.get(i_virt).unwrap();
                            let i_virt_occ = occupation.get(i_virt).unwrap()/2.0;
                            let j_virt_occ = occupation.get(j_virt).unwrap()/2.0;
                            if (1.0-i_virt_occ).abs() > 1.0e-6 && (1.0-j_virt_occ).abs() > 1.0e-6 {
                                let j_virt_eigen = eigenvalues.get(j_virt).unwrap();
                                let ij_virt_eigen = i_virt_eigen + j_virt_eigen;

                                let mut double_gap = (ij_virt_eigen - ij_state_eigen);
                                if double_gap.abs()<=1.0E-6 {
                                    println!("Warning: too close to degeneracy");
                                    double_gap = 1.0e-6;
                                };
                                double_gap /= (i_state_occ*j_state_occ*(1.0-i_virt_occ)*(1.0-j_virt_occ));

                                // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                                // the indices in rimo are shifted
                                let i_loc_virt = i_virt-vir_range.start;
                                let j_loc_virt = j_virt-vir_range.start;

                                let e_mp2_a = eri_virt.get2d([i_loc_virt,j_loc_virt]).unwrap();
                                let e_mp2_b = eri_virt.get2d([j_loc_virt,i_loc_virt]).unwrap();
                                e_mp2_term_ss += (e_mp2_a - e_mp2_b) * e_mp2_a / double_gap;
                                e_mp2_term_os += e_mp2_a * e_mp2_a / double_gap;
                            }
                            if i_state != j_state {
                                e_mp2_term_ss *= 2.0;
                                e_mp2_term_os *= 2.0;
                            }
                            s.send((e_mp2_term_os, e_mp2_term_ss)).unwrap()
                        });
                        receiver.into_iter().for_each(|(e_mp2_term_os,e_mp2_term_ss)| {
                            e_mp2_os -= e_mp2_term_os;
                            e_mp2_ss -= e_mp2_term_ss;
                        });
                    }
                    //s.send((e_mp2_term_os, e_mp2_term_ss)).unwrap()
                }
            };
        } else {
            panic!("RI3MO should be initialized before the PT2 calculations")
        };

        //// reuse the default omp_num_threads setting
        //utilities::omp_set_num_threads_wrapper(default_omp_num_threads);
        ////tmp_record.report_all();

        // sum up the ss and os contribution from the mpi tasks.
        let mut e_mp2_ss = mpi_reduce(&mpi_op.world, &mut [e_mp2_ss], 0, &SystemOperation::sum())[0];
        mpi_broadcast(&mpi_op.world, &mut e_mp2_ss, 0);
        let mut e_mp2_os = mpi_reduce(&mpi_op.world, &mut [e_mp2_os], 0, &SystemOperation::sum())[0];
        mpi_broadcast(&mpi_op.world, &mut e_mp2_os, 0);

        Ok([e_mp2_ss+e_mp2_os,e_mp2_os,e_mp2_ss])
    } else {
        close_shell_pt2_rayon(scf_data)
    }
    #[cfg(not(feature = "mpi"))]
    { close_shell_pt2_rayon(scf_data) }

}

#[cfg(feature = "mpi")]
fn check_conditions_25d(
    scf_data: &SCF,
    mpi_operator: &Option<MPIOperator>,
    dfa_family_pos: &crate::dft::DFAFamily,
) -> bool {
    let mut use_25d = true;

    let (mpi_op, mpi_ix) = match (&mpi_operator, &scf_data.mol.mpi_data) {
        (Some(op), Some(ix)) => (op, ix),
        _ => return false,
    };

    let my_rank = mpi_ix.rank;

    let local_n0_range = match &mpi_ix.auxbas {
        Some(loc_auxbas) => loc_auxbas[my_rank].clone(),
        None => {
            eprintln!("Auxiliary basis distribution not initialized");
            return false;
        }
    };

    let grid = mpi_op.initialize_grid();

    let ri3mo_vec = match &scf_data.ri3mo {
        Some(vec) => vec,
        None => {
            eprintln!("RI3MO not initialized, fallback to primitive PT2");
            return false;
        }
    };

    let (rimo, vir_range, occ_range) = &ri3mo_vec[0];
    let n0_local = rimo.size[0];
    let n1_global = rimo.size[1];
    let n2_global = rimo.size[2];

    if n0_local != local_n0_range.len()
        || n1_global != vir_range.len()
        || n2_global != occ_range.len()
    {
        eprintln!("Inconsistent RI3MO dimensions, fallback to primitive PT2");
        return false;
    }

    // The n_occ >= 20 preference exists so that small systems take the cheaper
    // 1D MPI path. Families WITHOUT a 1D MPI implementation (SCSRPA) must take
    // the 2.5D path at any size — the 1D dispatch is a panic for them.
    let has_1d_fallback = match dfa_family_pos {
        crate::dft::DFAFamily::PT2 | crate::dft::DFAFamily::SBGE2 => true,
        crate::dft::DFAFamily::SCSRPA => false,
        _ => false,
    };

    if has_1d_fallback && n2_global < 20 {
        eprintln!("num_occ smaller than 20, fallback to primitive PT2");
        use_25d = false;
    }

    if !has_1d_fallback {
        // Block-partition feasibility for the linear ownership path
        // (initialize_metadata_linear, k floor 1): num_block = k·P <= n_occ requires
        // n_occ >= P (P = cart rank count). If violated, error clearly instead of
        // tripping the assert deep inside.
        let P = grid.cart_comm.size() as usize;
        if n2_global < P {
            panic!(
                "The 2.5D MPI path is the only MPI path for this post-SCF family \
                 (DFAFamily::{:?}), but the system is too small for the current rank \
                 count: n_occ = {} < {} ranks. Reduce the MPI ranks or run without MPI.",
                dfa_family_pos, n2_global, P
            );
        }
    }

    if use_25d {
        let mut n0_global_tmp: u64 = 0;
        grid.cart_comm.all_reduce_into( &(n0_local as u64), &mut n0_global_tmp, &SystemOperation::sum());
        let n0_global = n0_global_tmp as usize;

        // Memory gate must mirror the redistribution the driver will actually use:
        // pair kernels (PT2/SBGE2) run initialize_metadata + swap_ownership
        // (row∪col, ≈ 2·M/√p); single-index kernels (SCSRPA/RPA) run
        // initialize_metadata_linear + redistribute_to_diag (≈ M/p).
        let ctx = if has_1d_fallback {
            initialize_metadata(&grid, n2_global)
        } else {
            initialize_metadata_linear(&grid, n2_global)
        };
        // Tensor part: estimated inside check_memory_25d from the ctx slice counts.
        let tensor_ok = check_memory_25d(&grid, &ctx, n0_global * n1_global, n2_global);
        // Response scratch part (replicated on every rank, independent of p):
        // partial + reduced + polar + integrand temporaries. dRPA ≈ 3-4×naux²;
        // SCSRPA open-shell λ-integration/series branch ≈ 6-8×naux² (conservative).
        let scratch_ok = if has_1d_fallback {
            true
        } else {
            let n_chi_sq = match dfa_family_pos {
                crate::dft::DFAFamily::RPA => 4,
                _ => 8, // SCSRPA 保守估计（含 OS 分支临时矩阵）
            };
            let required_bytes = n_chi_sq * n0_global * n0_global * std::mem::size_of::<f64>();
            let avail_bytes = detect_available_memory_mb() * 1024.0 * 1024.0;
            let local_ok = 4.0 * (required_bytes as f64) < (avail_bytes as f64);
            let mut global_ok = true;
            grid.cart_comm.all_reduce_into(&local_ok, &mut global_ok, &SystemOperation::logical_and());
            global_ok
        };
        if !tensor_ok {
            eprintln!("Not enough memory for 2.5D, fallback to primitive PT2");
            use_25d = false;
        }
        if !scratch_ok {
            eprintln!("Not enough memory for the replicated response matrices of the 2.5D path");
            use_25d = false;
        }
    }

    let mut global_use_25d = false;
    grid.cart_comm.all_reduce_into(
        &use_25d,
        &mut global_use_25d,
        &SystemOperation::logical_and(),
    );

    global_use_25d
}

#[cfg(feature = "mpi")]
pub fn open_shell_pt2_rayon_mpi_25d(scf_data: &SCF, mpi_operator: &Option<MPIOperator>) -> anyhow::Result<[f64;3]> {
    if let (Some(mpi_op), Some(mpi_ix)) = (&mpi_operator, &scf_data.mol.mpi_data)  {

        let my_rank = mpi_ix.rank;
        let size = mpi_ix.size;
        let local_n0_range = if let Some(loc_auxbas) = &mpi_ix.auxbas {
            loc_auxbas[my_rank].clone()
        } else {
            panic!("Memory distrubtion should be initalized for the auxiliary basis sets before post-SCF calculations")
        };

        let grid = mpi_op.initialize_grid();
        let mut global_term_os:f64 = 0.0_f64;
        let mut global_term_ss:f64 = 0.0_f64;

        let start_mo: usize = scf_data.mol.start_mo;
        let num_basis = scf_data.mol.num_basis;
        let num_state = scf_data.mol.num_state;

        if let Some(ri3mo_vec) = &scf_data.ri3mo {

            let alpha_eigenvalues = scf_data.eigenvalues.get(0).unwrap();
            let alpha_occupation = scf_data.occupation.get(0).unwrap();

            let alpha_homo = scf_data.homo.get(0).unwrap().clone();
            let alpha_lumo = scf_data.lumo.get(0).unwrap().clone();
            let alpha_num_occu = if scf_data.mol.num_elec[0 + 1] <= 1.0e-6 {0} else {alpha_homo + 1};

            let beta_eigenvalues = scf_data.eigenvalues.get(1).unwrap();
            let beta_occupation = scf_data.occupation.get(1).unwrap();

            let beta_homo = scf_data.homo.get(1).unwrap().clone();
            let beta_lumo = scf_data.lumo.get(1).unwrap().clone();
            let beta_num_occu = if scf_data.mol.num_elec[1 + 1] <= 1.0e-6 {0} else {beta_homo + 1};

            let (alpha_rimo, vir_range, occ_range) = &ri3mo_vec[0];
            let (beta_rimo, _, _) = &ri3mo_vec[1];

            let n0_local = alpha_rimo.size[0];
            let n1_global = alpha_rimo.size[1];
            let n2_global = alpha_rimo.size[2];

            let ctx = initialize_metadata(&grid, n2_global);
            let mut n0_global_tmp: u64 = 0;
            grid.cart_comm.all_reduce_into(&(n0_local as u64), &mut n0_global_tmp, &SystemOperation::sum());
            let n0_global = n0_global_tmp as usize;

            let redistributed_alpha_rimo = swap_ownership(&grid, &ctx, &alpha_rimo, n0_global, n1_global, n2_global, &local_n0_range);
            let redistributed_beta_rimo = swap_ownership(&grid, &ctx, &beta_rimo, n0_global, n1_global, n2_global, &local_n0_range);
            let term_aa = local_computation_ss_batch(&redistributed_alpha_rimo, &ctx, n2_global, alpha_eigenvalues, alpha_occupation, occ_range.start, vir_range.start, alpha_num_occu);
            let term_bb = local_computation_ss_batch(&redistributed_beta_rimo, &ctx, n2_global, beta_eigenvalues, beta_occupation, occ_range.start, vir_range.start, beta_num_occu);
            let local_term_ss = term_aa + term_bb;
            let local_term_os = local_computation_os_batch(&redistributed_alpha_rimo, &redistributed_beta_rimo, &ctx, n2_global,
                alpha_eigenvalues, beta_eigenvalues, alpha_occupation, beta_occupation, occ_range.start, vir_range.start, alpha_num_occu, beta_num_occu);

            grid.cart_comm.all_reduce_into(&local_term_ss, &mut global_term_ss, &SystemOperation::sum());
            grid.cart_comm.all_reduce_into(&local_term_os, &mut global_term_os, &SystemOperation::sum());

        }
        Ok([global_term_os + global_term_ss, global_term_os, global_term_ss])
    }
    else {
        panic!("MPI not initialized for 2.5d");
    }
}

#[cfg(feature = "mpi")]
pub fn restricted_open_shell_pt2_rayon_mpi_25d(scf_data: &SCF, mpi_operator: &Option<MPIOperator>) -> anyhow::Result<[f64;3]> {
    if let (Some(mpi_op), Some(mpi_ix)) = (&mpi_operator, &scf_data.mol.mpi_data)  {

        let my_rank = mpi_ix.rank;
        let size = mpi_ix.size;
        let local_n0_range = if let Some(loc_auxbas) = &mpi_ix.auxbas {
            loc_auxbas[my_rank].clone()
        } else {
            panic!("Memory distrubtion should be initalized for the auxiliary basis sets before post-SCF calculations")
        };

        let grid = mpi_op.initialize_grid();
        let mut global_term_os:f64 = 0.0_f64;
        let mut global_term_ss:f64 = 0.0_f64;

        let start_mo: usize = scf_data.mol.start_mo;
        let num_basis = scf_data.mol.num_basis;
        let num_state = scf_data.mol.num_state;

        if let Some(ri3mo_vec) = &scf_data.ri3mo {

            let alpha_eigenvalues = &scf_data.semi_eigenvalues.as_ref().unwrap()[0];
            let alpha_occupation = scf_data.occupation.get(0).unwrap();

            let alpha_homo = scf_data.homo.get(0).unwrap().clone();
            let alpha_lumo = scf_data.lumo.get(0).unwrap().clone();
            let alpha_num_occu = if scf_data.mol.num_elec[0 + 1] <= 1.0e-6 {0} else {alpha_homo + 1};

            let beta_eigenvalues = &scf_data.semi_eigenvalues.as_ref().unwrap()[1];
            let beta_occupation = scf_data.occupation.get(1).unwrap();

            let beta_homo = scf_data.homo.get(1).unwrap().clone();
            let beta_lumo = scf_data.lumo.get(1).unwrap().clone();
            let beta_num_occu = if scf_data.mol.num_elec[1 + 1] <= 1.0e-6 {0} else {beta_homo + 1};

            let (alpha_rimo, vir_range, occ_range) = &ri3mo_vec[0];
            let (beta_rimo, _, _) = &ri3mo_vec[1];

            let n0_local = alpha_rimo.size[0];
            let n1_global = alpha_rimo.size[1];
            let n2_global = alpha_rimo.size[2];

            let ctx = initialize_metadata(&grid, n2_global);
            let mut n0_global_tmp: u64 = 0;
            grid.cart_comm.all_reduce_into(&(n0_local as u64), &mut n0_global_tmp, &SystemOperation::sum());
            let n0_global = n0_global_tmp as usize;

            let redistributed_alpha_rimo = swap_ownership(&grid, &ctx, &alpha_rimo, n0_global, n1_global, n2_global, &local_n0_range);
            let redistributed_beta_rimo = swap_ownership(&grid, &ctx, &beta_rimo, n0_global, n1_global, n2_global, &local_n0_range);
            let term_aa = local_computation_ss_batch(&redistributed_alpha_rimo, &ctx, n2_global, alpha_eigenvalues, alpha_occupation, occ_range.start, vir_range.start, alpha_num_occu);
            let term_bb = local_computation_ss_batch(&redistributed_beta_rimo, &ctx, n2_global, beta_eigenvalues, beta_occupation, occ_range.start, vir_range.start, beta_num_occu);
            let local_term_ss = term_aa + term_bb;
            let local_term_os = local_computation_os_batch(&redistributed_alpha_rimo, &redistributed_beta_rimo, &ctx, n2_global,
                alpha_eigenvalues, beta_eigenvalues, alpha_occupation, beta_occupation, occ_range.start, vir_range.start, alpha_num_occu, beta_num_occu);

            grid.cart_comm.all_reduce_into(&local_term_ss, &mut global_term_ss, &SystemOperation::sum());
            grid.cart_comm.all_reduce_into(&local_term_os, &mut global_term_os, &SystemOperation::sum());

        }
        Ok([global_term_os + global_term_ss, global_term_os, global_term_ss])
    }
    else {
        panic!("MPI not initialized for 2.5d");
    }
}

#[cfg(feature = "mpi")]
pub fn close_shell_pt2_rayon_mpi_25d(scf_data: &SCF, mpi_operator: &Option<MPIOperator>) -> anyhow::Result<[f64;3]> {
    if let (Some(mpi_op), Some(mpi_ix)) = (&mpi_operator, &scf_data.mol.mpi_data)  {

        let my_rank = mpi_ix.rank;
        let size = mpi_ix.size;
        let local_n0_range = if let Some(loc_auxbas) = &mpi_ix.auxbas {
            loc_auxbas[my_rank].clone()
        } else {
            panic!("Memory distrubtion should be initalized for the auxiliary basis sets before post-SCF calculations")
        };

        let grid = mpi_op.initialize_grid();
        let mut global_term_os:f64 = 0.0_f64;
        let mut global_term_ss:f64 = 0.0_f64;
        if let Some(ri3mo_vec) = &scf_data.ri3mo {

            let eigenvector = scf_data.eigenvectors.get(0).unwrap();
            let eigenvalues = scf_data.eigenvalues.get(0).unwrap();
            let occupation = scf_data.occupation.get(0).unwrap();

            let homo = scf_data.homo.get(0).unwrap().clone();
            let lumo = scf_data.lumo.get(0).unwrap().clone();
            let num_basis = eigenvector.size.get(0).unwrap().clone();

            let num_state = eigenvector.size.get(1).unwrap().clone();
            let start_mo: usize = scf_data.mol.start_mo;

            let num_occu = if scf_data.mol.num_elec[0] <= 1.0e-6 {0} else {homo + 1};
            let (rimo, vir_range, occ_range) = &ri3mo_vec[0];
            let n0_local = rimo.size[0];
            let n1_global = rimo.size[1];
            let n2_global = rimo.size[2];

            //println!("rank:{}, cart_rank:{}, tensor_size=({},{},{}), local_n0_range:{:?})",
                //my_rank, grid.rank, n0_local, n1_global, n2_global, local_n0_range);

            let ctx = initialize_metadata(&grid, n2_global);
            let mut n0_global_tmp: u64 = 0;
            grid.cart_comm.all_reduce_into(&(n0_local as u64), &mut n0_global_tmp, &SystemOperation::sum());
            let n0_global = n0_global_tmp as usize;

            let redistributed_rimo = swap_ownership(&grid, &ctx, &rimo, n0_global, n1_global, n2_global, &local_n0_range);
            let (local_term_os, local_term_ss) = local_computation_close_shell_batch(&redistributed_rimo, &ctx, n2_global, eigenvalues, occupation, occ_range.start, vir_range.start);
            grid.cart_comm.all_reduce_into(&local_term_os, &mut global_term_os, &SystemOperation::sum());
            grid.cart_comm.all_reduce_into(&local_term_ss, &mut global_term_ss, &SystemOperation::sum());
        }
        Ok([global_term_os + global_term_ss, global_term_os, global_term_ss])
    }
    else {
        panic!("MPI not initialized for 2.5d");
    }
}

pub fn restricted_open_shell_pt2_rayon(scf_data: &SCF) -> anyhow::Result<[f64;3]> {
    let default_omp_num_threads = scf_data.mol.ctrl.num_threads.unwrap();

    // Calculate the contribution of singly excited states.
    let mut e_mp2_single_list = [0.0_f64, 0.0_f64];


    for i_spin in (0..2) {
        let eigenvalues_spin = &scf_data.semi_eigenvalues.as_ref().unwrap()[i_spin];
        let fock_spin = &scf_data.semi_fock.as_ref().unwrap()[i_spin];
        for i_occ in (0..scf_data.lumo[i_spin]) {
            for i_virt in (scf_data.lumo[i_spin]..scf_data.mol.num_state) {
                let single_gap = eigenvalues_spin[i_virt] - eigenvalues_spin[i_occ];
                e_mp2_single_list[i_spin] += - fock_spin[(i_virt,i_occ)].powf(2.0) / single_gap;
            }
        }
    }

    // ========================================

    let mut e_mp2_ss = 0.0_f64;
    let mut e_mp2_os = 0.0_f64;

    if let Some(ri3mo_vec) = &scf_data.ri3mo {
        
        let start_mo: usize = scf_data.mol.start_mo;
        let num_basis = scf_data.mol.num_basis;
        let num_state = scf_data.mol.num_state;
        let num_auxbas = scf_data.mol.num_auxbas;
        let spin_channel = scf_data.mol.spin_channel;
        let i_spin_pair: [(usize,usize);3] = [(0,0),(0,1),(1,1)];

        for (i_spin_1,i_spin_2) in i_spin_pair {
            if i_spin_1 == i_spin_2 {

                let i_spin = i_spin_1;
                let eigenvector = &scf_data.semi_eigenvectors.as_ref().unwrap()[i_spin];
                let eigenvalues = &scf_data.semi_eigenvalues.as_ref().unwrap()[i_spin];
                let occupation = scf_data.occupation.get(i_spin).unwrap();

                let homo = scf_data.homo.get(i_spin).unwrap().clone();
                let lumo = scf_data.lumo.get(i_spin).unwrap().clone();
                //let num_occu = homo + 1;
                //let num_occu = lumo;
                let num_occu = if scf_data.mol.num_elec[i_spin + 1] <= 1.0e-6 {0} else {homo + 1};

                let (rimo, vir_range, occ_range) = &ri3mo_vec[i_spin];

                //let mut rimo = riao.ao2mo(eigenvector).unwrap();
                let mut elec_pair: Vec<[usize;2]> = vec![];
                for i_state in start_mo..num_occu {
                    for j_state in i_state..num_occu {
                        elec_pair.push([i_state,j_state])
                    }
                };
                let (sender, receiver) = channel();
                elec_pair.par_iter().for_each_with(sender,|s,i_pair| {

                    omp_set_num_threads_wrapper(1);

                    let mut e_mp2_term_ss = 0.0_f64;

                    let i_state = i_pair[0];
                    let j_state = i_pair[1];
                    let i_state_eigen = eigenvalues.get(i_state).unwrap();
                    let j_state_eigen = eigenvalues.get(j_state).unwrap();
                    let ij_state_eigen = i_state_eigen + j_state_eigen;
                    let i_state_occ = occupation.get(i_state).unwrap();
                    let j_state_occ = occupation.get(j_state).unwrap();

                    if i_state_occ.abs() > 1.0e-6 && j_state_occ.abs() > 1.0e-6 {
                        // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                        // the indices in rimo are shifted
                        let i_loc_state = i_state-occ_range.start;
                        let j_loc_state = j_state-occ_range.start;
                        let ri_i = rimo.get_reducing_matrix(i_loc_state).unwrap();
                        let ri_j = rimo.get_reducing_matrix(j_loc_state).unwrap();
                        let mut eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                        _dgemm(
                            &ri_i, (0..num_auxbas,0..vir_range.len()), 'T', 
                            &ri_j,(0..num_auxbas,0..vir_range.len()) , 'N', 
                            &mut eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                            1.0,0.0);
                        //// ==== DEBUG IGOR ====
                        //if i_state == 1 && j_state== 2 {
                        //    eri_virt.formated_output(5, "full");
                        //}
                        //// ==== DEBUG IGOR ====
                        for i_virt in lumo..num_state {
                            let i_virt_eigen = eigenvalues[i_virt];
                            for j_virt in i_virt+1..num_state {
                                let j_virt_eigen = eigenvalues[j_virt];
                                let ij_virt_eigen = i_virt_eigen + j_virt_eigen;
                                let i_virt_occ = occupation.get(i_virt).unwrap();
                                let j_virt_occ = occupation.get(j_virt).unwrap();

                                if (1.0-i_virt_occ).abs() > 1.0e-6 && (1.0-j_virt_occ).abs() > 1.0e-6 {
                                    let mut double_gap = ij_virt_eigen - ij_state_eigen;
                                    if double_gap.abs()<=1.0E-6 {
                                        println!("Warning: too close to degeneracy");
                                        double_gap = 1.0e-6;
                                    };
                                    double_gap /= (i_state_occ*j_state_occ*(1.0-i_virt_occ)*(1.0-j_virt_occ));

                                    // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                                    // the indices in rimo are shifted
                                    let i_loc_virt = i_virt-vir_range.start;
                                    let j_loc_virt = j_virt-vir_range.start;
                                    let e_mp2_a = eri_virt.get2d([i_loc_virt,j_loc_virt]).unwrap();
                                    let e_mp2_b = eri_virt.get2d([j_loc_virt,i_loc_virt]).unwrap();
                                    e_mp2_term_ss += (e_mp2_a - e_mp2_b).powf(2.0) / double_gap;
                                }
                            }
                        }
                    }
                    s.send(e_mp2_term_ss).unwrap()
                });

                e_mp2_ss -= receiver.into_iter().sum::<f64>();

            } else {
                let eigenvector_1 = &scf_data.semi_eigenvectors.as_ref().unwrap()[i_spin_1];
                let eigenvalues_1 = &scf_data.semi_eigenvalues.as_ref().unwrap()[i_spin_1];
                let occupation_1 = scf_data.occupation.get(i_spin_1).unwrap();
                let homo_1 = scf_data.homo.get(i_spin_1).unwrap().clone();
                let lumo_1 = scf_data.lumo.get(i_spin_1).unwrap().clone();
                //let num_occu_1 = homo_1 + 1;
                //let num_occu_1 = lumo_1;
                let num_occu_1 = if scf_data.mol.num_elec[i_spin_1 + 1] <= 1.0e-6 {0} else {homo_1 + 1};
                let (rimo_1, vir_range, occ_range) = &ri3mo_vec[i_spin_1];

                let eigenvector_2 = &scf_data.semi_eigenvectors.as_ref().unwrap()[i_spin_2];
                let eigenvalues_2 = &scf_data.semi_eigenvalues.as_ref().unwrap()[i_spin_2];
                let occupation_2 = scf_data.occupation.get(i_spin_2).unwrap();
                let homo_2 = scf_data.homo.get(i_spin_2).unwrap().clone();
                let lumo_2 = scf_data.lumo.get(i_spin_2).unwrap().clone();
                //let num_occu_2 = homo_2 + 1;
                //let num_occu_2 = lumo_2;
                let num_occu_2 = if scf_data.mol.num_elec[i_spin_2 + 1] <= 1.0e-6 {0} else {homo_2 + 1};
                let (rimo_2, _, _) = &ri3mo_vec[i_spin_2];


                // prepare the elec_pair for the rayon parallelization
                let mut elec_pair: Vec<[usize;2]> = vec![];
                for i_state in start_mo..num_occu_1 {
                    for j_state in start_mo..num_occu_2 {
                        elec_pair.push([i_state,j_state])
                    }
                };
                let (sender, receiver) = channel();
                elec_pair.par_iter().for_each_with(sender,|s,i_pair| {
                    let mut e_mp2_term_os = 0.0_f64;
                    let i_state = i_pair[0];
                    let j_state = i_pair[1];
                    let i_state_eigen = eigenvalues_1.get(i_state).unwrap();
                    let j_state_eigen = eigenvalues_2.get(j_state).unwrap();
                    let ij_state_eigen = i_state_eigen + j_state_eigen;
                    let i_state_occ = occupation_1.get(i_state).unwrap();
                    let j_state_occ = occupation_2.get(j_state).unwrap();

                    if i_state_occ.abs() > 1.0e-6 && j_state_occ.abs() > 1.0e-6 {
                        // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                        // the indices in rimo are shifted
                        let i_loc_state = i_state-occ_range.start;
                        let j_loc_state = j_state-occ_range.start;
                        let ri_i = rimo_1.get_reducing_matrix(i_loc_state).unwrap();
                        let ri_j = rimo_2.get_reducing_matrix(j_loc_state).unwrap();
                        let mut eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                        _dgemm(
                            &ri_i, (0..num_auxbas,0..vir_range.len()), 'T', 
                            &ri_j,(0..num_auxbas,0..vir_range.len()) , 'N', 
                            &mut eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                            1.0,0.0);
                        //// ==== DEBUG IGOR ====
                        //if i_state == 0 && j_state== 0 {
                        //    eri_virt.formated_output(5, "full");
                        //}
                        // ==== DEBUG IGOR ====
                        for i_virt in lumo_1..num_state {
                            let i_virt_eigen = eigenvalues_1.get(i_virt).unwrap();
                            let i_virt_occ = occupation_1.get(i_virt).unwrap();
                            for j_virt in lumo_2..num_state {
                                let j_virt_eigen = eigenvalues_2.get(j_virt).unwrap();
                                let ij_virt_eigen = i_virt_eigen + j_virt_eigen;
                                let j_virt_occ = occupation_2.get(j_virt).unwrap();

                                //let mut test_value = 0.0; //DEBUG IGOR
                                if (1.0-i_virt_occ).abs() > 1.0e-6 && (1.0-j_virt_occ).abs() > 1.0e-6 {
                                    let mut double_gap = ij_virt_eigen - ij_state_eigen;
                                    if double_gap.abs()<=1.0E-6 {
                                        println!("Warning: too close to degeneracy")
                                    };
                                    double_gap /= (i_state_occ*j_state_occ*(1.0-i_virt_occ)*(1.0-j_virt_occ));

                                    // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                                    // the indices in rimo are shifted
                                    let i_loc_virt = i_virt-vir_range.start;
                                    let j_loc_virt = j_virt-vir_range.start;
                                    let e_mp2_a = eri_virt.get2d([i_loc_virt,j_loc_virt]).unwrap();
                                    e_mp2_term_os += e_mp2_a * e_mp2_a / double_gap;
                                    //test_value = e_mp2_a * e_mp2_a / double_gap; //DEBUG IGOR
                                }
                                //println!("debug e_mp2_term: {:16.8}, index: {} {}", test_value, i_virt, j_virt);
                            }
                        }
                    }
                    s.send(e_mp2_term_os).unwrap()
                });

                e_mp2_os -= receiver.into_iter().sum::<f64>();
            }
        }
    } else {
        panic!("RI3MO should be initialized before the PT2 calculations")
    };
    // reuse the default omp_num_threads setting
    omp_set_num_threads_wrapper(default_omp_num_threads);
    
    // Temporarily output the contribution of singly excited states here, to avoid modifying the pt2_c structure.
    if let Some(coeff) = &scf_data.mol.xc_data.dfa_paramr_adv {
        let hy_coeffi_pt2 = coeff.clone();
        println!("WARNING: The RO-PT2 method has been implemented.  
        The contribution parameter of singly excited states has been set to match 'para_mp2_ss': {}.  
        As a result, the contribution of singly excited states is {:16.8} * {} = {:16.8} Ha, with alpha: {:16.8} * {} = {:16.8} Ha, beta: {:16.8} * {} = {:16.8} Ha.  
        However, this contribution is not yet included in the final total energy.",  
    hy_coeffi_pt2[1],  
    e_mp2_single_list[0] + e_mp2_single_list[1], hy_coeffi_pt2[1], (e_mp2_single_list[0] + e_mp2_single_list[1]) * hy_coeffi_pt2[1],  
    e_mp2_single_list[0], hy_coeffi_pt2[1], e_mp2_single_list[0] * hy_coeffi_pt2[1],  
    e_mp2_single_list[1], hy_coeffi_pt2[1], e_mp2_single_list[1] * hy_coeffi_pt2[1]);  
    } else {
        println!("WARNING: The RO-PT2 method has been implemented.
        The contribution of singly excited states is {:16.8} Ha, with alpha: {:16.8} Ha, beta: {:16.8} Ha.
        However, this contribution is not yet included in the final total energy.",
        e_mp2_single_list[0]+e_mp2_single_list[1],
        e_mp2_single_list[0],
        e_mp2_single_list[1]);
    }
      
    Ok([e_mp2_ss+e_mp2_os+e_mp2_single_list[0]+e_mp2_single_list[1],e_mp2_os,e_mp2_ss])

}

fn restricted_open_shell_pt2_rayon_mpi(scf_data: &SCF, mpi_operator: &Option<MPIOperator>) -> anyhow::Result<[f64;3]> {
    #[cfg(feature = "mpi")]
    {
    let print_level = scf_data.mol.ctrl.print_level;
    
    if let (Some(mpi_op), Some(mpi_ix)) = (&mpi_operator, &scf_data.mol.mpi_data) {
        let num_threads = if let Some(nt) = scf_data.mol.ctrl.num_threads {nt} else {1};
        omp_set_num_threads_wrapper(num_threads);

        let mut e_mp2_ss = 0.0_f64;
        let mut e_mp2_os = 0.0_f64;

        // Contribution of singly excited states, identical to the serial
        // `restricted_open_shell_pt2_rayon`: -Σ_{i∈occ, a∈vir} F_{ai}^2 / (ε_a - ε_i).
        // The semi-canonical quantities are identical on all ranks.
        let mut e_mp2_single_list = [0.0_f64, 0.0_f64];
        for i_spin in (0..2) {
            let eigenvalues_spin = &scf_data.semi_eigenvalues.as_ref().unwrap()[i_spin];
            let fock_spin = &scf_data.semi_fock.as_ref().unwrap()[i_spin];
            for i_occ in (0..scf_data.lumo[i_spin]) {
                for i_virt in (scf_data.lumo[i_spin]..scf_data.mol.num_state) {
                    let single_gap = eigenvalues_spin[i_virt] - eigenvalues_spin[i_occ];
                    e_mp2_single_list[i_spin] += -fock_spin[(i_virt, i_occ)].powf(2.0) / single_gap;
                }
            }
        }

        let my_rank = mpi_ix.rank;
        let size = mpi_ix.size;
        let ran_auxbas_loc = if let Some(loc_auxbas) = &mpi_ix.auxbas {
            loc_auxbas[my_rank].clone()
        } else {
            panic!("Memory distrubtion should be initalized for the auxiliary basis sets before post-SCF calculations")
        };
        let num_auxbas_loc = ran_auxbas_loc.len();

        if let Some(ri3mo_vec) = &scf_data.ri3mo {
            
            let start_mo: usize = scf_data.mol.start_mo;
            let num_basis = scf_data.mol.num_basis;
            let num_state = scf_data.mol.num_state;
            //let num_auxbas = scf_data.mol.num_auxbas;
            let spin_channel = scf_data.mol.spin_channel;
            let i_spin_pair: [(usize,usize);3] = [(0,0),(0,1),(1,1)];
            

            for (i_spin_1,i_spin_2) in i_spin_pair {
                if i_spin_1 == i_spin_2 {

                    let i_spin = i_spin_1;
                    let eigenvector = &scf_data.semi_eigenvectors.as_ref().unwrap()[i_spin];
                    let eigenvalues = &scf_data.semi_eigenvalues.as_ref().unwrap()[i_spin];
                    let occupation = scf_data.occupation.get(i_spin).unwrap();

                    let homo = scf_data.homo.get(i_spin).unwrap().clone();
                    let lumo = scf_data.lumo.get(i_spin).unwrap().clone();
                    //let num_occu = homo + 1;
                    //let num_occu = lumo;
                    let num_occu = if scf_data.mol.num_elec[i_spin + 1] <= 1.0e-6 {0} else {homo + 1};

                    let (rimo, vir_range, occ_range) = &ri3mo_vec[i_spin];
                    let mut eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                    let mut loc_eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                    let virt_ss_pair = mpi_ix.distribution_same_spin_virtual_orbital_pair(lumo, num_state);

                    for i_state in start_mo..num_occu {
                        for j_state in i_state..num_occu {
                            let i_state_eigen = eigenvalues.get(i_state).unwrap();
                            let j_state_eigen = eigenvalues.get(j_state).unwrap();
                            let ij_state_eigen = i_state_eigen + j_state_eigen;
                            let i_state_occ = occupation.get(i_state).unwrap();
                            let j_state_occ = occupation.get(j_state).unwrap();

                            if i_state_occ.abs() > 1.0e-6 && j_state_occ.abs() > 1.0e-6 {
                                // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                                // the indices in rimo are shifted
                                let i_loc_state = i_state-occ_range.start;
                                let j_loc_state = j_state-occ_range.start;
                                let ri_i = rimo.get_reducing_matrix(i_loc_state).unwrap();
                                let ri_j = rimo.get_reducing_matrix(j_loc_state).unwrap();
                                //if print_level >= 2 {
                                //    println!("Debug: enter the preparation of eri_virt");
                                //}
                                //let mut eri_virt = {
                                    //let mut loc_eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                                _dgemm(
                                    &ri_i, (0..num_auxbas_loc,0..vir_range.len()), 'T', 
                                    &ri_j,(0..num_auxbas_loc,0..vir_range.len()) , 'N', 
                                    &mut loc_eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                                    1.0,0.0);
                                mpi_allreduce(&mpi_op.world, loc_eri_virt.data_ref().unwrap(), eri_virt.data_ref_mut().unwrap(), &SystemOperation::sum());
                                //let mut eri_virt = mpi_reduce(&mpi_op.world, &loc_eri_virt.data_ref().unwrap(), 0, &SystemOperation::sum());
                                //mpi_broadcast_vector(&mpi_op.world, &mut eri_virt, 0);
                                //MatrixFull::from_vec([vir_range.len(), vir_range.len()], eri_virt).unwrap()
                                //};
                                //if print_level >= 2 {
                                //    println!("Debug: leave the preparation of eri_virt");
                                //}
                                //// ==== DEBUG IGOR ====
                                //if i_state == 1 && j_state== 2 && my_rank == 0 {
                                //    eri_virt.formated_output(5, "full");
                                //}
                                //// ==== DEBUG IGOR ====

                                let (sender, receiver) = channel();
                                virt_ss_pair.par_iter().for_each_with(sender,|s,i_pair| {
                                    let mut e_mp2_term_ss = 0.0_f64;
                                    let i_virt = i_pair[0];
                                    let j_virt = i_pair[1];
                                    let i_virt_eigen = eigenvalues[i_virt];
                                    let j_virt_eigen = eigenvalues[j_virt];
                                    let ij_virt_eigen = i_virt_eigen + j_virt_eigen;
                                    let i_virt_occ = occupation.get(i_virt).unwrap();
                                    let j_virt_occ = occupation.get(j_virt).unwrap();

                                    if (1.0-i_virt_occ).abs() > 1.0e-6 && (1.0-j_virt_occ).abs() > 1.0e-6 {
                                        let mut double_gap = ij_virt_eigen - ij_state_eigen;
                                        if double_gap.abs()<=1.0E-6 {
                                            println!("Warning: too close to degeneracy");
                                            double_gap = 1.0e-6;
                                        };
                                        double_gap /= (i_state_occ*j_state_occ*(1.0-i_virt_occ)*(1.0-j_virt_occ));

                                        // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                                        // the indices in rimo are shifted
                                        let i_loc_virt = i_virt-vir_range.start;
                                        let j_loc_virt = j_virt-vir_range.start;
                                        let e_mp2_a = eri_virt.get2d([i_loc_virt,j_loc_virt]).unwrap();
                                        let e_mp2_b = eri_virt.get2d([j_loc_virt,i_loc_virt]).unwrap();
                                        e_mp2_term_ss += (e_mp2_a - e_mp2_b).powf(2.0) / double_gap;
                                    }
                                    s.send(e_mp2_term_ss).unwrap()
                                });
                                e_mp2_ss -= receiver.into_iter().sum::<f64>();
                                if print_level >= 2 {
                                    println!("Debug: ({},{}) with the same spin ({}) finishes ", i_state,j_state, i_spin_1);
                                }
                            }
                        }
                    }


                } else {
                    let eigenvector_1 = &scf_data.semi_eigenvectors.as_ref().unwrap()[i_spin_1];
                    let eigenvalues_1 = &scf_data.semi_eigenvalues.as_ref().unwrap()[i_spin_1];
                    let occupation_1 = scf_data.occupation.get(i_spin_1).unwrap();
                    let homo_1 = scf_data.homo.get(i_spin_1).unwrap().clone();
                    let lumo_1 = scf_data.lumo.get(i_spin_1).unwrap().clone();
                    //let num_occu_1 = homo_1 + 1;
                    //let num_occu_1 = lumo_1;
                    let num_occu_1 = if scf_data.mol.num_elec[i_spin_1 + 1] <= 1.0e-6 {0} else {homo_1 + 1};
                    let (rimo_1, vir_range, occ_range) = &ri3mo_vec[i_spin_1];

                    let eigenvector_2 = &scf_data.semi_eigenvectors.as_ref().unwrap()[i_spin_2];
                    let eigenvalues_2 = &scf_data.semi_eigenvalues.as_ref().unwrap()[i_spin_2];
                    let occupation_2 = scf_data.occupation.get(i_spin_2).unwrap();
                    let homo_2 = scf_data.homo.get(i_spin_2).unwrap().clone();
                    let lumo_2 = scf_data.lumo.get(i_spin_2).unwrap().clone();
                    //let num_occu_2 = homo_2 + 1;
                    //let num_occu_2 = lumo_2;
                    let num_occu_2 = if scf_data.mol.num_elec[i_spin_2 + 1] <= 1.0e-6 {0} else {homo_2 + 1};
                    let (rimo_2, _, _) = &ri3mo_vec[i_spin_2];

                    let mut eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                    let mut loc_eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                    let virt_os_pair = mpi_ix.distribution_opposite_spin_virtual_orbital_pair(lumo_1, lumo_2, num_state, scf_data.mol.ctrl.ri_pt2.mpi_mode);


                    // prepare the elec_pair for the rayon parallelization
                    //let mut elec_pair: Vec<[usize;2]> = vec![];
                    //for i_state in start_mo..num_occu_1 {
                    //    for j_state in start_mo..num_occu_2 {
                    //        elec_pair.push([i_state,j_state])
                    //    }
                    //};
                    //let (sender, receiver) = channel();
                    //elec_pair.par_iter().for_each_with(sender,|s,i_pair| {
                    for i_state in start_mo..num_occu_1 {
                        for j_state in start_mo..num_occu_2 {
                            //let i_state = i_pair[0];
                            //let j_state = i_pair[1];
                            let i_state_eigen = eigenvalues_1.get(i_state).unwrap();
                            let j_state_eigen = eigenvalues_2.get(j_state).unwrap();
                            let ij_state_eigen = i_state_eigen + j_state_eigen;
                            let i_state_occ = occupation_1.get(i_state).unwrap();
                            let j_state_occ = occupation_2.get(j_state).unwrap();

                            if i_state_occ.abs() > 1.0e-6 && j_state_occ.abs() > 1.0e-6 {
                                // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                                // the indices in rimo are shifted
                                let i_loc_state = i_state-occ_range.start;
                                let j_loc_state = j_state-occ_range.start;
                                let ri_i = rimo_1.get_reducing_matrix(i_loc_state).unwrap();
                                let ri_j = rimo_2.get_reducing_matrix(j_loc_state).unwrap();
                                //let mut eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                                //_dgemm(
                                //    &ri_i, (0..num_auxbas,0..vir_range.len()), 'T', 
                                //    &ri_j,(0..num_auxbas,0..vir_range.len()) , 'N', 
                                //    &mut eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                                //    1.0,0.0);
                                //if print_level >= 2 {
                                //    println!("Debug: enter the preparation of eri_virt");
                                //}
                                //let mut eri_virt = {
                                //    let mut loc_eri_virt = MatrixFull::new([vir_range.len(),vir_range.len()],0.0_f64);
                                //    _dgemm(
                                //        &ri_i, (0..num_auxbas_loc,0..vir_range.len()), 'T', 
                                //        &ri_j,(0..num_auxbas_loc,0..vir_range.len()) , 'N', 
                                //        &mut loc_eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                                //        1.0,0.0);
                                //    let mut eri_virt = mpi_reduce(&mpi_op.world, &loc_eri_virt.data_ref().unwrap(), 0, &SystemOperation::sum());
                                //    mpi_broadcast(&mpi_op.world, &mut eri_virt, 0);
                                //    MatrixFull::from_vec([vir_range.len(), vir_range.len()], eri_virt).unwrap()
                                //};
                                //if print_level >= 2 {
                                //    println!("Debug: leave the preparation of eri_virt");
                                //}
                                _dgemm(
                                    &ri_i, (0..num_auxbas_loc,0..vir_range.len()), 'T', 
                                    &ri_j,(0..num_auxbas_loc,0..vir_range.len()) , 'N', 
                                    &mut loc_eri_virt, (0..vir_range.len(),0..vir_range.len()), 
                                    1.0,0.0);
                                mpi_allreduce(&mpi_op.world, loc_eri_virt.data_ref().unwrap(), eri_virt.data_ref_mut().unwrap(), &SystemOperation::sum());
                                //// ==== DEBUG IGOR ====
                                //if i_state == 0 && j_state== 0 && my_rank == 0 {
                                //    println!("my rank = {}", my_rank);
                                //    eri_virt.formated_output(5, "full");
                                //}
                                //if i_state == 0 && j_state== 0 && my_rank == 1 {
                                //    println!("my rank = {}", my_rank);
                                //    eri_virt.formated_output(5, "full");
                                //}
                                //// ==== DEBUG IGOR ====
                                let (sender, receiver) = channel();
                                virt_os_pair.par_iter().for_each_with(sender,|s,i_pair| {
                                    let mut e_mp2_term_os = 0.0_f64;
                                    let i_virt = i_pair[0];
                                    let j_virt = i_pair[1];
                                    let i_virt_eigen = eigenvalues_1.get(i_virt).unwrap();
                                    let i_virt_occ = occupation_1.get(i_virt).unwrap();
                                    let j_virt_eigen = eigenvalues_2.get(j_virt).unwrap();
                                    let ij_virt_eigen = i_virt_eigen + j_virt_eigen;
                                    let j_virt_occ = occupation_2.get(j_virt).unwrap();

                                    if (1.0-i_virt_occ).abs() > 1.0e-6 && (1.0-j_virt_occ).abs() > 1.0e-6 {
                                        let mut double_gap = ij_virt_eigen - ij_state_eigen;
                                        if double_gap.abs()<=1.0E-6 {
                                            println!("Warning: too close to degeneracy")
                                        };
                                        double_gap /= (i_state_occ*j_state_occ*(1.0-i_virt_occ)*(1.0-j_virt_occ));

                                        // because we generate ri3mo for [lumo..num_state, start_mo..num_occ], 
                                        // the indices in rimo are shifted
                                        let i_loc_virt = i_virt-vir_range.start;
                                        let j_loc_virt = j_virt-vir_range.start;
                                        let e_mp2_a = eri_virt.get2d([i_loc_virt,j_loc_virt]).unwrap();
                                        e_mp2_term_os += e_mp2_a * e_mp2_a / double_gap;
                                    } //println!("debug e_mp2_term: {:16.8}, index: {:?}", e_mp2_term_os, i_pair);
                                    s.send(e_mp2_term_os).unwrap()
                                    
                                });
                                e_mp2_os -= receiver.into_iter().sum::<f64>();
                                if print_level >= 2 {
                                    println!("Debug: ({},{}) with the opposite spin ({},{}) finishes ", i_state,j_state, i_spin_1, i_spin_2);
                                }
                            }
                        }
                    };
                }
            }
        } else {
            panic!("RI3MO should be initialized before the PT2 calculations")
        };
        // reuse the default omp_num_threads setting
        //utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

        //// sum up the ss and os contribution from the mpi tasks.
        let mut e_mp2_ss = mpi_reduce(&mpi_op.world, &mut [e_mp2_ss], 0, &SystemOperation::sum())[0];
        mpi_broadcast(&mpi_op.world, &mut e_mp2_ss, 0);
        let mut e_mp2_os = mpi_reduce(&mpi_op.world, &mut [e_mp2_os], 0, &SystemOperation::sum())[0];
        mpi_broadcast(&mpi_op.world, &mut e_mp2_os, 0);
        Ok([e_mp2_ss+e_mp2_os+e_mp2_single_list[0]+e_mp2_single_list[1], e_mp2_os, e_mp2_ss])
    } else {
        restricted_open_shell_pt2_rayon(scf_data)
    }
    }
    #[cfg(not(feature = "mpi"))]
    { restricted_open_shell_pt2_rayon(scf_data) }
}

// ============================================================================
// Streaming PT2 main loop with M1-optimized ao2mo.
//
// These routines are an algorithmic alternative to the legacy
// `*_pt2_rayon[_mpi]` functions. Instead of materializing the full
// `ri3mo[naux, nvir, nocc]` tensor via `generate_ri3mo_rayon` and then doing
// the pair-contraction loop, they process occupied orbitals in blocks of
// `block_size`. For each (block_i, block_j) pair, only the corresponding
// columns of `ri3mo` are produced (via `ao2mo_rayon_m1`, which contracts the
// smaller occ side first); the full tensor is never stored.
//
// Memory model at peak:
//   rimatr (constant) + 2 * [naux, nvir, B] + per-thread scratch
//
// Total FLOPs are essentially identical to the non-streaming M1 path
// (i.e. ~40% lower than `ao2mo_rayon_v02` for the ao2mo stage). The only
// extra cost is `nocc / B` re-reads of `rimatr` columns, which adds < 1%
// overhead for typical B = 32..64.
// ============================================================================

/// Resolve the streaming block size: explicit override if given, otherwise
/// pick the smallest power of two such that `B^2 >= 4 * nthreads` (so each
/// block-pair has at least 4× more (i, j) pairs than threads for rayon to
/// balance). Clamped to [16, 128].
fn resolve_block_size(override_b: Option<usize>, nocc: usize) -> usize {
    if let Some(b) = override_b {
        return b.max(1).min(nocc.max(1));
    }
    let nthreads = rayon::current_num_threads();
    let lower = ((4 * nthreads) as f64).sqrt().ceil() as usize;
    let mut b = 16.max(lower);
    while b < 64 && b * b < (4 * nthreads) { b *= 2; }
    b = b.min(128).min(nocc.max(1));
    b.max(1)
}

/// Per-pair PT2 contribution for closed-shell.
///
/// Returns `(e_ss, e_os)` for one (i_state, j_state) pair. The caller is
/// responsible for the i ≠ j symmetry factor (multiply by 2).
///
/// Reuses the same accumulation pattern as `close_shell_pt2_rayon` but
/// parameterized by the two RI slices so it can be called from both the
/// legacy and the streaming drivers.
fn pt2_pair_contrib_closed(
    ri_i: &MatrixFullSlice<'_, f64>,
    ri_j: &MatrixFullSlice<'_, f64>,
    num_auxbas: usize,
    vir_range: &std::ops::Range<usize>,
    eigenvalues: &[f64],
    occupation: &[f64],
    num_state: usize,
    lumo: usize,
    i_state: usize,
    j_state: usize,
) -> (f64, f64) {
    let i_state_eigen = eigenvalues.get(i_state).unwrap();
    let j_state_eigen = eigenvalues.get(j_state).unwrap();
    let ij_state_eigen = i_state_eigen + j_state_eigen;
    let i_state_occ = occupation.get(i_state).unwrap() / 2.0;
    let j_state_occ = occupation.get(j_state).unwrap() / 2.0;

    let mut e_ss = 0.0_f64;
    let mut e_os = 0.0_f64;

    if i_state_occ.abs() > 1.0e-6 && j_state_occ.abs() > 1.0e-6 {
        let nvir = vir_range.len();
        let mut eri_virt = MatrixFull::new([nvir, nvir], 0.0_f64);
        _dgemm(
            ri_i, (0..num_auxbas, 0..nvir), 'T',
            ri_j, (0..num_auxbas, 0..nvir), 'N',
            &mut eri_virt, (0..nvir, 0..nvir),
            1.0, 0.0,
        );

        for i_virt in lumo..num_state {
            let i_virt_eigen = eigenvalues.get(i_virt).unwrap();
            let i_virt_occ = occupation.get(i_virt).unwrap() / 2.0;
            if (1.0 - i_virt_occ).abs() <= 1.0e-6 { continue; }
            let i_loc_virt = i_virt - vir_range.start;
            for j_virt in lumo..num_state {
                let j_virt_occ = occupation.get(j_virt).unwrap() / 2.0;
                if (1.0 - j_virt_occ).abs() <= 1.0e-6 { continue; }
                let j_virt_eigen = eigenvalues.get(j_virt).unwrap();
                let ij_virt_eigen = i_virt_eigen + j_virt_eigen;

                let mut double_gap = ij_virt_eigen - ij_state_eigen;
                if double_gap.abs() <= 1.0e-6 {
                    double_gap = 1.0e-6;
                }
                double_gap /= (i_state_occ * j_state_occ * (1.0 - i_virt_occ) * (1.0 - j_virt_occ));

                let j_loc_virt = j_virt - vir_range.start;
                let e_mp2_a = eri_virt.get2d([i_loc_virt, j_loc_virt]).unwrap();
                let e_mp2_b = eri_virt.get2d([j_loc_virt, i_loc_virt]).unwrap();
                e_ss += (e_mp2_a - e_mp2_b) * e_mp2_a / double_gap;
                e_os += e_mp2_a * e_mp2_a / double_gap;
            }
        }
    }
    (e_ss, e_os)
}
/// Streaming closed-shell PT2 driver with Fix B pipelining.
///
/// Pipelining strategy: for each (block_i, block_j) pair, the ao2mo for the
/// NEXT needed block is spawned as a scoped thread during the current PT2
/// contraction. The pre-fetch and PT2 share the global rayon pool via
/// work-stealing. Since PT2 is heavily compute-bound (~50 s/block-pair,
/// 190 FLOP/byte) and ao2mo is moderately memory-bound (~5 s/block,
/// 48 FLOP/byte), they complement each other: PT2 saturates FMA units
/// while ao2mo uses spare memory bandwidth.
///
/// Memory: block_i + block_j + pre-fetch-in-progress ≤ 3 × [naux, nvir, B].
/// For B=64, naux=7619, nvir=1517: 3 × 5.9 GB ≈ 18 GB.
pub fn close_shell_pt2_rayon_streaming(scf_data: &SCF, block_size: Option<usize>) -> anyhow::Result<[f64;3]> {
    let default_omp_num_threads = scf_data.mol.ctrl.num_threads.unwrap();
    let print_level = scf_data.mol.ctrl.print_level;

    let mut e_mp2_ss = 0.0_f64;
    let mut e_mp2_os = 0.0_f64;

    let (ri3ao, _basbas2baspar, _baspar2basbas) = match &scf_data.rimatr {
        Some(tuple) => (&tuple.0, &tuple.1, &tuple.2),
        None => panic!("close_shell_pt2_rayon_streaming: scf_data.rimatr is None; streaming requires rimatr to be materialized (set use_ri_symm = true and use_isdf = false)"),
    };

    let eigenvector = scf_data.eigenvectors.get(0).unwrap();
    let eigenvalues = scf_data.eigenvalues.get(0).unwrap();
    let occupation = scf_data.occupation.get(0).unwrap();

    let homo = scf_data.homo.get(0).unwrap().clone();
    let lumo = scf_data.lumo.get(0).unwrap().clone();
    let num_state = eigenvector.size.get(1).unwrap().clone();
    let start_mo: usize = scf_data.mol.start_mo;
    let num_occu = if scf_data.mol.num_elec[0] <= 1.0e-6 {0} else {homo + 1};

    let occ_range = start_mo..num_occu;
    let vir_range = lumo..num_state;
    let nocc = occ_range.len();
    let num_auxbas = scf_data.mol.num_auxbas;

    let b = resolve_block_size(block_size, nocc);
    if print_level > 1 {
        println!("[streaming-PT2] close_shell: nocc={}, nvir={}, naux={}, block_size={}",
                 nocc, vir_range.len(), num_auxbas, b);
    }

    let blocks: Vec<std::ops::Range<usize>> = (0..nocc)
        .step_by(b)
        .map(|s| {
            let start = occ_range.start + s;
            let end = (start + b).min(occ_range.end);
            start..end
        })
        .collect();
    let nblocks = blocks.len();
    if nblocks == 0 {
        return Ok([0.0, 0.0, 0.0]);
    }

    // ── Fix B: pipelined block-pair loop ─────────────────────────────────
    //
    // Block pair visitation order (upper triangular):
    //   (0,0) (0,1) ... (0,N-1)  (1,1) (1,2) ... (1,N-1)  ...  (N-1,N-1)
    //
    // Pre-fetch logic: during PT2(bi,bj), spawn ao2mo for the NEXT needed block:
    //   - If bj+1 < N:      next inner iter needs block_(bj+1)  → prefetch it
    //   - elif bi+1 < N:    next outer iter needs block_(bi+1)  → prefetch it
    //   - else:             no more blocks
    //
    // The pre-fetch scoped thread runs ao2mo_rayon_m1, which uses the global
    // rayon pool. The main thread's PT2 par_iter also uses the global pool.
    // Rayon's work-stealing naturally shares the 96 workers between both.
    std::thread::scope(|scope| -> anyhow::Result<()> {
        // block_i for current outer iteration.
        // For bi=0: compute synchronously. For bi>0: from pre-fetch of previous outer.
        let mut ri3mo_i = {
            let (r, _, _) = crate::scf_io::ao2mo_rayon_m1(
                eigenvector, ri3ao, vir_range.clone(), blocks[0].clone(),
            )?;
            r
        };
        // Pre-fetch handle for the NEXT block to be consumed.
        let mut prefetched: Option<std::thread::ScopedJoinHandle<RIFull<f64>>> = None;

        for bi_idx in 0..nblocks {
            // At outer-iter boundary (bi_idx > 0): consume pre-fetched block_i.
            if bi_idx > 0 {
                ri3mo_i = match prefetched.take() {
                    Some(handle) => handle.join().unwrap(),
                    None => {
                        let (r, _, _) = crate::scf_io::ao2mo_rayon_m1(
                            eigenvector, ri3ao, vir_range.clone(), blocks[bi_idx].clone(),
                        )?;
                        r
                    }
                };
            }

            for bj_idx in bi_idx..nblocks {
                let same_block = bi_idx == bj_idx;

                // Resolve block_j: alias (diagonal) or pre-fetched or synchronous.
                let ri3mo_j_owned: Option<RIFull<f64>> = if same_block {
                    None
                } else {
                    match prefetched.take() {
                        Some(handle) => Some(handle.join().unwrap()),
                        None => {
                            let (r, _, _) = crate::scf_io::ao2mo_rayon_m1(
                                eigenvector, ri3ao, vir_range.clone(), blocks[bj_idx].clone(),
                            )?;
                            Some(r)
                        }
                    }
                };
                let ri3mo_j: &RIFull<f64> = ri3mo_j_owned.as_ref().unwrap_or(&ri3mo_i);

                // Determine which block to pre-fetch during this PT2.
                let next_block_range: Option<std::ops::Range<usize>> = if bj_idx + 1 < nblocks {
                    // Same outer iter, next inner: pre-fetch block_(bj_idx+1).
                    Some(blocks[bj_idx + 1].clone())
                } else if bi_idx + 1 < nblocks {
                    // Last inner of this outer: pre-fetch block_(bi_idx+1) for next outer's block_i.
                    Some(blocks[bi_idx + 1].clone())
                } else {
                    None
                };

                // Spawn pre-fetch (background scoped thread shares rayon pool with PT2).
                prefetched = next_block_range.map(|occ_range| {
                    let eigvec = eigenvector;
                    let ri3ao_ref = ri3ao;
                    let vir = vir_range.clone();
                    scope.spawn(move || {
                        let (r, _, _) = crate::scf_io::ao2mo_rayon_m1(eigvec, ri3ao_ref, vir, occ_range).unwrap();
                        r
                    })
                });

                // ── PT2 contraction for (bi_idx, bj_idx) ────────────────────
                // This par_iter blocks the main thread; the pre-fetch scoped thread
                // runs concurrently, sharing the rayon pool via work-stealing.
                let occ_range_i = &blocks[bi_idx];
                let occ_range_j = &blocks[bj_idx];

                let mut pairs: Vec<(usize, usize, usize, usize)> = Vec::new();
                for (i_local, i_global) in occ_range_i.clone().enumerate() {
                    let j_local_start = if same_block { i_local } else { 0 };
                    for (j_local, j_global) in occ_range_j.clone().enumerate().skip(j_local_start) {
                        pairs.push((i_local, j_local, i_global, j_global));
                    }
                }

                let ri3mo_i_ref = &ri3mo_i;
                let (sender, receiver) = channel();
                pairs.par_iter().for_each_with(sender, |s, &(i_local, j_local, i_global, j_global)| {
                    omp_set_num_threads_wrapper(1);
                    let ri_i = ri3mo_i_ref.get_reducing_matrix(i_local).unwrap();
                    let ri_j = ri3mo_j.get_reducing_matrix(j_local).unwrap();
                    let (e_ss_pair, e_os_pair) = pt2_pair_contrib_closed(
                        &ri_i, &ri_j, num_auxbas, &vir_range,
                        eigenvalues, occupation, num_state, lumo,
                        i_global, j_global,
                    );
                    let (mut e_ss_pair, mut e_os_pair) = (e_ss_pair, e_os_pair);
                    if i_global != j_global {
                        e_ss_pair *= 2.0;
                        e_os_pair *= 2.0;
                    }
                    s.send((e_ss_pair, e_os_pair)).unwrap();
                });

                for (e_ss_pair, e_os_pair) in receiver.into_iter() {
                    e_mp2_ss -= e_ss_pair;
                    e_mp2_os -= e_os_pair;
                }

                // block_j no longer needed; drop before next iteration.
                drop(ri3mo_j_owned);
            }
            // End of outer iter: prefetched holds block_(bi_idx+1) for next outer's block_i.
        }

        // Drain any remaining pre-fetch (defensive; should be None at this point).
        if let Some(handle) = prefetched {
            let _ = handle.join();
        }

        Ok(())
    })?;

    omp_set_num_threads_wrapper(default_omp_num_threads);
    Ok([e_mp2_ss + e_mp2_os, e_mp2_os, e_mp2_ss])
}

/// Per-pair PT2 contribution for one spin channel (used in open-shell and ROHF).
///
/// Computes the SS or OS contribution for the spin pair (i_spin_1, i_spin_2)
/// at occupied pair (i_state, j_state). The `i_spin_1 == i_spin_2` case is
/// same-spin (SS); otherwise opposite-spin (OS).
///
/// Returns `(e_ss, e_os)` where the unused component is 0.
fn pt2_pair_contrib_spin(
    ri_i: &MatrixFullSlice<'_, f64>,
    ri_j: &MatrixFullSlice<'_, f64>,
    num_auxbas: usize,
    vir_range: &std::ops::Range<usize>,
    eigenvalues: &[f64],
    occupation: &[f64],
    num_state: usize,
    lumo: usize,
    i_state: usize,
    j_state: usize,
    same_spin: bool,
) -> (f64, f64) {
    let i_state_eigen = eigenvalues.get(i_state).unwrap();
    let j_state_eigen = eigenvalues.get(j_state).unwrap();
    let ij_state_eigen = i_state_eigen + j_state_eigen;
    let i_state_occ = occupation.get(i_state).unwrap();
    let j_state_occ = occupation.get(j_state).unwrap();

    let mut e_ss = 0.0_f64;
    let mut e_os = 0.0_f64;

    if i_state_occ.abs() > 1.0e-6 && j_state_occ.abs() > 1.0e-6 {
        let nvir = vir_range.len();
        let mut eri_virt = MatrixFull::new([nvir, nvir], 0.0_f64);
        _dgemm(
            ri_i, (0..num_auxbas, 0..nvir), 'T',
            ri_j, (0..num_auxbas, 0..nvir), 'N',
            &mut eri_virt, (0..nvir, 0..nvir),
            1.0, 0.0,
        );

        for i_virt in lumo..num_state {
            let i_virt_eigen = eigenvalues.get(i_virt).unwrap();
            let i_virt_occ = occupation.get(i_virt).unwrap();
            if (1.0 - i_virt_occ).abs() <= 1.0e-6 { continue; }
            let i_loc_virt = i_virt - vir_range.start;
            for j_virt in lumo..num_state {
                let j_virt_occ = occupation.get(j_virt).unwrap();
                if (1.0 - j_virt_occ).abs() <= 1.0e-6 { continue; }
                let j_virt_eigen = eigenvalues.get(j_virt).unwrap();
                let ij_virt_eigen = i_virt_eigen + j_virt_eigen;

                let mut double_gap = ij_virt_eigen - ij_state_eigen;
                if double_gap.abs() <= 1.0e-6 {
                    double_gap = 1.0e-6;
                }
                double_gap /= (i_state_occ * j_state_occ * (1.0 - i_virt_occ) * (1.0 - j_virt_occ));

                let j_loc_virt = j_virt - vir_range.start;
                let e_mp2_a = eri_virt.get2d([i_loc_virt, j_loc_virt]).unwrap();
                if same_spin {
                    let e_mp2_b = eri_virt.get2d([j_loc_virt, i_loc_virt]).unwrap();
                    e_ss += (e_mp2_a - e_mp2_b) * e_mp2_a / double_gap;
                } else {
                    e_os += e_mp2_a * e_mp2_a / double_gap;
                }
            }
        }
    }
    (e_ss, e_os)
}

/// Helper: process one spin-pair (i_spin_1, i_spin_2) for one (block_i, block_j)
/// combination in the open-shell streaming driver.
///
/// Returns (e_ss_total, e_os_total) accumulated for this block pair.
fn open_shell_pt2_streaming_block_pair(
    scf_data: &SCF,
    ri3ao: &MatrixFull<f64>,
    i_spin_1: usize,
    i_spin_2: usize,
    occ_range_i: &std::ops::Range<usize>,
    occ_range_j: &std::ops::Range<usize>,
    vir_range: &std::ops::Range<usize>,
    same_block: bool,
    ri3mo_i: &RIFull<f64>,
    ri3mo_j_owned: &Option<RIFull<f64>>,
) -> (f64, f64) {
    let ri3mo_j: &RIFull<f64> = ri3mo_j_owned.as_ref().unwrap_or(ri3mo_i);

    let eigenvector_1 = &scf_data.eigenvectors[i_spin_1];
    let eigenvector_2 = &scf_data.eigenvectors[i_spin_2];
    let _ = eigenvector_2;
    let eigenvalues_1 = &scf_data.eigenvalues[i_spin_1];
    let eigenvalues_2 = &scf_data.eigenvalues[i_spin_2];
    let occupation_1 = &scf_data.occupation[i_spin_1];
    let occupation_2 = &scf_data.occupation[i_spin_2];
    let lumo_1 = scf_data.lumo[i_spin_1];
    let lumo_2 = scf_data.lumo[i_spin_2];
    let num_state = scf_data.mol.num_state;
    let num_auxbas = scf_data.mol.num_auxbas;
    let _ = eigenvector_1;

    // Same spin pair (αα or ββ): SS only.
    // Cross spin pair (αβ): OS only.
    let same_spin = i_spin_1 == i_spin_2;
    let eigenvalues_j = if same_spin { eigenvalues_1 } else { eigenvalues_2 };
    let occupation_j = if same_spin { occupation_1 } else { occupation_2 };
    let lumo_j = if same_spin { lumo_1 } else { lumo_2 };
    let vir_range_j = if same_spin { vir_range.clone() } else { lumo_j..num_state };

    let _ = eigenvalues_j;
    let _ = occupation_j;

    // Build (i_local, j_local, i_global, j_global) pair list.
    let mut pairs: Vec<(usize, usize, usize, usize)> = Vec::new();
    for (i_local, i_global) in occ_range_i.clone().enumerate() {
        let j_local_start = if same_block { i_local } else { 0 };
        for (j_local, j_global) in occ_range_j.clone().enumerate().skip(j_local_start) {
            pairs.push((i_local, j_local, i_global, j_global));
        }
    }

    let (sender, receiver) = channel();
    pairs.par_iter().for_each_with(sender, |s, &(i_local, j_local, i_global, j_global)| {
        omp_set_num_threads_wrapper(1);
        let ri_i = ri3mo_i.get_reducing_matrix(i_local).unwrap();
        let ri_j = ri3mo_j.get_reducing_matrix(j_local).unwrap();

        // For same_spin, both spins use eigenvalues_1/occupation_1/vir_range.
        // For cross spin, i uses spin_1, j uses spin_2.
        let (e_ss_pair, e_os_pair);
        if same_spin {
            let (ss, _os) = pt2_pair_contrib_spin(
                &ri_i, &ri_j, num_auxbas, vir_range,
                eigenvalues_1, occupation_1, num_state, lumo_1,
                i_global, j_global, true,
            );
            e_ss_pair = ss; e_os_pair = 0.0;
        } else {
            // Cross-spin: i in spin_1, j in spin_2; uses different eigenvalues/occupation/vir_range.
            // pt2_pair_contrib_spin assumes same eigenvalues/occ for both i and j; for cross-spin
            // we need a custom accumulation. Fall back to manual computation here.
            let i_state_eigen = eigenvalues_1.get(i_global).unwrap();
            let j_state_eigen = eigenvalues_2.get(j_global).unwrap();
            let ij_state_eigen = i_state_eigen + j_state_eigen;
            let i_state_occ = occupation_1.get(i_global).unwrap();
            let j_state_occ = occupation_2.get(j_global).unwrap();

            let mut e_os_local = 0.0_f64;
            if i_state_occ.abs() > 1.0e-6 && j_state_occ.abs() > 1.0e-6 {
                let nvir_1 = vir_range.len();
                let nvir_2 = vir_range_j.len();
                let mut eri_virt = MatrixFull::new([nvir_1, nvir_2], 0.0_f64);
                _dgemm(
                    &ri_i, (0..num_auxbas, 0..nvir_1), 'T',
                    &ri_j, (0..num_auxbas, 0..nvir_2), 'N',
                    &mut eri_virt, (0..nvir_1, 0..nvir_2),
                    1.0, 0.0,
                );
                for i_virt in lumo_1..num_state {
                    let i_virt_eigen = eigenvalues_1.get(i_virt).unwrap();
                    let i_virt_occ = occupation_1.get(i_virt).unwrap();
                    if (1.0 - i_virt_occ).abs() <= 1.0e-6 { continue; }
                    let i_loc_virt = i_virt - vir_range.start;
                    for j_virt in lumo_2..num_state {
                        let j_virt_occ = occupation_2.get(j_virt).unwrap();
                        if (1.0 - j_virt_occ).abs() <= 1.0e-6 { continue; }
                        let j_virt_eigen = eigenvalues_2.get(j_virt).unwrap();
                        let ij_virt_eigen = i_virt_eigen + j_virt_eigen;

                        let mut double_gap = ij_virt_eigen - ij_state_eigen;
                        if double_gap.abs() <= 1.0e-6 { double_gap = 1.0e-6; }
                        double_gap /= (i_state_occ * j_state_occ * (1.0 - i_virt_occ) * (1.0 - j_virt_occ));

                        let j_loc_virt = j_virt - vir_range_j.start;
                        let e_mp2_a = eri_virt.get2d([i_loc_virt, j_loc_virt]).unwrap();
                        e_os_local += e_mp2_a * e_mp2_a / double_gap;
                    }
                }
            }
            e_ss_pair = 0.0; e_os_pair = e_os_local;
        }

        s.send((e_ss_pair, e_os_pair)).unwrap();
    });

    let mut e_ss = 0.0_f64;
    let mut e_os = 0.0_f64;
    for (e_ss_pair, e_os_pair) in receiver.into_iter() {
        e_ss -= e_ss_pair;
        e_os -= e_os_pair;
    }
    (e_ss, e_os)
}

/// Streaming open-shell (UKS) PT2 driver.
///
/// Loops over the three spin-pair types (αα, αβ, ββ). For each, runs the
/// closed-shell-style block streaming over occ blocks of the two spin channels.
pub fn open_shell_pt2_rayon_streaming(scf_data: &SCF, block_size: Option<usize>) -> anyhow::Result<[f64;3]> {
    let default_omp_num_threads = scf_data.mol.ctrl.num_threads.unwrap();
    let print_level = scf_data.mol.ctrl.print_level;

    let mut e_mp2_ss = 0.0_f64;
    let mut e_mp2_os = 0.0_f64;

    let (ri3ao, _basbas2baspar, _baspar2basbas) = match &scf_data.rimatr {
        Some(tuple) => (&tuple.0, &tuple.1, &tuple.2),
        None => panic!("open_shell_pt2_rayon_streaming: scf_data.rimatr is None; streaming requires rimatr to be materialized"),
    };

    let num_state = scf_data.mol.num_state;
    let start_mo: usize = scf_data.mol.start_mo;
    let num_auxbas = scf_data.mol.num_auxbas;
    let _ = num_auxbas;

    let i_spin_pair: [(usize, usize); 3] = [(0, 0), (0, 1), (1, 1)];

    for (i_spin_1, i_spin_2) in i_spin_pair {
        let eigenvector_1 = &scf_data.eigenvectors[i_spin_1];
        let homo_1 = scf_data.homo[i_spin_1];
        let lumo_1 = scf_data.lumo[i_spin_1];
        let num_occu_1 = if scf_data.mol.num_elec[i_spin_1 + 1] <= 1.0e-6 { 0 } else { homo_1 + 1 };
        let occ_range_1 = start_mo..num_occu_1;
        let vir_range_1 = lumo_1..num_state;

        let eigenvector_2 = &scf_data.eigenvectors[i_spin_2];
        let homo_2 = scf_data.homo[i_spin_2];
        let lumo_2 = scf_data.lumo[i_spin_2];
        let num_occu_2 = if scf_data.mol.num_elec[i_spin_2 + 1] <= 1.0e-6 { 0 } else { homo_2 + 1 };
        let occ_range_2 = start_mo..num_occu_2;
        let vir_range_2 = lumo_2..num_state;

        let nocc_1 = occ_range_1.len();
        let nocc_2 = occ_range_2.len();
        if nocc_1 == 0 || nocc_2 == 0 { continue; }

        let b1 = resolve_block_size(block_size, nocc_1);
        let b2 = resolve_block_size(block_size, nocc_2);
        if print_level > 1 {
            println!("[streaming-PT2] open_shell spin-pair ({}, {}): nocc=({}, {}), block_size=({}, {})",
                     i_spin_1, i_spin_2, nocc_1, nocc_2, b1, b2);
        }

        let blocks_1: Vec<std::ops::Range<usize>> = (0..nocc_1)
            .step_by(b1)
            .map(|s| {
                let start = occ_range_1.start + s;
                let end = (start + b1).min(occ_range_1.end);
                start..end
            })
            .collect();
        let blocks_2: Vec<std::ops::Range<usize>> = (0..nocc_2)
            .step_by(b2)
            .map(|s| {
                let start = occ_range_2.start + s;
                let end = (start + b2).min(occ_range_2.end);
                start..end
            })
            .collect();

        for (bi_idx, occ_range_i) in blocks_1.iter().enumerate() {
            let (ri3mo_i, _, _) = crate::scf_io::ao2mo_rayon_m1(
                eigenvector_1, ri3ao,
                vir_range_1.clone(), occ_range_i.clone(),
            )?;

            for (bj_idx, occ_range_j) in blocks_2.iter().enumerate() {
                // For (αα) and (ββ), enforce upper triangular block iteration.
                // For (αβ), iterate all (bi, bj) combinations.
                if i_spin_1 == i_spin_2 && bj_idx < bi_idx { continue; }
                let same_block = i_spin_1 == i_spin_2 && bi_idx == bj_idx;

                let ri3mo_j_owned: Option<RIFull<f64>> = if same_block {
                    None
                } else {
                    let (rj, _, _) = crate::scf_io::ao2mo_rayon_m1(
                        eigenvector_2, ri3ao,
                        vir_range_2.clone(), occ_range_j.clone(),
                    )?;
                    Some(rj)
                };

                let (e_ss, e_os) = open_shell_pt2_streaming_block_pair(
                    scf_data, ri3ao, i_spin_1, i_spin_2,
                    occ_range_i, occ_range_j, &vir_range_1,
                    same_block, &ri3mo_i, &ri3mo_j_owned,
                );
                e_mp2_ss += e_ss;
                e_mp2_os += e_os;
            }
        }
    }

    omp_set_num_threads_wrapper(default_omp_num_threads);
    Ok([e_mp2_ss + e_mp2_os, e_mp2_os, e_mp2_ss])
}

/// Streaming ROHF PT2 driver.
///
/// ROHF uses semi-canonical orbitals; the PT2 calculation treats α and β
/// channels as separate spin channels with the same spatial orbitals but
/// different occupation/eigenvalue vectors. The single-excitation correction
/// (CIS-like contribution from singly-occupied orbitals) is computed using
/// the same logic as `restricted_open_shell_pt2_rayon`.
pub fn restricted_open_shell_pt2_rayon_streaming(scf_data: &SCF, block_size: Option<usize>) -> anyhow::Result<[f64;3]> {
    let print_level = scf_data.mol.ctrl.print_level;

    // Single-excitation contribution (no ao2mo or rimatr needed; uses semi Fock).
    let mut e_mp2_single_list = [0.0_f64, 0.0_f64];
    for i_spin in 0..2 {
        let eigenvalues_spin = &scf_data.semi_eigenvalues.as_ref().unwrap()[i_spin];
        let fock_spin = &scf_data.semi_fock.as_ref().unwrap()[i_spin];
        for i_occ in 0..scf_data.lumo[i_spin] {
            for i_virt in scf_data.lumo[i_spin]..scf_data.mol.num_state {
                let single_gap = eigenvalues_spin[i_virt] - eigenvalues_spin[i_occ];
                e_mp2_single_list[i_spin] += -fock_spin[(i_virt, i_occ)].powf(2.0) / single_gap;
            }
        }
    }

    let default_omp_num_threads = scf_data.mol.ctrl.num_threads.unwrap();

    // Double-excitation part: treat as open-shell using semi-canonical orbitals.
    // semi_eigenvectors[0] = α, semi_eigenvectors[1] = β
    // For ROHF: occupation/semi_eigenvalues differ between α and β.
    // We reuse the open-shell streaming driver, but feed semi_eigenvectors.
    // For now, call open_shell streaming logic with a temporary SCF view.
    // Since the open-shell driver uses scf_data.eigenvectors/eigenvalues/occupation directly,
    // and ROHF stores semi quantities in separate fields, we need to dispatch manually.

    let mut e_mp2_ss = 0.0_f64;
    let mut e_mp2_os = 0.0_f64;

    let (ri3ao, _basbas2baspar, _baspar2basbas) = match &scf_data.rimatr {
        Some(tuple) => (&tuple.0, &tuple.1, &tuple.2),
        None => panic!("restricted_open_shell_pt2_rayon_streaming: scf_data.rimatr is None; streaming requires rimatr to be materialized"),
    };

    let num_state = scf_data.mol.num_state;
    let start_mo: usize = scf_data.mol.start_mo;
    let semi_eigenvectors = scf_data.semi_eigenvectors.as_ref().unwrap();
    let semi_eigenvalues = scf_data.semi_eigenvalues.as_ref().unwrap();
    // Use occupation from the semi-canonical alpha/beta channels.
    // For ROHF, occupation[0] = α occupation, occupation[1] = β occupation.
    let occupation = &scf_data.occupation;
    let num_auxbas = scf_data.mol.num_auxbas;
    let _ = num_auxbas;

    let i_spin_pair: [(usize, usize); 3] = [(0, 0), (0, 1), (1, 1)];

    for (i_spin_1, i_spin_2) in i_spin_pair {
        let eigenvector_1 = &semi_eigenvectors[i_spin_1];
        let eigenvalues_1 = &semi_eigenvalues[i_spin_1];
        let occupation_1 = &occupation[i_spin_1];
        let lumo_1 = scf_data.lumo[i_spin_1];
        let num_occu_1 = if scf_data.mol.num_elec[i_spin_1 + 1] <= 1.0e-6 { 0 } else { lumo_1 };
        let occ_range_1 = start_mo..num_occu_1;
        let vir_range_1 = lumo_1..num_state;

        let eigenvector_2 = &semi_eigenvectors[i_spin_2];
        let eigenvalues_2 = &semi_eigenvalues[i_spin_2];
        let occupation_2 = &occupation[i_spin_2];
        let lumo_2 = scf_data.lumo[i_spin_2];
        let num_occu_2 = if scf_data.mol.num_elec[i_spin_2 + 1] <= 1.0e-6 { 0 } else { lumo_2 };
        let occ_range_2 = start_mo..num_occu_2;
        let vir_range_2 = lumo_2..num_state;

        let nocc_1 = occ_range_1.len();
        let nocc_2 = occ_range_2.len();
        if nocc_1 == 0 || nocc_2 == 0 { continue; }

        let b1 = resolve_block_size(block_size, nocc_1);
        let b2 = resolve_block_size(block_size, nocc_2);
        if print_level > 1 {
            println!("[streaming-PT2] ROHF spin-pair ({}, {}): nocc=({}, {}), block_size=({}, {})",
                     i_spin_1, i_spin_2, nocc_1, nocc_2, b1, b2);
        }

        let same_spin = i_spin_1 == i_spin_2;

        let blocks_1: Vec<std::ops::Range<usize>> = (0..nocc_1)
            .step_by(b1)
            .map(|s| {
                let start = occ_range_1.start + s;
                let end = (start + b1).min(occ_range_1.end);
                start..end
            })
            .collect();
        let blocks_2: Vec<std::ops::Range<usize>> = (0..nocc_2)
            .step_by(b2)
            .map(|s| {
                let start = occ_range_2.start + s;
                let end = (start + b2).min(occ_range_2.end);
                start..end
            })
            .collect();

        for (bi_idx, occ_range_i) in blocks_1.iter().enumerate() {
            let (ri3mo_i, _, _) = crate::scf_io::ao2mo_rayon_m1(
                eigenvector_1, ri3ao,
                vir_range_1.clone(), occ_range_i.clone(),
            )?;

            for (bj_idx, occ_range_j) in blocks_2.iter().enumerate() {
                if same_spin && bj_idx < bi_idx { continue; }
                let same_block = same_spin && bi_idx == bj_idx;

                let ri3mo_j_owned: Option<RIFull<f64>> = if same_block {
                    None
                } else {
                    let (rj, _, _) = crate::scf_io::ao2mo_rayon_m1(
                        eigenvector_2, ri3ao,
                        vir_range_2.clone(), occ_range_j.clone(),
                    )?;
                    Some(rj)
                };
                let ri3mo_j: &RIFull<f64> = ri3mo_j_owned.as_ref().unwrap_or(&ri3mo_i);

                // Build pair list.
                let mut pairs: Vec<(usize, usize, usize, usize)> = Vec::new();
                for (i_local, i_global) in occ_range_i.clone().enumerate() {
                    let j_local_start = if same_block { i_local } else { 0 };
                    for (j_local, j_global) in occ_range_j.clone().enumerate().skip(j_local_start) {
                        pairs.push((i_local, j_local, i_global, j_global));
                    }
                }

                let (sender, receiver) = channel();
                pairs.par_iter().for_each_with(sender, |s, &(i_local, j_local, i_global, j_global)| {
                    omp_set_num_threads_wrapper(1);
                    let ri_i = ri3mo_i.get_reducing_matrix(i_local).unwrap();
                    let ri_j = ri3mo_j.get_reducing_matrix(j_local).unwrap();

                    let i_state_eigen = eigenvalues_1.get(i_global).unwrap();
                    let j_state_eigen = eigenvalues_2.get(j_global).unwrap();
                    let ij_state_eigen = i_state_eigen + j_state_eigen;
                    let i_state_occ = occupation_1.get(i_global).unwrap();
                    let j_state_occ = occupation_2.get(j_global).unwrap();

                    let mut e_ss_local = 0.0_f64;
                    let mut e_os_local = 0.0_f64;
                    if i_state_occ.abs() > 1.0e-6 && j_state_occ.abs() > 1.0e-6 {
                        let nvir_1 = vir_range_1.len();
                        let nvir_2 = vir_range_2.len();
                        let mut eri_virt = MatrixFull::new([nvir_1, nvir_2], 0.0_f64);
                        _dgemm(
                            &ri_i, (0..scf_data.mol.num_auxbas, 0..nvir_1), 'T',
                            &ri_j, (0..scf_data.mol.num_auxbas, 0..nvir_2), 'N',
                            &mut eri_virt, (0..nvir_1, 0..nvir_2),
                            1.0, 0.0,
                        );
                        for i_virt in lumo_1..num_state {
                            let i_virt_eigen = eigenvalues_1.get(i_virt).unwrap();
                            let i_virt_occ = occupation_1.get(i_virt).unwrap();
                            if (1.0 - i_virt_occ).abs() <= 1.0e-6 { continue; }
                            let i_loc_virt = i_virt - vir_range_1.start;
                            // Same-spin case mirrors `restricted_open_shell_pt2_rayon`:
                            // inner virtual loop is strict upper triangular (j_virt > i_virt),
                            // and the antisymmetrized integrand is squared:
                            //   (e_mp2_a - e_mp2_b)^2 / gap
                            // No factor 2 is applied at the occ-pair level (the upper
                            // triangular occ iteration already avoids double-counting).
                            // The OS (cross-spin) case iterates all (i_virt, j_virt).
                            let j_virt_range: Box<dyn Iterator<Item=usize> + Send> = if same_spin {
                                Box::new((i_virt + 1)..num_state)
                            } else {
                                Box::new(lumo_2..num_state)
                            };
                            for j_virt in j_virt_range {
                                let j_virt_occ = occupation_2.get(j_virt).unwrap();
                                if (1.0 - j_virt_occ).abs() <= 1.0e-6 { continue; }
                                let j_virt_eigen = eigenvalues_2.get(j_virt).unwrap();
                                let ij_virt_eigen = i_virt_eigen + j_virt_eigen;

                                let mut double_gap = ij_virt_eigen - ij_state_eigen;
                                if double_gap.abs() <= 1.0e-6 { double_gap = 1.0e-6; }
                                double_gap /= (i_state_occ * j_state_occ * (1.0 - i_virt_occ) * (1.0 - j_virt_occ));

                                let j_loc_virt = j_virt - vir_range_2.start;
                                let e_mp2_a = eri_virt.get2d([i_loc_virt, j_loc_virt]).unwrap();
                                if same_spin {
                                    let e_mp2_b = eri_virt.get2d([j_loc_virt, i_loc_virt]).unwrap();
                                    e_ss_local += (e_mp2_a - e_mp2_b).powf(2.0) / double_gap;
                                } else {
                                    e_os_local += e_mp2_a * e_mp2_a / double_gap;
                                }
                            }
                        }
                    }

                    // No occ-pair factor 2 for ROHF same-spin: the upper-triangular
                    // occ block iteration combined with strict upper-triangular virt
                    // iteration already accounts for unique (i, j, a, b) tuples.
                    let _ = i_global; // (no factor 2 applied)
                    s.send((e_ss_local, e_os_local)).unwrap();
                });

                for (e_ss_pair, e_os_pair) in receiver.into_iter() {
                    e_mp2_ss -= e_ss_pair;
                    e_mp2_os -= e_os_pair;
                }
            }
        }
    }

    // Single-excitation contribution: added to the total only (not to os or ss).
    // Mirrors `restricted_open_shell_pt2_rayon` at mod.rs:1464:
    //   Ok([ss + os + single, os, ss])
    let e_single = e_mp2_single_list[0] + e_mp2_single_list[1];

    omp_set_num_threads_wrapper(default_omp_num_threads);
    Ok([e_mp2_ss + e_mp2_os + e_single, e_mp2_os, e_mp2_ss])
}