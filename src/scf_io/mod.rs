#![warn(unused_imports)]
use crate::basis_io::ecp::ghost_effective_potential_matrix;
use crate::check_norm::force_state_occupation::adapt_occupation_with_force_projection;
use crate::check_norm::{self, generate_occupation_frac_occ, generate_occupation_integer, generate_occupation_sad, OCCType};
use crate::dft::gen_grids::prune::prune_by_rho;
use crate::dft::{DFTType, Grids};
use crate::geom_io::{calc_nuc_energy, calc_nuc_energy_with_ext_field, calc_nuc_energy_with_point_charges};
#[cfg(feature = "mpi")]
use crate::mpi_io::{mpi_broadcast, mpi_broadcast_matrixfull, mpi_broadcast_vector, mpi_reduce};
use crate::mpi_io::MPIOperator;
use crate::utilities::{self, TimeRecords};
use crate::utilities::memory_batch::*;
use crate::ctrl_io::ri_jk_io::*;

mod addons;
mod fchk;
pub mod print;
mod pyrest_scf_io;
pub mod scfrecord;
pub mod smear;
pub mod util;

#[cfg(feature = "mpi")]
use mpi::collective::SystemOperation;
use pyo3::{pyclass};
use tensors::matrix_blas_lapack::{_dgemm, _dgemm_full, _dgemv, _dspgvx, _dsymm, _dsyrk, _hamiltonian_fast_solver, _power_rayon_for_symmetric_matrix, _dsyevd};
use tensors::{map_upper_to_full, BasicMatrix, ERIFold4, MathMatrix, MatrixFull, MatrixFullSlice, MatrixUpper, MatrixUpperSlice, RIFull, TensorSliceMut};
use tensors::{TensorOpt,TensorSlice};
use itertools::{Itertools};
use rayon::prelude::*;
use std::collections::HashMap;
use crossbeam::{channel::{unbounded},thread::{scope}};
use std::sync::mpsc::{channel};
use crate::isdf::{prepare_for_ri_isdf, prepare_m_isdf};
use crate::molecule_io::{Molecule};
use crate::initial_guess::{initial_guess, update_basis_from_hdf5chk};
use crate::external_libs::dftd;
use crate::constants::{SQRT_THRESHOLD};
use crate::solvent::{PcmObject, PcmScf, solvent_prepare, debug_print_pcm};
use crate::x2c::RelativisticMethod;
use crate::ri_jk;
use tensors::matrix_blas_lapack::{omp_get_num_threads_wrapper,omp_set_num_threads_wrapper};
use log::{debug, info, trace, warn};
use scfrecord::ScfTraceRecord;
use self::util::occupied_orbital_count;
use self::util::norm;
use smear::apply_smearing;
use smear::annealed_sigma;

#[allow(unused_imports)]
use tensors::BasicMatUp;

#[pyclass]
#[derive(Clone)]
pub struct SCF {
    #[pyo3(get,set)]
    pub mol: Molecule,
    pub ovlp: MatrixUpper<f64>,
    pub h_core: MatrixUpper<f64>,
    //pub ijkl: Option<ERIFull<f64>>,
    pub ijkl: Option<ERIFold4<f64>>,
    pub ri3fn: Option<RIFull<f64>>,
    pub ri3fn_sr: Option<RIFull<f64>>,
    pub ri3fn_isdf: Option<RIFull<f64>>,
    pub tab_ao: Option<MatrixFull<f64>>,
    pub m: Option<MatrixFull<f64>>,
    pub rimatr: Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
    pub rimatr_sr: Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
    pub ri3mo: Option<Vec<(RIFull<f64>,std::ops::Range<usize> , std::ops::Range<usize>)>>,
    pub ri3mo_full:Option<Vec<(RIFull<f64>,std::ops::Range<usize> , std::ops::Range<usize>)>>,
    pub ri3fn_bse: Option<RIFull<f64>>,
    pub rimatr_bse: Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
    pub num_auxbas_bse: Option<usize>,
    #[pyo3(get,set)]
    pub eigenvalues: [Vec<f64>;2],
    //pub eigenvectors: Vec<Tensors<f64>>,
    pub eigenvectors: [MatrixFull<f64>;2],
    //pub density_matrix: Vec<Tensors<f64>>,
    //pub density_matrix: [MatrixFull<f64>;2],
    pub density_matrix: Vec<MatrixFull<f64>>,
    //pub hamiltonian: Vec<Tensors<f64>>,
    pub hamiltonian: [MatrixUpper<f64>;2],
    pub roothaan_hamiltonian: Option<MatrixUpper<f64>>,
    pub semi_eigenvalues: Option<[Vec<f64>; 2]>,
    pub semi_eigenvectors: Option<[MatrixFull<f64>; 2]>,
    pub semi_fock: Option<[MatrixFull<f64>; 2]>,
    pub scftype: SCFType,
    #[pyo3(get,set)]
    pub occupation: [Vec<f64>;2],
    #[pyo3(get,set)]
    pub homo: [usize;2],
    #[pyo3(get,set)]
    pub lumo: [usize;2],
    #[pyo3(get,set)]
    pub nuc_energy: f64,
    #[pyo3(get,set)]
    pub scf_energy: f64,
    pub smearing_entropy: f64,
    pub current_smear_sigma: f64,
    pub grad_dm: [MatrixFull<f64>; 2],
    pub grids: Option<Grids>,
    pub empirical_dispersion_energy: f64,
    pub energies: HashMap<String,Vec<f64>>,
    pub ref_eigenvectors: HashMap<String, ([MatrixFull<f64>;2], [usize;4])>,
    pub renormalized_singles_particles:Vec<f64>,
    pub gwqp:(Vec<f64>,Vec<f64>),
    pub algorithm_jk: AlgorithmJK,
    pub solvent_static_obj: Option<PcmObject>,
    pub solvent_scf: Option<PcmScf>,
}

#[derive(Clone,Copy)]
pub enum SCFType {
    RHF,
    ROHF,
    UHF
}


impl SCF {
    pub fn init_scf(mol: &Molecule) -> SCF {
        let mut scf_data = SCF {
            mol: mol.clone(),
            ovlp: MatrixUpper::new(1,0.0),
            h_core: MatrixUpper::new(1,0.0),
            ijkl: None,
            ri3fn: None,
            ri3fn_sr: None,
            ri3fn_isdf: None,
            tab_ao: None,
            m: None,
            rimatr: None,
            rimatr_sr: None,
            ri3mo: None,
            ri3mo_full:None,
            ri3fn_bse: None,
            rimatr_bse: None,
            num_auxbas_bse: None,
            eigenvalues: [vec![],vec![]],
            hamiltonian: [MatrixUpper::empty(),
                          MatrixUpper::empty()],
            roothaan_hamiltonian: None,
            semi_eigenvalues: None,
            semi_eigenvectors: None,
            semi_fock: None,
            eigenvectors: [MatrixFull::empty(),
                           MatrixFull::empty()],
            ref_eigenvectors: HashMap::new(),
            //density_matrix: [MatrixFull::new([1,1],0.0),
            //                     MatrixFull::new([1,1],0.0)],
            density_matrix: vec![MatrixFull::empty(),
                                 MatrixFull::empty()],
            scftype: SCFType::RHF,
            occupation: [vec![],vec![]],
            homo: [0,0],
            lumo: [0,0],
            nuc_energy: 0.0,
            scf_energy: 0.0,
            smearing_entropy: 0.0,
            current_smear_sigma: 0.0,
            grad_dm: [MatrixFull::empty(), MatrixFull::empty()],
            empirical_dispersion_energy: 0.0,
            grids: None,
            energies: HashMap::new(),
            renormalized_singles_particles:Vec::new(),
            gwqp:(Vec::new(),Vec::new()),
            algorithm_jk: AlgorithmJK::Default,
            solvent_static_obj: None,
            solvent_scf: None,
        };

        // at first check the scf type: RHF, ROHF or UHF
        scf_data.scftype = if mol.num_elec[1]==mol.num_elec[2] && ! mol.ctrl.spin_polarization {
            SCFType::RHF
        } else if mol.num_elec[1]!=mol.num_elec[2] && ! mol.ctrl.spin_polarization {
            SCFType::ROHF
        } else {      
            SCFType::UHF
        };
        match &scf_data.scftype {
            SCFType::RHF => {
                info!("Restricted Hartree-Fock (or Kohn-Sham) algorithm is invoked.")},
            SCFType::ROHF => {
                info!("Restricted open shell Hartree-Fock (or Kohn-Sham) algorithm is invoked.");
                // In ROHF, although the Roothaan Fock matrix is not separated into alpha and beta spin channels, 
                // it is derived based on the density matrices of the alpha and beta spin channels. 
                // Therefore, even though "spin_polarization=False" is specified as input, we handle it as "spin_channel=2".
                scf_data.mol.ctrl.spin_channel=2;
                scf_data.mol.spin_channel=2;
                scf_data.mol.xc_data.spin_channel=2;
            },
            SCFType::UHF => {
                info!("Unrestricted Hartree-Fock (or Kohn-Sham) algorithm is invoked.")
            },
        };

        scf_data
    }

    /// Determine the J/K algorithms based on user input and memory requirement.
    /// 
    /// Only in effective when
    /// 
    /// - Some user input is not given (i.e. field `algorithm_jk` is not specified).
    /// - Some user input is not a determined algorithm (i.e. field `algorithm_jk` is set to `ri`
    ///   instead of more-determined `ri-incore` or `ri-direct`).
    /// 
    /// The memory consumption for RI integrals is estimated as (nao, nao, naux) * 8 bytes.
    /// This estimation is twice larger than the actual memory consumption, to be conservative.
    pub fn update_jk_algorithms(&mut self) {
        let mol = &self.mol;

        // check algorithms of J/K
        let algorithm_jk = mol.ctrl.algorithm_jk;
        // by default, we will let it be RI
        let algorithm_jk = match algorithm_jk {
            AlgorithmJK::Default => AlgorithmJK::Ri,
            AlgorithmJK::Separated(algorithm_j, algorithm_k) => {
                let new_algorithm_j = if algorithm_j == AlgorithmJ::Default { AlgorithmJ::Ri } else { algorithm_j };
                let new_algorithm_k = if algorithm_k == AlgorithmK::Default { AlgorithmK::Ri } else { algorithm_k };
                AlgorithmJK::Separated(new_algorithm_j, new_algorithm_k)
            },
            _ => algorithm_jk,
        };
        // check memory requirement for RI
        let has_ri_non_specified = match algorithm_jk {
            AlgorithmJK::Ri => true,
            AlgorithmJK::Separated(algorithm_j, algorithm_k) => algorithm_j == AlgorithmJ::Ri || algorithm_k == AlgorithmK::Ri,
            _ => false,
        };
        let algorithm_jk = if has_ri_non_specified {
            info!("Checking memory requirement for RI J/K algorithms...");
            let nao = mol.num_basis;
            let naux = mol.num_auxbas;
            // range-separate hybrid functionals requires double memory for RI integrals due to the need of both standard RI and short-range RI integrals.
            let scale_rsh = if mol.xc_data.is_rsh() { 2.0 } else { 1.0 };
            // TODO: for safety, we add factor 1.5 to the memory requirement
            let mem_cderi_mb = 1.5 * scale_rsh * 8.0 * (0.5 * (nao * nao * naux) as f64) / 1024.0 / 1024.0;
            let mem_avail_mb = mol.ctrl.max_memory.map(|max_memory| {
                max_memory - detect_used_memory_mb("proc")
            }).unwrap_or_else(detect_available_memory_mb);
            let algorithm_jk = if mem_avail_mb  < mem_cderi_mb {
                info!("Memory available for 1.5 times of RI integrals ({:.2} MB) is less than required ({:.2} MB).", mem_avail_mb, mem_cderi_mb);
                info!("Switch to direct RI-J/K algorithms.");
                if algorithm_jk == AlgorithmJK::Ri {
                    AlgorithmJK::RiDirect
                } else if let AlgorithmJK::Separated(algorithm_j, algorithm_k) = algorithm_jk {
                    let new_algorithm_j = if algorithm_j == AlgorithmJ::Ri { AlgorithmJ::RiDirect } else { algorithm_j };
                    let new_algorithm_k = if algorithm_k == AlgorithmK::Ri { AlgorithmK::RiDirect } else { algorithm_k };
                    AlgorithmJK::Separated(new_algorithm_j, new_algorithm_k)
                } else {
                    algorithm_jk
                }
            } else {
                info!("Memory available for 1.5 times of RI integrals ({:.2} MB) is more than required ({:.2} MB).", mem_avail_mb, mem_cderi_mb);
                info!("Using standard incore RI-J/K algorithms.");
                if algorithm_jk == AlgorithmJK::Ri {
                    AlgorithmJK::RiIncore
                } else if let AlgorithmJK::Separated(algorithm_j, algorithm_k) = algorithm_jk {
                    let new_algorithm_j = if algorithm_j == AlgorithmJ::Ri { AlgorithmJ::RiIncore } else { algorithm_j };
                    let new_algorithm_k = if algorithm_k == AlgorithmK::Ri { AlgorithmK::RiIncore } else { algorithm_k };
                    AlgorithmJK::Separated(new_algorithm_j, new_algorithm_k)
                } else {
                    algorithm_jk
                }
            };
            algorithm_jk
        } else {
            algorithm_jk
        };

        // reassign the algorithm_jk to disable any ambiguity
        self.algorithm_jk = algorithm_jk;
    }


    pub fn prepare_necessary_integrals(&mut self, mpi_operator: &Option<MPIOperator>) {
        // prepare standard two, three, and four-center integrals.
        // for ISDF integrals, they needs density grids, and thus should be prepared after the grid initialization

        let print_level = self.mol.ctrl.print_level;

        //========================================
        // For nuclear energy, includin the interaction with the ghost atoms with point charges
        self.nuc_energy = calc_nuc_energy(&self.mol.geom, &self.mol.basis4elem);

        let nuc_energy_pc = calc_nuc_energy_with_point_charges(&self.mol.geom, &self.mol.basis4elem);
        let nuc_energy_ext_field = calc_nuc_energy_with_ext_field(&self.mol.geom, &self.mol.basis4elem);
        self.nuc_energy += nuc_energy_pc;
        self.nuc_energy += nuc_energy_ext_field;

        info!("Nuc_energy: {:16.8} Hartree",self.nuc_energy);
        if nuc_energy_pc.abs() > 1.0e-4 {
            info!("External potential due to point charges exists: {:16.8} Hartree", &nuc_energy_pc);
        }
        if nuc_energy_ext_field.abs() > 1.0e-10 {
            info!("External dipole field contribution to nuc energy exists: {:16.8} Hartree", &nuc_energy_ext_field);
        }
        //========================================
        // For emperial dispersion correction
        let mut disp_from_parse_xc = false;
        if let Some(dfadef) = &self.mol.dfadef {
        if dfadef.has_dispersion() {
            disp_from_parse_xc = true;
        }
        }
        let disp_from_ctrl = self.mol.ctrl.empirical_dispersion.is_some();
        if disp_from_ctrl || disp_from_parse_xc {
            // (energy, grad, sigma); fallback to this default value if dftd evaluation fails
            let default_disp = (0.0, None, None);
            let (engy_disp, grad_disp, sigma_disp) = dftd(self).unwrap_or(default_disp);

            let disp_name = if disp_from_parse_xc {
                self.mol.dfadef.as_ref().unwrap().get_dispersion().unwrap().func.clone()
            } else {
                self.mol.ctrl.empirical_dispersion.clone().unwrap()
            };
            debug!("The empirical dispersion energy of {} is {}.", disp_name.to_uppercase(), engy_disp);
            trace!("{:?}, {:?}", &grad_disp, &sigma_disp);
            self.empirical_dispersion_energy = engy_disp;
            
            // empirical dispersion energy added to the nuc_energy
            self.nuc_energy += engy_disp;
        }
        if !disp_from_ctrl && !disp_from_parse_xc {
            debug!("no empirical dispersion correction is employed");
        }
        //========================================
        // For two-center integrals
        self.ovlp = self.mol.int_ij_matrixupper(String::from("ovlp"));
        match self.mol.ctrl.rel {
            RelativisticMethod::SFX2C => {
                self.h_core = self.mol.generate_sfx2c_hamiltonian();
            }
            _ => {
                self.h_core = self.mol.int_ij_matrixupper(String::from("hcore"));
            }
        }
        //========================================
        // For ghost effective potential
        if self.mol.geom.ghost_ep_path.len() > 0 {
            let tmp_matr = ghost_effective_potential_matrix(
                &self.mol.cint_env, &self.mol.cint_atm, &self.mol.cint_bas,&self.mol.cint_type, self.mol.num_basis,
                &self.mol.geom.ghost_ep_path, &self.mol.geom.ghost_ep_pos
            );
            self.h_core.iter_mut().zip(tmp_matr.iter()).for_each(|(a,b)| *a += *b);
        } else {
            info!("No ghost effective potential");
        }

        // ========================================
        // For the external field
        if let Some(ext_field_dipole) = &self.mol.geom.ext_field.dipole {
            use crate::external_field::ExtField;
            let mut ext_field = ExtField::empty();
            ext_field.dipole = ext_field_dipole.clone().try_into().unwrap();
            let tmp_matr = ext_field.contribution_2c(&self.mol);
            let tmp_matr = tmp_matr.to_matrixupper();
            self.h_core.iter_mut().zip(tmp_matr.iter()).for_each(|(a,b)| *a += *b);
        }

        // For the ghost point charge term
        if self.mol.geom.ghost_pc_chrg.len() > 0 {
            info!("There are {} point charges specified", self.mol.geom.ghost_pc_chrg.len());
            let tmp_matr = self.mol.int_ij_matrixupper(String::from("point charge"));
            self.h_core.iter_mut().zip(tmp_matr.iter()).for_each(|(a,b)| *a += *b);
        } else {
            info!("No ghost point charges");
        }

        //========================================
        // For four-center integrals
        self.ijkl = if self.mol.ctrl.use_auxbas {
            None
        } else {
            Some(self.mol.int_ijkl_erifold4())
        };
        //========================================
        if self.mol.ctrl.print_level>3 {
            println!("The S matrix:");
            self.ovlp.formated_output(5, "lower");
            let mut kin = self.mol.int_ij_matrixupper(String::from("kinetic"));
            println!("The Kinetic matrix:");
            kin.formated_output(5, "lower");
            println!("The H-core matrix:");
            self.h_core.formated_output(5, "lower");
        }

        if self.mol.ctrl.print_level>4 {
            //(ij|kl)
            if let Some(tmp_eris) = &self.ijkl {
                println!("The four-center ERIs:");
                let mut tmp_num = 0;
                let (i_len,j_len) =  (self.mol.num_basis,self.mol.num_basis);
                let (k_len,l_len) =  (self.mol.num_basis,self.mol.num_basis);
                (0..k_len).into_iter().for_each(|k| {
                    (0..k+1).into_iter().for_each(|l| {
                        (0..i_len).into_iter().for_each(|i| {
                            (0..i+1).into_iter().for_each(|j| {
                                if let Some(tmp_value) = tmp_eris.get(&[i,j,k,l]) {
                                    if tmp_value.abs()>1.0e-1 {
                                        println!("I= {:2} J= {:2} K= {:2} L= {:2} Int= {:16.8}",i+1, j+1, k+1,l+1, tmp_value);
                                        tmp_num+= 1;
                                    }
                                } else {
                                    println!("Error: unknown value for eris[{},{},{},{}]",i,j,k,l)
                                };
                            })
                        })
                    })
                });
                println!("Print out {} ERIs", tmp_num);
            }
        }

        // update use_eri if some RI algorithms are specified
        let use_eri_jk = match self.algorithm_jk {
            AlgorithmJK::RiIncore => true,
            AlgorithmJK::Separated(algorithm_j, algorithm_k) => {
                let use_eri_j = algorithm_j == AlgorithmJ::RiIncore;
                let use_eri_k = algorithm_k == AlgorithmK::RiIncore;
                use_eri_j || use_eri_k
            },
            _ => false,
        };
        //let use_eri = true;
        let isdf = if use_eri_jk {self.mol.ctrl.eri_type.eq("ri_v") && self.mol.ctrl.use_isdf} else {false};
        let ri3fn_full = if use_eri_jk {self.mol.ctrl.use_auxbas && !self.mol.ctrl.use_ri_symm} else {false};
        let ri3fn_symm = if use_eri_jk {self.mol.ctrl.use_auxbas && self.mol.ctrl.use_ri_symm} else{false};

        let is_rsh = self.mol.xc_data.is_rsh();

        // For RSH: J uses on-the-fly shell-based ERI (generate_vj_on_the_fly_par).
        // K_full still needs standard rimatr (generate_vk_ri_direct has unresolved RSTSR bug).
        // K_erfc uses rimatr_sr.
        // TODO: when generate_vk_ri_direct is fixed, add `if !is_rsh` to skip standard rimatr for RSH.
        {
            // preparing the three-center integrals in the full format
            self.ri3fn = if ri3fn_full && !isdf {
                Some(self.mol.prepare_ri3fn_for_ri_v_full_rayon())
            } else if self.mol.ctrl.isdf_k_only {
                Some(self.mol.prepare_ri3fn_for_ri_v_full_rayon())
            } else {
                None
            };

            // preparing the three-center integrals using the symmetry
            self.rimatr = if ri3fn_symm && !isdf {
                // initialize the mpi distribution information for num_auxbas and num_baspar
                if let Some(local_mpi) = &mut self.mol.mpi_data {
                    let num_auxbas = self.mol.num_auxbas;
                    let num_basis = self.mol.num_basis;
                    let cint_bas = self.mol.cint_fdqc.clone();
                    local_mpi.distribute_rimatr_tasks(num_auxbas, num_basis, cint_bas);
                }
                let (rimatr, basbas2baspar, baspar2basbas) =
                    self.mol.prepare_rimatr_for_ri_v_mpi_rayon(None, mpi_operator);
                Some((rimatr, basbas2baspar, baspar2basbas))
            } else if ri3fn_symm && isdf {
                None
            } else {
                None
            };
        }

        // build short-range 3c RI integrals for range-separated hybrid (RSH) functionals
        if is_rsh && use_eri_jk {
            let omega = self.mol.xc_data.omega().unwrap();
            info!("Building short-range 3c RI integrals for RSH (omega = {:.4})", omega);
            if ri3fn_symm {
                // Note: SR RI omega is negative in libcint's convention.
                self.rimatr_sr = Some(self.mol.prepare_rimatr_for_ri_v_mpi_rayon(Some(-omega), mpi_operator));
            } else {
                self.ri3fn_sr = Some(self.mol.prepare_ri3fn_sr_rayon(omega));
            }
            info!("  SR 3c integrals built.");
        }

        // initial eigenvectors and eigenvalues
        let (eigenvectors, eigenvalues,n_found)=self.ovlp.to_matrixupperslicemut().lapack_dspevx().unwrap();

        if (n_found as usize) < self.mol.fdqc_bas.len() {
            info!("Overlap matrix is singular:");
            info!("  Using {} out of a possible {} specified basis functions",n_found, self.mol.fdqc_bas.len());
            info!("  Lowest remaining eigenvalue: {:16.8}",eigenvalues[0]);
            self.mol.num_state = n_found as usize;
        } else {
            info!("Overlap matrix is nonsigular:");
            info!("  Lowest eigenvalue: {:16.8} with the total number of basis functions: {:6}",eigenvalues[0],self.mol.num_state);
        };



    }

    pub fn prepare_bse_integrals(&mut self, mpi_operator: &Option<MPIOperator>) {
        // Check if BSE-specific auxiliary basis is set
        let bse_auxbas_path = if let Some(qp) = &self.mol.ctrl.quasiparticle_methods {
            if let Some(path) = &qp.bse_auxbas_path {
                path.clone()
            } else {
                return; // No BSE-specific auxiliary basis set
            }
        } else {
            return;
        };

        info!("Preparing BSE-specific RI integrals with auxiliary basis: {}", bse_auxbas_path);

        // Save original auxiliary basis information
        let original_auxbas_path = self.mol.ctrl.auxbas_path.clone();
        let original_num_auxbas = self.mol.num_auxbas;

        // Switch to BSE auxiliary basis
        self.mol.reload_auxbas(bse_auxbas_path);
        self.num_auxbas_bse = Some(self.mol.num_auxbas);

        // Generate BSE-specific RI integrals
        if self.mol.ctrl.use_ri_symm {
            // Initialize MPI distribution for BSE integrals if needed
            if let Some(local_mpi) = &mut self.mol.mpi_data {
                let num_auxbas = self.mol.num_auxbas;
                let num_basis = self.mol.num_basis;
                let cint_bas = self.mol.cint_fdqc.clone();
                local_mpi.distribute_rimatr_tasks(num_auxbas, num_basis, cint_bas);
            }

            let (rimatr, basbas2baspar, baspar2basbas) =
                self.mol.prepare_rimatr_for_ri_v_mpi_rayon(None, mpi_operator);
            self.rimatr_bse = Some((rimatr, basbas2baspar, baspar2basbas));
        } else {
            self.ri3fn_bse = Some(self.mol.prepare_ri3fn_for_ri_v_full_rayon());
        }

        // Restore original auxiliary basis
        self.mol.reload_auxbas(original_auxbas_path);

        info!("BSE-specific RI integrals prepared successfully");
        info!("BSE auxiliary basis size: {}, Regular auxiliary basis size: {}",
                 self.num_auxbas_bse.unwrap(), original_num_auxbas);
    }

    pub fn prepare_density_grids(&mut self) {

        self.grids = if self.mol.xc_data.is_dfa_scf() || self.mol.ctrl.use_isdf || self.mol.ctrl.initial_guess == "vsap" {
            let grids = Grids::build(&mut self.mol);
            info!("Grid size: {:}", grids.coordinates.len());
            Some(grids)
        } else {None};

        if let Some(grids) = &mut self.grids {
            grids.ao_cutoff = self.mol.ctrl.ao_cutoff;
            if grids.ao_cutoff > 0.0 {
                // sparse path: two-pass batch scan → compressed directly, no dense allocation
                grids.prepare_tabulated_ao_sparse(&self.mol);
            } else {
                // dense path: allocate full AO/AOP, then optionally build non0tab + compressed
                grids.prepare_tabulated_ao(&self.mol);
                grids.build_non0tab(&self.mol);
                grids.build_compressed_storage();
            }
            if self.mol.ctrl.drop_dense_ao && grids.ao_compressed.is_some() {
                if self.mol.ctrl.print_level >= 1 {
                    let (dense_bytes, comp_bytes, _) = grids.memory_footprint();
                    let ratio = if dense_bytes > 0 { comp_bytes as f64 / dense_bytes as f64 * 100.0 } else { 0.0 };
                    info!(" [non0tab] dropping dense ao/aop, dense={:.1}GB, compressed={:.1}GB ({:.1}%)",
                        dense_bytes as f64 / 1e9, comp_bytes as f64 / 1e9, ratio);
                }
                grids.ao = None;
                grids.aop = None;
            }
        }
    }

    pub fn prepare_isdf(&mut self, mpi_operator: &Option<MPIOperator>) {

        let use_eri = self.mol.use_eri;
        let isdf = if use_eri {self.mol.ctrl.eri_type.eq("ri_v") && self.mol.ctrl.use_isdf} else {false};
        let ri3fn_full = if use_eri {self.mol.ctrl.use_auxbas && !self.mol.ctrl.use_ri_symm} else {false};
        let ri3fn_symm = if use_eri {self.mol.ctrl.use_auxbas && self.mol.ctrl.use_ri_symm} else{false};

        if ! isdf {return}
        if let Some(grids) = &self.grids {
            if self.mol.ctrl.use_isdf {
                let init_fock = self.h_core.clone();
                if self.mol.spin_channel==1 {
                    self.hamiltonian = [init_fock,MatrixUpper::new(1,0.0)];
                } else {
                    let init_fock_beta = init_fock.clone();
                    self.hamiltonian = [init_fock,init_fock_beta];
                };
                (self.eigenvectors,self.eigenvalues, self.mol.num_state) = diagonalize_hamiltonian_outside(&self, mpi_operator);
                (self.occupation, self.homo, self.lumo) = generate_occupation_outside(&self);
                self.density_matrix = generate_density_matrix_outside(&self);

                self.grids = Some(prune_by_rho(grids, &self.density_matrix, self.mol.spin_channel));
                
            };


            self.ri3fn_isdf = if ri3fn_full && isdf && !self.mol.ctrl.isdf_new{
                if let Some(grids) = &self.grids {
                    Some(prepare_for_ri_isdf(self.mol.ctrl.isdf_k_mu, &self.mol, &grids))
                } else {
                    None
                }
            } else {
                None
            };

            (self.tab_ao, self.m) = if isdf && self.mol.ctrl.isdf_new{
                if let Some(grids) = &self.grids {
                    let isdf = prepare_m_isdf(self.mol.ctrl.isdf_k_mu, &self.mol, &grids);
                    (Some(isdf.0), Some(isdf.1))
                } else {
                    (None,None)
                }
            } else {
                (None,None)
            };
        } else {
            panic!("SCF.grids should be initialized before the preparation of ISDF");
        }

    }

    pub fn prepare_solvent_calculation(&mut self) {
        if self.mol.ctrl.solvent_enabled {
            self.solvent_static_obj = Some(solvent_prepare(&self.mol));

        }
    }

    pub fn build(mol: Molecule, mpi_operator: &Option<MPIOperator>) -> SCF {

        let mut new_scf = SCF::init_scf(&mol);
        //new_scf.generate_occupation();

        initialize_scf(&mut new_scf, mpi_operator);

        new_scf

    }


    pub fn generate_occupation(&mut self) {
        (self.occupation, self.homo, self.lumo) = generate_occupation_outside(&self);
    }
    
    pub fn generate_density_matrix(&mut self) {
        self.density_matrix = generate_density_matrix_outside(&self);
    }

    pub fn generate_vj_with_erifold4(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {
        let num_basis = self.mol.num_basis;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let mut vj: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
        let dm = &self.density_matrix;
        if let Some(ijkl) = &self.ijkl {
            for i_spin in (0..spin_channel) {
                vj[i_spin] = MatrixUpper::new(npair,0.0f64);
                for jc in (0..num_basis) {
                    for ic in (0..jc) {
                        let dm_ij = dm[i_spin].get1d(ic*num_basis + jc).unwrap() + 
                                        dm[i_spin].get1d(jc*num_basis + ic).unwrap();
                        let ijkl_start = (jc*(jc+1)/2+ic)*npair;
                        let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                        //let reduce_ij = ijkl.get4d_slice([0,0,ic,jc],npair).unwrap();
                        //unsafe{daxpy(npair as i32, dm_ij, reduce_ij, 1, vj[i_spin].to_slice_mut(), 1)};
                        vj[i_spin].data.iter_mut().zip(reduce_ij.iter()).for_each(|(vj_ij,eri_ij)| {
                            *vj_ij += eri_ij*dm_ij
                        });
                        // Rayon parallellism. 
                        //vj[i_spin].data.par_iter_mut().zip(reduce_ij.par_iter()).for_each(|(vj_ij,eri_ij)| {
                        //    *vj_ij += eri_ij*dm_ij
                        //});
                    }
                }
                for jc in (0..num_basis) {
                    let dm_ij = dm[i_spin].get1d(jc*num_basis + jc).unwrap(); 
                    let ijkl_start = (jc*(jc+1)/2+jc)*npair;
                    let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                    //let reduce_ij = ijkl.get4d_slice([0,0,jc,jc],npair).unwrap();
                    //unsafe{daxpy(npair as i32, *dm_ij, reduce_ij, 1, vj[i_spin].to_slice_mut(), 1)};
                    vj[i_spin].data.iter_mut().zip(reduce_ij.iter()).for_each(|(vj_ij,eri_ij)| {
                        *vj_ij += eri_ij*dm_ij
                    });
                    //vj[i_spin].data.par_iter_mut().zip(reduce_ij.par_iter()).for_each(|(vj_ij,eri_ij)| {
                    //    *vj_ij += eri_ij*dm_ij
                    //});
                }
            }
        }
        if scaling_factor!=1.0f64 {
            for i_spin in (0..spin_channel) {
                vj[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };

        //let i_spin:usize = 0;
        //for j in (0..num_basis) {
        //    for i in (0..j+1) {
        //        println!("i: {}, j: {}, vj_ij: {}", i,j, vj[0].get2d([i,j]).unwrap());
        //    }
        //}

        vj
    }
    pub fn generate_vj_with_erifold4_sync(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {
        let num_basis = self.mol.num_basis;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let mut vj: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
        let dm = &self.density_matrix;
        let num_para: usize = if let Some(num_para) = self.mol.ctrl.num_threads {
            num_para
        } else {
            1
        };
        let num_chunck = if num_basis%num_para==0 {
            (num_basis/num_para,num_basis/num_para)
        } else {
            (num_basis/num_para+1,num_basis-(num_basis/num_para+1)*(num_para-1))
        };
        if let Some(ijkl) = &self.ijkl {
            for i_spin in (0..spin_channel) {
                vj[i_spin] = MatrixUpper::new(npair,0.0f64);
                scope(|s_thread| {
                    let (tx_jc,rx_jc) = unbounded();
                    for f in (0..num_para-1) {
                        let jc_start_thread = f*num_chunck.0;
                        let jc_end_thread = jc_start_thread + num_chunck.0;
                        let tx_jc_thread = tx_jc.clone();
                        let handle = s_thread.spawn(move |_| {
                            let mut vj_thread = MatrixUpper::new(npair,0.0f64);
                            for jc in (jc_start_thread..jc_end_thread) {
                                for ic in (0..jc) {
                                    let dm_ij = dm[i_spin].get1d(ic*num_basis + jc).unwrap() + 
                                                    dm[i_spin].get1d(jc*num_basis + ic).unwrap();
                                    let ijkl_start = (jc*(jc+1)/2+ic)*npair;
                                    let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                                    vj_thread.data.iter_mut().zip(reduce_ij.iter()).for_each(|(vj_ij,eri_ij)| {
                                        *vj_ij += eri_ij*dm_ij
                                    });
                                }
                                let dm_ij = dm[i_spin].get1d(jc*num_basis + jc).unwrap(); 
                                let ijkl_start = (jc*(jc+1)/2+jc)*npair;
                                let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                                vj_thread.data.iter_mut().zip(reduce_ij.iter()).for_each(|(vj_ij,eri_ij)| {
                                    *vj_ij += eri_ij*dm_ij
                                });
                            }
                            tx_jc_thread.send(vj_thread).unwrap();
                        });
                    }
                    let jc_start_thread = (num_para-1)*num_chunck.0;
                    let jc_end_thread = jc_start_thread + num_chunck.1;
                    let tx_jc_thread = tx_jc;
                    let handle = s_thread.spawn(move |_| {
                        let mut vj_thread = MatrixUpper::new(npair,0.0f64);
                        for jc in (jc_start_thread..jc_end_thread) {
                            for ic in (0..jc) {
                                let dm_ij = dm[i_spin].get1d(ic*num_basis + jc).unwrap() + 
                                                dm[i_spin].get1d(jc*num_basis + ic).unwrap();
                                let ijkl_start = (jc*(jc+1)/2+ic)*npair;
                                let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                                vj_thread.data.iter_mut().zip(reduce_ij.iter()).for_each(|(vj_ij,eri_ij)| {
                                    *vj_ij += eri_ij*dm_ij
                                });
                            }
                            let dm_ij = dm[i_spin].get1d(jc*num_basis + jc).unwrap(); 
                            let ijkl_start = (jc*(jc+1)/2+jc)*npair;
                            let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                            vj_thread.data.iter_mut().zip(reduce_ij.iter()).for_each(|(vj_ij,eri_ij)| {
                                *vj_ij += eri_ij*dm_ij
                            });
                        }
                        tx_jc_thread.send(vj_thread).unwrap();
                    });
                    for received in rx_jc {
                        vj[i_spin].data.iter_mut()
                            .zip(received.data).for_each(|(i,j)| {*i += j});
                    }

                }).unwrap();
            }
        }
        if scaling_factor!=1.0f64 {
            for i_spin in (0..spin_channel) {
                vj[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };

        vj
    }

    pub fn generate_vj_on_the_fly(&self) -> Vec<MatrixUpper<f64>>{
        let num_shell = self.mol.cint_bas.len();
        //let num_shell = self.mol.cint_fdqc.len();
        let num_basis = self.mol.num_basis;
        let spin_channel = self.mol.spin_channel;
        let dm = &self.density_matrix;
        let mut vj: Vec<MatrixUpper<f64>> = vec![];
        let mol = &self.mol;
        for i_spin in 0..spin_channel{
            let mut vj_i = MatrixFull::new([num_basis, num_basis], 0.0);
            let mut dm_s = &self.density_matrix[i_spin];
            for k in 0..num_shell{
                let bas_start_k = mol.cint_fdqc[k][0];
                let bas_len_k = mol.cint_fdqc[k][1];

                for l in 0..num_shell{
                    let bas_start_l = mol.cint_fdqc[l][0];
                    let bas_len_l = mol.cint_fdqc[l][1];
                    let mut klij = &mol.int_ijkl_given_kl(k, l);
                    
                    // ao_k & ao_l are index of ao
                    let mut sum =0.0;
                    for ao_k in bas_start_k..bas_start_k+bas_len_k{
                        for ao_l in bas_start_l..bas_start_l+bas_len_l{
                            let mut sum =0.0;
                            let mut index_k = ao_k-bas_start_k;
                            let mut index_l = ao_l-bas_start_l;
                            let mut eri_cd = &klij[index_l * bas_len_k + index_k];
                            let eri_full = eri_cd.to_matrixupper().to_matrixfull().unwrap();
                            let mut v_cd = MatrixFull::new([num_basis, num_basis], 0.0);
                            v_cd.data.iter_mut().zip(dm_s.data.iter()).zip(eri_full.data.iter()).for_each(|((v,p),eri)|{
                                *v = *p * *eri
                            });

                            v_cd.data.iter().for_each(|x|{
                                sum += *x
                            });
                            vj_i[(ao_k,ao_l)] = sum;
                        }
                    }
                }
            }

            vj.push(vj_i.to_matrixupper());
        }
        if spin_channel == 1{
            vj.push(MatrixUpper::new(1, 0.0));       
        }
        vj
    }

    pub fn generate_vj_on_the_fly_par_old(&self) -> Vec<MatrixUpper<f64>>{
        //utilities::omp_set_num_threads_wrapper(1);
        let num_shell = self.mol.cint_bas.len();
        let num_basis = self.mol.num_basis;
        let spin_channel = self.mol.spin_channel;
        let dm = &self.density_matrix;
        let mut vj: Vec<MatrixUpper<f64>> = vec![];
        let mol = &self.mol;
        for i_spin in 0..spin_channel{
            let mut vj_i = MatrixFull::new([num_basis, num_basis], 0.0);
            let mut dm_s = &self.density_matrix[i_spin];
            let par_tasks = utilities::balancing(num_shell*num_shell, rayon::current_num_threads());
            let (sender, receiver) = channel();
            let mut index = vec![0usize; num_shell*num_shell];
            for i in 0..num_shell*num_shell{index[i] = i};
            index.par_iter().for_each_with(sender,|s,i|{
                let k = i/num_shell;
                let bas_start_k = mol.cint_fdqc[k][0];
                let bas_len_k = mol.cint_fdqc[k][1];
                let l = i%num_shell;
                let bas_start_l = mol.cint_fdqc[l][0];
                let bas_len_l = mol.cint_fdqc[l][1];
                let mut klij = &mol.int_ijkl_given_kl(k, l);
                let mut sum =0.0;
                //let mut out = vec![(0.0, 0usize, 0usize); ];
                let mut out:Vec<(f64, usize, usize)> = Vec::new();
                    for ao_k in bas_start_k..bas_start_k+bas_len_k{
                        for ao_l in bas_start_l..bas_start_l+bas_len_l{
                            let mut sum =0.0;
                            let mut index_k = ao_k-bas_start_k;
                            let mut index_l = ao_l-bas_start_l;
                            let mut eri_cd = &klij[index_l * bas_len_k + index_k];
                            let eri_full = eri_cd.to_matrixupper().to_matrixfull().unwrap();
                            let mut v_cd = MatrixFull::new([num_basis, num_basis], 0.0);
                            v_cd.data.iter_mut().zip(dm_s.data.iter()).zip(eri_full.data.iter()).for_each(|((v,p),eri)|{
                                *v = *p * *eri
                            });

                            v_cd.data.iter().for_each(|x|{
                                sum += *x
                            });
                            out.push((sum, ao_k, ao_l));
                        }
                    }
                s.send(out).unwrap();
            });
            receiver.into_iter().for_each(|out_vec| {
                out_vec.iter().for_each(|(value,index_k,index_l)|{
                    vj_i[(*index_k, *index_l)] = *value
                })
            });


            vj.push(vj_i.to_matrixupper());
        }
        if spin_channel == 1{
            vj.push(MatrixUpper::new(1, 0.0));
        }
        vj

    }

    pub fn generate_vj_on_the_fly_par_new(&self) -> Vec<MatrixUpper<f64>>{
        let num_shell = self.mol.cint_bas.len();
        let num_basis = self.mol.num_basis;
        let spin_channel = self.mol.spin_channel;
        let dm = &self.density_matrix;
        let mut vj: Vec<MatrixUpper<f64>> = vec![];
        let mol = &self.mol;
        //utilities::omp_set_num_threads_wrapper(1);
        for i_spin in 0..spin_channel{
            let mut vj_i = MatrixFull::new([num_basis, num_basis], 0.0);
            let dm_s = &self.density_matrix[i_spin];
            let par_tasks = utilities::balancing(num_shell*num_shell, rayon::current_num_threads());
            let (sender, receiver) = channel();
            let mut index = Vec::new();
            for l in 0..num_shell {
                for k in 0..l+1 {
                    index.push((k,l))
                }
            };
            index.par_iter().for_each_with(sender,|s,(k,l)|{
                let bas_start_k = mol.cint_fdqc[*k][0];
                let bas_len_k = mol.cint_fdqc[*k][1];
                let bas_start_l = mol.cint_fdqc[*l][0];
                let bas_len_l = mol.cint_fdqc[*l][1];

                let klij = mol.int_ijkl_given_kl_v02(*k, *l);
                let mut sum =0.0;
                //let mut out = vec![(0.0, 0usize, 0usize); ];
                //let mut out:Vec<(f64, usize, usize)> = Vec::new();
                let mut out = MatrixFull::new([bas_len_k, bas_len_l],0.0);
                //for ao_k in bas_start_k..bas_start_k+bas_len_k{
                //    for ao_l in bas_start_l..bas_start_l+bas_len_l{
                out.iter_columns_full_mut().enumerate().for_each(|(loc_l,x)|{
                    x.iter_mut().enumerate().for_each(|(loc_k,elem)|{
                        let ao_k = loc_k + bas_start_k;
                        let ao_l = loc_l + bas_start_l;
                        let eri_cd = klij.get(&[loc_k, loc_l]).unwrap();
                        let mut sum = dm_s.iter_matrixupper().unwrap()
                            .zip(eri_cd.iter_matrixupper().unwrap())
                            .fold(0.0,|sum, (p,eri)| {
                            sum + *p * *eri
                        });

                        let mut diagonal = dm_s.iter_diagonal().unwrap().zip(eri_cd.iter_diagonal().unwrap()).fold(0.0,|diagonal, (p,eri)| {
                            diagonal + *p * *eri
                        });

                        sum = sum*2.0 - diagonal;

                        *elem = sum;
                    })
                });
                s.send((out,*k,*l)).unwrap();
            });
            receiver.into_iter().for_each(|(out,k,l)| {
                let bas_start_k = mol.cint_fdqc[k][0];
                let bas_len_k = mol.cint_fdqc[k][1];
                let bas_start_l = mol.cint_fdqc[l][0];
                let bas_len_l = mol.cint_fdqc[l][1];
                vj_i.copy_from_matr(bas_start_k..bas_start_k+bas_len_k, bas_start_l..bas_start_l+bas_len_l, 
                    &out, 0..bas_len_k,0..bas_len_l);
                //vj_i.iter_submatrix_mut(bas_start_k..bas_start_k+bas_len_k, bas_start_l..bas_start_l+bas_len_l).zip(out.iter())
                //    .for_each(|(to, from)| {*to = *from});
            });
            

            vj.push(vj_i.to_matrixupper());
        }
        if spin_channel == 1{
            vj.push(MatrixUpper::new(1, 0.0));       
        }
        vj
    }

    pub fn generate_vj_on_the_fly_par(&self) -> Vec<MatrixUpper<f64>> {
        vj_on_the_fly_par(&self.mol, &self.density_matrix)
        // IGOR MARK: still has bugs in the batch_by_batch version
        //vj_on_the_fly_par_batch_by_batch(&self.mol, &self.density_matrix)
    }


    pub fn generate_vk_with_erifold4(&mut self, scaling_factor: f64) -> Vec<MatrixFull<f64>> {
        let num_basis = self.mol.num_basis;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let mut vk: Vec<MatrixFull<f64>> = vec![MatrixFull::new([1,1],0.0f64),MatrixFull::new([1,1],0.0f64)];
        let dm = &self.density_matrix;
        if let Some(ijkl) = &self.ijkl {
            for i_spin in (0..spin_channel) {
                vk[i_spin] = MatrixFull::new([num_basis,num_basis],0.0f64);
                for jc in (0..num_basis) {
                    for ic in (0..jc) {
                        let ijkl_start = (jc*(jc+1)/2+ic)*npair;
                        let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                        //let reduce_ij = ijkl.get4d_slice([0,0,ic,jc],npair).unwrap();
                        let dm_ic = dm[i_spin].get1d_slice(ic*num_basis,num_basis).unwrap();
                        let dm_jc = dm[i_spin].get1d_slice(jc*num_basis,num_basis).unwrap();
                        let mut vk_ic = vk[i_spin].get1d_slice_mut(ic*num_basis,num_basis).unwrap();
                        let mut kl = 0_usize;
                        for k in (0..num_basis) {
                            // The psuedo-code for the next several ten lines
                            //for l in (0..k) {
                            //    vk_ic[l] += reduce_ij[kl] *dm_jc[k];
                            //    vk_ic[k] += reduce_ij[kl] *dm_jc[l];
                            //    kl += 1;
                            //}
                            vk_ic[..k].iter_mut()
                                .zip(reduce_ij[kl..kl+k].iter())
                                .for_each(|(i,j)| {*i += j*dm_jc[k]});
                            vk_ic[k] += reduce_ij[kl..kl+k]
                                .iter()
                                .zip(dm_jc[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                            kl += k;
                            //============================================
                            vk_ic[k] += reduce_ij[kl] *dm_jc[k];
                            kl += 1;
                        }
                        let mut vk_jc = vk[i_spin].get1d_slice_mut(jc*num_basis,num_basis).unwrap();
                        let mut kl = 0_usize;
                        for k in (0..num_basis) {
                            // The psuedo-code for the next several ten lines
                            //for l in (0..k) {
                            //    vk_jc[l] += reduce_ij[kl] *dm_ic[k];
                            //    vk_jc[k] += reduce_ij[kl] *dm_ic[l];
                            //    kl += 1;
                            //}
                            vk_jc[..k].iter_mut()
                                .zip(reduce_ij[kl..kl+k].iter())
                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                            vk_jc[k] += reduce_ij[kl..kl+k]
                                .iter()
                                .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                            kl += k;
                            //============================================
                            vk_jc[k] += reduce_ij[kl] *dm_ic[k];
                            kl += 1;
                        }
                    }
                }
                for ic in (0..num_basis) {
                    let ijkl_start = (ic*(ic+1)/2+ic)*npair;
                    let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                    let dm_ic = dm[i_spin].get1d_slice(ic*num_basis,num_basis).unwrap();
                    let mut vk_ic = vk[i_spin].get1d_slice_mut(ic*num_basis,num_basis).unwrap();
                    let mut kl = 0_usize;
                    for k in (0..num_basis) {
                        // The psuedo-code for the next several ten lines
                        //for l in (0..k) {
                        //    vk_ic[l] += reduce_ij[kl] *dm_ic[k];
                        //    vk_ic[k] += reduce_ij[kl] *dm_ic[l];
                        //    kl += 1;
                        //}
                        vk_ic[..k].par_iter_mut()
                            .zip(reduce_ij[kl..kl+k].par_iter())
                            .for_each(|(i,j)| {*i += j*dm_ic[k]});
                        vk_ic[k] += reduce_ij[kl..kl+k]
                            .iter()
                            .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                        kl += k;
                        //=================================================
                        vk_ic[k] += reduce_ij[kl] *dm_ic[k];
                        kl += 1;
                    }
                }
            }
        }
        if scaling_factor!=1.0f64 {
            for i_spin in (0..spin_channel) {
                vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };
        vk
    }
    pub fn generate_vk_with_erifold4_v02(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {
        let num_basis = self.mol.num_basis;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
        let dm = &self.density_matrix;
        if let Some(ijkl) = &self.ijkl {
            for i_spin in (0..spin_channel) {
                vk[i_spin] = MatrixUpper::new(npair,0.0f64);
                for jc in (0..num_basis) {
                    for ic in (0..jc) {
                        let ijkl_start = (jc*(jc+1)/2+ic)*npair;
                        let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                        //let reduce_ij = ijkl.get4d_slice([0,0,ic,jc],npair).unwrap();
                        let dm_ic = dm[i_spin].get1d_slice(ic*num_basis,num_basis).unwrap();
                        let dm_jc = dm[i_spin].get1d_slice(jc*num_basis,num_basis).unwrap();
                        let mut vk_ic = vk[i_spin].get1d_slice_mut(ic*(ic+1)/2,ic+1).unwrap();
                        let mut kl = 0_usize;
                        // The psuedo-code for the next several ten lines
                        //for k in (0..num_basis) {
                        //    for l in (0..k) {
                        //        vk_ic[l] += reduce_ij[kl] *dm_jc[k];
                        //        vk_ic[k] += reduce_ij[kl] *dm_jc[l];
                        //        kl += 1;
                        //    }
                        //}
                        //    vk_ic[k] += reduce_ij[kl] *dm_jc[k];
                        //    kl += 1;
                        //==============================================
                        for k in (0..num_basis) {
                            if k<=ic {
                                vk_ic[..k].iter_mut()
                                    .zip(reduce_ij[kl..kl+k].iter())
                                    .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                vk_ic[k] += reduce_ij[kl..kl+k]
                                    .iter()
                                    .zip(dm_jc[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                kl += k;
                                vk_ic[k] += reduce_ij[kl] *dm_jc[k];
                                kl += 1;
                            } else {
                                vk_ic[..ic+1].iter_mut()
                                    .zip(reduce_ij[kl..kl+ic+1].iter())
                                    .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                kl += k+1;
                            }
                            //if ic==4 && k==35 {println!("{}",kl)};
                        }
                        //=================================================
                        // try rayon parallel version
                        //for k in (0..num_basis) {
                        //    if k<=ic {
                        //        vk_ic[..k].iter_mut()
                        //            .zip(reduce_ij[kl..kl+k].iter())
                        //            .for_each(|(i,j)| {*i += j*dm_jc[k]});
                        //        vk_ic[k] += reduce_ij[kl..kl+k]
                        //            .par_iter()
                        //            .zip(dm_jc[..k].par_iter()).map(|(i,j)| i*j).sum::<f64>();
                        //        kl += k;
                        //        vk_ic[k] += reduce_ij[kl] *dm_jc[k];
                        //        kl += 1;我们的描述子是1*10
                        //    } else {
                        //        vk_ic[..ic+1].iter_mut()
                        //            .zip(reduce_ij[kl..kl+ic+1].iter())
                        //            .for_each(|(i,j)| {*i += j*dm_jc[k]});
                        //        kl += k+1;
                        //    }
                        //}
                        //=================================================
                        let mut vk_jc = vk[i_spin].get1d_slice_mut(jc*(jc+1)/2,jc+1).unwrap();
                        let mut kl = 0_usize;
                        // The psuedo-code for the next several ten lines
                        //for k in (0..num_basis) {
                        //    for l in (0..k) {
                        //        vk_jc[l] += reduce_ij[kl] *dm_ic[k];
                        //        vk_jc[k] += reduce_ij[kl] *dm_ic[l];
                        //        kl += 1;
                        //    }
                        //    vk_jc[k] += reduce_ij[kl] *dm_ic[k];
                        //    kl += 1;
                        //}
                        for k in (0..num_basis) {
                            if k<=jc {
                                vk_jc[..k].iter_mut()
                                    .zip(reduce_ij[kl..kl+k].iter())
                                    .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                vk_jc[k] += reduce_ij[kl..kl+k]
                                    .iter()
                                    .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                kl += k;
                                vk_jc[k] += reduce_ij[kl] *dm_ic[k];
                                kl += 1;
                            } else {
                                vk_jc[..jc+1].iter_mut()
                                    .zip(reduce_ij[kl..kl+jc+1].iter())
                                    .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                kl += k+1;
                            }
                        }
                        //=================================================
                    }
                }
                for ic in (0..num_basis) {
                    let ijkl_start = (ic*(ic+1)/2+ic)*npair;
                    let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                    //let reduce_ij = ijkl.get4d_slice([0,0,ic,ic],npair).unwrap();
                    let dm_ic = dm[i_spin].get1d_slice(ic*num_basis,num_basis).unwrap();
                    let mut vk_ic = vk[i_spin].get1d_slice_mut(ic*(ic+1)/2,ic+1).unwrap();
                    let mut kl = 0_usize;
                    // The psuedo-code for the next several ten lines
                    //for k in (0..num_basis) {
                    //    for l in (0..k) {
                    //        vk_ic[l] += reduce_ij[kl] *dm_ic[k];
                    //        vk_ic[k] += reduce_ij[kl] *dm_ic[l];
                    //        kl += 1;
                    //    }
                    //    vk_ic[k] += reduce_ij[kl] *dm_ic[k];
                    //    kl += 1;
                    //}
                    for k in (0..num_basis) {
                        if k<=ic {
                            vk_ic[..k].iter_mut()
                                .zip(reduce_ij[kl..kl+k].iter())
                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                            vk_ic[k] += reduce_ij[kl..kl+k]
                                .iter()
                                .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                            kl += k;
                            vk_ic[k] += reduce_ij[kl] *dm_ic[k];
                            kl += 1;
                        } else {
                            vk_ic[..ic+1].iter_mut()
                                .zip(reduce_ij[kl..kl+ic+1].iter())
                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                            kl += k+1;
                        }
                    }
                    //=================================================
                }
            }
        }
        if scaling_factor!=1.0f64 {
            for i_spin in (0..spin_channel) {
                vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };
        vk
    }
    pub fn generate_vk_with_erifold4_sync(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {
        let num_basis = self.mol.num_basis;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
        let dm = &self.density_matrix;
        let num_para: usize = if let Some(num_para) = self.mol.ctrl.num_threads {
            num_para
        } else {
            1
        };
        let num_chunck = if num_basis%num_para==0 {
            (num_basis/num_para,num_basis/num_para)
        } else {
            (num_basis/num_para+1,num_basis-(num_basis/num_para+1)*(num_para-1))
        };
        println!("num_threads: ({},{}),num_chunck: ({},{})",
                num_para,num_basis,
                num_chunck.0,
                num_chunck.1);
                //if num_basis%num_para==0 {num_chunck.0} else {num_basis%num_para});
        if let Some(ijkl) = &self.ijkl {
            for i_spin in (0..spin_channel) {
                vk[i_spin] = MatrixUpper::new(npair,0.0f64);
                for jc in (0..num_basis) {
                    for ic in (0..jc) {
                        scope(|s_thread| {
                            let ijkl_start = (jc*(jc+1)/2+ic)*npair;
                            let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                            //let reduce_ij = ijkl.get4d_slice([0,0,ic,jc],npair).unwrap();
                            let dm_ic = dm[i_spin].get1d_slice(ic*num_basis,num_basis).unwrap();
                            let dm_jc = dm[i_spin].get1d_slice(jc*num_basis,num_basis).unwrap();
                            let (tx_ic,rx_ic) = unbounded();
                            let (tx_jc,rx_jc) = unbounded();
                            //println!("Main thread: {:?}",thread::current().id());
                            for f in (0..num_para-1) {
                                let ic_thread = ic;
                                let jc_thread = jc;
                                let k_start_thread = f*num_chunck.0;
                                let k_end_thread = k_start_thread + num_chunck.0;
                                let tx_ic_thread = tx_ic.clone();
                                let tx_jc_thread = tx_jc.clone();
                                let mut kl_thread = k_start_thread*(k_start_thread+1)/2;
                                let handle = s_thread.spawn(move |_| {
                                    let mut vk_ic_thread = vec![0.0;ic_thread+1];
                                    let mut vk_jc_thread = vec![0.0;jc_thread+1];
                                    //let handle_thread = thread::current();
                                    //if ic_thread == 4 {println!("Fork thread: {:?}, kl: {}, k: ({}, {})",handle_thread.id(),kl_thread, k_start_thread, k_end_thread)};
                                    for k in (k_start_thread..k_end_thread) {
                                        let mut kl_jc_thread = kl_thread;
                                        if k<=ic_thread {
                                            vk_ic_thread[..k].iter_mut()
                                                .zip(reduce_ij[kl_thread..kl_thread+k].iter())
                                                .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                            vk_ic_thread[k] += reduce_ij[kl_thread..kl_thread+k]
                                                .iter()
                                                .zip(dm_jc[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                            kl_thread += k;
                                            vk_ic_thread[k] += reduce_ij[kl_thread] *dm_jc[k];
                                            kl_thread += 1;
                                        } else {
                                            vk_ic_thread[..ic+1].iter_mut()
                                                .zip(reduce_ij[kl_thread..kl_thread+ic+1].iter())
                                                .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                            kl_thread += k+1;
                                        }
                                        if k<=jc_thread {
                                            vk_jc_thread[..k].iter_mut()
                                                .zip(reduce_ij[kl_jc_thread..kl_jc_thread+k].iter())
                                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                            vk_jc_thread[k] += reduce_ij[kl_jc_thread..kl_jc_thread+k]
                                                .iter()
                                                .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                            kl_jc_thread += k;
                                            vk_jc_thread[k] += reduce_ij[kl_jc_thread] *dm_ic[k];
                                            kl_jc_thread += 1;
                                        } else {
                                            vk_jc_thread[..jc+1].iter_mut()
                                                .zip(reduce_ij[kl_jc_thread..kl_jc_thread+jc_thread+1].iter())
                                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                            kl_jc_thread += k+1;
                                        }
                                    }
                                    //if ic_thread == 4 {println!("Fork thread: {:?}, kl: {}, k: ({}, {})",handle_thread.id(),kl_thread, k_start_thread, k_end_thread)};
                                    tx_ic_thread.send(vk_ic_thread).unwrap();
                                    tx_jc_thread.send(vk_jc_thread).unwrap();
                                });
                                //handles.push(handle);
                            }
                            let ic_thread = ic;
                            let jc_thread = jc;
                            let k_start_thread = (num_para-1)*num_chunck.0;
                            let k_end_thread = k_start_thread+num_chunck.1;
                            //let reduce_ij_thread = reduce_ij.clone();
                            //let dm_ic_thread = dm_ic.clone();
                            //let dm_jc_thread = dm_jc.clone();
                            let mut kl_thread = k_start_thread*(k_start_thread+1)/2;
                            let tx_ic_thread = tx_ic;
                            let tx_jc_thread = tx_jc;
                            let handle = s_thread.spawn(move |_| {
                                let mut vk_ic_thread = vec![0.0;ic_thread+1];
                                let mut vk_jc_thread = vec![0.0;jc_thread+1];
                                //let handle_thread = thread::current();
                                //if ic_thread == 4 {println!("Fork thread: {:?}, kl: {}, k: ({}, {})",handle_thread.id(),kl_thread, k_start_thread, k_end_thread)};
                                for k in (k_start_thread..k_end_thread) {
                                    let mut kl_jc_thread = kl_thread;
                                    if k<=ic_thread {
                                        vk_ic_thread[..k].iter_mut()
                                            .zip(reduce_ij[kl_thread..kl_thread+k].iter())
                                            .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                        vk_ic_thread[k] += reduce_ij[kl_thread..kl_thread+k]
                                            .iter()
                                            .zip(dm_jc[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                        kl_thread += k;
                                        vk_ic_thread[k] += reduce_ij[kl_thread] *dm_jc[k];
                                        kl_thread += 1;
                                    } else {
                                        vk_ic_thread[..ic+1].iter_mut()
                                            .zip(reduce_ij[kl_thread..kl_thread+ic+1].iter())
                                            .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                        kl_thread += k+1;
                                    }
                                    if k<=jc_thread {
                                        vk_jc_thread[..k].iter_mut()
                                            .zip(reduce_ij[kl_jc_thread..kl_jc_thread+k].iter())
                                            .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                        vk_jc_thread[k] += reduce_ij[kl_jc_thread..kl_jc_thread+k]
                                            .iter()
                                            .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                        kl_jc_thread += k;
                                        vk_jc_thread[k] += reduce_ij[kl_jc_thread] *dm_ic[k];
                                        kl_jc_thread += 1;
                                    } else {
                                        vk_jc_thread[..jc+1].iter_mut()
                                            .zip(reduce_ij[kl_jc_thread..kl_jc_thread+jc_thread+1].iter())
                                            .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                        kl_jc_thread += k+1;
                                    }
                                }
                                //if ic_thread == 4 {println!("Fork thread: {:?}, kl: {}, k: ({}, {})",handle_thread.id(),kl_thread, k_start_thread, k_end_thread)};
                                tx_ic_thread.send(vk_ic_thread).unwrap();
                                tx_jc_thread.send(vk_jc_thread).unwrap();
                            });
                            //handles.push(handle);
                            {
                                let mut vk_ic = vk[i_spin].get1d_slice_mut(ic*(ic+1)/2,ic+1).unwrap();
                                for received in rx_ic {
                                    vk_ic.iter_mut()
                                        .zip(received)
                                        .for_each(|(i,j)| {*i += j});
                                }
                            }
                            {
                                let mut vk_jc = vk[i_spin].get1d_slice_mut(jc*(jc+1)/2,jc+1).unwrap();
                                for received in rx_jc {
                                    vk_jc.iter_mut()
                                        .zip(received)
                                        .for_each(|(i,j)| {*i += j});
                                }
                            }
                        }).unwrap();
                    }
                }
                for ic in (0..num_basis) {
                    let ijkl_start = (ic*(ic+1)/2+ic)*npair;
                    let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                    //let reduce_ij = ijkl.get4d_slice([0,0,ic,ic],npair).unwrap();
                    let dm_ic = dm[i_spin].get1d_slice(ic*num_basis,num_basis).unwrap();
                    let mut vk_ic = vk[i_spin].get1d_slice_mut(ic*(ic+1)/2,ic+1).unwrap();
                    let mut kl = 0_usize;
                    // The psuedo-code for the next several ten lines
                    //for k in (0..num_basis) {
                    //    for l in (0..k) {
                    //        vk_ic[l] += reduce_ij[kl] *dm_ic[k];
                    //        vk_ic[k] += reduce_ij[kl] *dm_ic[l];
                    //        kl += 1;
                    //    }
                    //    vk_ic[k] += reduce_ij[kl] *dm_ic[k];
                    //    kl += 1;
                    //}
                    for k in (0..num_basis) {
                        if k<=ic {
                            vk_ic[..k].iter_mut()
                                .zip(reduce_ij[kl..kl+k].iter())
                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                            vk_ic[k] += reduce_ij[kl..kl+k]
                                .iter()
                                .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                            kl += k;
                            vk_ic[k] += reduce_ij[kl] *dm_ic[k];
                            kl += 1;
                        } else {
                            vk_ic[..ic+1].iter_mut()
                                .zip(reduce_ij[kl..kl+ic+1].iter())
                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                            kl += k+1;
                        }
                    }
                    //=================================================
                }
            }
        }
        if scaling_factor!=1.0f64 {
            for i_spin in (0..spin_channel) {
                vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };
        vk
    }
    pub fn generate_vk_with_erifold4_sync_v02(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {
        let num_basis = self.mol.num_basis;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
        let dm = &self.density_matrix;
        let num_para: usize = if let Some(num_para) = self.mol.ctrl.num_threads {
            num_para
        } else {
            1
        };
        let num_chunck = if num_basis%num_para==0 {
            (num_basis/num_para,num_basis/num_para)
        } else {
            (num_basis/num_para+1,num_basis-(num_basis/num_para+1)*(num_para-1))
        };
        //println!("num_threads: ({},{}),num_chunck: ({},{})",
        //        num_para,num_basis,
        //        num_chunck.0,
        //        num_chunck.1);
        if let Some(ijkl) = &self.ijkl {
            for i_spin in (0..spin_channel) {
                vk[i_spin] = MatrixUpper::new(npair,0.0f64);
                scope(|s_thread| {
                    let (tx_jc,rx_jc) = unbounded();
                    for f in (0..num_para-1) {
                        let jc_start_thread = f*num_chunck.0;
                        let jc_end_thread = jc_start_thread + num_chunck.0;
                        let tx_jc_thread = tx_jc.clone();
                        let handle = s_thread.spawn(move |_| {
                            let mut vk_thread =  MatrixUpper::new(npair,0.0f64);
                            for jc in (jc_start_thread..jc_end_thread) {
                                for ic in (0..jc) {
                                    let ijkl_start = (jc*(jc+1)/2+ic)*npair;
                                    let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                                    //let reduce_ij = ijkl.get4d_slice([0,0,ic,jc],npair).unwrap();
                                    let dm_ic = dm[i_spin].get1d_slice(ic*num_basis,num_basis).unwrap();
                                    let dm_jc = dm[i_spin].get1d_slice(jc*num_basis,num_basis).unwrap();
                                    let mut vk_ic = vk_thread.get1d_slice_mut(ic*(ic+1)/2,ic+1).unwrap();
                                    let mut kl = 0_usize;
                                    for k in (0..num_basis) {
                                        if k<=ic {
                                            vk_ic[..k].iter_mut()
                                                .zip(reduce_ij[kl..kl+k].iter())
                                                .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                            vk_ic[k] += reduce_ij[kl..kl+k]
                                                .iter()
                                                .zip(dm_jc[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                            kl += k;
                                            vk_ic[k] += reduce_ij[kl] *dm_jc[k];
                                            kl += 1;
                                        } else {
                                            vk_ic[..ic+1].iter_mut()
                                                .zip(reduce_ij[kl..kl+ic+1].iter())
                                                .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                            kl += k+1;
                                        }
                                    }
                                    //=================================================
                                    let mut vk_jc = vk_thread.get1d_slice_mut(jc*(jc+1)/2,jc+1).unwrap();
                                    let mut kl = 0_usize;
                                    for k in (0..num_basis) {
                                        if k<=jc {
                                            vk_jc[..k].iter_mut()
                                                .zip(reduce_ij[kl..kl+k].iter())
                                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                            vk_jc[k] += reduce_ij[kl..kl+k]
                                                .iter()
                                                .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                            kl += k;
                                            vk_jc[k] += reduce_ij[kl] *dm_ic[k];
                                            kl += 1;
                                        } else {
                                            vk_jc[..jc+1].iter_mut()
                                                .zip(reduce_ij[kl..kl+jc+1].iter())
                                                .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                            kl += k+1;
                                        }
                                    }
                                    //=================================================
                                }
                                let ijkl_start = (jc*(jc+1)/2+jc)*npair;
                                let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                                //let reduce_ij = ijkl.get4d_slice([0,0,ic,ic],npair).unwrap();
                                let dm_jc = dm[i_spin].get1d_slice(jc*num_basis,num_basis).unwrap();
                                let mut vk_jc = vk_thread.get1d_slice_mut(jc*(jc+1)/2,jc+1).unwrap();
                                let mut kl = 0_usize;
                                for k in (0..num_basis) {
                                    if k<=jc {
                                        vk_jc[..k].iter_mut()
                                            .zip(reduce_ij[kl..kl+k].iter())
                                            .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                        vk_jc[k] += reduce_ij[kl..kl+k]
                                            .iter()
                                            .zip(dm_jc[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                        kl += k;
                                        vk_jc[k] += reduce_ij[kl] *dm_jc[k];
                                        kl += 1;
                                    } else {
                                        vk_jc[..jc+1].iter_mut()
                                            .zip(reduce_ij[kl..kl+jc+1].iter())
                                            .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                        kl += k+1;
                                    }
                                }
                            }
                            tx_jc_thread.send(vk_thread);
                        });
                    }
                    let jc_start_thread = (num_para-1)*num_chunck.0;
                    let jc_end_thread = jc_start_thread + num_chunck.1;
                    let tx_jc_thread = tx_jc;
                    let handle = s_thread.spawn(move |_| {
                        let mut vk_thread =  MatrixUpper::new(npair,0.0f64);
                        for jc in (jc_start_thread..jc_end_thread) {
                            for ic in (0..jc) {
                                let ijkl_start = (jc*(jc+1)/2+ic)*npair;
                                let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                                //let reduce_ij = ijkl.get4d_slice([0,0,ic,jc],npair).unwrap();
                                let dm_ic = dm[i_spin].get1d_slice(ic*num_basis,num_basis).unwrap();
                                let dm_jc = dm[i_spin].get1d_slice(jc*num_basis,num_basis).unwrap();
                                let mut vk_ic = vk_thread.get1d_slice_mut(ic*(ic+1)/2,ic+1).unwrap();
                                let mut kl = 0_usize;
                                for k in (0..num_basis) {
                                    if k<=ic {
                                        vk_ic[..k].iter_mut()
                                            .zip(reduce_ij[kl..kl+k].iter())
                                            .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                        vk_ic[k] += reduce_ij[kl..kl+k]
                                            .iter()
                                            .zip(dm_jc[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                        kl += k;
                                        vk_ic[k] += reduce_ij[kl] *dm_jc[k];
                                        kl += 1;
                                    } else {
                                        vk_ic[..ic+1].iter_mut()
                                            .zip(reduce_ij[kl..kl+ic+1].iter())
                                            .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                        kl += k+1;
                                    }
                                }
                                //=================================================
                                let mut vk_jc = vk_thread.get1d_slice_mut(jc*(jc+1)/2,jc+1).unwrap();
                                let mut kl = 0_usize;
                                for k in (0..num_basis) {
                                    if k<=jc {
                                        vk_jc[..k].iter_mut()
                                            .zip(reduce_ij[kl..kl+k].iter())
                                            .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                        vk_jc[k] += reduce_ij[kl..kl+k]
                                            .iter()
                                            .zip(dm_ic[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                        kl += k;
                                        vk_jc[k] += reduce_ij[kl] *dm_ic[k];
                                        kl += 1;
                                    } else {
                                        vk_jc[..jc+1].iter_mut()
                                            .zip(reduce_ij[kl..kl+jc+1].iter())
                                            .for_each(|(i,j)| {*i += j*dm_ic[k]});
                                        kl += k+1;
                                    }
                                }
                                //=================================================
                            }
                            let ijkl_start = (jc*(jc+1)/2+jc)*npair;
                            let reduce_ij = ijkl.get1d_slice(ijkl_start,npair).unwrap();
                            //let reduce_ij = ijkl.get4d_slice([0,0,ic,ic],npair).unwrap();
                            let dm_jc = dm[i_spin].get1d_slice(jc*num_basis,num_basis).unwrap();
                            let mut vk_jc = vk_thread.get1d_slice_mut(jc*(jc+1)/2,jc+1).unwrap();
                            let mut kl = 0_usize;
                            for k in (0..num_basis) {
                                if k<=jc {
                                    vk_jc[..k].iter_mut()
                                        .zip(reduce_ij[kl..kl+k].iter())
                                        .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                    vk_jc[k] += reduce_ij[kl..kl+k]
                                        .iter()
                                        .zip(dm_jc[..k].iter()).fold(0.0, |acc,(i,j)| acc+i*j);
                                    kl += k;
                                    vk_jc[k] += reduce_ij[kl] *dm_jc[k];
                                    kl += 1;
                                } else {
                                    vk_jc[..jc+1].iter_mut()
                                        .zip(reduce_ij[kl..kl+jc+1].iter())
                                        .for_each(|(i,j)| {*i += j*dm_jc[k]});
                                    kl += k+1;
                                }
                            }
                        }
                        tx_jc_thread.send(vk_thread);
                    });
                    for received in rx_jc {
                        vk[i_spin].data.iter_mut()
                            .zip(received.data)
                            .for_each(|(i,j)| {*i += j});
                    }
                }).unwrap();
            }
        }
        if scaling_factor!=1.0f64 {
            for i_spin in (0..spin_channel) {
                vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };
        vk
    }

    pub fn generate_vk_with_isdf_new(&self, scaling_factor: f64) -> Vec<MatrixUpper<f64>>{
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        //let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let mut vk: Vec<MatrixUpper<f64>> = vec![];
        let spin_channel = self.mol.spin_channel;
        let m = self.m.clone().unwrap();
        let tab_ao = self.tab_ao.clone().unwrap();
        let n_ip = m.size[0];

        for i_spin in 0..spin_channel{
            let mut dm_s = &self.density_matrix[i_spin];
            let nw = occupied_orbital_count(&self.occupation[i_spin]);
            let mut kernel_mid = MatrixFull::new([n_ip,num_basis], 0.0);
            _dgemm(&tab_ao,(0..num_basis, 0..n_ip),'T',
                dm_s,(0..num_basis,0..num_basis),'N',
                &mut kernel_mid, (0..n_ip, 0..num_basis),
                1.0,0.0);

            let mut kernel = MatrixFull::new([n_ip,n_ip], 0.0);
            _dgemm(&kernel_mid,(0..n_ip, 0..num_basis),'N',
            &tab_ao, (0..num_basis, 0..n_ip),'N',
            &mut kernel, (0..n_ip,0..n_ip),
            1.0, 0.0);

            kernel.data.iter_mut().zip(m.data.iter()).for_each(|(x,y)|{
                *x *= *y * scaling_factor
            });

            let mut tmp = MatrixFull::new([num_basis, n_ip], 0.0);
            _dgemm(&tab_ao,(0..num_basis,0..n_ip),'N',
            &kernel,(0..n_ip,0..n_ip),'N',
            &mut tmp, (0..num_basis,0..n_ip),
            1.0, 0.0);
            let mut vk_i = MatrixFull::new([num_basis, num_basis], 0.0);
            _dgemm(&tmp,(0..num_basis,0..n_ip),'N',
            &tab_ao,(0..num_basis,0..n_ip),'T',
            &mut vk_i, (0..num_basis,0..num_basis),
            1.0, 0.0);
            vk.push(vk_i.to_matrixupper());
        }
        vk
    }

    pub fn generate_vk_with_isdf_dm_only(&mut self) -> Vec<MatrixUpper<f64>>{
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        //let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        let mut vk: Vec<MatrixUpper<f64>> = vec![];
        let eigv = &self.eigenvectors;
        let spin_channel = self.mol.spin_channel;
        let m = self.m.clone().unwrap();
        let tab_ao = self.tab_ao.clone().unwrap();
        let n_ip = m.size[0];

        for i_spin in 0..spin_channel{
            let occ_s =  &self.occupation[i_spin];
            let nw = occupied_orbital_count(occ_s);

            let mut tab_mo = MatrixFull::new([nw,n_ip], 0.0);
            _dgemm(&eigv[i_spin],(0..num_basis, 0..nw),'T',
                &tab_ao,(0..num_basis,0..n_ip),'N',
                &mut tab_mo, (0..nw, 0..n_ip),
                1.0,0.0);

            let mut zip_m_mo = MatrixFull::new([n_ip,n_ip], 0.0);
            _dgemm(&tab_mo,(0..nw, 0..n_ip),'T',
            &tab_mo, (0..nw, 0..n_ip),'N',
            &mut zip_m_mo, (0..n_ip,0..n_ip),
            1.0, 0.0);

            zip_m_mo.data.iter_mut().zip(m.data.iter()).for_each(|(x,y)|{
                *x *= *y * (-1.0)
            });

            let mut tmp = MatrixFull::new([num_basis, n_ip], 0.0);
            _dgemm(&tab_ao,(0..num_basis,0..n_ip),'N',
            &zip_m_mo,(0..n_ip,0..n_ip),'N',
            &mut tmp, (0..num_basis,0..n_ip),
            1.0, 0.0);

            let mut vk_i = MatrixFull::new([num_basis, num_basis], 0.0);
            _dgemm(&tmp,(0..num_basis,0..n_ip),'N',
            &tab_ao,(0..num_basis,0..n_ip),'T',
            &mut vk_i, (0..num_basis,0..num_basis),
            1.0, 0.0);

            vk.push(vk_i.to_matrixupper());

        }
        vk
    }
    pub fn generate_hf_hamiltonian_erifold4(&mut self) {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let spin_channel = self.mol.spin_channel;
        //let homo = &self.homo;
        //let vj = if self.mol.ctrl.num_threads>1 {
        //    self.generate_vj_with_erifold4_sync(1.0);
        //} else {
        //    self.generate_vj_with_erifold4(1.0)
        //};
        //let vk = if self.mol.ctrl.num_threads>1 {
        //    self.generate_vk_with_erifold4_sync_v02(-0.5);
        //} else {
        //    self.generate_vk_with_erifold4_v02(-0.5)
        //};
        let vj = self.generate_vj_with_erifold4_sync(1.0);
        let scaling_factor = match self.scftype {
            SCFType::RHF => -0.5,
            _ => -1.0,
        };
        let vk = self.generate_vk_with_erifold4_sync(scaling_factor);
        // let tmp_matrix = &self.h_core;
        // let mut tmp_num = 0;
        // let (i_len,j_len) =  (self.mol.num_basis,self.mol.num_basis);
        // let (k_len,l_len) =  (self.mol.num_basis,self.mol.num_basis);
        // tmp_matrix.data.iter().enumerate().for_each(|value| {
        //     if value.1.abs()>1.0e-1 {
        //         println!("I= {:2} Value= {:16.8}",value.0,value.1);
        //     }
        // });
        for i_spin in (0..spin_channel) {
            self.hamiltonian[i_spin] = self.h_core.clone();
            self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(vj[0].data.par_iter())
                            .for_each(|(h_ij,vj_ij)| {
                                *h_ij += vj_ij
                            });
            self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(vj[1].data.par_iter())
                            .for_each(|(h_ij,vj_ij)| {
                                *h_ij += vj_ij
                            });
            self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(vk[i_spin].data.par_iter())
                            .for_each(|(h_ij,vk_ij)| {
                                *h_ij += vk_ij
                            });
        };

        if let SCFType::ROHF = self.scftype {
            self.roothaan_hamiltonian = Some(self.generate_roothaan_fock());
        }
    }
    pub fn generate_hf_hamiltonian_ri_v(&mut self, mpi_operator: &Option<MPIOperator>) {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let spin_channel = self.mol.spin_channel;
        let dt1 = time::Local::now();
        let vj = if self.mol.ctrl.isdf_new {
            self.generate_vj_ri_direct(None)
        } else {
            match self.algorithm_jk {
                AlgorithmJK::RiIncore | AlgorithmJK::Separated(AlgorithmJ::RiIncore, _) => self.generate_vj_with_ri_v_sync(1.0, mpi_operator),
                AlgorithmJK::RiDirect | AlgorithmJK::Separated(AlgorithmJ::RiDirect, _) => self.generate_vj_ri_direct(None),
                _ => unreachable!("Other cases of algorithm_jk ({:?}) should have been ruled out. If this happens, it is a bug.", self.algorithm_jk),
            }
        };

        let dt2 = time::Local::now();
        let scaling_factor = match self.scftype {
            SCFType::RHF => -0.5,
            _ => -1.0,
        };

        let use_dm_only = self.mol.ctrl.use_dm_only;
        let vk = if self.mol.ctrl.use_isdf && !self.mol.ctrl.isdf_new {
            self.generate_vk_with_isdf(scaling_factor, use_dm_only)
        } else if self.mol.ctrl.isdf_new {
            self.generate_vk_with_isdf_new(scaling_factor)
        } else {
            match self.algorithm_jk {
                AlgorithmJK::RiIncore | AlgorithmJK::Separated(_, AlgorithmK::RiIncore) => self.generate_vk_with_ri_v(scaling_factor, use_dm_only, mpi_operator),
                AlgorithmJK::RiDirect | AlgorithmJK::Separated(_, AlgorithmK::RiDirect) => self.generate_vk_ri_direct(scaling_factor, use_dm_only, None, None),
                _ => unreachable!("Other cases of algorithm_jk ({:?}) should have been ruled out. If this happens, it is a bug.", self.algorithm_jk),
            }
        };


        let dt3 = time::Local::now();
        let timecost1 = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
        let timecost2 = (dt3.timestamp_millis()-dt2.timestamp_millis()) as f64 /1000.0;
        if self.mol.ctrl.print_level>2 {
            println!("The evaluation of Vj and Vk matrices cost {:10.2} and {:10.2} seconds, respectively",
                      timecost1,timecost2);
        }
        for i_spin in (0..spin_channel) {
            self.hamiltonian[i_spin] = self.h_core.clone();
            self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(vj[0].data.par_iter())
                            .for_each(|(h_ij,vj_ij)| {
                                *h_ij += vj_ij
                            });
            self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(vj[1].data.par_iter())
                            .for_each(|(h_ij,vj_ij)| {
                                *h_ij += vj_ij
                            });
            self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(vk[i_spin].data.par_iter())
                            .for_each(|(h_ij,vk_ij)| {
                                *h_ij += vk_ij
                            });
        };

        if let SCFType::ROHF = self.scftype {
            self.roothaan_hamiltonian = Some(self.generate_roothaan_fock());
        }
    }
    pub fn generate_hf_hamiltonian_ri_v_dm_only(&mut self, mpi_operator: &Option<MPIOperator>) {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let spin_channel = self.mol.spin_channel;
        //let homo = &self.homo;
        let dt1 = time::Local::now();
        let vj = if self.mol.ctrl.isdf_new || self.mol.ctrl.ri_k_only {
            self.generate_vj_ri_direct(None)
        } else {
            self.generate_vj_with_ri_v_sync(1.0, mpi_operator)
        };
        let dt2 = time::Local::now();
        let scaling_factor = match self.scftype {
            SCFType::RHF => -0.5,
            _ => -1.0,
        };
        let vk = self.generate_vk_with_ri_v(scaling_factor, true, mpi_operator);
        let dt3 = time::Local::now();
        let timecost1 = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
        let timecost2 = (dt3.timestamp_millis()-dt2.timestamp_millis()) as f64 /1000.0;
        if self.mol.ctrl.print_level>2 {
            println!("The evaluation of Vj and Vk matrices cost {:10.2} and {:10.2} seconds, respectively",
                      timecost1,timecost2);
        }
        for i_spin in (0..spin_channel) {
            self.hamiltonian[i_spin] = self.h_core.clone();
            self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(vj[0].data.par_iter())
                            .for_each(|(h_ij,vj_ij)| {
                                *h_ij += vj_ij
                            });
            self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(vj[1].data.par_iter())
                            .for_each(|(h_ij,vj_ij)| {
                                *h_ij += vj_ij
                            });
            self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(vk[i_spin].data.par_iter())
                            .for_each(|(h_ij,vk_ij)| {
                                *h_ij += vk_ij
                            });
        };

        if let SCFType::ROHF = self.scftype {
            self.roothaan_hamiltonian = Some(self.generate_roothaan_fock());
        }
    }

    pub fn generate_ks_hamiltonian_erifold4(&mut self) -> (f64,f64) {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let spin_channel = self.mol.spin_channel;
        let mut exc_total = 0.0;
        let mut vxc_total = 0.0;
        //let homo = &self.homo;
        for i_spin in (0..spin_channel) {
            self.hamiltonian[i_spin] = self.h_core.clone();
        }
        let dt1 = time::Local::now();
        //let vj = self.generate_vj_with_ri_v_sync(1.0);
        let vj = self.generate_vj_with_erifold4_sync(1.0);
        for i_spin in (0..spin_channel) {
            self.hamiltonian[i_spin].data
                .par_iter_mut()
                .zip(vj[0].data.par_iter())
                .for_each(|(h_ij,vj_ij)| {
                    *h_ij += vj_ij
                });
            self.hamiltonian[i_spin].data
                .par_iter_mut()
                .zip(vj[1].data.par_iter())
                .for_each(|(h_ij,vj_ij)| {
                    *h_ij += vj_ij
                });
        }
        let dt2 = time::Local::now();
        let scaling_factor = match self.scftype {
            SCFType::RHF => -0.5,
            _ => -1.0,
        }*self.mol.xc_data.dfa_hybrid_scf ;
        if ! scaling_factor.eq(&0.0) {
            //let vk = self.generate_vk_with_ri_v(scaling_factor);
            let vk = self.generate_vk_with_erifold4_sync(scaling_factor);
            for i_spin in (0..spin_channel) {
                self.hamiltonian[i_spin].data
                    .par_iter_mut()
                    .zip(vk[i_spin].data.par_iter())
                    .for_each(|(h_ij,vk_ij)| {
                        *h_ij += vk_ij
                    });
            };
        }
        let dt3 = time::Local::now();
        if self.mol.xc_data.dfa_compnt_scf.len()!=0 {
            let (exc,vxc) = self.generate_vxc(1.0);
            //println!("{:?}",vxc[0].data);
            for i_spin in (0..spin_channel) {
                self.hamiltonian[i_spin].data
                                .par_iter_mut()
                                .zip(vxc[i_spin].data.par_iter())
                                .for_each(|(h_ij,vk_ij)| {
                                    *h_ij += vk_ij
                                });
            };
            exc_total = exc;
            for i_spin in (0..spin_channel) {
                let dm_s = &self.density_matrix[i_spin];
                let dm_upper = dm_s.to_matrixupper();
                vxc_total += SCF::par_energy_contraction(&dm_upper, &vxc[i_spin]);
            }
        }

        let dt4 = time::Local::now();
        
        let timecost1 = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
        let timecost2 = (dt3.timestamp_millis()-dt2.timestamp_millis()) as f64 /1000.0;
        let timecost3 = (dt4.timestamp_millis()-dt3.timestamp_millis()) as f64 /1000.0;
        if self.mol.ctrl.print_level>2 {
            println!("The evaluation of Vj, Vk and Vxc matrices cost {:10.2}, {:10.2} and {:10.2} seconds, respectively",
                      timecost1,timecost2, timecost3);
        };

        if let SCFType::ROHF = self.scftype {
            self.roothaan_hamiltonian = Some(self.generate_roothaan_fock());
        }
        (exc_total, vxc_total)

    }
    pub fn generate_ks_hamiltonian_ri_v(&mut self, mpi_operator: &Option<MPIOperator>) -> (f64,f64) {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let spin_channel = self.mol.spin_channel;
        let mut exc_total = 0.0;
        let mut vxc_total = 0.0;
        let mut vk_total = 0.0;

        for i_spin in (0..spin_channel) {
            self.hamiltonian[i_spin] = self.h_core.clone();
        }

        // Coulomb J
        let dt1 = time::Local::now();
        let vj = match self.algorithm_jk {
            AlgorithmJK::RiIncore | AlgorithmJK::Separated(AlgorithmJ::RiIncore, _) => self.generate_vj_with_ri_v_sync(1.0, mpi_operator),
            AlgorithmJK::RiDirect | AlgorithmJK::Separated(AlgorithmJ::RiDirect, _) => self.generate_vj_ri_direct(None),
            _ => unreachable!("Other cases of algorithm_jk ({:?}) should have been ruled out. If this happens, it is a bug.", self.algorithm_jk),
        };

        for i_spin in (0..spin_channel) {
            self.hamiltonian[i_spin].data.par_iter_mut()
                .zip(vj[0].data.par_iter()).for_each(|(h,v)| *h += v);
            self.hamiltonian[i_spin].data.par_iter_mut()
                .zip(vj[1].data.par_iter()).for_each(|(h,v)| *h += v);
        }

        let dt2 = time::Local::now();

        // Standard hybrid exchange: K_total = hyb * K_full + (alpha-hyb) * K_erf
        // Since we have K_erfc (from SR rimatr), K_erf = K_full - K_erfc
        // So: K_total = alpha * K_full - (alpha-hyb) * K_erfc
        // F_ex = base_scaling * alpha * K_full + (-base_scaling) * (alpha-hyb) * K_erfc
        let base_scaling = match self.scftype { SCFType::RHF => -0.5, _ => -1.0 };

        if let Some((omega, alpha, _)) = self.mol.xc_data.rsh_params() {
            let hyb = self.mol.xc_data.dfa_hybrid_scf;
            let scaling_kfull = base_scaling * alpha;           // alpha * K_full
            let scaling_ksr = -base_scaling * (alpha - hyb);    // - (alpha-hyb) * K_erfc  (note: sign flipped)

            // Full K: alpha * K_full
            if scaling_kfull.abs() > 1e-10 {
                let use_dm_only = self.mol.ctrl.use_dm_only;
                let vk_full = match self.algorithm_jk {
                    AlgorithmJK::RiIncore | AlgorithmJK::Separated(_, AlgorithmK::RiIncore) => self.generate_vk_with_ri_v(scaling_kfull, use_dm_only, mpi_operator),
                    AlgorithmJK::RiDirect | AlgorithmJK::Separated(_, AlgorithmK::RiDirect) => self.generate_vk_ri_direct(scaling_kfull, use_dm_only, None, None),
                    _ => unreachable!("Other cases of algorithm_jk ({:?}) should have been ruled out. If this happens, it is a bug.", self.algorithm_jk),
                };
                for i_spin in 0..spin_channel {
                    self.hamiltonian[i_spin].data.par_iter_mut()
                        .zip(vk_full[i_spin].data.par_iter()).for_each(|(h,v)| *h += v);
                }
            }

            // SR correction: -(alpha-hyb) * K_erfc → scaling_ksr * K_erfc where scaling_ksr = -base_scaling*(alpha-hyb)
            // For ri-direct, pass omega so libcint computes erfc(ωr)/r integrals on the fly.
            // For ri-incore, use the pre-built rimatr_sr / ri3fn_sr.
            if scaling_ksr.abs() > 1e-10 {
                let vk_sr = match self.algorithm_jk {
                    AlgorithmJK::RiDirect | AlgorithmJK::Separated(_, AlgorithmK::RiDirect) => {
                        self.generate_vk_ri_direct(scaling_ksr, self.mol.ctrl.use_dm_only, None, Some(-omega))
                    }
                    _ => {
                        if self.rimatr_sr.is_some() {
                            let dm = &self.density_matrix;
                            vk_upper_with_rimatr_use_dm_only_sync(&self.rimatr_sr, dm, spin_channel, scaling_ksr)
                        } else if self.ri3fn_sr.is_some() {
                            let dm = &self.density_matrix;
                            vk_upper_with_ri_v_use_dm_only_sync(&self.ri3fn_sr, dm, spin_channel, scaling_ksr)
                        } else {
                            vec![MatrixUpper::empty(); spin_channel]
                        }
                    }
                };
                for i_spin in 0..spin_channel {
                    if !vk_sr[i_spin].data.is_empty() {
                        self.hamiltonian[i_spin].data.par_iter_mut()
                            .zip(vk_sr[i_spin].data.par_iter()).for_each(|(h,v)| *h += v);
                    }
                }
            }
        } else {
            let scaling_factor = base_scaling * self.mol.xc_data.dfa_hybrid_scf;
            if ! scaling_factor.eq(&0.0) {
                let use_dm_only = self.mol.ctrl.use_dm_only;
                let vk = match self.algorithm_jk {
                    AlgorithmJK::RiIncore | AlgorithmJK::Separated(_, AlgorithmK::RiIncore) => self.generate_vk_with_ri_v(scaling_factor, use_dm_only, mpi_operator),
                    AlgorithmJK::RiDirect | AlgorithmJK::Separated(_, AlgorithmK::RiDirect) => self.generate_vk_ri_direct(scaling_factor, use_dm_only, None, None),
                    _ => unreachable!("Other cases of algorithm_jk ({:?}) should have been ruled out. If this happens, it is a bug.", self.algorithm_jk),
                };
                for i_spin in (0..spin_channel) {
                    self.hamiltonian[i_spin].data.par_iter_mut()
                        .zip(vk[i_spin].data.par_iter()).for_each(|(h,v)| *h += v);
                };
            }
        }
        let dt3 = time::Local::now();
        if self.mol.xc_data.dfa_compnt_scf.len()!=0 {
            //let (exc,vxc) = self.generate_vxc_rayon_dm_only(1.0);
            let (_, exc,vxc) = if self.mol.ctrl.use_dm_only {
                self.generate_vxc_mpi_rayon_dm_only(1.0, mpi_operator)
            } else {
                self.generate_vxc_mpi_rayon(1.0, mpi_operator)
            };
            //let (exc,vxc) = self.generate_vxc(1.0);
            let _ = utilities::timing(&dt3, Some("evaluate vxc total"));
            for i_spin in (0..spin_channel) {
                self.hamiltonian[i_spin].data
                                .par_iter_mut()
                                .zip(vxc[i_spin].data.par_iter())
                                .for_each(|(h_ij,vk_ij)| {
                                    *h_ij += vk_ij
                                });
            };
            exc_total = exc;
            for i_spin in (0..spin_channel) {
                let dm_s = &self.density_matrix[i_spin];
                let dm_upper = dm_s.to_matrixupper();
                vxc_total += SCF::par_energy_contraction(&dm_upper, &vxc[i_spin]);
            }

        };

        let dt4 = time::Local::now();

        if let SCFType::ROHF = self.scftype {
            self.roothaan_hamiltonian = Some(self.generate_roothaan_fock());
            let dt5 = time::Local::now();
            let timecost4 = (dt5.timestamp_millis()-dt4.timestamp_millis()) as f64 /1000.0;
            if self.mol.ctrl.print_level > 2 {
                println!("The evaluation of Roothaan effective Fock Matrix costs {:10.2} seconds.", timecost4);
            }
        }
        
        let dt_solv0 = time::Local::now();
        //let mut esolv_total = 0.0;
        if self.mol.ctrl.solvent_enabled{
            if self.solvent_scf.is_some() {
                if let Some(solvent_scf) = self.solvent_scf.as_ref() {
                    for i_spin in 0..spin_channel {
                        self.hamiltonian[i_spin].data
                            .par_iter_mut()
                            .zip(solvent_scf.veff.data.par_iter())
                            .for_each(|(h_ij, veff_ij)| {
                                *h_ij += veff_ij;
                            });

                        let dm_s = &self.density_matrix[i_spin];
                        let dm_upper = dm_s.to_matrixupper();
                        //esolv_total -=  SCF::par_energy_contraction(&dm_upper, &solvent_scf.veff)
                    }
                    
                }
            }
        }
        //exc_total += esolv_total;
        
        let dt_solv1 = time::Local::now();
        let timecost_solv = (dt_solv1.timestamp_millis()-dt_solv0.timestamp_millis()) as f64 /1000.0;
        if self.mol.ctrl.solvent_enabled && self.mol.ctrl.print_level > 2 {
            println!("The evaluation of Solvent potential costs {:10.2} seconds.", timecost_solv);
        }


        let timecost1 = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
        let timecost2 = (dt3.timestamp_millis()-dt2.timestamp_millis()) as f64 /1000.0;
        let timecost3 = (dt4.timestamp_millis()-dt3.timestamp_millis()) as f64 /1000.0;
        if self.mol.ctrl.print_level>2 {
            println!("The evaluation of Vj, Vk and Vxc matrices cost {:10.2}, {:10.2} and {:10.2} seconds, respectively",
                      timecost1,timecost2, timecost3);
        };
        (exc_total, vxc_total)

    }

    pub fn compute_ghost_charge_forces(&self) -> Option<MatrixFull<f64>> {
        let geom = &self.mol.geom;
        if geom.ghost_pc_chrg.is_empty() {
            return None;
        }

        let num_ghosts = geom.ghost_pc_chrg.len();
        let ghost_pc_chrg = &geom.ghost_pc_chrg;
        let ghost_pc_pos = &geom.ghost_pc_pos;
        let mut matr_force = MatrixFull::new([3, num_ghosts], 0.0);
        let nao = self.mol.num_basis;

        let is_sp = self.mol.ctrl.spin_polarization;
        let dm = &self.density_matrix;
        let use_double_dm = is_sp && dm.len() > 1 && dm[1].size == [nao, nao];

        let nuclear_charges = crate::geom_io::get_charge(&geom.elem);
        let qm_position = &geom.position;
        let mut temp_mol = self.mol.clone();

        for i in 0..num_ghosts {
            let q_i = ghost_pc_chrg[i];
            let pos_i = [
                ghost_pc_pos[[0, i]],
                ghost_pc_pos[[1, i]],
                ghost_pc_pos[[2, i]],
            ];

            let mut deriv_hcore = vec![0.0; 3 * nao * nao];

            temp_mol.with_rinv_origin(pos_i, |mol_mut| {
                let cint = mol_mut.initialize_cint(false);
                let iprinv_out = cint.integrate("int1e_iprinv", "s1", None);
                
                if let Some(out_vec) = iprinv_out.out {
                    for t in 0..3 {
                        for nu in 0..nao {
                            for mu in 0..nao {
                                let idx = mu + nu * nao + t * (nao * nao);
                                if let Some(&val) = out_vec.get(idx) {
                                    deriv_hcore[t * nao * nao + mu * nao + nu] += q_i * val;
                                }
                            }
                        }
                    }
                }
            });

            let mut electron_force = [0.0; 3];
            for t in 0..3 {
                let mut sum_val = 0.0;
                for mu in 0..nao {
                    for nu in 0..nao {
                        let h_val = deriv_hcore[t * nao * nao + mu * nao + nu];
                        let dm_val = if !use_double_dm { 
                            dm[0][[mu, nu]] 
                        } else { 
                            dm[0][[mu, nu]] + dm[1][[mu, nu]] 
                        };
                        sum_val += h_val * dm_val;
                    }
                }
                electron_force[t] = 2.0 * sum_val;
            }

            let mut nuclear_force = [0.0; 3];
            for a in 0..nuclear_charges.len() {
                let pos_a = [qm_position[[0, a]], qm_position[[1, a]], qm_position[[2, a]]];
                let r_vec = [pos_i[0] - pos_a[0], pos_i[1] - pos_a[1], pos_i[2] - pos_a[2]];
                let r_sq = r_vec.iter().map(|&x| x * x).sum::<f64>();
                
                if r_sq > 1e-12 {
                    let prefactor = (nuclear_charges[a] * q_i) / (r_sq * r_sq.sqrt());
                    for t in 0..3 {
                        nuclear_force[t] += prefactor * r_vec[t];
                    }
                }
            }

            for t in 0..3 {
                matr_force[[t, i]] = electron_force[t] + nuclear_force[t];
            }
        }

        Some(matr_force)
    }
//    pub fn generate_ks_hamiltonian_ri_v_dm_only(&mut self, mpi_operator: &Option<MPIOperator>) -> (f64,f64) {
//        let num_basis = self.mol.num_basis;
//        let num_state = self.mol.num_state;
//        let spin_channel = self.mol.spin_channel;
//        let mut exc_total = 0.0;
//        let mut vxc_total = 0.0;
//        let mut vk_total = 0.0;
//        //let homo = &self.homo;
//        for i_spin in (0..spin_channel) {
//            self.hamiltonian[i_spin] = self.h_core.clone();
//        }
//        let dt1 = time::Local::now();
//        let vj = self.generate_vj_with_ri_v_sync(1.0);
//        for i_spin in (0..spin_channel) {
//            self.hamiltonian[i_spin].data
//                .par_iter_mut()
//                .zip(vj[0].data.par_iter())
//                .for_each(|(h_ij,vj_ij)| {
//                    *h_ij += vj_ij
//                });
//            self.hamiltonian[i_spin].data
//                .par_iter_mut()
//                .zip(vj[1].data.par_iter())
//                .for_each(|(h_ij,vj_ij)| {
//                    *h_ij += vj_ij
//                });
//        }
//        let dt2 = time::Local::now();
//        let scaling_factor = match self.scftype {
//            SCFType::RHF => -0.5,
//            _ => -1.0,
//        }*self.mol.xc_data.dfa_hybrid_scf ;
//        if ! scaling_factor.eq(&0.0) {
//            let use_dm_only = self.mol.ctrl.use_dm_only;
//            let vk = if self.mol.ctrl.use_isdf{
//                self.generate_vk_with_isdf(scaling_factor, use_dm_only)
//            }else{
//                self.generate_vk_with_ri_v(scaling_factor, use_dm_only)
//            };
//            for i_spin in (0..spin_channel) {
//                self.hamiltonian[i_spin].data
//                    .par_iter_mut()
//                    .zip(vk[i_spin].data.par_iter())
//                    .for_each(|(h_ij,vk_ij)| {
//                        *h_ij += vk_ij
//                    });
//            };
//        }
//        let dt3 = time::Local::now();
//        if self.mol.xc_data.dfa_compnt_scf.len()!=0 {
//            let (_, exc,vxc) = self.generate_vxc_mpi_rayon_dm_only(1.0, mpi_operator);
//            //let (exc,vxc) = self.generate_vxc(1.0);
//            let _ = utilities::timing(&dt3, Some("evaluate vxc total"));
//            for i_spin in (0..spin_channel) {
//                self.hamiltonian[i_spin].data
//                                .par_iter_mut()
//                                .zip(vxc[i_spin].data.par_iter())
//                                .for_each(|(h_ij,vk_ij)| {
//                                    *h_ij += vk_ij
//                                });
//            };
//            exc_total = exc;
//            for i_spin in (0..spin_channel) {
//                let dm_s = &self.density_matrix[i_spin];
//                let dm_upper = dm_s.to_matrixupper();
//                vxc_total += SCF::par_energy_contraction(&dm_upper, &vxc[i_spin]);
//            }
//        }
//
//        let dt4 = time::Local::now();
//        
//        let timecost1 = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
//        let timecost2 = (dt3.timestamp_millis()-dt2.timestamp_millis()) as f64 /1000.0;
//        let timecost3 = (dt4.timestamp_millis()-dt3.timestamp_millis()) as f64 /1000.0;
//        if self.mol.ctrl.print_level>2 {
//            println!("The evaluation of Vj, Vk and Vxc matrices cost {:10.2}, {:10.2} and {:10.2} seconds, respectively",
//                      timecost1,timecost2, timecost3);
//        };
//        (exc_total, vxc_total)
//
//    }

    pub fn generate_hf_hamiltonian_for_guess(&mut self) {
        if self.mol.xc_data.dfa_compnt_scf.len() == 0 {
            if self.mol.ctrl.eri_type.eq("analytic") {
                self.generate_hf_hamiltonian_erifold4();
            } else if  self.mol.ctrl.eri_type.eq("ri_v") {
                self.generate_hf_hamiltonian_ri_v_dm_only(&None);
            }
        } else {
            if self.mol.ctrl.eri_type.eq("analytic") {
                self.generate_ks_hamiltonian_erifold4();
            } else if  self.mol.ctrl.eri_type.eq("ri_v") {
                let origin_dm_only = self.mol.ctrl.use_dm_only;
                self.mol.ctrl.use_dm_only = true;
                self.generate_ks_hamiltonian_ri_v(&None);
                self.mol.ctrl.use_dm_only = origin_dm_only;
            }
        }
    }

    pub fn evaluate_hf_total_energy(&self) -> f64 {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let spin_channel = self.mol.spin_channel;
        let dm = &self.density_matrix;
        let mut total_energy = self.nuc_energy;
        match self.scftype {
            SCFType::RHF => {
                // D*(H^{core}+F)
                let dm_s = &dm[0];
                let hc = &self.h_core;
                let ht_s = &self.hamiltonian[0];
                let dm_upper = dm_s.to_matrixupper();
                let mut hc_and_ht = hc.clone();
                hc_and_ht.data.par_iter_mut().zip(ht_s.data.par_iter()).for_each(|value| {
                    *value.0 += value.1
                });
                total_energy += SCF::par_energy_contraction(&dm_upper, &hc_and_ht);
            },
            _ => {
                let dm_a = &dm[0];
                let dm_a_upper = dm_a.to_matrixupper();
                let dm_b = &dm[1];
                let dm_b_upper = dm_b.to_matrixupper();
                let mut dm_t_upper = dm_a_upper.clone();
                dm_t_upper.data.par_iter_mut().zip(dm_b_upper.data.par_iter()).for_each(|value| {*value.0+=value.1});

                // Now for D^{tot}*H^{core} term
                total_energy += SCF::par_energy_contraction(&dm_t_upper, &self.h_core);
                // Now for D^{alpha}*F^{alpha} term
                total_energy += SCF::par_energy_contraction(&dm_a_upper, &self.hamiltonian[0]);
                // Now for D^{beta}*F^{beta} term
                total_energy += SCF::par_energy_contraction(&dm_b_upper, &self.hamiltonian[1]);

            },
        }
        total_energy
    }

    pub fn generate_hf_hamiltonian(&mut self, mpi_operator: &Option<MPIOperator>) {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let spin_channel = self.mol.spin_channel;
        let mut exc_total = 0.0;
        let mut vxc_total = 0.0;
        if self.mol.xc_data.dfa_compnt_scf.len() == 0 {
            if self.mol.ctrl.eri_type.eq("analytic") {
                self.generate_hf_hamiltonian_erifold4();
            } else if  self.mol.ctrl.eri_type.eq("ri_v") {
                self.generate_hf_hamiltonian_ri_v(mpi_operator);
            }
        } else {
            if self.mol.ctrl.eri_type.eq("analytic") {
                //panic!("Hybrid DFA is not implemented with analytic ERI.");
                let (tmp_exc_total,tmp_vxc_total) = self.generate_ks_hamiltonian_erifold4();
                exc_total = tmp_exc_total;
                vxc_total = tmp_vxc_total;
            } else {
                let (tmp_exc_total,tmp_vxc_total) = self.generate_ks_hamiltonian_ri_v(mpi_operator);
                exc_total = tmp_exc_total;
                vxc_total = tmp_vxc_total;
            }
        }

        let dm = &self.density_matrix;

        // The following scf energy evaluation follow the formula presented in
        // the quantum chemistry book of Szabo A. and Ostlund N.S. P 150, Formula (3.184)
        self.scf_energy = self.nuc_energy;
        // for DFT calculations, we should replace the exchange-correlation (xc) potential by the xc energy
        //if self.mol.ctrl.print_level>1 {println!("Exc: {:?}, Vxc: {:?}", exc_total, vxc_total)};
        self.scf_energy = self.scf_energy - vxc_total + exc_total;
        //if let Some(local_grids) = &self.grids {
        //    let total_elec = numerical_density(&local_grids, &self.mol, dm, mpi_operator);
        //    if self.mol.ctrl.print_level>1 {
        //        if self.mol.spin_channel==1 {
        //            println!("total electron number: {:16.8}", total_elec[0])
        //        } else {
        //            println!("electron number in alpha-channel: {:12.8}", total_elec[0]);
        //            println!("electron number in beta-channel:  {:12.8}", total_elec[1]);
        //        }
        //    };
        //}
        //println!("==== IGOR debug for Exc[HF]====");
        //let exc_hf = self.evaluate_exact_exchange_ri_v(mpi_operator);
        //println!("Exc[HF] = {:16.8}", exc_hf);
        //println!("==== IGOR debug for Exc[HF]====");
        if self.mol.ctrl.solvent_enabled {
            if let Some(solvent_scf) = self.solvent_scf.as_ref() {
                self.scf_energy += solvent_scf.eng_nuc;
            }
            // SMD: add CDS energy from PcmStatic (geometry-dependent, computed once)
            if let Some(ref pstatic) = self.solvent_static_obj.as_ref().map(|s| &s.pstatic) {
                if let Some(e_cds) = pstatic.e_cds {
                    self.scf_energy += e_cds;
                }
            }
        }
        debug!("Exc: {:16.8}, Vxc: {:16.8}", exc_total, vxc_total);
        match self.scftype {
            SCFType::RHF => {
                // D*(H^{core}+F)
                let dm_s = &dm[0];
                let hc = &self.h_core;
                let ht_s = &self.hamiltonian[0];
                let dm_upper = dm_s.to_matrixupper();
                let mut hc_and_ht = hc.clone();
                hc_and_ht.data.par_iter_mut().zip(ht_s.data.par_iter()).for_each(|value| {
                    *value.0 += value.1
                });
                self.scf_energy += SCF::par_energy_contraction(&dm_upper, &hc_and_ht);
            },
            _ => {
                let dm_a = &dm[0];
                let dm_a_upper = dm_a.to_matrixupper();
                let dm_b = &dm[1];
                let dm_b_upper = dm_b.to_matrixupper();
                let mut dm_t_upper = dm_a_upper.clone();
                dm_t_upper.data.par_iter_mut().zip(dm_b_upper.data.par_iter()).for_each(|value| {*value.0+=value.1});

                // Now for D^{tot}*H^{core} term
                self.scf_energy += SCF::par_energy_contraction(&dm_t_upper, &self.h_core);
                // Now for D^{alpha}*F^{alpha} term
                self.scf_energy += SCF::par_energy_contraction(&dm_a_upper, &self.hamiltonian[0]);
                // Now for D^{beta}*F^{beta} term
                self.scf_energy += SCF::par_energy_contraction(&dm_b_upper, &self.hamiltonian[1]);

            },
        }

        match self.mol.ctrl.xc_type {
            DFTType::DeepLearning => {
                // For the Deep-Learning DFA (DL-DFA) developed by ShenBi  -- Coded by IGOR/ 2025/3/22
                // 1) evaluate the energy_components with respect to the current xc_data.dfa_paramr_scf
                let energy_components = self.evaluate_energy_components(mpi_operator);
                // 2) evaluate the DL-DFA total energy
                let exc_dldfa = crate::dft::deep_learning::dl_hybrid_xc_energy(&energy_components);
                // 3) update the DL-DFA xc potential 
                let next_dfa_paramr_scf = crate::dft::deep_learning::dl_hybrid_xc_param(&energy_components);
                self.mol.xc_data.dfa_hybrid_scf = next_dfa_paramr_scf[1];
                self.mol.xc_data.dfa_paramr_scf = next_dfa_paramr_scf[2..].to_vec();
            },
            _ => {}
        }
         
    }

    pub fn generate_roothaan_fock(&self) -> MatrixUpper<f64> {
        //generate Roothaan's effective Fock matrix
        // ======== ======== ====== =========
        // space     closed   open   virtual
        // ======== ======== ====== =========
        // closed      Fc      Fb     Fc
        // open        Fb      Fc     Fa
        // virtual     Fc      Fa     Fc
        // ======== ======== ====== =========
        // where Fc = (Fa + Fb) / 2, Roothaan's Fock matrix for core
        let num_basis = self.mol.num_basis;

        let hamiltonian_a: MatrixFull<f64> = self.hamiltonian[0].to_matrixfull().unwrap();
        let hamiltonian_b: MatrixFull<f64> = self.hamiltonian[1].to_matrixfull().unwrap();
        let hamiltonian_c: MatrixFull<f64> = (hamiltonian_a.clone() + hamiltonian_b.clone()) * 0.5;

        // Projector for core, open-shell, and virtual space
        // pc = dm_b * ovlp
        // po = (dm_a - dm_b) * ovlp
        // pv = eye - dm_a * ovlp
        let mut pc = MatrixFull::new([num_basis, num_basis], 0.0);
        let mut po = MatrixFull::new([num_basis, num_basis], 0.0);
        let mut pv = MatrixFull::new([num_basis, num_basis], 0.0);
        let ovlp_full = self.ovlp.to_matrixfull().unwrap();
        
        //_dgemm(&self.density_matrix[1], (0..num_basis,0..num_basis), 'N', &ovlp_full, (0..num_basis,0..num_basis), 'N', &mut pc, (0..num_basis,0..num_basis), 1.0, 0.0);
        _dsymm(&self.density_matrix[1], &ovlp_full, &mut pc, 'L', 'U', 1.0, 0.0);
        
        let dm_o = self.density_matrix[0].clone() - self.density_matrix[1].clone();
        //_dgemm(&dm_o, (0..num_basis,0..num_basis), 'N', &ovlp_full, (0..num_basis,0..num_basis), 'N', &mut po, (0..num_basis,0..num_basis), 1.0, 0.0);
        _dsymm(&dm_o, &ovlp_full, &mut po, 'L', 'U', 1.0, 0.0);
        
        pv.iter_diagonal_mut().unwrap().for_each(|x| *x = 1.0);
        //_dgemm(&self.density_matrix[0], (0..num_basis,0..num_basis), 'N', &ovlp_full, (0..num_basis,0..num_basis), 'N', &mut pv, (0..num_basis,0..num_basis), -1.0, 1.0);
        _dsymm(&self.density_matrix[0], &ovlp_full, &mut pv, 'L', 'U', -1.0, 1.0);
        
        let mut roothaan_fock = MatrixFull::new([num_basis, num_basis], 0.0);
        roothaan_fock += apply_projection_operator(&pc, &hamiltonian_c, &pc) * 0.5;
        roothaan_fock += apply_projection_operator(&po, &hamiltonian_c, &po) * 0.5;
        roothaan_fock += apply_projection_operator(&pv, &hamiltonian_c, &pv) * 0.5;
        roothaan_fock += apply_projection_operator(&po, &hamiltonian_b, &pc);
        roothaan_fock += apply_projection_operator(&po, &hamiltonian_a, &pv);
        roothaan_fock += apply_projection_operator(&pv, &hamiltonian_c, &pc);
        roothaan_fock = roothaan_fock.clone() + roothaan_fock.transpose();
        
        roothaan_fock.to_matrixupper()
    }

    pub fn evaluate_energy_components(&self, mpi_operator: &Option<MPIOperator>) -> Vec<f64> {
        /// 1. the ordering is given by E_noXC, Ex_HF, and a sequence of DFA components in self.xc_data.dfa_compnt_scf
        /// 2. this function should be invoked after generate_hf_hamiltonian()
        let exc_hf = if self.mol.ctrl.eri_type.eq("ri_v") {
            self.evaluate_exact_exchange_ri_v(mpi_operator)
        } else {
            panic!("Only RI_V version has been implemented for SCF::evaluate_energy_components")
        };
        //println!("Exc[HF] = {:16.8}", exc_hf);
        let current_dfa = &self.mol.xc_data;
        let xc_energy_list = if let Some(grids) = &self.grids {
            let xc_code_list = &current_dfa.dfa_compnt_scf;
            self.mol.xc_data.xc_exc_list(
                xc_code_list, 
                grids, 
                &self.density_matrix, 
                &self.eigenvectors, 
                &self.occupation
            )
        } else {
            let code_list = &self.mol.xc_data.dfa_compnt_scf;
            vec![[0.0,0.0];code_list.len()]
        };

        let mut xc_energy_list: Vec<f64> = xc_energy_list.iter().map(|energy| energy[0]+energy[1]).collect();

        #[cfg(feature = "mpi")]
        if let Some(mpi_world) = mpi_operator {
            let mut tot_xc_list = mpi_reduce(&mpi_world.world, &xc_energy_list, 0, &SystemOperation::sum());
            mpi_broadcast(&mpi_world.world, &mut tot_xc_list, 0);
            xc_energy_list = tot_xc_list;
        }

        let mut exc_total = exc_hf * current_dfa.dfa_hybrid_scf;
        xc_energy_list.iter().zip(current_dfa.dfa_paramr_scf.iter()).for_each(|(xc_energy, &param)| {
            exc_total += xc_energy * param
        });

        let e_noxc = self.scf_energy - exc_total;
        //let mut energy_components = vec![self.scf_energy-exc_hf];

        xc_energy_list.insert(0, exc_hf);
        xc_energy_list.insert(0, e_noxc);

        xc_energy_list
    }

    /// about total energy contraction:
    /// E0 = 1/2*\sum_{i}\sum_{j}a_{ij}*b_{ij}
    /// einsum('ij,ij')
    pub fn par_energy_contraction(a:&MatrixUpper<f64>, b:&MatrixUpper<f64>) -> f64 {
        let (sender, receiver) = channel();
        a.data.par_iter().zip(b.data.par_iter()).for_each_with(sender, |s,(dm,hc)| {
            let mut tmp_scf_energy = 0.0_f64;
            tmp_scf_energy += dm*(hc);
            s.send(tmp_scf_energy).unwrap();
        });
        let mut tmp_energy = 2.0_f64 * receiver.into_iter().sum::<f64>();
        let a_diag = a.get_diagonal_terms().unwrap();
        let b_diag = b.get_diagonal_terms().unwrap();
        let double_count = a_diag.par_iter().zip(b_diag.par_iter()).fold(|| 0.0_f64,|acc,(a,b)| {
            acc + *a*(*b)
        }).sum::<f64>();

        (tmp_energy - double_count) * 0.5
    }

    pub fn evaluate_exact_exchange_ri_v(&self, mpi_operator: &Option<MPIOperator>) -> f64 {
        let mut x_energy = 0.0;
        let use_dm_only = self.mol.ctrl.use_dm_only;
        //let mut vk = self.generate_vk_with_ri_v(1.0, use_dm_only);
        let mut vk = if self.mol.ctrl.use_isdf{
            self.generate_vk_with_isdf(1.0, use_dm_only)
        }else{
            match self.algorithm_jk {
                AlgorithmJK::RiDirect | AlgorithmJK::Separated(_, AlgorithmK::RiDirect) => self.generate_vk_ri_direct(1.0, use_dm_only, None, None),
                AlgorithmJK::RiIncore | AlgorithmJK::Separated(_, AlgorithmK::RiIncore) => self.generate_vk_with_ri_v(1.0, use_dm_only, mpi_operator),
                _ => unreachable!("Other cases of algorithm_jk ({:?}) should have been ruled out. If this happens, it is a bug.", self.algorithm_jk),
            }
        };
        let spin_channel = self.mol.spin_channel;
        for i_spin in 0..spin_channel {
            let dm_s = &self.density_matrix[i_spin];
            let dm_upper = dm_s.to_matrixupper();
            x_energy += SCF::par_energy_contraction(&dm_upper, &vk[i_spin]);
        }
        if self.mol.spin_channel==1 {
            // the factor of 0.5 is due to the use of full density matrix for the exchange energy evaluation
            x_energy*-0.5
        } else {
            x_energy*-1.0
        }
    }

    pub fn evaluate_xc_energy(&mut self, iop: usize, mpi_operator: &Option<MPIOperator>) -> f64 {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let num_auxbas = self.mol.num_auxbas;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        //let mut vxc: MatrixUpper<f64> = MatrixUpper::new(1,0.0f64);
        let mut exc_spin:Vec<f64> = vec![];
        let dm = &mut self.density_matrix;
        let mo = &mut self.eigenvectors;
        let occ = &mut self.occupation;
        if let Some(grids) = &mut self.grids {
            exc_spin = self.mol.xc_data.xc_exc(grids, spin_channel,dm, mo, occ,iop, mpi_operator);
        }
        let exc:f64 = exc_spin.iter().sum();
        exc
    }

    //pub fn evaluate_xc_energy
       

    pub fn diagonalize_hamiltonian(&mut self, mpi_operator: &Option<MPIOperator>) {
        (self.eigenvectors, self.eigenvalues, self.mol.num_state) = diagonalize_hamiltonian_outside(&self, mpi_operator);
    }

    pub fn semi_diagonalize_hamiltonian(&mut self) {
        (self.semi_eigenvectors, self.semi_eigenvalues, self.semi_fock, self.mol.num_state) = semi_diagonalize_hamiltonian_outside(&self);
    }

    pub fn check_scf_convergence(&self, scftracerecode: &ScfTraceRecord) -> [bool;2] {
        let spin_channel = self.mol.spin_channel;
        let num_basis = self.mol.num_basis as f64;
        let max_scf_cycle = self.mol.ctrl.max_scf_cycle;
        let scf_acc_rho   = self.mol.ctrl.scf_acc_rho;   
        let scf_acc_eev   = self.mol.ctrl.scf_acc_eev;  
        let scf_acc_etot  = self.mol.ctrl.scf_acc_etot;
        let scf_acc_g     = self.mol.ctrl.scf_acc_g;
        let scf_conv_criteria = self.mol.ctrl.scf_conv_criteria.as_str(); 

        let (cur_energy, pre_energy) = if self.mol.ctrl.smear.is_some() {
            let sigma = self.current_smear_sigma;
            (self.scf_energy - sigma * self.smearing_entropy,
             scftracerecode.scf_energy - sigma * scftracerecode.smearing_entropy)
        } else {
            (self.scf_energy, scftracerecode.scf_energy)
        };
        let diff_energy = cur_energy-pre_energy;

        let cur_energy = &self.eigenvalues;
        let pre_energy = &scftracerecode.eigenvalues;
        let mut eev_err = 0.0;
        for i_spin in 0..spin_channel {
            if self.mol.num_elec[i_spin+1] > 1.0E-5 {
                eev_err += cur_energy[i_spin].par_iter()
                    .zip(pre_energy[i_spin].par_iter())
                    .map(|(c,p)| (c-p).powf(2.0)).sum::<f64>();
            } else {
                eev_err += 0.0;
            }
        }
        eev_err = eev_err.sqrt();

        let mut dm_err = [0.0;2];
        let cur_dm = &self.density_matrix;
        let pre_dm = &scftracerecode.density_matrix[1];

        for i_spin in 0..spin_channel {
            dm_err[i_spin] = cur_dm[i_spin].data.par_iter()
                .zip(pre_dm[i_spin].data.par_iter())
                .map(|(c,p)| (c-p).powf(2.0)).sum::<f64>().sqrt()/num_basis;
        }

        let mut g_err = [0.0;2];
        for i_spin in 0..spin_channel {
            g_err[i_spin] = norm(&self.grad_dm[i_spin], "l2");
        }

        if spin_channel==1 {
            info!("SCF Change: DM {:10.5e}; eev {:10.5e} Ha; etot {:10.5e} Ha; grad {:10.5e} Ha",
                dm_err[0], eev_err, diff_energy, g_err[0])
        } else {
            info!("SCF Change: DM ({:10.5e},{:10.5e}); eev {:10.5e} Ha; etot {:10.5e} Ha; grad ({:10.5e},{:10.5e}) Ha",
                dm_err[0], dm_err[1], eev_err, diff_energy, g_err[0], g_err[1])
        };

        if scftracerecode.num_iter<2 {
            return [false,false]
        }

        let mut flag = [true,true];
        flag[0] = diff_energy.abs()<=scf_acc_etot;
        match scf_conv_criteria {
            "g" => {
                flag[0] = flag[0] && g_err[0] <= scf_acc_g;
                if spin_channel==2 { flag[0] = flag[0] && g_err[1] <= scf_acc_g; }
            }
            "dm" => {
                flag[0] = flag[0] && dm_err[0] <= scf_acc_rho;
                if spin_channel==2 { flag[0] = flag[0] && dm_err[1] <= scf_acc_rho; }
            }
            "dm,eev" => {
                flag[0] = flag[0] && dm_err[0] <= scf_acc_rho && eev_err <= scf_acc_eev;
                if spin_channel==2 { flag[0] = flag[0] && dm_err[1] <= scf_acc_rho; }
            }
            _ => {
                panic!("unknown scf_conv_criteria '{}', must be one of: dm,eev | dm | g", scf_conv_criteria);
            }
        }

        // Now check if max_scf_cycle is reached or not
        flag[1] = scftracerecode.num_iter >= max_scf_cycle;

        flag
    }

    // relevant to RI-V
    pub fn generate_vj_with_ri_v(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {

        let spin_channel = self.mol.spin_channel;
        let dm = &self.density_matrix;

        vj_upper_with_ri_v(&self.ri3fn, dm, spin_channel, scaling_factor)
    }

    pub fn generate_vj_with_ri_v_sync(&mut self, scaling_factor: f64, mpi_operator: &Option<MPIOperator>) -> Vec<MatrixUpper<f64>> {

        let spin_channel = self.mol.spin_channel;
        let dm = &self.density_matrix;

        if self.mol.ctrl.use_ri_symm {
            vj_upper_with_rimatr_sync_mpi(&self.rimatr, dm, spin_channel, scaling_factor, mpi_operator)
        } else {
            //vj_upper_with_ri_v_sync(&self.ri3fn, dm, spin_channel, scaling_factor)
            if self.mol.ctrl.use_isdf && !self.mol.ctrl.isdf_k_only && !self.mol.ctrl.isdf_new{
                vj_upper_with_ri_v_sync(&self.ri3fn_isdf, dm, spin_channel, scaling_factor)
            }else{
                vj_upper_with_ri_v_sync(&self.ri3fn, dm, spin_channel, scaling_factor)
            }
        }
    }

    pub fn generate_vj_with_isdf(&mut self, scaling_factor: f64) -> Vec<MatrixUpper<f64>> {

        let spin_channel = self.mol.spin_channel;
        let dm = &self.density_matrix;

        vj_upper_with_ri_v(&self.ri3fn_isdf, dm, spin_channel, scaling_factor)
    }

    pub fn generate_vk_with_ri_v(&self, scaling_factor: f64, use_dm_only: bool, mpi_operator: &Option<MPIOperator>) -> Vec<MatrixUpper<f64>> {
        //let num_basis = self.mol.num_basis;
        //let num_state = self.mol.num_state;
        //let num_auxbas = self.mol.num_auxbas;
        //let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;

        if self.mol.ctrl.use_ri_symm {
            if use_dm_only {
                let dm = &self.density_matrix;
                vk_upper_with_rimatr_use_dm_only_sync_mpi(&self.rimatr, dm, spin_channel, scaling_factor, mpi_operator)
            } else {
                let eigv = &self.eigenvectors;
                let occupation = &self.occupation;
                let num_elec = &self.mol.num_elec;
                //vk_upper_with_rimatr_sync(&mut self.rimatr, eigv, num_elec, occupation, spin_channel, scaling_factor)
                vk_upper_with_rimatr_sync_mpi(&self.rimatr, eigv, num_elec, occupation, spin_channel, scaling_factor,mpi_operator)
            }
        } else if self.mol.ctrl.isdf_new{
            self.generate_vk_with_isdf_new(scaling_factor)
        }else{
            if use_dm_only {
                let dm = &self.density_matrix;
                vk_upper_with_ri_v_use_dm_only_sync(&self.ri3fn, dm, spin_channel, scaling_factor)
            } else {
                let eigv = &self.eigenvectors;
                vk_upper_with_ri_v_sync(&self.ri3fn, eigv, &self.mol.num_elec, &self.occupation, 
                                        spin_channel, scaling_factor)
            }
        }

    

    }

    pub fn generate_vk_with_isdf(&self, scaling_factor: f64, use_dm_only: bool) -> Vec<MatrixUpper<f64>> {
        //let num_basis = self.mol.num_basis;
        //let num_state = self.mol.num_state;
        //let num_auxbas = self.mol.num_auxbas;
        //let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;

        if self.mol.ctrl.use_ri_symm {
            let dm = &self.density_matrix;
            vk_upper_with_rimatr_use_dm_only_sync(&self.rimatr, dm, spin_channel, scaling_factor)
        } else {
            if use_dm_only {
                //println!("use isdf to generate k");
                let dm = &self.density_matrix;
                //&dm[0].formated_output_e(5, "full");
                vk_upper_with_ri_v_use_dm_only_sync(&self.ri3fn_isdf, dm, spin_channel, scaling_factor)
            } else {
                let eigv = &self.eigenvectors;
                vk_upper_with_ri_v_sync(&self.ri3fn_isdf, eigv, &self.mol.num_elec, &self.occupation, 
                                        spin_channel, scaling_factor)
            }
        }
        

    }

    pub fn generate_vxc(&self, scaling_factor: f64) -> (f64, Vec<MatrixUpper<f64>>) {
        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let num_auxbas = self.mol.num_auxbas;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        //let mut vxc: MatrixUpper<f64> = MatrixUpper::new(1,0.0f64);
        let mut vxc:Vec<MatrixUpper<f64>> = vec![MatrixUpper::empty();spin_channel];
        let mut exc_spin:Vec<f64> = vec![];
        let mut exc_total:f64 = 0.0;
        let mut vxc_mf:Vec<MatrixFull<f64>> = vec![MatrixFull::empty();spin_channel];
        let dm = &self.density_matrix;
        let mo = &self.eigenvectors;
        let occ = &self.occupation;
        let print_level = self.mol.ctrl.print_level;
        if let Some(grids) = &self.grids {
            let dt0 = utilities::init_timing();
            let (exc,mut vxc_ao) = self.mol.xc_data.xc_exc_vxc(grids, spin_channel,dm, mo, occ, print_level);
            let dt1 = utilities::timing(&dt0, Some("Total vxc_ao time"));
            exc_spin = exc;
            if let Some(ao) = &grids.ao {
                // Evaluate the exchange-correlation energy
                //exc_total = izip!(grids.weights.iter(),exc.data.iter()).fold(0.0,|acc,(w,e)| {
                //    acc + w*e
                //});
                for i_spin in 0..spin_channel {
                    let vxc_mf_s = vxc_mf.get_mut(i_spin).unwrap();
                    *vxc_mf_s = MatrixFull::new([num_basis,num_basis],0.0f64);
                    let vxc_ao_s = vxc_ao.get(i_spin).unwrap();
                    _dgemm_full(ao, 'N', vxc_ao_s, 'T', vxc_mf_s, 1.0, 0.0);
                    //vxc_mf_s.lapack_dgemm(ao, vxc_ao_s, 'N', 'T', 1.0, 0.0);
                }
            }
            let dt2 = utilities::timing(&dt1, Some("From vxc_ao to vxc"));
        }


        let dt0 = utilities::init_timing();
        for i_spin in (0..spin_channel) {
            let mut vxc_s = vxc.get_mut(i_spin).unwrap();
            let mut vxc_mf_s = vxc_mf.get_mut(i_spin).unwrap();

            vxc_mf_s.self_add(&vxc_mf_s.transpose());
            vxc_mf_s.self_multiple(0.5);
            //vxc_mf_s.formated_output(10, "full");
            *vxc_s = vxc_mf_s.to_matrixupper();
        }

        utilities::timing(&dt0, Some("symmetrize vxc"));

        exc_total = exc_spin.iter().sum();


        if scaling_factor!=1.0f64 {
            exc_total *= scaling_factor;
            for i_spin in (0..spin_channel) {
                vxc[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };

        (exc_total, vxc)

    }

    pub fn generate_vxc_rayon_dm_only(&self, scaling_factor: f64) -> ([f64;2], f64, Vec<MatrixUpper<f64>>) {
        //In this subroutine, we call the lapack dgemm in a rayon parallel environment.
        //In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
        let default_omp_num_threads = self.mol.ctrl.num_threads.unwrap();

        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let num_auxbas = self.mol.num_auxbas;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        //let mut vxc: MatrixUpper<f64> = MatrixUpper::new(1,0.0f64);
        let mut vxc:Vec<MatrixUpper<f64>> = vec![MatrixUpper::empty();spin_channel];
        let mut exc_spin:Vec<f64> = vec![0.0;spin_channel];
        let mut total_elec = [0.0,0.0];
        let mut exc_total:f64 = 0.0;
        let mut vxc_mf:Vec<MatrixFull<f64>> = vec![MatrixFull::new([num_basis,num_basis],0.0);spin_channel];
        let dm = &self.density_matrix;
        let mo = &self.eigenvectors;
        let occ = &self.occupation;
        if let Some(grids) = &self.grids {
            let (sender, receiver) = channel();
            grids.parallel_balancing.par_iter().for_each_with(sender,|s,range_grids| {
                omp_set_num_threads_wrapper(1);
                // change the return of xc_exc_vxc, directly return vxc_mat [num_basis, num_basis]
                // let (exc,vxc_ao,total_elec) = self.mol.xc_data.xc_exc_vxc_slots_dm_only(range_grids.clone(), grids, spin_channel,dm, mo, occ);
                //exc_spin = exc;
                let (exc, vxc_mf, total_elec) = self.mol.xc_data.xc_exc_vxc_slots_dm_only(range_grids.clone(), grids, spin_channel,dm, mo, occ, self.mol.ctrl.print_level, self.mol.ctrl.vxc_screen_threshold);
                // let mut vxc_mf: Vec<MatrixFull<f64>> = vec![MatrixFull::new([num_basis,num_basis],0.0f64);spin_channel];;
                // if let Some(ao) = &grids.ao {
                //     for i_spin in 0..spin_channel {

                //         let vxc_mf_s = vxc_mf.get_mut(i_spin).unwrap();
                //         let vxc_ao_s = vxc_ao.get(i_spin).unwrap();
                //         rest_tensors::matrix::matrix_blas_lapack::_dgemm(
                //             ao,(0..num_basis, range_grids.clone()),'N',
                //             vxc_ao_s,(0..num_basis,0..range_grids.len()),'T',
                //             vxc_mf_s, (0..num_basis,0..num_basis),
                //             1.0,0.0);

                //         //vxc_mf_s.to_matrixfullslicemut().lapack_dgemm(
                //         //    &ao.to_matrixfullslice(), 
                //         //    &vxc_ao_s.to_matrixfullslice(),
                //         //    'N', 'T', 1.0, 0.0);
                //     }
                // }
                s.send((vxc_mf,exc,total_elec)).unwrap()
            });
            receiver.into_iter().for_each(|(vxc_mf_local,exc_local,loc_total_elec)| {
                vxc_mf.iter_mut().zip(vxc_mf_local.iter()).for_each(|(to_matr,from_matr)| {
                    to_matr.self_add(from_matr);
                });
                exc_spin.iter_mut().zip(exc_local.iter()).for_each(|(to_exc,from_exc)| {
                    *to_exc += from_exc
                });
                total_elec.iter_mut().zip(loc_total_elec.iter()).for_each(|(to_elec, from_elec)| {
                    *to_elec += from_elec

                })
            })
        }

        //if self.mol.ctrl.print_level>1 {
        //    if spin_channel==1 {
        //        println!("total electron number: {:16.8}", total_elec[0]);
        //    } else {
        //        println!("electron number in alpha-channel: {:12.8}", total_elec[0]);
        //        println!("electron number in beta-channel:  {:12.8}", total_elec[1]);
        //    }
        //}


        for i_spin in (0..spin_channel) {
            let mut vxc_s = vxc.get_mut(i_spin).unwrap();
            let mut vxc_mf_s = vxc_mf.get_mut(i_spin).unwrap();

            vxc_mf_s.self_add(&vxc_mf_s.transpose());
            vxc_mf_s.self_multiple(0.5);
            *vxc_s = vxc_mf_s.to_matrixupper();
        }

        exc_total = exc_spin.iter().sum();


        if scaling_factor!=1.0f64 {
            exc_total *= scaling_factor;
            for i_spin in (0..spin_channel) {
                vxc[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };

        omp_set_num_threads_wrapper(default_omp_num_threads);

        (total_elec, exc_total, vxc)

    }

    pub fn generate_vxc_mpi_rayon_dm_only(&self, scaling_factor: f64, mpi_operator: &Option<MPIOperator>) -> ([f64;2], f64, Vec<MatrixUpper<f64>>) {
        #[cfg(feature = "mpi")]
        let (total_elec, tot_exc, tot_xc) = if let Some(mpi_world) = mpi_operator {

            let world = &mpi_world.world;
            let my_rank = mpi_world.rank;

            let (mut total_elec, exc, mut vxc) = self.generate_vxc_rayon_dm_only(scaling_factor);

            let mut tot_exc = mpi_reduce(world, &[exc], 0, &SystemOperation::sum())[0];
            //mpi_broadcast(&world, &mut tot_exc, 0);

            let mut tot_elec = mpi_reduce(world, &total_elec, 0, &SystemOperation::sum());
            total_elec.iter_mut().zip(tot_elec.iter()).for_each(|(to, from)| *to = *from);
            //mpi_broadcast(&world, &mut total_elec, 0);

            //let mut tot_xc: Vec<MatrixUpper<f64>> = vec![MatrixUpper::empty(), MatrixUpper::empty()];
            for i_spin in 0..self.mol.spin_channel {
                let mut result= mpi_reduce(world, vxc[i_spin].data_ref().unwrap(), 0, &SystemOperation::sum());
                let mut xc_spin = vxc.get_mut(i_spin).unwrap();
                //mpi_broadcast_vector(&world, &mut result, 0);
                //if mpi_world.rank==0 {
                    xc_spin.data = result;
                //}
            } 

            (total_elec, tot_exc, vxc)

        } else {
            self.generate_vxc_rayon_dm_only(scaling_factor)
        };
        #[cfg(not(feature = "mpi"))]
        let (total_elec, tot_exc, tot_xc) = self.generate_vxc_rayon_dm_only(scaling_factor);

        if self.mol.spin_channel==1 {
            debug!("total electron number: {:16.8}", total_elec[0]);
        } else {
            debug!("electron number in alpha-channel: {:12.8}", total_elec[0]);
            debug!("electron number in beta-channel:  {:12.8}", total_elec[1]);
        }

        (total_elec, tot_exc, tot_xc)


    }

    pub fn generate_vxc_mpi_rayon(&self, scaling_factor: f64, mpi_operator: &Option<MPIOperator>) -> ([f64;2], f64, Vec<MatrixUpper<f64>>) {

        #[cfg(feature = "mpi")]
        let (total_elec, tot_exc, tot_xc) = if let Some(mpi_world) = mpi_operator {

            let world = &mpi_world.world;
            let my_rank = mpi_world.rank;

            let (mut total_elec, exc, mut vxc) = self.generate_vxc_rayon(scaling_factor);

            let mut tot_exc = mpi_reduce(world, &[exc], 0, &SystemOperation::sum())[0];
            mpi_broadcast(&world, &mut tot_exc, 0);

            let mut tot_elec = mpi_reduce(world, &total_elec, 0, &SystemOperation::sum());
            total_elec.iter_mut().zip(tot_elec.iter()).for_each(|(to, from)| *to = *from);
            mpi_broadcast(&world, &mut total_elec, 0);

            //let mut tot_xc: Vec<MatrixUpper<f64>> = vec![MatrixUpper::empty(), MatrixUpper::empty()];
            for i_spin in 0..self.mol.spin_channel {
                let mut result= mpi_reduce(world, vxc[i_spin].data_ref().unwrap(), 0, &SystemOperation::sum());

                let mut xc_spin = vxc.get_mut(i_spin).unwrap();
                //mpi_broadcast_vector(&world, &mut result, 0);
                if mpi_world.rank==0 {
                    xc_spin.data = result;
                } 
            } 

            (total_elec, tot_exc, vxc)

        } else {
            self.generate_vxc_rayon(scaling_factor)
        };
        #[cfg(not(feature = "mpi"))]
        let (total_elec, tot_exc, tot_xc) = self.generate_vxc_rayon(scaling_factor);

        if self.mol.spin_channel==1 {
            debug!("total electron number: {:16.8}", total_elec[0]);
        } else {
            debug!("electron number in alpha-channel: {:12.8}", total_elec[0]);
            debug!("electron number in beta-channel:  {:12.8}", total_elec[1]);
        }

        (total_elec, tot_exc, tot_xc)

    }


    pub fn generate_vxc_rayon(&self, scaling_factor: f64) -> ([f64;2], f64, Vec<MatrixUpper<f64>>) {
        //In this subroutine, we call the lapack dgemm in a rayon parallel environment.
        let default_omp_num_threads = self.mol.ctrl.num_threads.unwrap();

        let num_basis = self.mol.num_basis;
        let num_state = self.mol.num_state;
        let num_auxbas = self.mol.num_auxbas;
        let npair = num_basis*(num_basis+1)/2;
        let spin_channel = self.mol.spin_channel;
        //let mut vxc: MatrixUpper<f64> = MatrixUpper::new(1,0.0f64);
        let mut vxc:Vec<MatrixUpper<f64>> = vec![MatrixUpper::empty();spin_channel];
        let mut exc_spin:Vec<f64> = vec![0.0;spin_channel];
        let mut total_elec = [0.0,0.0];
        let mut exc_total:f64 = 0.0;
        let mut vxc_mf:Vec<MatrixFull<f64>> = vec![MatrixFull::new([num_basis,num_basis],0.0);spin_channel];
        let dm = &self.density_matrix;
        let mo = &self.eigenvectors;
        let occ = &self.occupation;
        if let Some(grids) = &self.grids {
            let (sender, receiver) = channel();
            grids.parallel_balancing.par_iter().for_each_with(sender,|s,range_grids| {

                // To ensure the efficiency, we disable the openmp ability of openblas within the parallel region
                omp_set_num_threads_wrapper(1);

                // change the return value of xc_exc_vxc by vxc_mat [num_basis, num_basis]
                // let (exc,vxc_ao,total_elec) = self.mol.xc_data.xc_exc_vxc_slots(range_grids.clone(), grids, spin_channel,dm, mo, occ);
                //exc_spin = exc;
                let (exc, vxc_mf, total_elec) = self.mol.xc_data.xc_exc_vxc_slots(range_grids.clone(), grids, spin_channel, dm, mo, occ, self.mol.ctrl.print_level, self.mol.ctrl.vxc_screen_threshold);
                // let mut vxc_mf: Vec<MatrixFull<f64>> = vec![MatrixFull::new([num_basis,num_basis],0.0f64);spin_channel];;
                // if let Some(ao) = &grids.ao {
                //     for i_spin in 0..spin_channel {

                //         let vxc_mf_s = vxc_mf.get_mut(i_spin).unwrap();
                //         let vxc_ao_s = vxc_ao.get(i_spin).unwrap();
                //         rest_tensors::matrix::matrix_blas_lapack::_dgemm(
                //             ao,(0..num_basis, range_grids.clone()),'N',
                //             vxc_ao_s,(0..num_basis,0..range_grids.len()),'T',
                //             vxc_mf_s, (0..num_basis,0..num_basis),
                //             1.0,0.0);

                //         //vxc_mf_s.to_matrixfullslicemut().lapack_dgemm(
                //         //    &ao.to_matrixfullslice(), 
                //         //    &vxc_ao_s.to_matrixfullslice(),
                //         //    'N', 'T', 1.0, 0.0);
                //     }
                // }
                s.send((vxc_mf,exc,total_elec)).unwrap()
            });
            receiver.into_iter().for_each(|(vxc_mf_local,exc_local,loc_total_elec)| {
                vxc_mf.iter_mut().zip(vxc_mf_local.iter()).for_each(|(to_matr,from_matr)| {
                    to_matr.self_add(from_matr);
                });
                exc_spin.iter_mut().zip(exc_local.iter()).for_each(|(to_exc,from_exc)| {
                    *to_exc += from_exc
                });
                total_elec.iter_mut().zip(loc_total_elec.iter()).for_each(|(to_elec, from_elec)| {
                    *to_elec += from_elec

                })
            })
        }

        //if self.mol.ctrl.print_level>1 {
        //    if spin_channel==1 {
        //        println!("total electron number: {:16.8}", total_elec[0]);
        //    } else {
        //        println!("electron number in alpha-channel: {:12.8}", total_elec[0]);
        //        println!("electron number in beta-channel:  {:12.8}", total_elec[1]);
        //    }
        //}


        for i_spin in (0..spin_channel) {
            let mut vxc_s = vxc.get_mut(i_spin).unwrap();
            let mut vxc_mf_s = vxc_mf.get_mut(i_spin).unwrap();

            vxc_mf_s.self_add(&vxc_mf_s.transpose());
            vxc_mf_s.self_multiple(0.5);
            *vxc_s = vxc_mf_s.to_matrixupper();
        }

        exc_total = exc_spin.iter().sum();


        if scaling_factor!=1.0f64 {
            exc_total *= scaling_factor;
            for i_spin in (0..spin_channel) {
                vxc[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
            }
        };

        omp_set_num_threads_wrapper(default_omp_num_threads);

        (total_elec,exc_total, vxc)

    }

    pub fn generate_ri3mo_rayon(&mut self, row_range: std::ops::Range<usize>, col_range: std::ops::Range<usize>) {
        if let SCFType::ROHF = self.scftype { //in ROHF case, generate semi-canonical eigenvectors for post SCF calculations.
            self.semi_diagonalize_hamiltonian(); 
        }

        if let Some((ref ri3ao, ref basbas2baspair, ref baspar2basbas))= &mut self.rimatr {
            let mut ri3mo: Vec<(RIFull<f64>,std::ops::Range<usize>, std::ops::Range<usize>)> = vec![];
            for i_spin in 0..self.mol.spin_channel {
                let eigenvector = match self.scftype {
                    SCFType::ROHF => &self.semi_eigenvectors.as_ref().unwrap()[i_spin],
                    _ => &self.eigenvectors[i_spin],
                };
                ri3mo.push(
                    ao2mo_rayon(
                        eigenvector, ri3ao, 
                        row_range.clone(), 
                        col_range.clone()
                    ).unwrap()
                )
            }

            //if let Some(my_data)=&self.mol.mpi_data {
            //    //if my_data.rank == 0 {self.eigenvectors[0].formated_output(5, "full")};
            //    let (dd, col, row) = &ri3mo[0];
            //    let ff = dd.get_reducing_matrix(0).unwrap();
            //    ff.iter_columns_full().enumerate().for_each(|(i,x)| {
            //        println!("i: {}", i);
            //        println!("x: {:?}", &x);
            //    })
            //} else {
            //    //self.eigenvectors[0].formated_output(5, "full");
            //    let (dd, col, row) = &ri3mo[0];
            //    let ff = dd.get_reducing_matrix(0).unwrap();
            //    ff.iter_columns_full().enumerate().for_each(|(i,x)| {
            //        println!("i: {}", i);
            //        println!("x: {:?}", &x);
            //    })
            //};

            // deallocate the rimatr to save the memory
            self.rimatr = None;
            self.ri3mo = Some(ri3mo);
        } else {
            // use rstsr::prelude::*;
            let mut ri3mo: Vec<(RIFull<f64>, std::ops::Range<usize>, std::ops::Range<usize>)> = vec![];
            let mut timerecords = TimeRecords::new();
            timerecords.new_item("ao2mo", "for the generation of RI3MO");
            match self.mol.spin_channel {
                1 => {
                    let eigenvectors = match self.scftype {
                        SCFType::ROHF => &self.semi_eigenvectors.as_ref().unwrap()[0],
                        _ => &self.eigenvectors[0],
                    };
                    let row_indices: Vec<usize> = row_range.clone().collect();
                    let col_indices: Vec<usize> = col_range.clone().collect();
                    let cderi = ri_jk::obtain_cderi_xvo_restricted(&self, &mut timerecords, Some(eigenvectors), Some(row_indices.as_slice()), Some(col_indices.as_slice()));
                    let shape = cderi.shape().to_vec().try_into().unwrap();
                    let data = cderi.into_shape(-1).into_raw();
                    let ri3ao = RIFull::from_vec(shape, data).unwrap();
                    ri3mo.push((ri3ao, row_range.clone(), col_range.clone()));
                },
                2 => {
                    let eigenvectors = match self.scftype {
                        SCFType::ROHF => [&self.semi_eigenvectors.as_ref().unwrap()[0], &self.semi_eigenvectors.as_ref().unwrap()[1]],
                        _ => [&self.eigenvectors[0], &self.eigenvectors[1]],
                    };
                    let row_indices: Vec<usize> = row_range.clone().collect();
                    let col_indices: Vec<usize> = col_range.clone().collect();
                    let cderi = ri_jk::obtain_cderi_xvo_unrestricted(
                        &self, &mut timerecords, Some(eigenvectors),
                        [Some(row_indices.as_slice()), Some(row_indices.as_slice())],
                        [Some(col_indices.as_slice()), Some(col_indices.as_slice())],
                    );

                    let [cderi_a, cderi_b] = cderi;
                    // handle alpha
                    let shape = cderi_a.shape().to_vec().try_into().unwrap();
                    let data = cderi_a.into_shape(-1).into_raw();
                    let ri3ao = RIFull::from_vec(shape, data).unwrap();
                    ri3mo.push((ri3ao, row_range.clone(), col_range.clone()));
                    // handle beta
                    let shape = cderi_b.shape().to_vec().try_into().unwrap();
                    let data = cderi_b.into_shape(-1).into_raw();
                    let ri3ao = RIFull::from_vec(shape, data).unwrap();
                    ri3mo.push((ri3ao, row_range.clone(), col_range.clone()));
                },
                _ => unreachable!()
            };
            self.ri3mo = Some(ri3mo);
        };
    }
    pub fn generate_ri3mo_rayon_for_multiple_times(&self, row_range: std::ops::Range<usize>, col_range: std::ops::Range<usize>)->Vec<(RIFull<f64>,std::ops::Range<usize>,std::ops::Range<usize>)> {

        let (mut ri3ao, mut basbas2baspair, mut baspar2basbas) =  if let Some((riao,basbas2baspair, baspar2basbas))=&self.rimatr {
            (riao,basbas2baspair, baspar2basbas)
        } else {
            panic!("rimatr should be initialized in the preparation of ri3mo");
        };
        let mut ri3mo: Vec<(RIFull<f64>,std::ops::Range<usize>, std::ops::Range<usize>)> = vec![];
        for i_spin in 0..self.mol.spin_channel {
            let eigenvector = match self.scftype {
                SCFType::ROHF => &self.semi_eigenvectors.as_ref().unwrap()[i_spin],
                _ => &self.eigenvectors[i_spin],
            };
            ri3mo.push(
                ao2mo_rayon(
                    eigenvector, ri3ao, 
                    row_range.clone(), 
                    col_range.clone()
                ).unwrap()
            )
        }
        ri3mo
    }

    pub fn generate_ri3mo_bse(&self, row_range: std::ops::Range<usize>, col_range: std::ops::Range<usize>)
        -> Vec<(RIFull<f64>, std::ops::Range<usize>, std::ops::Range<usize>)> {

        // Use BSE-specific RI integrals for AO→MO transformation
        if let Some((ri3ao, basbas2baspar, baspar2basbas)) = &self.rimatr_bse {
            // Using symmetric format BSE RI integrals
            let mut ri3mo = vec![];
            for i_spin in 0..self.mol.spin_channel {
                let eigenvector = match self.scftype {
                    SCFType::ROHF => &self.semi_eigenvectors.as_ref().unwrap()[i_spin],
                    _ => &self.eigenvectors[i_spin],
                };

                let tmp_ri3mo = ao2mo_rayon(
                    eigenvector, ri3ao,
                    row_range.clone(),
                    col_range.clone()
                ).unwrap();
                ri3mo.push(tmp_ri3mo);
            }
            ri3mo
        } else {
            // If rimatr_bse is not available, fall back to regular integrals
            // This maintains backward compatibility
            panic!("BSE RI integrals (rimatr_bse) not initialized. Only symmetric RI format is supported for BSE.");
        }
    }

    pub fn generate_ri3mo_full_rayon(&mut self, row_range: std::ops::Range<usize>, col_range: std::ops::Range<usize>) {
        let (mut ri3ao, mut basbas2baspair, mut baspar2basbas) =  if let Some((riao,basbas2baspair, baspar2basbas))=&mut self.rimatr {
            (riao,basbas2baspair, baspar2basbas)
        } else {
            panic!("rimatr should be initialized in the preparation of ri3mo");
        };
        let mut ri3mo: Vec<(RIFull<f64>,std::ops::Range<usize>, std::ops::Range<usize>)> = vec![];
        for i_spin in 0..self.mol.spin_channel {
            let eigenvector = &self.eigenvectors[i_spin];
            ri3mo.push(
                ao2mo_rayon(
                    eigenvector, ri3ao, 
                    row_range.clone(), 
                    col_range.clone()
                ).unwrap()
            )
        }
        self.ri3mo_full = Some(ri3mo);
        self.rimatr=None;

    }
    
    /// Generates J-matrix.
    /// 
    /// In function name:
    /// - `ri`: using RI-V method
    /// - `direct`: on-the-fly direct calculation
    /// 
    /// To activate this function, in the meantime when writing this function, in `ctrl.in`
    /// - specify `algorithm_j = ri-direct` or `algorithm_jk = ri-direct` to disable full storage of 3c-2e ERI (required);
    /// - specify `[ctrl]: max_memory` in MB for calculating `block_size` if not specified;
    fn generate_vj_ri_direct(&self, batch_size: Option<usize>) -> Vec<MatrixUpper<f64>> {
        let print_level = self.mol.ctrl.print_level;

        // compute batch_size
        let min_batch_size = 2 * rayon::current_num_threads();

        // estimate batch size based on available memory
        let nao = self.mol.num_basis;
        let naux = self.mol.num_auxbas;
        let nset = self.mol.spin_channel;
        let sys_info = sysinfo::System::new_all();
        let mem_avail = self.mol.ctrl.max_memory.map(|max_memory| {
            max_memory - detect_used_memory_mb("proc")
        });
        let mem_est = ri_jk::mem_estimate_vj_ri_direct(nao, naux, nset);
        let mut batch_size_estimate = calc_batch_size_from_mem_estimate::<f64>(&mem_est, mem_avail, None, true);

        // if estimated batch size is smaller than minimum, warn and set to minimum
        if batch_size_estimate < min_batch_size {
            warn!("in generate_vj_ri_direct, the estimated batch size ({batch_size_estimate}) is smaller than the minimum batch size ({min_batch_size}).");
            warn!("Setting batch size to {min_batch_size}. Memory could be insufficient.");
            batch_size_estimate = min_batch_size;
        }

        // if user specified batch size is smaller than minimum, warn and set to minimum
        let batch_size = if let Some(user_batch_size) = batch_size {
            if user_batch_size < min_batch_size {
                warn!("in generate_vj_ri_direct, the specified batch size ({user_batch_size}) is smaller than the minimum batch size ({min_batch_size}).");
                warn!("Setting batch size to {min_batch_size}.");
            }
            user_batch_size.max(min_batch_size)
        } else {
            batch_size_estimate
        };
        handle_memory_exceed(mem_est.estimate_mem::<f64>(batch_size), mem_avail, self.mol.ctrl.abort_on_mem_exceed);

        // batch size info output
        info!("in generate_vj_ri_direct, available memory: {:.2} MB", mem_avail.unwrap_or(f64::INFINITY));
        info!("in generate_vj_ri_direct, batch size      : {batch_size}");
        info!("in generate_vj_ri_direct, memory estimation");
        if print_level > 0 {
            mem_est.print_with_dtype::<f64>();
        }

        // compute vj only for specified spin channels
        let dms = &self.density_matrix[0..self.mol.spin_channel];
        let mol_obj = &self.mol;
        let mut vjs = ri_jk::generate_vj_ri_direct(dms, mol_obj, batch_size);

        // complete `vjs` if the spin channel is 1 (restricted, spin-unpolarized)
        if self.mol.spin_channel == 1 {
            vjs.push(MatrixUpper::new(1, 0.0f64));
        }

        vjs
    }

    fn generate_vk_ri_direct(&self, scaling_factor: f64, use_dm_only: bool, batch_size: Option<usize>, omega: Option<f64>) -> Vec<MatrixUpper<f64>> {
        let print_level = self.mol.ctrl.print_level;

        // compute batch_size
        let min_batch_size = 2 * rayon::current_num_threads();
        
        // estimate batch size based on available memory
        let nao = self.mol.num_basis;
        let naux = self.mol.num_auxbas;
        let nset = self.mol.spin_channel;
        let nocc_max = self.occupation.iter().map(|occ_s| occ_s.iter().filter(|&&x| x > f64::EPSILON).count()).max().unwrap_or(0);
        let sys_info = sysinfo::System::new_all();
        let mem_avail = self.mol.ctrl.max_memory.map(|max_memory| {
            max_memory - detect_used_memory_mb("proc")
        });
        let mem_est_direct = ri_jk::mem_estimate_vk_ri_direct_dm(nao, naux, nset);
        let mem_est_semi = ri_jk::mem_estimate_vk_ri_semi_direct_coeff(nao, naux, nocc_max, nset);
        let batch_size_estimate_direct = calc_batch_size_from_mem_estimate::<f64>(&mem_est_direct, mem_avail, None, true);
        let batch_size_estimate_semi = calc_batch_size_from_mem_estimate::<f64>(&mem_est_semi, mem_avail, None, true);

        // prefer semi-direct if possible
        let (alg_semi, mut batch_size_estimate) = if !use_dm_only && batch_size_estimate_semi >= min_batch_size {
            debug!("in generate_vk_ri_direct_dm, using semi-direct algorithm for vk computation.");
            (true, batch_size_estimate_semi)
        } else {
            debug!("in generate_vk_ri_direct_dm, using direct algorithm for vk computation.");
            (false, batch_size_estimate_direct)
        };

        // if estimated batch size is smaller than minimum, warn and set to minimum
        if batch_size_estimate < min_batch_size {
            warn!("in generate_vk_ri_direct_dm, the estimated batch size ({batch_size_estimate}) is smaller than the minimum batch size ({min_batch_size}).");
            warn!("Setting batch size to {min_batch_size}. Memory could be insufficient.");
            batch_size_estimate = min_batch_size
        }

        // if user specified batch size is smaller than minimum, warn and set to minimum
        let batch_size = if let Some(user_batch_size) = batch_size {
            if user_batch_size < min_batch_size {
                warn!("in generate_vk_ri_direct_dm, the specified batch size ({user_batch_size}) is smaller than the minimum batch size ({min_batch_size}).");
                warn!("Setting batch size to {min_batch_size}.");
            }
            user_batch_size.max(min_batch_size)
        } else {
            batch_size_estimate
        };
        let mem_est = if alg_semi { &mem_est_semi } else { &mem_est_direct };
        handle_memory_exceed(mem_est.estimate_mem::<f64>(batch_size), mem_avail, self.mol.ctrl.abort_on_mem_exceed);

        // info output
        debug!("in generate_vk_ri_direct_dm, available memory: {:.2} MB", mem_avail.unwrap_or(f64::INFINITY));
        debug!("in generate_vk_ri_direct_dm, batch size      : {batch_size}");
        debug!("in generate_vk_ri_direct_dm, memory estimation");
        if print_level > 1 {
            mem_est.print_with_dtype::<f64>();
        }

        // compute vk only for specified spin channels
        let mut vks = if alg_semi {
            // we assume copying molecular coefficients is cheap operation, to avoid lifetime issues
            let mut mo_coeff = self.eigenvectors[0..self.mol.spin_channel].iter().cloned().collect::<Vec<_>>();
            let mo_occ = &self.occupation[0..self.mol.spin_channel];
            let mol_obj = &self.mol;
            // special case for ROHF: copy alpha mo_coeff to beta (ROHF only store the alpha channel)
            match self.scftype {
                SCFType::ROHF => mo_coeff[1] = self.eigenvectors[0].clone(),
                _ => {}
            };
            let mo_coeff = &mo_coeff;
            ri_jk::generate_vk_ri_semi_direct_coeff(scaling_factor, mo_coeff, mo_occ, mol_obj, omega, batch_size)
        } else {
            let dms = &self.density_matrix[0..self.mol.spin_channel];
            let mol_obj = &self.mol;
            ri_jk::generate_vk_ri_direct_dm(scaling_factor, dms, mol_obj, omega, batch_size)
        };

        // complete `vks` if the spin channel is 1 (restricted, spin-unpolarized)
        if self.mol.spin_channel == 1 {
            vks.push(MatrixUpper::new(1, 0.0f64));
        }

        vks
    }
}

/// Applies a projection operator to a given matrix. Specifically, it calculates the
/// product \( a^T \cdot b \cdot c \), where \( a \), \( b \), and \( c \) are input matrices.
///
/// # Arguments
/// * `a` - The first matrix (used as a transpose in the calculation).
/// * `b` - The second matrix.
/// * `c` - The third matrix.
/// * `size` - The size of the matrices.
///
/// # Returns
/// A new matrix that represents the result of \( a^T \cdot b \cdot c \).
///
/// # Example
/// ```
/// let a = MatrixFull::new([size, size], ...);
/// let b = MatrixFull::new([size, size], ...);
/// let c = MatrixFull::new([size, size], ...);
/// let result = apply_projection_operator(&a, &b, &c, size);
/// ```
pub fn apply_projection_operator(a: &MatrixFull<f64>, b: &MatrixFull<f64>, c: &MatrixFull<f64>) -> MatrixFull<f64> {
    // Temporary matrix to store intermediate result of a^T * b
    let mut temp: MatrixFull<f64> = MatrixFull::new([a.size[1], b.size[1]], 0.0);
    
    // First multiplication: a^T * b
    _dgemm_full(a, 'T', b,  'N', &mut temp, 1.0, 0.0);
    
    // Second multiplication: (a^T * b) * c
    let mut final_result: MatrixFull<f64> = MatrixFull::new([a.size[1], c.size[1]], 0.0);
    _dgemm_full(&temp, 'N', c, 'N', &mut final_result, 1.0, 0.0);
    
    final_result
}

pub fn apply_guess_mix(scf_data: &mut SCF) {
    for (i_spin, &theta_deg) in scf_data.mol.ctrl.guess_mix_theta_deg.iter().enumerate() {
        if theta_deg < 0.0 || theta_deg > 45.0 {
            println!(
                "WARNING: theta for spin {} = {:.1}° is outside the recommended range (0°–45°); mixing may be ineffective or unstable.",
                i_spin, theta_deg
            );
        }

        let (cos_theta, sin_theta) = {
            let rad = theta_deg.to_radians();
            (rad.cos(), rad.sin())
        };

        let homo = scf_data.homo[i_spin];
        let lumo = scf_data.lumo[i_spin];
        let eigenvector_mut = scf_data.eigenvectors.get_mut(i_spin).unwrap();
        let homo_vec: Vec<f64> = eigenvector_mut.iter_column(homo).cloned().collect();
        let lumo_vec: Vec<f64> = eigenvector_mut.iter_column(lumo).cloned().collect();

        let (mixed_homo_vec, mixed_lumo_vec) = if i_spin == 0 {(
            homo_vec.iter().zip(&lumo_vec).map(|(h, l)| cos_theta * h + sin_theta * l).collect::<Vec<f64>>(),
            homo_vec.iter().zip(&lumo_vec).map(|(h, l)| -sin_theta * h + cos_theta * l).collect::<Vec<f64>>(),
        )} else {(
            homo_vec.iter().zip(&lumo_vec).map(|(h, l)| cos_theta * h - sin_theta * l).collect::<Vec<f64>>(),
            homo_vec.iter().zip(&lumo_vec).map(|(h, l)| sin_theta * h + cos_theta * l).collect::<Vec<f64>>(),
        )};

        for (val, slot) in mixed_homo_vec.iter().zip(eigenvector_mut.iter_column_mut(homo)) {
            *slot = *val;
        }
        for (val, slot) in mixed_lumo_vec.iter().zip(eigenvector_mut.iter_column_mut(lumo)) {
            *slot = *val;
        }
    }
}


/// return the occupation range and virtual range for the preparation of ri3mo;
pub fn determine_ri3mo_size_for_pt2_and_rpa(scf_data: &SCF) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
    let num_state = scf_data.mol.num_state;
    let mut homo = 0_usize;
    let mut lumo = num_state;
    let start_mo = scf_data.mol.start_mo;

    for i_spin in 0..scf_data.mol.spin_channel {

        let i_homo = scf_data.homo.get(i_spin).unwrap().clone();
        let i_lumo = scf_data.lumo.get(i_spin).unwrap().clone();

        homo = homo.max(i_homo);
        lumo = lumo.min(i_lumo);
    }

    (start_mo..homo+1, lumo..num_state)
}


pub fn generate_ri3mo_rayon_for_pt2_and_rpa(scf_data: &mut SCF) {
    use crate::post_scf_analysis::{split_indices_by_spin_occ, format_indices};

    let (occ_range, vir_range) = determine_ri3mo_size_for_pt2_and_rpa(&scf_data);
    if scf_data.mol.ctrl.print_level>1 {
        //println!("generate RI3MO only for occ_range:{:?}, vir_range:{:?}", &occ_range, &vir_range);
        let spin_orb_indices = split_indices_by_spin_occ(&scf_data.occupation, 0.5);
        let (alpha_occ, alpha_vir) = &spin_orb_indices[0];
        debug!("Occupied orbitals (alpha): {}", format_indices(alpha_occ));
        debug!("Virtual orbitals (alpha): {}", format_indices(alpha_vir));
        if matches!(scf_data.scftype, SCFType::UHF | SCFType::ROHF) {
            let (beta_occ,  beta_vir)  = &spin_orb_indices[1];
            debug!("Occupied orbitals (beta): {}", format_indices(beta_occ));
            debug!("Virtual orbitals (beta): {}", format_indices(beta_vir));
        }
    };
    scf_data.generate_ri3mo_rayon(vir_range, occ_range);
    debug!("Finish the RI3MO generation");
}


// vj, vk without dependency on SCF struct
//
pub fn vj_upper_with_ri_v(
                    ri3fn: &Option<RIFull<f64>>,
                    dm: &Vec<MatrixFull<f64>>, 
                    spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    
    let mut vj: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
    if let Some(ri3fn) = ri3fn {
        let num_basis = ri3fn.size[0];
        let num_auxbas = ri3fn.size[2];
        let npair = num_basis*(num_basis+1)/2;
        for i_spin in (0..spin_channel) {
            //let mut tmp_mu = vec![0.0f64;num_auxbas];
            let mut vj_spin = &mut vj[i_spin];
            *vj_spin = MatrixUpper::new(npair,0.0f64);
            ri3fn.iter_auxbas(0..num_auxbas).unwrap().enumerate().for_each(|(i,m)| {
                //prepare \sum_{kl}D_{kl}*M_{kl}^{\mu}
                let tmp_mu =
                    m.chunks_exact(num_basis).zip(dm[i_spin].data.chunks_exact(num_basis))
                        .fold(0.0_f64,|acc, (m,d)| {
                            acc + m.iter().zip(d.iter()).map(|value| value.0*value.1).sum::<f64>()
                        });
                // filter out the upper part of  M_{ij}^{\mu}
                let m_ij_upper = m.iter().enumerate().filter(|(i,v)| i%num_basis<=i/num_basis)
                    .map(|(i,v)| v );

                // fill vj[i_spin] with the contribution from the given {\mu}:
                //
                // M_{ij}^{\mu}*(\sum_{kl}D_{kl}*M_{kl}^{\mu})
                //
                vj_spin.data.iter_mut().zip(m_ij_upper)
                    .for_each(|value| *value.0 += *value.1*tmp_mu); 
            });
        }
    };

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vj[i_spin].data.iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };
    vj
}
pub fn vj_upper_with_ri_v_sync(
                ri3fn: &Option<RIFull<f64>>,
                dm: &Vec<MatrixFull<f64>>, 
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    let mut vj: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
    //// In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    //// In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    //let default_omp_num_threads = unsafe {openblas_get_num_threads()};
    //unsafe{openblas_set_num_threads(1)};
    

    if let Some(ri3fn) = ri3fn {
    let num_basis = ri3fn.size[0];
    let num_auxbas = ri3fn.size[2];
    let npair = num_basis*(num_basis+1)/2;
        for i_spin in (0..spin_channel) {
            //let mut tmp_mu = vec![0.0f64;num_auxbas];
            let mut vj_spin = &mut vj[i_spin];
            *vj_spin = MatrixUpper::new(npair,0.0f64);

            let (sender, receiver) = channel();
            ri3fn.par_iter_auxbas(0..num_auxbas).unwrap().enumerate().for_each_with(sender, |s, (i,m)| {
                //prepare \sum_{kl}D_{kl}*M_{kl}^{\mu} for each \mu -> tmp_mu
                let tmp_mu =
                    m.chunks_exact(num_basis).zip(dm[i_spin].data.chunks_exact(num_basis))
                        .fold(0.0_f64,|acc, (m,d)| {
                            acc + m.iter().zip(d.iter()).map(|value| value.0*value.1).sum::<f64>()
                        });
                // filter out the upper part (ij pair) of M_{ij}^{\mu} for each \mu -> m_ij_upper
                let m_ij_upper = m.iter().enumerate().filter(|(i,v)| i%num_basis<=i/num_basis)
                    .map(|(i,v)| v.clone() ).collect_vec();
                s.send((m_ij_upper,tmp_mu)).unwrap();
            });
            // fill vj[i_spin] with the contribution from the given {\mu}:
            //
            // M_{ij}^{\mu}*(\sum_{kl}D_{kl}*M_{kl}^{\mu})
            //
            receiver.iter().for_each(|(m_ij_upper, tmp_mu)| {
                vj_spin.data.iter_mut().zip(m_ij_upper.iter())
                    .for_each(|value| *value.0 += *value.1*tmp_mu); 
            });


            //vj_spin.data.par_iter_mut().zip(m_ij_upper.par_iter())
            //    .for_each(|value| *value.0 += *value.1*tmp_mu); 
        }
    };

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vj[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };

    //// reuse the default omp_num_threads setting
    //unsafe{openblas_set_num_threads(default_omp_num_threads)};

    vj
}

pub fn vj_upper_with_rimatr_sync_mpi(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                dm: &Vec<MatrixFull<f64>>, 
                spin_channel: usize, scaling_factor: f64,
                mpi_operator: &Option<MPIOperator>)  -> Vec<MatrixUpper<f64>> {
    #[cfg(feature = "mpi")]
    if let Some(mpi_op) = &mpi_operator {
        let mut vj_vec = vj_upper_with_rimatr_sync(ri3fn, dm, spin_channel, scaling_factor);
        for i_spin in 0..spin_channel {
            let vj = &mut vj_vec[i_spin];
            let mut tot_vj = mpi_reduce(&mpi_op.world, vj.data_ref().unwrap(), 0, &SystemOperation::sum());
            mpi_broadcast_vector(&mpi_op.world, &mut tot_vj, 0);
            //if mpi_op.rank == 0 {
                vj.data = tot_vj;
            //}
        }
        vj_vec
    } else
    {
        vj_upper_with_rimatr_sync(ri3fn, dm, spin_channel, scaling_factor)
    }
    #[cfg(not(feature = "mpi"))]
    { vj_upper_with_rimatr_sync(ri3fn, dm, spin_channel, scaling_factor) }
}

pub fn vj_upper_with_rimatr_sync(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                dm: &Vec<MatrixFull<f64>>, 
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    vj_upper_with_rimatr_sync_v02(ri3fn,dm,spin_channel,scaling_factor)
}

pub fn vj_upper_with_rimatr_sync_v01(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                dm: &Vec<MatrixFull<f64>>, 
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    let mut vj: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
    //// In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    //// In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    //let default_omp_num_threads = unsafe {openblas_get_num_threads()};
    //unsafe{openblas_set_num_threads(1)};
    
    if let Some((ri3fn,basbas2baspar,baspar2basbas)) = ri3fn {
        let num_basis = basbas2baspar.size[0];
        let num_baspar = ri3fn.size[0];
        let num_auxbas = ri3fn.size[1];
        //let npair = num_basis*(num_basis+1)/2;
        for i_spin in (0..spin_channel) {
            //let mut tmp_mu = vec![0.0f64;num_auxbas];
            let mut vj_spin = &mut vj[i_spin];
            *vj_spin = MatrixUpper::new(num_baspar,0.0f64);
            let dm_s = &dm[i_spin];

            let (sender, receiver) = channel();
            ri3fn.par_iter_columns_full().enumerate().for_each_with(sender, |s, (i,m)| {
                //prepare \sum_{kl}D_{kl}*M_{kl}^{\mu} for each \mu -> tmp_mu
                let riupper = MatrixUpperSlice::from_vec(m);
                let mut tmp_mu =
                    m.iter().zip(dm_s.iter_matrixupper().unwrap()).fold(0.0_f64, |acc,(m,d)| {
                        acc + *m * (*d)
                    });
                //let diagonal_term = riupper.get_diagonal_terms().unwrap()
                //    .iter().zip(dm[i_spin].iter_diagonal().unwrap()).fold(0.0f64, |acc, (v1,v2)| {
                //        acc + *v1*v2
                //});
                let diagonal_term = riupper.iter_diagonal()
                    .zip(dm[i_spin].iter_diagonal().unwrap()).fold(0.0f64, |acc, (v1,v2)| {
                        acc + *v1*v2
                });

                tmp_mu = 2.0_f64*tmp_mu - diagonal_term;

                let m_ij_upper = m.iter().map(|v| *v*tmp_mu).collect_vec();
                s.send(m_ij_upper).unwrap();
            });
            // fill vj[i_spin] with the contribution from the given {\mu}:
            //
            // M_{ij}^{\mu}*(\sum_{kl}D_{kl}*M_{kl}^{\mu})
            //
            receiver.iter().for_each(|(m_ij_upper)| {
                vj_spin.data.iter_mut().zip(m_ij_upper.iter())
                    .for_each(|value| *value.0 += *value.1); 
            });


            //vj_spin.data.par_iter_mut().zip(m_ij_upper.par_iter())
            //    .for_each(|value| *value.0 += *value.1*tmp_mu); 
        }
    };

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vj[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };

    //// reuse the default omp_num_threads setting
    //unsafe{openblas_set_num_threads(default_omp_num_threads)};
    //vj[0].formated_output(5, "full");

    vj
}

pub fn vj_upper_with_rimatr_sync_v02(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                dm: &Vec<MatrixFull<f64>>, 
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {

    //let default_omp_num_threads = omp_get_num_threads_wrapper();

    let mut vj: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
    if let Some((ri3fn,basbas2baspar,baspar2basbas)) = ri3fn {
        let num_basis = basbas2baspar.size[0];
        let num_baspar = ri3fn.size[0];
        let num_auxbas = ri3fn.size[1];
        for i_spin in (0..spin_channel) {
            //let mut tmp_mu = vec![0.0f64;num_auxbas];
            let mut vj_spin = &mut vj[i_spin];
            *vj_spin = MatrixUpper::new(num_baspar,0.0f64);
            //let mut dm_s = dm[i_spin].clone();
            //dm_s.iter_diagonal_mut().unwrap().for_each(|x| *x = *x/2.0);

            let mut dm_s_upper = MatrixUpper::from_vec(num_baspar,dm[i_spin].iter_matrixupper().unwrap().map(|x| *x).collect_vec()).unwrap();
            dm_s_upper.iter_diagonal_mut().for_each(|x| {*x = *x/2.0});

            let mut tmp_v = vec![0.0;num_auxbas];

            _dgemv(ri3fn, &dm_s_upper.data, &mut tmp_v, 'T', 2.0, 0.0, 1, 1);

            _dgemv(ri3fn, &tmp_v, &mut vj_spin.data, 'N',1.0,0.0,1,1);

        }
    }

    vj
}


// Just for test, no need to use vj_full because it's always symmetric
pub fn vj_full_with_ri_v(
                    ri3fn: &Option<RIFull<f64>>,
                    dm: &Vec<MatrixFull<f64>>, 
                    spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixFull<f64>> {
    
    let mut vj: Vec<MatrixFull<f64>> = vec![MatrixFull::new([1,1],0.0f64),MatrixFull::new([1,1],0.0f64)];
    if let Some(ri3fn) = ri3fn {
        let num_basis = ri3fn.size[0];
        let num_auxbas = ri3fn.size[2];
        //let npair = num_basis*(num_basis+1)/2;
        for i_spin in (0..spin_channel) {
            //let mut tmp_mu = vec![0.0f64;num_auxbas];
            let mut vj_spin = &mut vj[i_spin];
            *vj_spin = MatrixFull::new([num_basis, num_basis],0.0f64);
            ri3fn.iter_auxbas(0..num_auxbas).unwrap().enumerate().for_each(|(i,m)| {
                //prepare \sum_{kl}D_{kl}*M_{kl}^{\mu}
                let tmp_mu =
                    m.chunks_exact(num_basis).zip(dm[i_spin].data.chunks_exact(num_basis))
                        .fold(0.0_f64,|acc, (m,d)| {
                            acc + m.iter().zip(d.iter()).map(|value| value.0*value.1).sum::<f64>()
                        });

                // fill vj[i_spin] with the contribution from the given {\mu}:
                //
                // M_{ij}^{\mu} * (\sum_{kl}D_{kl}*M_{kl}^{\mu})
                //
                vj_spin.data.iter_mut().zip(m)
                    .for_each(|value| *value.0 += *value.1*tmp_mu); 
            });
        }
    };

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vj[i_spin].data.iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };
    vj
}
pub fn vk_full_fromdm_with_ri_v(
                    ri3fn: &Option<RIFull<f64>>,
                    dm: &Vec<MatrixFull<f64>>, 
                    spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixFull<f64>> {
    
    let mut vk: Vec<MatrixFull<f64>> = vec![MatrixFull::new([1,1],0.0f64),MatrixFull::new([1,1],0.0f64)];
    if let Some(ri3fn) = ri3fn {
        let num_basis = ri3fn.size[0];
        let num_auxbas = ri3fn.size[2];
        //let npair = num_basis*(num_basis+1)/2;
        for i_spin in (0..spin_channel) {
            //let mut tmp_mu = vec![0.0f64;num_auxbas];
            let mut vk_spin = &mut vk[i_spin];
            *vk_spin = MatrixFull::new([num_basis, num_basis],0.0f64);
            ri3fn.iter_auxbas(0..num_auxbas).unwrap().enumerate().for_each(|(i,m)| {
                //prepare \sum_{l}D_{jl}*M_{kl}^{\mu}
                let mut tmp_mu = MatrixFull::from_vec([num_basis, num_basis], m.to_vec()).unwrap();
                let mut dm_m = MatrixFull::new([num_basis, num_basis], 0.0f64);
                dm_m.lapack_dgemm(&mut dm[i_spin].clone(), &mut tmp_mu, 'N', 'T', 1.0, 0.0);

                // fill vk[i_spin] with the contribution from the given {\mu}:
                //
                // \sum_j M_{ij}^{\mu} * (\sum_{l}D_{jl}*M_{kl}^{\mu})
                //
                vk_spin.lapack_dgemm(&mut tmp_mu.clone(), &mut dm_m, 'N', 'N', 1.0, 1.0);
            });
        }
    };

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vk[i_spin].data.iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };
    vk
}

pub fn vk_upper_with_ri_v_use_dm_only_sync(
                ri3fn: &Option<RIFull<f64>>,
                dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    let default_omp_num_threads = omp_get_num_threads_wrapper();
    //let mut bm = RIFull::new([num_state,num_basis,num_auxbas], 0.0f64);
    let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];


    if let Some(ri3fn) = ri3fn {
        let num_basis = dm[0].size()[0];
        let num_baspair = (num_basis+1)*num_basis/2;
        let num_auxbas = ri3fn.size[2];
        for i_spin in 0..spin_channel {
            let mut vk_s = &mut vk[i_spin];
            *vk_s = MatrixUpper::new(num_baspair,0.0_f64);
            let dm_s = &dm[i_spin];
            //dm_s.formated_output(5, "upper");
            let (sender, receiver) = channel();
            ri3fn.par_iter_auxbas(0..num_auxbas).unwrap().for_each_with(sender,|s, m| {

                // To ensure the efficiency, we disable the openmp ability of openblase within the rayon parallel region
                omp_set_num_threads_wrapper(1);

                let mut tmp_mat = MatrixFull::new([num_basis,num_basis],0.0_f64);
                let mut reduced_ri3fn = MatrixFullSlice {
                    size:  &[num_basis,num_basis],
                    indicing: &[1,num_basis],
                    data: m,
                };
                //_dgemm(&reduced_ri3fn, (0..num_basis,0..num_basis), 'N', 
                //       dm_s, (0..num_basis,0..num_basis), 'N', 
                //       &mut tmp_mat, (0..num_basis,0..num_basis), 1.0, 0.0);
                //let mut vk_sm = MatrixFull::new([num_basis,num_basis],0.0_f64);
                //_dgemm(&tmp_mat, (0..num_basis,0..num_basis), 'N', 
                //       &reduced_ri3fn, (0..num_basis,0..num_basis), 'T', 
                //       &mut vk_sm, (0..num_basis,0..num_basis), 1.0, 0.0);
                //tmp_mat = ri3fn \cdot dm
                _dsymm(&reduced_ri3fn, dm_s, &mut tmp_mat, 'L', 'U', 1.0, 0.0);
                let mut vk_sm = MatrixFull::new([num_basis,num_basis],0.0_f64);
                //vk_sm = ri3fn \cdot dm \cdot ri3fn
                _dsymm(&reduced_ri3fn, &tmp_mat, &mut vk_sm, 'R', 'U', 1.0, 0.0);

                s.send(vk_sm.to_matrixupper()).unwrap();
            });

            receiver.into_iter().for_each(|vk_mu_upper| {
                vk_s.data.par_iter_mut()
                    .zip(vk_mu_upper.data.par_iter()).for_each(|value| {
                    *value.0 += *value.1
                })
            });
        }
    }

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };

    // reuse the default omp_num_threads setting
    omp_set_num_threads_wrapper(default_omp_num_threads);

    vk
}



pub fn vk_upper_with_rimatr_use_dm_only_sync(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {

    vk_upper_with_rimatr_use_dm_only_sync_v01(ri3fn, dm, spin_channel, scaling_factor)
}

pub fn vk_upper_with_rimatr_use_dm_only_sync_v01(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = omp_get_num_threads_wrapper();
    //let mut bm = RIFull::new([num_state,num_basis,num_auxbas], 0.0f64);
    let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];

    if let Some((ri3fn,basbas2baspar,baspar2basbas)) = ri3fn {
        let num_basis = dm[0].size()[0];
        let num_baspair = (num_basis+1)*num_basis/2;
        //let num_auxbas = ri3fn.size[2];
        for i_spin in 0..spin_channel {
            let mut vk_s = &mut vk[i_spin];
            *vk_s = MatrixUpper::new(num_baspair,0.0_f64);
            let dm_s = &dm[i_spin];
            let (sender, receiver) = channel();
            ri3fn.par_iter_columns_full().for_each_with(sender,|s, m| {

                // To ensure the efficiency, we disable the openmp ability of openblase within the rayon parallel region
                omp_set_num_threads_wrapper(1);

                let mut tmp_mat = MatrixFull::new([num_basis,num_basis],0.0_f64);
                let mut reduced_ri3fn = MatrixFull::new([num_basis,num_basis],0.0_f64);

                reduced_ri3fn.iter_matrixupper_mut().unwrap().zip(m.iter()).for_each(|(to, from)| {*to = *from});

                _dsymm(&reduced_ri3fn, dm_s, &mut tmp_mat, 'L', 'U', 1.0, 0.0);
                let mut vk_sm = MatrixFull::new([num_basis,num_basis],0.0_f64);
                _dsymm(&reduced_ri3fn, &tmp_mat, &mut vk_sm, 'R', 'U', 1.0, 0.0);

                s.send(vk_sm.to_matrixupper()).unwrap();
            });

            receiver.into_iter().for_each(|vk_mu_upper| {
                vk_s.data.iter_mut()
                    .zip(vk_mu_upper.data.iter()).for_each(|value| {
                    *value.0 += *value.1
                })
            });
        }
    }

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };

    // reuse the default omp_num_threads setting
    omp_set_num_threads_wrapper(default_omp_num_threads);

    vk
}

pub fn vk_upper_with_rimatr_use_dm_only_sync_v02(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = omp_get_num_threads_wrapper();
    //utilities::omp_set_num_threads_wrapper(1);
    //let mut bm = RIFull::new([num_state,num_basis,num_auxbas], 0.0f64);
    let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];

    if let Some((ri3fn,basbas2baspar,baspar2basbas)) = ri3fn {
        let num_basis = dm[0].size()[0];
        let num_baspair = ri3fn.size()[0];
        let num_auxbas = ri3fn.size()[1];
        //let num_auxbas = ri3fn.size[2];
        for i_spin in 0..spin_channel {
            let mut vk_s = &mut vk[i_spin];
            *vk_s = MatrixUpper::new(num_baspair,0.0_f64);
            //let dm_s = &dm[i_spin];
            let dm_s = _power_rayon_for_symmetric_matrix(&dm[i_spin], 0.5, SQRT_THRESHOLD).unwrap();
            let batch_num_auxbas = utilities::balancing(num_auxbas, rayon::current_num_threads());
            let (sender, receiver) = channel();
            batch_num_auxbas.par_iter().for_each_with(sender, |s,loc_auxbas| {

                // To ensure the efficiency, we disable the openmp ability of openblase within the rayon parallel region
                omp_set_num_threads_wrapper(1);

                let mut tmp_mat = MatrixFull::new([num_basis,num_basis],0.0_f64);
                let mut reduced_ri3fn = MatrixFull::new([num_basis,num_basis],0.0_f64);
                let mut vk_sm = MatrixFull::new([num_basis,num_basis],0.0_f64);
                ri3fn.iter_columns(loc_auxbas.clone()).for_each(|m| {
                    reduced_ri3fn.iter_matrixupper_mut().unwrap().zip(m.iter()).for_each(|(to, from)| {*to = *from});
                    //_dsymm(&reduced_ri3fn, dm_s, &mut tmp_mat, 'L', 'U', 1.0, 0.0);
                    //_dsymm(&reduced_ri3fn, &tmp_mat, &mut vk_sm, 'R', 'U', 1.0, 1.0);
                    _dsymm(&reduced_ri3fn, &dm_s, &mut tmp_mat, 'L', 'U', 1.0, 0.0);
                    _dsyrk(&tmp_mat, &mut vk_sm, 'U', 'N', 1.0, 1.0)
                });
                s.send(vk_sm.to_matrixupper()).unwrap();
            });

            receiver.into_iter().for_each(|vk_mu_upper| {
                vk_s.data.par_iter_mut()
                    .zip(vk_mu_upper.data.par_iter()).for_each(|value| {
                    *value.0 += *value.1
                })
            });
        }
    }

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };

    // reuse the default omp_num_threads setting
    omp_set_num_threads_wrapper(default_omp_num_threads);

    vk
}

pub fn vk_upper_with_rimatr_use_dm_only_sync_mpi(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64, mpi_operator: &Option<MPIOperator>)  -> Vec<MatrixUpper<f64>> {
    #[cfg(feature = "mpi")]
    if let Some(mpi_op) = &mpi_operator {
        let mut vk_vec = vk_upper_with_rimatr_use_dm_only_sync_v02(ri3fn, dm, spin_channel, scaling_factor);
        for i_spin in 0..spin_channel {
            let vk = &mut vk_vec[i_spin];
            let mut tot_vk = mpi_reduce(&mpi_op.world, vk.data_ref().unwrap(), 0, &SystemOperation::sum());
            mpi_broadcast(&mpi_op.world, &mut tot_vk, 0);
            //if mpi_op.rank == 0 {
                vk.data = tot_vk;
            //}
        };
        vk_vec
    } else
    {
        vk_upper_with_rimatr_use_dm_only_sync_v02(ri3fn, dm, spin_channel, scaling_factor)
    }
    #[cfg(not(feature = "mpi"))]
    { vk_upper_with_rimatr_use_dm_only_sync_v02(ri3fn, dm, spin_channel, scaling_factor) }
}

pub fn vk_upper_with_rimatr_sync_mpi(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                eigv: &[MatrixFull<f64>;2], 
                num_elec: &[f64;3], occupation: &[Vec<f64>;2],
                spin_channel: usize, scaling_factor: f64,
                mpi_operator: &Option<MPIOperator>)  -> Vec<MatrixUpper<f64>> {
    #[cfg(feature = "mpi")]
    if let Some(mpi_op) = &mpi_operator {
        let mut vk_vec = vk_upper_with_rimatr_sync_v03(ri3fn,eigv,num_elec,occupation,spin_channel,scaling_factor);
        for i_spin in 0..spin_channel {
            let vk = &mut vk_vec[i_spin];
            let mut tot_vk = mpi_reduce(&mpi_op.world, vk.data_ref().unwrap(), 0, &SystemOperation::sum());
            mpi_broadcast(&mpi_op.world, &mut tot_vk, 0);
            //if mpi_op.rank == 0 {
                vk.data = tot_vk;
            //}
        };
        vk_vec
    } else
    {
        vk_upper_with_rimatr_sync_v03(ri3fn,eigv,num_elec,occupation,spin_channel,scaling_factor)
    }
    #[cfg(not(feature = "mpi"))]
    { vk_upper_with_rimatr_sync_v03(ri3fn,eigv,num_elec,occupation,spin_channel,scaling_factor) }
}
pub fn vk_upper_with_rimatr_sync(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                eigv: &[MatrixFull<f64>;2], 
                num_elec: &[f64;3], occupation: &[Vec<f64>;2],
                //dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    vk_upper_with_rimatr_sync_v03(ri3fn,eigv,num_elec,occupation,spin_channel,scaling_factor)
}

pub fn vk_upper_with_rimatr_sync_v01(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                eigv: &[MatrixFull<f64>;2], 
                num_elec: &[f64;3], occupation: &[Vec<f64>;2],
                //dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = omp_get_num_threads_wrapper();
    //let mut bm = RIFull::new([num_state,num_basis,num_auxbas], 0.0f64);
    let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];

    if let Some((ri3fn,basbas2baspar,baspar2basbas)) = ri3fn {
        let num_basis = eigv[0].size()[0];
        let num_baspair = (num_basis+1)*num_basis/2;
        //let num_auxbas = ri3fn.size[2];
        for i_spin in 0..spin_channel {
            let mut vk_s = &mut vk[i_spin];
            *vk_s = MatrixUpper::new(num_baspair,0.0_f64);
            let eigv_s = &eigv[i_spin];
            let nw = occupied_orbital_count(&occupation[i_spin]);
            //let nw = num_elec[i_spin+1].ceil() as usize;
            if nw>0 {
                let mut tmp_mat = MatrixFull::new([num_basis,nw],0.0_f64);
                tmp_mat.data.iter_mut().zip(eigv_s.iter_submatrix(0..num_basis,0..nw))
                    .for_each(|value| {*value.0 = *value.1});
                let occ_s = &occupation[i_spin][0..nw];
                tmp_mat.data.par_chunks_exact_mut(tmp_mat.size[0]).zip(occ_s.par_iter()).for_each(|(to_value, from_value)| {
                        to_value.iter_mut().for_each(|to_value| {*to_value = *to_value*from_value.sqrt()});
                });
                let reduced_eigv_s = tmp_mat;

                //let dm_s = &dm[i_spin];

                let (sender, receiver) = channel();
                ri3fn.par_iter_columns_full().for_each_with(sender,|s, m| {

                    // To ensure the efficiency, we disable the openmp ability of openblase within the rayon parallel region
                    omp_set_num_threads_wrapper(1);

                    //let mut tmp_mat = MatrixFull::new([num_basis,num_basis],0.0_f64);
                    let mut reduced_ri3fn = MatrixFull::new([num_basis,num_basis],0.0_f64);

                    reduced_ri3fn.iter_matrixupper_mut().unwrap().zip(m.iter()).for_each(|(to, from)| {*to = *from});

                    let mut tmp_mc = MatrixFull::new([num_basis,nw],0.0_f64);
                    //tmp_mc = ri3fn \cdot eigv \cdot occ.sqrt()
                    _dsymm(&reduced_ri3fn, &reduced_eigv_s, &mut tmp_mc, 'L', 'U', 1.0, 0.0);

                    let mut vk_sm = MatrixFull::new([num_basis,num_basis],0.0_f64);
                    _dsyrk(&tmp_mc, &mut vk_sm, 'U', 'N', 1.0, 0.0);

                    s.send(vk_sm.to_matrixupper()).unwrap();
                });

                receiver.into_iter().for_each(|vk_mu_upper| {
                    vk_s.data.par_iter_mut()
                        .zip(vk_mu_upper.data.par_iter()).for_each(|value| {
                        *value.0 += *value.1
                    })
                });
            }
        }
    }

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };

    // reuse the default omp_num_threads setting
    omp_set_num_threads_wrapper(default_omp_num_threads);

    vk
}

/// a new vk version with the parallelization giving to openmk.
pub fn vk_upper_with_rimatr_sync_v02(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                eigv: &[MatrixFull<f64>;2], 
                num_elec: &[f64;3], occupation: &[Vec<f64>;2],
                //dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {

    let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];

    if let Some((ri3fn,basbas2baspar,baspar2basbas)) = ri3fn {
        let num_basis = eigv[0].size()[0];
        let num_baspair = (num_basis+1)*num_basis/2;
        //let num_auxbas = ri3fn.size[2];
        for i_spin in 0..spin_channel {
            let mut vk_s = &mut vk[i_spin];
            let mut vk_sm = MatrixFull::new([num_basis,num_basis],0.0_f64);
            //*vk_s = MatrixUpper::new(num_baspair,0.0_f64);
            let eigv_s = &eigv[i_spin];
            // now locate the highest obital that has electron with occupation largger than 1.0e-4
            let nw = occupied_orbital_count(&occupation[i_spin]);
            if nw>0 {
                let mut tmp_mat = MatrixFull::new([num_basis,nw],0.0_f64);
                tmp_mat.data.iter_mut().zip(eigv_s.iter_submatrix(0..num_basis,0..nw))
                    .for_each(|value| {*value.0 = *value.1});
                let occ_s = &occupation[i_spin][0..nw];
                tmp_mat.data.chunks_exact_mut(tmp_mat.size[0]).zip(occ_s.iter()).for_each(|(to_value, from_value)| {
                        to_value.iter_mut().for_each(|to_value| {*to_value = *to_value*from_value.sqrt()});
                });
                let reduced_eigv_s = tmp_mat;

                //let dm_s = &dm[i_spin];

                //let (sender, receiver) = channel();

                let mut tmp_mc = MatrixFull::new([num_basis,nw],0.0_f64);
                let mut reduced_ri3fn = MatrixFull::new([num_basis,num_basis],0.0_f64);

                ri3fn.iter_columns_full().for_each(|m| {
                    //let mut tmp_mat = MatrixFull::new([num_basis,num_basis],0.0_f64);
                    reduced_ri3fn.iter_matrixupper_mut().unwrap().zip(m.iter()).for_each(|(to, from)| {*to = *from});

                    //tmp_mc = ri3fn \cdot eigv
                    _dsymm(&reduced_ri3fn, &reduced_eigv_s, &mut tmp_mc, 'L', 'U', 1.0, 0.0);

                    _dsyrk(&tmp_mc, &mut vk_sm, 'U', 'N', 1.0, 1.0);

                    //s.send(vk_sm.to_matrixupper()).unwrap();
                });
                *vk_s = vk_sm.to_matrixupper();
            } else {
              *vk_s = MatrixUpper::new(num_baspair,0.0_f64);
            }
        }
    }

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };

    //// reuse the default omp_num_threads setting
    //utilities::omp_set_num_threads_wrapper(default_omp_num_threads);

    vk
}

pub fn vk_upper_with_rimatr_sync_v03(
                ri3fn: &Option<(MatrixFull<f64>,MatrixFull<usize>,Vec<[usize;2]>)>,
                eigv: &[MatrixFull<f64>;2], 
                num_elec: &[f64;3], occupation: &[Vec<f64>;2],
                //dm: &Vec<MatrixFull<f64>>,
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = omp_get_num_threads_wrapper();
    //utilities::omp_set_num_threads_wrapper(1);
    //let mut bm = RIFull::new([num_state,num_basis,num_auxbas], 0.0f64);
    //let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
    let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::empty(),MatrixUpper::empty()];

    if let Some((ri3fn,basbas2baspar,baspar2basbas)) = ri3fn {
        let num_basis = eigv[0].size()[0];
        let num_baspair = ri3fn.size()[0];
        let num_auxbas = ri3fn.size()[1];
        //let num_auxbas = ri3fn.size[2];
        for i_spin in 0..spin_channel {
            let mut vk_s = &mut vk[i_spin];
            *vk_s = MatrixUpper::new(num_baspair,0.0_f64);
            let eigv_s = if !eigv[i_spin].data.is_empty() {
                &eigv[i_spin]
            } else { // use the eigv[0] again for ROHF case.
                &eigv[0]
            };
            // now locate the highest obital that has electron with occupation largger than 1.0e-4
            let elec_spin = num_elec[i_spin+1].ceil() as usize;
            let nw = if elec_spin == 0 {0} else {occupied_orbital_count(&occupation[i_spin])};
            if nw>0 {
                let mut tmp_mat = MatrixFull::new([num_basis,nw],0.0_f64);
                tmp_mat.data.iter_mut().zip(eigv_s.iter_submatrix(0..num_basis,0..nw))
                    .for_each(|value| {*value.0 = *value.1});
                let occ_s = &occupation[i_spin][0..nw];
                tmp_mat.data.par_chunks_exact_mut(tmp_mat.size[0]).zip(occ_s.par_iter()).for_each(|(to_value, from_value)| {
                        to_value.iter_mut().for_each(|to_value| {*to_value = *to_value*from_value.sqrt()});
                });
                let reduced_eigv_s = tmp_mat;

                //let dm_s = &dm[i_spin];

                let batch_num_auxbas = utilities::balancing(num_auxbas, rayon::current_num_threads());
                let (sender, receiver) = channel();
                batch_num_auxbas.par_iter().for_each_with(sender, |s, loc_auxbas| {

                    // To ensure the efficiency, we disable the openmp ability of openblase within the rayon parallel region
                    omp_set_num_threads_wrapper(1);

                    let mut reduced_ri3fn = MatrixFull::new([num_basis,num_basis],0.0_f64);
                    let mut vk_sm = MatrixFull::new([num_basis,num_basis],0.0_f64);
                    let mut tmp_mc = MatrixFull::new([num_basis,nw],0.0_f64);
                    ri3fn.iter_columns(loc_auxbas.clone()).for_each(|m| {
                        reduced_ri3fn.iter_matrixupper_mut().unwrap().zip(m.iter()).for_each(|(to, from)| {*to = *from});
                        _dsymm(&reduced_ri3fn, &reduced_eigv_s, &mut tmp_mc, 'L', 'U', 1.0, 0.0);
                        _dsyrk(&tmp_mc, &mut vk_sm, 'U', 'N', 1.0, 1.0);
                    });
                    s.send(vk_sm.to_matrixupper()).unwrap();
                });

                receiver.into_iter().for_each(|vk_mu_upper| {
                    vk_s.data.par_iter_mut()
                        .zip(vk_mu_upper.data.par_iter()).for_each(|value| {
                        *value.0 += *value.1
                    })
                });
            }
        }
    }

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };

    // reuse the default omp_num_threads setting
    omp_set_num_threads_wrapper(default_omp_num_threads);

    vk
}

//==========================need to be checked=============================
pub fn vk_upper_with_ri_v_sync(
                ri3fn: &Option<RIFull<f64>>,
                eigv: &[MatrixFull<f64>;2], 
                num_elec: &[f64;3], occupation: &[Vec<f64>;2],
                spin_channel: usize, scaling_factor: f64)  -> Vec<MatrixUpper<f64>> {
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = omp_get_num_threads_wrapper();
    //utilities::omp_set_num_threads_wrapper(1);

    //let mut bm = RIFull::new([num_state,num_basis,num_auxbas], 0.0f64);
    let mut vk: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1,0.0f64),MatrixUpper::new(1,0.0f64)];
    
    if let Some(ri3fn) = ri3fn {
        let num_basis = eigv[0].size[0];
        let num_state = eigv[0].size[1];
        let num_auxbas = ri3fn.size[2];
        let npair = num_basis*(num_basis+1)/2;
        for i_spin in 0..spin_channel {
            let mut vk_s = &mut vk[i_spin];
            *vk_s = MatrixUpper::new(npair,0.0_f64);
            let eigv_s = &eigv[i_spin];
            let nw = occupied_orbital_count(&occupation[i_spin]);
            if nw>0 {
                let mut tmp_mat = MatrixFull::new([num_basis,nw],0.0_f64);
                tmp_mat.data.iter_mut().zip(eigv_s.iter_submatrix(0..num_basis,0..nw))
                    .for_each(|value| {*value.0 = *value.1});
                let reduced_eigv_s = tmp_mat;
                let occ_s = &occupation[i_spin][0..nw];
                //let mut tmp_b = MatrixFull::new([num_basis,num_basis],0.0_f64);
                let (sender, receiver) = channel();
                ri3fn.par_iter_auxbas(0..num_auxbas).unwrap().for_each_with(sender, |s, m| {

                    // To ensure the efficiency, we disable the openmp ability of openblase within the rayon parallel region
                    omp_set_num_threads_wrapper(1);

                    let mut reduced_ri3fn = MatrixFullSlice {
                        size:  &[num_basis,num_basis], 
                        indicing: &[1,num_basis],
                        data: m,
                    };
                    //tmp_mat: copy of related eigenvalue; reduced_ri3fn: certain part of ri3fn
                    let mut tmp_mc = MatrixFull::new([num_basis,nw],0.0_f64);
                    //tmp_mc = ri3fn \cdot eigv
                    tmp_mc.to_matrixfullslicemut().lapack_dgemm(&reduced_ri3fn, &reduced_eigv_s.to_matrixfullslice(), 'N', 'N', 1.0, 0.0);
                    //tmp_mat = tmp_mc (ri3fn \cdot eigv)
                    let mut tmp_mat = tmp_mc.clone();
                    //tmp_mat = tmp_mc * occ
                    tmp_mat.data.chunks_exact_mut(tmp_mat.size[0]).zip(occ_s.iter()).for_each(|(to_value, from_value)| {
                        to_value.iter_mut().for_each(|to_value| {*to_value = *to_value*from_value});
                    });

                    let mut vk_mu = MatrixFull::new([num_basis,num_basis],0.0_f64);
                    // vk_mu = tmp_mat \cdot tmp_mc.T  ((ri3fn \cdot eigv * occ) \cdot (ri3fn \cdot eigv)^T)
                    vk_mu.lapack_dgemm(&mut tmp_mat, &mut tmp_mc, 'N', 'T', 1.0, 0.0);

                    // filter out the upper part of vk_mu
                    let mut tmp_mat = MatrixUpper::from_vec(npair, vk_mu.data.iter().enumerate().filter(|(i,v)| i%num_basis<=i/num_basis)
                        .map(|(i,v)| v.clone() ).collect_vec()).unwrap();

                    s.send(tmp_mat).unwrap()
                });
                receiver.into_iter().for_each(|vk_mu_upper| {
                    vk_s.data.par_iter_mut()
                        .zip(vk_mu_upper.data.par_iter()).for_each(|value| {
                        *value.0 += *value.1
                    })
                });
            }
        }
        //// for each spin channel
        //vk.iter_mut().zip(eigv.iter()).for_each(|(vk_s,eigv_s)| {
        //});

    };

    if scaling_factor!=1.0f64 {
        for i_spin in (0..spin_channel) {
            vk[i_spin].data.par_iter_mut().for_each(|f| *f = *f*scaling_factor)
        }
    };

    // reuse the default omp_num_threads setting
    omp_set_num_threads_wrapper(default_omp_num_threads);


    vk
}

pub fn scf(mol:Molecule, mpi_operator: &Option<MPIOperator>) -> anyhow::Result<SCF> {
    let dt0 = time::Local::now();

    let mut scf_data = SCF::build(mol, mpi_operator);

    scf_without_build(&mut scf_data, mpi_operator);

    let dt2 = time::Local::now();
    if scf_data.mol.ctrl.print_level>0 {
        println!("the job costs {:16.2} seconds",(dt2.timestamp_millis()-dt0.timestamp_millis()) as f64 /1000.0)
    };

    //if scf_data.empirical_dispersion_energy != 0.0 {
    //    scf_data.scf_energy += scf_data.empirical_dispersion_energy
    //}

    Ok(scf_data)
}

fn ao2mo_rayon<'a, T, P>(eigenvector: &T, rimat_chunk: &P, row_dim: std::ops::Range<usize>, column_dim: std::ops::Range<usize>)
-> anyhow::Result<(RIFull<f64>, std::ops::Range<usize>, std::ops::Range<usize>)>
    where T: BasicMatrix<'a, f64>+std::marker::Sync,
          P: BasicMatrix<'a, f64>
{
    ao2mo_rayon_v02(eigenvector, rimat_chunk, row_dim, column_dim)
    //let mut 
    //ri_ao2mo_f
}

fn ao2mo_rayon_v01<'a, T, P>(eigenvector: &T, rimat_chunk: &P, row_dim: std::ops::Range<usize>, column_dim: std::ops::Range<usize>)
-> anyhow::Result<(RIFull<f64>, std::ops::Range<usize>, std::ops::Range<usize>)>
    where T: BasicMatrix<'a, f64>+std::marker::Sync,
          P: BasicMatrix<'a, f64>
{
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = omp_get_num_threads_wrapper();
    //utilities::omp_set_num_threads_wrapper(1);

    let num_basis = eigenvector.size()[0];
    let num_state = eigenvector.size()[1];
    let num_bpair = rimat_chunk.size()[0];
    let num_auxbs = rimat_chunk.size()[1];
    let num_loc_row = row_dim.len();
    let num_loc_col = column_dim.len();
    let mut rimo = RIFull::new([num_auxbs, num_loc_row, num_loc_col],0.0);
    let (sender, receiver) = channel();

    rimat_chunk.data_ref().unwrap().par_chunks_exact(num_bpair).enumerate().for_each_with(sender, |s, (i_auxbs, m)| {

        // To ensure the efficiency, we disable the openmp ability of openblase within the rayon parallel region
        omp_set_num_threads_wrapper(1);

        let mut loc_ri3mo = MatrixFull::new([row_dim.len(), column_dim.len()],0.0_f64);
        let mut reduced_ri = MatrixFull::new([num_basis, num_basis], 0.0_f64);
        reduced_ri.iter_matrixupper_mut().unwrap().zip(m.iter()).for_each(|(to, from)| {*to = *from});

        let mut tmp_mat = MatrixFull::new([num_basis,num_state], 0.0_f64);
        _dsymm(&reduced_ri, eigenvector, &mut tmp_mat, 'L', 'U', 1.0, 0.0);

        _dgemm(
            &tmp_mat, ((0..num_basis),row_dim.clone()), 'T',
            eigenvector, ((0..num_basis),column_dim.clone()), 'N',
            &mut loc_ri3mo, (0..row_dim.len(), 0..column_dim.len()),
            1.0, 0.0
        );
        s.send((loc_ri3mo, i_auxbs)).unwrap()
    });
    receiver.into_iter().for_each(|(loc_ri3mo, i_auxbs)| {
        rimo.copy_from_matr(0..num_loc_row, 0..num_loc_col, i_auxbs, 2, &loc_ri3mo, 0..num_loc_row, 0..num_loc_col)
    });

    omp_set_num_threads_wrapper(default_omp_num_threads);

    Ok((rimo, row_dim, column_dim))
}

fn ao2mo_rayon_v02<'a, T, P>(eigenvector: &T, rimat_chunk: &P, row_dim: std::ops::Range<usize>, column_dim: std::ops::Range<usize>)
-> anyhow::Result<(RIFull<f64>, std::ops::Range<usize>, std::ops::Range<usize>)>
    where T: BasicMatrix<'a, f64>+std::marker::Sync,
          P: BasicMatrix<'a, f64>
{
    // In this subroutine, we call the lapack dgemm in a rayon parallel environment.
    // In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
    let default_omp_num_threads = omp_get_num_threads_wrapper();
    //utilities::omp_set_num_threads_wrapper(1);

    let num_basis = eigenvector.size()[0];
    let num_state = eigenvector.size()[1];
    let num_bpair = rimat_chunk.size()[0];
    let num_auxbs = rimat_chunk.size()[1];
    let num_loc_row = row_dim.len();
    let num_loc_col = column_dim.len();
    let mut rimo = RIFull::new([num_auxbs, num_loc_row, num_loc_col],0.0);
    let (sender, receiver) = channel();

    rimat_chunk.data_ref().unwrap().par_chunks_exact(num_bpair).enumerate().for_each_with(sender, |s, (i_auxbs, m)| {

        // To ensure the efficiency, we disable the openmp ability of openblase within the rayon parallel region
        omp_set_num_threads_wrapper(1);

        let mut loc_ri3mo = MatrixFull::new([row_dim.len(), column_dim.len()],0.0_f64);
        let mut reduced_ri = MatrixFull::new([num_basis, num_basis], 0.0_f64);
        reduced_ri.iter_matrixupper_mut().unwrap().zip(m.iter()).for_each(|(to, from)| {*to = *from});

        let mut tmp_mat = MatrixFull::new([num_basis,num_state], 0.0_f64);
        _dsymm(&reduced_ri, eigenvector, &mut tmp_mat, 'L', 'U', 1.0, 0.0);

        _dgemm(
            &tmp_mat, ((0..num_basis),row_dim.clone()), 'T',
            eigenvector, ((0..num_basis),column_dim.clone()), 'N',
            &mut loc_ri3mo, (0..row_dim.len(), 0..column_dim.len()),
            1.0, 0.0
        );
        s.send((loc_ri3mo, i_auxbs)).unwrap()
    });
    receiver.into_iter().for_each(|(loc_ri3mo, i_auxbs)| {
        rimo.copy_from_matr(0..num_loc_row, 0..num_loc_col, i_auxbs, 2, &loc_ri3mo, 0..num_loc_row, 0..num_loc_col)
    });

    omp_set_num_threads_wrapper(default_omp_num_threads);

    Ok((rimo, row_dim, column_dim))
}

/// M1-optimized AO→MO transformation for a block of occupied (or virtual) orbitals.
///
/// Compared to `ao2mo_rayon_v02`, this routine contracts the *smaller* side
/// first in the dsymm step: instead of computing `tmp_mat = reduced_ri × eigenvector`
/// over the full eigenvector `[nao, nmo]` and then slicing, we pre-slice
/// `eigenvector[:, column_dim]` once (one-time copy of `[nao, |column_dim|]`)
/// and run dsymm on the smaller RHS. This saves a factor `nmo / |column_dim|`
/// in dsymm FLOPs.
///
/// Used by streaming PT2 (`*_pt2_rayon_streaming` in `ri_pt2/mod.rs`), where
/// `column_dim` is a small block of occupied orbitals (typical |column_dim| ≤ 64)
/// rather than the full `nocc`. The full `ri3mo` tensor is never materialized.
pub(crate) fn ao2mo_rayon_m1<'a, T, P>(
    eigenvector: &T,
    rimatr_chunk: &P,
    row_dim: std::ops::Range<usize>,
    column_dim: std::ops::Range<usize>,
) -> anyhow::Result<(RIFull<f64>, std::ops::Range<usize>, std::ops::Range<usize>)>
where T: BasicMatrix<'a, f64> + std::marker::Sync,
      P: BasicMatrix<'a, f64>
{
    let default_omp_num_threads = omp_get_num_threads_wrapper();

    let num_basis = eigenvector.size()[0];
    //let num_state = eigenvector.size()[1];
    let num_bpair = rimatr_chunk.size()[0];
    let num_auxbs = rimatr_chunk.size()[1];
    let num_loc_row = row_dim.len();
    let num_loc_col = column_dim.len();

    // M1 optimization: pre-slice eigenvector to the occ block.
    // Cost: one-time copy of [nao, |column_dim|] = num_basis * num_loc_col * 8 bytes.
    // For num_loc_col=64, num_basis=3035: ~1.5 MB. Negligible.
    let mut eigvec_occ_data = vec![0.0_f64; num_basis * num_loc_col];
    let eigvec_src = eigenvector.data_ref().expect("eigenvector must be contiguous");
    for j in 0..num_loc_col {
        let src_off = (column_dim.start + j) * num_basis;
        let dst_off = j * num_basis;
        eigvec_occ_data[dst_off..dst_off + num_basis]
            .copy_from_slice(&eigvec_src[src_off..src_off + num_basis]);
    }
    let eigvec_occ = MatrixFull::from_vec([num_basis, num_loc_col], eigvec_occ_data).unwrap();

    let mut rimo = RIFull::new([num_auxbs, num_loc_row, num_loc_col], 0.0);
    let (sender, receiver) = channel();

    rimatr_chunk.data_ref().unwrap().par_chunks_exact(num_bpair).enumerate().for_each_with(sender, |s, (i_auxbs, m)| {

        omp_set_num_threads_wrapper(1);

        let mut reduced_ri = MatrixFull::new([num_basis, num_basis], 0.0_f64);
        reduced_ri.iter_matrixupper_mut().unwrap().zip(m.iter()).for_each(|(to, from)| {*to = *from});

        // M1: dsymm with sliced eigenvector [nao, |column_dim|] (NOT full [nao, nmo])
        let mut tmp_mat = MatrixFull::new([num_basis, num_loc_col], 0.0_f64);
        _dsymm(&reduced_ri, &eigvec_occ, &mut tmp_mat, 'L', 'U', 1.0, 0.0);

        // dgemm: eigenvector[:, row_dim].T × tmp_mat -> [num_loc_row, num_loc_col]
        // loc_ri3mo[a, i] = sum_u eigenvector[u, a] * tmp_mat[u, i]
        let mut loc_ri3mo = MatrixFull::new([num_loc_row, num_loc_col], 0.0_f64);
        _dgemm(
            eigenvector, ((0..num_basis), row_dim.clone()), 'T',
            &tmp_mat, ((0..num_basis), (0..num_loc_col)), 'N',
            &mut loc_ri3mo, ((0..num_loc_row), (0..num_loc_col)),
            1.0, 0.0
        );
        s.send((loc_ri3mo, i_auxbs)).unwrap()
    });
    receiver.into_iter().for_each(|(loc_ri3mo, i_auxbs)| {
        rimo.copy_from_matr(0..num_loc_row, 0..num_loc_col, i_auxbs, 2, &loc_ri3mo, 0..num_loc_row, 0..num_loc_col)
    });

    omp_set_num_threads_wrapper(default_omp_num_threads);

    Ok((rimo, row_dim, column_dim))
}

pub fn diagonalize_hamiltonian_outside(scf_data: &SCF, mpi_operator: &Option<MPIOperator>) -> ([MatrixFull<f64>;2], [Vec<f64>;2], usize) {
    let mut eigenvectors = [MatrixFull::empty(),MatrixFull::empty()];
    let mut eigenvalues = [Vec::new(),Vec::new()];
    let mut num_state = 0;

    #[cfg(feature = "mpi")]
    if let Some(mpi_io) = mpi_operator {
        if mpi_io.rank == 0 {
            (eigenvectors, eigenvalues, num_state) = diagonalize_hamiltonian_outside_fast(scf_data);
        }
        for i_spin in 0..scf_data.mol.spin_channel {
            mpi_broadcast_vector(&mpi_io.world, &mut eigenvalues[i_spin], 0);
            mpi_broadcast_matrixfull(&mpi_io.world, &mut eigenvectors[i_spin], 0);
        }
        mpi_broadcast(&mpi_io.world, &mut num_state, 0);


    } else
    {
        (eigenvectors, eigenvalues, num_state) = diagonalize_hamiltonian_outside_fast(scf_data);
    }
    #[cfg(not(feature = "mpi"))]
    { (eigenvectors, eigenvalues, num_state) = diagonalize_hamiltonian_outside_fast(scf_data); }

    //println!("diagonalize_hamiltonian_outside: num_state {}", num_state);


    (eigenvectors, eigenvalues, scf_data.mol.num_state)
}

pub fn diagonalize_hamiltonian_outside_fast(scf_data: &SCF)  -> ([MatrixFull<f64>;2], [Vec<f64>;2], usize) {
    let spin_channel = scf_data.mol.spin_channel;
    let mut num_state = scf_data.mol.num_state;
    let dt1 = time::Local::now();

    let mut eigenvectors = [MatrixFull::empty(),MatrixFull::empty()];
    let mut eigenvalues = [Vec::new(),Vec::new()];

    match scf_data.scftype {
        SCFType::RHF | SCFType::UHF => {
            for i_spin in (0..spin_channel) {
                let (eigenvector_spin, eigenvalue_spin)=
                    _hamiltonian_fast_solver(&scf_data.hamiltonian[i_spin], &scf_data.ovlp, &mut num_state).unwrap();
                    //self.hamiltonian[i_spin].to_matrixupperslicemut()
                    //.lapack_dspgvx(self.ovlp.to_matrixupperslicemut(),num_state).unwrap();
                eigenvectors[i_spin] = eigenvector_spin;
                eigenvalues[i_spin] = eigenvalue_spin;
            }
        },
        SCFType::ROHF => {
            // diagonalize Roothaan Fock matrix
            let (eigenvector, eigenvalue)=
                _hamiltonian_fast_solver(scf_data.roothaan_hamiltonian.as_ref().unwrap(), &scf_data.ovlp, &mut num_state).unwrap();
            eigenvectors[0] = eigenvector;
            eigenvalues[0] = eigenvalue;
        }
    };
    (eigenvectors, eigenvalues, num_state)
}

pub fn diagonalize_hamiltonian_outside_rayon(scf_data: &SCF) -> ([MatrixFull<f64>;2], [Vec<f64>;2], usize) {
    let spin_channel = scf_data.mol.spin_channel;
    let num_state = scf_data.mol.num_state;
    //println!("diagonalize_hamiltonian_outside_rayon: num_state {}", num_state);
    let dt1 = time::Local::now();
    let mut num_state_out = num_state;

    let mut eigenvectors = [MatrixFull::empty(),MatrixFull::empty()];
    let mut eigenvalues = [Vec::new(),Vec::new()];

    match scf_data.scftype {
        SCFType::ROHF => {
            // diagonalize Roothaan Fock matrix
            let (eigenvector, eigenvalue, tmp_num_state_out)=
                _dspgvx(scf_data.roothaan_hamiltonian.as_ref().unwrap(), &scf_data.ovlp, num_state).unwrap();
            eigenvectors[0] = eigenvector;
            eigenvalues[0] = eigenvalue;
            if tmp_num_state_out < num_state_out {
                num_state_out = tmp_num_state_out;
            }
        },
        SCFType::RHF | SCFType::UHF => {
            for i_spin in (0..spin_channel) {
                let (eigenvector_spin, eigenvalue_spin, tmp_num_state_out)=
                    _dspgvx(&scf_data.hamiltonian[i_spin], &scf_data.ovlp, num_state).unwrap();
                    //self.hamiltonian[i_spin].to_matrixupperslicemut()
                    //.lapack_dspgvx(self.ovlp.to_matrixupperslicemut(),num_state).unwrap();
                eigenvectors[i_spin] = eigenvector_spin;
                eigenvalues[i_spin] = eigenvalue_spin;
                if tmp_num_state_out < num_state_out {
                    num_state_out = tmp_num_state_out;
                }
            }
        }
    }

    (eigenvectors,eigenvalues, num_state_out)
}

pub fn semi_diagonalize_hamiltonian_outside(scf_data: &SCF) -> (Option<[MatrixFull<f64>; 2]>, Option<[Vec<f64>; 2]>, Option<[MatrixFull<f64>; 2]>, usize) {
    // get the semi-canonical orbitals for RO-xDH calculations
    // See Knowles et al., Chem. Phys. Lett. 186(2), 130–136 (1991)
    let num_state = scf_data.mol.num_state;
    //let num_basis = scf_data.mol.num_basis; 
    let d_idx = 0..scf_data.lumo[1];
    let s_idx = scf_data.lumo[1]..scf_data.lumo[0];
    let v_idx = scf_data.lumo[0]..num_state;
    let ds_idx = 0..scf_data.lumo[0];
    let sv_idx = scf_data.lumo[1]..num_state;

    let fock = [scf_data.hamiltonian[0].to_matrixfull().unwrap(), scf_data.hamiltonian[1].to_matrixfull().unwrap()];

    let idx_list = [ds_idx, v_idx, d_idx, sv_idx];
    let fock_list = [&fock[0], &fock[0], &fock[1], &fock[1]];

    let mut semi_eigenvectors_list = [
        MatrixFull::new([num_state, scf_data.lumo[0]], 0.0),
        MatrixFull::new([num_state, num_state - scf_data.lumo[0]], 0.0),
        MatrixFull::new([num_state, scf_data.lumo[1]], 0.0),
        MatrixFull::new([num_state, num_state - scf_data.lumo[1]], 0.0),
    ];

    for (i, idx) in idx_list.iter().enumerate() {
        let c_slice = scf_data.eigenvectors[0].to_matrixfullslice_columns(idx.clone());
        let c = MatrixFull::from_vec(c_slice.size, c_slice.data.to_vec()).unwrap();

        let fock = apply_projection_operator(&c, fock_list[i], &c);
        let (eigenvectors, _eigenvalues, _) = _dsyevd(&fock, 'V');

        _dgemm_full(&c, 'N', &eigenvectors.unwrap(), 'N', &mut semi_eigenvectors_list[i], 1.0, 0.0);
    }

    let mut semi_eigenvectors: [MatrixFull<f64>; 2] = [
        semi_eigenvectors_list[0].clone(),
        semi_eigenvectors_list[2].clone(),
        ];

    semi_eigenvectors[0].append_column(&semi_eigenvectors_list[1]); // [ c_ds | c_v ]
    semi_eigenvectors[1].append_column(&semi_eigenvectors_list[3]); // [ c_d  | c_sv ]

    let mut semi_fock = [MatrixFull::new([num_state, num_state], 0.0), MatrixFull::new([num_state, num_state], 0.0)];
    let mut semi_eigenvalues = [Vec::new(), Vec::new()];

    for i_spin in 0..2 {
        let fock_tmp = apply_projection_operator(&semi_eigenvectors[i_spin], &fock[i_spin], &semi_eigenvectors[i_spin]);
    
        let diag_terms: Vec<f64> = fock_tmp.get_diagonal_terms().unwrap().into_iter().map(|&x| x).collect();
        
        semi_fock[i_spin] = fock_tmp;
        semi_eigenvalues[i_spin] = diag_terms;
    }
    
    (Some(semi_eigenvectors), Some(semi_eigenvalues), Some(semi_fock), num_state)
}

pub fn generate_occupation_outside(scf_data: &SCF) -> ([Vec<f64>;2], [usize;2], [usize;2]) {
    let mut occ = [vec![],vec![]];
    let mut homo = [0,0];
    let mut lumo = [0,0];
    match scf_data.mol.ctrl.occupation_type {
        OCCType::INTEGER => {
            (occ,homo,lumo) = generate_occupation_integer(&scf_data.mol,&scf_data.scftype);
        },
        OCCType::ATMSAD => {
            (occ,homo,lumo) = generate_occupation_sad(scf_data.mol.geom.elem.get(0).unwrap(),scf_data.mol.num_state, scf_data.mol.ecp_electrons);
        },
        OCCType::FRAC => {
            (occ,homo,lumo) = generate_occupation_frac_occ(&scf_data.mol,&scf_data.scftype, &scf_data.eigenvalues, scf_data.mol.ctrl.frac_tolerant);
        }
    }

    let mut force_occ = scf_data.mol.ctrl.force_state_occupation.clone();

    if force_occ.len()>0 {
        adapt_occupation_with_force_projection(
        &mut occ, &mut homo, &mut lumo,
        &mut force_occ, 
        &scf_data.scftype, 
        &scf_data.eigenvectors, 
        &scf_data.ovlp, 
        &scf_data.ref_eigenvectors);
        if scf_data.mol.ctrl.print_level>=2 {
            let mut window = [occ[0].len()-1,0];
            force_occ.iter().map(|x| x.get_check_window())
                .for_each(|[x,y]| {
                    if x< window[0] {window[0] = x};
                    if y>window[1] {window[1]=y}
                });
            println!("Occupation in Alpha Channel ({}-{}):", window[0], window[1]);
            let mut output = String::new();
            &occ[0][window[0]..window[1]].iter().enumerate().for_each(|(li,x)| {
                output = format!("{} ({:4}, {:6.3})", output, li+window[0], x);
                if (li+1)%5 == 0 {
                    output = format!("{}\n", output);
                }
            });
            println!("{}",output);
            if scf_data.mol.spin_channel == 2{
                println!("Occupation in Beta Channel: ({}-{}):", window[0], window[1]);
                let mut output = String::new();
                &occ[1][window[0]..window[1]].iter().enumerate().for_each(|(li,x)| {
                    output = format!("{} ({:4}, {:6.3})", output, li+window[0], x);
                    if (li+1)%5 == 0 {
                        output = format!("{}\n", output);
                    }
                });
                println!("{}",output);
            }
        }
    }



    (occ, homo, lumo)
}

pub fn generate_density_matrix_outside(scf_data: &SCF) -> Vec<MatrixFull<f64>>{

    // let num_basis = scf_data.mol.num_basis;
    // let num_state = scf_data.mol.num_state;
    let [num_basis, num_state] = scf_data.eigenvectors[0].size();
    let spin_channel = scf_data.mol.spin_channel;
    // let homo = &scf_data.homo;
    // println!("homo: {:?}", &homo);
    let mut dm = vec![
        MatrixFull::empty(),
        MatrixFull::empty()
        ];
    (0..spin_channel).into_iter().for_each(|i_spin| {
        let mut dm_s = &mut dm[i_spin];
        *dm_s = MatrixFull::new([num_basis,num_basis],0.0);
        let eigv_s = if let SCFType::ROHF = scf_data.scftype {
            &scf_data.eigenvectors[0]
        } else {
            &scf_data.eigenvectors[i_spin]
        };
        let occ_s =  &scf_data.occupation[i_spin];

        let nw = occupied_orbital_count(&scf_data.occupation[i_spin]);
        //println!("number of occupied orbitals from dm generation: {}", nw);

        let mut weight_eigv = MatrixFull::new([num_basis, num_state],0.0_f64);
        //let mut weight_eigv = eigv_s.clone();
        weight_eigv.par_iter_columns_mut(0..nw).unwrap().zip(eigv_s.par_iter_columns(0..nw).unwrap())
            .for_each(|value| {
                value.0.into_iter().zip(value.1.into_iter()).for_each(|value| {
                    *value.0 = *value.1
                })
            });

        // prepare weighted eigenvalue matrix wC
        weight_eigv.par_iter_columns_mut(0..nw).unwrap().zip(occ_s[0..nw].par_iter()).for_each(|(we,occ)| {
        //weight_eigv.data.chunks_exact_mut(weight_eigv.size[0]).zip(occ_s.iter()).for_each(|(we,occ)| {
            we.iter_mut().for_each(|c| *c = *c*occ);
        });

        // dm = wC*C^{T}
        _dgemm_full(&weight_eigv,'N',eigv_s, 'T',dm_s, 1.0, 0.0);
        //dm_s.lapack_dgemm(&mut weight_eigv, eigv_s, 'N', 'T', 1.0, 0.0);
        //dm_s.formated_output(5, "full");
    });
    //if let SCFType::ROHF = scf_data.scftype {dm[1]=dm[0].clone()};
    //scf_data.density_matrix = dm;

    dm

}


pub fn initialize_scf(scf_data: &mut SCF, mpi_operator: &Option<MPIOperator>) {

    // update the corresponding geometry information, which is crucial 
    // for preparing the following integrals accurately
    let position = &scf_data.mol.geom.position;
    scf_data.mol.cint_env = scf_data.mol.update_geom_poisition_in_cint_env(position);

    update_basis_from_hdf5chk(scf_data);

    // update the RI-JK algorithms if not clearly specified
    scf_data.update_jk_algorithms();

    let mut time_mark = utilities::TimeRecords::new();
    time_mark.new_item("Overall", "SCF Preparation");
    time_mark.count_start("Overall");

    time_mark.new_item("CInt", "Two, Three, and Four-center integrals");
    time_mark.count_start("CInt");
    scf_data.prepare_necessary_integrals(mpi_operator);
    time_mark.count("CInt");


    time_mark.new_item("DFT Grids", "Initialization of the tabulated Grids and AOs");
    time_mark.count_start("DFT Grids");
    scf_data.prepare_density_grids();
    time_mark.count("DFT Grids");

    time_mark.new_item("Solvent Calculation", "Initialization of the solvent calculation");
    time_mark.count_start("Solvent Calculation");
    scf_data.prepare_solvent_calculation();
    time_mark.count("Solvent Calculation");

    time_mark.new_item("ISDF", "ISDF initialization");
    time_mark.count_start("ISDF");
    scf_data.prepare_isdf(mpi_operator);
    time_mark.count("ISDF");

    time_mark.new_item("InitGuess", "Prepare initial guess");
    time_mark.count_start("InitGuess");
    initial_guess(scf_data, mpi_operator);
    if ! scf_data.mol.ctrl.atom_sad && scf_data.mol.ctrl.print_level>2 {
        println!("Initial density matrix by Atom SAD:");
        scf_data.density_matrix[0].formated_output(5, "full");
    }
    //println!("======== IGOR debug  for xc components after Initial Guess =======");
    //let dfa = crate::dft::DFA4REST::new_xc(scf_data.mol.spin_channel, scf_data.mol.ctrl.print_level);
    //let post_xc_energy = if let Some(grids) = &scf_data.grids {
    //    dfa.post_xc_exc(&scf_data.mol.ctrl.post_xc, grids, &scf_data.density_matrix, &scf_data.eigenvectors, &scf_data.occupation)
    //} else {
    //    vec![[0.0,0.0]]
    //};
    //post_xc_energy.iter().zip(scf_data.mol.ctrl.post_xc.iter()).for_each(|(energy, name)| {
    //    println!("{:<16}: {:16.8} Ha", name, energy[0]+energy[1]);
    //});
    //println!("======= IGOR debug ========");
    time_mark.count("InitGuess");

    time_mark.count("Overall");
    if scf_data.mol.ctrl.print_level>=2 {
        time_mark.report_all();
    }

}

pub fn scf_without_build(scf_data: &mut SCF, mpi_operator: &Option<MPIOperator>) {
    scf_data.generate_hf_hamiltonian(mpi_operator);
    scf_data.grad_dm = scf_data.get_grad_dm();

    let mut scf_records=ScfTraceRecord::initialize(&scf_data);

    if scf_data.mol.ctrl.print_level>0 {
        info!("The total energy: {:20.10} Ha by the initial guess",scf_data.scf_energy);
        if scf_data.mol.spin_channel==1 {
            info!("Initial grad_dm l2: {:10.5e} Ha", norm(&scf_data.grad_dm[0], "l2"));
        } else {
            info!("Initial grad_dm l2: ({:10.5e},{:10.5e}) Ha",
                norm(&scf_data.grad_dm[0], "l2"), norm(&scf_data.grad_dm[1], "l2"));
        }
    }
    //let mut scf_continue = true;
    if scf_data.mol.ctrl.noiter {
        println!("Warning: the SCF iteration is skipped!");
        return;
    }

    // now prepare the input density matrix for the first iteration and initialize the records
    scf_data.diagonalize_hamiltonian(mpi_operator);
    scf_data.generate_occupation();
    if let Some(st) = &scf_data.mol.ctrl.smear {
        apply_smearing(scf_data, *st, scf_data.mol.ctrl.smear_sigma.unwrap());
        scf_data.current_smear_sigma = scf_data.mol.ctrl.smear_sigma.unwrap();
    }

    // --- Apply guess_mix during the initial-guess stage ---
    // start_mix_cycle == 0 means: perform HOMO–LUMO mixing immediately
    // after the initial diagonalization (i.e. before the first SCF iteration).
    if scf_data.mol.ctrl.guess_mix && scf_data.mol.ctrl.start_mix_cycle == 0_usize {
        println!(">>> guess_mix activated: applying HOMO–LUMO mixing immediately after initial guess (start_mix_cycle = 0).");
        apply_guess_mix(scf_data);
    }    

    scf_data.generate_density_matrix();
    scf_records.update(&scf_data);

    //println!("======== IGOR debug for xc components =======");
    //let dfa = crate::dft::DFA4REST::new_xc(scf_data.mol.spin_channel, scf_data.mol.ctrl.print_level);
    //let post_xc_energy = if let Some(grids) = &scf_data.grids {
    //    dfa.post_xc_exc(&scf_data.mol.ctrl.post_xc, grids, &scf_data.density_matrix, &scf_data.eigenvectors, &scf_data.occupation)
    //} else {
    //    vec![[0.0,0.0]]
    //};
    //post_xc_energy.iter().zip(scf_data.mol.ctrl.post_xc.iter()).for_each(|(energy, name)| {
    //    println!("{:<16}: {:16.8} Ha", name, energy[0]+energy[1]);
    //});
    //println!("======= IGOR debug for xc components ========");

    let mut scf_converge = [false;2];
    let mut guess_mix_applied = (scf_data.mol.ctrl.start_mix_cycle == 0_usize); 
    while ! (scf_converge[0] || scf_converge[1]) {
        let dt1 = time::Local::now();

        scf_records.prepare_next_input(scf_data, mpi_operator);
        // scf_data.grad_dm = scf_data.get_grad_dm();

        let dt1_1 = time::Local::now();

        scf_data.diagonalize_hamiltonian(mpi_operator);
        let dt1_2 = time::Local::now();
        scf_data.generate_occupation();
        if let Some(st) = &scf_data.mol.ctrl.smear {
            let mut sigma = scf_data.mol.ctrl.smear_sigma.unwrap();
            if scf_data.mol.ctrl.smear_anneal {
                let anneal_start = scf_data.mol.ctrl.start_diis_cycle;
                let anneal_length = (scf_data.mol.ctrl.max_scf_cycle as f64 * 0.5) as usize;
                let sigma_min = scf_data.mol.ctrl.smear_sigma_min
                    .unwrap_or_else(|| f64::max(sigma * 0.01, 0.001));
                sigma = annealed_sigma(sigma, sigma_min, scf_records.num_iter, anneal_start, anneal_length);
            }
            apply_smearing(scf_data, *st, sigma);
            scf_data.current_smear_sigma = sigma;
        }

        // --- Apply guess_mix during SCF iterations ---
        // When start_mix_cycle > 0, perform HOMO–LUMO mixing exactly at the
        // specified SCF cycle number (num_iter == start_mix_cycle).
        if scf_data.mol.ctrl.guess_mix && !guess_mix_applied && (scf_records.num_iter as usize) == scf_data.mol.ctrl.start_mix_cycle {
            println!(">>> guess_mix activated at SCF iteration {}.", scf_records.num_iter);
            apply_guess_mix(scf_data);
            scf_records.refresh();
            guess_mix_applied = true;
        }

        scf_data.generate_density_matrix();

        let dt_solv0 = time::Local::now();
        if scf_data.mol.ctrl.solvent_enabled {
            if let Some(solvent_static) = scf_data.solvent_static_obj.as_ref() {
                let s_static = PcmScf::get_pcm_refresh(
                    &solvent_static.surface,
                    &scf_data.mol,
                    &scf_data.density_matrix,
                    &solvent_static.pstatic.K,
                    &solvent_static.pstatic.K_ipiv,
                    &solvent_static.pstatic.R,
                    &solvent_static.pstatic.v_grids_n,
                    &scf_data.mol.spin_channel,
                    &scf_data.mol.ctrl.max_memory,
                    &scf_data.mol.ctrl.solv_chunk,
                    scf_data.mol.ctrl.solvent_ri
                );
                // SMD: CDS energy from PcmStatic (computed once in solvent_prepare)
                let e_cds = solvent_static.pstatic.e_cds.unwrap_or(0.0);
                scf_data.energies.insert(String::from("solvent_energy"), vec![s_static.eng + e_cds]);
                scf_data.solvent_scf = Some(s_static);
            }
        }
        let dt_solv1 = time::Local::now();

        if scf_data.mol.ctrl.print_level>1 {
            scf_data.print_homo_lumo_gap()
        };
        let dt1_3 = time::Local::now();
        scf_converge = scf_data.check_scf_convergence(&scf_records);
        let dt1_4 = time::Local::now();
        
        // -------------------------
        // If SCF converged earlier than requested mix point,
        // but user requested guess_mix and it hasn't been applied yet,
        // apply mixing now and continue SCF (do NOT exit loop).
        // -------------------------
        if (scf_converge[0] || scf_converge[1]) && scf_data.mol.ctrl.guess_mix && !guess_mix_applied {
            println!(">>> guess_mix requested at start_mix_cycle = {}, but SCF converged after {} iterations. \
            Applying HOMO-LUMO mixing now and continuing SCF.", scf_data.mol.ctrl.start_mix_cycle, scf_records.num_iter - 1);

            // apply mixing and mark as applied
            apply_guess_mix(scf_data);
            scf_records.refresh();
            guess_mix_applied = true;

            // rebuild dependent quantities so subsequent SCF iterations are consistent
            scf_data.generate_density_matrix();
            scf_data.generate_hf_hamiltonian(mpi_operator);
            scf_data.grad_dm = scf_data.get_grad_dm();
            scf_data.diagonalize_hamiltonian(mpi_operator);
            scf_data.generate_occupation();

            // IMPORTANT: clear convergence so the while-loop continues
            scf_converge = [false, false];
        }

        scf_records.update(&scf_data);
        let dt1_5 = time::Local::now();


        let dt2 = time::Local::now();
        let timecost = (dt2.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
        if scf_data.mol.ctrl.print_level>0 {
            let have_smear = scf_data.mol.ctrl.smear.is_some();
            let e_tot = scf_records.scf_energy;
            if scf_data.mol.spin_channel == 2 {
                let [square_spin, spin_z] = evaluate_spin_angular_momentum(&scf_data.density_matrix, &scf_data.ovlp, scf_data.mol.spin_channel, &scf_data.mol.num_elec);
                if have_smear {
                    let sigma = scf_data.current_smear_sigma;
                    let s = scf_data.smearing_entropy;
                    println!("E(T) {:18.10}  E_free {:18.10}  E0 {:18.10}  S^2 {:5.3}  2S+1 {:5.3}  iter {:4}  {:8.2}s",
                         e_tot, e_tot - sigma * s, e_tot - 0.5 * sigma * s,
                         square_spin, spin_z,
                         scf_records.num_iter-1, timecost)
                } else {
                    println!("Energy: {:18.10} Ha with <S^2> = {:6.3} and <2S+1> = {:6.3} after {:4} iterations (in {:10.2} seconds).",
                         e_tot, square_spin, spin_z,
                         scf_records.num_iter-1, timecost)
                }
            } else {
                if have_smear {
                    let sigma = scf_data.current_smear_sigma;
                    let s = scf_data.smearing_entropy;
                    println!("E(T) {:18.10}  E_free {:18.10}  E0 {:18.10}  iter {:4}  {:8.2}s",
                         e_tot, e_tot - sigma * s, e_tot - 0.5 * sigma * s,
                         scf_records.num_iter-1, timecost)
                } else {
                    println!("Energy: {:18.10} Ha after {:4} iterations (in {:10.2} seconds).",
                         e_tot, scf_records.num_iter-1, timecost)
                }
            }
        };
        if scf_data.mol.ctrl.print_level>1 {
            println!("Detailed timing info in this SCF step:");
            let timecost = (dt1_1.timestamp_millis()-dt1.timestamp_millis()) as f64 /1000.0;
            println!("prepare_next_input:      {:10.2}s", timecost);
            let timecost = (dt1_2.timestamp_millis()-dt1_1.timestamp_millis()) as f64 /1000.0;
            println!("diagonalize_hamiltonian: {:10.2}s", timecost);
            let timecost = (dt1_3.timestamp_millis()-dt1_2.timestamp_millis()) as f64 /1000.0;
            println!("generate_density_matrix: {:10.2}s", timecost);
            let timecost = (dt1_4.timestamp_millis()-dt1_3.timestamp_millis()) as f64 /1000.0;
            println!("check_scf_convergence:   {:10.2}s", timecost);
            let timecost = (dt1_5.timestamp_millis()-dt1_4.timestamp_millis()) as f64 /1000.0;
            println!("scf_records.update:      {:10.2}s", timecost);
            let timecost = (dt_solv1.timestamp_millis()-dt_solv0.timestamp_millis()) as f64 /1000.0;
            println!("solvent_model.refresh:   {:10.2}s", timecost);
        }
    }
    if scf_converge[0] {
        info!("SCF is converged after {:4} iterations.", scf_records.num_iter-1);
        // Level shift is disabled before the final diagonalization to ensure accurate eigenvalues.
        // Formatted printing of eigenvalues and eigenvectors is now performed after re-diagonalizing the HF Hamiltonian.
    } else {
        //if scf_data.mol.ctrl.restart {save_chkfile(&scf_data)};
        println!("SCF does not converge within {:03} iterations",scf_records.num_iter);
    }
    match scf_data.mol.ctrl.occupation_type {
        OCCType::FRAC => {
            let (occupation, homo, lumo) = check_norm::generate_occupation_integer(&scf_data.mol, &scf_data.scftype);
            scf_data.occupation = occupation;
            scf_data.homo = homo;
            scf_data.lumo = lumo;
            scf_data.generate_density_matrix();
            scf_data.generate_hf_hamiltonian(mpi_operator);
            scf_data.grad_dm = scf_data.get_grad_dm();
            scf_data.diagonalize_hamiltonian(mpi_operator);
            scf_data.generate_occupation();

        }
        _ => {
            scf_data.generate_hf_hamiltonian(mpi_operator); 
            scf_data.grad_dm = scf_data.get_grad_dm();
            info!("Energy: {:20.10} Ha", scf_data.scf_energy);
            info!("grad_dm l2: {:10.5e} Ha", norm(&scf_data.grad_dm[0], "l2"));
            scf_data.diagonalize_hamiltonian(mpi_operator);
            scf_data.generate_occupation();
            // scf_data.generate_density_matrix();
            // scf_data.generate_hf_hamiltonian(mpi_operator);
            // scf_data.grad_dm = scf_data.get_grad_dm();
            // info!("Energy: {:20.10} Ha", scf_data.scf_energy);
            // info!("grad_dm l2: {:10.5e} Ha", norm(&scf_data.grad_dm[0], "l2"));
        }
    }

    //solvent debug
    if scf_data.mol.ctrl.solvent_enabled{

        //println!("solvent_static_obj: {:?}", scf_data.solvent_static_obj.is_some());
        //println!("solvent_scf: {:?}", scf_data.solvent_scf.is_some());

        if scf_data.solvent_static_obj.is_none() {
            println!("ERROR: solvent_static_obj is None");
        }
        if scf_data.solvent_scf.is_none() {
            println!("ERROR: solvent_scf is None");
        }

        if scf_data.mol.ctrl.print_level > 2{
            debug_print_pcm(&scf_data.solvent_static_obj.as_ref().unwrap().pstatic, &scf_data.solvent_scf.as_ref().unwrap());
        }
    }
    
    if scf_data.mol.ctrl.print_level>1 {
        scf_data.print_homo_lumo_gap();
        scf_data.formated_eigenvalues((scf_data.homo.iter().max().unwrap()+4).min(scf_data.mol.num_state));
    }
    if scf_data.mol.ctrl.print_level>3 {
        scf_data.formated_eigenvectors();
    }

}


pub fn vj_on_the_fly_par(mol: &Molecule, dm: &Vec<MatrixFull<f64>>) -> Vec<MatrixUpper<f64>>{

    let num_shell = mol.cint_bas.len();
    let num_basis = mol.num_basis;
    let spin_channel = mol.spin_channel;

    // establish the map between matrixupper and matrixfull
    let matupp_length = (mol.num_basis+1)*mol.num_basis/2;
    let matrixupper_index = map_upper_to_full(matupp_length).unwrap();

    let mut dm_upper = Vec::new();
    let mut dm_diagonal = Vec::new();
    for i_spin in 0..spin_channel {
        dm_upper.push(dm[i_spin].to_matrixupper());
        dm_diagonal.push(dm[i_spin].iter_diagonal().unwrap().map(|x| *x).collect::<Vec<f64>>())
    }
    //let dm_upper = dm.iter().map(|dm_s| dm_s.to_matrixupper()).collect::<Vec<MatrixUpper<f64>>>();
    //let dm_diagonal  = dm.iter().map(|dm_s| 
    //    dm_s.iter_diagonal().unwrap().map(|x| *x).collect::<Vec<f64>>()
    //).collect::<Vec<Vec<f64>>>();

    //let mut vj: Vec<MatrixUpper<f64>> = vec![MatrixUpper::new(1, 0.0), MatrixUpper::new(1, 0.0)];
    let mut vj: Vec<MatrixUpper<f64>> = vec![MatrixUpper::empty(), MatrixUpper::empty()];

    let mut vj_full: Vec<MatrixFull<f64>> = vec![
        MatrixFull::new([num_basis, num_basis], 0.0),
        if spin_channel==2 {
            MatrixFull::new([num_basis, num_basis], 0.0)
        } else {
            MatrixFull::empty()
        }
    ];


    // initialize the parallel tasks
    let mut index = Vec::new();
    for l in 0..num_shell {
        for k in 0..l+1 {
            index.push([mol.cint_fdqc[k][1]*mol.cint_fdqc[l][1],k,l])
        }
    };
    index = index.iter().sorted_by(|a,b| Ord::cmp(&a[0], &b[0]))
        .map(|x| *x).collect::<Vec<[usize;3]>>();

    let half_length = index.len()/2;
    let is_odd = index.len()%2;

    // re-arrange the tasks that mixed universally according the work loading.
    let mut index_new = Vec::new();
    if is_odd==1 {index_new.push(index[half_length])};
    index[0..half_length].iter().zip(index[half_length+is_odd..index.len()].iter().rev()).for_each(|(task1, task2)| {
        index_new.push(*task1);
        index_new.push(*task2);
    });

    let par_tasks = utilities::balancing(index_new.len(), rayon::current_num_threads());
    let (sender, receiver) = channel();

    par_tasks.par_iter().for_each_with(sender, |s,task_range| {
        //rayon::current_thread_index();
        let mut cint_data = mol.initialize_cint(false);
        cint_data.cint2e_optimizer_rust();

        let mut out_submatrix = Vec::new();

        index_new[task_range.clone()].iter().for_each(|[weight,k,l]| {
            let bas_start_k = mol.cint_fdqc[*k][0];
            let bas_len_k = mol.cint_fdqc[*k][1];
            let bas_start_l = mol.cint_fdqc[*l][0];
            let bas_len_l = mol.cint_fdqc[*l][1];


            let klij = mol.int_ijkl_given_kl_v03(*k, *l, &matrixupper_index, &mut cint_data);
            let mut out = vec![
                MatrixFull::new([bas_len_k, bas_len_l],0.0),
                if spin_channel==2 {
                    MatrixFull::new([bas_len_k, bas_len_l],0.0)
                } else {
                    MatrixFull::empty()
                }
            ];
            for i_spin in 0..spin_channel {
                let dm_s_upper = &dm_upper[i_spin];
                let dm_s_diagonal = &dm_diagonal[i_spin];
                let mut out_s = &mut out[i_spin];

                out_s.iter_columns_full_mut().enumerate().for_each(|(loc_l,x)|{
                    x.iter_mut().enumerate().for_each(|(loc_k,elem)|{
                        let ao_k = loc_k + bas_start_k;
                        let ao_l = loc_l + bas_start_l;
                        let eri_cd = klij.get(&[loc_k, loc_l]).unwrap();
                        let mut sum = dm_s_upper.data.iter().zip(eri_cd.iter())
                            .fold(0.0,|sum, (p,eri)| {
                            sum + *p * *eri
                        });
                        let mut diagonal = dm_s_diagonal.iter().zip(eri_cd.iter_diagonal()).fold(0.0,|diagonal, (p,eri)| {
                            diagonal + *p * *eri
                        });
                        sum = sum*2.0 - diagonal;

                        *elem = sum;
                    });
                });
            }

            out_submatrix.push((out,*k,*l));

        });

        cint_data.final_c2r();
        s.send(out_submatrix).unwrap();
    });

    receiver.into_iter().for_each(|out_submatrix| {
        out_submatrix.into_iter().for_each(|(out,k,l)| {
        let bas_start_k = mol.cint_fdqc[k][0];
        let bas_len_k = mol.cint_fdqc[k][1];
        let bas_start_l = mol.cint_fdqc[l][0];
        let bas_len_l = mol.cint_fdqc[l][1];
        for i_spin in 0..spin_channel {
            let mut vj_s = &mut vj_full[i_spin];
            let out_s = &out[i_spin];
            vj_s.copy_from_matr(bas_start_k..bas_start_k+bas_len_k, bas_start_l..bas_start_l+bas_len_l, 
                out_s, 0..bas_len_k,0..bas_len_l);
            //vj_i.iter_submatrix_mut(bas_start_k..bas_start_k+bas_len_k, bas_start_l..bas_start_l+bas_len_l).zip(out.iter())
            //    .for_each(|(to, from)| {*to = *from});
        }
        });
    });

    for i_spin in 0..spin_channel {
        vj[i_spin] = vj_full[i_spin].to_matrixupper();
    }

    vj
}

pub fn vj_on_the_fly_par_batch_by_batch(mol: &Molecule, dm: &Vec<MatrixFull<f64>>) -> Vec<MatrixUpper<f64>>{

    let matrixupper_index = map_upper_to_full((mol.num_basis+1)*mol.num_basis/2).unwrap();
    let batch_length = mol.ctrl.batch_size;
    let mut batches = Vec::new();
    let mut total_length = matrixupper_index.size as i32;
    let mut start = 0;
    while total_length >=0 {
        batches.push([start, batch_length]);
        start += batch_length;
        total_length -= (batch_length as i32);
    }
    let ind = batches.len()-1;
    if total_length < 0 {
        let [start, mut batch_length] = batches.pop().unwrap();
        batch_length -= (total_length.abs() as usize);
        if batch_length > 0 {
            batches.push([start,batch_length])
        }
    }

    //println!("{:?}", &batches);

    let num_shell = mol.cint_bas.len();
    let num_basis = mol.num_basis;
    let spin_channel = mol.spin_channel;
    //let dm = &self.density_matrix;
    let mut vj: Vec<MatrixUpper<f64>> = vec![];
    //let mol = &self.mol;
    //utilities::omp_set_num_threads_wrapper(1);
    for i_spin in 0..spin_channel{
        let mut vj_i = MatrixFull::new([num_basis, num_basis], 0.0);
        let dm_s_upper = dm[i_spin].to_matrixupper();
        let dm_s_diagnoal = dm[i_spin].iter_diagonal().unwrap().map(|x| *x).collect::<Vec<f64>>();
        let par_tasks = utilities::balancing(num_shell*num_shell, rayon::current_num_threads());
        let (sender, receiver) = channel();
        let mut index = Vec::new();
        for l in 0..num_shell {
            for k in 0..l+1 {
                index.push((k,l))
            }
        };

        //utilities::balancing_type_02(num_tasks, num_threads, per_communication);
        index.par_iter().for_each_with(sender,|s,(k,l)|{
            let bas_start_k = mol.cint_fdqc[*k][0];
            let bas_len_k = mol.cint_fdqc[*k][1];
            let bas_start_l = mol.cint_fdqc[*l][0];
            let bas_len_l = mol.cint_fdqc[*l][1];

            let mut out = MatrixFull::new([bas_len_k, bas_len_l],0.0);
            let mut output = String::new();
            for [start, batch_length] in &batches {
                //if *k==0 && *l == 0 {
                //    println!("batch: ({},{})",start, batch_length);
                //}
                let batch = (*start..*start+batch_length);
                let [str_row,str_col] = matrixupper_index[batch.start];
                let [end_row,end_col] = matrixupper_index[batch.end-1];
                let diag_list = if end_row < end_col {str_col..end_col} else {str_col..(end_col+1)};
                let klij = mol.int_ijkl_given_kl_batch(*k, *l, batch.clone(), &matrixupper_index);
                //if *k==0 && *l == 0 && (batch.start == 60|| batch.start==75) {
                //    println!("{:?}", &klij[(0,0)].data);
                //}
                let mut sum = 0.0;
                out.iter_columns_full_mut().enumerate().for_each(|(loc_l,x)|{
                    x.iter_mut().enumerate().for_each(|(loc_k,elem)|{
                        let ao_k = loc_k + bas_start_k;
                        let ao_l = loc_l + bas_start_l;
                        let eri_cd = &klij[(loc_k, loc_l)];
                        let mut sum = dm_s_upper.data[batch.clone()].iter().zip(eri_cd.iter())
                            .fold(0.0,|sum, (p,eri)| {
                        //let mut sum = dm_s_upper.data[batch.clone()].iter().enumerate().zip(eri_cd.iter())
                        //    .fold(0.0,|sum, ((i,p),eri)| {
                            //if ao_k == 0 && ao_l ==0 {
                            //    output = format!("{}, ({},{:8.4},{:8.4})", output, i+batch.start, p, eri);
                            //}
                            sum + *p * *eri
                        });

                        let mut diagonal = 0.0;
                        diag_list.clone().for_each(|i| {

                            let p = dm_s_diagnoal[i];

                            let i_in_mu = (i+1)*i/2 + i;
                            let i_in_smu = i_in_mu - eri_cd.global_range.start;

                            let eri = eri_cd.data[i_in_smu];

                            diagonal += p*eri;

                            //if ao_k == 0 && ao_l ==0 {
                            //    output = format!("{}, ({},{:8.4},{:8.4})", output, i, p, eri);
                            //}


                        });

                        *elem += sum*2.0 - diagonal;

                    });
                });

            }
            //if *k==0 && *l == 0 {
            //    println!("{}",output);
            //}
            //if *k==0 && *l==0 {
            //    out.formated_output(5, "full");
            //}

            s.send((out,*k,*l)).unwrap();
        });
        receiver.into_iter().for_each(|(out,k,l)| {
            let bas_start_k = mol.cint_fdqc[k][0];
            let bas_len_k = mol.cint_fdqc[k][1];
            let bas_start_l = mol.cint_fdqc[l][0];
            let bas_len_l = mol.cint_fdqc[l][1];
            vj_i.copy_from_matr(bas_start_k..bas_start_k+bas_len_k, bas_start_l..bas_start_l+bas_len_l, 
                &out, 0..bas_len_k,0..bas_len_l);
            //vj_i.iter_submatrix_mut(bas_start_k..bas_start_k+bas_len_k, bas_start_l..bas_start_l+bas_len_l).zip(out.iter())
            //    .for_each(|(to, from)| {*to = *from});
        });
        

        vj.push(vj_i.to_matrixupper());
    }
    if spin_channel == 1{
        vj.push(MatrixUpper::new(1, 0.0));       
    }
    vj
}

// evaluate the expectation value of squre spin angular momentum operator (S^2) as well as the expectation value of spin angular momentum operator along z axis
// the expectation value of S^2 is given by 3/4*Tr[(scf_data.density_matrix[0]] + scf_data.density_matrix[1])\dot scf_data.ovlp.to_matrixfull().unwrap()]
// the expectation value of S_z^2  is given by 1/2*Tr[(scf_data.density_matrix[0] -scf_data.[1])\dot scf_data.ovlp.to_matrixfull().unwrap()]
// use MatrixFull::iter_diagonal() to get the diagonal elements of a matrix for the `Tr` evaluation
pub fn evaluate_spin_angular_momentum_wrong(dm: &Vec<MatrixFull<f64>>, ovlp: &MatrixUpper<f64>,spin_channel: usize) -> [f64;2] {
    let mut tr_spin_angular_momentum = [0.0;2];
    let ovlp_full = ovlp.to_matrixfull().unwrap();

    if spin_channel==1 {
        let dm_s = dm.get(0).unwrap();
        let mut tmp_matrix = dm_s.clone();
        _dgemm_full(dm_s, 'N', &ovlp_full, 'N', &mut tmp_matrix, 1.0, 0.0);

        [0.75*tmp_matrix.iter_diagonal().unwrap().fold(0.0, |acc, val| acc + val),0.0]
    } else {
        let mut tr_spin_angular_momentum = [0.0;2];
        for i_spin in 0..spin_channel {
            let dm_s = dm.get(i_spin).unwrap();
            let mut tmp_matrix = dm_s.clone();
            _dgemm_full(dm_s, 'N', &ovlp_full, 'N', &mut tmp_matrix, 1.0, 0.0);
            tr_spin_angular_momentum[i_spin] = tmp_matrix.iter_diagonal().unwrap().fold(0.0, |acc, val| acc + val);
        }

        [0.75*(tr_spin_angular_momentum[0]+tr_spin_angular_momentum[1]),
         0.50*(tr_spin_angular_momentum[0]-tr_spin_angular_momentum[1])]
    }
}




pub fn evaluate_spin_angular_momentum(dm: &Vec<MatrixFull<f64>>, ovlp: &MatrixUpper<f64>, spin_channel: usize, num_elec: &[f64;3]) -> [f64;2] {

    let n_a = num_elec[1];
    let n_b = num_elec[2];
    let ms = (n_a-n_b)*0.5;
    let mut s2 = ms*(ms+1.0);
    let mut mlpy = 2.0*ms + 1.0;

    if spin_channel == 2 {
        let ovlp_full = ovlp.to_matrixfull().unwrap();

        s2 += n_b;

        let mut tmp_matr_a = dm[0].clone();
        let mut tmp_matr_b = dm[0].clone();

        _dgemm_full(&ovlp_full, 'N', &dm[0], 'N', &mut tmp_matr_a, 1.0, 0.0);
        _dgemm_full(&dm[1], 'N', &tmp_matr_a, 'N', &mut tmp_matr_b, 1.0, 0.0);
        _dgemm_full(&ovlp_full, 'N', &tmp_matr_b, 'N', &mut tmp_matr_a, 1.0, 0.0);

        s2 -= tmp_matr_a.iter_diagonal().unwrap().fold(0.0, |acc, x| acc+x);

        mlpy = (1.0 + 4.0* s2).powf(0.5);

    }

    [s2, mlpy]

}

/// Compute the DIIS error matrix for one spin channel:  e = F·D·S − S·D·F
pub fn get_grad_dm(
    fock: &MatrixFull<f64>,
    ovlp: &MatrixFull<f64>,
    density: &MatrixFull<f64>,
) -> MatrixFull<f64> {
    // FDS = F * D * S
    let mut error = fock.clone().ddot(&mut density.clone()).unwrap();
    error = error.ddot(&mut ovlp.clone()).unwrap();

    // SDF = S * D * F
    let mut sdf = ovlp.clone().ddot(&mut density.clone()).unwrap();
    sdf = sdf.ddot(&mut fock.clone()).unwrap();

    error.self_sub(&sdf);
    error
}

impl SCF {
    pub fn get_grad_dm(&self) -> [MatrixFull<f64>; 2] {
        let ovlp_full = self.ovlp.to_matrixfull().unwrap();
        let mut e = [MatrixFull::empty(), MatrixFull::empty()];
        for i_spin in 0..self.mol.spin_channel {
            let f = self.hamiltonian[i_spin].to_matrixfull().unwrap();
            e[i_spin] = get_grad_dm(&f, &ovlp_full, &self.density_matrix[i_spin]);
        }
        e
    }
}

impl SCF {
    /// Free the large tensors in SCF struct to save memory after SCF calculation is done.
    /// 
    /// This function is initially written for geometric optimization, in order to give
    /// approximately correct memory estimation for next step.
    pub fn free_large_tensors(&mut self) {
        self.ijkl = None;
        self.ri3fn = None;
        self.ri3fn_sr = None;
        self.ri3fn_isdf = None;
        self.tab_ao = None;
        self.m = None;
        self.rimatr = None;
        self.rimatr_sr = None;
        self.ri3mo = None;
        self.ri3mo_full = None;
        self.ri3fn_bse = None;
        self.rimatr_bse = None;
        self.grids = None;
        self.solvent_static_obj = None;
        self.solvent_scf = None;
    }
}
