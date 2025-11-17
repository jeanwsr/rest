
use pyo3::pyclass;
use serde::{Deserialize,Serialize};
use tensors::MatrixFull;
//use std::{fs, str::pattern::StrSearcher};
use std::{fs, sync::Arc};
use crate::ctrl_io::geometric_pyo3_io::parse_geometric_keywords;
use crate::ctrl_io::quasiparticle_methods::parse_quasiparticle_keywords;
use crate::{check_norm::force_state_occupation::ForceStateOccupation};
use crate::dft::{DFAFamily, DFTType, DFA4REST};
use crate::geom_io::{GeomCell, GeomUnit, MOrC, parse_geom_keywords};
use crate::utilities;
use rayon::ThreadPoolBuilder;
use crate::check_norm::OCCType;
use tensors::matrix_blas_lapack::{omp_set_num_threads_wrapper,omp_get_num_threads_wrapper};

use serde_json;
use toml;

pub mod flags;
pub use flags::*;

mod pyrest_ctrl_io;
mod geometric_pyo3_io;
pub mod quasiparticle_methods;
use geometric_pyo3_io::GeomeTRIC;
mod path_util;
use quasiparticle_methods::QuasiParticle;

pub fn parse_ctl(filename: String) -> anyhow::Result<(InputKeywords,GeomCell)> {
    let tmp_cont = fs::read_to_string(&filename[..])?;
    let tmp_keys = if let Ok(tmp_json) = serde_json::from_str::<serde_json::Value>(&tmp_cont[..]) {
        // input file in the json format
        tmp_json
    } else {
        // input file in the toml format
        toml::from_str::<serde_json::Value>(&tmp_cont[..])?
    };
    parse_ctl_from_json(&tmp_keys)
}

pub fn parse_ctl_from_json(tmp_keys: &serde_json::Value) -> anyhow::Result<(InputKeywords,GeomCell)> {
    let mut tmp_input = parse_ctrl_keywords(tmp_keys)?;
    let mut tmp_geomcell = parse_geom_keywords(tmp_keys)?;
    let mut tmp_geomtric = parse_geometric_keywords(tmp_keys)?;
    let mut tmp_quasiparticle=parse_quasiparticle_keywords(tmp_keys)?;
    if let Some(tmp_geomtric) = &mut tmp_geomtric {
        tmp_input.geometric_pyo3 = Some(std::mem::take(tmp_geomtric));
    }
    if let Some(tmp_quasiparticle) = &mut tmp_quasiparticle {
        tmp_input.quasiparticle_methods = Some(std::mem::take(tmp_quasiparticle));
    }
    Ok((tmp_input,tmp_geomcell))
}


#[derive(Clone,Copy,Debug, Deserialize, Serialize)]
pub enum JobType {
    SinglePoint,
    Force,
    NumDipole,
    GeomOpt,
}

/// **InputKeywords** for a specific calculation
///  ### System dependent keywords
///  - `print_level`:  default (1). `0` dose not print anything. larger number with more output information  
/// 
///  ### Basis set keywords
///  - `basis_path`:   `String`. The path where you can find the basis-set file in json format. If the basis-set file is missing, REST will try to download it from BasisSetExchange
///  - `basis_type`:   `String`. It can be `spheric` or `cartesian`
///  - `auxbas_path`:  `String`. The path where you can find the auxiliary basis-set file in json format. If the basis-set file is missing, REST will try to download it from BasisSetExchange
///  - `auxbas_type`:  `String`. It can be `spheric` or `cartesian`
///  - `even_tempered-basis`: `Bool`. True: turn on ETB to generate the auxiliary basis set
///  - `etb_start_atom_number`: `Usize`. Use ETB, for the element with atomic index larger than this value  
///  - `etb_beta`: `f64`. Relevant to the ETB basis set size. Smaller value indicates larger ETB basis set. NOTE: etb_beta should be larger than 1.0
#[derive(Debug,Clone,Serialize, Deserialize)]
#[pyclass]
pub struct InputKeywords {
    pub job_type: JobType,
    #[pyo3(get, set)]
    pub print_level: usize,
    // Keywords for the (aux) basis sets
    #[pyo3(get, set)]
    pub basis_path: String,
    #[pyo3(get, set)]
    pub basis_type: String,
    #[pyo3(get, set)]
    pub auxbas_path: String,
    #[pyo3(get, set)]
    pub auxbas_type: String,
    #[pyo3(get, set)]
    pub use_auxbas: bool,
    #[pyo3(get, set)]
    pub even_tempered_basis: bool,
    #[pyo3(get, set)]
    pub etb_start_atom_number: usize,
    #[pyo3(get, set)]
    pub etb_beta: f64,
    // Keywords for RI_K
    #[pyo3(get, set)]
    pub ri_k_only: bool,
    // Keywords for gradient, evaluate small contribution from auxiliary basis perturbation
    #[pyo3(get, set)]
    pub auxbasis_response: bool,
    // Keywords for gradient, whether evaluate from numerical or analytical derivative
    #[pyo3(get, set)]
    pub numerical_force: bool,
    // Keywords for IDSF
    #[pyo3(get, set)]
    pub use_isdf: bool,
    #[pyo3(get, set)]
    pub isdf_k_only: bool,
    #[pyo3(get, set)]
    pub isdf_k_mu: usize,
    // Keywords for systems
    #[pyo3(get, set)]
    pub isdf_new: bool,
    #[pyo3(get, set)]
    pub eri_type: String,
    #[pyo3(get, set)]
    pub use_ri_symm: bool,
    #[pyo3(get, set)]
    pub xc: String,
    pub xc_type: DFTType,
    // == for the non_standard setting of DFA ==
    pub xc_namelist: Option<Vec<String>>,
    pub xc_paralist: Option<Vec<f64>>,
    pub dfa_hybrid_scf: Option<f64>,
    // =========================================
    // == for the non_standard setting of DFA ==
    pub xc_model: Option<String>,
    // =========================================
    pub post_xc: Vec<String>,
    pub post_correlation: Vec<DFAFamily>,
    pub post_ai_correction: String,
    pub charge: f64,
    #[pyo3(get, set)]
    pub spin: f64,
    #[pyo3(get, set)]
    pub use_int_nelec: bool,
    pub spin_channel: usize,
    #[pyo3(get, set)]
    pub spin_polarization: bool,
    #[pyo3(get, set)]
    pub frozen_core_postscf: i32,
    #[pyo3(get, set)]
    pub frequency_points: usize,
    #[pyo3(get, set)]
    pub lambda_points: usize,
    #[pyo3(get, set)]
    pub freq_grid_type: usize,
    #[pyo3(get, set)]
    pub freq_cut_off: f64,
    // Keywords for DFT numerical integration
    #[pyo3(get, set)]
    pub radial_precision: f64,
    #[pyo3(get, set)]
    pub min_num_angular_points: usize,
    #[pyo3(get, set)]
    pub max_num_angular_points: usize,
    #[pyo3(get, set)]
    pub grid_gen_level: usize,
    #[pyo3(get, set)]
    pub hardness: usize,
    #[pyo3(get, set)]
    pub pruning: String,
    #[pyo3(get, set)]
    pub rad_grid_method: String,
    #[pyo3(get, set)]
    pub external_grids: String,
    // Keywords for the scf procedures
    #[pyo3(get, set)]
    pub mixer: String,
    #[pyo3(get, set)]
    pub mix_param: f64,
    #[pyo3(get, set)]
    pub num_max_diis: usize,
    #[pyo3(get, set)]
    pub start_diis_cycle: usize,
    #[pyo3(get, set)]
    pub start_check_oscillation: usize,
    #[pyo3(get, set)]
    pub level_shift: Option<f64>,
    #[pyo3(get, set)]
    pub max_scf_cycle: usize,
    #[pyo3(get, set)]
    pub scf_acc_rho: f64,
    #[pyo3(get, set)]
    pub scf_acc_eev: f64,
    #[pyo3(get, set)]
    pub scf_acc_etot:f64,
    #[pyo3(get, set)]
    pub restart: bool,
    #[pyo3(get, set)]
    // The initial MO coefficients and eigenvalues can be imported by setting chkfile
    pub chkfile: String,
    #[pyo3(get, set)]
    // At present, only the hdf5 format is available
    pub chkfile_type: String,
    #[pyo3(get, set)]
    // The initial density matrix can be imported by setting guessfile
    pub guessfile: String,
    #[pyo3(get, set)]
    // At present, only the hdf5 format is available
    pub guessfile_type: String,
    #[pyo3(get, set)]
    pub external_init_guess: bool,
    #[pyo3(get, set)]
    // There are three kinds of available initital guesses: 1) sad (default), 2) hcore, 3) vsap
    pub initial_guess: String,
    #[pyo3(get, set)]
    pub noiter: bool,
    #[pyo3(get, set)]
    pub check_stab: bool,
    #[pyo3(get, set)]
    pub use_dm_only: bool,
    pub algorithm_jk: AlgorithmJK,
    pub algorithm_j: AlgorithmJ,
    pub algorithm_k: AlgorithmK,
    // Keywords for fciqmc dump
    #[pyo3(get, set)]
    pub fciqmc_dump: bool,
    // Kyewords for post scf analysis
    pub outputs: Vec<String>,
    pub cube_orb_setting: [f64;2],
    pub cube_orb_indices: Vec<[usize;3]>,
    pub cube_orb_type: String,
    // keyword for rrs-pbc output
    pub pbc_eigenval: Option<String>,
    //pub output_wfn_in_real_space: usize,
    //pub output_cube: bool,
    //pub output_molden: bool,
    //pub output_fchk: bool,
    // Keywords for sad initial guess
    #[pyo3(get, set)]
    pub atom_sad: bool,
    pub empirical_dispersion: Option<String>,
    pub occupation_type: OCCType,
    pub frac_tolerant: f64,
    // Keywords for DeepPot
    #[pyo3(get, set)]
    pub deep_pot: bool,
    // Keywords for benchmarking various effective potentials, including ECP, ENXC, and Ghost EP
    #[pyo3(get, set)]
    pub bench_eps: bool,
    // Keywords for parallism
    #[pyo3(get, set)]
    pub num_threads: Option<usize>,
    // batch size for each thread
    pub batch_size: usize,
    pub nforce_displacement: f64,
    pub ndipole_displacement: f64,
    pub force_state_occupation: Vec<ForceStateOccupation>,
    pub auxiliary_reference_states: Vec<(String,usize)>,
    pub rpa_de_excitation_parameters: Option<[f64;4]>,
    pub pt2_mpi_mode: usize,
    /// Maximum memory available in MB, `None` if no limit.
    /// This option is only for single-node computation, and only works in some cases where algorithm awares memory usage and perform batched computation.
    /// For multi-node (MPI), this keyword is not fully discussed.
    pub max_memory: Option<f64>,
    pub guess_mix: bool,
    pub guess_mix_theta_deg: Vec<f64>,
    pub spin_correction_scheme: Option<String>,
    pub yamaguchi_triplet_type: Option<String>,
    /// External dipole field (x, y, z) intensity in atomic units
    pub ext_field_dipole: Option<[f64; 3]>,
    pub opt_engine: Option<String>,
    pub geometric_pyo3: Option<GeomeTRIC>,
    pub quasiparticle_methods:Option<QuasiParticle>,
}

impl Default for InputKeywords {
    fn default() -> Self {
        InputKeywords::init_ctrl()
    }
}

impl InputKeywords {
    pub fn init_ctrl() -> InputKeywords {
        InputKeywords{
            // keywords for machine and debug info
            print_level: 0,
            num_threads: Some(1),
            batch_size: 64,
            job_type: JobType::SinglePoint,
            nforce_displacement: 0.0013,
            ndipole_displacement: 3e-4,
            // Keywords for (aux)-basis sets
            basis_path: String::from("./STO-3G"),
            basis_type: String::from("spheric"),
            auxbas_path: String::from("./def2-SV(P)-JKFIT"),
            auxbas_type: String::from("spheric"),
            use_auxbas: true,
            auxbasis_response: true,
            numerical_force: false,
            use_isdf: false,
            ri_k_only: false,
            isdf_k_only: false,
            isdf_k_mu: 17,
            isdf_new: false,
            // Keywords associated with the method employed
            xc: String::from("x3lyp"),
            xc_type: DFTType::Standard,
            xc_namelist: None,
            xc_paralist: None,
            dfa_hybrid_scf: None,
            xc_model: None,
            empirical_dispersion: None,
            post_xc: vec![],
            post_correlation: vec![],
            post_ai_correction: String::from("none"),
            eri_type: String::from("ri_v"),
            use_ri_symm: true,
            charge: 0.0_f64,
            spin: 1.0_f64,
            use_int_nelec: true,
            spin_channel: 1_usize,
            spin_polarization: false,
            // Keywords for frozen-core algorithms
            frozen_core_postscf: 0_i32,
            // Keywords for RPA frequence tabulation
            frequency_points: 20_usize,
            freq_grid_type: 0_usize,
            freq_cut_off: 10.0_f64,
            // Keywords for scsRPA lambda tabulation
            lambda_points: 20_usize,
            // Keywords for DFT numerical integration
            radial_precision: 1.0e-12,
            min_num_angular_points: 110,
            max_num_angular_points: 110,
            hardness: 3,
            grid_gen_level: 3,
            pruning: String::from("nwchem"),
            rad_grid_method: String::from("treutler"),
            external_grids: "none".to_string(),
            // ETB for autogen the auxbasis
            even_tempered_basis: false,
            etb_start_atom_number: 37,
            etb_beta: 2.0,
            // Keywords for the scf procedures
            chkfile: String::from("none"),
            chkfile_type: String::from("hdf5"),
            guessfile: String::from("none"),
            guessfile_type: String::from("hdf5"),
            mixer: String::from("diis"),
            mix_param: 0.6,
            num_max_diis: 8,
            start_diis_cycle: 1,
            start_check_oscillation: 20,
            level_shift : None, 
            max_scf_cycle: 100,
            scf_acc_rho: 1.0e-6,
            scf_acc_eev: 1.0e-5,
            scf_acc_etot:1.0e-8,
            restart: false,
            external_init_guess: false,
            initial_guess: String::from("sad"),
            noiter: false,
            check_stab: false,
            // Kyewords for the manner to evaluate the Vk (and also Vxc) potentials
            // True:  using only density matrix in the evaluation
            // False: use coefficients as well with higher efficiency
            use_dm_only: false,
            algorithm_jk: AlgorithmJK::Default,
            algorithm_j: AlgorithmJ::Default,
            algorithm_k: AlgorithmK::Default,
            // Keywords for the fciqmc dump
            fciqmc_dump: false,
            // Keywords for post scf
            outputs: vec![],
            cube_orb_setting: [3.0,80.0],
            cube_orb_indices: Vec::new(),
            cube_orb_type: String::from("wavefunction"),
            // keyword for rrs-pbc output
            pbc_eigenval: None,
            //output_wfn_in_real_space: 0,
            //output_cube: false,
            //output_molden: false,
            //output_fchk: false,
            // Keyword to turn on atom calculations for the SAD initial guess
            atom_sad: false,
            // Derived keywords of identifying the method used
            //use_dft: false,
            //dft_type: None,
            deep_pot: false,
            bench_eps: false,
            occupation_type: OCCType::INTEGER,
            frac_tolerant: 1.0e-3,
            auxiliary_reference_states: Vec::new(),
            force_state_occupation: Vec::new(),
            rpa_de_excitation_parameters: None,
            pt2_mpi_mode: 0,
            max_memory: None,
            guess_mix: false,
            guess_mix_theta_deg: [15.0, 15.0].to_vec(),
            spin_correction_scheme: None,
            yamaguchi_triplet_type: None,
            ext_field_dipole: None,
            opt_engine: None,
            geometric_pyo3: None,
            quasiparticle_methods:None,
        }
    }

    pub fn formated_output_in_toml(&self) -> String {
        toml::to_string(self).unwrap()
    }

}


#[test]
fn iter_inputkeywords()  {
    let dd = InputKeywords::init_ctrl();
    let ff = toml::to_string(&dd).unwrap();
    println!("{}", ff);
}

pub fn overall_parse_and_report_on_ctrl_geom(ctrl: &mut InputKeywords, geom: &mut GeomCell) {
    println!("=========================================================");
    println!("      Input parameters for the REST calculation");
    println!("=========================================================");

    match ctrl.job_type.clone() {
        JobType::SinglePoint => {println!("Calculation type: Single-point energy")},
        JobType::Force => {println!("Calculation type: Force calculation")},
        JobType::NumDipole => {println!("Calculation type: Numerical dipole calculation")},
        JobType::GeomOpt => {
            println!("Calculation type: Geometry optimization");
            // opt_engine: available options: "lbfgs", "geometric-pyo3"; default: "geometric-pyo3"
            if let Some(opt_engine) = &mut ctrl.opt_engine {
                if ctrl.print_level >= 1 {
                    println!("Optimization engine: {}", opt_engine);
                }
            } else {
                ctrl.opt_engine = Some("geometric_pyo3".to_string());
                if ctrl.print_level >= 1 {
                    println!("Optimization engine: Default (geometric_pyo3)");
                }
            }
        },
    }

    // To make sure the geometric_pyo3 structure is initialized properly
    // The geometric_pyo3 block has higher priority than the corresponding settings in the ctrl block
    if let JobType::GeomOpt = ctrl.job_type.clone() {
        if let Some(opt_engine) = ctrl.opt_engine.clone() {
            if opt_engine.eq("geometric_pyo3") {
                println!("Geometric_pyo3 is used for geometry optimization");
                if let Some(geometric_pyo3) = &ctrl.geometric_pyo3 {
                    if geometric_pyo3.transition {
                        println!("Transition state optimization is enabled");
                    } else if geometric_pyo3.irc {
                        println!("IRC optimization is enabled");
                    } else {
                        println!("Standard geometry optimization is enabled");
                    }
                } else {
                    ctrl.geometric_pyo3 = Some(GeomeTRIC::default());
                    println!("Standard geometry optimization is enabled");
                }
            } else if opt_engine.eq("lbfgs") {
                println!("L-BFGS is used for geometry optimization");
            } else {
                panic!("Error:: Unknown optimization engine: {}", opt_engine);
            }
        }
    };

    if ctrl.cube_orb_type.eq("wavefunction") {
        println!("Cube output: wavefunction");
    } else if ctrl.cube_orb_type.eq("density") {
        println!("Cube output: density");
    } else {
        panic!("Error:: Unknown cube output type: {}", ctrl.cube_orb_type);
    }

    if ctrl.xc.eq("dl_dft") {
        ctrl.xc_type = DFTType::DeepLearning
    };
    match ctrl.xc_type {
        DFTType::Standard => {
            println!("The exchange-correlation method: {}", ctrl.xc);
        },
        DFTType::NonStandard => {
            if let (Some(xc_namelist), Some(xc_paralist), Some(dfa_hybrid_scf)) = (&ctrl.xc_namelist, &ctrl.xc_paralist, &ctrl.dfa_hybrid_scf) {
                println!("Nonstandard exchange-correlation method with {:16.8} exact exchange is employed:", dfa_hybrid_scf);
                xc_namelist.iter().zip(xc_paralist.iter()).for_each(|(xc_name, xc_para)| {
                    println!("    Component: {:>20}, parameter: {:16.8}", xc_name, xc_para);
                });
            } else {
                panic!("Error:: xc_namelist, xc_paralist and xc_hybrid_para should be specified for nonstandard xc method")
            }
        },
        DFTType::DeepLearning => {
            if let Some(xc_model) = &ctrl.xc_model {
                println!("Deep-learning exchange-correlation model is employed");
            } else {
                panic!("Error:: xc_model should be specified for deep-learning xc methods")
            }
        },
    };
    println!("Print level:                {}", ctrl.print_level);
    if let Some(num_threads) = ctrl.num_threads {
        println!("The number of threads used for parallelism:      {}", num_threads);
    } else {
        println!("The number of threads used for parallelism:      {}", rayon::current_num_threads());
    }
    println!("The {}-GTO basis set is taken from {}", ctrl.basis_type,ctrl.basis_path);
    if ctrl.use_auxbas {
        println!("The {}-GTO auxiliary basis set is taken from {}", ctrl.auxbas_type,ctrl.auxbas_path)
    };
    if ctrl.even_tempered_basis {
        println!("Even tempered basis generation starts at: {}", ctrl.etb_start_atom_number);
    }
    println!("Charge: {:3}; Spin: {:3}",ctrl.charge,ctrl.spin);
    if ctrl.spin_channel == 1 {
        println!("Spin polarization: Off")
    } else if ctrl.spin_channel == 2 {
        println!("Spin polarization: On")
    };

    println!("Input molecular structure (in Angstrom): ----------");
    println!("{}", geom.formated_geometry());
    println!("End of molecular structure ------------------------");

    if ctrl.print_level>0 {
        println!("ERI Type: {}", ctrl.eri_type);
        println!("SCF convergency thresholds: {:e} for density matrix", ctrl.scf_acc_rho);
        println!("                            {:e} Ha. for sum of eigenvalues", ctrl.scf_acc_eev);
        println!("                            {:e} Ha. for total energy", ctrl.scf_acc_etot);
        println!("Max. SCF cycle number:      {}", ctrl.max_scf_cycle);
        match geom.pbc {
            MOrC::Molecule => println!("It is a finite cluster calculation"),
            MOrC::Crystal => println!("It is a periodic calculation")
        }
        if ctrl.restart && ! std::path::Path::new(&ctrl.chkfile).exists() {
            println!("The specified checkfile is missing, which will be created after the SCF procedure \n({})",&ctrl.chkfile)
        } else if ctrl.restart && ! ctrl.external_init_guess {
            println!("The initial guess will be obtained from the existing checkfile \n({})",&ctrl.chkfile)
        } else {
            println!("The specified checkfile exists but is not loaded because the keyword 'external_init_guess' is specified");
            println!("It will be updated after the SCF procedure \n({})",&ctrl.chkfile)
        };


    }
    if ctrl.print_level>1 {
        if ctrl.use_ri_symm {
            println!("Turn on the basis pair symmetry for RI 3D-tensors")
        } else {
            println!("Turn off the basis pair symmetry for RI 3D-tensors")
        };
        println!("The pruning method is {}", ctrl.pruning);
        println!("The radial grid generation method is {}", ctrl.rad_grid_method);
        println!("min_num_angular_points: {}", ctrl.min_num_angular_points);
        println!("max_num_angular_points: {}", ctrl.max_num_angular_points);
        println!("hardness: {}", ctrl.hardness);
        println!("Grid generation level: {}", ctrl.grid_gen_level);
        println!("Even tempered basis generation: {}", ctrl.even_tempered_basis);
        let tmp_mixer = ctrl.mixer.clone();
        if tmp_mixer.eq(&"direct") {
            println!("No charge density mixing is employed for the SCF procedure");
        } else if tmp_mixer.eq(&"linear") {
            println!("The {} mixing is employed with the mixing parameter of {} for the SCF procedure", 
                      &tmp_mixer, &ctrl.mix_param);
        } else if tmp_mixer.eq(&"ddiis") 
               || tmp_mixer.eq(&"diis") {
            println!("The {} mixing with (param, max_vec_len) = ({}, {}) is employed for the SCF procedure", 
                      &tmp_mixer, &ctrl.mix_param, &ctrl.num_max_diis);
            println!("Turn on the {} mixing after {} step(s) of SCF iteractions with the linear mixing", 
                      &tmp_mixer, &ctrl.start_diis_cycle);
        } else {
            //ctrl.mixer = String::from("direct");
            panic!("Unknown charge density mixer ({})! No charge density mixing will be invoked.", ctrl.mixer);
        };
        println!("Initial guess is prepared by ({}).", &ctrl.initial_guess);

        if ctrl.external_init_guess {
            println!("The initial guess is obtained from the specified file \n({})", &ctrl.guessfile);
        }

        if ctrl.guess_mix {
            println!("Initial guess mixing enabled: HOMO-LUMO rotated with theta = {:.1}° (alpha), {:.1}° (beta) to induce symmetry breaking",
                ctrl.guess_mix_theta_deg[0], ctrl.guess_mix_theta_deg[1]);
        }

    }
    println!("=========================================================");

}

pub fn parse_ctrl_keywords(tmp_keys: &serde_json::Value) -> anyhow::Result<InputKeywords> { 
    let mut tmp_input = InputKeywords::init_ctrl();
    //==================================================================
    //
    //  parse the keywords from the "ctrl" block
    //
    //==================================================================
    match tmp_keys.get("ctrl").unwrap_or(&serde_json::Value::Null) {
        serde_json::Value::Object(tmp_ctrl) => {
            // =====================================
            //  Keywords for machine info and debug 
            // =====================================
            tmp_input.print_level = match tmp_ctrl.get("print_level").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(1_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(1) as usize},
                other => {1_usize},
            };
            //let default_rayon_current_num_threads = rayon::current_num_threads();
            tmp_input.num_threads = match tmp_ctrl.get("num_threads").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {Some(tmp_str.to_lowercase().parse().unwrap_or(1))},
                serde_json::Value::Number(tmp_num) => {Some(tmp_num.as_i64().unwrap_or(1) as usize)},
                other => {Some(1)},
            };
            tmp_input.batch_size = match tmp_ctrl.get("batch_size").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(64)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(64) as usize},
                other => {64},
            };
            tmp_input.pt2_mpi_mode = match tmp_ctrl.get("pt2_mpi_mode").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(0)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(0) as usize},
                other => {0},
            };
            if let Some(num_threads) = tmp_input.num_threads {
                //if tmp_input.print_level>0 {println!("The number of threads used for parallelism:      {}", num_threads)};
                // Now move the setting of rayon thread numbers to the main.rs
                //rayon::ThreadPoolBuilder::new().num_threads(num_threads);
                rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global().unwrap_or_else(|x| {println!("{:?}", &x)});
                omp_set_num_threads_wrapper(num_threads);
            } else {
                omp_set_num_threads_wrapper(rayon::current_num_threads());
                //if tmp_input.print_level>0 {println!("The default rayon num_threads value is used:      {}", rayon::current_num_threads())};
            };
            //println!("max_num_threads: {}, current_num_threads: {}", rayon::max_num_threads(), rayon::current_num_threads());
            // ====================================
            //  Keywords for the (aux) basis sets
            // ====================================

            // if env REST_BASIS_DIR is set, use it as rest_basis_dir, 
            // otherwise use the default path from rest docker's convention or $REST_HOME/rest/basis-set-pool/
            let rest_basis_dir = path_util::get_rest_basis_dir(tmp_input.print_level);

            tmp_input.basis_path = match tmp_ctrl.get("basis_path").unwrap_or(&serde_json::Value::Null) {
               serde_json::Value::String(tmp_bas) => {
                    path_util::get_valid_basis_path(&tmp_bas, &rest_basis_dir, "basis")
               },
               other => {
                    if ! std::path::Path::new(&String::from("./")).is_dir() {
                        panic!("The specified folder for the basis sets is missing. REST trys to find the basis set from the current folder: (./)");
                    };
                    String::from("./")
               }
            };
            tmp_input.basis_type = match tmp_ctrl.get("basis_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_type) => {tmp_type.to_lowercase()},
                other => {String::from("spheric")}
            };
            //if tmp_input.print_level> 0 {println!("The {}-GTO basis set is taken from {}", tmp_input.basis_type,tmp_input.basis_path)};

            tmp_input.pruning = match tmp_ctrl.get("pruning").unwrap_or(&serde_json::Value::Null){
                serde_json::Value::String(tmp_type) => {tmp_type.to_lowercase()},
                other => {String::from("nwchem")} //default prune method: sg1
            };
            //if tmp_input.print_level>0 {println!("The pruning method will be {}", tmp_input.pruning)};

            tmp_input.rad_grid_method = match tmp_ctrl.get("radial_grid_method").unwrap_or(&serde_json::Value::Null){
                serde_json::Value::String(tmp_type) => {tmp_type.to_lowercase()},
                other => {String::from("treutler")} //default prune method: sg1
            };
            //if tmp_input.print_level>0 {println!("The radial grid generation method will be {}", tmp_input.rad_grid_method)};

            tmp_input.eri_type = match tmp_ctrl.get("eri_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_eri) => {
                    if tmp_eri.to_lowercase().eq("ri_v") || tmp_eri.to_lowercase().eq("ri-v")
                    {
                        String::from("ri_v")
                    } else {tmp_eri.to_lowercase()}
                },
                other => {String::from("ri_v")},
            };
            let eri_type = tmp_input.eri_type.clone();
            if eri_type.eq(&String::from("ri_v"))
            {
                tmp_input.use_auxbas = true;
                tmp_input.use_isdf = false;
                tmp_input.ri_k_only = false;
            } else if eri_type.eq(&String::from("ri_k")) {
                tmp_input.eri_type = String::from("ri_v");
                tmp_input.use_auxbas = true;
                tmp_input.use_isdf = false;
                tmp_input.ri_k_only = true;
            } else if eri_type.eq(&String::from("isdf_full")) {
                // =========== for debug use by IGOR =================
                tmp_input.eri_type = String::from("ri_v");
                //====================================================
                tmp_input.use_auxbas = true;
                tmp_input.use_isdf = true;
            }else if eri_type.eq(&String::from("isdf_k_new")){
                    tmp_input.use_auxbas = true;
                    tmp_input.use_isdf = true;
                    tmp_input.isdf_k_only = true;
                    tmp_input.eri_type = String::from("ri_v");
                    tmp_input.isdf_new = true;
            }else if  eri_type.eq(&String::from("isdf_k")){
                    tmp_input.use_auxbas = true;
                    tmp_input.use_isdf = true;
                    tmp_input.isdf_k_only = true;
                    tmp_input.eri_type = String::from("ri_v");
                    tmp_input.isdf_new = false;
                    //println!("Initial use_isdf: {}", tmp_input.use_isdf);
            }else {
                tmp_input.use_auxbas = false;
                tmp_input.use_isdf = false;
            };
            //if tmp_input.print_level>0 {println!("ERI Type: {}", tmp_input.eri_type)};

            tmp_input.use_ri_symm = match tmp_ctrl.get("use_ri_symm").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {true},
            };
            //if tmp_input.print_level>0 {
            //    if tmp_input.use_ri_symm {
            //        println!("Turn on the basis pair symmetry for RI 3D-tensors")
            //    } else {
            //        println!("Turn off the basis pair symmetry for RI 3D-tensors")
            //    };
            //}
            tmp_input.isdf_k_mu = match tmp_ctrl.get("isdf_k_mu").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(8_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(8) as usize},
                other => {8_usize},
            };            

            tmp_input.auxbas_type = match tmp_ctrl.get("auxbas_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_type) => {tmp_type.to_lowercase()},
                other => {String::from("spheric")}
            };
            tmp_input.auxbas_path = match tmp_ctrl.get("auxbas_path").unwrap_or(&serde_json::Value::Null) {
               serde_json::Value::String(tmp_bas) => {
                    path_util::get_valid_basis_path(&tmp_bas, &rest_basis_dir, "auxiliary basis")
               },
               other => {
                    //if ! std::path::Path::new(&String::from("./")).is_dir() {
                    //    println!("The specified folder for the auxiliar basis sets is missing: (./)");
                    //};
                    println!("No auxiliary basis set is specified. REST will try to find the auxiliary basis from the current folder: (./)");
                    let default_bas = String::from("./");
                    if ! std::path::Path::new(&default_bas).is_dir() {
                        //tmp_input.use_auxbas = false;
                    } else {
                        //tmp_input.use_auxbas = true;
                    }
                    default_bas
               }
            };
            //if tmp_input.use_auxbas && tmp_input.print_level>0 {
            //    println!("The {}-GTO auxiliary basis set is taken from {}", tmp_input.auxbas_type,tmp_input.auxbas_path)
            //};
            // ===============================================
            //  Keywords for Gradient calculation
            // ==============================================
            tmp_input.auxbasis_response = match tmp_ctrl.get("auxbasis_response").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => *tmp_str,
                other => true,
            };
            tmp_input.numerical_force = match tmp_ctrl.get("numerical_force").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => *tmp_str,
                other => false,
            };
            // ==============================================
            //  JobType
            // ==============================================
            tmp_input.job_type = match tmp_ctrl.get("job_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_xc) => {
                    let tmp_xc_low = tmp_xc.to_lowercase();
                    if tmp_xc_low.eq("opt") || tmp_xc_low.eq("geometry optimization") || 
                       tmp_xc_low.eq("geometry relaxation") || tmp_xc_low.eq("geom_opt") ||
                       tmp_xc_low.eq("geom_relax") || tmp_xc_low.eq("relax") {
                        JobType::GeomOpt
                    } else if tmp_xc_low.eq("force") || tmp_xc_low.eq("gradient") {
                        JobType::Force
                    } else if tmp_xc_low.eq("numdipole") || tmp_xc_low.eq("numerical dipole") {
                        JobType::NumDipole
                    } else if tmp_xc_low.eq("energy") || tmp_xc_low.eq("single point") ||
                      tmp_xc_low.eq("single_point") {
                        JobType::SinglePoint 
                    } else {
                        JobType::SinglePoint
                    }
                },
                other => {JobType::SinglePoint},
            };
            tmp_input.nforce_displacement = match tmp_ctrl.get("nforce_displacement").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_nforce) => {tmp_nforce.to_lowercase().parse().unwrap_or(0.0013)},
                serde_json::Value::Number(tmp_nforce) => {tmp_nforce.as_f64().unwrap_or(0.0013)},
                other => {0.0013},
            };
            tmp_input.ndipole_displacement = match tmp_ctrl.get("nforce_displacement").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_nforce) => {tmp_nforce.to_lowercase().parse().unwrap_or(3e-4)},
                serde_json::Value::Number(tmp_nforce) => {tmp_nforce.as_f64().unwrap_or(3e-4)},
                serde_json::Value::Null => {3e-4},
                other => panic!("The ndipole_displacement is not recognized"),
            };
            // ==============================================
            //  Keywords associated with the method employed
            // ==============================================
            tmp_input.xc = match tmp_ctrl.get("xc").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_xc) => {tmp_xc.to_lowercase()},
                other => {String::from("hf")},
            };
            tmp_input.xc_type = match tmp_ctrl.get("xc_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_xc) => {
                    if tmp_xc.to_lowercase().eq("nonstandard") || tmp_xc.to_lowercase().eq("non-standard") {
                        DFTType::NonStandard
                    } else if tmp_xc.to_lowercase().eq("deep-learning")  
                           || tmp_xc.to_lowercase().eq("deep_learning") 
                           || tmp_xc.to_lowercase().eq("deep learning") 
                           || tmp_xc.to_lowercase().eq("machine learning") 
                           || tmp_xc.to_lowercase().eq("machine-learning") 
                           || tmp_xc.to_lowercase().eq("machine_learning") 
                    {
                        DFTType::DeepLearning
                    } else if tmp_xc.to_lowercase().eq("standard")  {
                        DFTType::Standard
                    } else {
                        println!("Unknown xc_type: ({}). xc_type is set to `standard`", tmp_xc);
                        DFTType::Standard
                    }
                },
                other => {
                    DFTType::Standard
                },
            };
            tmp_input.xc_namelist = match tmp_ctrl.get("xc_namelist").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_op) => {Some(vec![tmp_op.to_lowercase()])},
                serde_json::Value::Array(tmp_op) => {
                    let mut tmp_vec:Vec<String> = vec![];
                    tmp_op.iter().for_each(|x| {
                        let op_type = x.to_string();
                        let string_len = op_type.len();
                        tmp_vec.push(op_type[1..string_len-1].to_lowercase().to_string())
                    });
                    Some(tmp_vec)
                },
                other => {None},
            };
            tmp_input.xc_paralist = match tmp_ctrl.get("xc_paralist").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Array(tmp_op) => {
                    let tmp_vec:Vec<f64> = tmp_op.iter().map(|x| {
                        match x {
                            serde_json::Value::String(tmp_str) => {tmp_str.parse().unwrap_or(0.0)},
                            serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(0.0)},
                            other => {0.0},
                        }
                    }).collect::<Vec<f64>>();
                    Some(tmp_vec)
                },
                other => {None},
            };
            tmp_input.dfa_hybrid_scf = match tmp_ctrl.get("xc_hybrid_para").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {Some(tmp_str.parse().unwrap_or(0.0))},
                serde_json::Value::Number(tmp_num) => {Some(tmp_num.as_f64().unwrap_or(0.0))},
                other => {None}
            };
            tmp_input.xc_model = match tmp_ctrl.get("xc_model").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_xc) => {
                    tmp_input.xc_type=DFTType::DeepLearning; 
                    Some(tmp_xc.to_lowercase())
                },
                other => {None},
            };

            tmp_input.empirical_dispersion = match tmp_ctrl.get("empirical_dispersion").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_emp) => {
                    if tmp_emp.to_lowercase() == "none" {
                       None 
                    } else if tmp_emp.to_lowercase() == "true" {
                        Some("d3bj".to_string())
                    } else {
                       Some(tmp_emp.to_lowercase())
                    }
                },
                other => {None},
            };
            //let re0 = Regex::new(r"
            //                    (?P<elem>\w{1,2})\s*,?    # the element
            //                    \s+
            //                    (?P<x>[\+-]?\d+.\d+)\s*,? # the 'x' position
            //                    \s+
            //                    (?P<y>[\+-]?\d+.\d+)\s*,? # the 'y' position
            //                    \s+
            //                    (?P<z>[\+-]?\d+.\d+)\s*,? # the 'z' position
            //                    \s*").unwrap();
            tmp_input.post_xc = match tmp_ctrl.get("post_xc").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_xc) => {vec![tmp_xc.to_lowercase()]},
                serde_json::Value::Array(tmp_xc) => {
                    let mut tmp_vec:Vec<String> = vec![];
                    tmp_xc.iter().for_each(|x| {
                        let xc_method = x.to_string();
                        let string_len = xc_method.len();
                        tmp_vec.push(xc_method[1..string_len-1].to_string())
                    });
                    tmp_vec
                },
                other => {vec![]},
            };
            let post_corr = match tmp_ctrl.get("post_correlation").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_xc) => {vec![tmp_xc.to_lowercase()]},
                serde_json::Value::Array(tmp_xc) => {
                    let mut tmp_vec:Vec<String> = vec![];
                    tmp_xc.iter().for_each(|x| {
                        let xc_method = x.to_string();
                        let string_len = xc_method.len();
                        tmp_vec.push(xc_method[1..string_len-1].to_string())
                    });
                    tmp_vec
                },
                other => {vec![]},
            };
            tmp_input.post_correlation = vec![];
            post_corr.iter().for_each(|corr| {
                if corr.to_lowercase().eq("pt2") {
                    tmp_input.post_correlation.push(DFAFamily::PT2)
                } else if corr.to_lowercase().eq("sbge2") {
                    tmp_input.post_correlation.push(DFAFamily::SBGE2)
                } else if corr.to_lowercase().eq("rpa") {
                    tmp_input.post_correlation.push(DFAFamily::RPA)
                } else if corr.to_lowercase().eq("scsrpa") {
                    tmp_input.post_correlation.push(DFAFamily::SCSRPA)
                } else {
                    println!("WARNNING: Unknown post-scf correlation method: {}", corr)
                }
                //if corr.to_lowercase().eq(&pt2) 
            });
            tmp_input.post_ai_correction = match tmp_ctrl.get("post_ai_correction").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_xc) => {tmp_xc.to_lowercase()},
                other => {String::from("none")},
            };
            // ===============================================
            //  Keywords to determine the spin channel, which 
            //   is important to turn on RHF(RKS) or UHF(UKS)
            // ==============================================
            tmp_input.charge = match tmp_ctrl.get("charge").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_charge) => {tmp_charge.to_lowercase().parse().unwrap_or(0.0)},
                serde_json::Value::Number(tmp_charge) => {tmp_charge.as_f64().unwrap_or(0.0)},
                other => {0.0},
            };
            tmp_input.spin = match tmp_ctrl.get("spin").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_spin) => {tmp_spin.to_lowercase().parse().unwrap_or(0.0)},
                serde_json::Value::Number(tmp_spin) => {tmp_spin.as_f64().unwrap_or(0.0)},
                other => {0.0},
            };
            tmp_input.use_int_nelec = match tmp_ctrl.get("use_int_nelec").unwrap_or(&serde_json::Value::Null) {
                // serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(true)},
                serde_json::Value::Bool(tmp_bool) => tmp_bool.clone(),
                other => {true},
            };
            tmp_input.spin_polarization = match tmp_ctrl.get("spin_polarization").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value:: String(tmp_str) => tmp_str.to_lowercase().parse().unwrap_or(false),
                serde_json::Value:: Bool(tmp_bool) => tmp_bool.clone(),
                other => false,
            };
            tmp_input.spin_channel = if tmp_input.spin_polarization {
                //if tmp_input.print_level>0 {println!("Spin polarization: On")};
                2_usize
            } else {
                //if tmp_input.print_level>0 {println!("Spin polarization: Off")};
                1_usize
            };
            // ==============================================
            //  Keywords of setting the frozen-core algorithm
            // ==============================================
            tmp_input.frozen_core_postscf = match tmp_ctrl.get("frozen_core_postscf").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_fc) => {tmp_fc.to_lowercase().parse().unwrap_or(0)},
                serde_json::Value::Number(tmp_fc) => {tmp_fc.as_i64().unwrap_or(0) as i32},
                other => {0},
            };
            // ==============================================
            //  Keywords of setting the frequency tabulation
            // ==============================================
            tmp_input.frequency_points = match tmp_ctrl.get("frequency_points").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_fp) => {tmp_fp.to_lowercase().parse().unwrap_or(20_usize)},
                serde_json::Value::Number(tmp_fp) => {tmp_fp.as_i64().unwrap_or(20) as usize},
                other => {20_usize},
            };
            tmp_input.freq_grid_type = match tmp_ctrl.get("freq_grid_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_fg) => {tmp_fg.to_lowercase().parse().unwrap_or(0)},
                serde_json::Value::Number(tmp_fg) => {tmp_fg.as_i64().unwrap_or(0) as usize},
                other => {0},
            };
            tmp_input.freq_cut_off = match tmp_ctrl.get("freq_cut_off").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_fg) => {tmp_fg.to_lowercase().parse().unwrap_or(10.0)},
                serde_json::Value::Number(tmp_fg) => {tmp_fg.as_f64().unwrap_or(10.0)},
                other => {10.0},
            };
            tmp_input.lambda_points = match tmp_ctrl.get("lambda_points").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_fp) => {tmp_fp.to_lowercase().parse().unwrap_or(20_usize)},
                serde_json::Value::Number(tmp_fp) => {tmp_fp.as_i64().unwrap_or(20) as usize},
                other => {20_usize},
            };
            //===============================================
            // Keywords for fciqmc dump
            //===============================================
            tmp_input.fciqmc_dump = match tmp_ctrl.get("fciqmc_dump").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_bool) => {tmp_bool.clone()},
                other => {false},
            };


            // ==============================================
            //  Keywords associated with DFT grids
            // ==============================================
            tmp_input.radial_precision = match tmp_ctrl.get("radial_precision").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(1.0e-12)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.0e-12)},
                other => {1.0e-12}
            };
            tmp_input.min_num_angular_points = match tmp_ctrl.get("min_num_angular_points").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(110_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(110) as usize},
                other => {110_usize}
            };
            tmp_input.max_num_angular_points = match tmp_ctrl.get("max_num_angular_points").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(590_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(590) as usize},
                other => {590_usize}
            };
            tmp_input.hardness = match tmp_ctrl.get("hardness").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(3_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(3) as usize},
                other => {3_usize}
            };

            tmp_input.external_grids = match tmp_ctrl.get("external_grids").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_type) => {
                    if tmp_input.print_level>0 {println!("Read grids from the external file: {}", tmp_type)};
                    tmp_type.to_string()},
                other => {String::from("grids")}
            };

            tmp_input.grid_gen_level = match tmp_ctrl.get("grid_generation_level").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(3_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(3) as usize},
                other => {3_usize},
            };

            tmp_input.even_tempered_basis = match tmp_ctrl.get("even_tempered_basis").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };

            tmp_input.etb_start_atom_number = match tmp_ctrl.get("etb_start_atom_number").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(37_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(37) as usize},
                other => {37_usize},
            };

            tmp_input.etb_beta = match tmp_ctrl.get("etb_beta").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(2.0_f64)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(2.0_f64)},
                other => {2.0_f64},
            };

            // ==============================================
            //  Keywords associated with the SCF procedure
            // ==============================================
            tmp_input.max_scf_cycle = match tmp_ctrl.get("max_scf_cycle").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(100_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(100) as usize},
                other => {100_usize}
            };
            tmp_input.level_shift = match tmp_ctrl.get("level_shift").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {
                    let num = tmp_str.to_lowercase().parse().unwrap_or(0.0);
                    if num == 0.0 {
                        None
                    } else {
                        Some(num)
                    }
                },
                serde_json::Value::Number(tmp_num) => {
                    let num = tmp_num.as_f64().unwrap_or(0.0);
                    if num == 0.0 {
                        None
                    } else {
                        Some(num)
                    }
                },
                other => {None}
            };
            tmp_input.scf_acc_rho = match tmp_ctrl.get("scf_acc_rho").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(1.0e-6)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.0e-8)},
                other => {1.0e-8}
            };
            tmp_input.scf_acc_eev = match tmp_ctrl.get("scf_acc_eev").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(1.0e-6)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.0e-6)},
                other => {1.0e-6}
            };
            tmp_input.scf_acc_etot = match tmp_ctrl.get("scf_acc_etot").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(1.0e-8)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.0e-8)},
                other => {1.0e-8}
            };

            tmp_input.mixer = match tmp_ctrl.get("mixer").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase()},
                other => {String::from("diis")},
            };
            tmp_input.mix_param = match tmp_ctrl.get("mix_param").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(0.2)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(0.2)},
                other => {0.2}
            };
            tmp_input.num_max_diis = match tmp_ctrl.get("num_max_diis").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(8_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(8) as usize},
                other => {8_usize}
            };
            tmp_input.start_diis_cycle = match tmp_ctrl.get("start_diis_cycle").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(1_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(1) as usize},
                other => {1_usize}
            };
            tmp_input.start_check_oscillation = match tmp_ctrl.get("start_check_oscillation").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(20_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(20) as usize},
                other => {20_usize}
            };

            // Initial guess relevant keywords
            tmp_input.guessfile = match tmp_ctrl.get("guessfile").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_guess) => tmp_guess.clone(),
                other => String::from("none"),
            };
            tmp_input.guessfile_type = match tmp_ctrl.get("guessfile_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_guess) => tmp_guess.to_lowercase().clone(),
                other => String::from("none"),
            };

            // Fix a bug reported by Linyue Yu, 2024-09-03
            tmp_input.external_init_guess = (! tmp_input.guessfile.to_lowercase().eq(&"none") ) &&
                        std::path::Path::new(&tmp_input.guessfile).exists();

            tmp_input.chkfile = match tmp_ctrl.get("chkfile").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_chk) => tmp_chk.clone(),
                other => String::from("none"),
            };
            tmp_input.chkfile_type = match tmp_ctrl.get("chkfile_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_chk) => tmp_chk.to_lowercase().clone(),
                other => String::from("hdf5"),
            };

            tmp_input.restart = ! tmp_input.chkfile.to_lowercase().eq(&"none");

            tmp_input.initial_guess = match tmp_ctrl.get("initial_guess").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase()},
                other => {String::from("sad")},
            };

            tmp_input.noiter = match tmp_ctrl.get("noiter").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value:: String(tmp_str) => tmp_str.to_lowercase().parse().unwrap_or(false),
                serde_json::Value:: Bool(tmp_bool) => tmp_bool.clone(),
                other => false,
            };
            tmp_input.check_stab = match tmp_ctrl.get("check_stab").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value:: String(tmp_str) => tmp_str.to_lowercase().parse().unwrap_or(false),
                serde_json::Value:: Bool(tmp_bool) => tmp_bool.clone(),
                other => false,
            };
            tmp_input.use_dm_only = match tmp_ctrl.get("use_dm_only").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value:: String(tmp_str) => tmp_str.to_lowercase().parse().unwrap_or(false),
                serde_json::Value:: Bool(tmp_bool) => tmp_bool.clone(),
                other => false,
            };
            // setup and sanity check of J/K algorithms
            tmp_input.algorithm_jk = tmp_ctrl.get("algorithm_jk").map(serde_from_value).unwrap_or_default();
            tmp_input.algorithm_j = tmp_ctrl.get("algorithm_j").map(serde_from_value).unwrap_or_default();
            tmp_input.algorithm_k = tmp_ctrl.get("algorithm_k").map(serde_from_value).unwrap_or_default();
            if (tmp_input.algorithm_j != AlgorithmJ::Default || tmp_input.algorithm_k != AlgorithmK::Default) {
                if tmp_input.algorithm_jk != AlgorithmJK::Default {
                    println!("Warning: algorithm_j or algorithm_k are specified, the setting in algorithm_jk will be ignored.");
                }
                tmp_input.algorithm_jk = AlgorithmJK::Separated(tmp_input.algorithm_j, tmp_input.algorithm_k);
            }
            // ================================================
            //  Keywords associated with the elec occupation 
            // ================================================
            tmp_input.occupation_type = 
            match tmp_ctrl.get("occupation_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_type) => {
                    let tmp_occupation_type = tmp_type.to_lowercase();
                    if tmp_occupation_type.eq("integer") {
                        OCCType::INTEGER
                    } else if tmp_occupation_type.eq("sad") {
                        OCCType::ATMSAD
                    } else if tmp_occupation_type.eq("frac") {
                        OCCType::FRAC
                    } else {
                        OCCType::INTEGER
                    }
                },
                other => OCCType::INTEGER,
            };
            tmp_input.frac_tolerant = match tmp_ctrl.get("frac_tolerant").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(1.0e-3)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.0e-3)},
                other => {1.0e-3}
            };
            tmp_input.force_state_occupation = match tmp_ctrl.get("force_state_occupation").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(_) => vec![],
                serde_json::Value::Array(tmp_op) => {
                    let mut tmp_vec: Vec<ForceStateOccupation> = vec![];
                    tmp_op.iter().for_each(|x| {
                        let tmp_obj = match x {
                            serde_json::Value::Array(tmp_value) => {
                                match tmp_value.len() {
                                    5 => {
                                        // [ref_state, ref_spin, force_occ, min, max]
                                        let ref_state = tmp_value[0].as_u64().unwrap_or(0) as usize;
                                        let ref_spin = tmp_value[1].as_u64().unwrap_or(0) as usize;
                                        let target_spin = ref_spin;
                                        let force_occ = tmp_value[2].as_f64().unwrap_or(0.0);
                                        let check_min = tmp_value[3].as_u64().unwrap_or(0) as usize;
                                        let check_max = tmp_value[4].as_u64().unwrap_or(0) as usize;
                                        Some(ForceStateOccupation::init(
                                            tmp_input.chkfile.clone(),
                                            ref_state,
                                            ref_spin,
                                            target_spin,
                                            force_occ,
                                            check_min,
                                            check_max,
                                        ))
                                    }
                                    6 => {
                                        match &tmp_value[0] {
                                            serde_json::Value::String(ref reference) => {
                                                // ["ref.hdf5", ref_state, ref_spin, force_occ, min, max]
                                                let ref_state = tmp_value[1].as_u64().unwrap_or(0) as usize;
                                                let ref_spin = tmp_value[2].as_u64().unwrap_or(0) as usize;
                                                let target_spin = ref_spin;
                                                let force_occ = tmp_value[3].as_f64().unwrap_or(0.0);
                                                let check_min = tmp_value[4].as_u64().unwrap_or(0) as usize;
                                                let check_max = tmp_value[5].as_u64().unwrap_or(0) as usize;
                                                Some(ForceStateOccupation::init(
                                                    reference.to_string(),
                                                    ref_state,
                                                    ref_spin,
                                                    target_spin,
                                                    force_occ,
                                                    check_min,
                                                    check_max,
                                                ))
                                            }
                                            _ => {
                                                // [ref_state, ref_spin, target_spin, force_occ, min, max]
                                                let ref_state = tmp_value[0].as_u64().unwrap_or(0) as usize;
                                                let ref_spin = tmp_value[1].as_u64().unwrap_or(0) as usize;
                                                let target_spin = tmp_value[2].as_u64().unwrap_or(0) as usize;
                                                let force_occ = tmp_value[3].as_f64().unwrap_or(0.0);
                                                let check_min = tmp_value[4].as_u64().unwrap_or(0) as usize;
                                                let check_max = tmp_value[5].as_u64().unwrap_or(0) as usize;
                                                Some(ForceStateOccupation::init(
                                                    tmp_input.chkfile.clone(),
                                                    ref_state,
                                                    ref_spin,
                                                    target_spin,
                                                    force_occ,
                                                    check_min,
                                                    check_max,
                                                ))
                                            }
                                        }
                                    }
                                    7 => {
                                        // ["ref.hdf5", ref_state, ref_spin, target_spin, force_occ, min, max]
                                        let reference = tmp_value[0].as_str().unwrap_or("none").to_string();
                                        let ref_state = tmp_value[1].as_u64().unwrap_or(0) as usize;
                                        let ref_spin = tmp_value[2].as_u64().unwrap_or(0) as usize;
                                        let target_spin = tmp_value[3].as_u64().unwrap_or(0) as usize;
                                        let force_occ = tmp_value[4].as_f64().unwrap_or(0.0);
                                        let check_min = tmp_value[5].as_u64().unwrap_or(0) as usize;
                                        let check_max = tmp_value[6].as_u64().unwrap_or(0) as usize;
                                        Some(ForceStateOccupation::init(
                                            reference,
                                            ref_state,
                                            ref_spin,
                                            target_spin,
                                            force_occ,
                                            check_min,
                                            check_max,
                                        ))
                                    }
                                    _ => {
                                        panic!("ERROR:: incorrect force_state_occupation setting: {:?}", &tmp_value);
                                    }
                                }
                            }
                            _ => None,
                        };
                        if let Some(tmp_obj) = tmp_obj {
                            tmp_vec.push(tmp_obj);
                        }
                    });
                    tmp_vec
                }
                _ => vec![],
            };

            //
            tmp_input.auxiliary_reference_states = match tmp_ctrl.get("auxiliary_reference_states").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_chk) => vec![(String::from("none"),0)],
                serde_json::Value::Array(tmp_op) => {
                    let mut tmp_files = vec![];
                    tmp_op.iter().for_each(|x| {
                        match x {
                            serde_json::Value::String(tmp_str) => {tmp_files.push((tmp_str.clone(),0))},
                            serde_json::Value::Array(tmp_value) => {
                                let aux_file_name = tmp_value[0].as_str().unwrap().to_string();
                                let global_start = tmp_value[1].as_u64().unwrap() as usize;
                                tmp_files.push((aux_file_name,global_start));
                            },
                            _ => {}
                        }
                    });
                    tmp_files
                },
                other => Vec::new(),
            };
            tmp_input.rpa_de_excitation_parameters = match tmp_ctrl.get("rpa_de_excitation_parameters").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Array(tmp_op) => {
                    if tmp_op.len() == 4 {
                        let mut tmp_array = [0.0;4];
                        tmp_array.iter_mut().zip(tmp_op.iter()).for_each(|(to, from)| {
                            *to = from.as_f64().unwrap()
                        });
                        Some(tmp_array)
                    } else {
                        None
                    }
                },
                other => None,
            };
            // ================================================
            //  Keywords associated with the post-SCF analyais
            // ================================================
            tmp_input.pbc_eigenval = match tmp_ctrl.get("pbc_eigenval").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_op) => {
                    let tmp = match tmp_op.as_str() {
                        "none" => None,
                        "None" => None,
                        _ => Some(String::from(tmp_op))
                    };
                    tmp
                }
                _ => None
            };
            tmp_input.outputs = match tmp_ctrl.get("outputs").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_op) => {vec![tmp_op.to_lowercase()]},
                serde_json::Value::Array(tmp_op) => {
                    let mut tmp_vec:Vec<String> = vec![];
                    tmp_op.iter().for_each(|x| {
                        let op_type = x.to_string();
                        let string_len = op_type.len();
                        tmp_vec.push(op_type[1..string_len-1].to_lowercase().to_string())
                    });
                    tmp_vec
                },
                other => {vec![]},
            };
            tmp_input.cube_orb_type = match tmp_ctrl.get("cube_orb_type").unwrap_or(&serde_json::Value::Null) {
               serde_json::Value::String(tmp_type) => {
                    String::from(tmp_type).to_lowercase()
               },
               other => {
                    String::from("wavefunction")
               }
            };
            tmp_input.cube_orb_setting = match tmp_ctrl.get("cube_orb_setting").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Array(tmp_op) => {
                    let mut tmp_array = [0.0;2];
                    tmp_array.iter_mut().zip(tmp_op[0..2].iter()).for_each(|(to, from)| {
                        match from {
                            serde_json::Value::String(tmp_str) => {*to = tmp_str.parse().unwrap_or(0.0)},
                            serde_json::Value::Number(tmp_num) => {*to = tmp_num.as_f64().unwrap_or(0.0)},
                            other => {*to = 0.0},
                        }
                    });
                    tmp_array
                },
                other => {[3.0,80.0]},
            };
            tmp_input.cube_orb_indices = match tmp_ctrl.get("cube_orb_indices").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Array(tmp_op) => {
                    //let mut tmp_array = [0.0;2];
                    let mut tmp_indices = vec![[0;3];tmp_op.len()];
                    tmp_indices.iter_mut().zip(tmp_op.iter()).for_each(|(to, from)| {
                        let tmp_to = match from {
                            serde_json::Value::Array(tmp_opp) => {
                                let mut tmp_array = [0_usize;3];
                                tmp_array.iter_mut().zip(tmp_opp[0..3].iter()).for_each(|(to, from)| {
                                    match from {
                                        serde_json::Value::String(tmp_str) => {*to = tmp_str.parse().unwrap_or(0)},
                                        serde_json::Value::Number(tmp_num) => {*to = tmp_num.as_u64().unwrap_or(0) as usize},
                                        other => {*to = 0},
                                    }
                                });
                                Some(tmp_array)
                            },
                            other => {None},
                        };
                        if let Some(tmp_array) = tmp_to {
                            *to = tmp_array;
                        };
                    });
                    tmp_indices 
                },
                other => {vec![]},
            };
            tmp_input.deep_pot = match tmp_ctrl.get("deep_potential").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.bench_eps = match tmp_ctrl.get("bench_eps").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };

            // for atom_sad setting
            tmp_input.atom_sad = match tmp_ctrl.get("atom_sad").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };

            tmp_input.max_memory = match tmp_ctrl.get("max_memory").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => Some(tmp_num.as_f64().unwrap()),
                other => None,
            };
            
            // for guess_mix setting
            tmp_input.guess_mix = match tmp_ctrl.get("guess_mix").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_bool) => *tmp_bool,
                serde_json::Value::String(tmp_str) => tmp_str.to_lowercase().parse().unwrap_or(false),
                _ => false,
            };
            
            // for guess_mix_theta_deg: support number, string, or array; default to [15.0, 15.0]
            tmp_input.guess_mix_theta_deg = match tmp_ctrl.get("guess_mix_theta_deg").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => vec![n.as_f64().unwrap_or(15.0); 2],
                serde_json::Value::String(s) => {
                    let val = s.parse::<f64>().unwrap_or(15.0);
                    vec![val; 2]
                }
                serde_json::Value::Array(arr) => {
                    let mut vals = arr.iter().filter_map(|v| v.as_f64()).collect::<Vec<f64>>();
                    if vals.len() == 1 { vec![vals[0]; 2] }
                    else { vals.truncate(2); vals }
                }
                _ => vec![15.0, 15.0],
            };

            tmp_input.spin_correction_scheme = match tmp_ctrl.get("spin_correction_scheme").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_emp) => {Some(tmp_emp.to_lowercase())},
                other => {None},
            };

            tmp_input.yamaguchi_triplet_type = match tmp_ctrl.get("yamaguchi_triplet_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_emp) => {Some(tmp_emp.to_lowercase())},
                other => {None},
            };

            // opt_engine: available options: "lbfgs", "geometric-pyo3"; default: "geometric-pyo3"
            tmp_input.opt_engine = match tmp_ctrl.get("opt_engine").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => { 
                    let s = tmp_str.to_lowercase();
                    match s.to_lowercase().as_str() {
                        "lbfgs"  => Some(s.to_string()),
                        "geometric_pyo3" | "geometric-pyo3" => Some("geometric_pyo3".to_string()),
                        _ => panic!("Not recognized option for opt_engine: {}", s),
                    }
                },
                serde_json::Value::Null => { None },
                _ => panic!("Not recognized type for opt_engine"),
            };
            
            //===========================================================
            // Global check of ctrl keywords and futher modification
            //============================================================
            if tmp_input.even_tempered_basis == true {
                if tmp_input.etb_beta<=1.0f64 {
                    println!("WARNING: etb_beta cannot be below 1.0. REST will use etb_beta=2.0 instead in this calculation");
                    tmp_input.etb_beta=2.0f64;
                }
                //if tmp_input.print_level>0 {
                //    println!("Even tempered basis generation starts at: {}", tmp_input.etb_start_atom_number);
                //    println!("Even tempered basis beta is: {}", tmp_input.etb_beta);
                //}
            }
            if tmp_input.external_init_guess  {
                if ! std::path::Path::new(&tmp_input.guessfile).exists() {
                    println!("WARNING: Initial density matrix is required by the keyword of guessfile, which, however, does not exist: \n({}). \n The external initial guess will not be imported.\n",&tmp_input.guessfile);
                    tmp_input.external_init_guess = false;
                } else {
                    //if tmp_input.print_level>0 {
                    //    println!("The initial guess will be imported from \n({}).\n ",&tmp_input.guessfile);
                    //}
                }
            }
            if tmp_input.force_state_occupation.len()>0 {
                if ! tmp_input.restart {
                    panic!("ERROR: force_state_occupation can not be involved without an existing chkfile \'restart\'");
                } else if ! std::path::Path::new(&tmp_input.chkfile).exists() {
                    panic!("ERROR: force_state_occupation can not be involved without an existing chkfile \'restart\'");
                }
            }
        },
        other => {
            panic!("Error:: no 'ctrl' keyword or some inproper settings of the 'ctrl' keyword in the input file")
        },
    }

    Ok(tmp_input)
}
