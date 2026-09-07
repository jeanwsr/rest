//! Analytical derivative module for REST.
//!
//! This should serve as semi-independent module, handling property computations that requires (only)
//! derivatives. The energy components are considered to be linearly added, so that components serve
//! as independent operators (term of ORCA); trait design of this module will fully support this idea.
//!
//! This is a more modular and flexible design, and should be easier to maintain and extend.
//! The API design document is not written at this time, but will be available in the future.
//!
//! Currently only hessian property is implemented. This module may be full refactored in future to
//! support other types of derivatives and properties.
//!
//! This module should work in most cases, but still requires further testing and efficiency update.
//!
//! This module does not contain extensive detailed implementation. We defined traits,
//! some common implementations (hcore, nuc, ovlp), total hessian, important utilities.
//!
//! - For optimized RI-JK implementation, please refer to [`crate::ri_jk`] module.
//! - For DFT matmul implementation, please refer to [`crate::dft::numint_matmul`] module.
//!
//! We will also handle interface to REST.
//!
//! This module currently does not handle post-SCF derivatives.
//!
//! Some important utilities comes from other programs, and we acknowledge them here.
//! - `vib.rs`: Vibration analysis from Psi4, partially translated by AI, not fully reviewed by human.
//!   - TR/V (translation-rotation and vibration classification) is different to Psi4. We will use rotor-type
//!     to determine number of degrees of freedom (TR mode).
//! - `point_group_detect`: Point group detection from Psi4, translated by AI, not reviewed by human but have been tested.
//!   - Note some point group detection is minorly different (such as C3v).
//! - `krylov_block.rs`: Krylov solver (used in CP-SCF) from PySCF, translated with help by AI, reviewed
//!   by extensive testing.

#![warn(unused)]

// trait definitions
pub mod trait_rhess;
pub mod trait_uhess;
pub mod trait_util;

// core hess implementations
pub mod hcore;
pub mod nuc_repl;

// overlap hess implementations
pub mod ovlp;

// total hess implementations
pub mod rscf;
pub mod uscf;

// total hess interface to REST
pub mod interface;
pub mod rscf_interface;
pub mod uscf_interface;

// vibrational analysis
pub mod vib;
pub mod vib_interface;

// utilities
pub mod cint_handling;
pub mod config;
pub mod krylov_block;
pub mod point_group_detect;

#[allow(unused_imports)]
pub mod prelude {
    use super::*;

    pub use config::AnalDrvConfig;
    pub use hcore::{RHessHcore, UHessHcore};
    pub use nuc_repl::HessNucRepl;
    pub use ovlp::{RHessOvlp, UHessOvlp};
    pub use rscf::RHessSCF;
    pub use trait_rhess::{HessNucAPI, RHessCoreAPI, RHessElecInteractAPI};
    pub use trait_uhess::{UHessCoreAPI, UHessElecInteractAPI};
    pub use trait_util::HessUtilAPI;
    pub use uscf::UHessSCF;

    pub(super) use crate::ri_jk::util::{get_dm0_restricted, get_dme0_restricted};
    pub(super) use crate::utilities::rstsr_util::*;
    pub(super) use cint_handling::*;
    pub(super) use itertools::Itertools;
    pub(super) use krylov_block::krylov_block;
    pub(super) use rayon::prelude::*;
    pub(super) use rest_libcint::prelude::*;
    pub(super) use rstsr::prelude::*;
    pub(super) use std::collections::HashMap;
}
