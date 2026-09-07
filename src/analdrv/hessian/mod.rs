//! Hessian computation for REST analytical derivative module.
//!
//! This module contains all hessian-related functionalities:
//! - trait definitions for hessian components ([`trait_rhess`], [`trait_uhess`], [`trait_util`]);
//! - component hessian implementations ([`hcore`], [`nuc_repl`], [`ovlp`], and integral handling
//!   [`cint_handling`]);
//! - total hessian drivers for restricted and unrestricted SCF ([`rscf`], [`uscf`]), which also
//!   contain the CP-SCF response machinery;
//! - total hessian interface to REST ([`rscf_interface`], [`uscf_interface`]).
//!
//! Note that the CP-SCF parts (the response methods in [`rscf`] and [`uscf`]) are currently
//! embedded in the total hessian drivers; they may be decoupled into a separate module in future.

// trait definitions
pub mod trait_rhess;
pub mod trait_uhess;

// core hess implementations
pub mod hcore;
pub mod nuc_repl;

// overlap hess implementations
pub mod ovlp;

// integral handling for hess
pub mod cint_handling;

// total hess implementations
pub mod rscf;
pub mod uscf;

// total hess interface to REST
pub mod rscf_interface;
pub mod uscf_interface;

use crate::analdrv::config::AnalDrvConfig;
use crate::analdrv::vibration::vib::{GauThermoInfo, VibInfo};
use crate::scf_io::{SCFType, SCF};

pub fn hess_interface(scf_data: &SCF, analdrv_ctrl: &AnalDrvConfig) -> (Vec<f64>, VibInfo, Option<GauThermoInfo>) {
    use crate::analdrv::hessian::rscf_interface::rscf_hess_interface;
    use crate::analdrv::hessian::uscf_interface::uscf_hess_interface;

    eprintln!("[WARN] You are using analdrv module, which is still under development.");
    eprintln!("[WARN] Keywords of analdrv will be updated in future versions.");

    // simple guard, but currently many methods (solvent is not supported)
    if scf_data.mol.xc_data.is_fifth_dfa() {
        panic!("Normal modes calculation is currently not available for post-SCF methods.");
    }

    match scf_data.scftype {
        SCFType::RHF => rscf_hess_interface(&scf_data, analdrv_ctrl),
        SCFType::UHF => uscf_hess_interface(&scf_data, analdrv_ctrl),
        _ => unimplemented!("Normal modes calculation is only implemented for RHF and UHF SCF types."),
    }
}
