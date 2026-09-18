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
use crate::analdrv::response::rresp_interface::RRespSCF;
use crate::scf_io::{SCFType, SCF};
use crate::thermo::ThermoResult;

/// Output of the analytical Hessian task, with the derived vibrational/thermochemical analysis.
#[derive(Debug, Clone, Default)]
pub struct HessOutput {
    /// Flattened analytical Hessian matrix `[t, s, A, B]` (component xyz first, atom then).
    ///
    /// Consumed by the geometry-optimization initial guess; not carried into the results JSON.
    pub hessian: Vec<f64>,
    /// Vibrational frequencies in cm^-1; imaginary modes are reported as negative values.
    pub frequencies_cm: Vec<f64>,
    /// TR/V classification per mode: `"TR"` (translation/rotation), `"V"` (vibration), or `"-"`
    /// (near-zero force constant).
    pub modes_trv: Vec<&'static str>,
    /// Thermochemistry result, present if the `[thermo]` section is given in the control input.
    pub thermo: Option<ThermoResult>,
}

pub fn hess_interface<'a>(
    scf_data: &'a SCF,
    config: &AnalDrvConfig,
    resp_objs: Option<&mut RRespSCF<'a>>,
) -> HessOutput {
    use crate::analdrv::hessian::rscf_interface::rscf_hess_interface;
    use crate::analdrv::hessian::uscf_interface::uscf_hess_interface;

    eprintln!("[WARN] You are using analdrv module, which is still under development.");
    eprintln!("[WARN] Keywords of analdrv will be updated in future versions.");

    // simple guard, but currently many methods (solvent is not supported)
    if scf_data.mol.xc_data.is_fifth_dfa() {
        panic!("Normal modes calculation is currently not available for post-SCF methods.");
    }

    // The RHF hessian consumes the shared response object built by the caller; the UHF hessian
    // builds and owns its own (U-side) response machinery internally, so it takes the response
    // settings explicitly.
    let (hessian, vib, _) = match scf_data.scftype {
        SCFType::RHF => rscf_hess_interface(
            scf_data,
            &config.nucgrad,
            resp_objs.expect("internal error: the RHF hessian requires the shared response object"),
        ),
        SCFType::UHF => uscf_hess_interface(scf_data, &config.nucgrad, &config.resp),
        _ => unimplemented!("Normal modes calculation is only implemented for RHF and UHF SCF types."),
    };

    // vibrational frequencies; imaginary modes carry a negative sign
    let frequencies_cm: Vec<f64> =
        vib.omega.iter().zip(vib.imag.iter()).map(|(&f, &imag)| if imag { -f } else { f }).collect();

    // --- thermo analysis (shermo-style) --- //

    // this is activated by using `[thermo]` section in control input.
    let thermo = scf_data
        .mol
        .ctrl
        .thermo
        .as_ref()
        .map(|thermo_cfg| {
            let mut time_mark = crate::utilities::TimeRecords::new();
            crate::thermo::run_thermochemistry(scf_data, &frequencies_cm, thermo_cfg, &mut time_mark)
        })
        .flatten();

    HessOutput { hessian, frequencies_cm, modes_trv: vib.trv.clone(), thermo }
}
