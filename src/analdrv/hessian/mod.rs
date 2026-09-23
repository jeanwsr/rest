//! Hessian computation for REST analytical derivative module.
//!
//! This module contains all hessian-related functionalities:
//! - trait definitions for hessian components ([`trait_rhess`], [`trait_uhess`], [`trait_util`]);
//! - component hessian implementations ([`hcore`], [`nuc_repl`], [`ovlp`], and integral handling
//!   [`cint_handling`]);
//! - total hessian drivers for restricted and unrestricted SCF ([`rscf`], [`uscf`]);
//! - total hessian interface to REST ([`rscf_interface`], [`uscf_interface`]).
//!
//! The CP-SCF response machinery lives in the separate
//! [`response`](crate::analdrv::response) module; the hessian drivers consume its response
//! objects through the `resp` field.

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
use crate::analdrv::response::RespSCF;
use crate::scf_io::{SCFType, SCF};
use crate::thermo::ThermoResult;
use crate::utilities::rstsr_util::Tsr;

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

/// Print the hessian matrix and the driver timing report (shared by the restricted and
/// unrestricted interfaces), at `print_level >= 2`.
///
/// The matrix is printed in `[tA, sB]` format (component xyz first, atom then), 6 columns at a
/// time with an index header; the timing entries follow in call order.
fn print_hessian_and_timing(de_hess: &Tsr, timing: &[(String, f64)], print_level: usize) {
    if print_level < 2 {
        return;
    }
    println!("=== HESSIAN ===");
    println!("Print hessian in [tA, sB] format (component xyz first, atom then)");
    println!("");
    // print hessian matrix [t, s, A, B] -> [tA, sB]
    let natm = de_hess.shape()[3];
    let hess_mat = de_hess.transpose([0, 2, 1, 3]).into_shape((3 * natm, 3 * natm));
    // print 6 columns at a time, with index header
    for j in (0..3 * natm).step_by(6) {
        let j_end = (j + 6).min(3 * natm);
        let col_header = " ".repeat(6) + &(j..j_end).map(|i| format!("{:>12}", i)).collect::<String>();
        println!("{}", col_header);
        for i in 0..3 * natm {
            let row_str = format!("{i:>4}  ")
                + &(j..j_end).map(|j| format!("{:12.6}", hess_mat[[i, j]])).collect::<String>();
            println!("{}", row_str);
        }
        println!("");
    }

    // print timing information
    println!("Timing in Hessian calculation:");
    for (key, value) in timing.iter() {
        println!("    {:60}: {:10.6} seconds", key, value);
    }
}

pub fn hess_interface<'a>(
    scf_data: &'a SCF,
    config: &AnalDrvConfig,
    resp_objs: Option<&mut RespSCF<'a>>,
) -> HessOutput {
    use crate::analdrv::hessian::rscf_interface::rscf_hess_interface;
    use crate::analdrv::hessian::uscf_interface::uscf_hess_interface;

    eprintln!("[WARN] You are using analdrv module, which is still under development.");
    eprintln!("[WARN] Keywords of analdrv will be updated in future versions.");

    // simple guard, but currently many methods (solvent is not supported)
    if scf_data.mol.xc_data.is_fifth_dfa() {
        panic!("Normal modes calculation is currently not available for post-SCF methods.");
    }

    // The hessian consumes the shared response object built by the caller (the `RespSCF`
    // carrier, holding the restricted/unrestricted variant per the SCF type).
    let (hessian, vib, _) = match scf_data.scftype {
        SCFType::RHF => {
            let ctx = "internal error: the RHF hessian requires the shared response object";
            rscf_hess_interface(scf_data, &config.nucgrad, resp_objs.expect(ctx).expect_r_mut(ctx))
        },
        SCFType::UHF => {
            let ctx = "internal error: the UHF hessian requires the shared response object";
            uscf_hess_interface(scf_data, &config.nucgrad, resp_objs.expect(ctx).expect_u_mut(ctx))
        },
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
