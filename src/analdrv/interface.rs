//! Interface to REST other programs.

use crate::analdrv::config::{AnalDrvConfig, AnalDrvTask};
use crate::analdrv::hessian::hess_interface;
use crate::scf_io::SCF;
use crate::thermo::ThermoResult;

pub struct AnaldrvOutput {
    pub frequencies_cm: Vec<f64>,
    /// TR/V classification per mode: `"TR"` (translation/rotation), `"V"` (vibration), or `"-"` (near-zero force constant).
    pub modes_trv: Vec<&'static str>,
    /// Thermochemistry result, present if the `[thermo]` section is given in the control input.
    pub thermo: Option<ThermoResult>,
}

pub fn analdrv_interface(scf_data: &SCF, tasks: &[AnalDrvTask], config: &AnalDrvConfig) -> Option<AnaldrvOutput> {
    let mut output: Option<AnaldrvOutput> = None;
    for task in tasks {
        match task {
            AnalDrvTask::Hessian => {
                let (_, vib, _) = hess_interface(scf_data, config);
                let freqs: Vec<f64> = vib.omega.iter().zip(vib.imag.iter())
                    .map(|(&f, &imag)| if imag { -f } else { f })
                    .collect();

                // --- thermo analysis (shermo-style) --- //

                // this is activated by using `[thermo]` section in control input.
                let thermo = scf_data.mol.ctrl.thermo.as_ref().map(|thermo_cfg| {
                    let mut time_mark = crate::utilities::TimeRecords::new();
                    crate::thermo::run_thermochemistry(scf_data, &freqs, thermo_cfg, &mut time_mark)
                }).flatten();

                output = Some(AnaldrvOutput { frequencies_cm: freqs, modes_trv: vib.trv.clone(), thermo });
            }
        };
    }
    output
}
