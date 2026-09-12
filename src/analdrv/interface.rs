//! Interface to REST other programs.

use crate::analdrv::config::{AnalDrvConfig, AnalDrvTask, MultipoleRdm1Relax};
use crate::analdrv::hessian::hess_interface;
use crate::analdrv::multipole::interface::{multipole_interface, MultipoleOutput};
use crate::analdrv::response::rresp_interface::rscf_resp_interface;
use crate::scf_io::{SCFType, SCF};
use crate::thermo::ThermoResult;

use serde::Serialize;
use std::collections::HashMap;

/// Comprehensive output of the analdrv module: one entry per evaluated task.
///
/// The struct is serialized (auto-serde) into the `"analdrv"` key of the REST results JSON by
/// [`analdrv_json_interface`], so the JSON layout is exactly the serde layout of this struct:
/// the hessian frequencies/TRV tags keep their legacy flat position when the hessian task ran
/// (they are absent otherwise), and each other task nests under its own key. The thermochemistry
/// result keeps its legacy top-level `"thermo"` JSON key, and is therefore skipped in this
/// serialization.
#[derive(Debug, Clone, Default, Serialize)]
pub struct AnaldrvOutput {
    /// (Hessian task) Vibrational frequencies in cm^-1, imaginary modes as negative values.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequencies_cm: Option<Vec<f64>>,
    /// (Hessian task) TR/V classification per mode: `"TR"` (translation/rotation), `"V"`
    /// (vibration), or `"-"` (near-zero force constant).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modes_trv: Option<Vec<&'static str>>,
    /// Thermochemistry result, present if the `[thermo]` section is given in the control input
    /// and the hessian task ran. Serialized separately as the legacy top-level `"thermo"` key.
    #[serde(skip)]
    pub thermo: Option<ThermoResult>,
    /// (Multipole task) Electric multipole moments in atomic units, see [`MultipoleOutput`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multipole: Option<MultipoleOutput>,
}

/// Run the analdrv tasks on the (converged, post-SCF-evaluated) SCF data.
///
/// The results-JSON expansion of the returned [`AnaldrvOutput`] is provided separately by
/// [`analdrv_json_interface`], so this function is a pure task evaluation (computation and
/// stdout only), and the main driver holds no task-specific knowledge.
///
/// # Parameters
///
/// - `scf_data` : the converged SCF data. Taken by `&mut` only for the DFT grids regeneration
///   required by the post-SCF multipole increments (see below); the task evaluation itself only
///   reads.
/// - `tasks` : the task list from the `analdrv_tasks` control keyword.
/// - `config` : the `[analdrv]` section configuration.
///
/// # Shared objects
///
/// The RHF-level response object ([`crate::analdrv::response::rresp_interface::RRespSCF`]) is
/// built once and shared (sequentially, by `&mut`) between the task arms: the RHF hessian always
/// needs it, and the multipole task needs it for the relaxed DH increments. The UHF hessian
/// builds and owns its own (U-side) response machinery internally.
///
/// The multipole task on post-SCF (fifth-DFA) methods needs the DFT grids (for both the unrelaxed
/// and the relaxed DH increments), which `xdh_calculations` frees after the energy evaluation;
/// they are regenerated here if absent. This is grid generation only: the AO tabulation is
/// prepared by the respective consumers, not by the grid build.
pub fn analdrv_interface(
    scf_data: &mut SCF,
    tasks: &[AnalDrvTask],
    config: &AnalDrvConfig,
) -> AnaldrvOutput {
    let mut output = AnaldrvOutput::default();

    // --- preparation (the only phase taking scf_data mutably) --- //

    let has_multipole = tasks.iter().any(|task| matches!(task, AnalDrvTask::Multipole));
    let has_hessian = tasks.iter().any(|task| matches!(task, AnalDrvTask::Hessian));
    let is_fifth = scf_data.mol.xc_data.is_fifth_dfa();
    // the relaxed (Z-vector) DH multipole increments are the only multipole requirement on the
    // response object; the unrelaxed increments still need the DFT grids (regenerated below)
    let multipole_relaxed_dh =
        has_multipole && is_fifth && config.multipole.rdm1_relax == MultipoleRdm1Relax::Relaxed;
    if has_multipole && is_fifth && scf_data.grids.is_none() {
        scf_data.grids = Some(crate::dft::Grids::build(&mut scf_data.mol));
    }

    let need_resp = has_hessian || multipole_relaxed_dh;
    let mut resp_objs = if need_resp && matches!(scf_data.scftype, SCFType::RHF) {
        Some(rscf_resp_interface(scf_data, config))
    } else {
        None
    };

    // --- task evaluation --- //

    for task in tasks {
        match task {
            AnalDrvTask::Hessian => {
                let hess = hess_interface(scf_data, config, resp_objs.as_mut());
                output.frequencies_cm = Some(hess.frequencies_cm);
                output.modes_trv = Some(hess.modes_trv);
                output.thermo = hess.thermo;
            },
            AnalDrvTask::Multipole => {
                let resp = if multipole_relaxed_dh { resp_objs.as_mut() } else { None };
                output.multipole = Some(multipole_interface(scf_data, config, resp));
            },
        }
    }

    output
}

/// Build the results-JSON entries of the analdrv module from its [`AnaldrvOutput`].
///
/// Returns the map to be merged (plain insert semantics; same-named entries are overwritten)
/// into the results-JSON accumulator `json_extra` (see `crate::fileop::json_dump`): the
/// auto-serialization of `output` under the `"analdrv"` key, plus the thermochemistry result
/// under the legacy top-level `"thermo"` key when present.
pub fn analdrv_json_interface(output: &AnaldrvOutput) -> HashMap<String, serde_json::Value> {
    let mut json = HashMap::new();
    json.insert(
        "analdrv".to_string(),
        serde_json::to_value(output).unwrap_or(serde_json::Value::Null),
    );
    if let Some(thermo) = &output.thermo {
        json.insert(
            "thermo".to_string(),
            serde_json::to_value(thermo).unwrap_or(serde_json::Value::Null),
        );
    }
    json
}
