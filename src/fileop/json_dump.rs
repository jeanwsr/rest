use crate::scf_io::SCF;
use crate::constants::{AU2DEBYE, EV};
use serde_json::json;
use std::fs::File;
use std::io::Write;

pub fn dump_json(scf_data: &SCF, extra: &std::collections::HashMap<String, serde_json::Value>) {
    let mut result = json!({});

    result["scf_energy"] = json!(scf_data.scf_energy);
    result["nuc_energy"] = json!(scf_data.nuc_energy);

    result["total_energy"] = json!(scf_data.energies.get("total_energy")
        .map(|v| v[0])
        .unwrap_or(scf_data.scf_energy));

    result["energies"] = serde_json::to_value(&scf_data.energies).unwrap_or(json!({}));

    for (key, val) in extra {
        result[key] = json!(val);
    }

    if !scf_data.eigenvalues[0].is_empty() {
        let homo = scf_data.homo[0];
        let lumo = scf_data.lumo[0];
        if homo < scf_data.eigenvalues[0].len() && lumo < scf_data.eigenvalues[0].len() {
            let gap = (scf_data.eigenvalues[0][lumo] - scf_data.eigenvalues[0][homo]) * EV;
            result["hoco_luco_gap_ev"] = json!(gap);
        }
    }

    let dp_au = crate::post_scf_analysis::evaluate_dipole_moment(scf_data, None);
    let dp_debye: Vec<f64> = dp_au.iter().map(|x| x * AU2DEBYE).collect();
    result["dipole_debye"] = json!(dp_debye);

    if !scf_data.gwqp.0.is_empty() {
        result["gw"] = json!({
            "homo_qp": scf_data.gwqp.0,
            "lumo_qp": scf_data.gwqp.1,
        });
    }

    let json_str = serde_json::to_string_pretty(&result).unwrap_or_default();
    let base = scf_data.mol.ctrl.outname.as_deref().unwrap_or("rest_results");
    let filename = format!("{}.json", base);
    if let Ok(mut file) = File::create(filename) {
        let _ = file.write_all(json_str.as_bytes());
    }
}
