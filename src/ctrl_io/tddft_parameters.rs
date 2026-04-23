use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TDDFTParameters {
    pub tddft_method: String,       // "tda" or "lr" (full linear response)
    pub tddft_spin: String,         // "singlet" or "triplet"
    pub nroots: usize,              // number of excitation energies to compute
    pub davidson_tol: f64,          // Davidson convergence threshold
    pub davidson_max_iter: usize,   // maximum Davidson iterations
    pub davidson_max_subspace: usize, // maximum subspace dimension multiplier
}

impl Default for TDDFTParameters {
    fn default() -> Self {
        TDDFTParameters {
            tddft_method: String::from("lr"),
            tddft_spin: String::from("singlet"),
            nroots: 6,
            davidson_tol: 1.0e-6,
            davidson_max_iter: 50,
            davidson_max_subspace: 8,
        }
    }
}

pub fn parse_tddft_keywords(tmp_keys: &serde_json::Value) -> anyhow::Result<Option<TDDFTParameters>> {
    match tmp_keys.get("tddft").unwrap_or(&serde_json::Value::Null) {
        serde_json::Value::Object(tmp_ctrl) => {
            let mut p = TDDFTParameters::default();
            p.tddft_method = match tmp_ctrl.get("tddft_method").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.to_lowercase(),
                _ => String::from("lr"),
            };
            p.tddft_spin = match tmp_ctrl.get("tddft_spin").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.to_lowercase(),
                _ => String::from("singlet"),
            };
            p.nroots = match tmp_ctrl.get("nroots").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(6) as usize,
                _ => 6,
            };
            p.davidson_tol = match tmp_ctrl.get("davidson_tol").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1.0e-6),
                _ => 1.0e-6,
            };
            p.davidson_max_iter = match tmp_ctrl.get("davidson_max_iter").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(50) as usize,
                _ => 50,
            };
            p.davidson_max_subspace = match tmp_ctrl.get("davidson_max_subspace").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(8) as usize,
                _ => 8,
            };
            Ok(Some(p))
        },
        _ => Ok(None),
    }
}
