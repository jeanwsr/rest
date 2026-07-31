use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HessianParameters {
    /// If true, also compute and save vibrational frequencies after the Hessian.
    #[serde(default)]
    pub frequencies: bool,
    /// Verbose output level (0=silent, 1=normal, 2=debug) (default 1)
    #[serde(default = "default_verbose")]
    pub verbose: usize,
    /// Output path for the Hessian matrix txt file (default "./HessianMatrix.txt")
    #[serde(default = "default_hessian_matrix_path")]
    pub hessian_matrix_path: String,
    /// Output path for the vibrational eigenmodes txt file (default "./EigenModes.txt")
    #[serde(default = "default_eigenmodes_path")]
    pub eigenmodes_path: String,
}

fn default_verbose() -> usize { 1 }
fn default_hessian_matrix_path() -> String { String::from("./HessianMatrix.txt") }
fn default_eigenmodes_path() -> String { String::from("./EigenModes.txt") }

impl Default for HessianParameters {
    fn default() -> Self {
        HessianParameters {
            frequencies: false,
            verbose: default_verbose(),
            hessian_matrix_path: default_hessian_matrix_path(),
            eigenmodes_path: default_eigenmodes_path(),
        }
    }
}

pub fn parse_hessian_keywords(tmp_keys: &serde_json::Value) -> anyhow::Result<Option<HessianParameters>> {
    let section = tmp_keys.get("hessian").or_else(|| {
        tmp_keys.get("ctrl").and_then(|c| c.get("hessian"))
    }).unwrap_or(&serde_json::Value::Null);
    match section {
        serde_json::Value::Object(o) => {
            let mut p = HessianParameters::default();
            p.frequencies = match o.get("frequencies").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(b) => *b,
                _ => false,
            };
            p.verbose = match o.get("verbose").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(1) as usize,
                _ => 1,
            };
            p.hessian_matrix_path = match o.get("hessian_matrix_path").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => default_hessian_matrix_path(),
            };
            p.eigenmodes_path = match o.get("eigenmodes_path").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => default_eigenmodes_path(),
            };
            Ok(Some(p))
        },
        _ => Ok(None),
    }
}
