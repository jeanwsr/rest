use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPHFParameters {
    /// Solver type: "krylov" or "dense" (default "krylov")
    pub solver: String,
    /// Calculation task: "response", "hessian", "ck7", "ck8", "cphf_hess"
    #[serde(default = "default_calculation")]
    pub calculation: String,
    /// Krylov solver maximum iterations (default 50)
    pub krylov_max_cycle: usize,
    /// Krylov convergence tolerance (default 1e-12)
    pub krylov_tol: f64,
    /// Verbose output level (0=silent, 1=normal, 2=debug)
    pub verbose: usize,
    /// Output path for the Hessian matrix txt file
    #[serde(default = "default_hessian_matrix_path")]
    pub hessian_matrix_path: String,
    /// Output path for the vibrational eigenmodes txt file
    #[serde(default = "default_eigenmodes_path")]
    pub eigenmodes_path: String,
}

fn default_calculation() -> String { String::from("response") }
fn default_hessian_matrix_path() -> String { String::from("./HessianMatrix.txt") }
fn default_eigenmodes_path() -> String { String::from("./EigenModes.txt") }

impl Default for CPHFParameters {
    fn default() -> Self {
        CPHFParameters {
            solver: String::from("krylov"),
            calculation: String::from("response"),
            krylov_max_cycle: 50,
            krylov_tol: 1.0e-12,
            verbose: 1,
            hessian_matrix_path: default_hessian_matrix_path(),
            eigenmodes_path: default_eigenmodes_path(),
        }
    }
}

pub fn parse_cphf_keywords(tmp_keys: &serde_json::Value) -> anyhow::Result<Option<CPHFParameters>> {
    let cphf_section = tmp_keys.get("cphf").or_else(|| {
        tmp_keys.get("ctrl").and_then(|c| c.get("cphf"))
    }).unwrap_or(&serde_json::Value::Null);
    match cphf_section {
        serde_json::Value::Object(tmp_ctrl) => {
            let mut p = CPHFParameters::default();
            p.solver = match tmp_ctrl.get("solver").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.to_lowercase(),
                _ => String::from("krylov"),
            };
            p.calculation = match tmp_ctrl.get("calculation").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.to_lowercase(),
                _ => String::from("response"),
            };
            p.krylov_max_cycle = match tmp_ctrl.get("krylov_max_cycle").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(50) as usize,
                _ => 50,
            };
            p.krylov_tol = match tmp_ctrl.get("krylov_tol").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1.0e-12),
                _ => 1.0e-12,
            };
            p.verbose = match tmp_ctrl.get("verbose").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(1) as usize,
                _ => 1,
            };
            p.hessian_matrix_path = match tmp_ctrl.get("hessian_matrix_path").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => default_hessian_matrix_path(),
            };
            p.eigenmodes_path = match tmp_ctrl.get("eigenmodes_path").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => default_eigenmodes_path(),
            };
            Ok(Some(p))
        },
        _ => Ok(None),
    }
}
