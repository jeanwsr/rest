use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPHFParameters {
    /// Solver type: "krylov" or "dense" (default "krylov")
    pub solver: String,
    /// Krylov solver maximum iterations (default 50)
    pub krylov_max_cycle: usize,
    /// Krylov convergence tolerance (default 1e-12)
    pub krylov_tol: f64,
    /// Verbose output level (0=silent, 1=normal, 2=debug)
    pub verbose: usize,
}

impl Default for CPHFParameters {
    fn default() -> Self {
        CPHFParameters {
            solver: String::from("krylov"),
            krylov_max_cycle: 50,
            krylov_tol: 1.0e-12,
            verbose: 1,
        }
    }
}

pub fn parse_cphf_keywords(tmp_keys: &serde_json::Value) -> anyhow::Result<Option<CPHFParameters>> {
    match tmp_keys.get("cphf").unwrap_or(&serde_json::Value::Null) {
        serde_json::Value::Object(tmp_ctrl) => {
            let mut p = CPHFParameters::default();
            p.solver = match tmp_ctrl.get("solver").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.to_lowercase(),
                _ => String::from("krylov"),
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
            Ok(Some(p))
        },
        _ => Ok(None),
    }
}
