use serde::{Deserialize, Serialize};
use serde_inline_default::serde_inline_default;

#[serde_inline_default]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HessianParameters {
    /// Solver type: "krylov" or "dense" (default "krylov")
    #[serde(default = "default_solver")]
    pub solver: String,
    /// If true, also compute and save vibrational frequencies after the Hessian.
    #[serde(default)]
    pub frequencies: bool,
    /// Krylov solver maximum iterations (default 50)
    #[serde_inline_default(50usize)]
    pub krylov_max_cycle: usize,
    /// Krylov convergence tolerance (default 1e-12)
    #[serde_inline_default(1e-9)]
    pub krylov_tol: f64,
    /// Krylov residual inflation factor (default 1000.0)
    #[serde_inline_default(1000.0)]
    pub krylov_tol_inflation: f64,
    /// Krylov linear dependence threshold (default 1e-15)
    #[serde_inline_default(1e-15)]
    pub krylov_lindep: f64,
    /// Verbose output level (0=silent, 1=normal, 2=debug) (default 1)
    #[serde_inline_default(1usize)]
    pub verbose: usize,
    /// Output path for the Hessian matrix txt file (default "./HessianMatrix.txt")
    #[serde(default = "default_hessian_matrix_path")]
    pub hessian_matrix_path: String,
    /// Output path for the vibrational eigenmodes txt file (default "./EigenModes.txt")
    #[serde(default = "default_eigenmodes_path")]
    pub eigenmodes_path: String,
}

fn default_solver() -> String { String::from("krylov") }
fn default_hessian_matrix_path() -> String { String::from("./HessianMatrix.txt") }
fn default_eigenmodes_path() -> String { String::from("./EigenModes.txt") }

impl Default for HessianParameters {
    fn default() -> Self {
        HessianParameters {
            solver: default_solver(),
            frequencies: false,
            krylov_max_cycle: 50,
            krylov_tol: 1e-9,
            krylov_tol_inflation: 1000.0,
            krylov_lindep: 1e-15,
            verbose: 1,
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
            p.solver = match o.get("solver").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.to_lowercase(),
                _ => String::from("krylov"),
            };
            p.frequencies = match o.get("frequencies").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(b) => *b,
                _ => false,
            };
            p.krylov_max_cycle = match o.get("krylov_max_cycle").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(50) as usize,
                _ => 50,
            };
            p.krylov_tol = match o.get("krylov_tol").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1.0e-12),
                _ => 1.0e-9,
            };
            p.krylov_tol_inflation = match o.get("krylov_tol_inflation").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1000.0),
                _ => 1000.0,
            };
            p.krylov_lindep = match o.get("krylov_lindep").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1e-15),
                _ => 1e-15,
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
