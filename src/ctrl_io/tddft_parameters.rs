use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TDDFTParameters {
    pub tddft_method: String,       // "tda" or "lr" (full linear response)
    pub tddft_spin: String,         // "singlet" or "triplet"
    pub nroots: usize,              // number of excitation energies to compute
    pub davidson_tol: f64,          // Davidson convergence threshold
    pub davidson_max_iter: usize,   // maximum Davidson iterations
    pub davidson_max_subspace: usize, // maximum subspace dimension multiplier
    // Damped (frequency-domain) TDDFT controls
    pub damped_tddft: bool,         // enable damped TDDFT response calculation
    pub damped_tddft_solver: String, // "pople", "gmres", "klopper", or "dense"
    pub damped_tddft_tol: f64,      // iterative solver convergence tolerance
    pub damped_tddft_max_iter: usize, // maximum iterations for iterative solver
    pub external_field_freq: f64,   // external field frequency ω (Hartree)
    pub lifetime_gamma: f64,        // lifetime broadening γ (Hartree)
    // Damped TDDFT grid sampling parameters (for Polarized_Density_Grids.txt export)
    pub damped_tddft_x_start: f64,
    pub damped_tddft_x_end: f64,
    pub damped_tddft_x_points: usize,
    pub damped_tddft_y_start: f64,
    pub damped_tddft_y_end: f64,
    pub damped_tddft_y_points: usize,
    pub damped_tddft_z_start: f64,
    pub damped_tddft_z_end: f64,
    pub damped_tddft_z_points: usize,
    pub damped_tddft_grids: Vec<[f64; 3]>,
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
            damped_tddft: false,
            damped_tddft_solver: String::from("klopper"),
            damped_tddft_tol: 1.0e-6,
            damped_tddft_max_iter: 200,
            external_field_freq: 0.5,
            lifetime_gamma: 0.001,
            damped_tddft_x_start: 0.0,
            damped_tddft_x_end: 1.0,
            damped_tddft_x_points: 2,
            damped_tddft_y_start: 0.0,
            damped_tddft_y_end: 1.0,
            damped_tddft_y_points: 2,
            damped_tddft_z_start: 0.0,
            damped_tddft_z_end: 1.0,
            damped_tddft_z_points: 2,
            damped_tddft_grids: Vec::new(),
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
            // Damped TDDFT parameters
            p.damped_tddft = match tmp_ctrl.get("damped_tddft").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(b) => *b,
                _ => false,
            };
            p.damped_tddft_solver = match tmp_ctrl.get("damped_tddft_solver").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.to_lowercase(),
                _ => String::from("klopper"),
            };
            p.damped_tddft_tol = match tmp_ctrl.get("damped_tddft_tol").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1.0e-6),
                _ => 1.0e-6,
            };
            p.damped_tddft_max_iter = match tmp_ctrl.get("damped_tddft_max_iter").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(200) as usize,
                _ => 200,
            };
            p.external_field_freq = match tmp_ctrl.get("external_field_freq").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.5),
                _ => 0.5,
            };
            p.lifetime_gamma = match tmp_ctrl.get("lifetime_gamma").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.001),
                _ => 0.001,
            };
            // Damped TDDFT grid sampling parameters
            p.damped_tddft_x_start = match tmp_ctrl.get("damped_tddft_x_start").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
                _ => 0.0,
            };
            p.damped_tddft_x_end = match tmp_ctrl.get("damped_tddft_x_end").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1.0),
                _ => 1.0,
            };
            p.damped_tddft_x_points = match tmp_ctrl.get("damped_tddft_x_points").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(2) as usize,
                _ => 2,
            };
            p.damped_tddft_y_start = match tmp_ctrl.get("damped_tddft_y_start").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
                _ => 0.0,
            };
            p.damped_tddft_y_end = match tmp_ctrl.get("damped_tddft_y_end").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1.0),
                _ => 1.0,
            };
            p.damped_tddft_y_points = match tmp_ctrl.get("damped_tddft_y_points").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(2) as usize,
                _ => 2,
            };
            p.damped_tddft_z_start = match tmp_ctrl.get("damped_tddft_z_start").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
                _ => 0.0,
            };
            p.damped_tddft_z_end = match tmp_ctrl.get("damped_tddft_z_end").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1.0),
                _ => 1.0,
            };
            p.damped_tddft_z_points = match tmp_ctrl.get("damped_tddft_z_points").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(2) as usize,
                _ => 2,
            };
            // Generate grids: OUTER LOOP X, MIDDLE LOOP Y, INNER LOOP Z
            let x_step = if p.damped_tddft_x_points > 1 {
                (p.damped_tddft_x_end - p.damped_tddft_x_start) / (p.damped_tddft_x_points - 1) as f64
            } else { 0.0 };
            let y_step = if p.damped_tddft_y_points > 1 {
                (p.damped_tddft_y_end - p.damped_tddft_y_start) / (p.damped_tddft_y_points - 1) as f64
            } else { 0.0 };
            let z_step = if p.damped_tddft_z_points > 1 {
                (p.damped_tddft_z_end - p.damped_tddft_z_start) / (p.damped_tddft_z_points - 1) as f64
            } else { 0.0 };
            let mut grids = Vec::with_capacity(p.damped_tddft_x_points * p.damped_tddft_y_points * p.damped_tddft_z_points);
            for ix in 0..p.damped_tddft_x_points {
                let x = p.damped_tddft_x_start + ix as f64 * x_step;
                for iy in 0..p.damped_tddft_y_points {
                    let y = p.damped_tddft_y_start + iy as f64 * y_step;
                    for iz in 0..p.damped_tddft_z_points {
                        let z = p.damped_tddft_z_start + iz as f64 * z_step;
                        grids.push([x, y, z]);
                    }
                }
            }
            p.damped_tddft_grids = grids;
            Ok(Some(p))
        },
        _ => Ok(None),
    }
}
