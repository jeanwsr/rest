use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TDDFTParameters {
    pub tddft_method: String,       // "tda" or "lr" (full linear response)
    pub tddft_spin: String,         // "singlet" or "triplet"
    pub nroots: usize,              // number of excitation energies to compute
    pub davidson_tol: f64,          // Davidson convergence: ||r|| < sqrt(tol), |de| < tol
    pub davidson_max_iter: usize,   // maximum Davidson iterations
    pub davidson_max_subspace: usize, // maximum subspace dimension multiplier
    // Response (frequency-domain) TDDFT controls
    pub response_tddft: bool,         // enable response TDDFT calculation
    pub response_tddft_solver: String, // "pople", "gmres", "klopper", or "dense"
    pub response_tddft_tol: f64,      // iterative solver convergence tolerance
    pub response_tddft_max_iter: usize, // maximum iterations for iterative solver
    pub external_field_freq: f64,   // external field frequency ω (Hartree)
    pub lifetime_gamma: f64,        // lifetime broadening γ (Hartree)
    // Response TDDFT grid sampling parameters (for Polarized_Density_Grids.txt export)
    pub response_tddft_x_start: f64,
    pub response_tddft_x_end: f64,
    pub response_tddft_x_points: usize,
    pub response_tddft_y_start: f64,
    pub response_tddft_y_end: f64,
    pub response_tddft_y_points: usize,
    pub response_tddft_z_start: f64,
    pub response_tddft_z_end: f64,
    pub response_tddft_z_points: usize,
    pub response_tddft_grids: Vec<[f64; 3]>,
    // FEAST solver controls
    pub tddft_feast_solver: bool,
    pub tddft_feast_eigenrange_min: f64,
    pub tddft_feast_eigenrange_max: f64,
    pub tddft_feast_m_expected: usize,
    pub tddft_feast_max_iter: usize,
    pub tddft_feast_tol: f64,
    pub tddft_feast_gmres_restart: usize,
    pub tddft_feast_gmres_max_iter: usize,
    pub tddft_feast_cg_max_iter: usize,
    pub tddft_feast_cg_tol: f64,
    pub tddft_feast_init_guess_type: String,
    pub tddft_feast_gaussian_width_factor: f64,
    // Use optimized (rayon-parallel) fxc_matvec kernel
    pub tddft_use_optimized_fxc: bool,
    // Virtual orbital energy cutoff (Hartree); orbitals with KS eigenvalue
    // above this are excluded from the TDDFT excitation space. Default 1e6
    // (effectively no cutoff).
    pub tddft_cutoff_energy: f64,
    /// If true, export TDDFT results to rest_pysoc_export.json for PySOC.
    /// Requires tddft_spin = "both". Both Cartesian and spheric orbital
    /// basis sets are supported (the PySOC export is always written in
    /// Cartesian format; REST transforms spheric MOs when necessary).
    pub pysoc: bool,
}

/// Default Davidson subspace size for the TDDFT eigen-solvers.
///
/// The full-LR Davidson needs a subspace large enough to converge before the
/// first restart (each iteration adds ~nroots residual vectors); with the
/// previous default of 8 (effective ~4*nroots) the subspace collapsed to an
/// empty set and the LR solver failed for all functionals. 60 converges the
/// full-LR Davidson in ~8 iterations for typical valence-excitation problems.
pub const DEFAULT_DAVIDSON_MAX_SUBSPACE: usize = 60;

impl Default for TDDFTParameters {
    fn default() -> Self {
        TDDFTParameters {
            tddft_method: String::from("lr"),
            tddft_spin: String::from("singlet"),
            nroots: 6,
            davidson_tol: 1.0e-10,
            davidson_max_iter: 50,
            // The full-LR Davidson needs a subspace large enough to converge
            // before the first restart (each iteration adds ~nroots vectors);
            // with the previous default of 8 (effective ~4*nroots) the solver
            // collapsed to an empty subspace and failed for all functionals.
            davidson_max_subspace: 60,
            response_tddft: false,
            response_tddft_solver: String::from("klopper"),
            response_tddft_tol: 1.0e-6,
            response_tddft_max_iter: 200,
            external_field_freq: 0.5,
            lifetime_gamma: 0.001,
            response_tddft_x_start: 0.0,
            response_tddft_x_end: 1.0,
            response_tddft_x_points: 2,
            response_tddft_y_start: 0.0,
            response_tddft_y_end: 1.0,
            response_tddft_y_points: 2,
            response_tddft_z_start: 0.0,
            response_tddft_z_end: 1.0,
            response_tddft_z_points: 2,
            response_tddft_grids: Vec::new(),
            tddft_feast_solver: false,
            tddft_feast_eigenrange_min: 0.0,
            tddft_feast_eigenrange_max: 0.5,
            tddft_feast_m_expected: 20,
            tddft_feast_max_iter: 30,
            tddft_feast_tol: 1.0e-8,
            tddft_feast_gmres_restart: 200,
            tddft_feast_gmres_max_iter: 500,
            tddft_feast_cg_max_iter: 100,
            tddft_feast_cg_tol: 1.0e-8,
            tddft_feast_init_guess_type: String::from("random"),
            tddft_feast_gaussian_width_factor: 0.5,
            tddft_use_optimized_fxc: true,
            tddft_cutoff_energy: 1.0e6,
            pysoc: false,
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
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1.0e-10),
                _ => 1.0e-10,
            };
            p.davidson_max_iter = match tmp_ctrl.get("davidson_max_iter").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(50) as usize,
                _ => 50,
            };
            p.davidson_max_subspace = match tmp_ctrl.get("davidson_max_subspace").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(DEFAULT_DAVIDSON_MAX_SUBSPACE as u64) as usize,
                _ => DEFAULT_DAVIDSON_MAX_SUBSPACE,
            };
            // Response TDDFT parameters
            p.response_tddft = match tmp_ctrl.get("response_tddft").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(b) => *b,
                _ => false,
            };
            p.response_tddft_solver = match tmp_ctrl.get("response_tddft_solver").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.to_lowercase(),
                _ => String::from("klopper"),
            };
            p.response_tddft_tol = match tmp_ctrl.get("response_tddft_tol").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1.0e-6),
                _ => 1.0e-6,
            };
            p.response_tddft_max_iter = match tmp_ctrl.get("response_tddft_max_iter").unwrap_or(&serde_json::Value::Null) {
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
            // Response TDDFT grid sampling parameters
            p.response_tddft_x_start = match tmp_ctrl.get("response_tddft_x_start").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
                _ => 0.0,
            };
            p.response_tddft_x_end = match tmp_ctrl.get("response_tddft_x_end").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1.0),
                _ => 1.0,
            };
            p.response_tddft_x_points = match tmp_ctrl.get("response_tddft_x_points").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(2) as usize,
                _ => 2,
            };
            p.response_tddft_y_start = match tmp_ctrl.get("response_tddft_y_start").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
                _ => 0.0,
            };
            p.response_tddft_y_end = match tmp_ctrl.get("response_tddft_y_end").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1.0),
                _ => 1.0,
            };
            p.response_tddft_y_points = match tmp_ctrl.get("response_tddft_y_points").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(2) as usize,
                _ => 2,
            };
            p.response_tddft_z_start = match tmp_ctrl.get("response_tddft_z_start").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
                _ => 0.0,
            };
            p.response_tddft_z_end = match tmp_ctrl.get("response_tddft_z_end").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1.0),
                _ => 1.0,
            };
            p.response_tddft_z_points = match tmp_ctrl.get("response_tddft_z_points").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(2) as usize,
                _ => 2,
            };
            // Generate grids: OUTER LOOP X, MIDDLE LOOP Y, INNER LOOP Z
            let x_step = if p.response_tddft_x_points > 1 {
                (p.response_tddft_x_end - p.response_tddft_x_start) / (p.response_tddft_x_points - 1) as f64
            } else { 0.0 };
            let y_step = if p.response_tddft_y_points > 1 {
                (p.response_tddft_y_end - p.response_tddft_y_start) / (p.response_tddft_y_points - 1) as f64
            } else { 0.0 };
            let z_step = if p.response_tddft_z_points > 1 {
                (p.response_tddft_z_end - p.response_tddft_z_start) / (p.response_tddft_z_points - 1) as f64
            } else { 0.0 };
            let mut grids = Vec::with_capacity(p.response_tddft_x_points * p.response_tddft_y_points * p.response_tddft_z_points);
            for ix in 0..p.response_tddft_x_points {
                let x = p.response_tddft_x_start + ix as f64 * x_step;
                for iy in 0..p.response_tddft_y_points {
                    let y = p.response_tddft_y_start + iy as f64 * y_step;
                    for iz in 0..p.response_tddft_z_points {
                        let z = p.response_tddft_z_start + iz as f64 * z_step;
                        grids.push([x, y, z]);
                    }
                }
            }
            p.response_tddft_grids = grids;
            // FEAST solver parameters
            p.tddft_feast_solver = match tmp_ctrl.get("tddft_feast_solver").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(b) => *b,
                _ => false,
            };
            p.tddft_feast_eigenrange_min = match tmp_ctrl.get("tddft_feast_eigenrange_min").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
                _ => 0.0,
            };
            p.tddft_feast_eigenrange_max = match tmp_ctrl.get("tddft_feast_eigenrange_max").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.5),
                _ => 0.5,
            };
            p.tddft_feast_m_expected = match tmp_ctrl.get("tddft_feast_m_expected").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(20) as usize,
                _ => 20,
            };
            p.tddft_feast_max_iter = match tmp_ctrl.get("tddft_feast_max_iter").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(30) as usize,
                _ => 30,
            };
            p.tddft_feast_tol = match tmp_ctrl.get("tddft_feast_tol").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1.0e-8),
                _ => 1.0e-8,
            };
            p.tddft_feast_gmres_restart = match tmp_ctrl.get("tddft_feast_gmres_restart").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(200) as usize,
                _ => 200,
            };
            p.tddft_feast_gmres_max_iter = match tmp_ctrl.get("tddft_feast_gmres_max_iter").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(500) as usize,
                _ => 500,
            };
            p.tddft_feast_cg_max_iter = match tmp_ctrl.get("tddft_feast_cg_max_iter").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(100) as usize,
                _ => 100,
            };
            p.tddft_feast_cg_tol = match tmp_ctrl.get("tddft_feast_cg_tol").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1.0e-8),
                _ => 1.0e-8,
            };
            p.tddft_feast_init_guess_type = match tmp_ctrl.get("tddft_feast_init_guess_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => String::from("random"),
            };
            p.tddft_feast_gaussian_width_factor = match tmp_ctrl.get("tddft_feast_gaussian_width_factor").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.5),
                _ => 0.5,
            };
            // Optimized fxc_matvec kernel
            p.tddft_use_optimized_fxc = match tmp_ctrl.get("tddft_use_optimized_fxc").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(b) => *b,
                _ => true,
            };
            // Virtual orbital energy cutoff
            p.tddft_cutoff_energy = match tmp_ctrl.get("tddft_cutoff_energy").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1.0e6),
                _ => 1.0e6,
            };
            p.pysoc = match tmp_ctrl.get("pysoc").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(b) => *b,
                _ => false,
            };
            Ok(Some(p))
        },
        _ => Ok(None),
    }
}
