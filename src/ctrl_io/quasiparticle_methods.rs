use pyo3::types::PyDict;
use pyo3::{Py, PyResult, Python};
use serde::{Deserialize,Serialize};
use crate::geom_io::GeomCell;
use crate::ctrl_io::InputKeywords;

#[derive(Debug,Clone,Serialize, Deserialize)]
pub struct QuasiParticle {
    pub gw_scheme:String,
    pub homo_lumo_gw_qp:bool,
    pub x_alpha:f64,
    pub save_qp:bool,
    pub bse_davidson_solver:bool,
    pub davidson_target_excitations:usize,
    pub davidson_converge_threshold:f64,
    pub davidson_maximum_subspace_size:usize,
    pub davidson_restart_dimensions:usize,
    pub davidson_max_iter:usize,
    pub davidson_add_dimensions:usize,
    pub bse_tda:bool,
    pub bse_spin:String,
    pub bse_cutoff_energy:f64,
    pub save_bse_terms:bool,
    pub obtain_vx_vc_terms:bool,
    pub obtain_pure_exchange:bool,
    pub obtain_ks_homos:bool,
    pub gw_linearize_shift:f64,
    pub gw_linearize_derivative_h:f64,
    pub renormalized_singles:bool,
    pub w_rs:bool,
    pub scgw:String,
    pub gw:bool,
    pub bse_exchange_rescaling:f64,
    pub save_bse_excitations:bool,
    pub evgw_rounds:usize,
    pub save_gw_homo_lumo_qp:bool,
    pub save_qp_path:String,
    pub save_first_excitation:bool,
    pub save_first_excitation_path:String,
    pub use_low_rank_contour:bool,
    pub low_rank_grid_type:String,   // "linear" or "quadratic" (power-law, denser near zero)
    pub nomega_chi_real:usize,
    pub low_rank_tolerance:f64,
    pub omega_chi_max:f64,          // real-axis freq grid max (Ha); 0=auto from de_max
    pub nomega_sigma:usize,         // number of sigma sampling points on each side (de_max scan)
    pub step_sigma:f64,             // spacing of sigma grid in Ha (de_max scan)
    pub fourier_self_energy:bool,
    pub fse_sin_coeff_path:String,
    pub fse_cos_coeff_path:String,
    pub hermite_self_energy:bool,
    pub hermite_coeff_path:String,
    pub parse_qp_path:String,
    pub bse_qp_polarization:bool,
    pub threshold:f64,
    pub external_field_freq:f64,
    pub lifetime_gamma:f64,
    pub gw_or_bse:String,
    pub gw_span_energy:f64,
    pub gw_search_grid:usize,
    pub bse_max_ang_momentum:usize,
    pub gw_rootfinder:String,
    pub simplified_bse:bool,
    pub self_energy_spectrum_test:bool,
    pub spectrum_test_start:f64,
    pub spectrum_test_end:f64,
    pub spectrum_test_step:f64,
    pub pysoc:bool,
    pub gw_imag_rayon:bool,
    pub bse_auxbas_path: Option<String>,
    pub bse_feast_renormalized_doubles: bool,
    pub bse_renormalized_doubles_extra_width: f64,
    pub bse_feast_precondition_type: String,
    pub bse_feast_inner_gmres_tol: f64,
    pub bse_feast_inner_gmres_restart: usize,
    pub bse_feast_inner_gmres_max_iter: usize,
    // QSGW-specific controls
    pub qsgw_max_iter: usize,
    pub qsgw_energy_tol: f64,
    pub qsgw_mix_param: f64,
    pub qsgw_eta: f64,
    /// Lorentzian broadening (Ha) for CD-GW real-axis contour deformation.
    /// Default 0.0 (no broadening, backward compatible). 0.01 recommended for stability.
    pub cdgw_eta: f64,
    // response BSE grid sampling parameters
    pub response_bse_x_start: f64,
    pub response_bse_x_end: f64,
    pub response_bse_x_points: usize,
    pub response_bse_y_start: f64,
    pub response_bse_y_end: f64,
    pub response_bse_y_points: usize,
    pub response_bse_z_start: f64,
    pub response_bse_z_end: f64,
    pub response_bse_z_points: usize,
    pub response_bse_grids: Vec<[f64; 3]>,
    // response BSE solver selection
    pub response_bse_solver: String,
    pub response_bse_tol: f64,
    pub response_bse_max_iter: usize,
    // FEAST solver control and parameters for BSE
    pub bse_feast_solver: bool,
    pub bse_eigenrange_min: f64,
    pub bse_eigenrange_max: f64,
    pub bse_m_expected: usize,
    pub bse_max_feast_iter: usize,
    pub bse_tol_feast: f64,
    pub bse_feast_cg_max_iter: usize,
    pub bse_feast_cg_tol: f64,
    pub bse_feast_gmres_restart: usize,
    pub bse_feast_gmres_max_iter: usize,
    // FEAST initial guess type: "random" (default) or "gaussian"
    pub bse_feast_init_guess_type: String,
    // Gaussian width = (step * width_factor)²  (default 0.5 → half-spacing)
    pub bse_feast_gaussian_width_factor: f64,
    // Parallelise over quadrature points via rayon (default true).
    // If false, quadrature points are solved one by one in serial.
    pub bse_feast_contour_rayon: bool,
    // Number of Gauss-Legendre quadrature points for contour integration (default 8)
    pub bse_feast_n_quad: usize,
    // NLFEAST (nonlinear BSE) control parameters
    pub nonlinear_bse: bool,
    pub nlfeast_centre: f64,
    pub nlfeast_radius: f64,
    pub nlfeast_m0: usize,
    pub nlfeast_n_quad: usize,
    pub nlfeast_max_iter: usize,
    pub nlfeast_tol: f64,
    pub nlfeast_gmres_restart: usize,
    pub nlfeast_gmres_max_it: usize,
    pub nlfeast_gmres_tol: f64,
    pub export_matvec_count: bool,
    pub gw_switch_fallback_threshold: f64,
}

impl Default for QuasiParticle {
    fn default() -> Self {
        QuasiParticle {
            gw_scheme:String::from("no gw"),
            homo_lumo_gw_qp:false,
            x_alpha:0.5,
            save_qp:false,
            bse_davidson_solver:false,
            davidson_target_excitations:6,
            davidson_converge_threshold:1e-6,
            davidson_maximum_subspace_size:2,
            davidson_restart_dimensions:5,
            davidson_add_dimensions:4,
            davidson_max_iter:20,
            bse_tda:false,
            bse_spin:String::from("none"),
            bse_cutoff_energy:1000000.0,
            save_bse_terms:false,
            obtain_vx_vc_terms:false,
            obtain_pure_exchange:false,
            obtain_ks_homos:false,
            gw_linearize_shift:1e-2,
            gw_linearize_derivative_h:1e-10,
            renormalized_singles:false,
            w_rs:false,
            scgw:String::from("g0w0"),
            gw:false,
            save_bse_excitations:false, 
            evgw_rounds:0,
            save_gw_homo_lumo_qp:false,
            save_qp_path:String::from("single_qp_path.txt"),
            save_first_excitation:false,
            save_first_excitation_path:String::from("first_excitation_save.txt"),
            use_low_rank_contour:false,
            low_rank_grid_type:String::from("linear"),
            nomega_chi_real:6,
            low_rank_tolerance:1e-3,
            omega_chi_max:0.0,
            nomega_sigma:10,
            step_sigma:0.05,
            parse_qp_path:String::from("./qp_energies"),
            fourier_self_energy:false,
            fse_sin_coeff_path:String::from("./fse_sin_coeff.txt"),
            fse_cos_coeff_path:String::from("./fse_cos_coeff.txt"),
            hermite_self_energy:false,
            hermite_coeff_path:String::from("./hermite_coeff.txt"),
            bse_qp_polarization:false,
            threshold:0.1,
            gw_or_bse:String::new(),
            gw_span_energy:0.2,
            bse_exchange_rescaling:1.0,
            gw_search_grid:51,
            gw_rootfinder:"newton".to_string(),
            simplified_bse:false,
            bse_max_ang_momentum:10,
            self_energy_spectrum_test:false,
            spectrum_test_start:-1.0,
            spectrum_test_end:0.0,
            spectrum_test_step:0.01,
            pysoc:false,
            external_field_freq:0.5,
            lifetime_gamma:0.001,
            gw_imag_rayon:true,
            bse_auxbas_path: None,
            bse_feast_renormalized_doubles: false,
            bse_renormalized_doubles_extra_width: 0.1,
            bse_feast_precondition_type: String::from("inner_gmres"),
            bse_feast_inner_gmres_tol: 0.0001,
            bse_feast_inner_gmres_restart: 50,
            bse_feast_inner_gmres_max_iter: 100,
            qsgw_max_iter: 50,
            qsgw_energy_tol: 1e-5,
            qsgw_mix_param: 0.5,
            qsgw_eta: 0.001,
            cdgw_eta: 0.0,
            // response BSE grid sampling parameters (default: 2 points per dimension)
            response_bse_x_start: 0.0,
            response_bse_x_end: 1.0,
            response_bse_x_points: 2,
            response_bse_y_start: 0.0,
            response_bse_y_end: 1.0,
            response_bse_y_points: 2,
            response_bse_z_start: 0.0,
            response_bse_z_end: 1.0,
            response_bse_z_points: 2,
            response_bse_grids: Vec::new(),
            response_bse_solver: String::from("klopper"),
            response_bse_tol: 1e-6,
            response_bse_max_iter: 200,
            bse_feast_solver: false,
            bse_eigenrange_min: 0.0,
            bse_eigenrange_max: 0.5,
            bse_m_expected: 20,
            bse_max_feast_iter: 30,
            bse_tol_feast: 1e-8,
            bse_feast_cg_max_iter: 100,
            bse_feast_cg_tol: 1e-8,
            bse_feast_gmres_restart: 200,
            bse_feast_gmres_max_iter: 500,
            bse_feast_init_guess_type: String::from("random"),
            bse_feast_gaussian_width_factor: 0.5,
            bse_feast_contour_rayon: true,
            bse_feast_n_quad: 8,
            nonlinear_bse: false,
            nlfeast_centre: 0.0,
            nlfeast_radius: 0.5,
            nlfeast_m0: 20,
            nlfeast_n_quad: 12,
            nlfeast_max_iter: 20,
            nlfeast_tol: 1e-8,
            nlfeast_gmres_restart: 200,
            nlfeast_gmres_max_it: 500,
            nlfeast_gmres_tol: 1e-6,
            export_matvec_count: false,
            gw_switch_fallback_threshold: 1e6,
        }
    }
}

impl QuasiParticle { 
    pub fn to_toml(&self) -> toml::Value {
        let mut table = toml::map::Map::new();
        
        table.insert("gw_scheme".to_string(), toml::Value::String(self.gw_scheme.clone()));
        table.insert("homo_lumo_gw_qp".to_string(), toml::Value::Boolean(self.homo_lumo_gw_qp));
        table.insert("x_alpha".to_string(), toml::Value::Float(self.x_alpha));
        table.insert("save_qp".to_string(), toml::Value::Boolean(self.save_qp));
        table.insert("bse_davidson_solver".to_string(), toml::Value::Boolean(self.bse_davidson_solver));
        table.insert("davidson_target_excitations".to_string(), toml::Value::Integer(self.davidson_target_excitations as i64));
        table.insert("davidson_converge_threshold".to_string(), toml::Value::Float(self.davidson_converge_threshold));
        table.insert("davidson_maximum_subspace_size".to_string(), toml::Value::Integer(self.davidson_maximum_subspace_size as i64));
        table.insert("davidson_restart_dimensions".to_string(), toml::Value::Integer(self.davidson_restart_dimensions as i64));
        table.insert("davidson_add_dimensions".to_string(), toml::Value::Integer(self.davidson_add_dimensions as i64));
        table.insert("davidson_max_iter".to_string(), toml::Value::Integer(self.davidson_max_iter as i64));
        table.insert("bse_tda".to_string(), toml::Value::Boolean(self.bse_tda));
        table.insert("bse_spin".to_string(), toml::Value::String(self.bse_spin.clone()));
        table.insert("bse_cutoff_energy".to_string(), toml::Value::Float(self.bse_cutoff_energy));
        table.insert("save_bse_terms".to_string(), toml::Value::Boolean(self.save_bse_terms));
        table.insert("obtain_vx_vc_terms".to_string(), toml::Value::Boolean(self.obtain_vx_vc_terms));
        table.insert("obtain_pure_exchange".to_string(), toml::Value::Boolean(self.obtain_pure_exchange));
        table.insert("obtain_ks_homos".to_string(), toml::Value::Boolean(self.obtain_ks_homos));
        table.insert("gw_linearize_shift".to_string(), toml::Value::Float(self.gw_linearize_shift));
        table.insert("gw_linearize_derivative_h".to_string(), toml::Value::Float(self.gw_linearize_derivative_h));
        table.insert("renormalized_singles".to_string(), toml::Value::Boolean(self.renormalized_singles));
        table.insert("w_rs".to_string(), toml::Value::Boolean(self.w_rs));
        table.insert("scgw".to_string(), toml::Value::String(self.scgw.clone()));
        table.insert("gw_rootfinder".to_string(), toml::Value::String(self.gw_rootfinder.clone()));
        table.insert("gw".to_string(), toml::Value::Boolean(self.gw));
        table.insert("save_bse_excitations".to_string(), toml::Value::Boolean(self.save_bse_excitations));
        table.insert("evgw_rounds".to_string(), toml::Value::Integer(self.evgw_rounds as i64));
        table.insert("save_gw_homo_lumo_qp".to_string(), toml::Value::Boolean(self.save_gw_homo_lumo_qp));
        table.insert("save_qp_path".to_string(), toml::Value::String(self.save_qp_path.clone()));
        table.insert("save_first_excitation".to_string(), toml::Value::Boolean(self.save_first_excitation));
        table.insert("save_first_excitation_path".to_string(), toml::Value::String(self.save_first_excitation_path.clone()));
        table.insert("use_low_rank_contour".to_string(), toml::Value::Boolean(self.use_low_rank_contour));
        table.insert("low_rank_grid_type".to_string(), toml::Value::String(self.low_rank_grid_type.clone()));
        table.insert("nomega_chi_real".to_string(), toml::Value::Integer(self.nomega_chi_real as i64));
        table.insert("low_rank_tolerance".to_string(), toml::Value::Float(self.low_rank_tolerance));
        table.insert("omega_chi_max".to_string(), toml::Value::Float(self.omega_chi_max));
        table.insert("nomega_sigma".to_string(), toml::Value::Integer(self.nomega_sigma as i64));
        table.insert("step_sigma".to_string(), toml::Value::Float(self.step_sigma));
        table.insert("fse_sin_coeff_path".to_string(), toml::Value::String(self.fse_sin_coeff_path.clone()));
        table.insert("fse_cos_coeff_path".to_string(), toml::Value::String(self.fse_cos_coeff_path.clone()));
        table.insert("hermite_self_energy".to_string(), toml::Value::Boolean(self.hermite_self_energy));
        table.insert("hermite_coeff_path".to_string(), toml::Value::String(self.hermite_coeff_path.clone()));
        table.insert("parse_qp_path".to_string(), toml::Value::String(self.parse_qp_path.clone()));
        table.insert("bse_qp_polarization".to_string(), toml::Value::Boolean(self.bse_qp_polarization));
        table.insert("threshold".to_string(), toml::Value::Float(self.threshold));
        table.insert("gw_span_energy".to_string(), toml::Value::Float(self.gw_span_energy));
        table.insert("external_field_freq".to_string(), toml::Value::Float(self.external_field_freq));
        table.insert("lifetime_gamma".to_string(), toml::Value::Float(self.lifetime_gamma));
        table.insert("gw_or_bse".to_string(), toml::Value::String(self.gw_or_bse.clone()));
        table.insert("gw_search_grid".to_string(), toml::Value::Integer(self.gw_search_grid as i64));
        table.insert("simplified_bse".to_string(), toml::Value::Boolean(self.simplified_bse));
        table.insert("pysoc".to_string(), toml::Value::Boolean(self.pysoc));
        table.insert("gw_imag_rayon".to_string(), toml::Value::Boolean(self.gw_imag_rayon));
        table.insert("fourier_self_energy".to_string(), toml::Value::Boolean(self.fourier_self_energy));
        table.insert("bse_exchange_rescaling".to_string(), toml::Value::Float(self.bse_exchange_rescaling));
        table.insert("self_energy_spectrum_test".to_string(), toml::Value::Boolean(self.self_energy_spectrum_test));
        table.insert("spectrum_test_start".to_string(), toml::Value::Float(self.spectrum_test_start));
        table.insert("spectrum_test_end".to_string(), toml::Value::Float(self.spectrum_test_end));
        table.insert("spectrum_test_step".to_string(), toml::Value::Float(self.spectrum_test_step));
        table.insert("bse_max_ang_momentum".to_string(), toml::Value::Integer(self.bse_max_ang_momentum as i64));
        if let Some(path) = &self.bse_auxbas_path {
            table.insert("bse_auxbas_path".to_string(), toml::Value::String(path.clone()));
        }
        table.insert("bse_feast_renormalized_doubles".to_string(), toml::Value::Boolean(self.bse_feast_renormalized_doubles));
        table.insert("bse_renormalized_doubles_extra_width".to_string(), toml::Value::Float(self.bse_renormalized_doubles_extra_width));
        table.insert("bse_feast_precondition_type".to_string(), toml::Value::String(self.bse_feast_precondition_type.clone()));
        table.insert("bse_feast_inner_gmres_tol".to_string(), toml::Value::Float(self.bse_feast_inner_gmres_tol));
        table.insert("bse_feast_inner_gmres_restart".to_string(), toml::Value::Integer(self.bse_feast_inner_gmres_restart as i64));
        table.insert("bse_feast_inner_gmres_max_iter".to_string(), toml::Value::Integer(self.bse_feast_inner_gmres_max_iter as i64));
        table.insert("qsgw_max_iter".to_string(), toml::Value::Integer(self.qsgw_max_iter as i64));
        table.insert("qsgw_energy_tol".to_string(), toml::Value::Float(self.qsgw_energy_tol));
        table.insert("qsgw_mix_param".to_string(), toml::Value::Float(self.qsgw_mix_param));
        table.insert("qsgw_eta".to_string(), toml::Value::Float(self.qsgw_eta));
        table.insert("cdgw_eta".to_string(), toml::Value::Float(self.cdgw_eta));
        table.insert("response_bse_x_start".to_string(), toml::Value::Float(self.response_bse_x_start));
        table.insert("response_bse_x_end".to_string(), toml::Value::Float(self.response_bse_x_end));
        table.insert("response_bse_x_points".to_string(), toml::Value::Integer(self.response_bse_x_points as i64));
        table.insert("response_bse_y_start".to_string(), toml::Value::Float(self.response_bse_y_start));
        table.insert("response_bse_y_end".to_string(), toml::Value::Float(self.response_bse_y_end));
        table.insert("response_bse_y_points".to_string(), toml::Value::Integer(self.response_bse_y_points as i64));
        table.insert("response_bse_z_start".to_string(), toml::Value::Float(self.response_bse_z_start));
        table.insert("response_bse_z_end".to_string(), toml::Value::Float(self.response_bse_z_end));
        table.insert("response_bse_z_points".to_string(), toml::Value::Integer(self.response_bse_z_points as i64));
        table.insert("bse_eigenrange_min".to_string(), toml::Value::Float(self.bse_eigenrange_min));
        table.insert("bse_feast_solver".to_string(), toml::Value::Boolean(self.bse_feast_solver));
        table.insert("bse_eigenrange_max".to_string(), toml::Value::Float(self.bse_eigenrange_max));
        table.insert("bse_m_expected".to_string(), toml::Value::Integer(self.bse_m_expected as i64));
        table.insert("bse_max_feast_iter".to_string(), toml::Value::Integer(self.bse_max_feast_iter as i64));
        table.insert("bse_tol_feast".to_string(), toml::Value::Float(self.bse_tol_feast));
        table.insert("bse_feast_cg_max_iter".to_string(), toml::Value::Integer(self.bse_feast_cg_max_iter as i64));
        table.insert("bse_feast_cg_tol".to_string(), toml::Value::Float(self.bse_feast_cg_tol));
        table.insert("bse_feast_gmres_restart".to_string(), toml::Value::Integer(self.bse_feast_gmres_restart as i64));
        table.insert("bse_feast_gmres_max_iter".to_string(), toml::Value::Integer(self.bse_feast_gmres_max_iter as i64));
        table.insert("bse_feast_init_guess_type".to_string(), toml::Value::String(self.bse_feast_init_guess_type.clone()));
        table.insert("bse_feast_gaussian_width_factor".to_string(), toml::Value::Float(self.bse_feast_gaussian_width_factor));
        table.insert("bse_feast_contour_rayon".to_string(), toml::Value::Boolean(self.bse_feast_contour_rayon));
        table.insert("bse_feast_n_quad".to_string(), toml::Value::Integer(self.bse_feast_n_quad as i64));
        table.insert("response_bse_solver".to_string(), toml::Value::String(self.response_bse_solver.clone()));
        table.insert("response_bse_tol".to_string(), toml::Value::Float(self.response_bse_tol));
        table.insert("response_bse_max_iter".to_string(), toml::Value::Integer(self.response_bse_max_iter as i64));
        table.insert("nonlinear_bse".to_string(), toml::Value::Boolean(self.nonlinear_bse));
        table.insert("nlfeast_centre".to_string(), toml::Value::Float(self.nlfeast_centre));
        table.insert("nlfeast_radius".to_string(), toml::Value::Float(self.nlfeast_radius));
        table.insert("nlfeast_m0".to_string(), toml::Value::Integer(self.nlfeast_m0 as i64));
        table.insert("nlfeast_n_quad".to_string(), toml::Value::Integer(self.nlfeast_n_quad as i64));
        table.insert("nlfeast_max_iter".to_string(), toml::Value::Integer(self.nlfeast_max_iter as i64));
        table.insert("nlfeast_tol".to_string(), toml::Value::Float(self.nlfeast_tol));
        table.insert("nlfeast_gmres_restart".to_string(), toml::Value::Integer(self.nlfeast_gmres_restart as i64));
        table.insert("nlfeast_gmres_max_it".to_string(), toml::Value::Integer(self.nlfeast_gmres_max_it as i64));
        table.insert("nlfeast_gmres_tol".to_string(), toml::Value::Float(self.nlfeast_gmres_tol));
        table.insert("export_matvec_count".to_string(), toml::Value::Boolean(self.export_matvec_count));
        table.insert("gw_switch_fallback_threshold".to_string(), toml::Value::Float(self.gw_switch_fallback_threshold));
        toml::Value::Table(table)
    }
}

pub fn parse_quasiparticle_keywords(tmp_keys: &serde_json::Value) -> anyhow::Result<Option<QuasiParticle>> { 
    match tmp_keys.get("quasiparticle_methods").unwrap_or(&serde_json::Value::Null) {
        serde_json::Value::Object(tmp_ctrl) => {
            let mut tmp_input = QuasiParticle::default();
            tmp_input.gw_scheme = match tmp_ctrl.get("gw_scheme").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => String::from("no gw"),
            };
            tmp_input.homo_lumo_gw_qp=match tmp_ctrl.get("homo_lumo_gw_qp").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.x_alpha=match tmp_ctrl.get("x_alpha").unwrap_or(&serde_json::Value::Null){
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(2.0_f64)},
                other => {0.5},
            };
            tmp_input.save_qp = match tmp_ctrl.get("save_qp").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.bse_davidson_solver = match tmp_ctrl.get("bse_davidson_solver").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.bse_tda = match tmp_ctrl.get("bse_tda").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.self_energy_spectrum_test= match tmp_ctrl.get("self_energy_spectrum_test").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.spectrum_test_start = match tmp_ctrl.get("spectrum_test_start").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.5_f64)},
                other => {-1.0},
            };
            tmp_input.spectrum_test_end = match tmp_ctrl.get("spectrum_test_end").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.5_f64)},
                other => {0.0},
            };
            tmp_input.spectrum_test_step = match tmp_ctrl.get("spectrum_test_step").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.5_f64)},
                other => {0.01},
            };
            tmp_input.bse_spin = match tmp_ctrl.get("bse_spin").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => String::from("none"),
            };
            tmp_input.bse_cutoff_energy = match tmp_ctrl.get("bse_cutoff_energy").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1000000.0_f64)},
                other => {1000000.0},
            };
            tmp_input.bse_exchange_rescaling = match tmp_ctrl.get("bse_exchange_rescaling").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.0_f64)},
                other => {1.0},
            };
            tmp_input.external_field_freq = match tmp_ctrl.get("external_field_freq").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(0.5_f64)},
                other => {0.5},
            };tmp_input.lifetime_gamma = match tmp_ctrl.get("lifetime_gamma").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(0.001_f64)},
                other => {0.001},
            };
            tmp_input.davidson_converge_threshold = match tmp_ctrl.get("davidson_converge_threshold").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1e-6_f64)},
                other => {1e-6},
            };
            tmp_input.davidson_target_excitations = match tmp_ctrl.get("davidson_target_excitations").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(6_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(6) as usize},
                other => {6}
            };
            tmp_input.davidson_max_iter = match tmp_ctrl.get("davidson_max_iter").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(6_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(6) as usize},
                other => {20}
            };
            let maximum_subspace_size=(((tmp_input.davidson_target_excitations as f64)*3.0).ceil() as usize);
            tmp_input.davidson_maximum_subspace_size = match tmp_ctrl.get("davidson_maximum_subspace_size").unwrap_or(&serde_json::Value::Null) {
        
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(maximum_subspace_size) as usize},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(maximum_subspace_size as i64) as usize},
                other => {maximum_subspace_size}
            };
            let restart_size=(((tmp_input.davidson_target_excitations as f64)*1.5).ceil() as usize);
            tmp_input.davidson_restart_dimensions = match tmp_ctrl.get("davidson_restart_dimensions").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(restart_size) as usize},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(restart_size as i64) as usize},
                other => {restart_size}
            };
            tmp_input.davidson_add_dimensions = match tmp_ctrl.get("davidson_add_dimensions").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(restart_size) as usize},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(restart_size as i64) as usize},
                other => {restart_size}
            };
            tmp_input.save_bse_terms = match tmp_ctrl.get("save_bse_terms").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.fourier_self_energy = match tmp_ctrl.get("fourier_self_energy").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.hermite_self_energy = match tmp_ctrl.get("hermite_self_energy").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.obtain_vx_vc_terms = match tmp_ctrl.get("obtain_vx_vc_terms").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.pysoc = match tmp_ctrl.get("pysoc").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.gw_imag_rayon = match tmp_ctrl.get("gw_imag_rayon").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {true},
            };
            tmp_input.obtain_pure_exchange = match tmp_ctrl.get("obtain_pure_exchange").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.obtain_ks_homos = match tmp_ctrl.get("obtain_ks_homos").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.gw_linearize_shift= match tmp_ctrl.get("gw_linearize_shift").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(0.01_f64)},
                other => {0.01},
            };
            tmp_input.gw_linearize_derivative_h= match tmp_ctrl.get("gw_linearize_derivative_h").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1e-10_f64)},
                other => {1e-10},
            };
            tmp_input.threshold= match tmp_ctrl.get("threshold").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(0.1)},
                other => {0.1},
            };
            tmp_input.gw_span_energy= match tmp_ctrl.get("gw_span_energy").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(0.1)},
                other => {0.1},
            };
            tmp_input.renormalized_singles = match tmp_ctrl.get("renormalized_singles").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.w_rs = match tmp_ctrl.get("w_rs").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.scgw = match tmp_ctrl.get("scgw").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => String::from("g0w0"),
            };
            tmp_input.gw = match tmp_ctrl.get("gw").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };

            tmp_input.evgw_rounds = match tmp_ctrl.get("evgw_rounds").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(4_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(4) as usize},
                other => {0}
            };
            tmp_input.bse_max_ang_momentum = match tmp_ctrl.get("bse_max_ang_momentum").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(10_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(10) as usize},
                other => {10}
            };
            tmp_input.gw_search_grid = match tmp_ctrl.get("gw_search_grid").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(21_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(21) as usize},
                other => {21}
            };
            tmp_input.save_gw_homo_lumo_qp = match tmp_ctrl.get("save_gw_homo_lumo_qp").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.save_bse_excitations = match tmp_ctrl.get("save_bse_excitations").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.simplified_bse = match tmp_ctrl.get("simplified_bse").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.save_qp_path = match tmp_ctrl.get("save_qp_path").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => String::from("single_qp_save.txt"),
            };
            tmp_input.fse_sin_coeff_path = match tmp_ctrl.get("fse_sin_coeff_path").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => String::from("fse_sin_coeff.txt"),
            };
            tmp_input.fse_cos_coeff_path = match tmp_ctrl.get("fse_cos_coeff_path").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => String::from("fse_cos_coeff.txt"),
            };
            tmp_input.hermite_coeff_path = match tmp_ctrl.get("hermite_coeff_path").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => String::from("hermite_coeff.txt"),
            };
            tmp_input.gw_rootfinder = match tmp_ctrl.get("gw_rootfinder").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => String::from("newton".to_string()),
            };
            tmp_input.save_first_excitation = match tmp_ctrl.get("save_first_excitation").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.save_first_excitation_path = match tmp_ctrl.get("save_first_excitation_path").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => String::from("first_excitation_save.txt"),
            };
            tmp_input.use_low_rank_contour = match tmp_ctrl.get("use_low_rank_contour").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                _ => {false},
            };
            tmp_input.low_rank_grid_type = match tmp_ctrl.get("low_rank_grid_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => {
                    let lower = s.to_lowercase();
                    match lower.as_str() {
                        "quadratic" => String::from("quadratic"),
                        _ => String::from("linear"),
                    }
                },
                _ => String::from("linear"),
            };
            tmp_input.nomega_chi_real = match tmp_ctrl.get("nomega_chi_real").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(6_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(6) as usize},
                _ => {6}
            };
            tmp_input.low_rank_tolerance = match tmp_ctrl.get("low_rank_tolerance").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1e-3)},
                _ => {1e-3},
            };
            tmp_input.omega_chi_max = match tmp_ctrl.get("omega_chi_max").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(0.0)},
                _ => {0.0},
            };
            tmp_input.nomega_sigma = match tmp_ctrl.get("nomega_sigma").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_u64().unwrap_or(10) as usize},
                _ => {10},
            };
            tmp_input.step_sigma = match tmp_ctrl.get("step_sigma").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(0.05)},
                _ => {0.05},
            };
            tmp_input.parse_qp_path = match tmp_ctrl.get("parse_qp_path").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => String::from("./qp_energies"),
            };
            tmp_input.gw_or_bse = match tmp_ctrl.get("gw_or_bse").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => String::new(),
            };
            tmp_input.bse_qp_polarization = match tmp_ctrl.get("bse_qp_polarization").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.bse_auxbas_path = match tmp_ctrl.get("bse_auxbas_path").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => Some(s.clone()),
                _ => None,
            };
            tmp_input.bse_feast_renormalized_doubles = match tmp_ctrl.get("bse_feast_renormalized_doubles").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(b) => *b,
                _ => false,
            };
            tmp_input.bse_renormalized_doubles_extra_width = match tmp_ctrl.get("bse_renormalized_doubles_extra_width").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.1),
                _ => 0.1,
            };
            tmp_input.bse_feast_precondition_type = match tmp_ctrl.get("bse_feast_precondition_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => {
                    let lower = s.to_lowercase();
                    match lower.as_str() {
                        "inner_gmres" | "diagonal" | "diag" => {
                            if lower == "diag" { String::from("diagonal") }
                            else { lower }
                        },
                        _ => String::from("diagonal"),
                    }
                },
                _ => String::from("diagonal"),
            };
            tmp_input.bse_feast_inner_gmres_tol = match tmp_ctrl.get("bse_feast_inner_gmres_tol").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0001),
                _ => 0.0001,
            };
            tmp_input.bse_feast_inner_gmres_restart = match tmp_ctrl.get("bse_feast_inner_gmres_restart").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(50) as usize,
                _ => 50,
            };
            tmp_input.bse_feast_inner_gmres_max_iter = match tmp_ctrl.get("bse_feast_inner_gmres_max_iter").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(100) as usize,
                _ => 100,
            };
            tmp_input.qsgw_max_iter = match tmp_ctrl.get("qsgw_max_iter").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(50) as usize,
                _ => 50,
            };
            tmp_input.qsgw_energy_tol = match tmp_ctrl.get("qsgw_energy_tol").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1e-5),
                _ => 1e-5,
            };
            tmp_input.qsgw_mix_param = match tmp_ctrl.get("qsgw_mix_param").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.5),
                _ => 0.5,
            };
            tmp_input.qsgw_eta = match tmp_ctrl.get("qsgw_eta").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.001),
                _ => 0.001,
            };
            tmp_input.cdgw_eta = match tmp_ctrl.get("cdgw_eta").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
                _ => 0.0,
            };
            // Parse response BSE grid sampling parameters
            tmp_input.response_bse_x_start = match tmp_ctrl.get("response_bse_x_start").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
                _ => 0.0,
            };
            tmp_input.response_bse_x_end = match tmp_ctrl.get("response_bse_x_end").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1.0),
                _ => 1.0,
            };
            tmp_input.response_bse_x_points = match tmp_ctrl.get("response_bse_x_points").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(2) as usize,
                _ => 2,
            };
            tmp_input.response_bse_y_start = match tmp_ctrl.get("response_bse_y_start").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
                _ => 0.0,
            };
            tmp_input.response_bse_y_end = match tmp_ctrl.get("response_bse_y_end").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1.0),
                _ => 1.0,
            };
            tmp_input.response_bse_y_points = match tmp_ctrl.get("response_bse_y_points").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(2) as usize,
                _ => 2,
            };
            tmp_input.response_bse_z_start = match tmp_ctrl.get("response_bse_z_start").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
                _ => 0.0,
            };
            tmp_input.response_bse_z_end = match tmp_ctrl.get("response_bse_z_end").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1.0),
                _ => 1.0,
            };
            tmp_input.response_bse_z_points = match tmp_ctrl.get("response_bse_z_points").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(2) as usize,
                _ => 2,
            };
            // Parse FEAST solver control and parameters
            tmp_input.bse_feast_solver = match tmp_ctrl.get("bse_feast_solver").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(b) => *b,
                _ => false,
            };
            tmp_input.bse_eigenrange_min = match tmp_ctrl.get("bse_eigenrange_min").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
                _ => 0.0,
            };
            tmp_input.bse_eigenrange_max = match tmp_ctrl.get("bse_eigenrange_max").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.5),
                _ => 0.5,
            };
            tmp_input.bse_m_expected = match tmp_ctrl.get("bse_m_expected").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(20) as usize,
                _ => 20,
            };
            tmp_input.bse_max_feast_iter = match tmp_ctrl.get("bse_max_feast_iter").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(30) as usize,
                _ => 30,
            };
            tmp_input.bse_tol_feast = match tmp_ctrl.get("bse_tol_feast").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1e-8),
                _ => 1e-8,
            };
            tmp_input.bse_feast_cg_max_iter = match tmp_ctrl.get("bse_feast_cg_max_iter").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(100) as usize,
                _ => 100,
            };
            tmp_input.bse_feast_cg_tol = match tmp_ctrl.get("bse_feast_cg_tol").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1e-8),
                _ => 1e-8,
            };
            tmp_input.bse_feast_gmres_restart = match tmp_ctrl.get("bse_feast_gmres_restart").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(200) as usize,
                _ => 200,
            };
            tmp_input.bse_feast_gmres_max_iter = match tmp_ctrl.get("bse_feast_gmres_max_iter").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(500) as usize,
                _ => 500,
            };
            tmp_input.bse_feast_init_guess_type = match tmp_ctrl.get("bse_feast_init_guess_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => {
                    let lower = s.to_lowercase();
                    match lower.as_str() {
                        "gaussian" => String::from("gaussian"),
                        _ => String::from("random"),
                    }
                },
                _ => String::from("random"),
            };
            tmp_input.bse_feast_gaussian_width_factor = match tmp_ctrl.get("bse_feast_gaussian_width_factor").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.5),
                _ => 0.5,
            };
            tmp_input.bse_feast_contour_rayon = match tmp_ctrl.get("bse_feast_contour_rayon").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(b) => *b,
                _ => true,
            };
            tmp_input.bse_feast_n_quad = match tmp_ctrl.get("bse_feast_n_quad").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(8) as usize,
                _ => 8,
            };
            // Generate grids: OUTER LOOP X, MIDDLE LOOP Y, INNER LOOP Z
            let x_step = if tmp_input.response_bse_x_points > 1 {
                (tmp_input.response_bse_x_end - tmp_input.response_bse_x_start) / (tmp_input.response_bse_x_points - 1) as f64
            } else {
                0.0
            };
            let y_step = if tmp_input.response_bse_y_points > 1 {
                (tmp_input.response_bse_y_end - tmp_input.response_bse_y_start) / (tmp_input.response_bse_y_points - 1) as f64
            } else {
                0.0
            };
            let z_step = if tmp_input.response_bse_z_points > 1 {
                (tmp_input.response_bse_z_end - tmp_input.response_bse_z_start) / (tmp_input.response_bse_z_points - 1) as f64
            } else {
                0.0
            };
            let mut grids = Vec::with_capacity(tmp_input.response_bse_x_points * tmp_input.response_bse_y_points * tmp_input.response_bse_z_points);
            for ix in 0..tmp_input.response_bse_x_points {
                let x = tmp_input.response_bse_x_start + ix as f64 * x_step;
                for iy in 0..tmp_input.response_bse_y_points {
                    let y = tmp_input.response_bse_y_start + iy as f64 * y_step;
                    for iz in 0..tmp_input.response_bse_z_points {
                        let z = tmp_input.response_bse_z_start + iz as f64 * z_step;
                        grids.push([x, y, z]);
                    }
                }
            }
            tmp_input.response_bse_grids = grids;
            tmp_input.response_bse_solver = match tmp_ctrl.get("response_bse_solver").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone().to_lowercase(),
                _ => String::from("klopper"),
            };
            tmp_input.response_bse_tol = match tmp_ctrl.get("response_bse_tol").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1e-6),
                _ => 1e-6,
            };
            tmp_input.response_bse_max_iter = match tmp_ctrl.get("response_bse_max_iter").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(200) as usize,
                _ => 200,
            };
            // NLFEAST (nonlinear BSE) control parameters
            tmp_input.nonlinear_bse = match tmp_ctrl.get("nonlinear_bse").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(b) => *b,
                _ => false,
            };
            tmp_input.nlfeast_centre = match tmp_ctrl.get("nlfeast_centre").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
                _ => 0.0,
            };
            tmp_input.nlfeast_radius = match tmp_ctrl.get("nlfeast_radius").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.5),
                _ => 0.5,
            };
            tmp_input.nlfeast_m0 = match tmp_ctrl.get("nlfeast_m0").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(20) as usize,
                _ => 20,
            };
            tmp_input.nlfeast_n_quad = match tmp_ctrl.get("nlfeast_n_quad").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(12) as usize,
                _ => 12,
            };
            tmp_input.nlfeast_max_iter = match tmp_ctrl.get("nlfeast_max_iter").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(20) as usize,
                _ => 20,
            };
            tmp_input.nlfeast_tol = match tmp_ctrl.get("nlfeast_tol").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1e-8),
                _ => 1e-8,
            };
            tmp_input.nlfeast_gmres_restart = match tmp_ctrl.get("nlfeast_gmres_restart").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(200) as usize,
                _ => 200,
            };
            tmp_input.nlfeast_gmres_max_it = match tmp_ctrl.get("nlfeast_gmres_max_it").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(500) as usize,
                _ => 500,
            };
            tmp_input.export_matvec_count = match tmp_ctrl.get("export_matvec_count").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(b) => *b,
                _ => false,
            };
            tmp_input.nlfeast_gmres_tol = match tmp_ctrl.get("nlfeast_gmres_tol").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1e-6),
                _ => 1e-6,
            };
            tmp_input.gw_switch_fallback_threshold = match tmp_ctrl.get("gw_switch_fallback_threshold").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1e6),
                _ => 1e6,
            };
            return Ok(Some(tmp_input));
        },
        other => {
            return Ok(None);
        },
    }
    
}