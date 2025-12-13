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
    pub parse_qp_path:String,
    pub bse_qp_polarization:bool,
    pub threshold:f64,
    pub gw_or_bse:String,
    pub gw_span_energy:f64,
    pub gw_search_grid:usize,
    pub bse_max_ang_momentum:usize,
    pub gw_rootfinder:String,
    pub simplified_bse:bool,
    pub pysoc:bool
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
            parse_qp_path:String::from("./qp_energies"),
            bse_qp_polarization:false,
            threshold:0.1,
            gw_or_bse:String::new(),
            gw_span_energy:0.2,
            bse_exchange_rescaling:1.0,
            gw_search_grid:51,
            gw_rootfinder:"newton".to_string(),
            simplified_bse:false,
            bse_max_ang_momentum:10,
            pysoc:false
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
        table.insert("parse_qp_path".to_string(), toml::Value::String(self.parse_qp_path.clone()));
        table.insert("bse_qp_polarization".to_string(), toml::Value::Boolean(self.bse_qp_polarization));
        table.insert("threshold".to_string(), toml::Value::Float(self.threshold));
        table.insert("gw_span_energy".to_string(), toml::Value::Float(self.gw_span_energy));
        table.insert("gw_or_bse".to_string(), toml::Value::String(self.gw_or_bse.clone()));
        table.insert("gw_search_grid".to_string(), toml::Value::Integer(self.gw_search_grid as i64));
        table.insert("simplified_bse".to_string(), toml::Value::Boolean(self.simplified_bse));
        table.insert("pysoc".to_string(), toml::Value::Boolean(self.pysoc));
        table.insert("bse_exchange_rescaling".to_string(), toml::Value::Float(self.bse_exchange_rescaling));
        table.insert("bse_max_ang_momentum".to_string(), toml::Value::Integer(self.bse_max_ang_momentum as i64));
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
            tmp_input.bse_spin = match tmp_ctrl.get("bse_spin").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => String::from("none"),
            };
            tmp_input.bse_cutoff_energy = match tmp_ctrl.get("bse_cutoff_energy").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.5_f64)},
                other => {1000000.0},
            };
            tmp_input.bse_exchange_rescaling = match tmp_ctrl.get("bse_exchange_rescaling").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.0_f64)},
                other => {1.0},
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
            tmp_input.obtain_vx_vc_terms = match tmp_ctrl.get("obtain_vx_vc_terms").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.pysoc = match tmp_ctrl.get("pysoc").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
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
            return Ok(Some(tmp_input));
        },
        other => {
            return Ok(None);
        },
    }
    
}