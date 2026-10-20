use crate::dft::parse_xc::{DFAComponent, DFAdef, ComponentType};
use serde_json::Value;
// use crate::dft::parse_xc::xc_helper::ALIAS_WITH_DISP;
use std::collections::HashMap;
use lazy_static::lazy_static;

lazy_static! {
    pub static ref ALIAS_WITH_DISP: HashMap<&'static str, &'static str> = HashMap::from([
        ("cf22d", "cf22d + D3(version=zero, xc=cf22d)"),
        ("wb97x-d", "unsupported"),
        ("wb97x-d3bj", "wb97x_v + D3(version=bj, xc=wb97x)"),
        ("wb97x-d3", "wb97x_d3 + D3(version=zero, xc=wb97x)"),
        ("wb97x-v", "wb97x_v + VV10(xc=wb97x_v)"),
        ("wb97m-d3bj", "wb97m_v + D3(version=bj, xc=wb97m)"),
        ("wb97m-v", "wb97m_v + VV10(xc=wb97m_v)"),
        ("scan-vv10", "scan,scan_vv10 + VV10(xc=scan_vv10)"),
        ("scan-rvv10", "scan,scan_rvv10 + VV10(xc=scan_rvv10)"),
        ("revscan-vv10", "revscan,revscan_vv10 + VV10(xc=revscan_vv10)"),
    ]);
    pub static ref SUF2CODE: HashMap<&'static str, (&'static str, &'static str)> = HashMap ::from([
        ("-d3zero", ("D3", "zero")),
        ("-d3bj", ("D3", "bj")),
        ("-d4", ("D4", "bj")),
        ("-v", ("VV10", "")),
        ("-vv10", ("VV10", "")),
        ("-rvv10", ("rVV10", "")),
    ]);
}

pub fn check_disp_suffix(xc: &str) -> String {
    if ALIAS_WITH_DISP.contains_key(xc) {
        let new_xc = ALIAS_WITH_DISP.get(xc).unwrap().to_string();
        if new_xc.contains("unsupported") {
            panic!("Error: functional {} is currently not supported in REST", xc);
        }
        new_xc
    } else {
        // let suffixes = ["-d3zero", "-d3bj"];
        for suf in SUF2CODE.keys() {
            if xc.ends_with(suf) {
                let base = xc.trim_end_matches(suf);
                let (disp_engine, version) = SUF2CODE.get(suf).unwrap();
                if base.contains(",") || base.contains('+') {
                    panic!("Error: functional {} has both dispersion suffix and multiple components, which is not supported in current implementation", xc);
                }
                let result = format!("{} + {}(version={}, xc={})", base, disp_engine, version, base);
                println!("Detected dispersion suffix {}. Convert xc to {}", suf, result);
                return result;
            }
        }
        xc.to_string()
    }
}

impl DFAComponent {
    pub fn get_dftd_params(&self) -> Value {
        // params other than version and xc
        let mut params = self.param_keyword.clone();
        // params.remove("version");
        // params.remove("xc");
        Value::Object(params.into_iter().map(|(k, v)| (k, v)).collect())
    }

    pub fn is_nlc(&self) -> bool {
        match self.component_type {
            ComponentType::Libxc => {
                use libxc::prelude::*;
                let xcfunc = LibXCFunctional::from_number(self.id as i32, LibXCSpin::Unpolarized);
                xcfunc.flags().contains(LibXCFlags::VV10)
            },
            _ => false,
        }
    }
}

impl DFAdef {
    /// The `(b, C)` parameters of the NLC (VV10) kernel carried by this definition's libxc
    /// components: the single VV10-flagged functional in `xc_scf` (e.g. 466/531/584), which
    /// must appear with unit factor. `None` when no VV10-flagged component is present.
    /// Mirrors the `nlc_vv10` population in `DFAdef::to_scf_xc_data`.
    pub fn vv10_params(&self) -> Option<(f64, f64)> {
        let comps = self.xc_scf.as_ref()?;
        let mut nlc_comps = comps
            .iter()
            .filter(|comp| comp.component_type == ComponentType::Libxc && comp.is_nlc());
        let comp = match (nlc_comps.next(), nlc_comps.next()) {
            (Some(comp), None) => comp,
            _ => panic!("VV10 requires exactly one libxc functional carrying the VV10 kernel."),
        };
        if (comp.factor - 1.0).abs() > 1.0e-12 {
            panic!("VV10 component with non-unit factor is not supported: {}", comp.func_full_name);
        }
        use crate::dft::libxc_helper::xc_func_init;
        xc_func_init(comp.id, 1).vv10_coef()
    }

    pub fn has_dispersion(&self) -> bool {
        if let Some(components) = &self.xc_scf {
            for comp in components.iter() {
                if comp.component_type == ComponentType::Disp {
                    return true;
                }
            }
        }
        if let Some(components) = &self.xc_nscf {
            for comp in components.iter() {
                if comp.component_type == ComponentType::Disp {
                    return true;
                }
            }
        }
        false
    }

    pub fn get_dispersion(&self) -> Option<DFAComponent> {
        let mut disp = None;
        if let Some(components) = &self.xc_scf {
            for comp in components.iter() {
                if comp.component_type == ComponentType::Disp {
                    if disp.is_some() {
                        panic!("Error: more than one dispersion component found in SCF functional, which is not supported in current implementation");
                    } else {
                        disp = Some(comp.clone());
                    }
                }
            }
        }
        if let Some(components) = &self.xc_nscf {
            for comp in components.iter() {
                if comp.component_type == ComponentType::Disp {
                    if disp.is_some() {
                        panic!("Error: more than one dispersion component found in final energy functional, which is not supported in current implementation");
                    } else {
                        disp = Some(comp.clone());
                    }
                }
            }
        }
        disp
    }

    pub fn is_nlc(&self) -> bool {
        if self.has_dispersion() {
            match self.get_dispersion().unwrap().func.as_str() {
                "VV10" => return true,
                _ => return false,
            }
        }
        if let Some(components) = &self.xc_scf {
            for comp in components.iter() {
                if comp.is_nlc() {
                    return true;
                }
            }
        }
        if let Some(components) = &self.xc_nscf {
            for comp in components.iter() {
                if comp.is_nlc() {
                    return true;
                }
            }
        }
        false
    }
    
}