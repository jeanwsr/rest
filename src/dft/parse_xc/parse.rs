use core::{panic};
// use anyhow::{anyhow, Error};
use regex;
use std::collections::HashMap;
use lazy_static::lazy_static;
use crate::dft::libxc_helper::xc_func_init;
use crate::dft::parse_xc::xc_helper::{ALIAS, AVAIL_FUNC, CODES,
    ComponentType, 
    MULTISTEP, MULTISTEP_ALIAS, MULTISTEP_NAME_WITH_DASH, 
    NAME_WITH_DASH, NAME_WITHOUT_UNDERSCORE, NSCF_COMPONENTS, WHITELIST_NONLIBXC, 
    get_name, load_user_json_functionals};
use crate::dft::parse_xc::dispersion::{check_disp_suffix};
// use cached::proc_macro::cached;
use crate::dft::DFA4REST;
use crate::dft::DFAFamily;
use libxc::prelude::LibXCFunctional;
use serde_json::Value;

#[derive(Clone)]
pub struct DFAComponent {
    pub factor: f64,
    pub func: String,
    pub func_full_name: String,
    pub id: usize,
    pub param_positional: Vec<f64>,
    pub param_keyword: HashMap<String, Value>,
    pub component_type: ComponentType,
    // pub xcfunc: Option<XcFuncType>,
    pub reference: Vec<String>,
}


impl DFAComponent {
    pub fn new(factor: f64, func: String) -> Self {
        DFAComponent {
            factor,
            func,
            func_full_name: String::new(),
            id: 0,
            param_positional: Vec::new(),
            param_keyword: HashMap::new(),
            component_type: ComponentType::Unknown,
            // xcfunc: None,
            reference: Vec::new(),
        }
    }

    pub fn formatted_output(&self) -> String {
        // output line by line
        let mut result = format!("type: {}, factor: {}, func: {}, ", self.component_type.as_str(), self.factor, self.func);
        if self.id != 0 {
            result.push_str(&format!("id: {}, ", self.id));
        }
        if !self.func_full_name.is_empty() {
            result.push_str(&format!("func_full_name: {}, ", self.func_full_name));
        }
        result.push_str("\n    ");
        if !self.param_positional.is_empty() {
            result.push_str(&format!("param_positional: {:?}, ", self.param_positional));
        }
        if !self.param_keyword.is_empty() {
            result.push_str(&format!("param_keyword: {:?}, ", self.param_keyword));
        }
        if !self.reference.is_empty() {
            result.push_str(&format!("reference: {:?}", self.reference));
        } else {
            result.push_str("reference: Not available yet");
        }
        result
    }

    pub fn has_parameter(&self) -> bool {
        !self.param_keyword.is_empty() || !self.param_positional.is_empty()
    }

    pub fn is_unknown(&self) -> bool {
        self.component_type == ComponentType::Unknown
    }

    pub fn is_nonlibxc(&self) -> bool {
        // HF, PT2, RPA, SBGE2
        self.component_type != ComponentType::Libxc && self.component_type != ComponentType::Unknown
    }

    pub fn check_whitelist(&mut self, functype: &str) -> &mut Self {
        // check whitelist
        if WHITELIST_NONLIBXC.contains_key(self.func.as_str()) {
            self.component_type = WHITELIST_NONLIBXC.get(self.func.as_str()).unwrap().clone();
            return self;
        }
        // check pre-defined codes
        if CODES.contains_key(self.func.as_str()) {
            let code = CODES.get(self.func.as_str()).unwrap();
            self.func_full_name = get_name(*code);
            self.id = *code;
            self.component_type = ComponentType::Libxc;
            return self;
        }
        return self;
    }
    pub fn to_valid_name(&mut self, functype: &str) -> &mut Self {
        // search libxc full name
        // let prefix = format!("{}_", functype);
        let prefix = format!("{}_", functype);
        let illegal_prefix = ILLEGAL_SHORT_PREFIX.get(functype).unwrap();
        let (possible_complete_prefix, illegal_complete_prefix) = get_possible_prefix(&functype);
        // condition 1
        // check if self.func starts with any of the possible_complete_prefix
        for p in possible_complete_prefix.iter() {
            if self.func.starts_with(p) {
                self.func_full_name = self.func.clone();
                if let Some(id) = AVAIL_FUNC.get(&self.func_full_name) {
                    self.id = *id;
                    self.component_type = ComponentType::Libxc;
                } else {
                    panic!("Error: functional {} not found in libxc", self.func_full_name);
                }
                break;
            } 
        }
        for p in illegal_complete_prefix.iter() {
            if self.func.starts_with(p) {
                panic!("Error: functional {} has illegal prefix for type {}", self.func, functype);
            }
        }
        // end of condition 1

        // condition 2,3
        // check if self.func starts with any of the illegal_prefix
        for p in illegal_prefix.iter() {
            if self.func.starts_with(p) {
                panic!("Error: functional {} has illegal prefix for type {}", self.func, functype);
            }
        }
        // check if self.func starts with any short prefix 
        let mut possible_full_names = Vec::new();
        for (short, prefixes) in POSSIBLE_PREFIX.iter() {
            let current_prefix = format!("{}_", short);
            if self.func.starts_with(current_prefix.as_str()) {
                let cutted_name = self.func.trim_start_matches(&current_prefix);
                for p in prefixes.iter() {
                    let full_name = format!("{}{}", p, cutted_name);
                    possible_full_names.push(full_name);
                }
                break;
            }
        }
        // if not, try to add prefix
        if possible_full_names.is_empty() {
            for p in possible_complete_prefix.iter() {
                let full_name = format!("{}{}", p, self.func);
                possible_full_names.push(full_name);
            }
        }      
        // println!("Possible full names for {}: {:?}", self.func, possible_full_names);
        // if self.func.starts_with(&prefix) {
        //     tmp_func = self.func.trim_start_matches(&prefix).to_string();
        // }
        // // generate all possible full names
        // let possible_full_names: Vec<String> = possible_complete_prefix.iter().map(|p| {
        //     format!("{}{}", p, tmp_func)
        // }).collect();
        // search in AVAIL_FUNC
        let matches: Vec<_> = possible_full_names.into_iter()
            .filter(|name| AVAIL_FUNC.contains_key(name))
            .collect();

        if matches.is_empty() {
            // panic!("Error: functional {} not found in libxc", tmp_func);
            // do nothing, leave id as 0
        } else if matches.len() == 1 {
            self.func_full_name = matches[0].clone();
            self.id = *AVAIL_FUNC.get(&self.func_full_name).unwrap();
            self.component_type = ComponentType::Libxc;
        } else {
            panic!("Error: functional {} is ambiguous, possible matches: {:?}", self.func, matches);
        }
        // end of condition 2,3

        // println!("After to_valid_name: {}", self.formatted_output());

        self.reference = self.get_reference();
        
        return self;
    }

    // #[cached]
    pub fn get_hybrid(&self, spin_channel: usize) -> f64 {
        if self.is_rsh() {
            let (omega, alpha, beta) = self.get_rsh(spin_channel);
            return self.factor * (alpha + beta);
        }
        if self.component_type == ComponentType::HF {
            return self.factor;
        } else if self.component_type == ComponentType::Libxc {
            let xcfunc = xc_func_init(self.id, spin_channel);
            let hybrid_coef = if let Some((_omega, alpha, beta)) = xcfunc.cam_coef() {
                alpha + beta // for RSH, return alpha + beta (matches pyscf)
            } else {
                // for non-RSH, return hybrid coefficient; if not available, return 0.0
                xcfunc.hyb_exx_coef().unwrap_or(0.0)
            };
            let hyb =  self.factor * hybrid_coef;
            return hyb;
        } else {
            return 0.0;
        }
    }

    pub fn get_rsh(&self, spin_channel: usize) -> (Option<f64>, f64, f64) {
        if !self.is_rsh() {
            return (None, self.get_hybrid(spin_channel), 0.0);
        }
        if self.component_type == ComponentType::Libxc {
            let xcfunc = xc_func_init(self.id, spin_channel);
            if let Some((omega, alpha, beta)) = xcfunc.cam_coef() {
                return (Some(omega), alpha, beta);
            } else {
                panic!("Error: component {} is detected as RSH but cam_coef is not available in libxc", self.func);
            }
        } else if self.component_type == ComponentType::RSHF {
            let param = &self.param_positional;
            match self.func.as_str() {
                "LR_HF" | "SR_HF" => {
                    if param.len() != 1 {
                        panic!("Error: SR/LR HF component with func {} should have exactly 1 positional parameter, but got {}", self.func, param.len());
                    }
                },
                "RSH" => {
                    if param.len() != 3 {
                        panic!("Error: RSH component should have exactly 3 positional parameters (omega, alpha, beta), but got {}", param.len());
                    }
                },
                _ => {},
            }
            match self.func.as_str() {
                "LR_HF" => return (Some(param[0]), self.factor, -1.0*self.factor), 
                "SR_HF" => return (Some(param[0]), 0.0, self.factor),
                "RSH" => return (Some(param[0]), param[1], param[2]), 
                _ => return (None, 0.0, 0.0),
            }
        } else {
            return (None, 0.0, 0.0);
        }
    }

    /// `(omega, c_LR, c_SR)` of a component in the consumer convention of
    /// `DFA4REST::get_rsh_libxc` / `DFA4REST::dfa_rsh_scf`: `c_LR` is the long-range
    /// (full-range) HF fraction and `c_SR` the short-range fraction. Per libxc's
    /// convention (xc.h) `cam_alpha` is the full HF fraction and `cam_beta` the
    /// short-range-only extra (SR = cam_alpha + cam_beta, LR = cam_alpha), so the raw
    /// `get_rsh` slots are `(c_LR, c_SR - c_LR)` for every component type and the
    /// conversion below just restores the absolute `c_SR`. Non-RSH components carry
    /// their (range-independent) hybrid in both slots.
    pub fn get_rsh_lrsr(&self, spin_channel: usize) -> (Option<f64>, f64, f64) {
        let (omega, c_lr, c_sr_minus_lr) = self.get_rsh(spin_channel);
        (omega, c_lr, c_sr_minus_lr + c_lr)
    }

    pub fn is_rsh(&self) -> bool {
        match self.component_type {
            ComponentType::RSHF => {
                true
            },
            ComponentType::Libxc => {
                let xcfunc = xc_func_init(self.id, 1);
                xcfunc.is_hyb_cam()
            },
            _ => false,
        }
    }

    pub fn use_laplacian(&self) -> bool {
        match self.component_type {
            ComponentType::Libxc => {
                let xcfunc = xc_func_init(self.id, 1);
                xcfunc.needs_laplacian()
            },
            _ => false,
        }
    }

    pub fn get_reference(&self) -> Vec<String> {
        if self.component_type == ComponentType::Libxc && self.id != 0 {
            let xcfunc = xc_func_init(self.id, 1);
            return xcfunc.references().iter().map(|r| r.ref_text.clone()).collect();
        } else {
            return Vec::new();
        }
    }

    // pub fn init_libxc(&self, spin_channel: usize) -> Option<XcFuncType> {
    //     if self.component_type == ComponentType::Libxc && self.id != 0 {
    //         let xcfunc = XcFuncType::xc_func_init(self.id, spin_channel);
    //         Some(xcfunc)
    //     } else {
    //         None
    //     }
    // }

    pub fn canonicalize(&self) -> Self {
        match self.component_type {
            ComponentType::RSHF => {
                return self.normalize_rsh();
            }
            ComponentType::Disp => {
                // todo: check default
                let mut comp = self.clone();
                if !self.param_positional.is_empty() {
                    panic!("Error: positional parameters for Disp component is not supported, but got {:?}", self.param_positional);
                }
                match self.func.to_uppercase().as_str() {
                    "D3" => {
                        comp.func = "DFTD3".to_string();
                    },
                    "D4" => {
                        comp.func = "DFTD4".to_string();
                    },
                    _ => {},
                }
                return comp;
            },
            ComponentType::PT2 | ComponentType::SCSRPA | ComponentType::SBGE2 => {
                return self.normalize_pt2_param();
            },
            _ => {},
        }
        self.clone()
    }

    pub fn normalize_rsh(&self) -> Self {
        assert!(self.component_type == ComponentType::RSHF);
        let mut new_component = self.clone();
        let (omega, alpha, beta) = self.get_rsh(1);
        new_component.func = "RSH".to_string();
        new_component.param_positional = vec![omega.unwrap(), alpha, beta];
        new_component.factor = 1.0;
        new_component
    }

    pub fn normalize_pt2_param(&self) -> Self {
        let mut new_component = DFAComponent::new(1.0, self.func.clone());
        match self.component_type {
            ComponentType::PT2 | ComponentType::SCSRPA | ComponentType::SBGE2 => {},
            _ => { panic!("Error: only PT2, SCSRPA, and SBGE2 components can be normalized, but got {}", self.component_type.as_str()); },
        }

        match self.func.to_uppercase().as_str() {
            "MP2" | "SCSRPA" | "SBGE2" => {
                // panic if have both positional and keyword parameters
                if !self.param_positional.is_empty() && !self.param_keyword.is_empty() {
                    panic!("Error: PT2 component cannot have both positional and keyword parameters");
                }
                if !self.param_keyword.is_empty() {
                    let mut new_param = vec![0.0; 2];
                    for (k, v) in self.param_keyword.iter() {
                        match k.as_str() {
                            "os" => new_param[0] = v.as_f64().unwrap() * self.factor,
                            "ss" => new_param[1] = v.as_f64().unwrap() * self.factor,
                            _ => panic!("Error: unknown parameter {} for PT2 component", k),
                        }
                    }
                    new_component.param_positional = new_param;
                }                
                if !self.param_positional.is_empty() {
                    if self.param_positional.len() != 2 {
                        panic!("Error: PT2 component should have 2 positional parameters, but got {}", self.param_positional.len());
                    }
                    // multiply factor to parameters
                    let new_param = self.param_positional.iter().map(|p| p * self.factor).collect();
                    new_component.param_positional = new_param;
                } else {
                    // if no parameter, set to default value 1.0
                    new_component.param_positional = vec![self.factor, self.factor];
                }

            },
            "MP2_OS" => {
                if !self.param_positional.is_empty() || !self.param_keyword.is_empty() {
                    panic!("Error: MP2_OS component should not have parameters");
                }
                new_component.func = "MP2".to_string();
                new_component.param_positional = vec![self.factor, 0.0];
            },
            "MP2_SS" => {
                if !self.param_positional.is_empty() || !self.param_keyword.is_empty() {
                    panic!("Error: MP2_SS component should not have parameters");
                }
                new_component.func = "MP2".to_string();
                new_component.param_positional = vec![0.0, self.factor];
            },
            _ => panic!("Error: unknown PT2 component type {}", self.func),
        }

        new_component.component_type = self.component_type.clone();
        new_component

    }

    pub fn is_normalized(&self) -> bool {
        match self.component_type {
            ComponentType::PT2 | ComponentType::SCSRPA | ComponentType::SBGE2 => {
                self.factor == 1.0
            },
            ComponentType::RSHF => {
                self.factor == 1.0 && self.func == "RSH" && self.param_positional.len() == 3
            },
            _ => panic!("Error: only PT2, SCSRPA, SBGE2, and RSHF components can be checked for normalization, but got {}", self.component_type.as_str()),
        }
    }

}

trait Addable {
    fn is_addable_with(&self, other: &Self) -> bool;
}

impl Addable for DFAComponent {
    fn is_addable_with(&self, other: &Self) -> bool {
        if self.component_type != other.component_type {
            return false;
        }
        if self.id != other.id {
            return false;
        }
        // cannot add if any of them has xcfunc initialized
        // if self.xcfunc.is_some() || other.xcfunc.is_some() {
        //     return false;
        // }
        // todo: more precise check for parameters
        if self.param_keyword != other.param_keyword {
            return false;
        }
        if self.param_positional != other.param_positional {
            match self.component_type {
                ComponentType::PT2 | ComponentType::SCSRPA => {
                    if self.is_normalized() && other.is_normalized() {
                        return true;
                    } else {
                        return false;
                    }
                },
                ComponentType::RSHF => {
                    if self.is_normalized() && other.is_normalized() && self.param_positional[0] == other.param_positional[0] {
                        return true;
                    } else {
                        return false;
                    }
                }
                _ => {
                    return false;
                },
            }
        }
        true
    }
}

impl std::ops::Add for DFAComponent {
    type Output = DFAComponent;

    fn add(self, other: DFAComponent) -> DFAComponent {
        // if !self.is_addable_with(&other) {
        //     panic!("Error: cannot add two different DFAComponents");
        // }
        match self.component_type {
            ComponentType::PT2 | ComponentType::SCSRPA => {
                        DFAComponent {
            factor: 1.0, // should be normalized before add
            func: self.func.clone(),
            func_full_name: self.func_full_name.clone(),
            id: self.id,
            // direct add positional parameters
            param_positional: self.param_positional.iter().zip(other.param_positional.iter()).map(|(a, b)| a + b).collect(),
            param_keyword: self.param_keyword.clone(), // should be empty
            component_type: self.component_type.clone(),
            // xcfunc: None,
            reference: self.reference.clone(), // todo: check if reference is the same
                }
            },
            ComponentType::RSHF => {
                DFAComponent {
                    factor: 1.0, // should be normalized before add
                    func: self.func.clone(),
                    func_full_name: self.func_full_name.clone(),
                    id: self.id,
                    // direct add positional parameters
                    param_positional: {
                        let mut p = self.param_positional.clone();
                        p[1] += other.param_positional[1];
                        p[2] += other.param_positional[2];
                        p
                    },
                    param_keyword: self.param_keyword.clone(), // should be empty
                    component_type: self.component_type.clone(),
                    // xcfunc: None,
                    reference: self.reference.clone(), // todo: check if reference is the same
                }
            },
            _ => {
        DFAComponent {
            factor: self.factor + other.factor,
            func: self.func.clone(),
            func_full_name: self.func_full_name.clone(),
            id: self.id,
            param_positional: self.param_positional.clone(),
            param_keyword: self.param_keyword.clone(),
            component_type: self.component_type.clone(),
            // xcfunc: None,
            reference: self.reference.clone(), // todo: check if reference is the same
        }
            }
        }
    }
    
}

impl DFAComponent {
    /// `(libxc functional code, scale factor)` of a Libxc component; `None` for the
    /// non-libxc component types (HF/RSHF/PT2/RPA/SCSRPA/SBGE2/Disp/Unknown).
    pub fn libxc_code_param(&self) -> Option<(usize, f64)> {
        match self.component_type {
            ComponentType::Libxc => Some((self.id, self.factor)),
            _ => None,
        }
    }

    /// Initialized libxc functional of a Libxc component (spin per `spin_channel`:
    /// 1 = unpolarized, 2 = polarized); `None` for other component types.
    pub fn to_libxc_functional(&self, spin_channel: usize) -> Option<LibXCFunctional> {
        self.libxc_code_param()
            .map(|(code, _factor)| xc_func_init(code, spin_channel))
    }
}

#[derive(Clone)]
pub struct DFAdef {
    pub xc_scf: Option<Vec<DFAComponent>>,
    pub xc_nscf: Option<Vec<DFAComponent>>,
    pub reference: Vec<String>,
    pub spin_channel: usize,
    pub dfa_hybrid_scf: f64,
    // (omega, alpha, beta) in libxc convention; note beta is usually not used in computation.
    pub dfa_rsh_scf: (Option<f64>, f64, f64),
    pub dfa_hybrid_nscf: Option<f64>,
}


impl DFAdef {
    pub fn new() -> Self {
        DFAdef {
            xc_scf: None,
            xc_nscf: None,
            reference: Vec::new(),
            spin_channel: 1,
            dfa_hybrid_scf: 0.0,
            dfa_rsh_scf: (None, 0.0, 0.0),
            dfa_hybrid_nscf: None,
        }
    }

    /// Reconstruct a DFAdef record from a (full) DFA4REST, e.g. the output of the legacy
    /// parser or the deep-learning constructor.
    ///
    /// This is the transitional bridge of the DFAdef-primary refactor: those constructors
    /// keep producing a full DFA4REST that is used directly as `Molecule.xc_data`, while
    /// `Molecule.dfadef` stores this reconstructed definition. The reconstruction is exact
    /// over the DFA4REST fields except `nlc_vv10`, which has no DFAdef counterpart here
    /// (legacy VV10 keeps flowing through `xc_data.nlc_vv10`).
    pub fn from_dfa4rest(dfa: &DFA4REST) -> DFAdef {
        let libxc_comp = |id: usize, factor: f64| DFAComponent {
            factor,
            func: get_name(id),
            func_full_name: get_name(id),
            id,
            param_positional: Vec::new(),
            param_keyword: HashMap::new(),
            component_type: ComponentType::Libxc,
            reference: Vec::new(),
        };
        let mut dfadef = DFAdef::new();
        dfadef.spin_channel = dfa.spin_channel;
        dfadef.dfa_hybrid_scf = dfa.dfa_hybrid_scf;
        dfadef.dfa_rsh_scf = (
            dfa.dfa_rsh_scf.map(|(omega, _, _)| omega),
            dfa.dfa_rsh_scf.map_or(0.0, |(_, alpha, _)| alpha),
            dfa.dfa_rsh_scf.map_or(0.0, |(_, _, beta)| beta),
        );
        dfadef.xc_scf = Some(
            dfa.dfa_compnt_scf
                .iter()
                .zip(dfa.dfa_paramr_scf.iter())
                .map(|(&id, &factor)| libxc_comp(id, factor))
                .collect(),
        );
        if let Some(family) = &dfa.dfa_family_pos {
            let (component_type, func) = match family {
                DFAFamily::PT2 => (ComponentType::PT2, "MP2".to_string()),
                DFAFamily::RPA => (ComponentType::RPA, "RPA".to_string()),
                DFAFamily::SCSRPA => (ComponentType::SCSRPA, "SCSRPA".to_string()),
                DFAFamily::SBGE2 => (ComponentType::SBGE2, "SBGE2".to_string()),
                other => panic!(
                    "DFAdef::from_dfa4rest: family {other:?} is not a fifth-rung family"
                ),
            };
            let empty_vec: Vec<f64> = Vec::new();
            let mut nscf_components: Vec<DFAComponent> = dfa
                .dfa_compnt_pos
                .as_ref()
                .unwrap_or(&Vec::new())
                .iter()
                .zip(dfa.dfa_paramr_pos.as_ref().unwrap_or(&empty_vec).iter())
                .map(|(&id, &factor)| libxc_comp(id, factor))
                .collect();
            nscf_components.push(DFAComponent {
                factor: 1.0,
                func,
                func_full_name: String::new(),
                id: 0,
                param_positional: dfa.dfa_paramr_adv.clone().unwrap_or_default(),
                param_keyword: HashMap::new(),
                component_type,
                reference: Vec::new(),
            });
            dfadef.xc_nscf = Some(nscf_components);
            dfadef.dfa_hybrid_nscf = dfa.dfa_hybrid_pos;
        }
        dfadef
    }

    /// Backend shared by [`Self::to_dfa4rest`] and [`Self::to_xc_data`]: the SCF-side
    /// DFA4REST (libxc components, hybrid, RSH, VV10) with no post-SCF state.
    fn to_scf_xc_data(&self) -> DFA4REST {
        let mut xc_data = DFA4REST::new_xc(self.spin_channel, 0);
        if let Some(scf_components) = &self.xc_scf {
            for comp in scf_components {
                match comp.component_type {
                    ComponentType::Libxc => {
                        xc_data.dfa_compnt_scf.push(comp.id);
                        xc_data.dfa_paramr_scf.push(comp.factor);
                    },
                    _ => {},
                }
            }
        }
        xc_data.dfa_hybrid_scf = self.dfa_hybrid_scf;
        // An explicit VV10 dispersion component (e.g. from the `-v` suffix of
        // `wb97x-v`) activates non-local correlation. Its (b, C) parameters are
        // read from the libxc VV10-flagged component, not from user keywords.
        if let Some(disp) = self.get_dispersion() {
            if disp.func.eq_ignore_ascii_case("VV10") {
                // The VV10 (b, C) parameters come from the single libxc
                // functional that carries the VV10 kernel. This covers both
                // combined functionals (466/531 for wb97x-v/wb97m-v, where the
                // semilocal part and VV10 live in one functional) and composite
                // forms such as `scan-vv10` (263 MGGA_X_SCAN + 584
                // MGGA_C_SCAN_VV10): in both cases exactly one component has the
                // VV10 flag. The VV10 energy is evaluated separately on the NLC
                // grid, so the internal correlation bundled in 584 is not
                // double counted.
                if let Some(scf_components) = &self.xc_scf {
                    // Exactly one libxc component must carry the VV10 kernel
                    // (e.g. 466/531/584); the VV10 Disp component is not
                    // counted here.
                    let nlc_comps: Vec<_> = scf_components
                        .iter()
                        .filter(|comp| comp.component_type == ComponentType::Libxc && comp.is_nlc())
                        .collect();
                    if nlc_comps.len() == 1 {
                        let comp = nlc_comps[0];
                        if (comp.factor - 1.0).abs() > 1.0e-12 {
                            panic!("VV10 component with non-unit factor is not supported: {}", comp.func_full_name);
                        }
                        xc_data.nlc_vv10 = xc_func_init(comp.id, 1).vv10_coef();
                    } else {
                        panic!("VV10 requires exactly one libxc functional carrying the VV10 kernel.");
                    }
                }
            }
        }
        let alpha = self.dfa_rsh_scf.1;
        let beta = self.dfa_rsh_scf.2;
        if let Some(omega) = self.dfa_rsh_scf.0 {
            xc_data.dfa_rsh_scf = Some((omega, alpha, beta));
        } else {
            xc_data.dfa_rsh_scf = None;
        }
        xc_data
    }

    pub fn to_dfa4rest(&self) -> DFA4REST {
        let mut xc_data = self.to_scf_xc_data();
        if let Some(nscf_components) = &self.xc_nscf {
            let mut dfa_compnt_pos = Vec::new();
            let mut dfa_paramr_pos = Vec::new();
            let mut got_pos_family = false;
            for comp in nscf_components {
                
                match comp.component_type {
                    ComponentType::Libxc => {
                        dfa_compnt_pos.push(comp.id);
                        dfa_paramr_pos.push(comp.factor);
                    },
                    ComponentType::PT2 | ComponentType::RPA | ComponentType::SCSRPA | ComponentType::SBGE2 => {
                        if got_pos_family {
                            panic!("Error: more than one fifth rung functional in final energy functional, which is not supported in DFA4REST");
                        }
                        got_pos_family = true;
                        xc_data.dfa_family_pos = Some(comp.component_type.to_dfa_family().unwrap());
                        // let new_comp = comp.normalize_pt2_param();
                        xc_data.dfa_paramr_adv = Some(comp.param_positional.clone());
                    },
                    _ => {},
                }
            }
            xc_data.dfa_compnt_pos = Some(dfa_compnt_pos);
            xc_data.dfa_paramr_pos = Some(dfa_paramr_pos);
            xc_data.dfa_hybrid_pos = self.dfa_hybrid_nscf;
        }
        xc_data
    }

    /// Narrowed SCF exc/vxc [`DFA4REST`]: same SCF-side content as [`Self::to_dfa4rest`]
    /// but carrying no fifth-rung state (`dfa_family_pos`/`dfa_compnt_pos`/`dfa_paramr_pos`/
    /// `dfa_hybrid_pos`/`dfa_paramr_adv` stay at defaults); the post-SCF definition remains
    /// in `xc_nscf` and is read through the DFAdef accessors (`family_pos`, `fifth_rung_component`, ...).
    pub fn to_xc_data(&self) -> DFA4REST {
        self.to_scf_xc_data()
    }

    /// The fifth-rung (post-SCF) family carried by the final-energy components
    /// (`xc_nscf`); `None` for a purely variational SCF functional.
    pub fn family_pos(&self) -> Option<DFAFamily> {
        self.xc_nscf
            .as_ref()?
            .iter()
            .find_map(|comp| comp.component_type.to_dfa_family())
    }

    /// `true` when the SCF functional evaluates DFT XC on a grid (has at least one
    /// libxc component), mirroring `DFA4REST::is_dfa_scf`.
    pub fn is_dfa_scf(&self) -> bool {
        self.xc_scf
            .as_ref()
            .map_or(false, |comps| comps.iter().any(|comp| comp.component_type == ComponentType::Libxc))
    }

    /// `true` only for the RPA family (SCSRPA does not count), mirroring `DFA4REST::is_rpa`.
    pub fn is_rpa(&self) -> bool {
        matches!(self.family_pos(), Some(DFAFamily::RPA))
    }

    /// The fifth-rung component of the final-energy functional - the single component
    /// of type PT2/RPA/SCSRPA/SBGE2, matching what `to_dfa4rest` projects into
    /// `dfa_paramr_adv`. Its `param_positional` holds the family's auxiliary factors
    /// (for PT2/SCSRPA/SBGE2: the opposite-spin / same-spin scaling `(c_os, c_ss)`).
    pub fn fifth_rung_component(&self) -> Option<&DFAComponent> {
        self.xc_nscf
            .as_ref()?
            .iter()
            .find(|comp| comp.component_type.to_dfa_family().is_some())
    }

    /// Override the PT2 `(c_os, c_ss)` factors (e.g. from the `ri_pt2` `os_factor`/`ss_factor`
    /// control keywords). No-op when the definition carries no PT2 component; a component
    /// without positional parameters defaults to `[1.0, 1.0]`.
    pub fn update_pt2_params(&mut self, os_factor: Option<f64>, ss_factor: Option<f64>) {
        let comps = match self.xc_nscf.as_mut() {
            Some(comps) => comps,
            None => return,
        };
        let idx = match comps.iter().position(|comp| comp.component_type == ComponentType::PT2) {
            Some(idx) => idx,
            None => return,
        };
        let mut params = std::mem::take(&mut comps[idx].param_positional);
        if params.is_empty() {
            params = vec![1.0, 1.0];
        }
        if let Some(osf) = os_factor {
            params[0] = osf;
        }
        if let Some(ssf) = ss_factor {
            params[1] = ssf;
        }
        comps[idx].param_positional = params;
    }

    pub fn formatted_output(&self) -> String {
        let mut result = String::new();
        if let Some(scf_components) = &self.xc_scf {
            result.push_str("SCF components:\n");
            for comp in scf_components {
                result.push_str(&format!("  {}\n", comp.formatted_output()));
            }
        }
        // println!("Info for SCF functional:");
        // let hyb_0 = self.get_hybrid_scf(1);
        result.push_str(&format!("  Total hybrid: {}\n", self.dfa_hybrid_scf));
        if self.is_rsh() {
            result.push_str(&format!("  RSH parameters (omega, alpha, beta): {} {} {}\n", self.dfa_rsh_scf.0.unwrap(), self.dfa_rsh_scf.1, self.dfa_rsh_scf.2));
        }
        if let Some(nscf_components) = &self.xc_nscf {
            result.push_str("Final energy components:\n");
            for comp in nscf_components {
                result.push_str(&format!("  {}\n", comp.formatted_output()));
            }
            result.push_str(&format!("  Total hybrid: {}\n", self.dfa_hybrid_nscf.unwrap()));
        }
        if !self.reference.is_empty() {
            result.push_str("References:\n");
            for r in &self.reference {
                result.push_str(&format!("  {}\n", r));
            }
        }
        result
    }
    
    // #[cached]
    pub fn get_hybrid_scf(&self, spin_channel: usize) -> f64 {
        // sum up all hybrid components in self.xc
        if let Some(components) = &self.xc_scf {
            components.iter()
                .map(|comp| comp.get_hybrid(spin_channel))
                .sum()
        } else {
            0.0
        }
    }

    pub fn get_rsh_scf(&self, spin_channel: usize) -> (Option<f64>, f64, f64) {
        // check if any component in self.xc is RSH, if so, return (omega, c_LR, c_SR)
        // in the consumer convention of DFA4REST::dfa_rsh_scf / rsh_params (the same
        // normalization as DFA4REST::get_rsh_libxc); see DFAComponent::get_rsh_lrsr.
        let mut result = (None, 0.0, 0.0);
        if let Some(components) = &self.xc_scf {
            for comp in components.iter() {
                let (omega, c_lr, c_sr) = comp.get_rsh_lrsr(spin_channel);
                // println!("Component {}: omega = {:?}, c_LR = {}, c_SR = {}", comp.func, omega, c_lr, c_sr);
                // only if c_LR and c_SR are both close to zero, we consider it as not range-separated (pure zero).
                if c_lr.abs() < 1e-10 && c_sr.abs() < 1e-10 {
                    continue;
                }
                if let Some(omega) = omega {
                    if result.0.is_some() {
                        if result.0.unwrap() != omega {
                            panic!("Multiple omega detected in the DFA components for SCF. Currently this is not supported.");
                        }
                    } else {
                        result.0 = Some(omega);
                    }
                }
                result.1 += c_lr;
                result.2 += c_sr;
            }
        }
        result
    }

    pub fn get_hybrid_nscf(&self, spin_channel: usize) -> f64 {
        // sum up all hybrid components in self.xc
        if let Some(components) = &self.xc_nscf {
            components.iter()
                .map(|comp| comp.get_hybrid(spin_channel))
                .sum()
        } else {
            0.0
        }
    }

    pub fn is_hybrid(&self) -> bool {
        // todo: what if not initialized?
        self.dfa_hybrid_scf.abs() >= 1e-6
    }

    pub fn is_rsh(&self) -> bool {
        if let Some(scf_components) = &self.xc_scf {
            for comp in scf_components {
                if comp.is_rsh() {
                    return true;
                }
            }
        }
        if let Some(nscf_components) = &self.xc_nscf {
            for comp in nscf_components {
                if comp.is_rsh() {
                    return true;
                }
            }
        }
        false
    }

    pub fn use_laplacian(&self) -> bool {
        if let Some(scf_components) = &self.xc_scf {
            for comp in scf_components {
                if comp.use_laplacian() {
                    return true;
                }
            }
        }
        if let Some(nscf_components) = &self.xc_nscf {
            for comp in nscf_components {
                if comp.use_laplacian() {
                    return true;
                }
            }
        }
        false
    }


    pub fn is_fifth_dfa(&self) -> bool {
        // check if any component in self.xc is a fifth rung functional
        if let Some(components) = &self.xc_nscf {
            for comp in components.iter() {
                match comp.component_type {
                    ComponentType::PT2 | ComponentType::RPA | ComponentType::SCSRPA | ComponentType::SBGE2 => {
                        return true;
                    },
                    _ => {},
                }
            }
        }
        false
    }

    pub fn has_nscf(&self) -> bool {
        self.xc_nscf.is_some()
    }

    pub fn summary(&self) {
        println!("{}", self.formatted_output());
    }

    pub fn check_sanity(&self) -> (bool, Vec<String>) {
        let mut err_strings = Vec::new();
        if self.use_laplacian() {
            err_strings.push("Error: functionals that use Laplacian are not supported in current implementation".to_string());
        }
        (err_strings.is_empty(), err_strings)
    }

    // For calculation api, muted

    // pub fn init_libxc(&mut self) -> &mut Self {
    //     for comp in self.xc.iter_mut() {
    //         comp.init_libxc(self.spin_channel);
    //     }
    //     self
    // }
    // pub fn nscf_xcfunc_iter(&self) -> impl Iterator<Item = XcFuncType> + '_ {
    //     self.xc_nscf.as_ref().into_iter()
    //         .flat_map(|components| {
    //             components.iter()
    //                 .filter_map(|component| component.init_libxc(self.spin_channel))
    //         })
    // }

    // pub fn scf_xcfunc_iter(&self) -> impl Iterator<Item = XcFuncType> + '_ {
    //     self.xc_scf.as_ref().into_iter()
    //         .flat_map(|components| {
    //             components.iter()
    //                 .filter_map(|component| component.init_libxc(self.spin_channel))
    //         })
    // }

    // pub fn use_density_gradient(&self) -> bool {
    //     let scf_use_rhog = self.scf_xcfunc_iter().any(
    //         |xcfunc| xcfunc.use_density_gradient());
    //     let nscf_use_rhog = self.nscf_xcfunc_iter().any(
    //         |xcfunc| xcfunc.use_density_gradient());
    //     scf_use_rhog || nscf_use_rhog
    // }
}

lazy_static! {
    static ref POSSIBLE_PREFIX: HashMap<&'static str, Vec<&'static str>> = HashMap::from([
        ("X", vec!["LDA_X_", "GGA_X_", "MGGA_X_", "HYB_GGA_X_", "HYB_MGGA_X_"]),
        ("C", vec!["LDA_C_", "GGA_C_", "MGGA_C_"]),
        ("XC", vec!["LDA_XC_", "GGA_XC_", "MGGA_XC_", "HYB_LDA_XC_", "HYB_GGA_XC_", "HYB_MGGA_XC_"]),
    ]);
    static ref ILLEGAL_SHORT_PREFIX: HashMap<&'static str, Vec<&'static str>> = HashMap::from([
        ("X", vec!["C_", "XC_"]),
        ("C", vec!["X_", "XC_"]),
        // ("XC", vec!["X_", "C_"]),
        ("XC", vec![]),
        ("any", vec![]),
    ]);
}

pub fn get_possible_prefix(functype: &str) -> (Vec<&'static str>, Vec<&'static str>) {
    let mut possible_prefix = Vec::new();
    let mut illegal_prefix = Vec::new();
    if functype == "X" {
        possible_prefix = POSSIBLE_PREFIX.get("X").unwrap().clone();
        illegal_prefix = POSSIBLE_PREFIX.get("C").unwrap().clone();
        illegal_prefix.extend(POSSIBLE_PREFIX.get("XC").unwrap().clone());
    } else if functype == "C" {
        possible_prefix = POSSIBLE_PREFIX.get("C").unwrap().clone();
        illegal_prefix = POSSIBLE_PREFIX.get("X").unwrap().clone();
        illegal_prefix.extend(POSSIBLE_PREFIX.get("XC").unwrap().clone());
    } else if functype == "XC" {
        possible_prefix = POSSIBLE_PREFIX.get("XC").unwrap().clone();
        // illegal_prefix = POSSIBLE_PREFIX.get("X").unwrap().clone();
        // illegal_prefix.extend(POSSIBLE_PREFIX.get("C").unwrap().clone());
        possible_prefix.extend(POSSIBLE_PREFIX.get("X").unwrap().clone());
        possible_prefix.extend(POSSIBLE_PREFIX.get("C").unwrap().clone());
    } else if functype == "any" {
        possible_prefix = POSSIBLE_PREFIX.get("X").unwrap().clone();
        possible_prefix.extend(POSSIBLE_PREFIX.get("C").unwrap().clone());
        possible_prefix.extend(POSSIBLE_PREFIX.get("XC").unwrap().clone());
    } else {
        panic!("Error: unknown functype {}", functype);
    }
    (possible_prefix, illegal_prefix)
}

// pub fn check_type_sanity(name: &str, functype: &str) -> bool {
//     let mut sanity = true;
//     let parts: Vec<&str> = name.split(',').collect();
//     let n_part_notempty = parts.iter().filter(|p| !p.trim().is_empty()).count();
//     if (functype == "X" || functype == "C") && n_part_notempty > 1 {
//         sanity =  false;
//     }
//     sanity
// }


pub fn parse_tokens(mut components: Vec<DFAComponent>, functype: &str) -> Vec<DFAComponent> {
    let mut result = Vec::new();
    let allow_alias = functype == "XC" || functype == "any";
    for comp in components.iter_mut() {
        // comp.to_valid_name(functype);
        comp.check_whitelist(functype);
        
        if comp.is_nonlibxc() {
            result.push(comp.canonicalize());
        } else if allow_alias && ALIAS.contains_key(comp.func.as_str()) {
            if comp.has_parameter() {
                panic!("Error: functional {} has parameters, cannot be filtered by alias", comp.func);
            }
            let alias_xc = ALIAS.get(comp.func.as_str()).unwrap();
            // check if X func is aliased to XC
            // if !check_type_sanity(alias_xc, functype) {
            //     panic!("Error: functional {} is aliased to a different type, which is not allowed in type {}", comp.func, functype);
            // }
            let alias_components = parse_1step(alias_xc);
            for mut alias_comp in alias_components {
                alias_comp.factor *= comp.factor;
                result.push(alias_comp);
            }   
        } else {
            comp.to_valid_name(functype);
            if !comp.is_unknown() {
                result.push(comp.clone());
            } else {
                panic!("Error: functional {} not found in libxc and not in alias/whitelist", comp.func);
            }
        }
    }
    // for comp in result.iter_mut() {
    //     comp.to_valid_name(functype);
    // }
    result
}

// pub enum ParsedResult {
//     VecDFAComponent(Vec<DFAComponent>),
//     DFA2step(DFA2step),
// }

/// `(libxc code, factor)` pairs of the Libxc components, in order — the bridge from the
/// component-typed DFAdef world to the code-array currency of the exc/vxc/numint layer
/// (e.g. `XCData` construction in the DH gradients).
pub fn components_to_codes_params(components: &[DFAComponent]) -> (Vec<usize>, Vec<f64>) {
    components.iter().filter_map(|comp| comp.libxc_code_param()).unzip()
}

pub fn parse_and_derive(xc: &str, spin_channel: usize, print_level: usize) -> DFAdef {
    let mut dfa = parse(xc.to_lowercase().as_str());
    dfa.spin_channel = spin_channel;
    // restore intermediate variables
    // dfa.init_libxc();
    dfa.dfa_hybrid_scf = dfa.get_hybrid_scf(spin_channel);
    dfa.dfa_rsh_scf = dfa.get_rsh_scf(spin_channel);
    if dfa.has_nscf() {
        dfa.dfa_hybrid_nscf = Some(dfa.get_hybrid_nscf(spin_channel));
    }
    dfa.summary();
    dfa
}


pub fn parse(xc: &str) -> DFAdef {
    println!("Parsing xc: {}", xc);
    let mut dfa = DFAdef::new();
    // check MULTISTEP
    // let mut xc = xc.to_lowercase();
    // println!("multistep functionals: {:?}", MULTISTEP.keys().collect::<Vec<&String>>());
    let mut multistep_alias_all = MULTISTEP_NAME_WITH_DASH.clone();
    multistep_alias_all.extend(MULTISTEP_ALIAS.clone().into_iter().map(|(k, v)| (k.to_string(), v)));
    let mut multistep = MULTISTEP.clone();
    if let Some(user_functionals) = load_user_json_functionals() {
        multistep.extend(user_functionals.into_iter());
    }

    if multistep.contains_key(xc) || multistep_alias_all.contains_key(xc) {
        let steps = if multistep.contains_key(xc) {
            multistep.get(xc).unwrap()
        } else {
            let alias = multistep_alias_all.get(xc).unwrap();
            multistep.get(*alias).unwrap()
        };
        println!("Detected multi-step functional {}, which is parsed to:", xc);
        if steps.code_scf.is_empty() {
            println!("Step for SCF         : derived from final energy functional");
        } else {
            println!("Step for SCF         : {}", steps.code_scf);
        }
        println!("Step for final energy: {}", steps.code);
        // println!("Reference: {:?}", steps.reference);
        let final_components = parse_1step(&steps.code);
        let final_components_scf = if steps.code_scf.is_empty() {
            filter_out_nscf(final_components.clone())
        } else {
            parse_1step(&steps.code_scf)
        };
        // let final_components_scf = parse_1step(&steps.code_scf);
        // let mut reference = Vec::new();
        // reference.push(steps.reference.clone());
        dfa.xc_scf = Some(merge_components(final_components_scf));
        dfa.xc_nscf = Some(merge_components(final_components));
        dfa.reference = steps.reference.clone();
    } else {
        let xc1 = check_disp_suffix(xc);
        let mut final_components = parse_1step(&xc1);
        final_components = merge_components(final_components);
        dfa.xc_scf = Some(final_components);
        dfa.reference = Vec::new();
    }
    dfa
}


pub fn filter_out_nscf(components: Vec<DFAComponent>) -> Vec<DFAComponent> {
    components.into_iter().filter(|c| {
        !NSCF_COMPONENTS.contains(&c.component_type)
    }).collect()
}

pub fn parse_1step(xc: &str) -> Vec<DFAComponent> {
    // replace dash in name by searching NAME_WITH_DASH
    let mut xc = xc.to_uppercase();
    if xc.contains('-') {
        for (name_with_dash, name_without_dash) in NAME_WITH_DASH.iter() {
            // if xc.contains(name_with_dash) {
            xc = xc.replace(name_with_dash, name_without_dash);
            // }
        }
    }
    // replace non-underscore name by searching NAME_WITHOUT_UNDERSCORE
    for (name_without_underscore, name_with_underscore) in NAME_WITHOUT_UNDERSCORE.iter() {
        if xc.contains(name_without_underscore) {
            xc = xc.replace(name_without_underscore, name_with_underscore);
        }
    }
    let (xc_pass1, captures) = parse_pass1(&xc);
    let parts = parse_pass2(&xc_pass1);
    let mut final_components:Vec<DFAComponent> = Vec::new();
    if parts.len() == 2 {
        let (xfac, xfuncs, xparams) = parse_pass3(parts[0], &captures);
        let mut x_components:Vec<DFAComponent> = to_dfa_component_raw(xfac, xfuncs, xparams);
        x_components = parse_tokens(x_components, "X");
        
        let (cfac, cfuncs, cparams) = parse_pass3(parts[1], &captures);
        let mut c_components:Vec<DFAComponent> = to_dfa_component_raw(cfac, cfuncs, cparams);
        c_components = parse_tokens(c_components, "C");

        // x_components.iter().for_each(|c| {
        //     println!("X component: {}", c.formatted_output());
        // });
        // c_components.iter().for_each(|c| {
        //     println!("C component: {}", c.formatted_output());
        // });
        final_components.extend(x_components);
        final_components.extend(c_components);
    } else {
        let (xcfac, xcfuncs, xcparams) = parse_pass3(parts[0], &captures);
        let mut xc_components:Vec<DFAComponent> = to_dfa_component_raw(xcfac, xcfuncs, xcparams);
        xc_components = parse_tokens(xc_components, "XC");
        // xc_components.iter().for_each(|c| {
        //     println!("XC component: {}", c.formatted_output());
        // });
        final_components.extend(xc_components);
    }
    final_components
}

pub fn merge_components(components: Vec<DFAComponent>) -> Vec<DFAComponent> {
    let mut merged: Vec<DFAComponent> = Vec::new();
    // let mut components_n = components.clone();
    // components_n.iter_mut().for_each(|c| {
    //     if c.component_type == ComponentType::PT2 || c.component_type == ComponentType::SCSRPA {
    //         *c = c.normalize_pt2_param();
    //     }
    // });
    for comp in components {
        let mut found = false;
        
        for m in merged.iter_mut() {
            
            if m.is_addable_with(&comp) {
                *m = m.clone() + comp.clone();
                found = true;
                break;
            }
        }
        if !found {
            merged.push(comp);
        }
    }
    // get references
    // merged.iter_mut().for_each(|c| {
    //     let refs = c.get_reference();
    //     c.reference = refs;
    // });
    merged
}

pub fn parse_pass1(xc: &str) -> (String, Vec<String>) {
    // find "( )" in xc with regex and substitute with ":n"
    // record the content in "( )" in a vector
    let mut result = String::new();
    let re = regex::Regex::new(r"\((.*?)\)").unwrap();
    let mut count = 1;
    let mut last_index = 0;
    let mut captures = Vec::new();
    for cap in re.captures_iter(xc) {
        let m = cap.get(0).unwrap();
        result.push_str(&xc[last_index..m.start()]);
        result.push_str(&format!(":{}", count));
        captures.push(cap[1].to_string());
        count += 1;
        last_index = m.end();
    }
    result.push_str(&xc[last_index..]);
    // println!("pass1");
    // println!("xc: {}", result);
    // println!("captured params: {:?}", captures);

    (result, captures)
}

pub fn parse_pass2(xc: &str) -> Vec<&str> {
    // xc: "0.5*K1 + 0.5*K2:1, K3"
    // split by ","
    // panic if more than one ","
    let parts: Vec<&str> = xc.split(',').collect();
    if parts.len() > 2 {
        panic!("Error: more than one ',' in xc");
    }
    // println!("pass2");
    // println!("parts: {:?}", parts);
    parts
}

pub fn parse_pass3(xc: &str, param_captures: &Vec<String>) -> (Vec<f64>, Vec<String>, Vec<String>) {
    // xc: "21.5*K1 + 0.56*K2:1 + K3"
    // parsed to fac=[0.5, 0.5, 1.0], funcs=["K1", "K2:1", "K3"]
    let mut fac = Vec::new();
    let mut funcs = Vec::new();
    // let re = regex::Regex::new("([+-]?\\d*\\.?\\d*)\\*([A-Za-z0-9_:]+)").unwrap();
    let re = regex::Regex::new(r"([+-]?\s?\d*\.?\d*)\s?\*?\s?([A-Za-z0-9_:]+)").unwrap();
    for cap in re.captures_iter(xc) {
        //remove whitespace in cap[1]
        let cap1 = cap[1].replace(" ", "");
        let factor = if &cap1 == "" || &cap1 == "+" {
            1.0
        } else if &cap1 == "-" {
            -1.0
        } else {
            cap1.parse::<f64>().unwrap()
        };
        fac.push(factor);
        funcs.push(cap[2].to_string()//.to_uppercase()
            );
    }
    // println!("pass3");
    // println!("factors: {:?}", fac);
    // println!("funcs: {:?}", funcs);

    // fac=[0.5, 0.5, 1.0], funcs=["K1", "K2:1", "K3"], param_captures=["x=1,y=2"]
    // remove ":n" in funcs and create a new vector ["", "x=1,y=2", ""]
    let mut params = Vec::new();
    for func in funcs.iter_mut() {
        if func.contains(":") {
            if let Some((key,index)) = func.split_once(":") {
                let index: usize = index.parse().unwrap();
                if index == 0 || index > param_captures.len() {
                    panic!("Error: index out of range in func {}", func);
                }
                params.push(param_captures[index - 1].to_lowercase());
                *func = key.to_string();
            } else {
                panic!("Error: invalid func format {}", func);
            }
        } else {
            params.push(String::new());
        }
    }
    // println!("pass3.1");
    // println!("factors: {:?}", fac);
    // println!("funcs: {:?}", funcs);
    // println!("params: {:?}", params);

    (fac, funcs, params)
            
}

pub fn to_dfa_component_raw(fac: Vec<f64>, funcs: Vec<String>, params: Vec<String>) -> Vec<DFAComponent> {
    if fac.len() != funcs.len() || fac.len() != params.len() {
        panic!("Error: length of fac, funcs, params do not match");
    }
    let mut components = Vec::new();
    for i in 0..fac.len() {
        let (param_keyword, param_positional) = parse_arguments(&params[i]);
        let mut component = DFAComponent::new(fac[i], funcs[i].clone());
        component.param_keyword = param_keyword;
        component.param_positional = param_positional;
        components.push(component);
    }
    components
}

fn parse_arguments(input: &str) -> (HashMap<String, Value>, Vec<f64>) {
    // println!("Parsing arguments: {}", input);
    let re = regex::Regex::new(r"\s*(\w+\s*=\s*[^,]+)\s*|\s*([^,]+)\s*").unwrap();
    
    let mut keyword_args = HashMap::new();
    let mut positional_args = Vec::new();
    let mut has_keyword = false;
    let mut has_positional = false;

    for cap in re.captures_iter(input) {
        // println!("Captured: {:?}", cap);
        if let Some(keyword_match) = cap.get(1) {
            // Parse keyword argument
            let part = keyword_match.as_str().trim();
            if let Some((key, value)) = part.split_once('=') {
                let key = key.trim().to_string();
                // let value = value.trim().parse::<f64>().unwrap();
                let value_str = value.trim();
                let value = if let Ok(v) = value_str.parse::<f64>() {
                    Value::from(v)
                } else {
                    Value::from(value_str)
                };
                keyword_args.insert(key, value);
                has_keyword = true;
            }
        } else if let Some(positional_match) = cap.get(2) {
            let value = positional_match.as_str().trim().parse::<f64>().expect(&format!("Error: failed to parse positional argument '{}' as f64", positional_match.as_str().trim()));
            positional_args.push(value);
            has_positional = true;
        }

        // Check for mixed usage and panic if detected
        if has_keyword && has_positional {
            panic!("Mixed usage of keyword and positional arguments detected in: '{}'", input);
        }
    }

    (keyword_args, positional_args)
}

#[cfg(test)]
mod tests {
    use super::{parse_and_derive, DFA4REST, DFAComponent, DFAdef, DFAFamily, ComponentType};
    use libxc::prelude::LibXCSpin;
    use std::collections::HashMap;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1.0e-8
    }

    #[test]
    fn rsh_lrsr_rshf_decomposition() {
        let hf_comp = |factor: f64, func: &str| DFAComponent {
            factor,
            func: func.to_string(),
            func_full_name: String::new(),
            id: 0,
            param_positional: vec![0.33],
            param_keyword: HashMap::new(),
            component_type: ComponentType::RSHF,
            reference: Vec::new(),
        };
        let mut dfa = DFAdef::new();
        dfa.spin_channel = 1;
        dfa.xc_scf = Some(vec![hf_comp(0.65, "SR_HF"), hf_comp(0.19, "LR_HF")]);
        let (omega, c_lr, c_sr) = dfa.get_rsh_scf(1);
        assert_eq!(omega, Some(0.33));
        assert!(close(c_lr, 0.19), "c_LR = {c_lr}");
        assert!(close(c_sr, 0.65), "c_SR = {c_sr}");
        assert!(dfa.is_rsh());
    }

    fn assert_round_trip(original: &DFA4REST, name: &str) {
        let rt = DFAdef::from_dfa4rest(original).to_dfa4rest();
        assert_eq!(original.spin_channel, rt.spin_channel, "spin_channel for {name}");
        assert_eq!(original.dfa_compnt_scf, rt.dfa_compnt_scf, "compnt_scf for {name}");
        assert_eq!(original.dfa_paramr_scf, rt.dfa_paramr_scf, "paramr_scf for {name}");
        assert!(close(original.dfa_hybrid_scf, rt.dfa_hybrid_scf), "hybrid_scf for {name}");
        assert_eq!(original.dfa_rsh_scf, rt.dfa_rsh_scf, "rsh for {name}");
        assert_eq!(original.dfa_family_pos, rt.dfa_family_pos, "family_pos for {name}");
        assert_eq!(original.dfa_compnt_pos, rt.dfa_compnt_pos, "compnt_pos for {name}");
        assert_eq!(original.dfa_paramr_pos, rt.dfa_paramr_pos, "paramr_pos for {name}");
        match (original.dfa_hybrid_pos, rt.dfa_hybrid_pos) {
            (None, None) => {},
            (Some(a), Some(b)) => assert!(close(a, b), "hybrid_pos for {name}"),
            _ => panic!("hybrid_pos presence mismatch for {name}"),
        }
        match (&original.dfa_paramr_adv, &rt.dfa_paramr_adv) {
            (None, None) => {},
            (Some(a), Some(b)) => {
                assert_eq!(a.len(), b.len(), "paramr_adv len for {name}");
                for (x, y) in a.iter().zip(b.iter()) {
                    assert!(close(*x, *y), "paramr_adv for {name}");
                }
            },
            _ => panic!("paramr_adv presence mismatch for {name}"),
        }
    }

    #[test]
    fn rsh_parity_parse_xc_vs_legacy() {
        for name in ["cam-b3lyp", "lc-wpbe"] {
            let legacy = DFA4REST::new(name, 1, 0);
            let pxc = parse_and_derive(name, 1, 0).to_dfa4rest();
            assert_eq!(legacy.dfa_rsh_scf, pxc.dfa_rsh_scf, "rsh triple mismatch for {name}");
            assert!(
                close(legacy.dfa_hybrid_scf, pxc.dfa_hybrid_scf),
                "hybrid mismatch for {name}"
            );
        }
    }

    #[test]
    fn from_dfa4rest_round_trip_legacy() {
        for name in ["hf", "b3lyp", "pbe0", "cam-b3lyp", "wb97x-v", "xyg3", "xygjos", "b2plyp", "scsrpa"] {
            let original = DFA4REST::new(name, 1, 0);
            assert_round_trip(&original, name);
        }
    }

    #[test]
    fn from_dfa4rest_round_trip_deep_learning() {
        let original = DFA4REST::parse_scf_dldft(&"dl_dfa_scf".to_string(), 1);
        assert_round_trip(&original, "dl_dfa_scf");
    }

    #[test]
    fn parse_standard_functionals() {
        let dfa = parse_and_derive("hf", 1, 0);
        assert!(close(dfa.dfa_hybrid_scf, 1.0));
        assert!(dfa.family_pos().is_none());
        assert!(!dfa.is_dfa_scf());
        assert!(!dfa.is_fifth_dfa());

        let dfa = parse_and_derive("b3lyp", 1, 0);
        assert!(close(dfa.dfa_hybrid_scf, 0.2));
        assert!(dfa.is_dfa_scf());
        assert!(dfa.family_pos().is_none());

        let dfa = parse_and_derive("pbe0", 1, 0);
        assert!(close(dfa.dfa_hybrid_scf, 0.25));
        assert!(dfa.is_dfa_scf());

        let dfa = parse_and_derive("wb97x_v", 1, 0);
        assert!(dfa.is_dfa_scf());
        assert!(dfa.is_rsh());
    }

    #[test]
    fn parse_fifth_rung() {
        let dfa = parse_and_derive("xyg3", 1, 0);
        assert_eq!(dfa.family_pos(), Some(DFAFamily::PT2));
        assert!(dfa.is_fifth_dfa());
        assert!(!dfa.is_rpa());
        assert!(dfa.is_dfa_scf());
        assert!(close(dfa.dfa_hybrid_scf, 0.2));
        assert!(close(dfa.dfa_hybrid_nscf.unwrap(), 0.8033));
        let pt2 = dfa.fifth_rung_component().expect("xyg3 must carry a PT2 component");
        assert_eq!(pt2.param_positional.len(), 2);
        assert!(close(pt2.param_positional[0], 0.3211));
        assert!(close(pt2.param_positional[1], 0.3211));

        let dfa = parse_and_derive("dsd-blyp-d3bj", 1, 0);
        assert_eq!(dfa.family_pos(), Some(DFAFamily::PT2));
        let pt2 = dfa.fifth_rung_component().expect("dsd-blyp-d3bj must carry a PT2 component");
        assert!(close(pt2.param_positional[0], 0.47));
        assert!(close(pt2.param_positional[1], 0.40));
        assert!(dfa.has_dispersion());

        let dfa = parse_and_derive("rpa@b3lyp", 1, 0);
        assert_eq!(dfa.family_pos(), Some(DFAFamily::RPA));
        assert!(dfa.is_rpa());

        let dfa = parse_and_derive("scsrpa", 1, 0);
        assert_eq!(dfa.family_pos(), Some(DFAFamily::SCSRPA));
        assert!(dfa.is_fifth_dfa());
        assert!(!dfa.is_rpa());
        assert!(dfa.fifth_rung_component().is_some());
    }

    #[test]
    fn parse_dispersion() {
        let dfa = parse_and_derive("pbe0-d3bj", 1, 0);
        let disp = dfa.get_dispersion().expect("pbe0-d3bj must carry a Disp component");
        assert_eq!(disp.func.to_uppercase(), "DFTD3");
        assert!(!dfa.is_nlc());

        let dfa = parse_and_derive("wb97x-v", 1, 0);
        assert!(dfa.is_nlc());
        let (b, c) = dfa.vv10_params().expect("wb97x-v must carry VV10 parameters");
        assert!(b > 0.0 && c > 0.0);
    }

    #[test]
    fn update_pt2_params_overrides() {
        let mut dfa = parse_and_derive("xyg3", 1, 0);
        dfa.update_pt2_params(Some(0.5), None);
        let pt2 = dfa.fifth_rung_component().unwrap();
        assert!(close(pt2.param_positional[0], 0.5));
        assert!(close(pt2.param_positional[1], 0.3211));

        let mut dfa = parse_and_derive("xyg3", 1, 0);
        dfa.update_pt2_params(None, None);
        let pt2 = dfa.fifth_rung_component().unwrap();
        assert!(close(pt2.param_positional[0], 0.3211));
        assert!(close(pt2.param_positional[1], 0.3211));

        let mut dfa = parse_and_derive("b3lyp", 1, 0);
        dfa.update_pt2_params(Some(0.5), Some(0.5));
        assert!(dfa.fifth_rung_component().is_none());
    }

    #[test]
    fn to_xc_data_matches_to_dfa4rest_scf_side() {
        for name in ["hf", "b3lyp", "pbe0", "cam-b3lyp", "wb97x-v", "pbe0-d3bj", "xyg3"] {
            let dfa = parse_and_derive(name, 1, 0);
            let full = dfa.to_dfa4rest();
            let narrow = dfa.to_xc_data();
            assert_eq!(full.dfa_compnt_scf, narrow.dfa_compnt_scf, "compnt_scf mismatch for {name}");
            assert_eq!(full.dfa_paramr_scf, narrow.dfa_paramr_scf, "paramr_scf mismatch for {name}");
            assert!(close(full.dfa_hybrid_scf, narrow.dfa_hybrid_scf), "hybrid mismatch for {name}");
            assert_eq!(full.dfa_rsh_scf, narrow.dfa_rsh_scf, "rsh mismatch for {name}");
            assert_eq!(full.nlc_vv10, narrow.nlc_vv10, "nlc mismatch for {name}");
        }
    }

    #[test]
    fn func_list_door_matches_legacy_arrays() {
        for name in ["xyg3", "dsd-blyp-d3bj", "rpa@b3lyp"] {
            let dfa = parse_and_derive(name, 1, 0);
            let full = dfa.to_dfa4rest();
            let legacy: Option<Vec<(f64, usize)>> =
                match (&full.dfa_compnt_pos, &full.dfa_paramr_pos) {
                    (Some(code), Some(param)) if code.is_empty() => None,
                    (Some(code), Some(param)) => Some(
                        code.iter()
                            .zip(param.iter())
                            .map(|(&c, &p)| (p, c))
                            .collect(),
                    ),
                    (None, None) => None,
                    _ => panic!("pos arrays must be present or absent together"),
                };
            let door = crate::analdrv::response::rgfock_interface::dh_xc_func_list_from_components(
                dfa.xc_nscf.as_deref(),
                LibXCSpin::Unpolarized,
            )
            .map(|l| {
                l.into_iter()
                    .map(|(f, func)| (f, func.number() as usize))
                    .collect::<Vec<_>>()
            });
            assert_eq!(legacy, door, "func list mismatch for {name}");
        }
    }
}
