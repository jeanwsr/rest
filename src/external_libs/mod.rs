mod ffi_mokit;
use crate::external_libs::ffi_mokit::*;
use crate::geom_io::get_charge;
use std::ffi::{c_double, c_int, c_char, CStr, CString};
use crate::scf_io::SCF;
use crate::dft::parse_xc::{ComponentType, DFAComponent};
#[cfg(feature = "dftd3")]
use dftd3::prelude::*;
#[cfg(feature = "dftd4")]
use dftd4::prelude::*;
use toml;

pub fn py2fch(
    fchname: String,
    nbf: usize,
    nif: usize,
    eigenvector:&[f64], 
    ab: char,
    eigenvalues:&[f64],
    natorb: usize,
    gen_density: usize) 
{
    //let fchname_cstring = CString::new(fchname).expect("CString::new failed");
    let fchname_chars:&Vec<c_char> = &fchname.chars().map(|c| c as c_char).collect();
    unsafe{rest2fch_(
        //fchname_cstring.as_ptr(),
        fchname_chars.as_ptr(),
        &(fchname.len() as i32),
        &(nbf as i32),
        &(nif as i32),
        eigenvector.as_ptr(), 
        &(ab as c_char),
        eigenvalues.as_ptr(), 
        &(natorb as i32),
        &(gen_density as i32)
    )
    }
}

pub fn dftd(scf_data: &SCF) -> (f64, Option<Vec<f64>>, Option<Vec<f64>>) {
    let mut disp_from_parse_xc = false;
    if let Some(dfadef) = &scf_data.mol.dfadef {
        if dfadef.has_dispersion() {
            disp_from_parse_xc = true;
        }
    }
    let disp_from_ctrl = scf_data.mol.ctrl.empirical_dispersion.is_some();
    if disp_from_parse_xc && disp_from_ctrl {
        panic!("Empirical dispersion is specified in both xc and empirical_dispersion. Please specify it in only one place to avoid ambiguity.");
    }


    if disp_from_parse_xc {
        println!("Empirical dispersion is specified in xc. Use the dispersion from parse_xc.");
        let disp = scf_data.mol.dfadef.as_ref().unwrap().get_dispersion().unwrap();
        match disp.func.to_lowercase().as_str() {
            "dftd3"  => {
                return dftd3_atm_from_parse_xc(scf_data, &disp);
            },
            "dftd4" => {
                return dftd4_atm_from_parse_xc(scf_data, &disp);
            },
            _ => {
                panic!("Invalid input for empirical dispersion in xc: {}.", disp.func);
            }
        }
    }
    if disp_from_ctrl {
        println!("Empirical dispersion is specified in empirical_dispersion. Use the dispersion from empirical_dispersion.");
    
    // "d3" or "d3bj", then use dftd3_atm;
    // "d4" use dftd4,
    // else panic on invalid input
    if let Some(tmp_emprical) = &scf_data.mol.ctrl.empirical_dispersion {
        if tmp_emprical == "d3" || tmp_emprical == "d3bj" {
            return dftd3_atm_from_scf(scf_data);
        } else if tmp_emprical == "d4" {
            return dftd4_atm_from_scf(scf_data); 
        } else {
            panic!("Invalid input for empirical_dispersion: {}.\nDo not invoke the empirical dispersion evaluation!", tmp_emprical);
        }
    } else {
        println!("No empirical_dispersion.");
        return (0.0, None, None);
    }
    }
    (0.0, None, None)
}

#[cfg(feature = "dftd3")]
pub fn prepare_d3model(scf_data: &SCF) -> DFTD3Model {
    let numbers = get_charge(&scf_data.mol.geom.elem).iter().map(|x| *x as usize).collect::<Vec<usize>>();
    let positions = &scf_data.mol.geom.position.data;
    let lattice = None;
    let periodic = None;
    DFTD3Model::new(&numbers, positions, lattice, periodic)
}

#[cfg(feature = "dftd4")]
pub fn prepare_d4model(scf_data: &SCF) -> DFTD4Model {
        let numbers = get_charge(&scf_data.mol.geom.elem).iter().map(|x| *x as usize).collect::<Vec<usize>>();
        let positions = &scf_data.mol.geom.position.data;
        let charges = None;
        let lattice = None;
        let periodic = None;
        DFTD4Model::new(&numbers, positions, charges, lattice, periodic)
}

pub fn dftd3_atm_from_scf(scf_data: &SCF) -> (f64, Option<Vec<f64>>, Option<Vec<f64>>) {
    #[cfg(feature = "dftd3")]
    {   
        let d3_model = prepare_d3model(scf_data);
        //let xc = scf_data.mol.ctrl.xc.as_str();
        // reshape the name for some DFAs, which cannot be recognized by the dftd library.
        let mut xc = scf_data.mol.ctrl.xc.as_str();
        if xc.eq("m05-2x") {
            xc = "m052x"
        } else if xc.eq("m06-2x") {
            xc = "m062x"
        };
        let version = scf_data.mol.ctrl.empirical_dispersion.clone().unwrap();
        // handle special case: d3 -> d3zero
        let version = if version == "d3" {
            "d3zero".to_string()
        } else {
            version
        };
        let params = dftd3_load_param(&version, xc, true);
        d3_model.get_dispersion(&params, true).into()
    }
    #[cfg(not(feature = "dftd3"))]
    {
        println!("Do not invoke the empirical dispersion evaluation!");
        panic!("dftd3 is not enabled in the build.");
    }
}

pub fn dftd3_atm_from_parse_xc(scf_data: &SCF, disp_component: &DFAComponent) -> (f64, Option<Vec<f64>>, Option<Vec<f64>>) {
    #[cfg(feature = "dftd3")]
    {
        let d3_model = prepare_d3model(scf_data);
        let disp_params = disp_component.get_dftd_params();
        let dftd3_param = if disp_params.get("xc").is_some() {
            let version = disp_params.get("version").unwrap().as_str().unwrap();
            let xc = disp_params.get("xc").unwrap().as_str().unwrap();
            dftd3_load_param(version, xc, true)
        } else {
            let toml_params = toml::Value::try_from(disp_params).expect("Failed to convert dftd parameters to toml value.");
            let toml_string = toml::to_string(&toml_params).unwrap();
            let damping_param = dftd3_parse_damping_param_from_toml(toml_string.as_str());
            damping_param.new_param()
        };
        d3_model.get_dispersion(&dftd3_param, true).into()
    }
    #[cfg(not(feature = "dftd3"))]
    {
        println!("Do not invoke the empirical dispersion evaluation!");
        panic!("dftd3 is not enabled in the build.");
    }
}


pub fn dftd4_atm_from_scf(scf_data: &SCF) -> (f64, Option<Vec<f64>>, Option<Vec<f64>>) {
    #[cfg(feature = "dftd4")]
    {
        let d4_model = prepare_d4model(scf_data);
        let xc = scf_data.mol.ctrl.xc.as_str();
        let params = DFTD4Param::load_rational_damping(xc, true);
        d4_model.get_dispersion(&params, true).into()
    }
    #[cfg(not(feature = "dftd4"))]
    {
        println!("Do not invoke the empirical dispersion evaluation!");
        panic!("dftd4 is not enabled in the build.");
    }
}

pub fn dftd4_atm_from_parse_xc(scf_data: &SCF, disp_component: &DFAComponent) -> (f64, Option<Vec<f64>>, Option<Vec<f64>>) {
    #[cfg(feature = "dftd4")]
    {
        let d4_model = prepare_d4model(scf_data);
        let disp_params = disp_component.get_dftd_params();
        let dftd4_param = if disp_params.get("xc").is_some() {
            let xc = disp_params.get("xc").unwrap().as_str().unwrap();
            DFTD4Param::load_rational_damping(xc, true)
        } else {
            let toml_params = toml::Value::try_from(disp_params).expect("Failed to convert dftd parameters to toml value.");
            let toml_string = toml::to_string(&toml_params).unwrap();
            dftd4_parse_damping_param_from_toml(toml_string.as_str()).new_param()
        };
        d4_model.get_dispersion(&dftd4_param, true).into()
    }
    #[cfg(not(feature = "dftd4"))]
    {
        println!("Do not invoke the empirical dispersion evaluation!");
        panic!("dftd4 is not enabled in the build.");
    }
}
