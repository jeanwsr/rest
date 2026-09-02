use crate::dft::parse_xc::DFAComponent;
use crate::geom_io::get_charge;
use crate::molecule_io::Molecule;
#[cfg(feature = "dftd3")]
use dftd3::prelude::*;
#[cfg(feature = "dftd4")]
use dftd4::prelude::*;
use toml;

/// Specification of the empirical dispersion correction, resolved from the input.
///
/// The dispersion can be specified in two ways: as a dispersion component parsed from the `xc`
/// keyword (like `b3lyp-d3bj`), or by the `empirical_dispersion` control keyword. Specifying
/// both is ambiguous and rejected.
pub enum DispSpec {
    /// Dispersion component parsed from the `xc` keyword
    /// (like the `D3(version=bj, xc=b3lyp)` part of `b3lyp-d3bj`).
    FromXc(DFAComponent),
    /// Dispersion specified by the `empirical_dispersion` keyword ("d3", "d3bj" or "d4").
    FromCtrl(String),
}

impl DispSpec {
    /// Resolve the dispersion specification from the molecule.
    /// Returns `None` if no empirical dispersion is specified.
    pub fn resolve(mol: &Molecule) -> Option<Self> {
        let disp_from_parse_xc = mol.dfadef.as_ref().map(|dfadef| dfadef.has_dispersion()).unwrap_or(false);
        let disp_from_ctrl = mol.ctrl.empirical_dispersion.is_some();
        if disp_from_parse_xc && disp_from_ctrl {
            panic!("Empirical dispersion is specified in both xc and empirical_dispersion. Please specify it in only one place to avoid ambiguity.");
        }

        if disp_from_parse_xc {
            return Some(DispSpec::FromXc(mol.dfadef.as_ref().unwrap().get_dispersion().unwrap()));
        }
        if disp_from_ctrl {
            return Some(DispSpec::FromCtrl(mol.ctrl.empirical_dispersion.clone().unwrap()));
        }
        None
    }

    /// Evaluate the dispersion (energy, gradient, sigma) for the geometry currently in `mol`.
    ///
    /// The gradient is the derivative of the dispersion energy with respect to nuclear
    /// coordinates (`natom * 3`, atom-major). This function is quiet (no info messages), so it
    /// is suitable for repeated evaluation at displaced geometries, like finite-difference
    /// Hessians.
    pub fn evaluate(&self, mol: &Molecule) -> (f64, Option<Vec<f64>>, Option<Vec<f64>>) {
        match self {
            DispSpec::FromXc(disp) => match disp.func.to_lowercase().as_str() {
                "dftd3" => dftd3_atm_from_parse_xc(mol, disp),
                "dftd4" => dftd4_atm_from_parse_xc(mol, disp),
                _ => panic!("Invalid input for empirical dispersion in xc: {}.", disp.func),
            },
            DispSpec::FromCtrl(version) => match version.as_str() {
                // "d3" or "d3bj" uses dftd3_atm; "d4" uses dftd4.
                "d3" | "d3bj" => dftd3_atm_from_mol(mol),
                "d4" => dftd4_atm_from_mol(mol),
                _ => panic!(
                    "Invalid input for empirical_dispersion: {}.\nDo not invoke the empirical dispersion evaluation!",
                    version
                ),
            },
        }
    }
}

/// Evaluate the empirical dispersion correction for the molecule.
///
/// Returns `None` if no empirical dispersion is specified. For repeated evaluation at
/// different geometries, resolve a [`DispSpec`] once and use its `evaluate` method.
pub fn dftd(mol: &Molecule) -> Option<(f64, Option<Vec<f64>>, Option<Vec<f64>>)> {
    let spec = DispSpec::resolve(mol)?;
    match &spec {
        DispSpec::FromXc(_) => {
            println!("Empirical dispersion is specified in xc. Use the dispersion from parse_xc.");
        },
        DispSpec::FromCtrl(_) => {
            println!(
                "Empirical dispersion is specified in empirical_dispersion. Use the dispersion from empirical_dispersion."
            );
        },
    }
    Some(spec.evaluate(mol))
}

#[cfg(feature = "dftd3")]
pub fn prepare_d3model(mol: &Molecule) -> DFTD3Model {
    let numbers = get_charge(&mol.geom.elem).iter().map(|x| *x as usize).collect::<Vec<usize>>();
    let positions = &mol.geom.position.data;
    let lattice = None;
    let periodic = None;
    DFTD3Model::new(&numbers, positions, lattice, periodic)
}

#[cfg(feature = "dftd4")]
pub fn prepare_d4model(mol: &Molecule) -> DFTD4Model {
    let numbers = get_charge(&mol.geom.elem).iter().map(|x| *x as usize).collect::<Vec<usize>>();
    let positions = &mol.geom.position.data;
    let charges = None;
    let lattice = None;
    let periodic = None;
    DFTD4Model::new(&numbers, positions, charges, lattice, periodic)
}

/// DFTD3 dispersion at the geometry of `mol`, with damping parameters loaded from the
/// ctrl keywords (`xc` and `empirical_dispersion`).
pub fn dftd3_atm_from_mol(mol: &Molecule) -> (f64, Option<Vec<f64>>, Option<Vec<f64>>) {
    #[cfg(feature = "dftd3")]
    {
        let d3_model = prepare_d3model(mol);
        //let xc = mol.ctrl.xc.as_str();
        // reshape the name for some DFAs, which cannot be recognized by the dftd library.
        let mut xc = mol.ctrl.xc.as_str();
        if xc.eq("m05-2x") {
            xc = "m052x"
        } else if xc.eq("m06-2x") {
            xc = "m062x"
        };
        let version = mol.ctrl.empirical_dispersion.clone().unwrap();
        // handle special case: d3 -> d3zero
        let version = if version == "d3" { "d3zero".to_string() } else { version };
        let params = dftd3_load_param(&version, xc, true);
        d3_model.get_dispersion(&params, true).into()
    }
    #[cfg(not(feature = "dftd3"))]
    {
        println!("Do not invoke the empirical dispersion evaluation!");
        panic!("dftd3 is not enabled in the build.");
    }
}

pub fn dftd3_atm_from_parse_xc(
    mol: &Molecule,
    disp_component: &DFAComponent,
) -> (f64, Option<Vec<f64>>, Option<Vec<f64>>) {
    #[cfg(feature = "dftd3")]
    {
        let d3_model = prepare_d3model(mol);
        let disp_params = disp_component.get_dftd_params();
        let dftd3_param = if disp_params.get("xc").is_some() {
            let version = disp_params.get("version").unwrap().as_str().unwrap();
            let xc = disp_params.get("xc").unwrap().as_str().unwrap();
            dftd3_load_param(version, xc, true)
        } else {
            let toml_params =
                toml::Value::try_from(disp_params).expect("Failed to convert dftd parameters to toml value.");
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

/// DFTD4 dispersion at the geometry of `mol`, with damping parameters loaded from the
/// ctrl keyword `xc`.
pub fn dftd4_atm_from_mol(mol: &Molecule) -> (f64, Option<Vec<f64>>, Option<Vec<f64>>) {
    #[cfg(feature = "dftd4")]
    {
        let d4_model = prepare_d4model(mol);
        let xc = mol.ctrl.xc.as_str();
        let params = DFTD4Param::load_rational_damping(xc, true);
        d4_model.get_dispersion(&params, true).into()
    }
    #[cfg(not(feature = "dftd4"))]
    {
        println!("Do not invoke the empirical dispersion evaluation!");
        panic!("dftd4 is not enabled in the build.");
    }
}

pub fn dftd4_atm_from_parse_xc(
    mol: &Molecule,
    disp_component: &DFAComponent,
) -> (f64, Option<Vec<f64>>, Option<Vec<f64>>) {
    #[cfg(feature = "dftd4")]
    {
        let d4_model = prepare_d4model(mol);
        let disp_params = disp_component.get_dftd_params();
        let dftd4_param = if disp_params.get("xc").is_some() {
            let xc = disp_params.get("xc").unwrap().as_str().unwrap();
            DFTD4Param::load_rational_damping(xc, true)
        } else {
            let toml_params =
                toml::Value::try_from(disp_params).expect("Failed to convert dftd parameters to toml value.");
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
