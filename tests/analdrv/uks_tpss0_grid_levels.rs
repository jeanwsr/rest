//! UKS (TPSS0, MGGA) Hessian with explicit `grid_level_cpscf` / `grid_level_skeleton`.
//!
//! Exercises the MGGA skeleton-grid regeneration (`grid_gen_level + 2` default) and the
//! dedicated CP-SCF grid path on the UKS side.

use pyrest::analdrv::config::{AnalDrvCpscfCfg, AnalDrvConfig, AnalDrvNucgradCfg};
use pyrest::analdrv::uscf_interface::uscf_hess_interface;

use pyrest::ctrl_io;
use pyrest::molecule_io::Molecule;
use pyrest::scf_io::{self, scf_without_build};

use rstsr::prelude::*;

static INPUT_NH3: &str = r##"
[ctrl]
    print_level =          2
    num_threads =          16
    xc =                   "TPSS0"
    basis_path =           "def2-tzvp"
    auxbas_path =          "def2-universal-jkfit"
    eri_type =             "ri-v"
    charge =               2.0
    spin =                 3.0
    spin_polarization =    true
    auxbasis_response =    true
    mixer =                "diis"
    num_max_diis =         8
    start_diis_cycle =     3
    mix_param =            0.8
    max_scf_cycle =        100
    xc_parser =            "parse_xc"

[geom]
    name = "NH3"
    unit = "Angstrom"
    position = """
        N  0.0  0.0  0.0
        H  1.0  0.1  0.2
        H  0.3  1.1  0.2
        H  0.1  0.1  1.2
    """
"##;

fn run_with_config(config: AnalDrvConfig) -> Vec<f64> {
    let keys = toml::from_str::<serde_json::Value>(&INPUT_NH3[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);
    let (de, _vib, _th) = uscf_hess_interface(&mut scf_data, &config);
    de
}

#[test]
fn test_nh3_mgga_default_skeleton() {
    // Default skeleton level for MGGA = grid_gen_level + 2 = 5; cpscf = 1 -> ni_cpks path.
    let config = AnalDrvConfig::default();
    let de = run_with_config(config);

    let natm = 4;
    let de = rt::asarray((&de, [3, 3, natm, natm]));
    println!("Hessian (MGGA default grid levels):\n{:12.6}", de.t());
    assert_eq!(de.shape().to_vec(), vec![3, 3, natm, natm]);
}

#[test]
fn test_nh3_mgga_explicit_grid_levels() {
    let config = AnalDrvConfig {
        nucgrad: AnalDrvNucgradCfg { grid_level_skeleton: Some(5), ..Default::default() },
        cpscf: AnalDrvCpscfCfg { grid_level: Some(2), ..Default::default() },
        ..Default::default()
    };
    let de = run_with_config(config);

    let natm = 4;
    let de = rt::asarray((&de, [3, 3, natm, natm]));
    println!("Hessian (MGGA explicit grid levels):\n{:12.6}", de.t());
    assert_eq!(de.shape().to_vec(), vec![3, 3, natm, natm]);
}
