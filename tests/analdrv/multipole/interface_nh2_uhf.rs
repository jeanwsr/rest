//! Interface-level (through `analdrv_interface`) multipole test: SCF-level UHF NH2, plus the
//! rejection of the unrestricted post-SCF (fifth-DFA) path.
//!
//! Drives the full unrestricted task path — task dispatch, explicit `multipole_origin`, stdout
//! print, and the results-JSON export. The moment references are identical to
//! `tests/analdrv/multipole/nh2_uhf.rs` (pyscf 2.14.0, RI-JK with def2-universal-jkfit).

use pyrest::analdrv::config::{AnalDrvConfig, AnalDrvTask};
use pyrest::analdrv::interface::{analdrv_interface, analdrv_json_interface};
use pyrest::ctrl_io;
use pyrest::molecule_io::Molecule;
use pyrest::scf_io::{self, scf_without_build};

static INPUT_NH2: &str = r##"
[ctrl]
    print_level =          2
    num_threads =          4
    xc =                   "hf"
    basis_path =           "def2-tzvp"
    auxbas_path =          "def2-universal-jkfit"
    eri_type =             "ri-v"
    charge =               0.0
    spin =                 2.0
    spin_polarization =    true

[geom]
    name = "NH2"
    unit = "Angstrom"
    position = """
    N  0.0   0.0   0.0
    H  0.58  0.20  0.85
    H -0.50 -0.25  0.83
    """
"##;

static INPUT_NH2_XYG3: &str = r##"
[ctrl]
    print_level =          2
    num_threads =          4
    xc =                   "xyg3"
    basis_path =           "def2-tzvp"
    auxbas_path =          "def2-universal-jkfit"
    eri_type =             "ri-v"
    charge =               0.0
    spin =                 2.0
    spin_polarization =    true

[ctrl.ri_pt2]
    new_driver = true

[geom]
    name = "NH2"
    unit = "Angstrom"
    position = """
    N  0.0   0.0   0.0
    H  0.58  0.20  0.85
    H -0.50 -0.25  0.83
    """
"##;

fn allclose(v: &[f64], ref_: &[f64], tol: f64) {
    assert_eq!(v.len(), ref_.len());
    for (a, b) in v.iter().zip(ref_.iter()) {
        assert!((a - b).abs() < tol, "component mismatch: {a} vs {b}");
    }
}

fn build_scf(input: &str) -> scf_io::SCF {
    let keys = toml::from_str::<serde_json::Value>(input).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);
    scf_data
}

#[test]
fn test_nh2_uhf_interface() {
    let mut scf_data = build_scf(INPUT_NH2);
    let tasks = vec![AnalDrvTask::Multipole];

    // --- explicit origin [0, 0, 0]: identical settings to the driver-level nh2_uhf test --- //
    let mut config = AnalDrvConfig::default();
    config.multipole.orders = vec![1, 2];
    config.multipole.origin = Some([0.0; 3]);
    let output = analdrv_interface(&mut scf_data, &tasks, &config);
    let json_extra = analdrv_json_interface(&output);

    let mp = output.multipole.as_ref().expect("multipole task output");
    assert_eq!(mp.origin, [0.0; 3]);
    let dip = mp.dipole.as_ref().unwrap();
    // SCF level: no corr/resp increments at all
    assert!(dip.corr.is_none());
    assert!(dip.resp.is_none());
    // reference: pyscf RI-JK UHF (def2-tzvp / def2-universal-jkfit)
    allclose(&dip.nuc, &[0.151178089965, -0.094486306228, 3.174739889269], 1e-8);
    allclose(&dip.tot, &[0.016656856517, -0.038475737329, 0.915089522375], 1e-6);
    allclose(
        &mp.quadrupole.as_ref().unwrap().tot,
        &[
            -5.193509415837, 0.020912254543, 0.050343803281, 0.020912254543, -5.239324563951,
            -0.030545457675, 0.050343803281, -0.030545457675, -4.210872999273,
        ],
        1e-6,
    );
    // traceless quadrupole: 3/2 (Q - Tr(Q)/3 I), XX component from the raw reference above
    let quad_traceless = mp.quadrupole.as_ref().unwrap().traceless.as_ref().unwrap();
    allclose(&quad_traceless[..1], &[-0.468410634225], 1e-6);

    // results-JSON layout: no corr/resp keys for SCF-level unrestricted, no hessian keys
    let anal = &json_extra["analdrv"];
    assert!(anal.get("frequencies_cm").is_none());
    let dip_json = &anal["multipole"]["dipole"];
    assert!(dip_json["corr"].is_null());
    assert!(dip_json["resp"].is_null());
    allclose(
        &dip_json["tot"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>(),
        &[0.016656856517, -0.038475737329, 0.915089522375],
        1e-6,
    );

    // --- default origin: the coordinate origin [0, 0, 0] --- //
    let mut config = AnalDrvConfig::default();
    config.multipole.orders = vec![1];
    let output = analdrv_interface(&mut scf_data, &tasks, &config);
    let mp = output.multipole.as_ref().unwrap();
    assert_eq!(mp.origin, [0.0; 3]);
    allclose(&mp.dipole.as_ref().unwrap().tot, &[0.016656856517, -0.038475737329, 0.915089522375], 1e-6);
}

#[test]
#[should_panic(expected = "Multipole evaluation is not implemented for unrestricted post-SCF")]
fn test_nh2_uhf_fifth_dfa_panics() {
    let mut scf_data = build_scf(INPUT_NH2_XYG3);
    let tasks = vec![AnalDrvTask::Multipole];
    let mut config = AnalDrvConfig::default();
    config.multipole.orders = vec![1];
    let _ = analdrv_interface(&mut scf_data, &tasks, &config);
}
