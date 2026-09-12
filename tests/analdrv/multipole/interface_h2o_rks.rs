//! Interface-level (through `analdrv_interface`) multipole test: SCF-level B3LYP H2O.
//!
//! Drives the full task path — task dispatch, explicit `multipole_origin`, the SCF-only
//! construction (no gfock/resp objects for usual DFT), stdout print, and the results-JSON
//! export. The moment references are identical to `tests/analdrv/multipole/h2o_rks.rs`
//! (pyscf 2.14.0, RI-JK with def2-universal-jkfit, origin at [0, 0, 0]); the default-origin
//! (center of nuclear mass) check uses the IUPAC 2021 average atomic weights.

use pyrest::analdrv::config::{AnalDrvConfig, AnalDrvTask};
use pyrest::analdrv::interface::{analdrv_interface, analdrv_json_interface};
use pyrest::ctrl_io;
use pyrest::molecule_io::Molecule;
use pyrest::scf_io::{self, scf_without_build};


static INPUT_H2O: &str = r##"
[ctrl]
    print_level =          2
    num_threads =          4
    xc =                   "b3lyp"
    basis_path =           "def2-tzvp"
    auxbas_path =          "def2-universal-jkfit"
    eri_type =             "ri-v"
    charge =               0.0
    spin =                 1.0
    spin_polarization =    false
    auxbasis_response =    true
    mixer =                "diis"
    num_max_diis =         8
    start_diis_cycle =     3
    mix_param =            0.8
    max_scf_cycle =        100

[geom]
    name = "H2O"
    unit = "Angstrom"
    position = """
    O  0.0   0.0   0.0
    H  0.75  0.55  0.10
    H -0.70  0.60 -0.05
    """
"##;

fn allclose(v: &[f64], ref_: &[f64], tol: f64) {
    assert_eq!(v.len(), ref_.len());
    for (a, b) in v.iter().zip(ref_.iter()) {
        assert!((a - b).abs() < tol, "component mismatch: {a} vs {b}");
    }
}

#[test]
fn test_h2o_interface() {
    let keys = toml::from_str::<serde_json::Value>(&INPUT_H2O[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);

    let tasks = vec![AnalDrvTask::Multipole];

    // --- explicit origin [0, 0, 0]: identical settings to the driver-level h2o_rks test --- //
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
    // reference: pyscf RI-JK B3LYP (def2-tzvp / def2-universal-jkfit)
    allclose(&dip.tot, &[0.031738377072, 0.814840804724, 0.035041217679], 1e-5);
    allclose(
        &mp.quadrupole.as_ref().unwrap().tot,
        &[
            -3.358418468690, -0.029750730554, 0.231736911888, -0.029750730554, -4.303781449787,
            0.046590665639, 0.231736911888, 0.046590665639, -5.554802927163,
        ],
        1e-5,
    );
    // traceless quadrupole: 3/2 (Q - Tr(Q)/3 I), XX component from the raw reference above
    let quad_traceless = mp.quadrupole.as_ref().unwrap().traceless.as_ref().unwrap();
    allclose(&quad_traceless[..1], &[1.570873719784], 1e-5);

    // results-JSON layout: no hessian keys at all (the hessian task did not run), the multipole
    // result nested under its own key; no "thermo" key without a [thermo] section
    let anal = &json_extra["analdrv"];
    assert!(anal.get("frequencies_cm").is_none());
    assert!(anal.get("modes_trv").is_none());
    let dip_json = &anal["multipole"]["dipole"];
    assert!(dip_json["corr"].is_null());
    assert!(dip_json["resp"].is_null());
    allclose(
        &dip_json["tot"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>(),
        &[0.031738377072, 0.814840804724, 0.035041217679],
        1e-5,
    );
    assert!(json_extra.get("thermo").is_none());

    // --- default origin: center of nuclear mass from IUPAC 2021 average weights --- //
    let mut config = AnalDrvConfig::default();
    config.multipole.orders = vec![1];
    let output = analdrv_interface(&mut scf_data, &tasks, &config);

    let mp = output.multipole.as_ref().unwrap();
    // expected CoM (Bohr): (1.008 * (0.75 - 0.70), 1.008 * (0.55 + 0.60), 1.008 * (0.10 - 0.05))
    // Angstrom / (15.999 + 2 * 1.008) / 0.52917721092 (REST's BOHR)
    let expected_com = [0.005286827459, 0.121597031562, 0.005286827459];
    allclose(&mp.origin, &expected_com, 1e-9);
    // neutral molecule: dipole is origin-independent, so the total is unchanged
    allclose(&mp.dipole.as_ref().unwrap().tot, &[0.031738377072, 0.814840804724, 0.035041217679], 1e-5);
}
