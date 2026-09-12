//! Interface-level (through `analdrv_interface`) multipole test: XYG3 (xDH double hybrid) NH3.
//!
//! Drives the full DH task path — grids regeneration after `xdh_calculations`, the shared
//! response object, the relaxed (default) and unrelaxed `multipole_rdm1_relax` modes, and the
//! results-JSON export. The dipole part references are identical to
//! `tests/analdrv/multipole/nh3_xyg3.rs` / `tests/analdrv/response/xyg3_rgfock.rs`
//! (pyscf-forge master @ 0566d43, pyscf 2.14.0); the unrelaxed total is the sum of the
//! nuc/scf/corr references.

use pyrest::analdrv::config::{AnalDrvConfig, AnalDrvTask, MultipoleRdm1Relax};
use pyrest::analdrv::interface::{analdrv_interface, analdrv_json_interface};
use pyrest::ctrl_io;
use pyrest::molecule_io::Molecule;
use pyrest::scf_io::{self, scf_without_build};
use pyrest::{ri_pt2};


static INPUT_NH3: &str = r##"
[ctrl]
    print_level =          2
    num_threads =          4
    xc =                   "xyg3"
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

[ctrl.ri_pt2]
    new_driver = true

[geom]
    name = "NH3"
    unit = "Angstrom"
    position = """
    N 0.0 0.0 0.0
    H 0.9 0.0 0.0
    H 0.0 1.0 0.0
    H 0.0 0.0 1.1
    """
"##;

fn allclose(v: &[f64], ref_: &[f64], tol: f64) {
    assert_eq!(v.len(), ref_.len());
    for (a, b) in v.iter().zip(ref_.iter()) {
        assert!((a - b).abs() < tol, "component mismatch: {a} vs {b}");
    }
}

#[test]
fn test_nh3_xyg3_interface() {
    let keys = toml::from_str::<serde_json::Value>(&INPUT_NH3[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);

    // XYG3 total energy (side product; also frees the DFT grids, which analdrv_interface must
    // regenerate for the DH increments)
    let eng = ri_pt2::xdh_calculations(&mut scf_data, &None).unwrap();
    assert!((eng - (-56.513619197590)).abs() < 2e-6, "XYG3 total energy mismatch");

    let tasks = vec![AnalDrvTask::Multipole];

    // --- relaxed mode (default): nuc + scf + corr + resp (Z-vector) --- //
    let mut config = AnalDrvConfig::default();
    config.multipole.orders = vec![1, 2];
    config.multipole.origin = Some([0.0; 3]);
    let output = analdrv_interface(&mut scf_data, &tasks, &config);
    let json_extra = analdrv_json_interface(&output);

    let mp = output.multipole.as_ref().expect("multipole task output");
    assert_eq!(mp.origin, [0.0; 3]);
    let dip = mp.dipole.as_ref().unwrap();
    allclose(&dip.nuc, &[1.700753512109, 1.889726124565, 2.078698737022], 1e-8);
    allclose(&dip.scf, &[-1.156948912236, -1.364097621601, -1.579483131449], 1e-5);
    allclose(&dip.corr.as_ref().unwrap(), &[-0.002948597510, -0.004975193666, -0.007063388846], 1e-6);
    allclose(&dip.resp.as_ref().unwrap(), &[0.009523402231, 0.014507684671, 0.020086193759], 1e-5);
    allclose(&dip.tot, &[0.550379404594, 0.535160993970, 0.512238410486], 1e-5);

    // results-JSON: the DH decomposition is carried per part
    let dip_json = &json_extra["analdrv"]["multipole"]["dipole"];
    allclose(
        &dip_json["tot"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>(),
        &[0.550379404594, 0.535160993970, 0.512238410486],
        1e-5,
    );
    assert!(dip_json["corr"].is_array());
    assert!(dip_json["resp"].is_array());
    assert!(json_extra["analdrv"]["multipole"]["quadrupole"]["traceless"].is_array());

    // --- unrelaxed mode: nuc + scf + corr only, no CP-SCF --- //
    config.multipole.rdm1_relax = MultipoleRdm1Relax::Unrelaxed;
    let output = analdrv_interface(&mut scf_data, &tasks, &config);
    let json_extra = analdrv_json_interface(&output);

    let dip = output.multipole.as_ref().unwrap().dipole.as_ref().unwrap();
    assert!(dip.resp.is_none());
    // sum of the nuc/scf/corr references above
    allclose(&dip.tot, &[0.540856002363, 0.520653309298, 0.492152216727], 1e-5);
    assert!(json_extra["analdrv"]["multipole"]["dipole"]["resp"].is_null());
}
