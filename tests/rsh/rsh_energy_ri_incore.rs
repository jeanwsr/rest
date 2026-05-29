//! SCF energy test: RSH functionals vs pyscf (NH3/STO-3G)

use pyrest::ctrl_io;
use pyrest::molecule_io::Molecule;
use pyrest::scf_io::{self, SCF};

static CTRL_TEMPLATE: &str = r##"
[ctrl]
    print_level =          1
    xc =                   "XC_PLACEHOLDER"
    basis_path =           "basis-set-pool/sto-3g"
    auxbas_path =          "basis-set-pool/def2-universal-jkfit"
    eri_type =             "ri-v"
    charge =               0.0
    algorithm_jk =         "ri-incore"
    spin =                 1
    spin_polarization =    false
    initial_guess =        "sad"
    num_threads =          4
    max_scf_cycle =        50

[geom]
    name = "NH3"
    unit = "Angstrom"
    position = """
        N  0.0  0.0  0.0
        H  0.0  1.5  1.0
        H  1.4  1.1  0.0
        H  1.2  0.0  1.3
    """
"##;

fn run_scf(xc: &str) -> SCF {
    let ctrl_str = CTRL_TEMPLATE.replace("XC_PLACEHOLDER", xc);
    let keys = toml::from_str::<serde_json::Value>(&ctrl_str).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf = SCF::build(mol, &None);
    scf_io::scf_without_build(&mut scf, &None);
    scf
}

#[test]
fn test_cam_b3lyp_energy() {
    let scf = run_scf("cam-b3lyp");
    let pyscf_ref = -55.315789868819;
    println!("REST CAM-B3LYP: {:.12}", scf.scf_energy);
    println!("pyscf:          {:.12}", pyscf_ref);
    assert!((scf.scf_energy - pyscf_ref).abs() < 1e-5,
        "CAM-B3LYP energy mismatch: REST={} pyscf={}", scf.scf_energy, pyscf_ref);
}

#[test]
fn test_wb97x_energy() {
    let scf = run_scf("wb97x");
    let pyscf_ref = -55.333317760468;
    println!("REST wB97X: {:.12}", scf.scf_energy);
    println!("pyscf:      {:.12}", pyscf_ref);
    assert!((scf.scf_energy - pyscf_ref).abs() < 1e-5,
        "wB97X energy mismatch: REST={} pyscf={}", scf.scf_energy, pyscf_ref);
}

#[test]
fn test_lc_blyp_energy() {
    let scf = run_scf("lc-blyp");
    let pyscf_ref = -55.1845937756802;
    println!("REST LC-BLYP: {:.12}", scf.scf_energy);
    println!("pyscf:        {:.12}", pyscf_ref);
    assert!((scf.scf_energy - pyscf_ref).abs() < 1e-5,
        "LC-BLYP energy mismatch: REST={} pyscf={}", scf.scf_energy, pyscf_ref);
}
