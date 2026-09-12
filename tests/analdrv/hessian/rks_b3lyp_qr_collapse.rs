use pyrest::analdrv::hessian::rscf_interface::rscf_hess_interface;
use pyrest::analdrv::response::rresp_interface::rscf_resp_interface;
// use pyrest::analdrv::config::AnalDrvConfig;

use pyrest::ctrl_io;
use pyrest::molecule_io::Molecule;
use pyrest::scf_io::{self, scf_without_build};

static INPUT_H2O: &str = r##"
[ctrl]
    print_level =          2
    num_threads =          4
    xc =                   "b3lyp"
    basis_path =           "6-31g"
    auxbas_path =          "def2-universal-jkfit"
    eri_type =             "ri-v"
    charge =               0.0
    spin =                 1.0
    spin_polarization =    false
    auxbasis_response =    true
    mixer =                "diis"
    max_scf_cycle =        100

[analdrv]
    # legacy cphf_* key names, to verify they keep working as aliases of cpscf_*
    cphf_tol = 1e-9
    cphf_tol_inflation = 1e3
    #cphf_lindep = 1e-15

[geom]
    name = "H2O"
    unit = "Angstrom"
    position = """
        O  0.0  0.000  0.000
        H  0.0 -0.757  0.587
        H  0.0  0.757  0.587
    """
"##;

#[test]
fn test_h2o_qr_collapse() {
    let keys = toml::from_str::<serde_json::Value>(&INPUT_H2O[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);

    // let config = AnalDrvConfig::default();
    let config = scf_data.mol.ctrl.analdrv.as_ref().unwrap().clone();
    let mut resp_objs = rscf_resp_interface(&scf_data, &config);
    let (_de, vib, _) = rscf_hess_interface(&scf_data, &config.nucgrad, &mut resp_objs);
    println!("freqs: {:?}", vib.omega)
}
