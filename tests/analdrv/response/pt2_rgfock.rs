use pyrest::molecule_io::Molecule;
use pyrest::ri_jk::util::get_cint_mol;
use pyrest::scf_io::{self, scf_without_build};
use pyrest::utilities::rstsr_util::RestTensorToRstsrTsrAPI;
use pyrest::{ctrl_io, ri_pt2};
use rstsr::prelude::*;

static INPUT_NH3: &str = r##"
[ctrl]
    print_level =          2
    num_threads =          4
    xc =                   "mp2"
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

#[test]
fn test_nh3() {
    let keys = toml::from_str::<serde_json::Value>(&INPUT_NH3[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);
    let eng = ri_pt2::xdh_calculations(&mut scf_data, &None).unwrap();
    println!("MP2 energy: {}", eng);

    let device = rt::DeviceBLAS::default();

    // 1. nuclear contribution
    let dip_nuc = scf_data.mol.geom.evaluate_dipole_moment(None).0;
    let dip_nuc = rt::asarray((dip_nuc.to_vec(), &device));
    println!("Dipole from nuclear contribution: {:16.12}", dip_nuc);
    let dip_nuc_ref = rt::asarray((vec![1.700753512109, 1.889726124565, 2.078698737022], &device));
    assert!(rt::allclose(&dip_nuc, &dip_nuc_ref, None));

    // 2. scf density contribution
    let dm = scf_data.density_matrix[0].to_rstsr(&device);
    let mol = get_cint_mol(&scf_data.mol);
    let int1e_r = {
        let (out, shape) = mol.integrate("int1e_r", None, None).into();
        rt::asarray((out, shape, &device))
    };
    let dip_dm_scf = -(int1e_r * dm).sum_axes([0, 1]);
    println!("Dipole from SCF density matrix: {:16.12}", dip_dm_scf);
    let dip_dm_scf_ref = rt::asarray((vec![-1.138682776261, -1.343280625287, -1.559934460339], &device));
    assert!(rt::allclose(&dip_dm_scf, &dip_dm_scf_ref, None));
}

// --- following is utilities for developing dipole evaluation --- //
