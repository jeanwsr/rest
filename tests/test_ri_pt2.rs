use pyrest::ctrl_io;
use pyrest::molecule_io::Molecule;
use pyrest::ri_pt2::{self, xdh_calculations};
use pyrest::scf_io::{self, scf_without_build, SCF};
use pyrest::utilities::TimeRecords;

#[test]
fn test_restricted() {
    let mut scf_data = initialize_nh3();
    let mut timerecords = TimeRecords::new();
    let eng_lst = ri_pt2::pt2_pair_eng::evaluate_ript2_eng::<f64>(&scf_data, &mut timerecords);
    timerecords.report_all();
    println!("RIPT2 energy for NH3: {:?}", eng_lst);

    assert!((eng_lst[1] - -0.24939630795012804).abs() < 1e-6);
    assert!((eng_lst[0] - -0.31670803380267337).abs() < 1e-6);

    scf_data.rimatr = None;
    let eng_lst = ri_pt2::pt2_pair_eng::evaluate_ript2_eng::<f64>(&scf_data, &mut timerecords);
    timerecords.report_all();
    println!("RIPT2 energy for NH3 without rimatr: {:?}", eng_lst);

    assert!((eng_lst[1] - -0.24939630795012804).abs() < 1e-6);
    assert!((eng_lst[0] - -0.31670803380267337).abs() < 1e-6);

    assert!(scf_data.rimatr.is_none());
    assert!(scf_data.ri3mo.is_none());
    assert!(scf_data.ri3fn.is_none());
}

#[test]
fn test_unrestricted() {
    let mut scf_data = initialize_nh3_cation();
    let mut timerecords = TimeRecords::new();
    let eng_lst = ri_pt2::pt2_pair_eng::evaluate_riupt2_eng::<f64>(&scf_data, &mut timerecords);
    timerecords.report_all();
    println!("RIUPT2 energy for NH3 cation: {:?}", eng_lst);

    assert!((eng_lst[1] - -0.11474641974264627).abs() < 1e-6);
    assert!((eng_lst[0] - -0.150330650003985).abs() < 1e-6);

    scf_data.rimatr = None;
    let eng_lst = ri_pt2::pt2_pair_eng::evaluate_riupt2_eng::<f64>(&scf_data, &mut timerecords);
    timerecords.report_all();
    println!("RIUPT2 energy for NH3 cation without rimatr: {:?}", eng_lst);

    assert!((eng_lst[1] - -0.11474641974264627).abs() < 1e-6);
    assert!((eng_lst[0] - -0.150330650003985).abs() < 1e-6);

    assert!(scf_data.rimatr.is_none());
    assert!(scf_data.ri3mo.is_none());
    assert!(scf_data.ri3fn.is_none());
}

#[test]
fn test_hybrid_and_dh_direct() {
    // TODO: this test has not passed.
    let input_token = CTRL_STR
        .replace("SPIN_POLARIZATION", "false")
        .replace("CHARGE", "0.0")
        .replace("SPIN", "1.0")
        .replace("mp2", "xyg3");

    println!("=== usual calculation ===");
    let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);
    let _ = xdh_calculations(&mut scf_data, &None);
    println!("Energy table: {:#?}", scf_data.energies);
    assert!((scf_data.energies["xdh_energy"][0] - -56.19186092022621).abs() < 1e-6);
    assert!((scf_data.energies["pt2"][0] - -0.6356496452322714).abs() < 1e-6);

    println!("=== ri-direct calculation ===");
    let input_token = input_token.replace("ri-incore", "ri-direct");
    let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = SCF::build(mol, &None);
    assert!(scf_data.rimatr.is_none());
    scf_without_build(&mut scf_data, &None);
    assert!(scf_data.rimatr.is_none());
    let _ = xdh_calculations(&mut scf_data, &None);
    assert!(scf_data.rimatr.is_none());
    println!("Energy table: {:#?}", scf_data.energies);
    assert!((scf_data.energies["xdh_energy"][0] - -56.19186092022621).abs() < 1e-6);
    assert!((scf_data.energies["pt2"][0] - -0.6356496452322714).abs() < 1e-6);
}

#[test]
fn test_hybrid_and_dh_direct_unrestricted() {
    // TODO: this test has not passed.
    let input_token = CTRL_STR
        .replace("SPIN_POLARIZATION", "true")
        .replace("CHARGE", "2.0")
        .replace("SPIN", "3.0")
        .replace("mp2", "xyg3");

    println!("=== usual calculation ===");
    let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);
    let _ = xdh_calculations(&mut scf_data, &None);
    println!("Energy table: {:#?}", scf_data.energies);
    assert!((scf_data.energies["xdh_energy"][0] - -55.04682982561314).abs() < 1e-6);
    assert!((scf_data.energies["pt2"][0] - -0.2831221826466716).abs() < 1e-6);

    println!("=== ri-direct calculation ===");
    let input_token = input_token.replace("ri-incore", "ri-direct");
    let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = SCF::build(mol, &None);
    assert!(scf_data.rimatr.is_none());
    scf_without_build(&mut scf_data, &None);
    assert!(scf_data.rimatr.is_none());
    let _ = xdh_calculations(&mut scf_data, &None);
    assert!(scf_data.rimatr.is_none());
    println!("Energy table: {:#?}", scf_data.energies);
    assert!((scf_data.energies["xdh_energy"][0] - -55.04682982561314).abs() < 1e-6);
    assert!((scf_data.energies["pt2"][0] - -0.2831221826466716).abs() < 1e-6);
}

#[test]
fn test_scs_rpa_without_rimatr() {
    let input_token = CTRL_STR
        .replace("SPIN_POLARIZATION", "false")
        .replace("CHARGE", "0.0")
        .replace("SPIN", "1.0")
        .replace("mp2", "scsrpa");

    println!("=== usual calculation ===");
    let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = SCF::build(mol.clone(), &None);
    scf_io::scf_without_build(&mut scf_data, &None);
    let _ = ri_pt2::xdh_calculations(&mut scf_data, &None);
    println!("Energy table: {:#?}", scf_data.energies);
    assert!((scf_data.energies["scsrpa"][0] - -0.4654137443706769).abs() < 1e-6);

    println!("=== ri-direct calculation ===");
    let input_token = input_token.replace("ri-incore", "ri-direct");
    let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = SCF::build(mol.clone(), &None);
    scf_io::scf_without_build(&mut scf_data, &None);
    assert!(scf_data.rimatr.is_none());
    let _ = ri_pt2::xdh_calculations(&mut scf_data, &None);
    assert!(scf_data.rimatr.is_none());
    assert!((scf_data.energies["scsrpa"][0] - -0.4654137443706769).abs() < 1e-6);
}

#[test]
fn test_scs_rpa_without_rimatr_unrestricted() {
    let input_token = CTRL_STR
        .replace("SPIN_POLARIZATION", "true")
        .replace("CHARGE", "2.0")
        .replace("SPIN", "3.0")
        .replace("mp2", "scsrpa");

    println!("=== usual calculation ===");
    let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = SCF::build(mol.clone(), &None);
    scf_io::scf_without_build(&mut scf_data, &None);
    let _ = ri_pt2::xdh_calculations(&mut scf_data, &None);
    println!("Energy table: {:#?}", scf_data.energies);
    assert!((scf_data.energies["scsrpa"][0] - -0.31052645965358466).abs() < 1e-6);

    println!("=== ri-direct calculation ===");
    let input_token = input_token.replace("ri-incore", "ri-direct");
    let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = SCF::build(mol.clone(), &None);
    scf_io::scf_without_build(&mut scf_data, &None);
    assert!(scf_data.rimatr.is_none());
    let _ = ri_pt2::xdh_calculations(&mut scf_data, &None);
    assert!(scf_data.rimatr.is_none());
    println!("Energy table: {:#?}", scf_data.energies);
    assert!((scf_data.energies["scsrpa"][0] - -0.31052645965358466).abs() < 1e-6);
}

static CTRL_STR: &str = r##"
[ctrl]
    print_level =          2
    xc =                   "mp2"
    basis_path =           "basis-set-pool/def2-tzvp"
    auxbas_path =          "basis-set-pool/def2-universal-jkfit"
    eri_type =             "ri-v"
    charge =               CHARGE
    algorithm_jk =         "ri-incore"
    spin =                 SPIN
    spin_polarization =    SPIN_POLARIZATION
    initial_guess=         "sad"
    mixer =                "diis"
    num_threads =          16

[ctrl.ri_pt2]
fp_mode = "FP64"

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

fn initialize_nh3() -> SCF {
    let input_token = CTRL_STR.replace("SPIN_POLARIZATION", "false").replace("CHARGE", "0.0").replace("SPIN", "1.0");
    let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = SCF::build(mol, &None);
    scf_io::scf_without_build(&mut scf_data, &None);
    return scf_data;
}

fn initialize_nh3_cation() -> SCF {
    let input_token = CTRL_STR.replace("SPIN_POLARIZATION", "true").replace("CHARGE", "2.0").replace("SPIN", "3.0");
    let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = SCF::build(mol, &None);
    scf_io::scf_without_build(&mut scf_data, &None);
    return scf_data;
}
