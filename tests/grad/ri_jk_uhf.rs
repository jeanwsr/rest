#![allow(non_snake_case)]

use pyrest::grad::uhf::*;

use pyrest::ctrl_io;
use pyrest::molecule_io::Molecule;
use pyrest::scf_io::{self, scf_without_build, SCF};
use rstsr::prelude::*;

static INPUT_NH3: &str = r##"
[ctrl]
    print_level =          2
    num_threads =          16
    xc =                   "hf"
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

[ctrl.j2c_decomp]
policy = "POLICY"
uplo = "UPLO"

[geom]
    name = "NH3"
    unit = "Angstrom"
    position = """
        N   0.0         0.0         0.0       
        H   1.0         0.0         0.0       
        H  -0.34202014  0.0         0.93969262
        H  -0.34202014  0.81379768 -0.46984631
    """
"##;

fn test_nh3_with_arg(policy: &str, uplo: &str) {
    let input_token = INPUT_NH3.replace("POLICY", policy).replace("UPLO", uplo);
    let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);

    let time = std::time::Instant::now();
    let scf_grad = test_with_scf(&scf_data);
    println!("Time elapsed: {:?}", time.elapsed());

    let de = scf_grad.result.get("de").unwrap().clone();
    let de = rt::asarray((de.data, de.size));
    #[rustfmt::skip]
    let de_ref = vec![
        -0.1359097671,  0.1266556604,  0.0731246885,
        -0.0600737556,  0.0271455131,  0.0156724662,
         0.0979917607, -0.0268348940, -0.1311149038,
         0.0979917620, -0.1269662794,  0.0423177491,
    ];
    let de_ref = rt::asarray((&de_ref, [3, 4]));
    println!("Maximum Error {:?}", (&de_ref - &de).abs().max_all());
    assert!((de_ref - de).abs().max_all() < 1.0e-5);
}

#[test]
fn test_nh3_eig() {
    test_nh3_with_arg("eig", "Upper");
}

#[test]
fn test_nh3_cd_upper() {
    test_nh3_with_arg("cd", "Upper");
}

#[test]
fn test_nh3_cd_lower() {
    test_nh3_with_arg("cd", "Lower");
}

#[test]
fn test_hi() {
    let scf_data = initialize_hi();
    let time = std::time::Instant::now();
    let scf_grad = test_with_scf(&scf_data);
    println!("Time elapsed: {:?}", time.elapsed());

    let de = scf_grad.result.get("de").unwrap().clone();
    let de = rt::asarray((de.data, de.size));
    #[rustfmt::skip]
    let de_ref = vec![
        -0.0000000000, 0.0000000000,  0.0021577095,
         0.0000000000, 0.0000000000, -0.0021577095,
    ];
    let de_ref = rt::asarray((&de_ref, [3, 2]));
    assert!((de_ref - de).abs().max_all() < 1.0e-5);
}

#[test]
#[ignore = "stress test"]
fn test_c12h26() {
    let scf_data = initialize_c12h26();
    let time = std::time::Instant::now();
    test_with_scf(&scf_data);
    println!("Time elapsed: {:?}", time.elapsed());
}

fn test_with_scf(scf_data: &'_ SCF) -> RIUHFGradient<'_> {
    let mut scf_grad = RIUHFGradient::new(&scf_data);
    scf_grad.calc();

    println!("=== de ===");
    let de = scf_grad.result.get("de").unwrap().clone();
    let de = rt::asarray((de.data, de.size));
    println!("{:12.6}", de.t());

    println!("=== de_nuc ===");
    scf_grad.result.get("de_nuc").map(|de| {
        let de = rt::asarray((&de.data, de.size));
        println!("{:12.6}", de.t());
    });

    println!("=== de_ovlp ===");
    scf_grad.result.get("de_ovlp").map(|de| {
        let de = rt::asarray((&de.data, de.size));
        println!("{:12.6}", de.t());
    });

    println!("=== de_hcore ===");
    scf_grad.result.get("de_hcore").map(|de| {
        let de = rt::asarray((&de.data, de.size));
        println!("{:12.6}", de.t());
    });

    println!("=== de_j ===");
    scf_grad.result.get("de_j").map(|de| {
        let de = rt::asarray((&de.data, de.size));
        println!("{:12.6}", de.t());
    });

    println!("=== de_k ===");
    scf_grad.result.get("de_k").map(|de| {
        let de = rt::asarray((&de.data, de.size));
        println!("{:12.6}", de.t());
    });

    println!("=== de_jaux ===");
    scf_grad.result.get("de_jaux").map(|de| {
        let de = rt::asarray((&de.data, de.size));
        println!("{:12.6}", de.t());
    });

    println!("=== de_kaux ===");
    scf_grad.result.get("de_kaux").map(|de| {
        let de = rt::asarray((&de.data, de.size));
        println!("{:12.6}", de.t());
    });

    return scf_grad;
}

fn initialize_hi() -> SCF {
    let input_token = r##"
[ctrl]
    print_level =               2
    num_threads =               16
    xc =                        "mp2"
    basis_path =                "def2-tzvp"
    auxbas_path =               "def2-universal-jkfit"
    charge =                    -1.0
    spin =                      2.0
    spin_polarization =         true
    mixer =                     "diis"
    num_max_diis =              8
    start_diis_cycle =          1
    mix_param =                 0.6
    max_scf_cycle =             100
    initial_guess =             "hcore"
    auxbasis_response =         true

[geom]
    name = "HI"
    unit = "Angstrom"
    position = """
        H   0.0  0.0  0.0
        I   0.0  0.0  3.0
    """
"##;
    let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);
    return scf_data;
}

fn initialize_c12h26() -> SCF {
    let input_token = r##"
[ctrl]
    print_level =          2
    xc =                   "mp2"
    basis_path =           "def2-tzvp"
    auxbas_path =          "def2-universal-jkfit"
    basis_type =           "spheric"
    eri_type =             "ri-v"
    auxbas_type =          "spheric"
    guessfile =            "none"
    chkfile =              "none"
    charge =               2.0
    spin =                 3.0
    spin_polarization =    true
    external_grids =       "none"
    initial_guess=         "sad"
    mixer =                "diis"
    num_max_diis =         8
    start_diis_cycle =     3
    mix_param =            0.8
    max_scf_cycle =        10
    scf_acc_rho =          1.0e-6
    scf_acc_eev =          1.0e-9
    scf_acc_etot =         1.0e-11
    num_threads =          16

[geom]
    name = "C12H26"
    unit = "Angstrom"
    position = """
        C          0.99590        0.00874        0.02912
        C          2.51497        0.01491        0.04092
        C          3.05233        0.96529        1.10755
        C          4.57887        0.97424        1.12090
        C          5.10652        1.92605        2.19141
        C          6.63201        1.93944        2.20819
        C          7.15566        2.89075        3.28062
        C          8.68124        2.90423        3.29701
        C          9.20897        3.85356        4.36970
        C         10.73527        3.86292        4.38316
        C         11.27347        4.81020        5.45174
        C         12.79282        4.81703        5.46246
        H          0.62420       -0.67624       -0.73886
        H          0.60223        1.00743       -0.18569
        H          0.59837       -0.31563        0.99622
        H          2.88337        0.31565       -0.94674
        H          2.87961       -1.00160        0.22902
        H          2.67826        0.66303        2.09347
        H          2.68091        1.98005        0.91889
        H          4.95608        1.27969        0.13728
        H          4.95349       -0.03891        1.31112
        H          4.72996        1.62033        3.17524
        H          4.73142        2.93925        2.00202
        H          7.00963        2.24697        1.22537
        H          7.00844        0.92672        2.39728
        H          6.77826        2.58280        4.26344
        H          6.77905        3.90354        3.09209
        H          9.05732        3.21220        2.31361
        H          9.05648        1.89051        3.48373
        H          8.83233        3.54509        5.35255
        H          8.83406        4.86718        4.18229
        H         11.10909        4.16750        3.39797
        H         11.10701        2.84786        4.56911
        H         10.90576        4.50686        6.43902
        H         10.90834        5.82716        5.26681
        H         13.18649        3.81699        5.67312
        H         13.16432        5.49863        6.23161
        H         13.18931        5.14445        4.49536
    """
"##;
    let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);
    return scf_data;
}
