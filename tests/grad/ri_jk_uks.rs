#![allow(non_snake_case)]

use pyrest::grad::uhf::RIUHFGradient;

use pyrest::ctrl_io;
use pyrest::molecule_io::Molecule;
use pyrest::scf_io::{self, scf_without_build, SCF};
use rstsr::prelude::*;

type TsrCpu<T> = Tensor<T, DeviceCpu, IxD>;

static INPUT_NH3: &str = r##"
[ctrl]
    print_level =          2
    num_threads =          16
    xc =                   "FUNCTIONAL"
    basis_path =           "def2-tzvp"
    auxbas_path =          "def2-universal-jkfit"
    eri_type =             "ri-v"
    charge =               1.0
    spin =                 2.0
    spin_polarization =    true
    auxbasis_response =    true

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

fn test_nh3_with_arg(xc: &str, policy: &str, uplo: &str) -> TsrCpu<f64> {
    let input_token = INPUT_NH3.replace("FUNCTIONAL", xc).replace("POLICY", policy).replace("UPLO", uplo);
    let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);

    let time = std::time::Instant::now();
    let scf_grad = test_with_scf(&scf_data);
    println!("Time elapsed: {:?}", time.elapsed());

    let de = scf_grad.result.get("de").unwrap().clone();
    rt::asarray((de.data, de.size))
}

#[test]
fn test_nh3_eig_b3lyp() {
    let de = test_nh3_with_arg("b3lyp", "eig", "Upper");
    #[rustfmt::skip]
    let de_ref = rt::tensor_from_nested!([
        [-0.0134772394, -0.0368314087, -0.0212654492],
        [-0.0278440671,  0.0219159674,  0.0126533722],
        [ 0.0206610545,  0.0228746967, -0.0223963316],
        [ 0.0206613567, -0.0079582572,  0.0310082458],
    ]).into_reverse_axes();
    // rtol: 1e-4, atol: 1e-6
    assert!(rt::allclose(&de, &de_ref, (1e-4, 1e-6)));
}

#[test]
fn test_nh3_cd_upper() {
    let de = test_nh3_with_arg("camb3lyp", "cd", "Upper");
    #[rustfmt::skip]
    let de_ref = rt::tensor_from_nested!([
        [-0.0136363757, -0.0372487878, -0.0215063   ],
        [-0.0278315693,  0.0220667284,  0.0127403215],
        [ 0.0207343684,  0.0230230275, -0.0223452723],
        [ 0.0207345219, -0.0078399791,  0.0311111784],
    ]).into_reverse_axes();
    // rtol: 1e-4, atol: 1e-6
    assert!(rt::allclose(&de, &de_ref, (1e-4, 1e-6)));
}

fn test_with_scf(scf_data: &'_ SCF) -> RIUHFGradient<'_> {
    let mut scf_grad = RIUHFGradient::new(scf_data);
    scf_grad.calc_uks();

    for grad_key in
        ["de", "de_nuc", "de_ovlp", "de_hcore", "de_j", "de_k", "de_sr", "de_jaux", "de_kaux", "de_sraux", "de_xc", "de_r", "de_raux"]
    {
        if let Some(de) = scf_grad.result.get(grad_key) {
            let de = rt::asarray((&de.data, de.size));
            println!("=== {} ===", grad_key);
            println!("{:12.6}", de.t());
        }
    }

    return scf_grad;
}
