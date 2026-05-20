#![allow(non_snake_case)]

use pyrest::grad::rhf::RIRHFGradient;

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
    charge =               0.0
    spin =                 1.0
    spin_polarization =    false
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
        [ 0.0058725056,  0.0109097212,  0.0062976214],
        [-0.0091754497, -0.0017783237, -0.0010264264],
        [ 0.0016517872, -0.0005450584, -0.009598417 ],
        [ 0.0016520901, -0.008584801 ,  0.0043271104],
    ]).into_reverse_axes();
    // rtol: 1e-4, atol: 1e-6
    assert!(rt::allclose(&de, &de_ref, (1e-4, 1e-6)));
}

#[test]
fn test_nh3_cd_upper() {
    let de = test_nh3_with_arg("camb3lyp", "cd", "Upper");
    #[rustfmt::skip]
    let de_ref = rt::tensor_from_nested!([
        [ 0.0051224864,  0.0089901717,  0.005189553 ],
        [-0.0082151627, -0.0013529381, -0.0007809509],
        [ 0.0015466826, -0.0001238216, -0.0086028265],
        [ 0.0015468436, -0.007512061 ,  0.0041941748],
    ]).into_reverse_axes();
    // rtol: 1e-4, atol: 1e-6
    assert!(rt::allclose(&de, &de_ref, (1e-4, 1e-6)));
}

fn test_with_scf(scf_data: &'_ SCF) -> RIRHFGradient<'_> {
    let mut scf_grad = RIRHFGradient::new(scf_data);
    scf_grad.calc_rks();

    for grad_key in
        ["de", "de_nuc", "de_ovlp", "de_hcore", "de_j", "de_k", "de_jaux", "de_kaux", "de_xc", "de_r", "de_raux"]
    {
        if let Some(de) = scf_grad.result.get(grad_key) {
            let de = rt::asarray((&de.data, de.size));
            println!("=== {} ===", grad_key);
            println!("{:12.6}", de.t());
        }
    }

    return scf_grad;
}
