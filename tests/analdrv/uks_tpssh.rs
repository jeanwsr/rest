use pyrest::analdrv::config::AnalDrvConfig;
use pyrest::analdrv::uscf_interface::uscf_hess_interface;

use pyrest::ctrl_io;
use pyrest::molecule_io::Molecule;
use pyrest::scf_io::{self, scf_without_build};

use rstsr::prelude::*;

static INPUT_NH3: &str = r##"
[ctrl]
    print_level =          2
    num_threads =          16
    xc =                   "TPSSh"
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
    xc_parser =            "parse_xc"

[geom]
    name = "NH3"
    unit = "Angstrom"
    position = """
        N  0.0  0.0  0.0
        H  1.0  0.1  0.2
        H  0.3  1.1  0.2
        H  0.1  0.1  1.2
    """
"##;

#[test]
fn test_nh3() {
    // Note this test requires 10+ seconds to run.
    let keys = toml::from_str::<serde_json::Value>(&INPUT_NH3[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);

    let config = AnalDrvConfig::default();
    let (de, vib, _th) = uscf_hess_interface(&mut scf_data, &config);

    let natm = 4;
    let de = rt::asarray((&de, [3, 3, natm, natm]));
    println!("Hessian:\n{:12.6}", de.t());

    // reference freqs from gaussian
    // we allow positive frequencies to be in 2cm^-1 error for this case
    let ref_freqs = [-5082.3895, -742.4607, 1195.1428, 2035.0250, 2250.3643, 3429.0955];
    for (k, &i) in vib.vib_indices().iter().enumerate() {
        if !vib.imag[i] {
            assert!((vib.omega[i] - ref_freqs[k]).abs() < 2.0, "freq {}: {} != {}", i, vib.omega[i], ref_freqs[k]);
        }
    }
}
