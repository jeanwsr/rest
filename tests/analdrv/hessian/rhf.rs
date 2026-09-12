use pyrest::analdrv::config::AnalDrvConfig;
use pyrest::analdrv::hessian::rscf_interface::rscf_hess_interface;
use pyrest::analdrv::response::rresp_interface::rscf_resp_interface;

use pyrest::ctrl_io;
use pyrest::molecule_io::Molecule;
use pyrest::scf_io::{self, scf_without_build};

use rstsr::prelude::*;

static INPUT_NH3: &str = r##"
[ctrl]
    print_level =          2
    num_threads =          4
    xc =                   "hf"
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

[analdrv]
gau_thermo = true
resp_tol = 1e-9
resp_tol_inflation = 1000

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
    let keys = toml::from_str::<serde_json::Value>(&INPUT_NH3[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);

    let cfg = scf_data.mol.ctrl.analdrv.as_ref().unwrap().clone();
    let mut resp_objs = rscf_resp_interface(&scf_data, &cfg);
    let (de, vib, th) = rscf_hess_interface(&scf_data, &cfg.nucgrad, &mut resp_objs);
    let th = th.unwrap();

    let ref_freqs = [1263.343780, 1367.102321, 1424.072405, 2132.997526, 2443.140863, 3517.051480];
    println!("freqs: {:?}", vib.omega);
    for (k, &i) in vib.vib_indices().iter().enumerate() {
        assert!((vib.omega[i] - ref_freqs[k]).abs() < 1e-2, "freq {}: {} != {}", i, vib.omega[i], ref_freqs[k]);
    }

    pub const TRANS: usize = 1;
    pub const VIB: usize = 3;

    assert!((th.zpe[VIB] - 0.027674515751).abs() < 1e-6);
    assert!((th.s[TRANS] - 0.054886124656).abs() < 1e-6);
    assert!((th.cv_tot - 0.010125123641).abs() < 1e-6);
    assert!((th.cp_tot - 0.013291934132).abs() < 1e-6);

    let natm = 4;
    let de = rt::asarray((&de, [3, 3, natm, natm]));
    println!("Hessian:\n{:12.6}", de.t());
}

static INPUT_SBH3_HBR: &str = r##"
[ctrl]
    print_level =          2
    num_threads =          16
    xc =                   "hf"
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
    name = "BIH3"
    unit = "Angstrom"
    position = """
Sb         0.00000000      0.00000000      0.71474217
H          0.00000000      0.00000000     -2.10597083
Br         0.00000000      0.00000000     -3.52882583
H          0.69982699      1.21213591      1.64001815
H          0.69982699     -1.21213591      1.64001815
H         -1.39965399      0.00000000      1.64001815
    """
"##;

#[test]
fn test_sbh3_hbr() {
    let keys = toml::from_str::<serde_json::Value>(&INPUT_SBH3_HBR[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);

    let config = AnalDrvConfig::default();
    let mut resp_objs = rscf_resp_interface(&scf_data, &config);
    let (_, vib, _) = rscf_hess_interface(&scf_data, &config.nucgrad, &mut resp_objs);

    // reference value from gaussian 16
    // we allow 1 cm^-1 difference due to RI-JK/conventional difference
    let ref_freqs = [
        -169.4952, -169.4951, 65.6069, 333.1334, 333.1334, 871.2882, 922.0154, 922.0154, 2152.3976, 2152.3977,
        2166.9802, 2645.4606,
    ];
    for (k, &i) in vib.vib_indices().iter().enumerate() {
        // imag freq multiplies by negative
        let f = if vib.imag[i] { -vib.omega[i] } else { vib.omega[i] };
        assert!((f - ref_freqs[k]).abs() < 1.0, "freq {}: {} != {}", i, f, ref_freqs[k]);
    }
}
