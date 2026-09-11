//! XYG3 (xDH double hybrid) multipole moments of NH3 (def2-TZVP), driving the `RMultipoleDH`
//! evaluator with the `RGFockDH` composite and the `RRespSCF` response object.
//!
//! Dipole decomposition reference values are those recorded by
//! `local-runs/260910-xyg3/xyg3_dipole_contribs.py` (pyscf-forge master @ 0566d43, pyscf 2.14.0)
//! and already asserted contribution-by-contribution in `tests/analdrv/response/xyg3_rgfock.rs`;
//! the quadrupole SCF-density reference is from `local-runs/260911-multipole/gen_refs.py`
//! (pyscf 2.14.0, B3LYP RI-JK). The correlation/response quadrupole increments have no external
//! reference yet and are printed only.

use pyrest::analdrv::config::AnalDrvConfig;
use pyrest::analdrv::multipole::rmultipole::RMultipoleDH;
use pyrest::analdrv::response::rgfock_interface::rgfock_dh_interface;
use pyrest::analdrv::response::rresp_interface::rscf_resp_interface;
use pyrest::dft::Grids;
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

#[test]
fn test_nh3() {
    let keys = toml::from_str::<serde_json::Value>(&INPUT_NH3[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);

    let device = rt::DeviceBLAS::default();

    // 0. XYG3 total energy (side product; also frees the DFT grids inside)
    let eng = ri_pt2::xdh_calculations(&mut scf_data, &None).unwrap();
    println!("XYG3 total energy: {}", eng);
    // reference: pyscf-forge DFDH (def2-universal-jkfit for both JK and RI)
    assert!((eng - (-56.513619197590)).abs() < 2e-6, "XYG3 total energy mismatch");

    // regenerate the common SCF grids (xdh_calculations frees them for memory)
    scf_data.grids = Some(Grids::build(&mut scf_data.mol));

    // 1. driver assembly: SCF orbitals + DH composite + response object
    let mo_coeff = (&scf_data.eigenvectors[0]).to_rstsr(&device);
    let mo_occ = (&scf_data.occupation[0]).to_rstsr(&device);
    let mol_cint = get_cint_mol(&scf_data.mol);
    let config = AnalDrvConfig::default();
    let mut resp_objs = rscf_resp_interface(&scf_data, &config);
    let mut rgfock = rgfock_dh_interface::<f64>(&scf_data);

    let mut rmultipole =
        RMultipoleDH::new(&mol_cint, mo_coeff, mo_occ, [0.0; 3], Some(&mut rgfock), Some(&mut resp_objs));
    let dip_tot = rmultipole.make_dipole();
    let quad_tot = rmultipole.make_quadrupole();
    rmultipole.print_multipole(2);

    // 2. dipole decomposition (refs identical to tests/analdrv/response/xyg3_rgfock.rs)
    assert!(rt::allclose(
        &rmultipole.result["dip_nuc"],
        &rt::asarray((vec![1.700753512109, 1.889726124565, 2.078698737022], &device)),
        (1e-8, 1e-9)
    ));
    assert!(rt::allclose(
        &rmultipole.result["dip_scf"],
        &rt::asarray((vec![-1.156948912236, -1.364097621601, -1.579483131449], &device)),
        (1e-5, 1e-6)
    ));
    assert!(rt::allclose(
        &rmultipole.result["dip_corr"],
        &rt::asarray((vec![-0.002948597510, -0.004975193666, -0.007063388846], &device)),
        (1e-6, 5e-7)
    ));
    assert!(rt::allclose(
        &rmultipole.result["dip_resp"],
        &rt::asarray((vec![0.009523402231, 0.014507684671, 0.020086193759], &device)),
        (1e-5, 5e-6)
    ));
    assert!(rt::allclose(
        &dip_tot,
        &rt::asarray((vec![0.550379404594, 0.535160993970, 0.512238410486], &device)),
        (1e-5, 5e-6)
    ));

    // 3. quadrupole: nuclear part exact, SCF density part against pyscf B3LYP (grid tolerance);
    //    correlation/response increments printed only (no external reference yet)
    println!("Quadrupole (SCF density): {:16.12}", rmultipole.result["quad_scf"]);
    assert!(rt::allclose(
        &rmultipole.result["quad_nuc"],
        &rt::asarray((
            vec![2.892562508950, 0.0, 0.0, 0.0, 3.571064825864, 0.0, 0.0, 0.0, 4.320988439295],
            &device
        ))
        .into_shape((3, 3)),
        (1e-8, 1e-9)
    ));
    // tuple order is (rtol, atol); the atol floor carries the REST/pyscf grid difference, which
    // dominates for the near-zero off-diagonal components (pure rtol gives ~2e-6 there)
    assert!(rt::allclose(
        &rmultipole.result["quad_scf"],
        &rt::asarray((
            vec![-8.351340196824, -0.160186507520, -0.160276871973, -0.160186507520,
                -8.914433938876, -0.151145410624, -0.160276871973, -0.151145410624,
                -9.552865520333],
            &device
        ))
        .into_shape((3, 3)),
        (1e-4, 1e-5)
    ));
    println!("Quadrupole (corr. increment): {:16.12}", rmultipole.result["quad_corr"]);
    println!("Quadrupole (resp. increment): {:16.12}", rmultipole.result["quad_resp"]);
    println!("Quadrupole (total): {:16.12}", quad_tot);

    for (label, t) in &rmultipole.timing {
        println!("Timing {label}: {t:.3} s");
    }
}
