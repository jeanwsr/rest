//! RI-MP2 (relaxed, Z-vector) electric moments of HI (def2-TZVP with def2-ECP on iodine),
//! driven through the full `analdrv_interface` task path.
//!
//! This is the ECP integration test of the multipole task: the nuclear moments are built from
//! the ECP-effective charges (`CInt::atom_charges`, Z - n_ecp = 25 for iodine), so both the
//! nuclear charge contribution and the total moments must match Gaussian 16, which uses the same
//! valence-electron convention for ECP systems.
//!
//! References: Gaussian 16 Legacy RevB.01, `#p MP2(Full)/def2TZVP Density=MP2 NoSymm` (relaxed
//! MP2 density; note G16's default MP2 freezes the I 4s/4p subshells, so `Full` is required to
//! match REST's `start_mo = 0`), geometry H(0,0,0), I(0,0,1.609 Angstrom), run recorded at
//! `local-runs/260912-hi-ecp/g16`:
//!
//! - E(SCF) = -297.242009239, E2 = -0.5911611131, EUMP2 = -297.83317035213
//! - dipole Z = -0.5630 D (G16 prints 4 decimals; 1e-5 a.u. rounding)
//! - quadrupole XX = YY = -28.8681, ZZ = -25.7420 D.Ang
//! - traceless quadrupole XX = -1.0420, ZZ = 2.0841 D.Ang
//!
//! Conversion to a.u. divides by 2.54174623 (dipole) and 2.54174623 * 0.529177210903
//! (quadrupole). Two conventions must be respected when comparing:
//!
//! - G16 prints the (raw and traceless) quadrupole about the **coordinate origin**, not the
//!   center of mass (verified against REST by the mu * Delta-origin shift law); the dipole of a
//!   neutral molecule is origin-independent. The test therefore pins
//!   `multipole_origin = [0, 0, 0]`.
//! - G16's "Traceless Quadrupole" is `Q - Tr(Q)/3 * I` **without** the 3/2 factor of REST's
//!   `quadrupole_to_traceless`, so the G16 traceless values are scaled by 1.5 below.
//!
//! The RI tolerance (`def2-universal-jkfit` for both RI-J and RI-MP2) dominates the deviation:
//! ~1.3e-4 Ha on the correlation energy and ~8e-4 a.u. on the dipole total.

use pyrest::analdrv::config::{AnalDrvConfig, AnalDrvTask};
use pyrest::analdrv::interface::analdrv_interface;
use pyrest::ctrl_io;
use pyrest::molecule_io::Molecule;
use pyrest::ri_pt2;
use pyrest::scf_io::{self, scf_without_build};

static INPUT_HI: &str = r##"
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
    start_diis_cycle =     1
    mix_param =            0.6
    max_scf_cycle =        100

[ctrl.ri_pt2]
    # the multipole property path follows this keyword (FP32 is the global default); pinned to
    # FP64 here for deterministic thread-independent references
    fp_mode =              "FP64"

[geom]
    name = "HI"
    unit = "Angstrom"
    position = """
    H  0.0  0.0  0.0
    I  0.0  0.0  1.609
    """
"##;

fn allclose(v: &[f64], ref_: &[f64], tol: f64) {
    assert_eq!(v.len(), ref_.len());
    for (a, b) in v.iter().zip(ref_.iter()) {
        assert!((a - b).abs() < tol, "component mismatch: {a} vs {b}");
    }
}

#[test]
fn test_hi_rmp2_ecp() {
    let keys = toml::from_str::<serde_json::Value>(&INPUT_HI[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);

    // RI-MP2 energy (also materializes `rimatr` consumed by the multipole task); G16 reference
    // EUMP2 = -297.83317035213, RI fitting error ~1.3e-4 Ha
    let eng = ri_pt2::xdh_calculations(&mut scf_data, &None).unwrap();
    println!("RI-MP2 total energy: {eng}");
    assert!((eng - (-297.833170352)).abs() < 3e-4, "RI-MP2 total energy mismatch vs G16");

    let tasks = vec![AnalDrvTask::Multipole];
    let mut config = AnalDrvConfig::default();
    config.multipole.orders = vec![1, 2];
    // G16 evaluates its multipole output at the coordinate origin
    config.multipole.origin = Some([0.0; 3]);
    // `rdm1_relax` stays at its default (relaxed): the Z-vector response increment is included

    let output = analdrv_interface(&mut scf_data, &tasks, &config);
    let mp = output.multipole.as_ref().expect("multipole task output");

    // --- nuclear dipole: the ECP check. Iodine contributes Z_eff = 53 - 28 = 25 (not 53) at
    // z = 1.609 Angstrom; a full-charge convention would give 53/25 * 76.014 instead. --- //
    let bohr = 0.52917721092; // REST's constants::BOHR
    let dip_nuc_expected = 25.0 * 1.609 / bohr;
    let dip = mp.dipole.as_ref().expect("dipole output");
    allclose(&dip.nuc[..2], &[0.0, 0.0], 1e-9);
    assert!(
        (dip.nuc[2] - dip_nuc_expected).abs() < 1e-8,
        "nuclear dipole must use the ECP-effective charge 25, got {}",
        dip.nuc[2]
    );

    // relaxed RI-MP2: both the unrelaxed correlation and the Z-vector response increments exist
    let dip_corr = dip.corr.as_ref().expect("relaxed mode must carry the corr. increment");
    let dip_resp = dip.resp.as_ref().expect("relaxed mode must carry the response increment");
    allclose(dip_corr, &[0.0, 0.0, -0.000033189610], 1e-8);
    allclose(dip_resp, &[0.0, 0.0, -0.001018252093], 5e-7);

    // total dipole vs G16 relaxed MP2(Full) density: -0.5630 D = -0.2215013 a.u.; the deviation
    // (~8e-4 a.u.) is dominated by the RI fit of the SCF density and of the PT2 increments
    allclose(&dip.tot, &[0.0, 0.0, -0.222299437], 5e-7);
    assert!(
        (dip.tot[2] - (-0.221501263)).abs() < 2e-3,
        "total dipole deviates from G16 beyond the RI tolerance: {}",
        dip.tot[2]
    );

    // --- quadrupole (raw second moments) vs G16 at the coordinate origin --- //
    let quad = mp.quadrupole.as_ref().expect("quadrupole output");
    allclose(
        &quad.tot,
        &[
            -21.463558461, -0.0, -0.0, -0.0, -21.463558461, -0.0, -0.0, -0.0, -19.141266783,
        ],
        5e-3,
    );
    assert!((quad.tot[0] - (-21.462726)).abs() < 5e-3, "quadrupole XX vs G16");
    assert!(
        (quad.tot[8] - (-19.138547)).abs() < 1e-2,
        "quadrupole ZZ deviates from G16 beyond the RI tolerance: {}",
        quad.tot[8]
    );

    // traceless form: REST = 3/2 * (Q - Tr(Q)/3 * I); G16's traceless section omits the 3/2, so
    // the reference is (G16 traceless) * 1.5
    let traceless = quad.traceless.as_ref().expect("traceless quadrupole output");
    allclose(traceless, &[-1.161145839, 0.0, 0.0, 0.0, -1.161145839, 0.0, 0.0, 0.0, 2.322291678], 5e-3);
    assert!((traceless[0] - (-1.162052)).abs() < 5e-3, "traceless XX vs 1.5 * G16");
    assert!((traceless[8] - 2.324216).abs() < 5e-3, "traceless ZZ vs 1.5 * G16");
}
