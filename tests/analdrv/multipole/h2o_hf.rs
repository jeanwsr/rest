//! SCF-level (HF, RI-JK) multipole moments of an asymmetric H2O (def2-TZVP): dipole and
//! quadrupole verified against pyscf (`local-runs/260911-multipole/gen_refs.py`, pyscf 2.14.0,
//! RI-JK with def2-universal-jkfit), octupole and hexadecapole against Gaussian 16 RevB.01
//! exact HF (`local-runs/260911-multipole/g16/`; default `#p` output contains the moments
//! through hexadecapole, so no extra keyword is needed), plus origin-invariance (dipole) and
//! origin-shift-formula (quadrupole) checks.

use pyrest::analdrv::multipole::rmultipole::RMultipoleDH;
use pyrest::ctrl_io;
use pyrest::molecule_io::Molecule;
use pyrest::ri_jk::util::get_cint_mol;
use pyrest::scf_io::{self, scf_without_build};
use pyrest::utilities::rstsr_util::RestTensorToRstsrTsrAPI;

use rstsr::prelude::*;

static INPUT_H2O: &str = r##"
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

[geom]
    name = "H2O"
    unit = "Angstrom"
    position = """
    O  0.0   0.0   0.0
    H  0.75  0.55  0.10
    H -0.70  0.60 -0.05
    """
"##;

#[test]
fn test_h2o() {
    let keys = toml::from_str::<serde_json::Value>(&INPUT_H2O[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);

    let device = rt::DeviceBLAS::default();
    let mo_coeff = (&scf_data.eigenvectors[0]).to_rstsr(&device);
    let mo_occ = (&scf_data.occupation[0]).to_rstsr(&device);
    let mol_cint = get_cint_mol(&scf_data.mol);

    let mut rmultipole =
        RMultipoleDH::new(&mol_cint, mo_coeff.clone(), mo_occ.clone(), [0.0; 3], None, None);
    let dip_tot = rmultipole.make_dipole();
    let quad_tot = rmultipole.make_quadrupole();
    rmultipole.print_multipole(2);

    // reference: pyscf RI-JK HF (def2-tzvp / def2-universal-jkfit)
    assert!(rt::allclose(
        &rmultipole.result["dip_nuc"],
        &rt::asarray((vec![0.094486306228, 2.173185043250, 0.094486306228], &device)),
        (1e-8, 1e-9)
    ));
    assert!(rt::allclose(
        &rmultipole.result["dip_scf"],
        &rt::asarray((vec![-0.060972059624, -1.328171036861, -0.058084567563], &device)),
        (1e-6, 1e-7)
    ));
    assert!(rt::allclose(
        &dip_tot,
        &rt::asarray((vec![0.033514246605, 0.845014006389, 0.036401738665], &device)),
        (1e-6, 1e-7)
    ));

    // quadrupole refs are written in (xx, xy, xz, yx, ...) reading order; the tensors are
    // symmetric, so the column-major reshape is insensitive to the flattening order
    assert!(rt::allclose(
        &rmultipole.result["quad_nuc"],
        &rt::asarray((
            vec![3.758545729222, -0.026782986194, 0.392817130845, -0.026782986194, 2.365830447135,
                0.089276620647, 0.392817130845, 0.089276620647, 0.044638310323],
            &device
        ))
        .into_shape((3, 3)),
        (1e-8, 1e-9)
    ));
    assert!(rt::allclose(
        &rmultipole.result["quad_scf"],
        &rt::asarray((
            vec![-6.979259222626, -0.002473868942, -0.156278660899, -0.002473868942,
                -6.538641850577, -0.041120055342, -0.156278660899, -0.041120055342,
                -5.506810140372],
            &device
        ))
        .into_shape((3, 3)),
        (1e-6, 1e-7)
    ));
    assert!(rt::allclose(
        &quad_tot,
        &rt::asarray((
            vec![-3.220713493404, -0.029256855136, 0.236538469946, -0.029256855136,
                -4.172811403443, 0.048156565305, 0.236538469946, 0.048156565305,
                -5.462171830049],
            &device
        ))
        .into_shape((3, 3)),
        (1e-6, 1e-7)
    ));
    assert!(rt::allclose(
        &rmultipole.result["quad_tot_traceless"],
        &rt::asarray((
            vec![1.596778123341, -0.043885282705, 0.354807704919, -0.043885282705, 0.168631258284,
                0.072234847957, 0.354807704919, 0.072234847957, -1.765409381625],
            &device
        ))
        .into_shape((3, 3)),
        (1e-6, 1e-7)
    ));

    // octupole and hexadecapole: totals referenced to Gaussian 16 RevB.01 exact HF
    // (`local-runs/260911-multipole/g16/`, 4-decimal print; note G16's default output contains
    // these sections without any extra keyword, and its "Traceless Quadrupole" uses Q - Tr/3 I
    // without the 3/2 factor). Tolerances carry the exact-vs-RI-JK density difference (dipole/
    // quadrupole calibration: 8e-5 / 2e-4 a.u.) plus the G16 print rounding. The nuclear parts
    // are closed-form and asserted exactly against an independent computation.
    let oct_tot = rmultipole.make_octupole();
    assert!(rt::allclose(
        &rmultipole.result["oct_nuc"],
        &rt::asarray((
            vec![0.532274883, 4.071776326, 0.214259620, 4.071776326, -0.169551904, 0.420083822,
                0.214259620, 0.420083822, 0.038802923, 4.071776326, -0.169551904, 0.420083822,
                -0.169551904, 2.580394402, 0.082667098, 0.420083822, 0.082667098, 0.047238341,
                0.214259620, 0.420083822, 0.038802923, 0.420083822, 0.082667098, 0.047238341,
                0.038802923, 0.047238341, 0.005904793],
            &device
        ))
        .into_shape((3, 3, 3)),
        (1e-8, 1e-9)
    ));
    assert!(rt::allclose(
        &oct_tot,
        &rt::asarray((
            vec![0.121810, 1.158675, 0.065893, 1.158675, -0.112538, 0.184050, 0.065893, 0.184050,
                -0.021496, 1.158675, -0.112538, 0.184050, -0.112538, -0.741260, 0.008008,
                0.184050, 0.008008, -0.603854, 0.065893, 0.184050, -0.021496, 0.184050, 0.008008,
                -0.603854, -0.021496, -0.603854, -0.082471],
            &device
        ))
        .into_shape((3, 3, 3)),
        (1e-3, 5e-4)
    ));

    let hex_tot = rmultipole.make_hexadecapole();
    assert!(rt::allclose(
        &rmultipole.result["hex_nuc"],
        &rt::asarray((
            vec![7.096848175, 0.334514120, 0.756701706, 0.334514120, 4.419459962, 0.207068784,
                0.756701706, 0.207068784, 0.087354652, 0.334514120, 4.419459962, 0.207068784,
                4.419459962, -0.336905215, 0.450003985, 0.207068784, 0.450003985, 0.039213950,
                0.756701706, 0.207068784, 0.087354652, 0.207068784, 0.450003985, 0.039213950,
                0.087354652, 0.039213950, 0.010680222, 0.334514120, 4.419459962, 0.207068784,
                4.419459962, -0.336905215, 0.450003985, 0.207068784, 0.450003985, 0.039213950,
                4.419459962, -0.336905215, 0.450003985, -0.336905215, 2.819658336, 0.074442742,
                0.450003985, 0.074442742, 0.050053578, 0.207068784, 0.450003985, 0.039213950,
                0.450003985, 0.074442742, 0.050053578, 0.039213950, 0.050053578, 0.006057439,
                0.756701706, 0.207068784, 0.087354652, 0.207068784, 0.450003985, 0.039213950,
                0.087354652, 0.039213950, 0.010680222, 0.207068784, 0.450003985, 0.039213950,
                0.450003985, 0.074442742, 0.050053578, 0.039213950, 0.050053578, 0.006057439,
                0.087354652, 0.039213950, 0.010680222, 0.039213950, 0.050053578, 0.006057439,
                0.010680222, 0.006057439, 0.001354954],
            &device
        ))
        .into_shape((3, 3, 3, 3)),
        (1e-8, 1e-9)
    ));
    assert!(rt::allclose(
        &hex_tot,
        &rt::asarray((
            vec![-14.953222, 0.114165, 0.272403, 0.114165, -3.900990, 0.091066, 0.272403,
                0.091066, -5.793474, 0.114165, -3.900990, 0.091066, -3.900990, -0.232312,
                0.186115, 0.091066, 0.186115, 0.003186, 0.272403, 0.091066, -5.793474, 0.091066,
                0.186115, 0.003186, -5.793474, 0.003186, -0.148414, 0.114165, -3.900990,
                0.091066, -3.900990, -0.232312, 0.186115, 0.091066, 0.186115, 0.003186,
                -3.900990, -0.232312, 0.186115, -0.232312, -16.950047, -0.012478, 0.186115,
                -0.012478, -5.730551, 0.091066, 0.186115, 0.003186, 0.186115, -0.012478,
                -5.730551, 0.003186, -5.730551, -0.048852, 0.272403, 0.091066, -5.793474,
                0.091066, 0.186115, 0.003186, -5.793474, 0.003186, -0.148414, 0.091066, 0.186115,
                0.003186, 0.186115, -0.012478, -5.730551, 0.003186, -5.730551, -0.048852,
                -5.793474, 0.003186, -0.148414, 0.003186, -5.730551, -0.048852, -0.148414,
                -0.048852, -16.091951],
            &device
        ))
        .into_shape((3, 3, 3, 3)),
        (1e-3, 1e-3)
    ));

    // second evaluation at a shifted origin: the dipole must be invariant (neutral molecule),
    // and the quadrupole must follow Theta' = Theta - R mu^T - mu R^T + Q R R^T (Q = 0 here)
    let origin2 = [0.3, -0.2, 0.15];
    let mut rmultipole2 = RMultipoleDH::new(&mol_cint, mo_coeff, mo_occ, origin2, None, None);
    let dip2 = rmultipole2.make_dipole();
    let quad2 = rmultipole2.make_quadrupole();

    assert!(rt::allclose(&dip2, &dip_tot, (1e-9, 1e-10)));

    let r = rt::asarray((origin2.to_vec(), &device));
    let quad_shift =
        &(r.i((.., None)) * dip_tot.i((None, ..))) + &(dip_tot.i((.., None)) * r.i((None, ..)));
    let quad_pred = &quad_tot - &quad_shift;
    assert!(rt::allclose(&quad2, &quad_pred, (1e-9, 1e-9)));

    for (label, t) in &rmultipole.timing {
        println!("Timing {label}: {t:.3} s");
    }
}
