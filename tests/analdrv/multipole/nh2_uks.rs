//! SCF-level (UKS B3LYP, RI-JK) multipole moments of the NH2 radical (doublet, def2-TZVP),
//! verified against pyscf (2.14.0, RI-JK with def2-universal-jkfit). Tolerances carry the
//! REST/pyscf grid differences (~2e-6 on the small off-diagonal quadrupole components).

use pyrest::analdrv::multipole::umultipole::UMultipoleDH;
use pyrest::ctrl_io;
use pyrest::molecule_io::Molecule;
use pyrest::ri_jk::util::get_cint_mol;
use pyrest::scf_io::{self, scf_without_build};
use pyrest::utilities::rstsr_util::RestTensorToRstsrTsrAPI;

use rstsr::prelude::*;

static INPUT_NH2: &str = r##"
[ctrl]
    print_level =          2
    num_threads =          4
    xc =                   "b3lyp"
    basis_path =           "def2-tzvp"
    auxbas_path =          "def2-universal-jkfit"
    eri_type =             "ri-v"
    charge =               0.0
    spin =                 2.0
    spin_polarization =    true

[geom]
    name = "NH2"
    unit = "Angstrom"
    position = """
    N  0.0   0.0   0.0
    H  0.58  0.20  0.85
    H -0.50 -0.25  0.83
    """
"##;

#[test]
fn test_nh2_uks() {
    let keys = toml::from_str::<serde_json::Value>(&INPUT_NH2[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);

    let device = rt::DeviceBLAS::default();
    let mo_coeff = [(&scf_data.eigenvectors[0]).to_rstsr(&device), (&scf_data.eigenvectors[1]).to_rstsr(&device)];
    let mo_occ = [(&scf_data.occupation[0]).to_rstsr(&device), (&scf_data.occupation[1]).to_rstsr(&device)];
    let mol_cint = get_cint_mol(&scf_data.mol);

    let mut umultipole = UMultipoleDH::new(&mol_cint, mo_coeff, mo_occ, [0.0; 3]);
    let dip_tot = umultipole.make_dipole();
    let quad_tot = umultipole.make_quadrupole();
    umultipole.print_multipole(2);

    // reference: pyscf RI-JK UKS B3LYP (def2-tzvp / def2-universal-jkfit)
    assert!(rt::allclose(
        &umultipole.result["dip_scf"],
        &rt::asarray((vec![-0.136575312767, 0.057794212957, -2.312851619633], &device)),
        (1e-5, 1e-6)
    ));
    assert!(rt::allclose(
        &dip_tot,
        &rt::asarray((vec![0.014602777199, -0.036692093271, 0.861888269636], &device)),
        (1e-5, 1e-6)
    ));
    assert!(rt::allclose(
        &quad_tot,
        &rt::asarray((
            vec![-5.264151352116, 0.026124544967, 0.047625109070, 0.026124544967, -5.320049075464,
                -0.028624095927, 0.047625109070, -0.028624095927, -4.355249946458],
            &device
        ))
        .into_shape((3, 3)),
        (1e-5, 5e-6)
    ));

    for (label, t) in &umultipole.timing {
        println!("Timing {label}: {t:.3} s");
    }
}
