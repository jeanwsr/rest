//! SCF-level (B3LYP, RI-JK) multipole moments of an asymmetric H2O (def2-TZVP), verified against
//! pyscf (`local-runs/260911-multipole/gen_refs.py`, pyscf 2.14.0, RI-JK with
//! def2-universal-jkfit). Tolerances carry the REST/pyscf grid differences (~1e-5).

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
    xc =                   "b3lyp"
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

    let mut rmultipole = RMultipoleDH::new(&mol_cint, mo_coeff, mo_occ, [0.0; 3], None, None);
    let dip_tot = rmultipole.make_dipole();
    let quad_tot = rmultipole.make_quadrupole();
    rmultipole.print_multipole(2);

    // reference: pyscf RI-JK B3LYP (def2-tzvp / def2-universal-jkfit)
    assert!(rt::allclose(
        &rmultipole.result["dip_scf"],
        &rt::asarray((vec![-0.062747929156, -1.358344238525, -0.059445088550], &device)),
        (1e-5, 1e-6)
    ));
    assert!(rt::allclose(
        &dip_tot,
        &rt::asarray((vec![0.031738377072, 0.814840804724, 0.035041217679], &device)),
        (1e-5, 1e-6)
    ));
    assert!(rt::allclose(
        &quad_tot,
        &rt::asarray((
            vec![-3.358418468690, -0.029750730554, 0.231736911888, -0.029750730554,
                -4.303781449787, 0.046590665639, 0.231736911888, 0.046590665639,
                -5.554802927163],
            &device
        ))
        .into_shape((3, 3)),
        (1e-5, 1e-6)
    ));

    for (label, t) in &rmultipole.timing {
        println!("Timing {label}: {t:.3} s");
    }
}
