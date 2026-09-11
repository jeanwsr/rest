//! CO2 (HF, RI-JK, def2-TZVP): the neutral linear molecule has vanishing dipole, so its raw
//! quadrupole is origin-invariant. Quadrupole reference from pyscf
//! (`local-runs/260911-multipole/gen_refs.py`, pyscf 2.14.0).

use pyrest::analdrv::multipole::rmultipole::RMultipoleDH;
use pyrest::ctrl_io;
use pyrest::molecule_io::Molecule;
use pyrest::ri_jk::util::get_cint_mol;
use pyrest::scf_io::{self, scf_without_build};
use pyrest::utilities::rstsr_util::RestTensorToRstsrTsrAPI;

use rstsr::prelude::*;

static INPUT_CO2: &str = r##"
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
    name = "CO2"
    unit = "Angstrom"
    position = """
    C  0.0  0.0  0.0
    O  0.0  0.0  1.16
    O  0.0  0.0 -1.16
    """
"##;

#[test]
fn test_co2() {
    let keys = toml::from_str::<serde_json::Value>(&INPUT_CO2[..]).unwrap();
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

    // vanishing dipole
    assert!((&dip_tot * &dip_tot).sum().sqrt() < 1e-8);

    // reference: pyscf RI-JK HF (def2-tzvp / def2-universal-jkfit)
    assert!(rt::allclose(
        &quad_tot,
        &rt::asarray((
            vec![-10.954502654334, 0.0, 0.0, 0.0, -10.954502654334, 0.0, 0.0, 0.0,
                -14.803346728346],
            &device
        ))
        .into_shape((3, 3)),
        (1e-6, 1e-7)
    ));

    // mu = 0 (and Q = 0): the raw quadrupole is exactly origin-invariant
    let origin2 = [0.3, -0.2, 0.15];
    let mut rmultipole2 = RMultipoleDH::new(&mol_cint, mo_coeff, mo_occ, origin2, None, None);
    let dip2 = rmultipole2.make_dipole();
    let quad2 = rmultipole2.make_quadrupole();
    assert!(rt::allclose(&dip2, &dip_tot, (1e-9, 1e-10)));
    assert!(rt::allclose(&quad2, &quad_tot, (1e-8, 1e-9)));
}
