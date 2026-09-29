//! SCF-level (UHF, RI-JK) multipole moments of the NH2 radical (doublet, ²B1, def2-TZVP),
//! verified against pyscf (2.14.0, RI-JK with def2-universal-jkfit), plus origin-invariance
//! (dipole). NH2 is a non-degenerate open shell, unlike OH whose ²Π SOMO makes UHF/UKS
//! properties ill-conditioned.

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
    xc =                   "hf"
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
fn test_nh2_uhf() {
    let keys = toml::from_str::<serde_json::Value>(&INPUT_NH2[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);

    let device = rt::DeviceBLAS::default();
    let mo_coeff = [(&scf_data.eigenvectors[0]).to_rstsr(&device), (&scf_data.eigenvectors[1]).to_rstsr(&device)];
    let mo_occ = [(&scf_data.occupation[0]).to_rstsr(&device), (&scf_data.occupation[1]).to_rstsr(&device)];
    let mol_cint = get_cint_mol(&scf_data.mol);

    let mut umultipole = UMultipoleDH::new(&mol_cint, mo_coeff.clone(), mo_occ.clone(), [0.0; 3]);
    let dip_tot = umultipole.make_dipole();
    let quad_tot = umultipole.make_quadrupole();
    umultipole.print_multipole(2);

    // reference: pyscf RI-JK UHF (def2-tzvp / def2-universal-jkfit)
    assert!(rt::allclose(
        &umultipole.result["dip_nuc"],
        &rt::asarray((vec![0.151178089965, -0.094486306228, 3.174739889269], &device)),
        (1e-8, 1e-9)
    ));
    assert!(rt::allclose(
        &umultipole.result["dip_scf"],
        &rt::asarray((vec![-0.134521233449, 0.056010568899, -2.259650366894], &device)),
        (1e-6, 1e-7)
    ));
    assert!(rt::allclose(
        &dip_tot,
        &rt::asarray((vec![0.016656856517, -0.038475737329, 0.915089522375], &device)),
        (1e-6, 1e-7)
    ));

    assert!(rt::allclose(
        &umultipole.result["quad_nuc"],
        &rt::asarray((
            vec![2.094072413886, 0.860626623033, 0.278543056417, 0.860626623033, 0.366034144651,
                -0.133914930970, 0.278543056417, -0.133914930970, 5.040200895224],
            &device
        ))
        .into_shape((3, 3)),
        (1e-8, 1e-9)
    ));
    assert!(rt::allclose(
        &umultipole.result["quad_scf"],
        &rt::asarray((
            vec![-7.287581829724, -0.839714368491, -0.228199253137, -0.839714368491, -5.605358708602,
                0.103369473295, -0.228199253137, 0.103369473295, -9.251073894497],
            &device
        ))
        .into_shape((3, 3)),
        (1e-6, 1e-7)
    ));
    assert!(rt::allclose(
        &quad_tot,
        &rt::asarray((
            vec![-5.193509415837, 0.020912254543, 0.050343803281, 0.020912254543, -5.239324563951,
                -0.030545457675, 0.050343803281, -0.030545457675, -4.210872999273],
            &device
        ))
        .into_shape((3, 3)),
        (1e-6, 1e-7)
    ));
    assert!(rt::allclose(
        &umultipole.result["quad_tot_traceless"],
        &rt::asarray((
            vec![-0.468410634225, 0.031368381814, 0.075515704921, 0.031368381814, -0.537133356396,
                -0.045818186512, 0.075515704921, -0.045818186512, 1.005543990621],
            &device
        ))
        .into_shape((3, 3)),
        (1e-6, 1e-7)
    ));

    // shifted origin: the dipole of the neutral radical is origin-independent
    let mut umultipole2 = UMultipoleDH::new(&mol_cint, mo_coeff, mo_occ, [0.3, -0.2, 0.15]);
    let dip2 = umultipole2.make_dipole();
    assert!(rt::allclose(&dip2, &dip_tot, (1e-9, 1e-10)));

    for (label, t) in &umultipole.timing {
        println!("Timing {label}: {t:.3} s");
    }
}
