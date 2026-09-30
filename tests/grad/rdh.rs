//! Doubly-hybrid (`xc = "mp2"`) analytic gradient of distorted NH3 (restricted RI-MP2).
//!
//! Reference: pyscf-forge `DFDH(xc="MP2")` analytic gradient (pyscf 2.14.0 + pyscf-forge),
//! def2-SVP, def2-universal-jkfit for both the SCF JK fitting and the RI (3-center) part,
//! CPKS tolerance 1e-8. The JSON reference was dumped per atom in Hartree/bohr (row-major
//! `[natom, 3]`); REST stores the gradient as `[3, natm]`.
//! The correlation part below is the `"DH"` entry of `eval_force` (the `"SCF"` entry covers
//! the HF part). The rimatr must be CD-decomposed (see `src/grad/rdh.rs`).

use pyrest::grad::rdh::RDHGradient;
use pyrest::grad::traits::GradAPI;
use pyrest::molecule_io::Molecule;
use pyrest::ri_pt2;
use pyrest::scf_io::{self, scf_without_build};
use pyrest::ctrl_io;
use rstsr::prelude::*;

static INPUT_NH3: &str = r##"
[ctrl]
    print_level =          0
    num_threads =          4
    xc =                   "mp2"
    basis_path =           "def2-SVP"
    auxbas_path =          "def2-universal-jkfit"
    eri_type =             "ri-v"
    charge =               0.0
    spin =                 1.0
    spin_polarization =    false

[ctrl.j2c_decomp]
policy = "cd"

[ctrl.ri_pt2]
new_driver = true

[geom]
    name = "NH3"
    unit = "Angstrom"
    position = """
    N   0.0000000000    0.0000000000    0.0000000000
    H   0.0000000000    0.0000000000    1.1500000000
    H   0.9674007198    0.0000000000   -0.2902341250
    H  -0.4917018117    0.8516525201   -0.3062961203
    """
"##;

static INPUT_XYG3: &str = r##"
[ctrl]
    print_level =          0
    num_threads =          4
    xc =                   "xyg3"
    basis_path =           "def2-SVP"
    auxbas_path =          "def2-universal-jkfit"
    eri_type =             "ri-v"
    charge =               0.0
    spin =                 1.0
    spin_polarization =    false

[ctrl.j2c_decomp]
policy = "cd"

[ctrl.ri_pt2]
new_driver = true

[geom]
    name = "NH3"
    unit = "Angstrom"
    position = """
    N   0.0000000000    0.0000000000    0.0000000000
    H   0.0000000000    0.0000000000    1.1500000000
    H   0.9674007198    0.0000000000   -0.2902341250
    H  -0.4917018117    0.8516525201   -0.3062961203
    """
"##;

#[test]
fn test_nh3_rmp2_grad() {
    let keys = toml::from_str::<serde_json::Value>(&INPUT_NH3[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);

    // total energy (RI-MP2); also puts the SCF into the post-SCF state the gradient expects
    let eng = ri_pt2::xdh_calculations(&mut scf_data, &None).unwrap();
    println!("RI-MP2 total energy: {eng}");
    assert!((eng - (-56.32508171804932)).abs() < 2e-6, "RI-MP2 total energy mismatch");

    let mut grad = RDHGradient::new(&scf_data, &None);
    grad.calc();
    let de = grad.result.get("de").unwrap().clone();
    let de = rt::asarray((de.data, de.size));

    // pyscf-forge `gradient_corr_only_hartree_per_bohr` (natom, 3) -> REST [3, natm]
    #[rustfmt::skip]
    let de_ref = vec![
         0.0065780504,  0.0111234158,  0.0057757713,
        -0.0007435066, -0.0011691955, -0.0095653436,
        -0.0100640547, -0.0008857724,  0.0018050477,
         0.0042295109, -0.0090684480,  0.0019845245,
    ];
    let de_ref = rt::asarray((de_ref, [3, 4]));
    println!("DH correlation gradient (REST): {:16.9}", de.t());
    println!("DH correlation gradient (ref) : {:16.9}", de_ref.t());
    println!("Maximum Error {:?}", (&de_ref - &de).abs().max_all());

    // parts are also cached for inspection
    for key in ["de_h", "de_s1", "de_jk", "de_rint"] {
        let part = grad.result.get(key).unwrap().clone();
        println!("part {key}: {:16.9}", rt::asarray((part.data, part.size)).t());
    }
    assert!((&de_ref - &de).abs().max_all() < 1.0e-5);

    // PT2 correlation energy is carried along as the entry energy
    assert!((grad.get_energy() - (-0.19179525779814563)).abs() < 2e-6, "PT2 correlation energy mismatch");
}

/// Doubly-hybrid (`xc = "xyg3"`) analytic gradient of distorted NH3 (restricted).
///
/// Reference: pyscf-forge `DFDH(xc="XYG3")` analytic gradient (`ref_analytic_rxyg3.json` of the
/// working notes), def2-SVP, def2-universal-jkfit for both the SCF JK fitting and the RI part,
/// grid level 3 (REST's default `grid_gen_level`, same point set as the pyscf reference), CPKS
/// tolerance 1e-8. The asserted total is the `"SCF"` (DF-B3LYP) entry plus the `"DH"` entry;
/// the residual against pyscf is at the same quadrature/RI noise level as pyscf's own
/// finite-difference check (6.8e-7).
#[test]
fn test_nh3_rxyg3_grad() {
    let keys = toml::from_str::<serde_json::Value>(&INPUT_XYG3[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);

    let eng = ri_pt2::xdh_calculations(&mut scf_data, &None).unwrap();
    println!("XYG3 total energy: {eng}");
    assert!((eng - (-56.445621659465374)).abs() < 2e-6, "XYG3 total energy mismatch");

    // regenerate the common SCF grids (xdh_calculations frees them for memory)
    scf_data.grids = Some(pyrest::dft::Grids::build(&mut scf_data.mol));

    use pyrest::grad::rhf::RIRHFGradient;
    let mut grad_scf = RIRHFGradient::new(&scf_data, &None);
    grad_scf.calc_rks();

    let mut grad = RDHGradient::new(&scf_data, &None);
    grad.calc();

    let de = rt::asarray((&grad.result["de"].data, grad.result["de"].size))
        + rt::asarray((&grad_scf.result["de"].data, grad_scf.result["de"].size));

    // pyscf-forge XYG3 total gradient (natom, 3) -> REST [3, natm]
    #[rustfmt::skip]
    let de_ref = vec![
         0.0169586567,  0.0023709868, -0.0760256914,
        -0.0014295009, -0.0040509894,  0.0822119912,
        -0.0013968304, -0.0076827463,  0.0000330828,
        -0.0141324624,  0.0093628946, -0.0062191910,
    ];
    let de_ref = rt::asarray((de_ref, [3, 4]));
    println!("XYG3 total gradient (REST): {:16.9}", de.t());
    println!("XYG3 total gradient (ref) : {:16.9}", de_ref.t());
    println!("Maximum Error {:?}", (&de_ref - &de).abs().max_all());

    // parts are also cached for inspection
    for key in ["de_h", "de_s1", "de_jk", "de_rint", "de_k_dh", "de_xc_n", "de_resp", "de_ovlp_dh"] {
        let part = grad.result.get(key).unwrap().clone();
        println!("part {key}: {:16.9}", rt::asarray((part.data, part.size)).t());
    }
    assert!((&de_ref - &de).abs().max_all() < 1.0e-6, "XYG3 total gradient mismatch");
}
