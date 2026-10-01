//! Doubly-hybrid (`xc = "mp2"`) analytic gradient of distorted NH3+ (unrestricted RI-MP2).
//!
//! Reference: pyscf-forge `UDFDH(xc="MP2")` analytic gradient (pyscf 2.14.0 + pyscf-forge),
//! def2-SVP, def2-universal-jkfit for both the SCF JK fitting and the RI (3-center) part,
//! CPKS tolerance 1e-8. The JSON reference was dumped per atom in Hartree/bohr (row-major
//! `[natom, 3]`); REST stores the gradient as `[3, natm]`.
//! The correlation part below is the `"DH"` entry of `eval_force` (the `"SCF"` entry covers
//! the UHF part). The rimatr must be CD-decomposed (see `src/grad/udh.rs`).

use pyrest::ctrl_io;
use pyrest::grad::traits::GradAPI;
use pyrest::grad::udh::UDHGradient;
use pyrest::molecule_io::Molecule;
use pyrest::ri_pt2;
use pyrest::scf_io::{self, scf_without_build};
use rstsr::prelude::*;

static INPUT_NH3: &str = r##"
[ctrl]
    print_level =          0
    num_threads =          4
    xc =                   "mp2"
    basis_path =           "def2-SVP"
    auxbas_path =          "def2-universal-jkfit"
    eri_type =             "ri-v"
    charge =               1.0
    spin =                 2.0
    spin_polarization =    true

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
fn test_nh3_ump2_grad() {
    let keys = toml::from_str::<serde_json::Value>(&INPUT_NH3[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);

    // total energy (RI-UMP2); also puts the SCF into the post-SCF state the gradient expects
    let eng = ri_pt2::xdh_calculations(&mut scf_data, &None).unwrap();
    println!("RI-UMP2 total energy: {eng}");
    assert!((eng - (-55.95341829313824)).abs() < 2e-6, "RI-UMP2 total energy mismatch");

    let mut grad = UDHGradient::new(&scf_data, &None);
    grad.calc();
    let de = grad.result.get("de").unwrap().clone();
    let de = rt::asarray((de.data, de.size));

    // pyscf-forge `gradient_corr_only_hartree_per_bohr` (natom, 3) -> REST [3, natm]
    #[rustfmt::skip]
    let de_ref = vec![
         0.0069061512,  0.0118056890,  0.0063273326,
        -0.0009358131, -0.0015731210, -0.0092567167,
        -0.0095325681, -0.0013691064,  0.0013982791,
         0.0035622300, -0.0088634616,  0.0015311050,
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
    assert!((grad.get_energy() - (-0.1488233019937465)).abs() < 2e-6, "PT2 correlation energy mismatch");

    // total gradient: the "SCF" (UHF) entry plus the "DH" entry, against pyscf-forge's total
    use pyrest::grad::uhf::RIUHFGradient;
    let mut grad_scf = RIUHFGradient::new(&scf_data, &None);
    grad_scf.calc();
    let de_tot = rt::asarray((grad.result["de"].data.clone(), grad.result["de"].size))
        + rt::asarray((grad_scf.result["de"].data.clone(), grad_scf.result["de"].size));

    // pyscf-forge `gradient_hartree_per_bohr` (natom, 3) -> REST [3, natm]
    #[rustfmt::skip]
    let de_tot_ref = vec![
        -0.0109435544, -0.0477856558, -0.1027269995,
         0.0114885212,  0.0186658009,  0.0664058305,
        -0.0155263679,  0.0185574053,  0.0212383810,
         0.0149814011,  0.0105624497,  0.0150827880,
    ];
    let de_tot_ref = rt::asarray((de_tot_ref, [3, 4]));
    println!("UMP2 total gradient (REST): {:16.9}", de_tot.t());
    println!("UMP2 total gradient (ref) : {:16.9}", de_tot_ref.t());
    println!("Maximum Error {:?}", (&de_tot_ref - &de_tot).abs().max_all());
    assert!((&de_tot_ref - &de_tot).abs().max_all() < 1.0e-5, "UMP2 total gradient mismatch");
}
