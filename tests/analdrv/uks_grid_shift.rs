//! UKS Hessian with the Becke grid-shift derivatives (`grid_shift_deriv`), the unrestricted
//! sibling of `rks_grid_shift`: magnitude of the correction (B3LYP) and translational
//! invariance of the assembled Hessian (B3LYP/TPSSh) and of the per-spin skeleton terms.

use pyrest::analdrv::config::AnalDrvConfig;
use pyrest::analdrv::uscf_interface::uscf_hess_interface;
use pyrest::ctrl_io;
use pyrest::molecule_io::Molecule;
use pyrest::scf_io::{self, scf_without_build};

use rstsr::prelude::*;

use libxc::prelude::*;

static INPUT_NH3_B3LYP: &str = r##"
[ctrl]
    print_level =          2
    num_threads =          16
    xc =                   "b3lyp"
    basis_path =           "def2-tzvp"
    auxbas_path =          "def2-universal-jkfit"
    eri_type =             "ri-v"
    charge =               2.0
    spin =                 3.0
    spin_polarization =    true
    auxbasis_response =    true
    mixer =                "diis"
    num_max_diis =         8
    start_diis_cycle =     3
    mix_param =            0.8
    max_scf_cycle =        100

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

static INPUT_NH3_TPSSH: &str = r##"
[ctrl]
    print_level =          2
    num_threads =          16
    xc =                   "TPSSh"
    basis_path =           "def2-tzvp"
    auxbas_path =          "def2-universal-jkfit"
    eri_type =             "ri-v"
    charge =               2.0
    spin =                 3.0
    spin_polarization =    true
    auxbasis_response =    true
    mixer =                "diis"
    num_max_diis =         8
    start_diis_cycle =     3
    mix_param =            0.8
    max_scf_cycle =        100
    xc_parser =            "parse_xc"

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

fn run_scf(input: &str) -> scf_io::SCF {
    let keys = toml::from_str::<serde_json::Value>(&input[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);
    scf_data
}

fn run_hessian(scf_data: &mut scf_io::SCF, config: &AnalDrvConfig) -> Tensor<f64, DeviceBLAS> {
    let (de, _, _) = uscf_hess_interface(scf_data, config);
    let natm = 4;
    rt::asarray((de, [3, 3, natm, natm], &DeviceBLAS::default()))
}

fn xc_func_list(scf_data: &scf_io::SCF) -> Vec<(f64, LibXCFunctional)> {
    scf_data
        .mol
        .xc_data
        .dfa_compnt_scf
        .iter()
        .zip(scf_data.mol.xc_data.dfa_paramr_scf.iter())
        .map(|(&code, &param)| (param, LibXCFunctional::from_number(code as _, LibXCSpin::Polarized)))
        .collect()
}

#[test]
fn test_b3lyp_grid_shift_magnitude_and_invariance() {
    let mut scf_data = run_scf(INPUT_NH3_B3LYP);

    let de_on = run_hessian(&mut scf_data, &AnalDrvConfig::default());
    let de_off = run_hessian(&mut scf_data, &AnalDrvConfig { grid_shift_deriv: false, ..Default::default() });

    // the grid-shift correction is small compared to the Hessian itself
    let max_diff = (&de_on - &de_off).abs().max();
    println!("b3lyp: max |de_on - de_off| = {:?}", max_diff);
    assert!(max_diff < 1.0e-2, "grid-shift correction too large: {max_diff:?}");

    // [t, s, A, B] -> [t, s]: the full Hessian is translationally invariant
    let max_sum = de_on.sum_axes([-1, -2]).abs().max();
    println!("b3lyp: max |sum_AB de_on| = {:?}", max_sum);
    assert!(max_sum < 1.0e-6, "full hessian atom-sum not invariant: {max_sum:?}");
}

#[test]
fn test_tpssh_invariance() {
    // MGGA: with the grid-shift on (default) the skeleton grid stays at the SCF level;
    // grid_gen_level + 2 is only the default when grid_shift_deriv = false.
    let mut scf_data = run_scf(INPUT_NH3_TPSSH);
    let de_on = run_hessian(&mut scf_data, &AnalDrvConfig::default());

    let max_sum = de_on.sum_axes([-1, -2]).abs().max();
    println!("tpssh: max |sum_AB de_on| = {:?}", max_sum);
    assert!(max_sum < 1.0e-6, "full hessian atom-sum not invariant: {max_sum:?}");
}

/// Skeleton-level invariance of the UKS XC terms themselves: `de_xc_skeleton` summed over the
/// atom pair, and each spin's `vmat_deriv1_grid_a/b` summed over its atom axis, must vanish at
/// a much tighter threshold than the assembled Hessian (see `rks_grid_shift`).
#[test]
fn test_skeleton_invariance() {
    use pyrest::dft::numint_matmul::hess_uks::UHessKSNIMatmul;
    use pyrest::dft::numint_matmul::nimatmul::{regroup_grids_by_atom, NIMatmul};
    use pyrest::ri_jk::util::get_cint_mol;

    for input in [INPUT_NH3_B3LYP, INPUT_NH3_TPSSH] {
        let scf_data = run_scf(input);
        let device = DeviceBLAS::default();
        let mo_coeff = [
            rt::asarray((&scf_data.eigenvectors[0].data, scf_data.eigenvectors[0].size, &device)).into_contig(ColMajor),
            rt::asarray((&scf_data.eigenvectors[1].data, scf_data.eigenvectors[1].size, &device)).into_contig(ColMajor),
        ];
        let mo_occ = [
            rt::asarray((&scf_data.occupation[0], [scf_data.occupation[0].len()], &device)).into_contig(ColMajor),
            rt::asarray((&scf_data.occupation[1], [scf_data.occupation[1].len()], &device)).into_contig(ColMajor),
        ];

        let mol_obj = &scf_data.mol;
        let cint_mol = get_cint_mol(mol_obj);
        let grids = scf_data.grids.as_ref().unwrap();
        let (coordinates, weights, atm_idx, quadrature_weights) = regroup_grids_by_atom(
            grids.coordinates.clone(),
            grids.weights.clone(),
            grids.atm_idx.clone(),
            grids.quadrature_weights.clone(),
            cint_mol.natm(),
        );
        let ni = NIMatmul::new(&cint_mol, &coordinates, &weights, &atm_idx, &quadrature_weights);

        let xc_func_list = xc_func_list(&scf_data);

        let mut hess_obj = UHessKSNIMatmul::new(&cint_mol, xc_func_list, ni, true, false);
        hess_obj.make_hessian_setup(
            &[mo_coeff[0].view(), mo_coeff[1].view()],
            &[mo_occ[0].view(), mo_occ[1].view()],
            None,
        );

        let de_sk = hess_obj.intmd["de_xc_skeleton"].view();
        let max_sum_sk = de_sk.sum_axes([-1, -2]).abs().max();
        println!("skeleton: max |sum_AB de_xc_skeleton| = {:?}", max_sum_sk);
        assert!(max_sum_sk < 1.0e-9, "skeleton atom-sum not invariant: {max_sum_sk:?}");

        for key in ["vmat_deriv1_grid_a", "vmat_deriv1_grid_b"] {
            let vmat = hess_obj.intmd[key].view();
            let max_sum_v = vmat.sum_axes(-1).abs().max();
            println!("skeleton: max |sum_A {key}| = {:?}", max_sum_v);
            assert!(max_sum_v < 1.0e-9, "{key} atom-sum not invariant: {max_sum_v:?}");
        }
    }
}
