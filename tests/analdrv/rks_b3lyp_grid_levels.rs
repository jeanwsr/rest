//! RKS (B3LYP, GGA) Hessian with explicit `grid_level_cphf` / `grid_level_skeleton`.
//!
//! Exercises the dedicated CP-KS grid path (`ni_cpks = Some`) and the skeleton-grid
//! regeneration path for a GGA functional.

use pyrest::analdrv::config::AnalDrvConfig;
use pyrest::analdrv::rscf_interface::rscf_hess_interface;

use pyrest::ctrl_io;
use pyrest::dft::numint_matmul::hess_rks::{get_hess_ncomp_ao_dm0, get_rho_vxc_fxc, make_cpks_vxc_fxc};
use pyrest::dft::numint_matmul::nimatmul::NIMatmul;
use pyrest::dft::xceff::prelude::determine_den_type_from_list;
use pyrest::molecule_io::Molecule;
use pyrest::scf_io::{self, scf_without_build};

use rstsr::prelude::*;

static INPUT_NH3: &str = r##"
[ctrl]
    print_level =          2
    num_threads =          16
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
    name = "NH3"
    unit = "Angstrom"
    position = """
        N  0.0  0.0  0.0
        H  1.0  0.1  0.2
        H  0.3  1.1  0.2
        H  0.1  0.1  1.2
    """
"##;

fn run_with_config(config: AnalDrvConfig) -> Vec<f64> {
    let keys = toml::from_str::<serde_json::Value>(&INPUT_NH3[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);
    let (de, _, _) = rscf_hess_interface(&mut scf_data, &config);
    de
}

#[test]
fn test_nh3_explicit_grid_levels() {
    // For GGA the default skeleton level equals grid_gen_level (3); force both grids to differ
    // from each other and from the SCF grid so the regeneration + ni_cpks paths are exercised.
    let config = AnalDrvConfig { grid_level_skeleton: Some(4), grid_level_cphf: Some(2), ..Default::default() };
    let de = run_with_config(config);

    let natm = 4;
    let de = rt::asarray((&de, [3, 3, natm, natm]));
    println!("Hessian (explicit grid levels):\n{:12.6}", de.t());
    assert_eq!(de.shape().to_vec(), vec![3, 3, natm, natm]);
}

#[test]
fn test_nh3_default_grid_levels() {
    // Default config: skeleton = grid_gen_level (3), cphf = grid_gen_level.max(3) - 2 = 1.
    // This activates the ni_cpks path with a coarse cphf grid.
    let config = AnalDrvConfig::default();
    let de = run_with_config(config);

    let natm = 4;
    let de = rt::asarray((&de, [3, 3, natm, natm]));
    println!("Hessian (default grid levels):\n{:12.6}", de.t());
    assert_eq!(de.shape().to_vec(), vec![3, 3, natm, natm]);
}

#[test]
fn test_nh3_cphf_equals_skeleton() {
    // cphf level == skeleton level (both 3) -> ni_cpks = None fast path (reuse skeleton vxc/fxc).
    let config = AnalDrvConfig { grid_level_skeleton: Some(3), grid_level_cphf: Some(3), ..Default::default() };
    let de = run_with_config(config);

    let natm = 4;
    let de = rt::asarray((&de, [3, 3, natm, natm]));
    println!("Hessian (cphf == skeleton):\n{:12.6}", de.t());
    assert_eq!(de.shape().to_vec(), vec![3, 3, natm, natm]);
}

/// Numerical-equivalence check: the lean braket-based [`make_cpks_vxc_fxc`] (which forms rho from
/// occupied MOs via `make_rho_from_homogeneous_braket`) must produce the same `vxc` / `fxc` as the
/// dm0-based [`get_rho_vxc_fxc`] on the same grid.
#[test]
fn test_cpks_vxc_fxc_matches_dm0_path() {
    use itertools::Itertools;
    use libxc::prelude::*;
    use pyrest::ri_jk::util::get_dm0_restricted;

    let keys = toml::from_str::<serde_json::Value>(&INPUT_NH3[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);

    let device = DeviceBLAS::default();
    let mo_coeff =
        rt::asarray((&scf_data.eigenvectors[0].data, scf_data.eigenvectors[0].size, &device)).into_contig(ColMajor);
    let mo_occ = rt::asarray((&scf_data.occupation[0], [scf_data.occupation[0].len()], &device)).into_contig(ColMajor);

    let mol_obj = &scf_data.mol;
    let cint_mol = pyrest::ri_jk::util::get_cint_mol(mol_obj);
    let grid_coords = &scf_data.grids.as_ref().unwrap().coordinates;
    let grid_weights = &scf_data.grids.as_ref().unwrap().weights;
    let mut ni = NIMatmul::new(&cint_mol, grid_coords, grid_weights);

    let xc_func_list: Vec<(f64, LibXCFunctional)> = scf_data
        .mol
        .xc_data
        .dfa_compnt_scf
        .iter()
        .zip(scf_data.mol.xc_data.dfa_paramr_scf.iter())
        .map(|(&code, &param)| (param, LibXCFunctional::from_number(code as _, LibXCSpin::Unpolarized)))
        .collect();

    // reference: dm0 -> ao_dm0 -> get_rho_vxc_fxc
    let xc_type = determine_den_type_from_list(&xc_func_list.iter().map(|(_, f)| f).collect_vec());
    let ncomp_ao_dm0 = get_hess_ncomp_ao_dm0(xc_type);
    let dm0 = get_dm0_restricted(mo_coeff.view(), mo_occ.view());
    let ao = ni.get_cached_ao(xc_type.num_ao_deriv());
    let ao_dm0 = ao.i((Ellipsis, ..ncomp_ao_dm0)) % &dm0;
    let (_rho_ref, vxc_ref, fxc_ref) = get_rho_vxc_fxc(&xc_func_list, ao.view(), ao_dm0.view());

    // lean: braket from mo_coeff/mo_occ (clear AO cache so the braket path re-evaluates freshly)
    ni.cache_tensor.clear();
    let (vxc, fxc) = make_cpks_vxc_fxc(&xc_func_list, &mut ni, mo_coeff.view(), mo_occ.view());

    assert!(rt::allclose(&vxc, &vxc_ref, None));
    assert!(rt::allclose(&fxc, &fxc_ref, None));
}
