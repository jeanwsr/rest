use pyrest::analdrv::config::AnalDrvConfig;
use pyrest::analdrv::response::rgfock_interface::solve_z_vector;
use pyrest::analdrv::response::rresp_interface::rscf_resp_interface;
use pyrest::analdrv::response::trait_rgfock::{GFockFlags, RGFockAPI};
use pyrest::molecule_io::Molecule;
use pyrest::ri_jk::get_ao2mo_s2ij_to_s1_notrans;
use pyrest::ri_jk::util::get_cint_mol;
use pyrest::ri_pt2::rgfock_pt2::RGFockPT2;
use pyrest::scf_io::{self, scf_without_build};
use pyrest::utilities::rstsr_util::{RestTensorToRstsrTsrAPI, RestTensorToRstsrViewAPI};
use pyrest::{ctrl_io, ri_pt2};
use rstsr::prelude::*;

static INPUT_NH3: &str = r##"
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
    start_diis_cycle =     3
    mix_param =            0.8
    max_scf_cycle =        100

[ctrl.ri_pt2]
    new_driver = true

[geom]
    name = "NH3"
    unit = "Angstrom"
    position = """
    N 0.0 0.0 0.0
    H 0.9 0.0 0.0
    H 0.0 1.0 0.0
    H 0.0 0.0 1.1
    """
"##;

#[test]
fn test_nh3() {
    let keys = toml::from_str::<serde_json::Value>(&INPUT_NH3[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);
    let eng = ri_pt2::xdh_calculations(&mut scf_data, &None).unwrap();
    println!("MP2 energy: {}", eng);

    let device = rt::DeviceBLAS::default();

    // 1. nuclear contribution
    let dip_nuc = scf_data.mol.geom.evaluate_dipole_moment(None).0;
    let dip_nuc = rt::asarray((dip_nuc.to_vec(), &device));
    println!("Dipole from nuclear contribution: {:16.12}", dip_nuc);
    let dip_nuc_ref = rt::asarray((vec![1.700753512109, 1.889726124565, 2.078698737022], &device));
    assert!(rt::allclose(&dip_nuc, &dip_nuc_ref, None));

    // 2. scf density contribution
    let dm = scf_data.density_matrix[0].to_rstsr_view(&device);
    let mol = get_cint_mol(&scf_data.mol);
    let int1e_r = {
        let (out, shape) = mol.integrate("int1e_r", None, None).into();
        rt::asarray((out, shape, &device))
    };
    let dip_dm_scf = -(&int1e_r * &dm).sum_axes([0, 1]);
    println!("Dipole from SCF density matrix: {:16.12}", dip_dm_scf);
    let dip_dm_scf_ref = rt::asarray((vec![-1.138682776261, -1.343280625287, -1.559934460339], &device));
    assert!(rt::allclose(&dip_dm_scf, &dip_dm_scf_ref, None));

    // after 2. pt2 preparation
    let j3c = scf_data.rimatr.as_ref().unwrap().0.to_rstsr_view(&device);
    println!("j3c shape: {:?}", j3c.shape());
    let mo_energy = (&scf_data.eigenvalues[0]).to_rstsr(&device);
    let mo_coeff = (&scf_data.eigenvectors[0]).to_rstsr(&device);

    let idx_core = scf_data.mol.start_mo;
    let idx_lumo = scf_data.lumo[0];
    let num_mo = mo_energy.size();
    let occ_list: Vec<usize> = (idx_core..idx_lumo).collect();
    let vir_list: Vec<usize> = (idx_lumo..num_mo).collect();

    let occ_energy = mo_energy.index_select(-1, &occ_list).into_contig(ColMajor);
    let vir_energy = mo_energy.index_select(-1, &vir_list).into_contig(ColMajor);
    let occ_coeff = mo_coeff.index_select(-1, &occ_list).into_contig(ColMajor);
    let vir_coeff = mo_coeff.index_select(-1, &vir_list).into_contig(ColMajor);
    let cderi_vox = {
        let mut cderi_vox_lst =
            get_ao2mo_s2ij_to_s1_notrans(j3c.view(), Upper, &[vir_coeff.view()], &[occ_coeff.view()], |x| x);
        cderi_vox_lst.remove(0)
    };
    println!("cderi_vox shape: {:?}", cderi_vox.shape());

    use pyrest::ri_pt2::pure_pt2_r_elecderiv::*;
    let input = RPT2ElecDerivIncoreInp {
        cderi: j3c.view(),
        // cderi_vox: Some(cderi_vox.view()),
        cderi_vox: None,
        occ_coeff: occ_coeff.view(),
        vir_coeff: vir_coeff.view(),
        occ_energy: occ_energy.view(),
        vir_energy: vir_energy.view(),
        index_occ_outer_vec: &[0, 2, 5],
    };
    let arg = RPT2ElecDerivIncoreArg { c_os: 1.0, c_ss: 1.0 };
    let output = get_rpt2_elec_deriv_incore(&input, &arg, |x| x);
    println!("MP2 correlation energy: {}", output.e_corr);
    let e_corr_ref = -0.245426806393;
    assert!((output.e_corr - e_corr_ref).abs() < 1e-6, "MP2 correlation energy mismatch");

    let rdm1_corr_ao = mo_coeff.view() % output.rdm1_corr.view() % mo_coeff.view().t();
    let dip_rdm1_corr = -(&int1e_r * &rdm1_corr_ao).sum_axes([0, 1]);
    println!("Dipole from PT2 density matrix: {:16.12}", dip_rdm1_corr);
    let dip_rdm1_corr_ref = rt::asarray((vec![-0.002351686599, -0.003862564554, -0.005077785238], &device));
    assert!(rt::allclose(&dip_rdm1_corr, &dip_rdm1_corr_ref, None));

    // 4. response (Z-vector / CP-SCF) contribution, through RGFockPT2
    // reference values (pyscf-forge DFDH): lag_vo fro = 0.18726125698236695,
    // Z_vo fro = 0.09429288418092462, dip_resp = [-0.009197291377 -0.004474412016 0.008052718901]
    let config = AnalDrvConfig::default();
    let mut resp_objs = rscf_resp_interface(&scf_data, &config);

    let nocc_full = occ_list.len();
    let mut mo_occ = rt::zeros(([num_mo].f(), &device));
    mo_occ.i_mut(idx_core..idx_lumo).fill(2.0);

    let mut rgfock = RGFockPT2::<f64>::new(
        mo_coeff.to_owned(),
        mo_occ,
        mo_energy.to_owned(),
        j3c.view().into_cow(),
        None,
        vec![0, 2, 5],
        1.0,
        1.0,
    );
    resp_objs.make_cpscf_preparation(mo_coeff.view(), rgfock.mo_occ.view(), mo_energy.view());

    let so_full = rt::slice!(0, nocc_full);
    let sv_full = rt::slice!(nocc_full, num_mo);

    // trait-level access: unrelaxed rdm1 and generalized Fock (OV/VO blocks filled)
    let rdm1_trait = rgfock.make_rdm1();
    assert!(rt::allclose(&rdm1_trait, &output.rdm1_corr, None));
    let gfock = rgfock.make_gfock(Some(&mut resp_objs), GFockFlags::OV | GFockFlags::VO);
    println!("gfock (VO block): {:16.12}", gfock.i((sv_full, so_full)));

    let lag_vo = rgfock.make_lagrangian_vo(&mut resp_objs);
    let lag_fro = (&lag_vo * &lag_vo).sum().sqrt();
    println!("Lagrangian (lag_vo) fro: {lag_fro} (ref 0.18726125698236695)");

    let z_vo = solve_z_vector(lag_vo.view(), &mut resp_objs);
    let z_fro = (&z_vo * &z_vo).sum().sqrt();
    println!("Z-vector (z_vo) fro: {z_fro} (ref 0.09429288418092462)");

    // relaxed density response: dipole contribution of the Z-vector part
    let rdm1_resp = rgfock.make_rdm1_resp(&mut resp_objs);
    let dz_mo = &rdm1_resp - &output.rdm1_corr;
    let dz_ao = mo_coeff.view() % dz_mo.view() % mo_coeff.view().t();
    let dip_resp = -(&int1e_r * &dz_ao).sum_axes([0, 1]);
    println!("Dipole from response density: {:16.12}", dip_resp);
    // the recorded reference carries an irreducible ~2e-7 discrepancy
    let dip_resp_ref = rt::asarray((vec![-0.009197291377, -0.004474412016, 0.008052718901], &device));
    assert!(rt::allclose(&dip_resp, &dip_resp_ref, (1e-5, 2e-6)));

    // total dipole = sum of all contributions
    let dip_total = &dip_nuc + &dip_dm_scf + &dip_rdm1_corr + &dip_resp;
    println!("Total dipole: {:16.12}", dip_total);
    let dip_total_ref = rt::asarray((vec![0.550521757872, 0.538108522708, 0.521739210345], &device));
    assert!(rt::allclose(&dip_total, &dip_total_ref, (1e-5, 2e-6)));
}

/// FP32 counterpart of [`test_nh3`]: the same electronic-derivative machinery with the
/// working type `O = f32` (the input integrals stay f64), both through the pure function
/// and through [`RGFockPT2<f32>`] (supplying an f32 `cderi_vox`, produced directly by the
/// ao2mo, so no f64 tensor has to be re-cast). Reference values are the recorded f64 numbers
/// of [`test_nh3`]; tolerances are relaxed to the fp32 level.
#[test]
fn test_nh3_fp32() {
    let keys = toml::from_str::<serde_json::Value>(&INPUT_NH3[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);
    let eng = ri_pt2::xdh_calculations(&mut scf_data, &None).unwrap();
    println!("MP2 energy (ctrl fp_mode default, new driver): {}", eng);

    let device = rt::DeviceBLAS::default();

    let mol = get_cint_mol(&scf_data.mol);
    let int1e_r = {
        let (out, shape) = mol.integrate("int1e_r", None, None).into();
        rt::asarray((out, shape, &device))
    };

    let j3c = scf_data.rimatr.as_ref().unwrap().0.to_rstsr_view(&device);
    let mo_energy = (&scf_data.eigenvalues[0]).to_rstsr(&device);
    let mo_coeff = (&scf_data.eigenvectors[0]).to_rstsr(&device);

    let idx_core = scf_data.mol.start_mo;
    let idx_lumo = scf_data.lumo[0];
    let num_mo = mo_energy.size();
    let occ_list: Vec<usize> = (idx_core..idx_lumo).collect();
    let vir_list: Vec<usize> = (idx_lumo..num_mo).collect();

    let occ_energy = mo_energy.index_select(-1, &occ_list).into_contig(ColMajor);
    let vir_energy = mo_energy.index_select(-1, &vir_list).into_contig(ColMajor);
    let occ_coeff = mo_coeff.index_select(-1, &occ_list).into_contig(ColMajor);
    let vir_coeff = mo_coeff.index_select(-1, &vir_list).into_contig(ColMajor);
    // f32 pre-transformed integrals, generated directly by the ao2mo (f64 contraction,
    // cast at the end); used for `RGFockPT2<f32>` below
    let cderi_vox = {
        let mut cderi_vox_lst =
            get_ao2mo_s2ij_to_s1_notrans(j3c.view(), Upper, &[vir_coeff.view()], &[occ_coeff.view()], |x| x as f32);
        cderi_vox_lst.remove(0)
    };

    // 1. pure function with the f32 working type
    use pyrest::ri_pt2::pure_pt2_r_elecderiv::*;
    let input = RPT2ElecDerivIncoreInp {
        cderi: j3c.view(),
        cderi_vox: None,
        occ_coeff: occ_coeff.view(),
        vir_coeff: vir_coeff.view(),
        occ_energy: occ_energy.view(),
        vir_energy: vir_energy.view(),
        index_occ_outer_vec: &[0, 2, 5],
    };
    let arg = RPT2ElecDerivIncoreArg { c_os: 1.0, c_ss: 1.0 };
    let output = get_rpt2_elec_deriv_incore(&input, &arg, |x| x as f32);
    let e_corr_ref = -0.245426806393;
    println!("MP2 correlation energy (f32): {}", output.e_corr);
    println!("  deviation from f64 reference: {}", output.e_corr - e_corr_ref);
    assert!((output.e_corr - e_corr_ref).abs() < 1e-7, "MP2 correlation energy mismatch (f32)");

    let rdm1_corr_ao = mo_coeff.view() % output.rdm1_corr.view() % mo_coeff.view().t();
    let dip_rdm1_corr = -(&int1e_r * &rdm1_corr_ao).sum_axes([0, 1]);
    println!("Dipole from PT2 density matrix (f32): {:16.12}", dip_rdm1_corr);
    let dip_rdm1_corr_ref = rt::asarray((vec![-0.002351686599, -0.003862564554, -0.005077785238], &device));
    println!("  deviation from f64 reference: {}", &dip_rdm1_corr - &dip_rdm1_corr_ref);
    assert!(rt::allclose(&dip_rdm1_corr, &dip_rdm1_corr_ref, (1e-5, 1e-7)));

    // 2. RGFockPT2 in the f32 working type, with the f32 pre-transformed cderi_vox
    let config = AnalDrvConfig::default();
    let mut resp_objs = rscf_resp_interface(&scf_data, &config);

    let nocc_full = occ_list.len();
    let mut mo_occ = rt::zeros(([num_mo].f(), &device));
    mo_occ.i_mut(idx_core..idx_lumo).fill(2.0);

    let mut rgfock = RGFockPT2::new(
        mo_coeff.to_owned(),
        mo_occ,
        mo_energy.to_owned(),
        j3c.view().into_cow(),
        Some(cderi_vox.view().into_cow()),
        vec![0, 2, 5],
        1.0,
        1.0,
    );
    resp_objs.make_cpscf_preparation(mo_coeff.view(), rgfock.mo_occ.view(), mo_energy.view());

    let so_full = rt::slice!(0, nocc_full);
    let sv_full = rt::slice!(nocc_full, num_mo);

    // consistency with the direct f32 call (the supplied f32 `cderi_vox` and the in-kernel
    // regeneration both transform in f64 and cast at the end, so they agree closely)
    let rdm1_trait = rgfock.make_rdm1();
    assert!(rt::allclose(&rdm1_trait, &output.rdm1_corr, (1e-5, 1e-6)));
    let gfock = rgfock.make_gfock(Some(&mut resp_objs), GFockFlags::OV | GFockFlags::VO);
    println!("gfock (VO block, f32): {:16.12}", gfock.i((sv_full, so_full)));

    // references (pyscf-forge DFDH, f64): lag_vo fro = 0.18726125698236695,
    // Z_vo fro = 0.09429288418092462, dip_resp = [-0.009197291377 -0.004474412016 0.008052718901]
    let lag_vo = rgfock.make_lagrangian_vo(&mut resp_objs);
    let lag_fro = (&lag_vo * &lag_vo).sum().sqrt();
    println!("Lagrangian (lag_vo) fro (f32): {lag_fro} (ref 0.18726125698236695)");
    assert!((lag_fro - 0.18726125698236695).abs() < 1e-6);

    let z_vo = solve_z_vector(lag_vo.view(), &mut resp_objs);
    let z_fro = (&z_vo * &z_vo).sum().sqrt();
    println!("Z-vector (z_vo) fro (f32): {z_fro} (ref 0.09429288418092462)");
    assert!((z_fro - 0.09429288418092462).abs() < 1e-6);

    // relaxed density response: dipole contribution of the Z-vector part
    let rdm1_resp = rgfock.make_rdm1_resp(&mut resp_objs);
    let dz_mo = &rdm1_resp - &output.rdm1_corr;
    let dz_ao = mo_coeff.view() % dz_mo.view() % mo_coeff.view().t();
    let dip_resp = -(&int1e_r * &dz_ao).sum_axes([0, 1]);
    println!("Dipole from response density (f32): {:16.12}", dip_resp);
    let dip_resp_ref = rt::asarray((vec![-0.009197291377, -0.004474412016, 0.008052718901], &device));
    println!("  deviation from f64 reference: {}", &dip_resp - &dip_resp_ref);
    assert!(rt::allclose(&dip_resp, &dip_resp_ref, (1e-4, 1e-6)));
}

// --- following is utilities for developing dipole evaluation --- //
