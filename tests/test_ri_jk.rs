use pyrest::ctrl_io;
use pyrest::molecule_io::Molecule;
use pyrest::ri_jk::*;
use pyrest::scf_io::{self, SCF};
use pyrest::utilities::rstsr_util::*;
use rest_libcint::prelude::*;
use rstsr::prelude::*;

#[test]
fn test_nh3_j() {
    let scf_data = initialize_nh3();
    let device = DeviceBLAS::default();

    let dm = [&scf_data.density_matrix[0]].as_ref().to_rstsr(&device);
    let mol_obj = &scf_data.mol;

    let mol = &util::get_cint_mol(mol_obj);
    let aux = &util::get_cint_aux(mol_obj);

    // incore
    let rimatr = scf_data.rimatr.as_ref().unwrap().0.to_rstsr_view(&device);
    let j_rstsr = get_vj_ri_incore(rimatr, dm.view());
    let fp = fingerprint_f64(j_rstsr.i((.., .., 0)));
    let ref_fp = 37.83424292927407;
    assert!((fp / ref_fp - 1.0).abs() < 1e-5);

    // direct, full batch
    let j_rstsr = get_vj_ri_direct(dm.view(), mol, aux, 10000);
    let fp = fingerprint_f64(j_rstsr.i((.., .., 0)));
    let ref_fp = 37.83424292927407;
    assert!((fp / ref_fp - 1.0).abs() < 1e-5);

    // direct, small batch
    let j_rstsr = get_vj_ri_direct(dm.view(), mol, aux, 16);
    let fp = fingerprint_f64(j_rstsr.i((.., .., 0)));
    let ref_fp = 37.83424292927407;
    assert!((fp / ref_fp - 1.0).abs() < 1e-5);
}

#[test]
fn test_nh3_k() {
    let scf_data = initialize_nh3();
    let device = DeviceBLAS::default();

    let mo_coeff = [&scf_data.eigenvectors[0]].as_ref().to_rstsr(&device);
    let mo_occ = (&scf_data.occupation[0]).to_rstsr(&device).into_slice((.., None));
    let dms = [&scf_data.density_matrix[0]].as_ref().to_rstsr(&device);
    let mol_obj = &scf_data.mol;

    let mol = &util::get_cint_mol(mol_obj);
    let aux = &util::get_cint_aux(mol_obj);

    // incore, full batch, coeff
    let rimatr = scf_data.rimatr.as_ref().unwrap().0.to_rstsr_view(&device);
    let k_rstsr = get_vk_ri_incore_coeff(rimatr.view(), mo_coeff.view(), mo_occ.view(), 10000);
    let fp = fingerprint_f64(k_rstsr.i((.., .., 0)));
    let ref_fp = 12.950224351107128;
    assert!((fp / ref_fp - 1.0).abs() < 1e-5);

    // incore, small batch, coeff
    let k_rstsr = get_vk_ri_incore_coeff(rimatr.view(), mo_coeff.view(), mo_occ.view(), 16);
    let fp = fingerprint_f64(k_rstsr.i((.., .., 0)));
    let ref_fp = 12.950224351107128;
    assert!((fp / ref_fp - 1.0).abs() < 1e-5);

    // incore, full batch, dm
    let k_rstsr = get_vk_ri_incore_dm(rimatr.view(), dms.view(), 10000);
    let fp = fingerprint_f64(k_rstsr.i((.., .., 0)));
    let ref_fp = 12.950224351107128;
    assert!((fp / ref_fp - 1.0).abs() < 1e-5);

    // incore, small batch, dm
    let k_rstsr = get_vk_ri_incore_dm(rimatr.view(), dms.view(), 16);
    let fp = fingerprint_f64(k_rstsr.i((.., .., 0)));
    let ref_fp = 12.950224351107128;
    assert!((fp / ref_fp - 1.0).abs() < 1e-5);

    // semi-direct, full batch
    let k_rstsr = get_vk_ri_semi_direct_coeff(mo_coeff.view(), mo_occ.view(), mol, aux, 10000);
    let fp = fingerprint_f64(k_rstsr.i((.., .., 0)));
    let ref_fp = 12.950224351107128;
    assert!((fp / ref_fp - 1.0).abs() < 1e-5);

    // semi-direct, small batch
    let k_rstsr = get_vk_ri_semi_direct_coeff(mo_coeff.view(), mo_occ.view(), mol, aux, 16);
    let fp = fingerprint_f64(k_rstsr.i((.., .., 0)));
    let ref_fp = 12.950224351107128;
    assert!((fp / ref_fp - 1.0).abs() < 1e-5);

    // direct, full batch, dm version
    let k_rstsr = get_vk_ri_direct_dm(dms.view(), mol, aux, 10000);
    let fp = fingerprint_f64(k_rstsr.i((.., .., 0)));
    let ref_fp = 12.950224351107128;
    assert!((fp / ref_fp - 1.0).abs() < 1e-5);

    // direct, small batch, dm version
    let k_rstsr = get_vk_ri_direct_dm(dms.view(), mol, aux, 16);
    let fp = fingerprint_f64(k_rstsr.i((.., .., 0)));
    let ref_fp = 12.950224351107128;
    assert!((fp / ref_fp - 1.0).abs() < 1e-5);
}

#[test]
fn test_solved_j3c() {
    let scf_data = initialize_nh3();
    let mol_obj = &scf_data.mol;
    let device = DeviceBLAS::default();

    let mol = util::get_cint_mol(mol_obj);
    let aux = util::get_cint_aux(mol_obj);

    let j3c = {
        let (out, shape) = CInt::integrate_cross("int3c2e", [&mol, &mol, &aux], "s2ij", None).into();
        rt::asarray((out, shape.f(), &device))
    };
    let j4c = {
        let (out, shape) = mol.integrate("int2e", "s4", None).into();
        rt::asarray((out, shape.f(), &device))
    };

    // cholesky way, upper
    let j3c_ = j3c.clone();
    let ptr_j3c = j3c_.as_ptr();
    let j2c_decomp_option = J2CDecompOption { policy: J2CDecompPolicy::Cd, threshold: None, uplo: Upper };
    let j2c_decomp = get_j2c_decomp(&aux, &device, j2c_decomp_option);
    let j3c_solved = get_solved_j3c(j3c_, &j2c_decomp, false);
    let ptr_j3c_solved = j3c_solved.as_ptr();
    assert!(core::ptr::eq(ptr_j3c, ptr_j3c_solved));
    let j4c_recon = j3c_solved.view() % j3c_solved.t();
    // for this specific case, rtol=1e-2, atol=3e-2 should work
    assert!(rt::allclose(j4c_recon.view(), j4c.view(), (1e-2, 3e-2)));

    // cholesky way, lower
    let j3c_ = j3c.clone();
    let ptr_j3c = j3c_.as_ptr();
    let j2c_decomp_option = J2CDecompOption { policy: J2CDecompPolicy::Cd, threshold: None, uplo: Upper };
    let j2c_decomp = get_j2c_decomp(&aux, &device, j2c_decomp_option);
    let j3c_solved = get_solved_j3c(j3c_, &j2c_decomp, false);
    let ptr_j3c_solved = j3c_solved.as_ptr();
    assert!(core::ptr::eq(ptr_j3c, ptr_j3c_solved));
    let j4c_recon = j3c_solved.view() % j3c_solved.t();
    // for this specific case, rtol=1e-2, atol=3e-2 should work
    assert!(rt::allclose(j4c_recon.view(), j4c.view(), (1e-2, 3e-2)));

    // eigen way
    let j3c_ = j3c.clone();
    let ptr_j3c = j3c_.as_ptr();
    let j2c_decomp = get_j2c_decomp(
        &aux,
        &device,
        J2CDecompOption { policy: J2CDecompPolicy::Eig, threshold: Some(1e-13), uplo: Upper },
    );
    let j3c_solved = get_solved_j3c(j3c_, &j2c_decomp, false);
    let ptr_j3c_solved = j3c_solved.as_ptr();
    assert!(core::ptr::eq(ptr_j3c, ptr_j3c_solved));
    let j4c_recon = j3c_solved.view() % j3c_solved.t();
    // for this specific case, rtol=1e-2, atol=3e-2 should work
    assert!(rt::allclose(j4c_recon.view(), j4c.view(), (1e-2, 3e-2)));

    // cholesky way, non f-contiguous j3c should still work
    // (in this case, for 2-dim j3c, c-contiguous will still not perform copy)
    let j3c_ = j3c.to_contig(RowMajor).to_owned();
    let j2c_decomp_option = J2CDecompOption { policy: J2CDecompPolicy::Cd, threshold: None, uplo: Upper };
    let j2c_decomp = get_j2c_decomp(&aux, &device, j2c_decomp_option);
    let j3c_solved = get_solved_j3c(j3c_, &j2c_decomp, false);
    let j4c_recon = j3c_solved.view() % j3c_solved.t();
    // for this specific case, rtol=1e-2, atol=3e-2 should work
    assert!(rt::allclose(j4c_recon.view(), j4c.view(), (1e-2, 3e-2)));
}

#[test]
fn test_ao2mo() {
    let scf_data = initialize_nh3();
    let mol_obj = &scf_data.mol;
    let device = DeviceBLAS::default();

    let mol = util::get_cint_mol(mol_obj);
    let aux = util::get_cint_aux(mol_obj);

    let j3c = {
        let (out, shape) = CInt::integrate_cross("int3c2e", [&mol, &mol, &aux], "s2ij", None).into();
        rt::asarray((out, shape.f(), &device))
    };
    let nocc = (&scf_data.occupation[0].iter().sum::<f64>() / 2.0) as usize;
    println!("nocc: {nocc}");
    let mo_coeff = (&scf_data.eigenvectors[0]).to_rstsr(&device);
    let occ_coeff = mo_coeff.i((.., ..nocc));
    let vir_coeff = mo_coeff.i((.., nocc..));
    let out = get_ao2mo_s2ij_to_s1_trans(j3c.view(), Upper, &[occ_coeff.view()], &[vir_coeff.view()], None, |x| x);
    println!("=== aux, occ, vir ===\n{out:12.6?}");
    let out =
        get_ao2mo_s2ij_to_s1_trans(j3c.view(), Upper, &[vir_coeff.view()], &[occ_coeff.view()], None, |x| x as f32);
    println!("=== aux, vir, occ ===\n{out:12.6?}");
}

fn initialize_nh3() -> SCF {
    let input_token = r##"
[ctrl]
     print_level =          2
     xc =                   "hf"
     basis_path =           "basis-set-pool/def2-tzvp"
     auxbas_path =          "basis-set-pool/def2-universal-jkfit"
     eri_type =             "ri-v"
     charge =               0.0
     spin =                 1.0
     spin_polarization =    false
     initial_guess=         "sad"
     mixer =                "diis"
     num_threads =          16

[geom]
    name = "NH3"
    unit = "Angstrom"
    position = """
        N  0.0  0.0  0.0
        H  0.0  1.5  1.0
        H  1.4  1.1  0.0
        H  1.2  0.0  1.3
    """
"##;
    let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = SCF::build(mol, &None);
    scf_io::scf_without_build(&mut scf_data, &None);
    return scf_data;
}
