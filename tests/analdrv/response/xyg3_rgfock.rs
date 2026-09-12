//! XYG3 (xDH double hybrid) generalized Fock / relaxed dipole of NH3 (def2-TZVP), verified
//! contribution-by-contribution against pyscf-forge DFDH.
//!
//! Reference values were recorded by
//! `local-runs/260910-xyg3/xyg3_dipole_contribs.py` (pyscf-forge master @ 0566d43, pyscf 2.14.0):
//! XYG3 on B3LYP orbitals, basis def2-TZVP, auxiliary basis def2-universal-jkfit for both the JK
//! fitting and the RI-MP2 (3-center) part. The Lagrangian decomposition follows pyscf-forge's
//! `prepare_lagrangian`: `L_pt2 = W3 + W4 + Ax0(D_rdm1)` and
//! `L_xc_n = 4 Cv (h_core + J + V_xc_n) Co` (the SCF-functional part cancels through the canonical
//! diagonality of the converged B3LYP orbitals).
//!
//! Tolerances are looser than the RI-MP2 test: the SCF density, the XC numint fock and the DFT
//! response parts all carry grid differences between REST and pyscf integrations.

use pyrest::analdrv::config::AnalDrvConfig;
use pyrest::analdrv::response::rgfock_interface::rgfock_dh_interface;
use pyrest::analdrv::response::rresp_interface::rscf_resp_interface;
use pyrest::analdrv::response::trait_rgfock::{GFockFlags, RGFockAPI};
use pyrest::dft::Grids;
use pyrest::molecule_io::Molecule;
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
    xc =                   "xyg3"
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

    let device = rt::DeviceBLAS::default();

    // 0. XYG3 total energy (side product; also frees the DFT grids inside)
    let eng = ri_pt2::xdh_calculations(&mut scf_data, &None).unwrap();
    println!("XYG3 total energy: {}", eng);
    // reference: pyscf-forge DFDH (def2-universal-jkfit for both JK and RI)
    assert!((eng - (-56.513619197590)).abs() < 2e-6, "XYG3 total energy mismatch");

    // regenerate the common SCF grids (xdh_calculations frees them for memory)
    scf_data.grids = Some(Grids::build(&mut scf_data.mol));

    // 1. nuclear contribution
    let dip_nuc = scf_data.mol.geom.evaluate_dipole_moment(None).0;
    let dip_nuc = rt::asarray((dip_nuc.to_vec(), &device));
    println!("Dipole from nuclear contribution: {:16.12}", dip_nuc);
    let dip_nuc_ref = rt::asarray((vec![1.700753512109, 1.889726124565, 2.078698737022], &device));
    assert!(rt::allclose(&dip_nuc, &dip_nuc_ref, (1e-8, 1e-9)));

    // 2. scf density contribution
    let dm = scf_data.density_matrix[0].to_rstsr_view(&device);
    let mol_cint = get_cint_mol(&scf_data.mol);
    let int1e_r = {
        let (out, shape) = mol_cint.integrate("int1e_r", None, None).into();
        rt::asarray((out, shape, &device))
    };
    let dip_dm_scf = -(&int1e_r * &dm).sum_axes([0, 1]);
    println!("Dipole from SCF density matrix: {:16.12}", dip_dm_scf);
    let dip_dm_scf_ref = rt::asarray((vec![-1.156948912236, -1.364097621601, -1.579483131449], &device));
    assert!(rt::allclose(&dip_dm_scf, &dip_dm_scf_ref, (1e-5, 1e-6)));

    let mo_coeff = (&scf_data.eigenvectors[0]).to_rstsr(&device);

    // 3. generalized Fock / Lagrangian decomposition: standalone PT2 contribution first, then the
    // DH composite (which internally rebuilds the same PT2 element inside its contribution list)
    let config = AnalDrvConfig::default();
    let mut resp_objs = rscf_resp_interface(&scf_data, &config);

    // PT2 (correlation) contribution: unrelaxed rdm1 and Lagrangian
    let [c_os, c_ss]: [f64; 2] = scf_data
        .mol
        .xc_data
        .dfa_paramr_adv
        .clone()
        .expect("xyg3 carries the PT2 spin factors.")
        .try_into()
        .expect("dfa_paramr_adv must have exactly two entries.");
    let mo_occ = (&scf_data.occupation[0]).to_rstsr(&device);
    let mo_energy = (&scf_data.eigenvalues[0]).to_rstsr(&device);
    let j3c = scf_data.rimatr.as_ref().unwrap().0.to_rstsr_view(&device);
    let nocc = mo_occ.view().greater(0).sum();
    let mut pt2 = RGFockPT2::<f64>::new(
        mo_coeff.to_owned(),
        mo_occ,
        mo_energy,
        j3c.into_cow(),
        None,
        vec![0, nocc],
        c_os,
        c_ss,
    );
    resp_objs.make_cpscf_preparation(mo_coeff.view(), pt2.mo_occ.view(), pt2.mo_energy.view());
    let rdm1_corr = pt2.make_rdm1();
    let rdm1_corr_ao = mo_coeff.view() % rdm1_corr.view() % mo_coeff.view().t();
    let dip_rdm1_corr = -(&int1e_r * &rdm1_corr_ao).sum_axes([0, 1]);
    println!("Dipole from PT2 density matrix: {:16.12}", dip_rdm1_corr);
    let dip_rdm1_corr_ref = rt::asarray((vec![-0.002948597510, -0.004975193666, -0.007063388846], &device));
    assert!(rt::allclose(&dip_rdm1_corr, &dip_rdm1_corr_ref, (1e-6, 5e-7)));

    let lag_pt2 = pt2.make_lagrangian_vo(&mut resp_objs);
    let lag_pt2_fro = (&lag_pt2 * &lag_pt2).sum().sqrt();
    println!("Lagrangian PT2 part (fro): {lag_pt2_fro} (ref 0.08654174685311)");
    assert!((lag_pt2_fro - 0.08654174685311).abs() < 1e-6, "PT2 Lagrangian fro mismatch");

    let mut rgfock = rgfock_dh_interface(&scf_data);
    rgfock.make_response_preparation(&mut resp_objs);

    // the composite's summed unrelaxed rdm1 (only the PT2 element contributes) agrees with the
    // standalone PT2 evaluation
    assert!(rt::allclose(&rgfock.make_rdm1(), &rdm1_corr, None));

    let lag_total = rgfock.make_lagrangian(Some(&mut resp_objs));
    let lag_total_fro = (&lag_total * &lag_total).sum().sqrt();
    println!("Lagrangian total (fro): {lag_total_fro} (ref 0.33338836293131)");
    assert!((lag_total_fro - 0.33338836293131).abs() < 1e-5, "total Lagrangian fro mismatch");

    let lag_xc_n = &lag_total - &lag_pt2;
    let lag_xc_n_fro = (&lag_xc_n * &lag_xc_n).sum().sqrt();
    println!("Lagrangian xc_n part (fro): {lag_xc_n_fro} (ref 0.40098670506446)");
    assert!((lag_xc_n_fro - 0.40098670506446).abs() < 2e-5, "xc_n Lagrangian fro mismatch");

    // generalized Fock of the DH composite, restricted to the OV/VO parts (the PT2 element
    // rejects OO/VV, which are not implemented)
    let gfock = rgfock.make_gfock(Some(&mut resp_objs), GFockFlags::OV | GFockFlags::VO);
    let nocc = rgfock.nocc();
    let nmo = rgfock.nmo();
    let so = rt::slice!(0, nocc);
    let sv = rt::slice!(nocc, nmo);
    println!("gfock (VO block): {:16.12}", gfock.i((sv, so)));

    // 4. response (Z-vector / CP-SCF) contribution through the relaxed density
    let rdm1_resp = rgfock.make_rdm1_resp(&mut resp_objs);
    let dz_mo = &rdm1_resp - &rdm1_corr;
    let dz_ao = mo_coeff.view() % dz_mo.view() % mo_coeff.view().t();
    let dip_resp = -(&int1e_r * &dz_ao).sum_axes([0, 1]);
    println!("Dipole from response density: {:16.12}", dip_resp);
    let dip_resp_ref = rt::asarray((vec![0.009523402231, 0.014507684671, 0.020086193759], &device));
    // tuple order is (rtol, atol); the deviation carries the REST/pyscf grid difference
    assert!(rt::allclose(&dip_resp, &dip_resp_ref, (1e-5, 1e-5)));

    // 5. total dipole = sum of all contributions
    let dip_total = &dip_nuc + &dip_dm_scf + &dip_rdm1_corr + &dip_resp;
    println!("Total dipole: {:16.12}", dip_total);
    let dip_total_ref = rt::asarray((vec![0.550379404594, 0.535160993970, 0.512238410486], &device));
    assert!(rt::allclose(&dip_total, &dip_total_ref, (1e-5, 1e-5)));

    for (label, t) in &rgfock.timing {
        println!("Timing {label}: {t:.3} s");
    }
}
