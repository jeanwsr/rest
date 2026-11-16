//! Static-term consistency test for range-separated hybrid (RSH) GW references.
//!
//! The GW quasiparticle equation starts from the HF-like static diagonal
//! `e_n + (K_full - K_ref)_nn - v_xc,nn`.  For *any* reference (HF, global
//! hybrid or RSH) this must equal the diagonal of the full-HF Fock matrix in
//! the reference MO basis, because the reference KS eigenvalues contain
//! `K_ref,nn + v_xc,nn`.  This test checks that identity for several RSH
//! functionals, i.e. that the short-range exchange completion built by
//! `ri_gw::exchange_matrix_from_scf` is consistent with the SCF Hamiltonian.

use std::sync::Mutex;

use pyrest::ctrl_io;
use pyrest::molecule_io::Molecule;
use pyrest::ri_gw;
use pyrest::scf_io::{self, SCF};

/// The RSH SCF/libcint path is not safe to exercise from several threads at
/// once in one process; keep the two tests in this target serialised.
static RSH_GW_TEST_LOCK: Mutex<()> = Mutex::new(());

static CTRL_TEMPLATE: &str = r##"
[ctrl]
    print_level =          0
    xc =                   "XC_PLACEHOLDER"
    basis_path =           "basis-set-pool/sto-3g"
    auxbas_path =          "basis-set-pool/def2-universal-jkfit"
    eri_type =             "ri-v"
    charge =               0.0
    algorithm_jk =         "ri-incore"
    use_ri_symm =          true
    spin =                 1
    spin_polarization =    false
    initial_guess =        "sad"
    num_threads =          1
    max_scf_cycle =        100
    scf_acc_rho =          1.0e-9
    scf_acc_eev =          1.0e-8
    scf_acc_etot =         1.0e-11

[geom]
    name = "H2O"
    unit = "Angstrom"
    position = """
        O  0.0000000000  0.0000000000  0.1177900000
        H  0.0000000000  0.7571000000 -0.4711600000
        H  0.0000000000 -0.7571000000 -0.4711600000
    """
"##;

fn run_scf(xc: &str) -> SCF {
    let ctrl_str = CTRL_TEMPLATE.replace("XC_PLACEHOLDER", xc);
    let keys = toml::from_str::<serde_json::Value>(&ctrl_str).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf = SCF::build(mol, &None);
    scf_io::scf_without_build(&mut scf, &None);
    scf
}

/// Full-HF Fock matrix transformed into the reference MO basis.
fn hf_fock_mo(scf: &mut SCF) -> (Vec<f64>, Vec<f64>) {
    let saved_h = scf.hamiltonian[0].clone();
    let saved_roots = scf.roothaan_hamiltonian.clone();
    scf.generate_hf_hamiltonian_ri_v(&None);
    let f_ao = scf.hamiltonian[0].to_matrixfull().unwrap();
    scf.hamiltonian[0] = saved_h;
    scf.roothaan_hamiltonian = saved_roots;

    let c = &scf.eigenvectors[0];
    let nbas = c.size[0];
    let nmo = c.size[1];
    let mut f_mo = vec![0.0; nmo * nmo];
    for p in 0..nmo {
        for q in 0..nmo {
            let mut acc = 0.0;
            for mu in 0..nbas {
                for nu in 0..nbas {
                    acc += c[[mu, p]] * f_ao[[mu, nu]] * c[[nu, q]];
                }
            }
            f_mo[p + q * nmo] = acc;
        }
    }
    (f_mo, scf.eigenvalues[0].clone())
}

#[test]
fn rsh_static_term_matches_full_hf_fock_diagonal() {
    let _guard = RSH_GW_TEST_LOCK.lock().unwrap();
    for xc in ["cam-b3lyp", "wb97x", "hse06", "lc-blyp"] {
        let mut scf = run_scf(xc);
        assert!(
            scf.mol.dfadef.is_rsh(),
            "{xc}: expected an RSH reference"
        );
        assert!(
            scf.rimatr_sr.is_some(),
            "{xc}: the SCF must have built rimatr_sr"
        );

        let vxc_nn = ri_gw::vxc_ao2mo(&scf);
        let (_start_mo, num_state, _occ, _vir, homo, _lumo) =
            ri_gw::get_occupation_parameters(&scf, 'Y');
        let v_ex = ri_gw::exchange_matrix_from_scf(&scf);
        let e_ks = scf.eigenvalues[0].clone();
        let (f_mo, _) = hf_fock_mo(&mut scf);

        let mut max_err = 0.0_f64;
        for n in 0..num_state {
            let static_n = e_ks[n] + ri_gw::static_exchange(&scf, &v_ex, n, homo) - vxc_nn[n];
            let f_nn = f_mo[n + n * num_state];
            max_err = max_err.max((static_n - f_nn).abs());
        }
        println!("{xc}: max |static - F_HF,diag| = {max_err:.3e} Ha");
        assert!(
            max_err < 1.0e-8,
            "{xc}: RSH static term differs from the full-HF Fock diagonal by {max_err:.3e} Ha"
        );
    }
}

/// End-to-end smoke test: G0W0 built on an RSH reference must run through the
/// contour-deformation path and produce finite quasiparticle energies.
#[test]
fn rsh_g0w0_runs_and_is_finite() {
    let _guard = RSH_GW_TEST_LOCK.lock().unwrap();
    let qp_block = r##"
[quasiparticle_methods]
gw_or_bse = "gw"
gw_scheme = "extrapolated"
scgw = "g0w0"
gw_variant = "cd"
gw_extrapolate_occ_threshold = 100.0
gw_extrapolate_vir_threshold = 100.0
"##;
    let mut scf = run_scf("cam-b3lyp");
    let keys = toml::from_str::<serde_json::Value>(qp_block).unwrap();
    let qp = ctrl_io::quasiparticle_methods::parse_quasiparticle_keywords(&keys)
        .unwrap()
        .expect("quasiparticle block");
    scf.mol.ctrl.quasiparticle_methods = Some(qp);

    ri_gw::initialize_qp_g_w(&mut scf);
    let vxc_nn = ri_gw::vxc_ao2mo(&scf);
    ri_gw::gw_main(&mut scf, &vxc_nn, &None);

    let (_, num_state, occ_size, _, _homo, _lumo) =
        ri_gw::get_occupation_parameters(&scf, 'Y');
    let qp_energies = &scf.gwqp.0;
    assert_eq!(qp_energies.len(), num_state);
    assert!(
        qp_energies.iter().all(|e| e.is_finite()),
        "non-finite QP energies: {qp_energies:?}"
    );
    let homo_qp = qp_energies[occ_size - 1];
    let homo_ks = scf.eigenvalues[0][occ_size - 1];
    println!(
        "CAM-B3LYP/STO-3G G0W0: HOMO KS = {homo_ks:.6} Ha, QP = {homo_qp:.6} Ha, \
         first 4 QP = {:?}",
        &qp_energies[..4.min(qp_energies.len())]
    );
    assert!(
        (homo_qp - homo_ks).abs() < 2.0,
        "HOMO QP shift is implausibly large: {homo_qp} vs {homo_ks}"
    );
}
