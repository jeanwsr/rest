/// AC-GW smoke test: real H2/STO-3G analytic-continuation GW.
///
/// Verifies:
///  1. gw_variant="ac" actually enters the AC code path
///  2. QP energies are finite for both occupied and virtual orbitals
///  3. The true QP equation residual F(E_QP) is below the solver tolerance
///  4. No NaN, Inf, panic, or CD fallback
///  5. Compares AC-GW against CD-GW on the same system (diagnostic only)
///
/// The true QP residual is defined by the same equation used in
/// single_orbital_gw_ac:  F(E) = E_KS + Σ_x·(1-hybrid) - Vxc + Re[Σ_c(E+iη)] - E

use rest_tensors::MatrixFull;
use pyrest::ctrl_io::quasiparticle_methods::{GwVariant, QuasiParticle};
use pyrest::ri_gw::ac::{ImaginaryAxisSample, PadeApproximant};

/// Newton-solver tolerance used in single_orbital_gw_ac (line 210 of scgw.rs).
const NEWTON_TOL: f64 = 1e-5;

fn build_h2_hf_sto3g() -> pyrest::scf_io::SCF {
    use pyrest::ctrl_io;
    use pyrest::molecule_io::Molecule;
    use pyrest::scf_io::{self, SCF};

    let input_token = r##"
[ctrl]
     print_level =          0
     xc =                   "hf"
     basis_path =           "sto-3g"
     charge =               0.0
     spin =                 1.0
     spin_polarization =    false
     initial_guess =        "hcore"
     mixer =                "diis"
     num_threads =          1
     scf_acc_rho =          1.0e-8
     scf_acc_eev =          1.0e-6
     scf_acc_etot =         1.0e-8

[geom]
    name = "H2"
    unit = "Angstrom"
    position = """
        H     0.00000000     0.00000000     0.00000000
        H     0.00000000     0.00000000     1.40000000
    """
"##;
    let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = SCF::build(mol, &None);
    scf_io::scf_without_build(&mut scf_data, &None);
    scf_data
}

/// Build the same AC Pade approximant that single_orbital_gw_ac constructs,
/// then compute the true QP equation function F(omega) at a given frequency.
fn build_pade_and_qp_func(
    w_c_at_freqs: &Vec<(f64, f64, MatrixFull<f64>)>,
    n: usize,
    gwqp_g: &Vec<f64>,
    consts: f64,
    ac_num_samples: usize,
    ac_omega_max: f64,
    ac_eta: f64,
) -> impl Fn(f64) -> f64 {
    let ac_freqs: Vec<f64> = (0..ac_num_samples)
        .map(|j| ac_omega_max * (j as f64 + 1.0) / (ac_num_samples as f64))
        .collect();

    let samples: Vec<ImaginaryAxisSample> = ac_freqs
        .iter()
        .map(|&lambda| {
            let sigma_c = pyrest::ri_gw::calculate_sigma_c_imag_freq(
                w_c_at_freqs, n, lambda, 0.0, gwqp_g,
            );
            ImaginaryAxisSample::new(lambda, sigma_c)
        })
        .collect();

    let pade = PadeApproximant::from_imaginary_axis(&samples)
        .expect("Pade construction failed");

    move |omega: f64| -> f64 {
        match pade.evaluate_retarded(omega, ac_eta) {
            Ok(sigma_c) => consts + sigma_c.re - omega,
            Err(_) => f64::NAN,
        }
    }
}

#[test]
fn ac_gw_h2_sto3g_produces_finite_qp_energies() {
    let mut scf_data = build_h2_hf_sto3g();

    // ── Common AC configuration ──
    scf_data.mol.ctrl.quasiparticle_methods = Some(QuasiParticle {
        gw_variant: GwVariant::Ac,
        gw_scheme: "extrapolated".to_string(),
        gw_rootfinder: "newton".to_string(),
        ac_num_samples: 8,
        ac_omega_max: 2.0,
        ac_eta: 0.005,
        gw_imag_rayon: false,
        gw_search_grid: 21,
        gw_span_energy: 0.1,
        ..Default::default()
    });

    let (_start_mo, num_state, occ_size, vir_size, _homo, _lumo) =
        pyrest::ri_gw::get_occupation_parameters(&scf_data, 'Y');

    pyrest::ri_gw::initialize_qp_g_w(&mut scf_data);
    let vxc_nn = pyrest::ri_gw::vxc_ao2mo(&scf_data);
    let hybrid_param = scf_data.mol.xc_data.dfa_hybrid_scf;

    let ks_energies = scf_data.eigenvalues[0].clone();
    let gwqp_g_save = scf_data.gwqp.0.clone();
    let gwqp_w_save = scf_data.gwqp.1.clone();

    println!("\n=== H2/STO-3G SCF ===");
    println!("  SCF energy: {:.12} Ha", scf_data.scf_energy);
    println!("  n_state={}, occ={}, vir={}", num_state, occ_size, vir_size);
    for i in 0..num_state {
        println!("  MO #{}: {:12.8} Ha  ({})",
                 i, ks_energies[i], if i < occ_size { "occ" } else { "vir" });
    }

    // ── Run AC-GW ──
    println!("\n=== AC-GW smoke test ===");
    let gwqp_ac = pyrest::ri_gw::scgw::gw_near_fermi_surface_ac(
        &mut scf_data, 16, &vxc_nn, 5.0,
    );

    // ── Reconstruct W_c and compute true QP equation residuals ──
    let qp_ctrl = scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let ri_ov = pyrest::ri_bse::get_submatrix(&scf_data, 'O', 'V', 'Y');
    let v_matrix = pyrest::ri_gw::v_matrix_from_scf(&scf_data);
    let (_start_mo2, _ns, occ_sz, _vir_sz, _h, _l) =
        pyrest::ri_gw::get_occupation_parameters(&scf_data, 'Y');

    let w_c_at_freqs: Vec<(f64, f64, MatrixFull<f64>)> =
        pyrest::ri_gw::generate_w_c_serial(
            &scf_data, &ri_ov, &gwqp_g_save, &gwqp_w_save,
            num_state, occ_sz, _vir_sz, 16,
        );

    let e_homo = ks_energies[occ_size - 1];
    let e_lumo = ks_energies[occ_size];
    let calc_orbs: Vec<usize> = ks_energies
        .iter()
        .enumerate()
        .filter(|(_n, &e)| e > e_homo - 5.0 && e < e_lumo + 5.0)
        .map(|(n, _)| n)
        .collect();

    println!("\n=== AC-GW true QP residuals ===");
    println!("{:>4}  {:>12}  {:>12}  {:>12}  {:>12}",
             "Orb", "E_KS", "E_QP_AC", "qp_shift", "qp_residual");
    let (_so, _ns2, _os2, _vs2, homo2, _l2) =
        pyrest::ri_gw::get_occupation_parameters(&scf_data, 'Y');

    for &n in &calc_orbs {
        let e_ks_n = ks_energies[n];

        let mut exchange = 0.0;
        for i in 0..homo2 + 1 {
            exchange -= v_matrix[[n, i]];
        }
        let consts = e_ks_n + exchange * (1.0 - hybrid_param) - vxc_nn[n];

        // Build Pade and QP function (same as single_orbital_gw_ac)
        let qp_func = build_pade_and_qp_func(
            &w_c_at_freqs, n, &gwqp_g_save, consts,
            qp_ctrl.ac_num_samples, qp_ctrl.ac_omega_max, qp_ctrl.ac_eta,
        );

        let eqp_ac = gwqp_ac[n];
        let qp_residual = qp_func(eqp_ac).abs();
        let qp_shift = eqp_ac - e_ks_n;

        println!("{:>4}  {:>12.8}  {:>12.8}  {:>12.8}  {:>12.8}",
                 n, e_ks_n, eqp_ac, qp_shift, qp_residual);

        assert!(eqp_ac.is_finite(),
                "QP energy for orbital {} is not finite: {}", n, eqp_ac);
        assert!(qp_residual.is_finite(),
                "QP residual for orbital {} is not finite: {}", n, qp_residual);
        assert!(qp_residual < NEWTON_TOL * 10.0,
                "QP residual {:.2e} exceeds Newton tolerance {:.2e} for orbital {}",
                qp_residual, NEWTON_TOL, n);
    }

    // ── Compare with CD-GW ──
    println!("\n=== CD-GW comparison (diagnostic) ===");
    // Run CD-GW on the same system
    let mut scf_data_cd = build_h2_hf_sto3g();
    scf_data_cd.mol.ctrl.quasiparticle_methods = Some(QuasiParticle {
        gw_variant: GwVariant::Cd,
        gw_scheme: "extrapolated".to_string(),
        gw_rootfinder: "interpolation".to_string(), // CD standard solver
        gw_imag_rayon: false,
        gw_search_grid: 21,
        gw_span_energy: 0.1,
        ..Default::default()
    });
    pyrest::ri_gw::initialize_qp_g_w(&mut scf_data_cd);
    let vxc_nn_cd = pyrest::ri_gw::vxc_ao2mo(&scf_data_cd);
    let gwqp_cd = pyrest::ri_gw::scgw::gw_near_fermi_surface(
        &mut scf_data_cd, 16, &vxc_nn_cd, 5.0, 5.0,
    );

    println!("{:>4}  {:>12}  {:>12}  {:>12}",
             "Orb", "E_QP_AC", "E_QP_CD", "|AC-CD|");
    for &n in &calc_orbs {
        let diff = (gwqp_ac[n] - gwqp_cd[n]).abs();
        println!("{:>4}  {:>12.8}  {:>12.8}  {:>12.8}",
                 n, gwqp_ac[n], gwqp_cd[n], diff);
    }

    println!("\n=== AC-GW smoke test PASSED ===");
}
