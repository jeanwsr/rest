//! Diagnostic for the full-CD engine's non-frontier-orbital gradient defect.
//!
//! Decomposition of the analytic gradient
//!     d Omega_n/dR = Z * (d consts_n/dR + d Sigma_sub/dR |_omega)
//! into the fixed-omega pullback and the `Z` factor, each compared with its
//! own finite difference.

use pyrest::ri_gw::gw_grad::{GwCdGradEngine, GwGradConfig};
use pyrest::scf_io::SCF;

const H2O: &str = "
        O     0.000000000000     0.000000000000     0.117300000000
        H     0.000000000000     0.757200000000    -0.469200000000
        H     0.000000000000    -0.757200000000    -0.469200000000
    ";

fn build_mol(name: &str, geom: &str) -> SCF {
    use pyrest::ctrl_io;
    use pyrest::molecule_io::Molecule;
    use pyrest::scf_io;

    let input_token = format!(
        r##"
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
     scf_acc_rho =          1.0e-12
     scf_acc_eev =          1.0e-10
     scf_acc_etot =         1.0e-14

[geom]
    name = "{}"
    unit = "Angstrom"
    position = """{}"""
"##,
        name, geom
    );
    let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = SCF::build(mol, &None);
    scf_io::scf_without_build(&mut scf_data, &None);
    scf_data
}

fn set_gw_ctrl(scf: &mut SCF) {
    scf.mol.ctrl.quasiparticle_methods =
        Some(pyrest::ctrl_io::quasiparticle_methods::QuasiParticle {
            gw_scheme: "extrapolated".to_string(),
            use_low_rank_contour: false,
            low_rank_tolerance: 1.0e-10,
            nomega_chi_real: 6,
            ..Default::default()
        });
}

fn cfg_of(scf: &SCF) -> GwGradConfig {
    let mut cfg = GwGradConfig::from_scf(scf);
    cfg.qpe_tol = 1.0e-13;
    cfg
}

fn displaced(scf_data: &SCF, atm: usize, comp: usize, h: f64) -> SCF {
    let mut new_scf = scf_data.clone();
    let mut vec_xyz = vec![0.0f64; 3];
    vec_xyz[comp] = h;
    new_scf.mol.geom.geom_shift(atm, vec_xyz);
    new_scf.mol.ctrl.print_level = 0;
    new_scf.mol.ctrl.initial_guess = String::from("inherit");
    pyrest::scf_io::initialize_scf(&mut new_scf, &None);
    pyrest::scf_io::scf_without_build(&mut new_scf, &None);
    new_scf
}

/// Per-orbital decomposition:
///  * `fixed-omega pullback` vs FD of `e_n(R) + Re Sigma_sub(omega_ref, R)`
///  * `Z` vs FD of `omega(R)` through `1/(1 - d Sigma/d omega)`
///  * full analytic gradient vs FD of `omega(R)`
#[test]
fn h2o_cd_decomposition() {
    let mut scf_data = build_mol("H2O", H2O);
    set_gw_ctrl(&mut scf_data);
    let engine = GwCdGradEngine::new(&scf_data, cfg_of(&scf_data));
    let natm = scf_data.mol.geom.elem.len();
    let nmo = scf_data.eigenvalues[0].len();
    let h = 1.0e-4;

    let targets: Vec<usize> = match std::env::var("REST_CD_TARGET") {
        Ok(s) => vec![s.parse().unwrap()],
        Err(_) => (0..nmo).collect(),
    };

    for t in targets {
        let cache = engine.build_target_cache(t);
        let omega_ref = cache.omega;
        let z_ana = cache.z_factor;
        let (sig_ref, dsig_ref) = engine.sigma_sub_at(omega_ref, t);
        let z_ana2 = 1.0 / (1.0 - dsig_ref);
        let g_fixed = engine.fixed_omega_gradient(&cache);
        let (g_full, om_full, z_full) = engine.analytic_gradient(t);
        println!(
            "=== target {} : omega {:.10}  Sigma(omega_ref) {:+.10}  Z(ana) {:.10} / {:.10}",
            t, omega_ref, sig_ref, z_ana, z_ana2
        );
        // finite differences
        let mut fd_fixed = vec![0.0f64; natm * 3];
        let mut fd_full = vec![0.0f64; natm * 3];
        let mut fd_omega = vec![0.0f64; natm * 3];
        for atm in 0..natm {
            for comp in 0..3 {
                let ep = displaced(&scf_data, atm, comp, h);
                let em = displaced(&scf_data, atm, comp, -h);
                let epp = GwCdGradEngine::new(&ep, cfg_of(&ep));
                let emm = GwCdGradEngine::new(&em, cfg_of(&em));
                let phi_p = epp.base.e[t] + epp.sigma_sub_at(omega_ref, t).0;
                let phi_m = emm.base.e[t] + emm.sigma_sub_at(omega_ref, t).0;
                fd_fixed[atm * 3 + comp] = (phi_p - phi_m) / (2.0 * h);
                fd_full[atm * 3 + comp] = (epp.qp_energy_of(t) - emm.qp_energy_of(t)) / (2.0 * h);
                fd_omega[atm * 3 + comp] = (epp.qp_energy_of(t) - emm.qp_energy_of(t)) / (2.0 * h);
            }
        }
        let mut worst_fix = (0.0f64, 0usize);
        let mut worst_full = (0.0f64, 0usize);
        for i in 0..g_full.len() {
            let df = (g_fixed[i] - fd_fixed[i]).abs();
            let dg = (g_full[i] - fd_full[i]).abs();
            if df > worst_fix.0 {
                worst_fix = (df, i);
            }
            if dg > worst_full.0 {
                worst_full = (dg, i);
            }
            println!(
                "    comp {:2}: fixed-omega ana {:+.8e} fd {:+.8e} (d {:.2e}) | full ana {:+.8e} fd {:+.8e} (d {:.2e})",
                i, g_fixed[i], fd_fixed[i], df, g_full[i], fd_full[i], dg
            );
        }
        println!(
            "    WORST fixed-omega {:.3e} at comp {} | WORST full {:.3e} at comp {} | Z(analytic) {:.10}",
            worst_fix.0, worst_fix.1, worst_full.0, worst_full.1, z_full
        );
        assert!(
            worst_full.0 < 1.0e-6,
            "CD gradient for target {} does not match the finite difference: {:.3e} > 1e-6 \
             at component {}",
            t,
            worst_full.0,
            worst_full.1
        );
        let _ = fd_omega;
    }
}
