//! Validation of the LR-CD G0W0 analytic gradient (src/ri_gw/gw_grad.rs):
//! analytic gradient vs central finite differences of the QP energy
//! (H2 HOMO, H2O HOMO and LUMO).

use pyrest::ri_gw::gw_grad::{GwGradConfig, GwGradEngine};
use pyrest::scf_io::SCF;

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
            use_low_rank_contour: true,
            low_rank_tolerance: 1.0e-10,
            // REST default grid; finer grids move the piecewise jumps of the
            // quantised residue protocol and can invalidate FD windows
            nomega_chi_real: 6,
            ..Default::default()
        });
}

fn qp_energy_displaced(scf_data: &SCF, atm: usize, comp: usize, h: f64, target: usize) -> f64 {
    let mut new_scf = scf_data.clone();
    let mut vec_xyz = vec![0.0f64; 3];
    vec_xyz[comp] = h;
    new_scf.mol.geom.geom_shift(atm, vec_xyz);
    new_scf.mol.ctrl.print_level = 0;
    new_scf.mol.ctrl.initial_guess = String::from("inherit");
    pyrest::scf_io::initialize_scf(&mut new_scf, &None);
    pyrest::scf_io::scf_without_build(&mut new_scf, &None);
    let mut cfg = GwGradConfig::from_scf(&new_scf);
    cfg.qpe_tol = 1.0e-13;
    cfg.exact_residue_z = true;
    let engine = GwGradEngine::new(&new_scf, cfg);
    engine.qp_energy_of(target)
}

fn check_gradient(scf_data: &SCF, target: usize, tol: f64) {
    let mut cfg = GwGradConfig::from_scf(scf_data);
    cfg.qpe_tol = 1.0e-13; // tighter than production: reduces FD jitter
    cfg.exact_residue_z = true; // smooth functional: consistent FD validation
    let engine = GwGradEngine::new(scf_data, cfg);
    let (grad_ana, omega, z) = engine.analytic_gradient(target);
    println!("target {} : omega = {:.12}, Z = {:.12}", target, omega, z);
    println!("analytic grad = {:?}", grad_ana);
    assert!(omega.is_finite());

    let natm = engine.natm;
    let mut grad_fd = vec![0.0f64; natm * 3];
    let h = 1.0e-4;
    for atm in 0..natm {
        for comp in 0..3 {
            // The REST protocol quantises residue frequencies to the nearest
            // real-axis grid point (piecewise constant).  If a +-h window
            // straddles a grid midpoint the finite difference is spoiled;
            // shrink h until the difference is stable.
            let mut hh = h;
            for _ in 0..4 {
                let ep = qp_energy_displaced(scf_data, atm, comp, hh, target);
                let em = qp_energy_displaced(scf_data, atm, comp, -hh, target);
                if std::env::var("REST_TEST_DBG").is_ok() {
                    println!("[fd DBG] target={} atm={} comp={} hh={:.0e}: E(+h)={:.10} E(-h)={:.10}",
                        target, atm, comp, hh, ep, em);
                }
                let g = (ep - em) / (2.0 * hh);
                let hh2 = hh * 0.1;
                let ep2 = qp_energy_displaced(scf_data, atm, comp, hh2, target);
                let em2 = qp_energy_displaced(scf_data, atm, comp, -hh2, target);
                let g2 = (ep2 - em2) / (2.0 * hh2);
                hh = hh2;
                if (g - g2).abs() < tol * 10.0 {
                    grad_fd[atm * 3 + comp] = g2;
                    break;
                }
                grad_fd[atm * 3 + comp] = g2;
            }
        }
    }
    println!("FD grad       = {:?}", grad_fd);
    let mut worst = (0.0f64, 0usize);
    for i in 0..grad_ana.len() {
        let diff = (grad_ana[i] - grad_fd[i]).abs();
        if diff > worst.0 {
            worst = (diff, i);
        }
        println!(
            "  comp {:2}: ana = {:+.10e}, fd = {:+.10e}, diff = {:.3e}",
            i, grad_ana[i], grad_fd[i], diff
        );
        assert!(
            diff < tol,
            "GW gradient component {} mismatch: ana={} fd={} (diff {:.3e} > {:.1e})",
            i,
            grad_ana[i],
            grad_fd[i],
            diff,
            tol
        );
    }
    println!("worst diff = {:.3e} at component {}", worst.0, worst.1);
}

#[test]
fn gw_grad_h2_homo_matches_finite_difference() {
    let mut scf_data = build_mol(
        "H2",
        "
        H     0.00000000     0.00000000     0.00000000
        H     0.00000000     0.00000000     1.40000000
    ",
    );
    set_gw_ctrl(&mut scf_data);
    let nocc = scf_data.homo[0] + 1;
    check_gradient(&scf_data, nocc - 1, 1.0e-6);
}

#[test]
fn gw_grad_h2o_homo_and_lumo_match_finite_difference() {
    let mut scf_data = build_mol(
        "H2O",
        "
        O     0.00000000     0.00000000     0.11730000
        H     0.00000000     0.75720000    -0.46920000
        H     0.00000000    -0.75720000    -0.46920000
    ",
    );
    set_gw_ctrl(&mut scf_data);
    let nocc = scf_data.homo[0] + 1;
    let lumo = scf_data.lumo[0];
    // symmetry-zero components carry FD noise ~5e-6, nonzero components
    // agree to ~1e-7
    check_gradient(&scf_data, nocc - 1, 2.0e-5); // HOMO
    check_gradient(&scf_data, lumo, 2.0e-5); // LUMO
}
