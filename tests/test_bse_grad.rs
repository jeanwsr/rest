//! Validation of the static-BSE analytic gradient (src/ri_bse/bse_grad.rs):
//!   * Dense vs Davidson consistency
//!   * analytic gradient vs central finite differences of the excitation energy

use pyrest::ri_bse::bse_grad::{BseGradEngine, BseSolver, ScreeningEnergy};
use pyrest::ri_gw::gw_grad::{GwGradConfig, GwGradEngine};
use pyrest::scf_io::SCF;
use pyrest::solvers::davidson::DavidsonConfig;

fn build_h2o() -> SCF {
    use pyrest::ctrl_io;
    use pyrest::molecule_io::Molecule;
    use pyrest::scf_io;

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
     scf_acc_rho =          1.0e-12
     scf_acc_eev =          1.0e-10
     scf_acc_etot =         1.0e-14

[geom]
    name = "H2O"
    unit = "Angstrom"
    position = """
        O     0.00000000     0.00000000     0.11730000
        H     0.00000000     0.75720000    -0.46920000
        H     0.00000000    -0.75720000    -0.46920000
    """
"##;
    let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = SCF::build(mol, &None);
    scf_io::scf_without_build(&mut scf_data, &None);
    scf_data.mol.ctrl.quasiparticle_methods =
        Some(pyrest::ctrl_io::quasiparticle_methods::QuasiParticle {
            gw_scheme: "extrapolated".to_string(),
            use_low_rank_contour: true,
            low_rank_tolerance: 1.0e-10,
            ..Default::default()
        });
    scf_data
}

fn bse_energies_displaced(
    scf_data: &SCF,
    atm: usize,
    comp: usize,
    h: f64,
    nroots: usize,
    screening: ScreeningEnergy,
    solver: BseSolver,
    vir_cutoff: f64,
) -> Vec<f64> {
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
    let gw = GwGradEngine::new(&new_scf, cfg);
    let mut bse = BseGradEngine::new_with_cutoff(&new_scf, &gw, 's', screening, vir_cutoff);
    bse.solve(nroots, solver, &DavidsonConfig::default());
    bse.excitation_energies()
}

fn check_bse_gradient(screening: ScreeningEnergy, solver: BseSolver, tol: f64) {
    check_bse_gradient_cutoff(screening, solver, tol, f64::INFINITY)
}

fn check_bse_gradient_cutoff(
    screening: ScreeningEnergy,
    solver: BseSolver,
    tol: f64,
    vir_cutoff: f64,
) {
    let scf_data = build_h2o();
    let cfg = {
        let mut c = GwGradConfig::from_scf(&scf_data);
        c.qpe_tol = 1.0e-13;
        c.exact_residue_z = true;
        c
    };
    let gw = GwGradEngine::new(&scf_data, cfg);
    let natm = gw.natm;
    let mut bse =
        BseGradEngine::new_with_cutoff(&scf_data, &gw, 's', screening, vir_cutoff);
    let nroots = 2usize;
    bse.solve(nroots, solver, &DavidsonConfig::default());
    let (grads, omegas) = bse.analytic_gradients();
    println!(
        "omegas = {:?} (nvir {}/{} active)",
        omegas,
        bse.screen.nvir,
        bse.screen.nvir_full
    );

    // The two lowest roots of this minimal-basis system are nearly degenerate,
    // so individual root gradients are ill-conditioned under FD (root swaps).
    // The SUM over tracked roots is gauge- and crossing-invariant, so we
    // validate the analytic sum against the FD of the summed excitation
    // energies, and also report individual components.
    let h = 3.0e-5;
    let mut grad_fd_sum = vec![0.0f64; natm * 3];
    for atm in 0..natm {
        for comp in 0..3 {
            let ep = bse_energies_displaced(
                &scf_data, atm, comp, h, nroots, screening, solver, vir_cutoff,
            );
            let em = bse_energies_displaced(
                &scf_data, atm, comp, -h, nroots, screening, solver, vir_cutoff,
            );
            grad_fd_sum[atm * 3 + comp] =
                ((ep.iter().sum::<f64>()) - (em.iter().sum::<f64>())) / (2.0 * h);
        }
    }
    let mut grad_ana_sum = vec![0.0f64; natm * 3];
    for k in 0..nroots {
        for i in 0..grad_ana_sum.len() {
            grad_ana_sum[i] += grads[k][i];
        }
    }
    println!("--- roots 0..{} ({:?}, {:?}) ---", nroots, screening, solver);
    println!("analytic sum = {:?}", grad_ana_sum);
    println!("FD sum       = {:?}", grad_fd_sum);
    for i in 0..grad_ana_sum.len() {
        let diff = (grad_ana_sum[i] - grad_fd_sum[i]).abs();
        assert!(
            diff < tol,
            "BSE summed gradient component {} mismatch: ana={} fd={} (diff {:.3e})",
            i, grad_ana_sum[i], grad_fd_sum[i], diff
        );
    }
    println!("summed analytic matches FD within {:.1e}", tol);
}

#[test]
fn bse_grad_h2o_dense_reference_screening_matches_fd() {
    check_bse_gradient(ScreeningEnergy::Reference, BseSolver::Dense, 1.0e-7);
}

#[test]
fn bse_grad_h2o_reference_cutoff_matches_fd() {
    // bse_cutoff_energy-style truncation: restrict the transition space to a
    // strict subset of the virtuals (cutoff between the first two virtual
    // reference energies) with reference-energy screening; the screening
    // response itself stays full.
    let scf_data = build_h2o();
    let eps = &scf_data.eigenvalues[0];
    let nocc = 5usize; // sto-3g H2O
    assert!(eps.len() > nocc + 1, "need at least two virtuals");
    let cutoff = 0.5 * (eps[nocc] + eps[nocc + 1]);
    check_bse_gradient_cutoff(ScreeningEnergy::Reference, BseSolver::Dense, 1.0e-7, cutoff);
}

#[test]
fn diag_bse_residual() {
    let scf_data = build_h2o();
    let cfg = {
        let mut c = GwGradConfig::from_scf(&scf_data);
        c.qpe_tol = 1.0e-12;
        c
    };
    let gw = GwGradEngine::new(&scf_data, cfg);
    let mut bse = BseGradEngine::new(&scf_data, &gw, 's', ScreeningEnergy::Reference);
    bse.solve(2, BseSolver::Dense, &DavidsonConfig::default());
    let qp: Vec<f64> = bse.qp_caches.iter().map(|c| c.omega).collect();
    let (a, b) = pyrest::ri_bse::bse_grad::build_bse_matrices(&bse.screen, &qp, 2.0);
    let dim = bse.screen.nocc * bse.screen.nvir;
    for (k, root) in bse.roots.iter().enumerate() {
        let mut r1 = vec![0.0f64; dim];
        for r in 0..dim {
            let mut ax = 0.0; let mut by = 0.0;
            for c in 0..dim {
                ax += a[r + c*dim]*root.x[c];
                by += b[r + c*dim]*root.y[c];
            }
            r1[r] = ax + by - root.omega*root.x[r];
        }
        let n1: f64 = r1.iter().map(|v| v*v).sum::<f64>().sqrt();
        let mut r2 = vec![0.0f64; dim];
        for r in 0..dim {
            let mut bx = 0.0; let mut ay = 0.0;
            for c in 0..dim {
                bx += b[r + c*dim]*root.x[c];
                ay += a[r + c*dim]*root.y[c];
            }
            r2[r] = bx + ay + root.omega*root.y[r];
        }
        let n2: f64 = r2.iter().map(|v| v*v).sum::<f64>().sqrt();
        let met: f64 = root.x.iter().zip(root.x.iter()).map(|(a,b)| a*b).sum::<f64>()
            - root.y.iter().zip(root.y.iter()).map(|(a,b)| a*b).sum::<f64>();
        println!("[res] root {}: |A X + B Y - w X| = {:.3e}, |B X + A Y - w Y| = {:.3e}, metric = {:.6}, omega = {:.8}",
            k, n1, n2, met, root.omega);
    }
}

#[test]
fn bse_grad_h2o_davidson_matches_dense() {
    let scf_data = build_h2o();
    let cfg = {
        let mut c = GwGradConfig::from_scf(&scf_data);
        c.qpe_tol = 1.0e-13;
        c.exact_residue_z = true;
        c
    };
    let gw = GwGradEngine::new(&scf_data, cfg);
    let mut bse_d = BseGradEngine::new(&scf_data, &gw, 's', ScreeningEnergy::Reference);
    bse_d.solve(2, BseSolver::Dense, &DavidsonConfig::default());
    let mut bse_v = BseGradEngine::new(&scf_data, &gw, 's', ScreeningEnergy::Reference);
    bse_v.solve(2, BseSolver::Davidson, &DavidsonConfig::default());
    let (gd, od) = bse_d.analytic_gradients();
    let (gv, ov) = bse_v.analytic_gradients();
    println!("dense    omegas = {:?}", od);
    println!("davidson omegas = {:?}", ov);
    // solve_bse_davidson requests extra roots internally (standard Davidson
    // practice) and verifies the BSE residuals, so the two lowest roots must
    // agree with the dense reference.
    for k in 0..2 {
        assert!(
            (od[k] - ov[k]).abs() < 1.0e-8,
            "root {} omega mismatch: dense={} davidson={}",
            k, od[k], ov[k]
        );
        for i in 0..gd[k].len() {
            let diff = (gd[k][i] - gv[k][i]).abs();
            assert!(
                diff < 1.0e-6,
                "root {} grad component {} dense/davidson mismatch: {:.3e}",
                k, i, diff
            );
        }
    }
    println!("davidson root-0/1 gradients match dense within 1e-6");
}

#[test]
fn bse_grad_h2o_qp_screening_matches_fd() {
    check_bse_gradient(ScreeningEnergy::Qp, BseSolver::Dense, 1.0e-7);
}
