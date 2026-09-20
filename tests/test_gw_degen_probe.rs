//! Degeneracy diagnostics for the CD-G0W0 analytic gradient.
//!
//! Control case: H2O (no degenerate frontier orbitals).
//! Target case : C6H6 (D6h; the e1g HOMO is a symmetry-protected degenerate
//!               pair, and several occupied/virtual shells are degenerate).

use pyrest::ri_gw::gw_grad::{GwCdGradEngine, GwGradConfig};
use pyrest::scf_io::SCF;

/// Exactly D6h benzene (C-C = 1.39 A, C-H = 1.09 A), coordinates written with
/// identical strings for symmetry-equivalent atoms so that the degenerate
/// shells are degenerate to machine precision.
const C6H6: &str = "
        C     0.000000000000     1.390000000000     0.000000000000
        C    -1.203775311260     0.695000000000     0.000000000000
        C    -1.203775311260    -0.695000000000     0.000000000000
        C     0.000000000000    -1.390000000000     0.000000000000
        C     1.203775311260    -0.695000000000     0.000000000000
        C     1.203775311260     0.695000000000     0.000000000000
        H     0.000000000000     2.480000000000     0.000000000000
        H    -2.147743001385     1.240000000000     0.000000000000
        H    -2.147743001385    -1.240000000000     0.000000000000
        H     0.000000000000    -2.480000000000     0.000000000000
        H     2.147743001385    -1.240000000000     0.000000000000
        H     2.147743001385     1.240000000000     0.000000000000
    ";

const H2O: &str = "
        O     0.00000000     0.00000000     0.11730000
        H     0.00000000     0.75720000    -0.46920000
        H     0.00000000    -0.75720000    -0.46920000
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

/// Groups of orbitals whose energies agree within `tol`.
fn blocks(scf: &SCF, tol: f64) -> Vec<Vec<usize>> {
    let e = &scf.eigenvalues[0];
    let mut out: Vec<Vec<usize>> = Vec::new();
    for i in 0..e.len() {
        let push_new = match out.last() {
            Some(b) => (e[i] - e[*b.last().unwrap()]).abs() >= tol,
            None => true,
        };
        if push_new {
            out.push(vec![i]);
        } else {
            out.last_mut().unwrap().push(i);
        }
    }
    out
}

fn report_degeneracies(scf: &SCF, tol: f64) {
    let nocc = scf.homo[0] + 1;
    let e = &scf.eigenvalues[0];
    println!("nocc = {}, nmo = {}", nocc, e.len());
    for b in blocks(scf, tol) {
        if b.len() > 1 {
            println!(
                "  degenerate block {:?} ({}) e = {:+.10}  spread = {:.3e}",
                b,
                if b[0] < nocc { "occ" } else { "vir" },
                e[b[0]],
                e[*b.last().unwrap()] - e[b[0]]
            );
        }
    }
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

/// The GW-gradient config used by the probe: tight QP root, and the smooth
/// (exact moving-residue) protocol when `REST_EXACT_RES` is set — the
/// quantised nearest-grid residues make the QP energy piecewise constant in
/// `R`, which invalidates finite differences away from the frontier.
fn probe_cfg(scf: &SCF) -> GwGradConfig {
    let mut cfg = GwGradConfig::from_scf(scf);
    cfg.qpe_tol = 1.0e-13;
    cfg.exact_residue_z = std::env::var("REST_EXACT_RES").is_ok();
    cfg
}

fn fd_grad(scf_data: &SCF, h: f64, f: &dyn Fn(&GwCdGradEngine) -> f64) -> Vec<f64> {
    fd_grad_multi(scf_data, h, &[f]).remove(0)
}

/// Central differences of several functionals from ONE set of displaced
/// geometries.
fn fd_grad_multi(
    scf_data: &SCF,
    h: f64,
    fs: &[&dyn Fn(&GwCdGradEngine) -> f64],
) -> Vec<Vec<f64>> {
    let natm = scf_data.mol.geom.elem.len();
    let mut g = vec![vec![0.0f64; natm * 3]; fs.len()];
    for atm in 0..natm {
        for comp in 0..3 {
            let ep = displaced(scf_data, atm, comp, h);
            let em = displaced(scf_data, atm, comp, -h);
            let engp = GwCdGradEngine::new(&ep, probe_cfg(&ep));
            let engm = GwCdGradEngine::new(&em, probe_cfg(&em));
            for (k, f) in fs.iter().enumerate() {
                g[k][atm * 3 + comp] = (f(&engp) - f(&engm)) / (2.0 * h);
            }
        }
    }
    g
}

#[test]
fn c6h6_mo_table() {
    let mut scf_data = build_mol("C6H6", C6H6);
    set_gw_ctrl(&mut scf_data);
    let e = &scf_data.eigenvalues[0];
    let nocc = scf_data.homo[0] + 1;
    println!("nocc = {}, nmo = {}", nocc, e.len());
    for i in 0..e.len() {
        let d = if i + 1 < e.len() { e[i + 1] - e[i] } else { f64::NAN };
        println!(
            "  mo {:3} {} e = {:+.10}  gap_next = {:.3e}{}",
            i,
            if i < nocc { "occ" } else { "vir" },
            e[i],
            d,
            if d.abs() < 1.0e-6 { "   <-- degenerate" } else { "" }
        );
    }
    report_degeneracies(&scf_data, 1.0e-6);
}

fn mean(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| 0.5 * (x + y)).collect()
}

/// Analytic gradient of `t` vs
///  (a) the central FD of the QP energy of `t` itself,
///  (b) the central FD of the mean QP energy of `t`'s degenerate shell
///      (well defined for any shell size; for a 2-fold shell it equals (a)).
fn probe_target(scf_data: &SCF, t: usize) {
    let engine = GwCdGradEngine::new(scf_data, probe_cfg(scf_data));
    let shell = engine.base.degenerate_shell(t);
    let d = shell.len() as f64;
    println!("--- target {}   shell {:?} (size {})", t, shell, shell.len());
    let (g, om, z) = engine.analytic_gradient(t);
    println!("    omega = {:.12}  Z = {:.12}", om, z);
    let f_t = |eng: &GwCdGradEngine| eng.qp_energy_of(t);
    let f_shell =
        |eng: &GwCdGradEngine| shell.iter().map(|&m| eng.qp_energy_of(m)).sum::<f64>() / d;
    let fds = fd_grad_multi(scf_data, 1.0e-4, &[&f_t, &f_shell]);
    let (fd_t, fd_s) = (&fds[0], &fds[1]);
    let mut w_t = 0.0f64;
    let mut w_s = 0.0f64;
    for i in 0..g.len() {
        let dt = (g[i] - fd_t[i]).abs();
        let ds = (g[i] - fd_s[i]).abs();
        w_t = w_t.max(dt);
        w_s = w_s.max(ds);
        println!(
            "    comp {:2}: ana {:+.10e} | fd_target {:+.10e} (d {:.2e})  fd_shell {:+.10e} (d {:.2e})",
            i, g[i], fd_t[i], dt, fd_s[i], ds
        );
    }
    println!(
        "    WORST |ana - fd_target| = {:.3e}   |ana - fd_shell| = {:.3e}",
        w_t, w_s
    );
}

/// Compare the analytic gradient of one target with
///  (a) central FD of that target's QP energy,
///  (b) central FD of the mean of the block's QP energies.
#[test]
fn c6h6_gradient_probe() {
    let mut scf_data = build_mol("C6H6", C6H6);
    set_gw_ctrl(&mut scf_data);
    report_degeneracies(&scf_data, 1.0e-6);
    let nocc = scf_data.homo[0] + 1;
    // HOMO shell (e1g) and a non-degenerate orbital as a control
    match std::env::var("REST_PROBE_TARGET") {
        Ok(s) => probe_target(&scf_data, s.parse().unwrap()),
        Err(_) => {
            probe_target(&scf_data, nocc - 1); // HOMO (degenerate)
            probe_target(&scf_data, 16); // a1g-ish, non-degenerate
        }
    }
}

#[test]
fn h2o_control_probe() {
    let mut scf_data = build_mol("H2O", H2O);
    set_gw_ctrl(&mut scf_data);
    report_degeneracies(&scf_data, 1.0e-6);
    let nocc = scf_data.homo[0] + 1;
    probe_target(&scf_data, nocc - 1);
}

/// Is the analytic gradient reliable for *every* orbital, or only for the
/// frontier ones?  H2O has no degeneracy at all, so any mismatch here is
/// unrelated to the degenerate-shell problem.
#[test]
fn h2o_all_orbitals_probe() {
    let mut scf_data = build_mol("H2O", H2O);
    set_gw_ctrl(&mut scf_data);
    let nmo = scf_data.eigenvalues[0].len();
    for t in 0..nmo {
        probe_target(&scf_data, t);
    }
}

/// Is the finite difference itself converged, or is the analytic gradient
/// wrong?  Scan the displacement for every H2O orbital.
#[test]
fn h2o_fd_h_scan() {
    let mut scf_data = build_mol("H2O", H2O);
    set_gw_ctrl(&mut scf_data);
    let nmo = scf_data.eigenvalues[0].len();
    let engine = GwCdGradEngine::new(&scf_data, probe_cfg(&scf_data));
    for t in 0..nmo {
        let (g, om, _z) = engine.analytic_gradient(t);
        println!("--- target {}  omega {:.10}", t, om);
        for &h in &[1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6] {
            let f_t = |eng: &GwCdGradEngine| eng.qp_energy_of(t);
            let fd = fd_grad(&scf_data, h, &f_t);
            let worst = (0..g.len()).map(|i| (g[i] - fd[i]).abs()).fold(0.0f64, f64::max);
            println!(
                "    h = {:.0e}: worst |ana - fd| = {:.3e}   fd[2] = {:+.8e} fd[7] = {:+.8e}",
                h, worst, fd[2], fd[7]
            );
        }
        println!("    ana[2] = {:+.8e}  ana[7] = {:+.8e}", g[2], g[7]);
    }
}

/// Split the gradient into the orbital-response part (`eps1`, from the CP-HF
/// + Roothaan restoration) and the GW pullback part: compare `eps1[n]` with
/// the finite difference of the SCF eigenvalue of orbital `n`.
#[test]
fn h2o_eps1_vs_fd() {
    let mut scf_data = build_mol("H2O", H2O);
    set_gw_ctrl(&mut scf_data);
    let nmo = scf_data.eigenvalues[0].len();
    let engine = GwCdGradEngine::new(&scf_data, probe_cfg(&scf_data));
    let responses = engine.base.canonical_response_batch();
    let natm = scf_data.mol.geom.elem.len();
    let h = 1.0e-4;
    println!("SCF eigenvalues: {:?}", scf_data.eigenvalues[0]);
    for n in 0..nmo {
        let mut worst = (0.0f64, 0usize, 0.0, 0.0);
        for atm in 0..natm {
            for comp in 0..3 {
                let ep = displaced(&scf_data, atm, comp, h);
                let em = displaced(&scf_data, atm, comp, -h);
                let fd = (ep.eigenvalues[0][n] - em.eigenvalues[0][n]) / (2.0 * h);
                let ana = responses[atm * 3 + comp].1[n];
                let d = (ana - fd).abs();
                if d > worst.0 {
                    worst = (d, atm * 3 + comp, ana, fd);
                }
            }
        }
        println!(
            "  mo {:2}: worst |eps1 - d e/dR| = {:.3e} at comp {} (ana {:+.8e} fd {:+.8e})",
            n, worst.0, worst.1, worst.2, worst.3
        );
    }
}

/// Same check with the low-rank engine (`GwGradEngine`) to localise whether a
/// mismatch lives in the shared response or in the CD pullback.
#[test]
fn h2o_lr_engine_probe() {
    use pyrest::ri_gw::gw_grad::GwGradEngine;
    let mut scf_data = build_mol("H2O", H2O);
    set_gw_ctrl(&mut scf_data);
    if let Some(qp) = scf_data.mol.ctrl.quasiparticle_methods.as_mut() {
        qp.use_low_rank_contour = true;
    }
    let cfg = probe_cfg(&scf_data);
    let engine = GwGradEngine::new(&scf_data, cfg);
    for t in [0usize, 1, 2, 3, 4, 5, 6] {
        let (g, om, _z) = engine.analytic_gradient(t);
        println!("--- LR target {} omega {:.10}", t, om);
        for &h in &[1.0e-4, 1.0e-5] {
            let f_t = |eng: &GwGradEngine| eng.qp_energy_of(t);
            let natm = scf_data.mol.geom.elem.len();
            let mut worst = 0.0f64;
            for atm in 0..natm {
                for comp in 0..3 {
                    let ep = displaced(&scf_data, atm, comp, h);
                    let em = displaced(&scf_data, atm, comp, -h);
                    let cp = probe_cfg(&ep);
                    let cm = probe_cfg(&em);
                    let gp = f_t(&GwGradEngine::new(&ep, cp));
                    let gm = f_t(&GwGradEngine::new(&em, cm));
                    let fd = (gp - gm) / (2.0 * h);
                    worst = worst.max((g[atm * 3 + comp] - fd).abs());
                }
            }
            println!("    h = {:.0e}: worst |ana - fd| = {:.3e}", h, worst);
        }
    }
}

/// Realistic input geometry: coordinates rounded to 1e-6 A, so the
/// symmetry-protected shells are split by ~1e-5..1e-6 Ha instead of being
/// exactly degenerate.  Compares the legacy response (degenerate_tol = 0)
/// with the shell treatment and with finite differences of several steps.
#[test]
fn c6h6_rounded_geometry_scan() {
    const ROUNDED: &str = "
        C     0.00000000     1.39000000     0.00000000
        C     1.20384900     0.69500000     0.00000000
        C     1.20384900    -0.69500000     0.00000000
        C     0.00000000    -1.39000000     0.00000000
        C    -1.20384900    -0.69500000     0.00000000
        C    -1.20384900     0.69500000     0.00000000
        H     0.00000000     2.48000000     0.00000000
        H     2.14773500     1.24000000     0.00000000
        H     2.14773500    -1.24000000     0.00000000
        H     0.00000000    -2.48000000     0.00000000
        H    -2.14773500    -1.24000000     0.00000000
        H    -2.14773500     1.24000000     0.00000000
    ";
    let mut scf_data = build_mol("C6H6r", ROUNDED);
    set_gw_ctrl(&mut scf_data);
    report_degeneracies(&scf_data, 1.0e-4);
    let nocc = scf_data.homo[0] + 1;
    let t = nocc - 1;
    let e = &scf_data.eigenvalues[0];
    println!(
        "HOMO pair gap: e[{}] - e[{}] = {:.3e} Ha",
        t,
        t - 1,
        e[t] - e[t - 1]
    );
    for &tol in &[0.0f64, 1.0e-4] {
        let mut cfg = probe_cfg(&scf_data);
        cfg.degenerate_tol = tol;
        let engine = GwCdGradEngine::new(&scf_data, cfg);
        let shell = engine.base.degenerate_shell(t);
        let (g, om, _z) = engine.analytic_gradient(t);
        println!(
            "--- degenerate_tol = {:.1e}: shell {:?} omega {:.10}  sum(g over x) = {:+.6e}",
            tol,
            shell,
            om,
            (0..12).map(|a| g[a * 3]).sum::<f64>()
        );
        let d = shell.len() as f64;
        let f_t = |eng: &GwCdGradEngine| eng.qp_energy_of(t);
        let f_shell =
            |eng: &GwCdGradEngine| shell.iter().map(|&m| eng.qp_energy_of(m)).sum::<f64>() / d;
        for &h in &[1.0e-3, 1.0e-4, 1.0e-5] {
            let fds = fd_grad_multi(&scf_data, h, &[&f_t, &f_shell]);
            let dt = (0..g.len()).map(|i| (g[i] - fds[0][i]).abs()).fold(0.0f64, f64::max);
            let ds = (0..g.len()).map(|i| (g[i] - fds[1][i]).abs()).fold(0.0f64, f64::max);
            println!(
                "    h = {:.0e}: worst |ana - fd(index)| = {:.3e}   worst |ana - fd(shell mean)| = {:.3e}",
                h, dt, ds
            );
        }
    }
}
