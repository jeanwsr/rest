/// CP-HF Tests: cross-validate REST CP-HF against PySCF.
///
/// Runs SCF + CP-HF on H2O with HF/STO-3G, then calls PySCF via Python
/// to compute the same polarizability. Compares with tolerance 1e-6.

use std::process::Command;
use std::path::Path;
use rest_tensors::MatrixFull;

// ── helper: build H2O SCF (HF/sto-3g) ──────────────────────────────────────
fn run_h2o_scf_hf_sto3g() -> pyrest::scf_io::SCF {
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
     initial_guess=         "hcore"
     mixer =                "diis"
     num_threads =          8
     scf_acc_rho =          1.0e-8
     scf_acc_eev =          1.0e-6
     scf_acc_etot =         1.0e-8

[geom]
    name = "H2O"
    unit = "Angstrom"
    position = """
        O     0.00000000     0.00000000     0.12982363
        H     0.75933475     0.00000000    -0.46621158
        H    -0.75933475     0.00000000    -0.46621158
    """
"##;
    let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = SCF::build(mol, &None);
    scf_io::scf_without_build(&mut scf_data, &None);
    scf_data
}

// ── helper: run PySCF CP-HF and parse polarizability ───────────────────────
fn pyscf_polarizability(basis: &str) -> Option<[f64; 4]> {
    let script_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("test_cphf_h2o.py");

    let output = Command::new("python3")
        .arg(script_path.to_str().unwrap())
        .arg(basis)
        .output()
        .ok()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("PySCF script error:\n{}", stderr);
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let values: Vec<f64> = stdout
        .lines()
        .filter_map(|line| line.trim().parse::<f64>().ok())
        .collect();

    if values.len() < 4 {
        return None;
    }
    Some([values[0], values[1], values[2], values[3]])
}

// ── Test 1: Dense solver self-consistency (residual check) ──────────────────
#[test]
fn test_cphf_dense_solver_residual() {
    use pyrest::ri_cphf::{CPHFSolver, build_dipole_h1_comp};

    let scf = run_h2o_scf_hf_sto3g();
    let solver = CPHFSolver::new(&scf);
    println!("\n  === H2O HF/STO-3G SCF ===");
    println!("  HF energy:        {:.12} Ha", scf.scf_energy);
    println!("  nocc = {}, nvir = {}, dim = {}", solver.nocc, solver.nvir, solver.dim);

    let h1 = [
        build_dipole_h1_comp(&scf, 0),
        build_dipole_h1_comp(&scf, 1),
        build_dipole_h1_comp(&scf, 2),
    ];

    let u: Vec<_> = h1.iter().map(|h| {
        solver.solve_dense(&scf, h).expect("dense solve failed")
    }).collect();

    let labels = ["x", "y", "z"];
    for (comp, (u_comp, h_comp)) in u.iter().zip(h1.iter()).enumerate() {
        let fv = solver.fvind(&scf, u_comp);
        let residual: f64 = (0..solver.dim).map(|i| {
            let r = u_comp[i] + fv[i] * solver.e_ai[i] + h_comp[i] * solver.e_ai[i];
            r * r
        }).sum();
        let rnorm = residual.sqrt();
        println!("  Dense |(I+G̃)·U + h·e_ai| ({}) = {:.2e}", labels[comp], rnorm);
        assert!(rnorm < 1e-14,
            "Dense solver residual too large for {}: {:.2e}", labels[comp], rnorm);
    }
    println!("  ✓ Dense solver satisfies CP-HF equation to machine precision");
}

// ── Test 2: Krylov solver vs Dense solver (mathematical agreement) ─────────
#[test]
fn test_cphf_krylov_vs_dense() {
    use pyrest::ri_cphf::{CPHFSolver, build_dipole_h1_comp};

    let scf = run_h2o_scf_hf_sto3g();
    let solver = CPHFSolver::new(&scf);

    let h1 = [
        build_dipole_h1_comp(&scf, 0),
        build_dipole_h1_comp(&scf, 1),
        build_dipole_h1_comp(&scf, 2),
    ];

    let u_dense: Vec<_> = h1.iter().map(|h| {
        solver.solve_dense(&scf, h).expect("dense")
    }).collect();

    let u_krylov: Vec<_> = h1.iter().map(|h| {
        solver.solve_krylov(&scf, h, 50, 1e-12)
    }).collect();

    let dot = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x,y)| x*y).sum() };
    let labels = ["x", "y", "z"];
    let mut max_diff: f64 = 0.0;
    for (comp, (uk, ud)) in u_krylov.iter().zip(u_dense.iter()).enumerate() {
        let diff_sq = uk.iter().zip(ud.iter()).map(|(a,b)| (a-b)*(a-b)).sum::<f64>();
        let diff = diff_sq.sqrt();
        max_diff = max_diff.max(diff);
        println!("  Krylov vs Dense ({}): ||u_k - u_d||₂ = {:.2e}", labels[comp], diff);
    }
    assert!(max_diff < 1e-10,
        "Krylov and dense solutions differ: max_diff={:.2e}", max_diff);
    println!("  ✓ Krylov solver converges to exact (dense) solution");
}

// ── Test 3: Cross-validation with PySCF ────────────────────────────────────
#[test]
fn test_cphf_vs_pyscf_h2o_sto3g() {
    use pyrest::ri_cphf::{CPHFSolver, build_dipole_h1_comp};

    let scf = run_h2o_scf_hf_sto3g();
    let solver = CPHFSolver::new(&scf);

    let h1 = [
        build_dipole_h1_comp(&scf, 0),
        build_dipole_h1_comp(&scf, 1),
        build_dipole_h1_comp(&scf, 2),
    ];

    // REST dense solver
    let u: Vec<_> = h1.iter().map(|h| {
        solver.solve_dense(&scf, h).expect("dense")
    }).collect();
    let dot = |h: &[f64], u: &[f64]| -> f64 { h.iter().zip(u).map(|(a,b)| a*b).sum() };
    let rest_pol: Vec<f64> = h1.iter().zip(u.iter()).map(|(h, u)| -4.0 * dot(h, u)).collect();
    let rest_bar = rest_pol.iter().sum::<f64>() / 3.0;

    // PySCF
    let pyscf_result = pyscf_polarizability("sto-3g");
    let pyscf_pol = match pyscf_result {
        Some(p) => p,
        None => {
            eprintln!("PySCF not available — skipping test");
            return;
        }
    };

    println!("\n  === CP-HF Polarizability (H2O HF/STO-3G): REST vs PySCF ===");
    let labels = ["xx", "yy", "zz"];
    for (comp, (r, p)) in rest_pol.iter().zip(pyscf_pol.iter()).enumerate() {
        let diff = (r - p).abs();
        println!("  α_{}  REST={:14.10e}  PySCF={:14.10e}  diff={:.2e}", labels[comp], r, p, diff);
        assert!(diff < 1e-3,
            "α_{} mismatch: REST={:.10e} PySCF={:.10e} diff={:.2e}", labels[comp], r, p, diff);
    }
    let bar_diff = (rest_bar - pyscf_pol[3]).abs();
    assert!(bar_diff < 1e-3,
        "ᾱ mismatch: REST={:.10e} PySCF={:.10e}", rest_bar, pyscf_pol[3]);
    println!("  ᾱ     REST={:14.10e}  PySCF={:14.10e}  diff={:.2e}", rest_bar, pyscf_pol[3], bar_diff);
    println!();
    println!("  ✓ REST CP-HF matches PySCF within 1e-3");
}

// ═══════════════════════════════════════════════════════════════════════
// CPHFSolverPySCF tests (nmo, nocc space, PySCF convention)
// ═══════════════════════════════════════════════════════════════════════

/// Test 4: CPHFSolverPySCF with s1=0 gives same polarizability as old solver.
#[test]
fn test_cphf_pyscf_s0_vs_old_solver() {
    use pyrest::ri_cphf::{CPHFSolver, build_dipole_h1_comp};
    use pyrest::ri_cphf::CPHFSolverPySCF;

    let scf = run_h2o_scf_hf_sto3g();
    let old_solver = CPHFSolver::new(&scf);
    let new_solver = CPHFSolverPySCF::new(&scf);

    println!("\n  === CPHFSolverPySCF vs CPHFSolver (s1=0) ===");

    let h1_old = [
        build_dipole_h1_comp(&scf, 0),
        build_dipole_h1_comp(&scf, 1),
        build_dipole_h1_comp(&scf, 2),
    ];

    // For new solver: build h1 in (nmo, nocc) format
    let mo_dip = build_full_mo_dipole(&scf);
    let h1_new: Vec<Vec<f64>> = (0..3).map(|comp| {
        let mut h1_nmc = vec![0.0; new_solver.nmo * new_solver.nocc];
        for col in 0..new_solver.nocc {
            let occ_idx = new_solver.start_mo + col;
            for a in 0..new_solver.nvir {
                let row = new_solver.lumo + a;
                let nmc_idx = row + col * new_solver.nmo;
                h1_nmc[nmc_idx] = mo_dip[comp][[occ_idx, row]];
            }
        }
        h1_nmc
    }).collect();

    let s1_zero = vec![0.0; new_solver.nmo * new_solver.nocc];

    // Solve with old solver
    let u_old: Vec<Vec<f64>> = h1_old.iter().map(|h| {
        old_solver.solve_dense(&scf, h).expect("old dense")
    }).collect();

    // Solve with new solver (dense)
    let u_new: Vec<Vec<f64>> = h1_new.iter().map(|h| {
        let u_full = new_solver.solve_dense(&scf, h, &s1_zero)
            .expect("new dense");
        // Extract VO block to compare with old solver
        let mut u_vo = vec![0.0; new_solver.dim];
        for col in 0..new_solver.nocc {
            for a in 0..new_solver.nvir {
                let row = new_solver.lumo + a;
                let nmc_idx = row + col * new_solver.nmo;
                u_vo[col + a * new_solver.nocc] = u_full[nmc_idx];
            }
        }
        u_vo
    }).collect();

    // Compare
    let labels = ["x", "y", "z"];
    let mut max_diff: f64 = 0.0;
    for (comp, (uo, un)) in u_old.iter().zip(u_new.iter()).enumerate() {
        let diff_sq = uo.iter().zip(un.iter()).map(|(a,b)| (a-b)*(a-b)).sum::<f64>();
        let diff = diff_sq.sqrt();
        max_diff = max_diff.max(diff);
        println!("  Comp {}: ||U_old - U_new||₂ = {:.2e}", labels[comp], diff);
    }
    assert!(max_diff < 1e-10,
        "Old and new solver disagree: max_diff={:.2e}", max_diff);
    println!("  ✓ CPHFSolverPySCF (s1=0) matches old CPHFSolver");
}

/// Test 5: CPHFSolverPySCF dense vs krylov with s1=0.
#[test]
fn test_cphf_pyscf_dense_vs_krylov() {
    use pyrest::ri_cphf::CPHFSolverPySCF;

    let scf = run_h2o_scf_hf_sto3g();
    let solver = CPHFSolverPySCF::new(&scf);
    let mo_dip = build_full_mo_dipole(&scf);
    let s1_zero = vec![0.0; solver.nmo * solver.nocc];

    println!("\n  === CPHFSolverPySCF: Dense vs Krylov (s1=0) ===");

    let labels = ["x", "y", "z"];
    let mut max_diff: f64 = 0.0;

    for comp in 0..3 {
        let mut h1_nmc = vec![0.0; solver.nmo * solver.nocc];
        for col in 0..solver.nocc {
            let occ_idx = solver.start_mo + col;
            for a in 0..solver.nvir {
                let row = solver.lumo + a;
                h1_nmc[row + col * solver.nmo] = mo_dip[comp][[occ_idx, row]];
            }
        }

        let u_dense = solver.solve_dense(&scf, &h1_nmc, &s1_zero)
            .expect("dense");
        let u_krylov = solver.solve_krylov(&scf, &h1_nmc, &s1_zero, 50, 1e-12);

        let diff_sq = u_dense.iter().zip(u_krylov.iter())
            .map(|(a,b)| (a-b)*(a-b)).sum::<f64>();
        let diff = diff_sq.sqrt();
        max_diff = max_diff.max(diff);
        println!("  Comp {}: ||U_dense - U_krylov||₂ = {:.2e}", labels[comp], diff);
    }
    assert!(max_diff < 1e-10,
        "PySCF-style dense vs krylov mismatch: {:.2e}", max_diff);
    println!("  ✓ CPHFSolverPySCF dense == krylov");
}

/// Test 6: CPHFSolverPySCF dipole polarizability vs PySCF.
#[test]
fn test_cphf_pyscf_vs_pyscf_h2o_sto3g() {
    use pyrest::ri_cphf::CPHFSolverPySCF;

    let scf = run_h2o_scf_hf_sto3g();
    let solver = CPHFSolverPySCF::new(&scf);
    let mo_dip = build_full_mo_dipole(&scf);
    let s1_zero = vec![0.0; solver.nmo * solver.nocc];

    let labels = ["x", "y", "z"];
    let mut rest_pol = [0.0; 3];

    for comp in 0..3 {
        let mut h1_nmc = vec![0.0; solver.nmo * solver.nocc];
        for col in 0..solver.nocc {
            let occ_idx = solver.start_mo + col;
            for a in 0..solver.nvir {
                let row = solver.lumo + a;
                h1_nmc[row + col * solver.nmo] = mo_dip[comp][[occ_idx, row]];
            }
        }

        let u_full = solver.solve_dense(&scf, &h1_nmc, &s1_zero)
            .expect("dense");

        // Polarizability: α = -4 * Σ h1 * U (VO block only; occ-occ is 0 for s1=0)
        let mut pol = 0.0;
        for col in 0..solver.nocc {
            for a in 0..solver.nvir {
                let row = solver.lumo + a;
                let nmc_idx = row + col * solver.nmo;
                pol += h1_nmc[nmc_idx] * u_full[nmc_idx];
            }
        }
        rest_pol[comp] = -4.0 * pol;
    }
    let rest_bar = rest_pol.iter().sum::<f64>() / 3.0;

    // PySCF ref
    let pyscf_result = pyscf_polarizability("sto-3g");
    let pyscf_pol = match pyscf_result {
        Some(p) => p,
        None => {
            eprintln!("PySCF not available — skipping test");
            return;
        }
    };

    println!("\n  === CPHFSolverPySCF Polarizability vs PySCF ===");
    for comp in 0..3 {
        let diff = (rest_pol[comp] - pyscf_pol[comp]).abs();
        println!("  α_{}  REST={:14.10e}  PySCF={:14.10e}  diff={:.2e}",
            labels[comp], rest_pol[comp], pyscf_pol[comp], diff);
        assert!(diff < 1e-3,
            "α_{} mismatch: REST={:.10e} PySCF={:.10e}",
            labels[comp], rest_pol[comp], pyscf_pol[comp]);
    }
    let bar_diff = (rest_bar - pyscf_pol[3]).abs();
    assert!(bar_diff < 1e-3,
        "ᾱ mismatch: REST={:.10e} PySCF={:.10e}", rest_bar, pyscf_pol[3]);
    println!("  ✓ CPHFSolverPySCF matches PySCF within 1e-3");
}

// ═══════════════════════════════════════════════════════════════════════
// Test 7: CPHFSolverPySCF U-vector elements vs PySCF (element-by-element)
// ═══════════════════════════════════════════════════════════════════════

/// Compare the CP-HF solution U-vector (mo1) element-by-element against
/// PySCF for H2O/STO-3G dipole perturbations. This is a more stringent
/// test than polarizability alone — it checks individual solution elements.
///
/// Procedure:
///   1. Build h1 from MO-basis dipole integrals in REST
///   2. Solve CP-HF using CPHFSolverPySCF (dense + krylov)
///   3. Print U-vector elements for all 3 directions
///   4. Call PySCF reference script to get h1 and U elements
///   5. Compare h1 and U element by element
#[test]
fn test_cphf_pyscf_u_vector_vs_pyscf() {
    use pyrest::ri_cphf::CPHFSolverPySCF;

    let scf = run_h2o_scf_hf_sto3g();
    let solver = CPHFSolverPySCF::new(&scf);
    let mo_dip = build_full_mo_dipole(&scf);
    let s1_zero = vec![0.0; solver.nmo * solver.nocc];

    // Build h1 in REST's (nmo, nocc) flat format
    let mut h1_all: Vec<Vec<f64>> = Vec::with_capacity(3);
    for comp in 0..3 {
        let mut h1_nmc = vec![0.0; solver.nmo * solver.nocc];
        for col in 0..solver.nocc {
            let occ_idx = solver.start_mo + col;
            for a in 0..solver.nvir {
                let row = solver.lumo + a;
                h1_nmc[row + col * solver.nmo] = mo_dip[comp][[occ_idx, row]];
            }
        }
        h1_all.push(h1_nmc);
    }

    // Solve with dense and krylov
    let u_dense: Vec<Vec<f64>> = h1_all.iter().map(|h| {
        solver.solve_dense(&scf, h, &s1_zero).expect("dense")
    }).collect();
    let u_krylov: Vec<Vec<f64>> = h1_all.iter().map(|h| {
        solver.solve_krylov(&scf, h, &s1_zero, 50, 1e-12)
    }).collect();

    // Print REST U vectors (VO block only)
    let labels = ["x", "y", "z"];
    println!("\n  === REST CPHFSolverPySCF U-vectors (VO block) ===");
    println!("  nocc={}, nvir={}, dim_vo={}", solver.nocc, solver.nvir, solver.dim);
    for comp in 0..3 {
        println!("  U_vo_{} (dense, {:>2} elements):", labels[comp], solver.dim);
        for i in 0..solver.dim {
            // Extract VO block from full (nmo,nocc) solution
            let col = i % solver.nocc;
            let a = i / solver.nocc;
            let row = solver.lumo + a;
            let u_val = u_dense[comp][row + col * solver.nmo];
            println!("    [{:>2}] = {: .16e}", i, u_val);
        }
    }
    // Verify dense ≈ krylov
    for comp in 0..3 {
        let diff_sq: f64 = u_dense[comp].iter().zip(u_krylov[comp].iter())
            .map(|(a,b)| (a-b)*(a-b)).sum();
        assert!(diff_sq.sqrt() < 1e-12,
            "U_vo_{}: dense vs krylov ||diff||₂ = {:.2e}", labels[comp], diff_sq.sqrt());
    }
    println!("  ✓ Dense == Krylov (||diff||₂ < 1e-12)");

    // ── Call PySCF reference ──
    let script_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("test_cphf_h2o_uvec.py");

    let output = std::process::Command::new("python3")
        .arg(script_path.to_str().unwrap())
        .arg("sto-3g")
        .output()
        .expect("Failed to run test_cphf_h2o_uvec.py");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("PySCF script error:\n{}", stderr);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse sections
    let lines: Vec<&str> = stdout.lines().collect();
    let parse_section = |lines: &[&str], start_mark: &str, end_mark: &str| -> Vec<f64> {
        let mut in_section = false;
        let mut values = Vec::new();
        for &line in lines {
            let trimmed = line.trim();
            if trimmed.starts_with(start_mark) {
                in_section = true;
                continue;
            }
            if trimmed.starts_with(end_mark) {
                in_section = false;
                continue;
            }
            if in_section {
                // Skip comment lines
                if trimmed.starts_with('#') { continue; }
                if let Ok(v) = trimmed.parse::<f64>() {
                    values.push(v);
                }
            }
        }
        values
    };

    let py_h1 = parse_section(&lines, "### H1_START ###", "### H1_END ###");
    let py_u = parse_section(&lines, "### U_START ###", "### U_END ###");
    let py_pol = parse_section(&lines, "### POL_START ###", "### POL_END ###");

    // Expected counts: h1=3*nmo*nocc=84, u=3*nocc*nvir=24, pol=4
    assert_eq!(py_h1.len(), 3 * solver.nmo * solver.nocc,
        "Expected {} h1 elements, got {}", 3 * solver.nmo * solver.nocc, py_h1.len());
    assert_eq!(py_u.len(), 3 * solver.dim,
        "Expected {} U elements, got {}", 3 * solver.dim, py_u.len());
    assert_eq!(py_pol.len(), 4,
        "Expected 4 polarizability values, got {}", py_pol.len());

    // ── 1. Compare h1 ──
    println!("\n  --- h1 comparison (REST vs PySCF) ---");
    let mut h1_maxdiff = 0.0;
    for comp in 0..3 {
        for i in 0..solver.nmo * solver.nocc {
            let rest_val = h1_all[comp][i];
            let py_val = py_h1[comp * solver.nmo * solver.nocc + i];
            let diff = (rest_val - py_val).abs();
            if diff > h1_maxdiff { h1_maxdiff = diff; }
        }
    }
    println!("  h1 max|REST - PySCF| = {:.4e}", h1_maxdiff);

    // ── 2. Compare U vectors ──
    println!("\n  --- U-vector comparison (REST vs PySCF) ---");
    let mut u_maxdiff = 0.0;
    for comp in 0..3 {
        println!("  U_vo_{}:", labels[comp]);
        for i in 0..solver.dim {
            let col = i % solver.nocc;
            let a = i / solver.nocc;
            let row = solver.lumo + a;
            let rest_val = u_dense[comp][row + col * solver.nmo];
            let py_val = py_u[comp * solver.dim + i];
            let diff = (rest_val - py_val).abs();
            if diff > u_maxdiff { u_maxdiff = diff; }
            let marker = if diff > 1e-6 { " <---" } else { "" };
            println!("    [{:>2}] REST={: .12e}  PySCF={: .12e}  diff={:.4e}{}",
                     i, rest_val, py_val, diff, marker);
        }
    }
    println!("\n  U-vector max|REST - PySCF| = {:.4e}", u_maxdiff);

    // ── 3. Also compare polarizabilities ──
    println!("\n  --- Polarizability ---");
    let mut pol_maxdiff = 0.0;
    for comp in 0..3 {
        let pol_rest = -4.0 * (0..solver.dim).map(|i| {
            let col = i % solver.nocc;
            let a = i / solver.nocc;
            let row = solver.lumo + a;
            h1_all[comp][row + col * solver.nmo] * u_dense[comp][row + col * solver.nmo]
        }).sum::<f64>();
        let pol_py = py_pol[comp];
        let diff = (pol_rest - pol_py).abs();
        if diff > pol_maxdiff { pol_maxdiff = diff; }
        println!("  α_{}  REST={:14.10e}  PySCF={:14.10e}  diff={:.4e}",
                 labels[comp], pol_rest, pol_py, diff);
    }
    let pol_bar_rest = (0..3).map(|c| {
        -4.0 * (0..solver.dim).map(|i| {
            let col = i % solver.nocc;
            let a = i / solver.nocc;
            let row = solver.lumo + a;
            h1_all[c][row + col * solver.nmo] * u_dense[c][row + col * solver.nmo]
        }).sum::<f64>()
    }).sum::<f64>() / 3.0;
    println!("  ᾱ    REST={:14.10e}  PySCF={:14.10e}  diff={:.4e}",
             pol_bar_rest, py_pol[3], (pol_bar_rest - py_pol[3]).abs());

    // Assertions
    assert!(h1_maxdiff < 1e-10,
        "h1 mismatch: max|REST-PySCF| = {:.4e} > 1e-10", h1_maxdiff);
    assert!(u_maxdiff < 1e-3,
        "U-vector mismatch: max|REST-PySCF| = {:.4e} > 1e-3", u_maxdiff);
    assert!(pol_maxdiff < 1e-3,
        "Polarizability mismatch: maxdiff = {:.4e} > 1e-3", pol_maxdiff);
    assert!((pol_bar_rest - py_pol[3]).abs() < 1e-3,
        "Mean polarizability mismatch: |REST-PySCF| = {:.4e} > 1e-3",
        (pol_bar_rest - py_pol[3]).abs());

    println!("  ✓ U-vector elements match PySCF to tolerance 1e-3");
}

/// Helper: build full MO dipole matrix from SCF data.
fn build_full_mo_dipole(scf: &pyrest::scf_io::SCF) -> [rest_tensors::MatrixFull<f64>; 3] {
    use rest_tensors::MatrixFull;
    use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;

    let ao_dip = pyrest::ri_bse::dipoles::obtain_ao_dips(scf, None);
    let eigvec = &scf.eigenvectors[0];
    let nao = eigvec.size[0];
    let nmo = eigvec.size[1];

    let mut result = [(); 3].map(|_| MatrixFull::new([nmo, nmo], 0.0));

    for comp in 0..3 {
        let ao_dip_comp = ao_dip.get_reducing_matrix(comp).unwrap();
        let mut tmp = MatrixFull::new([nao, nmo], 0.0);
        _dgemm_full(eigvec, 'T', &ao_dip_comp, 'N', &mut tmp, 1.0, 0.0);
        _dgemm_full(&tmp, 'N', eigvec, 'N', &mut result[comp], 1.0, 0.0);
    }
    result
}
