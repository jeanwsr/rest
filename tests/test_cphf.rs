/// CP-HF Tests: cross-validate REST CP-HF against PySCF.
///
/// Runs SCF + CP-HF on H2O with HF/STO-3G, then calls PySCF via Python
/// to compute the same polarizability. Compares with tolerance 1e-6.

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
        let u_full = new_solver.solve_dense(&scf, None, h, &s1_zero)
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
    let mut max_l2_diff: f64 = 0.0;
    let mut max_abs_diff: f64 = 0.0;
    for (comp, (uo, un)) in u_old.iter().zip(u_new.iter()).enumerate() {
        let diffs: Vec<f64> = uo.iter().zip(un.iter()).map(|(a,b)| (a-b)).collect();
        let l2 = diffs.iter().map(|d| d*d).sum::<f64>().sqrt();
        let max_abs = diffs.iter().fold(0.0f64, |m, d| m.max(d.abs()));
        max_l2_diff = max_l2_diff.max(l2);
        max_abs_diff = max_abs_diff.max(max_abs);
        println!("  Comp {}: ||U_old - U_new||₂ = {:.2e}, max|diff| = {:.2e}",
                 labels[comp], l2, max_abs);
    }
    assert!(max_l2_diff < 1e-10,
        "Old and new solver disagree: max_l2_diff={:.2e}", max_l2_diff);
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
    println!("  nmo={}, nocc={}, nvir={}, dim={}, start_mo={}, lumo={}",
             solver.nmo, solver.nocc, solver.nvir, solver.dim,
             solver.start_mo, solver.lumo);

    // ── Diagnostic: compare VO×VO block of dense matrix vs Krylov matvec ──
    let nfrozen = solver.start_mo;
    println!("  nfrozen={}, dim_total={}", nfrozen, (nfrozen + solver.nvir) * solver.nocc);

    // Build dense matrix's VO×VO block column by column
    use rest_tensors::MatrixFull;
    let mut dense_vo_vo = MatrixFull::new([solver.dim, solver.dim], 0.0);
    let mut z_full = vec![0.0; solver.nmo * solver.nocc];
    for jb in 0..solver.dim {
        z_full.fill(0.0);
        let a_vir = jb / solver.nocc;
        let occ_j = jb % solver.nocc;
        let row = solver.lumo + a_vir;
        let col_jb = occ_j;
        z_full[row + col_jb * solver.nmo] = 1.0;
        let resp_full = solver.fvind_nmo_nocc(&scf, None, &z_full);
        for ia in 0..solver.dim {
            let ia_col = ia / solver.nocc;
            let ia_row = ia % solver.nocc;
            let r = solver.lumo + ia_col;
            dense_vo_vo[[ia, jb]] = resp_full[r + ia_row * solver.nmo] * solver.e_ai[ia];
        }
        dense_vo_vo[[jb, jb]] += 1.0; // identity
    }

    // Build Krylov matvec matrix for the VO subspace
    let mut krylov_mat = MatrixFull::new([solver.dim, solver.dim], 0.0);
    for jb in 0..solver.dim {
        let mut z_vo = vec![0.0; solver.dim];
        z_vo[jb] = 1.0;
        let mut z_full2 = vec![0.0; solver.nmo * solver.nocc];
        for i in 0..solver.dim {
            let a = i / solver.nocc;
            let occ = i % solver.nocc;
            let r = solver.lumo + a;
            z_full2[r + occ * solver.nmo] = z_vo[i];
        }
        let resp_full2 = solver.fvind_nmo_nocc(&scf, None, &z_full2);
        for i in 0..solver.dim {
            let col = i / solver.nocc;
            let row = i % solver.nocc;
            let r = solver.lumo + col;
            let c = row;
            krylov_mat[[i, jb]] = resp_full2[r + c * solver.nmo] * solver.e_ai[i];
        }
        krylov_mat[[jb, jb]] += 1.0; // (I + G̃)
    }

    // Compare matrices
    let mut mat_maxdiff = 0.0;
    for i in 0..solver.dim {
        for j in 0..solver.dim {
            let d = (dense_vo_vo[[i, j]] - krylov_mat[[i, j]]).abs();
            if d > mat_maxdiff { mat_maxdiff = d; }
        }
    }
    println!("  VO×VO matrix max|dense - krylov_matvec| = {:.2e}", mat_maxdiff);
    if mat_maxdiff > 1e-14 {
        // Print the matrices
        println!("  Dense VO×VO matrix:");
        for i in 0..solver.dim {
            print!("    ");
            for j in 0..solver.dim { print!(" {:12.6e}", dense_vo_vo[[i, j]]); }
            println!();
        }
        println!("  Krylov matvec matrix:");
        for i in 0..solver.dim {
            print!("    ");
            for j in 0..solver.dim { print!(" {:12.6e}", krylov_mat[[i, j]]); }
            println!();
        }
        println!("  Difference:");
        for i in 0..solver.dim {
            print!("    ");
            for j in 0..solver.dim { print!(" {:12.6e}", dense_vo_vo[[i, j]] - krylov_mat[[i, j]]); }
            println!();
        }
    }


    // ── Try with new_full (no frozen) to isolate the issue ──
    let solver_full = CPHFSolverPySCF::new_full(&scf);
    println!("\n  --- With new_full (start_mo=0): nmo={}, nocc={}, nvir={}, dim={} ---",
             solver_full.nmo, solver_full.nocc, solver_full.nvir, solver_full.dim);
    let s1_zero_full = vec![0.0; solver_full.nmo * solver_full.nocc];
    let mo_dip_full = build_full_mo_dipole(&scf);
    let mut max_diff_full: f64 = 0.0;
    for comp in 0..3 {
        let mut h1_nmc_full = vec![0.0; solver_full.nmo * solver_full.nocc];
        for col in 0..solver_full.nocc {
            let occ_idx = solver_full.start_mo + col;
            for a in 0..solver_full.nvir {
                let row = solver_full.lumo + a;
                h1_nmc_full[row + col * solver_full.nmo] = mo_dip_full[comp][[occ_idx, row]];
            }
        }
        let u_dense_f = solver_full.solve_dense(&scf, None, &h1_nmc_full, &s1_zero_full).expect("dense");
        let u_krylov_f = solver_full.solve_krylov(&scf, None, &h1_nmc_full, &s1_zero_full, 50, 1e-12);
        let diffs: Vec<f64> = u_dense_f.iter().zip(u_krylov_f.iter()).map(|(a,b)| a-b).collect();
        let l2 = diffs.iter().map(|d| d*d).sum::<f64>().sqrt();
        let max_abs = diffs.iter().fold(0.0f64, |m, d| m.max(d.abs()));
        max_diff_full = max_diff_full.max(l2);
        println!("  Comp {} (new_full): ||U_dense - U_krylov||₂ = {:.2e}, max|diff| = {:.2e}",
                 ["x", "y", "z"][comp], l2, max_abs);
    }
    println!("  new_full: max_diff = {:.2e}", max_diff_full);

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

        let u_dense = solver.solve_dense(&scf, None, &h1_nmc, &s1_zero)
            .expect("dense");
        let u_krylov = solver.solve_krylov(&scf, None, &h1_nmc, &s1_zero, 50, 1e-12);

        let diffs: Vec<f64> = u_dense.iter().zip(u_krylov.iter())
            .map(|(a,b)| a-b).collect();
        let l2 = diffs.iter().map(|d| d*d).sum::<f64>().sqrt();
        let max_abs = diffs.iter().fold(0.0f64, |m, d| m.max(d.abs()));
        max_diff = max_diff.max(l2);
        println!("  Comp {}: ||U_dense - U_krylov||₂ = {:.2e}, max|diff| = {:.2e}",
                 labels[comp], l2, max_abs);
    }
    assert!(max_diff < 1e-10,
        "PySCF-style dense vs krylov mismatch: {:.2e}", max_diff);
    println!("  ✓ CPHFSolverPySCF dense == krylov");
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
