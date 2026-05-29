//! Integration test: non0tab sparse-grid SCF correctness
//!
//! Tests sparse-grid parameters (ao_cutoff, non0tab_blksize, drop_dense_ao)
//! across representative systems:
//!
//! Small-molecule systems:
//!   NH3_SVWN  — LDA, singlet, cc-pVDZ (29 BF)
//!   NH3_SCAN  — mGGA, singlet, cc-pVDZ (29 BF)
//!   H2_triplet — Hybrid GGA, triplet UHF, cc-pVDZ (10 BF)
//!
//! H₂ linear-chain systems (B3LYP, cc-pVDZ):
//!   H2x2 — 2×H₂, 4 atoms, 20 BF, ~40K grids
//!
//! NOTE: Run with --test-threads=1 to avoid Rayon pool contention.

use pyrest::molecule_io::Molecule;
use pyrest::scf_io::scf;
use std::env;
use std::path::{Path, PathBuf};

const BENCH_POOL: &str = "/home/igor/Documents/Package-Pool/rest_workspace/rest_regression/bench_pool";

const NH3_SVWN: &str = "NH3_SVWN";
const NH3_SCAN: &str = "NH3_SCAN";
const H2_TRIPLET: &str = "H2_triplet_X3LYP";
const H2X2: &str = "H2x2_B3LYP";

const SELF_TOL: f64 = 1e-6;

fn bench_dir(name: &str) -> PathBuf {
    Path::new(BENCH_POOL).join(name)
}

fn run_scf_with_params(
    system_name: &str,
    ao_cutoff: f64,
    blksize: usize,
    drop_dense_ao: bool,
    num_threads: usize,
) -> anyhow::Result<f64> {
    let dir = bench_dir(system_name);
    let original_dir = env::current_dir().unwrap();
    env::set_current_dir(&dir).unwrap();

    let mut mol = Molecule::build("ctrl.in".to_string(), None)
        .expect(&format!("Failed to build Molecule for {system_name}"));

    mol.ctrl.ao_cutoff = ao_cutoff;
    mol.ctrl.non0tab_blksize = blksize;
    mol.ctrl.drop_dense_ao = drop_dense_ao;
    mol.ctrl.num_threads = Some(num_threads);
    mol.ctrl.print_level = 0;
    mol.ctrl.vxc_screen_threshold = 0.0;
    mol.ctrl.use_dm_only = true;

    let scf_result = scf(mol, &None)?;
    env::set_current_dir(original_dir).unwrap();
    Ok(scf_result.scf_energy)
}

fn dense_energy(system_name: &str) -> f64 {
    run_scf_with_params(system_name, 0.0, 0, false, 1).unwrap()
}

fn assert_within(energy: f64, reference: f64, tol: f64, sys: &str, desc: &str) {
    let diff = (energy - reference).abs();
    assert!(
        diff < tol,
        "[{}] {}: energy={:14.8} ref={:14.8} |Δ|={:.2e}",
        sys, desc, energy, reference, diff,
    );
}

// ═════════════════════════════════════════════════════════════════════
// 1. Dense baseline matches bench_pool reference energies
// ═════════════════════════════════════════════════════════════════════
#[test]
fn test_dense_baseline() {
    for (sys, expected) in [
        (NH3_SVWN, -56.0659625063),
        (NH3_SCAN, -56.5245527233),
        (H2_TRIPLET, -0.9973483683),
        (H2X2, -2.3466127877),
    ] {
        let energy = dense_energy(sys);
        assert_within(energy, expected, 1e-7, sys, "dense vs reference");
    }
}

// ═════════════════════════════════════════════════════════════════════
// 2. Sparse ao_cutoff=1e-10, auto blksize
// ═════════════════════════════════════════════════════════════════════
#[test]
fn test_sparse_cutoff_1e10() {
    for sys in &[NH3_SVWN, NH3_SCAN, H2X2] {
        let e_dense = dense_energy(sys);
        let e_sparse = run_scf_with_params(sys, 1e-10, 0, false, 1).unwrap();
        assert_within(e_sparse, e_dense, SELF_TOL, sys, "sparse ao_cutoff=1e-10");
    }
}

// ═════════════════════════════════════════════════════════════════════
// 3. Sparse ao_cutoff=1e-12 (tighter)
// ═════════════════════════════════════════════════════════════════════
#[test]
fn test_sparse_cutoff_1e12() {
    for sys in &[NH3_SVWN, NH3_SCAN, H2X2] {
        let e_dense = dense_energy(sys);
        let e_sparse = run_scf_with_params(sys, 1e-12, 0, false, 1).unwrap();
        assert_within(e_sparse, e_dense, SELF_TOL, sys, "sparse ao_cutoff=1e-12");
    }
}

// ═════════════════════════════════════════════════════════════════════
// 4. Sparse with drop_dense_ao=true (full memory reduction)
// ═════════════════════════════════════════════════════════════════════
#[test]
fn test_sparse_drop_dense_ao() {
    for sys in &[NH3_SVWN, NH3_SCAN, H2X2] {
        let e_dense = dense_energy(sys);
        let e_sparse = run_scf_with_params(sys, 1e-10, 128, true, 1).unwrap();
        assert_within(e_sparse, e_dense, SELF_TOL, sys, "sparse drop_dense_ao=true");
    }
}

// ═════════════════════════════════════════════════════════════════════
// 5. Explicit blksize values (batch-size robustness)
// ═════════════════════════════════════════════════════════════════════
#[test]
fn test_sparse_blksize_values() {
    for sys in &[NH3_SVWN, NH3_SCAN, H2X2] {
        let e_dense = dense_energy(sys);
        for blk in [64, 128, 256] {
            let e_sparse = run_scf_with_params(sys, 1e-10, blk, false, 1).unwrap();
            assert_within(e_sparse, e_dense, SELF_TOL, sys,
                &format!("sparse blksize={blk}"));
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// 6. Rayon multi-thread determinism
// ═════════════════════════════════════════════════════════════════════
#[test]
fn test_multithread_consistency() {
    for sys in &[NH3_SVWN, NH3_SCAN, H2X2] {
        let e1 = run_scf_with_params(sys, 1e-10, 128, false, 1).unwrap();
        let e4 = run_scf_with_params(sys, 1e-10, 128, false, 4).unwrap();
        let diff = (e1 - e4).abs();
        assert!(
            diff < 1e-8,
            "[{}] 1-thread={:14.8} 4-thread={:14.8} |Δ|={:.2e}",
            sys, e1, e4, diff,
        );
    }
}

// ═════════════════════════════════════════════════════════════════════
// 7. UHF open-shell (H2 triplet X3LYP) — sparse vs dense
// ═════════════════════════════════════════════════════════════════════
#[test]
fn test_uhf_sparse() {
    let e_dense = dense_energy(H2_TRIPLET);
    let e_sparse = run_scf_with_params(H2_TRIPLET, 1e-12, 128, false, 1).unwrap();
    assert_within(e_sparse, e_dense, SELF_TOL, H2_TRIPLET, "UHF sparse");
}
