use pyrest::molecule_io::Molecule;
use pyrest::scf_io::scf;
use std::env;
use std::path::{Path, PathBuf};

/// Locate a regression case in the sibling `rest_regression` repository.
///
/// The path is resolved from the crate root (`CARGO_MANIFEST_DIR`), so the test never depends on a
/// machine-specific location, and the case is searched one level down because the cases live in
/// `bench_pool/<category>/<name>`. `None` means "case not present in this checkout" and the caller
/// skips it, which also covers a standalone `rest` checkout without the sibling repository.
fn case_dir(system: &str) -> Option<PathBuf> {
    let pool = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../rest_regression/bench_pool");
    let direct = pool.join(system);
    if direct.is_dir() {
        return Some(direct);
    }
    for entry in std::fs::read_dir(&pool).ok()?.flatten() {
        let candidate = entry.path().join(system);
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    None
}

fn run_with_stats(dir: &Path, ao_cutoff: f64, blksize: usize, use_dm_only: bool) -> (f64, bool, String) {
    let saved = env::current_dir().unwrap();
    env::set_current_dir(dir).unwrap();

    let mut mol = Molecule::build("ctrl.in".to_string(), None).unwrap();
    mol.ctrl.ao_cutoff = ao_cutoff;
    mol.ctrl.non0tab_blksize = blksize;
    mol.ctrl.drop_dense_ao = false;
    mol.ctrl.num_threads = Some(1);
    mol.ctrl.print_level = 2;
    mol.ctrl.vxc_screen_threshold = 0.0;
    mol.ctrl.use_dm_only = use_dm_only;

    let result = scf(mol, &None).unwrap();
    let energy = result.scf_energy;

    let (sparse_active, stats) = if let Some(ref grids) = result.grids {
        if let Some(ref nt) = grids.non0tab {
            let ao_pct = nt.sparsity_ratio * 100.0;
            let has_comp = grids.ao_compressed.is_some();
            let has_aop_comp = grids.aop_compressed.is_some();
            let (dense_b, comp_b, _) = grids.memory_footprint();
            (has_comp,
             format!("non0tab=YES  ao-sparsity={:.1}% ({}/{})  aop-compressed={}  mem={:.1}MB→{:.1}MB",
                     ao_pct, nt.total_nonzero_ao, nt.total_elements,
                     has_aop_comp,
                     dense_b as f64/1e6, comp_b as f64/1e6))
        } else {
            (false, String::from("non0tab=NONE (ao too dense, sparse NOT triggered)"))
        }
    } else {
        (false, String::from("grids=None"))
    };

    env::set_current_dir(&saved).unwrap();
    (energy, sparse_active, stats)
}

#[test]
fn test_all_paths_with_sparsity() {
    // NH3_SVWN: at 1e-10 sparsity=91.2% (>90% threshold) → sparse skipped
    // Test with 1e-12 to force sparse activation
    for (sys, cutoff, blk, expect_sparse) in [
        ("H2x2_B3LYP",        1e-10, 128, true),
        ("H2Ox3_B3LYP",       1e-10, 128, true),
        ("H2_triplet_X3LYP",  1e-10, 128, true),
        ("NH3_SVWN",          1e-10, 128, false),   // 91.2% → skipped
        ("NH3_SVWN",          1e-12, 128, false),   // still >90% for this compact molecule
    ] {
        let Some(dir) = case_dir(sys) else {
            println!("skip {}: not found under rest_regression/bench_pool/<category>/", sys);
            continue;
        };
        println!("\n══════ {}  cutoff={:.0e}  blksize={} ══════", sys, cutoff, blk);

        let (e_d, _act_d, s_d) = run_with_stats(&dir, 0.0, 0, true);
        println!("  dense(dm_only) {:14.8}  | {}", e_d, s_d);

        let (e_s_dm, act_s_dm, s_s_dm) = run_with_stats(&dir, cutoff, blk, true);
        println!("  sparse(dm)    {:14.8}  Δ={:.2e}  | {}", e_s_dm, (e_s_dm-e_d).abs(), s_s_dm);
        assert_eq!(act_s_dm, expect_sparse, "sparse(dm) active mismatch: expected {} got {}", expect_sparse, act_s_dm);

        let (e_s_co, act_s_co, s_s_co) = run_with_stats(&dir, cutoff, blk, false);
        println!("  sparse(coeff) {:14.8}  Δ={:.2e}  | {}", e_s_co, (e_s_co-e_d).abs(), s_s_co);
        assert_eq!(act_s_co, expect_sparse, "sparse(coeff) active mismatch");

        let tol = 1e-6;
        assert!((e_s_dm - e_d).abs() < tol, "sparse(dm) energy differs");
        assert!((e_s_co - e_d).abs() < tol, "sparse(coeff) energy differs");
    }
}
