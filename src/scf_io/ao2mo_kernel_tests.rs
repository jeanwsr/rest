//! Element-wise comparison and timing of the AO -> MO transformation kernels.
//!
//! Covers the three implementations used by `ao2mo_rayon`:
//!   * `ao2mo_rayon_v02`  : column-first, right-hand side is the full `[nao, nmo]`
//!   * `ao2mo_rayon_m1`   : column-first, right-hand side is the `column_dim` block
//!   * `ao2mo_rayon_m2`   : row-first, right-hand side is the `row_dim` block
//!
//! and the dispatch rule of `ao2mo_rayon` (narrower side first, `v02` fallback).
//!
//! Run with:
//!   cargo test --release --lib ao2mo_kernel_tests -- --nocapture

use super::*;
use std::time::Instant;

/// Deterministic pseudo-random fill, so the tests do not depend on `rand`.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self { Lcg(seed.wrapping_mul(6364136223846793005).wrapping_add(1)) }
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    }
}

/// Synthetic `(eigenvector, rimatr)` pair with realistic shapes.
fn synthetic(nao: usize, nmo: usize, naux: usize, seed: u64) -> (MatrixFull<f64>, MatrixFull<f64>) {
    let mut rng = Lcg::new(seed);
    let nbaspar = nao * (nao + 1) / 2;
    let mut eigvec = MatrixFull::new([nao, nmo], 0.0_f64);
    eigvec.data.iter_mut().for_each(|x| *x = rng.next_f64());
    let mut rimatr = MatrixFull::new([nbaspar, naux], 0.0_f64);
    rimatr.data.iter_mut().for_each(|x| *x = rng.next_f64());
    (eigvec, rimatr)
}

fn max_abs_diff(a: &RIFull<f64>, b: &RIFull<f64>) -> f64 {
    assert_eq!(a.data.len(), b.data.len(), "output sizes differ");
    a.data.iter().zip(b.data.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
}

fn max_abs(a: &RIFull<f64>) -> f64 {
    a.data.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
}

/// Shapes actually requested by the consumers, at a smaller size for speed.
struct Case {
    label: &'static str,
    row: std::ops::Range<usize>,
    col: std::ops::Range<usize>,
}

fn cases(nao: usize) -> Vec<Case> {
    let nocc = nao / 5;
    let nvir = nao - nocc;
    vec![
        Case { label: "PT2/RPA (vir, occ)", row: nocc..nao, col: 0..nocc },
        Case { label: "TDDFT/BSE OO (occ, occ)", row: 0..nocc, col: 0..nocc },
        Case { label: "TDDFT/BSE OV (occ, vir)", row: 0..nocc, col: nocc..nao },
        Case { label: "TDDFT/BSE VV (vir, vir)", row: nocc..nao, col: nocc..nao },
        Case { label: "FF (all, all)", row: 0..nao, col: 0..nao },
        Case { label: "GW per-row (1, all)", row: 0..1, col: 0..nao },
        Case { label: "GW block (nao/8, all)", row: 0..(nao / 8).max(1), col: 0..nao },
    ]
}

#[test]
fn kernels_agree_elementwise_and_dispatch_picks_expected_one() {
    let nao = 60;
    let naux = 32;
    let (eigvec, rimatr) = synthetic(nao, nao, naux, 20260918);

    for case in cases(nao) {
        let (v02, ..) = ao2mo_rayon_v02(&eigvec, &rimatr, case.row.clone(), case.col.clone()).unwrap();
        let (m1, ..) = ao2mo_rayon_m1(&eigvec, &rimatr, case.row.clone(), case.col.clone()).unwrap();
        let (m2, ..) = ao2mo_rayon_m2(&eigvec, &rimatr, case.row.clone(), case.col.clone()).unwrap();
        let (dispatch, ..) = ao2mo_rayon(&eigvec, &rimatr, case.row.clone(), case.col.clone()).unwrap();

        let scale = max_abs(&v02).max(1.0e-12);
        let d_m1 = max_abs_diff(&v02, &m1) / scale;
        let d_m2 = max_abs_diff(&v02, &m2) / scale;
        println!("{:<26} rel diff v02-m1 {:8.2e}  v02-m2 {:8.2e}", case.label, d_m1, d_m2);

        assert!(d_m1 < 1.0e-12, "{}: m1 differs from v02 by {}", case.label, d_m1);
        assert!(d_m2 < 1.0e-12, "{}: m2 differs from v02 by {}", case.label, d_m2);

        // the dispatch must reproduce exactly the kernel it selects
        let narrow_is_row = case.row.len() < case.col.len();
        let d_disp = if narrow_is_row {
            assert_eq!(max_abs_diff(&dispatch, &m2), 0.0, "{}: dispatch is not m2", case.label);
            max_abs_diff(&dispatch, &m2)
        } else {
            assert_eq!(max_abs_diff(&dispatch, &m1), 0.0, "{}: dispatch is not m1", case.label);
            max_abs_diff(&dispatch, &m1)
        };
        assert_eq!(d_disp, 0.0);
    }
}

/// `v02` must still be reachable through a non-contiguous (strided) eigenvector.
#[test]
fn dispatch_falls_back_to_v02_for_non_contiguous_input() {
    let nao = 40;
    let naux = 16;
    let (eigvec, rimatr) = synthetic(nao, nao, naux, 424242);

    // Strided view: same logical shape, leading dimension nao + 1.
    let mut padded = vec![0.0_f64; nao * (nao + 1)];
    for j in 0..nao {
        for i in 0..nao {
            padded[i + j * (nao + 1)] = eigvec.data[i + j * nao];
        }
    }
    let size = [nao, nao];
    let indicing = [1, nao + 1];
    let view = MatrixFullSlice { size: &size, indicing: &indicing, data: &padded };

    let row = 0..(nao / 5);
    let col = nao / 5..nao;
    let (expect_v02, ..) = ao2mo_rayon_v02(&view, &rimatr, row.clone(), col.clone()).unwrap();
    let (got, ..) = ao2mo_rayon(&view, &rimatr, row.clone(), col.clone()).unwrap();
    assert_eq!(max_abs_diff(&expect_v02, &got), 0.0, "non-contiguous input did not fall back to v02");
}

/// Wall time per kernel on one realistic shape set, for the migration report.
#[test]
fn kernel_timing_by_shape() {
    let nao = 240;
    let naux = 320;
    let (eigvec, rimatr) = synthetic(nao, nao, naux, 777);
    let nocc = nao / 5;
    let reps = 3;

    println!("{:<26} {:>12} {:>12} {:>12}", "shape", "v02 (s)", "m1 (s)", "m2 (s)");
    for case in cases(nao) {
        let mut best = [f64::MAX; 3];
        for _ in 0..reps {
            let t = Instant::now();
            let _ = ao2mo_rayon_v02(&eigvec, &rimatr, case.row.clone(), case.col.clone()).unwrap();
            best[0] = best[0].min(t.elapsed().as_secs_f64());

            let t = Instant::now();
            let _ = ao2mo_rayon_m1(&eigvec, &rimatr, case.row.clone(), case.col.clone()).unwrap();
            best[1] = best[1].min(t.elapsed().as_secs_f64());

            let t = Instant::now();
            let _ = ao2mo_rayon_m2(&eigvec, &rimatr, case.row.clone(), case.col.clone()).unwrap();
            best[2] = best[2].min(t.elapsed().as_secs_f64());
        }
        println!("{:<26} {:12.4} {:12.4} {:12.4}", case.label, best[0], best[1], best[2]);
    }
    // nocc is used by cases(); keep the compiler honest about the intent.
    assert!(nocc > 0);
}
