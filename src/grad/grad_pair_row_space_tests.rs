//! Unit tests of the row-space projections that the hybrid gradient applies to a storage-level
//! pruned in-core RI tensor.
//!
//! The pruned energy is the energy of a tensor whose dropped rows are zero at every geometry, so
//! its derivative contracts the stored rows only. These tests pin the three projections down on
//! synthetic data: the compacted rows, the masked full-length packed vector and the masked AO
//! pair matrix, plus the fail-closed guard of the consumers that are not wired yet.
//!
//! Run with:
//!   cargo test --release --lib grad_pair_row_space_tests -- --nocapture

use super::rhf::{get_grad_dao_k_int3c2e_ip1, RIMatrRowSpace};
use crate::ri_jk::PairMap;
use rstsr::prelude::*;

type Tsr<T> = Tensor<T, DeviceBLAS, IxD>;

/// `(mu, nu)` of every packed pair of an `nao`-function basis, in packed order.
fn packed_pairs(nao: usize) -> Vec<[usize; 2]> {
    let mut pairs = Vec::with_capacity(nao * (nao + 1) / 2);
    for mu in 0..nao {
        for nu in 0..=mu {
            pairs.push([mu, nu]);
        }
    }
    pairs
}

/// A map that stores every pair of an `nao`-function basis.
fn full_map(nao: usize) -> (PairMap, Vec<[usize; 2]>) {
    let pairs = packed_pairs(nao);
    let kept: Vec<usize> = (0..pairs.len()).collect();
    let bounds = vec![1.0_f64; pairs.len()];
    (PairMap::from_kept(pairs.len(), &kept, &pairs, &bounds), pairs)
}

/// A map that stores the pairs listed in `kept` of an `nao`-function basis.
fn partial_map(nao: usize, kept: &[usize]) -> (PairMap, Vec<[usize; 2]>) {
    let pairs = packed_pairs(nao);
    let bounds = vec![1.0_f64; pairs.len()];
    (PairMap::from_kept(pairs.len(), kept, &pairs, &bounds), pairs)
}

fn arange(n: usize, shift: f64) -> Tsr<f64> {
    let device = DeviceBLAS::default();
    let data = (0..n).map(|i| shift + i as f64).collect::<Vec<f64>>();
    rt::asarray((data, [n], &device))
}

#[test]
fn kept_indices_lists_the_stored_rows_in_ascending_order() {
    let pairs = packed_pairs(5);
    // keep four pairs, deliberately with a gap, in ascending order
    let kept = vec![0_usize, 3, 4, pairs.len() - 1];
    let bounds = vec![1.0_f64; pairs.len()];
    let map = PairMap::from_kept(pairs.len(), &kept, &pairs, &bounds);

    assert_eq!(map.len(), kept.len());
    assert_eq!(map.kept_indices(), kept);
    assert_eq!(map.num_baspar_full(), pairs.len());
    for (row, &full) in kept.iter().enumerate() {
        assert_eq!(map.row_of_full(full), Some(row));
    }
    assert_eq!(map.row_of_full(1), None);
}

#[test]
fn full_row_space_leaves_every_quantity_alone() {
    let nao = 6;
    let (map, _) = full_map(nao);
    let device = DeviceBLAS::default();
    let rows = RIMatrRowSpace::new(nao, &map, &device);
    assert_eq!(rows.len(), nao * (nao + 1) / 2);

    let packed = arange(nao * (nao + 1) / 2, 1.0);
    let projected = rows.project_packed(&packed);
    assert_eq!(projected.shape().to_vec(), vec![packed.shape()[0]]);
    for i in 0..packed.shape()[0] {
        assert_eq!(projected[[i]], packed[[i]]);
    }

    let masked = rows.mask_packed(&packed);
    for i in 0..packed.shape()[0] {
        assert_eq!(masked[[i]], packed[[i]]);
    }

    let ao = arange(nao * nao, 1.0).into_shape([nao, nao]);
    let masked_ao = rows.mask_ao(ao.view());
    for i in 0..nao {
        for j in 0..nao {
            assert_eq!(masked_ao[[i, j]], ao[[i, j]]);
        }
    }
}

#[test]
fn dropped_rows_are_the_only_thing_the_projections_remove() {
    let nao = 6;
    let pairs = packed_pairs(nao);
    // drop every pair that contains function 4
    let kept = (0..pairs.len())
        .filter(|&p| pairs[p][0] != 4 && pairs[p][1] != 4)
        .collect::<Vec<usize>>();
    let dropped = (0..pairs.len()).filter(|p| !kept.contains(p)).collect::<Vec<usize>>();
    assert!(!dropped.is_empty() && dropped.len() < pairs.len());
    let (map, _) = partial_map(nao, &kept);
    let device = DeviceBLAS::default();
    let rows = RIMatrRowSpace::new(nao, &map, &device);

    // the compacted packed quantity carries the stored rows, in stored order
    let packed = arange(pairs.len(), 1.0);
    let projected = rows.project_packed(&packed);
    assert_eq!(projected.shape()[0], kept.len());
    for (row, &full) in kept.iter().enumerate() {
        assert_eq!(projected[[row]], packed[[full]]);
    }

    // the masked full-length quantity keeps the stored rows and zeroes the others
    let masked = rows.mask_packed(&packed);
    for p in 0..pairs.len() {
        let expected = if kept.contains(&p) { packed[[p]] } else { 0.0 };
        assert_eq!(masked[[p]], expected);
    }

    // the AO mask is symmetric and covers exactly the stored pairs
    let one = rt::ones(([nao, nao], &device));
    let masked_ao = rows.mask_ao(one.view());
    for p in 0..pairs.len() {
        let [mu, nu] = pairs[p];
        let expected = if kept.contains(&p) { 1.0 } else { 0.0 };
        assert_eq!(masked_ao[[mu, nu]], expected);
        assert_eq!(masked_ao[[nu, mu]], expected);
    }
}

#[test]
#[should_panic(expected = "still reads the in-core RI tensor")]
fn unwired_consumers_fail_closed_on_a_pruned_tensor() {
    let (map, _) = full_map(4);
    crate::ri_jk::require_unpruned_rimatr(&Some(map), "a test consumer");
}

#[test]
fn masked_k_ao_derivative_equals_the_restricted_sum() {
    // The K AO-derivative term sums ip1[mu, nu, p, t] * A[mu, nu, p] over the AO pairs. Under a
    // pruned tensor the sum runs over the stored pairs only, and the mask is how the helper says
    // so, so the masked result has to equal a plain Rust sum over the kept pairs.
    let (nao, naux) = (5, 3);
    let pairs = packed_pairs(nao);
    let kept = vec![0_usize, 2, 5, 9, 11, pairs.len() - 1];
    let (map, _) = partial_map(nao, &kept);
    let device = DeviceBLAS::default();
    let rows = RIMatrRowSpace::new(nao, &map, &device);

    let ip1_data = (0..nao * nao * naux * 3).map(|i| ((i % 17) as f64) - 8.0).collect::<Vec<f64>>();
    let ip1 = rt::asarray((ip1_data, [nao, nao, naux, 3], &device));
    let a_data = (0..nao * nao * naux).map(|i| ((i % 13) as f64) - 6.0).collect::<Vec<f64>>();
    let a = rt::asarray((a_data, [nao, nao, naux], &device));

    let got = get_grad_dao_k_int3c2e_ip1(ip1.view(), a.view(), Some(&rows.ao_mask));
    assert_eq!(got.shape().to_vec(), vec![nao, 3]);

    // brute-force the same sum on the masked pairs
    let keep_pair = |mu: usize, nu: usize| -> bool {
        kept.iter().any(|&p| pairs[p] == [mu, nu] || pairs[p] == [nu, mu])
    };
    for mu in 0..nao {
        for t in 0..3 {
            let mut expected = 0.0_f64;
            for nu in 0..nao {
                if !keep_pair(mu, nu) {
                    continue;
                }
                for p in 0..naux {
                    // read through the tensor interface: the test must not assume a layout
                    expected += -2.0 * ip1[[mu, nu, p, t]] * a[[mu, nu, p]];
                }
            }
            assert!(
                (got[[mu, t]] - expected).abs() < 1e-10,
                "mu = {}, t = {}: masked {} vs restricted {}",
                mu,
                t,
                got[[mu, t]],
                expected
            );
        }
    }
}
