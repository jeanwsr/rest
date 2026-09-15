//! Deterministic mathematical tests for ISDF and its linear algebra helpers.

#[cfg(test)]
mod tests {
    use crate::isdf::{
        cvt_classification, cvt_find_corresponding_point, cvt_isdf_v2, cvt_update_cmu, dgemm_ffi,
        index_of_min, prod_states_gw, tabulated_density, tabulated_density_batch,
    };
    use std::ffi::c_char;
    use tensors::MatrixFull;

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 1.0e-12 * expected.abs().max(1.0);
            assert!(
                actual.is_finite() && (actual - expected).abs() <= tolerance,
                "element {i}: expected {expected}, got {actual} (tolerance {tolerance})"
            );
        }
    }

    fn assert_matrix_close(actual: &MatrixFull<f64>, size: [usize; 2], expected: &[f64]) {
        assert_eq!(actual.size, size);
        assert_close(&actual.data, expected);
    }

    fn multiply(a: &MatrixFull<f64>, b: &MatrixFull<f64>) -> MatrixFull<f64> {
        assert_eq!(a.size[1], b.size[0]);
        let mut result = MatrixFull::new([a.size[0], b.size[1]], 0.0);
        result.lapack_dgemm(&mut a.clone(), &mut b.clone(), 'N', 'N', 1.0, 0.0);
        result
    }

    #[test]
    fn test_index_of_min() {
        assert_eq!(index_of_min(&mut vec![2.0, 1.0, -3.0, 6.1, 7.2]), 2);
        assert_eq!(index_of_min(&mut vec![2.0, -3.0, -3.0]), 1);
        assert_eq!(index_of_min(&mut vec![4.0]), 0);
    }

    #[test]
    fn test_prod_states_gw() {
        let phi = MatrixFull::from_vec([2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
        let psi = MatrixFull::from_vec([2, 2], vec![7.0, 9.0, 8.0, 10.0]).unwrap();
        // Column-major order: grid index, then psi state, then phi state.
        assert_matrix_close(
            &prod_states_gw(&phi, &psi),
            [2, 6],
            &[
                7.0, 36.0, 8.0, 40.0, 14.0, 45.0, 16.0, 50.0, 21.0, 54.0, 24.0, 60.0,
            ],
        );
    }

    #[test]
    #[should_panic(expected = "row dimensions of Phi and Psi do not match")]
    fn test_prod_states_gw_rejects_mismatched_grids() {
        let phi = MatrixFull::new([2, 1], 1.0);
        let psi = MatrixFull::new([3, 1], 1.0);
        prod_states_gw(&phi, &psi);
    }

    #[test]
    fn test_dgemm_transpose_and_scaling() {
        let mut a = MatrixFull::from_vec([3, 4], (1..=12).map(f64::from).collect()).unwrap();
        let mut b = MatrixFull::from_vec([3, 2], (1..=6).map(f64::from).collect()).unwrap();
        let mut c = MatrixFull::new([4, 2], 1.0);
        c.lapack_dgemm(&mut a, &mut b, 'T', 'N', 2.0, 0.5);
        assert_matrix_close(
            &c,
            [4, 2],
            &[28.5, 64.5, 100.5, 136.5, 64.5, 154.5, 244.5, 334.5],
        );
    }

    #[test]
    fn test_dgemm_ffi_transpose_and_scaling() {
        let a: Vec<f64> = (1..=12).map(f64::from).collect();
        let b: Vec<f64> = (1..=6).map(f64::from).collect();
        let mut c = vec![1.0; 8];
        // A is 3x4, B is 3x2, and C = 2 A^T B + 0.5 C is 4x2.
        dgemm_ffi(
            &a,
            &b,
            &mut c,
            &(b'T' as c_char),
            &(b'N' as c_char),
            &4,
            &2,
            &3,
            &2.0,
            &3,
            &3,
            &0.5,
            &4,
        );
        assert_close(&c, &[28.5, 64.5, 100.5, 136.5, 64.5, 154.5, 244.5, 334.5]);
    }

    #[test]
    fn test_lapack_dgesv_multiple_right_hand_sides() {
        let a = MatrixFull::from_vec([2, 2], vec![1.0, 3.0, 2.0, 5.0]).unwrap();
        // Two RHS columns must each contain two rows.
        let b = MatrixFull::from_vec([2, 2], vec![1.0, 2.0, 5.0, 13.0]).unwrap();
        let solution = a.clone().lapack_dgesv(&mut b.clone(), 2);
        assert_matrix_close(&solution, [2, 2], &[-1.0, 1.0, 1.0, 2.0]);
        assert_matrix_close(&multiply(&a, &solution), b.size, &b.data);
    }

    #[test]
    fn test_pinv_full_rank() {
        let a = MatrixFull::from_vec([2, 2], vec![1.0, 3.0, 2.0, 5.0]).unwrap();
        let inverse = a.clone().pinv(1.0e-12);
        assert_matrix_close(&inverse, [2, 2], &[-5.0, 3.0, 2.0, -1.0]);
        let identity = [1.0, 0.0, 0.0, 1.0];
        assert_matrix_close(&multiply(&a, &inverse), [2, 2], &identity);
        assert_matrix_close(&multiply(&inverse, &a), [2, 2], &identity);
    }

    #[test]
    fn test_pinv_rank_deficient() {
        let a = MatrixFull::from_vec([2, 2], vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        let inverse = a.clone().pinv(1.0e-12);
        assert_matrix_close(&inverse, [2, 2], &[0.04, 0.08, 0.08, 0.16]);
        // Moore-Penrose identities for a matrix with dependent columns.
        let aa_inv = multiply(&a, &inverse);
        let inv_a = multiply(&inverse, &a);
        assert_matrix_close(&multiply(&aa_inv, &a), a.size, &a.data);
        assert_matrix_close(&multiply(&inv_a, &inverse), inverse.size, &inverse.data);
        assert_matrix_close(&aa_inv.transpose(), aa_inv.size, &aa_inv.data);
        assert_matrix_close(&inv_a.transpose(), inv_a.size, &inv_a.data);
    }

    #[test]
    fn test_pinv_singular_value_cutoff() {
        let mut a =
            MatrixFull::from_vec([3, 3], vec![4.0, 0.0, 0.0, 0.0, 1.0e-8, 0.0, 0.0, 0.0, 0.0])
                .unwrap();
        assert_matrix_close(
            &a.pinv(1.0e-6),
            [3, 3],
            &[0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
    }

    #[test]
    fn test_cvt_classification() {
        let grids = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 4.0],
        ];
        let centers = vec![[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]];
        let (assignments, distances) = cvt_classification(&grids, &vec![1.0; 6], &centers);
        assert_eq!(assignments, vec![0, 0, 1, 1, 0, 0]);
        assert_close(&distances, &[0.0, 1.0, 1.0, 0.0, 2.0, 5.0]);
    }

    #[test]
    fn test_cvt_update_cmu_weighted_centroids_and_empty_cluster() {
        let grids = vec![
            [0.0, 0.0, 0.0],
            [2.0, 4.0, 6.0],
            [8.0, 10.0, 12.0],
            [10.0, 14.0, 18.0],
        ];
        let centers = vec![[0.0; 3], [10.0; 3], [20.0, 21.0, 22.0]];
        let updated = cvt_update_cmu(
            &grids,
            &vec![1.0, 3.0, 3.0, 1.0],
            &centers,
            &vec![0, 0, 1, 1],
        );
        assert_eq!(updated.len(), 3);
        assert_close(&updated[0], &[1.5, 3.0, 4.5]);
        assert_close(&updated[1], &[8.5, 11.0, 13.5]);
        assert_eq!(updated[2], centers[2]);
    }

    #[test]
    fn test_cvt_update_cmu_weight_threshold() {
        let grids = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let centers = vec![[10.0; 3], [20.0; 3]];
        let updated = cvt_update_cmu(&grids, &vec![1.0e-9, 1.0e-8], &centers, &vec![0, 1]);
        assert_eq!(updated.len(), 2);
        assert_eq!(updated[0], centers[0]);
        assert_close(&updated[1], &grids[1]);
    }

    #[test]
    fn test_cvt_find_corresponding_point() {
        let grids = vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ];
        let weights = vec![1.0; 4];
        let assignments = vec![0, 0, 1, 1];
        let mut centers = vec![[1.5, 0.0, 0.0], [8.5, 0.0, 0.0], [100.0, 0.0, 0.0]];
        // The empty third cluster falls back to the nearest point in the full grid.
        assert_eq!(
            cvt_find_corresponding_point(&grids, &weights, &centers, &assignments),
            vec![1, 2, 3]
        );
        centers[0] = [0.25, 0.0, 0.0];
        assert_eq!(
            cvt_find_corresponding_point(&grids, &weights, &centers, &assignments),
            vec![0, 2, 3]
        );
    }

    #[test]
    fn test_cvt_isdf_v2_weighted_grid_selection() {
        let grids = vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [200.0, 0.0, 0.0],
        ];
        let weights = vec![1.0, 3.0, 2.0, 0.0, 1.0e-9];
        // The retained points have centroid x = 7/3, nearest to the point x = 2.
        let (points, selected_weights) = cvt_isdf_v2(&grids, &weights, 1);
        assert_eq!(points, vec![[2.0, 0.0, 0.0]]);
        assert_close(&selected_weights, &[3.0]);
    }

    #[test]
    fn test_density_contraction_and_batch_boundaries() {
        let coordinates = vec![[0.0; 3]; 5];
        let ao = MatrixFull::from_vec(
            [2, 5],
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 2.0, -1.0, 1.0, 2.0, -1.0],
        )
        .unwrap();
        let dm = vec![
            MatrixFull::from_vec([2, 2], vec![2.0, 0.5, 0.5, 3.0]).unwrap(),
            MatrixFull::from_vec([2, 2], vec![1.0, -0.25, -0.25, 4.0]).unwrap(),
        ];
        // rho_s(r) = ao(r)^T dm_s ao(r), with separate spin columns.
        let expected = [2.0, 3.0, 16.0, 4.0, 9.0, 1.0, 4.0, 16.0, 5.5, 9.0];
        assert_matrix_close(
            &tabulated_density(&coordinates, &ao, &dm, 2),
            [5, 2],
            &expected,
        );
        for batch_size in [None, Some(1), Some(2), Some(5), Some(8)] {
            let density = tabulated_density_batch(&coordinates, &ao, &dm, 2, batch_size);
            assert_matrix_close(&density, [5, 2], &expected);
        }
    }
}
