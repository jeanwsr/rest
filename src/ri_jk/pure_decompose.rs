use super::decompose::*;
use super::prelude_dev::*;

/// Decompose 2c-2e ERI matrix using eigen decomposition.
///
/// Eigenvalues smaller than the threshold will be discarded, and the corresponding eigenvectors
/// will be ignored. The -1/2 power of the 2c-2e ERI matrix is returned.
pub fn decomp_j2c_eig(j2c: TsrView<f64>, threshold: f64) -> J2CDecompose {
    assert_eq!(j2c.device().default_order(), ColMajor);
    let (j2c_e, j2c_v) = rt::linalg::eigh(j2c.view()).into();
    // eigen values should be always in ascending order
    let n = j2c_e.less(threshold).sum();
    let e_pow = j2c_e.i(n..).pow(-0.5);
    let v = j2c_v.i((.., n..));
    let j2c_l_inv = (&v * e_pow.i((None, ..))) % v.t();
    J2CDecompose::Eig { j2c_l_inv, j2c_e: Some(j2c_e), j2c_v: Some(j2c_v) }
}

/// Decompose 2c-2e ERI matrix using Cholesky decomposition.
///
/// - If `threshold` is None, will directly perform Cholesky decomposition, and return error if it
///   fails.
/// - If `threshold` is Some, will first try Cholesky decomposition, if it fails or some diagonal
///   element is smaller than the threshold, will make matrix sufficiently positive-definite with
///   the given threshold and then perform Cholesky decomposition again.
pub fn decomp_j2c_cd(j2c: TsrView<f64>, threshold: Option<f64>, uplo: FlagUpLo) -> J2CDecompose {
    let j2c_l_result = rt::linalg::cholesky_f((j2c.view(), uplo));
    if let Some(threshold) = threshold {
        if let Ok(j2c_l) = j2c_l_result {
            // check diagonal elements if smaller than threshold
            if j2c_l.diagonal(None).min() > threshold {
                return J2CDecompose::Cd { j2c_l, uplo, j2c_l_inv: None };
            }
        }

        // usual cholesky fails, use eigen decomposition to force positive-definite and then cholesky
        // if even eigen fails, there's not much we can do
        let (j2c_e, j2c_v) = rt::linalg::eigh(j2c.view()).into();
        let j2c_e = j2c_e.maximum(threshold);
        let j2c = (&j2c_v * j2c_e.i((None, ..))) % j2c_v.t();
        // we will not check again the diagonal elements, they may be smaller than threshold but should be
        // sufficiently positive-definite for triangular solve (TRSM) work
        let j2c_l = rt::linalg::cholesky((j2c.view(), uplo));
        J2CDecompose::Cd { j2c_l, uplo, j2c_l_inv: None }
    } else {
        if let Ok(j2c_l) = j2c_l_result {
            return J2CDecompose::Cd { j2c_l, uplo, j2c_l_inv: None };
        } else {
            panic!("Cholesky decomposition failed for j2c matrix, and no threshold is provided for fallback.");
        }
    }
}

/// Decompose the 2c-2e ERI matrix using Cholesky or eigen decomposition based on the specified
/// threshold.
pub fn get_j2c_decomp(mol: &CInt, device: &DeviceBLAS, j2c_decomp_option: J2CDecompOption) -> J2CDecompose {
    let j2c = {
        let (out, shape) = mol.integrate("int2c2e", "s1", None).into();
        rt::asarray((out, shape.f(), device))
    };

    // first try cholesky decomposition, fall back by policy
    match j2c_decomp_option.policy {
        J2CDecompPolicy::Cd => decomp_j2c_cd(j2c.view(), j2c_decomp_option.threshold, j2c_decomp_option.uplo),
        J2CDecompPolicy::Eig => decomp_j2c_eig(j2c.view(), j2c_decomp_option.threshold.unwrap_or(J2C_THRESH)),
    }
}

/// Transform 3c-2e ERI (j3c), use solve/inv-matmul to decomposed 3c-2e ERI (cderi).
pub fn get_solved_j3c<T>(j3c: Tsr<T>, j2c_decomp: &J2CDecompose, flip_uplo: bool) -> Tsr<T>
where
    T: BlasFloat + FromPrimitive + 'static,
    DeviceBLAS: LapackDriverAPI<T>,
{
    match j2c_decomp {
        J2CDecompose::Cd { j2c_l, uplo, .. } => {
            // cast type anyway, this is not bottleneck
            let j2c_l = j2c_l.mapv(|x| T::from_f64(x).unwrap());
            // get j3c shape, and reshape to 2d for triangular solve;
            // note we assume j3c to be something similar to (x, x, naux).
            let j3c_shape = j3c.shape().clone();
            assert_eq!(
                *j3c_shape.last().unwrap(),
                j2c_l.shape()[0],
                "Last dimension of j3c should match the shape of j2c_l (both to be naux)."
            );
            let naux = j2c_l.shape()[0];
            let j3c_2d = j3c.into_shape((-1, naux)).into_reverse_axes(); // transposed to (naux, -1)]
            let j3c_2d = match (uplo, flip_uplo) {
                (Upper, false) => rt::linalg::solve_triangular((j2c_l.t(), j3c_2d, Lower)),
                (Lower, false) => rt::linalg::solve_triangular((j2c_l, j3c_2d, Lower)),
                (Upper, true) => rt::linalg::solve_triangular((j2c_l, j3c_2d, Upper)),
                (Lower, true) => rt::linalg::solve_triangular((j2c_l.t(), j3c_2d, Upper)),
            };
            j3c_2d.into_reverse_axes().into_shape(j3c_shape) // reverse back and reshape back
        },
        J2CDecompose::Eig { j2c_l_inv, .. } => {
            // we need to perform inplace matmul at this case
            // however, inplace matmul is not integrated at BLAS level, we need to batch it manually

            // batch size is currently fixed, not related to available menory at this time:
            // > max(1/25 remaining size, naux)
            // - we assume the API caller leaves at least 4% of memory for storing j3c;
            // - we assume size requirement of j2c copy is acceptable.

            // cast type anyway, this is not bottleneck
            let j2c_l_inv = j2c_l_inv.mapv(|x| T::from_f64(x).unwrap());
            // get j3c shape, and reshape to 2d for matmul;
            // note we assume j3c to be something similar to (x, x, naux).
            let j3c_shape = j3c.shape().clone();
            assert_eq!(
                *j3c_shape.last().unwrap(),
                j2c_l_inv.shape()[0],
                "Last dimension of j3c should match the shape of j2c_l_inv (both to be naux)."
            );
            let naux = j2c_l_inv.shape()[0];
            let device = j3c.device().clone();
            let mut j3c_2d = j3c.into_shape((-1, naux)); // not transposed
            let n = j3c_2d.shape()[0];
            // determine batch size
            let nbatch = ((n as f64 * 0.04).ceil() as usize).max(naux);
            let mut scratch_vec: Vec<T> = unsafe { uninitialized_vec(nbatch * naux).unwrap() };
            // perform batched inplace-matmul
            for start in (0..n).step_by(nbatch) {
                let end = (start + nbatch).min(n);
                let mut j3c_batch = j3c_2d.i_mut((start..end, ..));
                let mut scratch = rt::asarray((&mut scratch_vec, [end - start, naux].f(), &device));
                scratch.matmul_from(j3c_batch.view(), j2c_l_inv.view(), T::one(), T::zero());
                j3c_batch.assign(&scratch);
            }
            j3c_2d.into_shape(j3c_shape) // reshape back
        },
    }
}

#[test]
fn test_check_recoverable() {
    // this will just test if we can recover the original matrix by the decomposed factors.
    let mut device = DeviceBLAS::default();
    device.set_default_order(ColMajor);

    let vec = vec![10.0, 1.0, 2.0, 1.0, 20.0, 3.0, 2.0, 3.0, 30.0];
    let j2c = rt::asarray((vec, [3, 3].f(), &device));
    let decomp_cd = decomp_j2c_cd(j2c.view(), None, Upper);
    let decomp_eig = decomp_j2c_eig(j2c.view(), 1e-5);

    let j2c_recon_cd = match decomp_cd {
        J2CDecompose::Cd { j2c_l, uplo: Upper, .. } => j2c_l.t() % j2c_l.view(),
        _ => panic!("Expected Cholesky decomposition"),
    };
    assert!(rt::allclose(j2c_recon_cd.view(), j2c.view(), None));

    let j2c_recon_eig = match decomp_eig {
        J2CDecompose::Eig { j2c_l_inv, .. } => {
            let j2c_l = rt::linalg::inv(j2c_l_inv.view());
            j2c_l.t() % j2c_l.view()
        },
        _ => panic!("Expected Eigen decomposition"),
    };
    assert!(rt::allclose(j2c_recon_eig.view(), j2c.view(), None));
}
