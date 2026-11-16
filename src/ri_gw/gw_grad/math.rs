/// Plain inner product of two flat buffers.
#[inline]
pub fn flat_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// `s_metric · q` for a `[naux, ncol]` right-hand side, applied in auxiliary-column blocks.
///
/// Working block-wise holds one `naux * block` temporary (a few MB) and writes
/// straight into the output, instead of materialising a second full-size buffer
/// for `qt = s_metric · q` (`naux * nmo^2`, hundreds of MB on production
/// systems).
fn metric_transform_blocked(s_metric: &[f64], q: &[f64], naux: usize, ncol: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; naux * ncol];
    let sm = to_mat(s_metric, naux, naux);
    let block = 512usize.min(ncol).max(1);
    let mut c0 = 0;
    while c0 < ncol {
        let c1 = (c0 + block).min(ncol);
        let nb = c1 - c0;
        let bm = to_mat(&q[c0 * naux..c1 * naux], naux, nb);
        let mut om = MatrixFull::new([naux, nb], 0.0);
        _dgemm_full(&sm, 'N', &bm, 'N', &mut om, 1.0, 0.0);
        out[c0 * naux..c1 * naux].copy_from_slice(&om.data);
        c0 = c1;
    }
    out
}

/// Symmetric eigendecomposition, ascending eigenvalues, column eigenvectors.
///
/// Uses LAPACK `dsyevd`; the third tuple element of `_dsyevd` is the matrix
/// dimension, not `info`, so the eigenvalues are validated explicitly.
fn eigh_symmetric(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mat = to_mat(a, n, n);
    let (vecs, vals, _dim) = _dsyevd(&mat, 'V');
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "gw_grad: eigensolver produced non-finite eigenvalues"
    );
    (vals, vecs.expect("gw_grad: dsyev vectors").data)
}

/// Symmetry-protected degenerate shells of the orbital energies `e`.
///
/// The SCF sorts its eigenvalues, so a shell is a maximal *contiguous* run of
/// energies agreeing within `tol`; `tol <= 0` puts every orbital in a shell of
/// its own.  Returns `(shells, shell_of)`.
pub fn degenerate_shells(e: &[f64], tol: f64) -> (Vec<Vec<usize>>, Vec<usize>) {
    let mut shells: Vec<Vec<usize>> = Vec::new();
    let mut shell_of = vec![0usize; e.len()];
    for p in 0..e.len() {
        let start_new = match shells.last() {
            Some(s) => tol <= 0.0 || (e[p] - e[*s.last().unwrap()]).abs() > tol,
            None => true,
        };
        if start_new {
            shells.push(vec![p]);
        } else {
            shells.last_mut().unwrap().push(p);
        }
        shell_of[p] = shells.len() - 1;
    }
    (shells, shell_of)
}
