/// Plain inner product of two flat buffers.
#[inline]
pub fn flat_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn to_mat(v: &[f64], r: usize, c: usize) -> MatrixFull<f64> {
    MatrixFull::from_vec([r, c], v.to_vec()).unwrap()
}

/// `A^T B` for flat col-major buffers `A [ar, ac]`, `B [br, bc]`, `ar == br`.
/// The contraction runs over the FIRST dimension (REST `_dgemm_full` 'T'
/// convention), the result is `[ac, bc]` col-major.
///
/// The operands are borrowed as **views** (`MatrixFullSlice` over the caller's slice) instead of
/// being wrapped in `MatrixFull` through `to_mat`, which copies its whole buffer: in the GW
/// gradient `A` is the raw three-centre integral tensor `[nao^2, naux]` (three calls per adjoint
/// build) and `B` is the metric-transformed `qt` in the low-rank build, so a copy costs
/// `nao^2 * naux` doubles (874 MB at naphthalene/def2-TZVP) of purely transient memory each time.
///
/// **This used to be a tensor-library matmul (`rt::asarray(..).t() % ..`), which is a parallel
/// design defect on a shared-memory node.**  `rstsr`'s `%` routes every product through
/// `with_num_threads(pool_size, ..)`, i.e. it writes the *process-global* OpenBLAS thread count
/// twice, under a mutex, from whichever thread issues the product -- including threads inside a
/// rayon parallel region, where the write is a cross-thread side effect on shared state.  On the
/// polyene/def2-TZVPP gradient card at 40 threads, one run issued **43,686** such writes
/// (30,784 of them from this function's call sites); neutralising them from outside the process
/// (`LD_PRELOAD`) dropped the card from 141 s to 106 s and the CPU from 1934% to 614%, with
/// bit-identical results.  `_dgemm_full` calls the same OpenBLAS `dgemm` directly and leaves the
/// thread policy to whoever owns it (`ctrl_io`, or the phase-local raise in `bse_grad.rs`).
fn gemm_nt(a: &[f64], ar: usize, ac: usize, b: &[f64], br: usize, bc: usize) -> Vec<f64> {
    debug_assert_eq!(ar, br, "gemm_nt: contraction dimension mismatch");
    let (sa, si_a, sb, si_b) = ([ar, ac], [1usize, ar.max(1)], [br, bc], [1usize, br.max(1)]);
    let am = MatrixFullSlice { size: &sa, indicing: &si_a, data: a };
    let bm = MatrixFullSlice { size: &sb, indicing: &si_b, data: b };
    let mut out = MatrixFull::new([ac, bc], 0.0);
    _dgemm_full(&am, 'T', &bm, 'N', &mut out, 1.0, 0.0);
    out.data
}

/// `A B` for flat col-major buffers `A [ar, ac]`, `B [br, bc]`, `ac == br`; result `[ar, bc]`.
///
/// Borrowed views for the same reason as [`gemm_nt`]: the operands include `q`/`qt`/`i3`
/// (`nao^2 * naux` doubles) and the copies would dominate the engine-build and gradient peaks.
/// Direct BLAS for the same reason as [`gemm_nt`] — see the note there on the global OpenBLAS
/// thread count that the tensor library's `%` rewrites on every call.
fn gemm_nn(a: &[f64], ar: usize, ac: usize, b: &[f64], br: usize, bc: usize) -> Vec<f64> {
    debug_assert_eq!(ac, br);
    let (sa, si_a, sb, si_b) = ([ar, ac], [1usize, ar.max(1)], [br, bc], [1usize, br.max(1)]);
    let am = MatrixFullSlice { size: &sa, indicing: &si_a, data: a };
    let bm = MatrixFullSlice { size: &sb, indicing: &si_b, data: b };
    let mut out = MatrixFull::new([ar, bc], 0.0);
    _dgemm_full(&am, 'N', &bm, 'N', &mut out, 1.0, 0.0);
    out.data
}

/// `s_metric · q` for a `[naux, ncol]` right-hand side, applied in auxiliary-column blocks.
///
/// `gemm_nn` (and the tensor matmul behind it) materialises the result as a *second* full-size
/// buffer while the first is still alive, and converts its layout with a third: for the engine's
/// `qt = s_metric · q` (`naux * nmo^2` = 874 MB at naphthalene/def2-TZVP) that transient was the
/// highest point of the whole gradient. Working block-wise holds one `naux * block` temporary
/// (3.7 MB) and writes straight into the output.
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
/// The eigensolver's third tuple element is the matrix dimension, not the
/// LAPACK `info`; the eigenvalues are validated explicitly instead.
///
/// Driver choice: `dsyevd` (divide and conquer), not `dsyev`.  The chi0
/// factorisation in [`build_lr_factor`] is called once per (orbital, frequency
/// node) pair -- ~2000 times per excited-state gradient at azobenzene/
/// def2-TZVPP, with `naux = 1234`.  `dsyev` is dominated by the *serial*
/// tridiagonal QR iteration and hardly threads at all (benchmarked on this
/// machine: 2.86 s single-thread, 2.37 s at 16 threads), whereas `dsyevd` is
/// 0.50 s / 0.20 s -- an order of magnitude, and it uses the pool.
fn eigh_symmetric(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mat = to_mat(a, n, n);
    let (vecs, vals, _info) = _dsyevd(&mat, 'V');
    let vals = vals;
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
/// its own (legacy behaviour).  Returns `(shells, shell_of)` where `shells` is
/// the list of index groups (each of size >= 1) and `shell_of[p]` is the
/// position of orbital `p` in `shells`.
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

// ---------------------------------------------------------------------------
// low-rank screened response (REST real-symmetric convention)
