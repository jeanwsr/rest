//! Thin flat-buffer linear-algebra adapters shared by the GW/BSE gradients.
//!
//! These wrap `rest_tensors`' BLAS entry point and keep the memory-layout
//! conventions in one place: `MatrixFull` is column-major, first index fastest,
//! and the gemm wrappers borrow their operands instead of copying them.

use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;
use tensors::{MatrixFull, MatrixFullSlice};

/// Wrap a flat column-major buffer as a `MatrixFull` (copying it).
#[inline]
pub(crate) fn to_mat(v: &[f64], r: usize, c: usize) -> MatrixFull<f64> {
    MatrixFull::from_vec([r, c], v.to_vec()).expect("to_mat: shape mismatch")
}

/// `A B` for flat column-major buffers `A [ar, ac]`, `B [br, bc]`, `ac == br`.
#[inline]
pub(crate) fn gemm_nn(
    a: &[f64], ar: usize, ac: usize,
    b: &[f64], br: usize, bc: usize,
) -> Vec<f64> {
    debug_assert_eq!(ac, br, "gemm_nn: contraction dimension mismatch");
    let (sa, ia) = ([ar, ac], [1usize, ar.max(1)]);
    let (sb, ib) = ([br, bc], [1usize, br.max(1)]);
    let am = MatrixFullSlice { size: &sa, indicing: &ia, data: a };
    let bm = MatrixFullSlice { size: &sb, indicing: &ib, data: b };
    let mut out = MatrixFull::new([ar, bc], 0.0);
    _dgemm_full(&am, 'N', &bm, 'N', &mut out, 1.0, 0.0);
    out.data
}

/// `A^T B` for flat column-major buffers `A [ar, ac]`, `B [br, bc]`, `ar == br`.
#[inline]
pub(crate) fn gemm_nt(
    a: &[f64], ar: usize, ac: usize,
    b: &[f64], br: usize, bc: usize,
) -> Vec<f64> {
    debug_assert_eq!(ar, br, "gemm_nt: contraction dimension mismatch");
    let (sa, ia) = ([ar, ac], [1usize, ar.max(1)]);
    let (sb, ib) = ([br, bc], [1usize, br.max(1)]);
    let am = MatrixFullSlice { size: &sa, indicing: &ia, data: a };
    let bm = MatrixFullSlice { size: &sb, indicing: &ib, data: b };
    let mut out = MatrixFull::new([ac, bc], 0.0);
    _dgemm_full(&am, 'T', &bm, 'N', &mut out, 1.0, 0.0);
    out.data
}
