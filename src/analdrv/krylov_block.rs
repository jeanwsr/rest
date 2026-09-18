//! Block Krylov subspace solver for `(1 + A) x = b`.
//!
//! Thin wrapper around `crate::solvers::krylov::krylov_tsr`.  Preserves
//! the original call signature for `analdrv/hessian/rscf.rs` and `analdrv/hessian/uscf.rs`.

use super::prelude::*;
use crate::solvers::krylov::{krylov_tsr, KrylovConfig};

/// Solve `(I + aop) x = b` by a block Krylov subspace method with hard restarts.
///
/// # Parameters
///
/// - `aop` : Linear operator. Given a `[n, nblock]` input it must return a `[n, nblock]` output
///   (the action of `A` applied column-wise).
/// - `b` : Right-hand sides, shape `[n, nset]`. Each column is one RHS.
/// - `x0` : Optional initial guess, shape `[n, nset]`. Zero initial guess is used if not provided.
/// - `tol` : Convergence tolerance on `max(||new_trial_vec_i||)`.
/// - `max_cycle` : Maximum **total** number of inner cycles, summed across restarts.
/// - `max_space` : Maximum subspace size in cycles before a hard restart is triggered.
/// - `lindep` : Vectors with `||v||^2 < lindep` are dropped from the subspace.
///
/// # Returns
///
/// `x` of shape `[n, nset]`, an approximate solution of `(I + aop) x = b`.
pub fn krylov_block(
    mut aop: impl FnMut(TsrView) -> Tsr,
    b: TsrView,
    x0: Option<TsrView>,
    tol: f64,
    max_cycle: usize,
    max_space: usize,
    lindep: f64,
    tol_inflation: f64,
) -> Tsr {
    let config = KrylovConfig {
        tol,
        max_cycle,
        max_space: Some(max_space),
        lindep,
        max_residual_factor: tol_inflation,
        ..Default::default()
    };
    let (x, _result) = krylov_tsr(&mut aop, b, x0, &config);
    x
}
