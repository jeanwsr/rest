//! Sandbox: distributed (ScaLAPACK) Cholesky factorization and triangular solve of the
//! 2c-2e Coulomb metric for MPI-parallel rimatr builds.
//!
//! ## Triggering (strictly isolated)
//!
//! This module is only compiled when the `mpi` and `scalapack` features are both enabled,
//! and only invoked when `J2CDecompOption::distributed` selects it (see
//! [`use_distributed_j2c`]):
//!
//! - `Auto` (default): requires `naux >= J2C_DISTRIBUTED_NAUX_MIN` **and**
//!   `nproc >= J2C_DISTRIBUTED_NPROC_MIN` (large systems only; typical runs and all
//!   tests never trigger),
//! - `On`: forced (mainly for testing the distributed path on small systems),
//! - `Off`: never (explicit fallback).
//!
//! Only `policy = Cd` (Cholesky) is supported; the eigen path and the serial/rank-0
//! path are untouched.
//!
//! ## Layout notes
//!
//! The full-range and SR slots produce `tmp_ri3fn = [naux, loc_n_baspar]` per rank:
//! **full** auxiliary rows and the rank's **contiguous** basis-pair columns. The
//! ScaLAPACK triangular solve requires both operands in block-cyclic layout, so:
//!
//! 1. the columns are redistributed (contiguous → block-cyclic) into the distributed
//!    RHS `B` ([`scatter_to_blockcyclic`]),
//! 2. `pdtrtrs` solves `U^T · X = B` in the block-cyclic layout,
//! 3. `X` is gathered back to the contiguous per-rank columns ([`gather_from_blockcyclic`]).
//!
//! All communication uses matching synchronous send/receive pairs over the same
//! iteration order on every rank, so no request lifetime management is needed.

use std::ops::Range;

use mpi::topology::SimpleCommunicator;
use mpi::traits::{Communicator, Destination, Source};
use tensors::matrix::distributedmatrixfull::{DistributedMatrixFull, pdpotrf, pdtrtrs};
use tensors::matrix_scalapack::CblacsGrid;
use tensors::{BasicMatrix, MatrixFull};

use crate::ri_jk::{J2CDistributedMode, J2CDecompOption, J2CDecompPolicy};

/// Decide whether the distributed Cholesky solve should be used.
///
/// Sandbox guard: the distributed path is used only when
/// - `policy = Cd` (Cholesky-only),
/// - the mode is `On`, or `Auto` with a large enough auxiliary basis and process count.
pub fn use_distributed_j2c(option: &J2CDecompOption, naux: usize, nproc: usize) -> bool {
    if option.policy != J2CDecompPolicy::Cd {
        return false;
    }
    match option.distributed {
        J2CDistributedMode::On => true,
        J2CDistributedMode::Off => false,
        J2CDistributedMode::Auto => {
            naux >= crate::ri_jk::J2C_DISTRIBUTED_NAUX_MIN
                && nproc >= crate::ri_jk::J2C_DISTRIBUTED_NPROC_MIN
        }
    }
}

/// Adaptive block size for the block-cyclic layouts (must be < every global dimension).
fn block_size(naux: usize, n_cols: usize) -> i32 {
    let max_block = 256_i32;
    let limit = naux.min(n_cols).saturating_sub(1).max(1) as i32;
    max_block.min(limit)
}

/// Distributed Cholesky factorization of the 2c-2e Coulomb metric.
///
/// Every rank must supply the **full** metric `j2c` (e.g. computed on rank 0 and
/// broadcast); the block-cyclic layout is built locally without communication. The
/// returned distributed matrix holds the upper Cholesky factor `U` (`J = U^T U`) in
/// its upper triangle.
pub fn distributed_cholesky(
    grid: &CblacsGrid,
    j2c: &MatrixFull<f64>,
) -> DistributedMatrixFull<f64> {
    let naux = j2c.size[0];
    debug_assert_eq!(j2c.size[1], naux);
    let block = block_size(naux, naux);
    let mut u_dist = DistributedMatrixFull::from_matrixfull(grid, j2c, block, block, 0, 0);
    let info = pdpotrf(grid, 'U', &mut u_dist, 1, 1);
    assert_eq!(info, 0, "pdpotrf failed with info = {}", info);
    u_dist
}

/// Distributed triangular solve `U^T · X = B` in place (Cholesky convention of the
/// Cd policy: `cderi = j3c · L^{-T}` with `L = U^T`).
pub fn distributed_triangular_solve(
    grid: &CblacsGrid,
    u_dist: &DistributedMatrixFull<f64>,
    b: &mut DistributedMatrixFull<f64>,
) {
    let info = pdtrtrs(grid, 'U', 'T', 'N', u_dist, 1, 1, b, 1, 1);
    assert_eq!(info, 0, "pdtrtrs failed with info = {}", info);
}

/// Local block-cyclic column index of the global column `j` (owner `j/nb % npcol`).
fn bc_col_index(j: usize, nb: usize, npcol: usize) -> usize {
    let block_j = j / nb;
    (block_j / npcol) * nb + (j % nb)
}

/// Redistribute the columns of `local` (`[naux, loc_n_col]`, the rank's **contiguous**
/// global column block `col_ranges[rank]`) into the block-cyclic layout, i.e. build
/// the distributed right-hand side for the triangular solve. Rows are kept full on
/// every rank, so only column data is communicated.
pub fn scatter_to_blockcyclic(
    world: &SimpleCommunicator,
    grid: &CblacsGrid,
    nb: i32,
    col_ranges: &[Range<usize>],
    local: &MatrixFull<f64>,
) -> DistributedMatrixFull<f64> {
    let naux = local.size[0];
    let n_cols_total = col_ranges[col_ranges.len() - 1].end;
    let mb = nb as usize;
    let npcol = grid.npcol as usize;
    let nprow = grid.nprow as usize;
    let myrow = grid.myrow as usize;
    let mycol = grid.mycol as usize;
    let my_rank = world.rank() as usize;
    let nprocs = world.size() as usize;

    let mut b_dist = DistributedMatrixFull::new(grid, naux as i32, n_cols_total as i32, nb, nb, 0, 0, 0.0_f64);

    // full all-to-all with matching synchronous send/receive order on every rank
    for src in 0..nprocs {
        for dst in 0..nprocs {
            if src == dst {
                continue;
            }
            // columns of the contiguous block of `src` that belong to the
            // block-cyclic columns owned by `dst` (rank -> column-owner: mycol = dst % npcol)
            let cols: Vec<usize> = col_ranges[src]
                .clone()
                .filter(|&j| ((j / nb as usize) % npcol) == dst % npcol)
                .collect();
            if src == my_rank {
                let mut data = Vec::with_capacity(cols.len() * naux);
                for &j in cols.iter() {
                    let off = (j - col_ranges[my_rank].start) * naux;
                    data.extend_from_slice(&local.data[off..off + naux]);
                }
                world.process_at_rank(dst as i32).send(&data[..]);
            }
            if dst == my_rank {
                let mut data = vec![0.0_f64; cols.len() * naux];
                world.process_at_rank(src as i32).receive_into(&mut data[..]);
                for (k, &j) in cols.iter().enumerate() {
                    let lj = bc_col_index(j, nb as usize, npcol);
                    for i in 0..naux {
                        let block_i = i / mb;
                        if block_i % nprow != myrow {
                            continue;
                        }
                        let li = (block_i / nprow) * mb + (i % mb);
                        b_dist.data[li + lj * b_dist.lld as usize] = data[k * naux + i];
                    }
                }
            }
        }
    }
    // self-to-self columns
    for j in col_ranges[my_rank].clone() {
        if ((j / nb as usize) % npcol) == mycol {
            let lj = bc_col_index(j, nb as usize, npcol);
            let off = (j - col_ranges[my_rank].start) * naux;
            for i in 0..naux {
                let block_i = i / mb;
                if block_i % nprow != myrow {
                    continue;
                }
                let li = (block_i / nprow) * mb + (i % mb);
                b_dist.data[li + lj * b_dist.lld as usize] = local.data[off + i];
            }
        }
    }
    b_dist
}

/// Gather the solution `X` (block-cyclic layout) back to the rank's **contiguous**
/// column block `col_ranges[rank]` with full rows.
///
/// Two stages are needed because the block-cyclic layout scatters both rows and
/// columns: first the row blocks of each block-cyclic column are collected by the
/// column-owning rank (so full columns exist there), then the full columns are sent
/// to their contiguous owner.
pub fn gather_from_blockcyclic(
    world: &SimpleCommunicator,
    grid: &CblacsGrid,
    nb: i32,
    col_ranges: &[Range<usize>],
    x_dist: &DistributedMatrixFull<f64>,
) -> MatrixFull<f64> {
    let naux = x_dist.desc[2] as usize;
    let n_cols_total = x_dist.desc[3] as usize;
    let mb = nb as usize;
    let npcol = grid.npcol as usize;
    let nprow = grid.nprow as usize;
    let myrow = grid.myrow as usize;
    let mycol = grid.mycol as usize;
    let my_rank = world.rank() as usize;
    let nprocs = world.size() as usize;
    let my_contig = col_ranges[my_rank].clone();
    let mut result = MatrixFull::new([naux, my_contig.len()], 0.0);

    let n_row_blocks = naux.div_ceil(mb);
    let n_col_blocks = n_cols_total.div_ceil(nb as usize);

    // Each block (r, c) of the block-cyclic solution lives on rank
    // (r % nprow, c % npcol). Every rank needs the full rows of its own contiguous
    // columns, so the holder of block (r, c) sends the block rows of the columns of
    // block c that fall into the contiguous range of `dst`, directly to `dst`.
    // All ranks iterate the (r, c, dst) triples in the same order, so the matching
    // synchronous send/receive pairs cannot deadlock.
    for r in 0..n_row_blocks {
        for c in 0..n_col_blocks {
            let row_owner = r % nprow;
            let col_owner = c % npcol;
            let holder = row_owner * npcol + col_owner;
            let block_cols: Vec<usize> =
                ((c * nb as usize)..((c + 1) * nb as usize).min(n_cols_total)).collect();
            for dst in 0..nprocs {
                let cols: Vec<usize> = block_cols
                    .iter()
                    .copied()
                    .filter(|&j| col_ranges[dst].contains(&j))
                    .collect();
                if cols.is_empty() {
                    continue;
                }
                // rows of block r available locally
                let lr = (r / nprow) * mb;
                let valid_rows = mb.min(naux - r * mb);
                if holder == my_rank {
                    let mut data = Vec::with_capacity(cols.len() * valid_rows);
                    for &j in cols.iter() {
                        let lj = bc_col_index(j, nb as usize, npcol);
                        let base = lr + lj * x_dist.lld as usize;
                        for ii in 0..valid_rows {
                            data.push(x_dist.data[base + ii]);
                        }
                    }
                    if dst == my_rank {
                        // self block: copy locally
                        for (k, &j) in cols.iter().enumerate() {
                            let off = (j - my_contig.start) * naux;
                            for ii in 0..valid_rows {
                                result.data[off + r * mb + ii] = data[k * valid_rows + ii];
                            }
                        }
                    } else {
                        world.process_at_rank(dst as i32).send(&data[..]);
                    }
                } else if dst == my_rank {
                    let valid_rows = mb.min(naux - r * mb);
                    let mut data = vec![0.0_f64; cols.len() * valid_rows];
                    world.process_at_rank(holder as i32).receive_into(&mut data[..]);
                    for (k, &j) in cols.iter().enumerate() {
                        let off = (j - my_contig.start) * naux;
                        for ii in 0..valid_rows {
                            result.data[off + r * mb + ii] = data[k * valid_rows + ii];
                        }
                    }
                }
            }
        }
    }

    let _ = mycol;
    result
}

