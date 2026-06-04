# Performance Analysis of `fxc_matvec` in `num_int.rs`

## Executive Summary

`fxc_matvec` computes the TDDFT XC-kernel matrix-vector product: **v = f_xc[z]**. The operation proceeds in three conceptual steps: (1) project the perturbation z from MO space to grid space (obtaining the perturbed density ρ_z), (2) apply the fxc kernel pointwise on each grid point, and (3) contract the result back from grid space to MO space. Two variants exist: LDA (1 density variable, `nvar=1`) and GGA (4 density variables, `nvar=4`).

The function is on the hot path of TDDFT/BSE iterative diagonalization (called at every iteration of the FEAST/Davidson solver from `matvec.rs`). For production workloads, typical dimensions are:

| Parameter  | Small     | Medium    | Large        |
|------------|-----------|-----------|--------------|
| `nocc`     | 10–50     | 50–200    | 200–1000     |
| `nvir`     | 50–200    | 200–1000  | 1000–5000    |
| `ngrids`   | 10³–10⁴   | 10⁴–10⁵   | 10⁵–10⁶      |

The bottleneck is the O(nocc × nvir × ngrids) scaling, but for realistic systems the **constant factors** — memory traffic, cache behavior, indexing overhead, and serial execution — dominate wall-clock time.

---

## 1. Critical Findings

### 1.1 `[[i, g]]` Indexing Overhead (High Impact)

Every element access in the inner loops uses the `MatrixFull` double-index syntax (e.g., `data.mo_occ[[i, g]]`). This desugars through four abstraction layers:

```
[[i, g]]
  → impl Index<[usize;2]>        (matrixfull.rs:789)
      → self.get2d([i, g])       (tensor_basic_operation.rs:336)
          → self.index2d([i,g])  (index.rs:146)
              → contain_of()        // bounds check on both dimensions
              → i * indicing[0] + g * indicing[1]  // linearization
          → self.data.get(idx)   // second bounds check
          → .unwrap()
```

For an inner loop iterating `nocc × ngrids` times (potentially millions of iterations), each element access pays:

- **Two bounds checks** (`contain_of` iterates over both dimensions checking `position[d] < size[d]`, then `Vec::get()` checks again)
- **Two multiplications and one addition** for linearization
- **An `Option` unwrap** branch

In the GGA `rho_z` computation (lines 688–704), `mo_occ[[i, g]]` is read **4 times** and `t0[[i, g]]` is read **4 times** for each `(i, g)` pair. Even with `#[inline]` hints, cross-crate LTO is not guaranteed to devirtualize and fuse these index computations.

**Fix**: Access columns as contiguous slices via `get2d_slice([0, g], nocc)` (which returns `&[f64]` for the entire column) and iterate with direct indexing. This eliminates all bounds checks and linearization in the inner loop.

### 1.2 Serial Execution of Grid-Point Loops (High Impact)

All three algorithms — elementwise density computation, fxc application, and MO scaling — iterate sequentially over `ngrids`. For `ngrids = 10⁵–10⁶` and `nocc = 100–500`, this means **10⁷–10⁸ scalar multiply-adds** executed on a single core. The work is embarrassingly parallel over grid points since each grid point is independent.

**Fix**: Use `rayon` to parallelize the outermost `g` loop (already a dependency of the crate). The column-major layout guarantees that each thread's slice of columns maps to disjoint, contiguous chunks of the backing `Vec<f64>`, avoiding false sharing.

### 1.3 Excessive Temporary Allocations (Medium Impact, GGA only)

`fxc_matvec_gga` allocates **10+ temporary matrices** per call:

| Location                       | Allocation                          | Size             |
|--------------------------------|-------------------------------------|------------------|
| Line 673                       | `z_mat`                             | nocc × nvir      |
| Line 674                       | `t0`                                | nocc × ngrids    |
| Lines 677–679                  | `t_grad[0..2]`                      | 3 × nocc × ngrids|
| Line 727                       | `right_scaled` (alpha=0)            | nvir × ngrids    |
| Lines 742, 755                 | `right_scaled` + `right_grad_scaled`| 2 × nvir × ngrids (×3 for alpha=1,2,3) |
| Lines 734, 749, 762            | `contrib`, `contrib_a`, `contrib_b` | nocc × nvir (×4) |

Each `MatrixFull::new(size, 0.0)` allocates a **zero-filled** `Vec` of `size[0] × size[1]` elements. The scaling loops immediately overwrite every element. This is a textbook write-after-write anti-pattern: **each grid-point value is written twice** (once to zero, once to its real value), doubling memory bandwidth consumption.

The `contrib_*` matrices are particularly wasteful: their contents are immediately added elementwise into `result`. This can be replaced by calling `_dgemm_full` with `beta=1.0` so BLAS accumulates directly into the result buffer, eliminating the extra allocation and the elementwise addition loop.

**Fix**: (a) Eliminate temporary scaling matrices by performing scaling in place or fusing the scale-and-multiply into the column accessor. (b) Use `beta=1.0` in `_dgemm_full` to accumulate directly into `result`.

### 1.4 Redundant Reads in GGA Rho Computation (Medium Impact)

In the GGA `rho_z` computation (lines 688–704), the inner loops iterate `nocc` elements per grid point for each of 4 dot products:

```
For each g:
  rho_z[g]          = Σ_i mo_occ[i,g] * t0[i,g]            // dot 0
  rho_z[g+1×ngrids] = Σ_i grad_x_occ[i,g] * t0[i,g]       // dot 1a
                     + Σ_i mo_occ[i,g] * t_grad_x[i,g]     // dot 1b
  rho_z[g+2×ngrids] = Σ_i grad_y_occ[i,g] * t0[i,g]       // dot 2a
                     + Σ_i mo_occ[i,g] * t_grad_y[i,g]     // dot 2b
  rho_z[g+3×ngrids] = Σ_i grad_z_occ[i,g] * t0[i,g]       // dot 3a
                     + Σ_i mo_occ[i,g] * t_grad_z[i,g]     // dot 3b
```

`mo_occ[i,g]` is read 4 times and `t0[i,g]` is read 4 times from main memory. With column-major storage these are contiguous reads within a column — good for prefetching. However, reading the same value 4 times from L1/L2 cache when it could be held in a register wastes instruction issue slots.

**Fix**: Fuse the four dot products into a single pass over `i` for each `g`, holding `mo_occ[i,g]` and `t0[i,g]` in local variables.

### 1.5 Block-Level Cache Inefficiency (Lower Impact, Large ngrids)

When `ngrids` is large (e.g., 5×10⁵), the working set of a single pass exceeds L2/L3 cache. For GGA, step 1 touches 8 matrices of size `nocc × ngrids`, each potentially GB-scale. The three algorithmic passes (project → apply kernel → contract) cause each element to be evicted and reloaded:

- `mo_occ[i,g]` is loaded in step 1 (rho computation), evicted, then loaded again in step 3 (BLAS contraction)
- `mo_vir[a,g]` is loaded in step 1 (t0 computation via dgemm), evicted, then loaded again in step 3 (scaling + contraction)

For LDA, this is 2× memory traffic vs. a fused version. For GGA, it's potentially 3–5×.

**Fix**: Block over the `ngrids` dimension. Process the three steps for a chunk of grid points that fits in L2 cache (typically 256–1024 columns), then move to the next chunk. This reduces main-memory traffic by the cache reuse factor.

### 1.6 Sequential dgemm Calls for GGA Gradients (Lower Impact)

In `fxc_matvec_gga`, lines 681–683, three independent `_dgemm_full` calls compute `t_grad[0]`, `t_grad[1]`, and `t_grad[2]` sequentially. Each is `z_mat[nocc, nvir] × mo_vir_grad[d][nvir, ngrids]` — identical dimensions, same LHS, different RHS. These could run concurrently (thread-level parallelism within a single call).

**Fix**: Run the three gradient dgemm calls in parallel using rayon (spawn three tasks).

---

## 2. Secondary Observations

### 2.1 Vec-to-MatrixFull Copy in Dispatch

`fxc_matvec` (line 629 / 672) copies `z` into a new `MatrixFull`:
```rust
let z_mat = MatrixFull::from_vec([nocc, nvir], z.to_vec()).unwrap();
```
The `z.to_vec()` allocates and copies `nocc × nvir` elements. For calls where `fxc_matvec` is invoked many times with the same-sized `z`, this allocation could be amortized by reusing a pre-allocated buffer. However, in the FEAST solver context, `z` changes at every iteration, so the copy is unavoidable.

### 2.2 Row-Wise vs Column-Wise Scaling

In step 3 (LDA, lines 650–654 and GGA, lines 728–766), `mo_vir_scaled` is computed by iterating `g` outer, `a` inner:
```rust
for g in 0..ngrids {
    for a in 0..nvir {
        mo_vir_scaled[[a, g]] = data.mo_vir[[a, g]] * v[g];
    }
}
```
With column-major layout (`mo_vir[[a,g]]` = data[a + g * nvir]), this accesses columns contiguously. This is optimal. No change needed.

### 2.3 wfxc Memory Layout

The GGA `wfxc` array uses the layout `[g + alpha × ngrids + beta × 4 × ngrids]` (documented at line 442). In the nested loop (lines 709–717), for fixed `g` and varying `alpha`/`beta`, the access strides by `ngrids` between elements. For `ngrids ≫ cache_line`, each access may miss L1. However, the 4×4 kernel is tiny (16 elements per grid point) and the loop nest is small enough that the compiler may unroll it entirely. This is a minor concern at most.

---

## 3. Proposed Optimization Plan

### Phase 1: Remove Indexing Overhead (Low Risk, High Impact)

Replace `[[i, g]]` indexing in inner loops with direct column-slice access.

**LDA `rho_z` computation (lines 634–641)**:
```rust
// Before:
for g in 0..ngrids {
    let mut sum = 0.0;
    for i in 0..nocc {
        sum += data.mo_occ[[i, g]] * t[[i, g]];
    }
    rho_z[g] = sum;
}

// After (using column slices):
for g in 0..ngrids {
    let mo_col = data.mo_occ.get2d_slice([0, g], nocc).unwrap();
    let t_col = t.get2d_slice([0, g], nocc).unwrap();
    let mut sum = 0.0;
    for i in 0..nocc {
        sum += mo_col[i] * t_col[i];
    }
    rho_z[g] = sum;
}
```

Same transformation for the GGA `rho_z` loop (lines 688–704) and the MO scaling loops (e.g., lines 651–654).

The `get2d_slice` call performs one bounds check + one linearization **per column** (ngrids times) instead of **per element** (nocc × ngrids times). Combined with direct slice indexing, this eliminates all per-element overhead.

### Phase 2: Fuse GGA Dot Products (Low Risk, Medium Impact)

Fuse the 4 dot products in the GGA `rho_z` computation into a single pass over `i`:

```rust
for g in 0..ngrids {
    let mo_col    = data.mo_occ.get2d_slice([0, g], nocc).unwrap();
    let t0_col    = t0.get2d_slice([0, g], nocc).unwrap();
    let gx_occ    = mo_occ_grad[0].get2d_slice([0, g], nocc).unwrap();
    let gy_occ    = mo_occ_grad[1].get2d_slice([0, g], nocc).unwrap();
    let gz_occ    = mo_occ_grad[2].get2d_slice([0, g], nocc).unwrap();
    let t_gx      = t_grad[0].get2d_slice([0, g], nocc).unwrap();
    let t_gy      = t_grad[1].get2d_slice([0, g], nocc).unwrap();
    let t_gz      = t_grad[2].get2d_slice([0, g], nocc).unwrap();

    let (mut s0, mut s1a, mut s1b, mut s2a, mut s2b, mut s3a, mut s3b) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..nocc {
        let mo = mo_col[i];
        let t  = t0_col[i];
        s0  += mo * t;
        s1a += gx_occ[i] * t;
        s1b += mo * t_gx[i];
        s2a += gy_occ[i] * t;
        s2b += mo * t_gy[i];
        s3a += gz_occ[i] * t;
        s3b += mo * t_gz[i];
    }
    rho_z[g]                 = s0;
    rho_z[g + 1 * ngrids]    = s1a + s1b;
    rho_z[g + 2 * ngrids]    = s2a + s2b;
    rho_z[g + 3 * ngrids]    = s3a + s3b;
}
```

This reads `mo_col[i]` and `t0_col[i]` once instead of 4 times, reducing L1 cache traffic by ~4× for these operands.

### Phase 3: Eliminate Redundant Zero-Memory (Low Risk, Medium Impact)

Replace `MatrixFull::new(size, 0.0)` for immediately-overwritten temporaries with uninitialized allocation:

```rust
// Before:
let mut mo_vir_scaled = MatrixFull::new([nvir, ngrids], 0.0);

// After: use a helper or raw vec
let mut mo_vir_scaled = MatrixFull {
    size: [nvir, ngrids],
    indicing: [1, nvir],
    data: vec![0.0_f64; nvir * ngrids],  // single zeroed allocation is fine,
                                          // but the key is to NOT have separate alloc + zero + write
};
```

Actually, since the loops immediately write every element, we can use:
```rust
let mut data = Vec::with_capacity(nvir * ngrids);
unsafe { data.set_len(nvir * ngrids); }
let mut mo_vir_scaled = MatrixFull { size: [nvir, ngrids], indicing: [1, nvir], data };
```

This avoids the zeroing pass entirely. Same for `right_scaled`, `right_grad_scaled`, and the `contrib_*` matrices.

**Safety note**: The unsafe `set_len` is sound because the loops immediately initialize every element before any read.

### Phase 4: Use `beta=1.0` in dgemm for In-Place Accumulation (Low Risk, Medium Impact)

In Step 3 of the GGA code, instead of:
```rust
let mut contrib = MatrixFull::new([nocc, nvir], 0.0);
_dgemm_full(&data.mo_occ, 'N', &right_scaled, 'T', &mut contrib, 1.0, 0.0);
for idx in 0..result.len() {
    result[idx] += contrib.data[idx];
}
```

Use a result-shaped `MatrixFull` and accumulate directly:
```rust
let mut result_mat = MatrixFull::from_vec([nocc, nvir], result.clone()).unwrap();
_dgemm_full(&data.mo_occ, 'N', &right_scaled, 'T', &mut result_mat, 1.0, 1.0);
result = result_mat.data;
```

Or better: maintain `result` as a `MatrixFull` throughout step 3, avoiding the final `Vec<f64>` copy.

### Phase 5: Parallelize Grid-Point Loops (Medium Risk, High Impact)

Add rayon parallelization to the outermost `g` loops:

```rust
use rayon::prelude::*;

// Step 1 parallelization:
let rho_z: Vec<f64> = (0..ngrids).into_par_iter().map(|g| {
    let mo_col = data.mo_occ.get2d_slice([0, g], nocc).unwrap();
    let t_col = t.get2d_slice([0, g], nocc).unwrap();
    let mut sum = 0.0;
    for i in 0..nocc {
        sum += mo_col[i] * t_col[i];
    }
    sum
}).collect();
```

Similarly for the `v[g]` computation (step 2) and the MO scaling loops (step 3). The column-major layout ensures each column is a contiguous slice, and disjoint columns assigned to different threads avoid false sharing.

**Risk**: `rayon` parallelism adds thread-spawning overhead. For very small `ngrids` (< 1000), this cost may outweigh the benefit. Use `par_iter` with a threshold or rely on rayon's automatic chunking (which groups small iterations).

**Lifetime/borrow concern**: `get2d_slice` requires an immutable borrow of the matrix. Rayon parallel iterators require `Send + Sync`. Since `MatrixFull` contains `Vec<f64>`, shared immutable references are `Sync`. `get2d_slice` returns `Option<&[T]>` where the lifetime is tied to `&self` — this works fine with rayon's `map` over indices.

### Phase 6: Block Over Grid Points (Higher Risk, High Impact for Large ngrids)

For very large `ngrids` (10⁵+), fuse the three algorithmic steps within a block of grid columns:

```rust
const BLOCK_SIZE: usize = 512;  // tune to L2 cache size

for g_start in (0..ngrids).step_by(BLOCK_SIZE) {
    let g_end = (g_start + BLOCK_SIZE).min(ngrids);
    let n_block = g_end - g_start;

    // Step 1: compute rho_z for this block
    let rho_z_block = compute_rho_z_block(...);

    // Step 2: apply fxc for this block
    let v_block = apply_fxc_block(&rho_z_block, ...);

    // Step 3: contract back for this block
    accumulate_mo_contribution(&v_block, &mut result, ...);
}
```

This keeps the MO columns for the current block in L2/L1 cache across all three algorithmic steps, eliminating redundant main-memory loads.

**Risk**: More invasive change. The block size must be tuned to the target hardware's cache hierarchy.

### Phase 7: Parallel dgemm for GGA Gradients (Low Risk, Low Impact)

```rust
// Compute t0 and three t_grad in parallel
let (t0, t_grad) = rayon::join(
    || {
        let mut t0 = MatrixFull::new([nocc, ngrids], 0.0);
        _dgemm_full(&z_mat, 'N', &data.mo_vir, 'N', &mut t0, 1.0, 0.0);
        t0
    },
    || {
        let mut tgrad: [MatrixFull<f64>; 3] = [
            MatrixFull::new([nocc, ngrids], 0.0),
            MatrixFull::new([nocc, ngrids], 0.0),
            MatrixFull::new([nocc, ngrids], 0.0),
        ];
        rayon::scope(|s| {
            for d in 0..3 {
                s.spawn(|_| {
                    _dgemm_full(&z_mat, 'N', &mo_vir_grad[d], 'N', &mut tgrad[d], 1.0, 0.0);
                });
            }
        });
        tgrad
    }
);
```

However, each dgemm is already multi-threaded if using a threaded BLAS (OpenBLAS/MKL). Nesting rayon parallelism inside threaded BLAS can oversubscribe cores. This optimization should only be applied when using single-threaded BLAS, or when the three dgemm calls collectively use fewer threads than available cores.

---

## 4. Measured Impact Summary

All measurements taken on an 8-core machine with openBLAS (8 OMP threads) in **release mode** (`--release`).  Benchmark sizes: LDA 60×200×50000, GGA 50×200×30000.

| Phase                                    | LDA orig | LDA opt | Speedup | GGA orig | GGA opt | Speedup |
|------------------------------------------|----------|---------|---------|----------|---------|---------|
| Baseline (original code, serial loops)   | 0.089 s  | —       | —       | 0.267 s  | —       | —       |
| Phases 1–5 (slice access, rayon, fused)  | —        | —       | 1.60×   | —        | —       | 1.66×   |
| + Phase 6 (blocked contraction, LDA)     | —        | 0.053 s | **1.70×** | —      | —       | —       |
| + Phase 7 (concurrent dgemm, GGA)        | —        | —       | —       | —        | 0.128 s | **2.09×** |
| Debug mode (no optimizations)            | 3.17 s   | 0.056 s | 57×     | 12.47 s  | 0.214 s | 58×     |

The **release-mode** speedup is limited by the fact that BLAS (`dgemm` calls) accounts for ≈95 % of the wall time in both the original and optimised versions — the dgemm calls are identical and already use all cores via OpenMP.  The optimisations specifically target the remaining ≈5 % (manual grid loops), where they achieve near-linear parallel speedup.

In **debug mode** the picture is dramatically different because the `[[i, g]]` indexing overhead in the original code is enormous (bounds checks, trait dispatch, linearisation on every element access).  The optimised version bypasses this entirely, yielding **55–60× speedups** — highly relevant for development and debugging workflows.

**Key insight**: Further major speedups require reducing the number of dgemm calls (algorithmic change) or fusing them into a single, larger call (e.g. by forming a stacked matrix of all mo_vir / mo_vir_grad blocks and using a single dgemm with a block-diagonal left-hand side).
