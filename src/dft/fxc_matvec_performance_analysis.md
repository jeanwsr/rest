# `fxc_matvec` Performance Analysis and Optimization Plan

## 1. Overview

`fxc_matvec` (num_int.rs:611–771) computes the exchange-correlation kernel matrix-vector product for TDDFT linear response. It is called inside iterative eigensolvers (Davidson/FEAST), so it is a **hot-path function** whose performance directly determines time-to-solution.

The function dispatches by functional class:
- **LDA** (`nvar=1`): `fxc_matvec_lda` (lines 623–661)
- **GGA** (`nvar=4`): `fxc_matvec_gga` (lines 664–771)

All MO-value matrices stored in `FXCMatvecData` use `MatrixFull<f64>`, which is **column-major** `[nmo, ngrids]`. This means `matrix[[i, g]]` maps to memory offset `g + i * ngrids`.

---

## 2. Performance Issues

### 2.1 Strided Memory Access in Manual Loops (CRITICAL)

**Affected: LDA Step 1/3, GGA Step 1/3**

All manual nested loops follow the pattern:
```rust
for g in 0..ngrids {      // outer: grid points (100k+)
    for i in 0..nocc {    // inner: orbital index (10–500)
        sum += mo[[i, g]] * t[[i, g]];   // stride = ngrids
    }
}
```

Since `MatrixFull` is column-major, `mo[[i, g]]` is at offset `g + i * ngrids`. When the inner loop increments `i`, successive accesses jump by **`ngrids * 8` bytes** (e.g., 800 KB for `ngrids=100000`). This causes:

| Issue | Mechanism | Impact |
|-------|-----------|--------|
| **L1 cache misses** | Each iteration touches a different cache line; no spatial locality | 10–50× slower than contiguous access |
| **TLB thrashing** | Stride > page size (4 KB) means every access misses the TLB | Adds 10–100 cycles per access |
| **No SIMD auto-vectorization** | Non-unit stride prevents compiler vectorization | 2–8× lost throughput |

This pattern appears at:
- LDA line 635–641: `rho_z[g] = Σ_i mo_occ[[i,g]] * t[[i,g]]`
- LDA line 651–654: `mo_vir_scaled[[a,g]] = mo_vir[[a,g]] * v[g]`
- GGA line 688–703: `rho_z` computation with 4 density components
- GGA line 728–732, 743–747, 756–759: `right_scaled` / `right_grad_scaled` per alpha

**Fix**: Swap loop order so that the contiguous grid dimension is innermost.

---

### 2.2 Copy of Input Vector `z` on Every Call (HIGH)

**Affected: Both LDA and GGA, line 629/672**

```rust
let z_mat = MatrixFull::from_vec([nocc, nvir], z.to_vec()).unwrap();
```

`z.to_vec()` heap-allocates and copies the full `z` vector (length = `nocc * nvir`) on every invocation. For a typical system with 50 occupied × 200 virtual = 10,000 elements, that is 80 KB copied. In an iterative solver with 100+ iterations, this adds up.

**Fix**: See Section 3.3.

---

### 2.3 Allocation-Heavy Per-Call Design (HIGH)

**Affected: Both LDA and GGA**

Every call to `fxc_matvec` performs multiple heap allocations. The lifetime of each allocation is the duration of one function call:

| Function | Allocations per call | Total fresh heap |
|----------|---------------------|------------------|
| LDA | 6 (`z_mat`, `t`, `rho_z`, `v`, `mo_vir_scaled`, `result`) | ~`(nocc*nvir) + (nocc+nvir)*ngrids*2 + 2*ngrids` |
| GGA | 20+ (`z_mat`, `t0`, `t_grad×3`, `rho_z`, `fxc_eff_grid`, `result`, plus per-alpha `right_scaled/right_grad_scaled/contrib×3`) | ~`3×` LDA |

For a typical system (nocc=50, nvir=200, ngrids=150000):
- LDA: ~60 MB allocated and freed per call
- GGA: ~180 MB allocated and freed per call

In a Davidson solver doing 200 iterations, LDA alone allocates 12 GB total.

**Fix**: See Section 3.4.

---

### 2.4 GGA 4×4 Kernel Application: Poor Loop Nesting (MEDIUM)

**Affected: GGA Step 2, lines 709–718**

```rust
for g in 0..ngrids {          // ~150k iterations
    for alpha in 0..4 {       // 4 iterations
        for beta in 0..4 {    // 4 iterations (16 total)
            let w_idx = g + alpha*ngrids + beta*4*ngrids;
            sum += data.wfxc[w_idx] * rho_z[g + beta*ngrids];
        }
    }
}
```

Issues:
1. **Outer loop over grids with tiny inner work**: 150k loop overhead for 16 FLOPs per iteration → dominated by branch/latency overhead.
2. **wfxc access pattern**: `wfxc[g + alpha*ngrids + beta*4*ngrids]` — varying `beta` (innermost) jumps `4*ngrids`, varying `alpha` jumps `ngrids`. Both are **non-contiguous**.
3. **rho_z access**: `rho_z[g + beta*ngrids]` — `beta` varies innermost, jumping by `ngrids` → cache-unfriendly.

This is a 4×4 matrix-vector product per grid point. Since 4×4 is tiny, the loop overhead and memory latency dominate.

**Fix**: See Section 3.5.

---

### 2.5 GGA Alpha Loop: Redundant GEMM + Scaling (MEDIUM)

**Affected: GGA Step 3, lines 723–768**

For `alpha > 0`, each alpha iteration performs:
1. Scale `mo_vir` by `fxc_a[g]` → `right_scaled`
2. GEMM: `mo_occ_grad[d] × right_scaled^T` → `contrib_a`
3. Scale `mo_vir_grad[d]` by `fxc_a[g]` → `right_grad_scaled`
4. GEMM: `mo_occ × right_grad_scaled^T` → `contrib_b`

The `right_scaled` matrix is computed identically for `alpha=1,2,3` (scaling `mo_vir` by different `fxc_a` each time) — this is 3 separate passes over `mo_vir`. Similarly, `right_grad_scaled` is 3 separate passes over each `mo_vir_grad[d]`.

Total for alpha>0: **6 scaling passes + 6 GEMMs** = 12 O(ngrids·nvir·nocc) operations, but the 6 scaling passes are entirely memory-bound (just multiply by scalar).

**Fix**: Fuse the scaling into the GEMM where possible, or at minimum fuse the `right_scaled` construction for all alphas in one pass.

---

### 2.6 No Parallelism (MEDIUM)

**Affected: Both LDA and GGA**

The manual loops (grid-point iteration, scaling) are all serial. While BLAS GEMM calls may use threaded OpenBLAS internally, the intermediate scaling loops and rho_z accumulation run single-threaded. For GGA, the 4 alpha iterations in Step 3 are independent and could run in parallel.

**Fix**: See Section 3.6.

---

## 3. Proposed Optimization Plan

### 3.1 Loop Reordering for Column-Major Access

**Priority: CRITICAL** | **Effort: Low** | **Expected gain: 5–50× on manual loops**

Swap inner/outer loops so that `g` (the contiguous dimension) is innermost:

```rust
// Before (cache-hostile):
for g in 0..ngrids {
    for i in 0..nocc {
        sum += data.mo_occ[[i, g]] * t[[i, g]];
    }
}

// After (cache-friendly):
for i in 0..nocc {
    for g in 0..ngrids {
        sum_per_i[g] += data.mo_occ[[i, g]] * t[[i, g]];  // OR accumulate to rho_z after
    }
}
```

**Concrete changes:**
- LDA `rho_z` (lines 635–641): iterate `i` outer, `g` inner; accumulate into `rho_z[g]` directly.
- LDA `mo_vir_scaled` (lines 651–654): iterate `a` outer, `g` inner.
- GGA `rho_z` (lines 688–703): iterate `i` outer, `g` inner for all 4 density components simultaneously (fuse the 4 accumulations into one pass).
- GGA `right_scaled`/`right_grad_scaled` (lines 728–759): iterate `a` outer, `g` inner.

This costs no extra memory and is a pure code transformation.

### 3.2 Replace Manual Element-wise Loops with BLAS/BLAS-like Primitives

**Priority: HIGH** | **Effort: Medium** | **Expected gain: 2–4× additional**

Where possible, express operations as matrix primitives:

- **`rho_z = column_sum(mo_occ ⊙ t)`** (Hadamard product → column reduce):
  Could use a custom kernel, or restructure as `diag(mo_occ^T · t)`. Better: after loop reordering (Section 3.1), the inner loop is already vectorizable by the compiler.

- **`mo_vir_scaled = diag(v) · mo_vir`** (row-wise scaling):
  After loop reordering, the inner loop over `g` runs on contiguous memory and the compiler should auto-vectorize. Alternatively, use a rank-1 update formulation:
  - `mo_vir_scaled = mo_vir ⊙ (1 ⊗ v)` i.e., broadcast `v` across rows of `mo_vir`, then element-wise multiply.

- **GGA `fxc_eff_grid = wfxc_block @ rho_z_block` (4×4 matvec per grid point)**:
  After restructuring layout (Section 3.5), this becomes a single GEMM call.

### 3.3 Eliminate `z.to_vec()` Copy

**Priority: HIGH** | **Effort: Medium** | **Expected gain: saves 1 allocation + 1 copy per call**

Options:
1. If `MatrixFull::from_vec` takes ownership, change the API to accept `z: Vec<f64>` (move semantics), pushing the copy responsibility to the caller.
2. Implement `MatrixFull::from_slice` that stores a reference (zero-copy view) — if the downstream GEMM supports it.
3. Pre-allocate a reusable `MatrixFull` buffer for `z_mat` in a workspace struct and copy data into it with `memcpy`-style write, avoiding the double allocation.

Option 3 is the most practical: add a `z_mat: MatrixFull<f64>` field to `FXCMatvecData` (initialized once in `prepare_fxc_data`), and in `fxc_matvec`, overwrite its `.data` slice with `z` before calling GEMM.

### 3.4 Pre-allocate Workspace Buffers

**Priority: HIGH** | **Effort: Medium** | **Expected gain: eliminates ~90% of allocations**

Add reusable buffers to `FXCMatvecData` or a companion `FXCMatvecWorkspace`:

```rust
pub struct FXCMatvecWorkspace {
    // LDA buffers
    pub z_mat: MatrixFull<f64>,        // [nocc, nvir]
    pub t: MatrixFull<f64>,            // [nocc, ngrids]
    pub rho_z: Vec<f64>,               // [ngrids]
    pub v: Vec<f64>,                   // [ngrids] (or reuse rho_z in-place)
    pub mo_vir_scaled: MatrixFull<f64>, // [nvir, ngrids]
    pub result: MatrixFull<f64>,        // [nocc, nvir]

    // GGA additional buffers
    pub t_grad: [MatrixFull<f64>; 3],          // [nocc, ngrids] each
    pub fxc_eff_grid: Vec<f64>,                 // [ngrids * 4]
    pub right_scaled: MatrixFull<f64>,          // [nvir, ngrids] reusable
    pub right_grad_scaled: MatrixFull<f64>,     // [nvir, ngrids] reusable
    pub contrib: MatrixFull<f64>,               // [nocc, nvir] reusable
}
```

**Critical detail**: Ensure `FXCMatvecWorkspace` is NOT `Clone`, and use `&mut` in `fxc_matvec`. The caller becomes responsible for creating one workspace per thread (for multi-threaded solvers).

Alternatively, embed the LDA buffers directly in `FXCMatvecData` since they are always needed; add GGA buffers behind `Option<>`.

### 3.5 Restructure GGA Kernel Layout

**Priority: MEDIUM** | **Effort: Medium** | **Expected gain: 2–5× on Step 2**

**Current layout**: `wfxc[g + alpha*ngrids + beta*4*ngrids]` — grid-major, then alpha, then beta.

**Proposed layout**: Either:
- **(a)** Store the 4×4 matrix as `[alpha, beta, g]` so each `(alpha,beta)` pair has contiguous grid-point access: `wfxc[alpha*4*ngrids + beta*ngrids + g]`
- **(b)** Transpose on-the-fly in `prepare_fxc_data` to layout (a).
- **(c)** Restructure the loop: iterate alpha outer, beta outer, g inner:

```rust
for alpha in 0..4 {
    for beta in 0..4 {
        let w = &wfxc[beta*4*ngrids + alpha*ngrids..][..ngrids];  // after transpose
        let r = &rho_z[beta*ngrids..][..ngrids];
        for g in 0..ngrids {
            fxc_eff_grid[alpha*ngrids + g] += w[g] * r[g];
        }
    }
}
```

With layout (a), both `w` and `r` accesses are contiguous, enabling auto-vectorization.

Even better: treat the whole operation as a batch matrix-vector product. With layout (a):
- `fxc_eff_grid` is `[4, ngrids]` column-major
- `rho_z` is `[4, ngrids]` column-major
- `wfxc` is `[4*4, ngrids]` = 16 rows, each row is a grid function
- The operation at each `g` is a 4×4 @ 4×1 matvec → can be done as 4 dot products over the 4 beta components. This is small enough to just manually unroll the 4×4 loop.

### 3.6 Parallelize Independent Work

**Priority: MEDIUM** | **Effort: Low–Medium** | **Expected gain: 2–4× on scaling loops**

The following loops are embarrassingly parallel and large enough (ngrids ≫ 104) to benefit from threading:

1. **LDA rho_z computation**: Each `i` (orbital) is an independent reduction. Could be parallelized with `rayon` over `i` with atomic or per-thread partial sums.
2. **LDA mo_vir_scaled construction**: Each `a` iteration is independent.
3. **GGA alpha loop (Step 3)**: The 4 alpha iterations are independent — use `rayon::par_iter` over `0..4`.
4. **GGA rho_z computation**: Same as LDA, each `i` is independent.

**Caveat**: OpenBLAS GEMM already uses multiple threads internally. Avoid nested parallelism by:
- Using `rayon` only on loops NOT inside BLAS calls, OR
- Configuring OpenBLAS to single-thread (`OPENBLAS_NUM_THREADS=1`) and using `rayon` globally, OR
- Using `rayon::scope` with controlled thread counts and serial GEMM for parallel-for-over-alpha approach.

**Recommendation**: Start with parallelism only on the purely scalar loops (scaling, rho accumulation), leaving GEMMs to use OpenBLAS threads. The alpha loop in GGA Step 3 is the best candidate — each iteration is a scaling+GEMM+GEMM triple, and each alpha is fully independent.

### 3.7 Fuse GGA Scaling Passes

**Priority: LOW** | **Effort: Medium** | **Expected gain: 20–30% on GGA Step 3**

Currently, for each `alpha > 0`, `right_scaled` is recomputed from `mo_vir` with the same structure (only the scalar `fxc_a[g]` differs). Fuse into one pass:

```rust
// One pass over (a, g) to compute right_scaled for all 4 alphas:
// Store as [alpha, nvir, ngrids] or process sequentially and immediately GEMM.
```

However, since each alpha's `right_scaled` is immediately consumed by a GEMM, the best approach is to compute one alpha's `right_scaled`, do the GEMM, then reuse the same buffer for the next alpha — this is already close to optimal if the GEMM dominates. The real saving comes from not re-allocating buffers (Section 3.4).

### 3.8 Consider Blocking Over Grids

**Priority: LOW** | **Effort: High** | **Expected gain: Variable**

For very large `ngrids` (> 500k) where even `nocc * ngrids` matrices exceed L3 cache, consider blocking over grid batches:

```rust
for grid_block in 0..nblocks {
    let g_start = grid_block * block_size;
    let g_end = min(g_start + block_size, ngrids);
    // Process this block: mo_occ[:, g_start..g_end], etc.
    // Each block fits in L3 cache → better reuse
}
```

This helps when multiple passes over the same grid range (e.g., GGA Step 1 computing `rho_z` for all 4 components, then immediately reading them in Step 2) can stay in cache. However, this adds significant complexity to coordinate GEMM calls on sub-matrices.

### 3.9 Eliminate `MatrixFull::from_vec` for Intermediate Matrices

**Priority: LOW** | **Effort: Low** | **Expected gain: Minor**

Several places construct a `MatrixFull` from a slice just to pass it to GEMM:
```rust
let aop_d = MatrixFull::from_vec([num_basis, ngrids], aop_d_slice.iter().cloned().collect()).unwrap();
```
This is in `prepare_fxc_data` (called once) so it is not critical, but worth noting the pattern.

---

## 4. Recommended Implementation Order

| Order | Task | Impact | Effort |
|-------|------|--------|--------|
| 1 | Loop reordering (Section 3.1) | **CRITICAL** | Low |
| 2 | Pre-allocate workspace buffers (Section 3.4) | **HIGH** | Medium |
| 3 | Eliminate `z.to_vec()` copy (Section 3.3) | **HIGH** | Medium |
| 4 | Restructure GGA kernel layout + loop (Section 3.5) | MEDIUM | Medium |
| 5 | Parallelize independent loops (Section 3.6) | MEDIUM | Low–Medium |
| 6 | Fuse GGA scaling passes (Section 3.7) | LOW | Low (after 3.4) |

**Expected cumulative speedup**: 5–20× for LDA, 5–30× for GGA, depending on system size and hardware.

---

## 5. Risk/Correctness Considerations

- **Loop reordering**: Since both loop orders compute the same sum (addition is commutative and associative in exact arithmetic), floating-point results may differ in the last 1–2 bits due to different summation order. This is acceptable for quantum chemistry where FEAST/Davidson solvers already converge to a tolerance (e.g., 10⁻⁶).
- **Workspace pre-allocation**: Requires `&mut` access to workspace, making `fxc_matvec` no longer `&self`-only. Callers must be updated. Thread safety requires one workspace per thread.
- **GGA layout change**: Requires corresponding changes in `prepare_fxc_data` (where `wfxc` is computed). Both paths must stay consistent.
- **Parallelism**: Must avoid nested parallelism with BLAS threads. Benchmark to find the right thread partitioning.

---

## 6. Profiling Data (To Collect)

Before implementing, collect baseline data to validate these hypotheses:

```bash
# Cache miss profiling
perf stat -e cache-misses,cache-references,L1-dcache-load-misses,LLC-load-misses \
    -e dtlb_load_misses.miss_causes_a_walk \
    ./tddft_binary --input benchmark.inp

# Flame graph
perf record -g ./tddft_binary --input benchmark.inp
perf script | FlameGraph/stackcollapse-perf.pl | FlameGraph/flamegraph.pl > fxc_flame.svg
```

Comparing cache-miss rates before and after loop reordering will directly confirm the primary diagnosis.
