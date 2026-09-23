# Solver Unification

> Verify with: `cargo test -p rest --test test_cphf --test test_analdrv`

---

## 1. Davidson: `ri_bse` + `rest_tensors` → `solvers/davidson.rs`

- Moved `ri_bse/davidson_solver.rs` to `solvers/davidson.rs`. Removed 3 dead `use crate::ri_bse::*` imports.
- `rest_tensors::davidson_solve` replaced by `davidson_solver` in `scf_io/addons.rs`. `Box<dyn FnMut>` → `impl FnMut`.
- Removed `rayon` parallel matvec. Relaxed `Fn + Send + Sync` → `FnMut`.

```rust
pub struct DavidsonConfig {
    pub max_subspace: usize,      // 60
    pub add_dim: usize,           // 2: vectors added per iteration
    pub restart_dim: usize,       // 6: subspace size after restart
    pub max_iter: usize,          // 100: max Davidson iterations
    pub tol: f64,                 // 1e-10: convergence: ||r|| < sqrt(tol), |de| < tol
    pub lindep: f64,              // 1e-14: drop trial vectors with norm^2 < lindep
    pub use_mgs: bool,            // true: Modified GS (default); false: Classical GS
    pub divergence_restart: bool, // true: detect |r|/|r_last| > 3 and restart
    pub track_states: bool,       // false: reorder eigenstates by overlap between iterations
}
```

Improvements ported from `rest_tensors/davidson.rs`:

- Configurable `tol` (was hardcoded `1e-10`)
- `lindep` filtering: trial vectors with `||v||^2 < lindep` are dropped
- Preconditioner clamp: `|diag - e|` clamped to `>= 1e-8` before division
- `use_mgs`: MGS vs CGS toggle for search direction orthogonalization
- `divergence_restart`: restore previous eigenvectors and restart on blowup
- `track_states`: overlap-matrix based eigenstate reordering for near-degenerate cases
- Relaxed closure bounds: `Fn + Send + Sync` → `FnMut`

---

## 2. Krylov: 4 implementations → `solvers/krylov.rs`

| # | File | Solver | Status |
|---|------|--------|--------|
| 3 | `ri_cphf/cphf_solver_pyscf.rs` | `solve_krylov_batched` | → `solvers::krylov` |
| 4 | `analdrv/krylov_block.rs` | `krylov_block` | → `solvers::krylov_tsr` (5-line wrapper) |

### Architecture

```
solvers/krylov.rs
├── KrylovConfig           ← tol, max_cycle, max_space, lindep, max_residual_factor
├── krylov_tsr()           ← Tsr/TsrView core (full algorithm)
│   ├── Shared-subspace block Krylov (from #3)
│   │   ├── orth_block()   ← MGS + QR, drops lin-dep directions
│   │   ├── projected_solve() ← Galerkin: H·c=g via solve_general, x=xs·c
│   │   └── Hard restart at max_space cap (from #4)
│   ├── per_rhs_phase()    ← activated when QR collapse detected
│   │   ├── Per-RHS independent subspaces (batched aop, per-RHS CGS/QR)
│   │   ├── solve_one_rhs() ← projected solve + true residual per RHS
│   │   └── Recursive krylov_tsr for stalled RHS
│   └── Post-solve: true residual check + warn! if above tol
├── KrylovResult           ← cycles, aop_calls, per_root_solves, residual
├── krylov_vec()           ← Vec<f64> wrapper (CP-HF callers)
└── krylov() = krylov_vec  ← re-export alias

ri_cphf/cphf_solver_pyscf.rs
└── solve_krylov_batched() ← → solvers::krylov (production)

analdrv/krylov_block.rs
└── krylov_block()         ← 5-line wrapper → krylov_tsr
```

```rust
pub struct KrylovConfig {
    pub tol: f64,                  // 1e-9
    pub max_cycle: usize,          // 50
    pub max_space: Option<usize>,  // hard restart cap
    pub lindep: f64,               // 1e-15
    pub max_residual_factor: f64,  // 1000.0: accept ||r|| < factor * tol
}
```

### Caller defaults

| Field | Struct default | analdrv (`krylov_block`) | hessian (`solve_krylov_batched`) |
|-------|:---:|:---:|:---:|
| `tol` | `1e-9` | `cphf_tol: 1e-9` | `krylov_tol: 1e-9` |
| `lindep` | `1e-15` | `cphf_lindep: 1e-15` | `krylov_lindep: 1e-15` |
| `max_residual_factor` | `1000.0` | `cphf_tol_inflation: 1000.0` | `krylov_tol_inflation: 1000.0` |

### Per-RHS recovery

QR collapse detected when `max ||r|| > max_residual_factor * tol`.  Per-RHS
phase runs independent subspaces; RHS with `||r|| < factor * tol` are accepted
without recursive solve.  Stalled RHS get recursive single-RHS `krylov_tsr`
solves (fresh subspace, no shared coupling).

### Feature parity

| Feature | #1 | #2 | #3 | #4 | Unified |
|---------|:--:|:--:|:--:|:--:|:--:|
| Multi RHS | - | - | Shared | Shared | Shared + per-RHS fallback |
| Hard restart | - | - | - | Subspace cap | max_space cap + per-RHS |
| Generic closure API | - | - | - | TsrView→Tsr | TsrView→Tsr |
| Vec wrapper | ✓ | ✓ | ✓ | - | krylov_vec() |
| Debug output | - | ✓ | ✓+profile | Per-cycle | log::debug!/trace! |
| Lin-dep filtering | - | - | QR | MGS | QR |
| QR collapse recovery | - | - | - | - | Per-RHS + recursive per-root |
| Residual threshold | - | - | - | - | max_residual_factor * tol |

All solvers use the same Galerkin projection: `H = xs^T·(I+A)·xs`, `g = xs^T·b`, solve `H·c = g`, reconstruct `x = xs·c`. #4 and unified solve at each restart boundary.

### Production callers

| Caller | Calls | Notes |
|--------|-------|-------|
| `hessian/rhf.rs` | `solve_krylov_batched()` | Analytic Hessian, 3×natom RHS |
| `analdrv/rscf.rs` | `krylov_block()` | RHF/RKS CP-HF |
| `analdrv/uscf.rs` | `krylov_block()` | UHF/UKS CP-HF |

---

## 3. FEAST: `ri_bse/feast_solver.rs` core → `solvers/feast.rs`

The context-independent part of `ri_bse/feast_solver.rs` (Gauss–Legendre nodes,
preconditioned CG, restarted GMRES, the block-diagonal / inner-GMRES
preconditioners, and the `feast()` outer loop) moved to `solvers/feast.rs`.
Only the FEAST eigenvalue solve is shared; every BSE-specific pre-/post-processing
step stays in `ri_bse/feast_solver.rs`.

```
solvers/feast.rs
├── gauss_legendre_nodes()        ← quadrature on [-1, 1]
├── cg()                          ← SPD preconditioned CG (Vec<f64>)
├── gmres()                       ← restarted GMRES(m) with Givens rotations
├── BlockDiagPrecond              ← 2×2 block-diagonal real embedding
├── InnerGmresPrecond             ← implicit inverse via inner GMRES
├── GmresPrecond                  ← Diag | InnerGmres
├── feast()                       ← linear FEAST outer loop (closures only)
└── extract_eigenpairs()          ← interval selection + ascending sort

ri_bse/feast_solver.rs            ← unchanged pre-/post-processing
├── feast_solve_bse_{singlet,triplet,spin}
├── feast_solve_bse_{tda,nontda}  ← build BSE matvecs, call solvers::feast
├── rayleigh_ritz_refine()        ← renormalized-doubles full-basis projection
├── reconstruct_nontda_pairs_to_xy()
└── filter_eigenpairs() / print_ritz_eigenpairs()

ri_tddft/feast_solver.rs          ← TDDFT wrappers call solvers::feast
```

The renormalized-doubles trick is preserved verbatim: Round 1 runs
`feast_solve_bse_*` (s-only auxiliary basis) through `solvers::feast`, then
`rayleigh_ritz_refine()` projects onto the full auxiliary basis.  No caller
defaults change.

---

## 4. NLFEAST: `ri_bse/nonlinbse.rs` core → `solvers/nlfeast.rs`

The context-independent part of `ri_bse/nonlinbse.rs` (the nonlinear FEAST
subspace iteration of Gavin–Międlar–Polizzi 2018) moved to
`solvers/nlfeast.rs`.  The kernel is generic over an `NlepOperator` trait and
owns the selection, convergence test and contour-integral subspace update; the
BSE-specific matvecs/projected solve/shifted solve are supplied by a
`BseNlepOperator` adapter in `ri_bse/nonlinbse.rs`.

```
solvers/nlfeast.rs
├── ContourNode / NLFeastResult
├── trait NlepOperator
│     projected_solve()  ← Qᴴ T(λ) Q y = 0
│     t_real()           ← real-axis residual T(λ)x
│     solve_shifted()    ← T(z)u = rhs
├── qr_orthonormalise()  ← MGS QR, drops linearly dependent columns
└── nlfeast()            ← Algorithm 1 subspace iteration

ri_bse/nonlinbse.rs
├── ContourNodeData / prepare_contour_nodes()
├── compute_real_matvec() / compute_real_tda_matvec()
├── solve_projected_bse() / solve_projected_tda()
├── BlockDiagPrecond / gmres / make_shifted_*_matvec()
├── BseNlepOperator      ← implements NlepOperator for BSE
└── nlfeast_bse()        ← builds operator + initial subspace, calls kernel
```

### Bug found and fixed while testing

`tests/test_nlfeast.rs` builds linear, Hermitian-quadratic, quartic and random
quartic NLEVPs and compares against dense companion linearisations.  The kernel
originally counted **spurious** Ritz values (Ritz pairs with no exact
counterpart, residual ≫ tol) as eigenvalues inside the contour.  A single
spurious value blocked the convergence test forever and then leaked into the
final result.  The kernel now excludes interior Ritz pairs whose real-axis
residual exceeds `SPURIOUS_RESIDUAL_TOL = 1e-2` from both the convergence test
and the final result (the exclusion the original paper describes in §5).  All
seven stress cases then find exactly the reference in-window spectrum.

---

## 5. Test coverage

| Test suite | Count | Validates |
|-----------|-------|-----------|
| `test_cphf` | 1 | Krylov vs dense |
| `test_davidson` | 1 | Davidson vs dsyev |
| `test_analdrv` | 12 | Krylov (Hessian frequencies) |
| `test_analdrv` | 12 | Hessian frequencies via `krylov_block` → `krylov_tsr`. Includes QR collapse test (H2O B3LYP/6-31G). |
| `test_nlfeast` | 7 | NLFEAST vs dense companion linearisation: linear, Hermitian quadratic, quartic, random quartic, multiple windows, wide window (11 roots), clustered (1e-4 spacing) |
| `bench_pool/gw_bse` FEAST | 2 | `NH3_BSE_FEAST` + `NH3_BSE_FEAST_Nontda` (renormalized doubles, TDA & non-TDA) match reference |
