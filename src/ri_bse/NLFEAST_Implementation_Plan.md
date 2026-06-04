# NLFEAST Implementation Plan — Faithful to the Li & Polizzi (2025) Algorithm

## 0. Motivation

The existing `src/main.rs` implements a **hybrid** algorithm: it solves the projected
nonlinear eigenvalue problem (NLEVP) inside the subspace via companion linearisation
and uses contour integration *only* for subspace refinement.  It does **not**
implement the canonical FEAST spectral-projection algorithm described in
`NLFEAST_Instructions.tex`.

This document describes a **new implementation** that faithfully follows the
FEAST algorithm for nonlinear eigenvalue problems as derived in
`NLFEAST_Instructions.tex` (Algorithm 1 / §4.3–§4.4).

**Key property of the new design:** The user provides **two** matvec
closures — one for real shifts (residual computation on the real axis)
and one for complex shifts (quadrature nodes on the contour).  This
two-function split reflects the physics: the problem is real-valued on the
real axis and only acquires complex values when analytically continued to
the contour.  The algorithm treats the nonlinearity as a black box and
works for **any** holomorphic `T(λ)`.

---

## 1. The Algorithm (from `NLFEAST_Instructions.tex` §4.4)

### 1.1 Pseudocode

```
Algorithm: FEAST for Nonlinear T(λ)·x = 0

Input:  T_real(λ)    – matvec on the real axis:  λ∈ℝ, x∈ℝⁿ → T(λ)·x ∈ ℝⁿ
        T_cplx(z)    – matvec on the contour:    z∈ℂ, x∈ℝⁿ → T(z)·x ∈ ℂⁿ
        Γ            – closed contour in ℂ enclosing wanted eigenvalues
        {z_j, w_j}   – N_q quadrature nodes and weights on Γ
        X            – n×m₀ initial subspace (random real), m₀ ≥ m
        τ            – convergence tolerance

Output: {(λ_i, x_i)} for eigenvalues inside Γ  (λ_i ∈ ℝ, x_i ∈ ℝⁿ)

 1:  Λ ← 0          (real, length-m₀)
 2:  R ← X          (real n×m₀; first iteration: RHS = X)
 3:  while not converged:
 4:      // ─── Parallel loop over quadrature nodes ───
 5:      for j = 1 to N_q:
 6:          Solve  T_cplx(z_j) · X_j = R       ▷ X_j ∈ ℂ^{n×m₀}, R ∈ ℝ^{n×m₀}
 7:          Y_j = (X − X_j) · (z_j·I − Λ)⁻¹    ▷ diagonal weighting, complex
 8:      end for
 9:
10:      // ─── Assemble spectral projectors ───
11:      Q₀ = Σ_j  w_j · Y_j                    ▷ Q₀ ∈ ℂ^{n×m₀}
12:      Q₁ = Σ_j  w_j · z_j · Y_j              ▷ Q₁ ∈ ℂ^{n×m₀}
13:
14:      // ─── Thin QR ───
15:      [q, r] = QR(Q₀)       ▷ q ∈ ℂ^{n×m₀} orthonormal, r ∈ ℂ^{m₀×m₀}
16:
17:      // ─── Form and solve reduced eigenproblem ───
18:      C = q^H · Q₁ · r⁻¹                     ▷ C ∈ ℂ^{m₀×m₀}
19:      Solve C · W = W · Λ̃                     ▷ zgeev; Λ̃ ∈ ℂ (→ ℝ in exact arithmetic)
20:
21:      // ─── Rayleigh–Ritz update ───
22:      X ← Re(q · W)                           ▷ project back to real
23:      Λ ← Re(Λ̃)                               ▷ keep real part of eigenvalues
24:
25:      // ─── Residuals (real axis) ───
26:      for i = 1 to m₀:
27:          r_i = T_real(λ_i) · x_i             ▷ ℝⁿ → ℝⁿ, full nonlinear T
28:      end for
29:      R ← [r_1, …, r_{m₀}]                    ▷ ℝ^{n×m₀}
30:
31:      // ─── Convergence check ───
32:      if ‖r_i‖ < τ  for all i with λ_i inside Γ:  break
33:  end while
34:  return {(λ_i, x_i) | λ_i inside Γ}
```

**Critical difference from the naive single-matvec approach:**

| Location | λ domain | T(λ)·x result | Which matvec |
|----------|----------|---------------|--------------|
| Residual (line 27) | λ_i ∈ ℝ | ℝⁿ | `T_real(λ)` |
| Shifted solve (line 6) | z_j ∈ ℂ | ℂⁿ | `T_cplx(z_j)` |

The RHS `R` is **always real** (computed from `T_real` on the real axis).
The shifted linear system `T_cplx(z_j)·X_j = R` has a complex matrix applied
to a real right-hand side, yielding a complex solution `X_j`.

### 1.2 Why this works for arbitrary nonlinear `T(λ)`

| Mechanism | Explanation |
|-----------|-------------|
| **Nonlinearity "frozen" at z_j** | At each quadrature node, `T_cplx(z_j)` is a **constant complex matrix**. The fact that `T` depends nonlinearly on `z` is irrelevant during the linear solve — `T(z_j)` is just data. |
| **Spectral projection via Keldysh's theorem** | `∮ T⁻¹(z) dz` projects onto the invariant subspace of eigenvalues inside Γ, **regardless** of whether `T(z)` is linear, polynomial, or general nonlinear in `z`. |
| **Residual-driven correction** | The term `T⁻¹(z)·R` injects Newton-type correction information. When `R → 0`, the projection becomes exact. |
| **Real-axis residuals** | The nonlinear `T_real(λ)` is evaluated **only** on the real axis (line 27), costing `m₀` real matvecs per iteration, not `N_q` complex matvecs. |
| **Real↔Complex separation** | `T_real` is a function ℝ → ℝ^{n×n}; `T_cplx` is its analytic continuation ℂ → ℂ^{n×n}. Both represent the same abstract operator `T(λ)`, just evaluated on different domains. Physically, the problem is real on the real energy axis; complex values only appear when evaluating inside the contour integral. |

### 1.3 Optional preconditioner simplification (§4.4.4 of tex)

If the user can supply cheaper approximations, the linear solves in line 6
can use a preconditioner `P(z_j)` instead of the full `T_cplx(z_j)`:

```
Solve  P_cplx(z_j) · X_j = R      ▷ cheaper preconditioner on the contour
```

The full `T_real(λ)` is still used in the residual computation (line 27).
This is crucial for GW problems where `Σ^C(z)` is expensive to evaluate at
each quadrature node.

Symmetrically, if the residual evaluation `T_real(λ)` itself has an expensive
component, the user may optionally split it:

```
T_real(λ) = T₀(λ) + ΔT(λ)      where ΔT is the expensive nonlinear part
```

The residual can be computed as `r_i = T₀(λ_i)·x_i + ΔT(λ_i)·x_i`, allowing
the expensive part to be compiled separately or computed with a different
precision.

---

## 2. Software Architecture

### 2.1 Crate structure

```
src/
├── main.rs                  # Demo / integration test
├── nlfeast/
│   ├── mod.rs               # Re-exports
│   ├── algorithm.rs         # Core NLFEAST iteration (Algorithm 1)
│   ├── contour.rs           # Contour types + quadrature generators
│   ├── linear_solver.rs     # GMRES (real-embedded) + solver abstraction
│   └── result.rs            # NLFeastResult + convergence diagnostics
└── problems/
    ├── mod.rs               # Problem trait
    ├── quadratic.rs         # Quadratic EVP: T(λ) = λ²M + λC + K
    └── gw.rs                # GW placeholder (future work)
```

### 2.2 Core trait: `NonLinearEVP`

This is the **only** interface the user must implement.

```rust
/// A nonlinear eigenvalue problem  T(λ)·x = 0.
///
/// The problem is real-valued on the real axis (where the physics lives)
/// and complex-valued on the contour (where the spectral projection happens).
///
/// `T_real: ℝ → ℝ^{n×n}`  and  `T_cplx: ℂ → ℂ^{n×n}`  are the same abstract
/// operator evaluated on different domains.
pub trait NonLinearEVP {
    /// Problem dimension (T(λ) is n×n).
    fn dim(&self) -> usize;

    // ── Real-axis matvec (residual computation) ──

    /// Evaluate  T(λ)·x  at a **real** shift λ with a real vector x.
    ///
    /// Returns a real vector of length n.
    /// Used only in the convergence check (line 27 of the pseudocode).
    fn t_matvec_real(&self, lambda: f64, x: &[f64]) -> Vec<f64>;

    // ── Complex-contour matvec (shifted linear solves) ──

    /// Evaluate  T(z)·x  at a **complex** shift z with a real vector x.
    ///
    /// Returns a complex vector of length n.
    /// Used to build the GMRES matvec for solving T(z_j)·X_j = R  at each
    /// quadrature node (line 6 of the pseudocode).
    fn t_matvec_cplx(&self, z: Complex<f64>, x: &[f64]) -> Vec<Complex<f64>>;

    // ── Optional: preconditioner on the contour ──

    /// Evaluate  P(z)·x  where P(z) ≈ T(z) is a cheaper approximation
    /// for the shifted linear solves on the contour.
    ///
    /// Default implementation calls `t_matvec_cplx` (i.e., no preconditioner).
    fn preconditioner_matvec_cplx(&self, z: Complex<f64>, x: &[f64]) -> Vec<Complex<f64>> {
        self.t_matvec_cplx(z, x)
    }
}
```

**Why two matvecs are needed (and sufficient):**

| Method | Called from | Domain | Frequency | Cost per iteration |
|--------|------------|--------|-----------|-------------------|
| `t_matvec_real(λ, x)` | Residual check | λ ∈ ℝ | m₀ times | m₀ × cost(T_real) |
| `t_matvec_cplx(z, x)` | GMRES matvec at each z_j | z ∈ ℂ | ≈ N_q × (GMRES iters) | many, but preconditioner can reduce |
| `preconditioner_matvec_cplx(z, x)` | GMRES (optional) | z ∈ ℂ | ≈ N_q × (GMRES iters) | cheaper than full T_cplx |

No structural knowledge of `T` is required — the algorithm only needs to
apply the operator, never to inspect its entries.

### 2.3 Contour abstraction

```rust
/// A closed contour in the complex plane.
pub trait Contour {
    /// Number of quadrature nodes.
    fn n_nodes(&self) -> usize;

    /// Generate the j-th quadrature node and its weight.
    fn node(&self, j: usize) -> (Complex<f64>, Complex<f64>);
}

/// Circular contour: centre c, radius r, trapezoidal rule.
pub struct CircleContour {
    pub centre: f64,
    pub radius: f64,
    pub n_quad: usize,
}

/// Elliptical contour with Gauss-Legendre quadrature.
pub struct EllipseContour {
    pub centre: f64,
    pub a: f64,          // semi-major axis (horizontal)
    pub b: f64,          // semi-minor axis (vertical)
    pub n_quad: usize,
}
```

### 2.4 Linear solver abstraction

```rust
/// Solves  A·x = b  for multiple right-hand sides.
///
/// The matrix A is accessed only through `matvec`.
pub trait LinearSolver {
    /// Solve  matvec(x) = rhs  for a single right-hand side.
    fn solve(
        &self,
        matvec: &dyn Fn(&[f64]) -> Vec<f64>,
        rhs: &[f64],
    ) -> Vec<f64>;

    /// Solve for multiple columns of RHS.  Default: loop over columns.
    fn solve_multi(
        &self,
        matvec: &dyn Fn(&[f64]) -> Vec<f64>,
        rhs: &MatrixFull<f64>,
    ) -> MatrixFull<f64> { /* column loop */ }
}
```

The default implementation will be `RestartedGmres` (already implemented in
the existing codebase).  Users can provide a direct sparse solver if the
matrix is available explicitly.

### 2.5 Data flow diagram

```
User provides
    │
    ├── t_matvec_real(λ, x) → ℝⁿ        (real axis only)
    └── t_matvec_cplx(z, x) → ℂⁿ        (contour only)

                   │
                   ▼
┌──────────────────────────────────────────────────┐
│ NLFeastSolver                                     │
│                                                   │
│  // All X, Λ, R are real                          │
│  X ∈ ℝ^{n×m₀}    (eigenvector approximations)     │
│  Λ ∈ ℝ^{m₀}      (eigenvalue approximations)      │
│  R ∈ ℝ^{n×m₀}    (residuals, from t_matvec_real)  │
│                                                   │
│  while not converged:                             │
│    ┌───────────────────────────────────────┐      │
│    │ for each z_j ∈ ℂ (parallel):           │      │
│    │   build shifted matvec (2n×2n real)   │      │
│    │     from t_matvec_cplx(z_j, ·)        │      │
│    │   GMRES.solve(shifted_matvec, R)       │      │
│    │   → X_j ∈ ℂ^{n×m₀}                    │      │
│    │   Y_j = (X − X_j)·(z_j·I − Λ)⁻¹ ∈ ℂ  │      │
│    └───────────────────────────────────────┘      │
│    Q₀ = Σ w_j·Y_j  ∈ ℂ^{n×m₀}                    │
│    Q₁ = Σ w_j·z_j·Y_j  ∈ ℂ^{n×m₀}               │
│    [q, r] = QR(Q₀)    q ∈ ℂ^{n×m₀}               │
│    C = q^H·Q₁·r⁻¹     C ∈ ℂ^{m₀×m₀}             │
│    zgeev(C) → (W, Λ̃)   W ∈ ℂ, Λ̃ ∈ ℂ              │
│    X ← Re(q·W)         ▷ back to ℝ^{n×m₀}        │
│    Λ ← Re(Λ̃)           ▷ keep real part           │
│    for each i:                                    │
│      r_i = t_matvec_real(λ_i, x_i)  → ℝⁿ        │
│    R ← [r_1, …, r_{m₀}]  ∈ ℝ^{n×m₀}            │
│    check_convergence(R, Λ, Γ)                     │
│                                                   │
│  return {(λ_i ∈ ℝ, x_i ∈ ℝⁿ) | λ_i inside Γ}     │
└──────────────────────────────────────────────────┘
```

**Data type summary:**

| Quantity | Domain | Stored as |
|----------|--------|-----------|
| X, R | ℝ^{n×m₀} | `MatrixFull<f64>` |
| Λ | ℝ^{m₀} | `Vec<f64>` |
| X_j, Y_j, Q₀, Q₁, q | ℂ^{n×m₀} | `(MatrixFull<f64>, MatrixFull<f64>)` |
| r, C, W, Λ̃ | ℂ^{m₀×m₀} | small dense complex (native `Complex<f64>`) |

Since m₀ ≪ n, everything m₀×m₀ is cheap to handle with native complex arithmetic.
The heavy parts are the n×m₀ real-embedded GMRES solves and the n×m₀ complex
matrix accumulations.

---

## 3. Key Implementation Details

### 3.1 Complex arithmetic: real embedding vs. native complex

The existing codebase uses **real embedding**: a complex `n×n` system is
represented as a `2n×2n` real system.  This is done because `rest_tensors`
matrices are real-valued.

**Key simplification from the real/complex matvec split:** Since `R` is
always real (computed from `t_matvec_real` on the real axis), the GMRES
right-hand side for `T_cplx(z_j)·X_j = R` has the form:

```
b = [R_col;  0]   ∈ ℝ^{2n}     (bottom half is zero)
```

The GMRES iterate `x = [vr; vi]` will acquire a non-zero imaginary part as
the iteration proceeds (because `T_cplx(z_j)` is complex), so the full
2n×2n embedding is still required.  Only the initial RHS is simplified.

**Recommendation:** Keep the real embedding for linear solves (GMRES).
For intermediate complex quantities with small column counts (m₀ ≈ 10–50),
use native `Complex<f64>` arithmetic — this simplifies the QR, C-matrix
assembly, and `zgeev` call.  For the large n×m₀ quantities (X_j, Y_j, Q₀, Q₁, q),
store as pairs of real matrices:

```rust
struct ComplexMatrix {
    re: MatrixFull<f64>,  // n × k
    im: MatrixFull<f64>,  // n × k
}
```

Operations on `ComplexMatrix`:
- `add`, `sub`, `scale` (real and complex scalar)
- `qr()` → `(ComplexMatrix, ComplexMatrix)` (both q and r are complex in general)
- `matmul` — dense matrix multiply for assembling Q₀, Q₁
- `column_scale` — multiply each column by a complex scalar

### 3.1b Shifting real↔complex in the matvec builder

The shifted GMRES matvec at node `z_j` is built from `t_matvec_cplx(z_j, ·)`:

```rust
fn make_shifted_matvec(
    n: usize,
    t_matvec_cplx: &impl Fn(Complex<f64>, &[f64]) -> Vec<Complex<f64>>,
    z: Complex<f64>,
) -> impl Fn(&[f64]) -> Vec<f64> + '_ {
    move |v: &[f64]| -> Vec<f64> {
        let vr = &v[0..n];
        let vi = &v[n..2 * n];
        let t_vr = t_matvec_cplx(z, vr);   // ← complex matvec here
        let t_vi = t_matvec_cplx(z, vi);
        let mut out = vec![0.0; 2 * n];
        for i in 0..n {
            out[i]     = t_vr[i].re - t_vi[i].im;
            out[n + i] = t_vr[i].im + t_vi[i].re;
        }
        out
    }
}
```

Note: this uses `t_matvec_cplx` (not `t_matvec_real`), because the shift
`z_j` lies on the complex contour.  The RHS vector `b` being real just
means the initial GMRES residual has `b_im = 0`.

### 3.2 First iteration: Λ = 0 special case

When `Λ = 0`, the weighting matrix is:

```
(z_j·I − 0)⁻¹ = diag(1/z_j, …, 1/z_j)
```

This is well-conditioned as long as the contour does not pass through the
origin.  No special handling is needed — the general formula works.

### 3.3 The term `(z_j·I − Λ)⁻¹` (diagonal weighting)

Λ is real (length m₀).  For each quadrature node `z_j ∈ ℂ` and each current
eigenvalue `λ_k ∈ ℝ`:

```
d_{j,k} = 1 / (z_j − λ_k)   ∈ ℂ
```

Then `Y_j = (X − X_j) · D_j` where `D_j = diag(d_{j,1}, …, d_{j,m₀}) ∈ ℂ^{m₀×m₀}`.

In code:
```rust
for col in 0..m0 {
    let denom = z_j - Complex::new(lambda[col], 0.0);
    let inv = Complex::new(1.0, 0.0) / denom;   // Complex division
    for row in 0..n {
        // y_j_re[row][col] = inv.re * (x[row][col] - xj_re[row][col])
        //                   - inv.im * (0        - xj_im[row][col])
        // y_j_im[row][col] = inv.re * (0        - xj_im[row][col])
        //                   + inv.im * (x[row][col] - xj_re[row][col])
    }
}
```

Note: `X` is real, so its imaginary part is zero. `X_j` is complex.
This is column scaling in complex arithmetic — O(n·m₀) per node.

### 3.4 Quadrature weights (factor of `1/(2πi)`)

The contour integral formula is:

```
Q_k = (1/(2πi)) ∮ z^k · [X − T⁻¹(z)·R] · (zI − Λ)⁻¹ dz
```

With quadrature `{z_j, w_j}` where `w_j` are the **geometric** weights (e.g.,
from the trapezoidal rule: `w_j = (r·e^{iθ_j} / N_q) · i`), the full
quadrature weight including the `1/(2πi)` prefactor is `w_j / (2πi)`.

For the trapezoidal rule on a circle `z(θ) = c + r·e^{iθ}`:
- `dz = r·i·e^{iθ} dθ`
- Geometric weight: `w_j = r·i·e^{iθ_j} · (2π/N_q)` = `(z_j − c) · (2πi/N_q)`
- Full quadrature weight: `w_j / (2πi) = (z_j − c) / N_q`

So the update at each node is:

```
Q₀ += ((z_j − c) / N_q) · Y_j
Q₁ += ((z_j − c) / N_q) · z_j · Y_j
```

### 3.5 Eigenvalue filtering (real eigenvalues)

After diagonalising `C`, we take `Λ̃ = Re(diag(Λ̃))`.  In exact arithmetic,
eigenvalues of the original real-axis problem are real, so the imaginary
parts from `zgeev` are numerical noise (≈ 10⁻¹⁵).  Retaining only the real
part keeps the downstream arithmetic cleaner.

The interior test for a circular contour:

```
|λ_i − c| < r      where λ_i = Re(λ̃_i)
```

where `c` is the contour centre and `r` is the radius.

The algorithm preserves `m₀` eigenvectors in `X` at all times.  Eigenvalues
outside Γ are carried along passively and may drift out of the contour in
subsequent iterations — this is expected behaviour.

### 3.6 Handling of the reduced eigenproblem

Matrix `C` is `m₀×m₀` complex (non-Hermitian in general).  We diagonalise it
with `zgeev` (LAPACK):

```rust
let (_, w, _, vr, info) = _zgeev(&c_mat, 'N', 'V');
// w[j]   = eigenvalue  λ̃_j ∈ ℂ
// vr[:,j] = right eigenvector  w_j ∈ ℂ^{m₀}
```

Then extract the real parts of eigenvalues and Ritz vectors:

```rust
let lambda: Vec<f64> = w.iter().map(|c| c.re).collect();
let x_new = q_real_part * w_real_part;   // X ← Re(q·W)
```

The Q₀ QR produces a complex q (n×m₀) and complex r (m₀×m₀).  C is formed
in complex arithmetic.  The final `X` is real (eigenvectors of the
real-axis problem).

### 3.7 Convergence criteria

Two criteria from the tex document:

1. **Residual-based** (primary): `‖T_real(λ_i)·x_i‖ < τ` for all `λ_i` inside Γ.
   The residual is computed with `t_matvec_real` — it is a real vector,
   and its 2-norm is the standard Euclidean norm.  Set `τ` close to machine
   epsilon (e.g., `10⁻¹²`).

2. **Stagnation detection**: If eigenvalue changes `|λ_i^{(k+1)} − λ_i^{(k)}|`
   are all below a threshold (e.g., `10⁻¹²`) for 2–3 consecutive iterations,
   terminate.  This handles cases where residuals stall.

### 3.8 Parallelism

The `N_q` linear solves (line 6) are **independent** and can run in parallel.
Each solve uses `t_matvec_cplx(z_j, ·)` to build the shifted matvec.
Implementation: `rayon` parallel iterator over quadrature nodes.

```rust
use rayon::prelude::*;

let results: Vec<(usize, ComplexMatrix)> = quad_nodes
    .par_iter()
    .enumerate()
    .map(|(j, node)| {
        let shifted_mv = make_shifted_matvec(n, &t_matvec_cplx, node.z);
        let xj = solve_shifted_columns(&shifted_mv, &r_mat, &gmres_params);
        let yj = form_yj(&x_mat, &xj, node.z, &lambda);
        (j, yj)
    })
    .collect();
```

### 3.9 Preconditioner simplification (split real/complex)

For GW-type problems, the user may provide two levels of cost reduction:

**Level 1 — Contour preconditioner:** Replace `T_cplx(z_j)` with a cheaper
`P_cplx(z_j)` in the GMRES matvec during shifted solves:

```rust
fn preconditioner_matvec_cplx(&self, z: Complex<f64>, x: &[f64]) -> Vec<Complex<f64>> {
    // e.g., omit Σ^C(z) from T(z) at quadrature nodes
}
```

The residual still uses the full `t_matvec_real`.  This is the GW `Σ^C`-omission
trick (§4.4.4 of tex).

**Level 2 — Residual decomposition (optional):** If the real-axis operator
also has cheap + expensive parts:

```
T_real(λ)·x = T_cheap(λ)·x + T_expensive(λ)·x
```

The user can implement `t_matvec_real` to call both, or the framework can
cache the cheap part across iterations.  This is problem-specific and not
required for correctness.

---

## 4. Implementation Phases

### Phase 1: Complex matrix helpers (`ComplexMatrix`)

- `new(n, k)`, `clone`, accessors
- `add`, `sub`, `scale` (real and complex scalar)
- `qr()` — modified Gram-Schmidt producing `(q: ComplexMatrix, r: ComplexMatrix)` (r is complex in general)
- `matmul` — dense matrix multiply for assembling Q₀, Q₁
- `column_scale` — multiply each column by a complex scalar

### Phase 2: Contour and quadrature

- `CircleContour` with trapezoidal rule
- `EllipseContour` with Gauss-Legendre quadrature
- Unit tests comparing quadrature against known contour integrals

### Phase 3: Linear solver integration

- Refactor existing `gmres` to accept complex RHS via real embedding
- Wrap in a clean `solve_shifted_system(z, rhs_matrix, gmres_params)` function
- Test: solve `(zI − A)·X = RHS` for a known matrix A, compare with direct dense solve

### Phase 4: Core NLFEAST iteration

- Implement the while-loop exactly as in §1.1
- `make_shifted_matvec` using `t_matvec_cplx` (not `t_matvec_real`)
- `form_yj` (complex X_j, real X, real Λ → complex Y_j)
- `assemble_q0_q1` (complex accumulation with quadrature weights)
- QR of complex Q₀ → `(q_complex, r_complex)`
- `form_c_matrix` in native `Complex<f64>` (m₀×m₀, small)
- `solve_reduced_evp` via `zgeev`
- `X ← Re(q·W)`, `Λ ← Re(Λ̃)` (back to real)
- Residual computation: loop over columns, calling `t_matvec_real(λ_i, x_i)`
- Convergence check + stagnation detection

### Phase 5: QuadraticEVP test problem

- Implement `NonLinearEVP` for the quadratic EVP with BOTH matvecs:
  - `t_matvec_real(λ, x)` → `(λ²M + λC + K)·x`  (all real)
  - `t_matvec_cplx(z, x)` → `(z²M + zC + K)·x`  (complex z, real x → complex)
- Compare results against reference `_dgeev` on companion form
- Verify that `N_q ≈ 8–16` and 3–5 iterations give machine precision
- Verify that final `Im(λ_i) < 10⁻¹²` for all interior eigenvalues
- Test with/without preconditioner (e.g., `P(z) = z²M + zC` omitting K)

### Phase 6: GW problem stub

- Skeleton implementation of `NonLinearEVP` for GW
- `t_matvec_real(ε, x)` evaluates `[εS − h − Σ^X − Σ^C(ε)]·x` (real ε, real result)
- `t_matvec_cplx(z, x)` evaluates the same with complex continuation of `Σ^C(z)`
- `preconditioner_matvec_cplx(z, x)` optionally omits `Σ^C(z)` on the contour
- Document what a full implementation needs (mean-field orbitals, dielectric function, etc.)

---

## 5. API Summary (User-Facing)

```rust
use nlfeast::{NLFeastSolver, NLFeastConfig, CircleContour, NonLinearEVP};

// 1. Implement NonLinearEVP for your problem — TWO matvecs required
struct MyProblem { /* … */ }
impl NonLinearEVP for MyProblem {
    fn dim(&self) -> usize { /* … */ }

    // Real-axis matvec:  T(λ)·x  for λ ∈ ℝ, x ∈ ℝⁿ  →  ℝⁿ
    fn t_matvec_real(&self, lambda: f64, x: &[f64]) -> Vec<f64> {
        // Compute T(λ)·x  on the real axis  (used for residuals)
    }

    // Complex-contour matvec:  T(z)·x  for z ∈ ℂ, x ∈ ℝⁿ  →  ℂⁿ
    fn t_matvec_cplx(&self, z: Complex<f64>, x: &[f64]) -> Vec<Complex<f64>> {
        // Compute T(z)·x  at complex shift  (used for shifted solves)
    }

    // Optional: cheaper contour matvec for the shifted linear solves
    fn preconditioner_matvec_cplx(&self, z: Complex<f64>, x: &[f64]) -> Vec<Complex<f64>> {
        // e.g., omit expensive Σ^C(z) term on the contour
        self.t_matvec_cplx(z, x)   // default: use full T
    }
}

// 2. Configure the solver
let config = NLFeastConfig {
    problem: &my_problem,
    contour: CircleContour { centre: c, radius: r, n_quad: 12 },
    m0: 16,                    // subspace size (> expected #evals in contour)
    max_iter: 20,
    tol: 1e-12,
    gmres_restart: 40,
    gmres_max_it: 500,
    gmres_tol: 1e-8,
};

// 3. Solve
let result = NLFeastSolver::solve(&config)?;

// 4. Inspect — eigenvalues and eigenvectors are real
println!("Found {} eigenvalues inside contour", result.n_found);
for (lam, x, residual) in result.iter() {
    println!("λ = {:+.12},  ‖T(λ)x‖ = {:.2e}", lam, residual);
}
```

---

## 6. Comparison: Current Code vs. New Implementation

| Aspect | Current (`src/main.rs`) | New (this plan) |
|--------|------------------------|-----------------|
| Eigenpair extraction | `solve_projected` + companion linearisation | Contour integral → Q₀, Q₁ → C matrix |
| Contour integral role | Subspace update only | **Spectral projector** (core) |
| Matvec interface | Single `t_matvec(Complex, …) → Complex` | **Two functions**: `t_matvec_real(f64, …) → ℝⁿ` + `t_matvec_cplx(Complex, …) → ℂⁿ` |
| Generality | Requires `solve_projected` closure (NLEVP-specific) | Requires two matvecs, no structural knowledge |
| C matrix (`q^H·Q₁·r⁻¹`) | Not computed | **Computed and diagonalised** |
| Algorithm source | Gavin et al. 2018 (hybrid) | Li & Polizzi 2025 / Gavin et al. 2018 (canonical) |
| Preconditioner support | No | Yes (contour preconditioner via `preconditioner_matvec_cplx`) |
| Real-axis physics | Not separated | Explicit: real eigenvalues, real eigenvectors, real residuals |
| Complex only for contour | N/A | Yes — complex arithmetic isolated to quadrature nodes |

---

## 7. Testing Strategy

### 7.1 Unit tests
- Contour quadrature: verify `∮ 1/(z−a) dz = 2πi` for a inside/outside
- Complex matrix operations: QR, matmul, column scaling
- GMRES with complex matvec + real RHS: solve `T_cplx(z)·x = b` vs. direct dense solve
- Verify `t_matvec_real` and `t_matvec_cplx` agree on the real axis: for `λ ∈ ℝ`, `T_cplx(λ)·x ≈ T_real(λ)·x + 0i`

### 7.2 Integration tests (Quadratic EVP)
- `n = 20`, random M, C, K
- Implement both `t_matvec_real` and `t_matvec_cplx`
- Compare all eigenvalues inside Γ against `_dgeev` on companion form
- Verify: `max|λ_nlfeast − λ_ref| < 10⁻¹²`
- Verify: `max‖T_real(λ)·x‖ < 10⁻¹²`
- Verify: all interior `λ_i` have `|Im(λ_i)| < 10⁻¹²`
- Test with/without preconditioner (use `P_cplx(z) = z²M + zC` omitting K)

### 7.3 Convergence tests
- Vary `N_q` (4, 8, 12, 16, 24): verify final accuracy is independent of `N_q`
- Vary `m₀`: verify at least m₀ ≥ m_inside is sufficient
- Verify that over-relaxation in GMRES tolerance (e.g., tol=1e-2) still yields final accuracy tol=1e-12
- Test stagnation detection on hard problems

---

## 8. Required Changes to `Cargo.toml`

```toml
[dependencies]
rest_tensors = { path = "/opt/rest_workspace/rest_tensors" }
rand = "0.8"
num-complex = "0.4"
rayon = "1"            # NEW: parallel quadrature
lapack = "0.14"        # NEW: zgeev for reduced eigenproblem
```

(Note: if `rest_tensors` already provides `_zgeev`, the `lapack` crate may not
be needed.  Check whether `rest_tensors::matrix::matrix_blas_lapack` exports
complex LAPACK routines.)
