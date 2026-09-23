// ============================================================================
// Stress tests for the nonlinear FEAST (NLFEAST) solver.
//
// These tests build several *different* nonlinear eigenvalue problems whose
// spectra are known independently (closed form or a dense companion
// linearisation) and check that NLFEAST finds every eigenvalue inside the
// search contour, with small residuals and no spurious results.
//
// The algorithm under test is the context-independent kernel
// `pyrest::solvers::nlfeast::nlfeast` (Gavin–Międlar–Polizzi 2018), which is the
// same subspace iteration used by the BSE driver `ri_bse::nonlinbse::nlfeast_bse`.
//
// Reference: docs/FEASTnonlinear.md
// ============================================================================

use num::Complex;
use pyrest::solvers::nlfeast::{nlfeast, qr_orthonormalise, ContourNode, NlepOperator};
use rest_tensors::matrix::matrix_blas_lapack::{_dgeev, _dinverse};
use rest_tensors::matrix::MatrixFull;

// ---------------------------------------------------------------------------
// Tiny deterministic RNG (no external dependency, reproducible across runs)
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }
    fn next_u64(&mut self) -> u64 {
        // SplitMix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.f64().max(1e-300);
        let u2 = self.f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Small dense helpers (row-major n×n)
// ---------------------------------------------------------------------------

fn identity(n: usize) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        a[i * n + i] = 1.0;
    }
    a
}

fn matvec(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += a[i * n + j] * x[j];
        }
        y[i] = s;
    }
    y
}

/// Random orthogonal n×n matrix (row-major), columns orthonormal.
fn random_orthogonal(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg::new(seed);
    let mut a = vec![0.0; n * n];
    for v in a.iter_mut() {
        *v = rng.normal();
    }
    // Modified Gram–Schmidt on the columns
    for j in 0..n {
        for i in 0..j {
            let mut dot = 0.0;
            for k in 0..n {
                dot += a[k * n + j] * a[k * n + i];
            }
            for k in 0..n {
                a[k * n + j] -= dot * a[k * n + i];
            }
        }
        let nrm: f64 = (0..n).map(|k| a[k * n + j] * a[k * n + j]).sum::<f64>().sqrt();
        for k in 0..n {
            a[k * n + j] /= nrm;
        }
    }
    a
}

fn tridiag_3_minus1(n: usize) -> Vec<f64> {
    let mut t = vec![0.0; n * n];
    for i in 0..n {
        t[i * n + i] = 3.0;
        if i > 0 {
            t[i * n + (i - 1)] = -1.0;
        }
        if i + 1 < n {
            t[i * n + (i + 1)] = -1.0;
        }
    }
    t
}

/// Coefficients c_0..c_k (monic, c_k = 1) of ∏(λ − r).
fn poly_from_roots(roots: &[f64]) -> Vec<f64> {
    let mut c = vec![1.0];
    for &r in roots {
        let mut next = vec![0.0; c.len() + 1];
        for (i, &ci) in c.iter().enumerate() {
            next[i] -= ci * r;
            next[i + 1] += ci;
        }
        c = next;
    }
    c
}

// ---------------------------------------------------------------------------
// Generic matrix-polynomial operator (monic: coefficient of λ^k is I)
// ---------------------------------------------------------------------------

/// T(λ) = Σ_{i=0}^{k} λ^i A_i, with A_k = I and A_i real symmetric.
struct PolyOperator {
    n: usize,
    /// `coeffs[i]` is A_i row-major (n×n); `coeffs[k] == I`.
    coeffs: Vec<Vec<f64>>,
}

impl PolyOperator {
    fn new(n: usize, coeffs: Vec<Vec<f64>>) -> Self {
        let k = coeffs.len() - 1;
        for i in 0..n {
            assert!((coeffs[k][i * n + i] - 1.0).abs() < 1e-14, "operator must be monic");
        }
        PolyOperator { n, coeffs }
    }

    fn degree(&self) -> usize {
        self.coeffs.len() - 1
    }

    /// Evaluate T(λ)·x by Horner's rule.
    fn eval(&self, lambda: f64, x: &[f64]) -> Vec<f64> {
        let k = self.degree();
        let mut y = matvec(&self.coeffs[k], x, self.n);
        for i in (0..k).rev() {
            for v in y.iter_mut() {
                *v *= lambda;
            }
            let ai = matvec(&self.coeffs[i], x, self.n);
            for j in 0..self.n {
                y[j] += ai[j];
            }
        }
        y
    }

    /// Full-size companion matrix of the (monic) matrix polynomial, size k·n.
    fn full_companion(&self) -> MatrixFull<f64> {
        let n = self.n;
        let k = self.degree();
        let km = k * n;
        let mut c = MatrixFull::new([km, km], 0.0);
        for t in 0..k {
            let a = &self.coeffs[k - 1 - t];
            for r in 0..n {
                for cc in 0..n {
                    c[[r, t * n + cc]] = -a[r * n + cc];
                }
            }
        }
        for rr in 1..k {
            for r in 0..n {
                c[[rr * n + r, (rr - 1) * n + r]] = 1.0;
            }
        }
        c
    }

    /// Reference spectrum from the full companion linearisation.
    fn reference_eigenvalues(&self) -> Vec<Complex<f64>> {
        let c = self.full_companion();
        let (_, wr, wi, _, _, info) = _dgeev(&c, 'N', 'N');
        assert_eq!(info, 0, "reference dgeev failed");
        (0..wr.len()).map(|j| Complex::new(wr[j], wi[j])).collect()
    }
}

impl NlepOperator for PolyOperator {
    fn dim(&self) -> usize {
        self.n
    }

    fn projected_solve(
        &self,
        q: &MatrixFull<f64>,
        _lambda_init: &[f64],
    ) -> (Vec<Complex<f64>>, MatrixFull<f64>) {
        let n = self.n;
        let m = q.size[1];
        let k = self.degree();

        // Projected coefficients B_i = Qᵀ A_i Q  (m×m).  Because Q is
        // orthonormal and A_k = I, B_k = I, so the projected problem is monic.
        let mut b = vec![vec![0.0; m * m]; k + 1];
        for i in 0..=k {
            // aq = A_i · Q  (n×m)
            let mut aq = vec![0.0; n * m];
            for col in 0..m {
                for r in 0..n {
                    let mut s = 0.0;
                    for cc in 0..n {
                        s += self.coeffs[i][r * n + cc] * q[[cc, col]];
                    }
                    aq[r * m + col] = s;
                }
            }
            // B_i = Qᵀ · aq
            for p in 0..m {
                for r in 0..m {
                    let mut s = 0.0;
                    for a in 0..n {
                        s += q[[a, p]] * aq[a * m + r];
                    }
                    b[i][p * m + r] = s;
                }
            }
        }

        // Companion matrix of the projected monic matrix polynomial.
        let km = k * m;
        let mut c = MatrixFull::new([km, km], 0.0);
        for t in 0..k {
            for r in 0..m {
                for cc in 0..m {
                    c[[r, t * m + cc]] = -b[k - 1 - t][r * m + cc];
                }
            }
        }
        for rr in 1..k {
            for r in 0..m {
                c[[rr * m + r, (rr - 1) * m + r]] = 1.0;
            }
        }

        let (_, wr, wi, _, vr, info) = _dgeev(&c, 'N', 'V');
        assert_eq!(info, 0, "projected dgeev failed");

        let mut lambdas = Vec::with_capacity(km);
        for j in 0..km {
            lambdas.push(Complex::new(wr[j], wi[j]));
        }

        // Ritz vectors: the last block of the companion eigenvector is the
        // coefficient vector y; x = Q·y.  (Complex pairs share real/imag
        // storage; only the real part is used, matching the real-spectrum use
        // case targeted by these tests.)
        let mut ritz = MatrixFull::new([n, km], 0.0);
        for j in 0..km {
            let mut y = vec![0.0; m];
            for a in 0..m {
                y[a] = vr[[(k - 1) * m + a, j]];
            }
            for r in 0..n {
                let mut s = 0.0;
                for a in 0..m {
                    s += q[[r, a]] * y[a];
                }
                ritz[[r, j]] = s;
            }
        }
        (lambdas, ritz)
    }

    fn t_real(&self, lambda: f64, x: &[f64]) -> Vec<f64> {
        self.eval(lambda, x)
    }

    fn solve_shifted(&self, _node_index: usize, z: Complex<f64>, rhs: &[f64]) -> Vec<f64> {
        let n = self.n;
        let n2 = 2 * n;
        let k = self.degree();

        // Complex T(z) = Σ z^i A_i, split into real/imaginary parts.
        let mut tr = vec![0.0; n * n];
        let mut ti = vec![0.0; n * n];
        let mut zp = Complex::new(1.0, 0.0);
        for i in 0..=k {
            for idx in 0..n * n {
                tr[idx] += zp.re * self.coeffs[i][idx];
                ti[idx] += zp.im * self.coeffs[i][idx];
            }
            zp *= z;
        }

        // Real embedding  M₂ = [[Tr, −Ti], [Ti, Tr]].
        let mut m2 = MatrixFull::new([n2, n2], 0.0);
        for r in 0..n {
            for c in 0..n {
                m2[[r, c]] = tr[r * n + c];
                m2[[r, n + c]] = -ti[r * n + c];
                m2[[n + r, c]] = ti[r * n + c];
                m2[[n + r, n + c]] = tr[r * n + c];
            }
        }
        let inv = _dinverse(&m2).expect("shifted system is singular on the contour");

        // Solve M₂·[Re u; Im u] = [rhs; 0].
        let mut out = vec![0.0; n2];
        for r in 0..n2 {
            let mut s = 0.0;
            for c in 0..n {
                s += inv[[r, c]] * rhs[c];
            }
            out[r] = s;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Contour / driver helpers
// ---------------------------------------------------------------------------

/// Trapezoidal nodes on the circle z(θ) = centre + r·exp(iθ); weight
/// w_j = (z_j − centre)/n_quad (same convention as the BSE driver).
fn circle_nodes(centre: f64, radius: f64, n_quad: usize) -> Vec<ContourNode> {
    (0..n_quad)
        .map(|j| {
            let theta = 2.0 * std::f64::consts::PI * j as f64 / n_quad as f64;
            let (ct, st) = (theta.cos(), theta.sin());
            ContourNode {
                z_re: centre + radius * ct,
                z_im: radius * st,
                w_re: radius * ct / n_quad as f64,
                w_im: radius * st / n_quad as f64,
            }
        })
        .collect()
}

/// Random orthonormal initial subspace (n × m0).
fn random_subspace(n: usize, m0: usize, seed: u64) -> MatrixFull<f64> {
    let mut rng = Lcg::new(seed);
    let mut q = MatrixFull::new([n, m0], 0.0);
    for j in 0..m0 {
        for i in 0..n {
            q[[i, j]] = rng.normal();
        }
    }
    qr_orthonormalise(&q)
}

/// Linear standard operator T(λ) = λI − A with A = U·diag(spectrum)·Uᵀ, so the
/// exact spectrum is `spectrum` regardless of the random orthogonal U.
fn linear_operator_from_spectrum(spectrum: &[f64], seed: u64) -> PolyOperator {
    let n = spectrum.len();
    let u = random_orthogonal(n, seed);
    let mut a = vec![0.0; n * n];
    for r in 0..n {
        for c in 0..n {
            let mut s = 0.0;
            for t in 0..n {
                s += u[r * n + t] * spectrum[t] * u[c * n + t];
            }
            a[r * n + c] = s;
        }
    }
    let mut a0 = a.clone();
    for v in a0.iter_mut() {
        *v = -*v;
    }
    PolyOperator::new(n, vec![a0, identity(n)])
}

/// Reference real eigenvalues inside the closed contour |λ − centre| ≤ radius.
fn reference_in_window(op: &PolyOperator, centre: f64, radius: f64) -> Vec<f64> {
    let mut v: Vec<f64> = op
        .reference_eigenvalues()
        .into_iter()
        .filter(|z| z.im.abs() < 1e-8 && (z.re - centre).abs() <= radius)
        .map(|z| z.re)
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

/// Assert that the found eigenvalues equal the reference set and that every
/// returned residual is below `res_tol`.
fn assert_found_all(
    label: &str,
    found: &[f64],
    residuals: &[f64],
    expected: &[f64],
    eig_tol: f64,
    res_tol: f64,
) {
    let mut f = found.to_vec();
    f.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(
        f.len(),
        expected.len(),
        "{label}: found {} eigenvalues {:?}, expected {} {:?}",
        f.len(),
        f,
        expected.len(),
        expected
    );
    for (a, b) in f.iter().zip(expected.iter()) {
        assert!(
            (a - b).abs() < eig_tol,
            "{label}: eigenvalue mismatch {a} vs {b} (all found {f:?}, expected {expected:?})"
        );
    }
    for (k, r) in residuals.iter().enumerate() {
        assert!(
            *r < res_tol,
            "{label}: residual[{k}] = {r:.3e} exceeds {res_tol:.1e}"
        );
    }
    eprintln!("{label}: OK — found all {} in-window eigenvalues", f.len());
}

// ===========================================================================
// Test 1: linear standard problem T(λ) = λI − A, A symmetric
// ===========================================================================

#[test]
fn nlfeast_linear_standard_problem() {
    let n = 40;
    let dim = n;
    // A = U·diag(0..n-1)·Uᵀ  → spectrum is exactly {0,1,...,n-1}.
    let u = random_orthogonal(n, 12345);
    let mut a = vec![0.0; n * n];
    for r in 0..n {
        for c in 0..n {
            let mut s = 0.0;
            for t in 0..n {
                s += u[r * n + t] * (t as f64) * u[c * n + t];
            }
            a[r * n + c] = s;
        }
    }
    let mut a0 = a.clone();
    for v in a0.iter_mut() {
        *v = -*v;
    }
    let op = PolyOperator::new(n, vec![a0, identity(n)]);

    let centre = 20.0;
    let radius = 2.5;
    let expected: Vec<f64> = (0..n).map(|i| i as f64).filter(|d| (d - centre).abs() <= radius).collect();
    assert_eq!(expected.len(), 5);

    let q0 = random_subspace(dim, 10, 999);
    let nodes = circle_nodes(centre, radius, 16);
    let res = nlfeast(&op, q0, vec![centre; 10], &nodes, centre, radius, 100, 1e-10);

    assert_found_all(
        "linear standard",
        &res.eigenvalues,
        &res.residuals,
        &expected,
        1e-7,
        1e-6,
    );
}

// ===========================================================================
// Test 2: Hermitian quadratic eigenvalue problem (mass-spring type)
//         T(λ) = λ²I + λA₁ + A₀, all-real spectrum
// ===========================================================================

#[test]
fn nlfeast_hermitian_quadratic() {
    let n = 30;
    let t = tridiag_3_minus1(n);
    let mut a1 = t.clone();
    for v in a1.iter_mut() {
        *v *= 2.0;
    }
    let mut a0 = t.clone();
    for v in a0.iter_mut() {
        *v *= 0.5;
    }
    let op = PolyOperator::new(n, vec![a0, a1, identity(n)]);

    let centre = -5.0;
    let radius = 0.7;
    let expected = reference_in_window(&op, centre, radius);
    assert_eq!(expected.len(), 4, "test window should contain 4 eigenvalues");

    let q0 = random_subspace(n, 10, 4242);
    let nodes = circle_nodes(centre, radius, 16);
    let res = nlfeast(&op, q0, vec![centre; 10], &nodes, centre, radius, 100, 1e-10);

    assert_found_all(
        "hermitian quadratic",
        &res.eigenvalues,
        &res.residuals,
        &expected,
        1e-6,
        1e-7,
    );
}

// ===========================================================================
// Test 3: quartic eigenvalue problem, real spectrum
//         T(λ) = λ⁴I + λ³A₃ + λ²A₂ + λA₁ + A₀
// ===========================================================================

#[test]
fn nlfeast_quartic_real_spectrum() {
    let n = 12;
    let u = random_orthogonal(n, 777);

    // Per-eigenvector scalar monic quartic with four real roots.
    let roots_per_index: Vec<[f64; 4]> = (0..n)
        .map(|r| {
            let r = r as f64;
            [r, r + 9.37, r + 18.74, r + 28.11]
        })
        .collect();

    // coefficient-of-λ^i for eigenvalue index r
    let mut coeff_by_eig = vec![vec![0.0; n]; 5];
    for (r, roots) in roots_per_index.iter().enumerate() {
        let c = poly_from_roots(roots);
        for i in 0..5 {
            coeff_by_eig[i][r] = c[i];
        }
    }

    // A_i = U·diag(coeff_by_eig[i])·Uᵀ
    let mut coeffs: Vec<Vec<f64>> = Vec::new();
    for i in 0..5 {
        let mut a = vec![0.0; n * n];
        for r in 0..n {
            for c in 0..n {
                let mut s = 0.0;
                for t in 0..n {
                    s += u[r * n + t] * coeff_by_eig[i][t] * u[c * n + t];
                }
                a[r * n + c] = s;
            }
        }
        coeffs.push(a);
    }
    let op = PolyOperator::new(n, coeffs);

    let centre = 20.0;
    let radius = 2.0;
    let expected = reference_in_window(&op, centre, radius);
    assert_eq!(expected.len(), 7, "test window should contain 7 eigenvalues");

    let q0 = random_subspace(n, 12, 5150);
    let nodes = circle_nodes(centre, radius, 16);
    let res = nlfeast(&op, q0, vec![centre; 12], &nodes, centre, radius, 200, 1e-10);

    assert_found_all(
        "quartic",
        &res.eigenvalues,
        &res.residuals,
        &expected,
        1e-6,
        1e-7,
    );
}

// ===========================================================================
// Test 4: random symmetric quartic (non-diagonalisable-by-construction).
//         The search circle is chosen so that it contains only real
//         eigenvalues (verified against the full companion linearisation).
// ===========================================================================

#[test]
fn nlfeast_random_symmetric_quartic() {
    let n = 10;
    let mut rng = Lcg::new(7);
    let mut coeffs: Vec<Vec<f64>> = Vec::new();
    for _ in 0..4 {
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let v = rng.normal();
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        }
        coeffs.push(a);
    }
    coeffs.push(identity(n));
    let op = PolyOperator::new(n, coeffs);

    let centre = 1.5;
    let radius = 0.5;
    let expected = reference_in_window(&op, centre, radius);
    assert_eq!(expected.len(), 2, "test window should contain 2 real eigenvalues");

    // Also make sure no *complex* eigenvalue hides inside the circle.
    for z in op.reference_eigenvalues() {
        if (z - Complex::new(centre, 0.0)).norm() <= radius {
            assert!(z.im.abs() < 1e-8, "window contains complex eigenvalue {z}");
        }
    }

    let q0 = random_subspace(n, 6, 31415);
    let nodes = circle_nodes(centre, radius, 16);
    let res = nlfeast(&op, q0, vec![centre; 6], &nodes, centre, radius, 200, 1e-10);

    assert_found_all(
        "random quartic",
        &res.eigenvalues,
        &res.residuals,
        &expected,
        1e-6,
        1e-7,
    );
}

// ===========================================================================
// Test 5: several disjoint windows on the same linear problem — every window
//         must independently return exactly its own eigenvalues.
// ===========================================================================

#[test]
fn nlfeast_multiple_disjoint_windows() {
    let n = 40;
    let u = random_orthogonal(n, 2024);
    let mut a = vec![0.0; n * n];
    for r in 0..n {
        for c in 0..n {
            let mut s = 0.0;
            for t in 0..n {
                s += u[r * n + t] * (t as f64) * u[c * n + t];
            }
            a[r * n + c] = s;
        }
    }
    let mut a0 = a.clone();
    for v in a0.iter_mut() {
        *v = -*v;
    }
    let op = PolyOperator::new(n, vec![a0, identity(n)]);

    for &(centre, radius) in &[(5.0, 1.4), (20.0, 2.4), (30.0, 1.4)] {
        let expected: Vec<f64> = (0..n)
            .map(|i| i as f64)
            .filter(|d| (d - centre).abs() <= radius)
            .collect();
        assert!(!expected.is_empty());
        let m0 = expected.len() + 5;
        let q0 = random_subspace(n, m0, 1000 + centre as u64);
        let nodes = circle_nodes(centre, radius, 16);
        let res = nlfeast(&op, q0, vec![centre; m0], &nodes, centre, radius, 200, 1e-10);
        assert_found_all(
            &format!("window centre={centre}"),
            &res.eigenvalues,
            &res.residuals,
            &expected,
            1e-7,
            1e-6,
        );
    }
}

// ===========================================================================
// Test 6: wide window — many eigenvalues inside a single contour.
//         Stresses the Rayleigh–Ritz selection and the accuracy of the
//         contour quadrature (cf. Table 2 of the NLFEAST paper).
// ===========================================================================

#[test]
fn nlfeast_wide_window_many_eigenvalues() {
    let n = 40;
    let spectrum: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let op = linear_operator_from_spectrum(&spectrum, 555);

    let centre = 20.0;
    let radius = 5.5;
    // 15,16,...,25 → 11 eigenvalues with a 0.5 margin from the contour.
    let expected: Vec<f64> = (15..=25).map(|i| i as f64).collect();
    assert_eq!(expected.len(), 11);

    let m0 = 16; // > number of expected interior eigenvalues
    let q0 = random_subspace(n, m0, 111);
    let nodes = circle_nodes(centre, radius, 32);
    let res = nlfeast(&op, q0, vec![centre; m0], &nodes, centre, radius, 300, 1e-10);

    assert_found_all(
        "wide window",
        &res.eigenvalues,
        &res.residuals,
        &expected,
        1e-7,
        1e-6,
    );
}

// ===========================================================================
// Test 7: tightly clustered (nearly degenerate) eigenvalues.  A contour of
//         radius 1e-2 must resolve four eigenvalues separated by 1e-4.
// ===========================================================================

#[test]
fn nlfeast_clustered_spectrum() {
    let n = 30;
    let mut spectrum: Vec<f64> = (0..n).map(|i| i as f64).collect();
    spectrum[10] = 10.0;
    spectrum[11] = 10.0001;
    spectrum[12] = 10.0002;
    spectrum[13] = 10.0003;
    let op = linear_operator_from_spectrum(&spectrum, 888);

    let centre = 10.00015;
    let radius = 1e-2;
    let expected = vec![10.0, 10.0001, 10.0002, 10.0003];

    let m0 = 8;
    let q0 = random_subspace(n, m0, 222);
    let nodes = circle_nodes(centre, radius, 16);
    let res = nlfeast(&op, q0, vec![centre; m0], &nodes, centre, radius, 300, 1e-10);

    assert_found_all(
        "clustered",
        &res.eigenvalues,
        &res.residuals,
        &expected,
        1e-7,
        1e-6,
    );
}
