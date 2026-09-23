// ============================================================================
// Nonlinear FEAST (NLFEAST) kernel for nonlinear eigenvalue problems
//
//   T(λ) x = 0,   λ ∈ search contour C
//
// This is the context-independent part of the algorithm of Gavin, Międlar and
// Polizzi, "FEAST eigensolver for nonlinear eigenvalue problems",
// J. Comput. Sci. (2018); see docs/FEASTnonlinear.md.  It follows Algorithm 1:
//
//   Step 0  orthonormal initial subspace Q (built by the caller)
//   Step 1  solve the projected NLEVP  Qᴴ T(λ) Q y = 0      (operator)
//   Step 2  keep the m₀ Ritz pairs whose λ is closest to the contour centre
//   Step 3  stop when every in-contour Ritz pair satisfies ‖T(λ)x‖ ≤ tol
//   Step 4  contour update
//             Q ← Σ_j ω_j (X − T(z_j)⁻¹ T(X,Λ)) (z_j I − Λ)⁻¹
//   Step 5  QR-orthonormalise Q and repeat
//
// Everything specific to a physical problem (how T(λ) is applied, how the
// projected problem is solved, how the shifted systems are solved) is supplied
// through the `NlepOperator` trait.  This keeps the FEAST subspace iteration
// itself reusable and independently testable.
// ============================================================================

use num::Complex;
use rest_tensors::matrix::{MathMatrix, MatrixFull};

/// A single quadrature node z_j = z_re + i·z_im with complex weight
/// ω_j = w_re + i·w_im on the search contour.
#[derive(Clone, Copy, Debug)]
pub struct ContourNode {
    pub z_re: f64,
    pub z_im: f64,
    pub w_re: f64,
    pub w_im: f64,
}

/// Result of the NLFEAST solver.
pub struct NLFeastResult {
    /// Eigenvalues located inside the search contour, sorted by distance to
    /// the contour centre.
    pub eigenvalues: Vec<f64>,
    /// Corresponding eigenvectors (one column per eigenvalue).
    pub eigenvectors: MatrixFull<f64>,
    /// Number of eigenvalues found inside the contour.
    pub n_found: usize,
    /// Number of outer FEAST iterations performed.
    pub iterations: usize,
    /// ‖T(λ_k) x_k‖ for every returned eigenpair.
    pub residuals: Vec<f64>,
}

/// Operator interface consumed by the NLFEAST kernel.
///
/// A physical problem implements this trait to describe T(λ); the kernel owns
/// the subspace iteration, selection, convergence test and contour update.
pub trait NlepOperator {
    /// Dimension n of the eigenproblem.
    fn dim(&self) -> usize;

    /// Solve the projected nonlinear eigenproblem `Qᴴ T(λ) Q y = 0` in the
    /// current orthonormal basis `q` (n × m).  `lambda_init` holds one current
    /// eigenvalue estimate per column of `q`.
    ///
    /// Returns all candidate eigenvalues together with the corresponding Ritz
    /// vectors `q·y` (n × n_candidates).  The kernel performs the selection of
    /// the m₀ Ritz pairs closest to the contour centre, so implementations may
    /// return more candidates than the subspace dimension.
    fn projected_solve(
        &self,
        q: &MatrixFull<f64>,
        lambda_init: &[f64],
    ) -> (Vec<Complex<f64>>, MatrixFull<f64>);

    /// Apply T(λ) to a real vector at a real λ (the residual operator).
    fn t_real(&self, lambda: f64, x: &[f64]) -> Vec<f64>;

    /// Solve the shifted linear system T(z) u = rhs for a complex shift z.
    ///
    /// The solution is returned in the real-embedded layout `[Re(u); Im(u)]`
    /// of length 2n, matching the real-embedded complex arithmetic used by the
    /// contour integral.  `node_index` identifies which contour node is being
    /// processed (some operators hold node-dependent data).
    fn solve_shifted(&self, node_index: usize, z: Complex<f64>, rhs: &[f64]) -> Vec<f64>;
}

/// Modified Gram–Schmidt QR orthonormalisation.  Linearly dependent columns are
/// dropped, so the returned matrix may have fewer columns than the input.
pub fn qr_orthonormalise(a: &MatrixFull<f64>) -> MatrixFull<f64> {
    let n = a.size[0];
    let k = a.size[1];
    let mut q = a.clone();

    for j in 0..k {
        for i in 0..j {
            let mut dot = 0.0;
            for ii in 0..n {
                dot += q[[ii, j]] * q[[ii, i]];
            }
            for ii in 0..n {
                q[[ii, j]] -= dot * q[[ii, i]];
            }
        }
        let mut nrm = 0.0;
        for ii in 0..n {
            nrm += q[[ii, j]] * q[[ii, j]];
        }
        nrm = nrm.sqrt();
        if nrm > 1e-14 {
            for ii in 0..n {
                q[[ii, j]] /= nrm;
            }
        } else {
            for ii in 0..n {
                q[[ii, j]] = 0.0;
            }
        }
    }

    let mut valid = Vec::new();
    for j in 0..k {
        let mut nrm = 0.0;
        for i in 0..n {
            nrm += q[[i, j]] * q[[i, j]];
        }
        if nrm.sqrt() > 1e-14 {
            valid.push(j);
        }
    }
    if valid.len() == k {
        return q;
    }
    let nk = valid.len();
    let mut qt = MatrixFull::new([n, nk], 0.0);
    for (pj, &j) in valid.iter().enumerate() {
        for i in 0..n {
            qt[[i, pj]] = q[[i, j]];
        }
    }
    qt
}

/// Collect the converged in-contour Ritz pairs into an `NLFeastResult`.
fn build_final_result(
    n: usize,
    lambda_cur: &[f64],
    x_mat: &MatrixFull<f64>,
    residuals: &[f64],
    inside_flags: &[bool],
    centre: f64,
    radius: f64,
) -> NLFeastResult {
    let mut pairs: Vec<(f64, f64, usize)> = Vec::new();
    for (j, &inside) in inside_flags.iter().enumerate() {
        if !inside {
            continue;
        }
        let dist = (lambda_cur[j] - centre).abs();
        if dist <= radius * 1.05 {
            pairs.push((dist, lambda_cur[j], j));
        }
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let n_found = pairs.len();
    let mut eigvals = Vec::with_capacity(n_found);
    let mut eigvecs = MatrixFull::new([n, n_found], 0.0);
    let mut final_res = Vec::with_capacity(n_found);
    for (k, (_, lam, orig_j)) in pairs.iter().enumerate() {
        eigvals.push(*lam);
        for i in 0..n {
            eigvecs[[i, k]] = x_mat[[i, *orig_j]];
        }
        final_res.push(residuals[*orig_j]);
    }
    NLFeastResult {
        eigenvalues: eigvals,
        eigenvectors: eigvecs,
        n_found,
        iterations: 0,
        residuals: final_res,
    }
}

/// Ritz pairs whose real-axis residual exceeds this value are treated as
/// *spurious* and are not counted as eigenvalues inside the contour.  The
/// nonlinear FEAST contour integral can produce such spurious Ritz values
/// (they have no counterpart in the exact spectrum); the original paper
/// excludes them the same way (Gavin–Międlar–Polizzi, §5, "excluding spurious
/// eigenvalues").  Genuine but not-yet-converged eigenvalues have residuals
/// below this bound and still block the convergence test.
const SPURIOUS_RESIDUAL_TOL: f64 = 1e-2;

/// Run the nonlinear FEAST subspace iteration.
///
/// * `op` — the problem-specific operator T(λ).
/// * `q_mat` — initial subspace (n × m₀), ideally already orthonormalised.
/// * `lambda_init` — one initial eigenvalue estimate per column of `q_mat`.
/// * `nodes` — quadrature nodes/weights for the search contour.
/// * `centre`, `radius` — circular search contour |λ − centre| ≤ radius.
/// * `max_iter` — maximum number of outer iterations.
/// * `tol` — convergence tolerance on the real-axis residual ‖T(λ)x‖.
pub fn nlfeast<O: NlepOperator + ?Sized>(
    op: &O,
    mut q_mat: MatrixFull<f64>,
    mut lambda_init: Vec<f64>,
    nodes: &[ContourNode],
    centre: f64,
    radius: f64,
    max_iter: usize,
    tol: f64,
) -> NLFeastResult {
    let n = op.dim();
    let centre_cplx = Complex::new(centre, 0.0);

    let mut n_iter = 0;
    let mut prev_lambda: Vec<f64> = Vec::new();
    let mut last_lambda_real: Vec<f64> = Vec::new();
    let mut last_x_mat = MatrixFull::new([n, 0], 0.0);
    let mut last_residuals: Vec<f64> = Vec::new();

    for it in 0..max_iter {
        n_iter = it + 1;
        let m0_eff = q_mat.size[1];
        if m0_eff == 0 {
            break;
        }
        // Truncate lambda_init to match compressed subspace size
        lambda_init.truncate(m0_eff);

        // ---- Step 1: Solve the projected NLEVP at current lambda_init ----
        let (all_lambda, all_x) = op.projected_solve(&q_mat, &lambda_init);
        if it == 0 {
            eprint!("  Initial eigenvalue estimates (first 8):");
            for j in 0..8.min(all_lambda.len()) {
                eprint!(" {:.4}", all_lambda[j].re);
            }
            eprintln!();
        }

        // ---- Step 2: Select m0_eff eigenvalues closest to contour centre ----
        let n_all = all_lambda.len();
        let mut with_dist: Vec<(f64, usize)> = (0..n_all)
            .map(|idx| ((all_lambda[idx] - centre_cplx).norm(), idx))
            .collect();
        with_dist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let lambda_cur: Vec<Complex<f64>> = with_dist[..m0_eff.min(n_all)]
            .iter()
            .map(|(_, idx)| all_lambda[*idx])
            .collect();
        let mut x_mat = MatrixFull::new([n, lambda_cur.len()], 0.0);
        for (pj, (_, idx)) in with_dist[..lambda_cur.len()].iter().enumerate() {
            for i in 0..n {
                x_mat[[i, pj]] = all_x[[i, *idx]];
            }
        }
        let m0_eff = lambda_cur.len();

        // ---- Step 3: Convergence check (real-axis residuals) ----
        let mut n_inside = 0;
        let mut n_inside_converged = 0;
        let mut max_resid_inside = 0.0_f64;
        let mut residuals = Vec::with_capacity(m0_eff);
        let mut inside_flags = vec![false; m0_eff];

        for j in 0..m0_eff {
            let lam = lambda_cur[j];
            let lam_real = lam.re;
            let xj: Vec<f64> = (0..n).map(|i| x_mat[[i, j]]).collect();
            let txj = op.t_real(lam_real, &xj);
            let res: f64 = txj.iter().map(|&v| v * v).sum::<f64>().sqrt();
            residuals.push(res);

            if (lam - centre_cplx).norm() <= radius {
                // Exclude spurious Ritz pairs (large residual) from the
                // interior set so they neither block convergence nor end up in
                // the final result.  Genuine, not-yet-converged pairs have a
                // residual below SPURIOUS_RESIDUAL_TOL and still block.
                if res <= SPURIOUS_RESIDUAL_TOL {
                    inside_flags[j] = true;
                    n_inside += 1;
                    if res <= tol {
                        n_inside_converged += 1;
                    }
                    if res > max_resid_inside {
                        max_resid_inside = res;
                    }
                }
            }
        }

        eprintln!(
            "NLFEAST iter {}: max ‖T(λ)x‖_inside = {:.2e}, inside {}/{}",
            n_iter, max_resid_inside, n_inside_converged, n_inside
        );

        if n_inside > 0 && n_inside_converged == n_inside {
            eprintln!("  → All {} interior eigenvalues converged.", n_inside);
            let mut result = build_final_result(
                n,
                &lambda_cur.iter().map(|l| l.re).collect::<Vec<_>>(),
                &x_mat,
                &residuals,
                &inside_flags,
                centre,
                radius,
            );
            result.iterations = n_iter;
            return result;
        }

        // Stagnation detection
        let lambda_real: Vec<f64> = lambda_cur.iter().map(|l| l.re).collect();
        let improved = if prev_lambda.len() == lambda_real.len() {
            let mut d = 0.0;
            for j in 0..lambda_real.len() {
                d += (lambda_real[j] - prev_lambda[j]).abs();
            }
            d / lambda_real.len() as f64
        } else {
            1.0
        };
        if n_iter > 3 && improved < 1e-12 && n_inside > 0 {
            eprintln!(
                "  → Eigenvalues stabilised (Δλ ≈ {:.2e}). Returning interior results.",
                improved
            );
            let mut result = build_final_result(
                n,
                &lambda_real,
                &x_mat,
                &residuals,
                &inside_flags,
                centre,
                radius,
            );
            result.iterations = n_iter;
            return result;
        }
        // Save best results for fallback
        last_lambda_real = lambda_real.clone();
        last_x_mat = x_mat.clone();
        last_residuals = residuals.clone();

        // Update lambda_init for the next projected solve
        lambda_init = lambda_real.clone();
        prev_lambda = lambda_real;

        // ---- Step 4: Contour integration to update the subspace ----
        //   Q_new = Σ_j w_j · (X − T(z_j)⁻¹·T(X,Λ)) · (z_j I − Λ)⁻¹
        //
        // First compute the block residual T(Λ)·X on the real axis, then solve
        // T(z_j)·U_j = T(Λ)·X for each node and accumulate.
        let mut tx_mat = MatrixFull::new([n, m0_eff], 0.0);
        for j in 0..m0_eff {
            let lam = lambda_cur[j].re;
            let xj: Vec<f64> = (0..n).map(|i| x_mat[[i, j]]).collect();
            let txj = op.t_real(lam, &xj);
            for i in 0..n {
                tx_mat[[i, j]] = txj[i];
            }
        }

        let mut q_new = MatrixFull::new([n, m0_eff], 0.0);

        for (node_index, node) in nodes.iter().enumerate() {
            let z = Complex::new(node.z_re, node.z_im);
            let mut u_re = MatrixFull::new([n, m0_eff], 0.0);
            let mut u_im = MatrixFull::new([n, m0_eff], 0.0);

            for col in 0..m0_eff {
                // RHS is always real: [tx_col; 0] in the 2N embedding
                let mut b_2n = vec![0.0; 2 * n];
                for i in 0..n {
                    b_2n[i] = tx_mat[[i, col]];
                }
                let x_2n = op.solve_shifted(node_index, z, &b_2n);
                for i in 0..n {
                    u_re[[i, col]] = x_2n[i];
                    u_im[[i, col]] = x_2n[n + i];
                }
            }

            // Accumulate: Q_new += w_j · (X − U_j) · (z_j I − Λ)⁻¹
            for col in 0..m0_eff {
                let lam = lambda_cur[col];
                let dz_re = node.z_re - lam.re;
                let dz_im = node.z_im - lam.im;
                let denom = dz_re * dz_re + dz_im * dz_im;
                if denom < 1e-30 {
                    continue;
                }
                let inv_re = dz_re / denom;
                let inv_im = -dz_im / denom;

                // Combined factor f = w_j · (z_j − λ)⁻¹
                let f_re = node.w_re * inv_re - node.w_im * inv_im;
                let f_im = node.w_re * inv_im + node.w_im * inv_re;

                for i in 0..n {
                    // (X − U_j): X is real, U_j complex
                    let dx_re = x_mat[[i, col]] - u_re[[i, col]];
                    let dx_im = -u_im[[i, col]];
                    q_new[[i, col]] += f_re * dx_re - f_im * dx_im;
                }
            }
        }

        // ---- Step 5: QR orthogonalise Q_new ----
        q_mat = qr_orthonormalise(&q_new);
        if q_mat.size[1] < m0_eff / 2 {
            eprintln!("  Warning: subspace collapsed, stopping.");
            break;
        }
    }

    // Fallback: return best available interior results from the last iteration
    if !last_lambda_real.is_empty() {
        let mut last_inside_flags = vec![false; last_lambda_real.len()];
        for (j, &lam) in last_lambda_real.iter().enumerate() {
            // Same spurious-pair exclusion as the main convergence test.
            if (lam - centre).abs() <= radius && last_residuals[j] <= SPURIOUS_RESIDUAL_TOL {
                last_inside_flags[j] = true;
            }
        }
        let mut result = build_final_result(
            n,
            &last_lambda_real,
            &last_x_mat,
            &last_residuals,
            &last_inside_flags,
            centre,
            radius,
        );
        result.iterations = n_iter;
        return result;
    }
    NLFeastResult {
        eigenvalues: vec![],
        eigenvectors: MatrixFull::new([n, 0], 0.0),
        n_found: 0,
        iterations: n_iter,
        residuals: vec![],
    }
}
