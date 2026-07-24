//! 3×3 real-symmetric eigensolver. Verbatim port of Psi4's
//! `vecutil.diagonalize3x3symmat` (cyclic Jacobi, 50-iteration cap).
//! Pure std. Used to diagonalize the inertia tensor for principal axes.

use super::matrix::Matrix3;

/// Diagonalize a real symmetric 3×3 matrix. Returns `(eigenvalues, Q)` where
/// the **columns** of `Q` are the eigenvectors: eigenvector `i` is
/// `[Q[0][i], Q[1][i], Q[2][i]]`. (Matches the reference: `Q[r][p]`.)
pub fn diagonalize3x3symmat(m: &Matrix3) -> ([f64; 3], Matrix3) {
    let mut a = *m; // working copy (only upper triangle is used)
    let mut eig = super::matrix::identity();
    let mut w = [a[0][0], a[1][1], a[2][2]];

    // (reference computes SQR(tr(A)) as `sd` but never uses it; omitted.)

    for n_iter in 0..50 {
        // test for convergence: sum of |off-diagonal|
        let mut so = 0.0;
        for p in 0..3 {
            for q in (p + 1)..3 {
                so += a[p][q].abs();
            }
        }
        if so == 0.0 {
            return (w, eig);
        }

        let thresh = if n_iter < 4 { 0.2 * so / 9.0 } else { 0.0 };

        for p in 0..3 {
            for q in (p + 1)..3 {
                let g = 100.0 * a[p][q].abs();
                if n_iter > 4
                    && (w[p].abs() + g == w[p].abs())
                    && (w[q].abs() + g == w[q].abs())
                {
                    a[p][q] = 0.0;
                } else if a[p][q].abs() > thresh {
                    let h = w[q] - w[p];
                    let t;
                    if h.abs() + g == h.abs() {
                        t = a[p][q] / h;
                    } else {
                        let theta = 0.5 * h / a[p][q];
                        if theta < 0.0 {
                            t = -1.0 / ((1.0 + theta * theta).sqrt() - theta);
                        } else {
                            t = 1.0 / ((1.0 + theta * theta).sqrt() + theta);
                        }
                    }
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = t * c;
                    let z = t * a[p][q];

                    a[p][q] = 0.0;
                    w[p] -= z;
                    w[q] += z;

                    for r in 0..p {
                        let tr = a[r][p];
                        a[r][p] = c * tr - s * a[r][q];
                        a[r][q] = s * tr + c * a[r][q];
                    }
                    for r in (p + 1)..q {
                        let tr = a[p][r];
                        a[p][r] = c * tr - s * a[r][q];
                        a[r][q] = s * tr + c * a[r][q];
                    }
                    for r in (q + 1)..3 {
                        let tr = a[p][r];
                        a[p][r] = c * tr - s * a[q][r];
                        a[q][r] = s * tr + c * a[q][r];
                    }
                    for r in 0..3 {
                        let tr = eig[r][p];
                        eig[r][p] = c * tr - s * eig[r][q];
                        eig[r][q] = s * tr + c * eig[r][q];
                    }
                }
            }
        }
    }

    // not converged within 50 iterations; return best-effort (reference returns None)
    (w, eig)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::matrix::matmul;

    #[test]
    fn diagonal_diag_matrix() {
        let m = [[2.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, -3.0]];
        let (w, q) = diagonalize3x3symmat(&m);
        // eigenvalues (unordered): 5, 2, -3
        let mut ws = w;
        ws.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((ws[0] - (-3.0)).abs() < 1e-12);
        assert!((ws[1] - 2.0).abs() < 1e-12);
        assert!((ws[2] - 5.0).abs() < 1e-12);
        // Q is orthogonal
        let qt = super::super::matrix::transpose(&q);
        let prod = matmul(&q, &qt);
        for i in 0..3 {
            for j in 0..3 {
                let target = if i == j { 1.0 } else { 0.0 };
                assert!((prod[i][j] - target).abs() < 1e-12, "Q not orthogonal");
            }
        }
    }

    #[test]
    fn diagonal_known_symmetric() {
        // [[2,1,0],[1,2,0],[0,0,3]] -> eigenvalues 1, 3, 3
        let m = [[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
        let (w, _q) = diagonalize3x3symmat(&m);
        let mut ws = w;
        ws.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((ws[0] - 1.0).abs() < 1e-10);
        assert!((ws[1] - 3.0).abs() < 1e-10);
        assert!((ws[2] - 3.0).abs() < 1e-10);
    }
}
