//! Cheap conditioning probe for the overlap matrix.
//!
//! `prepare_necessary_integrals` needs two numbers from the overlap matrix S:
//! its lowest eigenvalue (a diagnostic that flags a nearly linearly dependent
//! basis) and the number of independent basis functions.  The previous
//! implementation obtained both from a full dense symmetric eigendecomposition
//! (O(n^3) plus an n x n eigenvector block); only the lowest eigenvalue was
//! ever consumed.  A Lanczos run with full reorthogonalization reaches the
//! same diagnostic in O(m n^2) with m ~ 20.

use rest_tensors::{BasicMatrix, MatrixUpper};

/// Cyclic Jacobi eigenvalues of a small dense symmetric matrix (row-major
/// `m x m` in `a`).  Used for the m x m Lanczos tridiagonal, where m is tiny.
fn jacobi_eigenvalues(a: &mut [f64], m: usize) -> Vec<f64> {
    let at = |i: usize, j: usize| i * m + j;
    for _sweep in 0..100 {
        let mut off = 0.0_f64;
        for i in 0..m {
            for j in (i + 1)..m {
                off += a[at(i, j)] * a[at(i, j)];
            }
        }
        if off <= 1.0e-30 {
            break;
        }
        for p in 0..m {
            for q in (p + 1)..m {
                let apq = a[at(p, q)];
                if apq.abs() < 1.0e-18 {
                    continue;
                }
                let theta = (a[at(q, q)] - a[at(p, p)]) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let app = a[at(p, p)];
                let aqq = a[at(q, q)];
                a[at(p, p)] = app - t * apq;
                a[at(q, q)] = aqq + t * apq;
                a[at(p, q)] = 0.0;
                a[at(q, p)] = 0.0;
                for r in 0..m {
                    if r != p && r != q {
                        let arp = a[at(r, p)];
                        let arq = a[at(r, q)];
                        a[at(r, p)] = c * arp - s * arq;
                        a[at(p, r)] = a[at(r, p)];
                        a[at(r, q)] = s * arp + c * arq;
                        a[at(q, r)] = a[at(r, q)];
                    }
                }
            }
        }
    }
    (0..m).map(|i| a[at(i, i)]).collect()
}

/// Lowest eigenvalue of the symmetric matrix stored in upper-triangular form.
///
/// `n` is the number of basis functions (the stored matrix is `n x n`).
/// Returns `f64::INFINITY` when the stored matrix is empty, so the caller's
/// "not singular" branch is taken.
pub fn lowest_eigenvalue(ovlp: &MatrixUpper<f64>, n: usize) -> f64 {
    let full = match ovlp.to_matrixfull() {
        Some(m) => m,
        None => return f64::INFINITY,
    };
    let ncol = full.size[1];
    let n = n.min(full.size[0]).min(ncol);
    if n == 0 {
        return f64::INFINITY;
    }
    let data = &full.data;

    // y = S x, S symmetric (column-major storage: S[i,j] at i + j*ncol).
    let matvec = |x: &[f64]| -> Vec<f64> {
        let mut y = vec![0.0_f64; n];
        for j in 0..n {
            let xj = x[j];
            if xj == 0.0 {
                continue;
            }
            let col = &data[j * ncol..j * ncol + n];
            for i in 0..n {
                y[i] += col[i] * xj;
            }
        }
        y
    };

    let m = 20.min(n);
    let mut q: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
    let mut alpha = vec![0.0_f64; m];
    let mut beta = vec![0.0_f64; m];
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut vnorm = 1.0_f64;

    for k in 0..m {
        let mut qk: Vec<f64> = v.iter().map(|x| x / vnorm).collect();
        // full reorthogonalization (two passes) keeps the basis orthonormal
        for _pass in 0..2 {
            for prev in q.iter() {
                let d: f64 = qk.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
                if d != 0.0 {
                    for i in 0..n {
                        qk[i] -= d * prev[i];
                    }
                }
            }
        }
        let nrm2: f64 = qk.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nrm2 > 0.0 {
            for x in qk.iter_mut() {
                *x /= nrm2;
            }
        }
        let w = matvec(&qk);
        let a_k: f64 = qk.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        alpha[k] = a_k;
        q.push(qk.clone());
        if k + 1 == m {
            break;
        }
        let mut w2 = w;
        for i in 0..n {
            w2[i] -= a_k * qk[i];
        }
        if k > 0 {
            let b_prev = beta[k - 1];
            let prev = &q[k - 1];
            for i in 0..n {
                w2[i] -= b_prev * prev[i];
            }
        }
        let bnorm = w2.iter().map(|x| x * x).sum::<f64>().sqrt();
        beta[k] = bnorm;
        if bnorm < 1.0e-14 {
            alpha.truncate(k + 1);
            beta.truncate(k + 1);
            return ritz_min(&alpha, &beta);
        }
        v = w2;
        vnorm = bnorm;
    }
    alpha.truncate(m);
    beta.truncate(m);
    ritz_min(&alpha, &beta)
}

/// Smallest Ritz value of the Lanczos tridiagonal built from `alpha`/`beta`.
fn ritz_min(alpha: &[f64], beta: &[f64]) -> f64 {
    let m = alpha.len();
    let mut t = vec![0.0_f64; m * m];
    for i in 0..m {
        t[i * m + i] = alpha[i];
        if i + 1 < m {
            t[i * m + i + 1] = beta[i];
            t[(i + 1) * m + i] = beta[i];
        }
    }
    let eig = jacobi_eigenvalues(&mut t, m);
    eig.iter().cloned().fold(f64::INFINITY, f64::min)
}
