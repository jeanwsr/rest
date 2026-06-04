use rest_tensors::MatrixFull;

// -------------------------------
// p = a[order]*x^order + ... + a[1]*x + a[0]
// -------------------------------
#[inline]
fn poly_value1_f64(a: &[f64], order: usize, x: f64) -> f64 {
    let mut p = a[order];
    for i in 1..=order {
        p = p * x + a[order - i];
    }
    p
}

// -------------------------------
// R_dnode：括弧 + 割线/二分 混合迭代（与 libcint 等价）
// 输入 a[0..order] 为升幂系数；roots 提供初值并被覆盖为解
// 返回 Ok(()) 成功；Err 失败
// -------------------------------
fn r_dnode(a: &[f64], roots: &mut [f64], order: usize) -> Result<(), String> {
    let accrt: f64 = 1e-15;
    let mut x0: f64;
    let mut x1: f64;
    let mut xi: f64;
    let mut x1init: f64 = 0.0;
    let mut p0: f64;
    let mut p1: f64;
    let mut pi: f64;
    let mut p1init: f64 = a[0];

    for m in 0..order {
        x0 = x1init;
        p0 = p1init;
        x1init = roots[m];
        p1init = poly_value1_f64(a, order, x1init);

        // 全零行的短路
        if p1init == 0.0 {
            continue;
        }
        if p0 * p1init > 0.0 {
            return Err(format!(
                "R_dnode: ROOT {} NOT BRACKETED for ORDER {}",
                m, order
            ));
        }

        if x0 <= x1init {
            x1 = x1init;
            p1 = p1init;
        } else {
            x1 = x0;
            p1 = p0;
            x0 = x1init;
            p0 = p1init;
        }

        if p1 == 0.0 {
            roots[m] = x1;
            continue;
        } else if p0 == 0.0 {
            roots[m] = x0;
            continue;
        } else {
            xi = x0 + (x0 - x1) / (p1 - p0) * p0;
        }

        let mut n_iter = 0;
        while (x1 - x0).abs() > x1.abs() * accrt {
            n_iter += 1;
            if n_iter > 200 {
                return Err("R_dnode: NO CONV after 200 iterations".to_string());
            }

            pi = poly_value1_f64(a, order, xi);
            if pi == 0.0 {
                break;
            } else if p0 * pi <= 0.0 {
                x1 = xi;
                p1 = pi;
                xi = x0 * 0.25 + xi * 0.75;
            } else {
                x0 = xi;
                p0 = pi;
                xi = xi * 0.75 + x1 * 0.25;
            }

            pi = poly_value1_f64(a, order, xi);
            if pi == 0.0 {
                break;
            } else if p0 * pi <= 0.0 {
                x1 = xi;
                p1 = pi;
            } else {
                x0 = xi;
                p0 = pi;
            }

            xi = x0 + (x0 - x1) / (p1 - p0) * p0;
        }
        roots[m] = xi;
    }
    Ok(())
}

// -------------------------------
// Hessenberg-QR：单步隐式 QR（与 libcint 等价）
// A 为 n×n，上 Hessenberg；就地修改
// -------------------------------
fn qr_step(A: &mut [f64], n: usize, n0: usize, n1: usize, shift: f64) {
    let idx = |i: usize, j: usize| -> usize { i * n + j };

    let m1 = n0 + 1;
    let mut c = A[idx(n0, n0)] - shift;
    let mut s = A[idx(m1, n0)];
    let mut v = (c * c + s * s).sqrt();
    let mut x;
    let mut y;

    if v == 0.0 {
        v = 1.0;
        c = 1.0;
        s = 0.0;
    }
    v = 1.0 / v;
    c *= v;
    s *= v;

    for k in n0..n {
        // 左乘 Givens
        x = A[idx(n0, k)];
        y = A[idx(m1, k)];
        A[idx(n0, k)] = c * x + s * y;
        A[idx(m1, k)] = c * y - s * x;
    }

    let mut m3 = n1.min(n0 + 3);
    for k in 0..m3 {
        // 右乘 Givens
        x = A[idx(k, n0)];
        y = A[idx(k, m1)];
        A[idx(k, n0)] = c * x + s * y;
        A[idx(k, m1)] = c * y - s * x;
    }

    for j in n0..(n1 - 2) {
        let j1 = j + 1;
        let j2 = j + 2;

        c = A[idx(j1, j)];
        s = A[idx(j2, j)];
        v = (c * c + s * s).sqrt();
        A[idx(j1, j)] = v;
        A[idx(j2, j)] = 0.0;

        if v == 0.0 {
            v = 1.0;
            c = 1.0;
            s = 0.0;
        }
        v = 1.0 / v;
        c *= v;
        s *= v;

        for k in j1..n {
            // 左乘 Givens
            x = A[idx(j1, k)];
            y = A[idx(j2, k)];
            A[idx(j1, k)] = c * x + s * y;
            A[idx(j2, k)] = c * y - s * x;
        }

        m3 = n1.min(j + 4);
        for k in 0..m3 {
            // 右乘 Givens
            x = A[idx(k, j1)];
            y = A[idx(k, j2)];
            A[idx(k, j1)] = c * x + s * y;
            A[idx(k, j2)] = c * y - s * x;
        }
    }
}

// -------------------------------
// 实 Hessenberg 隐式 QR（到实 Schur）
// 成功后对角线上是特征值（实根）
// -------------------------------
fn hessenberg_qr(A: &mut [f64], n: usize) -> Result<(), String> {
    let eps: f64 = 1e-15;
    let maxits: usize = 30;
    let idx = |i: usize, j: usize| -> usize { i * n + j };

    let mut n0: usize = 0;
    let mut n1: usize = n;
    let mut its: usize = 0;

    for _ic in 0..(n * maxits) {
        let mut k = n0;
        while k + 1 < n1 {
            let s = A[idx(k, k)].abs() + A[idx(k + 1, k + 1)].abs();
            if A[idx(k + 1, k)].abs() < eps * s {
                break;
            }
            k += 1;
        }
        let k1 = k + 1;

        if k1 < n1 {
            // deflation at (k+1, k)
            A[idx(k1, k)] = 0.0;
            n0 = k1;
            its = 0;

            if n0 + 1 >= n1 {
                // 一个块（≤2）已收敛
                n0 = 0;
                n1 = k1;
                if n1 < 2 {
                    return Ok(());
                }
            }
        } else {
            let m1 = n1 - 1;
            let m2 = n1 - 2;
            let a11 = A[idx(m1, m1)];
            let a22 = A[idx(m2, m2)];
            let t = a11 + a22;
            let mut s = (a11 - a22) * (a11 - a22) + 4.0 * A[idx(m1, m2)] * A[idx(m2, m1)];
            let shift: f64;
            if s > 0.0 {
                s = s.sqrt();
                let a = 0.5 * (t + s);
                let b = 0.5 * (t - s);
                shift = if (a11 - a).abs() > (a11 - b).abs() {
                    b
                } else {
                    a
                };
            } else {
                if n1 == 2 {
                    return Err("hessenberg_qr: failed to find real roots".to_string());
                }
                shift = 0.5 * t;
            }
            its += 1;
            qr_step(A, n, n0, n1, shift);
            if its > maxits {
                return Err(format!(
                    "hessenberg_qr: failed to converge after {} steps",
                    its
                ));
            }
        }
    }
    Err("hessenberg_qr: iteration overflow".to_string())
}

// -------------------------------
// 公开接口：从 R_dsmit 系数表求根（等价 libcint）
// 输入：cs 为 (nroots+1)×(nroots+1)，第 nroots 行是目标多项式系数 a[0..nroots]（升幂）
// 返回：长度为 nroots 的根数组（f64）
// -------------------------------
pub fn find_polyroots(cs: &MatrixFull<f64>, nroots: usize) -> Vec<f64> {
    if nroots == 0 || cs.size[0] < nroots + 1 || cs.size[1] < nroots + 1 {
        return Vec::new();
    }

    // nroots = 1：解析
    if nroots == 1 {
        let a0 = cs[(1, 0)];
        let a1 = cs[(1, 1)];
        if a1 == 0.0 || !a1.is_finite() {
            return Vec::new();
        }
        return vec![-a0 / a1];
    }

    // nroots = 2：解析
    if nroots == 2 {
        let a0 = cs[(2, 0)];
        let a1 = cs[(2, 1)];
        let a2 = cs[(2, 2)];
        if a2 == 0.0 || !a2.is_finite() {
            return Vec::new();
        }
        let disc = a1 * a1 - 4.0 * a0 * a2;
        if disc < 0.0 {
            return Vec::new();
        }
        let rdisc = disc.sqrt();
        let r0 = (-a1 - rdisc) / (2.0 * a2);
        let r1 = (-a1 + rdisc) / (2.0 * a2);
        return vec![r0, r1];
    }

    // nroots >= 3：companion + Hessenberg-QR
    let a_row: Vec<f64> = (0..=nroots).map(|j| cs[(nroots, j)]).collect();
    let an = a_row[nroots];

    if an != 0.0 && an.is_finite() {
        let mut A = vec![0.0_f64; nroots * nroots];
        let idx = |i: usize, j: usize| -> usize { i * nroots + j };
        let fac = -1.0 / an;

        // 第一行：反转低阶系数并缩放
        for i in 0..nroots {
            A[idx(0, nroots - 1 - i)] = a_row[i] * fac;
        }
        // 次对角置 1
        for i in 0..(nroots - 1) {
            A[idx(i + 1, i)] = 1.0;
        }

        if hessenberg_qr(&mut A, nroots).is_ok() {
            let mut roots = Vec::with_capacity(nroots);
            for i in 0..nroots {
                roots.push(A[idx(i, i)]);
            }
            return roots;
        }
        // 失败则回退
    }

    // 回退：R_dnode 逐阶增长（以 2 阶解析根作为初值）
    let mut roots = vec![0.0_f64; nroots];
    {
        let a0 = cs[(2, 0)];
        let a1 = cs[(2, 1)];
        let a2 = cs[(2, 2)];
        if a2 == 0.0 || !a2.is_finite() {
            return Vec::new();
        }

        let mut disc = a1 * a1 - 4.0 * a0 * a2;
        if disc < 0.0 {
            disc = 0.0;
        }
        let rdisc = disc.sqrt();

        roots[0] = 0.5 * (-a1 - rdisc) / a2;
        roots[1] = 0.5 * (-a1 + rdisc) / a2;
        for i in 2..nroots {
            roots[i] = 1.0;
        }

        for k in 2..nroots {
            let order = k + 1;
            let a_k: Vec<f64> = (0..=order).map(|j| cs[(order, j)]).collect();
            if r_dnode(&a_k, &mut roots[..order], order).is_err() {
                break;
            }
        }
    }

    roots
}
