use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct RecurrenceCoeffs {
    pub b10: f64,  // B10
    pub b01p: f64, // B01'
    pub b00: f64,  // B00
    pub c00: f64,  // C00
    pub c00p: f64, // C00'
}

impl RecurrenceCoeffs {
    pub fn zeros() -> Self {
        Self {
            b10: 0.0,
            b01p: 0.0,
            b00: 0.0,
            c00: 0.0,
            c00p: 0.0,
        }
    }
}

pub trait CoeffProvider {
    #[allow(clippy::too_many_arguments)]
    fn coeffs_for(
        &self,
        root: &f64,
        p_center: &f64,
        q_center: &f64,
        a_center: &f64,
        b_center: &f64,
        c_center: &f64,
        d_center: &f64,
        alpha: &f64,
        beta: &f64,
        gamma: &f64,
        delta: &f64,
        rho: &f64,
    ) -> RecurrenceCoeffs;
}

#[derive(Clone, Copy, Debug)]
pub struct PanelSpec {
    pub ni_max: u32, // = la + lb
    pub nk_max: u32, // = lc + ld
}

#[allow(clippy::too_many_arguments)]
pub fn build_1d_panel_for_t<C: CoeffProvider>(
    coeffs: &C,
    root: &f64,
    spec: PanelSpec,
    p: &f64,
    q: &f64,
    ax: &f64,
    bx: &f64,
    cx: &f64,
    dx: &f64,
    alpha: &f64,
    beta: &f64,
    gamma: &f64,
    delta: &f64,
    rho: &f64,
) -> HashMap<(u32, u32), f64> {
    let rc = coeffs.coeffs_for(root, p, q, ax, bx, cx, dx, alpha, beta, gamma, delta, rho);

    let mut z = HashMap::<(u32, u32), f64>::new();

    // G[0,0] = 1
    z.insert((0, 0), 1.0);

    /* ----- (48) G[n+1,0] = n*B10*G[n-1,0] + C00*G[n,0] ----- */
    {
        let v = rc.c00 * z[&(0, 0)];
        z.insert((1, 0), v);
    }
    let mut g_nm1 = z[&(0, 0)]; // G[n-1,0]
    let mut g_n = z[&(1, 0)]; // G[n,0]
    for n in 1..spec.ni_max {
        let term1 = (n as f64) * rc.b10 * g_nm1;
        let term2 = rc.c00 * g_n;
        let cur = term1 + term2;
        z.insert((n + 1, 0), cur);
        g_nm1 = g_n;
        g_n = cur;
    }

    /* ----- (49) G[0,m+1] = m*B01'*G[0,m-1] + C00'*G[0,m] ----- */
    {
        let v = rc.c00p * z[&(0, 0)];
        z.insert((0, 1), v);
    }
    let mut g_mn1 = z[&(0, 0)]; // G[0,m-1]
    let mut g_m = z[&(0, 1)]; // G[0,m]
    for m in 1..spec.nk_max {
        let term1 = (m as f64) * rc.b01p * g_mn1;
        let term2 = rc.c00p * g_m;
        let cur = term1 + term2;
        z.insert((0, m + 1), cur);
        g_mn1 = g_m;
        g_m = cur;
    }

    /* ----- (50) m=1：G[n,1] = n*B00*G[n-1,0] + C00'*G[n,0] ----- */
    for n in 0..=spec.ni_max {
        let g_n0 = z[&(n, 0)];
        let mut v = 0.0;

        if n > 0 {
            let g_nm10 = z[&(n - 1, 0)];
            v = (n as f64) * rc.b00 * g_nm10;
        }
        v += rc.c00p * g_n0;
        z.insert((n, 1), v);
    }

    /* ----- (51) n=1：G[1,m] = m*B00*G[0,m-1] + C00*G[0,m] ----- */
    for m in 0..=spec.nk_max {
        let g_0m = z[&(0, m)];
        let mut v = 0.0;

        if m > 0 {
            let g_0mn1 = z[&(0, m - 1)];
            v = (m as f64) * rc.b00 * g_0mn1;
        }
        v += rc.c00 * g_0m;
        z.insert((1, m), v);
    }

    /* ----- (44) G[n,m+1] = (m*B01')G[n,m-1] + (n*B00)G[n-1,m] + C00'G[n,m] ----- */
    for n in 1..=spec.ni_max {
        for m in 1..spec.nk_max {
            let g_nm1 = z[&(n, m - 1)];
            let g_n1m = z[&(n - 1, m)];
            let g_nm = z[&(n, m)];

            let t1 = (m as f64) * rc.b01p * g_nm1;
            let t2 = (n as f64) * rc.b00 * g_n1m;
            let t3 = rc.c00p * g_nm;

            let val = t1 + t2 + t3;
            z.insert((n, m + 1), val);
        }
    }

    z
}
/* =========================
 * Transfer：i → j
 * ========================= */
pub fn transfer_i_to_j(
    table: &mut HashMap<(u32, u32, u32, u32), f64>,
    xi_minus_xj: &f64,
    seed_ni_max: u32, // = la + lb
    nj_max: u32,      // = lb
    nk_fix: u32,
    nl_fix: u32,
) {
    let eps = 1e-18_f64;

    debug_assert!(
        table.contains_key(&(seed_ni_max, 0, nk_fix, nl_fix)),
        "i→j: j=0 seeds must cover i=0..la+lb"
    );

    if xi_minus_xj.abs() <= eps {
        // same_center：Z(ni,nj) = Z(ni+nj, 0)
        for nj in 1..=nj_max {
            let i_upper = seed_ni_max - nj;
            for ni in 0..=i_upper {
                let src_i = ni + nj;
                let val = if src_i > seed_ni_max {
                    0.0
                } else {
                    *table
                        .get(&(src_i, 0, nk_fix, nl_fix))
                        .expect("i→j seed missing")
                };
                table.insert((ni, nj, nk_fix, nl_fix), val);
            }
        }
        return;
    }

    // diff_center：Z(ni,nj) = (xi-xj)*Z(ni,nj-1) + Z(ni+1,nj-1)
    for nj in 1..=nj_max {
        let i_upper = seed_ni_max - nj;
        for ni in 0..=i_upper {
            let left = *table
                .get(&(ni, nj - 1, nk_fix, nl_fix))
                .expect("i→j: missing (ni,nj-1,...)");
            let right = *table
                .get(&(ni + 1, nj - 1, nk_fix, nl_fix))
                .expect("i→j: missing (ni+1,nj-1,...)");

            let val = right + (*xi_minus_xj) * left;
            table.insert((ni, nj, nk_fix, nl_fix), val);
        }
    }
}

/* =========================
 * Transfer：k → l
 * ========================= */
pub fn transfer_k_to_l(
    table: &mut HashMap<(u32, u32, u32, u32), f64>,
    xk_minus_xl: &f64,
    seed_ni_max: u32, // = la + lb
    nj_max: u32,      // = lb
    seed_nk_max: u32, // = lc + ld
    nl_max: u32,      // = ld
) {
    let eps = 1e-18_f64;

    debug_assert!(
        (0..=seed_nk_max).all(|nk| (0..=nj_max).all(|nj| {
            let i_upper = seed_ni_max.saturating_sub(nj);
            (0..=i_upper).all(|ni| table.contains_key(&(ni, nj, nk, 0)))
        })),
        "k→l: nl=0 seeds (triangular domain) are not fully populated"
    );

    if xk_minus_xl.abs() <= eps {
        // same_center：Z(...,nk,nl) = Z(..., nk+nl, 0)
        for nl in 1..=nl_max {
            let k_upper = seed_nk_max.saturating_sub(nl);
            for nk in 0..=k_upper {
                for nj in 0..=nj_max {
                    let i_upper = seed_ni_max.saturating_sub(nj);
                    for ni in 0..=i_upper {
                        let val = *table
                            .get(&(ni, nj, nk + nl, 0))
                            .expect("k→l coincident: missing Z(ni,nj,nk+nl,0)");
                        table.insert((ni, nj, nk, nl), val);
                    }
                }
            }
        }
        return;
    }

    // diff_center：Z(...,nk,nl) = (xk-xl)*Z(...,nk,nl-1) + Z(...,nk+1,nl-1)
    for nl in 1..=nl_max {
        let k_upper = seed_nk_max - nl;
        for nk in 0..=k_upper {
            for nj in 0..=nj_max {
                let i_upper = seed_ni_max.saturating_sub(nj);
                for ni in 0..=i_upper {
                    let left = *table
                        .get(&(ni, nj, nk, nl - 1))
                        .expect("k→l: missing (..,nl-1)");
                    let right = *table
                        .get(&(ni, nj, nk + 1, nl - 1))
                        .expect("k→l: missing (..,nk+1,nl-1)");

                    let val = right + (*xk_minus_xl) * left;
                    table.insert((ni, nj, nk, nl), val);
                }
            }
        }
    }
}
