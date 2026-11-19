// ===========================================================================
// Full (non-low-rank) CD-G0W0 analytic gradients: pyscf.gw.gw_cd_grad port
// ===========================================================================
//
// [`GwCdGradEngine`] differentiates the *static-subtracted continuous*
// CD-G0W0 quasiparticle energy of `pyscf/gw/gw_cd_grad*.py`
//
//     E_n = e_n + B_n + Re Sigma_sub_nn(E_n)
//     Sigma_sub = sum_m [ -1/pi sum_k w_k K(t_m, u_k) (W_m(iu_k) - W_m(0))
//                         + (1/2 - f_m) W_m(0)
//                         + s_m (W_m(zeta_m) - W_m(0)) ]
//     t_m = omega - e_m - 1j*eta*sign(ef - e_m),   zeta_m = |e_m - omega|,
//     s_m = -1 (occupied m above omega), +1 (virtual m below omega), else 0,
//
// where every `W_m(z) = q_nm^T [(J - Q(z))^-1 - J^-1] q_nm` is evaluated by
// DIRECT factorisation of the full (naux x naux) screening matrix — no
// low-rank truncation — at the exact moving residue frequency zeta_m
// (REST's res_tol softening window and near-threshold half weight included,
// matching `gw_cd_grad_optimized.sigma_subtracted`).  The screening matrix
// is real symmetric on the imaginary axis (Cholesky, factored once per
// geometry) and complex symmetric for z > 0 (LU through zgetrf/zgetrs).
// The gradient pullback ports `weighted_qp_vjp`: the J^-1 (metric)
// contributions of all three terms carry the total coefficient (1/2 - f_m)
// and are pulled back once per cache; the imaginary-axis buckets are
// accumulated frequency-major; the moving residues are rank-one pullbacks.
// HF, pure-DFA and global-hybrid references are supported (the v_xc part of
// the static term flows through the shared `b_x` covector of
// `canonical_response_batch`); range-separated hybrids are rejected.
//
// # Branch policy of the QP solve
//
// The QP equation of a high-lying virtual can have several roots inside one
// self-energy pole cluster, and which one a Newton iteration from `e_n + 0.02`
// lands on changes with the geometry (measured on NH3/def2-SVP: `Omega` of a
// high virtual moves by 0.03-0.12 Ha -- and its analytic gradient by up to
// 0.5 Ha/Bohr -- for a 1e-4 Bohr displacement, with no convergence warning).
// Differentiating such a root produces a functional that is not a function of
// the geometry, so three branches are possible and this engine takes them in
// this order:
//
// * **deep occupied levels** (`e_n < 0` and `-e_n > static_fallback_threshold`):
//   the static on-shell energy `consts_n + Re Sigma_sub(e_n)` (Z = 1);
// * **a converged root with `Z = 1/(1 - dSigma/domega) < z_min`**: the same
//   static on-shell energy -- the root has (almost) no quasiparticle weight
//   left and is not a level, see `GwGradConfig::z_min`;
// * otherwise the root itself.
//
// The static on-shell branch is a closed-form functional of the geometry (no
// root search), and [`GwCdGradEngine::cd_qp_pullback`] differentiates it
// exactly (`cache.static_fallback` adds the `dSigma/domega * de_n/dR` channel).
// It is **not** the AC/WAC engine's linearized (one-step Newton) branch, which
// keeps `Z` and needs the `omega`-derivative pullback channel
// (`Z^2 * lin_shift * d_R d_omega Sigma`) that this engine does not implement.

extern "C" {
    fn zgetrs_(
        trans: *const c_char,
        n: *const i32,
        nrhs: *const i32,
        a: *const Complex<f64>,
        lda: *const i32,
        ipiv: *const i32,
        b: *mut Complex<f64>,
        ldb: *const i32,
        info: *mut i32,
    );
}

#[inline]
/// PySCF-convention residue screening diagonal
/// `2 [(z + delta + 2i eta)^-1 + (-z + delta)^-1]`, `delta = e_i - e_a`.
fn residue_d(delta: f64, z: f64, eta: f64) -> Complex<f64> {
    2.0 * (Complex::new(z + delta, 2.0 * eta).inv() + Complex::new(delta - z, 0.0).inv())
}

/// `d/dz` of the residue diagonal: `2 [-(z+delta+2i eta)^-2 + (z-delta)^-2]`.
fn residue_dz(delta: f64, z: f64, eta: f64) -> Complex<f64> {
    let t2 = Complex::new(z + delta, 2.0 * eta);
    2.0 * (-(t2 * t2).inv() + Complex::new((z - delta) * (z - delta), 0.0).inv())
}

/// `d/ddelta` of the residue diagonal:
/// `2 [-(z+delta+2i eta)^-2 - (z-delta)^-2]`.
fn residue_de(delta: f64, z: f64, eta: f64) -> Complex<f64> {
    let t2 = Complex::new(z + delta, 2.0 * eta);
    2.0 * (-(t2 * t2).inv() - Complex::new((z - delta) * (z - delta), 0.0).inv())
}

/// Static (z = 0) screening diagonal `2 [(delta + 2i eta)^-1 + delta^-1]`.
fn static_d0(delta: f64, eta: f64) -> Complex<f64> {
    2.0 * (Complex::new(delta, 2.0 * eta).inv() + Complex::new(delta, 0.0).inv())
}

/// `d/ddelta` of [`static_d0`]: `2 [-(delta+2i eta)^-2 - delta^-2]`.
fn static_d0e(delta: f64, eta: f64) -> Complex<f64> {
    let t2 = Complex::new(delta, 2.0 * eta);
    2.0 * (-(t2 * t2).inv() - Complex::new(delta * delta, 0.0).inv())
}

/// Gram matrix `sum_k w_k q_k q_k^T` of the [naux, ncol] col-major buffer.
fn scaled_gram(qia: &[f64], naux: usize, ncol: usize, w: &[f64]) -> MatrixFull<f64> {
    let mut scaled = vec![0.0f64; naux * ncol];
    for col in 0..ncol {
        let wc = w[col];
        for r in 0..naux {
            scaled[r + col * naux] = qia[r + col * naux] * wc;
        }
    }
    let sm = to_mat(&scaled, naux, ncol);
    let qm = to_mat(qia, naux, ncol);
    let mut out = MatrixFull::new([naux, naux], 0.0);
    _dgemm_full(&sm, 'N', &qm, 'T', &mut out, 1.0, 0.0);
    out
}

/// Complex-symmetric LU factorisation (zgetrf/zgetrs) shared by all
/// right-hand sides of one screening matrix.
struct CdLuFactor {
    n: usize,
    lu: MatrixFull<Complex<f64>>,
    ipiv: Vec<i32>,
}

impl CdLuFactor {
    fn factor(a_re: &[f64], a_im: &[f64], n: usize) -> Self {
        let data: Vec<Complex<f64>> = a_re
            .iter()
            .zip(a_im.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();
        let mut lu = MatrixFull::from_vec([n, n], data).expect("CdLuFactor: shape");
        let ipiv = lu.lapack_zgetrf();
        Self { n, lu, ipiv }
    }

    /// Overwrite the [n, ncol] col-major right-hand sides with `A^-1 B`.
    fn solve_in_place(&self, b: &mut [Complex<f64>], ncol: usize) {
        let (nn, nc) = (self.n as i32, ncol as i32);
        let mut info = 0i32;
        let trans = b'N' as c_char;
        unsafe {
            zgetrs_(
                &trans, &nn, &nc, self.lu.data.as_ptr(), &nn, self.ipiv.as_ptr(),
                b.as_mut_ptr(), &nn, &mut info,
            )
        };
        assert!(
            info == 0,
            "cd_gw_grad: complex screening solve failed (info={info})"
        );
    }
}

/// Geometry-fixed full-CD screening: Cholesky factors of the real symmetric
/// `J - Q(iu)` at every quadrature point and the LU factor of the complex
/// symmetric `J - Q(0)`.
pub struct CdScreening {
    chol_imag: Vec<MatrixFull<f64>>,
    lu_static: CdLuFactor,
}

impl CdScreening {
    fn new(base: &GwGradEngine) -> Self {
        let naux = base.naux;
        let nov = base.de_ia.len();
        let eta = base.config.eta;
        // imaginary-axis factors: d_I = 4 delta/(u^2 + delta^2) with
        // delta = e_i - e_a = -de_ia  =>  -4 de_ia/(u^2 + de_ia^2) (real)
        let mut chol_imag = Vec::with_capacity(base.quad.len());
        for &(u, _) in &base.quad {
            let w: Vec<f64> = base
                .de_ia
                .iter()
                .map(|&x| -4.0 * x / (u * u + x * x))
                .collect();
            let g = scaled_gram(&base.qia, naux, nov, &w);
            let mut a = base.j_mat.clone();
            for (av, gv) in a.data.iter_mut().zip(g.data.iter()) {
                *av -= gv;
            }
            _dpotrf(&mut a, 'L');
            chol_imag.push(a);
        }
        // static factor (z = 0): d0 = 2 [(delta + 2i eta)^-1 + delta^-1]
        let mut wre = vec![0.0f64; nov];
        let mut wim = vec![0.0f64; nov];
        for (col, &x) in base.de_ia.iter().enumerate() {
            let d0 = static_d0(-x, eta);
            wre[col] = d0.re;
            wim[col] = d0.im;
        }
        let gr = scaled_gram(&base.qia, naux, nov, &wre);
        let gi = scaled_gram(&base.qia, naux, nov, &wim);
        let mut are = base.j_mat.data.clone();
        let mut aim = vec![0.0f64; naux * naux];
        for k in 0..naux * naux {
            are[k] -= gr.data[k];
            aim[k] -= gi.data[k];
        }
        let lu_static = CdLuFactor::factor(&are, &aim, naux);
        Self { chol_imag, lu_static }
    }

    /// `A(iu)^-1 B` in place, `B` real [naux, ncol] col-major.
    fn solve_imag(&self, wi: usize, naux: usize, b: &mut [f64], ncol: usize) {
        let mut bm = to_mat(b, naux, ncol);
        let ok1 = _dtrtrs(&self.chol_imag[wi], &mut bm, 'L', 'N', 'N');
        let ok2 = _dtrtrs(&self.chol_imag[wi], &mut bm, 'L', 'T', 'N');
        assert!(ok1 && ok2, "cd_gw_grad: imaginary-axis triangular solve failed");
        b.copy_from_slice(&bm.data);
    }

    /// `A(0)^-1 B`, `B` real [naux, ncol]; returns complex columns.
    fn solve_static(&self, b: &[f64], ncol: usize) -> Vec<Complex<f64>> {
        let mut bc: Vec<Complex<f64>> =
            b.iter().map(|&v| Complex::new(v, 0.0)).collect();
        self.lu_static.solve_in_place(&mut bc, ncol);
        bc
    }

    /// `A(zeta)^-1 rhs` with a transient LU factor (moving residue
    /// frequency).
    fn solve_residue(&self, base: &GwGradEngine, zeta: f64, rhs: &[f64]) -> Vec<Complex<f64>> {
        let naux = base.naux;
        let nov = base.de_ia.len();
        let eta = base.config.eta;
        let mut wre = vec![0.0f64; nov];
        let mut wim = vec![0.0f64; nov];
        for (col, &x) in base.de_ia.iter().enumerate() {
            let d = residue_d(-x, zeta, eta);
            wre[col] = d.re;
            wim[col] = d.im;
        }
        let gr = scaled_gram(&base.qia, naux, nov, &wre);
        let gi = scaled_gram(&base.qia, naux, nov, &wim);
        let mut are = base.j_mat.data.clone();
        let mut aim = vec![0.0f64; naux * naux];
        for k in 0..naux * naux {
            are[k] -= gr.data[k];
            aim[k] -= gi.data[k];
        }
        let factor = CdLuFactor::factor(&are, &aim, naux);
        let mut b: Vec<Complex<f64>> =
            rhs.iter().map(|&v| Complex::new(v, 0.0)).collect();
        factor.solve_in_place(&mut b, 1);
        b
    }
}

/// One active residue of the static-subtracted self-energy.
pub struct CdResidue {
    pub m: usize,
    /// signed pole factor `s_m * pole_factor`
    pub s: f64,
    /// residue frequency `zeta_m = |e_m - omega|`
    pub zeta: f64,
    /// `A(zeta)^-1 q_ext[:, m]`, complex [naux]
    pub y: Vec<Complex<f64>>,
}

/// One converged QP state of the full-CD gradient: energy, Z factor and the
/// response data needed by the pullback (port of the saved fields of
/// `gw_cd_grad_optimized._CDCache`).
pub struct CdQpCache {
    pub target: usize,
    /// the quasiparticle energy this cache describes
    pub omega: f64,
    /// the frequency the self-energy was evaluated at (the root, or `e_n` for
    /// the static on-shell fallback) — see [`QpCache::freq`]
    pub freq: f64,
    pub z_factor: f64,
    /// `d Re Sigma_sub / d omega` at [`Self::freq`] — see [`QpCache::dsigma_domega`]
    pub dsigma_domega: f64,
    /// the energy is the **static on-shell** value `consts_n + Re Sigma_sub(e_n)`
    /// (`Z = 1`, no root search) — see [`QpCache::static_fallback`]
    pub static_fallback: bool,
    /// linearized (one-step Newton) branch — see [`QpCache::linearized`]
    pub linearized: bool,
    /// `consts_n + Re Sigma_sub(e_n) - e_n` — see [`QpCache::lin_shift`]
    pub lin_shift: f64,
    /// `W_c[n, m](iu)` at every quadrature point [n_freq][nmo]
    pub w_imag: Vec<Vec<f64>>,
    /// `A(iu)^-1 q_ext` per quadrature point, real [naux * nmo] (columns m)
    pub y_imag: Vec<Vec<f64>>,
    /// `W_c[n, m](0)`, complex [nmo]
    pub w0: Vec<Complex<f64>>,
    /// `A(0)^-1 q_ext`, complex [naux * nmo]
    pub y0_static: Vec<Complex<f64>>,
    /// `J^-1 q_ext`, real [naux * nmo]
    pub y0_metric: Vec<f64>,
    pub residues: Vec<CdResidue>,
}

/// Full (non-low-rank) CD-G0W0 analytic-gradient engine for one geometry,
/// wrapping the shared raw-RI/CPHF machinery of [`GwGradEngine`].
pub struct GwCdGradEngine<'a> {
    pub base: GwGradEngine<'a>,
    pub screening: CdScreening,
}

/// Pullback of `Tr[(V_re + i V_im) dQ]` with a complex screening diagonal
/// (PySCF `weighted_qp_vjp::polar_pullback`): `bq[ov] += 2 Re(vq * d)`,
/// `be[occ] += bg`, `be[vir] -= bg` with `bg = qia : (vq * dd)` (real part),
/// and `bJ -= V_re`.
#[allow(clippy::too_many_arguments)]
fn cd_polar_pullback_complex(
    qia: &[f64],
    naux: usize,
    nov: usize,
    nocc: usize,
    nmo: usize,
    bq: &mut [f64],
    be: &mut [f64],
    bj: &mut [f64],
    v_re: &[f64],
    v_im: &[f64],
    d_re: &[f64],
    d_im: &[f64],
    de_re: &[f64],
    de_im: &[f64],
) {
    let vqm_r = to_mat(v_re, naux, naux);
    let vqm_i = to_mat(v_im, naux, naux);
    let qm = to_mat(qia, naux, nov);
    let mut vqr = MatrixFull::new([naux, nov], 0.0);
    let mut vqi = MatrixFull::new([naux, nov], 0.0);
    _dgemm_full(&vqm_r, 'N', &qm, 'N', &mut vqr, 1.0, 0.0);
    _dgemm_full(&vqm_i, 'N', &qm, 'N', &mut vqi, 1.0, 0.0);
    for col in 0..nov {
        let i = col % nocc;
        let a_mo = nocc + col / nocc;
        let colp = (i + a_mo * nmo) * naux;
        for p in 0..naux {
            bq[colp + p] += 2.0
                * (vqr.data[p + col * naux] * d_re[col] - vqi.data[p + col * naux] * d_im[col]);
        }
        let mut bg = 0.0f64;
        for p in 0..naux {
            bg += qia[p + col * naux]
                * (vqr.data[p + col * naux] * de_re[col]
                    - vqi.data[p + col * naux] * de_im[col]);
        }
        be[i] += bg;
        be[a_mo] -= bg;
    }
    for (bv, av) in bj.iter_mut().zip(v_re.iter()) {
        *bv -= av;
    }
}

impl<'a> GwCdGradEngine<'a> {
    pub fn new(scf: &'a SCF, config: GwGradConfig) -> Self {
        let raw = build_raw_ri_tensors(scf);
        Self::new_with_raw(scf, config, raw)
    }

    /// Engine with externally supplied raw RI tensors (for validation).
    pub fn new_with_raw(scf: &'a SCF, config: GwGradConfig, raw: RawRiTensors) -> Self {
        let vxc = GwGradEngine::precompute_vxc(scf);
        Self::new_with_raw_and_vxc(scf, config, raw, vxc)
    }

    /// Full-CD engine with the grid-derived `v_xc` supplied by the caller.
    ///
    /// Use this with [`GwGradEngine::precompute_vxc`] when the caller wants to release the
    /// SCF grid's tabulated AO tables before the engine is built (see
    /// [`Grids::release_tabulated_ao`](crate::dft::Grids::release_tabulated_ao)).
    pub fn new_with_vxc(scf: &'a SCF, config: GwGradConfig, vxc: GwVxc) -> Self {
        let raw = build_raw_ri_tensors(scf);
        Self::new_with_raw_and_vxc(scf, config, raw, vxc)
    }

    /// As [`Self::new_with_raw`], with the grid-derived `v_xc` supplied by the caller.
    pub fn new_with_raw_and_vxc(
        scf: &'a SCF,
        config: GwGradConfig,
        raw: RawRiTensors,
        vxc: GwVxc,
    ) -> Self {
        // the full-CD path never touches the low-rank machinery
        assert!(
            config.gw_variant != GwVariant::Ac && config.gw_variant != GwVariant::Wac,
            "GwCdGradEngine implements only the contour-deformation self-energy; \
             use GwGradEngine for gw_variant = \"ac\" or \"wac\""
        );
        let mut config = config;
        config.skip_lr_cache = true;
        let base = GwGradEngine::new_with_raw_and_vxc(scf, config, raw, vxc);
        let screening = CdScreening::new(&base);
        Self { base, screening }
    }

    fn pair_col(&self, p: usize, q: usize) -> usize {
        self.base.pair_col(p, q)
    }

    /// Static-subtracted CD self-energy `Re Sigma_sub(omega)`, its omega
    /// derivative, and the full response cache at `omega`.  Port of
    /// `gw_cd_grad_optimized.sigma_subtracted` +
    /// `d_sigma_subtracted_domega` in one pass.
    fn sigma_sub(&self, omega: f64, n: usize) -> (f64, f64, CdQpCache) {
        let base = &self.base;
        let naux = base.naux;
        let nmo = base.nmo;
        let nocc = base.nocc;
        let e = &base.e;
        let eta = base.config.eta;
        let res_tol = base.config.res_tol;
        let ef = 0.5 * (e[nocc - 1] + e[nocc]);
        let qext = base.with_q(|q| base.gather_row(q, n));
        let mut y0m = qext.clone();
        base.solve_metric_batch(&mut y0m, nmo);

        // imaginary-axis rows
        let mut w_imag: Vec<Vec<f64>> = Vec::with_capacity(base.quad.len());
        let mut y_imag: Vec<Vec<f64>> = Vec::with_capacity(base.quad.len());
        for wi in 0..base.quad.len() {
            let mut y = qext.clone();
            self.screening.solve_imag(wi, naux, &mut y, nmo);
            let mut w = vec![0.0f64; nmo];
            for m in 0..nmo {
                let off = m * naux;
                let mut acc = 0.0f64;
                for p in 0..naux {
                    acc += qext[off + p] * (y[off + p] - y0m[off + p]);
                }
                w[m] = acc;
            }
            w_imag.push(w);
            y_imag.push(y);
        }
        // static row W(0) for every m
        let y0_static = self.screening.solve_static(&qext, nmo);
        let mut w0 = vec![Complex::new(0.0, 0.0); nmo];
        for m in 0..nmo {
            let off = m * naux;
            let mut acc = Complex::new(0.0, 0.0);
            for p in 0..naux {
                acc += qext[off + p] * (y0_static[off + p] - Complex::new(y0m[off + p], 0.0));
            }
            w0[m] = acc;
        }

        let mut residues: Vec<CdResidue> = Vec::new();
        let mut sigma = Complex::new(0.0, 0.0);
        let mut dsigma = 0.0f64;
        for m in 0..nmo {
            let occupied = m < nocc;
            let f_m = if occupied { 1.0 } else { 0.0 };
            let t = Complex::new(omega - e[m], -eta * (ef - e[m]).signum());
            let t2 = t * t;
            for (wi, &(u, wt)) in base.quad.iter().enumerate() {
                let u2 = Complex::new(u * u, 0.0);
                let kk = t / (t2 + u2);
                let kt = (u2 - t2) / ((t2 + u2) * (t2 + u2));
                let dw = Complex::new(w_imag[wi][m], 0.0) - w0[m];
                sigma -= (wt * kk * dw) / std::f64::consts::PI;
                dsigma -= wt * (kt * dw).re / std::f64::consts::PI;
            }
            sigma += (0.5 - f_m) * w0[m];
            let zeta = (e[m] - omega).abs();
            let s_m = if occupied && omega < e[m] + res_tol {
                -1.0
            } else if !occupied && omega > e[m] - res_tol {
                1.0
            } else {
                0.0
            };
            if s_m == 0.0 {
                continue;
            }
            let pole_factor = if zeta < res_tol { 0.5 } else { 1.0 };
            let qcol = &qext[m * naux..(m + 1) * naux];
            let y_res = self.screening.solve_residue(&self.base, zeta, qcol);
            let mut w_res = Complex::new(0.0, 0.0);
            for p in 0..naux {
                w_res += qcol[p] * (y_res[p] - Complex::new(y0m[m * naux + p], 0.0));
            }
            // The softened (half-weight) residue scales the **whole**
            // `W(zeta_m) - W(0)` subtraction, not just `W(zeta_m)`: at
            // `zeta_m -> 0` the term then cancels exactly (`W(0) = W0_m` by
            // construction), which is the continuity the static subtraction
            // exists for.  Scaling only `W_res` (the historical form) leaves
            // a `(1 - pole_factor) s_m W_m(0)` value whose pullback the
            // engine does not reproduce: the residue bucket's `W(0)`
            // coefficient in [`Self::cd_qp_pullback`] is `-s_m * pole_factor`
            // (`Coeffs::staticc`), so with `pole_factor = 1` the two agree but
            // at the threshold they differ by `0.5 s_m W_m(0)` -- in the
            // value, in the `J^-1` channel (the residue bucket must contribute
            // zero to it) and in the gradient.  `pole_factor = 0.5` is not a
            // corner case: the static on-shell branch evaluates Sigma at
            // `omega = e_n`, so the target's own residue always sits exactly
            // at the threshold.
            sigma += s_m * pole_factor * (w_res - w0[m]);
            // d Sigma/d omega through the moving zeta_m = |e_m - omega|
            let nov = base.de_ia.len();
            let mut wz = 0.0f64;
            for col in 0..nov {
                let base_col = col * naux;
                let mut pr = 0.0f64;
                let mut pim = 0.0f64;
                for p in 0..naux {
                    pr += y_res[p].re * base.qia[p + base_col];
                    pim += y_res[p].im * base.qia[p + base_col];
                }
                let p2re = pr * pr - pim * pim;
                let p2im = 2.0 * pr * pim;
                let dz = residue_dz(-base.de_ia[col], zeta, eta);
                wz += p2re * dz.re - p2im * dz.im;
            }
            let dzeta_domega = -(e[m] - omega).signum();
            dsigma += s_m * pole_factor * wz * dzeta_domega;
            residues.push(CdResidue { m, s: s_m * pole_factor, zeta, y: y_res });
        }
        let cache = CdQpCache {
            target: n,
            omega,
            freq: omega,
            z_factor: 0.0,
            dsigma_domega: dsigma,
            static_fallback: false,
            linearized: false,
            lin_shift: 0.0,
            w_imag,
            y_imag,
            w0,
            y0_static,
            y0_metric: y0m,
            residues,
        };
        (sigma.re, dsigma, cache)
    }

    /// Static term `consts_n = e_n + (1-hyb) Sigma^x_RI,nn - v_xc,nn` without
    /// the LR-only `qt` cache (this engine sets `skip_lr_cache`): the RI
    /// exchange of one row is formed from the raw `q` columns through the
    /// metric transform, identical in value to `GwGradEngine::consts`.
    pub fn consts(&self, n: usize) -> f64 {
        let base = &self.base;
        let naux = base.naux;
        let exchange = base.with_q(|q| {
            let mut acc = 0.0f64;
            for i in 0..base.nocc {
                let qni = base.gather_pair(q, n, i);
                let sq = gemm_nn(&base.s_metric, naux, naux, &qni, naux, 1);
                acc -= sq.iter().map(|x| x * x).sum::<f64>();
            }
            acc
        });
        base.e[n] + exchange * (1.0 - base.fock_hyb) - base.vxc_nn[n]
    }

    /// Validation helper: the QP equation residual `omega - consts_n -
    /// Re Sigma_sub(omega)` at a fixed external frequency (see
    /// [`GwGradEngine::qp_equation_value`]).
    pub fn qp_equation_value(&self, omega: f64, n: usize) -> f64 {
        omega - self.consts(n) - self.sigma_sub_at(omega, n).0
    }

    /// The **static on-shell** (`Z = 1`) branch: `consts_n + Re Sigma_sub(e_n)`
    /// evaluated at the mean-field frequency, i.e. the MolGW `EVSC` /
    /// `se%nomega = 0` rule.
    ///
    /// A closed-form functional of the geometry (no root search), so it is
    /// smooth in `R` and [`Self::cd_qp_pullback`] differentiates it exactly
    /// (`cache.static_fallback` adds the `dSigma/domega * de_n/dR` channel).
    /// Taken for deep occupied levels (`static_fallback_threshold`) and for a
    /// converged root whose `Z` is below `z_min` (see the module header).
    fn static_fallback_cache(&self, n: usize) -> CdQpCache {
        let e_n = self.base.e[n];
        let consts = self.consts(n);
        let (sigma, dsigma, mut cache) = self.sigma_sub(e_n, n);
        cache.omega = consts + sigma;
        cache.freq = e_n;
        cache.z_factor = 1.0;
        cache.dsigma_domega = dsigma;
        cache.static_fallback = true;
        cache.linearized = false;
        cache.lin_shift = 0.0;
        cache
    }

    /// Solve the QP equation of the static-subtracted continuous energy and
    /// assemble the full response cache on the branch selected by the policy in
    /// the module header (root, or the static on-shell branch).  With
    /// `GwGradConfig::qpe_graphical` the root comes from REST's production
    /// graphical linear-interpolation solver instead of the
    /// analytic-derivative Newton iteration (PySCF `qp_energy_subtracted`).
    pub fn build_target_cache(&self, n: usize) -> CdQpCache {
        let e_n = self.base.e[n];
        let side = if n < self.base.nocc { -1.0 } else { 1.0 };
        let consts = self.consts(n);
        let t0 = Instant::now();
        // Branch policy (see the module header):
        //  * deep occupied level (`gw_switch_fallback_threshold`): static
        //    on-shell `consts_n + Re Sigma_sub(e_n)`, `Z = 1`;
        //  * converged root with `Z < z_min`: the same static on-shell branch;
        //  * otherwise the root.
        // The static on-shell branch is the only smooth branch this engine can
        // differentiate exactly (the linearized one needs the omega-derivative
        // pullback channel).
        let deeply_occupied =
            e_n < 0.0 && -e_n > self.base.config.static_fallback_threshold;
        if deeply_occupied {
            return self.static_fallback_cache(n);
        }
        if self.base.config.qpe_graphical {
            let fq = |om: f64| om - consts - self.sigma_sub_at(om, n).0;
            if !self.base.config.graphical_rescue {
                println!(
                    "warning!!! cd_gw_grad Newton did not converge for orbital {n}; \
                     using the static on-shell energy (graphical_rescue = false)"
                );
                return self.static_fallback_cache(n);
            }
            let (have_crossing, xg) =
                crate::ri_gw::linear_interpolation_solver(&fq, e_n, side, 21, 0.1);
            if !have_crossing {
                println!(
                    "warning!!! cd_gw_grad graphical solver found no crossing for orbital {n}!"
                );
            }
            let (_sigma, dsigma, mut cache) = self.sigma_sub(xg, n);
            cache.omega = xg;
            cache.freq = xg;
            cache.z_factor = 1.0 / (1.0 - dsigma);
            cache.dsigma_domega = dsigma;
            if grad_timing() {
                eprintln!(
                    "[cd_gw_grad timing]   target {n} graphical solve + cache {:8.2}s",
                    t0.elapsed().as_secs_f64()
                );
            }
        }
        // Newton on the analytic derivative of the QP equation.
        let mut x = e_n + side * 0.02;
        let mut converged = false;
        for iter in 0..self.base.config.qpe_max_iter {
            let (sigma, dsigma, _) = self.sigma_sub(x, n);
            let f = x - consts - sigma;
            let fp = 1.0 - dsigma;
            let shift = -f / fp;
            if std::env::var("REST_GWGRAD_NEWTON").is_ok() {
                println!(
                    "[cd newton DBG] n={} it={} x={:.10} f={:+.3e} shift={:+.3e}",
                    n, iter, x, f, shift
                );
            }
            x += shift;
            if shift.abs() < self.base.config.qpe_tol {
                converged = true;
                break;
            }
        }
        if !converged {
            // The full-CD engine has no omega-derivative pullback channel, so it
            // cannot differentiate the linearized branch; the root is also
            // unusable (the Newton iterate did not settle).  The static
            // on-shell energy is smooth and differentiable, so it is the only
            // branch left.
            println!(
                "warning!!! cd_gw_grad Newton solver did not converge for orbital {n}; \
                 using the static on-shell energy (the full-CD gradient does not \
                 implement the linearized branch)"
            );
            return self.static_fallback_cache(n);
        }
        let (_sigma, dsigma, mut cache) = self.sigma_sub(x, n);
        let z_root = 1.0 / (1.0 - dsigma);
        if self.base.config.z_min > 0.0 && z_root < self.base.config.z_min {
            // The converged root is not a quasiparticle: `Z -> 0` means it sits
            // on top of a self-energy pole, where the QP equation has several
            // roots and this Newton iteration picks one that depends on the
            // geometry -- measured on NH3/def2-SVP: `Omega_26` moves by 0.12 Ha
            // and the analytic gradient by 0.5 Ha/Bohr for a 1e-4 Bohr
            // displacement.  `Omega_n(R)` is then not a function of R, so
            // neither its analytic derivative nor a finite difference of it
            // means anything.
            //
            // Take the **static on-shell** (`Z = 1`) branch instead:
            //
            //     E_n = consts_n(R) + Re Sigma_sub(e_n(R); R),
            //
            // a closed-form functional of the geometry with no root search, so
            // it is smooth in R; [`Self::static_fallback_cache`] builds it and
            // the pullback already differentiates it exactly (the extra
            // `dSigma/domega * de_n/dR` channel of `cache.static_fallback`).
            // This is deliberately **not** the AC/WAC engine's linearized
            // (one-step Newton) branch `e_n + Z [consts + Sigma(e_n) - e_n]`:
            // that one keeps `Z` and needs the `omega`-derivative pullback
            // channel (`Z^2 * lin_shift * d_R d_omega Sigma`) which this
            // engine does not implement.  The static branch trades the
            // spectral weight for differentiability, which is the right trade
            // for orbitals that have essentially no quasiparticle weight left.
            println!(
                "warning!!! cd_gw_grad orbital {n}: converged Z = {z_root:.4} is below \
                 z_min = {:.3} (the root sits on a self-energy pole); using the static \
                 on-shell (Z = 1) energy consts_n + Re Sigma_sub(e_n) instead of the root.",
                self.base.config.z_min
            );
            let cache = self.static_fallback_cache(n);
            let shift = cache.omega - e_n;
            println!(
                "    cd_gw_grad orbital {n}: on-shell energy = {:.9} Ha (shift {shift:+.3e} \
                 Ha from e_n)",
                cache.omega
            );
            // A quasiparticle-like shift is at most a fraction of a Hartree
            // (valence levels: 0.05-0.3 Ha; the largest single-particle shift
            // in a molecule is bounded by its bandwidth).  A shift above 1 Ha
            // means `e_n` itself sits next to a self-energy pole: the on-shell
            // value is then still deterministic and smooth, but it is not a
            // level, and its gradient -- though exact for this functional -- is
            // steep and of no physical use.  Such orbitals should stay out of
            // the BSE transition space.
            if shift.abs() > 1.0 {
                println!(
                    "warning!!! cd_gw_grad orbital {n}: the static on-shell point lies {shift:+.2} \
                     Ha away from e_n -- it sits next to a self-energy pole, so this level is not \
                     a quasiparticle; keep it out of the BSE transition space (bse_cutoff_energy \
                     / bse_active_vir_end) rather than differentiating it on its own."
                );
            }
            return cache;
        }
        cache.omega = x;
        cache.freq = x;
        cache.z_factor = z_root;
        cache.dsigma_domega = dsigma;
        if grad_timing() {
            eprintln!(
                "[cd_gw_grad timing]   target {} newton+cache {:8.2}s",
                n,
                t0.elapsed().as_secs_f64()
            );
        }
        cache
    }

    /// Pull weighted QP energies back onto the raw `(q, J, e, B)` covectors.
    /// Port of `gw_cd_grad_optimized.weighted_qp_vjp` (all returned
    /// covectors are real; the complex symmetric screening is contracted
    /// through its real/imaginary parts with the transpose, never the
    /// conjugate transpose).
    pub fn cd_qp_pullback(
        &self,
        caches: &[&CdQpCache],
        weights: &[Vec<f64>],
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let t_all = Instant::now();
        let base = &self.base;
        let naux = base.naux;
        let nmo = base.nmo;
        let nocc = base.nocc;
        let nov = base.de_ia.len();
        let eta = base.config.eta;
        let nk = weights.len();
        for w in weights {
            assert_eq!(w.len(), caches.len(), "cd_qp_pullback: weight row length mismatch");
        }
        // Guard: this engine never builds a linearized cache (it takes the
        // static on-shell branch for `Z < z_min`, see the module header), and a
        // linearized cache handed in from elsewhere would silently produce a
        // wrong gradient because the `Z^2 * lin_shift * d_R d_omega Sigma`
        // channel is not implemented here.
        for c in caches.iter() {
            assert!(
                !c.linearized,
                "gw_grad: the full-CD GW analytic gradient does not implement the \
                 omega-derivative channel of the linearized (one-step Newton) QP \
                 branch (orbital {} was built on it); the CD engine uses the static \
                 on-shell branch for Z < gradient_z_min instead.  Use \
                 gw_variant = \"ac\" when the linearized branch is required.",
                c.target
            );
        }
        let mask = base.config.phase_mask;
        let (do_imag, do_static, do_res) = (mask & 1 != 0, mask & 2 != 0, mask & 4 != 0);

        let mut bq = vec![vec![0.0f64; naux * nmo * nmo]; nk];
        let mut bj = vec![vec![0.0f64; naux * naux]; nk];
        let mut be = vec![vec![0.0f64; nmo]; nk];
        let mut bb = vec![vec![0.0f64; nmo]; nk];

        // per-cache coefficient tables (independent of the weight row k)
        struct Coeffs {
            /// imag[wi][m] = -wts[wi] K(t_m, u_wi) / pi
            imag: Vec<Vec<Complex<f64>>>,
            /// static_coeff[m] = (1/2 - f_m) - sum_wi imag[wi][m] - active[m]
            staticc: Vec<Complex<f64>>,
            /// explicit t_m -> eps1[m] term of the imaginary-axis integral
            explicit: Vec<f64>,
        }
        let coeffs: Vec<Coeffs> = caches
            .iter()
            .map(|cache| {
                let ef = 0.5 * (base.e[nocc - 1] + base.e[nocc]);
                let mut active = vec![0.0f64; nmo];
                for r in &cache.residues {
                    active[r.m] += r.s;
                }
                let mut imag = vec![vec![Complex::new(0.0, 0.0); nmo]; base.quad.len()];
                let mut staticc = vec![Complex::new(0.0, 0.0); nmo];
                let mut explicit = vec![0.0f64; nmo];
                for m in 0..nmo {
                    let half_minus_f = if m < nocc { -0.5 } else { 0.5 };
                    let t = Complex::new(
                        cache.freq - base.e[m],
                        -eta * (ef - base.e[m]).signum(),
                    );
                    let t2 = t * t;
                    let mut sum_a = Complex::new(0.0, 0.0);
                    for (wi, &(u, wt)) in base.quad.iter().enumerate() {
                        let u2 = Complex::new(u * u, 0.0);
                        let kk = t / (t2 + u2);
                        let kt = (u2 - t2) / ((t2 + u2) * (t2 + u2));
                        let a = -wt * kk / std::f64::consts::PI;
                        imag[wi][m] = a;
                        sum_a += a;
                        let dw = Complex::new(cache.w_imag[wi][m], 0.0) - cache.w0[m];
                        explicit[m] += wt * (kt * dw).re / std::f64::consts::PI;
                    }
                    // `W(0)` coefficient of the (masked) functional
                    //   sum_m [ d_i sum_wi a (W_wi - W0) + d_s (1/2 - f_m) W0
                    //           + d_r sum_r s (W_res - W0) ]
                    let f_term = if do_static { half_minus_f } else { 0.0 };
                    let s_a = if do_imag { sum_a } else { Complex::new(0.0, 0.0) };
                    let act = if do_res { active[m] } else { 0.0 };
                    staticc[m] = Complex::new(f_term, 0.0) - s_a - Complex::new(act, 0.0);
                }
                Coeffs { imag, staticc, explicit }
            })
            .collect();

        // ---- phase A: target be/bb, explicit be and the metric (J^-1)
        //      pullback with the exact total coefficient (1/2 - f_m) ----
        let t_a = Instant::now();
        for (j, cache) in caches.iter().enumerate() {
            let n = cache.target;
            // `g_ni = J^-1 q_ni` and `sum_i g_ni g_ni^T` do not depend on the BSE
            // state `k`, so they are built once per cache instead of once per
            // `(k, i)`.  The rank-1 sum is one GEMM (`G G^T`): the scalar form
            // wrote the whole `naux^2` buffer once per occupied orbital with
            // stride-`naux` stores -- the pattern that was already removed from
            // `qp_pullback_chunk` (see the note there), but had been left here.
            // `g_ni` also goes through the Cholesky factor
            // (`solve_metric_batch`) rather than two `S = J^-1/2` GEMMs, which
            // halves the flops of this loop and stops allocating per pair.
            let exch = (1.0 - base.fock_hyb).abs() > 1.0e-14;
            let (gmat, gg) = if exch {
                let na = naux;
                let mut gm = MatrixFull::new([na, nocc], 0.0);
                for i in 0..nocc {
                    let qni = base.with_q(|q| base.gather_pair(q, n, i));
                    let mut g = qni;
                    base.solve_metric_batch(&mut g, 1);
                    gm.data[i * na..(i + 1) * na].copy_from_slice(&g);
                }
                let mut g2 = MatrixFull::new([na, na], 0.0);
                _dgemm_full(&gm, 'N', &gm, 'T', &mut g2, 1.0, 0.0);
                (Some(gm), Some(g2))
            } else {
                (None, None)
            };
            for k in 0..nk {
                let c = weights[k][j] * cache.z_factor;
                if c == 0.0 {
                    continue;
                }
                be[k][n] += c;
                bb[k][n] += c;
                if cache.static_fallback {
                    // see `QpCache::static_fallback`: the mean-field frequency
                    // moves with the perturbation
                    be[k][n] += weights[k][j] * cache.dsigma_domega;
                }
                if do_imag {
                    for m in 0..nmo {
                        be[k][m] += c * coeffs[j].explicit[m];
                    }
                }
                // ---- (1-alpha) RI-exchange part of the static term --------
                // `B_n = e_n + (1-alpha) Sigma_x,nn - v_xc,nn` with
                // `Sigma_x,nn = -sum_i q_ni^T J^-1 q_ni`, so this half of
                // `d B_n/dR` is `-(1-alpha)[2 sum_i g_ni^T dq_ni
                // - sum_i g_ni^T dJ g_ni]` with `g_ni = J^-1 q_ni`.  The v_xc
                // half leaves through the `b_x` covector (`bb`), and for an HF
                // reference `fock_hyb == 1` drops the whole block (the exchange
                // then already sits in `e_n`).  Mirrors the LR pullback
                // (`qp_pullback_chunk`); Omitting it made every DFT reference's
                // full-CD gradient wrong by the exchange-derivative term.
                if let (Some(gm), Some(g2)) = (gmat.as_ref(), gg.as_ref()) {
                    let na = naux;
                    let ex_scale = c * (1.0 - base.fock_hyb);
                    for i in 0..nocc {
                        let col_i = self.pair_col(n, i) * na;
                        let g_ni = &gm.data[i * na..(i + 1) * na];
                        for p in 0..na {
                            bq[k][col_i + p] -= 2.0 * ex_scale * g_ni[p];
                        }
                    }
                    for (d, s) in bj[k].iter_mut().zip(g2.data.iter()) {
                        *d += ex_scale * s;
                    }
                }
                if !do_static {
                    continue;
                }
                let mut scaled = vec![0.0f64; naux * nmo];
                for m in 0..nmo {
                    let am = -c * (if m < nocc { -0.5 } else { 0.5 });
                    let off = m * naux;
                    let col = self.pair_col(n, m) * naux;
                    for p in 0..naux {
                        let v = am * cache.y0_metric[off + p];
                        scaled[off + p] = v;
                        bq[k][col + p] += 2.0 * v;
                    }
                }
                let sm = to_mat(&scaled, naux, nmo);
                let ym = to_mat(&cache.y0_metric, naux, nmo);
                let mut acc = MatrixFull::new([naux, naux], 0.0);
                _dgemm_full(&sm, 'N', &ym, 'T', &mut acc, 1.0, 0.0);
                for (bv, av) in bj[k].iter_mut().zip(acc.data.iter()) {
                    *bv -= av;
                }
            }
        }
        let t_a = t_a.elapsed().as_secs_f64();

        // ---- phase B: frequency-major imaginary-axis buckets ----
        let t_b = Instant::now();
        let imag_factors: Vec<(Vec<f64>, Vec<f64>)> = base
            .quad
            .iter()
            .map(|&(u, _)| {
                let mut d = vec![0.0f64; nov];
                let mut dd = vec![0.0f64; nov];
                for (idx, &x) in base.de_ia.iter().enumerate() {
                    d[idx] = -4.0 * x / (x * x + u * u);
                    dd[idx] = -4.0 * (u * u - x * x) / ((x * x + u * u) * (x * x + u * u));
                }
                (d, dd)
            })
            .collect();
        for wi in 0..if do_imag { base.quad.len() } else { 0 } {
            let (d, dd) = &imag_factors[wi];
            for k in 0..nk {
                let mut v = MatrixFull::new([naux, naux], 0.0);
                for (j, cache) in caches.iter().enumerate() {
                    let c = weights[k][j] * cache.z_factor;
                    if c == 0.0 {
                        continue;
                    }
                    let n = cache.target;
                    let y = &cache.y_imag[wi];
                    let mut scaled = vec![0.0f64; naux * nmo];
                    for m in 0..nmo {
                        let a = c * coeffs[j].imag[wi][m].re;
                        let off = m * naux;
                        let col = self.pair_col(n, m) * naux;
                        for p in 0..naux {
                            let val = a * y[off + p];
                            scaled[off + p] = val;
                            bq[k][col + p] += 2.0 * val;
                        }
                    }
                    let sm = to_mat(&scaled, naux, nmo);
                    let ym = to_mat(y, naux, nmo);
                    _dgemm_full(&sm, 'N', &ym, 'T', &mut v, 1.0, 1.0);
                }
                // bJ -= V and the polarizability pullback of Tr[V dQ]
                for (bv, av) in bj[k].iter_mut().zip(v.data.iter()) {
                    *bv -= av;
                }
                base.polar_pullback(&mut bq, &mut be, k, &v.data, d, dd, false);
            }
        }
        let t_b = t_b.elapsed().as_secs_f64();

        // ---- phase C: static (z = 0) bucket ----
        let t_c = Instant::now();
        let (d0_re, d0_im, d0e_re, d0e_im): (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) = {
            let mut dr = vec![0.0f64; nov];
            let mut di = vec![0.0f64; nov];
            let mut er = vec![0.0f64; nov];
            let mut ei = vec![0.0f64; nov];
            for (col, &x) in base.de_ia.iter().enumerate() {
                let d0 = static_d0(-x, eta);
                let d0e = static_d0e(-x, eta);
                dr[col] = d0.re;
                di[col] = d0.im;
                er[col] = d0e.re;
                ei[col] = d0e.im;
            }
            (dr, di, er, ei)
        };
        for (j, cache) in caches.iter().enumerate() {
            let n = cache.target;
            let yr: Vec<f64> = cache.y0_static.iter().map(|z| z.re).collect();
            let yi: Vec<f64> = cache.y0_static.iter().map(|z| z.im).collect();
            for k in 0..nk {
                let c = weights[k][j] * cache.z_factor;
                if c == 0.0 {
                    continue;
                }
                // P = Re(c a_m y_m), Q = Im(c a_m y_m); V = (P + iQ)(yr + i yi)^T
                let mut pre = vec![0.0f64; naux * nmo];
                let mut pim = vec![0.0f64; naux * nmo];
                for m in 0..nmo {
                    let a = c * coeffs[j].staticc[m];
                    let off = m * naux;
                    let col = self.pair_col(n, m) * naux;
                    for p in 0..naux {
                        let vr = a.re * yr[off + p] - a.im * yi[off + p];
                        let vi = a.im * yr[off + p] + a.re * yi[off + p];
                        pre[off + p] = vr;
                        pim[off + p] = vi;
                        bq[k][col + p] += 2.0 * vr;
                    }
                }
                let prm = to_mat(&pre, naux, nmo);
                let pimm = to_mat(&pim, naux, nmo);
                let yrm = to_mat(&yr, naux, nmo);
                let yimm = to_mat(&yi, naux, nmo);
                let mut vr = MatrixFull::new([naux, naux], 0.0);
                let mut vi = MatrixFull::new([naux, naux], 0.0);
                _dgemm_full(&prm, 'N', &yrm, 'T', &mut vr, 1.0, 0.0);
                _dgemm_full(&pimm, 'N', &yimm, 'T', &mut vr, -1.0, 1.0);
                _dgemm_full(&pimm, 'N', &yrm, 'T', &mut vi, 1.0, 0.0);
                _dgemm_full(&prm, 'N', &yimm, 'T', &mut vi, 1.0, 1.0);
                cd_polar_pullback_complex(
                    &base.qia, naux, nov, nocc, nmo, &mut bq[k], &mut be[k], &mut bj[k],
                    &vr.data, &vi.data, &d0_re, &d0_im, &d0e_re, &d0e_im,
                );
            }
        }
        let t_c = t_c.elapsed().as_secs_f64();

        // ---- phase D: rank-one moving residues ----
        let t_d = Instant::now();
        for (j, cache) in caches.iter().enumerate().take(if do_res { usize::MAX } else { 0 }) {
            let n = cache.target;
            if cache.residues.is_empty() {
                continue;
            }
            let nres = cache.residues.len();
            let mut yre = vec![0.0f64; naux * nres];
            let mut yim = vec![0.0f64; naux * nres];
            for (ri, r) in cache.residues.iter().enumerate() {
                for p in 0..naux {
                    yre[ri * naux + p] = r.y[p].re;
                    yim[ri * naux + p] = r.y[p].im;
                }
            }
            // proj[res, col] = y_res^T qia[:, col]; `gemm_nt` returns the
            // col-major [nres, nov] buffer of that product, so element
            // (res, col) lives at `res + col*nres` (NOT `res*nov + col`).
            let proj_re = gemm_nt(&yre, naux, nres, &base.qia, naux, nov);
            let proj_im = gemm_nt(&yim, naux, nres, &base.qia, naux, nov);
            // per-residue diagonals (PySCF delta convention, delta = -de_ia)
            let mut diag: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> =
                Vec::with_capacity(nres);
            for r in &cache.residues {
                let mut dr = vec![0.0f64; nov];
                let mut di = vec![0.0f64; nov];
                let mut er = vec![0.0f64; nov];
                let mut ei = vec![0.0f64; nov];
                let mut zr = vec![0.0f64; nov];
                let mut zi = vec![0.0f64; nov];
                for (col, &x) in base.de_ia.iter().enumerate() {
                    let dv = residue_d(-x, r.zeta, eta);
                    let dev = residue_de(-x, r.zeta, eta);
                    let dzv = residue_dz(-x, r.zeta, eta);
                    dr[col] = dv.re;
                    di[col] = dv.im;
                    er[col] = dev.re;
                    ei[col] = dev.im;
                    zr[col] = dzv.re;
                    zi[col] = dzv.im;
                }
                diag.push((dr, di, er, ei, zr, zi));
            }
            for (ri, r) in cache.residues.iter().enumerate() {
                let (dr, di, er, ei, zr, zi) = &diag[ri];
                // w_zeta = Re sum_ia proj^2 dz (k-independent)
                let mut wz = 0.0f64;
                for col in 0..nov {
                    let pr = proj_re[ri + col * nres];
                    let pi = proj_im[ri + col * nres];
                    let p2re = pr * pr - pi * pi;
                    let p2im = 2.0 * pr * pi;
                    wz += p2re * zr[col] - p2im * zi[col];
                }
                let zsign = (base.e[r.m] - cache.freq).signum();
                for k in 0..nk {
                    let c = weights[k][j] * cache.z_factor;
                    if c == 0.0 {
                        continue;
                    }
                    let a = c * r.s;
                    if a == 0.0 {
                        continue;
                    }
                    // bq of the external pair (n, m)
                    let coln = self.pair_col(n, r.m) * naux;
                    for p in 0..naux {
                        bq[k][coln + p] += 2.0 * a * yre[ri * naux + p];
                    }
                    // bJ -= a Re(y y^T) (rank one)
                    for p in 0..naux {
                        let yr = yre[ri * naux + p];
                        let yi = yim[ri * naux + p];
                        for q in 0..naux {
                            let yr2 = yre[ri * naux + q];
                            let yi2 = yim[ri * naux + q];
                            bj[k][p + q * naux] -= a * (yr * yr2 - yi * yi2);
                        }
                    }
                    // polarizability pullback of y^T dQ y and the explicit
                    // zeta_m -> eps1[m] term
                    for col in 0..nov {
                        let pr = proj_re[ri + col * nres];
                        let pi = proj_im[ri + col * nres];
                        let p2re = pr * pr - pi * pi;
                        let p2im = 2.0 * pr * pi;
                        let pd_re = pr * dr[col] - pi * di[col];
                        let pd_im = pi * dr[col] + pr * di[col];
                        let i = col % nocc;
                        let a_mo = nocc + col / nocc;
                        let colp = self.pair_col(i, a_mo) * naux;
                        for p in 0..naux {
                            bq[k][colp + p] +=
                                2.0 * a * (yre[ri * naux + p] * pd_re - yim[ri * naux + p] * pd_im);
                        }
                        let bg = a * (p2re * er[col] - p2im * ei[col]);
                        be[k][i] += bg;
                        be[k][a_mo] -= bg;
                    }
                    be[k][r.m] += a * wz * zsign;
                }
            }
        }
        let t_d = t_d.elapsed().as_secs_f64();

        // symmetrise the covectors (minimum-norm symmetric representation)
        for k in 0..nk {
            symmetrise_pairs(&mut bq[k], naux, nmo);
            symmetrise_square(&mut bj[k], naux);
        }
        if grad_timing() {
            eprintln!(
                "[cd_gw_grad timing]   pullback: coeffs+metric {:7.2}s  imag {:7.2}s  static {:7.2}s  residues {:7.2}s  (total {:7.2}s)",
                t_a,
                t_b,
                t_c,
                t_d,
                t_all.elapsed().as_secs_f64()
            );
        }
        (bq, bj, be, bb)
    }

    /// Analytic gradient of one full-CD G0W0 QP energy.
    ///
    /// Returns `(grad, omega, z_factor)` with `grad[atm*3 + comp]` in
    /// Hartree/Bohr (row-major over atoms and Cartesian components).
    ///
    /// For a `target` inside a symmetry-protected degenerate shell the
    /// gradient is the basis-independent **shell average** — see
    /// [`GwGradEngine::analytic_gradient_ac_wac`].
    pub fn analytic_gradient(&self, target: usize) -> (Vec<f64>, f64, f64) {
        let shell = self.base.degenerate_shell(target);
        let caches: Vec<CdQpCache> =
            shell.iter().map(|&m| self.build_target_cache(m)).collect();
        let w = 1.0 / caches.len() as f64;
        let refs: Vec<&CdQpCache> = caches.iter().collect();
        let grad = self.cd_gradient_with_caches(&refs, &[vec![w; caches.len()]]);
        let own = &caches[shell.iter().position(|&m| m == target).unwrap()];
        (grad, own.omega, own.z_factor)
    }

    /// Analytic gradient from an already-converged [`CdQpCache`].
    ///
    /// The CP-HF batch (`canonical_response_batch`) runs BEFORE the QP
    /// pullback so that the analdrv workspaces and the `bq` covector (plus
    /// the per-perturbation `di`/`qx` transients) never coexist — this keeps
    /// the peak memory of the gradient phase close to PySCF's.
    ///
    /// For a target inside a degenerate shell this rebuilds the partner
    /// caches and returns the shell average (use
    /// [`GwCdGradEngine::analytic_gradient`] to avoid the extra work).
    pub fn analytic_gradient_with_cache(&self, cache: &CdQpCache) -> Vec<f64> {
        let shell = self.base.degenerate_shell(cache.target);
        if shell.len() == 1 {
            return self.cd_gradient_with_caches(&[cache], &[vec![1.0]]);
        }
        let caches: Vec<CdQpCache> =
            shell.iter().map(|&m| self.build_target_cache(m)).collect();
        let w = 1.0 / caches.len() as f64;
        let refs: Vec<&CdQpCache> = caches.iter().collect();
        self.cd_gradient_with_caches(&refs, &[vec![w; caches.len()]])
    }

    /// Shared gradient assembly: one covector pullback of the weighted cache
    /// set followed by the per-perturbation contraction.  `weights[0][j]` is
    /// the coefficient of cache `j`; a single covector set is returned and
    /// `contract_perturbation` sums over the (here one) sets.
    fn cd_gradient_with_caches(&self, caches: &[&CdQpCache], weights: &[Vec<f64>]) -> Vec<f64> {
        let responses = self.base.canonical_response_batch();
        let (bq, bj, be, bb) = self.cd_qp_pullback(caches, weights);
        // fold weight row 0 once (the full-CD driver only ever returns `g[0]`)
        let folded = if self.base.config.folded_qx {
            Some(self.base.fold_covector(&bq[0]))
        } else {
            None
        };
        let mut grad = vec![0.0f64; self.base.natm * 3];
        for atm in 0..self.base.natm {
            let blocks = self.base.raw.d_atom_blocks(atm);
            for comp in 0..3 {
                let dj = self.base.raw.d_j_atom(atm, comp);
                let (u, eps1, b_x) = responses[atm * 3 + comp].clone();
                let g = match &folded {
                    Some(f) => self.base.folded_perturbation_gradient(
                        f, &blocks, comp, &u, &bj[0], &be[0], &bb[0], &dj, &eps1, &b_x,
                    ),
                    None => {
                        let qx = self.base.qx_from_u_blocks(&blocks, comp, &u);
                        self.base.contract_perturbation(
                            &bq, &bj, &be, &bb, &qx, &dj, &eps1, &b_x,
                        )[0]
                    }
                };
                grad[atm * 3 + comp] = g;
            }
        }
        grad
    }

    /// QP energy of one orbital (the scalar whose gradient is returned by
    /// [`GwCdGradEngine::analytic_gradient`]).
    pub fn qp_energy_of(&self, target: usize) -> f64 {
        self.build_target_cache(target).omega
    }

    /// Validation helper: `Re Sigma_sub(omega)` and `d Re Sigma_sub/d omega`
    /// at a **fixed** external frequency (no QP root search).
    pub fn sigma_sub_at(&self, omega: f64, n: usize) -> (f64, f64) {
        let (sigma, dsigma, _cache) = self.sigma_sub(omega, n);
        (sigma, dsigma)
    }

}

