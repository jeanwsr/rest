// ---------------------------------------------------------------------------

/// Low-rank factorisation of the screened response kernel in the
/// metric-transformed representation:
/// `(I - Pi0)^-1 - I = U diag(lam) U^T` with orthonormal `U` columns.
pub struct LrFactor {
    pub omega: f64,
    /// eigenvectors [naux, n_keep], col-major
    pub eigvec: Vec<f64>,
    /// RPA eigenvalues `lam = lam0 / (1 - lam0)`
    pub eigval: Vec<f64>,
    pub n_aux: usize,
}

/// REST response factor per (i,a) transition, spin-restricted total:
/// imaginary axis: `-4 de / (de^2 + omega^2)`;
/// real axis (REST Lorentzian):
/// `-4 de (de^2-w^2+eta^2) / [(de^2-w^2)^2 + 2 eta^2 (de^2+w^2) + eta^4]`.
fn rho_response_diagonal(de: &[f64], omega: f64, part: char, eta: f64) -> Vec<f64> {
    de.iter()
        .map(|&de| {
            if part == 'I' {
                -4.0 * de / (de * de + omega * omega)
            } else {
                real_response_and_deriv_single(de, omega, eta).0
            }
        })
        .collect()
}

/// REST real-axis response factor and its de-derivative at fixed frequency.
fn real_response_and_deriv_single(de: f64, z: f64, eta: f64) -> (f64, f64) {
    let de2 = de * de;
    let z2 = z * z;
    let eta2 = eta * eta;
    let num = de2 - z2 + eta2;
    let den = (de2 - z2) * (de2 - z2) + 2.0 * eta2 * (de2 + z2) + eta2 * eta2;
    let d = -4.0 * de * num / den;
    let dnum = 2.0 * de;
    let dden = 4.0 * de * (de2 - z2) + 4.0 * eta2 * de;
    let dd = -4.0 * (num + de * dnum) / den + 4.0 * de * num * dden / (den * den);
    (d, dd)
}

pub fn real_response_and_deriv(de: &[f64], z: f64, eta: f64) -> (Vec<f64>, Vec<f64>) {
    de.iter()
        .map(|&de| real_response_and_deriv_single(de, z, eta))
        .unzip()
}

/// Derivative of the REST real-axis response factor w.r.t. the residue
/// frequency z at fixed transition energy `de`:
/// `d_C = -4 de num/den`, `num = de^2 - z^2 + eta^2`,
/// `den = (de^2 - z^2)^2 + 2 eta^2 (de^2 + z^2) + eta^4`.
pub fn dd_response_dz_single(de: f64, z: f64, eta: f64) -> f64 {
    let de2 = de * de;
    let z2 = z * z;
    let eta2 = eta * eta;
    let num = de2 - z2 + eta2;
    let den = (de2 - z2) * (de2 - z2) + 2.0 * eta2 * (de2 + z2) + eta2 * eta2;
    let dnum = -2.0 * z;
    let dden = -4.0 * z * (de2 - z2) + 4.0 * eta2 * z;
    -4.0 * de * (dnum * den - num * dden) / (den * den)
}

/// Build the low-rank factor of REST's real-symmetric response at `omega`.
///
/// `qt` are the metric-transformed (`S q`) occupied-virtual pairs
/// [naux, n_trans] with `ia = i + a*nocc`.
/// Build the imaginary-axis and real-axis low-rank factor sets from the occupied-virtual block
/// `qtia` (shared by the engine constructor and the lazy rebuild after `evict_qt`).
#[allow(clippy::too_many_arguments)]
/// Low-rank factor of **one** real-axis grid point (see
/// [`GwGradEngine::ensure_lr_grid_factor`]).
fn build_lr_factor_at(
    qtia: &[f64],
    naux: usize,
    nov: usize,
    de_ia: &[f64],
    z: f64,
    eta: f64,
    tol: f64,
) -> LrFactor {
    build_lr_factor(qtia, naux, nov, de_ia, z, 'R', eta, tol)
}

fn build_lr_factor(
    qt: &[f64],
    naux: usize,
    ntrans: usize,
    de: &[f64],
    omega: f64,
    part: char,
    eta: f64,
    tol: f64,
) -> LrFactor {
    let d = rho_response_diagonal(de, omega, part, eta);
    // chi0 = sum_ia qt_ia d_ia qt_ia^T  (blocked GEMM accumulation)
    let mut chi0 = MatrixFull::new([naux, naux], 0.0);
    let block = 512usize.min(ntrans).max(1);
    let mut c0 = 0;
    while c0 < ntrans {
        let c1 = (c0 + block).min(ntrans);
        let nb = c1 - c0;
        let mut scaled = vec![0.0f64; naux * nb];
        for k in 0..nb {
            let s = d[c0 + k];
            for r in 0..naux {
                scaled[r + k * naux] = qt[r + (c0 + k) * naux] * s;
            }
        }
        let am = to_mat(&qt[c0 * naux..c1 * naux], naux, nb);
        let bm = to_mat(&scaled, naux, nb);
        _dgemm_full(&am, 'N', &bm, 'T', &mut chi0, 1.0, 1.0);
        c0 = c1;
    }
    let (vals, vecs) = eigh_symmetric(&chi0.data, naux);
    // RPA eigenvalues lam = lam0/(1-lam0), keep |lam| > tol, |lam|-descending
    let mut order: Vec<usize> = (0..naux).collect();
    order.sort_by(|&a, &b| {
        let lam = |v: f64| (v / (1.0 - v)).abs();
        lam(vals[b]).partial_cmp(&lam(vals[a])).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut eigval = Vec::new();
    let mut cols = Vec::new();
    for &v in &order {
        let lam = vals[v] / (1.0 - vals[v]);
        if lam.abs() <= tol {
            break;
        }
        eigval.push(lam);
        for r in 0..naux {
            cols.push(vecs[r + v * naux]);
        }
    }
    if std::env::var("REST_GWGRAD_LRDBG").is_ok() {
        // how close is this (node) frequency to an RPA pole of the low-rank
        // Dyson equation `lam = lam0/(1-lam0)`?  `min|1-lam0|` is the distance
        // of the closest retained `chi0` eigenvalue to the pole.
        let max_lam = eigval.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let min_gap = order
            .iter()
            .map(|&v| (1.0 - vals[v]).abs())
            .fold(f64::INFINITY, f64::min);
        println!(
            "[lr] part={part} z={omega:.8} keep={} max|lam|={max_lam:.3e} min|1-lam0|={min_gap:.3e}",
            eigval.len()
        );
    }
    LrFactor { omega, eigvec: cols, eigval, n_aux: naux }
}

impl LrFactor {
    /// Response `y = (J - Q(omega))^-1 rhs` in the raw representation:
    /// `y = S (qt_rhs + U diag(lam) U^T qt_rhs)` with `qt_rhs = S rhs`.
    /// `rhs` holds `n_col` right-hand sides of length naux (col-major).
    pub fn response_vectors(&self, s: &[f64], rhs: &[f64], n_col: usize) -> Vec<f64> {
        self.response_vectors_parts(s, rhs, n_col).2
    }

    /// [`Self::response_vectors`] together with its two intermediates, so a
    /// caller that wants to store the result can keep only `qt` (half the size)
    /// and rebuild `y0`/`y` on demand.  For the empty-eigenvalue factor the
    /// result is `qt` itself and both later entries are empty.
    ///
    /// With `y0 = s * s * rhs` (the metric solve) the identities are
    /// `qt = s * rhs = s * y0 * s` (element-wise, `s` diagonal) and
    /// `y = s * (qt + E diag(eig) E^T qt)`.
    pub fn response_vectors_parts(
        &self,
        s: &[f64],
        rhs: &[f64],
        n_col: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let naux = self.n_aux;
        let qt = gemm_nn(s, naux, naux, rhs, naux, n_col);
        if self.eigval.is_empty() {
            return (qt.clone(), Vec::new(), qt);
        }
        let corr = self.correction_from_qt(&qt, n_col);
        let mut sum = qt.clone();
        for (a, b) in sum.iter_mut().zip(corr.iter()) {
            *a += b;
        }
        let y = gemm_nn(s, naux, naux, &sum, naux, n_col);
        (qt, sum, y)
    }

    /// Low-rank correction `E diag(eig) E^T qt` of the response, i.e.
    /// `y - y0 = s * corr` in the notation of [`Self::response_vectors_parts`].
    ///
    /// The self-energy value only needs the *element-wise* product of this
    /// correction with `qt` (`w = sum_P qt[P,m] corr[P,m]`), so a caller that
    /// does not want the response vectors never has to form them.
    pub fn correction_from_qt(&self, qt: &[f64], n_col: usize) -> Vec<f64> {
        let naux = self.n_aux;
        if self.eigval.is_empty() {
            return vec![0.0f64; naux * n_col];
        }
        let nk = self.eigval.len();
        let mut proj = gemm_nt(&self.eigvec, naux, nk, qt, naux, n_col);
        for c in 0..n_col {
            for v in 0..nk {
                proj[v + c * nk] *= self.eigval[v];
            }
        }
        gemm_nn(&self.eigvec, naux, nk, &proj, nk, n_col)
    }

    /// `w_c = sum_P qt[P,c] corr[P,c]` — the `W_c` value of each of the `n_col`
    /// right-hand sides whose metric intermediate `qt = s_metric * rhs` is
    /// given — **without** forming the correction, using
    /// `qt^T corr = sum_v eigval_v (E_v^T qt)^2`.
    ///
    /// The self-energy *value* reads nothing else from the response, so this
    /// skips the back-transformation gemm of [`Self::correction_from_qt`] on
    /// the Newton path (where only the value is needed).
    pub fn response_value_from_qt(&self, qt: &[f64], n_col: usize) -> Vec<f64> {
        let naux = self.n_aux;
        let mut w = vec![0.0f64; n_col];
        if self.eigval.is_empty() {
            return w;
        }
        let nk = self.eigval.len();
        let proj = gemm_nt(&self.eigvec, naux, nk, qt, naux, n_col);
        for c in 0..n_col {
            let mut acc = 0.0f64;
            for v in 0..nk {
                let pv = proj[v + c * nk];
                acc += self.eigval[v] * pv * pv;
            }
            w[c] = acc;
        }
        w
    }

    /// Rebuild the response vectors of a low-rank factor from the stored
    /// intermediates `qt` (see [`Self::response_vectors_parts`]).
    pub fn response_vectors_from_qt(&self, s: &[f64], qt: &[f64], n_col: usize) -> Vec<f64> {
        let naux = self.n_aux;
        let corr = self.correction_from_qt(qt, n_col);
        if self.eigval.is_empty() {
            return qt.to_vec();
        }
        let mut sum = qt.to_vec();
        for (a, b) in sum.iter_mut().zip(corr.iter()) {
            *a += b;
        }
        gemm_nn(s, naux, naux, &sum, naux, n_col)
    }
}

// ---------------------------------------------------------------------------
// per-geometry GW gradient engine
// ---------------------------------------------------------------------------

/// One low-rank response factor that contributes to a residue pole.
///
/// The exact-residue and nearest-grid protocols use a single entry per pole;
/// the linear-interpolation protocol uses the two bracketing grid factors, so
/// the pole's response is the convex combination
/// `(1-t) W(z_lo) + t W(z_hi)` and the gradient of each factor is pulled back
/// with its own weight.
#[derive(Clone)]
pub struct ResidueEntry {
    /// signed weight `sign(e_m - omega) * pole_factor * split` (`split` is
    /// `1` for a single-entry protocol and `1-t` / `t` for the linear one)
    pub weight: f64,
    /// the frequency this factor was built at
    pub z: f64,
    /// `d node(z) / d de_max` of this factor's grid node (`0` in the
    /// exact-residue protocol and whenever the grid is pinned).
    pub node_sens: f64,
    /// response vector `y = (J - Q)^{-1} q_nm`
    pub y: Vec<f64>,
    /// metric partner `y0 = J^{-1} q_nm`
    pub y0: Vec<f64>,
}

/// One active residue pole of the real-axis contour term, as consumed by the
/// gradient pullback.
#[derive(Clone)]
pub struct Residue {
    /// mean-field pole index `m`
    pub m: usize,
    /// `sign(e_m - omega) * pole_factor` of the pole (the sum of the entry
    /// weights), used by the static-subtraction bookkeeping
    pub pole_weight: f64,
    /// `d W_m / d omega` of the **interpolation weight**, divided by the
    /// `-zsign = -sign(e_m - omega)` that the pullback applies, i.e.
    /// plus, with `smooth_pole`, the `d(pole_factor)/dz * W` half of the same
    /// derivative (the historical hard ladder has a zero there).
    ///
    /// * `exact`   : `pole_weight * dW/dz` (the factor frequency follows `e_m`);
    /// * `linear`  : `pole_weight * (W_hi - W_lo)/(z_hi - z_lo)`;
    /// * `nearest` : `0` (a grid node does not move with `e_m`).
    pub weight_slope: f64,
    /// Interpolation weight `t` of the upper bracketing node (`0` when the
    /// pole has a single entry).  Together with `secant` it lets the pullback
    /// rebuild the grid-node channel `d W_m/d de_max` from the entries'
    /// response vectors, so the (comparatively expensive) `d W/dz` contraction
    /// is only paid once per cache instead of once per Newton step.
    pub interp_t: f64,
    /// Secant `(W_hi - W_lo)/(z_hi - z_lo)` of the linear rule (`0` otherwise)
    pub secant: f64,
    pub entries: Vec<ResidueEntry>,
}

