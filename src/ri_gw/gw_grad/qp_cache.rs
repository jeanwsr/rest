/// One converged QP state: energy, renormalisation factor and the response
/// data needed by the pullback.
#[derive(Clone)]
pub struct QpCache {
    pub target: usize,
    /// the quasiparticle energy this cache describes
    pub omega: f64,
    /// The frequency the self-energy was evaluated at: the QP root for a solved
    /// state, and the **mean-field** `e_n` for the static on-shell fallback
    /// ([`Self::static_fallback`]).  Everything in the pullback that reads
    /// `omega - e_m` (the axis kernel, `pole_factor`, the residue sign) must use
    /// this, not [`Self::omega`].
    pub freq: f64,
    /// `1/(1 - d Sigma_c/d omega)` at [`Self::freq`] for a solved root, and
    /// exactly `1` for the static fallback.
    pub z_factor: f64,
    /// `d Sigma_c/d omega` at [`Self::freq`] — the quantity behind
    /// [`Self::z_factor`].  The static fallback's gradient needs it as the
    /// weight of `d e_n/dR`.
    pub dsigma_domega: f64,
    /// The QP energy is the **static on-shell** value `consts_n + Re Sigma_c(e_n)`
    /// (no root search, `Z = 1`) rather than a root of the QP equation.
    pub static_fallback: bool,
    /// The QP energy comes from the **linearized (one-step Newton)** functional
    /// at the mean-field frequency `freq = e_n`, `Z = 1/(1 - dSigma_c/domega)`,
    /// `omega = e_n + Z * lin_shift`.  Its gradient carries an extra
    /// `Z^2 * lin_shift * d_R d_omega Sigma_c` term (the omega-derivative
    /// pullback channel) on top of the usual `Z`-scaled one.
    pub linearized: bool,
    /// `consts_n + Re Sigma_c(e_n) - e_n`, the bracket the linearized branch
    /// multiplies by `Z`.  Zero for every other branch.
    pub lin_shift: f64,
    /// W_c[n, m](iu) at every quadrature point [n_freq][nmo]
    pub w_imag: Vec<Vec<f64>>,
    /// (u, y[naux*nmo], y0[naux*nmo]) per quadrature point.  Empty when the
    /// cache stores [`QpCache::m_imag`] instead.
    pub y_imag: Vec<(f64, Vec<f64>, Vec<f64>)>,
    /// **Compact** interaction storage: the single metric intermediate
    /// `m = s_metric * q_n` (`[naux, nmo]`, col-major).  It does *not* depend
    /// on the quadrature frequency, and both `y0 = s_metric * m` and
    /// `y = y0 + s_metric * corr(m, u)` are functions of it
    /// (`GwGradEngine::row_response_from_m`), so one `[naux, nmo]` buffer replaces
    /// the `2 * nquad * naux * nmo` doubles of `y_imag` — a factor
    /// `2*nquad` (40x at the production 20-point quadrature) per cache.  Used
    /// by the BSE gradient, whose caches are the dominant memory term.
    pub m_imag: Vec<f64>,
    /// Static row `W_c[n, m](0)` of the static-subtracted functional
    /// (`GwGradConfig::subtracted`); empty when the engine runs raw/full-CD.
    pub w0: Vec<f64>,
    /// Static response row `y0s = (J - Q(0))^-1 q_n` (`[naux, nmo]`), stored
    /// only in the full (`y_imag`) representation — the compact caches rebuild
    /// it from `m_imag` through the static low-rank factor.  Empty when the
    /// engine runs raw/full-CD.
    pub y0_static: Vec<f64>,
    /// active residue poles (see [`Residue`])
    pub residues: Vec<Residue>,
    /// Analytic-continuation (Padé) response data; empty for the CD functional.
    pub ac: AcQpData,
    /// CD-WAC (Padé continuation of `W`) response data; `None` otherwise.
    pub wac: Option<WacQpData>,
}

/// Derivative information of the analytic-continuation (Padé) self-energy of
/// one QP cache.
///
/// The AC self-energy is a **linear functional of the imaginary-axis screened
/// rows** `W_c[n,m](iu_k)` (the Padé approximant is built from the samples
/// `Sigma_c(i lambda_s)`, each of which is a linear combination of those rows).
/// The pullback therefore only needs the per-`(k, m)` coefficient of that
/// functional, plus the explicit dependence on the mean-field energies
/// (`e_m` and the Fermi shift `ef`).  See [`crate::ri_gw::ac_grad`] for the
/// derivative construction.
#[derive(Clone, Debug, Default)]
pub struct AcQpData {
    /// `d Re Sigma_c / d W_c[n,m](iu_k)`, `[n_quad][nmo]`.  Empty when the
    /// cache uses the contour-deformation functional.
    pub w_coeff: Vec<Vec<f64>>,
    /// `d Re Sigma_c / d e_m` at fixed `W` (kernel derivative plus the
    /// `ef(R)` chain, `ef = (e_HOMO + e_LUMO)/2`), `[nmo]`.
    pub be: Vec<f64>,
    /// `d/d omega` of [`Self::w_coeff`], the omega channel of the linearized
    /// branch, `[n_quad][nmo]`.
    pub w_coeff_omega: Vec<Vec<f64>>,
    /// `d/d omega` of [`Self::be`], `[nmo]`.
    pub be_omega: Vec<f64>,
    /// `d^2 Re Sigma_c / d omega^2` (the `d e_n/dR` chain of the linearized
    /// omega channel).
    pub d2sigma_domega2: f64,
}

impl AcQpData {
    /// `true` when this cache carries AC (Padé) response data.
    pub fn is_ac(&self) -> bool {
        !self.w_coeff.is_empty()
    }
}

/// One active residue of the CD-WAC functional.
///
/// The Padé approximant of `W_c,nm` is a linear combination of the reference
/// values with coefficients `alpha[p] = d Re P_m / d W_nm(z_p)`; `weight` is
/// the signed CD pole factor and `beta` the explicit `d/d e_m` of the residue
/// argument (the dependence of the reference values on `e_m` flows through the
/// response pullback).
#[derive(Clone, Debug)]
pub struct WacEntry {
    pub m: usize,
    pub weight: f64,
    pub alpha: Vec<f64>,
    pub beta: f64,
}

/// CD-WAC response data of one QP cache.
#[derive(Clone, Debug, Default)]
pub struct WacQpData {
    pub entries: Vec<WacEntry>,
    pub n_ref: usize,
    /// `d R / d omega_max` of the automatic reference grid (0 when pinned)
    pub node_coeff: f64,
}

/// Imaginary-axis samples of one AC (Padé) self-energy row.
struct AcSetup {
    /// `lambda_s`, the imaginary frequency of every sample
    lambdas: Vec<f64>,
    /// `Sigma_c,nn(i lambda_s)`
    values: Vec<Complex<f64>>,
}

/// Complex RPA response `chi0(z) = sum_ia qt_ia d_ia(z) qt_ia^T` in the
/// metric-transformed representation, with
/// `d_ia(z) = -4 de_ia/(de_ia^2 - z^2)` (the same convention as the CD
/// imaginary-axis response and its Lorentzian real-axis limit).
fn wac_chi0_weights(
    qtia: &[f64],
    naux: usize,
    nov: usize,
    w: &[Complex<f64>],
) -> MatrixFull<Complex<f64>> {
    let mut scaled = vec![Complex::new(0.0, 0.0); naux * nov];
    for col in 0..nov {
        let d = w[col];
        for r in 0..naux {
            let q = qtia[r + col * naux];
            scaled[r + col * naux] = Complex::new(q * d.re, q * d.im);
        }
    }
    let qm_data: Vec<Complex<f64>> = qtia.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let qm = MatrixFull::from_vec([naux, nov], qm_data).unwrap();
    let sm = MatrixFull::from_vec([naux, nov], scaled).unwrap();
    let mut chi = MatrixFull::from_vec(
        [naux, naux],
        vec![Complex::new(0.0, 0.0); naux * naux],
    )
    .unwrap();
    chi.lapack_zgemm(
        &sm,
        &qm,
        'N',
        'T',
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
    );
    chi
}

fn wac_chi0(
    qtia: &[f64],
    naux: usize,
    nov: usize,
    de_ia: &[f64],
    z: Complex<f64>,
) -> MatrixFull<Complex<f64>> {
    let w: Vec<Complex<f64>> = de_ia
        .iter()
        .map(|&de| {
            let den = Complex::new(de * de, 0.0) - z * z;
            Complex::new(-4.0 * de, 0.0) / den
        })
        .collect();
    wac_chi0_weights(qtia, naux, nov, &w)
}

/// Reference data of the CD-WAC continuation of `W`.
///
/// `z[p]` are the reference frequencies (imaginary axis first, then the
/// near-real points) and `u[p] = z[p]^2` the Padé nodes.  `lu[p]` factors the
/// complex screening matrix `I - chi0(z[p])` in the metric-transformed
/// (`S q`) representation, so that the row of any target orbital at every
/// reference frequency is one triangular solve away.
struct WacReference {
    z: Vec<Complex<f64>>,
    u: Vec<Complex<f64>>,
    lu: Vec<CdLuFactor>,
    /// number of leading entries that lie on the imaginary axis (their values
    /// are the already available `QpRowTable::w_imag`)
    n_imag: usize,
    /// quadrature index of every imaginary-axis reference
    imag_index: Vec<usize>,
    /// `d u_p / d omega_max` for the automatic (geometry-dependent) real grid;
    /// all zero when `wac_omega_max` is pinned by the input card.
    du_domega_max: Vec<Complex<f64>>,
    /// orbital indices that set the automatic `omega_max = e_last - e_first`
    omega_max_movers: (usize, usize),
    /// `d z_p / d omega_max` of the reference frequencies (0 when pinned)
    dz_domega_max: Vec<f64>,
}

/// Value and response coefficients of the AC self-energy at one frequency.
struct AcParts {
    sigma: f64,
    dsigma_domega: f64,
    d2sigma_domega2: f64,
    w_coeff: Vec<Vec<f64>>,
    be: Vec<f64>,
    w_coeff_omega: Vec<Vec<f64>>,
    be_omega: Vec<f64>,
}

#[derive(Clone, Copy)]
pub struct SyncGw<'a, 'b>(pub &'a GwGradEngine<'b>);

unsafe impl Send for SyncGw<'_, '_> {}
unsafe impl Sync for SyncGw<'_, '_> {}

impl<'a, 'b> std::ops::Deref for SyncGw<'a, 'b> {
    type Target = GwGradEngine<'b>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'a, 'b> SyncGw<'a, 'b> {
    /// Inherent (shadowing) forwarders.
    ///
    /// The `Deref` impl would send these calls through `GwGradEngine<'b>`, and
    /// `&GwGradEngine<'b>` is not `Sync`, so every closure that captures the
    /// wrapper must reach the engine through an inherent method — the receiver
    /// then stays `SyncGw`, which *is* `Sync`.
    #[inline]
    pub fn qp_pullback_scratch(&self, nk: usize) -> QpPullbackScratch {
        self.0.qp_pullback_scratch(nk)
    }
    #[inline]
    pub fn qp_pullback_scratch_banded(&self, nk: usize, nrow_pair: usize) -> QpPullbackScratch {
        self.0.qp_pullback_scratch_banded(nk, nrow_pair)
    }
    #[inline]
    pub fn qp_pullback_scratch_clear(&self, scr: &mut QpPullbackScratch) {
        self.0.qp_pullback_scratch_clear(scr)
    }
    #[inline]
    pub fn qp_pullback_scratch_reset_page(&self, scr: &mut QpPullbackScratch) {
        self.0.qp_pullback_scratch_reset_page(scr)
    }
    #[inline]
    pub fn qp_pullback_scratch_merge(&self, dst: &mut QpPullbackScratch, src: QpPullbackScratch) {
        self.0.qp_pullback_scratch_merge(dst, src)
    }
    #[inline]
    pub fn qp_pullback_chunk_public(
        &self,
        caches: &[&QpCache],
        weights: &[f64],
        col0: usize,
        scr: &mut QpPullbackScratch,
    ) {
        self.0.qp_pullback_chunk_public(caches, weights, col0, scr)
    }
    #[inline]
    pub fn qp_pullback_chunk_omega_public(
        &self,
        caches: &[&QpCache],
        weights: &[f64],
        col0: usize,
        scr: &mut QpPullbackScratch,
    ) {
        self.0
            .qp_pullback_chunk_omega_public(caches, weights, col0, scr)
    }
    #[inline]
    pub fn qp_pullback_streaming_finish(
        &self,
        scr: QpPullbackScratch,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        self.0.qp_pullback_streaming_finish(scr)
    }
    #[inline]
    pub fn naux(&self) -> usize {
        self.0.naux
    }
    #[inline]
    pub fn nmo(&self) -> usize {
        self.0.nmo
    }
    #[inline]
    pub fn d_atom_blocks(&self, atm: usize) -> AtomDerivBlocks {
        self.0.raw.d_atom_blocks(atm)
    }
    #[inline]
    pub fn d_j_atom(&self, atm: usize, comp: usize) -> Vec<f64> {
        self.0.raw.d_j_atom(atm, comp)
    }
    #[inline]
    pub fn d_i_from_blocks(&self, blocks: &AtomDerivBlocks, comp: usize) -> Vec<f64> {
        self.0.raw.d_i_from_blocks(blocks, comp)
    }
    #[inline]
    pub fn canonical_skeleton_mo(
        &self,
        atm: usize,
        comp: usize,
        h1r: &[f64],
        s1r: &[Vec<f64>],
        di: &[f64],
        dj: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        self.0.canonical_skeleton_mo(atm, comp, h1r, s1r, di, dj)
    }
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn canonical_finish(
        &self,
        f1_skel_mo: &[f64],
        s1_mo: &[f64],
        mo1: &[f64],
        vxc_d1_mo: Option<&[f64]>,
        fxc_cache: Option<&crate::dft::response::FxcHessianCache>,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        self.0
            .canonical_finish(f1_skel_mo, s1_mo, mo1, vxc_d1_mo, fxc_cache)
    }
}

/// The grid-derived `v_xc` data of a GW/BSE gradient reference.
///
/// Both fields come from the SCF's DFT grid tabulated AO (`generate_vxc_rayon`), which is
/// why [`GwGradEngine::precompute_vxc`] exists: a driver that owns the `SCF` mutably can
/// compute them, release the (multi-GB) grid tables and only then build the engine.  See
/// [`Grids::release_tabulated_ao`](crate::dft::Grids::release_tabulated_ao).
#[derive(Clone, Debug, Default)]
pub struct GwVxc {
    /// `v_xc` diagonal in the MO basis, `[nmo]` (`<n| v_xc |n>`); all-zero for HF.
    pub nn: Vec<f64>,
    /// Full `v_xc` matrix in the MO basis `[nmo*nmo]` col-major; empty for an HF reference.
    pub mo: Vec<f64>,
}

pub struct GwGradEngine<'a> {
    pub scf: &'a SCF,
    pub config: GwGradConfig,
    pub raw: RawRiTensors,
    pub nao: usize,
    pub naux: usize,
    pub nmo: usize,
    pub nocc: usize,
    pub natm: usize,
    /// mean-field orbital energies (poles and screening energies of G0W0)
    pub e: Vec<f64>,
    /// Quasiparticle energies handed in by the caller (`scf_data.gwqp.0`, the
    /// vector the BSE Hamiltonian is built from).  With
    /// [`GwGradConfig::pin_qp_from_checkpoint`] set and an entry for orbital
    /// `n`, the cache of `n` is built **at** that energy -- no root search -- so
    /// the gradient differentiates the same quasiparticle solution the energy
    /// path used.  Empty = solve the QP equation here (historical behaviour).
    pub pinned_qp: Vec<f64>,
    /// MO coefficients [nao, nmo] col-major
    pub c_mo: MatrixFull<f64>,
    /// raw three-center MO integrals (pair-symmetrised).
    ///
    /// Behind a [`RefCell`] so that a task which no longer needs them can release
    /// `naux * nmo^2` doubles (874 MB at naphthalene/def2-TZVP) with [`Self::evict_q`];
    /// readers go through [`Self::with_q`], which rebuilds them from `raw.i3` on demand.
    pub q: RefCell<Vec<f64>>,
    /// the same integrals repacked pair-slowest (`[nmo, nmo*naux]` col-major
    /// view) so that the `U^T q + q U` contraction of `qx_from_u` is one
    /// large gemm
    pub q_pairmajor: Vec<f64>,
    /// auxiliary Coulomb metric
    pub j_mat: MatrixFull<f64>,
    /// lower Cholesky factor of the auxiliary metric (computed once; used
    /// with multi-right-hand-side triangular solves for the skeleton
    /// derivatives, matching scipy `solve(..., assume_a='pos')`)
    pub j_chol: MatrixFull<f64>,
    /// J^{-1/2}
    pub s_metric: Vec<f64>,
    /// Metric-transformed integrals, restricted to the **occupied MO columns**
    /// (`naux * nmo * nocc`, same pair-slowest/aux-fastest layout as `q`): this is all the
    /// QP-cache construction reads (`consts` needs `qt[P, n, i]` for `i < nocc`). Evictable like
    /// [`Self::q`] (see [`Self::evict_qt`]).
    pub qt: RefCell<Vec<f64>>,
    /// raw occupied-virtual block [naux, nov], ia = i + a*nocc
    pub qia: Vec<f64>,
    /// screening transition energies de_ia = e_a - e_i
    pub de_ia: Vec<f64>,
    /// imaginary-axis quadrature (omega, weight)
    pub quad: Vec<(f64, f64)>,
    /// real-axis sampling grid (REST protocol)
    pub z_grid: Vec<f64>,
    /// `d z_grid[i] / d de_max` (see [`RealAxisGrid::node_sens`])
    pub z_grid_sens: Vec<f64>,
    /// the transition energy that sets `de_max` (see [`GridMover`])
    pub z_grid_mover: Option<GridMover>,
    /// Cached low-rank factors on the imaginary axis. Evictable with `evict_qt`: only the QP-cache
    /// construction reads them, and `with_lr` rebuilds them from `qtia` on demand.
    pub lr_imag: RefCell<Vec<LrFactor>>,
    /// Cached low-rank factors on the real-axis grid (evictable like
    /// [`Self::lr_imag`]).
    ///
    /// The real-axis factors are built **on demand**, one grid point at a time
    /// ([`Self::ensure_lr_grid_factor`]): the contour part of a self-energy
    /// reads at most a handful of them (`nearest_grid_value` maps every pole of
    /// a row onto the grid), while building the whole grid costs one
    /// `naux x naux` eigendecomposition per point — the dominant term of the
    /// engine construction (7.0 s of 9.2 s at naphthalene/def2-SVP).
    /// Real-axis low-rank factor per grid point, **each slot independently
    /// lazily initialised**.
    ///
    /// `OnceLock` rather than `RefCell<Vec<Option<_>>>` because a self-energy row
    /// only ever reads the one or two grid nodes bracketing its pole
    /// (`lr_bracket`), while building the *whole* grid costs
    /// `z_grid.len() * naux^2` doubles and dominates the engine construction.
    /// Per-slot `get_or_init` keeps that lazy **and** makes the fill safe to
    /// trigger from several threads at once — which is what a parallel QP-cache
    /// sweep needs (a single `RefCell` forced the caller to pre-materialise every
    /// node, measured at vitamin C/def2-TZVP as 428 s of build against 75 s
    /// serial).
    pub lr_grid: Vec<OnceLock<LrFactor>>,
    /// Low-rank factor of the **static** (`z = 0`) response, the third
    /// ingredient of the static-subtracted self-energy
    /// ([`GwGradConfig::subtracted`]).  Built on demand from `qtia` with the
    /// same real-axis Lorentzian diagonal the production protocol evaluates at
    /// grid point `z = 0` (`d_ia = -4 de_ia/(de_ia^2 + eta^2)`); evicted with
    /// [`Self::evict_qt`] and rebuilt by [`Self::ensure_lr_static`].
    pub lr_static: RefCell<Option<LrFactor>>,
    /// metric-transformed occupied-virtual block (kept for on-the-fly factors).
    /// Evictable together with `qt` ([`Self::evict_qt`]).
    pub qtia: RefCell<Vec<f64>>,
    /// CD-WAC reference screening factors (built lazily, once per geometry).
    /// Only [`GwVariant::Wac`] reads them.
    wac_ref: RefCell<Option<WacReference>>,
    /// hybrid parameter alpha of the reference, as used by REST's GW
    /// `consts` (`dfa_hybrid_scf`)
    pub hybrid: f64,
    /// exchange scaling of the SCF Fock response (1.0 for an HF reference,
    /// `dfa_hybrid_scf` otherwise) — matches `dft::response::gen_vind`
    pub fock_hyb: f64,
    /// v_xc diagonal in the MO basis (cached once)
    vxc_nn: Vec<f64>,
    /// Full v_xc matrix in the MO basis `[nmo*nmo]` col-major (empty for an
    /// HF reference).  The static term `B = (1-alpha) Sigma_x - v_xc` of the
    /// QP equation is differentiated through the `b_x` covector, whose
    /// orbital-response piece `2 sum_p U[p,n] v_xc[p,n]` reads this matrix;
    /// the exchange part flows through the (bq, bJ) pullback instead.
    vxc_mo: Vec<f64>,
    /// symmetry-protected degenerate shells of `e` (see
    /// [`degenerate_shells`]); shells of size 1 are non-degenerate orbitals
    pub shells: Vec<Vec<usize>>,
    /// index of the shell containing each orbital
    pub shell_of: Vec<usize>,
}

    /// Bytes of one fully built [`QpCache`]: `nquad` pairs of `[naux, nmo]` vectors
/// plus the residue vectors.  The residue count is not known before the cache
/// is built, so the estimate is the imaginary-axis part, which dominates
/// (`nres` is at most a few).
pub fn qp_cache_bytes_estimate(naux: usize, nmo: usize, nquad: usize) -> usize {
    nquad * 2 * naux * nmo * 8
}

/// One chunk of a **streaming** QP pullback: the running covectors
    /// `(bq, bJ, be, bB)` plus the deferred polarizability workspace.
    ///
    /// The non-streaming [`Self::qp_pullback`] is the special case of a single
    /// chunk; the BSE gradient uses several chunks so that the QP caches (the
    /// largest geometry-fixed object of that path, `O(ncache·nquad·naux·nmo)`)
    /// never have to be resident all at once.  See
    /// [`Self::qp_pullback_streaming`].
    pub struct QpPullbackScratch {
        nk: usize,
        naux: usize,
        nmo: usize,
        nocc: usize,
        nov: usize,
        /// Width of the stored pair band: `bq` keeps only the packed pairs whose smaller index
        /// is `< nrow_pair` (see [`banded_pair_count`]); `nmo` means the full packed layout.
        nrow_pair: usize,
        bq: Vec<Vec<f64>>,
        bj: Vec<Vec<f64>>,
        be: Vec<Vec<f64>>,
        bb: Vec<Vec<f64>>,
        /// polarizability coefficient matrices `V[k][wi] = sum_m a_m y_m y_m^T`
        /// of the imaginary axis; they must survive the whole cache loop
        /// because every cache adds a rank-`nmo` contribution
        v_imag: Vec<Vec<Vec<f64>>>,
        /// static (`z = 0`) bucket of the subtracted functional,
        /// `V_static[k] = sum_m staticc_m y0s_m y0s_m^T`
        v_static: Vec<Vec<f64>>,
        /// residue frequency buckets `[k]`, parallel to `z_order[k]`
        v_res: Vec<Vec<Vec<f64>>>,
        z_order: Vec<Vec<f64>>,
        /// `(-4 de/(de^2+u^2), -4(u^2-de^2)/(de^2+u^2)^2)` per quadrature point
        imag_factors: Vec<(Vec<f64>, Vec<f64>)>,
        /// CD-WAC reference-frequency buckets `V[k][p]` (reused by every cache)
        v_wac: Vec<Vec<Vec<f64>>>,
    }

    impl QpPullbackScratch {
        /// `true` when every per-state buffer has been allocated (a
        /// [`Self::default_empty`] placeholder has not).
        pub fn is_shaped(&self, nk: usize) -> bool {
            self.bq.len() == nk
                && self.bq.iter().all(|v| !v.is_empty())
                && self.bj.len() == nk
                && self.bj.iter().all(|v| !v.is_empty())
                && self.be.len() == nk
                && self.bb.len() == nk
                && self.v_imag.len() == nk
                && self.v_static.len() == nk
                && self.v_res.len() == nk
                && !self.imag_factors.is_empty()
        }

        /// A scratch with the right shapes but **no** quadrature tables.
        ///
        /// Only used as the placeholder that a parallel pullback worker swaps in
        /// after handing its accumulated partial to the merge: the worker calls
        /// [`GwGradEngine::qp_pullback_scratch_clear`] before its next page, so
        /// the frequency tables are never read.
        pub fn default_empty(nk: usize) -> Self {
            QpPullbackScratch {
                nk,
                naux: 0,
                nmo: 0,
                nocc: 0,
                nov: 0,
                nrow_pair: 0,
                bq: Vec::new(),
                bj: Vec::new(),
                be: Vec::new(),
                bb: Vec::new(),
                v_imag: Vec::new(),
                v_static: Vec::new(),
                v_res: Vec::new(),
                z_order: Vec::new(),
                imag_factors: Vec::new(),
                v_wac: Vec::new(),
            }
        }

        fn new(engine: &GwGradEngine, nk: usize) -> Self {
            Self::new_banded(engine, nk, engine.nmo)
        }

        /// As [`Self::new`], keeping only the pair band `min(p, q) < nrow_pair` of `bq`.
        fn new_banded(engine: &GwGradEngine, nk: usize, nrow_pair: usize) -> Self {
            let (naux, nmo, nocc) = (engine.naux, engine.nmo, engine.nocc);
            let nrow_pair = nrow_pair.min(nmo);
            let nov = nocc * (nmo - nocc);
            let imag_factors = engine
                .quad
                .iter()
                .map(|&(u, _)| {
                    let mut d = vec![0.0f64; nov];
                    let mut dd = vec![0.0f64; nov];
                    for (idx, &de) in engine.de_ia.iter().enumerate() {
                        d[idx] = -4.0 * de / (de * de + u * u);
                        dd[idx] =
                            -4.0 * (u * u - de * de) / ((de * de + u * u) * (de * de + u * u));
                    }
                    (d, dd)
                })
                .collect();
            let nquad = engine.quad.len();
            let n_wac = if engine.config.gw_variant == GwVariant::Wac {
                engine.ensure_wac_reference();
                engine
                    .wac_ref
                    .borrow()
                    .as_ref()
                    .map(|r| r.z.len())
                    .unwrap_or(0)
            } else {
                0
            };
            QpPullbackScratch {
                nk,
                naux,
                nmo,
                nocc,
                nov,
                nrow_pair,
                // the pair covector accumulates directly in its packed
                // (lower-triangle) layout: the pullback writes the two
                // directions of a pair into the same slot and the finish pass
                // halves the off-diagonal pairs, which is exactly the
                // minimum-norm symmetric representative the symmetrisation used
                // to build — at half the memory.  Only the band
                // `min(p, q) < nrow_pair` is allocated; that band is a prefix of
                // the packed layout, and the covector is zero outside it (no QP
                // weight reaches those rows), so the arithmetic is unchanged.
                bq: vec![vec![0.0f64; naux * banded_pair_count(nmo, nrow_pair)]; nk],
                bj: vec![vec![0.0f64; naux * naux]; nk],
                be: vec![vec![0.0f64; nmo]; nk],
                bb: vec![vec![0.0f64; nmo]; nk],
                v_imag: vec![vec![vec![0.0f64; naux * naux]; nquad]; nk],
                v_static: vec![vec![0.0f64; naux * naux]; nk],
                v_res: vec![Vec::new(); nk],
                z_order: vec![Vec::new(); nk],
                imag_factors,
                v_wac: vec![vec![vec![0.0f64; naux * naux]; n_wac]; nk],
            }
        }
    }

