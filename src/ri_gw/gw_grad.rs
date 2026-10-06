//! Analytic nuclear gradients of low-rank contour-deformation (LR-CD) G0W0
//! quasiparticle energies.
//!
//! This module ports the validated PySCF implementation
//! `pyscf/gw/gw_lr_cd_grad.py` (+ `gw_cd_grad*.py`) to REST.  It reproduces
//! REST's own LR-CD G0W0 energy functional (`ri_gw::gw_calculations_lowrank`
//! / `newton_solver_lowrank_v2`) — the contour-deformation decomposition with
//! REST's real-symmetric screened response, REST's imaginary-axis quadrature
//! grids, REST's real-axis residue interpolation (`low_rank_interp`: nearest
//! grid point or linear between bracketing nodes, the latter by default) and
//! REST's finite-difference Newton root finder — and differentiates it exactly.
//!
//! By default the differentiated functional is the **static-subtracted**
//! self-energy of the full-CD engine (see [`GwCdGradEngine`]),
//! evaluated through the same low-rank response factors,
//!
//!     Sigma_sub = sum_m [ -1/pi sum_k w_k K(t_m, u_k) (W_m(iu_k) - W_m(0))
//!                         + (1/2 - f_m) W_m(0)
//!                         + s_m pf_m (W_m(zeta_m) - W_m(0)) ],
//!
//! with the *same* residue protocol (pole selection, `pole_factor` smoothing,
//! `low_rank_interp` interpolation) as REST's raw LR energy.  The static
//! subtraction removes the artificial pole-selection jump of the raw
//! decomposition at `omega = e_m`, making the QP energy — and therefore its
//! gradient — well defined even when a quasiparticle energy coincides with a
//! mean-field pole (degenerate shells, near-zero QP shifts).  Wherever no
//! pole is crossed the two forms are numerically identical:
//! `Sigma_sub - Sigma_raw = sum_m [A_m + 1/2 - f_m - s_m pf_m] W_m(0)` with
//! `A_m = 1/pi sum_k w_k K(t_m,u_k) = 1/2 sign(t_m) + O(1e-9)` on REST's
//! transformed Gauss-Legendre grids, so finite differences of the *raw*
//! production energy match the subtracted analytic gradient (validated in
//! `tests/gw_bse/test_gw_grad_subtracted.rs`).  `GwGradConfig::subtracted =
//! false` restores the legacy raw functional.
//!
//! Key design points
//! -----------------
//! * Everything is expressed in the **raw RI representation** `(mu nu | P)`
//!   (PySCF `gw_cd_grad` style): `q[P,pq] = sum_mu nu I C C`, metric
//!   `J[P,Q] = (P|Q)`.  The screened correlation part is
//!   `W_c[n,m](z) = q_nm^T [(J - Q(z))^-1 - J^-1] q_nm`, so no derivative of
//!   the `J^{-1/2}` metric factorisation ever appears.
//! * REST stores metric-transformed (`J^{-1/2}`) 3-center integrals; the
//!   low-rank spectral factorisation of REST's real-symmetric response,
//!   `W_c = S U diag(lam) U^T S` with `S = J^{-1/2}` and `lam = lam0/(1-lam0)`,
//!   provides the response vectors `y = (J - Q)^{-1} q` cheaply, exactly as
//!   PySCF's `gw_lr_cd_grad`.  The same low-rank truncation
//!   (`low_rank_tolerance`) as the energy is applied, so the analytic
//!   gradient matches finite differences of REST's low-rank energy.  Use a
//!   tight tolerance (1e-10) for validation runs.
//! * The pole selection, `pole_factor` smoothing, frequency grids, real-axis
//!   grid construction (de_max scan, linear/quadratic nodes) and the QP Newton
//!   iteration all follow REST's production low-rank path
//!   (`ri_gw::contour_rayon_lowrank`, `generate_real_axis_vchiv`,
//!   `newton_solver_lowrank`).
//! * [`GwGradConfig::low_rank_interp`] mirrors the energy path's
//!   `low_rank_interp` keyword.  The real-axis nodes are built from `de_max`,
//!   which depends on the reference geometry, so the gradient also carries the
//!   node-motion channel `d z_i / dR = (d z_i / d de_max)(d de_max / dR)`; the
//!   interpolation weight adds `d t / dR`.  Both are exact, and the FD
//!   validation in `tests/gw_bse/test_gw_bse_interp.rs` covers the two rules.
//! * The reference is a restricted closed-shell density-fitted calculation
//!   without frozen orbitals: HF, pure-DFA (GGA) and global-hybrid references
//!   are supported.  The static term `B_n = (1-alpha) Sigma_x,nn - v_xc,nn`
//!   of the QP equation is differentiated in full: the `(1-alpha)` RI
//!   exchange through the (bq, bJ) pullback, the v_xc part (explicit grid
//!   derivative, orbital rotation and XC-kernel density response) through
//!   the `b_x` covector of `canonical_response_batch` (port of
//!   `pyscf/gw/gw_cd_grad.py::_b_nuclear_deriv`).  Range-separated hybrids
//!   are rejected.
//! * The orbital response (U, eps1) is the full canonical CPHF response:
//!   the CP-HF equation is solved by REST's `analdrv` driver (`RHessSCF`
//!   block-Krylov over all perturbations, RI-JK response on `rimatr`), and
//!   the remaining blocks are restored from the differentiated Roothaan
//!   equation, exactly as `pyscf/gw/gw_cd_grad*.py::_cp_response_atom`.
//!
//! The module also provides [`GwCdGradEngine`], the port of the FULL
//! (non-low-rank) CD-G0W0 gradients of `pyscf/gw/gw_cd_grad*.py`: the
//! static-subtracted continuous self-energy with direct (naux x naux)
//! factorisation of the screening matrix — no low-rank truncation — and
//! exact moving residue frequencies.  See the section header below.

use crate::ri_jk::util::{get_cint_aux, get_cint_mol};
use crate::ri_rpa::{gauss_legendre_grids, logarithmic_grid, trans_gauss_legendre_grids};
use crate::scf_io::SCF;
use rest_libcint::prelude::*;
use rest_tensors::matrix::matrix_blas_lapack::{
    _dgemm_full, _dpotrf, _dsolve, _dsyev_inplace, _dtrtrs,
};
use tensors::{MathMatrix, MatrixFull};
use num_complex::Complex;

use std::cell::RefCell;
use std::os::raw::c_char;
use std::time::Instant;

use crate::ri_bse::bse_grad::grad_timing;

// sub-step accumulators of the canonical-response assembly (seconds):
// [0] j_deriv_from, [1] k_deriv_from, [2] compute_j_upper, [3] compute_k_upper
thread_local! {
    static T_TIMER: std::cell::RefCell<[f64; 4]> = const { std::cell::RefCell::new([0.0; 4]) };
    /// `[response, gram, accumulate, bq/be]` of the streaming QP pullback
    /// (`REST_GRAD_TIMING=1`).
    static T_PULL: std::cell::RefCell<[f64; 4]> = const { std::cell::RefCell::new([0.0; 4]) };
}

fn timer_add(idx: usize, t0: Instant) {
    if grad_timing() {
        T_TIMER.with(|c| c.borrow_mut()[idx] += t0.elapsed().as_secs_f64());
    }
}

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

// ---------------------------------------------------------------------------
// configuration
// ---------------------------------------------------------------------------

/// Numerical options of the LR-CD G0W0 gradient.  The defaults mirror REST's
/// `QuasiParticle` control block so the module reproduces `ri_gw` results.
#[derive(Clone, Debug)]
pub struct GwGradConfig {
    /// number of imaginary-axis quadrature points (REST `num_freq`)
    pub num_freq: usize,
    /// imaginary-axis grid selector, mirrors `mol.ctrl.freq_grid_type`
    pub freq_grid_type: usize,
    /// frequency cut-off for grids 1/2, mirrors `mol.ctrl.freq_cut_off`
    pub freq_cut_off: f64,
    /// Lorentzian broadening of the real-axis response (REST `cdgw_eta`)
    pub eta: f64,
    /// pole-detection tolerance (REST `cdgw_res_tol`)
    pub res_tol: f64,
    /// number of real-axis sampling points (REST `nomega_chi_real`)
    pub nomega_chi_real: usize,
    /// RPA eigenvalue truncation of the low-rank screening
    pub low_rank_tolerance: f64,
    /// 0 = linear real-axis grid, 1 = quadratic (REST `low_rank_grid_type`)
    pub grid_type: usize,
    /// REST `nomega_sigma` (de_max scan width)
    pub nomega_sigma: usize,
    /// REST `step_sigma` (de_max scan step)
    pub step_sigma: f64,
    /// REST `selfenergy_state_range` (de_max scan state window)
    pub selfenergy_state_range: usize,
    /// user override of the real-axis grid maximum; 0.0 = automatic
    pub omega_chi_max: f64,
    /// QP Newton tolerance (REST hardcodes 1e-5; use tight values for FD)
    pub qpe_tol: f64,
    /// QP Newton maximum iterations
    pub qpe_max_iter: usize,
    /// Evaluate residues at the exact frequencies |e_m - omega| instead of
    /// REST's nearest-grid quantisation.  The quantised protocol has a
    /// piecewise-constant residue term (piecewise differentiable gradient);
    /// the exact mode is the smooth continuous-CD limit (PySCF-like) and is
    /// better suited to finite-difference validation.
    pub exact_residue_z: bool,
    /// How the low-rank real-axis response is interpolated **between** the grid
    /// points, mirroring the energy path's `RealAxisVChiV::interp` and REST's
    /// `low_rank_interp` control keyword:
    ///
    /// * `0` — nearest grid point (piecewise constant, the historical rule and
    ///   MolGW's `sf_interpolate_vsqrt_chi_vsqrt`);
    /// * `1` — linear interpolation between the two bracketing grid points.
    ///
    /// The gradient must differentiate the *same* functional whose root the QP
    /// solver found, so this has to agree with the energy path.  REST's
    /// `low_rank_interp` defaults to `"linear"`, and the two rules only
    /// converge as the real-axis grid is refined (`nomega_chi_real`), so a
    /// mismatch shows up as an analytic/FD discrepancy that cannot be removed
    /// by tightening any other knob.  Ignored by the exact-residue protocol
    /// (`exact_residue_z`), which never evaluates the grid.
    pub low_rank_interp: usize,
    /// Enable the **graphical scan** as a rescue when the Newton iteration on the
    /// QP equation stalls (`GwGradConfig::qpe_graphical` selects it as the
    /// primary solver instead).
    ///
    /// With `false` the cascade is strictly `Newton -> linearized (one-step
    /// Newton) -> static on-shell`.  The graphical scan is derivative-free, so
    /// it can bracket a sign change that lies *inside* a residue jump -- a point
    /// where the self-energy is not differentiable and where the analytic
    /// gradient (which needs `dSigma_c/domega` and the node sensitivities) has
    /// no derivative to offer.  Turning it off keeps every branch of the cascade
    /// a smooth function of the geometry, at the price of the accuracy the scan
    /// would have recovered.
    pub graphical_rescue: bool,
    /// Replace the hard `0 / 0.5 / 1` residue `pole_factor` ladder by the C^1
    /// cubic switch with the same endpoints and the same `0.5` at `z = 0`.
    ///
    /// The ladder is a pair of genuine **value jumps** in `Sigma_c` at
    /// `|e_m - omega| = res_tol`, and a QP root of a high virtual sits right on
    /// top of its own pole (`z = 0`), so the finite difference of such a root
    /// does not converge (measured `|analytic - FD| ~ 15 * h` for orbital 23 of
    /// water).  The analytic gradient cannot represent a jump; the switch
    /// removes it, and `Residue::weight_slope` then also carries the missing
    /// `d(pole_factor)/dz * W` term, so the analytic gradient is again the exact
    /// derivative of the evaluated functional.
    pub smooth_pole: bool,
    /// `|E_KS|` above which the QP energy is taken from the **static on-shell**
    /// fallback instead of a root of the QP equation (REST
    /// `gw_switch_fallback_threshold`, default `1e6` = never):
    ///
    /// ```text
    /// E_qp = E_KS + Sigma_x - V_xc + Re Sigma_c(E_KS)      (Z = 1)
    /// ```
    ///
    /// MolGW's `EVSC` / `se%nomega = 0` rule.  The same evaluation is used when
    /// a root finder cannot bracket a crossing, and it is what the energy path
    /// (`ri_gw::scgw::single_orbital_gw` and its low-rank variants) returns in
    /// both situations.  `0.0` selects it for every orbital.
    ///
    /// The gradient is then the derivative of that functional, which is **not**
    /// the root gradient: the self-energy frequency is the mean-field `e_n`
    /// (it does not follow the QP energy) and there is no `Z` factor, while the
    /// explicit `e_n`-dependence adds `(d Sigma_c/d omega) d e_n/dR`.
    ///
    /// Only **occupied** levels are eligible: the rule fires when
    /// `e_n < 0` **and** `-e_n > static_fallback_threshold`.  Virtual levels
    /// are never forced onto it by this test.  Default **1.0 Ha**, so a deep
    /// core level (e.g. the O `1s` of water at -20.5 Ha) is treated on-shell
    /// while valence levels keep the root functional.
    pub static_fallback_threshold: f64,
    /// Tolerance (Hartree) for recognising symmetry-protected **degenerate**
    /// orbital shells of the reference (default `1e-8`).
    ///
    /// Inside a shell the canonical-orbital response
    /// `u[p,q] = (F1[p,q] - e_q S1[p,q]) / (e_q - e_p)` is singular, so the
    /// shell response is taken from degenerate perturbation theory and the
    /// gradient requested for a shell member is returned as the
    /// basis-independent **shell average** — which is exactly what a central
    /// finite difference of that orbital's QP energy converges to (the finite
    /// difference of the sorted level follows the other branch for `-h`).
    /// `0.0` disables the treatment and restores the legacy behaviour (raw
    /// `0/0` rotation, single-member gradient, orthogonality panic).
    pub degenerate_tol: f64,
    /// Validation bitmask for the full-CD pullback (`GwCdGradEngine`).
    ///
    /// With `S = Sigma_sub` written as
    /// `sum_m [ sum_wi a_wi (W_wi - W0) + (1/2 - f_m) W0 + sum_r s_r (W_res - W0) ]`,
    /// bit 0 keeps the imaginary-axis term, bit 1 the static `W(0)` term and
    /// bit 2 the residue term; `0b111` (the default) is the complete
    /// production gradient.  Because every `W` carries the same
    /// `-q^T J^-1 q` piece, a masked pullback is the exact gradient of the
    /// masked functional (the shared piece is scaled by `(1/2 - f_m)` only
    /// when bit 1 is set), so each term can be validated against the finite
    /// difference of its own contribution.
    pub phase_mask: u32,
    /// Store the QP-cache interaction data in its **compact** form: only the
    /// metric intermediate `m = s_metric * q_n` per quadrature point, with `y`
    /// and `y0` rebuilt on demand during the pullback.  Halves the resident
    /// size of a cache at the cost of one gemm per `(cache, quadrature)` in the
    /// pullback.  Used by the BSE gradient.
    pub compact_qp_cache: bool,
    /// Skip the low-rank machinery of the LR engine (`qt`, `qtia`, the
    /// real-axis grid and the low-rank factors).  Only the full-CD engine
    /// ([`GwCdGradEngine`]) sets this; the LR-only methods then find empty
    /// caches and must not be called.
    pub skip_lr_cache: bool,
    /// Evaluate the `qx` channel through the **folded** (AO-basis) covector
    /// instead of materialising `qx = C^T dI C + U^T q + q U` per perturbation.
    ///
    /// With the perturbation-independent folds
    /// `B[P] = C bq_P C^T` and `W = sum_P q_P bq_P` (one per weight row, built
    /// once per geometry), the channel is
    /// `sum bq:qx = sum_P <B[P], dI[P]> + 2 Tr[U^T W]`, i.e. an `O(naux nao^2)`
    /// dot plus an `O(nmo^2)` dot per perturbation, instead of the two MO
    /// transformations of `dI` (`O(naux nao^3)` per perturbation).  This
    /// removes the `O(N^5)` scaling of the `qx` channel
    /// (`O(naux nao^3)` once + `O(3 natm * naux nao^2)` instead of
    /// `O(3 natm * naux nao^3)`) without changing the memory profile.
    /// Mathematically identical to the `qx_from_u_blocks` route.
    pub folded_qx: bool,
    /// Evaluate the CP-HF channel through the adjoint ("Z-vector") solve
    /// ([`GwGradEngine::build_cphf_adjoint`]): one adjoint CP-HF solve plus
    /// `O(naux nao^2)` per-perturbation contractions replace the
    /// `3 natm`-column Krylov solve and the per-perturbation skeleton assembly.
    /// Requires a non-degenerate HF reference; otherwise the gradient falls
    /// back to the per-perturbation route.
    pub adjoint_cphf: bool,
    /// Locality-aware evaluation of the `dI` contraction
    /// ([`GwGradEngine::adjoint_perturbation_gradient_local`]).  `dI` is
    /// nonzero only on the perturbed atom's own AO rows/columns
    /// (`int3c2e_ip1`) and, for the atom's own auxiliary functions, on the
    /// whole `[nao,nao]` slice (`int3c2e_ip2`); contracting the covectors
    /// directly against those sliced blocks avoids materialising a dense
    /// `[nao,nao]` slice for every auxiliary function, lowering the
    /// per-perturbation cost from `O(naux nao^2)` to
    /// `O(naux nao nrow_A + naux_A nao^2)` and the whole gradient from
    /// `O(3 natm naux nao^2)` to `O(naux nao^2)`.
    pub local_di: bool,
    /// Keep the individual adjoint skeleton tensors of the CP-HF channel in
    /// [`CphfAdjoint`] (`Theta` alongside its folded combination `di_cov`, and the
    /// MO-basis covectors), so that the piecewise identities can be checked term by term.
    ///
    /// Off by default: `Theta` duplicates an `naux * nao^2` tensor (874 MB at
    /// naphthalene/def2-TZVP) that the gradient itself never reads after the fold.
    /// This only affects memory, not the numbers.
    pub keep_adjoint_pieces: bool,
    /// Memory budget (MiB) of the RI-JK response-kernel cache used inside the adjoint CP-HF build.
    ///
    /// The cache trades memory for speed on the Z-vector solve. Its two tiers are the unpacked
    /// three-centre ERI (`nao^2 * naux` doubles — 874 MB at naphthalene/def2-TZVP) and the much
    /// smaller occupied half-transforms (`(nao+nocc) * naux * nocc`). The default here is
    /// deliberately small: with a large budget the unpacked-`cderi` tier is resident *while the
    /// folded covector `B_ao` (`naux*nao^2` doubles) is also alive*, which sets the gradient's peak.
    /// 256 MiB keeps the half-transforms (the per-call win) and drops the big tier, costing about
    /// 0.5 s on the Z solve; raise it on a memory-rich machine.
    pub resp_cache_budget_mib: f64,
    /// Cache the bra-independent intermediates of the RI-JK response kernel
    /// (`analdrv`'s `cpscf_cache_kernel`).  The CP-HF route applies that kernel
    /// to a single right-hand side per Krylov iteration, so it rebuilds the
    /// unpacked 3c-ERI and its occupied half-transforms (an `O(nao^2 naux)`
    /// pass) on every iteration unless they are cached.  This only affects
    /// speed and memory, not the numbers.
    pub resp_cache_kernel: bool,
    /// Evaluate the adjoint CP-HF channel without the per-perturbation skeleton
    /// right-hand side tensor: `rhs_d` is linear in the derivative integrals,
    /// so `sum_d <Z, rhs_d>` is folded into direction-independent AO-basis
    /// covectors (`W = Zbra C_occ^T` and its `dh1`/`dS`/`dJ`/`dK` partners) that
    /// contract with the *same* streamed `dI` slices the gradient already
    /// touches.  This removes `compute_dimless_cpscf_rhs` (the RI-JK skeleton
    /// derivative machinery, measured at ~2 s for C2H4/def2-TZVPP and ~7.5 s for
    /// C6H6/def2-TZVPP) from the gradient path.
    pub fold_cphf_rhs: bool,
    /// Solve the QP equation with REST's production **graphical
    /// linear-interpolation solver** (`ri_gw::linear_interpolation_solver`:
    /// one damped Newton step centres a uniform 21-point scan of +-0.1 Ha,
    /// the root is the linear interpolation of the sign change, the crossing
    /// of largest spectral weight wins) instead of the analytic-derivative
    /// Newton iteration.  A scan without a sign change falls back to the
    /// static-coupling estimate (`e_n + Sigma(e_n)`) and reports the root as
    /// unconverged (`qp_equation_value` then has a large residual).
    pub qpe_graphical: bool,
    /// Orbital indices whose quasiparticle energy must be taken from the
    /// **static on-shell fallback** instead of a root of the quasiparticle
    /// equation (see [`QpCache::static_fallback`]).
    ///
    /// The branch that [`Self::build_target_cache`] selects depends on the
    /// nuclear geometry (the graphical scan finds a sign change at some
    /// displacements and not at others), so the QP energy — and therefore the
    /// quantity being differentiated — is a *discontinuous* function of the
    /// geometry.  A central finite difference then evaluates a different
    /// functional on the two sides and cannot reproduce the analytic gradient.
    /// Pinning this set to the one the reference geometry selected puts both
    /// sides on the same branch, which is the only way to validate the
    /// remaining (continuous) derivative machinery.  Empty by default.
    pub force_static_fallback: Vec<usize>,
    /// Validation pin of the **linearized** branch: orbitals listed here always
    /// take [`Self::linearized_cache`], independent of the root search.  Like
    /// [`Self::force_static_fallback`] it exists so that a central finite
    /// difference evaluates one and the same functional at every geometry (a
    /// root whose `Z` sits near `z_min` can otherwise flip branch between the
    /// two displaced geometries).  Empty by default.
    pub force_linearized: Vec<usize>,
    /// Spectral-weight floor of the root branch.  When the converged root
    /// carries `Z = 1/(1 - dSigma_c/domega) < z_min`, the quasiparticle energy
    /// is instead taken from the **linearized (one-step Newton)** functional
    ///
    /// ```text
    /// E_lin = e_n + Z(e_n) * [ consts_n + Re Sigma_c(e_n) - e_n ]
    /// ```
    ///
    /// evaluated at the *mean-field* level `e_n` rather than at the root.  The
    /// same branch is taken whenever the root search itself fails to converge.
    /// `0.0` (or negative) disables the rule and restores the historical
    /// behaviour (root, or the static on-shell fallback on failure).
    pub z_min: f64,
    /// Define the LR-CD self-energy through the **static-subtracted continuous**
    /// form of [`GwCdGradEngine`] (the default), evaluated with REST's low-rank
    /// response factors — see the module header.  Numerically identical to the
    /// raw decomposition wherever no mean-field pole coincides with the QP
    /// energy (to ~1e-9 `W(0)` on the production grids), but continuous and
    /// differentiable there, so the analytic gradient is always defined.
    /// `false` restores the legacy raw LR functional.  Ignored by the full-CD
    /// engine ([`GwCdGradEngine`]), which is static-subtracted by construction.
    pub subtracted: bool,
}

impl Default for GwGradConfig {
    fn default() -> Self {
        GwGradConfig {
            num_freq: 20,
            freq_grid_type: 0,
            freq_cut_off: 50.0,
            eta: 0.001,
            res_tol: 0.001,
            nomega_chi_real: 6,
            low_rank_tolerance: 1.0e-10,
            grid_type: 0,
            nomega_sigma: 10,
            step_sigma: 0.05,
            selfenergy_state_range: 100000,
            omega_chi_max: 0.0,
            qpe_tol: 1.0e-11,
            qpe_max_iter: 100,
            exact_residue_z: false,
            static_fallback_threshold: 1.0,
            low_rank_interp: 1,
            graphical_rescue: false,
            smooth_pole: false,
            degenerate_tol: 1.0e-8,
            phase_mask: 0b111,
            skip_lr_cache: false,
            folded_qx: true,
            adjoint_cphf: true,
            local_di: true,
            keep_adjoint_pieces: false,
            resp_cache_kernel: true,
            resp_cache_budget_mib: 256.0,
            fold_cphf_rhs: true,
            compact_qp_cache: false,
            subtracted: true,
            qpe_graphical: false,
            force_static_fallback: Vec::new(),
            force_linearized: Vec::new(),
            z_min: 0.3,
        }
    }
}

impl GwGradConfig {
    /// Read the protocol options from REST's control block (the same values
    /// the production low-rank GW driver would use).
    pub fn from_scf(scf: &SCF) -> Self {
        let qp = scf
            .mol
            .ctrl
            .quasiparticle_methods
            .clone()
            .expect("gw_grad: quasiparticle_methods ctrl block required");
        GwGradConfig {
            freq_grid_type: scf.mol.ctrl.freq_grid_type,
            freq_cut_off: scf.mol.ctrl.freq_cut_off,
            eta: qp.cdgw_eta,
            res_tol: qp.cdgw_res_tol,
            nomega_chi_real: qp.nomega_chi_real,
            low_rank_tolerance: qp.low_rank_tolerance,
            grid_type: if qp.low_rank_grid_type == "quadratic" { 1 } else { 0 },
            nomega_sigma: qp.nomega_sigma,
            step_sigma: qp.step_sigma,
            selfenergy_state_range: qp.selfenergy_state_range,
            omega_chi_max: qp.omega_chi_max,
            static_fallback_threshold: qp.gw_switch_fallback_threshold,
            low_rank_interp: if qp.low_rank_interp.eq_ignore_ascii_case("nearest") { 0 } else { 1 },
            graphical_rescue: qp.gradient_graphical_rescue,
            smooth_pole: qp.gradient_smooth_pole,
            ..Default::default()
        }
    }
}

/// Memory probe for the gradient phases: with `REST_GWGRAD_MEMDBG` set, print the
/// resident set (`VmRSS`) and the high-water mark (`VmHWM`) of the process, so the
/// peak of a gradient run can be attributed to a phase.  No effect otherwise.
///
/// The same line also carries the wall time elapsed since the previous probe
/// (`dt`) and since the first probe of the process (`T`), so the per-stage cost
/// breakdown comes from the very same instrumentation as the memory one.
fn mem_probe(label: &str) {
    if std::env::var("REST_GWGRAD_MEMDBG").is_err() {
        return;
    }
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;
    static CLOCK: OnceLock<Mutex<(Instant, Instant)>> = OnceLock::new();
    let now = Instant::now();
    let (dt, total) = {
        let clock = CLOCK.get_or_init(|| Mutex::new((now, now)));
        let mut g = clock.lock().unwrap();
        let dt = now.duration_since(g.0).as_secs_f64();
        let total = now.duration_since(g.1).as_secs_f64();
        g.0 = now;
        (dt, total)
    };
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let kb = |key: &str| -> f64 {
        status
            .lines()
            .find_map(|l| l.strip_prefix(key))
            .and_then(|r| r.trim().trim_end_matches(" kB").trim().parse::<f64>().ok())
            .unwrap_or(0.0)
            / 1024.0
    };
    println!(
        "[memdbg] {label:46} dt {dt:8.2}s | T {total:8.2}s | rss {:8.1} MB | peak {:8.1} MB",
        kb("VmRSS:"),
        kb("VmHWM:")
    );
}

// ---------------------------------------------------------------------------
// raw RI tensors and their nuclear derivatives
// ---------------------------------------------------------------------------

/// Geometry-fixed raw RI data: `(mu nu | P)`, `J = (P|Q)` and the machinery
/// needed to assemble any nuclear derivative `dI`, `dJ`.
///
/// The electronic derivative tensors `dI1`/`dI2` are NOT stored (they would
/// cost 2 x 3 nao^2 naux doubles, several GB at naphthalene scale).
/// Instead, [`RawRiTensors::d_atom_blocks`] evaluates the derivative
/// integrals shell-sliced on the perturbed atom's centers on the fly —
/// the total integral work is identical to one full `dI1`/`dI2` build, but
/// only one atom's blocks (O(nao_A nao naux)) are resident at a time.
pub struct RawRiTensors {
    pub nao: usize,
    pub naux: usize,
    pub natm: usize,
    /// I[mu + nu*nao + P*nao*nao], col-major [nao, nao, naux].
    ///
    /// Held behind a [`RefCell`] so that a task which has finished with it can release
    /// `nao^2 * naux` doubles (874 MB at naphthalene/def2-TZVP) with
    /// [`RawRiTensors::evict_i3`]; every reader goes through [`RawRiTensors::with_i3`],
    /// which re-evaluates the three-centre integrals on demand.
    pub i3: RefCell<Vec<f64>>,
    /// J[P + Q*naux]
    pub j2: Vec<f64>,
    /// dJ1[P + Q*naux + t*naux*naux]
    pub dj1: Vec<f64>,
    /// AO index -> atom
    pub ao_atom: Vec<usize>,
    /// auxiliary AO index -> atom
    pub aux_atom: Vec<usize>,
    /// integral engines for the on-the-fly derivative assembly
    pub mol: CInt,
    pub auxmol: CInt,
    /// per-atom [shl0, shl1] of the orbital basis
    pub ao_shl_range: Vec<[usize; 2]>,
    /// per-atom [ao0, ao1] of the orbital basis
    pub ao_range: Vec<[usize; 2]>,
    /// per-atom [shl0, shl1] of the auxiliary basis
    pub aux_shl_range: Vec<[usize; 2]>,
    /// per-atom [aux0, aux1]
    pub aux_range: Vec<[usize; 2]>,
}

/// Shell-sliced derivative integral blocks of ONE perturbed atom, shared by
/// its three Cartesian components (PySCF `gw_cd_grad_optimized.
/// _AtomThreeCenterDerivatives`).
pub struct AtomDerivBlocks {
    /// dI1 slice: [nrow, nao, naux, 3] col-major, derivative on the atom's
    /// own (bra) centers
    pub d1: Vec<f64>,
    /// dI2 slice: [nao, nao, naux_a, 3] col-major, derivative on the atom's
    /// auxiliary centers
    pub d2: Vec<f64>,
    /// atom index
    pub atm: usize,
    /// first AO of the atom
    pub row0: usize,
    /// number of AOs of the atom
    pub nrow: usize,
    /// number of auxiliary functions of the atom
    pub naux_a: usize,
    /// first auxiliary function of the atom
    pub aux0: usize,
}

fn atom_map_from_slices(slices: &[[usize; 4]], nao: usize) -> Vec<usize> {
    let mut map = vec![0usize; nao];
    for (atm, s) in slices.iter().enumerate() {
        for p in s[2] as usize..s[3] as usize {
            map[p] = atm;
        }
    }
    map
}

/// Build the raw RI tensors with the libcint electronic-derivative integrals.
pub fn build_raw_ri_tensors(scf: &SCF) -> RawRiTensors {
    let mol = get_cint_mol(&scf.mol);
    let aux = get_cint_aux(&scf.mol);
    let nao = mol.nao();
    let naux = aux.nao();
    let natm = scf.mol.geom.nfree;

    // The three-centre integrals are evaluated **on demand**: the engine build and the CD-GW
    // gradient consume them in auxiliary-atom blocks (`for_each_i3_atom_block`, so the dense
    // `nao^2 * naux` array — 874 MB at naphthalene/def2-TZVP — is never materialised), and only the
    // per-perturbation route and the TDDFT gradient read them densely (`with_i3`, which evaluates
    // them once and keeps them). Leaving this empty therefore costs those paths one integral
    // evaluation and keeps the block out of every other task's peak.

    let (j2, shapej) = aux.integrate("int2c2e", "s1", None).into();
    assert_eq!(shapej, vec![naux, naux], "int2c2e shape");
    let (dj1, shapedj) = aux.integrate("int2c2e_ip1", "s1", None).into();
    assert_eq!(shapedj, vec![naux, naux, 3], "int2c2e_ip1 shape");

    let ao_slices = scf.mol.aoslice_by_atom();
    let aux_slices = scf.mol.make_auxmol_fake().aoslice_by_atom();
    let range = |s: &[[usize; 4]], k0: usize, k1: usize| -> Vec<[usize; 2]> {
        s.iter().map(|r| [r[k0], r[k1]]).collect()
    };

    RawRiTensors {
        nao,
        naux,
        natm,
        i3: RefCell::new(Vec::new()),
        j2,
        dj1,
        ao_atom: atom_map_from_slices(&ao_slices, nao),
        aux_atom: atom_map_from_slices(&aux_slices, naux),
        mol,
        auxmol: aux,
        ao_shl_range: range(&ao_slices, 0, 1),
        ao_range: range(&ao_slices, 2, 3),
        aux_shl_range: range(&aux_slices, 0, 1),
        aux_range: range(&aux_slices, 2, 3),
    }
}

impl RawRiTensors {
    /// Run `f` on the raw three-centre integrals, re-evaluating them if a previous
    /// [`Self::evict_i3`] released them.
    ///
    /// The closure runs while the buffer is borrowed; nested readers are fine (they only take
    /// shared borrows), but nothing inside may evict.
    pub fn with_i3<R>(&self, f: impl FnOnce(&[f64]) -> R) -> R {
        self.ensure_i3();
        f(&self.i3.borrow())
    }

    /// Rebuild the three-centre integrals when they are not resident.
    ///
    /// Callers that keep a [`std::cell::Ref`] guard for a whole routine (the TDDFT gradient's
    /// Coulomb/`Y` helpers) call this once on entry, then borrow `i3` directly.
    pub fn ensure_i3(&self) {
        if self.i3.borrow().is_empty() {
            let t0 = Instant::now();
            let (i3, shape3) =
                CInt::integrate_cross("int3c2e", [&self.mol, &self.mol, &self.auxmol], "s1", None).into();
            if std::env::var("REST_GWGRAD_MEMDBG").is_ok() {
                println!(
                    "[memdbg]   i3 re-evaluation (int3c2e, {:.1} MB): {:.3}s",
                    (self.nao * self.nao * self.naux) as f64 * 8.0 / 1048576.0,
                    t0.elapsed().as_secs_f64()
                );
            }
            assert_eq!(shape3, vec![self.nao, self.nao, self.naux], "int3c2e s1 shape");
            *self.i3.borrow_mut() = i3;
        }
    }

    /// Evaluate the raw three-centre integrals **auxiliary-atom block by auxiliary-atom block** and
    /// hand each block to `f` as `(aux_offset, naux_block, block)`, with `block` col-major
    /// `[nao, nao, naux_block]` (auxiliary functions ordered as in the full tensor, i.e. starting at
    /// `aux_offset`).
    ///
    /// This is the streaming alternative to [`Self::with_i3`] for callers that only *contract* the
    /// integrals: the full `nao^2 * naux` array (874 MB at naphthalene/def2-TZVP) is never
    /// materialised, at the price of evaluating the same shell triplets in a few more libcint calls
    /// (the per-call shell-pair setup is negligible against the integral work). The integral count
    /// is identical to one full `int3c2e` call.
    pub fn for_each_i3_atom_block<F>(&self, mut f: F)
    where
        F: FnMut(usize, usize, &[f64]),
    {
        let nao = self.nao;
        for atm in 0..self.natm {
            let fun = self.aux_range[atm];
            let nb = fun[1] - fun[0];
            if nb == 0 {
                continue;
            }
            let shls = [[0, self.mol.nbas()], [0, self.mol.nbas()], self.aux_shl_range[atm]];
            let (blk, shape) =
                CInt::integrate_cross("int3c2e", [&self.mol, &self.mol, &self.auxmol], "s1", &shls).into();
            assert_eq!(shape, vec![nao, nao, nb], "auxiliary-atom int3c2e block shape");
            f(fun[0], nb, &blk);
        }
    }

    /// Stream the **AO-centre derivative** blocks (`int3c2e_ip1`) of one perturbed atom over
    /// auxiliary-shell blocks, handing `f(aux_offset, naux_block, block)` with `block` col-major
    /// `[nrow_A, nao, naux_block, 3]`.
    ///
    /// The per-direction contraction only *reads* these blocks, so streaming them keeps
    /// `nrow_A * nao * naux * 3` doubles out of the gradient's peak (254 MB for a carbon atom at
    /// naphthalene/def2-TZVP, against a few MB per block). The integral work is the same
    /// shell-triplet sum as one `int3c2e_ip1` call, split into a few more libcint calls.
    pub fn for_each_d1_block<F>(&self, atm: usize, mut f: F)
    where
        F: FnMut(usize, usize, &[f64]),
    {
        let row = self.ao_range[atm];
        let nrow = row[1] - row[0];
        let nshl_aux = self.auxmol.nbas();
        // a few dozen auxiliary shells per block keeps the block (~60 MB) far below the full tensor
        // (254 MB) while keeping the number of libcint calls small
        let step = 64usize;
        let mut a0 = 0usize;
        let mut fun_off = 0usize;
        while a0 < nshl_aux {
            let a1 = (a0 + step).min(nshl_aux);
            let shls = [self.ao_shl_range[atm], [0, self.mol.nbas()], [a0, a1]];
            let (blk, shape) =
                CInt::integrate_cross("int3c2e_ip1", [&self.mol, &self.mol, &self.auxmol], "s1", &shls).into();
            let nb = shape[2];
            assert_eq!(shape, vec![nrow, self.nao, nb, 3], "streamed int3c2e_ip1 block shape");
            f(fun_off, nb, &blk);
            fun_off += nb;
            a0 = a1;
        }
        debug_assert_eq!(fun_off, self.naux);
    }

    /// Stream the **auxiliary-centre derivative** blocks (`int3c2e_ip2`) of one perturbed atom over
    /// its own auxiliary shells; `block` is col-major `[nao, nao, naux_block, 3]`.
    pub fn for_each_d2_block<F>(&self, atm: usize, mut f: F)
    where
        F: FnMut(usize, usize, &[f64]),
    {
        let (shl0, shl1) = (self.aux_shl_range[atm][0], self.aux_shl_range[atm][1]);
        let fun0 = self.aux_range[atm][0];
        // the atom's own auxiliary shells are only a handful, so small blocks are cheap here and
        // cut the `nao^2 * naux_A * 3` tensor (135 MB for a carbon atom) to a few tens of MB
        let step = 4usize;
        let mut a0 = shl0;
        let mut fun_off = 0usize;
        while a0 < shl1 {
            let a1 = (a0 + step).min(shl1);
            let shls = [[0, self.mol.nbas()], [0, self.mol.nbas()], [a0, a1]];
            let (blk, shape) =
                CInt::integrate_cross("int3c2e_ip2", [&self.mol, &self.mol, &self.auxmol], "s1", &shls).into();
            let nb = shape[2];
            assert_eq!(shape, vec![self.nao, self.nao, nb, 3], "streamed int3c2e_ip2 block shape");
            f(fun0 + fun_off, nb, &blk);
            fun_off += nb;
            a0 = a1;
        }
    }

    /// Release the raw three-centre integrals (`nao^2 * naux` doubles).
    ///
    /// They are re-evaluated on demand by [`Self::with_i3`], so this only trades memory for the
    /// integral work of whoever needs them next. Tasks that hold their own copies of what they
    /// need (the CD-GW adjoint gradient after its Coulomb/`Y` contractions) can drop them for
    /// the rest of the calculation.
    pub fn evict_i3(&self) {
        self.i3.replace(Vec::new());
    }

    /// Shell-sliced derivative integral blocks of one perturbed atom,
    /// shared by its three Cartesian components.  The total integral work
    /// summed over atoms equals one full `dI1`/`dI2` build, but only one
    /// atom's blocks are resident at a time.
    /// Atom-sliced `int3c2e_ip1` block only (AO-derivative tables).  The caller
    /// usually needs either the `ip1` or the `ip2` block; evaluating both every
    /// time doubles the integral cost for no reason.
    pub fn d1_blocks(&self, atm: usize) -> AtomDerivBlocks {
        let row = self.ao_range[atm];
        let nrow = row[1] - row[0];
        let aux_r = self.aux_range[atm];
        let shls1 = [self.ao_shl_range[atm], [0, self.mol.nbas()], [0, self.auxmol.nbas()]];
        let (d1, shape1) = CInt::integrate_cross(
            "int3c2e_ip1",
            [&self.mol, &self.mol, &self.auxmol],
            "s1",
            &shls1,
        )
        .into();
        assert_eq!(shape1, vec![nrow, self.nao, self.naux, 3], "sliced int3c2e_ip1 shape");
        AtomDerivBlocks {
            d1,
            d2: Vec::new(),
            atm,
            row0: row[0],
            nrow,
            naux_a: aux_r[1] - aux_r[0],
            aux0: aux_r[0],
        }
    }

    /// Atom-sliced `int3c2e_ip2` block only (auxiliary-basis derivative).
    pub fn d2_blocks(&self, atm: usize) -> AtomDerivBlocks {
        let row = self.ao_range[atm];
        let aux_r = self.aux_range[atm];
        let naux_a = aux_r[1] - aux_r[0];
        let shls2 = [[0, self.mol.nbas()], [0, self.mol.nbas()], self.aux_shl_range[atm]];
        let (d2, shape2) = CInt::integrate_cross(
            "int3c2e_ip2",
            [&self.mol, &self.mol, &self.auxmol],
            "s1",
            &shls2,
        )
        .into();
        assert_eq!(shape2, vec![self.nao, self.nao, naux_a, 3], "sliced int3c2e_ip2 shape");
        AtomDerivBlocks {
            d1: Vec::new(),
            d2,
            atm,
            row0: row[0],
            nrow: row[1] - row[0],
            naux_a,
            aux0: aux_r[0],
        }
    }

    pub fn d_atom_blocks(&self, atm: usize) -> AtomDerivBlocks {
        let row = self.ao_range[atm];
        let nrow = row[1] - row[0];
        let aux_r = self.aux_range[atm];
        let naux_a = aux_r[1] - aux_r[0];
        let shls1 = [self.ao_shl_range[atm], [0, self.mol.nbas()], [0, self.auxmol.nbas()]];
        let (d1, shape1) = CInt::integrate_cross(
            "int3c2e_ip1",
            [&self.mol, &self.mol, &self.auxmol],
            "s1",
            &shls1,
        )
        .into();
        assert_eq!(
            shape1,
            vec![nrow, self.nao, self.naux, 3],
            "sliced int3c2e_ip1 shape"
        );
        let shls2 = [[0, self.mol.nbas()], [0, self.mol.nbas()], self.aux_shl_range[atm]];
        let (d2, shape2) = CInt::integrate_cross(
            "int3c2e_ip2",
            [&self.mol, &self.mol, &self.auxmol],
            "s1",
            &shls2,
        )
        .into();
        assert_eq!(
            shape2,
            vec![self.nao, self.nao, naux_a, 3],
            "sliced int3c2e_ip2 shape"
        );
        AtomDerivBlocks { d1, d2, atm, row0: row[0], nrow, naux_a, aux0: aux_r[0] }
    }

    /// Total nuclear derivative `dI_{mu nu P}/dR_{atm,comp}` (PySCF
    /// `gw_cd_grad._dI_atom`) assembled from the atom's sliced blocks: the
    /// Gaussian nuclear derivative is minus the electronic derivative on
    /// the corresponding center.
    pub fn d_i_from_blocks(&self, blocks: &AtomDerivBlocks, comp: usize) -> Vec<f64> {
        let nao = self.nao;
        let naux = self.naux;
        let nao2 = nao * nao;
        let nrow = blocks.nrow;
        let stride1 = nrow * nao * naux;
        let d1c = &blocks.d1[comp * stride1..(comp + 1) * stride1];
        let stride2 = nao2 * blocks.naux_a;
        let d2c = &blocks.d2[comp * stride2..(comp + 1) * stride2];
        let mut di = vec![0.0f64; nao2 * naux];
        for mu in blocks.row0..blocks.row0 + nrow {
            for nu in 0..nao {
                for p in 0..naux {
                    di[mu + nu * nao + p * nao2] -=
                        d1c[(mu - blocks.row0) + nu * nrow + p * nrow * nao];
                }
            }
        }
        for nu in blocks.row0..blocks.row0 + nrow {
            for mu in 0..nao {
                for p in 0..naux {
                    di[mu + nu * nao + p * nao2] -=
                        d1c[(nu - blocks.row0) + mu * nrow + p * nrow * nao];
                }
            }
        }
        for p_l in 0..blocks.naux_a {
            let p = blocks.aux0 + p_l;
            for k in 0..nao2 {
                di[p * nao2 + k] -= d2c[p_l * nao2 + k];
            }
        }
        di
    }

    /// One-shot derivative of one perturbation (blocks are rebuilt; prefer
    /// [`RawRiTensors::d_atom_blocks`] + [`RawRiTensors::d_i_from_blocks`]
    /// inside the per-atom loop).
    pub fn d_i_atom(&self, atm: usize, comp: usize) -> Vec<f64> {
        let blocks = self.d_atom_blocks(atm);
        self.d_i_from_blocks(&blocks, comp)
    }

    /// Assemble the `[nao, nao]` slice of `dI` for ONE auxiliary function
    /// `p` into `out` (col-major, overwritten).  Streaming variant of
    /// [`RawRiTensors::d_i_from_blocks`]: lets the qx U-part-gemm consume
    /// `dI` block by block so the full `nao^2 naux` derivative never
    /// coexists with the full `qx`.
    pub fn d_i_block(
        &self,
        blocks: &AtomDerivBlocks,
        comp: usize,
        p: usize,
        out: &mut [f64],
    ) {
        let nao = self.nao;
        let nao2 = nao * nao;
        let nrow = blocks.nrow;
        let stride1 = nrow * nao * self.naux;
        let d1c = &blocks.d1[comp * stride1..(comp + 1) * stride1];
        let stride2 = nao2 * blocks.naux_a;
        let d2c = &blocks.d2[comp * stride2..(comp + 1) * stride2];
        out.fill(0.0);
        for mu in blocks.row0..blocks.row0 + nrow {
            for nu in 0..nao {
                out[mu + nu * nao] -= d1c[(mu - blocks.row0) + nu * nrow + p * nrow * nao];
            }
        }
        for nu in blocks.row0..blocks.row0 + nrow {
            for mu in 0..nao {
                out[mu + nu * nao] -= d1c[(nu - blocks.row0) + mu * nrow + p * nrow * nao];
            }
        }
        if p >= blocks.aux0 && p < blocks.aux0 + blocks.naux_a {
            let base = (p - blocks.aux0) * nao2;
            for k in 0..nao2 {
                out[k] -= d2c[base + k];
            }
        }
    }

    /// Total nuclear derivative `dJ_{PQ}/dR_{atm,comp}`.
    pub fn d_j_atom(&self, atm: usize, comp: usize) -> Vec<f64> {
        let naux = self.naux;
        let naux2 = naux * naux;
        let mut dj = vec![0.0f64; naux2];
        let djc = &self.dj1[comp * naux2..(comp + 1) * naux2];
        for p in 0..naux {
            if self.aux_atom[p] != atm {
                continue;
            }
            for q in 0..naux {
                dj[p + q * naux] -= djc[p + q * naux];
                dj[q + p * naux] -= djc[p + q * naux];
            }
        }
        dj
    }
}

// ---------------------------------------------------------------------------
// small dense helpers (flat col-major buffers, first index fastest)
// ---------------------------------------------------------------------------

/// Plain inner product of two flat buffers.
#[inline]
pub fn flat_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn solve_linear(a: &MatrixFull<f64>, b: &[f64]) -> Vec<f64> {
    _dsolve(a, b).expect("gw_grad: singular linear system")
}

fn to_mat(v: &[f64], r: usize, c: usize) -> MatrixFull<f64> {
    MatrixFull::from_vec([r, c], v.to_vec()).unwrap()
}

/// `A^T B` for flat col-major buffers `A [ar, ac]`, `B [br, bc]`, `ar == br`.
/// The contraction runs over the FIRST dimension (REST `_dgemm_full` 'T'
/// convention), the result is `[ac, bc]` col-major.
///
/// The operands are borrowed as **views** (`rt::asarray` on a slice) instead of being wrapped in
/// `MatrixFull` through `to_mat`, which copies its whole buffer: in the GW gradient `A` is the raw
/// three-centre integral tensor `[nao^2, naux]` (three calls per adjoint build) and `B` is the
/// metric-transformed `qt` in the low-rank build, so a copy costs `nao^2 * naux` doubles
/// (874 MB at naphthalene/def2-TZVP) of purely transient memory each time.
fn gemm_nt(a: &[f64], ar: usize, ac: usize, b: &[f64], br: usize, bc: usize) -> Vec<f64> {
    use rstsr::prelude::*;

    debug_assert_eq!(ar, br, "gemm_nt: contraction dimension mismatch");
    let device = DeviceBLAS::default();
    let am = rt::asarray((a, [ar, ac].f(), &device));
    let bm = rt::asarray((b, [br, bc].f(), &device));
    let out = am.t() % bm;
    out.into_contig(ColMajor).into_shape(-1).into_vec()
}

/// Explicit transpose of a flat col-major `[r, c]` buffer.
fn transpose_buf(a: &[f64], r: usize, c: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j + i * c] = a[i + j * r];
        }
    }
    out
}

/// `A B` for flat col-major buffers `A [ar, ac]`, `B [br, bc]`, `ac == br`; result `[ar, bc]`.
///
/// Borrowed views for the same reason as [`gemm_nt`]: the operands include `q`/`qt`/`i3`
/// (`nao^2 * naux` doubles) and the copies would dominate the engine-build and gradient peaks.
fn gemm_nn(a: &[f64], ar: usize, ac: usize, b: &[f64], br: usize, bc: usize) -> Vec<f64> {
    use rstsr::prelude::*;

    debug_assert_eq!(ac, br);
    let device = DeviceBLAS::default();
    let am = rt::asarray((a, [ar, ac].f(), &device));
    let bm = rt::asarray((b, [br, bc].f(), &device));
    let out = am % bm;
    out.into_contig(ColMajor).into_shape(-1).into_vec()
}

/// `s_metric · q` for a `[naux, ncol]` right-hand side, applied in auxiliary-column blocks.
///
/// `gemm_nn` (and the tensor matmul behind it) materialises the result as a *second* full-size
/// buffer while the first is still alive, and converts its layout with a third: for the engine's
/// `qt = s_metric · q` (`naux * nmo^2` = 874 MB at naphthalene/def2-TZVP) that transient was the
/// highest point of the whole gradient. Working block-wise holds one `naux * block` temporary
/// (3.7 MB) and writes straight into the output.
fn metric_transform_blocked(s_metric: &[f64], q: &[f64], naux: usize, ncol: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; naux * ncol];
    let sm = to_mat(s_metric, naux, naux);
    let block = 512usize.min(ncol).max(1);
    let mut c0 = 0;
    while c0 < ncol {
        let c1 = (c0 + block).min(ncol);
        let nb = c1 - c0;
        let bm = to_mat(&q[c0 * naux..c1 * naux], naux, nb);
        let mut om = MatrixFull::new([naux, nb], 0.0);
        _dgemm_full(&sm, 'N', &bm, 'N', &mut om, 1.0, 0.0);
        out[c0 * naux..c1 * naux].copy_from_slice(&om.data);
        c0 = c1;
    }
    out
}

/// Symmetric eigendecomposition, ascending eigenvalues, column eigenvectors.
///
/// NOTE: the LAPACK bindings of this build leave `info = n` even on trivial
/// input while returning correct results (REST's production code therefore
/// ignores `info` as well); we validate the eigenvalues instead.
fn eigh_symmetric(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mat = to_mat(a, n, n);
    let (vecs, vals, _info) = _dsyev_inplace(mat, 'V');
    let vals = vals;
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "gw_grad: eigensolver produced non-finite eigenvalues"
    );
    (vals, vecs.expect("gw_grad: dsyev vectors").data)
}

/// Symmetry-protected degenerate shells of the orbital energies `e`.
///
/// The SCF sorts its eigenvalues, so a shell is a maximal *contiguous* run of
/// energies agreeing within `tol`; `tol <= 0` puts every orbital in a shell of
/// its own (legacy behaviour).  Returns `(shells, shell_of)` where `shells` is
/// the list of index groups (each of size >= 1) and `shell_of[p]` is the
/// position of orbital `p` in `shells`.
pub fn degenerate_shells(e: &[f64], tol: f64) -> (Vec<Vec<usize>>, Vec<usize>) {
    let mut shells: Vec<Vec<usize>> = Vec::new();
    let mut shell_of = vec![0usize; e.len()];
    for p in 0..e.len() {
        let start_new = match shells.last() {
            Some(s) => tol <= 0.0 || (e[p] - e[*s.last().unwrap()]).abs() > tol,
            None => true,
        };
        if start_new {
            shells.push(vec![p]);
        } else {
            shells.last_mut().unwrap().push(p);
        }
        shell_of[p] = shells.len() - 1;
    }
    (shells, shell_of)
}

// ---------------------------------------------------------------------------
// low-rank screened response (REST real-symmetric convention)
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

#[allow(dead_code)]
fn build_lr_factors(
    qtia: &[f64],
    naux: usize,
    nov: usize,
    de_ia: &[f64],
    quad: &[(f64, f64)],
    z_grid: &[f64],
    eta: f64,
    tol: f64,
) -> (Vec<LrFactor>, Vec<LrFactor>) {
    let lr_imag = quad
        .iter()
        .map(|&(u, _)| build_lr_factor(qtia, naux, nov, de_ia, u, 'I', 0.0, tol))
        .collect();
    let lr_grid = z_grid
        .iter()
        .map(|&z| build_lr_factor(qtia, naux, nov, de_ia, z, 'R', eta, tol))
        .collect();
    (lr_imag, lr_grid)
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
}

/// The LR-CD G0W0 analytic-gradient engine for one geometry.
///
/// Pair layout used throughout: `q[Q + (p + q_mo*nmo)*naux]` for the raw
/// integrals and `qt` (metric-transformed), and
/// `qia[Q + (i + a*nocc)*naux]` for the occupied-virtual block.
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
    pub lr_grid: RefCell<Vec<Option<LrFactor>>>,
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
    }

    impl QpPullbackScratch {
        fn new(engine: &GwGradEngine, nk: usize) -> Self {
            let (naux, nmo, nocc) = (engine.naux, engine.nmo, engine.nocc);
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
            QpPullbackScratch {
                nk,
                naux,
                nmo,
                nocc,
                nov,
                // the pair covector accumulates directly in its packed
                // (lower-triangle) layout: the pullback writes the two
                // directions of a pair into the same slot and the finish pass
                // halves the off-diagonal pairs, which is exactly the
                // minimum-norm symmetric representative the symmetrisation used
                // to build — at half the memory
                bq: vec![vec![0.0f64; naux * packed_pair_count(nmo)]; nk],
                bj: vec![vec![0.0f64; naux * naux]; nk],
                be: vec![vec![0.0f64; nmo]; nk],
                bb: vec![vec![0.0f64; nmo]; nk],
                v_imag: vec![vec![vec![0.0f64; naux * naux]; nquad]; nk],
                v_static: vec![vec![0.0f64; naux * naux]; nk],
                v_res: vec![Vec::new(); nk],
                z_order: vec![Vec::new(); nk],
                imag_factors,
            }
        }
    }

/// Frequency-independent data of one orbital row's self-energy.
///
/// `m = s_metric * q_n` and the imaginary-axis table
/// `w_imag[wi][m] = W_c[n, m](iu_wi)` do **not** depend on the trial QP energy,
/// so the Newton iterations of one orbital share them.  Building the table is
/// the dominant cost of a self-energy evaluation (`nquad` projections against
/// the low-rank factors) and used to be repeated for every trial `omega` — a
/// factor equal to the iteration count.
pub struct QpRowTable {
    /// `m = s_metric * q_n`, `[naux, nmo]` col-major
    pub m: Vec<f64>,
    /// `W_c[n, m](iu_wi)` per quadrature point
    pub w_imag: Vec<Vec<f64>>,
    /// Static row `W_c[n, m](0)` (empty when the engine runs raw/full-CD).
    /// Shared by every trial frequency of the Newton loop, so it is computed
    /// once with the table instead of once per evaluation.
    pub w0: Vec<f64>,
}

impl<'a> GwGradEngine<'a> {
    pub fn new(scf: &'a SCF, config: GwGradConfig) -> Self {
        let raw = build_raw_ri_tensors(scf);
        Self::new_with_raw(scf, config, raw)
    }

    /// Engine with externally supplied raw RI tensors (allows perturbing the
    /// integrals directly, e.g. for validation).
    pub fn new_with_raw(scf: &'a SCF, mut config: GwGradConfig, raw: RawRiTensors) -> Self {
        assert_eq!(scf.mol.spin_channel, 1, "gw_grad: restricted closed-shell only");
        assert_eq!(scf.mol.start_mo, 0, "gw_grad: no frozen orbitals supported");
        if config.num_freq == 0 {
            config.num_freq = 20;
        }
        let nao = raw.nao;
        let naux = raw.naux;
        let nmo = scf.mol.num_state;
        let nocc = scf.homo[0] + 1;
        let natm = raw.natm;
        let e = scf.eigenvalues[0][..nmo].to_vec();
        let c_mo = scf.eigenvectors[0].clone();

        // ---- raw q[P, p, q] = sum_mu nu I[mu nu P] C[mu p] C[nu q] ----
        // Two dgemms per auxiliary function (`dI_P C` then `C^T (dI_P C)`)
        // instead of the naive O(nao^2 nmo) triple loop: this is the dominant
        // setup cost for large basis sets (BLAS runs it at cache speed).
        let mut q = vec![0.0f64; naux * nmo * nmo];
        {
            // The three-centre integrals are consumed **auxiliary-atom block by block**: the dense
            // `nao^2 * naux` tensor (874 MB at naphthalene/def2-TZVP) is never materialised, which
            // removes it from the engine-build peak. The integral work is the same shell-triplet
            // sum as one `int3c2e` call.
            let n2 = nao * nao;
            let mut ip = MatrixFull::new([nao, nao], 0.0);
            let mut tmp = MatrixFull::new([nao, nmo], 0.0);
            let mut sl = MatrixFull::new([nmo, nmo], 0.0);
            raw.for_each_i3_atom_block(|p0, nb, blk| {
                for pl in 0..nb {
                    let p_aux = p0 + pl;
                    ip.data.copy_from_slice(&blk[pl * n2..(pl + 1) * n2]);
                    // tmp[mu, q_mo] = sum_nu I[mu, nu, P] C[nu, q_mo]
                    _dgemm_full(&ip, 'N', &c_mo, 'N', &mut tmp, 1.0, 0.0);
                    // sl[p_mo, q_mo] = sum_mu C[mu, p_mo] tmp[mu, q_mo]
                    _dgemm_full(&c_mo, 'T', &tmp, 'N', &mut sl, 1.0, 0.0);
                    for col in 0..nmo * nmo {
                        q[col * naux + p_aux] = sl.data[col];
                    }
                }
            });
        }
        symmetrise_pairs(&mut q, naux, nmo);
        // pair-slowest repack (strided once, then contiguous for the gemms).
        // Only the legacy `qx_from_u_blocks` route reads it (`U^T q + q U` as one
        // gemm); that route repacks per auxiliary block on the fly inside
        // `qx_from_u` when this is empty. The folded route (`folded_qx`) never
        // touches `q_pairmajor`, so it is not built there — that keeps
        // `naux * nmo^2` doubles (874 MB at naphthalene/def2-TZVP) out of the
        // resident set of every optimised gradient. Same for the full-CD
        // engine (`skip_lr_cache`).
        let q_pairmajor: Vec<f64> = if config.skip_lr_cache || config.folded_qx {
            Vec::new()
        } else {
            let npair = nmo * nmo;
            let mut v = vec![0.0f64; naux * npair];
            for pair in 0..npair {
                for p in 0..naux {
                    v[pair + p * npair] = q[pair * naux + p];
                }
            }
            v
        };

        mem_probe("  build: after q");
        // ---- metric and J^{-1/2} ----
        let j_mat = MatrixFull::from_vec([naux, naux], raw.j2.clone()).unwrap();
        let mut j_chol = j_mat.clone();
        _dpotrf(&mut j_chol, 'L');
        let (vals, vecs) = eigh_symmetric(&j_mat.data, naux);
        // `S = J^{-1/2} = V diag(1/sqrt(lam)) V^T` as two BLAS-3 products: the
        // explicit `O(naux^3)` scalar triple loop this replaces (with its
        // branch on the eigenvalue cutoff) cost ≈ 1.2 s at naphthalene/def2-SVP.
        let mut vd = vec![0.0f64; naux * naux];
        for v in 0..naux {
            let scale = if vals[v] > 1.0e-10 { 1.0 / vals[v].sqrt() } else { 0.0 };
            for a in 0..naux {
                vd[a + v * naux] = vecs[a + v * naux] * scale;
            }
        }
        let mut sm = MatrixFull::new([naux, naux], 0.0);
        {
            let vd_m = to_mat(&vd, naux, naux);
            let v_m = to_mat(&vecs, naux, naux);
            _dgemm_full(&vd_m, 'N', &v_m, 'T', &mut sm, 1.0, 0.0);
        }
        // keep `S` exactly symmetric: the self-energy value identities
        // (`sum_P m corr`, with `m = S q_n`) rely on it
        let mut s_metric = sm.data;
        for a in 0..naux {
            for b in (a + 1)..naux {
                let avg = 0.5 * (s_metric[a + b * naux] + s_metric[b + a * naux]);
                s_metric[a + b * naux] = avg;
                s_metric[b + a * naux] = avg;
            }
        }

        // ---- metric-transformed integrals (LR-only), occupied MO columns only ----
        // `consts(n)` reads `qt[P, n, i]` for occupied `i` only, i.e. the pair columns
        // `pair_col(n, i) = n + i*nmo < nocc*nmo` — the first `nocc*nmo` columns of `q`; and `qtia`
        // is the (`n >= nocc`, `i < nocc`) block, which (because `q` is pair-symmetrised, so
        // `q[(i,a)] == q[(a,i)]` exactly) is the (`n = a`, `i < nocc`) block of that same prefix.
        // Keeping only the prefix stores `naux*nmo*nocc` doubles (87 MB at naphthalene/def2-TZVP)
        // instead of `naux*nmo^2` (874 MB), which is what used to dominate the engine build.
        let ncol_occ = nocc * nmo;
        let qt = if config.skip_lr_cache {
            Vec::new()
        } else {
            metric_transform_blocked(&s_metric, &q[..ncol_occ * naux], naux, ncol_occ)
        };

        mem_probe("  build: after qt");
        // ---- occupied-virtual blocks ----
        let nov = nocc * (nmo - nocc);
        let mut qia = vec![0.0f64; naux * nov];
        let mut qtia = vec![0.0f64; if config.skip_lr_cache { 0 } else { naux * nov }];
        for a in nocc..nmo {
            for i in 0..nocc {
                let src = (i + a * nmo) * naux;
                let dst = (i + (a - nocc) * nocc) * naux;
                qia[dst..dst + naux].copy_from_slice(&q[src..src + naux]);
                if !config.skip_lr_cache {
                    // occupied-column prefix index of the same number (`q` is pair-symmetric)
                    let src_occ = (a + i * nmo) * naux;
                    qtia[dst..dst + naux].copy_from_slice(&qt[src_occ..src_occ + naux]);
                }
            }
        }
        let mut de_ia: Vec<f64> = Vec::with_capacity(nov);
        for a in nocc..nmo {
            for i in 0..nocc {
                de_ia.push(e[a] - e[i]);
            }
        }
        assert_eq!(de_ia.len(), nov);

        mem_probe("  build: after qia/qtia");
        // ---- imaginary-axis quadrature (REST grids) ----
        let quad: Vec<(f64, f64)> = match config.freq_grid_type {
            1 => {
                let (om, w) =
                    gauss_legendre_grids([0.0, config.freq_cut_off], config.num_freq);
                om.into_iter().zip(w).collect()
            }
            2 => {
                let (om, w) = logarithmic_grid([0.0, config.freq_cut_off], config.num_freq);
                om.into_iter().zip(w).collect()
            }
            _ => {
                let (om, w) = trans_gauss_legendre_grids(1.0, config.num_freq);
                om.into_iter().zip(w).collect()
            }
        };

        // ---- REST real-axis grid (de_max scan + linear/quadratic nodes;
        //      LR-only) ----
        let (z_grid, z_grid_sens, z_grid_mover) = if config.skip_lr_cache {
            (Vec::new(), Vec::new(), None)
        } else {
            let nsemin = (nocc - 1).saturating_sub(config.selfenergy_state_range);
            let nsemax = (nocc + config.selfenergy_state_range).min(nmo - 1);
            let g = build_real_axis_grid(
                &e,
                nocc,
                config.nomega_chi_real,
                nsemin,
                nsemax,
                config.nomega_sigma,
                config.step_sigma,
                config.res_tol,
                config.grid_type,
                config.omega_chi_max,
                config.degenerate_tol,
            );
            // `REST_GWGRAD_ZGRID=1` dumps the real-axis layout: the node count,
            // the range and the spacing decide how well the quantised residue
            // protocol resolves a pole, and the auto rule rarely does what the
            // `nomega_chi_real` value suggests.
            if std::env::var("REST_GWGRAD_ZGRID").is_ok() {
                eprintln!(
                    "[zgrid] n={} range={:.6}..{:.6} d={:.6}",
                    g.nodes.len(),
                    g.nodes.first().copied().unwrap_or(0.0),
                    g.nodes.last().copied().unwrap_or(0.0),
                    g.nodes.windows(2).next().map(|w| w[1] - w[0]).unwrap_or(0.0)
                );
            }
            (g.nodes, g.node_sens, g.mover)
        };

        // ---- reference static-term data and low-rank factors (LR-only) ----
        let hybrid = scf.mol.xc_data.dfa_hybrid_scf;
        // a plain HF reference counts as a hybrid with alpha = 1: its orbital
        // energies already carry the exact exchange (see `ri_gw::exchange_fraction`)
        let fock_hyb = crate::ri_gw::exchange_fraction(scf);
        let vxc_nn = crate::ri_gw::vxc_ao2mo(scf);
        let vxc_mo: Vec<f64> = if scf.mol.xc_data.dfa_compnt_scf.is_empty() {
            Vec::new()
        } else {
            let vxc_ao = scf
                .generate_vxc_rayon(1.0)
                .2[0]
                .to_matrixfull()
                .expect("gw_grad: v_xc AO matrix")
                .data;
            let t = gemm_nt(&c_mo.data, nao, nmo, &vxc_ao, nao, nao);
            gemm_nn(&t, nmo, nao, &c_mo.data, nao, nmo)
        };
        let (shells, shell_of) = degenerate_shells(&e, config.degenerate_tol);
        if config.degenerate_tol > 0.0 && std::env::var("REST_GWGRAD_DBG").is_ok() {
            let deg: Vec<&Vec<usize>> = shells.iter().filter(|s| s.len() > 1).collect();
            println!(
                "[new DBG] degenerate shells (tol {:.1e}): {:?}",
                config.degenerate_tol, deg
            );
        }
        if config.skip_lr_cache {
            GwGradEngine {
                scf,
                config,
                raw,
                nao,
                naux,
                nmo,
                nocc,
                natm,
                e,
                c_mo,
                q: RefCell::new(q),
                q_pairmajor,
                j_mat,
                j_chol,
                s_metric,
                qt: RefCell::new(qt),
                qia,
                de_ia,
                quad,
                z_grid,
                z_grid_sens,
                z_grid_mover,
                lr_imag: RefCell::new(Vec::new()),
                lr_grid: RefCell::new(Vec::new()),
                lr_static: RefCell::new(None),
                qtia: RefCell::new(qtia),
                hybrid,
                fock_hyb,
                vxc_nn,
                vxc_mo,
                shells,
                shell_of,
            }
        } else {
            if std::env::var("REST_GWGRAD_DBG").is_ok() {
                println!("[new DBG] qtia max = {}", qtia.iter().fold(0.0f64, |a, &x| a.max(x.abs())));
            }
            // The imaginary-axis factors are needed by every self-energy
            // evaluation; the real-axis grid factors are built on demand
            // (`ensure_lr_grid_factor`), because a self-energy row reads at
            // most a handful of grid points while a full grid costs one
            // `naux x naux` eigendecomposition per point.
            let lr_imag: Vec<LrFactor> = quad
                .iter()
                .map(|&(u, _)| {
                    build_lr_factor(
                        &qtia, naux, nov, &de_ia, u, 'I', 0.0, config.low_rank_tolerance,
                    )
                })
                .collect();
            let lr_grid: Vec<Option<LrFactor>> = (0..z_grid.len()).map(|_| None).collect();
            mem_probe("  build: after all factors");
            if std::env::var("REST_GWGRAD_DBG").is_ok() {
                println!("[new DBG] lr_imag nkeeps = {:?}", lr_imag.iter().map(|f| f.eigval.len()).collect::<Vec<_>>());
            }
            GwGradEngine {
                scf,
                config,
                raw,
                nao,
                naux,
                nmo,
                nocc,
                natm,
                e,
                c_mo,
                q: RefCell::new(q),
                q_pairmajor,
                j_mat,
                j_chol,
                s_metric,
                qt: RefCell::new(qt),
                qia,
                de_ia,
                quad,
                z_grid,
                z_grid_sens,
                z_grid_mover,
                lr_imag: RefCell::new(lr_imag),
                lr_grid: RefCell::new(lr_grid),
                lr_static: RefCell::new(None),
                qtia: RefCell::new(qtia),
                hybrid,
                fock_hyb,
                vxc_nn,
                vxc_mo,
                shells,
                shell_of,
            }
        }
    }

    fn pair_col(&self, p: usize, q: usize) -> usize {
        p + q * self.nmo
    }

    /// True when `p` and `q` belong to the same (size > 1) degenerate shell.
    pub fn same_degenerate_shell(&self, p: usize, q: usize) -> bool {
        p != q && self.shell_of[p] == self.shell_of[q] && self.shells[self.shell_of[p]].len() > 1
    }

    /// The symmetry-protected degenerate shell containing `target`
    /// (`vec![target]` when the orbital is non-degenerate).
    pub fn degenerate_shell(&self, target: usize) -> Vec<usize> {
        self.shells[self.shell_of[target]].clone()
    }

    /// Gather the block of one MO index against all MOs: `[naux, nmo]` with
    /// columns `src[:, n, m]`.
    fn gather_row(&self, src: &[f64], n: usize) -> Vec<f64> {
        let naux = self.naux;
        let mut out = vec![0.0f64; naux * self.nmo];
        for m in 0..self.nmo {
            let base = self.pair_col(n, m) * naux;
            out[m * naux..(m + 1) * naux].copy_from_slice(&src[base..base + naux]);
        }
        out
    }

    /// Gather one raw pair column `src[:, n, m]`.
    fn gather_pair(&self, src: &[f64], n: usize, m: usize) -> Vec<f64> {
        let base = self.pair_col(n, m) * self.naux;
        src[base..base + self.naux].to_vec()
    }

    /// `y0 = J^{-1} rhs` for n_col right-hand sides of length naux.
    fn metric_solve(&self, rhs: &[f64], n_col: usize) -> Vec<f64> {
        let qt = gemm_nn(&self.s_metric, self.naux, self.naux, rhs, self.naux, n_col);
        gemm_nn(&self.s_metric, self.naux, self.naux, &qt, self.naux, n_col)
    }

    /// `J^{-1} B` through the pre-computed Cholesky factor, with
    /// `B [naux, ncol]` col-major, overwritten with the solution
    /// (multi-right-hand-side triangular solves, one factorisation per
    /// geometry).
    pub(crate) fn solve_metric_batch(&self, b: &mut [f64], ncol: usize) {
        let naux = self.naux;
        let mut bm = to_mat(b, naux, ncol);
        let ok1 = _dtrtrs(&self.j_chol, &mut bm, 'L', 'N', 'N');
        let ok2 = _dtrtrs(&self.j_chol, &mut bm, 'L', 'T', 'N');
        assert!(ok1 && ok2, "gw_grad: metric triangular solve failed");
        b.copy_from_slice(&bm.data);
    }

    /// Public single-rhs metric solve (verification helper).
    pub fn solve_metric_public(&self, rhs: &mut [f64]) {
        self.solve_metric_batch(rhs, 1);
    }

    /// `J^{-1} rhs` for a single right-hand side.
    pub(crate) fn solve_metric(&self, rhs: &[f64]) -> Vec<f64> {
        let mut b = rhs.to_vec();
        self.solve_metric_batch(&mut b, 1);
        b
    }

    /// Ensure the low-rank factors are resident (rebuilds them from `qtia` after `evict_qt`).
    fn ensure_lr(&self) {
        if self.lr_grid.borrow().is_empty() && !self.config.skip_lr_cache {
            self.with_qtia(|qtia| {
                let imag: Vec<LrFactor> = self
                    .quad
                    .iter()
                    .map(|&(u, _)| {
                        build_lr_factor(
                            qtia,
                            self.naux,
                            self.nocc * (self.nmo - self.nocc),
                            &self.de_ia,
                            u,
                            'I',
                            0.0,
                            self.config.low_rank_tolerance,
                        )
                    })
                    .collect();
                *self.lr_imag.borrow_mut() = imag;
                // the real-axis factors are built on demand
                *self.lr_grid.borrow_mut() =
                    (0..self.z_grid.len()).map(|_| None).collect();
            });
        }
    }

    /// Make the real-axis factor of grid point `idx` resident, building it if
    /// this is the first time it is read.
    fn ensure_lr_grid_factor(&self, idx: usize) {
        self.ensure_lr();
        if self.lr_grid.borrow()[idx].is_some() {
            return;
        }
        let z = self.z_grid[idx];
        let factor = self.with_qtia(|qtia| {
            build_lr_factor_at(
                qtia,
                self.naux,
                self.nocc * (self.nmo - self.nocc),
                &self.de_ia,
                z,
                self.config.eta,
                self.config.low_rank_tolerance,
            )
        });
        self.lr_grid.borrow_mut()[idx] = Some(factor);
    }

    /// Make the static (`z = 0`) low-rank factor resident, building it on
    /// first use (see [`Self::lr_static`]).  The `z = 0` Lorentzian diagonal
    /// `-4 de/(de^2 + eta^2)` is exactly what the production real-axis grid
    /// evaluates at its first node, so the subtracted functional stays
    /// consistent with REST's own response protocol.
    fn ensure_lr_static(&self) {
        if self.lr_static.borrow().is_some() || self.config.skip_lr_cache {
            return;
        }
        let factor = self.with_qtia(|qtia| {
            build_lr_factor(
                qtia,
                self.naux,
                self.nocc * (self.nmo - self.nocc),
                &self.de_ia,
                0.0,
                'R',
                self.config.eta,
                self.config.low_rank_tolerance,
            )
        });
        *self.lr_static.borrow_mut() = Some(factor);
    }

    /// Static response row values `W_m(0)` of one orbital row from its metric
    /// intermediate `m = S q_n` (empty when the engine runs raw/full-CD).
    fn lr_static_row_values(&self, m: &[f64]) -> Vec<f64> {
        if !self.config.subtracted || self.config.skip_lr_cache {
            return Vec::new();
        }
        self.ensure_lr_static();
        let guard = self.lr_static.borrow();
        guard
            .as_ref()
            .expect("gw_grad: static low-rank factor missing")
            .response_value_from_qt(m, self.nmo)
    }

    /// Static response vectors `y0s = (J - Q(0))^-1 q_n` of one orbital row
    /// from its metric intermediate `m = S q_n`.
    fn lr_static_row_response(&self, m: &[f64]) -> Vec<f64> {
        self.ensure_lr_static();
        let guard = self.lr_static.borrow();
        guard
            .as_ref()
            .expect("gw_grad: static low-rank factor missing")
            .response_vectors_from_qt(&self.s_metric, m, self.nmo)
    }

    fn lr_factor_nearest_idx(&self, z: f64) -> usize {
        let zs = &self.z_grid;
        let n = zs.len();
        if n <= 1 || z <= zs[0] {
            return 0;
        }
        if z >= zs[n - 1] {
            return n - 1;
        }
        let (mut lo, mut hi) = (0usize, n - 1);
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if zs[mid] < z {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        if (z - zs[lo]) < (zs[hi] - z) {
            lo
        } else {
            hi
        }
    }

    /// Run `f` on the raw MO integrals `q[P, p, q]` (pair-symmetrised, aux-fastest).
    ///
    /// Rebuilds them from `raw.i3` when a previous task released them with [`Self::evict_q`].
    pub fn with_q<R>(&self, f: impl FnOnce(&[f64]) -> R) -> R {
        self.ensure_q();
        f(&self.q.borrow())
    }

    /// Make sure `q` is resident (see [`Self::with_q`]).
    pub fn ensure_q(&self) {
        if self.q.borrow().is_empty() {
            let built = self.build_q();
            *self.q.borrow_mut() = built;
        }
    }

    /// Rebuild `q[P, p, q] = sum_{mu,nu} (mu nu | P) C[mu,p] C[nu,q]` (pair-symmetrised) from the
    /// raw three-centre integrals; mirrors the construction in the engine's `new_with_raw`.
    fn build_q(&self) -> Vec<f64> {
        let (nao, nmo, naux) = (self.nao, self.nmo, self.naux);
        let n2 = nao * nao;
        let mut q = vec![0.0f64; naux * nmo * nmo];
        self.raw.with_i3(|i3| {
            let mut ip = MatrixFull::new([nao, nao], 0.0);
            let mut tmp = MatrixFull::new([nao, nmo], 0.0);
            let mut sl = MatrixFull::new([nmo, nmo], 0.0);
            for p_aux in 0..naux {
                ip.data.copy_from_slice(&i3[p_aux * n2..(p_aux + 1) * n2]);
                _dgemm_full(&ip, 'N', &self.c_mo, 'N', &mut tmp, 1.0, 0.0);
                _dgemm_full(&self.c_mo, 'T', &tmp, 'N', &mut sl, 1.0, 0.0);
                for col in 0..nmo * nmo {
                    q[col * naux + p_aux] = sl.data[col];
                }
            }
        });
        symmetrise_pairs(&mut q, naux, nmo);
        q
    }

    /// Make sure `qt` is resident (see [`Self::with_qt`]).
    fn ensure_qt(&self) {
        if self.qt.borrow().is_empty() && !self.config.skip_lr_cache {
            self.ensure_q();
            // same occupied-column prefix as the constructor (see `new_with_raw`)
            let ncol_occ = self.nocc * self.nmo;
            let built = metric_transform_blocked(
                &self.s_metric,
                &self.q.borrow()[..ncol_occ * self.naux],
                self.naux,
                ncol_occ,
            );
            *self.qt.borrow_mut() = built;
        }
    }

    /// Run `f` on the metric-transformed integrals `qt` (see [`Self::evict_qt`]).
    pub fn with_qt<R>(&self, f: impl FnOnce(&[f64]) -> R) -> R {
        self.ensure_qt();
        f(&self.qt.borrow())
    }

    /// Run `f` on the occupied-virtual block `qtia` (see [`Self::evict_qt`]).
    pub fn with_qtia<R>(&self, f: impl FnOnce(&[f64]) -> R) -> R {
        if self.qtia.borrow().is_empty() && !self.config.skip_lr_cache {
            self.ensure_qt();
            let (naux, nmo, nocc) = (self.naux, self.nmo, self.nocc);
            let qt = self.qt.borrow();
            let mut qtia = vec![0.0f64; naux * nocc * (nmo - nocc)];
            for a in nocc..nmo {
                for i in 0..nocc {
                    // occupied-column prefix index of the same number (see `new_with_raw`)
                    let src = (a + i * nmo) * naux;
                    let dst = (i + (a - nocc) * nocc) * naux;
                    qtia[dst..dst + naux].copy_from_slice(&qt[src..src + naux]);
                }
            }
            *self.qtia.borrow_mut() = qtia;
        }
        f(&self.qtia.borrow())
    }

    /// Release `qt` and `qtia` (`naux * nmo^2` + `naux * nov` doubles): they are only read while
    /// building a QP cache (the static `consts` term and the exact-residue on-the-fly factors),
    /// never by a gradient's own contractions. [`Self::with_qt`]/[`Self::with_qtia`] rebuild them.
    pub fn evict_qt(&self) {
        self.qt.replace(Vec::new());
        self.qtia.replace(Vec::new());
        // the low-rank factors are read only while building a QP cache as well
        self.lr_imag.replace(Vec::new());
        self.lr_grid.replace(Vec::new());
        self.lr_static.replace(None);
    }

    /// Release only the metric-transformed integral blocks (`qt`, `qtia`) and
    /// keep the cached low-rank factors.
    ///
    /// A driver that has finished building its QP caches but still needs the
    /// factors for the gradient pullback (the BSE gradient) can drop
    /// `naux*nmo^2 + naux*nov` doubles here — 76 MB at naphthalene/def2-SVP —
    /// without giving up the factors.  `with_qt`/`with_qtia` rebuild the blocks
    /// from `q` if a later cache build asks for them.
    pub fn evict_qt_blocks(&self) {
        self.qt.replace(Vec::new());
        self.qtia.replace(Vec::new());
    }

    /// Release `q` (`naux * nmo^2` doubles): the folded and adjoint gradient routes read it only
    /// up to the covector fold. [`Self::with_q`] rebuilds it on demand.
    pub fn evict_q(&self) {
        self.q.replace(Vec::new());
    }

    /// REST static term `consts_n = eps_n + (1-alpha) Sigma_x,nn - vxc_nn`
    /// with the RI exchange `Sigma_x,nn = -sum_i (n i | n i)`.
    pub fn consts(&self, n: usize) -> f64 {
        let exchange = self.with_qt(|qt| {
            let mut acc = 0.0;
            for i in 0..self.nocc {
                let base = self.pair_col(n, i) * self.naux;
                acc -= qt[base..base + self.naux].iter().map(|x| x * x).sum::<f64>();
            }
            acc
        });
        // For HF references REST's ctrl sets dfa_hybrid_scf = 0 while the
        // HF eigenvalues already contain the exact exchange, so the static
        // term must use the true hybrid fraction (cf. dft/response.rs).
        self.e[n] + exchange * (1.0 - self.fock_hyb) - self.vxc_nn[n]
    }

    /// Bracket `(lo, hi, t)` of the real-axis grid around `z`, mirroring the
    /// energy path's `ri_gw::interpolate_vchiv_bracket`: the interpolated
    /// operator is `(1-t) M(grid[lo]) + t M(grid[hi])`, and `t = 0` outside the
    /// grid (`lo == hi`).
    fn lr_bracket(&self, z: f64) -> (usize, usize, f64) {
        let zs = &self.z_grid;
        let n = zs.len();
        if n <= 1 || z <= zs[0] {
            return (0, 0, 0.0);
        }
        if z >= zs[n - 1] {
            return (n - 1, n - 1, 0.0);
        }
        let (mut lo, mut hi) = (0usize, n - 1);
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if zs[mid] < z {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let t = ((z - zs[lo]) / (zs[hi] - zs[lo])).clamp(0.0, 1.0);
        (lo, hi, t)
    }

    /// Grid sensitivity `dz/d(de_max)` of the node nearest to `z` (the
    /// quantised protocol reads the node value, which moves with the geometry
    /// whenever the grid is built from `de_max`).
    fn lr_node_sens(&self, idx: usize) -> f64 {
        self.z_grid_sens.get(idx).copied().unwrap_or(0.0)
    }

    /// `d W_m / d z` of a pole whose response vector is `y`: the response
    /// diagonal is the only `z`-dependent ingredient, so
    /// `sum_ia (y . q_ia)^2 d d_ia(z)/dz`.
    fn residue_dwdz(&self, y: &[f64], z: f64) -> f64 {
        let naux = self.naux;
        let nov = self.de_ia.len();
        let mut sum = 0.0f64;
        for col in 0..nov {
            let mut proj = 0.0f64;
            for p in 0..naux {
                proj += y[p] * self.qia[p + col * naux];
            }
            sum += proj * proj * dd_response_dz_single(self.de_ia[col], z, self.config.eta);
        }
        sum
    }

    /// `y = s_metric * (qt + corr)` and `y0 = s_metric * qt` of one pole from
    /// the factor's metric intermediate `qt = S q_nm`.
    fn residue_vectors(&self, factor: &LrFactor, qt: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let naux = self.naux;
        let corr = factor.correction_from_qt(qt, 1);
        let y0 = gemm_nn(&self.s_metric, naux, naux, qt, naux, 1);
        let sc = gemm_nn(&self.s_metric, naux, naux, &corr, naux, 1);
        let y: Vec<f64> = y0.iter().zip(sc.iter()).map(|(a, b)| a + b).collect();
        (y, y0)
    }

    /// One value-only residue record (the static-subtracted rebalance reads
    /// only the pole index and the weight; the pullback never runs on it).
    fn push_value_residue(
        m: usize,
        pole_weight: f64,
        z: f64,
        residues: &mut Vec<Residue>,
    ) {
        residues.push(Residue {
            m,
            pole_weight,
            weight_slope: 0.0,
            interp_t: 0.0,
            secant: 0.0,
            entries: vec![ResidueEntry {
                weight: pole_weight,
                z,
                node_sens: 0.0,
                y: Vec::new(),
                y0: Vec::new(),
            }],
        });
    }

    /// One pole contribution of the contour term in the **exact-residue**
    /// protocol (`z = |e_m - omega|`, a fresh diagonalisation per pole).
    fn add_pole(
        &self,
        m: usize,
        de: f64,
        n: usize,
        collect: bool,
        contour: &mut f64,
        residues: &mut Vec<Residue>,
    ) {
        let (pole_factor, pole_factor_slope) = self.pole_factor_and_slope(de);
        if pole_factor == 0.0 && pole_factor_slope == 0.0 {
            return;
        }
        let sign = if m < self.nocc { -1.0 } else { 1.0 };
        let pole_weight = sign * pole_factor;
        let pole_weight_slope = sign * pole_factor_slope;
        let z_used = de.abs();
        let qnm = self.with_q(|q| self.gather_pair(q, n, m));
        // `qt = s_metric * qnm` (the intermediate of the factor) and
        // `w = sum_P qnm[P] (y[P] - y0[P]) = sum_P qt[P] corr[P]`
        let mm = gemm_nn(&self.s_metric, self.naux, self.naux, &qnm, self.naux, 1);
        // fresh diagonalisation at the true residue frequency
        let factor = self.with_qtia(|qtia| {
            build_lr_factor(
                qtia, self.naux, self.de_ia.len(), &self.de_ia,
                z_used, 'R', self.config.eta, self.config.low_rank_tolerance,
            )
        });
        let w = factor.response_value_from_qt(&mm, 1)[0];
        *contour += pole_weight * w;
        if !collect {
            Self::push_value_residue(m, pole_weight, z_used, residues);
            return;
        }
        let (y, y0) = self.residue_vectors(&factor, &mm);
        let dwdz = self.residue_dwdz(&y, z_used);
        residues.push(Residue {
            m,
            pole_weight,
            // the factor frequency follows `e_m`, so `dW/de_m = dwdz * zsign`;
            // `pole_weight_slope * w` is the `d(pole_factor)/dz` half.
            weight_slope: pole_weight * dwdz + pole_weight_slope * w,
            interp_t: 0.0,
            secant: 0.0,
            // the exact protocol never reads the grid, so the entries carry no
            // node sensitivity
            entries: vec![ResidueEntry {
                weight: pole_weight,
                z: z_used,
                node_sens: 0.0,
                y,
                y0,
            }],
        });
    }

    /// Residue (contour) part of the self-energy of orbital `n` at `omega`.
    ///
    /// The active poles are evaluated **together, grouped by the response
    /// factors they share**: in the quantised protocols many transition
    /// energies map onto the same pair of cached low-rank factors, and
    /// gathering the pole columns into one `[naux, ncol]` block turns the
    /// per-pole matrix-vector products against `s_metric` (one full `naux^2`
    /// read each) into a handful of BLAS-3 products.  The exact-residue
    /// protocol needs a fresh factor per pole and keeps the per-pole route.
    ///
    /// The interpolation rule is [`GwGradConfig::low_rank_interp`] and must
    /// match the energy path, which defaults to `"linear"`.
    fn add_poles(
        &self,
        omega: f64,
        n: usize,
        collect: bool,
        contour: &mut f64,
        residues: &mut Vec<Residue>,
    ) {
        let nocc = self.nocc;
        let ef = 0.5 * (self.e[nocc - 1] + self.e[nocc]);
        if self.config.exact_residue_z {
            let branch: Box<dyn Iterator<Item = usize>> = if omega > ef {
                Box::new(nocc..self.nmo)
            } else {
                Box::new(0..nocc)
            };
            for m in branch {
                let de = if omega > ef { omega - self.e[m] } else { self.e[m] - omega };
                self.add_pole(m, de, n, collect, contour, residues);
            }
            return;
        }
        if self.config.low_rank_interp == 1 {
            self.add_poles_linear(omega, n, collect, contour, residues);
        } else {
            self.add_poles_nearest(omega, n, collect, contour, residues);
        }
    }

    /// Active residue poles of orbital `n` at `omega`: `(m, signed pole
    /// factor, |e_m - omega|)`, with REST's branch and `pole_factor` rules.
    /// `pole_factor` of a residue pole and its derivative with respect to the
    /// pole frequency `z = |e_m - omega|`.
    ///
    /// Historical **hard ladder** (`smooth_pole = false`): `0` for
    /// `de < -res_tol`, `0.5` for `|de| < res_tol`, `1` above.  That is two
    /// value jumps in `Sigma_c`, and they sit exactly where the QP root of a
    /// high virtual lives, so the analytic gradient (a derivative *between* the
    /// jumps) cannot match a central difference.
    ///
    /// **C^1 switch** (`smooth_pole = true`): the cubic smoothstep with the same
    /// endpoints and the same `0.5` at `z = 0`, but continuous everywhere.  The
    /// pullback then receives the matching `d(pole_factor)/dz * W` contribution
    /// through `Residue::weight_slope`.
    fn pole_factor_and_slope(&self, de: f64) -> (f64, f64) {
        let w = self.config.res_tol;
        if w <= 0.0 {
            return (1.0, 0.0);
        }
        if !self.config.smooth_pole {
            if de < -w {
                return (0.0, 0.0);
            }
            return (if de.abs() < w { 0.5 } else { 1.0 }, 0.0);
        }
        let t = (de / w).clamp(-1.0, 1.0);
        let f = 0.5 * (1.0 + 0.5 * (3.0 * t - t * t * t));
        let df = if t.abs() >= 1.0 {
            0.0
        } else {
            0.25 * (3.0 - 3.0 * t * t) / w
        };
        (f, df)
    }

    /// Active residue poles of orbital `n` at `omega`:
    /// `(m, signed pole factor, |e_m - omega|, d(pole factor)/dz)`.
    fn active_poles(&self, omega: f64) -> Vec<(usize, f64, f64, f64)> {
        let nocc = self.nocc;
        let ef = 0.5 * (self.e[nocc - 1] + self.e[nocc]);
        let above = omega > ef;
        let branch: Box<dyn Iterator<Item = usize>> =
            if above { Box::new(nocc..self.nmo) } else { Box::new(0..nocc) };
        let mut out = Vec::new();
        for m in branch {
            let de = if above { omega - self.e[m] } else { self.e[m] - omega };
            let (f, df) = self.pole_factor_and_slope(de);
            if !self.config.smooth_pole && de < -self.config.res_tol {
                continue;
            }
            if f == 0.0 && df == 0.0 {
                continue;
            }
            let sign = if m < nocc { -1.0 } else { 1.0 };
            out.push((m, sign * f, de.abs(), sign * df));
        }
        out
    }

    /// Quantised protocol with the **nearest grid point** rule: the response
    /// factor is the one at `nearest_grid_value(z_grid, |de|)`.
    fn add_poles_nearest(
        &self,
        omega: f64,
        n: usize,
        collect: bool,
        contour: &mut f64,
        residues: &mut Vec<Residue>,
    ) {
        let naux = self.naux;
        let poles = self.active_poles(omega);
        if poles.is_empty() {
            return;
        }
        // group the poles by the index of their nearest grid frequency
        let mut groups: std::collections::BTreeMap<usize, Vec<(usize, f64, f64, f64)>> =
            std::collections::BTreeMap::new();
        for (m, pw, de, pws) in poles {
            let z_used = nearest_grid_value(&self.z_grid, de);
            groups
                .entry(self.lr_factor_nearest_idx(z_used))
                .or_default()
                .push((m, pw, z_used, pws));
        }
        self.ensure_lr();
        self.ensure_q();
        for idx in groups.keys() {
            self.ensure_lr_grid_factor(*idx);
        }
        let lr_grid = self.lr_grid.borrow();
        let q_all = self.q.borrow();
        for (idx, poles) in groups.iter() {
            let factor = lr_grid[*idx]
                .as_ref()
                .expect("gw_grad: real-axis factor missing");
            let sens = self.lr_node_sens(*idx);
            let ncol = poles.len();
            // gather the `q[:, n, m]` columns of this group
            let mut qmat = vec![0.0f64; naux * ncol];
            for (c, &(m, ..)) in poles.iter().enumerate() {
                let base = self.pair_col(n, m) * naux;
                qmat[c * naux..(c + 1) * naux].copy_from_slice(&q_all[base..base + naux]);
            }
            let qt = gemm_nn(&self.s_metric, naux, naux, &qmat, naux, ncol);
            let w = factor.response_value_from_qt(&qt, ncol);
            for (c, &(_, pw, _, _)) in poles.iter().enumerate() {
                *contour += pw * w[c];
            }
            if !collect {
                for &(m, pw, z_used, _) in poles.iter() {
                    Self::push_value_residue(m, pw, z_used, residues);
                }
                continue;
            }
            let y0 = gemm_nn(&self.s_metric, naux, naux, &qt, naux, ncol);
            let corr = factor.correction_from_qt(&qt, ncol);
            let sc = gemm_nn(&self.s_metric, naux, naux, &corr, naux, ncol);
            for (c, &(m, pw, z_used, pw_slope)) in poles.iter().enumerate() {
                let y0c = y0[c * naux..(c + 1) * naux].to_vec();
                let y: Vec<f64> = y0c.iter().zip(sc[c * naux..(c + 1) * naux].iter())
                    .map(|(a, b)| a + b)
                    .collect();
                residues.push(Residue {
                    m,
                    pole_weight: pw,
                    // a grid node does not follow `e_m`; only the smooth
                    // `pole_factor` moves with the pole
                    weight_slope: pw_slope * w[c],
                    interp_t: 0.0,
                    secant: 0.0,
                    entries: vec![ResidueEntry { weight: pw, z: z_used, node_sens: sens, y, y0: y0c }],
                });
            }
        }
    }

    /// Quantised protocol with the **linear interpolation** rule (the energy
    /// path's default): the pole response is
    /// `(1-t) W(z_lo) + t W(z_hi)` with `t` from the bracketing grid nodes, and
    /// the gradient carries both the weight derivative (`dt/dR`) and the
    /// movement of the two nodes.
    fn add_poles_linear(
        &self,
        omega: f64,
        n: usize,
        collect: bool,
        contour: &mut f64,
        residues: &mut Vec<Residue>,
    ) {
        let naux = self.naux;
        let poles = self.active_poles(omega);
        if poles.is_empty() {
            return;
        }
        // group by the bracketing pair of grid nodes; `t` stays per column
        let mut groups: std::collections::BTreeMap<(usize, usize), Vec<(usize, f64, f64, f64, f64)>> =
            std::collections::BTreeMap::new();
        for (m, pw, z, pws) in poles {
            let (lo, hi, t) = self.lr_bracket(z);
            groups.entry((lo, hi)).or_default().push((m, pw, t, z, pws));
        }
        self.ensure_lr();
        self.ensure_q();
        for (lo, hi) in groups.keys() {
            self.ensure_lr_grid_factor(*lo);
            self.ensure_lr_grid_factor(*hi);
        }
        let lr_grid = self.lr_grid.borrow();
        let q_all = self.q.borrow();
        for ((lo, hi), poles) in groups.iter() {
            let single = lo == hi;
            let (z_lo, sens_lo) = (self.z_grid[*lo], self.lr_node_sens(*lo));
            let (z_hi, sens_hi) = (self.z_grid[*hi], self.lr_node_sens(*hi));
            let f_lo = lr_grid[*lo].as_ref().expect("gw_grad: real-axis factor missing");
            let f_hi = lr_grid[*hi].as_ref().expect("gw_grad: real-axis factor missing");
            let ncol = poles.len();
            let mut qmat = vec![0.0f64; naux * ncol];
            for (c, &(m, ..)) in poles.iter().enumerate() {
                let base = self.pair_col(n, m) * naux;
                qmat[c * naux..(c + 1) * naux].copy_from_slice(&q_all[base..base + naux]);
            }
            let qt = gemm_nn(&self.s_metric, naux, naux, &qmat, naux, ncol);
            let w_lo = f_lo.response_value_from_qt(&qt, ncol);
            let w_hi = if single { None } else { Some(f_hi.response_value_from_qt(&qt, ncol)) };
            for (c, &(_, pw, t, _, _)) in poles.iter().enumerate() {
                let wv = match w_hi.as_ref() {
                    None => w_lo[c],
                    Some(wh) => (1.0 - t) * w_lo[c] + t * wh[c],
                };
                *contour += pw * wv;
            }
            if !collect {
                for &(m, pw, _, z, _) in poles.iter() {
                    Self::push_value_residue(m, pw, z, residues);
                }
                continue;
            }
            let y0 = gemm_nn(&self.s_metric, naux, naux, &qt, naux, ncol);
            let corr_lo = f_lo.correction_from_qt(&qt, ncol);
            let sc_lo = gemm_nn(&self.s_metric, naux, naux, &corr_lo, naux, ncol);
            let hi_pair = if single {
                None
            } else {
                let ch = f_hi.correction_from_qt(&qt, ncol);
                let sh = gemm_nn(&self.s_metric, naux, naux, &ch, naux, ncol);
                Some(sh)
            };
            for (c, &(m, pw, t, _, pw_slope)) in poles.iter().enumerate() {
                let wv = match w_hi.as_ref() {
                    None => w_lo[c],
                    Some(wh) => (1.0 - t) * w_lo[c] + t * wh[c],
                };
                let col = c * naux..(c + 1) * naux;
                let y0c = y0[col.clone()].to_vec();
                let y_lo: Vec<f64> =
                    y0c.iter().zip(sc_lo[col.clone()].iter()).map(|(a, b)| a + b).collect();
                if single {
                    // outside the grid (or a single-node grid) the clamped value
                    // only follows its one node
                    residues.push(Residue {
                        m,
                        pole_weight: pw,
                        weight_slope: 0.0,
                        interp_t: 0.0,
                        secant: 0.0,
                        entries: vec![ResidueEntry {
                            weight: pw,
                            z: z_lo,
                            node_sens: sens_lo,
                            y: y_lo,
                            y0: y0c,
                        }],
                    });
                    continue;
                }
                let sc_hi = hi_pair.as_ref().expect("gw_grad: hi factor missing");
                let y_hi: Vec<f64> =
                    y0c.iter().zip(sc_hi[col].iter()).map(|(a, b)| a + b).collect();
                let wh = w_hi.as_ref().expect("gw_grad: hi value missing");
                let secant = (wh[c] - w_lo[c]) / (z_hi - z_lo);
                residues.push(Residue {
                    m,
                    pole_weight: pw,
                    weight_slope: pw * secant + pw_slope * wv,
                    interp_t: t,
                    secant,
                    entries: vec![
                        ResidueEntry {
                            weight: pw * (1.0 - t),
                            z: z_lo,
                            node_sens: sens_lo,
                            y: y_lo,
                            y0: y0c.clone(),
                        },
                        ResidueEntry {
                            weight: pw * t,
                            z: z_hi,
                            node_sens: sens_hi,
                            y: y_hi,
                            y0: y0c,
                        },
                    ],
                });
            }
        }
    }

    /// Build the [`QpRowTable`] of orbital `n` (see there).
    fn qp_row_table(&self, n: usize) -> QpRowTable {
        let qn = self.with_q(|q| self.gather_row(q, n));
        let m = gemm_nn(&self.s_metric, self.naux, self.naux, &qn, self.naux, self.nmo);
        self.ensure_lr();
        let lr_imag = self.lr_imag.borrow();
        let w_imag = (0..self.quad.len())
            .map(|wi| lr_imag[wi].response_value_from_qt(&m, self.nmo))
            .collect();
        let w0 = self.lr_static_row_values(&m);
        QpRowTable { m, w_imag, w0 }
    }

    /// All pieces of the self-energy at a fixed `omega` from a precomputed
    /// [`QpRowTable`]; optionally collects the response data needed by the
    /// gradient.  Returns `(contour, imag, y_imag, residues)`.
    ///
    /// The value of the self-energy only needs the table's `w_imag` numbers,
    /// which follow from the *element-wise* product of `m` with the low-rank
    /// correction (`y - y0 = s_metric * corr`), so the `y`/`y0` vectors (2
    /// `[naux,nmo]` buffers and one `[naux,naux]` gemm each per quadrature point)
    /// are only built when `want_y` asks for them.
    #[allow(clippy::type_complexity)]
    fn sigma_c_parts_with(
        &self,
        omega: f64,
        n: usize,
        collect: bool,
        want_y: bool,
        table: &QpRowTable,
    ) -> (
        f64,
        f64,
        Vec<(f64, Vec<f64>, Vec<f64>)>,
        Vec<Residue>,
    ) {
        let want_y = want_y && collect;
        let mut imag = 0.0f64;
        let mut y_imag_all: Vec<(f64, Vec<f64>, Vec<f64>)> = Vec::new();
        if want_y {
            let y0 = gemm_nn(&self.s_metric, self.naux, self.naux, &table.m, self.naux, self.nmo);
            self.ensure_lr();
            let lr_imag = self.lr_imag.borrow();
            for (wi, &(u, _wgt)) in self.quad.iter().enumerate() {
                let corr = lr_imag[wi].correction_from_qt(&table.m, self.nmo);
                let sc = gemm_nn(&self.s_metric, self.naux, self.naux, &corr, self.naux, self.nmo);
                let y: Vec<f64> = y0.iter().zip(sc.iter()).map(|(a, b)| a + b).collect();
                y_imag_all.push((u, y, y0.clone()));
            }
        }
        for (wi, &(u, wgt)) in self.quad.iter().enumerate() {
            let w = &table.w_imag[wi];
            let mut sum_m = 0.0f64;
            for mm in 0..self.nmo {
                let t = omega - self.e[mm];
                sum_m += 2.0 * t / (t * t + u * u) * w[mm];
            }
            imag += sum_m * wgt / TWO_PI;
        }
        // contour (residue) part with REST pole selection and nearest grid
        let mut contour = 0.0f64;
        let mut residues: Vec<Residue> = Vec::new();
        self.add_poles(omega, n, collect, &mut contour, &mut residues);
        // ---- static-subtracted rebalance (see the module header) ----
        // Sigma_sub = Sigma_raw + sum_m [A_m + 1/2 - f_m - a_m] W_m(0),
        // A_m = 1/pi sum_k w_k K(t_m,u_k), a_m = s_m pole_factor of the
        // active poles; the residues metadata carries a_m also in value-only
        // calls.  Away from pole thresholds A_m = 1/2 sign(t_m) to ~1e-9 on
        // REST's grids, so the two functionals coincide there.
        if !table.w0.is_empty() {
            for m in 0..self.nmo {
                let f_m = if m < self.nocc { 1.0 } else { 0.0 };
                let t = omega - self.e[m];
                let mut a_m = 0.0f64;
                for &(u, wgt) in self.quad.iter() {
                    a_m += wgt / TWO_PI * 2.0 * t / (t * t + u * u);
                }
                imag -= a_m * table.w0[m];
                contour += (0.5 - f_m) * table.w0[m];
            }
            for r in residues.iter() {
                contour -= r.pole_weight * table.w0[r.m];
            }
        }
        (contour, imag, y_imag_all, residues)
    }

    /// [`Self::sigma_c_parts_with`] with the row table built on the fly.
    #[allow(clippy::type_complexity)]
    fn sigma_c_parts(
        &self,
        omega: f64,
        n: usize,
        collect: bool,
        want_y: bool,
    ) -> (
        f64,
        f64,
        Vec<Vec<f64>>,
        Vec<(f64, Vec<f64>, Vec<f64>)>,
        Vec<Residue>,
        Vec<f64>,
    ) {
        let table = self.qp_row_table(n);
        let (contour, imag, y_imag_all, residues) =
            self.sigma_c_parts_with(omega, n, collect, want_y, &table);
        (contour, imag, table.w_imag, y_imag_all, residues, table.m)
    }

    /// Self-energy value from a precomputed row table: the imaginary-axis part
    /// is a weighted sum over the table (`O(nquad*nmo)`, no gemms) and only the
    /// residue part is re-evaluated, so scanning the QP equation costs
    /// microseconds per point instead of a full self-energy evaluation.
    fn sigma_c_with_table(&self, omega: f64, n: usize, table: &QpRowTable) -> f64 {
        let (contour, imag, _y, _res) = self.sigma_c_parts_with(omega, n, false, false, table);
        contour - imag
    }

    /// [`Self::sigma_c_and_d`] from a precomputed row table (the Newton loop).
    fn sigma_c_and_d_with(&self, omega: f64, n: usize, table: &QpRowTable) -> (f64, f64) {
        let (contour, imag, _y, residues) = self.sigma_c_parts_with(omega, n, true, false, table);
        let ds = self.sigma_d_from_parts(omega, &table.w_imag, &table.w0, &residues);
        (contour - imag, ds)
    }

    /// Compact interaction storage of a cache: the single metric intermediate
    /// `m = s_metric * q_n`, or empty when the full `y`/`y0` pair is kept.
    ///
    /// `m` does not depend on the quadrature frequency (`y0 = s_metric * m`,
    /// `y = y0 + s_metric * corr(m, u)`), so one `[naux, nmo]` buffer replaces
    /// the `nquad` identical copies the cache used to hold.
    fn compact_m_imag(&self, m: &[f64], compact: bool) -> Vec<f64> {
        if !compact {
            return Vec::new();
        }
        m.to_vec()
    }

    /// Static response row `y0s = (J - Q(0))^-1 q_n` of a cache: stored in the
    /// full (`y_imag`) representation, rebuilt on demand from `m` in the
    /// compact one.  Empty when the engine runs raw (`w0` empty).
    fn compact_y0_static(&self, m: &[f64], w0: &[f64], want_y: bool) -> Vec<f64> {
        if w0.is_empty() || !want_y {
            return Vec::new();
        }
        self.lr_static_row_response(m)
    }

    /// Sample the QP equation `f(omega) = consts + Sigma_c(omega) - omega` of
    /// one orbital on a uniform grid.  Validation aid for the Newton solver:
    /// the quantised residue protocol makes `f` piecewise continuous with
    /// jumps, so it is worth seeing where the sign changes.
    pub fn sample_qp_equation(&self, n: usize, lo: f64, hi: f64, npts: usize) -> Vec<(f64, f64)> {
        let consts = self.consts(n);
        (0..npts)
            .map(|i| {
                let x = lo + (hi - lo) * i as f64 / (npts - 1) as f64;
                (x, consts + self.sigma_c(x, n) - x)
            })
            .collect()
    }

    /// Sample `(f(omega), f'(omega))` of the QP equation of one orbital, where
    /// `f'` is the derivative the solver uses (`sigma_c_and_d`).  Validation
    /// aid: in a self-consistent protocol the analytic `f'` must agree with a
    /// central difference of `f`.
    pub fn qp_equation_consistency(
        &self,
        n: usize,
        omegas: &[f64],
        h: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let consts = self.consts(n);
        let f = |om: f64| consts + self.sigma_c(om, n) - om;
        let mut vals = Vec::with_capacity(omegas.len());
        let mut ana = Vec::with_capacity(omegas.len());
        let mut num = Vec::with_capacity(omegas.len());
        for &om in omegas {
            vals.push(f(om));
            let (_sc, ds) = self.sigma_c_and_d(om, n);
            ana.push(ds - 1.0);
            num.push((f(om + h) - f(om - h)) / (2.0 * h));
        }
        (vals, ana, num)
    }

    /// The metric intermediate `m = s_metric * q_n` of one orbital row.
    pub fn row_metric_intermediate(&self, n: usize) -> Vec<f64> {
        self.ensure_q();
        let qn = self.with_q(|q| self.gather_row(q, n));
        gemm_nn(&self.s_metric, self.naux, self.naux, &qn, self.naux, self.nmo)
    }

    /// Rebuild `(y, y0)` of one quadrature point from `m`
    /// (see [`Self::screened_row_compact`]).
    pub fn row_response_from_m(&self, wi: usize, m: &[f64]) -> (Vec<f64>, Vec<f64>) {
        self.ensure_lr();
        let lr_imag = self.lr_imag.borrow();
        let y = lr_imag[wi].response_vectors_from_qt(&self.s_metric, m, self.nmo);
        let y0 = gemm_nn(&self.s_metric, self.naux, self.naux, m, self.naux, self.nmo);
        (y, y0)
    }

    /// Raw CD self-energy `Sigma_c(omega) = contour(omega) - imag(omega)`.
    pub fn sigma_c(&self, omega: f64, n: usize) -> f64 {
        let (contour, imag, _, _, _, _) = self.sigma_c_parts(omega, n, false, false);
        contour - imag
    }

    /// Decomposed self-energy at fixed omega (for validation).  With the
    /// static-subtracted functional the parts are already rebalanced: the
    /// returned `imag` carries `W(iu) - W(0)` kernels and `contour` the static
    /// and residue `W(0)` bookkeeping, so `contour - imag` is `Sigma_sub`.
    pub fn sigma_c_decomposed(&self, omega: f64, n: usize) -> (f64, f64) {
        let (contour, imag, _, _, _, _) = self.sigma_c_parts(omega, n, false, false);
        (contour, imag)
    }

    /// Omega-derivative pieces from already-collected response data:
    /// `w_imag[wi][m]` from the imaginary pass and the residue vectors.
    ///
    /// With the static-subtracted functional (`w0` non-empty) the axis kernel
    /// multiplies `W_m(iu) - W_m(0)`, adding `sum_k w_k K'/pi * W_m(0)`; the
    /// static and residue subtractions themselves are omega-independent.
    fn sigma_d_from_parts(
        &self,
        omega: f64,
        w_imag: &[Vec<f64>],
        w0: &[f64],
        residues: &[Residue],
    ) -> f64 {
        let mut d = 0.0f64;
        for (wi, &(u, wgt)) in self.quad.iter().enumerate() {
            let w = &w_imag[wi];
            let mut sum_m = 0.0;
            for m in 0..self.nmo {
                let t = omega - self.e[m];
                sum_m += 2.0 * (u * u - t * t) / ((t * t + u * u) * (t * t + u * u)) * w[m];
            }
            d -= sum_m * wgt / TWO_PI;
            if !w0.is_empty() {
                let mut sum_0 = 0.0;
                for m in 0..self.nmo {
                    let t = omega - self.e[m];
                    sum_0 += 2.0 * (u * u - t * t) / ((t * t + u * u) * (t * t + u * u))
                        * w0[m];
                }
                d += sum_0 * wgt / TWO_PI;
            }
        }
        // Residue contribution to `d Sigma / d omega`.  `weight_slope` is
        // `dW_m/dz` of the interpolation rule (the factor is evaluated at the
        // exact `|e_m - omega|` in the exact protocol, so its frequency follows
        // `omega`; a grid node does not move with `omega`, so the nearest rule
        // contributes nothing here), and `z = |e_m - omega|` adds `-zsign`.
        if self.config.exact_residue_z || self.config.low_rank_interp == 1 {
            for r in residues {
                let zsign = if self.e[r.m] - omega >= 0.0 { 1.0 } else { -1.0 };
                d += (-zsign) * r.weight_slope;
            }
        }
        d
    }

    /// `d^2 Sigma_c / d omega^2` at fixed geometry, from already-collected
    /// response data (`w_imag[wi][m]` and the static row `w0`).
    ///
    /// Only the imaginary-axis integral contributes: the residue weights of the
    /// linear interpolation rule have a **constant** slope on each bracketing
    /// interval (the grid nodes do not move with `omega`), and the
    /// exact-residue protocol is rejected by the omega channel of the pullback.
    /// The static `W(0)` subtraction enters through `w0` exactly as in
    /// [`Self::sigma_d_from_parts`].
    fn sigma_d2_from_parts(&self, omega: f64, w_imag: &[Vec<f64>], w0: &[f64]) -> f64 {
        let mut d2 = 0.0f64;
        for (wi, &(u, wgt)) in self.quad.iter().enumerate() {
            let w = &w_imag[wi];
            let mut sum_m = 0.0;
            for m in 0..self.nmo {
                let t = omega - self.e[m];
                let dd = t * t + u * u;
                // d^2/d omega^2 [ 2t/(t^2+u^2) ] with `t = omega - e_m`
                sum_m += -4.0 * t * (3.0 * u * u - t * t) / (dd * dd * dd) * w[m];
            }
            d2 -= sum_m * wgt / TWO_PI;
            if !w0.is_empty() {
                let mut sum_0 = 0.0;
                for m in 0..self.nmo {
                    let t = omega - self.e[m];
                    let dd = t * t + u * u;
                    sum_0 += -4.0 * t * (3.0 * u * u - t * t) / (dd * dd * dd) * w0[m];
                }
                d2 += sum_0 * wgt / TWO_PI;
            }
        }
        d2
    }

    /// `Sigma_c` and `d Sigma_c / d omega` in one pass: the imaginary-axis
    /// screened rows and the pole factors are shared between the value and
    /// the derivative (the Newton solver evaluates both at every step).
    pub fn sigma_c_and_d(&self, omega: f64, n: usize) -> (f64, f64) {
        let table = self.qp_row_table(n);
        self.sigma_c_and_d_with(omega, n, &table)
    }

    /// `d Sigma_c / d omega` at fixed geometry.  In the REST quantised
    /// protocol the residue frequencies are piecewise constant, so only the
    /// imaginary-axis part contributes away from orbital thresholds; in the
    /// exact mode the moving residue frequencies add the standard zeta term.
    pub fn d_sigma_d_omega(&self, omega: f64, n: usize) -> f64 {
        self.sigma_c_and_d(omega, n).1
    }

    /// Solve the QP equation with Newton's method (analytic derivative,
    /// graphical-solver fallback on the jumps of the quantised residue
    /// protocol) and assemble the gradient cache at the converged root.
    pub fn build_target_cache(&self, n: usize) -> QpCache {
        self.build_target_cache_with(n, !self.config.compact_qp_cache)
    }

    /// [`Self::build_target_cache`] with the **compact** interaction storage
    /// ([`QpCache::m_imag`]) regardless of `config.compact_qp_cache`.
    ///
    /// The BSE gradient uses this for every cache it builds: its caches are its
    /// dominant memory term (`nocc + nvir_act` orbitals) and the pullback
    /// rebuilds `y`/`y0` from `m` anyway, so the full `nquad`-copy
    /// representation is never needed there.
    pub fn build_target_cache_compact(&self, n: usize) -> QpCache {
        self.build_target_cache_with(n, false)
    }

    /// Cache of the **static on-shell** quasiparticle energy
    ///
    /// ```text
    /// E = consts_n + Re Sigma_c(e_n),      consts_n = e_n + (1-a)Sigma_x,nn - v_xc,nn
    /// ```
    ///
    /// i.e. MolGW's `EVSC` / `se%nomega = 0` rule: the self-energy is evaluated
    /// **once at the mean-field energy**, there is no root search and no `Z`
    /// renormalisation.  Selected by [`GwGradConfig::static_fallback_threshold`]
    /// for deep levels, and used whenever a root finder cannot bracket a
    /// crossing (which is what the energy path does in both cases).
    ///
    /// The frequency does **not** follow the QP energy, so the pullback must use
    /// [`QpCache::freq`] and the gradient carries the extra
    /// `(d Sigma_c/d omega) * d e_n/dR` term (`dsigma_domega`); see
    /// [`QpCache::static_fallback`].
    /// One-step Newton ("linearized") quasiparticle energy
    ///
    /// ```text
    /// f(w)   = w - consts_n - Sigma_c(w)
    /// E_lin  = e_n - f(e_n)/f'(e_n) = e_n + Z(e_n) [ consts_n + Sigma_c(e_n) - e_n ]
    /// Z(w)   = 1/(1 - dSigma_c/dw)
    /// ```
    ///
    /// Used whenever the root search does not converge, or when the converged
    /// root carries a spectral weight below `GwGradConfig::z_min`.  The
    /// self-energy frequency is the **mean-field** `e_n`, so the analytic
    /// gradient carries the extra `Z^2 * lin_shift * d_R d_omega Sigma_c` term
    /// (the omega-derivative pullback channel) in addition to the usual
    /// `Z`-scaled one.
    /// One-step Newton ("linearized") branch guarded by the **physical
    /// spectral weight**: the step is only meaningful while `Z > 0`.  A
    /// non-positive `Z` means the slope of the QP equation at the
    /// linearization point has the wrong sign (`d Sigma_c/d omega > 1`), so the
    /// step runs away from `e_n` instead of interpolating towards the root
    /// (observed: `Z = -62.9` moves orbital 18 of water/def2-SVP by -2.4 Ha).
    /// Such an orbital uses the static on-shell energy instead, which depends
    /// smoothly on the geometry.
    fn linearized_guarded(
        &self,
        n: usize,
        consts: f64,
        table: &QpRowTable,
        want_y: bool,
    ) -> QpCache {
        let (sigma_n, dsdw) = self.sigma_c_and_d_with(self.e[n], n, table);
        let z = 1.0 / (1.0 - dsdw);
        if !(z > 0.0) {
            let shift = consts + sigma_n - self.e[n];
            println!(
                "warning!!! gw_grad orbital {n}: linearized (one-step Newton) Z = {z:.4} \
                 is not positive (step would be {:+.3e} Ha); using the static on-shell energy",
                z * shift
            );
            return self.static_fallback_cache(n, consts, table, want_y);
        }
        self.linearized_cache(n, consts, table, want_y)
    }

    fn linearized_cache(&self, n: usize, consts: f64, table: &QpRowTable, want_y: bool) -> QpCache {
        let e_n = self.e[n];
        let (sigma_n, dsdw_n) = self.sigma_c_and_d_with(e_n, n, table);
        let z = 1.0 / (1.0 - dsdw_n);
        let shift = consts + sigma_n - e_n;
        let (_c, _i, y_imag, residues) = self.sigma_c_parts_with(e_n, n, true, want_y, table);
        QpCache {
            target: n,
            omega: e_n + z * shift,
            freq: e_n,
            z_factor: z,
            dsigma_domega: dsdw_n,
            static_fallback: false,
            linearized: true,
            lin_shift: shift,
            w_imag: table.w_imag.clone(),
            y_imag: if want_y { y_imag } else { Vec::new() },
            m_imag: self.compact_m_imag(&table.m, !want_y),
            w0: table.w0.clone(),
            y0_static: self.compact_y0_static(&table.m, &table.w0, want_y),
            residues,
        }
    }

    fn static_fallback_cache(
        &self,
        n: usize,
        consts: f64,
        table: &QpRowTable,
        want_y: bool,
    ) -> QpCache {
        let freq = self.e[n];
        let (contour, imag, y_imag, residues) =
            self.sigma_c_parts_with(freq, n, true, want_y, table);
        let dsdw = self.sigma_d_from_parts(freq, &table.w_imag, &table.w0, &residues);
        let m_imag = self.compact_m_imag(&table.m, !want_y);
        let y0_static = self.compact_y0_static(&table.m, &table.w0, want_y);
        QpCache {
            target: n,
            omega: consts + contour - imag,
            freq,
            z_factor: 1.0,
            dsigma_domega: dsdw,
            static_fallback: true,
            linearized: false,
            lin_shift: 0.0,
            w_imag: table.w_imag.clone(),
            y_imag: if want_y { y_imag } else { Vec::new() },
            m_imag,
            w0: table.w0.clone(),
            y0_static,
            residues,
        }
    }

    /// Shared body of [`Self::build_target_cache`] and
    /// [`Self::build_target_cache_compact`]; `want_y` keeps the full
    /// `y`/`y0` response of every quadrature point in the cache.
    fn build_target_cache_with(&self, n: usize, want_y: bool) -> QpCache {
        let consts = self.consts(n);
        let side = if n < self.nocc { -1.0 } else { 1.0 };
        let mut x = self.e[n] + side * 0.02;
        // The imaginary-axis table of this row does not depend on the trial QP
        // energy, so it is built once and reused by every Newton step (and by
        // the final cache assembly) instead of once per evaluation.
        let table = self.qp_row_table(n);
        // REST `gw_switch_fallback_threshold`: deep levels (and any state whose
        // QP equation cannot be solved) use the static on-shell energy instead
        // of a root — see `GwGradConfig::static_fallback_threshold`.
        // One-step on-shell (static) treatment: only **occupied** mean-field
        // levels below `-static_fallback_threshold` are forced onto it.
        let deeply_occupied = self.e[n] < 0.0
            && -self.e[n] > self.config.static_fallback_threshold;
        if deeply_occupied || self.config.force_static_fallback.contains(&n) {
            return self.static_fallback_cache(n, consts, &table, want_y);
        }
        // Validation pin (`gradient_force_linearized`): take the linearized
        // branch regardless of what the root search would do, so that a
        // displaced geometry cannot switch the branch under a finite
        // difference.  It is checked **before** the root solve so the branch is
        // the same functional at every geometry.
        if self.config.force_linearized.contains(&n) {
            return self.linearized_guarded(n, consts, &table, want_y);
        }
        // REST production **graphical solver** (`GwGradConfig::qpe_graphical`):
        // no Newton iteration at all — the linear-interpolation scan of
        // `ri_gw::linear_interpolation_solver` defines the root.  A scan
        // without a sign change returns the static-coupling fallback and the
        // root is flagged unconverged by its large `qp_equation_value`.
        if self.config.qpe_graphical {
            let fq = |om: f64| consts + self.sigma_c_with_table(om, n, &table) - om;
            let (have_crossing, xg) =
                crate::ri_gw::linear_interpolation_solver(&fq, self.e[n], side, 21, 0.1);
            if !have_crossing {
                println!(
                    "warning!!! gw_grad graphical solver found no crossing for orbital {n}!"
                );
            }
            let omega = xg;
            let (_contour, _imag, y_imag, residues) =
                self.sigma_c_parts_with(omega, n, true, want_y, &table);
            let (w_imag, m, w0) = (table.w_imag, table.m, table.w0);
            let dsdw = self.sigma_d_from_parts(omega, &w_imag, &w0, &residues);
            let m_imag = self.compact_m_imag(&m, !want_y);
            let y0_static = self.compact_y0_static(&m, &w0, want_y);
            let y_imag = if want_y { y_imag } else { Vec::new() };
            return QpCache {
                target: n,
                omega,
                freq: omega,
                z_factor: 1.0 / (1.0 - dsdw),
                dsigma_domega: dsdw,
                static_fallback: false,
                linearized: false,
                lin_shift: 0.0,
                w_imag,
                y_imag,
                m_imag,
                w0,
                y0_static,
                residues,
            };
        }
        // Smooth exact-residue mode: the residue frequencies follow the QP
        // energy, so the derivative carries the extra `d zeta/d omega` term of
        // `sigma_d_from_parts` / `d_sigma_d_omega` — both branches use the
        // analytic derivative.
        if self.config.exact_residue_z {
            let mut converged = 0;
            let mut iter = 0;
            while converged == 0 && iter < self.config.qpe_max_iter {
                // value and derivative share one response evaluation
                let (sc, dsc) = self.sigma_c_and_d_with(x, n, &table);
                let fv = consts + sc - x;
                let dfv = dsc - 1.0;
                let shift = -fv / dfv;
                if std::env::var("REST_GWGRAD_NEWTON").is_ok() {
                    println!("[newtonE DBG] n={} it={} x={:.10} f={:+.3e} shift={:+.3e}",
                        n, iter, x, fv, shift);
                }
                x += shift;
                if shift.abs() < self.config.qpe_tol {
                    converged += 1;
                }
                iter += 1;
            }
            if converged == 0 {
                // The QP equation of the quantised residue protocol is only
                // piecewise smooth: the residue term jumps whenever a pole
                // frequency crosses a real-axis grid point, so the Newton
                // iterate can keep crossing the same jump forever (observed for
                // the symmetric O-H combination of H2O/STO-3G, where the plain
                // Newton never settles and leaves the QP energy ~0.002 Ha off
                // the root).  Grid scanning is immune to that: reuse the
                // production graphical solver, which brackets the sign change
                // on a grid and interpolates linearly inside it.
                if self.config.exact_residue_z {
                    println!(
                        "warning!!! gw_grad Newton did not converge for orbital {n}; \
                         using the static on-shell energy (the continuous-residue \
                         protocol has no omega-derivative channel)"
                    );
                    return self.static_fallback_cache(n, consts, &table, want_y);
                }
                if !self.config.graphical_rescue {
                    println!(
                        "warning!!! gw_grad Newton did not converge for orbital {n}; \
                         using the linearized (one-step Newton) functional \
                         (graphical_rescue = false)"
                    );
                    return self.linearized_guarded(n, consts, &table, want_y);
                }
                let mut fq = |om: f64| consts + self.sigma_c(om, n) - om;
                let (have_crossing, xg) =
                    crate::ri_gw::linear_interpolation_solver(&fq, self.e[n], side, 21, 0.1);
                if have_crossing {
                    if std::env::var("REST_GWGRAD_NEWTON").is_ok() {
                        println!(
                            "[newtonE] n={} Newton stalled at {:.10}; graphical solver -> {:.10}",
                            n, x, xg
                        );
                    }
                    x = xg;
                    converged = 1;
                } else {
                    println!(
                        "warning!!! gw_grad Newton (exact mode) did not converge for orbital {n}; \
                         using the linearized (one-step Newton) functional"
                    );
                    return self.linearized_guarded(n, consts, &table, want_y);
                }
            }
            let omega = x;
            let (_contour, _imag, y_imag, residues) =
                self.sigma_c_parts_with(omega, n, true, want_y, &table);
            let (w_imag, m, w0) = (table.w_imag, table.m, table.w0);
            let dsdw = self.sigma_d_from_parts(omega, &w_imag, &w0, &residues);
            let m_imag = self.compact_m_imag(&m, !want_y);
            let y0_static = self.compact_y0_static(&m, &w0, want_y);
            let y_imag = if want_y { y_imag } else { Vec::new() };
            return QpCache {
                target: n,
                omega,
                freq: omega,
                z_factor: 1.0 / (1.0 - dsdw),
                dsigma_domega: dsdw,
                static_fallback: false,
                linearized: false,
                lin_shift: 0.0,
                w_imag,
                y_imag,
                m_imag,
                w0,
                y0_static,
                residues,
            };
        }
        // Newton on the **analytic** derivative of the QP equation.
        //
        // `sigma_c_and_d` returns the value and `d Sigma_c/d omega` from a
        // single self-energy pass, while the finite-difference derivative needs
        // three (`f(x)`, `f(x+h)`, `f(x-h)`) — and the old implementation also
        // formed the full `y`/`y0` response of every quadrature point on each of
        // them, most of that work being thrown away.  Both derivatives solve the
        // same equation, and the analytic one was already the documented
        // fallback when the FD derivative stalls on the jumps of the quantised
        // residue protocol.
        let mut converged = 0;
        let mut iter = 0;
        // `(x, f)` of the iterate before the last step: a Newton that stalls on
        // the residue jumps two-cycles, so these two bracket the discontinuity
        let mut prev: Option<(f64, f64)> = None;
        while converged == 0 && iter < self.config.qpe_max_iter {
            let (sc, dsc) = self.sigma_c_and_d_with(x, n, &table);
            let fv = consts + sc - x;
            let dfv = dsc - 1.0;
            let shift = -fv / dfv;
            if std::env::var("REST_GWGRAD_NEWTON").is_ok() {
                println!("[newton DBG] n={} it={} x={:.10} f={:+.3e} shift={:+.3e}",
                    n, iter, x, fv, shift);
            }
            prev = Some((x, fv));
            x += shift;
            if shift.abs() < self.config.qpe_tol {
                converged += 1;
            }
            iter += 1;
        }
        if converged == 0 {
            // The QP equation of the quantised residue protocol is only
            // piecewise smooth: the residue term jumps whenever a pole
            // frequency crosses a real-axis grid point, so a derivative-based
            // Newton iterate can keep crossing the same jump forever.  Grid
            // scanning is immune to that: reuse the production graphical
            // solver, which brackets the sign change on a grid and interpolates
            // linearly inside it.
            let fq = |om: f64| consts + self.sigma_c_with_table(om, n, &table) - om;
            if self.config.exact_residue_z {
                println!(
                    "warning!!! gw_grad Newton did not converge for orbital {n}; \
                     using the static on-shell energy (the continuous-residue \
                     protocol has no omega-derivative channel)"
                );
                return self.static_fallback_cache(n, consts, &table, want_y);
            }
            if !self.config.graphical_rescue {
                println!(
                    "warning!!! gw_grad Newton did not converge for orbital {n}; \
                     using the linearized (one-step Newton) functional \
                     (graphical_rescue = false)"
                );
                return self.linearized_guarded(n, consts, &table, want_y);
            }
            let (have_crossing, xg) =
                crate::ri_gw::linear_interpolation_solver(&fq, self.e[n], side, 21, 0.1);
            if have_crossing {
                x = xg;
                converged = 1;
            } else if let Some((xp, fp)) = prev {
                // The production scan steps 0.01 Ha, but a residue jump is only
                // as wide as the distance between two real-axis grid points
                // mapped back through the pole (1.6e-4 Ha for orbital 91 of
                // naphthalene/def2-TZVP), so the scan can step over it.  The
                // stalled two-cycle brackets the jump exactly, and the sign
                // change inside that bracket is what the graphical solver
                // reports — bisect it instead of giving up.
                let fc = fq(x);
                let (mut lo, mut hi, mut flo, fhi) = if xp <= x {
                    (xp, x, fp, fc)
                } else {
                    (x, xp, fc, fp)
                };
                if flo * fhi < 0.0 {
                    for _ in 0..200 {
                        if hi - lo < 1.0e-12 {
                            break;
                        }
                        let mid = 0.5 * (lo + hi);
                        let fm = fq(mid);
                        if fm == 0.0 {
                            lo = mid;
                            hi = mid;
                            break;
                        }
                        if flo * fm < 0.0 {
                            hi = mid;
                        } else {
                            lo = mid;
                            flo = fm;
                        }
                    }
                    x = 0.5 * (lo + hi);
                    converged = 1;
                    if grad_timing() {
                        eprintln!(
                            "[gw_grad] orbital {n}: Newton stalled across a residue jump; \
                             bisected it to {x:.12} (bracket {:.2e} wide)",
                            hi - lo
                        );
                    }
                } else {
                    println!(
                        "warning!!! gw_grad Newton solver did not converge for orbital {n}; \
                         using the linearized (one-step Newton) functional"
                    );
                    return self.linearized_guarded(n, consts, &table, want_y);
                }
            } else {
                println!(
                    "warning!!! gw_grad Newton solver did not converge for orbital {n}; \
                     using the linearized (one-step Newton) functional"
                );
                return self.linearized_guarded(n, consts, &table, want_y);
            }
        }
        let omega = x;

        let (_contour, _imag, y_imag, residues) =
            self.sigma_c_parts_with(omega, n, true, want_y, &table);
        // `dsdw` only needs the table by reference, and the `z_min` test may
        // hand the table to `linearized_cache`; destructure by move only after
        // that test.
        let dsdw = self.sigma_d_from_parts(omega, &table.w_imag, &table.w0, &residues);
        let z_root = 1.0 / (1.0 - dsdw);
        if self.config.z_min > 0.0
            && !self.config.exact_residue_z
            && z_root < self.config.z_min
        {
            println!(
                "warning!!! gw_grad orbital {n}: converged spectral weight Z = {z_root:.4} \
                 is below z_min = {:.3}; using the linearized (one-step Newton) functional",
                self.config.z_min
            );
            return self.linearized_guarded(n, consts, &table, want_y);
        }
        let (w_imag, m, w0) = (table.w_imag, table.m, table.w0);
        let m_imag = self.compact_m_imag(&m, !want_y);
        let y0_static = self.compact_y0_static(&m, &w0, want_y);
        let y_imag = if want_y { y_imag } else { Vec::new() };
        QpCache {
            target: n,
            omega,
            freq: omega,
            z_factor: z_root,
            dsigma_domega: dsdw,
            static_fallback: false,
            linearized: false,
            lin_shift: 0.0,
            w_imag,
            y_imag,
            m_imag,
            w0,
            y0_static,
            residues,
        }
    }

    // ------------------------------------------------------------------
    // VJP: pull weighted QP gradients back onto (q, J, e, B)
    // ------------------------------------------------------------------

    /// Covectors `(bq, bJ, be, bB)` such that for every state k
    /// `d Omega_k/dR = sum bq[k]:dq + sum bJ[k]:dJ + sum be[k][p] eps1[p]
    ///              + sum bB[k][p] b_x[p]`.
    ///

    /// Covectors `(bq, bJ, be, bB)` such that for every state k
    /// `d Omega_k/dR = sum bq[k]:dq + sum bJ[k]:dJ + sum be[k][p] eps1[p]
    ///              + sum bB[k][p] b_x[p]`.
    ///
    /// Streaming form: `pages` yields successive disjoint runs of QP caches
    /// (`(col0, page)`, `col0` = index of the first cache in the page) and `scr`
    /// accumulates the covectors, so a driver that can rebuild a cache on demand
    /// (the BSE gradient) never has to hold all of them at once.  `weights` is
    /// row-major `[nk, nmo]`.
    ///
    /// Call [`Self::qp_pullback_streaming_finish`] once after the last page.
    pub fn qp_pullback_streaming(
        &self,
        weights: &[f64],
        scr: &mut QpPullbackScratch,
        pages: impl Iterator<Item = (usize, Vec<QpCache>)>,
    ) {
        let nk = scr.nk;
        assert_eq!(
            weights.len(),
            nk * self.nmo,
            "qp_pullback_streaming: weight row length must be nk*nmo"
        );
        for (col0, page) in pages {
            if page.is_empty() {
                break;
            }
            let refs: Vec<&QpCache> = page.iter().collect();
            self.qp_pullback_chunk(&refs, weights, col0, scr, false);
        }
    }

    /// Process one page of caches (see [`Self::qp_pullback_streaming`]).
    /// `omega_mode` selects the **omega-derivative** channel: every kernel is
    /// replaced by its `d/d omega` counterpart and the per-row coefficients
    /// become `weights * Z^2 * lin_shift` (non-zero only for `linearized`
    /// caches).  The caller runs the chunk twice for a linearized cache and
    /// combines the two contractions; see the module plan.
    fn qp_pullback_chunk(
        &self,
        caches: &[&QpCache],
        weights: &[f64],
        col0: usize,
        scr: &mut QpPullbackScratch,
        omega_mode: bool,
    ) {
        let (nk, naux, nmo, nocc, nov) = (scr.nk, scr.naux, scr.nmo, scr.nocc, scr.nov);
        let dbg = |name: &str| std::env::var(name).is_ok();
        let no_be_expl = dbg("REST_GWGRAD_NOBEEXPL");
        let no_exch = dbg("REST_GWGRAD_NOEXCH");
        let no_res = dbg("REST_GWGRAD_NORES");
        let no_imag = dbg("REST_GWGRAD_NOIMAG");
        let no_static = dbg("REST_GWGRAD_NOSTATIC");
        let QpPullbackScratch { bq, bj, be, bb, v_imag, v_static, v_res, z_order, .. } = scr;

        for (jl, cache) in caches.iter().enumerate() {
            let j = col0 + jl;
            let n = cache.target;
            // `c` of this cache in every state (`0` for orbitals without QP
            // weight, e.g. virtuals above the BSE cutoff) — a cache that is
            // inactive in every state is skipped entirely.
            let cs: Vec<f64> = if omega_mode {
                (0..nk)
                    .map(|k| {
                        if cache.linearized {
                            let z = cache.z_factor;
                            weights[k * nmo + j] * z * z * cache.lin_shift
                        } else {
                            0.0
                        }
                    })
                    .collect()
            } else {
                (0..nk).map(|k| weights[k * nmo + j] * cache.z_factor).collect()
            };
            if cs.iter().all(|&c| c == 0.0) {
                continue;
            }
            // Omega-mode `eps1[n]` coefficient of the linearized branch.
            //
            // `dE_lin/dR = Z (C' + d_R Sigma_c) + Z^2 S (d_R d_omega Sigma_c
            //              + Sigma_c'' * d e_n/dR)`,
            // so the omega channel must supply the **total** geometry derivative
            // of `Sigma_c'` at `omega = e_n(R)`: the partial `d_R d_omega Sigma_c`
            // (the omega-differentiated kernels below) plus this explicit chain
            // term, which moves the tangent point `e_n`.  `B` has no
            // omega-dependence, so its `b_x` and exchange channels contribute
            // nothing in omega mode (the standard pass already carries `Z`).
            let dsigma2 = if omega_mode && !no_imag {
                self.sigma_d2_from_parts(cache.freq, &cache.w_imag, &cache.w0)
            } else {
                0.0
            };
            for k in 0..nk {
                if cs[k] != 0.0 {
                    if omega_mode {
                        be[k][n] += cs[k] * dsigma2;
                    } else {
                        // d consts/dR = eps1[n] + b_x[n]
                        be[k][n] += cs[k];
                        bb[k][n] += cs[k];
                    }
                    if cache.static_fallback && !omega_mode {
                        // The static on-shell functional `consts_n + Sigma_c(e_n)`
                        // is evaluated at the **mean-field** frequency, which
                        // moves with the perturbation: its gradient adds
                        // `(d Sigma_c/d omega) d e_n/dR` (and carries no `Z`).
                        be[k][n] += weights[k * nmo + j] * cache.dsigma_domega;
                    }
                }
            }
            // packed-pair offsets of this cache's rows: pair `(n, m)` for every
            // `m`, i.e. the base column of each write below
            let cols: Vec<usize> = (0..nmo)
                .map(|m| packed_pair_index(nmo, n, m) * naux)
                .collect();
            let compact = cache.y_imag.is_empty() && !cache.m_imag.is_empty();
            // static-subtracted functional?  (`w0` is filled exactly when the
            // cache was built by a subtracted engine with a non-empty row)
            let sub = !cache.w0.is_empty() && (compact || !cache.y0_static.is_empty());
            // `y0 = s_metric * m` is the same for every quadrature point, so it
            // is rebuilt once per cache instead of once per (state, frequency).
            // The response buffers are `MatrixFull`s because the outer-product
            // accumulation below reads them as gemm operands.
            let mut y_m = MatrixFull::new([naux, nmo], 0.0);
            let mut y0_m = MatrixFull::new([naux, nmo], 0.0);
            if compact {
                let y0_own = gemm_nn(&self.s_metric, naux, naux, &cache.m_imag, naux, nmo);
                y0_m.data.copy_from_slice(&y0_own);
            }
            // ---- subtracted rows (built once per cache) ----
            // `y0s = (J - Q(0))^-1 q_n` is the response row of the static
            // `W(0)` terms, `y0m = J^-1 q_n` their metric partner (the axis
            // and residue terms keep the raw per-term pullback; the whole
            // static-subtraction bookkeeping folds into their coefficient
            // `staticc`).  Compact caches rebuild `y0s` from `m` through the
            // static low-rank factor; full ones read the stored row.
            let mut y0s_m = MatrixFull::new([naux, nmo], 0.0);
            if sub {
                if compact {
                    let y0s = self.lr_static_row_response(&cache.m_imag);
                    y0s_m.data.copy_from_slice(&y0s);
                } else {
                    y0s_m.data.copy_from_slice(&cache.y0_static);
                    y0_m.data.copy_from_slice(&cache.y_imag[0].2);
                }
            }
            // per-m static coefficient of the subtracted functional:
            // `staticc_m = (1/2 - f_m) + A_m - active_m` with
            // `A_m = 1/pi sum_k w_k K(t_m, u_k)` and `active_m` the signed
            // pole factor, so that the pullback of
            // `sum_wi a_wi W_wi + sum_m staticc_m W(0) + sum_res s pf W_res`
            // is exactly the gradient of `Sigma_sub` (module header).  The
            // `no_*` debug masks keep a masked pullback the exact gradient of
            // the masked functional, as in the full-CD `phase_mask`.
            // `staticc_d` is the omega-derivative `d A_m/d omega` of the same
            // coefficient (its `f_term` and `active` parts do not depend on
            // omega); the omega channel reweights `W(0)` by it.
            let (staticc, staticc_d): (Vec<f64>, Vec<f64>) = if sub {
                let mut active = vec![0.0f64; nmo];
                for r in cache.residues.iter() {
                    active[r.m] += r.pole_weight;
                }
                let mut sc = Vec::with_capacity(nmo);
                let mut sc_d = Vec::with_capacity(nmo);
                for m in 0..nmo {
                    let half_minus_f = if m < nocc { -0.5 } else { 0.5 };
                    let t = cache.freq - self.e[m];
                    let mut a_m = 0.0f64;
                    let mut a_d = 0.0f64;
                    for &(u, wgt) in self.quad.iter() {
                        let __d = t * t + u * u;
                        a_m += wgt / TWO_PI * 2.0 * t / __d;
                        // d/d omega [2t/(t^2+u^2)] = 2(u^2-t^2)/(t^2+u^2)^2
                        a_d += wgt / TWO_PI * 2.0 * (u * u - t * t) / (__d * __d);
                    }
                    let f_term = if no_static { 0.0 } else { half_minus_f };
                    let s_a = if no_imag { 0.0 } else { a_m };
                    let act = if no_res { 0.0 } else { active[m] };
                    sc.push(f_term + s_a - act);
                    sc_d.push(if no_imag { 0.0 } else { a_d });
                }
                (sc, sc_d)
            } else {
                (Vec::new(), Vec::new())
            };
            // `ya = diag(b) y`, `y0a = diag(b) y0` of the current frequency with
            // the **unit** weights `b_m` (see the accumulation note below)
            let mut ya = MatrixFull::new([naux, nmo], 0.0);
            let mut y0a = MatrixFull::new([naux, nmo], 0.0);
            let mut g0 = MatrixFull::new([naux, naux], 0.0);
            let mut v_unit = MatrixFull::new([naux, naux], 0.0);
            let mut sc = vec![0.0f64; naux * nmo];
            // the compact path rebuilds each frequency's response from `m`, which
            // reads the (evictable) low-rank factors
            if compact {
                self.ensure_lr();
            }
            for (wi, &(u, wgt)) in self
                .quad
                .iter()
                .enumerate()
                .take(if no_imag { 0 } else { usize::MAX })
            {
                // `y` and `y0` of this frequency, shared by every state `k`
                let t_resp = Instant::now();
                if compact {
                    let corr = {
                        let lr_imag = self.lr_imag.borrow();
                        lr_imag[wi].correction_from_qt(&cache.m_imag, nmo)
                    };
                    sc = gemm_nn(&self.s_metric, naux, naux, &corr, naux, nmo);
                    for (d, (a, b)) in y_m.data.iter_mut().zip(y0_m.data.iter().zip(sc.iter()))
                    {
                        *d = a + b;
                    }
                } else if !cache.y_imag.is_empty() {
                    let (_u, y, y0) = &cache.y_imag[wi];
                    y_m.data.copy_from_slice(y);
                    y0_m.data.copy_from_slice(y0);
                    for (d, (a, b)) in sc.iter_mut().zip(y.iter().zip(y0.iter())) {
                        *d = a - b;
                    }
                } else {
                    // empty placeholder: no contribution
                    continue;
                }
                if grad_timing() {
                    T_PULL.with(|c| c.borrow_mut()[0] += t_resp.elapsed().as_secs_f64());
                }
                // ---- outer-product accumulation with **unit** weights ----
                // `bj += sum_m a_m (y0_m y0_m^T - y_m y_m^T)` and
                // `V += sum_m a_m y_m y_m^T` (`a_m = c_k b_m`) are linear in the
                // per-state weight `c_k`, so the two Gram matrices are formed
                // once per (cache, frequency) from the `c = 1` coefficients and
                // scaled into every state below.  Writing them as
                // `Y0 diag(b) Y0^T` / `Y diag(b) Y^T` keeps the signed weights
                // inside the second gemm operand, so each needs a single BLAS-3
                // product (the sign-sorted `sqrt(|b|)` form needed four).
                // `V` is exactly the `y y^T` half of `bj`, so `bj` only
                // accumulates the `y0 y0^T` half here and the finish pass
                // subtracts the accumulated `V` matrices.
                let t_gram = Instant::now();
                for m in 0..nmo {
                    let t = cache.freq - self.e[m];
                    let b = {
                    let __d = t * t + u * u;
                    let (__k1, __k2) = if omega_mode {
                        (2.0 * (u * u - t * t) / (__d * __d),
                         -4.0 * t * (3.0 * u * u - t * t) / (__d * __d * __d))
                    } else {
                        (2.0 * t / __d, 2.0 * (u * u - t * t) / (__d * __d))
                    };
                    -wgt / TWO_PI * __k1
                    };
                    let base = m * naux;
                    for r in 0..naux {
                        ya.data[base + r] = b * y_m.data[base + r];
                        y0a.data[base + r] = b * y0_m.data[base + r];
                    }
                }
                g0.data.fill(0.0);
                _dgemm_full(&y0_m, 'N', &y0a, 'T', &mut g0, 1.0, 0.0);
                v_unit.data.fill(0.0);
                _dgemm_full(&y_m, 'N', &ya, 'T', &mut v_unit, 1.0, 0.0);
                if grad_timing() {
                    T_PULL.with(|c| c.borrow_mut()[1] += t_gram.elapsed().as_secs_f64());
                }
                let t_acc = Instant::now();
                for k in 0..nk {
                    let c = cs[k];
                    if c == 0.0 {
                        continue;
                    }
                    // dq covector of the pairs (n, m) and the explicit eps1 term
                    if !no_be_expl {
                        let f = c * wgt / TWO_PI;
                        for m in 0..nmo {
                            let t = cache.freq - self.e[m];
                    let __d = t * t + u * u;
                    let (__k1, __k2) = if omega_mode {
                        (2.0 * (u * u - t * t) / (__d * __d),
                         -4.0 * t * (3.0 * u * u - t * t) / (__d * __d * __d))
                    } else {
                        (2.0 * t / __d, 2.0 * (u * u - t * t) / (__d * __d))
                    };
                            be[k][m] += f * __k2 * cache.w_imag[wi][m];
                        }
                    }
                    // `sc = y - y0` is already the covector's source vector, so
                    // the update reads one stream instead of subtracting two
                    for m in 0..nmo {
                        let t = cache.freq - self.e[m];
                        let a = {
                    let __d = t * t + u * u;
                    let (__k1, __k2) = if omega_mode {
                        (2.0 * (u * u - t * t) / (__d * __d),
                         -4.0 * t * (3.0 * u * u - t * t) / (__d * __d * __d))
                    } else {
                        (2.0 * t / __d, 2.0 * (u * u - t * t) / (__d * __d))
                    };
                        -c * wgt / TWO_PI * __k1
                        };
                        let base = m * naux;
                        let col_m = cols[m];
                        for p in 0..naux {
                            bq[k][col_m + p] += 2.0 * a * sc[base + p];
                        }
                    }
                    let t_bj = Instant::now();
                    for (d, s) in bj[k].iter_mut().zip(g0.data.iter()) {
                        *d += c * s;
                    }
                    for (d, s) in v_imag[k][wi].iter_mut().zip(v_unit.data.iter()) {
                        *d += c * s;
                    }
                    if grad_timing() {
                        T_PULL.with(|c| c.borrow_mut()[2] += t_bj.elapsed().as_secs_f64());
                        T_PULL.with(|c| c.borrow_mut()[3] += t_acc.elapsed().as_secs_f64());
                    }
                }
            }

            for k in 0..nk {
                let c = cs[k];
                if c == 0.0 {
                    continue;
                }
                // ---- static `W(0)` terms of the subtracted functional ----
                // `sum_m c staticc_m W_m(0)` with `W(0) = q^T [(J-Q(0))^-1 -
                // J^-1] q`: the `A(0)^-1` half pulls through the static
                // response row `y0s` (deferred into the `v_static` bucket),
                // the metric half through `y0m = J^-1 q_n`, and the row
                // covector is their difference `2 (y0s - y0m)`.  The kernel
                // sum `A_m(t_m)` inside `staticc` carries the explicit
                // `-A'_m W(0)` eps1 term of the axis kernel.
                // The static `W(0)` bookkeeping carries its own omega-dependence
                // (`d A_m/d omega`), so the omega channel reweights the block by
                // `staticc_d[m]` instead of `staticc[m]` (and differentiates the
                // explicit kernel below).
                let static_w: &[f64] = if omega_mode { &staticc_d } else { &staticc };
                if sub && static_w.iter().any(|&v| v != 0.0) {
                    let mut pre = vec![0.0f64; naux * nmo];
                    let mut prm = vec![0.0f64; naux * nmo];
                    for m in 0..nmo {
                        let csm = c * static_w[m];
                        let base = m * naux;
                        let col_m = cols[m];
                        for p in 0..naux {
                            pre[base + p] = csm * y0s_m.data[base + p];
                            prm[base + p] = csm * y0_m.data[base + p];
                            bq[k][col_m + p] += 2.0 * (pre[base + p] - prm[base + p]);
                        }
                    }
                    let prem = to_mat(&pre, naux, nmo);
                    let prmm = to_mat(&prm, naux, nmo);
                    let ysm = to_mat(&y0s_m.data, naux, nmo);
                    let ymm = to_mat(&y0_m.data, naux, nmo);
                    let mut vr = MatrixFull::new([naux, naux], 0.0);
                    _dgemm_full(&prem, 'N', &ysm, 'T', &mut vr, 1.0, 0.0);
                    let mut vm = MatrixFull::new([naux, naux], 0.0);
                    _dgemm_full(&prmm, 'N', &ymm, 'T', &mut vm, 1.0, 0.0);
                    for (d, (a, b)) in bj[k].iter_mut().zip(vm.data.iter().zip(vr.data.iter()))
                    {
                        *d += a - b;
                    }
                    for (d, s) in v_static[k].iter_mut().zip(vr.data.iter()) {
                        *d += s;
                    }
                    if !no_be_expl {
                        for m in 0..nmo {
                            let t = cache.freq - self.e[m];
                            let mut ap = 0.0f64;
                            for &(u, wgt) in self.quad.iter() {
                                let __d = t * t + u * u;
                                // standard: `d/dt` of the `A_m` kernel
                                // `2t/(t^2+u^2)`; omega: its second derivative
                                let __k2 = if omega_mode {
                                    -4.0 * t * (3.0 * u * u - t * t) / (__d * __d * __d)
                                } else {
                                    2.0 * (u * u - t * t) / (__d * __d)
                                };
                                ap += wgt / TWO_PI * __k2;
                            }
                            be[k][m] -= c * ap * cache.w0[m];
                        }
                    }
                }
                // ---- residues (see `Residue`) ----
                for r in cache
                    .residues
                    .iter()
                    .take(if no_res { 0 } else { usize::MAX })
                {
                    // The factor frequency follows the pole (`z = |e_m - omega|`)
                    // in the exact protocol and the interpolation weight follows
                    // it in the linear one; `weight_slope` is the corresponding
                    // `dW/de_m` divided by `zsign` and is zero for the nearest
                    // (piecewise-constant) rule.
                    let zsign = if self.e[r.m] - cache.freq >= 0.0 { 1.0 } else { -1.0 };
                    if omega_mode {
                        // `d W_m/d omega` of the entry weights of the **linear**
                        // rule: the grid nodes do not move with omega, only the
                        // bracketing weight `t`, so
                        // `d split_lo/d omega = +zsign/(z_hi-z_lo)` and
                        // `d split_hi/d omega = -zsign/(z_hi-z_lo)`, times the
                        // pole weight.  `sum_i dh_i W_i` reproduces
                        // `sigma_d_from_parts`'s `(-zsign) weight_slope`.
                        // The exact protocol would need `d^2 W/dz^2` and is
                        // rejected loudly below; the single-entry rules have
                        // `dh = 0`.
                        assert!(
                            !self.config.exact_residue_z,
                            "gw_grad: the omega-derivative pullback of a linearized \
                             QP cache does not implement the exact-residue protocol \
                             (`exact_residue_z = true`)"
                        );
                    } else if r.weight_slope != 0.0 {
                        be[k][r.m] += c * r.weight_slope * zsign;
                    }
                    // Movement of the real-axis grid nodes with the reference:
                    // `d de_max/dR = 1.05 * sign * (d e_p - d e_m*)/dR`, so the
                    // node-frequency channel lands on the two scan indices.
                    // `dW/dz` is rebuilt here (once per cache and entry) rather
                    // than stored per Newton step.  The exact-residue protocol
                    // never reads the grid.
                    if !self.config.exact_residue_z {
                        if let Some(mv) = self.z_grid_mover.as_ref() {
                            let ent = &r.entries;
                            let grid_slope = if ent.len() == 1 {
                                if omega_mode {
                                    // the clamped/nearest node does not move
                                    // with omega
                                    0.0
                                } else {
                                    r.pole_weight
                                        * self.residue_dwdz(&ent[0].y, ent[0].z)
                                        * ent[0].node_sens
                                }
                            } else {
                                let (elo, ehi) = (&ent[0], &ent[1]);
                                let t = r.interp_t;
                                let (dlo, dhi) = (
                                    self.residue_dwdz(&elo.y, elo.z),
                                    self.residue_dwdz(&ehi.y, ehi.z),
                                );
                                if omega_mode {
                                    // `d/dt` of the standard bracket times
                                    // `dt/d omega = -zsign/(z_hi-z_lo)`
                                    let dz = ehi.z - elo.z;
                                    if dz.abs() < 1.0e-30 {
                                        0.0
                                    } else {
                                        r.pole_weight
                                            * (dhi * ehi.node_sens
                                                - dlo * elo.node_sens
                                                - r.secant * (ehi.node_sens - elo.node_sens))
                                            * (-zsign / dz)
                                    }
                                } else {
                                    r.pole_weight
                                        * ((1.0 - t) * dlo * elo.node_sens
                                            + t * dhi * ehi.node_sens
                                            - r.secant
                                                * ((1.0 - t) * elo.node_sens + t * ehi.node_sens))
                                }
                            };
                            if grid_slope != 0.0 {
                                // `de_max` is a **max** over the scan, so on a
                                // symmetry-degenerate tie it is not
                                // differentiable (its central difference is the
                                // average of the tied branches).  The
                                // basis-independent choice is the tie average,
                                // which is exact for the 1- and 2-fold
                                // degeneracies that dominate; a 3-fold tie
                                // leaves a residual of order the shell's
                                // `eps1` spread times the node sensitivity.
                                let g = c * mv.scale * mv.sign * grid_slope;
                                let wp = g / mv.p_set.len() as f64;
                                let wm = g / mv.m_set.len() as f64;
                                for &p in mv.p_set.iter() {
                                    be[k][p] += wp;
                                }
                                for &m in mv.m_set.iter() {
                                    be[k][m] -= wm;
                                }
                            }
                        }
                    }
                    // dq / dJ covectors and the V bucket of pair (n, m); a linear
                    // pole contributes one entry per bracketing grid node.
                    // In omega mode the entry weight is replaced by its
                    // omega-derivative (see above); the single-entry rules are
                    // omega-independent and contribute nothing.
                    let col_m = cols[r.m];
                    let entries_omega: [f64; 2] = if omega_mode && r.entries.len() == 2 {
                        let dz = r.entries[1].z - r.entries[0].z;
                        if dz.abs() < 1.0e-30 {
                            [0.0, 0.0]
                        } else {
                            let inv = r.pole_weight * zsign / dz;
                            [inv, -inv]
                        }
                    } else {
                        [0.0, 0.0]
                    };
                    for (ei, e) in r.entries.iter().enumerate() {
                        if e.y.is_empty() {
                            continue;
                        }
                        let a = if omega_mode {
                            c * if ei < 2 { entries_omega[ei] } else { 0.0 }
                        } else {
                            c * e.weight
                        };
                        if a == 0.0 {
                            continue;
                        }
                        for p in 0..naux {
                            bq[k][col_m + p] += 2.0 * a * (e.y[p] - e.y0[p]);
                        }
                        for p in 0..naux {
                            for q in 0..naux {
                                bj[k][p + q * naux] += a * (e.y0[p] * e.y0[q] - e.y[p] * e.y[q]);
                            }
                        }
                        let idx =
                            match z_order[k].iter().position(|&z| (z - e.z).abs() < 1.0e-14) {
                                Some(i) => i,
                                None => {
                                    z_order[k].push(e.z);
                                    v_res[k].push(vec![0.0f64; naux * naux]);
                                    z_order[k].len() - 1
                                }
                            };
                        let v = &mut v_res[k][idx];
                        for p in 0..naux {
                            let yp = e.y[p];
                            for q in 0..naux {
                                v[p + q * naux] += a * yp * e.y[q];
                            }
                        }
                    }
                }

                // ---- static term B_n = (1-alpha) Sigma_x,nn - v_xc,nn ----
                // REST `consts = eps + (1-alpha) Sigma_x,nn - v_xc,nn` with
                // `Sigma_x,nn = -sum_i (n i|n i) = -sum_i q_ni^T J^{-1} q_ni`,
                // so the derivative contributes
                //   -(1-alpha) [2 sum_i g_ni^T dq_ni - sum_i g_ni^T dJ g_ni],
                // g_ni = J^{-1} q_ni.  The v_xc relaxation term is not
                // implemented (references with a nonzero v_xc are rejected
                // in `canonical_response`).
                if (1.0 - self.fock_hyb).abs() > 1.0e-14 && !no_exch && !omega_mode {
                    for i in 0..nocc {
                        let qni = self.with_q(|q| self.gather_pair(q, n, i));
                        let g_ni = self.metric_solve(&qni, 1);
                        let col_i = packed_pair_index(nmo, n, i) * naux;
                        for p in 0..naux {
                            bq[k][col_i + p] -= 2.0 * c * (1.0 - self.fock_hyb) * g_ni[p];
                        }
                        for p in 0..naux {
                            for q in 0..naux {
                                bj[k][p + q * naux] +=
                                    c * (1.0 - self.fock_hyb) * g_ni[p] * g_ni[q];
                            }
                        }
                    }
                }
            }
        }
    }

    /// Close a streaming pullback: solve the deferred polarizability matrices,
    /// symmetrise the covectors and hand them back.
    pub fn qp_pullback_streaming_finish(
        &self,
        scr: QpPullbackScratch,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let nk = scr.nk;
        let (naux, nmo) = (scr.naux, scr.nmo);
        let dbg = |name: &str| std::env::var(name).is_ok();
        let no_res = dbg("REST_GWGRAD_NORES");
        let no_imag = dbg("REST_GWGRAD_NOIMAG");
        let QpPullbackScratch {
            mut bq,
            mut bj,
            mut be,
            bb,
            v_imag,
            v_static,
            v_res,
            z_order,
            imag_factors,
            ..
        } = scr;

        // ---- deferred polarizability pullbacks ----
        let t_polar = Instant::now();
        if grad_timing() {
            let [t_r, t_g, t_b, t_a] = T_PULL.with(|c| *c.borrow());
            eprintln!(
                "[gw_grad timing]   qp_pullback: response {:.2}s  gram {:.2}s  bj/v accum {:.2}s  bq/be+rest {:.2}s",
                t_r, t_g, t_b, (t_a - t_b).max(0.0)
            );
            T_PULL.with(|c| c.borrow_mut().fill(0.0));
        }
        for k in 0..nk {
            for (wi, _) in self.quad.iter().enumerate().take(if no_imag { 0 } else { usize::MAX }) {
                let (d, dd) = &imag_factors[wi];
                self.polar_pullback(&mut bq, &mut be, k, &v_imag[k][wi], d, dd, true);
            }
            if !no_res {
                for (zi, &z) in z_order[k].iter().enumerate() {
                    let (d, dd) = real_response_and_deriv(&self.de_ia, z, self.config.eta);
                    self.polar_pullback(&mut bq, &mut be, k, &v_res[k][zi], &d, &dd, true);
                }
            }
            // The static bucket is processed whenever it is nonzero: with the
            // `REST_GWGRAD_NOSTATIC` mask the coefficient keeps the axis and
            // residue `W(0)` compensations (`A_m` and `active_m`), so only the
            // fully masked combination leaves it empty.  `Q(0)` carries the
            // same real-axis Lorentzian diagonal at `z = 0` as the production
            // grid's first node.
            if v_static[k].iter().any(|&v| v != 0.0) {
                let (d, dd) = real_response_and_deriv(&self.de_ia, 0.0, self.config.eta);
                self.polar_pullback(&mut bq, &mut be, k, &v_static[k], &d, &dd, true);
            }
        }

        // The imaginary-axis outer products accumulated `bj += sum a_m y0 y0^T`
        // only; their `y y^T` half is exactly the dQ matrix, which the
        // frequency buckets `v_imag[k][wi]` already hold, so the subtraction is
        // one pass over the (small) `naux^2` covectors instead of a second Gram
        // product per cache and frequency.
        for k in 0..nk {
            for v in v_imag[k].iter() {
                for (d, s) in bj[k].iter_mut().zip(v.iter()) {
                    *d -= s;
                }
            }
        }

        // The packed pair layout accumulated **both** directions of every pair
        // into one slot (the pullback visits each row of each cache) while the
        // diagonal carries a single contribution.  The minimum-norm symmetric
        // representative the folded consumers expect stores the *average* of
        // the two directions, so halve the off-diagonal pairs — exactly the
        // values `symmetrise_pairs` used to write into the full buffer.
        for k in 0..nk {
            let mut tp = 0usize;
            for col in 0..nmo {
                for row in col..nmo {
                    if row != col {
                        let s = &mut bq[k][tp * naux..(tp + 1) * naux];
                        for v in s.iter_mut() {
                            *v *= 0.5;
                        }
                    }
                    tp += 1;
                }
            }
            symmetrise_square(&mut bj[k], naux);
        }
        if grad_timing() {
            eprintln!(
                "[gw_grad timing]   qp_pullback: deferred polar pullbacks {:8.2}s",
                t_polar.elapsed().as_secs_f64()
            );
        }
        (bq, bj, be, bb)
    }

    /// Fresh streaming scratch ([`Self::qp_pullback_streaming`]).
    pub fn qp_pullback_scratch(&self, nk: usize) -> QpPullbackScratch {
        QpPullbackScratch::new(self, nk)
    }

    /// Process one page of caches in an open streaming pullback.
    pub fn qp_pullback_chunk_public(
        &self,
        caches: &[&QpCache],
        weights: &[f64],
        col0: usize,
        scr: &mut QpPullbackScratch,
    ) {
        self.qp_pullback_chunk(caches, weights, col0, scr, false);
    }

    /// Omega-derivative counterpart of [`Self::qp_pullback_chunk_public`].
    /// Accumulates `d/d omega` of the same covectors, weighted by
    /// `weights * Z^2 * lin_shift` and restricted to `linearized` caches.
    pub fn qp_pullback_chunk_omega_public(
        &self,
        caches: &[&QpCache],
        weights: &[f64],
        col0: usize,
        scr: &mut QpPullbackScratch,
    ) {
        self.qp_pullback_chunk(caches, weights, col0, scr, true);
    }

    /// Bytes of one fully built [`QpCache`] of this engine.
    ///
    /// With `compact_qp_cache` a cache keeps a single `[naux, nmo]` metric
    /// intermediate instead of the `nquad` pairs of `[naux, nmo]` response
    /// vectors, so the page size of the streaming pullback is estimated from the
    /// compact footprint (plus the `w_imag` table and a residue budget).
    pub fn qp_cache_estimate_bytes(&self) -> usize {
        if self.config.compact_qp_cache {
            let nres = 4.min(self.nocc + self.nmo);
            self.naux * self.nmo * 8
                + self.quad.len() * self.nmo * 8
                + nres * 2 * self.naux * 8
        } else {
            qp_cache_bytes_estimate(self.naux, self.nmo, self.quad.len())
        }
    }

    /// Non-streaming [`Self::qp_pullback_streaming`]: every cache resident.
    ///
    /// `weights[k][j]` is the derivative of `Omega_k` with respect to the QP
    /// energy of cache `j`; the `Z` renormalisation of each cache is applied
    /// here.
    pub fn qp_pullback(
        &self,
        caches: &[&QpCache],
        weights: &[Vec<f64>],
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let nk = weights.len();
        for w in weights {
            assert_eq!(w.len(), caches.len(), "qp_pullback: weight row length mismatch");
        }
        let flat: Vec<f64> = weights.iter().flat_map(|w| w.iter().copied()).collect();
        let t_cache = Instant::now();
        let mut scr = QpPullbackScratch::new(self, nk);
        let owned: Vec<QpCache> = caches
            .iter()
            .map(|c| QpCache {
                target: c.target,
                omega: c.omega,
                freq: c.freq,
                z_factor: c.z_factor,
                dsigma_domega: c.dsigma_domega,
                static_fallback: c.static_fallback,
                linearized: c.linearized,
                lin_shift: c.lin_shift,
                w_imag: c.w_imag.clone(),
                y_imag: c.y_imag.clone(),
                m_imag: c.m_imag.clone(),
                w0: c.w0.clone(),
                y0_static: c.y0_static.clone(),
                residues: c.residues.clone(),
            })
            .collect();
        let refs: Vec<&QpCache> = owned.iter().collect();
        self.qp_pullback_chunk(&refs, &flat, 0, &mut scr, false);
        // Omega-derivative channel of the linearized caches: run the same
        // scratch a second time (its `cs` carries `weights * Z^2 * lin_shift`)
        // instead of keeping a second covector set resident, so the peak
        // memory of a linearized pullback equals the standard one.
        if refs.iter().any(|c| c.linearized) {
            self.qp_pullback_chunk(&refs, &flat, 0, &mut scr, true);
        }
        if grad_timing() {
            eprintln!(
                "[gw_grad timing]   qp_pullback: cache loop {:8.2}s",
                t_cache.elapsed().as_secs_f64()
            );
        }
        self.qp_pullback_streaming_finish(scr)
    }

    /// Add the pullback of `Tr[V dQ]` to `bq` and `be`, with
    /// `Q = sum_ia q_ia d_ia q_ia^T` and `deps_ia = deps_a - deps_i`.
    /// Deferred polarizability pullback.  `packed` selects the pair layout of
    /// `bq`: the LR streaming pullback accumulates the packed lower triangle,
    /// the CD-GW pullback the full pair matrix.
    #[allow(clippy::too_many_arguments)]
    fn polar_pullback(
        &self,
        bq: &mut [Vec<f64>],
        be: &mut [Vec<f64>],
        k: usize,
        v: &[f64],
        d: &[f64],
        dd: &[f64],
        packed: bool,
    ) {
        let naux = self.naux;
        let nov = self.de_ia.len();
        let vm = to_mat(v, naux, naux);
        let qiam = to_mat(&self.qia, naux, nov);
        let mut vq = MatrixFull::new([naux, nov], 0.0);
        _dgemm_full(&vm, 'N', &qiam, 'N', &mut vq, 1.0, 0.0);
        for col in 0..nov {
            let i = col % self.nocc;
            let a = self.nocc + col / self.nocc;
            let col_pair = if packed {
                packed_pair_index(self.nmo, i, a)
            } else {
                self.pair_col(i, a)
            } * naux;
            let mut bg = 0.0f64;
            for p in 0..naux {
                let vqv = vq.data[p + col * naux];
                bq[k][col_pair + p] += 2.0 * d[col] * vqv;
                bg += self.qia[p + col * naux] * vqv * dd[col];
            }
            be[k][i] -= bg;
            be[k][a] += bg;
        }
    }

    /// Contract the covectors with one nuclear perturbation: returns
    /// `sum_k [ bq_k : qx + bJ_k : dJ + be_k . eps1 + bB_k . b_x ]`.
    pub fn contract_perturbation(
        &self,
        bq: &[Vec<f64>],
        bj: &[Vec<f64>],
        be: &[Vec<f64>],
        bb: &[Vec<f64>],
        qx: &[f64],
        dj: &[f64],
        eps1: &[f64],
        b_x: &[f64],
    ) -> Vec<f64> {
        (0..bq.len())
            .map(|k| {
                bq[k].iter().zip(qx.iter()).map(|(a, b)| a * b).sum::<f64>()
                    + bj[k].iter().zip(dj.iter()).map(|(a, b)| a * b).sum::<f64>()
                    + be[k].iter().zip(eps1.iter()).map(|(a, b)| a * b).sum::<f64>()
                    + bb[k].iter().zip(b_x.iter()).map(|(a, b)| a * b).sum::<f64>()
            })
            .collect()
    }

    /// Derivative of the raw three-center MO integrals for one perturbation:
    /// `qx = C^T dI C + U^T q + q U` (pair-symmetrised), with `dI` supplied
    /// as a full [naux, nao*nao] buffer.
    pub fn qx_from_u(&self, di: &[f64], u: &[f64]) -> Vec<f64> {
        let n2 = self.nao * self.nao;
        self.qx_from_u_getter(&|p, out| out.copy_from_slice(&di[p * n2..(p + 1) * n2]), u)
    }

    /// Same contraction with `dI` assembled block-by-block from one atom's
    /// sliced derivative integrals — the full `dI` never coexists with `qx`.
    pub fn qx_from_u_blocks(
        &self,
        blocks: &AtomDerivBlocks,
        comp: usize,
        u: &[f64],
    ) -> Vec<f64> {
        self.qx_from_u_getter(&|p, out| self.raw.d_i_block(blocks, comp, p, out), u)
    }

    fn qx_from_u_getter(&self, di_getter: &dyn Fn(usize, &mut [f64]), u: &[f64]) -> Vec<f64> {
        let nao = self.nao;
        let naux = self.naux;
        let nmo = self.nmo;
        let n2 = nao * nao;
        let npair = nmo * nmo;
        // one borrow of `q` for the whole routine (`with_q` would rebuild it if a previous
        // gradient released it)
        self.ensure_q();
        let q_all = self.q.borrow();
        let mut qx = vec![0.0f64; naux * npair];

        // ---- dI part: qx[P, p + q*nmo] = (C^T dI_p C)[p, q], per P gemms ----
        {
            let mut ip = MatrixFull::new([nao, nao], 0.0);
            let mut tmp = MatrixFull::new([nao, nmo], 0.0);
            let mut sl = MatrixFull::new([nmo, nmo], 0.0);
            for p in 0..naux {
                di_getter(p, &mut ip.data);
                _dgemm_full(&ip, 'N', &self.c_mo, 'N', &mut tmp, 1.0, 0.0);
                _dgemm_full(&self.c_mo, 'T', &tmp, 'N', &mut sl, 1.0, 0.0);
                for col in 0..npair {
                    qx[p + col * naux] = sl.data[col];
                }
            }
        }
        // ---- U part: U^T q + q U.  q is pair-symmetric, so
        //      `(qU)[p,q] = (U^T q)[q,p]`; the pair-major [nmo, nmo*naux]
        //      view of q is contracted with U^T in auxiliary blocks (a
        //      full-length t1 buffer would double the qx memory footprint)
        //      and the transposed pair is added before symmetrising; qx
        //      stays in its native [naux, nmo*nmo] col-major layout.  The
        //      aux block p occupies the contiguous column range
        //      [p*nmo, (p+1)*nmo) of the pair-major view. ----
        let um = to_mat(u, nmo, nmo);
        let chunk_aux = (4_194_304 / (nmo * nmo)).clamp(1, naux);
        let resident_pairmajor = !self.q_pairmajor.is_empty();
        let mut p0 = 0;
        while p0 < naux {
            let n_chunk = (naux - p0).min(chunk_aux);
            let nc = n_chunk * nmo;
            // build the [nmo, n_chunk*nmo] pair-major view of the aux block:
            // element (pi, pj_m + p_l*nmo) = q[pair*naux + p0 + p_l]
            let q2c: Vec<f64> = if resident_pairmajor {
                self.q_pairmajor[p0 * npair..(p0 + n_chunk) * npair].to_vec()
            } else {
                let mut v = vec![0.0f64; nmo * nc];
                for pair in 0..npair {
                    let pi = pair % nmo;
                    let pj_m = pair / nmo;
                    let src = pair * naux + p0;
                    for (p_l, &val) in q_all[src..src + n_chunk].iter().enumerate() {
                        v[pi + (pj_m + p_l * nmo) * nmo] = val;
                    }
                }
                v
            };
            let q2cm = to_mat(&q2c, nmo, nc);
            let mut t1v = MatrixFull::new([nmo, nc], 0.0);
            _dgemm_full(&um, 'T', &q2cm, 'N', &mut t1v, 1.0, 0.0);
            for p in p0..p0 + n_chunk {
                let base = (p - p0) * npair;
                for pj in 0..npair {
                    let pj_swap = (pj % nmo) * nmo + pj / nmo;
                    qx[p + pj * naux] += t1v.data[base + pj] + t1v.data[base + pj_swap];
                }
                // symmetrise the pair block
                for pi in 0..nmo {
                    for pj in (pi + 1)..nmo {
                        let a = pi + pj * nmo;
                        let b = pj + pi * nmo;
                        let s = 0.5 * (qx[p + a * naux] + qx[p + b * naux]);
                        qx[p + a * naux] = s;
                        qx[p + b * naux] = s;
                    }
                }
            }
            p0 += n_chunk;
        }
        qx
    }
}

// ---------------------------------------------------------------------------
// Folded ("AO-basis") evaluation of the qx channel
// ---------------------------------------------------------------------------
//
// The gradient couples the perturbation-independent covector `bq` with the
// perturbation-dependent derivative of the raw three-center MO integrals
//
//     sum_{P,p,q} bq[P,p,q] qx[P,p,q],
//     qx = C^T (dI/dR) C + U^T q + q U .
//
// Materialising `qx` costs two MO transformations of `dI` per perturbation,
// `O(naux nao^2 nmo)` + `O(naux nao nmo^2)`, which summed over the `3 natm`
// perturbations is one power of `N` more than necessary: the mixing of the AO
// and MO bases does not depend on the perturbation at all.
//
// Folding the covector once per geometry removes it:
//
//   * dI channel.  With `B[P] = C bq_P C^T` in the AO basis,
//         sum_{P,pq} bq[P,pq] (C^T dI_P C)[pq]
//       = sum_P sum_{mu,nu} B[P,mu,nu] dI[P,mu,nu] ,
//     an `O(nao^2)` dot per auxiliary function, i.e. `O(naux nao^2)` per
//     perturbation, evaluated from the *same* streamed `[nao,nao]` dI slices
//     (`d_i_block`) that the old route fed into the gemms.
//
//   * orbital-rotation channel.  With `q` and `bq` pair-symmetric,
//         sum_{P,pq} bq[P,pq] (U^T q + q U)[P,pq] = 2 Tr[U^T W],
//         W[r,p] = sum_{P,q} q[P,r,q] bq[P,p,q] ,
//     an `O(nmo^2)` dot per perturbation for a once-per-geometry `O(naux nmo^3)`
//     preparation.
//
// Total: `O(naux nao^3)` once + `O(3 natm * naux nao^2)` instead of
// `O(3 natm * naux nao^3)`.

/// Perturbation-independent folded form of one `bq` covector (one weight row).
///
/// See the section header above for the derivation; both fields are built by
/// [`GwGradEngine::fold_covector`].
pub struct FoldedCovector {
    /// `B[P, mu + nu*nao] = sum_pq C[mu,p] bq[P,p,q] C[nu,q]`, col-major
    /// `[nao, nao]` per auxiliary function `P`.
    pub b_ao: Vec<f64>,
    /// `W[r + p*nmo] = sum_{P,q} q[P,r,q] bq[P,p,q]`, col-major `[nmo, nmo]`.
    pub w_rot: Vec<f64>,
}

impl GwGradEngine<'_> {
    /// Fold one weight row's `bq` covector into `(B, W)` (see
    /// [`FoldedCovector`]).  Cost `O(naux nao^2 nmo + naux nmo^3)`, paid once
    /// per geometry instead of once per perturbation.
    pub fn fold_covector(&self, bq_k: &[f64]) -> FoldedCovector {
        let nao = self.nao;
        let nmo = self.nmo;
        let naux = self.naux;
        let n2 = nao * nao;
        let npair = nmo * nmo;
        assert_eq!(bq_k.len(), naux * npair, "fold_covector: bq size mismatch");

        self.ensure_q();
        let q_all = self.q.borrow();
        let mut b_ao = vec![0.0f64; naux * n2];
        let mut x = MatrixFull::new([nao, nmo], 0.0);
        let mut bp = MatrixFull::new([nao, nao], 0.0);
        let mut qm = MatrixFull::new([nmo, nmo], 0.0);
        let mut bqm = MatrixFull::new([nmo, nmo], 0.0);
        let mut wm = MatrixFull::new([nmo, nmo], 0.0);
        for p in 0..naux {
            // `q` and `bq` are stored pair-slowest / aux-fastest
            // (`[pair*naux + P]`); gather the auxiliary function's
            // `[nmo, nmo]` block into the contiguous gemm buffers.
            for pair in 0..npair {
                let src = pair * naux + p;
                qm.data[pair] = q_all[src];
                bqm.data[pair] = bq_k[src];
            }
            // W += q_P bq_P  (both pair-symmetric, so this is
            // W[r,p] = sum_q q[P,r,q] bq[P,p,q])
            _dgemm_full(&qm, 'N', &bqm, 'N', &mut wm, 1.0, 1.0);
            // B_P = C bq_P C^T
            _dgemm_full(&self.c_mo, 'N', &bqm, 'N', &mut x, 1.0, 0.0);
            _dgemm_full(&x, 'N', &self.c_mo, 'T', &mut bp, 1.0, 0.0);
            b_ao[p * n2..(p + 1) * n2].copy_from_slice(&bp.data);
        }
        FoldedCovector { b_ao, w_rot: wm.data }
    }

    /// [`Self::fold_covector_packed`] without reading `q`.
    ///
    /// The orbital-rotation covector is `W = sum_P q_P bq_P = C^T X` with `X = sum_P I_P (C bq_P)`,
    /// which only needs the **raw** three-centre integrals (streamed auxiliary-atom block by block)
    /// instead of the transformed `q` (`naux*nmo^2` doubles, 874 MB at naphthalene/def2-TZVP).
    /// The adjoint gradient uses this variant so that `q` can be released before the fold; `B` is
    /// built with exactly the same operations as in [`Self::fold_covector`] (`Z_P = C bq_P`,
    /// `B_P = Z_P C^T`), so `di_cov` is unchanged, while `W` differs only in the order of the
    /// (mathematically identical) summations.
    pub fn fold_covector_packed_streamed(&self, bq_tp: &[f64]) -> FoldedCovector {
        let nao = self.nao;
        let nmo = self.nmo;
        let naux = self.naux;
        let n2 = nao * nao;
        let nmo_tp = nmo * (nmo + 1) / 2;
        assert_eq!(bq_tp.len(), naux * nmo_tp, "fold_covector_packed_streamed: bq size mismatch");

        let mut b_ao = vec![0.0f64; naux * n2];
        let mut bqm = MatrixFull::new([nmo, nmo], 0.0);
        let mut z = MatrixFull::new([nao, nmo], 0.0);
        let mut bp = MatrixFull::new([nao, nao], 0.0);
        let mut x_acc = MatrixFull::new([nao, nmo], 0.0);
        // one streaming pass over the auxiliary-atom integral blocks serves every `P`
        self.raw.for_each_i3_atom_block(|p0, nb, blk| {
            for pl in 0..nb {
                let p = p0 + pl;
                // rebuild the symmetric `bq` plane from its lower triangle (`bq` was symmetrised)
                let mut tp = 0usize;
                for col in 0..nmo {
                    for row in col..nmo {
                        let val = bq_tp[tp * naux + p];
                        bqm.data[row + col * nmo] = val;
                        bqm.data[col + row * nmo] = val;
                        tp += 1;
                    }
                }
                _dgemm_full(&self.c_mo, 'N', &bqm, 'N', &mut z, 1.0, 0.0); // Z_P = C bq_P
                let ip = to_mat(&blk[pl * n2..(pl + 1) * n2], nao, nao);
                _dgemm_full(&ip, 'N', &z, 'N', &mut x_acc, 1.0, 1.0); // X += I_P Z_P
                _dgemm_full(&z, 'N', &self.c_mo, 'T', &mut bp, 1.0, 0.0); // B_P = Z_P C^T
                b_ao[p * n2..(p + 1) * n2].copy_from_slice(&bp.data);
            }
        });
        let mut wm = MatrixFull::new([nmo, nmo], 0.0);
        _dgemm_full(&self.c_mo, 'T', &x_acc, 'N', &mut wm, 1.0, 0.0); // W = C^T X
        FoldedCovector { b_ao, w_rot: wm.data }
    }

    /// [`Self::fold_covector`] for a **pair-packed** `bq` (`[nmo*(nmo+1)/2, naux]`, lower triangle
    /// of the pair-symmetric covector, aux-fastest), as produced by [`pack_pairs_in_place`].
    ///
    /// The symmetric plane is rebuilt from the packed values, so the fold is exact and identical to
    /// the full-buffer version while half the covector memory is resident.
    pub fn fold_covector_packed(&self, bq_tp: &[f64]) -> FoldedCovector {
        let nao = self.nao;
        let nmo = self.nmo;
        let naux = self.naux;
        let n2 = nao * nao;
        let nmo_tp = nmo * (nmo + 1) / 2;
        assert_eq!(bq_tp.len(), naux * nmo_tp, "fold_covector_packed: bq size mismatch");

        self.ensure_q();
        let q_all = self.q.borrow();
        let mut b_ao = vec![0.0f64; naux * n2];
        let mut x = MatrixFull::new([nao, nmo], 0.0);
        let mut bp = MatrixFull::new([nao, nao], 0.0);
        let mut qm = MatrixFull::new([nmo, nmo], 0.0);
        let mut bqm = MatrixFull::new([nmo, nmo], 0.0);
        let mut wm = MatrixFull::new([nmo, nmo], 0.0);
        for p in 0..naux {
            for pair in 0..nmo * nmo {
                qm.data[pair] = q_all[pair * naux + p];
            }
            // rebuild the symmetric `bq` plane from its lower triangle (`bq` has been symmetrised,
            // so the mirrored entries are the same floating-point number)
            let mut tp = 0usize;
            for col in 0..nmo {
                for row in col..nmo {
                    let val = bq_tp[tp * naux + p];
                    bqm.data[row + col * nmo] = val;
                    bqm.data[col + row * nmo] = val;
                    tp += 1;
                }
            }
            // W += q_P bq_P, B_P = C bq_P C^T (same order as `fold_covector`)
            _dgemm_full(&qm, 'N', &bqm, 'N', &mut wm, 1.0, 1.0);
            _dgemm_full(&self.c_mo, 'N', &bqm, 'N', &mut x, 1.0, 0.0);
            _dgemm_full(&x, 'N', &self.c_mo, 'T', &mut bp, 1.0, 0.0);
            b_ao[p * n2..(p + 1) * n2].copy_from_slice(&bp.data);
        }
        FoldedCovector { b_ao, w_rot: wm.data }
    }

    /// Scalar produced by the `qx` channel of one perturbation for one weight
    /// row, evaluated in the folded form — identical (up to round-off) to
    /// `contract_perturbation(bq_k.., .., &qx_from_u_blocks(blocks, comp, u), ..)`
    /// for that row.
    ///
    /// `dj`, `eps1`, `b_x` and the `bJ`/`be`/`bB` covectors are contracted
    /// exactly as in [`GwGradEngine::contract_perturbation`].
    #[allow(clippy::too_many_arguments)]
    pub fn folded_perturbation_gradient(
        &self,
        fold: &FoldedCovector,
        blocks: &AtomDerivBlocks,
        comp: usize,
        u: &[f64],
        bj_k: &[f64],
        be_k: &[f64],
        bb_k: &[f64],
        dj: &[f64],
        eps1: &[f64],
        b_x: &[f64],
    ) -> f64 {
        let n2 = self.nao * self.nao;
        // dI channel: sum_P <B[P], dI[P]>, streamed slice by slice
        let mut ip = vec![0.0f64; n2];
        let mut g = 0.0f64;
        for p in 0..self.naux {
            self.raw.d_i_block(blocks, comp, p, &mut ip);
            let b = &fold.b_ao[p * n2..(p + 1) * n2];
            g += b.iter().zip(ip.iter()).map(|(x, y)| x * y).sum::<f64>();
        }
        // orbital-rotation channel: sum bq:(U^T q + q U) = 2 Tr[U^T W]
        g += 2.0
            * u.iter()
                .zip(fold.w_rot.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
        g += bj_k.iter().zip(dj.iter()).map(|(a, b)| a * b).sum::<f64>();
        g += be_k.iter().zip(eps1.iter()).map(|(a, b)| a * b).sum::<f64>();
        g += bb_k.iter().zip(b_x.iter()).map(|(a, b)| a * b).sum::<f64>();
        g
    }

    /// `qx` channel of one perturbation for **several** folded covectors at once.
    ///
    /// Same contraction as the `dI`/`U` part of
    /// [`GwGradEngine::folded_perturbation_gradient`], but the atom's sliced
    /// derivative integrals are summed **directly** against the covectors:
    /// with `dI[mu,nu,P] = -d1[mu-r0,nu,P] (mu on the atom)
    ///                      -d1[nu-r0,mu,P] (nu on the atom)
    ///                      -d2[mu,nu,P-p0]  (P on the atom)`
    /// (see [`RawRiTensors::d_i_block`]) no `[nao,nao]` `dI` plane is ever
    /// materialised, and the `d1`/`d2` blocks are read once for every root
    /// instead of once per root — the integral cost is paid once per
    /// perturbation rather than once per perturbation and root.
    ///
    /// Cost per perturbation: `O(naux*nao*nrow_A)` for the two `d1` terms plus
    /// `O(naux_A*nao^2)` for `d2`, against `O(naux*nao^2)` per root for the
    /// streamed `d_i_block` route.
    pub fn folded_qx_multi(
        &self,
        folds: &[FoldedCovector],
        blocks: &AtomDerivBlocks,
        comp: usize,
        u: &[f64],
    ) -> Vec<f64> {
        let nao = self.nao;
        let naux = self.naux;
        let n2 = nao * nao;
        let nrow = blocks.nrow;
        let row0 = blocks.row0;
        let stride1 = nrow * nao * naux;
        let d1c = &blocks.d1[comp * stride1..(comp + 1) * stride1];
        let stride2 = n2 * blocks.naux_a;
        let d2c = &blocks.d2[comp * stride2..(comp + 1) * stride2];

        let mut out = vec![0.0f64; folds.len()];
        for (fk, fold) in folds.iter().enumerate() {
            let b_ao = &fold.b_ao;
            let mut g = 0.0f64;
            for p in 0..naux {
                let b = &b_ao[p * n2..(p + 1) * n2];
                // mu on the perturbed atom: -sum_{q,nu} B[row0+q, nu] d1[q,nu,P]
                for nu in 0..nao {
                    let bcol = &b[row0 + nu * nao..row0 + nu * nao + nrow];
                    let off = nu * nrow + p * nrow * nao;
                    let mut acc = 0.0f64;
                    for q in 0..nrow {
                        acc += bcol[q] * d1c[off + q];
                    }
                    g -= acc;
                }
                // nu on the perturbed atom: -sum_{q,mu} B[mu, row0+q] d1[q,mu,P]
                for q in 0..nrow {
                    let brow = &b[(row0 + q) * nao..(row0 + q) * nao + nao];
                    let off = q + p * nrow * nao;
                    let mut acc = 0.0f64;
                    for mu in 0..nao {
                        acc += brow[mu] * d1c[off + mu * nrow];
                    }
                    g -= acc;
                }
                // P an auxiliary function of the perturbed atom
                if p >= blocks.aux0 && p < blocks.aux0 + blocks.naux_a {
                    let d2p = &d2c[(p - blocks.aux0) * n2..(p - blocks.aux0 + 1) * n2];
                    let mut acc = 0.0f64;
                    for k in 0..n2 {
                        acc += b[k] * d2p[k];
                    }
                    g -= acc;
                }
            }
            // orbital-rotation channel: sum bq:(U^T q + q U) = 2 Tr[U^T W]
            g += 2.0
                * u.iter()
                    .zip(fold.w_rot.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
            out[fk] = g;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// CPHF response and gradient driver
// ---------------------------------------------------------------------------

/// Frozen-density RI Coulomb derivative `d J[dm] / dR` (PySCF
/// `gw_cd_grad._j_deriv_from`).
fn j_deriv_from(engine: &GwGradEngine, di: &[f64], dj: &[f64], dm: &[f64]) -> Vec<f64> {
    engine.raw.with_i3(|i3| {
        j_deriv_from_impl(engine.nao, engine.naux, i3, di, dj, dm, &|b, n| {
            engine.solve_metric_batch(b, n)
        })
    })
}

fn j_deriv_from_impl(
    nao: usize,
    naux: usize,
    i3: &[f64],
    di: &[f64],
    dj: &[f64],
    dm: &[f64],
    solve: &dyn Fn(&mut [f64], usize),
) -> Vec<f64> {
    let n2 = nao * nao;
    // rho[P] = sum_k I[k, P] dm[k]: contract the natural [n2, naux] views (no `to_mat` copies)
    let mut c = gemm_nt(i3, n2, naux, dm, n2, 1);
    let drho = gemm_nt(di, n2, naux, dm, n2, 1);
    solve(&mut c, 1);
    let djc = gemm_nn(dj, naux, naux, &c, naux, 1);
    let mut dc: Vec<f64> = drho.iter().zip(djc.iter()).map(|(a, b)| a - b).collect();
    solve(&mut dc, 1);
    // out[k] = sum_P dI[k, P] c[P] + sum_P I[k, P] dc[P]
    let c_term = gemm_nn(di, n2, naux, &c, naux, 1);
    let dc_term = gemm_nn(i3, n2, naux, &dc, naux, 1);
    c_term.iter().zip(dc_term.iter()).map(|(a, b)| a + b).collect()
}

/// Frozen-orbital RI exchange derivative (PySCF `gw_cd_grad._k_deriv_from`):
/// with `dm = 2 sum_i C_i C_i^T` the derivative of K is
/// `dK = 2 sum_i [dY_i G_i + (dY_i G_i)^T - Y_i J^{-1} dJ G_i]` with
/// `Y_i[mu,P] = (mu i|P)` and `G_i = J^{-1} Y_i^T`.
///
/// All metric solves run through the pre-computed Cholesky factor with
/// `nao*nocc` right-hand sides at once (PySCF solves per occupied orbital;
/// factoring the metric once per perturbation instead of twice per AO
/// column is the main cost saving).
fn k_deriv_from(engine: &GwGradEngine, di: &[f64], dj: &[f64], cocc: &[f64]) -> Vec<f64> {
    engine.raw.with_i3(|i3| {
        k_deriv_from_impl(
            engine.nao,
            engine.naux,
            engine.nocc,
            i3,
            di,
            dj,
            cocc,
            &|b, n| engine.solve_metric_batch(b, n),
        )
    })
}

fn k_deriv_from_impl(
    nao: usize,
    naux: usize,
    nocc: usize,
    i3: &[f64],
    di: &[f64],
    dj: &[f64],
    cocc: &[f64],
    solve: &dyn Fn(&mut [f64], usize),
) -> Vec<f64> {
    let n2 = nao * nao;
    let ncol = nao * nocc; // metric-solve columns (i, mu) = i*nao + mu

    // Y^T and dY^T of all occupied orbitals side by side:
    //   yt[P, i*nao + mu] = sum_nu I3[mu + nu*nao + P*n2] Cocc[nu, i]
    // per auxiliary centre P: block = I3_p @ Cocc ([nao,nao] @ [nao,nocc])
    let cm = to_mat(cocc, nao, nocc);
    let mut yt = vec![0.0f64; naux * ncol];
    let mut dyt = vec![0.0f64; naux * ncol];
    {
        let mut ip = MatrixFull::new([nao, nao], 0.0);
        let mut block = MatrixFull::new([nao, nocc], 0.0);
        for p in 0..naux {
            ip.data.copy_from_slice(&i3[p * n2..(p + 1) * n2]);
            _dgemm_full(&ip, 'N', &cm, 'N', &mut block, 1.0, 0.0);
            for col in 0..ncol {
                yt[p + col * naux] = block.data[col];
            }
            ip.data.copy_from_slice(&di[p * n2..(p + 1) * n2]);
            _dgemm_full(&ip, 'N', &cm, 'N', &mut block, 1.0, 0.0);
            for col in 0..ncol {
                dyt[p + col * naux] = block.data[col];
            }
        }
    }
    // G = J^{-1} Y^T (in yt) and G2 = J^{-1} dJ G, one multi-column solve each
    let yt0 = yt.clone(); // unsolved Y^T, needed for the Y_i G2_i products
    let mut yt_b = yt;
    // one batched triangular solve over all `nao*nocc` columns (BLAS-3): the
    // per-column route ran `ncol` single-RHS `dtrsv` passes over the same
    // `naux^2` factor, which is memory-bound and dominated this routine
    solve(&mut yt_b, ncol);
    let djm = to_mat(dj, naux, naux);
    let g_pre = to_mat(&yt_b, naux, ncol);
    let mut djg = MatrixFull::new([naux, ncol], 0.0);
    _dgemm_full(&djm, 'N', &g_pre, 'N', &mut djg, 1.0, 0.0);
    let mut djg_flat = djg.data;
    solve(&mut djg_flat, ncol);

    // dK = 2 sum_i [dY_i G_i + (dY_i G_i)^T - Y_i G2_i]; the per-occupied
    // blocks are contiguous column blocks of yt/dyt/yt0/djg
    let mut out = vec![0.0f64; n2];
    let mut dk1 = vec![0.0f64; n2];
    let mut term = vec![0.0f64; n2];
    let mut yb = MatrixFull::new([naux, nao], 0.0);
    let mut dyb = MatrixFull::new([naux, nao], 0.0);
    let mut gb = MatrixFull::new([naux, nao], 0.0);
    let mut g2b = MatrixFull::new([naux, nao], 0.0);
    let mut acc = MatrixFull::new([nao, nao], 0.0);
    for i in 0..nocc {
        let off = i * nao * naux;
        dyb.data.copy_from_slice(&dyt[off..off + naux * nao]);
        yb.data.copy_from_slice(&yt0[off..off + naux * nao]);
        gb.data.copy_from_slice(&yt_b[off..off + naux * nao]);
        g2b.data.copy_from_slice(&djg_flat[off..off + naux * nao]);
        // dY_i G_i: dY_i = dyb^T ([nao, naux] = transpose of the [naux, nao] block)
        _dgemm_full(&dyb, 'T', &gb, 'N', &mut acc, 1.0, 0.0);
        dk1.copy_from_slice(&acc.data);
        _dgemm_full(&yb, 'T', &g2b, 'N', &mut acc, 1.0, 0.0);
        term.copy_from_slice(&acc.data);
        for row in 0..nao {
            for col in 0..nao {
                let k = row + col * nao;
                let k_t = col + row * nao;
                out[k] += 2.0 * (dk1[k] + dk1[k_t] - term[k]);
            }
        }
    }
    out
}

/// multi-right-hand-side metric solve on a flat [naux, ncol] col-major buffer
/// (kept for the validation harnesses that solve column by column)
#[allow(dead_code)]
fn solve_metric_flat(b: &mut [f64], naux: usize, ncol: usize, solve: &dyn Fn(&[f64]) -> Vec<f64>) {
    for col in 0..ncol {
        let sol = solve(&b[col * naux..(col + 1) * naux]);
        b[col * naux..(col + 1) * naux].copy_from_slice(&sol);
    }
}

impl GwGradEngine<'_> {
    /// Canonical response of every nuclear perturbation `(atm, comp)`,
    /// computed by the `analdrv` CP-HF driver: `RHessSCF` assembles the
    /// dimensionless CP-HF right-hand side and solves `U + resp(U) = rhs`
    /// for all perturbations at once through its block-Krylov solver
    /// (`krylov_block`), with the RI-JK response of `src/ri_jk/hess_r.rs`
    /// built on `rimatr`.
    ///
    /// Returns one `(U [nmo*nmo col-major], eps1 [nmo], b_x [nmo])` per
    /// perturbation, ordered `atm*3 + comp`; `U` and `eps1` follow
    /// `pyscf/gw/gw_cd_grad*.py::_cp_response_atom` and `b_x` is the v_xc
    /// derivative of the static term `B_nn` (zero for an HF reference; see
    /// the module header).
    pub fn canonical_response_batch(
        &self,
    ) -> Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        self.canonical_response_batch_impl(false).0
    }

    /// Same as [`GwGradEngine::canonical_response_batch`], but also returns the
    /// per-perturbation intermediates (`jd`, `kd`, `f1_skel_mo`, `s1_mo`,
    /// `mo1`, `rhs`) used by the adjoint cross-checks.
    pub fn canonical_response_batch_full(
        &self,
    ) -> (
        Vec<(Vec<f64>, Vec<f64>, Vec<f64>)>,
        Vec<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)>,
    ) {
        self.canonical_response_batch_impl(true)
    }

    /// Shared driver of [`Self::canonical_response_batch`] and
    /// [`Self::canonical_response_batch_full`]; `collect_debug` also keeps the
    /// per-perturbation intermediates (6 buffers per perturbation, ≈ 75 MB at
    /// naphthalene/def2-SVP), which only the validation harnesses read.
    #[allow(clippy::type_complexity)]
    fn canonical_response_batch_impl(
        &self,
        collect_debug: bool,
    ) -> (
        Vec<(Vec<f64>, Vec<f64>, Vec<f64>)>,
        Vec<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)>,
    ) {
        use crate::analdrv::prelude::{
            AnalDrvConfig, HessNucAPI, RHessCoreAPI, RHessElecInteractAPI, RHessHcore,
            RHessOvlp, RHessSCF,
        };
        use crate::analdrv::response::rresp_interface::rscf_resp_interface;
        use crate::ri_jk::hess_r::RHessRIJK;
        use crate::ri_jk::util::{get_cint_aux, get_cint_mol};
        use rstsr::prelude::*;

        let nao = self.nao;
        let nmo = self.nmo;
        let nocc = self.nocc;
        let natm = self.natm;

        // ---- DFT reference pieces of the CP-KS solve and the B_n covector --
        // The static term `B_n = (1-alpha) Sigma_x - v_xc` of the QP equation
        // is differentiated in three channels: the (1-alpha) RI-exchange part
        // through the (bq, bJ) pullback (see `qp_pullback_chunk`), and the
        // v_xc part through the `b_x` covector:
        //   b_x[n] = -[ <n|dv_xc^expl/dR|n> + 2 sum_p U[p,n] v_xc[p,n]
        //               + <n| fxc(D^x) |n> ],
        // the explicit grid derivative (frozen density and MOs), the orbital
        // rotation and the XC-kernel density response of the reference
        // (port of `pyscf/gw/gw_cd_grad.py::_b_nuclear_deriv`).  The same
        // v_xc pieces must also enter the CP-KS right-hand side and the
        // first-order Fock of the Roothaan restoration, so that `U`, `eps1`
        // are the KS response of the actual reference.
        let is_hf = self.scf.mol.xc_data.dfa_compnt_scf.is_empty();
        let vxc_d1_atoms: Option<Vec<Vec<f64>>> = if is_hf {
            None
        } else {
            assert!(
                self.scf.mol.xc_data.omega().is_none(),
                "gw_grad: range-separated hybrid references are not supported"
            );
            let t0 = Instant::now();
            let v = crate::hessian::rks::compute_vxc_h1ao(self.scf);
            if grad_timing() {
                eprintln!(
                    "[gw_grad timing]   cphf: vxc_deriv1 grid sweep {:8.2}s",
                    t0.elapsed().as_secs_f64()
                );
            }
            Some(v)
        };
        let fxc_cache = crate::hessian::rks::prepare_fxc_cache(self.scf);

        // ---- analdrv CP-HF driver with the RI-JK response ----
        let device = DeviceBLAS::default();
        let mol = get_cint_mol(&self.scf.mol);
        let aux = get_cint_aux(&self.scf.mol);
        let mo_coeff =
            rt::asarray((&self.c_mo.data, [nao, nmo], &device)).into_contig(ColMajor);
        let mo_occ =
            rt::asarray((&self.scf.occupation[0], [nmo], &device)).into_contig(ColMajor);
        let mo_energy = rt::asarray((&self.e, [nmo], &device)).into_contig(ColMajor);

        let mut ovlp_obj = RHessOvlp::new(&mol, &device);
        let mut hcore_obj = RHessHcore::new(&mol, &device);
        let rimatr = &self
            .scf
            .rimatr
            .as_ref()
            .expect("gw_grad: analdrv CP-HF requires rimatr (use_auxbas + use_ri_symm)")
            .0;
        use crate::utilities::rstsr_util::RestTensorToRstsrViewAPI;
        let cderi = rimatr.to_rstsr_view(&device).into_cow();
        let j2c_decomp = crate::ri_jk::get_j2c_decomp(
            &aux,
            &device,
            self.scf.mol.ctrl.j2c_decomp.clone(),
        );
        // the skeleton/response exchange weight follows the reference: the
        // frozen Fock is `h1 + J1 - 0.5*hyb*K1 (+ v_xc^x)` for a hybrid,
        // exactly `J - K` for HF (`fock_hyb` is 1.0 there)
        let mut rijk_obj =
            RHessRIJK::new_with_cderi(&mol, &aux, 1.0, self.fock_hyb, cderi, j2c_decomp);

        let mut config = AnalDrvConfig::default();
        config.resp.cache_kernel = self.config.resp_cache_kernel;
        config.resp.cache_budget = self.config.resp_cache_budget_mib;
        // upstream (analdrv refactor): the CP-SCF solve now lives in the shared
        // response object, which `RHessSCF` only borrows.
        let mut resp = rscf_resp_interface(self.scf, &config);
        let nuc_list: Vec<&mut dyn HessNucAPI> = Vec::new();
        let core_list: Vec<&mut dyn RHessCoreAPI> = vec![&mut hcore_obj];
        let el_list: Vec<&mut dyn RHessElecInteractAPI> = vec![&mut rijk_obj];
        // `hess` holds a mutable borrow of `resp`, so it is confined to this
        // scope; the block ends before `resp` is used directly for the solve.
        let (mut rhs_map, t_prep, t_rhs) = {
            let mut hess = RHessSCF::new(
                mo_coeff, mo_occ, mo_energy, &mut ovlp_obj, nuc_list, core_list, el_list,
                &mut resp, &config.nucgrad,
            );
            let t_prep0 = Instant::now();
            hess.make_response_preparation();
            let t_prep = t_prep0.elapsed().as_secs_f64();

            // ---- dimensionless CP-HF right-hand side (analdrv skeleton route)
            //      and one block-Krylov solve over all perturbations ----
            let t_rhs0 = Instant::now();
            let rhs_map = hess.compute_dimless_cpscf_rhs();
            if std::env::var("REST_GWGRAD_RIJK_TIMING").is_ok() {
                for (k, v) in rijk_obj.timing.iter() {
                    eprintln!("[rijk timing] {k:44} {v:8.3}s");
                }
            }
            let t_rhs = t_rhs0.elapsed().as_secs_f64();
            (rhs_map, t_prep, t_rhs)
        };
        // ---- v_xc skeleton of the CP-KS right-hand side ----
        // `compute_dimless_cpscf_rhs` only knows the core + RI-JK skeleton;
        // a DFT reference adds the frozen-density XC derivative, which enters
        // the dimensionless rhs exactly like `f1mo` does:
        // `rhs[vir,occ] -= (C^T v_xc^x C)[vir,occ] / (e_a - e_i + shift)`.
        if let Some(vxc_d1) = vxc_d1_atoms.as_ref() {
            let level_shift = config.resp.level_shift;
            let rhs_tsr_mut = rhs_map.get_mut("rhs").expect("gw_grad: rhs tensor");
            let off = rhs_tsr_mut.offset();
            let rhs_raw = &mut rhs_tsr_mut.raw_mut()[off..];
            for atm in 0..natm {
                for comp in 0..3 {
                    let vxc_d1_ao = vxc_block_ao(&vxc_d1[atm], comp, nao);
                    let t = gemm_nt(&self.c_mo.data, nao, nmo, &vxc_d1_ao, nao, nao);
                    let vxc_d1_mo = gemm_nn(&t, nmo, nao, &self.c_mo.data, nao, nmo);
                    let k = atm * 3 + comp;
                    let base = k * nmo * nocc;
                    for i in 0..nocc {
                        for a in nocc..nmo {
                            let denom = self.e[a] - self.e[i] + level_shift;
                            rhs_raw[base + a + i * nmo] -=
                                vxc_d1_mo[a + i * nmo] / denom;
                        }
                    }
                }
            }
        }
        let t_solve0 = Instant::now();
        let mo1_tsr = resp.solve_dimless_cpscf(rhs_map["rhs"].view()); // [nmo, nocc, 3, natm]
        let t_solve = t_solve0.elapsed().as_secs_f64();

        // ---- canonical-orbital assembly per perturbation ----
        // The analdrv tensors order perturbations as col-major [.., .., 3, natm],
        // i.e. k_pert = comp + atm*3.  The full-MO skeleton Fock/overlap
        // derivatives needed by the Roothaan restoration (all nmo columns)
        // are not available from the optimized RI-JK hessian object (only
        // `get_deriv1_bra` is implemented), so they come from the raw-RI
        // route here.
        let mo1_data = &mo1_tsr.raw()[mo1_tsr.offset()..];
        let rhs_tsr = &rhs_map["rhs"];
        let rhs_data = &rhs_tsr.raw()[rhs_tsr.offset()..];
        let n2 = nao * nao;
        let mut responses = Vec::with_capacity(natm * 3);
        let mut debug: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> =
            Vec::with_capacity(natm * 3);
        let (mut t_dij, mut t_skel, mut t_fin) = (0.0f64, 0.0f64, 0.0f64);
        T_TIMER.with(|c| c.borrow_mut().fill(0.0));
        for atm in 0..natm {
            let t0 = Instant::now();
            let h1r = crate::hessian::rhf::build_hcore_first_deriv(&self.scf.mol, atm);
            let s1r = crate::ri_cphf::cphf_solver_pyscf::build_s1ao_deriv(&self.scf.mol, atm);
            t_dij += t0.elapsed().as_secs_f64();
            // shell-sliced derivative integral blocks of this atom, shared
            // by the three Cartesian components
            let blocks = self.raw.d_atom_blocks(atm);
            for comp in 0..3 {
                let k_pert = atm * 3 + comp;
                let t0 = Instant::now();
                let di = self.raw.d_i_from_blocks(&blocks, comp);
                let dj = self.raw.d_j_atom(atm, comp);
                t_dij += t0.elapsed().as_secs_f64();
                let t0 = Instant::now();
                // v_xc skeleton of this perturbation in the MO basis (DFT
                // references): shared by the skeleton Fock and the b_x covector
                let vxc_d1_mo: Option<Vec<f64>> = vxc_d1_atoms.as_ref().map(|v| {
                    let vao = vxc_block_ao(&v[atm], comp, nao);
                    let t = gemm_nt(&self.c_mo.data, nao, nmo, &vao, nao, nao);
                    gemm_nn(&t, nmo, nao, &self.c_mo.data, nao, nmo)
                });
                let (mut f1_skel_mo, s1_mo, jd, kd) =
                    self.canonical_skeleton_mo(atm, comp, &h1r, &s1r, &di, &dj);
                if let Some(vd1) = vxc_d1_mo.as_ref() {
                    for (f, v) in f1_skel_mo.iter_mut().zip(vd1.iter()) {
                        *f += v;
                    }
                }
                t_skel += t0.elapsed().as_secs_f64();
                let mo1_slab = &mo1_data[k_pert * nmo * nocc..(k_pert + 1) * nmo * nocc];
                let t0 = Instant::now();
                let resp1 = self.canonical_finish(
                    &f1_skel_mo,
                    &s1_mo,
                    mo1_slab,
                    vxc_d1_mo.as_deref(),
                    fxc_cache.as_ref(),
                );
                responses.push(resp1);
                if collect_debug {
                    debug.push((
                        jd,
                        kd,
                        f1_skel_mo,
                        s1_mo,
                        mo1_slab.to_vec(),
                        rhs_data[k_pert * nmo * nocc..(k_pert + 1) * nmo * nocc].to_vec(),
                    ));
                } else {
                    // the intermediates are only read by the validation
                    // harnesses; dropping them here keeps the CP-HF work space
                    // (and its transient copies) off the production path
                    drop((jd, kd, f1_skel_mo, s1_mo));
                }
                t_fin += t0.elapsed().as_secs_f64();
            }
        }
        if grad_timing() {
            let [t_jd, t_kd, t_jup, t_kup] = T_TIMER.with(|c| *c.borrow());
            eprintln!(
                "[gw_grad timing]   cphf: prep {:7.2}s  rhs {:7.2}s  solve {:7.2}s | assembly: di/dj+h1/s1 {:7.2}s  skeleton {:7.2}s (j {:.2} k {:.2})  finish {:7.2}s (J {:.2} K {:.2})",
                t_prep, t_rhs, t_solve, t_dij, t_skel, t_jd, t_kd, t_fin, t_jup, t_kup
            );
        }
        let _ = n2;
        (responses, debug)
    }

    /// Full-MO skeleton Fock and overlap derivatives of one perturbation,
    /// from the raw-RI route: `f1_skel_mo = C^T (dh1/dR + d(J-K skeleton)/dR) C`
    /// and `s1_mo = C^T (dS/dR) C`, both [nmo*nmo] col-major.
    fn canonical_skeleton_mo(
        &self,
        atm: usize,
        comp: usize,
        h1r: &[f64],
        s1r: &[Vec<f64>],
        di: &[f64],
        dj: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let nao = self.nao;
        let nmo = self.nmo;
        let nocc = self.nocc;
        let n2 = nao * nao;

        let dm = self.scf.density_matrix[0].data.clone();
        let cocc: Vec<f64> = (0..nocc)
            .flat_map(|i| (0..nao).map(move |mu| self.c_mo[[mu, i]]))
            .collect();
        let (jd, kd) = {
            let t0 = Instant::now();
            let jd = j_deriv_from(self, di, dj, &dm);
            timer_add(0, t0);
            let t0 = Instant::now();
            let kd = k_deriv_from(self, di, dj, &cocc);
            timer_add(1, t0);
            (jd, kd)
        };
        // frozen Fock skeleton `h1 + J1 - 0.5*hyb*K1` — the exchange weight
        // follows the reference (`fock_hyb` is 1.0 for HF)
        let vfrozen: Vec<f64> = jd
            .iter()
            .zip(kd.iter())
            .map(|(a, b)| a - 0.5 * self.fock_hyb * b)
            .collect();
        // Fock skeleton in AO, col-major
        let mut f1_ao = vec![0.0f64; n2];
        for k in 0..n2 {
            let i = k % nao;
            let j = k / nao;
            f1_ao[k] = h1r[comp * n2 + i * nao + j] + vfrozen[k];
        }
        let f1_skel_mo = {
            let t = gemm_nt(&self.c_mo.data, nao, nmo, &f1_ao, nao, nao);
            gemm_nn(&t, nmo, nao, &self.c_mo.data, nao, nmo)
        };
        // overlap derivative in AO, col-major (s1r is row-major)
        let s1_ao: Vec<f64> = (0..n2)
            .map(|k| {
                let i = k % nao;
                let j = k / nao;
                s1r[comp][i * nao + j]
            })
            .collect();
        let s1_mo = {
            let t = gemm_nt(&self.c_mo.data, nao, nmo, &s1_ao, nao, nao);
            gemm_nn(&t, nmo, nao, &self.c_mo.data, nao, nmo)
        };
        (f1_skel_mo, s1_mo, jd, kd)
    }

    /// Canonical-orbital response `(u, eps1, b_x)` of one perturbation from
    /// the analdrv CP-HF solution: relaxed first-order density, complete
    /// first-order Fock and the Roothaan-style orbital rotations.  With a DFT
    /// reference (`vxc_d1_mo` given) the relaxed Fock carries the XC-kernel
    /// response `fxc(D^x)` and `b_x` collects the v_xc derivative of the
    /// static term `B_n = (1-alpha) Sigma_x - v_xc` (see
    /// `canonical_response_batch_impl`); for HF `b_x` stays zero.
    fn canonical_finish(
        &self,
        f1_skel_mo: &[f64], // skeleton F1 in MO basis [nmo*nmo], col-major
        s1_mo: &[f64],      // overlap derivative in MO basis [nmo*nmo], col-major
        mo1: &[f64],        // analdrv CP-HF solution [nmo*nocc], col-major
        vxc_d1_mo: Option<&[f64]>, // frozen-density v_xc skeleton, MO [nmo*nmo]
        fxc_cache: Option<&crate::dft::response::FxcHessianCache>,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let nao = self.nao;
        let nmo = self.nmo;
        let nocc = self.nocc;
        let n2 = nao * nao;

        // first-order density and complete first-order Fock
        let cocc: Vec<f64> = (0..nocc)
            .flat_map(|i| (0..nao).map(move |mu| self.c_mo[[mu, i]]))
            .collect();
        let mut c1 = vec![0.0f64; nao * nocc];
        for i in 0..nocc {
            for p in 0..nmo {
                let m1 = mo1[p + i * nmo];
                if m1 != 0.0 {
                    for mu in 0..nao {
                        c1[mu + i * nao] += self.c_mo[[mu, p]] * m1;
                    }
                }
            }
        }
        let mut dm1 = vec![0.0f64; n2];
        for mu in 0..nao {
            for nu in 0..nao {
                let mut acc = 0.0;
                for i in 0..nocc {
                    acc += c1[mu + i * nao] * cocc[nu + i * nao]
                        + cocc[mu + i * nao] * c1[nu + i * nao];
                }
                dm1[mu + nu * nao] = 2.0 * acc;
            }
        }
        let dm1_m = to_mat(&dm1, nao, nao);
        let t0 = Instant::now();
        let j_up = crate::dft::response::compute_j_upper(self.scf, &vec![dm1_m.clone()]);
        timer_add(2, t0);
        let t0 = Instant::now();
        let k_up = crate::dft::response::compute_k_upper(self.scf, &vec![dm1_m.clone()]);
        timer_add(3, t0);
        let j_full = j_up.to_matrixfull().expect("gw_grad: J upper").data;
        let k_full = k_up.to_matrixfull().expect("gw_grad: K upper").data;
        // XC-kernel response of the density change (DFT references only)
        let fxc_ao: Option<MatrixFull<f64>> = fxc_cache
            .map(|c| crate::dft::response::compute_fxc_response_ao_cached(c, &dm1_m));

        // relaxed part of the first-order Fock in the AO basis ...
        let mut f1_relaxed = vec![0.0f64; n2];
        for k in 0..n2 {
            f1_relaxed[k] = j_full[k] - 0.5 * self.fock_hyb * k_full[k];
        }
        if let Some(fa) = fxc_ao.as_ref() {
            for (f, v) in f1_relaxed.iter_mut().zip(fa.data.iter()) {
                *f += v;
            }
        }
        // ... transformed to MO and added to the skeleton derivative
        let mut f1_mo: Vec<f64> = {
            let relaxed = {
                let t = gemm_nt(&self.c_mo.data, nao, nmo, &f1_relaxed, nao, nao);
                gemm_nn(&t, nmo, nao, &self.c_mo.data, nao, nmo)
            };
            f1_skel_mo
                .iter()
                .zip(relaxed.iter())
                .map(|(a, b)| a + b)
                .collect()
        };
        // Both derivative matrices are derivatives of symmetric matrices, so
        // the Roothaan restoration only ever reads their symmetric part.  Any
        // asymmetry is numerical (the triangular `to_matrixfull` mirrors, the
        // MO transformation preserves symmetry up to rounding); averaging it
        // away makes the orthogonality identity `u + u^T + S1 = 0` exact for
        // **every** pair, whatever its gap.  Without it the identity residual
        // of a symmetry-degenerate (or merely nearly degenerate) reference is
        // the rounding noise divided by `e_q - e_p`, which can abort a valid
        // calculation through the diagnostic below.
        // how asymmetric the raw derivatives are (the identity residual of a
        // near-degenerate pair is this noise divided by the gap)
        let (mut f1_asym, mut s1_asym) = (0.0f64, 0.0f64);
        for p in 0..nmo {
            for q in 0..nmo {
                f1_asym = f1_asym.max((f1_mo[p + q * nmo] - f1_mo[q + p * nmo]).abs());
                s1_asym = s1_asym.max((s1_mo[p + q * nmo] - s1_mo[q + p * nmo]).abs());
            }
        }
        // `REST_GWGRAD_NO_SYMM` restores the historical behaviour (raw
        // derivatives) and exists only so the degeneracy diagnostic can be
        // reproduced in a test.
        if std::env::var("REST_GWGRAD_NO_SYMM").is_err() {
            symmetrise_square(&mut f1_mo, nmo);
        }
        let s1_mo: Vec<f64> = {
            let mut v = s1_mo.to_vec();
            if std::env::var("REST_GWGRAD_NO_SYMM").is_err() {
                symmetrise_square(&mut v, nmo);
            }
            v
        };
        if std::env::var("REST_GWGRAD_DBG").is_ok() {
            println!("[cphf DBG] raw f1/s1 asymmetry {f1_asym:.3e} / {s1_asym:.3e}");
        }

        let mut u = vec![0.0f64; nmo * nmo];
        let mut eps1 = vec![0.0f64; nmo];
        for p in 0..nmo {
            u[p + p * nmo] = -0.5 * s1_mo[p + p * nmo];
            eps1[p] = f1_mo[p + p * nmo] - self.e[p] * s1_mo[p + p * nmo];
        }
        // Non-degenerate pairs: the usual Roothaan restoration.  Pairs inside
        // one degenerate shell are excluded (their `e_q - e_p` vanishes) and
        // handled below.
        for p in 0..nmo {
            for q in 0..nmo {
                if p == q || self.same_degenerate_shell(p, q) {
                    continue;
                }
                u[p + q * nmo] = (f1_mo[p + q * nmo] - self.e[q] * s1_mo[p + q * nmo])
                    / (self.e[q] - self.e[p]);
            }
        }
        // Degenerate shells: the perturbation is not diagonal in the SCF
        // basis, so its shell block
        //     A[i,j] = F1[p_i, p_j] - e * S1[p_i, p_j]      (e = shell energy)
        // is diagonalised; its eigenvalues are the level derivatives
        // `eps1[p_i]` and its eigenvectors are the first-order rotation of the
        // shell basis (degenerate perturbation theory).  `tr A` — and hence
        // the shell-averaged gradient — is basis independent either way, but
        // the rotation makes each member's `eps1` the derivative of its own
        // split level instead of an arbitrary mixture of the two.
        for shell in self.shells.iter() {
            let d = shell.len();
            if d < 2 {
                continue;
            }
            let e0 = shell.iter().map(|&p| self.e[p]).sum::<f64>() / d as f64;
            let mut a = vec![0.0f64; d * d];
            for (ii, &p) in shell.iter().enumerate() {
                for (jj, &q) in shell.iter().enumerate() {
                    a[ii + jj * d] = f1_mo[p + q * nmo] - e0 * s1_mo[p + q * nmo];
                }
            }
            for ii in 0..d {
                for jj in (ii + 1)..d {
                    let s = 0.5 * (a[ii + jj * d] + a[jj + ii * d]);
                    a[ii + jj * d] = s;
                    a[jj + ii * d] = s;
                }
            }
            let (lam, v) = eigh_symmetric(&a, d);
            for (ii, &p) in shell.iter().enumerate() {
                eps1[p] = lam[ii];
            }
            // antisymmetric part of V (the pure first-order rotation)
            for ii in 0..d {
                for jj in 0..d {
                    if ii != jj {
                        let (p, q) = (shell[ii], shell[jj]);
                        u[p + q * nmo] = 0.5 * (v[ii + jj * d] - v[jj + ii * d]);
                    }
                }
            }
        }
        // The orthogonality identity `u + u^T + S1 = 0` is imposed by the
        // non-degenerate formula; inside a shell a rotation of the basis is a
        // free gauge (the shell functional is invariant under it), so those
        // pairs are excluded from the diagnostic.  With the symmetrised
        // `f1_mo`/`s1_mo` above the identity is algebraically exact, so what is
        // measured here is the solve residual only (diffuse bases amplify it by
        // the smallest orbital gap).
        let mut orth = 0.0f64;
        let mut worst = (0usize, 0usize, 0.0f64);
        for p in 0..nmo {
            for q in 0..nmo {
                if self.same_degenerate_shell(p, q) {
                    continue;
                }
                let e = (u[p + q * nmo] + u[q + p * nmo] + s1_mo[p + q * nmo]).abs();
                if e > worst.2 {
                    worst = (p, q, e);
                }
                orth = orth.max(e);
            }
        }
        if orth > 1.0e-2 {
            panic!(
                "gw_grad: canonical response orthogonality error {:.3e} at pair ({}, {}) \
                 (gap {:.3e}, degenerate_tol {:.1e})",
                orth,
                worst.0,
                worst.1,
                (self.e[worst.1] - self.e[worst.0]).abs(),
                self.config.degenerate_tol
            );
        }
        if orth > 1.0e-6 {
            println!(
                "warning!!! gw_grad canonical response orthogonality error {:.3e} at pair ({}, {})",
                orth, worst.0, worst.1
            );
        }
        // ---- v_xc part of the static-term covector b_x (DFT references) ----
        // `b_x[p] = -d v_xc,pp / dR`: explicit grid derivative (frozen density
        // and MOs), orbital rotation `2 sum_q U[q,p] v_xc[q,p]` and the
        // XC-kernel response `<p| fxc(D^x) |p>` of the first-order density.
        // The `(1-alpha) Sigma_x` half of `B_n` needs nothing here — its
        // complete derivative (explicit + orbital) flows through the
        // (bq, bJ) exchange pullback of `qp_pullback_chunk`.
        let mut b_x = vec![0.0f64; nmo];
        if let Some(vd1) = vxc_d1_mo {
            // MO diagonal of the fxc response: diag(C^T fxc_ao C)
            let fxc_diag: Vec<f64> = if let Some(fa) = fxc_ao.as_ref() {
                let t = gemm_nt(&self.c_mo.data, nao, nmo, &fa.data, nao, nao);
                (0..nmo)
                    .map(|p| {
                        let base = p * nao;
                        (0..nao)
                            .map(|mu| t[p + mu * nmo] * self.c_mo.data[mu + base])
                            .sum::<f64>()
                    })
                    .collect()
            } else {
                vec![0.0f64; nmo]
            };
            for p in 0..nmo {
                let mut s = vd1[p + p * nmo] + fxc_diag[p];
                let col = p * nmo;
                for q in 0..nmo {
                    s += 2.0 * u[q + col] * self.vxc_mo[q + col];
                }
                b_x[p] = -s;
            }
        }
        (u, eps1, b_x)
    }

    /// Analytic gradient of one LR-CD G0W0 QP energy.
    ///
    /// Returns `(grad, omega, z_factor)` with `grad[atm*3 + comp]` in
    /// Hartree/Bohr (row-major over atoms and Cartesian components).
    ///
    /// When `target` belongs to a symmetry-protected degenerate shell the
    /// returned gradient is the **shell average** `(1/d) sum_m d Omega_m/dR`
    /// and `omega` is the QP energy of `target` itself (all shell members are
    /// degenerate).  This is the only basis-independent definition of "the"
    /// gradient there, and it is what the central finite difference of the
    /// QP energy of orbital `target` converges to: displacing the nuclei
    /// splits the shell, the sorted index follows the *other* branch at `-h`,
    /// so the two-sided difference averages the two level derivatives.
    pub fn analytic_gradient(&self, target: usize) -> (Vec<f64>, f64, f64) {
        mem_probe("gradient: enter");
        let shell = self.degenerate_shell(target);
        let caches: Vec<QpCache> = shell.iter().map(|&m| self.build_target_cache(m)).collect();
        let w = 1.0 / caches.len() as f64;
        let refs: Vec<&QpCache> = caches.iter().collect();
        let adjoint_route = self.config.adjoint_cphf && self.supports_cphf_adjoint();
        // `qt`/`qtia` are read only while building a QP cache (`consts`, the exact-residue
        // on-the-fly factors); the gradient's own contractions never touch them. Release
        // `naux*nmo^2 + naux*nov` doubles (949 MB at naphthalene/def2-TZVP) for the rest of the
        // gradient; `with_qt`/`with_qtia` rebuild them on demand.
        self.evict_qt();
        // On the HF-only adjoint route the pullback never reads `q` (its only `q`-dependent term,
        // the static exchange derivative, is skipped for `fock_hyb == 1`) and the fold below uses
        // the streamed-integrals variant, so `q` (`naux*nmo^2` doubles, 874 MB at
        // naphthalene/def2-TZVP) can be released *before* the pullback instead of after the fold.
        // Any later reader rebuilds it through `with_q`.
        if adjoint_route {
            self.evict_q();
        }
        // The adjoint route rebuilds the raw integrals only inside `build_cphf_adjoint`, after the
        // response objects (and their kernel cache) have been released; keeping them out of the
        // covector/fold phase removes `nao^2 * naux` doubles from where this gradient peaks.
        // (The pullback and the fold read `q`/`qtia`/factors, never `i3`.)
        if adjoint_route {
            self.raw.evict_i3();
        }
        mem_probe("gradient: after QP caches");
        let (mut bq, bj, be, bb) = self.qp_pullback(&refs, &[vec![w; caches.len()]]);
        let own0 = &caches[shell.iter().position(|&m| m == target).unwrap()];
        mem_probe("gradient: after qp_pullback (bq/bj/be/bb)");
        // ---- adjoint (Z-vector) route --------------------------------
        // The reference guard is checked *before* folding (both the builder and
        // this branch use `supports_cphf_adjoint`): the folded covector is all
        // the adjoint route needs from `bq`, so the `naux * nmo^2` covector
        // buffer (874 MB at naphthalene/def2-TZVP) is released before the
        // adjoint workspace is built. The adjoint foldings and the
        // per-direction loop both then run without it resident. The
        // per-perturbation fallback below still needs `bq`, which is why the
        // release happens only on this branch.
        if adjoint_route {
            // `bq` was symmetrised by the pullback, so compact it in place to its lower triangle
            // before folding (the fold rebuilds the symmetric plane exactly): this releases
            // `naux*nmo^2/2` doubles (437 MB at naphthalene/def2-TZVP) for the fold, which is where
            // this gradient peaks. The route always returns below, so the (full-buffer) fallback
            // never sees the compacted form.
            // the pullback accumulates the pair covector directly in its packed
            // (lower-triangle) layout, so no repacking is needed here
            let packed_pairs = packed_pair_count(self.nmo);
            // no `q` here: the orbital-rotation channel is contracted against the raw integrals,
            // which are streamed auxiliary-atom block by block
            let mut fold = self.fold_covector_packed_streamed(&bq[0][..packed_pairs * self.naux]);
            // the folded covector replaces `bq` entirely, so release it here (the route returns
            // below; the full-buffer fallback never sees the compacted form)
            drop(bq);
            mem_probe("gradient: after fold_covector");
            // `q` is read only by the pullback and this fold, both done: release `naux*nmo^2`
            // doubles (874 MB at naphthalene/def2-TZVP) before the adjoint workspace is built.
            // `with_q` rebuilds it if a later task (another target, a BSE screening) needs it.
            self.evict_q();
            // `build_cphf_adjoint` takes `b_ao` out of `fold` and folds the exchange skeleton into
            // it in place, so its `naux * nao * nao` tensor becomes `adj.di_cov` instead of being
            // duplicated; only the small `w_rot` is left to release here.
            let adj = self
                .build_cphf_adjoint(&mut fold, &be[0])
                .expect("`supports_cphf_adjoint` and `build_cphf_adjoint` share this guard");
            drop(fold);
            mem_probe("gradient: after build_cphf_adjoint");
            let mut grad = vec![0.0f64; self.natm * 3];
            // HF reference only (guaranteed by `build_cphf_adjoint`)
            let zero_bx = vec![0.0f64; self.nmo];
            for atm in 0..self.natm {
                let h1r = crate::hessian::rhf::build_hcore_first_deriv(&self.scf.mol, atm);
                let s1r = crate::ri_cphf::cphf_solver_pyscf::build_s1ao_deriv(&self.scf.mol, atm);
                // the locality-aware variant streams the derivative blocks internally, so it does
                // not need (or allocate) the dense per-atom `AtomDerivBlocks`
                if atm == 0 {
                    mem_probe("gradient: inside per-direction loop (1st dir)");
                }
                if self.config.local_di {
                    let djs: Vec<Vec<f64>> = (0..3).map(|c| self.raw.d_j_atom(atm, c)).collect();
                    let g3 = self.adjoint_perturbation_gradient_local(
                        &adj, atm, &h1r, &s1r, &djs, &bj[0], &bb[0], &zero_bx, atm * 3,
                    );
                    grad[atm * 3..atm * 3 + 3].copy_from_slice(&g3);
                } else {
                    let blocks = self.raw.d_atom_blocks(atm);
                    for comp in 0..3 {
                        let dj = self.raw.d_j_atom(atm, comp);
                        grad[atm * 3 + comp] = self.adjoint_perturbation_gradient(
                            &adj, &blocks, comp, &h1r, &s1r[comp], &dj, &bj[0], &bb[0],
                            &zero_bx, atm * 3 + comp,
                        );
                    }
                }
            }
            mem_probe("gradient: after per-direction loop (adjoint)");
            return (grad, own0.omega, own0.z_factor);
        }
        // responses before the pullback: see analytic_gradient_with_cache
        // for the memory-peak rationale of this order
        let responses = self.canonical_response_batch();
        // Fold the (single) weight row's `bq` once: the AO-basis covector
        // `B = C bq C^T` and the rotation factor `W = sum_P q_P bq_P` replace
        // the per-perturbation `qx` assembly (see `FoldedCovector`).
        let folded = if self.config.folded_qx {
            Some(self.fold_covector_packed(&bq[0]))
        } else {
            None
        };
        // the legacy per-perturbation route contracts the **full** pair layout
        let bq_full: Vec<Vec<f64>> = if self.config.folded_qx {
            Vec::new()
        } else {
            bq.iter().map(|v| unpack_pairs(v, self.nmo, self.naux)).collect()
        };
        let mut grad = vec![0.0f64; self.natm * 3];
        for atm in 0..self.natm {
            let blocks = self.raw.d_atom_blocks(atm);
            for comp in 0..3 {
                let dj = self.raw.d_j_atom(atm, comp);
                let (u, eps1, b_x) = responses[atm * 3 + comp].clone();
                let g = match &folded {
                    Some(f) => self.folded_perturbation_gradient(
                        f, &blocks, comp, &u, &bj[0], &be[0], &bb[0], &dj, &eps1, &b_x,
                    ),
                    None => {
                        let qx = self.qx_from_u_blocks(&blocks, comp, &u);
                        self.contract_perturbation(
                            &bq_full, &bj, &be, &bb, &qx, &dj, &eps1, &b_x,
                        )[0]
                    }
                };
                grad[atm * 3 + comp] = g;
            }
        }
        let own = &caches[shell.iter().position(|&m| m == target).unwrap()];
        (grad, own.omega, own.z_factor)
    }

    /// QP energy of one orbital (the scalar whose gradient is returned by
    /// [`GwGradEngine::analytic_gradient`]).
    pub fn qp_energy_of(&self, target: usize) -> f64 {
        self.build_target_cache(target).omega
    }

    /// Validation helper: the QP equation residual `consts_n + Sigma_c(omega)
    /// - omega` at a fixed external frequency.  A root returned by the
    /// graphical solver has a residual of order the linear-interpolation
    /// error (~1e-5); an unconverged fallback root stands out at ~1e-2.
    pub fn qp_equation_value(&self, omega: f64, n: usize) -> f64 {
        self.consts(n) + self.sigma_c(omega, n) - omega
    }
}

// ---------------------------------------------------------------------------
// free helpers
// ---------------------------------------------------------------------------

/// One Cartesian component of the per-atom XC skeleton
/// (`hessian::rks::compute_vxc_h1ao`): the flattened col-major `[3*nao, nao]`
/// block reduced to a `[nao, nao]` col-major buffer of component `comp`
/// (rows `[comp*nao, (comp+1)*nao)`).
fn vxc_block_ao(flat: &[f64], comp: usize, nao: usize) -> Vec<f64> {
    let n2 = nao * nao;
    let mut out = vec![0.0f64; n2];
    let row0 = comp * nao;
    for j in 0..nao {
        for i in 0..nao {
            out[i + j * nao] = flat[row0 + i + j * 3 * nao];
        }
    }
    out
}

fn nearest_grid_value(zs: &[f64], z: f64) -> f64 {
    let n = zs.len();
    if n == 0 {
        return z;
    }
    if z <= zs[0] {
        return zs[0];
    }
    if z >= zs[n - 1] {
        return zs[n - 1];
    }
    let (mut lo, mut hi) = (0usize, n - 1);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if zs[mid] < z {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    if (z - zs[lo]) < (zs[hi] - z) {
        zs[lo]
    } else {
        zs[hi]
    }
}

// ---------------------------------------------------------------------------
// automatic real-axis grid (best practice)
// ---------------------------------------------------------------------------

/// Range margin of the automatic rule over the largest residue frequency the
/// requested state window actually reads.
pub const AUTO_GRID_RANGE_MARGIN: f64 = 2.5;
/// Smallest automatic grid range (Ha).
pub const AUTO_GRID_RANGE_FLOOR: f64 = 0.5;
/// Target node spacing (Ha) of the automatic rule.  Measured on
/// H2O/def2-SVP/PBE with the range capped at `2.5 z_need`: the frontier grid
/// error is 0.45 / 0.67 / 2.2 / 0.009 / 0.008 / 0.001 eV at 6 / 12 / 24 / 48 /
/// 96 / 192 nodes, so ~0.025 Ha puts the uniform rule in its converged regime
/// (`h = 0.025` -> 51 nodes for that case).  Note the non-monotonicity at
/// n = 24: the residual is still sensitive to where a node falls relative to
/// the RPA poles of `W`, which no spacing rule removes.
pub const AUTO_GRID_DELTA: f64 = 0.025;
/// Smallest automatic node count (the historical default).
pub const AUTO_GRID_N_MIN: usize = 6;
/// Largest automatic node count (cost cap).
pub const AUTO_GRID_N_MAX: usize = 400;

/// Symmetry-degenerate tie set of orbital `idx` within `lo..hi`.
fn tie_set(e: &[f64], idx: usize, lo: usize, hi: usize, tol: f64) -> Vec<usize> {
    (lo..hi).filter(|&j| (e[j] - e[idx]).abs() <= tol).collect()
}

/// Largest residue frequency the state window `nsemin..=nsemax` reads, and the
/// transition that attains it.
///
/// A QP state `n` (with `omega ~ e_n`) has active poles `m` on its own branch
/// with `de = |e_m - e_n| > 0`, and the residue evaluates the screened response
/// at `z = |e_m - omega|`.  This is the frequency range the low-rank factors
/// have to resolve — the **core-inflated `de_max` of the scan is usually far
/// larger** (the scan pairs every pole with every scan centre), and spreading
/// the nodes over that unused range makes the grid error stagnate at
/// O(0.1-1 eV) instead of converging.
///
/// Returns `(z_need, (p, q, sign, p_on_occupied_branch))` with
/// `d z_need/dR = sign * (eps1[p] - eps1[q])`.
pub fn residue_z_need(
    e: &[f64],
    nocc: usize,
    nsemin: usize,
    nsemax: usize,
) -> (f64, Option<(usize, usize, f64, bool)>) {
    let nmo = e.len();
    let mut z_need = 0.0f64;
    let mut winner: Option<(usize, usize, f64, bool)> = None;
    for n in nsemin..=nsemax {
        let occ = n < nocc;
        let (lo, hi) = if occ { (0, nocc) } else { (nocc, nmo) };
        for m in lo..hi {
            let de = if occ { e[m] - e[n] } else { e[n] - e[m] };
            if de > 0.0 && de > z_need {
                z_need = de;
                // occupied: `de = e_m - e_n` grows with `e_m`
                // virtual  : `de = e_n - e_m` grows with `e_n`
                winner = if occ {
                    Some((m, n, 1.0, true))
                } else {
                    Some((n, m, 1.0, false))
                };
            }
        }
    }
    (z_need, winner)
}

/// Node count of the automatic rule for a grid range.
fn auto_n(range: f64) -> usize {
    (((range / AUTO_GRID_DELTA).ceil() as usize) + 1).clamp(AUTO_GRID_N_MIN, AUTO_GRID_N_MAX)
}

/// Best-practice automatic real-axis grid, used when
/// `GwGradConfig::nomega_chi_real == 0` / `QuasiParticle::nomega_chi_real == 0`.
///
/// The rule is
///
/// ```text
/// z_need = max_n max_{active m} |e_m - e_n|          (state window)
/// range  = clamp(AUTO_GRID_RANGE_MARGIN * z_need, AUTO_GRID_RANGE_FLOOR, de_max)
/// n      = clamp(1 + ceil(range / AUTO_GRID_DELTA), AUTO_GRID_N_MIN, AUTO_GRID_N_MAX)
/// ```
///
/// i.e. the node count grows **linearly with the frequency range the residues
/// read**, not with `de_max` and not linearly with the number of levels (which
/// is what dilates `de_max` non-linearly through the deep core).  A user-pinned
/// `omega_chi_max > 0` wins over the range part of the rule (`n` is still
/// derived from it).
///
/// Returns `(n_points, grid_scale, mover)`; `None` when the rule does not apply
/// (`omega_chi_max` pins a range *and* an explicit `n` was given).
#[allow(clippy::too_many_arguments)]
pub fn auto_real_axis_grid(
    e: &[f64],
    nocc: usize,
    nsemin: usize,
    nsemax: usize,
    de_max: f64,
    de_mover: Option<GridMover>,
    omega_chi_max: f64,
    nomega_chi_real: usize,
    tie_tol: f64,
) -> Option<(usize, f64, Option<GridMover>)> {
    if nomega_chi_real != 0 {
        return None;
    }
    if omega_chi_max > 0.0 {
        return Some((auto_n(omega_chi_max), omega_chi_max, None));
    }
    let (z_need, winner) = residue_z_need(e, nocc, nsemin, nsemax);
    let want = (AUTO_GRID_RANGE_MARGIN * z_need).max(AUTO_GRID_RANGE_FLOOR);
    if !(want < de_max) || winner.is_none() {
        // the scan range is already tighter than the rule: keep it (and its
        // mover, whose `scale` is the `de_max` one)
        return Some((auto_n(de_max), de_max, de_mover));
    }
    let (p, q, sign, p_occ) = winner.unwrap();
    let p_set = if p_occ {
        tie_set(e, p, 0, nocc, tie_tol)
    } else {
        tie_set(e, p, nocc, e.len(), tie_tol)
    };
    let m_set = tie_set(e, q, 0, e.len(), tie_tol);
    Some((
        auto_n(want),
        want,
        Some(GridMover {
            p_set,
            m_set,
            sign,
            scale: AUTO_GRID_RANGE_MARGIN,
        }),
    ))
}

/// REST real-axis grid construction: de_max scan + linear/quadratic nodes.
///
/// Besides the nodes themselves this returns their **geometry dependence**, so
/// that the gradient can differentiate the grid-mediated part of the residue
/// term exactly (see [`RealAxisGrid`]).
pub struct RealAxisGrid {
    /// the grid nodes, in increasing order
    pub nodes: Vec<f64>,
    /// `d node_i / d de_max` of every node (all zero when the grid is pinned by
    /// `omega_chi_max > 0`)
    pub node_sens: Vec<f64>,
    /// The transition energy that attains the scan maximum (see
    /// [`GridMover`]).  `None` when `de_max` does not move with the geometry
    /// (`omega_chi_max > 0`) or when the scan found no transition above
    /// `res_tol`.
    pub mover: Option<GridMover>,
}

/// The scan maximum `de_max = 1.05 * |e_p - (e_m + i*step_sigma)| + 0.1` of
/// [`build_real_axis_grid`], together with the symmetry-degenerate orbitals
/// that tie for it.
///
/// `de_max` is a **max** over the scan, so at a degenerate geometry it is not
/// differentiable: displacing by `+h` follows the tied branch with the largest
/// derivative and `-h` the one with the smallest, and the central difference of
/// a max is the average of those two branches.  Storing the tied sets lets the
/// gradient reproduce exactly that.
#[derive(Clone, Debug)]
pub struct GridMover {
    /// tied orbitals on the pole side (`de` grows with their energy)
    pub p_set: Vec<usize>,
    /// tied scan centres (`de` shrinks with their energy)
    pub m_set: Vec<usize>,
    /// `+1` when the maximum is an occupied pole (`de = e_p - omega`), `-1` when
    /// it is a virtual pole (`de = omega - e_a`)
    pub sign: f64,
    /// `d(grid_scale)/d(Delta e)` where `Delta e = e_p - e_m` for the pole pair
    /// above: `1.05` for the `de_max = 1.05*D + 0.1` scan, and the range margin
    /// when the automatic rule caps the range at a multiple of `z_need`.
    pub scale: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn build_real_axis_grid(
    e: &[f64],
    nocc: usize,
    nomega_chi_real: usize,
    nsemin: usize,
    nsemax: usize,
    nomega_sigma: usize,
    step_sigma: f64,
    res_tol: f64,
    grid_type: usize,
    omega_chi_max: f64,
    tie_tol: f64,
) -> RealAxisGrid {
    let nmo = e.len();
    let mut de_max = 0.0f64;
    let mut winner: Option<(usize, usize, f64)> = None;
    for mstate in nsemin..=nsemax {
        let e0 = e[mstate];
        for io in -(nomega_sigma as isize)..=(nomega_sigma as isize) {
            let omega = e0 + io as f64 * step_sigma;
            for p in 0..nocc {
                let de = e[p] - omega;
                if de > res_tol && de > de_max {
                    de_max = de;
                    winner = Some((p, mstate, 1.0));
                }
            }
            for a in nocc..nmo {
                let de = omega - e[a];
                if de > res_tol && de > de_max {
                    de_max = de;
                    winner = Some((a, mstate, -1.0));
                }
            }
        }
    }
    // Tie sets of the winner: the maximum is attained by a whole
    // symmetry-degenerate shell whenever one is involved.
    let mover = winner.map(|(p, m, sign)| {
        let mut p_set = Vec::new();
        let mut m_set = Vec::new();
        if sign > 0.0 {
            for i in 0..nocc {
                if (e[i] - e[p]).abs() <= tie_tol {
                    p_set.push(i);
                }
            }
        } else {
            for a in nocc..nmo {
                if (e[a] - e[p]).abs() <= tie_tol {
                    p_set.push(a);
                }
            }
        }
        for j in 0..nmo {
            if (e[j] - e[m]).abs() <= tie_tol {
                m_set.push(j);
            }
        }
        GridMover { p_set, m_set, sign, scale: 1.05 }
    });
    de_max = de_max * 1.05 + 0.1;
    // Automatic mode (`nomega_chi_real == 0`) or the historical layout.
    // `omega_chi_max > 0` pins the node frequencies: the grid then does not
    // move with the reference geometry at all.
    let (n_points, grid_scale, pinned, mover) = match auto_real_axis_grid(
        e,
        nocc,
        nsemin,
        nsemax,
        de_max,
        mover.clone(),
        omega_chi_max,
        nomega_chi_real,
        tie_tol,
    ) {
        Some((n, scale, mv)) => (n, scale, false, mv),
        None if omega_chi_max > 0.0 => (nomega_chi_real, omega_chi_max, true, None),
        None => (nomega_chi_real, de_max, false, mover),
    };
    let mut nodes = Vec::with_capacity(n_points);
    let mut node_sens = Vec::with_capacity(n_points);
    for i in 0..n_points {
        if n_points > 1 {
            let t = i as f64 / (n_points - 1) as f64;
            if grid_type == 1 {
                nodes.push(grid_scale * t * t);
                node_sens.push(if pinned { 0.0 } else { t * t });
            } else {
                nodes.push(grid_scale * t);
                node_sens.push(if pinned { 0.0 } else { t });
            }
        } else {
            nodes.push(grid_scale);
            node_sens.push(if pinned { 0.0 } else { 1.0 });
        }
    }
    RealAxisGrid {
        nodes,
        node_sens,
        mover,
    }
}

/// Compact a **pair-symmetric** `[nmo*nmo, naux]` buffer (pair-slowest, `pair = row + col*nmo`)
/// in place to its lower triangle `[nmo*(nmo+1)/2, naux]`, returning the packed length.
///
/// Written in increasing packed order, so the destination offset never overtakes an unread source
/// (`tp <= pair`), and the tail is released with `truncate` + `shrink_to_fit`. Used by the adjoint
/// gradient for `bq`, which the pullback has already symmetrised: the fold reads only the symmetric
/// plane, so rebuilding it from the packed form is exact while `naux*nmo^2/2` doubles (437 MB at
/// naphthalene/def2-TZVP) leave the resident set before the fold — the phase this gradient peaks in.
/// Index of the unordered MO pair `{p, q}` in the **packed** (lower-triangle,
/// column-major) pair layout produced by [`pack_pairs_in_place`] and read by
/// [`GwGradEngine::fold_covector_packed`] and its streamed variant.
///
/// The layout is `for col { for row in col..nmo { … } }`, i.e. block `col`
/// starts at `col*nmo - col*(col-1)/2`.
#[inline]
pub fn packed_pair_index(nmo: usize, p: usize, q: usize) -> usize {
    let (col, row) = if p <= q { (p, q) } else { (q, p) };
    col * nmo - (col * col.wrapping_sub(1)) / 2 + (row - col)
}

/// Number of pairs in the packed layout of an `nmo`-dimensional pair space.
#[inline]
pub fn packed_pair_count(nmo: usize) -> usize {
    nmo * (nmo + 1) / 2
}

/// Expand a packed covector into the full symmetric `[nmo, nmo] x naux` layout
/// (the same value on both entries of each pair), i.e. the inverse of
/// [`pack_pairs_in_place`].  Provided for the legacy full-buffer contractions.
pub fn unpack_pairs(packed: &[f64], nmo: usize, naux: usize) -> Vec<f64> {
    assert_eq!(packed.len(), naux * packed_pair_count(nmo), "unpack_pairs: size mismatch");
    let mut v = vec![0.0f64; naux * nmo * nmo];
    let mut tp = 0usize;
    for col in 0..nmo {
        for row in col..nmo {
            let src = tp * naux;
            let dst = (row + col * nmo) * naux;
            v[dst..dst + naux].copy_from_slice(&packed[src..src + naux]);
            if row != col {
                let dst_t = (col + row * nmo) * naux;
                v[dst_t..dst_t + naux].copy_from_slice(&packed[src..src + naux]);
            }
            tp += 1;
        }
    }
    v
}

/// Pack a pair-symmetric full covector `[nmo, nmo] x naux` (aux fastest) into
/// its lower triangle `[nmo*(nmo+1)/2, naux]` **in place**, returning the packed
/// pair count.  The packed buffer holds the same value the full buffer stored at
/// each pair, so every consumer of the packed form
/// ([`GwGradEngine::fold_covector_packed`] and its streamed variant) sees exactly
/// the same covector while half the memory is resident.
pub fn pack_pairs_in_place(v: &mut Vec<f64>, nmo: usize, naux: usize) -> usize {
    let mut tp = 0usize;
    for col in 0..nmo {
        for row in col..nmo {
            let src = (row + col * nmo) * naux;
            let dst = tp * naux;
            if dst != src {
                v.copy_within(src..src + naux, dst);
            }
            tp += 1;
        }
    }
    v.truncate(tp * naux);
    v.shrink_to_fit();
    tp
}

fn symmetrise_pairs(v: &mut [f64], naux: usize, nmo: usize) {
    for pi in 0..nmo {
        for pj in (pi + 1)..nmo {
            let a = (pi + pj * nmo) * naux;
            let b = (pj + pi * nmo) * naux;
            for p in 0..naux {
                let s = 0.5 * (v[a + p] + v[b + p]);
                v[a + p] = s;
                v[b + p] = s;
            }
        }
    }
}

fn symmetrise_square(v: &mut [f64], n: usize) {
    for i in 0..n {
        for j in (i + 1)..n {
            let a = i + j * n;
            let b = j + i * n;
            let s = 0.5 * (v[a] + v[b]);
            v[a] = s;
            v[b] = s;
        }
    }
}

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

extern "C" {
    fn zgetrf_(
        m: *const i32,
        n: *const i32,
        a: *mut Complex<f64>,
        lda: *const i32,
        ipiv: *mut i32,
        info: *mut i32,
    );
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
fn cinv(b: Complex<f64>) -> Complex<f64> {
    let d = b.re * b.re + b.im * b.im;
    Complex::new(b.re / d, -b.im / d)
}

/// PySCF-convention residue screening diagonal
/// `2 [(z + delta + 2i eta)^-1 + (-z + delta)^-1]`, `delta = e_i - e_a`.
fn residue_d(delta: f64, z: f64, eta: f64) -> Complex<f64> {
    2.0 * (cinv(Complex::new(z + delta, 2.0 * eta)) + cinv(Complex::new(delta - z, 0.0)))
}

/// `d/dz` of the residue diagonal: `2 [-(z+delta+2i eta)^-2 + (z-delta)^-2]`.
fn residue_dz(delta: f64, z: f64, eta: f64) -> Complex<f64> {
    let t2 = Complex::new(z + delta, 2.0 * eta);
    2.0 * (-cinv(t2 * t2) + cinv(Complex::new((z - delta) * (z - delta), 0.0)))
}

/// `d/ddelta` of the residue diagonal:
/// `2 [-(z+delta+2i eta)^-2 - (z-delta)^-2]`.
fn residue_de(delta: f64, z: f64, eta: f64) -> Complex<f64> {
    let t2 = Complex::new(z + delta, 2.0 * eta);
    2.0 * (-cinv(t2 * t2) - cinv(Complex::new((z - delta) * (z - delta), 0.0)))
}

/// Static (z = 0) screening diagonal `2 [(delta + 2i eta)^-1 + delta^-1]`.
fn static_d0(delta: f64, eta: f64) -> Complex<f64> {
    2.0 * (cinv(Complex::new(delta, 2.0 * eta)) + cinv(Complex::new(delta, 0.0)))
}

/// `d/ddelta` of [`static_d0`]: `2 [-(delta+2i eta)^-2 - delta^-2]`.
fn static_d0e(delta: f64, eta: f64) -> Complex<f64> {
    let t2 = Complex::new(delta, 2.0 * eta);
    2.0 * (-cinv(t2 * t2) - cinv(Complex::new(delta * delta, 0.0)))
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
    lu: Vec<Complex<f64>>,
    ipiv: Vec<i32>,
}

impl CdLuFactor {
    fn factor(a_re: &[f64], a_im: &[f64], n: usize) -> Self {
        let mut lu: Vec<Complex<f64>> = a_re
            .iter()
            .zip(a_im.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();
        let mut ipiv = vec![0i32; n];
        let mut info = 0i32;
        let nn = n as i32;
        unsafe { zgetrf_(&nn, &nn, lu.as_mut_ptr(), &nn, ipiv.as_mut_ptr(), &mut info) };
        assert!(
            info == 0,
            "cd_gw_grad: complex screening factorisation failed (info={info})"
        );
        Self { n, lu, ipiv }
    }

    /// Overwrite the [n, ncol] col-major right-hand sides with `A^-1 B`.
    fn solve_in_place(&self, b: &mut [Complex<f64>], ncol: usize) {
        let (nn, nc) = (self.n as i32, ncol as i32);
        let mut info = 0i32;
        let trans = b'N' as c_char;
        unsafe {
            zgetrs_(
                &trans, &nn, &nc, self.lu.as_ptr(), &nn, self.ipiv.as_ptr(),
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
        // the full-CD path never touches the low-rank machinery
        let mut config = config;
        config.skip_lr_cache = true;
        let base = GwGradEngine::new_with_raw(scf, config, raw);
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
            sigma += s_m * (pole_factor * w_res - w0[m]);
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

    /// Solve the QP equation of the static-subtracted continuous energy
    /// and assemble the full response cache at the root.  With
    /// `GwGradConfig::qpe_graphical` the root comes from REST's production
    /// graphical linear-interpolation solver instead of the
    /// analytic-derivative Newton iteration (PySCF `qp_energy_subtracted`).
    /// One-step Newton ("linearized") quasiparticle energy for the full-CD
    /// engine — the counterpart of [`GwGradEngine::linearized_cache`].
    ///
    /// ```text
    /// E_lin = e_n + Z(e_n) [ consts_n + Re Sigma_sub(e_n) - e_n ],
    /// Z(e_n) = 1/(1 - d Re Sigma_sub/d omega)
    /// ```
    /// Full-CD static on-shell fallback: mean-field frequency and `Z = 1`
    /// (`E = consts_n + Re Sigma_sub(e_n)`), the branch used for deep levels and
    /// for a linearized step that is not a contraction.
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

    #[allow(dead_code)]
    /// Counterpart of [`GwGradEngine::linearized_guarded`] for the full-CD
    /// engine: a non-positive linearized `Z` is rejected and the static
    /// on-shell energy is used instead.
    fn linearized_guarded(&self, n: usize) -> CdQpCache {
        let e_n = self.base.e[n];
        let consts = self.consts(n);
        let (sigma, dsigma, _) = self.sigma_sub(e_n, n);
        let z = 1.0 / (1.0 - dsigma);
        if !(z > 0.0) {
            println!(
                "warning!!! cd_gw_grad orbital {n}: linearized (one-step Newton) Z = {z:.4} \
                 is not positive (step would be {:+.3e} Ha); using the static on-shell energy",
                z * (consts + sigma - e_n)
            );
            return self.static_fallback_cache(n);
        }
        self.linearized_cache(n)
    }

    #[allow(dead_code)]
    fn linearized_cache(&self, n: usize) -> CdQpCache {
        let e_n = self.base.e[n];
        let consts = self.consts(n);
        let (sigma, dsigma, mut cache) = self.sigma_sub(e_n, n);
        let z = 1.0 / (1.0 - dsigma);
        let shift = consts + sigma - e_n;
        cache.omega = e_n + z * shift;
        cache.freq = e_n;
        cache.z_factor = z;
        cache.dsigma_domega = dsigma;
        cache.static_fallback = false;
        cache.linearized = true;
        cache.lin_shift = shift;
        cache
    }

    pub fn build_target_cache(&self, n: usize) -> CdQpCache {
        let e_n = self.base.e[n];
        let side = if n < self.base.nocc { -1.0 } else { 1.0 };
        let consts = self.consts(n);
        let t0 = Instant::now();
        // REST `gw_switch_fallback_threshold`: deep levels use the static
        // on-shell energy `consts_n + Re Sigma_sub(e_n)` (`Z = 1`).
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
            // The `gradient_z_min` rule asks for the linearized (one-step
            // Newton) functional, whose analytic derivative needs the
            // omega-derivative pullback channel that only the low-rank engine
            // implements.  Rather than differentiate a branch this engine does
            // not support, keep the converged root (whose gradient **is**
            // implemented) and say so.
            println!(
                "warning!!! cd_gw_grad orbital {n}: converged Z = {z_root:.4} is below \
                 z_min = {:.3}, but the full-CD gradient does not implement the \
                 linearized (one-step Newton) branch; keeping the root.  Use \
                 gradient_engine = \"lr\" for the linearized branch.",
                self.base.config.z_min
            );
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
        // The full-CD gradient has no omega-derivative channel yet: a linearized
        // cache would silently produce a wrong gradient, so refuse loudly.
        for c in caches.iter() {
            assert!(
                !c.linearized,
                "gw_grad: the full-CD GW analytic gradient does not implement the \
                 omega-derivative channel of the linearized (one-step Newton) QP \
                 branch (orbital {} fell back to it).  Set \
                 [quasiparticle_methods] gradient_z_min = 0 to disable that branch, \
                 or use gradient_engine = \"lr\".",
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
                if (1.0 - base.fock_hyb).abs() > 1.0e-14 {
                    let na = naux;
                    let ex_scale = c * (1.0 - base.fock_hyb);
                    for i in 0..nocc {
                        let qni = base.with_q(|q| base.gather_pair(q, n, i));
                        let g_ni = base.metric_solve(&qni, 1);
                        let col_i = self.pair_col(n, i) * na;
                        for p in 0..na {
                            bq[k][col_i + p] -= 2.0 * ex_scale * g_ni[p];
                        }
                        for p in 0..na {
                            for q in 0..na {
                                bj[k][p + q * na] += ex_scale * g_ni[p] * g_ni[q];
                            }
                        }
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
    /// [`GwGradEngine::analytic_gradient`].
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

    /// Free the geometry-fixed screening factors.  They are only needed
    /// while QP caches are built ([`GwCdGradEngine::build_target_cache`]);
    /// calling this before the gradient phase (`analytic_gradient_with_cache`)
    /// lowers the peak memory.  Any further cache build afterwards would
    /// panic.
    pub fn release_screening(&mut self) {
        self.screening.chol_imag = Vec::new();
        self.screening.lu_static.lu = Vec::new();
        self.screening.lu_static.ipiv = Vec::new();
    }

    /// QP energy of one orbital (the scalar whose gradient is returned by
    /// [`GwCdGradEngine::analytic_gradient`]).
    pub fn qp_energy_of(&self, target: usize) -> f64 {
        self.build_target_cache(target).omega
    }

    /// Validation helper: the response cache (response vectors, `W(iu)`,
    /// `W(0)` and the active residues) evaluated at a **fixed** external
    /// frequency, without the QP root search.  Together with the public
    /// cache fields this lets a test rebuild each phase of the self-energy
    /// and finite-difference it independently.
    pub fn cache_at(&self, omega: f64, n: usize) -> CdQpCache {
        let (_sigma, dsigma, mut cache) = self.sigma_sub(omega, n);
        cache.z_factor = 1.0 / (1.0 - dsigma);
        cache
    }

    /// Validation helper: `Re Sigma_sub(omega)` and `d Re Sigma_sub/d omega`
    /// at a **fixed** external frequency (no QP root search).
    pub fn sigma_sub_at(&self, omega: f64, n: usize) -> (f64, f64) {
        let (sigma, dsigma, _cache) = self.sigma_sub(omega, n);
        (sigma, dsigma)
    }

    /// Validation helper: the pullback of one cache **without** the `Z`
    /// renormalisation, i.e. `d consts_n/dR + d Re Sigma_sub/dR` at the cache
    /// frequency.  This is the quantity a finite difference of
    /// `e_n(R) + Re Sigma_sub(omega_ref, R)` (fixed `omega_ref`) must
    /// reproduce, so it isolates the pullback from the `Z` factor / the
    /// omega-dependence of the QP equation.
    pub fn fixed_omega_gradient(&self, cache: &CdQpCache) -> Vec<f64> {
        self.cd_gradient_with_caches(&[cache], &[vec![1.0 / cache.z_factor]])
    }
}

// ===========================================================================
// Adjoint ("Z-vector") evaluation of the CP-HF channel
// ===========================================================================
//
// After the `qx` channel has been folded, the whole gradient still needing the
// per-perturbation CP-HF response is
//
//     G_d = 2 Tr[u_d^T W] + sum_p be[p] eps1_d[p] + <B, dI_d> + bJ:dJ_d + bB.b_x_d
//
// with `u_d`, `eps1_d` the Roothaan response of perturbation `d`.  Writing the
// Roothaan formulas out,
//
//   u[p,q] = (f1_mo[p,q] - e_q s1_mo[p,q]) / (e_q - e_p)   (p != q),
//   u[p,p] = -s1_mo[p,p]/2,   eps1[p] = f1_mo[p,p] - e_p s1_mo[p,p],
//   f1_mo  = C^T (dh1 + Jd - Kd/2 + vind(dm1)) C,
//   dm1    = 2 sum_i (c1_i c_i^T + c_i c1_i^T),   c1 = C mo1,
//
// the response part is a **linear functional of `mo1` with a
// perturbation-independent covector**:
//
//   G_d^resp = <Lambda, f1_mo_d> + <Lambda_s, s1_mo_d>
//            = <Lam_ao, dh1_d> + <Lam_ao, Jd_d - Kd_d/2> + Tr[Vtil dm1_d]
//              + <Lam_ao, vind(dm1_d)> + <Lam_s,ao, dS_d>            (1)
//
// with (W the folded rotation factor of the qx channel)
//
//   Lambda[p,q]   = 2 W[p,q]/(e_q - e_p)  (p != q),  Lambda[p,p] = be[p]
//   Lambda_s[p,q] = -2 W[p,q] e_q/(e_q - e_p) (p != q),
//   Lambda_s[p,p] = -W[p,p] - be[p] e_p
//   Lam_ao = C Lambda C^T,  Lam_s,ao = C Lambda_s C^T
//   Vtil   = vind(Lam_ao) = J[Lam_ao] - fock_hyb/2 K[Lam_ao]     (self-adjoint)
//
// and then, since `mo1_d = S rhs_d` with `S` the CP-HF solver map,
//
//   Tr[Vtil dm1_d] = <L, mo1_d> = <Z, rhs_d>,   L[p,i] = 4 (C^T Vtil C_occ)[p,i]
//   Z = S^T L                                                     (2)
//
// so **one** adjoint solve replaces the `3 natm`-column Krylov solve.
//
// The two skeleton contractions in (1) are also linear in the derivative
// integrals, so they are folded into precomputed `[naux,nao,nao]` / `[naux,naux]`
// tensors:
//
//   <Lam_ao, Jd_d>  = sum_P a_P c_P + sum_P lam_P dc_P ,
//        a_P = sum_munu Lam_ao dI[P], lam_P = sum_munu Lam_ao I[P],
//        c = J^-1 rho, dc = J^-1 (drho - dJ c), rho = I^T dm, drho = dI^T dm
//   <Lam_ao, Kd_d>  = 4 <Theta, dI_d> - 2 <Psi, dJ_d> ,
//        Theta_P = Lam_ao G_P Cocc^T ,  G_P = (J^-1 Y)_P ,  Y[mu,P] = (mu i|P) C_i
//        Psi = J^-1 Phi ,  Phi[P,P'] = sum_i Y_i^T Lam_ao G_i
//
// Cost: all precomputations are `O(naux nao^3)`-at-most once per geometry; each
// perturbation then costs one streamed pass over the `naux` `[nao,nao]` `dI`
// slices plus `O(naux^2 + nmo nocc)` scalars, i.e. `O(naux nao^2)`.  Total
// `O(naux nao^3 + 3 natm naux nao^2)` instead of `O(3 natm (naux nao^2 nocc +
// naux nao^3 + ...))`.
//
// Only non-degenerate references are supported: inside a symmetry-degenerate
// shell the response is defined through the eigen-decomposition of the shell
// block, which is *not* linear in the perturbation.

/// Per-geometry adjoint data of the CP-HF channel; see the section header.
pub struct CphfAdjoint {
    /// `[naux, nao, nao]`: **symmetrised** coefficient of `dI`,
    /// `(B - 2 fock_hyb Theta + c_P Lam_ao) + transpose`.  The Coulomb
    /// `sum_P a_P c_P` term (`a_P = <Lam_ao, dI_P>`) is folded in, and the
    /// symmetrisation lets both `int3c2e_ip1` bra centres be contracted in a
    /// single pass (see [`GwGradEngine::adjoint_perturbation_gradient_local`]).
    pub di_cov: Vec<f64>,
    /// `[nao, nao]`: `dm + dm^T` (the SCF density is symmetric, so this is
    /// `2 dm`); used together with the factor `1/2` of the symmetrised slicing.
    pub dm_sym: Vec<f64>,
    /// `[naux]`: `J^-1 lambda`; replaces the per-perturbation metric solve by
    /// `sum_P lambda_P dc_P = sum_P lam_tilde_P (drho - dJ c)_P`.
    pub lam_tilde: Vec<f64>,
    /// `[nao, nao]`: `Lam_ao` (coefficient of `dh1`).
    pub lam_ao: Vec<f64>,
    /// `[nao, nao]`: `Lam_s,ao` (coefficient of `dS`, col-major like `dS`).
    pub lams_ao: Vec<f64>,
    /// `[naux]`: `sum_munu Lam_ao I[mu,nu,P]`.
    pub lam_aux: Vec<f64>,
    /// `[naux, naux]`: `Psi` (coefficient of `dJ`).
    pub psi: Vec<f64>,
    /// `[naux]`: `J^-1 rho`, `rho_P = sum_munu I[mu,nu,P] dm[mu,nu]`.
    pub c_aux: Vec<f64>,
    /// `[nao, nao]` SCF density (col-major).
    pub dm: Vec<f64>,
    /// `[3 natm]`: `<Z, rhs_d>` (only filled when `fold_cphf_rhs` is off, for
    /// cross-validation against the folded route).
    pub rhs_dot: Vec<f64>,
    /// whether `<Z, rhs_d>` was folded into the covectors (`false` = the
    /// reference route, which keeps `rhs_dot` and must not add the `W` terms).
    pub fold_rhs: bool,
    /// `[nao, nao]`: `W = Zbra C_occ^T`, the AO covector replacing `<Z, rhs_d>`
    /// (coefficient of `dh1`).
    pub w_dh1: Vec<f64>,
    /// `[nao, nao]`: coefficient of `dS`.
    pub w_ds: Vec<f64>,
    /// flattened `[nmo, nocc, 3, natm]` analdrv skeleton Fock derivative (only
    /// filled when `fold_cphf_rhs` is off; used by the verification tests).
    pub f1mo_ref: Vec<f64>,
    /// exchange scale of the skeleton Fock derivative (always 0.5 here)
    pub half_k: f64,
    /// `[naux, nao, nao]`: exchange skeleton adjoint tensor `Theta`, kept **only** for
    /// verification of the piecewise identities when
    /// [`GwGradConfig::keep_adjoint_pieces`] is set; empty otherwise, because it is folded
    /// into [`Self::di_cov`] in place and the per-direction contraction never reads it.
    pub theta: Vec<f64>,
    /// wall-clock of the sub-phases (seconds):
    /// [0] covectors Lambda + Vtil = vind(Lam_ao), [1] analdrv object setup,
    /// [2] make_response_preparation, [3] compute_dimless_cpscf_rhs,
    /// [4] adjoint (Z) solve + <Z,rhs>, [5] skeleton foldings (Theta, Phi/Psi)
    pub timings: [f64; 6],
    /// verification fields (MO-basis covectors, relaxed-response operator, L and Z)
    pub lam_mo: Vec<f64>,
    pub lams_mo: Vec<f64>,
    pub vtil: Vec<f64>,
    pub l_mat: Vec<f64>,
    pub z_vec: Vec<f64>,
}

impl GwGradEngine<'_> {
    /// Whether the adjoint (Z-vector) CP-HF route applies to this reference.
    ///
    /// The route is the linear-response CP-HF solution of a **plain HF**
    /// reference; for a DFT reference it declines and the gradient falls back to
    /// the per-perturbation route. Callers check this *before* building the
    /// folded covector, so that the `naux * nmo^2` covector buffer can be
    /// released as soon as the fold has consumed it (see
    /// [`Self::analytic_gradient`]).
    ///
    /// This is the one predicate behind [`Self::build_cphf_adjoint`]'s early
    /// return: if that function ever declines for another reason, callers that
    /// rely on it must be revisited too.
    pub fn supports_cphf_adjoint(&self) -> bool {
        self.scf.mol.xc_data.dfa_compnt_scf.is_empty()
    }

    /// Build the adjoint data (see the section header).  Returns `None` when the
    /// reference has symmetry-degenerate shells (the route is not linear there)
    /// or when the reference is not a plain HF one.
    pub fn build_cphf_adjoint(
        &self,
        fold: &mut FoldedCovector,
        be_k: &[f64],
    ) -> Option<CphfAdjoint> {
        use crate::analdrv::prelude::{
            AnalDrvConfig, HessNucAPI, RHessCoreAPI, RHessElecInteractAPI, RHessHcore,
            RHessOvlp, RHessSCF,
        };
        use crate::analdrv::response::rresp_interface::rscf_resp_interface;
        use crate::ri_jk::hess_r::RHessRIJK;
        use crate::ri_jk::util::{get_cint_aux, get_cint_mol};
        use crate::utilities::rstsr_util::RestTensorToRstsrViewAPI;
        use rstsr::prelude::*;

        // Symmetry-degenerate shells are supported: for an equal-weight shell
        // average the covectors are isotropic on the shell (Schur's lemma), so
        // the shell block of the Roothaan response reduces to its trace,
        //     sum_i be[p_i] eps1[p_i] + 2 sum_ij W[p_i,p_j] u[p_i,p_j]
        //         = be * Tr[A] - w * sum_i s1[p_i,p_i] ,   A_ij = f1_ij - e0 s1_ij,
        // which is linear in the perturbation.  That is exactly what the
        // "skip the shell-internal off-diagonal pairs" rule below computes.
        if !self.supports_cphf_adjoint() {
            return None;
        }

        let nao = self.nao;
        let nmo = self.nmo;
        let nocc = self.nocc;
        let naux = self.naux;
        let n2 = nao * nao;
        let e = &self.e;

        // ---- (a) MO covectors Lambda, Lambda_s ---------------------------
        let mut lam = vec![0.0f64; nmo * nmo];
        let mut lams = vec![0.0f64; nmo * nmo];
        // NB: `f1_mo`/`s1_mo` are symmetric, so the covector is defined through
        // the full sum `sum_pq Lam[p,q] f1[p,q]`; each ordered pair (p,q) and
        // (q,p) of the Roothaan formulas contributes to the same matrix element.
        for p in 0..nmo {
            lam[p + p * nmo] = be_k[p];
            lams[p + p * nmo] = -fold.w_rot[p + p * nmo] - be_k[p] * e[p];
            for q in 0..nmo {
                // `p == q` and shell-internal pairs have `e_q == e_p`; the latter
                // are handled by the shell trace (see the guard comment above).
                if p == q || self.same_degenerate_shell(p, q) {
                    continue;
                }
                let den = e[q] - e[p];
                let wpq = fold.w_rot[p + q * nmo];
                let wqp = fold.w_rot[q + p * nmo];
                lam[p + q * nmo] = (wpq - wqp) / den;
                lams[p + q * nmo] = (e[p] * wqp - e[q] * wpq) / den;
            }
        }
        // Lam_ao = C Lam C^T, Lam_s,ao = C Lam_s C^T
        let to_ao = |m: &[f64]| -> Vec<f64> {
            let mm = to_mat(m, nmo, nmo);
            let mut tmp = MatrixFull::new([nao, nmo], 0.0);
            let mut out = MatrixFull::new([nao, nao], 0.0);
            _dgemm_full(&self.c_mo, 'N', &mm, 'N', &mut tmp, 1.0, 0.0);
            _dgemm_full(&tmp, 'N', &self.c_mo, 'T', &mut out, 1.0, 0.0);
            out.data
        };
        let lam_ao = to_ao(&lam);
        let lams_ao = to_ao(&lams);

        // ---- (b) Vtil = vind(Lam_ao) and L = 4 C^T Vtil C_occ ------------
        let lam_ao_m = to_mat(&lam_ao, nao, nao);
        let j_up = crate::dft::response::compute_j_upper(self.scf, &vec![lam_ao_m.clone()]);
        let k_up = crate::dft::response::compute_k_upper(self.scf, &vec![lam_ao_m]);
        let j_full = j_up.to_matrixfull().expect("adjoint: J upper").data;
        let k_full = k_up.to_matrixfull().expect("adjoint: K upper").data;
        let mut vtil = vec![0.0f64; n2];
        for k in 0..n2 {
            vtil[k] = j_full[k] - 0.5 * self.fock_hyb * k_full[k];
        }
        let cocc: Vec<f64> = (0..nocc)
            .flat_map(|i| (0..nao).map(move |mu| self.c_mo[[mu, i]]))
            .collect();
        let l_mat = {
            let vm = to_mat(&vtil, nao, nao);
            let cm = to_mat(&cocc, nao, nocc);
            let mut tmp = MatrixFull::new([nao, nocc], 0.0);
            let mut out = MatrixFull::new([nmo, nocc], 0.0);
            _dgemm_full(&vm, 'N', &cm, 'N', &mut tmp, 1.0, 0.0);
            _dgemm_full(&self.c_mo, 'T', &tmp, 'N', &mut out, 4.0, 0.0);
            out.data
        };

        // ---- (c) one adjoint CP-HF solve -> Z, and <Z, rhs_d> -------------
        let t_all = Instant::now();
        let mut timings = [0.0f64; 6];
        timings[0] = 0.0; // (a)+(b) are timed by the caller-visible total below
        let t_setup = Instant::now();
        let _ = t_all;
        let device = DeviceBLAS::default();
        let mol = get_cint_mol(&self.scf.mol);
        let aux = get_cint_aux(&self.scf.mol);
        let mo_coeff = rt::asarray((&self.c_mo.data, [nao, nmo], &device)).into_contig(ColMajor);
        let mo_occ =
            rt::asarray((&self.scf.occupation[0], [nmo], &device)).into_contig(ColMajor);
        let mo_energy = rt::asarray((&e[..], [nmo], &device)).into_contig(ColMajor);
        let mut ovlp_obj = RHessOvlp::new(&mol, &device);
        let mut hcore_obj = RHessHcore::new(&mol, &device);
        let rimatr = &self
            .scf
            .rimatr
            .as_ref()
            .expect("gw_grad: adjoint CP-HF requires rimatr (use_auxbas + use_ri_symm)")
            .0;
        let cderi = rimatr.to_rstsr_view(&device).into_cow();
        let j2c_decomp =
            crate::ri_jk::get_j2c_decomp(&aux, &device, self.scf.mol.ctrl.j2c_decomp.clone());
        let mut rijk_obj = RHessRIJK::new_with_cderi(&mol, &aux, 1.0, 1.0, cderi, j2c_decomp);
        let mut config = AnalDrvConfig::default();
        config.resp.cache_kernel = self.config.resp_cache_kernel;
        config.resp.cache_budget = self.config.resp_cache_budget_mib;
        // Everything from here to the end of the Z-vector correction lives in its own scope: the
        // response objects (and with them the RI-JK response-kernel cache, `nao^2 * naux` doubles —
        // 1030 MB at naphthalene/def2-TZVP) are released before the skeleton contractions below,
        // which never touch them. `z_vec`/`rhs_dot`/the timings are the only outputs needed later.
        let mut z_vec: Vec<f64>;
        let rhs_dot: Vec<f64>;
        let nvir: usize;
        let e_ai_shift: Vec<f64>;
        let f1mo_ref: Option<Vec<f64>>;
        let fold_rhs = self.config.fold_cphf_rhs;
        {
        let mut resp = rscf_resp_interface(self.scf, &config);
        timings[1] = t_setup.elapsed().as_secs_f64();

        timings[0] = t_all.elapsed().as_secs_f64() - t_setup.elapsed().as_secs_f64();
        let t_rhs = Instant::now();
        let rhs_full_opt = {
            let nuc_list: Vec<&mut dyn HessNucAPI> = Vec::new();
            let core_list: Vec<&mut dyn RHessCoreAPI> = vec![&mut hcore_obj];
            let el_list: Vec<&mut dyn RHessElecInteractAPI> = vec![&mut rijk_obj];
            let mut hess = RHessSCF::new(
                mo_coeff, mo_occ, mo_energy, &mut ovlp_obj, nuc_list, core_list, el_list,
                &mut resp, &config.nucgrad,
            );
            hess.make_response_preparation();
            timings[2] = t_rhs.elapsed().as_secs_f64();
            let t2 = Instant::now();
            if fold_rhs {
                // only `f1mo`/`s1mo` are unavailable here; the skeleton tensor
                // itself is never needed (the `Z` contraction is folded below)
                (None, None)
            } else {
                let rhs_map = hess.compute_dimless_cpscf_rhs();
                timings[3] = t2.elapsed().as_secs_f64();
                let f1 = rhs_map["f1mo"].to_owned();
                let fr = f1.raw()[f1.offset()..f1.offset() + nmo * nocc * 3 * self.natm].to_vec();
                (Some(rhs_map["rhs"].to_owned()), Some(fr))
            }
        };
        let (rhs_full, f1mo_ref_inner) = rhs_full_opt;
        f1mo_ref = f1mo_ref_inner;

        let t_solve = Instant::now();
        // dimensionless rhs for the adjoint: `rhs_vir = L / (e_a - e_i)`, occ = 0
        {
            let st = resp.cpscf_state();
            let nv = st.nmo - st.nocc;
            let cont = st.e_ai_shift.to_owned().into_contig(ColMajor);
            let s = cont.raw()[cont.offset()..cont.offset() + nv * st.nocc].to_vec();
            nvir = nv;
            e_ai_shift = s;
        }
        let mut rhs_z = rt::zeros(([nmo, nocc].f(), &device));
        {
            let mut buf = rhs_z.i_mut((rt::slice!(nocc, nmo), ..));
            let mut src = vec![0.0f64; nvir * nocc];
            for i in 0..nocc {
                for a in 0..nvir {
                    src[a + i * nvir] = l_mat[(nocc + a) + i * nmo] / e_ai_shift[a + i * nvir];
                }
            }
            let t = rt::asarray((&src, [nvir, nocc], &device)).into_contig(ColMajor);
            buf.assign(&t);
        }
        let t_asm = t_solve.elapsed();
        let t_sv = Instant::now();
        let z_raw = resp.solve_dimless_cpscf(rhs_z.view()); // [nmo, nocc]
        let t_sv = t_sv.elapsed();
        let z_cont = z_raw.into_contig(ColMajor);
        z_vec = vec![0.0f64; nmo * nocc];
        z_vec.copy_from_slice(&z_cont.raw()[z_cont.offset()..z_cont.offset() + nmo * nocc]);
        let t_occ0 = Instant::now();
        let z_occ_corr = {
            let mut x = rt::zeros(([nmo, nocc].f(), &device));
            let mut zv = vec![0.0f64; nvir * nocc];
            for i in 0..nocc {
                for a in 0..nvir {
                    zv[a + i * nvir] = z_vec[(nocc + a) + i * nmo];
                }
            }
            let t = rt::asarray((&zv, [nvir, nocc], &device)).into_contig(ColMajor);
            x.i_mut((rt::slice!(nocc, nmo), ..)).assign(&t);
            let r = resp.response_mo(x.view()).into_contig(ColMajor);
            let rr = r.raw();
            let ro = r.offset();
            // occ block of an [nmo, nocc] col-major tensor: rows 0..nocc
            let mut out = vec![0.0f64; nocc * nocc];
            for i in 0..nocc {
                for p in 0..nocc {
                    out[p + i * nocc] = rr[ro + p + i * nmo];
                }
            }
            out
        };
        for i in 0..nocc {
            for p in 0..nocc {
                z_vec[p + i * nmo] = l_mat[p + i * nmo] - z_occ_corr[p + i * nocc];
            }
            // Z_vir = D (D + A)^-1 L_vir = D * x_vir (D symmetric)
            for a in 0..nvir {
                z_vec[(nocc + a) + i * nmo] *= e_ai_shift[a + i * nvir];
            }
        }

        mem_probe("  adjoint: after Z solve");
        if std::env::var("REST_GWGRAD_ZDBG").is_ok() {
            eprintln!(
                "[zdbg] rhs-assembly {:.3}s solve {:.3}s occ-corr {:.3}s",
                t_asm.as_secs_f64(),
                t_sv.as_secs_f64(),
                t_occ0.elapsed().as_secs_f64()
            );
        }
        // <Z, rhs_d> for every perturbation (skipped on the folded route)
        rhs_dot = match &rhs_full {
            Some(rhs_full) => {
                let zt = rt::asarray((&z_vec, [nmo, nocc], &device)).into_contig(ColMajor);
                let prod = rhs_full * &zt.i((.., .., None, None));
                let s = prod.sum_axes([0, 1]); // [3, natm]
                let sr = s.raw();
                let so = s.offset();
                sr[so..so + 3 * self.natm].to_vec()
            }
            None => vec![0.0f64; 3 * self.natm],
        };

        timings[4] = t_solve.elapsed().as_secs_f64();
        }
        let t_skel = Instant::now();
        // ---- (c2) fold `sum_d <Z, rhs_d>` into AO covectors -------------
        //
        // `rhs_d` is built from the skeleton Fock/overlap derivatives,
        //   rhs[a,i] = -(f1mo[a,i] - s1mo[a,i] e_i)/D_ai ,  rhs[j,i] = -s1mo[j,i]/2,
        // so with `Zf[p,i] = -Z[p,i]/D_pi` (virtual p) and
        // `Zs[a,i] = Z[a,i] e_i/D_ai`, `Zs[j,i] = -Z[j,i]/2` (occupied j),
        //   sum_d <Z, rhs_d> = sum_{mu i} Zbra[mu,i] f1bra[mu,i] + <Zsbra, s1mo>
        // with `Zbra = C Zf`, `Zsbra = C Zs`.  Writing `f1bra = dh1·C_occ +
        // (Jd - Kd/2)·C_occ` and `s1mo = C^T dS C_occ` gives
        //   sum_d <Z, rhs_d> = <W, dh1_d + Jd_d - Kd_d/2> + <W_s, dS_d> ,
        //   W = Zbra C_occ^T ,  W_s = Zsbra C_occ^T ,
        // i.e. the same `dh1`/`dJ`/`dK`/`dS` contractions already built for
        // `Lambda_ao`, so `W` is simply added into those folded covectors.
        let (w_dh1, w_ds) = {
            let mut zf = vec![0.0f64; nmo * nocc];
            let mut zs = vec![0.0f64; nmo * nocc];
            for i in 0..nocc {
                for a in 0..nvir {
                    let p = nocc + a;
                    let d = e_ai_shift[a + i * nvir];
                    let z = z_vec[p + i * nmo];
                    zf[p + i * nmo] = -z / d;
                    zs[p + i * nmo] = z * e[i] / d;
                }
                for j in 0..nocc {
                    zs[j + i * nmo] = -0.5 * z_vec[j + i * nmo];
                }
            }
            let cm = to_mat(&cocc, nao, nocc);
            let to_ao = |m: &[f64]| -> Vec<f64> {
                let mm = to_mat(m, nmo, nocc);
                let mut tmp = MatrixFull::new([nao, nocc], 0.0);
                let mut out = MatrixFull::new([nao, nao], 0.0);
                _dgemm_full(&self.c_mo, 'N', &mm, 'N', &mut tmp, 1.0, 0.0);
                _dgemm_full(&tmp, 'N', &cm, 'T', &mut out, 1.0, 0.0);
                out.data
            };
            (to_ao(&zf), to_ao(&zs))
        };

        // ---- (d) folded skeleton tensors --------------------------------
        // Each contraction over the raw three-centre integrals is accumulated **one auxiliary-atom
        // block at a time**, so the full `nao^2 * naux` array (874 MB at naphthalene/def2-TZVP) is
        // never resident — this phase used to be the gradient's peak. The integral work is the same
        // shell-triplet sum as one `int3c2e` call, only split into `natm` calls.
        //
        //   Lam_aux = I^T Lam_ao   [naux]      rho = I^T dm   [naux]
        //   Y^T [naux, nocc*nao]   (G^T = J^-1 Y^T)
        let dm = self.scf.density_matrix[0].data.clone();
        let mut lam_aux = vec![0.0f64; naux];
        let mut rho = vec![0.0f64; naux];
        // `Lam_aux,W = I^T W` (the `Z` covector's partner) rides the same pass: one sweep over the
        // integral blocks serves all four contractions, so the shell triplets are evaluated once.
        let mut lam_aux_w = vec![0.0f64; naux];
        let mut yt = vec![0.0f64; naux * nocc * nao];
        {
            let cm = to_mat(&cocc, nao, nocc);
            let mut ip = MatrixFull::new([nao, nao], 0.0);
            let mut block = MatrixFull::new([nao, nocc], 0.0);
            self.raw.for_each_i3_atom_block(|p0, nb, blk| {
                lam_aux[p0..p0 + nb].copy_from_slice(&gemm_nt(blk, n2, nb, &lam_ao, n2, 1));
                rho[p0..p0 + nb].copy_from_slice(&gemm_nt(blk, n2, nb, &dm, n2, 1));
                lam_aux_w[p0..p0 + nb].copy_from_slice(&gemm_nt(blk, n2, nb, &w_dh1, n2, 1));
                for pl in 0..nb {
                    ip.data.copy_from_slice(&blk[pl * n2..(pl + 1) * n2]);
                    _dgemm_full(&ip, 'N', &cm, 'N', &mut block, 1.0, 0.0);
                    for col in 0..nao * nocc {
                        yt[(p0 + pl) + col * naux] = block.data[col];
                    }
                }
            });
        }
        let c_aux = self.solve_metric(&rho);
        let mut gt = yt.clone();
        self.solve_metric_batch(&mut gt, nao * nocc);

        // Folded exchange adjoint of an arbitrary AO covector `x_ao`:
        //   <x_ao, Kd> = 4 <Theta_x, dI> - 2 <Psi_x, dJ> ,
        //   Theta_x[P] = x_ao G_P Cocc^T ,  Phi_x[P,P'] = sum_i Y_i^T x_ao G_i ,
        //   Psi_x = J^-1 Phi_x .
        let cm = to_mat(&cocc, nao, nocc);
        // `x_theta` and `x_phi` are separate because `<X,Kd> = 2<Theta_{X+X^T},dI>
        // - 2<Psi_X,dJ>`: the `dY G^T` term symmetrises `X` (the partner is its
        // transpose) while the `Y G2^T` term does not.
        // The `dI` covector is `B - 2*Theta + c_P Lam_ao - 2*Theta_W + c_P W`, i.e. a sum of
        // two `[naux, nao, nao]` tensors plus rank-one-in-`P` terms. `Theta` and `B` are both
        // `[naux, nao, nao]` and neither is read afterwards, so the exchange fold accumulates
        // **into `B`'s buffer**: `b_ao` is taken out of `fold` and becomes `di_cov`. That keeps
        // a second `naux * nao^2` tensor (874 MB at naphthalene/def2-TZVP) out of the resident
        // set of the whole adjoint build and of the per-direction loop that follows.
        let mut di_cov = std::mem::take(&mut fold.b_ao);
        let keep_pieces = self.config.keep_adjoint_pieces;
        let mut theta_pieces: Vec<f64> = if keep_pieces { vec![0.0f64; naux * n2] } else { Vec::new() };
        let wfac = if fold_rhs { 1.0 } else { 0.0 };
        let mut fold_exchange = |x_theta: &[f64], x_phi: &[f64]| -> Vec<f64> {
            let xm = to_mat(x_theta, nao, nao);
            let mut gp = MatrixFull::new([nao, nocc], 0.0);
            let mut xb = MatrixFull::new([nao, nocc], 0.0);
            let mut th = MatrixFull::new([nao, nao], 0.0);
            for p in 0..naux {
                for i in 0..nocc {
                    for nu in 0..nao {
                        gp.data[nu + i * nao] = gt[p + (i * nao + nu) * naux];
                    }
                }
                _dgemm_full(&xm, 'N', &gp, 'N', &mut xb, 1.0, 0.0);
                _dgemm_full(&xb, 'N', &cm, 'T', &mut th, 1.0, 0.0);
                if keep_pieces {
                    theta_pieces[p * n2..(p + 1) * n2].copy_from_slice(&th.data);
                }
                // in-place: B_P -= 2 Theta_P, then add c_P Lam_ao (+ c_P W); the accumulated
                // value is symmetric in (mu, nu) (`a + at` and `at + a` are the same
                // floating-point number), so each unordered index pair is written once.
                let cp = c_aux[p];
                let off = p * n2;
                for mu in 0..nao {
                    for nu in 0..=mu {
                        let k = mu + nu * nao;
                        let kt = nu + mu * nao;
                        let a = di_cov[off + k] - 2.0 * th.data[k] + cp * lam_ao[k]
                            + wfac * cp * w_dh1[k];
                        let at = di_cov[off + kt] - 2.0 * th.data[kt] + cp * lam_ao[kt]
                            + wfac * cp * w_dh1[kt];
                        let v = a + at;
                        di_cov[off + k] = v;
                        di_cov[off + kt] = v;
                    }
                }
            }
            let xm = to_mat(x_phi, nao, nao);
            let mut acc = MatrixFull::new([naux, naux], 0.0);
            let mut yi = MatrixFull::new([nao, naux], 0.0);
            let mut gi = MatrixFull::new([nao, naux], 0.0);
            let mut xi = MatrixFull::new([nao, naux], 0.0);
            for i in 0..nocc {
                for p in 0..naux {
                    for nu in 0..nao {
                        yi.data[nu + p * nao] = yt[p + (i * nao + nu) * naux];
                        gi.data[nu + p * nao] = gt[p + (i * nao + nu) * naux];
                    }
                }
                _dgemm_full(&xm, 'N', &gi, 'N', &mut xi, 1.0, 0.0);
                // accumulate over occupied orbitals (beta = 1)
                _dgemm_full(&yi, 'T', &xi, 'N', &mut acc, 1.0, 1.0);
            }
            let mut phi_flat = acc.data;
            self.solve_metric_batch(&mut phi_flat, naux);
            phi_flat
        };
        // The `Z` covector `W` rides the same contractions, and both builders
        // are linear in their covector, so a single call each suffices:
        //   -2 Theta_{Lambda + (W+W^T)/2} = -2 Theta_Lambda - Theta_{W+W^T}
        //     (the `dY G^T` term symmetrises its covector, hence `(W+W^T)/2`),
        //   Psi_{Lambda + W} = Psi_Lambda + Psi_W .
        let mut x_theta = lam_ao.clone();
        let mut x_phi = lam_ao.clone();
        if fold_rhs {
            for mu in 0..nao {
                for nu in 0..nao {
                    let k = mu + nu * nao;
                    x_theta[k] += 0.5 * (w_dh1[k] + w_dh1[nu + mu * nao]);
                    x_phi[k] += w_dh1[k];
                }
            }
        }
        let psi = fold_exchange(&x_theta, &x_phi);

        timings[5] = t_skel.elapsed().as_secs_f64();
        if grad_timing() {
            eprintln!(
                "[gw_grad timing]   adjoint: covectors {:.2}s  setup {:.2}s  prep {:.2}s  rhs {:.2}s  Z solve {:.2}s  foldings {:.2}s",
                timings[0], timings[1], timings[2], timings[3], timings[4], timings[5]
            );
        }
        // dI covector = B - 2*fock_hyb*Theta (the skeleton uses the plain 1/2)
        // The Coulomb skeleton is <Lam_ao, Jd> = sum_P a_P c_P + sum_P lam_P dc_P
        // with a_P = <Lam_ao, dI_P>, c = J^-1 rho and dc = J^-1 (drho - dJ c).
        // J^-1 is symmetric, so sum_P lam_P dc_P = sum_P lam_tilde_P (drho - dJ c)_P
        // with lam_tilde = J^-1 lam; this removes the per-perturbation metric
        // solve and lets `c_P Lam_ao` be folded into the dI covector.
        let half_k = 0.5f64;
        // `sum_d <Z,rhs_d> = <W, dh1 + Jd - Kd/2> + <W_s, dS>` rides the same
        // folded contractions as `<Lambda_ao, ...>`, so the Coulomb covector
        // `lambda_W = I^T W`, the exchange tensors `Theta_W`/`Psi_W` and the
        // per-direction `dh1`/`dS` partners are simply added in.
        let mut lam_aux_total = lam_aux.clone();
        if fold_rhs {
            for (a, b) in lam_aux_total.iter_mut().zip(lam_aux_w.iter()) {
                *a += b;
            }
        }
        let lam_tilde = self.solve_metric(&lam_aux_total);
        // `dI^(a)` (first bra centre) and `dI^(b)` (second bra centre) are
        // transposes of each other and `dI^(c)` (auxiliary centre) is symmetric,
        // so the full `dI` slice is symmetric and contracting with the
        // symmetrised covector counts both bra centres at once:
        //   sum_munu X dI = sum X_sym dI^(a) + (1/2) sum X_sym dI^(c) .
        let mut dm_sym = vec![0.0f64; n2];
        for mu in 0..nao {
            for nu in 0..nao {
                dm_sym[mu + nu * nao] = dm[mu + nu * nao] + dm[nu + mu * nao];
            }
        }

        mem_probe("  adjoint: after folding (di_cov/lam/psi)");
        Some(CphfAdjoint {
            di_cov,
            lam_ao,
            lams_ao,
            lam_aux,
            psi,
            c_aux,
            dm,
            rhs_dot,
            fold_rhs,
            w_dh1,
            w_ds,
            f1mo_ref: f1mo_ref.unwrap_or_default(),
            half_k,
            timings,
            lam_tilde,
            dm_sym,
            // `Theta` was folded into `di_cov` in place; it exists only when a verification run
            // asked for the individual pieces (`keep_adjoint_pieces`), because it duplicates an
            // `naux * nao^2` tensor that the gradient itself never reads.
            theta: theta_pieces,
            lam_mo: lam,
            lams_mo: lams,
            vtil,
            l_mat,
            z_vec,
        })
    }

    /// One perturbation's gradient from the adjoint data; same value as
    /// `folded_perturbation_gradient` but with the CP-HF response contracted
    /// through `Z` (no per-perturbation response).
    ///
    /// This variant materialises a dense `[nao,nao]` `dI` slice for every
    /// auxiliary function; see [`GwGradEngine::adjoint_perturbation_gradient_local`]
    /// for the locality-aware (asymptotically cheaper) version.
    #[allow(clippy::too_many_arguments)]
    pub fn adjoint_perturbation_gradient(
        &self,
        adj: &CphfAdjoint,
        blocks: &AtomDerivBlocks,
        comp: usize,
        h1r: &[f64],
        s1_comp: &[f64],
        dj: &[f64],
        bj_k: &[f64],
        bb_k: &[f64],
        b_x: &[f64],
        dir: usize,
    ) -> f64 {
        let n2 = self.nao * self.nao;
        let naux = self.naux;
        let mut ip = vec![0.0f64; n2];
        let mut g = 0.0f64;
        let mut g_dm = 0.0f64;
        for p in 0..naux {
            self.raw.d_i_block(blocks, comp, p, &mut ip);
            let off = p * n2;
            // the dense `dI` slice is symmetric, so the symmetrised covectors
            // count it with a factor 1/2
            g += 0.5 * flat_dot(&adj.di_cov[off..off + n2], &ip);
            g_dm += 0.5 * adj.lam_tilde[p] * flat_dot(&adj.dm_sym, &ip);
        }
        self.finish_adjoint_gradient(adj, comp, h1r, s1_comp, dj, bj_k, bb_k, b_x, dir, g + g_dm)
    }

    /// Locality-aware variant of [`GwGradEngine::adjoint_perturbation_gradient`].
    ///
    /// `dI[mu,nu,P]` is nonzero only where a centre sits on the perturbed atom:
    /// `int3c2e_ip1` gives the atom's AO rows (`mu` on the atom, all `nu`) and
    /// columns (`nu` on the atom, all `mu`), while `int3c2e_ip2` is dense in
    /// `[mu,nu]` but only for the atom's own auxiliary functions `P`.
    /// Contracting the covectors directly against those sliced blocks costs
    /// `O(naux nrow_A nao + naux_A nao^2)` per perturbation instead of the
    /// `O(naux nao^2)` of the dense route; summed over the `3 natm`
    /// perturbations the whole gradient drops from `O(3 natm naux nao^2)` to
    /// `O(naux nao^2)`.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn adjoint_perturbation_gradient_local(
        &self,
        adj: &CphfAdjoint,
        atm: usize,
        h1r: &[f64],
        s1_all: &[Vec<f64>],
        djs: &[Vec<f64>],
        bj_k: &[f64],
        bb_k: &[f64],
        b_x: &[f64],
        dir0: usize,
    ) -> [f64; 3] {
        let nao = self.nao;
        let n2 = nao * nao;
        let row0 = self.raw.ao_range[atm][0];
        let nrow = self.raw.ao_range[atm][1] - row0;

        // all three Cartesian components are contracted from the *same* streamed integral blocks
        let mut g_di = [0.0f64; 3];
        let mut g_dm = [0.0f64; 3];
        // The two inner loops are pure reductions, and Rust/LLVM will not
        // reassociate `f64` additions on their own; four independent
        // accumulators give both SIMD width and enough ILP to run at memory
        // speed (measured ~3x over the single-accumulator form).
        //
        // The derivative blocks are *streamed* over auxiliary-shell blocks (see
        // `RawRiTensors::for_each_d1_block`/`for_each_d2_block`), so the per-atom tensors
        // (`nrow_A*nao*naux*3` + `nao^2*naux_A*3` doubles, ~390 MB for a carbon atom at
        // naphthalene/def2-TZVP) are never resident; each block is consumed immediately.
        let dm = &adj.dm_sym;
        self.raw.for_each_d1_block(atm, |p0, nb, d1b| {
            let stride1 = nrow * nao * nb;
            for comp in 0..3 {
            let d1c = &d1b[comp * stride1..(comp + 1) * stride1];
            for pl in 0..nb {
            let p = p0 + pl;
            let cov = &adj.di_cov[p * n2..(p + 1) * n2];
            let lt = adj.lam_tilde[p];
            let base1 = pl * nrow * nao;
            let mut adi = [0.0f64; 4];
            let mut adm = [0.0f64; 4];
            // int3c2e_ip1: rows `mu` on the atom, all `nu`.  `dI^(b)` is the
            // transpose of `dI^(a)` and is already counted by the symmetrised
            // covectors, so a single contiguous pass covers both bra centres.
            // nu outside / mu_l inside keeps every access contiguous.
            for nu in 0..nao {
                let doff = base1 + nu * nrow;
                let coff = row0 + nu * nao;
                let mut i = 0usize;
                while i + 4 <= nrow {
                    let d = &d1c[doff + i..doff + i + 4];
                    let c = &cov[coff + i..coff + i + 4];
                    let m = &dm[coff + i..coff + i + 4];
                    for j in 0..4 {
                        let v = -d[j];
                        adi[j] += c[j] * v;
                        adm[j] += m[j] * v;
                    }
                    i += 4;
                }
                while i < nrow {
                    let v = -d1c[doff + i];
                    let k = coff + i;
                    adi[0] += cov[k] * v;
                    adm[0] += dm[k] * v;
                    i += 1;
                }
            }
            g_di[comp] += adi.iter().sum::<f64>();
            g_dm[comp] += lt * adm.iter().sum::<f64>();
            }
            }
        });
        // int3c2e_ip2: dense [nao,nao] symmetric slices of the atom's own auxiliary functions, with
        // the 1/2 of the symmetrised contraction; streamed as well.
        self.raw.for_each_d2_block(atm, |p0, nb, d2b| {
            let stride2 = n2 * nb;
            for comp in 0..3 {
            let d2c = &d2b[comp * stride2..(comp + 1) * stride2];
            for pl in 0..nb {
                let p = p0 + pl;
                let cov = &adj.di_cov[p * n2..(p + 1) * n2];
                let lt = adj.lam_tilde[p];
                let d2 = &d2c[pl * n2..(pl + 1) * n2];
                let mut cdi = [0.0f64; 4];
                let mut cdm = [0.0f64; 4];
                let mut k = 0usize;
                while k + 4 <= n2 {
                    let d = &d2[k..k + 4];
                    let c = &cov[k..k + 4];
                    let m = &dm[k..k + 4];
                    for j in 0..4 {
                        let v = -0.5 * d[j];
                        cdi[j] += c[j] * v;
                        cdm[j] += m[j] * v;
                    }
                    k += 4;
                }
                while k < n2 {
                    let v = -0.5 * d2[k];
                    cdi[0] += cov[k] * v;
                    cdm[0] += dm[k] * v;
                    k += 1;
                }
                g_di[comp] += cdi.iter().sum::<f64>();
                g_dm[comp] += lt * cdm.iter().sum::<f64>();
            }
            }
        });
        let mut out = [0.0f64; 3];
        for comp in 0..3 {
            out[comp] = self.finish_adjoint_gradient(
                adj, comp, h1r, &s1_all[comp], &djs[comp], bj_k, bb_k, b_x, dir0 + comp,
                g_di[comp] + g_dm[comp],
            );
        }
        out
    }

    /// Shared tail of the two adjoint perturbation routines: everything except
    /// the `dI` contraction, whose value is passed in as `acc`.
    #[allow(clippy::too_many_arguments)]
    fn finish_adjoint_gradient(
        &self,
        adj: &CphfAdjoint,
        comp: usize,
        h1r: &[f64],
        s1_comp: &[f64],
        dj: &[f64],
        bj_k: &[f64],
        bb_k: &[f64],
        b_x: &[f64],
        dir: usize,
        acc: f64,
    ) -> f64 {
        let n2 = self.nao * self.nao;
        let naux = self.naux;
        let mut g = acc;
        // - sum_P lam_tilde_P (dJ c)_P
        let mut djc = 0.0f64;
        for p in 0..naux {
            let mut s = 0.0f64;
            for q in 0..naux {
                s += dj[p + q * naux] * adj.c_aux[q];
            }
            djc += adj.lam_tilde[p] * s;
        }
        g -= djc;
        // exchange skeleton remainder, core Hamiltonian and overlap
        g += 2.0 * adj.half_k * flat_dot(&adj.psi, dj);
        g += flat_dot(&adj.lam_ao, &h1r[comp * n2..(comp + 1) * n2]);
        g += flat_dot(&adj.lams_ao, s1_comp);
        // the folded `<Z, rhs_d>` channel: <W, dh1> + <W_s, dS> (the rest of
        // `<W, Jd - Kd/2>` rides the shared covectors)
        if adj.fold_rhs {
            g += flat_dot(&adj.w_dh1, &h1r[comp * n2..(comp + 1) * n2]);
            g += flat_dot(&adj.w_ds, s1_comp);
        }
        // metric channel, static term and the adjoint CP-HF channel
        g += flat_dot(bj_k, dj);
        g += flat_dot(bb_k, b_x);
        g += adj.rhs_dot[dir];
        g
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    fn rnd(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0
    }

    fn naive_j(nao: usize, naux: usize, i3: &[f64], di: &[f64], dj: &[f64], dm: &[f64], j: &MatrixFull<f64>) -> Vec<f64> {
        let n2 = nao * nao;
        let mut rho = vec![0.0f64; naux];
        let mut drho = vec![0.0f64; naux];
        for p in 0..naux {
            let mut a1 = 0.0;
            let mut a2 = 0.0;
            for k in 0..n2 {
                a1 += i3[k + p * n2] * dm[k];
                a2 += di[k + p * n2] * dm[k];
            }
            rho[p] = a1;
            drho[p] = a2;
        }
        let c = _dsolve(j, &rho).unwrap();
        let djc = gemm_nn(dj, naux, naux, &c, naux, 1);
        let rhs: Vec<f64> = drho.iter().zip(djc.iter()).map(|(a, b)| a - b).collect();
        let dc = _dsolve(j, &rhs).unwrap();
        let mut out = vec![0.0f64; n2];
        for p in 0..naux {
            for k in 0..n2 {
                out[k] += di[k + p * n2] * c[p] + i3[k + p * n2] * dc[p];
            }
        }
        out
    }

    fn naive_k(nao: usize, naux: usize, nocc: usize, i3: &[f64], di: &[f64], dj: &[f64], cocc: &[f64], j: &MatrixFull<f64>) -> Vec<f64> {
        let n2 = nao * nao;
        let mut out = vec![0.0f64; n2];
        for i in 0..nocc {
            let mut y_i = vec![0.0f64; nao * naux];
            let mut dy_i = vec![0.0f64; nao * naux];
            for p in 0..naux {
                for mu in 0..nao {
                    let mut a1 = 0.0;
                    let mut a2 = 0.0;
                    for nu in 0..nao {
                        let ci = cocc[nu + i * nao];
                        a1 += i3[mu + nu * nao + p * n2] * ci;
                        a2 += di[mu + nu * nao + p * n2] * ci;
                    }
                    y_i[mu + p * nao] = a1;
                    dy_i[mu + p * nao] = a2;
                }
            }
            let mut g = vec![0.0f64; naux * nao];
            for mu in 0..nao {
                let rhs: Vec<f64> = (0..naux).map(|p| y_i[mu + p * nao]).collect();
                let sol = _dsolve(j, &rhs).unwrap();
                for p in 0..naux {
                    g[p + mu * naux] = sol[p];
                }
            }
            let dk1 = gemm_nn(&dy_i, nao, naux, &g, naux, nao);
            let tmp = gemm_nn(dj, naux, naux, &g, naux, nao);
            let mut tmp2 = vec![0.0f64; naux * nao];
            for nu in 0..nao {
                let rhs: Vec<f64> = (0..naux).map(|p| tmp[p + nu * naux]).collect();
                let sol = _dsolve(j, &rhs).unwrap();
                for p in 0..naux {
                    tmp2[p + nu * naux] = sol[p];
                }
            }
            let term = gemm_nn(&y_i, nao, naux, &tmp2, naux, nao);
            for row in 0..nao {
                for col in 0..nao {
                    let k = row + col * nao;
                    let k_t = col + row * nao;
                    out[k] += 2.0 * (dk1[k] + dk1[k_t] - term[k]);
                }
            }
        }
        out
    }

    /// naive reference of the old scalar qx_from_u
    fn naive_qx(nao: usize, naux: usize, nmo: usize, c: &[f64], q: &[f64], di: &[f64], u: &[f64]) -> Vec<f64> {
        let n2 = nao * nao;
        let mut qx = vec![0.0f64; naux * nmo * nmo];
        for p_aux in 0..naux {
            let di_block = &di[p_aux * n2..(p_aux + 1) * n2];
            let mut tmp = vec![0.0f64; nao * nmo];
            for mu in 0..nao {
                for q_mo in 0..nmo {
                    let mut acc = 0.0;
                    for nu in 0..nao {
                        acc += di_block[mu + nu * nao] * c[nu + q_mo * nao];
                    }
                    tmp[mu + q_mo * nao] = acc;
                }
            }
            for q_mo in 0..nmo {
                for p_mo in 0..nmo {
                    let mut acc = 0.0;
                    for mu in 0..nao {
                        acc += c[mu + p_mo * nao] * tmp[mu + q_mo * nao];
                    }
                    qx[(p_mo + q_mo * nmo) * naux + p_aux] = acc;
                }
            }
        }
        let um = to_mat(u, nmo, nmo);
        for p_aux in 0..naux {
            let mut block = vec![0.0f64; nmo * nmo];
            for col in 0..nmo * nmo {
                block[col] = q[col * naux + p_aux];
            }
            let utq = gemm_nt(u, nmo, nmo, &block, nmo, nmo);
            let bmat = to_mat(&block, nmo, nmo);
            let mut qum = MatrixFull::new([nmo, nmo], 0.0);
            _dgemm_full(&bmat, 'N', &um, 'N', &mut qum, 1.0, 0.0);
            for col in 0..nmo * nmo {
                block[col] = qx[col * naux + p_aux] + utq[col] + qum.data[col];
            }
            for pi in 0..nmo {
                for pj in (pi + 1)..nmo {
                    let a = pi + pj * nmo;
                    let b = pj + pi * nmo;
                    let s = 0.5 * (block[a] + block[b]);
                    block[a] = s;
                    block[b] = s;
                }
            }
            for col in 0..nmo * nmo {
                qx[col * naux + p_aux] = block[col];
            }
        }
        qx
    }

    /// the new qx algorithm, copied verbatim from qx_from_u (synthetic inputs)
    fn new_qx(nao: usize, naux: usize, nmo: usize, c: &[f64], q: &[f64], di: &[f64], u: &[f64]) -> Vec<f64> {
        let n2 = nao * nao;
        let npair = nmo * nmo;
        let mut qx = vec![0.0f64; naux * npair];
        {
            let cm = to_mat(c, nao, nmo);
            let mut ip = MatrixFull::new([nao, nao], 0.0);
            let mut tmp = MatrixFull::new([nao, nmo], 0.0);
            let mut sl = MatrixFull::new([nmo, nmo], 0.0);
            for p in 0..naux {
                ip.data.copy_from_slice(&di[p * n2..(p + 1) * n2]);
                _dgemm_full(&ip, 'N', &cm, 'N', &mut tmp, 1.0, 0.0);
                _dgemm_full(&cm, 'T', &tmp, 'N', &mut sl, 1.0, 0.0);
                for col in 0..npair {
                    qx[p + col * naux] = sl.data[col];
                }
            }
        }
        // pair-slowest repack of q, as in GwGradEngine::new_with_raw
        let q_pm = {
            let mut v = vec![0.0f64; naux * npair];
            for pair in 0..npair {
                for p in 0..naux {
                    v[pair + p * npair] = q[pair * naux + p];
                }
            }
            v
        };
        // chunked pair-major gemm, as in qx_from_u
        let um = to_mat(u, nmo, nmo);
        let chunk_aux = (4_194_304 / (nmo * nmo)).clamp(1, naux);
        let mut p0 = 0;
        while p0 < naux {
            let n_chunk = (naux - p0).min(chunk_aux);
            let nc = n_chunk * nmo;
            let q2c = to_mat(&q_pm[p0 * npair..(p0 + n_chunk) * npair], nmo, nc);
            let mut t1v = MatrixFull::new([nmo, nc], 0.0);
            _dgemm_full(&um, 'T', &q2c, 'N', &mut t1v, 1.0, 0.0);
            for p in p0..p0 + n_chunk {
                let base = (p - p0) * npair;
                for pj in 0..npair {
                    let pj_swap = (pj % nmo) * nmo + pj / nmo;
                    qx[p + pj * naux] += t1v.data[base + pj] + t1v.data[base + pj_swap];
                }
                for pi in 0..nmo {
                    for pj in (pi + 1)..nmo {
                        let a = pi + pj * nmo;
                        let b = pj + pi * nmo;
                        let s = 0.5 * (qx[p + a * naux] + qx[p + b * naux]);
                        qx[p + a * naux] = s;
                        qx[p + b * naux] = s;
                    }
                }
            }
            p0 += n_chunk;
        }
        qx
    }

    #[test]
    fn j_k_qx_match_naive_reference() {
        let mut seed = 42u64;
        let nao = 12;
        let naux = 17;
        let nocc = 4;
        let nmo = nao;
        let n2 = nao * nao;
        // SPD metric
        let mut a = vec![0.0f64; naux * naux];
        for x in a.iter_mut() {
            *x = rnd(&mut seed);
        }
        let am = to_mat(&a, naux, naux);
        let mut spd = MatrixFull::new([naux, naux], 0.0);
        _dgemm_full(&am, 'T', &am, 'N', &mut spd, 1.0, 0.0);
        for i in 0..naux {
            spd.data[i + i * naux] += naux as f64;
        }
        // Cholesky factor + multi-RHS solve closure, as in GwGradEngine
        let mut chol = spd.clone();
        _dpotrf(&mut chol, 'L');
        let solve = |rhs: &[f64]| -> Vec<f64> {
            let mut b = rhs.to_vec();
            let mut bm = to_mat(&b, naux, 1);
            assert!(_dtrtrs(&chol, &mut bm, 'L', 'N', 'N'));
            assert!(_dtrtrs(&chol, &mut bm, 'L', 'T', 'N'));
            b.copy_from_slice(&bm.data);
            b
        };
        // sanity: cholesky solve == dgesv solve
        let rhs: Vec<f64> = (0..naux).map(|_| rnd(&mut seed)).collect();
        let x1 = solve(&rhs);
        let x2 = _dsolve(&spd, &rhs).unwrap();
        let err = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(err < 1.0e-10, "chol solve err {err}");

        let i3: Vec<f64> = (0..n2 * naux).map(|_| 0.2 * rnd(&mut seed)).collect();
        let di: Vec<f64> = (0..n2 * naux).map(|_| 0.2 * rnd(&mut seed)).collect();
        let dj: Vec<f64> = {
            let mut v: Vec<f64> = (0..naux * naux).map(|_| rnd(&mut seed)).collect();
            for i in 0..naux {
                for j in 0..i {
                    v[i + j * naux] = v[j + i * naux];
                }
            }
            v
        };
        let dm: Vec<f64> = {
            let mut v: Vec<f64> = (0..n2).map(|_| rnd(&mut seed)).collect();
            for i in 0..nao {
                for j in 0..i {
                    v[i + j * nao] = v[j + i * nao];
                }
            }
            v
        };
        let cocc: Vec<f64> = (0..nao * nocc).map(|_| rnd(&mut seed)).collect();
        let cmo: Vec<f64> = (0..nao * nmo).map(|_| rnd(&mut seed)).collect();
        let u: Vec<f64> = (0..nmo * nmo).map(|_| rnd(&mut seed)).collect();
        // pair-symmetric q
        let q: Vec<f64> = {
            let mut v: Vec<f64> = (0..naux * nmo * nmo).map(|_| rnd(&mut seed)).collect();
            for p in 0..naux {
                for pi in 0..nmo {
                    for pj in 0..pi {
                        let a = pi + pj * nmo;
                        let b = pj + pi * nmo;
                        let s = 0.5 * (v[a * naux + p] + v[b * naux + p]);
                        v[a * naux + p] = s;
                        v[b * naux + p] = s;
                    }
                }
            }
            v
        };

        let jn = naive_j(nao, naux, &i3, &di, &dj, &dm, &spd);
        let jnew = j_deriv_from_impl(nao, naux, &i3, &di, &dj, &dm, &solve);
        let ej = jn.iter().zip(jnew.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(ej < 1.0e-12, "j_deriv mismatch {ej}");

        let kn = naive_k(nao, naux, nocc, &i3, &di, &dj, &cocc, &spd);
        let knew = k_deriv_from_impl(nao, naux, nocc, &i3, &di, &dj, &cocc, &solve);
        let ek = kn.iter().zip(knew.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(ek < 1.0e-12, "k_deriv mismatch {ek}");

        let qn = naive_qx(nao, naux, nmo, &cmo, &q, &di, &u);
        let qnew = new_qx(nao, naux, nmo, &cmo, &q, &di, &u);
        let eq = qn.iter().zip(qnew.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(eq < 1.0e-12, "qx mismatch {eq}");
    }

    /// link + correctness smoke test of the complex symmetric LU solve used
    /// by the full-CD screening (zgetrf/zgetrs)
    #[test]
    fn cd_complex_lu_solve_matches_reference() {
        use num_complex::ComplexFloat;
        let n = 9usize;
        let mut seed = 7u64;
        // complex symmetric A = B B^T with complex B (symmetric, indefinite ok)
        let bmat: Vec<Complex<f64>> = (0..n * n).map(|_| Complex::new(rnd(&mut seed), rnd(&mut seed))).collect();
        let mut a_re = vec![0.0f64; n * n];
        let mut a_im = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = Complex::new(0.0, 0.0);
                for k in 0..n {
                    acc += bmat[i + k * n] * bmat[j + k * n];
                }
                a_re[i + j * n] = acc.re;
                a_im[i + j * n] = acc.im;
            }
        }
        let factor = CdLuFactor::factor(&a_re, &a_im, n);
        let ncol = 3usize;
        let rhs: Vec<Complex<f64>> = (0..n * ncol).map(|_| Complex::new(rnd(&mut seed), rnd(&mut seed))).collect();
        let mut x = rhs.clone();
        factor.solve_in_place(&mut x, ncol);
        // verify A X = B column by column
        for c in 0..ncol {
            for i in 0..n {
                let mut acc = Complex::new(0.0, 0.0);
                for j in 0..n {
                    acc += Complex::new(a_re[i + j * n], a_im[i + j * n]) * x[j + c * n];
                }
                let err = (acc - rhs[i + c * n]).abs();
                assert!(err < 1.0e-10, "complex LU solve mismatch {err}");
            }
        }
        // residue_dz/de consistency: d/dz of residue_d == residue_dz
        let (delta, z, eta) = (-0.7, 0.31, 1.0e-3);
        let h = 1.0e-6;
        let fd = (residue_d(delta, z + h, eta) - residue_d(delta, z - h, eta)) / (2.0 * h);
        let an = residue_dz(delta, z, eta);
        assert!((fd - an).abs() < 1.0e-8, "residue_dz mismatch {fd} vs {an}");
        let fdd = (residue_d(delta + h, z, eta) - residue_d(delta - h, z, eta)) / (2.0 * h);
        let and = residue_de(delta, z, eta);
        assert!((fdd - and).abs() < 1.0e-8, "residue_de mismatch {fdd} vs {and}");
        let fd0 = (static_d0(delta + h, eta) - static_d0(delta - h, eta)) / (2.0 * h);
        let an0 = static_d0e(delta, eta);
        assert!((fd0 - an0).abs() < 1.0e-8, "static_d0e mismatch {fd0} vs {an0}");
    }
}
