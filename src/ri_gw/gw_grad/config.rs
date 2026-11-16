/// Numerical options of the LR-CD G0W0 gradient.  The defaults mirror REST's
/// `QuasiParticle` control block so the module reproduces `ri_gw` results.
#[derive(Clone, Debug)]
pub struct GwGradConfig {
    /// number of imaginary-axis quadrature points (REST `num_freq`)
    pub num_freq: usize,
    /// How the correlation self-energy is represented:
    ///
    /// * [`GwVariant::Cd`] — the contour-deformation decomposition (imaginary
    ///   axis quadrature + real-axis residues) that this module was written
    ///   for;
    /// * [`GwVariant::Ac`] — analytic continuation: `Sigma_c` is sampled on
    ///   the imaginary axis and the retarded value is obtained from a Thiele
    ///   Padé approximant, exactly as the `gw_variant = "ac"` energy path
    ///   (`ri_gw::scgw::single_orbital_gw_ac`).  The gradient differentiates
    ///   the same Padé functional (see [`crate::ri_gw::ac_grad`]);
    /// * [`GwVariant::Fscd`] — the frequency-sampled CD residue evaluation.
    ///   The gradient currently differentiates the CD functional, and a
    ///   warning is printed once.
    pub gw_variant: GwVariant,
    /// Number of imaginary-axis samples of the Padé approximant (REST
    /// `ac_num_samples`), used only by [`GwVariant::Ac`].
    pub ac_num_samples: usize,
    /// Positive broadening `eta` of the retarded Padé evaluation
    /// (`Sigma_c(omega + i eta)`), REST `ac_eta`, used only by
    /// [`GwVariant::Ac`].
    pub ac_eta: f64,
    /// CD-WAC (`gw_variant = "wac"`): number of imaginary-axis reference
    /// frequencies for the continuation of `W` (0 = all quadrature points).
    pub wac_n_imag: usize,
    /// CD-WAC: number of real-axis reference frequencies.
    pub wac_n_real: usize,
    /// CD-WAC: largest real reference frequency (Ha; 0 = automatic).
    pub wac_omega_max: f64,
    /// CD-WAC: imaginary offset (Ha) of the near-real reference frequencies.
    pub wac_delta: f64,
    /// CD-WAC: real-axis reference-grid spacing rule (see
    /// `QuasiParticle::wac_grid`).  `0` quadratic (historical), `1` linear,
    /// `2` geometric, `3` Chebyshev, `4` pole-adapted.
    pub wac_grid: usize,
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
    /// Energy above which a **virtual** level takes the linearized (one-step
    /// Newton) functional *directly*, without any root search: the counterpart
    /// of [`Self::static_fallback_threshold`] for the empty states.
    ///
    /// ```text
    /// E_lin = e_n + Z(e_n) [ consts_n + Re Sigma_c(e_n) - e_n ],  Z = 1/(1 - dSigma_c/domega)
    /// ```
    ///
    /// High-lying diffuse virtuals are where the quantised-residue
    /// self-energy has several roots: at the S1 minimum of azobenzene/def2-TZVPP
    /// 21 virtuals (e_n >= +0.5 Ha) came out 0.1-0.5 Ha away from the root that
    /// the contour engine finds, and those bad QP energies poisoned the BSE
    /// gradient.  Taking the linear branch there keeps `Z` (the physical
    /// spectral weight) and is smooth in the geometry, exactly as the static
    /// branch is for the deep occupied levels.  A level is "high" when
    /// `n >= nocc` **and** `e_n > linear_fallback_threshold`; `f64::INFINITY`
    /// (the default) disables the rule.
    pub linear_fallback_threshold: f64,
    /// Use the quasiparticle energies that the **energy path** already solved
    /// (`scf_data.gwqp.0`, the vector the BSE Hamiltonian is built from) instead
    /// of re-solving the QP equation inside the gradient engine.
    ///
    /// The gradient needs, per orbital, the response cache (residues, `y/w`
    /// quadrature data, `Z`, `dSigma/domega`) -- not just the energy -- which is
    /// why [`BseGradEngine`] used to call `build_target_cache` and take
    /// `omega` from it.  That made the gradient re-run a *second*, independent
    /// root search: at the S1 minimum of azobenzene/def2-TZVPP the same orbital
    /// came out at +0.724 Ha in the energy path and at -0.247 Ha in the
    /// gradient.  When this flag is set the cache is built **at** the stored
    /// quasiparticle energy (no root search, one self-energy evaluation per
    /// orbital instead of one per Newton step), so the BSE diagonal, the printed
    /// spectrum and the gradient share one quasiparticle solution.
    pub pin_qp_from_checkpoint: bool,
    /// Upper bound on the linearized step `|Z * shift|` (Ha).  `Z` of a
    /// high-lying virtual can be negative and large (`Z = -24.8` was observed),
    /// in which case the linear extrapolation would move the level by ~1 Ha.
    /// The step is clamped to this value and a warning is printed; the clamped
    /// `Z` is stored in the cache (`z_factor`) so the analytic gradient uses
    /// the same functional that produced the energy.  `0` disables the clamp.
    pub linear_step_cap: f64,
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
    /// Extend the adjoint (Z-vector) CP-HF route to DFT and hybrid (CP-KS)
    /// references, whose response operator carries the XC kernel and whose
    /// static term `B_n` carries the `-v_xc` derivative.
    ///
    /// Three DFT channels are folded once per geometry (see the section header
    /// of `build_cphf_adjoint`): `f_xc(Lam_ao)` and `-f_xc(D_bb)` in `Vtil`, the
    /// `b_x` orbital-rotation covector into `Lambda`/`Lambda_s`, and the
    /// explicit v_xc skeleton into both the Roothaan covector and the `Z`
    /// right-hand side.  What remains per perturbation is the same `dI`
    /// contraction as for HF, replacing the `3 natm`-column canonical CP-KS
    /// solve.
    ///
    /// **On by default** (with `REST_ADJOINT_DFT=0` or the card keyword
    /// `gradient_adjoint_cphf = false` as the escape hatch): validated against
    /// the canonical CP-KS route to `~1e-8` relative on water/formaldehyde and
    /// to `1e-9` absolute on the production PBEh(45%) acene benchmarks
    /// (`tests/gw_bse/test_adjoint_dft.rs`), which is the direct-vs-folded
    /// contraction floor and far below the finite-difference accuracy of the
    /// canonical route itself.  Measured speedup at the production settings:
    /// 2.9x (benzene/def2-SVP, 36 directions), 5.5x (naphthalene/def2-SVP, 54)
    /// at 4 threads.
    pub adjoint_cphf_dft: bool,
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
    /// Krylov tolerance of the CP-HF/CP-KS response solve (`cpscf_tol`, default
    /// `1e-9`) and the inflation factor of its intermediate iterations
    /// (`cpscf_tol_inflation`, default `1000`).
    ///
    /// The **adjoint** route's gradient accuracy is set by these directly: its
    /// CP-HF channel is `<Z, rhs_d>` with `Z = S^T L`, so the Z-solve residual
    /// enters the gradient linearly, while the canonical route re-derives `u`
    /// from the relaxed Fock and is far less sensitive.  The default leaves the
    /// two DFT routes agreeing to ~1e-7 relative; `1e-12`/`10` takes that to the
    /// ~1e-8 folding floor (see `tests/gw_bse/test_adjoint_dft.rs`) for a few
    /// extra Krylov iterations on a single right-hand side.
    pub cpscf_tol: f64,
    /// Inflation of [`Self::cpscf_tol`] for the intermediate Krylov iterations.
    pub cpscf_tol_inflation: f64,
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
            gw_variant: GwVariant::Cd,
            ac_num_samples: 16,
            ac_eta: 0.001,
            wac_n_imag: 0,
            wac_n_real: 20,
            wac_omega_max: 50.0,
            wac_delta: 1.0e-3,
            wac_grid: 0,
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
            linear_fallback_threshold: f64::INFINITY,
            linear_step_cap: 0.5,
            pin_qp_from_checkpoint: true,
            low_rank_interp: 1,
            graphical_rescue: false,
            smooth_pole: false,
            degenerate_tol: 1.0e-8,
            phase_mask: 0b111,
            skip_lr_cache: false,
            folded_qx: true,
            adjoint_cphf: true,
            // Validated against the canonical CP-KS route to ~1e-8 relative
            // (`tests/gw_bse/test_adjoint_dft.rs`), so it is the default for DFT
            // references too; `REST_ADJOINT_DFT=0` (or the ctrl keyword
            // `gradient_adjoint_cphf = false`) restores the canonical route.
            adjoint_cphf_dft: std::env::var("REST_ADJOINT_DFT").map_or(true, |v| v != "0"),
            local_di: true,
            cpscf_tol: 1.0e-9,
            cpscf_tol_inflation: 1000.0,
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
            gw_variant: qp.gw_variant,
            ac_num_samples: qp.ac_num_samples.max(2),
            ac_eta: if qp.ac_eta > 0.0 { qp.ac_eta } else { 1.0e-3 },
            wac_n_imag: qp.wac_n_imag,
            wac_n_real: qp.wac_n_real,
            wac_omega_max: qp.wac_omega_max,
            wac_delta: if qp.wac_delta > 0.0 { qp.wac_delta } else { 1.0e-3 },
            wac_grid: qp.wac_grid,
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
            linear_fallback_threshold: qp.gradient_linear_threshold,
            linear_step_cap: qp.gradient_linear_step_cap,
            pin_qp_from_checkpoint: qp.gradient_qp_from_checkpoint,
            low_rank_interp: crate::ri_gw::low_rank_interp_code(&qp.low_rank_interp),
            graphical_rescue: qp.gradient_graphical_rescue,
            // `gradient_qp_graphical` (card) or `REST_QP_GRAPHICAL=1` (env): pick the
            // QP root with the production graphical scan instead of the Newton cascade.
            qpe_graphical: qp.gradient_qp_graphical
                || std::env::var("REST_QP_GRAPHICAL").map(|v| v != "0").unwrap_or(false),
            smooth_pole: qp.gradient_smooth_pole,
            // The input card is the authority for the route choice; the
            // `REST_ADJOINT_DFT` environment variable can still force it off
            // (it is read by `Default::default`, and the card value wins when
            // the keyword is present -- `gradient_adjoint_cphf` defaults to
            // true, so this is an explicit opt-out).
            adjoint_cphf_dft: qp.gradient_adjoint_cphf
                && std::env::var("REST_ADJOINT_DFT").map_or(true, |v| v != "0"),
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
pub(crate) fn mem_probe(label: &str) {
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
