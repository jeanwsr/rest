use serde::{Deserialize,Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GwVariant {
    Cd,
    Ac,
}

impl Default for GwVariant {
    fn default() -> Self {
        Self::Cd
    }
}

impl GwVariant {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cd => "cd",
            Self::Ac => "ac",
        }
    }
}

#[derive(Debug,Clone,Serialize, Deserialize)]
pub struct QuasiParticle {
    pub gw_scheme:String,
    pub gw_variant: GwVariant,
    pub homo_lumo_gw_qp:bool,
    pub x_alpha:f64,
    pub save_qp:bool,
    pub bse_davidson_solver:bool,
    pub davidson_target_excitations:usize,
    pub davidson_converge_threshold:f64,
    pub davidson_maximum_subspace_size:usize,
    pub davidson_restart_dimensions:usize,
    pub davidson_max_iter:usize,
    pub davidson_add_dimensions:usize,
    pub bse_tda:bool,
    pub print_nto:bool,
    /// BSE switch / spin channel.
    ///
    /// * `"none"` (default): no BSE is run.
    /// * Restricted reference: a genuine spin channel selector --
    ///   `"singlet"` (direct term, factor 2), `"triplet"` (no direct term,
    ///   factor 0) or `"both"`.  The restricted reference can be rotated into
    ///   the singlet/triplet subspaces, so both channels are physical.
    /// * Unrestricted reference (`spin_polarization = true`): **no channel
    ///   selection**.  The direct (screened Coulomb) kernel is spin-independent
    ///   and couples the alpha and beta blocks, so there is a single physical
    ///   channel; any value other than `"none"` just switches the run on, and
    ///   `"triplet"` / `"both"` are rejected (see `bse_main_unrestricted`).
    pub bse_spin:String,
    pub bse_cutoff_energy:f64,
    pub save_bse_terms:bool,
    pub obtain_vx_vc_terms:bool,
    pub obtain_pure_exchange:bool,
    pub obtain_ks_homos:bool,
    pub gw_linearize_shift:f64,
    pub gw_linearize_derivative_h:f64,
    pub renormalized_singles:bool,
    pub w_rs:bool,
    pub rs_full_space:bool,
    pub rs_use_rs_orbitals:bool,
    pub scgw:String,
    pub gw:bool,
    pub bse_exchange_rescaling:f64,
    pub save_bse_excitations:bool,
    pub evgw_rounds:usize,
    /// Which solver is used for the evGW outer loop (eigenvalue
    /// self-consistency on the quasiparticle energies).
    ///
    /// * `"molgw"` (default): MolGW's `GnWn` (EVSC) rule -- a *single*
    ///   evaluation `E_out = consts + Sigma_c(E_in)` at the previous
    ///   quasiparticle energy, with no root solve and no damping (`Z = 1`).
    ///   This is the branch MolGW really executes: `selfenergy_init` sets
    ///   `se%nomega = 0` for the `EVSC` technique, so
    ///   `find_qp_energy_linearization` always takes its `else` branch and the
    ///   Z factor is never applied (verified against MolGW's own tables, where
    ///   `E_qp = E0 + SigX-Vxc + SigC` with `E0` the constant KS energy).  Not
    ///   solving the quasiparticle equation inside the loop is what removes
    ///   REST's sensitivity to the coarse low-rank real-axis grid and its
    ///   satellite-root hopping.
    /// * `"diis"`: one full GW pass per round (historical root-solving map),
    ///   accelerated by Pulay/DIIS extrapolation of the whole
    ///   quasiparticle-energy vector -- the mechanism used by `pyscf.gw.evgw`.
    /// * `"z_update"`: one *linearized* (Z-damped) Newton step per orbital per
    ///   round, `E_out = E_in + Z (consts + Sigma_c(E_in) - E_in)` with
    ///   `Z = 1/(1 - dSigma_c/domega)` clamped to `[0, 1]`; no DIIS is applied.
    /// * `"molgw"`: MolGW's actual `GnWn` (EVSC) rule -- a *single* evaluation
    ///   `E_out = consts + Sigma_c(E_in)` at the previous quasiparticle energy,
    ///   with no root solve and no damping (`Z = 1`).  This is the branch
    ///   MolGW really executes: `selfenergy_init` sets `se%nomega = 0` for the
    ///   `EVSC` technique, so `find_qp_energy_linearization` always takes its
    ///   `else` branch and the Z factor is never applied.  No DIIS.
    pub evgw_solver:String,
    /// Linear mixing (damping) factor alpha in (0, 1] for the evGW outer loop.
    ///
    /// The Green's-function energy vector handed to the next GW pass is built as
    ///   E_in^(k+1) = (1 - alpha) E_in^(k) + alpha E_out^(k)
    /// where `E_out` is the result of one full GW pass.  `alpha = 1.0` is the
    /// historical, undamped fixed-point iteration; smaller values damp the
    /// eigenvalue self-consistency and are the scalar analogue of the
    /// Z-scaled (linearized) quasiparticle update used by MolGW's GnWn loop.
    pub evgw_damping:f64,
    /// Enable Pulay/DIIS (Anderson) extrapolation of the evGW quasiparticle
    /// energy vector, exactly as `pyscf.gw.evgw` does on `mo_energy`.
    ///
    /// It is applied **on top of** whatever per-round map `evgw_solver`
    /// selects.  That stacking is what makes a single parameter set work for
    /// both classes of REST evGW difficulty: `molgw`'s single evaluation
    /// removes the sensitivity to the coarse real-axis grid (CH4, CO2), while
    /// DIIS damps the antisymmetric mode of near-degenerate pairs that a scalar
    /// (diagonal) step cannot control (NH3, C6H6).
    pub evgw_diis:bool,
    /// Number of residual vectors kept in the DIIS history.
    pub evgw_diis_space:usize,
    /// Minimum number of stored residuals before DIIS extrapolation is used.
    pub evgw_diis_start:usize,
    /// Use the strict bracket-restricted fallback for the DIIS step.  Default
    /// `false` = standard Pulay/DIIS behaviour (free extrapolation, rejected
    /// only when non-finite or clearly runaway).
    pub evgw_diis_safeguard:bool,
    /// Tolerance (Ha) for the evGW degeneracy projection.
    ///
    /// evGW must keep symmetry-degenerate partners degenerate.  REST's integrals
    /// and DFT grid break that symmetry numerically (its Kohn-Sham partners
    /// differ by ~1e-5 Ha) and the evGW map amplifies the resulting
    /// antisymmetric mode: for C6H6 the E1g HOMO pair goes from a 1.8e-5 Ha KS
    /// splitting to 8.2e-3 Ha and then oscillates, which is what prevents
    /// convergence.  With a positive tolerance, orbitals whose *Kohn-Sham*
    /// energies lie within the tolerance are updated as a block (their
    /// quasiparticle energies are replaced by the group average) every round.
    /// MolGW does not need this because its integrals keep the pairs exactly
    /// degenerate.
    ///
    /// Grouping uses the fixed KS spectrum, so it cannot drift during the
    /// iteration.  Default `1e-4` Ha; set to `0.0` to disable.
    pub evgw_degeneracy_tol:f64,
    /// MolGW-style handling of the orbitals outside the explicitly computed
    /// window during evGW.
    ///
    /// * `false` (default): REST historical behaviour -- extrapolate their
    ///   quasiparticle energies with the rigid shift of the lowest/highest
    ///   computed orbital.
    /// * `true`: keep them at their initial (Kohn-Sham) energies, exactly as
    ///   MolGW's `find_qp_energy_linearization` does (it initialises
    ///   `energy_qp_z = energy0` and only overwrites `nsemin..nsemax`, so
    ///   out-of-window states are fed back unchanged).
    pub evgw_freeze_outside:bool,
    /// Finite-difference step (Ha) used by the MolGW-style `z_update` solver to
    /// evaluate `dSigma_c/domega` for the Z factor.  It must be larger than the
    /// scale on which the self-energy varies sharply (the smallest
    /// imaginary-axis quadrature frequency).
    /// Default 0.01 (MolGW's `step_sigma` default).
    pub evgw_z_step:f64,
    /// Optional hard cap (Ha) on the per-orbital quasiparticle-energy change of
    /// one evGW round.  `0.0` (default) = unlimited.  This is the practical
    /// analogue of MolGW's `Z <= 1` clamp: it bounds the step taken by the
    /// eigenvalue self-consistency and prevents a quasiparticle-equation root
    /// solve from jumping to a spurious distant root.
    pub evgw_max_step:f64,
    /// Convergence threshold (Ha) on max_n |E_n^out - E_n^in| for one evGW round.
    pub evgw_conv_tol:f64,
    /// Stop the evGW loop as soon as `evgw_conv_tol` is met (otherwise always
    /// run the full `evgw_rounds` passes).
    pub evgw_stop_on_convergence:bool,
    /// Print a per-round convergence report (|dE|, |dG|, HOMO/LUMO).
    pub evgw_report:bool,
    pub save_gw_homo_lumo_qp:bool,
    /// Write a GW/evGW checkpoint (archive) to `gw_checkpoint_path`.
    ///
    /// When `true`, REST writes the complete post-SCF state (compatibility
    /// metadata, the converged SCF arrays, the GW quasiparticle energies and
    /// the evGW outer-loop progress) to disk **after GW finishes** and **at the
    /// end of every evGW round**.  A later run with
    /// `resume_from_checkpoint = true` can then continue from that file
    /// instead of redoing the SCF and the GW/evGW work.  Default `false`.
    pub save_gw_checkpoint:bool,
    /// Path of the GW/evGW checkpoint file (HDF5).  Default `"./gw_checkpoint.h5"`.
    ///
    /// The file is written atomically: the archive goes to `<path>.tmp` first
    /// and is renamed onto `<path>` once it is closed, so a job killed in the
    /// middle of a write does not leave a corrupt checkpoint behind.
    pub gw_checkpoint_path:String,
    /// Read `gw_checkpoint_path` and resume instead of redoing previous work.
    ///
    /// * If the checkpoint represents a finished GW, the SCF iterations and the
    ///   whole GW/evGW calculation are skipped and the run continues directly
    ///   into the BSE/response step.
    /// * If it represents an interrupted evGW round, the SCF iterations are
    ///   skipped and the evGW loop continues from the saved round (DIIS history
    ///   included).
    ///
    /// The checkpoint is validated against the current input (basis, electron
    /// count, charge, spin, geometry, array dimensions) and a clear error is
    /// raised on any mismatch.  Default `false`.
    pub resume_from_checkpoint:bool,
    pub save_qp_path:String,
    pub save_first_excitation:bool,
    pub save_first_excitation_path:String,
    pub use_low_rank_contour:bool,
    pub low_rank_grid_type:String,   // "linear" or "quadratic" (power-law, denser near zero)
    pub nomega_chi_real:usize,
    /// How the real-axis low-rank `v*chi*v` is obtained between grid points.
    ///
    /// * `"linear"` (default): linear interpolation between the two bracketing
    ///   grid points.  Removes the artificial discontinuity of the
    ///   piecewise-constant rule and is the more accurate choice.
    /// * `"nearest"`: piecewise constant (nearest grid point).  Historical REST
    ///   behaviour and the rule MolGW uses in `sf_interpolate_vsqrt_chi_vsqrt`;
    ///   kept for reproducing older numbers and for A/B comparison.
    pub low_rank_interp:String,
    pub low_rank_tolerance:f64,
    pub omega_chi_max:f64,          // real-axis freq grid max (Ha); 0=auto from de_max
    pub nomega_sigma:usize,         // number of sigma sampling points on each side (de_max scan)
    pub step_sigma:f64,             // spacing of sigma grid in Ha (de_max scan)
    pub fourier_self_energy:bool,
    pub fse_sin_coeff_path:String,
    pub fse_cos_coeff_path:String,
    pub hermite_self_energy:bool,
    pub hermite_coeff_path:String,
    pub parse_qp_path:String,
    pub bse_qp_polarization:bool,
    pub gw_extrapolate_occ_threshold:f64,
    pub gw_extrapolate_vir_threshold:f64,
    pub external_field_freq:f64,
    pub lifetime_gamma:f64,
    pub gw_or_bse:String,
    pub gw_span_energy:f64,
    pub gw_search_grid:usize,
    pub bse_max_ang_momentum:usize,
    pub gw_rootfinder:String,
    pub simplified_bse:bool,
    pub self_energy_spectrum_test:bool,
    pub spectrum_test_start:f64,
    pub spectrum_test_end:f64,
    pub spectrum_test_step:f64,
    pub pysoc:bool,
    pub gw_imag_rayon:bool,
    pub bse_auxbas_path: Option<String>,
    pub bse_feast_renormalized_doubles: bool,
    pub bse_renormalized_doubles_extra_width: f64,
    pub bse_feast_precondition_type: String,
    pub bse_feast_inner_gmres_tol: f64,
    pub bse_feast_inner_gmres_restart: usize,
    pub bse_feast_inner_gmres_max_iter: usize,
    // QSGW-specific controls
    pub qsgw_max_iter: usize,
    pub qsgw_energy_tol: f64,
    pub qsgw_mix_param: f64,
    pub qsgw_eta: f64,
    /// Lorentzian broadening (Ha) for CD-GW real-axis contour deformation.
    /// Default 0.0 (no broadening, backward compatible). 0.01 recommended for stability.
    pub cdgw_eta: f64,
    /// Numerical tolerance (Ha) for residue pole detection in CD-GW:
    /// - de > -tol  → include the residue pole
    /// - |de| < tol → pole lies on the contour, apply half-weight (×0.5)
    /// - de_max scan: only collect de > tol
    /// Must be a small numerical tolerance (∼1e-3), NOT a physics broadening.
    /// Default 0.001 (matching MolGW). Separated from cdgw_eta to prevent
    /// large broadening from incorrectly halving off-contour residues.
    pub cdgw_res_tol: f64,
    /// Number of imaginary-axis self-energy samples used for Padé continuation.
    pub ac_num_samples: usize,
    /// Maximum external imaginary frequency (Ha) used for Padé sampling.
    pub ac_omega_max: f64,
    /// Positive broadening (Ha) used to evaluate Sigma(omega + i*eta).
    pub ac_eta: f64,
    /// Number of states below/above HOMO-LUMO for the self-energy evaluation
    /// and for the real-axis de_max scan. States far from the gap (e.g. core states)
    /// contribute negligibly to the real-axis residues and inflate de_max, making
    /// the low-rank grid excessively sparse. Restricting this range to ~20-50
    /// around HOMO gives a much finer de_max grid that captures the pole structure
    /// properly — exactly as MolGW does with selfenergy_state_range.
    /// Default 100000 (essentially all states, backward compatible).
    pub selfenergy_state_range: usize,
    /// Restrict the `de_max` scan that sizes the low-rank real-axis grid to the
    /// window of orbitals explicitly computed in the current run (only used by
    /// the `extrapolated` GW scheme).
    ///
    /// * `false` (default): historical REST behaviour -- scan all states.
    ///   `de_max` then reaches 10-20 Ha for molecules with a deep core orbital,
    ///   which makes the uniform real-axis grid far too coarse for the valence
    ///   region (measured single-pass low-rank error: 1.4e-2 Ha for CO2 and
    ///   1.4e-1 Ha for C6H6 at `nomega_chi_real = 64`).
    /// * `true`: scan only the states whose self-energy is actually evaluated,
    ///   as MolGW does (`de_max` is computed over `nsemin..nsemax`).  This
    ///   shrinks `de_max` to ~1 Ha and improves the single-pass low-rank error
    ///   by 170-640x; it turned CO2's evGW from non-converging into 13-round
    ///   convergence.  **But it is not a free win**: for CH4 the restriction
    ///   makes `de_max` so small that many residues are clamped to the grid
    ///   boundary, and CH4 goes from deterministic 21-round convergence to
    ///   nondeterministic behaviour.  Hence it is opt-in, not the default.
    pub low_rank_demax_window: bool,
    // response BSE grid sampling parameters
    pub response_bse_x_start: f64,
    pub response_bse_x_end: f64,
    pub response_bse_x_points: usize,
    pub response_bse_y_start: f64,
    pub response_bse_y_end: f64,
    pub response_bse_y_points: usize,
    pub response_bse_z_start: f64,
    pub response_bse_z_end: f64,
    pub response_bse_z_points: usize,
    pub response_bse_grids: Vec<[f64; 3]>,
    // response BSE solver selection
    pub response_bse_solver: String,
    pub response_bse_tol: f64,
    pub response_bse_max_iter: usize,
    // FEAST solver control and parameters for BSE
    pub bse_feast_solver: bool,
    pub bse_eigenrange_min: f64,
    pub bse_eigenrange_max: f64,
    pub bse_m_expected: usize,
    pub bse_max_feast_iter: usize,
    pub bse_tol_feast: f64,
    pub bse_feast_cg_max_iter: usize,
    pub bse_feast_cg_tol: f64,
    pub bse_feast_gmres_restart: usize,
    pub bse_feast_gmres_max_iter: usize,
    // FEAST initial guess type: "random" (default) or "gaussian"
    pub bse_feast_init_guess_type: String,
    // Gaussian width = (step * width_factor)²  (default 0.5 → half-spacing)
    pub bse_feast_gaussian_width_factor: f64,
    // Parallelise over quadrature points via rayon (default true).
    // If false, quadrature points are solved one by one in serial.
    pub bse_feast_contour_rayon: bool,
    // Number of Gauss-Legendre quadrature points for contour integration (default 8)
    pub bse_feast_n_quad: usize,
    // NLFEAST (nonlinear BSE) control parameters
    pub nonlinear_bse: bool,
    pub nlfeast_centre: f64,
    pub nlfeast_radius: f64,
    pub nlfeast_m0: usize,
    pub nlfeast_n_quad: usize,
    pub nlfeast_max_iter: usize,
    pub nlfeast_tol: f64,
    pub nlfeast_gmres_restart: usize,
    pub nlfeast_gmres_max_it: usize,
    pub nlfeast_gmres_tol: f64,
    /// Dynamical kernel used by `gw_or_bse = "dynamic_bse"`:
    ///   "bse"  — number-conserving RPA-pair kernel on top of the screened
    ///            static GW-BSE singles block (default);
    ///   "srpa" — bare-Coulomb sRPA: TDHF static singles block plus the bare
    ///            two-particle–two-hole coupling (Eq. (7) of Sangalli 2011).
    pub dynamic_bse_kernel: String,
    pub export_matvec_count: bool,
    /// Which implementation performs the implicit BSE matrix-vector products.
    ///
    /// * `"mo"` (default): the historical MO-basis matvec in
    ///   `ri_bse::matvec`, which materialises the `[n_aux, n_o, n_v]`,
    ///   `[n_aux, n_o, n_o]` and `[n_aux, n_v, n_v]` RI tensors together with
    ///   the `[n_o*n_v, n_o*n_v]` W matrices.
    /// * `"ao"`: the AO-basis matvec in `ri_bse::matvec_fast`, which folds the
    ///   MO coefficients directly into the packed AO-basis RI tensor
    ///   (`rimatr_bse`) and keeps the auxiliary index explicit, so neither the
    ///   `[n_o*n_v, n_o*n_v]` matrices nor any fully transformed tensor is ever
    ///   formed.
    ///
    /// The historical spellings `"fast"` / `"memory-efficient"` are still
    /// accepted as aliases for `"mo"` / `"ao"`.
    ///
    /// `"ao"` requires `use_ri_symm = true` and a BSE auxiliary basis
    /// (`bse_auxbas_path`); otherwise the run falls back to `"mo"` with a
    /// printed warning.
    pub bse_matvec_style: String,
    /// Which tensor representation the GW module uses for the RI three-centre
    /// integrals and the screened interaction.
    ///
    /// * `"mo"` (default): the historical route.  Every RI tensor is produced
    ///   by the MO transformation `ao2mo_rayon` (`scf_data.rimatr` →
    ///   `dsymm` against the *full* eigenvector, once per auxiliary function
    ///   and, in `w_c_matrix_from_scf`, once per block of target orbitals).
    /// * `"ao"`: the AO-basis route in `ri_gw::tensor_ao`.  The packed AO RI
    ///   tensor is folded with MO coefficients directly,
    ///   `(Q|nm) = sum_{mu,nu} J_Q[mu,nu] X[mu,n] X[nu,m]`, and the half
    ///   transformation `Y_Q[mu,m] = sum_nu J_Q[mu,nu] X[nu,m]` is computed
    ///   **once** and shared by every orbital row.  The cost of all `(Q|nm)`
    ///   drops from `O(n_aux n_bas^2 n_mo^2 / block)` to
    ///   `O(n_aux n_bas^2 n_mo + n_aux n_bas n_mo^2)`; no `[n_aux, n_mo, n_mo]`
    ///   array is ever materialised.
    ///
    /// This is the GW analogue of the AO-basis BSE matvec: the auxiliary index
    /// stays explicit and MO-coefficient folds replace pre-transformed
    /// MO-basis tensors.
    pub gw_tensor_style: String,
    /// AO-pair pre-screening threshold (a.u.) used by `gw_tensor_style = "ao"`.
    ///
    /// An AO pair `(mu,nu)` is dropped from every fold when
    /// `max_Q |J_Q[mu,nu]|` is below this threshold.  `0.0` disables screening
    /// (bit-for-bit the unscreened result).
    pub gw_ao_screening_tol: f64,
    pub gw_switch_fallback_threshold: f64,
}

impl Default for QuasiParticle {
    fn default() -> Self {
        QuasiParticle {
            gw_scheme:String::from("no gw"),
            gw_variant: GwVariant::Cd,
            homo_lumo_gw_qp:false,
            x_alpha:0.5,
            save_qp:false,
            bse_davidson_solver:false,
            davidson_target_excitations:6,
            davidson_converge_threshold:1e-10,
            davidson_maximum_subspace_size:60,
            davidson_restart_dimensions:6,
            davidson_add_dimensions:6,
            davidson_max_iter:100,
            bse_tda:false,
            print_nto:false,
            bse_spin:String::from("none"),
            bse_cutoff_energy:1000000.0,
            save_bse_terms:false,
            obtain_vx_vc_terms:false,
            obtain_pure_exchange:false,
            obtain_ks_homos:false,
            gw_linearize_shift:1e-2,
            gw_linearize_derivative_h:1e-10,
            renormalized_singles:false,
            w_rs:false,
            rs_full_space:false,
            rs_use_rs_orbitals:false,
            scgw:String::from("g0w0"),
            gw:false,
            save_bse_excitations:false, 
            evgw_rounds:0,
            evgw_solver:String::from("molgw"),
            evgw_damping:1.0,
            evgw_diis:true,
            evgw_diis_space:8,
            evgw_diis_start:2,
            evgw_diis_safeguard:false,
            evgw_freeze_outside:true,
            evgw_degeneracy_tol:1e-4,
            evgw_z_step:0.01,
            evgw_max_step:0.0,
            evgw_conv_tol:1e-5,
            evgw_stop_on_convergence:true,
            evgw_report:true,
            save_gw_homo_lumo_qp:false,
            save_gw_checkpoint:false,
            gw_checkpoint_path:String::from("./gw_checkpoint.h5"),
            resume_from_checkpoint:false,
            save_qp_path:String::from("single_qp_path.txt"),
            save_first_excitation:false,
            save_first_excitation_path:String::from("first_excitation_save.txt"),
            use_low_rank_contour:false,
            low_rank_grid_type:String::from("linear"),
            nomega_chi_real:6,
            low_rank_interp:String::from("linear"),
            low_rank_tolerance:1e-3,
            omega_chi_max:0.0,
            nomega_sigma:10,
            step_sigma:0.05,
            parse_qp_path:String::from("./qp_energies"),
            fourier_self_energy:false,
            fse_sin_coeff_path:String::from("./fse_sin_coeff.txt"),
            fse_cos_coeff_path:String::from("./fse_cos_coeff.txt"),
            hermite_self_energy:false,
            hermite_coeff_path:String::from("./hermite_coeff.txt"),
            bse_qp_polarization:false,
            gw_extrapolate_occ_threshold:0.1,
            gw_extrapolate_vir_threshold:0.1,
            gw_or_bse:String::new(),
            gw_span_energy:0.2,
            bse_exchange_rescaling:1.0,
            gw_search_grid:51,
            gw_rootfinder:"newton".to_string(),
            simplified_bse:false,
            bse_max_ang_momentum:10,
            self_energy_spectrum_test:false,
            spectrum_test_start:-1.0,
            spectrum_test_end:0.0,
            spectrum_test_step:0.01,
            pysoc:false,
            external_field_freq:0.5,
            lifetime_gamma:0.001,
            gw_imag_rayon:true,
            bse_auxbas_path: None,
            bse_feast_renormalized_doubles: false,
            bse_renormalized_doubles_extra_width: 0.5,
            bse_feast_precondition_type: String::from("inner_gmres"),
            bse_feast_inner_gmres_tol: 0.0001,
            bse_feast_inner_gmres_restart: 50,
            bse_feast_inner_gmres_max_iter: 100,
            qsgw_max_iter: 50,
            qsgw_energy_tol: 1e-5,
            qsgw_mix_param: 0.5,
            qsgw_eta: 0.001,
            cdgw_eta: 0.001,
            cdgw_res_tol: 0.001,
            ac_num_samples: 16,
            ac_omega_max: 5.0,
            ac_eta: 0.001,
            selfenergy_state_range: 100000,
            low_rank_demax_window: false,
            // response BSE grid sampling parameters (default: 2 points per dimension)
            response_bse_x_start: 0.0,
            response_bse_x_end: 1.0,
            response_bse_x_points: 2,
            response_bse_y_start: 0.0,
            response_bse_y_end: 1.0,
            response_bse_y_points: 2,
            response_bse_z_start: 0.0,
            response_bse_z_end: 1.0,
            response_bse_z_points: 2,
            response_bse_grids: Vec::new(),
            response_bse_solver: String::from("klopper"),
            response_bse_tol: 1e-6,
            response_bse_max_iter: 200,
            bse_feast_solver: false,
            bse_eigenrange_min: 0.0,
            bse_eigenrange_max: 0.5,
            bse_m_expected: 20,
            bse_max_feast_iter: 30,
            bse_tol_feast: 1e-8,
            bse_feast_cg_max_iter: 100,
            bse_feast_cg_tol: 1e-8,
            bse_feast_gmres_restart: 200,
            bse_feast_gmres_max_iter: 500,
            bse_feast_init_guess_type: String::from("random"),
            bse_feast_gaussian_width_factor: 0.5,
            bse_feast_contour_rayon: true,
            bse_feast_n_quad: 8,
            nonlinear_bse: false,
            nlfeast_centre: 0.0,
            nlfeast_radius: 0.5,
            nlfeast_m0: 20,
            nlfeast_n_quad: 12,
            nlfeast_max_iter: 20,
            nlfeast_tol: 1e-8,
            nlfeast_gmres_restart: 200,
            nlfeast_gmres_max_it: 500,
            nlfeast_gmres_tol: 1e-6,
            dynamic_bse_kernel: String::from("bse"),
            export_matvec_count: false,
            // BSE defaults to the AO-basis ("memory-efficient") matvec; GW keeps MO.
            bse_matvec_style: String::from("ao"),
            gw_tensor_style: String::from("mo"),
            gw_ao_screening_tol: 0.0,
            gw_switch_fallback_threshold: 1e6,
        }
    }
}

impl QuasiParticle { 
    pub fn to_toml(&self) -> toml::Value {
        let mut table = toml::map::Map::new();
        
        table.insert("gw_scheme".to_string(), toml::Value::String(self.gw_scheme.clone()));
        table.insert(
            "gw_variant".to_string(),
            toml::Value::String(self.gw_variant.as_str().to_string()),
        );
        table.insert("homo_lumo_gw_qp".to_string(), toml::Value::Boolean(self.homo_lumo_gw_qp));
        table.insert("x_alpha".to_string(), toml::Value::Float(self.x_alpha));
        table.insert("save_qp".to_string(), toml::Value::Boolean(self.save_qp));
        table.insert("print_nto".to_string(), toml::Value::Boolean(self.print_nto));
        table.insert("bse_davidson_solver".to_string(), toml::Value::Boolean(self.bse_davidson_solver));
        table.insert("davidson_target_excitations".to_string(), toml::Value::Integer(self.davidson_target_excitations as i64));
        table.insert("davidson_converge_threshold".to_string(), toml::Value::Float(self.davidson_converge_threshold));
        table.insert("davidson_maximum_subspace_size".to_string(), toml::Value::Integer(self.davidson_maximum_subspace_size as i64));
        table.insert("davidson_restart_dimensions".to_string(), toml::Value::Integer(self.davidson_restart_dimensions as i64));
        table.insert("davidson_add_dimensions".to_string(), toml::Value::Integer(self.davidson_add_dimensions as i64));
        table.insert("davidson_max_iter".to_string(), toml::Value::Integer(self.davidson_max_iter as i64));
        table.insert("bse_tda".to_string(), toml::Value::Boolean(self.bse_tda));
        table.insert("bse_spin".to_string(), toml::Value::String(self.bse_spin.clone()));
        table.insert("bse_cutoff_energy".to_string(), toml::Value::Float(self.bse_cutoff_energy));
        table.insert("save_bse_terms".to_string(), toml::Value::Boolean(self.save_bse_terms));
        table.insert("obtain_vx_vc_terms".to_string(), toml::Value::Boolean(self.obtain_vx_vc_terms));
        table.insert("obtain_pure_exchange".to_string(), toml::Value::Boolean(self.obtain_pure_exchange));
        table.insert("obtain_ks_homos".to_string(), toml::Value::Boolean(self.obtain_ks_homos));
        table.insert("gw_linearize_shift".to_string(), toml::Value::Float(self.gw_linearize_shift));
        table.insert("gw_linearize_derivative_h".to_string(), toml::Value::Float(self.gw_linearize_derivative_h));
        table.insert("renormalized_singles".to_string(), toml::Value::Boolean(self.renormalized_singles));
        table.insert("w_rs".to_string(), toml::Value::Boolean(self.w_rs));
        table.insert("rs_full_space".to_string(), toml::Value::Boolean(self.rs_full_space));
        table.insert("rs_use_rs_orbitals".to_string(), toml::Value::Boolean(self.rs_use_rs_orbitals));
        table.insert("scgw".to_string(), toml::Value::String(self.scgw.clone()));
        table.insert("gw_rootfinder".to_string(), toml::Value::String(self.gw_rootfinder.clone()));
        table.insert("gw".to_string(), toml::Value::Boolean(self.gw));
        table.insert("save_bse_excitations".to_string(), toml::Value::Boolean(self.save_bse_excitations));
        table.insert("evgw_rounds".to_string(), toml::Value::Integer(self.evgw_rounds as i64));
        table.insert("evgw_solver".to_string(), toml::Value::String(self.evgw_solver.clone()));
        table.insert("evgw_damping".to_string(), toml::Value::Float(self.evgw_damping));
        table.insert("evgw_diis".to_string(), toml::Value::Boolean(self.evgw_diis));
        table.insert("evgw_diis_space".to_string(), toml::Value::Integer(self.evgw_diis_space as i64));
        table.insert("evgw_diis_start".to_string(), toml::Value::Integer(self.evgw_diis_start as i64));
        table.insert("evgw_diis_safeguard".to_string(), toml::Value::Boolean(self.evgw_diis_safeguard));
        table.insert("evgw_freeze_outside".to_string(), toml::Value::Boolean(self.evgw_freeze_outside));
        table.insert("evgw_degeneracy_tol".to_string(), toml::Value::Float(self.evgw_degeneracy_tol));
        table.insert("evgw_z_step".to_string(), toml::Value::Float(self.evgw_z_step));
        table.insert("evgw_max_step".to_string(), toml::Value::Float(self.evgw_max_step));
        table.insert("evgw_conv_tol".to_string(), toml::Value::Float(self.evgw_conv_tol));
        table.insert("evgw_stop_on_convergence".to_string(), toml::Value::Boolean(self.evgw_stop_on_convergence));
        table.insert("evgw_report".to_string(), toml::Value::Boolean(self.evgw_report));
        table.insert("save_gw_homo_lumo_qp".to_string(), toml::Value::Boolean(self.save_gw_homo_lumo_qp));
        table.insert("save_gw_checkpoint".to_string(), toml::Value::Boolean(self.save_gw_checkpoint));
        table.insert("gw_checkpoint_path".to_string(), toml::Value::String(self.gw_checkpoint_path.clone()));
        table.insert("resume_from_checkpoint".to_string(), toml::Value::Boolean(self.resume_from_checkpoint));
        table.insert("save_qp_path".to_string(), toml::Value::String(self.save_qp_path.clone()));
        table.insert("save_first_excitation".to_string(), toml::Value::Boolean(self.save_first_excitation));
        table.insert("save_first_excitation_path".to_string(), toml::Value::String(self.save_first_excitation_path.clone()));
        table.insert("use_low_rank_contour".to_string(), toml::Value::Boolean(self.use_low_rank_contour));
        table.insert("low_rank_grid_type".to_string(), toml::Value::String(self.low_rank_grid_type.clone()));
        table.insert("low_rank_interp".to_string(), toml::Value::String(self.low_rank_interp.clone()));
        table.insert("nomega_chi_real".to_string(), toml::Value::Integer(self.nomega_chi_real as i64));
        table.insert("low_rank_tolerance".to_string(), toml::Value::Float(self.low_rank_tolerance));
        table.insert("omega_chi_max".to_string(), toml::Value::Float(self.omega_chi_max));
        table.insert("nomega_sigma".to_string(), toml::Value::Integer(self.nomega_sigma as i64));
        table.insert("step_sigma".to_string(), toml::Value::Float(self.step_sigma));
        table.insert("fse_sin_coeff_path".to_string(), toml::Value::String(self.fse_sin_coeff_path.clone()));
        table.insert("fse_cos_coeff_path".to_string(), toml::Value::String(self.fse_cos_coeff_path.clone()));
        table.insert("hermite_self_energy".to_string(), toml::Value::Boolean(self.hermite_self_energy));
        table.insert("hermite_coeff_path".to_string(), toml::Value::String(self.hermite_coeff_path.clone()));
        table.insert("parse_qp_path".to_string(), toml::Value::String(self.parse_qp_path.clone()));
        table.insert("bse_qp_polarization".to_string(), toml::Value::Boolean(self.bse_qp_polarization));
        table.insert("gw_extrapolate_occ_threshold".to_string(), toml::Value::Float(self.gw_extrapolate_occ_threshold));
        table.insert("gw_extrapolate_vir_threshold".to_string(), toml::Value::Float(self.gw_extrapolate_vir_threshold));
        table.insert("gw_span_energy".to_string(), toml::Value::Float(self.gw_span_energy));
        table.insert("external_field_freq".to_string(), toml::Value::Float(self.external_field_freq));
        table.insert("lifetime_gamma".to_string(), toml::Value::Float(self.lifetime_gamma));
        table.insert("gw_or_bse".to_string(), toml::Value::String(self.gw_or_bse.clone()));
        table.insert("gw_search_grid".to_string(), toml::Value::Integer(self.gw_search_grid as i64));
        table.insert("simplified_bse".to_string(), toml::Value::Boolean(self.simplified_bse));
        table.insert("pysoc".to_string(), toml::Value::Boolean(self.pysoc));
        table.insert("gw_imag_rayon".to_string(), toml::Value::Boolean(self.gw_imag_rayon));
        table.insert("fourier_self_energy".to_string(), toml::Value::Boolean(self.fourier_self_energy));
        table.insert("bse_exchange_rescaling".to_string(), toml::Value::Float(self.bse_exchange_rescaling));
        table.insert("self_energy_spectrum_test".to_string(), toml::Value::Boolean(self.self_energy_spectrum_test));
        table.insert("spectrum_test_start".to_string(), toml::Value::Float(self.spectrum_test_start));
        table.insert("spectrum_test_end".to_string(), toml::Value::Float(self.spectrum_test_end));
        table.insert("spectrum_test_step".to_string(), toml::Value::Float(self.spectrum_test_step));
        table.insert("bse_max_ang_momentum".to_string(), toml::Value::Integer(self.bse_max_ang_momentum as i64));
        if let Some(path) = &self.bse_auxbas_path {
            table.insert("bse_auxbas_path".to_string(), toml::Value::String(path.clone()));
        }
        table.insert("bse_feast_renormalized_doubles".to_string(), toml::Value::Boolean(self.bse_feast_renormalized_doubles));
        table.insert("bse_renormalized_doubles_extra_width".to_string(), toml::Value::Float(self.bse_renormalized_doubles_extra_width));
        table.insert("bse_feast_precondition_type".to_string(), toml::Value::String(self.bse_feast_precondition_type.clone()));
        table.insert("bse_feast_inner_gmres_tol".to_string(), toml::Value::Float(self.bse_feast_inner_gmres_tol));
        table.insert("bse_feast_inner_gmres_restart".to_string(), toml::Value::Integer(self.bse_feast_inner_gmres_restart as i64));
        table.insert("bse_feast_inner_gmres_max_iter".to_string(), toml::Value::Integer(self.bse_feast_inner_gmres_max_iter as i64));
        table.insert("qsgw_max_iter".to_string(), toml::Value::Integer(self.qsgw_max_iter as i64));
        table.insert("qsgw_energy_tol".to_string(), toml::Value::Float(self.qsgw_energy_tol));
        table.insert("qsgw_mix_param".to_string(), toml::Value::Float(self.qsgw_mix_param));
        table.insert("qsgw_eta".to_string(), toml::Value::Float(self.qsgw_eta));
        table.insert("cdgw_eta".to_string(), toml::Value::Float(self.cdgw_eta));
        table.insert("cdgw_res_tol".to_string(), toml::Value::Float(self.cdgw_res_tol));
        table.insert("ac_num_samples".to_string(), toml::Value::Integer(self.ac_num_samples as i64));
        table.insert("ac_omega_max".to_string(), toml::Value::Float(self.ac_omega_max));
        table.insert("ac_eta".to_string(), toml::Value::Float(self.ac_eta));
        table.insert("low_rank_demax_window".to_string(), toml::Value::Boolean(self.low_rank_demax_window));
        table.insert("selfenergy_state_range".to_string(), toml::Value::Integer(self.selfenergy_state_range as i64));
        table.insert("response_bse_x_start".to_string(), toml::Value::Float(self.response_bse_x_start));
        table.insert("response_bse_x_end".to_string(), toml::Value::Float(self.response_bse_x_end));
        table.insert("response_bse_x_points".to_string(), toml::Value::Integer(self.response_bse_x_points as i64));
        table.insert("response_bse_y_start".to_string(), toml::Value::Float(self.response_bse_y_start));
        table.insert("response_bse_y_end".to_string(), toml::Value::Float(self.response_bse_y_end));
        table.insert("response_bse_y_points".to_string(), toml::Value::Integer(self.response_bse_y_points as i64));
        table.insert("response_bse_z_start".to_string(), toml::Value::Float(self.response_bse_z_start));
        table.insert("response_bse_z_end".to_string(), toml::Value::Float(self.response_bse_z_end));
        table.insert("response_bse_z_points".to_string(), toml::Value::Integer(self.response_bse_z_points as i64));
        table.insert("bse_eigenrange_min".to_string(), toml::Value::Float(self.bse_eigenrange_min));
        table.insert("bse_feast_solver".to_string(), toml::Value::Boolean(self.bse_feast_solver));
        table.insert("bse_eigenrange_max".to_string(), toml::Value::Float(self.bse_eigenrange_max));
        table.insert("bse_m_expected".to_string(), toml::Value::Integer(self.bse_m_expected as i64));
        table.insert("bse_max_feast_iter".to_string(), toml::Value::Integer(self.bse_max_feast_iter as i64));
        table.insert("bse_tol_feast".to_string(), toml::Value::Float(self.bse_tol_feast));
        table.insert("bse_feast_cg_max_iter".to_string(), toml::Value::Integer(self.bse_feast_cg_max_iter as i64));
        table.insert("bse_feast_cg_tol".to_string(), toml::Value::Float(self.bse_feast_cg_tol));
        table.insert("bse_feast_gmres_restart".to_string(), toml::Value::Integer(self.bse_feast_gmres_restart as i64));
        table.insert("bse_feast_gmres_max_iter".to_string(), toml::Value::Integer(self.bse_feast_gmres_max_iter as i64));
        table.insert("bse_feast_init_guess_type".to_string(), toml::Value::String(self.bse_feast_init_guess_type.clone()));
        table.insert("bse_feast_gaussian_width_factor".to_string(), toml::Value::Float(self.bse_feast_gaussian_width_factor));
        table.insert("bse_feast_contour_rayon".to_string(), toml::Value::Boolean(self.bse_feast_contour_rayon));
        table.insert("bse_feast_n_quad".to_string(), toml::Value::Integer(self.bse_feast_n_quad as i64));
        table.insert("response_bse_solver".to_string(), toml::Value::String(self.response_bse_solver.clone()));
        table.insert("response_bse_tol".to_string(), toml::Value::Float(self.response_bse_tol));
        table.insert("response_bse_max_iter".to_string(), toml::Value::Integer(self.response_bse_max_iter as i64));
        table.insert("nonlinear_bse".to_string(), toml::Value::Boolean(self.nonlinear_bse));
        table.insert("nlfeast_centre".to_string(), toml::Value::Float(self.nlfeast_centre));
        table.insert("nlfeast_radius".to_string(), toml::Value::Float(self.nlfeast_radius));
        table.insert("nlfeast_m0".to_string(), toml::Value::Integer(self.nlfeast_m0 as i64));
        table.insert("nlfeast_n_quad".to_string(), toml::Value::Integer(self.nlfeast_n_quad as i64));
        table.insert("nlfeast_max_iter".to_string(), toml::Value::Integer(self.nlfeast_max_iter as i64));
        table.insert("nlfeast_tol".to_string(), toml::Value::Float(self.nlfeast_tol));
        table.insert("nlfeast_gmres_restart".to_string(), toml::Value::Integer(self.nlfeast_gmres_restart as i64));
        table.insert("nlfeast_gmres_max_it".to_string(), toml::Value::Integer(self.nlfeast_gmres_max_it as i64));
        table.insert("nlfeast_gmres_tol".to_string(), toml::Value::Float(self.nlfeast_gmres_tol));
        table.insert("dynamic_bse_kernel".to_string(), toml::Value::String(self.dynamic_bse_kernel.clone()));
        table.insert("export_matvec_count".to_string(), toml::Value::Boolean(self.export_matvec_count));
        table.insert("bse_matvec_style".to_string(), toml::Value::String(self.bse_matvec_style.clone()));
        table.insert("gw_tensor_style".to_string(), toml::Value::String(self.gw_tensor_style.clone()));
        table.insert("gw_ao_screening_tol".to_string(), toml::Value::Float(self.gw_ao_screening_tol));
        table.insert("gw_switch_fallback_threshold".to_string(), toml::Value::Float(self.gw_switch_fallback_threshold));
        toml::Value::Table(table)
    }
}

/// Normalise a `"mo"` / `"ao"` style keyword.
///
/// `aliases` lists the accepted spellings of the non-default style; anything
/// unrecognised (including a missing key) falls back to `default`.  The
/// canonical values stored in `QuasiParticle` are always the short forms
/// `"mo"` and `"ao"`.
pub fn normalise_style(
    value: Option<&serde_json::Value>,
    aliases: &[&str],
    default: &str,
) -> String {
    match value {
        Some(serde_json::Value::String(s)) => {
            let lower = s.trim().to_lowercase();
            if aliases.iter().any(|a| *a == lower) {
                String::from("ao")
            } else {
                default.to_string()
            }
        }
        _ => default.to_string(),
    }
}

/// `true` when the stored style selects the AO-basis implementation.
pub fn style_is_ao(style: &str) -> bool {
    matches!(
        style.trim().to_lowercase().as_str(),
        "ao" | "memory-efficient" | "memory_efficient" | "mem-efficient"
    )
}

pub fn parse_quasiparticle_keywords(tmp_keys: &serde_json::Value) -> anyhow::Result<Option<QuasiParticle>> { 
    match tmp_keys.get("quasiparticle_methods").unwrap_or(&serde_json::Value::Null) {
        serde_json::Value::Object(tmp_ctrl) => {
            let mut tmp_input = QuasiParticle::default();
            tmp_input.gw_scheme = match tmp_ctrl.get("gw_scheme").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => String::from("no gw"),
            };
            tmp_input.gw_variant = match tmp_ctrl
                .get("gw_variant")
                .unwrap_or(&serde_json::Value::Null)
            {
                serde_json::Value::String(value) => {
                    match value.trim().to_ascii_lowercase().as_str() {
                        "cd" => GwVariant::Cd,
                        "ac" => GwVariant::Ac,
                        other => {
                            anyhow::bail!(
                                "Invalid gw_variant '{}'. Supported values are 'cd' and 'ac'.",
                                other
                            );
                        }
                    }
                }
                serde_json::Value::Null => GwVariant::Cd,
                other => {
                    anyhow::bail!(
                        "Invalid type for gw_variant: {:?}. Expected the string 'cd' or 'ac'.",
                        other
                    );
                }
            };
            tmp_input.homo_lumo_gw_qp=match tmp_ctrl.get("homo_lumo_gw_qp").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.print_nto = match tmp_ctrl
                .get("print_nto")
                .unwrap_or(&serde_json::Value::Null)
            {
                serde_json::Value::Bool(value) => *value,
                _ => tmp_input.print_nto,
            };
            tmp_input.x_alpha=match tmp_ctrl.get("x_alpha").unwrap_or(&serde_json::Value::Null){
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(2.0_f64)},
                other => {0.5},
            };
            tmp_input.save_qp = match tmp_ctrl.get("save_qp").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.bse_davidson_solver = match tmp_ctrl.get("bse_davidson_solver").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.bse_tda = match tmp_ctrl.get("bse_tda").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.self_energy_spectrum_test= match tmp_ctrl.get("self_energy_spectrum_test").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.spectrum_test_start = match tmp_ctrl.get("spectrum_test_start").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.5_f64)},
                other => {-1.0},
            };
            tmp_input.spectrum_test_end = match tmp_ctrl.get("spectrum_test_end").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.5_f64)},
                other => {0.0},
            };
            tmp_input.spectrum_test_step = match tmp_ctrl.get("spectrum_test_step").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.5_f64)},
                other => {0.01},
            };
            tmp_input.bse_spin = match tmp_ctrl.get("bse_spin").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => String::from("none"),
            };
            tmp_input.bse_cutoff_energy = match tmp_ctrl.get("bse_cutoff_energy").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1000000.0_f64)},
                other => {1000000.0},
            };
            tmp_input.bse_exchange_rescaling = match tmp_ctrl.get("bse_exchange_rescaling").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.0_f64)},
                other => {1.0},
            };
            tmp_input.external_field_freq = match tmp_ctrl.get("external_field_freq").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(0.5_f64)},
                other => {0.5},
            };tmp_input.lifetime_gamma = match tmp_ctrl.get("lifetime_gamma").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(0.001_f64)},
                other => {0.001},
            };
            tmp_input.davidson_converge_threshold = match tmp_ctrl.get("davidson_converge_threshold").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1e-10_f64)},
                other => {1e-10},
            };
            tmp_input.davidson_target_excitations = match tmp_ctrl.get("davidson_target_excitations").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(6_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(6) as usize},
                other => {6}
            };
            tmp_input.davidson_max_iter = match tmp_ctrl.get("davidson_max_iter").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(100_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(100) as usize},
                other => {100}
            };
            let maximum_subspace_size = 100usize.max((tmp_input.davidson_target_excitations as f64 * 20.0).ceil() as usize);
            tmp_input.davidson_maximum_subspace_size = match tmp_ctrl.get("davidson_maximum_subspace_size").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(maximum_subspace_size) as usize},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(maximum_subspace_size as i64) as usize},
                other => {maximum_subspace_size}
            };
            let restart_size = 6usize.max((tmp_input.davidson_target_excitations as f64 * 1.5).ceil() as usize);
            tmp_input.davidson_restart_dimensions = match tmp_ctrl.get("davidson_restart_dimensions").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(restart_size) as usize},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(restart_size as i64) as usize},
                other => {restart_size}
            };
            tmp_input.davidson_add_dimensions = match tmp_ctrl.get("davidson_add_dimensions").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(restart_size) as usize},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(restart_size as i64) as usize},
                other => {restart_size}
            };
            tmp_input.save_bse_terms = match tmp_ctrl.get("save_bse_terms").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.fourier_self_energy = match tmp_ctrl.get("fourier_self_energy").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.hermite_self_energy = match tmp_ctrl.get("hermite_self_energy").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.obtain_vx_vc_terms = match tmp_ctrl.get("obtain_vx_vc_terms").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.pysoc = match tmp_ctrl.get("pysoc").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.gw_imag_rayon = match tmp_ctrl.get("gw_imag_rayon").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {true},
            };
            tmp_input.obtain_pure_exchange = match tmp_ctrl.get("obtain_pure_exchange").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.obtain_ks_homos = match tmp_ctrl.get("obtain_ks_homos").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.gw_linearize_shift= match tmp_ctrl.get("gw_linearize_shift").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(0.01_f64)},
                other => {0.01},
            };
            tmp_input.gw_linearize_derivative_h= match tmp_ctrl.get("gw_linearize_derivative_h").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1e-10_f64)},
                other => {1e-10},
            };
            tmp_input.gw_extrapolate_occ_threshold= match tmp_ctrl.get("gw_extrapolate_occ_threshold").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(0.1)},
                other => {0.1},
            };
            tmp_input.gw_extrapolate_vir_threshold= match tmp_ctrl.get("gw_extrapolate_vir_threshold").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(0.1)},
                other => {0.1},
            };
            tmp_input.gw_span_energy= match tmp_ctrl.get("gw_span_energy").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(0.1)},
                other => {0.1},
            };
            tmp_input.renormalized_singles = match tmp_ctrl.get("renormalized_singles").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.w_rs = match tmp_ctrl.get("w_rs").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.rs_full_space = match tmp_ctrl.get("rs_full_space").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.rs_use_rs_orbitals = match tmp_ctrl.get("rs_use_rs_orbitals").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.scgw = match tmp_ctrl.get("scgw").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => String::from("g0w0"),
            };
            tmp_input.gw = match tmp_ctrl.get("gw").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };

            tmp_input.evgw_rounds = match tmp_ctrl.get("evgw_rounds").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(4_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(4) as usize},
                other => {0}
            };
            tmp_input.evgw_solver = match tmp_ctrl.get("evgw_solver").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => {
                    match s.to_lowercase().as_str() {
                        "z_update" | "z-update" | "zupdate" => String::from("z_update"),
                        "molgw" | "molgw_update" | "plain_eval" | "eval" => String::from("molgw"),
                        _ => String::from("molgw"),
                    }
                },
                _ => String::from("molgw"),
            };
            tmp_input.evgw_damping = match tmp_ctrl.get("evgw_damping").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.parse().unwrap_or(1.0_f64)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1.0)},
                _ => {1.0}
            };
            tmp_input.evgw_diis = match tmp_ctrl.get("evgw_diis").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_bool) => {*tmp_bool},
                _ => {true},
            };
            tmp_input.evgw_diis_space = match tmp_ctrl.get("evgw_diis_space").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(8_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(8) as usize},
                _ => {8}
            };
            tmp_input.evgw_diis_start = match tmp_ctrl.get("evgw_diis_start").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(2_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(2) as usize},
                _ => {2}
            };
            tmp_input.evgw_diis_safeguard = match tmp_ctrl.get("evgw_diis_safeguard").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_bool) => {*tmp_bool},
                _ => {false},
            };
            tmp_input.evgw_freeze_outside = match tmp_ctrl.get("evgw_freeze_outside").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_bool) => {*tmp_bool},
                _ => {true},
            };
            tmp_input.evgw_degeneracy_tol = match tmp_ctrl.get("evgw_degeneracy_tol").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.parse().unwrap_or(1e-4_f64)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1e-4)},
                _ => {1e-4}
            };
            tmp_input.evgw_z_step = match tmp_ctrl.get("evgw_z_step").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.parse().unwrap_or(0.01_f64)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(0.01)},
                _ => {0.01}
            };
            tmp_input.evgw_max_step = match tmp_ctrl.get("evgw_max_step").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.parse().unwrap_or(0.0_f64)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(0.0)},
                _ => {0.0}
            };
            tmp_input.evgw_conv_tol = match tmp_ctrl.get("evgw_conv_tol").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.parse().unwrap_or(1e-5_f64)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1e-5)},
                _ => {1e-5}
            };
            tmp_input.evgw_stop_on_convergence = match tmp_ctrl.get("evgw_stop_on_convergence").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_bool) => {*tmp_bool},
                _ => {true},
            };
            tmp_input.evgw_report = match tmp_ctrl.get("evgw_report").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_bool) => {*tmp_bool},
                _ => {true},
            };
            tmp_input.bse_max_ang_momentum = match tmp_ctrl.get("bse_max_ang_momentum").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(10_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(10) as usize},
                other => {10}
            };
            tmp_input.gw_search_grid = match tmp_ctrl.get("gw_search_grid").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(21_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(21) as usize},
                other => {21}
            };
            tmp_input.save_gw_homo_lumo_qp = match tmp_ctrl.get("save_gw_homo_lumo_qp").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.save_gw_checkpoint = match tmp_ctrl.get("save_gw_checkpoint").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.resume_from_checkpoint = match tmp_ctrl.get("resume_from_checkpoint").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.gw_checkpoint_path = match tmp_ctrl.get("gw_checkpoint_path").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => tmp_input.gw_checkpoint_path.clone(),
            };
            tmp_input.save_bse_excitations = match tmp_ctrl.get("save_bse_excitations").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.simplified_bse = match tmp_ctrl.get("simplified_bse").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.save_qp_path = match tmp_ctrl.get("save_qp_path").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => String::from("single_qp_save.txt"),
            };
            tmp_input.fse_sin_coeff_path = match tmp_ctrl.get("fse_sin_coeff_path").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => String::from("fse_sin_coeff.txt"),
            };
            tmp_input.fse_cos_coeff_path = match tmp_ctrl.get("fse_cos_coeff_path").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => String::from("fse_cos_coeff.txt"),
            };
            tmp_input.hermite_coeff_path = match tmp_ctrl.get("hermite_coeff_path").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => String::from("hermite_coeff.txt"),
            };
            tmp_input.gw_rootfinder = match tmp_ctrl.get("gw_rootfinder").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => String::from("newton".to_string()),
            };
            tmp_input.save_first_excitation = match tmp_ctrl.get("save_first_excitation").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.save_first_excitation_path = match tmp_ctrl.get("save_first_excitation_path").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => String::from("first_excitation_save.txt"),
            };
            tmp_input.use_low_rank_contour = match tmp_ctrl.get("use_low_rank_contour").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                _ => {false},
            };
            tmp_input.low_rank_grid_type = match tmp_ctrl.get("low_rank_grid_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => {
                    let lower = s.to_lowercase();
                    match lower.as_str() {
                        "quadratic" => String::from("quadratic"),
                        _ => String::from("linear"),
                    }
                },
                _ => String::from("linear"),
            };
            tmp_input.nomega_chi_real = match tmp_ctrl.get("nomega_chi_real").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(tmp_str) => {tmp_str.to_lowercase().parse().unwrap_or(6_usize)},
                serde_json::Value::Number(tmp_num) => {tmp_num.as_i64().unwrap_or(6) as usize},
                _ => {6}
            };
            tmp_input.low_rank_interp = match tmp_ctrl.get("low_rank_interp").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => {
                    if s.eq_ignore_ascii_case("nearest") { String::from("nearest") } else { String::from("linear") }
                },
                _ => String::from("linear"),
            };
            tmp_input.low_rank_tolerance = match tmp_ctrl.get("low_rank_tolerance").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(1e-3)},
                _ => {1e-3},
            };
            tmp_input.omega_chi_max = match tmp_ctrl.get("omega_chi_max").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(0.0)},
                _ => {0.0},
            };
            tmp_input.nomega_sigma = match tmp_ctrl.get("nomega_sigma").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_u64().unwrap_or(10) as usize},
                _ => {10},
            };
            tmp_input.step_sigma = match tmp_ctrl.get("step_sigma").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_f64().unwrap_or(0.05)},
                _ => {0.05},
            };
            tmp_input.low_rank_demax_window = match tmp_ctrl.get("low_rank_demax_window").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_bool) => {*tmp_bool},
                _ => {false},
            };
            tmp_input.selfenergy_state_range = match tmp_ctrl.get("selfenergy_state_range").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(tmp_num) => {tmp_num.as_u64().unwrap_or(100000) as usize},
                _ => {100000},
            };
            tmp_input.parse_qp_path = match tmp_ctrl.get("parse_qp_path").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => String::from("./qp_energies"),
            };
            tmp_input.gw_or_bse = match tmp_ctrl.get("gw_or_bse").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone(),
                _ => String::new(),
            };
            tmp_input.bse_qp_polarization = match tmp_ctrl.get("bse_qp_polarization").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(tmp_str) => {*tmp_str},
                other => {false},
            };
            tmp_input.bse_auxbas_path = match tmp_ctrl.get("bse_auxbas_path").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => Some(s.clone()),
                _ => None,
            };
            tmp_input.bse_feast_renormalized_doubles = match tmp_ctrl.get("bse_feast_renormalized_doubles").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(b) => *b,
                _ => false,
            };
            tmp_input.bse_renormalized_doubles_extra_width = match tmp_ctrl.get("bse_renormalized_doubles_extra_width").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.5),
                _ => 0.5,
            };
            tmp_input.bse_feast_precondition_type = match tmp_ctrl.get("bse_feast_precondition_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => {
                    let lower = s.to_lowercase();
                    match lower.as_str() {
                        "inner_gmres" | "diagonal" | "diag" => {
                            if lower == "diag" { String::from("diagonal") }
                            else { lower }
                        },
                        _ => String::from("diagonal"),
                    }
                },
                _ => String::from("diagonal"),
            };
            tmp_input.bse_feast_inner_gmres_tol = match tmp_ctrl.get("bse_feast_inner_gmres_tol").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0001),
                _ => 0.0001,
            };
            tmp_input.bse_feast_inner_gmres_restart = match tmp_ctrl.get("bse_feast_inner_gmres_restart").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(50) as usize,
                _ => 50,
            };
            tmp_input.bse_feast_inner_gmres_max_iter = match tmp_ctrl.get("bse_feast_inner_gmres_max_iter").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(100) as usize,
                _ => 100,
            };
            tmp_input.qsgw_max_iter = match tmp_ctrl.get("qsgw_max_iter").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(50) as usize,
                _ => 50,
            };
            tmp_input.qsgw_energy_tol = match tmp_ctrl.get("qsgw_energy_tol").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1e-5),
                _ => 1e-5,
            };
            tmp_input.qsgw_mix_param = match tmp_ctrl.get("qsgw_mix_param").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.5),
                _ => 0.5,
            };
            tmp_input.qsgw_eta = match tmp_ctrl.get("qsgw_eta").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.001),
                _ => 0.001,
            };
            tmp_input.cdgw_eta = match tmp_ctrl.get("cdgw_eta").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
                _ => 0.0,
            };
            tmp_input.cdgw_res_tol = match tmp_ctrl.get("cdgw_res_tol").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.001),
                _ => 0.001,
            };
            tmp_input.ac_num_samples = match tmp_ctrl.get("ac_num_samples").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(16) as usize,
                _ => 16,
            };
            tmp_input.ac_omega_max = match tmp_ctrl.get("ac_omega_max").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(5.0),
                _ => 5.0,
            };
            tmp_input.ac_eta = match tmp_ctrl.get("ac_eta").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.001),
                _ => 0.001,
            };
            // Parse response BSE grid sampling parameters
            tmp_input.response_bse_x_start = match tmp_ctrl.get("response_bse_x_start").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
                _ => 0.0,
            };
            tmp_input.response_bse_x_end = match tmp_ctrl.get("response_bse_x_end").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1.0),
                _ => 1.0,
            };
            tmp_input.response_bse_x_points = match tmp_ctrl.get("response_bse_x_points").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(2) as usize,
                _ => 2,
            };
            tmp_input.response_bse_y_start = match tmp_ctrl.get("response_bse_y_start").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
                _ => 0.0,
            };
            tmp_input.response_bse_y_end = match tmp_ctrl.get("response_bse_y_end").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1.0),
                _ => 1.0,
            };
            tmp_input.response_bse_y_points = match tmp_ctrl.get("response_bse_y_points").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(2) as usize,
                _ => 2,
            };
            tmp_input.response_bse_z_start = match tmp_ctrl.get("response_bse_z_start").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
                _ => 0.0,
            };
            tmp_input.response_bse_z_end = match tmp_ctrl.get("response_bse_z_end").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1.0),
                _ => 1.0,
            };
            tmp_input.response_bse_z_points = match tmp_ctrl.get("response_bse_z_points").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(2) as usize,
                _ => 2,
            };
            // Parse FEAST solver control and parameters
            tmp_input.bse_feast_solver = match tmp_ctrl.get("bse_feast_solver").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(b) => *b,
                _ => false,
            };
            tmp_input.bse_eigenrange_min = match tmp_ctrl.get("bse_eigenrange_min").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
                _ => 0.0,
            };
            tmp_input.bse_eigenrange_max = match tmp_ctrl.get("bse_eigenrange_max").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.5),
                _ => 0.5,
            };
            tmp_input.bse_m_expected = match tmp_ctrl.get("bse_m_expected").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(20) as usize,
                _ => 20,
            };
            tmp_input.bse_max_feast_iter = match tmp_ctrl.get("bse_max_feast_iter").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(30) as usize,
                _ => 30,
            };
            tmp_input.bse_tol_feast = match tmp_ctrl.get("bse_tol_feast").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1e-8),
                _ => 1e-8,
            };
            tmp_input.bse_feast_cg_max_iter = match tmp_ctrl.get("bse_feast_cg_max_iter").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(100) as usize,
                _ => 100,
            };
            tmp_input.bse_feast_cg_tol = match tmp_ctrl.get("bse_feast_cg_tol").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1e-8),
                _ => 1e-8,
            };
            tmp_input.bse_feast_gmres_restart = match tmp_ctrl.get("bse_feast_gmres_restart").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(200) as usize,
                _ => 200,
            };
            tmp_input.bse_feast_gmres_max_iter = match tmp_ctrl.get("bse_feast_gmres_max_iter").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(500) as usize,
                _ => 500,
            };
            tmp_input.bse_feast_init_guess_type = match tmp_ctrl.get("bse_feast_init_guess_type").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => {
                    let lower = s.to_lowercase();
                    match lower.as_str() {
                        "gaussian" => String::from("gaussian"),
                        _ => String::from("random"),
                    }
                },
                _ => String::from("random"),
            };
            tmp_input.bse_feast_gaussian_width_factor = match tmp_ctrl.get("bse_feast_gaussian_width_factor").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.5),
                _ => 0.5,
            };
            tmp_input.bse_feast_contour_rayon = match tmp_ctrl.get("bse_feast_contour_rayon").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(b) => *b,
                _ => true,
            };
            tmp_input.bse_feast_n_quad = match tmp_ctrl.get("bse_feast_n_quad").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(8) as usize,
                _ => 8,
            };
            // Generate grids: OUTER LOOP X, MIDDLE LOOP Y, INNER LOOP Z
            let x_step = if tmp_input.response_bse_x_points > 1 {
                (tmp_input.response_bse_x_end - tmp_input.response_bse_x_start) / (tmp_input.response_bse_x_points - 1) as f64
            } else {
                0.0
            };
            let y_step = if tmp_input.response_bse_y_points > 1 {
                (tmp_input.response_bse_y_end - tmp_input.response_bse_y_start) / (tmp_input.response_bse_y_points - 1) as f64
            } else {
                0.0
            };
            let z_step = if tmp_input.response_bse_z_points > 1 {
                (tmp_input.response_bse_z_end - tmp_input.response_bse_z_start) / (tmp_input.response_bse_z_points - 1) as f64
            } else {
                0.0
            };
            let mut grids = Vec::with_capacity(tmp_input.response_bse_x_points * tmp_input.response_bse_y_points * tmp_input.response_bse_z_points);
            for ix in 0..tmp_input.response_bse_x_points {
                let x = tmp_input.response_bse_x_start + ix as f64 * x_step;
                for iy in 0..tmp_input.response_bse_y_points {
                    let y = tmp_input.response_bse_y_start + iy as f64 * y_step;
                    for iz in 0..tmp_input.response_bse_z_points {
                        let z = tmp_input.response_bse_z_start + iz as f64 * z_step;
                        grids.push([x, y, z]);
                    }
                }
            }
            tmp_input.response_bse_grids = grids;
            tmp_input.response_bse_solver = match tmp_ctrl.get("response_bse_solver").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => s.clone().to_lowercase(),
                _ => String::from("klopper"),
            };
            tmp_input.response_bse_tol = match tmp_ctrl.get("response_bse_tol").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1e-6),
                _ => 1e-6,
            };
            tmp_input.response_bse_max_iter = match tmp_ctrl.get("response_bse_max_iter").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(200) as usize,
                _ => 200,
            };
            // NLFEAST (nonlinear BSE) control parameters
            tmp_input.nonlinear_bse = match tmp_ctrl.get("nonlinear_bse").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(b) => *b,
                _ => false,
            };
            tmp_input.nlfeast_centre = match tmp_ctrl.get("nlfeast_centre").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
                _ => 0.0,
            };
            tmp_input.nlfeast_radius = match tmp_ctrl.get("nlfeast_radius").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.5),
                _ => 0.5,
            };
            tmp_input.nlfeast_m0 = match tmp_ctrl.get("nlfeast_m0").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(20) as usize,
                _ => 20,
            };
            tmp_input.nlfeast_n_quad = match tmp_ctrl.get("nlfeast_n_quad").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(12) as usize,
                _ => 12,
            };
            tmp_input.nlfeast_max_iter = match tmp_ctrl.get("nlfeast_max_iter").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(20) as usize,
                _ => 20,
            };
            tmp_input.nlfeast_tol = match tmp_ctrl.get("nlfeast_tol").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1e-8),
                _ => 1e-8,
            };
            tmp_input.nlfeast_gmres_restart = match tmp_ctrl.get("nlfeast_gmres_restart").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(200) as usize,
                _ => 200,
            };
            tmp_input.nlfeast_gmres_max_it = match tmp_ctrl.get("nlfeast_gmres_max_it").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_u64().unwrap_or(500) as usize,
                _ => 500,
            };
            tmp_input.export_matvec_count = match tmp_ctrl.get("export_matvec_count").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Bool(b) => *b,
                _ => false,
            };
            tmp_input.nlfeast_gmres_tol = match tmp_ctrl.get("nlfeast_gmres_tol").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1e-6),
                _ => 1e-6,
            };
            tmp_input.dynamic_bse_kernel = match tmp_ctrl.get("dynamic_bse_kernel").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::String(s) => {
                    let lower = s.trim().to_lowercase();
                    match lower.as_str() {
                        "srpa" | "bare" | "bare_v" => String::from("srpa"),
                        _ => String::from("bse"),
                    }
                }
                _ => String::from("bse"),
            };
            tmp_input.bse_matvec_style = normalise_style(
                tmp_ctrl.get("bse_matvec_style"),
                &["ao", "memory-efficient", "memory_efficient", "mem-efficient"],
                "ao",
            );
            tmp_input.gw_tensor_style = normalise_style(
                tmp_ctrl.get("gw_tensor_style"),
                &["ao", "memory-efficient", "memory_efficient", "mem-efficient"],
                "mo",
            );
            tmp_input.gw_ao_screening_tol = match tmp_ctrl.get("gw_ao_screening_tol").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
                _ => 0.0,
            };
            tmp_input.gw_switch_fallback_threshold = match tmp_ctrl.get("gw_switch_fallback_threshold").unwrap_or(&serde_json::Value::Null) {
                serde_json::Value::Number(n) => n.as_f64().unwrap_or(1e6),
                _ => 1e6,
            };
            return Ok(Some(tmp_input));
        },
        other => {
            return Ok(None);
        },
    }
    
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn gw_variant_defaults_to_cd() {
        let input = json!({
            "quasiparticle_methods": {}
        });

        let qp = parse_quasiparticle_keywords(&input).unwrap().unwrap();
        assert_eq!(qp.gw_variant, GwVariant::Cd);
    }

    #[test]
    fn gw_variant_accepts_ac_case_insensitively() {
        for value in ["ac", "AC", "Ac"] {
            let input = json!({
                "quasiparticle_methods": {
                    "gw_variant": value
                }
            });

            let qp = parse_quasiparticle_keywords(&input).unwrap().unwrap();
            assert_eq!(qp.gw_variant, GwVariant::Ac);
        }
    }

    #[test]
    fn gw_variant_rejects_unknown_value() {
        let input = json!({
            "quasiparticle_methods": {
                "gw_variant": "unknown"
            }
        });

        assert!(parse_quasiparticle_keywords(&input).is_err());
    }

    #[test]
    fn gw_checkpoint_keys_default_to_off() {
        let input = json!({
            "quasiparticle_methods": {}
        });

        let qp = parse_quasiparticle_keywords(&input).unwrap().unwrap();
        assert!(!qp.save_gw_checkpoint);
        assert!(!qp.resume_from_checkpoint);
        assert_eq!(qp.gw_checkpoint_path, "./gw_checkpoint.h5");
    }

    #[test]
    fn gw_checkpoint_keys_are_parsed() {
        let input = json!({
            "quasiparticle_methods": {
                "save_gw_checkpoint": true,
                "resume_from_checkpoint": true,
                "gw_checkpoint_path": "/scratch/run42/gw.h5"
            }
        });

        let qp = parse_quasiparticle_keywords(&input).unwrap().unwrap();
        assert!(qp.save_gw_checkpoint);
        assert!(qp.resume_from_checkpoint);
        assert_eq!(qp.gw_checkpoint_path, "/scratch/run42/gw.h5");

        // ... and round-trips through the TOML serialisation.
        let table = qp.to_toml();
        assert_eq!(table["save_gw_checkpoint"].as_bool(), Some(true));
        assert_eq!(table["resume_from_checkpoint"].as_bool(), Some(true));
        assert_eq!(
            table["gw_checkpoint_path"].as_str(),
            Some("/scratch/run42/gw.h5")
        );
    }
}
