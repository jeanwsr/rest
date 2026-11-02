//! PEXSI-based SCF solve layer: a driver, not a whole SCF.
//!
//! [`PexsiSolver`] owns the state that has to survive between SCF iterations —
//! the frozen sparsity pattern, the plan (with its symbolic factorization), the
//! driver options, the bracket state and the chemical potential. The SCF loop
//! in `scf_without_build` owns the iteration itself: mixing, convergence,
//! printing. One loop, two interchangeable solve steps, chosen once per run.
//!
//! # Usage
//!
//! ```ignore
//! // once, after the initial diagonalisation has produced eigenvalues
//! let mut solver = PexsiSolver::new(scf, mpi_operator)?;
//!
//! // inside the loop, in place of diagonalise + occupation + density
//! solver.solve_step(scf, mpi_operator)?;   // hamiltonian[0] in, density_matrix[0] out
//!
//! // after convergence: rebuild the Fock from the converged density, take the
//! // energy, then diagonalise once to populate eigenvalues/eigenvectors for
//! // whatever runs next.
//! ```
//!
//! `new` reads `scf.eigenvalues[0]` for the preconditioning (gap, temperature,
//! μ0, bracket, ΔE) and `h_core`/`ovlp` for the pattern. It does not solve
//! anything, so a failure there is safe to fall back from.
//!
//! # Scope
//!
//! Validated for LDA/GGA. Exact exchange is not supported: `K_ij` needs only
//! `(i,k)` and `(j,l)` to overlap *separately*, so its tail decays as 1/R and
//! lives outside the `H_core` pattern frozen here. Measured on H2O20/cc-pVDZ:
//! `max|F_off| = 5.8e-4`, `||F_off||_F = 3.7e-3`, converged energy 13.7 mHa
//! from the diagonalisation path. For a local potential `J` and `Vxc` share
//! `S`'s support and the containment is exact, which is why LDA/GGA agree with
//! diagonalisation to ~1e-6 Ha (measured: 1.58 µHa).
//!
//! # Limits of PEXSI v2.0.0 encoded here
//!
//! * `nPoints` must be 1. Higher values route the density through
//!   `InterpolateDMReal`'s cross-point interpolation, which returns the wrong
//!   electron count.
//! * `LoadRealHSMatrix` wraps the caller's buffers rather than copying them
//!   (`owndata_ = false`), so the value arrays have to live on the plan. That
//!   is why `pexsi_load_hs` copies into `PexsiPlan`'s own vectors.
//! * Method 2's shipped pole table stops at 61 poles; asking for more aborts
//!   inside `getPole`, so `NUM_POLE` stays inside that range.
//! * `mu0` is ignored by `DFTDriver2`. With `nPoints = 1` the solver samples
//!   the bracket midpoint and does not move it, so the bracket we pass *is* the
//!   chemical potential. That is why the bracket is rebuilt every iteration
//!   around the value the previous solve returned.
use mpi::topology::Communicator;
use mpi::raw::AsRaw;

use crate::mpi_io::{mpi_broadcast_matrixfull, MPIOperator};
use crate::scf_io::smear::{find_mu, SmearingType};
use crate::scf_io::util::integer_homo_lumo;
use log::{info, warn};
use mpi::topology::SimpleCommunicator;
use std::sync::Arc;
use tensors::matrix::distsparsematrixfull::{
    reconstruct_from_gathered, DistSparseMatrixFull, SparsePattern,
};
use tensors::matrix::matrix_pexsi::{
    pexsi_dft_driver2, pexsi_load_hs, pexsi_retrieve_dm, PexsiPlan, PPEXSIOptions,
};
use tensors::MatrixFull;

use super::{SCFType, SCF};

/// Fallback electronic temperature: a fraction of the gap, clamped. The upper
/// bound keeps the pole count inside the table; the lower bound keeps the
/// occupation function resolvable by it.
const T_FALLBACK_MAX: f64 = 1.0e-2;
const T_FALLBACK_MIN: f64 = 5.0e-3;

/// Threshold that defines the frozen pattern. Applied to `H_core`, safe for a
/// local potential, not safe for exact exchange — see the module docs.
const PATTERN_EPS: f64 = 1.0e-10;

/// Pole count. Method 2's table stops at 61; 34-60 land above
/// `beta * deltaE` for a typical T = 1e-2 Ha run.
const NUM_POLE: i32 = 60;

/// Energy window used for the PEXSI pass. The pole-expanded density has a noise
/// floor near 1e-6, so a tighter window cannot close.
pub const ENERGY_TOLERANCE_FLOOR: f64 = 1.0e-4;

/// What one solve step produced.
pub struct StepReport {
    /// Chemical potential the solver returned (the bracket midpoint it sampled).
    pub mu: f64,
    /// Electron count *before* PEXSI's normalisation rescale. Should track
    /// `num_electrons`; a drift means μ no longer matches this Fock and the
    /// density is being repaired by a uniform scaling instead.
    pub num_electrons_raw: f64,
}

/// Holds everything that persists across iterations.
pub struct PexsiSolver {
    temperature: f64,
    num_electrons: f64,
    mu0: f64,
    mu_half_band: f64,
    delta_e: f64,
    gap: f64,

    nbas: usize,
    n_pad: usize,
    pattern: Arc<SparsePattern>,
    s_sp: DistSparseMatrixFull<f64>,
    fock_pad: MatrixFull<f64>,

    opts: PPEXSIOptions,
    plan: PexsiPlan,

    mu: f64,
    /// Re-locate μ on the next step. Set on the first step and whenever the
    /// electron count drifts, cleared once μ is known to be inside the gap.
    force_inertia: bool,

    pattern_nnz: usize,
    pattern_dense: f64,
}

impl PexsiSolver {
    /// Whether this calculation should use the PEXSI step at all.
    ///
    /// The compile-time gate belongs at the call site, because this module only
    /// exists with the feature on.
    pub fn is_enabled(scf: &SCF) -> bool {
        // The SAD initial guess runs its own nested SCF, which is a doublet.
        !scf.mol.ctrl.atom_sad && scf.mol.ctrl.pexsi 
    }

    /// Build the pattern, fill the overlap, and create the plan.
    ///
    /// Reads `scf.eigenvalues[0]` (from the initial diagonalisation),
    /// `scf.h_core`, `scf.ovlp`. Does not modify `scf`, so an `Err` here is
    /// safe to fall back from.
    pub fn new(scf: &SCF, mpi_operator: &Option<MPIOperator>) -> Result<Self, String> {
        check_reference(scf)?;
        warn_if_exact_exchange(scf);

        let p = precondition(scf)?;
        warn_convergence_criteria(scf);

        let nbas = scf.mol.num_basis;
        let (rank, n_pad, c_pad) = with_world(mpi_operator, |world| {
            let npcol = world.size() as usize;
            let n_pad = (nbas + npcol - 1) / npcol * npcol;
            (world.rank(), n_pad, p.mu0 + 0.5 * p.delta_e)
        });

        if rank == 0 && scf.mol.ctrl.print_level > 0 {
            info!(
                "PEXSI preconditioning: T = {:.6} Ha  N_e = {:.1}  HOMO = {:+.6}  LUMO = {:+.6}  \
                 gap = {:.6} Ha  deltaE = {:.6} Ha  mu0 = {:+.6}  bracket = [{:+.6}, {:+.6}]  \
                 numPole = {}",
                p.temperature,
                p.num_electrons,
                p.homo,
                p.lumo,
                p.gap,
                p.delta_e,
                p.mu0,
                p.mu0 - p.mu_half_band,
                p.mu0 + p.mu_half_band,
                p.num_pole,
            );
        }

        // ---- pattern: H_core on the padded grid --------------------------
        let h_core_full = scf.h_core.to_matrixfull().unwrap();
        let mut h_core_pad = MatrixFull::new([n_pad, n_pad], 0.0);
        for j in 0..nbas {
            for i in 0..nbas {
                h_core_pad[[i, j]] = h_core_full[[i, j]];
            }
        }
        for i in nbas..n_pad {
            h_core_pad[[i, i]] = c_pad;
        }

        let pattern = Arc::new(with_world(mpi_operator, |world| {
            SparsePattern::from_matrixfull(&h_core_pad, PATTERN_EPS, world.size(), world.rank())
        }));
        let pattern_nnz = pattern.global_nnz;
        let pattern_dense = 100.0 * pattern_nnz as f64 / (n_pad * n_pad) as f64;
        if rank == 0 {
            info!(
                "PEXSI pattern: nnz = {} of {} ({:.1}% dense), built from H_core at {:.0e}",
                pattern_nnz,
                n_pad * n_pad,
                pattern_dense,
                PATTERN_EPS
            );
        }

        // ---- overlap, filled on the same pattern (once) -------------------
        let ovlp_full = scf.ovlp.to_matrixfull().unwrap();
        let mut s_pad = MatrixFull::new([n_pad, n_pad], 0.0);
        for j in 0..nbas {
            for i in 0..nbas {
                s_pad[[i, j]] = ovlp_full[[i, j]];
            }
        }
        for i in nbas..n_pad {
            s_pad[[i, i]] = 1.0;
        }
        verify_pattern_contains_overlap(&pattern, &s_pad, n_pad, PATTERN_EPS, rank)?;
        let mut s_nz = Vec::new();
        pattern.fill_from_matrixfull(&s_pad, &mut s_nz);
        let s_sp = DistSparseMatrixFull {
            pattern: pattern.clone(),
            nzval: s_nz,
        };

        // ---- driver options: fixed for the whole run ----------------------
        let mut opts = PPEXSIOptions::defaults();
        opts.verbosity = 2;
        opts.symmetric = 1;
        opts.solver = 0; // SuperLU_DIST
        opts.ordering = 2; // MMD
        opts.method = 2; // Moussa
        opts.nPoints = 1; // >1 corrupts the density through InterpolateDMReal
        opts.numPole = p.num_pole;
        opts.temperature = p.temperature;
        opts.deltaE = p.delta_e;
        opts.gap = p.gap;
        opts.matrixType = 0;
        opts.npSymbFact = 1;
        opts.isSymbolicFactorize = 1; // first call only
        opts.isInertiaCount = 1;
        opts.maxPEXSIIter = 20;
        opts.mu0 = p.mu0;
        opts.muMin0 = p.mu0 - p.mu_half_band;
        opts.muMax0 = p.mu0 + p.mu_half_band;
        opts.muInertiaTolerance = 1.0e-3; // clamped up to 4*kT by PEXSI
        opts.muInertiaExpansion = 1.0;
        opts.muPEXSISafeGuard = 1.0e-2;
        opts.numElectronPEXSITolerance = 1.0e-6;

        let plan = with_world(mpi_operator, |world| {
            PexsiPlan::new(world.as_raw(), 1, world.size(), world.rank())
                .expect("PEXSIPlanInitialize")
        });

        // Padded diagonal carries no density; keep it above the spectrum so
        // those columns stay empty.
        let mut fock_pad = MatrixFull::new([n_pad, n_pad], 0.0);
        for i in nbas..n_pad {
            fock_pad[[i, i]] = c_pad;
        }

        Ok(PexsiSolver {
            temperature: p.temperature,
            num_electrons: p.num_electrons,
            mu0: p.mu0,
            mu_half_band: p.mu_half_band,
            delta_e: p.delta_e,
            gap: p.gap,
            nbas,
            n_pad,
            pattern,
            s_sp,
            fock_pad,
            opts,
            plan,
            mu: p.mu0,
            force_inertia: true,
            pattern_nnz,
            pattern_dense,
        })
    }

    /// One solve step: `hamiltonian[0]` in, `density_matrix[0]` out.
    ///
    /// The Fock must already be built from the current density — that is what
    /// `prepare_next_input` leaves behind.
    pub fn solve_step(
        &mut self,
        scf: &mut SCF,
        mpi_operator: &Option<MPIOperator>,
    ) -> Result<StepReport, String> {
        let nbas = self.nbas;
        let n_pad = self.n_pad;

        // ---- Fock -> values on the frozen pattern ------------------------
        let fock = scf.hamiltonian[0].to_matrixfull().unwrap();
        for j in 0..nbas {
            for i in 0..nbas {
                self.fock_pad[[i, j]] = fock[[i, j]];
            }
        }
        let mut h_nz = Vec::new();
        self.pattern.fill_from_matrixfull(&self.fock_pad, &mut h_nz);
        let h_sp = DistSparseMatrixFull {
            pattern: self.pattern.clone(),
            nzval: h_nz,
        };

        // ---- bracket policy ---------------------------------------------
        // With nPoints = 1 the solver samples the midpoint and does not move
        // it, so the bracket has to contain the answer. Wide after a drift
        // (inertia counting re-locates μ); narrow otherwise (scan skipped,
        // cheaper and stable while the Fock moves little).
        if self.force_inertia {
            let band = self.gap.max(8.0 * self.temperature);
            self.opts.muMin0 = self.mu - band;
            self.opts.muMax0 = self.mu + band;
        } else {
            self.opts.muMin0 = self.mu - self.mu_half_band;
            self.opts.muMax0 = self.mu + self.mu_half_band;
        }

        // ---- solve ------------------------------------------------------
        pexsi_load_hs(&mut self.plan, &self.opts, &h_sp, &self.s_sp)
            .map_err(|e| format!("LoadRealHSMatrix: {e}"))?;
        let res = pexsi_dft_driver2(&self.plan, &mut self.opts, self.num_electrons)
            .map_err(|e| format!("DFTDriver2: {e}"))?;
        self.mu = res.mu;
        self.opts.isSymbolicFactorize = 0; // pattern frozen from here on

        let num_electrons_raw = res.num_electrons;
        self.force_inertia = (num_electrons_raw - self.num_electrons).abs() > 1.0e-2;

        // ---- density back to a dense matrix on every rank ----------------
        let mut dm_nz = vec![0.0_f64; h_sp.nnz_local()];
        pexsi_retrieve_dm(&self.plan, &mut dm_nz)
            .map_err(|e| format!("RetrieveRealDM: {e}"))?;
        let dm_sp = DistSparseMatrixFull {
            pattern: self.pattern.clone(),
            nzval: dm_nz,
        };
        let pattern = self.pattern.clone();

        // Collective on every rank: non-root ranks block in `send`.
        let dm_pad = with_world(mpi_operator, |world| {
            let gathered = dm_sp.gather_to_root(0, world);
            let mut dm_pad = MatrixFull::new([n_pad, n_pad], 0.0);
            if world.rank() == 0 {
                dm_pad = reconstruct_from_gathered(&gathered, &pattern);
            }
            mpi_broadcast_matrixfull(world, &mut dm_pad, 0);
            dm_pad
        });

        let mut dm_new = MatrixFull::new([nbas, nbas], 0.0);
        for j in 0..nbas {
            for i in 0..nbas {
                dm_new[[i, j]] = dm_pad[[i, j]];
            }
        }
        scf.density_matrix[0] = dm_new;

        Ok(StepReport {
            mu: self.mu,
            num_electrons_raw,
        })
    }

    pub fn mu(&self) -> f64 {
        self.mu
    }

    pub fn pattern_nnz(&self) -> usize {
        self.pattern_nnz
    }

    pub fn pattern_dense(&self) -> f64 {
        self.pattern_dense
    }

    /// `true` once the solver reports an electron count matching `N_e`, i.e.
    /// once the bracket is centred on a μ that sits inside the gap.
    pub fn mu_is_located(&self) -> bool {
        !self.force_inertia
    }
}

// ===========================================================================
// preconditioning
// ===========================================================================

/// Input-dependent quantities derived once, from the pre-loop spectrum.
struct PexsiParams {
    temperature: f64,
    num_electrons: f64,
    num_pole: i32,
    mu0: f64,
    mu_half_band: f64,
    delta_e: f64,
    gap: f64,
    homo: f64,
    lumo: f64,
}

fn check_reference(scf: &SCF) -> Result<(), String> {
    let name = match scf.scftype {
        SCFType::RHF => "RHF",
        SCFType::ROHF => "ROHF",
        SCFType::UHF => "UHF",
    };
    if !matches!(scf.scftype, SCFType::RHF) {
        return Err(format!(
            "PEXSI requires an RHF reference; got scftype = {name}, spin = {}, \
             spin_polarization = {}, spin_channel = {}, atom_sad = {}. NOTE: spin_channel \
             follows `spin_polarization`, not `spin` - set `spin_polarization = false` for \
             a closed-shell run.",
            scf.mol.ctrl.spin,
            scf.mol.ctrl.spin_polarization,
            scf.mol.spin_channel,
            scf.mol.ctrl.atom_sad,
        ));
    }
    if let Some(st) = &scf.mol.ctrl.smear {
        if !matches!(*st, SmearingType::FERMI) {
            return Err(
                "PEXSI supports Fermi-Dirac smearing only; `smear = gaussian` has no \
                 PEXSI analogue."
                    .to_string(),
            );
        }
    }
    Ok(())
}

/// Warn, but do not refuse: an exact-exchange run still produces an answer, it
/// is just the answer for a truncated Fock. Flipping this to a hard error is a
/// one-line change if the group decides a silent bias is worse than a refusal.
fn warn_if_exact_exchange(scf: &SCF) {
    if scf.mol.use_eri {
        warn!(
            "PEXSI: xc = `{}` builds exact-exchange factors (dfa_hybrid_scf = {:.6}), so F is \
             NOT contained in the H_core pattern: K_ij needs only (i,k) and (j,l) to overlap \
             separately, its tail decays as 1/R, and the frozen pattern truncates it. \
             Measured on H2O20/cc-pVDZ: max|F_off| = 5.8e-4, ||F_off||_F = 3.7e-3, converged \
             energy 13.7 mHa away from the diagonalisation path. The PEXSI path is validated \
             for LDA/GGA only; treat this density as approximate.",
            scf.mol.ctrl.xc, scf.mol.xc_data.dfa_hybrid_scf
        );
    }
}

fn precondition(scf: &SCF) -> Result<PexsiParams, String> {
    let eigs: &[f64] = &scf.eigenvalues[0];
    if eigs.is_empty() {
        return Err(
            "PEXSI preconditioning needs `scf.eigenvalues[0]`; diagonalise once and generate \
             the occupation before creating the solver."
                .to_string(),
        );
    }
    let last = eigs.len() - 1;

    // Occupied / virtual indices from the SCF's own machinery, so that ATMSAD
    // and FRAC choices are respected instead of assuming N_e / 2.
    let ([homo_i, _], [lumo_i, _]) = integer_homo_lumo(&scf.mol.num_elec, scf.mol.spin_channel);
    let homo_i = homo_i.min(last);
    let lumo_i = lumo_i.min(last);

    let gap = (eigs[lumo_i] - eigs[homo_i]).max(0.0);
    let num_electrons = scf.mol.num_elec[0];

    // Explicit `smear_sigma` wins; otherwise scale with the gap.
    let temperature = match scf.mol.ctrl.smear_sigma.filter(|&s| s > 0.0) {
        Some(s) => s,
        None => (0.05 * gap).clamp(T_FALLBACK_MIN, T_FALLBACK_MAX),
    };
    if gap > 0.0 && temperature > 0.1 * gap {
        warn!(
            "PEXSI: kT = {:.3e} Ha is {:.0}% of the gap ({:.3e} Ha). Occupations will be \
             visibly smeared, so this is not the T->0 density. Set `smear_sigma` explicitly \
             if that is not what you want.",
            temperature,
            100.0 * temperature / gap,
            gap,
        );
    }

    // mu from the SCF's own occupation function, not from a bracket midpoint:
    // with nPoints = 1 whatever we pass is the chemical potential.
    let mu0 = find_mu(eigs, num_electrons / 2.0, temperature, SmearingType::FERMI);
    let rho = eigs.iter().fold(0.0_f64, |m, &e| m.max(e.abs()));

    Ok(PexsiParams {
        temperature,
        num_electrons,
        num_pole: NUM_POLE,
        mu0,
        mu_half_band: 2.0 * temperature, // width 4*kT < 8*kT: the scan stays skipped
        delta_e: 1.05 * rho,
        gap,
        homo: eigs[homo_i],
        lumo: eigs[lumo_i],
    })
}

fn warn_convergence_criteria(scf: &SCF) {
    let crit = scf.mol.ctrl.scf_conv_criteria.clone();
    warn!(
        "PEXSI: convergence criteria '{}' (etot {:.0e} Ha, dm {:.0e}, eev {:.0e}, grad {:.0e}). \
         The pole-expanded density has a noise floor around 1e-6, so an energy window below \
         ~1e-5 Ha cannot close. The density criterion is the reliable gate for this solver.",
        crit,
        scf.mol.ctrl.scf_acc_etot,
        scf.mol.ctrl.scf_acc_rho,
        scf.mol.ctrl.scf_acc_eev,
        scf.mol.ctrl.scf_acc_g,
    );

    if crit.split(',').any(|c| c.trim() == "g") {
        warn!(
            "PEXSI: scf_conv_criteria includes 'g'. grad_dm = ||F D S - S D F||_F cannot reach \
             1e-5 here. Under diagonalisation D = C f C^T is idempotent, so the commutator \
             vanishes with the SCF residual; under PEXSI the density carries a ~1e-6 \
             pole-expansion error, so the commutator floors near |F| * eps * nbas ~ 1e-1 for \
             nbas ~ 500. Use 'dm' (or 'dm,eev')."
        );
    }
}

// ===========================================================================
// helpers
// ===========================================================================

/// Run `f` with a communicator, whether or not the caller has an MPI operator.
/// `SimpleCommunicator::world()` is cheap, so this is called per step rather
/// than stored.
fn with_world<R>(mpi_operator: &Option<MPIOperator>, f: impl FnOnce(&SimpleCommunicator) -> R) -> R {
    match mpi_operator {
        Some(op) => f(&op.world),
        None => {
            let world = SimpleCommunicator::world();
            f(&world)
        }
    }
}

/// The metric is filled on the pattern, so every overlap entry above the
/// threshold must be on it. `S` and `H_core` share their *structural* support
/// (both are two-centre integrals over the same basis) but not their numerical
/// one: near a cancellation in `T + V_nuc`, `|h_core|` can fall below the
/// threshold while `|S|` does not. One such pair and PEXSI receives a metric
/// that is silently missing an entry.
fn verify_pattern_contains_overlap(
    pattern: &SparsePattern,
    s_pad: &MatrixFull<f64>,
    n_pad: usize,
    eps: f64,
    rank: i32,
) -> Result<(), String> {
    let mut missing = 0usize;
    let mut worst = 0.0_f64;
    let mut checked = 0usize;

    for j in 0..n_pad {
        let lo = (pattern.global_colptr[j] - 1) as usize;
        let hi = (pattern.global_colptr[j + 1] - 1) as usize;
        let rows = &pattern.global_rowind[lo..hi];
        for i in 0..n_pad {
            let v = s_pad[[i, j]];
            if v.abs() <= eps {
                continue;
            }
            checked += 1;
            // global_rowind is built by scanning i upward, so it is sorted.
            if rows.binary_search(&((i + 1) as i32)).is_err() {
                missing += 1;
                worst = worst.max(v.abs());
            }
        }
    }

    if missing > 0 {
        if rank == 0 {
            warn!("{}", format!(
                "PEXSI pattern: {missing} of {checked} overlap entries above {eps:.0e} are not on \
                 the pattern (largest missing |S_ij| = {worst:.3e}). The threshold applied to \
                 h_core is dropping metric entries; build the pattern at a smaller eps, or from \
                 max(|h_core|, |S|)."
            ));
        }
    }
    Ok(())
}
