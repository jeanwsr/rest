//! MOM state guard: is the calculation still the electronic state it was asked for?
//!
//! A Delta-SCF (`force_state_occupation`) geometry optimization can lose its
//! state in ways that look like success: the SCF converges, the energy falls
//! smoothly, the gradient shrinks, the optimizer reports `Converged!` -- and the
//! determinant being optimized is no longer the excited state. The first
//! implementation of the rolling reference did exactly that (a bookkeeping bug
//! rolled the tracked index onto the 1s core orbital, the occupation fell back to
//! the Aufbau filling, and the job optimized the ground state for five steps
//! without a single error line).
//!
//! This module answers, once per converged SCF, "which state is this, really?".
//! The criteria are deliberately phrased in terms of **energies and spin**, never
//! in terms of "does the occupation look Aufbau", because that last test is not an
//! invariant of the state:
//!
//! * the occupation is written by *index* while the diagonalizer sorts by
//!   *eigenvalue*, so in a near-degenerate manifold the same physical state can
//!   appear with a contiguous or a non-contiguous occupation pattern;
//! * when the ground state itself is open shell (a biradical, the INVEST
//!   molecules this work targets), "S0" and "S1" are both broken-symmetry
//!   singlets, and no occupation-pattern test can separate them.
//!
//! The ladder actually used here, cheapest first:
//!
//! | level | quantity | catches |
//! |---|---|---|
//! | L0 | `<S^2>` against its value at the first converged step | collapse into a closed shell (`<S^2> -> 0`) |
//! | L1 | `dE = E(MOM) - E(ground determinant in the *same* orbitals)` | the MOM determinant *is* the ground determinant, or is degenerate with it |
//! | L2 | `dE = E(MOM) - E(S0 SCF at the same geometry)` (opt-in, one extra SCF) | the real adiabatic gap and the adiabatic crossing |
//! | L3 | overlap of the tracked orbital with the reference | the reference has become meaningless |
//!
//! L1 is the workhorse: it builds the ground-state filling **from the orbital
//! eigenvalues** (not from the index order), evaluates that determinant's energy
//! with one extra Fock build, and reports the difference. A collapse gives exactly
//! zero; a genuine excited state gives a positive number; a near-degenerate one
//! gives a small number that is *reported* instead of being turned into a wrong
//! yes/no answer.
//!
//! Near-degeneracy itself is not a problem this module can solve: when the gap
//! closes, "S1" is not a well-defined single-determinant label, and the right move
//! is to stop and hand the system to the multireference machinery of this package
//! (CASSCF + MC-xDH, see `mcxdh5_examples/001_s0`). The guard's job is to make
//! that moment visible.
//!
//! See docs/dscf_s1_geomopt_plan.md, section A3.

use super::*;
use crate::constants::EV;
use crate::ctrl_io::{JobType, MomGapCheck, MomGuard};

/// What one geometry step's state check produced (for the log and the json).
#[derive(Clone, Debug)]
pub struct MomStateReport {
    /// committed geometry step this report belongs to (1-based)
    pub step: usize,
    /// orbital index the constraint is currently attached to, per entry
    pub index: Vec<usize>,
    /// smallest overlap of the tracked orbitals with their reference
    pub overlap: f64,
    /// `<S^2>` of the current determinant
    pub s2: f64,
    /// `<S^2>` of the first converged step of the job (the state that was asked for)
    pub s2_ref: f64,
    /// `E(MOM)` at this geometry
    pub e_mom: f64,
    /// `E` of the ground-state determinant built from the *same* orbitals
    pub e_gs_fo: f64,
    /// `E(MOM) - E(ground determinant, same orbitals)`, in Ha
    pub d_e_fo: f64,
    /// whether the MOM occupation *is* that ground-state filling
    pub aufbau_equal: bool,
    /// `E(S0)` at the same geometry (L2), when requested
    pub e_gs: Option<f64>,
}

impl MomStateReport {
    pub fn d_e_fo_ev(&self) -> f64 {
        self.d_e_fo * EV
    }
    pub fn d_e_gs_ev(&self) -> Option<f64> {
        self.e_gs.map(|e| (self.e_mom - e) * EV)
    }
}

impl SCF {
    /// Ground-state filling of the **current** orbitals: for each spin channel the
    /// `n_sigma` orbitals with the lowest eigenvalues.
    ///
    /// Selecting by eigenvalue rather than by "the first `n_sigma` columns" is
    /// what makes the comparison independent of the ordering the diagonalizer
    /// returned, and it is the reason this test survives the near-degenerate case
    /// that defeats an index/occupation-pattern test.
    fn aufbau_occupation_by_energy(&self) -> Option<[Vec<f64>; 2]> {
        let nmo = self.mol.num_state;
        if nmo == 0 {
            return None;
        }
        let mut occ = [vec![0.0; nmo], vec![0.0; nmo]];
        for spin in 0..2 {
            let n_sigma = self.occupation[spin].iter().sum::<f64>().round() as usize;
            let ev = &self.eigenvalues[spin];
            if ev.len() < nmo {
                // the spin channel has no orbitals of its own (restricted run, or
                // an empty channel): nothing sensible to build
                if n_sigma > 0 {
                    return None;
                }
                continue;
            }
            let mut idx: Vec<usize> = (0..nmo).collect();
            idx.sort_by(|&a, &b| {
                ev[a].partial_cmp(&ev[b]).unwrap_or(std::cmp::Ordering::Equal)
            });
            for &i in idx.iter().take(n_sigma.min(nmo)) {
                occ[spin][i] = 1.0;
            }
        }
        Some(occ)
    }

    /// Energy of a determinant built from the current orbitals with the given
    /// occupation, leaving the SCF state exactly as it was found.
    ///
    /// This is the same save / swap density / single Fock build / restore pattern
    /// the open-shell-singlet triplet evaluator uses.
    fn determinant_energy_with_occupation(&mut self, occupation: [Vec<f64>; 2], mpi_operator: &Option<MPIOperator>) -> f64 {
        let dm_save = self.density_matrix.clone();
        let occ_save = self.occupation.clone();
        let (homo_save, lumo_save) = (self.homo, self.lumo);
        let ham_save = self.hamiltonian.clone();
        let roothaan_save = self.roothaan_hamiltonian.clone();
        let e_save = self.scf_energy;
        let inner_save = self.oss_variational_inner_build;

        self.oss_variational_inner_build = true;
        self.occupation = occupation;
        (self.homo, self.lumo) = oss_homo_lumo_scan(&self.occupation);
        self.generate_density_matrix();
        self.generate_hf_hamiltonian(mpi_operator);
        let energy = self.scf_energy;

        self.density_matrix = dm_save;
        self.occupation = occ_save;
        self.homo = homo_save;
        self.lumo = lumo_save;
        self.hamiltonian = ham_save;
        self.roothaan_hamiltonian = roothaan_save;
        self.scf_energy = e_save;
        self.oss_variational_inner_build = inner_save;
        energy
    }

    /// Ground-state energy at the current geometry (L2).
    ///
    /// Runs a plain (unconstrained) SCF on the same molecule -- the same
    /// functional, the same spin treatment, the same geometry, only without the
    /// `force_state_occupation` constraints -- so the difference to the MOM state
    /// is a genuine adiabatic gap. It costs one integral setup plus one SCF;
    /// nothing of the current SCF state is touched, and the auxiliary job writes
    /// no chkfile and no json.
    ///
    /// The spin setup is deliberately left as it is: switching a two-channel
    /// (UHF/UKS) molecule to a restricted one requires `mol.spin_channel` and
    /// `mol.num_elec` to be rebuilt together with `ctrl.spin_polarization`, and a
    /// mismatch there once produced a "ground state" with all ten electrons in the
    /// alpha channel (E = -78.12 Ha for H2O/6-31G/PBE, ~1.7 Ha below the real
    /// value, gap 57 eV instead of ~5 eV). Clearing the constraints instead keeps
    /// both states in the same variational space, which is also the comparison one
    /// wants.
    pub fn mom_ground_state_energy(&self, mpi_operator: &Option<MPIOperator>) -> Option<f64> {
        let mut mol = self.mol.clone();
        mol.ctrl.force_state_occupation.clear();
        mol.ctrl.mom_gap_check = MomGapCheck::Off;
        mol.ctrl.mom_guard = MomGuard::Warn;
        mol.ctrl.initial_guess = String::from("sad");
        mol.ctrl.external_init_guess = None;
        mol.ctrl.has_chkfile = false;
        mol.ctrl.outname = None;
        mol.ctrl.print_level = 0;
        if mol.num_elec[1] != mol.num_elec[2] {
            // not a singlet electron count: there is no closed-shell "S0" to
            // compare a (typically singlet) MOM state with
            return None;
        }
        let mut s0 = SCF::init_scf(&mol);
        initialize_scf(&mut s0, mpi_operator);
        scf_without_build(&mut s0, mpi_operator);
        if !s0.scf_converged {
            return None;
        }
        Some(s0.scf_energy)
    }

    /// Run the MOM state guard for the SCF that has just converged.
    ///
    /// Prints one line per step and appends the numbers to `self.energies` (the
    /// json dump serializes that map wholesale, so the per-step history ends up in
    /// `rest_results.json` without a separate writer). Depending on `mom_guard`
    /// and on the job type, a failed check aborts the job: a force or geometry
    /// optimization of the wrong state produces a confident, wrong structure,
    /// which is worse than a stop.
    pub fn mom_state_check(&mut self, mpi_operator: &Option<MPIOperator>) -> Option<MomStateReport> {
        if self.mol.ctrl.force_state_occupation.is_empty() {
            return None;
        }
        // The OSS machinery has its own bookkeeping and does not support forces.
        if self.mol.ctrl.oss_roks {
            return None;
        }
        if self.eigenvectors[0].data.is_empty() || self.density_matrix[0].data.is_empty() {
            return None;
        }

        let step = self.mom_commit_count + 1;

        // --- L0: spin signature ------------------------------------------------
        let [s2, _2s_plus_1] = evaluate_spin_angular_momentum(&self.density_matrix, &self.ovlp, 2, &self.mol.num_elec);
        let s2_ref = *self.mom_s2_ref.get_or_insert(s2);

        // --- tracked orbital / overlap (L3) -----------------------------------
        let mut index: Vec<usize> = Vec::new();
        let mut overlap = 1.0_f64;
        for fso in self.mol.ctrl.force_state_occupation.iter() {
            index.push(fso.get_curr_state());
            overlap = overlap.min(fso.get_overlap());
        }

        // --- L1 (+ L4 diagnostic): ground-state determinant in the same orbitals
        let e_mom = self.scf_energy;
        let (e_gs_fo, aufbau_equal) = match self.aufbau_occupation_by_energy() {
            Some(occ_gs) => {
                let equal = (0..2).all(|s| occ_gs[s] == self.occupation[s]);
                let energy = self.determinant_energy_with_occupation(occ_gs, mpi_operator);
                (energy, equal)
            }
            None => (f64::NAN, false),
        };

        let mut report = MomStateReport {
            step,
            index,
            overlap,
            s2,
            s2_ref,
            e_mom,
            e_gs_fo,
            d_e_fo: e_mom - e_gs_fo,
            aufbau_equal,
            e_gs: None,
        };

        // --- L2: S0 SCF at the same geometry (opt-in) -------------------------
        if self.mol.ctrl.mom_gap_check == MomGapCheck::Each {
            report.e_gs = self.mom_ground_state_energy(mpi_operator);
        }

        // --- verdict ------------------------------------------------------------
        let d_e_ev = report.d_e_fo_ev();
        let mut abort_reason: Option<String> = None;
        let mut warns: Vec<String> = Vec::new();

        // L1 is a *diagnostic*, not an abort condition. The reason is the one that
        // kills the naive "is the occupation Aufbau?" test it was meant to replace:
        // in a UKS broken-symmetry open-shell singlet the alpha and beta orbital
        // LISTS differ, so both channels can fill their own lowest n_sigma orbitals
        // while sitting on *different spatial orbitals*. Then
        //   occupation == per-channel Aufbau filling
        // holds for a perfectly healthy excited state, and dE_fo is exactly zero by
        // construction -- not because the excitation vanished. Observed on the
        // formaldehyde S1 started from a pyramidalized geometry: once n and pi*
        // crossed in the alpha channel, all three runs ("particle only",
        // "hole+particle", "all four ends") ended with |dE_fo| = 0 while the state
        // was still +2.30 eV above S0 and descending toward the planar S1 minimum.
        // The question "is this still an excited state" is therefore answered by L2
        // (the gap to the independently relaxed S0), never by dE_fo.
        let d_e_abs = d_e_ev.abs();
        if d_e_ev.is_finite() && !report.aufbau_equal && d_e_abs < self.mol.ctrl.mom_gap_warn {
            warns.push(format!(
                "small MOM-ground separation in the same orbitals (|dE| = {:.4} eV < warn threshold {:.4} eV, dE = {:+.4} eV): the two determinants are close, watch the dE(MOM - S0) column",
                d_e_abs, self.mol.ctrl.mom_gap_warn, d_e_ev
            ));
        }

        // L2: the only criterion that answers "is the state I am following still an
        // excited state". It costs one S0 SCF per step, hence opt-in
        // (`mom_gap_check`); when it is off, the guard can still catch a lost
        // *reference* (overlap) but not a state that became the ground state.
        if let Some(gap) = report.d_e_gs_ev() {
            if gap < self.mol.ctrl.mom_gap_abort {
                abort_reason = Some(format!(
                    "the MOM state is not above the ground state any more (E(MOM) - E(S0) = {:+.4} eV < {:+.4} eV): either the excitation has vanished (the determinant became the ground state) or the two states have crossed",
                    gap, self.mol.ctrl.mom_gap_abort
                ));
            } else if gap < self.mol.ctrl.mom_gap_warn {
                warns.push(format!(
                    "E(MOM) - E(S0) = {:+.4} eV (< warn threshold {:+.4} eV): approaching a degeneracy with the ground state",
                    gap, self.mol.ctrl.mom_gap_warn
                ));
            }
        }
        if report.overlap < self.mol.ctrl.mom_overlap_abort {
            abort_reason = Some(format!(
                "the tracked orbital overlaps its reference by only {:.4} (< abort threshold {:.4}): the reference no longer identifies it",
                report.overlap, self.mol.ctrl.mom_overlap_abort
            ));
        } else if report.overlap < self.mol.ctrl.mom_overlap_warn {
            warns.push(format!(
                "the tracked orbital overlaps its reference by only {:.4} (< warn threshold {:.4})",
                report.overlap, self.mol.ctrl.mom_overlap_warn
            ));
        }
        if report.s2_ref > 0.1 {
            // <S^2> is a *character* indicator, not an identity test, and it is
            // two-sided: <S^2> = 1 - |<phi_a|phi_b>|^2 for a broken-symmetry
            // determinant, so it drops when the hole and particle orbitals mix
            // (a legitimate relaxation of a singlet) and rises when the state
            // acquires triplet character (MOM can collapse onto T1 when S1 and T1
            // are close). Neither direction is by itself a collapse: the test for
            // "the state is no longer an excited state" is L2, and the test for
            // "the reference no longer identifies the orbital" is the overlap. So
            // this level only warns.
            let ratio = report.s2 / report.s2_ref;
            if !(0.5..=1.5).contains(&ratio) {
                warns.push(format!(
                    "<S^2> moved from {:.4} to {:.4} (ratio {:.2}); check whether the state kept its \
                     multiplicity character (down: mixing toward closed shell, up: triplet \
                     admixture -- S1 and T1 are close whenever the two SOMOs are)",
                    report.s2_ref, report.s2, ratio
                ));
            }
        }
        // Self-consistency of L2 against L1: the separately relaxed S0 is
        // variational, so it cannot lie above the frozen ground-state determinant
        // of the MOM orbitals -- L2 >= L1 always. A violation means the S0 SCF
        // settled on a different (higher) solution, and the gap it reports is
        // then not the one advertised. This is the cheap cross-check that turns
        // "a number came out" into "the number is the one we asked for".
        if report.aufbau_equal {
            // The occupation equals the per-channel Aufbau filling. This is NOT by
            // itself a collapse: a UKS broken-symmetry open-shell singlet keeps
            // alpha and beta orbitals distinct, so both channels filling their own
            // lowest n_sigma orbitals is a legitimate excited determinant.
            // Informational only -- L2 decides.
            warns.push(String::from(
                "occupation == per-channel Aufbau filling: dE_fo is zero by construction. For a \
                 broken-symmetry open-shell state this is normal (the alpha and beta orbital lists \
                 differ), so this line alone says nothing -- read E(MOM) - E(S0): if that is still \
                 clearly positive the state is alive and only its index pattern changed.",
            ));
        }
        if let Some(d_e_gs_ev) = report.d_e_gs_ev() {
            if d_e_gs_ev < d_e_ev - 1.0e-3 {
                warns.push(format!(
                    "L2 gap ({:+.4} eV) is below the frozen-orbital L1 gap ({:+.4} eV), which is impossible for a variational S0: the unconstrained SCF probably converged to a different solution (check its own output above)",
                    d_e_gs_ev, d_e_ev
                ));
            }
        }

        // --- report -------------------------------------------------------------
        if self.mol.ctrl.print_level >= 1 {
            let idx = report.index.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",");
            let mut line = format!(
                "[MOM] state check (step {}): orbital {} overlap {:.5} | <S^2> {:.4} (ref {:.4}) | dE(MOM - gs determinant, same orbitals) = {:+.4} eV",
                report.step, idx, report.overlap, report.s2, report.s2_ref, d_e_ev
            );
            if let Some(gap) = report.d_e_gs_ev() {
                line.push_str(&format!(" | dE(MOM - S0) = {:+.4} eV", gap));
            }
            println!("{}", line);
            if report.aufbau_equal && self.mol.ctrl.print_level >= 2 {
                println!("[MOM] note: the occupation coincides with the per-channel Aufbau filling, so the dE(MOM - gs determinant) column is zero by construction. For a broken-symmetry open-shell state (alpha and beta orbital lists differ) that is normal; the dE(MOM - S0) column is what decides whether the state is still excited.");
            }
            for w in &warns {
                println!("[MOM] WARNING: {}", w);
            }
        }

        // per-step history for the json dump (`energies` is serialized wholesale
        // by `fileop::json_dump`, so no separate writer is needed)
        self.energies.entry("mom_step".to_string()).or_insert_with(Vec::new).push(report.step as f64);
        self.energies.entry("mom_index".to_string()).or_insert_with(Vec::new)
            .push(report.index.first().map(|i| *i as f64).unwrap_or(f64::NAN));
        self.energies.entry("mom_overlap".to_string()).or_insert_with(Vec::new).push(report.overlap);
        self.energies.entry("mom_s2".to_string()).or_insert_with(Vec::new).push(report.s2);
        self.energies.entry("mom_d_e_fo_ev".to_string()).or_insert_with(Vec::new).push(d_e_ev);
        if let Some(gap) = report.d_e_gs_ev() {
            self.energies.entry("mom_d_e_s0_ev".to_string()).or_insert_with(Vec::new).push(gap);
        }

        if let Some(reason) = &abort_reason {
            let enforce = self.mol.ctrl.mom_guard == MomGuard::Enforce
                && matches!(self.mol.ctrl.job_type, JobType::Force | JobType::GeomOpt);
            if enforce {
                panic!(
                    "ERROR: the MOM state check failed at geometry step {}: {}.\n\
                     The calculation is no longer the electronic state it was asked for, so its \
                     energy and gradient must not be used. Inspect the [MOM] lines above, check \
                     whether the state crossed (dE to S0 in the log), and consider \
                     mom_strategy = \"initial\" or a different reference. Set [ctrl] \
                     mom_guard = \"warn\" to continue deliberately.",
                    report.step, reason
                );
            } else {
                println!("[MOM] ERROR: state check failed at step {}: {}", report.step, reason);
            }
        }

        self.mom_last_report = Some(report.clone());
        Some(report)
    }

    /// The `mom_gap_check = "last"` hook: report the adiabatic MOM-S0 gap at the
    /// geometry the optimization ended on.
    pub fn mom_final_gap_report(&mut self, mpi_operator: &Option<MPIOperator>) {
        if self.mol.ctrl.force_state_occupation.is_empty() {
            return;
        }
        if self.mol.ctrl.mom_gap_check != MomGapCheck::Last {
            return;
        }
        if self.mol.ctrl.print_level >= 1 {
            println!("");
            println!("[MOM] final gap check at the last geometry (mom_gap_check = \"last\")");
        }
        let e_mom = self.scf_energy;
        if let Some(e_s0) = self.mom_ground_state_energy(mpi_operator) {
            let gap = (e_mom - e_s0) * EV;
            if self.mol.ctrl.print_level >= 1 {
                println!("[MOM] E(MOM) = {:18.10} Ha, E(S0, same geometry) = {:18.10} Ha", e_mom, e_s0);
                println!("[MOM] adiabatic gap E(MOM) - E(S0) = {:+.4} eV", gap);
                if gap < 0.0 {
                    println!("[MOM] NOTE: the MOM state lies BELOW the unconstrained ground state at this geometry; check the state assignment (a crossing, or the unconstrained SCF found a different solution)");
                }
            }
            self.energies.entry("mom_final_gap_ev".to_string()).or_insert_with(Vec::new).push(gap);
            self.energies.entry("mom_final_e_s0".to_string()).or_insert_with(Vec::new).push(e_s0);
        } else {
            if self.mol.ctrl.print_level >= 1 {
                println!("[MOM] the unconstrained ground-state SCF did not converge at this geometry, or the electron count is not a singlet; no gap reported");
            }
        }
    }
}
