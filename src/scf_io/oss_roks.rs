//! Restricted open-shell singlet (OSS/ROKS) support.
//!
//! A two-determinant configuration state function written in a single
//! (restricted) set of spatial orbitals: alpha occupies the closed shell plus
//! orbital `a`, beta the closed shell plus orbital `b` (`a != b`).  The
//! physical state is the open-shell singlet
//!
//! ```text
//! E_S = 2 E_mix - E_T
//! ```
//!
//! where `E_mix` is the mixed determinant and `E_T` the same-orbital
//! (M_S = 1) triplet.  The SCF drives the variational E_S stationary point
//! with the effective Fock `G = 2 F^M - F^T` (see `generate_roothaan_fock_oss`);
//! `E_T` is evaluated single-shot by a build on the triplet occupation.
//!
//! This module owns everything that exists only for `oss_roks = true`:
//!
//! * the four-space (c / a / b / v) Roothaan Fock of the mixed determinant and
//!   the variational effective Fock of `E_S`;
//! * pinning of the open-shell pair across a degenerate manifold
//!   (`pin_oss_pair`, used post-SCF only);
//! * the post-SCF triplet quantities and the E_S / E_T / E_mix bookkeeping
//!   (`store_oss_energies`, `evaluate_oss_quantities`, ...);
//! * the occupation-indexed semi-canonicalization entry point used by the
//!   restricted open-shell PT2 drivers.
//!
//! The SCF struct itself (`SCF` fields `oss_*`) and the small dispatch
//! branches of `generate_occupation` / `generate_density_matrix` /
//! `check_scf_convergence` stay in `scf_io`.

use super::*;

/// Same-orbital triplet quantities of the open-shell singlet (DFT sum
/// route, Step 5').
///
/// E_xc of a semilocal functional is a nonlinear functional of the spin
/// densities, so the same-orbital triplet energy E_T(KS) is NOT
/// E_mix - K_ab (that identity holds only for HF, where the exchange part
/// is linear in the density); it must be evaluated single-shot with the
/// triplet occupation (alpha += b, beta -= b) over the same OSS orbitals.
/// The total density of the two states is identical -- only the spin
/// densities differ -- which is why the one-electron, Coulomb and nuclear
/// parts cancel in E_mix - E_T.
///
/// For the PT2 family (xDH / bDH) the method-level E_T(DH) additionally
/// needs the triplet EXX / XC components and the triplet PT2 correlation.
/// They are evaluated here while the DFT grids and the RI matrix are still
/// alive (`xdh_calculations` frees the grids right after its own XC
/// evaluations), and are combined with E_mix(DH) = xdh_energy by
/// `store_oss_energies_finalize` after `xdh_calculations`.
#[derive(Clone, Copy)]
pub struct OssTripletCache {
    /// single-shot triplet KS energy on the OSS orbitals
    pub e_t_ks: f64,
    /// exact-exchange energy of the triplet density (PT2 family only)
    pub x_t: f64,
    /// XC energy of the triplet density with the SCF functional (PT2 family only)
    pub xc_scf_t: f64,
    /// XC energy of the triplet density with the post-SCF (xDH) functional (PT2 family only)
    pub xc_xdh_t: f64,
    /// triplet PT2 correlation [total, opposite-spin, same-spin], evaluated
    /// on the OSS semi-canonical basis (F1: the same one-electron basis and
    /// denominators as E_mix's PT2) with the triplet occupation
    pub pt2_c: [f64; 3],
    /// method-level E_T / E_S, filled by `store_oss_energies_finalize`
    pub e_t_method: Option<f64>,
    pub e_s_method: Option<f64>,
}


/// Recompute (homo, lumo) from an occupation pattern by scanning for the
/// last occupied / first strictly empty state (the same convention as the
/// open-shell singlet branch of `generate_occupation_integer`; used for
/// the same-orbital triplet occupation in the Step 5' evaluators).
/// lumo must be the first EMPTY orbital: the PT2 drivers loop their
/// particle sums over lumo..num_state and exclude only singly-occupied
/// orbitals via their own filter, so a scan that returns a doubly
/// occupied (closed) orbital would leak it into the particle sums.
pub fn oss_homo_lumo_scan(occupation: &[Vec<f64>;2]) -> ([usize;2],[usize;2]) {
    let mut homo = [0_usize;2];
    let mut lumo = [0_usize;2];
    for i_spin in 0..2 {
        homo[i_spin] = occupation[i_spin].iter().enumerate()
            .filter(|(_i, occ)| **occ >= 1.0e-6)
            .map(|(i, _occ)| i).max().unwrap_or(0);
        lumo[i_spin] = occupation[i_spin].iter().enumerate()
            .filter(|(_i, occ)| **occ < 1.0e-6)
            .map(|(i, _occ)| i).min().unwrap_or(0);
    }
    (homo, lumo)
}


impl SCF {
    /// Pins the open-shell singlet orbital pair (a, b) across SCF
    /// iterations. On a strictly degenerate manifold (e.g. the pi_g pair
    /// of O2) the Fock block of the pair is proportional to the identity,
    /// so every diagonalization may rotate (a, b) arbitrarily, letting the
    /// density wander on the degenerate manifold and stalling the density
    /// convergence criterion. The columns of the previous Fock build are
    /// kept in oss_prev_ab. Two cases:
    ///
    /// 1. The pair is (nearly) stationary in the current Fock: either the
    ///    fresh eigenvalues are degenerate (|eps_b - eps_a| < 1e-6 Ha,
    ///    the manifold proper, e.g. the pi_g pair of O2), or the pair
    ///    rotation gradient F_ab = prev_a^T F' prev_b is below 1e-3 Ha.
    ///    The latter catches two regimes the eigenvalue gap cannot see:
    ///    the unresolvable regime (F_aa == F_bb, the fresh gap is
    ///    spuriously split by ~2|F_ab| and the diagonalizer rotates the
    ///    pair ~45 deg regardless of the true angle), and the
    ///    repelling-map regime (F_aa - F_bb is resolvable, but the 2x2
    ///    rotation map amplifies any noise rotation by
    ///    ~2|dF_ab/dtheta|/(F_aa - F_bb), so the pair escapes onto the
    ///    +-45 deg orbit within a few iterations; stretched H2 at 4 Ang,
    ///    amplification ~500x). In both cases the columns (a, b) are
    ///    replaced by the previous pair (making the density exactly
    ///    stationary), and when the Fock is available the eigenvalues are
    ///    overwritten with F_aa / F_bb so that the occupation, the
    ///    HOMO-LUMO gap print and the eigenvalue-based convergence check
    ///    see the pinned spectrum. Freezing a pair with |F_ab| < 1e-3
    ///    costs O(F_ab^2 / |d^2E/dtheta^2|) < 1e-6 Ha.
    /// 2. The pair is genuinely split (e.g. sigma_g/sigma_u of H2 at short
    ///    bond length): the fresh eigenvectors are kept; only a 2x2 overlap
    ///    check swaps them if the diagonalizer exchanged the two characters.
    ///
    /// Only a 2x2 check is performed: the pair is assumed to be isolated
    /// from the rest of the spectrum (true for the pi_g pair of O2).
    pub fn pin_oss_pair(&mut self) {
        if !self.mol.ctrl.oss_roks { return; }
        let Some(prev_ab) = self.oss_prev_ab.clone() else { return; };
        if self.eigenvectors[0].data.is_empty() { return; }
        let [orb_a, orb_b] = match self.mol.ctrl.oss_pair {
            Some(occ) => occ,
            None => return,
        };
        let num_basis = self.mol.num_basis;

        // columns (a, b) of the freshly diagonalized eigenvectors
        let mut new_ab = MatrixFull::new([num_basis, 2], 0.0);
        for (i, orb_x) in [orb_a, orb_b].iter().enumerate() {
            let start = orb_x * num_basis;
            for r in 0..num_basis {
                new_ab.data[r + i * num_basis] = self.eigenvectors[0].data[start + r];
            }
        }
        // 2x2 overlap M[i][j] = <prev_i | new_j> (both sets orthonormal)
        let ovlp_full = self.ovlp.to_matrixfull().unwrap();
        let m = apply_projection_operator(&prev_ab, &ovlp_full, &new_ab);
        // column-major: data[0] = M[0][0], data[2] = M[0][1]

        let de_pair = self.eigenvalues[0][orb_b] - self.eigenvalues[0][orb_a];
        // || P_prev new_a ||^2: 1 if the fresh pair is a rotation of the
        // previous one; guards against mid-SCF level crossings
        let in_prev_space = m.data[0] * m.data[0] + m.data[2] * m.data[2];

        // prev-pair elements of the current Fock: F_xy = prev_x^T F' prev_y.
        // F_ab is the gradient of the a<->b rotation at the previous pair
        // (the Fock cross-check of the report block); it vanishes exactly
        // when the pair is a stationary point of the current Fock. Three
        // regimes require the snap (see the decision below):
        //   (i)  the fresh pair is genuinely degenerate (|de_pair| < 1e-6,
        //        the pi_g pair of O2): the diagonalization returns an
        //        arbitrary rotation of the pair;
        //   (ii) the diagonal at the previous pair is unresolvable
        //        (|F_aa - F_bb| < 1e-4, stretched H2 at 6/9 Ang): the
        //        fresh gap is spuriously split by ~2|F_ab| and the
        //        manifold is flat, so the fresh pair is arbitrary as well;
        //   (iii) the diagonal is resolvable but the 2x2 rotation map is
        //        repelling (stretched H2 at 4 Ang): a noise rotation theta
        //        is amplified by ~2 |dF_ab/dtheta| / (F_aa - F_bb) per
        //        diagonalization (~500x, observed: theta creeps for ~5
        //        iterations, then jumps to the localized pair), which is
        //        detected by the fresh-pair gradient G_fresh growing an
        //        order of magnitude above F_ab itself.
        // A resolvable pair on an attracting map (e.g. the OSS pair of
        // 001: F_ab ~ 1e-7 but F_aa - F_bb ~ 6e-2 Ha) passes none of the
        // three tests, keeps the fresh columns and relaxes freely against
        // the closed/virtual spaces. The old |F_ab| < 1e-3 criterion fired
        // for such pairs too -- locking the SAD-guess orientation for the
        // whole SCF, whose permanent c-a/a-v gradient in the DIIS error
        // vector then produced a limit cycle (SCF never converged).
        // Whenever the snap fires, the eigenvalues are overwritten with
        // F_aa / F_bb so the occupation, the HOMO-LUMO gap print and the
        // eigenvalue-based convergence check see the pinned spectrum.
        // Freezing a pair with |F_ab| < 1e-3 costs
        // O(F_ab^2 / |d^2E/dtheta^2|) < 1e-6 Ha.
        let (f_ab, de_fock, f_aa, f_bb) = match self.roothaan_hamiltonian.as_ref() {
            Some(fock_up) => {
                let fock = fock_up.to_matrixfull().unwrap();
                let contract = |x_col: usize, y_col: usize| -> f64 {
                    let mut s = 0.0;
                    for j in 0..num_basis {
                        for i in 0..num_basis {
                            s += prev_ab.data[i + x_col * num_basis]
                                * fock.data[i + j * num_basis]
                                * prev_ab.data[j + y_col * num_basis];
                        }
                    }
                    s
                };
                let aa = contract(0, 0);
                let bb = contract(1, 1);
                let ab = contract(0, 1);
                (ab, aa - bb, aa, bb)
            }
            None => (f64::INFINITY, f64::INFINITY, 0.0, 0.0),
        };

        // fresh-pair element of the current Fock: G_fresh = fresh_a^T F' fresh_b.
        // The fresh columns come from diagonalizing the DIIS-extrapolated
        // Fock, so G_fresh is the pair's response to the extrapolation
        // shift: on a repelling rotation map it is amplified (regime (iii)
        // above), on an attracting map it stays at the noise level of F_ab
        // itself (the OSS pair of 001).
        let f_ab_fresh = match self.roothaan_hamiltonian.as_ref() {
            Some(fock_up) => {
                let fock = fock_up.to_matrixfull().unwrap();
                let mut s = 0.0;
                for j in 0..num_basis {
                    for i in 0..num_basis {
                        s += new_ab.data[i] * fock.data[i + j * num_basis] * new_ab.data[j + num_basis];
                    }
                }
                s
            }
            None => f64::INFINITY,
        };

        // Snap decision: restore the previous orientation when the fresh
        // pair cannot be trusted -- genuinely degenerate (arbitrary
        // rotation), unresolvable at the previous pair (flat manifold,
        // arbitrary rotation), or resolvable but repelling (escaping).
        // The pair must still live in the previous subspace (no level
        // crossing). A resolvable pair on an attracting map keeps the
        // fresh columns and relaxes freely against the closed/virtual
        // spaces.
        let repelling = f_ab.abs() < 1e-3 && f_ab_fresh.abs() > 10.0 * f_ab.abs() && f_ab_fresh.abs() > 1e-5;
        if in_prev_space > 0.99
            && (de_pair.abs() < 1e-6 || de_fock.abs() < 1e-4 || repelling)
        {
            // snap onto the previous columns; the eigenvalues are
            // overwritten with the prev-pair diagonal elements (when the
            // Fock is available) so that the occupation, the HOMO-LUMO
            // gap print and the eigenvalue-based convergence check see
            // the true pair spectrum instead of the spurious ~2|F_ab|
            // split. Freezing the orientation costs
            // O(F_ab^2 / |d^2E/dtheta^2|) < 1e-6 Ha.
            for (i, orb_x) in [orb_a, orb_b].iter().enumerate() {
                let start = orb_x * num_basis;
                for r in 0..num_basis {
                    self.eigenvectors[0].data[start + r] = prev_ab.data[r + i * num_basis];
                }
            }
            if de_fock.is_finite() {
                self.eigenvalues[0][orb_a] = f_aa;
                self.eigenvalues[0][orb_b] = f_bb;
            }
            if de_pair.abs() < 1e-6 {
                println!("rohf-oss: degenerate pair (dE = {:9.2e}): snapped orbitals {} and {} onto the previous pair", de_pair, orb_a, orb_b);
            } else if de_fock.abs() < 1e-4 {
                println!("rohf-oss: unresolvable pair (dF = {:9.2e}): snapped orbitals {} and {} onto the previous pair", de_fock, orb_a, orb_b);
            } else {
                println!("rohf-oss: repelling pair (G = {:9.2e}, G_fresh = {:9.2e}, dF = {:9.2e}): snapped orbitals {} and {} onto the previous pair", f_ab, f_ab_fresh, de_fock, orb_a, orb_b);
            }
        } else if m.data[2].abs() > m.data[0].abs() {
            // case 2: split pair -> keep the fresh eigenvectors, but undo a
            // character swap by the diagonalizer
            let start_a = orb_a * num_basis;
            let start_b = orb_b * num_basis;
            for r in 0..num_basis {
                self.eigenvectors[0].data.swap(start_a + r, start_b + r);
            }
            self.eigenvalues[0].swap(orb_a, orb_b);
            println!("rohf-oss: pinned the open-shell pair: swapped orbitals {} and {} (|overlap| {:.4} > {:.4})", orb_a, orb_b, m.data[2].abs(), m.data[0].abs());
        }
    }


    /// Build the four-space Roothaan Fock of the restricted open-shell
    /// singlet (the E_mix table), or the variational E_S effective Fock
    /// `G = 2 F^M - F^T` when the SCF is at its outer level.
    ///
    /// `hamiltonian_a` / `hamiltonian_b` hold F^M_alpha / F^M_beta of the
    /// build that just finished, `hamiltonian_c = (F^M_alpha + F^M_beta)/2`
    /// and `ovlp_full` the overlap matrix.
    pub(crate) fn generate_roothaan_fock_oss(
        &mut self,
        hamiltonian_a: MatrixFull<f64>,
        hamiltonian_b: MatrixFull<f64>,
        hamiltonian_c: MatrixFull<f64>,
        ovlp_full: MatrixFull<f64>,
        mpi_operator: &Option<MPIOperator>,
    ) -> MatrixUpper<f64> {
        let num_basis = self.mol.num_basis;

        // ======== ======== ====== ====== =========
        // space     closed   a      b      virtual
        // ======== ======== ====== ====== =========
        // closed      Fc      Fb     Fa     Fc
        // a           Fb      Fc     Fa-Fb  Fa
        // b           Fa      Fa-Fb  Fc     Fb
        // virtual     Fc      Fa     Fb     Fc
        // ======== ======== ====== ====== =========
        // where Fc = (Fa + Fb) / 2; the a-b block carries (Fa - Fb),
        // whose (a, b) element <ab|bb> - <aa|ab> is the energy gradient
        // of the a<->b rotation (restricted open-shell singlet).
        //
        // The first Fock build can precede the orbital guess: with a
        // density-only initial guess (e.g. SAD), the hamiltonian is
        // generated before the first diagonalization, so neither the
        // eigenvectors nor the open-shell orbital indices (a, b) exist
        // yet. Fall back to the spin-averaged Fock Fc, which is
        // well-defined from the density alone: the occupation step fixes
        // (a, b) right after this first diagonalization, and the 4-space
        // OSS Fock below is used from the next build on.
        if self.mol.ctrl.oss_pair.is_none() || self.eigenvectors[0].data.is_empty() {
            println!("rohf-oss: the Fock is built before the orbital guess is available; using the spin-averaged Fock Fc for this build");
            return hamiltonian_c.to_matrixupper();
        }
        let [orb_a, orb_b] = match self.mol.ctrl.oss_pair {
            Some(occ) => occ,
            None => panic!("ERROR: rohf-oss: the open-shell orbital indices (a, b) are not fixed yet when building the Roothaan Fock"),
        };

        // Stash the current (a, b) columns: on a strictly degenerate
        // manifold (e.g. the pi_g pair of O2) the next diagonalization
        // may rotate the pair arbitrarily, so the density step pins the
        // fresh columns against this stash by maximum overlap
        // (see SCF::pin_oss_pair).
        let mut prev_ab = MatrixFull::new([num_basis, 2], 0.0);
        for (i, orb_x) in [orb_a, orb_b].iter().enumerate() {
            let start = orb_x * num_basis;
            for r in 0..num_basis {
                prev_ab.data[r + i * num_basis] = self.eigenvectors[0].data[start + r];
            }
        }
        self.oss_prev_ab = Some(prev_ab);

        // Projectors for the closed (c), open-shell a, open-shell b and
        // virtual (v) spaces, in the one-electron density convention
        // P_x = D_x * S, so that in the orthonormal MO basis
        //   pc = sum_c |c><c|,  pa = |a><a|,  pb = |b><b|,
        //   pv = I - (D_c + |a><a| + |b><b|) S .
        let mut pa = MatrixFull::new([num_basis, num_basis], 0.0);
        let mut pb = MatrixFull::new([num_basis, num_basis], 0.0);
        for (orb_x, px) in [(orb_a, &mut pa), (orb_b, &mut pb)] {
            // c_x: column orb_x of the eigenvectors ([num_basis, num_state])
            let start = orb_x * num_basis;
            let c_x = &self.eigenvectors[0].data[start..start + num_basis];
            // P_x = |x><x| S = c_x (S c_x)^T  (column-major storage)
            for j in 0..num_basis {
                let mut sv_x = 0.0;
                for i in 0..num_basis {
                    sv_x += ovlp_full.data[i + j * num_basis] * c_x[i];
                }
                for i in 0..num_basis {
                    px.data[i + j * num_basis] = c_x[i] * sv_x;
                }
            }
        }

        let mut dm_a_s = MatrixFull::new([num_basis, num_basis], 0.0);
        _dsymm(&self.density_matrix[0], &ovlp_full, &mut dm_a_s, 'L', 'U', 1.0, 0.0);
        // pc = D_c S = (D_a - |a><a|) S
        let pc = dm_a_s.clone() - pa.clone();
        // pv = I - (D_c + |a><a| + |b><b|) S = I - D_a S - |b><b| S
        let mut pv = MatrixFull::new([num_basis, num_basis], 0.0);
        pv.iter_diagonal_mut().unwrap().for_each(|x| *x = 1.0);
        pv = pv - dm_a_s - pb.clone();

        // ====== ====== ====== ====== ==========
        // variational ROKS (Scheme B), E_S = 2 E_mix - E_T
        // ====== ====== ====== ====== ==========
        // Scheme B (always on): assemble the effective
        // Fock G of docs/oss_roks_variational.md Sec. 5 and hand it to
        // the SCF loop, so the orbitals converge to the E_S stationary
        // point shared by both channels (Q-Chem ROKS convention)
        // instead of the E_mix one. self.hamiltonian holds F^M from the
        // build that just finished; one extra build on the swapped
        // triplet density provides F^T. The F^T build's ROHF tail
        // re-enters this function with oss_variational_inner_build set
        // and falls through to the ordinary E_mix table below (that
        // output is discarded by the restore right after). Diagonal
        // blocks of G use Fc = (F^M_α + F^M_β)/2 as in the E_mix table
        // (sorting only, they do not affect the stationary point).
        if !self.oss_variational_inner_build {
            self.oss_variational_inner_build = true;

            // --- save the mixed-determinant state -----------------------
            let dm_mix = self.density_matrix.clone();
            let occ_mix = self.occupation.clone();
            let (homo_mix, lumo_mix) = (self.homo, self.lumo);
            let ham_mix = self.hamiltonian.clone();
            let roothaan_mix = self.roothaan_hamiltonian.clone();
            let e_mix = self.scf_energy;

            // --- triplet occupation: alpha += b, beta -= b --------------
            self.occupation[0][orb_b] = 1.0;
            self.occupation[1][orb_b] = 0.0;
            (self.homo, self.lumo) = oss_homo_lumo_scan(&self.occupation);
            let c_b: Vec<f64> = self.eigenvectors[0].data[orb_b * num_basis..(orb_b + 1) * num_basis].to_vec();
            for j in 0..num_basis {
                for i in 0..num_basis {
                    let p_ij = c_b[i] * c_b[j];
                    self.density_matrix[0].data[i + j * num_basis] = dm_mix[0].data[i + j * num_basis] + p_ij;
                    self.density_matrix[1].data[i + j * num_basis] = dm_mix[1].data[i + j * num_basis] - p_ij;
                }
            }
            // F^T on the triplet density (single build; the E_T(KS)
            // energy lands in scf_energy and is restored below)
            self.generate_hf_hamiltonian(mpi_operator);
            let e_t_ks = self.scf_energy;
            let f_t_a = self.hamiltonian[0].to_matrixfull().unwrap();
            let f_t_b = self.hamiltonian[1].to_matrixfull().unwrap();

            // --- restore the mixed-determinant state ---------------------
            self.density_matrix = dm_mix;
            self.occupation = occ_mix;
            self.homo = homo_mix;
            self.lumo = lumo_mix;
            self.hamiltonian = ham_mix;
            self.roothaan_hamiltonian = roothaan_mix;
            self.scf_energy = e_mix;
            self.oss_variational_inner_build = false;

            // --- combination matrices (docs/oss_roks_variational.md) -----
            let f_tilde_a = (hamiltonian_a.clone() * 2.0) - f_t_a.clone();   // F̃_α = 2F^M_α - F^T_α
            let f_tilde_b = (hamiltonian_b.clone() * 2.0) - f_t_b.clone();   // F̃_β = 2F^M_β - F^T_β
            let d_cb = (hamiltonian_a.clone() * 2.0) - f_t_b.clone();        // c-b: 2F^M_α - F^T_β
            let d_bv = (hamiltonian_b.clone() * 2.0) - f_t_a.clone();        // b-v: 2F^M_β - F^T_α
            let d_ab = (hamiltonian_a.clone() - hamiltonian_b.clone()) * 2.0; // a-b: 2(F^M_α - F^M_β)
            let f_cv = f_tilde_a.clone() + f_tilde_b.clone();                // c-v: F̃_α + F̃_β

            // --- assemble G per the E_S block table -----------------------
            let mut g_fock = MatrixFull::new([num_basis, num_basis], 0.0);
            g_fock += apply_projection_operator(&pc, &hamiltonian_c, &pc) * 0.5;
            g_fock += apply_projection_operator(&pa, &hamiltonian_c, &pa) * 0.5;
            g_fock += apply_projection_operator(&pb, &hamiltonian_c, &pb) * 0.5;
            g_fock += apply_projection_operator(&pv, &hamiltonian_c, &pv) * 0.5;
            g_fock += apply_projection_operator(&pa, &f_tilde_b, &pc);  // c-a: F̃_β
            g_fock += apply_projection_operator(&pb, &d_cb, &pc);       // c-b: 2F^M_α - F^T_β
            g_fock += apply_projection_operator(&pa, &f_tilde_a, &pv);  // a-v: F̃_α
            g_fock += apply_projection_operator(&pb, &d_bv, &pv);       // b-v: 2F^M_β - F^T_α
            g_fock += apply_projection_operator(&pa, &d_ab, &pb);       // a-b: 2(F^M_α - F^M_β)
            g_fock += apply_projection_operator(&pv, &f_cv, &pc);       // v-c: F̃_α + F̃_β
            let g_fock = g_fock.clone() + g_fock.transpose();
            // per-iteration trace of the E_S target: print_level > 1 (the
            // default level 1/2 keeps the log to the report block only)
            if self.mol.ctrl.print_level > 1 {
                println!("rohf-oss [variational ROKS] E_mix = {:18.10}, E_T(KS) = {:18.10}, E_S = 2*E_mix - E_T = {:18.10}",
                         e_mix, e_t_ks, 2.0 * e_mix - e_t_ks);
            }
            return g_fock.to_matrixupper();
        }

        let hamiltonian_ab = hamiltonian_a.clone() - hamiltonian_b.clone();

        let mut roothaan_fock = MatrixFull::new([num_basis, num_basis], 0.0);
        // diagonal blocks: Fc on each block (1/2 factor + transpose)
        roothaan_fock += apply_projection_operator(&pc, &hamiltonian_c, &pc) * 0.5;
        roothaan_fock += apply_projection_operator(&pa, &hamiltonian_c, &pa) * 0.5;
        roothaan_fock += apply_projection_operator(&pb, &hamiltonian_c, &pb) * 0.5;
        roothaan_fock += apply_projection_operator(&pv, &hamiltonian_c, &pv) * 0.5;
        // off-diagonal blocks (full weight + transpose)
        roothaan_fock += apply_projection_operator(&pa, &hamiltonian_b, &pc);   // c-a: Fb
        roothaan_fock += apply_projection_operator(&pb, &hamiltonian_a, &pc);   // c-b: Fa
        roothaan_fock += apply_projection_operator(&pa, &hamiltonian_a, &pv);   // a-v: Fa
        roothaan_fock += apply_projection_operator(&pb, &hamiltonian_b, &pv);   // b-v: Fb
        roothaan_fock += apply_projection_operator(&pa, &hamiltonian_ab, &pb);  // a-b: Fa-Fb
        roothaan_fock += apply_projection_operator(&pv, &hamiltonian_c, &pc);   // v-c: Fc
        roothaan_fock = roothaan_fock.clone() + roothaan_fock.transpose();

        return roothaan_fock.to_matrixupper();
    }

    /// Compute the open-shell singlet pair quantities from the pinned
    /// (a, b) pair: the exchange integral K_ab = (ab|ba), the Coulomb
    /// integral J_ab = (aa|bb), the gradient residual <aa|ab> - <bb|ab>
    /// and the Fock cross-check (F_a - F_b)(a,b). Returns None when the
    /// OSS setup is not available.
    ///
    /// The final re-diagonalization after the SCF loop (performed for
    /// eigenvalue printing) does not pin the (a, b) pair: on a degenerate
    /// manifold it returns an arbitrary rotation of the pair used to
    /// build the stored Fock matrices. Pin again so that all contractions
    /// below use the density-consistent pair.
    fn compute_oss_pair_quantities(&mut self) -> Option<(f64, f64, f64, f64, f64, f64)> {
        self.pin_oss_pair();
        let [orb_a, orb_b] = self.mol.ctrl.oss_pair?;
        if self.eigenvectors[0].data.is_empty() { return None; }
        let num_basis = self.mol.num_basis;

        // open-shell orbital columns and the orbital densities
        // D_a = c_a c_a^T, D_b = c_b c_b^T, and the symmetric cross
        // density D_ab = c_a c_b^T + c_b c_a^T. All three are symmetric,
        // as required by the K-build API (the batched builder mirrors the
        // off-diagonal blocks, so an unsymmetrized input would give a
        // labeling-dependent result). Under the code's K map
        // K[D]_mu,nu = sum_ls (mu l|nu s) D_ls:
        //   <aa|ab> = c_a^T K[D_a] c_b;   <bb|ab> = c_a^T K[D_b] c_b;
        //   K_ab     = c_a^T K[D_b] c_a = (ab|ba);
        //   J_ab     = c_a^T K[D_ab] c_b - K_ab = (aa|bb).
        let c_a: Vec<f64> = self.eigenvectors[0].data[orb_a * num_basis..(orb_a + 1) * num_basis].to_vec();
        let c_b: Vec<f64> = self.eigenvectors[0].data[orb_b * num_basis..(orb_b + 1) * num_basis].to_vec();
        let mut d_a = MatrixFull::new([num_basis, num_basis], 0.0);
        let mut d_b = MatrixFull::new([num_basis, num_basis], 0.0);
        let mut d_ab = MatrixFull::new([num_basis, num_basis], 0.0);
        for j in 0..num_basis {
            for i in 0..num_basis {
                d_a.data[i + j * num_basis] = c_a[i] * c_a[j];
                d_b.data[i + j * num_basis] = c_b[i] * c_b[j];
                d_ab.data[i + j * num_basis] = c_a[i] * c_b[j] + c_b[i] * c_a[j];
            }
        }

        // per-density exchange matrices: K[D_a], K[D_b], K[D_ab]
        let ks = ri_jk::generate_vk_ri_direct_dm(1.0, &[d_a, d_b, d_ab], &self.mol, None, self.mol.ctrl.batch_size);
        let ka = ks[0].to_matrixfull().unwrap();
        let kb = ks[1].to_matrixfull().unwrap();
        let kab = ks[2].to_matrixfull().unwrap();
        let contract = |m: &MatrixFull<f64>, x: &[f64], y: &[f64]| -> f64 {
            let mut s = 0.0;
            for j in 0..num_basis {
                for i in 0..num_basis {
                    s += x[i] * m.data[i + j * num_basis] * y[j];
                }
            }
            s
        };
        // <aa|ab> = c_a^T K[D_a] c_b;  <bb|ab> = c_a^T K[D_b] c_b;
        // K_ab = c_a^T K[D_b] c_a = (ab|ba);  J_ab = c_a^T K[D_ab] c_b - K_ab
        let aa_ab = contract(&ka, &c_a, &c_b);
        let bb_ab = contract(&kb, &c_a, &c_b);
        let k_ab = contract(&kb, &c_a, &c_a);
        let j_ab = contract(&kab, &c_a, &c_b) - k_ab;

        // cross-check against the Fock matrices: (F_a - F_b)(a,b) = <ab|bb> - <aa|ab>
        let fa = self.hamiltonian[0].to_matrixfull().unwrap();
        let fb = self.hamiltonian[1].to_matrixfull().unwrap();
        let fa_ab = contract(&fa, &c_a, &c_b);
        let fb_ab = contract(&fb, &c_a, &c_b);

        Some((k_ab, j_ab, aa_ab, bb_ab, fa_ab, fb_ab))
    }

    /// Compute and store the open-shell singlet energies right after the
    /// SCF loop, before chkfile / total-energy collection.
    ///
    /// HF path (bitwise unchanged from Step 4):
    ///   E_T = E_mix - K_ab             (exact same-orbital triplet identity
    ///                                    of the Fock operator)
    ///   E_S = E_mix + K_ab = 2 E_mix - E_T   (the ROKS total energy)
    /// The KS evaluator still runs once as a self-check (it must reproduce
    /// the identity to roundoff); its value is only reported, not stored.
    ///
    /// KS path (Step 5'): E_xc of a semilocal functional is nonlinear in
    /// the spin densities, so E_T(KS) is NOT E_mix - K_ab; it is evaluated
    /// single-shot with the triplet occupation (alpha += b, beta -= b) on
    /// the same OSS orbitals, and E_S = 2 E_mix - E_T(KS). For the PT2
    /// family the triplet EXX / XC components and the triplet PT2
    /// correlation are evaluated here as well -- this function runs while
    /// the DFT grids and the RI matrix are still alive, whereas
    /// `xdh_calculations` frees the grids right after its own XC
    /// evaluations -- and are cached for `store_oss_energies_finalize`.
    /// The pair quantities are cached in `oss_pair_cache` so the report
    /// block reuses them instead of rebuilding the K matrices.
    pub fn store_oss_energies(&mut self, mpi_operator: &Option<MPIOperator>) {
        if !self.mol.ctrl.oss_roks { return; }
        let Some(vals) = self.compute_oss_pair_quantities() else { return; };
        let (k_ab, _j_ab, _aa_ab, _bb_ab, _fa_ab, _fb_ab) = vals;
        let e_mix = self.scf_energy;
        self.energies.insert("oss_e_mix".to_string(), vec![e_mix]);
        self.energies.insert("oss_k_ab".to_string(), vec![k_ab]);
        self.oss_pair_cache = Some(vals);

        // HF fast path: keep the Step-4 values bitwise (all HF anchors
        // unchanged); the KS evaluator runs as a self-check only.
        let is_hf = self.mol.xc_data.dfa_compnt_scf.len() == 0;
        if is_hf {
            self.energies.insert("oss_e_triplet".to_string(), vec![e_mix - k_ab]);
            self.energies.insert("oss_e_singlet".to_string(), vec![e_mix + k_ab]);
            self.oss_triplet_cache = Some(OssTripletCache {
                e_t_ks: self.evaluate_triplet_ks_quantities(mpi_operator, false).0,
                x_t: 0.0, xc_scf_t: 0.0, xc_xdh_t: 0.0, pt2_c: [0.0; 3],
                e_t_method: None, e_s_method: None,
            });
            return;
        }

        // KS path: the same-orbital triplet is a full KS energy on the
        // triplet occupation (see the doc comment above). The RPA family
        // has no same-orbital triplet evaluator yet.
        if self.mol.xc_data.is_rpa() {
            panic!("ERROR: rohf-oss with the RPA correlation family is not yet supported (the same-orbital triplet E_T(RPA) is not implemented)");
        }

        let fifth = self.mol.xc_data.is_fifth_dfa();
        let (e_t_ks, x_t, xc_scf_t, xc_xdh_t) = self.evaluate_triplet_ks_quantities(mpi_operator, fifth);
        // Single-pass PT2 (oss_pt2_single_pass, Q-Chem ROKS structure): the
        // triplet PT2 pass is skipped -- E_S(DH) is assembled in
        // `store_oss_energies_finalize` as E_S(SCF) + PT2 -- one PT2 pass on
        // the OSS occupation (alpha in a, beta in b) -- the only PT2
        // correlation of the job, mirroring Q-Chem's one-SCF-one-MP2 ROKS
        // output. (Q-Chem's own ROKS-MP2 reference is undocumented and is
        // not matched bit-for-bit; see PR4_oss_roks.md "PT2 的参考态".) The double-pass
        // assembly (flag false) still needs the triplet correlation for
        // E_S(DH) = 2 E_mix(DH) - E_T(DH).
        let pt2_c = if fifth && !self.mol.ctrl.oss_pt2_single_pass {
            crate::ri_pt2::oss_triplet_pt2_correlation(self, mpi_operator)
        } else {
            [0.0; 3]
        };
        // SCF-level keys for now; `store_oss_energies_finalize` overwrites
        // them with the method-level values for the PT2 family (F2: a').
        self.energies.insert("oss_e_triplet".to_string(), vec![e_t_ks]);
        self.energies.insert("oss_e_singlet".to_string(), vec![2.0 * e_mix - e_t_ks]);
        self.oss_triplet_cache = Some(OssTripletCache {
            e_t_ks, x_t, xc_scf_t, xc_xdh_t, pt2_c,
            e_t_method: None, e_s_method: None,
        });
    }

    /// Evaluate the same-orbital triplet quantities of the KS path in one
    /// single-shot pass: swap the occupation to the triplet pattern
    /// (alpha += b, beta -= b), rebuild the triplet density (the total
    /// density is unchanged -- only the spin densities differ), evaluate
    /// the full KS energy through `generate_hf_hamiltonian`, and -- when
    /// `need_xc_components` (PT2 family) -- the triplet EXX and XC
    /// components on the same density. All swapped SCF fields (density,
    /// occupation, homo/lumo, hamiltonian, roothaan_hamiltonian,
    /// scf_energy) are restored before returning; the OSS state is left
    /// exactly as it was. (roothaan_hamiltonian needs the restore because
    /// the ROHF tails of the hamiltonian builders overwrite it with a Fock
    /// projected for the triplet occupation, while the report block's
    /// pin_oss_pair reads it; oss_prev_ab is re-stashed with identical
    /// content since the eigenvectors are untouched.) The solvent
    /// reaction field is intentionally NOT re-iterated for the triplet
    /// density: `generate_hf_hamiltonian` reads the stored field of the
    /// OSS SCF, i.e. E_T uses the frozen reaction field (F3).
    fn evaluate_triplet_ks_quantities(&mut self, mpi_operator: &Option<MPIOperator>, need_xc_components: bool) -> (f64, f64, f64, f64) {
        let [_, orb_b] = match self.mol.ctrl.oss_pair {
            Some(occ) => occ,
            None => panic!("ERROR: rohf-oss: the open-shell orbital indices (a, b) are not fixed yet"),
        };
        let num_basis = self.mol.num_basis;

        // --- save the mixed-determinant state -------------------------------
        let dm_mix = self.density_matrix.clone();
        let occ_mix = self.occupation.clone();
        let (homo_mix, lumo_mix) = (self.homo, self.lumo);
        let ham_mix = self.hamiltonian.clone();
        let roothaan_mix = self.roothaan_hamiltonian.clone();
        let e_mix = self.scf_energy;

        // --- triplet occupation: alpha += b, beta -= b -----------------------
        self.occupation[0][orb_b] = 1.0;
        self.occupation[1][orb_b] = 0.0;
        (self.homo, self.lumo) = oss_homo_lumo_scan(&self.occupation);

        // --- triplet density: D_T = D_mix + P_b (alpha), D_mix - P_b (beta) --
        // P_b = |b><b| is the projector on the open-shell orbital b
        let c_b: Vec<f64> = self.eigenvectors[0].data[orb_b * num_basis..(orb_b + 1) * num_basis].to_vec();
        for j in 0..num_basis {
            for i in 0..num_basis {
                let p_ij = c_b[i] * c_b[j];
                self.density_matrix[0].data[i + j * num_basis] = dm_mix[0].data[i + j * num_basis] + p_ij;
                self.density_matrix[1].data[i + j * num_basis] = dm_mix[1].data[i + j * num_basis] - p_ij;
            }
        }

        // --- single-shot triplet KS energy -----------------------------------
        // the energy formula D_t H + D_a F_a + D_b F_b (+ nuc - vxc + exc) is
        // the standard determinant energy for any alpha/beta density, so one
        // hamiltonian build on the triplet density yields E_T(KS) directly
        // (the inner gate keeps the ROHF tail out of the variational branch:
        // this swap must not trigger a second swap inside it)
        self.oss_variational_inner_build = true;
        self.generate_hf_hamiltonian(mpi_operator);
        self.oss_variational_inner_build = false;
        let e_t_ks = self.scf_energy;

        // --- triplet EXX / XC components (PT2 family only) -------------------
        let (mut x_t, mut xc_scf_t, mut xc_xdh_t) = (0.0, 0.0, 0.0);
        if need_xc_components {
            x_t = self.evaluate_exact_exchange_ri_v(mpi_operator);
            xc_scf_t = self.evaluate_xc_energy(0, mpi_operator);
            xc_xdh_t = self.evaluate_xc_energy(1, mpi_operator);
        }

        // --- restore the mixed-determinant state -----------------------------
        self.density_matrix = dm_mix;
        self.occupation = occ_mix;
        self.homo = homo_mix;
        self.lumo = lumo_mix;
        self.hamiltonian = ham_mix;
        self.roothaan_hamiltonian = roothaan_mix;
        self.scf_energy = e_mix;

        (e_t_ks, x_t, xc_scf_t, xc_xdh_t)
    }


    /// Finalize the method-level open-shell singlet energies after the
    /// post-SCF (PT2-family) step. The SCF-level E_T(KS) / E_S(KS) were
    /// stored by `store_oss_energies`. Two assemblies of the method-level
    /// E_S(DH) are supported, selected by `oss_pt2_single_pass`:
    ///
    /// single-pass (default, Q-Chem ROKS structure -- one SCF, one PT2):
    ///   E_S(DH) = E_S(SCF) + corr_S + Ec[PT2]
    /// where Ec[PT2] is the correlation evaluated on the OSS occupation
    /// (alpha in a, beta in b) in the OSS semi-canonical basis
    /// (the "xdh_pt2_energy" key,
    /// inserted by `xdh_calculations`); the triplet PT2 pass was skipped,
    /// so the method-level E_T stays at the KS value. corr_S raises the
    /// exchange/XC level of the SCF energy to the final hybrid of the
    /// fifth-rung family (hy_xdh != hy_scf, e.g. XYG3 with a B3LYP SCF
    /// base): the post-SCF corrections of `xdh_calculations` are per-state
    /// densities, so the E_S = 2 E_mix - E_T combination raises both
    /// determinants,
    ///   corr_S = 2 (x_hf [mix] dhy + dxc[mix])
    ///              - (x_t [triplet] dhy + dxc[triplet])
    /// with the mix-side values under the "x_hf" / "xc_energy_scf" /
    /// "xc_energy_xdh" keys of `xdh_calculations` (mixed occupation; see
    /// the OSS dispatch note at ri_pt2/mod.rs) and the triplet side cached
    /// by `evaluate_triplet_ks_quantities` (OssTripletCache.x_t / xc_scf_t
    /// / xc_xdh_t). Families with hy_xdh == hy_scf and identical XC
    /// components (PBE-QIDH, B2PLYP) have corr_S == 0 and reduce to the
    /// plain E_S(SCF) + Ec[PT2].
    ///
    /// double-pass (oss_pt2_single_pass = false, the F1-based assembly):
    /// combining the cached triplet components with the method-level mixed
    /// energy E_mix(DH) = xdh_energy (the same assembly as the total-energy
    /// formula of `xdh_calculations`):
    ///   E_T(DH) = E_T(KS) + x_t (hy_xdh - hy_scf)
    ///                     + (xc_xdh_t - xc_scf_t)
    ///                     + c_os PT2_t[os] + c_ss PT2_t[ss]
    ///   E_S(DH) = 2 E_mix(DH) - E_T(DH)
    /// PT2 reference conventions: the PT2 inside E_mix(DH) = xdh_energy is
    /// evaluated on the **OSS occupation** (alpha in a, beta in b -- the
    /// beta hole kept open) in the OSS semi-canonical basis. This is REST's
    /// own convention; Q-Chem's ROKS-MP2 reference is undocumented and its
    /// numbers differ by ~2.5 mHa (see PR4_oss_roks.md). The earlier
    /// "symmetric Aufbau" (beta hole filled) variant
    /// was removed on 2026-10-28: it built a closed-shell MP2 on orbitals
    /// that are stationary for E_S but not for that closed shell, and
    /// overestimated the correlation by ~1.74x (H2/6-31G B2PLYP).
    /// PT2_t is evaluated on the triplet occupation over the OSS
    /// semi-canonical basis (`oss_triplet_pt2_correlation`).
    /// F2 (a'): oss_e_triplet / oss_e_singlet are overwritten with the
    /// method-level values; oss_e_mix keeps the SCF E_mix by definition
    /// (E_mix(DH) stays available under the "xdh_energy" key). The
    /// SCF-level singlet (2 E_mix - E_T(KS)) is preserved under the
    /// "oss_e_singlet_scf" key before the overwrite, so that output_result
    /// can print it as "The ROKS SCF energy" next to the method-level
    /// total (the analogue of Q-Chem's ROKS "SCF energy in the final
    /// basis set" line). Nothing is printed here -- every ROKS energy is
    /// reported exactly once, either in the Mission-accomplished block or
    /// in the rohf-oss report table. For non-PT2-family methods the SCF
    /// level already IS the method level, and this function does nothing.
    pub fn store_oss_energies_finalize(&mut self) {
        if !self.mol.ctrl.oss_roks { return; }

        // Preserve the SCF-level singlet (2 E_mix - E_T(KS), stored by
        // store_oss_energies) under its own key before oss_e_singlet is
        // overwritten below: output_result prints "The ROKS energy (E_S)"
        // first (the method-level total) and "The ROKS SCF energy" from
        // this key, which corresponds to Q-Chem's ROKS "SCF energy in
        // the final basis set" line. For HF / non-PT2 KS the two keys
        // are identical and output_result prints a single line.
        if let Some(v) = self.energies.get("oss_e_singlet") {
            self.energies.insert("oss_e_singlet_scf".to_string(), v.clone());
        }

        if !self.mol.xc_data.is_fifth_dfa() { return; }
        let Some(cache) = &mut self.oss_triplet_cache else { return; };
        let hy_scf = self.mol.xc_data.dfa_hybrid_scf;
        let hy_xdh = self.mol.xc_data.dfa_hybrid_pos.unwrap_or(0.0);
        let pt2_coeff = self.mol.xc_data.dfa_paramr_adv.clone().unwrap_or(vec![0.0, 0.0]);
        let (e_t_dh, e_s_dh) = if self.mol.ctrl.oss_pt2_single_pass {
            // single-pass (Q-Chem ROKS structure): E_S(DH) = E_S(SCF) +
            // corr_S + Ec[PT2], the PT2 correlation on the OSS occupation
            // in the OSS semi-canonical basis (the only PT2
            // of the job; the triplet pass was skipped in
            // `store_oss_energies`).
            let x_hf = match self.energies.get("x_hf") {
                Some(e) => e[0],
                None => 0.0,
            };
            let xc_delta = match (self.energies.get("xc_energy_xdh"), self.energies.get("xc_energy_scf")) {
                (Some(ex), Some(es)) => ex[0] - es[0],
                _ => 0.0,
            };
            // corr_S: raise E_S(SCF) from the SCF-level hybrid to the final
            // one of the fifth-rung family. The post-SCF corrections of
            // `xdh_calculations` are evaluated per-state density, so the
            // E_S = 2 E_mix - E_T combination raises each determinant
            // separately: 2x the mix-side correction (x_hf / xc_delta
            // above, keys of `xdh_calculations`, mixed occupation) minus
            // the triplet-side correction (cached by
            // `evaluate_triplet_ks_quantities`). For PBE-QIDH / B2PLYP
            // (hy_xdh == hy_scf, same XC components) corr_S vanishes and
            // the assembly is the plain E_S(SCF) + Ec[PT2].
            let hy_delta = hy_xdh - hy_scf;
            let corr_s = 2.0 * (x_hf * hy_delta + xc_delta)
                       - (cache.x_t * hy_delta + (cache.xc_xdh_t - cache.xc_scf_t));
            let ec_pt2 = match self.energies.get("xdh_pt2_energy") {
                Some(e) => e[0],
                None => panic!("ERROR: rohf-oss: xdh_pt2_energy is missing at finalize time (the post-SCF step did not run?)"),
            };
            let e_s_scf = match self.energies.get("oss_e_singlet_scf") {
                Some(v) => v[0],
                None => panic!("ERROR: rohf-oss: oss_e_singlet_scf is missing at finalize time (store_oss_energies did not run?)"),
            };
            // E_T carries no PT2 in the single-pass structure: the
            // method-level value stays at the KS triplet energy.
            (cache.e_t_ks, e_s_scf + corr_s + ec_pt2)
        } else {
            // double-pass (F1-based): E_T(DH) from the cached triplet KS
            // components and the triplet PT2, E_S(DH) = 2 E_mix(DH) - E_T(DH).
            let e_mix_dh = match self.energies.get("xdh_energy") {
                Some(e) => e[0],
                None => panic!("ERROR: rohf-oss: xdh_energy is missing at finalize time (the post-SCF step did not run?)"),
            };
            let e_t_pt2 = cache.pt2_c[1] * pt2_coeff[0] + cache.pt2_c[2] * pt2_coeff[1];
            let e_t_dh = cache.e_t_ks + cache.x_t * (hy_xdh - hy_scf) + (cache.xc_xdh_t - cache.xc_scf_t) + e_t_pt2;
            (e_t_dh, 2.0 * e_mix_dh - e_t_dh)
        };
        self.energies.insert("oss_e_triplet".to_string(), vec![e_t_dh]);
        self.energies.insert("oss_e_singlet".to_string(), vec![e_s_dh]);
        cache.e_t_method = Some(e_t_dh);
        cache.e_s_method = Some(e_s_dh);
    }

    /// Post-SCF report for the restricted open-shell singlet. The center
    /// of the report is a 3-state x 2-level energy table:
    ///   states: E_mix (the converged mixed determinant of the OSS SCF,
    ///                   <S^2> = 1, an intermediate construction only),
    ///           E_T   (the same-orbital triplet, spin-pure),
    ///           E_S   (the ROKS open-shell singlet = 2 E_mix - E_T);
    ///   levels: the SCF level and -- for the PT2 family -- the method
    ///   level assembled by `store_oss_energies_finalize` (the ROKS total
    ///   energy). The K_ab / J_ab integrals, the a<->b gradient residual,
    ///   the Fock cross-check, and the semi-canonical orbital energies
    ///   are diagnostics of lower importance than these energies and are
    ///   printed only at print_level >= 3 (the default print_level = 2
    ///   hides them; the semi-canonical diagonalization still runs inside
    ///   `semi_diagonalize_hamiltonian_outside_oss` whenever the PT2
    ///   passes need that basis, so skipping it here costs nothing).
    pub fn evaluate_oss_quantities(&mut self) {
        if !self.mol.ctrl.oss_roks { return; }
        // The pair quantities were computed and cached right after the SCF
        // loop by store_oss_energies; recompute only when that never ran
        // (e.g. direct calls through the python bindings).
        let Some((k_ab, j_ab, aa_ab, bb_ab, fa_ab, fb_ab)) = (match self.oss_pair_cache {
            Some(vals) => Some(vals),
            None => self.compute_oss_pair_quantities(),
        }) else {
            return;
        };
        // keep the eigenvectors density-consistent with the cached pair
        self.pin_oss_pair();
        let [orb_a, orb_b] = match self.mol.ctrl.oss_pair { Some(occ) => occ, None => return };
        let [s2, mlpy] = evaluate_spin_angular_momentum(&self.density_matrix, &self.ovlp, 2, &self.mol.num_elec);
        let e_mix = self.scf_energy;
        let is_hf = self.mol.xc_data.dfa_compnt_scf.len() == 0;

        println!("");
        println!("====================================================");
        println!("          rohf-oss: open-shell singlet report");
        println!("====================================================");
        println!("Open-shell orbitals  : a = {}, b = {}", orb_a, orb_b);
        println!("<S^2> of E_mix      : {:18.10}  (<2S+1> = {:.6})", s2, mlpy);

        // 3-state x 2-level energy table. The SCF-level E_S is the spin-
        // purified 2 E_mix - E_T(KS); the method-level columns exist for
        // the PT2 family only (values cached by store_oss_energies_finalize).
        if let Some(cache) = &self.oss_triplet_cache {
            let e_t_ks = cache.e_t_ks;
            let e_s_ks = 2.0 * e_mix - e_t_ks;
            if self.mol.xc_data.is_fifth_dfa() {
                let e_mix_dh = match self.energies.get("xdh_energy") {
                    Some(e) => e[0],
                    None => panic!("ERROR: rohf-oss: xdh_energy is missing at report time (the post-SCF step did not run?)"),
                };
                let (e_t_dh, e_s_dh) = match (cache.e_t_method, cache.e_s_method) {
                    (Some(et), Some(es)) => (et, es),
                    _ => panic!("ERROR: rohf-oss: method-level energies are missing at report time (store_oss_energies_finalize did not run?)"),
                };
                println!("---------------------------------------------------------------");
                println!("  {:<20}{:<20}{:<20}", "State", "SCF level (KS)", "Method level (xDH)");
                println!("  ------------------------------------------------------------");
                println!("  {:<20}{:>18.10}  {:>18.10}", "E_mix  (mixed det.)", e_mix, e_mix_dh);
                println!("  {:<20}{:>18.10}  {:>18.10}", "E_T    (triplet)", e_t_ks, e_t_dh);
                println!("  {:<20}{:>18.10}  {:>18.10}   <- ROKS total", "E_S    (ROKS)", e_s_ks, e_s_dh);
                println!("  ------------------------------------------------------------");
                if self.mol.ctrl.oss_pt2_single_pass {
                    println!("  SCF level: E_S = 2 E_mix - E_T(KS). Method level (single-pass PT2,");
                    println!("  oss_pt2_single_pass): E_S(DH) = E_S(SCF) + Ec[PT2], with PT2 on the");
                    println!("  OSS occupation (alpha in a, beta in b) in the OSS semi-canonical");
                    println!("  basis -- the native OSS reference (the Q-Chem ROKS MP2 reference");
                    println!("  differs and is undocumented; E_T carries no PT2; the method-level");
                    println!("  E_T column repeats the KS value).");
                } else {
                    println!("  E_S = 2 E_mix - E_T at each level. The method level adds the");
                    println!("  PT2 correlation of the post-SCF family to the SCF level, with");
                    println!("  PT2(E_mix) on the OSS occupation (the REST OSS reference)");
                    println!("  and PT2(E_T) on the triplet occupation over the same OSS");
                    println!("  orbitals.");
                }
            } else if is_hf {
                // HF: the evaluator's E_T(KS) must reproduce the exact
                // same-orbital identity E_mix - K_ab (self-check line);
                // E_S = E_mix + K_ab is already the method level
                println!("---------------------------------------------------------------");
                println!("  E_T (triplet)     : {:18.10} Ha   (= E_mix - K_ab, same-orbital identity)", e_t_ks);
                println!("  E_S (ROKS)        : {:18.10} Ha   (= E_mix + K_ab, the ROKS total)", e_s_ks);
                println!("  HF self-check     : |E_T(evaluator) - (E_mix - K_ab)| = {:12.3e} Ha", (e_t_ks - (e_mix - k_ab)).abs());
                println!("---------------------------------------------------------------");
            } else {
                // KS without a PT2 family: E_T(KS) is a single-shot KS
                // energy on the triplet occupation of the same orbitals;
                // the HF-only identity E_mix - K_ab does not hold for KS
                println!("---------------------------------------------------------------");
                println!("  E_T (triplet)     : {:18.10} Ha   (single-shot KS on the triplet occupation)", e_t_ks);
                println!("  E_S (ROKS)        : {:18.10} Ha   (= 2 E_mix - E_T(KS), the ROKS total)", e_s_ks);
                println!("---------------------------------------------------------------");
            }
        } else {
            // python-binding path without store_oss_energies: HF-style only
            println!("---------------------------------------------------------------");
            println!("  E_T (triplet)     : {:18.10} Ha   (= E_mix - K_ab, same-orbital identity)", e_mix - k_ab);
            println!("  E_S (ROKS)        : {:18.10} Ha   (= E_mix + K_ab, the ROKS total)", e_mix + k_ab);
            println!("---------------------------------------------------------------");
        }
        if self.mol.ctrl.solvent_enabled {
            println!("WARNING: E_T is evaluated with the solvent reaction field frozen");
            println!("         from the OSS SCF (no solvent re-iteration for the triplet density)");
        }

        // --- detailed diagnostics (print_level >= 3) --------------------------
        if self.mol.ctrl.print_level >= 3 {
            println!("");
            println!("rohf-oss: pair diagnostics (exchange / Coulomb integrals of the open pair,");
            println!("         gradient residual of the a<->b rotation, Fock cross-check):");
            println!("Exchange integral    : K_ab = (ab|ba) = {:18.10} Ha", k_ab);
            println!("Coulomb integral     : J_ab = (aa|bb) = {:18.10} Ha", j_ab);
            println!("Gradient residual    : <aa|ab> - <bb|ab> = {:12.6e} Ha", aa_ab - bb_ab);
            println!("     <aa|ab> = {:12.6e} Ha;  <bb|ab> = {:12.6e} Ha", aa_ab, bb_ab);
            println!("Fock cross-check     : (F_a - F_b)(a,b) = {:12.6e} Ha (should equal <bb|ab> - <aa|ab>)", fa_ab - fb_ab);
            // Strict OS semi-canonicalization: each spin channel is
            // diagonalized independently on its own occupied / virtual
            // subspaces with F^alpha / F^beta, so alpha and beta have
            // different semi-canonical orbital energies and occupations.
            self.semi_diagonalize_hamiltonian();
            if let Some(semi_ev) = &self.semi_eigenvalues {
                println!("");
                println!("rohf-oss: semi-canonical orbital energies and occupations (per-spin OS semi-canonicalization):");
                println!("  index  occ(alpha)  occ(beta)  eps(alpha)    eps(beta)");
                let ncol = semi_ev[0].len();
                for i in 0..ncol {
                    let occ_a = self.occupation[0][i];
                    let occ_b = self.occupation[1][i];
                    println!("{:6}   {:8.4}   {:8.4}   {:12.6}   {:12.6}",
                             i, occ_a, occ_b, semi_ev[0][i], semi_ev[1][i]);
                }
            }
        }
    }
}

/// Entry point kept for the restricted open-shell singlet; see
/// [`semi_diagonalize_from_occupation`].
pub(crate) fn semi_diagonalize_hamiltonian_outside_oss(scf_data: &SCF) -> (Option<[MatrixFull<f64>; 2]>, Option<[Vec<f64>; 2]>, Option<[MatrixFull<f64>; 2]>, usize) {
    semi_diagonalize_from_occupation(scf_data)
}

