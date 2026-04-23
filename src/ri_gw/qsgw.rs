//! RI-QSGW implementation following the MolGW algorithm path.
//!
//! Reference MolGW routines:
//!   m_gw_selfenergy_analytic.f90:654  gw_selfenergy_qs
//!   m_selfenergy_tools.f90:723        setup_exchange_m_vxc
//!   m_selfenergy_tools.f90:813        apply_qs_approximation
//!   m_scf_loop.f90:181                QSGW injection into SCF Hamiltonian
//!
//! See ri_gw/QSGW_WARNING.md for all deviations from MolGW.

use rest_tensors::matrix::{MatrixFull, MathMatrix, matrixupper::MatrixUpper};
use rest_tensors::matrix::matrix_blas_lapack::{_dgemm_full, _dsyev};
use crate::scf_io::{SCF, diagonalize_hamiltonian_outside};
use crate::ri_bse;
use crate::ri_gw;

// ---------------------------------------------------------------------------
// 1. Index helpers
// ---------------------------------------------------------------------------

/// Flat index for an (occ i, vir a) pair.
/// Ordering: ia = a * occ_size + i  (confirmed from construct_energy_diag_for_a).
#[inline]
pub fn ia_index(i: usize, a: usize, occ_size: usize) -> usize {
    a * occ_size + i
}

/// Flat column index for a full (p, q) MO pair in B_full shaped [num_auxbas, num_state*num_state].
/// B_full column = p * num_state + q  (row-major on the MO pair).
#[inline]
pub fn pq_index(p: usize, q: usize, num_state: usize) -> usize {
    p * num_state + q
}

// ---------------------------------------------------------------------------
// 2. RPA pole diagonalization
// ---------------------------------------------------------------------------

/// Build energy-difference vector eps_ia = e_a - e_i.
/// Flat order: ia = a * occ_size + i  (same as construct_energy_diag_for_a).
pub fn build_rpa_energy_differences(
    qp_energies: &[f64],
    occ_size: usize,
    vir_size: usize,
) -> Vec<f64> {
    let mut eps = Vec::with_capacity(occ_size * vir_size);
    for a in 0..vir_size {
        for i in 0..occ_size {
            eps.push(qp_energies[occ_size + a] - qp_energies[i]);
        }
    }
    eps
}

/// Build the symmetrised RPA matrix  M_tilde:
///   M_tilde[ia,jb] = 2 * sqrt(eps_ia) * V[ia,jb] * sqrt(eps_jb)
///   M_tilde[ia,ia] += eps_ia^2
/// where V = B_ov^T * B_ov.
pub fn build_rpa_symmetrized_matrix(
    ri_ov: &MatrixFull<f64>,
    eps: &[f64],
    occ_size: usize,
    vir_size: usize,
) -> MatrixFull<f64> {
    let nov = occ_size * vir_size;
    let mut v = MatrixFull::new([nov, nov], 0.0_f64);
    _dgemm_full(ri_ov, 'T', ri_ov, 'N', &mut v, 1.0, 0.0);

    let sqrt_eps: Vec<f64> = eps.iter().map(|e| e.sqrt()).collect();
    for ia in 0..nov {
        for jb in 0..nov {
            v[[ia, jb]] *= 2.0 * sqrt_eps[ia] * sqrt_eps[jb];
        }
    }
    for ia in 0..nov {
        v[[ia, ia]] += eps[ia] * eps[ia];
    }
    v
}

/// Diagonalize M_tilde -> (omega2[s], eigenvectors).
/// omega2[s] = Omega_s^2.
/// NOTE: uses _dsyev (LAPACK DSYEV) instead of MolGW DSYEVD. See QSGW_WARNING.md.
pub fn diagonalize_rpa_poles(m_tilde: MatrixFull<f64>) -> (Vec<f64>, MatrixFull<f64>) {
    let (eigenvectors,eigenvalues,_)=_dsyev(&m_tilde, 'V');
    (eigenvalues,eigenvectors.unwrap())
}

/// Build (X+Y)[ia, s] = sqrt(eps_ia) * Z[ia,s] / sqrt(Omega_s).
/// Returns shape [nov, n_poles].
pub fn build_xpy(
    eigenvectors: &MatrixFull<f64>,
    omega2: &[f64],
    eps: &[f64],
    occ_size: usize,
    vir_size: usize,
) -> MatrixFull<f64> {
    let nov = occ_size * vir_size;
    let n_poles = omega2.len();
    let sqrt_eps: Vec<f64> = eps.iter().map(|e| e.sqrt()).collect();
    let inv_sqrt_omega: Vec<f64> = omega2.iter().map(|w2| 1.0 / w2.sqrt()).collect();
    let mut xpy = MatrixFull::new([nov, n_poles], 0.0_f64);
    for s in 0..n_poles {
        for ia in 0..nov {
            xpy[[ia, s]] = sqrt_eps[ia] * eigenvectors[[ia, s]] * inv_sqrt_omega[s];
        }
    }
    xpy
}

// ---------------------------------------------------------------------------
// 3. Screened coupling vectors  w[P, s] = B_ov * (X+Y)
// ---------------------------------------------------------------------------

/// w[P, s] = spin_factor * sum_{ia} B_ov[P, ia] * (X+Y)[ia, s].
/// Returns shape [num_auxbas, n_poles].
pub fn build_screened_couplings(
    ri_ov: &MatrixFull<f64>,
    xpy: &MatrixFull<f64>,
    spin_factor: f64,
) -> MatrixFull<f64> {
    let num_auxbas = ri_ov.size[0];
    let n_poles = xpy.size[1];
    let mut w = MatrixFull::new([num_auxbas, n_poles], 0.0_f64);
    _dgemm_full(ri_ov, 'N', xpy, 'N', &mut w, spin_factor, 0.0);
    w
}

// ---------------------------------------------------------------------------
// 4. Full off-diagonal correlation self-energy  Sigma_c[p,q]
// ---------------------------------------------------------------------------

/// Extract B_pi block of shape [num_auxbas, num_state] for fixed inner state i:
///   B_pi[P, p] = B_full[P, pq_index(p, i, num_state)]
fn extract_b_pi_block(
    ri_full: &MatrixFull<f64>,
    i: usize,
    num_state: usize,
) -> MatrixFull<f64> {
    let num_auxbas = ri_full.size[0];
    let mut b_pi = MatrixFull::new([num_auxbas, num_state], 0.0_f64);
    for p in 0..num_state {
        let col = pq_index(p, i, num_state);
        if col < ri_full.size[1] {
            for aux in 0..num_auxbas {
                b_pi[[aux, p]] = ri_full[[aux, col]];
            }
        }
    }
    b_pi
}

/// Contract W_i[s, q] = sum_P w[P,s] * B_pi[P,q].  Shape [n_poles, num_state].
fn contract_w_bpi(
    w: &MatrixFull<f64>,
    b_pi: &MatrixFull<f64>,
) -> MatrixFull<f64> {
    let n_poles = w.size[1];
    let num_state = b_pi.size[1];
    let mut wi = MatrixFull::new([n_poles, num_state], 0.0_f64);
    _dgemm_full(w, 'T', b_pi, 'N', &mut wi, 1.0, 0.0);
    wi
}

/// Build full off-diagonal Sigma_c[p,q] in MO basis (MolGW gw_selfenergy_qs path).
///
/// For each inner state i (all states, occupied sign +1, virtual sign -1):
///   Sigma_c[p,q] += sum_s W_i[s,p]*W_i[s,q] * sign_i
///                     * 0.5 * Re[1/(e_q-e_i-Omega_s+i*eta)
///                                + 1/(e_q-e_i+Omega_s+i*eta)]
///
/// The real-axis energy argument e_q is the diagonal QP energy of the column
/// state (Kotani QS approximation).
pub fn build_sigma_c_mo(
    ri_full: &MatrixFull<f64>,
    w: &MatrixFull<f64>,
    qp_energies: &[f64],
    omega: &[f64],
    occ_size: usize,
    num_state: usize,
    eta: f64,
) -> MatrixFull<f64> {
    let n_poles = omega.len();
    let mut sigma = MatrixFull::new([num_state, num_state], 0.0_f64);

    for i in 0..num_state {
        let b_pi = extract_b_pi_block(ri_full, i, num_state);
        let wi = contract_w_bpi(w, &b_pi); // [n_poles, num_state]
        let e_i = qp_energies[i];
        let sign_i = if i < occ_size { 1.0_f64 } else { -1.0_f64 };

        for q in 0..num_state {
            let e_q = qp_energies[q];
            for p in 0..num_state {
                let mut sum = 0.0_f64;
                for s in 0..n_poles {
                    let ww = wi[[s, p]] * wi[[s, q]];
                    let dm = e_q - e_i - omega[s];
                    let dp = e_q - e_i + omega[s];
                    // real-axis Lorentzian regularization
                    let rm = dm / (dm * dm + eta * eta);
                    let rp = dp / (dp * dp + eta * eta);
                    sum += ww * 0.5 * (rm + rp);
                }
                sigma[[p, q]] += sign_i * sum;
            }
        }
    }
    sigma
}

// ---------------------------------------------------------------------------
// 5. Kotani Hermitianization
// ---------------------------------------------------------------------------

/// Sigma_QS[p,q] = 0.5 * (Sigma_c[p,q] + Sigma_c[q,p]).
pub fn symmetrize_sigma_qs(sigma: &MatrixFull<f64>, num_state: usize) -> MatrixFull<f64> {
    let mut out = MatrixFull::new([num_state, num_state], 0.0_f64);
    for p in 0..num_state {
        for q in 0..num_state {
            out[[p, q]] = 0.5 * (sigma[[p, q]] + sigma[[q, p]]);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 6. Exchange – Vxc static operator in MO basis
// ---------------------------------------------------------------------------

/// V_x[p,q] = -sum_i sum_P B[P,p,i] * B[P,q,i]  (bare exchange in MO basis).
pub fn build_vx_mo(
    ri_full: &MatrixFull<f64>,
    occ_size: usize,
    num_state: usize,
) -> MatrixFull<f64> {
    let mut vx = MatrixFull::new([num_state, num_state], 0.0_f64);
    for i in 0..occ_size {
        let b_pi = extract_b_pi_block(ri_full, i, num_state);
        let mut contrib = MatrixFull::new([num_state, num_state], 0.0_f64);
        _dgemm_full(&b_pi, 'T', &b_pi, 'N', &mut contrib, -1.0, 0.0);
        for p in 0..num_state {
            for q in 0..num_state {
                vx[[p, q]] += contrib[[p, q]];
            }
        }
    }
    vx
}

/// Build H_QS[p,q] = V_x[p,q] - diag(V_xc)[p,p]*delta_{pq} + Sigma_QS[p,q].
/// vxc_diag: diagonal KS xc potential in MO basis (length num_state).
pub fn build_hqs_mo(
    vx: &MatrixFull<f64>,
    vxc_diag: &[f64],
    sigma_qs: &MatrixFull<f64>,
    num_state: usize,
) -> MatrixFull<f64> {
    let mut h = MatrixFull::new([num_state, num_state], 0.0_f64);
    for p in 0..num_state {
        for q in 0..num_state {
            h[[p, q]] = vx[[p, q]] + sigma_qs[[p, q]];
        }
        h[[p, p]] -= vxc_diag[p];
    }
    h
}

// ---------------------------------------------------------------------------
// 7. MO -> AO back-transform + inject into Fock
// ---------------------------------------------------------------------------

/// M_AO = C * M_MO * C^T  (num_basis x num_basis).
pub fn mo_to_ao(
    m_mo: &MatrixFull<f64>,
    c: &MatrixFull<f64>,
) -> MatrixFull<f64> {
    let num_basis = c.size[0];
    let num_state = c.size[1];
    let mut tmp = MatrixFull::new([num_basis, num_state], 0.0_f64);
    _dgemm_full(c, 'N', m_mo, 'N', &mut tmp, 1.0, 0.0);
    let mut m_ao = MatrixFull::new([num_basis, num_basis], 0.0_f64);
    _dgemm_full(&tmp, 'N', c, 'T', &mut m_ao, 1.0, 0.0);
    m_ao
}

/// Add a symmetric MatrixFull correction to a MatrixUpper Fock matrix.
/// correction is num_basis x num_basis (symmetric).
fn add_full_to_upper(fock: &mut MatrixUpper<f64>, correction: &MatrixFull<f64>, num_basis: usize) {
    for p in 0..num_basis {
        for q in 0..=p {
            let idx = p * (p + 1) / 2 + q;
            fock.data[idx] += correction[[p, q]];
        }
    }
}

// ---------------------------------------------------------------------------
// 8. Outer QSGW fixed-point loop
// ---------------------------------------------------------------------------

/// Run the QSGW outer loop.
///
/// Algorithm:
///   1. Build RPA poles from current QP energies.
///   2. Build screened couplings w_s.
///   3. Build full Sigma_c + Kotani symmetrization.
///   4. Build H_QS = Vx - Vxc + Sigma_QS in MO basis.
///   5. Back-transform to AO, add to base Fock, re-diagonalize.
///   6. Repeat until QP energies converge.
///
/// Returns the converged QP eigenvalues (length num_state).
pub fn qsgw_loop(
    scf_data: &mut SCF,
    vxc_nn: &[f64],
    mpi_operator: &Option<crate::mpi_io::MPIOperator>,
) -> Vec<f64> {
    let qp_ctrl = scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let max_iter   = qp_ctrl.qsgw_max_iter;
    let etol       = qp_ctrl.qsgw_energy_tol;
    let mix        = qp_ctrl.qsgw_mix_param;
    let eta        = qp_ctrl.qsgw_eta;
    let spin_channel = scf_data.mol.spin_channel;

    // Save the initial (KS/HF) Hamiltonian as the base Fock.
    // MolGW m_scf_loop.f90:181: at each QSGW step we restore to base then
    // add the new H_QS correction, so the total is:
    //   H_new = base_fock + (Vx - Vxc + Sigma_QS)_AO
    //         = (h_core + J + Vxc) + (Vx - Vxc + Sigma_QS)
    //         = h_core + J + Vx + Sigma_QS   (QSGW Hamiltonian)
    let base_fock: [MatrixUpper<f64>; 2] = [
        scf_data.hamiltonian[0].clone(),
        scf_data.hamiltonian[1].clone(),
    ];
    let num_basis  = scf_data.mol.num_basis;
    // For RHF/RKS (spin_channel==1) the screened interaction gets a factor of
    // 2 to account for both spin channels (MolGW nspin factor).
    let spin_factor = if spin_channel == 1 { 2.0_f64 } else { 1.0_f64 };

    // Current QP energies: start from KS eigenvalues.
    let mut qp: Vec<f64> = scf_data.eigenvalues[0].clone();

    println!("QSGW: starting outer loop (max_iter={}, etol={:.2e}, mix={:.2})",
             max_iter, etol, mix);

    for iter in 0..max_iter {

        // ------------------------------------------------------------
        // 1. Occupation parameters from the current SCF state
        // ------------------------------------------------------------
        let (start_mo, num_state, occ_size, vir_size, _homo, _lumo) =
            ri_gw::get_occupation_parameters(scf_data, 'Y');

        println!("QSGW iter {}: occ={} vir={} num_state={}",
                 iter + 1, occ_size, vir_size, num_state);

        // ------------------------------------------------------------
        // 2. AO->MO transform for current MO coefficients
        //    ri_ov  : [num_auxbas, occ_size * vir_size]
        //    ri_full: [num_auxbas, num_state^2]
        // ------------------------------------------------------------
        let ri_ov   = ri_bse::get_submatrix(scf_data, 'O', 'V', 'Y');
        let ri_full = ri_bse::get_submatrix(scf_data, 'F', 'F', 'Y');

        // ------------------------------------------------------------
        // 3. RPA energy differences eps[ia] = e_a - e_i
        // ------------------------------------------------------------
        let eps_raw = build_rpa_energy_differences(&qp, occ_size, vir_size);
        // Guard: negative or zero differences would break sqrt; clamp to 1e-6 a.u.
        let eps: Vec<f64> = eps_raw.iter().map(|&e| e.max(1.0e-6)).collect();

        // ------------------------------------------------------------
        // 4. Build symmetrized RPA matrix M_tilde and diagonalize
        //    -> (omega2[s], Z[ia,s])  MolGW: diagonalize_rpa_matrix
        // ------------------------------------------------------------
        let m_tilde = build_rpa_symmetrized_matrix(&ri_ov, &eps, occ_size, vir_size);
        let (omega2, eigvec) = diagonalize_rpa_poles(m_tilde);
        // Physical RPA poles: sqrt of positive eigenvalues
        let omega: Vec<f64> = omega2.iter().map(|&w2| w2.max(0.0).sqrt()).collect();

        // ------------------------------------------------------------
        // 5-6. (X+Y) transition vectors and screened couplings
        //      w[P,s] = spin_factor * B_ov * (X+Y)
        //      MolGW: build_w_matrix_qs
        // ------------------------------------------------------------
        let xpy = build_xpy(&eigvec, &omega2, &eps, occ_size, vir_size);
        let w   = build_screened_couplings(&ri_ov, &xpy, spin_factor);

        // ------------------------------------------------------------
        // 7. Full off-diagonal correlation self-energy Sigma_c[p,q]
        //    MolGW: gw_selfenergy_qs
        // ------------------------------------------------------------
        let sigma_c = build_sigma_c_mo(
            &ri_full, &w, &qp, &omega, occ_size, num_state, eta,
        );

        // ------------------------------------------------------------
        // 8. Kotani Hermitianization (MolGW: apply_qs_approximation)
        //    Sigma_QS[p,q] = 0.5*(Sigma_c[p,q] + Sigma_c[q,p])
        // ------------------------------------------------------------
        let sigma_qs = symmetrize_sigma_qs(&sigma_c, num_state);

        // ------------------------------------------------------------
        // 9. Bare exchange Vx[p,q] and QS Hamiltonian in MO basis
        //    H_QS = Vx - diag(Vxc) + Sigma_QS
        //    MolGW: setup_exchange_m_vxc
        // ------------------------------------------------------------
        let vx       = build_vx_mo(&ri_full, occ_size, num_state);
        let h_qs_mo  = build_hqs_mo(
            &vx,
            &vxc_nn[start_mo..start_mo + num_state],
            &sigma_qs,
            num_state,
        );

        // ------------------------------------------------------------
        // 10. Back-transform H_QS to AO basis: C * H_QS_MO * C^T
        //     MolGW: m_scf_loop.f90 inject
        // ------------------------------------------------------------
        let c       = scf_data.eigenvectors[0].clone();
        let h_qs_ao = mo_to_ao(&h_qs_mo, &c);

        // ------------------------------------------------------------
        // 11. Restore base Fock, then add H_QS_AO correction
        // ------------------------------------------------------------
        for i_spin in 0..spin_channel {
            scf_data.hamiltonian[i_spin] = base_fock[i_spin].clone();
            add_full_to_upper(
                &mut scf_data.hamiltonian[i_spin],
                &h_qs_ao,
                num_basis,
            );
        }

        // ------------------------------------------------------------
        // 12. Re-diagonalize H_QSGW -> new QP eigenvalues + MO coeffs
        // ------------------------------------------------------------
        let (new_eigvecs, new_eigvals, _) =
            diagonalize_hamiltonian_outside(scf_data, mpi_operator);
        for i_spin in 0..spin_channel {
            scf_data.eigenvectors[i_spin] = new_eigvecs[i_spin].clone();
            scf_data.eigenvalues[i_spin]  = new_eigvals[i_spin].clone();
        }

        // ------------------------------------------------------------
        // 13. Linear mixing and convergence check
        // ------------------------------------------------------------
        let new_qp  = &new_eigvals[0];
        let delta: f64 = qp.iter().zip(new_qp.iter())
            .map(|(old, &nw)| (nw - old).abs())
            .fold(0.0_f64, f64::max);

        // mixed = mix * new + (1-mix) * old
        let mixed: Vec<f64> = new_qp.iter().zip(qp.iter())
            .map(|(&nw, &old)| mix * nw + (1.0 - mix) * old)
            .collect();
        qp = mixed;
        // Also push mixed energies back so next iteration uses them
        scf_data.eigenvalues[0] = qp.clone();

        println!("QSGW iter {}: max |\u{0394}\u{03b5}| = {:.6e} Ha", iter + 1, delta);

        if delta < etol {
            println!("QSGW converged in {} iterations.", iter + 1);
            break;
        }

        if iter + 1 == max_iter {
            println!("QSGW warning: did not converge within {} iterations.", max_iter);
        }
    }

    // Store converged QP energies for downstream use.
    scf_data.gwqp = (qp.clone(), qp.clone());
    let (start_mo, num_state, occ_size, vir_size, _homo, _lumo) =
            ri_gw::get_occupation_parameters(scf_data, 'Y');
    ri_gw::display::full_quasiparticles(&qp,occ_size);
    qp
}