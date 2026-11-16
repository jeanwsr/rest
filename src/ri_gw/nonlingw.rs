// ============================================================================
// Nonlinear FEAST G0W0 (full off-diagonal quasiparticle self-energy)
//
// Solves the Dyson/quasiparticle equation
//
//     T(λ) x = 0,      T(λ) = λ I - H_static - Σ_c(λ)
//
// as a *nonlinear* eigenvalue problem on a complex FEAST contour.
//
// The implementation follows Li & Polizzi, Phys. Rev. B 111, 045137 (2025):
// the FEAST subspace iteration only needs shifted linear solves at the contour
// nodes and a real-axis residual.  As in the paper, the correlation term may
// be dropped from the shifted systems (they then act as preconditioners); by
// default REST includes it when the extra dense work is affordable, because
// that makes the contour update exact.  In both cases the full Σ_c(λ) is
// applied when forming the residual and the projected problem.
//
// REST normally computes GW with contour deformation and only the diagonal
// Σ_c,nn.  Here we need the complete matrix Σ_c,pq(λ) at a complex/real energy.
// We therefore use the fully analytic (spectral) representation of W_c:
//
//     W_c(ω) = Σ_s w_s w_s^T [ 1/(ω-Ω_s) - 1/(ω+Ω_s) ],
//
// with (Ω_s, w_s) the RPA (direct-ring) poles and residues in the auxiliary
// RI basis.  The RPA eigenproblem is the standard dRPA one,
//
//     A_ia,jb = Δ_ia δ + 4 (ia|jb),      B_ia,jb = 4 (ia|jb),
//     H = Δ²/4 + √Δ C √Δ,               C = Q^T Q,
//
// diagonalised as H z_s = (Ω_s²/4) z_s.  The residue is
//
//     w_s = √(2/Ω_s) Q (√Δ ⊙ z_s).
//
// `Q` is the RI three-centre matrix `(P|ia)` in the Coulomb-metric auxiliary
// basis used by REST (`v_matrix = Q^T Q`).  The resulting self-energy is
//
//   Σ_c,pq(λ) = Σ_{n,s} U_s[p,n] U_s[q,n]
//                 [ f_n/(λ-ε_n+Ω_s) + (1-f_n)/(λ-ε_n-Ω_s) ],
//
//   U_s[p,n] = Σ_P (P|p n) w_s[P].
//
// This is the RI/libcint transcription of Eq. (11) of the paper.  The static
// part is H_static = h + J + Σ_x, i.e. the *Hartree-Fock* Fock matrix built
// from the reference density; for a HF reference it is diagonal in the
// canonical MO basis, while for a DFT reference its off-diagonal blocks carry
// (Σ_x - v_xc).  This is exactly the upfolding/head matrix of MolGW.
// ============================================================================

use num::Complex;
use rayon::prelude::*;
use std::io::Write;
use std::time::Instant;

use rest_tensors::matrix::matrix_blas_lapack::{
    _dgemm_full, _dgemv, _dgeev, _dsyevd,
};
use tensors::{MathMatrix, MatrixFull};

use crate::ctrl_io::quasiparticle_methods::QuasiParticle;
use crate::ri_bse::nonlinbse::{gmres, BlockDiagPrecond};
use crate::scf_io::SCF;
use crate::solvers::nlfeast::{
    nlfeast as nlfeast_core, qr_orthonormalise, ContourNode, NlepOperator,
};

use super::{compute_ri3mo_row, get_occupation_parameters};

// ============================================================================
// RPA spectral representation of W_c
// ============================================================================

/// Poles and residues of the direct-RPA screened interaction in the auxiliary
/// RI basis.
///
/// `w_s` is an auxiliary-basis vector such that
/// `W_c(ω) = Σ_s w_s w_s^T [1/(ω-Ω_s) - 1/(ω+Ω_s)]`.
pub struct RpaSpectrum {
    pub omega: Vec<f64>,
    pub w_s: Vec<Vec<f64>>,
}

/// Build the RPA poles/residues from the transition-space RI matrix `q`
/// (`[n_aux, n_trans]`) and the transition energies `delta` (`n_trans`).
///
/// This is a dense symmetric diagonalisation of the dRPA matrix.  It is the
/// price of the fully analytic self-energy; use `feast_gw_rpa_energy_max` or
/// `feast_gw_rpa_occ_max/vir_max` to keep `n_trans` moderate.
pub fn build_rpa_spectrum(q: &MatrixFull<f64>, delta: &[f64]) -> RpaSpectrum {
    let n_aux = q.size[0];
    let n_trans = q.size[1];
    if n_trans == 0 || n_aux == 0 {
        return RpaSpectrum { omega: vec![], w_s: vec![] };
    }

    // C = Q^T Q
    let mut c = MatrixFull::new([n_trans, n_trans], 0.0);
    _dgemm_full(q, 'T', q, 'N', &mut c, 1.0, 0.0);

    // H = Δ²/4 + √Δ C √Δ
    let mut h = c.clone();
    let sqrt_delta: Vec<f64> = delta.iter().map(|d| d.max(0.0).sqrt()).collect();
    h.data
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, value)| {
            let i = idx % n_trans;
            let j = idx / n_trans;
            *value *= sqrt_delta[i] * sqrt_delta[j];
        });
    for i in 0..n_trans {
        h[[i, i]] += 0.25 * delta[i] * delta[i];
    }

    let (v_opt, evals, _ndim) = _dsyevd(&h, 'V');
    let v = match v_opt {
        Some(v) => v,
        None => return RpaSpectrum { omega: vec![], w_s: vec![] },
    };

    let mut omega = Vec::with_capacity(n_trans);
    let mut w_s = Vec::with_capacity(n_trans);
    // Work vector reused for Q (√Δ ⊙ z_s).
    let mut scaled = vec![0.0_f64; n_trans];
    for s in 0..n_trans {
        // dRPA poles are Ω = 2√h.  Negative/zero h is unphysical and is
        // dropped (it can only arise from round-off for a positive C).
        if evals[s] <= 1.0e-14 {
            continue;
        }
        let om = 2.0 * evals[s].sqrt();
        if !om.is_finite() || om < 1.0e-10 {
            continue;
        }
        for i in 0..n_trans {
            scaled[i] = sqrt_delta[i] * v[[i, s]];
        }
        // w_s = √(2/Ω) Q (√Δ z_s)
        let mut w = vec![0.0_f64; n_aux];
        _dgemv(q, &scaled, &mut w, 'N', (2.0 / om).sqrt(), 0.0, 1, 1);
        omega.push(om);
        w_s.push(w);
    }

    RpaSpectrum { omega, w_s }
}

/// RPA transition data shared by `feast_gw` and `fully_analytic_gw`.
pub struct RpaSetup {
    pub rpa: RpaSpectrum,
    pub n_aux: usize,
    pub n_trans: usize,
}

/// Build the dRPA spectral representation of `W_c` from the SCF reference.
///
/// The transition space is filtered by `feast_gw_rpa_energy_max`,
/// `feast_gw_rpa_occ_max` and `feast_gw_rpa_vir_max` (the same knobs are used
/// by both nonlinear-FEAST and fully analytic GW).
pub fn build_rpa_setup(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    qp_w: &[f64],
) -> RpaSetup {
    let (start_mo, num_state, _occ_size, _vir_size, homo, lumo) =
        get_occupation_parameters(scf_data, 'Y');
    let ri_ov = crate::ri_bse::get_submatrix(scf_data, 'O', 'V', 'Y');
    let n_aux = ri_ov.size[0];
    let n_occ = homo + 1 - start_mo;
    let n_vir = num_state - lumo;
    assert_eq!(ri_ov.size[1], n_occ * n_vir);

    let occ_max = if qp_ctrl.feast_gw_rpa_occ_max == 0 {
        n_occ
    } else {
        qp_ctrl.feast_gw_rpa_occ_max.min(n_occ)
    };
    let vir_max = if qp_ctrl.feast_gw_rpa_vir_max == 0 {
        n_vir
    } else {
        qp_ctrl.feast_gw_rpa_vir_max.min(n_vir)
    };
    // A reduced occupied space keeps the highest occupied (valence) orbitals.
    let occ_start = n_occ - occ_max;
    let e_max = qp_ctrl.feast_gw_rpa_energy_max;
    let mut trans_cols: Vec<usize> = Vec::new();
    let mut deltas: Vec<f64> = Vec::new();
    for a in 0..vir_max {
        let ea = qp_w[lumo + a];
        for i in occ_start..n_occ {
            let ei = qp_w[start_mo + i];
            let d = ea - ei;
            if d <= 0.0 || d > e_max {
                continue;
            }
            trans_cols.push(i + a * n_occ);
            deltas.push(d);
        }
    }
    let n_trans = trans_cols.len();
    if n_trans == 0 {
        panic!("[rpa] no RPA transitions survived the cutoffs");
    }
    let max_transitions = qp_ctrl.feast_gw_rpa_max_transitions.max(1);
    if n_trans > max_transitions {
        panic!(
            "[rpa] the dense RPA problem has {} transitions; this would need ~{:.1} GB. \
             The configured limit is {} transitions.  Raise \
             feast_gw_rpa_max_transitions if enough memory is available, or reduce \
             the transition space with feast_gw_rpa_energy_max, \
             feast_gw_rpa_occ_max or feast_gw_rpa_vir_max.",
            n_trans,
            (n_trans as f64 * n_trans as f64 * 8.0 * 2.0) / 1.0e9,
            max_transitions
        );
    }
    if n_trans > 1500 {
        println!(
            "  [rpa] warning: {} RPA transitions; the dense eigenproblem uses ~{:.1} GB. \
             Consider feast_gw_rpa_energy_max / feast_gw_rpa_occ_max / feast_gw_rpa_vir_max.",
            n_trans,
            (n_trans as f64 * n_trans as f64 * 8.0 * 2.0) / 1.0e9
        );
    }
    let mut q_red = MatrixFull::new([n_aux, n_trans], 0.0);
    for (k, &src) in trans_cols.iter().enumerate() {
        for p in 0..n_aux {
            q_red[[p, k]] = ri_ov[[p, src]];
        }
    }
    RpaSetup {
        rpa: build_rpa_spectrum(&q_red, &deltas),
        n_aux,
        n_trans,
    }
}

// ============================================================================
// Static Hamiltonian  H_static = h + J + Σ_x
// ============================================================================

/// Build the full static GW Hamiltonian in the MO basis.
///
/// `h + J + Σ_x` is the Hartree-Fock Fock matrix built from the *reference*
/// density.  REST can evaluate it with the existing RI J/K machinery; we save
/// and restore `scf_data.hamiltonian` so that no downstream step sees the
/// temporary HF Fock matrix.  Only closed-shell RHF is supported here (the
/// driver panics for `spin_channel != 1`).
fn build_static_hamiltonian(
    scf_data: &mut SCF,
    mpi_operator: &Option<crate::mpi_io::MPIOperator>,
) -> MatrixFull<f64> {
    let n_basis = scf_data.mol.num_basis;
    let n_state = scf_data.mol.num_state;
    let c = scf_data.eigenvectors[0].clone();
    assert_eq!(c.size, [n_basis, n_state]);

    let saved_h = scf_data.hamiltonian[0].clone();
    let saved_roothaan = scf_data.roothaan_hamiltonian.clone();

    if scf_data.mol.ctrl.print_level > 0 {
        println!("[feast_gw] building the static HF Fock matrix h + J + Σ_x (AO → MO)");
    }
    scf_data.generate_hf_hamiltonian_ri_v(mpi_operator);
    let f_ao = scf_data.hamiltonian[0].to_matrixfull().unwrap();

    // Restore the SCF Fock matrix immediately.
    scf_data.hamiltonian[0] = saved_h;
    scf_data.roothaan_hamiltonian = saved_roothaan;

    // F_MO = C^T F_AO C
    let mut tmp = MatrixFull::new([n_state, n_basis], 0.0);
    _dgemm_full(&c, 'T', &f_ao, 'N', &mut tmp, 1.0, 0.0);
    let mut f_mo = MatrixFull::new([n_state, n_state], 0.0);
    _dgemm_full(&tmp, 'N', &c, 'N', &mut f_mo, 1.0, 0.0);

    // Keep the matrix exactly Hermitian (it should be to round-off already).
    for i in 0..n_state {
        for j in 0..i {
            let avg = 0.5 * (f_mo[[i, j]] + f_mo[[j, i]]);
            f_mo[[i, j]] = avg;
            f_mo[[j, i]] = avg;
        }
    }
    f_mo
}

// ============================================================================
// Self-energy operator
// ============================================================================

/// `U_s[p,n]` stored as `[n_g, n_active]` for all RPA poles.
///
/// This layout makes the self-energy matvec two BLAS `dgemv` calls per pole:
/// `y_n = Σ_p U[n,p] x_p`, then `out_p = Σ_n U[n,p] d_n y_n`.
struct SelfEnergyData {
    u_s: Vec<MatrixFull<f64>>,
    g_energies: Vec<f64>,
    g_occupied: Vec<bool>,
    omega: Vec<f64>,
    n_active: usize,
    eta: f64,
    /// Diagnostic: keep only the diagonal of Σ_c (and, in the driver, of the
    /// static Hamiltonian).  This turns the nonlinear matrix problem into a
    /// set of independent scalar equations.
    diagonal_only: bool,
}

impl SelfEnergyData {
    /// Real-axis correlation self-energy matvec at `lambda`.
    fn sigma_matvec_real(&self, lambda: f64, x: &[f64]) -> Vec<f64> {
        let n_g = self.g_energies.len();
        let n_active = self.n_active;
        let mut out = vec![0.0_f64; n_active];
        let mut y = vec![0.0_f64; n_g];
        let mut z = vec![0.0_f64; n_active];

        if self.diagonal_only {
            for p in 0..n_active {
                let mut diag = 0.0_f64;
                for (s, u) in self.u_s.iter().enumerate() {
                    let om = self.omega[s];
                    for n in 0..n_g {
                        let shift = if self.g_occupied[n] { om } else { -om };
                        let denom = lambda - self.g_energies[n] + shift;
                        let d = if self.eta > 0.0 {
                            denom / (denom * denom + self.eta * self.eta)
                        } else if denom.abs() < 1.0e-14 {
                            0.0
                        } else {
                            1.0 / denom
                        };
                        let val = u[[n, p]];
                        diag += val * val * d;
                    }
                }
                out[p] = diag * x[p];
            }
            return out;
        }

        for (s, u) in self.u_s.iter().enumerate() {
            let om = self.omega[s];
            // y = U_s x
            _dgemv(u, x, &mut y, 'N', 1.0, 0.0, 1, 1);
            for n in 0..n_g {
                let shift = if self.g_occupied[n] { om } else { -om };
                let denom = lambda - self.g_energies[n] + shift;
                // Real part of 1/(denom - iη).  With η = 0 this is the
                // principal-value denominator used by the paper.
                let d = if self.eta > 0.0 {
                    denom / (denom * denom + self.eta * self.eta)
                } else if denom.abs() < 1.0e-14 {
                    0.0
                } else {
                    1.0 / denom
                };
                y[n] *= d;
            }
            // z = U_s^T y
            _dgemv(u, &y, &mut z, 'T', 1.0, 0.0, 1, 1);
            for p in 0..n_active {
                out[p] += z[p];
            }
        }
        out
    }

    /// Derivative of the real-axis correlation self-energy matvec,
    /// `d/dlambda [Sigma_c(lambda) x]`, used for the quasiparticle spectral
    /// weight.  The denominators are differentiated analytically so the result
    /// is stable even when `eta` is small.
    fn sigma_derivative_matvec(&self, lambda: f64, x: &[f64]) -> Vec<f64> {
        let n_g = self.g_energies.len();
        let n_active = self.n_active;
        let mut out = vec![0.0_f64; n_active];
        let mut y = vec![0.0_f64; n_g];
        let mut z = vec![0.0_f64; n_active];

        if self.diagonal_only {
            for p in 0..n_active {
                let mut diag = 0.0_f64;
                for (s, u) in self.u_s.iter().enumerate() {
                    let om = self.omega[s];
                    for n in 0..n_g {
                        let shift = if self.g_occupied[n] { om } else { -om };
                        let denom = lambda - self.g_energies[n] + shift;
                        let dd = if self.eta > 0.0 {
                            let den2 = denom * denom + self.eta * self.eta;
                            (self.eta * self.eta - denom * denom) / (den2 * den2)
                        } else if denom.abs() < 1.0e-14 {
                            0.0
                        } else {
                            -1.0 / (denom * denom)
                        };
                        let val = u[[n, p]];
                        diag += val * val * dd;
                    }
                }
                out[p] = diag * x[p];
            }
            return out;
        }

        for (s, u) in self.u_s.iter().enumerate() {
            let om = self.omega[s];
            _dgemv(u, x, &mut y, 'N', 1.0, 0.0, 1, 1);
            for n in 0..n_g {
                let shift = if self.g_occupied[n] { om } else { -om };
                let denom = lambda - self.g_energies[n] + shift;
                let dd = if self.eta > 0.0 {
                    let den2 = denom * denom + self.eta * self.eta;
                    (self.eta * self.eta - denom * denom) / (den2 * den2)
                } else if denom.abs() < 1.0e-14 {
                    0.0
                } else {
                    -1.0 / (denom * denom)
                };
                y[n] *= dd;
            }
            _dgemv(u, &y, &mut z, 'T', 1.0, 0.0, 1, 1);
            for p in 0..n_active {
                out[p] += z[p];
            }
        }
        out
    }

    /// Quasiparticle spectral weight (residue) for a normalized Ritz vector:
    ///
    ///   Z = 1 / (1 - x^T dSigma_c/dlambda x).
    ///
    /// A positive, non-negligible Z is one of the indicators of a physical
    /// quasiparticle pole; a negative or near-zero Z indicates a satellite or
    /// a resonance-like solution.
    fn spectral_weight(&self, lambda: f64, x: &[f64]) -> f64 {
        let dsig = self.sigma_derivative_matvec(lambda, x);
        let xdsig: f64 = x.iter().zip(dsig.iter()).map(|(a, b)| a * b).sum();
        let denom = 1.0 - xdsig;
        if denom.abs() < 1.0e-12 {
            f64::NAN
        } else {
            1.0 / denom
        }
    }

    /// Full complex correlation self-energy matrix at a complex shift.
    ///
    /// Used only when the shifted FEAST systems include `Σ_c` (exact contour
    /// update).  The denominator is evaluated with the physical `±iη`
    /// prescription written as a small negative imaginary part; on the
    /// contour this is far from the branch cut and `eta = 0` is normally used.
    fn sigma_matrix_complex(&self, z: Complex<f64>) -> MatrixFull<Complex<f64>> {
        let n_g = self.g_energies.len();
        let n_active = self.n_active;
        let mut out = MatrixFull::new([n_active, n_active], Complex::new(0.0, 0.0));
        let mut v_re = MatrixFull::new([n_g, n_active], 0.0);
        let mut v_im = MatrixFull::new([n_g, n_active], 0.0);
        let mut sig_re = MatrixFull::new([n_active, n_active], 0.0);
        let mut sig_im = MatrixFull::new([n_active, n_active], 0.0);

        if self.diagonal_only {
            for p in 0..n_active {
                let mut re = 0.0_f64;
                let mut im = 0.0_f64;
                for (s, u) in self.u_s.iter().enumerate() {
                    let om = self.omega[s];
                    for k in 0..n_g {
                        let shift = if self.g_occupied[k] { om } else { -om };
                        let denom = z - Complex::new(self.g_energies[k] - shift, self.eta);
                        let inv = Complex::new(1.0, 0.0) / denom;
                        let val = u[[k, p]];
                        re += val * val * inv.re;
                        im += val * val * inv.im;
                    }
                }
                out[[p, p]] = Complex::new(re, im);
            }
            return out;
        }

        for (s, u) in self.u_s.iter().enumerate() {
            let om = self.omega[s];
            for k in 0..n_g {
                let shift = if self.g_occupied[k] { om } else { -om };
                // Match the real-axis convention: occupied -> +Ω, virtual ->
                // -Ω.  The ±iη branch is represented by `eta`.
                let denom = z - Complex::new(self.g_energies[k] - shift, self.eta);
                let inv = Complex::new(1.0, 0.0) / denom;
                for p in 0..n_active {
                    let uval = u[[k, p]];
                    v_re[[k, p]] = uval * inv.re;
                    v_im[[k, p]] = uval * inv.im;
                }
            }
            // Σ += U diag(d) U^T for the real and imaginary parts.
            _dgemm_full(u, 'T', &v_re, 'N', &mut sig_re, 1.0, 1.0);
            _dgemm_full(u, 'T', &v_im, 'N', &mut sig_im, 1.0, 1.0);
        }
        for j in 0..n_active {
            for i in 0..n_active {
                out[[i, j]] = Complex::new(sig_re[[i, j]], sig_im[[i, j]]);
            }
        }
        out
    }
}

// ============================================================================
// Dense complex inverse used for the shifted (static) systems
// ============================================================================

/// Cached inverse of `T(z_j)` on every FEAST node.
///
/// Depending on `feast_gw_shift_correlation`, `T(z_j)` is either the static
/// `z_j I - H_static` (paper, Algorithm 1) or the exact
/// `z_j I - H_static - Σ_c(z_j)`.  Either way it is a fixed dense matrix per
/// node, so it is factorised/inverted once and reused for every RHS.
struct DenseShiftSolvers {
    inverse: Vec<MatrixFull<Complex<f64>>>,
    n: usize,
}

impl DenseShiftSolvers {
    fn build(
        h_static: &MatrixFull<f64>,
        nodes: &[ContourNode],
        sigma: Option<&SelfEnergyData>,
    ) -> Self {
        let n = h_static.size[0];
        let mut inverse = Vec::with_capacity(nodes.len());
        for node in nodes {
            let z = Complex::new(node.z_re, node.z_im);
            let corr = sigma.map(|s| s.sigma_matrix_complex(z));
            let mut a = MatrixFull::<Complex<f64>>::new([n, n], Complex::new(0.0, 0.0));
            for j in 0..n {
                for i in 0..n {
                    a[[i, j]] = -Complex::new(h_static[[i, j]], 0.0);
                    if let Some(c) = &corr {
                        a[[i, j]] -= c[[i, j]];
                    }
                }
                a[[j, j]] += z;
            }
            let ipiv = a.lapack_zgetrf();
            a.lapack_zgetri(&ipiv);
            inverse.push(a);
        }
        DenseShiftSolvers { inverse, n }
    }

    fn solve(&self, node_index: usize, rhs: &[f64]) -> Vec<f64> {
        let n = self.n;
        let a = &self.inverse[node_index];
        let mut out = vec![0.0_f64; 2 * n];
        // rhs is the real-embedded [Re(b); Im(b)] vector.  The generic FEAST
        // core always supplies a real RHS, but support a general one.
        for i in 0..n {
            let mut re = 0.0;
            let mut im = 0.0;
            for j in 0..n {
                let aij = a[[i, j]];
                let br = rhs[j];
                let bi = rhs[n + j];
                re += aij.re * br - aij.im * bi;
                im += aij.re * bi + aij.im * br;
            }
            out[i] = re;
            out[n + i] = im;
        }
        out
    }

    /// Apply the cached inverse to an entire block of real RHS columns at
    /// once.  This turns the contour update from O(m0) matrix-vector products
    /// per node into a single BLAS-3 complex GEMM, which is substantially
    /// faster for the rather small m0 used by FEAST.
    fn solve_block(&self, node_index: usize, rhs: &MatrixFull<f64>) -> MatrixFull<Complex<f64>> {
        let n = self.n;
        let m = rhs.size[1];
        let mut b = MatrixFull::<Complex<f64>>::new([n, m], Complex::new(0.0, 0.0));
        for i in 0..n {
            for k in 0..m {
                b[[i, k]] = Complex::new(rhs[[i, k]], 0.0);
            }
        }
        let mut u = MatrixFull::<Complex<f64>>::new([n, m], Complex::new(0.0, 0.0));
        u.lapack_zgemm(
            &self.inverse[node_index],
            &b,
            'N',
            'N',
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
        );
        u
    }
}

// ============================================================================
// NLEP operator
// ============================================================================

struct GwNlepOperator<'a> {
    h_static: &'a MatrixFull<f64>,
    sigma: &'a SelfEnergyData,
    n: usize,
    shift: ShiftSolver<'a>,
    projected_selfconsistent: bool,
    projected_max_inner: usize,
}

enum ShiftSolver<'a> {
    Direct(DenseShiftSolvers),
    Gmres {
        h_static: &'a MatrixFull<f64>,
        diag: Vec<f64>,
        restart: usize,
        max_it: usize,
        tol: f64,
    },
}

impl NlepOperator for GwNlepOperator<'_> {
    fn dim(&self) -> usize {
        self.n
    }

    fn projected_solve(
        &self,
        q: &MatrixFull<f64>,
        lambda_init: &[f64],
    ) -> (Vec<Complex<f64>>, MatrixFull<f64>) {
        let n = self.n;
        let m0 = q.size[1];
        assert_eq!(lambda_init.len(), m0);

        if !self.projected_selfconsistent {
            // Cheap linearisation: each column j uses its own λ_j and the
            // projected matrix is diagonalised once with dgeev.
            let mut aq = MatrixFull::new([n, m0], 0.0);
            let mut qj = vec![0.0_f64; n];
            for j in 0..m0 {
                let lam = lambda_init[j];
                for i in 0..n {
                    qj[i] = q[[i, j]];
                }
                let mut hq = vec![0.0_f64; n];
                _dgemv(self.h_static, &qj, &mut hq, 'N', 1.0, 0.0, 1, 1);
                let sig = self.sigma.sigma_matvec_real(lam, &qj);
                for i in 0..n {
                    aq[[i, j]] = hq[i] + sig[i];
                }
            }
            let mut m = MatrixFull::new([m0, m0], 0.0);
            _dgemm_full(q, 'T', &aq, 'N', &mut m, 1.0, 0.0);
            let (_dummy, wr, wi, _vl, vr, info) = _dgeev(&m, 'N', 'V');
            if info != 0 {
                eprintln!("[feast_gw] projected dgeev returned info={}", info);
            }
            let mut lambdas = Vec::with_capacity(m0);
            let mut ritz = MatrixFull::new([n, m0], 0.0);
            for j in 0..m0 {
                lambdas.push(Complex::new(wr[j], wi[j]));
                for i in 0..n {
                    let mut val = 0.0;
                    for k in 0..m0 {
                        val += q[[i, k]] * vr[[k, j]];
                    }
                    ritz[[i, j]] = val;
                }
            }
            return (lambdas, ritz);
        }

        // T(λ) = λI - A(λ) with A(λ) = H_static + Σ_c(λ).  The projected
        // problem is the nonlinear eigenvalue problem
        //
        //     M(λ) y = λ y,      M(λ) = Q^T A(λ) Q.
        //
        // We solve it by a small self-consistent Rayleigh–Ritz iteration:
        // diagonalise M(λ), pick the Ritz pair that continues the current
        // vector, and update λ.  This is more accurate than linearising M
        // with one (different) λ per column and is what makes the residual
        // of near-degenerate / strongly mixed states converge.
        //
        // H Q and Q^T H Q are λ-independent and are formed once.
        let mut hq = MatrixFull::new([n, m0], 0.0);
        {
            let mut col = vec![0.0_f64; n];
            for k in 0..m0 {
                for i in 0..n {
                    col[i] = q[[i, k]];
                }
                let mut hcol = vec![0.0_f64; n];
                _dgemv(self.h_static, &col, &mut hcol, 'N', 1.0, 0.0, 1, 1);
                for i in 0..n {
                    hq[[i, k]] = hcol[i];
                }
            }
        }
        let mut hq_proj = MatrixFull::new([m0, m0], 0.0);
        _dgemm_full(q, 'T', &hq, 'N', &mut hq_proj, 1.0, 0.0);

        let max_inner = self.projected_max_inner.max(1);
        let n_g = self.sigma.g_energies.len();
        let full_sigma = !self.sigma.diagonal_only;

        // Major optimization: precompute G_s = U_s^T Q once per projected
        // solve (Q is fixed inside this call), instead of applying
        // Sigma(lambda) to every Q column for every column/inner iteration.
        //
        //   Sigma(lambda) = sum_s U_s diag(d_s(lambda)) U_s^T
        //   Q^T Sigma(lambda) Q = sum_s G_s diag(d_s(lambda)) G_s^T
        //
        // The old code formed Sigma(lambda_j) q_k for every j and k, i.e.
        // O(m0^2) full self-energy matvecs per inner iteration.  Here the
        // expensive O(n_s n_g n_active m0) contraction is done once, and each
        // lambda evaluation only needs O(n_s n_g m0^2) BLAS-3 work.
        let g_s: Vec<MatrixFull<f64>> = if full_sigma {
            let mut gs = Vec::with_capacity(self.sigma.u_s.len());
            for u in &self.sigma.u_s {
                let mut g = MatrixFull::new([n_g, m0], 0.0);
                _dgemm_full(u, 'T', q, 'N', &mut g, 1.0, 0.0);
                gs.push(g);
            }
            gs
        } else {
            Vec::new()
        };

        let mut sigma_proj = MatrixFull::new([m0, m0], 0.0);
        let mut work = MatrixFull::new([n_g, m0], 0.0);
        let mut build_sigma_projected = |lam: f64, out: &mut MatrixFull<f64>| {
            for i in 0..m0 {
                for k in 0..m0 {
                    out[[i, k]] = 0.0;
                }
            }
            if full_sigma {
                for (s, g) in g_s.iter().enumerate() {
                    let om = self.sigma.omega[s];
                    for n in 0..n_g {
                        let shift = if self.sigma.g_occupied[n] { om } else { -om };
                        let denom = lam - self.sigma.g_energies[n] + shift;
                        let d = if self.sigma.eta > 0.0 {
                            denom / (denom * denom + self.sigma.eta * self.sigma.eta)
                        } else if denom.abs() < 1.0e-14 {
                            0.0
                        } else {
                            1.0 / denom
                        };
                        for k in 0..m0 {
                            work[[n, k]] = d * g[[n, k]];
                        }
                    }
                    _dgemm_full(g, 'T', &work, 'N', out, 1.0, 1.0);
                }
            } else {
                // Diagonal self-energy: Q^T diag(sigma_p) Q.  This preserves
                // the semantics of feast_gw_zero_offdiag.
                let mut diag = vec![0.0_f64; n];
                for (s, u) in self.sigma.u_s.iter().enumerate() {
                    let om = self.sigma.omega[s];
                    for k in 0..n_g {
                        let shift = if self.sigma.g_occupied[k] { om } else { -om };
                        let denom = lam - self.sigma.g_energies[k] + shift;
                        let d = if self.sigma.eta > 0.0 {
                            denom / (denom * denom + self.sigma.eta * self.sigma.eta)
                        } else if denom.abs() < 1.0e-14 {
                            0.0
                        } else {
                            1.0 / denom
                        };
                        for p in 0..n {
                            let val = u[[k, p]];
                            diag[p] += val * val * d;
                        }
                    }
                }
                for i in 0..n {
                    for k in 0..m0 {
                        work[[i, k]] = diag[i] * q[[i, k]];
                    }
                }
                _dgemm_full(q, 'T', &work, 'N', out, 1.0, 1.0);
            }
        };

        let mut lambdas = Vec::with_capacity(m0);
        let mut ritz = MatrixFull::new([n, m0], 0.0);

        for j in 0..m0 {
            let mut lam = lambda_init[j];
            // Start from the current subspace column; tracks the branch
            // through the inner iteration.
            let mut y: Vec<f64> = (0..m0)
                .map(|k| if k == j { 1.0 } else { 0.0 })
                .collect();
            for _inner in 0..max_inner {
                let mut m = hq_proj.clone();
                build_sigma_projected(lam, &mut sigma_proj);
                for i in 0..m0 {
                    for k in 0..m0 {
                        m[[i, k]] += sigma_proj[[i, k]];
                    }
                }
                let (v_opt, evals, _nd) = _dsyevd(&m, 'V');
                let v = match v_opt {
                    Some(v) => v,
                    None => break,
                };
                // Choose the eigenpair that best continues y.  For a
                // near-degenerate cluster this picks different vectors for
                // different columns; otherwise it reduces to the closest
                // eigenvalue.
                let mut best_k = 0;
                let mut best_score = f64::NEG_INFINITY;
                for k in 0..m0 {
                    let mut overlap = 0.0_f64;
                    for i in 0..m0 {
                        overlap += v[[i, k]] * y[i];
                    }
                    let dist = (evals[k] - lam).abs();
                    let score = overlap.abs() - 0.25 * dist;
                    if score > best_score {
                        best_score = score;
                        best_k = k;
                    }
                }
                let lam_new = evals[best_k];
                let mut y_new = vec![0.0_f64; m0];
                for i in 0..m0 {
                    y_new[i] = v[[i, best_k]];
                }
                let dlam = (lam_new - lam).abs();
                let dy: f64 = (0..m0)
                    .map(|i| (y_new[i] - y[i]).abs())
                    .sum::<f64>();
                lam = lam_new;
                y = y_new;
                if dlam < 1.0e-11 && dy < 1.0e-9 {
                    break;
                }
            }
            lambdas.push(Complex::new(lam, 0.0));
            for i in 0..n {
                let mut val = 0.0;
                for k in 0..m0 {
                    val += q[[i, k]] * y[k];
                }
                ritz[[i, j]] = val;
            }
        }
        if std::env::var("REST_FEAST_GW_DEBUG").is_ok() {
            eprintln!(
                "  [feast_gw debug] projected Ritz values: {}",
                lambdas
                    .iter()
                    .map(|l| format!("{:.6}", l.re))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        (lambdas, ritz)
    }

    fn t_real(&self, lambda: f64, x: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut hx = vec![0.0_f64; n];
        _dgemv(self.h_static, x, &mut hx, 'N', 1.0, 0.0, 1, 1);
        let sig = self.sigma.sigma_matvec_real(lambda, x);
        (0..n)
            .map(|i| lambda * x[i] - hx[i] - sig[i])
            .collect()
    }

    fn solve_shifted(&self, node_index: usize, z: Complex<f64>, rhs: &[f64]) -> Vec<f64> {
        match &self.shift {
            ShiftSolver::Direct(d) => d.solve(node_index, rhs),
            ShiftSolver::Gmres {
                h_static,
                diag,
                restart,
                max_it,
                tol,
            } => {
                let n = self.n;
                let z_re = z.re;
                let z_im = z.im;
                let h = *h_static;
                let matvec = |v: &[f64]| -> Vec<f64> {
                    let vr = &v[0..n];
                    let vi = &v[n..2 * n];
                    let mut hvr = vec![0.0_f64; n];
                    let mut hvi = vec![0.0_f64; n];
                    _dgemv(h, vr, &mut hvr, 'N', 1.0, 0.0, 1, 1);
                    _dgemv(h, vi, &mut hvi, 'N', 1.0, 0.0, 1, 1);
                    let mut out = vec![0.0_f64; 2 * n];
                    for i in 0..n {
                        out[i] = z_re * vr[i] - z_im * vi[i] - hvr[i];
                        out[n + i] = z_re * vi[i] + z_im * vr[i] - hvi[i];
                    }
                    out
                };
                let precond = BlockDiagPrecond::from_re_im(diag, &vec![-z_im; n]);
                gmres(matvec, rhs, *restart, *max_it, *tol, false, Some(&precond))
            }
        }
    }


}

// ============================================================================
// Main entry point
// ============================================================================

/// Top-level driver for `gw_or_bse = "feast_gw"`.
pub fn feast_gw_main(
    scf_data: &mut SCF,
    qp_ctrl: &QuasiParticle,
    mpi_operator: &Option<crate::mpi_io::MPIOperator>,
) {
    let start = Instant::now();
    if scf_data.mol.spin_channel != 1 {
        panic!(
            "[feast_gw] the nonlinear-FEAST GW driver currently supports only \
             closed-shell (spin-restricted) references."
        );
    }

    if scf_data.rimatr.is_none() && scf_data.ri3fn.is_none() {
        panic!(
            "[feast_gw] requires RI three-centre integrals (eri_type = \"ri-v\" and a \
             valid auxbas_path); neither rimatr nor ri3fn is available."
        );
    }

    super::initialize_qp_g_w(scf_data);

    let sem_mode = qp_ctrl.feast_gw_selfenergy_mode.trim().to_lowercase();
    if !matches!(sem_mode.as_str(), "analytic" | "spectral" | "rpa") {
        panic!(
            "[feast_gw] unsupported feast_gw_selfenergy_mode = \"{}\"; \
             the implemented fully analytic mode is \"analytic\".",
            qp_ctrl.feast_gw_selfenergy_mode
        );
    }

    let (start_mo, num_state, occ_size, vir_size, homo, lumo) =
        get_occupation_parameters(scf_data, 'Y');
    if occ_size == 0 || vir_size == 0 {
        panic!("[feast_gw] the occupied or virtual space is empty");
    }
    let n_state = num_state - start_mo;

    println!("========================================");
    println!("  Nonlinear FEAST G0W0 (full off-diagonal Σ, analytic W_c)");
    println!("========================================");
    println!(
        "  MO window: start={}, num_state={}, occ={}, vir={}, homo={}, lumo={}",
        start_mo, num_state, occ_size, vir_size, homo, lumo
    );

    let qp_g = scf_data.gwqp.0.clone();
    let qp_w = scf_data.gwqp.1.clone();

    // ── Static Hamiltonian ──────────────────────────────────────────────
    let t_static = Instant::now();
    let h_full = build_static_hamiltonian(scf_data, mpi_operator);
    println!("[feast_gw] static Hamiltonian built in {:?}", t_static.elapsed());

    let centre = qp_ctrl.nlfeast_centre;
    let radius = qp_ctrl.nlfeast_radius;
    if !(radius > 0.0) {
        panic!("[feast_gw] nlfeast_radius must be positive");
    }
    let margin = qp_ctrl.feast_gw_window_margin.abs();
    let all_states = qp_ctrl.feast_gw_window_margin < 0.0;

    // ── Active window: states whose static diagonal lies near the contour ─
    let lo = centre - radius - margin;
    let hi = centre + radius + margin;
    let mut window: Vec<usize> = if all_states {
        (start_mo..num_state).collect()
    } else {
        (start_mo..num_state)
            .filter(|&p| h_full[[p, p]] >= lo && h_full[[p, p]] <= hi)
            .collect()
    };
    if window.is_empty() {
        // Fall back to the single closest state.
        let mut best = start_mo;
        let mut best_d = f64::INFINITY;
        for p in start_mo..num_state {
            let d = (h_full[[p, p]] - centre).abs();
            if d < best_d {
                best_d = d;
                best = p;
            }
        }
        window.push(best);
    }
    window.sort_unstable();
    let n_active = window.len();
    if all_states {
        println!(
            "  Active window: all {} states (feast_gw_window_margin < 0; centre={:.4}, radius={:.4})",
            n_active, centre, radius
        );
    } else {
        println!(
            "  Active window: {} states in [{:.4}, {:.4}] Ha (centre={:.4}, radius={:.4}, margin={:.4})",
            n_active, lo, hi, centre, radius, margin
        );
    }

    // Keep only the active-window block of H_static.
    let mut h_static = MatrixFull::new([n_active, n_active], 0.0);
    for (a, &p) in window.iter().enumerate() {
        for (b, &q) in window.iter().enumerate() {
            h_static[[a, b]] = h_full[[p, q]];
        }
    }
    let zero_offdiag = qp_ctrl.feast_gw_zero_offdiag;
    if zero_offdiag {
        for a in 0..n_active {
            for b in 0..n_active {
                if a != b {
                    h_static[[a, b]] = 0.0;
                }
            }
        }
        println!(
            "  [diagnostic] feast_gw_zero_offdiag = true: static and correlation \
             self-energy off-diagonal elements are set to zero"
        );
    }

    // ── G states entering the self-energy sum ───────────────────────────
    let mut g_states: Vec<usize> = if qp_ctrl.feast_gw_selfenergy_state_range == 0 {
        (start_mo..num_state).collect()
    } else {
        let lo_win = *window.first().unwrap();
        let hi_win = *window.last().unwrap();
        let r = qp_ctrl.feast_gw_selfenergy_state_range;
        (start_mo..num_state)
            .filter(|&p| p + r >= lo_win && p <= hi_win + r)
            .collect()
    };
    if g_states.is_empty() {
        g_states = (start_mo..num_state).collect();
    }
    println!("  Green's-function states: {} (of {})", g_states.len(), n_state);

    // ── RPA transition space and spectral W_c ───────────────────────────
    let t_rpa = Instant::now();
    let rpa_setup = build_rpa_setup(scf_data, qp_ctrl, &qp_w);
    let rpa = rpa_setup.rpa;
    let n_aux = rpa_setup.n_aux;
    let n_trans = rpa_setup.n_trans;
    println!(
        "[feast_gw] RPA: {} transitions -> {} poles, n_aux={}, built in {:?}",
        n_trans,
        rpa.omega.len(),
        n_aux,
        t_rpa.elapsed()
    );

    // ── U_s[p,n] = Σ_P (P|p n) w_s[P] ──────────────────────────────────
    let t_u = Instant::now();
    let n_g = g_states.len();
    let mut u_s: Vec<MatrixFull<f64>> = rpa
        .w_s
        .iter()
        .map(|_| MatrixFull::new([n_g, n_active], 0.0))
        .collect();
    let mut w_mat = MatrixFull::new([n_aux, rpa.omega.len()], 0.0);
    for (s, w) in rpa.w_s.iter().enumerate() {
        for p in 0..n_aux {
            w_mat[[p, s]] = w[p];
        }
    }
    // For each active orbital p, contract its RI row with every residue at
    // once: U_all = W^T · R_p  ([n_poles, n_g]).
    let mut u_all = MatrixFull::new([rpa.omega.len(), n_g], 0.0);
    for (pa, &p_global) in window.iter().enumerate() {
        let rp = compute_ri3mo_row(scf_data, p_global - start_mo);
        // R_p is [n_aux, n_state] with column k = global state start_mo+k.
        // Extract only the requested G states.
        let mut r_g = MatrixFull::new([n_aux, n_g], 0.0);
        for (k, &g) in g_states.iter().enumerate() {
            let col = g - start_mo;
            let src = col * n_aux;
            let dst = k * n_aux;
            r_g.data[dst..dst + n_aux].copy_from_slice(&rp.data[src..src + n_aux]);
        }
        _dgemm_full(&w_mat, 'T', &r_g, 'N', &mut u_all, 1.0, 0.0);
        for s in 0..rpa.omega.len() {
            for k in 0..n_g {
                u_s[s][[k, pa]] = u_all[[s, k]];
            }
        }
    }
    println!("[feast_gw] self-energy integrals built in {:?}", t_u.elapsed());

    let sigma = SelfEnergyData {
        u_s,
        g_energies: g_states.iter().map(|&g| qp_g[g]).collect(),
        g_occupied: g_states.iter().map(|&g| g <= homo).collect(),
        omega: rpa.omega,
        n_active,
        eta: qp_ctrl.feast_gw_eta.abs(),
        diagonal_only: zero_offdiag,
    };

    // Optional element-by-element dump for comparison with MolGW.
    if !qp_ctrl.feast_gw_dump_sigma.is_empty() {
        let omega = qp_ctrl.feast_gw_dump_omega;
        let eta = qp_ctrl.feast_gw_eta.abs();
        // `sigma_matrix_complex` itself adds the physical -iη; pass a real
        // frequency here so the imaginary part is not cancelled.
        let z = Complex::new(omega, 0.0);
        let sig = sigma.sigma_matrix_complex(z);
        let path = qp_ctrl.feast_gw_dump_sigma.clone();
        let mut f = std::fs::File::create(&path)
            .unwrap_or_else(|e| panic!("[feast_gw] cannot create {}: {}", path, e));
        writeln!(f, "# REST_SIGMA_C").unwrap();
        writeln!(f, "# omega_ha {:.16e}", omega).unwrap();
        writeln!(f, "# eta_ha {:.16e}", eta).unwrap();
        writeln!(f, "# n_basis {}", scf_data.mol.num_basis).unwrap();
        writeln!(f, "# n_state {}", num_state).unwrap();
        writeln!(f, "# start_mo {}", start_mo).unwrap();
        writeln!(f, "# homo {}", homo).unwrap();
        writeln!(f, "# n_active {}", n_active).unwrap();
        writeln!(f, "# mo_indices {}", window.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")).unwrap();
        writeln!(f, "# energies").unwrap();
        for n in 0..num_state {
            let occ = if n <= homo { 1.0 } else { 0.0 };
            writeln!(f, "{} {:.16e} {:.1}", n, qp_g[n], occ).unwrap();
        }
        writeln!(f, "# mo_coefficients").unwrap();
        let c = &scf_data.eigenvectors[0];
        for mu in 0..c.size[0] {
            for n in 0..c.size[1] {
                writeln!(f, "{} {} {:.16e}", mu, n, c[[mu, n]]).unwrap();
            }
        }
        writeln!(f, "# h_static").unwrap();
        for (a, &p) in window.iter().enumerate() {
            for (b, &q) in window.iter().enumerate() {
                writeln!(f, "{} {} {:.16e}", p, q, h_static[[a, b]]).unwrap();
            }
        }
        writeln!(f, "# sigma_c").unwrap();
        for (a, &p) in window.iter().enumerate() {
            for (b, &q) in window.iter().enumerate() {
                let v = sig[[a, b]];
                writeln!(f, "{} {} {:.16e} {:.16e}", p, q, v.re, v.im).unwrap();
            }
        }
        println!(
            "[feast_gw] dumped full Σ_c(ω={:.9} Ha, η={:.3e}) to {}",
            omega, eta, path
        );
        return;
    }

    // Optional check that the fully analytic Σ_c,nn agrees with the
    // conventional contour-deformation Σ_c,nn at the same (real) energy.
    // This isolates the RPA spectral representation from the nonlinear solve.
    if qp_ctrl.feast_gw_validate_diagonal {
        println!("\n  [diagnostic] Σ_c,nn at the static diagonal (analytic vs CD):");
        let mut gw_ctx = super::GwContext::default();
        let route = gw_ctx.route(scf_data);
        let ri_ov_route = route.ri_ov(scf_data);
        let w_c_at_freqs = route.generate_w_c(
            scf_data,
            ri_ov_route.as_ref(),
            num_state,
            occ_size,
            vir_size,
            20,
            false,
        );
        let vxc_nn = super::vxc_ao2mo(scf_data);
        let v_mat = super::exchange_matrix_from_scf(scf_data);
        let alpha = super::static_exchange_fraction(scf_data);
        for (pa, &p_global) in window.iter().enumerate() {
            let e = h_static[[pa, pa]];
            let mut exchange = 0.0;
            for i in 0..=homo {
                exchange -= v_mat[[p_global, i]];
            }
            let static_ref = qp_g[p_global] + (1.0 - alpha) * exchange - vxc_nn[p_global];
            println!(
                "    MO {:>4}: H_static={:+.9}  ε+(1-α)Σx−vxc={:+.9}  Δstatic={:+.2e}",
                p_global, e, static_ref, e - static_ref
            );
            let mut unit = vec![0.0_f64; n_active];
            unit[pa] = 1.0;
            let sig_analytic = sigma.sigma_matvec_real(e, &unit)[pa];

            let ri_row = route.ri_row(scf_data, p_global);
            let sig_cd = route.contour(
                scf_data,
                ri_ov_route.as_ref(),
                e,
                p_global,
                &qp_g,
                &qp_w,
                occ_size,
                vir_size,
                num_state,
                &ri_row,
                qp_ctrl.cdgw_res_tol,
                qp_ctrl.cdgw_eta,
            ) - super::calculate_imag(&w_c_at_freqs, num_state, p_global, e, &qp_g, &qp_w);
            println!(
                "    MO {:>4}: ε={:+.9} Ha  Σ_analytic={:+.9}  Σ_CD={:+.9}  Δ={:+.2e}",
                p_global,
                e,
                sig_analytic,
                sig_cd,
                sig_analytic - sig_cd
            );
        }
    }

    // ── Contour nodes and shifted solvers ────────────────────────────────
    let n_quad = qp_ctrl.nlfeast_n_quad.max(4);
    let nodes: Vec<ContourNode> = (0..n_quad)
        .map(|j| {
            let theta = 2.0 * std::f64::consts::PI * j as f64 / n_quad as f64;
            let (ct, st) = (theta.cos(), theta.sin());
            ContourNode {
                z_re: centre + radius * ct,
                z_im: radius * st,
                w_re: radius * ct / n_quad as f64,
                w_im: radius * st / n_quad as f64,
            }
        })
        .collect();

    let shift_mode = qp_ctrl.feast_gw_shift_solver.trim().to_lowercase();
    let corr_mode = qp_ctrl.feast_gw_shift_correlation.trim().to_lowercase();
    let corr_cost = (n_active as f64).powi(2)
        * (sigma.omega.len() as f64)
        * (sigma.g_energies.len() as f64)
        * (n_quad as f64);
    let mut use_direct = match shift_mode.as_str() {
        "direct" | "lu" => true,
        "gmres" | "iterative" => false,
        _ => n_active <= 350,
    };
    let mut include_corr = match corr_mode.as_str() {
        "on" | "true" | "yes" => true,
        "off" | "false" | "no" => false,
        // Auto: include Σ when the extra dense builds are a small fraction of
        // the RPA/self-energy work and the window is not too large.
        _ => use_direct && corr_cost <= 2.0e9,
    };
    if include_corr && !use_direct {
        println!(
            "[feast_gw] Σ is not included in the shifted systems because GMRES was requested; \
             set feast_gw_shift_solver = \"direct\" to use the exact contour update."
        );
        include_corr = false;
    }
    let shift = if use_direct {
        println!(
            "[feast_gw] shifted solver: direct complex LU ({} nodes), correlation in shifted systems: {}",
            n_quad, include_corr
        );
        let corr = if include_corr { Some(&sigma) } else { None };
        ShiftSolver::Direct(DenseShiftSolvers::build(&h_static, &nodes, corr))
    } else {
        println!("[feast_gw] shifted solver: real-embedded GMRES (static preconditioner)");
        let diag: Vec<f64> = (0..n_active).map(|i| h_static[[i, i]]).collect();
        ShiftSolver::Gmres {
            h_static: &h_static,
            diag,
            restart: qp_ctrl.nlfeast_gmres_restart.max(20),
            max_it: qp_ctrl.nlfeast_gmres_max_it.max(50),
            tol: qp_ctrl.nlfeast_gmres_tol.max(1.0e-12),
        }
    };

    // ── Initial subspace: eigenvectors of H_static nearest the contour ──
    let (h_evec, h_eval, _ndim) = _dsyevd(&h_static, 'V');
    let h_evec = h_evec.expect("static Hamiltonian diagonalisation failed");
    let mut order: Vec<usize> = (0..n_active).collect();
    order.sort_by(|&a, &b| {
        (h_eval[a] - centre)
            .abs()
            .partial_cmp(&(h_eval[b] - centre).abs())
            .unwrap()
    });
    let mut m0 = qp_ctrl.nlfeast_m0.min(n_active).max(1);
    // For a small active window, using the full window as the FEAST subspace
    // makes the nonlinear projected problem exact and converges in one
    // iteration.  For large windows this would be too expensive, so the
    // user-provided m0 is respected there.
    if n_active <= 48 && m0 < n_active && std::env::var("REST_FEAST_GW_NO_AUTO_M0").is_err() {
        println!(
            "  FEAST: expanding m0 from {} to the full active-window size {} (small window)",
            m0, n_active
        );
        m0 = n_active;
    }
    let mut q0 = MatrixFull::new([n_active, m0], 0.0);
    let mut lambda_init = vec![0.0_f64; m0];
    for (j, &idx) in order.iter().take(m0).enumerate() {
        lambda_init[j] = h_eval[idx];
        for i in 0..n_active {
            q0[[i, j]] = h_evec[[i, idx]];
        }
    }
    q0 = qr_orthonormalise(&q0);
    if q0.size[1] < m0 {
        m0 = q0.size[1];
        lambda_init.truncate(m0);
    }
    println!(
        "  FEAST: m0={}, n_quad={}, max_iter={}, tol={:.1e}",
        m0, n_quad, qp_ctrl.nlfeast_max_iter, qp_ctrl.nlfeast_tol
    );
    println!(
        "  Initial λ estimates (Ha): {}",
        lambda_init
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let op = GwNlepOperator {
        h_static: &h_static,
        sigma: &sigma,
        n: n_active,
        shift,
        projected_selfconsistent: qp_ctrl.feast_gw_projected_solver != "linearized",
        projected_max_inner: qp_ctrl.feast_gw_projected_max_inner.max(1),
    };
    let result = nlfeast_core(
        &op,
        q0,
        lambda_init,
        &nodes,
        centre,
        radius,
        qp_ctrl.nlfeast_max_iter.max(1),
        qp_ctrl.nlfeast_tol.max(1.0e-14),
    );

    println!("\n  NLFEAST iterations: {}", result.iterations);

    if result.n_found == 0 {
        println!("  No quasiparticle eigenvalues found inside the contour.");
        return;
    }

    println!("\n  Found {} quasiparticle state(s) inside [{:.4}, {:.4}] Ha:",
        result.n_found, centre - radius, centre + radius);
    println!("  #    MO     E_QP (Ha)        E_QP (eV)       ‖T(λ)x‖       Z      LUMO     HOMO");
    println!("  ───  ─────  ───────────────  ─────────────  ──────────  ────────  ───────  ───────");
    let mut sorted: Vec<usize> = (0..result.n_found).collect();
    sorted.sort_by(|&a, &b| {
        result.eigenvalues[a]
            .partial_cmp(&result.eigenvalues[b])
            .unwrap()
    });

    // Per-root diagnostics kept for the optional CSV dump.  Each entry is
    // (root, E_Ha, E_eV, residual, Z, dominant MO, w_LUMO, w_HOMO, composition).
    let mut root_rows: Vec<(usize, f64, f64, f64, f64, usize, f64, f64, Vec<(usize, f64)>)> =
        Vec::with_capacity(result.n_found);

    for &k in &sorted {
        // Ritz vector in the active-window KS-MO basis, normalized to unit norm.
        let xk_raw: Vec<f64> = (0..n_active).map(|p| result.eigenvectors[[p, k]]).collect();
        let norm2: f64 = xk_raw.iter().map(|v| v * v).sum();
        let inv_norm = if norm2 > 0.0 { 1.0 / norm2.sqrt() } else { 0.0 };
        let xk: Vec<f64> = xk_raw.iter().map(|v| v * inv_norm).collect();
        let mut comp: Vec<(usize, f64)> = (0..n_active)
            .map(|p| (window[p], xk[p] * xk[p]))
            .collect();
        comp.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let best_mo = comp.first().map(|(mo, _)| *mo).unwrap_or(0);
        let w_lumo = comp
            .iter()
            .find(|(mo, _)| *mo == lumo)
            .map(|(_, w)| *w)
            .unwrap_or(0.0);
        let w_homo = comp
            .iter()
            .find(|(mo, _)| *mo == homo)
            .map(|(_, w)| *w)
            .unwrap_or(0.0);
        let z = sigma.spectral_weight(result.eigenvalues[k], &xk);

        println!(
            "  {:>3}  {:>5}  {:>15.9}  {:>13.6}  {:>9.2e}  {:>8.3}  {:>7.3}  {:>7.3}",
            k,
            best_mo,
            result.eigenvalues[k],
            result.eigenvalues[k] * crate::constants::EV,
            result.residuals[k],
            z,
            w_lumo,
            w_homo
        );
        let top = comp
            .iter()
            .take(6)
            .map(|(mo, w)| format!("MO{}:{:.3}", mo, w))
            .collect::<Vec<_>>()
            .join(", ");
        println!("       KS composition: {}", top);
        if std::env::var("REST_FEAST_GW_DEBUG").is_ok() {
            eprintln!(
                "      [debug] top coefficients: {}",
                comp.iter()
                    .take(6)
                    .map(|(mo, w)| format!("MO{}:{:.3}", mo, w))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        root_rows.push((
            k,
            result.eigenvalues[k],
            result.eigenvalues[k] * crate::constants::EV,
            result.residuals[k],
            z,
            best_mo,
            w_lumo,
            w_homo,
            comp,
        ));
    }

    if !qp_ctrl.feast_gw_dump_roots.is_empty() {
        let path = qp_ctrl.feast_gw_dump_roots.clone();
        let mut f = std::fs::File::create(&path)
            .unwrap_or_else(|e| panic!("[feast_gw] cannot create {}: {}", path, e));
        writeln!(
            f,
            "root,energy_ha,energy_ev,residual,Z,dominant_mo,w_lumo,w_homo,composition"
        )
        .unwrap();
        for (k, e_ha, e_ev, res, z, best_mo, w_lumo, w_homo, comp) in &root_rows {
            let comp_str = comp
                .iter()
                .take(20)
                .map(|(mo, w)| format!("MO{}:{:.6}", mo, w))
                .collect::<Vec<_>>()
                .join(";");
            writeln!(
                f,
                "{},{:.16e},{:.10e},{:.6e},{:.8e},{},{:.8e},{:.8e},\"{}\"",
                k, e_ha, e_ev, res, z, best_mo, w_lumo, w_homo, comp_str
            )
            .unwrap();
        }
        println!("[feast_gw] wrote root diagnostics to {}", path);
    }

    // Store the quasiparticle energies back into the SCF record, keyed by the
    // dominant MO of each eigenvector.
    let mut new_qp = qp_g.clone();
    for &k in &sorted {
        let mut best_p = 0;
        let mut best_w = 0.0;
        for p in 0..n_active {
            let w = result.eigenvectors[[p, k]].abs();
            if w > best_w {
                best_w = w;
                best_p = p;
            }
        }
        new_qp[window[best_p]] = result.eigenvalues[k];
    }
    scf_data.gwqp.0 = new_qp;

    // Optional diagonal diagnostic against the conventional CD-GW equation.
    if qp_ctrl.feast_gw_validate_diagonal {
        println!("\n  [diagnostic] diagonal self-energy at the FEAST solution:");
        for &k in &sorted {
            let mut best_p = 0;
            let mut best_w = 0.0;
            for p in 0..n_active {
                let w = result.eigenvectors[[p, k]].abs();
                if w > best_w {
                    best_w = w;
                    best_p = p;
                }
            }
            let lam = result.eigenvalues[k];
            let mut e = vec![0.0_f64; n_active];
            e[best_p] = 1.0;
            let sig = sigma.sigma_matvec_real(lam, &e);
            let h_nn = h_static[[best_p, best_p]];
            println!(
                "    MO {:>4}: λ={:.9} Ha, H_nn={:.9} Ha, Σ_c,nn={:.9} Ha, λ-H_nn-Σ_nn={:.2e} (diagonal only; full residual above)",
                window[best_p],
                lam,
                h_nn,
                sig[best_p],
                lam - h_nn - sig[best_p]
            );
        }
    }

    println!("\n[feast_gw] total time: {:?}", start.elapsed());
}

#[cfg(test)]
mod tests {
    use super::*;
    use rest_tensors::matrix::matrix_blas_lapack::_dinverse_inplace;

    /// The spectral representation of `W_c` must reproduce the direct
    /// response/inverse-dielectric construction at real frequencies.
    #[test]
    fn rpa_spectrum_matches_direct_inverse() {
        let n_aux = 7;
        let n_trans = 5;
        let mut q = MatrixFull::new([n_aux, n_trans], 0.0);
        for i in 0..n_aux {
            for j in 0..n_trans {
                q[[i, j]] = ((i as f64 + 1.0) * 0.37 + (j as f64 + 1.0) * 0.71).sin()
                    + 0.2 * ((i as f64 + 2.0) * (j as f64 + 3.0)).cos();
            }
        }
        let delta = vec![0.5, 1.0, 1.5, 2.0, 3.0];
        let spec = build_rpa_spectrum(&q, &delta);
        assert!(!spec.omega.is_empty());

        let frequencies = [0.0, 0.3, 2.5, 10.0];
        for &w in &frequencies {
            // Reference: ε = I - Q diag(4Δ/(ω²-Δ²)) Q^T, W_c = ε⁻¹ - I.
            let mut resp = MatrixFull::new([n_aux, n_aux], 0.0);
            for j in 0..n_trans {
                let d = 4.0 * delta[j] / (w * w - delta[j] * delta[j]);
                for a in 0..n_aux {
                    for b in 0..n_aux {
                        resp[[a, b]] += q[[a, j]] * d * q[[b, j]];
                    }
                }
            }
            let mut eps = MatrixFull::new([n_aux, n_aux], 0.0);
            for a in 0..n_aux {
                for b in 0..n_aux {
                    eps[[a, b]] = -resp[[a, b]] + if a == b { 1.0 } else { 0.0 };
                }
            }
            assert!(_dinverse_inplace(&mut eps));
            for a in 0..n_aux {
                eps[[a, a]] -= 1.0;
            }

            let mut spec_w = MatrixFull::new([n_aux, n_aux], 0.0);
            for s in 0..spec.omega.len() {
                let f = 1.0 / (w - spec.omega[s]) - 1.0 / (w + spec.omega[s]);
                for a in 0..n_aux {
                    for b in 0..n_aux {
                        spec_w[[a, b]] += spec.w_s[s][a] * f * spec.w_s[s][b];
                    }
                }
            }
            for a in 0..n_aux {
                for b in 0..n_aux {
                    assert!(
                        (eps[[a, b]] - spec_w[[a, b]]).abs() < 1.0e-10,
                        "ω={w}, ({a},{b}): direct={} spectral={}",
                        eps[[a, b]],
                        spec_w[[a, b]]
                    );
                }
            }
        }
    }
}
