// ============================================================================
// dynamicbse.rs — Dynamical BSE for double excitations (number-conserving
// RPA-pair downfolding), solved with the nonlinear FEAST (NLFEAST) kernel.
//
// Reference
// ---------
// D. Sangalli, P. Romaniello, G. Onida, A. Marini,
//   "Double excitations in correlated systems: A many-body approach",
//   J. Chem. Phys. 134, 034115 (2011).  See docs/DynBSE_DoubleExcitations.pdf.
//
// Method
// ------
// The doubles space is folded into the singles (electron-hole) space, giving
// the nonlinear eigenvalue problem
//
//     T(ω) x = [ S + Ξ_d(ω) − ω·I ] x = 0 ,
//
// where S is the *static* TDA BSE Hamiltonian (screened W, applied through an
// implicit matrix-vector product — no dense n×n matrix is ever built) and
//
//     Ξ_d(ω) = Σ_{ν1≠ν2} C^RPA_{ν1ν2} (C^RPA_{ν1ν2})ᵀ / (ω − (Ω_{ν1}+Ω_{ν2}+2iη))
//                                                                        (Eq. 23)
//
// is the number-conserving (NC) dynamical kernel built from *RPA excitation
// pairs*.  The coupling vectors are (Eq. 22)
//
//     C^RPA_{(ij),ν1ν2} = ½ Σ_{(nq),(mp)} [ (v_{(in),(mp)}δ_{j,q}
//                            + v_{(mp),(jq)}δ_{i,n}) R_{ν1,(np)} R_{ν2,(mq)}
//                            + {ν1↔ν2} ] ,
//
// with R_{ν,eh} and Ω_ν the residues and poles of the RPA response χ^RPA
// (Eq. 21), obtained here from the time-dependent-Hartree TDA eigenproblem
// (direct Coulomb only, singlet factor 2 / triplet 0).
//
// The old implementation (dynamicbse_matvec.rs) instead dressed a
// frequency-dependent W(ω) from *bare* electron-hole pairs — the non-NC
// approximation the paper identifies as the source of spurious poles — and
// never formed the RPA basis or the coupling vectors.  It has been removed.
//
// NLFEAST interface
// -----------------
// `DynamicBseOperator` implements `solvers::nlfeast::NlepOperator`:
//   * `t_real`          — T(ω)x on the real axis (convergence residual),
//   * `projected_solve` — exact linearisation of the projected rational
//                         eigenproblem QᵀT(ω)Q y = 0,
//   * `solve_shifted`   — real-embedded GMRES for T(z)u = rhs.
// All heavy objects are implicit matvecs; only the RPA vectors and the
// (window-truncated) coupling vectors are stored.
// ============================================================================

use num::Complex;
use rest_tensors::matrix::matrix_blas_lapack::{_dgemm_scaled, _dgeev, _dsyevd};
use rest_tensors::matrix::{MathMatrix, MatrixFull};
use std::time::Instant;

use crate::constants::EV;
use crate::ctrl_io::quasiparticle_methods::QuasiParticle;
use crate::ri_gw::get_occupation_parameters;
use crate::scf_io::SCF;
use crate::solvers::davidson::{generate_initial_guess, tda_davidson_solver, DavidsonConfig};
use crate::solvers::nlfeast::{nlfeast, ContourNode, NlepOperator, NLFeastResult};

use super::matvec::coulomb_contribution;
use super::nonlinbse::{gmres, BlockDiagPrecond};
use super::{construct_coulomb, construct_energy_diag_for_a, construct_inverse_dielectric, BseMatvec};

/// RPA poles with energy above `omega_window_max + PAIR_BAND` cannot contribute
/// a pair pole inside (or just above) the search window and are discarded.
const PAIR_BAND: f64 = 0.15;
/// RPA dimensions up to this value are diagonalised densely (robust for many
/// roots); larger ones use the implicit Davidson solver.
const DENSE_RPA_LIMIT: usize = 600;
/// Upper bound on the number of RPA excitations solved for (memory/time guard).
const MAX_RPA_ROOTS: usize = 400;
/// Upper bound on the number of (grouped) kernel coupling vectors kept; only
/// the poles closest to the search window are retained.
const MAX_KERNEL_TERMS: usize = 600;

/// Residues and poles of the RPA response: `vectors[ν]` is `R_ν` (length n,
/// index `i + a·occ`), `omegas[ν] = Ω_ν`.
struct RpaData {
    omegas: Vec<f64>,
    vectors: Vec<Vec<f64>>,
}

/// One (grouped) pole of the dynamical kernel: a common energy denominator
/// `pole` and the coupling vectors that share it.
struct KernelTerm {
    pole: f64,
    vectors: Vec<Vec<f64>>,
}

// ============================================================================
// RPA (time-dependent Hartree, TDA) excitation solver
// ============================================================================

/// Implicit TDH-TDA matrix-vector product:
///   A^TDH = diag(ε_a − ε_i) + c·v ,  c = 2 (singlet) / 1 (R) / 0 (triplet).
fn tdh_matvec(
    ri_ov: &MatrixFull<f64>,
    diag: &[f64],
    coulomb_factor: f64,
    x: &[f64],
) -> Vec<f64> {
    let mut y: Vec<f64> = diag.iter().zip(x.iter()).map(|(d, xi)| d * xi).collect();
    if coulomb_factor.abs() > 1e-30 {
        let v = coulomb_contribution(ri_ov, &x.to_vec());
        for i in 0..y.len() {
            y[i] += coulomb_factor * v[i];
        }
    }
    y
}

/// Solve for the RPA excitations up to `omega_max` (plus a small margin).
///
/// Small problems are diagonalised densely (symmetric `dsyevd`); larger ones use
/// the implicit Davidson solver, doubling the requested root count until the
/// highest returned pole exceeds the target or the problem is exhausted.
fn solve_rpa_excitations(
    ri_ov: &MatrixFull<f64>,
    diag: &[f64],
    coulomb_factor: f64,
    omega_max: f64,
    occ_size: usize,
    vir_size: usize,
) -> RpaData {
    let n = occ_size * vir_size;
    if n == 0 {
        return RpaData { omegas: vec![], vectors: vec![] };
    }
    let target = omega_max + PAIR_BAND;

    let roots: Vec<(f64, Vec<f64>)> = if n <= DENSE_RPA_LIMIT {
        // Build the (symmetric) TDH-TDA matrix column by column and diagonalise.
        let mut a = MatrixFull::new([n, n], 0.0);
        for j in 0..n {
            let mut e = vec![0.0; n];
            e[j] = 1.0;
            let col = tdh_matvec(ri_ov, diag, coulomb_factor, &e);
            for i in 0..n {
                a[[i, j]] = col[i];
            }
        }
        let (v_opt, evals, _) = _dsyevd(&a, 'V');
        let v = v_opt.unwrap();
        (0..n)
            .map(|j| (evals[j], (0..n).map(|i| v[[i, j]]).collect::<Vec<f64>>()))
            .collect()
    } else {
        let mut nroots = 40.min(n);
        let mut result: Vec<(f64, Vec<f64>)> = Vec::new();
        loop {
            let initial = generate_initial_guess(&diag.to_vec(), nroots);
            let config = DavidsonConfig {
                max_subspace: (3 * nroots).max(200).min(n),
                add_dim: 4,
                restart_dim: (nroots / 2).max(8),
                max_iter: 300,
                tol: 1e-9,
                ..Default::default()
            };
            let r = tda_davidson_solver(
                |x: &Vec<f64>| tdh_matvec(ri_ov, diag, coulomb_factor, x),
                nroots,
                &diag.to_vec(),
                initial,
                &config,
            );
            let highest = r.last().map(|(e, _)| *e).unwrap_or(f64::NEG_INFINITY);
            result = r;
            if highest >= target || nroots >= n || nroots >= MAX_RPA_ROOTS {
                break;
            }
            nroots = (nroots * 2).min(n).min(MAX_RPA_ROOTS);
        }
        result
    };

    // Keep only poles that can participate in a pair pole near the window.
    let mut filtered: Vec<(f64, Vec<f64>)> = roots
        .into_iter()
        .filter(|(e, _)| *e <= target + 1e-10)
        .collect();
    filtered.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    RpaData {
        omegas: filtered.iter().map(|(e, _)| *e).collect(),
        vectors: filtered.into_iter().map(|(_, v)| v).collect(),
    }
}

// ============================================================================
// Coupling vectors C^RPA (Eq. 22)
// ============================================================================

/// Contract the oo-ov Coulomb tensor with `r1` to form
///   U1[i,m] = Σ_{n,p} v_{(in),(mp)} R1[n,p] .
///
/// `v_oo_ov` is `construct_coulomb(ri_oo, ri_ov)`, size [no², no·nv], with the
/// code's pair orderings: oo pair = i + n·occ, ov pair = m + p·occ.
fn contract_u1(
    v_oo_ov: &MatrixFull<f64>,
    r1: &[f64],
    occ_size: usize,
    vir_size: usize,
) -> Vec<f64> {
    let no = occ_size;
    let nv = vir_size;
    let mut u1 = vec![0.0; no * no]; // [i, m]
    for i in 0..no {
        for m in 0..no {
            let mut acc = 0.0;
            for n in 0..no {
                let row = i + n * no;
                for p in 0..nv {
                    acc += v_oo_ov[[row, m + p * no]] * r1[n + p * no];
                }
            }
            u1[i + m * no] = acc;
        }
    }
    u1
}

/// Contract the vv-ov Coulomb tensor with `r2` to form
///   G2[j,p] = Σ_{m,q} v_{(jq),(mp)} R2[m,q] .
///
/// `v_vv_ov` is `construct_coulomb(ri_vv, ri_ov)`, size [nv², no·nv], with the
/// code's pair orderings: vv pair = j + q·vir, ov pair = m + p·occ.
fn contract_g2(
    v_vv_ov: &MatrixFull<f64>,
    r2: &[f64],
    occ_size: usize,
    vir_size: usize,
) -> Vec<f64> {
    let no = occ_size;
    let nv = vir_size;
    let mut g2 = vec![0.0; nv * nv]; // [j, p], both virtual
    for j in 0..nv {
        for p in 0..nv {
            let mut acc = 0.0;
            for q in 0..nv {
                let row = j + q * nv;
                for m in 0..no {
                    acc += v_vv_ov[[row, m + p * no]] * r2[m + q * no];
                }
            }
            g2[j * nv + p] = acc;
        }
    }
    g2
}

/// The ordered contribution `A(ν1,ν2) + B(ν1,ν2)` of Eq. (22):
///   [U1(R1)·R2]_{(ij)} + [R1·G2(R2)ᵀ]_{(ij)} .
fn coupling_ordered(
    v_oo_ov: &MatrixFull<f64>,
    v_vv_ov: &MatrixFull<f64>,
    r1: &[f64],
    r2: &[f64],
    occ_size: usize,
    vir_size: usize,
) -> Vec<f64> {
    let no = occ_size;
    let nv = vir_size;
    let u1 = contract_u1(v_oo_ov, r1, no, nv); // [i, m]
    let g2 = contract_g2(v_vv_ov, r2, no, nv); // [j, p]
    let mut out = vec![0.0; no * nv]; // index i + j·occ
    for i in 0..no {
        for j in 0..nv {
            // Term A: Σ_m U1[i,m] R2[m,j]
            let mut a = 0.0;
            for m in 0..no {
                a += u1[i + m * no] * r2[m + j * no];
            }
            // Term B: Σ_p R1[i,p] G2[j,p]   (j, p both virtual)
            let mut b = 0.0;
            for p in 0..nv {
                b += r1[i + p * no] * g2[j * nv + p];
            }
            out[i + j * no] = a + b;
        }
    }
    out
}

/// Full symmetrised coupling vector of Eq. (22), including `{ν1↔ν2}`.
fn coupling_vector(
    v_oo_ov: &MatrixFull<f64>,
    v_vv_ov: &MatrixFull<f64>,
    r1: &[f64],
    r2: &[f64],
    occ_size: usize,
    vir_size: usize,
) -> Vec<f64> {
    let a = coupling_ordered(v_oo_ov, v_vv_ov, r1, r2, occ_size, vir_size);
    let b = coupling_ordered(v_oo_ov, v_vv_ov, r2, r1, occ_size, vir_size);
    a.iter().zip(b.iter()).map(|(x, y)| 0.5 * (x + y)).collect()
}

/// Build the grouped kernel terms for the RPA pairs whose pole
/// `Ω_{ν1}+Ω_{ν2}` lies within `[omega_min - PAIR_BAND, omega_max + PAIR_BAND]`.
///
/// Only the `MAX_KERNEL_TERMS` pairs whose poles are closest to the window are
/// kept; this bounds the size of the linearised projected problem.
fn build_kernel_terms(
    scf_data: &SCF,
    occ_size: usize,
    vir_size: usize,
    rpa: &RpaData,
    omega_min: f64,
    omega_max: f64,
) -> Vec<KernelTerm> {
    let lo = omega_min - PAIR_BAND;
    let hi = omega_max + PAIR_BAND;
    let window_centre = 0.5 * (omega_min + omega_max);

    // Candidate pairs in the band, ordered by closeness to the window.
    let nrpa = rpa.omegas.len();
    let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
    for a in 0..nrpa {
        for b in (a + 1)..nrpa {
            let pole = rpa.omegas[a] + rpa.omegas[b];
            if pole < lo || pole > hi {
                continue;
            }
            candidates.push(((pole - window_centre).abs(), a, b));
        }
    }
    candidates.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    if candidates.len() > MAX_KERNEL_TERMS {
        eprintln!(
            "  [dynamic BSE] {} RPA pairs in the window band; keeping the {} closest to the centre.",
            candidates.len(),
            MAX_KERNEL_TERMS
        );
        candidates.truncate(MAX_KERNEL_TERMS);
    }

    let ri_oo = super::get_submatrix(scf_data, 'O', 'O', 'N');
    let ri_vv = super::get_submatrix(scf_data, 'V', 'V', 'N');
    let ri_ov = super::get_submatrix(scf_data, 'O', 'V', 'N');
    let v_oo_ov = construct_coulomb(&ri_oo, &ri_ov);
    let v_vv_ov = construct_coulomb(&ri_vv, &ri_ov);

    let mut raw: Vec<(f64, Vec<f64>)> = candidates
        .into_iter()
        .map(|(_, a, b)| {
            let pole = rpa.omegas[a] + rpa.omegas[b];
            let c = coupling_vector(
                &v_oo_ov,
                &v_vv_ov,
                &rpa.vectors[a],
                &rpa.vectors[b],
                occ_size,
                vir_size,
            );
            (pole, c)
        })
        .collect();

    // Group vectors that share (numerically) the same pole.
    raw.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut terms: Vec<KernelTerm> = Vec::new();
    for (pole, c) in raw {
        match terms.last_mut() {
            Some(t) if (t.pole - pole).abs() < 1e-10 => t.vectors.push(c),
            _ => terms.push(KernelTerm { pole, vectors: vec![c] }),
        }
    }
    terms
}

// ============================================================================
// Bare-Coulomb sRPA coupling and kernel (Sangalli 2011, Eq. (7))
// ============================================================================

/// Coupling vector `C_{(ij),(nq)(mp)}` of Eq. (7) for one double excitation:
///
/// ```text
/// C = ½ ( v_(in),(mp) δ_j,q + v_(jq),(mp) δ_i,n
///         − {n↔m} − {q↔p} + {(nq)↔(mp)} )
/// ```
///
/// The δ-functions make `C` sparse: it is non-zero only on the two columns
/// `j = q, p` and the two rows `i = n, m`.  Returns the dense length-`no·nv`
/// vector (index `i + j·occ`).
fn srpa_coupling_vector(
    v_oo_ov: &MatrixFull<f64>,
    v_vv_ov: &MatrixFull<f64>,
    n: usize,
    q: usize,
    m: usize,
    p: usize,
    occ_size: usize,
    vir_size: usize,
) -> Vec<f64> {
    let no = occ_size;
    let nv = vir_size;
    let mut c = vec![0.0; no * nv];
    let idx = |i: usize, j: usize| i + j * no;

    // Column j = q, rows i:  ½[ v_(in),(mp) − v_(im),(np) ]
    for i in 0..no {
        let a = v_oo_ov[[i + n * no, m + p * no]];
        let b = v_oo_ov[[i + m * no, n + p * no]];
        c[idx(i, q)] += 0.5 * (a - b);
    }
    // Column j = p, rows i:  ½[ −v_(in),(mq) + v_(im),(nq) ]
    for i in 0..no {
        let a = v_oo_ov[[i + n * no, m + q * no]];
        let b = v_oo_ov[[i + m * no, n + q * no]];
        c[idx(i, p)] += 0.5 * (-a + b);
    }
    // Row i = n, columns j:  ½[ v_(jq),(mp) − v_(jp),(mq) ]
    for j in 0..nv {
        let a = v_vv_ov[[j + q * nv, m + p * no]];
        let b = v_vv_ov[[j + p * nv, m + q * no]];
        c[idx(n, j)] += 0.5 * (a - b);
    }
    // Row i = m, columns j:  ½[ −v_(jq),(np) + v_(jp),(nq) ]
    for j in 0..nv {
        let a = v_vv_ov[[j + q * nv, n + p * no]];
        let b = v_vv_ov[[j + p * nv, n + q * no]];
        c[idx(m, j)] += 0.5 * (-a + b);
    }
    c
}

/// Build the bare-Coulomb sRPA kernel terms
///   `Ξ(ω) = Σ_{(nq)≤(mp)} C Cᵀ / (ω − (Δ_{nq} + Δ_{mp}))`
/// for all double excitations whose energy lies in the window band.
///
/// The two e-h pairs are taken in a canonical order (pair index `a ≤ b`) so
/// that each double excitation is counted once; identical pairs give `C = 0`
/// (Pauli).
fn build_srpa_kernel_terms(
    scf_data: &SCF,
    occ_size: usize,
    vir_size: usize,
    diag: &[f64],
    omega_min: f64,
    omega_max: f64,
) -> Vec<KernelTerm> {
    let lo = omega_min - PAIR_BAND;
    let hi = omega_max + PAIR_BAND;
    let window_centre = 0.5 * (omega_min + omega_max);
    let n_exc = occ_size * vir_size;

    // Candidate doubles in the band, closest to the window first.
    let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
    for a in 0..n_exc {
        for b in a..n_exc {
            let e = diag[a] + diag[b];
            if e < lo || e > hi {
                continue;
            }
            candidates.push(((e - window_centre).abs(), a, b));
        }
    }
    candidates.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    if candidates.len() > MAX_KERNEL_TERMS {
        eprintln!(
            "  [sRPA] {} double excitations in the window band; keeping the {} closest to the centre.",
            candidates.len(),
            MAX_KERNEL_TERMS
        );
        candidates.truncate(MAX_KERNEL_TERMS);
    }

    let ri_oo = super::get_submatrix(scf_data, 'O', 'O', 'N');
    let ri_vv = super::get_submatrix(scf_data, 'V', 'V', 'N');
    let ri_ov = super::get_submatrix(scf_data, 'O', 'V', 'N');
    let v_oo_ov = construct_coulomb(&ri_oo, &ri_ov);
    let v_vv_ov = construct_coulomb(&ri_vv, &ri_ov);

    let mut raw: Vec<(f64, Vec<f64>)> = Vec::new();
    for (_, a, b) in candidates {
        let (n, q) = (a % occ_size, a / occ_size);
        let (m, p) = (b % occ_size, b / occ_size);
        let pole = diag[a] + diag[b];
        let c = srpa_coupling_vector(&v_oo_ov, &v_vv_ov, n, q, m, p, occ_size, vir_size);
        let nrm: f64 = c.iter().map(|v| v * v).sum::<f64>().sqrt();
        if nrm > 1e-14 {
            raw.push((pole, c));
        }
    }

    raw.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut terms: Vec<KernelTerm> = Vec::new();
    for (pole, c) in raw {
        match terms.last_mut() {
            Some(t) if (t.pole - pole).abs() < 1e-10 => t.vectors.push(c),
            _ => terms.push(KernelTerm { pole, vectors: vec![c] }),
        }
    }
    terms
}
// ============================================================================

/// Exact linearisation of the projected rational eigenproblem
///
///   T_Q(ω) y = [ S_Q − ω·I + Σ_k c_k c_kᵀ/(ω − d_k) ] y = 0 .
///
/// With `t_k = (c_k·y)/(ω−d_k)` this is equivalent to the standard (symmetric)
/// eigenvalue problem
///
///   [ S_Q   C  ] [y]       [y]
///   [ Cᵀ    D  ] [t] = ω · [t] ,   D = diag(d_k),  C = [c_1 c_2 …] ,
///
/// of dimension `m + nterms`.  Returns the linearised matrix.
fn linearize_rational(
    s_q: &MatrixFull<f64>,
    c_mat: &MatrixFull<f64>,
    poles: &[f64],
) -> MatrixFull<f64> {
    let m = s_q.size[0];
    let nterms = poles.len();
    let dim = m + nterms;
    let mut lin = MatrixFull::new([dim, dim], 0.0);
    for i in 0..m {
        for j in 0..m {
            lin[[i, j]] = s_q[[i, j]];
        }
    }
    for k in 0..nterms {
        for r in 0..m {
            lin[[r, m + k]] = c_mat[[r, k]];
            lin[[m + k, r]] = c_mat[[r, k]];
        }
        lin[[m + k, m + k]] = poles[k];
    }
    lin
}

struct DynamicBseOperator<'a> {
    scf_data: &'a SCF,
    qp_ctrl: &'a QuasiParticle,
    mv: &'a BseMatvec,
    n: usize,
    energy_diag: Vec<f64>,
    terms: Vec<KernelTerm>,
    gmres_restart: usize,
    gmres_max_it: usize,
    gmres_tol: f64,
}

impl DynamicBseOperator<'_> {
    fn static_a(&self, x: &Vec<f64>) -> Vec<f64> {
        self.mv.a(self.scf_data, self.qp_ctrl, x)
    }

    /// Ξ_d(ω)·x for a real ω.
    fn kernel_real(&self, omega: f64, x: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut y = vec![0.0; n];
        for term in &self.terms {
            let denom = omega - term.pole;
            if denom.abs() < 1e-14 {
                continue;
            }
            let f = 1.0 / denom;
            for c in &term.vectors {
                let dot: f64 = c.iter().zip(x.iter()).map(|(ci, xi)| ci * xi).sum();
                let g = f * dot;
                for i in 0..n {
                    y[i] += g * c[i];
                }
            }
        }
        y
    }

    /// Ξ_d(z)·(xr + i·xi) for a complex z, returned as (real, imag).
    fn kernel_complex(&self, z: Complex<f64>, xr: &[f64], xi: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = self.n;
        let mut yr = vec![0.0; n];
        let mut yi = vec![0.0; n];
        for term in &self.terms {
            let inv = Complex::new(z.re - term.pole, z.im).inv();
            for c in &term.vectors {
                let mut sr = 0.0;
                let mut si = 0.0;
                for i in 0..n {
                    sr += c[i] * xr[i];
                    si += c[i] * xi[i];
                }
                let g = Complex::new(sr, si) * inv;
                for i in 0..n {
                    yr[i] += g.re * c[i];
                    yi[i] += g.im * c[i];
                }
            }
        }
        (yr, yi)
    }
}

impl NlepOperator for DynamicBseOperator<'_> {
    fn dim(&self) -> usize {
        self.n
    }

    fn projected_solve(
        &self,
        q: &MatrixFull<f64>,
        _lambda_init: &[f64],
    ) -> (Vec<Complex<f64>>, MatrixFull<f64>) {
        let n = self.n;
        let m = q.size[1];
        let nterms: usize = self.terms.iter().map(|t| t.vectors.len()).sum();

        // S_Q = Qᵀ S Q via the implicit static matvec.
        let mut aq = MatrixFull::new([n, m], 0.0);
        for j in 0..m {
            let qj: Vec<f64> = (0..n).map(|i| q[[i, j]]).collect();
            let aqj = self.static_a(&qj);
            for i in 0..n {
                aq[[i, j]] = aqj[i];
            }
        }
        let s_q = _dgemm_scaled(q, 'T', &aq, 'N', 1.0);

        // c_k = Qᵀ C_k for every coupling vector.
        let mut c_mat = MatrixFull::new([m, nterms], 0.0);
        let mut poles = Vec::with_capacity(nterms);
        let mut col = 0;
        for term in &self.terms {
            for c in &term.vectors {
                for r in 0..m {
                    let mut acc = 0.0;
                    for i in 0..n {
                        acc += q[[i, r]] * c[i];
                    }
                    c_mat[[r, col]] = acc;
                }
                poles.push(term.pole);
                col += 1;
            }
        }

        // Exact linearisation of the projected rational eigenproblem:
        //   [ S_Q   C ] [y]       [y]
        //   [ Cᵀ    D ] [t] = ω · [t] ,   D = diag(poles).
        let lin = linearize_rational(&s_q, &c_mat, &poles);
        let dim = lin.size[0];

        let (_, wr, wi, _, vr, info) = _dgeev(&lin, 'N', 'V');
        if info != 0 {
            eprintln!("Warning: dynamic-BSE projected dgeev returned info={}", info);
        }

        let mut lambdas = Vec::with_capacity(dim);
        let mut ritz = MatrixFull::new([n, dim], 0.0);
        for j in 0..dim {
            lambdas.push(Complex::new(wr[j], wi[j]));
            // y = first m components of the linearisation eigenvector, then the
            // Ritz vector Q·y normalised to unit norm.  The linearisation
            // eigenvector is normalised in the (m + nterms)-dimensional space,
            // so ‖Q·y‖ can be far below 1; without renormalisation the residual
            // ‖T(ω)x‖ would be artificially small and convergence would stall.
            let mut norm2 = 0.0;
            for r in 0..n {
                let mut acc = 0.0;
                for k in 0..m {
                    acc += q[[r, k]] * vr[[k, j]];
                }
                ritz[[r, j]] = acc;
                norm2 += acc * acc;
            }
            let nrm = norm2.sqrt();
            if nrm > 1e-30 {
                for r in 0..n {
                    ritz[[r, j]] /= nrm;
                }
            }
        }
        (lambdas, ritz)
    }

    fn t_real(&self, lambda: f64, x: &[f64]) -> Vec<f64> {
        let sx = self.static_a(&x.to_vec());
        let kx = self.kernel_real(lambda, x);
        (0..self.n)
            .map(|i| sx[i] - lambda * x[i] + kx[i])
            .collect()
    }

    fn solve_shifted(&self, _node_index: usize, z: Complex<f64>, rhs: &[f64]) -> Vec<f64> {
        let n = self.n;
        let zr = z.re;
        let zi = z.im;

        let matvec = |v: &[f64]| -> Vec<f64> {
            let xr = &v[0..n];
            let xi = &v[n..2 * n];
            let sr = self.static_a(&xr.to_vec());
            let si = self.static_a(&xi.to_vec());
            let (kr, ki) = self.kernel_complex(z, xr, xi);
            let mut out = vec![0.0; 2 * n];
            for i in 0..n {
                out[i] = sr[i] - zr * xr[i] + zi * xi[i] + kr[i];
                out[n + i] = si[i] - zr * xi[i] - zi * xr[i] + ki[i];
            }
            out
        };

        // 2×2 block-diagonal preconditioner from the diagonal of T(z):
        //   d_i = Δ_i − z + Σ_k c_ki² /(z − d_k)
        let mut re_part = Vec::with_capacity(n);
        let mut im_part = Vec::with_capacity(n);
        for i in 0..n {
            let mut acc = Complex::new(self.energy_diag[i] - zr, -zi);
            for term in &self.terms {
                let inv = Complex::new(z.re - term.pole, z.im).inv();
                for c in &term.vectors {
                    acc += c[i] * c[i] * inv;
                }
            }
            re_part.push(acc.re);
            im_part.push(acc.im);
        }
        let precond = BlockDiagPrecond::from_re_im(&re_part, &im_part);

        gmres(
            &matvec,
            rhs,
            self.gmres_restart,
            self.gmres_max_it,
            self.gmres_tol,
            false,
            Some(&precond),
        )
    }
}

// ============================================================================
// Contour nodes
// ============================================================================

fn circle_nodes(centre: f64, radius: f64, n_quad: usize) -> Vec<ContourNode> {
    (0..n_quad)
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
        .collect()
}

// ============================================================================
// Driver
// ============================================================================

fn spin_coulomb_factor(spin: &str) -> f64 {
    match spin {
        "singlet" => 2.0,
        "triplet" => 0.0,
        _ => 1.0,
    }
}

/// Top-level entry point for the dynamical BSE (double excitations).
pub fn dynamic_bse_main(scf_data: &SCF, qp_ctrl: &QuasiParticle) {
    let start = Instant::now();
    let (_start_mo, _num_state, occ_size, vir_size, _homo, _lumo) =
        get_occupation_parameters(scf_data, 'N');
    let n = occ_size * vir_size;

    let quasiparticle_energies = scf_data.gwqp.0.clone();

    let centre = qp_ctrl.nlfeast_centre;
    let radius = qp_ctrl.nlfeast_radius;
    let window_min = centre - radius;
    let window_max = centre + radius;
    let srpa = qp_ctrl.dynamic_bse_kernel.eq_ignore_ascii_case("srpa");

    // ── Energy denominators ──
    let ks_energies = scf_data.eigenvalues[0].clone();
    let mut epsilon = ks_energies.clone();
    if qp_ctrl.bse_qp_polarization {
        epsilon = quasiparticle_energies.clone();
    }
    // The static matvec uses the GW quasiparticle gaps on its diagonal
    // (`diagonal_elements_contribution`), so the kernel and the preconditioner
    // use the same one-particle energies.
    let diag = construct_energy_diag_for_a(&quasiparticle_energies, occ_size, vir_size);

    // ── Static singles block S (implicit matvec) ──
    //   "bse"  : screened GW-BSE (inverse dielectric from the RPA response)
    //   "srpa" : bare Coulomb (identity screening ⇒ TDHF TDA, no W)
    let mv = if srpa {
        let naux = super::get_submatrix(scf_data, 'O', 'V', 'N').size[0];
        let identity = super::identity_matrix(naux);
        BseMatvec::new(scf_data, &identity, false)
    } else {
        let inverse_dielectric = construct_inverse_dielectric(scf_data, &epsilon);
        BseMatvec::new(scf_data, &inverse_dielectric, false)
    };

    println!("=============================================");
    if srpa {
        println!("  bare-v sRPA (TDHF singles + 2p2h coupling) + NLFEAST");
    } else {
        println!("  Dynamical BSE (RPA-pair NC kernel + NLFEAST)");
    }
    println!("=============================================");
    println!("  occ_size = {}, vir_size = {}, n = {}", occ_size, vir_size, n);
    if qp_ctrl.bse_tda {
        println!("  Method: TDA  (S + Ξ(ω) − ωI)");
    } else {
        println!(
            "  WARNING: non-TDA requested; the dynamical kernel is derived in the TDA. \
             Running the TDA form."
        );
    }
    println!("  Search contour: centre={:.6}, radius={:.6} Ha", centre, radius);

    // ── Dynamical kernel ──
    let terms = if srpa {
        // Eq. (7): bare 2p2h coupling with the diagonal doubles energies.
        let t = build_srpa_kernel_terms(
            scf_data, occ_size, vir_size, &diag, window_min, window_max,
        );
        let nc: usize = t.iter().map(|x| x.vectors.len()).sum();
        println!(
            "  Bare-v sRPA kernel: {} pole groups, {} double-excitation coupling vectors",
            t.len(),
            nc
        );
        t
    } else {
        // Eq. (22): number-conserving coupling in the RPA-pair basis.
        let ri_ov = super::get_submatrix(scf_data, 'O', 'V', 'N');
        let coulomb_factor = spin_coulomb_factor(&qp_ctrl.bse_spin);
        let rpa = solve_rpa_excitations(
            &ri_ov,
            &diag,
            coulomb_factor,
            window_max,
            occ_size,
            vir_size,
        );
        println!(
            "  RPA excitations below {:.4} Ha: {} (spin={}, Coulomb factor={})",
            window_max + PAIR_BAND,
            rpa.omegas.len(),
            qp_ctrl.bse_spin,
            coulomb_factor
        );
        if !rpa.omegas.is_empty() {
            let show: Vec<String> = rpa.omegas.iter().take(8).map(|e| format!("{:.4}", e)).collect();
            println!("  Lowest RPA poles (Ha): {}", show.join(", "));
        }
        let t = build_kernel_terms(
            scf_data, occ_size, vir_size, &rpa, window_min, window_max,
        );
        let npairs: usize = t.iter().map(|x| x.vectors.len()).sum();
        println!(
            "  Dynamical kernel: {} pole groups, {} RPA-pair coupling vectors",
            t.len(),
            npairs
        );
        t
    };
    let npairs: usize = terms.iter().map(|t| t.vectors.len()).sum();
    if npairs == 0 {
        println!("  No double excitations contribute inside the window — nothing to do.");
        return;
    }

    let op = DynamicBseOperator {
        scf_data,
        qp_ctrl,
        mv: &mv,
        n,
        energy_diag: diag.clone(),
        terms,
        gmres_restart: qp_ctrl.nlfeast_gmres_restart,
        gmres_max_it: qp_ctrl.nlfeast_gmres_max_it,
        gmres_tol: qp_ctrl.nlfeast_gmres_tol,
    };

    // ── Initial subspace and eigenvalue estimates ──
    let m0 = qp_ctrl.nlfeast_m0;
    let q0 = {
        let u = crate::solvers::davidson::generate_initial_guess(
            &diag,
            m0.min(n).max(1),
        );
        crate::solvers::nlfeast::qr_orthonormalise(&u)
    };
    let lambda_init: Vec<f64> = {
        let mut d: Vec<(f64, f64)> = diag
            .iter()
            .map(|&e| ((e - centre).abs(), e))
            .collect();
        d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        d.iter().take(m0).map(|(_, e)| *e).collect()
    };

    let nodes = circle_nodes(centre, radius, qp_ctrl.nlfeast_n_quad);
    println!(
        "  Contour nodes: {} (trapezoidal), m0 = {}",
        nodes.len(),
        m0
    );

    super::matvec_trace::set_enabled(qp_ctrl.export_matvec_count);

    let result: NLFeastResult = nlfeast(
        &op,
        q0,
        lambda_init,
        &nodes,
        centre,
        radius,
        qp_ctrl.nlfeast_max_iter,
        qp_ctrl.nlfeast_tol,
    );

    println!("  Dynamical BSE total time: {:?}", start.elapsed());
    println!("  Outer iterations: {}", result.iterations);

    if result.n_found == 0 {
        println!("  No excitations found inside the contour.");
        return;
    }

    println!(
        "\n  Found {} excitation(s) inside [{:.4}, {:.4}] Ha:\n",
        result.n_found,
        window_min,
        window_max
    );
    println!("  #       Excitation energy (eV)    ‖T(ω)x‖");
    println!("  ───     ─────────────────────    ──────────");
    for k in 0..result.n_found {
        println!(
            "  {:>3}     {:>12.6} eV             {:>9.2e}",
            k,
            result.eigenvalues[k] * EV,
            result.residuals[k]
        );
    }

    if qp_ctrl.save_bse_excitations {
        let line = result
            .eigenvalues
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        use std::fs::OpenOptions;
        use std::io::Write;
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open("dynamic_bse_excitations.txt")
            .expect("Failed to open dynamic_bse_excitations.txt");
        writeln!(file, "{}", line).expect("Failed to write excitations");
    }

    if qp_ctrl.save_first_excitation && result.n_found > 0 {
        let save_path = qp_ctrl.save_first_excitation_path.clone();
        use std::fs::OpenOptions;
        use std::io::Write;
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(save_path)
            .expect("Failed to open save path");
        writeln!(file, "{}", result.eigenvalues[0]).expect("Failed to write");
    }
}

// ============================================================================
// Unit tests for the pure algebraic building blocks.
//
// The end-to-end driver needs a full SCF/GW/BSE setup, so these tests exercise
// the two pieces where an index or algebra mistake would be silent:
//   * the Eq. (22) coupling contraction, against a literal double loop, and
//   * the exact linearisation of the projected rational eigenproblem.
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    }

    /// Literal double-loop evaluation of Eq. (22) in the code's index
    /// conventions: oo pair = i+n·occ, ov pair = m+p·occ, vv pair = j+q·vir;
    /// `R[n+p·occ]`; output `i+j·occ`.
    fn coupling_bruteforce(
        v_oo_ov: &MatrixFull<f64>,
        v_vv_ov: &MatrixFull<f64>,
        r1: &[f64],
        r2: &[f64],
        no: usize,
        nv: usize,
    ) -> Vec<f64> {
        let oo = |i: usize, n: usize, m: usize, p: usize| v_oo_ov[[i + n * no, m + p * no]];
        let vv = |j: usize, q: usize, m: usize, p: usize| v_vv_ov[[j + q * nv, m + p * no]];
        let rr = |r: &[f64], n: usize, p: usize| r[n + p * no];
        let mut out = vec![0.0; no * nv];
        for i in 0..no {
            for j in 0..nv {
                let mut acc = 0.0;
                // Term A: v_(in),(mp) δ_jq R1_(np) R2_(mq)
                for n in 0..no {
                    for m in 0..no {
                        for p in 0..nv {
                            acc += oo(i, n, m, p) * rr(r1, n, p) * rr(r2, m, j);
                        }
                    }
                }
                // Term B: v_(mp),(jq) δ_in  =  vv_(jq),(mp)
                for m in 0..no {
                    for p in 0..nv {
                        for q in 0..nv {
                            acc += vv(j, q, m, p) * rr(r1, i, p) * rr(r2, m, q);
                        }
                    }
                }
                // {ν1↔ν2}
                for n in 0..no {
                    for m in 0..no {
                        for p in 0..nv {
                            acc += oo(i, n, m, p) * rr(r2, n, p) * rr(r1, m, j);
                        }
                    }
                }
                for m in 0..no {
                    for p in 0..nv {
                        for q in 0..nv {
                            acc += vv(j, q, m, p) * rr(r2, i, p) * rr(r1, m, q);
                        }
                    }
                }
                out[i + j * no] = 0.5 * acc;
            }
        }
        out
    }

    #[test]
    fn coupling_contraction_matches_eq22() {
        let no = 3;
        let nv = 2;
        let mut seed = 20240517u64;
        let mut v_oo_ov = MatrixFull::new([no * no, no * nv], 0.0);
        for i in 0..no * no {
            for j in 0..no * nv {
                v_oo_ov[[i, j]] = lcg(&mut seed);
            }
        }
        let mut v_vv_ov = MatrixFull::new([nv * nv, no * nv], 0.0);
        for i in 0..nv * nv {
            for j in 0..no * nv {
                v_vv_ov[[i, j]] = lcg(&mut seed);
            }
        }
        let r1: Vec<f64> = (0..no * nv).map(|_| lcg(&mut seed)).collect();
        let r2: Vec<f64> = (0..no * nv).map(|_| lcg(&mut seed)).collect();

        let got = coupling_vector(&v_oo_ov, &v_vv_ov, &r1, &r2, no, nv);
        let want = coupling_bruteforce(&v_oo_ov, &v_vv_ov, &r1, &r2, no, nv);
        for k in 0..got.len() {
            assert!(
                (got[k] - want[k]).abs() < 1e-12,
                "coupling[{k}] = {} vs {}",
                got[k],
                want[k]
            );
        }
    }

    #[test]
    fn linearisation_reproduces_rational_eigenvalues() {
        let m = 4;
        let nterms = 5;
        let mut seed = 987654321u64;

        // Random symmetric S_Q.
        let mut s = vec![vec![0.0; m]; m];
        for i in 0..m {
            for j in 0..=i {
                let val = lcg(&mut seed) * 2.0;
                s[i][j] = val;
                s[j][i] = val;
            }
        }
        let mut s_q = MatrixFull::new([m, m], 0.0);
        for i in 0..m {
            for j in 0..m {
                s_q[[i, j]] = s[i][j];
            }
        }

        // Random coupling vectors and poles.
        let mut c_mat = MatrixFull::new([m, nterms], 0.0);
        for k in 0..nterms {
            for r in 0..m {
                c_mat[[r, k]] = lcg(&mut seed);
            }
        }
        let poles: Vec<f64> = (0..nterms).map(|k| 1.0 + 0.7 * k as f64).collect();

        let lin = linearize_rational(&s_q, &c_mat, &poles);
        let (_, wr, wi, _, vr, info) = _dgeev(&lin, 'N', 'V');
        assert_eq!(info, 0);

        // Every linearisation eigenvalue ω with first-block eigenvector y must
        // satisfy T(ω) y ≈ 0,  T(ω) = S_Q − ωI + Σ c_k c_kᵀ/(ω−d_k).
        for j in 0..lin.size[0] {
            assert!(wi[j].abs() < 1e-8, "poles/eigenvalues should be real here");
            let omega = wr[j];
            let mut y = vec![0.0; m];
            for r in 0..m {
                y[r] = vr[[r, j]];
            }
            // Project out the (tiny) pole components: skip roots sitting on a pole.
            if poles.iter().any(|d| (omega - d).abs() < 1e-6) {
                continue;
            }
            let mut ty = vec![0.0; m];
            for r in 0..m {
                let mut acc = -omega * y[r];
                for c in 0..m {
                    acc += s[r][c] * y[c];
                }
                ty[r] = acc;
            }
            for k in 0..nterms {
                let dot: f64 = (0..m).map(|r| c_mat[[r, k]] * y[r]).sum();
                let f = dot / (omega - poles[k]);
                for r in 0..m {
                    ty[r] += f * c_mat[[r, k]];
                }
            }
            let nrm: f64 = ty.iter().map(|v| v * v).sum::<f64>().sqrt();
            let ynorm: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(
                nrm < 1e-6 * ynorm.max(1e-12),
                "linearised root ω={omega} has ‖T(ω)y‖={nrm} (‖y‖={ynorm})"
            );
        }
    }

    /// Literal evaluation of Eq. (7) for the bare-v sRPA coupling.
    #[allow(clippy::too_many_arguments)]
    fn srpa_coupling_bruteforce(
        v_oo_ov: &MatrixFull<f64>,
        v_vv_ov: &MatrixFull<f64>,
        n: usize,
        q: usize,
        m: usize,
        p: usize,
        no: usize,
        nv: usize,
    ) -> Vec<f64> {
        // v_(ab),(cd): oo pair = a+b·occ, ov pair = c+d·occ
        let oo = |a: usize, b: usize, c: usize, d: usize| v_oo_ov[[a + b * no, c + d * no]];
        // w_(ab),(cd): vv pair = a+b·vir, ov pair = c+d·occ
        let vv = |a: usize, b: usize, c: usize, d: usize| v_vv_ov[[a + b * nv, c + d * no]];
        let mut c = vec![0.0; no * nv];
        for i in 0..no {
            for j in 0..nv {
                let d_jq = if j == q { 1.0 } else { 0.0 };
                let d_jp = if j == p { 1.0 } else { 0.0 };
                let d_in = if i == n { 1.0 } else { 0.0 };
                let d_im = if i == m { 1.0 } else { 0.0 };
                let a = oo(i, n, m, p) * d_jq;
                let a_nm = oo(i, m, n, p) * d_jq;
                let a_qp = oo(i, n, m, q) * d_jp;
                let a_pairs = oo(i, m, n, q) * d_jp;
                let b = vv(j, q, m, p) * d_in;
                let b_nm = vv(j, q, n, p) * d_im;
                let b_qp = vv(j, p, m, q) * d_in;
                let b_pairs = vv(j, p, n, q) * d_im;
                c[i + j * no] =
                    0.5 * (a - a_nm - a_qp + a_pairs + b - b_nm - b_qp + b_pairs);
            }
        }
        c
    }

    #[test]
    fn srpa_coupling_matches_eq7() {
        let no = 3;
        let nv = 2;
        let mut seed = 424242u64;
        let mut v_oo_ov = MatrixFull::new([no * no, no * nv], 0.0);
        for i in 0..no * no {
            for j in 0..no * nv {
                v_oo_ov[[i, j]] = lcg(&mut seed);
            }
        }
        let mut v_vv_ov = MatrixFull::new([nv * nv, no * nv], 0.0);
        for i in 0..nv * nv {
            for j in 0..no * nv {
                v_vv_ov[[i, j]] = lcg(&mut seed);
            }
        }
        for n in 0..no {
            for q in 0..nv {
                for m in 0..no {
                    for p in 0..nv {
                        let got = srpa_coupling_vector(&v_oo_ov, &v_vv_ov, n, q, m, p, no, nv);
                        let want = srpa_coupling_bruteforce(
                            &v_oo_ov, &v_vv_ov, n, q, m, p, no, nv,
                        );
                        for k in 0..got.len() {
                            assert!(
                                (got[k] - want[k]).abs() < 1e-12,
                                "double ({n},{q},{m},{p}) coupling[{k}] = {} vs {}",
                                got[k],
                                want[k]
                            );
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Self-contained 8-level model (4 occupied + 4 virtual) with an explicit
    // two-body tensor.  The exact sRPA spectrum is obtained by diagonalising
    // [S C; Cᵀ D]; the production sRPA coupling + NLFEAST must reproduce the
    // in-window part of it.  This is the "same type of 8-level model" check.
    // ------------------------------------------------------------------

    struct ModelOp {
        s: MatrixFull<f64>,
        terms: Vec<KernelTerm>,
        n: usize,
    }

    impl ModelOp {
        fn smul(&self, x: &[f64]) -> Vec<f64> {
            let n = self.n;
            let mut y = vec![0.0; n];
            for i in 0..n {
                let mut acc = 0.0;
                for j in 0..n {
                    acc += self.s[[i, j]] * x[j];
                }
                y[i] = acc;
            }
            y
        }
        fn kernel_complex(&self, z: Complex<f64>, xr: &[f64], xi: &[f64]) -> (Vec<f64>, Vec<f64>) {
            let n = self.n;
            let mut yr = vec![0.0; n];
            let mut yi = vec![0.0; n];
            for term in &self.terms {
                let inv = Complex::new(z.re - term.pole, z.im).inv();
                for c in &term.vectors {
                    let mut sr = 0.0;
                    let mut si = 0.0;
                    for i in 0..n {
                        sr += c[i] * xr[i];
                        si += c[i] * xi[i];
                    }
                    let g = Complex::new(sr, si) * inv;
                    for i in 0..n {
                        yr[i] += g.re * c[i];
                        yi[i] += g.im * c[i];
                    }
                }
            }
            (yr, yi)
        }
    }

    impl NlepOperator for ModelOp {
        fn dim(&self) -> usize {
            self.n
        }
        fn projected_solve(
            &self,
            q: &MatrixFull<f64>,
            _lambda_init: &[f64],
        ) -> (Vec<Complex<f64>>, MatrixFull<f64>) {
            let n = self.n;
            let m = q.size[1];
            let nterms: usize = self.terms.iter().map(|t| t.vectors.len()).sum();
            let mut s_q = MatrixFull::new([m, m], 0.0);
            for p in 0..m {
                for r in 0..m {
                    let mut acc = 0.0;
                    for i in 0..n {
                        for j in 0..n {
                            acc += q[[i, p]] * self.s[[i, j]] * q[[j, r]];
                        }
                    }
                    s_q[[p, r]] = acc;
                }
            }
            let mut c_mat = MatrixFull::new([m, nterms], 0.0);
            let mut poles = Vec::with_capacity(nterms);
            let mut col = 0;
            for t in &self.terms {
                for c in &t.vectors {
                    for r in 0..m {
                        let mut acc = 0.0;
                        for i in 0..n {
                            acc += q[[i, r]] * c[i];
                        }
                        c_mat[[r, col]] = acc;
                    }
                    poles.push(t.pole);
                    col += 1;
                }
            }
            let lin = linearize_rational(&s_q, &c_mat, &poles);
            let dim = lin.size[0];
            let (_, wr, wi, _, vr, info) = _dgeev(&lin, 'N', 'V');
            assert_eq!(info, 0, "model projected dgeev failed");
            let mut lambdas = Vec::with_capacity(dim);
            let mut ritz = MatrixFull::new([n, dim], 0.0);
            for j in 0..dim {
                lambdas.push(Complex::new(wr[j], wi[j]));
                let mut norm2 = 0.0;
                for r in 0..n {
                    let mut acc = 0.0;
                    for k in 0..m {
                        acc += q[[r, k]] * vr[[k, j]];
                    }
                    ritz[[r, j]] = acc;
                    norm2 += acc * acc;
                }
                let nrm = norm2.sqrt();
                if nrm > 1e-30 {
                    for r in 0..n {
                        ritz[[r, j]] /= nrm;
                    }
                }
            }
            (lambdas, ritz)
        }
        fn t_real(&self, omega: f64, x: &[f64]) -> Vec<f64> {
            let sx = self.smul(x);
            let (kr, _) = self.kernel_complex(Complex::new(omega, 0.0), x, &vec![0.0; self.n]);
            (0..self.n)
                .map(|i| sx[i] - omega * x[i] + kr[i])
                .collect()
        }
        fn solve_shifted(&self, _node: usize, z: Complex<f64>, rhs: &[f64]) -> Vec<f64> {
            let n = self.n;
            // Assemble the dense complex T(z) = S − z + Σ c cᵀ/(z−d) and solve
            // its 2n real embedding exactly (n = 16 here, so this is cheap and
            // removes any iterative-solver error from the model check).
            let mut tr = vec![0.0; n * n];
            let mut ti = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    tr[i * n + j] = self.s[[i, j]];
                }
                tr[i * n + i] -= z.re;
                ti[i * n + i] -= z.im;
            }
            for t in &self.terms {
                let inv = Complex::new(z.re - t.pole, z.im).inv();
                for c in &t.vectors {
                    for i in 0..n {
                        for j in 0..n {
                            let g = c[i] * c[j] * inv;
                            tr[i * n + j] += g.re;
                            ti[i * n + j] += g.im;
                        }
                    }
                }
            }
            let n2 = 2 * n;
            let mut m = MatrixFull::new([n2, n2], 0.0);
            for i in 0..n {
                for j in 0..n {
                    m[[i, j]] = tr[i * n + j];
                    m[[i, n + j]] = -ti[i * n + j];
                    m[[n + i, j]] = ti[i * n + j];
                    m[[n + i, n + j]] = tr[i * n + j];
                }
            }
            let inv = rest_tensors::matrix::matrix_blas_lapack::_dinverse(&m)
                .expect("model shifted matrix is singular on the contour");
            let mut out = vec![0.0; n2];
            for i in 0..n2 {
                let mut acc = 0.0;
                for j in 0..n {
                    acc += inv[[i, j]] * rhs[j];
                }
                out[i] = acc;
            }
            out
        }
    }

    #[test]
    fn srpa_eight_level_model_self_consistency() {
        // ── 8-level model: 4 occupied + 4 virtual ──
        let no = 4usize;
        let nv = 4usize;
        let nlev = no + nv;
        let n = no * nv; // 16 singles
        let idx = |a: usize, b: usize, c: usize, d: usize| ((a * nlev + b) * nlev + c) * nlev + d;

        // Random *exactly* symmetric two-body tensor v[a,b,c,d], symmetric
        // under a↔b, c↔d and (ab)↔(cd).  The average is taken over the full
        // orbit of the symmetry group so that S is symmetric to machine
        // precision (otherwise the dsyevd reference and the matvec disagree).
        let mut seed = 20240601u64;
        let mut raw = vec![0.0f64; nlev * nlev * nlev * nlev];
        for x in raw.iter_mut() {
            *x = lcg(&mut seed);
        }
        let mut v = vec![0.0f64; nlev * nlev * nlev * nlev];
        for a in 0..nlev {
            for b in 0..nlev {
                for c in 0..nlev {
                    for d in 0..nlev {
                        let val = (raw[idx(a, b, c, d)]
                            + raw[idx(b, a, c, d)]
                            + raw[idx(a, b, d, c)]
                            + raw[idx(b, a, d, c)]
                            + raw[idx(c, d, a, b)]
                            + raw[idx(d, c, a, b)]
                            + raw[idx(c, d, b, a)]
                            + raw[idx(d, c, b, a)])
                            / 8.0;
                        v[idx(a, b, c, d)] = val;
                    }
                }
            }
        }

        // Orbital energies (randomised to avoid accidental degeneracies) and e-h gaps.
        let mut eseed = 987654321u64;
        let eps: Vec<f64> = (0..nlev)
            .map(|i| 0.05 + 0.03 * i as f64 + 0.004 * lcg(&mut eseed))
            .collect();
        let mut diag = vec![0.0; n];
        for i in 0..no {
            for a in 0..nv {
                diag[i + a * no] = eps[no + a] - eps[i];
            }
        }

        // Singles block S: TDHF TDA singlet  S = Δ + 2·direct − exchange.
        let mut s = MatrixFull::new([n, n], 0.0);
        for i in 0..no {
            for a in 0..nv {
                let p = i + a * no;
                for j in 0..no {
                    for b in 0..nv {
                        let q = j + b * no;
                        let direct = v[idx(i, a, b, j)];
                        let exch = v[idx(i, j, a, b)];
                        s[[p, q]] = 2.0 * direct - exch;
                        if p == q {
                            s[[p, q]] += diag[p];
                        }
                    }
                }
            }
        }

        // Coulomb blocks for Eq. (7).
        let mut v_oo_ov = MatrixFull::new([no * no, no * nv], 0.0);
        for i in 0..no {
            for nn in 0..no {
                for m in 0..no {
                    for p in 0..nv {
                        v_oo_ov[[i + nn * no, m + p * no]] = v[idx(i, nn, m, p)];
                    }
                }
            }
        }
        let mut v_vv_ov = MatrixFull::new([nv * nv, no * nv], 0.0);
        for j in 0..nv {
            for q in 0..nv {
                for m in 0..no {
                    for p in 0..nv {
                        v_vv_ov[[j + q * nv, m + p * no]] = v[idx(j, q, m, p)];
                    }
                }
            }
        }

        // Production sRPA coupling for every distinct unordered e-h pair.
        let mut terms: Vec<KernelTerm> = Vec::new();
        for a in 0..n {
            for b in (a + 1)..n {
                let (nn, q) = (a % no, a / no);
                let (m, p) = (b % no, b / no);
                let c =
                    srpa_coupling_vector(&v_oo_ov, &v_vv_ov, nn, q, m, p, no, nv);
                terms.push(KernelTerm {
                    pole: diag[a] + diag[b],
                    vectors: vec![c],
                });
            }
        }
        let nd = terms.len(); // 120 doubles

        // ── Exact sRPA reference: eigenvalues of [S C; Cᵀ D] ──
        let dim = n + nd;
        let mut full = MatrixFull::new([dim, dim], 0.0);
        for i in 0..n {
            for j in 0..n {
                full[[i, j]] = s[[i, j]];
            }
        }
        let mut col = 0;
        for t in &terms {
            for c in &t.vectors {
                for r in 0..n {
                    full[[r, n + col]] = c[r];
                    full[[n + col, r]] = c[r];
                }
                full[[n + col, n + col]] = t.pole;
                col += 1;
            }
        }
        let (_, ref_eigs, _) = _dsyevd(&full, 'N');
        assert_eq!(ref_eigs.len(), n + nd);

        // ── Static singles only ──
        let (_, s_eigs, _) = _dsyevd(&s, 'N');
        assert_eq!(s_eigs.len(), n);
        eprintln!(
            "8-level model: {} singles, {} doubles, exact total {} poles",
            n, nd, dim
        );
        eprintln!(
            "  static singles range [{:.4}, {:.4}] Ha",
            s_eigs[0], s_eigs[n - 1]
        );

        // ── Pick a window containing ~8-12 distinct exact poles (< n = 16) ──
        let mut chosen: Option<(f64, f64, Vec<f64>)> = None;
        'outer: for mid in 8..ref_eigs.len().saturating_sub(8) {
            for k in 4..=8usize {
                let lo = mid.saturating_sub(k);
                let hi = (mid + k).min(ref_eigs.len() - 1);
                let centre = 0.5 * (ref_eigs[lo] + ref_eigs[hi]);
                let radius = 0.5 * (ref_eigs[hi] - ref_eigs[lo]) + 1e-3;
                if radius < 0.02 {
                    continue;
                }
                let expected: Vec<f64> = ref_eigs
                    .iter()
                    .cloned()
                    .filter(|e| (e - centre).abs() <= radius)
                    .collect();
                if (8..=12).contains(&expected.len()) {
                    chosen = Some((centre, radius, expected));
                    break 'outer;
                }
            }
        }
        let (centre, radius, expected) = chosen.expect("no suitable model window found");
        let m0 = (expected.len() + 3).min(n);

        // ── Run the production NLFEAST on the model ──
        let q0 = {
            let mut rng = 1234567u64;
            let mut q = MatrixFull::new([n, m0], 0.0);
            for j in 0..m0 {
                for i in 0..n {
                    q[[i, j]] = lcg(&mut rng);
                }
            }
            crate::solvers::nlfeast::qr_orthonormalise(&q)
        };
        let lambda_init: Vec<f64> = {
            let mut d: Vec<(f64, f64)> =
                diag.iter().map(|&e| ((e - centre).abs(), e)).collect();
            d.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
            d.iter().take(m0).map(|(_, e)| *e).collect()
        };
        let op = ModelOp { s, terms, n };
        let nodes = circle_nodes(centre, radius, 64);
        let res = nlfeast(&op, q0, lambda_init, &nodes, centre, radius, 300, 1e-9);

        let mut found = res.eigenvalues.clone();
        found.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut want = expected.clone();
        want.sort_by(|a, b| a.partial_cmp(b).unwrap());

        eprintln!(
            "  window [{:.4}, {:.4}] Ha (r={:.4}, m0={}): exact {} poles, NLFEAST found {} in {} iters",
            centre - radius,
            centre + radius,
            radius,
            m0,
            want.len(),
            found.len(),
            res.iterations
        );
        assert_eq!(
            found.len(),
            want.len(),
            "NLFEAST found {:?}, exact window has {:?}",
            found,
            want
        );
        for (a, b) in found.iter().zip(want.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "model pole mismatch: {a} vs {b} (found {found:?}, exact {want:?})"
            );
        }
        for r in &res.residuals {
            assert!(*r < 1e-6, "model residual {r} too large");
        }
        // No spurious poles: the static block alone has only `n` poles, while
        // the exact model has `n + nd`; the window contains genuine doubles.
        assert!(want.iter().any(|e| {
            !s_eigs.iter().any(|se| (se - e).abs() < 1e-6)
        }), "window should contain at least one double excitation");
    }

    // ------------------------------------------------------------------
    // Exactly solvable *NC kernel* model.  Here the model is defined with the
    // doubles block diagonal in the RPA-pair basis (the assumption behind
    // Eq. (23)), so the NC kernel expansion is exact.  The reference spectrum
    // is an independent dense diagonalisation of
    //     H = [ S   C        ]
    //         [ Cᵀ  diag(E_ν+E_μ) ]
    // with C from the production Eq. (22) coupling and (E_ν, R_ν) from an
    // independently diagonalised TDH-TDA problem.
    // ------------------------------------------------------------------
    #[test]
    fn nc_kernel_eight_level_model_self_consistency() {
        let no = 4usize;
        let nv = 4usize;
        let nlev = no + nv;
        let n = no * nv; // 16 singles
        let idx = |a: usize, b: usize, c: usize, d: usize| ((a * nlev + b) * nlev + c) * nlev + d;

        // Exactly symmetric two-body tensor (same construction as the sRPA test).
        let mut seed = 1357911u64;
        let mut raw = vec![0.0f64; nlev * nlev * nlev * nlev];
        for x in raw.iter_mut() {
            *x = lcg(&mut seed);
        }
        let mut v = vec![0.0f64; nlev * nlev * nlev * nlev];
        for a in 0..nlev {
            for b in 0..nlev {
                for c in 0..nlev {
                    for d in 0..nlev {
                        v[idx(a, b, c, d)] = (raw[idx(a, b, c, d)]
                            + raw[idx(b, a, c, d)]
                            + raw[idx(a, b, d, c)]
                            + raw[idx(b, a, d, c)]
                            + raw[idx(c, d, a, b)]
                            + raw[idx(d, c, a, b)]
                            + raw[idx(c, d, b, a)]
                            + raw[idx(d, c, b, a)])
                            / 8.0;
                    }
                }
            }
        }

        let mut eseed = 246801357u64;
        let eps: Vec<f64> = (0..nlev)
            .map(|i| 0.05 + 0.03 * i as f64 + 0.004 * lcg(&mut eseed))
            .collect();
        let mut diag = vec![0.0; n];
        for i in 0..no {
            for a in 0..nv {
                diag[i + a * no] = eps[no + a] - eps[i];
            }
        }

        // Static singles block S (TDHF TDA singlet).
        let mut s = MatrixFull::new([n, n], 0.0);
        // Direct kernel D[(ia),(jb)] = v[i,a,b,j] (used by the RPA problem too).
        let mut direct = MatrixFull::new([n, n], 0.0);
        for i in 0..no {
            for a in 0..nv {
                let p = i + a * no;
                for j in 0..no {
                    for b in 0..nv {
                        let q = j + b * no;
                        direct[[p, q]] = v[idx(i, a, b, j)];
                        s[[p, q]] = 2.0 * v[idx(i, a, b, j)] - v[idx(i, j, a, b)];
                        if p == q {
                            s[[p, q]] += diag[p];
                        }
                    }
                }
            }
        }

        // TDH-TDA (singlet) RPA problem: A = diag + 2·direct, solved densely.
        let mut rpa = MatrixFull::new([n, n], 0.0);
        for p in 0..n {
            for q in 0..n {
                rpa[[p, q]] = 2.0 * direct[[p, q]];
            }
            rpa[[p, p]] += diag[p];
        }
        let (r_opt, rpa_e, _) = _dsyevd(&rpa, 'V');
        let r_mat = r_opt.unwrap();
        let rpa_vecs: Vec<Vec<f64>> = (0..n)
            .map(|k| (0..n).map(|i| r_mat[[i, k]]).collect())
            .collect();

        // Eq. (22) coupling for every RPA pair (production implementation).
        let mut v_oo_ov = MatrixFull::new([no * no, no * nv], 0.0);
        for i in 0..no {
            for nn in 0..no {
                for m in 0..no {
                    for p in 0..nv {
                        v_oo_ov[[i + nn * no, m + p * no]] = v[idx(i, nn, m, p)];
                    }
                }
            }
        }
        let mut v_vv_ov = MatrixFull::new([nv * nv, no * nv], 0.0);
        for j in 0..nv {
            for q in 0..nv {
                for m in 0..no {
                    for p in 0..nv {
                        v_vv_ov[[j + q * nv, m + p * no]] = v[idx(j, q, m, p)];
                    }
                }
            }
        }

        let mut terms: Vec<KernelTerm> = Vec::new();
        for a in 0..n {
            for b in (a + 1)..n {
                let c = coupling_vector(
                    &v_oo_ov,
                    &v_vv_ov,
                    &rpa_vecs[a],
                    &rpa_vecs[b],
                    no,
                    nv,
                );
                terms.push(KernelTerm {
                    pole: rpa_e[a] + rpa_e[b],
                    vectors: vec![c],
                });
            }
        }
        let npairs = terms.len(); // 120 RPA pairs

        // Exact NC model H = [[S, C], [Cᵀ, diag(poles)]].
        let dim = n + npairs;
        let mut full = MatrixFull::new([dim, dim], 0.0);
        for i in 0..n {
            for j in 0..n {
                full[[i, j]] = s[[i, j]];
            }
        }
        let mut col = 0;
        for t in &terms {
            for c in &t.vectors {
                for r in 0..n {
                    full[[r, n + col]] = c[r];
                    full[[n + col, r]] = c[r];
                }
                full[[n + col, n + col]] = t.pole;
                col += 1;
            }
        }
        let (_, ref_eigs, _) = _dsyevd(&full, 'N');
        assert_eq!(ref_eigs.len(), n + npairs);
        let (_, s_eigs, _) = _dsyevd(&s, 'N');
        eprintln!(
            "NC-kernel 8-level model: {} singles + {} RPA pairs = {} exact poles",
            n, npairs, dim
        );

        // Window with ~8-12 distinct exact poles (< n = 16).
        let mut chosen: Option<(f64, f64, Vec<f64>)> = None;
        'outer: for mid in 8..ref_eigs.len().saturating_sub(8) {
            for k in 4..=8usize {
                let lo = mid.saturating_sub(k);
                let hi = (mid + k).min(ref_eigs.len() - 1);
                let centre = 0.5 * (ref_eigs[lo] + ref_eigs[hi]);
                let radius = 0.5 * (ref_eigs[hi] - ref_eigs[lo]) + 1e-3;
                if radius < 0.02 {
                    continue;
                }
                let expected: Vec<f64> = ref_eigs
                    .iter()
                    .cloned()
                    .filter(|e| (e - centre).abs() <= radius)
                    .collect();
                if (8..=12).contains(&expected.len()) {
                    chosen = Some((centre, radius, expected));
                    break 'outer;
                }
            }
        }
        let (centre, radius, expected) = chosen.expect("no suitable NC model window found");
        let m0 = (expected.len() + 3).min(n);

        let q0 = {
            let mut rng = 778899u64;
            let mut q = MatrixFull::new([n, m0], 0.0);
            for j in 0..m0 {
                for i in 0..n {
                    q[[i, j]] = lcg(&mut rng);
                }
            }
            crate::solvers::nlfeast::qr_orthonormalise(&q)
        };
        let lambda_init: Vec<f64> = {
            let mut d: Vec<(f64, f64)> =
                diag.iter().map(|&e| ((e - centre).abs(), e)).collect();
            d.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
            d.iter().take(m0).map(|(_, e)| *e).collect()
        };
        let op = ModelOp { s, terms, n };
        let nodes = circle_nodes(centre, radius, 64);
        let res = nlfeast(&op, q0, lambda_init, &nodes, centre, radius, 300, 1e-9);

        let mut found = res.eigenvalues.clone();
        found.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut want = expected.clone();
        want.sort_by(|a, b| a.partial_cmp(b).unwrap());

        eprintln!(
            "  window [{:.4}, {:.4}] Ha (r={:.4}, m0={}): exact {} NC poles, NLFEAST found {} in {} iters",
            centre - radius,
            centre + radius,
            radius,
            m0,
            want.len(),
            found.len(),
            res.iterations
        );
        assert_eq!(
            found.len(),
            want.len(),
            "NC NLFEAST found {found:?}, exact window has {want:?}"
        );
        for (a, b) in found.iter().zip(want.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "NC pole mismatch: {a} vs {b} (found {found:?}, exact {want:?})"
            );
        }
        for r in &res.residuals {
            assert!(*r < 1e-6, "NC model residual {r} too large");
        }
        // The window must contain at least one genuine RPA-pair (double) pole
        // that the static singles block alone does not have.
        assert!(want.iter().any(|e| {
            !s_eigs.iter().any(|se| (se - e).abs() < 1e-6)
        }), "window should contain at least one NC RPA-pair excitation");
    }
}
