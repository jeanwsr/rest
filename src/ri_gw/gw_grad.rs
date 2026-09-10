//! Analytic nuclear gradients of low-rank contour-deformation (LR-CD) G0W0
//! quasiparticle energies.
//!
//! This module ports the validated PySCF implementation
//! `pyscf/gw/gw_lr_cd_grad.py` (+ `gw_cd_grad*.py`) to REST.  It reproduces
//! REST's own LR-CD G0W0 energy functional (`ri_gw::gw_calculations_lowrank`
//! / `newton_solver_lowrank_v2`) — the *raw* (non-static-subtracted)
//! contour-deformation decomposition with REST's real-symmetric screened
//! response, REST's imaginary-axis quadrature grids, REST's nearest-neighbour
//! real-axis residue interpolation and REST's finite-difference Newton root
//! finder — and differentiates it exactly.
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
//!   grid construction (de_max scan, linear/quadratic nodes, nearest-neighbour
//!   lookup) and the QP Newton iteration all follow REST's production
//!   low-rank path (`ri_gw::contour_rayon_lowrank`, `generate_real_axis_vchiv`,
//!   `newton_solver_lowrank`).
//! * The reference must be a restricted closed-shell density-fitted HF
//!   calculation without frozen orbitals (`consts_n = eps_n`, `B = 0`).  A
//!   guard rejects references for which the static term
//!   `B_n = (1-alpha) Sigma_x,nn - v_xc,nn` does not vanish; extending to DFT
//!   references only requires filling the `bB` covector.
//! * The orbital response (U, eps1) is the full canonical CPHF response:
//!   the CP-HF equation is solved by REST's `analdrv` driver (`RHessSCF`
//!   block-Krylov over all perturbations, RI-JK response on `rimatr`), and
//!   the remaining blocks are restored from the differentiated Roothaan
//!   equation, exactly as `pyscf/gw/gw_cd_grad*.py::_cp_response_atom`.

use crate::ri_jk::util::{get_cint_aux, get_cint_mol};
use crate::ri_rpa::{gauss_legendre_grids, logarithmic_grid, trans_gauss_legendre_grids};
use crate::scf_io::SCF;
use rest_libcint::prelude::*;
use rest_tensors::matrix::matrix_blas_lapack::{
    _dgemm_full, _dpotrf, _dsolve, _dsyev_inplace, _dtrtrs,
};
use tensors::{MathMatrix, MatrixFull};

use std::time::Instant;

use crate::ri_bse::bse_grad::grad_timing;

// sub-step accumulators of the canonical-response assembly (seconds):
// [0] j_deriv_from, [1] k_deriv_from, [2] compute_j_upper, [3] compute_k_upper
thread_local! {
    static T_TIMER: std::cell::RefCell<[f64; 4]> = const { std::cell::RefCell::new([0.0; 4]) };
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
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// raw RI tensors and their nuclear derivatives
// ---------------------------------------------------------------------------

/// Geometry-fixed raw RI data: `(mu nu | P)`, `J = (P|Q)` and the electronic
/// derivative tensors needed to assemble any nuclear derivative `dI`, `dJ`.
pub struct RawRiTensors {
    pub nao: usize,
    pub naux: usize,
    pub natm: usize,
    /// I[mu + nu*nao + P*nao*nao], col-major [nao, nao, naux]
    pub i3: Vec<f64>,
    /// electronic derivative on the first AO center,
    /// [mu + nu*nao + P*nao2 + t*nao2*naux], col-major [nao, nao, naux, 3]
    pub di1: Vec<f64>,
    /// same layout, derivative on the auxiliary center
    pub di2: Vec<f64>,
    /// J[P + Q*naux]
    pub j2: Vec<f64>,
    /// dJ1[P + Q*naux + t*naux*naux]
    pub dj1: Vec<f64>,
    /// AO index -> atom
    pub ao_atom: Vec<usize>,
    /// auxiliary AO index -> atom
    pub aux_atom: Vec<usize>,
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

    let (i3, shape3) = CInt::integrate_cross("int3c2e", [&mol, &mol, &aux], "s1", None).into();
    assert_eq!(shape3, vec![nao, nao, naux], "int3c2e s1 shape");
    let (di1, shape1) =
        CInt::integrate_cross("int3c2e_ip1", [&mol, &mol, &aux], "s1", None).into();
    assert_eq!(shape1, vec![nao, nao, naux, 3], "int3c2e_ip1 s1 shape");
    let (di2, shape2) =
        CInt::integrate_cross("int3c2e_ip2", [&mol, &mol, &aux], "s1", None).into();
    assert_eq!(shape2, vec![nao, nao, naux, 3], "int3c2e_ip2 s1 shape");

    let (j2, shapej) = aux.integrate("int2c2e", "s1", None).into();
    assert_eq!(shapej, vec![naux, naux], "int2c2e shape");
    let (dj1, shapedj) = aux.integrate("int2c2e_ip1", "s1", None).into();
    assert_eq!(shapedj, vec![naux, naux, 3], "int2c2e_ip1 shape");

    let ao_atom = atom_map_from_slices(&scf.mol.aoslice_by_atom(), nao);
    let aux_atom = atom_map_from_slices(&scf.mol.make_auxmol_fake().aoslice_by_atom(), naux);

    RawRiTensors {
        nao,
        naux,
        natm,
        i3,
        di1,
        di2,
        j2,
        dj1,
        ao_atom,
        aux_atom,
    }
}

impl RawRiTensors {
    /// Total nuclear derivative `dI_{mu nu P}/dR_{atm,comp}` (PySCF
    /// `gw_cd_grad._dI_atom`): the Gaussian nuclear derivative is minus the
    /// electronic derivative on the corresponding center.
    pub fn d_i_atom(&self, atm: usize, comp: usize) -> Vec<f64> {
        let nao = self.nao;
        let naux = self.naux;
        let nao2 = nao * nao;
        let nao2naux = nao2 * naux;
        let mut di = vec![0.0f64; nao2naux];
        let d1c = &self.di1[comp * nao2naux..(comp + 1) * nao2naux];
        let d2c = &self.di2[comp * nao2naux..(comp + 1) * nao2naux];
        for mu in 0..nao {
            if self.ao_atom[mu] != atm {
                continue;
            }
            for nu in 0..nao {
                for p in 0..naux {
                    di[mu + nu * nao + p * nao2] -= d1c[mu + nu * nao + p * nao2];
                }
            }
        }
        for nu in 0..nao {
            if self.ao_atom[nu] != atm {
                continue;
            }
            for mu in 0..nao {
                for p in 0..naux {
                    di[mu + nu * nao + p * nao2] -= d1c[nu + mu * nao + p * nao2];
                }
            }
        }
        for p in 0..naux {
            if self.aux_atom[p] != atm {
                continue;
            }
            for k in 0..nao2 {
                di[p * nao2 + k] -= d2c[p * nao2 + k];
            }
        }
        di
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

pub fn solve_linear(a: &MatrixFull<f64>, b: &[f64]) -> Vec<f64> {
    _dsolve(a, b).expect("gw_grad: singular linear system")
}

fn to_mat(v: &[f64], r: usize, c: usize) -> MatrixFull<f64> {
    MatrixFull::from_vec([r, c], v.to_vec()).unwrap()
}

/// `A^T B` for flat col-major buffers `A [ar, ac]`, `B [br, bc]`, `ar == br`.
/// The contraction runs over the FIRST dimension (REST `_dgemm_full` 'T'
/// convention), the result is `[ac, bc]`.
fn gemm_nt(a: &[f64], ar: usize, ac: usize, b: &[f64], br: usize, bc: usize) -> Vec<f64> {
    debug_assert_eq!(ar, br, "gemm_nt: contraction dimension mismatch");
    let am = to_mat(a, ar, ac);
    let bm = to_mat(b, br, bc);
    let mut out = MatrixFull::new([ac, bc], 0.0);
    _dgemm_full(&am, 'T', &bm, 'N', &mut out, 1.0, 0.0);
    out.data
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

/// `A B` for flat col-major buffers.
fn gemm_nn(a: &[f64], ar: usize, ac: usize, b: &[f64], br: usize, bc: usize) -> Vec<f64> {
    debug_assert_eq!(ac, br);
    let am = to_mat(a, ar, ac);
    let bm = to_mat(b, br, bc);
    let mut out = MatrixFull::new([ar, bc], 0.0);
    _dgemm_full(&am, 'N', &bm, 'N', &mut out, 1.0, 0.0);
    out.data
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
    LrFactor { omega, eigvec: cols, eigval, n_aux: naux }
}

impl LrFactor {
    /// Response `y = (J - Q(omega))^-1 rhs` in the raw representation:
    /// `y = S (qt_rhs + U diag(lam) U^T qt_rhs)` with `qt_rhs = S rhs`.
    /// `rhs` holds `n_col` right-hand sides of length naux (col-major).
    pub fn response_vectors(&self, s: &[f64], rhs: &[f64], n_col: usize) -> Vec<f64> {
        let naux = self.n_aux;
        let qt = gemm_nn(s, naux, naux, rhs, naux, n_col);
        if self.eigval.is_empty() {
            return qt;
        }
        let nk = self.eigval.len();
        let mut proj = gemm_nt(&self.eigvec, naux, nk, &qt, naux, n_col);
        for c in 0..n_col {
            for v in 0..nk {
                proj[v + c * nk] *= self.eigval[v];
            }
        }
        let corr = gemm_nn(&self.eigvec, naux, nk, &proj, nk, n_col);
        let mut sum = qt;
        for (a, b) in sum.iter_mut().zip(corr.iter()) {
            *a += b;
        }
        gemm_nn(s, naux, naux, &sum, naux, n_col)
    }
}

// ---------------------------------------------------------------------------
// per-geometry GW gradient engine
// ---------------------------------------------------------------------------

/// One converged QP state: energy, renormalisation factor and the response
/// data needed by the pullback.
pub struct QpCache {
    pub target: usize,
    pub omega: f64,
    pub z_factor: f64,
    /// W_c[n, m](iu) at every quadrature point [n_freq][nmo]
    pub w_imag: Vec<Vec<f64>>,
    /// (u, y[naux*nmo], y0[naux*nmo]) per quadrature point
    pub y_imag: Vec<(f64, Vec<f64>, Vec<f64>)>,
    /// (m, sign*pole_factor, z_used, y_res[naux], y0_metric[naux]) per
    /// active residue
    pub residues: Vec<(usize, f64, f64, Vec<f64>, Vec<f64>)>,
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
    /// raw three-center MO integrals (pair-symmetrised)
    pub q: Vec<f64>,
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
    /// metric-transformed integrals (same layout as `q`)
    pub qt: Vec<f64>,
    /// raw occupied-virtual block [naux, nov], ia = i + a*nocc
    pub qia: Vec<f64>,
    /// screening transition energies de_ia = e_a - e_i
    pub de_ia: Vec<f64>,
    /// imaginary-axis quadrature (omega, weight)
    pub quad: Vec<(f64, f64)>,
    /// real-axis sampling grid (REST protocol)
    pub z_grid: Vec<f64>,
    /// cached low-rank factors on the imaginary axis
    pub lr_imag: Vec<LrFactor>,
    /// cached low-rank factors on the real-axis grid
    pub lr_grid: Vec<LrFactor>,
    /// metric-transformed occupied-virtual block (kept for on-the-fly factors)
    pub qtia: Vec<f64>,
    /// hybrid parameter alpha of the reference, as used by REST's GW
    /// `consts` (`dfa_hybrid_scf`)
    pub hybrid: f64,
    /// exchange scaling of the SCF Fock response (1.0 for an HF reference,
    /// `dfa_hybrid_scf` otherwise) — matches `dft::response::gen_vind`
    pub fock_hyb: f64,
    /// v_xc diagonal in the MO basis (cached once)
    vxc_nn: Vec<f64>,
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
        let mut q = vec![0.0f64; naux * nmo * nmo];
        for p_aux in 0..naux {
            let i_block = &raw.i3[p_aux * nao * nao..(p_aux + 1) * nao * nao];
            // tmp[mu, q_mo] = sum_nu I[mu, nu, P] C[nu, q_mo]
            let mut tmp = vec![0.0f64; nao * nmo];
            for mu in 0..nao {
                for q_mo in 0..nmo {
                    let mut acc = 0.0;
                    for nu in 0..nao {
                        acc += i_block[mu + nu * nao] * c_mo[[nu, q_mo]];
                    }
                    tmp[mu + q_mo * nao] = acc;
                }
            }
            for q_mo in 0..nmo {
                for p_mo in 0..nmo {
                    let mut acc = 0.0;
                    for mu in 0..nao {
                        acc += c_mo[[mu, p_mo]] * tmp[mu + q_mo * nao];
                    }
                    q[(p_mo + q_mo * nmo) * naux + p_aux] = acc;
                }
            }
        }
        symmetrise_pairs(&mut q, naux, nmo);
        // pair-slowest repack (strided once, then contiguous for the gemms)
        let q_pairmajor: Vec<f64> = {
            let npair = nmo * nmo;
            let mut v = vec![0.0f64; naux * npair];
            for pair in 0..npair {
                for p in 0..naux {
                    v[pair + p * npair] = q[pair * naux + p];
                }
            }
            v
        };

        // ---- metric and J^{-1/2} ----
        let j_mat = MatrixFull::from_vec([naux, naux], raw.j2.clone()).unwrap();
        let mut j_chol = j_mat.clone();
        _dpotrf(&mut j_chol, 'L');
        let (vals, vecs) = eigh_symmetric(&j_mat.data, naux);
        let mut s_metric = vec![0.0f64; naux * naux];
        for a in 0..naux {
            for b in 0..naux {
                let mut acc = 0.0;
                for v in 0..naux {
                    if vals[v] > 1.0e-10 {
                        acc += vecs[a + v * naux] * vecs[b + v * naux] / vals[v].sqrt();
                    }
                }
                s_metric[a + b * naux] = acc;
            }
        }

        // ---- metric-transformed integrals ----
        let q_mat = to_mat(&q, naux, nmo * nmo);
        let qt = gemm_nn(&s_metric, naux, naux, &q_mat.data, naux, nmo * nmo);

        // ---- occupied-virtual blocks ----
        let nov = nocc * (nmo - nocc);
        let mut qia = vec![0.0f64; naux * nov];
        let mut qtia = vec![0.0f64; naux * nov];
        for a in nocc..nmo {
            for i in 0..nocc {
                let src = (i + a * nmo) * naux;
                let dst = (i + (a - nocc) * nocc) * naux;
                qia[dst..dst + naux].copy_from_slice(&q[src..src + naux]);
                qtia[dst..dst + naux].copy_from_slice(&qt[src..src + naux]);
            }
        }
        let mut de_ia: Vec<f64> = Vec::with_capacity(nov);
        for a in nocc..nmo {
            for i in 0..nocc {
                de_ia.push(e[a] - e[i]);
            }
        }
        assert_eq!(de_ia.len(), nov);

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

        // ---- REST real-axis grid (de_max scan + linear/quadratic nodes) ----
        let nsemin = (nocc - 1).saturating_sub(config.selfenergy_state_range);
        let nsemax = (nocc + config.selfenergy_state_range).min(nmo - 1);
        let z_grid = build_real_axis_grid(
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
        );

        // ---- low-rank factors ----
        if std::env::var("REST_GWGRAD_DBG").is_ok() {
            println!("[new DBG] qtia max = {}", qtia.iter().fold(0.0f64, |a, &x| a.max(x.abs())));
        }
        let lr_imag: Vec<LrFactor> = quad
            .iter()
            .map(|&(u, _)| {
                build_lr_factor(&qtia, naux, nov, &de_ia, u, 'I', 0.0, config.low_rank_tolerance)
            })
            .collect();
        let lr_grid: Vec<LrFactor> = z_grid
            .iter()
            .map(|&z| {
                build_lr_factor(&qtia, naux, nov, &de_ia, z, 'R', config.eta, config.low_rank_tolerance)
            })
            .collect();

        if std::env::var("REST_GWGRAD_DBG").is_ok() {
            println!("[new DBG] lr_imag nkeeps = {:?}", lr_imag.iter().map(|f| f.eigval.len()).collect::<Vec<_>>());
            println!("[new DBG] lr_grid nkeeps = {:?}", lr_grid.iter().map(|f| f.eigval.len()).collect::<Vec<_>>());
        }
        let hybrid = scf.mol.xc_data.dfa_hybrid_scf;
        let fock_hyb = if scf.mol.xc_data.dfa_compnt_scf.is_empty() { 1.0 } else { hybrid };
        let vxc_nn = crate::ri_gw::vxc_ao2mo(scf);

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
            q,
            q_pairmajor,
            j_mat,
            j_chol,
            s_metric,
            qt,
            qia,
            de_ia,
            quad,
            z_grid,
            lr_imag,
            lr_grid,
            qtia: qtia,
            hybrid,
            fock_hyb,
            vxc_nn,
        }
    }

    fn pair_col(&self, p: usize, q: usize) -> usize {
        p + q * self.nmo
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

    /// `J^{-1} rhs` for a single right-hand side.
    pub(crate) fn solve_metric(&self, rhs: &[f64]) -> Vec<f64> {
        let mut b = rhs.to_vec();
        self.solve_metric_batch(&mut b, 1);
        b
    }

    fn lr_factor_nearest(&self, z: f64) -> &LrFactor {
        let zs = &self.z_grid;
        let n = zs.len();
        if n <= 1 || z <= zs[0] {
            return &self.lr_grid[0];
        }
        if z >= zs[n - 1] {
            return &self.lr_grid[n - 1];
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
            &self.lr_grid[lo]
        } else {
            &self.lr_grid[hi]
        }
    }

    /// REST static term `consts_n = eps_n + (1-alpha) Sigma_x,nn - vxc_nn`
    /// with the RI exchange `Sigma_x,nn = -sum_i (n i | n i)`.
    pub fn consts(&self, n: usize) -> f64 {
        let mut exchange = 0.0;
        for i in 0..self.nocc {
            let base = self.pair_col(n, i) * self.naux;
            exchange -= self.qt[base..base + self.naux].iter().map(|x| x * x).sum::<f64>();
        }
        // For HF references REST's ctrl sets dfa_hybrid_scf = 0 while the
        // HF eigenvalues already contain the exact exchange, so the static
        // term must use the true hybrid fraction (cf. dft/response.rs).
        self.e[n] + exchange * (1.0 - self.fock_hyb) - self.vxc_nn[n]
    }

    /// One pole contribution of the contour term with REST's protocol
    /// (`pole_factor` smoothing, nearest-grid frequency, sign convention).
    #[allow(clippy::type_complexity)]
    fn add_pole(
        &self,
        m: usize,
        de: f64,
        n: usize,
        collect: bool,
        contour: &mut f64,
        residues: &mut Vec<(usize, f64, f64, Vec<f64>, Vec<f64>)>,
    ) {
        if de < -self.config.res_tol {
            return;
        }
        let pole_factor = if de.abs() < self.config.res_tol { 0.5 } else { 1.0 };
        let sign = if m < self.nocc { -1.0 } else { 1.0 };
        let z_used = if self.config.exact_residue_z {
            de.abs()
        } else {
            nearest_grid_value(&self.z_grid, de.abs())
        };
        // exact mode: factor at the true residue frequency (fresh
        // diagonalisation); quantised mode: nearest grid factor
        let factor_on_the_fly;
        let factor = if self.config.exact_residue_z {
            factor_on_the_fly = build_lr_factor(
                &self.qtia, self.naux, self.de_ia.len(), &self.de_ia,
                z_used, 'R', self.config.eta, self.config.low_rank_tolerance,
            );
            &factor_on_the_fly
        } else {
            self.lr_factor_nearest(z_used)
        };
        let qnm = self.gather_pair(&self.q, n, m);
        let y = factor.response_vectors(&self.s_metric, &qnm, 1);
        let y0 = self.metric_solve(&qnm, 1);
        let w: f64 = qnm
            .iter()
            .zip(y.iter())
            .zip(y0.iter())
            .map(|((&a, &b), &c)| a * (b - c))
            .sum();
        *contour += sign * pole_factor * w;
        if collect {
            // the signed factor carries pole_factor so that every consumer
            // (value, d/domega zeta term, pullback) stays consistent
            residues.push((m, sign * pole_factor, z_used, y, y0));
        }
    }

    /// All pieces of the self-energy; optionally collects the response data
    /// needed for the gradient.  Returns `(contour, imag, w_imag, y_imag,
    /// residues)`.
    #[allow(clippy::type_complexity)]
    fn sigma_c_parts(
        &self,
        omega: f64,
        n: usize,
        collect: bool,
    ) -> (
        f64,
        f64,
        Vec<Vec<f64>>,
        Vec<(f64, Vec<f64>, Vec<f64>)>,
        Vec<(usize, f64, f64, Vec<f64>, Vec<f64>)>,
    ) {
        let ef = 0.5 * (self.e[self.nocc - 1] + self.e[self.nocc]);
        // imaginary-axis part
        let mut imag = 0.0f64;
        let mut w_imag_all: Vec<Vec<f64>> = Vec::with_capacity(self.quad.len());
        let mut y_imag_all: Vec<(f64, Vec<f64>, Vec<f64>)> = Vec::new();
        for (wi, &(u, wgt)) in self.quad.iter().enumerate() {
            let (w, y, y0) = self.screened_row(Some(wi), u, n);
            let mut sum_m = 0.0;
            for m in 0..self.nmo {
                let t = omega - self.e[m];
                sum_m += 2.0 * t / (t * t + u * u) * w[m];
            }
            imag += sum_m * wgt / TWO_PI;
            w_imag_all.push(w);
            if collect {
                y_imag_all.push((u, y, y0));
            }
        }
        // contour (residue) part with REST pole selection and nearest grid
        let mut contour = 0.0f64;
        let mut residues: Vec<(usize, f64, f64, Vec<f64>, Vec<f64>)> = Vec::new();
        if omega > ef {
            for a in self.nocc..self.nmo {
                let de = omega - self.e[a];
                self.add_pole(a, de, n, collect, &mut contour, &mut residues);
            }
        } else {
            for i in 0..self.nocc {
                let de = self.e[i] - omega;
                self.add_pole(i, de, n, collect, &mut contour, &mut residues);
            }
        }
        (contour, imag, w_imag_all, y_imag_all, residues)
    }

    /// `W_c[n, m](z)` for all m plus the exact response vectors, evaluated
    /// through a low-rank factor (imaginary axis or nearest real grid point).
    fn screened_row(&self, wi: Option<usize>, z: f64, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let factor = match wi {
            Some(w) => &self.lr_imag[w],
            None => self.lr_factor_nearest(z),
        };
        let qn = self.gather_row(&self.q, n);
        let y = factor.response_vectors(&self.s_metric, &qn, self.nmo);
        let y0 = self.metric_solve(&qn, self.nmo);
        let mut w = vec![0.0f64; self.nmo];
        for m in 0..self.nmo {
            let base = m * self.naux;
            w[m] = qn[base..base + self.naux]
                .iter()
                .zip(y[base..base + self.naux].iter())
                .zip(y0[base..base + self.naux].iter())
                .map(|((&a, &b), &c)| a * (b - c))
                .sum();
        }
        (w, y, y0)
    }

    /// Raw CD self-energy `Sigma_c(omega) = contour(omega) - imag(omega)`.
    pub fn sigma_c(&self, omega: f64, n: usize) -> f64 {
        let (contour, imag, _, _, _) = self.sigma_c_parts(omega, n, false);
        contour - imag
    }

    /// Decomposed self-energy at fixed omega (for validation).
    pub fn sigma_c_decomposed(&self, omega: f64, n: usize) -> (f64, f64) {
        let (contour, imag, _, _, _) = self.sigma_c_parts(omega, n, false);
        (contour, imag)
    }

    /// `d imag(omega)/d omega` at fixed geometry.
    fn imag_d_omega(&self, omega: f64, n: usize) -> f64 {
        let mut d = 0.0f64;
        for (wi, &(u, wgt)) in self.quad.iter().enumerate() {
            let (w, _y, _y0) = self.screened_row(Some(wi), u, n);
            let mut sum_m = 0.0;
            for m in 0..self.nmo {
                let t = omega - self.e[m];
                sum_m += 2.0 * (u * u - t * t) / ((t * t + u * u) * (t * t + u * u)) * w[m];
            }
            d += sum_m * wgt / TWO_PI;
        }
        d
    }

    /// Omega-derivative pieces from already-collected response data:
    /// `w_imag[wi][m]` from the imaginary pass and the residue vectors.
    fn sigma_d_from_parts(
        &self,
        omega: f64,
        w_imag: &[Vec<f64>],
        residues: &[(usize, f64, f64, Vec<f64>, Vec<f64>)],
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
        }
        if self.config.exact_residue_z {
            let nov = self.de_ia.len();
            for &(m, sign, z_used, ref y_r, ref _y0) in residues {
                let zsign = if self.e[m] - omega >= 0.0 { 1.0 } else { -1.0 };
                let mut sum = 0.0f64;
                for col in 0..nov {
                    let mut proj = 0.0f64;
                    for p in 0..self.naux {
                        proj += y_r[p] * self.qia[p + col * self.naux];
                    }
                    sum += proj * proj
                        * dd_response_dz_single(self.de_ia[col], z_used, self.config.eta);
                }
                d += sign * (-zsign) * sum;
                let _ = m;
            }
        }
        d
    }

    /// `Sigma_c` and `d Sigma_c / d omega` in one pass: the imaginary-axis
    /// screened rows and the pole factors are shared between the value and
    /// the derivative (the Newton solver evaluates both at every step).
    pub fn sigma_c_and_d(&self, omega: f64, n: usize) -> (f64, f64) {
        let (contour, imag, w_imag, _y_imag, residues) = self.sigma_c_parts(omega, n, true);
        let ds = self.sigma_d_from_parts(omega, &w_imag, &residues);
        (contour - imag, ds)
    }

    /// `d Sigma_c / d omega` at fixed geometry.  In the REST quantised
    /// protocol the residue frequencies are piecewise constant, so only the
    /// imaginary-axis part contributes away from orbital thresholds; in the
    /// exact mode the moving residue frequencies add the standard zeta term.
    pub fn d_sigma_d_omega(&self, omega: f64, n: usize) -> f64 {
        let mut d = -self.imag_d_omega(omega, n);
        if self.config.exact_residue_z {
            let (_c, _i, _w, _yi, res) = self.sigma_c_parts(omega, n, true);
            let nov = self.de_ia.len();
            for &(m, sign, z_used, ref y_r, ref _y0) in &res {
                let zsign = if self.e[m] - omega >= 0.0 { 1.0 } else { -1.0 };
                // dW/dz = 2 sum_ia proj_ia^2 dd_ia/dz with proj = y^T q_ia
                let mut sum = 0.0f64;
                for col in 0..nov {
                    let mut proj = 0.0f64;
                    for p in 0..self.naux {
                        proj += y_r[p] * self.qia[p + col * self.naux];
                    }
                    sum += proj * proj
                        * dd_response_dz_single(self.de_ia[col], z_used, self.config.eta);
                }
                // dzeta/domega = -zsign
                d += sign * (-zsign) * sum;
                let _ = m;
            }
        }
        d
    }

    /// Solve the QP equation with REST's finite-difference Newton iteration
    /// and assemble the full gradient cache at the converged root.
    pub fn build_target_cache(&self, n: usize) -> QpCache {
        let consts = self.consts(n);
        let side = if n < self.nocc { -1.0 } else { 1.0 };
        let h = 1.0e-6;
        let mut x = self.e[n] + side * 0.02;
        let f = |om: f64| consts + self.sigma_c(om, n) - om;
        // Smooth exact-residue mode: solve with the analytic derivative
        // directly (robust; the finite-difference derivative is only needed
        // to mirror REST's production Newton on the quantised protocol).
        if self.config.exact_residue_z {
            let mut converged = 0;
            let mut iter = 0;
            while converged == 0 && iter < self.config.qpe_max_iter {
                // value and derivative share one response evaluation
                let (sc, dsc) = self.sigma_c_and_d(x, n);
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
                println!("warning!!! gw_grad Newton (exact mode) did not converge for orbital {}!", n);
            }
            let omega = x;
            let (_contour, _imag, w_imag, y_imag, residues) = self.sigma_c_parts(omega, n, true);
            let dsdw = self.sigma_d_from_parts(omega, &w_imag, &residues);
            return QpCache {
                target: n,
                omega,
                z_factor: 1.0 / (1.0 - dsdw),
                w_imag,
                y_imag,
                residues,
            };
        }
        let mut y = f(x);
        let mut yp = f(x + h);
        let mut ym = f(x - h);
        let mut converged = 0;
        let mut iter = 0;
        while converged == 0 && iter < self.config.qpe_max_iter {
            let deriv = (yp - ym) / (2.0 * h);
            let shift = -y / deriv;
                x += shift;
            y = f(x);
            if shift.abs() < self.config.qpe_tol {
                converged += 1;
            }
            yp = f(x + h);
            ym = f(x - h);
            iter += 1;
        }
        if converged == 0 {
            // The finite-difference derivative can stall on the jumps of the
            // piecewise-constant residue protocol; retry with the analytic
            // derivative of the QP equation.
            let df = |om: f64| self.d_sigma_d_omega(om, n) - 1.0;
            let mut x2 = x;
            for it2 in 0..self.config.qpe_max_iter {
                let fv = f(x2);
                let dv = df(x2);
                let shift = -fv / dv;
                if std::env::var("REST_GWGRAD_NEWTON").is_ok() {
                    println!("[newton2 DBG] n={} it={} x={:.10} f={:+.3e} shift={:+.3e}",
                        n, it2, x2, fv, shift);
                }
                x2 += shift;
                if shift.abs() < self.config.qpe_tol {
                    converged += 1;
                    break;
                }
            }
            x = x2;
        }
        if converged == 0 {
            println!("warning!!! gw_grad Newton solver did not converge for orbital {}!", n);
        }
        let omega = x;

        let (_contour, _imag, w_imag, y_imag, residues) = self.sigma_c_parts(omega, n, true);
        let dsdw = self.sigma_d_from_parts(omega, &w_imag, &residues);
        QpCache {
            target: n,
            omega,
            z_factor: 1.0 / (1.0 - dsdw),
            w_imag,
            y_imag,
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
        let naux = self.naux;
        let nmo = self.nmo;
        let nocc = self.nocc;
        let nov = nocc * (nmo - nocc);
        let pair = naux * nmo * nmo;
        let dbg = |name: &str| std::env::var(name).is_ok();
        let no_be_expl = dbg("REST_GWGRAD_NOBEEXPL");
        let no_exch = dbg("REST_GWGRAD_NOEXCH");
        let no_res = dbg("REST_GWGRAD_NORES");
        let no_imag = dbg("REST_GWGRAD_NOIMAG");

        let mut bq: Vec<Vec<f64>> = vec![vec![0.0; pair]; nk];
        let mut bj: Vec<Vec<f64>> = vec![vec![0.0; naux * naux]; nk];
        let mut be: Vec<Vec<f64>> = vec![vec![0.0; nmo]; nk];
        let mut bb: Vec<Vec<f64>> = vec![vec![0.0; nmo]; nk];

        // Polarizability-coefficient matrices V[k] = sum_m a_m y_m y_m^T are
        // accumulated over ALL QP targets first; their contraction with q_ia
        // (the expensive polarizability pullback) runs once per (k, frequency)
        // at the end — the PySCF low_rank_weighted_screening structure.
        let nquad = self.quad.len();
        let mut v_imag: Vec<Vec<Vec<f64>>> =
            vec![vec![vec![0.0; naux * naux]; nquad]; nk];
        let mut z_order: Vec<Vec<f64>> = vec![Vec::new(); nk];
        let mut v_res: Vec<Vec<Vec<f64>>> = vec![Vec::new(); nk];

        let t_cache = Instant::now();
        for (j, cache) in caches.iter().enumerate() {
            let n = cache.target;
            for k in 0..nk {
                let c = weights[k][j] * cache.z_factor;
                // orbitals without QP weight (e.g. virtuals above the BSE
                // cutoff, holding dummy caches) contribute nothing
                if c == 0.0 {
                    continue;
                }

                // d consts/dR = eps1[n] + b_x[n]
                be[k][n] += c;
                bb[k][n] += c;

                // ---- imaginary-axis integral ----
                // Sigma_c contains -imag, so the coefficient of dWc[n,m] is
                //   a_m = -c w/(2 pi) g(t_m, u),
                // and the explicit eps1 dependence of t_m contributes
                //   be[m] += c w/(2 pi) g'(t_m) Wc[n,m].
                for (wi, &(u, wgt)) in self.quad.iter().enumerate().take(if no_imag { 0 } else { usize::MAX }) {
                    let (_u, y, y0) = &cache.y_imag[wi];
                    // J covector  bj += sum_m a_m (y0_m y0_m^T - y_m y_m^T)
                    // and dQ matrix V += sum_m a_m y_m y_m^T as BLAS-3
                    // products with the sign-sorted column scaling
                    // sqrt(|a_m|); the two sign groups are accumulated
                    // separately because outer products cannot carry a sign.
                    let mut y0p = vec![0.0f64; naux * nmo];
                    let mut yp = vec![0.0f64; naux * nmo];
                    let mut y0m = vec![0.0f64; naux * nmo];
                    let mut ym = vec![0.0f64; naux * nmo];
                    for m in 0..nmo {
                        let t = cache.omega - self.e[m];
                        let a = -c * wgt / TWO_PI * 2.0 * t / (t * t + u * u);
                        let base = m * naux;
                        // dq covector of pair (n, m) and explicit eps1 term
                        let col_m = self.pair_col(n, m) * naux;
                        for p in 0..naux {
                            bq[k][col_m + p] += 2.0 * a * (y[base + p] - y0[base + p]);
                        }
                        let gprime =
                            2.0 * (u * u - t * t) / ((t * t + u * u) * (t * t + u * u));
                        if !no_be_expl {
                            be[k][m] += c * wgt / TWO_PI * gprime * cache.w_imag[wi][m];
                        }
                        let s = a.abs().sqrt();
                        let (d0, dy) = if a >= 0.0 { (&mut y0p, &mut yp) } else { (&mut y0m, &mut ym) };
                        for r in 0..naux {
                            d0[base + r] = s * y0[base + r];
                            dy[base + r] = s * y[base + r];
                        }
                    }
                    // bj += Y0p Y0p^T - Y0m Y0m^T - (Yp Yp^T - Ym Ym^T)
                    {
                        let m0p = to_mat(&y0p, naux, nmo);
                        let m0m = to_mat(&y0m, naux, nmo);
                        let mp = to_mat(&yp, naux, nmo);
                        let mm = to_mat(&ym, naux, nmo);
                        let mut acc = MatrixFull::new([naux, naux], 0.0);
                        _dgemm_full(&m0p, 'N', &m0p, 'T', &mut acc, 1.0, 1.0);
                        _dgemm_full(&m0m, 'N', &m0m, 'T', &mut acc, -1.0, 1.0);
                        _dgemm_full(&mp, 'N', &mp, 'T', &mut acc, -1.0, 1.0);
                        _dgemm_full(&mm, 'N', &mm, 'T', &mut acc, 1.0, 1.0);
                        for (d, s) in bj[k].iter_mut().zip(acc.data.iter()) {
                            *d += s;
                        }
                    }
                    // v += Yp Yp^T - Ym Ym^T
                    {
                        let v = &mut v_imag[k][wi];
                        let mp = to_mat(&yp, naux, nmo);
                        let mm = to_mat(&ym, naux, nmo);
                        let mut acc = MatrixFull::new([naux, naux], 0.0);
                        _dgemm_full(&mp, 'N', &mp, 'T', &mut acc, 1.0, 1.0);
                        _dgemm_full(&mm, 'N', &mm, 'T', &mut acc, -1.0, 1.0);
                        for (d, s) in v.iter_mut().zip(acc.data.iter()) {
                            *d += s;
                        }
                    }
                }

                // ---- residues (quantised frequencies; no d_zeta term) ----
                for &(m, sign, z_used, ref y_r, ref y0_r) in cache.residues.iter().take(if no_res { 0 } else { usize::MAX }) {
                    let a = c * sign;
                    // exact mode: the moving residue frequency contributes
                    // dzeta/dR = zsign * eps1[m] through dW/dz
                    if self.config.exact_residue_z {
                        let zsign = if self.e[m] - cache.omega >= 0.0 { 1.0 } else { -1.0 };
                        let mut sum = 0.0f64;
                        for col in 0..nov {
                            let mut proj = 0.0f64;
                            for p in 0..naux {
                                proj += y_r[p] * self.qia[p + col * naux];
                            }
                            sum += proj * proj
                                * dd_response_dz_single(self.de_ia[col], z_used, self.config.eta);
                        }
                        be[k][m] += c * sign * zsign * sum;
                    }
                    // dq covector of pair (n, m)
                    let col_m = self.pair_col(n, m) * naux;
                    for p in 0..naux {
                        bq[k][col_m + p] += 2.0 * a * (y_r[p] - y0_r[p]);
                    }
                    for p in 0..naux {
                        for q in 0..naux {
                            bj[k][p + q * naux] += a * (y0_r[p] * y0_r[q] - y_r[p] * y_r[q]);
                        }
                    }
                    // accumulate the V bucket of this residue frequency
                    let idx = match z_order[k].iter().position(|&z| (z - z_used).abs() < 1.0e-14) {
                        Some(i) => i,
                        None => {
                            z_order[k].push(z_used);
                            v_res[k].push(vec![0.0f64; naux * naux]);
                            z_order[k].len() - 1
                        }
                    };
                    let v = &mut v_res[k][idx];
                    for p in 0..naux {
                        let yp = y_r[p];
                        for q in 0..naux {
                            v[p + q * naux] += a * yp * y_r[q];
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
                if (1.0 - self.fock_hyb).abs() > 1.0e-14 && !no_exch {
                    for i in 0..nocc {
                        let qni = self.gather_pair(&self.q, n, i);
                        let g_ni = self.metric_solve(&qni, 1);
                        let col_i = self.pair_col(n, i) * naux;
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

        // ---- deferred polarizability pullbacks ----
        let t_cache = t_cache.elapsed().as_secs_f64();
        let t_polar = Instant::now();
        let imag_factors: Vec<(Vec<f64>, Vec<f64>)> = self
            .quad
            .iter()
            .map(|&(u, _)| {
                let mut d = vec![0.0f64; nov];
                let mut dd = vec![0.0f64; nov];
                for (idx, &de) in self.de_ia.iter().enumerate() {
                    d[idx] = -4.0 * de / (de * de + u * u);
                    dd[idx] =
                        -4.0 * (u * u - de * de) / ((de * de + u * u) * (de * de + u * u));
                }
                (d, dd)
            })
            .collect();
        for k in 0..nk {
            for (wi, _) in self.quad.iter().enumerate().take(if no_imag { 0 } else { usize::MAX }) {
                let (d, dd) = &imag_factors[wi];
                self.polar_pullback(&mut bq, &mut be, k, &v_imag[k][wi], d, dd);
            }
            if !no_res {
                for (zi, &z) in z_order[k].iter().enumerate() {
                    let (d, dd) = real_response_and_deriv(&self.de_ia, z, self.config.eta);
                    self.polar_pullback(&mut bq, &mut be, k, &v_res[k][zi], &d, &dd);
                }
            }
        }

        // symmetrise the covectors (minimum-norm symmetric representation)
        for k in 0..nk {
            symmetrise_pairs(&mut bq[k], naux, nmo);
            symmetrise_square(&mut bj[k], naux);
        }
        if grad_timing() {
            eprintln!(
                "[gw_grad timing]   qp_pullback: cache loop {:8.2}s  polar pullbacks {:8.2}s",
                t_cache,
                t_polar.elapsed().as_secs_f64()
            );
        }
        (bq, bj, be, bb)
    }

    /// Add the pullback of `Tr[V dQ]` to `bq` and `be`, with
    /// `Q = sum_ia q_ia d_ia q_ia^T` and `deps_ia = deps_a - deps_i`.
    fn polar_pullback(
        &self,
        bq: &mut [Vec<f64>],
        be: &mut [Vec<f64>],
        k: usize,
        v: &[f64],
        d: &[f64],
        dd: &[f64],
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
            let col_pair = self.pair_col(i, a) * naux;
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
    /// `qx = C^T dI C + U^T q + q U` (pair-symmetrised).
    pub fn qx_from_u(&self, di: &[f64], u: &[f64]) -> Vec<f64> {
        let nao = self.nao;
        let naux = self.naux;
        let nmo = self.nmo;
        let n2 = nao * nao;
        let npair = nmo * nmo;
        let mut qx = vec![0.0f64; naux * npair];

        // ---- dI part: qx[P, p + q*nmo] = (C^T dI_p C)[p, q], per P gemms ----
        {
            let mut ip = MatrixFull::new([nao, nao], 0.0);
            let mut tmp = MatrixFull::new([nao, nmo], 0.0);
            let mut sl = MatrixFull::new([nmo, nmo], 0.0);
            for p in 0..naux {
                ip.data.copy_from_slice(&di[p * n2..(p + 1) * n2]);
                _dgemm_full(&ip, 'N', &self.c_mo, 'N', &mut tmp, 1.0, 0.0);
                _dgemm_full(&self.c_mo, 'T', &tmp, 'N', &mut sl, 1.0, 0.0);
                for col in 0..npair {
                    qx[p + col * naux] = sl.data[col];
                }
            }
        }
        // ---- U part: U^T q + q U.  q is pair-symmetric, so
        //      `(qU)[p,q] = (U^T q)[q,p]`; one large gemm of the pair-major
        //      [nmo, nmo*naux] view of q with U^T and the transposed pair add.
        //      t1 is [nmo, nmo*naux] col-major (pair blocks per P); qx stays
        //      in its native [naux, nmo*nmo] col-major layout ----
        let um = to_mat(u, nmo, nmo);
        let q2 = to_mat(&self.q_pairmajor, nmo, nmo * naux);
        let mut t1 = MatrixFull::new([nmo, nmo * naux], 0.0);
        _dgemm_full(&um, 'T', &q2, 'N', &mut t1, 1.0, 0.0);
        for p in 0..naux {
            let base = p * npair;
            for pj in 0..npair {
                let pj_swap = (pj % nmo) * nmo + pj / nmo;
                qx[p + pj * naux] += t1.data[base + pj] + t1.data[base + pj_swap];
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
        qx
    }
}

// ---------------------------------------------------------------------------
// CPHF response and gradient driver
// ---------------------------------------------------------------------------

/// Frozen-density RI Coulomb derivative `d J[dm] / dR` (PySCF
/// `gw_cd_grad._j_deriv_from`).
fn j_deriv_from(engine: &GwGradEngine, di: &[f64], dj: &[f64], dm: &[f64]) -> Vec<f64> {
    j_deriv_from_impl(engine.nao, engine.naux, &engine.raw.i3, di, dj, dm, &|r| {
        engine.solve_metric(r)
    })
}

fn j_deriv_from_impl(
    nao: usize,
    naux: usize,
    i3: &[f64],
    di: &[f64],
    dj: &[f64],
    dm: &[f64],
    solve: &dyn Fn(&[f64]) -> Vec<f64>,
) -> Vec<f64> {
    let n2 = nao * nao;
    // rho[P] = sum_k I[k, P] dm[k]: contract the natural [n2, naux] views
    let i3v = to_mat(i3, n2, naux);
    let div = to_mat(di, n2, naux);
    let dmv = to_mat(dm, n2, 1);
    let mut rho = MatrixFull::new([naux, 1], 0.0);
    _dgemm_full(&i3v, 'T', &dmv, 'N', &mut rho, 1.0, 0.0);
    let mut drho = MatrixFull::new([naux, 1], 0.0);
    _dgemm_full(&div, 'T', &dmv, 'N', &mut drho, 1.0, 0.0);
    let c = solve(&rho.data);
    let djc = gemm_nn(dj, naux, naux, &c, naux, 1);
    let rhs: Vec<f64> = drho.data.iter().zip(djc.iter()).map(|(a, b)| a - b).collect();
    let dc = solve(&rhs);
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
    k_deriv_from_impl(
        engine.nao,
        engine.naux,
        engine.nocc,
        &engine.raw.i3,
        di,
        dj,
        cocc,
        &|r| engine.solve_metric(r),
    )
}

fn k_deriv_from_impl(
    nao: usize,
    naux: usize,
    nocc: usize,
    i3: &[f64],
    di: &[f64],
    dj: &[f64],
    cocc: &[f64],
    solve: &dyn Fn(&[f64]) -> Vec<f64>,
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
    solve_metric_flat(&mut yt_b, naux, ncol, solve);
    let djm = to_mat(dj, naux, naux);
    let g_pre = to_mat(&yt_b, naux, ncol);
    let mut djg = MatrixFull::new([naux, ncol], 0.0);
    _dgemm_full(&djm, 'N', &g_pre, 'N', &mut djg, 1.0, 0.0);
    let mut djg_flat = djg.data;
    solve_metric_flat(&mut djg_flat, naux, ncol, solve);

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
    /// `pyscf/gw/gw_cd_grad*.py::_cp_response_atom` and `b_x` is the
    /// derivative of the static term `B_nn` (identically zero for an HF
    /// reference, which is the only supported reference).
    pub fn canonical_response_batch(
        &self,
    ) -> Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        use crate::analdrv::prelude::{
            AnalDrvConfig, HessNucAPI, RHessCoreAPI, RHessElecInteractAPI, RHessHcore,
            RHessOvlp, RHessSCF,
        };
        use crate::ri_jk::hess_r::RHessRIJK;
        use crate::ri_jk::util::{get_cint_aux, get_cint_mol};
        use rstsr::prelude::*;

        let nao = self.nao;
        let nmo = self.nmo;
        let nocc = self.nocc;
        let natm = self.natm;

        // reference guard: the v_xc relaxation of the static term is not
        // implemented, so only exchange-only references are supported
        let mut vxc_abs = 0.0f64;
        for v in &self.vxc_nn[..self.nmo] {
            vxc_abs = vxc_abs.max(v.abs());
        }
        assert!(
            vxc_abs < 1.0e-12,
            "gw_grad: reference has a nonzero v_xc (|v_xc| = {:.3e}); the \
             derivative of B_n = (1-alpha) Sigma_x - v_xc is only implemented \
             for exchange-only (HF) references",
            vxc_abs
        );

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
        let is_hf = self.scf.mol.xc_data.dfa_compnt_scf.is_empty();
        assert!(is_hf, "gw_grad: analdrv CP-HF path supports HF references only");
        let mut rijk_obj = RHessRIJK::new_with_cderi(&mol, &aux, 1.0, 1.0, cderi, j2c_decomp);

        let config = AnalDrvConfig::default();
        let nuc_list: Vec<&mut dyn HessNucAPI> = Vec::new();
        let core_list: Vec<&mut dyn RHessCoreAPI> = vec![&mut hcore_obj];
        let el_list: Vec<&mut dyn RHessElecInteractAPI> = vec![&mut rijk_obj];
        let mut hess = RHessSCF::new(
            mo_coeff, mo_occ, mo_energy, &mut ovlp_obj, nuc_list, core_list, el_list,
            &config,
        );
        let t_prep0 = Instant::now();
        hess.make_response_preparation();
        let t_prep = t_prep0.elapsed().as_secs_f64();

        // ---- dimensionless CP-HF right-hand side (analdrv skeleton route)
        //      and one block-Krylov solve over all perturbations ----
        let t_rhs0 = Instant::now();
        let rhs_map = hess.compute_dimless_cphf_rhs();
        let t_rhs = t_rhs0.elapsed().as_secs_f64();
        let t_solve0 = Instant::now();
        let mo1_tsr = hess.solve_dimless_cphf(rhs_map["rhs"].view()); // [nmo, nocc, 3, natm]
        let t_solve = t_solve0.elapsed().as_secs_f64();

        // ---- canonical-orbital assembly per perturbation ----
        // The analdrv tensors order perturbations as col-major [.., .., 3, natm],
        // i.e. k_pert = comp + atm*3.  The full-MO skeleton Fock/overlap
        // derivatives needed by the Roothaan restoration (all nmo columns)
        // are not available from the optimized RI-JK hessian object (only
        // `get_deriv1_bra` is implemented), so they come from the raw-RI
        // route here.
        let mo1_data = &mo1_tsr.raw()[mo1_tsr.offset()..];
        let n2 = nao * nao;
        let mut responses = Vec::with_capacity(natm * 3);
        let (mut t_dij, mut t_skel, mut t_fin) = (0.0f64, 0.0f64, 0.0f64);
        T_TIMER.with(|c| c.borrow_mut().fill(0.0));
        for atm in 0..natm {
            let t0 = Instant::now();
            let h1r = crate::hessian::rhf::build_hcore_first_deriv(&self.scf.mol, atm);
            let s1r = crate::ri_cphf::cphf_solver_pyscf::build_s1ao_deriv(&self.scf.mol, atm);
            t_dij += t0.elapsed().as_secs_f64();
            for comp in 0..3 {
                let k_pert = atm * 3 + comp;
                let t0 = Instant::now();
                let di = self.raw.d_i_atom(atm, comp);
                let dj = self.raw.d_j_atom(atm, comp);
                t_dij += t0.elapsed().as_secs_f64();
                let t0 = Instant::now();
                let (f1_skel_mo, s1_mo) =
                    self.canonical_skeleton_mo(atm, comp, &h1r, &s1r, &di, &dj);
                t_skel += t0.elapsed().as_secs_f64();
                let mo1_slab = &mo1_data[k_pert * nmo * nocc..(k_pert + 1) * nmo * nocc];
                let t0 = Instant::now();
                responses.push(self.canonical_finish(&f1_skel_mo, &s1_mo, mo1_slab));
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
        responses
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
    ) -> (Vec<f64>, Vec<f64>) {
        let nao = self.nao;
        let nmo = self.nmo;
        let nocc = self.nocc;
        let n2 = nao * nao;

        let dm = self.scf.density_matrix[0].data.clone();
        let cocc: Vec<f64> = (0..nocc)
            .flat_map(|i| (0..nao).map(move |mu| self.c_mo[[mu, i]]))
            .collect();
        let vfrozen = {
            let t0 = Instant::now();
            let jd = j_deriv_from(self, di, dj, &dm);
            timer_add(0, t0);
            let t0 = Instant::now();
            let kd = k_deriv_from(self, di, dj, &cocc);
            timer_add(1, t0);
            jd.iter().zip(kd.iter()).map(|(a, b)| a - 0.5 * b).collect::<Vec<f64>>()
        };
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
        (f1_skel_mo, s1_mo)
    }

    /// Canonical-orbital response `(u, eps1, b_x)` of one perturbation from
    /// the analdrv CP-HF solution: relaxed first-order density, complete
    /// first-order Fock and the Roothaan-style orbital rotations.
    fn canonical_finish(
        &self,
        f1_skel_mo: &[f64], // skeleton F1 in MO basis [nmo*nmo], col-major
        s1_mo: &[f64],      // overlap derivative in MO basis [nmo*nmo], col-major
        mo1: &[f64],        // analdrv CP-HF solution [nmo*nocc], col-major
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
        let k_up = crate::dft::response::compute_k_upper(self.scf, &vec![dm1_m]);
        timer_add(3, t0);
        let j_full = j_up.to_matrixfull().expect("gw_grad: J upper").data;
        let k_full = k_up.to_matrixfull().expect("gw_grad: K upper").data;

        // relaxed part of the first-order Fock in the AO basis ...
        let mut f1_relaxed = vec![0.0f64; n2];
        for k in 0..n2 {
            f1_relaxed[k] = j_full[k] - 0.5 * self.fock_hyb * k_full[k];
        }
        // ... transformed to MO and added to the skeleton derivative
        let f1_mo: Vec<f64> = {
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

        let mut u = vec![0.0f64; nmo * nmo];
        let mut eps1 = vec![0.0f64; nmo];
        for p in 0..nmo {
            u[p + p * nmo] = -0.5 * s1_mo[p + p * nmo];
            for q in 0..nmo {
                if p != q {
                    u[p + q * nmo] = (f1_mo[p + q * nmo] - self.e[q] * s1_mo[p + q * nmo])
                        / (self.e[q] - self.e[p]);
                }
            }
            eps1[p] = f1_mo[p + p * nmo] - self.e[p] * s1_mo[p + p * nmo];
        }
        let mut orth = 0.0f64;
        for p in 0..nmo {
            for q in 0..nmo {
                orth = orth.max((u[p + q * nmo] + u[q + p * nmo] + s1_mo[p + q * nmo]).abs());
            }
        }
        // diffuse bases amplify the CP-HF Krylov residual by the smallest
        // orbital gap, so the canonical response can lose a little more than
        // machine precision without any degeneracy problem
        if orth > 1.0e-2 {
            panic!(
                "gw_grad: canonical response orthogonality error {:.3e} (degenerate orbitals?)",
                orth
            );
        }
        if orth > 1.0e-6 {
            println!("warning!!! gw_grad canonical response orthogonality error {:.3e}", orth);
        }
        (u, eps1, vec![0.0f64; nmo])
    }

    /// Analytic gradient of one LR-CD G0W0 QP energy.
    ///
    /// Returns `(grad, omega, z_factor)` with `grad[atm*3 + comp]` in
    /// Hartree/Bohr (row-major over atoms and Cartesian components).
    pub fn analytic_gradient(&self, target: usize) -> (Vec<f64>, f64, f64) {
        let cache = self.build_target_cache(target);
        let (bq, bj, be, bb) = self.qp_pullback(&[&cache], &[vec![1.0]]);
        let responses = self.canonical_response_batch();
        let mut grad = vec![0.0f64; self.natm * 3];
        for atm in 0..self.natm {
            for comp in 0..3 {
                let di = self.raw.d_i_atom(atm, comp);
                let dj = self.raw.d_j_atom(atm, comp);
                let (u, eps1, b_x) = responses[atm * 3 + comp].clone();
                let qx = self.qx_from_u(&di, &u);
                let g = self.contract_perturbation(&bq, &bj, &be, &bb, &qx, &dj, &eps1, &b_x);
                grad[atm * 3 + comp] = g[0];
            }
        }
        (grad, cache.omega, cache.z_factor)
    }

    /// QP energy of one orbital (the scalar whose gradient is returned by
    /// [`GwGradEngine::analytic_gradient`]).
    pub fn qp_energy_of(&self, target: usize) -> f64 {
        self.build_target_cache(target).omega
    }
}

// ---------------------------------------------------------------------------
// free helpers
// ---------------------------------------------------------------------------

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

/// REST real-axis grid construction: de_max scan + linear/quadratic nodes.
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
) -> Vec<f64> {
    let nmo = e.len();
    let mut de_max = 0.0f64;
    for mstate in nsemin..=nsemax {
        let e0 = e[mstate];
        for io in -(nomega_sigma as isize)..=(nomega_sigma as isize) {
            let omega = e0 + io as f64 * step_sigma;
            for p in 0..nocc {
                let de = e[p] - omega;
                if de > res_tol {
                    de_max = de_max.max(de);
                }
            }
            for a in nocc..nmo {
                let de = omega - e[a];
                if de > res_tol {
                    de_max = de_max.max(de);
                }
            }
        }
    }
    de_max = de_max * 1.05 + 0.1;
    let grid_scale = if omega_chi_max > 0.0 { omega_chi_max } else { de_max };
    (0..nomega_chi_real)
        .map(|i| {
            if nomega_chi_real > 1 {
                let t = i as f64 / (nomega_chi_real - 1) as f64;
                if grid_type == 1 {
                    grid_scale * t * t
                } else {
                    grid_scale * t
                }
            } else {
                grid_scale
            }
        })
        .collect()
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
        let um = to_mat(u, nmo, nmo);
        let q2 = to_mat(&q_pm, nmo, nmo * naux);
        let mut t1 = MatrixFull::new([nmo, nmo * naux], 0.0);
        _dgemm_full(&um, 'T', &q2, 'N', &mut t1, 1.0, 0.0);
        for p in 0..naux {
            let base = p * npair;
            for pj in 0..npair {
                let pj_swap = (pj % nmo) * nmo + pj / nmo;
                qx[p + pj * naux] += t1.data[base + pj] + t1.data[base + pj_swap];
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
}
