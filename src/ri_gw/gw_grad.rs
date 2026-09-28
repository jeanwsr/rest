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
            degenerate_tol: 1.0e-8,
            phase_mask: 0b111,
            skip_lr_cache: false,
            folded_qx: true,
            adjoint_cphf: true,
            local_di: true,
            keep_adjoint_pieces: false,
            resp_cache_kernel: true,
            fold_cphf_rhs: true,
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

/// Memory probe for the gradient phases: with `REST_GWGRAD_MEMDBG` set, print the
/// resident set (`VmRSS`) and the high-water mark (`VmHWM`) of the process, so the
/// peak of a gradient run can be attributed to a phase.  No effect otherwise.
fn mem_probe(label: &str) {
    if std::env::var("REST_GWGRAD_MEMDBG").is_err() {
        return;
    }
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let kb = |key: &str| -> f64 {
        status
            .lines()
            .find_map(|l| l.strip_prefix(key))
            .and_then(|r| r.trim().trim_end_matches(" kB").trim().parse::<f64>().ok())
            .unwrap_or(0.0)
            / 1024.0
    };
    println!("[memdbg] {label:46} rss {:8.1} MB | peak {:8.1} MB", kb("VmRSS:"), kb("VmHWM:"));
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

    let (i3, shape3) = CInt::integrate_cross("int3c2e", [&mol, &mol, &aux], "s1", None).into();
    assert_eq!(shape3, vec![nao, nao, naux], "int3c2e s1 shape");

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
        i3: RefCell::new(i3),
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
    /// metric-transformed integrals (same layout as `q`). Evictable like [`Self::q`]
    /// (see [`Self::evict_qt`]); only the QP-cache construction reads it.
    pub qt: RefCell<Vec<f64>>,
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
    /// symmetry-protected degenerate shells of `e` (see
    /// [`degenerate_shells`]); shells of size 1 are non-degenerate orbitals
    pub shells: Vec<Vec<usize>>,
    /// index of the shell containing each orbital
    pub shell_of: Vec<usize>,
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
            let n2 = nao * nao;
            let i3_local = raw.i3.borrow();
            let mut ip = MatrixFull::new([nao, nao], 0.0);
            let mut tmp = MatrixFull::new([nao, nmo], 0.0);
            let mut sl = MatrixFull::new([nmo, nmo], 0.0);
            for p_aux in 0..naux {
                ip.data.copy_from_slice(&i3_local[p_aux * n2..(p_aux + 1) * n2]);
                // tmp[mu, q_mo] = sum_nu I[mu, nu, P] C[nu, q_mo]
                _dgemm_full(&ip, 'N', &c_mo, 'N', &mut tmp, 1.0, 0.0);
                // sl[p_mo, q_mo] = sum_mu C[mu, p_mo] tmp[mu, q_mo]
                _dgemm_full(&c_mo, 'T', &tmp, 'N', &mut sl, 1.0, 0.0);
                for col in 0..nmo * nmo {
                    q[col * naux + p_aux] = sl.data[col];
                }
            }
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
        // `i3` has served its only purpose in the engine build (the `q` contraction above); the
        // remaining construction (metric, `qt`, occupied-virtual blocks, low-rank factors) does not
        // read it. Releasing `nao^2 * naux` doubles here (874 MB at naphthalene/def2-TZVP) removes
        // them from the factor-construction phase, which is otherwise the highest point of the
        // engine build. Whoever needs the raw integrals later (the adjoint skeleton contractions,
        // the per-perturbation route, TDDFT) gets them back through `with_i3`.
        raw.evict_i3();
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

        // ---- metric-transformed integrals (LR-only) ----
        let qt = if config.skip_lr_cache {
            Vec::new()
        } else {
            metric_transform_blocked(&s_metric, &q, naux, nmo * nmo)
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
                    qtia[dst..dst + naux].copy_from_slice(&qt[src..src + naux]);
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
        let z_grid = if config.skip_lr_cache {
            Vec::new()
        } else {
            let nsemin = (nocc - 1).saturating_sub(config.selfenergy_state_range);
            let nsemax = (nocc + config.selfenergy_state_range).min(nmo - 1);
            build_real_axis_grid(
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
            )
        };

        // ---- reference static-term data and low-rank factors (LR-only) ----
        let hybrid = scf.mol.xc_data.dfa_hybrid_scf;
        let fock_hyb = if scf.mol.xc_data.dfa_compnt_scf.is_empty() { 1.0 } else { hybrid };
        let vxc_nn = crate::ri_gw::vxc_ao2mo(scf);
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
                lr_imag: Vec::new(),
                lr_grid: Vec::new(),
                qtia: RefCell::new(qtia),
                hybrid,
                fock_hyb,
                vxc_nn,
                shells,
                shell_of,
            }
        } else {
            if std::env::var("REST_GWGRAD_DBG").is_ok() {
                println!("[new DBG] qtia max = {}", qtia.iter().fold(0.0f64, |a, &x| a.max(x.abs())));
            }
            let lr_imag: Vec<LrFactor> = quad
                .iter()
                .map(|&(u, _)| {
                    build_lr_factor(&qtia, naux, nov, &de_ia, u, 'I', 0.0, config.low_rank_tolerance)
                })
                .collect();
            mem_probe("  build: after imaginary factors");
            let lr_grid: Vec<LrFactor> = z_grid
                .iter()
                .map(|&z| {
                    build_lr_factor(&qtia, naux, nov, &de_ia, z, 'R', config.eta, config.low_rank_tolerance)
                })
                .collect();
            mem_probe("  build: after all factors");
            if std::env::var("REST_GWGRAD_DBG").is_ok() {
                println!("[new DBG] lr_imag nkeeps = {:?}", lr_imag.iter().map(|f| f.eigval.len()).collect::<Vec<_>>());
                println!("[new DBG] lr_grid nkeeps = {:?}", lr_grid.iter().map(|f| f.eigval.len()).collect::<Vec<_>>());
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
                lr_imag,
                lr_grid,
                qtia: RefCell::new(qtia),
                hybrid,
                fock_hyb,
                vxc_nn,
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
            let built = metric_transform_blocked(
                &self.s_metric,
                &self.q.borrow(),
                self.naux,
                self.nmo * self.nmo,
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
                    let src = (i + a * nmo) * naux;
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
            factor_on_the_fly = self.with_qtia(|qtia| {
                build_lr_factor(
                    qtia, self.naux, self.de_ia.len(), &self.de_ia,
                    z_used, 'R', self.config.eta, self.config.low_rank_tolerance,
                )
            });
            &factor_on_the_fly
        } else {
            self.lr_factor_nearest(z_used)
        };
        let qnm = self.with_q(|q| self.gather_pair(q, n, m));
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
        let qn = self.with_q(|q| self.gather_row(q, n));
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
                        let qni = self.with_q(|q| self.gather_pair(q, n, i));
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
}

// ---------------------------------------------------------------------------
// CPHF response and gradient driver
// ---------------------------------------------------------------------------

/// Frozen-density RI Coulomb derivative `d J[dm] / dR` (PySCF
/// `gw_cd_grad._j_deriv_from`).
fn j_deriv_from(engine: &GwGradEngine, di: &[f64], dj: &[f64], dm: &[f64]) -> Vec<f64> {
    engine.raw.with_i3(|i3| {
        j_deriv_from_impl(engine.nao, engine.naux, i3, di, dj, dm, &|r| engine.solve_metric(r))
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
    // rho[P] = sum_k I[k, P] dm[k]: contract the natural [n2, naux] views (no `to_mat` copies)
    let rho = gemm_nt(i3, n2, naux, dm, n2, 1);
    let drho = gemm_nt(di, n2, naux, dm, n2, 1);
    let c = solve(&rho);
    let djc = gemm_nn(dj, naux, naux, &c, naux, 1);
    let rhs: Vec<f64> = drho.iter().zip(djc.iter()).map(|(a, b)| a - b).collect();
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
    engine.raw.with_i3(|i3| {
        k_deriv_from_impl(
            engine.nao,
            engine.naux,
            engine.nocc,
            i3,
            di,
            dj,
            cocc,
            &|r| engine.solve_metric(r),
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
        self.canonical_response_batch_full().0
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

        let mut config = AnalDrvConfig::default();
        config.resp.cache_kernel = self.config.resp_cache_kernel;
        // upstream (analdrv refactor): the CP-SCF solve now lives in the shared
        // response object, which `RHessSCF` only borrows.
        let mut resp = rscf_resp_interface(self.scf, &config);
        let nuc_list: Vec<&mut dyn HessNucAPI> = Vec::new();
        let core_list: Vec<&mut dyn RHessCoreAPI> = vec![&mut hcore_obj];
        let el_list: Vec<&mut dyn RHessElecInteractAPI> = vec![&mut rijk_obj];
        // `hess` holds a mutable borrow of `resp`, so it is confined to this
        // scope; the block ends before `resp` is used directly for the solve.
        let (rhs_map, t_prep, t_rhs) = {
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
                let (f1_skel_mo, s1_mo, jd, kd) =
                    self.canonical_skeleton_mo(atm, comp, &h1r, &s1r, &di, &dj);
                t_skel += t0.elapsed().as_secs_f64();
                let mo1_slab = &mo1_data[k_pert * nmo * nocc..(k_pert + 1) * nmo * nocc];
                let t0 = Instant::now();
                let resp1 = self.canonical_finish(&f1_skel_mo, &s1_mo, mo1_slab);
                responses.push(resp1);
                debug.push((
                    jd,
                    kd,
                    f1_skel_mo,
                    s1_mo,
                    mo1_slab.to_vec(),
                    rhs_data[k_pert * nmo * nocc..(k_pert + 1) * nmo * nocc].to_vec(),
                ));
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
        let vfrozen: Vec<f64> = jd.iter().zip(kd.iter()).map(|(a, b)| a - 0.5 * b).collect();
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
        // pairs are excluded from the diagnostic.
        let mut orth = 0.0f64;
        for p in 0..nmo {
            for q in 0..nmo {
                if self.same_degenerate_shell(p, q) {
                    continue;
                }
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
            let packed_pairs = pack_pairs_in_place(&mut bq[0], self.nmo, self.naux);
            let mut fold = self.fold_covector_packed(&bq[0][..packed_pairs * self.naux]);
            mem_probe("gradient: after fold_covector");
            drop(bq);
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
                let blocks = self.raw.d_atom_blocks(atm);
                for comp in 0..3 {
                    let dj = self.raw.d_j_atom(atm, comp);
                    if atm == 0 && comp == 0 {
                        mem_probe("gradient: inside per-direction loop (1st dir)");
                    }
                    grad[atm * 3 + comp] = if self.config.local_di {
                        self.adjoint_perturbation_gradient_local(
                            &adj, &blocks, comp, &h1r, &s1r[comp], &dj, &bj[0], &bb[0],
                            &zero_bx, atm * 3 + comp,
                        )
                    } else {
                        self.adjoint_perturbation_gradient(
                            &adj, &blocks, comp, &h1r, &s1r[comp], &dj, &bj[0], &bb[0],
                            &zero_bx, atm * 3 + comp,
                        )
                    };
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
            Some(self.fold_covector(&bq[0]))
        } else {
            None
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
                            &bq, &bj, &be, &bb, &qx, &dj, &eps1, &b_x,
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

/// Compact a **pair-symmetric** `[nmo*nmo, naux]` buffer (pair-slowest, `pair = row + col*nmo`)
/// in place to its lower triangle `[nmo*(nmo+1)/2, naux]`, returning the packed length.
///
/// Written in increasing packed order, so the destination offset never overtakes an unread source
/// (`tp <= pair`), and the tail is released with `truncate` + `shrink_to_fit`. Used by the adjoint
/// gradient for `bq`, which the pullback has already symmetrised: the fold reads only the symmetric
/// plane, so rebuilding it from the packed form is exact while `naux*nmo^2/2` doubles (437 MB at
/// naphthalene/def2-TZVP) leave the resident set before the fold — the phase this gradient peaks in.
fn pack_pairs_in_place(v: &mut Vec<f64>, nmo: usize, naux: usize) -> usize {
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
// Only exchange-only (HF) references are supported (B_n = 0), as for the
// low-rank engine.

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
    pub omega: f64,
    pub z_factor: f64,
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
            z_factor: 0.0,
            w_imag,
            y_imag,
            w0,
            y0_static,
            y0_metric: y0m,
            residues,
        };
        (sigma.re, dsigma, cache)
    }

    /// Solve the QP equation of the static-subtracted continuous energy
    /// with the analytic-derivative Newton iteration and assemble the full
    /// response cache at the converged root (PySCF `qp_energy_subtracted`).
    pub fn build_target_cache(&self, n: usize) -> CdQpCache {
        let e_n = self.base.e[n];
        let side = if n < self.base.nocc { -1.0 } else { 1.0 };
        let mut x = e_n + side * 1.0e-2;
        let t0 = Instant::now();
        let mut converged = false;
        for iter in 0..self.base.config.qpe_max_iter {
            let (sigma, dsigma, _) = self.sigma_sub(x, n);
            let f = x - e_n - sigma;
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
            println!(
                "warning!!! cd_gw_grad Newton solver did not converge for orbital {}!",
                n
            );
        }
        let (_sigma, dsigma, mut cache) = self.sigma_sub(x, n);
        cache.omega = x;
        cache.z_factor = 1.0 / (1.0 - dsigma);
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
                        cache.omega - base.e[m],
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
                if do_imag {
                    for m in 0..nmo {
                        be[k][m] += c * coeffs[j].explicit[m];
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
                base.polar_pullback(&mut bq, &mut be, k, &v.data, d, dd);
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
                let zsign = (base.e[r.m] - cache.omega).signum();
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
    pub fn adjoint_perturbation_gradient_local(
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
        let nao = self.nao;
        let naux = self.naux;
        let n2 = nao * nao;
        let row0 = blocks.row0;
        let nrow = blocks.nrow;
        let aux0 = blocks.aux0;
        let naux_a = blocks.naux_a;
        let stride1 = nrow * nao * naux;
        let d1c = &blocks.d1[comp * stride1..(comp + 1) * stride1];
        let stride2 = n2 * naux_a;
        let d2c = &blocks.d2[comp * stride2..(comp + 1) * stride2];

        let mut g_di = 0.0f64;
        let mut g_dm = 0.0f64;
        // The two inner loops are pure reductions, and Rust/LLVM will not
        // reassociate `f64` additions on their own; four independent
        // accumulators give both SIMD width and enough ILP to run at memory
        // speed (measured ~3x over the single-accumulator form).
        let dm = &adj.dm_sym;
        for p in 0..naux {
            let cov = &adj.di_cov[p * n2..(p + 1) * n2];
            let lt = adj.lam_tilde[p];
            let base1 = p * nrow * nao;
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
            g_di += adi.iter().sum::<f64>();
            g_dm += lt * adm.iter().sum::<f64>();
            // int3c2e_ip2: dense [nao,nao] symmetric slice, atom's own aux
            // functions only, with the 1/2 of the symmetrised contraction.
            if p >= aux0 && p < aux0 + naux_a {
                let d2 = &d2c[(p - aux0) * n2..(p - aux0 + 1) * n2];
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
                g_di += cdi.iter().sum::<f64>();
                g_dm += lt * cdm.iter().sum::<f64>();
            }
        }
        self.finish_adjoint_gradient(adj, comp, h1r, s1_comp, dj, bj_k, bb_k, b_x, dir, g_di + g_dm)
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
