//! AO-basis matrix-vector products for TDDFT (`tddft_mode = "ao"`)
//!
//! The MO-based path (`matvec.rs`) pre-transforms three-center RI integrals
//! into the MO basis, costing O(naux·(nocc·nvir + nocc² + nvir²)) memory.
//! This AO path keeps the Davidson iteration in the MO amplitude space but
//! evaluates every kernel contribution from the **AO transition density**:
//!
//! P_z = C_occ · z_mat · C_vir^T
//!
//! - Coulomb:  F_J[μν] = Σ_Q B_{μν,Q} · (Σ_λσ B_{λσ,Q} P_λσ)
//!   evaluated with two DGEMVs on the folded-pair RI tensor; correct for
//!   non-symmetric transition densities (2×off-diagonal + 1×diagonal rule).
//! - Exchange: F_K[μν] = Σ_Q Σ_λσ P_λσ B_{μλ,Q} B_{νσ,Q} = Σ_Q M_Q·P·M_Q
//!   where M_Q is the full symmetric expansion of aux column Q.
//!   A-block uses P; B-block uses Pᵀ:
//!   C_occᵀ·K[Pᵀ]·C_vir gives K_B[ia] = Σ_jb (ib|aj) z_jb.
//! - fxc:      ρ_z(g) = Σ_μν P_μν φ_μ(g)φ_ν(g) → ×wfxc[g] → contract back
//!   with AO values on grids (LDA nvar=1 / GGA nvar=4).
//!
//! Final projection back to MO amplitudes: result = C_occᵀ · F · C_vir.

use rest_tensors::{MatrixFull, RIFull};
use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;
use rest_tensors::matrixupper::map_upper_to_full;
use rayon::prelude::*;

use crate::scf_io::{SCF, SCFType};
use crate::dft::num_int::FXCMatvecData;
use crate::dft::Grids;
use crate::dft::numint_matmul::nimatmul::NIMatmul;
use crate::dft::numint_matmul::hess_rks::eval_vxc_fxc_from_rho;
use crate::dft::xceff::prelude::{XCDenType, XCSpin};
use crate::ri_jk::util::get_cint_mol;
use crate::ri_tddft::tddft::FxcDriver;
use crate::ri_tddft::utils::{
    tddft_occupation_parameters, tddft_occupation_parameters_u, TddftSector,
};
use crate::ri_tddft::{TDDFTData, TDDFTMode};
use crate::utilities::rstsr_util::{RestTensorToRstsrTsrAPI, RestTensorToRstsrViewAPI, Tsr, TsrView};
use rstsr::prelude::*;
use libxc::prelude::*;

/// Type alias matching `scf.rimatr`:
/// (ri3fn folded [npair, naux], basbas2baspar map, baspar2basbas list)
pub type RimatrTuple = Option<(MatrixFull<f64>, MatrixFull<usize>, Vec<[usize; 2]>)>;


// ══════════════════════════════════════════════════════════════════
// Timing instrumentation (debug-gated; see `ao_timing_report`)
// ══════════════════════════════════════════════════════════════════

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Cumulative nanoseconds: transition-density build + symmetrization.
pub static T_TDEN: AtomicU64 = AtomicU64::new(0);
/// Cumulative nanoseconds: batched RI-J (Coulomb).
pub static T_J: AtomicU64 = AtomicU64::new(0);
/// Cumulative nanoseconds: batched RI-K (exchange, all drivers).
pub static T_K: AtomicU64 = AtomicU64::new(0);
/// Cumulative nanoseconds: batched fxc (total).
pub static T_FXC: AtomicU64 = AtomicU64::new(0);
/// Cumulative nanoseconds: fxc `make_rho_from_dm` part.
pub static T_FXC_RHO: AtomicU64 = AtomicU64::new(0);
/// Cumulative nanoseconds: fxc `make_fxc_pot_with_eff` part.
pub static T_FXC_POT: AtomicU64 = AtomicU64::new(0);
/// Cumulative nanoseconds: per-column assembly + `contract_back`.
pub static T_CONT: AtomicU64 = AtomicU64::new(0);
/// Cumulative nanoseconds: whole batched A/B matvec closures (diagonal + kernel).
pub static T_CLOS: AtomicU64 = AtomicU64::new(0);
/// Number of batched A/B matvec closure calls.
pub static N_AO_CLOS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn add_ns(counter: &AtomicU64, started: Instant) {
    counter.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

pub(crate) fn s_of(counter: &AtomicU64) -> f64 {
    counter.load(Ordering::Relaxed) as f64 / 1.0e9
}

/// Print the accumulated AO-kernel timing table (debug level, visible with
/// `print_level = 2`). Percentages are relative to the total closure time
/// (batched A/B matvecs, diagonal included).
pub fn ao_timing_report() {
    if !log::log_enabled!(log::Level::Debug) {
        return;
    }
    let n = N_AO_CLOS.load(Ordering::Relaxed);
    let clos = s_of(&T_CLOS);
    log::debug!("AO matvec timing ({} batched A/B calls, total {:.3} s):", n, clos);
    let rows = [
        ("tden build", &T_TDEN),
        ("J (Coulomb)", &T_J),
        ("K (exchange)", &T_K),
        ("fxc total", &T_FXC),
        ("  fxc rho", &T_FXC_RHO),
        ("  fxc pot", &T_FXC_POT),
        ("assemble+contract", &T_CONT),
    ];
    for (name, counter) in rows {
        let t = s_of(counter);
        let pct = if clos > 0.0 { 100.0 * t / clos } else { 0.0 };
        log::debug!("  {:<18} {:>10.3} s  ({:>5.1}% of closure)", name, t, pct);
    }
}


// ══════════════════════════════════════════════════════════════════
// Core contractions
// ══════════════════════════════════════════════════════════════════

/// Transition density P = C_occ · z_mat · C_virᵀ with z[i + a*occ].
pub fn transition_density(
    c_occ: &MatrixFull<f64>,
    c_vir: &MatrixFull<f64>,
    z: &[f64],
    nao: usize,
    occ_size: usize,
    vir_size: usize,
) -> MatrixFull<f64> {
    let z_mat = MatrixFull::from_vec([occ_size, vir_size], z.to_vec()).unwrap();
    let mut t1 = MatrixFull::new([nao, vir_size], 0.0);
    _dgemm_full(c_occ, 'N', &z_mat, 'N', &mut t1, 1.0, 0.0);
    let mut p = MatrixFull::new([nao, nao], 0.0);
    _dgemm_full(&t1, 'N', c_vir, 'T', &mut p, 1.0, 0.0);
    p
}

/// Batched Coulomb (J) over all trial vectors: $J[D^{\mathbb{A}}]$ for
/// $\mathbb{A} = 0..m$ in one ri_jk call, amortizing the cderi unpack +
/// `to_rstsr` conversions across nset.
///
/// Returns `[nao*nao, m]` with column $\mathbb{A}$ = flattened J (column-major
/// `[nao,nao]`), matching the `f_fxc_block` layout (`base = \mathbb{A}·nao·nao`).
fn get_j_ao_batched(scf: &SCF, p_block: &[MatrixFull<f64>]) -> MatrixFull<f64> {
    let device = DeviceBLAS::default();
    let (ri3fn, _, _) = scf.rimatr.as_ref()
        .expect("rimatr must be initialized for AO-mode TDDFT");
    let cderi = ri3fn.to_rstsr_view(&device);
    let dms_slice: &[MatrixFull<f64>] = p_block;
    let dms = dms_slice.to_rstsr(&device);               // [nao, nao, m]
    let nao = scf.mol.num_basis;
    let js = crate::ri_jk::pure_incore::get_vj_ri_incore_nonsym(cderi, dms.view()); // [nao,nao,m]
    let mut out = MatrixFull::new([nao * nao, p_block.len()], 0.0);
    out.data.copy_from_slice(js.raw());
    out
}

/// Batched Exchange (K) over all trial vectors. Exact route: one
/// `get_vk_ri_incore_dm` over the whole `[nao,nao,m]` block. Low-rank route
/// (SVD is per-vector, so it cannot be batched): loops per vector and stacks.
///
/// `c_occ`/`c_vir` are the sector's occupied/virtual MO coefficients — for an
/// unrestricted reference each spin sector calls this with its own orbitals
/// (exchange acts within a spin sector only).
///
/// Returns `[nao*nao, m]` (column $\mathbb{A}$ = flattened K, matching
/// `f_fxc_block`). Always evaluates $K[D^{\mathbb{A}}]$ with the untransposed
/// density; the B-block exchange is derived by the caller from the identity
/// $K[D^{\mathrm{T}}] = K[D]^{\mathrm{T}}$ (each $M_Q$ is symmetric).
fn get_k_ao_batched(
    scf: &SCF,
    c_occ: &MatrixFull<f64>,
    c_vir: &MatrixFull<f64>,
    z_block: &MatrixFull<f64>,
    p_block: &[MatrixFull<f64>],
) -> MatrixFull<f64> {
    let tddft_ctrl = scf.mol.ctrl.tddft.as_ref();
    let driver = tddft_ctrl.map_or("dm", |t| {
        let d = t.tddft_ao_rik_driver.as_str();
        if d != "dm" && d != "semitrans" && d != "lowrank" {
            log::warn!("Unknown tddft_ao_rik_driver = \"{}\"; falling back to \"dm\"", d);
        }
        d
    });
    let svd_tol = tddft_ctrl.map_or(1.0e-6, |t| t.tddft_svd_tol);
    let nao = scf.mol.num_basis;
    let m = p_block.len();
    let device = DeviceBLAS::default();
    let (ri3fn, _, _) = scf.rimatr.as_ref()
        .expect("rimatr must be initialized for AO-mode TDDFT");
    let cderi = ri3fn.to_rstsr_view(&device);
    let naux = ri3fn.size[1];
    let occ_size = c_occ.size[1];
    let vir_size = c_vir.size[1];

    let mut out = MatrixFull::new([nao * nao, m], 0.0);
    match driver {
        "semitrans" => {
            // Semi-transformation driver: fold the amplitudes into occ-side
            // coefficients first:
            //   CX_s = C_vir · X_sᵀ  (nao × occ), so that P_s = CX_s · C_occᵀ holds
            //   exactly (rank ≤ nocc, no SVD needed), and
            //   K_s = Σ_Q (M_Q CX_s)(M_Q C_occ)ᵀ  via ri_jk::get_vk_ri_incore_coeff_pair.
            // The fold always uses the A-side amplitude (never refolds at k = nvir);
            // K[Pᵀ] = K[P]ᵀ is exploited by transposing the output below.
            let c_vir_v = c_vir.to_rstsr_view(&device);
            let c_occ_v = c_occ.to_rstsr_view(&device);
            // x_t [vir, occ, m]: the amplitudes in [vir, occ] order per set
            // (per-set transpose of the [occ, vir] columns).
            let x_tsr = rt::asarray((&z_block.data, [occ_size, vir_size, m].f(), &device))
                .swapaxes(0, 1)
                .into_contig(FlagOrder::F);
            // CX block [nao, occ, m]
            let mut cx = rt::zeros(([nao, occ_size, m].f(), &device));
            for s in 0..m {
                cx.i_mut((.., .., s)).matmul_from(&c_vir_v, &x_tsr.i((.., .., s)), 1.0, 0.0);
            }
            let ks = crate::ri_jk::pure_incore::get_vk_ri_incore_coeff_pair(
                cderi, cx.view(), c_occ_v.view(), naux,
            );
            // The fold always produces K[Pᵀ] (= K[C_vir Xᵀ C_occᵀ]); transpose
            // each set (axis swap (mu,nu)->(nu,mu)) to return K[P].
            let ks_out = ks.swapaxes(0, 1).into_contig(FlagOrder::F);
            out.data.copy_from_slice(ks_out.raw());
        }
        "lowrank" => {
            // Low-rank: per-vector SVD (cannot batch the SVD truncation).
            let c_occ_v = c_occ.to_rstsr_view(&device);
            let c_vir_v = c_vir.to_rstsr_view(&device);
            for s in 0..m {
                let z_col: Vec<f64> = (0..z_block.size[0]).map(|r| z_block[[r, s]]).collect();
                let z_mat = MatrixFull::from_vec([occ_size, vir_size], z_col).unwrap();
                let (c_left, c_right) = (c_occ_v.view(), c_vir_v.view());
                let k2 = crate::ri_jk::pure_incore::get_vk_ri_incore_dm_lowrank(
                    cderi.view(),
                    c_left,
                    c_right,
                    z_mat.to_rstsr_view(&device),
                    svd_tol,
                    naux,
                );
                out.data[s * nao * nao..(s + 1) * nao * nao].copy_from_slice(k2.raw());
            }
        }
        _ => {
            // Exact: one batched K over all trial vectors.
            let dms_slice: &[MatrixFull<f64>] = p_block;
            let dms = dms_slice.to_rstsr(&device);            // [nao,nao,m]
            let ks = crate::ri_jk::pure_incore::get_vk_ri_incore_dm(cderi, dms.view(), naux);
            out.data.copy_from_slice(ks.raw());
        }
    }
    out
}

/// Project an AO-basis matrix back to MO amplitudes: r[i,a] = C_occᵀ·F·C_vir.
pub fn contract_back(
    f_full: &MatrixFull<f64>,
    c_occ: &MatrixFull<f64>,
    c_vir: &MatrixFull<f64>,
    occ_size: usize,
    vir_size: usize,
) -> Vec<f64> {
    let nao = f_full.size[0];
    let mut tmp = MatrixFull::new([occ_size, nao], 0.0);
    _dgemm_full(c_occ, 'T', f_full, 'N', &mut tmp, 1.0, 0.0);
    let mut r = MatrixFull::new([occ_size, vir_size], 0.0);
    _dgemm_full(&tmp, 'N', c_vir, 'N', &mut r, 1.0, 0.0);
    r.data
}


// ══════════════════════════════════════════════════════════════════
// Public A/B matvecs (drop-in replacements for matvec.rs versions)
// ══════════════════════════════════════════════════════════════════

/// Full A-block matvec in AO mode, applied to a block of trial vectors.
///
/// `z_block` is `[dim, m]`; returns `[dim, m]`. The fxc term is batched via
/// `NIMatmul` (all m vectors in one pass); RI-J/K stay per-vector.
/// Symmetrize a transition density: P_sym = (P + Pᵀ)/2.
fn symmetrize_density(p: &MatrixFull<f64>) -> MatrixFull<f64> {
    let [n, _] = p.size;
    let mut p_sym = MatrixFull::new([n, n], 0.0);
    for i in 0..n {
        for j in 0..n {
            p_sym[[i, j]] = 0.5 * (p[[i, j]] + p[[j, i]]);
        }
    }
    p_sym
}

/// Batched AO-mode fxc kernel over a block of transition densities via
/// NIMatmul (`make_rho_from_dm` + `make_fxc_pot_with_eff`, PySCF `vind(zs)`
/// style). When `grid_batch` is set, the evaluation runs over grid batches
/// (`split_batch`) so the full `[ngrids, nao, ncomp]` AO cache is never
/// materialized.
///
/// `p_sectors[sigma]` holds the m transition densities of spin sector sigma
/// (one sector for RHF, two for UHF). The kernel application branches on the
/// reference type: RHF applies the unpolarized kernel `[g,nvar,nvar]` to
/// `rho1 [g,nvar,m]`; UHF applies the spin-polarized kernel
/// `[g,nvar,2,nvar,2]` to `rho1 [g,nvar,2,m]` — the spin-σ potential of one
/// trial vector couples BOTH its spin densities through the f^{στ} blocks.
///
/// Returns `[nao, nao*n_sec*m]` (flattened column-major `[nao, nao, n_sec, m]`):
/// the block (σ, s) is at data offset `(σ·m + s)·nao²` (matching the assembly
/// loops).
fn fxc_matvec_ao_batched(
    ao_data: &mut TDDFTData,
    p_sectors: &[Vec<MatrixFull<f64>>],
    device: &DeviceBLAS,
) -> MatrixFull<f64> {
    use crate::dft::xceff::prelude::XCSpin;
    let is_uhf = ao_data.is_uhf();
    let n_sec = p_sectors.len();
    let m = p_sectors[0].len();
    let nao = p_sectors[0][0].size[0];
    let ni = ao_data.ni.as_mut().expect("AO-mode fxc requires NIMatmul");
    let ngrids = ni.coords.len();
    let den_type = ao_data.den_type.expect("AO-mode fxc requires den_type");
    let nvar = den_type.num_nvar();
    let n_tot = n_sec * m;

    // Flatten sector densities [sector0 × m, sector1 × m, ...] and symmetrize:
    // make_rho_from_dm assumes a symmetric density for its SIGMA response —
    // exact, the fxc kernel is symmetric in μν so the response only sees the
    // symmetric part.
    let mut p_all: Vec<MatrixFull<f64>> = Vec::with_capacity(n_tot);
    for p_sec in p_sectors {
        p_all.extend(p_sec.iter().map(|p| symmetrize_density(p)));
    }
    let dms = p_all.as_slice().to_rstsr(device); // [nao, nao, n_tot]
    let fxc_eff_view = ao_data.fxc_eff.as_ref().unwrap().view();

    // out block (σ, s) at (σ·m + s)·nao²
    let mut out = MatrixFull::new([nao, nao * n_tot], 0.0);
    if !ao_data.grid_batch {
        // Full-grid path: one batched call over all sets.
        let dm_views: Vec<TsrView> = (0..n_tot).map(|s| dms.i((.., .., s))).collect();
        let t0 = Instant::now();
        let rho1 = ni.make_rho_from_dm(&dm_views, den_type); // [g, nvar, n_tot]
        add_ns(&T_FXC_RHO, t0);
        let t0 = Instant::now();
        let f_fxc = if is_uhf {
            // set = σ*m + s: a pure f-order reinterpretation
            // [g,nvar,n_tot] → [g,nvar,n_sec,m]
            let rho1_u = rho1.into_shape([ngrids, nvar, n_sec, m]);
            ni.make_fxc_pot_with_eff(fxc_eff_view, rho1_u.view(), den_type, XCSpin::Polarized)
        } else {
            ni.make_fxc_pot_with_eff(fxc_eff_view, rho1.view(), den_type, XCSpin::Unpolarized)
        };
        add_ns(&T_FXC_POT, t0);
        for s in 0..m {
            if is_uhf {
                for sigma in 0..n_sec {
                    let blk = f_fxc.i((.., .., sigma, s)); // [nao, nao]
                    let base = (sigma * m + s) * nao * nao;
                    for (r, v) in blk.iter().enumerate() {
                        out.data[base + r] = *v;
                    }
                }
            } else {
                let base = s * nao * nao;
                for (r, v) in f_fxc.i((.., .., s)).iter().enumerate() {
                    out.data[base + r] = *v;
                }
            }
        }
    } else {
        // Grid-batched path: evaluate per grid batch (all sets), accumulating.
        let nbatch = ni.nbatch;
        for start in (0..ngrids).step_by(nbatch) {
            let end = (start + nbatch).min(ngrids);
            let nb = end - start;
            let mut ni_batch = ni.split_batch(start, end);
            let dm_views: Vec<TsrView> = (0..n_tot).map(|s| dms.i((.., .., s))).collect();
            let t0 = Instant::now();
            let rho1_b = ni_batch.make_rho_from_dm(&dm_views, den_type); // [nb, nvar, n_tot]
            add_ns(&T_FXC_RHO, t0);
            let fxc_eff_b = fxc_eff_view.i((start..end));
            let t0 = Instant::now();
            let f_b = if is_uhf {
                let rho1_bu = rho1_b.into_shape([nb, nvar, n_sec, m]);
                ni_batch.make_fxc_pot_with_eff(fxc_eff_b, rho1_bu.view(), den_type, XCSpin::Polarized)
            } else {
                ni_batch.make_fxc_pot_with_eff(fxc_eff_b, rho1_b.view(), den_type, XCSpin::Unpolarized)
            };
            add_ns(&T_FXC_POT, t0);
            for s in 0..m {
                if is_uhf {
                    for sigma in 0..n_sec {
                        let blk = f_b.i((.., .., sigma, s));
                        let base = (sigma * m + s) * nao * nao;
                        for (r, v) in blk.iter().enumerate() {
                            out.data[base + r] += *v;
                        }
                    }
                } else {
                    let base = s * nao * nao;
                    for (r, v) in f_b.i((.., .., s)).iter().enumerate() {
                        out.data[base + r] += *v;
                    }
                }
            }
        }
    }
    out
}

/// "mo"/"semitrans" fxc drivers: the occ/vir-reduced kernel application using
/// the cached per-sector occ-side MO-on-grid projection tables
/// (`psi_occ`/`psi_occ_grad`).
///
/// Per sector sigma (RHF: one sector; UHF: alpha/beta), with
/// $\psi_i(g)=\sum_\mu C^{\sigma}_{\mu i}\varphi_\mu(g)$:
///
/// $$\rho^{\sigma}_{0}(g) = \sum_{ia} z^{\sigma}_{ia}\,\psi^{\sigma}_i(g)\psi^{\sigma}_a(g), \qquad
///   \rho^{\sigma}_{d+1}(g) = \sum_{ia} z^{\sigma}_{ia}\,(\partial_d\psi_i\,\psi_a + \psi_i\,\partial_d\psi_a)(g)$$
///
/// $$v^{\sigma}_{1,\alpha}(g) = w(g)\sum_{\tau,\beta} f^{\rm xc}[g,\alpha,\sigma,\beta,\tau]\,\rho^{\tau}_{\beta}(g), \qquad
///   E^{\sigma}_{ia} = \sum_g \Lambda^{\sigma,\alpha}_{ia}(g)\,v^{\sigma}_{1,\alpha}(g)$$
///
/// (RHF: sigma = tau = 0 and the kernel table is `[g,nvar,nvar]` — the offset
/// strides below reduce exactly to the 3D case, keeping the restricted
/// arithmetic unchanged.)
///
/// Every contraction runs in the (n_occ, n_vir, n_grid) space —
/// no `[nao, nao, m]` intermediates, no `contract_back`.
/// Returns MO amplitudes `[dim_total, m]` (added directly to the matvec result).
fn fxc_mo_matvec(
    scf: &SCF,
    ao_data: &TDDFTData,
    z_block: &MatrixFull<f64>,
    sectors: &[TddftSector],
    device: &DeviceBLAS,
) -> MatrixFull<f64> {
    let n_sec = sectors.len();
    let is_uhf = ao_data.is_uhf();
    let psi_all = ao_data.psi_occ.as_ref().expect("mo fxc requires psi_occ");
    let pog_all = ao_data.psi_occ_grad.as_ref(); // per sector [3, ng, occ] (GGA only)
    let gga = pog_all.is_some();
    let nvar = if gga { 4 } else { 1 };
    let fxc_eff = ao_data.fxc_eff.as_ref().unwrap().view(); // RHF [ng,nvar,nvar] / UHF [ng,nvar,2,nvar,2]
    let dim = z_block.size[0];
    let m = z_block.size[1];
    let ng = psi_all[0].shape()[0];
    let weights = &scf.grids.as_ref().expect("DFT grids required for mo fxc").weights;
    let fxc_raw = fxc_eff.raw();
    let fxc_off = fxc_eff.offset();
    let deriv = if gga { 1 } else { 0 };
    let st = ao_data.fxc_driver == Some(FxcDriver::SEMITRANS);

    // fxc table strides: RHF [g, var1, var2] vs UHF [g, var1, s1, var2, s2].
    let (stride_s1, stride_v2, stride_s2, n_tau) = if is_uhf {
        (nvar * ng, 2 * nvar * ng, 2 * nvar * nvar * ng, 2usize)
    } else {
        (0, nvar * ng, 0, 1usize)
    };

    // Sector row offsets in the concatenated amplitude block.
    let mut sector_row0: Vec<usize> = Vec::with_capacity(n_sec);
    {
        let mut acc = 0usize;
        for sec in sectors {
            sector_row0.push(acc);
            acc += sec.dim();
        }
    }

    // ── per-sector set-stacked right operands (built ONCE per call) ──
    //   "mo":        right_s = Z_s[(s*nocc+i), b]              (contig f-order)
    //   "semitrans": right_s = Z_s · C_vir_s^T  [(s*nocc+i), nao]
    let mut right_s: Vec<Tsr> = Vec::with_capacity(n_sec);
    for (i_sec, sec) in sectors.iter().enumerate() {
        let z_sec = slice_z_rows(z_block, sector_row0[i_sec], sector_row0[i_sec] + sec.dim());
        let z_stack = rt::asarray((&z_sec.data, [sec.occ_size, sec.vir_size, m].f(), device))
            .swapaxes(1, 2)
            .into_contig(FlagOrder::F)
            .into_shape([m * sec.occ_size, sec.vir_size]);
        if st {
            let c_vir_v = ao_data.c_vir[i_sec].to_rstsr_view(device);
            let mut zt = rt::zeros(([m * sec.occ_size, ao_data.c_vir[i_sec].size[0]].f(), device));
            zt.matmul_from(&z_stack, &c_vir_v.t(), 1.0, 0.0);
            right_s.push(zt);
        } else {
            right_s.push(z_stack.into_owned());
        }
    }

    let ni = ao_data.ni.as_ref().expect("mo fxc requires NIMatmul");
    let nbatch = ni.nbatch;
    let nchunk = 1536usize;
    let nb_max = nbatch.min(ng);
    // "mo": reusable vir-side batch buffers per sector (allocated once per
    // call, refilled per batch).
    let mut pv_all = if st {
        None
    } else {
        Some(sectors.iter()
            .map(|sec| rt::zeros(([nb_max, sec.vir_size].f(), device)))
            .collect::<Vec<_>>())
    };
    let mut pvg_all = if st || !gga {
        None
    } else {
        Some(sectors.iter()
            .map(|sec| (0..3)
                .map(|_| rt::zeros(([nb_max, sec.vir_size].f(), device)))
                .collect::<Vec<_>>())
            .collect::<Vec<_>>())
    };

    let mut result = MatrixFull::new([dim, m], 0.0);
    let mut g0 = 0usize;
    while g0 < ng {
        let g1 = (g0 + nbatch).min(ng);
        let nb = g1 - g0;
        // ── serial: evaluate AO for this batch; "mo" also projects the vir side ──
        let t_proj = Instant::now();
        let mut ni_b = ni.split_batch(g0, g1);
        let ao_b = ni_b.get_cached_ao(deriv); // [nb, nao, ncomp]; alive through the chunk loop
        if !st {
            let pv = pv_all.as_mut().unwrap();
            for (i_sec, sec) in sectors.iter().enumerate() {
                let c_vir_v = ao_data.c_vir[i_sec].to_rstsr_view(device);
                pv[i_sec]
                    .i_mut((..nb, ..))
                    .matmul_from(&ao_b.i((.., .., 0)), &c_vir_v, 1.0, 0.0);
            }
            if gga {
                let pvg = pvg_all.as_mut().unwrap();
                for d in 0..3 {
                    for (i_sec, sec) in sectors.iter().enumerate() {
                        let c_vir_v = ao_data.c_vir[i_sec].to_rstsr_view(device);
                        pvg[i_sec][d]
                            .i_mut((..nb, ..))
                            .matmul_from(&ao_b.i((.., .., 1 + d)), &c_vir_v, 1.0, 0.0);
                    }
                }
            }
        }
        add_ns(&T_FXC_RHO, t_proj);

        // ── parallel chunks within this batch (set-stacked kernels) ──
        let ntask_b = nb.div_ceil(nchunk);
        let chunk_results: Vec<Vec<Vec<f64>>> = (0..ntask_b)
            .into_par_iter()
            .map(|ic| {
                let cg0 = g0 + ic * nchunk;
                let cg1 = (cg0 + nchunk).min(g1);
                let cg = cg1 - cg0;
                let lg0 = cg0 - g0;
                let lg1 = cg1 - g0;
                let phi_c = ao_b.i((lg0..lg1, .., 0)); // [cg, nao]
                let mut e_out: Vec<Vec<f64>> = sectors.iter()
                    .map(|sec| vec![0.0_f64; m * sec.occ_size * sec.vir_size])
                    .collect();

                // ── per-sector response densities rho^sigma_var(g) ──
                // rho_bufs[sigma][s][beta][g]
                let mut rho_bufs: Vec<Vec<Vec<Vec<f64>>>> = sectors.iter()
                    .map(|_| vec![vec![vec![0.0; cg]; 4]; m])
                    .collect();
                for (i_sec, sec) in sectors.iter().enumerate() {
                    let occ_s = sec.occ_size;
                    if occ_s == 0 {
                        continue;
                    }
                    let psi_s = &psi_all[i_sec]; // [ng, occ]
                    let po_c = psi_s.i((cg0..cg1, ..)); // [cg, occ]
                    // driver-dependent GEMM operands:
                    //   "mo":        left = psi_vir_c  [cg, nvir], right = Z_s
                    //   "semitrans": left = phi_c       [cg, nao],  right = Z_st_s
                    let left_c = if st {
                        ao_b.i((lg0..lg1, .., 0))
                    } else {
                        pv_all.as_ref().unwrap()[i_sec].i((lg0..lg1, ..))
                    };
                    let right = &right_s[i_sec];

                    // ── rho build: ONE stacked GEMM per component (all m sets at once) ──
                    // q[g, s*nocc+i] = sum_b left[g,b] * right[s*nocc+i, b]
                    let mut q_stack = rt::zeros(([cg, m * occ_s].f(), device));
                    q_stack.matmul_from(&left_c, &right.t(), 1.0, 0.0);
                    let mut qd_stack = if gga {
                        Some(rt::zeros(([cg, m * occ_s].f(), device)))
                    } else {
                        None
                    };
                    for s in 0..m {
                        let q_s = q_stack.i((.., s * occ_s..(s + 1) * occ_s));
                        let mut rho0 = rt::zeros(([cg], device));
                        rho0.i_mut((..)).vecdot_from(&q_s, &po_c, 1);
                        for g in 0..cg {
                            rho_bufs[i_sec][s][0][g] = rho0[[g]];
                        }
                    }
                    if gga {
                        let pog = &pog_all.as_ref().unwrap()[i_sec]; // [3, ng, occ]
                        for d in 0..3 {
                            // "mo": grad left operand from the projected batch buffer;
                            // "semitrans": raw AO gradient slab of this batch.
                            let grad_c = if st {
                                ao_b.i((lg0..lg1, .., 1 + d))
                            } else {
                                pvg_all.as_ref().unwrap()[i_sec][d].i((lg0..lg1, ..))
                            };
                            qd_stack
                                .as_mut()
                                .unwrap()
                                .matmul_from(&grad_c, &right.t(), 1.0, 0.0);
                            for s in 0..m {
                                let q_s = q_stack.i((.., s * occ_s..(s + 1) * occ_s));
                                let qd_s = qd_stack
                                    .as_ref()
                                    .unwrap()
                                    .i((.., s * occ_s..(s + 1) * occ_s));
                                // term 1: rho1_d[g] = sum_i (d_d psi_occ)[g,i] * q[g,i]
                                let mut r_d = rt::zeros(([cg], device));
                                r_d.i_mut((..)).vecdot_from(&pog.i((d, cg0..cg1, ..)), &q_s, 1);
                                // term 2: rho2_d[g] = sum_i psi_occ[g,i] * qd[g,i]
                                let mut r_t = rt::zeros(([cg], device));
                                r_t.i_mut((..)).vecdot_from(&po_c, &qd_s, 1);
                                for g in 0..cg {
                                    rho_bufs[i_sec][s][1 + d][g] = r_d[[g]] + r_t[[g]];
                                }
                            }
                        }
                    }
                }

                // ── weighted kernel per set:
                // v1^sigma_alpha(g) = w(g) Σ_{tau,beta} fxc[g,alpha,sigma,beta,tau] rho^tau_beta(g)
                // (RHF: sigma = tau = 0, n_tau = 1 — identical to the 3D kernel loop)
                let mut v1_bufs: Vec<Vec<Vec<Vec<f64>>>> = sectors.iter()
                    .map(|_| vec![vec![vec![0.0; cg]; nvar]; m])
                    .collect();
                for sigma in 0..n_sec {
                    for s in 0..m {
                        for alpha in 0..nvar {
                            for tau in 0..n_tau {
                                for beta in 0..nvar {
                                    let base = fxc_off + alpha * ng + sigma * stride_s1
                                        + beta * stride_v2 + tau * stride_s2 + cg0;
                                    let r = &rho_bufs[tau][s][beta];
                                    let v = &mut v1_bufs[sigma][s][alpha];
                                    for g in 0..cg {
                                        v[g] += fxc_raw[base + g] * r[g];
                                    }
                                }
                            }
                            for g in 0..cg {
                                v1_bufs[sigma][s][alpha][g] *= weights[cg0 + g];
                            }
                        }
                    }
                }

                // ── per-sector back-projection: stacked scalings + GEMM per term ──
                //   "mo":        E_stack += S^a * left^a_c              (one GEMM)
                //   "semitrans": G = S^a * phi_c; E_stack += G * C_vir_s (two GEMMs;
                //                g_stack must be REWRITTEN (beta = 0) per term)
                let t_pot = Instant::now();
                for (i_sec, sec) in sectors.iter().enumerate() {
                    let occ_s = sec.occ_size;
                    let vir_s = sec.vir_size;
                    if occ_s == 0 {
                        continue;
                    }
                    let psi_s = &psi_all[i_sec]; // [ng, occ]
                    let po_c = psi_s.i((cg0..cg1, ..)); // [cg, occ]
                    let left_c = if st {
                        ao_b.i((lg0..lg1, .., 0))
                    } else {
                        pv_all.as_ref().unwrap()[i_sec].i((lg0..lg1, ..))
                    };
                    let c_vir_s = &ao_data.c_vir[i_sec];
                    let c_vir_v = c_vir_s.to_rstsr_view(device);
                    let mut s_stack = vec![0.0_f64; m * occ_s * cg]; // reused build buffer
                    let mut e_stack = rt::zeros(([m * occ_s, vir_s].f(), device));
                    let mut g_stack = if st {
                        Some(rt::zeros(([m * occ_s, c_vir_s.size[0]].f(), device)))
                    } else {
                        None
                    };
                    // var1 = 0: S[(s*nocc+i), g] = v1_0^s(g) * psi[g,i]
                    // (measured: the (g,i) order below beats the tilted (i,g-block) order —
                    // the scattered po_c reads are L2-resident hits, cheaper than the
                    // reordered write pattern)
                    for s in 0..m {
                        let v = &v1_bufs[i_sec][s][0];
                        let rbase = s * occ_s;
                        for g in 0..cg {
                            let w = v[g];
                            for i in 0..occ_s {
                                // f-order [m*nocc, cg]: idx = row + col*nrow
                                s_stack[rbase + i + g * (m * occ_s)] = po_c[[g, i]] * w;
                            }
                        }
                    }
                    let s0 = rt::asarray((&s_stack, [m * occ_s, cg].f(), device));
                    if st {
                        let g_s = g_stack.as_mut().unwrap();
                        g_s.matmul_from(&s0, &left_c, 1.0, 0.0);
                        e_stack.matmul_from(g_s, &c_vir_v, 1.0, 0.0);
                    } else {
                        e_stack.matmul_from(&s0, &left_c, 1.0, 0.0);
                    }
                    if gga {
                        let pog = &pog_all.as_ref().unwrap()[i_sec]; // [3, ng, occ]
                        for d in 0..3 {
                            let grad_c = if st {
                                ao_b.i((lg0..lg1, .., 1 + d))
                            } else {
                                pvg_all.as_ref().unwrap()[i_sec][d].i((lg0..lg1, ..))
                            };
                            // term 1: S = v1_{d+1}^s(g) * (d_d psi_occ)[g,i]
                            for s in 0..m {
                                let v = &v1_bufs[i_sec][s][1 + d];
                                let rbase = s * occ_s;
                                for g in 0..cg {
                                    let w = v[g];
                                    for i in 0..occ_s {
                                        s_stack[rbase + i + g * (m * occ_s)] = pog[[d, cg0 + g, i]] * w;
                                    }
                                }
                            }
                            let s1 = rt::asarray((&s_stack, [m * occ_s, cg].f(), device));
                            if st {
                                let g_s = g_stack.as_mut().unwrap();
                                g_s.matmul_from(&s1, &left_c, 1.0, 0.0);
                                e_stack.matmul_from(g_s, &c_vir_v, 1.0, 1.0);
                            } else {
                                e_stack.matmul_from(&s1, &left_c, 1.0, 1.0);
                            }
                            // term 2: S = v1_{d+1}^s(g) * psi_occ[g,i]
                            for s in 0..m {
                                let v = &v1_bufs[i_sec][s][1 + d];
                                let rbase = s * occ_s;
                                for g in 0..cg {
                                    let w = v[g];
                                    for i in 0..occ_s {
                                        s_stack[rbase + i + g * (m * occ_s)] = po_c[[g, i]] * w;
                                    }
                                }
                            }
                            let s2 = rt::asarray((&s_stack, [m * occ_s, cg].f(), device));
                            if st {
                                let g_s = g_stack.as_mut().unwrap();
                                g_s.matmul_from(&s2, &grad_c, 1.0, 0.0);
                                e_stack.matmul_from(g_s, &c_vir_v, 1.0, 1.0);
                            } else {
                                e_stack.matmul_from(&s2, &grad_c, 1.0, 1.0);
                            }
                        }
                    }

                    // scatter E_stack[(s*nocc+i), a] -> e_out[i_sec][s][i,a]
                    for s in 0..m {
                        let obase = s * occ_s * vir_s;
                        for a in 0..vir_s {
                            for i in 0..occ_s {
                                e_out[i_sec][obase + i * vir_s + a] = e_stack[[s * occ_s + i, a]];
                            }
                        }
                    }
                }
                add_ns(&T_FXC_POT, t_pot);
                e_out
            })
            .collect::<Vec<_>>();

        for e_out in chunk_results.iter() {
            for (i_sec, sec) in sectors.iter().enumerate() {
                for s in 0..m {
                    let obase = s * sec.occ_size * sec.vir_size;
                    for a in 0..sec.vir_size {
                        for i in 0..sec.occ_size {
                            result[[sector_row0[i_sec] + i + a * sec.occ_size, s]]
                                += e_out[i_sec][obase + i * sec.vir_size + a];
                        }
                    }
                }
            }
        }
        g0 = g1;
    }
    result
}


/// Extract a contiguous row range `[r0, r1)` of every column as a new matrix
/// (used to split the concatenated unrestricted amplitude block per sector).
fn slice_z_rows(z_block: &MatrixFull<f64>, r0: usize, r1: usize) -> MatrixFull<f64> {
    let nrow = z_block.size[0];
    let m = z_block.size[1];
    let nr = r1 - r0;
    let mut out = MatrixFull::new([nr, m], 0.0);
    for s in 0..m {
        out.data[s * nr..(s + 1) * nr]
            .copy_from_slice(&z_block.data[s * nrow + r0..s * nrow + r1]);
    }
    out
}


/// AO-mode kernel block over a block of trial vectors — unified for both
/// reference types.
///
/// The amplitude block `z_block` is `[dim_total, m]` with per-sector rows
/// laid out as `i + a*occ_s` (RHF: one sector of `dim` rows; UHF: alpha rows
/// first, then beta — PySCF `tdscf/uhf.py` collinear response):
/// - **fxc**: RHF applies the `[g,nvar,nvar]` kernel; UHF the spin-polarized
///   `[g,nvar,2,nvar,2]` blocks $f^{\sigma\tau}$ coupling the two spin
///   response densities (DM driver: batched `NIMatmul`; SEMITRANS:
///   per-sector grid-chunked application in the MO amplitude space);
/// - **Coulomb**: RHF: `coulomb_factor × J[P_s]` (xlet: singlet 2 / R 1 /
///   triplet 0). UHF: J is spin-blind — it responds to the total transition
///   density $P^\alpha_s + P^\beta_s$, so ONE batched RI-J call over all
///   n_sec·m densities serves every sector (each gets $J[P^\alpha_s] +
///   J[P^\beta_s]$, unit weight — no restricted singlet factor);
/// - **Exchange**: acts within a spin sector only — one batched RI-K call per
///   sector with that sector's MO coefficients.
fn ao_kernel_block(
    scf: &SCF,
    ao_data: &mut TDDFTData,
    z_block: &MatrixFull<f64>,
    xlet: char,
    is_b: bool,
) -> MatrixFull<f64> {
    let alpha_hybrid = ao_data.alpha_hybrid;
    let is_uhf = ao_data.is_uhf();
    let n_sec = ao_data.n_sectors();
    let sectors = crate::ri_tddft::utils::tddft_sector_params(scf);
    let dim_total: usize = sectors.iter().map(|sec| sec.dim()).sum();
    let m = z_block.size[1];
    let nao = scf.mol.num_basis;
    // Restricted singlet/triplet Coulomb weight; unrestricted: unit weight.
    let coulomb_factor = if xlet == 'S' { 2.0 } else if xlet == 'R' { 1.0 } else { 0.0 };

    let mut result = MatrixFull::new([dim_total, m], 0.0);
    if dim_total == 0 || m == 0 {
        return result;
    }

    // ── per-sector z blocks and transition densities ──
    // (coefficient borrows are scoped to each call so the fxc step below can
    // take `ao_data` mutably for the NIMatmul cache)
    let device = DeviceBLAS::default();
    let t0 = Instant::now();
    let mut z_sectors: Vec<MatrixFull<f64>> = Vec::with_capacity(n_sec);
    {
        let mut base = 0usize;
        for sec in sectors.iter() {
            z_sectors.push(slice_z_rows(z_block, base, base + sec.dim()));
            base += sec.dim();
        }
    }
    let build_p = |c_o: &MatrixFull<f64>, c_v: &MatrixFull<f64>,
                   z_sec: &MatrixFull<f64>, occ: usize, vir: usize|
        -> Vec<MatrixFull<f64>> {
        (0..m)
            .map(|s| {
                if occ == 0 || vir == 0 {
                    return MatrixFull::new([nao, nao], 0.0);
                }
                let z: Vec<f64> = (0..occ * vir).map(|r| z_sec[[r, s]]).collect();
                transition_density(c_o, c_v, &z, nao, occ, vir)
            })
            .collect()
    };
    let p_sectors: Vec<Vec<MatrixFull<f64>>> = sectors.iter().enumerate()
        .map(|(i_sec, sec)| build_p(
            &ao_data.c_occ[i_sec], &ao_data.c_vir[i_sec],
            &z_sectors[i_sec], sec.occ_size, sec.vir_size))
        .collect();
    add_ns(&T_TDEN, t0);

    // ── fxc: DM (batched NIMatmul, AO block) or MO/SEMITRANS (occ/vir-reduced,
    //      direct MO-amplitude output) ──
    let t0 = Instant::now();
    let f_fxc_block = match ao_data.fxc_driver {
        Some(FxcDriver::DM) => Some(fxc_matvec_ao_batched(ao_data, &p_sectors, &device)),
        _ => None,
    };
    let fxc_mo_block = match ao_data.fxc_driver {
        Some(FxcDriver::MO | FxcDriver::SEMITRANS) => {
            Some(fxc_mo_matvec(scf, ao_data, z_block, &sectors, &device))
        }
        _ => None,
    };
    add_ns(&T_FXC, t0);

    // ── J: one batched call over ALL sectors' transition densities; columns
    //      [σ·m .. (σ+1)·m) = J[P^σ_s] ──
    let t0 = Instant::now();
    let need_j = is_uhf || coulomb_factor != 0.0;
    let j_block = if need_j {
        let mut p_all: Vec<MatrixFull<f64>> = Vec::with_capacity(n_sec * m);
        for p_sec in &p_sectors {
            p_all.extend(p_sec.iter().cloned());
        }
        Some(get_j_ao_batched(scf, &p_all)) // [nao*nao, n_sec*m]
    } else {
        None
    };
    add_ns(&T_J, t0);

    // ── K per spin sector (exchange is same-spin only) ──
    let t0 = Instant::now();
    let k_sectors: Vec<Option<MatrixFull<f64>>> = if alpha_hybrid.abs() > 1e-15 {
        sectors.iter().enumerate()
            .map(|(i_sec, sec)| {
                if sec.occ_size > 0 && sec.vir_size > 0 {
                    Some(get_k_ao_batched(
                        scf,
                        &ao_data.c_occ[i_sec],
                        &ao_data.c_vir[i_sec],
                        &z_sectors[i_sec],
                        &p_sectors[i_sec],
                    ))
                } else {
                    None
                }
            })
            .collect()
    } else {
        (0..n_sec).map(|_| None).collect()
    };
    add_ns(&T_K, t0);

    // ── per-sector assembly + contract back ──
    let t0 = Instant::now();
    for (i_sec, sec) in sectors.iter().enumerate() {
        let occ = sec.occ_size;
        let vir = sec.vir_size;
        if occ == 0 || vir == 0 {
            continue;
        }
        let row0 = sector_row0_of(&sectors, i_sec);
        let dim = occ * vir;
        let k_blk = &k_sectors[i_sec];
        for s in 0..m {
            let mut f_total = MatrixFull::new([nao, nao], 0.0);
            // Coulomb: RHF — coulomb_factor × J[P_s]; UHF — J responds to BOTH
            // spin densities (unit weight each).
            if let Some(jb) = &j_block {
                if is_uhf {
                    for tau in 0..n_sec {
                        let base = (tau * m + s) * nao * nao;
                        for idx in 0..nao * nao {
                            f_total.data[idx] += jb.data[base + idx];
                        }
                    }
                } else {
                    let base = s * nao * nao;
                    for idx in 0..nao * nao {
                        f_total.data[idx] += coulomb_factor * jb.data[base + idx];
                    }
                }
            }
            if !is_b {
                if let Some(kb) = k_blk {
                    let base = s * nao * nao;
                    for idx in 0..nao * nao {
                        f_total.data[idx] -= alpha_hybrid * kb.data[base + idx];
                    }
                }
            }
            if let Some(fb) = &f_fxc_block {
                // spin-σ fxc potential of trial vector s
                let base = (i_sec * m + s) * nao * nao;
                for idx in 0..nao * nao {
                    f_total.data[idx] += fb.data[base + idx];
                }
            }
            let kernel_mo = contract_back(&f_total, &ao_data.c_occ[i_sec], &ao_data.c_vir[i_sec], occ, vir);
            for r in 0..dim {
                result[[row0 + r, s]] += kernel_mo[r];
            }
            if is_b {
                // B-block exchange: -alpha (C_vir^σᵀ K C_occ^σ)ᵀ (from
                // K[Pᵀ] = K[P]ᵀ), same index trick as the restricted path.
                if let Some(kb) = k_blk {
                    let base = s * nao * nao;
                    let k_col = MatrixFull::from_vec(
                        [nao, nao],
                        kb.data[base..base + nao * nao].to_vec(),
                    )
                    .unwrap();
                    let k_mo_v = contract_back(
                        &k_col,
                        &ao_data.c_vir[i_sec],
                        &ao_data.c_occ[i_sec],
                        vir,
                        occ,
                    );
                    for a in 0..vir {
                        for i in 0..occ {
                            result[[row0 + i + a * occ, s]] -= alpha_hybrid * k_mo_v[a + i * vir];
                        }
                    }
                }
            }
            if let Some(fm) = &fxc_mo_block {
                // MO/SEMITRANS fxc is already in MO amplitudes (concatenated layout)
                for r in 0..dim {
                    result[[row0 + r, s]] += fm[[row0 + r, s]];
                }
            }
        }
    }
    add_ns(&T_CONT, t0);
    result
}

/// Cumulative row offset of sector `i_sec` in the concatenated amplitude block.
fn sector_row0_of(sectors: &[TddftSector], i_sec: usize) -> usize {
    sectors.iter().take(i_sec).map(|sec| sec.dim()).sum()
}


pub fn a_matvec_ao_batched(
    scf: &SCF,
    ao_data: &mut TDDFTData,
    z_block: &MatrixFull<f64>,
    xlet: char,
) -> MatrixFull<f64> {
    let t_clos = Instant::now();
    let sectors = crate::ri_tddft::utils::tddft_sector_params(scf);
    let m = z_block.size[1];

    let mut result = ao_kernel_block(scf, ao_data, z_block, xlet, false);

    // Per-sector diagonal contribution (each sector uses its own spin's
    // orbital energies; elementwise on the block)
    let mut base = 0usize;
    for (i_sec, sec) in sectors.iter().enumerate() {
        let ks = &scf.eigenvalues[i_sec];
        for s in 0..m {
            for a in 0..sec.vir_size {
                for i in 0..sec.occ_size {
                    let idx = base + i + a * sec.occ_size;
                    result[[idx, s]] += (ks[sec.lumo + a] - ks[sec.start_mo + i]) * z_block[[idx, s]];
                }
            }
        }
        base += sec.dim();
    }
    add_ns(&T_CLOS, t_clos);
    N_AO_CLOS.fetch_add(1, Ordering::Relaxed);
    result
}

/// Build the full A matrix `[dim, dim]` directly (dense small-system path).
///
/// $A_{ia,jb} = (\varepsilon_a - \varepsilon_i)\delta_{ij}\delta_{ab} + \text{kernel}$,
/// constructed by applying the AO kernel block to the identity — one batched
/// J/K/fxc call across all `dim` columns instead of per-vector matvecs.
/// Unrestricted: the concatenated `[dim_a + dim_b]` space with per-sector
/// diagonal blocks.
pub fn build_a_ao(scf: &SCF, ao_data: &mut TDDFTData, xlet: char) -> MatrixFull<f64> {
    let sectors = crate::ri_tddft::utils::tddft_sector_params(scf);
    let dim_total: usize = sectors.iter().map(|sec| sec.dim()).sum();

    // Identity block: each column is one unit amplitude vector e_(ia).
    let mut identity = MatrixFull::new([dim_total, dim_total], 0.0);
    for idx in 0..dim_total {
        identity[[idx, idx]] = 1.0;
    }

    let mut a_full = ao_kernel_block(scf, ao_data, &identity, xlet, false);

    // Per-sector diagonal blocks (ε_a − ε_i).
    let mut base = 0usize;
    for (i_sec, sec) in sectors.iter().enumerate() {
        let ks = &scf.eigenvalues[i_sec];
        for a in 0..sec.vir_size {
            for i in 0..sec.occ_size {
                let idx = base + i + a * sec.occ_size;
                a_full[[idx, idx]] += ks[sec.lumo + a] - ks[sec.start_mo + i];
            }
        }
        base += sec.dim();
    }
    a_full
}

pub fn b_matvec_ao_batched(
    scf: &SCF,
    ao_data: &mut TDDFTData,
    z_block: &MatrixFull<f64>,
    xlet: char,
) -> MatrixFull<f64> {
    let t_clos = Instant::now();
    let result = ao_kernel_block(scf, ao_data, z_block, xlet, true);
    add_ns(&T_CLOS, t_clos);
    N_AO_CLOS.fetch_add(1, Ordering::Relaxed);
    result
}

/// Full A-block matvec, unrestricted (UKS) AO mode: kernel block + per-sector
pub fn build_b_ao(scf: &SCF, ao_data: &mut TDDFTData, xlet: char) -> MatrixFull<f64> {
    let sectors = crate::ri_tddft::utils::tddft_sector_params(scf);
    let dim_total: usize = sectors.iter().map(|sec| sec.dim()).sum();
    let mut identity = MatrixFull::new([dim_total, dim_total], 0.0);
    for idx in 0..dim_total {
        identity[[idx, idx]] = 1.0;
    }
    ao_kernel_block(scf, ao_data, &identity, xlet, true)
}

// ══════════════════════════════════════════════════════════════════
// Tests: validate AO contractions against naive four-index references
// ══════════════════════════════════════════════════════════════════

