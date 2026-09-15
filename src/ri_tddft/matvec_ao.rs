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

use crate::scf_io::SCF;
use crate::dft::num_int::FXCMatvecData;
use crate::dft::Grids;
use crate::dft::numint_matmul::nimatmul::NIMatmul;
use crate::dft::numint_matmul::resp_rks::eval_vxc_fxc_from_rho;
use crate::dft::xceff::prelude::{XCDenType, XCSpin};
use crate::ri_jk::util::get_cint_mol;
use crate::ri_tddft::tddft::FxcDriver;
use crate::ri_tddft::utils::tddft_occupation_parameters;
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
/// Dimensions are inferred from `c_occ`/`c_vir` (`nao = c_occ.rows`, `occ = c_occ.cols`, `vir = c_vir.cols`).
pub fn transition_density(
    c_occ: &MatrixFull<f64>,
    c_vir: &MatrixFull<f64>,
    z: &[f64],
) -> MatrixFull<f64> {
    let nao = c_occ.size[0];
    let occ_size = c_occ.size[1];
    let vir_size = c_vir.size[1];
    assert_eq!(c_vir.size[0], nao, "c_occ and c_vir must share nao");
    assert_eq!(z.len(), occ_size * vir_size, "z length must be occ*vir");
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
/// Returns `[nao*nao, m]` (column $\mathbb{A}$ = flattened K, matching
/// `f_fxc_block`). Always evaluates $K[D^{\mathbb{A}}]$ with the untransposed
/// density; the B-block exchange is derived by the caller from the identity
/// $K[D^{\mathrm{T}}] = K[D]^{\mathrm{T}}$ (each $M_Q$ is symmetric).
fn get_k_ao_batched(
    scf: &SCF,
    ao_data: &TDDFTData,
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
    let occ_size = ao_data.c_occ.as_ref().unwrap().size[1];
    let vir_size = ao_data.c_vir.as_ref().unwrap().size[1];

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
            let c_vir_v = ao_data.c_vir.as_ref().unwrap().to_rstsr_view(&device);
            let c_occ_v = ao_data.c_occ.as_ref().unwrap().to_rstsr_view(&device);
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
            let c_occ_v = ao_data.c_occ.as_ref().unwrap().to_rstsr_view(&device);
            let c_vir_v = ao_data.c_vir.as_ref().unwrap().to_rstsr_view(&device);
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
/// Dimensions are inferred from `c_occ`/`c_vir` (`occ = c_occ.cols`, `vir = c_vir.cols`).
pub fn contract_back(
    f_full: &MatrixFull<f64>,
    c_occ: &MatrixFull<f64>,
    c_vir: &MatrixFull<f64>,
) -> Vec<f64> {
    let occ_size = c_occ.size[1];
    let vir_size = c_vir.size[1];
    let nao = f_full.size[0];
    assert_eq!(c_occ.size[0], nao, "c_occ rows must match F rows");
    assert_eq!(c_vir.size[0], nao, "c_vir rows must match F rows");
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
/// Returns `[nao, nao*m]` with column block $\mathbb{A}$ at
/// `data[\mathbb{A}·nao·nao ..]` (matching the assembly loops' `base`).
fn fxc_matvec_ao_batched(
    ao_data: &mut TDDFTData,
    p_block: &[MatrixFull<f64>],
    device: &DeviceBLAS,
) -> MatrixFull<f64> {
    use crate::dft::xceff::prelude::XCSpin;
    let nao = p_block[0].size[0];
    let m = p_block.len();
    let ni = ao_data.ni.as_mut().expect("AO-mode fxc requires NIMatmul");
    let ngrids = ni.coords.len();
    let den_type = ao_data.den_type.expect("AO-mode fxc requires den_type");

    let mut out = MatrixFull::new([nao, nao * m], 0.0);
    // make_rho_from_dm assumes a symmetric density for its SIGMA response;
    // symmetrize the transition densities first (exact — the fxc kernel is
    // symmetric in μν so the fxc response only sees the symmetric part).
    let p_sym_block: Vec<MatrixFull<f64>> = p_block.iter().map(symmetrize_density).collect();
    let dms_slice: &[MatrixFull<f64>] = &p_sym_block;
    let dms = dms_slice.to_rstsr(device); // [nao, nao, m]
    let fxc_eff_view = ao_data.fxc_eff.as_ref().unwrap().view();

    if !ao_data.grid_batch {
        // Full-grid path: one batched call over all sets.
        let dm_views: Vec<TsrView> = (0..m).map(|s| dms.i((.., .., s))).collect();
        let t0 = Instant::now();
        let rho1 = ni.make_rho_from_dm(&dm_views, den_type); // [ngrids, nvar, m]
        add_ns(&T_FXC_RHO, t0);
        let t0 = Instant::now();
        let f_fxc = ni.make_fxc_pot_with_eff(fxc_eff_view, rho1.view(), den_type, XCSpin::Unpolarized);
        add_ns(&T_FXC_POT, t0);
        for s in 0..m {
            for (r, v) in f_fxc.i((.., .., s)).iter().enumerate() {
                out.data[s * nao * nao + r] = *v;
            }
        }
    } else {
        // Grid-batched path: evaluate per grid batch (all sets), accumulating.
        let nbatch = ni.nbatch;
        for start in (0..ngrids).step_by(nbatch) {
            let end = (start + nbatch).min(ngrids);
            let mut ni_batch = ni.split_batch(start, end);
            let dm_views: Vec<TsrView> = (0..m).map(|s| dms.i((.., .., s))).collect();
            let t0 = Instant::now();
            let rho1_b = ni_batch.make_rho_from_dm(&dm_views, den_type); // [nb, nvar, m]
            add_ns(&T_FXC_RHO, t0);
            let fxc_eff_b = fxc_eff_view.i((start..end));
            let t0 = Instant::now();
            let f_b = ni_batch.make_fxc_pot_with_eff(fxc_eff_b, rho1_b.view(), den_type, XCSpin::Unpolarized);
            add_ns(&T_FXC_POT, t0);
            for s in 0..m {
                for (r, v) in f_b.i((.., .., s)).iter().enumerate() {
                    out.data[s * nao * nao + r] += *v;
                }
            }
        }
    }
    out
}

/// "mo" fxc driver: the MO-mode fxc algorithm (occ/vir-reduced kernel
/// application) using the cached
/// Cached occ-side MO-on-grid projection tables (`psi_occ`/`psi_occ_grad`);
///
/// Per trial vector $z_{ia}$, with $\psi_i(g)=\sum_\mu C_{\mu i}\varphi_\mu(g)$:
///
/// $$\rho_0(g) = \sum_{ia} z_{ia}\,\psi_i(g)\psi_a(g), \qquad
///   \rho_{d+1}(g) = \sum_{ia} z_{ia}\,(\partial_d\psi_i\,\psi_a + \psi_i\,\partial_d\psi_a)(g)$$
///
/// $$v_{1,\alpha}(g) = w(g)\sum_\beta f^{\rm xc}_{\alpha\beta}(g)\,\rho_\beta(g), \qquad
///   E_{ia} = \sum_g \Lambda^\alpha_{ia}(g)\,v_{1,\alpha}(g)$$
///
/// with $\Lambda^0_{ia} = \psi_i\psi_a$ and $\Lambda^{d+1}_{ia} = \partial_d\psi_i\,\psi_a + \psi_i\,\partial_d\psi_a$.
/// Every contraction runs in the $(n_\mathrm{occ}, n_\mathrm{vir}, n_\mathrm{grid})$ space —
/// no $[n_\mathrm{ao}, n_\mathrm{ao}, m]$ intermediates, no `contract_back`.
///
/// Returns MO amplitudes `[dim, m]` (added directly to the matvec result).
fn fxc_mo_matvec(
    scf: &SCF,
    ao_data: &TDDFTData,
    z_block: &MatrixFull<f64>,
    device: &DeviceBLAS,
) -> MatrixFull<f64> {
    // fxc matvec for the occ/vir-reduced drivers (`"mo"` and `"semitrans"`).
    // Both cache the small occ-side tables (psi_occ [+grads]); they differ only
    // in how the vir side is provided, branching at four points:
    //   * `"mo"`: psi_vir(+grads) is projected per grid batch into reusable
    //     buffers and consumed by every GEMM (batch buffers freed per batch).
    //   * `"semitrans"`: C_vir is folded into the amplitudes
    //     (Z_st = Z_stack * C_vir^T, one GEMM per call) and the vir side stays
    //     in the RAW AO basis — the batch AO slab is kept alive through the
    //     chunk loop and no psi_vir is ever formed.
    // Per chunk (all m sets stacked; shapes use the driver's "left operand",
    // psi_vir_c [cg,nvir] or phi_c [cg,nao]):
    //   Q^b[g, s*nocc+i] = sum_b left^b[g,b] * Z^right[s*nocc+i, b]
    //   rho_0[g]         = sum_i psi_occ[g,i] * Q_0[g, i-block]
    //   rho_{d+1}[g]     = vecdot(d_d psi_occ, Q_0) + vecdot(psi_occ, Q_{d+1})
    //   v1_a(g)          = w(g) sum_b fxc_eff[g,a,b] rho_b(g)
    //   S^a[(s*nocc+i), g] = v1_a^s(g) * psi^a_i(g)
    //   "mo":        E_stack += S^a * left^a_c
    //   "semitrans": E_stack += (S^a * phi_c) * C_vir^T
    let psi_occ = ao_data.psi_occ.as_ref().expect("mo fxc requires psi_occ"); // [ng, occ] cached
    let pog_cached = ao_data.psi_occ_grad.as_ref(); // [3, ng, occ] cached (GGA only)
    let gga = pog_cached.is_some();
    let nvar = if gga { 4 } else { 1 };
    let c_vir = ao_data.c_vir.as_ref().expect("mo fxc requires c_vir");
    let fxc_eff = ao_data.fxc_eff.as_ref().unwrap().view(); // [ng, nvar, nvar]
    let dim = z_block.size[0];
    let m = z_block.size[1];
    let occ_size = psi_occ.shape()[1];
    let vir_size = c_vir.size[1];
    let ng = psi_occ.shape()[0];
    let weights = &scf.grids.as_ref().expect("DFT grids required for mo fxc").weights;
    let fxc_raw = fxc_eff.raw();
    let fxc_off = fxc_eff.offset();
    let deriv = if gga { 1 } else { 0 };
    let st = ao_data.fxc_driver == Some(FxcDriver::SEMITRANS);

    // Set-stacked amplitudes: Z_stack[(s*nocc+i), b] = z_s[i,b] — built ONCE per call,
    // so the per-(chunk, set) z re-assembly disappears and every GEMM below runs with
    // N = m*nocc (fat on both axes).
    let z_stack = rt::asarray((&z_block.data, [occ_size, vir_size, m].f(), device))
        .swapaxes(1, 2)
        .into_contig(FlagOrder::F)
        .into_shape([m * occ_size, vir_size]);
    // semitrans: fold C_vir into the amplitude side.
    let ztilde_stack = if st {
        let c_vir_v = c_vir.to_rstsr_view(device);
        let mut zt = rt::zeros(([m * occ_size, c_vir.size[0]].f(), device));
        zt.matmul_from(&z_stack, &c_vir_v.t(), 1.0, 0.0);
        Some(zt)
    } else {
        None
    };

    let ni = ao_data.ni.as_ref().expect("mo fxc requires NIMatmul");
    let c_vir_v = c_vir.to_rstsr_view(device);
    let nbatch = ni.nbatch;
    let nchunk = 1536usize;
    let nb_max = nbatch.min(ng);
    // "mo": reusable vir-side batch buffers (allocated once per call, refilled per batch).
    let mut pv_all = if st {
        None
    } else {
        Some(rt::zeros(([nb_max, vir_size].f(), device)))
    };
    let mut pvg_all = if st || !gga {
        None
    } else {
        Some(
            (0..3)
                .map(|_| rt::zeros(([nb_max, vir_size].f(), device)))
                .collect::<Vec<_>>(),
        )
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
            pv.i_mut((..nb, ..))
                .matmul_from(&ao_b.i((.., .., 0)), &c_vir_v, 1.0, 0.0);
            if gga {
                let pvg = pvg_all.as_mut().unwrap();
                for d in 0..3 {
                    pvg[d]
                        .i_mut((..nb, ..))
                        .matmul_from(&ao_b.i((.., .., 1 + d)), &c_vir_v, 1.0, 0.0);
                }
            }
        }
        add_ns(&T_FXC_RHO, t_proj);

        // ── parallel chunks within this batch (set-stacked kernels) ──
        let ntask_b = nb.div_ceil(nchunk);
        let chunk_results: Vec<Vec<f64>> = (0..ntask_b)
            .into_par_iter()
            .map(|ic| {
                let cg0 = g0 + ic * nchunk;
                let cg1 = (cg0 + nchunk).min(g1);
                let cg = cg1 - cg0;
                let lg0 = cg0 - g0;
                let lg1 = cg1 - g0;
                let mut e_out = vec![0.0_f64; m * occ_size * vir_size];

                let po_c = psi_occ.i((cg0..cg1, ..)); // [cg, occ]
                // driver-dependent GEMM operands:
                //   "mo":        left = psi_vir_c  [cg, nvir], right = Z_stack
                //   "semitrans": left = phi_c       [cg, nao],  right = Z_st
                let left_c = if st {
                    ao_b.i((lg0..lg1, .., 0))
                } else {
                    pv_all.as_ref().unwrap().i((lg0..lg1, ..))
                };
                let right = if st {
                    ztilde_stack.as_ref().unwrap()
                } else {
                    &z_stack
                };

                // ── rho build: ONE stacked GEMM per component (all m sets at once) ──
                // q[g, s*nocc+i] = sum_b left[g,b] * right[s*nocc+i, b]
                // (GEMM [cg,nao|nvir]-[nao|nvir, m*nocc], K = nvir|nao, N = m*nocc)
                let mut q_stack = rt::zeros(([cg, m * occ_size].f(), device));
                q_stack.matmul_from(&left_c, &right.t(), 1.0, 0.0);
                let mut qd_stack = if gga {
                    Some(rt::zeros(([cg, m * occ_size].f(), device)))
                } else {
                    None
                };
                // rho_bufs[s][beta][g]
                let mut rho_bufs: Vec<Vec<Vec<f64>>> =
                    vec![vec![vec![0.0; cg]; 4]; m];
                for s in 0..m {
                    let q_s = q_stack.i((.., s * occ_size..(s + 1) * occ_size));
                    let mut rho0 = rt::zeros(([cg], device));
                    rho0.i_mut((..)).vecdot_from(&q_s, &po_c, 1);
                    for g in 0..cg {
                        rho_bufs[s][0][g] = rho0[[g]];
                    }
                }
                if gga {
                    let pog = pog_cached.unwrap(); // [3, ng, occ]
                    for d in 0..3 {
                        // "mo": grad left operand from the projected batch buffer;
                        // "semitrans": raw AO gradient slab of this batch.
                        let grad_c = if st {
                            ao_b.i((lg0..lg1, .., 1 + d))
                        } else {
                            pvg_all.as_ref().unwrap()[d].i((lg0..lg1, ..))
                        };
                        qd_stack
                            .as_mut()
                            .unwrap()
                            .matmul_from(&grad_c, &right.t(), 1.0, 0.0);
                        for s in 0..m {
                            let q_s = q_stack.i((.., s * occ_size..(s + 1) * occ_size));
                            let qd_s = qd_stack
                                .as_ref()
                                .unwrap()
                                .i((.., s * occ_size..(s + 1) * occ_size));
                            // term 1: rho1_d[g] = sum_i (d_d psi_occ)[g,i] * q[g,i]
                            let mut r_d = rt::zeros(([cg], device));
                            r_d.i_mut((..)).vecdot_from(&pog.i((d, cg0..cg1, ..)), &q_s, 1);
                            // term 2: rho2_d[g] = sum_i psi_occ[g,i] * qd[g,i]
                            let mut r_t = rt::zeros(([cg], device));
                            r_t.i_mut((..)).vecdot_from(&po_c, &qd_s, 1);
                            for g in 0..cg {
                                rho_bufs[s][1 + d][g] = r_d[[g]] + r_t[[g]];
                            }
                        }
                    }
                }

                // ── weighted kernel per set: v1_a(g) = w(g) sum_b fxc_eff[g,a,b] rho_b(g) ──
                let mut v1_bufs: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; cg]; 4]; m];
                for s in 0..m {
                    for alpha in 0..nvar {
                        for beta in 0..nvar {
                            let base = fxc_off + alpha * ng + beta * nvar * ng + cg0;
                            let r = &rho_bufs[s][beta];
                            let v = &mut v1_bufs[s][alpha];
                            for g in 0..cg {
                                v[g] += fxc_raw[base + g] * r[g];
                            }
                        }
                        for g in 0..cg {
                            v1_bufs[s][alpha][g] *= weights[cg0 + g];
                        }
                    }
                }

                // ── back-projection: stacked scalings + GEMM per term (all sets) ──
                //   "mo":        E_stack += S^a * left^a_c              (one GEMM)
                //   "semitrans": G = S^a * phi_c; E_stack += G * C_vir   (two GEMMs;
                //                g_stack must be REWRITTEN (beta = 0) per term)
                let t_pot = Instant::now();
                let mut s_stack = vec![0.0_f64; m * occ_size * cg]; // reused build buffer
                let mut e_stack = rt::zeros(([m * occ_size, vir_size].f(), device));
                let mut g_stack = if st {
                    Some(rt::zeros(([m * occ_size, c_vir.size[0]].f(), device)))
                } else {
                    None
                };
                // alpha = 0: S[(s*nocc+i), g] = v1_0^s(g) * psi_occ[g,i]
                // (measured: the (g,i) order below beats the tilted (i,g-block) order —
                // the scattered po_c reads are L2-resident hits, cheaper than the
                // reordered write pattern)
                for s in 0..m {
                    let v = &v1_bufs[s][0];
                    let rbase = s * occ_size;
                    for g in 0..cg {
                        let w = v[g];
                        for i in 0..occ_size {
                            // f-order [m*nocc, cg]: idx = row + col*nrow
                            s_stack[rbase + i + g * (m * occ_size)] = po_c[[g, i]] * w;
                        }
                    }
                }
                let s0 = rt::asarray((&s_stack, [m * occ_size, cg].f(), device));
                if st {
                    let g_s = g_stack.as_mut().unwrap();
                    g_s.matmul_from(&s0, &left_c, 1.0, 0.0);
                    e_stack.matmul_from(g_s, &c_vir_v, 1.0, 0.0);
                } else {
                    e_stack.matmul_from(&s0, &left_c, 1.0, 0.0);
                }
                if gga {
                    let pog = pog_cached.unwrap(); // [3, ng, occ]
                    for d in 0..3 {
                        let grad_c = if st {
                            ao_b.i((lg0..lg1, .., 1 + d))
                        } else {
                            pvg_all.as_ref().unwrap()[d].i((lg0..lg1, ..))
                        };
                        // term 1: S = v1_{d+1}^s(g) * (d_d psi_occ)[g,i]
                        for s in 0..m {
                            let v = &v1_bufs[s][1 + d];
                            let rbase = s * occ_size;
                            for g in 0..cg {
                                let w = v[g];
                                for i in 0..occ_size {
                                    s_stack[rbase + i + g * (m * occ_size)] = pog[[d, cg0 + g, i]] * w;
                                }
                            }
                        }
                        let s1 = rt::asarray((&s_stack, [m * occ_size, cg].f(), device));
                        if st {
                            let g_s = g_stack.as_mut().unwrap();
                            g_s.matmul_from(&s1, &left_c, 1.0, 0.0);
                            e_stack.matmul_from(g_s, &c_vir_v, 1.0, 1.0);
                        } else {
                            e_stack.matmul_from(&s1, &left_c, 1.0, 1.0);
                        }
                        // term 2: S = v1_{d+1}^s(g) * psi_occ[g,i]
                        for s in 0..m {
                            let v = &v1_bufs[s][1 + d];
                            let rbase = s * occ_size;
                            for g in 0..cg {
                                let w = v[g];
                                for i in 0..occ_size {
                                    s_stack[rbase + i + g * (m * occ_size)] = po_c[[g, i]] * w;
                                }
                            }
                        }
                        let s2 = rt::asarray((&s_stack, [m * occ_size, cg].f(), device));
                        if st {
                            let g_s = g_stack.as_mut().unwrap();
                            g_s.matmul_from(&s2, &grad_c, 1.0, 0.0);
                            e_stack.matmul_from(g_s, &c_vir_v, 1.0, 1.0);
                        } else {
                            e_stack.matmul_from(&s2, &grad_c, 1.0, 1.0);
                        }
                    }
                }

                // scatter E_stack[(s*nocc+i), a] -> e_out[s][i,a]
                for s in 0..m {
                    let obase = s * occ_size * vir_size;
                    for a in 0..vir_size {
                        for i in 0..occ_size {
                            e_out[obase + i * vir_size + a] = e_stack[[s * occ_size + i, a]];
                        }
                    }
                }
                e_out
            })
            .collect::<Vec<_>>();

        for e_out in chunk_results.iter() {
            for s in 0..m {
                let obase = s * occ_size * vir_size;
                for a in 0..vir_size {
                    for i in 0..occ_size {
                        result[[i + a * occ_size, s]] += e_out[obase + i * vir_size + a];
                    }
                }
            }
        }
        g0 = g1;
    }
    result
}


fn ao_kernel_block(
    scf: &SCF,
    ao_data: &mut TDDFTData,
    z_block: &MatrixFull<f64>,
    xlet: char,
    is_b: bool,
) -> MatrixFull<f64> {
    let alpha_hybrid = ao_data.alpha_hybrid;
    let (_start_mo, _, occ_size, vir_size, _homo, _lumo) = tddft_occupation_parameters(scf);
    let dim = occ_size * vir_size;
    let m = z_block.size[1];
    let nao = scf.mol.num_basis;
    let coulomb_factor = if xlet == 'S' { 2.0 } else if xlet == 'R' { 1.0 } else { 0.0 };

    let mut result = MatrixFull::new([dim, m], 0.0);

    // Build P for each column, and batch the fxc over all columns.
    let device = DeviceBLAS::default();
    let t0 = Instant::now();
    let mut p_block: Vec<MatrixFull<f64>> = Vec::with_capacity(m);
    for s in 0..m {
        let z: Vec<f64> = (0..dim).map(|r| z_block[[r, s]]).collect();
        p_block.push(transition_density(ao_data.c_occ.as_ref().unwrap(), ao_data.c_vir.as_ref().unwrap(), &z));
    }
    add_ns(&T_TDEN, t0);

    // fxc: "dm" (assembled-density NIMatmul, AO-basis block) or "mo"
    // (occ/vir-reduced, direct MO-amplitude output).
    let t0 = Instant::now();
    let f_fxc_block = match ao_data.fxc_driver {
        Some(FxcDriver::DM) => Some(fxc_matvec_ao_batched(ao_data, &p_block, &device)),
        _ => None,
    };
    let fxc_mo_block = match ao_data.fxc_driver {
        Some(FxcDriver::MO | FxcDriver::SEMITRANS) => {
            Some(fxc_mo_matvec(scf, ao_data, z_block, &device))
        }
        _ => None,
    };
    add_ns(&T_FXC, t0);

    // Batched RI-J/K over all trial vectors (single ri_jk call).
    let t0 = Instant::now();
    let j_block = if coulomb_factor != 0.0 {
        Some(get_j_ao_batched(scf, &p_block))
    } else {
        None
    };
    add_ns(&T_J, t0);

    let t0 = Instant::now();
    // K[P] with the untransposed density for BOTH blocks; the B-block exchange
    // is derived below via K[Pᵀ] = K[P]ᵀ (each M_Q symmetric).
    let k_block = if alpha_hybrid.abs() > 1e-15 {
        Some(get_k_ao_batched(scf, ao_data, z_block, &p_block))
    } else {
        None
    };
    add_ns(&T_K, t0);

    // Per-column assembly + contract back
    let t0 = Instant::now();
    for s in 0..m {
        let base = s * nao * nao;
        let mut f_total = MatrixFull::new([nao, nao], 0.0);
        if let Some(jb) = &j_block {
            for idx in 0..nao * nao {
                f_total.data[idx] += coulomb_factor * jb.data[base + idx];
            }
        }
        if !is_b {
            if let Some(kb) = &k_block {
                for idx in 0..nao * nao {
                    f_total.data[idx] -= alpha_hybrid * kb.data[base + idx];
                }
            }
        }
        if let Some(fb) = &f_fxc_block {
            // fxc contribution for this vector (column s of the batched fxc block)
            for idx in 0..nao * nao {
                f_total.data[idx] += fb.data[base + idx];
            }
        }
        let kernel_mo = contract_back(&f_total, ao_data.c_occ.as_ref().unwrap(), ao_data.c_vir.as_ref().unwrap());
        for r in 0..dim {
            result[[r, s]] += kernel_mo[r];
        }
        if is_b {
            // B-block exchange: -alpha (C_virᵀ K C_occ)ᵀ  (from K[Pᵀ] = K[P]ᵀ,
            // M_Q symmetric). contract_back with swapped roles gives C_virᵀ K C_occ
            // at flat index a + i*vir; the transpose to [i + a*occ] is the loop.
            if let Some(kb) = &k_block {
                let k_col = MatrixFull::from_vec(
                    [nao, nao],
                    kb.data[base..base + nao * nao].to_vec(),
                )
                .unwrap();
                let k_mo_v = contract_back(
                    &k_col,
                    ao_data.c_vir.as_ref().unwrap(),
                    ao_data.c_occ.as_ref().unwrap(),
                );
                for a in 0..vir_size {
                    for i in 0..occ_size {
                        result[[i + a * occ_size, s]] -= alpha_hybrid * k_mo_v[a + i * vir_size];
                    }
                }
            }
        }
        if let Some(fm) = &fxc_mo_block {
            // "mo" fxc is already in MO amplitudes
            for r in 0..dim {
                result[[r, s]] += fm[[r, s]];
            }
        }
    }
    add_ns(&T_CONT, t0);
    result
}

pub fn a_matvec_ao_batched(
    scf: &SCF,
    ao_data: &mut TDDFTData,
    z_block: &MatrixFull<f64>,
    xlet: char,
) -> MatrixFull<f64> {
    let t_clos = Instant::now();
    let (start_mo, _, occ_size, vir_size, _homo, lumo) = tddft_occupation_parameters(scf);
    let dim = occ_size * vir_size;
    let m = z_block.size[1];
    let ks = &scf.eigenvalues[0];

    let mut result = ao_kernel_block(scf, ao_data, z_block, xlet, false);

    // Diagonal contribution (elementwise on the block)
    for s in 0..m {
        for a in 0..vir_size {
            for i in 0..occ_size {
                let idx = i + a * occ_size;
                result[[idx, s]] += (ks[lumo + a] - ks[start_mo + i]) * z_block[[idx, s]];
            }
        }
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
pub fn build_a_ao(scf: &SCF, ao_data: &mut TDDFTData, xlet: char) -> MatrixFull<f64> {
    let (start_mo, _, occ_size, vir_size, _homo, lumo) = tddft_occupation_parameters(scf);
    let dim = occ_size * vir_size;
    let ks = &scf.eigenvalues[0];

    // Identity block: each column is one unit amplitude vector e_(ia).
    let mut identity = MatrixFull::new([dim, dim], 0.0);
    for idx in 0..dim {
        identity[[idx, idx]] = 1.0;
    }

    let mut a_full = ao_kernel_block(scf, ao_data, &identity, xlet, false);

    // Diagonal matrix (ε_a − ε_i).
    for a in 0..vir_size {
        for i in 0..occ_size {
            a_full[[i + a * occ_size, i + a * occ_size]] += ks[lumo + a] - ks[start_mo + i];
        }
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

/// Build the full B matrix `[dim, dim]` directly (dense small-system path),
/// by applying the B kernel block to the identity.
pub fn build_b_ao(scf: &SCF, ao_data: &mut TDDFTData, xlet: char) -> MatrixFull<f64> {
    let (_start_mo, _, occ_size, vir_size, _homo, _lumo) = tddft_occupation_parameters(scf);
    let dim = occ_size * vir_size;
    let mut identity = MatrixFull::new([dim, dim], 0.0);
    for idx in 0..dim {
        identity[[idx, idx]] = 1.0;
    }
    ao_kernel_block(scf, ao_data, &identity, xlet, true)
}

// ══════════════════════════════════════════════════════════════════
// Tests: validate AO contractions against naive four-index references
// ══════════════════════════════════════════════════════════════════

