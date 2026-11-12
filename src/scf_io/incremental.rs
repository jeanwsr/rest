//! δD 增量 Fock v1：`[ctrl] incremental_fock = true` 激活。
//! J(D) 与 K(D) 对密度线性；当本次密度与上次实际构建时的密度之差
//! max|ΔD| < scf_acc_rho（与收敛判据同源）时重用缓存，误差与收敛精度同级。
//! 状态放进程级缓存，不侵入 SCF 构造路径。v2 方向：按 AO 对的 |ΔD| 加权界与部分重算
//! （LinK 风格）；PCM（对密度线性）接入在 solvent 侧，v1 未覆盖。
use std::sync::{Mutex, OnceLock};
use rayon::prelude::*;
use tensors::{MatrixFull, MatrixUpper, BasicMatrix};
use crate::mpi_io::MPIOperator;
use super::SCF;

struct Cache {
    dm: Vec<MatrixFull<f64>>,
    vj: Vec<MatrixUpper<f64>>,
    vk: Vec<MatrixUpper<f64>>,
    /// consecutive reuses since the last full rebuild
    stale: u32,
    /// the decision taken by this iteration's J call; K follows it so the two stay consistent
    reuse_now: bool,
    /// consecutive K spectral-increment updates since the last full K rebuild
    inc_streak: u32,
    /// density at the last K evaluation (full build or increment). The shared `dm`
    /// is refreshed by the J path before K runs, so the K increment must compare
    /// against its own base, not `dm`.
    dm_k: Option<Vec<MatrixFull<f64>>>,
    /// one-shot request for a full rebuild, set when the SCF criteria passed while
    /// the cached Fock was not fresh (increment or reuse). The energy reported at a
    /// "converged" point must come from a Fock built at exactly that density.
    force_full: bool,
}

/// reuse threshold as a multiple of `scf_acc_rho`: the Fock may lag the density by a
/// controlled amount mid-SCF (the direct-SCF "fock drift" trade), and the periodic
/// rebuild keeps the convergence criteria clean at the end. Tightened 10x -> 3x after
/// w30 measured a 8.4e-5 Ha energy lag at 10x (target 1e-5).
const REUSE_TOL_FACTOR: f64 = 3.0;
const MAX_STALE: u32 = 4;

/// J reuse tolerance multiplier, configurable via REST_J_REUSE_TOL.
fn j_reuse_tol_factor() -> f64 {
    std::env::var("REST_J_REUSE_TOL").ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(REUSE_TOL_FACTOR)
}

fn cache() -> &'static Mutex<Option<Cache>> {
    static C: OnceLock<Mutex<Option<Cache>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(None))
}

fn delta_max(cur: &[MatrixFull<f64>], prev: &[MatrixFull<f64>]) -> Option<f64> {
    if cur.len() != prev.len() || cur.is_empty() { return None; }
    let mut m = 0.0_f64;
    for (c, p) in cur.iter().zip(prev.iter()) {
        if c.size() != p.size() { return None; }
        for (a, b) in c.data.iter().zip(p.data.iter()) {
            let d = (a - b).abs();
            if d > m { m = d; }
        }
    }
    Some(m)
}

fn dm_delta(scf: &SCF, c: &Cache) -> Option<f64> {
    delta_max(&scf.density_matrix, &c.dm)
}

pub fn cached_vj(scf: &mut SCF, mpi_operator: &Option<MPIOperator>) -> Vec<MatrixUpper<f64>> {
    let mut g = cache().lock().unwrap();
    let tol = j_reuse_tol_factor() * scf.mol.ctrl.scf_acc_rho;
    if let Some(c) = g.as_mut() {
        // Once the density itself has reached the convergence criterion
        // (max|ΔD| < scf_acc_rho) the SCF is about to declare convergence: force a full
        // rebuild so the final energy is computed from a fresh Fock, not a reused one.
        let near_conv = dm_delta(scf, c).map_or(true, |d| d < scf.mol.ctrl.scf_acc_rho);
        let hit = !c.vj.is_empty()
            && !near_conv
            && !c.force_full
            && c.stale < MAX_STALE
            && dm_delta(scf, c).map_or(false, |d| d < tol);
        if hit {
            c.reuse_now = true;
            c.stale += 1;
            println!("[incremental] J reused (max|ΔD| = {:.2e} < {:.1e}, streak {})",
                dm_delta(scf, c).unwrap_or(0.0), tol, c.stale);
            return c.vj.clone();
        }
        if near_conv && !c.vj.is_empty() {
            println!("[incremental] density at convergence level (max|ΔD| < {:.1e}): full rebuild for the final Fock",
                scf.mol.ctrl.scf_acc_rho);
        } else if c.stale >= MAX_STALE && !c.vj.is_empty() {
            println!("[incremental] periodic full rebuild after {} reuses", c.stale);
        }
    }
    let vj = scf.generate_vj_with_ri_v_sync(1.0, mpi_operator);
    match g.as_mut() {
        Some(c) => {
            // this build used the current density, so the cache's reference density must be it
            c.reuse_now = false;
            if c.force_full { println!("[incremental] fresh-Fock validation rebuild (J)"); }
            c.force_full = false;
            c.stale = 0;
            c.dm = scf.density_matrix.clone();
            c.vj = vj.clone();
        }
        None => *g = Some(Cache { dm: scf.density_matrix.clone(), vj: vj.clone(), vk: Vec::new(), stale: 0, reuse_now: false, inc_streak: 0, dm_k: None, force_full: false }),
    }
    vj
}

pub fn cached_vk(scf: &mut SCF, scaling_factor: f64, mpi_operator: &Option<MPIOperator>) -> Vec<MatrixUpper<f64>> {
    let mut g = cache().lock().unwrap();
    // follow the decision the J call of this iteration already took, so J and K never mix
    // a stale and a fresh half within one Fock build
    if let Some(c) = g.as_ref() {
        if c.reuse_now && !c.vk.is_empty() && !c.force_full {
            println!("[incremental] K reused");
            return c.vk.clone();
        }
    }
    // ---- spectral K increment (REST_K_INC=1) ------------------------------------------
    // K is linear in the density, so K(D_new) = K(D_base) + K(ΔD) with ΔD = D_new − D_base.
    // ΔD is indefinite, so it is split spectrally into Δ⁺ − Δ⁻ (both PSD); the occupied-
    // orbital columns W√λ of each part feed the pair-screened kernel directly, and the
    // screening criterion is ABSOLUTE against the full-density scale (eps × scale(D))
    // rather than relative to Δ itself — a uniformly small Δ then retains fewer and fewer
    // rows as the SCF converges, which is the whole point of the update.
    let forced = g.as_ref().map_or(false, |c| c.force_full);
    if let Some(vk_inc) = if forced { None } else { k_spectral_increment(scf, g.as_ref(), scaling_factor) } {
        let mut g = g;
        let c = g.as_mut().unwrap();
        c.inc_streak += 1;
        c.dm_k = Some(scf.density_matrix.clone());
        c.vk = vk_inc;
        c.reuse_now = false;
        return c.vk.clone();
    }
    let vk = scf.generate_vk_with_ri_v(scaling_factor, scf.mol.ctrl.use_dm_only, mpi_operator);
    match g.as_mut() {
        Some(c) => { c.reuse_now = false; c.dm = scf.density_matrix.clone(); c.dm_k = Some(scf.density_matrix.clone()); c.vk = vk.clone(); if c.inc_streak > 0 { println!("[incremental] full K rebuild after {} increments", c.inc_streak); } c.inc_streak = 0; if c.force_full { println!("[incremental] fresh-Fock validation rebuild (K)"); c.force_full = false; } }
        None => *g = Some(Cache { dm: scf.density_matrix.clone(), vj: Vec::new(), vk: vk.clone(), stale: 0, reuse_now: false, inc_streak: 0, dm_k: Some(scf.density_matrix.clone()), force_full: false }),
    }
    vk
}

/// Try the spectral K increment. `None` = "not applicable / disabled", the caller falls
/// back to a full K build.
fn k_spectral_increment(scf: &SCF, cache: Option<&Cache>, scaling_factor: f64) -> Option<Vec<MatrixUpper<f64>>> {
    if std::env::var("REST_K_INC").as_deref() != Ok("1") { return None; }
    let c = cache?;
    if c.vk.is_empty() || c.vk.iter().any(|v| v.data.is_empty()) { return None; }
    let tol_on = std::env::var("REST_K_INC_TOL").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.0e-4);
    let eps = std::env::var("REST_K_INC_EPS").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.0e-10);
    let rebuild_every: u32 = std::env::var("REST_K_INC_REBUILD").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
    let base = c.dm_k.as_ref().unwrap_or(&c.dm);
    let dd = delta_max(&scf.density_matrix, base)?;
    // activate only in the mid-to-late regime: too-large Δ is cheaper to rebuild fully,
    // and near convergence the fresh rebuild of the J decision above wins
    if !(dd < tol_on && dd >= 3.0 * scf.mol.ctrl.scf_acc_rho) { return None; }
    if c.inc_streak >= rebuild_every { return None; }
    // in-core RI tensor as expected by the pair-screened kernel: [stored rows, naux]
    let ri3fn = scf.rimatr.as_ref().map(|(m, _, _)| m)?;
    let map = scf.rimatr_pair_map.as_ref()?;
    let spin_channel = scf.mol.spin_channel;
    let mut out = vec![MatrixUpper::empty(); 2];
    for i_spin in 0..spin_channel {
        // ΔD = D − D_base
        let mut ddm = scf.density_matrix[i_spin].clone();
        ddm.data.par_iter_mut().zip(base[i_spin].data.par_iter())
            .for_each(|(to, from)| *to -= *from);
        // symmetric part only (D is symmetric; guard roundoff asymmetry)
        let n = ddm.size[0];
        let (w_opt, lam, _n) = tensors::matrix_blas_lapack::_dsyevd(&ddm, 'V');
        let w_mat = w_opt?;
        // split columns: C⁺ = W√λ (λ > tiny), C⁻ = W√|λ| (λ < −tiny)
        let tiny = 1.0e-14_f64;
        let mut npos = 0usize; let mut nneg = 0usize;
        for &l in lam.iter() { if l > tiny { npos += 1 } else if l < -tiny { nneg += 1 } }
        let mut cpos = MatrixFull::<f64>::new([n, npos.max(1)], 0.0);
        let mut cneg = MatrixFull::<f64>::new([n, nneg.max(1)], 0.0);
        let mut ip = 0usize; let mut iq = 0usize;
        for (j, &l) in lam.iter().enumerate() {
            let (dst, col) = if l > tiny { ip += 1; (&mut cpos, l.sqrt()) }
                else if l < -tiny { iq += 1; (&mut cneg, (-l).sqrt()) }
                else { continue };
            for r in 0..n { dst.data[r + (if l > tiny { ip - 1 } else { iq - 1 }) * n] = w_mat.data[r + j * n] * col; }
        }
        // absolute screening scale: from the CURRENT occupied orbitals (full density)
        let w_full = reduced_occ_weights(scf, i_spin)?;
        let scale_d = crate::ri_jk::orbital_scale(map, &w_full);
        let mut kspin = c.vk[i_spin].clone();
        // the cached vk already carries the caller's scaling_factor (e.g. -1.0 for
        // UKS exchange); the raw kernel output does NOT. Multiply the increments by
        // the same factor before accumulating, otherwise the sign flips for UKS
        // (which silently steers the SCF to a wrong solution).
        if npos > 0 {
            let wpos = crate::ri_jk::orbital_weights(&cpos);
            let kp = crate::ri_jk::get_vk_pair_screened(ri3fn, map, &cpos, &wpos, eps, scale_d);
            kspin.data.par_iter_mut().zip(kp.data.par_iter())
                .for_each(|(a, b)| *a += b * scaling_factor);
        }
        if nneg > 0 {
            let wneg = crate::ri_jk::orbital_weights(&cneg);
            let kn = crate::ri_jk::get_vk_pair_screened(ri3fn, map, &cneg, &wneg, eps, scale_d);
            kspin.data.par_iter_mut().zip(kn.data.par_iter())
                .for_each(|(a, b)| *a -= b * scaling_factor);
        }
        out[i_spin] = kspin;
    }
    println!("[incremental] K spectral increment (max|ΔD| = {:.2e}, streak {})", dd, c.inc_streak + 1);
    Some(out)
}

/// Per-AO weights of the current occupied orbitals (occ × sqrt(occ)-scaled columns),
/// mirroring the main K path's screening weights.
fn reduced_occ_weights(scf: &SCF, i_spin: usize) -> Option<Vec<f64>> {
    let num_basis = scf.eigenvectors[0].size[0];
    let eigv_s = if !scf.eigenvectors[i_spin].data.is_empty() { &scf.eigenvectors[i_spin] } else { &scf.eigenvectors[0] };
    let occ = &scf.occupation[i_spin];
    let nw = occ.iter().filter(|&&o| o >= 1.0e-6).count();
    if nw == 0 { return None; }
    let mut tmp = MatrixFull::<f64>::new([num_basis, nw], 0.0);
    for j in 0..nw {
        let s = occ[j].sqrt();
        for r in 0..num_basis {
            tmp.data[r + j * num_basis] = eigv_s.data[r + j * num_basis] * s;
        }
    }
    Some(crate::ri_jk::orbital_weights(&tmp))
}


/// Whether the cached Fock state is not fresh (an increment or a reuse was the last
/// update). Callers should demand a fresh full rebuild before accepting convergence.
pub fn fock_is_stale() -> bool {
    cache().lock().unwrap().as_ref().map_or(false, |c| c.inc_streak > 0 || c.reuse_now)
}

/// Request a full J and K rebuild on the next evaluation (the "fresh-Fock validation
/// pass"): the SCF loop calls this when the convergence criteria pass but the cached
/// Fock lags the density, and simply runs one more iteration.
pub fn request_full_rebuild() {
    if let Some(c) = cache().lock().unwrap().as_mut() {
        c.force_full = true;
    }
}
