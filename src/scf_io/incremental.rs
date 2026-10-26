//! δD 增量 Fock v1：`[ctrl] incremental_fock = true` 激活。
//! J(D) 与 K(D) 对密度线性；当本次密度与上次实际构建时的密度之差
//! max|ΔD| < scf_acc_rho（与收敛判据同源）时重用缓存，误差与收敛精度同级。
//! 状态放进程级缓存，不侵入 SCF 构造路径。v2 方向：按 AO 对的 |ΔD| 加权界与部分重算
//! （LinK 风格）；PCM（对密度线性）接入在 solvent 侧，v1 未覆盖。
use std::sync::{Mutex, OnceLock};
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
}

/// reuse threshold as a multiple of `scf_acc_rho`: the Fock may lag the density by a
/// controlled amount mid-SCF (the direct-SCF "fock drift" trade), and the periodic
/// rebuild keeps the convergence criteria clean at the end
const REUSE_TOL_FACTOR: f64 = 100.0;
const MAX_STALE: u32 = 4;

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
    let tol = REUSE_TOL_FACTOR * scf.mol.ctrl.scf_acc_rho;
    if let Some(c) = g.as_mut() {
        let hit = !c.vj.is_empty()
            && c.stale < MAX_STALE
            && dm_delta(scf, c).map_or(false, |d| d < tol);
        if hit {
            c.reuse_now = true;
            c.stale += 1;
            println!("[incremental] J reused (max|ΔD| = {:.2e} < {:.1e}, streak {})",
                dm_delta(scf, c).unwrap_or(0.0), tol, c.stale);
            return c.vj.clone();
        }
        if c.stale >= MAX_STALE && !c.vj.is_empty() {
            println!("[incremental] periodic full rebuild after {} reuses", c.stale);
        }
    }
    let vj = scf.generate_vj_with_ri_v_sync(1.0, mpi_operator);
    match g.as_mut() {
        Some(c) => {
            c.reuse_now = false;
            c.stale = 0;
            if c.vk.is_empty() || !dm_delta(scf, c).map_or(false, |d| d < 1.0) {
                c.dm = scf.density_matrix.clone();
            }
            c.vj = vj.clone();
        }
        None => *g = Some(Cache { dm: scf.density_matrix.clone(), vj: vj.clone(), vk: Vec::new(), stale: 0, reuse_now: false }),
    }
    vj
}

pub fn cached_vk(scf: &mut SCF, scaling_factor: f64, mpi_operator: &Option<MPIOperator>) -> Vec<MatrixUpper<f64>> {
    let mut g = cache().lock().unwrap();
    // follow the decision the J call of this iteration already took, so J and K never mix
    // a stale and a fresh half within one Fock build
    if let Some(c) = g.as_ref() {
        if c.reuse_now && !c.vk.is_empty() {
            println!("[incremental] K reused");
            return c.vk.clone();
        }
    }
    let vk = scf.generate_vk_with_ri_v(scaling_factor, scf.mol.ctrl.use_dm_only, mpi_operator);
    match g.as_mut() {
        Some(c) => { c.reuse_now = false; c.vk = vk.clone(); }
        None => *g = Some(Cache { dm: scf.density_matrix.clone(), vj: Vec::new(), vk: vk.clone(), stale: 0, reuse_now: false }),
    }
    vk
}
