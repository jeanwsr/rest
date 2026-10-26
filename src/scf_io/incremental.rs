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

fn dm_matches(scf: &SCF, c: &Cache) -> bool {
    delta_max(&scf.density_matrix, &c.dm).map_or(false, |d| d < scf.mol.ctrl.scf_acc_rho)
}

pub fn cached_vj(scf: &mut SCF, mpi_operator: &Option<MPIOperator>) -> Vec<MatrixUpper<f64>> {
    let mut g = cache().lock().unwrap();
    if let Some(c) = g.as_ref() {
        if !c.vj.is_empty() && dm_matches(scf, c) {
            println!("[incremental] J reused (max|ΔD| < {:.1e})", scf.mol.ctrl.scf_acc_rho);
            return c.vj.clone();
        }
    }
    let vj = scf.generate_vj_with_ri_v_sync(1.0, mpi_operator);
    match g.as_mut() {
        Some(c) if dm_matches(scf, c) => c.vj = vj.clone(),
        _ => *g = Some(Cache { dm: scf.density_matrix.clone(), vj: vj.clone(), vk: Vec::new() }),
    }
    vj
}

pub fn cached_vk(scf: &mut SCF, scaling_factor: f64, mpi_operator: &Option<MPIOperator>) -> Vec<MatrixUpper<f64>> {
    let mut g = cache().lock().unwrap();
    if let Some(c) = g.as_ref() {
        if !c.vk.is_empty() && dm_matches(scf, c) {
            println!("[incremental] K reused (max|ΔD| < {:.1e})", scf.mol.ctrl.scf_acc_rho);
            return c.vk.clone();
        }
    }
    let vk = scf.generate_vk_with_ri_v(scaling_factor, scf.mol.ctrl.use_dm_only, mpi_operator);
    match g.as_mut() {
        Some(c) if dm_matches(scf, c) => c.vk = vk.clone(),
        _ => *g = Some(Cache { dm: scf.density_matrix.clone(), vj: Vec::new(), vk: vk.clone() }),
    }
    vk
}
