use crate::basis_io::etb::{etb_gen_for_atom_list, get_etb_elem};
use crate::basis_io::Basis4Elem;
use crate::constants::{AUXBAS_THRESHOLD, BOHR, SQRT_THRESHOLD};
use crate::geom_io::GeomCell;
use crate::molecule_io::Molecule;
use crate::scf_io::{
    vj_upper_with_rimatr_sync_v02, vk_upper_with_rimatr_sync_v03,
    vk_upper_with_rimatr_use_dm_only_sync_v02,
};
use crate::utilities::memory_batch::detect_available_memory_mb;
use rayon::prelude::*;
use rest_tensors::matrix_blas_lapack::{
    _dsolve, _dsyev, _dsyrk, omp_get_num_threads_wrapper, omp_set_num_threads_wrapper,
};
use rest_tensors::{MatrixFull, MatrixFullSlice, MatrixUpper, RIFull, TensorOpt, TensorOptMut};
use libm::erf;
use std::array;
use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::thread;
use std::time::Instant;
mod basis;
use crate::lib_rint::basis::{
    load_aux_molecule_shell_shared_from_raw, load_aux_rint_shells_from_raw,
    load_cartesian_rhf_basis_and_density_shell_shared,
    load_cartesian_rhf_rint_shells_basis_and_density_shell_shared,
    load_molecule_rint_shells_from_raw, load_molecule_shell_shared_from_raw,
    transform_density_to_cartesian_shell_shared, transform_mo_coeff_to_cartesian_shell_shared,
    transform_operator_from_cartesian_shell_shared,
};
pub use basis::{make_contracted_coeffs_for_shell, BasisFunction, RintContractedShell, RintShell};
mod find_polyroots;
use find_polyroots::find_polyroots;
mod rys_coeffs_provider;
mod rys_recursion;
use rys_coeffs_provider::RysCoeffProvider;
use rys_recursion::{
    build_1d_panel_for_t, transfer_i_to_j, transfer_k_to_l, CoeffProvider, PanelSpec,
};
// MP2 numerator
mod legacy_var;
pub use legacy_var::{
    var_vee_doubles_hf_r, var_vee_doubles_hf_r_geom, var_vee_hf_r_ri3mo, var_vee_hf_r_ri3mo_geom,
};

#[derive(Clone, Copy, Debug)]
pub struct RhfVeeObservables {
    pub ej: f64,
    pub ek: f64,
    pub total: f64,
}
#[derive(Clone, Debug)]
pub struct UhfVeeJkMatrices {
    pub j_total: MatrixFull<f64>,
    pub k_spin: Vec<MatrixFull<f64>>,
}
#[derive(Clone, Debug)]
pub struct RhfVeeJkIntermediate {
    pub observables: RhfVeeObservables,
    pub j: MatrixFull<f64>,
    pub k: MatrixFull<f64>,
}
#[derive(Clone, Debug)]
pub struct UhfVeeJkIntermediate {
    pub observables: RhfVeeObservables,
    pub jk: UhfVeeJkMatrices,
}
#[derive(Clone, Copy, Debug)]
pub struct RhfJkObservables {
    pub r: RhfVeeObservables,
    pub r2: RhfVeeObservables,
}
const DEFAULT_R2_ETB_BETA: f64 = 1.7;
const LIBCINT_DEFAULT_EXPCUTOFF: f64 = 60.0;
const R2_DIRECT_MEMORY_FACTOR: f64 = 0.70;
const R2_SEMIDIRECT_REGRESSION_TOL: f64 = 1.0e-10;
const R2_SEMIDIRECT_MEDIUM_REGRESSION_TOL: f64 = 1.0e-8;
const TWO_PI_POW_2P5: f64 = 34.986_836_655_249_725_f64;

impl RhfJkObservables {
    pub fn print(&self) {
        print_rhf_vee_observables("r", &self.r);
        print_rhf_vee_observables("r2", &self.r2);
    }
}
pub fn print_rhf_vee_observables(label: &str, observables: &RhfVeeObservables) {
    match label {
        "r" => {
            println!("E_J(r)         = {:.16e}", observables.ej);
            println!("E_K(r)         = {:.16e}", observables.ek);
            println!("<1/r_12>       = {:.16e}", observables.total);
        }
        "r2" => {
            println!("E_J(r2)        = {:.16e}", observables.ej);
            println!("E_K(r2)        = {:.16e}", observables.ek);
            println!("<1/r_12^2>     = {:.16e}", observables.total);
        }
        _ => {
            println!("E_J({label})         = {:.16e}", observables.ej);
            println!("E_K({label})         = {:.16e}", observables.ek);
            println!("E({label})           = {:.16e}", observables.total);
        }
    }
}
pub fn load_saved_rhf_r_observables_from_energies(
    energies: &HashMap<String, Vec<f64>>,
) -> Option<RhfVeeObservables> {
    let ej = energies.get("lib_rint_vee_r_ej")?.first().copied()?;
    let ek = energies.get("lib_rint_vee_r_ek")?.first().copied()?;
    let total = energies
        .get("lib_rint_vee_r_total")
        .and_then(|values| values.first().copied())
        .unwrap_or(ej + ek);
    Some(RhfVeeObservables { ej, ek, total })
}
pub fn load_saved_rhf_vee_r_from_energies(
    energies: &HashMap<String, Vec<f64>>,
) -> Option<RhfVeeObservables> {
    load_saved_rhf_r_observables_from_energies(energies)
}
pub fn load_saved_rhf_r_observables_from_chkfile(chkfile: &str) -> Option<RhfVeeObservables> {
    let file = hdf5::File::open(chkfile).ok()?;
    let scf = file.group("scf").ok()?;
    let ej = scf
        .dataset("lib_rint_vee_r_ej")
        .ok()?
        .read_raw::<f64>()
        .ok()?
        .first()
        .copied()?;
    let ek = scf
        .dataset("lib_rint_vee_r_ek")
        .ok()?
        .read_raw::<f64>()
        .ok()?
        .first()
        .copied()?;
    let total = scf
        .dataset("lib_rint_vee_r_total")
        .ok()
        .and_then(|dataset| dataset.read_raw::<f64>().ok())
        .and_then(|values| values.first().copied())
        .unwrap_or(ej + ek);
    Some(RhfVeeObservables { ej, ek, total })
}
pub fn load_saved_rhf_vee_r_from_chkfile(chkfile: &str) -> Option<RhfVeeObservables> {
    load_saved_rhf_r_observables_from_chkfile(chkfile)
}

fn build_default_r2_etb_auxbasis(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
) -> Option<Vec<Basis4Elem>> {
    let mut mol = Molecule::init_mol();
    mol.geom = geom.clone();
    mol.basis4elem = basis4elem.to_vec();
    let etb_elem = get_etb_elem(&mol.geom, &1_usize);
    let etb = etb_gen_for_atom_list(&mol, &DEFAULT_R2_ETB_BETA, &etb_elem);
    if etb.elements.is_empty() {
        return None;
    }
    geom.elem
        .iter()
        .enumerate()
        .map(|(atom_idx, elem)| {
            etb.elements.get(elem).map(|basis| {
                let mut basis = basis.clone();
                basis.global_index = (atom_idx, atom_idx);
                basis
            })
        })
        .collect()
}
// ============================================================
// Boys function F_m(T) for 1/r kernel, f64 implementation
// ============================================================
pub fn boys_vec(mmax: usize, t: f64) -> Vec<f64> {
    let mut f = vec![0.0_f64; mmax + 1];
    // Very small T: use the limit F_m(0) = 1 / (2m + 1).
    if t <= 1e-12 {
        for m in 0..=mmax {
            f[m] = 1.0 / (2.0 * (m as f64) + 1.0);
        }
        return f;
    }
    // Small T: power series F_m(T) = sum_k (-T)^k / (k! (2m + 2k + 1)).
    if t <= 5.0 {
        for m in 0..=mmax {
            let mut sum = 0.0_f64;
            let mut term = 1.0 / (2.0 * (m as f64) + 1.0);
            sum += term;
            let mut k: u32 = 0;
            loop {
                k += 1;
                let ratio = -(t / (k as f64));
                let prev_d = 2.0 * (m as f64) + 2.0 * ((k - 1) as f64) + 1.0;
                let next_d = 2.0 * (m as f64) + 2.0 * (k as f64) + 1.0;
                let extra = prev_d / next_d;
                term *= ratio * extra;
                sum += term;
                if term.abs() < 1e-30 || k > 100 {
                    break;
                }
            }
            f[m] = if sum.is_finite() { sum } else { 0.0 };
        }
        if t > 1.0e-6 {
            let sqrt_pi = std::f64::consts::PI.sqrt();
            let sqrt_t = t.sqrt();
            f[0] = 0.5 * sqrt_pi * erf(sqrt_t) / sqrt_t;
            if mmax >= 1 {
                let f1 = (f[0] - (-t).exp()) / (2.0 * t);
                f[1] = if f1.is_finite() { f1 } else { 0.0 };
            }
        }
        return f;
    }
    // Large T: asymptotic start + downward recursion + F0 rescaling.
    if t >= 50.0 {
        let large_thresh = 200.0_f64;
        let sqrt_pi = std::f64::consts::PI.sqrt();
        if t >= large_thresh {
            for m in 0..=mmax {
                let mut df = 1.0_f64;
                for j in 1..=m {
                    df *= (2 * j - 1) as f64;
                }
                let t_pow = t.powi(m as i32) * t.sqrt();
                let denom = 2.0_f64.powi(m as i32 + 1);
                let val = sqrt_pi * df / (denom * t_pow);
                f[m] = if val.is_finite() { val } else { 0.0 };
            }
            return f;
        }
        let extra = ((t / 100.0).min(30.0)).round() as usize;
        let mstart = mmax + extra;
        let mut g = vec![0.0_f64; mstart + 1];
        let mut df = 1.0_f64;
        for j in 1..=mstart {
            df *= (2 * j - 1) as f64;
        }
        let t_pow = t.powi(mstart as i32) * t.sqrt();
        let denom = 2.0_f64.powi(mstart as i32 + 1);
        g[mstart] = sqrt_pi * df / (denom * t_pow);
        let exp_minus_t = (-t).exp();
        for mm in (0..mstart).rev() {
            g[mm] = (2.0 * t * g[mm + 1] + exp_minus_t) / (2.0 * (mm as f64) + 1.0);
        }
        let sqrt_t = t.sqrt();
        let f0_exact = 0.5 * sqrt_pi * erf(sqrt_t) / sqrt_t;
        let scale = f0_exact / g[0];
        for m in 0..=mmax {
            let val = g[m] * scale;
            f[m] = if val.is_finite() { val } else { 0.0 };
        }
        return f;
    }
    // Moderate T: exact F0 + scaled upward recursion.
    let sqrt_pi = std::f64::consts::PI.sqrt();
    let sqrt_t = t.sqrt();
    let f0 = 0.5 * sqrt_pi * erf(sqrt_t) / sqrt_t;
    if mmax == 0 {
        f[0] = if f0.is_finite() { f0 } else { 0.0 };
        return f;
    }
    let e_t = t.exp();
    let e_minus_t = (-t).exp();
    let mut e = vec![0.0_f64; mmax + 1];
    e[0] = f0 * e_t;
    for m in 0..mmax {
        let num = (2.0 * (m as f64) + 1.0) * e[m] - 1.0;
        e[m + 1] = num / (2.0 * t);
    }
    for m in 0..=mmax {
        let val = e[m] * e_minus_t;
        f[m] = if val.is_finite() { val } else { 0.0 };
    }
    f
}

fn boys_slice(mmax: usize, t: f64, f: &mut [f64]) {
    debug_assert!(f.len() > mmax);
    f[..=mmax].fill(0.0);
    if t <= 1e-12 {
        for m in 0..=mmax {
            f[m] = 1.0 / (2.0 * (m as f64) + 1.0);
        }
        return;
    }
    if t <= 5.0 {
        for m in 0..=mmax {
            let mut sum = 0.0_f64;
            let mut term = 1.0 / (2.0 * (m as f64) + 1.0);
            sum += term;
            let mut k: u32 = 0;
            loop {
                k += 1;
                let ratio = -(t / (k as f64));
                let prev_d = 2.0 * (m as f64) + 2.0 * ((k - 1) as f64) + 1.0;
                let next_d = 2.0 * (m as f64) + 2.0 * (k as f64) + 1.0;
                term *= ratio * (prev_d / next_d);
                sum += term;
                if term.abs() < 1e-30 || k > 100 {
                    break;
                }
            }
            f[m] = if sum.is_finite() { sum } else { 0.0 };
        }
        if t > 1.0e-6 {
            let sqrt_pi = std::f64::consts::PI.sqrt();
            let sqrt_t = t.sqrt();
            f[0] = 0.5 * sqrt_pi * erf(sqrt_t) / sqrt_t;
            if mmax >= 1 {
                let f1 = (f[0] - (-t).exp()) / (2.0 * t);
                f[1] = if f1.is_finite() { f1 } else { 0.0 };
            }
        }
        return;
    }
    if t >= 50.0 {
        let large_thresh = 200.0_f64;
        let sqrt_pi = std::f64::consts::PI.sqrt();
        if t >= large_thresh {
            for m in 0..=mmax {
                let mut df = 1.0_f64;
                for j in 1..=m {
                    df *= (2 * j - 1) as f64;
                }
                let t_pow = t.powi(m as i32) * t.sqrt();
                let denom = 2.0_f64.powi(m as i32 + 1);
                let val = sqrt_pi * df / (denom * t_pow);
                f[m] = if val.is_finite() { val } else { 0.0 };
            }
            return;
        }
        let extra = ((t / 100.0).min(30.0)).round() as usize;
        let mstart = mmax + extra;
        if mstart >= 64 {
            let tmp = boys_vec(mmax, t);
            f[..=mmax].copy_from_slice(&tmp);
            return;
        }
        let mut g = [0.0_f64; 64];
        let mut df = 1.0_f64;
        for j in 1..=mstart {
            df *= (2 * j - 1) as f64;
        }
        let t_pow = t.powi(mstart as i32) * t.sqrt();
        let denom = 2.0_f64.powi(mstart as i32 + 1);
        g[mstart] = sqrt_pi * df / (denom * t_pow);
        let exp_minus_t = (-t).exp();
        for mm in (0..mstart).rev() {
            g[mm] = (2.0 * t * g[mm + 1] + exp_minus_t) / (2.0 * (mm as f64) + 1.0);
        }
        let sqrt_t = t.sqrt();
        let f0_exact = 0.5 * sqrt_pi * erf(sqrt_t) / sqrt_t;
        let scale = f0_exact / g[0];
        for m in 0..=mmax {
            let val = g[m] * scale;
            f[m] = if val.is_finite() { val } else { 0.0 };
        }
        return;
    }

    let sqrt_pi = std::f64::consts::PI.sqrt();
    let sqrt_t = t.sqrt();
    let f0 = 0.5 * sqrt_pi * erf(sqrt_t) / sqrt_t;
    if mmax == 0 {
        f[0] = if f0.is_finite() { f0 } else { 0.0 };
        return;
    }
    let e_t = t.exp();
    let e_minus_t = (-t).exp();
    let mut e = [0.0_f64; 64];
    if mmax >= e.len() {
        let tmp = boys_vec(mmax, t);
        f[..=mmax].copy_from_slice(&tmp);
        return;
    }
    e[0] = f0 * e_t;
    for m in 0..mmax {
        e[m + 1] = ((2.0 * (m as f64) + 1.0) * e[m] - 1.0) / (2.0 * t);
    }
    for m in 0..=mmax {
        let val = e[m] * e_minus_t;
        f[m] = if val.is_finite() { val } else { 0.0 };
    }
}

#[inline(always)]
fn boys_f0(t: f64) -> f64 {
    if t <= 1.0e-12 {
        return 1.0;
    }
    if t <= 1.0e-6 {
        let mut sum = 1.0_f64;
        let mut term = 1.0_f64;
        let mut k = 0_u32;
        loop {
            k += 1;
            let prev_d = 2.0 * ((k - 1) as f64) + 1.0;
            let next_d = 2.0 * (k as f64) + 1.0;
            term *= -(t / (k as f64)) * (prev_d / next_d);
            sum += term;
            if term.abs() < 1.0e-30 || k > 100 {
                break;
            }
        }
        return if sum.is_finite() { sum } else { 0.0 };
    }
    let sqrt_t = t.sqrt();
    let value = 0.5 * std::f64::consts::PI.sqrt() * erf(sqrt_t) / sqrt_t;
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

#[inline(always)]
fn boys_f0_f1(t: f64) -> (f64, f64) {
    if t <= 1.0e-12 {
        return (1.0, 1.0 / 3.0);
    }
    if t <= 1.0e-6 {
        let mut f0 = 0.0_f64;
        let mut f1 = 0.0_f64;
        for m in 0..=1 {
            let mut sum = 0.0_f64;
            let mut term = 1.0 / (2.0 * (m as f64) + 1.0);
            sum += term;
            let mut k = 0_u32;
            loop {
                k += 1;
                let prev_d = 2.0 * (m as f64) + 2.0 * ((k - 1) as f64) + 1.0;
                let next_d = 2.0 * (m as f64) + 2.0 * (k as f64) + 1.0;
                term *= -(t / (k as f64)) * (prev_d / next_d);
                sum += term;
                if term.abs() < 1.0e-30 || k > 100 {
                    break;
                }
            }
            if m == 0 {
                f0 = if sum.is_finite() { sum } else { 0.0 };
            } else {
                f1 = if sum.is_finite() { sum } else { 0.0 };
            }
        }
        return (f0, f1);
    }
    if t <= 1.0 {
        let f0 = boys_f0(t);
        let f1 = (f0 - (-t).exp()) / (2.0 * t);
        return (f0, if f1.is_finite() { f1 } else { 0.0 });
    }
    let f0 = boys_f0(t);
    let f1 = (f0 - (-t).exp()) / (2.0 * t);
    (f0, if f1.is_finite() { f1 } else { 0.0 })
}
// ============================================================
// Beta / G_m(T) for 1/r^2 kernel in your code, f64 version
// ============================================================
/// Beta(m+1, 1/2), m >= 0
fn beta(m: usize) -> f64 {
    // B(1, 1/2) = 2
    let mut b = 2.0_f64;
    if m == 0 {
        return b;
    }
    // B(a+1,1/2) = a / (a+1/2) * B(a,1/2)
    for a in 1..=m {
        let a_f = a as f64;
        let denom = a_f + 0.5;
        b *= a_f / denom;
    }
    b
}
fn gm_r2_single(m: usize, t: f64) -> f64 {
    let t_abs = t.abs();
    // G_m(0) = 1/2 * B(m+1,1/2)
    if t_abs <= 1e-8 {
        return 0.5 * beta(m);
    }
    let t_split = 10.0_f64;
    if t_abs <= t_split {
        // Beta + (-T)^k/k! expansion
        let mut sum = 0.0_f64;
        let mut beta_val = beta(m);
        let mut pow = 1.0_f64; // (-T)^0
        let mut fact = 1.0_f64; // 0!
        let half = 0.5_f64;
        let max_k = 120usize;
        let tol = 1e-20_f64;
        for k in 0..=max_k {
            let term = half * beta_val * pow / fact;
            sum += term;
            if term.abs() < tol * sum.abs().max(1e-10) {
                break;
            }
            // (-T)^{k+1}
            pow *= -t;
            // (k+1)!
            let k1 = (k + 1) as f64;
            fact *= k1;
            // Update beta: B(m+k+2,1/2) from B(m+k+1,1/2)
            // B(m+k+2,1/2) = (m+k+1)/(m+k+3/2) * B(m+k+1,1/2)
            let mk1 = m + k + 1;
            let num = mk1 as f64;
            let denom = (2.0 * ((m + k) as f64) + 3.0) / 2.0;
            beta_val *= num / denom;
        }
        if sum.is_finite() {
            sum
        } else {
            0.0
        }
    } else {
        // Asymptotic expansion in 1/T
        // G_m(T) ~ sum_n c_n * 1/2 * (m+n)! / T^(m+n+1)
        let half = 0.5_f64;
        // (m)! = Gamma(m+1)
        let mut gamma_val = 1.0_f64;
        for k in 1..=m {
            gamma_val *= k as f64;
        }
        // T^{m+1}
        let mut tpow = t_abs.powi((m as i32) + 1);
        // c_0
        let mut c = 1.0_f64;
        let max_n = 8usize;
        let tol = 1e-20_f64;
        let mut sum = 0.0_f64;
        for n in 0..=max_n {
            let term = half * c * gamma_val / tpow;
            sum += term;
            if term.abs() < tol * sum.abs().max(1e-10) {
                break;
            }
            // c_{n+1} = c_n * (2n+1)/(2n+2)
            let two_n = 2.0 * (n as f64);
            c *= (two_n + 1.0) / (two_n + 2.0);
            // (m+n+1)! update
            let mn1 = m + n + 1;
            gamma_val *= mn1 as f64;
            // T^{m+n+2}
            tpow *= t_abs;
        }
        if sum.is_finite() {
            sum
        } else {
            0.0
        }
    }
}
pub fn boys_vec_r2(mmax: usize, t: f64) -> Vec<f64> {
    let mut g = vec![0.0_f64; mmax + 1];
    for m in 0..=mmax {
        g[m] = gm_r2_single(m, t);
    }
    g
}
// ============================================================
// Helpers for Rys roots/weights
// ============================================================
#[inline]
fn poly_value1(a: &[f64], order: usize, x: f64) -> f64 {
    let mut p = a[order];
    for i in 1..=order {
        p = p * x + a[order - i];
    }
    if p.is_finite() {
        p
    } else {
        0.0
    }
}
#[inline]
fn zeros_1d(size: usize) -> Vec<f64> {
    vec![0.0_f64; size]
}
#[inline]
fn zeros_2d(rows: usize, cols: usize) -> MatrixFull<f64> {
    MatrixFull::new([rows, cols], 0.0_f64)
}
/// Compute the cs matrix via modified Gram-Schmidt (your R_dsmit), f64 version.
pub fn R_dsmit(s: &MatrixFull<f64>, n: usize) -> Result<MatrixFull<f64>, String> {
    let mut cs = zeros_2d(n, n);
    for j in 0..n {
        let mut fac = s[(j, j)];
        let mut v = zeros_1d(j);
        for k in 0..j {
            // dot = sum_{i=0..j} cs[k,i] * s[j,i]
            let mut dot = 0.0_f64;
            for i in 0..=j {
                dot += cs[(k, i)] * s[(j, i)];
            }
            for i in 0..j {
                v[i] -= dot * cs[(k, i)];
            }
            fac -= dot * dot;
        }
        if fac < 0.0 || !fac.is_finite() {
            return Err(format!("R_dsmit fac < 0 or non-finite, fac = {}", fac));
        }
        let fac_sqrt = fac.sqrt();
        if fac_sqrt == 0.0 {
            return Err(format!("R_dsmit fac_sqrt is zero, fac = {}", fac));
        }
        let inv = 1.0 / fac_sqrt;
        cs[(j, j)] = inv;
        for i in 0..j {
            cs[(j, i)] = inv * v[i];
        }
    }
    Ok(cs)
}

fn solve_3x3(mut matrix: [[f64; 3]; 3], mut rhs: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let mut pivot = col;
        let mut pivot_abs = matrix[col][col].abs();
        for row in (col + 1)..3 {
            let value_abs = matrix[row][col].abs();
            if value_abs > pivot_abs {
                pivot = row;
                pivot_abs = value_abs;
            }
        }
        if pivot_abs <= 1.0e-24 || !pivot_abs.is_finite() {
            return None;
        }
        if pivot != col {
            matrix.swap(col, pivot);
            rhs.swap(col, pivot);
        }
        let pivot_value = matrix[col][col];
        for row in (col + 1)..3 {
            let factor = matrix[row][col] / pivot_value;
            matrix[row][col] = 0.0;
            for k in (col + 1)..3 {
                matrix[row][k] -= factor * matrix[col][k];
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    let mut x = [0.0_f64; 3];
    for row in (0..3).rev() {
        let mut value = rhs[row];
        for col in (row + 1)..3 {
            value -= matrix[row][col] * x[col];
        }
        let diag = matrix[row][row];
        if diag.abs() <= 1.0e-24 || !diag.is_finite() {
            return None;
        }
        x[row] = value / diag;
        if !x[row].is_finite() {
            return None;
        }
    }
    Some(x)
}

fn solve_linear_system(mut matrix: Vec<f64>, mut rhs: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    debug_assert_eq!(matrix.len(), n * n);
    debug_assert_eq!(rhs.len(), n);
    let mut col_major = vec![0.0_f64; n * n];
    for row in 0..n {
        for col in 0..n {
            col_major[row + n * col] = matrix[row * n + col];
        }
    }
    if let Some(solution) =
        MatrixFull::from_vec([n, n], col_major).and_then(|mat| _dsolve(&mat, &rhs))
    {
        return Some(solution);
    }

    let idx = |row: usize, col: usize| -> usize { row * n + col };

    for col in 0..n {
        let mut pivot = col;
        let mut pivot_abs = matrix[idx(col, col)].abs();
        for row in (col + 1)..n {
            let value_abs = matrix[idx(row, col)].abs();
            if value_abs > pivot_abs {
                pivot = row;
                pivot_abs = value_abs;
            }
        }
        if pivot_abs <= 1.0e-24 || !pivot_abs.is_finite() {
            return None;
        }
        if pivot != col {
            for k in col..n {
                let lhs = idx(col, k);
                let rhs_idx = idx(pivot, k);
                matrix.swap(lhs, rhs_idx);
            }
            rhs.swap(col, pivot);
        }
        let pivot_value = matrix[idx(col, col)];
        for row in (col + 1)..n {
            let factor = matrix[idx(row, col)] / pivot_value;
            matrix[idx(row, col)] = 0.0;
            for k in (col + 1)..n {
                let row_k = idx(row, k);
                matrix[row_k] -= factor * matrix[idx(col, k)];
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    let mut x = vec![0.0_f64; n];
    for row in (0..n).rev() {
        let mut value = rhs[row];
        for col in (row + 1)..n {
            value -= matrix[idx(row, col)] * x[col];
        }
        let diag = matrix[idx(row, row)];
        if diag.abs() <= 1.0e-24 || !diag.is_finite() {
            return None;
        }
        x[row] = value / diag;
        if !x[row].is_finite() {
            return None;
        }
    }
    Some(x)
}

fn cubic_three_real_roots(a: f64, b: f64, c: f64) -> Option<[f64; 3]> {
    let third = 1.0 / 3.0;
    let p = b - a * a * third;
    let q = (2.0 * a * a * a) / 27.0 - (a * b) * third + c;
    let discriminant = 0.25 * q * q + (p * third).powi(3);
    if !p.is_finite() || !q.is_finite() || !discriminant.is_finite() {
        return None;
    }
    if discriminant > 1.0e-22 || p >= 0.0 {
        return None;
    }
    let cos_arg = ((3.0 * q) / (2.0 * p) * (-3.0 / p).sqrt()).clamp(-1.0, 1.0);
    let theta = cos_arg.acos() * third;
    let scale = 2.0 * (-p * third).sqrt();
    let shift = -a * third;
    let mut roots = [
        shift + scale * theta.cos(),
        shift + scale * (theta - 2.0 * std::f64::consts::PI * third).cos(),
        shift + scale * (theta - 4.0 * std::f64::consts::PI * third).cos(),
    ];
    if roots.iter().all(|root| root.is_finite()) {
        roots.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
        Some(roots)
    } else {
        None
    }
}

fn rys_roots_weights_r3_from_moments(ff: &[f64]) -> Option<([f64; 3], [f64; 3])> {
    if ff.len() < 6 {
        return None;
    }
    let [a, b, c] = solve_3x3(
        [
            [ff[2], ff[1], ff[0]],
            [ff[3], ff[2], ff[1]],
            [ff[4], ff[3], ff[2]],
        ],
        [-ff[3], -ff[4], -ff[5]],
    )?;
    let roots = cubic_three_real_roots(a, b, c)?;
    let mut weights = [0.0_f64; 3];
    for i in 0..3 {
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        let denom = (roots[i] - roots[j]) * (roots[i] - roots[k]);
        if denom.abs() <= 1.0e-24 || !denom.is_finite() {
            return None;
        }
        weights[i] = (ff[2] - ff[1] * (roots[j] + roots[k]) + ff[0] * roots[j] * roots[k]) / denom;
        if !weights[i].is_finite() {
            return None;
        }
    }
    Some((roots, weights))
}

fn poly_eval_ascending(coeffs: &[f64], x: f64) -> f64 {
    coeffs
        .iter()
        .rev()
        .fold(0.0_f64, |acc, coeff| acc * x + coeff)
}

fn polynomial_derivative_ascending(coeffs: &[f64]) -> Vec<f64> {
    if coeffs.len() <= 1 {
        return Vec::new();
    }
    coeffs
        .iter()
        .enumerate()
        .skip(1)
        .map(|(power, coeff)| *coeff * power as f64)
        .collect()
}

fn bisect_polynomial_root(coeffs: &[f64], mut left: f64, mut right: f64) -> Option<f64> {
    let mut f_left = poly_eval_ascending(coeffs, left);
    let f_right = poly_eval_ascending(coeffs, right);
    if !f_left.is_finite() || !f_right.is_finite() {
        return None;
    }
    if f_left.abs() <= 1.0e-14 {
        return Some(left);
    }
    if f_right.abs() <= 1.0e-14 {
        return Some(right);
    }
    if f_left * f_right > 0.0 {
        return None;
    }
    for _ in 0..100 {
        let mid = 0.5 * (left + right);
        let f_mid = poly_eval_ascending(coeffs, mid);
        if !f_mid.is_finite() {
            return None;
        }
        if f_mid.abs() <= 1.0e-15 || (right - left).abs() <= 1.0e-15 {
            return Some(mid);
        }
        if f_left * f_mid <= 0.0 {
            right = mid;
        } else {
            left = mid;
            f_left = f_mid;
        }
    }
    Some(0.5 * (left + right))
}

fn polynomial_roots_unit_interval(coeffs: &[f64]) -> Vec<f64> {
    let degree = coeffs.len().saturating_sub(1);
    if degree == 0 {
        return Vec::new();
    }
    if degree == 1 {
        let denom = coeffs[1];
        if denom.abs() <= 1.0e-24 {
            return Vec::new();
        }
        let root = -coeffs[0] / denom;
        return if root > -1.0e-12 && root < 1.0 + 1.0e-12 && root.is_finite() {
            vec![root.clamp(0.0, 1.0)]
        } else {
            Vec::new()
        };
    }

    let derivative = polynomial_derivative_ascending(coeffs);
    let mut points = vec![0.0_f64];
    points.extend(polynomial_roots_unit_interval(&derivative));
    points.push(1.0);
    points.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    points.dedup_by(|left, right| (*left - *right).abs() <= 1.0e-13);

    let mut roots: Vec<f64> = Vec::new();
    for window in points.windows(2) {
        let left = window[0];
        let right = window[1];
        if right - left <= 1.0e-14 {
            continue;
        }
        if let Some(root) = bisect_polynomial_root(coeffs, left, right) {
            if root > -1.0e-12
                && root < 1.0 + 1.0e-12
                && !roots.iter().any(|prev| (*prev - root).abs() <= 1.0e-10)
            {
                roots.push(root.clamp(0.0, 1.0));
            }
        }
    }
    roots.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    roots
}

fn rys_roots_weights_from_moments_golub_welsch(
    nroots: usize,
    moments: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    if nroots == 0 || moments.len() < 2 * nroots || moments[0] <= 0.0 || !moments[0].is_finite() {
        return None;
    }

    let mut s = zeros_2d(nroots, nroots);
    for row in 0..nroots {
        for col in 0..nroots {
            s[(row, col)] = moments[row + col];
        }
    }
    let cs = R_dsmit(&s, nroots).ok()?;

    let mut jacobi = vec![0.0_f64; nroots * nroots];
    let idx = |row: usize, col: usize| -> usize { row + nroots * col };
    for j in 0..nroots {
        let mut alpha = 0.0_f64;
        for a in 0..=j {
            for b in 0..=j {
                alpha += cs[(j, a)] * cs[(j, b)] * moments[a + b + 1];
            }
        }
        if !alpha.is_finite() {
            return None;
        }
        jacobi[idx(j, j)] = alpha;

        if j > 0 {
            let mut beta = 0.0_f64;
            for a in 0..=j {
                for b in 0..j {
                    beta += cs[(j, a)] * cs[(j - 1, b)] * moments[a + b + 1];
                }
            }
            if !beta.is_finite() {
                return None;
            }
            let beta_abs = beta.abs();
            jacobi[idx(j - 1, j)] = beta_abs;
            jacobi[idx(j, j - 1)] = beta_abs;
        }
    }

    let jacobi_matrix = MatrixFull::from_vec([nroots, nroots], jacobi)?;
    let (vectors_opt, roots, info) = _dsyev(&jacobi_matrix, 'V');
    if info != nroots as i32 || roots.len() != nroots {
        return None;
    }
    let vectors = vectors_opt?;
    if roots
        .iter()
        .any(|root| !root.is_finite() || *root < -1.0e-10 || *root > 1.0 + 1.0e-10)
    {
        return None;
    }

    let mut weights = Vec::with_capacity(nroots);
    for col in 0..nroots {
        let v0 = vectors[(0, col)];
        let weight = moments[0] * v0 * v0;
        if !weight.is_finite() || weight < -1.0e-12 {
            return None;
        }
        weights.push(weight.max(0.0));
    }
    validate_quadrature_moments(nroots, &roots, &weights, moments, 5.0e-10)?;
    Some((roots, weights))
}

fn validate_quadrature_moments(
    nroots: usize,
    roots: &[f64],
    weights: &[f64],
    moments: &[f64],
    rel_tol: f64,
) -> Option<()> {
    if roots.len() != nroots || weights.len() != nroots || moments.len() < 2 * nroots {
        return None;
    }
    for moment_idx in 0..=(2 * nroots - 1) {
        let got = roots
            .iter()
            .zip(weights.iter())
            .map(|(root, weight)| weight * root.powi(moment_idx as i32))
            .sum::<f64>();
        let expect = moments[moment_idx];
        if (got - expect).abs() > rel_tol * expect.abs().max(1.0) {
            return None;
        }
    }
    Some(())
}

const STIELTJES_BASE_N: usize = 192;

#[derive(Clone, Copy)]
enum RysMomentMeasure {
    RInv,
    RInv2Code,
}

fn gauss_legendre_unit_nodes_weights() -> &'static [(f64, f64)] {
    static CACHE: OnceLock<Vec<(f64, f64)>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let n = STIELTJES_BASE_N;
        let half = (n + 1) / 2;
        let mut nodes_weights = vec![(0.0_f64, 0.0_f64); n];
        for i in 0..half {
            let i_f = i as f64;
            let n_f = n as f64;
            let mut z = (std::f64::consts::PI * (i_f + 0.75) / (n_f + 0.5)).cos();
            let mut pp = 0.0_f64;
            for _ in 0..64 {
                let mut p1 = 1.0_f64;
                let mut p2 = 0.0_f64;
                for j in 1..=n {
                    let j_f = j as f64;
                    let p3 = p2;
                    p2 = p1;
                    p1 = ((2.0 * j_f - 1.0) * z * p2 - (j_f - 1.0) * p3) / j_f;
                }
                pp = n_f * (z * p1 - p2) / (z * z - 1.0);
                let z_next = z - p1 / pp;
                if (z_next - z).abs() <= 1.0e-15 {
                    z = z_next;
                    break;
                }
                z = z_next;
            }
            let x_left = 0.5 * (1.0 - z);
            let x_right = 0.5 * (1.0 + z);
            let weight = 1.0 / ((1.0 - z * z) * pp * pp);
            nodes_weights[i] = (x_left, weight);
            nodes_weights[n - 1 - i] = (x_right, weight);
        }
        nodes_weights
    })
}

fn rys_roots_weights_stieltjes(
    nroots: usize,
    t: f64,
    measure: RysMomentMeasure,
    moments: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    if nroots == 0 || moments.len() < 2 * nroots || !t.is_finite() || t < 0.0 {
        return None;
    }
    let base = gauss_legendre_unit_nodes_weights();
    let mut nodes = Vec::with_capacity(base.len());
    let mut weights = Vec::with_capacity(base.len());
    let t_measure = match measure {
        RysMomentMeasure::RInv2Code if t <= 1.0e-8 => 0.0,
        _ => t,
    };
    for &(s, legendre_weight) in base {
        let (node, weight) = match measure {
            RysMomentMeasure::RInv => {
                let node = s * s;
                (node, legendre_weight * (-t_measure * node).exp())
            }
            RysMomentMeasure::RInv2Code => {
                let node = 1.0 - s * s;
                (node, legendre_weight * (-t_measure * node).exp())
            }
        };
        if !node.is_finite() || !weight.is_finite() || weight < 0.0 {
            return None;
        }
        nodes.push(node.clamp(0.0, 1.0));
        weights.push(weight);
    }

    let mu0 = weights.iter().sum::<f64>();
    if mu0 <= 0.0 || !mu0.is_finite() {
        return None;
    }
    let mut p_prev = vec![0.0_f64; nodes.len()];
    let mut p_curr = vec![1.0 / mu0.sqrt(); nodes.len()];
    let mut beta_prev = 0.0_f64;
    let mut jacobi = vec![0.0_f64; nroots * nroots];
    let idx = |row: usize, col: usize| -> usize { row + nroots * col };

    for k in 0..nroots {
        let alpha = nodes
            .iter()
            .zip(weights.iter())
            .zip(p_curr.iter())
            .map(|((node, weight), p)| weight * node * p * p)
            .sum::<f64>();
        if !alpha.is_finite() {
            return None;
        }
        jacobi[idx(k, k)] = alpha;
        if k + 1 < nroots {
            let mut next = vec![0.0_f64; nodes.len()];
            for i in 0..nodes.len() {
                next[i] = (nodes[i] - alpha) * p_curr[i] - beta_prev * p_prev[i];
            }
            let norm_sq = next
                .iter()
                .zip(weights.iter())
                .map(|(value, weight)| weight * value * value)
                .sum::<f64>();
            if norm_sq <= 0.0 || !norm_sq.is_finite() {
                return None;
            }
            let beta = norm_sq.sqrt();
            jacobi[idx(k, k + 1)] = beta;
            jacobi[idx(k + 1, k)] = beta;
            for value in &mut next {
                *value /= beta;
            }
            p_prev = p_curr;
            p_curr = next;
            beta_prev = beta;
        }
    }

    let jacobi_matrix = MatrixFull::from_vec([nroots, nroots], jacobi)?;
    let (vectors_opt, roots, info) = _dsyev(&jacobi_matrix, 'V');
    if info != nroots as i32 || roots.len() != nroots {
        return None;
    }
    let vectors = vectors_opt?;
    if roots
        .iter()
        .any(|root| !root.is_finite() || *root < -1.0e-10 || *root > 1.0 + 1.0e-10)
    {
        return None;
    }
    let mut quad_weights = (0..nroots)
        .map(|col| {
            let v0 = vectors[(0, col)];
            mu0 * v0 * v0
        })
        .collect::<Vec<_>>();
    let sumw = quad_weights.iter().sum::<f64>();
    if sumw == 0.0 || !sumw.is_finite() {
        return None;
    }
    let scale = moments[0] / sumw;
    for weight in &mut quad_weights {
        *weight *= scale;
    }
    if quad_weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < -1.0e-12)
    {
        return None;
    }
    validate_quadrature_moments(nroots, &roots, &quad_weights, moments, 5.0e-8)?;
    Some((roots, quad_weights))
}

fn r2_stable_moments_from_measure(mmax: usize, t: f64) -> Option<Vec<f64>> {
    if !t.is_finite() || t < 0.0 {
        return None;
    }
    if t <= 1.0e-8 {
        return Some((0..=mmax).map(|m| 0.5 * beta(m)).collect());
    }
    let mut moments = vec![0.0_f64; mmax + 1];
    for &(s, legendre_weight) in gauss_legendre_unit_nodes_weights() {
        let node = (1.0 - s * s).clamp(0.0, 1.0);
        let scale = legendre_weight * (-t * node).exp();
        if !scale.is_finite() || scale < 0.0 {
            return None;
        }
        let mut power = 1.0_f64;
        for moment in &mut moments {
            *moment += scale * power;
            power *= node;
        }
    }
    Some(moments)
}

fn rys_roots_weights_r_from_moments(nroots: usize, ff: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    if nroots < 3 {
        return None;
    }
    if let Some((roots, weights)) = rys_roots_weights_from_moments_golub_welsch(nroots, ff) {
        return Some((roots, weights));
    }
    if ff.len() >= 2 * nroots + 1 {
        let nroots1 = nroots + 1;
        let mut s = zeros_2d(nroots1, nroots1);
        for j in 0..nroots1 {
            for i in 0..nroots1 {
                s[(i, j)] = ff[i + j];
            }
        }
        let cs = R_dsmit(&s, nroots1).ok()?;
        let poly_coeffs = (0..=nroots)
            .map(|col| cs[(nroots, col)])
            .collect::<Vec<_>>();
        let mut roots = polynomial_roots_unit_interval(&poly_coeffs);
        if roots.len() != nroots || roots.iter().any(|root| !root.is_finite()) {
            roots = find_polyroots(&cs, nroots);
        }
        if roots.len() != nroots || roots.iter().any(|root| !root.is_finite()) {
            return None;
        }
        roots.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
        let weights = rys_weights_from_orthogonal_polys(nroots, &roots, &cs, ff)
            .or_else(|| rys_weights_from_roots_moments(nroots, &roots, ff))?;
        return Some((roots, weights));
    }

    if ff.len() < 2 * nroots {
        return None;
    }

    let mut hankel = vec![0.0_f64; nroots * nroots];
    let mut rhs = vec![0.0_f64; nroots];
    for row in 0..nroots {
        for col in 0..nroots {
            hankel[row * nroots + col] = ff[row + col];
        }
        rhs[row] = -ff[row + nroots];
    }
    let coeffs = solve_linear_system(hankel, rhs, nroots)?;

    let mut poly_coeffs = coeffs;
    poly_coeffs.push(1.0);
    let roots = polynomial_roots_unit_interval(&poly_coeffs);
    if roots.len() != nroots || roots.iter().any(|root| !root.is_finite()) {
        return None;
    }

    let mut vandermonde = vec![0.0_f64; nroots * nroots];
    let mut weight_rhs = vec![0.0_f64; nroots];
    for row in 0..nroots {
        weight_rhs[row] = ff[row];
        for (col, root) in roots.iter().enumerate() {
            vandermonde[row * nroots + col] = root.powi(row as i32);
        }
    }
    let weights = solve_linear_system(vandermonde, weight_rhs, nroots)?;
    if weights.iter().any(|weight| !weight.is_finite()) {
        return None;
    }

    for moment_idx in 0..=(2 * nroots - 1) {
        let got = roots
            .iter()
            .zip(weights.iter())
            .map(|(root, weight)| weight * root.powi(moment_idx as i32))
            .sum::<f64>();
        let expect = ff[moment_idx];
        if (got - expect).abs() > 1.0e-8 * expect.abs().max(1.0) {
            return None;
        }
    }
    Some((roots, weights))
}

fn rys_weights_from_orthogonal_polys(
    nroots: usize,
    roots: &[f64],
    cs: &MatrixFull<f64>,
    ff: &[f64],
) -> Option<Vec<f64>> {
    if roots.len() != nroots || cs.size[0] < nroots || cs.size[1] < nroots || ff.len() < 2 * nroots
    {
        return None;
    }
    let mut weights = zeros_1d(nroots);
    for i in 0..nroots {
        let root = roots[i];
        let mut denom = if ff[0] == 0.0 { 0.0 } else { 1.0 / ff[0] };
        for j in 1..nroots {
            let row: Vec<f64> = (0..=j).map(|col| cs[(j, col)]).collect();
            let poly = poly_value1(&row, j, root);
            denom += poly * poly;
        }
        weights[i] = if denom == 0.0 || !denom.is_finite() {
            return None;
        } else {
            1.0 / denom
        };
    }
    let sumw = weights.iter().sum::<f64>();
    if sumw == 0.0 || !sumw.is_finite() {
        return None;
    }
    let scale = ff[0] / sumw;
    for weight in &mut weights {
        *weight *= scale;
    }
    if weights.iter().any(|weight| !weight.is_finite()) {
        return None;
    }
    for moment_idx in 0..=(2 * nroots - 1) {
        let got = roots
            .iter()
            .zip(weights.iter())
            .map(|(root, weight)| weight * root.powi(moment_idx as i32))
            .sum::<f64>();
        let expect = ff[moment_idx];
        if (got - expect).abs() > 1.0e-5 * expect.abs().max(1.0) {
            return None;
        }
    }
    Some(weights)
}

fn rys_weights_from_roots_moments(nroots: usize, roots: &[f64], ff: &[f64]) -> Option<Vec<f64>> {
    if roots.len() != nroots || ff.len() < 2 * nroots {
        return None;
    }
    if roots
        .iter()
        .any(|root| !root.is_finite() || *root < -1.0e-10 || *root > 1.0 + 1.0e-10)
    {
        return None;
    }

    let mut vandermonde = vec![0.0_f64; nroots * nroots];
    let mut rhs = vec![0.0_f64; nroots];
    for row in 0..nroots {
        rhs[row] = ff[row];
        for (col, root) in roots.iter().enumerate() {
            vandermonde[row * nroots + col] = root.powi(row as i32);
        }
    }
    let weights = solve_linear_system(vandermonde, rhs, nroots)?;
    if weights.iter().any(|weight| !weight.is_finite()) {
        return None;
    }
    for moment_idx in 0..=(2 * nroots - 1) {
        let got = roots
            .iter()
            .zip(weights.iter())
            .map(|(root, weight)| weight * root.powi(moment_idx as i32))
            .sum::<f64>();
        let expect = ff[moment_idx];
        if (got - expect).abs() > 1.0e-7 * expect.abs().max(1.0) {
            return None;
        }
    }
    Some(weights)
}

fn rys_roots_weights_r_into(
    nroots: usize,
    t: f64,
    roots: &mut [f64],
    weights: &mut [f64],
) -> usize {
    if nroots == 0 || roots.len() < nroots || weights.len() < nroots {
        return 0;
    }

    if nroots == 1 {
        let mut ff = [0.0_f64; 2];
        boys_slice(1, t, &mut ff);
        roots[0] = if ff[0] == 0.0 {
            0.0
        } else {
            let value = ff[1] / ff[0];
            if value.is_finite() {
                value
            } else {
                0.0
            }
        };
        weights[0] = ff[0];
        return 1;
    }

    if nroots == 2 {
        let mut ff = [0.0_f64; 4];
        boys_slice(3, t, &mut ff);
        let det = ff[1] * ff[1] - ff[2] * ff[0];
        if det.abs() > 1.0e-24 && det.is_finite() {
            let poly_a = (ff[3] * ff[0] - ff[2] * ff[1]) / det;
            let poly_b = (ff[2] * ff[2] - ff[1] * ff[3]) / det;
            let discriminant = poly_a * poly_a - 4.0 * poly_b;
            if discriminant >= 0.0 && discriminant.is_finite() {
                let sqrt_disc = discriminant.sqrt();
                roots[0] = (-poly_a - sqrt_disc) * 0.5;
                roots[1] = (-poly_a + sqrt_disc) * 0.5;
                let root_delta = roots[0] - roots[1];
                if root_delta.abs() > 1.0e-24 && roots[0].is_finite() && roots[1].is_finite() {
                    weights[0] = (ff[1] - ff[0] * roots[1]) / root_delta;
                    weights[1] = ff[0] - weights[0];
                    if weights[0].is_finite() && weights[1].is_finite() {
                        return 2;
                    }
                }
            }
        }
    }

    if nroots == 3 {
        let mut ff = [0.0_f64; 6];
        boys_slice(5, t, &mut ff);
        if let Some((roots_array, weights_array)) = rys_roots_weights_r3_from_moments(&ff) {
            roots[..3].copy_from_slice(&roots_array);
            weights[..3].copy_from_slice(&weights_array);
            return 3;
        }
    }

    if (4..=6).contains(&nroots) {
        let mut ff = [0.0_f64; 16];
        let mmax = 2 * nroots;
        boys_slice(mmax, t, &mut ff);
        if let Some((roots_vec, weights_vec)) =
            rys_roots_weights_r_from_moments(nroots, &ff[..=mmax])
        {
            roots[..nroots].copy_from_slice(&roots_vec);
            weights[..nroots].copy_from_slice(&weights_vec);
            return nroots;
        }
    }

    let (roots_vec, weights_vec) = rys_roots_weights_r(nroots, t);
    if roots_vec.len() != nroots || weights_vec.len() != nroots {
        return 0;
    }
    roots[..nroots].copy_from_slice(&roots_vec);
    weights[..nroots].copy_from_slice(&weights_vec);
    nroots
}

pub fn rys_roots_weights_r(nroots: usize, t: f64) -> (Vec<f64>, Vec<f64>) {
    if nroots == 1 {
        let ff = boys_vec(1, t);
        let root = if ff[0] == 0.0 {
            0.0
        } else {
            let value = ff[1] / ff[0];
            if value.is_finite() {
                value
            } else {
                0.0
            }
        };
        return (vec![root], vec![ff[0]]);
    }
    if nroots == 2 {
        let ff = boys_vec(3, t);
        let det = ff[1] * ff[1] - ff[2] * ff[0];
        if det.abs() > 1.0e-24 && det.is_finite() {
            let poly_a = (ff[3] * ff[0] - ff[2] * ff[1]) / det;
            let poly_b = (ff[2] * ff[2] - ff[1] * ff[3]) / det;
            let discriminant = poly_a * poly_a - 4.0 * poly_b;
            if discriminant >= 0.0 && discriminant.is_finite() {
                let sqrt_disc = discriminant.sqrt();
                let root1 = (-poly_a - sqrt_disc) * 0.5;
                let root2 = (-poly_a + sqrt_disc) * 0.5;
                let root_delta = root1 - root2;
                if root_delta.abs() > 1.0e-24 && root1.is_finite() && root2.is_finite() {
                    let weight1 = (ff[1] - ff[0] * root2) / root_delta;
                    let weight2 = ff[0] - weight1;
                    if weight1.is_finite() && weight2.is_finite() {
                        return (vec![root1, root2], vec![weight1, weight2]);
                    }
                }
            }
        }
    }
    if nroots == 3 {
        let ff = boys_vec(5, t);
        if let Some((roots, weights)) = rys_roots_weights_r3_from_moments(&ff) {
            return (roots.to_vec(), weights.to_vec());
        }
    }
    if (4..=6).contains(&nroots) {
        let ff = boys_vec(2 * nroots, t);
        if let Some((roots, weights)) = rys_roots_weights_r_from_moments(nroots, &ff) {
            return (roots, weights);
        }
    }
    let m = nroots * 2;
    let ff = boys_vec(m, t); // F_m(T)
    if let Some((roots, weights)) =
        rys_roots_weights_stieltjes(nroots, t, RysMomentMeasure::RInv, &ff)
    {
        return (roots, weights);
    }
    let nroots1 = nroots + 1;
    let mut s = zeros_2d(nroots1, nroots1);
    for j in 0..nroots1 {
        for i in 0..nroots1 {
            s[(i, j)] = ff[i + j]; // S_ij = F_{i+j}(T)
        }
    }
    let cs = match R_dsmit(&s, nroots1) {
        Ok(val) => val,
        Err(_) => return (zeros_1d(nroots), zeros_1d(nroots)),
    };
    let mut rt = find_polyroots(&cs, nroots);
    if rt.len() != nroots || rt.iter().any(|root| !root.is_finite()) {
        return (zeros_1d(nroots), zeros_1d(nroots));
    }
    rt.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    if let Some(weights) = rys_weights_from_roots_moments(nroots, &rt, &ff) {
        return (rt, weights);
    }
    let mut weights = zeros_1d(nroots);
    for i in 0..nroots {
        let root = rt[i];
        let mut dum = if ff[0] == 0.0 { 0.0 } else { 1.0 / ff[0] };
        for j in 1..nroots {
            let row: Vec<f64> = (0..=j).map(|col| cs[(j, col)]).collect();
            let poly = poly_value1(&row, j, root);
            dum += poly * poly;
        }
        weights[i] = if dum == 0.0 || !dum.is_finite() {
            0.0
        } else {
            1.0 / dum
        };
    }
    // normalization: sum_i w_i = F0(T)
    let f0 = ff[0];
    let sumw: f64 = weights.iter().sum();
    let scale = if sumw == 0.0 { 0.0 } else { f0 / sumw };
    for w in &mut weights {
        *w *= scale;
    }
    (rt[..nroots].to_vec(), weights)
}
pub fn rys_roots_weights_r2(nroots: usize, t: f64) -> (Vec<f64>, Vec<f64>) {
    let m = nroots * 2;
    let gg = boys_vec_r2(m, t); // G_m(T)
    if let Some((roots, weights)) = rys_roots_weights_r2_from_moments(nroots, &gg, true) {
        return (roots, weights);
    }
    if let Some(stable_moments) = r2_stable_moments_from_measure(m, t) {
        if let Some((roots, weights)) =
            rys_roots_weights_stieltjes(nroots, t, RysMomentMeasure::RInv2Code, &gg)
        {
            return (roots, weights);
        }
        if let Some((roots, weights)) =
            rys_roots_weights_stieltjes(nroots, t, RysMomentMeasure::RInv2Code, &stable_moments)
        {
            return (roots, weights);
        }
    }
    (zeros_1d(nroots), zeros_1d(nroots))
}

fn rys_roots_weights_r2_from_moments(
    nroots: usize,
    gg: &[f64],
    validate: bool,
) -> Option<(Vec<f64>, Vec<f64>)> {
    if nroots == 0 || gg.len() < 2 * nroots + 1 {
        return None;
    }
    let nroots1 = nroots + 1;
    let mut s = zeros_2d(nroots1, nroots1);
    for j in 0..nroots1 {
        for i in 0..nroots1 {
            s[(i, j)] = gg[i + j]; // S_ij = G_{i+j}(T)
        }
    }
    let cs = R_dsmit(&s, nroots1).ok()?;
    let rt = find_polyroots(&cs, nroots);
    if rt.len() != nroots {
        return None;
    }
    let mut weights = zeros_1d(nroots);
    for i in 0..nroots {
        let root = rt[i];
        let mut dum = if gg[0] == 0.0 { 0.0 } else { 1.0 / gg[0] };
        for j in 1..nroots {
            let row: Vec<f64> = (0..=j).map(|col| cs[(j, col)]).collect();
            let poly = poly_value1(&row, j, root);
            dum += poly * poly;
        }
        weights[i] = if dum == 0.0 || !dum.is_finite() {
            0.0
        } else {
            1.0 / dum
        };
    }
    // normalization: sum_i w_i = G0(T)
    let g0 = gg[0];
    let sumw: f64 = weights.iter().sum();
    let scale = if sumw == 0.0 { 0.0 } else { g0 / sumw };
    for w in &mut weights {
        *w *= scale;
    }
    let roots = rt[..nroots].to_vec();
    if validate {
        if roots
            .iter()
            .any(|root| !root.is_finite() || *root < -1.0e-10 || *root > 1.0 + 1.0e-10)
            || weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight < -1.0e-12)
        {
            return None;
        }
        validate_quadrature_moments(nroots, &roots, &weights, gg, 5.0e-8)?;
    }
    Some((roots, weights))
}
// ============================================================
// Geometry helpers (f64)
// ============================================================
pub fn distance_squared(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    let v = dx * dx + dy * dy + dz * dz;
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

pub fn calculate_nroots(
    bf1: &BasisFunction,
    bf2: &BasisFunction,
    bf3: &BasisFunction,
    bf4: &BasisFunction,
) -> usize {
    let lsum = bf1.lx
        + bf1.ly
        + bf1.lz
        + bf2.lx
        + bf2.ly
        + bf2.lz
        + bf3.lx
        + bf3.ly
        + bf3.lz
        + bf4.lx
        + bf4.ly
        + bf4.lz;
    (lsum / 2 + 1) as usize
}
const ZERO_ANG_MOM: [u32; 3] = [0, 0, 0];
#[derive(Clone, Copy, Debug)]
enum TwoElectronKernel {
    RInv,
    RInv2,
}
fn calculate_nroots_from_angmom(la: [u32; 3], lb: [u32; 3], lc: [u32; 3], ld: [u32; 3]) -> usize {
    let lsum = la[0]
        + la[1]
        + la[2]
        + lb[0]
        + lb[1]
        + lb[2]
        + lc[0]
        + lc[1]
        + lc[2]
        + ld[0]
        + ld[1]
        + ld[2];
    (lsum / 2 + 1) as usize
}
fn calculate_nroots_3c(la: [u32; 3], lb: [u32; 3], lc: [u32; 3]) -> usize {
    calculate_nroots_from_angmom(la, lb, lc, ZERO_ANG_MOM)
}
fn calculate_nroots_2c(la: [u32; 3], lc: [u32; 3]) -> usize {
    calculate_nroots_from_angmom(la, ZERO_ANG_MOM, lc, ZERO_ANG_MOM)
}
pub fn gaussian_product_center(alpha: f64, beta: f64, a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    let denom = alpha + beta;
    if denom == 0.0 || !denom.is_finite() {
        return [0.0, 0.0, 0.0];
    }
    array::from_fn(|i| {
        let num = alpha * a[i] + beta * b[i];
        let v = num / denom;
        if v.is_finite() {
            v
        } else {
            0.0
        }
    })
}
// ============================================================
// 1D Rys integral assembly (VRR seeds + transfers), f64
// ============================================================
pub fn compute_rys_integral_1d<C: CoeffProvider>(
    la: u32,
    lb: u32,
    lc: u32,
    ld: u32,
    root: &f64,
    p_center: f64,
    q_center: f64,
    a_center: f64,
    b_center: f64,
    c_center: f64,
    d_center: f64,
    alpha: &f64,
    beta: &f64,
    gamma: &f64,
    delta: &f64,
    rho: &f64,
    coeffs: &C,
) -> f64 {
    let ni_max = la;
    let nj_max = lb;
    let nk_max = lc;
    let nl_max = ld;
    let need_ni = ni_max + nj_max; // la + lb
    let need_nk = nk_max + nl_max; // lc + ld
    let z_panel = build_1d_panel_for_t(
        coeffs,
        root,
        PanelSpec {
            ni_max: need_ni,
            nk_max: need_nk,
        },
        &p_center,
        &q_center,
        &a_center,
        &b_center,
        &c_center,
        &d_center,
        alpha,
        beta,
        gamma,
        delta,
        rho,
    );
    // table(ni,0,nk,0)
    let mut table = HashMap::<(u32, u32, u32, u32), f64>::new();
    for nk in 0..=need_nk {
        for ni in 0..=need_ni {
            let key_1d = (ni, nk);
            let val = *z_panel.get(&key_1d).unwrap_or_else(|| {
                panic!(
                    "VRR seed missing for (u,0,v,0): need (ni={},0,nk={},0) but z_panel[{:?}] not present",
                    ni, nk, key_1d
                )
            });
            table.insert((ni, 0, nk, 0), val);
        }
    }
    // Transfers: i->j, k->l
    let xi_minus_xj = a_center - b_center;
    let xk_minus_xl = c_center - d_center;
    for nk in 0..=need_nk {
        transfer_i_to_j(&mut table, &xi_minus_xj, need_ni, nj_max, nk, 0);
    }
    transfer_k_to_l(&mut table, &xk_minus_xl, need_ni, nj_max, need_nk, nl_max);
    *table.get(&(la, lb, lc, ld)).unwrap_or(&0.0)
}
// ============================================================
// Primitive ERI kernel
// ============================================================
fn gaussian_primitive_kernel(
    kernel: TwoElectronKernel,
    nroots: usize,
    alpha: f64,
    beta: f64,
    gamma: f64,
    delta: f64,
    rab2: f64,
    rcd2: f64,
    rpq2: f64,
    la: [u32; 3],
    lb: [u32; 3],
    lc: [u32; 3],
    ld: [u32; 3],
    p_center: &[f64; 3],
    q_center: &[f64; 3],
    a_center: &[f64; 3],
    b_center: &[f64; 3],
    c_center: &[f64; 3],
    d_center: &[f64; 3],
) -> f64 {
    let pi = std::f64::consts::PI;
    let p_sum = alpha + beta;
    let q_sum = gamma + delta;
    let prod_ab = alpha * beta;
    let p_fac = prod_ab / p_sum; // P in your code
    let prod_cd = gamma * delta;
    let q_fac = prod_cd / q_sum; // Q in your code
    let p_sum_q = p_sum + q_sum;
    let pq_mul = p_sum * q_sum;
    let pref = match kernel {
        TwoElectronKernel::RInv => {
            let denom_r1 = pq_mul * (p_sum_q.sqrt());
            2.0 * pi.powf(2.5) / denom_r1
        }
        TwoElectronKernel::RInv2 => {
            let denom_r2 = (pq_mul.sqrt()) * p_sum_q;
            2.0 * pi.powf(3.0) / denom_r2
        }
    };
    let exp_ab = (-p_fac * rab2).exp();
    let exp_cd = (-q_fac * rcd2).exp();
    let exp_abcd = exp_ab * exp_cd;
    let rho = pq_mul / p_sum_q;
    let t = rho * rpq2;
    let (roots, weights) = match kernel {
        TwoElectronKernel::RInv => rys_roots_weights_r(nroots, t),
        TwoElectronKernel::RInv2 => rys_roots_weights_r2(nroots, t),
    };
    let vrr = RysCoeffProvider;
    let mut eri = 0.0_f64;
    for r in 0..nroots {
        let root = &roots[r];
        let weight = &weights[r];
        let ix = compute_rys_integral_1d(
            la[0],
            lb[0],
            lc[0],
            ld[0],
            root,
            p_center[0],
            q_center[0],
            a_center[0],
            b_center[0],
            c_center[0],
            d_center[0],
            &alpha,
            &beta,
            &gamma,
            &delta,
            &rho,
            &vrr,
        );
        let iy = compute_rys_integral_1d(
            la[1],
            lb[1],
            lc[1],
            ld[1],
            root,
            p_center[1],
            q_center[1],
            a_center[1],
            b_center[1],
            c_center[1],
            d_center[1],
            &alpha,
            &beta,
            &gamma,
            &delta,
            &rho,
            &vrr,
        );
        let iz = compute_rys_integral_1d(
            la[2],
            lb[2],
            lc[2],
            ld[2],
            root,
            p_center[2],
            q_center[2],
            a_center[2],
            b_center[2],
            c_center[2],
            d_center[2],
            &alpha,
            &beta,
            &gamma,
            &delta,
            &rho,
            &vrr,
        );
        let term = (*weight) * ix * iy * iz;
        if term.is_finite() {
            eri += term;
        }
    }
    let result = (exp_abcd * pref) * eri;
    if result.is_finite() {
        result
    } else {
        0.0
    }
}
pub fn gaussian_4c_r(
    nroots: usize,
    alpha: &f64,
    beta: &f64,
    gamma: &f64,
    delta: &f64,
    rab2: &f64,
    rcd2: &f64,
    rpq2: &f64,
    la: [u32; 3],
    lb: [u32; 3],
    lc: [u32; 3],
    ld: [u32; 3],
    p_center: &[f64; 3],
    q_center: &[f64; 3],
    a_center: &[f64; 3],
    b_center: &[f64; 3],
    c_center: &[f64; 3],
    d_center: &[f64; 3],
    _i: usize,
    _j: usize,
    _k: usize,
    _l: usize,
) -> f64 {
    gaussian_primitive_kernel(
        TwoElectronKernel::RInv,
        nroots,
        *alpha,
        *beta,
        *gamma,
        *delta,
        *rab2,
        *rcd2,
        *rpq2,
        la,
        lb,
        lc,
        ld,
        p_center,
        q_center,
        a_center,
        b_center,
        c_center,
        d_center,
    )
}
pub fn gaussian_4c_r2(
    nroots: usize,
    alpha: &f64,
    beta: &f64,
    gamma: &f64,
    delta: &f64,
    rab2: &f64,
    rcd2: &f64,
    rpq2: &f64,
    la: [u32; 3],
    lb: [u32; 3],
    lc: [u32; 3],
    ld: [u32; 3],
    p_center: &[f64; 3],
    q_center: &[f64; 3],
    a_center: &[f64; 3],
    b_center: &[f64; 3],
    c_center: &[f64; 3],
    d_center: &[f64; 3],
    _i: usize,
    _j: usize,
    _k: usize,
    _l: usize,
) -> f64 {
    gaussian_primitive_kernel(
        TwoElectronKernel::RInv2,
        nroots,
        *alpha,
        *beta,
        *gamma,
        *delta,
        *rab2,
        *rcd2,
        *rpq2,
        la,
        lb,
        lc,
        ld,
        p_center,
        q_center,
        a_center,
        b_center,
        c_center,
        d_center,
    )
}
pub fn gaussian_3c_r(
    nroots: usize,
    alpha: &f64,
    beta: &f64,
    gamma: &f64,
    rab2: &f64,
    rpq2: &f64,
    la: [u32; 3],
    lb: [u32; 3],
    lc: [u32; 3],
    p_center: &[f64; 3],
    a_center: &[f64; 3],
    b_center: &[f64; 3],
    c_center: &[f64; 3],
) -> f64 {
    gaussian_primitive_kernel(
        TwoElectronKernel::RInv,
        nroots,
        *alpha,
        *beta,
        *gamma,
        0.0,
        *rab2,
        0.0,
        *rpq2,
        la,
        lb,
        lc,
        ZERO_ANG_MOM,
        p_center,
        c_center,
        a_center,
        b_center,
        c_center,
        c_center,
    )
}
pub fn gaussian_3c_r2(
    nroots: usize,
    alpha: &f64,
    beta: &f64,
    gamma: &f64,
    rab2: &f64,
    rpq2: &f64,
    la: [u32; 3],
    lb: [u32; 3],
    lc: [u32; 3],
    p_center: &[f64; 3],
    a_center: &[f64; 3],
    b_center: &[f64; 3],
    c_center: &[f64; 3],
) -> f64 {
    gaussian_primitive_kernel(
        TwoElectronKernel::RInv2,
        nroots,
        *alpha,
        *beta,
        *gamma,
        0.0,
        *rab2,
        0.0,
        *rpq2,
        la,
        lb,
        lc,
        ZERO_ANG_MOM,
        p_center,
        c_center,
        a_center,
        b_center,
        c_center,
        c_center,
    )
}
pub fn gaussian_2c_r(
    nroots: usize,
    alpha: &f64,
    gamma: &f64,
    rpq2: &f64,
    la: [u32; 3],
    lc: [u32; 3],
    a_center: &[f64; 3],
    c_center: &[f64; 3],
) -> f64 {
    gaussian_primitive_kernel(
        TwoElectronKernel::RInv,
        nroots,
        *alpha,
        0.0,
        *gamma,
        0.0,
        0.0,
        0.0,
        *rpq2,
        la,
        ZERO_ANG_MOM,
        lc,
        ZERO_ANG_MOM,
        a_center,
        c_center,
        a_center,
        a_center,
        c_center,
        c_center,
    )
}
pub fn gaussian_2c_r2(
    nroots: usize,
    alpha: &f64,
    gamma: &f64,
    rpq2: &f64,
    la: [u32; 3],
    lc: [u32; 3],
    a_center: &[f64; 3],
    c_center: &[f64; 3],
) -> f64 {
    gaussian_primitive_kernel(
        TwoElectronKernel::RInv2,
        nroots,
        *alpha,
        0.0,
        *gamma,
        0.0,
        0.0,
        0.0,
        *rpq2,
        la,
        ZERO_ANG_MOM,
        lc,
        ZERO_ANG_MOM,
        a_center,
        c_center,
        a_center,
        a_center,
        c_center,
        c_center,
    )
}
pub fn eri_ao_4c_r(bfs: &[BasisFunction], i: usize, j: usize, k: usize, l: usize) -> f64 {
    let bf1 = &bfs[i];
    let bf2 = &bfs[j];
    let bf3 = &bfs[k];
    let bf4 = &bfs[l];
    let rab2 = distance_squared(&bf1.center, &bf2.center);
    let rcd2 = distance_squared(&bf3.center, &bf4.center);
    let la = [bf1.lx, bf1.ly, bf1.lz];
    let lb = [bf2.lx, bf2.ly, bf2.lz];
    let lc = [bf3.lx, bf3.ly, bf3.lz];
    let ld = [bf4.lx, bf4.ly, bf4.lz];
    let nroots = calculate_nroots(bf1, bf2, bf3, bf4);
    let mut eri = 0.0_f64;
    for (&a_exp, &a_coeff) in bf1.exponents.iter().zip(bf1.coefficients.iter()) {
        let alpha = a_exp;
        for (&b_exp, &b_coeff) in bf2.exponents.iter().zip(bf2.coefficients.iter()) {
            let beta = b_exp;
            // p_center
            let p_center = gaussian_product_center(alpha, beta, &bf1.center, &bf2.center);
            for (&c_exp, &c_coeff) in bf3.exponents.iter().zip(bf3.coefficients.iter()) {
                let gamma = c_exp;
                for (&d_exp, &d_coeff) in bf4.exponents.iter().zip(bf4.coefficients.iter()) {
                    let delta = d_exp;
                    // q_center/rpq2
                    let q_center = gaussian_product_center(gamma, delta, &bf3.center, &bf4.center);
                    let rpq2 = distance_squared(&p_center, &q_center);
                    // primitive kernel
                    let eri_prim = gaussian_4c_r(
                        nroots,
                        &alpha,
                        &beta,
                        &gamma,
                        &delta,
                        &rab2,
                        &rcd2,
                        &rpq2,
                        la,
                        lb,
                        lc,
                        ld,
                        &p_center,
                        &q_center,
                        &bf1.center,
                        &bf2.center,
                        &bf3.center,
                        &bf4.center,
                        i,
                        j,
                        k,
                        l,
                    );
                    // contraction
                    let coeff_prod = a_coeff * b_coeff * c_coeff * d_coeff;
                    let term = coeff_prod * eri_prim;
                    if term.is_finite() {
                        eri += term;
                    }
                }
            }
        }
    }
    eri
}
pub fn eri_ao_3c_r(
    ao_bfs: &[BasisFunction],
    aux_bfs: &[BasisFunction],
    i: usize,
    j: usize,
    k: usize,
) -> f64 {
    let bf1 = &ao_bfs[i];
    let bf2 = &ao_bfs[j];
    let bf3 = &aux_bfs[k];
    let rab2 = distance_squared(&bf1.center, &bf2.center);
    let la = [bf1.lx, bf1.ly, bf1.lz];
    let lb = [bf2.lx, bf2.ly, bf2.lz];
    let lc = [bf3.lx, bf3.ly, bf3.lz];
    let nroots = calculate_nroots_3c(la, lb, lc);
    let mut eri = 0.0_f64;
    for (&a_exp, &a_coeff) in bf1.exponents.iter().zip(bf1.coefficients.iter()) {
        let alpha = a_exp;
        for (&b_exp, &b_coeff) in bf2.exponents.iter().zip(bf2.coefficients.iter()) {
            let beta = b_exp;
            let p_center = gaussian_product_center(alpha, beta, &bf1.center, &bf2.center);
            for (&c_exp, &c_coeff) in bf3.exponents.iter().zip(bf3.coefficients.iter()) {
                let gamma = c_exp;
                let rpq2 = distance_squared(&p_center, &bf3.center);
                let eri_prim = gaussian_3c_r(
                    nroots,
                    &alpha,
                    &beta,
                    &gamma,
                    &rab2,
                    &rpq2,
                    la,
                    lb,
                    lc,
                    &p_center,
                    &bf1.center,
                    &bf2.center,
                    &bf3.center,
                );
                let term = a_coeff * b_coeff * c_coeff * eri_prim;
                if term.is_finite() {
                    eri += term;
                }
            }
        }
    }
    eri
}
pub fn eri_ao_4c_r2(bfs: &[BasisFunction], i: usize, j: usize, k: usize, l: usize) -> f64 {
    let bf1 = &bfs[i];
    let bf2 = &bfs[j];
    let bf3 = &bfs[k];
    let bf4 = &bfs[l];
    let rab2 = distance_squared(&bf1.center, &bf2.center);
    let rcd2 = distance_squared(&bf3.center, &bf4.center);
    let la = [bf1.lx, bf1.ly, bf1.lz];
    let lb = [bf2.lx, bf2.ly, bf2.lz];
    let lc = [bf3.lx, bf3.ly, bf3.lz];
    let ld = [bf4.lx, bf4.ly, bf4.lz];
    let nroots = calculate_nroots(bf1, bf2, bf3, bf4);
    let mut eri = 0.0_f64;
    for (&a_exp, &a_coeff) in bf1.exponents.iter().zip(bf1.coefficients.iter()) {
        let alpha = a_exp;
        for (&b_exp, &b_coeff) in bf2.exponents.iter().zip(bf2.coefficients.iter()) {
            let beta = b_exp;
            // p_center
            let p_center = gaussian_product_center(alpha, beta, &bf1.center, &bf2.center);
            for (&c_exp, &c_coeff) in bf3.exponents.iter().zip(bf3.coefficients.iter()) {
                let gamma = c_exp;
                for (&d_exp, &d_coeff) in bf4.exponents.iter().zip(bf4.coefficients.iter()) {
                    let delta = d_exp;
                    // q_center/rpq2
                    let q_center = gaussian_product_center(gamma, delta, &bf3.center, &bf4.center);
                    let rpq2 = distance_squared(&p_center, &q_center);
                    // primitive kernel
                    let eri_prim = gaussian_4c_r2(
                        nroots,
                        &alpha,
                        &beta,
                        &gamma,
                        &delta,
                        &rab2,
                        &rcd2,
                        &rpq2,
                        la,
                        lb,
                        lc,
                        ld,
                        &p_center,
                        &q_center,
                        &bf1.center,
                        &bf2.center,
                        &bf3.center,
                        &bf4.center,
                        i,
                        j,
                        k,
                        l,
                    );
                    // contraction
                    let coeff_prod = a_coeff * b_coeff * c_coeff * d_coeff;
                    let term = coeff_prod * eri_prim;
                    if term.is_finite() {
                        eri += term;
                    }
                }
            }
        }
    }
    eri
}
pub fn eri_ao_3c_r2(
    ao_bfs: &[BasisFunction],
    aux_bfs: &[BasisFunction],
    i: usize,
    j: usize,
    k: usize,
) -> f64 {
    let bf1 = &ao_bfs[i];
    let bf2 = &ao_bfs[j];
    let bf3 = &aux_bfs[k];
    let rab2 = distance_squared(&bf1.center, &bf2.center);
    let la = [bf1.lx, bf1.ly, bf1.lz];
    let lb = [bf2.lx, bf2.ly, bf2.lz];
    let lc = [bf3.lx, bf3.ly, bf3.lz];
    let nroots = calculate_nroots_3c(la, lb, lc);
    let mut eri = 0.0_f64;
    for (&a_exp, &a_coeff) in bf1.exponents.iter().zip(bf1.coefficients.iter()) {
        let alpha = a_exp;
        for (&b_exp, &b_coeff) in bf2.exponents.iter().zip(bf2.coefficients.iter()) {
            let beta = b_exp;
            let p_center = gaussian_product_center(alpha, beta, &bf1.center, &bf2.center);
            for (&c_exp, &c_coeff) in bf3.exponents.iter().zip(bf3.coefficients.iter()) {
                let gamma = c_exp;
                let rpq2 = distance_squared(&p_center, &bf3.center);
                let eri_prim = gaussian_3c_r2(
                    nroots,
                    &alpha,
                    &beta,
                    &gamma,
                    &rab2,
                    &rpq2,
                    la,
                    lb,
                    lc,
                    &p_center,
                    &bf1.center,
                    &bf2.center,
                    &bf3.center,
                );
                let term = a_coeff * b_coeff * c_coeff * eri_prim;
                if term.is_finite() {
                    eri += term;
                }
            }
        }
    }
    eri
}
pub fn eri_ao_2c_r(aux_bfs: &[BasisFunction], i: usize, j: usize) -> f64 {
    let bf1 = &aux_bfs[i];
    let bf3 = &aux_bfs[j];
    let la = [bf1.lx, bf1.ly, bf1.lz];
    let lc = [bf3.lx, bf3.ly, bf3.lz];
    let nroots = calculate_nroots_2c(la, lc);
    let mut eri = 0.0_f64;
    for (&a_exp, &a_coeff) in bf1.exponents.iter().zip(bf1.coefficients.iter()) {
        let alpha = a_exp;
        for (&c_exp, &c_coeff) in bf3.exponents.iter().zip(bf3.coefficients.iter()) {
            let gamma = c_exp;
            let rpq2 = distance_squared(&bf1.center, &bf3.center);
            let eri_prim = gaussian_2c_r(
                nroots,
                &alpha,
                &gamma,
                &rpq2,
                la,
                lc,
                &bf1.center,
                &bf3.center,
            );
            let term = a_coeff * c_coeff * eri_prim;
            if term.is_finite() {
                eri += term;
            }
        }
    }
    eri
}
pub fn eri_ao_2c_r2(aux_bfs: &[BasisFunction], i: usize, j: usize) -> f64 {
    let bf1 = &aux_bfs[i];
    let bf3 = &aux_bfs[j];
    let la = [bf1.lx, bf1.ly, bf1.lz];
    let lc = [bf3.lx, bf3.ly, bf3.lz];
    let nroots = calculate_nroots_2c(la, lc);
    let mut eri = 0.0_f64;
    for (&a_exp, &a_coeff) in bf1.exponents.iter().zip(bf1.coefficients.iter()) {
        let alpha = a_exp;
        for (&c_exp, &c_coeff) in bf3.exponents.iter().zip(bf3.coefficients.iter()) {
            let gamma = c_exp;
            let rpq2 = distance_squared(&bf1.center, &bf3.center);
            let eri_prim = gaussian_2c_r2(
                nroots,
                &alpha,
                &gamma,
                &rpq2,
                la,
                lc,
                &bf1.center,
                &bf3.center,
            );
            let term = a_coeff * c_coeff * eri_prim;
            if term.is_finite() {
                eri += term;
            }
        }
    }
    eri
}

#[inline]
fn rint_shell_coefficients(shell: &RintShell) -> &[f64] {
    shell
        .shell
        .coefficients
        .first()
        .expect("RintShell must contain exactly one normalized coefficient column")
}

fn rint_shell_log_abs_coefficients(shell: &RintShell) -> Vec<f64> {
    rint_shell_coefficients(shell)
        .iter()
        .map(|coeff| {
            if *coeff == 0.0 {
                f64::NEG_INFINITY
            } else {
                coeff.abs().ln()
            }
        })
        .collect()
}

fn libcint_pair_log_bound(left: &RintShell, right: &RintShell, rab2: f64) -> f64 {
    let min_exp_sum = left
        .shell
        .exponents
        .iter()
        .flat_map(|alpha| right.shell.exponents.iter().map(move |beta| alpha + beta))
        .fold(f64::INFINITY, f64::min);
    if !min_exp_sum.is_finite() || min_exp_sum <= 0.0 {
        return f64::INFINITY;
    }
    let lij = left.shell.ang_type + right.shell.ang_type;
    let mut log_bound = 1.7 - 1.5 * min_exp_sum.ln();
    if lij > 0 {
        log_bound += (lij as f64) * (rab2.sqrt() + 1.0).ln();
    }
    log_bound
}

fn libcint_pair_cceij(
    alpha: f64,
    beta: f64,
    log_abs_left_coeff: f64,
    log_abs_right_coeff: f64,
    rab2: f64,
    pair_log_bound: f64,
) -> f64 {
    if !log_abs_left_coeff.is_finite() || !log_abs_right_coeff.is_finite() {
        return f64::INFINITY;
    }
    let aij = alpha + beta;
    if aij <= 0.0 {
        return f64::INFINITY;
    }
    let eij = rab2 * alpha * beta / aij;
    eij - pair_log_bound - log_abs_left_coeff - log_abs_right_coeff
}

fn eri_rint_shell_2c_r2(
    left: &RintShell,
    left_cart: usize,
    right: &RintShell,
    right_cart: usize,
) -> f64 {
    let left_coeffs = rint_shell_coefficients(left);
    let right_coeffs = rint_shell_coefficients(right);
    let la = left.cart_components[left_cart];
    let lc = right.cart_components[right_cart];
    let nroots = calculate_nroots_2c(la, lc);
    let rpq2 = distance_squared(&left.center, &right.center);
    let mut eri = 0.0_f64;
    for (&alpha, &a_coeff) in left.shell.exponents.iter().zip(left_coeffs.iter()) {
        for (&gamma, &c_coeff) in right.shell.exponents.iter().zip(right_coeffs.iter()) {
            let eri_prim = gaussian_2c_r2(
                nroots,
                &alpha,
                &gamma,
                &rpq2,
                la,
                lc,
                &left.center,
                &right.center,
            );
            let term = a_coeff * c_coeff * eri_prim;
            if term.is_finite() {
                eri += term;
            }
        }
    }
    eri
}

fn build_rys_transfer_panel_2c(
    root: &f64,
    ni_max: u32,
    nk_max: u32,
    a_center_axis: f64,
    c_center_axis: f64,
    alpha: &f64,
    gamma: &f64,
    rho: &f64,
    coeffs: &RysCoeffProvider,
) -> HashMap<(u32, u32), f64> {
    let zero = 0.0_f64;
    build_1d_panel_for_t(
        coeffs,
        root,
        PanelSpec { ni_max, nk_max },
        &a_center_axis,
        &c_center_axis,
        &a_center_axis,
        &a_center_axis,
        &c_center_axis,
        &c_center_axis,
        alpha,
        &zero,
        gamma,
        &zero,
        rho,
    )
}

struct RysTransferTable2c {
    ni_dim: usize,
    nk_dim: usize,
    data: Vec<f64>,
}

impl RysTransferTable2c {
    fn new(ni_max: u32, nk_max: u32) -> Self {
        let ni_dim = ni_max as usize + 1;
        let nk_dim = nk_max as usize + 1;
        Self {
            ni_dim,
            nk_dim,
            data: vec![0.0; ni_dim * nk_dim],
        }
    }

    #[inline(always)]
    fn index(&self, ni: u32, nk: u32) -> usize {
        nk as usize * self.ni_dim + ni as usize
    }

    #[inline(always)]
    fn get(&self, ni: u32, nk: u32) -> f64 {
        if ni as usize >= self.ni_dim || nk as usize >= self.nk_dim {
            0.0
        } else {
            self.data[self.index(ni, nk)]
        }
    }

    #[inline(always)]
    fn set(&mut self, ni: u32, nk: u32, value: f64) {
        let idx = self.index(ni, nk);
        self.data[idx] = value;
    }
}

#[allow(clippy::too_many_arguments)]
fn build_rys_transfer_table_2c(
    root: &f64,
    ni_max: u32,
    nk_max: u32,
    a_center_axis: f64,
    c_center_axis: f64,
    alpha: &f64,
    gamma: &f64,
    rho: &f64,
    coeffs: &RysCoeffProvider,
) -> RysTransferTable2c {
    let zero = 0.0_f64;
    let rc = coeffs.coeffs_for(
        root,
        &a_center_axis,
        &c_center_axis,
        &a_center_axis,
        &a_center_axis,
        &c_center_axis,
        &c_center_axis,
        alpha,
        &zero,
        gamma,
        &zero,
        rho,
    );

    let mut table = RysTransferTable2c::new(ni_max, nk_max);
    table.set(0, 0, 1.0);

    if ni_max >= 1 {
        table.set(1, 0, rc.c00);
        let mut g_nm1 = 1.0_f64;
        let mut g_n = rc.c00;
        for n in 1..ni_max {
            let cur = (n as f64) * rc.b10 * g_nm1 + rc.c00 * g_n;
            table.set(n + 1, 0, cur);
            g_nm1 = g_n;
            g_n = cur;
        }
    }

    if nk_max >= 1 {
        table.set(0, 1, rc.c00p);
        let mut g_mn1 = 1.0_f64;
        let mut g_m = rc.c00p;
        for m in 1..nk_max {
            let cur = (m as f64) * rc.b01p * g_mn1 + rc.c00p * g_m;
            table.set(0, m + 1, cur);
            g_mn1 = g_m;
            g_m = cur;
        }
    }

    if nk_max >= 1 {
        for n in 0..=ni_max {
            let mut value = rc.c00p * table.get(n, 0);
            if n > 0 {
                value += (n as f64) * rc.b00 * table.get(n - 1, 0);
            }
            table.set(n, 1, value);
        }
    }

    if ni_max >= 1 {
        for m in 0..=nk_max {
            let mut value = rc.c00 * table.get(0, m);
            if m > 0 {
                value += (m as f64) * rc.b00 * table.get(0, m - 1);
            }
            table.set(1, m, value);
        }
    }

    for n in 1..=ni_max {
        for m in 1..nk_max {
            let value = (m as f64) * rc.b01p * table.get(n, m - 1)
                + (n as f64) * rc.b00 * table.get(n - 1, m)
                + rc.c00p * table.get(n, m);
            table.set(n, m + 1, value);
        }
    }

    table
}

fn add_primitive_2c_r2_shell_block(
    block: &mut MatrixFull<f64>,
    left: &RintShell,
    right: &RintShell,
    alpha: f64,
    gamma: f64,
    coeff_prod: f64,
    rpq2: f64,
) {
    let pi = std::f64::consts::PI;
    let p_sum = alpha;
    let q_sum = gamma;
    let p_sum_q = p_sum + q_sum;
    let pq_mul = p_sum * q_sum;
    if p_sum <= 0.0 || q_sum <= 0.0 || p_sum_q <= 0.0 || pq_mul <= 0.0 {
        return;
    }

    let pref = 2.0 * pi.powf(3.0) / (pq_mul.sqrt() * p_sum_q);
    let rho = pq_mul / p_sum_q;
    let t = rho * rpq2;
    let nroots = ((left.shell.ang_type + right.shell.ang_type) / 2 + 1) as usize;
    let (roots, weights) = rys_roots_weights_r2(nroots, t);
    let vrr = RysCoeffProvider;
    let ni_max = left.shell.ang_type;
    let nk_max = right.shell.ang_type;

    for root_idx in 0..nroots {
        let root = &roots[root_idx];
        let weight = weights[root_idx];
        let table_x = build_rys_transfer_table_2c(
            root,
            ni_max,
            nk_max,
            left.center[0],
            right.center[0],
            &alpha,
            &gamma,
            &rho,
            &vrr,
        );
        let table_y = build_rys_transfer_table_2c(
            root,
            ni_max,
            nk_max,
            left.center[1],
            right.center[1],
            &alpha,
            &gamma,
            &rho,
            &vrr,
        );
        let table_z = build_rys_transfer_table_2c(
            root,
            ni_max,
            nk_max,
            left.center[2],
            right.center[2],
            &alpha,
            &gamma,
            &rho,
            &vrr,
        );

        let primitive_scale = coeff_prod * pref * weight;
        for j in 0..right.ao_len {
            let right_ang = right.cart_components[j];
            let col_offset = j * left.ao_len;
            for i in 0..left.ao_len {
                let left_ang = left.cart_components[i];
                let ix = table_x.get(left_ang[0], right_ang[0]);
                let iy = table_y.get(left_ang[1], right_ang[1]);
                let iz = table_z.get(left_ang[2], right_ang[2]);
                let term = primitive_scale * ix * iy * iz;
                if term.is_finite() {
                    block.data[col_offset + i] += term;
                }
            }
        }
    }
}

fn int2c_r2_shell_block_batched(left: &RintShell, right: &RintShell) -> MatrixFull<f64> {
    let mut block = MatrixFull::new([left.ao_len, right.ao_len], 0.0_f64);
    let left_coeffs = rint_shell_coefficients(left);
    let right_coeffs = rint_shell_coefficients(right);
    let rpq2 = distance_squared(&left.center, &right.center);

    for (&alpha, &a_coeff) in left.shell.exponents.iter().zip(left_coeffs.iter()) {
        for (&gamma, &c_coeff) in right.shell.exponents.iter().zip(right_coeffs.iter()) {
            add_primitive_2c_r2_shell_block(
                &mut block,
                left,
                right,
                alpha,
                gamma,
                a_coeff * c_coeff,
                rpq2,
            );
        }
    }
    block
}

fn eri_rint_shell_3c_r2(
    left: &RintShell,
    left_cart: usize,
    right: &RintShell,
    right_cart: usize,
    aux: &RintShell,
    aux_cart: usize,
) -> f64 {
    let left_coeffs = rint_shell_coefficients(left);
    let right_coeffs = rint_shell_coefficients(right);
    let aux_coeffs = rint_shell_coefficients(aux);
    let la = left.cart_components[left_cart];
    let lb = right.cart_components[right_cart];
    let lc = aux.cart_components[aux_cart];
    let nroots = calculate_nroots_3c(la, lb, lc);
    let rab2 = distance_squared(&left.center, &right.center);
    let mut eri = 0.0_f64;
    for (&alpha, &a_coeff) in left.shell.exponents.iter().zip(left_coeffs.iter()) {
        for (&beta, &b_coeff) in right.shell.exponents.iter().zip(right_coeffs.iter()) {
            let p_center = gaussian_product_center(alpha, beta, &left.center, &right.center);
            let rpq2 = distance_squared(&p_center, &aux.center);
            for (&gamma, &c_coeff) in aux.shell.exponents.iter().zip(aux_coeffs.iter()) {
                let eri_prim = gaussian_3c_r2(
                    nroots,
                    &alpha,
                    &beta,
                    &gamma,
                    &rab2,
                    &rpq2,
                    la,
                    lb,
                    lc,
                    &p_center,
                    &left.center,
                    &right.center,
                    &aux.center,
                );
                let term = a_coeff * b_coeff * c_coeff * eri_prim;
                if term.is_finite() {
                    eri += term;
                }
            }
        }
    }
    eri
}

struct RysTransferTable3c {
    ni_dim: usize,
    nj_dim: usize,
    nk_dim: usize,
    data: Vec<f64>,
}

impl RysTransferTable3c {
    fn new(ni_max: u32, nj_max: u32, nk_max: u32) -> Self {
        let ni_dim = ni_max as usize + 1;
        let nj_dim = nj_max as usize + 1;
        let nk_dim = nk_max as usize + 1;
        Self {
            ni_dim,
            nj_dim,
            nk_dim,
            data: vec![0.0; ni_dim * nj_dim * nk_dim],
        }
    }

    #[inline(always)]
    fn index(&self, ni: u32, nj: u32, nk: u32) -> usize {
        ((nk as usize * self.nj_dim + nj as usize) * self.ni_dim) + ni as usize
    }

    #[inline(always)]
    fn get(&self, ni: u32, nj: u32, nk: u32) -> f64 {
        if ni as usize >= self.ni_dim || nj as usize >= self.nj_dim || nk as usize >= self.nk_dim {
            0.0
        } else {
            self.data[self.index(ni, nj, nk)]
        }
    }

    #[inline(always)]
    fn set(&mut self, ni: u32, nj: u32, nk: u32, value: f64) {
        let idx = self.index(ni, nj, nk);
        self.data[idx] = value;
    }
}

#[allow(clippy::too_many_arguments)]
fn build_rys_transfer_table_3c(
    root: &f64,
    ni_max: u32,
    nj_max: u32,
    nk_max: u32,
    p_center_axis: f64,
    a_center_axis: f64,
    b_center_axis: f64,
    c_center_axis: f64,
    alpha: &f64,
    beta: &f64,
    gamma: &f64,
    rho: &f64,
    coeffs: &RysCoeffProvider,
) -> RysTransferTable3c {
    let zero_delta = 0.0_f64;
    let rc = coeffs.coeffs_for(
        root,
        &p_center_axis,
        &c_center_axis,
        &a_center_axis,
        &b_center_axis,
        &c_center_axis,
        &c_center_axis,
        alpha,
        beta,
        gamma,
        &zero_delta,
        rho,
    );

    let mut table = RysTransferTable3c::new(ni_max, nj_max, nk_max);

    table.set(0, 0, 0, 1.0);
    if ni_max >= 1 {
        table.set(1, 0, 0, rc.c00);
        let mut g_nm1 = 1.0_f64;
        let mut g_n = rc.c00;
        for n in 1..ni_max {
            let cur = (n as f64) * rc.b10 * g_nm1 + rc.c00 * g_n;
            table.set(n + 1, 0, 0, cur);
            g_nm1 = g_n;
            g_n = cur;
        }
    }

    if nk_max >= 1 {
        table.set(0, 0, 1, rc.c00p);
        let mut g_mn1 = 1.0_f64;
        let mut g_m = rc.c00p;
        for m in 1..nk_max {
            let cur = (m as f64) * rc.b01p * g_mn1 + rc.c00p * g_m;
            table.set(0, 0, m + 1, cur);
            g_mn1 = g_m;
            g_m = cur;
        }
    }

    if nk_max >= 1 {
        for n in 0..=ni_max {
            let mut val = rc.c00p * table.get(n, 0, 0);
            if n > 0 {
                val += (n as f64) * rc.b00 * table.get(n - 1, 0, 0);
            }
            table.set(n, 0, 1, val);
        }
    }

    if ni_max >= 1 {
        for m in 0..=nk_max {
            let mut val = rc.c00 * table.get(0, 0, m);
            if m > 0 {
                val += (m as f64) * rc.b00 * table.get(0, 0, m - 1);
            }
            table.set(1, 0, m, val);
        }
    }

    for n in 1..=ni_max {
        for m in 1..nk_max {
            let val = (m as f64) * rc.b01p * table.get(n, 0, m - 1)
                + (n as f64) * rc.b00 * table.get(n - 1, 0, m)
                + rc.c00p * table.get(n, 0, m);
            table.set(n, 0, m + 1, val);
        }
    }

    let xi_minus_xj = a_center_axis - b_center_axis;
    if xi_minus_xj.abs() <= 1.0e-18 {
        for nk in 0..=nk_max {
            for nj in 1..=nj_max {
                let i_upper = ni_max - nj;
                for ni in 0..=i_upper {
                    let val = table.get(ni + nj, 0, nk);
                    table.set(ni, nj, nk, val);
                }
            }
        }
    } else {
        for nk in 0..=nk_max {
            for nj in 1..=nj_max {
                let i_upper = ni_max - nj;
                for ni in 0..=i_upper {
                    let val =
                        table.get(ni + 1, nj - 1, nk) + xi_minus_xj * table.get(ni, nj - 1, nk);
                    table.set(ni, nj, nk, val);
                }
            }
        }
    }
    table
}

#[derive(Clone, Copy)]
struct Rint3cBlockEntry {
    data_idx: usize,
    left_ang: [u32; 3],
    right_ang: [u32; 3],
    aux_ang: [u32; 3],
}

fn build_3c_block_entries(
    left: &RintShell,
    right: &RintShell,
    aux: &RintShell,
) -> Vec<Rint3cBlockEntry> {
    let pair_rows = left.ao_len * right.ao_len;
    let mut entries = Vec::with_capacity(pair_rows * aux.ao_len);
    for p in 0..aux.ao_len {
        let aux_ang = aux.cart_components[p];
        let col_offset = p * pair_rows;
        for j in 0..right.ao_len {
            let right_ang = right.cart_components[j];
            let row_offset = col_offset + j * left.ao_len;
            for i in 0..left.ao_len {
                entries.push(Rint3cBlockEntry {
                    data_idx: row_offset + i,
                    left_ang: left.cart_components[i],
                    right_ang,
                    aux_ang,
                });
            }
        }
    }
    entries
}

#[derive(Clone, Copy)]
struct Rint4cBlockEntry {
    data_idx: usize,
    x_idx: usize,
    y_idx: usize,
    z_idx: usize,
}

#[inline(always)]
fn rys_transfer_table_4c_idx(
    ni: u32,
    nj: u32,
    nk: u32,
    nl: u32,
    nj_dim: usize,
    nk_dim: usize,
    nl_dim: usize,
) -> usize {
    (((ni as usize * nj_dim + nj as usize) * nk_dim + nk as usize) * nl_dim) + nl as usize
}

fn build_4c_block_entries(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
) -> Vec<Rint4cBlockEntry> {
    let left_rows = a.ao_len * b.ao_len;
    let right_cols = c.ao_len * d.ao_len;
    let mut entries = Vec::with_capacity(left_rows * right_cols);
    build_4c_block_entries_into(a, b, c, d, &mut entries);
    entries
}

fn build_4c_block_entries_into(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
    entries: &mut Vec<Rint4cBlockEntry>,
) {
    entries.clear();
    let left_rows = a.ao_len * b.ao_len;
    let right_cols = c.ao_len * d.ao_len;
    entries.reserve(left_rows * right_cols);
    let nj_dim = (b.shell.ang_type + 1) as usize;
    let nk_dim = (c.shell.ang_type + d.shell.ang_type + 1) as usize;
    let nl_dim = (d.shell.ang_type + 1) as usize;
    for l in 0..d.ao_len {
        let d_ang = d.cart_components[l];
        for k in 0..c.ao_len {
            let c_ang = c.cart_components[k];
            let col_offset = (l * c.ao_len + k) * left_rows;
            for j in 0..b.ao_len {
                let b_ang = b.cart_components[j];
                let row_offset = col_offset + j * a.ao_len;
                for i in 0..a.ao_len {
                    let a_ang = a.cart_components[i];
                    entries.push(Rint4cBlockEntry {
                        data_idx: row_offset + i,
                        x_idx: rys_transfer_table_4c_idx(
                            a_ang[0], b_ang[0], c_ang[0], d_ang[0], nj_dim, nk_dim, nl_dim,
                        ),
                        y_idx: rys_transfer_table_4c_idx(
                            a_ang[1], b_ang[1], c_ang[1], d_ang[1], nj_dim, nk_dim, nl_dim,
                        ),
                        z_idx: rys_transfer_table_4c_idx(
                            a_ang[2], b_ang[2], c_ang[2], d_ang[2], nj_dim, nk_dim, nl_dim,
                        ),
                    });
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Rint4cContractedBlockEntry {
    a_cart: usize,
    b_cart: usize,
    c_cart: usize,
    d_cart: usize,
    x_idx: usize,
    y_idx: usize,
    z_idx: usize,
}

#[derive(Clone, Copy)]
struct TotalAng2ContractedTarget4c {
    a_cart: usize,
    b_cart: usize,
    c_cart: usize,
    d_cart: usize,
    kind: u8,
    ang: [u32; 3],
    left_axis: usize,
    right_axis: usize,
}

fn build_4c_contracted_block_entries_into(
    a: &RintContractedShell,
    b: &RintContractedShell,
    c: &RintContractedShell,
    d: &RintContractedShell,
    entries: &mut Vec<Rint4cContractedBlockEntry>,
) {
    entries.clear();
    entries.reserve(a.cart_len * b.cart_len * c.cart_len * d.cart_len);
    let nj_dim = (b.l + 1) as usize;
    let nk_dim = (c.l + d.l + 1) as usize;
    let nl_dim = (d.l + 1) as usize;
    for l in 0..d.cart_len {
        let d_ang = d.cart_components[l];
        for k in 0..c.cart_len {
            let c_ang = c.cart_components[k];
            for j in 0..b.cart_len {
                let b_ang = b.cart_components[j];
                for i in 0..a.cart_len {
                    let a_ang = a.cart_components[i];
                    entries.push(Rint4cContractedBlockEntry {
                        a_cart: i,
                        b_cart: j,
                        c_cart: k,
                        d_cart: l,
                        x_idx: rys_transfer_table_4c_idx(
                            a_ang[0], b_ang[0], c_ang[0], d_ang[0], nj_dim, nk_dim, nl_dim,
                        ),
                        y_idx: rys_transfer_table_4c_idx(
                            a_ang[1], b_ang[1], c_ang[1], d_ang[1], nj_dim, nk_dim, nl_dim,
                        ),
                        z_idx: rys_transfer_table_4c_idx(
                            a_ang[2], b_ang[2], c_ang[2], d_ang[2], nj_dim, nk_dim, nl_dim,
                        ),
                    });
                }
            }
        }
    }
}

fn build_total_ang2_contracted_targets_4c(
    a: &RintContractedShell,
    b: &RintContractedShell,
    c: &RintContractedShell,
    d: &RintContractedShell,
) -> Option<([TotalAng2ContractedTarget4c; 16], usize)> {
    let mut targets = [TotalAng2ContractedTarget4c {
        a_cart: 0,
        b_cart: 0,
        c_cart: 0,
        d_cart: 0,
        kind: 0,
        ang: [0; 3],
        left_axis: 0,
        right_axis: 0,
    }; 16];
    let mut target_len = 0_usize;
    let mut push_target = |target: TotalAng2ContractedTarget4c| -> Option<()> {
        if target_len >= targets.len() {
            return None;
        }
        targets[target_len] = target;
        target_len += 1;
        Some(())
    };

    if a.l == 2 {
        for i in 0..a.cart_len {
            push_target(TotalAng2ContractedTarget4c {
                a_cart: i,
                b_cart: 0,
                c_cart: 0,
                d_cart: 0,
                kind: 0,
                ang: a.cart_components[i],
                left_axis: 0,
                right_axis: 0,
            })?;
        }
    } else if b.l == 2 {
        for j in 0..b.cart_len {
            push_target(TotalAng2ContractedTarget4c {
                a_cart: 0,
                b_cart: j,
                c_cart: 0,
                d_cart: 0,
                kind: 1,
                ang: b.cart_components[j],
                left_axis: 0,
                right_axis: 0,
            })?;
        }
    } else if c.l == 2 {
        for k in 0..c.cart_len {
            push_target(TotalAng2ContractedTarget4c {
                a_cart: 0,
                b_cart: 0,
                c_cart: k,
                d_cart: 0,
                kind: 2,
                ang: c.cart_components[k],
                left_axis: 0,
                right_axis: 0,
            })?;
        }
    } else if d.l == 2 {
        for l in 0..d.cart_len {
            push_target(TotalAng2ContractedTarget4c {
                a_cart: 0,
                b_cart: 0,
                c_cart: 0,
                d_cart: l,
                kind: 3,
                ang: d.cart_components[l],
                left_axis: 0,
                right_axis: 0,
            })?;
        }
    } else if a.l == 1 && b.l == 1 {
        for j in 0..b.cart_len {
            let b_axis = single_p_axis(b.cart_components[j]);
            for i in 0..a.cart_len {
                push_target(TotalAng2ContractedTarget4c {
                    a_cart: i,
                    b_cart: j,
                    c_cart: 0,
                    d_cart: 0,
                    kind: 4,
                    ang: [0; 3],
                    left_axis: single_p_axis(a.cart_components[i]),
                    right_axis: b_axis,
                })?;
            }
        }
    } else if a.l == 1 && c.l == 1 {
        for k in 0..c.cart_len {
            let c_axis = single_p_axis(c.cart_components[k]);
            for i in 0..a.cart_len {
                push_target(TotalAng2ContractedTarget4c {
                    a_cart: i,
                    b_cart: 0,
                    c_cart: k,
                    d_cart: 0,
                    kind: 5,
                    ang: [0; 3],
                    left_axis: single_p_axis(a.cart_components[i]),
                    right_axis: c_axis,
                })?;
            }
        }
    } else if a.l == 1 && d.l == 1 {
        for l in 0..d.cart_len {
            let d_axis = single_p_axis(d.cart_components[l]);
            for i in 0..a.cart_len {
                push_target(TotalAng2ContractedTarget4c {
                    a_cart: i,
                    b_cart: 0,
                    c_cart: 0,
                    d_cart: l,
                    kind: 6,
                    ang: [0; 3],
                    left_axis: single_p_axis(a.cart_components[i]),
                    right_axis: d_axis,
                })?;
            }
        }
    } else if b.l == 1 && c.l == 1 {
        for k in 0..c.cart_len {
            let c_axis = single_p_axis(c.cart_components[k]);
            for j in 0..b.cart_len {
                push_target(TotalAng2ContractedTarget4c {
                    a_cart: 0,
                    b_cart: j,
                    c_cart: k,
                    d_cart: 0,
                    kind: 7,
                    ang: [0; 3],
                    left_axis: single_p_axis(b.cart_components[j]),
                    right_axis: c_axis,
                })?;
            }
        }
    } else if b.l == 1 && d.l == 1 {
        for l in 0..d.cart_len {
            let d_axis = single_p_axis(d.cart_components[l]);
            for j in 0..b.cart_len {
                push_target(TotalAng2ContractedTarget4c {
                    a_cart: 0,
                    b_cart: j,
                    c_cart: 0,
                    d_cart: l,
                    kind: 8,
                    ang: [0; 3],
                    left_axis: single_p_axis(b.cart_components[j]),
                    right_axis: d_axis,
                })?;
            }
        }
    } else {
        for l in 0..d.cart_len {
            let d_axis = single_p_axis(d.cart_components[l]);
            for k in 0..c.cart_len {
                push_target(TotalAng2ContractedTarget4c {
                    a_cart: 0,
                    b_cart: 0,
                    c_cart: k,
                    d_cart: l,
                    kind: 9,
                    ang: [0; 3],
                    left_axis: single_p_axis(c.cart_components[k]),
                    right_axis: d_axis,
                })?;
            }
        }
    }
    Some((targets, target_len))
}

struct RysTransferTable4c {
    data: Vec<f64>,
    nj_dim: usize,
    nk_dim: usize,
    nl_dim: usize,
}

impl RysTransferTable4c {
    fn new(ni_max: u32, nj_max: u32, nk_max: u32, nl_max: u32) -> Self {
        let data_len = (ni_max as usize + 1)
            * (nj_max as usize + 1)
            * (nk_max as usize + 1)
            * (nl_max as usize + 1);
        Self {
            data: vec![0.0; data_len],
            nj_dim: (nj_max + 1) as usize,
            nk_dim: (nk_max + 1) as usize,
            nl_dim: (nl_max + 1) as usize,
        }
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.data.fill(0.0);
    }

    #[inline(always)]
    fn idx(&self, ni: u32, nj: u32, nk: u32, nl: u32) -> usize {
        rys_transfer_table_4c_idx(ni, nj, nk, nl, self.nj_dim, self.nk_dim, self.nl_dim)
    }

    #[inline(always)]
    fn get(&self, ni: u32, nj: u32, nk: u32, nl: u32) -> f64 {
        let idx = self.idx(ni, nj, nk, nl);
        debug_assert!(idx < self.data.len());
        unsafe { *self.data.get_unchecked(idx) }
    }

    #[inline(always)]
    fn set(&mut self, ni: u32, nj: u32, nk: u32, nl: u32, value: f64) {
        let idx = self.idx(ni, nj, nk, nl);
        debug_assert!(idx < self.data.len());
        unsafe {
            *self.data.get_unchecked_mut(idx) = value;
        }
    }
}

struct RysSeedPanel2d {
    data: Vec<f64>,
    nk_dim: usize,
}

impl RysSeedPanel2d {
    fn new(ni_max: u32, nk_max: u32) -> Self {
        Self {
            data: vec![0.0; (ni_max as usize + 1) * (nk_max as usize + 1)],
            nk_dim: (nk_max + 1) as usize,
        }
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.data.fill(0.0);
    }

    #[inline(always)]
    fn idx(&self, ni: u32, nk: u32) -> usize {
        ni as usize * self.nk_dim + nk as usize
    }

    #[inline(always)]
    fn get(&self, ni: u32, nk: u32) -> f64 {
        let idx = self.idx(ni, nk);
        debug_assert!(idx < self.data.len());
        unsafe { *self.data.get_unchecked(idx) }
    }

    #[inline(always)]
    fn set(&mut self, ni: u32, nk: u32, value: f64) {
        let idx = self.idx(ni, nk);
        debug_assert!(idx < self.data.len());
        unsafe {
            *self.data.get_unchecked_mut(idx) = value;
        }
    }
}

struct RysTransferWorkspace4c {
    table_x: RysTransferTable4c,
    table_y: RysTransferTable4c,
    table_z: RysTransferTable4c,
    seed_panel: RysSeedPanel2d,
}

impl RysTransferWorkspace4c {
    fn new(ni_max: u32, nj_max: u32, nk_max: u32, nl_max: u32) -> Self {
        Self {
            table_x: RysTransferTable4c::new(ni_max, nj_max, nk_max, nl_max),
            table_y: RysTransferTable4c::new(ni_max, nj_max, nk_max, nl_max),
            table_z: RysTransferTable4c::new(ni_max, nj_max, nk_max, nl_max),
            seed_panel: RysSeedPanel2d::new(ni_max, nk_max),
        }
    }
}

#[derive(Clone, Copy)]
struct RintPrimitivePair4c {
    exp_sum: f64,
    scaled_coeff_over_exp_sum: f64,
    center: [f64; 3],
}

fn build_primitive_pairs_4c(
    left: &RintShell,
    right: &RintShell,
    left_coeffs: &[f64],
    right_coeffs: &[f64],
    rab2: f64,
) -> Vec<RintPrimitivePair4c> {
    let mut pairs = Vec::with_capacity(left.shell.exponents.len() * right.shell.exponents.len());
    for (&alpha, &left_coeff) in left.shell.exponents.iter().zip(left_coeffs.iter()) {
        for (&beta, &right_coeff) in right.shell.exponents.iter().zip(right_coeffs.iter()) {
            let p_sum = alpha + beta;
            if p_sum <= 0.0 {
                continue;
            }
            let coeff = left_coeff * right_coeff;
            if coeff == 0.0 {
                continue;
            }
            let p_fac = alpha * beta / p_sum;
            let exp_factor = (-p_fac * rab2).exp();
            if !exp_factor.is_finite() {
                continue;
            }
            let scaled_coeff = coeff * exp_factor;
            pairs.push(RintPrimitivePair4c {
                exp_sum: p_sum,
                scaled_coeff_over_exp_sum: scaled_coeff / p_sum,
                center: gaussian_product_center(alpha, beta, &left.center, &right.center),
            });
        }
    }
    pairs
}

#[derive(Clone)]
struct RintContractedPrimitivePair4c {
    exp_sum: f64,
    scaled_exp_over_exp_sum: f64,
    center: [f64; 3],
    coeff_products: Vec<f64>,
}

fn build_contracted_primitive_pairs_4c(
    left: &RintContractedShell,
    right: &RintContractedShell,
    rab2: f64,
) -> Vec<RintContractedPrimitivePair4c> {
    let left_nctr = left.nctr();
    let right_nctr = right.nctr();
    let mut pairs = Vec::with_capacity(left.exponents.len() * right.exponents.len());
    for (alpha_idx, &alpha) in left.exponents.iter().enumerate() {
        for (beta_idx, &beta) in right.exponents.iter().enumerate() {
            let p_sum = alpha + beta;
            if p_sum <= 0.0 {
                continue;
            }
            let p_fac = alpha * beta / p_sum;
            let exp_factor = (-p_fac * rab2).exp();
            if !exp_factor.is_finite() {
                continue;
            }
            let mut coeff_products = Vec::with_capacity(left_nctr * right_nctr);
            let mut any_nonzero = false;
            for right_col in right.coeff_columns.iter() {
                for left_col in left.coeff_columns.iter() {
                    let coeff = left_col[alpha_idx] * right_col[beta_idx];
                    any_nonzero |= coeff != 0.0;
                    coeff_products.push(coeff);
                }
            }
            if !any_nonzero {
                continue;
            }
            pairs.push(RintContractedPrimitivePair4c {
                exp_sum: p_sum,
                scaled_exp_over_exp_sum: exp_factor / p_sum,
                center: gaussian_product_center(alpha, beta, &left.center, &right.center),
                coeff_products,
            });
        }
    }
    pairs
}

#[derive(Clone, Copy)]
struct RysAxisCoeffs4c {
    b10: f64,
    b01p: f64,
    b00: f64,
    c00: f64,
    c00p: f64,
}

#[derive(Clone, Copy)]
struct RysRootScalars4c {
    b10: f64,
    b01p: f64,
    b00: f64,
    q_over_pq: f64,
    p_over_pq: f64,
}

#[derive(Clone, Copy)]
struct PTripletTarget4c {
    data_idx: usize,
    axes: [usize; 3],
    ids: [usize; 3],
}

#[derive(Clone, Copy)]
struct PQuartetTarget4c {
    data_idx: usize,
    axes: [usize; 4],
}

#[derive(Clone, Copy)]
struct TotalAng2Target4c {
    data_idx: usize,
    kind: u8,
    ang: [u32; 3],
    left_axis: usize,
    right_axis: usize,
}

enum Rint4cCachedBlockPlan {
    Ssss,
    TotalAng1 {
        p_shell_id: usize,
        targets: [(usize, usize); 3],
        target_len: usize,
    },
    TotalAng2 {
        targets: [TotalAng2Target4c; 16],
        target_len: usize,
    },
    ThreePOneS {
        targets: [PTripletTarget4c; 81],
        target_len: usize,
    },
    FourP {
        targets: [PQuartetTarget4c; 81],
        target_len: usize,
    },
    Generic(Vec<Rint4cBlockEntry>),
}

#[inline(always)]
fn rys_root_scalars_4c(root: f64, p_sum: f64, q_sum: f64, p_sum_q: f64) -> RysRootScalars4c {
    let inv_p_sum_q = p_sum_q.recip();
    let half_inv_p = 0.5 / p_sum;
    let half_inv_q = 0.5 / q_sum;
    RysRootScalars4c {
        b10: half_inv_p - q_sum * half_inv_p * inv_p_sum_q * root,
        b01p: half_inv_q - p_sum * half_inv_q * inv_p_sum_q * root,
        b00: 0.5 * inv_p_sum_q * root,
        q_over_pq: q_sum * inv_p_sum_q,
        p_over_pq: p_sum * inv_p_sum_q,
    }
}

#[inline(always)]
fn rys_axis_coeffs_4c_from_scalars(
    scalars: RysRootScalars4c,
    root: f64,
    p_center: f64,
    q_center: f64,
    a_center: f64,
    c_center: f64,
) -> RysAxisCoeffs4c {
    let q_minus_p = q_center - p_center;
    RysAxisCoeffs4c {
        b10: scalars.b10,
        b01p: scalars.b01p,
        b00: scalars.b00,
        c00: (p_center - a_center) + scalars.q_over_pq * q_minus_p * root,
        c00p: (q_center - c_center) - scalars.p_over_pq * q_minus_p * root,
    }
}

#[inline(always)]
fn rys_axis_coeffs_4c(
    root: f64,
    p_sum: f64,
    q_sum: f64,
    p_sum_q: f64,
    p_center: f64,
    q_center: f64,
    a_center: f64,
    c_center: f64,
) -> RysAxisCoeffs4c {
    rys_axis_coeffs_4c_from_scalars(
        rys_root_scalars_4c(root, p_sum, q_sum, p_sum_q),
        root,
        p_center,
        q_center,
        a_center,
        c_center,
    )
}

#[inline(always)]
fn single_p_axis(ang: [u32; 3]) -> usize {
    if ang[0] == 1 {
        0
    } else if ang[1] == 1 {
        1
    } else {
        2
    }
}

fn build_three_p_one_s_targets_4c(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
) -> Option<([PTripletTarget4c; 81], usize)> {
    let mut targets = [PTripletTarget4c {
        data_idx: 0,
        axes: [0; 3],
        ids: [0; 3],
    }; 81];
    let mut target_len = 0_usize;
    let left_rows = a.ao_len * b.ao_len;
    for l in 0..d.ao_len {
        let d_ang = d.cart_components[l];
        let col_base = l * c.ao_len * left_rows;
        for k in 0..c.ao_len {
            let c_ang = c.cart_components[k];
            let col_offset = col_base + k * left_rows;
            for j in 0..b.ao_len {
                let b_ang = b.cart_components[j];
                let row_offset = col_offset + j * a.ao_len;
                for i in 0..a.ao_len {
                    if target_len >= targets.len() {
                        return None;
                    }
                    let a_ang = a.cart_components[i];
                    let mut axes = [0_usize; 3];
                    let mut ids = [0_usize; 3];
                    let mut n = 0_usize;
                    if a.shell.ang_type == 1 {
                        axes[n] = single_p_axis(a_ang);
                        ids[n] = 0;
                        n += 1;
                    }
                    if b.shell.ang_type == 1 {
                        axes[n] = single_p_axis(b_ang);
                        ids[n] = 1;
                        n += 1;
                    }
                    if c.shell.ang_type == 1 {
                        axes[n] = single_p_axis(c_ang);
                        ids[n] = 2;
                        n += 1;
                    }
                    if d.shell.ang_type == 1 {
                        axes[n] = single_p_axis(d_ang);
                        ids[n] = 3;
                        n += 1;
                    }
                    debug_assert_eq!(n, 3);
                    targets[target_len] = PTripletTarget4c {
                        data_idx: row_offset + i,
                        axes,
                        ids,
                    };
                    target_len += 1;
                }
            }
        }
    }
    Some((targets, target_len))
}

fn build_four_p_targets_4c(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
) -> Option<([PQuartetTarget4c; 81], usize)> {
    let mut targets = [PQuartetTarget4c {
        data_idx: 0,
        axes: [0; 4],
    }; 81];
    let mut target_len = 0_usize;
    let left_rows = a.ao_len * b.ao_len;
    for l in 0..d.ao_len {
        let d_axis = single_p_axis(d.cart_components[l]);
        let col_base = l * c.ao_len * left_rows;
        for k in 0..c.ao_len {
            let c_axis = single_p_axis(c.cart_components[k]);
            let col_offset = col_base + k * left_rows;
            for j in 0..b.ao_len {
                let b_axis = single_p_axis(b.cart_components[j]);
                let row_offset = col_offset + j * a.ao_len;
                for i in 0..a.ao_len {
                    if target_len >= targets.len() {
                        return None;
                    }
                    targets[target_len] = PQuartetTarget4c {
                        data_idx: row_offset + i,
                        axes: [single_p_axis(a.cart_components[i]), b_axis, c_axis, d_axis],
                    };
                    target_len += 1;
                }
            }
        }
    }
    Some((targets, target_len))
}

fn build_total_ang2_targets_4c(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
) -> Option<([TotalAng2Target4c; 16], usize)> {
    let mut targets = [TotalAng2Target4c {
        data_idx: 0,
        kind: 0,
        ang: [0; 3],
        left_axis: 0,
        right_axis: 0,
    }; 16];
    let mut target_len = 0_usize;
    let left_rows = a.ao_len * b.ao_len;
    let mut push_target = |target: TotalAng2Target4c| -> Option<()> {
        if target_len >= targets.len() {
            return None;
        }
        targets[target_len] = target;
        target_len += 1;
        Some(())
    };

    if a.shell.ang_type == 2 {
        for i in 0..a.ao_len {
            push_target(TotalAng2Target4c {
                data_idx: i,
                kind: 0,
                ang: a.cart_components[i],
                left_axis: 0,
                right_axis: 0,
            })?;
        }
    } else if b.shell.ang_type == 2 {
        for j in 0..b.ao_len {
            push_target(TotalAng2Target4c {
                data_idx: j * a.ao_len,
                kind: 1,
                ang: b.cart_components[j],
                left_axis: 0,
                right_axis: 0,
            })?;
        }
    } else if c.shell.ang_type == 2 {
        for k in 0..c.ao_len {
            push_target(TotalAng2Target4c {
                data_idx: k * left_rows,
                kind: 2,
                ang: c.cart_components[k],
                left_axis: 0,
                right_axis: 0,
            })?;
        }
    } else if d.shell.ang_type == 2 {
        for l in 0..d.ao_len {
            push_target(TotalAng2Target4c {
                data_idx: l * c.ao_len * left_rows,
                kind: 3,
                ang: d.cart_components[l],
                left_axis: 0,
                right_axis: 0,
            })?;
        }
    } else if a.shell.ang_type == 1 && b.shell.ang_type == 1 {
        for j in 0..b.ao_len {
            let b_axis = single_p_axis(b.cart_components[j]);
            let row_offset = j * a.ao_len;
            for i in 0..a.ao_len {
                push_target(TotalAng2Target4c {
                    data_idx: row_offset + i,
                    kind: 4,
                    ang: [0; 3],
                    left_axis: single_p_axis(a.cart_components[i]),
                    right_axis: b_axis,
                })?;
            }
        }
    } else if a.shell.ang_type == 1 && c.shell.ang_type == 1 {
        for k in 0..c.ao_len {
            let c_axis = single_p_axis(c.cart_components[k]);
            let col_offset = k * left_rows;
            for i in 0..a.ao_len {
                push_target(TotalAng2Target4c {
                    data_idx: col_offset + i,
                    kind: 5,
                    ang: [0; 3],
                    left_axis: single_p_axis(a.cart_components[i]),
                    right_axis: c_axis,
                })?;
            }
        }
    } else if a.shell.ang_type == 1 && d.shell.ang_type == 1 {
        for l in 0..d.ao_len {
            let d_axis = single_p_axis(d.cart_components[l]);
            let col_offset = l * c.ao_len * left_rows;
            for i in 0..a.ao_len {
                push_target(TotalAng2Target4c {
                    data_idx: col_offset + i,
                    kind: 6,
                    ang: [0; 3],
                    left_axis: single_p_axis(a.cart_components[i]),
                    right_axis: d_axis,
                })?;
            }
        }
    } else if b.shell.ang_type == 1 && c.shell.ang_type == 1 {
        for k in 0..c.ao_len {
            let c_axis = single_p_axis(c.cart_components[k]);
            let col_offset = k * left_rows;
            for j in 0..b.ao_len {
                push_target(TotalAng2Target4c {
                    data_idx: col_offset + j * a.ao_len,
                    kind: 7,
                    ang: [0; 3],
                    left_axis: single_p_axis(b.cart_components[j]),
                    right_axis: c_axis,
                })?;
            }
        }
    } else if b.shell.ang_type == 1 && d.shell.ang_type == 1 {
        for l in 0..d.ao_len {
            let d_axis = single_p_axis(d.cart_components[l]);
            let col_offset = l * c.ao_len * left_rows;
            for j in 0..b.ao_len {
                push_target(TotalAng2Target4c {
                    data_idx: col_offset + j * a.ao_len,
                    kind: 8,
                    ang: [0; 3],
                    left_axis: single_p_axis(b.cart_components[j]),
                    right_axis: d_axis,
                })?;
            }
        }
    } else {
        for l in 0..d.ao_len {
            let d_axis = single_p_axis(d.cart_components[l]);
            let col_base = l * c.ao_len * left_rows;
            for k in 0..c.ao_len {
                push_target(TotalAng2Target4c {
                    data_idx: col_base + k * left_rows,
                    kind: 9,
                    ang: [0; 3],
                    left_axis: single_p_axis(c.cart_components[k]),
                    right_axis: d_axis,
                })?;
            }
        }
    }
    Some((targets, target_len))
}

fn build_total_ang1_targets_4c(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
) -> (usize, [(usize, usize); 3], usize) {
    let left_rows = a.ao_len * b.ao_len;
    let p_shell_id = if a.shell.ang_type == 1 {
        0
    } else if b.shell.ang_type == 1 {
        1
    } else if c.shell.ang_type == 1 {
        2
    } else {
        3
    };
    let mut p_targets = [(0_usize, 0_usize); 3];
    let p_target_len = match p_shell_id {
        0 => {
            for (target, (i, &ang)) in p_targets
                .iter_mut()
                .zip(a.cart_components.iter().enumerate())
            {
                *target = (i, single_p_axis(ang));
            }
            a.ao_len
        }
        1 => {
            for (target, (j, &ang)) in p_targets
                .iter_mut()
                .zip(b.cart_components.iter().enumerate())
            {
                *target = (j * a.ao_len, single_p_axis(ang));
            }
            b.ao_len
        }
        2 => {
            for (target, (k, &ang)) in p_targets
                .iter_mut()
                .zip(c.cart_components.iter().enumerate())
            {
                *target = (k * left_rows, single_p_axis(ang));
            }
            c.ao_len
        }
        _ => {
            for (target, (l, &ang)) in p_targets
                .iter_mut()
                .zip(d.cart_components.iter().enumerate())
            {
                *target = (l * c.ao_len * left_rows, single_p_axis(ang));
            }
            d.ao_len
        }
    };
    (p_shell_id, p_targets, p_target_len)
}

#[derive(Clone, Copy)]
struct RysTotalAng2Values {
    a1: [f64; 3],
    b1: [f64; 3],
    c1: [f64; 3],
    d1: [f64; 3],
    a2: [f64; 3],
    b2: [f64; 3],
    c2: [f64; 3],
    d2: [f64; 3],
    ab_same: [f64; 3],
    ac_same: [f64; 3],
    ad_same: [f64; 3],
    bc_same: [f64; 3],
    bd_same: [f64; 3],
    cd_same: [f64; 3],
}

#[inline(always)]
fn d_component_value(ang: [u32; 3], first: &[f64; 3], second_same_axis: &[f64; 3]) -> f64 {
    if ang[0] == 2 {
        second_same_axis[0]
    } else if ang[1] == 2 {
        second_same_axis[1]
    } else if ang[2] == 2 {
        second_same_axis[2]
    } else if ang[0] == 1 && ang[1] == 1 {
        first[0] * first[1]
    } else if ang[0] == 1 && ang[2] == 1 {
        first[0] * first[2]
    } else {
        first[1] * first[2]
    }
}

#[inline(always)]
fn p_pair_component_value(
    left_axis: usize,
    right_axis: usize,
    left_first: &[f64; 3],
    right_first: &[f64; 3],
    same_axis: &[f64; 3],
) -> f64 {
    if left_axis == right_axis {
        same_axis[left_axis]
    } else {
        left_first[left_axis] * right_first[right_axis]
    }
}

#[inline(always)]
fn total_ang2_target_value(target: TotalAng2Target4c, values: &RysTotalAng2Values) -> f64 {
    match target.kind {
        0 => d_component_value(target.ang, &values.a1, &values.a2),
        1 => d_component_value(target.ang, &values.b1, &values.b2),
        2 => d_component_value(target.ang, &values.c1, &values.c2),
        3 => d_component_value(target.ang, &values.d1, &values.d2),
        4 => p_pair_component_value(
            target.left_axis,
            target.right_axis,
            &values.a1,
            &values.b1,
            &values.ab_same,
        ),
        5 => p_pair_component_value(
            target.left_axis,
            target.right_axis,
            &values.a1,
            &values.c1,
            &values.ac_same,
        ),
        6 => p_pair_component_value(
            target.left_axis,
            target.right_axis,
            &values.a1,
            &values.d1,
            &values.ad_same,
        ),
        7 => p_pair_component_value(
            target.left_axis,
            target.right_axis,
            &values.b1,
            &values.c1,
            &values.bc_same,
        ),
        8 => p_pair_component_value(
            target.left_axis,
            target.right_axis,
            &values.b1,
            &values.d1,
            &values.bd_same,
        ),
        _ => p_pair_component_value(
            target.left_axis,
            target.right_axis,
            &values.c1,
            &values.d1,
            &values.cd_same,
        ),
    }
}

#[inline(always)]
fn p_pair_covariance(id_a: usize, id_b: usize, scalars: RysRootScalars4c) -> f64 {
    match (id_a <= 1, id_b <= 1) {
        (true, true) => scalars.b10,
        (false, false) => scalars.b01p,
        _ => scalars.b00,
    }
}

#[inline(always)]
fn p_pair_axis_value(
    axis: usize,
    id_a: usize,
    id_b: usize,
    first: &[[f64; 3]; 4],
    scalars: RysRootScalars4c,
) -> f64 {
    first[id_a][axis] * first[id_b][axis] + p_pair_covariance(id_a, id_b, scalars)
}

#[inline(always)]
fn p_triplet_component_value(
    axes: [usize; 3],
    ids: [usize; 3],
    first: &[[f64; 3]; 4],
    scalars: RysRootScalars4c,
) -> f64 {
    let m0 = first[ids[0]][axes[0]];
    let m1 = first[ids[1]][axes[1]];
    let m2 = first[ids[2]][axes[2]];
    if axes[0] == axes[1] && axes[1] == axes[2] {
        m0 * m1 * m2
            + p_pair_covariance(ids[0], ids[1], scalars) * m2
            + p_pair_covariance(ids[0], ids[2], scalars) * m1
            + p_pair_covariance(ids[1], ids[2], scalars) * m0
    } else if axes[0] == axes[1] {
        p_pair_axis_value(axes[0], ids[0], ids[1], first, scalars) * m2
    } else if axes[0] == axes[2] {
        p_pair_axis_value(axes[0], ids[0], ids[2], first, scalars) * m1
    } else if axes[1] == axes[2] {
        p_pair_axis_value(axes[1], ids[1], ids[2], first, scalars) * m0
    } else {
        m0 * m1 * m2
    }
}

#[inline(always)]
fn p_axis_covariance(
    left_axis: usize,
    right_axis: usize,
    left_id: usize,
    right_id: usize,
    scalars: RysRootScalars4c,
) -> f64 {
    if left_axis == right_axis {
        p_pair_covariance(left_id, right_id, scalars)
    } else {
        0.0
    }
}

#[inline(always)]
fn p_quartet_component_value(
    axes: [usize; 4],
    ids: [usize; 4],
    first: &[[f64; 3]; 4],
    scalars: RysRootScalars4c,
) -> f64 {
    let m0 = first[ids[0]][axes[0]];
    let m1 = first[ids[1]][axes[1]];
    let m2 = first[ids[2]][axes[2]];
    let m3 = first[ids[3]][axes[3]];
    let c01 = p_axis_covariance(axes[0], axes[1], ids[0], ids[1], scalars);
    let c02 = p_axis_covariance(axes[0], axes[2], ids[0], ids[2], scalars);
    let c03 = p_axis_covariance(axes[0], axes[3], ids[0], ids[3], scalars);
    let c12 = p_axis_covariance(axes[1], axes[2], ids[1], ids[2], scalars);
    let c13 = p_axis_covariance(axes[1], axes[3], ids[1], ids[3], scalars);
    let c23 = p_axis_covariance(axes[2], axes[3], ids[2], ids[3], scalars);

    m0 * m1 * m2 * m3
        + c01 * m2 * m3
        + c02 * m1 * m3
        + c03 * m1 * m2
        + c12 * m0 * m3
        + c13 * m0 * m2
        + c23 * m0 * m1
        + c01 * c23
        + c02 * c13
        + c03 * c12
}

#[allow(clippy::too_many_arguments)]
fn rys_total_ang2_values(
    root: f64,
    p_sum: f64,
    q_sum: f64,
    p_sum_q: f64,
    p_center: &[f64; 3],
    q_center: &[f64; 3],
    a_center: &[f64; 3],
    b_center: &[f64; 3],
    c_center: &[f64; 3],
    d_center: &[f64; 3],
) -> RysTotalAng2Values {
    let scalars = rys_root_scalars_4c(root, p_sum, q_sum, p_sum_q);
    let mut values = RysTotalAng2Values {
        a1: [0.0; 3],
        b1: [0.0; 3],
        c1: [0.0; 3],
        d1: [0.0; 3],
        a2: [0.0; 3],
        b2: [0.0; 3],
        c2: [0.0; 3],
        d2: [0.0; 3],
        ab_same: [0.0; 3],
        ac_same: [0.0; 3],
        ad_same: [0.0; 3],
        bc_same: [0.0; 3],
        bd_same: [0.0; 3],
        cd_same: [0.0; 3],
    };

    for axis in 0..3 {
        let rc = rys_axis_coeffs_4c_from_scalars(
            scalars,
            root,
            p_center[axis],
            q_center[axis],
            a_center[axis],
            c_center[axis],
        );
        let ab = a_center[axis] - b_center[axis];
        let cd = c_center[axis] - d_center[axis];
        values.a1[axis] = rc.c00;
        values.b1[axis] = rc.c00 + ab;
        values.c1[axis] = rc.c00p;
        values.d1[axis] = rc.c00p + cd;
        values.a2[axis] = rc.b10 + rc.c00 * rc.c00;
        values.b2[axis] = values.a2[axis] + 2.0 * ab * rc.c00 + ab * ab;
        values.c2[axis] = rc.b01p + rc.c00p * rc.c00p;
        values.d2[axis] = values.c2[axis] + 2.0 * cd * rc.c00p + cd * cd;
        let ac_same = rc.c00 * rc.c00p + rc.b00;
        values.ab_same[axis] = values.a2[axis] + ab * rc.c00;
        values.ac_same[axis] = ac_same;
        values.ad_same[axis] = ac_same + cd * rc.c00;
        values.bc_same[axis] = ac_same + ab * rc.c00p;
        values.bd_same[axis] = ac_same + ab * rc.c00p + cd * values.b1[axis];
        values.cd_same[axis] = values.c2[axis] + cd * rc.c00p;
    }

    values
}
#[allow(clippy::too_many_arguments)]
fn add_primitive_4c_total_ang2_fast(
    block_data: &mut [f64],
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
    primitive_coeff: f64,
    pref: f64,
    t: f64,
    p_sum: f64,
    q_sum: f64,
    p_sum_q: f64,
    p_center: &[f64; 3],
    q_center: &[f64; 3],
) -> bool {
    let mut roots = [0.0_f64; 2];
    let mut weights = [0.0_f64; 2];
    let nroots = rys_roots_weights_r_into(2, t, &mut roots, &mut weights);
    if nroots != 2 {
        return false;
    }
    let left_rows = a.ao_len * b.ao_len;
    for root_idx in 0..2 {
        let values = rys_total_ang2_values(
            roots[root_idx],
            p_sum,
            q_sum,
            p_sum_q,
            p_center,
            q_center,
            &a.center,
            &b.center,
            &c.center,
            &d.center,
        );
        let primitive_scale = primitive_coeff * pref * weights[root_idx];
        if !primitive_scale.is_finite() {
            continue;
        }

        if a.shell.ang_type == 2 {
            for i in 0..a.ao_len {
                block_data[i] += primitive_scale
                    * d_component_value(a.cart_components[i], &values.a1, &values.a2);
            }
        } else if b.shell.ang_type == 2 {
            for j in 0..b.ao_len {
                block_data[j * a.ao_len] += primitive_scale
                    * d_component_value(b.cart_components[j], &values.b1, &values.b2);
            }
        } else if c.shell.ang_type == 2 {
            for k in 0..c.ao_len {
                block_data[k * left_rows] += primitive_scale
                    * d_component_value(c.cart_components[k], &values.c1, &values.c2);
            }
        } else if d.shell.ang_type == 2 {
            for l in 0..d.ao_len {
                block_data[l * c.ao_len * left_rows] += primitive_scale
                    * d_component_value(d.cart_components[l], &values.d1, &values.d2);
            }
        } else if a.shell.ang_type == 1 && b.shell.ang_type == 1 {
            for j in 0..b.ao_len {
                let b_axis = single_p_axis(b.cart_components[j]);
                let row_offset = j * a.ao_len;
                for i in 0..a.ao_len {
                    let a_axis = single_p_axis(a.cart_components[i]);
                    block_data[row_offset + i] += primitive_scale
                        * p_pair_component_value(
                            a_axis,
                            b_axis,
                            &values.a1,
                            &values.b1,
                            &values.ab_same,
                        );
                }
            }
        } else if a.shell.ang_type == 1 && c.shell.ang_type == 1 {
            for k in 0..c.ao_len {
                let c_axis = single_p_axis(c.cart_components[k]);
                let col_offset = k * left_rows;
                for i in 0..a.ao_len {
                    let a_axis = single_p_axis(a.cart_components[i]);
                    block_data[col_offset + i] += primitive_scale
                        * p_pair_component_value(
                            a_axis,
                            c_axis,
                            &values.a1,
                            &values.c1,
                            &values.ac_same,
                        );
                }
            }
        } else if a.shell.ang_type == 1 && d.shell.ang_type == 1 {
            for l in 0..d.ao_len {
                let d_axis = single_p_axis(d.cart_components[l]);
                let col_offset = l * c.ao_len * left_rows;
                for i in 0..a.ao_len {
                    let a_axis = single_p_axis(a.cart_components[i]);
                    block_data[col_offset + i] += primitive_scale
                        * p_pair_component_value(
                            a_axis,
                            d_axis,
                            &values.a1,
                            &values.d1,
                            &values.ad_same,
                        );
                }
            }
        } else if b.shell.ang_type == 1 && c.shell.ang_type == 1 {
            for k in 0..c.ao_len {
                let c_axis = single_p_axis(c.cart_components[k]);
                let col_offset = k * left_rows;
                for j in 0..b.ao_len {
                    let b_axis = single_p_axis(b.cart_components[j]);
                    block_data[col_offset + j * a.ao_len] += primitive_scale
                        * p_pair_component_value(
                            b_axis,
                            c_axis,
                            &values.b1,
                            &values.c1,
                            &values.bc_same,
                        );
                }
            }
        } else if b.shell.ang_type == 1 && d.shell.ang_type == 1 {
            for l in 0..d.ao_len {
                let d_axis = single_p_axis(d.cart_components[l]);
                let col_offset = l * c.ao_len * left_rows;
                for j in 0..b.ao_len {
                    let b_axis = single_p_axis(b.cart_components[j]);
                    block_data[col_offset + j * a.ao_len] += primitive_scale
                        * p_pair_component_value(
                            b_axis,
                            d_axis,
                            &values.b1,
                            &values.d1,
                            &values.bd_same,
                        );
                }
            }
        } else {
            for l in 0..d.ao_len {
                let d_axis = single_p_axis(d.cart_components[l]);
                let col_base = l * c.ao_len * left_rows;
                for k in 0..c.ao_len {
                    let c_axis = single_p_axis(c.cart_components[k]);
                    block_data[col_base + k * left_rows] += primitive_scale
                        * p_pair_component_value(
                            c_axis,
                            d_axis,
                            &values.c1,
                            &values.d1,
                            &values.cd_same,
                        );
                }
            }
        }
    }
    true
}

fn int4c_r_four_p_block_into_data_with_pairs(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
    ab_pairs: &[RintPrimitivePair4c],
    cd_pairs: &[RintPrimitivePair4c],
    block_data: &mut [f64],
) -> bool {
    if a.shell.ang_type != 1
        || b.shell.ang_type != 1
        || c.shell.ang_type != 1
        || d.shell.ang_type != 1
    {
        return false;
    }

    let Some((targets, target_len)) = build_four_p_targets_4c(a, b, c, d) else {
        return false;
    };
    int4c_r_four_p_block_into_data_with_pairs_targets(
        a, b, c, d, ab_pairs, cd_pairs, block_data, &targets, target_len,
    )
}

#[allow(clippy::too_many_arguments)]
fn int4c_r_four_p_block_into_data_with_pairs_targets(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
    ab_pairs: &[RintPrimitivePair4c],
    cd_pairs: &[RintPrimitivePair4c],
    block_data: &mut [f64],
    targets: &[PQuartetTarget4c; 81],
    target_len: usize,
) -> bool {
    let ids = [0_usize, 1_usize, 2_usize, 3_usize];
    for ab_pair in ab_pairs {
        let p_sum = ab_pair.exp_sum;
        let p_center = ab_pair.center;
        for cd_pair in cd_pairs {
            let q_sum = cd_pair.exp_sum;
            let p_sum_q = p_sum + q_sum;
            let pq_mul = p_sum * q_sum;
            debug_assert!(p_sum > 0.0 && q_sum > 0.0 && p_sum_q > 0.0 && pq_mul > 0.0);

            let rho = pq_mul / p_sum_q;
            let t = rho * distance_squared(&p_center, &cd_pair.center);
            let mut roots = [0.0_f64; 3];
            let mut weights = [0.0_f64; 3];
            let nroots = rys_roots_weights_r_into(3, t, &mut roots, &mut weights);
            if nroots != 3 {
                return false;
            }

            let pref = TWO_PI_POW_2P5 / p_sum_q.sqrt();
            let primitive_coeff =
                ab_pair.scaled_coeff_over_exp_sum * cd_pair.scaled_coeff_over_exp_sum;
            for root_idx in 0..3 {
                let root = roots[root_idx];
                let scalars = rys_root_scalars_4c(root, p_sum, q_sum, p_sum_q);
                let mut first = [[0.0_f64; 3]; 4];
                for axis in 0..3 {
                    let rc = rys_axis_coeffs_4c_from_scalars(
                        scalars,
                        root,
                        p_center[axis],
                        cd_pair.center[axis],
                        a.center[axis],
                        c.center[axis],
                    );
                    first[0][axis] = rc.c00;
                    first[1][axis] = rc.c00 + a.center[axis] - b.center[axis];
                    first[2][axis] = rc.c00p;
                    first[3][axis] = rc.c00p + c.center[axis] - d.center[axis];
                }

                let primitive_scale = primitive_coeff * pref * weights[root_idx];
                if !primitive_scale.is_finite() {
                    continue;
                }
                for target in targets[..target_len].iter() {
                    let value = p_quartet_component_value(target.axes, ids, &first, scalars);
                    let term = primitive_scale * value;
                    if term.is_finite() {
                        block_data[target.data_idx] += term;
                    }
                }
            }
        }
    }
    true
}

fn int4c_r_ssss_block_into_data_with_pairs(
    ab_pairs: &[RintPrimitivePair4c],
    cd_pairs: &[RintPrimitivePair4c],
    block_data: &mut [f64],
) {
    let mut value = 0.0_f64;
    for ab_pair in ab_pairs {
        let p_sum = ab_pair.exp_sum;
        let p_center = ab_pair.center;
        for cd_pair in cd_pairs {
            let q_sum = cd_pair.exp_sum;
            let p_sum_q = p_sum + q_sum;
            let pq_mul = p_sum * q_sum;
            debug_assert!(p_sum > 0.0 && q_sum > 0.0 && p_sum_q > 0.0 && pq_mul > 0.0);
            let rho = pq_mul / p_sum_q;
            let t = rho * distance_squared(&p_center, &cd_pair.center);
            let pref = TWO_PI_POW_2P5 / p_sum_q.sqrt();
            let term = ab_pair.scaled_coeff_over_exp_sum
                * cd_pair.scaled_coeff_over_exp_sum
                * pref
                * boys_f0(t);
            if term.is_finite() {
                value += term;
            }
        }
    }
    block_data[0] = value;
}

fn int4c_r_total_ang1_block_into_data_with_pairs(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
    ab_pairs: &[RintPrimitivePair4c],
    cd_pairs: &[RintPrimitivePair4c],
    block_data: &mut [f64],
) {
    let (p_shell_id, p_targets, p_target_len) = build_total_ang1_targets_4c(a, b, c, d);
    int4c_r_total_ang1_block_into_data_with_pairs_targets(
        a,
        b,
        c,
        d,
        ab_pairs,
        cd_pairs,
        block_data,
        p_shell_id,
        &p_targets,
        p_target_len,
    );
}

#[allow(clippy::too_many_arguments)]
fn int4c_r_total_ang1_block_into_data_with_pairs_targets(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
    ab_pairs: &[RintPrimitivePair4c],
    cd_pairs: &[RintPrimitivePair4c],
    block_data: &mut [f64],
    p_shell_id: usize,
    p_targets: &[(usize, usize); 3],
    p_target_len: usize,
) {
    for ab_pair in ab_pairs {
        let p_sum = ab_pair.exp_sum;
        let p_center = ab_pair.center;
        for cd_pair in cd_pairs {
            let q_sum = cd_pair.exp_sum;
            let p_sum_q = p_sum + q_sum;
            let pq_mul = p_sum * q_sum;
            debug_assert!(p_sum > 0.0 && q_sum > 0.0 && p_sum_q > 0.0 && pq_mul > 0.0);

            let rho = pq_mul / p_sum_q;
            let t = rho * distance_squared(&p_center, &cd_pair.center);
            let (f0, f1) = boys_f0_f1(t);
            if f0 == 0.0 {
                continue;
            }
            let root = f1 / f0;
            if !root.is_finite() {
                continue;
            }
            let pref = TWO_PI_POW_2P5 / p_sum_q.sqrt();
            let primitive_scale =
                ab_pair.scaled_coeff_over_exp_sum * cd_pair.scaled_coeff_over_exp_sum * pref * f0;
            if !primitive_scale.is_finite() {
                continue;
            }

            let scalars = rys_root_scalars_4c(root, p_sum, q_sum, p_sum_q);
            let mut a1 = [0.0_f64; 3];
            let mut b1 = [0.0_f64; 3];
            let mut c1 = [0.0_f64; 3];
            let mut d1 = [0.0_f64; 3];
            for axis in 0..3 {
                let rc = rys_axis_coeffs_4c_from_scalars(
                    scalars,
                    root,
                    p_center[axis],
                    cd_pair.center[axis],
                    a.center[axis],
                    c.center[axis],
                );
                a1[axis] = rc.c00;
                b1[axis] = rc.c00 + a.center[axis] - b.center[axis];
                c1[axis] = rc.c00p;
                d1[axis] = rc.c00p + c.center[axis] - d.center[axis];
            }

            let first_values = match p_shell_id {
                0 => &a1,
                1 => &b1,
                2 => &c1,
                _ => &d1,
            };
            for &(idx, axis) in p_targets[..p_target_len].iter() {
                block_data[idx] += primitive_scale * first_values[axis];
            }
        }
    }
}

fn int4c_r_total_ang2_block_into_data_with_pairs(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
    ab_pairs: &[RintPrimitivePair4c],
    cd_pairs: &[RintPrimitivePair4c],
    block_data: &mut [f64],
) -> bool {
    let Some((targets, target_len)) = build_total_ang2_targets_4c(a, b, c, d) else {
        return false;
    };
    int4c_r_total_ang2_block_into_data_with_pairs_targets(
        a, b, c, d, ab_pairs, cd_pairs, block_data, &targets, target_len,
    )
}

#[allow(clippy::too_many_arguments)]
fn int4c_r_total_ang2_block_into_data_with_pairs_targets(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
    ab_pairs: &[RintPrimitivePair4c],
    cd_pairs: &[RintPrimitivePair4c],
    block_data: &mut [f64],
    targets: &[TotalAng2Target4c; 16],
    target_len: usize,
) -> bool {
    for ab_pair in ab_pairs {
        let p_sum = ab_pair.exp_sum;
        let p_center = ab_pair.center;
        for cd_pair in cd_pairs {
            let q_sum = cd_pair.exp_sum;
            let p_sum_q = p_sum + q_sum;
            let pq_mul = p_sum * q_sum;
            debug_assert!(p_sum > 0.0 && q_sum > 0.0 && p_sum_q > 0.0 && pq_mul > 0.0);

            let rho = pq_mul / p_sum_q;
            let t = rho * distance_squared(&p_center, &cd_pair.center);
            let pref = TWO_PI_POW_2P5 / p_sum_q.sqrt();
            let primitive_coeff =
                ab_pair.scaled_coeff_over_exp_sum * cd_pair.scaled_coeff_over_exp_sum;
            let mut roots = [0.0_f64; 2];
            let mut weights = [0.0_f64; 2];
            let nroots = rys_roots_weights_r_into(2, t, &mut roots, &mut weights);
            if nroots != 2 {
                return false;
            }
            for root_idx in 0..2 {
                let values = rys_total_ang2_values(
                    roots[root_idx],
                    p_sum,
                    q_sum,
                    p_sum_q,
                    &p_center,
                    &cd_pair.center,
                    &a.center,
                    &b.center,
                    &c.center,
                    &d.center,
                );
                let primitive_scale = primitive_coeff * pref * weights[root_idx];
                if !primitive_scale.is_finite() {
                    continue;
                }
                for target in targets[..target_len].iter() {
                    let term = primitive_scale * total_ang2_target_value(*target, &values);
                    if term.is_finite() {
                        block_data[target.data_idx] += term;
                    }
                }
            }
        }
    }
    true
}

fn int4c_r_three_p_one_s_block_into_data_with_pairs(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
    ab_pairs: &[RintPrimitivePair4c],
    cd_pairs: &[RintPrimitivePair4c],
    block_data: &mut [f64],
) -> bool {
    let p_shell_count = [a, b, c, d]
        .iter()
        .filter(|shell| shell.shell.ang_type == 1)
        .count();
    let s_shell_count = [a, b, c, d]
        .iter()
        .filter(|shell| shell.shell.ang_type == 0)
        .count();
    if p_shell_count != 3 || s_shell_count != 1 {
        return false;
    }

    let Some((targets, target_len)) = build_three_p_one_s_targets_4c(a, b, c, d) else {
        return false;
    };
    int4c_r_three_p_one_s_block_into_data_with_pairs_targets(
        a, b, c, d, ab_pairs, cd_pairs, block_data, &targets, target_len,
    )
}

#[allow(clippy::too_many_arguments)]
fn int4c_r_three_p_one_s_block_into_data_with_pairs_targets(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
    ab_pairs: &[RintPrimitivePair4c],
    cd_pairs: &[RintPrimitivePair4c],
    block_data: &mut [f64],
    targets: &[PTripletTarget4c; 81],
    target_len: usize,
) -> bool {
    for ab_pair in ab_pairs {
        let p_sum = ab_pair.exp_sum;
        let p_center = ab_pair.center;
        for cd_pair in cd_pairs {
            let q_sum = cd_pair.exp_sum;
            let p_sum_q = p_sum + q_sum;
            let pq_mul = p_sum * q_sum;
            debug_assert!(p_sum > 0.0 && q_sum > 0.0 && p_sum_q > 0.0 && pq_mul > 0.0);

            let rho = pq_mul / p_sum_q;
            let t = rho * distance_squared(&p_center, &cd_pair.center);
            let mut roots = [0.0_f64; 2];
            let mut weights = [0.0_f64; 2];
            let nroots = rys_roots_weights_r_into(2, t, &mut roots, &mut weights);
            if nroots != 2 {
                return false;
            }

            let pref = TWO_PI_POW_2P5 / p_sum_q.sqrt();
            let primitive_coeff =
                ab_pair.scaled_coeff_over_exp_sum * cd_pair.scaled_coeff_over_exp_sum;
            for root_idx in 0..2 {
                let root = roots[root_idx];
                let scalars = rys_root_scalars_4c(root, p_sum, q_sum, p_sum_q);
                let mut first = [[0.0_f64; 3]; 4];
                for axis in 0..3 {
                    let rc = rys_axis_coeffs_4c_from_scalars(
                        scalars,
                        root,
                        p_center[axis],
                        cd_pair.center[axis],
                        a.center[axis],
                        c.center[axis],
                    );
                    first[0][axis] = rc.c00;
                    first[1][axis] = rc.c00 + a.center[axis] - b.center[axis];
                    first[2][axis] = rc.c00p;
                    first[3][axis] = rc.c00p + c.center[axis] - d.center[axis];
                }

                let primitive_scale = primitive_coeff * pref * weights[root_idx];
                if !primitive_scale.is_finite() {
                    continue;
                }
                for target in targets[..target_len].iter() {
                    let value = p_triplet_component_value(target.axes, target.ids, &first, scalars);
                    let term = primitive_scale * value;
                    if term.is_finite() {
                        block_data[target.data_idx] += term;
                    }
                }
            }
        }
    }
    true
}

fn build_rint_shell_coefficient_cache(shells: &[RintShell]) -> Vec<&[f64]> {
    shells.iter().map(rint_shell_coefficients).collect()
}

fn build_shell_pair_primitive_pair_cache(
    shells: &[RintShell],
    shell_coeffs: &[&[f64]],
) -> Vec<Vec<RintPrimitivePair4c>> {
    let pair_count = shells.len() * (shells.len() + 1) / 2;
    let mut cache = vec![Vec::new(); pair_count];
    for left_idx in 0..shells.len() {
        for right_idx in 0..=left_idx {
            let rank = shell_pair_rank(left_idx, right_idx);
            cache[rank] = build_primitive_pairs_4c(
                &shells[left_idx],
                &shells[right_idx],
                &shell_coeffs[left_idx],
                &shell_coeffs[right_idx],
                distance_squared(&shells[left_idx].center, &shells[right_idx].center),
            );
        }
    }
    cache
}

fn build_single_contraction_shell_views(shells: &[RintContractedShell]) -> Vec<RintShell> {
    shells
        .iter()
        .map(|shell| RintShell {
            atom_idx: shell.atom_idx,
            center: shell.center,
            shell: basis::Shell {
                ang_type: shell.l,
                exponents: shell.exponents.clone(),
                coefficients: vec![shell
                    .coeff_columns
                    .first()
                    .cloned()
                    .unwrap_or_else(Vec::new)],
            },
            column_idx: 0,
            cart_components: shell.cart_components.clone(),
            ao_start: shell.ao_start,
            ao_len: shell.cart_len,
            is_aux: shell.is_aux,
        })
        .collect()
}

fn build_contracted_shell_pair_primitive_pair_cache(
    shells: &[RintContractedShell],
) -> Vec<Vec<RintContractedPrimitivePair4c>> {
    let pair_count = shells.len() * (shells.len() + 1) / 2;
    let mut cache = vec![Vec::new(); pair_count];
    for left_idx in 0..shells.len() {
        for right_idx in 0..=left_idx {
            let rank = shell_pair_rank(left_idx, right_idx);
            cache[rank] = build_contracted_primitive_pairs_4c(
                &shells[left_idx],
                &shells[right_idx],
                distance_squared(&shells[left_idx].center, &shells[right_idx].center),
            );
        }
    }
    cache
}

#[allow(clippy::too_many_arguments)]
fn build_dense_1d_panel_for_t_into<C: CoeffProvider>(
    z: &mut RysSeedPanel2d,
    coeffs: &C,
    root: &f64,
    ni_max: u32,
    nk_max: u32,
    p: f64,
    q: f64,
    ax: f64,
    bx: f64,
    cx: f64,
    dx: f64,
    alpha: &f64,
    beta: &f64,
    gamma: &f64,
    delta: &f64,
    rho: &f64,
) {
    let rc = coeffs.coeffs_for(
        root, &p, &q, &ax, &bx, &cx, &dx, alpha, beta, gamma, delta, rho,
    );
    z.set(0, 0, 1.0);

    if ni_max >= 1 {
        z.set(1, 0, rc.c00);
        let mut g_nm1 = z.get(0, 0);
        let mut g_n = z.get(1, 0);
        for n in 1..ni_max {
            let cur = (n as f64) * rc.b10 * g_nm1 + rc.c00 * g_n;
            z.set(n + 1, 0, cur);
            g_nm1 = g_n;
            g_n = cur;
        }
    }

    if nk_max >= 1 {
        z.set(0, 1, rc.c00p);
        let mut g_mn1 = z.get(0, 0);
        let mut g_m = z.get(0, 1);
        for m in 1..nk_max {
            let cur = (m as f64) * rc.b01p * g_mn1 + rc.c00p * g_m;
            z.set(0, m + 1, cur);
            g_mn1 = g_m;
            g_m = cur;
        }
    }

    if nk_max >= 1 {
        for n in 0..=ni_max {
            let mut value = rc.c00p * z.get(n, 0);
            if n > 0 {
                value += (n as f64) * rc.b00 * z.get(n - 1, 0);
            }
            z.set(n, 1, value);
        }
    }

    if ni_max >= 1 {
        for m in 0..=nk_max {
            let mut value = rc.c00 * z.get(0, m);
            if m > 0 {
                value += (m as f64) * rc.b00 * z.get(0, m - 1);
            }
            z.set(1, m, value);
        }
    }

    for n in 1..=ni_max {
        for m in 1..nk_max {
            let value = (m as f64) * rc.b01p * z.get(n, m - 1)
                + (n as f64) * rc.b00 * z.get(n - 1, m)
                + rc.c00p * z.get(n, m);
            z.set(n, m + 1, value);
        }
    }
}

fn build_dense_1d_panel_for_4c_into(
    z: &mut RysSeedPanel2d,
    rc: RysAxisCoeffs4c,
    ni_max: u32,
    nk_max: u32,
) {
    z.set(0, 0, 1.0);

    if ni_max >= 1 {
        z.set(1, 0, rc.c00);
        let mut g_nm1 = z.get(0, 0);
        let mut g_n = z.get(1, 0);
        for n in 1..ni_max {
            let cur = (n as f64) * rc.b10 * g_nm1 + rc.c00 * g_n;
            z.set(n + 1, 0, cur);
            g_nm1 = g_n;
            g_n = cur;
        }
    }

    if nk_max >= 1 {
        z.set(0, 1, rc.c00p);
        let mut g_mn1 = z.get(0, 0);
        let mut g_m = z.get(0, 1);
        for m in 1..nk_max {
            let cur = (m as f64) * rc.b01p * g_mn1 + rc.c00p * g_m;
            z.set(0, m + 1, cur);
            g_mn1 = g_m;
            g_m = cur;
        }
    }

    if nk_max >= 1 {
        for n in 0..=ni_max {
            let mut value = rc.c00p * z.get(n, 0);
            if n > 0 {
                value += (n as f64) * rc.b00 * z.get(n - 1, 0);
            }
            z.set(n, 1, value);
        }
    }

    if ni_max >= 1 {
        for m in 0..=nk_max {
            let mut value = rc.c00 * z.get(0, m);
            if m > 0 {
                value += (m as f64) * rc.b00 * z.get(0, m - 1);
            }
            z.set(1, m, value);
        }
    }

    for n in 1..=ni_max {
        for m in 1..nk_max {
            let value = (m as f64) * rc.b01p * z.get(n, m - 1)
                + (n as f64) * rc.b00 * z.get(n - 1, m)
                + rc.c00p * z.get(n, m);
            z.set(n, m + 1, value);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_rys_transfer_table_4c<C: CoeffProvider>(
    table: &mut RysTransferTable4c,
    z_panel: &mut RysSeedPanel2d,
    root: &f64,
    ni_max: u32,
    nj_max: u32,
    nk_max: u32,
    nl_max: u32,
    p_center: f64,
    q_center: f64,
    a_center: f64,
    b_center: f64,
    c_center: f64,
    d_center: f64,
    alpha: &f64,
    beta: &f64,
    gamma: &f64,
    delta: &f64,
    rho: &f64,
    coeffs: &C,
) {
    build_dense_1d_panel_for_t_into(
        z_panel, coeffs, root, ni_max, nk_max, p_center, q_center, a_center, b_center, c_center,
        d_center, alpha, beta, gamma, delta, rho,
    );
    for nk in 0..=nk_max {
        for ni in 0..=ni_max {
            table.set(ni, 0, nk, 0, z_panel.get(ni, nk));
        }
    }
    let xi_minus_xj = a_center - b_center;
    let xk_minus_xl = c_center - d_center;

    if xi_minus_xj.abs() <= 1.0e-18 {
        for nk in 0..=nk_max {
            for nj in 1..=nj_max {
                let i_upper = ni_max - nj;
                for ni in 0..=i_upper {
                    let value = table.get(ni + nj, 0, nk, 0);
                    table.set(ni, nj, nk, 0, value);
                }
            }
        }
    } else {
        for nk in 0..=nk_max {
            for nj in 1..=nj_max {
                let i_upper = ni_max - nj;
                for ni in 0..=i_upper {
                    let value = table.get(ni + 1, nj - 1, nk, 0)
                        + xi_minus_xj * table.get(ni, nj - 1, nk, 0);
                    table.set(ni, nj, nk, 0, value);
                }
            }
        }
    }

    if xk_minus_xl.abs() <= 1.0e-18 {
        for nl in 1..=nl_max {
            let k_upper = nk_max - nl;
            for nk in 0..=k_upper {
                for nj in 0..=nj_max {
                    let i_upper = ni_max.saturating_sub(nj);
                    for ni in 0..=i_upper {
                        let value = table.get(ni, nj, nk + nl, 0);
                        table.set(ni, nj, nk, nl, value);
                    }
                }
            }
        }
    } else {
        for nl in 1..=nl_max {
            let k_upper = nk_max - nl;
            for nk in 0..=k_upper {
                for nj in 0..=nj_max {
                    let i_upper = ni_max.saturating_sub(nj);
                    for ni in 0..=i_upper {
                        let value = table.get(ni, nj, nk + 1, nl - 1)
                            + xk_minus_xl * table.get(ni, nj, nk, nl - 1);
                        table.set(ni, nj, nk, nl, value);
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_rys_transfer_table_4c_direct(
    table: &mut RysTransferTable4c,
    z_panel: &mut RysSeedPanel2d,
    coeffs: RysAxisCoeffs4c,
    ni_max: u32,
    nj_max: u32,
    nk_max: u32,
    nl_max: u32,
    a_center: f64,
    b_center: f64,
    c_center: f64,
    d_center: f64,
) {
    build_dense_1d_panel_for_4c_into(z_panel, coeffs, ni_max, nk_max);
    for nk in 0..=nk_max {
        for ni in 0..=ni_max {
            table.set(ni, 0, nk, 0, z_panel.get(ni, nk));
        }
    }
    let xi_minus_xj = a_center - b_center;
    let xk_minus_xl = c_center - d_center;

    if xi_minus_xj.abs() <= 1.0e-18 {
        for nk in 0..=nk_max {
            for nj in 1..=nj_max {
                let i_upper = ni_max - nj;
                for ni in 0..=i_upper {
                    let value = table.get(ni + nj, 0, nk, 0);
                    table.set(ni, nj, nk, 0, value);
                }
            }
        }
    } else {
        for nk in 0..=nk_max {
            for nj in 1..=nj_max {
                let i_upper = ni_max - nj;
                for ni in 0..=i_upper {
                    let value = table.get(ni + 1, nj - 1, nk, 0)
                        + xi_minus_xj * table.get(ni, nj - 1, nk, 0);
                    table.set(ni, nj, nk, 0, value);
                }
            }
        }
    }

    if xk_minus_xl.abs() <= 1.0e-18 {
        for nl in 1..=nl_max {
            let k_upper = nk_max - nl;
            for nk in 0..=k_upper {
                for nj in 0..=nj_max {
                    let i_upper = ni_max.saturating_sub(nj);
                    for ni in 0..=i_upper {
                        let value = table.get(ni, nj, nk + nl, 0);
                        table.set(ni, nj, nk, nl, value);
                    }
                }
            }
        }
    } else {
        for nl in 1..=nl_max {
            let k_upper = nk_max - nl;
            for nk in 0..=k_upper {
                for nj in 0..=nj_max {
                    let i_upper = ni_max.saturating_sub(nj);
                    for ni in 0..=i_upper {
                        let value = table.get(ni, nj, nk + 1, nl - 1)
                            + xk_minus_xl * table.get(ni, nj, nk, nl - 1);
                        table.set(ni, nj, nk, nl, value);
                    }
                }
            }
        }
    }
}

fn add_primitive_3c_r2_shell_block(
    block: &mut MatrixFull<f64>,
    left: &RintShell,
    right: &RintShell,
    aux: &RintShell,
    entries: &[Rint3cBlockEntry],
    alpha: f64,
    beta: f64,
    gamma: f64,
    coeff_prod: f64,
    rab2: f64,
) {
    let pi = std::f64::consts::PI;
    let p_sum = alpha + beta;
    let q_sum = gamma;
    let p_sum_q = p_sum + q_sum;
    let pq_mul = p_sum * q_sum;
    if p_sum <= 0.0 || q_sum <= 0.0 || p_sum_q <= 0.0 || pq_mul <= 0.0 {
        return;
    }

    let p_fac = alpha * beta / p_sum;
    let p_center = gaussian_product_center(alpha, beta, &left.center, &right.center);
    let rpq2 = distance_squared(&p_center, &aux.center);
    let pref = 2.0 * pi.powf(3.0) / (pq_mul.sqrt() * p_sum_q);
    let exp_ab = (-p_fac * rab2).exp();
    let rho = pq_mul / p_sum_q;
    let t = rho * rpq2;
    let nroots =
        ((left.shell.ang_type + right.shell.ang_type + aux.shell.ang_type) / 2 + 1) as usize;
    let (roots, weights) = rys_roots_weights_r2(nroots, t);
    let vrr = RysCoeffProvider;
    let ni_max = left.shell.ang_type + right.shell.ang_type;
    let nj_max = right.shell.ang_type;
    let nk_max = aux.shell.ang_type;

    for root_idx in 0..nroots {
        let root = &roots[root_idx];
        let weight = weights[root_idx];
        let table_x = build_rys_transfer_table_3c(
            root,
            ni_max,
            nj_max,
            nk_max,
            p_center[0],
            left.center[0],
            right.center[0],
            aux.center[0],
            &alpha,
            &beta,
            &gamma,
            &rho,
            &vrr,
        );
        let table_y = build_rys_transfer_table_3c(
            root,
            ni_max,
            nj_max,
            nk_max,
            p_center[1],
            left.center[1],
            right.center[1],
            aux.center[1],
            &alpha,
            &beta,
            &gamma,
            &rho,
            &vrr,
        );
        let table_z = build_rys_transfer_table_3c(
            root,
            ni_max,
            nj_max,
            nk_max,
            p_center[2],
            left.center[2],
            right.center[2],
            aux.center[2],
            &alpha,
            &beta,
            &gamma,
            &rho,
            &vrr,
        );

        let primitive_scale = coeff_prod * exp_ab * pref * weight;
        for entry in entries {
            let ix = table_x.get(entry.left_ang[0], entry.right_ang[0], entry.aux_ang[0]);
            let iy = table_y.get(entry.left_ang[1], entry.right_ang[1], entry.aux_ang[1]);
            let iz = table_z.get(entry.left_ang[2], entry.right_ang[2], entry.aux_ang[2]);
            let term = primitive_scale * ix * iy * iz;
            if term.is_finite() {
                block.data[entry.data_idx] += term;
            }
        }
    }
}

fn int3c_r2_shell_block_batched(
    left: &RintShell,
    right: &RintShell,
    aux: &RintShell,
) -> MatrixFull<f64> {
    int3c_r2_shell_block_batched_with_expcutoff(left, right, aux, LIBCINT_DEFAULT_EXPCUTOFF)
}

fn int3c_r2_shell_block_batched_unscreened(
    left: &RintShell,
    right: &RintShell,
    aux: &RintShell,
) -> MatrixFull<f64> {
    int3c_r2_shell_block_batched_with_expcutoff(left, right, aux, f64::INFINITY)
}

fn int3c_r2_shell_block_batched_with_expcutoff(
    left: &RintShell,
    right: &RintShell,
    aux: &RintShell,
    expcutoff: f64,
) -> MatrixFull<f64> {
    let pair_rows = left.ao_len * right.ao_len;
    let mut block = MatrixFull::new([pair_rows, aux.ao_len], 0.0_f64);
    let entries = build_3c_block_entries(left, right, aux);
    int3c_r2_shell_block_batched_into_with_expcutoff(
        left, right, aux, &entries, &mut block, expcutoff,
    );
    block
}

fn int3c_r2_shell_block_batched_into(
    left: &RintShell,
    right: &RintShell,
    aux: &RintShell,
    entries: &[Rint3cBlockEntry],
    block: &mut MatrixFull<f64>,
) {
    int3c_r2_shell_block_batched_into_with_expcutoff(
        left,
        right,
        aux,
        entries,
        block,
        LIBCINT_DEFAULT_EXPCUTOFF,
    );
}

fn int3c_r2_shell_block_batched_into_direct(
    left: &RintShell,
    right: &RintShell,
    aux: &RintShell,
    entries: &[Rint3cBlockEntry],
    block: &mut MatrixFull<f64>,
) {
    int3c_r2_shell_block_batched_into_with_expcutoff(
        left,
        right,
        aux,
        entries,
        block,
        r2_direct_3c_expcutoff(),
    );
}

fn r2_direct_3c_expcutoff() -> f64 {
    static R2_DIRECT_3C_EXPCUTOFF: OnceLock<f64> = OnceLock::new();
    *R2_DIRECT_3C_EXPCUTOFF.get_or_init(|| {
        std::env::var("REST_R2_DIRECT_3C_EXPCUTOFF")
            .ok()
            .and_then(|value| value.trim().parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value >= 0.0)
            .unwrap_or(LIBCINT_DEFAULT_EXPCUTOFF)
    })
}

fn r2_direct_shell_screen_enabled() -> bool {
    std::env::var("REST_R2_DIRECT_SHELL_SCREEN")
        .ok()
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            matches!(value.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn r2_shell_pair_all_primitives_screened(
    left: &RintShell,
    right: &RintShell,
    expcutoff: f64,
) -> bool {
    if !expcutoff.is_finite() {
        return false;
    }
    let rab2 = distance_squared(&left.center, &right.center);
    let pair_log_bound = libcint_pair_log_bound(left, right, rab2);
    if !pair_log_bound.is_finite() {
        return false;
    }
    let left_log_coeffs = rint_shell_log_abs_coefficients(left);
    let right_log_coeffs = rint_shell_log_abs_coefficients(right);
    let mut min_cceij = f64::INFINITY;
    for (alpha, left_log_coeff) in left.shell.exponents.iter().zip(left_log_coeffs.iter()) {
        for (beta, right_log_coeff) in right.shell.exponents.iter().zip(right_log_coeffs.iter()) {
            let cceij = libcint_pair_cceij(
                *alpha,
                *beta,
                *left_log_coeff,
                *right_log_coeff,
                rab2,
                pair_log_bound,
            );
            min_cceij = min_cceij.min(cceij);
        }
    }
    min_cceij > expcutoff
}

fn int3c_r2_shell_block_batched_into_with_expcutoff(
    left: &RintShell,
    right: &RintShell,
    aux: &RintShell,
    entries: &[Rint3cBlockEntry],
    block: &mut MatrixFull<f64>,
    expcutoff: f64,
) {
    block.data.fill(0.0);
    let left_coeffs = rint_shell_coefficients(left);
    let right_coeffs = rint_shell_coefficients(right);
    let aux_coeffs = rint_shell_coefficients(aux);
    let left_log_coeffs = rint_shell_log_abs_coefficients(left);
    let right_log_coeffs = rint_shell_log_abs_coefficients(right);
    let rab2 = distance_squared(&left.center, &right.center);
    let pair_log_bound = if expcutoff.is_finite() {
        libcint_pair_log_bound(left, right, rab2)
    } else {
        f64::INFINITY
    };

    for (alpha_idx, (&alpha, &a_coeff)) in left
        .shell
        .exponents
        .iter()
        .zip(left_coeffs.iter())
        .enumerate()
    {
        for (beta_idx, (&beta, &b_coeff)) in right
            .shell
            .exponents
            .iter()
            .zip(right_coeffs.iter())
            .enumerate()
        {
            if expcutoff.is_finite() {
                let cceij = libcint_pair_cceij(
                    alpha,
                    beta,
                    left_log_coeffs[alpha_idx],
                    right_log_coeffs[beta_idx],
                    rab2,
                    pair_log_bound,
                );
                if cceij > expcutoff {
                    continue;
                }
            }
            for (&gamma, &c_coeff) in aux.shell.exponents.iter().zip(aux_coeffs.iter()) {
                let coeff_prod = a_coeff * b_coeff * c_coeff;
                if coeff_prod == 0.0 {
                    continue;
                }
                add_primitive_3c_r2_shell_block(
                    block, left, right, aux, entries, alpha, beta, gamma, coeff_prod, rab2,
                );
            }
        }
    }
}

fn add_primitive_4c_r_shell_block(
    block_data: &mut [f64],
    workspace: &mut RysTransferWorkspace4c,
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
    entries: &[Rint4cBlockEntry],
    ab_pair: &RintPrimitivePair4c,
    cd_pair: &RintPrimitivePair4c,
) {
    let p_sum = ab_pair.exp_sum;
    let q_sum = cd_pair.exp_sum;
    let p_sum_q = p_sum + q_sum;
    let pq_mul = p_sum * q_sum;
    debug_assert!(p_sum > 0.0 && q_sum > 0.0 && p_sum_q > 0.0 && pq_mul > 0.0);

    let p_center = ab_pair.center;
    let q_center = cd_pair.center;
    let rpq2 = distance_squared(&p_center, &q_center);
    let pref = TWO_PI_POW_2P5 / p_sum_q.sqrt();
    let primitive_coeff = ab_pair.scaled_coeff_over_exp_sum * cd_pair.scaled_coeff_over_exp_sum;
    let rho = pq_mul / p_sum_q;
    let t = rho * rpq2;
    let ni_max = a.shell.ang_type + b.shell.ang_type;
    let nj_max = b.shell.ang_type;
    let nk_max = c.shell.ang_type + d.shell.ang_type;
    let nl_max = d.shell.ang_type;
    if ni_max == 0 && nj_max == 0 && nk_max == 0 && nl_max == 0 {
        let term = primitive_coeff * pref * boys_f0(t);
        if term.is_finite() {
            block_data[0] += term;
        }
        return;
    }

    let total_ang = a.shell.ang_type + b.shell.ang_type + c.shell.ang_type + d.shell.ang_type;
    if total_ang == 1 {
        let (f0, f1) = boys_f0_f1(t);
        if f0 == 0.0 {
            return;
        }
        let root = f1 / f0;
        if !root.is_finite() {
            return;
        }
        let primitive_scale = primitive_coeff * pref * f0;
        if !primitive_scale.is_finite() {
            return;
        }

        let scalars = rys_root_scalars_4c(root, p_sum, q_sum, p_sum_q);
        let mut a1 = [0.0_f64; 3];
        let mut b1 = [0.0_f64; 3];
        let mut c1 = [0.0_f64; 3];
        let mut d1 = [0.0_f64; 3];
        for axis in 0..3 {
            let rc = rys_axis_coeffs_4c_from_scalars(
                scalars,
                root,
                p_center[axis],
                q_center[axis],
                a.center[axis],
                c.center[axis],
            );
            a1[axis] = rc.c00;
            b1[axis] = rc.c00 + a.center[axis] - b.center[axis];
            c1[axis] = rc.c00p;
            d1[axis] = rc.c00p + c.center[axis] - d.center[axis];
        }

        if a.shell.ang_type == 1 {
            for i in 0..a.ao_len {
                let axis = single_p_axis(a.cart_components[i]);
                block_data[i] += primitive_scale * a1[axis];
            }
        } else if b.shell.ang_type == 1 {
            for j in 0..b.ao_len {
                let axis = single_p_axis(b.cart_components[j]);
                block_data[j * a.ao_len] += primitive_scale * b1[axis];
            }
        } else if c.shell.ang_type == 1 {
            let left_rows = a.ao_len * b.ao_len;
            for k in 0..c.ao_len {
                let axis = single_p_axis(c.cart_components[k]);
                block_data[k * left_rows] += primitive_scale * c1[axis];
            }
        } else {
            let left_rows = a.ao_len * b.ao_len;
            for l in 0..d.ao_len {
                let axis = single_p_axis(d.cart_components[l]);
                block_data[l * c.ao_len * left_rows] += primitive_scale * d1[axis];
            }
        }
        return;
    }

    if total_ang == 2
        && add_primitive_4c_total_ang2_fast(
            block_data,
            a,
            b,
            c,
            d,
            primitive_coeff,
            pref,
            t,
            p_sum,
            q_sum,
            p_sum_q,
            &p_center,
            &q_center,
        )
    {
        return;
    }

    let nroots_required = (total_ang / 2 + 1) as usize;
    let mut roots_stack = [0.0_f64; 16];
    let mut weights_stack = [0.0_f64; 16];
    let mut roots_heap = Vec::new();
    let mut weights_heap = Vec::new();
    let nroots = if nroots_required <= roots_stack.len() {
        rys_roots_weights_r_into(nroots_required, t, &mut roots_stack, &mut weights_stack)
    } else {
        let (roots, weights) = rys_roots_weights_r(nroots_required, t);
        if roots.len() == nroots_required && weights.len() == nroots_required {
            roots_heap = roots;
            weights_heap = weights;
            nroots_required
        } else {
            0
        }
    };
    if nroots == 0 {
        return;
    }

    for root_idx in 0..nroots {
        let (root, weight) = if nroots_required <= roots_stack.len() {
            (roots_stack[root_idx], weights_stack[root_idx])
        } else {
            (roots_heap[root_idx], weights_heap[root_idx])
        };
        let scalars = rys_root_scalars_4c(root, p_sum, q_sum, p_sum_q);
        build_rys_transfer_table_4c_direct(
            &mut workspace.table_x,
            &mut workspace.seed_panel,
            rys_axis_coeffs_4c_from_scalars(
                scalars,
                root,
                p_center[0],
                q_center[0],
                a.center[0],
                c.center[0],
            ),
            ni_max,
            nj_max,
            nk_max,
            nl_max,
            a.center[0],
            b.center[0],
            c.center[0],
            d.center[0],
        );
        build_rys_transfer_table_4c_direct(
            &mut workspace.table_y,
            &mut workspace.seed_panel,
            rys_axis_coeffs_4c_from_scalars(
                scalars,
                root,
                p_center[1],
                q_center[1],
                a.center[1],
                c.center[1],
            ),
            ni_max,
            nj_max,
            nk_max,
            nl_max,
            a.center[1],
            b.center[1],
            c.center[1],
            d.center[1],
        );
        build_rys_transfer_table_4c_direct(
            &mut workspace.table_z,
            &mut workspace.seed_panel,
            rys_axis_coeffs_4c_from_scalars(
                scalars,
                root,
                p_center[2],
                q_center[2],
                a.center[2],
                c.center[2],
            ),
            ni_max,
            nj_max,
            nk_max,
            nl_max,
            a.center[2],
            b.center[2],
            c.center[2],
            d.center[2],
        );

        let primitive_scale = primitive_coeff * pref * weight;
        for entry in entries {
            debug_assert!(entry.x_idx < workspace.table_x.data.len());
            debug_assert!(entry.y_idx < workspace.table_y.data.len());
            debug_assert!(entry.z_idx < workspace.table_z.data.len());
            let ix = unsafe { *workspace.table_x.data.get_unchecked(entry.x_idx) };
            let iy = unsafe { *workspace.table_y.data.get_unchecked(entry.y_idx) };
            let iz = unsafe { *workspace.table_z.data.get_unchecked(entry.z_idx) };
            let term = primitive_scale * ix * iy * iz;
            if term.is_finite() {
                block_data[entry.data_idx] += term;
            }
        }
    }
}

fn int4c_r_shell_block_batched_into_data(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
    entries: &mut Vec<Rint4cBlockEntry>,
    block_data: &mut Vec<f64>,
) -> [usize; 2] {
    let a_coeffs = rint_shell_coefficients(a);
    let b_coeffs = rint_shell_coefficients(b);
    let c_coeffs = rint_shell_coefficients(c);
    let d_coeffs = rint_shell_coefficients(d);
    let rab2 = distance_squared(&a.center, &b.center);
    let rcd2 = distance_squared(&c.center, &d.center);
    let ab_pairs = build_primitive_pairs_4c(a, b, &a_coeffs, &b_coeffs, rab2);
    let cd_pairs = build_primitive_pairs_4c(c, d, &c_coeffs, &d_coeffs, rcd2);
    int4c_r_shell_block_batched_into_data_with_pairs(
        a, b, c, d, &ab_pairs, &cd_pairs, entries, block_data,
    )
}

fn int4c_r_shell_block_batched_into_data_with_pairs(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
    ab_pairs: &[RintPrimitivePair4c],
    cd_pairs: &[RintPrimitivePair4c],
    entries: &mut Vec<Rint4cBlockEntry>,
    block_data: &mut Vec<f64>,
) -> [usize; 2] {
    let ni_max = a.shell.ang_type + b.shell.ang_type;
    let nj_max = b.shell.ang_type;
    let nk_max = c.shell.ang_type + d.shell.ang_type;
    let nl_max = d.shell.ang_type;
    let mut workspace = RysTransferWorkspace4c::new(ni_max, nj_max, nk_max, nl_max);
    int4c_r_shell_block_batched_into_data_with_pairs_and_workspace(
        a,
        b,
        c,
        d,
        ab_pairs,
        cd_pairs,
        entries,
        block_data,
        &mut workspace,
    )
}

fn int4c_r_shell_block_batched_into_data_with_pairs_and_workspace(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
    ab_pairs: &[RintPrimitivePair4c],
    cd_pairs: &[RintPrimitivePair4c],
    entries: &mut Vec<Rint4cBlockEntry>,
    block_data: &mut Vec<f64>,
    workspace: &mut RysTransferWorkspace4c,
) -> [usize; 2] {
    if let Some(shape) = try_int4c_r_shell_block_fast_path_into_data_with_pairs(
        a, b, c, d, ab_pairs, cd_pairs, block_data,
    ) {
        return shape;
    }
    build_4c_block_entries_into(a, b, c, d, entries);
    int4c_r_shell_block_batched_into_data_with_pairs_workspace_entries_generic(
        a, b, c, d, ab_pairs, cd_pairs, entries, block_data, workspace,
    )
}

fn int4c_r_shell_block_batched_into_data_with_pairs_workspace_entries(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
    ab_pairs: &[RintPrimitivePair4c],
    cd_pairs: &[RintPrimitivePair4c],
    entries: &[Rint4cBlockEntry],
    block_data: &mut Vec<f64>,
    workspace: &mut RysTransferWorkspace4c,
) -> [usize; 2] {
    if let Some(shape) = try_int4c_r_shell_block_fast_path_into_data_with_pairs(
        a, b, c, d, ab_pairs, cd_pairs, block_data,
    ) {
        return shape;
    }
    int4c_r_shell_block_batched_into_data_with_pairs_workspace_entries_generic(
        a, b, c, d, ab_pairs, cd_pairs, entries, block_data, workspace,
    )
}

fn build_cached_4c_block_plan(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
) -> Rint4cCachedBlockPlan {
    let total_ang = a.shell.ang_type + b.shell.ang_type + c.shell.ang_type + d.shell.ang_type;
    if total_ang == 0 {
        return Rint4cCachedBlockPlan::Ssss;
    }
    if total_ang == 1 {
        let (p_shell_id, targets, target_len) = build_total_ang1_targets_4c(a, b, c, d);
        return Rint4cCachedBlockPlan::TotalAng1 {
            p_shell_id,
            targets,
            target_len,
        };
    }
    if total_ang == 2 {
        if let Some((targets, target_len)) = build_total_ang2_targets_4c(a, b, c, d) {
            return Rint4cCachedBlockPlan::TotalAng2 {
                targets,
                target_len,
            };
        }
    }
    if total_ang == 3 {
        let p_shell_count = [a, b, c, d]
            .iter()
            .filter(|shell| shell.shell.ang_type == 1)
            .count();
        let s_shell_count = [a, b, c, d]
            .iter()
            .filter(|shell| shell.shell.ang_type == 0)
            .count();
        if p_shell_count == 3 && s_shell_count == 1 {
            if let Some((targets, target_len)) = build_three_p_one_s_targets_4c(a, b, c, d) {
                return Rint4cCachedBlockPlan::ThreePOneS {
                    targets,
                    target_len,
                };
            }
        }
    }
    if total_ang == 4
        && a.shell.ang_type == 1
        && b.shell.ang_type == 1
        && c.shell.ang_type == 1
        && d.shell.ang_type == 1
    {
        if let Some((targets, target_len)) = build_four_p_targets_4c(a, b, c, d) {
            return Rint4cCachedBlockPlan::FourP {
                targets,
                target_len,
            };
        }
    }

    let mut entries = Vec::new();
    build_4c_block_entries_into(a, b, c, d, &mut entries);
    Rint4cCachedBlockPlan::Generic(entries)
}

#[allow(clippy::too_many_arguments)]
fn int4c_r_shell_block_batched_into_data_with_pairs_cached_plan(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
    ab_pairs: &[RintPrimitivePair4c],
    cd_pairs: &[RintPrimitivePair4c],
    plan: &Rint4cCachedBlockPlan,
    block_data: &mut Vec<f64>,
    workspace: &mut RysTransferWorkspace4c,
) -> [usize; 2] {
    let left_rows = a.ao_len * b.ao_len;
    let right_cols = c.ao_len * d.ao_len;
    let shape = [left_rows, right_cols];
    match plan {
        Rint4cCachedBlockPlan::Ssss => {
            block_data.resize(left_rows * right_cols, 0.0_f64);
            block_data.fill(0.0);
            int4c_r_ssss_block_into_data_with_pairs(ab_pairs, cd_pairs, block_data);
            shape
        }
        Rint4cCachedBlockPlan::TotalAng1 {
            p_shell_id,
            targets,
            target_len,
        } => {
            block_data.resize(left_rows * right_cols, 0.0_f64);
            block_data.fill(0.0);
            int4c_r_total_ang1_block_into_data_with_pairs_targets(
                a,
                b,
                c,
                d,
                ab_pairs,
                cd_pairs,
                block_data,
                *p_shell_id,
                targets,
                *target_len,
            );
            shape
        }
        Rint4cCachedBlockPlan::TotalAng2 {
            targets,
            target_len,
        } => {
            block_data.resize(left_rows * right_cols, 0.0_f64);
            block_data.fill(0.0);
            if int4c_r_total_ang2_block_into_data_with_pairs_targets(
                a,
                b,
                c,
                d,
                ab_pairs,
                cd_pairs,
                block_data,
                targets,
                *target_len,
            ) {
                shape
            } else {
                let mut entries = Vec::new();
                build_4c_block_entries_into(a, b, c, d, &mut entries);
                int4c_r_shell_block_batched_into_data_with_pairs_workspace_entries_generic(
                    a, b, c, d, ab_pairs, cd_pairs, &entries, block_data, workspace,
                )
            }
        }
        Rint4cCachedBlockPlan::ThreePOneS {
            targets,
            target_len,
        } => {
            block_data.resize(left_rows * right_cols, 0.0_f64);
            block_data.fill(0.0);
            if int4c_r_three_p_one_s_block_into_data_with_pairs_targets(
                a,
                b,
                c,
                d,
                ab_pairs,
                cd_pairs,
                block_data,
                targets,
                *target_len,
            ) {
                shape
            } else {
                let mut entries = Vec::new();
                build_4c_block_entries_into(a, b, c, d, &mut entries);
                int4c_r_shell_block_batched_into_data_with_pairs_workspace_entries_generic(
                    a, b, c, d, ab_pairs, cd_pairs, &entries, block_data, workspace,
                )
            }
        }
        Rint4cCachedBlockPlan::FourP {
            targets,
            target_len,
        } => {
            block_data.resize(left_rows * right_cols, 0.0_f64);
            block_data.fill(0.0);
            if int4c_r_four_p_block_into_data_with_pairs_targets(
                a,
                b,
                c,
                d,
                ab_pairs,
                cd_pairs,
                block_data,
                targets,
                *target_len,
            ) {
                shape
            } else {
                let mut entries = Vec::new();
                build_4c_block_entries_into(a, b, c, d, &mut entries);
                int4c_r_shell_block_batched_into_data_with_pairs_workspace_entries_generic(
                    a, b, c, d, ab_pairs, cd_pairs, &entries, block_data, workspace,
                )
            }
        }
        Rint4cCachedBlockPlan::Generic(entries) => {
            int4c_r_shell_block_batched_into_data_with_pairs_workspace_entries_generic(
                a, b, c, d, ab_pairs, cd_pairs, entries, block_data, workspace,
            )
        }
    }
}

fn int4c_r_shell_block_batched_into_data_with_pairs_entry_cache(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
    ab_pairs: &[RintPrimitivePair4c],
    cd_pairs: &[RintPrimitivePair4c],
    plan_cache: &mut HashMap<[u32; 4], Rint4cCachedBlockPlan>,
    block_data: &mut Vec<f64>,
    workspace: &mut RysTransferWorkspace4c,
) -> [usize; 2] {
    let entry_key = [
        a.shell.ang_type,
        b.shell.ang_type,
        c.shell.ang_type,
        d.shell.ang_type,
    ];
    let plan = plan_cache
        .entry(entry_key)
        .or_insert_with(|| build_cached_4c_block_plan(a, b, c, d));
    int4c_r_shell_block_batched_into_data_with_pairs_cached_plan(
        a, b, c, d, ab_pairs, cd_pairs, plan, block_data, workspace,
    )
}

fn try_int4c_r_shell_block_fast_path_into_data_with_pairs(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
    ab_pairs: &[RintPrimitivePair4c],
    cd_pairs: &[RintPrimitivePair4c],
    block_data: &mut Vec<f64>,
) -> Option<[usize; 2]> {
    let left_rows = a.ao_len * b.ao_len;
    let right_cols = c.ao_len * d.ao_len;
    let shape = [left_rows, right_cols];
    block_data.resize(left_rows * right_cols, 0.0_f64);
    block_data.fill(0.0);
    let total_ang = a.shell.ang_type + b.shell.ang_type + c.shell.ang_type + d.shell.ang_type;
    if a.shell.ang_type == 0
        && b.shell.ang_type == 0
        && c.shell.ang_type == 0
        && d.shell.ang_type == 0
    {
        int4c_r_ssss_block_into_data_with_pairs(ab_pairs, cd_pairs, block_data);
        return Some(shape);
    }
    if total_ang == 1 {
        int4c_r_total_ang1_block_into_data_with_pairs(a, b, c, d, ab_pairs, cd_pairs, block_data);
        return Some(shape);
    }
    if total_ang == 2
        && int4c_r_total_ang2_block_into_data_with_pairs(a, b, c, d, ab_pairs, cd_pairs, block_data)
    {
        return Some(shape);
    }
    if total_ang == 3
        && int4c_r_three_p_one_s_block_into_data_with_pairs(
            a, b, c, d, ab_pairs, cd_pairs, block_data,
        )
    {
        return Some(shape);
    }
    if total_ang == 4
        && int4c_r_four_p_block_into_data_with_pairs(a, b, c, d, ab_pairs, cd_pairs, block_data)
    {
        return Some(shape);
    }
    None
}

fn int4c_r_shell_block_batched_into_data_with_pairs_workspace_entries_generic(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
    ab_pairs: &[RintPrimitivePair4c],
    cd_pairs: &[RintPrimitivePair4c],
    entries: &[Rint4cBlockEntry],
    block_data: &mut Vec<f64>,
    workspace: &mut RysTransferWorkspace4c,
) -> [usize; 2] {
    let left_rows = a.ao_len * b.ao_len;
    let right_cols = c.ao_len * d.ao_len;
    let shape = [left_rows, right_cols];
    block_data.resize(left_rows * right_cols, 0.0_f64);
    block_data.fill(0.0);

    for ab_pair in ab_pairs {
        for cd_pair in cd_pairs {
            add_primitive_4c_r_shell_block(
                block_data, workspace, a, b, c, d, entries, ab_pair, cd_pair,
            );
        }
    }
    shape
}

#[inline(always)]
fn add_contracted_4c_cart_term_by_carts(
    block_data: &mut [f64],
    a: &RintContractedShell,
    b: &RintContractedShell,
    c: &RintContractedShell,
    d: &RintContractedShell,
    a_cart: usize,
    b_cart: usize,
    c_cart: usize,
    d_cart: usize,
    ab_pair: &RintContractedPrimitivePair4c,
    cd_pair: &RintContractedPrimitivePair4c,
    cart_term: f64,
) {
    if cart_term == 0.0 || !cart_term.is_finite() {
        return;
    }
    let a_nctr = a.nctr();
    let b_nctr = b.nctr();
    let c_nctr = c.nctr();
    let d_nctr = d.nctr();
    let left_rows = a.ao_len * b.ao_len;
    for d_ctr in 0..d_nctr {
        let d_local = d_ctr * d.cart_len + d_cart;
        for c_ctr in 0..c_nctr {
            let cd_coeff = cd_pair.coeff_products[d_ctr * c_nctr + c_ctr];
            if cd_coeff == 0.0 {
                continue;
            }
            let c_local = c_ctr * c.cart_len + c_cart;
            let col_offset = (d_local * c.ao_len + c_local) * left_rows;
            let cd_term = cart_term * cd_coeff;
            for b_ctr in 0..b_nctr {
                let b_local = b_ctr * b.cart_len + b_cart;
                let row_offset = col_offset + b_local * a.ao_len;
                for a_ctr in 0..a_nctr {
                    let ab_coeff = ab_pair.coeff_products[b_ctr * a_nctr + a_ctr];
                    if ab_coeff == 0.0 {
                        continue;
                    }
                    let term = cd_term * ab_coeff;
                    if term.is_finite() {
                        let a_local = a_ctr * a.cart_len + a_cart;
                        block_data[row_offset + a_local] += term;
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn add_contracted_4c_cart_term(
    block_data: &mut [f64],
    a: &RintContractedShell,
    b: &RintContractedShell,
    c: &RintContractedShell,
    d: &RintContractedShell,
    entry: &Rint4cContractedBlockEntry,
    ab_pair: &RintContractedPrimitivePair4c,
    cd_pair: &RintContractedPrimitivePair4c,
    cart_term: f64,
) {
    add_contracted_4c_cart_term_by_carts(
        block_data,
        a,
        b,
        c,
        d,
        entry.a_cart,
        entry.b_cart,
        entry.c_cart,
        entry.d_cart,
        ab_pair,
        cd_pair,
        cart_term,
    );
}

fn int4c_r_contracted_total_ang1_block_into_data_with_pairs(
    a: &RintContractedShell,
    b: &RintContractedShell,
    c: &RintContractedShell,
    d: &RintContractedShell,
    ab_pairs: &[RintContractedPrimitivePair4c],
    cd_pairs: &[RintContractedPrimitivePair4c],
    block_data: &mut [f64],
) {
    let p_shell_id = if a.l == 1 {
        0
    } else if b.l == 1 {
        1
    } else if c.l == 1 {
        2
    } else {
        3
    };
    for ab_pair in ab_pairs {
        let p_sum = ab_pair.exp_sum;
        let p_center = ab_pair.center;
        for cd_pair in cd_pairs {
            let q_sum = cd_pair.exp_sum;
            let p_sum_q = p_sum + q_sum;
            let pq_mul = p_sum * q_sum;
            debug_assert!(p_sum > 0.0 && q_sum > 0.0 && p_sum_q > 0.0 && pq_mul > 0.0);

            let rho = pq_mul / p_sum_q;
            let t = rho * distance_squared(&p_center, &cd_pair.center);
            let (f0, f1) = boys_f0_f1(t);
            if f0 == 0.0 {
                continue;
            }
            let root = f1 / f0;
            if !root.is_finite() {
                continue;
            }
            let pref = TWO_PI_POW_2P5 / p_sum_q.sqrt();
            let primitive_scale =
                ab_pair.scaled_exp_over_exp_sum * cd_pair.scaled_exp_over_exp_sum * pref * f0;
            if !primitive_scale.is_finite() {
                continue;
            }

            let scalars = rys_root_scalars_4c(root, p_sum, q_sum, p_sum_q);
            let mut first = [[0.0_f64; 3]; 4];
            for axis in 0..3 {
                let rc = rys_axis_coeffs_4c_from_scalars(
                    scalars,
                    root,
                    p_center[axis],
                    cd_pair.center[axis],
                    a.center[axis],
                    c.center[axis],
                );
                first[0][axis] = rc.c00;
                first[1][axis] = rc.c00 + a.center[axis] - b.center[axis];
                first[2][axis] = rc.c00p;
                first[3][axis] = rc.c00p + c.center[axis] - d.center[axis];
            }

            match p_shell_id {
                0 => {
                    for i in 0..a.cart_len {
                        let axis = single_p_axis(a.cart_components[i]);
                        add_contracted_4c_cart_term_by_carts(
                            block_data,
                            a,
                            b,
                            c,
                            d,
                            i,
                            0,
                            0,
                            0,
                            ab_pair,
                            cd_pair,
                            primitive_scale * first[0][axis],
                        );
                    }
                }
                1 => {
                    for j in 0..b.cart_len {
                        let axis = single_p_axis(b.cart_components[j]);
                        add_contracted_4c_cart_term_by_carts(
                            block_data,
                            a,
                            b,
                            c,
                            d,
                            0,
                            j,
                            0,
                            0,
                            ab_pair,
                            cd_pair,
                            primitive_scale * first[1][axis],
                        );
                    }
                }
                2 => {
                    for k in 0..c.cart_len {
                        let axis = single_p_axis(c.cart_components[k]);
                        add_contracted_4c_cart_term_by_carts(
                            block_data,
                            a,
                            b,
                            c,
                            d,
                            0,
                            0,
                            k,
                            0,
                            ab_pair,
                            cd_pair,
                            primitive_scale * first[2][axis],
                        );
                    }
                }
                _ => {
                    for l in 0..d.cart_len {
                        let axis = single_p_axis(d.cart_components[l]);
                        add_contracted_4c_cart_term_by_carts(
                            block_data,
                            a,
                            b,
                            c,
                            d,
                            0,
                            0,
                            0,
                            l,
                            ab_pair,
                            cd_pair,
                            primitive_scale * first[3][axis],
                        );
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn int4c_r_contracted_total_ang2_block_into_data_with_pairs_targets(
    a: &RintContractedShell,
    b: &RintContractedShell,
    c: &RintContractedShell,
    d: &RintContractedShell,
    ab_pairs: &[RintContractedPrimitivePair4c],
    cd_pairs: &[RintContractedPrimitivePair4c],
    block_data: &mut [f64],
    targets: &[TotalAng2ContractedTarget4c; 16],
    target_len: usize,
) -> bool {
    for ab_pair in ab_pairs {
        let p_sum = ab_pair.exp_sum;
        let p_center = ab_pair.center;
        for cd_pair in cd_pairs {
            let q_sum = cd_pair.exp_sum;
            let p_sum_q = p_sum + q_sum;
            let pq_mul = p_sum * q_sum;
            debug_assert!(p_sum > 0.0 && q_sum > 0.0 && p_sum_q > 0.0 && pq_mul > 0.0);

            let rho = pq_mul / p_sum_q;
            let t = rho * distance_squared(&p_center, &cd_pair.center);
            let pref = TWO_PI_POW_2P5 / p_sum_q.sqrt();
            let primitive_coeff = ab_pair.scaled_exp_over_exp_sum * cd_pair.scaled_exp_over_exp_sum;
            let mut roots = [0.0_f64; 2];
            let mut weights = [0.0_f64; 2];
            let nroots = rys_roots_weights_r_into(2, t, &mut roots, &mut weights);
            if nroots != 2 {
                return false;
            }
            for root_idx in 0..2 {
                let values = rys_total_ang2_values(
                    roots[root_idx],
                    p_sum,
                    q_sum,
                    p_sum_q,
                    &p_center,
                    &cd_pair.center,
                    &a.center,
                    &b.center,
                    &c.center,
                    &d.center,
                );
                let primitive_scale = primitive_coeff * pref * weights[root_idx];
                if !primitive_scale.is_finite() {
                    continue;
                }
                for target in targets[..target_len].iter() {
                    let value = total_ang2_target_value(
                        TotalAng2Target4c {
                            data_idx: 0,
                            kind: target.kind,
                            ang: target.ang,
                            left_axis: target.left_axis,
                            right_axis: target.right_axis,
                        },
                        &values,
                    );
                    add_contracted_4c_cart_term_by_carts(
                        block_data,
                        a,
                        b,
                        c,
                        d,
                        target.a_cart,
                        target.b_cart,
                        target.c_cart,
                        target.d_cart,
                        ab_pair,
                        cd_pair,
                        primitive_scale * value,
                    );
                }
            }
        }
    }
    true
}

fn add_primitive_4c_r_contracted_shell_block(
    block_data: &mut [f64],
    workspace: &mut RysTransferWorkspace4c,
    a: &RintContractedShell,
    b: &RintContractedShell,
    c: &RintContractedShell,
    d: &RintContractedShell,
    entries: &[Rint4cContractedBlockEntry],
    ab_pair: &RintContractedPrimitivePair4c,
    cd_pair: &RintContractedPrimitivePair4c,
) {
    let p_sum = ab_pair.exp_sum;
    let q_sum = cd_pair.exp_sum;
    let p_sum_q = p_sum + q_sum;
    let pq_mul = p_sum * q_sum;
    debug_assert!(p_sum > 0.0 && q_sum > 0.0 && p_sum_q > 0.0 && pq_mul > 0.0);

    let p_center = ab_pair.center;
    let q_center = cd_pair.center;
    let rpq2 = distance_squared(&p_center, &q_center);
    let pref = TWO_PI_POW_2P5 / p_sum_q.sqrt();
    let primitive_coeff = ab_pair.scaled_exp_over_exp_sum * cd_pair.scaled_exp_over_exp_sum;
    let rho = pq_mul / p_sum_q;
    let t = rho * rpq2;
    let ni_max = a.l + b.l;
    let nj_max = b.l;
    let nk_max = c.l + d.l;
    let nl_max = d.l;
    if ni_max == 0 && nj_max == 0 && nk_max == 0 && nl_max == 0 {
        let cart_term = primitive_coeff * pref * boys_f0(t);
        if let Some(entry) = entries.first() {
            add_contracted_4c_cart_term(block_data, a, b, c, d, entry, ab_pair, cd_pair, cart_term);
        }
        return;
    }

    let total_ang = a.l + b.l + c.l + d.l;
    let nroots_required = (total_ang / 2 + 1) as usize;
    let mut roots_stack = [0.0_f64; 16];
    let mut weights_stack = [0.0_f64; 16];
    let mut roots_heap = Vec::new();
    let mut weights_heap = Vec::new();
    let nroots = if nroots_required <= roots_stack.len() {
        rys_roots_weights_r_into(nroots_required, t, &mut roots_stack, &mut weights_stack)
    } else {
        let (roots, weights) = rys_roots_weights_r(nroots_required, t);
        if roots.len() == nroots_required && weights.len() == nroots_required {
            roots_heap = roots;
            weights_heap = weights;
            nroots_required
        } else {
            0
        }
    };
    if nroots == 0 {
        return;
    }

    for root_idx in 0..nroots {
        let (root, weight) = if nroots_required <= roots_stack.len() {
            (roots_stack[root_idx], weights_stack[root_idx])
        } else {
            (roots_heap[root_idx], weights_heap[root_idx])
        };
        let scalars = rys_root_scalars_4c(root, p_sum, q_sum, p_sum_q);
        build_rys_transfer_table_4c_direct(
            &mut workspace.table_x,
            &mut workspace.seed_panel,
            rys_axis_coeffs_4c_from_scalars(
                scalars,
                root,
                p_center[0],
                q_center[0],
                a.center[0],
                c.center[0],
            ),
            ni_max,
            nj_max,
            nk_max,
            nl_max,
            a.center[0],
            b.center[0],
            c.center[0],
            d.center[0],
        );
        build_rys_transfer_table_4c_direct(
            &mut workspace.table_y,
            &mut workspace.seed_panel,
            rys_axis_coeffs_4c_from_scalars(
                scalars,
                root,
                p_center[1],
                q_center[1],
                a.center[1],
                c.center[1],
            ),
            ni_max,
            nj_max,
            nk_max,
            nl_max,
            a.center[1],
            b.center[1],
            c.center[1],
            d.center[1],
        );
        build_rys_transfer_table_4c_direct(
            &mut workspace.table_z,
            &mut workspace.seed_panel,
            rys_axis_coeffs_4c_from_scalars(
                scalars,
                root,
                p_center[2],
                q_center[2],
                a.center[2],
                c.center[2],
            ),
            ni_max,
            nj_max,
            nk_max,
            nl_max,
            a.center[2],
            b.center[2],
            c.center[2],
            d.center[2],
        );

        let primitive_scale = primitive_coeff * pref * weight;
        if primitive_scale == 0.0 || !primitive_scale.is_finite() {
            continue;
        }
        for entry in entries {
            debug_assert!(entry.x_idx < workspace.table_x.data.len());
            debug_assert!(entry.y_idx < workspace.table_y.data.len());
            debug_assert!(entry.z_idx < workspace.table_z.data.len());
            let ix = unsafe { *workspace.table_x.data.get_unchecked(entry.x_idx) };
            let iy = unsafe { *workspace.table_y.data.get_unchecked(entry.y_idx) };
            let iz = unsafe { *workspace.table_z.data.get_unchecked(entry.z_idx) };
            let cart_term = primitive_scale * ix * iy * iz;
            add_contracted_4c_cart_term(block_data, a, b, c, d, entry, ab_pair, cd_pair, cart_term);
        }
    }
}

fn int4c_r_contracted_shell_block_batched_into_data_with_pairs_workspace_entries(
    a: &RintContractedShell,
    b: &RintContractedShell,
    c: &RintContractedShell,
    d: &RintContractedShell,
    ab_pairs: &[RintContractedPrimitivePair4c],
    cd_pairs: &[RintContractedPrimitivePair4c],
    entries: &[Rint4cContractedBlockEntry],
    block_data: &mut Vec<f64>,
    workspace: &mut RysTransferWorkspace4c,
) -> [usize; 2] {
    let left_rows = a.ao_len * b.ao_len;
    let right_cols = c.ao_len * d.ao_len;
    let shape = [left_rows, right_cols];
    block_data.resize(left_rows * right_cols, 0.0_f64);
    block_data.fill(0.0);
    let total_ang = a.l + b.l + c.l + d.l;
    if total_ang == 1 {
        int4c_r_contracted_total_ang1_block_into_data_with_pairs(
            a, b, c, d, ab_pairs, cd_pairs, block_data,
        );
        return shape;
    }
    if total_ang == 2 {
        if let Some((targets, target_len)) = build_total_ang2_contracted_targets_4c(a, b, c, d) {
            if int4c_r_contracted_total_ang2_block_into_data_with_pairs_targets(
                a, b, c, d, ab_pairs, cd_pairs, block_data, &targets, target_len,
            ) {
                return shape;
            }
        }
    }
    for ab_pair in ab_pairs {
        for cd_pair in cd_pairs {
            add_primitive_4c_r_contracted_shell_block(
                block_data, workspace, a, b, c, d, entries, ab_pair, cd_pair,
            );
        }
    }
    shape
}

fn int4c_r_shell_block_batched(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
) -> MatrixFull<f64> {
    let mut entries = Vec::new();
    let mut data = Vec::new();
    let shape = int4c_r_shell_block_batched_into_data(a, b, c, d, &mut entries, &mut data);
    let block = unsafe { MatrixFull::from_vec_unchecked(shape, data) };
    block
}

fn eri_rint_shell_4c_r2(
    a: &RintShell,
    a_cart: usize,
    b: &RintShell,
    b_cart: usize,
    c: &RintShell,
    c_cart: usize,
    d: &RintShell,
    d_cart: usize,
) -> f64 {
    let a_coeffs = rint_shell_coefficients(a);
    let b_coeffs = rint_shell_coefficients(b);
    let c_coeffs = rint_shell_coefficients(c);
    let d_coeffs = rint_shell_coefficients(d);
    let la = a.cart_components[a_cart];
    let lb = b.cart_components[b_cart];
    let lc = c.cart_components[c_cart];
    let ld = d.cart_components[d_cart];
    let nroots = calculate_nroots_from_angmom(la, lb, lc, ld);
    let rab2 = distance_squared(&a.center, &b.center);
    let rcd2 = distance_squared(&c.center, &d.center);
    let mut eri = 0.0_f64;
    for (&alpha, &a_coeff) in a.shell.exponents.iter().zip(a_coeffs.iter()) {
        for (&beta, &b_coeff) in b.shell.exponents.iter().zip(b_coeffs.iter()) {
            let p_center = gaussian_product_center(alpha, beta, &a.center, &b.center);
            for (&gamma, &c_coeff) in c.shell.exponents.iter().zip(c_coeffs.iter()) {
                for (&delta, &d_coeff) in d.shell.exponents.iter().zip(d_coeffs.iter()) {
                    let q_center = gaussian_product_center(gamma, delta, &c.center, &d.center);
                    let rpq2 = distance_squared(&p_center, &q_center);
                    let eri_prim = gaussian_4c_r2(
                        nroots,
                        &alpha,
                        &beta,
                        &gamma,
                        &delta,
                        &rab2,
                        &rcd2,
                        &rpq2,
                        la,
                        lb,
                        lc,
                        ld,
                        &p_center,
                        &q_center,
                        &a.center,
                        &b.center,
                        &c.center,
                        &d.center,
                        a.ao_start + a_cart,
                        b.ao_start + b_cart,
                        c.ao_start + c_cart,
                        d.ao_start + d_cart,
                    );
                    let term = a_coeff * b_coeff * c_coeff * d_coeff * eri_prim;
                    if term.is_finite() {
                        eri += term;
                    }
                }
            }
        }
    }
    eri
}

/// Shell-block window for r2 two-center auxiliary integrals.
///
/// The returned matrix is column-major with shape `[left.ao_len, right.ao_len]`.
/// Element `(i, j)` is `(left_i | right_j)` for the `1/r12^2` kernel.
pub fn int2c_r2_shell_block(left: &RintShell, right: &RintShell) -> MatrixFull<f64> {
    int2c_r2_shell_block_batched(left, right)
}

/// Shell-block window for r2 three-center integrals.
///
/// The returned matrix is column-major with shape
/// `[left.ao_len * right.ao_len, aux.ao_len]`.  Row
/// `j * left.ao_len + i` stores `(left_i right_j | aux_p)`.
pub fn int3c_r2_shell_block(
    left: &RintShell,
    right: &RintShell,
    aux: &RintShell,
) -> MatrixFull<f64> {
    int3c_r2_shell_block_batched(left, right, aux)
}

/// Shell-block window for exact Coulomb four-center integrals.
///
/// The returned matrix is column-major with shape
/// `[a.ao_len * b.ao_len, c.ao_len * d.ao_len]`. Row
/// `j * a.ao_len + i` and column `l * c.ao_len + k` store
/// `(a_i b_j | c_k d_l)` for the `1/r12` kernel.
pub fn int4c_r_shell_block(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
) -> MatrixFull<f64> {
    int4c_r_shell_block_batched(a, b, c, d)
}

#[inline(always)]
pub fn int4c_r_full_index(mu: usize, nu: usize, lam: usize, sig: usize, nao: usize) -> usize {
    (((mu * nao + nu) * nao + lam) * nao) + sig
}

#[inline(always)]
fn shell_pair_rank(a: usize, b: usize) -> usize {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    hi * (hi + 1) / 2 + lo
}

#[inline(always)]
fn set_eri4_symmetry(
    data: &mut [f64],
    nao: usize,
    mu: usize,
    nu: usize,
    lam: usize,
    sig: usize,
    value: f64,
) {
    data[int4c_r_full_index(mu, nu, lam, sig, nao)] = value;
    data[int4c_r_full_index(nu, mu, lam, sig, nao)] = value;
    data[int4c_r_full_index(mu, nu, sig, lam, nao)] = value;
    data[int4c_r_full_index(nu, mu, sig, lam, nao)] = value;
    data[int4c_r_full_index(lam, sig, mu, nu, nao)] = value;
    data[int4c_r_full_index(sig, lam, mu, nu, nao)] = value;
    data[int4c_r_full_index(lam, sig, nu, mu, nao)] = value;
    data[int4c_r_full_index(sig, lam, nu, mu, nao)] = value;
}

fn build_unique_4c_shell_quartet_tasks(
    ao_shells: &[RintShell],
) -> Vec<(usize, usize, usize, usize)> {
    let mut tasks = Vec::new();
    for a_idx in 0..ao_shells.len() {
        for b_idx in 0..=a_idx {
            let ab_rank = shell_pair_rank(a_idx, b_idx);
            for c_idx in 0..ao_shells.len() {
                for d_idx in 0..=c_idx {
                    if shell_pair_rank(c_idx, d_idx) <= ab_rank {
                        tasks.push((a_idx, b_idx, c_idx, d_idx));
                    }
                }
            }
        }
    }
    tasks
}

fn build_unique_4c_shell_quartet_tasks_for_count(
    nshell: usize,
) -> Vec<(usize, usize, usize, usize)> {
    let mut tasks = Vec::new();
    for a_idx in 0..nshell {
        for b_idx in 0..=a_idx {
            let ab_rank = shell_pair_rank(a_idx, b_idx);
            for c_idx in 0..nshell {
                for d_idx in 0..=c_idx {
                    if shell_pair_rank(c_idx, d_idx) <= ab_rank {
                        tasks.push((a_idx, b_idx, c_idx, d_idx));
                    }
                }
            }
        }
    }
    tasks
}

/// Full exact Coulomb four-center tensor generated from shell-block quartets.
///
/// The returned vector stores `(mu nu | lam sig)` at
/// `int4c_r_full_index(mu, nu, lam, sig, nao)`, where `nao` is the AO basis
/// count implied by `ao_shells`.  The implementation computes only unique
/// shell-pair quartets and fills the eight exact permutation symmetries.
pub fn int4c_r_full_from_shell_blocks(ao_shells: &[RintShell]) -> Vec<f64> {
    let nao = rint_shell_basis_count(ao_shells);
    let mut eri = vec![0.0_f64; nao * nao * nao * nao];
    let mut block_data = Vec::new();
    let shell_coeffs = build_rint_shell_coefficient_cache(ao_shells);
    let primitive_pair_cache = build_shell_pair_primitive_pair_cache(ao_shells, &shell_coeffs);
    let mut workspace_cache: HashMap<[u32; 4], RysTransferWorkspace4c> = HashMap::new();
    let mut entry_cache: HashMap<[u32; 4], Vec<Rint4cBlockEntry>> = HashMap::new();

    for (a_idx, b_idx, c_idx, d_idx) in build_unique_4c_shell_quartet_tasks(ao_shells) {
        let a_shell = &ao_shells[a_idx];
        let b_shell = &ao_shells[b_idx];
        let c_shell = &ao_shells[c_idx];
        let d_shell = &ao_shells[d_idx];
        let ab_pairs = &primitive_pair_cache[shell_pair_rank(a_idx, b_idx)];
        let cd_pairs = &primitive_pair_cache[shell_pair_rank(c_idx, d_idx)];
        let workspace_key = [
            a_shell.shell.ang_type + b_shell.shell.ang_type,
            b_shell.shell.ang_type,
            c_shell.shell.ang_type + d_shell.shell.ang_type,
            d_shell.shell.ang_type,
        ];
        let workspace = workspace_cache.entry(workspace_key).or_insert_with(|| {
            RysTransferWorkspace4c::new(
                workspace_key[0],
                workspace_key[1],
                workspace_key[2],
                workspace_key[3],
            )
        });
        let entry_key = [
            a_shell.shell.ang_type,
            b_shell.shell.ang_type,
            c_shell.shell.ang_type,
            d_shell.shell.ang_type,
        ];
        let entries = entry_cache.entry(entry_key).or_insert_with(|| {
            let mut entries = Vec::new();
            build_4c_block_entries_into(a_shell, b_shell, c_shell, d_shell, &mut entries);
            entries
        });
        let [left_rows, _right_cols] =
            int4c_r_shell_block_batched_into_data_with_pairs_workspace_entries(
                a_shell,
                b_shell,
                c_shell,
                d_shell,
                ab_pairs,
                cd_pairs,
                entries,
                &mut block_data,
                workspace,
            );
        for l in 0..d_shell.ao_len {
            let sig = d_shell.ao_start + l;
            for k in 0..c_shell.ao_len {
                let lam = c_shell.ao_start + k;
                let col = l * c_shell.ao_len + k;
                for j in 0..b_shell.ao_len {
                    let nu = b_shell.ao_start + j;
                    for i in 0..a_shell.ao_len {
                        let mu = a_shell.ao_start + i;
                        let row = j * a_shell.ao_len + i;
                        set_eri4_symmetry(
                            &mut eri,
                            nao,
                            mu,
                            nu,
                            lam,
                            sig,
                            block_data[col * left_rows + row],
                        );
                    }
                }
            }
        }
    }
    eri
}

/// Full exact Coulomb four-center tensor generated from multi-contraction shell blocks.
///
/// This preserves the AO ordering of `expand_shell_shared_to_rint_shells`
/// (contraction column first, Cartesian component second) while avoiding the
/// front-end split of multi-contraction shells.
pub fn int4c_r_full_from_contracted_shell_blocks(ao_shells: &[RintContractedShell]) -> Vec<f64> {
    let nao = rint_contracted_shell_basis_count(ao_shells);
    let mut eri = vec![0.0_f64; nao * nao * nao * nao];
    let mut block_data = Vec::new();
    let primitive_pair_cache = build_contracted_shell_pair_primitive_pair_cache(ao_shells);
    let single_shells = build_single_contraction_shell_views(ao_shells);
    let single_shell_coeffs = build_rint_shell_coefficient_cache(&single_shells);
    let single_primitive_pair_cache =
        build_shell_pair_primitive_pair_cache(&single_shells, &single_shell_coeffs);
    let mut workspace_cache: HashMap<[u32; 4], RysTransferWorkspace4c> = HashMap::new();
    let mut entry_cache: HashMap<[u32; 4], Vec<Rint4cContractedBlockEntry>> = HashMap::new();
    let mut plan_cache: HashMap<[u32; 4], Rint4cCachedBlockPlan> = HashMap::new();

    for (a_idx, b_idx, c_idx, d_idx) in
        build_unique_4c_shell_quartet_tasks_for_count(ao_shells.len())
    {
        let a_shell = &ao_shells[a_idx];
        let b_shell = &ao_shells[b_idx];
        let c_shell = &ao_shells[c_idx];
        let d_shell = &ao_shells[d_idx];
        let workspace_key = [
            a_shell.l + b_shell.l,
            b_shell.l,
            c_shell.l + d_shell.l,
            d_shell.l,
        ];
        let workspace = workspace_cache.entry(workspace_key).or_insert_with(|| {
            RysTransferWorkspace4c::new(
                workspace_key[0],
                workspace_key[1],
                workspace_key[2],
                workspace_key[3],
            )
        });
        let [left_rows, _right_cols] = if a_shell.nctr() == 1
            && b_shell.nctr() == 1
            && c_shell.nctr() == 1
            && d_shell.nctr() == 1
        {
            int4c_r_shell_block_batched_into_data_with_pairs_entry_cache(
                &single_shells[a_idx],
                &single_shells[b_idx],
                &single_shells[c_idx],
                &single_shells[d_idx],
                &single_primitive_pair_cache[shell_pair_rank(a_idx, b_idx)],
                &single_primitive_pair_cache[shell_pair_rank(c_idx, d_idx)],
                &mut plan_cache,
                &mut block_data,
                workspace,
            )
        } else {
            let ab_pairs = &primitive_pair_cache[shell_pair_rank(a_idx, b_idx)];
            let cd_pairs = &primitive_pair_cache[shell_pair_rank(c_idx, d_idx)];
            let entry_key = [a_shell.l, b_shell.l, c_shell.l, d_shell.l];
            let entries = entry_cache.entry(entry_key).or_insert_with(|| {
                let mut entries = Vec::new();
                build_4c_contracted_block_entries_into(
                    a_shell,
                    b_shell,
                    c_shell,
                    d_shell,
                    &mut entries,
                );
                entries
            });
            int4c_r_contracted_shell_block_batched_into_data_with_pairs_workspace_entries(
                a_shell,
                b_shell,
                c_shell,
                d_shell,
                ab_pairs,
                cd_pairs,
                entries,
                &mut block_data,
                workspace,
            )
        };
        for l in 0..d_shell.ao_len {
            let sig = d_shell.ao_start + l;
            for k in 0..c_shell.ao_len {
                let lam = c_shell.ao_start + k;
                let col = l * c_shell.ao_len + k;
                for j in 0..b_shell.ao_len {
                    let nu = b_shell.ao_start + j;
                    for i in 0..a_shell.ao_len {
                        let mu = a_shell.ao_start + i;
                        let row = j * a_shell.ao_len + i;
                        set_eri4_symmetry(
                            &mut eri,
                            nao,
                            mu,
                            nu,
                            lam,
                            sig,
                            block_data[col * left_rows + row],
                        );
                    }
                }
            }
        }
    }
    eri
}

/// Parallel full exact Coulomb four-center tensor generated from shell-block quartets.
///
/// This computes the same layout as `int4c_r_full_from_shell_blocks`.  Each
/// unique shell-pair quartet is evaluated independently; the final symmetry
/// scatter is synchronized per shell block.
pub fn int4c_r_full_from_shell_blocks_parallel(ao_shells: &[RintShell]) -> Vec<f64> {
    let nao = rint_shell_basis_count(ao_shells);
    let eri = Mutex::new(vec![0.0_f64; nao * nao * nao * nao]);
    let tasks = build_unique_4c_shell_quartet_tasks(ao_shells);

    tasks.par_iter().for_each_init(
        || (Vec::<Rint4cBlockEntry>::new(), Vec::<f64>::new()),
        |(entries, block_data), &(a_idx, b_idx, c_idx, d_idx)| {
            let a_shell = &ao_shells[a_idx];
            let b_shell = &ao_shells[b_idx];
            let c_shell = &ao_shells[c_idx];
            let d_shell = &ao_shells[d_idx];
            let [left_rows, _right_cols] = int4c_r_shell_block_batched_into_data(
                a_shell, b_shell, c_shell, d_shell, entries, block_data,
            );

            let mut eri = eri.lock().expect("failed to lock exact 4c tensor");
            for l in 0..d_shell.ao_len {
                let sig = d_shell.ao_start + l;
                for k in 0..c_shell.ao_len {
                    let lam = c_shell.ao_start + k;
                    let col = l * c_shell.ao_len + k;
                    for j in 0..b_shell.ao_len {
                        let nu = b_shell.ao_start + j;
                        for i in 0..a_shell.ao_len {
                            let mu = a_shell.ao_start + i;
                            let row = j * a_shell.ao_len + i;
                            set_eri4_symmetry(
                                &mut eri,
                                nao,
                                mu,
                                nu,
                                lam,
                                sig,
                                block_data[col * left_rows + row],
                            );
                        }
                    }
                }
            }
        },
    );

    eri.into_inner()
        .expect("failed to unwrap exact 4c tensor mutex")
}

/// Shell-block window for r2 four-center integrals.
///
/// The returned matrix is column-major with shape
/// `[a.ao_len * b.ao_len, c.ao_len * d.ao_len]`.  Row
/// `j * a.ao_len + i` and column `l * c.ao_len + k` store
/// `(a_i b_j | c_k d_l)`.
pub fn int4c_r2_shell_block(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
) -> MatrixFull<f64> {
    int4c_r2_shell_block_batched(a, b, c, d)
}

fn r2_primitive_prefactor(p_sum: f64, q_sum: f64, p_sum_q: f64) -> f64 {
    2.0 * std::f64::consts::PI.powi(3) / ((p_sum * q_sum).sqrt() * p_sum_q)
}

fn add_primitive_4c_r2_shell_block(
    block_data: &mut [f64],
    workspace: &mut RysTransferWorkspace4c,
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
    entries: &[Rint4cBlockEntry],
    ab_pair: &RintPrimitivePair4c,
    cd_pair: &RintPrimitivePair4c,
) {
    let p_sum = ab_pair.exp_sum;
    let q_sum = cd_pair.exp_sum;
    let p_sum_q = p_sum + q_sum;
    let pq_mul = p_sum * q_sum;
    debug_assert!(p_sum > 0.0 && q_sum > 0.0 && p_sum_q > 0.0 && pq_mul > 0.0);

    let p_center = ab_pair.center;
    let q_center = cd_pair.center;
    let rpq2 = distance_squared(&p_center, &q_center);
    let pref = r2_primitive_prefactor(p_sum, q_sum, p_sum_q);
    let primitive_coeff =
        ab_pair.scaled_coeff_over_exp_sum * cd_pair.scaled_coeff_over_exp_sum * pq_mul;
    let rho = pq_mul / p_sum_q;
    let t = rho * rpq2;
    let ni_max = a.shell.ang_type + b.shell.ang_type;
    let nj_max = b.shell.ang_type;
    let nk_max = c.shell.ang_type + d.shell.ang_type;
    let nl_max = d.shell.ang_type;
    let total_ang = a.shell.ang_type + b.shell.ang_type + c.shell.ang_type + d.shell.ang_type;
    let nroots_required = (total_ang / 2 + 1) as usize;
    if ni_max == 0 && nj_max == 0 && nk_max == 0 && nl_max == 0 {
        let term = primitive_coeff * pref * gm_r2_single(0, t);
        if term.is_finite() {
            block_data[0] += term;
        }
        return;
    }

    let (roots, weights) = rys_roots_weights_r2(nroots_required, t);
    if roots.len() != nroots_required || weights.len() != nroots_required {
        return;
    }

    for root_idx in 0..nroots_required {
        let root = roots[root_idx];
        let weight = weights[root_idx];
        let scalars = rys_root_scalars_4c(root, p_sum, q_sum, p_sum_q);
        build_rys_transfer_table_4c_direct(
            &mut workspace.table_x,
            &mut workspace.seed_panel,
            rys_axis_coeffs_4c_from_scalars(
                scalars,
                root,
                p_center[0],
                q_center[0],
                a.center[0],
                c.center[0],
            ),
            ni_max,
            nj_max,
            nk_max,
            nl_max,
            a.center[0],
            b.center[0],
            c.center[0],
            d.center[0],
        );
        build_rys_transfer_table_4c_direct(
            &mut workspace.table_y,
            &mut workspace.seed_panel,
            rys_axis_coeffs_4c_from_scalars(
                scalars,
                root,
                p_center[1],
                q_center[1],
                a.center[1],
                c.center[1],
            ),
            ni_max,
            nj_max,
            nk_max,
            nl_max,
            a.center[1],
            b.center[1],
            c.center[1],
            d.center[1],
        );
        build_rys_transfer_table_4c_direct(
            &mut workspace.table_z,
            &mut workspace.seed_panel,
            rys_axis_coeffs_4c_from_scalars(
                scalars,
                root,
                p_center[2],
                q_center[2],
                a.center[2],
                c.center[2],
            ),
            ni_max,
            nj_max,
            nk_max,
            nl_max,
            a.center[2],
            b.center[2],
            c.center[2],
            d.center[2],
        );

        let primitive_scale = primitive_coeff * pref * weight;
        if primitive_scale == 0.0 || !primitive_scale.is_finite() {
            continue;
        }
        for entry in entries {
            debug_assert!(entry.x_idx < workspace.table_x.data.len());
            debug_assert!(entry.y_idx < workspace.table_y.data.len());
            debug_assert!(entry.z_idx < workspace.table_z.data.len());
            let ix = unsafe { *workspace.table_x.data.get_unchecked(entry.x_idx) };
            let iy = unsafe { *workspace.table_y.data.get_unchecked(entry.y_idx) };
            let iz = unsafe { *workspace.table_z.data.get_unchecked(entry.z_idx) };
            let term = primitive_scale * ix * iy * iz;
            if term.is_finite() {
                block_data[entry.data_idx] += term;
            }
        }
    }
}

fn int4c_r2_shell_block_batched_into_data_with_pairs_workspace_entries(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
    ab_pairs: &[RintPrimitivePair4c],
    cd_pairs: &[RintPrimitivePair4c],
    entries: &[Rint4cBlockEntry],
    block_data: &mut Vec<f64>,
    workspace: &mut RysTransferWorkspace4c,
) -> [usize; 2] {
    let left_rows = a.ao_len * b.ao_len;
    let right_cols = c.ao_len * d.ao_len;
    let shape = [left_rows, right_cols];
    block_data.resize(left_rows * right_cols, 0.0_f64);
    block_data.fill(0.0);

    for ab_pair in ab_pairs {
        for cd_pair in cd_pairs {
            add_primitive_4c_r2_shell_block(
                block_data, workspace, a, b, c, d, entries, ab_pair, cd_pair,
            );
        }
    }
    shape
}

fn int4c_r2_shell_block_batched(
    a: &RintShell,
    b: &RintShell,
    c: &RintShell,
    d: &RintShell,
) -> MatrixFull<f64> {
    let a_coeffs = rint_shell_coefficients(a);
    let b_coeffs = rint_shell_coefficients(b);
    let c_coeffs = rint_shell_coefficients(c);
    let d_coeffs = rint_shell_coefficients(d);
    let rab2 = distance_squared(&a.center, &b.center);
    let rcd2 = distance_squared(&c.center, &d.center);
    let ab_pairs = build_primitive_pairs_4c(a, b, a_coeffs, b_coeffs, rab2);
    let cd_pairs = build_primitive_pairs_4c(c, d, c_coeffs, d_coeffs, rcd2);
    let mut entries = Vec::new();
    build_4c_block_entries_into(a, b, c, d, &mut entries);
    let workspace_key = [
        a.shell.ang_type + b.shell.ang_type,
        b.shell.ang_type,
        c.shell.ang_type + d.shell.ang_type,
        d.shell.ang_type,
    ];
    let mut workspace = RysTransferWorkspace4c::new(
        workspace_key[0],
        workspace_key[1],
        workspace_key[2],
        workspace_key[3],
    );
    let mut data = Vec::new();
    let shape = int4c_r2_shell_block_batched_into_data_with_pairs_workspace_entries(
        a,
        b,
        c,
        d,
        &ab_pairs,
        &cd_pairs,
        &entries,
        &mut data,
        &mut workspace,
    );
    unsafe { MatrixFull::from_vec_unchecked(shape, data) }
}

#[inline(always)]
fn contracted_shell_local_to_ctr_cart(shell: &RintContractedShell, local: usize) -> (usize, usize) {
    (local / shell.cart_len, local % shell.cart_len)
}

fn eri_rint_contracted_shell_4c_r2(
    a: &RintContractedShell,
    a_local: usize,
    b: &RintContractedShell,
    b_local: usize,
    c: &RintContractedShell,
    c_local: usize,
    d: &RintContractedShell,
    d_local: usize,
) -> f64 {
    let (a_ctr, a_cart) = contracted_shell_local_to_ctr_cart(a, a_local);
    let (b_ctr, b_cart) = contracted_shell_local_to_ctr_cart(b, b_local);
    let (c_ctr, c_cart) = contracted_shell_local_to_ctr_cart(c, c_local);
    let (d_ctr, d_cart) = contracted_shell_local_to_ctr_cart(d, d_local);
    let a_coeffs = &a.coeff_columns[a_ctr];
    let b_coeffs = &b.coeff_columns[b_ctr];
    let c_coeffs = &c.coeff_columns[c_ctr];
    let d_coeffs = &d.coeff_columns[d_ctr];
    let la = a.cart_components[a_cart];
    let lb = b.cart_components[b_cart];
    let lc = c.cart_components[c_cart];
    let ld = d.cart_components[d_cart];
    let nroots = calculate_nroots_from_angmom(la, lb, lc, ld);
    let rab2 = distance_squared(&a.center, &b.center);
    let rcd2 = distance_squared(&c.center, &d.center);
    let mut eri = 0.0_f64;
    for (&alpha, &a_coeff) in a.exponents.iter().zip(a_coeffs.iter()) {
        for (&beta, &b_coeff) in b.exponents.iter().zip(b_coeffs.iter()) {
            let p_center = gaussian_product_center(alpha, beta, &a.center, &b.center);
            for (&gamma, &c_coeff) in c.exponents.iter().zip(c_coeffs.iter()) {
                for (&delta, &d_coeff) in d.exponents.iter().zip(d_coeffs.iter()) {
                    let q_center = gaussian_product_center(gamma, delta, &c.center, &d.center);
                    let rpq2 = distance_squared(&p_center, &q_center);
                    let eri_prim = gaussian_4c_r2(
                        nroots,
                        &alpha,
                        &beta,
                        &gamma,
                        &delta,
                        &rab2,
                        &rcd2,
                        &rpq2,
                        la,
                        lb,
                        lc,
                        ld,
                        &p_center,
                        &q_center,
                        &a.center,
                        &b.center,
                        &c.center,
                        &d.center,
                        a.ao_start + a_local,
                        b.ao_start + b_local,
                        c.ao_start + c_local,
                        d.ao_start + d_local,
                    );
                    let term = a_coeff * b_coeff * c_coeff * d_coeff * eri_prim;
                    if term.is_finite() {
                        eri += term;
                    }
                }
            }
        }
    }
    eri
}

fn add_primitive_4c_r2_contracted_shell_block(
    block_data: &mut [f64],
    workspace: &mut RysTransferWorkspace4c,
    a: &RintContractedShell,
    b: &RintContractedShell,
    c: &RintContractedShell,
    d: &RintContractedShell,
    entries: &[Rint4cContractedBlockEntry],
    ab_pair: &RintContractedPrimitivePair4c,
    cd_pair: &RintContractedPrimitivePair4c,
) {
    let p_sum = ab_pair.exp_sum;
    let q_sum = cd_pair.exp_sum;
    let p_sum_q = p_sum + q_sum;
    let pq_mul = p_sum * q_sum;
    debug_assert!(p_sum > 0.0 && q_sum > 0.0 && p_sum_q > 0.0 && pq_mul > 0.0);

    let p_center = ab_pair.center;
    let q_center = cd_pair.center;
    let rpq2 = distance_squared(&p_center, &q_center);
    let pref = r2_primitive_prefactor(p_sum, q_sum, p_sum_q);
    let primitive_coeff =
        ab_pair.scaled_exp_over_exp_sum * cd_pair.scaled_exp_over_exp_sum * pq_mul;
    let rho = pq_mul / p_sum_q;
    let t = rho * rpq2;
    let ni_max = a.l + b.l;
    let nj_max = b.l;
    let nk_max = c.l + d.l;
    let nl_max = d.l;
    let total_ang = a.l + b.l + c.l + d.l;
    let nroots_required = (total_ang / 2 + 1) as usize;
    if ni_max == 0 && nj_max == 0 && nk_max == 0 && nl_max == 0 {
        let cart_term = primitive_coeff * pref * gm_r2_single(0, t);
        if let Some(entry) = entries.first() {
            add_contracted_4c_cart_term(block_data, a, b, c, d, entry, ab_pair, cd_pair, cart_term);
        }
        return;
    }

    let (roots, weights) = rys_roots_weights_r2(nroots_required, t);
    if roots.len() != nroots_required || weights.len() != nroots_required {
        return;
    }

    for root_idx in 0..nroots_required {
        let root = roots[root_idx];
        let weight = weights[root_idx];
        let scalars = rys_root_scalars_4c(root, p_sum, q_sum, p_sum_q);
        build_rys_transfer_table_4c_direct(
            &mut workspace.table_x,
            &mut workspace.seed_panel,
            rys_axis_coeffs_4c_from_scalars(
                scalars,
                root,
                p_center[0],
                q_center[0],
                a.center[0],
                c.center[0],
            ),
            ni_max,
            nj_max,
            nk_max,
            nl_max,
            a.center[0],
            b.center[0],
            c.center[0],
            d.center[0],
        );
        build_rys_transfer_table_4c_direct(
            &mut workspace.table_y,
            &mut workspace.seed_panel,
            rys_axis_coeffs_4c_from_scalars(
                scalars,
                root,
                p_center[1],
                q_center[1],
                a.center[1],
                c.center[1],
            ),
            ni_max,
            nj_max,
            nk_max,
            nl_max,
            a.center[1],
            b.center[1],
            c.center[1],
            d.center[1],
        );
        build_rys_transfer_table_4c_direct(
            &mut workspace.table_z,
            &mut workspace.seed_panel,
            rys_axis_coeffs_4c_from_scalars(
                scalars,
                root,
                p_center[2],
                q_center[2],
                a.center[2],
                c.center[2],
            ),
            ni_max,
            nj_max,
            nk_max,
            nl_max,
            a.center[2],
            b.center[2],
            c.center[2],
            d.center[2],
        );

        let primitive_scale = primitive_coeff * pref * weight;
        if primitive_scale == 0.0 || !primitive_scale.is_finite() {
            continue;
        }
        for entry in entries {
            debug_assert!(entry.x_idx < workspace.table_x.data.len());
            debug_assert!(entry.y_idx < workspace.table_y.data.len());
            debug_assert!(entry.z_idx < workspace.table_z.data.len());
            let ix = unsafe { *workspace.table_x.data.get_unchecked(entry.x_idx) };
            let iy = unsafe { *workspace.table_y.data.get_unchecked(entry.y_idx) };
            let iz = unsafe { *workspace.table_z.data.get_unchecked(entry.z_idx) };
            let cart_term = primitive_scale * ix * iy * iz;
            add_contracted_4c_cart_term(block_data, a, b, c, d, entry, ab_pair, cd_pair, cart_term);
        }
    }
}

fn int4c_r2_contracted_shell_block_batched_into_data_with_pairs_workspace_entries(
    a: &RintContractedShell,
    b: &RintContractedShell,
    c: &RintContractedShell,
    d: &RintContractedShell,
    ab_pairs: &[RintContractedPrimitivePair4c],
    cd_pairs: &[RintContractedPrimitivePair4c],
    entries: &[Rint4cContractedBlockEntry],
    block_data: &mut Vec<f64>,
    workspace: &mut RysTransferWorkspace4c,
) -> [usize; 2] {
    let left_rows = a.ao_len * b.ao_len;
    let right_cols = c.ao_len * d.ao_len;
    let shape = [left_rows, right_cols];
    block_data.resize(left_rows * right_cols, 0.0_f64);
    block_data.fill(0.0);

    for ab_pair in ab_pairs {
        for cd_pair in cd_pairs {
            add_primitive_4c_r2_contracted_shell_block(
                block_data, workspace, a, b, c, d, entries, ab_pair, cd_pair,
            );
        }
    }
    shape
}

/// Shell-block window for contracted-shell r2 four-center integrals.
///
/// The AO order matches `expand_shell_shared_to_rint_shells`: contraction
/// column first, Cartesian component second.
pub fn int4c_r2_contracted_shell_block(
    a: &RintContractedShell,
    b: &RintContractedShell,
    c: &RintContractedShell,
    d: &RintContractedShell,
) -> MatrixFull<f64> {
    let ab_pairs =
        build_contracted_primitive_pairs_4c(a, b, distance_squared(&a.center, &b.center));
    let cd_pairs =
        build_contracted_primitive_pairs_4c(c, d, distance_squared(&c.center, &d.center));
    let mut entries = Vec::new();
    build_4c_contracted_block_entries_into(a, b, c, d, &mut entries);
    let workspace_key = [a.l + b.l, b.l, c.l + d.l, d.l];
    let mut workspace = RysTransferWorkspace4c::new(
        workspace_key[0],
        workspace_key[1],
        workspace_key[2],
        workspace_key[3],
    );
    let mut data = Vec::new();
    let shape = int4c_r2_contracted_shell_block_batched_into_data_with_pairs_workspace_entries(
        a,
        b,
        c,
        d,
        &ab_pairs,
        &cd_pairs,
        &entries,
        &mut data,
        &mut workspace,
    );
    unsafe { MatrixFull::from_vec_unchecked(shape, data) }
}

/// Full R2 four-center tensor generated from split shell-block quartets.
pub fn int4c_r2_full_from_shell_blocks(ao_shells: &[RintShell]) -> Vec<f64> {
    let nao = rint_shell_basis_count(ao_shells);
    let mut eri = vec![0.0_f64; nao * nao * nao * nao];
    let mut block_data = Vec::new();
    let shell_coeffs = build_rint_shell_coefficient_cache(ao_shells);
    let primitive_pair_cache = build_shell_pair_primitive_pair_cache(ao_shells, &shell_coeffs);
    let mut workspace_cache: HashMap<[u32; 4], RysTransferWorkspace4c> = HashMap::new();
    let mut entry_cache: HashMap<[u32; 4], Vec<Rint4cBlockEntry>> = HashMap::new();

    for (a_idx, b_idx, c_idx, d_idx) in build_unique_4c_shell_quartet_tasks(ao_shells) {
        let a_shell = &ao_shells[a_idx];
        let b_shell = &ao_shells[b_idx];
        let c_shell = &ao_shells[c_idx];
        let d_shell = &ao_shells[d_idx];
        let ab_pairs = &primitive_pair_cache[shell_pair_rank(a_idx, b_idx)];
        let cd_pairs = &primitive_pair_cache[shell_pair_rank(c_idx, d_idx)];
        let workspace_key = [
            a_shell.shell.ang_type + b_shell.shell.ang_type,
            b_shell.shell.ang_type,
            c_shell.shell.ang_type + d_shell.shell.ang_type,
            d_shell.shell.ang_type,
        ];
        let workspace = workspace_cache.entry(workspace_key).or_insert_with(|| {
            RysTransferWorkspace4c::new(
                workspace_key[0],
                workspace_key[1],
                workspace_key[2],
                workspace_key[3],
            )
        });
        let entry_key = [
            a_shell.shell.ang_type,
            b_shell.shell.ang_type,
            c_shell.shell.ang_type,
            d_shell.shell.ang_type,
        ];
        let entries = entry_cache.entry(entry_key).or_insert_with(|| {
            let mut entries = Vec::new();
            build_4c_block_entries_into(a_shell, b_shell, c_shell, d_shell, &mut entries);
            entries
        });
        let left_rows = a_shell.ao_len * b_shell.ao_len;
        int4c_r2_shell_block_batched_into_data_with_pairs_workspace_entries(
            a_shell,
            b_shell,
            c_shell,
            d_shell,
            ab_pairs,
            cd_pairs,
            entries,
            &mut block_data,
            workspace,
        );
        for l in 0..d_shell.ao_len {
            let sig = d_shell.ao_start + l;
            for k in 0..c_shell.ao_len {
                let lam = c_shell.ao_start + k;
                let col = l * c_shell.ao_len + k;
                for j in 0..b_shell.ao_len {
                    let nu = b_shell.ao_start + j;
                    for i in 0..a_shell.ao_len {
                        let mu = a_shell.ao_start + i;
                        let row = j * a_shell.ao_len + i;
                        set_eri4_symmetry(
                            &mut eri,
                            nao,
                            mu,
                            nu,
                            lam,
                            sig,
                            block_data[col * left_rows + row],
                        );
                    }
                }
            }
        }
    }
    eri
}

/// Full R2 four-center tensor generated from contracted shell-block quartets.
pub fn int4c_r2_full_from_contracted_shell_blocks(ao_shells: &[RintContractedShell]) -> Vec<f64> {
    let nao = rint_contracted_shell_basis_count(ao_shells);
    let mut eri = vec![0.0_f64; nao * nao * nao * nao];
    let mut block_data = Vec::new();
    let primitive_pair_cache = build_contracted_shell_pair_primitive_pair_cache(ao_shells);
    let single_shells = build_single_contraction_shell_views(ao_shells);
    let single_shell_coeffs = build_rint_shell_coefficient_cache(&single_shells);
    let single_primitive_pair_cache =
        build_shell_pair_primitive_pair_cache(&single_shells, &single_shell_coeffs);
    let mut workspace_cache: HashMap<[u32; 4], RysTransferWorkspace4c> = HashMap::new();
    let mut entry_cache: HashMap<[u32; 4], Vec<Rint4cContractedBlockEntry>> = HashMap::new();
    let mut single_entry_cache: HashMap<[u32; 4], Vec<Rint4cBlockEntry>> = HashMap::new();

    for (a_idx, b_idx, c_idx, d_idx) in
        build_unique_4c_shell_quartet_tasks_for_count(ao_shells.len())
    {
        let a_shell = &ao_shells[a_idx];
        let b_shell = &ao_shells[b_idx];
        let c_shell = &ao_shells[c_idx];
        let d_shell = &ao_shells[d_idx];
        let workspace_key = [
            a_shell.l + b_shell.l,
            b_shell.l,
            c_shell.l + d_shell.l,
            d_shell.l,
        ];
        let workspace = workspace_cache.entry(workspace_key).or_insert_with(|| {
            RysTransferWorkspace4c::new(
                workspace_key[0],
                workspace_key[1],
                workspace_key[2],
                workspace_key[3],
            )
        });
        let left_rows = a_shell.ao_len * b_shell.ao_len;
        if a_shell.nctr() == 1 && b_shell.nctr() == 1 && c_shell.nctr() == 1 && d_shell.nctr() == 1
        {
            let a_single = &single_shells[a_idx];
            let b_single = &single_shells[b_idx];
            let c_single = &single_shells[c_idx];
            let d_single = &single_shells[d_idx];
            let entry_key = [
                a_single.shell.ang_type,
                b_single.shell.ang_type,
                c_single.shell.ang_type,
                d_single.shell.ang_type,
            ];
            let entries = single_entry_cache.entry(entry_key).or_insert_with(|| {
                let mut entries = Vec::new();
                build_4c_block_entries_into(a_single, b_single, c_single, d_single, &mut entries);
                entries
            });
            int4c_r2_shell_block_batched_into_data_with_pairs_workspace_entries(
                a_single,
                b_single,
                c_single,
                d_single,
                &single_primitive_pair_cache[shell_pair_rank(a_idx, b_idx)],
                &single_primitive_pair_cache[shell_pair_rank(c_idx, d_idx)],
                entries,
                &mut block_data,
                workspace,
            );
        } else {
            let ab_pairs = &primitive_pair_cache[shell_pair_rank(a_idx, b_idx)];
            let cd_pairs = &primitive_pair_cache[shell_pair_rank(c_idx, d_idx)];
            let entry_key = [a_shell.l, b_shell.l, c_shell.l, d_shell.l];
            let entries = entry_cache.entry(entry_key).or_insert_with(|| {
                let mut entries = Vec::new();
                build_4c_contracted_block_entries_into(
                    a_shell,
                    b_shell,
                    c_shell,
                    d_shell,
                    &mut entries,
                );
                entries
            });
            int4c_r2_contracted_shell_block_batched_into_data_with_pairs_workspace_entries(
                a_shell,
                b_shell,
                c_shell,
                d_shell,
                ab_pairs,
                cd_pairs,
                entries,
                &mut block_data,
                workspace,
            );
        }
        for l in 0..d_shell.ao_len {
            let sig = d_shell.ao_start + l;
            for k in 0..c_shell.ao_len {
                let lam = c_shell.ao_start + k;
                let col = l * c_shell.ao_len + k;
                for j in 0..b_shell.ao_len {
                    let nu = b_shell.ao_start + j;
                    for i in 0..a_shell.ao_len {
                        let mu = a_shell.ao_start + i;
                        let row = j * a_shell.ao_len + i;
                        set_eri4_symmetry(
                            &mut eri,
                            nao,
                            mu,
                            nu,
                            lam,
                            sig,
                            block_data[col * left_rows + row],
                        );
                    }
                }
            }
        }
    }
    eri
}
#[inline(always)]
fn c_mu_p(c: &[MatrixFull<f64>; 2], spin: usize, mu: usize, p: usize) -> f64 {
    *c[spin]
        .get(&[mu, p])
        .expect("MO coefficient index out of bounds")
}
//chmical sign : (pq|rs)
pub fn eri_mo_4c_r(
    bfs: &[BasisFunction],
    c: &[MatrixFull<f64>; 2],
    spin: usize,
    p: usize,
    q: usize,
    r: usize,
    s: usize,
) -> f64 {
    assert!(spin < 2);
    let nao = bfs.len();
    let mut val = 0.0_f64;
    for mu in 0..nao {
        let c1 = c_mu_p(c, spin, mu, p);
        if c1 == 0.0 {
            continue;
        }
        for nu in 0..nao {
            let c2 = c_mu_p(c, spin, nu, q);
            let w12 = c1 * c2;
            if w12 == 0.0 {
                continue;
            }
            for lam in 0..nao {
                let c3 = c_mu_p(c, spin, lam, r);
                if c3 == 0.0 {
                    continue;
                }
                for sig in 0..nao {
                    let c4 = c_mu_p(c, spin, sig, s);
                    let w = w12 * c3 * c4;
                    if w == 0.0 {
                        continue;
                    }
                    let eri_ao = eri_ao_4c_r(bfs, mu, nu, lam, sig);
                    let term = w * eri_ao;
                    if term.is_finite() {
                        val += term;
                    }
                }
            }
        }
    }
    val
}
//chmical sign : (pq|rs)
pub fn eri_mo_4c_r2(
    bfs: &[BasisFunction],
    c: &[MatrixFull<f64>; 2],
    spin: usize,
    p: usize,
    q: usize,
    r: usize,
    s: usize,
) -> f64 {
    assert!(spin < 2);
    let nao = bfs.len();
    let mut val = 0.0_f64;
    for mu in 0..nao {
        let c1 = c_mu_p(c, spin, mu, p);
        if c1 == 0.0 {
            continue;
        }
        for nu in 0..nao {
            let c2 = c_mu_p(c, spin, nu, q);
            let w12 = c1 * c2;
            if w12 == 0.0 {
                continue;
            }
            for lam in 0..nao {
                let c3 = c_mu_p(c, spin, lam, r);
                if c3 == 0.0 {
                    continue;
                }
                for sig in 0..nao {
                    let c4 = c_mu_p(c, spin, sig, s);
                    let w = w12 * c3 * c4;
                    if w == 0.0 {
                        continue;
                    }
                    let eri_ao = eri_ao_4c_r2(bfs, mu, nu, lam, sig);
                    let term = w * eri_ao;
                    if term.is_finite() {
                        val += term;
                    }
                }
            }
        }
    }
    val
}
// ==============================
// Generic JK builder (kernel as fn ptr)
// ==============================
pub type EriAoKernel = fn(&[BasisFunction], usize, usize, usize, usize) -> f64;
pub fn build_jk_from_p_kernel(
    bfs: &[BasisFunction],
    p: &MatrixFull<f64>, // density used for contraction
    eri: EriAoKernel,    // eri_ao_4c_r or eri_ao_4c_r2
) -> (MatrixFull<f64>, MatrixFull<f64>) {
    let nao = bfs.len();
    let mut j = MatrixFull::new([nao, nao], 0.0);
    let mut k = MatrixFull::new([nao, nao], 0.0);
    for mu in 0..nao {
        for nu in 0..nao {
            let mut j_munu = 0.0_f64;
            let mut k_munu = 0.0_f64;
            for lam in 0..nao {
                for sig in 0..nao {
                    let p_lamsig = *p.get(&[lam, sig]).unwrap();
                    j_munu += p_lamsig * eri(bfs, mu, nu, lam, sig);
                    k_munu += p_lamsig * eri(bfs, mu, lam, nu, sig);
                }
            }
            j.set2d([mu, nu], j_munu);
            k.set2d([mu, nu], k_munu);
        }
    }
    (j, k)
}

pub fn build_jk_from_full_4c_tensor(
    eri4: &[f64],
    p: &MatrixFull<f64>,
    nao: usize,
) -> (MatrixFull<f64>, MatrixFull<f64>) {
    assert_eq!(
        p.size,
        [nao, nao],
        "density shape must match full 4c tensor AO dimension"
    );
    assert_eq!(
        eri4.len(),
        nao * nao * nao * nao,
        "full 4c tensor length must be nao^4"
    );
    let mut j = MatrixFull::new([nao, nao], 0.0);
    let mut k = MatrixFull::new([nao, nao], 0.0);
    for mu in 0..nao {
        for nu in 0..nao {
            let mut j_munu = 0.0_f64;
            let mut k_munu = 0.0_f64;
            for lam in 0..nao {
                for sig in 0..nao {
                    let p_lamsig = *p.get(&[lam, sig]).unwrap();
                    j_munu += p_lamsig * eri4[int4c_r_full_index(mu, nu, lam, sig, nao)];
                    k_munu += p_lamsig * eri4[int4c_r_full_index(mu, lam, nu, sig, nao)];
                }
            }
            j.set2d([mu, nu], j_munu);
            k.set2d([mu, nu], k_munu);
        }
    }
    (j, k)
}

pub fn build_jk_from_r2_contracted_shell_block_full_tensor(
    ao_shells: &[RintContractedShell],
    p: &MatrixFull<f64>,
) -> (MatrixFull<f64>, MatrixFull<f64>) {
    let nao = rint_contracted_shell_basis_count(ao_shells);
    let eri4 = int4c_r2_full_from_contracted_shell_blocks(ao_shells);
    build_jk_from_full_4c_tensor(&eri4, p, nao)
}

pub fn build_jk_from_r2_shell_block_full_tensor(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    p_cart: &MatrixFull<f64>,
) -> (MatrixFull<f64>, MatrixFull<f64>) {
    let ao_shells =
        crate::lib_rint::basis::load_molecule_contracted_rint_shells_from_raw(geom, basis4elem)
            .expect("failed to build contracted rint shells from GeomCell/BasCell");
    build_jk_from_r2_contracted_shell_block_full_tensor(&ao_shells, p_cart)
}
#[inline(always)]
fn dot2(a: &MatrixFull<f64>, b: &MatrixFull<f64>) -> f64 {
    let [nrow, ncol] = a.size;
    assert_eq!([nrow, ncol], b.size);
    let mut s = 0.0_f64;
    for i in 0..nrow {
        for j in 0..ncol {
            s += a.get(&[i, j]).unwrap() * b.get(&[i, j]).unwrap();
        }
    }
    s
}
// ==============================
// Energy from P, J, K (RHF convention)
// ==============================
// Assumes RHF total density P includes a factor of 2:
//   P_{mu,nu} = 2 * sum_{i in occ} C_{mu i} C_{nu i}
// Then:
//   E = 1/2 <P,J> - 1/4 <P,K>
//
// If your RHF P does NOT include factor 2, use instead:
//   E = 1.0 <P,J> - 0.5 <P,K>
#[inline(always)]
pub fn ej_ek_from_p_jk_rhf(
    p: &MatrixFull<f64>,
    j: &MatrixFull<f64>,
    k: &MatrixFull<f64>,
) -> (f64, f64, f64) {
    let ej = 0.5 * dot2(p, j);
    let ek = -0.25 * dot2(p, k);
    (ej, ek, ej + ek)
}
#[inline(always)]
pub fn ej_ek_from_p_jk_uhf(
    p_spin: &[MatrixFull<f64>],
    j_total: &MatrixFull<f64>,
    k_spin: &[MatrixFull<f64>],
) -> (f64, f64, f64) {
    assert_eq!(
        p_spin.len(),
        k_spin.len(),
        "UHF energy contraction expects one K matrix for each spin density"
    );
    // `dot2` contracts full symmetric matrices, so each off-diagonal term is
    // counted twice compared with SCF::par_energy_contraction on MatrixUpper.
    let ej = 0.25 * p_spin.iter().map(|p| dot2(p, j_total)).sum::<f64>();
    let ek = -0.25
        * p_spin
            .iter()
            .zip(k_spin.iter())
            .map(|(p, k)| dot2(p, k))
            .sum::<f64>();
    (ej, ek, ej + ek)
}
#[inline(always)]
pub fn jk_pair_square_sum(j: &MatrixFull<f64>, k: &MatrixFull<f64>) -> f64 {
    let [nrow, ncol] = j.size;
    assert_eq!([nrow, ncol], k.size);
    let mut s = 0.0_f64;
    for i in 0..nrow {
        for p in 0..ncol {
            let d = j.get(&[i, p]).unwrap() - k.get(&[i, p]).unwrap();
            let t = d * d;
            if t.is_finite() {
                s += t;
            }
        }
    }
    s
}
// ==============================
// Convenience: build JK and compute E for r and r2 kernels (RHF)
// ==============================
#[inline(always)]
fn baspair_index(i: usize, j: usize) -> usize {
    if i < j {
        (j + 1) * j / 2 + i
    } else {
        (i + 1) * i / 2 + j
    }
}
fn prepare_baspair_map(num_basis: usize) -> (MatrixFull<usize>, Vec<[usize; 2]>) {
    let num_baspar = (num_basis + 1) * num_basis / 2;
    let mut basbas2baspar = MatrixFull::new([num_basis, num_basis], 0_usize);
    let mut baspar2basbas = vec![[0_usize; 2]; num_baspar];
    for j in 0..num_basis {
        for i in 0..num_basis {
            let baspar_ind = baspair_index(i, j);
            basbas2baspar.set2d([i, j], baspar_ind);
            baspar2basbas[baspar_ind] = [i, j];
        }
    }
    (basbas2baspar, baspar2basbas)
}
fn contiguous_partitions(len: usize, batch_size: usize) -> Vec<[usize; 2]> {
    assert!(batch_size > 0, "batch_size should be larger than zero");
    let mut parts = Vec::new();
    let mut start = 0;
    while start < len {
        let end = (start + batch_size).min(len);
        parts.push([start, end]);
        start = end;
    }
    parts
}
fn pack_density_for_ri_j(dm: &[MatrixFull<f64>]) -> MatrixFull<f64> {
    let spin_channel = dm.len();
    let nao = dm[0].size[0];
    let num_baspar = (nao + 1) * nao / 2;
    let mut packed = MatrixFull::new([num_baspar, spin_channel], 0.0_f64);
    for (iset, dm_s) in dm.iter().enumerate() {
        let mut ind = 0;
        for j in 0..nao {
            for i in 0..=j {
                let factor = if i == j { 1.0 } else { 2.0 };
                packed.data[iset * num_baspar + ind] = factor * dm_s.get(&[i, j]).unwrap();
                ind += 1;
            }
        }
    }
    packed
}
fn vj_upper_with_rimatr_sync_internal(
    ri3fn: &Option<(MatrixFull<f64>, MatrixFull<usize>, Vec<[usize; 2]>)>,
    dm: &Vec<MatrixFull<f64>>,
    spin_channel: usize,
    scaling_factor: f64,
) -> Vec<MatrixUpper<f64>> {
    let mut vj = vec![MatrixUpper::new(1, 0.0_f64); spin_channel];
    if let Some((ri3fn, _, _)) = ri3fn {
        let num_baspar = ri3fn.size[0];
        let num_auxbas = ri3fn.size[1];
        for i_spin in 0..spin_channel {
            let mut dm_s_upper = MatrixUpper::from_vec(
                num_baspar,
                dm[i_spin].iter_matrixupper().unwrap().map(|x| *x).collect(),
            )
            .unwrap();
            dm_s_upper.iter_diagonal_mut().for_each(|x| *x *= 0.5);
            let mut tmp_v = vec![0.0_f64; num_auxbas];
            for aux in 0..num_auxbas {
                let mut acc = 0.0_f64;
                for baspar in 0..num_baspar {
                    acc += ri3fn.get(&[baspar, aux]).unwrap() * dm_s_upper.data[baspar];
                }
                tmp_v[aux] = 2.0 * acc;
            }
            let mut vj_spin = MatrixUpper::new(num_baspar, 0.0_f64);
            for baspar in 0..num_baspar {
                let mut acc = 0.0_f64;
                for aux in 0..num_auxbas {
                    acc += ri3fn.get(&[baspar, aux]).unwrap() * tmp_v[aux];
                }
                vj_spin.data[baspar] = acc * scaling_factor;
            }
            vj[i_spin] = vj_spin;
        }
    }
    vj
}
fn vk_upper_with_rimatr_sync_internal(
    ri3fn: &Option<(MatrixFull<f64>, MatrixFull<usize>, Vec<[usize; 2]>)>,
    dm: &Vec<MatrixFull<f64>>,
    spin_channel: usize,
    scaling_factor: f64,
) -> Vec<MatrixUpper<f64>> {
    let mut vk = vec![MatrixUpper::new(1, 0.0_f64); spin_channel];
    if let Some((ri3fn, basbas2baspar, baspar2basbas)) = ri3fn {
        let num_basis = basbas2baspar.size[0];
        let num_auxbas = ri3fn.size[1];
        for i_spin in 0..spin_channel {
            let Some(dm_sqrt) = dm[i_spin].clone().lapack_power(0.5, SQRT_THRESHOLD) else {
                continue;
            };
            let mut vk_full = MatrixFull::new([num_basis, num_basis], 0.0_f64);
            for aux in 0..num_auxbas {
                let mut b_p = MatrixFull::new([num_basis, num_basis], 0.0_f64);
                for (baspar, [mu, nu]) in baspar2basbas.iter().enumerate() {
                    let value = *ri3fn.get(&[baspar, aux]).unwrap();
                    b_p.set2d([*mu, *nu], value);
                    if mu != nu {
                        b_p.set2d([*nu, *mu], value);
                    }
                }
                let mut tmp = MatrixFull::new([num_basis, num_basis], 0.0_f64);
                tmp.to_matrixfullslicemut().lapack_dgemm(
                    &b_p.to_matrixfullslice(),
                    &dm_sqrt.to_matrixfullslice(),
                    'N',
                    'N',
                    1.0,
                    0.0,
                );
                let mut vk_p = MatrixFull::new([num_basis, num_basis], 0.0_f64);
                vk_p.to_matrixfullslicemut().lapack_dgemm(
                    &tmp.to_matrixfullslice(),
                    &tmp.to_matrixfullslice(),
                    'N',
                    'T',
                    1.0,
                    0.0,
                );
                vk_full
                    .data
                    .iter_mut()
                    .zip(vk_p.data.iter())
                    .for_each(|(to, from)| *to += *from);
            }
            if scaling_factor != 1.0_f64 {
                vk_full.data.iter_mut().for_each(|f| *f *= scaling_factor);
            }
            vk[i_spin] = vk_full.to_matrixupper();
        }
    }
    vk
}

/// Density-linear RI exchange contraction used by variational R2 energies.
///
/// The generic density-only K builder factors D as sqrt(D)sqrt(D)^T.  That is
/// efficient for positive semidefinite SCF densities, but eigenvalue
/// thresholding makes the map nonlinear at the occupied/virtual boundary and
/// breaks finite-difference derivatives.  Contracting B_P D B_P^T directly is
/// algebraically equivalent for physical densities and remains linear for
/// arbitrary symmetric perturbations.
fn vk_upper_with_rimatr_r2_linear_sync_internal(
    ri3fn: &Option<(MatrixFull<f64>, MatrixFull<usize>, Vec<[usize; 2]>)>,
    dm: &Vec<MatrixFull<f64>>,
    spin_channel: usize,
    scaling_factor: f64,
) -> Vec<MatrixUpper<f64>> {
    let mut vk = vec![MatrixUpper::new(1, 0.0_f64); spin_channel];
    if let Some((ri3fn, basbas2baspar, baspar2basbas)) = ri3fn {
        let num_basis = basbas2baspar.size[0];
        let num_auxbas = ri3fn.size[1];
        for i_spin in 0..spin_channel {
            let mut vk_full = MatrixFull::new([num_basis, num_basis], 0.0_f64);
            for aux in 0..num_auxbas {
                let mut b_p = MatrixFull::new([num_basis, num_basis], 0.0_f64);
                for (baspar, [mu, nu]) in baspar2basbas.iter().enumerate() {
                    let value = *ri3fn.get(&[baspar, aux]).unwrap();
                    b_p.set2d([*mu, *nu], value);
                    if mu != nu {
                        b_p.set2d([*nu, *mu], value);
                    }
                }
                let mut tmp = MatrixFull::new([num_basis, num_basis], 0.0_f64);
                tmp.to_matrixfullslicemut().lapack_dgemm(
                    &b_p.to_matrixfullslice(),
                    &dm[i_spin].to_matrixfullslice(),
                    'N',
                    'N',
                    1.0,
                    0.0,
                );
                let mut vk_p = MatrixFull::new([num_basis, num_basis], 0.0_f64);
                vk_p.to_matrixfullslicemut().lapack_dgemm(
                    &tmp.to_matrixfullslice(),
                    &b_p.to_matrixfullslice(),
                    'N',
                    'T',
                    1.0,
                    0.0,
                );
                vk_full
                    .data
                    .iter_mut()
                    .zip(vk_p.data.iter())
                    .for_each(|(to, from)| *to += *from);
            }
            if scaling_factor != 1.0_f64 {
                vk_full
                    .data
                    .iter_mut()
                    .for_each(|value| *value *= scaling_factor);
            }
            vk[i_spin] = vk_full.to_matrixupper();
        }
    }
    vk
}
pub fn prepare_rimatr_for_r_sync(
    ao_bfs: &[BasisFunction],
    aux_bfs: &[BasisFunction],
) -> Option<(MatrixFull<f64>, MatrixFull<usize>, Vec<[usize; 2]>)> {
    if ao_bfs.is_empty() || aux_bfs.is_empty() {
        return None;
    }
    let num_basis = ao_bfs.len();
    let num_auxbas = aux_bfs.len();
    let num_baspar = (num_basis + 1) * num_basis / 2;
    let (basbas2baspar, baspar2basbas) = prepare_baspair_map(num_basis);
    let mut aux_v = MatrixFull::new([num_auxbas, num_auxbas], 0.0_f64);
    for q in 0..num_auxbas {
        for p in 0..=q {
            let value = eri_ao_2c_r(aux_bfs, p, q);
            aux_v.set2d([p, q], value);
            aux_v.set2d([q, p], value);
        }
    }
    let aux_v_inv_sqrt = aux_v.lapack_power(-0.5, AUXBAS_THRESHOLD)?;
    let mut raw_ri3fn = MatrixFull::new([num_baspar, num_auxbas], 0.0_f64);
    for nu in 0..num_basis {
        for mu in 0..=nu {
            let baspar = baspair_index(mu, nu);
            for aux in 0..num_auxbas {
                raw_ri3fn.set2d([baspar, aux], eri_ao_3c_r(ao_bfs, aux_bfs, mu, nu, aux));
            }
        }
    }
    let mut rimatr = MatrixFull::new([num_baspar, num_auxbas], 0.0_f64);
    rimatr.to_matrixfullslicemut().lapack_dgemm(
        &raw_ri3fn.to_matrixfullslice(),
        &aux_v_inv_sqrt.to_matrixfullslice(),
        'N',
        'N',
        1.0,
        0.0,
    );
    Some((rimatr, basbas2baspar, baspar2basbas))
}
pub fn prepare_rimatr_for_r(
    ao_bfs: &[BasisFunction],
    aux_bfs: &[BasisFunction],
) -> Option<(MatrixFull<f64>, MatrixFull<usize>, Vec<[usize; 2]>)> {
    prepare_rimatr_for_r_sync(ao_bfs, aux_bfs)
}
type Int2cShellBlockFn = fn(&RintShell, &RintShell) -> MatrixFull<f64>;
type Int3cShellBlockIntoFn =
    fn(&RintShell, &RintShell, &RintShell, &[Rint3cBlockEntry], &mut MatrixFull<f64>);

#[derive(Clone, Copy)]
struct RiShellBlockKernel {
    int2c: Int2cShellBlockFn,
    int3c_into: Int3cShellBlockIntoFn,
}

fn rint_shell_basis_count(shells: &[RintShell]) -> usize {
    shells
        .iter()
        .map(|shell| shell.ao_start + shell.ao_len)
        .max()
        .unwrap_or(0)
}

fn rint_contracted_shell_basis_count(shells: &[RintContractedShell]) -> usize {
    shells
        .iter()
        .map(|shell| shell.ao_start + shell.ao_len)
        .max()
        .unwrap_or(0)
}

fn build_aux_metric_from_shell_blocks_parallel(
    aux_shells: &[RintShell],
    num_auxbas: usize,
    kernel: RiShellBlockKernel,
) -> MatrixFull<f64> {
    let shell_pairs: Vec<(usize, usize)> = aux_shells
        .iter()
        .enumerate()
        .flat_map(|(q_shell_idx, _)| {
            (0..=q_shell_idx).map(move |p_shell_idx| (p_shell_idx, q_shell_idx))
        })
        .collect();
    // semi-direct 的 2c metric 也使用外层 worker；默认跟随 ctrl.num_threads
    // 初始化后的 Rayon 线程数，避免在调度器暴露更多 CPU 时误开过多线程。
    let requested_threads = r2_direct_requested_workers();
    let num_workers = requested_threads.min(shell_pairs.len()).max(1);
    let next_pair = AtomicUsize::new(0);

    let blocks: Vec<(usize, usize, MatrixFull<f64>)> = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            handles.push(scope.spawn(|| {
                let mut local_blocks = Vec::new();
                loop {
                    let pair_idx = next_pair.fetch_add(1, Ordering::Relaxed);
                    if pair_idx >= shell_pairs.len() {
                        break;
                    }
                    let (p_shell_idx, q_shell_idx) = shell_pairs[pair_idx];
                    let p_shell = &aux_shells[p_shell_idx];
                    let q_shell = &aux_shells[q_shell_idx];
                    let block = (kernel.int2c)(p_shell, q_shell);
                    local_blocks.push((p_shell_idx, q_shell_idx, block));
                }
                local_blocks
            }));
        }

        let mut blocks = Vec::with_capacity(shell_pairs.len());
        for handle in handles {
            blocks.extend(
                handle
                    .join()
                    .expect("RI-r2 2c metric worker thread panicked unexpectedly"),
            );
        }
        blocks
    });

    let mut aux_v = MatrixFull::new([num_auxbas, num_auxbas], 0.0_f64);
    for (p_shell_idx, q_shell_idx, block) in blocks {
        let p_shell = &aux_shells[p_shell_idx];
        let q_shell = &aux_shells[q_shell_idx];
        for q_loc in 0..q_shell.ao_len {
            let q = q_shell.ao_start + q_loc;
            for p_loc in 0..p_shell.ao_len {
                let p = p_shell.ao_start + p_loc;
                let value = block[(p_loc, q_loc)];
                aux_v[(p, q)] = value;
                aux_v[(q, p)] = value;
            }
        }
    }
    aux_v
}

fn build_fitted_ri3fn_from_shell_blocks(
    ao_shells: &[RintShell],
    aux_shells: &[RintShell],
    num_baspar: usize,
    num_auxbas: usize,
    kernel: RiShellBlockKernel,
    aux_v_inv_sqrt: &MatrixFull<f64>,
) -> MatrixFull<f64> {
    if ao_shells.is_empty() || aux_shells.is_empty() {
        return MatrixFull::new([num_baspar, num_auxbas], 0.0_f64);
    }

    //  worker 数必须显式从 r2_direct_requested_workers() 取得，而不能读可见 CPU 数。
    let requested_threads = r2_direct_requested_workers();
    let mut right_shell_tasks: Vec<(usize, usize)> = ao_shells
        .iter()
        .enumerate()
        .filter(|(_, shell)| shell.ao_len > 0)
        .map(|(right_shell_idx, right_shell)| {
            let right_end = right_shell.ao_start + right_shell.ao_len;
            let left_cost = ao_shells
                .iter()
                .take_while(|left_shell| left_shell.ao_start < right_end)
                .map(|left_shell| {
                    let primitive_cost =
                        left_shell.shell.exponents.len() * right_shell.shell.exponents.len();
                    left_shell.ao_len * right_shell.ao_len * primitive_cost.max(1)
                })
                .sum::<usize>();
            let aux_cost = aux_shells
                .iter()
                .map(|aux_shell| aux_shell.ao_len * aux_shell.shell.exponents.len().max(1))
                .sum::<usize>()
                .max(1);
            (right_shell_idx, left_cost * aux_cost)
        })
        .collect();
    right_shell_tasks.sort_by(|(_, left_cost), (_, right_cost)| right_cost.cmp(left_cost));

    let num_workers = requested_threads.min(right_shell_tasks.len()).max(1);

    let next_right_shell_task = AtomicUsize::new(0);
    let mut rimatr = MatrixFull::new([num_baspar, num_auxbas], 0.0_f64);
    let inflight_blocks = std::env::var("REST_R2_DIRECT_INFLIGHT_BLOCKS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| (num_workers * 2).max(1));
    let (sender, receiver) = mpsc::sync_channel(inflight_blocks);
    let default_omp_num_threads = omp_get_num_threads_wrapper();
    omp_set_num_threads_wrapper(1);

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let sender = sender.clone();
            let next_right_shell_task = &next_right_shell_task;
            let right_shell_tasks = &right_shell_tasks;
            handles.push(scope.spawn(move || {
                let mut entries_cache: HashMap<
                    (u32, u32, u32, usize, usize, usize),
                    Vec<Rint3cBlockEntry>,
                > = HashMap::new();
                let mut pair_index_map_cache: HashMap<
                    (usize, usize, usize, usize, usize),
                    Vec<(usize, usize)>,
                > = HashMap::new();
                let mut block_buffer = MatrixFull::empty();
                loop {
                    let task_idx = next_right_shell_task.fetch_add(1, Ordering::Relaxed);
                    if task_idx >= right_shell_tasks.len() {
                        break;
                    }
                    let right_shell_idx = right_shell_tasks[task_idx].0;
                    let right_shell = &ao_shells[right_shell_idx];

                    let right_start = right_shell.ao_start;
                    let right_end = right_shell.ao_start + right_shell.ao_len;
                    let global_pair_start = right_start * (right_start + 1) / 2;
                    let global_pair_end = right_end * (right_end + 1) / 2;
                    let pair_len = global_pair_end - global_pair_start;
                    let mut local_raw = MatrixFull::new([pair_len, num_auxbas], 0.0_f64);

                    for left_shell in ao_shells.iter() {
                        if left_shell.ao_start >= right_end {
                            continue;
                        }
                        for aux_shell in aux_shells.iter() {
                            let pair_rows = left_shell.ao_len * right_shell.ao_len;
                            if block_buffer.size != [pair_rows, aux_shell.ao_len] {
                                block_buffer = MatrixFull::new([pair_rows, aux_shell.ao_len], 0.0);
                            }
                            let entries_key = (
                                left_shell.shell.ang_type,
                                right_shell.shell.ang_type,
                                aux_shell.shell.ang_type,
                                left_shell.ao_len,
                                right_shell.ao_len,
                                aux_shell.ao_len,
                            );
                            let entries = entries_cache.entry(entries_key).or_insert_with(|| {
                                build_3c_block_entries(left_shell, right_shell, aux_shell)
                            });
                            (kernel.int3c_into)(
                                left_shell,
                                right_shell,
                                aux_shell,
                                entries,
                                &mut block_buffer,
                            );
                            for aux_loc in 0..aux_shell.ao_len {
                                let aux = aux_shell.ao_start + aux_loc;
                                for nu_loc in 0..right_shell.ao_len {
                                    let nu = right_shell.ao_start + nu_loc;
                                    for mu_loc in 0..left_shell.ao_len {
                                        let mu = left_shell.ao_start + mu_loc;
                                        if mu <= nu {
                                            let block_row = nu_loc * left_shell.ao_len + mu_loc;
                                            let local_pair =
                                                baspair_index(mu, nu) - global_pair_start;
                                            local_raw[(local_pair, aux)] =
                                                block_buffer[(block_row, aux_loc)];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    let mut local_fitted = MatrixFull::new([pair_len, num_auxbas], 0.0_f64);
                    local_fitted.to_matrixfullslicemut().lapack_dgemm(
                        &local_raw.to_matrixfullslice(),
                        &aux_v_inv_sqrt.to_matrixfullslice(),
                        'N',
                        'N',
                        1.0,
                        0.0,
                    );
                    sender
                        .send((global_pair_start, pair_len, local_fitted))
                        .expect("failed to send fitted RI-r2 shell block");
                }
            }));
        }
        drop(sender);

        for (global_pair_start, pair_len, local_fitted) in receiver {
            for aux in 0..num_auxbas {
                let to_start = aux * num_baspar + global_pair_start;
                let from_start = aux * pair_len;
                rimatr.data[to_start..to_start + pair_len]
                    .copy_from_slice(&local_fitted.data[from_start..from_start + pair_len]);
            }
        }
        for handle in handles {
            handle
                .join()
                .expect("RI-r2 fitted 3c worker thread panicked unexpectedly");
        }
    });
    omp_set_num_threads_wrapper(default_omp_num_threads);

    rimatr
}

/// 按 kept 区间生成三中心拟合块（local_fitted_q），并以 shell block 为粒度回调消费。
/// 该函数是 kept-batch-first 数据流的底层并行遍历器。
fn for_each_fitted_ri3fn_shell_block_rayon_kept_range<F>(
    ao_shells: &[RintShell],
    aux_shells: &[RintShell],
    _num_auxbas: usize,
    kernel: RiShellBlockKernel,
    metric_factor: &MatrixFull<f64>,
    kept_start: usize,
    kept_len: usize,
    mut consume: F,
) where
    F: FnMut(usize, usize, MatrixFull<f64>),
{
    if ao_shells.is_empty() || aux_shells.is_empty() || kept_len == 0 {
        return;
    }
    debug_assert!(kept_start + kept_len <= metric_factor.size[1]);

    // kept-batch-first 的主并行入口：worker 数默认等于输入文件中的 num_threads。
    // 如需单独调 semi-direct 外层并行，可用 REST_R2_DIRECT_WORKERS 覆盖。
    let requested_threads = r2_direct_requested_workers();
    let mut right_shell_tasks: Vec<(usize, usize)> = ao_shells
        .iter()
        .enumerate()
        .filter(|(_, shell)| shell.ao_len > 0)
        .map(|(right_shell_idx, right_shell)| {
            let right_end = right_shell.ao_start + right_shell.ao_len;
            let left_cost = ao_shells
                .iter()
                .take_while(|left_shell| left_shell.ao_start < right_end)
                .map(|left_shell| {
                    let primitive_cost =
                        left_shell.shell.exponents.len() * right_shell.shell.exponents.len();
                    left_shell.ao_len * right_shell.ao_len * primitive_cost.max(1)
                })
                .sum::<usize>();
            let aux_cost = aux_shells
                .iter()
                .map(|aux_shell| aux_shell.ao_len * aux_shell.shell.exponents.len().max(1))
                .sum::<usize>()
                .max(1);
            (right_shell_idx, left_cost * aux_cost)
        })
        .collect();
    right_shell_tasks.sort_by(|(_, left_cost), (_, right_cost)| right_cost.cmp(left_cost));
    if right_shell_tasks.is_empty() {
        return;
    }

    let num_workers = requested_threads.min(right_shell_tasks.len()).max(1);
    let total_task_cost = right_shell_tasks
        .iter()
        .map(|(_, cost)| *cost)
        .sum::<usize>();
    // 将较小的 right-shell 任务合并成 group，减少 worker 间抢任务的同步开销。
    let min_task_cost = std::env::var("REST_R2_DIRECT_TASK_COST_MIN")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| (total_task_cost / (num_workers * 4).max(1)).max(1));
    let mut grouped_tasks: Vec<Vec<usize>> = Vec::new();
    let mut current_group = Vec::new();
    let mut current_group_cost = 0_usize;
    for (right_shell_idx, task_cost) in right_shell_tasks.iter().copied() {
        current_group.push(right_shell_idx);
        current_group_cost = current_group_cost.saturating_add(task_cost);
        if current_group_cost >= min_task_cost {
            grouped_tasks.push(std::mem::take(&mut current_group));
            current_group_cost = 0;
        }
    }
    if !current_group.is_empty() {
        grouped_tasks.push(current_group);
    }
    let log_worker_load = std::env::var("REST_R2_DIRECT_LOG_WORKER_LOAD")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false);
    let worker_group_counts = Arc::new(
        (0..num_workers)
            .map(|_| AtomicUsize::new(0))
            .collect::<Vec<_>>(),
    );
    let worker_shell_counts = Arc::new(
        (0..num_workers)
            .map(|_| AtomicUsize::new(0))
            .collect::<Vec<_>>(),
    );

    let next_right_shell_task = AtomicUsize::new(0);
    let inflight_blocks = r2_direct_inflight_blocks(num_workers);
    let (sender, receiver) = mpsc::sync_channel(inflight_blocks);
    let shell_screen_enabled = r2_direct_shell_screen_enabled();
    let direct_3c_expcutoff = r2_direct_3c_expcutoff();
    let default_omp_num_threads = omp_get_num_threads_wrapper();
    // 外层 worker 和内层 BLAS/OpenMP 不能同时无限放大；auto 模式默认优先外层并行。
    let omp_mode = std::env::var("REST_R2_DIRECT_OMP_MODE")
        .map(|value| value.trim().to_ascii_lowercase())
        .unwrap_or_else(|_| "auto".to_string());
    let (effective_mode, omp_threads) = match omp_mode.as_str() {
        "hybrid" => std::env::var("REST_R2_DIRECT_OMP_THREADS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
            .map(|threads| ("hybrid".to_string(), threads))
            .unwrap_or_else(|| ("hybrid".to_string(), (default_omp_num_threads / 2).max(1))),
        "inherit" => ("inherit".to_string(), default_omp_num_threads.max(1)),
        "outer" => ("outer".to_string(), 1),
        "auto" => {
            let auto_threads = if num_workers <= 4 && default_omp_num_threads >= 2 {
                2
            } else {
                1
            };
            let mode = if auto_threads > 1 {
                "auto(hybrid)".to_string()
            } else {
                "auto(outer)".to_string()
            };
            (mode, auto_threads)
        }
        _ => ("outer".to_string(), 1),
    };
    omp_set_num_threads_wrapper(omp_threads);
    println!(
        "RI-r2 semi-direct parallel plan: workers={} grouped_tasks={} min_task_cost={} omp_mode={} omp_threads={} inflight_blocks={}",
        num_workers,
        grouped_tasks.len(),
        min_task_cost,
        effective_mode,
        omp_threads,
        inflight_blocks
    );

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(num_workers);
        for worker_idx in 0..num_workers {
            let sender = sender.clone();
            let next_right_shell_task = &next_right_shell_task;
            let grouped_tasks = &grouped_tasks;
            let metric_factor = metric_factor;
            let shell_screen_enabled = shell_screen_enabled;
            let direct_3c_expcutoff = direct_3c_expcutoff;
            let worker_group_counts = Arc::clone(&worker_group_counts);
            let worker_shell_counts = Arc::clone(&worker_shell_counts);
            handles.push(scope.spawn(move || {
                let mut entries_cache: HashMap<
                    (u32, u32, u32, usize, usize, usize),
                    Vec<Rint3cBlockEntry>,
                > = HashMap::new();
                let mut pair_index_map_cache: HashMap<
                    (usize, usize, usize, usize, usize),
                    Vec<(usize, usize)>,
                > = HashMap::new();
                let mut block_buffer = MatrixFull::empty();
                let mut raw_compact = MatrixFull::empty();
                let mut metric_compact = MatrixFull::empty();
                let mut fit_delta = MatrixFull::empty();
                loop {
                    let task_idx = next_right_shell_task.fetch_add(1, Ordering::Relaxed);
                    if task_idx >= grouped_tasks.len() {
                        break;
                    }
                    worker_group_counts[worker_idx].fetch_add(1, Ordering::Relaxed);
                    for right_shell_idx in grouped_tasks[task_idx].iter().copied() {
                        worker_shell_counts[worker_idx].fetch_add(1, Ordering::Relaxed);
                        let right_shell = &ao_shells[right_shell_idx];
                        let right_start = right_shell.ao_start;
                        let right_end = right_shell.ao_start + right_shell.ao_len;
                        let global_pair_start = right_start * (right_start + 1) / 2;
                        let global_pair_end = right_end * (right_end + 1) / 2;
                        let pair_len = global_pair_end - global_pair_start;
                        let mut local_fitted = MatrixFull::new([pair_len, kept_len], 0.0_f64);

                        for left_shell in ao_shells.iter() {
                            if left_shell.ao_start >= right_end {
                                continue;
                            }
                            if shell_screen_enabled
                                && r2_shell_pair_all_primitives_screened(
                                    left_shell,
                                    right_shell,
                                    direct_3c_expcutoff,
                                )
                            {
                                continue;
                            }
                            for aux_shell in aux_shells.iter() {
                                let pair_rows = left_shell.ao_len * right_shell.ao_len;
                                let pair_map_key = (
                                    left_shell.ao_start,
                                    left_shell.ao_len,
                                    right_shell.ao_start,
                                    right_shell.ao_len,
                                    global_pair_start,
                                );
                                let pair_index_map =
                                    pair_index_map_cache.entry(pair_map_key).or_insert_with(|| {
                                        let mut mapping = Vec::with_capacity(pair_rows);
                                        for nu_loc in 0..right_shell.ao_len {
                                            let nu = right_shell.ao_start + nu_loc;
                                            for mu_loc in 0..left_shell.ao_len {
                                                let mu = left_shell.ao_start + mu_loc;
                                                if mu <= nu {
                                                    let block_row =
                                                        nu_loc * left_shell.ao_len + mu_loc;
                                                    let local_pair =
                                                        baspair_index(mu, nu) - global_pair_start;
                                                    mapping.push((block_row, local_pair));
                                                }
                                            }
                                        }
                                        mapping
                                    });
                                if block_buffer.size != [pair_rows, aux_shell.ao_len] {
                                    block_buffer =
                                        MatrixFull::new([pair_rows, aux_shell.ao_len], 0.0);
                                }
                                let entries_key = (
                                    left_shell.shell.ang_type,
                                    right_shell.shell.ang_type,
                                    aux_shell.shell.ang_type,
                                    left_shell.ao_len,
                                    right_shell.ao_len,
                                    aux_shell.ao_len,
                                );
                                let entries =
                                    entries_cache.entry(entries_key).or_insert_with(|| {
                                        build_3c_block_entries(left_shell, right_shell, aux_shell)
                                    });
                                (kernel.int3c_into)(
                                    left_shell,
                                    right_shell,
                                    aux_shell,
                                    entries,
                                    &mut block_buffer,
                                );
                                r2_accumulate_streamed_fitted_aux_shell_block(
                                    &block_buffer,
                                    pair_index_map,
                                    aux_shell,
                                    metric_factor,
                                    kept_start,
                                    kept_len,
                                    &mut local_fitted,
                                    &mut raw_compact,
                                    &mut metric_compact,
                                    &mut fit_delta,
                                );
                            }
                        }

                        sender
                            .send((global_pair_start, pair_len, local_fitted))
                            .expect(
                                "failed to send kept-range fitted RI-r2 semi-direct shell block",
                            );
                    }
                }
            }));
        }
        drop(sender);

        for (global_pair_start, pair_len, local_fitted) in receiver {
            consume(global_pair_start, pair_len, local_fitted);
        }
        for handle in handles {
            handle
                .join()
                .expect("RI-r2 kept-range fitted shell block worker thread panicked unexpectedly");
        }
    });
    if log_worker_load {
        let group_counts: Vec<usize> = worker_group_counts
            .iter()
            .map(|counter| counter.load(Ordering::Relaxed))
            .collect();
        let shell_counts: Vec<usize> = worker_shell_counts
            .iter()
            .map(|counter| counter.load(Ordering::Relaxed))
            .collect();
        let group_min = group_counts.iter().copied().min().unwrap_or(0);
        let group_max = group_counts.iter().copied().max().unwrap_or(0);
        let shell_min = shell_counts.iter().copied().min().unwrap_or(0);
        let shell_max = shell_counts.iter().copied().max().unwrap_or(0);
        println!(
            "RI-r2 semi-direct worker load: groups_per_worker={:?} shells_per_worker={:?} group_min/max={}/{} shell_min/max={}/{}",
            group_counts, shell_counts, group_min, group_max, shell_min, shell_max
        );
    }
    omp_set_num_threads_wrapper(default_omp_num_threads);
}

fn r2_accumulate_streamed_fitted_aux_shell_block(
    block_buffer: &MatrixFull<f64>,
    pair_index_map: &[(usize, usize)],
    aux_shell: &RintShell,
    metric_factor: &MatrixFull<f64>,
    kept_start: usize,
    kept_len: usize,
    local_fitted: &mut MatrixFull<f64>,
    raw_compact: &mut MatrixFull<f64>,
    metric_compact: &mut MatrixFull<f64>,
    fit_delta: &mut MatrixFull<f64>,
) {
    let valid_pairs = pair_index_map.len();
    let aux_len = aux_shell.ao_len;
    if valid_pairs == 0 || aux_len == 0 || kept_len == 0 {
        return;
    }

    r2_prepare_scratch_matrix(raw_compact, [valid_pairs, aux_len], false);
    for aux_loc in 0..aux_len {
        for (pair_idx, (block_row, _local_pair)) in pair_index_map.iter().copied().enumerate() {
            raw_compact[(pair_idx, aux_loc)] = block_buffer[(block_row, aux_loc)];
        }
    }

    r2_prepare_scratch_matrix(metric_compact, [aux_len, kept_len], false);
    for kept_loc in 0..kept_len {
        let kept = kept_start + kept_loc;
        for aux_loc in 0..aux_len {
            let aux = aux_shell.ao_start + aux_loc;
            metric_compact[(aux_loc, kept_loc)] = metric_factor[(aux, kept)];
        }
    }

    r2_prepare_scratch_matrix(fit_delta, [valid_pairs, kept_len], false);
    fit_delta.to_matrixfullslicemut().lapack_dgemm(
        &raw_compact.to_matrixfullslice(),
        &metric_compact.to_matrixfullslice(),
        'N',
        'N',
        1.0,
        0.0,
    );

    for kept_loc in 0..kept_len {
        for (pair_idx, (_block_row, local_pair)) in pair_index_map.iter().copied().enumerate() {
            local_fitted[(local_pair, kept_loc)] += fit_delta[(pair_idx, kept_loc)];
        }
    }
}

fn prepare_rimatr_from_shell_blocks_sync_with_auxbas_threshold(
    ao_shells: &[RintShell],
    aux_shells: &[RintShell],
    kernel: RiShellBlockKernel,
    auxbas_threshold: f64,
) -> Option<(MatrixFull<f64>, MatrixFull<usize>, Vec<[usize; 2]>)> {
    if ao_shells.is_empty() || aux_shells.is_empty() {
        return None;
    }
    let num_basis = rint_shell_basis_count(ao_shells);
    let num_auxbas = rint_shell_basis_count(aux_shells);
    if num_basis == 0 || num_auxbas == 0 {
        return None;
    }

    let num_baspar = (num_basis + 1) * num_basis / 2;
    let (basbas2baspar, baspar2basbas) = prepare_baspair_map(num_basis);

    let aux_v = build_aux_metric_from_shell_blocks_parallel(aux_shells, num_auxbas, kernel);
    let aux_v_inv_sqrt = r2_aux_metric_pinv_factor_cached(&aux_v, auxbas_threshold)?;
    let rimatr = build_fitted_ri3fn_from_shell_blocks(
        ao_shells,
        aux_shells,
        num_baspar,
        num_auxbas,
        kernel,
        &aux_v_inv_sqrt,
    );
    Some((rimatr, basbas2baspar, baspar2basbas))
}

pub fn prepare_rimatr_for_r2_shell_blocks_sync_with_auxbas_threshold(
    ao_shells: &[RintShell],
    aux_shells: &[RintShell],
    auxbas_threshold: f64,
) -> Option<(MatrixFull<f64>, MatrixFull<usize>, Vec<[usize; 2]>)> {
    prepare_rimatr_from_shell_blocks_sync_with_auxbas_threshold(
        ao_shells,
        aux_shells,
        RiShellBlockKernel {
            int2c: int2c_r2_shell_block,
            int3c_into: int3c_r2_shell_block_batched_into,
        },
        auxbas_threshold,
    )
}

pub fn prepare_rimatr_for_r2_shell_blocks_sync(
    ao_shells: &[RintShell],
    aux_shells: &[RintShell],
) -> Option<(MatrixFull<f64>, MatrixFull<usize>, Vec<[usize; 2]>)> {
    prepare_rimatr_for_r2_shell_blocks_sync_with_auxbas_threshold(
        ao_shells,
        aux_shells,
        AUXBAS_THRESHOLD,
    )
}

pub fn prepare_rimatr_for_r2(
    ao_shells: &[RintShell],
    aux_shells: &[RintShell],
) -> Option<(MatrixFull<f64>, MatrixFull<usize>, Vec<[usize; 2]>)> {
    prepare_rimatr_for_r2_shell_blocks_sync(ao_shells, aux_shells)
}

/// 读取 RI-r2 direct 路径的可用内存上限（MB）。
fn r2_memory_limit_mb() -> f64 {
    std::env::var("REST_R2_DIRECT_MEMORY_MB")
        .ok()
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| *value > 0.0)
        .unwrap_or_else(detect_available_memory_mb)
}

/// 读取 direct 路径的内存安全系数（0~1）。
fn r2_direct_memory_factor() -> f64 {
    std::env::var("REST_R2_DIRECT_MEMORY_FACTOR")
        .ok()
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| *value > 0.0 && *value <= 1.0)
        .unwrap_or(R2_DIRECT_MEMORY_FACTOR)
}

/// 估算 RI-r2 incore 的峰值内存（MB）：
/// total = fitted_3c + metric + metric_factor。
fn r2_incore_memory_estimate_mb(num_basis: usize, num_auxbas: usize) -> (f64, f64, f64, f64) {
    let num_baspar = (num_basis + 1) * num_basis / 2;
    let fitted_ri_mb = 8.0 * (num_baspar as f64) * (num_auxbas as f64) / 1024.0 / 1024.0;
    let metric_mb = 8.0 * (num_auxbas as f64) * (num_auxbas as f64) / 1024.0 / 1024.0;
    let metric_factor_mb = metric_mb;
    let total_mb = fitted_ri_mb + metric_mb + metric_factor_mb;
    (total_mb, fitted_ri_mb, metric_mb, metric_factor_mb)
}

/// 决定是否启用 semi-direct：
/// 1) 可被 REST_R2_DIRECT 强制覆盖；
/// 2) 否则根据 incore 估算是否超过预算自动判断。
fn r2_use_semidirect(num_basis: usize, num_auxbas: usize) -> bool {
    match std::env::var("REST_R2_DIRECT") {
        Ok(value) => {
            let value = value.trim().to_ascii_lowercase();
            if matches!(
                value.as_str(),
                "1" | "true" | "yes" | "on" | "semi-direct" | "semidirect" | "direct"
            ) {
                println!("RI-r2 semi-direct requested by REST_R2_DIRECT={value}");
                return true;
            }
            if matches!(value.as_str(), "0" | "false" | "no" | "off" | "incore") {
                println!("RI-r2 incore forced by REST_R2_DIRECT={value}");
                return false;
            }
        }
        Err(_) => {}
    }

    let (incore_mb, fitted_ri_mb, metric_mb, metric_factor_mb) =
        r2_incore_memory_estimate_mb(num_basis, num_auxbas);
    let avail_mb = r2_memory_limit_mb();
    let factor = r2_direct_memory_factor();
    let threshold_mb = avail_mb * factor;
    let use_semidirect = incore_mb > threshold_mb;
    println!(
        "RI-r2 memory estimate: incore_peak={:.2} MB (fitted_3c={:.2} MB metric={:.2} MB metric_factor={:.2} MB), limit={:.2} MB factor={:.2} incore_budget={:.2} MB -> {}",
        incore_mb,
        fitted_ri_mb,
        metric_mb,
        metric_factor_mb,
        avail_mb,
        factor,
        threshold_mb,
        if use_semidirect {
            "semi-direct: incore estimate exceeds budget"
        } else {
            "incore: incore estimate fits budget"
        }
    );
    use_semidirect
}

fn r2_aux_metric_pinv_factor(aux_v: &MatrixFull<f64>, threshold: f64) -> Option<MatrixFull<f64>> {
    if aux_v.size[0] != aux_v.size[1] {
        return None;
    }
    let n = aux_v.size[0];
    let mut sorted_aux_v = aux_v.clone();
    sorted_aux_v
        .data
        .iter_mut()
        .for_each(|value| *value *= -1.0);
    let (eigenvectors, mut eigenvalues, info_n) =
        sorted_aux_v.to_matrixfullslicemut().lapack_dsyev()?;
    if info_n as usize != n {
        panic!("Found unphysical eigenvalues in RI-r2 auxiliary metric");
    }

    let mut nkept = 0_usize;
    for value in eigenvalues.iter_mut() {
        *value *= -1.0;
        if *value >= threshold {
            nkept += 1;
        }
    }
    if nkept == 0 {
        return None;
    }
    if nkept != n {
        println!("n_nonsigular: {}", nkept);
    }

    let mut factor = MatrixFull::new([n, nkept], 0.0_f64);
    for col in 0..nkept {
        let scale = eigenvalues[col].powf(-0.5);
        for row in 0..n {
            factor[(row, col)] = eigenvectors[(row, col)] * scale;
        }
    }
    Some(factor)
}

struct R2MetricFactorCacheEntry {
    n: usize,
    threshold_bits: u64,
    data_hash: u64,
    factor: MatrixFull<f64>,
}

static R2_METRIC_FACTOR_CACHE: OnceLock<Mutex<Option<R2MetricFactorCacheEntry>>> = OnceLock::new();

fn r2_metric_factor_cache_enabled() -> bool {
    std::env::var("REST_R2_METRIC_FACTOR_CACHE")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on" | "force"
            )
        })
        .unwrap_or(false)
}

fn r2_metric_matrix_hash(aux_v: &MatrixFull<f64>, threshold: f64) -> (usize, u64, u64) {
    let mut hasher = DefaultHasher::new();
    aux_v.size.hash(&mut hasher);
    threshold.to_bits().hash(&mut hasher);
    for value in aux_v.data.iter() {
        value.to_bits().hash(&mut hasher);
    }
    (aux_v.size[0], threshold.to_bits(), hasher.finish())
}

/// 计算并可选缓存 RI-r2 的 metric 伪逆因子（eigh/pinv）。
/// 缓存命中仅用于完全相同的 metric 矩阵和阈值。
fn r2_aux_metric_pinv_factor_cached(
    aux_v: &MatrixFull<f64>,
    threshold: f64,
) -> Option<MatrixFull<f64>> {
    if !r2_metric_factor_cache_enabled() {
        return r2_aux_metric_pinv_factor(aux_v, threshold);
    }
    let (n, threshold_bits, data_hash) = r2_metric_matrix_hash(aux_v, threshold);
    let cache = R2_METRIC_FACTOR_CACHE.get_or_init(|| Mutex::new(None));
    if let Ok(guard) = cache.lock() {
        if let Some(entry) = guard.as_ref() {
            if entry.n == n
                && entry.threshold_bits == threshold_bits
                && entry.data_hash == data_hash
            {
                println!("RI-r2 semi-direct metric factor cache hit: n={n}");
                return Some(entry.factor.clone());
            }
        }
    }

    let factor = r2_aux_metric_pinv_factor(aux_v, threshold)?;
    if let Ok(mut guard) = cache.lock() {
        *guard = Some(R2MetricFactorCacheEntry {
            n,
            threshold_bits,
            data_hash,
            factor: factor.clone(),
        });
    }
    Some(factor)
}

fn r2_prepare_occ_coeffs(
    coeff: &[MatrixFull<f64>],
    occupation: &[Vec<f64>],
    num_basis: usize,
) -> Option<Vec<MatrixFull<f64>>> {
    if coeff.len() != occupation.len() || coeff.is_empty() {
        return None;
    }
    let mut occ_coeffs = Vec::with_capacity(coeff.len());
    for (c, occ) in coeff.iter().zip(occupation.iter()) {
        if c.size[0] != num_basis || c.size[1] != occ.len() {
            return None;
        }
        let nocc = occ.iter().filter(|value| **value > f64::EPSILON).count();
        if nocc == 0 {
            return None;
        }
        let mut c_occ = MatrixFull::new([num_basis, nocc], 0.0_f64);
        let mut occ_col = 0;
        for (mo, occ_value) in occ.iter().enumerate() {
            if *occ_value <= f64::EPSILON {
                continue;
            }
            let scale = occ_value.sqrt();
            for mu in 0..num_basis {
                c_occ[(mu, occ_col)] = c[(mu, mo)] * scale;
            }
            occ_col += 1;
        }
        occ_coeffs.push(c_occ);
    }
    Some(occ_coeffs)
}

fn triangular_pair_start_to_ao(pair_start: usize) -> usize {
    let mut low = 0_usize;
    let mut high = 1_usize;
    while high * (high + 1) / 2 < pair_start {
        high *= 2;
    }
    while low < high {
        let mid = (low + high) / 2;
        let tri = mid * (mid + 1) / 2;
        if tri < pair_start {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    low
}

struct R2HFitScratch {
    c_right: MatrixFull<f64>,
    c_right_key: Option<(usize, usize, usize)>,
    b_direct: MatrixFull<f64>,
    b_trans: MatrixFull<f64>,
    direct: MatrixFull<f64>,
    c_left: MatrixFull<f64>,
    c_left_key: Option<(usize, usize)>,
    trans: MatrixFull<f64>,
}

impl Default for R2HFitScratch {
    fn default() -> Self {
        Self {
            c_right: MatrixFull::empty(),
            c_right_key: None,
            b_direct: MatrixFull::empty(),
            b_trans: MatrixFull::empty(),
            direct: MatrixFull::empty(),
            c_left: MatrixFull::empty(),
            c_left_key: None,
            trans: MatrixFull::empty(),
        }
    }
}

#[derive(Default)]
struct R2HFitStepTimings {
    pack: f64,
    gemm: f64,
    scatter: f64,
    c_pack: f64,
    b_pack: f64,
    direct_gemm: f64,
    trans_gemm: f64,
    direct_scatter: f64,
    trans_scatter: f64,
}

fn r2_prepare_scratch_matrix(matrix: &mut MatrixFull<f64>, size: [usize; 2], fill_zero: bool) {
    if matrix.size != size {
        *matrix = MatrixFull::new(size, 0.0_f64);
    } else if fill_zero {
        matrix.data.fill(0.0);
    }
}

fn r2_accumulate_hfit_from_fitted_shell_block_kept_range_with_scratch(
    global_pair_start: usize,
    pair_len: usize,
    local_fitted: &MatrixFull<f64>,
    kept_start: usize,
    kept_len: usize,
    h_fit_kept_start: usize,
    occ_coeff: &MatrixFull<f64>,
    h_fit: &mut MatrixFull<f64>,
    num_basis: usize,
    scratch: &mut R2HFitScratch,
) -> R2HFitStepTimings {
    let mut timings = R2HFitStepTimings::default();
    let right_start = triangular_pair_start_to_ao(global_pair_start);
    let right_end = triangular_pair_start_to_ao(global_pair_start + pair_len);
    let nocc = occ_coeff.size[1];
    let width = right_end - right_start;
    if width == 0 || kept_len == 0 {
        return timings;
    }
    let h_rows = num_basis * nocc;

    let timer = Instant::now();
    let c_right_key = (right_start, right_end, nocc);
    if scratch.c_right_key != Some(c_right_key) || scratch.c_right.size != [width, nocc] {
        r2_prepare_scratch_matrix(&mut scratch.c_right, [width, nocc], false);
        for occ in 0..nocc {
            let from_start = occ * num_basis + right_start;
            let to_start = occ * width;
            scratch.c_right.data[to_start..to_start + width]
                .copy_from_slice(&occ_coeff.data[from_start..from_start + width]);
        }
        scratch.c_right_key = Some(c_right_key);
    }
    let c_right_pack = timer.elapsed().as_secs_f64();
    timings.c_pack += c_right_pack;
    timings.pack += c_right_pack;

    let timer = Instant::now();
    r2_prepare_scratch_matrix(&mut scratch.b_direct, [right_end * kept_len, width], true);
    r2_prepare_scratch_matrix(&mut scratch.b_trans, [right_end, width * kept_len], true);
    for (nu_loc, nu) in (right_start..right_end).enumerate() {
        let pair_base = baspair_index(0, nu) - global_pair_start;
        let copy_len = nu + 1;
        for kept_loc in 0..kept_len {
            let from_start = (kept_start + kept_loc) * pair_len + pair_base;
            let direct_start = nu_loc * right_end * kept_len + right_end * kept_loc;
            scratch.b_direct.data[direct_start..direct_start + copy_len]
                .copy_from_slice(&local_fitted.data[from_start..from_start + copy_len]);
            if nu > 0 {
                let trans_start = (nu_loc + width * kept_loc) * right_end;
                scratch.b_trans.data[trans_start..trans_start + nu]
                    .copy_from_slice(&local_fitted.data[from_start..from_start + nu]);
            }
        }
    }
    let b_pack = timer.elapsed().as_secs_f64();
    timings.b_pack += b_pack;
    timings.pack += b_pack;

    r2_prepare_scratch_matrix(&mut scratch.direct, [right_end * kept_len, nocc], false);
    let timer = Instant::now();
    scratch.direct.to_matrixfullslicemut().lapack_dgemm(
        &scratch.b_direct.to_matrixfullslice(),
        &scratch.c_right.to_matrixfullslice(),
        'N',
        'N',
        1.0,
        0.0,
    );
    let direct_gemm = timer.elapsed().as_secs_f64();
    timings.direct_gemm += direct_gemm;
    timings.gemm += direct_gemm;
    let timer = Instant::now();
    for kept_loc in 0..kept_len {
        let h_fit_kept = h_fit_kept_start + kept_loc;
        for occ in 0..nocc {
            let from_start = occ * right_end * kept_len + right_end * kept_loc;
            let to_start = h_fit_kept * h_rows + occ * num_basis;
            for (to, from) in h_fit.data[to_start..to_start + right_end]
                .iter_mut()
                .zip(scratch.direct.data[from_start..from_start + right_end].iter())
            {
                *to += *from;
            }
        }
    }
    let direct_scatter = timer.elapsed().as_secs_f64();
    timings.direct_scatter += direct_scatter;
    timings.scatter += direct_scatter;

    let timer = Instant::now();
    let c_left_key = (right_end, nocc);
    if scratch.c_left_key != Some(c_left_key) || scratch.c_left.size != [right_end, nocc] {
        r2_prepare_scratch_matrix(&mut scratch.c_left, [right_end, nocc], false);
        for occ in 0..nocc {
            let from_start = occ * num_basis;
            let to_start = occ * right_end;
            scratch.c_left.data[to_start..to_start + right_end]
                .copy_from_slice(&occ_coeff.data[from_start..from_start + right_end]);
        }
        scratch.c_left_key = Some(c_left_key);
    }
    r2_prepare_scratch_matrix(&mut scratch.trans, [width * kept_len, nocc], false);
    let c_left_pack = timer.elapsed().as_secs_f64();
    timings.c_pack += c_left_pack;
    timings.pack += c_left_pack;
    let timer = Instant::now();
    scratch.trans.to_matrixfullslicemut().lapack_dgemm(
        &scratch.b_trans.to_matrixfullslice(),
        &scratch.c_left.to_matrixfullslice(),
        'T',
        'N',
        1.0,
        0.0,
    );
    let trans_gemm = timer.elapsed().as_secs_f64();
    timings.trans_gemm += trans_gemm;
    timings.gemm += trans_gemm;
    let timer = Instant::now();
    for kept_loc in 0..kept_len {
        let h_fit_kept = h_fit_kept_start + kept_loc;
        for occ in 0..nocc {
            let from_start = occ * width * kept_len + width * kept_loc;
            let to_start = h_fit_kept * h_rows + occ * num_basis + right_start;
            for (to, from) in h_fit.data[to_start..to_start + width]
                .iter_mut()
                .zip(scratch.trans.data[from_start..from_start + width].iter())
            {
                *to += *from;
            }
        }
    }
    let trans_scatter = timer.elapsed().as_secs_f64();
    timings.trans_scatter += trans_scatter;
    timings.scatter += trans_scatter;
    timings
}

/// kept_batch 的硬上限（用于避免过大批次导致工作集失控）。
fn r2_direct_max_kept_block_size(nkept: usize) -> usize {
    std::env::var("REST_R2_DIRECT_MAX_KEPT_BLOCK")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .map(|value| value.min(nkept))
        .unwrap_or_else(|| 4096_usize.min(nkept).max(1))
}

/// semi-direct 外层 worker 数。
///
/// 默认读取 Rayon 当前线程数，也就是 ctrl 中 num_threads 初始化出的并行度；
/// 不再直接读取系统/LSF 可见 CPU 数，避免 `num_threads = 16` 时仍开 128 个 worker。
/// 如需针对 semi-direct 单独覆盖，可设置 REST_R2_DIRECT_WORKERS。
fn r2_direct_requested_workers() -> usize {
    std::env::var("REST_R2_DIRECT_WORKERS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(rayon::current_num_threads)
        .max(1)
}

fn r2_direct_inflight_blocks(workers: usize) -> usize {
    std::env::var("REST_R2_DIRECT_INFLIGHT_BLOCKS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| (workers * 2).max(1))
}

fn r2_direct_kept_memory_fraction() -> f64 {
    std::env::var("REST_R2_DIRECT_KEPT_MEMORY_FRACTION")
        .ok()
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| *value > 0.0 && *value <= 1.0)
        .unwrap_or(0.60)
}

fn r2_max_shell_block_pair_len(ao_shells: &[RintShell], num_basis: usize) -> usize {
    ao_shells
        .iter()
        .filter(|shell| shell.ao_len > 0)
        .map(|shell| {
            let right_start = shell.ao_start.min(num_basis);
            let right_end = (shell.ao_start + shell.ao_len).min(num_basis);
            let global_pair_start = right_start * (right_start + 1) / 2;
            let global_pair_end = right_end * (right_end + 1) / 2;
            global_pair_end.saturating_sub(global_pair_start)
        })
        .max()
        .unwrap_or_else(|| num_basis.max(1))
        .max(1)
}

fn r2_max_aux_shell_len(aux_shells: &[RintShell]) -> usize {
    aux_shells
        .iter()
        .map(|shell| shell.ao_len)
        .max()
        .unwrap_or(1)
        .max(1)
}

/// 根据常驻内存、并行 shell worker 临时块、H/K scratch 和 batch cache 估算 kept_batch。
fn r2_direct_kept_block_size(
    nkept: usize,
    num_basis: usize,
    num_auxbas: usize,
    num_baspar: usize,
    occ_coeffs: &[MatrixFull<f64>],
    max_pair_len: usize,
    max_aux_shell_len: usize,
) -> usize {
    if nkept == 0 {
        return 0;
    }

    let nocc_sum = occ_coeffs.iter().map(|coeff| coeff.size[1]).sum::<usize>();
    let nocc_max = occ_coeffs
        .iter()
        .map(|coeff| coeff.size[1])
        .max()
        .unwrap_or(1)
        .max(1);
    let h_values_per_kept = occ_coeffs
        .iter()
        .map(|coeff| num_basis.saturating_mul(coeff.size[1]))
        .sum::<usize>();
    let k_scratch_values_per_kept = num_basis
        .saturating_mul(nocc_sum)
        .saturating_add(num_basis.saturating_mul(nocc_max))
        .saturating_add(num_basis);
    let batch_cache_values_per_kept = num_baspar;
    // 内存计划必须和实际 worker 数使用同一个来源，否则 kept_batch 会被错误估大或估小。
    let workers = r2_direct_requested_workers();
    let inflight_blocks = r2_direct_inflight_blocks(workers);
    let worker_fitted_values_per_kept = max_pair_len.saturating_mul(
        workers
            .saturating_mul(2)
            .saturating_add(inflight_blocks)
            .saturating_add(1),
    );
    let live_values_per_kept = h_values_per_kept
        .saturating_add(k_scratch_values_per_kept)
        .saturating_add(batch_cache_values_per_kept)
        .saturating_add(worker_fitted_values_per_kept);
    if live_values_per_kept == 0 {
        return 512_usize.min(nkept).max(1);
    }

    let limit_mb = r2_memory_limit_mb();
    let factor = r2_direct_memory_factor();
    let kept_fraction = r2_direct_kept_memory_fraction();
    let total_budget_bytes = ((limit_mb * factor) * 1024.0 * 1024.0) as usize;
    let fixed_metric_factor_bytes = num_auxbas
        .saturating_mul(nkept)
        .saturating_mul(std::mem::size_of::<f64>());
    let output_values = num_baspar
        .saturating_add(num_basis.saturating_mul(num_basis))
        .saturating_mul(occ_coeffs.len().max(1));
    let output_bytes = output_values.saturating_mul(std::mem::size_of::<f64>());
    let worker_raw_bytes = workers
        .saturating_mul(max_pair_len)
        .saturating_mul(max_aux_shell_len)
        .saturating_mul(2)
        .saturating_mul(std::mem::size_of::<f64>());
    let fixed_bytes = fixed_metric_factor_bytes
        .saturating_add(output_bytes)
        .saturating_add(worker_raw_bytes);
    let remaining_budget_bytes = total_budget_bytes.saturating_sub(fixed_bytes);
    let kept_budget_bytes = ((remaining_budget_bytes as f64) * kept_fraction).floor() as usize;
    let live_budget_mb = bytes_to_mb(kept_budget_bytes);
    let bytes_per_kept = (live_values_per_kept as f64) * std::mem::size_of::<f64>() as f64;
    let estimated = ((kept_budget_bytes as f64) / bytes_per_kept).floor() as usize;
    let max_kept = r2_direct_max_kept_block_size(nkept);
    let kept_block = estimated.max(1).min(max_kept).min(nkept).max(1);
    println!(
        "RI-r2 semi-direct K kept_batch auto: kept_batch={} nkept={} limit={:.2} MB factor={:.2} kept_fraction={:.2} fixed={:.2} MB fixed_metric_factor={:.2} MB worker_raw_streamed={:.2} MB output={:.2} MB kept_budget={:.2} MB live_per_kept={:.2} KB H_per_kept={} K_scratch_per_kept={} batch_cache_per_kept={} worker_fitted_per_kept={} max_pair_len={} max_aux_shell_len={} workers={} inflight_blocks={} nocc_sum={} nocc_max={} cap={}",
        kept_block,
        nkept,
        limit_mb,
        factor,
        kept_fraction,
        bytes_to_mb(fixed_bytes),
        bytes_to_mb(fixed_metric_factor_bytes),
        bytes_to_mb(worker_raw_bytes),
        bytes_to_mb(output_bytes),
        live_budget_mb,
        bytes_per_kept / 1024.0,
        h_values_per_kept,
        k_scratch_values_per_kept,
        batch_cache_values_per_kept,
        worker_fitted_values_per_kept,
        max_pair_len,
        max_aux_shell_len,
        workers,
        inflight_blocks,
        nocc_sum,
        nocc_max,
        max_kept,
    );
    kept_block
}

#[derive(Default)]
struct R2SemiDirectTimings {
    metric_2c: f64,
    metric_eigh: f64,
    raw_3c: f64,
    fit_b: f64,
    rho: f64,
    half_transform: f64,
    fit_h: f64,
    fit_h_pack: f64,
    fit_h_gemm: f64,
    fit_h_scatter: f64,
    fit_h_c_pack: f64,
    fit_h_b_pack: f64,
    fit_h_direct_gemm: f64,
    fit_h_trans_gemm: f64,
    fit_h_direct_scatter: f64,
    fit_h_trans_scatter: f64,
    j_contract: f64,
    k_pack: f64,
    k_dgemm: f64,
    k_scatter: f64,
    cache_write: f64,
    cache_read_j: f64,
    cache_read_k: f64,
    hfit_cache_build: f64,
    cache_write_bytes: usize,
    cache_read_j_bytes: usize,
    cache_read_k_bytes: usize,
    hfit_cache_bytes: usize,
    regenerated_blocks_j: usize,
    regenerated_blocks_k: usize,
    kept_batches: usize,
    effective_kept_batch: usize,
    hfit_cache_read_failures: usize,
    hfit_cache_write_failures: usize,
    raw_cache_read_failures: usize,
    raw_cache_write_failures: usize,
    fitted_shell_block_passes: usize,
    hfit_build_source: &'static str,
}

impl R2SemiDirectTimings {
    fn accumulate_hfit_step(&mut self, step: R2HFitStepTimings) {
        self.fit_h_pack += step.pack;
        self.fit_h_gemm += step.gemm;
        self.fit_h_scatter += step.scatter;
        self.fit_h_c_pack += step.c_pack;
        self.fit_h_b_pack += step.b_pack;
        self.fit_h_direct_gemm += step.direct_gemm;
        self.fit_h_trans_gemm += step.trans_gemm;
        self.fit_h_direct_scatter += step.direct_scatter;
        self.fit_h_trans_scatter += step.trans_scatter;
    }

    fn print(&self) {
        let total = self.metric_2c
            + self.metric_eigh
            + self.raw_3c
            + self.fit_b
            + self.rho
            + self.half_transform
            + self.fit_h
            + self.j_contract
            + self.k_pack
            + self.k_dgemm
            + self.k_scatter;
        println!(
            "RI-r2 semi-direct timings: metric_2c={:.3}s metric_eigh={:.3}s raw_3c={:.3}s fit_b={:.3}s rho={:.3}s half_transform={:.3}s fit_h={:.3}s fit_h_pack={:.3}s fit_h_gemm={:.3}s fit_h_scatter={:.3}s J={:.3}s K_pack={:.3}s K_dgemm={:.3}s K_scatter={:.3}s total_profiled={:.3}s",
            self.metric_2c,
            self.metric_eigh,
            self.raw_3c,
            self.fit_b,
            self.rho,
            self.half_transform,
            self.fit_h,
            self.fit_h_pack,
            self.fit_h_gemm,
            self.fit_h_scatter,
            self.j_contract,
            self.k_pack,
            self.k_dgemm,
            self.k_scatter,
            total
        );
        println!(
            "RI-r2 semi-direct cache/profile: cache_write={:.3}s/{:.2} MB cache_read_J={:.3}s/{:.2} MB cache_read_K={:.3}s/{:.2} MB H_cache_build={:.3}s/{:.2} MB regenerated_J={} regenerated_K={} kept_batches={} effective_kept_batch={} fitted_shell_block_passes={}",
            self.cache_write,
            bytes_to_mb(self.cache_write_bytes),
            self.cache_read_j,
            bytes_to_mb(self.cache_read_j_bytes),
            self.cache_read_k,
            bytes_to_mb(self.cache_read_k_bytes),
            self.hfit_cache_build,
            bytes_to_mb(self.hfit_cache_bytes),
            self.regenerated_blocks_j,
            self.regenerated_blocks_k,
            self.kept_batches,
            self.effective_kept_batch,
            self.fitted_shell_block_passes,
        );
        println!(
            "RI-r2 semi-direct cache/fallbacks: H_build_source={} raw_read_failures={} raw_write_failures={} H_read_failures={} H_write_failures={}",
            if self.hfit_build_source.is_empty() {
                "none"
            } else {
                self.hfit_build_source
            },
            self.raw_cache_read_failures,
            self.raw_cache_write_failures,
            self.hfit_cache_read_failures,
            self.hfit_cache_write_failures,
        );
        println!(
            "RI-r2 semi-direct H-build profile: c_pack={:.3}s b_pack={:.3}s direct_gemm={:.3}s trans_gemm={:.3}s direct_scatter={:.3}s trans_scatter={:.3}s",
            self.fit_h_c_pack,
            self.fit_h_b_pack,
            self.fit_h_direct_gemm,
            self.fit_h_trans_gemm,
            self.fit_h_direct_scatter,
            self.fit_h_trans_scatter,
        );
    }
}

fn bytes_to_mb(bytes: usize) -> f64 {
    bytes as f64 / 1024.0 / 1024.0
}

#[derive(Clone, Copy)]
struct R2SemiDirectCachePlan {
    kept_batch: usize,
    use_batch_fitted_cache: bool,
}

/// 判断是否启用“当前 kept batch 的 fitted block 临时缓存”。
/// 该缓存仅在单个 batch 生命周期内存在，用于避免 J 的第二遍 block 生成。
fn r2_batch_fitted_cache_enabled(
    num_baspar: usize,
    kept_batch: usize,
    h_batch_bytes: usize,
    k_scratch_peak_bytes: usize,
    fixed_bytes: usize,
) -> bool {
    if num_baspar == 0 || kept_batch == 0 {
        return false;
    }
    let mode = std::env::var("REST_R2_DIRECT_BATCH_FITTED_CACHE")
        .map(|value| value.trim().to_ascii_lowercase())
        .unwrap_or_else(|_| "auto".to_string());
    let force_on = matches!(mode.as_str(), "1" | "true" | "yes" | "on" | "force");
    let force_off = matches!(mode.as_str(), "0" | "false" | "no" | "off");
    if force_off {
        println!("RI-r2 semi-direct batch fitted cache disabled by REST_R2_DIRECT_BATCH_FITTED_CACHE={mode}");
        return false;
    }

    let limit_mb = r2_memory_limit_mb();
    let factor = r2_direct_memory_factor();
    let kept_fraction = r2_direct_kept_memory_fraction();
    let total_budget_bytes = ((limit_mb * factor) * 1024.0 * 1024.0) as usize;
    let budget_bytes =
        ((total_budget_bytes.saturating_sub(fixed_bytes) as f64) * kept_fraction).floor() as usize;
    let b_batch_bytes = num_baspar
        .saturating_mul(kept_batch)
        .saturating_mul(std::mem::size_of::<f64>());
    let live_peak_bytes = h_batch_bytes
        .saturating_add(k_scratch_peak_bytes)
        .saturating_add(b_batch_bytes);
    let fits_budget = live_peak_bytes <= budget_bytes;

    if force_on && !fits_budget {
        println!(
            "RI-r2 semi-direct batch fitted cache requested but disabled by budget: live_peak={:.2} MB fixed={:.2} MB budget={:.2} MB",
            bytes_to_mb(live_peak_bytes),
            bytes_to_mb(fixed_bytes),
            bytes_to_mb(budget_bytes),
        );
        return false;
    }

    let enabled = if force_on { true } else { fits_budget };
    println!(
        "RI-r2 semi-direct batch fitted cache {}: mode={} batch_cache={:.2} MB H_batch={:.2} MB K_scratch_peak={:.2} MB live_peak={:.2} MB fixed={:.2} MB budget={:.2} MB",
        if enabled { "enabled" } else { "disabled" },
        mode,
        bytes_to_mb(b_batch_bytes),
        bytes_to_mb(h_batch_bytes),
        bytes_to_mb(k_scratch_peak_bytes),
        bytes_to_mb(live_peak_bytes),
        bytes_to_mb(fixed_bytes),
        bytes_to_mb(budget_bytes),
    );
    enabled
}

/// 构建 semi-direct 的轻量内存计划（当前主要用于选择 kept_batch 并打印预算信息）。
fn r2_make_semidirect_cache_plan(
    num_basis: usize,
    num_auxbas: usize,
    nkept: usize,
    occ_coeffs: &[MatrixFull<f64>],
    max_pair_len: usize,
    max_aux_shell_len: usize,
) -> R2SemiDirectCachePlan {
    let limit_mb = r2_memory_limit_mb();
    let factor = r2_direct_memory_factor();
    let (incore_peak_mb, _fitted_ri_mb, _metric_mb, _metric_factor_mb) =
        r2_incore_memory_estimate_mb(num_basis, num_auxbas);
    let incore_peak_bytes = (incore_peak_mb * 1024.0 * 1024.0) as usize;
    let num_baspar = (num_basis + 1) * num_basis / 2;
    let kept_batch = r2_direct_kept_block_size(
        nkept,
        num_basis,
        num_auxbas,
        num_baspar,
        occ_coeffs,
        max_pair_len,
        max_aux_shell_len,
    );
    let h_batch_values = occ_coeffs
        .iter()
        .map(|occ_coeff| {
            num_basis
                .saturating_mul(occ_coeff.size[1])
                .saturating_mul(kept_batch)
        })
        .sum::<usize>();
    let h_batch_bytes = h_batch_values.saturating_mul(std::mem::size_of::<f64>());
    let nocc_sum = occ_coeffs.iter().map(|coeff| coeff.size[1]).sum::<usize>();
    let nocc_max = occ_coeffs
        .iter()
        .map(|coeff| coeff.size[1])
        .max()
        .unwrap_or(1)
        .max(1);
    let k_scratch_peak_values = num_basis
        .saturating_mul(nocc_sum)
        .saturating_mul(kept_batch)
        .saturating_add(
            num_basis
                .saturating_mul(nocc_max)
                .saturating_mul(kept_batch),
        )
        .saturating_add(num_basis.saturating_mul(kept_batch));
    let k_scratch_peak_bytes = k_scratch_peak_values.saturating_mul(std::mem::size_of::<f64>());
    // 这里估算每个 worker 的 raw/fitted 临时块，需与实际 semi-direct worker 数保持一致。
    let workers = r2_direct_requested_workers();
    let fixed_metric_factor_bytes = num_auxbas
        .saturating_mul(nkept)
        .saturating_mul(std::mem::size_of::<f64>());
    let output_values = num_baspar
        .saturating_add(num_basis.saturating_mul(num_basis))
        .saturating_mul(occ_coeffs.len().max(1));
    let output_bytes = output_values.saturating_mul(std::mem::size_of::<f64>());
    let worker_raw_bytes = workers
        .saturating_mul(max_pair_len)
        .saturating_mul(max_aux_shell_len)
        .saturating_mul(2)
        .saturating_mul(std::mem::size_of::<f64>());
    let fixed_bytes = fixed_metric_factor_bytes
        .saturating_add(output_bytes)
        .saturating_add(worker_raw_bytes);
    let use_batch_fitted_cache = r2_batch_fitted_cache_enabled(
        num_baspar,
        kept_batch,
        h_batch_bytes,
        k_scratch_peak_bytes,
        fixed_bytes,
    );
    println!(
        "RI-r2 semi-direct memory plan (kept-batch-first): incore_peak={:.2} MB fixed={:.2} MB H_batch={:.2} MB K_scratch_peak={:.2} MB limit={:.2} MB factor={:.2} kept_fraction={:.2} max_aux_shell_len={}",
        bytes_to_mb(incore_peak_bytes),
        bytes_to_mb(fixed_bytes),
        bytes_to_mb(h_batch_bytes),
        bytes_to_mb(k_scratch_peak_bytes),
        limit_mb,
        factor,
        r2_direct_kept_memory_fraction(),
        max_aux_shell_len,
    );
    println!(
        "RI-r2 semi-direct cache plan (kept-batch-first): kept_batch={} batch_fitted_cache={}",
        kept_batch, use_batch_fitted_cache,
    );

    R2SemiDirectCachePlan {
        kept_batch,
        use_batch_fitted_cache,
    }
}

/// 将 batch 级 H 拟合矩阵重排为 `nao x (nocc*kept_len)`，并执行 `K += H H^T`。
fn r2_contract_k_from_h_fit_batch(
    h_fit_batch: &MatrixFull<f64>,
    kept_len: usize,
    occ_coeff: &MatrixFull<f64>,
    num_basis: usize,
    h_flat_scratch: &mut MatrixFull<f64>,
    k_full: &mut MatrixFull<f64>,
    timings: &mut R2SemiDirectTimings,
) {
    let nocc = occ_coeff.size[1];
    if nocc == 1 {
        let timer = Instant::now();
        _dsyrk(h_fit_batch, k_full, 'U', 'N', 1.0, 1.0);
        timings.k_dgemm += timer.elapsed().as_secs_f64();
        return;
    }

    let block_cols = std::env::var("REST_R2_DIRECT_K_PACK_COLS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(256);
    let kept_block = block_cols.div_ceil(nocc).max(1);
    let h_rows = num_basis * nocc;
    let mut kept_loc = 0_usize;
    while kept_loc < kept_len {
        let chunk_kept = (kept_len - kept_loc).min(kept_block);
        let chunk_cols = chunk_kept * nocc;
        r2_prepare_scratch_matrix(h_flat_scratch, [num_basis, chunk_cols], false);
        let timer = Instant::now();
        for chunk_q in 0..chunk_kept {
            let kept_idx = kept_loc + chunk_q;
            for occ in 0..nocc {
                let col = occ + nocc * chunk_q;
                let from_start = kept_idx * h_rows + occ * num_basis;
                let to_start = col * num_basis;
                h_flat_scratch.data[to_start..to_start + num_basis]
                    .copy_from_slice(&h_fit_batch.data[from_start..from_start + num_basis]);
            }
        }
        timings.k_pack += timer.elapsed().as_secs_f64();
        let timer = Instant::now();
        _dsyrk(h_flat_scratch, k_full, 'U', 'N', 1.0, 1.0);
        timings.k_dgemm += timer.elapsed().as_secs_f64();
        kept_loc += chunk_kept;
    }
}

struct R2RhoScratch {
    dm_block: MatrixFull<f64>,
    rho_delta: MatrixFull<f64>,
}

struct R2JScratch {
    j_block: MatrixFull<f64>,
}

struct R2BatchFittedBlock {
    global_pair_start: usize,
    fitted_q: MatrixFull<f64>,
}

impl Default for R2RhoScratch {
    fn default() -> Self {
        Self {
            dm_block: MatrixFull::empty(),
            rho_delta: MatrixFull::empty(),
        }
    }
}

impl Default for R2JScratch {
    fn default() -> Self {
        Self {
            j_block: MatrixFull::empty(),
        }
    }
}

/// 对单个 fitted shell block 执行 J-like 线性压缩：
/// `rho_q += B_q^T * D_pair`（含三角对的对称权重处理）。
fn r2_accumulate_rho_from_fitted_block(
    global_pair_start: usize,
    local_fitted: &MatrixFull<f64>,
    baspar2basbas: &[[usize; 2]],
    dm: &[MatrixFull<f64>],
    rho_kept: &mut MatrixFull<f64>,
    scratch: &mut R2RhoScratch,
) {
    let pair_len = local_fitted.size[0];
    let nkept = local_fitted.size[1];
    let spin_channel = dm.len();
    debug_assert_eq!(rho_kept.size, [nkept, spin_channel]);
    r2_prepare_scratch_matrix(&mut scratch.dm_block, [pair_len, spin_channel], false);
    for local_pair in 0..pair_len {
        let global_pair = global_pair_start + local_pair;
        let [mu, nu] = baspar2basbas[global_pair];
        for i_spin in 0..spin_channel {
            scratch.dm_block[(local_pair, i_spin)] = if mu == nu {
                dm[i_spin][(mu, nu)]
            } else {
                2.0 * dm[i_spin][(mu, nu)]
            };
        }
    }

    r2_prepare_scratch_matrix(&mut scratch.rho_delta, [nkept, spin_channel], false);
    scratch.rho_delta.to_matrixfullslicemut().lapack_dgemm(
        &local_fitted.to_matrixfullslice(),
        &scratch.dm_block.to_matrixfullslice(),
        'T',
        'N',
        1.0,
        0.0,
    );
    for i_spin in 0..spin_channel {
        for kept in 0..nkept {
            rho_kept[(kept, i_spin)] += scratch.rho_delta[(kept, i_spin)];
        }
    }
}

/// 使用当前 block 的 fitted 数据与 rho_q 回写 J：
/// `J_pair += B_q * rho_q`。
fn r2_contract_j_from_fitted_block(
    global_pair_start: usize,
    local_fitted: &MatrixFull<f64>,
    rho_matrix: &MatrixFull<f64>,
    j_upper: &mut [MatrixUpper<f64>],
    scratch: &mut R2JScratch,
) {
    let pair_len = local_fitted.size[0];
    let spin_channel = rho_matrix.size[1];
    r2_prepare_scratch_matrix(&mut scratch.j_block, [pair_len, spin_channel], false);
    scratch.j_block.to_matrixfullslicemut().lapack_dgemm(
        &local_fitted.to_matrixfullslice(),
        &rho_matrix.to_matrixfullslice(),
        'N',
        'N',
        1.0,
        0.0,
    );
    for local_pair in 0..pair_len {
        let global_pair = global_pair_start + local_pair;
        for i_spin in 0..spin_channel {
            j_upper[i_spin].data[global_pair] += scratch.j_block[(local_pair, i_spin)];
        }
    }
}

/// RI-r2 semi-direct 主实现（kept-batch-first）：
/// - Pass A: 生成 B_q 并累积 rho_q/H_q，然后完成 K_q
/// - Pass B: 重新生成 B_q 并完成 J_q
/// 该路径保持数值上使用 metric_eigh/pinv，不引入近似分解。
fn r2_jk_direct_from_shell_blocks_with_auxbas_threshold_kept_batch_first(
    ao_shells: &[RintShell],
    aux_shells: &[RintShell],
    dm: &[MatrixFull<f64>],
    coeff: Option<&[MatrixFull<f64>]>,
    occupation: Option<&[Vec<f64>]>,
    auxbas_threshold: f64,
) -> Option<(Vec<MatrixUpper<f64>>, Vec<MatrixUpper<f64>>)> {
    if ao_shells.is_empty() || aux_shells.is_empty() || dm.is_empty() {
        return None;
    }
    let num_basis = rint_shell_basis_count(ao_shells);
    let num_auxbas = rint_shell_basis_count(aux_shells);
    if num_basis == 0 || num_auxbas == 0 {
        return None;
    }
    let spin_channel = dm.len();
    let num_baspar = (num_basis + 1) * num_basis / 2;
    let occ_coeffs = coeff
        .zip(occupation)
        .and_then(|(coeff, occupation)| r2_prepare_occ_coeffs(coeff, occupation, num_basis))?;
    if occ_coeffs.len() != spin_channel {
        return None;
    }

    let mut timings = R2SemiDirectTimings::default();
    let timer = Instant::now();
    let aux_v = build_aux_metric_from_shell_blocks_parallel(
        aux_shells,
        num_auxbas,
        RiShellBlockKernel {
            int2c: int2c_r2_shell_block,
            int3c_into: int3c_r2_shell_block_batched_into,
        },
    );
    timings.metric_2c += timer.elapsed().as_secs_f64();
    let timer = Instant::now();
    let metric_factor = r2_aux_metric_pinv_factor_cached(&aux_v, auxbas_threshold)?;
    timings.metric_eigh += timer.elapsed().as_secs_f64();
    drop(aux_v);
    let nkept = metric_factor.size[1];
    println!("RI-r2 semi-direct metric factor: method=eigen/pinv nkept={nkept}");
    if nkept == 0 {
        return None;
    }
    let (_, baspar2basbas) = {
        let (_basbas2baspar, baspar2basbas) = prepare_baspair_map(num_basis);
        (_basbas2baspar, baspar2basbas)
    };
    let kernel = RiShellBlockKernel {
        int2c: int2c_r2_shell_block,
        int3c_into: int3c_r2_shell_block_batched_into_direct,
    };
    let max_pair_len = r2_max_shell_block_pair_len(ao_shells, num_basis);
    let max_aux_shell_len = r2_max_aux_shell_len(aux_shells);
    let plan = r2_make_semidirect_cache_plan(
        num_basis,
        num_auxbas,
        nkept,
        &occ_coeffs,
        max_pair_len,
        max_aux_shell_len,
    );
    let kept_block = plan.kept_batch.max(1).min(nkept);
    timings.effective_kept_batch = kept_block;
    timings.hfit_build_source = "kept_batch_first";
    println!(
        "RI-r2 semi-direct kept-batch-first path: kept_batch={} nkept={}",
        kept_block, nkept
    );

    let mut j_upper = vec![MatrixUpper::new(num_baspar, 0.0_f64); spin_channel];
    let mut k_full = (0..spin_channel)
        .map(|_| MatrixFull::new([num_basis, num_basis], 0.0_f64))
        .collect::<Vec<_>>();
    let mut h_flat_scratch = (0..spin_channel)
        .map(|_| MatrixFull::empty())
        .collect::<Vec<_>>();
    let mut rho_scratch = R2RhoScratch::default();
    let mut h_fit_scratch = (0..spin_channel)
        .map(|_| R2HFitScratch::default())
        .collect::<Vec<_>>();
    let mut j_scratch = R2JScratch::default();
    let mut h_fit_batches = occ_coeffs
        .iter()
        .map(|_| MatrixFull::empty())
        .collect::<Vec<_>>();
    let mut kept_start = 0_usize;
    while kept_start < nkept {
        let kept_len = (nkept - kept_start).min(kept_block);
        timings.kept_batches += 1;
        let mut rho_q = MatrixFull::new([kept_len, spin_channel], 0.0_f64);
        for i_spin in 0..spin_channel {
            r2_prepare_scratch_matrix(
                &mut h_fit_batches[i_spin],
                [num_basis * occ_coeffs[i_spin].size[1], kept_len],
                true,
            );
        }
        let mut batch_fitted_blocks = Vec::new();

        let pass_a_timer = Instant::now();
        timings.fitted_shell_block_passes += 1;
        for_each_fitted_ri3fn_shell_block_rayon_kept_range(
            ao_shells,
            aux_shells,
            num_auxbas,
            kernel,
            &metric_factor,
            kept_start,
            kept_len,
            |global_pair_start, pair_len, local_fitted_q| {
                debug_assert_eq!(local_fitted_q.size[1], kept_len);
                r2_accumulate_rho_from_fitted_block(
                    global_pair_start,
                    &local_fitted_q,
                    &baspar2basbas,
                    dm,
                    &mut rho_q,
                    &mut rho_scratch,
                );
                for i_spin in 0..spin_channel {
                    let hfit_step =
                        r2_accumulate_hfit_from_fitted_shell_block_kept_range_with_scratch(
                            global_pair_start,
                            pair_len,
                            &local_fitted_q,
                            0,
                            kept_len,
                            0,
                            &occ_coeffs[i_spin],
                            &mut h_fit_batches[i_spin],
                            num_basis,
                            &mut h_fit_scratch[i_spin],
                        );
                    timings.accumulate_hfit_step(hfit_step);
                }
                if plan.use_batch_fitted_cache {
                    batch_fitted_blocks.push(R2BatchFittedBlock {
                        global_pair_start,
                        fitted_q: local_fitted_q,
                    });
                }
            },
        );
        let pass_a_elapsed = pass_a_timer.elapsed().as_secs_f64();
        timings.raw_3c += pass_a_elapsed;
        timings.fit_h += pass_a_elapsed;

        for i_spin in 0..spin_channel {
            r2_contract_k_from_h_fit_batch(
                &h_fit_batches[i_spin],
                kept_len,
                &occ_coeffs[i_spin],
                num_basis,
                &mut h_flat_scratch[i_spin],
                &mut k_full[i_spin],
                &mut timings,
            );
        }

        let rho_matrix_q = rho_q;
        let pass_b_timer = Instant::now();
        if plan.use_batch_fitted_cache && !batch_fitted_blocks.is_empty() {
            for cached in batch_fitted_blocks.iter() {
                r2_contract_j_from_fitted_block(
                    cached.global_pair_start,
                    &cached.fitted_q,
                    &rho_matrix_q,
                    &mut j_upper,
                    &mut j_scratch,
                );
            }
        } else {
            timings.fitted_shell_block_passes += 1;
            for_each_fitted_ri3fn_shell_block_rayon_kept_range(
                ao_shells,
                aux_shells,
                num_auxbas,
                kernel,
                &metric_factor,
                kept_start,
                kept_len,
                |global_pair_start, _pair_len, local_fitted_q| {
                    debug_assert_eq!(local_fitted_q.size[1], kept_len);
                    r2_contract_j_from_fitted_block(
                        global_pair_start,
                        &local_fitted_q,
                        &rho_matrix_q,
                        &mut j_upper,
                        &mut j_scratch,
                    );
                },
            );
        }
        timings.j_contract += pass_b_timer.elapsed().as_secs_f64();
        kept_start += kept_len;
    }

    let mut k_upper = Vec::with_capacity(spin_channel);
    for mut k_full_spin in k_full {
        let timer = Instant::now();
        k_upper.push(k_full_spin.to_matrixupper());
        timings.k_scatter += timer.elapsed().as_secs_f64();
    }
    timings.print();

    debug_assert_eq!(j_upper[0].size, num_baspar);
    Some((j_upper, k_upper))
}

fn r2_jk_direct_from_shell_blocks_with_auxbas_threshold(
    ao_shells: &[RintShell],
    aux_shells: &[RintShell],
    dm: &[MatrixFull<f64>],
    coeff: Option<&[MatrixFull<f64>]>,
    occupation: Option<&[Vec<f64>]>,
    auxbas_threshold: f64,
) -> Option<(Vec<MatrixUpper<f64>>, Vec<MatrixUpper<f64>>)> {
    r2_jk_direct_from_shell_blocks_with_auxbas_threshold_kept_batch_first(
        ao_shells,
        aux_shells,
        dm,
        coeff,
        occupation,
        auxbas_threshold,
    )
}

pub fn vj_upper_with_rimatr_r_sync(
    ri3fn: &Option<(MatrixFull<f64>, MatrixFull<usize>, Vec<[usize; 2]>)>,
    dm: &Vec<MatrixFull<f64>>,
    spin_channel: usize,
    scaling_factor: f64,
) -> Vec<MatrixUpper<f64>> {
    vj_upper_with_rimatr_sync_internal(ri3fn, dm, spin_channel, scaling_factor)
}
pub fn vk_upper_with_rimatr_r_sync(
    ri3fn: &Option<(MatrixFull<f64>, MatrixFull<usize>, Vec<[usize; 2]>)>,
    dm: &Vec<MatrixFull<f64>>,
    spin_channel: usize,
    scaling_factor: f64,
) -> Vec<MatrixUpper<f64>> {
    vk_upper_with_rimatr_sync_internal(ri3fn, dm, spin_channel, scaling_factor)
}
pub fn vj_upper_with_rimatr_r2_sync(
    ri3fn: &Option<(MatrixFull<f64>, MatrixFull<usize>, Vec<[usize; 2]>)>,
    dm: &Vec<MatrixFull<f64>>,
    spin_channel: usize,
    scaling_factor: f64,
) -> Vec<MatrixUpper<f64>> {
    vj_upper_with_rimatr_sync_v02(ri3fn, dm, spin_channel, scaling_factor)
}
pub fn vk_upper_with_rimatr_r2_sync(
    ri3fn: &Option<(MatrixFull<f64>, MatrixFull<usize>, Vec<[usize; 2]>)>,
    dm: &Vec<MatrixFull<f64>>,
    spin_channel: usize,
    scaling_factor: f64,
) -> Vec<MatrixUpper<f64>> {
    vk_upper_with_rimatr_r2_linear_sync_internal(ri3fn, dm, spin_channel, scaling_factor)
}
pub fn lib_vee_rhf_r(geom: &GeomCell, basis4elem: &[Basis4Elem], p_rhf: &MatrixFull<f64>) -> f64 {
    let observables = lib_vee_rhf_r_observables(geom, basis4elem, p_rhf);
    print_rhf_vee_observables("r", &observables);
    observables.total
}
pub fn lib_vee_rhf_r_observables_exact(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    p_rhf: &MatrixFull<f64>,
) -> RhfVeeObservables {
    let (bfs, p_cart) = load_cartesian_rhf_basis_and_density_shell_shared(
        geom,
        basis4elem,
        p_rhf,
        "lib_vee_rhf_r_exact",
    );
    let (j, k) = build_jk_from_p_kernel(&bfs, &p_cart, eri_ao_4c_r);
    let (ej, ek, e) = ej_ek_from_p_jk_rhf(&p_cart, &j, &k);
    RhfVeeObservables { ej, ek, total: e }
}
pub fn lib_vee_rhf_r_observables_with_auxbasis(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    auxbasis4elem: &[Basis4Elem],
    p_rhf: &MatrixFull<f64>,
) -> RhfVeeObservables {
    let exact = || lib_vee_rhf_r_observables_exact(geom, basis4elem, p_rhf);
    let (bfs, p_cart) = load_cartesian_rhf_basis_and_density_shell_shared(
        geom,
        basis4elem,
        p_rhf,
        "lib_vee_rhf_r_ri",
    );

    let aux_bfs = match load_aux_molecule_shell_shared_from_raw(geom, auxbasis4elem) {
        Ok(aux_bfs) if !aux_bfs.is_empty() => aux_bfs,
        _ => return exact(),
    };
    let ri3fn = prepare_rimatr_for_r_sync(&bfs, &aux_bfs);
    if ri3fn.is_none() {
        return exact();
    }
    let dm = vec![p_cart.clone()];
    let j_upper = vj_upper_with_rimatr_r_sync(&ri3fn, &dm, 1, 1.0);
    let k_upper = vk_upper_with_rimatr_r_sync(&ri3fn, &dm, 1, 1.0);
    let Some(j) = j_upper.get(0).and_then(|mat| mat.to_matrixfull()) else {
        return exact();
    };
    let Some(k) = k_upper.get(0).and_then(|mat| mat.to_matrixfull()) else {
        return exact();
    };
    let (ej, ek, e) = ej_ek_from_p_jk_rhf(&p_cart, &j, &k);
    RhfVeeObservables { ej, ek, total: e }
}
pub fn lib_vee_rhf_r_observables(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    p_rhf: &MatrixFull<f64>,
) -> RhfVeeObservables {
    lib_vee_rhf_r_observables_exact(geom, basis4elem, p_rhf)
}
fn transform_spin_densities_to_cartesian(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    dm_spin: &[MatrixFull<f64>],
    nao_cart: usize,
    caller: &str,
) -> Vec<MatrixFull<f64>> {
    dm_spin
        .iter()
        .enumerate()
        .map(|(i_spin, dm)| {
            transform_density_to_cartesian_shell_shared(
                geom,
                basis4elem,
                dm,
                nao_cart,
                &format!("{caller}[spin={i_spin}]"),
                false,
            )
        })
        .collect()
}
fn sum_density_matrices(dm_spin: &[MatrixFull<f64>]) -> MatrixFull<f64> {
    assert!(
        !dm_spin.is_empty(),
        "sum_density_matrices expects at least one density matrix"
    );
    let mut total = dm_spin[0].clone();
    for dm in dm_spin.iter().skip(1) {
        total
            .data
            .iter_mut()
            .zip(dm.data.iter())
            .for_each(|(to, from)| *to += *from);
    }
    total
}
fn upper_mats_to_full(mats: &[MatrixUpper<f64>], caller: &str) -> Option<Vec<MatrixFull<f64>>> {
    mats.iter()
        .enumerate()
        .map(|(i_spin, mat)| {
            mat.to_matrixfull().or_else(|| {
                eprintln!("{caller}: failed to expand MatrixUpper for spin channel {i_spin}");
                None
            })
        })
        .collect()
}
fn uhf_jk_matrices_to_current_ao(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    target_nao: usize,
    j_total_cart: &MatrixFull<f64>,
    k_spin_cart: &[MatrixFull<f64>],
    caller: &str,
) -> UhfVeeJkMatrices {
    let j_total = transform_operator_from_cartesian_shell_shared(
        geom,
        basis4elem,
        j_total_cart,
        target_nao,
        &format!("{caller}[J_total]"),
        false,
    );
    let k_spin = k_spin_cart
        .iter()
        .enumerate()
        .map(|(spin, k)| {
            transform_operator_from_cartesian_shell_shared(
                geom,
                basis4elem,
                k,
                target_nao,
                &format!("{caller}[K_spin={spin}]"),
                false,
            )
        })
        .collect();
    UhfVeeJkMatrices { j_total, k_spin }
}
pub fn lib_vee_uhf_r_observables_exact(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    dm_spin: &[MatrixFull<f64>],
) -> RhfVeeObservables {
    assert_eq!(
        dm_spin.len(),
        2,
        "lib_vee_uhf_r_observables_exact expects alpha and beta density matrices"
    );
    let ao_shells = load_molecule_rint_shells_from_raw(geom, basis4elem)
        .expect("failed to build rint shells from GeomCell/BasCell");
    let bfs = crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&ao_shells)
        .expect("failed to expand rint shells to BasisFunction list");
    let dm_cart = transform_spin_densities_to_cartesian(
        geom,
        basis4elem,
        dm_spin,
        bfs.len(),
        "lib_vee_uhf_r_exact",
    );
    let p_total = sum_density_matrices(&dm_cart);
    let (j_total, _) = build_jk_from_p_kernel(&bfs, &p_total, eri_ao_4c_r);
    let (_, k_alpha) = build_jk_from_p_kernel(&bfs, &dm_cart[0], eri_ao_4c_r);
    let (_, k_beta) = build_jk_from_p_kernel(&bfs, &dm_cart[1], eri_ao_4c_r);
    let k_spin = vec![k_alpha, k_beta];
    let (ej, ek, total) = ej_ek_from_p_jk_uhf(&dm_cart, &j_total, &k_spin);
    RhfVeeObservables { ej, ek, total }
}
pub fn lib_vee_uhf_r_observables_with_auxbasis(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    auxbasis4elem: &[Basis4Elem],
    dm_spin: &[MatrixFull<f64>],
) -> RhfVeeObservables {
    let exact = || lib_vee_uhf_r_observables_exact(geom, basis4elem, dm_spin);
    assert_eq!(
        dm_spin.len(),
        2,
        "lib_vee_uhf_r_observables_with_auxbasis expects alpha and beta density matrices"
    );

    let ao_shells = load_molecule_rint_shells_from_raw(geom, basis4elem)
        .expect("failed to build rint shells from GeomCell/BasCell");
    let bfs = crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&ao_shells)
        .expect("failed to expand rint shells to BasisFunction list");
    let dm_cart = transform_spin_densities_to_cartesian(
        geom,
        basis4elem,
        dm_spin,
        bfs.len(),
        "lib_vee_uhf_r_ri",
    );

    let aux_bfs = match load_aux_molecule_shell_shared_from_raw(geom, auxbasis4elem) {
        Ok(aux_bfs) if !aux_bfs.is_empty() => aux_bfs,
        _ => return exact(),
    };

    let ri3fn = prepare_rimatr_for_r_sync(&bfs, &aux_bfs);
    if ri3fn.is_none() {
        return exact();
    }

    let j_spin_upper = vj_upper_with_rimatr_r_sync(&ri3fn, &dm_cart, 2, 1.0);
    let k_spin_upper = vk_upper_with_rimatr_r_sync(&ri3fn, &dm_cart, 2, 1.0);
    let Some(j_spin_full) =
        upper_mats_to_full(&j_spin_upper, "lib_vee_uhf_r_observables_with_auxbasis[J]")
    else {
        return exact();
    };
    let Some(k_spin_full) =
        upper_mats_to_full(&k_spin_upper, "lib_vee_uhf_r_observables_with_auxbasis[K]")
    else {
        return exact();
    };

    let j_total = sum_density_matrices(&j_spin_full);
    let (ej, ek, total) = ej_ek_from_p_jk_uhf(&dm_cart, &j_total, &k_spin_full);
    RhfVeeObservables { ej, ek, total }
}
pub fn lib_vee_uhf_r_observables(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    dm_spin: &[MatrixFull<f64>],
) -> RhfVeeObservables {
    lib_vee_uhf_r_observables_exact(geom, basis4elem, dm_spin)
}
pub fn lib_vee_rhf_r2(geom: &GeomCell, basis4elem: &[Basis4Elem], p_rhf: &MatrixFull<f64>) -> f64 {
    let observables = lib_vee_rhf_r2_observables(geom, basis4elem, p_rhf);
    print_rhf_vee_observables("r2", &observables);
    observables.total
}
fn lib_vee_rhf_r2_observables_exact(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    p_rhf: &MatrixFull<f64>,
) -> RhfVeeObservables {
    let ao_shells =
        crate::lib_rint::basis::load_molecule_contracted_rint_shells_from_raw(geom, basis4elem)
            .expect("failed to build contracted rint shells from GeomCell/BasCell");
    let nao_cart = rint_contracted_shell_basis_count(&ao_shells);
    let p_cart = transform_density_to_cartesian_shell_shared(
        geom,
        basis4elem,
        p_rhf,
        nao_cart,
        "lib_vee_rhf_r2_exact",
        false,
    );
    let (j2, k2) = build_jk_from_r2_contracted_shell_block_full_tensor(&ao_shells, &p_cart);
    let (ej2, ek2, e2) = ej_ek_from_p_jk_rhf(&p_cart, &j2, &k2);
    RhfVeeObservables {
        ej: ej2,
        ek: ek2,
        total: e2,
    }
}
fn lib_vee_rhf_r2_jk_matrices_exact(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    p_rhf: &MatrixFull<f64>,
) -> (MatrixFull<f64>, MatrixFull<f64>) {
    let target_nao = p_rhf.size[0];
    let ao_shells =
        crate::lib_rint::basis::load_molecule_contracted_rint_shells_from_raw(geom, basis4elem)
            .expect("failed to build contracted rint shells from GeomCell/BasCell");
    let nao_cart = rint_contracted_shell_basis_count(&ao_shells);
    let p_cart = transform_density_to_cartesian_shell_shared(
        geom,
        basis4elem,
        p_rhf,
        nao_cart,
        "lib_vee_rhf_r2_jk_matrices_exact",
        false,
    );
    let (j2_cart, k2_cart) =
        build_jk_from_r2_contracted_shell_block_full_tensor(&ao_shells, &p_cart);
    (
        transform_operator_from_cartesian_shell_shared(
            geom,
            basis4elem,
            &j2_cart,
            target_nao,
            "lib_vee_rhf_r2_jk_matrices_exact[J]",
            false,
        ),
        transform_operator_from_cartesian_shell_shared(
            geom,
            basis4elem,
            &k2_cart,
            target_nao,
            "lib_vee_rhf_r2_jk_matrices_exact[K]",
            false,
        ),
    )
}
pub fn lib_vee_rhf_r2_observables_with_auxbasis(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    auxbasis4elem: &[Basis4Elem],
    p_rhf: &MatrixFull<f64>,
) -> RhfVeeObservables {
    let exact = || lib_vee_rhf_r2_observables_exact(geom, basis4elem, p_rhf);

    let (ao_shells, _bfs, p_cart) = load_cartesian_rhf_rint_shells_basis_and_density_shell_shared(
        geom,
        basis4elem,
        p_rhf,
        "lib_vee_rhf_r2",
    );

    let aux_shells = match load_aux_rint_shells_from_raw(geom, auxbasis4elem) {
        Ok(aux_shells) if !aux_shells.is_empty() => aux_shells,
        _ => return exact(),
    };
    let dm = vec![p_cart.clone()];
    if r2_use_semidirect(
        rint_shell_basis_count(&ao_shells),
        rint_shell_basis_count(&aux_shells),
    ) {
        if let Some((j_upper, k_upper)) = r2_jk_direct_from_shell_blocks_with_auxbas_threshold(
            &ao_shells,
            &aux_shells,
            &dm,
            None,
            None,
            AUXBAS_THRESHOLD,
        ) {
            let Some(j2) = j_upper.get(0).and_then(|mat| mat.to_matrixfull()) else {
                return exact();
            };
            let Some(k2) = k_upper.get(0).and_then(|mat| mat.to_matrixfull()) else {
                return exact();
            };
            let (ej2, ek2, e2) = ej_ek_from_p_jk_rhf(&p_cart, &j2, &k2);
            return RhfVeeObservables {
                ej: ej2,
                ek: ek2,
                total: e2,
            };
        }
    }
    let ri3fn = prepare_rimatr_for_r2_shell_blocks_sync(&ao_shells, &aux_shells);
    if ri3fn.is_none() {
        return exact();
    }

    let j_upper = vj_upper_with_rimatr_r2_sync(&ri3fn, &dm, 1, 1.0);
    let k_upper = vk_upper_with_rimatr_r2_sync(&ri3fn, &dm, 1, 1.0);

    let Some(j2) = j_upper.get(0).and_then(|mat| mat.to_matrixfull()) else {
        return exact();
    };
    let Some(k2) = k_upper.get(0).and_then(|mat| mat.to_matrixfull()) else {
        return exact();
    };

    let (ej2, ek2, e2) = ej_ek_from_p_jk_rhf(&p_cart, &j2, &k2);

    RhfVeeObservables {
        ej: ej2,
        ek: ek2,
        total: e2,
    }
}
pub fn lib_vee_rhf_r2_observables_with_auxbasis_advanced(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    auxbasis4elem: &[Basis4Elem],
    p_rhf: &MatrixFull<f64>,
    coeff_spin: Option<&[MatrixFull<f64>; 2]>,
    occupation: Option<&[Vec<f64>; 2]>,
) -> RhfVeeObservables {
    let (j, k) = lib_vee_rhf_r2_jk_matrices_with_auxbasis_advanced(
        geom,
        basis4elem,
        auxbasis4elem,
        p_rhf,
        coeff_spin,
        occupation,
    );
    let (ej2, ek2, e2) = ej_ek_from_p_jk_rhf(p_rhf, &j, &k);
    RhfVeeObservables {
        ej: ej2,
        ek: ek2,
        total: e2,
    }
}

pub fn lib_vee_rhf_r2_jk_matrices_with_auxbasis_advanced(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    auxbasis4elem: &[Basis4Elem],
    p_rhf: &MatrixFull<f64>,
    coeff_spin: Option<&[MatrixFull<f64>; 2]>,
    occupation: Option<&[Vec<f64>; 2]>,
) -> (MatrixFull<f64>, MatrixFull<f64>) {
    let target_nao = p_rhf.size[0];
    let to_current_ao = |matrix: &MatrixFull<f64>, label: &str| {
        transform_operator_from_cartesian_shell_shared(
            geom, basis4elem, matrix, target_nao, label, false,
        )
    };
    let exact = || lib_vee_rhf_r2_jk_matrices_exact(geom, basis4elem, p_rhf);

    let (ao_shells, _bfs, p_cart) = load_cartesian_rhf_rint_shells_basis_and_density_shell_shared(
        geom,
        basis4elem,
        p_rhf,
        "lib_vee_rhf_r2_jk_matrices_advanced",
    );
    let aux_shells = match load_aux_rint_shells_from_raw(geom, auxbasis4elem) {
        Ok(aux_shells) if !aux_shells.is_empty() => aux_shells,
        _ => return exact(),
    };
    let use_semidirect = r2_use_semidirect(
        rint_shell_basis_count(&ao_shells),
        rint_shell_basis_count(&aux_shells),
    );
    let dm = vec![p_cart.clone()];

    if use_semidirect {
        let semidirect_result =
            if let (Some(coeff_spin), Some(occupation)) = (coeff_spin, occupation) {
                let num_basis = rint_shell_basis_count(&ao_shells);
                let coeff_cart_alpha = transform_mo_coeff_to_cartesian_shell_shared(
                    geom,
                    basis4elem,
                    &coeff_spin[0],
                    num_basis,
                    "lib_vee_rhf_r2_jk_matrices_advanced[alpha]",
                    false,
                );
                let occ = [occupation[0].clone()];
                r2_jk_direct_from_shell_blocks_with_auxbas_threshold(
                    &ao_shells,
                    &aux_shells,
                    &dm,
                    Some(&[coeff_cart_alpha][..]),
                    Some(&occ),
                    AUXBAS_THRESHOLD,
                )
            } else {
                r2_jk_direct_from_shell_blocks_with_auxbas_threshold(
                    &ao_shells,
                    &aux_shells,
                    &dm,
                    None,
                    None,
                    AUXBAS_THRESHOLD,
                )
            };
        if let Some((j_upper, k_upper)) = semidirect_result {
            if let (Some(j2), Some(k2)) = (
                j_upper.get(0).and_then(|mat| mat.to_matrixfull()),
                k_upper.get(0).and_then(|mat| mat.to_matrixfull()),
            ) {
                return (
                    to_current_ao(&j2, "lib_vee_rhf_r2_jk_matrices_advanced_direct[J]"),
                    to_current_ao(&k2, "lib_vee_rhf_r2_jk_matrices_advanced_direct[K]"),
                );
            }
        }
    }

    let ri3fn = prepare_rimatr_for_r2_shell_blocks_sync(&ao_shells, &aux_shells);
    let Some(ri3fn) = ri3fn else {
        return exact();
    };
    let ri3fn = Some(ri3fn);
    let j_upper = vj_upper_with_rimatr_r2_sync(&ri3fn, &dm, 1, 1.0);
    let k_upper = if let (Some(coeff_spin), Some(occupation)) = (coeff_spin, occupation) {
        let num_basis = rint_shell_basis_count(&ao_shells);
        let coeff_cart_alpha = transform_mo_coeff_to_cartesian_shell_shared(
            geom,
            basis4elem,
            &coeff_spin[0],
            num_basis,
            "lib_vee_rhf_r2_jk_matrices_advanced[alpha]",
            false,
        );
        let coeff_cart = [coeff_cart_alpha, MatrixFull::empty()];
        let occ_cart = [occupation[0].clone(), Vec::new()];
        let num_elec_alpha = occupation[0].iter().sum::<f64>();
        let num_elec = [num_elec_alpha, num_elec_alpha, 0.0_f64];
        vk_upper_with_rimatr_sync_v03(&ri3fn, &coeff_cart, &num_elec, &occ_cart, 1, 1.0)
    } else {
        vk_upper_with_rimatr_r2_sync(&ri3fn, &dm, 1, 1.0)
    };

    let Some(j2) = j_upper.get(0).and_then(|mat| mat.to_matrixfull()) else {
        return exact();
    };
    let Some(k2) = k_upper.get(0).and_then(|mat| mat.to_matrixfull()) else {
        return exact();
    };
    (
        to_current_ao(&j2, "lib_vee_rhf_r2_jk_matrices_advanced[J]"),
        to_current_ao(&k2, "lib_vee_rhf_r2_jk_matrices_advanced[K]"),
    )
}

pub fn lib_vee_rhf_r2_jk_matrices_advanced(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    p_rhf: &MatrixFull<f64>,
    coeff_spin: Option<&[MatrixFull<f64>; 2]>,
    occupation: Option<&[Vec<f64>; 2]>,
) -> (MatrixFull<f64>, MatrixFull<f64>) {
    let auxbasis4elem = build_default_r2_etb_auxbasis(geom, basis4elem)
        .expect("failed to build default ETB(beta=1.7) auxiliary basis for r2 RI path");
    lib_vee_rhf_r2_jk_matrices_with_auxbasis_advanced(
        geom,
        basis4elem,
        &auxbasis4elem,
        p_rhf,
        coeff_spin,
        occupation,
    )
}

/// Compute RHF R2 observables and their variational J/K matrices from one RI
/// construction.  The R2-specific K contraction is density-linear, so the
/// energy contraction and returned potential remain derivative-consistent.
pub fn lib_vee_rhf_r2_intermediate(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    p_rhf: &MatrixFull<f64>,
) -> RhfVeeJkIntermediate {
    let (j, k) = lib_vee_rhf_r2_jk_matrices_advanced(geom, basis4elem, p_rhf, None, None);
    let (ej, ek, total) = ej_ek_from_p_jk_rhf(p_rhf, &j, &k);
    RhfVeeJkIntermediate {
        observables: RhfVeeObservables { ej, ek, total },
        j,
        k,
    }
}

pub fn lib_vee_rhf_r2_observables(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    p_rhf: &MatrixFull<f64>,
) -> RhfVeeObservables {
    let auxbasis4elem = build_default_r2_etb_auxbasis(geom, basis4elem)
        .expect("failed to build default ETB(beta=1.7) auxiliary basis for r2 RI path");
    lib_vee_rhf_r2_observables_with_auxbasis(geom, basis4elem, &auxbasis4elem, p_rhf)
}
pub fn lib_vee_rhf_r2_observables_advanced(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    p_rhf: &MatrixFull<f64>,
    coeff_spin: Option<&[MatrixFull<f64>; 2]>,
    occupation: Option<&[Vec<f64>; 2]>,
) -> RhfVeeObservables {
    let auxbasis4elem = build_default_r2_etb_auxbasis(geom, basis4elem)
        .expect("failed to build default ETB(beta=1.7) auxiliary basis for r2 RI path");
    lib_vee_rhf_r2_observables_with_auxbasis_advanced(
        geom,
        basis4elem,
        &auxbasis4elem,
        p_rhf,
        coeff_spin,
        occupation,
    )
}
fn lib_vee_uhf_r2_observables_exact(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    dm_spin: &[MatrixFull<f64>],
) -> RhfVeeObservables {
    assert_eq!(
        dm_spin.len(),
        2,
        "lib_vee_uhf_r2_observables_exact expects alpha and beta density matrices"
    );
    let ao_shells =
        crate::lib_rint::basis::load_molecule_contracted_rint_shells_from_raw(geom, basis4elem)
            .expect("failed to build contracted rint shells from GeomCell/BasCell");
    let nao_cart = rint_contracted_shell_basis_count(&ao_shells);
    let dm_cart = transform_spin_densities_to_cartesian(
        geom,
        basis4elem,
        dm_spin,
        nao_cart,
        "lib_vee_uhf_r2_exact",
    );
    let p_total = sum_density_matrices(&dm_cart);
    let eri4 = int4c_r2_full_from_contracted_shell_blocks(&ao_shells);
    let (j_total, _) = build_jk_from_full_4c_tensor(&eri4, &p_total, nao_cart);
    let (_, k_alpha) = build_jk_from_full_4c_tensor(&eri4, &dm_cart[0], nao_cart);
    let (_, k_beta) = build_jk_from_full_4c_tensor(&eri4, &dm_cart[1], nao_cart);
    let k_spin = vec![k_alpha, k_beta];
    let (ej, ek, total) = ej_ek_from_p_jk_uhf(&dm_cart, &j_total, &k_spin);
    RhfVeeObservables { ej, ek, total }
}
pub fn lib_vee_uhf_r2_observables_with_auxbasis(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    auxbasis4elem: &[Basis4Elem],
    dm_spin: &[MatrixFull<f64>],
) -> RhfVeeObservables {
    let exact = || lib_vee_uhf_r2_observables_exact(geom, basis4elem, dm_spin);
    assert_eq!(
        dm_spin.len(),
        2,
        "lib_vee_uhf_r2_observables_with_auxbasis expects alpha and beta density matrices"
    );

    let ao_shells = load_molecule_rint_shells_from_raw(geom, basis4elem)
        .expect("failed to build rint shells from GeomCell/BasCell");
    let bfs = crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&ao_shells)
        .expect("failed to expand rint shells to BasisFunction list");
    let dm_cart = transform_spin_densities_to_cartesian(
        geom,
        basis4elem,
        dm_spin,
        bfs.len(),
        "lib_vee_uhf_r2_ri",
    );

    let aux_shells = match load_aux_rint_shells_from_raw(geom, auxbasis4elem) {
        Ok(aux_shells) if !aux_shells.is_empty() => aux_shells,
        _ => return exact(),
    };
    if r2_use_semidirect(
        rint_shell_basis_count(&ao_shells),
        rint_shell_basis_count(&aux_shells),
    ) {
        if let Some((j_spin_upper, k_spin_upper)) =
            r2_jk_direct_from_shell_blocks_with_auxbas_threshold(
                &ao_shells,
                &aux_shells,
                &dm_cart,
                None,
                None,
                AUXBAS_THRESHOLD,
            )
        {
            let Some(j_spin_full) = upper_mats_to_full(
                &j_spin_upper,
                "lib_vee_uhf_r2_observables_with_auxbasis_direct[J]",
            ) else {
                return exact();
            };
            let Some(k_spin_full) = upper_mats_to_full(
                &k_spin_upper,
                "lib_vee_uhf_r2_observables_with_auxbasis_direct[K]",
            ) else {
                return exact();
            };
            let j_total = sum_density_matrices(&j_spin_full);
            let (ej, ek, total) = ej_ek_from_p_jk_uhf(&dm_cart, &j_total, &k_spin_full);
            return RhfVeeObservables { ej, ek, total };
        }
    }
    let ri3fn = prepare_rimatr_for_r2_shell_blocks_sync(&ao_shells, &aux_shells);
    if ri3fn.is_none() {
        return exact();
    }

    let j_spin_upper = vj_upper_with_rimatr_r2_sync(&ri3fn, &dm_cart, 2, 1.0);
    let k_spin_upper = vk_upper_with_rimatr_r2_sync(&ri3fn, &dm_cart, 2, 1.0);
    let Some(j_spin_full) =
        upper_mats_to_full(&j_spin_upper, "lib_vee_uhf_r2_observables_with_auxbasis[J]")
    else {
        return exact();
    };
    let Some(k_spin_full) =
        upper_mats_to_full(&k_spin_upper, "lib_vee_uhf_r2_observables_with_auxbasis[K]")
    else {
        return exact();
    };

    let j_total = sum_density_matrices(&j_spin_full);
    let (ej, ek, total) = ej_ek_from_p_jk_uhf(&dm_cart, &j_total, &k_spin_full);
    RhfVeeObservables { ej, ek, total }
}
pub fn lib_vee_uhf_r2_observables_with_auxbasis_advanced(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    auxbasis4elem: &[Basis4Elem],
    dm_spin: &[MatrixFull<f64>],
    coeff_spin: Option<&[MatrixFull<f64>; 2]>,
    occupation: Option<&[Vec<f64>; 2]>,
) -> RhfVeeObservables {
    let jk = lib_vee_uhf_r2_jk_matrices_with_auxbasis_advanced(
        geom,
        basis4elem,
        auxbasis4elem,
        dm_spin,
        coeff_spin,
        occupation,
    );
    let (ej, ek, total) = ej_ek_from_p_jk_uhf(dm_spin, &jk.j_total, &jk.k_spin);
    RhfVeeObservables { ej, ek, total }
}
pub fn lib_vee_uhf_r2_observables(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    dm_spin: &[MatrixFull<f64>],
) -> RhfVeeObservables {
    let auxbasis4elem = build_default_r2_etb_auxbasis(geom, basis4elem)
        .expect("failed to build default ETB(beta=1.7) auxiliary basis for r2 RI path");
    lib_vee_uhf_r2_observables_with_auxbasis(geom, basis4elem, &auxbasis4elem, dm_spin)
}
pub fn lib_vee_uhf_r2_observables_advanced(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    dm_spin: &[MatrixFull<f64>],
    coeff_spin: Option<&[MatrixFull<f64>; 2]>,
    occupation: Option<&[Vec<f64>; 2]>,
) -> RhfVeeObservables {
    let auxbasis4elem = build_default_r2_etb_auxbasis(geom, basis4elem)
        .expect("failed to build default ETB(beta=1.7) auxiliary basis for r2 RI path");
    lib_vee_uhf_r2_observables_with_auxbasis_advanced(
        geom,
        basis4elem,
        &auxbasis4elem,
        dm_spin,
        coeff_spin,
        occupation,
    )
}
fn lib_vee_uhf_r2_jk_matrices_exact(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    dm_spin: &[MatrixFull<f64>],
) -> UhfVeeJkMatrices {
    assert_eq!(
        dm_spin.len(),
        2,
        "lib_vee_uhf_r2_jk_matrices_exact expects alpha and beta density matrices"
    );
    let target_nao = dm_spin[0].size[0];
    let ao_shells =
        crate::lib_rint::basis::load_molecule_contracted_rint_shells_from_raw(geom, basis4elem)
            .expect("failed to build contracted rint shells from GeomCell/BasCell");
    let nao_cart = rint_contracted_shell_basis_count(&ao_shells);
    let dm_cart = transform_spin_densities_to_cartesian(
        geom,
        basis4elem,
        dm_spin,
        nao_cart,
        "lib_vee_uhf_r2_jk_matrices_exact",
    );
    let p_total = sum_density_matrices(&dm_cart);
    let eri4 = int4c_r2_full_from_contracted_shell_blocks(&ao_shells);
    let (j_total, _) = build_jk_from_full_4c_tensor(&eri4, &p_total, nao_cart);
    let (_, k_alpha) = build_jk_from_full_4c_tensor(&eri4, &dm_cart[0], nao_cart);
    let (_, k_beta) = build_jk_from_full_4c_tensor(&eri4, &dm_cart[1], nao_cart);
    uhf_jk_matrices_to_current_ao(
        geom,
        basis4elem,
        target_nao,
        &j_total,
        &[k_alpha, k_beta],
        "lib_vee_uhf_r2_jk_matrices_exact",
    )
}

pub fn lib_vee_uhf_r2_jk_matrices_with_auxbasis_advanced(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    auxbasis4elem: &[Basis4Elem],
    dm_spin: &[MatrixFull<f64>],
    coeff_spin: Option<&[MatrixFull<f64>; 2]>,
    occupation: Option<&[Vec<f64>; 2]>,
) -> UhfVeeJkMatrices {
    let exact = || lib_vee_uhf_r2_jk_matrices_exact(geom, basis4elem, dm_spin);
    assert_eq!(
        dm_spin.len(),
        2,
        "lib_vee_uhf_r2_jk_matrices_with_auxbasis_advanced expects alpha and beta density matrices"
    );
    let target_nao = dm_spin[0].size[0];
    let ao_shells = load_molecule_rint_shells_from_raw(geom, basis4elem)
        .expect("failed to build rint shells from GeomCell/BasCell");
    let bfs = crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&ao_shells)
        .expect("failed to expand rint shells to BasisFunction list");
    let dm_cart = transform_spin_densities_to_cartesian(
        geom,
        basis4elem,
        dm_spin,
        bfs.len(),
        "lib_vee_uhf_r2_jk_matrices_advanced",
    );
    let aux_shells = match load_aux_rint_shells_from_raw(geom, auxbasis4elem) {
        Ok(aux_shells) if !aux_shells.is_empty() => aux_shells,
        _ => return exact(),
    };
    let use_semidirect = r2_use_semidirect(
        rint_shell_basis_count(&ao_shells),
        rint_shell_basis_count(&aux_shells),
    );
    let coeff_cart = coeff_spin.map(|coeff_spin| {
        transform_spin_coefficients_to_cartesian(
            geom,
            basis4elem,
            coeff_spin,
            bfs.len(),
            "lib_vee_uhf_r2_jk_matrices_advanced",
        )
    });
    if use_semidirect {
        if let Some((j_spin_upper, k_spin_upper)) =
            r2_jk_direct_from_shell_blocks_with_auxbas_threshold(
                &ao_shells,
                &aux_shells,
                &dm_cart,
                coeff_cart.as_ref().map(|coeff| &coeff[..]),
                occupation.map(|occupation| &occupation[..]),
                AUXBAS_THRESHOLD,
            )
        {
            let Some(j_spin_full) = upper_mats_to_full(
                &j_spin_upper,
                "lib_vee_uhf_r2_jk_matrices_with_auxbasis_advanced_direct[J]",
            ) else {
                return exact();
            };
            let Some(k_spin_full) = upper_mats_to_full(
                &k_spin_upper,
                "lib_vee_uhf_r2_jk_matrices_with_auxbasis_advanced_direct[K]",
            ) else {
                return exact();
            };
            let j_total = sum_density_matrices(&j_spin_full);
            return uhf_jk_matrices_to_current_ao(
                geom,
                basis4elem,
                target_nao,
                &j_total,
                &k_spin_full,
                "lib_vee_uhf_r2_jk_matrices_with_auxbasis_advanced_direct",
            );
        }
    }
    let ri3fn = prepare_rimatr_for_r2_shell_blocks_sync(&ao_shells, &aux_shells);
    if ri3fn.is_none() {
        return exact();
    }
    let j_spin_upper = vj_upper_with_rimatr_r2_sync(&ri3fn, &dm_cart, 2, 1.0);
    let k_spin_upper =
        if let (Some(coeff_cart), Some(occupation)) = (coeff_cart.as_ref(), occupation) {
            let num_elec_alpha = occupation[0].iter().sum::<f64>();
            let num_elec_beta = occupation[1].iter().sum::<f64>();
            let num_elec = [
                num_elec_alpha + num_elec_beta,
                num_elec_alpha,
                num_elec_beta,
            ];
            vk_upper_with_rimatr_sync_v03(&ri3fn, coeff_cart, &num_elec, occupation, 2, 1.0)
        } else {
            vk_upper_with_rimatr_r2_sync(&ri3fn, &dm_cart, 2, 1.0)
        };
    let Some(j_spin_full) = upper_mats_to_full(
        &j_spin_upper,
        "lib_vee_uhf_r2_jk_matrices_with_auxbasis_advanced[J]",
    ) else {
        return exact();
    };
    let Some(k_spin_full) = upper_mats_to_full(
        &k_spin_upper,
        "lib_vee_uhf_r2_jk_matrices_with_auxbasis_advanced[K]",
    ) else {
        return exact();
    };
    let j_total = sum_density_matrices(&j_spin_full);
    uhf_jk_matrices_to_current_ao(
        geom,
        basis4elem,
        target_nao,
        &j_total,
        &k_spin_full,
        "lib_vee_uhf_r2_jk_matrices_with_auxbasis_advanced",
    )
}

pub fn lib_vee_uhf_r2_jk_matrices_advanced(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    dm_spin: &[MatrixFull<f64>],
    coeff_spin: Option<&[MatrixFull<f64>; 2]>,
    occupation: Option<&[Vec<f64>; 2]>,
) -> UhfVeeJkMatrices {
    let auxbasis4elem = build_default_r2_etb_auxbasis(geom, basis4elem)
        .expect("failed to build default ETB(beta=1.7) auxiliary basis for r2 RI path");
    lib_vee_uhf_r2_jk_matrices_with_auxbasis_advanced(
        geom,
        basis4elem,
        &auxbasis4elem,
        dm_spin,
        coeff_spin,
        occupation,
    )
}

/// Compute UHF R2 observables and spin-resolved variational J/K matrices from
/// one RI construction.
pub fn lib_vee_uhf_r2_intermediate(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    dm_spin: &[MatrixFull<f64>],
) -> UhfVeeJkIntermediate {
    let jk = lib_vee_uhf_r2_jk_matrices_advanced(geom, basis4elem, dm_spin, None, None);
    let (ej, ek, total) = ej_ek_from_p_jk_uhf(dm_spin, &jk.j_total, &jk.k_spin);
    UhfVeeJkIntermediate {
        observables: RhfVeeObservables { ej, ek, total },
        jk,
    }
}
fn transform_spin_coefficients_to_cartesian(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    c: &[MatrixFull<f64>; 2],
    nao_cart: usize,
    caller: &str,
) -> [MatrixFull<f64>; 2] {
    [
        transform_mo_coeff_to_cartesian_shell_shared(
            geom,
            basis4elem,
            &c[0],
            nao_cart,
            &format!("{caller}[alpha]"),
            false,
        ),
        transform_mo_coeff_to_cartesian_shell_shared(
            geom,
            basis4elem,
            &c[1],
            nao_cart,
            &format!("{caller}[beta]"),
            false,
        ),
    ]
}

/// First-term approximation over occupied electron pairs in RHF closed shell.
///
/// For `i == j`, there is one opposite-spin pair `(alpha_i, beta_i)`:
///   term_ii = J2_ii - J_ii^2
///
/// For `i < j`, follow the current folded decomposition in terms of one
/// same-spin channel plus the doubly counted opposite-spin channel:
///   <W_ij^2>_ret = (J2_ij - K2_ij) + 2 J2_ij = 3 J2_ij - K2_ij
///   <W_ij>^2_diag = (J_ij - K_ij)^2 + (2 J_ij)^2
///   term_ij = (3 J2_ij - K2_ij) - (J_ij - K_ij)^2 - (2 J_ij)^2
fn lib_vee_rhf_occpair_r_decompose_impl(
    bfs: &[BasisFunction],
    c: &[MatrixFull<f64>; 2],
    spin: usize,
    occ_list: &[usize],
) -> (Vec<(usize, f64)>, Vec<((usize, usize), f64)>, f64) {
    let nao_c = c[spin].size[0];
    let nmo_c = c[spin].size[1];
    assert_eq!(
        bfs.len(),
        nao_c,
        "lib_vee_rhf_occpair_r: MO/AO size mismatch, C has nao={}, bfs has {}",
        nao_c,
        bfs.len()
    );
    assert!(
        !occ_list.is_empty(),
        "lib_vee_rhf_occpair_r: occ_list is empty"
    );
    for &orb in occ_list {
        assert!(
            orb < nmo_c,
            "lib_vee_rhf_occpair_r: occ orbital index out of range, orb={}, nmo={}",
            orb,
            nmo_c
        );
    }
    let mut same_terms: Vec<(usize, f64)> = Vec::with_capacity(occ_list.len());
    let mut pair_terms: Vec<((usize, usize), f64)> = Vec::new();
    let mut s1_occ_raw = 0.0_f64;
    for (ii, &i_orb) in occ_list.iter().enumerate() {
        let j_ii = eri_mo_4c_r(bfs, c, spin, i_orb, i_orb, i_orb, i_orb);
        let j2_ii = eri_mo_4c_r2(bfs, c, spin, i_orb, i_orb, i_orb, i_orb);
        let term_same_orb = j2_ii - j_ii * j_ii;
        same_terms.push((i_orb, term_same_orb));
        if term_same_orb.is_finite() {
            s1_occ_raw += term_same_orb;
        }
        for &j_orb in &occ_list[(ii + 1)..] {
            let j_r = eri_mo_4c_r(bfs, c, spin, i_orb, i_orb, j_orb, j_orb);
            let k_r = eri_mo_4c_r(bfs, c, spin, i_orb, j_orb, j_orb, i_orb);
            let j_r2 = eri_mo_4c_r2(bfs, c, spin, i_orb, i_orb, j_orb, j_orb);
            let k_r2 = eri_mo_4c_r2(bfs, c, spin, i_orb, j_orb, j_orb, i_orb);
            let tmp = (j_r2 - k_r2) - (j_r - k_r) * (j_r - k_r) + j_r2 - j_r * j_r;
            let term = 2.0 * tmp;
            pair_terms.push(((i_orb, j_orb), term));
            if term.is_finite() {
                s1_occ_raw += term;
            }
        }
    }
    (same_terms, pair_terms, s1_occ_raw)
}
pub fn lib_vee_rhf_occpair_r_decompose(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    c: &[MatrixFull<f64>; 2],
    spin: usize,
    occ_list: &[usize],
) -> (Vec<(usize, f64)>, Vec<((usize, usize), f64)>, f64) {
    assert!(spin < 2);
    let bfs = load_molecule_shell_shared_from_raw(geom, basis4elem)
        .expect("failed to build shell_shared bfs from GeomCell/BasCell");
    let c_cart = transform_spin_coefficients_to_cartesian(
        geom,
        basis4elem,
        c,
        bfs.len(),
        "lib_vee_rhf_occpair_r_decompose",
    );
    lib_vee_rhf_occpair_r_decompose_impl(&bfs, &c_cart, spin, occ_list)
}
pub fn lib_vee_rhf_occpair_r(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    c: &[MatrixFull<f64>; 2],
    spin: usize,
    occ_list: &[usize],
) -> f64 {
    let (_, _, s1_occ_raw) = lib_vee_rhf_occpair_r_decompose(geom, basis4elem, c, spin, occ_list);
    println!(
        "<occ-pair first term (RHF closed-shell, spatial-orbital folded)> = {:.16e}",
        s1_occ_raw
    );
    s1_occ_raw
}

#[cfg(test)]
mod tests;
