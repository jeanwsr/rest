use crate::basis_io::etb::{etb_gen_for_atom_list, get_etb_elem};
use crate::basis_io::Basis4Elem;
use crate::constants::{AUXBAS_THRESHOLD, SQRT_THRESHOLD};
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
use statrs::function::erf::erf;
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
};
pub use basis::{make_contracted_coeffs_for_shell, BasisFunction, RintShell};
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

    let left_rows = a.ao_len * b.ao_len;
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
                                }
                                debug_assert_eq!(n + usize::from(d.shell.ang_type == 1), 3);
                                let value = p_triplet_component_value(axes, ids, &first, scalars);
                                let term = primitive_scale * value;
                                if term.is_finite() {
                                    block_data[row_offset + i] += term;
                                }
                            }
                        }
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
    build_4c_block_entries_into(a, b, c, d, entries);
    int4c_r_shell_block_batched_into_data_with_pairs_workspace_entries(
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
    let left_rows = a.ao_len * b.ao_len;
    let right_cols = c.ao_len * d.ao_len;
    let shape = [left_rows, right_cols];
    block_data.resize(left_rows * right_cols, 0.0_f64);
    block_data.fill(0.0);
    if a.shell.ang_type == 0
        && b.shell.ang_type == 0
        && c.shell.ang_type == 0
        && d.shell.ang_type == 0
    {
        int4c_r_ssss_block_into_data_with_pairs(ab_pairs, cd_pairs, block_data);
        return shape;
    }
    if a.shell.ang_type + b.shell.ang_type + c.shell.ang_type + d.shell.ang_type == 1 {
        int4c_r_total_ang1_block_into_data_with_pairs(a, b, c, d, ab_pairs, cd_pairs, block_data);
        return shape;
    }
    if a.shell.ang_type + b.shell.ang_type + c.shell.ang_type + d.shell.ang_type == 3
        && int4c_r_three_p_one_s_block_into_data_with_pairs(
            a, b, c, d, ab_pairs, cd_pairs, block_data,
        )
    {
        return shape;
    }

    for ab_pair in ab_pairs {
        for cd_pair in cd_pairs {
            add_primitive_4c_r_shell_block(
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
    let left_rows = a.ao_len * b.ao_len;
    let right_cols = c.ao_len * d.ao_len;
    let mut block = MatrixFull::new([left_rows, right_cols], 0.0_f64);
    for l in 0..d.ao_len {
        for k in 0..c.ao_len {
            let col = l * c.ao_len + k;
            let col_offset = col * left_rows;
            for j in 0..b.ao_len {
                let row_offset = col_offset + j * a.ao_len;
                for i in 0..a.ao_len {
                    block.data[row_offset + i] = eri_rint_shell_4c_r2(a, i, b, j, c, k, d, l);
                }
            }
        }
    }
    block
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
    num_auxbas: usize,
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
                let metric_factor_columns =
                    metric_factor.to_matrixfullslice_columns(kept_start..kept_start + kept_len);
                let metric_factor_view = MatrixFullSlice {
                    size: &metric_factor_columns.size,
                    indicing: &metric_factor_columns.indicing,
                    data: metric_factor_columns.data,
                };
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
                        let mut local_raw = MatrixFull::new([pair_len, num_auxbas], 0.0_f64);

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
                                for aux_loc in 0..aux_shell.ao_len {
                                    let aux = aux_shell.ao_start + aux_loc;
                                    for (block_row, local_pair) in pair_index_map.iter().copied() {
                                        local_raw[(local_pair, aux)] =
                                            block_buffer[(block_row, aux_loc)];
                                    }
                                }
                            }
                        }

                        let mut local_fitted = MatrixFull::new([pair_len, kept_len], 0.0_f64);
                        local_fitted.to_matrixfullslicemut().lapack_dgemm(
                            &local_raw.to_matrixfullslice(),
                            &metric_factor_view,
                            'N',
                            'N',
                            1.0,
                            0.0,
                        );
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
            if matches!(value.as_str(), "semi-direct" | "semidirect" | "direct") {
                println!("RI-r2 semi-direct requested by REST_R2_DIRECT={value}");
                return true;
            }
            if matches!(value.as_str(), "incore") {
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

/// 根据常驻内存、并行 shell worker 临时块、H/K scratch 和 batch cache 估算 kept_batch。
fn r2_direct_kept_block_size(
    nkept: usize,
    num_basis: usize,
    num_auxbas: usize,
    num_baspar: usize,
    occ_coeffs: &[MatrixFull<f64>],
    max_pair_len: usize,
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
    let worker_fitted_values_per_kept =
        max_pair_len.saturating_mul(workers.saturating_add(inflight_blocks).saturating_add(1));
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
        .saturating_mul(num_auxbas)
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
        "RI-r2 semi-direct K kept_batch auto: kept_batch={} nkept={} limit={:.2} MB factor={:.2} kept_fraction={:.2} fixed={:.2} MB fixed_metric_factor={:.2} MB worker_raw={:.2} MB output={:.2} MB kept_budget={:.2} MB live_per_kept={:.2} KB H_per_kept={} K_scratch_per_kept={} batch_cache_per_kept={} worker_fitted_per_kept={} max_pair_len={} workers={} inflight_blocks={} nocc_sum={} nocc_max={} cap={}",
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
        .saturating_mul(num_auxbas)
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
        "RI-r2 semi-direct memory plan (kept-batch-first): incore_peak={:.2} MB fixed={:.2} MB H_batch={:.2} MB K_scratch_peak={:.2} MB limit={:.2} MB factor={:.2} kept_fraction={:.2}",
        bytes_to_mb(incore_peak_bytes),
        bytes_to_mb(fixed_bytes),
        bytes_to_mb(h_batch_bytes),
        bytes_to_mb(k_scratch_peak_bytes),
        limit_mb,
        factor,
        r2_direct_kept_memory_fraction(),
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
    let plan =
        r2_make_semidirect_cache_plan(num_basis, num_auxbas, nkept, &occ_coeffs, max_pair_len);
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
    vk_upper_with_rimatr_use_dm_only_sync_v02(ri3fn, dm, spin_channel, scaling_factor)
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
pub fn lib_vee_rhf_r2_observables_exact(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    p_rhf: &MatrixFull<f64>,
) -> RhfVeeObservables {
    let (bfs, p_cart) = load_cartesian_rhf_basis_and_density_shell_shared(
        geom,
        basis4elem,
        p_rhf,
        "lib_vee_rhf_r2_exact",
    );
    let (j2, k2) = build_jk_from_p_kernel(&bfs, &p_cart, eri_ao_4c_r2);
    let (ej2, ek2, e2) = ej_ek_from_p_jk_rhf(&p_cart, &j2, &k2);
    RhfVeeObservables {
        ej: ej2,
        ek: ek2,
        total: e2,
    }
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
    let Some(coeff_spin) = coeff_spin else {
        return lib_vee_rhf_r2_observables_with_auxbasis(geom, basis4elem, auxbasis4elem, p_rhf);
    };
    let Some(occupation) = occupation else {
        return lib_vee_rhf_r2_observables_with_auxbasis(geom, basis4elem, auxbasis4elem, p_rhf);
    };
    let exact = || lib_vee_rhf_r2_observables_exact(geom, basis4elem, p_rhf);
    let (ao_shells, _bfs, p_cart) = load_cartesian_rhf_rint_shells_basis_and_density_shell_shared(
        geom,
        basis4elem,
        p_rhf,
        "lib_vee_rhf_r2_advanced",
    );
    let aux_shells = match load_aux_rint_shells_from_raw(geom, auxbasis4elem) {
        Ok(aux_shells) if !aux_shells.is_empty() => aux_shells,
        _ => return exact(),
    };
    let use_semidirect = r2_use_semidirect(
        rint_shell_basis_count(&ao_shells),
        rint_shell_basis_count(&aux_shells),
    );
    let num_basis = rint_shell_basis_count(&ao_shells);
    let coeff_cart_alpha = transform_mo_coeff_to_cartesian_shell_shared(
        geom,
        basis4elem,
        &coeff_spin[0],
        num_basis,
        "lib_vee_rhf_r2_advanced[alpha]",
        false,
    );
    let coeff_cart = [coeff_cart_alpha.clone(), MatrixFull::empty()];
    let occ = [occupation[0].clone()];
    let dm = vec![p_cart.clone()];
    if use_semidirect {
        if let Some((j_upper, k_upper)) = r2_jk_direct_from_shell_blocks_with_auxbas_threshold(
            &ao_shells,
            &aux_shells,
            &dm,
            Some(&coeff_cart[..1]),
            Some(&occ),
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
        };
    }
    let ri3fn = prepare_rimatr_for_r2_shell_blocks_sync(&ao_shells, &aux_shells);
    if ri3fn.is_none() {
        return exact();
    }
    let j_upper = vj_upper_with_rimatr_r2_sync(&ri3fn, &dm, 1, 1.0);
    let occ_cart = [occupation[0].clone(), Vec::new()];
    let num_elec_alpha = occupation[0].iter().sum::<f64>();
    let num_elec = [num_elec_alpha, num_elec_alpha, 0.0_f64];
    let k_upper = vk_upper_with_rimatr_sync_v03(&ri3fn, &coeff_cart, &num_elec, &occ_cart, 1, 1.0);
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
pub fn lib_vee_uhf_r2_observables_exact(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    dm_spin: &[MatrixFull<f64>],
) -> RhfVeeObservables {
    assert_eq!(
        dm_spin.len(),
        2,
        "lib_vee_uhf_r2_observables_exact expects alpha and beta density matrices"
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
        "lib_vee_uhf_r2_exact",
    );
    let p_total = sum_density_matrices(&dm_cart);
    let (j_total, _) = build_jk_from_p_kernel(&bfs, &p_total, eri_ao_4c_r2);
    let (_, k_alpha) = build_jk_from_p_kernel(&bfs, &dm_cart[0], eri_ao_4c_r2);
    let (_, k_beta) = build_jk_from_p_kernel(&bfs, &dm_cart[1], eri_ao_4c_r2);
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
    let Some(coeff_spin) = coeff_spin else {
        return lib_vee_uhf_r2_observables_with_auxbasis(geom, basis4elem, auxbasis4elem, dm_spin);
    };
    let Some(occupation) = occupation else {
        return lib_vee_uhf_r2_observables_with_auxbasis(geom, basis4elem, auxbasis4elem, dm_spin);
    };
    let exact = || lib_vee_uhf_r2_observables_exact(geom, basis4elem, dm_spin);
    assert_eq!(
        dm_spin.len(),
        2,
        "lib_vee_uhf_r2_observables_with_auxbasis_advanced expects alpha and beta density matrices"
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
        "lib_vee_uhf_r2_advanced",
    );
    let aux_shells = match load_aux_rint_shells_from_raw(geom, auxbasis4elem) {
        Ok(aux_shells) if !aux_shells.is_empty() => aux_shells,
        _ => return exact(),
    };
    let use_semidirect = r2_use_semidirect(
        rint_shell_basis_count(&ao_shells),
        rint_shell_basis_count(&aux_shells),
    );
    let coeff_cart = transform_rhf_coefficients_to_cartesian(
        geom,
        basis4elem,
        coeff_spin,
        bfs.len(),
        "lib_vee_uhf_r2_advanced",
    );
    if use_semidirect {
        if let Some((j_spin_upper, k_spin_upper)) =
            r2_jk_direct_from_shell_blocks_with_auxbas_threshold(
                &ao_shells,
                &aux_shells,
                &dm_cart,
                Some(&coeff_cart),
                Some(&occupation[..]),
                AUXBAS_THRESHOLD,
            )
        {
            let Some(j_spin_full) = upper_mats_to_full(
                &j_spin_upper,
                "lib_vee_uhf_r2_observables_with_auxbasis_advanced_direct[J]",
            ) else {
                return exact();
            };
            let Some(k_spin_full) = upper_mats_to_full(
                &k_spin_upper,
                "lib_vee_uhf_r2_observables_with_auxbasis_advanced_direct[K]",
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
    let num_elec_alpha = occupation[0].iter().sum::<f64>();
    let num_elec_beta = occupation[1].iter().sum::<f64>();
    let num_elec = [
        num_elec_alpha + num_elec_beta,
        num_elec_alpha,
        num_elec_beta,
    ];
    let k_spin_upper =
        vk_upper_with_rimatr_sync_v03(&ri3fn, &coeff_cart, &num_elec, occupation, 2, 1.0);
    let Some(j_spin_full) = upper_mats_to_full(
        &j_spin_upper,
        "lib_vee_uhf_r2_observables_with_auxbasis_advanced[J]",
    ) else {
        return exact();
    };
    let Some(k_spin_full) = upper_mats_to_full(
        &k_spin_upper,
        "lib_vee_uhf_r2_observables_with_auxbasis_advanced[K]",
    ) else {
        return exact();
    };
    let j_total = sum_density_matrices(&j_spin_full);
    let (ej, ek, total) = ej_ek_from_p_jk_uhf(&dm_cart, &j_total, &k_spin_full);
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
fn transform_rhf_coefficients_to_cartesian(
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
    let c_cart = transform_rhf_coefficients_to_cartesian(
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

fn occ_closure_occupied_coeff_matrix(
    coeff: &MatrixFull<f64>,
    occupation: &[f64],
) -> MatrixFull<f64> {
    let nao = coeff.size[0];
    let occ_idx = occupation
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| if *value > 1.0e-8 { Some(idx) } else { None })
        .collect::<Vec<_>>();
    let mut c_occ = MatrixFull::new([nao, occ_idx.len()], 0.0_f64);
    for (i_occ, &i_mo) in occ_idx.iter().enumerate() {
        for mu in 0..nao {
            c_occ[(mu, i_occ)] = coeff[(mu, i_mo)];
        }
    }
    c_occ
}

fn occ_closure_fragment_rimatr_operator_covariance(
    rimatr: &MatrixFull<f64>,
    nao: usize,
    aoslice: &[[usize; 4]],
    atoms: &[usize],
    c_occ: &MatrixFull<f64>,
    s_inv: &MatrixFull<f64>,
) -> MatrixFull<f64> {
    let nmode = rimatr.size[1];
    let nocc = c_occ.size[1];
    let mut frag_aos = Vec::new();
    for &atom in atoms {
        assert!(
            atom < aoslice.len(),
            "X_AB atom index {atom} is out of range for {} atoms",
            aoslice.len()
        );
        let [_shl0, _shl1, p0, p1] = aoslice[atom];
        frag_aos.extend(p0..p1);
    }

    let mut y_flat = MatrixFull::new([nao * nocc, nmode], 0.0_f64);
    for &mu in &frag_aos {
        for &nu in &frag_aos {
            let pair = baspair_index(mu.max(nu), mu.min(nu));
            for q in 0..nmode {
                let value = rimatr[(pair, q)];
                if value.abs() < 1.0e-18 {
                    continue;
                }
                for i in 0..nocc {
                    y_flat[(mu + nao * i, q)] += value * c_occ[(nu, i)];
                }
            }
        }
    }

    let mut z_flat = MatrixFull::new([nao * nocc, nmode], 0.0_f64);
    for q in 0..nmode {
        for i in 0..nocc {
            for mu in 0..nao {
                let mut value = 0.0_f64;
                for nu in 0..nao {
                    value += s_inv[(mu, nu)] * y_flat[(nu + nao * i, q)];
                }
                z_flat[(mu + nao * i, q)] = value;
            }
        }
    }

    let mut occ_flat = MatrixFull::new([nocc * nocc, nmode], 0.0_f64);
    for q in 0..nmode {
        for j in 0..nocc {
            for i in 0..nocc {
                let mut value = 0.0_f64;
                for mu in 0..nao {
                    value += c_occ[(mu, i)] * y_flat[(mu + nao * j, q)];
                }
                occ_flat[(i + nocc * j, q)] = value;
            }
        }
    }

    let mut closure = MatrixFull::new([nmode, nmode], 0.0_f64);
    closure.to_matrixfullslicemut().lapack_dgemm(
        &y_flat.to_matrixfullslice(),
        &z_flat.to_matrixfullslice(),
        'T',
        'N',
        1.0,
        0.0,
    );
    let mut occupied = MatrixFull::new([nmode, nmode], 0.0_f64);
    occupied.to_matrixfullslicemut().lapack_dgemm(
        &occ_flat.to_matrixfullslice(),
        &occ_flat.to_matrixfullslice(),
        'T',
        'N',
        1.0,
        0.0,
    );

    let mut cov = MatrixFull::new([nmode, nmode], 0.0_f64);
    for r in 0..nmode {
        for q in 0..nmode {
            cov[(q, r)] = 2.0 * (closure[(q, r)] - occupied[(q, r)]);
        }
    }
    cov
}

pub fn lib_vee_rhf_occ_closure_connected_ri_coulomb_x(
    scf_data: &crate::scf_io::SCF,
    atoms_a: &[usize],
    atoms_b: &[usize],
) -> f64 {
    assert!(
        !atoms_a.is_empty() && !atoms_b.is_empty(),
        "X_AB needs non-empty atom lists for fragments A and B"
    );
    let (ao_bfs, _p_cart) = load_cartesian_rhf_basis_and_density_shell_shared(
        &scf_data.mol.geom,
        &scf_data.mol.basis4elem,
        &scf_data.density_matrix[0],
        "lib_vee_rhf_occ_closure_connected_ri_coulomb_x",
    );
    let auxbasis4elem = build_default_r2_etb_auxbasis(&scf_data.mol.geom, &scf_data.mol.basis4elem)
        .expect("default ETB auxiliary basis should be available for X_AB");
    let aux_bfs = load_aux_molecule_shell_shared_from_raw(&scf_data.mol.geom, &auxbasis4elem)
        .expect("failed to build auxiliary basis for X_AB");
    let (rimatr, _basbas2baspar, _baspar2basbas) =
        prepare_rimatr_for_r_sync(&ao_bfs, &aux_bfs).expect("failed to build RI-r matrix for X_AB");

    let coeff_cart = transform_mo_coeff_to_cartesian_shell_shared(
        &scf_data.mol.geom,
        &scf_data.mol.basis4elem,
        &scf_data.eigenvectors[0],
        ao_bfs.len(),
        "lib_vee_rhf_occ_closure_connected_ri_coulomb_x[coeff]",
        false,
    );
    let c_occ = occ_closure_occupied_coeff_matrix(&coeff_cart, &scf_data.occupation[0]);
    let mut ovlp = scf_data
        .ovlp
        .to_matrixfull()
        .expect("overlap MatrixUpper -> full");
    let s_inv = ovlp
        .lapack_inverse()
        .expect("overlap inverse should exist for X_AB");
    let aoslice = scf_data.mol.aoslice_by_atom();
    let cov_a = occ_closure_fragment_rimatr_operator_covariance(
        &rimatr,
        ao_bfs.len(),
        &aoslice,
        atoms_a,
        &c_occ,
        &s_inv,
    );
    let cov_b = occ_closure_fragment_rimatr_operator_covariance(
        &rimatr,
        ao_bfs.len(),
        &aoslice,
        atoms_b,
        &c_occ,
        &s_inv,
    );

    cov_a
        .data
        .iter()
        .zip(cov_b.data.iter())
        .fold(0.0_f64, |acc, (a, b)| acc + a * b)
}

#[cfg(test)]
mod kernel_worst_quartet_tests {
    use super::*;
    use crate::basis_io::{BasCell, Basis4Elem};
    use crate::lib_rint::basis::Shell;
    use crate::scf_io::{
        scf_without_build, vj_upper_with_rimatr_sync, vk_upper_with_rimatr_use_dm_only_sync_v02,
        SCF,
    };
    use crate::Molecule;
    use rest_libcint::prelude::rest_libcint_wrapper::int1e_r;
    use std::fs;
    use std::process::Command;
    use std::time::Instant;

    #[test]
    fn r2_direct_requested_workers_defaults_to_configured_rayon_threads() {
        std::env::remove_var("REST_R2_DIRECT_WORKERS");
        std::env::remove_var("RAYON_NUM_THREADS");

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .expect("failed to build test rayon pool");

        pool.install(|| {
            assert_eq!(r2_direct_requested_workers(), rayon::current_num_threads());
        });
    }

    fn write_temp_ctrl_h2(basis_dir: &str) -> String {
        let pid = std::process::id();
        let path = format!("/tmp/lib_rint_kernel_check_{basis_dir}_{pid}.toml")
            .replace('(', "")
            .replace(')', "")
            .replace('/', "_")
            .replace(' ', "_");
        let text = format!(
            "[ctrl]\nprint_level = 0\nxc = \"hf\"\nbasis_path = \"/home/cfh/rest_workspace/rest/basis-set-pool/{basis_dir}\"\nbasis_type = \"Cartesian\"\nuse_auxbas = false\neven_tempered_basis = false\ncharge = 0.0\nspin = 1.0\nspin_polarization = false\nnum_threads = 1\nrun_lib_rint = false\n\n[geom]\nname = \"H2\"\nunit = \"angstrom\"\nposition = [\n    \"H   0.0000000000   0.0000000000   0.0000000000\",\n    \"H   0.0000000000   0.0000000000   1.4000000000\",\n]\n"
        );
        fs::write(&path, text).unwrap();
        path
    }
    fn write_temp_ctrl_h2o(basis_dir: &str) -> String {
        let pid = std::process::id();
        let path = format!("/tmp/lib_rint_kernel_h2o_check_{basis_dir}_{pid}.toml")
            .replace('(', "")
            .replace(')', "")
            .replace('/', "_")
            .replace(' ', "_");
        let text = format!(
            "[ctrl]\nprint_level = 0\nxc = \"hf\"\nbasis_path = \"/home/cfh/rest_workspace/rest/basis-set-pool/{basis_dir}\"\nbasis_type = \"Cartesian\"\nuse_auxbas = false\neven_tempered_basis = false\ncharge = 0.0\nspin = 1.0\nspin_polarization = false\nnum_threads = 1\nrun_lib_rint = false\n\n[geom]\nname = \"H2O\"\nunit = \"angstrom\"\nposition = [\n    \"O   0.0000000000   0.0000000000   0.0000000000\",\n    \"H   0.7586020000   0.0000000000   0.5042840000\",\n    \"H  -0.7586020000   0.0000000000   0.5042840000\",\n]\n"
        );
        fs::write(&path, text).unwrap();
        path
    }
    fn write_temp_high_l_he_basis(l: u32) -> (String, String) {
        let pid = std::process::id();
        let basis_dir = format!("/tmp/lib_rint_high_l_basis_l{l}_{pid}");
        fs::create_dir_all(&basis_dir).unwrap();
        let basis_path = format!("{basis_dir}/He.json");
        let basis_json = format!(
            r#"{{
    "electron_shells": [
        {{
            "function_type": "gto",
            "region": "",
            "angular_momentum": [{l}],
            "exponents": ["0.75"],
            "coefficients": [["1.0"]]
        }}
    ],
    "references": null,
    "ecp_potentials": null,
    "ecp_electrons": null
}}"#
        );
        fs::write(&basis_path, basis_json).unwrap();

        let ctrl_path = format!("/tmp/lib_rint_high_l_ctrl_l{l}_{pid}.toml");
        let ctrl_text = format!(
            "[ctrl]\nprint_level = 0\nxc = \"hf\"\nbasis_path = \"{basis_dir}\"\nbasis_type = \"Cartesian\"\nuse_auxbas = false\neven_tempered_basis = false\ncharge = 0.0\nspin = 1.0\nspin_polarization = false\nnum_threads = 1\nrun_lib_rint = false\n\n[geom]\nname = \"He_high_l\"\nunit = \"angstrom\"\nposition = [\n    \"He   0.0000000000   0.0000000000   0.0000000000\",\n]\n"
        );
        fs::write(&ctrl_path, ctrl_text).unwrap();
        (ctrl_path, basis_dir)
    }

    fn assert_high_l_4c_shell_block_matches_scalar_and_libcint(l: u32) {
        let (ctrl_path, basis_dir) = write_temp_high_l_he_basis(l);
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();

        let ao_shells =
            crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
                .expect("failed to build AO rint shells");
        let ao_bfs = crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&ao_shells)
            .expect("failed to expand AO rint shells");
        assert_eq!(ao_shells.len(), 1);
        let shell = &ao_shells[0];
        assert_eq!(shell.shell.ang_type, l);

        let t0 = Instant::now();
        let block = int4c_r_shell_block(shell, shell, shell, shell);
        let librint_s = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let libcint_full = mol.int_ijkl_erifull();
        let libcint_s = t0.elapsed().as_secs_f64();
        let mut max_abs_scalar = 0.0_f64;
        let mut max_abs_libcint = 0.0_f64;
        let mut max_value = 0.0_f64;
        let sample_locs = [
            0_usize,
            shell.ao_len / 3,
            shell.ao_len / 2,
            shell.ao_len.saturating_sub(1),
        ];
        for l_loc in 0..shell.ao_len {
            for k_loc in 0..shell.ao_len {
                let col = l_loc * shell.ao_len + k_loc;
                for j_loc in 0..shell.ao_len {
                    for i_loc in 0..shell.ao_len {
                        let row = j_loc * shell.ao_len + i_loc;
                        let got = block[(row, col)];
                        let i_ao = shell.ao_start + i_loc;
                        let j_ao = shell.ao_start + j_loc;
                        let k_ao = shell.ao_start + k_loc;
                        let l_ao = shell.ao_start + l_loc;
                        let libcint = *libcint_full.get(&[i_ao, j_ao, k_ao, l_ao]).unwrap();
                        max_abs_libcint = max_abs_libcint.max((got - libcint).abs());
                        max_value = max_value.max(got.abs()).max(libcint.abs());
                        if sample_locs.contains(&i_loc)
                            && sample_locs.contains(&j_loc)
                            && sample_locs.contains(&k_loc)
                            && sample_locs.contains(&l_loc)
                        {
                            let scalar = eri_ao_4c_r(&ao_bfs, i_ao, j_ao, k_ao, l_ao);
                            max_abs_scalar = max_abs_scalar.max((got - scalar).abs());
                        }
                    }
                }
            }
        }
        let tol = 5.0e-8 * max_value.max(1.0);
        let nroots = ((4 * l) / 2 + 1) as usize;
        println!(
            "TMP_EXACT4C_HIGH_L_ERROR l={l} nroots={nroots} ao_len={} libcint_s={libcint_s:.6} lib_rint_s={librint_s:.6} lib_rint_over_libcint={:.3} max_abs_libcint={max_abs_libcint:.3e} max_abs_scalar={max_abs_scalar:.3e} max_value={max_value:.3e} tol={tol:.3e}",
            shell.ao_len,
            librint_s / libcint_s.max(1.0e-12)
        );
        assert!(
            max_abs_scalar <= tol,
            "high-l shell block vs scalar mismatch l={l}: max_abs={max_abs_scalar:.3e}, tol={tol:.3e}"
        );
        assert!(
            max_abs_libcint <= tol,
            "high-l shell block vs libcint mismatch l={l}: max_abs={max_abs_libcint:.3e}, tol={tol:.3e}"
        );

        let _ = fs::remove_file(ctrl_path);
        let _ = fs::remove_dir_all(basis_dir);
    }

    fn single_component_shell(l: u32, component: [u32; 3]) -> RintShell {
        RintShell {
            atom_idx: 0,
            center: [0.0, 0.0, 0.0],
            shell: Shell {
                ang_type: l,
                exponents: vec![0.75],
                coefficients: vec![vec![1.0]],
            },
            column_idx: 0,
            cart_components: vec![component],
            ao_start: 0,
            ao_len: 1,
            is_aux: false,
        }
    }

    fn single_component_basis_function(component: [u32; 3]) -> BasisFunction {
        BasisFunction {
            atom_idx: 0,
            lx: component[0],
            ly: component[1],
            lz: component[2],
            exponents: vec![0.75],
            coefficients: vec![1.0],
            center: [0.0, 0.0, 0.0],
        }
    }

    fn write_temp_ctrl_water_cluster(basis_dir: &str, nwater: usize) -> String {
        let pid = std::process::id();
        let path = format!("/tmp/lib_rint_kernel_h2o_cluster_{nwater}_{basis_dir}_{pid}.toml")
            .replace('(', "")
            .replace(')', "")
            .replace('/', "_")
            .replace(' ', "_");
        let mut positions = Vec::with_capacity(3 * nwater);
        for i in 0..nwater {
            let shift = 2.8 * i as f64;
            positions.push(format!(
                "    \"O   {shift:.10}   0.0000000000   0.0000000000\""
            ));
            positions.push(format!(
                "    \"H   {:.10}   0.0000000000   0.5042840000\"",
                shift + 0.758602
            ));
            positions.push(format!(
                "    \"H   {:.10}   0.0000000000   0.5042840000\"",
                shift - 0.758602
            ));
        }
        let text = format!(
            "[ctrl]\nprint_level = 0\nxc = \"hf\"\nbasis_path = \"/home/cfh/rest_workspace/rest/basis-set-pool/{basis_dir}\"\nbasis_type = \"Cartesian\"\nuse_auxbas = false\neven_tempered_basis = false\ncharge = 0.0\nspin = 1.0\nspin_polarization = false\nnum_threads = 1\nrun_lib_rint = false\n\n[geom]\nname = \"H2O_cluster_{nwater}\"\nunit = \"angstrom\"\nposition = [\n{}\n]\n",
            positions.join(",\n")
        );
        fs::write(&path, text).unwrap();
        path
    }

    fn write_temp_ctrl_he2(basis_dir: &str, distance_ang: f64, label: &str) -> String {
        let pid = std::process::id();
        let path =
            format!("/tmp/lib_rint_kernel_he2_{label}_{distance_ang:.3}_{basis_dir}_{pid}.toml")
                .replace('(', "")
                .replace(')', "")
                .replace('/', "_")
                .replace(' ', "_");
        let text = format!(
            "[ctrl]\nprint_level = 0\nxc = \"hf\"\nbasis_path = \"/home/cfh/rest_workspace/rest/basis-set-pool/{basis_dir}\"\nauxbas_path = \"/home/cfh/rest_workspace/rest/basis-set-pool/def2-SV(P)-JKFIT\"\nbasis_type = \"Cartesian\"\nuse_auxbas = false\neven_tempered_basis = false\ncharge = 0.0\nspin = 1.0\nspin_polarization = false\nnum_threads = 1\nrun_lib_rint = false\n\n[geom]\nname = \"He2_{distance_ang:.3}\"\nunit = \"angstrom\"\nposition = [\n    \"He   0.0000000000   0.0000000000   0.0000000000\",\n    \"He   0.0000000000   0.0000000000   {distance_ang:.10}\",\n]\n"
        );
        fs::write(&path, text).unwrap();
        path
    }

    fn write_temp_ctrl_diatomic_dimer(
        elem: &str,
        basis_dir: &str,
        distance_ang: f64,
        label: &str,
    ) -> String {
        let pid = std::process::id();
        let path = format!(
            "/tmp/lib_rint_kernel_{elem}2_{label}_{distance_ang:.3}_{basis_dir}_{pid}.toml"
        )
        .replace('(', "")
        .replace(')', "")
        .replace('/', "_")
        .replace(' ', "_");
        let text = format!(
            "[ctrl]\nprint_level = 0\nxc = \"hf\"\nbasis_path = \"/home/cfh/rest_workspace/rest/basis-set-pool/{basis_dir}\"\nauxbas_path = \"/home/cfh/rest_workspace/rest/basis-set-pool/def2-SV(P)-JKFIT\"\nbasis_type = \"Cartesian\"\nuse_auxbas = false\neven_tempered_basis = false\ncharge = 0.0\nspin = 1.0\nspin_polarization = false\nnum_threads = 1\nrun_lib_rint = false\n\n[geom]\nname = \"{elem}2_{distance_ang:.3}\"\nunit = \"angstrom\"\nposition = [\n    \"{elem}   0.0000000000   0.0000000000   0.0000000000\",\n    \"{elem}   0.0000000000   0.0000000000   {distance_ang:.10}\",\n]\n"
        );
        fs::write(&path, text).unwrap();
        path
    }

    fn write_temp_ctrl_methane_dimer(basis_dir: &str, distance_ang: f64, label: &str) -> String {
        let pid = std::process::id();
        let path = format!(
            "/tmp/lib_rint_kernel_ch4_dimer_{label}_{distance_ang:.3}_{basis_dir}_{pid}.toml"
        )
        .replace('(', "")
        .replace(')', "")
        .replace('/', "_")
        .replace(' ', "_");
        let h = 1.09_f64 / 3.0_f64.sqrt();
        let mut positions = Vec::new();
        for z0 in [0.0_f64, distance_ang] {
            positions.push(format!(
                "    \"C   0.0000000000   0.0000000000   {z0:.10}\""
            ));
            for (x, y, z) in [(h, h, h), (h, -h, -h), (-h, h, -h), (-h, -h, h)] {
                positions.push(format!("    \"H   {x:.10}   {y:.10}   {:.10}\"", z0 + z));
            }
        }
        let text = format!(
            "[ctrl]\nprint_level = 0\nxc = \"hf\"\nbasis_path = \"/home/cfh/rest_workspace/rest/basis-set-pool/{basis_dir}\"\nauxbas_path = \"/home/cfh/rest_workspace/rest/basis-set-pool/def2-SV(P)-JKFIT\"\nbasis_type = \"Cartesian\"\nuse_auxbas = false\neven_tempered_basis = false\ncharge = 0.0\nspin = 1.0\nspin_polarization = false\nnum_threads = 1\nrun_lib_rint = false\n\n[geom]\nname = \"CH4_dimer_{distance_ang:.3}\"\nunit = \"angstrom\"\nposition = [\n{}\n]\n",
            positions.join(",\n")
        );
        fs::write(&path, text).unwrap();
        path
    }

    fn restrict_operator_to_fragment(
        op: &MatrixFull<f64>,
        aoslice: &[[usize; 4]],
        atoms: &[usize],
    ) -> MatrixFull<f64> {
        let nao = op.size[0];
        let mut mask = vec![false; nao];
        for &atom in atoms {
            let [_shl0, _shl1, p0, p1] = aoslice[atom];
            for ao in p0..p1 {
                mask[ao] = true;
            }
        }
        let mut frag_op = MatrixFull::new([nao, nao], 0.0_f64);
        for nu in 0..nao {
            if !mask[nu] {
                continue;
            }
            for mu in 0..nao {
                if mask[mu] {
                    frag_op[(mu, nu)] = op[(mu, nu)];
                }
            }
        }
        frag_op
    }

    fn transform_one_body_operator_to_mo(
        op_ao: &MatrixFull<f64>,
        coeff: &MatrixFull<f64>,
    ) -> MatrixFull<f64> {
        let nao = coeff.size[0];
        let nmo = coeff.size[1];
        assert_eq!(op_ao.size, [nao, nao]);
        let mut tmp = MatrixFull::new([nao, nmo], 0.0_f64);
        tmp.to_matrixfullslicemut().lapack_dgemm(
            &op_ao.to_matrixfullslice(),
            &coeff.to_matrixfullslice(),
            'N',
            'N',
            1.0,
            0.0,
        );
        let mut op_mo = MatrixFull::new([nmo, nmo], 0.0_f64);
        op_mo.to_matrixfullslicemut().lapack_dgemm(
            &coeff.to_matrixfullslice(),
            &tmp.to_matrixfullslice(),
            'T',
            'N',
            1.0,
            0.0,
        );
        op_mo
    }

    fn occupied_coeff_matrix(coeff: &MatrixFull<f64>, occupation: &[f64]) -> MatrixFull<f64> {
        let nao = coeff.size[0];
        let occ_idx = occupation
            .iter()
            .enumerate()
            .filter_map(|(idx, value)| if *value > 1.0e-8 { Some(idx) } else { None })
            .collect::<Vec<_>>();
        let mut c_occ = MatrixFull::new([nao, occ_idx.len()], 0.0_f64);
        for (i_occ, &i_mo) in occ_idx.iter().enumerate() {
            for mu in 0..nao {
                c_occ[(mu, i_occ)] = coeff[(mu, i_mo)];
            }
        }
        c_occ
    }

    fn matmul_nn(a: &MatrixFull<f64>, b: &MatrixFull<f64>) -> MatrixFull<f64> {
        assert_eq!(a.size[1], b.size[0]);
        let mut out = MatrixFull::new([a.size[0], b.size[1]], 0.0_f64);
        out.to_matrixfullslicemut().lapack_dgemm(
            &a.to_matrixfullslice(),
            &b.to_matrixfullslice(),
            'N',
            'N',
            1.0,
            0.0,
        );
        out
    }

    fn matmul_tn(a: &MatrixFull<f64>, b: &MatrixFull<f64>) -> MatrixFull<f64> {
        assert_eq!(a.size[0], b.size[0]);
        let mut out = MatrixFull::new([a.size[1], b.size[1]], 0.0_f64);
        out.to_matrixfullslicemut().lapack_dgemm(
            &a.to_matrixfullslice(),
            &b.to_matrixfullslice(),
            'T',
            'N',
            1.0,
            0.0,
        );
        out
    }

    fn trace_ct_x(c: &MatrixFull<f64>, x: &MatrixFull<f64>) -> f64 {
        assert_eq!(c.size, x.size);
        let mut trace = 0.0_f64;
        for i in 0..c.size[1] {
            for mu in 0..c.size[0] {
                trace += c[(mu, i)] * x[(mu, i)];
            }
        }
        trace
    }

    fn trace_product(a: &MatrixFull<f64>, b: &MatrixFull<f64>) -> f64 {
        assert_eq!(a.size[0], a.size[1]);
        assert_eq!(a.size, b.size);
        let mut trace = 0.0_f64;
        for i in 0..a.size[0] {
            for j in 0..a.size[1] {
                trace += a[(i, j)] * b[(j, i)];
            }
        }
        trace
    }

    fn fragment_dipole_fluctuation_tensor(scf_data: &SCF, atoms: &[usize]) -> [[f64; 3]; 3] {
        let mut cint_data = scf_data.mol.initialize_cint(false);
        let (dipole_raw, dipole_shape) = cint_data.integral_s1::<int1e_r>(None);
        let dipoles = RIFull::from_vec(dipole_shape.try_into().unwrap(), dipole_raw)
            .expect("dipole integral tensor should match libcint shape");
        let dipole_full = (0..3)
            .map(|axis| {
                let mat = dipoles
                    .get_reducing_matrix(axis)
                    .expect("dipole tensor should have three Cartesian components");
                MatrixFull::from_vec(mat.size.try_into().unwrap(), mat.data.to_vec())
                    .expect("dipole slice -> MatrixFull")
            })
            .collect::<Vec<_>>();
        let aoslice = scf_data.mol.aoslice_by_atom();
        let coeff = &scf_data.eigenvectors[0];
        let occ = &scf_data.occupation[0];
        let c_occ = occupied_coeff_matrix(coeff, occ);
        let mut ovlp = scf_data
            .ovlp
            .to_matrixfull()
            .expect("overlap MatrixUpper -> full");
        let s_inv = ovlp
            .lapack_inverse()
            .expect("overlap inverse should exist for closure dipole test");
        let dipole_frag = dipole_full
            .iter()
            .map(|op| restrict_operator_to_fragment(op, &aoslice, atoms))
            .collect::<Vec<_>>();
        let dipole_occ = dipole_frag
            .iter()
            .map(|op| transform_one_body_operator_to_mo(op, &c_occ))
            .collect::<Vec<_>>();
        let mut cov = [[0.0_f64; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                let mu_b_c = matmul_nn(&dipole_frag[b], &c_occ);
                let s_inv_mu_b_c = matmul_nn(&s_inv, &mu_b_c);
                let mu_a_s_inv_mu_b_c = matmul_nn(&dipole_frag[a], &s_inv_mu_b_c);
                let closure_all = trace_ct_x(&c_occ, &mu_a_s_inv_mu_b_c);
                let occupied_projector = trace_product(&dipole_occ[a], &dipole_occ[b]);
                cov[a][b] = 2.0 * (closure_all - occupied_projector);
            }
        }
        cov
    }

    fn dipole_coupling_tensor_z(distance_bohr: f64) -> [[f64; 3]; 3] {
        let r2 = distance_bohr * distance_bohr;
        let r5 = distance_bohr.powi(5);
        let r = [0.0_f64, 0.0_f64, distance_bohr];
        let mut tensor = [[0.0_f64; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                let delta = if a == b { 1.0 } else { 0.0 };
                tensor[a][b] = (3.0 * r[a] * r[b] - delta * r2) / r5;
            }
        }
        tensor
    }

    fn fragment_connected_dipole_x(
        cov_a: [[f64; 3]; 3],
        cov_b: [[f64; 3]; 3],
        distance_bohr: f64,
    ) -> f64 {
        let t = dipole_coupling_tensor_z(distance_bohr);
        let mut x = 0.0_f64;
        for a in 0..3 {
            for b in 0..3 {
                for g in 0..3 {
                    for d in 0..3 {
                        x += t[a][b] * t[g][d] * cov_a[a][g] * cov_b[b][d];
                    }
                }
            }
        }
        x
    }

    fn fragment_mode_operators_from_rimatr(
        rimatr: &MatrixFull<f64>,
        nao: usize,
        aoslice: &[[usize; 4]],
        atoms: &[usize],
    ) -> Vec<MatrixFull<f64>> {
        let mut mask = vec![false; nao];
        for &atom in atoms {
            let [_shl0, _shl1, p0, p1] = aoslice[atom];
            for ao in p0..p1 {
                mask[ao] = true;
            }
        }

        let nmode = rimatr.size[1];
        let mut ops = (0..nmode)
            .map(|_| MatrixFull::new([nao, nao], 0.0_f64))
            .collect::<Vec<_>>();
        for mu in 0..nao {
            if !mask[mu] {
                continue;
            }
            for nu in 0..=mu {
                if !mask[nu] {
                    continue;
                }
                let pair = baspair_index(mu, nu);
                for q in 0..nmode {
                    let value = rimatr[(pair, q)];
                    ops[q][(mu, nu)] = value;
                    ops[q][(nu, mu)] = value;
                }
            }
        }
        ops
    }

    fn one_body_operator_closure_covariance(
        ops: &[MatrixFull<f64>],
        c_occ: &MatrixFull<f64>,
        s_inv: &MatrixFull<f64>,
    ) -> MatrixFull<f64> {
        let nmode = ops.len();
        let nao = c_occ.size[0];
        let nocc = c_occ.size[1];
        let mut y_flat = MatrixFull::new([nao * nocc, nmode], 0.0_f64);
        let mut z_flat = MatrixFull::new([nao * nocc, nmode], 0.0_f64);
        let mut occ_flat = MatrixFull::new([nocc * nocc, nmode], 0.0_f64);

        for (q, op) in ops.iter().enumerate() {
            let y = matmul_nn(op, c_occ);
            let z = matmul_nn(s_inv, &y);
            let occ = matmul_tn(c_occ, &y);
            for i in 0..nocc {
                for mu in 0..nao {
                    let row = mu + nao * i;
                    y_flat[(row, q)] = y[(mu, i)];
                    z_flat[(row, q)] = z[(mu, i)];
                }
            }
            for j in 0..nocc {
                for i in 0..nocc {
                    occ_flat[(i + nocc * j, q)] = occ[(i, j)];
                }
            }
        }

        let mut closure = MatrixFull::new([nmode, nmode], 0.0_f64);
        closure.to_matrixfullslicemut().lapack_dgemm(
            &y_flat.to_matrixfullslice(),
            &z_flat.to_matrixfullslice(),
            'T',
            'N',
            1.0,
            0.0,
        );
        let mut occupied = MatrixFull::new([nmode, nmode], 0.0_f64);
        occupied.to_matrixfullslicemut().lapack_dgemm(
            &occ_flat.to_matrixfullslice(),
            &occ_flat.to_matrixfullslice(),
            'T',
            'N',
            1.0,
            0.0,
        );

        let mut cov = MatrixFull::new([nmode, nmode], 0.0_f64);
        for r in 0..nmode {
            for q in 0..nmode {
                cov[(q, r)] = 2.0 * (closure[(q, r)] - occupied[(q, r)]);
            }
        }
        cov
    }

    fn fragment_rimatr_operator_closure_covariance(
        rimatr: &MatrixFull<f64>,
        nao: usize,
        aoslice: &[[usize; 4]],
        atoms: &[usize],
        c_occ: &MatrixFull<f64>,
        s_inv: &MatrixFull<f64>,
    ) -> MatrixFull<f64> {
        let nmode = rimatr.size[1];
        let nocc = c_occ.size[1];
        let mut frag_aos = Vec::new();
        for &atom in atoms {
            let [_shl0, _shl1, p0, p1] = aoslice[atom];
            frag_aos.extend(p0..p1);
        }

        let mut y_flat = MatrixFull::new([nao * nocc, nmode], 0.0_f64);
        for &mu in &frag_aos {
            for &nu in &frag_aos {
                let pair = baspair_index(mu.max(nu), mu.min(nu));
                for q in 0..nmode {
                    let value = rimatr[(pair, q)];
                    if value.abs() < 1.0e-18 {
                        continue;
                    }
                    for i in 0..nocc {
                        y_flat[(mu + nao * i, q)] += value * c_occ[(nu, i)];
                    }
                }
            }
        }

        let mut z_flat = MatrixFull::new([nao * nocc, nmode], 0.0_f64);
        for q in 0..nmode {
            for i in 0..nocc {
                for mu in 0..nao {
                    let mut value = 0.0_f64;
                    for nu in 0..nao {
                        value += s_inv[(mu, nu)] * y_flat[(nu + nao * i, q)];
                    }
                    z_flat[(mu + nao * i, q)] = value;
                }
            }
        }

        let mut occ_flat = MatrixFull::new([nocc * nocc, nmode], 0.0_f64);
        for q in 0..nmode {
            for j in 0..nocc {
                for i in 0..nocc {
                    let mut value = 0.0_f64;
                    for mu in 0..nao {
                        value += c_occ[(mu, i)] * y_flat[(mu + nao * j, q)];
                    }
                    occ_flat[(i + nocc * j, q)] = value;
                }
            }
        }

        let mut closure = MatrixFull::new([nmode, nmode], 0.0_f64);
        closure.to_matrixfullslicemut().lapack_dgemm(
            &y_flat.to_matrixfullslice(),
            &z_flat.to_matrixfullslice(),
            'T',
            'N',
            1.0,
            0.0,
        );
        let mut occupied = MatrixFull::new([nmode, nmode], 0.0_f64);
        occupied.to_matrixfullslicemut().lapack_dgemm(
            &occ_flat.to_matrixfullslice(),
            &occ_flat.to_matrixfullslice(),
            'T',
            'N',
            1.0,
            0.0,
        );

        let mut cov = MatrixFull::new([nmode, nmode], 0.0_f64);
        for r in 0..nmode {
            for q in 0..nmode {
                cov[(q, r)] = 2.0 * (closure[(q, r)] - occupied[(q, r)]);
            }
        }
        cov
    }

    fn fragment_connected_ri_coulomb_x(
        scf_data: &SCF,
        atoms_a: &[usize],
        atoms_b: &[usize],
    ) -> f64 {
        let (ao_bfs, _p_cart) =
            crate::lib_rint::basis::load_cartesian_rhf_basis_and_density_shell_shared(
                &scf_data.mol.geom,
                &scf_data.mol.basis4elem,
                &scf_data.density_matrix[0],
                "fragment_connected_ri_coulomb_x",
            );
        let auxbasis4elem =
            build_default_r2_etb_auxbasis(&scf_data.mol.geom, &scf_data.mol.basis4elem)
                .expect("default ETB auxiliary basis should be available");
        let aux_bfs = crate::lib_rint::basis::load_aux_molecule_shell_shared_from_raw(
            &scf_data.mol.geom,
            &auxbasis4elem,
        )
        .expect("failed to build shell-shared auxiliary basis");
        let (rimatr, _basbas2baspar, _baspar2basbas) =
            prepare_rimatr_for_r_sync(&ao_bfs, &aux_bfs).expect("failed to build RI-r matrix");

        let coeff_cart = transform_mo_coeff_to_cartesian_shell_shared(
            &scf_data.mol.geom,
            &scf_data.mol.basis4elem,
            &scf_data.eigenvectors[0],
            ao_bfs.len(),
            "fragment_connected_ri_coulomb_x[coeff]",
            false,
        );
        let c_occ = occupied_coeff_matrix(&coeff_cart, &scf_data.occupation[0]);
        let mut ovlp = scf_data
            .ovlp
            .to_matrixfull()
            .expect("overlap MatrixUpper -> full");
        let s_inv = ovlp
            .lapack_inverse()
            .expect("overlap inverse should exist for RI Coulomb closure test");
        let aoslice = scf_data.mol.aoslice_by_atom();
        let cov_a = fragment_rimatr_operator_closure_covariance(
            &rimatr,
            ao_bfs.len(),
            &aoslice,
            atoms_a,
            &c_occ,
            &s_inv,
        );
        let cov_b = fragment_rimatr_operator_closure_covariance(
            &rimatr,
            ao_bfs.len(),
            &aoslice,
            atoms_b,
            &c_occ,
            &s_inv,
        );

        cov_a
            .data
            .iter()
            .zip(cov_b.data.iter())
            .fold(0.0_f64, |acc, (a, b)| acc + a * b)
    }

    fn fragment_mask_from_rint_shells(
        ao_shells: &[crate::lib_rint::basis::RintShell],
        nao: usize,
        atoms: &[usize],
    ) -> Vec<bool> {
        let mut mask = vec![false; nao];
        for shell in ao_shells {
            if atoms.contains(&shell.atom_idx) {
                for ao in shell.ao_start..(shell.ao_start + shell.ao_len) {
                    mask[ao] = true;
                }
            }
        }
        mask
    }

    fn project_density_to_fragment(p: &MatrixFull<f64>, mask: &[bool]) -> MatrixFull<f64> {
        let nao = p.size[0];
        assert_eq!(p.size, [nao, nao]);
        assert_eq!(mask.len(), nao);
        let mut p_frag = MatrixFull::new([nao, nao], 0.0_f64);
        for nu in 0..nao {
            if !mask[nu] {
                continue;
            }
            for mu in 0..nao {
                if mask[mu] {
                    p_frag[(mu, nu)] = p[(mu, nu)];
                }
            }
        }
        p_frag
    }

    fn fragment_cross_r2_jk_features(
        scf_data: &SCF,
        atoms_a: &[usize],
        atoms_b: &[usize],
    ) -> (f64, f64) {
        let (ao_shells, _ao_bfs, p_cart) =
            crate::lib_rint::basis::load_cartesian_rhf_rint_shells_basis_and_density_shell_shared(
                &scf_data.mol.geom,
                &scf_data.mol.basis4elem,
                &scf_data.density_matrix[0],
                "fragment_cross_r2_jk_features",
            );
        let auxbasis4elem =
            build_default_r2_etb_auxbasis(&scf_data.mol.geom, &scf_data.mol.basis4elem)
                .expect("default r2 ETB auxiliary basis should be available");
        let aux_shells = crate::lib_rint::basis::load_aux_rint_shells_from_raw(
            &scf_data.mol.geom,
            &auxbasis4elem,
        )
        .expect("failed to build r2 auxiliary shells");
        let ri_r2 = prepare_rimatr_for_r2_shell_blocks_sync(&ao_shells, &aux_shells);
        assert!(
            ri_r2.is_some(),
            "failed to build RI-r2 matrix for ablation features"
        );

        let nao = p_cart.size[0];
        let mask_a = fragment_mask_from_rint_shells(&ao_shells, nao, atoms_a);
        let mask_b = fragment_mask_from_rint_shells(&ao_shells, nao, atoms_b);
        let p_a = project_density_to_fragment(&p_cart, &mask_a);
        let p_b = project_density_to_fragment(&p_cart, &mask_b);
        let dm_a = vec![p_a.clone()];
        let dm_b = vec![p_b.clone()];

        let j_a = vj_upper_with_rimatr_r2_sync(&ri_r2, &dm_a, 1, 1.0)
            .remove(0)
            .to_matrixfull()
            .expect("RI-r2 J_A upper -> full");
        let j_b = vj_upper_with_rimatr_r2_sync(&ri_r2, &dm_b, 1, 1.0)
            .remove(0)
            .to_matrixfull()
            .expect("RI-r2 J_B upper -> full");
        let k_a = vk_upper_with_rimatr_r2_sync(&ri_r2, &dm_a, 1, 1.0)
            .remove(0)
            .to_matrixfull()
            .expect("RI-r2 K_A upper -> full");
        let k_b = vk_upper_with_rimatr_r2_sync(&ri_r2, &dm_b, 1, 1.0)
            .remove(0)
            .to_matrixfull()
            .expect("RI-r2 K_B upper -> full");

        let ej_ab = 0.5_f64 * (dot2(&p_a, &j_b) + dot2(&p_b, &j_a));
        let ek_ab = -0.25_f64 * (dot2(&p_a, &k_b) + dot2(&p_b, &k_a));
        (ej_ab, ek_ab)
    }

    fn log_log_slope(points: &[(f64, f64)]) -> f64 {
        let n = points.len() as f64;
        let sum_x = points.iter().map(|(x, _)| x.ln()).sum::<f64>();
        let sum_y = points.iter().map(|(_, y)| y.abs().ln()).sum::<f64>();
        let sum_xx = points.iter().map(|(x, _)| x.ln() * x.ln()).sum::<f64>();
        let sum_xy = points
            .iter()
            .map(|(x, y)| x.ln() * y.abs().ln())
            .sum::<f64>();
        (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    }

    fn log_log_slope_finite(points: &[(f64, f64)]) -> Option<f64> {
        let finite = points
            .iter()
            .copied()
            .filter(|(x, y)| x.is_finite() && *x > 0.0 && y.is_finite() && y.abs() > 1.0e-300)
            .collect::<Vec<_>>();
        if finite.len() < 3 {
            None
        } else {
            Some(log_log_slope(&finite))
        }
    }

    fn run_ablation_feature_scaling_case<F>(
        label: &str,
        distances_ang: &[f64],
        make_ctrl: F,
        atoms_a: &[usize],
        atoms_b: &[usize],
    ) where
        F: Fn(f64) -> String,
    {
        let mut x_points = Vec::new();
        let mut ej_points = Vec::new();
        let mut ek_points = Vec::new();
        for &distance_ang in distances_ang {
            let ctrl_path = make_ctrl(distance_ang);
            let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
            let mut scf_data = SCF::build(mol, &None);
            scf_without_build(&mut scf_data, &None);
            let distance_bohr = distance_ang / crate::constants::ANG;
            let x = fragment_connected_ri_coulomb_x(&scf_data, atoms_a, atoms_b);
            let (ej_r2, ek_r2) = fragment_cross_r2_jk_features(&scf_data, atoms_a, atoms_b);
            println!(
                "ablation features {label}: R={distance_ang:.3} Ang ({distance_bohr:.6} bohr) X_AB^RI={x:.16e} E_J^r2_cross={ej_r2:.16e} E_K^r2_cross={ek_r2:.16e} logR={:.8} log|X|={:.8} log|EJ_r2|={:.8} log|EK_r2|={:.8}",
                distance_bohr.ln(),
                x.abs().ln(),
                ej_r2.abs().ln(),
                ek_r2.abs().ln(),
            );
            x_points.push((distance_bohr, x));
            ej_points.push((distance_bohr, ej_r2));
            ek_points.push((distance_bohr, ek_r2));
            let _ = fs::remove_file(ctrl_path);
        }

        let x_slope = log_log_slope(&x_points);
        let ej_slope = log_log_slope_finite(&ej_points);
        let ek_slope = log_log_slope_finite(&ek_points);
        println!(
            "ablation feature slopes {label}: X_AB^RI={x_slope:.6} E_J^r2_cross={} E_K^r2_cross={}",
            ej_slope
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "n/a".to_string()),
            ek_slope
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "n/a".to_string()),
        );
        assert!(
            (x_slope + 6.0).abs() < 1.0,
            "expected X_AB^RI slope near -6 for {label}, got {x_slope}"
        );
    }

    fn write_temp_ctrl_h2_with_aux(basis_dir: &str, aux_basis_dir: &str) -> String {
        let pid = std::process::id();
        let path = format!("/tmp/lib_rint_kernel_ri_check_{basis_dir}_{aux_basis_dir}_{pid}.toml")
            .replace('(', "")
            .replace(')', "")
            .replace('/', "_")
            .replace(' ', "_");
        let text = format!(
            r#"[ctrl]
print_level = 0
xc = "hf"
basis_path = "/home/cfh/rest_workspace/rest/basis-set-pool/{basis_dir}"
auxbas_path = "/home/cfh/rest_workspace/rest/basis-set-pool/{aux_basis_dir}"
basis_type = "Cartesian"
auxbas_type = "Cartesian"
use_auxbas = true
even_tempered_basis = false
charge = 0.0
spin = 1.0
spin_polarization = false
num_threads = 1
run_lib_rint = false
[geom]
name = "H2"
unit = "angstrom"
position = [
    "H   0.0000000000   0.0000000000   0.0000000000",
    "H   0.0000000000   0.0000000000   1.4000000000",
]
"#
        );
        fs::write(&path, text).unwrap();
        path
    }

    fn compare_r2_direct_with_incore_for_ctrl(ctrl_path: &str, label: &str) -> [f64; 2] {
        let mol = Molecule::build(ctrl_path.to_string(), None).unwrap();
        let mut scf_data = SCF::build(mol, &None);
        scf_without_build(&mut scf_data, &None);

        let (ao_shells, _bfs, p_cart) =
            crate::lib_rint::basis::load_cartesian_rhf_rint_shells_basis_and_density_shell_shared(
                &scf_data.mol.geom,
                &scf_data.mol.basis4elem,
                &scf_data.density_matrix[0],
                label,
            );
        let auxbasis4elem =
            build_default_r2_etb_auxbasis(&scf_data.mol.geom, &scf_data.mol.basis4elem)
                .expect("default r2 ETB auxiliary basis should be available");
        let aux_shells = crate::lib_rint::basis::load_aux_rint_shells_from_raw(
            &scf_data.mol.geom,
            &auxbasis4elem,
        )
        .expect("failed to build aux rint shells");
        let dm = vec![p_cart];

        let ri = prepare_rimatr_for_r2_shell_blocks_sync(&ao_shells, &aux_shells)
            .expect("failed to build incore RI-r2 matrix");
        let j_incore_u = vj_upper_with_rimatr_r2_sync(&Some(ri.clone()), &dm, 1, 1.0);
        let k_incore_u = vk_upper_with_rimatr_r2_sync(&Some(ri), &dm, 1, 1.0);
        let j_incore = j_incore_u[0].to_matrixfull().unwrap();
        let k_incore = k_incore_u[0].to_matrixfull().unwrap();

        let coeff_cart = [transform_mo_coeff_to_cartesian_shell_shared(
            &scf_data.mol.geom,
            &scf_data.mol.basis4elem,
            &scf_data.eigenvectors[0],
            rint_shell_basis_count(&ao_shells),
            label,
            false,
        )];
        let occupation = [scf_data.occupation[0].clone()];
        let (j_semi_u, k_semi_u) = r2_jk_direct_from_shell_blocks_with_auxbas_threshold(
            &ao_shells,
            &aux_shells,
            &dm,
            Some(&coeff_cart),
            Some(&occupation),
            AUXBAS_THRESHOLD,
        )
        .expect("failed to build semi-direct RI-r2 J/K");
        let j_semi = j_semi_u[0].to_matrixfull().unwrap();
        let k_semi = k_semi_u[0].to_matrixfull().unwrap();
        [
            max_abs_diff_matrix(&j_incore, &j_semi),
            max_abs_diff_matrix(&k_incore, &k_semi),
        ]
    }

    #[test]
    fn r2_direct_matches_incore_for_h2_sto3g() {
        let ctrl_path = write_temp_ctrl_h2("sto-3g");
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
        let mut scf_data = SCF::build(mol, &None);
        scf_without_build(&mut scf_data, &None);

        let (ao_shells, _bfs, p_cart) =
            crate::lib_rint::basis::load_cartesian_rhf_rint_shells_basis_and_density_shell_shared(
                &scf_data.mol.geom,
                &scf_data.mol.basis4elem,
                &scf_data.density_matrix[0],
                "r2_direct_matches_incore_for_h2_sto3g",
            );
        let auxbasis4elem =
            build_default_r2_etb_auxbasis(&scf_data.mol.geom, &scf_data.mol.basis4elem)
                .expect("default r2 ETB auxiliary basis should be available");
        let aux_shells = crate::lib_rint::basis::load_aux_rint_shells_from_raw(
            &scf_data.mol.geom,
            &auxbasis4elem,
        )
        .expect("failed to build aux rint shells");
        let dm = vec![p_cart];

        let ri = prepare_rimatr_for_r2_shell_blocks_sync(&ao_shells, &aux_shells)
            .expect("failed to build incore RI-r2 matrix");
        let j_incore_u = vj_upper_with_rimatr_r2_sync(&Some(ri.clone()), &dm, 1, 1.0);
        let k_incore_u = vk_upper_with_rimatr_r2_sync(&Some(ri), &dm, 1, 1.0);

        let j_incore = j_incore_u[0].to_matrixfull().unwrap();
        let k_incore = k_incore_u[0].to_matrixfull().unwrap();

        let coeff_cart = [transform_mo_coeff_to_cartesian_shell_shared(
            &scf_data.mol.geom,
            &scf_data.mol.basis4elem,
            &scf_data.eigenvectors[0],
            rint_shell_basis_count(&ao_shells),
            "r2_direct_matches_incore_for_h2_sto3g[semi-direct]",
            false,
        )];
        let occupation = [scf_data.occupation[0].clone()];
        let (j_semi_u, k_semi_u) = r2_jk_direct_from_shell_blocks_with_auxbas_threshold(
            &ao_shells,
            &aux_shells,
            &dm,
            Some(&coeff_cart),
            Some(&occupation),
            AUXBAS_THRESHOLD,
        )
        .expect("failed to build semi-direct RI-r2 J/K");
        let j_semi = j_semi_u[0].to_matrixfull().unwrap();
        let k_semi = k_semi_u[0].to_matrixfull().unwrap();
        let d_j_semi = max_abs_diff_matrix(&j_incore, &j_semi);
        let d_k_semi = max_abs_diff_matrix(&k_incore, &k_semi);
        println!("H2/STO-3G RI-r2 semi-direct vs incore: dJ={d_j_semi:.3e} dK={d_k_semi:.3e}");
        assert!(
            d_j_semi < R2_SEMIDIRECT_REGRESSION_TOL,
            "RI-r2 semi-direct J drifted from incore: {d_j_semi}"
        );
        assert!(
            d_k_semi < R2_SEMIDIRECT_REGRESSION_TOL,
            "RI-r2 semi-direct K drifted from incore: {d_k_semi}"
        );

        let _ = fs::remove_file(ctrl_path);
    }

    #[test]
    fn r2_incore_occ_coeff_k_matches_semidirect_h2_sto3g() {
        let ctrl_path = write_temp_ctrl_h2("sto-3g");
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
        let mut scf_data = SCF::build(mol, &None);
        scf_without_build(&mut scf_data, &None);

        let (ao_shells, _bfs, p_cart) =
            crate::lib_rint::basis::load_cartesian_rhf_rint_shells_basis_and_density_shell_shared(
                &scf_data.mol.geom,
                &scf_data.mol.basis4elem,
                &scf_data.density_matrix[0],
                "r2_incore_occ_coeff_k_matches_semidirect_h2_sto3g",
            );
        let auxbasis4elem =
            build_default_r2_etb_auxbasis(&scf_data.mol.geom, &scf_data.mol.basis4elem)
                .expect("default r2 ETB auxiliary basis should be available");
        let aux_shells = crate::lib_rint::basis::load_aux_rint_shells_from_raw(
            &scf_data.mol.geom,
            &auxbasis4elem,
        )
        .expect("failed to build aux rint shells");
        let dm = vec![p_cart];

        let ri = prepare_rimatr_for_r2_shell_blocks_sync(&ao_shells, &aux_shells)
            .expect("failed to build incore RI-r2 matrix");
        let coeff_alpha = transform_mo_coeff_to_cartesian_shell_shared(
            &scf_data.mol.geom,
            &scf_data.mol.basis4elem,
            &scf_data.eigenvectors[0],
            rint_shell_basis_count(&ao_shells),
            "r2_incore_occ_coeff_k_matches_semidirect_h2_sto3g[alpha]",
            false,
        );
        let coeff_cart = [coeff_alpha.clone(), MatrixFull::empty()];
        let occupation = [scf_data.occupation[0].clone(), Vec::new()];
        let num_elec_alpha = occupation[0].iter().sum::<f64>();
        let num_elec = [num_elec_alpha, num_elec_alpha, 0.0_f64];
        let k_occ_u =
            vk_upper_with_rimatr_sync_v03(&Some(ri), &coeff_cart, &num_elec, &occupation, 1, 1.0);
        let k_occ = k_occ_u[0].to_matrixfull().unwrap();

        let coeff_semi = [coeff_alpha];
        let occupation_semi = [occupation[0].clone()];
        let (_j_semi_u, k_semi_u) = r2_jk_direct_from_shell_blocks_with_auxbas_threshold(
            &ao_shells,
            &aux_shells,
            &dm,
            Some(&coeff_semi),
            Some(&occupation_semi),
            AUXBAS_THRESHOLD,
        )
        .expect("failed to build semi-direct RI-r2 J/K");
        let k_semi = k_semi_u[0].to_matrixfull().unwrap();
        let d_k = max_abs_diff_matrix(&k_occ, &k_semi);
        println!("H2/STO-3G RI-r2 incore occ-coeff K vs semi-direct: dK={d_k:.3e}");
        assert!(
            d_k < R2_SEMIDIRECT_REGRESSION_TOL,
            "RI-r2 incore occ-coeff K drifted from semi-direct: {d_k}"
        );

        let _ = fs::remove_file(ctrl_path);
    }

    #[test]
    fn r2_direct_matches_incore_for_medium_sto3g_molecules() {
        let cases = [
            ("H2O/STO-3G", write_temp_ctrl_h2o("sto-3g")),
            ("NH3/STO-3G", write_temp_ctrl_nh3("sto-3g")),
        ];

        for (label, ctrl_path) in cases {
            let [d_j_semi, d_k_semi] = compare_r2_direct_with_incore_for_ctrl(&ctrl_path, label);
            println!("{label} RI-r2 semi-direct vs incore: dJ={d_j_semi:.3e} dK={d_k_semi:.3e}");
            assert!(
                d_j_semi < R2_SEMIDIRECT_MEDIUM_REGRESSION_TOL,
                "{label} RI-r2 semi-direct J drifted from incore: {d_j_semi}"
            );
            assert!(
                d_k_semi < R2_SEMIDIRECT_MEDIUM_REGRESSION_TOL,
                "{label} RI-r2 semi-direct K drifted from incore: {d_k_semi}"
            );
            let _ = fs::remove_file(ctrl_path);
        }
    }

    #[test]
    #[ignore]
    fn occ_closure_dipole_connected_he2_loglog_slope() {
        let distances_ang = [4.0_f64, 5.0, 6.0, 7.0, 8.0, 10.0];
        let mut points = Vec::new();
        for distance_ang in distances_ang {
            let ctrl_path = write_temp_ctrl_he2("cc-pVDZ", distance_ang, "dipole_connected");
            let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
            let mut scf_data = SCF::build(mol, &None);
            scf_without_build(&mut scf_data, &None);
            let cov_a = fragment_dipole_fluctuation_tensor(&scf_data, &[0]);
            let cov_b = fragment_dipole_fluctuation_tensor(&scf_data, &[1]);
            let distance_bohr = distance_ang / crate::constants::ANG;
            let x = fragment_connected_dipole_x(cov_a, cov_b, distance_bohr);
            println!(
                "occ-closure dipole X_disp(He2): R={distance_ang:.3} Ang ({distance_bohr:.6} bohr) X={x:.16e} logR={:.8} log|X|={:.8}",
                distance_bohr.ln(),
                x.abs().ln(),
            );
            points.push((distance_bohr, x));
            let _ = fs::remove_file(ctrl_path);
        }
        let slope = log_log_slope(&points);
        println!("occ-closure dipole connected He2 log|X| vs log R slope = {slope:.6}");
        assert!(
            (slope + 6.0).abs() < 0.5,
            "expected long-range dipole connected descriptor slope near -6, got {slope}"
        );
    }

    #[test]
    #[ignore]
    fn occ_closure_ri_coulomb_connected_he2_loglog_slope() {
        let distances_ang = [4.0_f64, 5.0, 6.0, 7.0, 8.0, 10.0];
        let mut points = Vec::new();
        for distance_ang in distances_ang {
            let ctrl_path = write_temp_ctrl_he2("cc-pVDZ", distance_ang, "ri_coulomb_connected");
            let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
            let mut scf_data = SCF::build(mol, &None);
            scf_without_build(&mut scf_data, &None);
            let distance_bohr = distance_ang / crate::constants::ANG;
            let x = fragment_connected_ri_coulomb_x(&scf_data, &[0], &[1]);
            println!(
                "occ-closure RI Coulomb X_disp(He2): R={distance_ang:.3} Ang ({distance_bohr:.6} bohr) X={x:.16e} logR={:.8} log|X|={:.8}",
                distance_bohr.ln(),
                x.abs().ln(),
            );
            points.push((distance_bohr, x));
            let _ = fs::remove_file(ctrl_path);
        }
        let slope = log_log_slope(&points);
        println!("occ-closure RI Coulomb connected He2 log|X| vs log R slope = {slope:.6}");
        assert!(
            (slope + 6.0).abs() < 1.0,
            "expected long-range RI Coulomb connected descriptor slope near -6, got {slope}"
        );
    }

    #[test]
    fn public_occ_closure_ri_coulomb_x_ab_he2_is_positive() {
        let ctrl_path = write_temp_ctrl_he2("cc-pVDZ", 4.0, "public_x_ab");
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
        let mut scf_data = SCF::build(mol, &None);
        scf_without_build(&mut scf_data, &None);

        let x_ab =
            crate::lib_rint::lib_vee_rhf_occ_closure_connected_ri_coulomb_x(&scf_data, &[0], &[1]);

        assert!(
            x_ab > 0.0,
            "expected a positive occupied-closure connected RI Coulomb X_AB, got {x_ab}"
        );
        let _ = fs::remove_file(ctrl_path);
    }

    #[test]
    #[ignore]
    fn occ_closure_ri_coulomb_connected_noble_dimers_loglog_slope() {
        for (elem, basis_dir, distances_ang) in [("Ne", "cc-pVDZ", [4.0_f64, 5.0, 7.0, 10.0])] {
            let mut points = Vec::new();
            for distance_ang in distances_ang {
                let ctrl_path = write_temp_ctrl_diatomic_dimer(
                    elem,
                    basis_dir,
                    distance_ang,
                    "ri_coulomb_connected",
                );
                let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
                let mut scf_data = SCF::build(mol, &None);
                scf_without_build(&mut scf_data, &None);
                let distance_bohr = distance_ang / crate::constants::ANG;
                let x = fragment_connected_ri_coulomb_x(&scf_data, &[0], &[1]);
                println!(
                    "occ-closure RI Coulomb X_disp({elem}2/{basis_dir}): R={distance_ang:.3} Ang ({distance_bohr:.6} bohr) X={x:.16e} logR={:.8} log|X|={:.8}",
                    distance_bohr.ln(),
                    x.abs().ln(),
                );
                points.push((distance_bohr, x));
                let _ = fs::remove_file(ctrl_path);
            }
            let slope = log_log_slope(&points);
            println!(
                "occ-closure RI Coulomb connected {elem}2/{basis_dir} log|X| vs log R slope = {slope:.6}"
            );
            assert!(
                (slope + 6.0).abs() < 1.0,
                "expected long-range RI Coulomb connected descriptor slope near -6 for {elem}2/{basis_dir}, got {slope}"
            );
        }
    }

    #[test]
    #[ignore]
    fn occ_closure_ri_coulomb_connected_methane_dimer_loglog_slope() {
        let distances_ang = [5.0_f64, 6.0, 7.0, 8.0, 10.0, 12.0];
        let mut points = Vec::new();
        for distance_ang in distances_ang {
            let ctrl_path =
                write_temp_ctrl_methane_dimer("sto-3g", distance_ang, "ri_coulomb_connected");
            let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
            let mut scf_data = SCF::build(mol, &None);
            scf_without_build(&mut scf_data, &None);
            let distance_bohr = distance_ang / crate::constants::ANG;
            let x = fragment_connected_ri_coulomb_x(&scf_data, &[0, 1, 2, 3, 4], &[5, 6, 7, 8, 9]);
            println!(
                "occ-closure RI Coulomb X_disp(CH4--CH4/STO-3G): R={distance_ang:.3} Ang ({distance_bohr:.6} bohr) X={x:.16e} logR={:.8} log|X|={:.8}",
                distance_bohr.ln(),
                x.abs().ln(),
            );
            points.push((distance_bohr, x));
            let _ = fs::remove_file(ctrl_path);
        }
        let slope = log_log_slope(&points);
        println!(
            "occ-closure RI Coulomb connected CH4--CH4/STO-3G log|X| vs log R slope = {slope:.6}"
        );
        assert!(
            (slope + 6.0).abs() < 1.0,
            "expected long-range RI Coulomb connected descriptor slope near -6 for CH4 dimer, got {slope}"
        );
    }

    #[test]
    #[ignore]
    fn occ_closure_ablation_feature_scaling_loglog() {
        run_ablation_feature_scaling_case(
            "He2/cc-pVDZ",
            &[4.0_f64, 5.0, 6.0, 7.0, 8.0, 10.0],
            |distance_ang| write_temp_ctrl_he2("cc-pVDZ", distance_ang, "ablation_features"),
            &[0],
            &[1],
        );
        run_ablation_feature_scaling_case(
            "Ne2/cc-pVDZ",
            &[4.0_f64, 5.0, 7.0, 10.0],
            |distance_ang| {
                write_temp_ctrl_diatomic_dimer("Ne", "cc-pVDZ", distance_ang, "ablation_features")
            },
            &[0],
            &[1],
        );
        run_ablation_feature_scaling_case(
            "CH4--CH4/STO-3G",
            &[5.0_f64, 6.0, 7.0, 8.0, 10.0, 12.0],
            |distance_ang| {
                write_temp_ctrl_methane_dimer("sto-3g", distance_ang, "ablation_features")
            },
            &[0, 1, 2, 3, 4],
            &[5, 6, 7, 8, 9],
        );
    }

    #[test]
    fn r2_shell_block_windows_match_ao_scalar_windows() {
        let ctrl_path = write_temp_ctrl_h2("sto-3g");
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();

        let ao_shells =
            crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
                .expect("failed to build AO rint shells");
        let ao_bfs = crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&ao_shells)
            .expect("failed to expand AO rint shells");
        let auxbasis4elem = build_default_r2_etb_auxbasis(&mol.geom, &mol.basis4elem)
            .expect("default r2 ETB auxiliary basis should be available");
        let aux_shells =
            crate::lib_rint::basis::load_aux_rint_shells_from_raw(&mol.geom, &auxbasis4elem)
                .expect("failed to build auxiliary rint shells");
        let aux_bfs = crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&aux_shells)
            .expect("failed to expand auxiliary rint shells");

        let ao_a = &ao_shells[0];
        let ao_b = ao_shells.get(1).unwrap_or(ao_a);
        let aux_a = &aux_shells[0];
        let aux_b = aux_shells.get(1).unwrap_or(aux_a);

        let block_2c = int2c_r2_shell_block(aux_a, aux_b);
        for j in 0..aux_b.ao_len {
            for i in 0..aux_a.ao_len {
                let got = block_2c[(i, j)];
                let expect = eri_ao_2c_r2(&aux_bfs, aux_a.ao_start + i, aux_b.ao_start + j);
                assert!(
                    (got - expect).abs() < 1.0e-12,
                    "2c shell block mismatch at ({i},{j}): {got} vs {expect}"
                );
            }
        }

        let block_3c = int3c_r2_shell_block(ao_a, ao_b, aux_a);
        for p in 0..aux_a.ao_len {
            for j in 0..ao_b.ao_len {
                for i in 0..ao_a.ao_len {
                    let row = j * ao_a.ao_len + i;
                    let got = block_3c[(row, p)];
                    let expect = eri_ao_3c_r2(
                        &ao_bfs,
                        &aux_bfs,
                        ao_a.ao_start + i,
                        ao_b.ao_start + j,
                        aux_a.ao_start + p,
                    );
                    assert!(
                        (got - expect).abs() < 1.0e-12,
                        "3c shell block mismatch at row={row}, p={p}: {got} vs {expect}"
                    );
                }
            }
        }

        let block_4c = int4c_r2_shell_block(ao_a, ao_b, ao_a, ao_b);
        for l in 0..ao_b.ao_len {
            for k in 0..ao_a.ao_len {
                let col = l * ao_a.ao_len + k;
                for j in 0..ao_b.ao_len {
                    for i in 0..ao_a.ao_len {
                        let row = j * ao_a.ao_len + i;
                        let got = block_4c[(row, col)];
                        let expect = eri_ao_4c_r2(
                            &ao_bfs,
                            ao_a.ao_start + i,
                            ao_b.ao_start + j,
                            ao_a.ao_start + k,
                            ao_b.ao_start + l,
                        );
                        assert!(
                            (got - expect).abs() < 1.0e-12,
                            "4c shell block mismatch at row={row}, col={col}: {got} vs {expect}"
                        );
                    }
                }
            }
        }

        let _ = fs::remove_file(ctrl_path);
    }

    #[test]
    fn r_shell_block_4c_matches_ao_scalar_window() {
        let ctrl_path = write_temp_ctrl_h2("sto-3g");
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();

        let ao_shells =
            crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
                .expect("failed to build AO rint shells");
        let ao_bfs = crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&ao_shells)
            .expect("failed to expand AO rint shells");

        let shell_a = &ao_shells[0];
        let shell_b = ao_shells.get(1).unwrap_or(shell_a);
        let shell_c = ao_shells.get(2).unwrap_or(shell_a);
        let shell_d = ao_shells.get(3).unwrap_or(shell_b);

        let block_4c = int4c_r_shell_block(shell_a, shell_b, shell_c, shell_d);
        for l in 0..shell_d.ao_len {
            for k in 0..shell_c.ao_len {
                let col = l * shell_c.ao_len + k;
                for j in 0..shell_b.ao_len {
                    for i in 0..shell_a.ao_len {
                        let row = j * shell_a.ao_len + i;
                        let got = block_4c[(row, col)];
                        let expect = eri_ao_4c_r(
                            &ao_bfs,
                            shell_a.ao_start + i,
                            shell_b.ao_start + j,
                            shell_c.ao_start + k,
                            shell_d.ao_start + l,
                        );
                        assert!(
                            (got - expect).abs() < 1.0e-12,
                            "1/r 4c shell block mismatch at row={row}, col={col}: {got} vs {expect}"
                        );
                    }
                }
            }
        }

        let _ = fs::remove_file(ctrl_path);
    }

    #[test]
    fn r_shell_block_into_data_matches_allocating_shell_block() {
        let ctrl_path = write_temp_ctrl_h2("sto-3g");
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();

        let ao_shells =
            crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
                .expect("failed to build AO rint shells");

        let shell_a = &ao_shells[0];
        let shell_b = ao_shells.get(1).unwrap_or(shell_a);
        let shell_c = ao_shells.get(2).unwrap_or(shell_a);
        let shell_d = ao_shells.get(3).unwrap_or(shell_b);

        let expected = int4c_r_shell_block(shell_a, shell_b, shell_c, shell_d);
        let mut entries = Vec::new();
        let mut data = Vec::new();
        let shape = int4c_r_shell_block_batched_into_data(
            shell_a,
            shell_b,
            shell_c,
            shell_d,
            &mut entries,
            &mut data,
        );

        assert_eq!(shape, expected.size);
        assert_eq!(data.len(), expected.data.len());
        for (idx, (got, expect)) in data.iter().zip(expected.data.iter()).enumerate() {
            assert!(
                (*got - *expect).abs() < 1.0e-12,
                "reused 1/r 4c shell block mismatch at data[{idx}]: {got} vs {expect}"
            );
        }

        let _ = fs::remove_file(ctrl_path);
    }

    #[test]
    fn r_shell_block_with_cached_primitive_pairs_matches_default() {
        let ctrl_path = write_temp_ctrl_h2("sto-3g");
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();

        let ao_shells =
            crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
                .expect("failed to build AO rint shells");

        let shell_a = &ao_shells[0];
        let shell_b = ao_shells.get(1).unwrap_or(shell_a);
        let shell_c = ao_shells.get(2).unwrap_or(shell_a);
        let shell_d = ao_shells.get(3).unwrap_or(shell_b);
        let expected = int4c_r_shell_block(shell_a, shell_b, shell_c, shell_d);

        let shell_coeffs = build_rint_shell_coefficient_cache(&ao_shells);
        let ab_pairs = build_primitive_pairs_4c(
            shell_a,
            shell_b,
            &shell_coeffs[0],
            &shell_coeffs[ao_shells
                .iter()
                .position(|shell| std::ptr::eq(shell, shell_b))
                .unwrap_or(0)],
            distance_squared(&shell_a.center, &shell_b.center),
        );
        let c_idx = ao_shells
            .iter()
            .position(|shell| std::ptr::eq(shell, shell_c))
            .unwrap_or(0);
        let d_idx = ao_shells
            .iter()
            .position(|shell| std::ptr::eq(shell, shell_d))
            .unwrap_or(0);
        let cd_pairs = build_primitive_pairs_4c(
            shell_c,
            shell_d,
            &shell_coeffs[c_idx],
            &shell_coeffs[d_idx],
            distance_squared(&shell_c.center, &shell_d.center),
        );
        let mut entries = Vec::new();
        let mut data = Vec::new();
        let shape = int4c_r_shell_block_batched_into_data_with_pairs(
            shell_a,
            shell_b,
            shell_c,
            shell_d,
            &ab_pairs,
            &cd_pairs,
            &mut entries,
            &mut data,
        );

        assert_eq!(shape, expected.size);
        for (idx, (got, expect)) in data.iter().zip(expected.data.iter()).enumerate() {
            assert!(
                (*got - *expect).abs() < 1.0e-12,
                "cached-pair 1/r 4c shell block mismatch at data[{idx}]: {got} vs {expect}"
            );
        }

        let _ = fs::remove_file(ctrl_path);
    }

    #[test]
    fn r_shell_block_4c_matches_ao_scalar_high_ang_ccpvdz() {
        let ctrl_path = write_temp_ctrl_h2o("cc-pVDZ");
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();

        let ao_shells =
            crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
                .expect("failed to build AO rint shells");
        let ao_bfs = crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&ao_shells)
            .expect("failed to expand AO rint shells");

        let shell_a = ao_shells
            .iter()
            .max_by_key(|shell| shell.shell.ang_type)
            .expect("missing AO shell");
        let shell_b = shell_a;
        let shell_c = ao_shells
            .iter()
            .find(|shell| shell.shell.ang_type == 1)
            .unwrap_or(shell_a);
        let shell_d = ao_shells
            .iter()
            .find(|shell| shell.shell.ang_type == 0)
            .unwrap_or(shell_a);

        let block_4c = int4c_r_shell_block(shell_a, shell_b, shell_c, shell_d);
        for l in 0..shell_d.ao_len {
            for k in 0..shell_c.ao_len {
                let col = l * shell_c.ao_len + k;
                for j in 0..shell_b.ao_len {
                    for i in 0..shell_a.ao_len {
                        let row = j * shell_a.ao_len + i;
                        let got = block_4c[(row, col)];
                        let expect = eri_ao_4c_r(
                            &ao_bfs,
                            shell_a.ao_start + i,
                            shell_b.ao_start + j,
                            shell_c.ao_start + k,
                            shell_d.ao_start + l,
                        );
                        assert!(
                            (got - expect).abs() < 1.0e-10,
                            "high-ang 1/r 4c shell block mismatch at row={row}, col={col}: {got} vs {expect}"
                        );
                    }
                }
            }
        }

        let _ = fs::remove_file(ctrl_path);
    }

    #[test]
    #[ignore = "expensive diagnostic; use targeted full-tensor mismatch diagnostics first"]
    fn r_shell_block_4c_matches_ao_scalar_def2_tzvpp_all_shells() {
        let ctrl_path = write_temp_ctrl_h2o("def2-tzvpp");
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();

        let ao_shells =
            crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
                .expect("failed to build AO rint shells");
        let ao_bfs = crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&ao_shells)
            .expect("failed to expand AO rint shells");

        for (a_idx, shell_a) in ao_shells.iter().enumerate() {
            for (b_idx, shell_b) in ao_shells.iter().enumerate() {
                for (c_idx, shell_c) in ao_shells.iter().enumerate() {
                    for (d_idx, shell_d) in ao_shells.iter().enumerate() {
                        let block_4c = int4c_r_shell_block(shell_a, shell_b, shell_c, shell_d);
                        for l in 0..shell_d.ao_len {
                            for k in 0..shell_c.ao_len {
                                let col = l * shell_c.ao_len + k;
                                for j in 0..shell_b.ao_len {
                                    for i in 0..shell_a.ao_len {
                                        let row = j * shell_a.ao_len + i;
                                        let got = block_4c[(row, col)];
                                        let expect = eri_ao_4c_r(
                                            &ao_bfs,
                                            shell_a.ao_start + i,
                                            shell_b.ao_start + j,
                                            shell_c.ao_start + k,
                                            shell_d.ao_start + l,
                                        );
                                        assert!(
                                            (got - expect).abs()
                                                <= 1.0e-10 * expect.abs().max(1.0),
                                            "def2-tzvpp shell block mismatch shells=({a_idx},{b_idx},{c_idx},{d_idx}) ang=({},{},{},{}) local=({i},{j},{k},{l}) row={row} col={col} got={got:.16e} expect={expect:.16e}",
                                            shell_a.shell.ang_type,
                                            shell_b.shell.ang_type,
                                            shell_c.shell.ang_type,
                                            shell_d.shell.ang_type,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let _ = fs::remove_file(ctrl_path);
    }

    #[test]
    fn rys_roots_weights_r_reproduce_boys_moments_through_nine_roots() {
        for nroots in 1..=9 {
            for t in [0.0_f64, 1.0e-8, 0.2, 1.0384010260937269, 2.0, 20.0, 80.0] {
                let (roots, weights) = rys_roots_weights_r(nroots, t);
                let moments = boys_vec(2 * nroots - 1, t);
                assert_eq!(roots.len(), nroots);
                assert_eq!(weights.len(), nroots);
                for moment_idx in 0..=(2 * nroots - 1) {
                    let got = roots
                        .iter()
                        .zip(weights.iter())
                        .map(|(root, weight)| weight * root.powi(moment_idx as i32))
                        .sum::<f64>();
                    let expect = moments[moment_idx];
                    let tol = if nroots <= 6 { 1.0e-9 } else { 5.0e-8 } * expect.abs().max(1.0);
                    assert!(
                        (got - expect).abs() <= tol,
                        "Rys moment mismatch nroots={nroots}, t={t}, moment={moment_idx}: {got} vs {expect}"
                    );
                }
            }
        }
    }

    #[test]
    fn rys_roots_weights_r_reproduce_boys_moments_through_twenty_five_roots() {
        for nroots in 4..=25 {
            for t in [0.0_f64, 1.0e-8, 0.2, 1.0384010260937269, 2.0, 20.0, 80.0] {
                let moments = boys_vec(2 * nroots, t);
                let (roots, weights) = rys_roots_weights_r(nroots, t);
                for idx in 0..nroots {
                    assert!(
                        roots[idx].is_finite()
                            && roots[idx] >= -1.0e-12
                            && roots[idx] <= 1.0 + 1.0e-12,
                        "invalid root nroots={nroots}, t={t}, idx={idx}: got={:.16e}",
                        roots[idx]
                    );
                    assert!(
                        weights[idx].is_finite() && weights[idx] >= -1.0e-10,
                        "invalid weight nroots={nroots}, t={t}, idx={idx}: got={:.16e}",
                        weights[idx]
                    );
                }
                for moment_idx in 0..=(2 * nroots - 1) {
                    let got = roots
                        .iter()
                        .zip(weights.iter())
                        .map(|(root, weight)| weight * root.powi(moment_idx as i32))
                        .sum::<f64>();
                    let expect = moments[moment_idx];
                    assert!(
                        (got - expect).abs() <= 5.0e-8 * expect.abs().max(1.0),
                        "generic moment mismatch nroots={nroots}, t={t}, moment={moment_idx}: got={got:.16e}, expect={expect:.16e}"
                    );
                }
            }
        }
    }

    #[test]
    fn rys_roots_weights_r2_reproduce_moments_through_six_roots() {
        for nroots in 1..=6 {
            for t in [0.0_f64, 1.0e-8, 0.2, 1.0384010260937269, 2.0, 20.0, 80.0] {
                let moments = boys_vec_r2(2 * nroots, t);
                let (roots, weights) = rys_roots_weights_r2(nroots, t);
                assert_eq!(roots.len(), nroots);
                assert_eq!(weights.len(), nroots);
                for idx in 0..nroots {
                    assert!(
                        roots[idx].is_finite()
                            && roots[idx] >= -1.0e-12
                            && roots[idx] <= 1.0 + 1.0e-12,
                        "invalid R2 root nroots={nroots}, t={t}, idx={idx}: got={:.16e}",
                        roots[idx]
                    );
                    assert!(
                        weights[idx].is_finite() && weights[idx] >= -1.0e-10,
                        "invalid R2 weight nroots={nroots}, t={t}, idx={idx}: got={:.16e}",
                        weights[idx]
                    );
                }
                for moment_idx in 0..=(2 * nroots - 1) {
                    let got = roots
                        .iter()
                        .zip(weights.iter())
                        .map(|(root, weight)| weight * root.powi(moment_idx as i32))
                        .sum::<f64>();
                    let expect = moments[moment_idx];
                    assert!(
                        (got - expect).abs() <= 5.0e-8 * expect.abs().max(1.0),
                        "R2 moment mismatch nroots={nroots}, t={t}, moment={moment_idx}: got={got:.16e}, expect={expect:.16e}"
                    );
                }
            }
        }
    }

    #[test]
    fn rys_roots_weights_r2_keeps_legacy_low_root_path() {
        for nroots in 1..=6 {
            for t in [0.0_f64, 1.0e-8, 0.2, 1.0384010260937269, 2.0, 20.0, 80.0] {
                let moments = boys_vec_r2(2 * nroots, t);
                let (roots_ref, weights_ref) =
                    rys_roots_weights_r2_from_moments(nroots, &moments, false)
                        .expect("legacy R2 roots/weights should be available");
                let (roots, weights) = rys_roots_weights_r2(nroots, t);
                assert_eq!(roots.len(), roots_ref.len());
                assert_eq!(weights.len(), weights_ref.len());
                for idx in 0..nroots {
                    assert!(
                        (roots[idx] - roots_ref[idx]).abs() <= 1.0e-13,
                        "R2 low-root baseline root changed nroots={nroots}, t={t}, idx={idx}: got={:.16e}, legacy={:.16e}",
                        roots[idx],
                        roots_ref[idx]
                    );
                    assert!(
                        (weights[idx] - weights_ref[idx]).abs() <= 1.0e-13,
                        "R2 low-root baseline weight changed nroots={nroots}, t={t}, idx={idx}: got={:.16e}, legacy={:.16e}",
                        weights[idx],
                        weights_ref[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn rys_roots_weights_r2_handles_high_roots_with_stable_moments() {
        for nroots in [7_usize, 9, 15, 25, 33, 49, 65] {
            for t in [0.0_f64, 1.0e-8, 0.2, 2.0, 20.0, 80.0] {
                let moments = r2_stable_moments_from_measure(2 * nroots, t)
                    .expect("failed to build stable R2 moments");
                let (roots, weights) = rys_roots_weights_r2(nroots, t);
                assert_eq!(roots.len(), nroots);
                assert_eq!(weights.len(), nroots);
                for idx in 0..nroots {
                    assert!(
                        roots[idx].is_finite()
                            && roots[idx] >= -1.0e-12
                            && roots[idx] <= 1.0 + 1.0e-12,
                        "invalid stable R2 root nroots={nroots}, t={t}, idx={idx}: got={:.16e}",
                        roots[idx]
                    );
                    assert!(
                        weights[idx].is_finite() && weights[idx] >= -1.0e-10,
                        "invalid stable R2 weight nroots={nroots}, t={t}, idx={idx}: got={:.16e}",
                        weights[idx]
                    );
                }
                for moment_idx in 0..=(2 * nroots - 1) {
                    let got = roots
                        .iter()
                        .zip(weights.iter())
                        .map(|(root, weight)| weight * root.powi(moment_idx as i32))
                        .sum::<f64>();
                    let expect = moments[moment_idx];
                    assert!(
                        (got - expect).abs() <= 5.0e-8 * expect.abs().max(1.0),
                        "stable R2 moment mismatch nroots={nroots}, t={t}, moment={moment_idx}: got={got:.16e}, expect={expect:.16e}"
                    );
                }
            }
        }
    }

    #[test]
    fn r2_stable_moments_match_legacy_gm_convention() {
        for mmax in [12_usize, 24, 50, 98, 130] {
            for t in [0.0_f64, 1.0e-8, 0.2, 2.0, 8.0] {
                let legacy = boys_vec_r2(mmax, t);
                let stable = r2_stable_moments_from_measure(mmax, t)
                    .expect("failed to build stable R2 moments");
                for m in 0..=mmax {
                    let expect = legacy[m];
                    let got = stable[m];
                    let tol = 5.0e-10 * expect.abs().max(1.0);
                    assert!(
                        (got - expect).abs() <= tol,
                        "stable R2 moment changed legacy convention mmax={mmax}, m={m}, t={t}: got={got:.16e}, legacy={expect:.16e}, tol={tol:.3e}"
                    );
                }
            }
        }
    }

    #[test]
    fn boys_f0_f1_matches_general_boys_slice() {
        for t in [
            0.0_f64, 1.0e-12, 1.0e-8, 1.0e-6, 1.0e-5, 0.2, 1.0, 2.0, 20.0, 80.0,
        ] {
            let mut reference = [0.0_f64; 2];
            boys_slice(1, t, &mut reference);
            let (f0, f1) = boys_f0_f1(t);
            assert!(
                (f0 - reference[0]).abs() <= 1.0e-12 * reference[0].abs().max(1.0),
                "F0 mismatch at t={t}: {f0} vs {}",
                reference[0]
            );
            assert!(
                (f1 - reference[1]).abs() <= 1.0e-10 * reference[1].abs().max(1.0),
                "F1 mismatch at t={t}: {f1} vs {}",
                reference[1]
            );
        }
    }

    #[test]
    fn boys_f0_matches_general_boys_slice() {
        for t in [
            0.0_f64, 1.0e-12, 1.0e-8, 1.0e-6, 1.0e-5, 0.2, 0.8, 1.0, 2.0, 20.0, 80.0,
        ] {
            let mut reference = [0.0_f64; 1];
            boys_slice(0, t, &mut reference);
            let f0 = boys_f0(t);
            assert!(
                (f0 - reference[0]).abs() <= 1.0e-13 * reference[0].abs().max(1.0),
                "F0 mismatch at t={t}: {f0} vs {}",
                reference[0]
            );
        }
    }

    #[test]
    fn exact4c_scalar_def2_tzvpp_pfff_matches_libcint() {
        let ctrl_path = write_temp_ctrl_h2o("def2-tzvpp");
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();

        let ao_shells =
            crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
                .expect("failed to build AO rint shells");
        let ao_bfs = crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&ao_shells)
            .expect("failed to expand AO rint shells");

        let indices = [41_usize, 26_usize, 26_usize, 26_usize];
        let got = eri_ao_4c_r(&ao_bfs, indices[0], indices[1], indices[2], indices[3]);
        let eri4_libcint = mol.int_ijkl_erifull();
        let expect = *eri4_libcint
            .get(&[indices[0], indices[1], indices[2], indices[3]])
            .unwrap();
        assert!(
            (got - expect).abs() <= 1.0e-10 * expect.abs().max(1.0),
            "def2-tzvpp p-f-f-f exact 4c mismatch at {:?}: got={got:.16e}, libcint={expect:.16e}",
            indices
        );

        let _ = fs::remove_file(ctrl_path);
    }

    #[test]
    fn exact4c_shell_block_def2_qzvp_gggg_matches_scalar() {
        let ctrl_path = write_temp_ctrl_h2o("def2-qzvp");
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();

        let ao_shells =
            crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
                .expect("failed to build AO rint shells");
        let ao_bfs = crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&ao_shells)
            .expect("failed to expand AO rint shells");

        let shell = ao_shells
            .iter()
            .find(|shell| shell.shell.ang_type == 4)
            .expect("def2-qzvp H2O should contain g shells");
        let block = int4c_r_shell_block(shell, shell, shell, shell);
        let got = block[(0, 0)];
        let ao = shell.ao_start;
        let expect = eri_ao_4c_r(&ao_bfs, ao, ao, ao, ao);
        assert!(
            got.abs() > 1.0e-8,
            "def2-qzvp g-g-g-g shell block unexpectedly vanished for ao={ao}: got={got:.16e}"
        );
        assert!(
            (got - expect).abs() <= 1.0e-10 * expect.abs().max(1.0),
            "def2-qzvp g-g-g-g shell block mismatch ao={ao}: got={got:.16e}, scalar={expect:.16e}"
        );

        let _ = fs::remove_file(ctrl_path);
    }

    #[test]
    fn exact4c_shell_block_single_component_l5_to_l8_matches_scalar() {
        for l in 5_u32..=8 {
            let component = [l, 0, 0];
            let shell = single_component_shell(l, component);
            let ao_bfs = vec![single_component_basis_function(component)];
            let block = int4c_r_shell_block(&shell, &shell, &shell, &shell);
            let got = block[(0, 0)];
            let expect = eri_ao_4c_r(&ao_bfs, 0, 0, 0, 0);
            assert!(
                (got - expect).abs() <= 1.0e-10 * expect.abs().max(1.0),
                "single-component exact 4c mismatch l={l}: got={got:.16e}, scalar={expect:.16e}"
            );
        }
    }

    #[test]
    fn r2_4c_shell_block_single_component_l5_to_l12_matches_scalar() {
        for l in 5_u32..=12 {
            let component = [l, 0, 0];
            let shell = single_component_shell(l, component);
            let ao_bfs = vec![single_component_basis_function(component)];
            let block = int4c_r2_shell_block(&shell, &shell, &shell, &shell);
            let got = block[(0, 0)];
            let expect = eri_ao_4c_r2(&ao_bfs, 0, 0, 0, 0);
            assert!(
                (got - expect).abs() <= 1.0e-10 * expect.abs().max(1.0),
                "single-component R2 exact 4c mismatch l={l}: got={got:.16e}, scalar={expect:.16e}"
            );
        }
    }

    #[test]
    #[ignore = "artificial R2 l=13..20 shell-block comparison is too slow for default debug tests"]
    fn r2_4c_shell_block_single_component_l13_to_l20_matches_scalar() {
        for l in 13_u32..=20 {
            let component = [l, 0, 0];
            let shell = single_component_shell(l, component);
            let ao_bfs = vec![single_component_basis_function(component)];
            let block = int4c_r2_shell_block(&shell, &shell, &shell, &shell);
            let got = block[(0, 0)];
            let expect = eri_ao_4c_r2(&ao_bfs, 0, 0, 0, 0);
            assert!(
                (got - expect).abs() <= 1.0e-10 * expect.abs().max(1.0),
                "single-component R2 exact 4c mismatch l={l}: got={got:.16e}, scalar={expect:.16e}"
            );
        }
    }

    #[test]
    #[ignore = "full h/i Cartesian shell-block comparison is too slow for debug default tests"]
    fn exact4c_shell_block_artificial_h_and_i_match_scalar_and_libcint() {
        for l in [5_u32, 6_u32] {
            assert_high_l_4c_shell_block_matches_scalar_and_libcint(l);
        }
    }

    #[test]
    fn rys_roots_weights_r_into_matches_allocating_api() {
        for nroots in 1..=9 {
            for t in [0.0_f64, 1.0e-8, 0.2, 2.0, 20.0, 80.0] {
                let (roots_ref, weights_ref) = rys_roots_weights_r(nroots, t);
                let mut roots = [0.0_f64; 16];
                let mut weights = [0.0_f64; 16];
                let got_nroots = rys_roots_weights_r_into(nroots, t, &mut roots, &mut weights);
                assert_eq!(got_nroots, nroots);
                for i in 0..nroots {
                    assert!(
                        (roots[i] - roots_ref[i]).abs() < 1.0e-11,
                        "root mismatch nroots={nroots}, t={t}, i={i}: {} vs {}",
                        roots[i],
                        roots_ref[i]
                    );
                    assert!(
                        (weights[i] - weights_ref[i]).abs() < 1.0e-11,
                        "weight mismatch nroots={nroots}, t={t}, i={i}: {} vs {}",
                        weights[i],
                        weights_ref[i]
                    );
                }
            }
        }
    }

    #[test]
    fn r_full_shell_blocks_match_ao_scalar_h2_sto3g() {
        let ctrl_path = write_temp_ctrl_h2("sto-3g");
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();

        let ao_shells =
            crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
                .expect("failed to build AO rint shells");
        let ao_bfs = crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&ao_shells)
            .expect("failed to expand AO rint shells");

        let full = int4c_r_full_from_shell_blocks(&ao_shells);
        let nao = ao_bfs.len();
        for sig in 0..nao {
            for lam in 0..nao {
                for nu in 0..nao {
                    for mu in 0..nao {
                        let got = full[int4c_r_full_index(mu, nu, lam, sig, nao)];
                        let expect = eri_ao_4c_r(&ao_bfs, mu, nu, lam, sig);
                        assert!(
                            (got - expect).abs() < 1.0e-12,
                            "full 1/r 4c mismatch at ({mu},{nu},{lam},{sig}): {got} vs {expect}"
                        );
                    }
                }
            }
        }

        let _ = fs::remove_file(ctrl_path);
    }

    #[test]
    fn r_full_shell_blocks_parallel_matches_serial_h2_sto3g() {
        let ctrl_path = write_temp_ctrl_h2("sto-3g");
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();

        let ao_shells =
            crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
                .expect("failed to build AO rint shells");

        let serial = int4c_r_full_from_shell_blocks(&ao_shells);
        let parallel = int4c_r_full_from_shell_blocks_parallel(&ao_shells);

        assert_eq!(parallel.len(), serial.len());
        for (idx, (got, expect)) in parallel.iter().zip(serial.iter()).enumerate() {
            assert!(
                (*got - *expect).abs() < 1.0e-12,
                "parallel full 1/r 4c mismatch at data[{idx}]: {got} vs {expect}"
            );
        }

        let _ = fs::remove_file(ctrl_path);
    }

    #[test]
    #[ignore = "temporary exact 4c libcint vs lib_rint release benchmark"]
    fn tmp_bench_exact4c_libcint_vs_librint() {
        use std::hint::black_box;

        let cases_env =
            std::env::var("TMP_EXACT4C_CASES").unwrap_or_else(|_| String::from("sto-3g"));
        let cases = cases_env
            .split(',')
            .map(str::trim)
            .filter(|case| !case.is_empty())
            .map(|case| (case.to_string(), write_temp_ctrl_h2o(case)))
            .collect::<Vec<_>>();
        let repeat = std::env::var("TMP_EXACT4C_REPEAT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1);
        let parallel = std::env::var("TMP_EXACT4C_PARALLEL")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mode = if parallel { "parallel" } else { "serial" };
        let profile_shells = std::env::var("TMP_EXACT4C_PROFILE_SHELLS")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let diagnose_mismatch = std::env::var("TMP_EXACT4C_DIAGNOSE_MISMATCH")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        for (basis_name, ctrl_path) in cases {
            let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
            let ao_shells = crate::lib_rint::basis::load_molecule_rint_shells_from_raw(
                &mol.geom,
                &mol.basis4elem,
            )
            .expect("failed to build AO rint shells");
            let nao = ao_shells.iter().map(|shell| shell.ao_len).sum::<usize>();
            let full_nint = nao * nao * nao * nao;

            if profile_shells {
                let shell_coeffs = build_rint_shell_coefficient_cache(&ao_shells);
                let primitive_pair_cache =
                    build_shell_pair_primitive_pair_cache(&ao_shells, &shell_coeffs);
                let mut entries = Vec::new();
                let mut block_data = Vec::new();
                let mut workspace_cache =
                    std::collections::HashMap::<[u32; 4], RysTransferWorkspace4c>::new();
                let mut shell_profile =
                    std::collections::HashMap::<[u32; 5], (usize, usize, usize, f64)>::new();
                for (a_idx, b_idx, c_idx, d_idx) in build_unique_4c_shell_quartet_tasks(&ao_shells)
                {
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
                    let t0 = Instant::now();
                    let shape = int4c_r_shell_block_batched_into_data_with_pairs_and_workspace(
                        a_shell,
                        b_shell,
                        c_shell,
                        d_shell,
                        ab_pairs,
                        cd_pairs,
                        &mut entries,
                        &mut block_data,
                        workspace,
                    );
                    let elapsed = t0.elapsed().as_secs_f64();
                    black_box(&block_data);
                    let nroots = ((a_shell.shell.ang_type
                        + b_shell.shell.ang_type
                        + c_shell.shell.ang_type
                        + d_shell.shell.ang_type)
                        / 2
                        + 1) as u32;
                    let key = [
                        a_shell.shell.ang_type,
                        b_shell.shell.ang_type,
                        c_shell.shell.ang_type,
                        d_shell.shell.ang_type,
                        nroots,
                    ];
                    let item = shell_profile.entry(key).or_insert((0, 0, 0, 0.0));
                    item.0 += 1;
                    item.1 += ab_pairs.len() * cd_pairs.len();
                    item.2 += shape[0] * shape[1];
                    item.3 += elapsed;
                }
                let mut rows = shell_profile.into_iter().collect::<Vec<_>>();
                rows.sort_by(|left, right| {
                    let left_seconds = left.1 .3;
                    let right_seconds = right.1 .3;
                    right_seconds
                        .partial_cmp(&left_seconds)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                for (rank, (key, (count, prim_quartets, block_values, seconds))) in
                    rows.into_iter().take(16).enumerate()
                {
                    println!(
                        "TMP_EXACT4C_SHELL_PROFILE basis={basis_name} rank={} la={} lb={} lc={} ld={} nroots={} shell_quartets={} primitive_quartets={} block_values={} seconds={:.6}",
                        rank + 1,
                        key[0],
                        key[1],
                        key[2],
                        key[3],
                        key[4],
                        count,
                        prim_quartets,
                        block_values,
                        seconds,
                    );
                }
            }

            let mut best_libcint = f64::INFINITY;
            let mut best_librint = f64::INFINITY;
            let mut libcint_checksum = 0.0_f64;
            let mut librint_checksum = 0.0_f64;
            let mut libcint_ref = None;
            let mut librint_full = vec![0.0_f64; full_nint];

            for _ in 0..repeat {
                let t0 = Instant::now();
                let eri4_libcint = mol.int_ijkl_erifull();
                let libcint_s = t0.elapsed().as_secs_f64();
                best_libcint = best_libcint.min(libcint_s);
                libcint_checksum = eri4_libcint
                    .data
                    .iter()
                    .enumerate()
                    .fold(0.0_f64, |acc, (idx, value)| {
                        acc + *value * ((idx % 17 + 1) as f64)
                    });
                black_box(libcint_checksum);
                libcint_ref = Some(eri4_libcint);

                let t0 = Instant::now();
                librint_full = if parallel {
                    int4c_r_full_from_shell_blocks_parallel(&ao_shells)
                } else {
                    int4c_r_full_from_shell_blocks(&ao_shells)
                };
                let librint_s = t0.elapsed().as_secs_f64();
                best_librint = best_librint.min(librint_s);
                librint_checksum = librint_full
                    .iter()
                    .enumerate()
                    .fold(0.0_f64, |acc, (idx, value)| {
                        acc + *value * ((idx % 17 + 1) as f64)
                    });
                black_box(librint_checksum);
            }

            let eri4_libcint = libcint_ref.expect("missing libcint reference");
            let mut max_abs = 0.0_f64;
            let mut max_rel = 0.0_f64;
            let mut ao_to_shell = vec![0_usize; nao];
            for (shell_idx, shell) in ao_shells.iter().enumerate() {
                for local in 0..shell.ao_len {
                    ao_to_shell[shell.ao_start + local] = shell_idx;
                }
            }
            let mut error_by_nroots =
                std::collections::BTreeMap::<usize, (usize, f64, f64, f64)>::new();
            let mut max_case = (
                0_usize, 0_usize, 0_usize, 0_usize, 0_usize, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64,
            );
            for sig in 0..nao {
                for lam in 0..nao {
                    for nu in 0..nao {
                        for mu in 0..nao {
                            let idx = int4c_r_full_index(mu, nu, lam, sig, nao);
                            let lib_value = librint_full[idx];
                            let ref_value = *eri4_libcint.get(&[mu, nu, lam, sig]).unwrap();
                            let abs = (lib_value - ref_value).abs();
                            let rel = abs / ref_value.abs().max(1.0e-12);
                            let a_shell = &ao_shells[ao_to_shell[mu]];
                            let b_shell = &ao_shells[ao_to_shell[nu]];
                            let c_shell = &ao_shells[ao_to_shell[lam]];
                            let d_shell = &ao_shells[ao_to_shell[sig]];
                            let nroots = ((a_shell.shell.ang_type
                                + b_shell.shell.ang_type
                                + c_shell.shell.ang_type
                                + d_shell.shell.ang_type)
                                / 2
                                + 1) as usize;
                            let item = error_by_nroots.entry(nroots).or_insert((0, 0.0, 0.0, 0.0));
                            item.0 += 1;
                            item.1 = item.1.max(abs);
                            item.2 = item.2.max(rel);
                            item.3 += abs * abs;
                            if abs > max_abs {
                                max_abs = abs;
                                max_case = (mu, nu, lam, sig, idx, lib_value, ref_value, abs, rel);
                            }
                            max_rel = max_rel.max(rel);
                        }
                    }
                }
            }
            for (nroots, (count, bucket_max_abs, bucket_max_rel, sum_abs_sq)) in &error_by_nroots {
                let rms_abs = (sum_abs_sq / (*count as f64)).sqrt();
                println!(
                    "TMP_EXACT4C_ERROR_BY_NROOTS basis={basis_name} nroots={nroots} count={count} max_abs={bucket_max_abs:.3e} max_rel_floor_1e-12={bucket_max_rel:.3e} rms_abs={rms_abs:.3e}"
                );
            }
            if diagnose_mismatch && max_abs > 1.0e-8 {
                let ao_bfs =
                    crate::lib_rint::basis::expand_rint_shells_to_basis_functions(&ao_shells)
                        .expect("failed to expand AO rint shells");
                let mut ao_to_shell_local = vec![(0_usize, 0_usize); nao];
                for (shell_idx, shell) in ao_shells.iter().enumerate() {
                    for local in 0..shell.ao_len {
                        ao_to_shell_local[shell.ao_start + local] = (shell_idx, local);
                    }
                }
                let (mu, nu, lam, sig, idx, lib_value, ref_value, abs, rel) = max_case;
                let (a_idx, i) = ao_to_shell_local[mu];
                let (b_idx, j) = ao_to_shell_local[nu];
                let (c_idx, k) = ao_to_shell_local[lam];
                let (d_idx, l) = ao_to_shell_local[sig];
                let scalar_value = eri_ao_4c_r(&ao_bfs, mu, nu, lam, sig);
                println!(
                    "TMP_EXACT4C_MISMATCH basis={basis_name} idx={idx} ao=({mu},{nu},{lam},{sig}) shells=({},{},{},{}) locals=({},{},{},{}) ang=({},{},{},{}) lib_rint_full={:.16e} lib_rint_scalar={:.16e} libcint={:.16e} scalar_minus_cint={:.3e} abs={:.3e} rel={:.3e}",
                    a_idx,
                    b_idx,
                    c_idx,
                    d_idx,
                    i,
                    j,
                    k,
                    l,
                    ao_shells[a_idx].shell.ang_type,
                    ao_shells[b_idx].shell.ang_type,
                    ao_shells[c_idx].shell.ang_type,
                    ao_shells[d_idx].shell.ang_type,
                    lib_value,
                    scalar_value,
                    ref_value,
                    scalar_value - ref_value,
                    abs,
                    rel,
                );

                let eval_direct = |mu0: usize, nu0: usize, lam0: usize, sig0: usize| -> f64 {
                    let (a0_idx, i0) = ao_to_shell_local[mu0];
                    let (b0_idx, j0) = ao_to_shell_local[nu0];
                    let (c0_idx, k0) = ao_to_shell_local[lam0];
                    let (d0_idx, l0) = ao_to_shell_local[sig0];
                    let a0 = &ao_shells[a0_idx];
                    let b0 = &ao_shells[b0_idx];
                    let c0 = &ao_shells[c0_idx];
                    let d0 = &ao_shells[d0_idx];
                    let block = int4c_r_shell_block(a0, b0, c0, d0);
                    let row = j0 * a0.ao_len + i0;
                    let col = l0 * c0.ao_len + k0;
                    block[(row, col)]
                };
                for (label, mu0, nu0, lam0, sig0) in [
                    ("ab_cd", mu, nu, lam, sig),
                    ("ba_cd", nu, mu, lam, sig),
                    ("ab_dc", mu, nu, sig, lam),
                    ("ba_dc", nu, mu, sig, lam),
                    ("cd_ab", lam, sig, mu, nu),
                    ("dc_ab", sig, lam, mu, nu),
                    ("cd_ba", lam, sig, nu, mu),
                    ("dc_ba", sig, lam, nu, mu),
                ] {
                    let direct = eval_direct(mu0, nu0, lam0, sig0);
                    let full = librint_full[int4c_r_full_index(mu0, nu0, lam0, sig0, nao)];
                    let cint = *eri4_libcint.get(&[mu0, nu0, lam0, sig0]).unwrap();
                    println!(
                        "TMP_EXACT4C_MISMATCH_PERM basis={basis_name} perm={label} ao=({},{},{},{}) direct={:.16e} full={:.16e} libcint={:.16e} direct_minus_cint={:.3e} full_minus_cint={:.3e}",
                        mu0,
                        nu0,
                        lam0,
                        sig0,
                        direct,
                        full,
                        cint,
                        direct - cint,
                        full - cint,
                    );
                }
            }
            println!(
                "TMP_EXACT4C_LIBCINT_BENCH basis={basis_name} mode={mode} nao={nao} nshell={} full_nint={full_nint} repeat={repeat} libcint_direct_s={:.6} lib_rint_shell_s={:.6} lib_rint_over_libcint={:.3} libcint_over_lib_rint={:.3} max_abs={:.3e} max_rel_floor_1e-12={:.3e} checksum_libcint={:.12e} checksum_librint={:.12e}",
                ao_shells.len(),
                best_libcint,
                best_librint,
                best_librint / best_libcint.max(1.0e-12),
                best_libcint / best_librint.max(1.0e-12),
                max_abs,
                max_rel,
                libcint_checksum,
                librint_checksum,
            );
            assert!(
                max_abs <= 1.0e-8,
                "exact 4c mismatch for {basis_name}: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}"
            );

            let _ = fs::remove_file(ctrl_path);
        }
    }

    #[test]
    fn r2_3c_libcint_pair_screening_matches_unscreened_h2o_sto3g() {
        let ctrl_path = write_temp_ctrl_h2o("sto-3g");
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();

        let ao_shells =
            crate::lib_rint::basis::load_molecule_rint_shells_from_raw(&mol.geom, &mol.basis4elem)
                .expect("failed to build AO rint shells");
        let auxbasis4elem = build_default_r2_etb_auxbasis(&mol.geom, &mol.basis4elem)
            .expect("default r2 ETB auxiliary basis should be available");
        let aux_shells =
            crate::lib_rint::basis::load_aux_rint_shells_from_raw(&mol.geom, &auxbasis4elem)
                .expect("failed to build auxiliary rint shells");

        let mut max_diff = 0.0_f64;
        for left_shell in ao_shells.iter() {
            for right_shell in ao_shells.iter() {
                let left_min = left_shell.ao_start;
                let right_max = right_shell.ao_start + right_shell.ao_len - 1;
                if left_min > right_max {
                    continue;
                }
                for aux_shell in aux_shells.iter() {
                    let screened = int3c_r2_shell_block(left_shell, right_shell, aux_shell);
                    let unscreened =
                        int3c_r2_shell_block_batched_unscreened(left_shell, right_shell, aux_shell);
                    max_diff = max_diff.max(max_abs_diff_matrix(&screened, &unscreened));
                }
            }
        }

        println!("H2O/STO-3G r2 3c libcint-pair-screening max_diff={max_diff:.3e}");
        assert!(
            max_diff < 1.0e-12,
            "libcint-like primitive pair screening drifted from unscreened 3c blocks: {max_diff}"
        );

        let _ = fs::remove_file(ctrl_path);
    }

    fn benchmark_water_cluster_ccpvdz_scf_vs_r2_semidirect_shell_blocks(nwater: usize, tag: &str) {
        let ctrl_path = write_temp_ctrl_water_cluster("cc-pVDZ", nwater);
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
        let mut scf_data = SCF::build(mol, &None);

        let t_scf = Instant::now();
        scf_without_build(&mut scf_data, &None);
        let scf_secs = t_scf.elapsed().as_secs_f64();

        let (ao_shells, _ao_bfs, p_cart) =
            crate::lib_rint::basis::load_cartesian_rhf_rint_shells_basis_and_density_shell_shared(
                &scf_data.mol.geom,
                &scf_data.mol.basis4elem,
                &scf_data.density_matrix[0],
                tag,
            );
        let auxbasis4elem =
            build_default_r2_etb_auxbasis(&scf_data.mol.geom, &scf_data.mol.basis4elem)
                .expect("default r2 ETB auxiliary basis should be available");
        let aux_shells = crate::lib_rint::basis::load_aux_rint_shells_from_raw(
            &scf_data.mol.geom,
            &auxbasis4elem,
        )
        .expect("failed to build auxiliary rint shells");
        let coeff_cart = [transform_mo_coeff_to_cartesian_shell_shared(
            &scf_data.mol.geom,
            &scf_data.mol.basis4elem,
            &scf_data.eigenvectors[0],
            rint_shell_basis_count(&ao_shells),
            tag,
            false,
        )];
        let occupation = [scf_data.occupation[0].clone()];
        let dm = vec![p_cart.clone()];

        let t_r2 = Instant::now();
        let (j_upper, k_upper) = r2_jk_direct_from_shell_blocks_with_auxbas_threshold(
            &ao_shells,
            &aux_shells,
            &dm,
            Some(&coeff_cart),
            Some(&occupation),
            AUXBAS_THRESHOLD,
        )
        .expect("failed to build semi-direct RI-r2 J/K");
        let r2_secs = t_r2.elapsed().as_secs_f64();

        let j = j_upper[0].to_matrixfull().unwrap();
        let k = k_upper[0].to_matrixfull().unwrap();
        let (_, _, r2_total) = ej_ek_from_p_jk_rhf(&p_cart, &j, &k);
        println!(
            "(H2O){nwater}/cc-pVDZ SCF vs semi-direct RI-r2: nbas={} naux={} scf={:.6}s r2_semidirect={:.6}s scf_over_r2={:.3}x r2_over_scf={:.3}x r2_total={:.16e}",
            rint_shell_basis_count(&ao_shells),
            rint_shell_basis_count(&aux_shells),
            scf_secs,
            r2_secs,
            scf_secs / r2_secs.max(1.0e-12),
            r2_secs / scf_secs.max(1.0e-12),
            r2_total
        );

        let _ = fs::remove_file(ctrl_path);
    }

    #[test]
    #[ignore = "benchmark only; runs a medium water cluster RI-r2 semi-direct timing case"]
    fn benchmark_water4_ccpvdz_scf_vs_r2_semidirect_shell_blocks() {
        benchmark_water_cluster_ccpvdz_scf_vs_r2_semidirect_shell_blocks(
            4,
            "benchmark_water4_ccpvdz_scf_vs_r2_semidirect_shell_blocks",
        );
    }

    #[test]
    #[ignore = "benchmark only; runs a larger water cluster RI-r2 semi-direct timing case"]
    fn benchmark_water8_ccpvdz_scf_vs_r2_semidirect_shell_blocks() {
        benchmark_water_cluster_ccpvdz_scf_vs_r2_semidirect_shell_blocks(
            8,
            "benchmark_water8_ccpvdz_scf_vs_r2_semidirect_shell_blocks",
        );
    }

    fn write_temp_ctrl_h2_with_etb(basis_dir: &str, etb_beta: f64) -> (String, String) {
        let pid = std::process::id();
        let safe_basis = basis_dir
            .replace('(', "")
            .replace(')', "")
            .replace('/', "_")
            .replace(' ', "_");
        let aux_dir = format!("/tmp/lib_rint_etb_aux_{safe_basis}_{pid}");
        let path = format!("/tmp/lib_rint_kernel_etb_check_{safe_basis}_{pid}.toml");
        let text = format!(
            r#"[ctrl]
print_level = 0
xc = "hf"
basis_path = "/home/cfh/rest_workspace/rest/basis-set-pool/{basis_dir}"
auxbas_path = "{aux_dir}"
basis_type = "Cartesian"
auxbas_type = "Cartesian"
use_auxbas = true
even_tempered_basis = true
etb_start_atom_number = 1
etb_beta = {etb_beta}
charge = 0.0
spin = 1.0
spin_polarization = false
num_threads = 1
run_lib_rint = false
[geom]
name = "H2"
unit = "angstrom"
position = [
    "H   0.0000000000   0.0000000000   0.0000000000",
    "H   0.0000000000   0.0000000000   1.4000000000",
]
"#
        );
        fs::write(&path, text).unwrap();
        (path, aux_dir)
    }
    fn write_temp_ctrl_nh3(basis_dir: &str) -> String {
        let pid = std::process::id();
        let path = format!("/tmp/lib_rint_kernel_nh3_{basis_dir}_{pid}.toml")
            .replace('(', "")
            .replace(')', "")
            .replace('/', "_")
            .replace(' ', "_");
        let text = format!(
            "[ctrl]\nprint_level = 0\nxc = \"hf\"\nbasis_path = \"/home/cfh/rest_workspace/rest/basis-set-pool/{basis_dir}\"\nbasis_type = \"Cartesian\"\nuse_auxbas = false\neven_tempered_basis = false\ncharge = 0.0\nspin = 1.0\nspin_polarization = false\nnum_threads = 1\nrun_lib_rint = false\n\n[geom]\nname = \"NH3\"\nunit = \"angstrom\"\nposition = [\n    \"N   0.0000000000   0.0000000000   0.0000000000\",\n    \"H   0.0000000000   1.5000000000   1.0000000000\",\n    \"H   1.4000000000   1.1000000000   0.0000000000\",\n    \"H   1.2000000000   0.0000000000   1.3000000000\",\n]\n"
        );
        fs::write(&path, text).unwrap();
        path
    }
    fn load_pyscf_autoaux_basis4elem(mol: &Molecule, basis_name: &str) -> Vec<Basis4Elem> {
        let atoms = crate::lib_rint::basis::parse_geomcell(&mol.geom)
            .expect("failed to parse geometry for PySCF autoaux bridge");
        let atom_spec = atoms
            .iter()
            .map(|(elem, center)| {
                format!(
                    "{} {:.16} {:.16} {:.16}",
                    elem, center[0], center[1], center[2]
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        let basis_spec = basis_name.to_lowercase();
        let script = r#"
import json
import sys
from pyscf import gto
from pyscf.df.autoaux import autoaux

atom_spec = sys.argv[1]
basis_spec = sys.argv[2]
mol = gto.M(atom=atom_spec, basis=basis_spec, cart=True)
aux = autoaux(mol)

payload = {}
for elem, shells in aux.items():
    out_shells = []
    for shell in shells:
        l = int(shell[0])
        exps = []
        coeffs = []
        for pair in shell[1:]:
            exps.append(float(pair[0]))
            coeffs.append(float(pair[1]))
        out_shells.append({
            "l": l,
            "exponents": exps,
            "coefficients": coeffs,
        })
    payload[elem] = out_shells
print(json.dumps(payload))
"#;
        let output = Command::new("python")
            .arg("-c")
            .arg(script)
            .arg(&atom_spec)
            .arg(&basis_spec)
            .output()
            .expect("failed to run python for PySCF autoaux");
        assert!(
            output.status.success(),
            "PySCF autoaux bridge failed: stdout={}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let payload: serde_json::Value = serde_json::from_slice(&output.stdout)
            .expect("failed to parse JSON from PySCF autoaux");

        atoms
            .iter()
            .enumerate()
            .map(|(atom_idx, (elem, _))| {
                let shells_json = payload
                    .get(elem)
                    .unwrap_or_else(|| panic!("PySCF autoaux output missing element {elem}"));
                let electron_shells = shells_json
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|shell| {
                        let l = shell["l"].as_i64().unwrap() as i32;
                        let exponents = shell["exponents"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_f64().unwrap())
                            .collect::<Vec<_>>();
                        let coeffs = shell["coefficients"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_f64().unwrap())
                            .collect::<Vec<_>>();
                        BasCell {
                            function_type: Some(String::from("gto")),
                            region: None,
                            angular_momentum: vec![l],
                            exponents,
                            coefficients: vec![coeffs.clone()],
                            native_coefficients: vec![coeffs],
                        }
                    })
                    .collect::<Vec<_>>();
                Basis4Elem {
                    electron_shells,
                    references: None,
                    ecp_potentials: None,
                    ecp_electrons: None,
                    global_index: (atom_idx, atom_idx),
                }
            })
            .collect()
    }
    fn compare_ri_r_with_fixed_density(
        ao_bfs: &[BasisFunction],
        p_cart: &MatrixFull<f64>,
        aux_bfs: &[BasisFunction],
    ) -> (f64, f64, f64, f64, f64, usize) {
        let dm_vec = vec![p_cart.clone()];
        let ri_r = prepare_rimatr_for_r_sync(ao_bfs, aux_bfs)
            .expect("failed to build RI-r matrix for fixed-density comparison");
        let j_ri_u = vj_upper_with_rimatr_r_sync(&Some(ri_r.clone()), &dm_vec, 1, 1.0);
        let k_ri_u = vk_upper_with_rimatr_r_sync(&Some(ri_r), &dm_vec, 1, 1.0);
        let j_ri = j_ri_u[0].to_matrixfull().unwrap();
        let k_ri = k_ri_u[0].to_matrixfull().unwrap();
        let (j_ex, k_ex) = build_jk_from_p_kernel(ao_bfs, p_cart, eri_ao_4c_r);
        let (ej_ri, ek_ri, e_ri) = ej_ek_from_p_jk_rhf(p_cart, &j_ri, &k_ri);
        let (ej_ex, ek_ex, e_ex) = ej_ek_from_p_jk_rhf(p_cart, &j_ex, &k_ex);

        (
            max_abs_diff_matrix(&j_ri, &j_ex),
            max_abs_diff_matrix(&k_ri, &k_ex),
            (ej_ri - ej_ex).abs(),
            (ek_ri - ek_ex).abs(),
            (e_ri - e_ex).abs(),
            aux_bfs.len(),
        )
    }
    fn reconstruct_eri_ao_4c_from_rimatr(
        rimatr: &MatrixFull<f64>,
        basbas2baspar: &MatrixFull<usize>,
        mu: usize,
        nu: usize,
        lam: usize,
        sig: usize,
    ) -> f64 {
        let munu = *basbas2baspar.get(&[mu, nu]).unwrap();
        let lamsig = *basbas2baspar.get(&[lam, sig]).unwrap();
        let naux = rimatr.size[1];
        let mut value = 0.0_f64;
        for aux in 0..naux {
            value += rimatr.get(&[munu, aux]).unwrap() * rimatr.get(&[lamsig, aux]).unwrap();
        }
        value
    }
    fn build_jk_from_reconstructed_rimatr(
        rimatr: &MatrixFull<f64>,
        basbas2baspar: &MatrixFull<usize>,
        p: &MatrixFull<f64>,
    ) -> (MatrixFull<f64>, MatrixFull<f64>) {
        let nao = p.size[0];
        let mut j = MatrixFull::new([nao, nao], 0.0_f64);
        let mut k = MatrixFull::new([nao, nao], 0.0_f64);
        for mu in 0..nao {
            for nu in 0..nao {
                let mut j_munu = 0.0_f64;
                let mut k_munu = 0.0_f64;
                for lam in 0..nao {
                    for sig in 0..nao {
                        let p_lamsig = *p.get(&[lam, sig]).unwrap();
                        let eri_j = reconstruct_eri_ao_4c_from_rimatr(
                            rimatr,
                            basbas2baspar,
                            mu,
                            nu,
                            lam,
                            sig,
                        );
                        let eri_k = reconstruct_eri_ao_4c_from_rimatr(
                            rimatr,
                            basbas2baspar,
                            mu,
                            lam,
                            nu,
                            sig,
                        );
                        j_munu += p_lamsig * eri_j;
                        k_munu += p_lamsig * eri_k;
                    }
                }
                j.set2d([mu, nu], j_munu);
                k.set2d([mu, nu], k_munu);
            }
        }
        (j, k)
    }
    fn write_temp_ctrl_h2o_sto3g_cartesian() -> String {
        let pid = std::process::id();
        let path = format!("/tmp/lib_rint_grid_h2o_sto3g_cart_{pid}.toml");
        let text = r#"[ctrl]
print_level = 0
xc = "hf"
basis_path = "/home/cfh/rest_workspace/rest/basis-set-pool/sto-3g"
basis_type = "Cartesian"
use_auxbas = false
even_tempered_basis = false
charge = 0.0
spin = 1.0
spin_polarization = false
num_threads = 1
run_lib_rint = false

[geom]
name = "H2O"
unit = "angstrom"
position = [
    "O   0.0000000000   0.0000000000   0.0000000000",
    "H   0.0000000000  -0.7571600000   0.5862600000",
    "H   0.0000000000   0.7571600000   0.5862600000",
]
"#;
        fs::write(&path, text).unwrap();
        path
    }
    fn write_temp_ctrl_li_open_shell_with_aux(basis_dir: &str, aux_basis_dir: &str) -> String {
        let pid = std::process::id();
        let path =
            format!("/tmp/lib_rint_open_shell_ri_check_{basis_dir}_{aux_basis_dir}_{pid}.toml")
                .replace('(', "")
                .replace(')', "")
                .replace('/', "_")
                .replace(' ', "_");
        let text = format!(
            r#"[ctrl]
print_level = 0
xc = "hf"
eri_type = "ri_v"
basis_path = "/home/cfh/rest_workspace/rest/basis-set-pool/{basis_dir}"
auxbas_path = "/home/cfh/rest_workspace/rest/basis-set-pool/{aux_basis_dir}"
basis_type = "spheric"
auxbas_type = "spheric"
use_auxbas = true
even_tempered_basis = false
charge = 0.0
spin = 2.0
spin_polarization = true
initial_guess = "sad"
mixer = "diis"
num_max_diis = 8
start_diis_cycle = 1
mix_param = 0.6
max_scf_cycle = 80
scf_acc_rho = 1.0e-8
scf_acc_eev = 1.0e-8
scf_acc_etot = 1.0e-10
num_threads = 1
run_lib_rint = false
[geom]
name = "Li"
unit = "angstrom"
position = [
    "Li   0.0000000000   0.0000000000   0.0000000000",
]
"#
        );
        fs::write(&path, text).unwrap();
        path
    }
    fn generate_pyscf_ref_4c(basis_name: &str, out4: &str) {
        let script = r#"
import sys
from pyscf import gto
basis, out4 = sys.argv[1:3]
mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis=basis, unit='Angstrom', cart=True)
eri4 = mol.intor('int2e_cart', aosym='s1')
with open(out4, 'w', encoding='utf-8') as fh:
    for i in range(eri4.shape[0]):
        for j in range(eri4.shape[1]):
            for k in range(eri4.shape[2]):
                for l in range(eri4.shape[3]):
                    fh.write(f'{i} {j} {k} {l} {eri4[i,j,k,l]:.15e}\n')
"#;
        let status = Command::new("python3")
            .args(["-c", script, basis_name, out4])
            .status()
            .expect("failed to launch python3");
        assert!(
            status.success(),
            "PySCF 4c reference generation failed for {basis_name}"
        );
    }
    fn generate_pyscf_ref_2c_3c_4c(basis_name: &str, out2: &str, out3: &str, out4: &str) {
        let script = r#"
import sys
from pyscf import gto, df
basis, out2, out3, out4 = sys.argv[1:5]
atom = 'H 0 0 0; H 0 0 1.4'
mol = gto.M(atom=atom, basis=basis, unit='Angstrom', cart=True)
auxmol = gto.M(atom=atom, basis=basis, unit='Angstrom', cart=True)
eri2 = auxmol.intor('int2c2e_cart')
eri3 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_cart', aosym='s1')
eri4 = mol.intor('int2e_cart', aosym='s1')
with open(out2, 'w', encoding='utf-8') as fh:
    for i in range(eri2.shape[0]):
        for j in range(eri2.shape[1]):
            fh.write(f'{i} {j} {eri2[i,j]:.15e}\n')
with open(out3, 'w', encoding='utf-8') as fh:
    for i in range(eri3.shape[0]):
        for j in range(eri3.shape[1]):
            for k in range(eri3.shape[2]):
                fh.write(f'{i} {j} {k} {eri3[i,j,k]:.15e}\n')
with open(out4, 'w', encoding='utf-8') as fh:
    for i in range(eri4.shape[0]):
        for j in range(eri4.shape[1]):
            for k in range(eri4.shape[2]):
                for l in range(eri4.shape[3]):
                    fh.write(f'{i} {j} {k} {l} {eri4[i,j,k,l]:.15e}\n')
"#;
        let status = Command::new("python3")
            .args(["-c", script, basis_name, out2, out3, out4])
            .status()
            .expect("failed to launch python3");
        assert!(
            status.success(),
            "PySCF 2c/3c/4c reference generation failed for {basis_name}"
        );
    }
    fn read_ref_values(path: &str, expected: usize) -> Vec<f64> {
        let text = fs::read_to_string(path).unwrap();
        let mut out = Vec::with_capacity(expected);
        for line in text.lines() {
            let value = line
                .split_whitespace()
                .last()
                .unwrap()
                .parse::<f64>()
                .unwrap();
            out.push(value);
        }
        assert_eq!(
            out.len(),
            expected,
            "unexpected reference length for {path}"
        );
        out
    }
    #[derive(Clone, Copy, Debug)]
    struct DiffStats {
        count: usize,
        bad: usize,
        max_abs: f64,
        max_rel: f64,
        sum_abs: f64,
        sum_rel: f64,
    }
    impl DiffStats {
        fn new() -> Self {
            Self {
                count: 0,
                bad: 0,
                max_abs: 0.0,
                max_rel: 0.0,
                sum_abs: 0.0,
                sum_rel: 0.0,
            }
        }
        fn push(&mut self, got: f64, reference: f64, tol_abs: f64, tol_rel: f64) {
            let abs = (got - reference).abs();
            let rel = abs / reference.abs().max(1.0e-300);
            self.count += 1;
            self.sum_abs += abs;
            self.sum_rel += rel;
            self.max_abs = self.max_abs.max(abs);
            self.max_rel = self.max_rel.max(rel);
            if abs > tol_abs && rel > tol_rel {
                self.bad += 1;
            }
        }
        fn summary(&self) -> String {
            format!(
                "count={} bad_count={} max_abs={:.12e} avg_abs={:.12e} max_rel={:.12e} avg_rel={:.12e}",
                self.count,
                self.bad,
                self.max_abs,
                if self.count > 0 {
                    self.sum_abs / self.count as f64
                } else {
                    0.0
                },
                self.max_rel,
                if self.count > 0 {
                    self.sum_rel / self.count as f64
                } else {
                    0.0
                },
            )
        }
    }
    fn max_abs_diff_matrix(a: &MatrixFull<f64>, b: &MatrixFull<f64>) -> f64 {
        assert_eq!(a.size, b.size);
        a.data
            .iter()
            .zip(b.data.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum GridKernel {
        R,
        R2,
    }
    #[derive(Clone, Debug)]
    struct GridQuartetRef {
        label: String,
        kernel: GridKernel,
        ijkl: [usize; 4],
        value: f64,
    }
    fn read_grid_quartet_refs(path: &str) -> Vec<GridQuartetRef> {
        let text = fs::read_to_string(path).unwrap();
        let mut rows = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 7 {
                continue;
            }
            let kernel = match fields[1] {
                "1/r12" | "K=1/r12" => GridKernel::R,
                "1/r12²" | "1/r12^2" | "K=1/r12^2" => GridKernel::R2,
                _ => continue,
            };
            let n = fields.len();
            rows.push(GridQuartetRef {
                label: fields[0].to_string(),
                kernel,
                ijkl: [
                    fields[n - 5].parse().unwrap(),
                    fields[n - 4].parse().unwrap(),
                    fields[n - 3].parse().unwrap(),
                    fields[n - 2].parse().unwrap(),
                ],
                value: fields[n - 1].parse().unwrap(),
            });
        }
        rows
    }

    #[test]
    fn compare_li_open_shell_ri_r_with_rest_standard_path() {
        let basis_name = "def2-TZVP";
        let aux_name = "def2-TZVP-RIFIT";
        let ctrl_path = write_temp_ctrl_li_open_shell_with_aux(basis_name, aux_name);
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
        let mut scf_data = crate::scf_io::SCF::build(mol, &None);
        scf_without_build(&mut scf_data, &None);

        assert!(
            scf_data.density_matrix.len() >= 2,
            "open-shell SCF should provide alpha and beta density matrices"
        );

        let dm_spin = scf_data.density_matrix[0..2].to_vec();
        let lib_obs = lib_vee_uhf_r_observables_with_auxbasis(
            &scf_data.mol.geom,
            &scf_data.mol.basis4elem,
            &scf_data.mol.auxbas4elem,
            &dm_spin,
        );

        let ri_rest = Some(scf_data.mol.prepare_rimatr_for_ri_v_rayon(None));
        let j_rest_u = vj_upper_with_rimatr_sync(&ri_rest, &dm_spin, 2, 1.0);
        let k_rest_u = vk_upper_with_rimatr_use_dm_only_sync_v02(&ri_rest, &dm_spin, 2, 1.0);

        let dm_a_upper = dm_spin[0].to_matrixupper();
        let dm_b_upper = dm_spin[1].to_matrixupper();
        let mut j_total_upper = j_rest_u[0].clone();
        j_total_upper
            .data
            .iter_mut()
            .zip(j_rest_u[1].data.iter())
            .for_each(|(to, from)| *to += *from);

        let ej_rest = 0.5
            * (SCF::par_energy_contraction(&dm_a_upper, &j_total_upper)
                + SCF::par_energy_contraction(&dm_b_upper, &j_total_upper));
        let ek_rest = -0.5
            * (SCF::par_energy_contraction(&dm_a_upper, &k_rest_u[0])
                + SCF::par_energy_contraction(&dm_b_upper, &k_rest_u[1]));
        let e_rest = ej_rest + ek_rest;

        let d_ej = (lib_obs.ej - ej_rest).abs();
        let d_ek = (lib_obs.ek - ek_rest).abs();
        let d_e = (lib_obs.total - e_rest).abs();

        println!(
            "Li open-shell {basis_name} + {aux_name}: lib_rint(EJ, EK, E) = ({:.16e}, {:.16e}, {:.16e})",
            lib_obs.ej, lib_obs.ek, lib_obs.total
        );
        println!(
            "Li open-shell {basis_name} + {aux_name}: rest_std(EJ, EK, E) = ({:.16e}, {:.16e}, {:.16e})",
            ej_rest, ek_rest, e_rest
        );
        println!(
            "Li open-shell {basis_name} + {aux_name}: |dEJ| = {:.3e}, |dEK| = {:.3e}, |dE| = {:.3e}",
            d_ej, d_ek, d_e
        );

        assert!(
            d_ej < 2.0e-5 && d_ek < 1.0e-5 && d_e < 1.0e-5,
            "open-shell lib_rint r-energy mismatch vs REST standard path: dEJ={d_ej:.3e}, dEK={d_ek:.3e}, dE={d_e:.3e}"
        );

        let _ = fs::remove_file(ctrl_path);
    }

    #[test]
    fn diagnose_h2_rest_internal_ri_r_vs_exact_with_rifit_and_converged_rhf_density_shell_shared() {
        let cases = [
            ("cc-pVDZ", "cc-pVDZ-RIFIT"),
            ("def2-TZVP", "def2-TZVP-RIFIT"),
        ];

        for (basis_name, aux_name) in cases {
            let ctrl_path = write_temp_ctrl_h2_with_aux(basis_name, aux_name);
            let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
            let mut scf_data = SCF::build(mol, &None);
            scf_without_build(&mut scf_data, &None);

            let (ao_bfs, p_cart) =
                crate::lib_rint::basis::load_cartesian_rhf_basis_and_density_shell_shared(
                    &scf_data.mol.geom,
                    &scf_data.mol.basis4elem,
                    &scf_data.density_matrix[0],
                    "diagnose_h2_rest_internal_ri_r_vs_exact_with_rifit_and_converged_rhf_density_shell_shared",
                );

            let dm_vec_rest = vec![scf_data.density_matrix[0].clone()];
            let ri_rest = Some(scf_data.mol.prepare_rimatr_for_ri_v_rayon(None));
            let j_rest_u = vj_upper_with_rimatr_sync(&ri_rest, &dm_vec_rest, 1, 1.0);
            let k_rest_u =
                vk_upper_with_rimatr_use_dm_only_sync_v02(&ri_rest, &dm_vec_rest, 1, 1.0);

            let mut j_total_u = j_rest_u[0].clone();
            j_total_u
                .data
                .iter_mut()
                .zip(j_rest_u[1].data.iter())
                .for_each(|(to, from)| *to += *from);
            let j_rest = j_total_u.to_matrixfull().unwrap();
            let k_rest = k_rest_u[0].to_matrixfull().unwrap();

            let (j_ex, k_ex) = build_jk_from_p_kernel(&ao_bfs, &p_cart, eri_ao_4c_r);
            let (ej_rest, ek_rest, e_rest) = ej_ek_from_p_jk_rhf(&p_cart, &j_rest, &k_rest);
            let (ej_ex, ek_ex, e_ex) = ej_ek_from_p_jk_rhf(&p_cart, &j_ex, &k_ex);

            let d_j = max_abs_diff_matrix(&j_rest, &j_ex);
            let d_k = max_abs_diff_matrix(&k_rest, &k_ex);
            let d_ej = (ej_rest - ej_ex).abs();
            let d_ek = (ek_rest - ek_ex).abs();
            let d_e = (e_rest - e_ex).abs();

            println!(
                "{basis_name} + {aux_name} converged RHF density (SCF energy = {:.16e}) REST-internal-RI-r vs exact-r: dJ={:.12e} dK={:.12e} dEj={:.12e} dEk={:.12e} dE={:.12e}",
                scf_data.scf_energy, d_j, d_k, d_ej, d_ek, d_e
            );

            let _ = fs::remove_file(ctrl_path);
        }
    }

    #[test]
    fn diagnose_h2_ri_r_vs_exact_with_rifit_and_converged_rhf_density_shell_shared() {
        let cases = [
            ("cc-pVDZ", "cc-pVDZ-RIFIT"),
            ("def2-TZVP", "def2-TZVP-RIFIT"),
        ];

        for (basis_name, aux_name) in cases {
            let ctrl_path = write_temp_ctrl_h2_with_aux(basis_name, aux_name);
            let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
            let mut scf_data = SCF::build(mol, &None);
            scf_without_build(&mut scf_data, &None);

            let (ao_bfs, p_cart) =
                crate::lib_rint::basis::load_cartesian_rhf_basis_and_density_shell_shared(
                    &scf_data.mol.geom,
                    &scf_data.mol.basis4elem,
                    &scf_data.density_matrix[0],
                    "diagnose_h2_ri_r_vs_exact_with_rifit_and_converged_rhf_density_shell_shared",
                );
            let aux_bfs = crate::lib_rint::basis::load_aux_molecule_shell_shared_from_raw(
                &scf_data.mol.geom,
                &scf_data.mol.auxbas4elem,
            )
            .expect("failed to load auxiliary basis (shell_shared_from_raw)");

            let (d_j, d_k, d_ej, d_ek, d_e, naux) =
                compare_ri_r_with_fixed_density(&ao_bfs, &p_cart, &aux_bfs);

            println!(
                "{basis_name} + {aux_name} converged RHF density (SCF energy = {:.16e}) RI-r vs exact-r: naux={} dJ={:.12e} dK={:.12e} dEj={:.12e} dEk={:.12e} dE={:.12e}",
                scf_data.scf_energy, naux, d_j, d_k, d_ej, d_ek, d_e
            );

            let _ = fs::remove_file(ctrl_path);
        }
    }
    #[test]
    fn diagnose_h2_ri_r_with_converged_rhf_density_across_aux_generators() {
        let cases = [
            ("cc-pVDZ", "cc-pVDZ-RIFIT"),
            ("def2-TZVP", "def2-TZVP-RIFIT"),
        ];

        for (basis_name, aux_name) in cases {
            let ctrl_path = write_temp_ctrl_h2(basis_name);
            let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
            let mut scf_data = SCF::build(mol, &None);
            scf_without_build(&mut scf_data, &None);

            let dm = scf_data.density_matrix[0].clone();
            let (ao_bfs, p_cart) =
                crate::lib_rint::basis::load_cartesian_rhf_basis_and_density_shell_shared(
                    &scf_data.mol.geom,
                    &scf_data.mol.basis4elem,
                    &dm,
                    "diagnose_h2_ri_r_with_converged_rhf_density_across_aux_generators",
                );

            let ctrl_rifit = write_temp_ctrl_h2_with_aux(basis_name, aux_name);
            let mol_rifit = Molecule::build(ctrl_rifit.clone(), None).unwrap();
            let aux_bfs_rifit = crate::lib_rint::basis::load_aux_molecule_shell_shared_from_raw(
                &scf_data.mol.geom,
                &mol_rifit.auxbas4elem,
            )
            .expect("failed to expand RIFIT aux basis");
            let (d_j_rifit, d_k_rifit, d_ej_rifit, d_ek_rifit, d_e_rifit, naux_rifit) =
                compare_ri_r_with_fixed_density(&ao_bfs, &p_cart, &aux_bfs_rifit);

            let (ctrl_etb, aux_dir_etb) = write_temp_ctrl_h2_with_etb(basis_name, 1.3);
            let mol_etb = Molecule::build(ctrl_etb.clone(), None).unwrap();
            let aux_bfs_etb = crate::lib_rint::basis::load_aux_molecule_shell_shared_from_raw(
                &scf_data.mol.geom,
                &mol_etb.auxbas4elem,
            )
            .expect("failed to expand ETB aux basis");
            let (d_j_etb, d_k_etb, d_ej_etb, d_ek_etb, d_e_etb, naux_etb) =
                compare_ri_r_with_fixed_density(&ao_bfs, &p_cart, &aux_bfs_etb);

            let autoaux_basis4elem = load_pyscf_autoaux_basis4elem(&scf_data.mol, basis_name);
            let aux_bfs_autoaux = crate::lib_rint::basis::load_aux_molecule_shell_shared_from_raw(
                &scf_data.mol.geom,
                &autoaux_basis4elem,
            )
            .expect("failed to expand PySCF autoaux basis");
            let (d_j_autoaux, d_k_autoaux, d_ej_autoaux, d_ek_autoaux, d_e_autoaux, naux_autoaux) =
                compare_ri_r_with_fixed_density(&ao_bfs, &p_cart, &aux_bfs_autoaux);

            println!(
                "{basis_name} converged RHF density (exact SCF energy = {:.16e})",
                scf_data.scf_energy
            );
            println!(
                "  RIFIT        naux={} dJ={:.12e} dK={:.12e} dEj={:.12e} dEk={:.12e} dE={:.12e}",
                naux_rifit, d_j_rifit, d_k_rifit, d_ej_rifit, d_ek_rifit, d_e_rifit
            );
            println!(
                "  ETB(beta=1.3) naux={} dJ={:.12e} dK={:.12e} dEj={:.12e} dEk={:.12e} dE={:.12e}",
                naux_etb, d_j_etb, d_k_etb, d_ej_etb, d_ek_etb, d_e_etb
            );
            println!(
                "  PySCF autoaux naux={} dJ={:.12e} dK={:.12e} dEj={:.12e} dEk={:.12e} dE={:.12e}",
                naux_autoaux, d_j_autoaux, d_k_autoaux, d_ej_autoaux, d_ek_autoaux, d_e_autoaux
            );

            let _ = fs::remove_file(ctrl_path);
            let _ = fs::remove_file(ctrl_rifit);
            let _ = fs::remove_file(ctrl_etb);
            let _ = fs::remove_dir_all(aux_dir_etb);
        }
    }
    #[test]
    fn diagnose_h2_ri_r_with_converged_rhf_density_with_etb_shell_shared() {
        let basis_cases = ["cc-pVDZ", "def2-TZVP"];
        let beta_cases = [2.0_f64, 1.7_f64, 1.5_f64, 1.3_f64];

        for basis_name in basis_cases {
            let ctrl_path = write_temp_ctrl_h2(basis_name);
            let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
            let mut scf_data = SCF::build(mol, &None);
            scf_without_build(&mut scf_data, &None);

            let dm = scf_data.density_matrix[0].clone();
            let (ao_bfs, p_cart) =
                crate::lib_rint::basis::load_cartesian_rhf_basis_and_density_shell_shared(
                    &scf_data.mol.geom,
                    &scf_data.mol.basis4elem,
                    &dm,
                    "diagnose_h2_ri_r_with_converged_rhf_density_with_etb_shell_shared",
                );

            for etb_beta in beta_cases {
                let (ctrl_path_etb, aux_dir) = write_temp_ctrl_h2_with_etb(basis_name, etb_beta);
                let mol_etb = Molecule::build(ctrl_path_etb.clone(), None).unwrap();
                let aux_bfs = crate::lib_rint::basis::load_aux_molecule_shell_shared_from_raw(
                    &scf_data.mol.geom,
                    &mol_etb.auxbas4elem,
                )
                .expect("failed to load ETB auxiliary basis (shell_shared_from_raw)");

                assert_eq!(ao_bfs.len(), p_cart.size[0]);
                assert_eq!(ao_bfs.len(), p_cart.size[1]);
                assert_eq!(aux_bfs.len(), mol_etb.num_auxbas);
                assert!(
                    !aux_bfs.is_empty(),
                    "ETB auxiliary basis should not be empty for {basis_name}"
                );

                let (d_j, d_k, d_ej, d_ek, d_e, naux) =
                    compare_ri_r_with_fixed_density(&ao_bfs, &p_cart, &aux_bfs);

                println!(
                    "{basis_name} + ETB(beta={etb_beta:.3}) converged RHF density (exact SCF energy = {:.16e}): naux={} dJ={:.12e} dK={:.12e} dEj={:.12e} dEk={:.12e} dE={:.12e}",
                    scf_data.scf_energy, naux, d_j, d_k, d_ej, d_ek, d_e
                );

                let _ = fs::remove_file(ctrl_path_etb);
                let _ = fs::remove_dir_all(aux_dir);
            }

            let _ = fs::remove_file(ctrl_path);
        }
    }
    #[test]
    fn default_r2_path_uses_etb_beta_1p7_when_auxbasis_is_missing() {
        let basis_name = "cc-pVDZ";
        let ctrl_path = write_temp_ctrl_h2(basis_name);
        let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
        let mut scf_data = SCF::build(mol, &None);
        scf_without_build(&mut scf_data, &None);

        let default_obs = lib_vee_rhf_r2_observables(
            &scf_data.mol.geom,
            &scf_data.mol.basis4elem,
            &scf_data.density_matrix[0],
        );

        let auxbasis4elem =
            build_default_r2_etb_auxbasis(&scf_data.mol.geom, &scf_data.mol.basis4elem)
                .expect("default r2 ETB auxiliary basis should be available");
        let explicit_obs = lib_vee_rhf_r2_observables_with_auxbasis(
            &scf_data.mol.geom,
            &scf_data.mol.basis4elem,
            &auxbasis4elem,
            &scf_data.density_matrix[0],
        );

        assert!((default_obs.ej - explicit_obs.ej).abs() < 1.0e-12);
        assert!((default_obs.ek - explicit_obs.ek).abs() < 1.0e-12);
        assert!((default_obs.total - explicit_obs.total).abs() < 1.0e-12);

        let _ = fs::remove_file(ctrl_path);
    }
    #[test]
    fn diagnose_h2_ri_r2_with_rifit_and_etb_auxbasis() {
        let cases = [
            ("cc-pVDZ", "cc-pVDZ-RIFIT"),
            ("def2-TZVP", "def2-TZVP-RIFIT"),
            ("def2-QZVPP", "def2-QZVPP-RIFIT"),
        ];

        for (basis_name, aux_name) in cases {
            let ctrl_path = write_temp_ctrl_h2(basis_name);
            let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
            let mut scf_data = SCF::build(mol, &None);
            scf_without_build(&mut scf_data, &None);

            let ctrl_rifit = write_temp_ctrl_h2_with_aux(basis_name, aux_name);
            let mol_rifit = Molecule::build(ctrl_rifit.clone(), None).unwrap();
            let rifit_obs = lib_vee_rhf_r2_observables_with_auxbasis(
                &scf_data.mol.geom,
                &scf_data.mol.basis4elem,
                &mol_rifit.auxbas4elem,
                &scf_data.density_matrix[0],
            );

            println!(
                "{basis_name} RI-r2 RIFIT: naux={} EJ={:.16e} EK={:.16e} E={:.16e}",
                mol_rifit.num_auxbas, rifit_obs.ej, rifit_obs.ek, rifit_obs.total
            );

            for etb_beta in [1.3_f64, 1.7_f64] {
                let (ctrl_etb, aux_dir_etb) = write_temp_ctrl_h2_with_etb(basis_name, etb_beta);
                let mol_etb = Molecule::build(ctrl_etb.clone(), None).unwrap();
                let etb_obs = lib_vee_rhf_r2_observables_with_auxbasis(
                    &scf_data.mol.geom,
                    &scf_data.mol.basis4elem,
                    &mol_etb.auxbas4elem,
                    &scf_data.density_matrix[0],
                );
                println!(
                    "  ETB(beta={etb_beta:.1}) naux={} EJ={:.16e} EK={:.16e} E={:.16e} |vs RIFIT| dEJ={:.12e} dEK={:.12e} dE={:.12e}",
                    mol_etb.num_auxbas,
                    etb_obs.ej,
                    etb_obs.ek,
                    etb_obs.total,
                    (etb_obs.ej - rifit_obs.ej).abs(),
                    (etb_obs.ek - rifit_obs.ek).abs(),
                    (etb_obs.total - rifit_obs.total).abs()
                );
                let _ = fs::remove_file(ctrl_etb);
                let _ = fs::remove_dir_all(aux_dir_etb);
            }

            let _ = fs::remove_file(ctrl_path);
            let _ = fs::remove_file(ctrl_rifit);
        }
    }
    #[test]
    fn r2_optimized_contractions_match_reference_shell_shared() {
        let basis_cases = ["cc-pVDZ", "def2-TZVP"];

        for basis_name in basis_cases {
            let ctrl_path = write_temp_ctrl_h2(basis_name);
            let mol = Molecule::build(ctrl_path.clone(), None).unwrap();
            let mut scf_data = SCF::build(mol, &None);
            scf_without_build(&mut scf_data, &None);

            let dm = scf_data.density_matrix[0].clone();
            let (ao_shells, _ao_bfs, p_cart) =
                crate::lib_rint::basis::load_cartesian_rhf_rint_shells_basis_and_density_shell_shared(
                    &scf_data.mol.geom,
                    &scf_data.mol.basis4elem,
                    &dm,
                    "r2_optimized_contractions_match_reference_shell_shared",
                );
            let auxbasis4elem =
                build_default_r2_etb_auxbasis(&scf_data.mol.geom, &scf_data.mol.basis4elem)
                    .expect("default r2 ETB auxiliary basis should be available");
            let aux_shells = crate::lib_rint::basis::load_aux_rint_shells_from_raw(
                &scf_data.mol.geom,
                &auxbasis4elem,
            )
            .expect("failed to build default ETB auxiliary rint shells");
            let ri_r2 = prepare_rimatr_for_r2_shell_blocks_sync(&ao_shells, &aux_shells)
                .expect("failed to build RI-r2 matrix for contraction regression");
            let dm_vec = vec![p_cart.clone()];

            let j_ref_u = vj_upper_with_rimatr_sync_internal(&Some(ri_r2.clone()), &dm_vec, 1, 1.0);
            let k_ref_u = vk_upper_with_rimatr_sync_internal(&Some(ri_r2.clone()), &dm_vec, 1, 1.0);
            let j_opt_u = vj_upper_with_rimatr_r2_sync(&Some(ri_r2.clone()), &dm_vec, 1, 1.0);
            let k_opt_u = vk_upper_with_rimatr_r2_sync(&Some(ri_r2), &dm_vec, 1, 1.0);

            let j_ref = j_ref_u[0].to_matrixfull().unwrap();
            let k_ref = k_ref_u[0].to_matrixfull().unwrap();
            let j_opt = j_opt_u[0].to_matrixfull().unwrap();
            let k_opt = k_opt_u[0].to_matrixfull().unwrap();

            let (ej_ref, ek_ref, e_ref) = ej_ek_from_p_jk_rhf(&p_cart, &j_ref, &k_ref);
            let (ej_opt, ek_opt, e_opt) = ej_ek_from_p_jk_rhf(&p_cart, &j_opt, &k_opt);

            assert!(
                max_abs_diff_matrix(&j_ref, &j_opt) < 1.0e-10,
                "{basis_name} optimized r2 J contraction drifted from reference"
            );
            assert!(
                max_abs_diff_matrix(&k_ref, &k_opt) < 1.0e-10,
                "{basis_name} optimized r2 K contraction drifted from reference"
            );
            assert!(
                (ej_ref - ej_opt).abs() < 1.0e-10,
                "{basis_name} optimized r2 E_J drifted from reference"
            );
            assert!(
                (ek_ref - ek_opt).abs() < 1.0e-10,
                "{basis_name} optimized r2 E_K drifted from reference"
            );
            assert!(
                (e_ref - e_opt).abs() < 1.0e-10,
                "{basis_name} optimized r2 total energy drifted from reference"
            );

            let _ = fs::remove_file(ctrl_path);
        }
    }
}
