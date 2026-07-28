use crate::basis_io::{BasCell, Basis4Elem};
use crate::constants::c2s_matrix_const;
use crate::geom_io::GeomCell;
use rest_tensors::{MatrixFull, TensorOpt, TensorOptMut};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::{error::Error, fs};
// =========================
// Data structures
// =========================

#[derive(Debug, Deserialize, Clone)]
pub struct Shell {
    /// Angular momentum type: 0=s,1=p,2=d,3=f,...
    pub ang_type: u32,
    /// Primitive exponents
    pub exponents: Vec<f64>,
    /// coefficients[col][i]
    pub coefficients: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct BasisFunction {
    pub atom_idx: usize,        // Which atom this AO belongs to
    pub lx: u32,                // Cartesian angular momentum power in x
    pub ly: u32,                // Cartesian angular momentum power in y
    pub lz: u32,                // Cartesian angular momentum power in z
    pub exponents: Vec<f64>,    // Primitive exponents alpha_i
    pub coefficients: Vec<f64>, // Normalized contraction coefficients aligned with exponents
    pub center: [f64; 3],       // AO center coordinates (a.u.)
}

#[derive(Debug, Clone)]
pub struct RawShellShared {
    pub atom_idx: usize,
    pub center: [f64; 3],
    pub l: u32,
    pub exponents: Vec<f64>,
    pub coeff_columns: Vec<Vec<f64>>,
    pub is_aux: bool,
}

#[derive(Debug, Clone)]
pub struct RintShell {
    pub atom_idx: usize,
    pub center: [f64; 3],
    pub shell: Shell,
    pub column_idx: usize,
    pub cart_components: Vec<[u32; 3]>,
    pub ao_start: usize,
    pub ao_len: usize,
    pub is_aux: bool,
}

#[derive(Debug, Clone)]
pub struct RintContractedShell {
    pub atom_idx: usize,
    pub center: [f64; 3],
    pub l: u32,
    pub exponents: Vec<f64>,
    pub coeff_columns: Vec<Vec<f64>>,
    pub cart_components: Vec<[u32; 3]>,
    pub ao_start: usize,
    pub cart_len: usize,
    pub ao_len: usize,
    pub is_aux: bool,
}

impl RintContractedShell {
    pub fn nctr(&self) -> usize {
        self.coeff_columns.len()
    }
}

#[derive(Debug, Clone, Default)]
pub struct ShellInventoryBucket {
    pub shells: usize,
    pub contraction_columns: usize,
    pub cartesian_aos: usize,
}

#[derive(Debug, Clone, Default)]
pub struct ShellInventorySummary {
    pub total_shells: usize,
    pub total_contraction_columns: usize,
    pub total_cartesian_aos: usize,
    pub per_l: BTreeMap<u32, ShellInventoryBucket>,
}

#[derive(Debug, Clone)]
pub struct ShellCoeffDebugRow {
    pub atom_idx: usize,
    pub shell_idx: usize,
    pub l: u32,
    pub column_idx: usize,
    pub exponents: Vec<f64>,
    pub native_coefficients: Vec<f64>,
    pub rest_coefficients: Vec<f64>,
    pub rebuilt_shell_shared: Vec<f64>,
    pub is_aux: bool,
}

// =========================
// Math utilities (f64 only)
// =========================

/// Factorial n! as u128 (sufficient for small n used in normalization).
/// For larger n you should switch to log-factorial to avoid overflow.
pub fn factorial_u128(n: usize) -> u128 {
    if n <= 1 {
        return 1;
    }
    let mut result: u128 = 1;
    for i in 2..=n {
        result = result.saturating_mul(i as u128);
    }
    result
}

/// Double factorial n!! as u128.
/// For n <= 0, return 1.
pub fn double_factorial_u128(n: i32) -> u128 {
    if n <= 0 {
        return 1;
    }
    let mut result: u128 = 1;
    let mut k = n;
    while k > 0 {
        result = result.saturating_mul(k as u128);
        k -= 2;
    }
    result
}

/// log((2k-1)!!) accumulated form, used for log-safe normalization.
/// Here input is n where we mean (2n-1)!! for odd double factorial.
pub fn log_double_factorial_odd(n: u32) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let mut s = 0.0_f64;
    for k in 1..=n {
        let odd = 2.0 * (k as f64) - 1.0;
        s += odd.ln();
    }
    s
}

/// Cartesian primitive Gaussian normalization constant N(alpha,lx,ly,lz)
/// such that integral |N x^lx y^ly z^lz e^{-alpha r^2}|^2 d^3r = 1.
pub fn primitive_norm_cartesian(alpha: f64, lx: u32, ly: u32, lz: u32) -> f64 {
    assert!(alpha > 0.0, "alpha must be > 0");

    let pi = std::f64::consts::PI;
    let two = 2.0_f64;
    let four = 4.0_f64;

    // logN = (3/4) ln(2 alpha/pi)
    //      + (1/2)(lx+ly+lz) ln(4 alpha)
    //      - (1/2)[ ln((2lx-1)!!)+ln((2ly-1)!!)+ln((2lz-1)!!) ]
    let mut log_n = ((two * alpha / pi).ln()) * 0.75;

    let lsum = (lx + ly + lz) as f64;
    log_n += (four * alpha).ln() * (0.5 * lsum);

    let ldf =
        log_double_factorial_odd(lx) + log_double_factorial_odd(ly) + log_double_factorial_odd(lz);
    log_n -= 0.5 * ldf;

    log_n.exp()
}

/// Overlap integral for two Cartesian primitive Gaussians with the same center.
/// S_ij^(lx,ly,lz) = product over axes of axis_factor(n_axis)
fn primitive_overlap_same_center(lx: u32, ly: u32, lz: u32, alpha_i: f64, alpha_j: f64) -> f64 {
    let gamma = alpha_i + alpha_j;

    let sqrt_pi = std::f64::consts::PI.sqrt();

    // Gamma(n+1/2)
    let axis_factor = |n: u32, gamma: f64| -> f64 {
        let n_i32 = n as i32;

        let df = double_factorial_u128(2 * n_i32 - 1) as f64;

        // 2^n
        let two_pow_n = (1u128 << (n as u32)) as f64;

        // Gamma(n+1/2)
        let gpow = gamma.powf(n as f64 + 0.5);

        df * sqrt_pi / (two_pow_n * gpow)
    };

    let mut s = axis_factor(lx, gamma);
    s *= axis_factor(ly, gamma);
    s *= axis_factor(lz, gamma);

    s
}

// =========================
// Radial-shell normalization (approx for L>=2)
// =========================

/// Gamma(L+3/2) computed using half-integer closed form:
/// Gamma(n+1/2) = ((2n)! / (4^n n!)) * sqrt(pi), with n = L+1.
fn gamma_half_integer_for_L(L: u32) -> f64 {
    let n = (L + 1) as usize;

    // Compute (2n)! and n! using u128 (safe for small n).
    let fact_2n = factorial_u128(2 * n) as f64;
    let fact_n = factorial_u128(n) as f64;

    // 4^n = (2^2)^n = 2^(2n)
    let four_pow_n = (1u128 << (2 * n as u32)) as f64;

    let sqrt_pi = std::f64::consts::PI.sqrt();

    // Gamma(n+1/2)
    (fact_2n / (four_pow_n * fact_n)) * sqrt_pi
}

/// Radial shell normalization used for L>=2:
/// I(L) = 1/2 * sum_ij c_i c_j * Gamma(L+3/2) * (alpha_i+alpha_j)^(-(2L+3)/2)
/// A_shell = 1 / sqrt(I(L))
fn radial_shell_normalization(alphas: &[f64], coeffs: &[f64], L: u32) -> f64 {
    assert_eq!(alphas.len(), coeffs.len());

    let gamma_val = gamma_half_integer_for_L(L);
    let pow_exp = -((2 * L + 3) as f64) / 2.0;

    let mut I = 0.0_f64;
    for i in 0..alphas.len() {
        for j in 0..alphas.len() {
            let beta = alphas[i] + alphas[j];
            let beta_term = beta.powf(pow_exp);
            I += (coeffs[i] * coeffs[j]) * beta_term;
        }
    }

    I *= 0.5;
    I *= gamma_val;

    1.0 / I.sqrt()
}

// =========================
// Contraction coefficient normalization
// =========================

/// For a single Cartesian AO (lx,ly,lz), convert raw contraction coefficients c_i and exponents alpha_i
/// into the final normalized contraction coefficients.
/// - For L<=1 (s/p): do exact Cartesian normalization using primitive normalization + overlap normalization
/// - For L>=2: fall back to radial-shell approximation
pub fn make_contracted_coeffs_for_shell(
    alphas: &[f64],
    coeffs: &[f64], // raw contraction coefficients c_i (not multiplied by primitive normalization)
    lx: u32,
    ly: u32,
    lz: u32,
) -> Vec<f64> {
    assert_eq!(alphas.len(), coeffs.len());

    let L = lx + ly + lz;

    // Case 1: L<=1, exact Cartesian normalization
    if L <= 1 {
        let n = alphas.len();

        // 1) Primitive normalization: d0_i = c_i * N_prim(alpha_i; lx,ly,lz)
        let mut d0: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            let nprim = primitive_norm_cartesian(alphas[i], lx, ly, lz);
            d0.push(coeffs[i] * nprim);
        }

        // 2) Norm^2 of the unnormalized contracted function: quad = <psi|psi>
        // quad = sum_ij d0_i d0_j S_ij
        let mut quad = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let sij = primitive_overlap_same_center(lx, ly, lz, alphas[i], alphas[j]);
                quad += (d0[i] * d0[j]) * sij;
            }
        }

        // 3) Nc = 1 / sqrt(quad)
        let nc = if (quad.abs() - 1.0).abs() > 1e-8 {
            1.0 / quad.sqrt()
        } else {
            1.0
        };

        // 4) final coefficients = d0_i * Nc
        let mut final_coeffs: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            final_coeffs.push(d0[i] * nc);
        }
        return final_coeffs;
    }

    // Case 2: L>=2, radial-shell approximation
    let a_shell = radial_shell_normalization(alphas, coeffs, L);
    coeffs.iter().map(|&c| a_shell * c).collect()
}

// =========================
// Geom / Basis
// =========================

/// Parse GeomCell into
/// Supports geom.position as 3xnatom or natomx3.
pub fn parse_geomcell(geom: &GeomCell) -> Result<Vec<(String, [f64; 3])>, Box<dyn Error>> {
    let natom = geom.elem.len();
    let nrow = geom.position.size[0];
    let ncol = geom.position.size[1];
    if natom == 0 {
        return Err("GeomCell.elem is empty".into());
    }
    let mut atoms = Vec::with_capacity(natom);
    for i in 0..natom {
        let (x, y, z) = if nrow == 3 && ncol == natom {
            (
                geom.position[(0, i)],
                geom.position[(1, i)],
                geom.position[(2, i)],
            )
        } else if nrow == natom && ncol == 3 {
            (
                geom.position[(i, 0)],
                geom.position[(i, 1)],
                geom.position[(i, 2)],
            )
        } else {
            return Err(format!(
                "GeomCell.position shape mismatch: got {}x{}, natom={}. Expect 3xnatom or natomx3",
                nrow, ncol, natom
            )
            .into());
        };
        atoms.push((geom.elem[i].clone(), [x, y, z]));
    }
    Ok(atoms)
}

/// Find the Basis4Elem for a given atom index.
fn basis4elem_for_atom<'a>(
    basis4elem: &'a [Basis4Elem],
    atom_idx: usize,
) -> Result<&'a Basis4Elem, Box<dyn Error>> {
    basis4elem.get(atom_idx).ok_or_else(|| {
        format!(
            "basis4elem.len()={} < atom_idx+1={}",
            basis4elem.len(),
            atom_idx + 1
        )
        .into()
    })
}

/// Read all shells for a given atom index from Basis4Elem array.
pub fn read_shells_for_atom_from_basis4elem(
    basis4elem: &[Basis4Elem],
    atom_idx: usize,
) -> Result<Vec<Shell>, Box<dyn Error>> {
    let b4e = basis4elem_for_atom(basis4elem, atom_idx)?;
    read_shells_from_basis4elem(b4e)
}

/// Read all shells from a single atom's Basis4Elem (Basis4Elem.electron_shells).
pub fn read_shells_from_basis4elem(b4e: &Basis4Elem) -> Result<Vec<Shell>, Box<dyn Error>> {
    let mut shells_all: Vec<Shell> = Vec::new();

    for bas in b4e.electron_shells.iter() {
        let shells = bascell_to_shells(bas)?;
        shells_all.extend(shells);
    }

    Ok(shells_all)
}
/// Convert one BasCell into one or multiple Shell(s).
pub fn bascell_to_shells(bas: &BasCell) -> Result<Vec<Shell>, Box<dyn Error>> {
    if bas.angular_momentum.is_empty() {
        return Err("BasCell.angular_momentum is empty".into());
    }
    if bas.exponents.is_empty() {
        return Err("BasCell.exponents is empty".into());
    }
    if bas.native_coefficients.is_empty() {
        return Err("BasCell.coefficients is empty".into());
    }

    let nprim = bas.exponents.len();

    // Case 1: Non-Pople (one L, multiple contraction columns)
    if bas.angular_momentum.len() == 1 {
        let l = bas.angular_momentum[0];
        if l < 0 {
            return Err(format!("Invalid angular momentum: {}", l).into());
        }

        for (col_idx, col) in bas.native_coefficients.iter().enumerate() {
            if col.len() != nprim {
                return Err(format!(
                    "BasCell L={} col{}: coeff len {} != nprim {}",
                    l,
                    col_idx,
                    col.len(),
                    nprim
                )
                .into());
            }
        }

        return Ok(vec![Shell {
            ang_type: l as u32,
            exponents: bas.exponents.clone(),
            coefficients: bas.native_coefficients.clone(),
        }]);
    }

    // Case 2: Pople/SP-like (multiple L share same exponents; one coeff column per L)
    if bas.angular_momentum.len() != bas.native_coefficients.len() {
        return Err(format!(
            "Pople-like BasCell mismatch: angular_momentum.len()={} but coefficients.len()={}",
            bas.angular_momentum.len(),
            bas.native_coefficients.len()
        )
        .into());
    }

    let mut shells = Vec::with_capacity(bas.angular_momentum.len());
    for (k, &l_k) in bas.angular_momentum.iter().enumerate() {
        if l_k < 0 {
            return Err(format!("Invalid angular momentum at k={}: {}", k, l_k).into());
        }
        let coeff_col = &bas.native_coefficients[k];
        if coeff_col.len() != nprim {
            return Err(format!(
                "BasCell Pople k={} L={}: coeff len {} != nprim {}",
                k,
                l_k,
                coeff_col.len(),
                nprim
            )
            .into());
        }

        shells.push(Shell {
            ang_type: l_k as u32,
            exponents: bas.exponents.clone(),
            coefficients: vec![coeff_col.clone()], // one contraction column
        });
    }

    Ok(shells)
}

pub fn primitive_norm_shell_l(alpha: f64, l: u32) -> f64 {
    primitive_norm_cartesian(alpha, l, 0, 0)
}

pub fn shell_overlap_same_center_l(l: u32, alpha_i: f64, alpha_j: f64) -> f64 {
    primitive_overlap_same_center(l, 0, 0, alpha_i, alpha_j)
}

fn raw_shells_from_bascell(
    bas: &BasCell,
    atom_idx: usize,
    center: [f64; 3],
    is_aux: bool,
) -> Result<Vec<RawShellShared>, Box<dyn Error>> {
    if bas.angular_momentum.is_empty() {
        return Err("BasCell.angular_momentum is empty".into());
    }
    if bas.exponents.is_empty() {
        return Err("BasCell.exponents is empty".into());
    }
    if bas.native_coefficients.is_empty() {
        return Err("BasCell.native_coefficients is empty".into());
    }

    let nprim = bas.exponents.len();

    // -------------------------
    // Case 1: non-Pople
    // -------------------------
    if bas.angular_momentum.len() == 1 {
        let l = bas.angular_momentum[0];
        if l < 0 {
            return Err(format!("Invalid angular momentum: {}", l).into());
        }

        for (col_idx, col) in bas.native_coefficients.iter().enumerate() {
            if col.len() != nprim {
                return Err(format!(
                    "BasCell L={} col{}: coeff len {} != nprim {}",
                    l,
                    col_idx,
                    col.len(),
                    nprim
                )
                .into());
            }
        }

        // 关键修复：
        // standard 和 auxiliary 在这里都保留“原始完整 contraction 列语义”，
        // 不再做 claimed_by_later_pure_shell 的裁剪/置零。
        return Ok(vec![RawShellShared {
            atom_idx,
            center,
            l: l as u32,
            exponents: bas.exponents.clone(),
            coeff_columns: bas.native_coefficients.clone(),
            is_aux,
        }]);
    }

    // -------------------------
    // Case 2: Pople/SP-like
    // -------------------------
    if bas.angular_momentum.len() != bas.native_coefficients.len() {
        return Err(format!(
            "Pople-like BasCell mismatch: angular_momentum.len()={} but native_coefficients.len()={}",
            bas.angular_momentum.len(),
            bas.native_coefficients.len()
        )
        .into());
    }

    // 每个 L 保留一个原始完整列，不做 segmented 重解释
    let mut shells = Vec::with_capacity(bas.angular_momentum.len());

    for (k, &l_k) in bas.angular_momentum.iter().enumerate() {
        if l_k < 0 {
            return Err(format!("Invalid angular momentum at k={}: {}", k, l_k).into());
        }

        let coeff_col = &bas.native_coefficients[k];
        if coeff_col.len() != nprim {
            return Err(format!(
                "BasCell Pople k={} L={}: coeff len {} != nprim {}",
                k,
                l_k,
                coeff_col.len(),
                nprim
            )
            .into());
        }

        shells.push(RawShellShared {
            atom_idx,
            center,
            l: l_k as u32,
            exponents: bas.exponents.clone(),
            coeff_columns: vec![coeff_col.clone()],
            is_aux,
        });
    }

    Ok(shells)
}

pub fn read_raw_shells_from_basis4elem(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    atom_idx: usize,
    is_aux: bool,
) -> Result<Vec<RawShellShared>, Box<dyn Error>> {
    let atoms = parse_geomcell(geom)?;
    let (_, center) = atoms
        .get(atom_idx)
        .ok_or_else(|| format!("atom_idx {} out of range {}", atom_idx, atoms.len()))?;
    let b4e = basis4elem_for_atom(basis4elem, atom_idx)?;

    let mut shells_all = Vec::new();
    for bas in b4e.electron_shells.iter() {
        let shells = raw_shells_from_bascell(bas, atom_idx, *center, is_aux)?;
        shells_all.extend(shells);
    }
    Ok(shells_all)
}

fn read_all_raw_shells_from_basis4elem(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    is_aux: bool,
) -> Result<Vec<RawShellShared>, Box<dyn Error>> {
    let mut all = Vec::new();
    for atom_idx in 0..geom.elem.len() {
        all.extend(read_raw_shells_from_basis4elem(
            geom, basis4elem, atom_idx, is_aux,
        )?);
    }
    Ok(all)
}

fn update_shell_inventory(summary: &mut ShellInventorySummary, l: u32, ncols: usize) {
    let cart_dim = ((l as usize + 1) * (l as usize + 2)) / 2;
    summary.total_shells += 1;
    summary.total_contraction_columns += ncols;
    summary.total_cartesian_aos += cart_dim * ncols;
    let bucket = summary.per_l.entry(l).or_default();
    bucket.shells += 1;
    bucket.contraction_columns += ncols;
    bucket.cartesian_aos += cart_dim * ncols;
}

pub fn summarize_shell_inventory_shell_shared_from_raw(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    is_aux: bool,
) -> Result<ShellInventorySummary, Box<dyn Error>> {
    let mut summary = ShellInventorySummary::default();
    let shells = read_all_raw_shells_from_basis4elem(geom, basis4elem, is_aux)?;
    for shell in shells {
        update_shell_inventory(&mut summary, shell.l, shell.coeff_columns.len());
    }
    Ok(summary)
}

pub fn collect_shell_shared_debug_rows(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    is_aux: bool,
) -> Result<Vec<ShellCoeffDebugRow>, Box<dyn Error>> {
    let atoms = parse_geomcell(geom)?;
    let mut rows = Vec::new();
    for atom_idx in 0..geom.elem.len() {
        let (_, center) = atoms
            .get(atom_idx)
            .ok_or_else(|| format!("atom_idx {} out of range {}", atom_idx, atoms.len()))?;
        let b4e = basis4elem_for_atom(basis4elem, atom_idx)?;
        let mut shell_idx = 0usize;
        for bas in b4e.electron_shells.iter() {
            if bas.angular_momentum.is_empty() {
                continue;
            }
            if bas.angular_momentum.len() == 1 {
                let l = bas.angular_momentum[0];
                if l < 0 {
                    return Err(format!(
                        "Invalid angular momentum {} in shell_idx {}",
                        l, shell_idx
                    )
                    .into());
                }
                for col_idx in 0..bas.native_coefficients.len() {
                    let native = bas.native_coefficients[col_idx].clone();
                    let rest = bas.coefficients[col_idx].clone();
                    let rebuilt =
                        make_shell_shared_coeffs_from_raw(&bas.exponents, &native, l as u32)?;
                    rows.push(ShellCoeffDebugRow {
                        atom_idx,
                        shell_idx,
                        l: l as u32,
                        column_idx: col_idx,
                        exponents: bas.exponents.clone(),
                        native_coefficients: native,
                        rest_coefficients: rest,
                        rebuilt_shell_shared: rebuilt,
                        is_aux,
                    });
                }
                shell_idx += 1;
                continue;
            }

            if bas.angular_momentum.len() != bas.native_coefficients.len()
                || bas.angular_momentum.len() != bas.coefficients.len()
            {
                return Err(format!(
                    "Pople-like BasCell mismatch in debug rows: l={} native={} rest={}",
                    bas.angular_momentum.len(),
                    bas.native_coefficients.len(),
                    bas.coefficients.len()
                )
                .into());
            }

            for k in 0..bas.angular_momentum.len() {
                let l = bas.angular_momentum[k];
                if l < 0 {
                    return Err(format!(
                        "Invalid angular momentum {} in shell_idx {}",
                        l, shell_idx
                    )
                    .into());
                }
                let native = bas.native_coefficients[k].clone();
                let rest = bas.coefficients[k].clone();
                let rebuilt = make_shell_shared_coeffs_from_raw(&bas.exponents, &native, l as u32)?;
                rows.push(ShellCoeffDebugRow {
                    atom_idx,
                    shell_idx,
                    l: l as u32,
                    column_idx: 0,
                    exponents: bas.exponents.clone(),
                    native_coefficients: native,
                    rest_coefficients: rest,
                    rebuilt_shell_shared: rebuilt,
                    is_aux,
                });
                shell_idx += 1;
            }
        }
        let _ = center;
    }
    Ok(rows)
}

pub fn primitive_norm_shell_shared(alpha: f64, l: u32) -> f64 {
    assert!(alpha > 0.0, "alpha must be > 0");
    let denom = (double_factorial_u128(2 * l as i32 - 1) as f64).sqrt();
    (2.0 * alpha / std::f64::consts::PI).powf(0.75) * (4.0 * alpha).powf(0.5 * l as f64) / denom
}

pub fn primitive_overlap_same_center_shell_shared(alpha_i: f64, alpha_j: f64, l: u32) -> f64 {
    let df = double_factorial_u128(2 * l as i32 - 1) as f64;
    let pi_3_2 = std::f64::consts::PI.powf(1.5);
    let two_pow_l = (1u128 << l) as f64;
    let beta = alpha_i + alpha_j;
    df * pi_3_2 / (two_pow_l * beta.powf(l as f64 + 1.5))
}

pub fn make_shell_shared_coeffs_from_raw(
    exponents: &[f64],
    coeffs_raw: &[f64],
    l: u32,
) -> Result<Vec<f64>, Box<dyn Error>> {
    assert_eq!(exponents.len(), coeffs_raw.len());
    if exponents.is_empty() {
        return Ok(vec![]);
    }

    let d0: Vec<f64> = exponents
        .iter()
        .zip(coeffs_raw.iter())
        .map(|(alpha, coeff)| coeff * primitive_norm_shell_shared(*alpha, l))
        .collect();

    let mut q = 0.0_f64;
    for i in 0..exponents.len() {
        for j in 0..exponents.len() {
            q += d0[i]
                * d0[j]
                * primitive_overlap_same_center_shell_shared(exponents[i], exponents[j], l);
        }
    }

    if q <= 0.0 {
        return Err(format!("Non-positive shell-shared norm Q={} for l={}", q, l).into());
    }

    let nc = 1.0 / q.sqrt();
    // let sp_fix = match l {
    //     0 => (4.0 * std::f64::consts::PI).sqrt(),
    //     1 => (4.0 * std::f64::consts::PI / 3.0).sqrt(),
    //     _ => 1.0,
    // };
    Ok(d0.into_iter().map(|v| nc * v).collect())
}

pub fn normalize_raw_shells_to_shell_shared(
    shells: &[RawShellShared],
) -> Result<Vec<RawShellShared>, Box<dyn Error>> {
    let mut normalized = Vec::with_capacity(shells.len());
    for shell in shells.iter() {
        let mut coeff_columns = Vec::with_capacity(shell.coeff_columns.len());
        for coeff_col in shell.coeff_columns.iter() {
            if coeff_col.len() != shell.exponents.len() {
                return Err(format!(
                    "shell l={} coeff len {} != exps len {}",
                    shell.l,
                    coeff_col.len(),
                    shell.exponents.len()
                )
                .into());
            }
            coeff_columns.push(make_shell_shared_coeffs_from_raw(
                &shell.exponents,
                coeff_col,
                shell.l,
            )?);
        }
        normalized.push(RawShellShared {
            atom_idx: shell.atom_idx,
            center: shell.center,
            l: shell.l,
            exponents: shell.exponents.clone(),
            coeff_columns,
            is_aux: shell.is_aux,
        });
    }
    Ok(normalized)
}

pub fn expand_shell_shared_to_basis_functions(
    shells: &[RawShellShared],
) -> Result<Vec<BasisFunction>, Box<dyn Error>> {
    let mut all_basis = Vec::new();
    for shell in shells.iter() {
        for coeff_col in shell.coeff_columns.iter() {
            if coeff_col.len() != shell.exponents.len() {
                return Err(format!(
                    "shell l={} coeff len {} != exps len {}",
                    shell.l,
                    coeff_col.len(),
                    shell.exponents.len()
                )
                .into());
            }
            for (lx, ly, lz) in cartesian_components(shell.l) {
                all_basis.push(BasisFunction {
                    atom_idx: shell.atom_idx,
                    lx,
                    ly,
                    lz,
                    exponents: shell.exponents.clone(),
                    coefficients: coeff_col.clone(),
                    center: shell.center,
                });
            }
        }
    }
    Ok(all_basis)
}

pub fn expand_shell_shared_to_rint_shells(
    shells: &[RawShellShared],
) -> Result<Vec<RintShell>, Box<dyn Error>> {
    let mut all_shells = Vec::new();
    let mut ao_start = 0usize;
    for shell in shells.iter() {
        let components: Vec<[u32; 3]> = cartesian_components(shell.l)
            .into_iter()
            .map(|(lx, ly, lz)| [lx, ly, lz])
            .collect();
        let ao_len = components.len();
        for (column_idx, coeff_col) in shell.coeff_columns.iter().enumerate() {
            if coeff_col.len() != shell.exponents.len() {
                return Err(format!(
                    "shell l={} coeff len {} != exps len {}",
                    shell.l,
                    coeff_col.len(),
                    shell.exponents.len()
                )
                .into());
            }
            all_shells.push(RintShell {
                atom_idx: shell.atom_idx,
                center: shell.center,
                shell: Shell {
                    ang_type: shell.l,
                    exponents: shell.exponents.clone(),
                    coefficients: vec![coeff_col.clone()],
                },
                column_idx,
                cart_components: components.clone(),
                ao_start,
                ao_len,
                is_aux: shell.is_aux,
            });
            ao_start += ao_len;
        }
    }
    Ok(all_shells)
}

pub fn expand_shell_shared_to_contracted_rint_shells(
    shells: &[RawShellShared],
) -> Result<Vec<RintContractedShell>, Box<dyn Error>> {
    let mut all_shells = Vec::new();
    let mut ao_start = 0usize;
    for shell in shells.iter() {
        let components: Vec<[u32; 3]> = cartesian_components(shell.l)
            .into_iter()
            .map(|(lx, ly, lz)| [lx, ly, lz])
            .collect();
        let cart_len = components.len();
        for coeff_col in shell.coeff_columns.iter() {
            if coeff_col.len() != shell.exponents.len() {
                return Err(format!(
                    "shell l={} coeff len {} != exps len {}",
                    shell.l,
                    coeff_col.len(),
                    shell.exponents.len()
                )
                .into());
            }
        }
        let ao_len = cart_len * shell.coeff_columns.len();
        all_shells.push(RintContractedShell {
            atom_idx: shell.atom_idx,
            center: shell.center,
            l: shell.l,
            exponents: shell.exponents.clone(),
            coeff_columns: shell.coeff_columns.clone(),
            cart_components: components,
            ao_start,
            cart_len,
            ao_len,
            is_aux: shell.is_aux,
        });
        ao_start += ao_len;
    }
    Ok(all_shells)
}

pub fn expand_rint_shells_to_basis_functions(
    shells: &[RintShell],
) -> Result<Vec<BasisFunction>, Box<dyn Error>> {
    let mut all_basis = Vec::new();
    for shell in shells.iter() {
        let Some(coeff_col) = shell.shell.coefficients.first() else {
            return Err(format!(
                "rint shell atom={} l={} has no coefficient column",
                shell.atom_idx, shell.shell.ang_type
            )
            .into());
        };
        if coeff_col.len() != shell.shell.exponents.len() {
            return Err(format!(
                "rint shell l={} coeff len {} != exps len {}",
                shell.shell.ang_type,
                coeff_col.len(),
                shell.shell.exponents.len()
            )
            .into());
        }
        for [lx, ly, lz] in shell.cart_components.iter().copied() {
            all_basis.push(BasisFunction {
                atom_idx: shell.atom_idx,
                lx,
                ly,
                lz,
                exponents: shell.shell.exponents.clone(),
                coefficients: coeff_col.clone(),
                center: shell.center,
            });
        }
    }
    Ok(all_basis)
}

/// For a given angular momentum L, generate all (lx,ly,lz) such that lx+ly+lz = L.
pub fn cartesian_components(L: u32) -> Vec<(u32, u32, u32)> {
    let mut v = Vec::new();
    for lx in (0..=L).rev() {
        for lz in 0..=(L - lx) {
            let ly = L - lx - lz;
            v.push((lx, ly, lz));
        }
    }
    v
}

/// Small labels for debugging / printing.
pub fn cart_label(lx: u32, ly: u32, lz: u32) -> &'static str {
    match (lx, ly, lz) {
        (0, 0, 0) => "s",
        (1, 0, 0) => "px",
        (0, 1, 0) => "py",
        (0, 0, 1) => "pz",
        (2, 0, 0) => "d_xx",
        (0, 2, 0) => "d_yy",
        (0, 0, 2) => "d_zz",
        (1, 1, 0) => "d_xy",
        (1, 0, 1) => "d_xz",
        (0, 1, 1) => "d_yz",
        (3, 0, 0) => "f_xxx",
        (2, 1, 0) => "f_xxy",
        (2, 0, 1) => "f_xxz",
        (1, 2, 0) => "f_xyy",
        (1, 1, 1) => "f_xyz",
        (1, 0, 2) => "f_xzz",
        (0, 3, 0) => "f_yyy",
        (0, 2, 1) => "f_yyz",
        (0, 1, 2) => "f_yzz",
        (0, 0, 3) => "f_zzz",
        _ => "h",
    }
}

pub fn dump_rust_ao_labels(basis: &[BasisFunction]) {
    for (i, bf) in basis.iter().enumerate() {
        println!(
            "{:>3}  {:>4}   ({},{},{})   [{:>10.5e},{:>10.5e},{:>10.5e}]",
            i, bf.atom_idx, bf.lx, bf.ly, bf.lz, bf.center[0], bf.center[1], bf.center[2]
        );
    }
}

pub fn push_cart_shell(
    all_basis: &mut Vec<BasisFunction>,
    atom_idx: usize,
    coord: [f64; 3],
    exps: &[f64],
    coeffs: &[f64],
    L: u32,
) {
    for (lx, ly, lz) in cartesian_components(L) {
        let final_coeffs = make_contracted_coeffs_for_shell(exps, coeffs, lx, ly, lz);

        all_basis.push(BasisFunction {
            atom_idx,
            lx,
            ly,
            lz,
            exponents: exps.to_vec(),
            coefficients: final_coeffs,
            center: coord,
        });
    }
}

pub fn load_molecule_shell_shared_from_raw(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
) -> Result<Vec<BasisFunction>, Box<dyn Error>> {
    let natm = geom.elem.len();
    let mut raw_shells = Vec::new();
    for atom_idx in 0..natm {
        raw_shells.extend(read_raw_shells_from_basis4elem(
            geom, basis4elem, atom_idx, false,
        )?);
    }
    let normalized = normalize_raw_shells_to_shell_shared(&raw_shells)?;
    expand_shell_shared_to_basis_functions(&normalized)
}

pub fn load_molecule_rint_shells_from_raw(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
) -> Result<Vec<RintShell>, Box<dyn Error>> {
    let natm = geom.elem.len();
    let mut raw_shells = Vec::new();
    for atom_idx in 0..natm {
        raw_shells.extend(read_raw_shells_from_basis4elem(
            geom, basis4elem, atom_idx, false,
        )?);
    }
    let normalized = normalize_raw_shells_to_shell_shared(&raw_shells)?;
    expand_shell_shared_to_rint_shells(&normalized)
}

pub fn load_molecule_contracted_rint_shells_from_raw(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
) -> Result<Vec<RintContractedShell>, Box<dyn Error>> {
    let natm = geom.elem.len();
    let mut raw_shells = Vec::new();
    for atom_idx in 0..natm {
        raw_shells.extend(read_raw_shells_from_basis4elem(
            geom, basis4elem, atom_idx, false,
        )?);
    }
    let normalized = normalize_raw_shells_to_shell_shared(&raw_shells)?;
    expand_shell_shared_to_contracted_rint_shells(&normalized)
}

pub fn load_aux_molecule_shell_shared_from_raw(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
) -> Result<Vec<BasisFunction>, Box<dyn Error>> {
    let natm = geom.elem.len();
    let mut raw_shells = Vec::new();
    for atom_idx in 0..natm {
        raw_shells.extend(read_raw_shells_from_basis4elem(
            geom, basis4elem, atom_idx, true,
        )?);
    }
    let normalized = normalize_raw_shells_to_shell_shared(&raw_shells)?;
    expand_shell_shared_to_basis_functions(&normalized)
}

pub fn load_aux_rint_shells_from_raw(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
) -> Result<Vec<RintShell>, Box<dyn Error>> {
    let natm = geom.elem.len();
    let mut raw_shells = Vec::new();
    for atom_idx in 0..natm {
        raw_shells.extend(read_raw_shells_from_basis4elem(
            geom, basis4elem, atom_idx, true,
        )?);
    }
    let normalized = normalize_raw_shells_to_shell_shared(&raw_shells)?;
    expand_shell_shared_to_rint_shells(&normalized)
}

pub fn count_spheric_ao_shell_shared(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    is_aux: bool,
) -> usize {
    let shells = read_all_raw_shells_from_basis4elem(geom, basis4elem, is_aux)
        .expect("failed to read raw shell_shared metadata from Basis4Elem");
    let mut nao_sph = 0usize;
    for shell in shells {
        nao_sph += (2 * shell.l as usize + 1) * shell.coeff_columns.len();
    }
    nao_sph
}

pub fn build_cart_from_spheric_transform_shell_shared(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    nao_cart: usize,
    nao_sph: usize,
    is_aux: bool,
) -> MatrixFull<f64> {
    let shells = read_all_raw_shells_from_basis4elem(geom, basis4elem, is_aux)
        .expect("failed to read raw shell_shared metadata from Basis4Elem");
    let mut transform = MatrixFull::new([nao_cart, nao_sph], 0.0);
    let mut cart_offset = 0usize;
    let mut sph_offset = 0usize;
    for shell in shells {
        let l = shell.l as usize;
        let cart_dim = (l + 1) * (l + 2) / 2;
        let sph_dim = 2 * l + 1;
        let c2s_owned = c2s_matrix_const(l);
        let c2s = c2s_owned.to_matrixfullslice();
        for _ in shell.coeff_columns.iter() {
            for sph in 0..sph_dim {
                for cart in 0..cart_dim {
                    transform[(cart_offset + cart, sph_offset + sph)] =
                        *c2s.get2d([cart, sph]).expect("invalid c2s index");
                }
            }
            cart_offset += cart_dim;
            sph_offset += sph_dim;
        }
    }
    assert_eq!(
        cart_offset, nao_cart,
        "shell_shared cart transform ended at {}, expected {}",
        cart_offset, nao_cart
    );
    assert_eq!(
        sph_offset, nao_sph,
        "shell_shared sph transform ended at {}, expected {}",
        sph_offset, nao_sph
    );
    transform
}

pub fn transform_density_to_cartesian_shell_shared(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    p_rhf: &MatrixFull<f64>,
    nao_cart: usize,
    caller: &str,
    is_aux: bool,
) -> MatrixFull<f64> {
    let nao_sph = count_spheric_ao_shell_shared(geom, basis4elem, is_aux);
    if p_rhf.size == [nao_cart, nao_cart] {
        return p_rhf.clone();
    }
    assert_eq!(
        p_rhf.size,
        [nao_sph, nao_sph],
        "{}: shell_shared density size mismatch, got {:?}, expected either [{}, {}] or [{}, {}]",
        caller,
        p_rhf.size,
        nao_cart,
        nao_cart,
        nao_sph,
        nao_sph
    );
    let transform =
        build_cart_from_spheric_transform_shell_shared(geom, basis4elem, nao_cart, nao_sph, is_aux);
    let mut work = MatrixFull::new([nao_cart, nao_sph], 0.0);
    work.to_matrixfullslicemut().lapack_dgemm(
        &transform.to_matrixfullslice(),
        &p_rhf.to_matrixfullslice(),
        'N',
        'N',
        1.0,
        0.0,
    );
    let mut p_cart = MatrixFull::new([nao_cart, nao_cart], 0.0);
    p_cart.to_matrixfullslicemut().lapack_dgemm(
        &work.to_matrixfullslice(),
        &transform.to_matrixfullslice(),
        'N',
        'T',
        1.0,
        0.0,
    );
    p_cart
}

pub fn transform_operator_from_cartesian_shell_shared(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    v_cart: &MatrixFull<f64>,
    target_nao: usize,
    caller: &str,
    is_aux: bool,
) -> MatrixFull<f64> {
    if v_cart.size == [target_nao, target_nao] {
        return v_cart.clone();
    }
    let nao_cart = v_cart.size[0];
    assert_eq!(
        v_cart.size,
        [nao_cart, nao_cart],
        "{}: shell_shared operator must be square, got {:?}",
        caller,
        v_cart.size,
    );
    let nao_sph = count_spheric_ao_shell_shared(geom, basis4elem, is_aux);
    assert_eq!(
        target_nao, nao_sph,
        "{}: shell_shared operator target size mismatch, got {}, expected {}",
        caller, target_nao, nao_sph
    );
    let transform =
        build_cart_from_spheric_transform_shell_shared(geom, basis4elem, nao_cart, nao_sph, is_aux);
    let mut work = MatrixFull::new([nao_cart, nao_sph], 0.0);
    work.to_matrixfullslicemut().lapack_dgemm(
        &v_cart.to_matrixfullslice(),
        &transform.to_matrixfullslice(),
        'N',
        'N',
        1.0,
        0.0,
    );
    let mut v_sph = MatrixFull::new([nao_sph, nao_sph], 0.0);
    v_sph.to_matrixfullslicemut().lapack_dgemm(
        &transform.to_matrixfullslice(),
        &work.to_matrixfullslice(),
        'T',
        'N',
        1.0,
        0.0,
    );
    v_sph
}

pub fn transform_mo_coeff_to_cartesian_shell_shared(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    c_mo: &MatrixFull<f64>,
    nao_cart: usize,
    caller: &str,
    is_aux: bool,
) -> MatrixFull<f64> {
    let nao_sph = count_spheric_ao_shell_shared(geom, basis4elem, is_aux);
    let nmo = c_mo.size[1];
    if c_mo.size[0] == nao_cart {
        return c_mo.clone();
    }
    assert_eq!(
        c_mo.size[0], nao_sph,
        "{}: shell_shared MO/AO size mismatch, got {:?}, expected first dimension {} or {}",
        caller, c_mo.size, nao_cart, nao_sph
    );
    let transform =
        build_cart_from_spheric_transform_shell_shared(geom, basis4elem, nao_cart, nao_sph, is_aux);
    let mut c_cart = MatrixFull::new([nao_cart, nmo], 0.0);
    c_cart.to_matrixfullslicemut().lapack_dgemm(
        &transform.to_matrixfullslice(),
        &c_mo.to_matrixfullslice(),
        'N',
        'N',
        1.0,
        0.0,
    );
    c_cart
}

pub fn load_cartesian_rhf_basis_and_density_shell_shared(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    p_rhf: &MatrixFull<f64>,
    caller: &str,
) -> (Vec<BasisFunction>, MatrixFull<f64>) {
    let bfs = load_molecule_shell_shared_from_raw(geom, basis4elem)
        .expect("failed to build shell_shared bfs from GeomCell/BasCell");
    let p_cart = transform_density_to_cartesian_shell_shared(
        geom,
        basis4elem,
        p_rhf,
        bfs.len(),
        caller,
        false,
    );
    (bfs, p_cart)
}

pub fn load_cartesian_rhf_rint_shells_basis_and_density_shell_shared(
    geom: &GeomCell,
    basis4elem: &[Basis4Elem],
    p_rhf: &MatrixFull<f64>,
    caller: &str,
) -> (Vec<RintShell>, Vec<BasisFunction>, MatrixFull<f64>) {
    let shells = load_molecule_rint_shells_from_raw(geom, basis4elem)
        .expect("failed to build rint shells from GeomCell/BasCell");
    let bfs = expand_rint_shells_to_basis_functions(&shells)
        .expect("failed to expand rint shells to BasisFunction list");
    let p_cart = transform_density_to_cartesian_shell_shared(
        geom,
        basis4elem,
        p_rhf,
        bfs.len(),
        caller,
        false,
    );
    (shells, bfs, p_cart)
}
