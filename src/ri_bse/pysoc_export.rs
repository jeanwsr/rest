//! PySOC export module — writes BSE results to a JSON file for PySOC perturbative SOC calculation.
//!
//! The JSON file contains: geometry, Gaussian-format basis set, MO coefficients/energies,
//! and CI coefficients (X+Y / X-Y) in alpha/beta spin format compatible with soc_td.
//!
//! Requirements:
//! - Cartesian basis (basis_type = "cartesian") — molsoc only computes Cartesian integrals
//! - BSE with bse_spin = "both" (singlet + triplet excitations)
//! - No frozen core (REST BSE doesn't freeze core)

use crate::scf_io::SCF;
use crate::ri_gw::get_occupation_parameters;
use rest_libcint::CINTR2CDATA;
use rest_libcint::prelude::CintType;
use serde::Serialize;
use std::fs::File;
use std::io::Write;

/// Hartree → eV conversion factor.
const HARTREE_TO_EV: f64 = 27.21138505;
/// Bohr → Angstrom conversion factor (same as constants::ANG).
const BOHR_TO_ANG: f64 = 0.5291772083;

/// Top-level JSON structure exported for PySOC.
#[derive(Serialize)]
struct PysocExport {
    program: String,
    method: String,
    basis_type: String,
    geometry: Vec<(String, f64, f64, f64)>,
    basis_set_gaussian: String,
    ao_basis: Vec<usize>,
    num_orbitals: usize,
    num_occupied_orbitals: usize,
    num_virtual_orbitals: usize,
    num_frozen_orbitals: usize,
    mo_energies_hartree: Vec<f64>,
    mo_coefficients: Vec<f64>,
    singlet_states: Vec<(usize, f64)>,
    triplet_states: Vec<(usize, f64)>,
    ci_singlet_xpy: Vec<f64>,
    ci_singlet_xmy: Vec<f64>,
    ci_triplet_xpy: Vec<f64>,
    ci_triplet_xmy: Vec<f64>,
}

/// Entry point: export BSE singlet + triplet results to a PySOC-compatible JSON file.
///
/// Called from `bse_main` after `bse_both_spins` returns, when `qp_ctrl.pysoc == true`.
///
/// # Arguments
/// * `scf` - SCF data (provides geometry, basis, MO coefficients, MO energies)
/// * `singlet_excitations` - `[(energy_hartree, eigenvector), ...]` from BSE singlet solve
/// * `triplet_excitations` - `[(energy_hartree, eigenvector), ...]` from BSE triplet solve
/// * `tda` - true if TDA was used (Y=0), false for full LR
/// * `output_path` - path for the output JSON file
pub fn export_pysoc_json(
    scf: &SCF,
    singlet_excitations: &[(f64, Vec<f64>)],
    triplet_excitations: &[(f64, Vec<f64>)],
    tda: bool,
    output_path: &str,
) {
    // 1. Enforce Cartesian basis (molsoc only computes Cartesian SOC integrals)
    if !matches!(scf.mol.cint_type, CintType::Cartesian) {
        panic!(
            "PySOC export requires Cartesian basis. Please set basis_type = \"cartesian\" in the input."
        );
    }

    let (start_mo, num_state, occ_size, vir_size, _homo, _lumo) =
        get_occupation_parameters(scf, 'N');
    let nao = scf.mol.num_basis;

    // 2. Geometry: REST stores in Bohr, convert to Angstrom for molsoc (ANG keyword)
    let geometry = export_geometry(scf);

    // 3. Basis set in Gaussian GFInput format + shell sizes
    let (basis_set_gaussian, ao_basis) = export_basis_gaussian(scf);

    // 4. MO coefficients: active orbitals only, column-major flat
    //    Scale Cartesian D/F mixed components to match molsoc's individual normalization.
    //    libcint uses uniform normalization (all components share gto_norm(l,alpha)),
    //    so <d_xy|d_xy> = 1/3 while molsoc normalizes each component to 1.
    //    Scale factor for component (a,b,c) with a+b+c=l:
    //      sqrt((2a-1)!!*(2b-1)!!*(2c-1)!! / (2l-1)!!)
    let mo_coefficients = export_mo_coefficients(scf, start_mo, num_state, nao);

    // 5. MO energies: Hartree, active orbitals only (soc_td converts to eV internally)
    let mo_energies_hartree = export_mo_energies(scf, start_mo, num_state);

    // 6. Excitation energies: convert Hartree → eV
    let singlet_states: Vec<(usize, f64)> = singlet_excitations
        .iter()
        .enumerate()
        .map(|(i, (e, _))| (i + 1, e * HARTREE_TO_EV))
        .collect();
    let triplet_states: Vec<(usize, f64)> = triplet_excitations
        .iter()
        .enumerate()
        .map(|(i, (e, _))| (i + 1, e * HARTREE_TO_EV))
        .collect();

    // 7. CI coefficients: normalize + transpose + alpha/beta conversion
    let (ci_singlet_xpy, ci_singlet_xmy) =
        convert_ci_set(singlet_excitations, tda, occ_size, vir_size, true);
    let (ci_triplet_xpy, ci_triplet_xmy) =
        convert_ci_set(triplet_excitations, tda, occ_size, vir_size, false);

    let export = PysocExport {
        program: "REST".to_string(),
        method: "BSE".to_string(),
        basis_type: "cartesian".to_string(),
        geometry,
        basis_set_gaussian,
        ao_basis,
        num_orbitals: nao,
        num_occupied_orbitals: occ_size,
        num_virtual_orbitals: vir_size,
        num_frozen_orbitals: 0,
        mo_energies_hartree,
        mo_coefficients,
        singlet_states,
        triplet_states,
        ci_singlet_xpy,
        ci_singlet_xmy,
        ci_triplet_xpy,
        ci_triplet_xmy,
    };

    let json = serde_json::to_string_pretty(&export).expect("JSON serialization failed");
    let mut file = File::create(output_path).expect("Cannot create PySOC export file");
    write!(file, "{}", json).expect("Cannot write PySOC export file");
    println!("PySOC export written to: {}", output_path);
}

/// Extract geometry as (element, x, y, z) in Angstrom.
fn export_geometry(scf: &SCF) -> Vec<(String, f64, f64, f64)> {
    scf.mol
        .geom
        .elem
        .iter()
        .zip(scf.mol.geom.position.iter_columns_full())
        .map(|(elem, pos)| {
            (
                elem.clone(),
                pos[0] * BOHR_TO_ANG,
                pos[1] * BOHR_TO_ANG,
                pos[2] * BOHR_TO_ANG,
            )
        })
        .collect()
}

/// Export basis set in Gaussian GFInput format and compute shell sizes.
///
/// Each shell in REST's `basis4elem` becomes one shell in the output.
/// SP shells are already split into separate S and P by REST's parser.
/// Coefficients are de-normalized (divided by gto_norm) to recover native values.
fn export_basis_gaussian(scf: &SCF) -> (String, Vec<usize>) {
    let mut text = String::new();
    let mut ao_basis = Vec::new();

    for (i_atom, bas4elem) in scf.mol.basis4elem.iter().enumerate() {
        let elem = &scf.mol.geom.elem[i_atom];
        text.push_str(&format!("{}  0\n", elem));

        for shell in &bas4elem.electron_shells {
            let l = shell.angular_momentum[0];
            let label = match l {
                0 => "S",
                1 => "P",
                2 => "D",
                3 => "F",
                _ => panic!("PySOC export: unsupported angular momentum l={}", l),
            };

            // Each contraction column becomes one shell entry (typically 1 per BasCell)
            // Use native_coefficients (original un-normalized values from basis set file)
            // rather than de-normalizing `coefficients` (which may have been modified).
            let native = &shell.native_coefficients;
            for (icol, coeff_col) in native.iter().enumerate() {
                let nprim = shell.exponents.len();
                text.push_str(&format!("{}   {}   1.00\n", label, nprim));

                for (&exp, &coeff) in shell.exponents.iter().zip(coeff_col.iter()) {
                    text.push_str(&format!("  {:>18.12}  {:>18.12}\n", exp, coeff));
                }

                // Cartesian shell size: (l+1)*(l+2)/2
                let n_cart = ((l + 1) * (l + 2) / 2) as usize;
                ao_basis.push(n_cart);
            }
        }
        text.push_str("****\n");
    }

    // Remove trailing "****\n" — write_molsoc_basis adds "END" automatically
    if text.ends_with("****\n") {
        text.truncate(text.len() - 5);
    }

    (text, ao_basis)
}

/// Extract active MO coefficients as a flat column-major array.
///
/// Returns `nao * num_state` values. Column `j` (0-based) corresponds to MO `start_mo + j`.
/// Layout matches soc_td's `mo_coeff.dat`: [MO1_AO1, MO1_AO2, ..., MO1_AOnao, MO2_AO1, ...].
///
/// **Cartesian normalization scaling**: libcint uses uniform normalization where all Cartesian
/// components of a shell share the same `gto_norm(l, alpha)`. This makes mixed components
/// like d_xy have self-overlap 1/3 (not 1). molsoc normalizes each component independently,
/// giving self-overlap 1 for all. We scale MO coefficients for mixed components to bridge
/// this difference.
fn export_mo_coefficients(scf: &SCF, start_mo: usize, num_state: usize, nao: usize) -> Vec<f64> {
    let eigenvectors = &scf.eigenvectors[0];

    // Build per-AO scaling factors for Cartesian normalization conversion.
    let scales = build_ao_scales(scf);

    let mut mo_coefficients = Vec::with_capacity(nao * num_state);

    for j in start_mo..start_mo + num_state {
        let col_start = j * nao;
        let col_end = (j + 1) * nao;
        for (i, (&coeff, &scale)) in eigenvectors.data[col_start..col_end]
            .iter()
            .zip(scales.iter())
            .enumerate()
        {
            mo_coefficients.push(coeff * scale);
        }
    }

    mo_coefficients
}

/// Build per-AO scaling factors to convert from libcint's uniform Cartesian normalization
/// to molsoc's individual Cartesian normalization.
///
/// For a Cartesian component (a, b, c) with a+b+c=l, the scale factor is:
///   sqrt((2a-1)!! * (2b-1)!! * (2c-1)!! / (2l-1)!!)
///
/// This is 1 for pure components (l,0,0) and <1 for mixed components.
fn build_ao_scales(scf: &SCF) -> Vec<f64> {
    let mut scales = Vec::new();

    for bas4elem in &scf.mol.basis4elem {
        for shell in &bas4elem.electron_shells {
            let l = shell.angular_momentum[0] as usize;
            for _ in &shell.coefficients {
                // Cartesian component ordering: (lx, ly, lz) with lx descending,
                // then ly descending within each lx.
                // This matches REST's cartesian_gto_std and molsoc's genop.f.
                for lx in (0..=l).rev() {
                    let rl = l - lx;
                    for ly in (0..=rl).rev() {
                        let lz = rl - ly;
                        scales.push(cartesian_norm_scale(lx, ly, lz));
                    }
                }
            }
        }
    }

    assert_eq!(
        scales.len(),
        scf.mol.num_basis,
        "AO scale count mismatch: {} vs nao {}",
        scales.len(),
        scf.mol.num_basis
    );

    scales
}

/// Compute the normalization scale factor for Cartesian component (lx, ly, lz).
/// Returns sqrt(double_factorial(2lx-1) * double_factorial(2ly-1) * double_factorial(2lz-1)
///               / double_factorial(2*(lx+ly+lz)-1))
/// where double_factorial(-1) = 1 (convention).
fn cartesian_norm_scale(lx: usize, ly: usize, lz: usize) -> f64 {
    let l = lx + ly + lz;
    let numerator = double_factorial_odd(2 * lx as i32 - 1)
        * double_factorial_odd(2 * ly as i32 - 1)
        * double_factorial_odd(2 * lz as i32 - 1);
    let denominator = double_factorial_odd(2 * l as i32 - 1);
    (numerator as f64 / denominator as f64).sqrt()
}

/// Compute the odd double factorial (2n-1)!! = 1 * 3 * 5 * ... * (2n-1).
/// For n < 0 (i.e., input -1), returns 1 (convention for 0!! = (-1)!! = 1).
fn double_factorial_odd(n: i32) -> i32 {
    if n <= 0 {
        return 1;
    }
    let mut result = 1;
    let mut k = 1;
    while k <= n {
        result *= k;
        k += 2;
    }
    result
}

/// Extract active MO energies in Hartree.
///
/// Uses `gwqp.0` (GW quasiparticle energies) which is always populated before BSE.
/// soc_td converts Hartree → eV internally for `qm_flag != 'tddftb'`.
fn export_mo_energies(scf: &SCF, start_mo: usize, num_state: usize) -> Vec<f64> {
    let energies = if scf.gwqp.0.is_empty() {
        &scf.eigenvalues[0]
    } else {
        &scf.gwqp.0
    };
    energies[start_mo..start_mo + num_state].to_vec()
}

/// Convert a set of BSE eigenvectors to PySOC's CI coefficient format.
///
/// Steps per state:
/// 1. Split into X and Y (TDA: Y=0; LR: first half=X, second half=Y)
/// 2. Transpose from REST layout (i fastest, a slowest) to soc_td layout (a fastest, i slowest)
/// 3. Normalize to |X|² - |Y|² = 1/2 (so soc_td's check `sum(X+Y·X-Y) == 1` passes)
/// 4. Construct X+Y and X-Y
/// 5. Expand to alpha/beta format: singlet → α=β, triplet → α=-β
///
/// Returns (xpy_all, xmy_all), each of length `n_states * 2 * occ_size * vir_size`.
fn convert_ci_set(
    excitations: &[(f64, Vec<f64>)],
    tda: bool,
    occ_size: usize,
    vir_size: usize,
    is_singlet: bool,
) -> (Vec<f64>, Vec<f64>) {
    let n = occ_size * vir_size;
    let mut xpy_all = Vec::new();
    let mut xmy_all = Vec::new();

    for (_energy, eigvec) in excitations {
        // 1. Split X, Y (REST layout: index = i + a * occ_size, i fastest)
        let (x_rest, y_rest): (Vec<f64>, Vec<f64>) = if tda {
            (eigvec.clone(), vec![0.0; n])
        } else {
            assert_eq!(
                eigvec.len(),
                2 * n,
                "Full LR eigenvector length mismatch: expected {}, got {}",
                2 * n,
                eigvec.len()
            );
            (eigvec[0..n].to_vec(), eigvec[n..2 * n].to_vec())
        };

        // 2. Transpose to soc_td layout: index = i * vir_size + a (a fastest)
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..occ_size {
            for a in 0..vir_size {
                let idx_rest = i + a * occ_size;
                let idx_soc = i * vir_size + a;
                x[idx_soc] = x_rest[idx_rest];
                y[idx_soc] = y_rest[idx_rest];
            }
        }

        // 3. Normalize to |X|² - |Y|² = 1/2
        let x_norm: f64 = x.iter().map(|v| v * v).sum();
        let y_norm: f64 = y.iter().map(|v| v * v).sum();
        let diff = x_norm - y_norm;
        if diff <= 0.0 {
            eprintln!(
                "  Warning: |X|² - |Y|² = {:.6e} <= 0 for a state, using absolute value",
                diff
            );
        }
        let norm_factor = (diff.abs() * 2.0_f64).sqrt();
        let x: Vec<f64> = x.iter().map(|v| v / norm_factor).collect();
        let y: Vec<f64> = y.iter().map(|v| v / norm_factor).collect();

        // 4. Construct X+Y and X-Y
        let xpy: Vec<f64> = x.iter().zip(y.iter()).map(|(xi, yi)| xi + yi).collect();
        let xmy: Vec<f64> = x.iter().zip(y.iter()).map(|(xi, yi)| xi - yi).collect();

        // 5. Alpha/beta format: singlet α=β, triplet α=-β
        xpy_all.extend(&xpy); // alpha
        if is_singlet {
            xpy_all.extend(&xpy); // beta = alpha
        } else {
            xpy_all.extend(xpy.iter().map(|v| -v)); // beta = -alpha
        }

        xmy_all.extend(&xmy); // alpha
        if is_singlet {
            xmy_all.extend(&xmy); // beta = alpha
        } else {
            xmy_all.extend(xmy.iter().map(|v| -v)); // beta = -alpha
        }
    }

    (xpy_all, xmy_all)
}
