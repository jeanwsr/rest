//! PySOC export module — writes BSE results to a JSON file for PySOC perturbative SOC calculation.
//!
//! The JSON file contains: geometry, Gaussian-format basis set, MO coefficients/energies,
//! and CI coefficients (X+Y / X-Y) in alpha/beta spin format compatible with soc_td.
//!
//! Requirements:
//! - Molsoc only computes Cartesian integrals, so the exported basis is always
//!   Cartesian. REST calculations may use either `basis_type = "cartesian"` or
//!   `basis_type = "spheric"`; in the latter case MO coefficients are transformed
//!   from REST's spherical AOs to the exported Cartesian AOs here.
//! - BSE with bse_spin = "both" (singlet + triplet excitations)
//! - No frozen core (REST BSE doesn't freeze core)

use crate::constants::c2s_matrix_const;
use crate::scf_io::SCF;
use rest_libcint::prelude::CintType;
use rest_libcint::CINTR2CDATA;
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use tensors::TensorOpt;

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
    method: &str,
    start_mo: usize,
    num_state: usize,
    occ_size: usize,
    vir_size: usize,
) {
    if matches!(scf.mol.cint_type, CintType::Spinor) {
        panic!("PySOC export does not support spinor basis sets.");
    }

    let n_active = occ_size + vir_size; // number of active MOs to export

    // 1. Geometry: REST stores in Bohr, convert to Angstrom for molsoc (ANG keyword)
    let geometry = export_geometry(scf);

    // 2. Basis set in Gaussian GFInput format + Cartesian shell sizes.
    //    Molsoc only computes Cartesian integrals, so the JSON is always
    //    written in Cartesian format regardless of REST's internal basis type.
    let (basis_set_gaussian, ao_basis) = export_basis_gaussian(scf);
    let nao = ao_basis.iter().sum::<usize>();
    let nao_cart = count_cartesian_aos(scf);
    assert_eq!(
        nao, nao_cart,
        "PySOC export: Cartesian shell-size count mismatch: {} vs {}",
        nao, nao_cart
    );

    // 3. MO coefficients: active orbitals only, column-major flat.
    //    For a spheric REST calculation, expand each spherical MO into the
    //    Cartesian AO basis first (C_cart = C2S * C_sph). For a Cartesian REST
    //    calculation, scale D/F mixed components to match molsoc's individual
    //    normalization (the C2S expansion already produces molsoc-normalized
    //    Cartesian components for the spheric case).
    let mo_coefficients = export_mo_coefficients(scf, start_mo, n_active, nao);

    // 4. MO energies: Hartree, active orbitals only (soc_td converts to eV internally)
    let mo_energies_hartree = export_mo_energies(scf, start_mo, n_active, method);

    // 5. Excitation energies: convert Hartree → eV
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

    // 6. CI coefficients: normalize + transpose + alpha/beta conversion
    let (ci_singlet_xpy, ci_singlet_xmy) =
        convert_ci_set(singlet_excitations, tda, occ_size, vir_size, true);
    let (ci_triplet_xpy, ci_triplet_xmy) =
        convert_ci_set(triplet_excitations, tda, occ_size, vir_size, false);

    let export = PysocExport {
        program: "REST".to_string(),
        method: method.to_string(),
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

/// Number of Cartesian AOs corresponding to `scf.mol.basis4elem`.
///
/// This equals `scf.mol.num_basis` for a Cartesian REST calculation; for a
/// spheric calculation it is the size of the Cartesian basis exported to PySOC.
fn count_cartesian_aos(scf: &SCF) -> usize {
    scf.mol
        .basis4elem
        .iter()
        .flat_map(|bas4elem| bas4elem.electron_shells.iter())
        .map(|shell| {
            let l = shell.angular_momentum[0] as usize;
            let n_cart = ((l + 1) * (l + 2) / 2) as usize;
            n_cart * shell.native_coefficients.len()
        })
        .sum()
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
/// Returns `nao * num_state` values where `nao` is the number of exported
/// Cartesian AOs. Column `j` (0-based) corresponds to MO `start_mo + j`.
/// Layout matches soc_td's `mo_coeff.dat`:
/// [MO1_AO1, MO1_AO2, ..., MO1_AOnao, MO2_AO1, ...].
///
/// For `basis_type = "spheric"` the spherical MO column is first expanded to
/// the Cartesian basis (see `expand_spheric_mo_to_cartesian`).
///
/// **Cartesian normalization scaling**: libcint uses uniform normalization where all Cartesian
/// components of a shell share the same `gto_norm(l, alpha)`. This makes mixed components
/// like d_xy have self-overlap 1/3 (not 1). molsoc normalizes each component independently,
/// giving self-overlap 1 for all. We scale MO coefficients for mixed components to bridge
/// this difference.
fn export_mo_coefficients(scf: &SCF, start_mo: usize, num_state: usize, nao: usize) -> Vec<f64> {
    let eigenvectors = &scf.eigenvectors[0];
    let nao_internal = scf.mol.num_basis;

    // Build per-AO scaling factors for Cartesian normalization conversion.
    let scales = build_ao_scales(scf, nao);

    let mut mo_coefficients = Vec::with_capacity(nao * num_state);

    for j in start_mo..start_mo + num_state {
        let col_start = j * nao_internal;
        let col_end = (j + 1) * nao_internal;
        let mo_column = match scf.mol.cint_type {
            CintType::Cartesian => eigenvectors.data[col_start..col_end].to_vec(),
            CintType::Spheric => {
                expand_spheric_mo_to_cartesian(scf, &eigenvectors.data[col_start..col_end], nao)
            }
            CintType::Spinor => {
                panic!("PySOC export does not support spinor basis sets.")
            }
        };
        assert_eq!(mo_column.len(), nao);
        if matches!(scf.mol.cint_type, CintType::Cartesian) {
            // libcint's Cartesian D/F components use one uniform gto_norm
            // per shell, whereas molsoc normalizes every Cartesian component
            // individually. The C2S expansion for a spheric calculation
            // already yields molsoc's individually-normalized Cartesian
            // components, so no extra scaling is applied in that case.
            mo_coefficients.extend(
                mo_column
                    .iter()
                    .zip(scales.iter())
                    .map(|(&coeff, &scale)| coeff * scale),
            );
        } else {
            mo_coefficients.extend(mo_column);
        }
    }

    mo_coefficients
}

/// Expand one spherical MO coefficient column into the Cartesian AO basis.
///
/// The block-diagonal transformation matrix has one C2S block per contracted
/// shell; the blocks follow the same shell order and Cartesian component order
/// as `export_basis_gaussian` and molsoc.
fn expand_spheric_mo_to_cartesian(scf: &SCF, sph_mo: &[f64], nao_cart: usize) -> Vec<f64> {
    assert_eq!(sph_mo.len(), scf.mol.num_basis);
    let mut cart_mo = vec![0.0; nao_cart];
    let mut sph_offset = 0usize;
    let mut cart_offset = 0usize;

    for bas4elem in &scf.mol.basis4elem {
        for shell in &bas4elem.electron_shells {
            let l = shell.angular_momentum[0] as usize;
            let sph_dim = 2 * l + 1;
            let cart_dim = (l + 1) * (l + 2) / 2;
            let c2s_owned = c2s_matrix_const(l);
            let c2s = c2s_owned.to_matrixfullslice();

            for _ in 0..shell.native_coefficients.len() {
                for cart in 0..cart_dim {
                    let mut value = 0.0;
                    for sph in 0..sph_dim {
                        value += *c2s
                            .get2d([cart, sph])
                            .expect("PySOC export: invalid C2S index")
                            * sph_mo[sph_offset + sph];
                    }
                    cart_mo[cart_offset + cart] = value;
                }
                sph_offset += sph_dim;
                cart_offset += cart_dim;
            }
        }
    }

    assert_eq!(
        sph_offset, scf.mol.num_basis,
        "PySOC export: spherical AO count mismatch during C2S expansion"
    );
    assert_eq!(
        cart_offset, nao_cart,
        "PySOC export: Cartesian AO count mismatch during C2S expansion"
    );
    cart_mo
}

/// Build per-AO scaling factors to convert from libcint's uniform Cartesian normalization
/// to molsoc's individual Cartesian normalization.
///
/// For a Cartesian component (a, b, c) with a+b+c=l, the scale factor is:
///   sqrt((2a-1)!! * (2b-1)!! * (2c-1)!! / (2l-1)!!)
///
/// This is 1 for pure components (l,0,0) and <1 for mixed components.
fn build_ao_scales(scf: &SCF, nao_cart: usize) -> Vec<f64> {
    let mut scales = Vec::new();

    for bas4elem in &scf.mol.basis4elem {
        for shell in &bas4elem.electron_shells {
            let l = shell.angular_momentum[0] as usize;
            for _ in &shell.native_coefficients {
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
        nao_cart,
        "AO scale count mismatch: {} vs exported Cartesian nao {}",
        scales.len(),
        nao_cart
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
/// For BSE: uses `gwqp.0` (GW quasiparticle energies).
/// For TDDFT: uses `eigenvalues[0]` (KS eigenvalues).
/// soc_td converts Hartree → eV internally for `qm_flag != 'tddftb'`.
fn export_mo_energies(scf: &SCF, start_mo: usize, num_state: usize, method: &str) -> Vec<f64> {
    let energies = if method == "TDDFT" {
        &scf.eigenvalues[0]
    } else if !scf.gwqp.0.is_empty() {
        &scf.gwqp.0
    } else {
        &scf.eigenvalues[0]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis_io::{BasCell, Basis4Elem};
    use crate::molecule_io::Molecule;
    use tensors::MatrixFull;

    fn scf_with_shell(cint_type: CintType, l: usize, sph_mo: Vec<f64>) -> SCF {
        let n_sph = 2 * l + 1;
        let shell = BasCell {
            function_type: None,
            region: None,
            angular_momentum: vec![l as i32],
            exponents: vec![1.0],
            coefficients: vec![vec![1.0]],
            native_coefficients: vec![vec![1.0]],
        };
        let mut mol = Molecule::init_mol();
        mol.cint_type = cint_type;
        mol.num_basis = if matches!(cint_type, CintType::Spheric) {
            n_sph
        } else {
            (l + 1) * (l + 2) / 2
        };
        mol.num_state = mol.num_basis;
        mol.basis4elem = vec![Basis4Elem {
            electron_shells: vec![shell],
            references: None,
            ecp_potentials: None,
            ecp_electrons: None,
            global_index: (0, 0),
        }];
        let mut scf = SCF::init_scf(&mol);
        scf.eigenvectors[0] = MatrixFull::from_vec([mol.num_basis, 1], sph_mo).unwrap();
        scf
    }

    #[test]
    fn spheric_d_shell_is_expanded_to_molsoc_cartesian_order() {
        let scf = scf_with_shell(CintType::Spheric, 2, vec![1.0, 0.0, 0.0, 0.0, 0.0]);
        let cart = expand_spheric_mo_to_cartesian(&scf, &scf.eigenvectors[0].data, 6);
        // REST's d(-2) component corresponds to the Cartesian xy component.
        let expected = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        for (got, want) in cart.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1.0e-12, "{} != {}", got, want);
        }

        // Spheric MOs are already in molsoc's individual Cartesian
        // normalization after C2S expansion, so no further scaling occurs.
        let exported = export_mo_coefficients(&scf, 0, 1, 6);
        assert_eq!(exported, cart);
    }

    #[test]
    fn cartesian_d_mixed_component_is_scaled_for_molsoc() {
        let scf = scf_with_shell(CintType::Cartesian, 2, vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let exported = export_mo_coefficients(&scf, 0, 1, 6);
        let scale = 1.0_f64 / 3.0_f64.sqrt();
        let expected = [0.0, scale, 0.0, 0.0, 0.0, 0.0];
        for (got, want) in exported.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1.0e-12, "{} != {}", got, want);
        }
    }
}
