use tensors::{MatrixFull, MathMatrix};
use rest_tensors::matrix::matrix_blas_lapack::{_dgemm_full, _dgemv};

use crate::ri_tddft::fxc_kernel::TDDFTData;
use crate::ri_bse::matvec::coulomb_contribution;
use crate::dft::xc_deriv::XCType;

/// fxc matvec: computes fxc * z in the (ia) space
pub fn fxc_matvec(data: &TDDFTData, is_singlet: bool, z: &Vec<f64>) -> Vec<f64> {
    let occ = data.occ_size;
    let vir = data.vir_size;
    let ng = data.num_grids;
    let nvar = data.nvar;
    let wfxc = if is_singlet { &data.wfxc_singlet } else { &data.wfxc_triplet };

    // Z_mat: [occ, vir] (column-major)
    let z_mat = MatrixFull::from_vec([occ, vir], z.clone()).unwrap();

    // T = Z^T * mo_occ -> [vir, ng]
    let mut t_mat = MatrixFull::new([vir, ng], 0.0);
    _dgemm_full(&z_mat, 'T', &data.mo_occ, 'N', &mut t_mat, 1.0, 0.0);

    // Build transition density components rho_z: [nvar, ng]
    let mut rho_z = vec![0.0; nvar * ng];

    // Component 0: rho_z[0,r] = sum_a T[a,r] * mo_vir[a,r]
    for r in 0..ng {
        let mut val = 0.0;
        for a in 0..vir {
            val += t_mat[[a, r]] * data.mo_vir[[a, r]];
        }
        rho_z[r] = val;
    }

    if nvar > 1 {
        // GGA: gradient components
        let mo_og = data.mo_occ_grad.as_ref().unwrap();
        let mo_vg = data.mo_vir_grad.as_ref().unwrap();
        for d in 0..3 {
            // T_gd = Z^T * mo_occ_grad_d -> [vir, ng]
            let mut t_gd = MatrixFull::new([vir, ng], 0.0);
            _dgemm_full(&z_mat, 'T', &mo_og[d], 'N', &mut t_gd, 1.0, 0.0);
            for r in 0..ng {
                let mut val = 0.0;
                for a in 0..vir {
                    val += t_gd[[a, r]] * data.mo_vir[[a, r]]
                         + t_mat[[a, r]] * mo_vg[d][[a, r]];
                }
                rho_z[(d + 1) * ng + r] = val;
            }
        }
    }

    // Apply fxc kernel: sigma_z[alpha, r] = sum_beta wfxc[r, alpha, beta] * rho_z[beta, r]
    let mut sigma_z = vec![0.0; nvar * ng];
    for r in 0..ng {
        for alpha in 0..nvar {
            let mut val = 0.0;
            for beta in 0..nvar {
                val += wfxc[(r * nvar + alpha) * nvar + beta] * rho_z[beta * ng + r];
            }
            sigma_z[alpha * ng + r] = val;
        }
    }

    // Contract back to (ia) space
    if nvar == 1 {
        // LDA: weighted_vir[a,r] = mo_vir[a,r] * sigma_z[r]
        let mut weighted_vir = MatrixFull::new([vir, ng], 0.0);
        for r in 0..ng {
            for a in 0..vir {
                weighted_vir[[a, r]] = data.mo_vir[[a, r]] * sigma_z[r];
            }
        }
        // result = mo_occ * weighted_vir^T -> [occ, vir]
        let mut result_mat = MatrixFull::new([occ, vir], 0.0);
        _dgemm_full(&data.mo_occ, 'N', &weighted_vir, 'T', &mut result_mat, 1.0, 0.0);
        result_mat.data
    } else {
        // GGA
        let mo_og = data.mo_occ_grad.as_ref().unwrap();
        let mo_vg = data.mo_vir_grad.as_ref().unwrap();
        let mut result_mat = MatrixFull::new([occ, vir], 0.0);

        // Component 0
        let mut wv0 = MatrixFull::new([vir, ng], 0.0);
        for r in 0..ng {
            for a in 0..vir {
                wv0[[a, r]] = data.mo_vir[[a, r]] * sigma_z[r];
            }
        }
        _dgemm_full(&data.mo_occ, 'N', &wv0, 'T', &mut result_mat, 1.0, 0.0);

        // Gradient components: for each direction d,
        // result += mo_occ_grad_d * (mo_vir ⊙ σ_z[d+1])^T
        //         + mo_occ * (mo_vir_grad_d ⊙ σ_z[d+1])^T
        for d in 0..3 {
            let mut wv_d = MatrixFull::new([vir, ng], 0.0);
            for r in 0..ng {
                let sd = sigma_z[(d + 1) * ng + r];
                for a in 0..vir {
                    wv_d[[a, r]] = data.mo_vir[[a, r]] * sd;
                }
            }
            _dgemm_full(&mo_og[d], 'N', &wv_d, 'T', &mut result_mat, 1.0, 1.0);

            let mut wv_gd = MatrixFull::new([vir, ng], 0.0);
            for r in 0..ng {
                let sd = sigma_z[(d + 1) * ng + r];
                for a in 0..vir {
                    wv_gd[[a, r]] = mo_vg[d][[a, r]] * sd;
                }
            }
            _dgemm_full(&data.mo_occ, 'N', &wv_gd, 'T', &mut result_mat, 1.0, 1.0);
        }
        result_mat.data
    }
}

/// Exchange K^A matvec: Σ_{jb} (ij|ab) z_{jb}
/// ri_oo: [occ*num_auxbas, occ] (pre-reshaped)
/// ri_vv: [num_auxbas*vir, vir] (pre-reshaped)
pub fn exchange_a_matvec(
    ri_oo: &MatrixFull<f64>,
    ri_vv: &MatrixFull<f64>,
    occ_size: usize,
    vir_size: usize,
    z: &Vec<f64>,
) -> Vec<f64> {
    let num_auxbas = ri_vv.size[0] / vir_size;
    let z_mat = MatrixFull::from_vec([occ_size, vir_size], z.clone()).unwrap();

    // T = ri_vv * Z^T -> [num_auxbas*vir, occ]
    let mut t_tensor = MatrixFull::new([num_auxbas * vir_size, occ_size], 0.0);
    _dgemm_full(ri_vv, 'N', &z_mat, 'T', &mut t_tensor, 1.0, 0.0);

    // Transpose and reshape: [occ, num_auxbas*vir] -> [occ*num_auxbas, vir]
    t_tensor = t_tensor.transpose_and_drop();
    t_tensor.reshape([occ_size * num_auxbas, vir_size]);

    // result = ri_oo^T * T -> [occ, vir]
    let mut result = MatrixFull::new([occ_size, vir_size], 0.0);
    _dgemm_full(ri_oo, 'T', &t_tensor, 'N', &mut result, 1.0, 0.0);
    result.data
}

/// Exchange K^B matvec: Σ_{jb} (ib|aj) z_{jb}
/// ri_ov_a: [num_auxbas, occ*vir] (standard layout)
/// ri_ov_b: [num_auxbas*occ, vir] (reshaped)
pub fn exchange_b_matvec(
    ri_ov_a: &MatrixFull<f64>,
    ri_ov_b: &MatrixFull<f64>,
    occ_size: usize,
    vir_size: usize,
    z: &Vec<f64>,
) -> Vec<f64> {
    let num_auxbas = ri_ov_a.size[0];
    let z_mat = MatrixFull::from_vec([occ_size, vir_size], z.clone()).unwrap();

    // T = ri_ov_b * Z^T -> [num_auxbas*occ, occ]
    // ri_ov_b[Q*i, b] * Z[j, b]^T = Σ_b ri_ov[Q, i*vir+b] * z[j+b*occ]
    let mut t_tensor = MatrixFull::new([num_auxbas * occ_size, occ_size], 0.0);
    _dgemm_full(ri_ov_b, 'N', &z_mat, 'T', &mut t_tensor, 1.0, 0.0);

    // Transpose block structure: [num_auxbas*occ, occ] -> [num_auxbas*occ, occ]
    // We need to swap the two occ indices
    let mut t_swapped = vec![0.0; num_auxbas * occ_size * occ_size];
    for j in 0..occ_size {
        for q in 0..num_auxbas {
            for i in 0..occ_size {
                // source: t_tensor[q + i*num_auxbas, j]
                // target: t_swapped[q + j*num_auxbas, i]
                t_swapped[(q + j * num_auxbas) + i * num_auxbas * occ_size] =
                    t_tensor[[q + i * num_auxbas, j]];
            }
        }
    }
    let t_swapped = MatrixFull::from_vec(
        [num_auxbas * occ_size, occ_size], t_swapped
    ).unwrap();

    // result = ri_ov_b^T * t_swapped -> [vir, occ] ... no, we need [occ, vir]
    // Actually: result[i, a] = Σ_{Q,j} ri_ov[Q, a*occ+j] * T_swapped[Q*j, i]
    // = Σ_{Q,j} ri_ov_b[Q*j, a]^T * T_swapped[Q*j, i]
    // = (ri_ov_b^T * T_swapped)[a, i] -> need transpose
    let mut result_at = MatrixFull::new([vir_size, occ_size], 0.0);
    _dgemm_full(ri_ov_b, 'T', &t_swapped, 'N', &mut result_at, 1.0, 0.0);

    // Transpose to [occ, vir] layout and flatten
    let mut result = vec![0.0; occ_size * vir_size];
    for i in 0..occ_size {
        for a in 0..vir_size {
            result[i + a * occ_size] = result_at[[a, i]];
        }
    }
    result
}

/// Diagonal energy contribution: (ε_a - ε_i) * z_{ia}
fn energy_diag_matvec(
    eigenvalues: &[f64],
    occ_size: usize,
    vir_size: usize,
    z: &Vec<f64>,
) -> Vec<f64> {
    let mut result = vec![0.0; occ_size * vir_size];
    for a in 0..vir_size {
        for i in 0..occ_size {
            let idx = i + a * occ_size;
            result[idx] = (eigenvalues[occ_size + a] - eigenvalues[i]) * z[idx];
        }
    }
    result
}

/// Full TDA A-block matvec
/// Singlet: A z = (ε_a - ε_i) z + 2*v*z + fxc*z - α*K^A*z
/// Triplet: A z = (ε_a - ε_i) z + fxc_triplet*z - α*K^A*z
pub fn tddft_a_matvec(
    data: &TDDFTData,
    ri_ov: &MatrixFull<f64>,
    ri_oo: &MatrixFull<f64>,
    ri_vv: &MatrixFull<f64>,
    eigenvalues: &[f64],
    is_singlet: bool,
    z: &Vec<f64>,
) -> Vec<f64> {
    let n = data.occ_size * data.vir_size;
    let mut result = energy_diag_matvec(eigenvalues, data.occ_size, data.vir_size, z);

    // fxc contribution
    let fxc_z = fxc_matvec(data, is_singlet, z);
    for i in 0..n {
        result[i] += fxc_z[i];
    }

    // Coulomb (only singlet, factor 2)
    if is_singlet {
        let v_z = coulomb_contribution(ri_ov, z);
        for i in 0..n {
            result[i] += 2.0 * v_z[i];
        }
    }

    // HF exchange (if hybrid)
    if data.alpha_hybrid.abs() > 1.0e-10 {
        let k_z = exchange_a_matvec(ri_oo, ri_vv, data.occ_size, data.vir_size, z);
        for i in 0..n {
            result[i] -= data.alpha_hybrid * k_z[i];
        }
    }

    result
}

/// Full LR (A+B)-block matvec
/// Singlet: (A+B) z = (ε_a - ε_i) z + 4*v*z + 2*fxc*z - α*(K^A + K^B)*z
/// Triplet: (A+B) z = (ε_a - ε_i) z + 2*fxc_triplet*z - α*(K^A + K^B)*z
pub fn tddft_apb_matvec(
    data: &TDDFTData,
    ri_ov: &MatrixFull<f64>,
    ri_oo: &MatrixFull<f64>,
    ri_vv: &MatrixFull<f64>,
    ri_ov_b: &MatrixFull<f64>,
    eigenvalues: &[f64],
    is_singlet: bool,
    z: &Vec<f64>,
) -> Vec<f64> {
    let n = data.occ_size * data.vir_size;
    let mut result = energy_diag_matvec(eigenvalues, data.occ_size, data.vir_size, z);

    // fxc contribution (factor 2)
    let fxc_z = fxc_matvec(data, is_singlet, z);
    for i in 0..n {
        result[i] += 2.0 * fxc_z[i];
    }

    // Coulomb (only singlet, factor 4)
    if is_singlet {
        let v_z = coulomb_contribution(ri_ov, z);
        for i in 0..n {
            result[i] += 4.0 * v_z[i];
        }
    }

    // HF exchange (if hybrid): K^A + K^B
    if data.alpha_hybrid.abs() > 1.0e-10 {
        let ka_z = exchange_a_matvec(ri_oo, ri_vv, data.occ_size, data.vir_size, z);
        let kb_z = exchange_b_matvec(ri_ov, ri_ov_b, data.occ_size, data.vir_size, z);
        for i in 0..n {
            result[i] -= data.alpha_hybrid * (ka_z[i] + kb_z[i]);
        }
    }

    result
}
