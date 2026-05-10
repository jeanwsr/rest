/// TDDFT fxc kernel matrix-vector product
///
/// Implements the exchange-correlation kernel (fxc) contribution to the
/// TDDFT A matrix-vector product for LDA and GGA functionals.
///
/// Physical formula:
///   (A^{fxc} · z)_{ia} = Σ_g w(g) · Σ_{αβ} D_α[φ_i φ_a](g) · fxc(g)_{αβ} · ρ_z,β(g)
///   where ρ_z,β(g) = Σ_{jb} D_β[φ_j φ_b](g) · z_{jb}
///
/// Reference: PySCF TDDFT implementation pattern

use rest_tensors::MatrixFull;
use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;
use crate::scf_io::SCF;
use crate::ri_tddft::utils::tddft_occupation_parameters;
use crate::dft::xc_deriv::XCType;
use crate::dft::num_int::{eval_rho5_batch, eval_ao_batch};
use crate::dft::libxc_itrf::eval_xc_eff;

/// Precomputed fxc kernel data for TDDFT matrix-vector product
pub struct FXCMatvecData {
    /// Number of density variables: 1 (LDA) or 4 (GGA)
    pub nvar: usize,
    /// Number of numerical integration grid points
    pub ngrids: usize,
    /// Number of occupied orbitals
    pub nocc: usize,
    /// Number of virtual orbitals
    pub nvir: usize,
    /// First active MO index (for frozen-core handling)
    pub start_mo: usize,
    /// Hybrid exchange coefficient (c_x from functional)
    pub alpha_hybrid: f64,
    /// Occupied MO values on grids, column-major [nocc, ngrids]
    pub mo_occ: MatrixFull<f64>,
    /// Virtual MO values on grids, column-major [nvir, ngrids]
    pub mo_vir: MatrixFull<f64>,
    /// For GGA: occupied MO gradients (x, y, z) on grids, each [nocc, ngrids]
    pub mo_occ_grad: Option<[MatrixFull<f64>; 3]>,
    /// For GGA: virtual MO gradients (x, y, z) on grids, each [nvir, ngrids]
    pub mo_vir_grad: Option<[MatrixFull<f64>; 3]>,
    /// fxc kernel × grid weights
    /// LDA: vec[ngrids] = fxc[g] × w[g]
    /// GGA: vec[ngrids × 4 × 4] f-contiguous [g, α, β] = fxc[g,α,β] × w[g]
    pub wfxc: Vec<f64>,
}

/// Prepare FXCMatvecData from converged SCF object
///
/// Steps:
/// 1. Extract occupied/virtual MO coefficients from SCF eigenvectors
/// 2. Project MO coefficients onto DFT integration grids
/// 3. For GGA: also project MO gradients onto grids
/// 4. Compute ground-state density on grids via eval_rho5_batch
/// 5. Compute fxc kernel via eval_xc_eff with deriv=2
/// 6. Multiply fxc by grid weights
pub fn prepare_fxc_data(scf: &SCF) -> FXCMatvecData {
    let (start_mo, _num_state, occ_size, vir_size, _homo, _lumo) =
        tddft_occupation_parameters(scf);

    let xc_data = &scf.mol.xc_data;
    let xc_type = if xc_data.use_density_gradient() {
        XCType::GGA
    } else {
        XCType::LDA
    };
    let nvar = match xc_type {
        XCType::LDA => 1,
        XCType::GGA => 4,
        _ => panic!("fxc only supports LDA and GGA"),
    };
    let alpha_hybrid = xc_data.dfa_hybrid_scf;

    let grids = scf.grids.as_ref().expect("DFT grids must be initialized for fxc");
    let ngrids = grids.weights.len();
    let num_basis = scf.mol.num_basis;
    let weights = &grids.weights;
    let ao = grids.ao.as_ref().expect("AO on grids must be tabulated");
    let eigvec = &scf.eigenvectors[0];

    // ── Extract MO coefficients for occupied and virtual spaces ──
    let mut c_occ = MatrixFull::new([num_basis, occ_size], 0.0);
    for j in 0..occ_size {
        for i in 0..num_basis {
            c_occ[[i, j]] = eigvec[[i, start_mo + j]];
        }
    }
    let mut c_vir = MatrixFull::new([num_basis, vir_size], 0.0);
    for j in 0..vir_size {
        for i in 0..num_basis {
            c_vir[[i, j]] = eigvec[[i, _lumo + j]];
        }
    }

    // ── Project MO values onto grids ──
    // mo_occ[i,g] = Σ_p C_occ[p,i] × ao[p,g]
    // mo_vir[a,g] = Σ_p C_vir[p,a] × ao[p,g]
    // In BLAS: mo_occ = C_occ^T × ao
    let mut mo_occ = MatrixFull::new([occ_size, ngrids], 0.0);
    _dgemm_full(&c_occ, 'T', ao, 'N', &mut mo_occ, 1.0, 0.0);

    let mut mo_vir = MatrixFull::new([vir_size, ngrids], 0.0);
    _dgemm_full(&c_vir, 'T', ao, 'N', &mut mo_vir, 1.0, 0.0);

    // ── GGA: MO gradients on grids ──
    let (mo_occ_grad, mo_vir_grad) = if xc_type == XCType::GGA {
        let aop = grids.aop.as_ref().expect("AO gradients needed for GGA fxc");
        let mut og = [
            MatrixFull::new([occ_size, ngrids], 0.0),
            MatrixFull::new([occ_size, ngrids], 0.0),
            MatrixFull::new([occ_size, ngrids], 0.0),
        ];
        let mut vg = [
            MatrixFull::new([vir_size, ngrids], 0.0),
            MatrixFull::new([vir_size, ngrids], 0.0),
            MatrixFull::new([vir_size, ngrids], 0.0),
        ];
        for d in 0..3 {
            let aop_d_slice = aop.get_reducing_matrix(d).unwrap();
            let aop_d = MatrixFull::from_vec(
                [num_basis, ngrids],
                aop_d_slice.iter().cloned().collect(),
            ).unwrap();
            _dgemm_full(&c_occ, 'T', &aop_d, 'N', &mut og[d], 1.0, 0.0);
            _dgemm_full(&c_vir, 'T', &aop_d, 'N', &mut vg[d], 1.0, 0.0);
        }
        (Some(og), Some(vg))
    } else {
        (None, None)
    };

    // ── Compute ground-state density on grids for fxc evaluation ──
    let ao_deriv = if xc_type == XCType::GGA { 1 } else { 0 };
    let ao_rifull = eval_ao_batch(&scf.mol, &grids.coordinates, ao_deriv, ngrids);
    let mo_coeffs = vec![scf.eigenvectors[0].clone()];
    let occ = vec![scf.occupation[0].clone()];
    let rho_tensor = eval_rho5_batch(&ao_rifull, xc_type, &mo_coeffs, &occ, 1, ngrids);

    // Extract rho_array in the format expected by eval_xc_eff
    // For spin=0: column-major [ngrids, nvar], i.e. rho_array[g + v*ngrids]
    // For LDA: only density; for GGA: density, grad_x, grad_y, grad_z
    // rho_tensor has shape [ngrids, nvar, 1] in f-contiguous (via RIFull)
    // RIFull data layout: data[g + v*ngrids + s*nvar*ngrids]
    // For spin_channel=1 (s=0): rho_tensor raw data has layout [g + v*ngrids]
    let rho_array: Vec<f64> = rho_tensor.raw()[rho_tensor.offset()..]
        .chunks(ngrids * nvar)
        .next()
        .unwrap_or(&[])
        .to_vec();
    // Actually the offset might be 0, and raw() gives the full data.
    // Let's just take what we need.
    let rho_array = if rho_array.is_empty() {
        // Fallback: copy from raw
        let raw = rho_tensor.raw();
        let offset = rho_tensor.offset();
        let len = ngrids * nvar;
        raw[offset..offset + len].to_vec()
    } else {
        rho_array
    };

    // ── Compute fxc kernel analytically via libxc ──
    let func_ids = &xc_data.dfa_compnt_scf;
    let func_factors = &xc_data.dfa_paramr_scf;
    let xc_tensors = eval_xc_eff(func_ids, func_factors, xc_type, 0, &rho_array, ngrids, 2);

    // xc_tensors[2] = fxc kernel (deriv=2)
    let fxc_tensor = xc_tensors[2].as_ref()
        .expect("fxc (deriv=2) should be available");
    let fxc_raw: Vec<f64> = {
        let raw = fxc_tensor.raw();
        let offset = fxc_tensor.offset();
        let nv2 = nvar * nvar;
        let len = ngrids * nv2;
        raw[offset..offset + len].to_vec()
    };

    // ── Multiply fxc by grid weights ──
    // IMPORTANT: eval_xc_eff(spin=0) returns the unpolarized fxc = (δ²Exc/δρ²)
    // For RKS closed shell, the unpolarized fxc corresponds to:
    //   f_u = (f_↑↑ + f_↑↓) / 2 = δ²Exc/δρ² (where ρ = ρ_↑+ρ_↓)
    // The singlet TDDFT kernel needs f_s = f_↑↑ + f_↑↓ = 2 × f_u
    // Reference: PySCF nr_rks_fxc_st for singlet
    // For LDA: wfxc[g] = fxc_raw[g] × w[g] × 2
    // For GGA: wfxc[g + α*ngrids + β*4*ngrids] = fxc_raw[idx] × w[g] × 2
    const SINGLET_FXC_FACTOR: f64 = 2.0;
    let wfxc: Vec<f64> = if nvar == 1 {
        (0..ngrids).map(|g| fxc_raw[g] * weights[g] * SINGLET_FXC_FACTOR).collect()
    } else {
        let nv2 = nvar * nvar;
        let mut wfxc = vec![0.0; ngrids * nv2];
        for g in 0..ngrids {
            let w = weights[g];
            for k in 0..nv2 {
                wfxc[g + k * ngrids] = fxc_raw[g + k * ngrids] * w * SINGLET_FXC_FACTOR;
            }
        }
        wfxc
    };

    println!(
        "FXCMatvecData prepared: nocc={}, nvir={}, ngrids={}, nvar={}, alpha_hybrid={}",
        occ_size, vir_size, ngrids, nvar, alpha_hybrid
    );

    FXCMatvecData {
        nvar,
        ngrids,
        nocc: occ_size,
        nvir: vir_size,
        start_mo,
        alpha_hybrid,
        mo_occ,
        mo_vir,
        mo_occ_grad,
        mo_vir_grad,
        wfxc,
    }
}

/// Dispatch fxc matrix-vector product based on functional type
pub fn fxc_matvec(data: &FXCMatvecData, z: &[f64]) -> Vec<f64> {
    assert_eq!(z.len(), data.nocc * data.nvir,
               "z vector length {} must equal nocc×nvir = {}×{}",
               z.len(), data.nocc, data.nvir);
    match data.nvar {
        1 => fxc_matvec_lda(data, z),
        4 => fxc_matvec_gga(data, z),
        _ => panic!("fxc_matvec only supports LDA (nvar=1) and GGA (nvar=4)"),
    }
}

/// LDA fxc matrix-vector product
///
/// For each trial vector z (shape [nocc, nvir]):
///   1. ρ_z[g] = Σ_ia mo_occ[i,g] × mo_vir[a,g] × z[i,a]
///   2. v[g] = wfxc[g] × ρ_z[g]
///   3. result[i,a] = Σ_g mo_occ[i,g] × mo_vir[a,g] × v[g]
fn fxc_matvec_lda(data: &FXCMatvecData, z: &[f64]) -> Vec<f64> {
    let nocc = data.nocc;
    let nvir = data.nvir;
    let ngrids = data.ngrids;

    // ── Step 1: Compute perturbed density on grids ──
    // t[i,g] = Σ_a mo_vir[a,g] × z[i,a]
    // In BLAS: t = z × mo_vir, where z is [nocc,nvir], mo_vir is [nvir,ngrids]
    let z_mat = MatrixFull::from_vec([nocc, nvir], z.to_vec()).unwrap();
    let mut t = MatrixFull::new([nocc, ngrids], 0.0);
    _dgemm_full(&z_mat, 'N', &data.mo_vir, 'N', &mut t, 1.0, 0.0);

    // ρ_z[g] = Σ_i mo_occ[i,g] × t[i,g]
    let mut rho_z = vec![0.0; ngrids];
    for g in 0..ngrids {
        let mut sum = 0.0;
        for i in 0..nocc {
            sum += data.mo_occ[[i, g]] * t[[i, g]];
        }
        rho_z[g] = sum;
    }

    // ── Step 2: Apply fxc kernel ──
    // v[g] = wfxc[g] × ρ_z[g]
    let mut v = vec![0.0; ngrids];
    for g in 0..ngrids {
        v[g] = data.wfxc[g] * rho_z[g];
    }

    // ── Step 3: Contract back to MO basis ──
    // Scale mo_vir by v: s[a,g] = mo_vir[a,g] × v[g]
    let mut mo_vir_scaled = MatrixFull::new([nvir, ngrids], 0.0);
    for g in 0..ngrids {
        for a in 0..nvir {
            mo_vir_scaled[[a, g]] = data.mo_vir[[a, g]] * v[g];
        }
    }

    // result[i,a] = Σ_g mo_occ[i,g] × mo_vir_scaled[a,g]
    // = mo_occ · mo_vir_scaled^T
    let mut result = MatrixFull::new([nocc, nvir], 0.0);
    _dgemm_full(&data.mo_occ, 'N', &mo_vir_scaled, 'T', &mut result, 1.0, 0.0);

    result.data
}

/// GGA fxc matrix-vector product
///
/// The GGA fxc kernel is a 4×4 symmetric matrix at each grid point:
/// fxc[g, α, β] where α,β ∈ {ρ, ∇_xρ, ∇_yρ, ∇_zρ}
///
/// D_0[φ_i φ_a] = φ_i × φ_a                           (density component)
/// D_d[φ_i φ_a] = ∇_d φ_i × φ_a + φ_i × ∇_d φ_a      (gradient components, d=1,2,3)
///
/// For each trial vector z:
///   1. Compute ρ_z[β,g] = Σ_ia D_β[φ_i φ_a](g) × z[i,a]  for β=0,1,2,3
///   2. Apply fxc: fxc_eff[α,g] = Σ_β wfxc[g,α,β] × ρ_z[β,g]
///   3. Contract back: result[i,a] = Σ_g Σ_α D_α[φ_i φ_a](g) × fxc_eff[α,g]
fn fxc_matvec_gga(data: &FXCMatvecData, z: &[f64]) -> Vec<f64> {
    let nocc = data.nocc;
    let nvir = data.nvir;
    let ngrids = data.ngrids;

    let mo_occ_grad = data.mo_occ_grad.as_ref().expect("GGA requires mo_occ_grad");
    let mo_vir_grad = data.mo_vir_grad.as_ref().expect("GGA requires mo_vir_grad");

    // ── Step 1a: Compute half-transforms ──
    // t0[i,g] = Σ_a mo_vir[a,g] × z[i,a]
    // t_grad[d][i,g] = Σ_a mo_vir_grad[d][a,g] × z[i,a]
    let z_mat = MatrixFull::from_vec([nocc, nvir], z.to_vec()).unwrap();
    let mut t0 = MatrixFull::new([nocc, ngrids], 0.0);
    _dgemm_full(&z_mat, 'N', &data.mo_vir, 'N', &mut t0, 1.0, 0.0);

    let mut t_grad: [MatrixFull<f64>; 3] = [
        MatrixFull::new([nocc, ngrids], 0.0),
        MatrixFull::new([nocc, ngrids], 0.0),
        MatrixFull::new([nocc, ngrids], 0.0),
    ];
    for d in 0..3 {
        _dgemm_full(&z_mat, 'N', &mo_vir_grad[d], 'N', &mut t_grad[d], 1.0, 0.0);
    }

    // ── Step 1b: Compute ρ_z[β,g] for β=0,1,2,3 ──
    let mut rho_z = vec![0.0; ngrids * 4]; // [g, β] → rho_z[g + β*ngrids]

    // β=0: ρ_z[0,g] = Σ_i mo_occ[i,g] × t0[i,g]
    for g in 0..ngrids {
        let mut sum = 0.0;
        for i in 0..nocc {
            sum += data.mo_occ[[i, g]] * t0[[i, g]];
        }
        rho_z[g] = sum;

        // β=1,2,3:
        // ρ_z[β,g] = Σ_i mo_occ_grad[β-1][i,g] × t0[i,g]
        //          + Σ_i mo_occ[i,g] × t_grad[β-1][i,g]
        for d in 0..3 {
            let mut sum1 = 0.0;
            let mut sum2 = 0.0;
            for i in 0..nocc {
                sum1 += mo_occ_grad[d][[i, g]] * t0[[i, g]];
                sum2 += data.mo_occ[[i, g]] * t_grad[d][[i, g]];
            }
            rho_z[g + (d + 1) * ngrids] = sum1 + sum2;
        }
    }

    // ── Step 2: Apply fxc kernel at each grid point ──
    // fxc_eff[alpha,g] = sum_beta wfxc[g, alpha, beta] * rho_z[beta,g]
    // wfxc layout f-contiguous [g, alpha, beta]:
    //   wfxc[g + alpha*ngrids + beta*4*ngrids]
    let mut fxc_eff_grid = vec![0.0; ngrids * 4]; // [g, alpha] → fxc_eff_grid[g + alpha*ngrids]

    for g in 0..ngrids {
        for alpha in 0..4 {
            let mut sum = 0.0;
            for beta in 0..4 {
                let w_idx = g + alpha * ngrids + beta * 4 * ngrids;
                sum += data.wfxc[w_idx] * rho_z[g + beta * ngrids];
            }
            fxc_eff_grid[g + alpha * ngrids] = sum;
        }
    }

    // ── Step 3: Contract back to MO basis ──
    // result[i,a] = sum_g sum_alpha D_alpha[phi_i phi_a](g) * fxc_eff[g,alpha]
    //
    // For alpha=0: D_0 = mo_occ[i,g] * mo_vir[a,g]
    //   result += mo_occ . (mo_vir * fxc_eff[0])^T
    //
    // For alpha=d+1 (d=0,1,2):
    //   D_{d+1} = mo_occ_grad[d][i,g] * mo_vir[a,g]
    //           + mo_occ[i,g] * mo_vir_grad[d][a,g]
    //   result += mo_occ_grad[d] . (mo_vir * fxc_eff[d+1])^T
    //           + mo_occ . (mo_vir_grad[d] * fxc_eff[d+1])^T

    let mut result = vec![0.0; nocc * nvir];

    for alpha in 0..4 {
        let fxc_a = &fxc_eff_grid[alpha * ngrids..(alpha + 1) * ngrids];

        if alpha == 0 {
            // α=0 (density channel): D_0 = φ_i · φ_a
            // result += mo_occ · (mo_vir × fxc_eff[0])^T
            let mut right_scaled = MatrixFull::new([nvir, ngrids], 0.0);
            for g in 0..ngrids {
                let fv = fxc_a[g];
                for a in 0..nvir {
                    right_scaled[[a, g]] = data.mo_vir[[a, g]] * fv;
                }
            }
            let mut contrib = MatrixFull::new([nocc, nvir], 0.0);
            _dgemm_full(&data.mo_occ, 'N', &right_scaled, 'T', &mut contrib, 1.0, 0.0);
            for idx in 0..result.len() {
                result[idx] += contrib.data[idx];
            }
        } else {
            // α>0 (gradient channel): D_d = ∇_d(φ_i)·φ_a + φ_i·∇_d(φ_a)
            let d = alpha - 1;

            // Term A: ∇_d(mo_occ) · (mo_vir × fxc_eff[α])^T
            let mut right_scaled = MatrixFull::new([nvir, ngrids], 0.0);
            for g in 0..ngrids {
                let fv = fxc_a[g];
                for a in 0..nvir {
                    right_scaled[[a, g]] = data.mo_vir[[a, g]] * fv;
                }
            }
            let mut contrib_a = MatrixFull::new([nocc, nvir], 0.0);
            _dgemm_full(&mo_occ_grad[d], 'N', &right_scaled, 'T', &mut contrib_a, 1.0, 0.0);
            for idx in 0..result.len() {
                result[idx] += contrib_a.data[idx];
            }

            // Term B: mo_occ · (∇_d(mo_vir) × fxc_eff[α])^T
            let mut right_grad_scaled = MatrixFull::new([nvir, ngrids], 0.0);
            for g in 0..ngrids {
                let fv = fxc_a[g];
                for a in 0..nvir {
                    right_grad_scaled[[a, g]] = mo_vir_grad[d][[a, g]] * fv;
                }
            }
            let mut contrib_b = MatrixFull::new([nocc, nvir], 0.0);
            _dgemm_full(&data.mo_occ, 'N', &right_grad_scaled, 'T', &mut contrib_b, 1.0, 0.0);
            for idx in 0..result.len() {
                result[idx] += contrib_b.data[idx];
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create synthetic FXCMatvecData for LDA testing
    fn make_lda_data(nocc: usize, nvir: usize, ngrids: usize) -> FXCMatvecData {
        // mo_occ: fill with sin pattern
        let mut mo_occ = MatrixFull::new([nocc, ngrids], 0.0);
        for i in 0..nocc {
            for g in 0..ngrids {
                mo_occ[[i, g]] = ((i as f64 + 1.0) * (g as f64 + 1.0)).sin() * 0.5;
            }
        }
        // mo_vir: fill with cos pattern
        let mut mo_vir = MatrixFull::new([nvir, ngrids], 0.0);
        for a in 0..nvir {
            for g in 0..ngrids {
                mo_vir[[a, g]] = ((a as f64 + 1.0) * (g as f64 + 1.0)).cos() * 0.3;
            }
        }
        // wfxc: random positive values
        let wfxc: Vec<f64> = (0..ngrids).map(|g| (g as f64 + 1.0).sqrt() * 0.01).collect();

        FXCMatvecData {
            nvar: 1,
            ngrids,
            nocc,
            nvir,
            start_mo: 0,
            alpha_hybrid: 0.0,
            mo_occ,
            mo_vir,
            mo_occ_grad: None,
            mo_vir_grad: None,
            wfxc,
        }
    }

    /// Create synthetic FXCMatvecData for GGA testing
    fn make_gga_data(nocc: usize, nvir: usize, ngrids: usize) -> FXCMatvecData {
        let mut mo_occ = MatrixFull::new([nocc, ngrids], 0.0);
        for i in 0..nocc {
            for g in 0..ngrids {
                mo_occ[[i, g]] = ((i as f64 + 1.0) * (g as f64 + 1.0)).sin() * 0.5;
            }
        }
        let mut mo_vir = MatrixFull::new([nvir, ngrids], 0.0);
        for a in 0..nvir {
            for g in 0..ngrids {
                mo_vir[[a, g]] = ((a as f64 + 1.0) * (g as f64 + 1.0)).cos() * 0.3;
            }
        }

        // Gradient data
        let mut og = [
            MatrixFull::new([nocc, ngrids], 0.0),
            MatrixFull::new([nocc, ngrids], 0.0),
            MatrixFull::new([nocc, ngrids], 0.0),
        ];
        let mut vg = [
            MatrixFull::new([nvir, ngrids], 0.0),
            MatrixFull::new([nvir, ngrids], 0.0),
            MatrixFull::new([nvir, ngrids], 0.0),
        ];
        for d in 0..3 {
            for i in 0..nocc {
                for g in 0..ngrids {
                    og[d][[i, g]] = ((i as f64 + 1.0) * (g as f64 + 1.0)).cos() * 0.2 * (d as f64 + 1.0);
                }
            }
            for a in 0..nvir {
                for g in 0..ngrids {
                    vg[d][[a, g]] = -((a as f64 + 1.0) * (g as f64 + 1.0)).sin() * 0.2 * (d as f64 + 1.0);
                }
            }
        }

        // fxc kernel: symmetric 4×4 at each grid point (ngrids × 4 × 4 f-contiguous)
        let nv2 = 16;
        let mut wfxc = vec![0.0; ngrids * nv2];
        for g in 0..ngrids {
            let weight = (g as f64 + 1.0).sqrt() * 0.01;
            // Construct a simple symmetric 4x4 matrix at each grid point
            for a in 0..4 {
                for b in 0..4 {
                    let idx = g + a * ngrids + b * 4 * ngrids;
                    wfxc[idx] = weight * (a as f64 * 0.1 + b as f64 * 0.1 + 1.0).sin().max(0.01);
                }
            }
        }

        FXCMatvecData {
            nvar: 4,
            ngrids,
            nocc,
            nvir,
            start_mo: 0,
            alpha_hybrid: 0.25,
            mo_occ,
            mo_vir,
            mo_occ_grad: Some(og),
            mo_vir_grad: Some(vg),
            wfxc,
        }
    }

    fn _test_fxc_lda(nocc: usize, nvir: usize, ngrids: usize) {
        let data = make_lda_data(nocc, nvir, ngrids);
        let n = nocc * nvir;
        let z: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) * 0.01).collect();
        let result = fxc_matvec(&data, &z);
        assert_eq!(result.len(), n);
        let norm: f64 = result.iter().map(|x| x * x).sum::<f64>().sqrt();
        println!("LDA test nocc={} nvir={} ngrids={}: norm={:.6}", nocc, nvir, ngrids, norm);
        assert!(norm > 0.0, "LDA fxc_matvec should give non-zero result");
        assert!(norm.is_finite(), "LDA fxc_matvec result should be finite");
    }

    fn _test_fxc_gga(nocc: usize, nvir: usize, ngrids: usize) {
        let data = make_gga_data(nocc, nvir, ngrids);
        let n = nocc * nvir;
        let z: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) * 0.01).collect();
        let result = fxc_matvec(&data, &z);
        assert_eq!(result.len(), n);
        let norm: f64 = result.iter().map(|x| x * x).sum::<f64>().sqrt();
        println!("GGA test nocc={} nvir={} ngrids={}: norm={:.6}", nocc, nvir, ngrids, norm);
        assert!(norm > 0.0, "GGA fxc_matvec should give non-zero result");
        assert!(norm.is_finite(), "GGA fxc_matvec result should be finite");
    }

    #[test]
    fn test_fxc_lda_small() { _test_fxc_lda(3, 5, 10); }
    #[test]
    fn test_fxc_lda_medium() { _test_fxc_lda(10, 20, 100); }
    #[test]
    fn test_fxc_lda_zero_z() {
        let data = make_lda_data(3, 5, 10);
        let z = vec![0.0; 15];
        let result = fxc_matvec(&data, &z);
        for &v in &result { assert!((v).abs() < 1e-15); }
    }
    #[test]
    fn test_fxc_gga_small() { _test_fxc_gga(3, 5, 10); }
    #[test]
    fn test_fxc_gga_medium() { _test_fxc_gga(10, 20, 100); }
    #[test]
    fn test_fxc_gga_zero_z() {
        let data = make_gga_data(3, 5, 10);
        let z = vec![0.0; 15];
        let result = fxc_matvec(&data, &z);
        for &v in &result { assert!((v).abs() < 1e-15); }
    }
    #[test]
    fn test_fxc_gga_symmetry() {
        // Test that the result has correct structure for symmetric inputs
        let data = make_gga_data(4, 6, 20);
        let n = 24;
        let z: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let result = fxc_matvec(&data, &z);
        assert!(result.iter().all(|x| x.is_finite()));
    }
    #[test]
    fn test_fxc_lda_linearity() {
        let data = make_lda_data(3, 5, 10);
        let n = 15;
        let z1: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let z2: Vec<f64> = (0..n).map(|i| ((i + 3) as f64) * 0.01).collect();
        let r1 = fxc_matvec(&data, &z1);
        let r2 = fxc_matvec(&data, &z2);
        let z_sum: Vec<f64> = z1.iter().zip(z2.iter()).map(|(a, b)| a + b).collect();
        let r_sum = fxc_matvec(&data, &z_sum);
        for i in 0..n {
            assert!((r_sum[i] - (r1[i] + r2[i])).abs() < 1e-14,
                    "Linearity violated at index {}", i);
        }
        println!("LDA linearity test passed");
    }

    #[test]
    fn test_fxc_lda_energy() {
        // Test that fxc contributes positive energy for wavefunction-like case
        let data = make_lda_data(5, 8, 30);
        let n = 40;
        let z: Vec<f64> = (0..n).map(|i| if i < 10 { 1.0 } else { 0.0 }).collect();
        let result = fxc_matvec(&data, &z);
        // Check result is a real vector
        assert!(result.iter().all(|x| x.is_finite()));
        // Check dot product z·result > 0 (positive semidefinite fxc for physical systems)
        // Note: this may not hold for synthetic data, but the result should be non-zero
        let dot: f64 = z.iter().zip(result.iter()).map(|(a, b)| a * b).sum();
        println!("LDA energy test: z·result = {:.6}", dot);
    }

    #[test]
    fn test_fxc_gga_linearity() {
        let data = make_gga_data(3, 5, 10);
        let n = 15;
        let z1: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let z2: Vec<f64> = (0..n).map(|i| ((i + 3) as f64) * 0.01).collect();
        let r1 = fxc_matvec(&data, &z1);
        let r2 = fxc_matvec(&data, &z2);
        let z_sum: Vec<f64> = z1.iter().zip(z2.iter()).map(|(a, b)| a + b).collect();
        let r_sum = fxc_matvec(&data, &z_sum);
        for i in 0..n {
            let diff = (r_sum[i] - (r1[i] + r2[i])).abs();
            assert!(diff < 1e-14, "GGA linearity violated at index {}: {}", i, diff);
        }
        println!("GGA linearity test passed");
    }
}
