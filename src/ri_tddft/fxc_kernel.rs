use tensors::{MatrixFull, MathMatrix};
use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;
use rest_tensors::RIFull;
use rstsr::prelude::*;

use crate::scf_io::SCF;
use crate::dft::Grids;
use crate::dft::xc_deriv::XCType;
use crate::dft::libxc_itrf::eval_xc_eff;
use crate::dft::num_int::{eval_rho5_batch, eval_ao_batch};
use crate::ri_gw::get_occupation_parameters;

pub struct TDDFTData {
    pub xc_type: XCType,
    pub nvar: usize,
    pub num_grids: usize,
    pub occ_size: usize,
    pub vir_size: usize,
    pub start_mo: usize,
    pub alpha_hybrid: f64,
    pub mo_occ: MatrixFull<f64>,
    pub mo_vir: MatrixFull<f64>,
    pub mo_occ_grad: Option<[MatrixFull<f64>; 3]>,
    pub mo_vir_grad: Option<[MatrixFull<f64>; 3]>,
    pub wfxc_singlet: Vec<f64>,
    pub wfxc_triplet: Vec<f64>,
}

pub fn prepare_tddft_data(scf_data: &SCF) -> TDDFTData {
    let (start_mo, num_state, occ_size, vir_size, homo, lumo) =
        get_occupation_parameters(scf_data, 'N');

    let xc_data = &scf_data.mol.xc_data;
    let xc_type = if xc_data.use_density_gradient() {
        XCType::GGA
    } else {
        XCType::LDA
    };
    let nvar = match xc_type {
        XCType::LDA => 1,
        XCType::GGA => 4,
        _ => panic!("TDDFT fxc only supports LDA and GGA"),
    };
    let alpha_hybrid = xc_data.dfa_hybrid_scf;
    let func_ids = &xc_data.dfa_compnt_scf;
    let func_factors = &xc_data.dfa_paramr_scf;

    let grids = scf_data.grids.as_ref().expect("Grids must be initialized");
    let num_grids = grids.weights.len();
    let weights = &grids.weights;
    let ao = grids.ao.as_ref().expect("AO on grids must be tabulated");
    let num_basis = scf_data.mol.num_basis;
    let eigvec = &scf_data.eigenvectors[0];

    // Extract MO coefficients
    let mut c_occ = MatrixFull::new([num_basis, occ_size], 0.0);
    for j in 0..occ_size {
        for i in 0..num_basis {
            c_occ[[i, j]] = eigvec[[i, start_mo + j]];
        }
    }
    let mut c_vir = MatrixFull::new([num_basis, vir_size], 0.0);
    for j in 0..vir_size {
        for i in 0..num_basis {
            c_vir[[i, j]] = eigvec[[i, lumo + j]];
        }
    }

    // MO on grids: C^T * ao
    let mut mo_occ = MatrixFull::new([occ_size, num_grids], 0.0);
    _dgemm_full(&c_occ, 'T', ao, 'N', &mut mo_occ, 1.0, 0.0);
    let mut mo_vir = MatrixFull::new([vir_size, num_grids], 0.0);
    _dgemm_full(&c_vir, 'T', ao, 'N', &mut mo_vir, 1.0, 0.0);

    // GGA: MO gradients on grids
    let (mo_occ_grad, mo_vir_grad) = if xc_type == XCType::GGA {
        let aop = grids.aop.as_ref().expect("AO gradients needed for GGA");
        let mut og = [
            MatrixFull::new([occ_size, num_grids], 0.0),
            MatrixFull::new([occ_size, num_grids], 0.0),
            MatrixFull::new([occ_size, num_grids], 0.0),
        ];
        let mut vg = [
            MatrixFull::new([vir_size, num_grids], 0.0),
            MatrixFull::new([vir_size, num_grids], 0.0),
            MatrixFull::new([vir_size, num_grids], 0.0),
        ];
        for d in 0..3 {
            let aop_d = aop.get_reducing_matrix(d).unwrap();
            let aop_d_owned = MatrixFull::from_vec(
                [num_basis, num_grids],
                aop_d.iter().cloned().collect(),
            ).unwrap();
            _dgemm_full(&c_occ, 'T', &aop_d_owned, 'N', &mut og[d], 1.0, 0.0);
            _dgemm_full(&c_vir, 'T', &aop_d_owned, 'N', &mut vg[d], 1.0, 0.0);
        }
        (Some(og), Some(vg))
    } else {
        (None, None)
    };

    // Compute ground-state density for fxc
    let ao_deriv = if xc_type == XCType::GGA { 1 } else { 0 };
    let ao_rifull = eval_ao_batch(&scf_data.mol, &grids.coordinates, ao_deriv, num_grids);
    let mo_coeffs = vec![scf_data.eigenvectors[0].clone()];
    let occ = vec![scf_data.occupation[0].clone()];
    let rho_tensor = eval_rho5_batch(&ao_rifull, xc_type, &mo_coeffs, &occ, 1, num_grids);
    let rho_array: Vec<f64> = rho_tensor.to_vec();

    // Singlet fxc: spin=0, deriv=2
    let xc_singlet = eval_xc_eff(func_ids, func_factors, xc_type, 0, &rho_array, num_grids, 2);
    let fxc_s = xc_singlet[2].as_ref().expect("fxc must be available");
    let wfxc_singlet = compute_weighted_fxc(fxc_s, weights, num_grids, nvar, false);

    // Triplet fxc: need spin=1 to separate αα and αβ
    // Duplicate RKS density for both spin channels (each gets half)
    let mut rho_spin = vec![0.0; num_grids * nvar * 2];
    for g in 0..num_grids {
        for v in 0..nvar {
            let val = rho_array[g * nvar + v] * 0.5;
            rho_spin[(g * nvar + v) * 2 + 0] = val;
            rho_spin[(g * nvar + v) * 2 + 1] = val;
        }
    }
    let xc_spin = eval_xc_eff(func_ids, func_factors, xc_type, 1, &rho_spin, num_grids, 2);
    let fxc_t = xc_spin[2].as_ref().expect("spin-resolved fxc must be available");
    let wfxc_triplet = compute_weighted_fxc(fxc_t, weights, num_grids, nvar, true);

    println!("TDDFT data prepared: occ={}, vir={}, grids={}, xc_type={:?}, alpha_hybrid={}",
             occ_size, vir_size, num_grids, xc_type, alpha_hybrid);

    TDDFTData {
        xc_type, nvar, num_grids, occ_size, vir_size, start_mo, alpha_hybrid,
        mo_occ, mo_vir, mo_occ_grad, mo_vir_grad,
        wfxc_singlet, wfxc_triplet,
    }
}

fn compute_weighted_fxc(
    fxc_tensor: &Tensor<f64, DeviceBLAS>,
    weights: &[f64],
    num_grids: usize,
    nvar: usize,
    is_triplet: bool,
) -> Vec<f64> {
    let mut wfxc = vec![0.0; num_grids * nvar * nvar];
    let fxc_data = fxc_tensor.to_vec();

    if !is_triplet {
        // fxc shape: [num_grids, nvar, nvar]
        // Factor 2: eval_xc_eff(spin=0) returns ∂²(ρε)/∂ρ² w.r.t. total density,
        // but the RKS singlet kernel is f_xc^{αα} + f_xc^{αβ} = 2 * ∂²(ρε)/∂ρ².
        for g in 0..num_grids {
            for a in 0..nvar {
                for b in 0..nvar {
                    let idx = (g * nvar + a) * nvar + b;
                    wfxc[idx] = 2.0 * fxc_data[idx] * weights[g];
                }
            }
        }
    } else {
        // fxc shape: [num_grids, nvar, 2, nvar, 2]
        // triplet = fxc[g,a,α,b,α] - fxc[g,a,α,b,β]
        let s_g = nvar * 2 * nvar * 2;
        let s_a = 2 * nvar * 2;
        let s_s1 = nvar * 2;
        let s_b = 2;
        for g in 0..num_grids {
            for a in 0..nvar {
                for b in 0..nvar {
                    let idx_aa = g * s_g + a * s_a + 0 * s_s1 + b * s_b + 0;
                    let idx_ab = g * s_g + a * s_a + 0 * s_s1 + b * s_b + 1;
                    let out_idx = (g * nvar + a) * nvar + b;
                    wfxc[out_idx] = (fxc_data[idx_aa] - fxc_data[idx_ab]) * weights[g];
                }
            }
        }
    }
    wfxc
}
