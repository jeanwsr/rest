use tensors::{MatrixFull, MathMatrix};
use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;

use crate::scf_io::SCF;
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
    let rho_array: Vec<f64> = rho_tensor.reshape(-1).to_vec();

    // Compute fxc via numerical differentiation of vxc (deriv=1),
    // because eval_xc_eff with deriv=2 triggers a buffer overflow bug in merge_xc.
    let delta = 1.0e-5;
    let inv_2delta = 0.5 / delta;

    // Singlet fxc: d²Exc/drho² (total density second derivative)
    let wfxc_singlet = numerical_fxc_singlet(
        func_ids, func_factors, xc_type, &rho_array, weights, num_grids, nvar, delta, inv_2delta,
    );

    // Triplet fxc: fxc_αα - fxc_αβ (via spin magnetization perturbation)
    let wfxc_triplet = numerical_fxc_triplet(
        func_ids, func_factors, xc_type, &rho_array, weights, num_grids, nvar, delta, inv_2delta,
    );

    println!("TDDFT data prepared: occ={}, vir={}, grids={}, xc_type={:?}, alpha_hybrid={}",
             occ_size, vir_size, num_grids, xc_type, alpha_hybrid);

    TDDFTData {
        xc_type, nvar, num_grids, occ_size, vir_size, start_mo, alpha_hybrid,
        mo_occ, mo_vir, mo_occ_grad, mo_vir_grad,
        wfxc_singlet, wfxc_triplet,
    }
}

fn numerical_fxc_singlet(
    func_ids: &Vec<usize>,
    func_factors: &Vec<f64>,
    xc_type: XCType,
    rho_array: &[f64],
    weights: &[f64],
    num_grids: usize,
    nvar: usize,
    delta: f64,
    inv_2delta: f64,
) -> Vec<f64> {
    // fxc_singlet[g, a, b] = (vxc[g,a](rho + δ*e_b) - vxc[g,a](rho - δ*e_b)) / (2δ) * w[g]
    // wfxc layout: row-major [num_grids, nvar, nvar]
    // rho_array layout: column-major [num_grids, nvar], i.e. rho[g, v] = rho_array[g + v * num_grids]
    // vxc layout: column-major [num_grids, nvar], i.e. vxc[g, a] = vxc_data[g + a * num_grids]
    let mut wfxc = vec![0.0; num_grids * nvar * nvar];
    let np = num_grids;

    for b in 0..nvar {
        let mut rho_plus = rho_array.to_vec();
        let mut rho_minus = rho_array.to_vec();
        for g in 0..np {
            rho_plus[g + b * np] += delta;
            rho_minus[g + b * np] -= delta;
        }

        let xc_plus = eval_xc_eff(func_ids, func_factors, xc_type, 0, &rho_plus, np, 1);
        let xc_minus = eval_xc_eff(func_ids, func_factors, xc_type, 0, &rho_minus, np, 1);

        let vxc_plus = xc_plus[1].as_ref().unwrap().reshape(-1).to_vec();
        let vxc_minus = xc_minus[1].as_ref().unwrap().reshape(-1).to_vec();

        for g in 0..np {
            for a in 0..nvar {
                let out_idx = (g * nvar + a) * nvar + b;
                let vp = vxc_plus[g + a * np];
                let vm = vxc_minus[g + a * np];
                wfxc[out_idx] = (vp - vm) * inv_2delta * weights[g];
            }
        }
    }
    wfxc
}

fn numerical_fxc_triplet(
    func_ids: &Vec<usize>,
    func_factors: &Vec<f64>,
    xc_type: XCType,
    rho_array: &[f64],
    weights: &[f64],
    num_grids: usize,
    nvar: usize,
    delta: f64,
    inv_2delta: f64,
) -> Vec<f64> {
    // Triplet fxc = fxc_αα - fxc_αβ
    // Perturb spin magnetization: rho_α = rho/2 + δ*e_b, rho_β = rho/2 - δ*e_b
    // fxc_triplet[g,a,b] = (vxc_α(+δ) - vxc_α(-δ))[g,a] / (2δ) * w[g]
    //
    // rho_spin layout: column-major [np, nvar, 2]
    //   rho_spin[g + v * np] = alpha, rho_spin[g + v * np + np * nvar] = beta
    // vxc_spin layout: column-major [np, nvar, 2]
    //   vxc[g + a * np] = alpha, vxc[g + a * np + np * nvar] = beta
    let mut wfxc = vec![0.0; num_grids * nvar * nvar];
    let np = num_grids;

    for b in 0..nvar {
        let mut rho_plus = vec![0.0; np * nvar * 2];
        let mut rho_minus = vec![0.0; np * nvar * 2];
        for v in 0..nvar {
            let pert = if v == b { delta } else { 0.0 };
            for g in 0..np {
                let val = rho_array[g + v * np] * 0.5;
                // alpha block: offset 0
                rho_plus[g + v * np] = val + pert;
                // beta block: offset np * nvar
                rho_plus[g + v * np + np * nvar] = val - pert;
                // reversed perturbation
                rho_minus[g + v * np] = val - pert;
                rho_minus[g + v * np + np * nvar] = val + pert;
            }
        }

        let xc_plus = eval_xc_eff(func_ids, func_factors, xc_type, 1, &rho_plus, np, 1);
        let xc_minus = eval_xc_eff(func_ids, func_factors, xc_type, 1, &rho_minus, np, 1);

        // vxc shape for spin=1: column-major [np, nvar, 2]
        let vxc_plus = xc_plus[1].as_ref().unwrap().reshape(-1).to_vec();
        let vxc_minus = xc_minus[1].as_ref().unwrap().reshape(-1).to_vec();

        for g in 0..np {
            for a in 0..nvar {
                let out_idx = (g * nvar + a) * nvar + b;
                // alpha component: offset 0
                let vp_alpha = vxc_plus[g + a * np];
                let vm_alpha = vxc_minus[g + a * np];
                wfxc[out_idx] = (vp_alpha - vm_alpha) * inv_2delta * weights[g];
            }
        }
    }
    wfxc
}
