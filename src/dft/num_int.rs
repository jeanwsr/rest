use std::sync::{Arc, Mutex};
use rstsr::prelude::*;
use rest_tensors::{MatrixFull, RIFull};
use rest_tensors::matrix_blas_lapack::{ _einsum_01_serial, _einsum_02_serial};
use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;
use tensors::matrix_blas_lapack::{_dgemm};
use crate::scf_io::SCF;
use crate::molecule_io::Molecule;
use crate::basis_io::{spheric_gto_deriv_batch_serial};
use crate::dft::{Grids, DFA4REST};
use crate::dft::xc_deriv::XCType;
use crate::dft::libxc_itrf::eval_xc_eff;
use crate::ri_tddft::utils::tddft_occupation_parameters;


pub fn eval_ao_batch(mol:&Molecule, coords:&[[f64; 3]], ao_deriv:usize, num_grids:usize) -> RIFull<f64> {
    // let default_omp_num_threads = unsafe {utilities::openblas_get_num_threads()};
    // let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    // utilities::omp_set_num_threads_wrapper(1);

    let n_components_deriv = (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) / 6;
    
    let mut loc_ao = RIFull::new([mol.num_basis, num_grids, n_components_deriv], 0.0);

    mol.basis4elem.iter()
    .zip(mol.geom.position.iter_columns_full())
    .for_each(
        |(elem, geom)|
        {
            let ind_glb_bas = elem.global_index.0;
            let loc_num_bas = elem.global_index.1;
            let start = ind_glb_bas;
            let end = start + loc_num_bas;
            let mut tmp_geom = [0.0; 3];
            tmp_geom.iter_mut().zip(geom.iter()).for_each(|value| {*value.0 = *value.1});
            let tab_ao_elem = spheric_gto_deriv_batch_serial(&coords, &tmp_geom, elem, ao_deriv);

            for ic in 0..n_components_deriv {
                let gto_ic = tab_ao_elem.get(ic).unwrap();
                loc_ao.copy_from_matr(start..end, 0..num_grids, ic, 0, 
                   gto_ic, 0..loc_num_bas, 0..num_grids);
            }
            // loc_ao.copy_from_ri(start..end, 0..num_grids, 0..n_components_cart, &tab_ao_elem, 0..loc_num_bas, 0..num_grids, 0..n_components_cart);
        }
    );

    loc_ao
}


pub fn eval_rho5_batch(ao:&RIFull<f64>, xc_type:XCType, mo:&Vec<MatrixFull<f64>>, occ:&Vec<Vec<f64>>, spin_channel:usize, num_grids:usize) -> Tensor<f64> {
    /*
        Evaluate density on a batch of ao grids value for given mo coefficients and occupation.
        Args:
            ao: gto value on grids [num_grids, num_basis, nderiv],
                nderiv = 1 for ao0, 4 for ao0, ao_x, ao_y, ao_z, 10 for 2nd order derivatives (symmetric)
            xc_type: exchange-correlation dfa type
            mo: orbital coeffients [num_basis, num_state]
            occ: occupation number [num_state]
            spin_channel: number of spins, 1 for Restricted, 2 for Unrestricted
            range_grids: range of grids
        Return:
            rho [num_grids, nvar, spin_channel]
            where nvar = 1 for LDA, 4 for GGA, 5 for MGGA 
            tabulated as rho, rho_x, rho_y, rho_z, tau
        Note:
            laplacian is not implemented
    */
    let num_basis = mo[0].size.get(0).unwrap();
    let num_state = mo[0].size.get(1).unwrap();

    let nvar = match xc_type {
        XCType::HF => 1, 
        XCType::LDA => 1,
        XCType::GGA => 4,
        XCType::MGGA => 5,
    };
    let mut cur_rho = RIFull::new([num_grids, nvar, spin_channel], 0.0);

    let ao0 = &ao.get_reducing_matrix(0).unwrap();

    for i_spin in 0..spin_channel {
        let mo_s = mo.get(i_spin).unwrap();
        let homo_s = occ[i_spin].iter()
            .enumerate()
            .filter(|(i,occ)| **occ >=1.0e-6)
            .map(|(i,occ)| i).max();
        let mut occ_s = if let Some(homo_s) = homo_s {
            occ.get(i_spin).unwrap()[0..homo_s+1].iter().map(|occ| occ.sqrt()).collect::<Vec<f64>>()
        } else {
            // In this case, no electrons in the i_spin channel, for which homo_s = None
            vec![]
        };

        let num_occ = occ_s.len();
        let mut wmo = _einsum_01_serial(&mo_s.to_matrixfullslice(), &occ_s);
        let mut tmo = MatrixFull::new([num_occ, num_grids], 0.0);
        // tmo = C.T matmul ao (half transform)
        _dgemm(
            &wmo, (0..wmo.size[0], 0..wmo.size[1]), 'T', 
            ao0, (0..ao0.size[0], 0..num_grids), 'N',
            &mut tmo, (0..wmo.size[1], 0..num_grids),
            1.0, 0.0
        );
        // spin case: rho_s = tmo * tmo (C_ug * C_vg * phi_ug * phi_vg => rho_g) 
        let rho_s = _einsum_02_serial(&tmo.to_matrixfullslice(), &tmo.to_matrixfullslice());
        // cur_rho[ispin][0] = rho_s
        cur_rho.get_reducing_matrix_mut(i_spin).unwrap()
        .iter_mut_j(0)
        .zip(rho_s.iter())
        .for_each(
            |(to, from)| {*to = *from}
        );
        if nvar > 1 { // gga and upper case 
            for ic in 0..3 { // x, y, z
                let mut tmop = MatrixFull::new([num_occ, num_grids], 0.0);
                let aop_ic = &ao.get_reducing_matrix(ic+1).unwrap();
                _dgemm(
                    &wmo, (0..wmo.size[0], 0..wmo.size[1]), 'T',
                    aop_ic, (0..aop_ic.size[0], 0..num_grids), 'N',
                    &mut tmop, (0..wmo.size[1], 0..num_grids),
                    1.0, 0.0
                );
                let rhop_ic_s = _einsum_02_serial(&tmop.to_matrixfullslice(), &tmo.to_matrixfullslice());
                // cur_rho[ispin][ic+1] = rhop_ic_s
                cur_rho.get_reducing_matrix_mut(i_spin).unwrap()
                .iter_mut_j(ic + 1)
                .zip(rhop_ic_s.iter())
                .for_each(
                    |(to, from)| {*to = *from * 2.0}
                );
                if nvar == 5 { // mgga
                    let tau_s = _einsum_02_serial(&tmop.to_matrixfullslice(), &tmop.to_matrixfullslice());
                    // cur_rho[ispin][5] = tau_s
                    // tau = 1/2 * |grad phi|^2 sum over x,y,z
                    cur_rho.get_reducing_matrix_mut(i_spin).unwrap()
                    .iter_mut_j(4)
                    .zip(tau_s.iter())
                    .for_each(
                        |(to, from)| {*to += *from*0.5}
                    );
                }
            }
        }
    }
    let rho_ten = rt::asarray((&cur_rho.data, [num_grids, nvar, spin_channel])).to_owned();
    rho_ten

}


pub fn eval_rho5_dm_only_batch(ao:&RIFull<f64>, xc_type:XCType, dms:&Vec<MatrixFull<f64>>, spin_channel:usize, num_grids:usize) -> Tensor<f64> {
    /*
        Evaluate density on grids for a given set of density matrices.
        Args:
            ao: gto value on grids [num_grids, num_basis, nderiv] (column major),
                nderiv = 1 for ao0, 4 for ao0, ao_x, ao_y, ao_z, 10 for 2nd order derivatives (symmetric)
            xc_method: exchange-correlation dfa type
            dms: density matrices of shape [num_basis, num_bais, spin_channel, nsets] 
            spin_channel: number of spins, 1 for Restricted, 2 for Unrestricted
            range_grids: range of grids
        Return:
            rho [num_grids, nvar, spin_channel]
            where nvar = 1 for LDA, 4 for GGA, 5 for MGGA 
            tabulated as rho, rho_x, rho_y, rho_z, tau
        Note:
            laplacian is not implemented
    */

    let num_basis = dms[0].size[0];

    let nvar = match xc_type {
        XCType::HF => 1, 
        XCType::LDA => 1,
        XCType::GGA => 4,
        XCType::MGGA => 5,
    };

    let mut cur_rho = RIFull::new([num_grids, nvar, spin_channel], 0.0);
    
    let ao0 = &ao.get_reducing_matrix(0).unwrap();

    for i_spin in 0..spin_channel {
        // rho 
        let dm_s = &dms[i_spin];
        let mut wao = MatrixFull::new([num_basis, num_grids],0.0);
        _dgemm(
            dm_s, (0..num_basis, 0..num_basis), 'N',
            ao0, (0..num_basis, 0..num_grids), 'N',
            &mut wao,  (0..num_basis, 0..num_grids), 
            1.0, 0.0
        );
        ao0.iter_columns(0..num_grids).unwrap()
        .zip(wao.iter_columns_full())
        .map(|(ao_r,wao_r)| (ao_r, wao_r))
        .zip(cur_rho.get_reducing_matrix_mut(i_spin).unwrap().iter_column_mut(0))
        .for_each(
            |((ao_r, wao_r), cur_rho_s)| 
            {
                *cur_rho_s = wao_r.iter().zip(ao_r.iter()).fold(0.0, |acc, (a,b)| {acc + a*b})
            }
        );
        if nvar > 1 {
            for ic in 0..3usize {
                let mut aop_ic = &ao.get_reducing_matrix(ic + 1).unwrap();
                let mut wao = MatrixFull::new([num_basis, num_grids],0.0);
                _dgemm(
                    dm_s, (0..num_basis, 0..num_basis), 'N',
                    aop_ic, (0..num_basis, 0..num_grids), 'N',
                    &mut wao, (0..num_basis, 0..num_grids), 1.0, 0.0
                );
            
                ao0.iter_columns(0..num_grids).unwrap()
                .zip(wao.iter_columns_full())
                .map(|(ao_r, wao_r)| (ao_r, wao_r))
                .zip(cur_rho.get_reducing_matrix_mut(i_spin).unwrap().iter_column_mut(ic + 1))
                .for_each(
                    |((ao_r, wao_r), cur_rhop_r)| 
                    {
                        *cur_rhop_r = 2.0 * wao_r.iter().zip(ao_r.iter()).fold(0.0, |acc,(wao,ao)| {acc + wao*ao})
                    }
                );
                if nvar == 5 {
                    aop_ic.iter_columns(0..num_grids).unwrap()
                    .zip(wao.iter_columns_full())
                    .map(|(aop_r, wao_r)|(aop_r, wao_r))
                    .zip(cur_rho.get_reducing_matrix_mut(i_spin).unwrap().iter_column_mut(4))
                    .for_each(
                        |((aop_r, wao_r), cur_tau_s)|
                        {
                            *cur_tau_s += 0.5 * wao_r.iter().zip(aop_r.iter()).fold(0.0, |acc, (a,b)| {acc + a*b})
                        }
                    );
                }
            }
        }
    } // end spin case 
    let rho_ten = rt::asarray((&cur_rho.data, [num_grids, nvar, spin_channel])).to_owned();
    rho_ten
}


pub fn eval_rho5_spin_batch(ao:&RIFull<f64>, xc_type:XCType, mo:&MatrixFull<f64>, occ:&Vec<f64>, num_grids:usize) -> Tensor<f64> {
    /*
        Evaluate density on a batch of ao grids value for given mo coefficients and occupation in one spin channel.
        Args:
            ao: gto value on grids [num_grids, num_basis, nderiv],
                nderiv = 1 for ao0, 4 for ao0, ao_x, ao_y, ao_z, 10 for 2nd order derivatives (symmetric)
            xc_type: exchange-correlation dfa type
            mo: orbital coeffients [num_basis, num_state]
            occ: occupation number [num_state]
            range_grids: range of grids
        Return:
            rho [num_grids, nvar]
            where nvar = 1 for LDA, 4 for GGA, 5 for MGGA 
            tabulated as rho, rho_x, rho_y, rho_z, tau
        Note:
            laplacian is not implemented
    */

    let num_basis = mo.size.get(0).unwrap();
    let num_state = mo.size.get(1).unwrap();

    let nvar = match xc_type {
        XCType::HF => 1, 
        XCType::LDA => 1,
        XCType::GGA => 4,
        XCType::MGGA => 5,
    };
    let mut cur_rho = MatrixFull::new([num_grids, nvar], 0.0);

    let ao0 = &ao.get_reducing_matrix(0).unwrap();

    
    
    let homo = occ.iter().enumerate()
        .filter(|(i,occ)| **occ >=1.0e-6)
        .map(|(i,occ)| i).max();
    let mut occ_tmp = if let Some(homo) = homo {
            occ[0..homo+1].iter().map(|occ| occ.sqrt()).collect::<Vec<f64>>()
    } else {
        // In this case, no electrons in the i_spin channel, for which homo_s = None
        vec![]
    };

    let num_occ = occ_tmp.len();
    let mut wmo = _einsum_01_serial(&mo.to_matrixfullslice(), &occ_tmp);
    let mut tmo = MatrixFull::new([num_occ, num_grids], 0.0);
    // tmo = C.T matmul ao (half transform)
    _dgemm(
        &wmo, (0..wmo.size[0], 0..wmo.size[1]), 'T', 
        ao0, (0..ao0.size[0], 0..num_grids), 'N',
        &mut tmo, (0..wmo.size[1], 0..num_grids),
        1.0, 0.0
    );
    // rho_0 = tmo * tmo (C_ug * C_vg * phi_ug * phi_vg => rho_g) 
    let rho_0 = _einsum_02_serial(&tmo.to_matrixfullslice(), &tmo.to_matrixfullslice());
    // cur_rho[0] = rho_0
    cur_rho.iter_column_mut(0).zip(rho_0.iter())
    .for_each(
        |(to, from)| {*to = *from}
    );
        
    if nvar > 1 { // gga and upper case 
        for ic in 0..3 { // x, y, z
            let mut tmop = MatrixFull::new([num_occ, num_grids], 0.0);
            let aop_ic = &ao.get_reducing_matrix(ic+1).unwrap();
            _dgemm(
                &wmo, (0..wmo.size[0], 0..wmo.size[1]), 'T',
                aop_ic, (0..aop_ic.size[0], 0..num_grids), 'N',
                &mut tmop, (0..wmo.size[1], 0..num_grids),
                1.0, 0.0
            );
            let rhop_ic = _einsum_02_serial(&tmop.to_matrixfullslice(), &tmo.to_matrixfullslice());
            // cur_rho[ic+1] = rhop_ic
            cur_rho.iter_column_mut(ic + 1).zip(rhop_ic.iter())
            .for_each(
                |(to, from)| {*to = *from * 2.0}
            );
            if nvar == 5 { // mgga
                let tau = _einsum_02_serial(&tmop.to_matrixfullslice(), &tmop.to_matrixfullslice());
                // cur_rho[5] = tau
                // tau = 1/2 * |grad phi|^2 sum over x,y,z
                cur_rho.iter_column_mut(4).zip(tau.iter())
                .for_each(
                    |(to, from)| {*to += *from*0.5}
                );
            }
        }
    }

    let rho_ten = rt::asarray((cur_rho.data, [num_grids, nvar]));
    rho_ten 
}


pub fn eval_rho5_spin_dm_only_batch(ao:&RIFull<f64>, xc_type:XCType, dm:&MatrixFull<f64>, num_grids:usize) -> Tensor<f64> {
    let num_basis = dm.size[0];

    let nvar = match xc_type {
        XCType::HF => 1, 
        XCType::LDA => 1,
        XCType::GGA => 4,
        XCType::MGGA => 5,
    };

    let mut cur_rho = MatrixFull::new([num_grids, nvar], 0.0);
    let ao0 = &ao.get_reducing_matrix(0).unwrap();
    // rho0 
    let mut wao = MatrixFull::new([num_basis, num_grids],0.0);
    _dgemm(
        dm, (0..num_basis, 0..num_basis), 'N',
        ao0, (0..num_basis, 0..num_grids), 'N',
        &mut wao,  (0..num_basis, 0..num_grids), 
        1.0, 0.0
    );
    ao0.iter_columns(0..num_grids).unwrap()
    .zip(wao.iter_columns_full())
    .map(|(ao_r,wao_r)| (ao_r, wao_r))
    .zip(cur_rho.iter_column_mut(0))
    .for_each(
        |((ao_r, wao_r), cur_rho_0)| 
        {
            *cur_rho_0 = wao_r.iter().zip(ao_r.iter()).fold(0.0, |acc, (a,b)| {acc + a*b})
        }
    );
    // gga and upper case
    if nvar > 1 {
        // nabla rho 
        for ic in 0..3usize {
            let mut aop_ic = &ao.get_reducing_matrix(ic + 1).unwrap();
            let mut wao = MatrixFull::new([num_basis, num_grids],0.0);
            _dgemm(
                dm, (0..num_basis, 0..num_basis), 'N',
                aop_ic, (0..num_basis, 0..num_grids), 'N',
                &mut wao, (0..num_basis, 0..num_grids), 1.0, 0.0
            );
            ao0.iter_columns(0..num_grids).unwrap()
            .zip(wao.iter_columns_full())
            .map(|(ao_r, wao_r)| (ao_r, wao_r))
            .zip(cur_rho.iter_column_mut(ic + 1))
            .for_each(
                |((ao_r, wao_r), cur_rhop_r)| 
                {
                    *cur_rhop_r = 2.0 * wao_r.iter().zip(ao_r.iter()).fold(0.0, |acc,(wao,ao)| {acc + wao*ao})
                }
            );
            // tau 
            if nvar == 5 {
                aop_ic.iter_columns(0..num_grids).unwrap()
                .zip(wao.iter_columns_full())
                .map(|(aop_r, wao_r)|(aop_r, wao_r))
                .zip(cur_rho.iter_column_mut(4))
                .for_each(
                    |((aop_r, wao_r), cur_tau)|
                    {
                        *cur_tau += 0.5 * wao_r.iter().zip(aop_r.iter()).fold(0.0, |acc, (a,b)| {acc + a*b})
                    }
                );
            }
        }
    }

    let rho_ten = rt::asarray((cur_rho.data, [num_grids, nvar]));
    rho_ten
}

// ══════════════════════════════════════════════════════════════════════════════
// fxc kernel for TDDFT/CP-HF: exchange-correlation kernel matrix-vector product
// ══════════════════════════════════════════════════════════════════════════════
//
// Physical formula:
//   (A^{fxc} · z)_{ia} = Σ_g w(g) · Σ_{αβ} D_α[φ_i φ_a](g) · fxc(g)_{αβ} · ρ_z,β(g)
//   where ρ_z,β(g) = Σ_{jb} D_β[φ_j φ_b](g) · z_{jb}
//
// Reference: PySCF TDDFT implementation pattern

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
    /// If true, `fxc_matvec` will use the optimized implementation
    /// (`fxc_matvec_opt`) instead of the original.
    pub use_opt: bool,
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
    let rho_array: Vec<f64> = rho_tensor.raw()[rho_tensor.offset()..]
        .chunks(ngrids * nvar)
        .next()
        .unwrap_or(&[])
        .to_vec();
    let rho_array = if rho_array.is_empty() {
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
        use_opt: scf.mol.ctrl.use_fxc_opt,
    }
}

/// Dispatch fxc matrix-vector product based on functional type.
///
/// When `data.use_opt` is `true`, the optimized implementation
/// (`fxc_matvec_opt`) is used; otherwise the original (`fxc_matvec_old`)
/// is used.
pub fn fxc_matvec(data: &FXCMatvecData, z: &[f64]) -> Vec<f64> {
    assert_eq!(z.len(), data.nocc * data.nvir,
               "z vector length {} must equal nocc×nvir = {}×{}",
               z.len(), data.nocc, data.nvir);
    if data.use_opt {
        let mut ws = prepare_fxc_workspace(data);
        fxc_matvec_opt(data, z, &mut ws)
    } else {
        match data.nvar {
            1 => fxc_matvec_lda(data, z),
            4 => fxc_matvec_gga(data, z),
            _ => panic!("fxc_matvec only supports LDA (nvar=1) and GGA (nvar=4)"),
        }
    }
}

/// Original fxc matrix-vector product (kept for reference).
pub fn fxc_matvec_old(data: &FXCMatvecData, z: &[f64]) -> Vec<f64> {
    assert_eq!(z.len(), data.nocc * data.nvir,
               "z vector length {} must equal nocc×nvir = {}×{}",
               z.len(), data.nocc, data.nvir);
    match data.nvar {
        1 => fxc_matvec_lda(data, z),
        4 => fxc_matvec_gga(data, z),
        _ => panic!("fxc_matvec_old only supports LDA (nvar=1) and GGA (nvar=4)"),
    }
}

/// LDA fxc matrix-vector product
fn fxc_matvec_lda(data: &FXCMatvecData, z: &[f64]) -> Vec<f64> {
    let nocc = data.nocc;
    let nvir = data.nvir;
    let ngrids = data.ngrids;

    // ── Step 1: Compute perturbed density on grids ──
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
    let mut v = vec![0.0; ngrids];
    for g in 0..ngrids {
        v[g] = data.wfxc[g] * rho_z[g];
    }

    // ── Step 3: Contract back to MO basis ──
    let mut mo_vir_scaled = MatrixFull::new([nvir, ngrids], 0.0);
    for g in 0..ngrids {
        for a in 0..nvir {
            mo_vir_scaled[[a, g]] = data.mo_vir[[a, g]] * v[g];
        }
    }

    let mut result = MatrixFull::new([nocc, nvir], 0.0);
    _dgemm_full(&data.mo_occ, 'N', &mo_vir_scaled, 'T', &mut result, 1.0, 0.0);

    result.data
}

/// GGA fxc matrix-vector product
fn fxc_matvec_gga(data: &FXCMatvecData, z: &[f64]) -> Vec<f64> {
    let nocc = data.nocc;
    let nvir = data.nvir;
    let ngrids = data.ngrids;

    let mo_occ_grad = data.mo_occ_grad.as_ref().expect("GGA requires mo_occ_grad");
    let mo_vir_grad = data.mo_vir_grad.as_ref().expect("GGA requires mo_vir_grad");

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

    let mut rho_z = vec![0.0; ngrids * 4];

    // β=0: ρ_z[0,g] = Σ_i mo_occ[i,g] × t0[i,g]
    for g in 0..ngrids {
        let mut sum = 0.0;
        for i in 0..nocc {
            sum += data.mo_occ[[i, g]] * t0[[i, g]];
        }
        rho_z[g] = sum;

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
    let mut fxc_eff_grid = vec![0.0; ngrids * 4];

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
    let mut result = vec![0.0; nocc * nvir];

    for alpha in 0..4 {
        let fxc_a = &fxc_eff_grid[alpha * ngrids..(alpha + 1) * ngrids];

        if alpha == 0 {
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
            let d = alpha - 1;

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

// ═══════════════════════════════════════════════════════════════════════
// Optimized fxc_matvec with pre-allocated workspace and parallelism
// ═══════════════════════════════════════════════════════════════════════
//
// Key optimizations over the original:
//   1. Pre-allocated workspace eliminates per-call heap allocations.
//   2. z is copied (not cloned via to_vec()) into pre-allocated z_mat.
//   3. GGA wfxc is packed [g][16] so the 4×4 kernel per grid point is
//      contiguous in memory (eliminating MB-stride access).
//   4. Rayon parallelism on independent loops (rho_z, 4×4 kernel,
//      column-wise scaling).  The GEMMs remain handled by OpenBLAS.

/// Pre-allocated workspace for `fxc_matvec_opt`.
///
/// All buffers are sized once at construction and reused across calls.
#[derive(Debug)]
pub struct FXCMatvecWorkspace {
    /// z matrix [nocc, nvir] — avoids `z.to_vec()` allocation per call
    pub z_mat: MatrixFull<f64>,
    /// t = z_mat @ mo_vir [nocc, ngrids]
    pub t: MatrixFull<f64>,
    /// Density response: [ngrids] for LDA, [ngrids*4] for GGA
    pub rho_z: Vec<f64>,
    /// Scaled virtuals for LDA contraction [nvir, ngrids]
    pub mo_vir_scaled: MatrixFull<f64>,
    /// Result buffer [nocc * nvir]
    pub result: Vec<f64>,

    // GGA-specific:
    /// t_grad[d] = z_mat @ mo_vir_grad[d]  [nocc, ngrids] each
    pub t_grad: [MatrixFull<f64>; 3],
    /// fxc kernel result on grids [ngrids * 4] — stored [g][α] (grid-major)
    pub fxc_eff_grid: Vec<f64>,
    /// right-scaled virtuals [nvir, ngrids]
    pub right_scaled: MatrixFull<f64>,
    /// right-scaled virtual gradients [nvir, ngrids]
    pub right_grad_scaled: MatrixFull<f64>,
    /// Packed wfxc [ngrids * 16]: each grid point's 4×4 kernel contiguous.
    /// Derived from data.wfxc (strided [g,α,β] → packed [g][α][β]).
    pub wfxc_packed: Option<Vec<f64>>,
}

/// Allocate workspace buffers sized for `data`.
///
/// For GGA this also pre-computes the packed wfxc layout.
pub fn prepare_fxc_workspace(data: &FXCMatvecData) -> FXCMatvecWorkspace {
    let nocc = data.nocc;
    let nvir = data.nvir;
    let ngrids = data.ngrids;
    let is_gga = data.nvar == 4;

    let wfxc_packed = if is_gga {
        // Original layout: idx = g + α * ngrids + β * 4 * ngrids
        // Packed  layout:  idx = g * 16 + α * 4 + β
        let src = &data.wfxc;
        let mut packed = vec![0.0; ngrids * 16];
        for g in 0..ngrids {
            for a in 0..4 {
                for b in 0..4 {
                    packed[g * 16 + a * 4 + b] =
                        src[g + a * ngrids + b * 4 * ngrids];
                }
            }
        }
        Some(packed)
    } else {
        None
    };

    FXCMatvecWorkspace {
        z_mat: MatrixFull::new([nocc, nvir], 0.0),
        t: MatrixFull::new([nocc, ngrids], 0.0),
        rho_z: vec![0.0; if is_gga { ngrids * 4 } else { ngrids }],
        mo_vir_scaled: MatrixFull::new([nvir, ngrids], 0.0),
        result: vec![0.0; nocc * nvir],
        t_grad: [
            MatrixFull::new([nocc, ngrids], 0.0),
            MatrixFull::new([nocc, ngrids], 0.0),
            MatrixFull::new([nocc, ngrids], 0.0),
        ],
        fxc_eff_grid: vec![0.0; ngrids * 4],
        right_scaled: MatrixFull::new([nvir, ngrids], 0.0),
        right_grad_scaled: MatrixFull::new([nvir, ngrids], 0.0),
        wfxc_packed,
    }
}

/// Optimized fxc matrix-vector product using pre-allocated workspace.
///
/// The result vector is taken from the workspace's internal buffer.
/// Call `prepare_fxc_workspace` once before calling this function in a loop.
pub fn fxc_matvec_opt(
    data: &FXCMatvecData,
    z: &[f64],
    ws: &mut FXCMatvecWorkspace,
) -> Vec<f64> {
    let nocc = data.nocc;
    let nvir = data.nvir;
    assert_eq!(z.len(), nocc * nvir, "z length mismatch");

    // Copy z into pre-allocated z_mat, avoiding the `z.to_vec()` allocation.
    ws.z_mat.data.copy_from_slice(z);

    match data.nvar {
        1 => fxc_matvec_lda_opt(data, ws),
        4 => fxc_matvec_gga_opt(data, ws),
        _ => panic!("fxc_matvec_opt only supports LDA (nvar=1) and GGA (nvar=4)"),
    }
    std::mem::take(&mut ws.result)
}

// ── LDA optimized ────────────────────────────────────────────────────

fn fxc_matvec_lda_opt(data: &FXCMatvecData, ws: &mut FXCMatvecWorkspace) {
    use rayon::prelude::*;
    let nocc = data.nocc;
    let nvir = data.nvir;
    let ngrids = data.ngrids;

    // Step 1a: t = z_mat @ mo_vir  (BLAS GEMM — thread-safe)
    ws.t.data.fill(0.0);
    _dgemm_full(&ws.z_mat, 'N', &data.mo_vir, 'N', &mut ws.t, 1.0, 0.0);

    // Step 1b: rho_z[g] = Σ_i mo_occ[i,g] × t[i,g]   (parallel over g)
    //
    // MatrixFull is column-major [nocc, ngrids]: column g has its nocc
    // elements contiguously at data[g*nocc .. (g+1)*nocc].
    let mo_occ = &data.mo_occ;
    let t = &ws.t;
    ws.rho_z
        .par_iter_mut()
        .enumerate()
        .for_each(|(g, rho)| {
            let base = g * nocc;
            let mo_col = &mo_occ.data[base..base + nocc];
            let t_col = &t.data[base..base + nocc];
            let mut sum = 0.0;
            for i in 0..nocc {
                sum += mo_col[i] * t_col[i];
            }
            *rho = sum;
        });

    // Step 2: v[g] = wfxc[g] × rho_z[g]   (element-wise, reuse rho_z buf)
    for g in 0..ngrids {
        ws.rho_z[g] = data.wfxc[g] * ws.rho_z[g];
    }

    // Step 3a: mo_vir_scaled[a,g] = mo_vir[a,g] × v[g]  (parallel over columns)
    //
    // mo_vir is [nvir, ngrids] column-major: column g is at data[g*nvir..].
    // par_chunks_mut(nvir) splits the data into ngrids chunks, one per column.
    let mo_vir = &data.mo_vir;
    let v = &ws.rho_z;
    ws.mo_vir_scaled
        .data
        .par_chunks_mut(nvir)
        .enumerate()
        .for_each(|(g, col)| {
            let vg = v[g];
            for a in 0..nvir {
                col[a] = mo_vir.data[a + g * nvir] * vg;
            }
        });

    // Step 3b: result = mo_occ × mo_vir_scaled^T  (BLAS GEMM)
    //
    // NOTE: allocate a fresh Vec rather than taking from ws.result,
    // because ws.result may be empty after a previous fxc_matvec_opt call
    // drained it for return.  We assign the GEMM result into ws.result
    // so it is valid for subsequent calls.
    let mut result_mat = MatrixFull {
        size: [nocc, nvir],
        indicing: [nocc, nvir],
        data: vec![0.0; nocc * nvir],
    };
    _dgemm_full(
        mo_occ,
        'N',
        &ws.mo_vir_scaled,
        'T',
        &mut result_mat,
        1.0,
        0.0,
    );
    ws.result = result_mat.data;
}

// ── GGA optimized ────────────────────────────────────────────────────

fn fxc_matvec_gga_opt(data: &FXCMatvecData, ws: &mut FXCMatvecWorkspace) {
    use rayon::prelude::*;
    let nocc = data.nocc;
    let nvir = data.nvir;
    let ngrids = data.ngrids;

    let mo_occ_grad = data
        .mo_occ_grad
        .as_ref()
        .expect("GGA requires mo_occ_grad");
    let mo_vir_grad = data
        .mo_vir_grad
        .as_ref()
        .expect("GGA requires mo_vir_grad");
    let wfxc_packed = ws
        .wfxc_packed
        .as_ref()
        .expect("GGA requires packed wfxc");

    // Step 1a: t0 = z_mat @ mo_vir  (BLAS)
    ws.t.data.fill(0.0);
    _dgemm_full(&ws.z_mat, 'N', &data.mo_vir, 'N', &mut ws.t, 1.0, 0.0);

    // Step 1b: t_grad[d] = z_mat @ mo_vir_grad[d]  (BLAS × 3)
    for d in 0..3 {
        ws.t_grad[d].data.fill(0.0);
        _dgemm_full(
            &ws.z_mat,
            'N',
            &mo_vir_grad[d],
            'N',
            &mut ws.t_grad[d],
            1.0,
            0.0,
        );
    }

    // Step 1c: rho_z[β, g] — parallel over the 4 β components.
    //
    // rho_z is stored as 4 contiguous slabs of length ngrids.
    // par_chunks_mut(ngrids) → 4 chunks, one per β.
    let mo_occ = &data.mo_occ;
    let t0 = &ws.t;
    ws.rho_z
        .par_chunks_mut(ngrids)
        .enumerate()
        .for_each(|(beta, rho_beta)| {
            if beta == 0 {
                // β=0:  ρ_z[0,g] = Σ_i mo_occ[i,g] × t0[i,g]
                for g in 0..ngrids {
                    let base = g * nocc;
                    let mut s = 0.0;
                    for i in 0..nocc {
                        s += mo_occ.data[base + i] * t0.data[base + i];
                    }
                    rho_beta[g] = s;
                }
            } else {
                let d = beta - 1;
                let tg = &ws.t_grad[d];
                let mog = &mo_occ_grad[d];
                // β=1..3:  ρ_z[β,g] =
                //   Σ_i ∇mo[d,i,g] × t0[i,g]  +  mo_occ[i,g] × ∇t[d,i,g]
                for g in 0..ngrids {
                    let base = g * nocc;
                    let mut s = 0.0;
                    for i in 0..nocc {
                        s += mog.data[base + i] * t0.data[base + i]
                            + mo_occ.data[base + i] * tg.data[base + i];
                    }
                    rho_beta[g] = s;
                }
            }
        });

    // Step 2: 4×4 matvec per grid point  (parallel over g)
    //
    // wfxc_packed[g*16 + α*4 + β] — all 16 values per g are contiguous.
    //
    // fxc_eff_grid is written in [g][α] (grid-major) layout:
    //   fxc_eff_grid[g*4 + α] = Σ_β wfxc_packed[g*16 + α*4 + β] * rho_z[g + β*ngrids]
    let fxc_eff = &mut ws.fxc_eff_grid;
    fxc_eff
        .par_chunks_mut(4)
        .enumerate()
        .for_each(|(g, blk)| {
            let wb = g * 16;
            let r0 = ws.rho_z[g];
            let r1 = ws.rho_z[g + 1 * ngrids];
            let r2 = ws.rho_z[g + 2 * ngrids];
            let r3 = ws.rho_z[g + 3 * ngrids];

            blk[0] = wfxc_packed[wb] * r0
                + wfxc_packed[wb + 1] * r1
                + wfxc_packed[wb + 2] * r2
                + wfxc_packed[wb + 3] * r3;
            blk[1] = wfxc_packed[wb + 4] * r0
                + wfxc_packed[wb + 5] * r1
                + wfxc_packed[wb + 6] * r2
                + wfxc_packed[wb + 7] * r3;
            blk[2] = wfxc_packed[wb + 8] * r0
                + wfxc_packed[wb + 9] * r1
                + wfxc_packed[wb + 10] * r2
                + wfxc_packed[wb + 11] * r3;
            blk[3] = wfxc_packed[wb + 12] * r0
                + wfxc_packed[wb + 13] * r1
                + wfxc_packed[wb + 14] * r2
                + wfxc_packed[wb + 15] * r3;
        });

    // Step 3: Contract back to MO basis (sequential α loop, parallel scaling)
    //
    // Reallocate result Vec: ws.result may be empty after a previous
    // fxc_matvec_opt call drained it for return.
    ws.result = vec![0.0; nocc * nvir];
    let mo_occ = &data.mo_occ;
    let mo_vir = &data.mo_vir;

    // fxc_eff is [g][α], so α-component at grid point g is fxc_eff[g*4 + α].
    for alpha in 0..4 {
        if alpha == 0 {
            // right_scaled[a,g] = mo_vir[a,g] × fxc_eff[g*4]  (column-wise)
            ws.right_scaled
                .data
                .par_chunks_mut(nvir)
                .enumerate()
                .for_each(|(g, col)| {
                    let fv = fxc_eff[g * 4];
                    for a in 0..nvir {
                        col[a] = mo_vir.data[a + g * nvir] * fv;
                    }
                });
            // contrib = mo_occ × right_scaled^T
            let mut contrib = MatrixFull {
                size: [nocc, nvir],
                indicing: [nocc, nvir],
                data: vec![0.0; nocc * nvir],
            };
            _dgemm_full(mo_occ, 'N', &ws.right_scaled, 'T', &mut contrib, 1.0, 0.0);
            for i in 0..ws.result.len() {
                ws.result[i] += contrib.data[i];
            }
        } else {
            let d = alpha - 1;

            // right_scaled = mo_vir × fxc_eff[g*4 + α]
            ws.right_scaled
                .data
                .par_chunks_mut(nvir)
                .enumerate()
                .for_each(|(g, col)| {
                    let fv = fxc_eff[g * 4 + alpha];
                    for a in 0..nvir {
                        col[a] = mo_vir.data[a + g * nvir] * fv;
                    }
                });
            let mut contrib_a = MatrixFull {
                size: [nocc, nvir],
                indicing: [nocc, nvir],
                data: vec![0.0; nocc * nvir],
            };
            _dgemm_full(
                &mo_occ_grad[d],
                'N',
                &ws.right_scaled,
                'T',
                &mut contrib_a,
                1.0,
                0.0,
            );
            for i in 0..ws.result.len() {
                ws.result[i] += contrib_a.data[i];
            }

            // right_grad_scaled = mo_vir_grad[d] × fxc_eff[g*4 + α]
            ws.right_grad_scaled
                .data
                .par_chunks_mut(nvir)
                .enumerate()
                .for_each(|(g, col)| {
                    let fv = fxc_eff[g * 4 + alpha];
                    for a in 0..nvir {
                        col[a] = mo_vir_grad[d].data[a + g * nvir] * fv;
                    }
                });
            let mut contrib_b = MatrixFull {
                size: [nocc, nvir],
                indicing: [nocc, nvir],
                data: vec![0.0; nocc * nvir],
            };
            _dgemm_full(
                mo_occ,
                'N',
                &ws.right_grad_scaled,
                'T',
                &mut contrib_b,
                1.0,
                0.0,
            );
            for i in 0..ws.result.len() {
                ws.result[i] += contrib_b.data[i];
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct BlockSettings {
    pub ao_deriv: usize,
    pub max_memory: usize, // in MB
    pub blksize: usize, // block size of batched grids
}

pub struct BlockData<'a> {
    pub coords: &'a [[f64; 3]],
    pub weights: &'a [f64],
    pub ao: RIFull<f64>, // atomic orbitals evaluated on grids
}

pub struct XCData<'a> {
    pub xc_type: XCType,
    pub xc_code: &'a Vec<usize>,
    pub xc_params: &'a Vec<f64>,
    pub device: DeviceOpenBLAS,
    pub dm: &'a Vec<MatrixFull<f64>>, 
    pub mo_coeffs: Option<Vec<MatrixFull<f64>>>,
    pub occ: Option<Vec<Vec<f64>>>,
}


pub struct GridIterator<'a> {
    mol: &'a Molecule,
    grids: &'a Grids,
    settings: BlockSettings,
    current: usize,
    total_grids: usize,
}

impl<'a> GridIterator<'a> {
    pub fn new(
        mol: &'a Molecule,
        grids: &'a Grids,
        settings: BlockSettings,
    ) -> Self {
        let total_grids = grids.weights.len();
        Self {
            mol,
            grids,
            settings,
            current: 0,
            total_grids,
        }
    }
}

impl<'a> Iterator for GridIterator<'a> {
    type Item = BlockData<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.total_grids {
            return None;
        }
        let start = self.current;
        let end = (start + self.settings.blksize).min(self.total_grids);
        self.current = end;

        let coords = &self.grids.coordinates[start..end];
        let weights = &self.grids.weights[start..end];
        let num_grids = weights.len();
        let ao = eval_ao_batch(self.mol, coords, self.settings.ao_deriv, num_grids);

        Some(BlockData{
            coords: &self.grids.coordinates[start..end],
            weights: &self.grids.weights[start..end],
            ao: ao,
        })
    }
}

pub struct GridParallelIterator<'a> {
    mol: &'a Molecule,
    grids: &'a Grids,
    settings: BlockSettings,
    current: Arc<Mutex<usize>>, 
    total_grids: usize,
}

impl<'a> GridParallelIterator<'a> {
    pub fn new(
        mol: &'a Molecule,
        grids: &'a Grids,
        settings: BlockSettings,
    ) -> Self {
        let total_grids = grids.coordinates.len() / 3;
        Self {
            mol,
            grids,
            settings,
            current: Arc::new(Mutex::new(0)),
            total_grids,
        }
    }
}

// impl<'a> ParallelIterator for GridParallelIterator<'a> {
//     type Item = BlockData<'a>;

//     fn drive_unindexed<C>(self, consumer: C) -> C::Result
//     where
//         C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
//     {
//         rayon::iter::plumbing::bridge(self, consumer)
//     }

//     fn opt_len(&self) -> Option<usize> {
//         Some((self.total_grids + self.settings.blksize - 1) / self.settings.blksize)
//     }
// }

// impl<'a> rayon::iter::plumbing::ParallelIteratorBridge for GridParallelIterator<'a> {
//     fn fold_with<F>(self, folder: F) -> F
//     where
//         F: rayon::iter::plumbing::Folder<Self::Item>,
//     {
//         let blksize = self.settings.blksize;
//         let mut current = self.current.lock().unwrap();
            
//         while *current < self.total_grids {
//             let start = *current;
//             let end = (start + blksize).min(self.total_grids);
//             *current = end; 

//             drop(current);
                
//             let block = BlockData {
//                 ao: eval_ao_batch(
//                     self.mol,
//                     &self.grids.coordinates.slice(s![start..end]),
//                     self.settings.ao_deriv,
//                 ),
//                 weights: self.grids.weights.slice(s![start..end]),
//                 coords: self.grids.coordinates.slice(s![start..end]),
//             };

//             let folder = folder.consume(block);
//             current = self.current.lock().unwrap();
//         }
//         folder 
//     }
// }

pub trait NumInt<'a> {

    fn gen_xc_data(&'a self, scf_data:&'a SCF, spin:usize) -> XCData<'a>;

    fn get_vxc(&'a self, xc_data: &'a XCData, grids: &mut Grids, mol: &Molecule, spin: usize) -> (Vec<f64>, Vec<f64>, Vec<MatrixFull<f64>>);

    fn get_fxc(&'a self, xc_data0: &'a XCData, xc_data1: &'a XCData, grids: &mut Grids, mol: &Molecule, spin: usize) -> Vec<MatrixFull<f64>>;

    fn block_loop(
        &'a self,
        mol: &'a Molecule,
        grids: &'a mut Grids,
        settings: BlockSettings,
    ) -> GridIterator<'a> {
        GridIterator::new(mol, grids, settings)
    }

    fn par_blook_loop(
        &'a self,
        mol: &'a Molecule,
        grids: &'a mut Grids,
        settings: BlockSettings,
    ) -> GridParallelIterator<'a> {
        GridParallelIterator::new(mol, grids, settings)
    }
}


impl<'a> NumInt<'a> for DFA4REST {
    fn gen_xc_data(&'a self, scf_data: &'a SCF, spin: usize) -> XCData<'a> {
        let xc_type = if self.use_kinetic_density() {
            XCType::MGGA
        } else if self.use_density_gradient() {
            XCType::GGA
        } else {
            XCType::LDA
        };

        let xc_code = &self.dfa_compnt_scf;
        let xc_params = &self.dfa_paramr_scf;

        let density_matrix = &scf_data.density_matrix;
        let mo_coeffs = scf_data.eigenvectors.clone().to_vec();
        let mo_occ = scf_data.occupation.clone().to_vec();

        let device = DeviceOpenBLAS::default();

        XCData {
            xc_type,
            xc_code,
            xc_params,
            device: device,
            dm: density_matrix,
            mo_coeffs: Some(mo_coeffs),
            occ: Some(mo_occ),
        }
    }

    fn get_fxc(
        &'a self,
        xc_data0: &'a XCData,
        xc_data1: &'a XCData,
        grids: &mut Grids,
        mol: &Molecule,
        spin: usize,
    ) -> Vec<MatrixFull<f64>> {
        todo!("get_fxc has not yet implemented for DFA4REST");
    }

    fn get_vxc(
        &'a self,
        xc_data: &'a XCData,
        grids: &mut Grids,
        mol: &Molecule,
        spin: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<MatrixFull<f64>>) {

        let num_grids = grids.weights.len();
        let num_basis = mol.num_basis;
        // determine block settings 
        let ao_deriv = match xc_data.xc_type {
            XCType::HF => 0usize,
            XCType::LDA => 0usize,
            XCType::GGA => 1usize,
            XCType::MGGA => 1usize,
        };
        let max_memory = if let Some(max_memory) = mol.ctrl.max_memory {
            max_memory as usize
        } else {
            2000 as usize // default to 2000 MB
        };  
        let ao_comp = (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) / 6; // number of components for ao derivatives 
        let blksize = max_memory * 1_000_000 / 8 / ((ao_comp + 1) * num_basis);
        let blksize = blksize.min(num_grids).max(4); 
        let block_settings = BlockSettings {
            ao_deriv,
            max_memory,
            blksize,
        };

        let device = &xc_data.device;
        
        let mut nelec = vec![0.0; spin+1];
        let mut exc_sum = vec![0.0; spin+1];
        let mut vxc = rt::zeros(([num_basis, num_basis, spin + 1], device));
        let mut vxc_mgga_temp = match xc_data.xc_type {
            XCType::MGGA => Some(rt::zeros(([num_basis, num_basis, spin + 1], device))),
            _ => None,
        };
        let grids_iterator = self.block_loop(mol, grids, block_settings);
        // calculation starts here 
        let deriv = 1usize;          
        grids_iterator.into_iter().for_each(
            |block| 
            {
                let num_grids = block.weights.len();
                let loc_rho = if let Some(mo_coeffs) = &xc_data.mo_coeffs {
                    eval_rho5_batch(&block.ao, xc_data.xc_type, mo_coeffs, xc_data.occ.as_ref().unwrap(), spin, num_grids)
                } else {
                    eval_rho5_dm_only_batch(&block.ao, xc_data.xc_type, xc_data.dm, spin, num_grids)
                };
                let shape = loc_rho.shape().clone();

                let rho_array = loc_rho.into_vec();
                let loc_xc_tensor = eval_xc_eff(xc_data.xc_code, xc_data.xc_params, xc_data.xc_type, spin, &rho_array, num_grids, deriv);

                let loc_rho = rt::asarray((rho_array, shape, device));
                // let loc_exc = rt::asarray((&loc_xc_tensor[0].unwrap()));
                // let loc_vxc = rt::asarray((&loc_xc_tensor[1].unwrap()));

                let loc_exc = if let Some(exc) = loc_xc_tensor[0].as_ref() {
                    rt::asarray(exc)
                } else {
                    panic!("Exchange-correlation energy is not provided for the given xc_type: {:?}", xc_data.xc_type);
                };

                let loc_vxc = if let Some(vxc) = loc_xc_tensor[1].as_ref() {
                    rt::asarray(vxc)
                } else {
                    panic!("Exchange-correlation potential is not provided for the given xc_type: {:?}", xc_data.xc_type);
                };

                let loc_weights = rt::asarray((block.weights, device));

                let ao_shape = vec![num_basis, num_grids, ao_comp];
                let loc_ao = rt::asarray((&block.ao.data, ao_shape, device));
                

                // let loc_exc = loc_xc_tensor[0].as_ref().unwrap();
                // let loc_vxc = loc_xc_tensor[1].as_ref().unwrap();

                match spin {
                    0 => {
                        let loc_rho0 = loc_rho.i((0..num_grids, 0, 0));
                        let loc_den = loc_rho0 * &loc_weights;
                        let loc_nelec = rt::sum(&loc_den);
                        nelec[0] += loc_nelec;
                        let loc_exc_sum = (loc_exc * &loc_den).sum();
                        exc_sum[0] += loc_exc_sum;

                        let mut loc_wv = loc_vxc * &loc_weights.i((.., None)); 

                        match xc_data.xc_type {
                            XCType::LDA => {
                                let aow = &loc_ao.i((.., .., 0)) * &loc_wv; 
                                vxc.i_mut((.., .., 0)).matmul_from(
                                    &aow, &loc_ao.i((.., .., 0)).t(), 1.0, 1.0
                                );

                            },
                            XCType::GGA => {
                                loc_wv.i_mut((.., 0)).mul_assign(0.5);
                                
                                let aow = &loc_ao * &loc_wv.i((.., None, ..));
                                for ic in 0..=3 {
                                    vxc.i_mut(0).matmul_from(
                                        &aow.i((.., .., ic)), &loc_ao.i((.., .., ic)).t(), 1.0, 1.0
                                    );
                                }
                            },
                            XCType::MGGA => {
                                loc_wv.i_mut((.., 0)).mul_assign(0.5);
                                loc_wv.i_mut((.., 4)).mul_assign(0.5);

                                for ic in 0..=3 {
                                    let aow = &loc_ao.i((.., .., ic)) * &loc_wv.i((.., None, ic));
                                    vxc.i_mut(0).matmul_from(
                                        &loc_ao.i((.., .., 0)), &aow.t(), 1.0, 1.0
                                    );
                                }
                                let Some(v1) = vxc_mgga_temp.as_mut() else {
                                    panic!("vxc_mgga_temp is not initialized for MGGA type");
                                };
                                for ic in 0..=3 {
                                    let vtau_ket = &loc_ao.i((.., .., ic+1)) * &loc_wv.i((.., None, 4));
                                    v1.i_mut(0).matmul_from(
                                        &loc_ao.i((.., .., ic+1)), &vtau_ket.t(), 1.0, 1.0
                                    );
                                }
                            },
                            XCType::HF => {},
                        }
                    },
                    1 => {
                        let loc_rho_a = loc_rho.i((0..num_grids, 0, 0));
                        let loc_rho_b = loc_rho.i((0..num_grids, 0, 1));
                        let loc_den_a = loc_rho_a * &loc_weights;
                        let loc_den_b = loc_rho_b * &loc_weights;
                        let loc_nelec_a = rt::sum(&loc_den_a);
                        let loc_nelec_b = rt::sum(&loc_den_b);
                        nelec[0] += loc_nelec_a;
                        nelec[1] += loc_nelec_b;
                        let loc_exc_sum_a = (&loc_exc * &loc_den_a).sum();
                        let loc_exc_sum_b = (&loc_exc * &loc_den_b).sum();
                        exc_sum[0] += loc_exc_sum_a;
                        exc_sum[1] += loc_exc_sum_b;

                        //loc_vxc shape [num_grids, nderiv ,spin] nderiv = 1, 4, 5 for LDA (vrho), GGA (vrho, vnabla_rho), MGGA (.., vtau)
                        let mut loc_wv = &loc_vxc * &loc_weights.i((.., None, None)); 

                        match xc_data.xc_type {
                            XCType::LDA => {
                                let loc_wv_a = loc_wv.i((.., 0, 0));
                                let loc_wv_b = loc_wv.i((.., 0, 1));
                                let aow_a = &loc_ao.i((.., .., 0)) * &loc_wv_a.i((.., None)); 
                                let aow_b = &loc_ao.i((.., .., 0)) * &loc_wv_b.i((.., None));
                                vxc.i_mut((.., .., 0)).matmul_from(
                                    &aow_a, &loc_ao.i((.., .., 0)).t(), 1.0, 1.0
                                );
                                vxc.i_mut((.., .., 1)).matmul_from(
                                    &aow_b, &loc_ao.i((.., .., 0)).t(), 1.0, 1.0
                                );

                            },
                            XCType::GGA => {
                                loc_wv.i_mut((.., 0, ..)).mul_assign(0.5);
                                let loc_wv_a = loc_wv.i((.., .., 0));
                                let aow_a = &loc_ao.i((.., .., 0..4)) * &loc_wv_a.i((.., None, ..));
                                let loc_wv_b = loc_wv.i((.., .., 1));
                                let aow_b = &loc_ao .i((.., .., 0..4)) * &loc_wv_b.i((.., None, ..));
                                for ic in 0..4 {
                                    vxc.i_mut((.., .., 0)).matmul_from(
                                        &loc_ao.i((.., .., 0)), &aow_a.i((.., .., ic)).t(), 1.0, 1.0
                                    );
                                    vxc.i_mut((.., .., 1)).matmul_from(
                                        &loc_ao.i((.., .., 0)), &aow_b.i((.., .., ic)).t(), 1.0, 1.0
                                    );
                                }
                            },
                            XCType::MGGA => {
                                loc_wv.i_mut((.., 0, ..)).mul_assign(0.5);
                                loc_wv.i_mut((.., 4, ..)).mul_assign(0.5);
                                let Some(v1) = vxc_mgga_temp.as_mut() else {
                                    panic!("vxc_mgga_temp is not initialized for MGGA type");
                                };

                                let loc_wv_a = loc_wv.i((.., .., 0));
                                let aow_a = &loc_ao.i((.., .., 0..4)) * &loc_wv_a.i((.., None, 0..4));
                                let loc_wv_b = loc_wv.i((.., .., 1));
                                let aow_b = &loc_ao .i((.., .., 0..4)) * &loc_wv_b.i((.., None, 0..4));

                                for ic in 0..4 {
                                    vxc.i_mut((.., .., 0)).matmul_from(
                                        &loc_ao.i((.., .., 0)), &aow_a.i((.., .., ic)).t(), 1.0, 1.0
                                    );
                                    vxc.i_mut((.., .., 1)).matmul_from(
                                        &loc_ao.i((.., .., 0)), &aow_b.i((.., .., ic)).t(), 1.0, 1.0
                                    );
                                    let vtau_ket_a = &loc_ao.i((.., .., ic+1)) * &loc_wv_a.i((.., None, 4));
                                    let vtau_ket_b = &loc_ao.i((.., .., ic+1)) * &loc_wv_b.i((.., None, 4));
                                    v1.i_mut((.., .., 0)).matmul_from(
                                        &loc_ao.i((.., .., ic+1)), &vtau_ket_a.t(), 1.0, 1.0
                                    );
                                    v1.i_mut((.., .., 1)).matmul_from(
                                        &loc_ao.i((.., .., ic+1)), &vtau_ket_b.t(), 1.0, 1.0
                                    );
                                }
                            },
                            XCType::HF => {},
                        }

                    },
                    _ => unreachable!(),
                }
            }
        ); // end block loop

        let mut vmat = vec!();

        match xc_data.xc_type {
            XCType::LDA => {},
            XCType::GGA => {
                let vxc_trans = vxc.transpose([0, 2, 1]);
                vxc = &vxc + vxc_trans;
            },
            XCType::MGGA => {
                let vxc_trans = vxc.transpose([0, 2, 1]);
                vxc = &vxc + vxc_trans;
                let Some(v1) = vxc_mgga_temp.as_ref() else {
                    panic!("vxc_mgga_temp is not initialized for MGGA type");
                };
                vxc.add_assign(v1);
            },
            XCType::HF => {
                unreachable!()
            },
        }

        vmat.push(MatrixFull::from_vec([num_basis, num_basis], vxc.i(0).to_vec()).unwrap());
        if spin == 1 {
            vmat.push(MatrixFull::from_vec([num_basis, num_basis], vxc.i(1).to_vec()).unwrap());
        }

        (nelec, exc_sum, vmat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_lda_data(nocc: usize, nvir: usize, ngrids: usize) -> FXCMatvecData {
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
            use_opt: false,
        }
    }

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

        let nv2 = 16;
        let mut wfxc = vec![0.0; ngrids * nv2];
        for g in 0..ngrids {
            let weight = (g as f64 + 1.0).sqrt() * 0.01;
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
            use_opt: false,
        }
    }

    fn test_fxc_lda(nocc: usize, nvir: usize, ngrids: usize) {
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

    fn test_fxc_gga(nocc: usize, nvir: usize, ngrids: usize) {
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
    fn test_fxc_lda_small() { test_fxc_lda(3, 5, 10); }
    #[test]
    fn test_fxc_lda_medium() { test_fxc_lda(10, 20, 100); }
    #[test]
    fn test_fxc_lda_zero_z() {
        let data = make_lda_data(3, 5, 10);
        let z = vec![0.0; 15];
        let result = fxc_matvec(&data, &z);
        for &v in &result { assert!((v).abs() < 1e-15); }
    }
    #[test]
    fn test_fxc_gga_small() { test_fxc_gga(3, 5, 10); }
    #[test]
    fn test_fxc_gga_medium() { test_fxc_gga(10, 20, 100); }
    #[test]
    fn test_fxc_gga_zero_z() {
        let data = make_gga_data(3, 5, 10);
        let z = vec![0.0; 15];
        let result = fxc_matvec(&data, &z);
        for &v in &result { assert!((v).abs() < 1e-15); }
    }
    #[test]
    fn test_fxc_gga_symmetry() {
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
        let data = make_lda_data(5, 8, 30);
        let n = 40;
        let z: Vec<f64> = (0..n).map(|i| if i < 10 { 1.0 } else { 0.0 }).collect();
        let result = fxc_matvec(&data, &z);
        assert!(result.iter().all(|x| x.is_finite()));
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

    // ═════════════════════════════════════════════════════════════════
    // Tests for fxc_matvec_opt
    // ═════════════════════════════════════════════════════════════════

    fn setup_rayon_threads(n: usize) -> Option<rayon::ThreadPool> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .ok()
    }

    fn compare_results(name: &str, expected: &[f64], got: &[f64]) {
        assert_eq!(expected.len(), got.len(), "{}: length mismatch", name);
        for (i, (e, g)) in expected.iter().zip(got.iter()).enumerate() {
            let diff = (e - g).abs();
            assert!(
                diff < 1e-14,
                "{}: mismatch at [{}]: expected {} got {} (diff {:.2e})",
                name,
                i,
                e,
                g,
                diff
            );
        }
    }

    fn test_opt_lda_small(nocc: usize, nvir: usize, ngrids: usize) {
        let data = make_lda_data(nocc, nvir, ngrids);
        let n = nocc * nvir;
        let z: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) * 0.01).collect();
        let expected = fxc_matvec(&data, &z);
        let mut ws = prepare_fxc_workspace(&data);
        let got = fxc_matvec_opt(&data, &z, &mut ws);
        compare_results("LDA opt small", &expected, &got);
        println!("LDA opt small nocc={} nvir={} ngrids={}: ✓", nocc, nvir, ngrids);
    }

    fn test_opt_gga_small(nocc: usize, nvir: usize, ngrids: usize) {
        let data = make_gga_data(nocc, nvir, ngrids);
        let n = nocc * nvir;
        let z: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) * 0.01).collect();
        let expected = fxc_matvec(&data, &z);
        let mut ws = prepare_fxc_workspace(&data);
        let got = fxc_matvec_opt(&data, &z, &mut ws);
        compare_results("GGA opt small", &expected, &got);
        println!("GGA opt small nocc={} nvir={} ngrids={}: ✓", nocc, nvir, ngrids);
    }

    fn test_opt_lda_multiple_calls(nocc: usize, nvir: usize, ngrids: usize) {
        let data = make_lda_data(nocc, nvir, ngrids);
        let n = nocc * nvir;
        let mut ws = prepare_fxc_workspace(&data);
        for trial in 0..5 {
            let z: Vec<f64> = (0..n)
                .map(|i| ((i + trial * 17) % 11) as f64 * 0.01)
                .collect();
            let expected = fxc_matvec(&data, &z);
            let got = fxc_matvec_opt(&data, &z, &mut ws);
            compare_results(&format!("LDA reuse trial {}", trial), &expected, &got);
        }
        println!("LDA opt multiple-call reuse nocc={} nvir={} ngrids={}: ✓", nocc, nvir, ngrids);
    }

    fn test_opt_gga_multiple_calls(nocc: usize, nvir: usize, ngrids: usize) {
        let data = make_gga_data(nocc, nvir, ngrids);
        let n = nocc * nvir;
        let mut ws = prepare_fxc_workspace(&data);
        for trial in 0..5 {
            let z: Vec<f64> = (0..n)
                .map(|i| ((i + trial * 17) % 11) as f64 * 0.01)
                .collect();
            let expected = fxc_matvec(&data, &z);
            let got = fxc_matvec_opt(&data, &z, &mut ws);
            compare_results(&format!("GGA reuse trial {}", trial), &expected, &got);
        }
        println!("GGA opt multiple-call reuse nocc={} nvir={} ngrids={}: ✓", nocc, nvir, ngrids);
    }

    fn bench_opt(
        label: &str,
        nocc: usize,
        nvir: usize,
        ngrids: usize,
        ntrials: usize,
        nthreads: Option<usize>,
    ) {
        let pool = nthreads.and_then(|n| setup_rayon_threads(n));

        let data = if label.contains("GGA") {
            make_gga_data(nocc, nvir, ngrids)
        } else {
            make_lda_data(nocc, nvir, ngrids)
        };
        let n = nocc * nvir;
        let z: Vec<f64> = (0..n).map(|i| (i as f64 % 13.0) * 0.01).collect();
        let mut ws = prepare_fxc_workspace(&data);

        // Warm-up
        for _ in 0..3 {
            fxc_matvec(&data, &z);
            fxc_matvec_opt(&data, &z, &mut ws);
        }

        // Benchmark original (sequential)
        let start = std::time::Instant::now();
        for _ in 0..ntrials {
            std::hint::black_box(fxc_matvec(&data, &z));
        }
        let orig_time = start.elapsed().as_secs_f64() / ntrials as f64;

        // Benchmark optimized (may use rayon threads inside pool)
        let start = std::time::Instant::now();
        if let Some(ref pool) = pool {
            pool.install(|| {
                for _ in 0..ntrials {
                    std::hint::black_box(fxc_matvec_opt(&data, &z, &mut ws));
                }
            });
        } else {
            for _ in 0..ntrials {
                std::hint::black_box(fxc_matvec_opt(&data, &z, &mut ws));
            }
        }
        let opt_time = start.elapsed().as_secs_f64() / ntrials as f64;

        let speedup = orig_time / opt_time;
        let threads_str = nthreads.map_or("default".to_string(), |n| n.to_string());
        println!(
            "{} (nocc={} nvir={} ngrids={}, nthreads={}): \
             orig={:.4}s opt={:.4}s speedup={:.2}×",
            label, nocc, nvir, ngrids, threads_str, orig_time, opt_time, speedup
        );
    }

    // ── Correctness tests ──

    #[test]
    fn test_fxc_opt_lda_correctness() {
        test_opt_lda_small(3, 5, 10);
        test_opt_lda_small(10, 20, 100);
        test_opt_lda_small(50, 100, 500);
    }

    #[test]
    fn test_fxc_opt_gga_correctness() {
        test_opt_gga_small(3, 5, 10);
        test_opt_gga_small(10, 20, 100);
        test_opt_gga_small(20, 30, 200);
    }

    #[test]
    fn test_fxc_opt_lda_workspace_reuse() {
        test_opt_lda_multiple_calls(10, 20, 100);
    }

    #[test]
    fn test_fxc_opt_gga_workspace_reuse() {
        test_opt_gga_multiple_calls(10, 20, 100);
    }

    #[test]
    fn test_fxc_opt_lda_performance() {
        bench_opt("LDA", 50, 200, 10_000, 30, Some(8));
    }

    #[test]
    fn test_fxc_opt_gga_performance() {
        bench_opt("GGA", 30, 80, 5_000, 15, Some(8));
    }
}

