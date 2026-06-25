use num_complex::Complex;
use rest_tensors::{MatrixFull, MatrixUpper};
use tensors::BasicMatrix;
use tensors::matrix_blas_lapack::{_dgemm_full, _dspgvx, _dsyev};
use rest_libcint::prelude::*;
use rest_libcint::CINTR2CDATA;
use rest_libcint_wrapper::*;
use crate::basis_io::BasInfo;
use crate::constants::ENV_PRT_START;
use crate::molecule_io::{Molecule, get_basis_name};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RelativisticMethod {
    None,
    SFX2C,  // spin-free X2C
    // X2C,
}

impl Molecule {
    // Decontract the basis set: expand each contracted shell into individual primitive shells.
    // Returns (uncontracted_molecule, contraction_coefficient_matrix).
    
    // The contraction matrix C maps uncontracted basis functions to contracted basis functions:
    // χ_j = Σ_i C[i,j] * φ_i
    // where φ_i are normalized primitive Gaussians and χ_j are the original contracted functions.
    // After computing the X2C Hamiltonian in the uncontracted basis, contract back via:
    // h_contracted = C^T @ h_uncontracted @ C

    pub fn decontract_basis(&self) -> (Molecule, MatrixFull<f64>) {
        let natm = self.cint_atm.len();
        let env_atom_end = ENV_PRT_START as usize + natm * 4;

        let mut new_env: Vec<f64> = self.cint_env[..env_atom_end].to_vec();
        let mut new_bas: Vec<Vec<i32>> = vec![];
        let mut new_fdqc: Vec<Vec<usize>> = vec![];
        let mut new_fdqc_bas: Vec<BasInfo> = vec![];
        let mut new_basis_start = env_atom_end as i32;

        let mut c_blocks: Vec<MatrixFull<f64>> = vec![];
        let mut row_offsets: Vec<usize> = vec![0];
        let mut col_offsets: Vec<usize> = vec![0];
        let mut cur_uncontracted = 0_usize;
        let mut cur_contracted = 0_usize;

        let n_ang = |ang_mom: i32| -> usize {
            match self.cint_type {
                CintType::Spheric => (ang_mom * 2 + 1) as usize,
                CintType::Cartesian => ((ang_mom + 1) * (ang_mom + 2) / 2) as usize,
                CintType::Spinor => panic!("Spinor not yet implemented"),
            }
        };

        for bas_row in self.cint_bas.iter() {
            let atom_idx = bas_row[0] as usize;
            let ang_mom = bas_row[1];
            let n_prim = bas_row[2] as usize;
            let n_ctr = bas_row[3] as usize;
            let ptr_exp = bas_row[5] as usize;
            let ptr_coeff = bas_row[6] as usize;
            let n_angular = n_ang(ang_mom);

            let exponents: Vec<f64> = self.cint_env[ptr_exp..ptr_exp + n_prim].to_vec();

            let mut raw_coeffs = vec![vec![0.0f64; n_prim]; n_ctr];
            for c in 0..n_ctr {
                for p in 0..n_prim {
                    let env_coeff = self.cint_env[ptr_coeff + c * n_prim + p];
                    let norm = CINTR2CDATA::gto_norm(ang_mom, exponents[p]);
                    raw_coeffs[c][p] = env_coeff / norm;
                }
            }

            let shell_start_bas_idx = new_bas.len();
            for p in 0..n_prim {
                let norm = CINTR2CDATA::gto_norm(ang_mom, exponents[p]);

                new_env.push(exponents[p]);
                new_env.push(norm);

                let new_bas_row = vec![
                    atom_idx as i32,
                    ang_mom,
                    1,
                    1,
                    0,
                    new_basis_start,
                    new_basis_start + 1,
                    0,
                ];
                new_bas.push(new_bas_row);
                new_basis_start += 2;

                let tmp_start = if new_fdqc.is_empty() {
                    0
                } else {
                    new_fdqc[new_fdqc.len() - 1][0] + new_fdqc[new_fdqc.len() - 1][1]
                };
                new_fdqc.push(vec![tmp_start, n_angular]);

                for m in 0..n_angular {
                    new_fdqc_bas.push(BasInfo {
                        bas_name: get_basis_name(ang_mom as usize, &self.cint_type, m),
                        bas_type: String::from("Primitive"),
                        elem_index0: atom_idx,
                        cint_index0: new_bas.len() - 1,
                        cint_index1: m,
                    });
                }
            }

            let block_rows = n_prim * n_angular;
            let block_cols = n_ctr * n_angular;
            let mut block = MatrixFull::<f64>::new([block_rows, block_cols], 0.0);

            for p in 0..n_prim {
                for c in 0..n_ctr {
                    for m in 0..n_angular {
                        block[(p * n_angular + m, c * n_angular + m)] = raw_coeffs[c][p];
                    }
                }
            }

            let prev_row = *row_offsets.last().unwrap();
            let prev_col = *col_offsets.last().unwrap();
            row_offsets.push(prev_row + block_rows);
            col_offsets.push(prev_col + block_cols);

            c_blocks.push(block);
            cur_uncontracted += block_rows;
            cur_contracted += block_cols;
        }

        let n_total_uncontracted = cur_uncontracted;
        let n_total_contracted = cur_contracted;
        let mut contr_coeff =
            MatrixFull::<f64>::new([n_total_uncontracted, n_total_contracted], 0.0);

        for (i, block) in c_blocks.iter().enumerate() {
            let r0 = row_offsets[i];
            let c0 = col_offsets[i];
            let (br, bc) = (block.size()[0], block.size()[1]);
            for r in 0..br {
                for c in 0..bc {
                    contr_coeff[(r0 + r, c0 + c)] = block[(r, c)];
                }
            }
        }

        let mut xmol = self.clone();
        xmol.num_basis = n_total_uncontracted;
        xmol.num_auxbas = 0;
        xmol.fdqc_bas = new_fdqc_bas;
        xmol.cint_fdqc = new_fdqc;
        xmol.cint_bas = new_bas;
        xmol.cint_env = new_env;
        xmol.cint_ecpbas = None;
        xmol.basis4elem = vec![];
        xmol.auxbas4elem = vec![];
        xmol.fdqc_aux_bas = vec![];
        xmol.cint_aux_fdqc = vec![];
        xmol.cint_aux_atm = vec![];
        xmol.cint_aux_bas = vec![];
        xmol.cint_aux_env = vec![];

        (xmol, contr_coeff)
    }

    
    pub fn generate_sfx2c_hamiltonian(&self) -> MatrixUpper<f64> {
        let light_speed: f64 = 137.03599967994;

        let (xmol, contr_coeff) = self.decontract_basis();
        let n_contracted = self.num_basis;
        let mut cint_data = xmol.initialize_cint(false);

        // Note that in spin-free X2C, all matrices are real so that "dagger" operator can be replaced with "transposition" operator
        // The generalized eigenvalue problem of hamiltion h and overlap matrix M: hE = MCE
        // h = [V       T    ] =  [h_ll h_ls]     M = [S     0    ] = [s_ll s_ls]
        //     [T W/(4*c^2)-T]    [h_sl h_ss]         [0 T/(2*c^2)] = [s_sl s_ss]
        // for spin-free X2C, the one-electron integrals are 
        // S = <μ|ν> (overlap)
        // T = 0.5<μ|p·pν> (kinetic)
        // V = <μ|V|ν> (nuc)
        // W = <σ·pμ|V|σ·pν> (pnucp)

        let (out_s, _) = cint_data.integral_s2ij::<int1e_ovlp>(None);
        let (out_t, _) = cint_data.integral_s2ij::<int1e_kin>(None);
        let (out_v, _) = cint_data.integral_s2ij::<int1e_nuc>(None);
        let (out_w, _) = cint_data.integral_s2ij::<int1e_pnucp>(None);

        let s = MatrixUpper::from_vec(out_s.len(), out_s).unwrap().to_matrixfull().unwrap();
        let t = MatrixUpper::from_vec(out_t.len(), out_t).unwrap().to_matrixfull().unwrap();
        let v = MatrixUpper::from_vec(out_v.len(), out_v).unwrap().to_matrixfull().unwrap();
        let factor_h_ss = 0.25 / light_speed.powi(2);
        let w = MatrixUpper::from_vec(out_w.len(), out_w).unwrap().to_matrixfull().unwrap() * factor_h_ss;

        let n_uncontracted = s.size()[0];

        // h = [h_ll h_ls], M = [s_ll s_ls], 
        //     [h_sl h_ss]      [s_sl s_ss]

        // 4-component hamiltonian: h = [V, T; T, W/(4c^2)-T]

        let h_ss = w.clone() - t.clone();
        let four_component_h = merge_matrix(&v, &t, &t, &h_ss);

        // 4-component overlap: M = [S, 0; 0, T/(2c^2)]

        let s_ls = MatrixFull::new([n_uncontracted, n_uncontracted], 0.0);
        let s_sl = MatrixFull::new([n_uncontracted, n_uncontracted], 0.0);
        let factor_s_ss = 0.5 / light_speed.powi(2);
        let s_ss = t.clone() * factor_s_ss;
        let four_component_overlap = merge_matrix(&s, &s_ls, &s_sl, &s_ss);

        // Solve the generalized eigenvalue problem: hE = MCE 
        //     [h_ll h_ls] [C_L^- C_L^+] = [s_ll s_ls] [C_L^- C_L^+] [E^-   0 ]
        //     [h_sl h_ss] [C_S^- C_S^+]   [s_sl s_ss] [C_S^- C_S^+] [0    E^+]

        let (eigenvector_ghf, eigenvalue_ghf, _dim_out) =
            _dspgvx(&four_component_h.to_matrixupper(), &four_component_overlap.to_matrixupper(), 2 * n_uncontracted)
            .unwrap();

        // C = [C_L^-, C_L^+; C_S^-, C_S^+]

        let (_c_large_minus, mut c_large_plus, _c_small_minus, _c_small_plus) =
            split_matrix_by_component(&eigenvector_ghf);

                 // The eigenvalues are in ascending order, discard the negetive ones
        let mut e_positive = MatrixFull::<f64>::new([n_uncontracted, n_uncontracted], 0.0);
        for i in 0..n_uncontracted {
            e_positive[(i, i)] = eigenvalue_ghf[i + n_uncontracted];
        }

        // Taking C_L^+ (following referred as C) matrix as basis and rewrite the FW Hcore formula, to avoid inversing matrix
        // R^dag \tilde{S} R = S and 
        // R = S^{-1/2} [S^{-1/2}\tilde{S}S^{-1/2}]^{-1/2} S^{1/2}

        // Using C matrix as basis, the representation of R is
        // R[C] = (C^dag S C)^{1/2} = (C^dag S C)^{-1/2} C^dag S C
        // Construct h = R^C h1 C in two steps, first in basis C matrix, then transform back to AO basis
        // h  = (C^dag)^{-1} R[C]^dag (C^dag h1 C) R[C] C^{-1}         (0)
        // Using (C^dag)^{-1} = \tilde{S} C, h can be transformed to
        // h  = \tilde{S} C R[C]^dag C^dag h1 C R[C] C^dag \tilde{S}   (1)
        // Using R[C] = R[C]^{-1} C^dag S C,  Eq (0) turns to
        //      = S C R[C]^{-1}^dag C^dag h1 C R[C]^{-1} C^dag S
        //      = S C R[C]^{-1}^dag e_positive R[C]^{-1} C^dag S       (2)

        // Caculating the eigenvalue equation of C^\dag S C so that
        // C^dag S C = U W U^dag (U are eigenvectors and W are eigenvalues)
        let mut cl_s = MatrixFull::<f64>::new([n_uncontracted, n_uncontracted], 0.0);
        let mut cl_s_cl = MatrixFull::<f64>::new([n_uncontracted, n_uncontracted], 0.0);
        _dgemm_full(&c_large_plus, 'T', &s, 'N', &mut cl_s, 1.0, 0.0);
        _dgemm_full(&cl_s, 'N', &c_large_plus, 'N', &mut cl_s_cl, 1.0, 0.0);

        let (eigenvector_csc_opt, eigenvalue_csc, _dim_out2) = _dsyev(&cl_s_cl, 'V');
        let eigenvector_csc = eigenvector_csc_opt.unwrap();

        // R[C] = (C^dag S C)^{1/2} = U W U^\dag   (3)
        let mut eigenvalue_csc_invsqrt = MatrixFull::<f64>::new([n_uncontracted, n_uncontracted], 0.0);
        for i in 0..n_uncontracted {
            eigenvalue_csc_invsqrt[(i, i)] = 1.0 / eigenvalue_csc[i].sqrt();
        }

        // If r = R[C]^{-1} C^dag S = U W^{-1/2} U^dag A^dag S, eq(2) turns to 
        // h = r^dag e_positive r, which is exactly the spin-free x2c hamiltonian under AO basis

        // r = U W^{-1/2} U^dag A^dag S
        let mut r = MatrixFull::<f64>::new([n_uncontracted, n_uncontracted], 0.0);
        let mut r1 = MatrixFull::<f64>::new([n_uncontracted, n_uncontracted], 0.0);
        let mut r2 = MatrixFull::<f64>::new([n_uncontracted, n_uncontracted], 0.0);
        let mut r3 = MatrixFull::<f64>::new([n_uncontracted, n_uncontracted], 0.0);
        _dgemm_full(&eigenvector_csc, 'N', &eigenvalue_csc_invsqrt, 'N', &mut r1, 1.0, 0.0);
        _dgemm_full(&r1, 'N', &eigenvector_csc, 'T', &mut r2, 1.0, 0.0);
        _dgemm_full(&r2, 'N', &c_large_plus, 'T', &mut r3, 1.0, 0.0);
        _dgemm_full(&r3, 'N', &s, 'N', &mut r, 1.0, 0.0);

        // h = r^dag e_positive r
        let mut h1 = MatrixFull::<f64>::new([n_uncontracted, n_uncontracted], 0.0);
        let mut h_ao_uncontracted = MatrixFull::<f64>::new([n_uncontracted, n_uncontracted], 0.0);
        _dgemm_full(&r, 'T', &e_positive, 'N', &mut h1, 1.0, 0.0);
        _dgemm_full(&h1, 'N', &r, 'N', &mut h_ao_uncontracted, 1.0, 0.0);

        // Contract back to original basis: h_contracted = C^T h_uncontracted C
        let mut h_temp = MatrixFull::<f64>::new([n_contracted, n_uncontracted], 0.0);
        let mut h_ao_contracted = MatrixFull::<f64>::new([n_contracted, n_contracted], 0.0);
        _dgemm_full(&contr_coeff, 'T', &h_ao_uncontracted, 'N', &mut h_temp, 1.0, 0.0);
        _dgemm_full(&h_temp, 'N', &contr_coeff, 'N', &mut h_ao_contracted, 1.0, 0.0);

        h_ao_contracted.to_matrixupper()
    }
}

pub fn merge_matrix<T: Copy + Clone + Default>(
    block_ll: &MatrixFull<T>,
    block_ls: &MatrixFull<T>,
    block_sl: &MatrixFull<T>,
    block_ss: &MatrixFull<T>,
) -> MatrixFull<T> {
    assert!(
        block_ll.size() == block_ls.size()
            && block_ll.size() == block_sl.size()
            && block_ll.size() == block_ss.size(),
        "The dimension of four blocks must be identical!"
    );
    let size_block = block_ll.size()[0];
    let dim = 2 * size_block;
    let mut big = MatrixFull::<T>::new([dim, dim], T::default());
    let mut copy_block = |dst: &mut MatrixFull<T>,
                          row_offset: usize,
                          col_offset: usize,
                          block: &MatrixFull<T>| {
        for j in 0..size_block {
            let dst_col =
                &mut dst.slice_column_mut(col_offset + j)[row_offset..row_offset + size_block];
            let src_col = block.slice_column(j);
            dst_col.clone_from_slice(src_col);
        }
    };

    copy_block(&mut big, 0, 0, block_ll);
    copy_block(&mut big, 0, size_block, block_ls);
    copy_block(&mut big, size_block, 0, block_sl);
    copy_block(&mut big, size_block, size_block, block_ss);

    big
}

pub fn split_matrix_by_component<T: Copy + Clone + Default>(
    big: &MatrixFull<T>,
) -> (MatrixFull<T>, MatrixFull<T>, MatrixFull<T>, MatrixFull<T>) {
    let size_big = big.size();
    assert!(size_big[0] == size_big[1], "The input matrix must be square matrix!");
    assert!(size_big[0] % 2 == 0, "Dimension of the input matrix must be even!");

    let dim = size_big[0] / 2;

    let mut block_ll = MatrixFull::<T>::new([dim, dim], T::default());
    let mut block_ls = MatrixFull::<T>::new([dim, dim], T::default());
    let mut block_sl = MatrixFull::<T>::new([dim, dim], T::default());
    let mut block_ss = MatrixFull::<T>::new([dim, dim], T::default());

    for row in 0..dim {
        for col in 0..dim {
            block_ll[(row, col)] = big[(row, col)];
            block_ls[(row, col)] = big[(row, col + dim)];
            block_sl[(row, col)] = big[(row + dim, col)];
            block_ss[(row, col)] = big[(row + dim, col + dim)];
        }
    }

    (block_ll, block_ls, block_sl, block_ss)
}


pub fn split_matrix_by_spin<T: Copy + Clone + Default>(
    big: &MatrixFull<T>,
) -> (MatrixFull<T>, MatrixFull<T>, MatrixFull<T>, MatrixFull<T>) {
    let size_big = big.size();
    assert!(size_big[0] == size_big[1], "The input matrix must be square matrix!");
    assert!(size_big[0] % 2 == 0, "Dimension of the input matrix must be even!");

    let dim = size_big[0] / 2;

    let mut block_alpha_alpha = MatrixFull::<T>::new([dim, dim], T::default());
    let mut block_alpha_beta = MatrixFull::<T>::new([dim, dim], T::default());
    let mut block_beta_alpha = MatrixFull::<T>::new([dim, dim], T::default());
    let mut block_beta_beta = MatrixFull::<T>::new([dim, dim], T::default());

    for row in 0..dim {
        for col in 0..dim {
            block_alpha_alpha[(row, col)] = big[(2 * row, 2 * col)];
            block_alpha_beta[(row, col)] = big[(2 * row + 1, 2 * col)];
            block_beta_alpha[(row, col)] = big[(2 * row, 2 * col + 1)];
            block_beta_beta[(row, col)] = big[(2 * row + 1, 2 * col + 1)];
        }
    }

    (block_alpha_alpha, block_alpha_beta, block_beta_alpha, block_beta_beta)
}

