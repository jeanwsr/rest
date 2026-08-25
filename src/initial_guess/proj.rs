use rstsr::prelude::*;
use tensors::matrix::MatrixFull;
use tensors::BasicMatrix;
use crate::molecule_io::Molecule;
use crate::utilities::rstsr_util::{RestTensorToRstsrTsrAPI, RestTensorToRstsrViewAPI};

/// Solve S22 * X = B using Cholesky + triangular solve, with fallback to general solve.
/// Equivalent to pyscf's lib.cho_solve(a, b, strict_sym_pos=False).
fn cho_solve(s22: &MatrixFull<f64>, b: &MatrixFull<f64>, device: &DeviceBLAS) -> MatrixFull<f64> {
    let s22_tsr = s22.to_rstsr_view(device);
    let b_tsr = b.to_rstsr_view(device);

    match rt::linalg::cholesky_f((&s22_tsr, Lower)) {
        Ok(l) => {
            // println!("Cholesky decomposition succeeded, using triangular solve.");
            let y = rt::linalg::solve_triangular((&l, &b_tsr, Lower));
            let x = rt::linalg::solve_triangular((l.t(), &y, Upper));
            let shape: [usize; 2] = x.shape().to_vec().try_into().unwrap();
            MatrixFull::from_vec(shape, x.into_shape(-1).into_vec()).unwrap()
        },
        Err(_) => {
            println!("Cholesky decomposition failed, falling back to general solve.");
            let x = rt::linalg::solve_general((&s22_tsr, &b_tsr));
            let shape: [usize; 2] = x.shape().to_vec().try_into().unwrap();
            MatrixFull::from_vec(shape, x.into_shape(-1).into_vec()).unwrap()
        }
    }
}

/// Normalize MO columns so that mo^T S mo = I.
/// Equivalent to pyscf:
///   norm = numpy.einsum('pi,pi->i', mo.conj(), s.dot(mo))
///   mo /= numpy.sqrt(norm)
fn normalize_mo(mo: &mut MatrixFull<f64>, s: &MatrixFull<f64>, device: &DeviceBLAS) {
    let s_tsr = s.to_rstsr_view(device);
    let mo_tsr = mo.to_rstsr_view(device);

    let s_dot_mo = &s_tsr % &mo_tsr;
    let norm = (&mo_tsr * &s_dot_mo).sum_axes(0);

    let norm_vec: Vec<f64> = norm.to_vec();
    let inv_sqrt_norm: Vec<f64> = norm_vec.iter().map(|x| 1.0 / x.sqrt()).collect();
    let inv_sqrt_tsr = rt::asarray((inv_sqrt_norm.clone(), device));

    // println!("mo shape: {:?}, norm shape: {:?}, inv_sqrt_norm shape: {:?}", mo_tsr.shape(), norm.shape(), inv_sqrt_tsr.shape());
    // println!("inv_sqrt_norm: {:?}", inv_sqrt_norm[..5].to_vec());
    let mo_normalized = &mo_tsr * inv_sqrt_tsr.slice((None, ..));

    let shape: [usize; 2] = mo_normalized.shape().to_vec().try_into().unwrap();
    *mo = MatrixFull::from_vec(shape, mo_normalized.into_shape(-1).into_vec()).unwrap();
}

/// Project MO coefficients from a source molecule's basis to a target molecule's basis.
///
/// # Arguments
/// - `mol_target`: the target molecule (basis set 2)
/// - `mol_source`: the source molecule (basis set 1)
/// - `mo_source`: MO coefficients in the source basis
///
/// # Returns
/// - MO coefficients projected to the target basis (C2), normalized so that C2^T S22 C2 = I
pub fn proj_mo(mol_target: &Molecule, mol_source: &Molecule, mo_source: [MatrixFull<f64>;2]) -> [MatrixFull<f64>;2] {
    // S22 = target self-overlap
    let s22_full = mol_target.int_ij_matrixupper("ovlp".to_string()).to_matrixfull().unwrap();

    // S21 = <AO_target|AO_source>
    let s21 = mol_target.int_cross(mol_source, "ovlp".to_string());

    let device = DeviceBLAS::default();
    let s21_tsr = s21.to_rstsr(&device);

    let mut mo_target: [MatrixFull<f64>; 2] = [MatrixFull::empty(), MatrixFull::empty()];
    for spin in 0..2 {
        if mo_source[spin].size()[0] == 0 {
            continue;
        }
        let mo_source_tsr = mo_source[spin].to_rstsr_view(&device);
        let temp_tsr = &s21_tsr % &mo_source_tsr;
        let temp = MatrixFull::from_vec(
            temp_tsr.shape().to_vec().try_into().unwrap(),
            temp_tsr.into_shape(-1).into_vec(),
        ).unwrap();

        mo_target[spin] = cho_solve(&s22_full, &temp, &device);
        normalize_mo(&mut mo_target[spin], &s22_full, &device);
    }
    // mo_target[0].formated_output(5, "full");

    mo_target
}

pub fn check_proj_sanity(mol_target: &Molecule, mol_source: &Molecule) -> bool {
    let mut san = true;
    if mol_target.cint_type != mol_source.cint_type {
        println!("CintType mismatch between target and source molecule (target: {:?}, source: {:?})", mol_target.cint_type, mol_source.cint_type);
        san = false;
    }
    if mol_target.ecp_electrons != mol_source.ecp_electrons {
        println!("ecp_electrons mismatch between target and source molecule (target: {}, source: {})", mol_target.ecp_electrons, mol_source.ecp_electrons);
        san = false;
    }
    // if mol_target.has_ecp() != mol_source.has_ecp() {
    //     println!("ECP presence mismatch between target and source molecule (target has ECP: {}, source has ECP: {})", mol_target.has_ecp(), mol_source.has_ecp());
    //     san = false;
    // }
    san
}