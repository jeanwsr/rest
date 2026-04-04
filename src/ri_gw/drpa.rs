use rest_tensors::matrix::{MatrixFull,MathMatrix,matrixupper::MatrixUpper};
use rest_tensors::matrix::matrix_blas_lapack::{_dgeev,_dgemm_full, _dgemv,_dpotrf,_dsyev,_dinverse, _power_rayon_for_symmetric_matrix};
use std::time::Instant;
use crate::ri_bse::matvec;
use crate::scf_io::{SCF, SCFType};
use std::cmp;
use crate::ri_bse;
use crate::ri_gw;
use rayon::prelude::*;
use rayon::iter::ParallelBridge;
use itertools::Itertools;
use crate::ctrl_io::quasiparticle_methods::QuasiParticle;
use crate::constants::{INVERSE_THRESHOLD, SPECIES_INFO, SQRT_THRESHOLD};

pub fn construct_submat_a(scf_data:&SCF,quasiparticle_energies:&Vec<f64>)->MatrixFull<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=ri_gw::get_occupation_parameters(scf_data,'N');
    let ri_ov=ri_bse::get_submatrix(scf_data,'O','V','N');
    let mut v=ri_bse::construct_coulomb(&ri_ov,&ri_ov);
    v.self_multiple(2.0);
    let mut energy_diag=ri_bse::construct_energy_diag_for_a(quasiparticle_energies,occ_size,vir_size);
    v.iter_diagonal_mut().unwrap().zip(energy_diag.iter_mut()).for_each(|(x,e)|{(*x,*e)=(*x+*e,*e)});
    v
}
pub fn construct_submat_b(scf_data:&SCF)->MatrixFull<f64>{
    let ri_ov=ri_bse::get_submatrix(scf_data,'O','V','N');
    let mut v=ri_bse::construct_coulomb(&ri_ov,&ri_ov);
    v.self_multiple(2.0);
    v
}
