use crate::scf_io::{SCF, SCFType};
use crate::mpi_io::MPIOperator;
use std::path::Path;
use rest_libcint::prelude::rest_libcint_wrapper::int1e_r;
use tensors::{MathMatrix, MatrixFull, RIFull,MatrixFullSlice};
use crate::constants::{AU2DEBYE, SPECIES_INFO};
use itertools::Itertools;
use crate::ri_gw::get_occupation_parameters;
use rest_tensors::matrix::matrix_blas_lapack::{_dgemv};
use std::f64::consts::SQRT_2;


pub fn obtain_ao_dips(scf_data:&SCF,orig:Option<[f64;3]>)->RIFull<f64>{
    let (mut tot_dip, mass_tot) = scf_data.mol.geom.evaluate_dipole_moment(None);

    let mut dm = scf_data.density_matrix[0].clone();
    if scf_data.mol.spin_channel == 2 {
        dm.self_add(&scf_data.density_matrix[1]);
    }
    let mut cint_data = scf_data.mol.initialize_cint(false);

    let p_orig = cint_data.get_common_origin();

    let r_orig: [f64;3] = if let Some(u_orig) = orig {
        u_orig.try_into().unwrap()
    } else {
        p_orig.clone()
    };
    cint_data.set_common_origin(r_orig);
    let (out, out_shape)= cint_data.integral_s1::<int1e_r>(None);
    RIFull::from_vec(out_shape.try_into().unwrap(), out).unwrap()
}
pub fn obtain_mu_ia(eigenvectors:&MatrixFull<f64>,ao_dip:&RIFull<f64>,i:usize,a:usize)->[f64;3]{
    let num_state=eigenvectors.size[0];
    let mut mu_ia=[0.0;3];
    for d in 0..3{
        let ao_dip_tmp = ao_dip.get_reducing_matrix(d).unwrap();
        mu_ia[d]=(0..num_state).cartesian_product(0..num_state).fold(0.0,|acc,(m,n)|{
            let idx = m * ao_dip_tmp.size[0] + n;
            acc+eigenvectors[[m,i]]*eigenvectors[[n,a]]*ao_dip_tmp.data[idx]})
    }
    mu_ia
}
pub fn compute_dipole_matrix(scf_data:&SCF)->MatrixFull<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    if scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap().bse_spin=="triplet"{
        return MatrixFull::new([3,vir_size*occ_size],0.0)
    }
    let ao_dip=obtain_ao_dips(scf_data,None);
    //matrixfullslice_to_matrixfull(ao_dip.get_reducing_matrix(0).unwrap()).formated_output(1000,"full");
    //matrixfullslice_to_matrixfull(ao_dip.get_reducing_matrix(1).unwrap()).formated_output(1000,"full");
    //matrixfullslice_to_matrixfull(ao_dip.get_reducing_matrix(2).unwrap()).formated_output(1000,"full");
    let mut dipole_matrix=MatrixFull::new([3,vir_size*occ_size],0.0);
    let eigenvectors=scf_data.eigenvectors[0].clone();
    //eigenvectors.formated_output(1000,"full");
    (0..occ_size).cartesian_product(0..vir_size).for_each(|(i,a)|{
        // compact occupied `i` -> orbital `start_mo + i`; compact virtual `a`
        // -> orbital `lumo + a` (the virtual block spans `lumo..num_state`)
        let mu_ia=obtain_mu_ia(&eigenvectors,&ao_dip,start_mo+i,lumo+a);
        dipole_matrix[[0,i+a*occ_size]]=mu_ia[0];
        dipole_matrix[[1,i+a*occ_size]]=mu_ia[1];
        dipole_matrix[[2,i+a*occ_size]]=mu_ia[2];
    });
    //dipole_matrix.formated_output(1000,"full");
    dipole_matrix
}
/// Transition-dipole matrix for a CVS transition space.
///
/// Columns are laid out exactly as the CVS amplitudes, `i_local + a*ncore`,
/// where `i_local` indexes `core` and `a` indexes the virtual window
/// `vir_start..vir_start+vir_size`.  Triplet CVS matrices are returned as
/// zeros, matching the ordinary BSE convention (the transition is dark in the
/// non-relativistic dipole approximation).
pub fn compute_dipole_matrix_cvs(
    scf_data:&SCF,
    core:&[usize],
    vir_start:usize,
    vir_size:usize,
    triplet:bool,
)->MatrixFull<f64>{
    let ncore=core.len();
    if triplet{
        return MatrixFull::new([3,vir_size*ncore],0.0)
    }
    let ao_dip=obtain_ao_dips(scf_data,None);
    let eigenvectors=scf_data.eigenvectors[0].clone();
    let mut dipole_matrix=MatrixFull::new([3,vir_size*ncore],0.0);
    for (ii,&i) in core.iter().enumerate(){
        for a in 0..vir_size{
            let mu_ia=obtain_mu_ia(&eigenvectors,&ao_dip,i,vir_start+a);
            let idx=ii+a*ncore;
            dipole_matrix[[0,idx]]=mu_ia[0];
            dipole_matrix[[1,idx]]=mu_ia[1];
            dipole_matrix[[2,idx]]=mu_ia[2];
        }
    }
    dipole_matrix
}
pub fn transition_dipole_square(dipole_matrix:&MatrixFull<f64>,vec:&Vec<f64>,tda:bool)->f64{
    let mut mu=vec![0.0;3];
    let mut vector:Vec<f64>=Vec::new();
    if tda==false{
        let size=dipole_matrix.size[1];
        let x=&vec[0..size];
        let y=&vec[size..];
        vector=x.iter().zip(y.iter()).map(|(x_i,y_i)|x_i+y_i).collect();
    }else{
        vector=vec.to_vec();
    }
    vector=vector.iter().map(|x_i|x_i*2.0).collect();
    _dgemv(dipole_matrix, &vector, &mut mu, 'N', 1.0, 0.0, 1, 1);
    println!("\tDipole Moment Components:\n\tx:{}, y:{}, z:{}",mu[0],mu[1],mu[2]);
    mu[0].powf(2.0)+mu[1].powf(2.0)+mu[2].powf(2.0)
}
pub fn matrixfullslice_to_matrixfull(slice:MatrixFullSlice<f64>)->MatrixFull<f64>{
    MatrixFull{data:slice.data.to_vec(),indicing:[slice.indicing[0],slice.indicing[1]],size:[slice.size[0],slice.size[1]]}
}
pub fn normalize(vector:&[f64],tda:bool)->Vec<f64>{
    vector.to_vec();
    if tda==false{
        let num_state=vector.len()/2;
        let x_vec=&vector[0..num_state];
        let y_vec=&vector[num_state..];
        let x_norm=x_vec.iter().fold(0.0,|acc,x_i|acc+x_i.powf(2.0));
        let y_norm=y_vec.iter().fold(0.0,|acc,y_i|acc+y_i.powf(2.0));
        println!("x_norm-y_norm={}",x_norm-y_norm);
        let x_minus_y_root=(x_norm-y_norm).powf(0.5);
        vector.iter().map(|x_i|x_i/x_minus_y_root/SQRT_2).collect()
    }else{
        let x_norm=vector.iter().fold(0.0,|acc,x_i|acc+x_i.powf(2.0));
        vector.iter().map(|x_i|x_i/x_norm/SQRT_2).collect()
    }
}