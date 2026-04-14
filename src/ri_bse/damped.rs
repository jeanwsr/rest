use crate::scf_io::{SCF, SCFType};
use crate::mpi_io::MPIOperator;
use std::path::Path;
use rest_libcint::prelude::int1e_r;
use tensors::{MathMatrix, MatrixFull, RIFull,MatrixFullSlice};
use crate::constants::{ANG, AU2DEBYE, SPECIES_INFO};
use itertools::Itertools;
use crate::ri_gw::get_occupation_parameters;
use std::f64::consts::SQRT_2;
use rest_tensors::matrix::matrix_blas_lapack::{_dgeev, _dgemv,_dgemm_full,_dsolve};
use crate::ri_bse;
use crate::ri_bse::davidson_solver::{dot_product,num_product,vector_scaled_add};
use crate::ctrl_io::quasiparticle_methods::QuasiParticle;
use rayon::prelude::*;
use std::time::Instant;
use std::sync::atomic::{AtomicPtr, Ordering};


pub fn obtain_mu_ia_z(eigenvectors:&MatrixFull<f64>,ao_dip:&RIFull<f64>,i:usize,a:usize)->f64{
    let num_state=eigenvectors.size[0];
    let ao_dip_tmp = ao_dip.get_reducing_matrix(2).unwrap();
    let mu_ia:f64=(0..num_state).cartesian_product(0..num_state).fold(0.0,|acc,(m,n)|{
        let idx = m * ao_dip_tmp.size[0] + n;
        acc+eigenvectors[[m,i]]*eigenvectors[[n,a]]*ao_dip_tmp.data[idx]});
    mu_ia
}
pub fn compute_mu_z_vec(scf_data:&SCF)->Vec<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let ao_dip=ri_bse::dipoles::obtain_ao_dips(scf_data,None);
    let mut mu_z_vec=vec![0.0;occ_size*vir_size];
    let eigenvectors=scf_data.eigenvectors[0].clone();
    //eigenvectors.formated_output(1000,"full");
    (0..occ_size).cartesian_product(0..vir_size).for_each(|(i,a)|{
        let mu_ia=obtain_mu_ia_z(&eigenvectors,&ao_dip,i,a+occ_size);
        mu_z_vec[i+a*occ_size]=mu_ia;
    });
    //dipole_matrix.formated_output(1000,"full");
    mu_z_vec
}
pub fn prepare_p0_r_i(mu_z_vec:&Vec<f64>,gwqp:&Vec<f64>,omega:f64,gamma:f64,occ_size:usize,vir_size:usize)->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>){
    let mut p0rp=vec![0.0;occ_size*vir_size];
    let mut p0rm=vec![0.0;occ_size*vir_size];
    let mut p0ip=vec![0.0;occ_size*vir_size];
    let mut p0im=vec![0.0;occ_size*vir_size];
    (0..occ_size).cartesian_product(0..vir_size).for_each(|(i,a)|{
        let mu_ia_z=mu_z_vec[i+a*occ_size];
        let energy_gap=gwqp[a+occ_size]-gwqp[i];
        p0rp[i+a*occ_size]=(omega-energy_gap)/((omega-energy_gap).powf(2.0)+gamma.powf(2.0))*mu_ia_z;
        p0rm[i+a*occ_size]=-(omega+energy_gap)/((omega+energy_gap).powf(2.0)+gamma.powf(2.0))*mu_ia_z;
        p0ip[i+a*occ_size]=gamma/((omega-energy_gap).powf(2.0)+gamma.powf(2.0))*mu_ia_z;
        p0im[i+a*occ_size]=-gamma/((omega+energy_gap).powf(2.0)+gamma.powf(2.0))*mu_ia_z;
    });
    (p0rp,p0rm,p0ip,p0im)
}
pub fn a_block_matvec(scf_data:&SCF,ri_vv:&MatrixFull<f64>,ri_ov:&MatrixFull<f64>,ri_oo_tilde:&MatrixFull<f64>,z_vec:&Vec<f64>,qp_ctrl:&QuasiParticle)->Vec<f64>{
    let mut result=ri_bse::matvec::w_contribution_a_block_dgemm(scf_data,ri_vv,z_vec,ri_oo_tilde,&qp_ctrl);
    result=ri_bse::matvec::coulomb_contribution(ri_ov,z_vec).iter().zip(result.iter()).map(|(v_i,z_i)|v_i+z_i).collect();
    result
}
pub fn b_block_matvec(scf_data:&SCF,ri_ov_a:&MatrixFull<f64>,ri_ov_b:&MatrixFull<f64>,ri_ov_tilde:&MatrixFull<f64>,z_vec:&Vec<f64>)->Vec<f64>{
    let mut result=ri_bse::matvec::w_contribution_b_block_dgemm(scf_data,ri_ov_b,z_vec,ri_ov_tilde);
    result=ri_bse::matvec::coulomb_contribution(ri_ov_a,z_vec).iter().zip(result.iter()).map(|(v_i,z_i)|v_i+z_i).collect();
    result
}
pub fn pairvec_matvec(scf_data:&SCF,ri_vv:&MatrixFull<f64>,ri_ov:&MatrixFull<f64>,ri_oo_tilde:&MatrixFull<f64>,
    ri_ov_w:&MatrixFull<f64>,ri_ov_tilde:&MatrixFull<f64>,p_vec:&Vec<f64>,m_vec:&Vec<f64>,
    qp_ctrl:&QuasiParticle)->(Vec<f64>,Vec<f64>){
    let ap=a_block_matvec(scf_data,ri_vv,ri_ov,ri_oo_tilde,p_vec,qp_ctrl);
    let am=a_block_matvec(scf_data,ri_vv,ri_ov,ri_oo_tilde,m_vec,qp_ctrl);
    let bp=b_block_matvec(scf_data,ri_ov,ri_ov_w,ri_ov_tilde,p_vec);
    let bm=b_block_matvec(scf_data,ri_ov,ri_ov_w,ri_ov_tilde,m_vec);
    (ap.iter().enumerate().map(|(k,apk)|apk+bm[k]).collect(),bp.iter().enumerate().map(|(k,bpk)|bpk+am[k]).collect())
}
pub fn update_w_vecs<F1>(rp:&Vec<f64>,rm:&Vec<f64>,ip:&Vec<f64>,im:&Vec<f64>,pair_matvec:F1,
    gwqp:&Vec<f64>,omega:f64,gamma:f64,occ_size:usize,vir_size:usize)->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)
    where F1:Fn(&(Vec<f64>,Vec<f64>))->(Vec<f64>,Vec<f64>){
    let preal=(rp.clone(),rm.clone());
    let pimag=(ip.clone(),im.clone());
    let (krp,krm)=pair_matvec(&preal);
    let (kip,kim)=pair_matvec(&pimag);
    let mut wrp=vec![0.0;occ_size*vir_size];
    let mut wrm=vec![0.0;occ_size*vir_size];
    let mut wip=vec![0.0;occ_size*vir_size];
    let mut wim=vec![0.0;occ_size*vir_size];
    (0..occ_size).cartesian_product(0..vir_size).for_each(|(i,a)|{
        let energy_gap=gwqp[a+occ_size]-gwqp[i];
        let prefactor_p:f64=1.0/((omega-energy_gap).powf(2.0)+gamma.powf(2.0));
        let prefactor_m:f64=-1.0/((omega+energy_gap).powf(2.0)+gamma.powf(2.0));
        wrp[i+a*occ_size]=prefactor_p*((omega-energy_gap)*krp[i+a*occ_size]+gamma*kip[i+a*occ_size]);
        wrm[i+a*occ_size]=prefactor_m*((omega+energy_gap)*krm[i+a*occ_size]+gamma*kim[i+a*occ_size]);
        wip[i+a*occ_size]=prefactor_p*((omega-energy_gap)*kip[i+a*occ_size]+gamma*krp[i+a*occ_size]);
        wim[i+a*occ_size]=prefactor_m*((omega+energy_gap)*kim[i+a*occ_size]+gamma*krm[i+a*occ_size]);
    });
    (wrp,wrm,wip,wim)
} 
pub fn fourvec_dot_product(fourvec_1:&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>),fourvec_2:&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>))->f64{
    dot_product(&fourvec_1.0,&fourvec_2.0)+dot_product(&fourvec_1.1,&fourvec_2.1)+dot_product(&fourvec_1.2,&fourvec_2.2)+dot_product(&fourvec_1.3,&fourvec_2.3)
}
pub fn fourvec_scaled_add(fourvec_1:&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>),scale1:f64,fourvec_2:&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>),scale2:f64)->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>){
    (vector_scaled_add(&fourvec_1.0,scale1,&fourvec_2.0,scale2),vector_scaled_add(&fourvec_1.1,scale1,&fourvec_2.1,scale2),vector_scaled_add(&fourvec_1.2,scale1,&fourvec_2.2,scale2),vector_scaled_add(&fourvec_1.3,scale1,&fourvec_2.3,scale2))
}
pub fn poples_numerical_trick<F1>(p0:&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>),update_w:F1)->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)
    where F1:Fn(&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>))->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>){
    let p0p0sqrt=fourvec_dot_product(p0,p0).sqrt();
    let u0=(num_product(&p0.0,p0p0sqrt),num_product(&p0.1,p0p0sqrt),num_product(&p0.2,p0p0sqrt),num_product(&p0.3,p0p0sqrt));
    let occvir=p0.0.len();
    let mut p_result=(vec![0.0;occvir],vec![0.0;occvir],vec![0.0;occvir],vec![0.0;occvir]);
    let mut u_vecs:Vec<(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)>=vec![u0.clone()];
    let mut w_vecs:Vec<(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)>=Vec::new();
    let mut iter_count=0;
    loop{
        let un=u_vecs[u_vecs.len()-1].clone();
        let wn=update_w(&un);
        w_vecs.push(wn.clone());
        let mut unp1=wn.clone();
        u_vecs.iter().for_each(|uk|{
            let ukuk=fourvec_dot_product(uk,uk);
            let ukwn=fourvec_dot_product(uk,&wn);
            unp1=fourvec_scaled_add(&unp1,1.0,uk,-ukwn/ukuk);
        });
        let coeff=solve_for_coeff(&u_vecs,&w_vecs,&p0);
        let residue=calculate_residue(&u_vecs,&w_vecs,&p0,&coeff);
        u_vecs.push(unp1);
        if residue<0.000001{
            (0..coeff.len()).for_each(|k|{
               p_result=fourvec_scaled_add(&p_result,1.0,&u_vecs[k],coeff[k]); 
            });
            break
        }
        iter_count+=1;
        println!("Now is iteration {}, residue={},converge threshold is 0.000001",iter_count,residue);
    }
    p_result
}
pub fn solve_for_coeff(u_vecs:&Vec<(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)>,w_vecs:&Vec<(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)>,p0:&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>))->Vec<f64>{
    let n=u_vecs.len();
    let mut matrix=MatrixFull::new([n,n],0.0);
    (0..n).cartesian_product(0..n).for_each(|(i,k)|{
        let wkui=fourvec_dot_product(&u_vecs[i],&w_vecs[k]);
        matrix[[i,k]]=if i==k{wkui+1.0}else {wkui};
    });
    let mut b=vec![0.0;n];
    (0..n).for_each(|i|{
        b[i]=fourvec_dot_product(&u_vecs[i],p0);
    });
    _dsolve(&matrix,&b).unwrap()
}
pub fn calculate_residue(u_vecs:&Vec<(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)>,w_vecs:&Vec<(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)>,
    p0:&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>),coeff:&Vec<f64>)->f64{
    let mut residue=p0.clone();
    (0..u_vecs.len()).for_each(|k|{
        residue=fourvec_scaled_add(&u_vecs[k],coeff[k],&residue,1.0);
        residue=fourvec_scaled_add(&w_vecs[k],coeff[k],&residue,1.0);
    });
    fourvec_dot_product(&residue,&residue).sqrt()
}
pub fn damped_bse(scf_data:&SCF)->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>){
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let inverse_dielectric=ri_bse::construct_inverse_dielectric(scf_data,&scf_data.eigenvalues[0]);
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    println!("Now begins damped BSE calculation. Parameters:\nOcc Size={},Vir Size={},External Field Freq={},Lifetime Gamma={}",occ_size,vir_size,qp_ctrl.external_field_freq,qp_ctrl.lifetime_gamma);
    let mu_z_vec=compute_mu_z_vec(scf_data);
    let ri_oo=ri_bse::get_submatrix(scf_data,'O','O','N');
    println!("num_auxbas={}",ri_oo.size[0]);
    let num_auxbas=ri_oo.size[0];
    let mut ri_oo_tilde:MatrixFull<f64>=MatrixFull::new(ri_oo.size,0.0);
    _dgemm_full(&inverse_dielectric,'N',&ri_oo,'N',&mut ri_oo_tilde,1.0,0.0);
    drop(ri_oo);
    ri_oo_tilde.reshape([num_auxbas*occ_size,occ_size]);
    ri_oo_tilde=ri_oo_tilde.transpose_and_drop();
    ri_oo_tilde.reshape([occ_size*num_auxbas,occ_size]);
    let ri_ov=ri_bse::get_submatrix(scf_data,'O','V','N');
    let mut ri_ov_w=ri_ov.clone();
    ri_ov_w.reshape([num_auxbas*occ_size,vir_size]);
    let mut ri_vv=ri_bse::get_submatrix(scf_data,'V','V','N');
    ri_vv.reshape([num_auxbas*vir_size,vir_size]);
    let mut ri_ov_tilde:MatrixFull<f64>=MatrixFull::new(ri_ov.size,0.0);
    _dgemm_full(&inverse_dielectric,'N',&ri_ov,'N',&mut ri_ov_tilde,1.0,0.0);
    drop(inverse_dielectric);
    ri_ov_tilde.reshape([num_auxbas*occ_size,vir_size]);
    let p0=prepare_p0_r_i(&mu_z_vec,&scf_data.gwqp.0,qp_ctrl.external_field_freq,qp_ctrl.lifetime_gamma,occ_size,vir_size);
    let wrapped_pair_matvec = |pairvec:&(Vec<f64>,Vec<f64>)| -> (Vec<f64>,Vec<f64>) {
        pairvec_matvec(scf_data,&ri_vv,&ri_ov,&ri_oo_tilde,
            &ri_ov_w,&ri_ov_tilde,&pairvec.0,&pairvec.1,
            &qp_ctrl)
    };
    let wrapped_update_w=|u:&(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>)|->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>){
        update_w_vecs(&u.0,&u.1,&u.2,&u.3,|x|wrapped_pair_matvec(&x),
            &scf_data.gwqp.0,qp_ctrl.external_field_freq,qp_ctrl.lifetime_gamma,occ_size,vir_size)
    };
    poples_numerical_trick(&p0,|z|wrapped_update_w(&z))
}