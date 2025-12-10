//use std::simd::num;
use crate::constants::PI;
use itertools::Itertools;
use std::ops::Range;
use crate::utilities;
use crate::scf_io::SCF;
//use std::slice::Iter::<'_, f64>;
use std::fs::OpenOptions;
use std::io::Write;
use rayon::iter::ParallelBridge;
use libc::seccomp_notif;
use rayon::result;
use reqwest::blocking::Response;
use std::sync::{Arc, Mutex};
use rest_tensors::{RIFull};
use tensors::{matrix_blas_lapack::{_dinverse,_dsyev}, ri, MathMatrix, MatrixFull};
//use rest::molecule_io::Molecule;
use rest_tensors::matrix::matrix_blas_lapack::{_dgees,_dgemm_full,_dgemv};
use crate::tensors::matrix_blas_lapack::omp_get_num_threads_wrapper;
use rest_tensors::MatrixUpper;
use crate::ri_bse;
use crate::ri_rpa;
use crate::molecule_io;
use std::{fs, io};
use crate::scf_io;
use crate::dft;
use crate::dft::DFA4REST;
use std::sync::mpsc::channel;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
pub mod renormalized_singles;
pub mod scgw;
pub mod display;
use crate::mpi_io::MPIOperator;

pub fn gw_main(scf_data:&mut SCF,vxc_nn:&Vec<f64>,mpi_operator:&Option<MPIOperator>){
    let printlevel=scf_data.mol.ctrl.print_level.clone();
    show_energy_levels(scf_data);
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(&scf_data,'Y');
    let mut rs_particles:Vec<f64>=Vec::new();
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let renormalized_singles=qp_ctrl.renormalized_singles;
    if renormalized_singles==true{
        println!("Starts renormalized singles calculations!");
        let w_rs=qp_ctrl.w_rs;
        //println!("Now starting to compute rs_particles");
        rs_particles=renormalized_singles::renormalized_singles_diagonalization(scf_data,w_rs,mpi_operator);
        if printlevel>0{
            println!("Renormalized Singles particles:{:#?}",rs_particles);
        }
        scf_data.renormalized_singles_particles=rs_particles
    }
    initialize_qp_g_w(scf_data);
    let gw_scheme=qp_ctrl.gw_scheme.clone();
    let scgw=qp_ctrl.scgw.clone();
    let quasiparticle_energies:Vec<f64>=
        if scgw=="g0w0"&&renormalized_singles==false{
            println!("You are doing G0W0 calculations of entire energy spectrum");
            scgw::g0w0(scf_data,20,&vxc_nn,true)
        }else if scgw=="evgw"{
            println!("You are doing evGW calculations of entire energy spectrum");
            scgw::evgw(scf_data,20,&vxc_nn,qp_ctrl.evgw_rounds)
        }else if scgw=="g0w0"&&renormalized_singles==true{
            scgw::g0w0(scf_data,20,&vxc_nn,true)
        }else{panic!("invalid expression for scgw!")};
    if gw_scheme !="no gw"{
        println!("One round of GW by {} scheme has finished.",gw_scheme);
    }
    if qp_ctrl.save_qp_path.len()>0{
        let save_path=qp_ctrl.save_qp_path.clone();
        let mut file = OpenOptions::new().append(true).create(true).open(save_path);
        scf_data.gwqp.0.iter().for_each(|qp|{
            writeln!(file.as_ref().expect("write failure"), "{}",qp);
        });
    }
}
pub fn initialize_qp_g_w(scf_data:&mut SCF){
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    if qp_ctrl.renormalized_singles==true{
        let rsp=scf_data.renormalized_singles_particles.clone();
        if qp_ctrl.w_rs==true{
            scf_data.gwqp=(rsp.clone(),rsp);
        }else{
            let eigenenergies=scf_data.eigenvalues[0].clone();
            scf_data.gwqp=(rsp,eigenenergies);
        }
    }else{
        let eigenenergies=scf_data.eigenvalues[0].clone();
        scf_data.gwqp=(eigenenergies.clone(),eigenenergies);
    }
}
pub fn show_energy_levels(scf_data:&SCF){
    let num_state = scf_data.mol.num_state;
    let mut homo = 0_usize;
    let mut lumo = num_state;
    let start_mo = scf_data.mol.start_mo;
    for i_spin in 0..scf_data.mol.spin_channel {

        let i_homo = scf_data.homo.get(i_spin).unwrap().clone();
        let i_lumo = scf_data.lumo.get(i_spin).unwrap().clone();

        homo = homo.max(i_homo);
        lumo = lumo.min(i_lumo);
    }
    let occ_size=homo-start_mo+1;
    let vir_size=num_state-lumo;
    let eigenvalues=scf_data.eigenvalues[0].clone();
    for i_occ in 0..occ_size{
        if scf_data.mol.ctrl.print_level>1{
            println!("{}th occupied orbital, energy={}",i_occ+1,eigenvalues[start_mo+i_occ])   
        }
    }
    for i_vir in 0..vir_size{
        if scf_data.mol.ctrl.print_level>1{
            println!("{}th virtual orbital, energy={}",i_vir+1,eigenvalues[lumo+i_vir])
        }
    }
    let eigenvalues=scf_data.eigenvalues[0].clone();
    if scf_data.mol.ctrl.print_level>1{
        println!("homo:{},lumo:{},occ_size:{},vir_size:{},start_mo:{},num_state:{}",homo,lumo,occ_size,vir_size,start_mo,num_state);
    }
    
}

pub fn gw_calculations(scf_data:&mut SCF,num_freq:usize,vxc_nn:&Vec<f64>,cancel_dfa_xc:bool)->Vec<f64>{
    let v_matrix=v_matrix(&scf_data);
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'Y');
    let (start_mo,num_state_cutoff,occ_size,vir_size_cutoff,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let hatree=27.2113863;
    let spin_channel=scf_data.mol.ctrl.spin_channel;
    let mut quasiparticle_energies_g:Vec<f64>=Vec::new();
    let mut quasiparticle_energies_w:Vec<f64>=Vec::new();
    let ri_mat:MatrixFull<f64>=ri_bse::get_submatrix(scf_data,'F','F','Y');
    let start:usize=0;
    let end:usize=6*occ_size;
    if scf_data.mol.ctrl.print_level>2{
        println!("ri_mat full:");
        ri_mat.formated_output(1000,"full");
    }
    let mut quasiparticle_energies_g=scf_data.gwqp.0.clone();
    let quasiparticle_energies_w=scf_data.gwqp.1.clone();
    let ri_ov:MatrixFull<f64>=ri_bse::get_submatrix(scf_data,'O','V','Y');
    //display_and_save_quasiparticles(scf_data,&quasiparticle_energies_g,0);
    let w_c_at_freqs=generate_w_c(scf_data,&ri_ov,&ri_mat,&quasiparticle_energies_g,&quasiparticle_energies_w,num_state,occ_size,vir_size,num_freq);
    let mut save_energies:Vec<f64>=vec![0.0;num_state_cutoff];
    save_energies=(0..num_state_cutoff).map(|n|{
        let consts=if cancel_dfa_xc==true{
            let mut exchange=0.0;
            for i in 0..homo+1{
                exchange-=v_matrix[[n,i]];
            }
            scf_data.eigenvalues[0][n]+exchange-vxc_nn[n]
        }else{quasiparticle_energies_g[n]};
        //for first round, if non-RS, then of course qp==scf;
        //if RS, testings showed that solving omega=scf-v_xc+sigma_x+sigma_c_RS would be better for fisrt round
        //To confirm, the consts DO UPDATE over self-consistent GW
        let side=if n>=occ_size{1.0}else{-1.0};
        let printlevel=scf_data.mol.ctrl.print_level.clone();
        let mut real_qp=0.0;
        let mut have_crossing=true;
        let qp_eq_func=|omega: f64|{
            quasiparticle_equation(omega,n,consts,&ri_ov,&ri_mat,&quasiparticle_energies_g,&quasiparticle_energies_w,occ_size,vir_size,num_state,&w_c_at_freqs)
        };
        (have_crossing,real_qp)=linear_interpolation_solver(qp_eq_func,scf_data.eigenvalues[0][n],side,21,0.1);
        //real_qp=newton_solver(quasiparticle_equation,n,consts,&ri_ov,&ri_mat,&quasiparticle_energies_g,&quasiparticle_energies_w,occ_size,vir_size,num_state,&w_c_at_freqs,scf_data.eigenvalues[0][n],0.00001,50,side,printlevel);
        println!("for n={}, quasiparticle equation yields:qp energy={}",n,real_qp);
        real_qp
    }).collect::<Vec<f64>>().clone();
    quasiparticle_energies_g=save_energies.clone();
    scf_data.gwqp.0=quasiparticle_energies_g.clone();
    display::full_quasiparticles(&quasiparticle_energies_g,occ_size);
    quasiparticle_energies_g
} 


pub fn vxc_ao2mo(scf_data:&SCF)->Vec<f64>{
    let eigenvecs=scf_data.eigenvectors.clone();
    let vxc_ao=scf_data.generate_vxc_rayon(1.0).2[0].to_matrixfull().unwrap().clone();
    let dimensions=vxc_ao.size[0];
    let mut vxc_nn=vec![0.0;dimensions];
    let mut element=0.0;
    for i in (0..dimensions){
        //println!("now computing:{}-{}",i,i);
        element=0.0;
        for mu in 0..dimensions{
            for nu in 0..dimensions{
                element+=(eigenvecs[0][[mu,i]]*eigenvecs[0][[nu,i]]*vxc_ao[[mu,nu]]);
            }
        }
        vxc_nn[i]=element;
    }
    vxc_nn
    //No need to worry about cutoff
}
pub fn vxc_ao2mo_rayon(scf_data:&SCF)->MatrixFull<f64>{
    let eigenvecs=scf_data.eigenvectors.clone();
    let vxc_ao=scf_data.generate_vxc_rayon(1.0).2[0].to_matrixfull().unwrap().clone();
    let dimensions=vxc_ao.size[0];
    let mut vxc_mo:MatrixFull<f64>=MatrixFull::new([dimensions,dimensions],0.0);
    vxc_mo.data.par_iter_mut().enumerate().for_each(|(index, element)| {
        let i = index / dimensions;
        let j = index % dimensions;
        println!("now computing:{}-{}",i,j);
        // 计算element的值并赋值
        for mu in 0..dimensions{
            for nu in 0..dimensions{
                *element+=(eigenvecs[0][[mu,i]]*eigenvecs[0][[nu,j]]*vxc_ao[[mu,nu]]);
            }
        }
    });
    vxc_mo
    //Cutoff will help, but not necessary
}
pub fn v_mn_matrix_element(scf_data:&SCF,m:usize,n:usize,ri_mat:&MatrixFull<f64>)->f64{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'Y');
    let mut v_mn_matrix_element=0.0;
    for i in 0..(ri_mat.size[0]){
        v_mn_matrix_element+=ri_mat[[i as usize,m*num_state+n]].powf(2.0);
    }
    v_mn_matrix_element
}
pub fn v_matrix(scf_data:&SCF)->MatrixFull<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'Y');
    let ri_mat:MatrixFull<f64>=ri_bse::get_submatrix(scf_data,'F','F','Y');
    let mut v_matrix:MatrixFull<f64>=MatrixFull::new([num_state,num_state],0.0);
    for i in 0..num_state{
        for j in 0..num_state{
            v_matrix[[i,j]]=v_mn_matrix_element(scf_data,i,j,&ri_mat);
        }
    }
    v_matrix
    //No need to worry about cutoff
}
pub fn generate_w_c(scf_data:&SCF,ri_ov:&MatrixFull<f64>,ri_full:&MatrixFull<f64>,quasiparticle_energies_g:&Vec<f64>,quasiparticle_energies_w:&Vec<f64>,num_state:usize,occ_size:usize,vir_size:usize,num_freq:usize)->Vec<(f64,f64,MatrixFull<f64>)>{
    omp_get_num_threads_wrapper();
    let freq_grid_type = scf_data.mol.ctrl.freq_grid_type;
    let max_freq = scf_data.mol.ctrl.freq_cut_off;
    let mut sp = format!("The frequency integration is tabulated by {:3} grids using", num_freq);
    let (mut omega_1,weight) = if freq_grid_type==0 {
        sp = format!("{} the modified Gauss-Legendre grids",sp);
        ri_rpa::trans_gauss_legendre_grids(1.0, num_freq)
    } else if freq_grid_type==1 {
        sp = format!("{} the standard Gauss-Legendre grids",sp);
        ri_rpa::gauss_legendre_grids([0.0,max_freq], num_freq)
    } else if freq_grid_type== 2 {
        sp = format!("{} the logarithmic grids",sp);
        ri_rpa::logarithmic_grid([0.0,max_freq], num_freq)
    } else {
        sp = format!("{} the modified Gauss-Legendre grids",sp);
        ri_rpa::trans_gauss_legendre_grids(1.0, num_freq)
    };
    if scf_data.mol.ctrl.print_level>1 {
        println!("{}", sp);
    }
    let (sender,receiver) = channel();
    rayon::prelude::IndexedParallelIterator::zip(omega_1.par_iter(), weight.par_iter()).for_each_with(sender, |s, (omega_1, weight)| {
        let response=response_matrix(quasiparticle_energies_w,occ_size,vir_size,ri_ov,*omega_1,'I');
        let inverse_dielectric=inverse_dielectric_matrix(&response,'I');
        let w_c=w_c_matrix(&inverse_dielectric,num_state,ri_full);
        s.send((*omega_1,*weight,w_c)).expect("unsuccessful collection of w_c")
    });
    let w_c_at_freqs:Vec<(f64,f64,MatrixFull<f64>)>=receiver.into_iter().collect();
    omp_get_num_threads_wrapper();
    w_c_at_freqs
}
pub fn calculate_imag(w_c_at_freqs:&Vec<(f64,f64,MatrixFull<f64>)>,num_state:usize,n:usize,omega:f64,quasiparticle_energies_g:&Vec<f64>,quasiparticle_energies_w:&Vec<f64>)->f64{
    omp_get_num_threads_wrapper();
    let (sender,receiver)=channel();
    //println!("0th freq:  calculating imag: w_c size:{},{}; quasiparticles_g size:{}",w_c_at_freqs[0].2.size[0],w_c_at_freqs[0].2.size[1],quasiparticle_energies_g.len());
    //println!("10th freq: calculating imag: w_c size:{},{}; quasiparticles_g size:{}",w_c_at_freqs[10].2.size[0],w_c_at_freqs[10].2.size[1],quasiparticle_energies_g.len());
    w_c_at_freqs.par_iter().for_each_with(sender, |s, (omega_1,weight,w_c)|{
        let imag_n_omega=(quasiparticle_energies_g.iter().enumerate().map(|(m,qp_m)|{
            let rpod=omega-qp_m;
            let gfc=rpod/(rpod.powf(2.0)+omega_1.powf(2.0));
            2.0*gfc*(w_c[[n,m]])/(2.0*PI)
        }).sum::<f64>())*weight;
        s.send(imag_n_omega).expect("unsuccessful collection of imag_n_omega")
    });
    let imag_n=receiver.into_iter().sum();
    omp_get_num_threads_wrapper();
    imag_n
    //No need to worry about cutoff
}
pub fn get_occupation_parameters(scf_data:&SCF,response_or_not:char)->(usize,usize,usize,usize,usize,usize){
    let cutoff=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap().bse_cutoff_energy;
    let mut num_state = scf_data.mol.num_state;
    let mut homo = 0_usize;
    let mut lumo = num_state;
    let start_mo = scf_data.mol.start_mo;
    for i_spin in 0..scf_data.mol.spin_channel {
        let i_homo = scf_data.homo.get(i_spin).unwrap().clone();
        let i_lumo = scf_data.lumo.get(i_spin).unwrap().clone();
        homo = homo.max(i_homo);
        lumo = lumo.min(i_lumo);
    }
    let occ_size=homo-start_mo+1;
    let mut vir_size=0;
    if response_or_not=='Y'{
        vir_size=num_state-occ_size;
    }
    
    else if response_or_not=='N'{
        num_state=scf_data.eigenvalues[0].clone().iter().filter(|x|**x<cutoff).count();
        vir_size=num_state-occ_size;
    }
    (start_mo,num_state,occ_size,vir_size,homo,lumo)
}
pub fn display_and_save_quasiparticles(scf_data:&mut SCF,quasiparticle_energies_g:&Vec<f64>,round:usize){
    for i in 0..quasiparticle_energies_g.len(){
        println!("quasiparticles:{:#?}",quasiparticle_energies_g[i]);
        //scf_data.eigenvalues[0][i]=quasiparticle_energies[i];
    }
}
pub fn response_matrix(quasiparticle_energies_w:&Vec<f64>,occ_size:usize,vir_size:usize,ri_ov:&MatrixFull<f64>,omega:f64,part:char)->MatrixFull<f64>{
    let num_auxbas=ri_ov.size[0];
    let mut ri_to_be_processed=MatrixFull::new([num_auxbas,0],0.0);
    let mut ri_as_vecs:Vec<(usize,Vec<f64>)>=ri_ov.iter_columns_full().enumerate().par_bridge().map(|(n,ri_n)|{
        let i=n%occ_size;
        let a=occ_size+n/occ_size;
        let energy_gap=quasiparticle_energies_w[a]-quasiparticle_energies_w[i];
        let mut multiply_number=if part=='I'{-2.0*energy_gap/(energy_gap.powf(2.0)+omega.powf(2.0))}
            else{-2.0*energy_gap/(energy_gap.powf(2.0)-omega.powf(2.0))};
        let mut ri_n_vec=ri_n.to_vec();
        ri_n_vec.par_iter_mut().for_each(|x| *x *= multiply_number);
        (n,ri_n_vec)
    }).collect();
    ri_as_vecs.sort_by_key(|(i, _)| *i);
    for ri_vec in ri_as_vecs {
        ri_to_be_processed.push_column(&ri_vec.1);
    }
    let mut response:MatrixFull<f64>=MatrixFull::new([num_auxbas,num_auxbas],0.0);
    _dgemm_full(ri_ov,'N',&ri_to_be_processed,'T',&mut response,1.0,0.0); 
    response.self_multiple(2.0);
    //println!("a response has been collected, its size is:{},{}",response.size[0],response.size[1]);
    response
}
pub fn inverse_dielectric_matrix(response:&MatrixFull<f64>,part:char)->MatrixFull<f64>{
    let mut dielectric:MatrixFull<f64>=response.clone();
    dielectric.self_multiple(-1.0);
    let num_auxbas=response.size[0];
    dielectric+=ri_bse::identity_matrix(num_auxbas);
    let mut inverse_dielectric=_dinverse(&dielectric).expect("unsuccessful _dinverse");
    if part=='C'||part=='I'{
        inverse_dielectric-=ri_bse::identity_matrix(num_auxbas);
    }
    inverse_dielectric
}
pub fn w_c_matrix(inverse_dielectric:&MatrixFull<f64>,num_state:usize,ri_full:&MatrixFull<f64>)->MatrixFull<f64>{
    let iterator=ri_full.iter_columns((0..(num_state*num_state))).enumerate();
    let num_auxbas=inverse_dielectric.size[0];
    let mut w_c=MatrixFull::new([num_state,num_state],0.0);
    let mut first_product=vec![0.0;num_auxbas];
    let mut check_matrix=MatrixFull::new([num_state,num_state],0.0);
    for (a,vec) in iterator{
        let n=a/num_state;
        let m=a%num_state;
        if check_matrix[[m,n]]<0.5{
            _dgemv(inverse_dielectric,vec,&mut first_product,'N', 1.0, 0.0, 1, 1);
            w_c[[m,n]]=first_product.iter().zip(vec.iter()).map(|(a,b)|a*b).sum();
            w_c[[n,m]]=w_c[[m,n]];
            check_matrix[[m,n]]=1.0;
            check_matrix[[n,m]]=1.0;
        }
    }
    w_c
}
pub fn contour_rayon(omega:f64,n:usize,quasiparticle_energies_g:&Vec<f64>,quasiparticle_energies_w:&Vec<f64>,occ_size:usize,vir_size:usize,num_state:usize,ri_ov:&MatrixFull<f64>,ri_full:&MatrixFull<f64>)->f64{
    let fermi_energy=(quasiparticle_energies_g[occ_size-1]+quasiparticle_energies_g[occ_size])/2.0;
    let sign=if omega>fermi_energy{1}else{-1};
    let mut contour:f64=0.0;
    let num_auxbas=ri_ov.size[0];
    let mut residue_count=0;
    let mut contour=0.0;
    if sign==1{
        contour=(0..vir_size).into_par_iter().map(|a|{
            let mut contour:f64=0.0;
            let mut residue=0.0;
            if quasiparticle_energies_g[occ_size+a]<omega{
                let mut exist=false;
                let gap=omega-quasiparticle_energies_g[occ_size+a];
                let response=response_matrix(quasiparticle_energies_w,occ_size,vir_size,ri_ov,gap,'C');
                let inverse_dielectric=inverse_dielectric_matrix(&response,'C');
                let vec:Vec<f64>=ri_full.iter_column(n+(occ_size+a)*num_state).copied().collect::<Vec<f64>>();
                let mut first_product=vec![0.0;num_auxbas];
                _dgemv(&inverse_dielectric, &vec, &mut first_product, 'N', 1.0, 0.0, 1, 1);
                residue=first_product.iter().zip(vec.iter()).map(|(a,b)|a*b).sum::<f64>();
                
            }
            residue*=(sign as f64);
            residue
        }).sum()
    }else{
        contour=(0..occ_size).into_par_iter().map(|i|{
            let mut contour:f64=0.0;
            let mut residue=0.0;
            if quasiparticle_energies_g[i]>omega{
                let mut exist=false;
                let gap=quasiparticle_energies_g[i]-omega;
                let response=response_matrix(quasiparticle_energies_w,occ_size,vir_size,ri_ov,gap,'C');
                let inverse_dielectric=inverse_dielectric_matrix(&response,'C');
                let vec:Vec<f64>=ri_full.iter_column(n+i*num_state).copied().collect::<Vec<f64>>();
                let mut first_product=vec![0.0;num_auxbas];
                _dgemv(&inverse_dielectric, &vec, &mut first_product, 'N', 1.0, 0.0, 1, 1);
                residue=first_product.iter().zip(vec.iter()).map(|(a,b)|a*b).sum::<f64>();
            }
            residue*=(sign as f64);
            residue
        }).sum()
    }
    contour
}
pub fn newton_solver<F>(mut f:F,n:usize,consts:f64,ri_ov:&MatrixFull<f64>,ri_full:&MatrixFull<f64>,quasiparticle_energies_g:&Vec<f64>,quasiparticle_energies_w:&Vec<f64>,occ_size:usize,vir_size:usize,num_state:usize,w_c_at_freqs:&Vec<(f64,f64,MatrixFull<f64>)>,starting_point:f64,tol:f64,max_iter:usize,side:f64,printlevel:usize)->f64 where F:Fn(f64,usize,f64,&MatrixFull<f64>,&MatrixFull<f64>,&Vec<f64>,&Vec<f64>,usize,usize,usize,&Vec<(f64,f64,MatrixFull<f64>)>)->f64,{
    let h =0.000001;
    let delta=0.02;
    let mut x_curr=starting_point+side*delta;
    let mut y_curr=f(x_curr,n,consts,ri_ov,ri_full,quasiparticle_energies_g,quasiparticle_energies_w,occ_size,vir_size,num_state,w_c_at_freqs);
    let mut y_plus=f(x_curr+h,n,consts,ri_ov,ri_full,quasiparticle_energies_g,quasiparticle_energies_w,occ_size,vir_size,num_state,w_c_at_freqs);
    let mut y_minus=f(x_curr-h,n,consts,ri_ov,ri_full,quasiparticle_energies_g,quasiparticle_energies_w,occ_size,vir_size,num_state,w_c_at_freqs);
    let mut converge=0;
    let mut iter_times=0;
    loop{
        let derivative=(y_plus-y_minus)/(2.0*h);
        let shift=-y_curr/derivative;
        if printlevel>1{
            println!("newton now:n={},x_curr={},y_curr={},derivative={},shift={}",n,x_curr,y_curr,derivative,shift);
        }
        x_curr=x_curr+shift;
        y_curr=f(x_curr,n,consts,ri_ov,ri_full,quasiparticle_energies_g,quasiparticle_energies_w,occ_size,vir_size,num_state,w_c_at_freqs);
        if shift.abs()<tol{
            converge+=1;
            if printlevel>1{
                println!("convergence: x_curr={},shift={}, y_curr={}",x_curr,shift,y_curr);
            }
        }
        y_plus=f(x_curr+h,n,consts,ri_ov,ri_full,quasiparticle_energies_g,quasiparticle_energies_w,occ_size,vir_size,num_state,w_c_at_freqs);
        y_minus=f(x_curr-h,n,consts,ri_ov,ri_full,quasiparticle_energies_g,quasiparticle_energies_w,occ_size,vir_size,num_state,w_c_at_freqs);
        iter_times+=1;
        if converge==1 || iter_times==max_iter{
            break
        }
    }
    if iter_times==max_iter{
        println!("warning!!! newton solver did not converge for orbital {}!",n);
    }
    x_curr
}
pub fn linear_interpolation_solver<F>(mut f:F,starting_point:f64,side:f64,grid_freqs:usize,span_energy:f64)->(bool,f64) where F:Fn(f64)->f64{
    let h=0.000001;
    let delta=0.02;
    let mut x_curr=starting_point+side*delta;
    let y_curr=f(x_curr);
    let y_minus=f(x_curr-h);
    let y_plus=f(x_curr+h);
    let derivative=(y_plus-y_minus)/(2.0*h);
    let shift=-y_curr/derivative;
    x_curr=x_curr+shift;
    let energy_step=span_energy*2.0/((grid_freqs-1)as f64);
    let grid_results:Vec<(f64,f64)>=(0..grid_freqs).map(|i|{
            let x=x_curr-span_energy+((i as f64)*energy_step);
            let fx=f(x);
            //println!("omega={}, qp_eq={}",x,fx);
            (x,fx)
    }).collect();
    let mut xing=vec![0.0;0];
    let mut answer=0.0;
    for i in (0..grid_freqs-1){
        if grid_results[i].1*grid_results[i+1].1<0.0{
            let slope=(grid_results[i+1].1-grid_results[i].1)/(grid_results[i+1].0-grid_results[i].0);
            xing.push(grid_results[i].0-(grid_results[i].1/slope));
        }
    }
    let mut have_crossing=true;
    //println!("---------------\nfound {} crossings",xing.len());
    if xing.len()>1{
        let mut spectral_weights:Vec<(f64,f64)>=xing.iter().map(|e|{
            let y_minus=f(e-h);
            let y_plus=f(e+h);
            let derivative=(y_plus-y_minus)/(2.0*h)+1.0;
            let spectral_weight=(1.0-derivative).powf(-1.0);
            println!("crossing at {}, spectral value={}",e,spectral_weight);
            (spectral_weight,*e)
        }).collect();
        spectral_weights.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        answer=spectral_weights[0].1;
    }else if xing.len()==1{
        answer=xing[0];
        let y_minus=f(answer-h);
        let y_plus=f(answer+h);
        let derivative=(y_plus-y_minus)/(2.0*h)+1.0;
        let spectral_weight=(1.0-derivative).powf(-1.0);
        println!("crossing at {}, derivative={}, spectral value={}",answer,derivative,spectral_weight);
    }else{
        have_crossing=false;
    }
    (have_crossing,answer)
}
pub fn quasiparticle_equation(omega:f64,n:usize,consts:f64,ri_ov:&MatrixFull<f64>,ri_full:&MatrixFull<f64>,quasiparticle_energies_g:&Vec<f64>,quasiparticle_energies_w:&Vec<f64>,occ_size:usize,vir_size:usize,num_state:usize,w_c_at_freqs:&Vec<(f64,f64,MatrixFull<f64>)>)->f64{
    let contour=contour_rayon(omega,n,quasiparticle_energies_g,quasiparticle_energies_g,occ_size,vir_size,num_state,ri_ov,ri_full);
    let imag=calculate_imag(&w_c_at_freqs,num_state,n,omega,quasiparticle_energies_g,quasiparticle_energies_w);
    //println!("quasiparticle equation residue now:={}",consts+contour-imag-omega);
    consts+contour-imag-omega
}

pub fn spectrum_test(scf_data:&SCF,num_freq:usize){
    let eigenenergies:Vec<f64>=scf_data.eigenvalues[0].clone();
    let mut quasiparticle_energies:Vec<f64>=Vec::new();
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'Y');
    for n in 0..num_state{
        quasiparticle_energies.push(eigenenergies[n]);
    }
    let ri_full:MatrixFull<f64>=ri_bse::get_submatrix(scf_data,'F','F','Y');
    let ri_ov:MatrixFull<f64>=ri_bse::get_submatrix(scf_data,'O','V','Y');
    let start_freq:f64=-0.8;
    let end_freq:f64=0.8;
    let step:f64=0.8/1000.0;
    let w_c_at_freqs=generate_w_c(scf_data,&ri_ov,&ri_full,&quasiparticle_energies,&quasiparticle_energies,num_state,occ_size,vir_size,num_freq);
    for n in (6..15){
        println!("Now is the spectrum of orbital #{}",n);
        (0..2000).into_par_iter().for_each(|w|{
            let freq=start_freq+(w as f64)*step;
            let contour=contour_rayon(freq,n,&quasiparticle_energies,&quasiparticle_energies,occ_size,vir_size,num_state,&ri_ov,&ri_full);
            let imag=calculate_imag(&w_c_at_freqs,num_state,n,freq,&quasiparticle_energies,&quasiparticle_energies);
            let sigma=contour-imag;
            println!("{},{}",freq,sigma);
        });
    }
}
pub fn x_alpha_gw(scf_data:&mut SCF)->Vec<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'Y');
    let ks_energies:Vec<f64>=scf_data.eigenvalues[0].clone();
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let xc_name = scf_data.mol.ctrl.xc.to_lowercase();
    let exchange_under_dfa=get_pure_x_or_c_of_xc(scf_data,&xc_name,'X');
    let x_alpha=qp_ctrl.x_alpha;
    let v_matrix=v_matrix(&scf_data);
    ks_energies.into_iter().enumerate().map(|(n,e_n)|{
        let mut exchange=0.0;
        for i in 0..homo+1{
            exchange-=v_matrix[[n,i]];
        }
        e_n+x_alpha*(exchange-exchange_under_dfa[n])
    }).collect()
}
pub fn linearized_gw(scf_data:&mut SCF,num_freq:usize,vxc_nn:&Vec<f64>,cancel_dfa_xc:bool)->Vec<f64>{
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let delta=qp_ctrl.gw_linearize_shift;
    let v_matrix=v_matrix(&scf_data);
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'Y');
    let mut num_state_qp_range=num_state;
    if qp_ctrl.scgw=="g0w0"{
        let (start_mo_n,num_state_n,occ_size_n,vir_size_n,homo_n,lumo_n)=get_occupation_parameters(scf_data,'N');
        num_state_qp_range=num_state_n;
    }
    let eigenenergies:Vec<f64>=scf_data.eigenvalues[0].clone();
    let ri_mat:MatrixFull<f64>=ri_bse::get_submatrix(scf_data,'F','F','Y');
    let mut quasiparticle_energies_g:Vec<f64>=scf_data.gwqp.0.clone();
    let mut quasiparticle_energies_w:Vec<f64>=scf_data.gwqp.1.clone();
    let ri_ov:MatrixFull<f64>=ri_bse::get_submatrix(scf_data,'O','V','Y');
    //display_and_save_quasiparticles(scf_data,&quasiparticle_energies_g,0);
    let w_c_at_freqs=generate_w_c(scf_data,&ri_ov,&ri_mat,&quasiparticle_energies_g,&quasiparticle_energies_w,num_state,occ_size,vir_size,num_freq);
    let mut save_energies:Vec<f64>=vec![0.0;num_state];
    let h=qp_ctrl.gw_linearize_derivative_h;
    save_energies=(0..num_state_qp_range).map(|n|{
        let side=if n>=occ_size{1.0}else{-1.0};
        let omega_shifted=scf_data.eigenvalues[0][n]-delta*side;
        let contour=contour_rayon(omega_shifted,n,&quasiparticle_energies_g,&quasiparticle_energies_w,occ_size,vir_size,num_state,&ri_ov,&ri_mat);
        let imag_n=calculate_imag(&w_c_at_freqs,num_state,n,omega_shifted,&quasiparticle_energies_g,&quasiparticle_energies_w);
        let mut exchange=0.0;
        
        let contour_plus_h=contour_rayon(omega_shifted+h,n,&quasiparticle_energies_g,&quasiparticle_energies_w,occ_size,vir_size,num_state,&ri_ov,&ri_mat);
        let contour_minus_h=contour_rayon(omega_shifted-h,n,&quasiparticle_energies_g,&quasiparticle_energies_w,occ_size,vir_size,num_state,&ri_ov,&ri_mat);
        let imag_plus_h=calculate_imag(&w_c_at_freqs,num_state,n,omega_shifted+h,&quasiparticle_energies_g,&quasiparticle_energies_w);
        let imag_minus_h=calculate_imag(&w_c_at_freqs,num_state,n,omega_shifted-h,&quasiparticle_energies_g,&quasiparticle_energies_w);
        let self_energy_plus_h=contour_plus_h-imag_plus_h;
        let self_energy_minus_h=contour_minus_h-imag_minus_h;
        let derivative=(self_energy_plus_h-self_energy_minus_h)/(2.0*h);
        if scf_data.mol.ctrl.print_level>1{
            println!("derivative at n={}: {}",n,derivative);
        }
        let spectral_weight=(1.0-derivative).powf(-1.0);
        println!("spectral weight at n={}: {}",n,spectral_weight);
        let real_qp=if cancel_dfa_xc==true{
            let mut exchange=0.0;
            for i in 0..homo+1{
                exchange-=v_matrix[[n,i]];
            }
            //println!("for n={},exchange={}",n,exchange);
            omega_shifted+(spectral_weight*(contour-imag_n+exchange-vxc_nn[n]))
        }else{omega_shifted+(spectral_weight*(contour-imag_n))};
        println!("for n={}, linearized gw yields:qp energy={}",n,real_qp);
        real_qp
    }).collect::<Vec<f64>>().clone();
    quasiparticle_energies_g=save_energies.clone();
    display::full_quasiparticles(&quasiparticle_energies_g,occ_size);
    //display_and_save_quasiparticles(scf_data,&quasiparticle_energies_g,1);
    quasiparticle_energies_g
}
pub fn get_pure_x_or_c_of_xc(scf_data:&mut SCF,name:&str,choice:char)->Vec<f64>{
    let eigenvecs=scf_data.eigenvectors.clone();
    let x_or_c=if choice=='X'{1}else if choice=='C'{2}else{panic!("invalid choice for pure x/c!")};
    let dfa_compnt_save=scf_data.mol.xc_data.dfa_compnt_scf.clone();
    let dfa_paramr_save=scf_data.mol.xc_data.dfa_paramr_scf.clone();
    let mut new_paramr=vec![0.0,0.0,0.0];
    new_paramr[x_or_c]=1.0;
    scf_data.mol.xc_data.dfa_compnt_scf=DFA4REST::libxc_code_fdqc(name).to_vec();
    scf_data.mol.xc_data.dfa_paramr_scf=new_paramr;
    let v_x_or_c_ao=scf_data.generate_vxc_rayon(1.0).2[0].to_matrixfull().unwrap().clone();
    let dimensions=v_x_or_c_ao.size[0];
    let mut v_x_or_c_nn=vec![0.0;dimensions];
    let mut element=0.0;
    for i in (0..dimensions){
        element=0.0;
        for mu in 0..dimensions{
            for nu in 0..dimensions{
                element+=(eigenvecs[0][[mu,i]]*eigenvecs[0][[nu,i]]*v_x_or_c_ao[[mu,nu]]);
            }
        }
        v_x_or_c_nn[i]=element;
    }
    scf_data.mol.xc_data.dfa_compnt_scf=dfa_compnt_save;
    scf_data.mol.xc_data.dfa_paramr_scf=dfa_paramr_save;
    v_x_or_c_nn
}
pub fn get_homo_vx_or_vc(scf_data:&mut SCF,name:&str,choice:char)->f64{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'Y');
    let eigenvecs=scf_data.eigenvectors.clone();
    let x_or_c=if choice=='X'{1}else if choice=='C'{2}else{panic!("invalid choice for pure x/c!")};
    //println!("About to save DFA4REST");
    let dfa_compnt_save=scf_data.mol.xc_data.dfa_compnt_scf.clone();
    let dfa_paramr_save=scf_data.mol.xc_data.dfa_paramr_scf.clone();
    //println!("About to obtain xc_code and change DFA4REST");
    scf_data.mol.xc_data.dfa_compnt_scf=vec![DFA4REST::libxc_code_fdqc(name)[x_or_c]];
    //println!("xc_code has been obtained");
    scf_data.mol.xc_data.dfa_paramr_scf=vec![1.0];
    //println!("About to generate vxc");
    let v_x_or_c_ao=scf_data.generate_vxc_rayon(1.0).2[0].to_matrixfull().unwrap().clone();
    //println!("vxc has been generated");
    let dimensions=v_x_or_c_ao.size[0];
    let mut element=0.0;
    for mu in 0..dimensions{
        for nu in 0..dimensions{
            element+=(eigenvecs[0][[mu,homo]]*eigenvecs[0][[nu,homo]]*v_x_or_c_ao[[mu,nu]]);
        }
    }
    //println!("About to recover DFA4REST");
    scf_data.mol.xc_data.dfa_compnt_scf=dfa_compnt_save;
    scf_data.mol.xc_data.dfa_paramr_scf=dfa_paramr_save;
    //println!("DFA4REST has been recovered");
    element
}
pub fn get_homo_lumo_qp_only(scf_data:&mut SCF,num_freq:usize,vxc_nn:&Vec<f64>,mpi_operator:&Option<MPIOperator>){
    let printlevel=scf_data.mol.ctrl.print_level.clone();
    let v_matrix=v_matrix(&scf_data);
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'Y');
    let mut rs_particles:Vec<f64>=Vec::new();
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let renormalized_singles=qp_ctrl.renormalized_singles;
    if renormalized_singles==true{
        println!("Starts renormalized singles calculations!");
        let w_rs=qp_ctrl.w_rs;
        //println!("Now starting to compute rs_particles");
        rs_particles=renormalized_singles::renormalized_singles_diagonalization(scf_data,w_rs,mpi_operator);
        if printlevel>0{
            println!("rs_particles:{:?}",rs_particles);
        }
        scf_data.renormalized_singles_particles=rs_particles
    }
    let eigenenergies:Vec<f64>=scf_data.eigenvalues[0].clone();
    let mut quasiparticle_energies_g:Vec<f64>=Vec::new();
    let mut quasiparticle_energies_w:Vec<f64>=Vec::new();
    let ri_mat:MatrixFull<f64>=ri_bse::get_submatrix(scf_data,'F','F','Y');
    if scf_data.mol.ctrl.print_level>2{
        println!("ri_mat full:");
        ri_mat.formated_output(1000,"full");
    }
    
    if qp_ctrl.renormalized_singles==true{
        quasiparticle_energies_g=scf_data.renormalized_singles_particles.clone();
        if qp_ctrl.w_rs==true{
            quasiparticle_energies_w=scf_data.renormalized_singles_particles.clone();
        }else{
            for n in 0..num_state{
                quasiparticle_energies_w.push(eigenenergies[n]);
            }
        }
    }else{
        for n in 0..num_state{
            quasiparticle_energies_g.push(eigenenergies[n]);
            quasiparticle_energies_w.push(eigenenergies[n]);
        }
    }
    let ri_ov:MatrixFull<f64>=ri_bse::get_submatrix(scf_data,'O','V','Y');
    let w_c_at_freqs=generate_w_c(scf_data,&ri_ov,&ri_mat,&quasiparticle_energies_g,&quasiparticle_energies_w,num_state,occ_size,vir_size,num_freq);
    let n=homo;
    let mut exchange=0.0;
    for i in 0..homo+1{
        exchange-=v_matrix[[n,i]];
    }
    let consts=eigenenergies[n]+exchange-vxc_nn[n];
    if scf_data.mol.ctrl.print_level>1{
        println!("for n={},exchange={}",n,exchange);
    }
    let side=if n>=occ_size{1.0}else{-1.0};
    let printlevel=scf_data.mol.ctrl.print_level;
    let homo_qp=newton_solver(quasiparticle_equation,n,consts,&ri_ov,&ri_mat,&quasiparticle_energies_g,&quasiparticle_energies_w,occ_size,vir_size,num_state,&w_c_at_freqs,eigenenergies[n],0.00001,50,side,printlevel);
    //println!("for n={}, quasiparticle equation yields:qp energy={}",n,homo_qp);
    let save_path=qp_ctrl.save_qp_path.clone();
    println!("The QP energy of HOMO obtained by GWA is {}",homo_qp);
    let n=lumo;
    let mut exchange=0.0;
    for i in 0..homo+1{
        exchange-=v_matrix[[n,i]];
    }
    let consts=eigenenergies[n]+exchange-vxc_nn[n];
    if scf_data.mol.ctrl.print_level>1{
        println!("for n={},exchange={}",n,exchange);   
    }
    let side=if n>=occ_size{1.0}else{-1.0};
    let printlevel=scf_data.mol.ctrl.print_level;
    let lumo_qp=newton_solver(quasiparticle_equation,n,consts,&ri_ov,&ri_mat,&quasiparticle_energies_g,&quasiparticle_energies_w,occ_size,vir_size,num_state,&w_c_at_freqs,eigenenergies[n],0.00001,50,side,printlevel);
    let save_path=qp_ctrl.save_qp_path.clone();
    if qp_ctrl.save_gw_homo_lumo_qp==true{
        let mut file = OpenOptions::new().append(true).create(true).open(save_path);
        writeln!(file.expect("write failure"), "{},{}",homo_qp,lumo_qp);
    }
    println!("The QP energy of LUMO obtained by GWA is {}",lumo_qp);
}
pub fn get_homo_gw_exchange(scf_data:&SCF)->f64{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'Y');
    let v_matrix=v_matrix(&scf_data);
    let mut exchange=0.0;
    let n=homo;
    for i in 0..homo+1{
        exchange-=v_matrix[[n,i]];
    }
    exchange
}
pub fn obtain_vx_vc_terms(scf_data:&mut SCF){
    println!("starts obtaining HOMO vx and vc terms from various DFAs");
    let mut file = OpenOptions::new().append(true).create(true).open("lda_x_slater.txt");
    let value=get_homo_vx_or_vc(scf_data,"lda_x_slater",'X');
    writeln!(file.expect("write failure"), "{}", value);
    let mut file = OpenOptions::new().append(true).create(true).open("gga_x_b88.txt");
    writeln!(file.expect("write failure"), "{}", get_homo_vx_or_vc(scf_data,"gga_x_b88",'X'));
    let mut file = OpenOptions::new().append(true).create(true).open("gga_x_pbe.txt");
    writeln!(file.expect("write failure"), "{}", get_homo_vx_or_vc(scf_data,"gga_x_pbe",'X'));
    let mut file = OpenOptions::new().append(true).create(true).open("gga_x_xpbe.txt");
    writeln!(file.expect("write failure"), "{}", get_homo_vx_or_vc(scf_data,"gga_x_xpbe",'X'));
    let mut file = OpenOptions::new().append(true).create(true).open("lda_c_vwn.txt");
    writeln!(file.expect("write failure"), "{}", get_homo_vx_or_vc(scf_data,"lda_c_vwn",'C'));
    let mut file = OpenOptions::new().append(true).create(true).open("lda_c_vwn_rpa.txt");
    writeln!(file.expect("write failure"), "{}", get_homo_vx_or_vc(scf_data,"lda_c_vwn_rpa",'C'));
    let mut file = OpenOptions::new().append(true).create(true).open("gga_c_lyp.txt");
    writeln!(file.expect("write failure"), "{}", get_homo_vx_or_vc(scf_data,"gga_c_lyp",'C'));
    let mut file = OpenOptions::new().append(true).create(true).open("gga_c_pbe.txt");
    writeln!(file.expect("write failure"), "{}", get_homo_vx_or_vc(scf_data,"gga_c_pbe",'C'));
    let mut file = OpenOptions::new().append(true).create(true).open("gga_c_xpbe.txt");
    writeln!(file.expect("write failure"), "{}", get_homo_vx_or_vc(scf_data,"gga_c_xpbe",'C'));
}
pub fn read_floats(path: &str) -> io::Result<Vec<f64>> {
    fs::read_to_string(path)?
        .lines()
        .map(|s| s.parse().map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e)))
        .collect()
}