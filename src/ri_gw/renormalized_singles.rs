use crate::constants::PI;
use itertools::Itertools;
use std::ops::Range;
use crate::utilities;
use crate::scf_io::SCF;
use libc::seccomp_notif;
use rayon::result;
use reqwest::blocking::Response;
use rest_tensors::{RIFull};
use tensors::{matrix_blas_lapack::{_dinverse,_dsyev}, ri, MathMatrix, MatrixFull};
use rest_tensors::matrix::matrix_blas_lapack::{_dgees,_dgemm,_dgemv};
use rest_tensors::MatrixUpper;
use crate::ri_bse;
use crate::ri_rpa;
use crate::molecule_io;
use crate::scf_io;
use crate::dft;
use crate::dft::DFA4REST;
use std::sync::mpsc::channel;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use crate::mpi_io::{MPIOperator,MPIData};
use crate::ri_gw;

pub fn generate_rs_hamiltonian(scf_data:&mut SCF,mpi_operator:&Option<MPIOperator>)->(MatrixFull<f64>,MatrixFull<f64>){
    println!("Starts generating rs hamiltonian!");
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=ri_gw::get_occupation_parameters(scf_data,'Y');
    //let dfa_oo_hamiltonian=hamiltonian_ao2mo(scf_data,'O');
    //let dfa_vv_hamiltonian=hamiltonian_ao2mo(scf_data,'V');
    //println!("DFA oo hamiltonian");
    //dfa_oo_hamiltonian.formated_output(20,"full");
    //println!("DFA vv hamiltonian");
    //dfa_vv_hamiltonian.formated_output(20,"full");
    scf_data.mol.xc_data.dfa_compnt_scf=vec![];
    scf_io::SCF::generate_hf_hamiltonian_ri_v_dm_only(scf_data,mpi_operator);
    let hf_oo_hamiltonian=hamiltonian_ao2mo(scf_data,'O');
    let hf_vv_hamiltonian=hamiltonian_ao2mo(scf_data,'V');
    (hf_oo_hamiltonian,hf_vv_hamiltonian)
}
pub fn renormalized_singles_diagonalization(scf_data:&mut SCF,w_rs:bool,mpi_operator:&Option<MPIOperator>)->Vec<f64>{
    println!("You are doing renormalized singles GW calculations suggested by Weitao Yang's group:");
    println!("Ye Jin, Neil Qiang Su, and Weitao Yang, J. Phys. Chem. Lett. 10, 3, 447–452 (2019)");
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=ri_gw::get_occupation_parameters(scf_data,'Y');
    let (hf_oo_hamiltonian,hf_vv_hamiltonian)=generate_rs_hamiltonian(scf_data,mpi_operator);
    if scf_data.mol.ctrl.print_level>1{
        println!("HF oo hamiltonian");
        hf_oo_hamiltonian.formated_output(20,"full");
        println!("HF vv hamiltonian");
        hf_vv_hamiltonian.formated_output(20,"full");
    }
    let (occ_rs_values,oo_renormalized)=diagonalize_and_renormalize(scf_data,&hf_oo_hamiltonian,'O');
    let (vir_rs_values,vv_renormalized)=diagonalize_and_renormalize(scf_data,&hf_vv_hamiltonian,'V');
    println!("Renormalized Singles Results:");
    if scf_data.mol.ctrl.print_level>1{
        println!("eigenenergies:{:?}",scf_data.eigenvalues.clone()[0]);
    }
    if scf_data.mol.ctrl.print_level>2{
        println!("oo:");
    oo_renormalized.formated_output(1000,"full");
    println!("vv:");
    vv_renormalized.formated_output(1000,"full");
    }
    if w_rs==true{
        for i in 0..num_state{
            for j in 0..num_state{
                if i<occ_size{
                    scf_data.eigenvectors[0][[j,i]]=oo_renormalized[[j,i]]
                }else{
                    scf_data.eigenvectors[0][[j,i]]=vv_renormalized[[j,i-occ_size]]
                }            
            }
        }
    }
    occ_rs_values.into_iter().chain(vir_rs_values.into_iter()).collect()
}
pub fn hamiltonian_ao2mo(scf_data:&SCF,choice:char)->MatrixFull<f64>{
    println!("now computing:{}",choice);
    let ao_hamiltonian=scf_data.hamiltonian[0].clone().to_matrixfull().unwrap();
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=ri_gw::get_occupation_parameters(scf_data,'Y');
    let mut range_a:Range<usize>=0..1;
    let mut range_b:Range<usize>=0..1;
    let eigenvecs=scf_data.eigenvectors[0].clone();
    let dimensions = match choice {
        'O' => occ_size,
        'V' => vir_size,
        _ => panic!("invalid choice of RS subspace!"),
    };
    let mut hamiltonian=MatrixFull::new([dimensions,dimensions],0.0);
    let mut element=0.0;
    if scf_data.mol.ctrl.print_level>1{
        println!("Now staring to do ao2mo of {}, have a little patience when doing 'V'",choice);
    }
    let ao_dimensions=ao_hamiltonian.size[0];
    if scf_data.mol.ctrl.print_level>1{
        println!("ao dimensions={}",ao_dimensions);
        println!("mo dimensions={}",dimensions);
    }
    let mut c_t_h_ao=MatrixFull::new([dimensions,ao_dimensions],0.0);
    let mut h_mo=MatrixFull::new([dimensions,dimensions],0.0);
    if choice=='O'{
        _dgemm(&eigenvecs,((0..ao_dimensions),(0..occ_size)),'T',
               &ao_hamiltonian,((0..ao_dimensions),(0..ao_dimensions)),'N',
               &mut c_t_h_ao,((0..occ_size),(0..ao_dimensions)),1.0,0.0);
        _dgemm(&c_t_h_ao,((0..occ_size),(0..ao_dimensions)),'N',
               &eigenvecs,((0..ao_dimensions),(0..occ_size)),'N',
               &mut h_mo,((0..occ_size),(0..occ_size)),1.0,0.0);
    }else{
        _dgemm(&eigenvecs,((0..ao_dimensions),(occ_size..ao_dimensions)),'T',
               &ao_hamiltonian,((0..ao_dimensions),(0..ao_dimensions)),'N',
               &mut c_t_h_ao,((0..dimensions),(0..ao_dimensions)),1.0,0.0);
        _dgemm(&c_t_h_ao,((0..dimensions),(0..ao_dimensions)),'N',
               &eigenvecs,((0..ao_dimensions),(occ_size..ao_dimensions)),'N',
               &mut h_mo,((0..dimensions),(0..dimensions)),1.0,0.0);
    }
    
    h_mo
}
pub fn diagonalize_and_renormalize(scf_data:&SCF,subspace_hamiltonian:&MatrixFull<f64>,choice:char)->(Vec<f64>,MatrixFull<f64>){
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=ri_gw::get_occupation_parameters(scf_data,'Y');
    let mut eigenvalues_for_scf: Vec<f64> = Vec::new();
    let spin_channel=scf_data.mol.ctrl.spin_channel;
    let previous_eigenvecs=scf_data.eigenvectors[0].clone();
    let (mut eigenvectors,mut eigenvalues, ndim) = _dsyev(subspace_hamiltonian, 'V');
    let num_state = scf_data.mol.num_state;
    let previous_coeff=previous_eigenvecs.clone();
    let mut renormalized_singles_mo=eigenvectors.unwrap();
    let mut norm:f64=0.0;
    let ndim=ndim as usize;
    for i in 0..ndim{
        for j in 0..ndim{
            norm+=(&renormalized_singles_mo[[j,i]].powf(2.0));
        }
        norm=norm.sqrt();
        //println!("norm at {} has been collected!",i);
        for j in 0..ndim{
            //println!("processing {}th dimension at {} with norm",j,i);
            renormalized_singles_mo[[j,i]]*=norm.powf(-1.0);
        }
        norm=0.0;
    }
    let mut vectors:MatrixFull<f64>=MatrixFull::new([num_state,ndim],0.0);
    for i in 0..ndim{
        for mu in 0..num_state{
            let mut sum=0.0;
            for j in 0..ndim{
                let ind = match choice {
                'O' => j,
                'V' => num_state - ndim + j,
                _ => panic!("无效的选项: {}", choice), // 或返回默认值如 0
                };
                sum+=renormalized_singles_mo[[i,j]]*previous_coeff[[mu,ind]];
            }
            vectors[[mu,i]]=sum;
            sum=0.0;
        }
    }
    /*eigenvectors_for_scf=vectors;
    if choice=='O'{
        for i in start_mo..homo+1{
            for spin in 0..spin_channel{
                scf_data.eigenvalues[spin][i]=eigenvalues_for_scf[spin][i];
            }
        }
    }else if choice=='V'{
        for i in lumo..num_state{
            for spin in 0..spin_channel{
                scf_data.eigenvalues[spin][i]=eigenvalues_for_scf[spin][i-lumo];
            }
        }
    }
    eigenvectors_for_scf*/
    (eigenvalues,vectors)
}