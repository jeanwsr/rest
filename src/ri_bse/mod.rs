//use std::simd::num;
use std::ops::Range;
use crate::scf_io::SCF;
use libc::seccomp_notif;
use rayon::result;
use rest_tensors::{RIFull};
use std::ops::Index;
use tensors::{matrix_blas_lapack::_dinverse, MathMatrix, MatrixFull};
use crate::ri_gw::get_occupation_parameters;
//use libc::select;
//use rest::molecule_io::Molecule;
use crate::ri_gw;
use crate::molecule_io::Molecule;
use rest_tensors::matrix::matrix_blas_lapack::{_dgeev,_dgemm_full,_newton_schulz_inverse_square_root_v02};
use std::fs::OpenOptions;
use std::time::Instant;
use std::{f64, fs::File, io::Write};
pub mod dipoles;
pub mod davidson_solver;
pub mod matvec;
pub mod sbse;

pub fn bse_main(scf_data:&mut SCF){
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let quasiparticle_energies=scf_data.gwqp.0.clone();
    let dipole_matrix=dipoles::compute_dipole_matrix(scf_data);
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    if qp_ctrl.bse_spin =="none"{
        println!("No BSE Calculations are triggered");
    }else{
        println!("Specific BSE calculations are triggered");
        prepare_ri3mo(scf_data,'N');
        let bse_spin=qp_ctrl.bse_spin.clone();
        println!("BSE Type:{}",bse_spin);
        let xlet=if bse_spin=="triplet"{'T'}else if bse_spin=="singlet"{'S'}else{panic!("invalid choice for bse_spin!")};
        if qp_ctrl.simplified_bse==true{
            println!("Now using: Simplified BSE scheme");
            let (ri3ao, mut basbas2baspair, mut baspar2basbas) =  if let Some((riao,basbas2baspair, baspar2basbas))=&scf_data.rimatr {
                (riao,basbas2baspair, baspar2basbas)
            } else {
                panic!("rimatr should be initialized in the preparation of riao");
            };
            let ri3ao=ri3ao.transpose();
            let mut epsilon:Vec<f64>=scf_data.eigenvalues[0].clone();
            let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
            if qp_ctrl.bse_qp_polarization==true{
                epsilon=quasiparticle_energies.clone();
            }
            let energy_diag=construct_energy_diag_for_a(&scf_data.gwqp.0,occ_size,vir_size);
            let inverse_dielectric=construct_inverse_dielectric(scf_data,&epsilon);
            let w_ao=sbse::w_ao_basis(&inverse_dielectric,epsilon.len(),ri3ao.clone());
            let mo_coeff=scf_data.eigenvectors[0].clone();
            let ovlp=scf_data.ovlp.clone().to_matrixfull().unwrap();
            let level_shift = vec![0.0];
            let (ovlp_invsqrt, n_singular, converged_flag) = _newton_schulz_inverse_square_root_v02(&ovlp, 1.0e-8, 1.0, 100, &level_shift);
            let ovlp_sqrt=_dinverse(&ovlp_invsqrt).expect("unsuccessful _dinverse");
            let mut ortho_mo_coeff=MatrixFull::new(ovlp.size,0.0);
            _dgemm_full(&ovlp_sqrt,'N',&mo_coeff,'N',&mut ortho_mo_coeff,1.0,0.0);
            println!("MO coefficients:");
            ortho_mo_coeff.formated_output(1000,"full");
            let ri_ov=get_submatrix(scf_data,'O','V','N');
            let initial_guess=davidson_solver::generate_initial_guess(&energy_diag,qp_ctrl.davidson_target_excitations);
            let excitations=davidson_solver::tda_davidson_solver(scf_data.mol.ctrl.print_level,|z|matvec::sbse_matvec(scf_data,&ortho_mo_coeff,&w_ao,occ_size,vir_size,&ri_ov,&z),qp_ctrl.davidson_target_excitations,&energy_diag,initial_guess,&qp_ctrl);
            if scf_data.mol.ctrl.print_level>2{
                show_all_eigenpairs(&excitations);
            }
            let number=excitations.len().min(30);
            println!("First {} excitations:",number);
            excitations[0..number].iter().for_each(|(e,vec)|{
            let v=dipoles::normalize(vec,true);
            println!("excitation energy={}",e);
            let dipole_square=dipoles::transition_dipole_square(&dipole_matrix,&v,true);
            println!("Transition Dipole Square:{}; Oscillator Strength:{}",dipole_square,dipole_square*e*2.0/3.0);
            leading_components(&v,occ_size,vir_size)});
            println!("The first excitation obtained by BSE is {}",excitations[0].0);
            if qp_ctrl.save_bse_excitations==true{
                let line = excitations.iter().map(|(num,vec)| num.to_string()).collect::<Vec<_>>().join(",");
                let mut file = OpenOptions::new().append(true).create(true).open("bse_excitations.txt");
                writeln!(file.expect("write failure"), "{}", line);
            }
            if qp_ctrl.save_first_excitation==true{
                let save_path=qp_ctrl.save_first_excitation_path.clone();
                let mut file = OpenOptions::new().append(true).create(true).open(save_path);
                writeln!(file.expect("write failure"), "{}", excitations[0].0);
            }
        }else if qp_ctrl.bse_tda==false{
            let mut excitations=non_tda_calculations(&scf_data,&quasiparticle_energies,xlet);
            if scf_data.mol.ctrl.print_level>2{
                show_all_eigenpairs(&excitations);
            }
            let mid = excitations.len() / 2;
            excitations=excitations[mid..].to_vec();
            let number=excitations.len().min(30);
            println!("First {} excitations:",number);
            excitations[0..number].iter().for_each(|(e,vec)|{println!("excitation energy={}",e);
            let v=dipoles::normalize(vec,false);
            let dipole_square=dipoles::transition_dipole_square(&dipole_matrix,&v,false);
            println!("Transition Dipole Square:{}; Oscillator Strength:{}",dipole_square,dipole_square*e*2.0/3.0);
            leading_components(&v,occ_size,vir_size)});
            println!("The first excitation obtained by BSE is {}",excitations[0].0);
            if qp_ctrl.save_bse_excitations==true{
                let line = excitations.iter().map(|(num,vec)| num.to_string()).collect::<Vec<_>>().join(",");
                let mut file = OpenOptions::new().append(true).create(true).open("bse_excitations.txt");
                writeln!(file.expect("write failure"), "{}", line);
            }
            if qp_ctrl.save_first_excitation==true{
                let save_path=qp_ctrl.save_first_excitation_path.clone();
                let mut file = OpenOptions::new().append(true).create(true).open(save_path);
                writeln!(file.expect("write failure"), "{}", excitations[0].0);
            }
        }else{
            let excitations=tda_calculations(&scf_data,&quasiparticle_energies,xlet);
            if scf_data.mol.ctrl.print_level>2{
                show_all_eigenpairs(&excitations);
            }
            let number=excitations.len().min(30);
            println!("First {} excitations:",number);
            excitations[0..number].iter().for_each(|(e,vec)|{
                let v=dipoles::normalize(vec,true);
                println!("excitation energy={}",e);
                let dipole_square=dipoles::transition_dipole_square(&dipole_matrix,&v,true);
                println!("Transition Dipole Square:{}; Oscillator Strength:{}",dipole_square,dipole_square*e*2.0/3.0);
                leading_components(&v,occ_size,vir_size)});
            println!("The first excitation obtained by BSE is {}",excitations[0].0);
            if qp_ctrl.save_bse_excitations==true{
                let line = excitations.iter().map(|(num,vec)| num.to_string()).collect::<Vec<_>>().join(",");
                let mut file = OpenOptions::new().append(true).create(true).open("bse_excitations.txt");
                writeln!(file.expect("write failure"), "{}", line);
            }
            if qp_ctrl.save_first_excitation==true{
                let save_path=qp_ctrl.save_first_excitation_path.clone();
                let mut file = OpenOptions::new().append(true).create(true).open(save_path);
                writeln!(file.expect("write failure"), "{}", excitations[0].0);
            }
        }
    }
}
pub fn prepare_ri3mo(scf_data:&mut SCF,response_or_not:char){
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=ri_gw::get_occupation_parameters(scf_data,response_or_not);
    let (start_mo,num_state_response,occ_size,vir_size_response,homo,lumo)=ri_gw::get_occupation_parameters(scf_data,'Y');
    let mut range:(Range<usize>, Range<usize>);
    let range_ov=(start_mo..homo+1, lumo..num_state_response);
    let range_ff=(start_mo..num_state,start_mo..num_state);
    let mut rimatr=scf_data.rimatr.clone();
    scf_data.generate_ri3mo_rayon(range_ov.0,range_ov.1);
    scf_data.rimatr=rimatr.clone();
    scf_data.generate_ri3mo_full_rayon(range_ff.0,range_ff.1);
    scf_data.rimatr=rimatr;
}
pub fn get_submatrix(scf_data:&SCF,choice_a:char,choice_b:char,response_or_not:char)->MatrixFull<f64>{
    let mut vector:Vec<(RIFull<f64>,Range<usize>,Range<usize>)>=Vec::new();
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=ri_gw::get_occupation_parameters(scf_data,'N');
    let range_oo=(start_mo..homo+1, start_mo..homo+1);
    let range_vv=(lumo..num_state, lumo..num_state);
    let range_ov=(start_mo..homo+1, lumo..num_state);
    if choice_a=='O'&&choice_b=='V'&&response_or_not=='Y'{
        vector=scf_data.ri3mo.clone().unwrap();
    }else if choice_a=='F'&&choice_b=='F'{
        vector=scf_data.ri3mo_full.clone().unwrap();
    }else if choice_a=='O'&&choice_b=='O'{
        let mut scf_clone=scf_data.clone();
        scf_clone.generate_ri3mo_rayon(range_oo.0,range_oo.1);
        vector=scf_clone.ri3mo.clone().unwrap();
    }else if choice_a=='V'&&choice_b=='V'{
        let mut scf_clone=scf_data.clone();
        scf_clone.generate_ri3mo_rayon(range_vv.0,range_vv.1);
        vector=scf_clone.ri3mo.clone().unwrap();
    }else if choice_a=='O'&&choice_b=='V'&&response_or_not=='N'{
        let mut scf_clone=scf_data.clone();
        scf_clone.generate_ri3mo_rayon(range_ov.0,range_ov.1);
        vector=scf_clone.ri3mo.clone().unwrap();
    }else {
        panic!("invalid choice of ri subspace!")
    };
    let matrix:MatrixFull<f64>=vector[0].0.rifull_to_matfull_i_jk();
    matrix
}
pub fn construct_raw_w(ri_left:&MatrixFull<f64>,ri_right:&MatrixFull<f64>,inverse_dielectric:&MatrixFull<f64>)->MatrixFull<f64>{
    let mut first_product:MatrixFull<f64>=MatrixFull::new([ri_left.size[1],ri_left.size[0]],0.0);
    let mut w:MatrixFull<f64>=MatrixFull::new([ri_left.size[1],ri_right.size[1]],0.0);
    _dgemm_full(ri_left,'T',inverse_dielectric,'N',&mut first_product,1.0,0.0);
    _dgemm_full(&first_product,'N',ri_right,'N',&mut w,1.0,0.0);
    w
}
pub fn construct_coulomb(ri_left:&MatrixFull<f64>,ri_right:&MatrixFull<f64>)->MatrixFull<f64>{
    let mut coulomb:MatrixFull<f64>=MatrixFull::new([ri_left.size[1],ri_right.size[1]],0.0);
    _dgemm_full(ri_left,'T',ri_right,'N',&mut coulomb,1.0,0.0);
    coulomb
}
pub fn reorganize_w(w:MatrixFull<f64>,part:char,occ_size:usize,vir_size:usize)->MatrixFull<f64>{
    //println!("starts reoorganizing W!!!");
    let mut reorganized_w:MatrixFull<f64>=MatrixFull::new([occ_size*vir_size,occ_size*vir_size],0.0);
    if part=='A'{
        for i in 0..occ_size {
            for j in 0..occ_size {
                for a in 0..vir_size {
                    for b in 0..vir_size {
                        // 根据给定的转换规则计算目标位置
                        //let target_row = i * vir_size + k;
                        let target_row=a*occ_size+i;
                        //let target_col = j * vir_size + l;
                        let target_col=b*occ_size+j;
                        // 计算在原矩阵a中的位置
                        //let source_row = i * occ_size + j;
                        let source_row=j*occ_size+i;
                        let source_col=b*vir_size+a;
                       // let source_col = k * vir_size + l;
                        // 将值从a复制到b中相应的位置
                        reorganized_w[[target_row,target_col]] = w[[source_row,source_col]];
                    }
                }
            }
        }
    }
    if part=='B'{
        for i in 0..occ_size {
            for j in 0..occ_size {
                for a in 0..vir_size {
                    for b in 0..vir_size {
                        
                        // 根据给定的转换规则计算目标位置
                        let target_row=a*occ_size+i;
                        let target_col=b*occ_size+j;
                        //TEMPORARY INDICIES:
                        //let temp_row=a*occ_size+i;
                        //let temp_col=j*vir_size+b;


                        // 计算在原矩阵a中的位置
                        //SAVED SCHEME
                        //let source_row=b*occ_size+i;
                        //let source_col=j*vir_size+a;
                        //TEST SCHEME
                        let source_row=b*occ_size+i;
                        let source_col=a*occ_size+j;
                        // 将值从a复制到b中相应的位置
                        reorganized_w[[target_row,target_col]] = w[[source_row,source_col]];
                    }
                }
            }
        }
    }
    reorganized_w
}
pub fn identity_matrix(n: usize) -> MatrixFull<f64> {
    // 创建一个大小为 n * n 的矩阵，初始值为 0.0
    let mut mat = MatrixFull::new([n, n], 0.0);
    // 设置主对角线上的元素为 1.0
    for i in 0..n {
        mat[[i, i]] = 1.0;
    }
    mat
}
pub fn construct_energy_diag_for_a(quasiparticle_energies:&Vec<f64>,occ_size:usize,vir_size:usize)->Vec<f64>{
    let mut energy_diag = Vec::new();
    for a in 0..vir_size{
        for i in 0..occ_size{
            let energy_gap=quasiparticle_energies[occ_size+a]-quasiparticle_energies[i];
            energy_diag.push(energy_gap);
        }
    }
    energy_diag
}
pub fn construct_inverse_dielectric(scf_data:&SCF,epsilon:&Vec<f64>)->MatrixFull<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=ri_gw::get_occupation_parameters(scf_data,'Y');
    let ri_ov=get_submatrix(scf_data,'O','V','Y');
    if scf_data.mol.ctrl.print_level>1{
        println!("occ_size={},vir_size(for response)={}",occ_size,vir_size);
    }
    let response=ri_gw::response_matrix(epsilon,occ_size,vir_size,&ri_ov,0.0,'R');
    let inverse_dielectric=ri_gw::inverse_dielectric_matrix(&response,'R');
    inverse_dielectric
}
pub fn construct_submat_a(scf_data:&SCF,inverse_dielectric:&MatrixFull<f64>,quasiparticle_energies:&Vec<f64>,xlet:char)->MatrixFull<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let ri_ov=get_submatrix(scf_data,'O','V','N');
    let mut v=construct_coulomb(&ri_ov,&ri_ov);
    v.self_multiple(2.0);
    let ri_oo=get_submatrix(scf_data,'O','O','N');
    let ri_vv=get_submatrix(scf_data,'V','V','N');
    let raw_w=construct_raw_w(&ri_oo,&ri_vv,&inverse_dielectric);
    let mut w=reorganize_w(raw_w, 'A', occ_size, vir_size);
    w.self_multiple(-1.0);
    let mut energy_diag=construct_energy_diag_for_a(quasiparticle_energies,occ_size,vir_size);
    let mut a=w;
    a.iter_diagonal_mut().unwrap().zip(energy_diag.iter_mut()).for_each(|(x,e)|{(*x,*e)=(*x+*e,*e)});
    if xlet=='S'{
        a=MatrixFull::add(&v,&a).unwrap();
    }
    a
}
pub fn construct_submat_b(scf_data:&SCF,xlet:char,inverse_dielectric:&MatrixFull<f64>)->MatrixFull<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let ri_ov=get_submatrix(scf_data,'O','V','N');
    let mut v=construct_coulomb(&ri_ov,&ri_ov);
    v.self_multiple(2.0);
    let raw_w=construct_raw_w(&ri_ov,&ri_ov,&inverse_dielectric);
    let mut w=reorganize_w(raw_w, 'B', occ_size, vir_size);
    w.self_multiple(-1.0);
    let mut b=w;
    if xlet=='S'{
        b=MatrixFull::add(&b,&v).unwrap();
    }
    b
}
pub fn construct_full_bse_hamitonian(scf_data:&SCF,xlet:char,inverse_dielectric:&MatrixFull<f64>,quasiparticle_energies:&Vec<f64>)->MatrixFull<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let mut hamiltonian:MatrixFull<f64>=MatrixFull::new([2*occ_size*vir_size,2*occ_size*vir_size],0.0);
    let a:MatrixFull<f64>=construct_submat_a(scf_data,inverse_dielectric,quasiparticle_energies, xlet);
    let mut minus_a=a.transpose();
    let b=construct_submat_b(scf_data,xlet,inverse_dielectric);
    let transpose_b=b.transpose();
    let mut minus_b=b;
    minus_a.self_multiple(-1.0);
    minus_b.self_multiple(-1.0);
    for i in 0..occ_size*vir_size {
        for j in 0..vir_size*occ_size{
            hamiltonian[[i,j]]=a[[i,j]];
            hamiltonian[[occ_size*vir_size+i,j]]=minus_b[[i,j]];
            hamiltonian[[i,occ_size*vir_size+j]]=transpose_b[[i,j]];
            hamiltonian[[occ_size*vir_size+i,occ_size*vir_size+j]]=minus_a[[i,j]];
        }
    }
    hamiltonian
}
pub fn construct_tdhf_tda_hamiltonian(scf_data:&SCF,xlet:char,quasiparticle_energies:&Vec<f64>)->MatrixFull<f64>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let ri_ov=get_submatrix(scf_data,'O','V','N');
    let mut v=construct_coulomb(&ri_ov,&ri_ov);
    v.self_multiple(2.0);
    let ri_oo=get_submatrix(scf_data,'O','O','N');
    let ri_vv=get_submatrix(scf_data,'V','V','N');
    let raw_exchange=construct_coulomb(&ri_oo,&ri_vv);
    let mut exchange=reorganize_w(raw_exchange, 'A', occ_size, vir_size);
    exchange.self_multiple(-1.0);
    let mut energy_diag=construct_energy_diag_for_a(quasiparticle_energies,occ_size,vir_size);
    let mut a=exchange;
    a.iter_diagonal_mut().unwrap().zip(energy_diag.iter_mut()).for_each(|(x,e)| {(*x,*e)=(*x+*e,*e)});
    if xlet=='S'{
        a=MatrixFull::add(&v,&a).unwrap();
    }
    a
}
extern "C" fn select(ar: *const f64, ai: *const f64) -> i32 {
    unsafe{if ((*ar)*(*ar)).sqrt() < 1.0e-8 || ((*ai)*(*ai)).sqrt() > 1.0e-8 { 0 } else { 1 }} // Example criterion
}
pub fn evaluate_all_excitations(scf_data:&SCF,quasiparticle_energies:&Vec<f64>,xlet:char)->(Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>,Vec<f64>){
    let ks_energies:Vec<f64>=scf_data.eigenvalues[0].clone();
    let mut epsilon=ks_energies.clone();
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    if qp_ctrl.bse_qp_polarization==true{
        epsilon=quasiparticle_energies.clone();
    }
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    println!("homo:{},lumo:{},occ_size:{},vir_size:{},start_mo:{},num_state:{}",homo,lumo,occ_size,vir_size,start_mo,num_state);
    println!("--------------------------------");
    println!("starts contructing Full BSE hamiltonian!!!");
    let inverse_dielectric=construct_inverse_dielectric(scf_data,&epsilon);
    println!("the inverse dielectric that is passed into CFBH is of size:{},{}",inverse_dielectric.size[0],inverse_dielectric.size[1]);
    let bse_hamiltonian=construct_full_bse_hamitonian(scf_data, xlet,&inverse_dielectric,quasiparticle_energies);
    println!("--------------------------------");
    let (matr_b_1, wr_1, wi_1,vl_1,vr_1,info_1)=_dgeev(&bse_hamiltonian, 'N', 'V');
    println!("--------------------------------");
    println!("starts contructing TDA BSE hamiltonian!!!");
    let tda_bse_hamiltonian=construct_submat_a(scf_data,&inverse_dielectric,quasiparticle_energies, xlet);
    println!("--------------------------------");
    //tda_bse_hamiltonian.formated_output(occ_size*vir_size,"full");
    let (matr_b_2, wr_2, wi_2,vl_2,vr_2,info_2)=_dgeev(&tda_bse_hamiltonian, 'N', 'V');
    //println!("_dgees has been finished!!!");
    //let excitation_energies=excitations.2;
    //excitation_energies
    println!("--------------------------------");
    println!("starts contructing TDA TDHF hamiltonian!!!");
    let tda_tdhf_hamiltonian=construct_tdhf_tda_hamiltonian(scf_data, xlet, quasiparticle_energies);
    println!("--------------------------------");
    //tda_tdhf_hamiltonian.formated_output(occ_size*vir_size,"full");
    let (matr_b_3, wr_3, wi_3,vl_3,vr_3,info_3)=_dgeev(&tda_bse_hamiltonian, 'N', 'V');
    (wr_1,wi_1,wr_2,wi_2,wr_3,wi_3)
}
pub fn non_tda_calculations(scf_data:&SCF,quasiparticle_energies:&Vec<f64>,xlet:char)->Vec<(f64,Vec<f64>)>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let ks_energies:Vec<f64>=scf_data.eigenvalues[0].clone();
    let mut epsilon=ks_energies.clone();
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    if qp_ctrl.bse_qp_polarization==true{
        epsilon=quasiparticle_energies.clone();
    }
    let inverse_dielectric=construct_inverse_dielectric(scf_data,&epsilon);
    let mut eigenpairs:Vec<(f64,Vec<f64>)>=Vec::new();
    if qp_ctrl.bse_davidson_solver==true{
        let ri_oo=get_submatrix(scf_data,'O','O','N');
        let mut ri_oo_tilde:MatrixFull<f64>=MatrixFull::new(ri_oo.size,0.0);
        _dgemm_full(&inverse_dielectric,'N',&ri_oo,'N',&mut ri_oo_tilde,1.0,0.0);
        drop(ri_oo);
        let ri_ov=get_submatrix(scf_data,'V','V','N');
        let ri_vv=get_submatrix(scf_data,'V','V','N');
        let mut ri_ov_tilde:MatrixFull<f64>=MatrixFull::new(ri_ov.size,0.0);
        _dgemm_full(&inverse_dielectric,'N',&ri_ov,'N',&mut ri_ov_tilde,1.0,0.0);
        drop(inverse_dielectric);
        let energy_diag=construct_energy_diag_for_a(&scf_data.gwqp.0,occ_size,vir_size);
        let initial_guess=davidson_solver::generate_initial_guess(&energy_diag,qp_ctrl.davidson_target_excitations);
        eigenpairs=davidson_solver::lr_davidson_solver(scf_data.mol.ctrl.print_level,|z|matvec::a_block_matvec(scf_data,&ri_vv,&ri_ov,&ri_oo_tilde,&z),|z|matvec::b_block_matvec(scf_data,&ri_ov,&ri_ov_tilde,&z),qp_ctrl.davidson_target_excitations,&energy_diag,initial_guess);
    }else{
        println!("starts contructing Full BSE hamiltonian!!!");
        let bse_hamiltonian=construct_full_bse_hamitonian(scf_data, xlet,&inverse_dielectric,quasiparticle_energies);
        let (matr_b_1, wr_1, wi_1,vl_1,vr_1,info_1)=_dgeev(&bse_hamiltonian, 'N', 'V');
        eigenpairs=zip_and_sort(&wr_1,&vr_1);
    }
    eigenpairs
}
pub fn tda_calculations(scf_data:&SCF,quasiparticle_energies:&Vec<f64>,xlet:char)->Vec<(f64,Vec<f64>)>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let ks_energies:Vec<f64>=scf_data.eigenvalues[0].clone();
    let mut epsilon=ks_energies.clone();
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    if qp_ctrl.bse_qp_polarization==true{
        epsilon=quasiparticle_energies.clone();
    }
    let inverse_dielectric=construct_inverse_dielectric(scf_data,&epsilon);
    let mut eigenpairs:Vec<(f64,Vec<f64>)>=Vec::new();
    if qp_ctrl.bse_davidson_solver==true{
        let start=Instant::now();
        let ri_ov=get_submatrix(scf_data,'O','V','N');
        let duration1=start.elapsed();
        println!("RI-OV耗时: {:?}", duration1);
        let mut ri_vv=get_submatrix(scf_data,'V','V','N');
        let num_auxbas=inverse_dielectric.size[0];
        //ri_vv.reshape([num_auxbas*vir_size,vir_size]);
        let duration2=start.elapsed();
        println!("RI-VV作耗时: {:?}", duration2-duration1);
        let ri_oo=get_submatrix(scf_data,'O','O','N');
        let duration3=start.elapsed();
        println!("RI-OO耗时: {:?}", duration3-duration2);
        let mut ri_oo_tilde:MatrixFull<f64>=MatrixFull::new(ri_oo.size,0.0);
        _dgemm_full(&inverse_dielectric,'N',&ri_oo,'N',&mut ri_oo_tilde,1.0,0.0);
        drop(inverse_dielectric);
        ri_oo_tilde.reshape([num_auxbas*occ_size,occ_size]);
        ri_oo_tilde=ri_oo_tilde.transpose_and_drop();
        ri_oo_tilde.reshape([occ_size*num_auxbas,occ_size]);
        ri_vv.reshape([num_auxbas*vir_size,vir_size]);
        //ri_oo_tilde.reshape([num_auxbas*occ_size,occ_size]);
        let energy_diag=construct_energy_diag_for_a(&scf_data.gwqp.0,occ_size,vir_size);
        let initial_guess=davidson_solver::generate_initial_guess(&energy_diag,qp_ctrl.davidson_target_excitations);
        eigenpairs=davidson_solver::tda_davidson_solver(scf_data.mol.ctrl.print_level,|z|matvec::a_block_matvec(scf_data,&ri_vv,&ri_ov,&ri_oo_tilde,&z),qp_ctrl.davidson_target_excitations,&energy_diag,initial_guess,&qp_ctrl);
    }else{
        println!("starts contructing TDA BSE hamiltonian!!!");
        let tda_bse_hamiltonian=construct_submat_a(scf_data,&inverse_dielectric,quasiparticle_energies, xlet);
        let (matr_b_1, wr_1, wi_1,vl_1,vr_1,info_1)=_dgeev(&tda_bse_hamiltonian, 'N', 'V');
        eigenpairs=zip_and_sort(&wr_1,&vr_1);
    }
    eigenpairs
}
pub fn zip_and_sort(
    eigenvalues: &Vec<f64>,
    eigenvectors: &MatrixFull<f64>
    ) -> Vec<(f64,Vec<f64>)> {
    let mut eigens:Vec<(f64,Vec<f64>)>=eigenvalues.iter().zip(eigenvectors.iter_columns_full()).map(|(e,v)|(*e,v.to_vec())).collect();
    eigens.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    eigens
}
pub fn leading_components(eigenvector: &Vec<f64>,occ_size:usize, vir_size: usize){
    let mut components: Vec<(usize, usize, f64)> = eigenvector
        .iter()
        .enumerate()
        .map(|(n, x)| {
            let index = n; // 从0开始的索引
            let j = occ_size+index / occ_size;  // 整除
            let i = index % occ_size;  // 取余
            (i, j, *x)
        })
        .collect();
    
    components.sort_by(|a, b| {
        b.2.abs().partial_cmp(&a.2.abs())  // 降序排列（绝对值最大的在前）
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let length=components.len().min(5);
    for i in 0..length{
        println!("      #{}->#{},amplitude={}",components[i].0,components[i].1,components[i].2);
    }
}
pub fn show_all_eigenpairs(eigenpairs:&Vec<(f64,Vec<f64>)>){
    eigenpairs.iter().for_each(|(val,vec)|println!("eigenvalue:{},eigenvector:{:#?}",val,vec))
}
