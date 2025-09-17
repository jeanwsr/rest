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
use rest_tensors::matrix::matrix_blas_lapack::{_dgeev,_dgemm_full};
use std::fs::OpenOptions;
use std::{f64, fs::File, io::Write};
//pub mod desert;

pub fn bse_main(scf_data:&mut SCF){
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let quasiparticle_energies=scf_data.gwqp.0.clone();
    if scf_data.mol.ctrl.bse_all==true{
        prepare_ri3mo(scf_data,'N');
        println!("NOW STARTS TRIPLET BSE CALCULATION!!!");
        let (mut wr_1,wi_1,mut wr_2,wi_2,mut wr_3,wi_3)=evaluate_all_excitations(scf_data,&quasiparticle_energies,'T');
        println!("BSE evaluation finished!!!");
        let n=wr_1.len();
        println!("Triplet full diagonalization results:");
        wr_1.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let number=wr_1.len().min(30);
        println!("diag:{:#?}",wr_1.iter().filter(|a|**a>0.0).copied().collect::<Vec<f64>>()[0..number].to_vec());
        println!("complexities:");
        println!("wi_1:{:?}",wi_1);
        println!("end of full diagonalization results");
        println!("~~~~~results of another method~~~~~~");
        wr_2.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let number=wr_2.len().min(30);
        println!("Triplet TDA-BSE diagonalization results:{:#?}",wr_2[0..number].to_vec());
        println!("complexities:");
        println!("wi_2:{:?}",wi_2);
        println!("end of TDA-BSE diagonalization results");
        println!("~~~~~results of another method~~~~~~");
        wr_3.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let number=wr_3.len().min(30);
        println!("Triplet TDA-TDHF diagonalization results:{:#?}",wr_3[0..number].to_vec());
        println!("complexities:");
        println!("wi_3:{:?}",wi_3);
        println!("end of TDA-TDHF diagonalization results");

        println!("NOW STARTS  SINGLET BSE CALCULATION!!!");
        let (mut wr_1,wi_1,mut wr_2,wi_2,mut wr_3,wi_3)=evaluate_all_excitations(scf_data,&quasiparticle_energies,'S');
        println!("BSE evaluation finished!!!");
        let n=wr_1.len();
        println!("Singlet full diagonalization results:");
        wr_1.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let number=wr_1.len().min(30);
        println!("diag:{:#?}",wr_1.iter().filter(|a|**a>0.0).copied().collect::<Vec<f64>>()[0..number].to_vec());
        println!("complexities:");
        println!("wi_1:{:?}",wi_1);
        println!("end of full diagonalization results");
        println!("~~~~~results of another method~~~~~~");
        wr_2.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let number=wr_2.len().min(30);
        println!("Singlet TDA-BSE diagonalization results:{:#?}",wr_2[0..number].to_vec());
        println!("complexities:");
        println!("wi_2:{:?}",wi_2);
        println!("end of TDA-BSE diagonalization results");
        println!("~~~~~results of another method~~~~~~");
        wr_3.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let number=wr_1.len().min(30);
        println!("Singlet TDA-TDHF diagonalization results:{:#?}",wr_3[0..number].to_vec());
        println!("complexities:");
        println!("wi_3:{:?}",wi_3);
        println!("end of TDA-TDHF diagonalization results");
    }else if scf_data.mol.ctrl.bse_spin =="none"{
        println!("No BSE Calculations are triggered");
    }else{
        println!("Specific BSE calculations are triggered");
        prepare_ri3mo(scf_data,'N');
        let bse_spin=scf_data.mol.ctrl.bse_spin.clone();
        println!("BSE Type:{}",bse_spin);
        let xlet=if bse_spin=="triplet"{'T'}else if bse_spin=="singlet"{'S'}else{panic!("invalid choice for bse_spin!")};
        if scf_data.mol.ctrl.bse_tda==false{
            let mut eigens=non_tda_calculations(&scf_data,&quasiparticle_energies,xlet);
            let mut excitations=zip_and_sort(&eigens.0,&eigens.1);
            if scf_data.mol.ctrl.print_level>2{
                show_all_eigenpairs(&excitations);
            }
            let mid = excitations.len() / 2;
            excitations=excitations[mid..].to_vec();
            let number=excitations.len().min(30);
            println!("First {} excitations:",number);
            excitations[0..number].iter().for_each(|(e,v)|{println!("excitation energy={}",e);leading_components(v,occ_size,vir_size)});
            println!("The first excitation obtained by BSE is {}",excitations[0].0);
            if scf_data.mol.ctrl.save_bse_excitations==true{
                let line = excitations.iter().map(|(num,vec)| num.to_string()).collect::<Vec<_>>().join(",");
                let mut file = OpenOptions::new().append(true).create(true).open("bse_excitations.txt");
                writeln!(file.expect("write failure"), "{}", line);
            }
            if scf_data.mol.ctrl.save_first_excitation==true{
                let save_path=scf_data.mol.ctrl.save_first_excitation_path.clone();
                let mut file = OpenOptions::new().append(true).create(true).open(save_path);
                writeln!(file.expect("write failure"), "{}", excitations[0].0);
            }
        }else{
            let mut eigens=tda_calculations(&scf_data,&quasiparticle_energies,xlet);
            let excitations=zip_and_sort(&eigens.0,&eigens.1);
            if scf_data.mol.ctrl.print_level>2{
                show_all_eigenpairs(&excitations);
            }
            let number=excitations.len().min(30);
            println!("First {} excitations:",number);
            excitations[0..number].iter().for_each(|(e,v)|{println!("excitation energy={}",e);leading_components(v,occ_size,vir_size)});
            println!("The first excitation obtained by BSE is {}",excitations[0].0);
            if scf_data.mol.ctrl.save_bse_excitations==true{
                let line = excitations.iter().map(|(num,vec)| num.to_string()).collect::<Vec<_>>().join(",");
                let mut file = OpenOptions::new().append(true).create(true).open("bse_excitations.txt");
                writeln!(file.expect("write failure"), "{}", line);
            }
            if scf_data.mol.ctrl.save_first_excitation==true{
                let save_path=scf_data.mol.ctrl.save_first_excitation_path.clone();
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
    if scf_data.mol.ctrl.bse_qp_polarization==true{
        epsilon=quasiparticle_energies.clone();
    }
    let inverse_dielectric=construct_inverse_dielectric(scf_data,&epsilon);
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
pub fn non_tda_calculations(scf_data:&SCF,quasiparticle_energies:&Vec<f64>,xlet:char)->(Vec<f64>,MatrixFull<f64>){
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let ks_energies:Vec<f64>=scf_data.eigenvalues[0].clone();
    let mut epsilon=ks_energies.clone();
    if scf_data.mol.ctrl.bse_qp_polarization==true{
        epsilon=quasiparticle_energies.clone();
    }
    let inverse_dielectric=construct_inverse_dielectric(scf_data,&epsilon);
    println!("starts contructing Full BSE hamiltonian!!!");
    let bse_hamiltonian=construct_full_bse_hamitonian(scf_data, xlet,&inverse_dielectric,quasiparticle_energies);
    let (matr_b_1, wr_1, wi_1,vl_1,vr_1,info_1)=_dgeev(&bse_hamiltonian, 'N', 'V');
    (wr_1,vr_1)
}
pub fn tda_calculations(scf_data:&SCF,quasiparticle_energies:&Vec<f64>,xlet:char)->(Vec<f64>,MatrixFull<f64>){
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let ks_energies:Vec<f64>=scf_data.eigenvalues[0].clone();
    let mut epsilon=ks_energies.clone();
    if scf_data.mol.ctrl.bse_qp_polarization==true{
        epsilon=quasiparticle_energies.clone();
    }
    let inverse_dielectric=construct_inverse_dielectric(scf_data,&epsilon);
    println!("starts contructing TDA BSE hamiltonian!!!");
    let tda_bse_hamiltonian=construct_submat_a(scf_data,&inverse_dielectric,quasiparticle_energies, xlet);
    let (matr_b_1, wr_1, wi_1,vl_1,vr_1,info_1)=_dgeev(&tda_bse_hamiltonian, 'N', 'V');
    (wr_1,vr_1)
}
pub fn zip_and_sort<'a>(eigenvalues:&'a Vec<f64>,eigenvectors:&'a MatrixFull<f64>)->Vec<(f64,&'a [f64])>{
    let mut eigens:Vec<(f64,&[f64])>=eigenvalues.iter().cloned().zip(eigenvectors.iter_columns_full()).collect();
    eigens.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    eigens
}
pub fn leading_components(eigenvector: &[f64],occ_size:usize, vir_size: usize){
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
pub fn show_all_eigenpairs<'a>(eigenpairs:&Vec<(f64,&'a [f64])>){
    eigenpairs.iter().for_each(|(val,vec)|println!("eigenvalue:{},eigenverctor:{:#?}",val,vec))
}
