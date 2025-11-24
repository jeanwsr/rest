use rest_tensors::matrix::{MatrixFull,MathMatrix,matrixupper::MatrixUpper};
use rest_tensors::matrix::matrix_blas_lapack::{_dgeev,_dgemm_full, _dgemv,_dpotrf,_dsyev,_dinverse};
use std::time::Instant;
use crate::ri_bse::matvec;
use crate::scf_io::{SCF, SCFType};
use crate::ri_bse::get_occupation_parameters;
use std::cmp;
use crate::ri_bse;
use itertools::Itertools;
use crate::ctrl_io::quasiparticle_methods::QuasiParticle;

pub fn zip_and_sort(
    eigenvalues: &Vec<f64>,
    eigenvectors: &MatrixFull<f64>
    ) -> Vec<(f64, Vec<f64>)> {
    let mut eigens: Vec<_> = eigenvalues.iter()
        .cloned()
        .zip(eigenvectors.iter_columns_full())
        .collect();
    eigens.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    eigens.into_iter().map(|(e,v)|(e,v.to_vec())).collect()
}
pub fn generate_initial_guess(diag:&Vec<f64>,nroots_ctrl:usize)->MatrixFull<f64>{
    let mut indicies:Vec<_>=diag.iter().enumerate().collect();
    indicies.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let nroots=cmp::max(nroots_ctrl,2);
    let indicies:Vec<_>=indicies.iter().take(nroots).map(|(i,_)|*i).collect();
    let mut initial_guess=MatrixFull::new([diag.len(),nroots],0.0);
    indicies.iter().enumerate().for_each(|(k,i)|{
        initial_guess[[*i,k]]=1.0;
    });
    initial_guess
}
pub fn tda_davidson_solver<F1>(print_level:usize,a_matvec:F1,nroots:usize,diag:&Vec<f64>,initial_guess:MatrixFull<f64>,qp_ctrl:&QuasiParticle)->Vec<(f64,Vec<f64>)>
where F1:Fn(&Vec<f64>)->Vec<f64>{
    let mut ss=initial_guess;
    //ss:search space S, where each column in S is a search vector
    let occ_vir=diag.len();
    //occ_vir:length of dimension of a,b as well as x,y. In TDDFT/BSE it is occ_size*vir_size
    let mut x_solutions=MatrixFull::new([occ_vir,0],0.0);
    let mut eigenvalues:Vec<f64>=Vec::new();
    let mut iter_num=0;
    let qp_ctrl=
    loop{
        iter_num+=1;
        let m=ss.size[1];
        let mut a_ss=MatrixFull::new([occ_vir,0],0.0);
        //a_ss:AS
        ss.iter_columns_full().for_each(|ss_i|{
            let z=ss_i.to_vec();
            let a_z=a_matvec(&z);
            a_ss.push_column(&a_z);
        });
        if print_level>1{
            println!("m={}",m);
        }
        let mut ss_t_a_ss=MatrixFull::new([m,m],0.0);
        //ss_t_a_ss:projection matrix of a
        _dgemm_full(&ss,'T',&a_ss,'N',&mut ss_t_a_ss,1.0,0.0);
        if print_level>1{println!("A projection:");
        ss_t_a_ss.formated_output(1000,"full");}
        drop(a_ss);
        let (Some(x_proj),omega,_)=_dsyev(&ss_t_a_ss,'V')else{panic!("dsyev failure!")};
        //ginv_xpy:G^{-1}(X+Y),eigenvectors of G^T(A-B)G,denoted as u in some literature,omega2:square of desired excitation energies
        let mut x_proj_nroots=MatrixFull::new([m,0],0.0);
        let mut omega_nroots:Vec<f64>=Vec::new();
        if print_level>1{println!("omega={:#?}",omega);}
        omega.iter().zip(x_proj.iter_columns_full()).enumerate().for_each(|(n,(omega_i,x_i))|
            if n<nroots{
                omega_nroots.push(*omega_i);
                x_proj_nroots.push_column(x_i);
            });
        let m=ss_t_a_ss.size[0];
        //m:subspace dimensions
        if print_level>1{println!("omega:{:#?}",omega);}
        let mut x_full=MatrixFull::new([occ_vir,nroots],0.0);
        _dgemm_full(&ss,'N',&x_proj_nroots,'N',&mut x_full,1.0,0.0);
        //xmy_full,xpy_full:Real X+Y and X-Y spanned in MO-pair basis
        let mut ax_full=MatrixFull::new([occ_vir,0],0.0);
        x_full.iter_columns_full().for_each(|x_i|{
            let z=x_i.to_vec();
            let ax_i=a_matvec(&z);
            ax_full.push_column(&ax_i);
        });
        let mut residues=ax_full;
        let mut eigenvalue_matrix=MatrixFull::new([nroots,nroots],0.0);
        (0..nroots).for_each(|i|eigenvalue_matrix[[i,i]]=omega[i]);
        //diagonal matrix of eigenvalues, to make it possible to obtain residues with LAPACK
        let mut omega_x_full=MatrixFull::new([occ_vir,nroots],0.0);
        _dgemm_full(&x_full,'N',&eigenvalue_matrix,'N',&mut omega_x_full,1.0,0.0);
        //omega_x: Omega(diagonal)X
        let mut omega_x=MatrixFull::new([occ_vir,nroots],0.0);
        _dgemm_full(&x_full,'N',&eigenvalue_matrix,'N',&mut omega_x_full,1.0,0.0);
        residues=residues.scaled_add(&omega_x_full,-1.0).unwrap();
        println!("Now is iteration #{},current progress:",iter_num);
        println!("Residues:");
        residues.iter_columns_full().for_each(|vec|{
            println!("{}",vec.iter().map(|x|x.powf(2.0)).sum::<f64>().powf(0.5));
        });
        let mut converge=true;
        residues.iter_columns_full().for_each(|residue_i|{
            let norm=residue_i.iter().fold(0.0,|acc,val|acc+val.powf(2.0));
            if norm>1e-10{
                converge=false;
            }
        });
        if converge{
            x_solutions.append_column(&x_full);
            eigenvalues=omega;
            break;
        }
        let restart=if ss.size[1]>qp_ctrl.davidson_maximum_subspace_size{true}else{false};
        if !restart{residues.iter_columns_full().enumerate().for_each(|(i,residue_i)|{
            let mut preconditioned=residue_i.iter().enumerate().map(|(j,v_k)|v_k/(omega[i]-diag[j])).collect();
            //preconditioned:preconditioned vector:
            //(X-Y)=(omega-diag)
            if print_level>2{println!("Un-orthogonalized to add:{:#?}",preconditioned);}
            ss.iter_columns_full().for_each(|v_k|{
                let subspace_vec=v_k.to_vec();
                let projection_product=dot_product(&subspace_vec,&preconditioned);
                if print_level>2{println!("projection={}",projection_product);}
                preconditioned=vector_scaled_add(&preconditioned,1.0,&subspace_vec,-projection_product);
            });
            let orthogonalized_norm=dot_product(&preconditioned,&preconditioned).powf(0.5);
            //if print_level>1{println!("Before normalization:{:#?},norm={}",preconditioned,orthogonalized_norm);}
            if orthogonalized_norm>1e-8{
                preconditioned=num_product(&preconditioned,1.0/orthogonalized_norm);
                ss.push_column(&preconditioned);
                println!("A new search vector has been added. Search space now has {} vectors",ss.size[1]);
            }
        });}else{
            ss=MatrixFull::new([occ_vir,0],0.0);
            x_full.iter_columns_full().take(qp_ctrl.davidson_restart_dimensions).for_each(|x_i|ss.push_column(x_i));
        }
        if print_level>2{println!("Search Space:");
        ss.formated_output(1000,"full");}
        if iter_num>qp_ctrl.davidson_max_iter{
            break;
        }
    };
    let eigenvectors:Vec<_>=x_solutions.iter_columns_full().map(|v1|{
        let mut vec=v1.to_vec();
        vec
    }).collect();
    let mut eigenpairs:Vec<_>=eigenvalues.iter().zip(eigenvectors.iter()).map(|(value,vector)|(*value,vector.clone())).collect();
    eigenpairs.sort_by(|a,b|a.0.partial_cmp(&b.0).unwrap());
    eigenpairs
}



pub fn lr_davidson_solver<F1,F2>(print_level:usize,a_matvec:F1,b_matvec:F2,nroots:usize,diag:&Vec<f64>,initial_guess:MatrixFull<f64>)->Vec<(f64,Vec<f64>)>
where F1:Fn(&Vec<f64>)->Vec<f64>,F2:Fn(&Vec<f64>)->Vec<f64>{
    let mut ss=initial_guess;
    //ss:search space S, where each column in S is a search vector
    let occ_vir=diag.len();
    //occ_vir:length of dimension of a,b as well as x,y. In TDDFT/BSE it is occ_size*vir_size
    let mut x_solutions=MatrixFull::new([occ_vir,0],0.0);
    let mut y_solutions=MatrixFull::new([occ_vir,0],0.0);
    let mut eigenvalues:Vec<f64>=Vec::new();
    let mut iter_num=0;
    loop{
        iter_num+=1;
        let m=ss.size[1];
        let mut a_ss=MatrixFull::new([occ_vir,0],0.0);
        //a_ss:AS
        ss.iter_columns_full().for_each(|ss_i|{
            let z=ss_i.to_vec();
            let a_z=a_matvec(&z);
            a_ss.push_column(&a_z);
        });
        if print_level>1{
            println!("m={}",m);
        }
        let mut ss_t_a_ss=MatrixFull::new([m,m],0.0);
        //ss_t_a_ss:projection matrix of a
        _dgemm_full(&ss,'T',&a_ss,'N',&mut ss_t_a_ss,1.0,0.0);
        if print_level>1{println!("A projection:");
        ss_t_a_ss.formated_output(1000,"full");}
        drop(a_ss);
        let mut b_ss=MatrixFull::new([occ_vir,0],0.0);
        //b_ss:BS
        ss.iter_columns_full().for_each(|ss_i|{
            let z=ss_i.to_vec();
            let b_z=b_matvec(&z);
            b_ss.push_column(&b_z);
        });
        let mut ss_t_b_ss=MatrixFull::new([m,m],0.0);
        //ss_t_b_ss:projection matrix of b
        _dgemm_full(&ss,'T',&b_ss,'N',&mut ss_t_b_ss,1.0,0.0);
        if print_level>1{println!("B projection:");
        ss_t_b_ss.formated_output(1000,"full");}
        let ss_t_amb_ss=ss_t_a_ss.scaled_add(&ss_t_b_ss,-1.0).unwrap();
        let ss_t_apb_ss=ss_t_a_ss.scaled_add(&ss_t_b_ss,1.0).unwrap();
        if print_level>1{println!("AmB projection:");
        ss_t_amb_ss.formated_output(1000,"full");}
        let mut g=ss_t_amb_ss;
        _dpotrf(&mut g,'L');
        (0..m).cartesian_product(0..m).for_each(|(i,j)|{
            if i<j{
                g[[i,j]]=0.0;
            }
        });
        //g: A-B=GG^T,Cholesky Lower matrix from A-B
        if print_level>1{println!("ApB projection:");
        ss_t_apb_ss.formated_output(1000,"full");}
        let mut apb_g=MatrixFull::new([m,m],0.0);
        //apb_g:(A-B)G,where A-B is the subspace projection
        _dgemm_full(&ss_t_apb_ss,'N',&g,'N',&mut apb_g,1.0,0.0);
        let mut gt_apb_g=MatrixFull::new([m,m],0.0);
        //gt_apb_g:G^T(A-B)G,where A-B is the subspace projection.This is the matrix to dsyev!
        _dgemm_full(&g,'T',&apb_g,'N',&mut gt_apb_g,1.0,0.0);
        drop(apb_g);
        let (Some(ginv_xpy),omega2,_)=_dsyev(&gt_apb_g,'V')else{panic!("dsyev failure!")};
        //ginv_xpy:G^{-1}(X+Y),eigenvectors of G^T(A-B)G,denoted as u in some literature,omega2:square of desired excitation energies
        let mut ginv_xpy_positives=MatrixFull::new([m,0],0.0);
        let mut omega2_positives:Vec<f64>=Vec::new();
        if print_level>1{println!("omega square={:#?}",omega2);}
        omega2.iter().zip(ginv_xpy.iter_columns_full()).enumerate().for_each(|(n,(omega2_i,ginv_xpy))|if *omega2_i>0.0{
            if n<nroots && *omega2_i>0.0{
                omega2_positives.push(*omega2_i);
                ginv_xpy_positives.push_column(ginv_xpy);
            }
        });
        omega2_positives.iter().take(nroots).collect::<Vec<_>>();
    
        //Deleted solutions with negative omega^2 to increase robustness
        let m=g.size[0];
        //m:subspace dimensions
        let mut xpy=MatrixFull::new([m,nroots],0.0);
        _dgemm_full(&g,'N',&ginv_xpy_positives,'N',&mut xpy,1.0,0.0);
        if print_level>1{println!("Subspace xpy:");
        xpy.formated_output(1000,"full");}
        //xpy:X+Y, spanned in subspace
        let ginv=_dinverse(&g).expect("unsuccessful _dinverse");
        //ginv:G^{-1}
        let mut xmy=MatrixFull::new([m,nroots],0.0);
        //xmy:X-Y, spanned in subspace
        _dgemm_full(&ginv,'T',&ginv_xpy_positives,'N',&mut xmy,1.0,0.0);
        if print_level>1{println!("Subspace xmy:");
        xmy.formated_output(1000,"full");}
        //G^{-T}G{-1}(X+Y)=(A-B)^{-1}(X+Y)=omega(A-B)^{-1}(A-B)(X-Y)=omega(X-Y)
        let mut omega:Vec<f64>=Vec::new();
        (0..nroots).for_each(|j|{
            let omega_j=omega2_positives[j].powf(0.5);
            omega.push(omega_j);
            //constructing omega
            (0..m).for_each(|k|{
                xmy[[k,j]]=xmy[[k,j]]*omega_j.powf(0.5);
                xpy[[k,j]]=xpy[[k,j]]/omega_j.powf(0.5);
            });
            //Rescaling:to satisfy xmy*xpy=1
        });
        if print_level>1{println!("omega:{:#?}",omega);}
        let mut xmy_full=MatrixFull::new([occ_vir,nroots],0.0);
        let mut xpy_full=MatrixFull::new([occ_vir,nroots],0.0);
        _dgemm_full(&ss,'N',&xmy,'N',&mut xmy_full,1.0,0.0);
        _dgemm_full(&ss,'N',&xpy,'N',&mut xpy_full,1.0,0.0);
        //xmy_full,xpy_full:Real X+Y and X-Y spanned in MO-pair basis
        let mut axmy=MatrixFull::new([occ_vir,0],0.0);
        let mut bxmy=MatrixFull::new([occ_vir,0],0.0);
        xmy_full.iter_columns_full().for_each(|xmy_i|{
            let z=xmy_i.to_vec();
            let axmy_i=a_matvec(&z);
            axmy.push_column(&axmy_i);
            let bxmy_i=b_matvec(&z);
            bxmy.push_column(&bxmy_i);
        });
        let ambxmy=axmy.scaled_add(&bxmy,-1.0).unwrap();
        let mut axpy=MatrixFull::new([occ_vir,0],0.0);
        let mut bxpy=MatrixFull::new([occ_vir,0],0.0);
        xpy_full.iter_columns_full().for_each(|xpy_i|{
            let z=xpy_i.to_vec();
            let axpy_i=a_matvec(&z);
            axpy.push_column(&axpy_i);
            let bxpy_i=b_matvec(&z);
            bxpy.push_column(&bxpy_i);
        });
        let apbxpy=axpy.scaled_add(&bxpy,1.0).unwrap();
        //ambxmy,apbxpy:(A+B)(X+Y) and (A-B)(X-Y)
        //They are used to evaluate residues:(A+B)(X+Y)-omega(X-Y) and (A-B)(X-Y)-omega(X+Y)
        let mut left_residues=ambxmy;
        let mut right_residues=apbxpy;
        let mut eigenvalue_matrix=MatrixFull::new([nroots,nroots],0.0);
        (0..nroots).for_each(|i|eigenvalue_matrix[[i,i]]=omega[i]);
        //diagonal matrix of eigenvalues, to make it possible to obtain residues with LAPACK
        let mut omega_xpy=MatrixFull::new([occ_vir,nroots],0.0);
        _dgemm_full(&xpy_full,'N',&eigenvalue_matrix,'N',&mut omega_xpy,1.0,0.0);
        //omega_xpy: Omega(diagonal)(X+Y)
        let mut omega_xmy=MatrixFull::new([occ_vir,nroots],0.0);
        _dgemm_full(&xmy_full,'N',&eigenvalue_matrix,'N',&mut omega_xmy,1.0,0.0);
        //omega_xmy: Omega(diagonal)(X-Y)
        left_residues=left_residues.scaled_add(&omega_xpy,-1.0).unwrap();
        right_residues=right_residues.scaled_add(&omega_xmy,-1.0).unwrap();
        println!("Now is iteration #{},current progress:",iter_num);
        println!("Right residues:");
        right_residues.iter_columns_full().for_each(|vec|{
            println!("{}",vec.iter().map(|x|x.powf(2.0)).sum::<f64>().powf(0.5));
        });
        println!("Left residues:");
        left_residues.iter_columns_full().for_each(|vec|{
            println!("{}",vec.iter().map(|x|x.powf(2.0)).sum::<f64>().powf(0.5));
        });
        //R_left=(A-B)(X-Y)-Omega(X+Y)
        //R_right=(A+B)(X+Y)-Omega(X-Y)
        let mut converge=true;
        left_residues.iter_columns_full().for_each(|residue|{
            let norm=residue.iter().fold(0.0,|acc,val|acc+val.powf(2.0));
            if norm>1e-10{
                converge=false;
            }
        });
        if converge{
            right_residues.iter_columns_full().for_each(|residue|{
                let norm=residue.iter().fold(0.0,|acc,val|acc+val.powf(2.0));
                if norm>1e-10{
                    converge=false;
                }
            });
        }
        if converge{
            let mut x=xmy_full.scaled_add(&xpy_full,1.0).unwrap();
            let mut y=xpy_full.scaled_add(&xmy_full,-1.0).unwrap();
            x.self_multiple(0.5);
            y.self_multiple(0.5);
            x_solutions.append_column(&x);
            y_solutions.append_column(&y);
            eigenvalues=omega;
            break;
        }
        left_residues.iter_columns_full().enumerate().for_each(|(i,residue)|{
            let omega_m_diag=omega[i]-diag[i];
            let mut preconditioned=residue.iter().map(|v_k|v_k/omega_m_diag).collect();
            //preconditioned:preconditioned vector:
            //(X-Y)=(omega-diag)
            if print_level>1{println!("Un-orthogonalized to add:{:#?}",preconditioned);}
            ss.iter_columns_full().for_each(|v_k|{
                let subspace_vec=v_k.to_vec();
                let projection_product=dot_product(&subspace_vec,&preconditioned);
                if print_level>1{println!("projection={}",projection_product);}
                preconditioned=vector_scaled_add(&preconditioned,1.0,&subspace_vec,-projection_product);
            });
            let orthogonalized_norm=dot_product(&preconditioned,&preconditioned).powf(0.5);
            //if print_level>1{println!("Before normalization:{:#?},norm={}",preconditioned,orthogonalized_norm);}
            if orthogonalized_norm>1e-8{
                preconditioned=num_product(&preconditioned,1.0/orthogonalized_norm);
                ss.push_column(&preconditioned);
                println!("A new search vector has been added. Search space now has {} vectors",ss.size[1]);
            }
        });
        right_residues.iter_columns_full().enumerate().for_each(|(i,residue)|{
            let omega_m_diag=omega[i]-diag[i];
            let mut preconditioned=residue.iter().map(|v_k|v_k/omega_m_diag).collect();
            //preconditioned:preconditioned vector:
            //(X-Y)=(omega-diag)
            if print_level>1{println!("Un-orthogonalized to add:{:#?}",preconditioned);}
            ss.iter_columns_full().for_each(|v_k|{
                let subspace_vec=v_k.to_vec();
                let projection_product=dot_product(&subspace_vec,&preconditioned);
                if print_level>1{println!("projection={}",projection_product);}
                preconditioned=vector_scaled_add(&preconditioned,1.0,&subspace_vec,-projection_product);
            });
            let orthogonalized_norm=dot_product(&preconditioned,&preconditioned).powf(0.5);
            //if print_level>1{println!("Before normalization:{:#?},norm={}",preconditioned,orthogonalized_norm);}
            if orthogonalized_norm>1e-8{
                preconditioned=num_product(&preconditioned,1.0/orthogonalized_norm);
                ss.push_column(&preconditioned);
                println!("A new search vector has been added. Search space now has {} vectors",ss.size[1]);
            }
        });
        if print_level>1{println!("Search Space:");
        ss.formated_output(1000,"full");}
        if iter_num>20{
            break;
        }
    }
    let eigenvectors:Vec<_>=x_solutions.iter_columns_full().zip(y_solutions.iter_columns_full()).map(|(v1,v2)|{
        let mut vec=v1.to_vec();
        vec.extend(v2);
        vec
    }).collect();
    let mut eigenpairs:Vec<_>=eigenvalues.iter().zip(eigenvectors.iter()).map(|(value,vector)|(*value,vector.clone())).collect();
    eigenpairs.sort_by(|a,b|a.0.partial_cmp(&b.0).unwrap());
    eigenpairs
}
pub fn dot_product(vec1:&Vec<f64>,vec2:&Vec<f64>)->f64{
    vec1.iter().zip(vec2.iter()).fold(0.0,|acc,(x1,x2)|acc+x1*x2)
}
pub fn num_product(vec1:&Vec<f64>,num:f64)->Vec<f64>{
    vec1.iter().map(|x|x*num).collect()
}
pub fn vector_scaled_add(vec1:&Vec<f64>,scale1:f64,vec2:&Vec<f64>,scale2:f64)->Vec<f64>{
    vec1.iter().zip(vec2.iter()).map(|(x1,x2)|x1*scale1+x2*scale2).collect()
}