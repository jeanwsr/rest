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

pub fn form_initial_space(scf_data:&SCF)->Vec<Vec<f64>>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let len=occ_size*vir_size;
    let mut subspace:Vec<Vec<f64>>=if occ_size>1{vec![vec![0.0;len];4]}else{vec![vec![0.0;len];2]};
    if occ_size>1{
        (0..2).cartesian_product(0..2).enumerate().for_each(|(n,(i,a))|{
            let index=occ_size-1-i+(occ_size*a);
            subspace[n][index]=1.0;
        });
    }else{
        (0..2).enumerate().for_each(|(n,a)|{
            let index=a;
            subspace[n][index]=1.0;
        });
    }
    subspace
}
pub fn subspace_projection(scf_data:&SCF,ri_oo_tilde:&MatrixFull<f64>,subspace:&Vec<Vec<f64>>)->MatrixFull<f64>{
    let m=subspace.len();
    let mut projection_matrix=MatrixFull::from_vec([m,m],vec![0.0;m*m]).expect("failure when initializing zero projection matrix");
    let mut matrix_checks=MatrixFull::from_vec([m,m],vec![0.0;m*m]).expect("failure when initializing zero projection matrix");
    subspace.iter().enumerate().for_each(|(j,vj)|{
        let avj=matvec::a_block_matvec(scf_data,ri_oo_tilde,vj);
        subspace.iter().enumerate().for_each(|(i,vi)|{
            if matrix_checks[[i,j]]==0.0{
                let viavj=vecvec(&avj,vi);
                projection_matrix[[i,j]]=viavj;
                projection_matrix[[j,i]]=viavj;
                matrix_checks[[i,j]]=1.0;
                matrix_checks[[j,i]]=1.0;
            }
        })
    });
    projection_matrix
}
pub fn compute_ritz_pairs(projection_matrix:&MatrixFull<f64>,subspace:&Vec<Vec<f64>>,for_restart:bool)->Vec<(f64,Vec<f64>)>{
    let (_, wr_1, _, _, vr_1, _) = _dgeev(projection_matrix, 'N', 'V');
    println!("projection matrix size:{}",projection_matrix.size[0]);
    let eigenpairs = zip_and_sort(&wr_1, &vr_1);
    let ritz_pair_numbers=if for_restart{12}else{10};
    //ritz pairs have 2 uses: 1)assess the quality of current subspace, i.e. the current results;2)provide vectors to be preconditioned
    let ritz_pairs:Vec<(f64,Vec<f64>)>=eigenpairs.iter().take(ritz_pair_numbers).map(|(lambda,theta)|(*lambda,ritz_vector(subspace,theta))).collect();
    ritz_pairs
}
pub fn compute_residues(scf_data:&SCF,ritz_pairs:&Vec<(f64,Vec<f64>)>,ri_oo_tilde:&MatrixFull<f64>)->Vec<(f64,Vec<f64>)>{
    let compute_pairs=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap().davidson_target_excitations;
    ritz_pairs.iter().take(compute_pairs).map(|(lambda,ritz_vec)|{
        let axi=matvec::a_block_matvec(scf_data,ri_oo_tilde,ritz_vec);
        let lambda_x:Vec<f64>=ritz_vec.iter().map(|x_i|*x_i*lambda).collect();
        let residue_vec=axi.iter().zip(lambda_x.iter()).map(|(axik,lambda_xik)|axik-lambda_xik).collect();
        let residue_norm=vecvec(&residue_vec,&residue_vec).sqrt();
        (residue_norm,residue_vec)
    }).collect()
}
pub fn ritz_vector(subspace:&Vec<Vec<f64>>,projected_solution:&Vec<f64>)->Vec<f64>{
    let vector:Vec<f64>=vec![0.0;subspace[0].len()];
    vector.iter().enumerate().map(|(i,_)|projected_solution.iter().zip(subspace.iter()).map(|(theta_k,v_k)|theta_k*v_k[i]).sum()).collect()
}
pub fn vecvec(vec1:&Vec<f64>,vec2:&Vec<f64>)->f64{
    if vec1.len()!=vec2.len(){
        panic!("vector lengths don't match!")
    }
    vec1.iter().zip(vec2.iter()).map(|(x,y)|x*y).sum()
}
pub fn preconditioner(diag:&Vec<f64>,ritz_pairs:&Vec<(f64,Vec<f64>)>,residues:&Vec<(f64,Vec<f64>)>,print_level:usize)->Vec<Vec<f64>>{
    let mut ritz_pair_push:Vec<Vec<f64>>=Vec::new();
    let mut count=0;
    ritz_pairs.iter().zip(residues.iter()).for_each(
        |((lambda_i,x_i),(residue_val_i,residue_vec_i))|
        {if *residue_val_i>1e-7 && count<4{
            ritz_pair_push.push(residue_vec_i.iter().zip(diag.iter()).map(|(r_i_j,diag_j)|r_i_j*((diag_j-lambda_i).powf(-1.0))).collect());
            count+=1;
        }else if *residue_val_i<1e-7{
            if print_level>1{
                println!("skipping converged pair");
            }
        }
        });
        println!("this round of preconditioning will add {} dimensions to the search space",ritz_pair_push.len());
    ritz_pair_push
}
pub fn vmv(vec1:&Vec<f64>,vec2:&Vec<f64>)->Vec<f64>{
    vec1.iter().zip(vec2.iter()).map(|(v1,v2)|*v1-*v2).collect()
}
pub fn numvec(vec:&Vec<f64>,num:f64)->Vec<f64>{
    vec.iter().map(|x|*x * num).collect()
}
pub fn schmidt(subspace:&mut Vec<Vec<f64>>,ritz_vectors:&Vec<Vec<f64>>,print_level:usize){
    let mut count=0;
    let mut trivialty=0;
    ritz_vectors.iter().for_each(|ritzvec|{
        let mut addvec=ritzvec.clone();
        subspace.iter().for_each(|orthogvec|{
            let projection=vecvec(&ritzvec,&orthogvec);
            let modulus_square=vecvec(&orthogvec,&orthogvec);
            let minusvec=numvec(&orthogvec,projection/modulus_square);
            addvec=vmv(&addvec,&minusvec);
        });
        let norm = vecvec(&addvec, &addvec).sqrt();
        if norm > 1e-10 {
            subspace.push(numvec(&addvec, 1.0/norm));
            count+=1;
        }else{
            trivialty+=1;
        } // 否则丢弃零向量
    });
    if print_level>1{
        println!("{} vectors have been added to the search space,there are {} trivialties",count,trivialty);
    }
}
pub fn orthog_check(subspace:&Vec<Vec<f64>>)->MatrixFull<f64>{
    let m=subspace.len();
    let mut projection_matrix=MatrixFull::from_vec([m,m],vec![0.0;m*m]).expect("failure when initializing zero projection matrix");
    subspace.iter().enumerate().for_each(|(j,vj)|{
        subspace.iter().enumerate().for_each(|(i,vi)|projection_matrix[[i,j]]=vecvec(vj,vi))
    });
    projection_matrix
}
pub fn iteration(scf_data:&SCF,subspace:&mut Vec<Vec<f64>>,ri_oo_tilde:&MatrixFull<f64>)->Vec<(f64,Vec<f64>)>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let mut projection_matrix:MatrixFull<f64>=subspace_projection(scf_data,ri_oo_tilde,subspace);
    let mut ritz_pairs=compute_ritz_pairs(&projection_matrix,subspace,false);
    let mut residues=compute_residues(scf_data,&ritz_pairs,ri_oo_tilde);
    let mut rounds=0;
    //New Control Parameters:1)davidson_converge_threshold;2)davidson_target_excitations;3)maximum dimensions;4)restart dimensions
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let converge_threshold=qp_ctrl.davidson_converge_threshold;
    let mut compute_pairs=qp_ctrl.davidson_target_excitations;
    let max_subspace=qp_ctrl.davidson_maximum_subspace_size;
    let restart_dimensions=qp_ctrl.davidson_restart_dimensions;
    println!("Paramaters for this Davidson Algorithm:\n  Dimension of matrix={};\n  Converge threshold={};\n  Target excitations={};\n  Maximum Subspace Dimensions={}\n  Restart Dimensions={}",occ_size*vir_size,converge_threshold,compute_pairs,max_subspace,restart_dimensions);
    if compute_pairs>occ_size*vir_size{
        println!("Warning: You asked for {} eigenpairs computed from the Davidson Eigensolver, \nbut there are only {} eigenpairs in this system.\nOnly {} eigenpairs will be computed instead.",compute_pairs,occ_size*vir_size,cmp::max(occ_size*vir_size-2,1));
        compute_pairs=cmp::min(occ_size*vir_size-2,1);
    }
    println!("Davidson initial guess:");
    println!("First residue={}, Last residue={}",residues[0].0,residues[compute_pairs-1].0);
    loop{
        let mut converge=true;
        rounds+=1;
        println!("Now Round #{},",rounds);
        residues.iter().enumerate().for_each(|(i,(residue_val,_))|{
            //println!("residue={},ritz={}",residue_val,ritz_pairs[i].0);
            if *residue_val>converge_threshold && i<compute_pairs{
                println!("{}th Pair Has not converged yet.",i);
                converge=false;
            }
            if *residue_val<converge_threshold&& i<compute_pairs{
                println!("{}th Pair Has converged.",i);
            }
        });
        if converge{
            break
        }
        let diags=matvec::diagonal_elements_contribution(scf_data,&vec![1.0;occ_size*vir_size]);
        let mut for_restart=false;
        if subspace.len()<max_subspace{
            let ritz_space=preconditioner(&diags,&ritz_pairs,&residues,scf_data.mol.ctrl.print_level);
            schmidt(subspace,&ritz_space,scf_data.mol.ctrl.print_level);
        }
        else {
            let ritz_space=ritz_pairs.iter().map(|(ritz_val,ritz_vec)|ritz_vec.clone()).collect();
            *subspace = explicit_restart(&ritz_space,compute_pairs,restart_dimensions,scf_data.mol.ctrl.print_level); 
            for_restart=true;
        }
        let start1=Instant::now();
        projection_matrix=subspace_projection(scf_data,ri_oo_tilde,subspace);
        let projection_time=start1.elapsed();
        if scf_data.mol.ctrl.print_level>1{
            println!("projection time={:?}",projection_time);
        }
        ritz_pairs=compute_ritz_pairs(&projection_matrix,subspace,for_restart);
        let start2=Instant::now();
        residues=compute_residues(scf_data,&ritz_pairs,ri_oo_tilde);
        let compute_residue_time=start2.elapsed();
        if scf_data.mol.ctrl.print_level>1{
            println!("compute residue time={:?}",compute_residue_time);
        }
        println!("First residue={}, Last residue={}",residues[0].0,residues[compute_pairs-1].0);
    }
    //(0..compute_pairs).for_each(|m|check_eigen(scf_data,ri_oo_tilde,&ritz_pairs[m]));
    ritz_pairs.into_iter().take(compute_pairs).collect()
}
pub fn explicit_restart(ritz_space:&Vec<Vec<f64>>,compute_pairs:usize,restart_dimensions:usize,print_level:usize)->Vec<Vec<f64>>{
    println!("explicit restart!");
    let mut subspace:Vec<Vec<f64>>=Vec::new();
    schmidt(&mut subspace,&ritz_space.iter().take(restart_dimensions).cloned().collect(),print_level);
    subspace
}
pub fn ritz_approx(scf_data:&SCF,ritz_vec:&Vec<f64>,ri_oo_tilde:&MatrixFull<f64>)->f64{
    let av=matvec::a_block_matvec(scf_data,ri_oo_tilde,ritz_vec);
    let vav=vecvec(&av,ritz_vec);
    let vv=vecvec(ritz_vec,ritz_vec);
    vav/vv
}
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
pub fn check_eigen(scf_data:&SCF,ri_oo_tilde:&MatrixFull<f64>,ritz_pair:&(f64,Vec<f64>)){
    let eigenvec=ritz_pair.1.clone();
    let result=matvec::a_block_matvec(scf_data,ri_oo_tilde,&eigenvec);
    let error_norm=(0..eigenvec.len()).fold(0.0,|acc,i|(result[i]-(ritz_pair.0*eigenvec[i])).powf(2.0));
    println!("Ritz_value={}, Error_norm={}",ritz_pair.0,error_norm);
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
            let orthogonalizrd_norm=dot_product(&preconditioned,&preconditioned).powf(0.5);
            if print_level>1{println!("Before normalization:{:#?},norm={}",preconditioned,orthogonalizrd_norm);}
            if orthogonalizrd_norm>1e-8{
                preconditioned=num_product(&preconditioned,1.0/orthogonalizrd_norm);
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
            let orthogonalizrd_norm=dot_product(&preconditioned,&preconditioned).powf(0.5);
            if print_level>1{println!("Before normalization:{:#?},norm={}",preconditioned,orthogonalizrd_norm);}
            if orthogonalizrd_norm>1e-8{
                preconditioned=num_product(&preconditioned,1.0/orthogonalizrd_norm);
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