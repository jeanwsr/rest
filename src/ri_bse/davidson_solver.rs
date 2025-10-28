use rest_tensors::matrix::MatrixFull;
use rest_tensors::matrix::matrix_blas_lapack::{_dgeev, _dgemv};
use std::time::Instant;
use crate::ri_bse::matvec;
use crate::scf_io::{SCF, SCFType};
use crate::ri_bse::get_occupation_parameters;
use std::cmp;
use crate::ri_bse;
use itertools::Itertools;

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
pub fn subspace_projection(scf_data:&SCF,inverse_dielectric:&MatrixFull<f64>,subspace:&Vec<Vec<f64>>)->MatrixFull<f64>{
    let m=subspace.len();
    let mut projection_matrix=MatrixFull::from_vec([m,m],vec![0.0;m*m]).expect("failure when initializing zero projection matrix");
    let mut matrix_checks=MatrixFull::from_vec([m,m],vec![0.0;m*m]).expect("failure when initializing zero projection matrix");
    subspace.iter().enumerate().for_each(|(j,vj)|{
        let avj=matvec::a_block_matvec(scf_data,inverse_dielectric,vj);
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
pub fn compute_residues(scf_data:&SCF,ritz_pairs:&Vec<(f64,Vec<f64>)>,inverse_dielectric:&MatrixFull<f64>)->Vec<(f64,Vec<f64>)>{
    ritz_pairs.iter().take(10).map(|(lambda,ritz_vec)|{
        let axi=matvec::a_block_matvec(scf_data,inverse_dielectric,ritz_vec);
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
pub fn get_diag_elements(matrix:&MatrixFull<f64>)->Vec<f64>{
    let n=matrix.size[0];
    (0..n).map(|i|matrix[[i,i]]).collect()
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
pub fn iteration(scf_data:&SCF,subspace:&mut Vec<Vec<f64>>,inverse_dielectric:&MatrixFull<f64>)->Vec<(f64,Vec<f64>)>{
    let (start_mo,num_state,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let mut projection_matrix:MatrixFull<f64>=subspace_projection(scf_data,inverse_dielectric,subspace);
    //4->8->12->16->20--->12->16->20--->12
    let mut ritz_pairs=compute_ritz_pairs(&projection_matrix,subspace,false);
    let mut residues=compute_residues(scf_data,&ritz_pairs,inverse_dielectric);
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
        projection_matrix=subspace_projection(scf_data,inverse_dielectric,subspace);
        let projection_time=start1.elapsed();
        if scf_data.mol.ctrl.print_level>1{
            println!("projection time={:?}",projection_time);
        }
        ritz_pairs=compute_ritz_pairs(&projection_matrix,subspace,for_restart);
        let start2=Instant::now();
        residues=compute_residues(scf_data,&ritz_pairs,inverse_dielectric);
        let compute_residue_time=start2.elapsed();
        if scf_data.mol.ctrl.print_level>1{
            println!("compute residue time={:?}",compute_residue_time);
        }
        println!("First residue={}, Last residue={}",residues[0].0,residues[compute_pairs-1].0);
    }
    //(0..compute_pairs).for_each(|m|check_eigen(scf_data,inverse_dielectric,&ritz_pairs[m]));
    ritz_pairs.into_iter().take(compute_pairs).collect()
}
pub fn explicit_restart(ritz_space:&Vec<Vec<f64>>,compute_pairs:usize,restart_dimensions:usize,print_level:usize)->Vec<Vec<f64>>{
    println!("explicit restart!");
    let mut subspace:Vec<Vec<f64>>=Vec::new();
    schmidt(&mut subspace,&ritz_space.iter().take(restart_dimensions).cloned().collect(),print_level);
    subspace
}
pub fn ritz_approx(scf_data:&SCF,ritz_vec:&Vec<f64>,inverse_dielectric:&MatrixFull<f64>)->f64{
    let av=matvec::a_block_matvec(scf_data,inverse_dielectric,ritz_vec);
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
pub fn check_eigen(scf_data:&SCF,inverse_dielectric:&MatrixFull<f64>,ritz_pair:&(f64,Vec<f64>)){
    let eigenvec=ritz_pair.1.clone();
    let result=matvec::a_block_matvec(scf_data,inverse_dielectric,&eigenvec);
    let error_norm=(0..eigenvec.len()).fold(0.0,|acc,i|(result[i]-(ritz_pair.0*eigenvec[i])).powf(2.0));
    println!("Ritz_value={}, Error_norm={}",ritz_pair.0,error_norm);
}