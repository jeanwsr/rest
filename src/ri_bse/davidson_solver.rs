use rest_tensors::matrix::{MatrixFull,MathMatrix,matrixupper::MatrixUpper};
use rest_tensors::matrix::matrix_blas_lapack::{_dgeev,_dgemm_full, _dgemv,_dpotrf,_dsyev,_dinverse};
use std::time::Instant;
use crate::ri_bse::matvec;
use crate::scf_io::{SCF, SCFType};
use crate::ri_bse::get_occupation_parameters;
use std::cmp;
use crate::ri_bse;
use rayon::prelude::*;
use rayon::iter::ParallelBridge;
use itertools::Itertools;

/// Davidson solver configuration, decoupled from QuasiParticle/TDDFTParameters.
#[derive(Debug, Clone)]
pub struct DavidsonConfig {
    pub max_subspace: usize,
    pub add_dim: usize,
    pub restart_dim: usize,
    pub max_iter: usize,
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
pub fn tda_davidson_solver<F1>(print_level:usize,a_matvec:F1,nroots:usize,diag:&Vec<f64>,initial_guess:MatrixFull<f64>,config:&DavidsonConfig)->Vec<(f64,Vec<f64>)>
where F1:Fn(&Vec<f64>)->Vec<f64>+Send+Sync{
    let mut ss=initial_guess;
    //ss:search space S, where each column in S is a search vector
    let occ_vir=diag.len();
    //occ_vir:length of dimension of a,b as well as x,y. In TDDFT/BSE it is occ_size*vir_size
    let mut x_solutions=MatrixFull::new([occ_vir,0],0.0);
    let mut eigenvalues:Vec<f64>=Vec::new();
    let mut iter_num=0;
    loop{
        let start=Instant::now();
        let restart=if ss.size[1]>config.max_subspace-config.add_dim{true}else{false};
        iter_num+=1;
        let m=ss.size[1];
        let mut a_ss=MatrixFull::new([occ_vir,0],0.0);
        //a_ss:AS
        let mut results: Vec<(usize,Vec<f64>)> = ss.iter_columns_full()
            .enumerate().par_bridge()  // 将普通迭代器转换为并行迭代器
            .map(|(i,ss_i)| {
                let z = ss_i.to_vec();
                (i,a_matvec(&z))  // 直接返回结果向量
            })
            .collect();
        // 按顺序推入结果
        results.sort_by_key(|(i, _)| *i);
        for result in results {
            a_ss.push_column(&result.1);
        }
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
        let collect_sol_num=if restart{config.restart_dim}else{cmp::min(ss.size[1],config.add_dim)};
        omega.iter().zip(x_proj.iter_columns_full()).enumerate().for_each(|(n,(omega_i,x_i))|
            if n<collect_sol_num{
                omega_nroots.push(*omega_i);
                x_proj_nroots.push_column(x_i);
            });
        let m=ss_t_a_ss.size[0];
        //m:subspace dimensions
        if print_level>1{println!("omega:{:#?}",omega);}
        let mut x_full=MatrixFull::new([occ_vir,collect_sol_num],0.0);
        _dgemm_full(&ss,'N',&x_proj_nroots,'N',&mut x_full,1.0,0.0);
        //xmy_full,xpy_full:Real X+Y and X-Y spanned in MO-pair basis
        let mut ax_full=MatrixFull::new([occ_vir,0],0.0);
        let mut results: Vec<(usize,Vec<f64>)> = x_full.iter_columns_full()
            .enumerate().par_bridge()  // 将普通迭代器转换为并行迭代器
            .map(|(i,x_i)| {
                let z = x_i.to_vec();
                (i,a_matvec(&z))  // 直接返回结果向量
            })
            .collect();
        // 按顺序推入结果
        results.sort_by_key(|(i, _)| *i);
        for result in results {
            ax_full.push_column(&result.1);
        }
        let mut residues=ax_full;
        let mut eigenvalue_matrix=MatrixFull::new([collect_sol_num,collect_sol_num],0.0);
        (0..collect_sol_num).for_each(|i|eigenvalue_matrix[[i,i]]=omega[i]);
        //diagonal matrix of eigenvalues, to make it possible to obtain residues with LAPACK
        let mut omega_x_full=MatrixFull::new([occ_vir,collect_sol_num],0.0);
        _dgemm_full(&x_full,'N',&eigenvalue_matrix,'N',&mut omega_x_full,1.0,0.0);
        //omega_x: Omega(diagonal)X
        let mut omega_x=MatrixFull::new([occ_vir,collect_sol_num],0.0);
        _dgemm_full(&x_full,'N',&eigenvalue_matrix,'N',&mut omega_x_full,1.0,0.0);
        residues=residues.scaled_add(&omega_x_full,-1.0).unwrap();println!("Now is iteration #{},current progress:",iter_num);
        if print_level>1{println!("Residues:");
        residues.iter_columns_full().enumerate().for_each(|(n,vec)|{
            if n<nroots{
                println!("{}",vec.iter().map(|x|x.powf(2.0)).sum::<f64>().powf(0.5));
            }
        });}
        let mut converge=true;
        let mut converge_pairs=0;
        let n_residues = residues.size[1];
        residues.iter_columns_full().enumerate().for_each(|(n,residue_i)|{
            if n < nroots.min(n_residues) {
                let norm=residue_i.iter().fold(0.0,|acc,val|acc+val.powf(2.0));
                if norm>1e-10{
                    converge=false;
                }else{
                    converge_pairs+=1;
                }
            }
        });
        if n_residues < nroots {
            converge = false;
            if print_level > 0 {
                println!("  Warning: only {} trial vectors for {} roots", n_residues, nroots);
            }
        }
        if converge{
            x_full.iter_columns_full().enumerate().for_each(|(n,vec)|if n<nroots{x_solutions.push_column(vec)});
            eigenvalues=omega[..nroots].to_vec();
            println!("Davidson Solver has converged. This final round took {:?}",start.elapsed());
            break;
        }
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
                if print_level>1{println!("A new search vector has been added. Search space now has {} vectors",ss.size[1]);}
            }
        });}else{
            println!("Explicit Restart");
            ss=MatrixFull::new([occ_vir,0],0.0);
            x_full.iter_columns_full().take(config.restart_dim).for_each(|x_i|ss.push_column(x_i));
            residues.iter_columns_full().enumerate().for_each(|(i,residue_i)|{
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
                    if print_level>1{println!("A new search vector has been added. Search space now has {} vectors",ss.size[1]);}
                }
            });
        }
        if print_level>2{println!("Search Space:");
        ss.formated_output(1000,"full");}
        if iter_num>config.max_iter{
            break;
        }
        println!("Converged Pairs:{} out of the {} desired solutions",converge_pairs,nroots);
        println!("This iteration took {:?}",start.elapsed());
    };
    let eigenvectors:Vec<_>=x_solutions.iter_columns_full().map(|v1|{
        let mut vec=v1.to_vec();
        vec
    }).collect();
    let mut eigenpairs:Vec<_>=eigenvalues.iter().zip(eigenvectors.iter()).map(|(value,vector)|(*value,vector.clone())).collect();
    eigenpairs.sort_by(|a,b|a.0.partial_cmp(&b.0).unwrap());
    eigenpairs
}



pub fn lr_davidson_solver<F1,F2>(print_level:usize,a_matvec:F1,b_matvec:F2,nroots:usize,diag:&Vec<f64>,initial_guess:MatrixFull<f64>,config:&DavidsonConfig)->Vec<(f64,Vec<f64>)>
where F1:Fn(&Vec<f64>)->Vec<f64>+ Send + Sync,F2:Fn(&Vec<f64>)->Vec<f64>+ Send + Sync{
    let mut ss=initial_guess;
    //ss:search space S, where each column in S is a search vector
    let occ_vir=diag.len();
    //occ_vir:length of dimension of a,b as well as x,y. In TDDFT/BSE it is occ_size*vir_size
    let mut x_solutions=MatrixFull::new([occ_vir,0],0.0);
    let mut y_solutions=MatrixFull::new([occ_vir,0],0.0);
    let mut eigenvalues:Vec<f64>=Vec::new();
    let mut iter_num=0;
    loop{
        let start=Instant::now();
        iter_num+=1;
        let m=ss.size[1];
        let mut a_ss=MatrixFull::new([occ_vir,0],0.0);
        //a_ss:AS
        let mut results: Vec<(usize,Vec<f64>)> = ss.iter_columns_full()
            .enumerate().par_bridge()  // 将普通迭代器转换为并行迭代器
            .map(|(i,ss_i)| {
                let z = ss_i.to_vec();
                (i,a_matvec(&z))  // 直接返回结果向量
            })
            .collect();
        // 按顺序推入结果
        results.sort_by_key(|(i, _)| *i);
        for result in results {
            a_ss.push_column(&result.1);
        }
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
        let mut results: Vec<(usize,Vec<f64>)> = ss.iter_columns_full()
            .enumerate().par_bridge()  // 将普通迭代器转换为并行迭代器
            .map(|(i,ss_i)| {
                let z = ss_i.to_vec();
                (i,b_matvec(&z))  // 直接返回结果向量
            })
            .collect();
        // 按顺序推入结果
        results.sort_by_key(|(i, _)| *i);
        for result in results {
            b_ss.push_column(&result.1);
        }
        let mut ss_t_b_ss=MatrixFull::new([m,m],0.0);
        //ss_t_b_ss:projection matrix of b
        _dgemm_full(&ss,'T',&b_ss,'N',&mut ss_t_b_ss,1.0,0.0);
        if print_level>1{println!("B projection:");
        ss_t_b_ss.formated_output(1000,"full");}
        let ss_t_amb_ss=ss_t_a_ss.scaled_add(&ss_t_b_ss,-1.0).unwrap();
        let ss_t_apb_ss=ss_t_a_ss.scaled_add(&ss_t_b_ss,1.0).unwrap();
        if print_level>1{println!("AmB projection:");
        ss_t_amb_ss.formated_output(1000,"full");}
        let mut g=ss_t_amb_ss.clone();
        // Try Cholesky decomposition; catch panic if not positive definite
        let cholesky_ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut g_chol = ss_t_amb_ss.clone();
            _dpotrf(&mut g_chol, 'L');
            (0..m).cartesian_product(0..m).for_each(|(i,j)|{
                if i<j{ g_chol[[i,j]] = 0.0; }
            });
            g = g_chol;
        }));
        if cholesky_ok.is_err() {
            if print_level > 0 {
                println!("  Cholesky failed (A-B not positive-definite): this can happen with hybrid functionals.");
                println!("  Falling back to TDA approximation for this iteration.");
            }
        }
        let cholesky_result: Option<(MatrixFull<f64>, MatrixFull<f64>, Vec<f64>)> = if cholesky_ok.is_ok() {
            // Cholesky succeeded: use standard approach
            // (A-B) = GG^T, solve G^T(A+B)G Z = ω²Z
            if print_level>1{println!("ApB projection:");
            ss_t_apb_ss.formated_output(1000,"full");}
            let mut apb_g=MatrixFull::new([m,m],0.0);
            _dgemm_full(&ss_t_apb_ss,'N',&g,'N',&mut apb_g,1.0,0.0);
            let mut gt_apb_g=MatrixFull::new([m,m],0.0);
            _dgemm_full(&g,'T',&apb_g,'N',&mut gt_apb_g,1.0,0.0);
            drop(apb_g);
            let (Some(ginv_xpy), omega2, _) = _dsyev(&gt_apb_g, 'V') else { panic!("dsyev failure!") };

            let mut ginv_xpy_pos = MatrixFull::new([m,0], 0.0);
            let mut omega2_pos: Vec<f64> = Vec::new();
            for (n, (w2, col)) in omega2.iter().zip(ginv_xpy.iter_columns_full()).enumerate() {
                if *w2 > 0.0 && omega2_pos.len() < nroots {
                    omega2_pos.push(*w2);
                    ginv_xpy_pos.push_column(col);
                }
            }
            if omega2_pos.is_empty() {
                // No positive eigenvalues; fall through to dgeev approach
                None
            } else {
                let m_sub = g.size[0];
                let mut xpy_sub = MatrixFull::new([m_sub, nroots], 0.0);
                _dgemm_full(&g, 'N', &ginv_xpy_pos, 'N', &mut xpy_sub, 1.0, 0.0);
                let ginv = _dinverse(&g).expect("_dinverse");
                let mut xmy_sub = MatrixFull::new([m_sub, nroots], 0.0);
                _dgemm_full(&ginv, 'T', &ginv_xpy_pos, 'N', &mut xmy_sub, 1.0, 0.0);
                let omega_vec: Vec<f64> = omega2_pos.iter().map(|w| w.sqrt()).collect();
                Some((xpy_sub, xmy_sub, omega_vec))
            }
        } else {
            if print_level > 0 {
                println!("  Cholesky failed (A-B not positive-definite), switching to direct subspace solver");
            }
            None
        };

        let (mut xpy, mut xmy, mut omega) = match cholesky_result {
            Some((xp, xm, ow)) => (xp, xm, ow),
            None => {
                // Fallback: solve [A B; -B -A] on the projected subspace directly
                // Build 2m × 2m matrix H = [A_proj, B_proj; -B_proj, -A_proj]
                let mut h_full = MatrixFull::new([2*m, 2*m], 0.0);
                // Upper-left: A_proj = ss_t_a_ss
                for i in 0..m { for j in 0..m { h_full[[i, j]] = ss_t_a_ss[[i, j]]; }}
                // Upper-right: B_proj = ss_t_b_ss
                for i in 0..m { for j in 0..m { h_full[[i, m + j]] = ss_t_b_ss[[i, j]]; }}
                // Lower-left: -B_proj
                for i in 0..m { for j in 0..m { h_full[[m + i, j]] = -ss_t_b_ss[[i, j]]; }}
                // Lower-right: -A_proj
                for i in 0..m { for j in 0..m { h_full[[m + i, m + j]] = -ss_t_a_ss[[i, j]]; }}

                let (_, wr, wi, _vl, vr, _info) = _dgeev(&h_full, 'N', 'V');

                // Collect positive-real eigenvalues and corresponding right eigenvectors
                let mut pairs: Vec<(f64, Vec<f64>)> = wr.iter().zip(wi.iter().zip(vr.iter_columns_full()))
                    .filter(|(wr_i, (wi_i, _))| **wr_i > 1e-4 && wi_i.abs() < 1e-6)
                    .map(|(wr_i, (_, v))| (*wr_i, v.to_vec()))
                    .collect();
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                pairs.truncate(nroots);

                if pairs.is_empty() {
                    println!("  WARNING: No positive eigenvalues found in subspace.");
                    // Return empty results
                    return vec![];
                }

                let n_found = pairs.len();
                let mut xp = MatrixFull::new([m, n_found], 0.0);
                let mut xm = MatrixFull::new([m, n_found], 0.0);
                let mut ow: Vec<f64> = Vec::new();

                for (k, (val, vec)) in pairs.iter().enumerate() {
                    ow.push(*val);
                    // vec[0..m] = X, vec[m..2m] = Y from the dgeev eigenvector [X; Y]
                    for i in 0..m {
                        let vx = vec[i];           // X component
                        let vy = vec[m + i];        // Y component
                        xp[[i, k]] = vx + vy;       // X+Y
                        xm[[i, k]] = vx - vy;       // X-Y
                    }
                    // Standard rescaling: xmy *= sqrt(ω), xpy /= sqrt(ω)
                    let omega_sqrt = val.sqrt();
                    for i in 0..m {
                        xp[[i, k]] /= omega_sqrt;
                        xm[[i, k]] *= omega_sqrt;
                    }
                }
                (xp, xm, ow)
            }
        };
        if print_level>1{println!("Subspace xmy:");
        xmy.formated_output(1000,"full");}
        // Rescale to satisfy xmy·xpy = 1 (half-weight normalization)
        let n_omega = omega.len();
        for j in 0..n_omega {
            let eig = omega[j];
            for i in 0..m {
                xmy[[i,j]] *= eig.sqrt();
                xpy[[i,j]] /= eig.sqrt();
            }
        }
        let n_omega = omega.len();
        if n_omega == 0 {
            if print_level > 0 { println!("  No TDDFT roots found in subspace."); }
            break;
        }
        if print_level>1{println!("omega:{:#?}",omega);}
        let mut xmy_full=MatrixFull::new([occ_vir,n_omega],0.0);
        let mut xpy_full=MatrixFull::new([occ_vir,n_omega],0.0);
        _dgemm_full(&ss,'N',&xmy,'N',&mut xmy_full,1.0,0.0);
        _dgemm_full(&ss,'N',&xpy,'N',&mut xpy_full,1.0,0.0);
        //xmy_full,xpy_full:Real X+Y and X-Y spanned in MO-pair basis
        let mut axmy=MatrixFull::new([occ_vir,0],0.0);
        let mut bxmy=MatrixFull::new([occ_vir,0],0.0);
        let mut results: Vec<(usize,Vec<f64>,Vec<f64>)> = xmy_full.iter_columns_full()
            .enumerate().par_bridge()
            .map(|(i,xmy_i)| {
                let z=xmy_i.to_vec();
                (i,a_matvec(&z),b_matvec(&z))
            })
            .collect();
        results.sort_by_key(|(i, _,_)| *i);
        for result in results {
            axmy.push_column(&result.1);
            bxmy.push_column(&result.2);
        }
        let ambxmy=axmy.scaled_add(&bxmy,-1.0).unwrap();
        let mut axpy=MatrixFull::new([occ_vir,0],0.0);
        let mut bxpy=MatrixFull::new([occ_vir,0],0.0);
        let mut results: Vec<(usize,Vec<f64>,Vec<f64>)> = xpy_full.iter_columns_full()
            .enumerate().par_bridge()
            .map(|(i,xpy_i)| {
                let z=xpy_i.to_vec();
                (i,a_matvec(&z),b_matvec(&z))
            })
            .collect();
        results.sort_by_key(|(i, _,_)| *i);
        for result in results {
            axpy.push_column(&result.1);
            bxpy.push_column(&result.2);
        }
        let apbxpy=axpy.scaled_add(&bxpy,1.0).unwrap();
        let mut left_residues=ambxmy;
        let mut right_residues=apbxpy;
        let mut eigenvalue_matrix=MatrixFull::new([n_omega,n_omega],0.0);
        (0..n_omega).for_each(|i|eigenvalue_matrix[[i,i]]=omega[i]);
        let mut omega_xpy=MatrixFull::new([occ_vir,n_omega],0.0);
        _dgemm_full(&xpy_full,'N',&eigenvalue_matrix,'N',&mut omega_xpy,1.0,0.0);
        let mut omega_xmy=MatrixFull::new([occ_vir,n_omega],0.0);
        _dgemm_full(&xmy_full,'N',&eigenvalue_matrix,'N',&mut omega_xmy,1.0,0.0);
        //omega_xmy: Omega(diagonal)(X-Y)
        left_residues=left_residues.scaled_add(&omega_xpy,-1.0).unwrap();
        right_residues=right_residues.scaled_add(&omega_xmy,-1.0).unwrap();
        println!("Now is iteration #{},current progress:",iter_num);
        if print_level>1{println!("Right residues:");
        right_residues.iter_columns_full().for_each(|vec|{
            println!("{}",vec.iter().map(|x|x.powf(2.0)).sum::<f64>().powf(0.5));
        });
        println!("Left residues:");
        left_residues.iter_columns_full().for_each(|vec|{
            println!("{}",vec.iter().map(|x|x.powf(2.0)).sum::<f64>().powf(0.5));
        });}
        //R_left=(A-B)(X-Y)-Omega(X+Y)
        //R_right=(A+B)(X+Y)-Omega(X-Y)
        let mut converge=true;
        let mut left_converge_pair=0;
        let mut right_converge_pair=0;
        left_residues.iter_columns_full().for_each(|residue|{
            let norm=residue.iter().fold(0.0,|acc,val|acc+val.powf(2.0));
            if norm>1e-10{
                converge=false;
            }else{
                left_converge_pair+=1;
            }
        });
        if converge{
            right_residues.iter_columns_full().for_each(|residue|{
                let norm=residue.iter().fold(0.0,|acc,val|acc+val.powf(2.0));
                if norm>1e-10{
                    converge=false;
                }else{
                    right_converge_pair+=1;
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
            println!("Davidson Solver has converged. This final iteration took {:?}",start.elapsed());
            break;
        }
        left_residues.iter_columns_full().enumerate().for_each(|(i,residue)|{
            let mut preconditioned=residue.iter().enumerate().map(|(k,v_k)|v_k/(omega[i]-diag[k])).collect();
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
                if print_level>1{println!("A new search vector has been added. Search space now has {} vectors",ss.size[1]);}
            }
        });
        right_residues.iter_columns_full().enumerate().for_each(|(i,residue)|{
            let mut preconditioned=residue.iter().enumerate().map(|(k,v_k)|v_k/(omega[i]-diag[k])).collect();
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
                if print_level>1{println!("A new search vector has been added. Search space now has {} vectors",ss.size[1]);}
            }
        });
        if print_level>1{println!("Search Space:");
        ss.formated_output(1000,"full");}
        if iter_num>config.max_iter{
            break;
        }
        println!("For Left Residues, {} out of the {} desired solutions have converged",left_converge_pair,nroots);
        println!("For Right Residues, {} out of the {} desired solutions have converged",right_converge_pair,nroots);
        println!("This iteration took {:?}",start.elapsed());
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