use crate::scf_io::SCF;
use num_complex::Complex;
use rest_tensors::{MatrixFull,MatrixUpper};
use crate::tensors::BasicMatrix;
use std::ffi::c_char;
use std::f64::consts::PI;
use rayon::prelude::*;

pub fn rrs_pbc_1d(scf: &SCF, core_hamil_size: usize, num_cluster: usize, p_vec: Vec<Vec<f64>>, k_points: usize) -> Vec<Vec<f64>> {
    // real num_cluster = 2*num_cluster + 1
    let real_num_cluster = 2 * num_cluster + 1;
    let mut fock = scf.hamiltonian[0].clone().to_matrixfull().unwrap();
    let mut ovlp = scf.ovlp.clone().to_matrixfull().unwrap();
    //generate p_vec and k_vec
    let p_vec_1d = p_vec[0].clone();
    let pi = PI;
    let mut k_value = vec![0.0;k_points];
    k_value = (1..(k_points+1)).map(|idx| { -pi + 2.0*pi*(idx as f64)/(k_points as f64) }).collect();
    let mut p_value = vec![0.0;num_cluster];
    p_value = (-(num_cluster as i32) .. (num_cluster as i32 + 1)).map(|idx| { idx as f64 }).collect();
    //build factor matrix [N_k, N_p]
    let mut complex_factor = vec![vec![Complex::new(0.0,0.0);real_num_cluster];k_points];
    complex_factor = k_value.iter().map(|&k| { p_value.iter().map(|&p| Complex::new(0.0,k*p).exp()).collect::<Vec<Complex<f64>>>()}).collect();
    //build core hamiltonian
    let full_hamil_size = fock.size[0] as usize;
    let full_core_size = (core_hamil_size * real_num_cluster) as usize;
    let half_core_size = (core_hamil_size * num_cluster) as usize;
    let edge_size = ((full_hamil_size - full_core_size) / 2) as usize;
    let core_fock_vec = fock.iter_submatrix(edge_size+half_core_size..edge_size+half_core_size+core_hamil_size,edge_size..edge_size+full_core_size).copied().collect::<Vec<f64>>();
    let core_ovlp_vec = ovlp.iter_submatrix(edge_size+half_core_size..edge_size+half_core_size+core_hamil_size,edge_size..edge_size+full_core_size).copied().collect::<Vec<f64>>();
    //fold fock and ovlp to Vec<Vec>
    //N_p, N_hamil * N_hamil
    let mut fold_fock = vec![vec![0.0;core_hamil_size*core_hamil_size];real_num_cluster];
    let mut fold_ovlp = vec![vec![0.0;core_hamil_size*core_hamil_size];real_num_cluster];
    for idx in 0..real_num_cluster {
        fold_fock[idx] = core_fock_vec[idx*core_hamil_size*core_hamil_size..(idx+1)*core_hamil_size*core_hamil_size].to_vec();
        fold_ovlp[idx] = core_ovlp_vec[idx*core_hamil_size*core_hamil_size..(idx+1)*core_hamil_size*core_hamil_size].to_vec();
    };
    let mut complex_fold_fock = vec![vec![Complex::new(0.0,0.0);core_hamil_size*core_hamil_size];real_num_cluster];
    let mut complex_fold_ovlp = vec![vec![Complex::new(0.0,0.0);core_hamil_size*core_hamil_size];real_num_cluster];
    complex_fold_fock.iter_mut().zip(fold_fock.iter()).for_each(|(f,h)| {
        f.iter_mut().zip(h.iter()).for_each(|(ff,hh)| {
            *ff = Complex::new(*hh,0.0);
        });
    });
    complex_fold_ovlp.iter_mut().zip(fold_ovlp.iter()).for_each(|(f,h)| {
        f.iter_mut().zip(h.iter()).for_each(|(ff,hh)| {
            *ff = Complex::new(*hh,0.0);
        });
    });
    // post_xxx: N_k, N_hamil * N_hamil
    let mut post_fock = vec![vec![Complex::new(0.0,0.0);core_hamil_size*core_hamil_size];k_points];
    let mut post_ovlp = vec![vec![Complex::new(0.0,0.0);core_hamil_size*core_hamil_size];k_points];

    post_fock.iter_mut().zip(complex_factor.iter()).for_each(|(f, c)| { //for k
        c.iter().zip(complex_fold_fock.iter_mut()).for_each(|(cc,p)| { //for p
            p.iter_mut().zip(f.iter_mut()).for_each(|(pp,ff)| { //for ij
                *ff += *pp * *cc;
            });
        });
    });

    post_ovlp.iter_mut().zip(complex_factor.iter()).for_each(|(f, c)| { //for k
        c.iter().zip(complex_fold_ovlp.iter_mut()).for_each(|(cc,p)| { //for p
            p.iter_mut().zip(f.iter_mut()).for_each(|(pp,ff)| { //for ij
                *ff += *pp * *cc;
            });
        });
    });
    //make post_xxx hermitian
    let mut herm_post_fock = vec![vec![Complex::new(0.0,0.0);core_hamil_size*core_hamil_size];k_points];
    herm_post_fock.par_iter_mut().zip(post_fock.par_iter()).for_each(|(pf,f)| {
        for i in 0..core_hamil_size {
            for j in 0..core_hamil_size {
                pf[i+j*core_hamil_size] = (f[i+j*core_hamil_size] + f[j+i*core_hamil_size].conj())/2.0;
            };
        };
    });
    let mut post_fock = herm_post_fock;

    let mut herm_post_ovlp = vec![vec![Complex::new(0.0,0.0);core_hamil_size*core_hamil_size];k_points];
    herm_post_ovlp.par_iter_mut().zip(post_ovlp.par_iter()).for_each(|(pf,f)| {
        for i in 0..core_hamil_size {
            for j in 0..core_hamil_size {
                pf[i+j*core_hamil_size] = (f[i+j*core_hamil_size] + f[j+i*core_hamil_size].conj())/2.0;
            };
        };
    });
    let mut post_ovlp = herm_post_ovlp;

    //cholesky for ovlp
    let uplo = b'L' as c_char;
    let n: i32 = core_hamil_size.clone() as i32;
//    let mut complex_l_ovlp = post_ovlp.clone();
    post_ovlp.par_iter_mut().for_each(|ov| {
        let mut info = 0;
        unsafe {
            zpotrf_(
                &uplo as *const c_char,
                &n as *const i32,
                ov.as_mut_ptr(),
                &n as *const i32,
                &mut info as *mut i32,
                );
        };
        if info != 0 {
           panic!("Cholesky decomposition failed");
        };
    });
    let mut identity = vec![vec![Complex::new(0.0,0.0);core_hamil_size*core_hamil_size];k_points];
    identity.iter_mut().for_each(|ii| {
        for i in 0..core_hamil_size {
            for j in 0..core_hamil_size {
                if i == j {
                    ii[i+core_hamil_size*j] = Complex::new(1.0,0.0);
                };
            };
        };
    });

    //get inv_sqrt_ovlp
    let uplo = b'L' as c_char;
    let trans = b'N' as c_char;
    let diag = b'N' as c_char;
    let n = core_hamil_size.clone() as i32;
    let nrhs = core_hamil_size.clone() as i32;
    let lda = core_hamil_size.clone() as i32;
    let ldb = core_hamil_size.clone() as i32;
    post_ovlp.par_iter_mut().zip(identity.par_iter_mut()).for_each(|(ov,iid)| {
        let mut info = 0;
        unsafe {
            ztrtrs_(
                &uplo as *const c_char,
                &trans as *const c_char,
                &diag as *const c_char,
                &n as *const i32,
                &nrhs as *const i32,
                ov.as_mut_ptr(),
                &lda as *const i32,
                iid.as_mut_ptr(),
                &ldb as *const i32,
                &mut info as *mut i32,
                );
        };
        if info != 0 {
            panic!("Lapack ztrtrs failed");
        };
    });

    //generate L^-1 @ F
    let mut inv_sqrt_ovlp_1 = identity.clone();
    let mut inv_sqrt_ovlp_2 = identity.clone();
    let mut mid_fock = vec![vec![Complex::new(0.0,0.0);core_hamil_size*core_hamil_size];k_points];
    let transa = b'N' as c_char;
    let transb = b'N' as c_char;
    let m = core_hamil_size.clone() as i32;
    let n = core_hamil_size.clone() as i32;
    let k = core_hamil_size.clone() as i32;
    let alpha = Complex::new(1.0,0.0);
    let lda = core_hamil_size.clone() as i32;
    let beta = Complex::new(0.0,0.0);
    let ldc = core_hamil_size.clone() as i32;
    inv_sqrt_ovlp_1.par_iter_mut().zip(post_fock.par_iter_mut()).zip(mid_fock.par_iter_mut()).for_each(|((ivov,f),mf)| {
        unsafe {
            zgemm_(
            &transa as *const c_char,
            &transb as *const c_char,
            &m as *const i32,
            &n as *const i32,
            &k as *const i32,
            &alpha as *const Complex<f64>,
            ivov.as_mut_ptr(),
            &lda as *const i32,
            f.as_mut_ptr(),
            &ldb as *const i32,
            &beta as *const Complex<f64>,
            mf.as_mut_ptr(),
            &ldc as *const i32,
            );
        };
    });
    //generate L^-1 @ F @ L^-1^H
    let mut fin_fock = vec![vec![Complex::new(0.0,0.0);core_hamil_size*core_hamil_size];k_points];
    let transb = b'C' as c_char; //conj is needed
    mid_fock.par_iter_mut().zip(inv_sqrt_ovlp_2.par_iter_mut()).zip(fin_fock.par_iter_mut()).for_each(|((mf,ivov),ff)| {
        unsafe {
            zgemm_(
            &transa as *const c_char,
            &transb as *const c_char,
            &m as *const i32,
            &n as *const i32,
            &k as *const i32,
            &alpha as *const Complex<f64>,
            mf.as_mut_ptr(),
            &lda as *const i32,
            ivov.as_mut_ptr(),
            &ldb as *const i32,
            &beta as *const Complex<f64>,
            ff.as_mut_ptr(),
            &ldc as *const i32,
            );
        };
    });

    //get eigvalues and eigvectors
    let jobvl = b'N' as c_char;
    let jobvr = b'V' as c_char;
    let n = core_hamil_size.clone() as i32;
    //w: eigvalues
    let mut w = vec![vec![Complex::new(0.0,0.0);core_hamil_size];k_points];
    //vl is not used actually
    let mut vl = vec![vec![Complex::new(0.0,0.0);core_hamil_size*core_hamil_size];k_points];
    let ldvl = core_hamil_size.clone() as i32;
    let mut vr = vec![vec![Complex::new(0.0,0.0);core_hamil_size*core_hamil_size];k_points];
    let ldvr = core_hamil_size.clone() as i32;
    let lwork = 2 * n;

    fin_fock.par_iter_mut().zip(w.par_iter_mut()).zip(vl.par_iter_mut()).zip(vr.par_iter_mut()).for_each(|(((f,ww),vll),vrr)| {
        let mut info = 0;
        let mut work = vec![Complex::new(0.0,0.0);(2*n).try_into().unwrap()];
        let mut rwork = vec![0.0;(2*n).try_into().unwrap()];
        unsafe {
            zgeev_(
            &jobvl as *const c_char,
            &jobvr as *const c_char,
            &n as *const i32,
            f.as_mut_ptr(),
            &n as *const i32,
            ww.as_mut_ptr(),
            vll.as_mut_ptr(),
            &n as *const i32,
            vrr.as_mut_ptr(),
            &n as *const i32,
            work.as_mut_ptr(),
            &lwork as *const i32,
            rwork.as_mut_ptr(),
            &mut info as *mut i32)
        };
    });
   
    // Note that, ideally, fin_fock should be hermitian, and lead to REAL eigvalues w
    // Thus, only real part will be returned

    let mut real_w = vec![vec![0.0;core_hamil_size];k_points];
    real_w.iter_mut().zip(w.iter()).for_each(|(rw,cw)| {
        rw.iter_mut().zip(cw.iter()).for_each(|(rwi,cwi)| {
            *rwi = cwi.re;
        });
    });

    //sort eigvalues
    real_w.iter_mut().for_each(|ww| {
        ww.sort_by(|a,b| a.partial_cmp(b).unwrap());
    });
    real_w
}
    //get true l_ovlp
//    let mut complex_l_ovlp = vec![vec![Complex::new(0.0,0.0);core_hamil_size*core_hamil_size];k_points];
//    complex_l_ovlp.iter_mut().zip(post_ovlp.iter()).for_each(|l,ov| {
//        for i in 0..core_hamil_size {
//            for j in 0..core_hamil_size {
//                if i >= j {
//                    *l[i+core_hamil_size*j] = *ov[i+core_hamil_size*j];
//                };
//            };
//        };
//    });

#[link(name="lapack")]
extern "C" {
    fn zgemm_(
        transa: *const c_char,
        transb: *const c_char,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const Complex<f64>,
        a: *const Complex<f64>,
        lda: *const i32,
        b: *const Complex<f64>,
        ldb: *const i32,
        beta: *const Complex<f64>,
        c: *mut Complex<f64>,
        ldc: *const i32,
    );
}

#[link(name="lapack")]
extern "C" {
    fn zgeev_(
        jobvl: *const c_char,
        jobvr: *const c_char,
        n: *const i32,
        a: *const Complex<f64>,
        lda: *const i32,
        w: *mut Complex<f64>,
        vl: *mut Complex<f64>,
        ldvl: *const i32,
        vr: *mut Complex<f64>,
        ldvr: *const i32,
        work: *mut Complex<f64>,
        lwork: *const i32,
        rwork: *mut f64,
        info: *mut i32,
        );
}

#[link(name="lapack")]
extern "C" {
    fn zpotrf_(
        uplo: *const c_char,
        n: *const i32,
        a: *mut Complex<f64>,
        lda: *const i32,
        info: *mut i32);
}

#[link(name="lapack")]
extern "C" {
    fn ztrtrs_(
        uplo: *const c_char,
        trans: *const c_char,
        diag: *const c_char,
        n: *const i32,
        nrhs: *const i32,
        a: *mut Complex<f64>,
        lda: *const i32,
        b: *mut Complex<f64>,
        ldb: *const i32,
        info: *mut i32);
}
