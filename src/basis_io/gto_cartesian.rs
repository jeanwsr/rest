//! Primitive-level cartesian GTO evaluators: batch/tabulated values and first-,
//! second-, third-order derivatives of a single contracted-cartesian gaussian
//! `gau(a,l,c)` on coordinate batches (serial / rayon-parallel / libcint-norm variants).
use rest_tensors::{MatrixFull, RIFull};
use rayon::prelude::*;
use itertools::{Itertools, izip};
use tensors::{MathMatrix, ParMathMatrix};
use tensors::matrix_blas_lapack::{_dgemm_nn, _dgemm_nn_serial};
use std::convert::TryInto;
use std::f64::consts::{PI, E};
use crate::utilities::basic_math::{double_factorial, opowf, specific_double_factorial};
use crate::constants::{self, C2S_L0, C2S_L1, C2S_L2, C2S_L3, C2S_L4, CarBasInfo, c2s_matrix_const, cartesian_gto_const, cint_cartesian_gto_const};
use crate::utilities;
use super::{cint_norm_factor, cint_cart_norm_factor};

/// displacement vector and its squared norm between two points
#[inline]
fn delta_r(r1: &[f64;3], r2: &[f64;3]) -> ([f64;3],f64) {
    let mut dr = [0.0;3];
    let mut res = 0.0;
    for i in 0..3 {
        dr[i] = r1[i] - r2[i];
        res += dr[i]*dr[i];
    }
    return (dr, res);
}

/// produce the value of cartesian gaussian-type orbital "gau(a,l,c)" in a given coordinate "r=(x,y,z)"
///
/// ```text
///       gau(a,l,c) = Norm*x^{lx}*y^{ly}*z^{lz}*exp[-a*r^2],
///                  = Norm*Fang*exp[-a*r^2],
/// ```
///
/// The GTO is normalized with the normalization factor "Norm" as
///
///   Norm = (2a/Pi)^(3/4)*(4a)^{l/2}/((2*lx-1)!!(2*ly-1)!!(2*lz-1)!!)^{1/2}
///
pub fn cartesian_gto_std(a: f64, l: usize, c:&[f64;3],r:&[f64;3]) -> MatrixFull<f64> {
    // produce the value of cartesian gaussian-type orbital "gau(a,l,c)" in a given coordinate "r=(x,y,z)"
    //
    //       gau(a,l,c) = Norm*x^{lx}*y^{ly}*z^{lz}*exp[-a*r^2],
    //
    // The GTO is normalized with the normalization factor "Norm" as
    //
    //   Norm = (2a/Pi)^(3/4)*(4a)^{l/2}/((2*lx-1)!!(2*ly-1)!!(2*lz-1)!!)^{1/2}
    //

    // results in the same order that are used in libcint
    let mut tmp_v: Vec<f64> = vec![];
    let mut rr:[f64;3] = r.clone();
    rr.iter_mut().zip(c.iter()).for_each(|(r,c)| *r -= c);
    let mut rdot = rr.iter().fold(0.0f64,|acc,rr| acc + rr*rr);
    let exp_part = libm::exp(-a*rdot);
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    for lx in (0..=l).rev() {
        let xp = rr[0].powf(lx as f64);
        //debug here
        let normx = double_factorial((2*lx) as i32-1) as f64;
        let rl = l - lx;
        for ly in (0..=rl).rev() {
            let lz = rl - ly;
            let normy = double_factorial((2*ly) as i32-1) as f64;
            let normz = double_factorial((2*lz) as i32-1) as f64;

            //tmp_v.push(
            //        xp*rr[1].powf(ly as f64)*rr[2].powf(lz as f64)
            //        *exp_part);
            tmp_v.push(norm0/(normx*normy*normz).sqrt()
                    *xp*rr[1].powf(ly as f64)*rr[2].powf(lz as f64)
                    *exp_part);
            //println!("debug norm 2: {}", norm0/(normx*normy*normz).sqrt());
        }
    }
    unsafe{MatrixFull::from_vec_unchecked([(l+1)*(l+2)/2,1], tmp_v)}
}

pub fn cartesian_gto_batch(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> MatrixFull<f64> {
    // produce the value of cartesian gaussian-type orbital "gau(a,l,c)" in a given coordinate "r=(x,y,z)"
    //
    //       gau(a,l,c) = Norm*x^{lx}*y^{ly}*z^{lz}*exp[-a*r^2],
    //                  = Norm*Fang*exp[-a*r^2],
    //
    // The GTO is normalized with the normalization factor "Norm" as
    //
    //   Norm = (2a/Pi)^(3/4)*(4a)^{l/2}/((2*lx-1)!!(2*ly-1)!!(2*lz-1)!!)^{1/2}
    //

    // results in the same order that are used in libcint
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    //let mut tmp_v: Vec<f64> = vec![];
    let mut tmp_mat = MatrixFull::new([num_grids,num_bas],0.0);
    let mut i_bas:usize = 0;
    for lx in (0..=l).rev() {
        let normx = double_factorial((2*lx) as i32-1) as f64;
        let lxf = lx as f64;
        let rl = l - lx;
        for ly in (0..=rl).rev() {
            let lz = rl - ly;
            let normy = double_factorial((2*ly) as i32-1) as f64;
            let normz = double_factorial((2*lz) as i32-1) as f64;
            let norm  = norm0/(normx*normy*normz).sqrt();

            let lyf = ly as f64;
            let lzf = lz as f64;
            tmp_mat.iter_column_mut(i_bas).zip(r.iter()).for_each(|(bas_r,r)| {
                //r.iter().zip(c.iter())
                let mut rdot = 0.0;
                let mut fang = 1.0;
                izip!(r.iter(),c.iter(),[lxf,lyf,lzf].iter()).for_each(|(rx,cx,lx)| {
                    let rrx = rx-cx;
                    rdot += rrx*rrx;
                    fang *= rrx.powf(*lx);
                });
                let exp_part = libm::exp(-a*rdot);
                *bas_r = norm*fang*exp_part;
            });
            i_bas +=1;
        }
    }
    tmp_mat
}

#[inline]
pub fn cartesian_gto_batch_par(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> MatrixFull<f64> {
    // produce the value of cartesian gaussian-type orbital "gau(a,l,c)" in a given coordinate "r=(x,y,z)"
    //
    //       gau(a,l,c) = Norm*x^{lx}*y^{ly}*z^{lz}*exp[-a*r^2],
    //                  = Norm*Fang*exp[-a*r^2],
    //
    // The GTO is normalized with the normalization factor "Norm" as
    //
    //   Norm = (2a/Pi)^(3/4)*(4a)^{l/2}/((2*lx-1)!!(2*ly-1)!!(2*lz-1)!!)^{1/2}
    //

    // results in the same order that are used in libcint
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    //let mut tmp_v: Vec<f64> = vec![];
    let mut tmp_mat = MatrixFull::new([num_grids,num_bas],0.0);
    let mut i_bas:usize = 0;
    //let mut lf = [0.0,0.0,0.0];
    for lx in (0..=l).rev() {
        //let normx = double_factorial((2*lx) as i32-1) as f64;
        let normx = specific_double_factorial((2*lx) as i32-1);
        let lxf = lx as f64;
        //lf[0] = lx as f64;
        let rl = l - lx;
        for ly in (0..=rl).rev() {
            let lz = rl - ly;
            let normy = specific_double_factorial((2*ly) as i32-1);
            let normz = specific_double_factorial((2*lz) as i32-1);
            let norm  = norm0/(normx*normy*normz).sqrt();
            //println!("debug: {:16.8},{:16.8},{:2},{:16.8}, {:16.8}, {:16.8},{:16.8}",cint_norm_factor(l as i32, a), a,l, norm0, normx, normy,normz);

            //lf[1] = ly as f64;
            //lf[2] = (lz-ly) as f64;
            let lyf = ly as f64;
            let lzf = lz as f64;
            tmp_mat.par_iter_column_mut(i_bas).zip(r.par_iter()).for_each(|(bas_r,r)| {
                //r.iter().zip(c.iter())
                //let mut rdot = 0.0;
                //let mut fang = 1.0;
                //izip!(r.iter(),c.iter(),[lxf,lyf,lzf].iter()).for_each(|(rx,cx,lx)| {
                //    let rrx = rx-cx;
                //    rdot += rrx.powf(2.0);
                //    fang *= rrx.powf(*lx);
                //});
                let (rdot, fang) = izip!(r.iter(),c.iter(),[lxf,lyf,lzf].iter())
                //let (rdot, fang) = izip!(r.iter(),c.iter(),(&lf).iter())
                    .fold((0.0,1.0), |(rdot, fang), (rx,cx,lx)| {
                        let rrx = rx-cx;
                        (rdot + rrx.powf(2.0), fang * rrx.powf(*lx))
                });
                let exp_part = libm::exp(-a*rdot);
                *bas_r = norm*fang*exp_part;
            });
            i_bas +=1;
        }
    }
    tmp_mat
}
#[inline]
pub fn cartesian_gto_batch_par_v02(a: f64, l: usize, c:&[f64;3],r:&Vec<[f64;3]>) -> MatrixFull<f64> {
    // produce the value of cartesian gaussian-type orbital "gau(a,l,c)" in a given coordinate "r=(x,y,z)"
    //
    //       gau(a,l,c) = Norm*x^{lx}*y^{ly}*z^{lz}*exp[-a*r^2],
    //                  = Norm*Fang*exp[-a*r^2],
    //
    // The GTO is normalized with the normalization factor "Norm" as
    //
    //   Norm = (2a/Pi)^(3/4)*(4a)^{l/2}/((2*lx-1)!!(2*ly-1)!!(2*lz-1)!!)^{1/2}
    //

    // results in the same order that are used in libcint
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);

    let binding = cartesian_gto_const(l);
    let basinfo = &binding.to_matrixfullslice();;

    let mut tmp_mat = MatrixFull::new([num_grids,num_bas],0.0_f64);
    tmp_mat.iter_columns_full_mut().zip(basinfo.iter_columns_full()).for_each(|(bas_r,bas_i)| {
        bas_r.par_iter_mut().zip(r.par_iter()).for_each(|(bas_r,r)| {
            let (rdot, fang) = izip!(r.iter(),c.iter(),bas_i[0..3].iter())
                .fold((0.0,1.0), |(rdot, fang), (rx,cx,lx)| {
                    let rrx = rx-cx;
                    (rdot + rrx.powf(2.0), fang * rrx.powf(*lx))
            });
            let exp_part = libm::exp(-a*rdot);
            *bas_r = norm0*bas_i[3]*fang*exp_part;
        })
    });
    tmp_mat
}
#[inline]
pub fn cartesian_gto_batch_par_v03(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> MatrixFull<f64> {
    // produce the value of cartesian gaussian-type orbital "gau(a,l,c)" in a given coordinate "r=(x,y,z)"
    //
    //       gau(a,l,c) = Norm*x^{lx}*y^{ly}*z^{lz}*exp[-a*r^2],
    //                  = Norm*Fang*exp[-a*r^2],
    //
    // The GTO is normalized with the normalization factor "Norm" as
    //
    //   Norm = (2a/Pi)^(3/4)*(4a)^{l/2}/((2*lx-1)!!(2*ly-1)!!(2*lz-1)!!)^{1/2}
    //

    // results in the same order that are used in libcint
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    let cut_off:f64 = (constants::E9).log(constants::E)/a;

    let binding = cartesian_gto_const(l);
    let basinfo = &binding.to_matrixfullslice();;

    let mut tmp_mat = MatrixFull::new([num_bas,num_grids],0.0_f64);
    tmp_mat.par_iter_columns_full_mut().zip(r.par_iter())
        .filter(|(r_bas,r)| {
            let rr = r.iter().zip(c.iter()).fold(0.0, |rdot, (r,c)| {rdot + (r-c).powf(2.0_f64)});
            rr < cut_off
            //true
        }).for_each(|(r_bas,r)| {
            let mut rr = [0.0;3];
            let rdot = izip!(rr.iter_mut(),r.iter(),c.iter()).fold(0.0, |rdot, (rr,r,c)| {
                *rr= r-c;
                rdot + rr.powf(2.0_f64)}
            );
            let exp_part = norm0*libm::exp(-a*rdot);
            r_bas.iter_mut().zip(basinfo.iter_columns_full()).for_each(|(bas_r,bas_i)| {
                let fang = rr.iter().zip(bas_i[0..3].iter())
                    .fold(1.0, |fang, (rr,lx)| {
                        fang * rr.powf(*lx)
                });
                *bas_r = bas_i[3]*fang*exp_part;
            });
    });

    tmp_mat.transpose_and_drop()
}

pub fn cartesian_gto_batch_v03(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> MatrixFull<f64> {
    // produce the value of cartesian gaussian-type orbital "gau(a,l,c)" in a given coordinate "r=(x,y,z)"
    //
    //       gau(a,l,c) = Norm*x^{lx}*y^{ly}*z^{lz}*exp[-a*r^2],
    //                  = Norm*Fang*exp[-a*r^2],
    //
    // The GTO is normalized with the normalization factor "Norm" as
    //
    //   Norm = (2a/Pi)^(3/4)*(4a)^{l/2}/((2*lx-1)!!(2*ly-1)!!(2*lz-1)!!)^{1/2}
    //

    // results in the same order that are used in libcint
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    let cut_off:f64 = (constants::E9).log(constants::E)/a;

    let binding = cartesian_gto_const(l);
    let basinfo = &binding.to_matrixfullslice();;

    let mut tmp_mat = MatrixFull::new([num_bas,num_grids],0.0_f64);
    tmp_mat.iter_columns_full_mut().zip(r.iter())
        //.filter(|(r_bas,r)| {
        //    let rr = r.iter().zip(c.iter()).fold(0.0, |rdot, (r,c)| {rdot + (r-c).powf(2.0_f64)});
        //    //rr < cut_off
        //    true
        //}).for_each(|(r_bas,r)| {
        .for_each(|(r_bas,r)| {
            let mut rr = [0.0;3];
            let rdot = izip!(rr.iter_mut(),r.iter(),c.iter()).fold(0.0, |rdot, (rr,r,c)| {
                *rr= r-c;
                rdot + rr.powf(2.0_f64)}
            );
            let exp_part = norm0*libm::exp(-a*rdot);
            r_bas.iter_mut().zip(basinfo.iter_columns_full()).for_each(|(bas_r,bas_i)| {
                let fang = rr.iter().zip(bas_i[0..3].iter())
                    .fold(1.0, |fang, (rr,lx)| {
                        fang * rr.powf(*lx)
                });
                *bas_r = bas_i[3]*fang*exp_part;
            });
    });
    tmp_mat.transpose_and_drop()

}

pub fn cartesian_gto_batch_v04(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> MatrixFull<f64> {
    // produce the value of cartesian gaussian-type orbital "gau(a,l,c)" in a given coordinate "r=(x,y,z)"
    // it will be used for the spheric GTO defined by libcint
    //
    //       gau(a,l,c) = Norm*(x-cx)^{lx}*(y-cy)^{ly}*(z-cz)^{lz}*exp[-a*|r-c|^2],
    //                  = Norm0*Fang*exp[-a*|r-c|^2],
    //
    // The GTO is normalized with the normalization factor "Norm" as
    //
    //   Norm = (2a/Pi)^(3/4)*(4a)^{l/2}/((2*lx-1)!!(2*ly-1)!!(2*lz-1)!!)^{1/2}
    //   Norm0 = (2a/Pi)^(3/4)*(4a)^{l/2}
    //   Fang = 1.0/((2*lx-1)!!(2*ly-1)!!(2*lz-1)!!)^{1/2} which has been evaluated and stored in constants/cartesian_gto.rs

    // results in the same order that are used in libcint
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    let norm0 =(2.0*a/PI).powf(0.75)*opowf((4.0*a),(l as f64)/2.0);

    let binding = cartesian_gto_const(l);
    let basinfo = &binding.to_matrixfullslice();;

    let mut tmp_mat = MatrixFull::new([num_bas,num_grids],0.0_f64);
    tmp_mat.iter_columns_full_mut().zip(r.iter())
        .for_each(|(r_bas,r)| {
            let (rr, rdot) = delta_r(r, c);
            let exp_part = norm0*libm::exp(-a*rdot);
            r_bas.iter_mut().zip(basinfo.iter_columns_full()).for_each(|(bas_r,bas_i)| {
                let fang = rr.iter().zip(&bas_i[0..3])
                    .fold(1.0, |fang, (rr,lx)| {
                        fang * opowf(*rr,*lx)
                });
                *bas_r = unsafe {bas_i.get_unchecked(3)*fang*exp_part};
            });
    });
    tmp_mat

}

pub fn cint_cartesian_gto_batch_v04(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> MatrixFull<f64> {
    // produce the value of cartesian gaussian-type orbital "gau(a,l,c)" in a given coordinate "r=(x,y,z)"
    // it will be used for the cartesian GTO defined by libcint
    //
    //       gau(a,l,c) = Norm*(x-cx)^{lx}*(y-cy)^{ly}*(z-cz)^{lz}*exp[-a*|r-c|^2],
    //                  = Norm0*Fang*exp[-a*|r-c|^2],
    //
    // The GTO is normalized with the normalization factor "Norm" as
    //
    //   Norm = (2a/Pi)^(3/4)*(4a)^{l/2}/((2*lx-1)!!(2*ly-1)!!(2*lz-1)!!)^{1/2}
    //   Norm0 = (2a/Pi)^(3/4)*(4a)^{l/2}
    //   Fang = 1.0/((2*lx-1)!!(2*ly-1)!!(2*lz-1)!!)^{1/2} which has been evaluated and stored in constants/cartesian_gto.rs

    // results in the same order that are used in libcint
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    // For l>=2 (cartesian), libcint leaves the GTO un-normalized (only gto_norm is
    // absorbed into the contraction coefficients). The tabulated GTO must therefore
    // be the *raw* polynomial x^lx y^ly z^lz exp(-a r^2) with no Norm0/Fang factor.
    let norm0 = if l >= 2 {
        1.0
    } else {
        (2.0*a/PI).powf(0.75)*opowf((4.0*a),(l as f64)/2.0)
    };

    // In libcint, the GTOs with l>=2 are not normalized. 
    // In concequence, the tabulated Cartesian GTOs prepared by REST are 
    //     0) normalized for l<2
    //     1) only normalized for the GTOs of x^{l}exp[-a*|r-c|^2)].
    let binding = cint_cartesian_gto_const(l);
    let basinfo = &binding.to_matrixfullslice();;

    let mut tmp_mat = MatrixFull::new([num_bas,num_grids],0.0_f64);
    tmp_mat.iter_columns_full_mut().zip(r.iter())
        .for_each(|(r_bas,r)| {
            let (rr, rdot) = delta_r(r, c);
            let exp_part = norm0*libm::exp(-a*rdot);
            r_bas.iter_mut().zip(basinfo.iter_columns_full()).for_each(|(bas_r,bas_i)| {
                let fang = rr.iter().zip(&bas_i[0..3])
                    .fold(1.0, |fang, (rr,lx)| {
                        fang * opowf(*rr,*lx)
                });
                *bas_r = fang*exp_part;
            });
    });
    tmp_mat

}

pub fn cartesian_gto_1st_batch(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> Vec<MatrixFull<f64>> {
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    //let mut pao = RIFull::new([num_grids,num_bas,3],0.0);
    let mut aox =  MatrixFull::new([num_grids,num_bas], 0.0);
    let mut aoy =  MatrixFull::new([num_grids,num_bas], 0.0);
    let mut aoz =  MatrixFull::new([num_grids,num_bas], 0.0);
    let ll = l as i32;
    let mut i_bas:usize = 0;
    for lx in (0..=ll).rev() {
        let lxf = lx as f64;
        let normx = double_factorial(2*lx-1) as f64;
        let rl = ll - lx;
        for ly in (0..=rl as i32).rev() {
            let lz = rl - ly;
            let lyf = ly as f64;
            let lzf = lz as f64;
            let normy = double_factorial(2*ly-1) as f64;
            let normz = double_factorial(2*lz-1) as f64;
            let norm  = norm0/(normx*normy*normz).sqrt();
            izip!(aox.iter_column_mut(i_bas),aoy.iter_column_mut(i_bas),aoz.iter_column_mut(i_bas),r.iter())
            .for_each(|(aox_r,aoy_r,aoz_r,r)| {

                let mut rdot = 0.0;
                let mut xyz = [0.0;3];
                let mut pxyz = [0.0;3];
                izip!(r.iter(),c.iter(),[lxf,lyf,lzf].iter(),xyz.iter_mut(),pxyz.iter_mut())
                    .for_each(|(rx,cx,lx,x,px)| {
                    let rrx = rx-cx;
                    rdot += rrx*rrx;
                    *x = rrx.powf(*lx);
                    *px = -2.0*a*rrx.powf(lx+1.0);
                    *px += if (*lx!=0.0) {lx*rrx.powf(lx-1.0)} else {0.0};
                });
                let exp_part = norm*libm::exp(-a*rdot);
                let [x,y,z] = xyz;
                let [px,py,pz] = pxyz;
                *aox_r = exp_part*px*y*z;
                *aoy_r = exp_part*x*py*z;
                *aoz_r = exp_part*x*y*pz;
            });
            i_bas +=1;
        }
    }
    vec![aox,aoy,aoz]
}
#[inline]
pub fn cartesian_gto_1st_batch_par(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> Vec<MatrixFull<f64>> {
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    //let mut pao = RIFull::new([num_grids,num_bas,3],0.0);
    let mut aox =  MatrixFull::new([num_grids,num_bas], 0.0);
    let mut aoy =  MatrixFull::new([num_grids,num_bas], 0.0);
    let mut aoz =  MatrixFull::new([num_grids,num_bas], 0.0);

    //let binding = cartesian_gto_const(l);
    //let basinfo = &binding.to_matrixfullslice();;
    let ll = l as i32;
    let mut i_bas:usize = 0;
    for lx in (0..=ll).rev() {
        let lxf = lx as f64;
        let normx = double_factorial(2*lx-1) as f64;
        let rl = ll - lx;
        for ly in (0..=rl as i32).rev() {
            let lz = rl - ly;
            let lyf = ly as f64;
            let lzf = lz as f64;
            let normy = double_factorial(2*ly-1) as f64;
            let normz = double_factorial(2*lz-1) as f64;
            let norm  = norm0/(normx*normy*normz).sqrt();

            //let (sender, receiver) = channel();
            aox.par_iter_column_mut(i_bas).zip(aoy.par_iter_column_mut(i_bas))
            .map(|(aox_r,aoy_r)| (aox_r,aoy_r)).zip(
            aoz.par_iter_column_mut(i_bas).zip(r.par_iter())
            .map(|(aoz_r,r)| (aoz_r,r)))
            .for_each(|((aox_r,aoy_r),(aoz_r,r))| {
                let mut rdot = 0.0;
                let mut xyz = [0.0;3];
                let mut pxyz = [0.0;3];
                izip!(r.iter(),c.iter(),[lxf,lyf,lzf].iter(),xyz.iter_mut(),pxyz.iter_mut())
                    .for_each(|(rx,cx,lx,x,px)| {
                    let rrx = rx-cx;
                    rdot += rrx*rrx;
                    *x = rrx.powf(*lx);
                    *px = -2.0*a*rrx.powf(lx+1.0);
                    *px += if (*lx!=0.0) {lx*rrx.powf(lx-1.0)} else {0.0};
                });
                let exp_part = norm*libm::exp(-a*rdot);
                let [x,y,z] = xyz;
                let [px,py,pz] = pxyz;
                *aox_r = exp_part*px*y*z;
                *aoy_r = exp_part*x*py*z;
                *aoz_r = exp_part*x*y*pz;

            });

            i_bas +=1;
        }
    }
    vec![aox,aoy,aoz]
}

#[inline]
pub fn cartesian_gto_1st_batch_par_v02(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> Vec<MatrixFull<f64>> {
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    let cut_off:f64 = (constants::E9).log(constants::E)/a;

    //let mut pao = RIFull::new([num_grids,num_bas,3],0.0);
    //let mut aox =  MatrixFull::new([num_grids,num_bas], 0.0);
    //let mut aoy =  MatrixFull::new([num_grids,num_bas], 0.0);
    //let mut aoz =  MatrixFull::new([num_grids,num_bas], 0.0);

    let binding = cartesian_gto_const(l);
    let basinfo = &binding.to_matrixfullslice();;

    let mut aox =  MatrixFull::new([num_bas,num_grids], 0.0);
    let mut aoy =  MatrixFull::new([num_bas,num_grids], 0.0);
    let mut aoz =  MatrixFull::new([num_bas,num_grids], 0.0);

    aox.par_iter_columns_full_mut()
    .zip(aoy.par_iter_columns_full_mut())
    .zip(aoz.par_iter_columns_full_mut())
    .zip(r.par_iter()).map(|(((aox_r,aoy_r),aoz_r),r)| (aox_r,aoy_r,aoz_r,r))
    .filter(|(aox_r,aoy_r,aoz_r,r)| {
            let rr = r.iter().zip(c.iter()).fold(0.0, |rdot, (r,c)| {rdot + (r-c).powf(2.0_f64)});
            rr < cut_off
            //true
    }).for_each(|(aox_r,aoy_r,aoz_r,r)| {
        let mut rr = [0.0;3];
        let rdot = izip!(rr.iter_mut(),r.iter(),c.iter()).fold(0.0, |rdot, (rr,r,c)| {
            *rr= r-c;
            rdot + rr.powf(2.0_f64)}
        );
        let exp_part = norm0*libm::exp(-a*rdot);
        let mut xyz = [0.0;3];
        let mut pxyz = [0.0;3];
        aox_r.iter_mut().zip(aoy_r.iter_mut()).zip(aoz_r.iter_mut())
        .zip(basinfo.iter_columns_full())
        .map(|((((aox_r,aoy_r),aoz_r)),bas_i)| (aox_r, aoy_r, aoz_r,bas_i))
        .for_each(|(aox_r,aoy_r,aoz_r,bas_i)| {
            //xyz = [0.0;3];
            //pxyz = [0.0;3];
            izip!(xyz.iter_mut(),pxyz.iter_mut(),bas_i[0..3].iter(),rr.iter())
                .for_each(|(x,px,lx,rrx)| {
                    *x = rrx.powf(*lx);
                    *px = -2.0*a*rrx.powf(lx+1.0);
                    *px += if (*lx!=0.0) {lx*rrx.powf(lx-1.0)} else {0.0};
                });
            let [x,y,z] = xyz;
            let [px,py,pz] = pxyz;
            *aox_r = bas_i[3]*exp_part*px*y*z;
            *aoy_r = bas_i[3]*exp_part*x*py*z;
            *aoz_r = bas_i[3]*exp_part*x*y*pz;
        });
    });

    vec![aox.transpose_and_drop(),aoy.transpose_and_drop(),aoz.transpose_and_drop()]
}

#[inline]
pub fn cartesian_gto_1st_batch_v02(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> Vec<MatrixFull<f64>> {
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    let norm0 = (2.0*a/PI).powf(0.75)*opowf((4.0*a),(l as f64)/2.0);
    let cut_off:f64 = (constants::E9).log(constants::E)/a;

    //let mut pao = RIFull::new([num_grids,num_bas,3],0.0);
    //let mut aox =  MatrixFull::new([num_grids,num_bas], 0.0);
    //let mut aoy =  MatrixFull::new([num_grids,num_bas], 0.0);
    //let mut aoz =  MatrixFull::new([num_grids,num_bas], 0.0);

    let binding = cartesian_gto_const(l);
    let basinfo = &binding.to_matrixfullslice();;

    let mut aox =  MatrixFull::new([num_bas,num_grids], 0.0);
    let mut aoy =  MatrixFull::new([num_bas,num_grids], 0.0);
    let mut aoz =  MatrixFull::new([num_bas,num_grids], 0.0);

    aox.iter_columns_full_mut()
    .zip(aoy.iter_columns_full_mut())
    .zip(aoz.iter_columns_full_mut())
    .zip(r.iter()).map(|(((aox_r,aoy_r),aoz_r),r)| (aox_r,aoy_r,aoz_r,r))
    //.filter(|(aox_r,aoy_r,aoz_r,r)| {
    //        let rr = r.iter().zip(c.iter()).fold(0.0, |rdot, (r,c)| {rdot + (r-c).powf(2.0_f64)});
    //        //rr < cut_off
    //        true
    //})
    .for_each(|(aox_r,aoy_r,aoz_r,r)| {
        //let mut rr = [0.0;3];
        //let rdot = izip!(rr.iter_mut(),r.iter(),c.iter()).fold(0.0, |rdot, (rr,r,c)| {
        //    *rr= r-c;
        //    rdot + rr.powf(2.0_f64)}
        //);
        let (rr, rdot) = delta_r(r, c);
        let exp_part = norm0*libm::exp(-a*rdot);
        let mut xyz = [0.0;3];
        let mut pxyz = [0.0;3];
        aox_r.iter_mut().zip(aoy_r.iter_mut()).zip(aoz_r.iter_mut())
        .zip(basinfo.iter_columns_full())
        .map(|((((aox_r,aoy_r),aoz_r)),bas_i)| (aox_r, aoy_r, aoz_r,bas_i))
        .for_each(|(aox_r,aoy_r,aoz_r,bas_i)| {
            //xyz = [0.0;3];
            //pxyz = [0.0;3];
            izip!(xyz.iter_mut(),pxyz.iter_mut(),bas_i[0..3].iter(),rr.iter())
                .for_each(|(x,px,lx,rrx)| {
                    *x = opowf(*rrx,*lx);
                    //*px = -2.0*a*rrx.powf(lx+1.0);
                    *px = -2.0*a*opowf(*rrx,lx+1.0);
                    //*px += if (*lx!=0.0) {lx*rrx.powf(lx-1.0)} else {0.0};
                    *px += if (*lx!=0.0) {lx*opowf(*rrx,lx-1.0)} else {0.0};
                });
            let [x,y,z] = xyz;
            let [px,py,pz] = pxyz;
            *aox_r = bas_i[3]*exp_part*px*y*z;
            *aoy_r = bas_i[3]*exp_part*x*py*z;
            *aoz_r = bas_i[3]*exp_part*x*y*pz;
        });
    });

    vec![aox.transpose_and_drop(),aoy.transpose_and_drop(),aoz.transpose_and_drop()]
}

#[inline]
pub fn cartesian_gto_1st_batch_v03(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> Vec<MatrixFull<f64>> {
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    //let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    let norm0 = (2.0*a/PI).powf(0.75)*opowf((4.0*a),(l as f64)/2.0);
    //let cut_off:f64 = (constants::E7).log(constants::E)/a;

    let binding = cartesian_gto_const(l);
    let basinfo = &binding.to_matrixfullslice();;

    let mut aox =  MatrixFull::new([num_bas,num_grids], 0.0);
    let mut aoy =  MatrixFull::new([num_bas,num_grids], 0.0);
    let mut aoz =  MatrixFull::new([num_bas,num_grids], 0.0);

    aox.iter_columns_full_mut()
    .zip(aoy.iter_columns_full_mut())
    .zip(aoz.iter_columns_full_mut())
    .zip(r.iter()).map(|(((aox_r,aoy_r),aoz_r),r)| (aox_r,aoy_r,aoz_r,r))
    //.filter(|(aox_r,aoy_r,aoz_r,r)| {
    //        let rr = r.iter().zip(c.iter()).fold(0.0, |rdot, (r,c)| {rdot + (r-c).powf(2.0_f64)});
    //        rr < cut_off
    //        //true
    //}).for_each(|(aox_r,aoy_r,aoz_r,r)| {
    .for_each(|(aox_r,aoy_r,aoz_r,r)| {
        let (rr, rdot) = delta_r(r, c);
        //let mut rr = [0.0;3];
        //let rdot = izip!(rr.iter_mut(),r.iter(),c.iter()).fold(0.0, |rdot, (rr,r,c)| {
        //    *rr= r-c;
        //    rdot + rr.powf(2.0_f64)}
        //);
        let exp_part = norm0*libm::exp(-a*rdot);
        let mut xyz = [0.0;3];
        let mut pxyz = [0.0;3];
        aox_r.iter_mut().zip(aoy_r.iter_mut()).zip(aoz_r.iter_mut())
        .zip(basinfo.iter_columns_full())
        .map(|((((aox_r,aoy_r),aoz_r)),bas_i)| (aox_r, aoy_r, aoz_r,bas_i))
        .for_each(|(aox_r,aoy_r,aoz_r,bas_i)| {
            //xyz = [0.0;3];
            //pxyz = [0.0;3];
            izip!(xyz.iter_mut(),pxyz.iter_mut(),bas_i[0..3].iter(),rr.iter())
                .for_each(|(x,px,lx,rrx)| {
                    //*x = rrx.powf(*lx);
                    //*px = -2.0*a*rrx.powf(lx+1.0);
                    //*px += if (*lx!=0.0) {lx*rrx.powf(lx-1.0)} else {0.0};
                    *x = opowf(*rrx,*lx);
                    *px = -2.0*a*opowf(*rrx,lx+1.0);
                    *px += if (*lx!=0.0) {lx*opowf(*rrx,lx-1.0)} else {0.0};
                });
            let [x,y,z] = xyz;
            let [px,py,pz] = pxyz;
            *aox_r = bas_i[3]*exp_part*px*y*z;
            *aoy_r = bas_i[3]*exp_part*x*py*z;
            *aoz_r = bas_i[3]*exp_part*x*y*pz;
        });
    });

    vec![aox,aoy,aoz]
}

#[inline]
pub fn cint_cartesian_gto_1st_batch_v03(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> Vec<MatrixFull<f64>> {
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    // For l>=2 (cartesian), the tabulated GTO gradient must match the raw polynomial
    // (no Norm0/Fang), consistent with cint_cartesian_gto_batch_v04.
    let norm0 = if l >= 2 {
        1.0
    } else {
        (2.0*a/PI).powf(0.75)*opowf((4.0*a),(l as f64)/2.0)
    };
    let cut_off:f64 = (constants::E7).log(constants::E)/a;

    let binding = cint_cartesian_gto_const(l);
    let basinfo = &binding.to_matrixfullslice();;

    let mut aox =  MatrixFull::new([num_bas,num_grids], 0.0);
    let mut aoy =  MatrixFull::new([num_bas,num_grids], 0.0);
    let mut aoz =  MatrixFull::new([num_bas,num_grids], 0.0);

    aox.iter_columns_full_mut()
    .zip(aoy.iter_columns_full_mut())
    .zip(aoz.iter_columns_full_mut())
    .zip(r.iter()).map(|(((aox_r,aoy_r),aoz_r),r)| (aox_r,aoy_r,aoz_r,r))
    //.filter(|(aox_r,aoy_r,aoz_r,r)| {
    //        let rr = r.iter().zip(c.iter()).fold(0.0, |rdot, (r,c)| {rdot + (r-c).powf(2.0_f64)});
    //        rr < cut_off
    //        //true
    //}).for_each(|(aox_r,aoy_r,aoz_r,r)| {
    .for_each(|(aox_r,aoy_r,aoz_r,r)| {
        let (rr, rdot) = delta_r(r, c);
        //let mut rr = [0.0;3];
        //let rdot = izip!(rr.iter_mut(),r.iter(),c.iter()).fold(0.0, |rdot, (rr,r,c)| {
        //    *rr= r-c;
        //    rdot + rr.powf(2.0_f64)}
        //);
        let exp_part = norm0*libm::exp(-a*rdot);
        let mut xyz = [0.0;3];
        let mut pxyz = [0.0;3];
        aox_r.iter_mut().zip(aoy_r.iter_mut()).zip(aoz_r.iter_mut())
        .zip(basinfo.iter_columns_full())
        .map(|((((aox_r,aoy_r),aoz_r)),bas_i)| (aox_r, aoy_r, aoz_r,bas_i))
        .for_each(|(aox_r,aoy_r,aoz_r,bas_i)| {
            //xyz = [0.0;3];
            //pxyz = [0.0;3];
            izip!(xyz.iter_mut(),pxyz.iter_mut(),bas_i[0..3].iter(),rr.iter())
                .for_each(|(x,px,lx,rrx)| {
                    //*x = rrx.powf(*lx);
                    //*px = -2.0*a*rrx.powf(lx+1.0);
                    //*px += if (*lx!=0.0) {lx*rrx.powf(lx-1.0)} else {0.0};
                    *x = opowf(*rrx,*lx);
                    *px = -2.0*a*opowf(*rrx,lx+1.0);
                    *px += if (*lx!=0.0) {lx*opowf(*rrx,lx-1.0)} else {0.0};
                });
            let [x,y,z] = xyz;
            let [px,py,pz] = pxyz;
            *aox_r = exp_part*px*y*z;
            *aoy_r = exp_part*x*py*z;
            *aoz_r = exp_part*x*y*pz;
        });
    });

    vec![aox,aoy,aoz]
}

pub fn _gto_nabla1(fx1:&mut [f64; 16], fy1:&mut [f64; 16], fz1:&mut [f64; 16], fx0:&[f64; 16], fy0:&[f64; 16], fz0:&[f64; 16], l:usize, a:f64) {
    let a2 = -2.0*a;
    fx1[0] = a2 * fx0[1];
    fy1[0] = a2 * fy0[1];
    fz1[0] = a2 * fz0[1];
    for lx in 1..=l {
        let lx_f64 = lx as f64;
        fx1[lx] = lx_f64 * fx0[lx-1] + a2 * fx0[lx+1];
        fy1[lx] = lx_f64 * fy0[lx-1] + a2 * fy0[lx+1];
        fz1[lx] = lx_f64 * fz0[lx-1] + a2 * fz0[lx+1];
    }
}

pub fn cartesian_gto_1st_batch_v04(a:f64, l:usize, c:&[f64;3], r:&[[f64;3]]) -> Vec<MatrixFull<f64>> {
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    let cut_off:f64 = (constants::E7).log(constants::E)/a;

    let binding = cartesian_gto_const(l);
    let basinfo = &binding.to_matrixfullslice();

    let mut ao   = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aox  = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoy  = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoz  = MatrixFull::new([num_bas, num_grids], 0.0);

    let mut fx0 = [0.0; 16];
    let mut fy0 = [0.0; 16];
    let mut fz0 = [0.0; 16];
    let mut fx1 = [0.0; 16];
    let mut fy1 = [0.0; 16];
    let mut fz1 = [0.0; 16];


    ao.iter_columns_full_mut()
    .zip(aox.iter_columns_full_mut())
    .zip(aoy.iter_columns_full_mut())
    .zip(aoz.iter_columns_full_mut())
    .zip(r.iter())
    .map(
        |((((ao, aox), aoy), aoz), r)| 
        (ao, aox, aoy, aoz, r)
    ).for_each(
        |(ao, aox, aoy, aoz, r)| {
            // exp part 
            let mut rr = [0.0; 3];
            let rdot = izip!(
                rr.iter_mut(), r.iter(), c.iter()).fold(0.0, |rdot, (rr, r, c)| 
                {
                    *rr= r - c;
                    rdot + rr.powf(2.0_f64)
                }
            );
            let e = norm0 * libm::exp(-a*rdot);
            // polynomial part 
            // initialize the 0th order 
            fx0[0] = 1.0; fy0[0] = 1.0; fz0[0] = 1.0;
            // check if it should be ri - rA
            for lx in 1..=l+2 {
                fx0[lx] = fx0[lx-1] * rr[0];
                fy0[lx] = fy0[lx-1] * rr[1];
                fz0[lx] = fz0[lx-1] * rr[2];
            }
            // first order
            _gto_nabla1(&mut fx1, &mut fy1, &mut fz1, &fx0, &fy0, &fz0, l, a);

            ao.iter_mut()
            .zip(aox.iter_mut())
            .zip(aoy.iter_mut())
            .zip(aoz.iter_mut())
            .zip(basinfo.iter_columns_full())
            .map(
                |((((ao_r, aox_r), aoy_r), aoz_r), bas_i)|
                (ao_r, aox_r, aoy_r, aoz_r, bas_i)
            )
            .for_each(|(ao_r, aox_r, aoy_r, aoz_r, bas_i)| {
                let lx = bas_i[0] as usize;
                let ly = bas_i[1] as usize;
                let lz = bas_i[2] as usize;
                let norm = bas_i[3];
                *ao_r   = norm * e * fx0[lx] * fy0[ly] * fz0[lz];
                *aox_r  = norm * e * fx1[lx] * fy0[ly] * fz0[lz];
                *aoy_r  = norm * e * fx0[lx] * fy1[ly] * fz0[lz];
                *aoz_r  = norm * e * fx0[lx] * fy0[ly] * fz1[lz];
            });
        }
    );
    vec![ao, aox, aoy, aoz]
}

pub fn cartesian_gto_2nd_batch_v04(a: f64, l: usize, c: &[f64;3], r:&[[f64;3]]) -> Vec<MatrixFull<f64>> {
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    let cut_off:f64 = (constants::E7).log(constants::E)/a;

    let binding = cartesian_gto_const(l);
    let basinfo = &binding.to_matrixfullslice();

    let mut ao   = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aox  = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoy  = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoz  = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoxx = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoxy = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoxz = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoyy = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoyz = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aozz = MatrixFull::new([num_bas, num_grids], 0.0);

    let mut fx0 = [0.0; 16];
    let mut fy0 = [0.0; 16];
    let mut fz0 = [0.0; 16];
    let mut fx1 = [0.0; 16];
    let mut fy1 = [0.0; 16];
    let mut fz1 = [0.0; 16];
    let mut fx2 = [0.0; 16];
    let mut fy2 = [0.0; 16];
    let mut fz2 = [0.0; 16];

    ao.iter_columns_full_mut()
    .zip(aox.iter_columns_full_mut())
    .zip(aoy.iter_columns_full_mut())
    .zip(aoz.iter_columns_full_mut())
    .zip(aoxx.iter_columns_full_mut())
    .zip(aoxy.iter_columns_full_mut())
    .zip(aoxz.iter_columns_full_mut())
    .zip(aoyy.iter_columns_full_mut())
    .zip(aoyz.iter_columns_full_mut())
    .zip(aozz.iter_columns_full_mut())
    .zip(r.iter())
    .map(
        |((((((((((ao, aox), aoy), aoz), aoxx), aoxy), aoxz), aoyy), aoyz), aozz), r)| 
        (ao, aox, aoy, aoz, aoxx, aoxy, aoxz, aoyy, aoyz, aozz, r)
    ).for_each(
        |(ao, aox, aoy, aoz, aoxx, aoxy, aoxz, aoyy, aoyz, aozz, r)| {
            // exp part 
            let mut rr = [0.0; 3];
            let rdot = izip!(
                rr.iter_mut(), r.iter(), c.iter()).fold(0.0, |rdot, (rr, r, c)| 
                {
                    *rr= r - c;
                    rdot + rr.powf(2.0_f64)
                }
            );
            let e = norm0 * libm::exp(-a*rdot);
            // polynomial part 
            // initialize the 0th order 
            fx0[0] = 1.0; fy0[0] = 1.0; fz0[0] = 1.0;
            // check if it should be ri - rA
            for lx in 1..=l+2 {
                fx0[lx] = fx0[lx-1] * rr[0];
                fy0[lx] = fy0[lx-1] * rr[1];
                fz0[lx] = fz0[lx-1] * rr[2];
            }
            // first order
            _gto_nabla1(&mut fx1, &mut fy1, &mut fz1, &fx0, &fy0, &fz0, l+1, a);
            // second order
            _gto_nabla1(&mut fx2, &mut fy2, &mut fz2, &fx1, &fy1, &fz1, l,   a);

            ao.iter_mut()
            .zip(aox.iter_mut())
            .zip(aoy.iter_mut())
            .zip(aoz.iter_mut())
            .zip(aoxx.iter_mut())
            .zip(aoxy.iter_mut())
            .zip(aoxz.iter_mut())
            .zip(aoyy.iter_mut())
            .zip(aoyz.iter_mut())
            .zip(aozz.iter_mut())
            .zip(basinfo.iter_columns_full())
            .map(
                |((((((((((ao_r, aox_r), aoy_r), aoz_r), aoxx_r), aoxy_r), aoxz_r), aoyy_r), aoyz_r), aozz_r), bas_i)|
                (ao_r, aox_r, aoy_r, aoz_r, aoxx_r, aoxy_r, aoxz_r, aoyy_r, aoyz_r, aozz_r, bas_i)
            )
            .for_each(|(ao_r, aox_r, aoy_r, aoz_r, aoxx_r, aoxy_r, aoxz_r, aoyy_r, aoyz_r, aozz_r, bas_i)| {
                let lx = bas_i[0] as usize;
                let ly = bas_i[1] as usize;
                let lz = bas_i[2] as usize;
                let norm = bas_i[3];
                *ao_r   = norm * e * fx0[lx] * fy0[ly] * fz0[lz];
                *aox_r  = norm * e * fx1[lx] * fy0[ly] * fz0[lz];
                *aoy_r  = norm * e * fx0[lx] * fy1[ly] * fz0[lz];
                *aoz_r  = norm * e * fx0[lx] * fy0[ly] * fz1[lz];
                *aoxx_r = norm * e * fx2[lx] * fy0[ly] * fz0[lz];
                *aoxy_r = norm * e * fx1[lx] * fy1[ly] * fz0[lz];
                *aoxz_r = norm * e * fx1[lx] * fy0[ly] * fz1[lz];
                *aoyy_r = norm * e * fx0[lx] * fy2[ly] * fz0[lz];
                *aoyz_r = norm * e * fx0[lx] * fy1[ly] * fz1[lz];
                *aozz_r = norm * e * fx0[lx] * fy0[ly] * fz2[lz];
            });
        }
    );
    vec![ao, aox, aoy, aoz, aoxx, aoxy, aoxz, aoyy, aoyz, aozz]
}

pub fn cartesian_gto_3rd_batch_v04(a: f64, l: usize, c: &[f64;3], r:&[[f64;3]]) -> Vec<MatrixFull<f64>> {
    let num_grids = r.len();
    let num_bas = (l+1)*(l+2)/2;
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    let cut_off:f64 = (constants::E7).log(constants::E)/a;

    let binding = cartesian_gto_const(l);
    let basinfo = &binding.to_matrixfullslice();

    let mut ao    = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aox   = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoy   = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoz   = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoxx  = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoxy  = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoxz  = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoyy  = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoyz  = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aozz  = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoxxx = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoxxy = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoxxz = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoxyy = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoxyz = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoxzz = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoyyy = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoyyz = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aoyzz = MatrixFull::new([num_bas, num_grids], 0.0);
    let mut aozzz = MatrixFull::new([num_bas, num_grids], 0.0);


    let mut fx0 = [0.0; 16];
    let mut fy0 = [0.0; 16];
    let mut fz0 = [0.0; 16];
    let mut fx1 = [0.0; 16];
    let mut fy1 = [0.0; 16];
    let mut fz1 = [0.0; 16];
    let mut fx2 = [0.0; 16];
    let mut fy2 = [0.0; 16];
    let mut fz2 = [0.0; 16];
    let mut fx3 = [0.0; 16];
    let mut fy3 = [0.0; 16];
    let mut fz3 = [0.0; 16];

    ao.iter_columns_full_mut()
    .zip(aox.iter_columns_full_mut())
    .zip(aoy.iter_columns_full_mut())
    .zip(aoz.iter_columns_full_mut())
    .zip(aoxx.iter_columns_full_mut())
    .zip(aoxy.iter_columns_full_mut())
    .zip(aoxz.iter_columns_full_mut())
    .zip(aoyy.iter_columns_full_mut())
    .zip(aoyz.iter_columns_full_mut())
    .zip(aozz.iter_columns_full_mut())
    .zip(aoxxx.iter_columns_full_mut())
    .zip(aoxxy.iter_columns_full_mut())
    .zip(aoxxz.iter_columns_full_mut())
    .zip(aoxyy.iter_columns_full_mut())
    .zip(aoxyz.iter_columns_full_mut())
    .zip(aoxzz.iter_columns_full_mut())
    .zip(aoyyy.iter_columns_full_mut())
    .zip(aoyyz.iter_columns_full_mut())
    .zip(aoyzz.iter_columns_full_mut())
    .zip(aozzz.iter_columns_full_mut())
    .zip(r.iter())
    .map(
        |((((((((((((((((((((
            ao, aox), aoy), aoz), 
            aoxx), aoxy), aoxz), aoyy), aoyz), aozz), 
            aoxxx), aoxxy), aoxxz), aoxyy), aoxyz), aoxzz), aoyyy), aoyyz), aoyzz), aozzz),
            r
        )| 
        (
            ao, aox, aoy, aoz, 
            aoxx, aoxy, aoxz, aoyy, aoyz, aozz, 
            aoxxx, aoxxy, aoxxz, aoxyy, aoxyz, aoxzz, aoyyy, aoyyz, aoyzz, aozzz, 
            r
        )
    ).for_each(
        |(ao, aox, aoy, aoz, aoxx, aoxy, aoxz, aoyy, aoyz, aozz, aoxxx, aoxxy, aoxxz, aoxyy, aoxyz, aoxzz, aoyyy, aoyyz, aoyzz, aozzz, r)|
        {
            // exp part
            let mut rr = [0.0; 3];
            let rdot = izip!(
                rr.iter_mut(), r.iter(), c.iter()).fold(0.0, |rdot, (rr, r, c)|
                {
                    *rr= r - c;
                    rdot + *rr*(*rr)
                }
            );
            let e = norm0 * libm::exp(-a*rdot);
            // polynomial part
            // initialize the 0th order
            fx0[0] = 1.0; fy0[0] = 1.0; fz0[0] = 1.0;
            // check if it should be ri - rA
            // NOTE: must initialize up to l+3 (not l+2) because the deepest _gto_nabla1
            // call for fx1 (with l_input=l+2) accesses fx0[l+3] when computing fx1[l+2].
            // For 2nd-order code (cartesian_gto_2nd_batch_v04), l+2 suffices since
            // the deepest call uses l_input=l+1 which only reads fx0[l+2].
            for lx in 1..=l+3 {
                fx0[lx] = fx0[lx-1] * rr[0];
                fy0[lx] = fy0[lx-1] * rr[1];
                fz0[lx] = fz0[lx-1] * rr[2];
            }
            // first order
            _gto_nabla1(&mut fx1, &mut fy1, &mut fz1, &fx0, &fy0, &fz0, l+2, a);
            // second order
            _gto_nabla1(&mut fx2, &mut fy2, &mut fz2, &fx1, &fy1, &fz1, l+1, a);
            // third order
            _gto_nabla1(&mut fx3, &mut fy3, &mut fz3, &fx2, &fy2, &fz2, l,   a);

            ao.iter_mut()
            .zip(aox.iter_mut())
            .zip(aoy.iter_mut())
            .zip(aoz.iter_mut())
            .zip(aoxx.iter_mut())
            .zip(aoxy.iter_mut())
            .zip(aoxz.iter_mut())
            .zip(aoyy.iter_mut())
            .zip(aoyz.iter_mut())
            .zip(aozz.iter_mut())
            .zip(aoxxx.iter_mut())
            .zip(aoxxy.iter_mut())
            .zip(aoxxz.iter_mut())
            .zip(aoxyy.iter_mut())
            .zip(aoxyz.iter_mut())
            .zip(aoxzz.iter_mut())
            .zip(aoyyy.iter_mut())
            .zip(aoyyz.iter_mut())
            .zip(aoyzz.iter_mut())
            .zip(aozzz.iter_mut())
            .zip(basinfo.iter_columns_full())
            .map(
                |((((((((((((((((((((ao_r, aox_r), aoy_r), aoz_r), aoxx_r), aoxy_r), aoxz_r), aoyy_r), aoyz_r), aozz_r), aoxxx_r), aoxxy_r), aoxxz_r), aoxyy_r), aoxyz_r), aoxzz_r), aoyyy_r), aoyyz_r), aoyzz_r), aozzz_r), bas_i)| (ao_r, aox_r, aoy_r, aoz_r, aoxx_r, aoxy_r, aoxz_r, aoyy_r, aoyz_r, aozz_r, aoxxx_r, aoxxy_r, aoxxz_r, aoxyy_r, aoxyz_r, aoxzz_r, aoyyy_r, aoyyz_r, aoyzz_r, aozzz_r, bas_i))
            .for_each(|(ao_r, aox_r, aoy_r, aoz_r, aoxx_r, aoxy_r, aoxz_r, aoyy_r, aoyz_r, aozz_r, aoxxx_r, aoxxy_r, aoxxz_r, aoxyy_r, aoxyz_r, aoxzz_r, aoyyy_r, aoyyz_r, aoyzz_r, aozzz_r, bas_i)|
            {
                let lx = bas_i[0] as usize;
                let ly = bas_i[1] as usize;
                let lz = bas_i[2] as usize;
                let norm = bas_i[3];
                *ao_r    = norm * e * fx0[lx] * fy0[ly] * fz0[lz];
                *aox_r   = norm * e * fx1[lx] * fy0[ly] * fz0[lz];
                *aoy_r   = norm * e * fx0[lx] * fy1[ly] * fz0[lz];
                *aoz_r   = norm * e * fx0[lx] * fy0[ly] * fz1[lz];
                *aoxx_r  = norm * e * fx2[lx] * fy0[ly] * fz0[lz];
                *aoxy_r  = norm * e * fx1[lx] * fy1[ly] * fz0[lz];
                *aoxz_r  = norm * e * fx1[lx] * fy0[ly] * fz1[lz];
                *aoyy_r  = norm * e * fx0[lx] * fy2[ly] * fz0[lz];
                *aoyz_r  = norm * e * fx0[lx] * fy1[ly] * fz1[lz];
                *aozz_r  = norm * e * fx0[lx] * fy0[ly] * fz2[lz];
                *aoxxx_r = norm * e * fx3[lx] * fy0[ly] * fz0[lz];
                *aoxxy_r = norm * e * fx2[lx] * fy1[ly] * fz0[lz];
                *aoxxz_r = norm * e * fx2[lx] * fy0[ly] * fz1[lz];
                *aoxyy_r = norm * e * fx1[lx] * fy2[ly] * fz0[lz];
                *aoxyz_r = norm * e * fx1[lx] * fy1[ly] * fz1[lz];
                *aoxzz_r = norm * e * fx1[lx] * fy0[ly] * fz2[lz];
                *aoyyy_r = norm * e * fx0[lx] * fy3[ly] * fz0[lz];
                *aoyyz_r = norm * e * fx0[lx] * fy2[ly] * fz1[lz];
                *aoyzz_r = norm * e * fx0[lx] * fy1[ly] * fz2[lz];
                *aozzz_r = norm * e * fx0[lx] * fy0[ly] * fz3[lz];
            });
        }
    );
    vec![ao, aox, aoy, aoz, aoxx, aoxy, aoxz, aoyy, aoyz, aozz, aoxxx, aoxxy, aoxxz, aoxyy, aoxyz, aoxzz, aoyyy, aoyyz, aoyzz, aozzz]
}

pub fn cartesian_gto_1st_std(a: f64, l: usize, c:&[f64;3],r:&[f64;3]) -> [MatrixFull<f64>;3] {
    let mut gto_dx: Vec<f64> = vec![];
    let mut gto_dy: Vec<f64> = vec![];
    let mut gto_dz: Vec<f64> = vec![];
    let mut rr:[f64;3] = r.clone();
    rr.iter_mut().zip(c.iter()).for_each(|(r,c)| *r -= c);
    let mut rdot = rr.iter().fold(0.0f64,|acc,rr| acc + rr*rr);
    let exp_part = libm::exp(-a*rdot);
    let norm0 = (2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0);
    for lx in (0..=l as i32).rev() {
        let lx_f64 = lx as f64;
        let xp = rr[0].powf(lx_f64);
        let mut xp_1s = -2.0*a*rr[0].powf(lx_f64+1.0);
        xp_1s += if (lx != 0) {lx_f64*rr[0].powf(lx_f64-1.0)} else {0.0};
        //debug here
        let normx = double_factorial(2*lx-1) as f64;
        let rl = l as i32 - lx;
        for ly in (0..=rl as i32).rev() {
            let lz = rl - ly;
            let ly_f64 = ly as f64;
            let lz_f64 = lz as f64;
            let normy = double_factorial(2*ly-1) as f64;
            let normz = double_factorial(2*lz-1) as f64;
            let norm = norm0/(normx*normy*normz).sqrt();

            let yp = rr[1].powf(ly_f64);
            let mut yp_1s = -2.0*a*rr[1].powf(ly_f64+1.0);
            yp_1s += if (ly != 0) {ly_f64*rr[1].powf(ly_f64-1.0)} else {0.0};
            let zp = rr[2].powf(lz_f64);
            let mut zp_1s = -2.0*a*rr[2].powf(lz_f64+1.0);
            zp_1s += if (lz != 0)  {lz_f64*rr[2].powf(lz_f64-1.0)} else {0.0};

            //tmp_v.push(
            //        xp*rr[1].powf(ly as f64)*rr[2].powf(lz as f64)
            //        *exp_part);
            gto_dx.push(norm*xp_1s*yp*zp*exp_part);
            gto_dy.push(norm*xp*yp_1s*zp*exp_part);
            gto_dz.push(norm*xp*yp*zp_1s*exp_part);
            //println!("debug norm 2: {}", norm0/(normx*normy*normz).sqrt());
            //println!("debug: {},{},{}", xp_1s, yp_1s, zp_1s);
            //println!("yp_1s comp: ly_f64: {}, rr[1]: {}, a: {}", ly_f64, rr[1], a);
            //println!("yp_1s comp: ly_f64: {}, rr[1]: {}, a: {}", ly_f64, rr[1].powf(ly_f64-1.0), a);
        }
    }
    //println!("debug: gto_d(x,y,z): ({:?}, {:?}, {:?})", &gto_dx, &gto_dy, &gto_dz);
    [unsafe{MatrixFull::from_vec_unchecked([(l+1)*(l+2)/2,1], gto_dx)},
     unsafe{MatrixFull::from_vec_unchecked([(l+1)*(l+2)/2,1], gto_dy)},
     unsafe{MatrixFull::from_vec_unchecked([(l+1)*(l+2)/2,1], gto_dz)}]

}

pub fn cartesian_gto_cint(a: f64, l: usize, c:&[f64;3],r:&[f64;3]) -> MatrixFull<f64> {
    // WARNNING: In libcint, the normalization of radial part should be defined in coefficients.
    //           In other words, the native GTO integrals from libcint are not normlized with respect to "\int dr^3 r^2 g(r)"
    //           In consequence, 
    let mut std_gto = cartesian_gto_std(a, l, c, r);
    std_gto.self_multiple(cint_norm_factor(l as i32, a));
    std_gto
}

pub fn cartesian_gto_cint_batch_rayon(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> MatrixFull<f64> {
    // WARNNING: In libcint, the normalization of radial part should be defined in coefficients.
    //           In other words, the native GTO integrals from libcint are not normlized with respect to "\int dr^3 r^2 g(r)"
    //           In consequence, 
    //let mut time_records = utilities::TimeRecords::new();

    //time_records.new_item("1-1-1", "cartesian_gto_batch_par");
    //time_records.count_start("1-1-1");
    let mut std_gto = cartesian_gto_batch_par_v03(a, l, c, r);

    //time_records.new_item("1-1-2", "self_multiple/norm_factor");
    //time_records.count("1-1-1");
    //time_records.count_start("1-1-2");
    std_gto.self_multiple(cint_norm_factor(l as i32, a));
    //time_records.count("1-1-2");

    //time_records.report_all();

    std_gto
}
pub fn cartesian_gto_cint_batch(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> MatrixFull<f64> {
    // WARNNING: In libcint, the normalization of radial part should be defined in coefficients.
    //           In other words, the native GTO integrals from libcint are not normlized with respect to "\int dr^3 r^2 g(r)"
    //           In consequence, 
    let mut std_gto = cartesian_gto_batch_v03(a, l, c, r);

    std_gto.self_multiple(cint_norm_factor(l as i32, a));

    std_gto
}

pub fn cartesian_gto_cint_batch_v04(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> MatrixFull<f64> {
    // WARNNING: In libcint, the normalization of radial part should be defined in coefficients.
    //           In other words, the native GTO integrals from libcint are not normlized with respect to "\int dr^3 r^2 g(r)"
    //           It prepares the tabulated cartesian gto which will be used for SPHERICAL gto defined in libcint
    let mut std_gto = cartesian_gto_batch_v04(a, l, c, r);

    std_gto.self_multiple(cint_norm_factor(l as i32, a));

    std_gto
}

pub fn cint_cartesian_gto_cint_batch_v04(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> MatrixFull<f64> {
    // WARNNING: In libcint, only s and p type gtos are normalized.
    //           For gtos with l>=2, the normalization is not defined for different combination of lx, ly, and lz.
    //           In consequence, the density matrix and orbital coeffiencts represented by the cartesian GTOs with libcint are not normalized.
    //           In order to meet this requirement, the density tabulation by REST should be non-normalized as well.
    //           It is the reason we have two subrutines to prepare the cartesian_gto_batch_v04.
    //           This subroutine prepares the tabulated cartesian gto which will be used directly later with the CARTESIAN gto defined in libcint
    let mut std_gto = cint_cartesian_gto_batch_v04(a, l, c, r);

    std_gto.self_multiple(cint_cart_norm_factor(l as i32, a));

    std_gto
}

pub fn cartesian_gto_1st_cint(a: f64, l: usize, c:&[f64;3],r:&[f64;3]) -> [MatrixFull<f64>;3] {
    // WARNNING: In libcint, the normalization of radial part should be defined in coefficients.
    //           In other words, the native GTO integrals from libcint are not normlized with respect to "\int dr^3 r^2 g(r)"
    //           In consequence, 
    let mut std_gto_1st = cartesian_gto_1st_std(a, l, c, r);
    //println!("debug: {:?}, {:?}, {:?}", &std_gto_1st[0].data,&std_gto_1st[1].data,&std_gto_1st[2].data);
    std_gto_1st.iter_mut().for_each(|std_gto| {
        std_gto.self_multiple(cint_norm_factor(l as i32, a))
    });
    std_gto_1st
}

pub fn cartesian_gto_1st_cint_batch(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> Vec<MatrixFull<f64>> {
    // WARNNING: In libcint, the normalization of radial part should be defined in coefficients.
    //           In other words, the native GTO integrals from libcint are not normlized with respect to "\int dr^3 r^2 g(r)"
    //           In consequence, 
    let mut std_gto_1st = cartesian_gto_1st_batch_par_v02(a, l, c, r);
    //println!("debug: {:?}, {:?}, {:?}", &std_gto_1st[0].data,&std_gto_1st[1].data,&std_gto_1st[2].data);
    let fac = cint_norm_factor(l as i32, a);
    //std_gto_1st.data.iter_mut().for_each(|value| {*value *=fac});
    std_gto_1st.iter_mut().for_each(|std_gto| {
        std_gto.self_multiple(fac)
    });
    std_gto_1st
}

pub fn cartesian_gto_1st_cint_batch_serial(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> Vec<MatrixFull<f64>> {
    // WARNNING: In libcint, the normalization of radial part should be defined in coefficients.
    //           In other words, the native GTO integrals from libcint are not normlized with respect to "\int dr^3 r^2 g(r)"
    //           In consequence, 
    let mut std_gto_1st = cartesian_gto_1st_batch_v02(a, l, c, r);
    //println!("debug: {:?}, {:?}, {:?}", &std_gto_1st[0].data,&std_gto_1st[1].data,&std_gto_1st[2].data);
    let fac = cint_norm_factor(l as i32, a);
    //std_gto_1st.data.iter_mut().for_each(|value| {*value *=fac});
    std_gto_1st.iter_mut().for_each(|std_gto| {
        std_gto.self_multiple(fac)
    });
    std_gto_1st
}

///MARK by IGOR
pub fn cartesian_gto_1st_cint_batch_serial_v03(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> Vec<MatrixFull<f64>> {
    // WARNNING: In libcint, the normalization of radial part should be defined in coefficients.
    //           In other words, the native GTO integrals from libcint are not normlized with respect to "\int dr^3 r^2 g(r)"
    //           In consequence, 
    let mut std_gto_1st = cartesian_gto_1st_batch_v03(a, l, c, r);
    //println!("debug: {:?}, {:?}, {:?}", &std_gto_1st[0].data,&std_gto_1st[1].data,&std_gto_1st[2].data);
    let fac = cint_norm_factor(l as i32, a);
    //std_gto_1st.data.iter_mut().for_each(|value| {*value *=fac});
    std_gto_1st.iter_mut().for_each(|std_gto| {
        std_gto.self_multiple(fac)
    });
    std_gto_1st
}

pub fn cint_cartesian_gto_1st_cint_batch_serial_v03(a: f64, l: usize, c:&[f64;3],r:&[[f64;3]]) -> Vec<MatrixFull<f64>> {
    // WARNNING: In libcint, only s and p type gtos are normalized.
    //           For gtos with l>=2, the normalization is not defined for different combination of lx, ly, and lz.
    //           In consequence, the density matrix and orbital coeffiencts represented by the cartesian GTOs with libcint are not normalized.
    //           In order to meet this requirement, the density tabulation by REST should be non-normalized as well.
    //           It is the reason we have two subrutines to prepare the cartesian_gto_batch_v04.
    //           This subroutine prepares the tabulated cartesian gto which will be used directly later with the CARTESIAN gto defined in libcint
    let mut std_gto_1st = cint_cartesian_gto_1st_batch_v03(a, l, c, r);
    let fac = cint_cart_norm_factor(l as i32, a);
    //std_gto_1st.data.iter_mut().for_each(|value| {*value *=fac});
    std_gto_1st.iter_mut().for_each(|std_gto| {
        std_gto.self_multiple(fac)
    });
    std_gto_1st
}
