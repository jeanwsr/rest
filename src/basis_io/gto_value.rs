//! Basis-set-level GTO evaluators: value / first derivative / arbitrary-order
//! derivative of spherical and cartesian GTOs on grids, taking `Basis4Elem`,
//! plus cint-style normalization factors.
use rest_tensors::MatrixFull;
use rayon::prelude::*;
use itertools::{Itertools, izip};
use tensors::{MathMatrix, ParMathMatrix};
use tensors::matrix_blas_lapack::{_dgemm, _dgemm_nn, _dgemm_nn_serial};
use std::convert::TryInto;
use std::f64::consts::{PI, E};
use libm;
use crate::utilities::basic_math::opowf;
use crate::constants::{C2S_L0, C2S_L1, C2S_L2, C2S_L3, C2S_L4, CarBasInfo, c2s_matrix_const, cartesian_gto_const, cint_cartesian_gto_const};
use crate::utilities;
use super::Basis4Elem;
use super::{
    cartesian_gto_1st_batch_v04, cartesian_gto_1st_cint_batch,
    cartesian_gto_1st_cint_batch_serial, cartesian_gto_1st_cint_batch_serial_v03,
    cartesian_gto_2nd_batch_v04, cartesian_gto_3rd_batch_v04, cartesian_gto_batch_v04,
    cartesian_gto_cint_batch, cartesian_gto_cint_batch_rayon,
    cartesian_gto_cint_batch_v04, cint_cartesian_gto_1st_cint_batch_serial_v03,
    cint_cartesian_gto_cint_batch_v04,
};

pub fn spheric_gto_value_matrixfull(r: &[[f64;3]],gto_center:&[f64;3], bas:&Basis4Elem) -> MatrixFull<f64> {

    //let mut time_records = utilities::TimeRecords::new();
    //time_records.new_item("1-1", "cartesian_gto_cint_batch");
    //time_records.new_item("1-2", "_dgemm_nn");

    let num_grids:usize = r.len();
    let mut num_bas_c = 0;
    let mut num_bas_s = 0;
    bas.electron_shells.iter().for_each(|ibas| {
        let iang = ibas.angular_momentum[0] as usize;
        num_bas_c += (iang+1)*(iang+2)/2*ibas.coefficients.len();
        num_bas_s += (iang*2+1)*ibas.coefficients.len();
    });
    //println!("debug num_bas_c and num_bas_s: {},{}", num_bas_c, num_bas_s);
    //let mut outmat_c = MatrixFull::new([num_grids,num_bas_c],0.0);
    // shells tile all columns via the copy below: no zeroing needed
    let mut outmat_s = uninit_matrix(num_grids, num_bas_s);
    let mut ibas_start = 0;
    for ibas in &bas.electron_shells {
        let iang = ibas.angular_momentum[0] as usize;
        //let mut c2s_mat = &mut c2s_matrix(iang);
        let c2s_mat = &c2s_matrix_const(iang);
        let s_len = 2*iang + 1;
        let c_len = (iang+1)*(iang+2)/2;
        for coeff in ibas.coefficients.iter() {
            let mut tmp_cart = MatrixFull::new([num_grids,c_len],0.0);
            coeff.iter().zip(ibas.exponents.iter()).for_each(|(icoeff,iexp)| {

                //time_records.count_start("1-1");
                let mut tmp_cart_0 = cartesian_gto_cint_batch_rayon(*iexp, iang, gto_center, r);
                //time_records.count("1-1");

                tmp_cart.par_self_scaled_add(&tmp_cart_0, *icoeff);
            });

            //time_records.count_start("1-2");
            // C[num_grids, s_len] = tmp_cart[num_grids, c_len] x c2s[c_len, s_len]
            let mut tmp_spheric = uninit_matrix(num_grids, s_len);
            _dgemm(&tmp_cart, (0..num_grids, 0..c_len), 'N',
                   &c2s_mat.to_matrixfullslice(), (0..c_len, 0..s_len), 'N',
                   &mut tmp_spheric, (0..num_grids, 0..s_len), 1.0, 0.0);
            //time_records.count("1-2");

            outmat_s.iter_columns_mut(ibas_start..ibas_start+s_len)
            .zip(tmp_spheric.iter_columns_full()).for_each(|(to,from)| {
                to.par_iter_mut().zip(from.par_iter()).for_each(|(to,from)| {*to = *from});
            });
            ibas_start += s_len;
        }
    }

    //time_records.report_all();

    outmat_s
}

pub fn gto_value_matrixfull_serial(r: &[[f64;3]],gto_center:&[f64;3], bas:&Basis4Elem, basis_type: &String) -> MatrixFull<f64> { 
    if basis_type.eq(&"spheric".to_string()) {
        return spheric_gto_value_matrixfull_serial(r, gto_center, bas);
    } else {
        return cartesian_gto_value_matrixfull_serial(r, gto_center, bas);
    } 
}

pub fn spheric_gto_value_matrixfull_serial(r: &[[f64;3]],gto_center:&[f64;3], bas:&Basis4Elem) -> MatrixFull<f64> {

    //let mut time_records = utilities::TimeRecords::new();
    //time_records.new_item("1-1", "cartesian_gto_cint_batch");
    //time_records.new_item("1-2", "_dgemm_nn");

    let num_grids:usize = r.len();
    let mut num_bas_c = 0;
    let mut num_bas_s = 0;
    bas.electron_shells.iter().for_each(|ibas| {
        let iang = ibas.angular_momentum[0] as usize;
        num_bas_c += (iang+1)*(iang+2)/2*ibas.coefficients.len();
        num_bas_s += (iang*2+1)*ibas.coefficients.len();
    });
    //println!("debug num_bas_c and num_bas_s: {},{}", num_bas_c, num_bas_s);
    //let mut outmat_c = MatrixFull::new([num_grids,num_bas_c],0.0);
    // shells tile all columns via the copy below: no zeroing needed
    let mut outmat_s = uninit_matrix(num_grids, num_bas_s);
    let mut ibas_start = 0;
    for ibas in &bas.electron_shells {
        let iang = ibas.angular_momentum[0] as usize;
        //let mut c2s_mat = &mut c2s_matrix(iang);
        let c2s_mat = &c2s_matrix_const(iang);
        let s_len = 2*iang + 1;
        let c_len = (iang+1)*(iang+2)/2;
        for coeff in ibas.coefficients.iter() {
            let mut tmp_cart = MatrixFull::new([num_grids,c_len],0.0);
            coeff.iter().zip(ibas.exponents.iter()).for_each(|(icoeff,iexp)| {

                //time_records.count_start("1-1");
                let mut tmp_cart_0 = cartesian_gto_cint_batch(*iexp, iang, gto_center, r);
                //time_records.count("1-1");

                tmp_cart.self_scaled_add(&tmp_cart_0, *icoeff);
            });

            //time_records.count_start("1-2");
            // C[num_grids, s_len] = tmp_cart[num_grids, c_len] x c2s[c_len, s_len]
            let mut tmp_spheric = uninit_matrix(num_grids, s_len);
            _dgemm(&tmp_cart, (0..num_grids, 0..c_len), 'N',
                   &c2s_mat.to_matrixfullslice(), (0..c_len, 0..s_len), 'N',
                   &mut tmp_spheric, (0..num_grids, 0..s_len), 1.0, 0.0);
            //time_records.count("1-2");

            outmat_s.iter_columns_mut(ibas_start..ibas_start+s_len)
            .zip(tmp_spheric.iter_columns_full()).for_each(|(to,from)| {
                to.iter_mut().zip(from.iter()).for_each(|(to,from)| {*to = *from});
            });
            ibas_start += s_len;
        }
    }
    //time_records.report_all();

    outmat_s
}

pub fn cartesian_gto_value_matrixfull_serial(r: &[[f64;3]],gto_center:&[f64;3], bas:&Basis4Elem) -> MatrixFull<f64> {

    //let mut time_records = utilities::TimeRecords::new();
    //time_records.new_item("1-1", "cartesian_gto_cint_batch");
    //time_records.new_item("1-2", "_dgemm_nn");

    let num_grids:usize = r.len();
    let mut num_bas_c = 0;
    let mut num_bas_s = 0;
    bas.electron_shells.iter().for_each(|ibas| {
        let iang = ibas.angular_momentum[0] as usize;
        num_bas_c += (iang+1)*(iang+2)/2*ibas.coefficients.len();
        num_bas_s += (iang*2+1)*ibas.coefficients.len();
    });
    //println!("debug num_bas_c and num_bas_s: {},{}", num_bas_c, num_bas_s);
    let mut outmat_c = MatrixFull::new([num_grids,num_bas_c],0.0);
    //let mut outmat_s = MatrixFull::new([num_grids,num_bas_s],0.0);
    let mut ibas_start = 0;
    for ibas in &bas.electron_shells {
        let iang = ibas.angular_momentum[0] as usize;
        //let mut c2s_mat = &mut c2s_matrix(iang);
        let c2s_mat = &c2s_matrix_const(iang);
        let s_len = 2*iang + 1;
        let c_len = (iang+1)*(iang+2)/2;
        for coeff in ibas.coefficients.iter() {
            let mut tmp_cart = MatrixFull::new([num_grids,c_len],0.0);
            coeff.iter().zip(ibas.exponents.iter()).for_each(|(icoeff,iexp)| {

                //time_records.count_start("1-1");
                let mut tmp_cart_0 = cartesian_gto_cint_batch(*iexp, iang, gto_center, r);
                //time_records.count("1-1");

                tmp_cart.self_scaled_add(&tmp_cart_0, *icoeff);
            });

            //time_records.count_start("1-2");
            //let tmp_spheric = _dgemm_nn_serial(&tmp_cart.to_matrixfullslice(), &c2s_mat.to_matrixfullslice());
            //time_records.count("1-2");

            outmat_c.iter_columns_mut(ibas_start..ibas_start+c_len)
            .zip(tmp_cart.iter_columns_full()).for_each(|(to,from)| {
                to.iter_mut().zip(from.iter()).for_each(|(to,from)| {*to = *from});
            });
            ibas_start += c_len;
        }
    }
    //time_records.report_all();

    outmat_c
}

#[inline]
fn uninit_matrix(rows: usize, cols: usize) -> MatrixFull<f64> {
    let n = rows * cols;
    let mut buf = Vec::with_capacity(n);
    unsafe { buf.set_len(n); }
    unsafe { MatrixFull::from_vec_unchecked([rows, cols], buf) }
}

pub fn gto_value_serial(r: &[[f64;3]],gto_center:&[f64;3], bas:&Basis4Elem, basis_type: &String) -> MatrixFull<f64> { 
    if basis_type.eq(&"spheric".to_string()) {
        return spheric_gto_value_serial(r, gto_center, bas);
    } else {
        return cartesian_gto_value_serial(r, gto_center, bas);
    } 
}

pub fn spheric_gto_value_serial(r: &[[f64;3]],gto_center:&[f64;3], bas:&Basis4Elem) -> MatrixFull<f64> {

    //let mut time_records = utilities::TimeRecords::new();
    //time_records.new_item("1-1", "cartesian_gto_cint_batch");
    //time_records.new_item("1-2", "_dgemm_nn");

    let num_grids:usize = r.len();
    let mut num_bas_c = 0;
    let mut num_bas_s = 0;
    bas.electron_shells.iter().for_each(|ibas| {
        let iang = ibas.angular_momentum[0] as usize;
        num_bas_c += (iang+1)*(iang+2)/2*ibas.coefficients.len();
        num_bas_s += (iang*2+1)*ibas.coefficients.len();
    });
    //println!("debug num_bas_c and num_bas_s: {},{}", num_bas_c, num_bas_s);
    //let mut outmat_c = MatrixFull::new([num_grids,num_bas_c],0.0);
    // shells tile the full [num_bas_s, num_grids] range via copy_from: no zeroing
    let mut outmat_s = uninit_matrix(num_bas_s, num_grids);
    let mut ibas_start = 0;
    for ibas in &bas.electron_shells {
        let iang = ibas.angular_momentum[0] as usize;
        //let mut c2s_mat = &mut c2s_matrix(iang);
        let c2s_mat = &c2s_matrix_const(iang);
        let s_len = 2*iang + 1;
        let c_len = (iang+1)*(iang+2)/2;
        for coeff in ibas.coefficients.iter() {
            let mut tmp_cart = MatrixFull::new([c_len,num_grids],0.0);
            coeff.iter().zip(ibas.exponents.iter()).for_each(|(icoeff,iexp)| {
                tmp_cart += cartesian_gto_cint_batch_v04(*iexp, iang, gto_center, r)* (*icoeff);
            });

            //time_records.count_start("1-2");
            let mut tmp_spheric = uninit_matrix(s_len, num_grids);
            // C[s_len, num_grids] = c2s[c_len, s_len]^T x tmp_cart[c_len, num_grids]
            _dgemm(&c2s_mat.to_matrixfullslice(), (0..c_len, 0..s_len), 'T',
                   &tmp_cart, (0..c_len, 0..num_grids), 'N',
                   &mut tmp_spheric, (0..s_len, 0..num_grids), 1.0, 0.0);

            outmat_s.copy_from_matr(ibas_start..ibas_start+s_len, 0..num_grids, 
                &tmp_spheric, 0..s_len, 0..num_grids);

            ibas_start += s_len;
        }
    }
    //time_records.report_all();

    outmat_s
}

pub fn cartesian_gto_value_serial(r: &[[f64;3]],gto_center:&[f64;3], bas:&Basis4Elem) -> MatrixFull<f64> {

    let num_grids:usize = r.len();
    let mut num_bas_c = 0;
    let mut num_bas_s = 0;
    bas.electron_shells.iter().for_each(|ibas| {
        let iang = ibas.angular_momentum[0] as usize;
        num_bas_c += (iang+1)*(iang+2)/2*ibas.coefficients.len();
        num_bas_s += (iang*2+1)*ibas.coefficients.len();
    });
    let mut outmat_c = MatrixFull::new([num_bas_c, num_grids],0.0);
    let mut ibas_start = 0;
    for ibas in &bas.electron_shells {
        let iang = ibas.angular_momentum[0] as usize;
        let c_len = (iang+1)*(iang+2)/2;
        // libcint cartesian convention for every l (the l<2 / l>=2 split the
        // historical if/else duplicated is already inside the evaluator)
        for coeff in ibas.coefficients.iter() {
            let mut tmp_cart = MatrixFull::new([c_len,num_grids],0.0);
            coeff.iter().zip(ibas.exponents.iter()).for_each(|(icoeff,iexp)| {
                    tmp_cart += cint_cartesian_gto_cint_batch_v04(*iexp, iang, gto_center, r)* (*icoeff);
            });

            outmat_c.copy_from_matr(ibas_start..ibas_start+c_len, 0..num_grids, 
                &tmp_cart, 0..c_len, 0..num_grids);
            ibas_start += c_len;
        }
    }

    outmat_c
}

pub fn gto_value(r:&[f64;3], gto_center:&[f64;3], bas:&Basis4Elem, basis_type: &String) -> Vec<f64> {
    // Single-point convenience wrapper around the batch tabulation. Both
    // branches (spheric / cartesian) must use the SAME normalization chain as
    // `gto_value_serial` so that the values are consistent with the libcint
    // integrals (the historical hand-rolled version applied the spheric-chain
    // normalization on the cartesian branch, which is wrong for l >= 2; see
    // AO_AOP_NORM_AUDIT.md §5-D).
    gto_value_serial(&[r.clone()], gto_center, bas, basis_type).data
}
pub fn spheric_gto_1st_value_batch(r:&[[f64;3]], gto_center:&[f64;3], bas:&Basis4Elem) -> Vec<MatrixFull<f64>> {
    let num_grids:usize = r.len();
    let mut num_bas_c = 0;
    let mut num_bas_s = 0;
    bas.electron_shells.iter().for_each(|ibas| {
        let iang = ibas.angular_momentum[0] as usize;
        num_bas_c += (iang+1)*(iang+2)/2*ibas.coefficients.len();
        num_bas_s += (iang*2+1)*ibas.coefficients.len();
    });
    //let mut paox = RIFull::new([num_grids,num_bas_s,3],0.0);
    let mut paox = vec![MatrixFull::new([num_grids,num_bas_s],0.0);3];
    //let mut paox = MatrixFull::new([num_grids,num_bas_s],0.0);
    //let mut aopy = MatrixFull::new([num_grids,num_bas_s],0.0);
    //let mut aopz = MatrixFull::new([num_grids,num_bas_s],0.0);
    let mut ibas_start = 0;
    for ibas in &bas.electron_shells {
        let iang = ibas.angular_momentum[0] as usize;
        //let mut c2s_mat = &mut c2s_matrix(iang);
        let mut c2s_mat = &c2s_matrix_const(iang);
        let s_len = 2*iang + 1;
        let c_len = (iang+1)*(iang+2)/2;
        for coeff in ibas.coefficients.iter() {
            let mut tmp_cart = vec![MatrixFull::new([num_grids,c_len],0.0);3];
            coeff.iter().zip(ibas.exponents.iter()).for_each(|(icoeff,iexp)| {
                let mut tmp_cart_0 = cartesian_gto_1st_cint_batch(*iexp, iang, gto_center, r);

                //tmp_cart.self_scaled_add(&tmp_cart_0, *icoeff);
                tmp_cart.iter_mut().zip(tmp_cart_0.iter()).for_each(|(to,from)| {
                    to.par_self_scaled_add(from, *icoeff);
                });
            });
            //let mut tmp_spheric = MatrixFull::new([num_grids,s_len],0.0);
            paox.iter_mut().zip(tmp_cart.iter_mut()).for_each(|(paox_x,tmp_cart_x)| {
                //tmp_spheric.lapack_dgemm(tmp_cart_x, &mut c2s_mat, 'N','N',1.0,0.0);
                // C[num_grids, s_len] = tmp_cart_x[num_grids, c_len] x c2s[c_len, s_len]
                let mut tmp_spheric = uninit_matrix(num_grids, s_len);
                _dgemm(&*tmp_cart_x, (0..num_grids, 0..c_len), 'N',
                       &c2s_mat.to_matrixfullslice(), (0..c_len, 0..s_len), 'N',
                       &mut tmp_spheric, (0..num_grids, 0..s_len), 1.0, 0.0);
                paox_x.iter_columns_mut(ibas_start..ibas_start+s_len)
                .zip(tmp_spheric.iter_columns_full()).for_each(|(to,from)| {
                    to.par_iter_mut().zip(from.par_iter()).for_each(|(to,from)| {*to = *from});
                });

            });
            ibas_start += s_len;
        }
    }
    paox
}
pub fn gto_1st_value_batch_serial(r:&[[f64;3]], gto_center:&[f64;3], bas:&Basis4Elem, basis_type:&String) -> Vec<MatrixFull<f64>> {
    if basis_type.eq(&"spheric".to_string()) {
        return spheric_gto_1st_value_batch_serial(r, gto_center, bas)
    } else {
        return cartesian_gto_1st_value_batch_serial(r, gto_center, bas)
    } 
}
pub fn spheric_gto_1st_value_batch_serial(r:&[[f64;3]], gto_center:&[f64;3], bas:&Basis4Elem) -> Vec<MatrixFull<f64>> {
    let num_grids:usize = r.len();
    let mut num_bas_c = 0;
    let mut num_bas_s = 0;
    bas.electron_shells.iter().for_each(|ibas| {
        let iang = ibas.angular_momentum[0] as usize;
        num_bas_c += (iang+1)*(iang+2)/2*ibas.coefficients.len();
        num_bas_s += (iang*2+1)*ibas.coefficients.len();
    });
    //let mut paox = RIFull::new([num_grids,num_bas_s,3],0.0);
    let mut paox = vec![MatrixFull::new([num_grids,num_bas_s],0.0);3];
    //let mut paox = MatrixFull::new([num_grids,num_bas_s],0.0);
    //let mut aopy = MatrixFull::new([num_grids,num_bas_s],0.0);
    //let mut aopz = MatrixFull::new([num_grids,num_bas_s],0.0);
    let mut ibas_start = 0;
    for ibas in &bas.electron_shells {
        let iang = ibas.angular_momentum[0] as usize;
        //let mut c2s_mat = &mut c2s_matrix(iang);
        let mut c2s_mat = &c2s_matrix_const(iang);
        let s_len = 2*iang + 1;
        let c_len = (iang+1)*(iang+2)/2;
        for coeff in ibas.coefficients.iter() {
            let mut tmp_cart = vec![MatrixFull::new([num_grids,c_len],0.0);3];
            coeff.iter().zip(ibas.exponents.iter()).for_each(|(icoeff,iexp)| {
                let mut tmp_cart_0 = cartesian_gto_1st_cint_batch_serial(*iexp, iang, gto_center, r);

                //tmp_cart.self_scaled_add(&tmp_cart_0, *icoeff);
                tmp_cart.iter_mut().zip(tmp_cart_0.iter()).for_each(|(to,from)| {
                    to.self_scaled_add(from, *icoeff);
                });
            });
            //let mut tmp_spheric = MatrixFull::new([num_grids,s_len],0.0);
            paox.iter_mut().zip(tmp_cart.iter_mut()).for_each(|(paox_x,tmp_cart_x)| {
                //tmp_spheric.lapack_dgemm(tmp_cart_x, &mut c2s_mat, 'N','N',1.0,0.0);
                // C[num_grids, s_len] = tmp_cart_x[num_grids, c_len] x c2s[c_len, s_len]
                let mut tmp_spheric = uninit_matrix(num_grids, s_len);
                _dgemm(&*tmp_cart_x, (0..num_grids, 0..c_len), 'N',
                       &c2s_mat.to_matrixfullslice(), (0..c_len, 0..s_len), 'N',
                       &mut tmp_spheric, (0..num_grids, 0..s_len), 1.0, 0.0);
                paox_x.iter_columns_mut(ibas_start..ibas_start+s_len)
                .zip(tmp_spheric.iter_columns_full()).for_each(|(to,from)| {
                    to.iter_mut().zip(from.iter()).for_each(|(to,from)| {*to = *from});
                });

            });
            ibas_start += s_len;
        }
    }
    paox
}

pub fn cartesian_gto_1st_value_batch_serial(r:&[[f64;3]], gto_center:&[f64;3], bas:&Basis4Elem) -> Vec<MatrixFull<f64>> {
    let num_grids:usize = r.len();
    let mut num_bas_c = 0;
    let mut num_bas_s = 0;
    bas.electron_shells.iter().for_each(|ibas| {
        let iang = ibas.angular_momentum[0] as usize;
        num_bas_c += (iang+1)*(iang+2)/2*ibas.coefficients.len();
        num_bas_s += (iang*2+1)*ibas.coefficients.len();
    });
    //let mut paox = RIFull::new([num_grids,num_bas_s,3],0.0);
    //let mut paox = vec![MatrixFull::new([num_grids,num_bas_s],0.0);3];
    let mut paox = vec![MatrixFull::new([num_grids,num_bas_c],0.0);3];
    //let mut paox = MatrixFull::new([num_grids,num_bas_s],0.0);
    //let mut aopy = MatrixFull::new([num_grids,num_bas_s],0.0);
    //let mut aopz = MatrixFull::new([num_grids,num_bas_s],0.0);
    let mut ibas_start = 0;
    for ibas in &bas.electron_shells {
        let iang = ibas.angular_momentum[0] as usize;
        //let mut c2s_mat = &mut c2s_matrix(iang);
        let mut c2s_mat = &c2s_matrix_const(iang);
        let s_len = 2*iang + 1;
        let c_len = (iang+1)*(iang+2)/2;
        for coeff in ibas.coefficients.iter() {
            let mut tmp_cart = vec![MatrixFull::new([num_grids,c_len],0.0);3];
            coeff.iter().zip(ibas.exponents.iter()).for_each(|(icoeff,iexp)| {
                let mut tmp_cart_0 = cartesian_gto_1st_cint_batch_serial(*iexp, iang, gto_center, r);

                //tmp_cart.self_scaled_add(&tmp_cart_0, *icoeff);
                tmp_cart.iter_mut().zip(tmp_cart_0.iter()).for_each(|(to,from)| {
                    to.self_scaled_add(from, *icoeff);
                });
            });
            //let mut tmp_spheric = MatrixFull::new([num_grids,s_len],0.0);
            paox.iter_mut().zip(tmp_cart.iter_mut()).for_each(|(paox_x,tmp_cart_x)| {
                //tmp_spheric.lapack_dgemm(tmp_cart_x, &mut c2s_mat, 'N','N',1.0,0.0);
                //let tmp_spheric = _dgemm_nn_serial(&tmp_cart_x.to_matrixfullslice(),&c2s_mat.to_matrixfullslice());
                paox_x.iter_columns_mut(ibas_start..ibas_start+c_len)
                .zip(tmp_cart_x.iter_columns_full()).for_each(|(to,from)| {
                    to.iter_mut().zip(from.iter()).for_each(|(to,from)| {*to = *from});
                });

            });
            ibas_start += c_len;
        }
    }
    paox
}

pub fn gto_1st_value_serial(r: &[[f64;3]],gto_center:&[f64;3], bas:&Basis4Elem, basis_type: &String) -> Vec<MatrixFull<f64>> { 
    if basis_type.eq(&"spheric".to_string()) {
        return spheric_gto_1st_value_serial(r, gto_center, bas);
    } else {
        return cartesian_gto_1st_value_serial(r, gto_center, bas);
    } 
}

pub fn cartesian_gto_1st_value_serial(r:&[[f64;3]], gto_center:&[f64;3], bas:&Basis4Elem) -> Vec<MatrixFull<f64>> {
    let num_grids:usize = r.len();
    let mut num_bas_c = 0;
    let mut num_bas_s = 0;
    bas.electron_shells.iter().for_each(|ibas| {
        let iang = ibas.angular_momentum[0] as usize;
        num_bas_c += (iang+1)*(iang+2)/2*ibas.coefficients.len();
        num_bas_s += (iang*2+1)*ibas.coefficients.len();
    });
    let mut paox = vec![MatrixFull::new([num_bas_c,num_grids],0.0);3];
    let mut ibas_start = 0;
    for ibas in &bas.electron_shells {
        let iang = ibas.angular_momentum[0] as usize;
        let c_len = (iang+1)*(iang+2)/2;
        // libcint cartesian convention for every l (the l<2 / l>=2 split the
        // historical if/else duplicated is already inside the evaluator)
        for coeff in ibas.coefficients.iter() {
            let mut tmp_cart = vec![MatrixFull::new([c_len, num_grids],0.0);3];
            coeff.iter().zip(ibas.exponents.iter()).for_each(|(icoeff,iexp)| {
                let mut tmp_cart_0 = cint_cartesian_gto_1st_cint_batch_serial_v03(*iexp, iang, gto_center, r);
                tmp_cart.iter_mut().zip(tmp_cart_0.iter()).for_each(|(to,from)| {
                    to.self_scaled_add(from, *icoeff);
                });
            });
            paox.iter_mut().zip(tmp_cart.iter_mut()).for_each(|(paox_x,tmp_cart_x)| {
                paox_x.copy_from_matr(ibas_start..ibas_start+c_len,0..num_grids,  tmp_cart_x, 0..c_len,0..num_grids);
            });
            ibas_start += c_len;
        }
    }
    paox
}

pub fn spheric_gto_1st_value_serial(r:&[[f64;3]], gto_center:&[f64;3], bas:&Basis4Elem) -> Vec<MatrixFull<f64>> {
    let num_grids:usize = r.len();
    let mut num_bas_c = 0;
    let mut num_bas_s = 0;
    bas.electron_shells.iter().for_each(|ibas| {
        let iang = ibas.angular_momentum[0] as usize;
        num_bas_c += (iang+1)*(iang+2)/2*ibas.coefficients.len();
        num_bas_s += (iang*2+1)*ibas.coefficients.len();
    });
    //let mut paox = RIFull::new([num_grids,num_bas_s,3],0.0);
    let mut paox = vec![MatrixFull::new([num_bas_s,num_grids],0.0);3];
    //let mut paox = MatrixFull::new([num_grids,num_bas_s],0.0);
    //let mut aopy = MatrixFull::new([num_grids,num_bas_s],0.0);
    //let mut aopz = MatrixFull::new([num_grids,num_bas_s],0.0);
    let mut ibas_start = 0;
    for ibas in &bas.electron_shells {
        let iang = ibas.angular_momentum[0] as usize;
        //let mut c2s_mat = &mut c2s_matrix(iang);
        let mut c2s_mat = &c2s_matrix_const(iang);
        let s_len = 2*iang + 1;
        let c_len = (iang+1)*(iang+2)/2;
        for coeff in ibas.coefficients.iter() {
            let mut tmp_cart = vec![MatrixFull::new([c_len, num_grids],0.0);3];
            coeff.iter().zip(ibas.exponents.iter()).for_each(|(icoeff,iexp)| {
                let mut tmp_cart_0 = cartesian_gto_1st_cint_batch_serial_v03(*iexp, iang, gto_center, r);

                //tmp_cart.self_scaled_add(&tmp_cart_0, *icoeff);
                tmp_cart.iter_mut().zip(tmp_cart_0.iter()).for_each(|(to,from)| {
                    to.self_scaled_add(from, *icoeff);
                });
            });
            //let mut tmp_spheric = MatrixFull::new([num_grids,s_len],0.0);
            //Stop here
            paox.iter_mut().zip(tmp_cart.iter_mut()).for_each(|(paox_x,tmp_cart_x)| {
                //tmp_spheric.lapack_dgemm(tmp_cart_x, &mut c2s_mat, 'N','N',1.0,0.0);
                // C[s_len, num_grids] = c2s[c_len, s_len]^T x tmp_cart_x[c_len, num_grids]
                let mut tmp_spheric = uninit_matrix(s_len, num_grids);
                _dgemm(&c2s_mat.to_matrixfullslice(), (0..c_len, 0..s_len), 'T',
                       &*tmp_cart_x, (0..c_len, 0..num_grids), 'N',
                       &mut tmp_spheric, (0..s_len, 0..num_grids), 1.0, 0.0);
                paox_x.copy_from_matr(ibas_start..ibas_start+s_len,0..num_grids,  &tmp_spheric, 0..s_len,0..num_grids);
                //let tmp_spheric = _dgemm_nn_serial(&tmp_cart_x.to_matrixfullslice(),&c2s_mat.to_matrixfullslice());
                //paox_x.iter_columns_mut(ibas_start..ibas_start+s_len)
                //.zip(tmp_spheric.iter_columns_full()).for_each(|(to,from)| {
                //    to.iter_mut().zip(from.iter()).for_each(|(to,from)| {*to = *from});
                //});

            });
            ibas_start += s_len;
        }
    }
    paox
}

pub fn spheric_gto_deriv_batch_serial(r:&[[f64;3]], gto_center:&[f64;3], bas:&Basis4Elem, ao_deriv:usize) -> Vec<MatrixFull<f64>> {
    let num_grids:usize = r.len();
    let mut num_bas_c = 0;
    let mut num_bas_s = 0;
    bas.electron_shells.iter().for_each(
        |ibas| 
        {
            let cur_l = ibas.angular_momentum[0] as usize;
            num_bas_c += (cur_l+1)*(cur_l+2) / 2*ibas.coefficients.len();
            num_bas_s += (cur_l*2+1)*ibas.coefficients.len();
        }
    );
    //let mut paox = RIFull::new([num_grids,num_bas_s,3],0.0);
    let n_components_cart = (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) / 6;
    let mut pao = vec![MatrixFull::new([num_bas_s, num_grids], 0.0); n_components_cart];
    let mut ibas_start = 0;

    for ibas in &bas.electron_shells {
        let cur_l = ibas.angular_momentum[0] as usize;
        let mut c2s_mat = &c2s_matrix_const(cur_l);
        let s_len = 2*cur_l + 1;
        let c_len = (cur_l+1)*(cur_l+2) / 2;
        for coeff in ibas.coefficients.iter() {
            let mut tmp_cart_res = vec![MatrixFull::new([c_len, num_grids], 0.0); n_components_cart];
            coeff.iter().zip(ibas.exponents.iter()).for_each(
                |(icoeff, iexp)| 
                {
                    // let mut tmp_cart_0 = cartesian_gto_1st_cint_batch_serial_v03(*iexp, iang, gto_center, r);
                    let mut tmp_cart_res_0 = match ao_deriv {
                        0 => {
                            let mut tmp_res = cartesian_gto_batch_v04(*iexp, cur_l, gto_center, r);
                            vec![tmp_res]
                        },
                        1 => {
                            let mut tmp_res = cartesian_gto_1st_batch_v04(*iexp, cur_l, gto_center, r);
                            tmp_res
                        },
                        2 => {
                            let mut tmp_res = cartesian_gto_2nd_batch_v04(*iexp, cur_l, gto_center, r);
                            tmp_res
                        },
                        3 => {
                            let mut tmp_res = cartesian_gto_3rd_batch_v04(*iexp, cur_l, gto_center, r);
                            tmp_res
                        },
                        _ => unreachable!()
                    };

                    let fac = cint_norm_factor(cur_l as i32, *iexp);
                    //std_gto_1st.data.iter_mut().for_each(|value| {*value *=fac});
                    tmp_cart_res_0.iter_mut().for_each(|std_gto| {
                        std_gto.self_multiple(fac)
                    });

                    //tmp_cart.self_scaled_add(&tmp_cart_0, *icoeff);
                    tmp_cart_res.iter_mut().zip(tmp_cart_res_0.iter()).for_each(
                        |(to, from)| 
                        {
                            to.self_scaled_add(from, *icoeff);
                        }
                    );
                }
            );
            //let mut tmp_spheric = MatrixFull::new([num_grids,s_len],0.0);
            //Stop here
            pao.iter_mut().zip(tmp_cart_res.iter_mut()).for_each(
                |(pao_x, tmp_cart_x)| 
                {
                    let mut tmp_spheric = MatrixFull::new([s_len, num_grids],0.0);
                    tmp_spheric.to_matrixfullslicemut().lapack_dgemm(
                        &c2s_mat.to_matrixfullslice(),
                        &tmp_cart_x.to_matrixfullslice(),
                        'T','N',1.0,0.0
                    );
                    
                    // pao_x.iter_columns_mut(ibas_start..ibas_start+s_len)
                    // .zip(tmp_spheric.iter_columns_full()).for_each(
                    //     |(to, from)| 
                    //     {
                    //         to.iter_mut().zip(from.iter()).for_each(|(to,from)| {*to = *from});
                    //     }
                    // );

                    pao_x.copy_from_matr(ibas_start..ibas_start+s_len,0..num_grids,  &tmp_spheric, 0..s_len,0..num_grids);
                }
            );
            ibas_start += s_len;
        }
    }
    pao 
}

/// sqrt(I_L(alpha)) with I_L = int_0^inf r^{2L+2} exp(-2 alpha r^2) dr, i.e.
/// the exact inverse of libcint's `CINTgto_norm(L, alpha)`.
///
/// Spheric chain: `basis_normalization` multiplies the contraction
/// coefficients by `CINTgto_norm`, while the tabulated primitives
/// (`cartesian_gto_*_v04`, raw `Norm0*Fang*monomial`) carry no radial
/// normalization. Multiplying the primitives by this factor makes the
/// product element-wise equal to libcint's spheric basis function, because
/// `sqrt(I_L)*Norm0 = sqrt((2L+1)!!)/(2 sqrt(pi))` cancels against the c2s
/// (Schlegel–Frisch) rows: `sqrt(I_L)*norm0*f*c2s = fac_sp*T_libcint`
/// (derivation: AO_AOP_NORM_AUDIT.md §4).
pub fn cint_norm_factor(iang: i32, alpha: f64) -> f64 {
    return _gaussian_int(iang*2+2, alpha*2.0).sqrt()
}

/// Cartesian-chain counterpart of [`cint_norm_factor`].
///
/// libcint's cartesian convention applies `CINTcommon_fac_sp(l)` to s and p
/// shells only (`fac_sp(0) = 1/(2 sqrt(pi))`, `fac_sp(1) = sqrt(3/(4 pi))`,
/// `fac_sp(l) = 1` for l >= 2). With `norm0*sqrt(I_L) = fac_sp(l)` for l < 2,
/// multiplying the l < 2 tabulated primitives by this factor completes
/// libcint's angular factor; for l >= 2 the factor is exactly 1 — the
/// tabulated rows stay the RAW polynomial x^lx y^ly z^lz exp(-a r^2),
/// matching the per-shell shared normalization of libcint's cartesian
/// integrals (the per-component normalization libcint skips for l >= 2
/// cannot be shared across one contraction; AO_AOP_NORM_AUDIT.md §2–§4).
pub fn cint_cart_norm_factor(iang: i32, alpha: f64) -> f64 {
    if iang == 0 || iang == 1 {
        return _gaussian_int(iang*2+2, alpha*2.0).sqrt()
    } else {
        return 1.0;
    }
}

pub fn _gaussian_int(iang: i32, alpha: f64) -> f64 {
    let dang = (iang as f64 + 1.0)*0.5;
    return libm::exp(libm::lgamma(dang))/(2.0*opowf(alpha,dang))
}
