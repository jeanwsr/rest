//! Basis-set-level GTO evaluators: value / first derivative / arbitrary-order
//! derivative of spherical and cartesian GTOs on grids, taking `Basis4Elem`,
//! plus cint-style normalization factors.
use rest_tensors::MatrixFull;
use rayon::prelude::*;
use itertools::{Itertools, izip};
use tensors::{MathMatrix, ParMathMatrix};
use tensors::matrix_blas_lapack::{_dgemm_nn, _dgemm_nn_serial};
use std::convert::TryInto;
use std::f64::consts::{PI, E};
use libm;
use crate::basis_io::basic_math::{delta_r, opowf, double_factorial, specific_double_factorial};
use crate::constants::{C2S_L0, C2S_L1, C2S_L2, C2S_L3, C2S_L4, CarBasInfo, c2s_matrix_const, cartesian_gto_const, cint_cartesian_gto_const};
use crate::utilities;
use super::Basis4Elem;
use super::{
    cartesian_gto_1st_batch_v04, cartesian_gto_1st_cint, cartesian_gto_1st_cint_batch,
    cartesian_gto_1st_cint_batch_serial, cartesian_gto_1st_cint_batch_serial_v03,
    cartesian_gto_2nd_batch_v04, cartesian_gto_3rd_batch_v04, cartesian_gto_batch_v04,
    cartesian_gto_cint, cartesian_gto_cint_batch, cartesian_gto_cint_batch_rayon,
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
    let mut outmat_s = MatrixFull::new([num_grids,num_bas_s],0.0);
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
            let tmp_spheric = _dgemm_nn(&tmp_cart.to_matrixfullslice(), &c2s_mat.to_matrixfullslice());
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
    let mut outmat_s = MatrixFull::new([num_grids,num_bas_s],0.0);
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
            let tmp_spheric = _dgemm_nn_serial(&tmp_cart.to_matrixfullslice(), &c2s_mat.to_matrixfullslice());
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
    let mut outmat_s = MatrixFull::new([num_bas_s, num_grids],0.0);
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
            let mut tmp_spheric= MatrixFull::new([s_len,num_grids],0.0);
            tmp_spheric.to_matrixfullslicemut().lapack_dgemm(
                &c2s_mat.to_matrixfullslice(),
                &tmp_cart.to_matrixfullslice(), 
                'T', 'N', 1.0, 0.0);

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
        if iang ==0 || iang==1 {
            for coeff in ibas.coefficients.iter() {
                let mut tmp_cart = MatrixFull::new([c_len,num_grids],0.0);
                coeff.iter().zip(ibas.exponents.iter()).for_each(|(icoeff,iexp)| {
                        tmp_cart += cint_cartesian_gto_cint_batch_v04(*iexp, iang, gto_center, r)* (*icoeff);
                });

                outmat_c.copy_from_matr(ibas_start..ibas_start+c_len, 0..num_grids, 
                    &tmp_cart, 0..c_len, 0..num_grids);
                ibas_start += c_len;
            }
        } else {
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
    }

    outmat_c
}

pub fn gto_value(r:&[f64;3], gto_center:&[f64;3], bas:&Basis4Elem, basis_type: &String) -> Vec<f64> {
    let mut value:Vec<f64> = vec![];
    for ibas in &bas.electron_shells {
        let iang = ibas.angular_momentum[0];
        for coeff in ibas.coefficients.iter() {
            let mut tmp_cart = MatrixFull::new([((iang+1)*(iang+2)/2) as usize,1],0.0);
            coeff.iter().zip(ibas.exponents.iter()).for_each(|(icoeff,iexp)| {
                //let scoeff = *icoeff * _gaussian_int(iang*2 + 2, *iexp*2.0).sqrt();
                let mut tmp_cart_0 = cartesian_gto_cint(*iexp, iang as usize, gto_center, r);
                //tmp_cart.data.iter_mut().zip(tmp_cart_0.data.iter()).for_each(|(to,from)| {
                //    *to += *from*icoeff
                //});
                tmp_cart.self_scaled_add(&tmp_cart_0, *icoeff);
            });
            if basis_type.eq(&"spheric".to_string()) {
                let mut tmp_spheric = MatrixFull::new([(2*iang + 1) as usize, 1],0.0);
                tmp_spheric.to_matrixfullslicemut().lapack_dgemm(
                    &tmp_cart.to_matrixfullslice(), 
                    &c2s_matrix_const(iang as usize).to_matrixfullslice(),
                     'T','N',1.0,1.0);
                //tmp_spheric.lapack_dgemm(&mut tmp_cart, &mut c2s_matrix(iang as usize), 'T', 'N', 1.0, 0.0);
                value.extend(tmp_spheric.data);
            } else {
                value.extend(tmp_cart.data);
            }
        }
        
    }
    value
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
                let tmp_spheric = _dgemm_nn(&tmp_cart_x.to_matrixfullslice(),&c2s_mat.to_matrixfullslice());
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
                let tmp_spheric = _dgemm_nn_serial(&tmp_cart_x.to_matrixfullslice(),&c2s_mat.to_matrixfullslice());
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
        if iang == 0 || iang == 1 { 
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
        } else {
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
                let mut tmp_spheric = MatrixFull::new([s_len,num_grids],0.0);
                tmp_spheric.to_matrixfullslicemut().lapack_dgemm(
                    &c2s_mat.to_matrixfullslice(),
                    &tmp_cart_x.to_matrixfullslice(),
                    'T','N',1.0,0.0);
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

pub fn gto_1st_value(r:&[f64;3], gto_center:&[f64;3], bas:&Basis4Elem, basis_type: &String) -> Vec<Vec<f64>> {
    let mut value:Vec<Vec<f64>> = vec![vec![];3];
    //let mut value:[Vec<f64>;3] = [Vec::new();3];
    for ibas in &bas.electron_shells {
        let iang = ibas.angular_momentum[0];
        for coeff in ibas.coefficients.iter() {
            let mut tmp_cart = [
                MatrixFull::new([((iang+1)*(iang+2)/2) as usize,1],0.0),
                MatrixFull::new([((iang+1)*(iang+2)/2) as usize,1],0.0),
                MatrixFull::new([((iang+1)*(iang+2)/2) as usize,1],0.0)
            ];
            coeff.iter().zip(ibas.exponents.iter()).for_each(|(icoeff,iexp)| {
                //let scoeff = *icoeff * _gaussian_int(iang*2 + 2, *iexp*2.0).sqrt();
                let mut tmp_cart_0 = cartesian_gto_1st_cint(*iexp, iang as usize, gto_center, r);
                //println!("debug: {:?}, {:?}, {:?}", &tmp_cart_0[0].data,&tmp_cart_0[1].data,&tmp_cart_0[2].data);
                tmp_cart.iter_mut().zip(tmp_cart_0.iter()).for_each(|(tmp_cart,tmp_cart_0)| {
                    tmp_cart.self_scaled_add(tmp_cart_0, *icoeff)
                });
            });
            if basis_type.eq(&"spheric".to_string()) {
                tmp_cart.iter().zip(value.iter_mut()).for_each(|(tmp_cart, value)| {
                    let mut tmp_spheric = MatrixFull::new([(2*iang + 1) as usize, 1],0.0);
                    //tmp_spheric.lapack_dgemm(tmp_cart, &mut c2s_matrix(iang as usize), 'T', 'N', 1.0, 0.0);
                    tmp_spheric.to_matrixfullslicemut().lapack_dgemm(
                        &tmp_cart.to_matrixfullslice(), 
                        &c2s_matrix_const(iang as usize).to_matrixfullslice(), 
                        'T', 'N', 1.0, 0.0);
                    value.extend(tmp_spheric.data.iter());
                });
            } else {
                tmp_cart.iter().zip(value.iter_mut()).for_each(|(tmp_cart, value)| {
                    value.extend(tmp_cart.data.iter())
                });
            }
        }
        
    }
    value
}
pub fn gto_value_debug(r:&[f64;3], gto_center:&[f64;3], bas:&Basis4Elem, basis_type: &String,count: usize) -> Vec<f64> {
    let mut value:Vec<f64> = vec![];
    for ibas in &bas.electron_shells {
        let iang = ibas.angular_momentum[0];
        for coeff in ibas.coefficients.iter() {
            let mut tmp_cart = MatrixFull::new([((iang+1)*(iang+2)/2) as usize,1],0.0);
            coeff.iter().zip(ibas.exponents.iter()).for_each(|(icoeff,iexp)| {
                //let scoeff = *icoeff * _gaussian_int(iang*2 + 2, *iexp*2.0).sqrt();
                let mut tmp_cart_0 = cartesian_gto_cint(*iexp, iang as usize, gto_center, r);
                //tmp_cart.data.iter_mut().zip(tmp_cart_0.data.iter()).for_each(|(to,from)| {
                //    *to += *from*icoeff
                //});
                tmp_cart.self_scaled_add(&tmp_cart_0, *icoeff);
                //if count<10 {println!("iang: {:2} coeff: {:8.4}, exp: {:8.4}, tmp_cart: {:?}", iang, scoeff, iexp, &tmp_cart)};
            });
            if basis_type.eq(&"spheric".to_string()) {
                let mut tmp_spheric = MatrixFull::new([(2*iang + 1) as usize, 1],0.0);
                //tmp_spheric.lapack_dgemm(&mut tmp_cart, &mut c2s_matrix(iang as usize), 'T', 'N', 1.0, 0.0);
                tmp_spheric.to_matrixfullslicemut().lapack_dgemm(
                    &tmp_cart.to_matrixfullslice(), 
                    &c2s_matrix_const(iang as usize).to_matrixfullslice(),
                    'T', 'N', 1.0, 0.0);
                value.extend(tmp_spheric.data);
            } else {
                value.extend(tmp_cart.data);
            }
        }
        
    }
    value
}

/// Normalization factor for GTO radial part g=r^l e^{-\alpha r^2}
/// 
/// \frac{1}{\sqrt{\int g^r r^2 dr}} = \sqrt{\frac{2^{2l+3} (l+1)! (2a)^{1+1.5}}{(2l+2)!\sqrt{\pi}}}
/// Ref: H. B. Schlegel and M. J. Frisch, Int. J. Quant. Chem., 54(1995, 83-87)
/// 
/// This part is absorbed in the BasCell.coefficients for libcint.
/// However, in REST, we prepare the tabulated GTOs which are normalized.
/// In consequence, this factor should be removed again for the use in REST. 
/// 
/// ===============================================================================================
/// NOTE::: It works for spheric basis, s and p type-cartesian basis.
/// NOTE::: for cartesian basis with l>=2, the normalization factor is not correct.
/// NOTE::: Therefore, for the calculation with basis_type = catesian, you need to use cint_cart_norm_factor()
/// ===============================================================================================
pub fn cint_norm_factor(iang: i32, alpha: f64) -> f64 {
    return _gaussian_int(iang*2+2, alpha*2.0).sqrt()
}

pub fn cint_cart_norm_factor(iang: i32, alpha: f64) -> f64 {
    //return _gaussian_int(iang*2+2, alpha*2.0).sqrt()
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
