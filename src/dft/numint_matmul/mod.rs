//! Naive matrix multiplication driver for DFT numerical integration.
//!
//! Though saying "naive", it should be sufficiently good for dense GTO grids - basis pairs (small systems).
//! For large molecules, we do not exploit sparsity here for code simplicity.

pub mod gfock_rks;
pub mod hess_rks;
pub mod hess_uks;
pub mod nimatmul;
pub mod pure_eval_rho;
pub mod pure_xcpot;
pub mod resp_rks;

#[allow(unused)]
pub mod prelude {
    use super::*;

    pub(super) use indexmap::IndexMap;
    pub(super) use itertools::Itertools;
    pub(super) use libxc::prelude::*;
    pub(super) use rayon::prelude::*;
    pub(super) use rest_libcint::prelude::*;
    pub(super) use rstsr::prelude::*;
    pub(super) use std::collections::HashMap;
    pub(super) use std::sync::{Arc, Mutex};

    pub(super) use super::nimatmul::*;
    pub(super) use super::pure_eval_rho::*;
    pub(super) use super::pure_xcpot::*;
    pub(super) use crate::dft::xceff::prelude::*;
    pub(super) use crate::ni_check_shape;
    pub(super) use crate::ri_jk::util::get_dm0_restricted;
    pub(super) use crate::utilities::buffer_pool::BufferPool;
    pub(super) use crate::utilities::rstsr_util::*;

    pub(super) type TsrView<'a, T = f64> = TensorView<'a, T, DeviceBLAS>;
    pub(super) type Tsr<T = f64> = Tensor<T, DeviceBLAS>;
}

/* #region utility ni_check_shape */

pub trait NIIntoUsizeVec {
    fn into_usize_vec(self) -> Vec<usize>;
}

impl NIIntoUsizeVec for i32 {
    fn into_usize_vec(self) -> Vec<usize> {
        vec![self as usize]
    }
}

impl NIIntoUsizeVec for usize {
    fn into_usize_vec(self) -> Vec<usize> {
        vec![self]
    }
}

impl NIIntoUsizeVec for &[usize] {
    fn into_usize_vec(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl<const N: usize> NIIntoUsizeVec for [usize; N] {
    fn into_usize_vec(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl NIIntoUsizeVec for Vec<usize> {
    fn into_usize_vec(self) -> Vec<usize> {
        self
    }
}

impl NIIntoUsizeVec for &Vec<usize> {
    fn into_usize_vec(self) -> Vec<usize> {
        self.clone()
    }
}

#[macro_export]
macro_rules! ni_check_shape {
    ($actual:expr, $expected:expr, $msg:expr) => {{
        use $crate::dft::numint_matmul::NIIntoUsizeVec;
        if $actual.into_usize_vec() != $expected.into_usize_vec() {
            let str_actual = stringify!($actual);
            let str_expected = stringify!($expected);
            panic!(
                "Shape mismatch: expected {} = {:?}, but got {} = {:?}; message: {}",
                str_expected,
                $expected.into_usize_vec(),
                str_actual,
                $actual.into_usize_vec(),
                $msg
            );
        }
    }};

    ($cond:expr, $msg:expr) => {{
        if !$cond {
            let str_cond = stringify!($cond);
            panic!("Condition failed: {}; message: {}", str_cond, $msg);
        }
    }};
}

/* #endregion */
