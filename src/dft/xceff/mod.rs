//! Generate effective XC potentials by LibXC driver.
//!
//! Functionality of this module is similar to `libxc_itrf.rs`, function `eval_xc_eff`.
//! We may handle refactor and merge in the future.

pub mod flags;
pub mod libxc_wrap;
pub mod xc_deriv;

pub mod prelude {
    use super::*;

    pub use flags::{XCDenType, XCPar, XCSpin, AO_DERIV_DIM};
    pub use libxc_wrap::{determine_den_type, determine_den_type_from_list, libxc_eval_eff};

    pub(super) use crate::ni_check_shape;
    pub(super) use crate::utilities::rstsr_util::*;
    pub(super) use itertools::Itertools;
    pub(super) use libxc::prelude::*;
    pub(super) use rayon::prelude::*;
    pub(super) use rstsr::prelude::*;

    pub(super) type TsrView<'a, T = f64> = TensorView<'a, T, DeviceBLAS>;
    pub(super) type Tsr<T = f64> = Tensor<T, DeviceBLAS>;
}
