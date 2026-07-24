//! Development prelude for RI/JK module.

// core math import
pub use crate::utilities::rstsr_util::*;
pub use itertools::Itertools;
pub use num::FromPrimitive;
pub use rayon::prelude::*;
pub use rest_libcint::prelude::*;
pub use rstsr::prelude::*;
pub use rstsr_core::prelude_dev::uninitialized_vec;
pub use std::collections::HashMap;

pub use rt::blas::{BlasFloat, LapackDriverAPI};

pub type Tsr<T = f64> = Tensor<T, DeviceBLAS, IxD>;
pub type TsrView<'a, T = f64> = TensorView<'a, T, DeviceBLAS, IxD>;

// utilities and logic-related imports
pub(super) use super::util;
pub use crate::molecule_io::Molecule;
pub use crate::utilities::memory_batch::*;
pub use core::fmt::Write;
pub use rest_tensors::{MatrixFull, MatrixUpper};

pub use FlagSide::L as Left;
pub use FlagSide::R as Right;
