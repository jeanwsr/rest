//! Pure functions for generalized Fock (restricted) and related methods for RI-PT2.

// #![warn(unused)]
use itertools::{izip, Itertools};
use num::ToPrimitive;
use rayon::prelude::*;
use rstsr::prelude::*;

use crate::utilities::buffer_pool::BufferPool;
use rt::blas::BlasFloat;

type Tsr<T, D = IxD> = Tensor<T, DeviceBLAS, D>;
type TsrView<'a, T, D = IxD> = TensorView<'a, T, DeviceBLAS, D>;
