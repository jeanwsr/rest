//! Empirical dispersion correction (DFT-D3/DFT-D4 and related dispersion components).
//!
//! - `energy`: evaluate the dispersion energy, gradient and virial from the dftd libraries;
//! - `grad`: the GradAPI wrapper for the dispersion gradient;
//! - `hess`: the HessNucAPI wrapper; the Hessian is numerical from the analytic gradient.

pub mod energy;
pub mod grad;
pub mod hess;
