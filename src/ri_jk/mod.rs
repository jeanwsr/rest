//! Resolution-of-Identity Functionalities for Fock (J/K)
//!
//! # Important Conventions
//!
//! - We will always use column-major order here. Please transform the input data to F-contiguous if
//!   encountered problems.
#![warn(unused)]

// prelude definitions
pub(self) mod prelude_dev;
pub mod util;

// pure functions
//
// - working functions that directly implement equations, does not involve complicated
//   structs/traits
// - should only involve basic data structures
// - based on RSTSR
pub mod pure_ao2mo;
pub mod pure_decompose;
pub mod pure_direct;
pub mod pure_incore;

// integrated functions
//
// - convertion to structs/traits, complicated logics
pub mod ao2mo;
pub mod decompose;
pub mod direct;
pub mod incore;

// hessian implementations
pub mod hess_r;
pub mod hess_u;

// response implementations
pub mod resp_r;

// generalized fock implementations
pub mod gfock_r;

// exports
pub use ao2mo::*;
pub use decompose::*;
pub use direct::*;
pub use incore::*;
pub use pure_ao2mo::*;
pub use pure_decompose::*;
pub use pure_direct::*;
pub use pure_incore::*;
