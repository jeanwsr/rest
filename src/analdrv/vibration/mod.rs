//! Vibrational analysis for REST analytical derivative module.
//!
//! This module contains the vibrational (normal mode) analysis [`vib`], and its interface to
//! REST [`vib_interface`]. It consumes the hessian matrix from the [`crate::analdrv::hessian`]
//! submodule, and is independent of how the hessian is produced.

// vibrational analysis
pub mod vib;

// vibrational analysis interface to REST
pub mod vib_interface;
