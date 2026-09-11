//! Electric multipole moment property for REST.
//!
//! This module evaluates electric multipole moments (currently dipole, quadrupole, octupole and
//! hexadecapole; higher orders can be added as further methods on the traits and drivers) of
//! restricted SCF and double-hybrid (DH) type methods. The total moment is assembled from three
//! kinds of contributions:
//!
//! - **Non-electronic (nuclear) part**: moments of the classical nuclear charges, described by
//!   the [`MultipoleNucAPI`](trait_multipole::MultipoleNucAPI) trait. Currently the only
//!   implementation is the nuclear charge
//!   [`MultipoleNucCharge`](nuc_charge::MultipoleNucCharge); other classical charge sources
//!   (like point charges of an external field source) can be added as further implementors in
//!   the future.
//! - **SCF density contribution**: contraction of the SCF density matrix with the multipole
//!   integral tensors.
//! - **Response density increment** (optional, for post-SCF/DH methods): contraction of the
//!   density increments of the
//!   [`RGFockDH`](crate::analdrv::response::rgfock_interface::RGFockDH) composite — the
//!   unrelaxed correlation rdm1 and the relaxed (Z-vector) increment — with the multipole
//!   integral tensors.
//!
//! Unlike the hessian, the multipole evaluation is fully incremental (every contribution is a
//! plain one-density contraction), so a single driver
//! [`RMultipoleDH`](rmultipole::RMultipoleDH) handles both the SCF and the DH levels; the DH
//! objects are simply optional fields of the driver.
//!
//! # Conventions
//!
//! - All quantities are in atomic units. The electronic contribution carries the electron charge
//!   sign: `dip = sum_A Z_A (R_A - origin) - sum_{mu nu} D_{mu nu} <mu| r |nu>`.
//! - The origin is an explicit parameter: a function-level parameter for the (closed-form)
//!   nuclear trait methods, and a construction-time field (default `[0.0; 3]`) of the driver.
//!   The dipole moment of a charge-neutral molecule is origin-independent, while higher moments
//!   shift as `Theta' = Theta - R mu^T - mu R^T + Q_tot R R^T` (and `mu' = mu - Q_tot R`);
//!   keeping the origin explicit allows these properties to be verified numerically.
//! - The quadrupole is evaluated and stored as the raw (second-moment) tensor of shape `[3, 3]`;
//!   the traceless form is a cheap conversion, see
//!   [`quadrupole_to_traceless`](rmultipole::quadrupole_to_traceless).
//! - Multipole integrals are evaluated by [`CInt::integrate`] (column-major), inside a common
//!   origin save/restore. The flat 9-component axis of `int1e_rr` is reshaped to trailing axes
//!   `(3, 3)` with the first Cartesian index the slowest (`comp = 3 t1 + t2`); this ordering is
//!   pinned numerically against PySCF in the module tests.

pub mod nuc_charge;
pub mod rmultipole;
pub mod trait_multipole;
