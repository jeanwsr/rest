/// Coupled-Perturbed Hartree-Fock (CP-HF) module
///
/// Implements the solution of the CP-HF equation for computing
/// first-order molecular orbital response to external perturbations.
///
/// Phase 1: Dense linear algebra solver (this module)
///   - Builds the full (I + G̃) matrix column-by-column
///   - Solves using LAPACK dgesv
///   - Used for verification before implementing the Krylov solver
///
/// Phase 2 (future): Krylov subspace iterative solver
///   - Pople-type subspace iteration
///   - Handles large systems efficiently

pub mod cphf_solver;

pub use cphf_solver::{CPHFSolver, build_dipole_h1, build_dipole_h1_comp, test_cphf_dense};
