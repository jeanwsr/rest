//! Flags for DFT evaluation.
//!
//! This file is currently in xceff module, which only LibXC implementation uses. But it should be
//! independent to some specific XC evaluation implementation, and can be used by other
//! implementations as well. So this file is free to be moved to a more general place in the future.

pub const AO_DERIV_DIM: [usize; 5] = [1, 4, 10, 20, 35];

/// Density type for XC functionals.
///
/// - RHO: only density
/// - SIGMA: density + gradient
/// - TAU: density + gradient + kinetic energy density
/// - LAPL: density + gradient + kinetic energy density + laplacian
///
/// Note for this enum, each higher-level density type also contains all components of the
/// lower-level types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum XCDenType {
    RHO,
    SIGMA,
    TAU,
    LAPL,
}

impl XCDenType {
    /// Returns the number of components in the output density for this XC type.
    ///
    /// - RHO: 1 component (density)
    /// - SIGMA: 4 components (density + 3 gradient components)
    /// - TAU: 5 components (density + 3 gradient components + kinetic energy density)
    /// - LAPL: 6 components (density + 3 gradient components + kinetic energy density + laplacian)
    pub fn num_nvar(&self) -> usize {
        match self {
            XCDenType::RHO => 1,
            XCDenType::SIGMA => 4,
            XCDenType::TAU => 5,
            XCDenType::LAPL => 6,
        }
    }

    /// Returns the required AO derivative level for this XC type.
    ///
    /// - RHO: 0th order
    /// - SIGMA: 1st order (gradient)
    /// - TAU: 1st order (gradient)
    /// - LAPL: 2nd order (Laplacian)
    pub fn num_ao_deriv(&self) -> usize {
        match self {
            XCDenType::RHO => 0,
            XCDenType::SIGMA => 1,
            XCDenType::TAU => 1,
            XCDenType::LAPL => 2,
        }
    }

    /// Returns the number of AO components needed for this XC type
    ///
    /// - RHO: 1 component (AO value)
    /// - SIGMA: 4 components (AO value + 3 gradient components)
    /// - TAU: 4 components (AO value + 3 gradient components) [
    /// - LAPL: 10 components (AO value + 3 gradient components + 6 second derivative components)
    pub fn num_ao_comp(&self) -> usize {
        AO_DERIV_DIM[self.num_ao_deriv()]
    }
}

/// Parallelization strategy for XC evaluation.
///
/// This enum allows three kinds of parallelization strategies by `From` trait implementations:
///
/// - usize number : parallel with given chunk size;
/// - None : Use default chunk size determined by the implementation function;
/// - bool : parallel with auto-chunking if true, or serial if false.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum XCPar {
    Par { chunk_size: Option<usize> },
    Serial,
}

impl From<usize> for XCPar {
    fn from(chunk_size: usize) -> Self {
        XCPar::Par { chunk_size: Some(chunk_size) }
    }
}

impl From<Option<usize>> for XCPar {
    fn from(chunk_size: Option<usize>) -> Self {
        XCPar::Par { chunk_size }
    }
}

impl From<bool> for XCPar {
    fn from(parallel: bool) -> Self {
        if parallel {
            XCPar::Par { chunk_size: None }
        } else {
            XCPar::Serial
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum XCSpin {
    Unpolarized,
    Polarized,
}
