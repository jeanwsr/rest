//! Trait for generalized Fock matrix (restricted).

use crate::analdrv::prelude::*;
use enumflags2::BitFlags;

/// Enumeration of the parts of the generalized Fock matrix.
///
/// - OO: occupied-occupied block $\mathscr{F}_{ij}$.
/// - OV: occupied-virtual block $\mathscr{F}_{ia}$.
/// - VO: virtual-occupied block $\mathscr{F}_{ai}$.
/// - VV: virtual-virtual block $\mathscr{F}_{ab}$.
#[enumflags2::bitflags]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
#[repr(u32)]
pub enum GFockParts {
    OO,
    OV,
    VO,
    VV,
}

/// Trait for generalized Fock matrix (restricted).
///
/// Generalized Fock matrix is defined as energy derivative with respect to the molecular
/// coefficients by ket, and contracted with the molecular coefficients by bra:
///
/// $$
/// \mathscr{F}_{pq} = \sum_{\mu} C_{\mu p} \frac{\partial E}{\partial C_{\mu q}}
/// $$
///
/// We do not use bra-ket (half-transformed) notation here, due to that post-SCF methods may not be
/// easily expressed in atomic basis form. So everything will work in molecular orbital basis.
///
/// Note the generalized Fock matrix is not necessarily Hermitian (symmetric). As exception, if
/// energy is variational with respect to the molecular coefficients, the generalized Fock matrix is
/// Hermitian (and is usual Fock matrix in molecular orbital basis).
///
/// This term is usually used for post-SCF perturbation-based methods, such as MP2, CCSD, etc. We do
/// not know if it can be utilized in MC-SCF or CI methods. This is usually not required in
/// SCF-related methods (it is usually overkill for SCF energy contributions).
///
/// ## Note to developers
///
/// For structs that implements `RGFockAPI`, you should explicitly pass `mo_coeff` and `mo_occ`, or
/// something equilvant, to the constructor of the struct, so that the generalized Fock matrix can
/// be computed in molecular orbital basis.
///
/// For MP2 or similar methods, if frozen orbitals are used, you should be noticed that generalized
/// Fock is usually only relavent to the active space. Frozen/active mask should also be passed to
/// the constructor of the struct.
///
/// Also note we named `make_` for every function, not `get_`. We assume these functions will store
/// computed results in the struct, and reuse them if the same function is called again. Make sure
/// the computed results are not polluted by other functions, and make sure double call will not
/// pollute the results and is expected to directly return the stored results.
///
/// We note that [`Self::make_gfock`] and [`Self::make_lagrangian`] may optionally requires a
/// response object that implements `RRespAPI`. But for [`Self::make_rdm1`], we have not decided if
/// it requires a response object (MP2 does not require this object).
pub trait RGFockAPI: AnalDrvBaseAPI {
    /// Make generalized Fock matrix in molecular orbital basis.
    ///
    /// Note you can use `GFockParts` as bit-flags to specify which parts of the generalized Fock
    /// matrix to compute.
    /// - For example, you can use `GFockParts::OO | GFockParts::OV` to compute both the
    ///   occupied-occupied and occupied-virtual blocks.
    /// - For lagrangian computation, you can use `GFockParts::OV | GFockParts::VO` and then
    ///   anti-symmetrize the result to get the lagrangian.
    ///
    /// # Parameters
    ///
    /// - `parts` : Bit-flags of `GFockParts` to specify which parts of the generalized Fock matrix
    ///   to compute.
    /// - `resp` : The response object that implements `RRespAPI`. The response object should
    ///   represent the SCF method that gives the molecular orbitals. This is optional depending on
    ///   the implementation of the generalized Fock matrix. For example, for MP2, the response
    ///   object is not needed when handling Fia (OV) and Fab (VV), but required for other cases.
    ///
    /// # Returns
    ///
    /// - `gfock` : shape (nmo, nmo). Generalized Fock matrix in molecular orbital basis. Depending
    ///   on the `parts` specified, some parts of the matrix may be zero.
    fn make_gfock(&mut self, resp: Option<&impl RRespAPI>, parts: impl Into<BitFlags<GFockParts>>) -> Tsr;

    /// Make reduced one-particle density matrix (rdm1) in molecular orbital basis.
    ///
    /// Please note this is the "unrelaxed", not the "relaxed", density matrix.
    ///
    /// By our definition, it is the density matrix that can represent skeleton energy derivative
    /// (the property derivation that does not change molecular orbitals). As comparasion, the
    /// relaxed density matrix can represent the total energy derivative (which includes the orbital
    /// relaxation and requires CP-SCF or Z-Vector).
    ///
    /// # Returns
    ///
    /// - `rdm1` : shape (nmo, nmo). Reduced one-particle density matrix in molecular orbital basis.
    fn make_rdm1(&mut self) -> Tsr;

    /// Make lagrangian matrix in molecular orbital basis.
    ///
    /// # Returns
    ///
    /// - `lagrangian` : shape (nvir, nocc). Lagrangian matrix in molecular orbital basis. Note the
    ///   `nvir` and `nocc` are the number of virtual and occupied orbitals: they are derived from
    ///   the struct internal data, not checked from the input parameters.
    /// - `resp` : The response object that implements `RRespAPI`. See [`Self::make_gfock`] for more
    ///   details.
    fn make_lagrangian(&mut self, resp: Option<&impl RRespAPI>) -> Tsr;
}
