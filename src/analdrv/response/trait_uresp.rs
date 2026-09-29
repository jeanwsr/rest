//! Trait for Fock and response matrix (unrestricted).

use crate::analdrv::prelude::*;

/// Abstract class for Fock and response matrix (unrestricted).
///
/// Unrestricted sibling of [`RRespAPI`](crate::analdrv::response::trait_rresp::RRespAPI): every
/// orbital-space quantity exists per spin, as `[TsrView; 2]` (input) / `[Tsr; 2]` (output) arrays
/// with index 0 = α and index 1 = β.
///
/// # Term Explanation
///
/// - **Fock**: the first-order derivative to energy wrt density matrix.
/// - **Resp** (response): the second-order derivative to energy wrt density matrix, contracted by
///   input rdm/bra.
/// - **rdm1**: reduced one-particle density matrix
/// - **bra**: Bra-ket (half side), usually refers to occupied molecular coefficients (as input) or
///   contracted fock/response matrix that is half-transformed by occupied molecular coefficients
///   (as output).
///
/// # Unrestricted conventions
///
/// - The Coulomb response sees the **total** (α+β) density: a single AO operator is produced from
///   `sum_s bra_s @ mocc_s.T` internally; the consumer applies the per-spin right
///   half-transformation and a `0.5` prefactor against the restricted convention (which carries
///   the occupation-2 factor). Exchange is strictly same-spin.
/// - The DFT XC kernel `fxc` carries its spin blocks `[nvar, 2, nvar, 2]`; the response carries the
///   UHF factor `2.0` (hermitian symmetry only, no spin degeneracy) against the restricted `4.0`.
/// - The hessian prefactors assembled from the CP-SCF solution are `2*` per spin against the
///   restricted `4*` (see the `get_cpscf_hess` of the hessian driver).
pub trait URespAPI: AnalDrvBaseAPI {
    /// Generate Fock matrix by density matrix (rdm, reduced one-particle density matrix).
    ///
    /// This is the canonical way to generate Fock matrix, however, in many cases it is not the most
    /// efficient way. Refer to [`URespAPI::get_fock_coeff`] for better way to generate Fock matrix.
    ///
    /// # Parameters
    ///
    /// - `rdm` : shape `[nao, nao]` per spin. Reduced one-particle (spin) density matrix.
    /// - `prec` : precision of the underlying resource: `true` for the high-precision (SCF-grade)
    ///   one, `false` for the low-precision response resource when attached (falling back to the
    ///   high-precision one otherwise).
    ///
    /// # Returns
    ///
    /// - `fock` : shape `[nao, nao]` per spin. Fock matrix; the α/β blocks share the Coulomb
    ///   potential of the total density and differ in the same-spin exchange.
    ///
    /// # Reserved for ugfock
    ///
    /// No unrestricted driver consumes this method yet; it is part of the trait surface reserved
    /// for the future unrestricted generalized-Fock machinery, mirroring how the restricted
    /// [`RRespAPI::get_fock_rdm`] serves [`RGFockAPI`](crate::analdrv::response::trait_rgfock::RGFockAPI).
    fn get_fock_rdm(&mut self, rdm: &[TsrView; 2], prec: bool) -> [Tsr; 2];

    /// Generate Fock matrix from molecular coefficients and occupation numbers.
    ///
    /// Override this function to leverage the algorithmic advantage by occupation number over
    /// molecular orbital number.
    ///
    /// # Parameters
    ///
    /// - `mo_coeff` : shape `[nao, nmo_s]` per spin. Molecular orbital coefficients.
    /// - `mo_occ` : shape `[nmo_s]` per spin. Molecular orbital occupation numbers.
    /// - `prec` : precision of the underlying resource, forwarded to
    ///   [`get_fock_rdm`](Self::get_fock_rdm).
    ///
    /// Note `nmo_s` can be set to `nocc_s` if only occupied orbitals are considered. This can save
    /// memory and speed up the calculation.
    ///
    /// # Returns
    ///
    /// - `fock` : shape `[nao, nao]` per spin. Fock matrix (the operator in AO basis, not
    ///   contracted by `mo_coeff`).
    fn get_fock_coeff(&mut self, mo_coeff: &[TsrView; 2], mo_occ: &[TsrView; 2], prec: bool) -> [Tsr; 2] {
        let rdm = [get_dm0_restricted(mo_coeff[0].view(), mo_occ[0].view()), get_dm0_restricted(mo_coeff[1].view(), mo_occ[1].view())];
        self.get_fock_rdm(&[rdm[0].view(), rdm[1].view()], prec)
    }

    /// Prepare the data for response calculation.
    ///
    /// Response (related to second order of density matrix derivative to energy) will be called
    /// multiple-times in CP-SCF solver and other places.
    ///
    /// Some methods (especially DFT) may be helpful to prepare some data for response calculation,
    /// and store them in the object.
    ///
    /// For Hartree-Fock methods, they usually also need to store the `mo_coeff` and `mo_occ`, so to
    /// make sure [`get_response_bra`](Self::get_response_bra) can be called with only bra as input.
    ///
    /// # Parameters
    ///
    /// - `mo_coeff` : shape `[nao, nmo_s]` per spin. Molecular orbital coefficients.
    /// - `mo_occ` : shape `[nmo_s]` per spin. Molecular orbital occupation numbers.
    /// - `prec` : precision of the resource the preparation builds on; must match the `prec` of
    ///   the subsequent response contraction calls (e.g. `false` throughout the CP-SCF solve).
    fn make_response_preparation(&mut self, mo_coeff: &[TsrView; 2], mo_occ: &[TsrView; 2], prec: bool);

    /// Generate response matrix.
    ///
    /// Call [`make_response_preparation`] before this function to make sure the data is ready.
    ///
    /// # Parameters
    ///
    /// - `rdm` : shape `[nao, nao, ...]` per spin. Reduced one-particle (spin) density matrix list.
    /// - `prec` : precision of the underlying resource: `true` for the high-precision (SCF-grade)
    ///   one, `false` for the low-precision response resource when attached (falling back to the
    ///   high-precision one otherwise).
    ///
    /// # Returns
    ///
    /// - `resp` : shape `[nao, nao, ...]` per spin. Response matrix.
    ///
    /// # Reserved for ugfock
    ///
    /// No unrestricted driver consumes this rdm-form entry yet; it is the form required by the
    /// future unrestricted generalized-Fock machinery, mirroring how the restricted
    /// [`RRespAPI::get_response_rdm`] serves the RI-PT2 generalized Fock.
    ///
    /// [`make_response_preparation`]: Self::make_response_preparation
    fn get_response_rdm(&mut self, rdm: &[TsrView; 2], prec: bool) -> [Tsr; 2];

    /// Generate response matrix in half-transformed MO basis.
    ///
    /// Override this function to leverage the algorithmic advantage by occupation number over
    /// molecular orbital number.
    /// Call [`make_response_preparation`] before this function to make sure the data is ready.
    ///
    /// # Parameters
    ///
    /// - `bra` : shape `[nao, nocc_s, ...]` per spin. Bra-ket (half side), usually refers to
    ///   occupied molecular coefficients (as input) or contracted fock/response matrix that is
    ///   half-transformed by occupied molecular coefficients (as output). This is usually the
    ///   derivative of MO coefficients (like $U_{\mu i}^\mathbb{A}$ given by CP-SCF).
    /// - `prec` : precision of the underlying resource: `true` for the high-precision (SCF-grade)
    ///   one, `false` for the low-precision response resource when attached (falling back to the
    ///   high-precision one otherwise). The CP-SCF machinery contracts this method with `false`.
    ///
    /// # Returns
    ///
    /// - `resp_bra` : shape `[nao, nocc_s, ...]` per spin. Response matrix in half-transformed MO
    ///   basis, carrying the unrestricted prefactor structure documented on the trait.
    ///
    /// # Notes
    ///
    /// This function may not work for fractional occupation.
    /// We have not prepared to propose a good API for fractional occupation.
    ///
    /// [`make_response_preparation`]: Self::make_response_preparation
    fn get_response_bra(&mut self, bra: &[TsrView; 2], prec: bool) -> [Tsr; 2];
}
