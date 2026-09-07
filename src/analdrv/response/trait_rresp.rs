//! Trait for Fock and response matrix (restricted).

use crate::analdrv::prelude::*;

/// Abstract class for Fock and response matrix (restricted).
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
pub trait RRespAPI: AnalDrvBaseAPI {
    /// Generate Fock matrix by density matrix (rdm, reduced one-particle density matrix).
    ///
    /// This is the canonical way to generate Fock matrix, however, in many cases it is not the most
    /// efficient way. Refer to [`RRespAPI::get_fock_coeff`] for better way to generate Fock matrix.
    ///
    /// # Parameters
    ///
    /// - `rdm` : shape `[nao, nao]`. Reduced one-particle density matrix.
    ///
    /// # Returns
    ///
    /// - `fock` : shape `[nao, nao]`. Fock matrix.
    fn get_fock_rdm(&mut self, rdm: TsrView) -> Tsr;

    /// Generate the first-order derivative of Fock matrix with respect to molecular coefficients
    /// (mo_coeff).
    ///
    /// Override this function to leverage the algorithmic advantage by occupation number over
    /// molecular orbital number.
    ///
    /// # Parameters
    ///
    /// - `mo_coeff` : shape `[nao, nmo]`. Molecular orbital coefficients.
    /// - `mo_occ` : shape `[nmo]`. Molecular orbital occupation numbers.
    ///
    /// Note `nmo` can be set to `nocc` if only occupied orbitals are considered. This can save
    /// memory and speed up the calculation.
    ///
    /// # Returns
    ///
    /// - `fock` : shape `[nao, nmo]`. Fock matrix.
    fn get_fock_coeff(&mut self, mo_coeff: TsrView, mo_occ: TsrView) -> Tsr {
        let rdm = get_dm0_restricted(mo_coeff, mo_occ);
        self.get_fock_rdm(rdm.view())
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
    /// - `mo_coeff` : shape `[nao, nmo]`. Molecular orbital coefficients.
    /// - `mo_occ` : shape `[nmo]`. Molecular orbital occupation numbers.
    fn make_response_preparation(&mut self, mo_coeff: TsrView, mo_occ: TsrView);

    /// Generate response matrix.
    ///
    /// Refer to [`RRespAPI::get_resp_bra`] for better way to leverage the algorithmic advantage by
    /// occupation number over molecular orbital number (but also notice the output is different in
    /// shape and meaning).
    /// Call [`make_response_preparation`] before this function to make sure the data is ready.
    ///
    /// # Parameters
    ///
    /// - `rdm` : shape `[nao, nao, ...]`. Reduced one-particle density matrix list.
    ///
    /// Note on SCF contribution term `mo_coeff` and `mo_occ`: Not useful if energy contribution is
    /// exactly second-order function of density matrix (such as coulomb or exchange), but is
    /// required if energy is high-order function of density matrix (such as DFT).
    ///
    /// Note `nmo` can be set to `nocc` if only occupied orbitals are considered. This can save
    /// memory and speed up the calculation.
    ///
    /// Note that we do not provide option to put rdm as SCF density matrix. You must use `mo_coeff`
    /// and `mo_occ` to represent SCF density matrix currently.
    ///
    /// # Returns
    ///
    /// - `resp` : shape `[nao, nao, ...]`. Response matrix.
    ///
    /// [`make_response_preparation`]: Self::make_response_preparation
    fn get_response_rdm(&mut self, rdm: TsrView) -> Tsr;

    /// Generate response matrix in half-transformed MO basis.
    ///
    /// Override this function to leverage the algorithmic advantage by occupation number over
    /// molecular orbital number.
    /// Call [`make_response_preparation`] before this function to make sure the data is ready.
    ///
    /// # Parameters
    ///
    /// - `bra` : shape `[nao, nocc, ...]`. Bra-ket (half side), usually refers to occupied
    ///   molecular coefficients (as input) or contracted fock/response matrix that is
    ///   half-transformed by occupied molecular coefficients (as output). This is usually the
    ///   derivative of MO coefficients (like $U_{\mu i}^\mathbb{A}$ given by CP-SCF).
    ///
    /// # Returns
    ///
    /// - `resp_bra` : shape `[nao, nocc, ...]`. Response matrix in half-transformed MO basis.
    ///
    /// # Notes
    ///
    /// This function may not work for fractional occupation.
    /// We have not prepared to propose a good API for fractional occupation.
    ///
    /// [`make_response_preparation`]: Self::make_response_preparation
    fn get_response_bra(&mut self, bra: TsrView) -> Tsr;
}
