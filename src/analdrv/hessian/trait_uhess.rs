use crate::analdrv::prelude::*;

/// Abstract class for Hessian-related API for unrestricted SCF core components.
///
/// Difference to [`RHessCoreAPI`] is that we may need different signature. Basic ideas are exactly
/// the same.
pub trait UHessCoreAPI: HessUtilAPI {
    /// Generate the **skeleton** contribution of Hessian for current SCF component.
    ///
    /// # Parameters
    ///
    /// - `mo_coeff` : shape `[nao, nmo_α]` and `[nao, nmo_β]`. Molecular orbital coefficients.
    /// - `mo_occ` : shape `[nmo_α]` and `[nmo_β]`. Molecular orbital occupation numbers. In usual
    ///   cases, the occupied orbitals should have occupation 1 (unrestricted), and virtual orbitals
    ///   should have occupation 0.
    /// - `atm_list` : optional list of atom indices to compute the Hessian for. If `None`, all
    ///   atoms are computed.
    ///
    /// # Returns
    ///
    /// - `hess` : shape `[3, 3, natm, natm]`. The Hessian matrix for current SCF component, where
    ///   `natm = atm_list.len()` if `atm_list` is `Some`, else `mol.natm()`.
    ///
    ///   Note the hessian should be of indices `[s, t, B, A]` for column major.
    ///
    /// # See also
    ///
    /// [`RHessCoreAPI::make_skeleton_hess`]. Signature difference: `mo_coeff` and `mo_occ` type
    /// different.
    fn make_skeleton_hess(&mut self, mo_coeff: &[TsrView; 2], mo_occ: &[TsrView; 2], atm_list: Option<&[usize]>)
        -> Tsr;

    /// Generate the function to compute the first-order derivative of core component.
    ///
    /// This function works in atomic basis, so should have same implementation to restricted case.
    ///
    /// # Parameters (in closure)
    ///
    /// - `A` : usize. The atom index (global, in original molecule) for which the derivative is
    ///   taken.
    ///
    /// # Returns (in closure)
    ///
    /// - `deriv1` : shape `[nao, nao, 3]`. The first-order derivative of core component with
    ///   respect to the position of atom `A`.
    ///
    /// # See also
    ///
    /// [`RHessCoreAPI::generator_deriv1`]. Signature difference: no difference.
    fn generator_deriv1(&self) -> Box<dyn FnMut(usize) -> Tsr>;
}

/// Abstract class for Hessian-related API for restricted SCF electronic interaction components.
///
/// Difference to [`RHessElecInteractAPI`] is that we may need different signature. Basic ideas are
/// exactly the same.
pub trait UHessElecInteractAPI: HessUtilAPI {
    /// Generate the **skeleton** contribution of Hessian for current SCF component.
    ///
    /// # Parameters
    ///
    /// - `mo_coeff` : shape `[nao, nmo_α]` and `[nao, nmo_β]`. Molecular orbital coefficients.
    /// - `mo_occ` : shape `[nmo_α]` and `[nmo_β]`. Molecular orbital occupation numbers. In usual
    ///   cases, the occupied orbitals should have occupation 1, and virtual orbitals should have
    ///   occupation 0.
    /// - `atm_list` : optional list of atom indices to compute the Hessian for. If `None`, all
    ///   atoms are computed.
    ///
    /// # Returns
    ///
    /// - `hess` : shape `[3, 3, natm, natm]`. The Hessian matrix for current SCF component.
    ///
    ///   Note the hessian should be of indices `[s, t, B, A]` for column major.
    ///
    /// # See also
    ///
    /// [`RHessElecInteractAPI::make_skeleton_hess`]. Signature difference: `mo_coeff` and `mo_occ`
    /// type different.
    fn make_skeleton_hess(&mut self, mo_coeff: &[TsrView; 2], mo_occ: &[TsrView; 2], atm_list: Option<&[usize]>)
        -> Tsr;

    /// First order skeleton derivative in AO basis.
    ///
    /// # Parameters
    ///
    /// - `mo_coeff` : shape `[nao, nmo_α]` and `[nao, nmo_β]`. Molecular orbital coefficients.
    /// - `mo_occ` : shape `[nmo_α]` and `[nmo_β]`. Molecular orbital occupation numbers.
    /// - `atm_list` : optional list of atom indices over which derivatives are computed.
    ///
    /// # Returns
    ///
    /// - `deriv_ao` : shape `[nao, nao, 3, natm]` and `[nao, nao, 3, natm]`. The first-order
    ///   skeleton derivative in AO basis.
    ///
    /// # See also
    ///
    /// [`RHessElecInteractAPI::get_deriv1_ao`]. Signature difference: `mo_coeff` and `mo_occ` type
    /// different, output shape different.
    fn get_deriv1_ao(&mut self, mo_coeff: &[TsrView; 2], mo_occ: &[TsrView; 2], atm_list: Option<&[usize]>)
        -> [Tsr; 2];

    /// First order skeleton derivative in half-transformed MO basis.
    ///
    /// # Parameters
    ///
    /// - `mo_coeff` : shape `[nao, nmo_α]` and `[nao, nmo_β]`. Molecular orbital coefficients.
    /// - `mo_occ` : shape `[nmo_α]` and `[nmo_β]`. Molecular orbital occupation numbers.
    /// - `atm_list` : optional list of atom indices over which derivatives are computed.
    ///
    /// # Returns
    ///
    /// - `deriv_bra` : shape `[nao, nocc_α, 3, natm]` and `[nao, nocc_β, 3, natm]`. The first-order
    ///   skeleton derivative in half-transformed MO basis. Note that this function will handle the
    ///   order of occupied orbitals. If occupation number is not sorted contiguously, you may be
    ///   extra cautious to this function.
    ///
    /// # See also
    ///
    /// [`RHessElecInteractAPI::get_deriv1_bra`]. Signature difference: `mo_coeff` and `mo_occ` type
    /// different, output type different.
    fn get_deriv1_bra(
        &mut self,
        mo_coeff: &[TsrView; 2],
        mo_occ: &[TsrView; 2],
        atm_list: Option<&[usize]>,
    ) -> [Tsr; 2] {
        // Using `a` and `b` may be confusing for spin description. Using non-ASCII may be better.
        // Decision:
        // - In code or comments, we may use Greek `α`, `β` (not Cyrillic). This violates UTS #39 and
        //   RFCS-2457 (also related to rust lint `mixed_script_confusables`). But I think it's prettier for
        //   reading, and more clear then 0/1 in meaning.
        // - In dictionary keys, we use `0` and `1` to avoid any potential issues.
        let [α, β] = [0, 1];

        let occidx = [mo_occ[α].view().greater(0).into_vec(), mo_occ[β].view().greater(0).into_vec()];
        let mocc = [mo_coeff[α].bool_select(-1, &occidx[α]), mo_coeff[β].bool_select(-1, &occidx[β])];
        let deriv1_ao = self.get_deriv1_ao(mo_coeff, mo_occ, atm_list);
        [&deriv1_ao[α] % &mocc[α], &deriv1_ao[β] % &mocc[β]]
    }

    /// Prepare the data for response calculation.
    ///
    /// # Parameters
    ///
    /// - `mo_coeff` : shape `[nao, nmo_α]` and `[nao, nmo_β]`. Molecular orbital coefficients.
    /// - `mo_occ` : shape `[nmo_α]` and `[nmo_β]`. Molecular orbital occupation numbers.
    ///
    /// # See also
    ///
    /// [`RHessElecInteractAPI::make_response_preparation`]. Signature difference: `mo_coeff` and
    /// `mo_occ` type different.
    fn make_response_preparation(&mut self, mo_coeff: &[TsrView; 2], mo_occ: &[TsrView; 2]);

    /// Get the response contribution for current SCF component.
    ///
    ///
    /// # Parameters
    ///
    /// - `bra` : shape `[nao, nocc_α, ...]` and `[nao, nocc_β, ...]`. The bra part for response
    ///   calculation.
    ///
    /// # Returns
    ///
    /// - `resp_bra` : shape `[nao, nocc_α, ...]` and `[nao, nocc_β, ...]`. The response potential
    ///   (related to second order of density matrix derivative to energy).
    ///
    /// # See also
    ///
    /// [`RHessElecInteractAPI::get_response_bra`]. Signature difference: input and output type
    /// different.
    fn get_response_bra(&mut self, bra: &[TsrView; 2]) -> [Tsr; 2];
}
