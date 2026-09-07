use crate::analdrv::prelude::*;

/// Abstract class for Hessian-related API for nuclear repulsion contribution.
///
/// # Term Explanation
///
/// **Nuc component** here actually means the term is of zero-order with right of (electron) density
/// matrix. Usually this is only related to nuclear repulsion, but should denote all components that
/// do not depend on density matrix (like DFT-D3).
///
/// This hessian contribution is not related to density, so no RHF/UHF/GHF distinguishment required.
pub trait HessNucAPI: HessUtilAPI {
    fn make_skeleton_hess(&mut self, atm_list: Option<&[usize]>) -> Tsr;
}

/// Abstract class for Hessian-related API for restricted SCF core components.
///
/// # Term Explanation
///
/// **Core component** here actually means the term is of first-order with right of (electron)
/// density matrix.
///
/// - Core Hamiltonian is first-order (linear to density matrix).
/// - External field may have nuclear and electronic contributions. For dipole field, as an example,
///   the electronic contribution is of first-order, and can be counted in core-hamiltonian in some
///   frameworks.
///
/// We have function `make_skeleton_hess` here to count the **skeleton** contribution of the
/// Hessian. We do not handle derivative of density matrix here, which is the responsibility of CP-SCF
/// solver.
pub trait RHessCoreAPI: HessUtilAPI {
    /// Generate the **skeleton** contribution of Hessian for current SCF component.
    ///
    /// # Parameters
    ///
    /// - `mo_coeff` : shape `[nao, nmo]`. Molecular orbital coefficients.
    /// - `mo_occ` : shape `[nmo]`. Molecular orbital occupation numbers. In usual cases, the
    ///   occupied orbitals should have occupation 2, and virtual orbitals should have occupation 0.
    /// - `atm_list` : optional list of atom indices to compute the Hessian for. If `None`, all
    ///   atoms are computed.
    ///
    /// # Returns
    ///
    /// - `hess` : shape `[3, 3, natm, natm]`. The Hessian matrix for current SCF component, where
    ///   `natm = atm_list.len()` if `atm_list` is `Some`, else `mol.natm()`.
    ///
    ///   Note the hessian should be of indices `[s, t, B, A]` for column major.
    fn make_skeleton_hess(&mut self, mo_coeff: TsrView, mo_occ: TsrView, atm_list: Option<&[usize]>) -> Tsr;

    /// Generate the function to compute the first-order derivative of core component.
    ///
    /// This function only works for first-order density matrix contribution (like hcore). If this
    /// component does not contribute (like nuclear repulsion), return None.
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
    fn generator_deriv1(&self) -> Box<dyn FnMut(usize) -> Tsr>;
}

/// Abstract class for Hessian-related API for restricted SCF electronic interaction components.
///
/// # Term Explanation
///
/// **Electronic interaction** here actually means the term is of two-order (or higher-order) with
/// right of (electron) density matrix.
///
/// - J/K contribution from Hartree-Fock is exactly two-order.
/// - DFT contribution is non-linear to density matrix, and should be counted as infinity-order.
/// - Implicit-solvent/VV10 is probably categorized here.
///
/// In SCF iteration, introducing two-order (or higher-order) contribution requires the program to
/// make some modification to Fock matrix construction. This kind of terms is substentially
/// different from zero/one-order core components, and should be handled separately.
pub trait RHessElecInteractAPI: HessUtilAPI {
    /// Generate the **skeleton** contribution of Hessian for current SCF component.
    ///
    /// # Parameters
    ///
    /// - `mo_coeff` : shape `[nao, nmo]`. Molecular orbital coefficients.
    /// - `mo_occ` : shape `[nmo]`. Molecular orbital occupation numbers. In usual cases, the
    ///   occupied orbitals should have occupation 2, and virtual orbitals should have occupation 0.
    /// - `atm_list` : optional list of atom indices to compute the Hessian for. If `None`, all
    ///   atoms are computed.
    ///
    /// # Returns
    ///
    /// - `hess` : shape `[3, 3, natm, natm]`. The Hessian matrix for current SCF component.
    ///
    ///   Note the hessian should be of indices `[s, t, B, A]` for column major.
    fn make_skeleton_hess(&mut self, mo_coeff: TsrView, mo_occ: TsrView, atm_list: Option<&[usize]>) -> Tsr;

    /// First order skeleton derivative in AO basis.
    ///
    /// # Parameters
    ///
    /// - `mo_coeff` : shape `[nao, nmo]`. Molecular orbital coefficients.
    /// - `mo_occ` : shape `[nmo]`. Molecular orbital occupation numbers.
    /// - `atm_list` : optional list of atom indices over which derivatives are computed.
    ///
    /// # Returns
    ///
    /// - `deriv_ao` : shape `[nao, nao, 3, natm]`. The first-order skeleton derivative in AO basis.
    fn get_deriv1_ao(&mut self, mo_coeff: TsrView, mo_occ: TsrView, atm_list: Option<&[usize]>) -> Tsr;

    /// First order skeleton derivative in half-transformed MO basis.
    ///
    /// # Parameters
    ///
    /// - `mo_coeff` : shape `[nao, nmo]`. Molecular orbital coefficients.
    /// - `mo_occ` : shape `[nmo]`. Molecular orbital occupation numbers.
    /// - `atm_list` : optional list of atom indices over which derivatives are computed.
    ///
    /// # Returns
    ///
    /// - `deriv_bra` : shape `[nao, nocc, 3, natm]`. The first-order skeleton derivative in
    ///   half-transformed MO basis. Note that this function will handle the order of occupied
    ///   orbitals. If occupation number is not sorted contiguously, you may be extra cautious to
    ///   this function.
    ///
    /// # Notes
    ///
    /// If [`get_deriv1_ao`] implemented, this function should behave like `deriv_bra = deriv_ao @
    /// mocc`, where `mocc` is the occupied molecular coefficients (as ket).
    ///
    /// However, in some cases, it is probably better to skip the usage of [`get_deriv1_ao`] and
    /// directly use this function. By ket half-transformation, some RI-JK or DFT methods will
    /// benefit from boost by using low-rank occupied orbitals, instead of using full AO basis.
    ///
    /// # See also
    ///
    /// [`get_deriv1_ao`]
    ///
    /// [`get_deriv1_ao`]: Self::get_deriv1_ao
    fn get_deriv1_bra(&mut self, mo_coeff: TsrView, mo_occ: TsrView, atm_list: Option<&[usize]>) -> Tsr {
        let occidx = mo_occ.view().greater(0).into_vec();
        let mocc = mo_coeff.bool_select(-1, &occidx);
        self.get_deriv1_ao(mo_coeff, mo_occ, atm_list) % mocc
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

    /// Get the response contribution for current SCF component.
    ///
    /// This function will be called multiple-times in CP-SCF solver and other places.
    /// Call [`make_response_preparation`] before this function to make sure the data is ready.
    ///
    /// Also, this function will not pass in the MO coefficients and occupation numbers.
    /// If you need them, you should store them in the object by function
    /// `make_response_preparation`.
    ///
    /// # Parameters
    ///
    /// - `bra` : shape `[nao, nocc, ...]`. The bra part for response calculation. This is usually
    ///   the derivative of MO coefficients (like $U_{\mu i}^\mathbb{A}$ given by CP-SCF).
    ///
    /// # Returns
    ///
    /// - `resp_bra` : shape `[nao, nocc, ...]`. The response potential (related to second order of
    ///   density matrix derivative to energy).
    ///
    /// # Notes
    ///
    /// This function may not work for fractional occupation.
    /// We have not prepared to propose a good API for fractional occupation.
    ///
    /// [`make_response_preparation`]: Self::make_response_preparation
    fn get_response_bra(&mut self, bra: TsrView) -> Tsr;
}
