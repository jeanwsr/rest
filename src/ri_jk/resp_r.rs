//! Response (Fock and response matrix) implementation for RI-JK, restricted.
//!
//! The response object [`RRespRIJK`] is a standalone object holding the minimal state for the
//! fock/response functionality of the RI-JK electronic-interaction contribution; it is fully
//! independent of the skeleton-hessian machinery of `RHessRIJK` (which owns its own copy of the
//! factors and `cderi`).
//!
//! Fock generation routes through the pure in-core functions (`get_vj_ri_incore`,
//! `get_vk_ri_incore_dm`/`get_vk_ri_incore_coeff`) on the stored `cderi`, instead of any SCF-level
//! hamiltonian driver. The J/K factors are absorbed into the returned fock tensor, consistent with
//! how the hessian-side `get_deriv1_bra` applies the factors.

use super::prelude_dev::*;
use crate::analdrv::prelude::*;
use crate::ri_jk::hess_r::get_rijk_response_bra_separated;
use crate::ri_jk::pure_incore::{get_vj_ri_incore, get_vk_ri_incore_coeff, get_vk_ri_incore_dm};
use crate::ri_jk::util::get_dm0_restricted;

/// Response (fock/response matrix) object for RI-JK, restricted.
///
/// This object deliberately holds only what the response contractions need: the Cholesky
/// decomposed ERI, the J/K factors, and the response intermediates. It does **not** need
/// `mol`/`aux`/`j2c_decomp` — those belong to the skeleton-hessian machinery of the hessian
/// object.
///
/// Main-thread only (not `Send`/`Sync`).
pub struct RRespRIJK<'a> {
    /// Coulomb factor, absorbed into the returned fock/response tensors.
    pub factor_j: f64,
    /// Exchange factor, absorbed into the returned fock/response tensors.
    pub factor_k: f64,
    /// Cholesky decomposed 3c-2e ERI, shape `[nao_tp, naux]`.
    pub cderi: TsrCow<'a>,
    /// Response intermediates: `mo_coeff [nao, nmo]` (RowMajor) and `mo_occ [nmo]`, stored by
    /// [`RRespAPI::make_response_preparation`].
    pub intmd: HashMap<String, Tsr>,
}

impl<'a> RRespRIJK<'a> {
    /// Create from a Cholesky-decomposed ERI (`cderi`).
    pub fn new_with_cderi(factor_j: f64, factor_k: f64, cderi: TsrCow<'a>) -> Self {
        Self { factor_j, factor_k, cderi, intmd: HashMap::new() }
    }
}

impl<'a> AnalDrvBaseAPI for RRespRIJK<'a> {}

impl<'a> RRespAPI for RRespRIJK<'a> {
    /// Fock from density matrix: `factor_j * J(rdm) - 0.5 * factor_k * K(rdm)`.
    ///
    /// The `rdm` is assumed to be symmetric, as required by the pure in-core functions.
    fn get_fock_rdm(&mut self, rdm: TsrView) -> Tsr {
        assert_eq!(rdm.ndim(), 2, "rdm must have 2 dimensions");
        let nao = rdm.shape()[0];
        let device = self.cderi.device();

        // pure functions require (nao, nao, nset) f-contiguous inputs
        let dms = rdm.into_contig(ColMajor).into_shape((nao, nao, 1));

        // TODO: batch size `72` should be tunable by max-memory.
        let vj = get_vj_ri_incore(self.cderi.view(), dms.view()).i((.., .., 0)).into_contig(ColMajor);
        let vk = get_vk_ri_incore_dm(self.cderi.view(), dms.view(), 72).i((.., .., 0)).into_contig(ColMajor);

        let mut fock = rt::zeros(([nao, nao], device));
        if self.factor_j != 0.0 {
            fock += self.factor_j * &vj;
        }
        if self.factor_k != 0.0 {
            fock -= 0.5 * self.factor_k * &vk;
        }
        fock
    }

    /// Fock from molecular coefficients: J by the dm route (no coeff-native vj exists), K by the
    /// coeff-native pure function.
    fn get_fock_coeff(&mut self, mo_coeff: TsrView, mo_occ: TsrView) -> Tsr {
        let [nao, nmo] = mo_coeff.shape().to_vec().try_into().unwrap();
        let device = self.cderi.device();

        let dm = get_dm0_restricted(mo_coeff.view(), mo_occ.view());
        let dms = dm.into_contig(ColMajor).into_shape((nao, nao, 1));
        let vj = get_vj_ri_incore(self.cderi.view(), dms.view()).i((.., .., 0)).into_contig(ColMajor);

        let coeff = mo_coeff.into_contig(ColMajor).into_shape((nao, nmo, 1));
        let occ = mo_occ.into_contig(ColMajor).into_shape((nmo, 1));

        // TODO: batch size `72` should be tunable by max-memory.
        let vk =
            get_vk_ri_incore_coeff(self.cderi.view(), coeff.view(), occ.view(), 72).i((.., .., 0)).into_contig(ColMajor);

        let mut fock = rt::zeros(([nao, nao], device));
        if self.factor_j != 0.0 {
            fock += self.factor_j * &vj;
        }
        if self.factor_k != 0.0 {
            fock -= 0.5 * self.factor_k * &vk;
        }
        fock
    }

    fn get_response_rdm(&mut self, _rdm: TsrView) -> Tsr {
        unimplemented!("Response matrix (rdm form) is not implemented for RI-JK yet.")
    }

    fn make_response_preparation(&mut self, mo_coeff: TsrView, mo_occ: TsrView) {
        self.intmd.insert("mo_coeff".to_string(), mo_coeff.into_contig(RowMajor));
        self.intmd.insert("mo_occ".to_string(), mo_occ.to_owned());
    }

    fn get_response_bra(&mut self, bra: TsrView) -> Tsr {
        let mo_coeff = self.intmd["mo_coeff"].view();
        let mo_occ = self.intmd["mo_occ"].view();
        let cderi = self.cderi.view();

        // RHF (single spin) assembly of the separated J/K response core.
        // - J (AO form, from total density) contracted with `mocc` and scaled by `factor_j`.
        // - K (same-spin bra form) scaled by `factor_k`; the core already bakes in the exchange sign.
        // - RHF exchange prefactor (occ = 2) is folded into `factor_k`, matching the naive convention.
        let shape_bra = bra.shape().to_vec();
        let nao = mo_coeff.shape()[0];
        let device = mo_coeff.device();
        let occidx = mo_occ.view().greater(0).into_vec();
        let mocc = mo_coeff.bool_select(-1, &occidx);
        let nocc = mocc.shape()[1];
        let nprop: usize = shape_bra[2..].iter().product();

        // TODO: batch size `72` should be tunable by max-memory.
        let (j_ao, k_bras) = get_rijk_response_bra_separated(
            cderi,
            &[mo_coeff.view()],
            &[mo_occ.view()],
            &[bra.view()],
            self.factor_j != 0.0,
            self.factor_k != 0.0,
            72,
        );

        let mut resp: Tsr = rt::zeros(([nao, nocc, nprop], device));
        if let Some(resp_ao_j) = j_ao {
            resp += self.factor_j * (resp_ao_j % &mocc);
        }
        if let Some(k_bra) = k_bras.first() {
            // K bra is returned in the original trailing shape; flatten trailing dims to (nao, nocc, nprop).
            resp += self.factor_k * k_bra.view().reshape((nao, nocc, nprop));
        }
        resp.into_shape(shape_bra)
    }
}
