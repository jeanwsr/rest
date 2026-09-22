//! Response (Fock and response matrix) implementation for RI-JK, unrestricted.
//!
//! The response object [`URespRIJK`] is a standalone object holding the minimal state for the
//! fock/response functionality of the RI-JK electronic-interaction contribution of an unrestricted
//! SCF; it is fully independent of the skeleton-hessian machinery of `UHessRIJK` (which owns its
//! own copy of the factors and `cderi`).
//!
//! The response-bra core delegates to the shared separated J/K kernel
//! [`get_rijk_response_bra_separated`](super::resp_r::get_rijk_response_bra_separated) (also used
//! by the RHF response object): J is produced once in AO form from the total (α+β) density
//! response and right half-transformed per spin with a `0.5` prefactor; K is produced per spin in
//! bra form (same-spin only).

use super::prelude_dev::*;
use crate::analdrv::prelude::*;
use crate::ri_jk::pure_incore::{get_vj_ri_incore, get_vk_ri_incore_coeff, get_vk_ri_incore_dm};
use crate::ri_jk::resp_r::get_rijk_response_bra_separated;
use crate::ri_jk::util::get_dm0_restricted;

/// Response (fock/response matrix) object for RI-JK, unrestricted.
///
/// This object deliberately holds only what the response contractions need: the Cholesky
/// decomposed ERI, the J/K factors, and the response intermediates. It does **not** need
/// `mol`/`aux`/`j2c_decomp` — those belong to the skeleton-hessian machinery of the hessian
/// object.
///
/// Main-thread only (not `Send`/`Sync`).
pub struct URespRIJK<'a> {
    /// Coulomb factor, absorbed into the returned fock/response tensors.
    pub factor_j: f64,
    /// Exchange factor, absorbed into the returned fock/response tensors.
    pub factor_k: f64,
    /// Cholesky decomposed 3c-2e ERI, shape `[nao_tp, naux]`.
    pub cderi: TsrCow<'a>,
    /// Response intermediates: `mo_coeff_s [nao, nmo_s]` (RowMajor) and `mo_occ_s [nmo_s]`, stored
    /// by [`URespAPI::make_response_preparation`].
    pub intmd: HashMap<String, Tsr>,
}

impl<'a> URespRIJK<'a> {
    /// Create from a Cholesky-decomposed ERI (`cderi`).
    pub fn new_with_cderi(factor_j: f64, factor_k: f64, cderi: TsrCow<'a>) -> Self {
        Self { factor_j, factor_k, cderi, intmd: HashMap::new() }
    }
}

impl<'a> AnalDrvBaseAPI for URespRIJK<'a> {}

impl<'a> URespAPI for URespRIJK<'a> {
    /// Fock per spin from spin density matrices: `factor_j * J(rdm_α + rdm_β) - factor_k *
    /// K(rdm_s)`.
    ///
    /// The Coulomb block sees the total density; the exchange block is same-spin. Unlike the
    /// restricted [`RRespRIJK`](super::resp_r::RRespRIJK) there is no `0.5` on the exchange: each
    /// spin density carries occupation 1.
    ///
    /// The `rdm` is assumed to be symmetric, as required by the pure in-core functions.
    fn get_fock_rdm(&mut self, rdm: &[TsrView; 2]) -> [Tsr; 2] {
        let [α, β] = [0, 1];
        for rdm_s in rdm {
            assert_eq!(rdm_s.ndim(), 2, "rdm must have 2 dimensions");
        }
        let nao = rdm[α].shape()[0];
        let device = self.cderi.device();

        let mut fock = [rt::zeros(([nao, nao], device)), rt::zeros(([nao, nao], device))];
        if self.factor_j != 0.0 {
            // J: total (α+β) density, spin-independent
            let dm_total = &rdm[α] + &rdm[β];
            let dms = dm_total.into_contig(ColMajor).into_shape((nao, nao, 1));
            let vj = get_vj_ri_incore(self.cderi.view(), dms.view()).i((.., .., 0)).into_contig(ColMajor);
            fock[α] += self.factor_j * &vj;
            fock[β] += self.factor_j * &vj;
        }
        if self.factor_k != 0.0 {
            // K: same-spin exchange
            for (s, fock_s) in fock.iter_mut().enumerate() {
                let dms = rdm[s].view().into_contig(ColMajor).into_shape((nao, nao, 1));
                // TODO: batch size `72` should be tunable by max-memory.
                let vk = get_vk_ri_incore_dm(self.cderi.view(), dms.view(), 72).i((.., .., 0)).into_contig(ColMajor);
                *fock_s -= self.factor_k * &vk;
            }
        }
        fock
    }

    /// Fock per spin from molecular coefficients: J by the dm route (no coeff-native vj exists), K
    /// by the coeff-native pure function per spin.
    fn get_fock_coeff(&mut self, mo_coeff: &[TsrView; 2], mo_occ: &[TsrView; 2]) -> [Tsr; 2] {
        let [α, β] = [0, 1];
        let nao = mo_coeff[α].shape()[0];
        let device = self.cderi.device();

        let mut fock = [rt::zeros(([nao, nao], device)), rt::zeros(([nao, nao], device))];
        if self.factor_j != 0.0 {
            // J: total (α+β) density, spin-independent
            let dm_total = get_dm0_restricted(mo_coeff[α].view(), mo_occ[α].view())
                + get_dm0_restricted(mo_coeff[β].view(), mo_occ[β].view());
            let dms = dm_total.into_contig(ColMajor).into_shape((nao, nao, 1));
            let vj = get_vj_ri_incore(self.cderi.view(), dms.view()).i((.., .., 0)).into_contig(ColMajor);
            fock[α] += self.factor_j * &vj;
            fock[β] += self.factor_j * &vj;
        }
        if self.factor_k != 0.0 {
            // K: same-spin exchange, coeff-native
            for (s, fock_s) in fock.iter_mut().enumerate() {
                let nmo = mo_coeff[s].shape()[1];
                let coeff = mo_coeff[s].view().into_contig(ColMajor).into_shape((nao, nmo, 1));
                let occ = mo_occ[s].view().into_contig(ColMajor).into_shape((nmo, 1));
                // TODO: batch size `72` should be tunable by max-memory.
                let vk =
                    get_vk_ri_incore_coeff(self.cderi.view(), coeff.view(), occ.view(), 72).i((.., .., 0)).into_contig(ColMajor);
                *fock_s -= self.factor_k * &vk;
            }
        }
        fock
    }

    /// Response matrix per spin from (symmetrizable) spin density matrices: `2 * J(rdm_α +
    /// rdm_β) - 2 * K(rdm_s)`.
    ///
    /// This is the rdm-form entry of the unrestricted response kernel, consistent with the
    /// restricted `4 * J - 2 * K` convention of [`RRespRIJK`](super::resp_r::RRespRIJK) and with
    /// this object's own bra form (whose `0.5` Coulomb prefactor carries the same convention):
    /// per-spin occupation 1 against the restricted 2 halves the Coulomb scale, while the
    /// two-term symmetrized exchange kernel keeps the restricted scale. In the closed-shell
    /// limit (`rdm_α = rdm_β = X`), each spin's response equals the restricted kernel on `X`.
    /// No preparation is required.
    fn get_response_rdm(&mut self, rdm: &[TsrView; 2]) -> [Tsr; 2] {
        for rdm_s in rdm {
            assert_eq!(rdm_s.ndim(), 2, "rdm must have 2 dimensions");
        }
        let [α, β] = [0, 1];
        let nao = rdm[α].shape()[0];
        let device = self.cderi.device();

        // symmetrize (the pure in-core functions require symmetric input); the Coulomb response
        // sees the total (α+β) density
        let rdm_sym = [
            ((&rdm[α] + &rdm[α].t()) * 0.5).into_contig(ColMajor).into_shape((nao, nao, 1)),
            ((&rdm[β] + &rdm[β].t()) * 0.5).into_contig(ColMajor).into_shape((nao, nao, 1)),
        ];
        let rdm_total = (&rdm_sym[α] + &rdm_sym[β]).into_contig(ColMajor);

        let mut resp = [rt::zeros(([nao, nao], device)), rt::zeros(([nao, nao], device))];
        if self.factor_j != 0.0 {
            let vj = get_vj_ri_incore(self.cderi.view(), rdm_total.view()).i((.., .., 0)).into_contig(ColMajor);
            // 2.0 against the restricted 4.0: occupation 1 vs 2 on the total-density Coulomb
            resp[α] += 2.0 * self.factor_j * &vj;
            resp[β] += 2.0 * self.factor_j * &vj;
        }
        if self.factor_k != 0.0 {
            // K: same-spin exchange, same 2.0 scale as restricted (two-term symmetrized kernel)
            for (s, resp_s) in resp.iter_mut().enumerate() {
                // TODO: batch size `72` should be tunable by max-memory.
                let vk = get_vk_ri_incore_dm(self.cderi.view(), rdm_sym[s].view(), 72).i((.., .., 0)).into_contig(ColMajor);
                *resp_s -= 2.0 * self.factor_k * &vk;
            }
        }
        resp
    }

    fn make_response_preparation(&mut self, mo_coeff: &[TsrView; 2], mo_occ: &[TsrView; 2]) {
        self.intmd.insert("mo_coeff_0".to_string(), mo_coeff[0].view().into_contig(RowMajor));
        self.intmd.insert("mo_coeff_1".to_string(), mo_coeff[1].view().into_contig(RowMajor));
        self.intmd.insert("mo_occ_0".to_string(), mo_occ[0].to_owned());
        self.intmd.insert("mo_occ_1".to_string(), mo_occ[1].to_owned());
    }

    fn get_response_bra(&mut self, bra: &[TsrView; 2]) -> [Tsr; 2] {
        let mo_coeff = [self.intmd["mo_coeff_0"].view(), self.intmd["mo_coeff_1"].view()];
        let mo_occ = [self.intmd["mo_occ_0"].view(), self.intmd["mo_occ_1"].view()];
        let cderi = self.cderi.view();
        let device = mo_coeff[0].device();
        // Shared separated J/K response core: J (AO form, from total density) + per-spin K (bra form).
        let (j_ao, k_bras) = get_rijk_response_bra_separated(
            cderi,
            &mo_coeff,
            &mo_occ,
            bra,
            self.factor_j != 0.0,
            self.factor_k != 0.0,
            72, // TODO: batch size `72` should be tunable by max-memory.
        );

        let nao = mo_coeff[0].shape()[0];
        let occidx = [mo_occ[0].view().greater(0.0).into_vec(), mo_occ[1].view().greater(0.0).into_vec()];
        let mocc = [
            mo_coeff[0].view().bool_select(-1, &occidx[0]).into_contig(ColMajor),
            mo_coeff[1].view().bool_select(-1, &occidx[1]).into_contig(ColMajor),
        ];
        let nocc = [mocc[0].shape()[1], mocc[1].shape()[1]];

        let mut resp = [None, None];
        for s in 0..2 {
            let shape = bra[s].shape().to_vec();
            let nprop: usize = shape[2..].iter().product();
            let mut r = rt::zeros(([nao, nocc[s], nprop], device));
            // J: spin-independent AO operator, right half-transformed by this spin's mocc.
            // The shared `j_ao` carries the RHF symmetrization factor (effective `4 * J1`); UHF
            // naive J uses `2 * J1`, so an extra `0.5` prefactor is applied here (occ = 1 vs 2).
            if let Some(j_ao) = j_ao.as_ref() {
                r += 0.5 * self.factor_j * (j_ao.view() % &mocc[s]);
            }
            // K: same-spin bra form (UHF occ = 1, so no 0.5 factor — unlike RHF). The core already
            // bakes in the exchange sign, so this is an additive contribution.
            if let Some(k_bra) = k_bras.get(s) {
                r += self.factor_k * k_bra.view().reshape((nao, nocc[s], nprop));
            }
            resp[s] = Some(r.into_shape(shape));
        }
        [resp[0].take().unwrap(), resp[1].take().unwrap()]
    }
}
