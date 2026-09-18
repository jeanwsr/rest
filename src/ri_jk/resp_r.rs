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
//!
//! The response-bra core [`get_rijk_response_bra_separated`] (shared by the RHF response object
//! and the UHF hessian's response path) also lives in this module.

use super::prelude_dev::*;
use crate::analdrv::prelude::*;
use crate::grad::rhf::pack_triu_tilde;
use crate::ri_jk::pure_incore::{get_vj_ri_incore, get_vk_ri_incore_coeff, get_vk_ri_incore_dm};
use crate::ri_jk::util::get_dm0_restricted;

/* #region response */

/// Separated J/K response-bra core, shared by RHF and UHF.
///
/// # Shapes
///
/// - `cderi`: `[nao_tp, naux]`
/// - `mo_coeff[s]`: `[nao, nmo_s]`, `mo_occ[s]`: `[nmo_s]`, `bra[s]`: `[nao, nocc_s, ...]` (the
///   trailing dimensions, collectively `nprop`, must agree across spins)
///
/// # Returns
///
/// A tuple `(j_ao, k_bras)`:
/// - `j_ao`: `Option<Tsr>` of shape `[nao, nao, nprop]` — the **spin-independent** Coulomb response
///   operator in AO basis, built from the total density response `sum_s bra_s @ mocc_s.T` (already
///   carrying the internal factor `2.0` from the symmetric cderi contraction; the consumer applies
///   `factor_j` and the per-spin right half-transform `... @ mocc_s`). `None` if `do_j` is false.
/// - `k_bras`: `Vec<Tsr>` (one entry per spin) of shape `[nao, nocc_s, nprop]` — the same-spin
///   exchange response in bra form (already carrying its internal sign/scale; the consumer applies
///   `factor_k`). Empty if `do_k` is false.
///
/// # Convention notes
///
/// - J sees the **total** density response, so a single AO operator is produced and shared across
///   spins; this is why UHF can reuse the RHF J path verbatim.
/// - K is strictly same-spin; each spin's bra form is produced independently.
/// - The internal factors (`2.0` on J, the two-term symmetrized sum on K) match the existing RHF
///   optimized response; the per-method `factor_j` / `factor_k` and the RHF `0.5` vs UHF `1.0`
///   exchange prefactor are applied by the consumer, not here.
#[allow(clippy::too_many_arguments)]
pub fn get_rijk_response_bra_separated(
    cderi: TsrView,
    mo_coeff: &[TsrView],
    mo_occ: &[TsrView],
    bra: &[TsrView],
    do_j: bool,
    do_k: bool,
    nbatch_aux: usize,
) -> (Option<Tsr>, Vec<Tsr>) {
    // notes on shape
    // - cderi: [nao_tp, naux]
    // - mo_coeff[s]: [nao, nmo_s]
    // - mo_occ[s]: [nmo_s]
    // - bra[s]: [nao, nocc_s, ...]  (trailing dims collectively `nprop`, same across spins)

    let nset = mo_coeff.len();
    assert_eq!(mo_occ.len(), nset);
    assert_eq!(bra.len(), nset);
    assert!(nset >= 1);

    let nao = mo_coeff[0].shape()[0];
    let naux = cderi.shape()[1];
    let nao_tp = nao * (nao + 1) / 2;
    assert_eq!(cderi.shape()[0], nao_tp);
    let device = cderi.device().clone();

    // per-spin occupied coefficients and reshaped bras
    let mocc: Vec<Tsr> = (0..nset)
        .map(|s| {
            let occidx = mo_occ[s].view().greater(0).into_vec();
            mo_coeff[s].view().bool_select(-1, &occidx)
        })
        .collect();
    let nocc: Vec<usize> = mocc.iter().map(|m| m.shape()[1]).collect();
    let bra_shape_orig: Vec<Vec<usize>> = bra.iter().map(|b| b.shape().to_vec()).collect();
    let bra: Vec<Tsr> = (0..nset).map(|s| bra[s].view().reshape((nao, nocc[s], -1)).into_contig(ColMajor)).collect();
    let nprop = bra[0].shape()[2];
    for s in 0..nset {
        assert_eq!(bra[s].shape()[2], nprop, "bra trailing dim (nprop) must agree across spins");
    }

    let mut j_ao: Option<Tsr> = None;
    let mut k_bras: Vec<Tsr> = Vec::new();

    // --- J contribution (spin-independent, AO form, from total density response) --- //

    if do_j {
        // dm1_total = sum_s (bra_s @ mocc_s.T), then symmetrize; pack with tilde; the symmetric
        // cderi contraction carries the internal factor 2.0 (matches the RHF optimized response).
        let mut dm1: Tsr = rt::zeros(([nao, nao, nprop], &device));
        for s in 0..nset {
            dm1 += &bra[s] % &mocc[s].t();
        }
        let dm1 = &dm1 + &dm1.swapaxes(0, 1);
        let dm1_tp = pack_triu_tilde(dm1.view());
        let itm_j_aux = cderi.t() % &dm1_tp;
        let resp_tp_j: Tsr = 2.0 * &cderi % itm_j_aux;
        j_ao = Some(resp_tp_j.unpack_tri(Upper, FlagSymm::Sy));
    }

    // --- K contribution (same-spin, bra form, two symmetrized terms) --- //

    if do_k {
        for s in 0..nset {
            let mocc_s = &mocc[s];
            let bra_s = &bra[s];
            let mut resp_bra_k: Tsr = rt::zeros_like(bra_s);
            for iaux_start in (0..naux).step_by(nbatch_aux) {
                let iaux_end = (iaux_start + nbatch_aux).min(naux);
                let slc = rt::slice!(iaux_start, iaux_end);
                // note: the following `naux` is the batch size, shadowing the outer one for brevity
                let naux = iaux_end - iaux_start;

                // - cderi: [nao, nao, naux]
                // - cderi_bxo: [nao, naux, nocc]
                // - cderi_oxo: [nocc, naux, nocc]
                // - cderi_box: [nao, nocc, naux]
                let cderi = cderi.i((.., slc)).unpack_tri(Upper, FlagSymm::Sy);
                let cderi_bxo = (cderi.reshape([nao, nao * naux]).t() % mocc_s).into_shape([nao, naux, nocc[s]]);
                let cderi_oxo =
                    (mocc_s.t() % cderi_bxo.reshape([nao, naux * nocc[s]])).into_shape([nocc[s], naux, nocc[s]]);

                for a in 0..nprop {
                    let bra_sa = bra_s.i((.., .., a));
                    let mut respka = resp_bra_k.i_mut((.., .., a));
                    // k contribution part 0: uPj, iPj -> ui
                    let cderi_bxo_1 = (cderi.reshape([nao, nao * naux]).t() % &bra_sa).into_shape([nao, naux, nocc[s]]);
                    respka -=
                        cderi_bxo_1.reshape([nao, naux * nocc[s]]) % cderi_oxo.reshape([nocc[s], naux * nocc[s]]).t();
                    // k contribution part 1: uPj, iPj -> ui (i from mocc, j from bra)
                    let cderi_oxo_1 =
                        (mocc_s.t() % cderi_bxo_1.reshape([nao, naux * nocc[s]])).into_shape([nocc[s], naux, nocc[s]]);
                    respka -=
                        cderi_bxo.reshape([nao, naux * nocc[s]]) % cderi_oxo_1.reshape([nocc[s], naux * nocc[s]]).t();
                }
            }
            // restore original trailing shape for this spin's bra
            let mut shape = bra_shape_orig[s].clone();
            shape[0] = nao;
            k_bras.push(resp_bra_k.into_shape(shape));
        }
    }

    (j_ao, k_bras)
}

/* #endregion */

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

    fn get_response_rdm(&mut self, rdm: TsrView) -> Tsr {
        // assume rdm can be symmetrized, 4 * J - 2 * K
        assert_eq!(rdm.ndim(), 2, "rdm must have 2 dimensions");
        let [nao, nao2] = rdm.shape().to_vec().try_into().unwrap();
        assert_eq!(nao, nao2, "rdm must be square");
        let device = self.cderi.device();

        let rdm_sym = ((&rdm + &rdm.t()) * 0.5).into_contig(ColMajor).into_shape((nao, nao, 1));

        let mut resp = rt::zeros(([nao, nao], device));
        if self.factor_j != 0.0 {
            let vj = get_vj_ri_incore(self.cderi.view(), rdm_sym.view()).i((.., .., 0)).into_contig(ColMajor);
            resp += 4.0 * self.factor_j * &vj;
        }
        // TODO: batch size `72` should be tunable by max-memory.
        if self.factor_k != 0.0 {
            let vk = get_vk_ri_incore_dm(self.cderi.view(), rdm_sym.view(), 72).i((.., .., 0)).into_contig(ColMajor);
            resp -= 2.0 * self.factor_k * &vk;
        }
        resp
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
