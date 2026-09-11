//! Generalized Fock matrix implementation for RI-JK (incore), restricted.
//!
//! The gfock object [`RGFockRIJK`] is a standalone object for the generalized Fock (and
//! Lagrangian) contribution of the RI-JK Coulomb/exchange part of an energy functional, evaluated
//! on a fixed set of molecular orbitals (typically the converged SCF orbitals of another — the
//! SCF-iteration — functional, as required by double-hybrid type methods). It is the
//! generalized-Fock counterpart of the fock/response object [`RRespRIJK`](super::resp_r::RRespRIJK).
//!
//! Only the incore implementation is provided: the fock contribution is assembled by the pure
//! in-core functions (`get_vj_ri_incore`, `get_vk_ri_incore_coeff_bra`) on the stored `cderi`,
//! with the J/K factors absorbed into the returned tensors.
//!
//! The energy contribution this object represents, $\mathrm{tr}(V D)$ with
//! $V = c_J J[D] - \tfrac{1}{2} c_K K[D]$, depends on the molecular orbitals only through the
//! occupied-orbital density. Its generalized Fock therefore fills only the occupied columns
//! (the OO and VO blocks, with the closed-shell prefactor 4), and its Lagrangian is just the VO
//! block since the OV block vanishes identically. Both are driven by the half-transformed
//! occupied-column fock $V C_o$ alone: the exchange part is generated directly in this bra form
//! (no `(nao, nao)` exchange matrix is materialized), mirroring the bra form of the response
//! objects.

use super::prelude_dev::*;
use crate::analdrv::prelude::*;
use crate::analdrv::response::trait_rgfock::{GFockParts, RGFockAPI};
use crate::ri_jk::pure_incore::{get_vj_ri_incore, get_vk_ri_incore_coeff_bra};
use crate::ri_jk::util::get_dm0_restricted;
use enumflags2::BitFlags;

/// Generalized Fock object for the RI-JK (incore) Coulomb/exchange contribution, restricted.
///
/// This object deliberately holds only what the generalized-Fock evaluation needs: the Cholesky
/// decomposed ERI, the J/K factors of the energy functional it represents, and the molecular
/// orbitals on which the energy is evaluated. It does **not** need `mol`/`aux`/`j2c_decomp`.
///
/// Main-thread only (not `Send`/`Sync`).
pub struct RGFockRIJK<'a> {
    /// Coulomb factor, absorbed into the returned fock/gfock tensors.
    pub factor_j: f64,
    /// Exchange factor, absorbed into the returned fock/gfock tensors.
    pub factor_k: f64,
    /// Cholesky decomposed 3c-2e ERI, shape `[nao_tp, naux]`.
    pub cderi: TsrCow<'a>,
    /// Molecular orbital coefficients, shape `[nao, nmo]`.
    pub mo_coeff: Tsr,
    /// Occupation numbers, shape `[nmo]`.
    pub mo_occ: Tsr,
    /// Cached results, keyed by tensor name (`fock_ao_occ` : the occupied columns of the AO-space
    /// fock contribution).
    pub intmd: HashMap<String, Tsr>,
    /// Timing information. Represented by wall time in second.
    pub timing: Vec<(String, f64)>,
}

impl<'a> RGFockRIJK<'a> {
    /// Create from a Cholesky-decomposed ERI (`cderi`).
    ///
    /// # Parameters
    ///
    /// - `factor_j` : Coulomb factor of the energy functional (e.g. `1.0`).
    /// - `factor_k` : exchange factor of the energy functional (hybrid coefficient).
    /// - `cderi` : Cholesky decomposed 3c-2e ERI, shape `[nao_tp, naux]`.
    /// - `mo_coeff` : shape `[nao, nmo]`. Molecular orbital coefficients.
    /// - `mo_occ` : shape `[nmo]`. Occupation numbers.
    pub fn new_with_cderi(
        factor_j: f64,
        factor_k: f64,
        cderi: TsrCow<'a>,
        mo_coeff: Tsr,
        mo_occ: Tsr,
    ) -> Self {
        Self { factor_j, factor_k, cderi, mo_coeff, mo_occ, intmd: HashMap::new(), timing: Vec::new() }
    }

    /// The occupied columns of the AO-space fock contribution,
    /// `(factor_j * J[D] - 0.5 * factor_k * K[D]) @ Co`, on the density of the stored molecular
    /// orbitals. Shape `[nao, nocc]`. Cached on first call.
    ///
    /// J is evaluated by the dm route (no coeff-native vj exists), then contracted by `Co`; K is
    /// generated directly in this bra form by the coeff-native pure function
    /// [`get_vk_ri_incore_coeff_bra`], so that no `(nao, nao)` exchange matrix is materialized —
    /// mirroring the bra form of the response objects.
    pub fn make_fock_ao_occ(&mut self) -> Tsr {
        if !self.intmd.contains_key("fock_ao_occ") {
            let t0 = std::time::Instant::now();
            let [nao, nmo] = self.mo_coeff.shape().to_vec().try_into().unwrap();

            let occidx = self.mo_occ.view().greater(0).into_vec();
            let mocc = self.mo_coeff.bool_select(-1, &occidx).into_contig(ColMajor);

            let mut fock_occ: Option<Tsr> = None;
            if self.factor_j != 0.0 {
                let dm = get_dm0_restricted(self.mo_coeff.view(), self.mo_occ.view());
                let dms = dm.into_contig(ColMajor).into_shape((nao, nao, 1));
                let vj = get_vj_ri_incore(self.cderi.view(), dms.view()).i((.., .., 0)).into_contig(ColMajor);
                fock_occ = Some(self.factor_j * (vj % mocc.view()));
            }
            if self.factor_k != 0.0 {
                let coeff = self.mo_coeff.view().into_contig(ColMajor).into_shape((nao, nmo, 1));
                let occ = self.mo_occ.view().into_contig(ColMajor).into_shape((nmo, 1));

                // TODO: batch size `72` should be tunable by max-memory.
                let vk_occ = get_vk_ri_incore_coeff_bra(self.cderi.view(), coeff.view(), occ.view(), mocc.view(), 72)
                    .i((.., .., 0))
                    .into_contig(ColMajor);
                fock_occ = Some(match fock_occ {
                    Some(fock) => fock - 0.5 * self.factor_k * &vk_occ,
                    None => -0.5 * self.factor_k * &vk_occ,
                });
            }
            let fock_occ = fock_occ.expect("At least one of factor_j/factor_k must be non-zero.");

            self.intmd.insert("fock_ao_occ".to_string(), fock_occ);
            self.timing.push(("in RGFockRIJK, make_fock_ao_occ".to_string(), t0.elapsed().as_secs_f64()));
        }
        self.intmd["fock_ao_occ"].to_owned()
    }

    /// Number of occupied orbitals (occupation number greater than zero).
    pub fn nocc(&self) -> usize {
        self.mo_occ.view().greater(0).sum()
    }

    /// Total number of molecular orbitals.
    pub fn nmo(&self) -> usize {
        self.mo_occ.shape()[0]
    }
}

impl<'a> AnalDrvBaseAPI for RGFockRIJK<'a> {}

impl<'a> RGFockAPI for RGFockRIJK<'a> {
    /// Generalized Fock of the RI-JK contribution: only the OO and VO blocks are filled
    /// ($4 C_p^T V C_q$ with $q$ occupied); the OV and VV blocks are identically zero.
    ///
    /// The expensive AO→occupied contraction is cached by [`Self::make_fock_ao_occ`]; the MO
    /// transformation of the blocks is repeated per call, and is deterministic on the cached
    /// tensor.
    fn make_gfock<'r>(&mut self, _resp: Option<&mut (dyn RRespAPI + 'r)>, parts: BitFlags<GFockParts>) -> Tsr {
        let t0 = std::time::Instant::now();
        let nocc = self.nocc();
        let nmo = self.nmo();
        let so = rt::slice!(0, nocc);
        let sv = rt::slice!(nocc, nmo);
        let device = self.mo_coeff.device().clone();

        let fock_ao_occ = self.make_fock_ao_occ();
        let mo = self.mo_coeff.view();
        let mut gfock: Tsr = rt::zeros(([nmo, nmo].f(), &device));
        if parts.contains(GFockParts::OO) {
            let block = 4.0 * (mo.i((.., so)).t() % fock_ao_occ.view());
            *&mut gfock.i_mut((so, so)) += &block;
        }
        if parts.contains(GFockParts::VO) {
            let block = 4.0 * (mo.i((.., sv)).t() % fock_ao_occ.view());
            *&mut gfock.i_mut((sv, so)) += &block;
        }
        self.timing.push(("in RGFockRIJK, make_gfock".to_string(), t0.elapsed().as_secs_f64()));
        gfock
    }

    /// Unrelaxed rdm1 of the RI-JK contribution: identically zero (the density contribution of
    /// this functional part is carried by the SCF density itself, not by a correction density).
    fn make_rdm1(&mut self) -> Tsr {
        let nmo = self.nmo();
        let device = self.mo_coeff.device().clone();
        rt::zeros(([nmo, nmo].f(), &device))
    }

    /// Lagrangian of the RI-JK contribution: $L_{ai} = \mathscr{F}_{ai} - \mathscr{F}_{ia} =
    /// 4 C_v^T V C_o$ (the OV block vanishes), shape `[nvir, nocc]`. Cached on first call.
    fn make_lagrangian<'r>(&mut self, _resp: Option<&mut (dyn RRespAPI + 'r)>) -> Tsr {
        if !self.intmd.contains_key("lagrangian") {
            let t0 = std::time::Instant::now();
            let nocc = self.nocc();
            let nmo = self.nmo();
            let sv = rt::slice!(nocc, nmo);
            let fock_ao_occ = self.make_fock_ao_occ();
            let mo = self.mo_coeff.view();
            let block: Tsr = mo.i((.., sv)).t() % fock_ao_occ.view();
            let lag: Tsr = (block * 4.0_f64).into_contig(ColMajor);
            self.intmd.insert("lagrangian".to_string(), lag);
            self.timing.push(("in RGFockRIJK, make_lagrangian".to_string(), t0.elapsed().as_secs_f64()));
        }
        self.intmd["lagrangian"].to_owned()
    }
}
