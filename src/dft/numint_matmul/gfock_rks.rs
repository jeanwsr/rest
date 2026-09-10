//! Generalized Fock matrix implementation for the RKS numint-matmul XC component.
//!
//! The gfock object [`RGFockKSNIMatmul`] is a standalone object for the generalized Fock (and
//! Lagrangian) contribution of the RKS numerical-integration XC part of an energy functional,
//! evaluated on a fixed set of molecular orbitals (typically the converged SCF orbitals of
//! another — the SCF-iteration — functional, as required by double-hybrid type methods). It is
//! the generalized-Fock counterpart of the fock/response object
//! [`RRespKSNIMatmul`](super::resp_rks::RRespKSNIMatmul).
//!
//! The generalized Fock is an energy evaluation, not a response: it is always evaluated on the
//! common (large) grid `ni` (usually the SCF grid); the response grid of the response objects is
//! irrelevant here.
//!
//! The XC energy depends on the molecular orbitals only through the occupied-orbital density, so
//! the generalized Fock fills only the occupied columns (the OO and VO blocks, with the
//! closed-shell prefactor 4), and the Lagrangian is just the VO block since the OV block
//! vanishes identically.

use super::prelude::*;
use crate::analdrv::prelude::*;
use crate::analdrv::response::trait_rgfock::{GFockParts, RGFockAPI};
use enumflags2::BitFlags;

use super::resp_rks::eval_vxc_fxc_from_rho;

/// Generalized Fock object for the RKS numint-matmul XC contribution.
pub struct RGFockKSNIMatmul<'a> {
    /// List of `(scale, functional)` pairs of the XC functional this object represents.
    pub xc_func_list: Vec<(f64, LibXCFunctional)>,
    /// Common (large) numerical-integration grid, usually the SCF grid.
    pub ni: NIMatmul<'a>,
    /// Molecular orbital coefficients, shape `[nao, nmo]`.
    pub mo_coeff: Tsr,
    /// Occupation numbers, shape `[nmo]`.
    pub mo_occ: Tsr,
    /// Cached results, keyed by tensor name (`fock_ao` : the AO-space fock contribution).
    pub intmd: HashMap<String, Tsr>,
    /// Timing information. Represented by wall time in second.
    pub timing: Vec<(String, f64)>,
}

impl<'a> RGFockKSNIMatmul<'a> {
    /// Create a new RKS generalized-Fock object.
    ///
    /// # Parameters
    ///
    /// - `xc_func_list` : list of `(scale, functional)` pairs of the energy functional.
    /// - `ni` : numerical-integration driver over the common (large) grid.
    /// - `mo_coeff` : shape `[nao, nmo]`. Molecular orbital coefficients.
    /// - `mo_occ` : shape `[nmo]`. Occupation numbers.
    pub fn new(
        xc_func_list: Vec<(f64, LibXCFunctional)>,
        ni: NIMatmul<'a>,
        mo_coeff: Tsr,
        mo_occ: Tsr,
    ) -> Self {
        Self { xc_func_list, ni, mo_coeff, mo_occ, intmd: HashMap::new(), timing: Vec::new() }
    }

    /// The AO-space XC numint fock contribution on the density of the stored molecular orbitals,
    /// evaluated on the common grid `ni`, shape `[nao, nao]`. Cached on first call.
    pub fn make_fock_ao(&mut self) -> Tsr {
        if !self.intmd.contains_key("fock_ao") {
            let t0 = std::time::Instant::now();
            let dm = get_dm0_restricted(self.mo_coeff.view(), self.mo_occ.view());
            let den_type =
                determine_den_type_from_list(&self.xc_func_list.iter().map(|(_, f)| f).collect_vec());
            let rho = self.ni.make_rho_from_dm(&[dm.view()], den_type);
            let (vxc, _fxc) = eval_vxc_fxc_from_rho(&self.xc_func_list, rho.i((.., .., 0)));
            let fock_ao = self.ni.make_vxc_pot_with_eff(vxc.view(), den_type, XCSpin::Unpolarized);
            self.intmd.insert("fock_ao".to_string(), fock_ao);
            self.timing
                .push(("in RGFockKSNIMatmul, make_fock_ao".to_string(), t0.elapsed().as_secs_f64()));
        }
        self.intmd["fock_ao"].to_owned()
    }

    /// The occupied columns of the AO-space fock contracted into MO coefficients,
    /// `4 * V @ Co`, shape `[nao, nocc]`.
    fn make_fock_ao_occ(&mut self) -> Tsr {
        let fock_ao = self.make_fock_ao();
        let occidx = self.mo_occ.view().greater(0).into_vec();
        let mocc = self.mo_coeff.bool_select(-1, &occidx).into_contig(ColMajor);
        &fock_ao % mocc.view()
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

impl<'a> AnalDrvBaseAPI for RGFockKSNIMatmul<'a> {}

impl<'a> RGFockAPI for RGFockKSNIMatmul<'a> {
    /// Generalized Fock of the XC contribution: only the OO and VO blocks are filled
    /// ($4 C_p^T V_{xc} C_q$ with $q$ occupied); the OV and VV blocks are identically zero.
    ///
    /// The expensive AO-space fock is cached by [`Self::make_fock_ao`]; the MO transformation of
    /// the blocks is repeated per call, and is deterministic on the cached fock.
    fn make_gfock(&mut self, _resp: Option<&mut dyn RRespAPI>, parts: BitFlags<GFockParts>) -> Tsr {
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
        self.timing.push(("in RGFockKSNIMatmul, make_gfock".to_string(), t0.elapsed().as_secs_f64()));
        gfock
    }

    /// Unrelaxed rdm1 of the XC contribution: identically zero (the density contribution of this
    /// functional part is carried by the SCF density itself, not by a correction density).
    fn make_rdm1(&mut self) -> Tsr {
        let nmo = self.nmo();
        let device = self.mo_coeff.device().clone();
        rt::zeros(([nmo, nmo].f(), &device))
    }

    /// Lagrangian of the XC contribution: $L_{ai} = \mathscr{F}_{ai} - \mathscr{F}_{ia} =
    /// 4 C_v^T V_{xc} C_o$ (the OV block vanishes), shape `[nvir, nocc]`. Cached on first call.
    fn make_lagrangian(&mut self, _resp: Option<&mut dyn RRespAPI>) -> Tsr {
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
            self.timing
                .push(("in RGFockKSNIMatmul, make_lagrangian".to_string(), t0.elapsed().as_secs_f64()));
        }
        self.intmd["lagrangian"].to_owned()
    }
}
