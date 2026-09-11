//! Generalized Fock matrix implementation for the core Hamiltonian, restricted.
//!
//! The gfock object [`RGFockHcore`] is a standalone object for the generalized Fock (and
//! Lagrangian) contribution of the core-Hamiltonian part of an energy functional, evaluated on a
//! fixed set of molecular orbitals (typically the converged SCF orbitals of another — the
//! SCF-iteration — functional, as required by double-hybrid type methods); it is assembled into
//! the double-hybrid composite by [`rgfock_dh_interface`](super::rgfock_interface).

use crate::analdrv::prelude::*;
use crate::analdrv::response::trait_rgfock::{GFockParts, RGFockAPI};
use enumflags2::BitFlags;

/// Generalized-Fock contribution of the core Hamiltonian, restricted.
///
/// The core-Hamiltonian energy $\mathrm{tr}(h D)$ is a fixed-operator density-only contribution:
/// its generalized Fock fills only the OO and VO blocks ($4 C_p^T h C_q$), its Lagrangian is
/// $4 C_v^T h C_o$, and its unrelaxed rdm1 is zero.
pub struct RGFockHcore {
    /// Core Hamiltonian, shape `[nao, nao]`.
    pub hcore: Tsr,
    /// Molecular orbital coefficients, shape `[nao, nmo]`.
    pub mo_coeff: Tsr,
    /// Occupation numbers, shape `[nmo]`.
    pub mo_occ: Tsr,
    /// Cached results, keyed by tensor name.
    pub intmd: HashMap<String, Tsr>,
    /// Timing information. Represented by wall time in second.
    pub timing: Vec<(String, f64)>,
}

impl RGFockHcore {
    /// Create from the AO-space core Hamiltonian.
    ///
    /// # Parameters
    ///
    /// - `hcore` : shape `[nao, nao]`. Core Hamiltonian.
    /// - `mo_coeff` : shape `[nao, nmo]`. Molecular orbital coefficients.
    /// - `mo_occ` : shape `[nmo]`. Occupation numbers.
    pub fn new(hcore: Tsr, mo_coeff: Tsr, mo_occ: Tsr) -> Self {
        Self { hcore, mo_coeff, mo_occ, intmd: HashMap::new(), timing: Vec::new() }
    }

    /// The occupied columns of the core Hamiltonian contracted into MO coefficients,
    /// `4 * h @ Co`, shape `[nao, nocc]`.
    fn make_fock_ao_occ(&mut self) -> Tsr {
        let occidx = self.mo_occ.view().greater(0).into_vec();
        let mocc = self.mo_coeff.bool_select(-1, &occidx).into_contig(ColMajor);
        &self.hcore % mocc.view()
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

impl AnalDrvBaseAPI for RGFockHcore {}

impl RGFockAPI for RGFockHcore {
    /// Generalized Fock of the core Hamiltonian: only the OO and VO blocks are filled
    /// ($4 C_p^T h C_q$ with $q$ occupied); the OV and VV blocks are identically zero.
    fn make_gfock<'r>(&mut self, _resp: Option<&mut (dyn RRespAPI + 'r)>, parts: BitFlags<GFockParts>) -> Tsr {
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
        gfock
    }

    /// Unrelaxed rdm1 of the core-Hamiltonian contribution: identically zero.
    fn make_rdm1(&mut self) -> Tsr {
        let nmo = self.nmo();
        let device = self.mo_coeff.device().clone();
        rt::zeros(([nmo, nmo].f(), &device))
    }

    /// Lagrangian of the core Hamiltonian: $4 C_v^T h C_o$, shape `[nvir, nocc]`. Cached on first
    /// call.
    fn make_lagrangian<'r>(&mut self, _resp: Option<&mut (dyn RRespAPI + 'r)>) -> Tsr {
        if !self.intmd.contains_key("lagrangian") {
            let nocc = self.nocc();
            let nmo = self.nmo();
            let sv = rt::slice!(nocc, nmo);
            let fock_ao_occ = self.make_fock_ao_occ();
            let mo = self.mo_coeff.view();
            let block: Tsr = mo.i((.., sv)).t() % fock_ao_occ.view();
            let lag: Tsr = (block * 4.0_f64).into_contig(ColMajor);
            self.intmd.insert("lagrangian".to_string(), lag);
        }
        self.intmd["lagrangian"].to_owned()
    }
}
