//! Generalized Fock (restricted) and related methods for RI-PT2.
//!
//! This module implements [`RGFockAPI`] for the RI-PT2 (RI-MP2-type) correlation contribution,
//! along with the Z-vector (relaxed density) machinery:
//!
//! - The generalized Fock contribution of PT2 has only the off-diagonal blocks filled:
//!   $\mathscr{F}_{ia} = - (W^\texttt{3}_{ai})^\top$ (OV) and $\mathscr{F}_{ai} =
//!   W^\texttt{4}_{ai}$ (VO), given by [`crate::ri_pt2::pure_pt2_r_elecderiv`]; the SCF
//!   response upon the correlation rdm1 ($A_{ai, pq} D_{pq}^{\mathrm{RDM}}$ term of the
//!   Lagrangian) is added to the VO block.
//! - The Lagrangian $L_{ai}$ is the antisymmetrized generalized Fock, which gives
//!   $L_{ai} = W^\texttt{3}_{ai} + W^\texttt{4}_{ai} + A_{ai, pq} D_{pq}^{\mathrm{RDM}}$.
//! - The Z-vector is obtained by solving the Z-vector (CP-SCF-type) equation
//!   $-(\varepsilon_a - \varepsilon_i) Z_{ai} - A_{ai, bj} Z_{bj} = L_{ai}$; this solve is
//!   DH-level machinery, provided by
//!   [`solve_z_vector`](crate::analdrv::response::rgfock_interface::solve_z_vector). The
//!   relaxed (response) density is then the unrelaxed rdm1 with its $[sv, so]$ block replaced
//!   by $Z_{ai}$.
//!
//! The object does **not** own the SCF response object: the response (and the CP-SCF state
//! behind it) belongs to the caller, and is passed through the assembling driver (e.g.
//! [`RGFockDH`] (crate::analdrv::response::rgfock_interface::RGFockDH), which likewise does not
//! hold it) into the functions that need it.
//!
//! Reference implementations: `libincoreri` `src/ri_mp2.cpp` (W3/W4 terms), `pyincoseri`
//! `mp2_polar.py` (`get_rdm1_corr_resp`), and pyscf-forge `dh/resp.py` (`prepare_lagrangian`,
//! `prepare_D_r`).

use crate::analdrv::prelude::*;
use crate::analdrv::response::rgfock_interface::solve_z_vector;
use crate::analdrv::response::trait_rgfock::{GFockFlags, RGFockAPI};
use crate::ri_pt2::pure_pt2_r_elecderiv::{
    get_rpt2_elec_deriv_incore, RPT2ElecDerivIncoreArg, RPT2ElecDerivIncoreInp,
};
use crate::utilities::rstsr_util::{Tsr, TsrCow, TsrView};
use enumflags2::BitFlags;
use itertools::Itertools;
use num::traits::NumAssignOps;
use num::{FromPrimitive, ToPrimitive};
use rstsr::prelude::*;
use rt::blas::BlasFloat;
use std::collections::HashMap;

/// Working solver and maintainer of all generalized-Fock (and related) components for RI-PT2.
///
/// The type parameter `O` is the working (floating-point) type of the incore
/// electronic-derivative kernel [`get_rpt2_elec_deriv_incore`] (`f64` or `f32`). The inputs in
/// `T`-position (`cderi` and the MO coefficients/energies) stay `f64`, and all outputs
/// (`e_corr`, `gfock_part`, `rdm1_corr`) are `f64` regardless of `O`.
///
/// The SCF response object is not stored: it is passed as a function argument wherever needed
/// ([`Self::make_axd_vo`]/[`Self::make_lagrangian_vo`]/[`Self::make_rdm1_resp`], and the
/// `resp` argument of the [`RGFockAPI`] methods). The response object is prepared by its owner
/// ([`RRespSCF::make_cpscf_preparation`]) with the same orbitals before these calls.
pub struct RGFockPT2<'a, O = f64>
where
    O: BlasFloat + 'static,
{
    /// Molecular orbital coefficients, shape `[nao, nmo]`.
    pub mo_coeff: Tsr,
    /// Occupation numbers, shape `[nmo]`.
    pub mo_occ: Tsr,
    /// Molecular orbital energies, shape `[nmo]`.
    pub mo_energy: Tsr,
    /// Cholesky-decomposed 3c2e ERI of the RI-PT2 auxiliary basis, shape `[nao_tp, naux]`.
    pub cderi: TsrCow<'a>,
    /// Optionally pre-transformed (vir, occ, aux) three-center integrals **in the working type
    /// `O`**; generated on the fly (transformed in f64, cast at the end) if not given. A caller
    /// holding an f64 tensor should regenerate it in `O` rather than re-cast, to avoid the extra
    /// allocation of a cast copy.
    pub cderi_vox: Option<TsrCow<'a, O>>,
    /// Batching (outer slice) indices of occupied orbitals; must start with `0` and end with
    /// `nocc`. Degenerate occupied orbitals must stay within one batch.
    pub index_occ_outer_vec: Vec<usize>,
    /// Opposite-spin correlation factor $c_\mathrm{OS}$.
    pub c_os: f64,
    /// Same-spin correlation factor $c_\mathrm{SS}$.
    pub c_ss: f64,
    /// Generalized Fock matrix in MO basis, accumulating the already-evaluated parts (only OV
    /// and VO are implemented); a zero matrix `[nmo, nmo]` at creation.
    pub gfock: Tsr,
    /// The parts of `gfock` that have been evaluated (only OV and VO can ever be set).
    pub gfock_flags: BitFlags<GFockFlags>,
    /// Cached results, keyed by tensor name.
    pub result: HashMap<String, Tsr>,
    /// Correlation energy of the most recent electronic-derivative evaluation, if any.
    pub e_corr: Option<f64>,
    /// Timing information. Represented by wall time in second.
    pub timing: Vec<(String, f64)>,
}

impl<'a, O> RGFockPT2<'a, O>
where
    O: BlasFloat + ToPrimitive + FromPrimitive + NumAssignOps + 'static,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mo_coeff: Tsr,
        mo_occ: Tsr,
        mo_energy: Tsr,
        cderi: TsrCow<'a>,
        cderi_vox: Option<TsrCow<'a, O>>,
        index_occ_outer_vec: Vec<usize>,
        c_os: f64,
        c_ss: f64,
    ) -> Self {
        let nmo = mo_occ.shape()[0];
        let device = mo_coeff.device().clone();
        Self {
            mo_coeff,
            mo_occ,
            mo_energy,
            cderi,
            cderi_vox,
            index_occ_outer_vec,
            c_os,
            c_ss,
            gfock: rt::zeros(([nmo, nmo].f(), &device)),
            gfock_flags: BitFlags::empty(),
            result: HashMap::new(),
            e_corr: None,
            timing: Vec::new(),
        }
    }

    /// Evaluate the incore electronic derivative (correlation energy, unrelaxed rdm1, and the
    /// partial generalized Fock with its OV/VO blocks filled by the energy-weighted terms), and
    /// cache the results. Repeated calls directly return the cached results.
    pub fn make_elec_deriv(&mut self) {
        if self.result.contains_key("rdm1") {
            return;
        }
        let t0 = std::time::Instant::now();

        let occidx = self.mo_occ.view().greater(0).into_vec();
        let viridx = occidx.iter().map(|&x| !x).collect_vec();
        let mocc = self.mo_coeff.bool_select(-1, &occidx).into_contig(ColMajor);
        let mvir = self.mo_coeff.bool_select(-1, &viridx).into_contig(ColMajor);
        let eocc = self.mo_energy.bool_select(-1, &occidx).into_contig(ColMajor);
        let evir = self.mo_energy.bool_select(-1, &viridx).into_contig(ColMajor);

        let input = RPT2ElecDerivIncoreInp {
            cderi: self.cderi.view(),
            cderi_vox: self.cderi_vox.as_ref().map(|x| x.view()),
            occ_coeff: mocc.view(),
            vir_coeff: mvir.view(),
            occ_energy: eocc.view(),
            vir_energy: evir.view(),
            index_occ_outer_vec: &self.index_occ_outer_vec,
        };
        let arg = RPT2ElecDerivIncoreArg { c_os: self.c_os, c_ss: self.c_ss };
        let out = get_rpt2_elec_deriv_incore(&input, &arg, |x| O::from_f64(x).unwrap());

        self.e_corr = Some(out.e_corr);
        self.result.insert("gfock_part".to_string(), out.gfock_part);
        self.result.insert("rdm1".to_string(), out.rdm1_corr);
        self.timing.push(("in RGFockPT2, make_elec_deriv".to_string(), t0.elapsed().as_secs_f64()));
    }

    /// SCF response upon the (full MO-space) correlation rdm1, contracted to the vir-occupied
    /// block: $A_{ai, pq} D_{pq}^{\mathrm{RDM}}$. Shape `[nvir, nocc]`.
    ///
    /// # Parameters
    ///
    /// - `resp` : response objects of the SCF-iteration functional.
    pub fn make_axd_vo<'r>(&mut self, resp: &mut (dyn RRespAPI + 'r)) -> Tsr {
        if !self.result.contains_key("axd_vo") {
            let t0 = std::time::Instant::now();
            self.make_elec_deriv();
            let rdm1 = self.result["rdm1"].view();
            let dm_ao = self.mo_coeff.view() % rdm1 % self.mo_coeff.view().t();
            let resp_ao = resp.get_response_rdm(dm_ao.view());
            let axd_mo = self.mo_coeff.view().t() % resp_ao % self.mo_coeff.view();
            let nocc = self.nocc();
            let nmo = self.nmo();
            let so = rt::slice!(0, nocc);
            let sv = rt::slice!(nocc, nmo);
            self.result.insert("axd_vo".to_string(), axd_mo.i((sv, so)).into_contig(ColMajor));
            self.timing.push(("in RGFockPT2, make_axd_vo".to_string(), t0.elapsed().as_secs_f64()));
        }
        self.result["axd_vo"].to_owned()
    }

    /// Lagrangian $L_{ai}$ of the PT2 contribution. Shape `[nvir, nocc]`.
    ///
    /// This is the antisymmetrized partial generalized Fock
    /// ($\mathscr{F}_{ai} - \mathscr{F}_{ia} = W^\texttt{3}_{ai} + W^\texttt{4}_{ai}$) plus the
    /// SCF response upon the correlation rdm1, $A_{ai, pq} D_{pq}^{\mathrm{RDM}}$.
    ///
    /// # Parameters
    ///
    /// - `resp` : response objects of the SCF-iteration functional.
    pub fn make_lagrangian_vo<'r>(&mut self, resp: &mut (dyn RRespAPI + 'r)) -> Tsr {
        if !self.result.contains_key("lagrangian") {
            let t0 = std::time::Instant::now();
            self.make_elec_deriv();
            let nocc = self.nocc();
            let nmo = self.nmo();
            let so = rt::slice!(0, nocc);
            let sv = rt::slice!(nocc, nmo);
            let gfock_part = self.result["gfock_part"].view();
            let mut lag = gfock_part.i((sv, so)).to_owned() - gfock_part.i((so, sv)).t();
            lag += self.make_axd_vo(resp).view();
            self.result.insert("lagrangian".to_string(), lag);
            self.timing.push(("in RGFockPT2, make_lagrangian_vo".to_string(), t0.elapsed().as_secs_f64()));
        }
        self.result["lagrangian"].to_owned()
    }

    /// Relaxed (response) 1-RDM of the PT2 contribution in MO basis, shape `[nmo, nmo]`.
    ///
    /// This is the unrelaxed rdm1 with its vir-occupied block replaced by the Z-vector
    /// $Z_{ai}$ (cf. pyscf-forge `prepare_D_r`: `D_r[sv, so] = solve_cpks(L)`), solved by the
    /// DH-level [`solve_z_vector`](crate::analdrv::response::rgfock_interface::solve_z_vector).
    ///
    /// # Parameters
    ///
    /// - `resp` : response objects of the SCF-iteration functional; must have been prepared
    ///   ([`RRespSCF::make_cpscf_preparation`]) with the same orbitals.
    pub fn make_rdm1_resp(&mut self, resp: &mut RRespSCF) -> Tsr {
        if !self.result.contains_key("rdm1_resp") {
            let t0 = std::time::Instant::now();
            let lag = self.make_lagrangian_vo(resp);
            let z = solve_z_vector(lag.view(), resp);
            self.result.insert("z_vector".to_string(), z.to_owned());
            let mut rdm1_resp = self.result["rdm1"].to_owned();
            let nocc = self.nocc();
            let nmo = self.nmo();
            let so = rt::slice!(0, nocc);
            let sv = rt::slice!(nocc, nmo);
            rdm1_resp.i_mut((sv, so)).assign(&z.i((sv, so)));
            self.result.insert("rdm1_resp".to_string(), rdm1_resp);
            self.timing.push(("in RGFockPT2, make_rdm1_resp".to_string(), t0.elapsed().as_secs_f64()));
        }
        self.result["rdm1_resp"].to_owned()
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

impl<O> AnalDrvBaseAPI for RGFockPT2<'_, O>
where
    O: BlasFloat + 'static,
{
}

impl<O> RGFockAPI for RGFockPT2<'_, O>
where
    O: BlasFloat + ToPrimitive + FromPrimitive + NumAssignOps + 'static,
{
    /// Generalized Fock of the PT2 contribution. Only the OV and VO parts are implemented
    /// ($\mathscr{F}_{ia} = - (W^\texttt{3}_{ai})^\top$ and $\mathscr{F}_{ai} =
    /// W^\texttt{4}_{ai}$, plus the SCF response upon the correlation rdm1
    /// $A_{ai, pq} D_{pq}^{\mathrm{RDM}}$ in the VO block); requesting the OO or VV parts is
    /// rejected. Parts that are not requested are left zero. Evaluated parts are accumulated in
    /// `self.gfock` (tracked by `self.gfock_flags`) and directly reused on later calls. The
    /// response object is required only when the VO part is actually evaluated.
    fn make_gfock<'r>(&mut self, resp: Option<&mut (dyn RRespAPI + 'r)>, flags: BitFlags<GFockFlags>) -> Tsr {
        assert!(
            !flags.contains(GFockFlags::OO) && !flags.contains(GFockFlags::VV),
            "RI-PT2 generalized Fock has only the OV and VO parts implemented; OO and VV are rejected"
        );

        let t0 = std::time::Instant::now();
        let nocc = self.nocc();
        let nmo = self.nmo();
        let so = rt::slice!(0, nocc);
        let sv = rt::slice!(nocc, nmo);

        // evaluate the requested parts that are not yet in `self.gfock`
        if flags.contains(GFockFlags::OV) && !self.gfock_flags.contains(GFockFlags::OV) {
            self.make_elec_deriv();
            *&mut self.gfock.i_mut((so, sv)) += &self.result["gfock_part"].view().i((so, sv));
            self.gfock_flags.insert(GFockFlags::OV);
        }
        if flags.contains(GFockFlags::VO) && !self.gfock_flags.contains(GFockFlags::VO) {
            self.make_elec_deriv();
            let resp = resp.expect(
                "RI-PT2 generalized Fock VO part requires the response object (for the SCF response upon rdm1_corr)",
            );
            // W^4 term, plus the SCF response upon rdm1_corr, in the vir-occupied block
            *&mut self.gfock.i_mut((sv, so)) += &self.result["gfock_part"].view().i((sv, so));
            let axd_vo = self.make_axd_vo(resp);
            *&mut self.gfock.i_mut((sv, so)) += &axd_vo;
            self.gfock_flags.insert(GFockFlags::VO);
        }

        // re-construct the matrix of the requested parts, not the accumulated `self.gfock`
        let device = self.mo_coeff.device().clone();
        let mut gfock: Tsr = rt::zeros(([nmo, nmo].f(), &device));
        if flags.contains(GFockFlags::OV) {
            gfock.i_mut((so, sv)).assign(&self.gfock.i((so, sv)));
        }
        if flags.contains(GFockFlags::VO) {
            gfock.i_mut((sv, so)).assign(&self.gfock.i((sv, so)));
        }
        self.timing.push(("in RGFockPT2, make_gfock".to_string(), t0.elapsed().as_secs_f64()));
        gfock
    }

    fn make_rdm1(&mut self) -> Tsr {
        self.make_elec_deriv();
        self.result["rdm1"].to_owned()
    }

    fn make_lagrangian<'r>(&mut self, resp: Option<&mut (dyn RRespAPI + 'r)>) -> Tsr {
        let resp = resp.expect("RI-PT2 Lagrangian requires the response object (for the SCF response upon rdm1_corr)");
        self.make_lagrangian_vo(resp)
    }
}
