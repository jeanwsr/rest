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
//!   $-(\varepsilon_a - \varepsilon_i) Z_{ai} - A_{ai, bj} Z_{bj} = L_{ai}$ with a block Krylov
//!   solver on the SCF response objects; the relaxed (response) density is then the
//!   unrelaxed rdm1 with its $[sv, so]$ block replaced by $Z_{ai}$.
//!
//! Reference implementations: `libincoreri` `src/ri_mp2.cpp` (W3/W4 terms), `pyincoreri`
//! `mp2_polar.py` (`get_rdm1_corr_resp`), and pyscf-forge `dh/resp.py` (`prepare_lagrangian`,
//! `prepare_D_r`).

use crate::analdrv::prelude::*;
use crate::analdrv::response::trait_rgfock::{GFockParts, RGFockAPI};
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
pub struct RPT2GFock<'a, 'b, O = f64>
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
    /// Response (fock/response) object of the underlying SCF method (the composite `RRespSCF`,
    /// borrowing the same SCF data as `cderi`). Used for the A-tensor contraction upon rdm1
    /// (Lagrangian) and the Z-vector (CP-SCF) solve.
    pub resp: &'b mut RRespSCF<'a>,
    /// Analytical-derivative configuration (CP-SCF solver settings).
    pub config: AnalDrvConfig,
    /// Cached results, keyed by tensor name.
    pub result: HashMap<String, Tsr>,
    /// Correlation energy of the most recent electronic-derivative evaluation, if any.
    pub e_corr: Option<f64>,
    /// Timing information. Represented by wall time in second.
    pub timing: Vec<(String, f64)>,
}

impl<'a, 'b, O> RPT2GFock<'a, 'b, O>
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
        resp: &'b mut RRespSCF<'a>,
        config: &AnalDrvConfig,
    ) -> Self {
        Self {
            mo_coeff,
            mo_occ,
            mo_energy,
            cderi,
            cderi_vox,
            index_occ_outer_vec,
            c_os,
            c_ss,
            resp,
            config: config.clone(),
            result: HashMap::new(),
            e_corr: None,
            timing: Vec::new(),
        }
    }

    /// Prepare the response of the SCF response object (must be called before the Z-vector
    /// solve or any response contraction). This also stores the orbital state in the response
    /// object for its inherent CP-SCF machinery.
    pub fn make_response_preparation(&mut self) {
        let t0 = std::time::Instant::now();
        self.resp.make_cpscf_preparation(self.mo_coeff.view(), self.mo_occ.view(), self.mo_energy.view());
        self.timing.push(("in RPT2GFock, make_response_preparation".to_string(), t0.elapsed().as_secs_f64()));
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
        self.timing.push(("in RPT2GFock, make_elec_deriv".to_string(), t0.elapsed().as_secs_f64()));
    }

    /// SCF response upon the (full MO-space) correlation rdm1, contracted to the vir-occupied
    /// block: $A_{ai, pq} D_{pq}^{\mathrm{RDM}}$. Shape `[nvir, nocc]`.
    pub fn make_axd_vo(&mut self) -> Tsr {
        if !self.result.contains_key("axd_vo") {
            let t0 = std::time::Instant::now();
            self.make_elec_deriv();
            let rdm1 = self.result["rdm1"].view();
            let dm_ao = self.mo_coeff.view() % rdm1 % self.mo_coeff.view().t();
            let resp_ao = self.resp.get_response_rdm(dm_ao.view());
            let axd_mo = self.mo_coeff.view().t() % resp_ao % self.mo_coeff.view();
            let nocc = self.nocc();
            let nmo = self.nmo();
            let so = rt::slice!(0, nocc);
            let sv = rt::slice!(nocc, nmo);
            self.result.insert("axd_vo".to_string(), axd_mo.i((sv, so)).into_contig(ColMajor));
            self.timing.push(("in RPT2GFock, make_axd_vo".to_string(), t0.elapsed().as_secs_f64()));
        }
        self.result["axd_vo"].to_owned()
    }

    /// Lagrangian $L_{ai}$ of the PT2 contribution. Shape `[nvir, nocc]`.
    ///
    /// This is the antisymmetrized partial generalized Fock
    /// ($\mathscr{F}_{ai} - \mathscr{F}_{ia} = W^\texttt{3}_{ai} + W^\texttt{4}_{ai}$) plus the
    /// SCF response upon the correlation rdm1, $A_{ai, pq} D_{pq}^{\mathrm{RDM}}$.
    pub fn make_lagrangian_vo(&mut self) -> Tsr {
        if !self.result.contains_key("lagrangian") {
            let t0 = std::time::Instant::now();
            self.make_elec_deriv();
            let nocc = self.nocc();
            let nmo = self.nmo();
            let so = rt::slice!(0, nocc);
            let sv = rt::slice!(nocc, nmo);
            let gfock_part = self.result["gfock_part"].view();
            let mut lag = gfock_part.i((sv, so)).to_owned() - gfock_part.i((so, sv)).t();
            lag += self.make_axd_vo().view();
            self.result.insert("lagrangian".to_string(), lag);
            self.timing.push(("in RPT2GFock, make_lagrangian_vo".to_string(), t0.elapsed().as_secs_f64()));
        }
        self.result["lagrangian"].to_owned()
    }

    /// Solve the Z-vector equation for the PT2 Lagrangian $L_{ai}$.
    ///
    /// The Z-vector equation (Handy-Schaefer) reads
    /// $-(\varepsilon_a - \varepsilon_i) Z_{ai} - A_{ai, bj} Z_{bj} = L_{ai}$, which is solved
    /// in the dimensionless form $Z + A(Z) / (\varepsilon_a - \varepsilon_i) = - L / (\varepsilon_a
    /// - \varepsilon_i)$ by the block Krylov solver of the response object
    /// ([`RRespSCF::solve_dimless_cpscf`]), where $A(Z)$ is the A-tensor action upon the
    /// perturbed density (cf. [`RRespAPI::get_response_bra`] and pyscf's `eri_cpks` response
    /// kernel).
    ///
    /// # Parameters
    ///
    /// - `lag_vo` : shape `[nvir, nocc]`. The Lagrangian $L_{ai}$.
    ///
    /// # Returns
    ///
    /// - `z_vo` : shape `[nvir, nocc]`. The Z-vector $Z_{ai}$.
    pub fn solve_z_vector(&mut self, lag_vo: TsrView) -> Tsr {
        let t0 = std::time::Instant::now();
        let device = self.mo_coeff.device().clone();
        let nocc = self.nocc();
        let nmo = self.nmo();
        let sv = rt::slice!(nocc, nmo);
        let (_, e_ai_shift) = self.e_ai();

        // dimensionless rhs: `- L / (e_a - e_i)` on the vir-occ block, zero on the occ block
        let mut rhs = rt::zeros(([nmo, nocc].f(), &device));
        rhs.i_mut(sv).assign(&(-lag_vo / &e_ai_shift));

        // solve `Z + A(Z)/(e_a - e_i) = rhs` by block Krylov on the response object
        let z = self.resp.solve_dimless_cpscf(rhs.view());

        self.result.insert("z_vector".to_string(), z.to_owned());
        self.timing.push(("in RPT2GFock, solve_z_vector".to_string(), t0.elapsed().as_secs_f64()));
        self.result["z_vector"].to_owned()
    }

    /// Relaxed (response) 1-RDM of the PT2 contribution in MO basis, shape `[nmo, nmo]`.
    ///
    /// This is the unrelaxed rdm1 with its vir-occupied block replaced by the Z-vector
    /// $Z_{ai}$ (cf. pyscf-forge `prepare_D_r`: `D_r[sv, so] = solve_cpks(L)`).
    pub fn make_rdm1_resp(&mut self) -> Tsr {
        if !self.result.contains_key("rdm1_resp") {
            let t0 = std::time::Instant::now();
            let lag = self.make_lagrangian_vo();
            let z = self.solve_z_vector(lag.view());
            let mut rdm1_resp = self.result["rdm1"].to_owned();
            let nocc = self.nocc();
            let nmo = self.nmo();
            let so = rt::slice!(0, nocc);
            let sv = rt::slice!(nocc, nmo);
            rdm1_resp.i_mut((sv, so)).assign(&z.i((sv, so)));
            self.result.insert("rdm1_resp".to_string(), rdm1_resp);
            self.timing.push(("in RPT2GFock, make_rdm1_resp".to_string(), t0.elapsed().as_secs_f64()));
        }
        self.result["rdm1_resp"].to_owned()
    }

    /// Number of occupied orbitals (occupation number greater than zero).
    pub fn nocc(&self) -> usize {
        self.mo_occ.view().greater(0).into_vec().iter().filter(|&&x| x).count()
    }

    /// Total number of molecular orbitals.
    pub fn nmo(&self) -> usize {
        self.mo_occ.shape()[0]
    }

    /// Orbital energy difference $(\varepsilon_a - \varepsilon_i)$ and its level-shifted
    /// version, both of shape `[nvir, nocc]`.
    fn e_ai(&self) -> (Tsr, Tsr) {
        let occidx = self.mo_occ.view().greater(0).into_vec();
        let viridx = occidx.iter().map(|&x| !x).collect_vec();
        let eocc = self.mo_energy.bool_select(-1, &occidx);
        let evir = self.mo_energy.bool_select(-1, &viridx);
        let e_ai = (&evir.i((.., None)) - &eocc.i((None, ..))).into_dim::<Ix2>();
        let level_shift = self.config.resp.level_shift;
        let e_ai_shift = if level_shift != 0.0 { &e_ai + level_shift } else { e_ai.view().to_owned() };
        (e_ai.into_dim::<IxD>(), e_ai_shift.into_dim::<IxD>())
    }
}

impl<O> AnalDrvBaseAPI for RPT2GFock<'_, '_, O>
where
    O: BlasFloat + 'static,
{
}

impl<O> RGFockAPI for RPT2GFock<'_, '_, O>
where
    O: BlasFloat + ToPrimitive + FromPrimitive + NumAssignOps + 'static,
{
    fn make_gfock(&mut self, _resp: Option<&impl RRespAPI>, parts: impl Into<BitFlags<GFockParts>>) -> Tsr {
        // The response object is held internally (`resp`); the optional argument is not used by
        // this implementation.
        let parts: BitFlags<GFockParts> = parts.into();
        assert!(
            parts.contains(GFockParts::OV) && parts.contains(GFockParts::VO),
            "RI-PT2 generalized Fock currently requires both OV and VO parts"
        );

        let t0 = std::time::Instant::now();
        self.make_elec_deriv();
        let mut gfock = self.result["gfock_part"].to_owned();
        // add SCF response upon rdm1_corr to the vir-occupied block
        let nocc = self.nocc();
        let nmo = self.nmo();
        let so = rt::slice!(0, nocc);
        let sv = rt::slice!(nocc, nmo);
        let axd_vo = self.make_axd_vo();
        *&mut gfock.i_mut((sv, so)) += &axd_vo;
        self.result.insert("gfock".to_string(), gfock.to_owned());
        self.timing.push(("in RPT2GFock, make_gfock".to_string(), t0.elapsed().as_secs_f64()));
        gfock
    }

    fn make_rdm1(&mut self) -> Tsr {
        self.make_elec_deriv();
        self.result["rdm1"].to_owned()
    }

    fn make_lagrangian(&mut self, _resp: Option<&impl RRespAPI>) -> Tsr {
        self.make_lagrangian_vo()
    }
}
