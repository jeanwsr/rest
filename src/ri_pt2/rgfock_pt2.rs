//! Generalized Fock (restricted) and related methods for RI-PT2.
//!
//! This module implements [`RGFockAPI`] for the RI-PT2 (RI-MP2-type) correlation contribution,
//! along with the Z-vector (relaxed density) machinery:
//!
//! - The generalized Fock contribution of PT2 comes in two modes. The default fills only the
//!   off-diagonal blocks: $\mathscr{F}_{ia} = - (W^\texttt{3}_{ai})^\top$ (OV) and
//!   $\mathscr{F}_{ai} = W^\texttt{4}_{ai}$ (VO), given by
//!   [`crate::ri_pt2::pure_pt2_r_elecderiv`]. Requests containing the OO or VV parts switch to
//!   the full four-block evaluation (batched over the auxiliary index), which carries the
//!   $2 \varepsilon \cdot \mathrm{rdm1}^{\mathrm{corr}}$ orbital-energy terms in the diagonal
//!   blocks. In both modes the SCF response upon the correlation rdm1
//!   ($A_{ai, pq} D_{pq}^{\mathrm{RDM}}$ term of the Lagrangian) is added to the VO block (and,
//!   in the full mode, to the OO block as well).
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
    /// `nocc`. The kernels are window-agnostic: any monotonic partition works, and degenerate
    /// occupied orbitals may be split across windows (each unordered pair is visited once and the
    /// per-window partial sums accumulate).
    pub index_occ_outer_vec: Vec<usize>,
    /// Opposite-spin correlation factor $c_\mathrm{OS}$.
    pub c_os: f64,
    /// Same-spin correlation factor $c_\mathrm{SS}$.
    pub c_ss: f64,
    /// Whether to additionally cache the amplitude intermediate $G_{ai}^\mathtt{P}$ under the
    /// result key `"g_vix"` (used by the analytic-gradient driver); costs an extra
    /// `nvir * nocc * naux` tensor. Must be set before the first evaluation (a cached kernel is
    /// not re-run; enabling it later panics rather than omitting `"g_vix"`).
    pub dump_g_vix: bool,
    /// Cached results, keyed by tensor name. Keys used by this struct: `"rdm1"`,
    /// `"gfock_part"`/`"gfock_part_full"` (electronic-derivative gfock contribution in the two
    /// modes of [`Self::make_elec_deriv`]), `"gfock_oo"`/`"gfock_ov"`/`"gfock_vo"`/`"gfock_vv"`
    /// (blocks of the generalized Fock), `"axd"`, `"axd_vo"`, `"lagrangian"`, `"z_vector"`,
    /// `"rdm1_resp"`.
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
        Self {
            mo_coeff,
            mo_occ,
            mo_energy,
            cderi,
            cderi_vox,
            index_occ_outer_vec,
            c_os,
            c_ss,
            dump_g_vix: false,
            result: HashMap::new(),
            e_corr: None,
            timing: Vec::new(),
        }
    }

    /// Evaluate the incore electronic derivative (correlation energy, unrelaxed rdm1, and the
    /// partial generalized Fock with its OV/VO blocks filled by the energy-weighted terms), and
    /// cache the results. Repeated calls directly return the cached results.
    pub fn make_elec_deriv(&mut self) {
        self.make_elec_deriv_with(false);
    }

    /// [`Self::make_elec_deriv`] with the gfock mode selected: `full_gfock` evaluates the full
    /// four-block contribution (OO+OV+VO+VV with the orbital-energy terms) in place of the
    /// W3/W4 off-diagonal blocks. The two modes cache their `gfock_part` under separate keys
    /// (`"gfock_part"`/`"gfock_part_full"`); a cached legacy evaluation is rerun on a full
    /// request (only `gfock_part` differs; e_corr and rdm1 are mode-independent), and a cached
    /// full evaluation serves any request.
    fn make_elec_deriv_with(&mut self, full_gfock: bool) {
        let key = if full_gfock { "gfock_part_full" } else { "gfock_part" };
        if self.result.contains_key(key) || (!full_gfock && self.result.contains_key("gfock_part_full")) {
            assert!(
                !self.dump_g_vix || self.result.contains_key("g_vix"),
                "dump_g_vix was enabled after the cached electronic-derivative evaluation; set it before the first make_elec_deriv call."
            );
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
        let arg = RPT2ElecDerivIncoreArg { c_os: self.c_os, c_ss: self.c_ss, full_gfock, dump_g_vix: self.dump_g_vix };
        let out = get_rpt2_elec_deriv_incore(&input, &arg, |x| O::from_f64(x).unwrap());

        self.e_corr = Some(out.e_corr);
        self.result.insert(key.to_string(), out.gfock_part);
        self.result.insert("rdm1".to_string(), out.rdm1_corr);
        if let Some(g_vix) = out.g_vix {
            self.result.insert("g_vix".to_string(), g_vix);
        }
        self.timing.push(("in RGFockPT2, make_elec_deriv".to_string(), t0.elapsed().as_secs_f64()));
    }

    /// The generalized-Fock contribution cached by [`Self::make_elec_deriv`]: the full
    /// four-block result if that mode ran (a cached full evaluation serves any request),
    /// otherwise the legacy W3/W4 off-diagonal result.
    fn gfock_part_cached(&self) -> &Tsr {
        self.result.get("gfock_part_full").or_else(|| self.result.get("gfock_part")).unwrap()
    }

    /// SCF response upon the (full MO-space) correlation rdm1, in MO basis:
    /// $A_{pq, \lambda} D_{\lambda}^{\mathrm{RDM}}$. Shape `[nmo, nmo]`.
    ///
    /// # Parameters
    ///
    /// - `resp` : response objects of the SCF-iteration functional.
    pub fn make_axd<'r>(&mut self, resp: &mut (dyn RRespAPI + 'r)) -> Tsr {
        if !self.result.contains_key("axd") {
            let t0 = std::time::Instant::now();
            self.make_elec_deriv();
            let rdm1 = self.result["rdm1"].view();
            let dm_ao = self.mo_coeff.view() % rdm1 % self.mo_coeff.view().t();
            // high precision: this rdm-form A-contraction belongs to the generalized-Fock
            // (energy-derivative) side and evaluates on the SCF-grade resource; a future input
            // keyword may open the low-precision (`prec = false`) window here as well
            let resp_ao = resp.get_response_rdm(dm_ao.view(), true);
            let axd_mo = self.mo_coeff.t() % resp_ao % self.mo_coeff.view();
            self.result.insert("axd".to_string(), axd_mo);
            self.timing.push(("in RGFockPT2, make_axd".to_string(), t0.elapsed().as_secs_f64()));
        }
        self.result["axd"].to_owned()
    }

    /// Vir-occupied block of [`Self::make_axd`]: $A_{ai, pq} D_{pq}^{\mathrm{RDM}}$. Shape
    /// `[nvir, nocc]`.
    ///
    /// # Parameters
    ///
    /// - `resp` : response objects of the SCF-iteration functional.
    pub fn make_axd_vo<'r>(&mut self, resp: &mut (dyn RRespAPI + 'r)) -> Tsr {
        if !self.result.contains_key("axd_vo") {
            let axd = self.make_axd(resp);
            let nocc = self.nocc();
            let nmo = self.nmo();
            let so = rt::slice!(0, nocc);
            let sv = rt::slice!(nocc, nmo);
            self.result.insert("axd_vo".to_string(), axd.i((sv, so)).into_contig(ColMajor));
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
            let gfock_part = self.gfock_part_cached().view();
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
    /// Generalized Fock of the PT2 contribution. Requests within OV/VO are served by the
    /// W3/W4 off-diagonal blocks; any request containing the OO or VV parts switches to the
    /// full four-block evaluation (batched over the auxiliary index), whose diagonal blocks
    /// carry the $2 \varepsilon \cdot \mathrm{rdm1}^{\mathrm{corr}}$ orbital-energy terms.
    /// The SCF response upon the correlation rdm1 ($A_{ai, pq} D_{pq}^{\mathrm{RDM}}$) enters
    /// the OO and VO blocks (the density does not respond to virtual-coefficient changes);
    /// OV and VV carry no response term. Parts that are not requested are left zero.
    ///
    /// Each block is evaluated once and cached in `self.result` (keys `"gfock_oo"`/
    /// `"gfock_ov"`/`"gfock_vo"`/`"gfock_vv"`); later calls, with whatever `flags`, directly
    /// reassemble the result from the cached blocks. The `resp` object is not part of the
    /// cache: it is consulted only while a block is actually being evaluated, and ignored
    /// (not compared, not re-contracted) on calls served from the cache. It is required only
    /// when the OO or VO part is evaluated and the SCF response term is not yet cached.
    fn make_gfock<'r>(&mut self, resp: Option<&mut (dyn RRespAPI + 'r)>, flags: BitFlags<GFockFlags>) -> Tsr {
        let t0 = std::time::Instant::now();
        let full = flags.contains(GFockFlags::OO) || flags.contains(GFockFlags::VV);
        self.make_elec_deriv_with(full);

        let nocc = self.nocc();
        let nmo = self.nmo();
        let so = rt::slice!(0, nocc);
        let sv = rt::slice!(nocc, nmo);

        // the SCF response term, required by the OO and VO blocks being evaluated
        let need_axd = (flags.contains(GFockFlags::OO) && !self.result.contains_key("gfock_oo"))
            || (flags.contains(GFockFlags::VO) && !self.result.contains_key("gfock_vo"));
        if need_axd && !self.result.contains_key("axd") {
            self.make_axd(resp.expect(
                "RI-PT2 generalized Fock OO/VO parts require the response object (for the SCF response upon rdm1_corr)",
            ));
        }

        // evaluate the requested blocks that are not yet cached
        let mut evaluated = false;
        for (flag, key, rows, cols, with_axd) in [
            (GFockFlags::OO, "gfock_oo", so, so, true),
            (GFockFlags::OV, "gfock_ov", so, sv, false),
            (GFockFlags::VO, "gfock_vo", sv, so, true),
            (GFockFlags::VV, "gfock_vv", sv, sv, false),
        ] {
            if !flags.contains(flag) || self.result.contains_key(key) {
                continue;
            }
            let mut block = self.gfock_part_cached().view().i((rows, cols)).into_contig(ColMajor);
            if with_axd {
                block += self.result["axd"].view().i((rows, cols));
            }
            self.result.insert(key.to_string(), block);
            evaluated = true;
        }

        // re-construct the matrix of the requested parts from the cached blocks
        let device = self.mo_coeff.device().clone();
        let mut gfock: Tsr = rt::zeros(([nmo, nmo].f(), &device));
        for (flag, key, rows, cols, _) in [
            (GFockFlags::OO, "gfock_oo", so, so, true),
            (GFockFlags::OV, "gfock_ov", so, sv, false),
            (GFockFlags::VO, "gfock_vo", sv, so, true),
            (GFockFlags::VV, "gfock_vv", sv, sv, false),
        ] {
            if flags.contains(flag) {
                gfock.i_mut((rows, cols)).assign(&self.result[key].view());
            }
        }
        if evaluated {
            self.timing.push(("in RGFockPT2, make_gfock".to_string(), t0.elapsed().as_secs_f64()));
        }
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
