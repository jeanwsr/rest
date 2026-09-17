//! Generalized Fock (restricted) driver and interface for double-hybrid type methods.
//!
//! This module assembles the generalized-Fock (and Lagrangian, Z-vector) machinery of a
//! restricted double-hybrid (DH) calculation from converged SCF data, in the same spirit as
//! [`rscf_resp_interface`](super::rresp_interface::rscf_resp_interface) assembles the response
//! objects. The composite [`RGFockDH`] sums the contributions of
//!
//! - the **final-energy (non-variational) functional's** parts that depend on the density matrix
//!   only — the core Hamiltonian ([`RGFockHcore`]), the RI-JK Coulomb/exchange ([`RGFockRIJK`])
//!   with the final functional's hybrid coefficient, and the DFT XC numint part
//!   ([`RGFockKSNIMatmul`]) on the common SCF grid. Each contributes only the OO/VO blocks of the
//!   generalized Fock ($4 C_p^T V C_q$) and an empty unrelaxed rdm1, so their Lagrangian is just
//!   the VO block $4 C_v^T V C_o$. The core-Hamiltonian/Coulomb parts are shared with the SCF
//!   functional; their Lagrangian contribution cancels through the canonical diagonality of the
//!   converged SCF Fock (orbital stationarity), exactly as in pyscf-forge's `prepare_lagrangian`.
//! - the **correlation (RI-PT2) contribution** ([`RGFockPT2`]), which fills the OV/VO blocks
//!   (`W`-terms), the unrelaxed rdm1, and the SCF response upon that rdm1 — an element of the
//!   contribution list like the others, not a special member.
//!
//! The driver does not own the SCF-iteration functional's response object ([`RRespSCF`]): it is
//! supplied by the caller as a mutable argument to the methods that need it — the A-tensor
//! contractions and the Z-vector (CP-SCF) solve ([`solve_z_vector`], the DH-level Handy-Schaefer
//! solve); the relaxed density is the unrelaxed rdm1 with its vir-occupied block replaced by the
//! Z-vector.
//!
//! The contribution list is open-ended: further post-SCF contributions (e.g. RPA-type) can be
//! appended to [`RGFockDH::gfock_list`] without changing the type.

use crate::analdrv::prelude::*;
use crate::analdrv::response::rgfock_hcore::RGFockHcore;
use crate::analdrv::response::trait_rgfock::{GFockFlags, RGFockAPI};
use crate::dft::numint_matmul::gfock_rks::RGFockKSNIMatmul;
use crate::dft::numint_matmul::nimatmul::NIMatmul;
use crate::dft::DFAFamily;
use crate::ri_jk::gfock_r::RGFockRIJK;
use crate::ri_jk::util::get_cint_mol;
use crate::ri_pt2::rgfock_pt2::RGFockPT2;
use crate::ri_pt2::PT2FPMode;
use crate::SCF;

use enumflags2::BitFlags;
use libxc::prelude::*;

/// J/K factors of the RI-JK part of the (restricted) final-energy functional of a DH.
///
/// The final functional carries the full Coulomb (`factor_j = 1`) and its own hybrid coefficient
/// `dfa_hybrid_pos` (for an HF final functional without DFT parts, the full exchange). Compare
/// [`scf_jk_factors`](super::rresp_interface::scf_jk_factors) for the SCF-iteration functional.
pub fn dh_jk_factors(scf_data: &SCF) -> (f64, f64) {
    let xc_data = &scf_data.mol.xc_data;
    // the final functional's total hybrid coefficient; absent `dfa_hybrid_pos` (post-SCF family
    // without its own functional), the final functional is the SCF functional. For an HF final
    // functional (e.g. MP2 correlation on HF), this correctly yields the full exchange 1.0.
    let factor_k = xc_data.dfa_hybrid_pos.unwrap_or(xc_data.dfa_hybrid_scf);
    (1.0, factor_k)
}

/// List of `(scale, functional)` pairs of the final-energy functional's DFT (XC numint) part
/// (spin-unpolarized); `None` when the final functional has no DFT part.
pub fn dh_xc_func_list(scf_data: &SCF) -> Option<Vec<(f64, LibXCFunctional)>> {
    let xc_data = &scf_data.mol.xc_data;
    let (xc_code, xc_params) = match (&xc_data.dfa_compnt_pos, &xc_data.dfa_paramr_pos) {
        // a present-but-empty component list (pure-MP2-family post-SCF methods) carries no DFT
        // part; treated the same as absent, so no empty KS contribution object is built
        (Some(code), Some(_)) if code.is_empty() => return None,
        (Some(code), Some(param)) => (code, param),
        (None, None) => return None,
        _ => panic!("dfa_compnt_pos and dfa_paramr_pos must be present or absent together."),
    };
    Some(
        xc_code
            .iter()
            .zip(xc_params.iter())
            .map(|(&code, &param)| (param, LibXCFunctional::from_number(code as _, LibXCSpin::Unpolarized)))
            .collect_vec(),
    )
}

/// Solve the Z-vector (CP-SCF) equation for a Lagrangian of the (combined) post-SCF contributions.
///
/// The Z-vector equation (Handy-Schaefer) reads
/// $-(\varepsilon_a - \varepsilon_i) Z_{ai} - A_{ai, bj} Z_{bj} = L_{ai}$, which is solved
/// in the dimensionless form $Z + A(Z) / (\varepsilon_a - \varepsilon_i) = - L / (\varepsilon_a -
/// \varepsilon_i)$ by the block Krylov solver of the response object
/// ([`RRespSCF::solve_dimless_cpscf`]), where $A(Z)$ is the A-tensor action upon the
/// perturbed density (cf. [`RRespAPI::get_response_bra`] and pyscf's `eri_cpks` response
/// kernel).
///
/// This is a double-hybrid level machinery — the orbital response of the SCF-iteration functional
/// upon the non-variational final-energy Lagrangian — not specific to any single correlation
/// contribution. The level-shifted orbital-energy differences and the occupied/virtual split are
/// taken from the CP-SCF state cached by [`RRespSCF::make_cpscf_preparation`], so the level shift
/// is that of the response object.
///
/// # Parameters
///
/// - `lag_vo` : shape `[nvir, nocc]`. The Lagrangian $L_{ai}$.
/// - `resp` : response objects of the SCF-iteration functional; must have been prepared
///   ([`RRespSCF::make_cpscf_preparation`]) with the same orbitals.
///
/// # Returns
///
/// - `z_vo` : shape `[nmo, nocc]`. The Z-vector $Z_{ai}$ (vir-occupied block).
pub fn solve_z_vector(lag_vo: TsrView, resp: &mut RRespSCF) -> Tsr {
    let state = resp.cpscf_state();
    let nmo = state.nmo;
    let nocc = state.nocc;
    let device = state.e_ai_shift.device().clone();
    let sv = rt::slice!(nocc, nmo);

    // dimensionless rhs: `- L / (e_a - e_i)` on the vir-occ block, zero on the occ block
    let mut rhs = rt::zeros(([nmo, nocc].f(), &device));
    rhs.i_mut(sv).assign(&(-lag_vo / &state.e_ai_shift));

    // solve `Z + A(Z)/(e_a - e_i) = rhs` by block Krylov on the response object
    resp.solve_dimless_cpscf(rhs.view())
}

/// Working solver and maintainer of all generalized-Fock (and related) components of a restricted
/// double-hybrid type method.
///
/// The type holds the generalized-Fock contribution objects of the final-energy functional's
/// density-only parts (core Hamiltonian, RI-JK, DFT XC; each filling only OO/VO blocks) and the
/// RI-PT2 correlation contribution ([`RGFockPT2`]; OV/VO blocks, unrelaxed rdm1) — all as
/// elements of the contribution list, treated uniformly through [`RGFockAPI`]. The composite
/// implementation sums the contributions; the relaxed (response) density is the unrelaxed rdm1
/// with its vir-occupied block replaced by the Z-vector solved from the combined Lagrangian.
///
/// The response object of the SCF-iteration functional is not stored (cf. [`RGFockPT2`], which
/// follows the same convention): it is passed as a mutable argument to the methods that need it.
///
/// # Structure
///
/// - `gfock_list` : the contribution objects (including the PT2 correlation one); open-ended for
///   further contributions.
///
pub struct RGFockDH<'a> {
    /// Generalized-Fock contribution objects: the final-energy functional's density-only parts
    /// (core Hamiltonian, RI-JK Coulomb/exchange, DFT XC numint) and the RI-PT2 correlation
    /// contribution.
    pub gfock_list: Vec<Box<dyn RGFockAPI + 'a>>,
    /// Molecular orbital coefficients, shape `[nao, nmo]`.
    pub mo_coeff: Tsr,
    /// Occupation numbers, shape `[nmo]`.
    pub mo_occ: Tsr,
    /// Molecular orbital energies, shape `[nmo]`.
    pub mo_energy: Tsr,
    /// Generalized Fock matrix in MO basis, accumulating the already-evaluated parts; a zero
    /// matrix `[nmo, nmo]` at creation.
    pub gfock: Tsr,
    /// The parts of `gfock` that have been evaluated.
    pub gfock_flags: BitFlags<GFockFlags>,
    /// Cached results, keyed by tensor name.
    pub result: HashMap<String, Tsr>,
    /// Timing information. Represented by wall time in second.
    pub timing: Vec<(String, f64)>,
}

impl<'a> RGFockDH<'a> {
    /// Create the composite from the contribution objects.
    ///
    /// # Parameters
    ///
    /// - `gfock_list` : the generalized-Fock contribution objects (density-only parts of the
    ///   final-energy functional plus the PT2 correlation contribution).
    /// - `mo_coeff` : shape `[nao, nmo]`. Molecular orbital coefficients.
    /// - `mo_occ` : shape `[nmo]`. Occupation numbers.
    /// - `mo_energy` : shape `[nmo]`. Molecular orbital energies.
    pub fn new(
        gfock_list: Vec<Box<dyn RGFockAPI + 'a>>,
        mo_coeff: Tsr,
        mo_occ: Tsr,
        mo_energy: Tsr,
    ) -> Self {
        let nmo = mo_occ.shape()[0];
        let device = mo_coeff.device().clone();
        Self {
            gfock_list,
            mo_coeff,
            mo_occ,
            mo_energy,
            gfock: rt::zeros(([nmo, nmo].f(), &device)),
            gfock_flags: BitFlags::empty(),
            result: HashMap::new(),
            timing: Vec::new(),
        }
    }

    /// Prepare the response object (must be called before the Z-vector solve or any response
    /// contraction). This also stores the orbital state in the response object for its inherent
    /// CP-SCF machinery, using the orbitals of this driver.
    ///
    /// # Parameters
    ///
    /// - `resp` : response objects of the SCF-iteration functional, mutably.
    pub fn make_response_preparation(&mut self, resp: &mut RRespSCF) {
        let t0 = std::time::Instant::now();
        resp.make_cpscf_preparation(self.mo_coeff.view(), self.mo_occ.view(), self.mo_energy.view());
        self.timing.push(("in RGFockDH, make_response_preparation".to_string(), t0.elapsed().as_secs_f64()));
    }

    /// Solve the Z-vector equation for the given Lagrangian $L_{ai}$ through the response object,
    /// caching the result under `z_vector`. Shape `[nmo, nocc]`; see [`solve_z_vector`].
    ///
    /// # Parameters
    ///
    /// - `lag_vo` : shape `[nvir, nocc]`. The Lagrangian $L_{ai}$.
    /// - `resp` : response objects of the SCF-iteration functional, mutably.
    pub fn solve_z_vector(&mut self, lag_vo: TsrView, resp: &mut RRespSCF) -> Tsr {
        let t0 = std::time::Instant::now();
        let z = solve_z_vector(lag_vo, resp);
        self.result.insert("z_vector".to_string(), z.to_owned());
        self.timing.push(("in RGFockDH, solve_z_vector".to_string(), t0.elapsed().as_secs_f64()));
        self.result["z_vector"].to_owned()
    }

    /// Relaxed (response) 1-RDM of the DH method in MO basis, shape `[nmo, nmo]`.
    ///
    /// This is the unrelaxed rdm1 (of the PT2 contribution; the density-only parts contribute
    /// none) with its vir-occupied block replaced by the Z-vector solved from the combined
    /// Lagrangian of all contributions.
    ///
    /// # Parameters
    ///
    /// - `resp` : response objects of the SCF-iteration functional, mutably.
    pub fn make_rdm1_resp(&mut self, resp: &mut RRespSCF) -> Tsr {
        if !self.result.contains_key("rdm1_resp") {
            let t0 = std::time::Instant::now();
            let lag = self.make_lagrangian(Some(resp));
            let z = self.solve_z_vector(lag.view(), resp);
            let mut rdm1_resp = self.make_rdm1();
            let nocc = self.nocc();
            let nmo = self.nmo();
            let so = rt::slice!(0, nocc);
            let sv = rt::slice!(nocc, nmo);
            rdm1_resp.i_mut((sv, so)).assign(&z.i((sv, so)));
            self.result.insert("rdm1_resp".to_string(), rdm1_resp);
            self.timing.push(("in RGFockDH, make_rdm1_resp".to_string(), t0.elapsed().as_secs_f64()));
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

impl AnalDrvBaseAPI for RGFockDH<'_> {}

impl RGFockAPI for RGFockDH<'_> {
    /// Generalized Fock of the DH method: for every requested part not yet evaluated, the part's
    /// block is computed as the sum of the contribution objects' blocks of that part (each
    /// contribution is called with the single-part flags, and the PT2 element rejects its
    /// unimplemented OO/VV parts), accumulated into `self.gfock`, and the matrix of exactly the
    /// requested parts is re-constructed from the accumulated one. Parts not requested are left
    /// zero in every contribution, even where computable (e.g. OO in RI-JK/DFT/hcore).
    ///
    /// The response object argument is passed through to the contribution objects; the PT2
    /// element requires it for its VO part.
    fn make_gfock<'r>(&mut self, mut resp: Option<&mut (dyn RRespAPI + 'r)>, flags: BitFlags<GFockFlags>) -> Tsr {
        let t0 = std::time::Instant::now();
        let nocc = self.nocc();
        let nmo = self.nmo();
        let so = rt::slice!(0, nocc);
        let sv = rt::slice!(nocc, nmo);
        let device = self.mo_coeff.device().clone();

        // evaluate the requested parts that are not yet in `self.gfock`
        for (flag, rows, cols) in [
            (GFockFlags::OO, so, so),
            (GFockFlags::OV, rt::slice!(0, nocc), rt::slice!(nocc, nmo)),
            (GFockFlags::VO, sv, rt::slice!(0, nocc)),
            (GFockFlags::VV, rt::slice!(nocc, nmo), rt::slice!(nocc, nmo)),
        ] {
            if !flags.contains(flag) || self.gfock_flags.contains(flag) {
                continue;
            }
            let mut block: Option<Tsr> = None;
            for gfock_obj in self.gfock_list.iter_mut() {
                let obj_gfock = gfock_obj.make_gfock(resp.as_deref_mut(), flag.into());
                let obj_block = obj_gfock.i((rows, cols)).into_contig(ColMajor);
                block = Some(match block {
                    Some(block) => block + obj_block,
                    None => obj_block,
                });
            }
            let block = block.expect("RGFockDH must hold at least one contribution object.");
            *&mut self.gfock.i_mut((rows, cols)) += &block;
            self.gfock_flags.insert(flag);
        }

        // re-construct the matrix of the requested parts, not the accumulated `self.gfock`
        let mut gfock: Tsr = rt::zeros(([nmo, nmo].f(), &device));
        for (flag, rows, cols) in [
            (GFockFlags::OO, so, so),
            (GFockFlags::OV, rt::slice!(0, nocc), rt::slice!(nocc, nmo)),
            (GFockFlags::VO, sv, rt::slice!(0, nocc)),
            (GFockFlags::VV, rt::slice!(nocc, nmo), rt::slice!(nocc, nmo)),
        ] {
            if flags.contains(flag) {
                gfock.i_mut((rows, cols)).assign(&self.gfock.i((rows, cols)));
            }
        }
        self.timing.push(("in RGFockDH, make_gfock".to_string(), t0.elapsed().as_secs_f64()));
        gfock
    }

    /// Unrelaxed rdm1 of the DH method: the sum of the contribution objects' rdm1s (the PT2
    /// correlation rdm1; the density-only parts contribute zero). Cached on first call.
    fn make_rdm1(&mut self) -> Tsr {
        if !self.result.contains_key("rdm1") {
            let mut rdm1: Option<Tsr> = None;
            for gfock_obj in self.gfock_list.iter_mut() {
                let rdm1_obj = gfock_obj.make_rdm1();
                rdm1 = Some(match rdm1 {
                    Some(rdm1) => rdm1 + rdm1_obj,
                    None => rdm1_obj,
                });
            }
            let rdm1 = rdm1.expect("RGFockDH must hold at least one contribution object.");
            self.result.insert("rdm1".to_string(), rdm1);
        }
        self.result["rdm1"].to_owned()
    }

    /// Lagrangian of the DH method: the sum of the contribution objects' Lagrangians (the PT2 one
    /// $W^\texttt{3} + W^\texttt{4} + A_{ai, pq} D_{pq}^{\mathrm{RDM}}$, the density-only ones
    /// $4 C_v^T V C_o$ each). Shape `[nvir, nocc]`. Cached on first call.
    ///
    /// The response object argument is passed through to the contribution objects; the PT2 element
    /// requires it.
    fn make_lagrangian<'r>(&mut self, mut resp: Option<&mut (dyn RRespAPI + 'r)>) -> Tsr {
        if !self.result.contains_key("lagrangian") {
            let t0 = std::time::Instant::now();
            let mut lagrangian: Option<Tsr> = None;
            for gfock_obj in self.gfock_list.iter_mut() {
                let lag_obj = gfock_obj.make_lagrangian(resp.as_deref_mut());
                lagrangian = Some(match lagrangian {
                    Some(lagrangian) => lagrangian + lag_obj,
                    None => lag_obj,
                });
            }
            let lagrangian =
                lagrangian.expect("RGFockDH must hold at least one contribution object.");
            self.result.insert("lagrangian".to_string(), lagrangian);
            self.timing
                .push(("in RGFockDH, make_lagrangian".to_string(), t0.elapsed().as_secs_f64()));
        }
        self.result["lagrangian"].to_owned()
    }
}

/// Build the generalized-Fock (and Lagrangian, Z-vector) machinery of a restricted double-hybrid
/// method from converged SCF data.
///
/// The returned [`RGFockDH`] holds the contribution objects of the final-energy functional (core
/// Hamiltonian, RI-JK with `dfa_hybrid_pos`, DFT XC of `dfa_compnt_pos` on the common SCF grid)
/// and the RI-PT2 correlation driver with the `dfa_paramr_adv` spin factors. The response object
/// of the SCF-iteration functional (e.g. from
/// [`rscf_resp_interface`](super::rresp_interface::rscf_resp_interface)) is not held by the
/// returned driver: pass it as a mutable argument to the driver methods that need one, after its
/// preparation with the same orbitals ([`RGFockDH::make_response_preparation`]).
///
/// The CP-SCF solver settings (including the Z-vector level shift) are those of the response
/// object supplied by the caller, captured when it was built.
///
/// The RI-PT2 working precision follows the `[ri_pt2] fp_mode` control keyword (FP32 by default);
/// all outputs of the PT2 contribution are f64 regardless.
///
/// # Parameters
///
/// - `scf_data` : converged SCF data. Must carry the cholesky decomposed ERI (`rimatr`); the DFT
///   part additionally requires the SCF grids (`scf_data.grids`) to be present — note that
///   [`xdh_calculations`](crate::ri_pt2::xdh_calculations) frees the grids, so this interface
///   must be called before it, or the grids must be regenerated in between.
pub fn rgfock_dh_interface<'a>(scf_data: &'a SCF) -> RGFockDH<'a> {
    let device = DeviceBLAS::default();

    // --- basic preparation and checks --- //

    let mo_coeff = scf_data.eigenvectors[0].to_rstsr(&device);
    let mo_occ = (&scf_data.occupation[0]).to_rstsr(&device);
    let mo_energy = (&scf_data.eigenvalues[0]).to_rstsr(&device);

    assert_eq!(
        scf_data.mol.start_mo, 0,
        "RGFockDH does not support frozen-core double hybrids yet (mol.start_mo != 0)."
    );
    match scf_data.mol.xc_data.dfa_family_pos {
        Some(DFAFamily::PT2) => {}
        _ => panic!("RGFockDH currently requires the PT2 post-SCF family (ri_pt2 correlation)."),
    }
    if scf_data.mol.xc_data.omega().is_some() {
        panic!(
            "RGFockDH does not support range-separated double hybrids yet: the final-energy \
             functional's range separation is not stored in xc_data (only the SCF functional's \
             dfa_rsh_scf), so its RI-JK exchange decomposition cannot be determined."
        );
    }
    let [c_os, c_ss] = scf_data
        .mol
        .xc_data
        .dfa_paramr_adv
        .clone()
        .expect("RGFockDH requires the PT2 spin factors (dfa_paramr_adv).")
        .try_into()
        .expect("dfa_paramr_adv must have exactly two entries (c_os, c_ss).");

    let nocc = mo_occ.view().greater(0).sum();

    let mut gfock_list: Vec<Box<dyn RGFockAPI + 'a>> = Vec::new();

    // --- core Hamiltonian --- //

    {
        let hcore = scf_data.h_core.to_matrixfull().unwrap().to_rstsr(&device);
        gfock_list.push(Box::new(RGFockHcore::new(hcore, mo_coeff.to_owned(), mo_occ.to_owned())));
    }

    // --- RI-JK --- //

    let (rimatr, _, _) = scf_data.rimatr.as_ref().expect(
        "This implementation requires cholesky decomposed ERI (or rimatr) to be available and stored in memory.",
    );
    {
        let (factor_j, factor_k) = dh_jk_factors(scf_data);
        gfock_list.push(Box::new(RGFockRIJK::new_with_cderi(
            factor_j,
            factor_k,
            rimatr.to_rstsr_view(&device).into_cow(),
            mo_coeff.to_owned(),
            mo_occ.to_owned(),
        )));
    }

    // --- DFT (final-energy functional, common SCF grid) --- //

    if let Some(xc_func_list) = dh_xc_func_list(scf_data) {
        let mol = get_cint_mol(&scf_data.mol);
        let grids = scf_data.grids.as_ref().expect(
            "The DH generalized Fock requires the SCF grids; note xdh_calculations frees them, so build this interface before it or regenerate the grids in between.",
        );
        let ni = NIMatmul::new(&mol, &grids.coordinates, &grids.weights, &grids.atm_idx, &grids.quadrature_weights);
        gfock_list.push(Box::new(RGFockKSNIMatmul::new(
            xc_func_list,
            ni,
            mo_coeff.to_owned(),
            mo_occ.to_owned(),
        )));
    }

    // --- RI-PT2 (correlation contribution; an ordinary element of the list) --- //

    // the RI-PT2 working precision follows the `[ri_pt2] fp_mode` control keyword (default FP32;
    // the pre-transformed f32 `cderi_vox` is not supplied here, so the kernel transforms in f64
    // and casts on the fly); all outputs (`gfock_part`, `rdm1_corr`, `e_corr`) are f64 regardless
    match scf_data.mol.ctrl.ri_pt2.fp_mode {
        PT2FPMode::FP64 => gfock_list.push(Box::new(RGFockPT2::<f64>::new(
            mo_coeff.to_owned(),
            mo_occ.to_owned(),
            mo_energy.to_owned(),
            rimatr.to_rstsr_view(&device).into_cow(),
            None,
            vec![0, nocc],
            c_os,
            c_ss,
        ))),
        PT2FPMode::FP32 => gfock_list.push(Box::new(RGFockPT2::<f32>::new(
            mo_coeff.to_owned(),
            mo_occ.to_owned(),
            mo_energy.to_owned(),
            rimatr.to_rstsr_view(&device).into_cow(),
            None,
            vec![0, nocc],
            c_os,
            c_ss,
        ))),
    }

    RGFockDH::new(gfock_list, mo_coeff, mo_occ, mo_energy)
}
