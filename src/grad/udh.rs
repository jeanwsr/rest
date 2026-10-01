//! Unrestricted doubly-hybrid (pure-MP2 family) analytic gradient; the unrestricted twin of
//! [`crate::grad::rdh`], following pyscf-forge `dh/grad/udfdh.py`.
//!
//! The SCF-iteration functional's gradient is provided by the existing `"SCF"` entry of
//! [`crate::main_driver::eval_force`]; this entry covers the D_r-carrying (correlation)
//! remainder, per spin $\sigma$ (`de_*` keys mirror the restricted module):
//!
//! - `de_h`: hcore-derivative term $\Sigma_\sigma \langle H_1(A), D_r^\sigma\rangle$;
//! - `de_s1`: overlap-Pulay term $\Sigma_\sigma \langle W^\sigma, S_1(A)\rangle$, with $W$ the
//!   unrestricted `W_I + W_II + W_III` assembly (forge `prepare_lagrangian`/`_get_gradient_pt2`);
//! - `de_jk`: `t1 + t2 + t4` of forge `get_gradient_jk`, restricted to the D_r-carrying weights:
//!   the RI-J bilinears `t1`/`t2` run over all four spin pairs (the total SCF density meets the
//!   per-spin relaxed densities), while the exchange `t4` is per spin with
//!   `C1^s = cx C^s D_r^{S,s}` (`cx = dfa_hybrid_scf`);
//! - `de_rint`: $\Sigma_\sigma \langle G^\sigma, Y_1^\sigma(A)\rangle$ with the unrestricted
//!   coefficient `1` (restricted: `4`), `G` carrying forge's `4`/`2` spin-block normalization.
//!
//! The relaxed density is solved per spin through the coupled CP-SCF of
//! [`uscf_resp_interface`](crate::analdrv::response::uresp_interface::uscf_resp_interface) (no
//! $\alpha\beta$ exchange in the kernel), driven by the DH-combined Lagrangian
//! $L^\sigma = \mathscr{F}^{\sigma}_{vo} - \mathscr{F}^{\sigma}_{ov\top} + A^\sigma(D^-)_{vo}
//! + 2 C_v^{\sigma\top} F_n^\sigma(D^\sigma) C_o^\sigma$.
//!
//! The final-functional (XYG3-family) terms mirror the restricted module
//! [`crate::grad::rdh`], per spin $\sigma$ and all absent for the pure-MP2 family:
//! `de_k_dh`: exchange-derivative difference upon the SCF density (`factor_k =
//! dfa_hybrid_pos - dfa_hybrid_scf`, with its aux-basis-response part); `de_xc_n`: XC skeleton
//! difference `xc_n - xc` upon D_scf; `de_resp`: $\langle D_{r,\mathrm{ao}}^\sigma, F_1
//! [\mathrm{xc}]\rangle$ through the UKS Hessian `vmat_deriv1` kernels (coefficient 1 per spin);
//! `de_ovlp_dh`: occupied-Pulay difference `Tr(S1, W_n^\sigma - dme0^\sigma)` with
//! $W_n^\sigma = C_o^\sigma F_n^\sigma(D_{scf})_{oo} C_o^{\sigma\top}$ (no occupancy factor,
//! against the restricted 2).

use crate::analdrv::response::rgfock_interface::dh_jk_factors;
use crate::analdrv::response::trait_uresp::URespAPI;
use crate::analdrv::response::uresp_interface::{
    dh_xc_func_list_uks, scf_xc_func_list_uks, uscf_resp_interface, URespSCF,
};
use crate::dft::num_int::XCData;
use crate::dft::numint_matmul::nimatmul::{regroup_grids_by_atom, NIMatmul};
use crate::dft::numint_matmul::resp_uks::URespKSNIMatmul;
use crate::dft::xc_deriv::XCType;
use crate::dft::xceff::prelude::{determine_den_type_from_list, XCDenType};
use crate::grad::rhf::{
    generator_deriv_hcore, get_dme0, get_grad_dao_ovlp, pack_triu_tilde, RIHFGradientFlagsBuilder,
};
use crate::grad::traits::GradAPI;
use crate::grad::uhf::RIUHFGradient;
use crate::mpi_io::MPIOperator;
use crate::ri_jk::resp_u::URespRIJK;
use crate::ri_jk::util;
use crate::ri_jk::util::get_dm0_restricted;
use crate::ri_jk::{get_j2c_decomp, J2CDecompose, J2C_THRESH};
use crate::ri_pt2::pure_pt2_u_elecderiv::{
    get_rupt2_elec_deriv_incore, UPT2ElecDerivIncoreArg, UPT2ElecDerivIncoreInp,
};
use crate::ri_pt2::{occ_batch_index, occ_batch_step};
use crate::scf_io::{SCF, SCFType};
use crate::utilities::memory_batch::{
    blocksize_partition, calc_batch_size, detect_used_memory_mb, handle_memory_exceed,
    xc_grad_block_mb,
};
use crate::utilities::rstsr_util::{RestTensorToRstsrTsrAPI, RestTensorToRstsrViewAPI, Tsr, TsrView};
use rest_libcint::prelude::*;
use rstsr::prelude::*;
use std::collections::HashMap;
use std::ops::Range;
use tensors::MatrixFull;

/// Analytic gradient of the PT2-correlation (RI-MP2) energy, unrestricted case.
pub struct UDHGradient<'a> {
    scf_data: &'a SCF,
    mpi_operator: &'a Option<MPIOperator>,
    /// Gradient parts, keyed by `de_h`/`de_s1`/`de_jk`/`de_rint`/`de`, each `[3, natm]`.
    pub result: HashMap<String, MatrixFull<f64>>,
    /// PT2 correlation energy (side product of the electronic-derivative evaluation).
    pub energy: f64,
    /// Total relaxed density (SCF density plus the relaxed correlation density) prepared for the
    /// fchk density dump, in fchk section order: `[Total MP2 Density, Spin MP2 Density]`
    /// (alpha + beta, alpha - beta). `Some` only when `[ctrl] outputs` requests `fchk`; the same
    /// density the multipole task dumps.
    pub rdm1_dump: Option<Vec<MatrixFull<f64>>>,
}

impl<'a> UDHGradient<'a> {
    pub fn new(scf_data: &'a SCF, mpi_operator: &'a Option<MPIOperator>) -> Self {
        Self { scf_data, mpi_operator, result: HashMap::new(), energy: 0.0, rdm1_dump: None }
    }

    pub fn calc(&mut self) -> &mut Self {
        let scf_data = self.scf_data;
        let mol_obj = &scf_data.mol;
        let device = DeviceBLAS::default();
        assert!(mol_obj.ctrl.spin_polarization, "UDHGradient supports only unrestricted (spin-polarization = true) references.");
        assert!(matches!(scf_data.scftype, SCFType::UHF), "UDHGradient supports only UHF references.");
        assert!(mol_obj.start_mo == 0, "UDHGradient does not support frozen core (start_mo).");
        assert!(self.mpi_operator.is_none(), "UDHGradient is not MPI-parallelized yet.");
        assert!(
            scf_data.mol.xc_data.omega().is_none(),
            "RSH doubly-hybrid gradients are not supported."
        );
        assert!(
            scf_data.mol.geom.ghost_pc_chrg.is_empty() && !scf_data.mol.ctrl.solvent_enabled,
            "The DH analytic gradient does not support QMMM point charges or solvent models yet."
        );
        assert!(
            !matches!(scf_data.mol.ctrl.j2c_decomp.uplo, Lower),
            "UDHGradient does not support lower-triangular Cholesky factors (developer option)."
        );
        // this driver reads `rimatr` through the full-space pair tables; a storage-level pruned
        // tensor stores its retained rows only, so refuse it rather than contract wrong rows
        crate::ri_jk::require_unpruned_rimatr(&scf_data.rimatr_pair_map, "the DH analytic gradient (UDHGradient)");
        let rimatr = scf_data.rimatr.as_ref().expect(
            "Decomposed ERI (rimatr) not found; the DH analytic gradient requires the streaming/new-driver RI-PT2 engine which keeps it alive.",
        );
        let ederi_utp = rt::asarray((&rimatr.0.data, rimatr.0.size, &device));
        let mol = util::get_cint_mol(mol_obj);
        let aux = util::get_cint_aux(mol_obj);
        let natm = mol_obj.geom.elem.len();
        let naux = aux.nao();

        let [c_os, c_ss]: [f64; 2] = mol_obj
            .xc_data
            .dfa_paramr_adv
            .clone()
            .expect("Doubly-hybrid gradient requires the PT2 spin factors (dfa_paramr_adv).")
            .try_into()
            .expect("dfa_paramr_adv must have exactly two entries.");

        // gates of the final-functional (doubly-hybrid) terms; the pure-MP2 family has none, and
        // a bDH whose final functional equals the SCF one is covered by the "SCF" entry entirely
        let xc_data = &mol_obj.xc_data;
        let delta_hyb = xc_data.dfa_hybrid_pos.unwrap_or(xc_data.dfa_hybrid_scf) - xc_data.dfa_hybrid_scf;
        let xc_func_pos = dh_xc_func_list_uks(scf_data);
        let xc_n_is_scf = match (&xc_data.dfa_compnt_pos, &xc_data.dfa_paramr_pos) {
            (Some(code), Some(param)) => {
                code.iter().eq(xc_data.dfa_compnt_scf.iter())
                    && param.len() == xc_data.dfa_paramr_scf.len()
                    && param.iter().zip(xc_data.dfa_paramr_scf.iter()).all(|(a, b)| (a - b).abs() < 1e-12)
            },
            _ => true,
        };
        let add_delta_k = delta_hyb.abs() > 1.0e-10;
        let add_xc_n = xc_func_pos.is_some() && !xc_n_is_scf;
        let has_final_functional = add_delta_k || add_xc_n;
        let add_resp = has_scf_xc(scf_data);

        // atom-grouped copy of the SCF grid, shared by the final-Fock and response terms
        let grids_regrouped = if has_final_functional || add_resp {
            let grids = scf_data.grids.as_ref().expect(
                "Doubly-hybrid gradient requires the DFT grids (`eval_force` regenerates them; `xdh_calculations` frees them).",
            );
            Some(regroup_grids_by_atom(
                grids.coordinates.clone(),
                grids.weights.clone(),
                grids.atm_idx.clone(),
                grids.quadrature_weights.clone(),
                natm,
            ))
        } else {
            None
        };

        // ── per-spin orbital data ── //
        let mo_coeff: [Tsr; 2] =
            [(&scf_data.eigenvectors[0]).to_rstsr(&device), (&scf_data.eigenvectors[1]).to_rstsr(&device)];
        let mo_occ: [Tsr; 2] =
            [(&scf_data.occupation[0]).to_rstsr(&device), (&scf_data.occupation[1]).to_rstsr(&device)];
        let mo_energy: [Tsr; 2] =
            [(&scf_data.eigenvalues[0]).to_rstsr(&device), (&scf_data.eigenvalues[1]).to_rstsr(&device)];
        let nao = mo_coeff[0].shape()[0];
        let nmo = mo_coeff[0].shape()[1];
        assert_eq!(mo_coeff[1].shape()[1], nmo, "UDHGradient expects equal MO counts per spin.");
        let nocc = [
            mo_occ[0].view().greater(0).sum(),
            mo_occ[1].view().greater(0).sum(),
        ];
        let nvir = [nmo - nocc[0], nmo - nocc[1]];
        assert!(nocc[0] > 0 && nocc[1] > 0, "UDHGradient expects both spin channels occupied.");
        let nao_tp = nao * (nao + 1) / 2;
        let so: [Range<usize>; 2] = [0..nocc[0], 0..nocc[1]];
        let sv: [Range<usize>; 2] = [nocc[0]..nmo, nocc[1]..nmo];
        let c0: [Tsr; 2] = [
            mo_coeff[0].i((.., so[0].clone())).into_contig(ColMajor),
            mo_coeff[1].i((.., so[1].clone())).into_contig(ColMajor),
        ];
        let cv: [Tsr; 2] = [
            mo_coeff[0].i((.., sv[0].clone())).into_contig(ColMajor),
            mo_coeff[1].i((.., sv[1].clone())).into_contig(ColMajor),
        ];
        let eo: [Tsr; 2] = [
            mo_energy[0].i(so[0].clone()).into_contig(ColMajor),
            mo_energy[1].i(so[1].clone()).into_contig(ColMajor),
        ];
        let ev: [Tsr; 2] = [
            mo_energy[0].i(sv[0].clone()).into_contig(ColMajor),
            mo_energy[1].i(sv[1].clone()).into_contig(ColMajor),
        ];
        let d_hf: [Tsr; 2] = [
            (&scf_data.density_matrix[0]).to_rstsr(&device),
            (&scf_data.density_matrix[1]).to_rstsr(&device),
        ];

        // F_n^s(D_scf) = h + J[D_a + D_b] - hyb_pos K^s[D^s] + v_xc[xc_n]^s; it enters the
        // DH-combined Lagrangian (the formal-SCF/Brillouin term) and the occupied-Pulay difference
        let fock_n: Option<[Tsr; 2]> = if has_final_functional {
            let (coords, weights, atm_idx_grids, quadrature_weights) = grids_regrouped.as_ref().unwrap();
            let (factor_j_dh, factor_k_dh) = dh_jk_factors(scf_data);
            let cderi_dh = rimatr.0.to_rstsr_view(&device).into_cow();
            let h_core = scf_data.h_core.to_matrixfull().unwrap().to_rstsr(&device);
            let mut resp_rijk = URespRIJK::new_with_cderi(factor_j_dh, factor_k_dh, cderi_dh);
            let f_jk = resp_rijk.get_fock_rdm(&[d_hf[0].view(), d_hf[1].view()], true);
            let mut fock_n = [(&h_core + &f_jk[0]).into_contig(ColMajor), (&h_core + &f_jk[1]).into_contig(ColMajor)];
            if let Some(xc_func_list) = dh_xc_func_list_uks(scf_data) {
                let ni_dh = NIMatmul::new(&mol, coords, weights, atm_idx_grids, quadrature_weights);
                let mut resp_ks = URespKSNIMatmul::new(xc_func_list, ni_dh, mol_obj.ctrl.print_level > 2);
                let f_ks = resp_ks.get_fock_rdm(&[d_hf[0].view(), d_hf[1].view()], true);
                *&mut fock_n[0] += &f_ks[0];
                *&mut fock_n[1] += &f_ks[1];
            }
            Some(fock_n)
        } else {
            None
        };

        // ── response objects, PT2 electronic derivative, relaxed correlation density ── //
        let config = mol_obj.ctrl.analdrv.clone().unwrap_or_default();
        let mut resp_objs: URespSCF = uscf_resp_interface(scf_data, &config);
        // the in-core electronic-derivative evaluation streams its amplitudes (storage bounded
        // by the `nvir * nocc * naux` class): the same-spin blocks window their outer
        // occupied index, the αβ block its α virtual index (full occupied range per window, cf.
        // the kernel docs). Size the windows so the per-window transients stay within a few
        // times the `nvir * nocc * naux` class and inside `max_memory`.
        let mem_avail = mol_obj.ctrl.max_memory.map(|m| m - detect_used_memory_mb("proc"));
        let abort = mol_obj.ctrl.abort_on_mem_exceed;
        let nocc_ = [c0[0].shape()[1], c0[1].shape()[1]];
        let nvir_ = [cv[0].shape()[1], cv[1].shape()[1]];
        let headroom = 4.0;
        let mut step_ss = [1usize; 2];
        for s in 0..2 {
            if nocc_[s] > 0 && nvir_[s] > 0 {
                let elems_per_step = 2 * nvir_[s] * nvir_[s] * nocc_[s] + nvir_[s] * naux;
                step_ss[s] = occ_batch_step(nocc_[s], elems_per_step, nvir_[s] * nocc_[s] * naux, mem_avail, headroom);
            }
        }
        // the αβ block holds its amplitude window plus ~3 layout copies of it, in each of its
        // two passes (α-side respectively β-side rdm1 Grams)
        let mut step_ab = [nvir_[0].max(1), nvir_[1].max(1)];
        let mut ab_elems_per_step = [0usize; 2];
        if nocc_[0] > 0 && nocc_[1] > 0 && nvir_[0] > 0 && nvir_[1] > 0 {
            ab_elems_per_step[0] = 4 * nocc_[0] * nvir_[1] * nocc_[1];
            step_ab[0] = occ_batch_step(nvir_[0], ab_elems_per_step[0], nvir_[0] * nocc_[0] * naux, mem_avail, headroom);
            ab_elems_per_step[1] = 4 * nvir_[0] * nocc_[0] * nocc_[1];
            step_ab[1] = occ_batch_step(nvir_[1], ab_elems_per_step[1], nvir_[1] * nocc_[1] * naux, mem_avail, headroom);
        }
        let index_occ_outer = [occ_batch_index(nocc_[0], step_ss[0]), occ_batch_index(nocc_[1], step_ss[1])];
        let index_vir_outer = [occ_batch_index(nvir_[0], step_ab[0]), occ_batch_index(nvir_[1], step_ab[1])];
        if mol_obj.ctrl.print_level > 1 {
            println!(
                "DH gradient amplitude windows: σσ occ steps {:?}, αβ virtual steps {:?} (nocc {nocc_:?}, nvir {nvir_:?}, naux {naux}).",
                step_ss, step_ab
            );
        }
        // guard: even the smallest possible window must fit when batching is floored at one
        let peak_elems = [
            if nocc_[0] > 0 && nvir_[0] > 0 { step_ss[0] * (2 * nvir_[0] * nvir_[0] * nocc_[0] + nvir_[0] * naux) } else { 0 },
            if nocc_[1] > 0 && nvir_[1] > 0 { step_ss[1] * (2 * nvir_[1] * nvir_[1] * nocc_[1] + nvir_[1] * naux) } else { 0 },
            step_ab[0] * ab_elems_per_step[0],
            step_ab[1] * ab_elems_per_step[1],
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        handle_memory_exceed(peak_elems as f64 * 8.0 / 1048576.0, mem_avail, abort);
        let j3c = rimatr.0.to_rstsr_view(&device);
        let out = get_rupt2_elec_deriv_incore(
            &UPT2ElecDerivIncoreInp {
                cderi: j3c.view(),
                cderi_vox: [None, None],
                occ_coeff: [c0[0].view(), c0[1].view()],
                vir_coeff: [cv[0].view(), cv[1].view()],
                occ_energy: [eo[0].view(), eo[1].view()],
                vir_energy: [ev[0].view(), ev[1].view()],
                index_occ_outer_vec: index_occ_outer,
                index_vir_outer_vec: index_vir_outer,
            },
            &UPT2ElecDerivIncoreArg { c_os, c_ss },
        );
        let rdm1 = out.rdm1_corr;
        let gfock_part = out.gfock_part;
        let g_vix = out.g_vix;
        self.energy = out.e_corr;

        resp_objs.make_cpscf_preparation(
            &[mo_coeff[0].view(), mo_coeff[1].view()],
            &[mo_occ[0].view(), mo_occ[1].view()],
            &[mo_energy[0].view(), mo_energy[1].view()],
        );
        // SCF response upon the unrelaxed correlation density (the Lagrangian's `Ax0` term), and
        // upon the *relaxed* one (the W_III term) below; both are rdm-form A-contractions
        let d_corr_ao: [Tsr; 2] = [
            (mo_coeff[0].view() % rdm1[0].view() % mo_coeff[0].view().t()).into_contig(ColMajor),
            (mo_coeff[1].view() % rdm1[1].view() % mo_coeff[1].view().t()).into_contig(ColMajor),
        ];
        let axd = {
            let resp_ao = resp_objs.get_response_rdm(&[d_corr_ao[0].view(), d_corr_ao[1].view()], true);
            [
                (mo_coeff[0].view().t() % resp_ao[0].view() % mo_coeff[0].view()).into_contig(ColMajor),
                (mo_coeff[1].view().t() % resp_ao[1].view() % mo_coeff[1].view()).into_contig(ColMajor),
            ]
        };

        // L^s = gfock[vo] - gfock[ov]^T + axd[vo]; Z-vector through the coupled CP-SCF
        let rdm1_resp: [Tsr; 2] = {
            let e_ai = [
                resp_objs.cpscf_state().e_ai_shift[0].to_owned(),
                resp_objs.cpscf_state().e_ai_shift[1].to_owned(),
            ];
            let mut rhs = [
                rt::zeros(([nmo, nocc[0]].f(), &device)),
                rt::zeros(([nmo, nocc[1]].f(), &device)),
            ];
            for s in 0..2 {
                let mut lag_vo = (&gfock_part[s].i((sv[s].clone(), so[s].clone()))
                    - &gfock_part[s].i((so[s].clone(), sv[s].clone())).t()
                    + &axd[s].i((sv[s].clone(), so[s].clone())))
                    .into_contig(ColMajor);
                // the DH-combined Lagrangian carries the formal-SCF (Brillouin-violation) term
                // of the final functional: `2 Cv^T F_n^s(D^s) Co` per spin
                if let Some(fock_n) = &fock_n {
                    *&mut lag_vo += &(2.0 * (cv[s].view().t() % fock_n[s].view() % c0[s].view()));
                }
                rhs[s].i_mut(sv[s].clone()).assign(&(-lag_vo / &e_ai[s]));
            }
            let z = resp_objs.solve_dimless_cpscf(&[rhs[0].view(), rhs[1].view()]);
            [
                {
                    let mut r = rdm1[0].to_owned();
                    r.i_mut((sv[0].clone(), so[0].clone())).assign(&z[0].i((sv[0].clone(), so[0].clone())));
                    r
                },
                {
                    let mut r = rdm1[1].to_owned();
                    r.i_mut((sv[1].clone(), so[1].clone())).assign(&z[1].i((sv[1].clone(), so[1].clone())));
                    r
                },
            ]
        };

        // symmetrized half of the relaxed density, and the D_r-carrying AO/MO objects
        let d_r_s: [Tsr; 2] = [
            ((&rdm1_resp[0] + &rdm1_resp[0].t()).mapv(|x| 0.5 * x)).into_contig(ColMajor),
            ((&rdm1_resp[1] + &rdm1_resp[1].t()).mapv(|x| 0.5 * x)).into_contig(ColMajor),
        ];
        // the exchange (K) part of the SCF functional linearized upon the relaxed density carries
        // the SCF hybrid coefficient (forge `C1^s = cx C D_r^{S,s}`; unity for the MP2/HF reference)
        let cx_scf = mol_obj.xc_data.dfa_hybrid_scf;
        let c1_dr: [Tsr; 2] = [
            (mo_coeff[0].view() % d_r_s[0].view()).mapv(|x| cx_scf * x).into_contig(ColMajor),
            (mo_coeff[1].view() % d_r_s[1].view()).mapv(|x| cx_scf * x).into_contig(ColMajor),
        ];
        let d_r_ao: [Tsr; 2] = [
            (mo_coeff[0].view() % d_r_s[0].view() % mo_coeff[0].view().t()).into_contig(ColMajor),
            (mo_coeff[1].view() % d_r_s[1].view() % mo_coeff[1].view().t()).into_contig(ColMajor),
        ];
        // per-spin total relaxed densities (SCF + relaxed correlation) and their fchk sections
        // (total and spin) for the density dump of force jobs; only materialized when the fchk
        // output is requested (`SCF::dh_rdm1_resp`)
        if mol_obj.ctrl.outputs.iter().any(|output| output.eq("fchk")) {
            let dm_alpha = &d_hf[0] + &d_r_ao[0];
            let dm_beta = &d_hf[1] + &d_r_ao[1];
            self.rdm1_dump = Some(vec![
                to_matrix_full(&dm_alpha + &dm_beta),
                to_matrix_full(&dm_alpha - &dm_beta),
            ]);
        }
        let axd_dr = {
            let resp_ao = resp_objs.get_response_rdm(&[d_r_ao[0].view(), d_r_ao[1].view()], true);
            [
                (mo_coeff[0].view().t() % resp_ao[0].view() % mo_coeff[0].view()).into_contig(ColMajor),
                (mo_coeff[1].view().t() % resp_ao[1].view() % mo_coeff[1].view()).into_contig(ColMajor),
            ]
        };

        // ── W = W_I + W_II + W_III (MO), per spin, from the `gfock_part` blocks ── //
        let mut w_ao: [Tsr; 2] = [rt::zeros_like(&d_hf[0]), rt::zeros_like(&d_hf[1])];
        for s in 0..2 {
            let eps2: Tsr = 2.0 * mo_energy[s].view();
            let mut w_mo: Tsr = rt::zeros(([nmo, nmo], &device));
            {
                let oo = (&gfock_part[s].i((so[s].clone(), so[s].clone()))
                    - &eps2.i((None, so[s].clone())) * rdm1[s].i((so[s].clone(), so[s].clone())))
                    .mapv(|x: f64| -0.5 * x);
                let vv = (&gfock_part[s].i((sv[s].clone(), sv[s].clone()))
                    - &eps2.i((None, sv[s].clone())) * rdm1[s].i((sv[s].clone(), sv[s].clone())))
                    .mapv(|x: f64| -0.5 * x);
                w_mo.i_mut((so[s].clone(), so[s].clone())).assign(&oo);
                w_mo.i_mut((sv[s].clone(), sv[s].clone())).assign(&vv);
                w_mo.i_mut((sv[s].clone(), so[s].clone())).assign(
                    &gfock_part[s].i((so[s].clone(), sv[s].clone())).t().mapv(|x: f64| -x),
                );
            }
            // forge W_II = -D_r (.) eps with the UNSYMMETRISED relaxed density and the raw energies
            *&mut w_mo -= &rdm1_resp[s] * &mo_energy[s].view().i((None, ..));
            *&mut w_mo.i_mut((so[s].clone(), so[s].clone())) -= 0.5 * axd_dr[s].i((so[s].clone(), so[s].clone()));
            w_ao[s] = (mo_coeff[s].view() % w_mo.view() % mo_coeff[s].view().t()).into_contig(ColMajor);
        }

        // ── de_h and de_s1 (spin-summed) ── //
        let mut de_h: Tsr = rt::zeros(([3, natm], &device));
        let mut de_s1: Tsr = rt::zeros(([3, natm], &device));
        let tsr_ipovlp = {
            let (out, shape) = mol.integrate("int1e_ipovlp", "s1", None).into();
            rt::asarray((out, shape, &device))
        };
        {
            let mut hcore_gen = generator_deriv_hcore(scf_data);
            let w_ao_tot = &w_ao[0] + &w_ao[1];
            let d_r_ao_tot = &d_r_ao[0] + &d_r_ao[1];
            let s1_mu = (&tsr_ipovlp * &w_ao_tot.i((.., .., None))).sum_axes(1);
            let s1_nu = (&tsr_ipovlp * &w_ao_tot.view().t().i((.., .., None))).sum_axes(1);
            let ao_slice = mol_obj.aoslice_by_atom();
            for atm in 0..natm {
                let h1 = hcore_gen(atm);
                de_h.i_mut((.., atm)).assign(&(&h1 * &d_r_ao_tot.i((.., .., None))).sum_axes([0, 1]));
                let [_, _, p0, p1] = ao_slice[atm];
                let part = (s1_mu.i(p0..p1).sum_axes(0) + s1_nu.i(p0..p1).sum_axes(0)).mapv(|x: f64| -x);
                de_s1.i_mut((.., atm)).assign(&part);
            }
        }

        // ── metric objects, following the rimatr convention (ctrl) ── //
        let j2c_decomp = get_j2c_decomp(&aux, &device, mol_obj.ctrl.j2c_decomp);
        let s_op: Tsr = match &j2c_decomp {
            J2CDecompose::Cd { j2c_l, uplo: Upper, .. } => {
                let eye = rt::eye((naux, &device));
                let u_inv = rt::linalg::solve_triangular((j2c_l.view(), eye.view(), Upper));
                u_inv.t().into_contig(ColMajor)
            },
            J2CDecompose::Eig { j2c_l_inv, .. } => j2c_l_inv.clone(),
            _ => panic!("UDHGradient does not support lower-triangular Cholesky factors (developer option)."),
        };
        let tsr_int2c2e_ip1 = {
            let (out, shape) = aux.integrate("int2c2e_ip1", "s1", None).into();
            rt::asarray((out, shape, &device))
        };
        // strictly-lower mask with 1/2 diagonal of forge's `generator_L_1`; only the Cholesky
        // convention uses it per atom (`Eig` folds the weight matrices once instead, below)
        let l1_cd = match &j2c_decomp {
            J2CDecompose::Cd { j2c_l, .. } => {
                let l_lower = j2c_l.t().into_contig(ColMajor);
                let mut l_mask: Tsr = rt::zeros(([naux, naux], &device));
                for i in 0..naux {
                    for j in 0..i {
                        l_mask[[i, j]] = 1.0;
                    }
                    l_mask[[i, i]] = 0.5;
                }
                Some((l_lower, l_mask))
            },
            J2CDecompose::Eig { .. } => None,
        };

        // ── persistent MO intermediates; density dots of the rimatr by two GEMVs per spin ── //
        // true peak of the partner intermediates in construction order per spin: each source
        // coexists with its solve transient, the transient with the permuted `[i, Q, x]` layout
        // replacing it, and all of them with the partners already built
        let mut resident = 0.0f64;
        let mut partner_peak = 0.0f64;
        for s in 0..2 {
            if nocc[s] == 0 {
                continue;
            }
            let (y_s, g_s) = ((nocc[s] * nmo * naux) as f64, (nvir[s] * nocc[s] * naux) as f64);
            partner_peak = partner_peak.max(resident + 2.0 * y_s);
            partner_peak = partner_peak.max(resident + y_s + 2.0 * g_s);
            resident += y_s + g_s;
        }
        handle_memory_exceed(partner_peak * 8.0 / 1048576.0, mem_avail, abort);
        let d_hf_tp: [Tsr; 2] =
            [pack_triu_tilde(d_hf[0].view()), pack_triu_tilde(d_hf[1].view())];
        let d_r_ao_tp: [Tsr; 2] =
            [pack_triu_tilde(d_r_ao[0].view()), pack_triu_tilde(d_r_ao[1].view())];
        let y_dot_d: [Tsr; 2] = [
            (ederi_utp.view().t() % d_hf_tp[0].view().reshape([nao_tp, 1])).into_shape([naux]),
            (ederi_utp.view().t() % d_hf_tp[1].view().reshape([nao_tp, 1])).into_shape([naux]),
        ];
        let y_dot_dr: [Tsr; 2] = [
            (ederi_utp.view().t() % d_r_ao_tp[0].view().reshape([nao_tp, 1])).into_shape([naux]),
            (ederi_utp.view().t() % d_r_ao_tp[1].view().reshape([nao_tp, 1])).into_shape([naux]),
        ];
        let mut y_ip: [Tsr; 2] = [
            rt::zeros(([nocc[0], nmo, naux], &device)),
            rt::zeros(([nocc[1], nmo, naux], &device)),
        ];
        let mut r_jk: Tsr = rt::zeros(([naux, naux], &device));
        let mut r_ri: Tsr = rt::zeros(([naux, naux], &device));
        let g2d = [g_vix[0].view().into_shape([nvir[0] * nocc[0], naux]), g_vix[1].view().into_shape([nvir[1] * nocc[1], naux])];
        for p in 0..naux {
            let col = ederi_utp.i((.., p)).unpack_tri(Upper, FlagSymm::Sy);
            for s in 0..2 {
                if nocc[s] == 0 {
                    continue;
                }
                y_ip[s].i_mut((.., .., p)).assign(&(c0[s].view().t() % &col % mo_coeff[s].view()));
                // forge RI-term metric coupling: R_ri[P, p] = sum_{i a} G[a, i, P] (i a|p),
                // spin-summed over the two channels
                let m_q = (c0[s].view().t() % &col % cv[s].view()).into_contig(ColMajor);
                let mt = m_q.view().t().into_contig(ColMajor);
                let part = g2d[s].t() % mt.view().reshape(nvir[s] * nocc[s]);
                if s == 0 {
                    r_ri.i_mut((.., p)).assign(&part);
                } else {
                    *&mut r_ri.i_mut((.., p)) += &part;
                }
            }
        }
        for s in 0..2 {
            for i in 0..nocc[s] {
                let y2t = y_ip[s].i((i, .., ..)).into_contig(ColMajor);
                let a_i = y2t.view().t() % d_r_s[s].view();
                *&mut r_jk += &a_i.view() % y2t.view();
            }
        }
        *&mut r_jk *= cx_scf;
        // `Eig` metric-derivative weights: the y-dot cross terms and the exchange weight form
        // `M_jk`, the RI weight `M_ri`; each is folded once by the Fréchet derivative of
        // `J^-1/2`, so the atom loop contracts the raw metric derivatives directly
        let l1_eig = match &j2c_decomp {
            J2CDecompose::Eig { j2c_e: Some(j2c_e), j2c_v: Some(j2c_v), .. } => {
                let thresh = mol_obj.ctrl.j2c_decomp.threshold.unwrap_or(J2C_THRESH);
                let y_dot_d_tot: Tsr = (&y_dot_d[0] + &y_dot_d[1]).into_owned();
                let y_dot_dr_tot: Tsr = (&y_dot_dr[0] + &y_dot_dr[1]).into_owned();
                let outer: Tsr = y_dot_dr_tot.view().i((.., None)) % y_dot_d_tot.view().i((None, ..));
                let m_jk: Tsr = (&outer + &outer.t()).mapv(|x: f64| -x) + 2.0 * r_jk.view();
                let m_ri: Tsr = -r_ri.view();
                Some((
                    l1_fold_eig(m_jk.view(), j2c_e.view(), j2c_v.view(), thresh),
                    l1_fold_eig(m_ri.view(), j2c_e.view(), j2c_v.view(), thresh),
                ))
            },
            J2CDecompose::Eig { .. } => panic!(
                "The eigen-decomposed 2c-2e ERI carries no stored eigenpairs; cannot build the DH metric-derivative kernels."
            ),
            J2CDecompose::Cd { .. } => None,
        };

        // ── pre-contractions `Ȳ = partner % S` in the single kept layout `[i, Q, x]`, per spin ──
        // one stored layout per partner: the solve result `[(i m), Q]` / `[(a i), Q]` is permuted
        // once into `[i, Q, m]` / `[i, Q, a]`, and each spin source is dropped before its permute
        let mut ybar_jk: [Option<Tsr>; 2] = [None, None];
        let mut ybar_ri: [Option<Tsr>; 2] = [None, None];
        let mut y_src = y_ip.map(Some);
        let mut g_src = g_vix.map(Some);
        for s in 0..2 {
            if nocc[s] == 0 {
                continue;
            }
            let src = y_src[s].take().unwrap();
            let solved: Tsr = src.view().into_shape([nocc[s] * nmo, naux]) % s_op.view(); // [(i m), Q]
            drop(src);
            ybar_jk[s] = Some(solved.view().into_shape([nocc[s], nmo, naux]).transpose([0, 2, 1]).into_contig(ColMajor)); // [i, Q, m]
            let src = g_src[s].take().unwrap();
            let solved: Tsr = src.view().into_shape([nvir[s] * nocc[s], naux]) % s_op.view(); // [(a i), Q]
            drop(src);
            ybar_ri[s] = Some(solved.view().into_shape([nvir[s], nocc[s], naux]).transpose([1, 2, 0]).into_contig(ColMajor)); // [i, Q, a]
        }

        // ── derivative-integral passes: shell batches over all atoms, atom scatter in-batch ── //
        let mol_loc = mol.ao_loc();
        let aux_loc = aux.ao_loc();
        let mol_ao_slice = mol_obj.aoslice_by_atom();
        let aux_slice = mol_obj.make_auxmol_fake().aoslice_by_atom();
        let nbas = mol.nbas();
        let nauxbas = aux.nbas();
        let ncomp = 3usize;
        let nvir_max = nvir[0].max(nvir[1]);
        let nocc_max = nocc[0].max(nocc[1]);
        let ip1_unit = 6 * nao * naux * ncomp + 2 * naux * nmo;
        let shell_batch = calc_batch_size::<f64>(ip1_unit, mem_avail, None, None).max(1);
        let ip2_unit = 3 * nao_tp + 3 * nao * nao + 3 * nocc_max * (nao + nmo + nvir_max);
        let aux_batch = calc_batch_size::<f64>(ip2_unit, mem_avail, None, None).max(1);
        handle_memory_exceed(ip1_unit as f64 * 8.0 / 1048576.0, mem_avail, abort);
        handle_memory_exceed(ip2_unit as f64 * 8.0 / 1048576.0, mem_avail, abort);

        let mut de_jk: Tsr = rt::zeros(([3, natm], &device));
        let mut de_rint: Tsr = rt::zeros(([3, natm], &device));
        let mut de_jk_acc = vec![0.0f64; 3 * natm];
        let mut de_rint_acc = vec![0.0f64; 3 * natm];
        let mut ydd_sol_atoms: [Vec<Tsr>; 2] = [Vec::with_capacity(natm), Vec::with_capacity(natm)];
        let mut ydr_sol_atoms: [Vec<Tsr>; 2] = [Vec::with_capacity(natm), Vec::with_capacity(natm)];

        // ip1 side: u-slot / v-slot of the derivative integrals, per atom and per spin
        for atm in 0..natm {
            let [shl_a0, shl_a1, _, _] = mol_ao_slice[atm];
            let mut ydd_raw: [Tsr; 2] =
                [rt::zeros(([naux, ncomp], &device)), rt::zeros(([naux, ncomp], &device))];
            let mut ydr_raw: [Tsr; 2] =
                [rt::zeros(([naux, ncomp], &device)), rt::zeros(([naux, ncomp], &device))];
            let mut t4_3c = [0.0f64; 3];
            let mut ri_3c = [0.0f64; 3];
            let shl_batches = blocksize_partition(&mol_loc[shl_a0..=shl_a1], shell_batch)
                .into_iter()
                .map(|[i0, i1]| [shl_a0 + i0, shl_a0 + i1]);
            for [s0, s1] in shl_batches {
                let ip1 = {
                    let shl_slices = [[s0, s1], [0, nbas], [0, nauxbas]];
                    let (out, shape) =
                        CInt::integrate_cross("int3c2e_ip1", [&mol, &mol, &aux], "s1", shl_slices).into();
                    rt::asarray((out, shape, &device))
                }; // [nub, naov, naux, ncomp]
                let nub = ip1.shape()[0];
                let naov = ip1.shape()[1];
                let ip1_mat = ip1.view().into_shape([nub * naov, naux * ncomp]);
                let r = mol_loc[s0]..mol_loc[s1];
                for s in 0..2 {
                    if nocc[s] == 0 {
                        continue;
                    }
                    let d_rows = d_hf[s].i((r.clone(), ..)).into_contig(ColMajor);
                    let ydd_seg = ip1_mat.view().t() % d_rows.view().reshape([nub * naov, 1]);
                    *&mut ydd_raw[s] += &ydd_seg.into_shape([naux, ncomp]);
                    let d_rows = d_r_ao[s].i((r.clone(), ..)).into_contig(ColMajor);
                    let ydr_seg = ip1_mat.view().t() % d_rows.view().reshape([nub * naov, 1]);
                    *&mut ydr_raw[s] += &ydr_seg.into_shape([naux, ncomp]);
                }
                // transposed copy `[naov, (nub naux ncomp)]`: the full-AO axis becomes a GEMM index
                let ip1_t = ip1.view().transpose([1, 0, 2, 3]).into_contig(ColMajor);
                let ip1_t2 = ip1_t.view().into_shape([naov, nub * naux * ncomp]);
                for s in 0..2 {
                    if nocc[s] == 0 {
                        continue;
                    }
                    // u-slot / v-slot of t4 (exchange upon D_r) and of the RI term, per spin
                    t4_3c = add_fold_ip1_batched(
                        ip1_t2.view(),
                        nub,
                        c0[s].i((r.clone(), ..)),
                        c1_dr[s].view(),
                        ybar_jk[s].as_ref().unwrap().view(),
                        2.0,
                        t4_3c,
                    );
                    t4_3c = add_fold_ip1_batched_v(
                        ip1_t2.view(),
                        nub,
                        c1_dr[s].i((r.clone(), ..)),
                        c0[s].view(),
                        ybar_jk[s].as_ref().unwrap().view(),
                        2.0,
                        t4_3c,
                    );
                    ri_3c = add_fold_ip1_batched(
                        ip1_t2.view(),
                        nub,
                        c0[s].i((r.clone(), ..)),
                        cv[s].view(),
                        ybar_ri[s].as_ref().unwrap().view(),
                        -1.0,
                        ri_3c,
                    );
                    ri_3c = add_fold_ip1_batched_v(
                        ip1_t2.view(),
                        nub,
                        cv[s].i((r.clone(), ..)),
                        c0[s].view(),
                        ybar_ri[s].as_ref().unwrap().view(),
                        -1.0,
                        ri_3c,
                    );
                }
            }
            for s in 0..2 {
                ydd_sol_atoms[s].push(s_op.view() % ydd_raw[s].view());
                ydr_sol_atoms[s].push(s_op.view() % ydr_raw[s].view());
            }
            for t in 0..ncomp {
                de_jk_acc[t * natm + atm] += t4_3c[t];
                de_rint_acc[t * natm + atm] += ri_3c[t];
            }
        }

        // ip2 side: Q-slot batches spanning all aux shells, scattered by atom afterwards
        let mut wdd_all: [Tsr; 2] =
            [rt::zeros(([naux, ncomp], &device)), rt::zeros(([naux, ncomp], &device))];
        let mut wdr_all: [Tsr; 2] =
            [rt::zeros(([naux, ncomp], &device)), rt::zeros(([naux, ncomp], &device))];
        let q_batches = blocksize_partition(&aux_loc[0..=nauxbas], aux_batch);
        for [q0, q1] in q_batches {
            let ip2 = {
                let shl_slices = [[0, nbas], [0, nbas], [q0, q1]];
                let (out, shape) =
                    CInt::integrate_cross("int3c2e_ip2", [&mol, &mol, &aux], "s2ij", shl_slices).into();
                rt::asarray((out, shape, &device))
            }; // [nao_tp, nb_q, ncomp]
            let nb_q = ip2.shape()[1];
            let ip2_2d = ip2.view().into_shape([nao_tp, nb_q * ncomp]);
            let wdd_batch: Vec<Tsr> = (0..2)
                .map(|s| {
                    (ip2_2d.view().t() % d_hf_tp[s].view().reshape([nao_tp, 1])).into_shape([nb_q, ncomp])
                })
                .collect();
            let wdr_batch: Vec<Tsr> = (0..2)
                .map(|s| {
                    (ip2_2d.view().t() % d_r_ao_tp[s].view().reshape([nao_tp, 1])).into_shape([nb_q, ncomp])
                })
                .collect();
            let (qa, qb) = (aux_loc[q0], aux_loc[q1]);
            for s in 0..2 {
                wdd_all[s].i_mut((qa..qb, ..)).assign(&wdd_batch[s]);
                wdr_all[s].i_mut((qa..qb, ..)).assign(&wdr_batch[s]);
            }
            let ip2_unp = ip2_2d.unpack_tri(Upper, FlagSymm::Sy); // [mu, nu, (q t)]
            for s in 0..2 {
                if nocc[s] == 0 {
                    continue;
                }
                let w1 = c0[s].view().t() % &ip2_unp; // [i, nu, (q t)]
                let f_jk = (w1.view() % c1_dr[s].view()).into_contig(ColMajor);
                let f_ri = (w1.view() % cv[s].view()).into_contig(ColMajor);
                for t in 0..ncomp {
                    // the partner is stored once, so the two contraction axes are paired
                    // individually (`(i m)` is not merged) in both halves of the reduction
                    let r_jk_q = rt::vecdot(
                        &ybar_jk[s].as_ref().unwrap().view().i((.., qa..qb, ..)),
                        &f_jk.i((.., .., t * nb_q..(t + 1) * nb_q)).transpose([0, 2, 1]),
                        ([0, 2], [0, 2]),
                    );
                    let r_ri_q = rt::vecdot(
                        &ybar_ri[s].as_ref().unwrap().view().i((.., qa..qb, ..)),
                        &f_ri.i((.., .., t * nb_q..(t + 1) * nb_q)).transpose([0, 2, 1]),
                        ([0, 2], [0, 2]),
                    );
                    for atm in 0..natm {
                        let [_, _, auq0, auq1] = aux_slice[atm];
                        let (lo, hi) = (qa.max(auq0), qb.min(auq1));
                        if lo >= hi {
                            continue;
                        }
                        let f0 = lo - qa;
                        let f1 = hi - qa;
                        de_jk_acc[t * natm + atm] += 2.0 * r_jk_q.i(f0..f1).sum_all();
                        de_rint_acc[t * natm + atm] -= r_ri_q.i(f0..f1).sum_all();
                    }
                }
            }
        }

        // ── per-atom metric couplings, L_1 terms and assembly ── //
        for atm in 0..natm {
            let [_, _, auq0, auq1] = aux_slice[atm];
            let s_op_cols = s_op.i((.., auq0..auq1));
            let ydd2: [Tsr; 2] = [
                &s_op_cols % wdd_all[0].i((auq0..auq1, ..)),
                &s_op_cols % wdd_all[1].i((auq0..auq1, ..)),
            ];
            let ydr2: [Tsr; 2] = [
                &s_op_cols % wdr_all[0].i((auq0..auq1, ..)),
                &s_op_cols % wdr_all[1].i((auq0..auq1, ..)),
            ];
            let mut ydd_l1: [Tsr; 2] =
                [rt::zeros(([naux, ncomp], &device)), rt::zeros(([naux, ncomp], &device))];
            let mut ydr_l1: [Tsr; 2] =
                [rt::zeros(([naux, ncomp], &device)), rt::zeros(([naux, ncomp], &device))];
            let mut t4_l1 = [0.0f64; 3];
            let mut ri_l1 = [0.0f64; 3];
            // `Cd`: forge's L_1 generator of this atom, contracted with the solve operator;
            // `Eig`: the raw metric derivative contracts against the folded weights directly
            for t in 0..ncomp {
                match &l1_cd {
                    Some((l_lower, l_mask)) => {
                        let m0 = &s_op_cols % tsr_int2c2e_ip1.i((auq0..auq1, .., t));
                        let mut m: Tsr = &m0 % s_op.view().t();
                        let m_t = m.view().t().into_owned();
                        *&mut m += &m_t;
                        let l1_t = (l_lower.view() % &(l_mask * &m)).mapv(|x: f64| -x);
                        let l1di = s_op.view() % l1_t.view();
                        for s in 0..2 {
                            ydd_l1[s].i_mut((.., t)).assign(&(l1di.view() % y_dot_d[s].view()));
                            ydr_l1[s].i_mut((.., t)).assign(&(l1di.view() % y_dot_dr[s].view()));
                        }
                        t4_l1[t] = 2.0 * rt::vecdot(&l1di, r_jk.view(), -1).sum_all();
                        ri_l1[t] = -rt::vecdot(&l1di, r_ri.view(), -1).sum_all();
                    },
                    None => {
                        let (k_jk, k_ri) =
                            l1_eig.as_ref().expect("the metric-derivative kernels must exist for a non-Cholesky policy.");
                        let dj = tsr_int2c2e_ip1.i((auq0..auq1, .., t));
                        t4_l1[t] = rt::vecdot(&dj, &k_jk.i((auq0..auq1, ..)), -1).sum_all();
                        ri_l1[t] = rt::vecdot(&dj, &k_ri.i((auq0..auq1, ..)), -1).sum_all();
                    },
                }
            }

            // forge t1 + t2 (all four spin pairs) + t4 (per spin), and the RI term
            let mut grad = [0.0f64; 3];
            let mut grad_ri = [0.0f64; 3];
            for t in 0..ncomp {
                let c_ydd = (&ydd2[0].i((.., t)) + &ydd_l1[0].i((.., t))
                    + &(&ydd2[1].i((.., t)) + &ydd_l1[1].i((.., t))))
                    .into_owned();
                let c_ydr = (&ydr2[0].i((.., t)) + &ydr_l1[0].i((.., t))
                    + &(&ydr2[1].i((.., t)) + &ydr_l1[1].i((.., t))))
                    .into_owned();
                let v_ydd = &ydd_sol_atoms[0][atm].i((.., t)) + &ydd_sol_atoms[1][atm].i((.., t));
                let v_ydr = &ydr_sol_atoms[0][atm].i((.., t)) + &ydr_sol_atoms[1][atm].i((.., t));
                let y_dot_d_tot = &y_dot_d[0] + &y_dot_d[1];
                let y_dot_dr_tot = &y_dot_dr[0] + &y_dot_dr[1];
                let s1: f64 = rt::vecdot(&y_dot_d_tot, &(&c_ydr + &(2.0 * &v_ydr)).view(), -1).sum_all();
                let s2: f64 = rt::vecdot(&y_dot_dr_tot, &(&c_ydd + &(2.0 * &v_ydd)).view(), -1).sum_all();
                grad[t] = -s1 - s2 + t4_l1[t] + de_jk_acc[t * natm + atm];
                grad_ri[t] = ri_l1[t] + de_rint_acc[t * natm + atm];
            }
            de_jk.i_mut((.., atm)).assign(&rt::asarray((grad.as_slice(), [3], &device)));
            de_rint.i_mut((.., atm)).assign(&rt::asarray((grad_ri.as_slice(), [3], &device)));
        }

        // ── doubly-hybrid (final-functional) terms, all absent for the pure-MP2 family ── //
        // `de_k_dh`: exchange-derivative difference upon the SCF density (`factor_k =
        //     dfa_hybrid_pos - dfa_hybrid_scf`; the J part is unchanged and covered by the "SCF"
        //     entry), i.e. the D_scf part of forge `get_gradient_jk`'s `0.5 cx_n C D_mo` piece;
        // `de_xc_n`: XC skeleton difference `xc_n - xc` upon D_scf, contracted per spin (forge
        //     `_get_gradient_gga` explicit part; the SCF-functional one is the "SCF" entry's
        //     `de_xc`);
        // `de_resp`: `<D_r_ao^s, F_1[xc]>` per spin, the explicit AO-derivative of the SCF
        //     functional's XC potential (forge `_get_gradient_gga` response part; its fxc-on-D_r
        //     half enters the CP-SCF kernel of the Z-vector instead);
        // `de_ovlp_dh`: occupied-Pulay difference `Tr(S1, W_n^s - dme0^s)` with
        //     `W_n^s = C_o^s F_n^s(D_scf)_oo C_o^sT` (no occupancy factor; forge
        //     `_get_gradient_enfunc`), against the "SCF" entry's `dme0` term.
        let mut de_k_dh: Tsr = rt::zeros(([3, natm], &device));
        let mut de_xc_n: Tsr = rt::zeros(([3, natm], &device));
        let mut de_resp: Tsr = rt::zeros(([3, natm], &device));
        let mut de_ovlp_dh: Tsr = rt::zeros(([3, natm], &device));
        if has_final_functional || add_resp {
            if add_delta_k {
                let mut grad_helper = uhf_grad_helper(scf_data, self.mpi_operator);
                grad_helper.flags.factor_k = Some(delta_hyb);
                grad_helper.calc_de_jk();
                let de_k = &grad_helper.result["de_k"];
                de_k_dh.assign(&rt::asarray((&de_k.data, de_k.size, &device)));
                // the auxiliary-basis-response part of the exchange gradient
                if let Some(de_kaux) = grad_helper.result.get("de_kaux") {
                    *&mut de_k_dh += &rt::asarray((&de_kaux.data, de_kaux.size, &device));
                }
            }
            if add_xc_n {
                // the DH total energy carries the *final* functional's XC skeleton explicitly
                // (forge `grad_gga`), while the "SCF" entry contributes the SCF functional's one;
                // add their difference, contracted with the per-spin SCF density
                let grids = scf_data.grids.as_ref().unwrap();
                let dao_xc_n = xc_skeleton_dao(scf_data, self.mpi_operator, false, grids);
                let dao_xc_scf = xc_skeleton_dao(scf_data, self.mpi_operator, true, grids);
                let mut dao_xc: Tsr = rt::zeros(([nao, 3], &device));
                for s in 0..2 {
                    *&mut dao_xc +=
                        &get_grad_dao_ovlp((&dao_xc_n[s] - &dao_xc_scf[s]).view(), d_hf[s].view());
                }
                for atm in 0..natm {
                    let [_, _, p0, p1] = mol_ao_slice[atm];
                    *&mut de_xc_n.i_mut((.., atm)) += &dao_xc.i(p0..p1).sum_axes(0);
                }
            }
            if add_resp {
                let (coords, weights, atm_idx_grids, quadrature_weights) = grids_regrouped.as_ref().unwrap();
                let dm0 = [
                    get_dm0_restricted(mo_coeff[0].view(), mo_occ[0].view()),
                    get_dm0_restricted(mo_coeff[1].view(), mo_occ[1].view()),
                ];
                let mut ni = NIMatmul::new(&mol, coords, weights, atm_idx_grids, quadrature_weights);
                // the explicit AO-derivative of the SCF functional's XC potential (forge
                // `hessian.uks._get_vxc_deriv1`), contracted with the relaxed density on the fly;
                // the grid-shift terms are absent, following the forge reference
                de_resp = de_xc_response_term(&mol, &scf_xc_func_list_uks(scf_data), &mut ni, &dm0, &d_r_ao);
            }
            if has_final_functional {
                let fock_n = fock_n.as_ref().unwrap();
                let mut w_n: Tsr = rt::zeros_like(&d_hf[0]);
                for s in 0..2 {
                    let f_n_oo = (c0[s].view().t() % fock_n[s].view() % c0[s].view()).into_contig(ColMajor);
                    *&mut w_n += &(c0[s].view() % &f_n_oo % c0[s].view().t());
                }
                // the "SCF" entry's `de_ovlp` carries the per-spin `dme0 = C (n eps) C^T` term,
                // while the final-functional Pulay reads `-Tr(S1_oo^s, F_n^s_oo)` per spin; add
                // the difference only (`2` = the two-half contraction pattern)
                let dme0 = get_dme0(mo_coeff[0].view(), mo_occ[0].view(), mo_energy[0].view())
                    + get_dme0(mo_coeff[1].view(), mo_occ[1].view(), mo_energy[1].view());
                let dao_ovlp: Tsr = get_grad_dao_ovlp(tsr_ipovlp.view(), (&w_n - &dme0).view());
                for atm in 0..natm {
                    let [_, _, p0, p1] = mol_ao_slice[atm];
                    *&mut de_ovlp_dh.i_mut((.., atm)) += &dao_ovlp.i(p0..p1).sum_axes(0);
                }
            }
        }

        // ── assemble the parts ── //
        let de = de_h.view() + de_s1.view() + de_jk.view() + de_rint.view() + de_k_dh.view()
            + de_xc_n.view() + de_resp.view() + de_ovlp_dh.view();
        for (key, val) in [
            ("de_h", de_h),
            ("de_s1", de_s1),
            ("de_jk", de_jk),
            ("de_rint", de_rint),
            ("de_k_dh", de_k_dh),
            ("de_xc_n", de_xc_n),
            ("de_resp", de_resp),
            ("de_ovlp_dh", de_ovlp_dh),
            ("de", de.into_owned()),
        ] {
            self.result.insert(key.to_string(), to_matrix_full(val));
        }
        self
    }
}

/// Device tensor (e.g. a `[3, natm]` gradient part or a `[nao, nao]` density) into a
/// `MatrixFull` of the same shape.
fn to_matrix_full(tsr: Tsr) -> MatrixFull<f64> {
    let shape: [usize; 2] = tsr.shape().to_vec().try_into().unwrap();
    MatrixFull::from_vec(shape, tsr.into_shape(-1).into_raw()).unwrap()
}

/// Fold a metric-derivative weight matrix for the eigen convention: the `L_1` term
/// `sum_weights . raw d(J^-1/2)` equals `<dJ, F(M S^-1)>` with `F` the (self-adjoint) Fréchet
/// derivative of `J^-1/2`, i.e. `V (D ∘ (V^T M^T V)) V^T` with
/// `D_ij = -1 / (sqrt(e_i) (sqrt(e_i) + sqrt(e_j)))`; the returned `G + G^T` pairs with one
/// atom block of the raw `int2c2e_ip1` (its transposed half included). Eigenpairs below
/// `threshold` are discarded, exactly as in the decomposition itself. Unrestricted twin of the
/// helper in [`crate::grad::rdh`].
fn l1_fold_eig(weight: TsrView, j2c_e: TsrView, j2c_v: TsrView, threshold: f64) -> Tsr {
    let n = j2c_e.view().less(threshold).sum();
    let sr = j2c_e.i(n..).pow(0.5);
    let v = j2c_v.i((.., n..)).into_contig(ColMajor);
    let ssum = &sr.i((.., None)) + &sr.i((None, ..));
    let d: Tsr = (&sr.i((.., None)) * &ssum).mapv(|x: f64| -1.0 / x);
    let p = v.view().t() % (weight.t() % v.view()); // V^T M^T V
    let g: Tsr = v.view() % &(&d * &p) % v.view().t();
    (&g + &g.t()).into_owned()
}

/// Whether the SCF-iteration functional carries a non-HF XC part (the gate of the
/// response-density XC term; the kernels handle LDA/GGA/MGGA alike, unlike forge's GGA-only
/// gate). Unrestricted twin of the helper in [`crate::grad::rdh`], duplicated to keep the
/// restricted module untouched.
fn has_scf_xc(scf_data: &SCF) -> bool {
    !scf_data.mol.xc_data.dfa_compnt_scf.is_empty()
}

/// A bare unrestricted gradient helper carrying the SCF control flags, for reuse of the UKS
/// gradient kernels (`calc_de_jk` with custom J/K factors, `get_vxc_rayon_new`); the unrestricted
/// twin of the helper in [`crate::grad::rdh`].
fn uhf_grad_helper<'a>(scf_data: &'a SCF, mpi_operator: &'a Option<MPIOperator>) -> RIUHFGradient<'a> {
    let mol_obj = &scf_data.mol;
    let flags = RIHFGradientFlagsBuilder::default()
        .print_level(mol_obj.ctrl.print_level)
        .max_memory(mol_obj.ctrl.max_memory)
        .auxbasis_response(mol_obj.ctrl.auxbasis_response)
        .factor_j(None)
        .factor_k(None)
        .build()
        .unwrap();
    RIUHFGradient { scf_data, flags, mpi_operator, result: HashMap::new() }
}

/// Per-spin grid contribution of the XC skeleton derivative, `vmat_s[nao, nao, 3]` (the
/// row-scatter, contracted with the per-spin SCF density and the two-half factor, is the
/// skeleton-gradient contribution); `scf` selects the SCF-iteration functional, else the final
/// (`pos`) one. Unrestricted twin of the helper in [`crate::grad::rdh`].
fn xc_skeleton_dao(
    scf_data: &SCF,
    mpi_operator: &Option<MPIOperator>,
    scf: bool,
    grids: &crate::dft::Grids,
) -> Vec<Tsr> {
    let mol_obj = &scf_data.mol;
    let xc_data = &mol_obj.xc_data;
    let (code, param) = if scf {
        (&xc_data.dfa_compnt_scf, &xc_data.dfa_paramr_scf)
    } else {
        (
            xc_data.dfa_compnt_pos.as_ref().expect("the final functional has no XC component list."),
            xc_data.dfa_paramr_pos.as_ref().expect("the final functional has no XC parameter list."),
        )
    };
    let xc_type = if code.iter().any(|&c| xc_data.init_libxc(&c).needs_tau()) {
        XCType::MGGA
    } else if code
        .iter()
        .any(|&c| !matches!(xc_data.init_libxc(&c).family(), libxc::enums::LibXCFamily::LDA | libxc::enums::LibXCFamily::HybLDA))
    {
        XCType::GGA
    } else {
        XCType::LDA
    };
    let xc_data_sel = XCData {
        xc_type,
        xc_code: code,
        xc_params: param,
        device: DeviceOpenBLAS::default(),
        dm: &scf_data.density_matrix,
        mo_coeffs: None,
        occ: None,
    };
    let mut grids_clone = grids.clone();
    // 与闭壳层一侧一致：块预算由 [ctrl] max_memory 推得
    let block_mb = xc_grad_block_mb(scf_data.mol.ctrl.max_memory, 16.0) as usize;
    crate::grad::uks::get_vxc_rayon_new(&uhf_grad_helper(scf_data, mpi_operator), &xc_data_sel, &mut grids_clone, mol_obj, block_mb)
}

/// Response-density XC term `<D_r_ao^s, F_1[A, t]>` of the DH gradient (forge
/// `_get_gradient_gga` response part, coefficient 1 per spin), with `F_1[A, t, mu, nu]` the
/// explicit AO-derivative of the SCF functional's XC potential (the gradient-level analogue of
/// forge `hessian.uks._get_vxc_deriv1`, including its fxc-upon-skeleton-density part; the
/// grid-shift terms are absent, following the forge reference).
///
/// Chunk-parallel over the (atom-grouped) grid, with the relaxed density contracted immediately,
/// so only the Hessian-level kernels of [`crate::dft::numint_matmul::hess_uks`] are reused and no
/// full-grid `[nao, nao, 3, natm]` accumulator is held. Unrestricted twin of the helper in
/// [`crate::grad::rdh`].
fn de_xc_response_term(
    mol: &CInt,
    xc_func_list: &[(f64, libxc::functional::LibXCFunctional)],
    ni: &mut NIMatmul,
    dm0: &[Tsr; 2],
    d_r_ao: &[Tsr; 2],
) -> Tsr {
    use crate::dft::numint_matmul::hess_rks::{get_drho, get_vmat_ip, get_vmat_vxc};
    use crate::dft::numint_matmul::hess_uks::{get_rho_exc_vxc_fxc_uks, get_vmat_fxc_uks};
    use crate::dft::xceff::prelude::XCDenType::*;

    let natm = mol.natm();
    let nao = mol.nao();
    let ngrids = ni.weights.len();
    let device = dm0[0].device().clone();
    let aoslices = mol.aoslice_by_atom();
    let func_refs: Vec<&libxc::functional::LibXCFunctional> = xc_func_list.iter().map(|(_, f)| f).collect();
    let xc_type = determine_den_type_from_list(&func_refs);
    // gradient-level AO derivative order (pyscf `grad.uks` convention; the Hessian needs one more)
    let ao_deriv = match xc_type {
        RHO => 1,
        _ => 2,
    };
    let ncomp_ao_dm0 = xc_type.num_ao_comp();

    let de_resp: Tsr = rt::zeros(([3, natm].f(), &device));
    let nchunk_target = (rayon::current_num_threads() * 8).max(1);
    let nsplit = ((ngrids + nchunk_target - 1) / nchunk_target).max(1);
    let chunks = (0..ngrids).step_by(nsplit).map(|s| (s, (s + nsplit).min(ngrids))).collect::<Vec<_>>();
    let guard = std::sync::Mutex::new(());
    use rayon::prelude::*;
    let weights_full = ni.weights.clone();
    let d_r_ao_ref = d_r_ao;
    let dm0_owned = [dm0[0].clone(), dm0[1].clone()];
    let ni_ref = &*ni;
    let device_ref = &device;
    let aoslices_ref = &aoslices;
    chunks.into_par_iter().for_each(|(start, end)| {
        let mut ni_chunk = ni_ref.split_batch(start, end);
        let weights = rt::asarray((&weights_full[start..end], device_ref));
        let ao = ni_chunk.get_cached_ao(ao_deriv);
        let ao_dm0_a = ao.i((Ellipsis, ..ncomp_ao_dm0)) % &dm0_owned[0];
        let ao_dm0_b = ao.i((Ellipsis, ..ncomp_ao_dm0)) % &dm0_owned[1];
        let (_rho, _exc, vxc, fxc) =
            get_rho_exc_vxc_fxc_uks(xc_func_list, ao.view(), ao_dm0_a.view(), ao_dm0_b.view());
        let wv_a = &weights * &vxc.i((.., .., 0));
        let wv_b = &weights * &vxc.i((.., .., 1));
        let wf = &weights * &fxc;
        let drho_a = get_drho(xc_type, ao.view(), ao_dm0_a.view(), aoslices_ref);
        let drho_b = get_drho(xc_type, ao.view(), ao_dm0_b.view(), aoslices_ref);
        let vmat_ip_a = get_vmat_ip(xc_type, ao.view(), wv_a.view());
        let vmat_ip_b = get_vmat_ip(xc_type, ao.view(), wv_b.view());
        let (vmat_fxc_a, vmat_fxc_b) =
            get_vmat_fxc_uks(xc_type, ao.view(), drho_a.view(), drho_b.view(), wf.view());
        let vmat_a = &vmat_fxc_a + &get_vmat_vxc(vmat_ip_a.view(), aoslices_ref);
        let vmat_b = &vmat_fxc_b + &get_vmat_vxc(vmat_ip_b.view(), aoslices_ref);
        let mut de_chunk: Tsr = rt::zeros(([3, natm].f(), &device));
        for atm in 0..natm {
            for t in 0..3 {
                let v0: f64 = rt::vecdot(
                    &d_r_ao_ref[0].view().into_shape([nao * nao]),
                    &vmat_a.i((.., .., t, atm)).into_shape([nao * nao]),
                    -1,
                )
                .sum_all();
                let v1: f64 = rt::vecdot(
                    &d_r_ao_ref[1].view().into_shape([nao * nao]),
                    &vmat_b.i((.., .., t, atm)).into_shape([nao * nao]),
                    -1,
                )
                .sum_all();
                de_chunk[[t, atm]] = v0 + v1;
            }
        }
        let _lock = guard.lock().unwrap();
        unsafe { *&mut de_resp.force_mut() += &de_chunk };
    });
    de_resp
}

/// One ip1 derivative-integral pass on an AO-shell batch, reduced against the pre-contracted
/// partner `Ȳ` (the metric solve is hoisted out of the atom loop); shared kernel with
/// [`crate::grad::rdh`], duplicated here to keep the restricted module untouched.
fn add_fold_ip1_batched(
    ip1_t2: TsrView,  // [naov, nub naux ncomp]
    nub: usize,
    c_row: TsrView,   // [nub, x], the atom-side MO weight
    c_col: TsrView,   // [naov, y], the full-AO-side MO weight
    ybar: TsrView,    // [x, naux, y], pre-contracted partner
    factor: f64,
    mut acc: [f64; 3],
) -> [f64; 3] {
    let device = ip1_t2.device().clone();
    let naov = ip1_t2.shape()[0];
    let naux = ip1_t2.shape()[1] / (nub * 3);
    let x = c_row.shape()[1];
    let y = c_col.shape()[1];
    let mut k_buf: Tsr = rt::zeros(([nub, naux * y], &device));
    k_buf.view_mut().matmul_from(&c_row, &ybar.reshape([x, naux * y]), 1.0, 0.0);
    let k_buf_view = k_buf.view();
    let k2 = k_buf_view.reshape([nub * naux, y]);
    let mut g_buf: Tsr = rt::zeros(([nub * naux, y], &device));
    for t in 0..3 {
        let blk = ip1_t2.i((.., t * nub * naux..(t + 1) * nub * naux));
        g_buf.view_mut().matmul_from(&blk.t(), &c_col, 1.0, 0.0);
        let s: f64 = rt::vecdot(&g_buf, &k2, -1).sum_all();
        acc[t] += factor * s;
    }
    acc
}

/// [`add_fold_ip1_batched`] counterpart for the `v` slot, where the partner `Ȳ[i, Q, m]` is
/// contracted over its slow `m` axis instead: the same two GEMMs are kept, but the partner
/// enters on the left of a single `[nocc naux, nmo] % [nmo, nub]` fold while the full-AO-side
/// fold is assembled from one `[y, naov] % [naov, naux]` GEMM per batch row `u` (each landing
/// contiguously in the paired buffer). Both stay within the `u` slot's GEMM structure and the
/// reduction remains a `rt::vecdot` over matching layouts, so no second partner layout is
/// stored; unrestricted twin of the kernel in [`crate::grad::rdh`].
fn add_fold_ip1_batched_v(
    ip1_t2: TsrView,  // [naov, nub naux ncomp]
    nub: usize,
    c_row: TsrView,   // [nub, x], the atom-side MO weight (contracted partner axis)
    c_col: TsrView,   // [naov, y], the full-AO-side MO weight (partner's leading axis)
    ybar: TsrView,    // [y, naux, x], pre-contracted partner in the single kept layout
    factor: f64,
    mut acc: [f64; 3],
) -> [f64; 3] {
    let device = ip1_t2.device().clone();
    let naov = ip1_t2.shape()[0];
    let naux = ip1_t2.shape()[1] / (nub * 3);
    let x = c_row.shape()[1];
    let y = c_col.shape()[1];
    let mut k_buf: Tsr = rt::zeros(([y * naux, nub], &device));
    k_buf.view_mut().matmul_from(&ybar.reshape([y * naux, x]), &c_row.t(), 1.0, 0.0);
    let mut g_buf: Tsr = rt::zeros(([y, naux, nub], &device));
    let c_col_t = c_col.t();
    let ip1_t3 = ip1_t2.reshape([naov, nub, naux * 3]); // [naov, u, (Q t)]
    for t in 0..3 {
        for u in 0..nub {
            let blk = ip1_t3.i((.., u, t * naux..(t + 1) * naux));
            g_buf.i_mut((.., .., u)).matmul_from(&c_col_t, &blk, 1.0, 0.0);
        }
        let s: f64 = rt::vecdot(&k_buf, &g_buf.view().reshape([y * naux, nub]), 0).sum_all();
        acc[t] += factor * s;
    }
    acc
}

impl GradAPI for UDHGradient<'_> {
    fn get_gradient(&self) -> MatrixFull<f64> {
        self.result["de"].clone()
    }
    fn get_energy(&self) -> f64 {
        self.energy
    }
}
