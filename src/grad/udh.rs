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
//! $\alpha\beta$ exchange in the kernel), driven by the Lagrangian
//! $L^\sigma = \mathscr{F}^{\sigma}_{vo} - \mathscr{F}^{\sigma}_{ov\top} + A^\sigma(D^-)_{vo}$.
//! The final-functional (XYG3-type) terms of the restricted module are not implemented yet; the
//! gates below panic until the next milestone, keeping the same gating structure.

use crate::analdrv::config::AnalDrvConfig;
use crate::analdrv::response::trait_uresp::URespAPI;
use crate::analdrv::response::uresp_interface::{uscf_resp_interface, URespSCF};
use crate::grad::rhf::{generator_deriv_hcore, pack_triu_tilde};
use crate::grad::traits::GradAPI;
use crate::mpi_io::MPIOperator;
use crate::ri_jk::util;
use crate::ri_jk::{get_j2c_decomp, J2CDecompose};
use crate::ri_pt2::pure_pt2_u_elecderiv::{
    get_rupt2_elec_deriv_incore, UPT2ElecDerivIncoreArg, UPT2ElecDerivIncoreInp,
};
use crate::scf_io::{SCF, SCFType};
use crate::utilities::memory_batch::{
    blocksize_partition, calc_batch_size, detect_used_memory_mb, handle_memory_exceed,
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
}

impl<'a> UDHGradient<'a> {
    pub fn new(scf_data: &'a SCF, mpi_operator: &'a Option<MPIOperator>) -> Self {
        Self { scf_data, mpi_operator, result: HashMap::new(), energy: 0.0 }
    }

    pub fn calc(&mut self) -> &mut Self {
        let scf_data = self.scf_data;
        let mol_obj = &scf_data.mol;
        let device = DeviceBLAS::default();
        assert!(mol_obj.ctrl.spin_polarization, "UDHGradient supports only unrestricted (spin-polarization = true) references.");
        assert!(matches!(scf_data.scftype, SCFType::UHF), "UDHGradient supports only UHF references.");
        assert!(mol_obj.start_mo == 0, "UDHGradient does not support frozen core (start_mo).");
        assert!(self.mpi_operator.is_none(), "UDHGradient is not MPI-parallelized yet.");
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

        // gates of the final-functional (doubly-hybrid) terms; not implemented on the
        // unrestricted side yet (next milestone), so the pure-MP2 family is the supported scope
        let xc_data = &mol_obj.xc_data;
        let delta_hyb = xc_data.dfa_hybrid_pos.unwrap_or(xc_data.dfa_hybrid_scf) - xc_data.dfa_hybrid_scf;
        let has_final_functional = delta_hyb.abs() > 1.0e-10
            || xc_data.dfa_compnt_pos.as_ref().map_or(false, |v| !v.is_empty());
        if has_final_functional {
            panic!("The unrestricted doubly-hybrid gradient currently supports the pure-MP2 family only (the final-functional terms are a separate milestone).");
        }

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

        // ── response objects, PT2 electronic derivative, relaxed correlation density ── //
        let config = AnalDrvConfig::default();
        let mut resp_objs: URespSCF = uscf_resp_interface(scf_data, &config);
        let j3c = rimatr.0.to_rstsr_view(&device);
        let out = get_rupt2_elec_deriv_incore(
            &UPT2ElecDerivIncoreInp {
                cderi: j3c.view(),
                cderi_vox: [None, None],
                occ_coeff: [c0[0].view(), c0[1].view()],
                vir_coeff: [cv[0].view(), cv[1].view()],
                occ_energy: [eo[0].view(), eo[1].view()],
                vir_energy: [ev[0].view(), ev[1].view()],
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
                let lag_vo = (&gfock_part[s].i((sv[s].clone(), so[s].clone()))
                    - &gfock_part[s].i((so[s].clone(), sv[s].clone())).t()
                    + &axd[s].i((sv[s].clone(), so[s].clone())))
                    .into_contig(ColMajor);
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
        let (l_lower, l_mask) = match &j2c_decomp {
            J2CDecompose::Cd { j2c_l, .. } => {
                let l_lower = j2c_l.t().into_contig(ColMajor);
                let mut l_mask: Tsr = rt::zeros(([naux, naux], &device));
                for i in 0..naux {
                    for j in 0..i {
                        l_mask[[i, j]] = 1.0;
                    }
                    l_mask[[i, i]] = 0.5;
                }
                (l_lower, l_mask)
            },
            _ => panic!(
                "The DH analytic gradient's metric-derivative generator currently supports the CD convention only. Please set\n[ctrl.j2c_decomp]\npolicy = \"cd\"\nin the ctrl input (the derivative-integral solves themselves follow the rimatr convention)."
            ),
        };

        // ── persistent MO intermediates; density dots of the rimatr by two GEMVs per spin ── //
        let mem_avail = mol_obj.ctrl.max_memory.map(|m| m - detect_used_memory_mb("proc"));
        let abort = mol_obj.ctrl.abort_on_mem_exceed;
        handle_memory_exceed(
            ((nocc[0] + nocc[1]) * nmo * naux + (nvir[0] * nocc[0] + nvir[1] * nocc[1]) * naux) as f64
                * 8.0
                / 1048576.0,
            mem_avail,
            abort,
        );
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

        // ── pre-contractions `Ȳ = partner % S` (replace all in-loop metric solves), per spin ── //
        let mut ybar_y2: [Option<Tsr>; 2] = [None, None];
        let mut ybar_g2i: [Option<Tsr>; 2] = [None, None];
        let mut ybar_jk_u: [Option<Tsr>; 2] = [None, None];
        let mut ybar_jk_v: [Option<Tsr>; 2] = [None, None];
        let mut ybar_ri_u: [Option<Tsr>; 2] = [None, None];
        let mut ybar_ri_v: [Option<Tsr>; 2] = [None, None];
        for s in 0..2 {
            if nocc[s] == 0 {
                continue;
            }
            let y2 = y_ip[s].view().into_shape([nocc[s] * nmo, naux]) % s_op.view();
            let g2i = g_vix[s]
                .view()
                .transpose([1, 0, 2])
                .into_contig(ColMajor)
                .view()
                .into_shape([nocc[s] * nvir[s], naux])
                % s_op.view();
            ybar_y2[s] = Some(y2);
            ybar_g2i[s] = Some(g2i);
            ybar_jk_u[s] = Some(
                ybar_y2[s]
                    .as_ref()
                    .unwrap()
                    .view()
                    .into_shape([nocc[s], nmo, naux])
                    .transpose([0, 2, 1])
                    .into_contig(ColMajor),
            );
            let yv = y_ip[s]
                .view()
                .transpose([1, 0, 2])
                .into_contig(ColMajor)
                .view()
                .into_shape([nmo * nocc[s], naux])
                % s_op.view();
            ybar_jk_v[s] = Some(
                yv.view()
                    .into_shape([nmo, nocc[s], naux])
                    .transpose([0, 2, 1])
                    .into_contig(ColMajor),
            );
            ybar_ri_u[s] = Some(
                ybar_g2i[s]
                    .as_ref()
                    .unwrap()
                    .view()
                    .into_shape([nocc[s], nvir[s], naux])
                    .transpose([0, 2, 1])
                    .into_contig(ColMajor),
            );
            let gv = g_vix[s].view().into_shape([nvir[s] * nocc[s], naux]) % s_op.view();
            ybar_ri_v[s] = Some(
                gv.view()
                    .into_shape([nvir[s], nocc[s], naux])
                    .transpose([0, 2, 1])
                    .into_contig(ColMajor),
            );
        }
        drop(y_ip);
        drop(g_vix);

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
                        ybar_jk_u[s].as_ref().unwrap().view(),
                        2.0,
                        t4_3c,
                    );
                    t4_3c = add_fold_ip1_batched(
                        ip1_t2.view(),
                        nub,
                        c1_dr[s].i((r.clone(), ..)),
                        c0[s].view(),
                        ybar_jk_v[s].as_ref().unwrap().view(),
                        2.0,
                        t4_3c,
                    );
                    ri_3c = add_fold_ip1_batched(
                        ip1_t2.view(),
                        nub,
                        c0[s].i((r.clone(), ..)),
                        cv[s].view(),
                        ybar_ri_u[s].as_ref().unwrap().view(),
                        -1.0,
                        ri_3c,
                    );
                    ri_3c = add_fold_ip1_batched(
                        ip1_t2.view(),
                        nub,
                        cv[s].i((r.clone(), ..)),
                        c0[s].view(),
                        ybar_ri_v[s].as_ref().unwrap().view(),
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
                let f_jk2 = f_jk.view().into_shape([nocc[s] * nmo, nb_q * ncomp]);
                let f_ri2 = f_ri.view().into_shape([nocc[s] * nvir[s], nb_q * ncomp]);
                for t in 0..ncomp {
                    let r_jk_q =
                        rt::vecdot(&f_jk2.i((.., t * nb_q..(t + 1) * nb_q)), ybar_y2[s].as_ref().unwrap().view().i((.., qa..qb)), 0);
                    let r_ri_q =
                        rt::vecdot(&f_ri2.i((.., t * nb_q..(t + 1) * nb_q)), ybar_g2i[s].as_ref().unwrap().view().i((.., qa..qb)), 0);
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
            for t in 0..ncomp {
                let m0 = &s_op_cols % tsr_int2c2e_ip1.i((auq0..auq1, .., t));
                let mut m: Tsr = &m0 % s_op.view().t();
                let m_t = m.view().t().into_owned();
                *&mut m += &m_t;
                let l1_t = (l_lower.view() % &(&l_mask * &m)).mapv(|x: f64| -x);
                let l1di = s_op.view() % l1_t.view();
                for s in 0..2 {
                    ydd_l1[s].i_mut((.., t)).assign(&(l1di.view() % y_dot_d[s].view()));
                    ydr_l1[s].i_mut((.., t)).assign(&(l1di.view() % y_dot_dr[s].view()));
                }
                t4_l1[t] = 2.0 * rt::vecdot(&l1di, r_jk.view(), -1).sum_all();
                ri_l1[t] = -rt::vecdot(&l1di, r_ri.view(), -1).sum_all();
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

        // ── assemble the parts ── //
        let de = de_h.view() + de_s1.view() + de_jk.view() + de_rint.view();
        for (key, val) in [
            ("de_h", de_h),
            ("de_s1", de_s1),
            ("de_jk", de_jk),
            ("de_rint", de_rint),
            ("de", de.into_owned()),
        ] {
            self.result.insert(key.to_string(), to_matrix_full(val));
        }
        self
    }
}

/// `[3, natm]` device tensor into a `MatrixFull` of the same shape.
fn to_matrix_full(tsr: Tsr) -> MatrixFull<f64> {
    let shape: [usize; 2] = tsr.shape().to_vec().try_into().unwrap();
    MatrixFull::from_vec(shape, tsr.into_shape(-1).into_raw()).unwrap()
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

impl GradAPI for UDHGradient<'_> {
    fn get_gradient(&self) -> MatrixFull<f64> {
        self.result["de"].clone()
    }
    fn get_energy(&self) -> f64 {
        self.energy
    }
}
