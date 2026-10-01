//! Restricted doubly-hybrid (XYG3-family and pure-MP2) analytic gradient.
//!
//! The SCF-iteration functional's gradient is provided by the existing `"SCF"` entry of
//! [`crate::main_driver::eval_force`]; this entry covers the D_r-carrying (correlation)
//! remainder plus the final-functional difference terms, following pyscf-forge
//! `dh/grad/dfdh.py`:
//!
//! - `de_h`: hcore-derivative term `<H_1(A), D_r>` (forge `_get_gradient_pt2`, first term);
//! - `de_s1`: overlap-Pulay term `<W, S_1(A)>` with `W = W_I + W_II + W_III` assembled from the
//!   full four-block PT2 generalized Fock, the orbital-energy terms and the SCF response upon
//!   the relaxed correlation density (forge `_get_gradient_pt2`, second term);
//! - `de_jk`: `t1 + t2 + t4` of forge `get_gradient_jk`, restricted to the D_r-carrying weights
//!   (`C1 = cx C D_r^S` with `cx = dfa_hybrid_scf`; the `0.5 cx_n C D_mo` exchange part belongs
//!   to the `"SCF"` entry and the difference term below);
//! - `de_rint`: `4 <G_ia, Y_1_ia(A)>` (forge `_get_gradient_pt2`, third term);
//! - `de_k_dh`: exchange-derivative difference upon the SCF density with `factor_k =
//!   dfa_hybrid_pos - dfa_hybrid_scf` (the D_scf part of forge `get_gradient_jk`'s `C D_mo`
//!   exchange piece; the J part is unchanged);
//! - `de_xc_n`: skeleton derivative of the final functional's XC part upon the SCF density
//!   minus the SCF functional's one (forge `_get_gradient_gga` explicit part; the latter is
//!   already carried by the `"SCF"` entry's `de_xc`);
//! - `de_resp`: `<D_r_ao, F_1[xc]>`, the explicit AO-derivative of the SCF functional's XC
//!   potential contracted with the relaxed density (forge `_get_gradient_gga` response part;
//!   the fxc-upon-D_r half enters the CP-SCF kernel of the Z-vector instead);
//! - `de_ovlp_dh`: occupied-Pulay difference, `Tr(S1, W_n - dme0)` with `W_n = C_oo F_n(D_scf)_oo
//!   C_oo^T` the final-functional energy-weighted density (forge `_get_gradient_enfunc`).
//!
//! The relaxed density itself is the unrelaxed PT2 rdm1 with its vir-occupied block replaced by
//! the DH-combined Z-vector: the Lagrangian carries the formal-SCF (Brillouin-violation) term
//! `4 Cv^T F_n(D_scf) Co` of the final functional on top of the PT2 part. All the
//! final-functional terms vanish for the pure-MP2 family (no `dfa_compnt_pos`, equal hybrid
//! coefficients), which leaves the previous RI-MP2-only behaviour untouched.
//!
//! Every in-loop metric solve of the old per-atom scheme is replaced by an exact reordering:
//! the persistent MO partners are pre-contracted once as `Ȳ = partner % S` (`S` = the solve
//! operator that generated `rimatr`, obtained from `get_j2c_decomp(ctrl.j2c_decomp)`), and each
//! shell batch reduces directly (`rt::vecdot`) against `Ȳ`, never materializing
//! `[naux, 3, x, y]` intermediates. Each partner is stored in a single layout `[i, Q, x]` (the
//! metric-solve output permuted once); both the `u` and the `v` fold slots and the ip2-side
//! reductions consume it in place, so the persistent cost stays at the two `Ȳ` tensors
//! themselves. The metric-derivative (`L_1`) term follows forge's Cholesky-generator identity
//! under the `Cd` convention; under `Eig` the same term is evaluated through the (self-adjoint)
//! Fréchet derivative of `J^{-1/2}`, which folds the weight matrices once so that the raw
//! metric-derivative integrals contract against them directly.

use crate::analdrv::response::rresp_interface::{rscf_resp_interface, scf_xc_func_list, RRespSCF};
use crate::analdrv::response::rgfock_interface::{dh_jk_factors, dh_xc_func_list, solve_z_vector};
use crate::analdrv::response::trait_rgfock::{GFockFlags, RGFockAPI};
use crate::analdrv::response::trait_rresp::RRespAPI;
use crate::dft::num_int::{NumInt, XCData};
use crate::dft::numint_matmul::nimatmul::{regroup_grids_by_atom, NIMatmul};
use crate::dft::numint_matmul::resp_rks::RRespKSNIMatmul;
use crate::dft::xc_deriv::XCType;
use crate::dft::xceff::prelude::{determine_den_type_from_list, XCDenType};
use crate::grad::rhf::{get_grad_dao_ovlp, RIRHFGradient, RIHFGradientFlagsBuilder};
use crate::grad::rks::get_vxc_rayon_new;
use crate::grad::traits::GradAPI;
use crate::mpi_io::MPIOperator;
use crate::ri_jk::resp_r::RRespRIJK;
use crate::ri_jk::util;
use crate::ri_jk::{get_j2c_decomp, J2CDecompose, J2C_THRESH};
use crate::ri_pt2::rgfock_pt2::RGFockPT2;
use crate::ri_pt2::{occ_batch_index, occ_batch_step};
use crate::scf_io::SCF;
use crate::utilities::memory_batch::{blocksize_partition, calc_batch_size, detect_used_memory_mb, handle_memory_exceed};
use crate::utilities::rstsr_util::{RestTensorToRstsrTsrAPI, RestTensorToRstsrViewAPI, Tsr, TsrView};
use rest_libcint::prelude::*;
use rstsr::prelude::*;
use std::collections::HashMap;
use tensors::MatrixFull;

/// Analytic gradient of the PT2-correlation (RI-MP2) energy, restricted case.
pub struct RDHGradient<'a> {
    scf_data: &'a SCF,
    mpi_operator: &'a Option<MPIOperator>,
    /// Gradient parts, keyed by `de_h`/`de_s1`/`de_jk`/`de_rint`/`de`, each `[3, natm]`.
    pub result: HashMap<String, MatrixFull<f64>>,
    /// PT2 correlation energy (side product of the generalized-Fock evaluation).
    pub energy: f64,
    /// Total relaxed density (SCF density plus the relaxed correlation density) prepared for the
    /// fchk density dump, in fchk section order: `[Total MP2 Density]`. `Some` only when
    /// `[ctrl] outputs` requests `fchk`; the same density the multipole task dumps.
    pub rdm1_dump: Option<Vec<MatrixFull<f64>>>,
}

impl<'a> RDHGradient<'a> {
    pub fn new(scf_data: &'a SCF, mpi_operator: &'a Option<MPIOperator>) -> Self {
        Self { scf_data, mpi_operator, result: HashMap::new(), energy: 0.0, rdm1_dump: None }
    }

    pub fn calc(&mut self) -> &mut Self {
        let scf_data = self.scf_data;
        let mol_obj = &scf_data.mol;
        let device = DeviceBLAS::default();
        assert!(!mol_obj.ctrl.spin_polarization, "RDHGradient supports only restricted (spin-polarization = false) references.");
        assert!(mol_obj.start_mo == 0, "RDHGradient does not support frozen core (start_mo).");
        assert!(self.mpi_operator.is_none(), "RDHGradient is not MPI-parallelized yet.");
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
            "RDHGradient does not support lower-triangular Cholesky factors (developer option)."
        );
        // this driver reads `rimatr` through the full-space pair tables; a storage-level pruned
        // tensor stores its retained rows only, so refuse it rather than contract wrong rows
        crate::ri_jk::require_unpruned_rimatr(&scf_data.rimatr_pair_map, "the DH analytic gradient (RDHGradient)");
        let rimatr = scf_data.rimatr.as_ref().expect(
            "Decomposed ERI (rimatr) not found; the DH analytic gradient requires the streaming/new-driver RI-PT2 engine which keeps it alive.",
        );
        let ederi_utp = rt::asarray((&rimatr.0.data, rimatr.0.size, &device));
        let mol = util::get_cint_mol(mol_obj);
        let aux = util::get_cint_aux(mol_obj);
        let natm = mol_obj.geom.elem.len();
        let naux = aux.nao();

        let mo_coeff = (&scf_data.eigenvectors[0]).to_rstsr(&device);
        let mo_occ = (&scf_data.occupation[0]).to_rstsr(&device);
        let mo_energy = (&scf_data.eigenvalues[0]).to_rstsr(&device);
        let nmo = mo_coeff.shape()[1];
        let nao = mo_coeff.shape()[0];
        let nocc = mo_occ.view().greater(0).sum();
        let nvir = nmo - nocc;
        let nao_tp = nao * (nao + 1) / 2;
        let so = rt::slice!(0, nocc);
        let sv = rt::slice!(nocc, nmo);
        let c0 = mo_coeff.i((.., so.clone())).into_contig(ColMajor);
        let cv = mo_coeff.i((.., sv.clone())).into_contig(ColMajor);
        let d_hf = (&scf_data.density_matrix[0]).to_rstsr(&device);

        // ── response objects, relaxed correlation density, full four-block gfock ──
        let config = mol_obj.ctrl.analdrv.clone().unwrap_or_default();
        let mut resp_objs: RRespSCF = rscf_resp_interface(scf_data, &config);
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
        let xc_func_pos = dh_xc_func_list(scf_data);
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

        // F_n(D_scf) = h + J - (hyb_pos / 2) K + v_xc[xc_n]; it enters the DH-combined Lagrangian
        // (the formal-SCF/Brillouin term `4 Cv^T F_n Co`) and the occupied-Pulay difference
        let fock_n = if has_final_functional {
            let (coords, weights, atm_idx_grids, quadrature_weights) = grids_regrouped.as_ref().unwrap();
            let (factor_j_dh, factor_k_dh) = dh_jk_factors(scf_data);
            let cderi_dh = rimatr.0.to_rstsr_view(&device).into_cow();
            let mut resp_rijk = RRespRIJK::new_with_cderi(factor_j_dh, factor_k_dh, cderi_dh);
            let h_core = scf_data.h_core.to_matrixfull().unwrap().to_rstsr(&device);
            let mut fock_n = &h_core + resp_rijk.get_fock_rdm(d_hf.view(), true);
            if xc_func_pos.is_some() {
                let ni_dh = NIMatmul::new(&mol, coords, weights, atm_idx_grids, quadrature_weights);
                let mut resp_ks = RRespKSNIMatmul::new(dh_xc_func_list(scf_data).unwrap(), ni_dh, mol_obj.ctrl.print_level > 2);
                *&mut fock_n += &resp_ks.get_fock_rdm(d_hf.view(), true);
            }
            Some(fock_n.into_contig(ColMajor))
        } else {
            None
        };
        // the PT2 kernel streams its amplitudes over occupied windows (storage bounded by the
        // `nvir * nocc * naux` class): size the windows so the per-window transients stay within
        // a few times that class and inside `max_memory`
        let mem_avail = mol_obj.ctrl.max_memory.map(|m| m - detect_used_memory_mb("proc"));
        let abort = mol_obj.ctrl.abort_on_mem_exceed;
        let elems_per_step = 2 * nvir * nvir * nocc + nvir * naux;
        handle_memory_exceed(elems_per_step as f64 * 8.0 / 1048576.0, mem_avail, abort);
        let step_pt2 = occ_batch_step(nocc, elems_per_step, nvir * nocc * naux, mem_avail, 4.0);
        if mol_obj.ctrl.print_level > 1 {
            println!("DH gradient amplitude windows: PT2 occ step {step_pt2} (nocc {nocc}, nvir {nvir}, naux {naux}).");
        }
        let index_occ_outer = occ_batch_index(nocc, step_pt2);
        let j3c = rimatr.0.to_rstsr_view(&device);
        let mut pt2 = RGFockPT2::<f64>::new(
            mo_coeff.clone(),
            mo_occ.clone(),
            mo_energy.clone(),
            j3c.into_cow(),
            None,
            index_occ_outer,
            c_os,
            c_ss,
        );
        pt2.dump_g_vix = true;
        resp_objs.make_cpscf_preparation(mo_coeff.view(), pt2.mo_occ.view(), pt2.mo_energy.view());
        // full four-block request (also caches `g_vix` and the SCF response upon rdm1_corr),
        // then the relaxed density through the DH-level Z-vector
        let _ = pt2.make_gfock(Some(&mut resp_objs), GFockFlags::OO | GFockFlags::VV);
        let rdm1 = pt2.result["rdm1"].to_owned();
        let rdm1_resp = match &fock_n {
            Some(fock_n) => {
                // DH-combined Lagrangian: the PT2 part plus the formal-SCF term `4 Cv^T F_n Co`
                let lag_vo = pt2.make_lagrangian_vo(&mut resp_objs);
                let lag_xc_n: Tsr<f64> = 4.0 * (cv.view().t() % fock_n.view() % c0.view());
                let lag_dh = (&lag_vo + &lag_xc_n).into_contig(ColMajor);
                let z = solve_z_vector(lag_dh.view(), &mut resp_objs);
                let mut r = rdm1.clone();
                r.i_mut((sv.clone(), so.clone())).assign(&z.i((sv.clone(), so.clone())));
                r
            },
            None => {
                let _ = pt2.make_rdm1_resp(&mut resp_objs);
                pt2.result["rdm1_resp"].to_owned()
            },
        };
        let gfock_part = pt2.result["gfock_part_full"].to_owned();
        let g_vix = pt2.result["g_vix"].to_owned(); // [nvir, nocc, naux]
        self.energy = pt2.e_corr.unwrap_or(0.0);
        drop(pt2);

        let d_r_sym = (&rdm1_resp + &rdm1_resp.t()).into_contig(ColMajor);
        let d_r_s: Tsr<f64> = d_r_sym.mapv(|x| 0.5 * x);
        // the exchange (K) part of the SCF functional linearized upon the relaxed density carries
        // the SCF hybrid coefficient (forge `C1 = cx C D_r_symm`; unity for the MP2/HF reference)
        let cx_scf = mol_obj.xc_data.dfa_hybrid_scf;
        let c1_dr: Tsr<f64> = (mo_coeff.view() % d_r_s.view()).mapv(|x| cx_scf * x).into_contig(ColMajor);
        let d_r_ao = (mo_coeff.view() % d_r_s.view() % mo_coeff.view().t()).into_contig(ColMajor);
        // total relaxed density (SCF + relaxed correlation) for the fchk density dump of force
        // jobs; only materialized when the fchk output is requested (`SCF::dh_rdm1_resp`)
        if mol_obj.ctrl.outputs.iter().any(|output| output.eq("fchk")) {
            self.rdm1_dump = Some(vec![to_matrix_full(&d_hf + &d_r_ao)]);
        }

        // SCF response upon the *relaxed* correlation density (forge W_III); the cached `axd`
        // of the generalized-Fock evaluation acts on the unrelaxed rdm1 instead.
        let axd_dr = {
            let resp_ao = resp_objs.get_response_rdm(d_r_ao.view(), true);
            (mo_coeff.view().t() % resp_ao % mo_coeff.view()).into_contig(ColMajor)
        };

        // ── W = W_I + W_II + W_III (MO), from the axd-free `gfock_part_full` blocks ──
        let eps2: Tsr<f64> = 2.0 * mo_energy.view();
        let mut w_mo: Tsr<f64> = rt::zeros(([nmo, nmo], &device));
        {
            let oo = (&gfock_part.i((so.clone(), so.clone())) - &eps2.i((None, so.clone())) * rdm1.i((so.clone(), so.clone())))
                .mapv(|x: f64| -0.5 * x);
            let vv = (&gfock_part.i((sv.clone(), sv.clone())) - &eps2.i((None, sv.clone())) * rdm1.i((sv.clone(), sv.clone())))
                .mapv(|x: f64| -0.5 * x);
            w_mo.i_mut((so.clone(), so.clone())).assign(&oo);
            w_mo.i_mut((sv.clone(), sv.clone())).assign(&vv);
            w_mo.i_mut((sv.clone(), so.clone())).assign(&gfock_part.i((so.clone(), sv.clone())).t().mapv(|x: f64| -x));
        }
        // forge W_II = -D_r (.) eps with the UNSYMMETRISED relaxed density (its [ov] half is
        // zero); the raw (non-doubled) orbital energies enter here
        *&mut w_mo -= &rdm1_resp * &mo_energy.view().i((None, ..));
        *&mut w_mo.i_mut((so.clone(), so.clone())) -= 0.5 * axd_dr.i((so.clone(), so.clone()));
        let w_ao: Tsr<f64> = (mo_coeff.view() % w_mo.view() % mo_coeff.view().t()).into_contig(ColMajor);

        // ── de_h and de_s1 ──
        let mut de_h: Tsr<f64> = rt::zeros(([3, natm], &device));
        let mut de_s1: Tsr<f64> = rt::zeros(([3, natm], &device));
        let tsr_ipovlp = {
            let (out, shape) = mol.integrate("int1e_ipovlp", "s1", None).into();
            rt::asarray((out, shape, &device))
        };
        {
            let mut hcore_gen = crate::grad::rhf::generator_deriv_hcore(scf_data);
            // two half-row contractions of the skeleton overlap derivative (forge S_1_ao)
            let s1_mu = (&tsr_ipovlp * &w_ao.i((.., .., None))).sum_axes(1);
            let s1_nu = (&tsr_ipovlp * &w_ao.view().t().i((.., .., None))).sum_axes(1);
            let ao_slice = mol_obj.aoslice_by_atom();
            for atm in 0..natm {
                let h1 = hcore_gen(atm);
                de_h.i_mut((.., atm)).assign(&(&h1 * &d_r_ao.i((.., .., None))).sum_axes([0, 1]));
                let [_, _, p0, p1] = ao_slice[atm];
                let part = (s1_mu.i(p0..p1).sum_axes(0) + s1_nu.i(p0..p1).sum_axes(0)).mapv(|x: f64| -x);
                de_s1.i_mut((.., atm)).assign(&part);
            }
        }

        // ── metric objects, following the rimatr convention (ctrl) ──
        let j2c_decomp = get_j2c_decomp(&aux, &device, mol_obj.ctrl.j2c_decomp);
        // `S` = the solve operator that generated the rimatr: `L_inv` (Cd) or `J^-1/2` (Eig)
        let s_op: Tsr<f64> = match &j2c_decomp {
            J2CDecompose::Cd { j2c_l, uplo: Upper, .. } => {
                let eye = rt::eye((naux, &device));
                let u_inv = rt::linalg::solve_triangular((j2c_l.view(), eye.view(), Upper));
                u_inv.t().into_contig(ColMajor)
            },
            J2CDecompose::Eig { j2c_l_inv, .. } => j2c_l_inv.clone(),
            _ => panic!("RDHGradient does not support lower-triangular Cholesky factors (developer option)."),
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
                let mut l_mask: Tsr<f64> = rt::zeros(([naux, naux], &device));
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

        // ── persistent MO intermediates; density dots of the rimatr by two GEMVs ──
        let mem_avail = mol_obj.ctrl.max_memory.map(|m| m - detect_used_memory_mb("proc"));
        let abort = mol_obj.ctrl.abort_on_mem_exceed;
        // true peak of the partner intermediates: each source coexists with its metric-solve
        // transient, which in turn coexists with the permuted `[i, Q, x]` layout replacing it
        // (the partner built first stays resident throughout the second one's construction)
        let (n_y, n_g) = ((nocc * nmo * naux) as f64, (nvir * nocc * naux) as f64);
        handle_memory_exceed((2.0 * n_y).max(n_y + 2.0 * n_g) * 8.0 / 1048576.0, mem_avail, abort);
        let d_hf_tp = crate::grad::rhf::pack_triu_tilde(d_hf.view());
        let d_r_ao_tp = crate::grad::rhf::pack_triu_tilde(d_r_ao.view());
        let y_dot_d: Tsr<f64> = (ederi_utp.view().t() % d_hf_tp.view().reshape([nao_tp, 1])).into_shape([naux]);
        let y_dot_dr: Tsr<f64> = (ederi_utp.view().t() % d_r_ao_tp.view().reshape([nao_tp, 1])).into_shape([naux]);
        let mut y_ip: Tsr<f64> = rt::zeros(([nocc, nmo, naux], &device));
        let mut r_jk: Tsr<f64> = rt::zeros(([naux, naux], &device));
        let mut r_ri: Tsr<f64> = rt::zeros(([naux, naux], &device));
        let g2d = g_vix.view().into_shape([nvir * nocc, naux]); // (a i)-packed rows, P columns
        for p in 0..naux {
            let col = ederi_utp.i((.., p)).unpack_tri(Upper, FlagSymm::Sy);
            y_ip.i_mut((.., .., p)).assign(&(c0.view().t() % &col % mo_coeff.view()));
            // forge RI-term metric coupling: R2[P, p] = sum_{i a} G[a, i, P] (i a|p)
            let m_q = (c0.view().t() % &col % cv.view()).into_contig(ColMajor); // [nocc(i), nvir(a)]
            let mt = m_q.view().t().into_contig(ColMajor); // [nvir, nocc], (a i)-packed on flatten
            r_ri.i_mut((.., p)).assign(&(g2d.t() % mt.view().reshape(nvir * nocc)));
        }
        // forge t4 metric coupling: R[P, Q] = sum_{i p q} Y_ip[P, i, p] D_r_s[p, q] Y_ip[Q, i, q]
        for i in 0..nocc {
            let y2t = y_ip.i((i, .., ..)).into_contig(ColMajor); // [nmo(p), naux(P)]
            let a_i = y2t.view().t() % d_r_s.view(); // [naux(P), nmo(q)]
            *&mut r_jk += &a_i.view() % y2t.view();
        }
        *&mut r_jk *= cx_scf;
        // `Eig` metric-derivative weights: the y-dot cross terms and the exchange weight form
        // `M_jk`, the RI weight `M_ri`; each is folded once by the Fréchet derivative of
        // `J^-1/2`, so the atom loop contracts the raw metric derivatives directly
        let l1_eig = match &j2c_decomp {
            J2CDecompose::Eig { j2c_e: Some(j2c_e), j2c_v: Some(j2c_v), .. } => {
                let thresh = mol_obj.ctrl.j2c_decomp.threshold.unwrap_or(J2C_THRESH);
                let outer: Tsr<f64> = y_dot_dr.view().i((.., None)) % y_dot_d.view().i((None, ..));
                let m_jk: Tsr<f64> = (&outer + &outer.t()).mapv(|x: f64| -x) + 2.0 * r_jk.view();
                let m_ri: Tsr<f64> = -4.0 * r_ri.view();
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

        // ── pre-contractions `Ȳ = partner % S` in the single kept layout `[i, Q, x]` ──
        // one stored layout per partner: the solve result `[(i m), Q]` / `[(a i), Q]` is permuted
        // once into `[i, Q, m]` / `[i, Q, a]`, and each source is dropped before the permutation
        let ybar_jk: Tsr<f64> = {
            let solved: Tsr<f64> = y_ip.view().into_shape([nocc * nmo, naux]) % s_op.view(); // [(i m), Q]
            drop(y_ip);
            let ybar = solved.view().into_shape([nocc, nmo, naux]).transpose([0, 2, 1]).into_contig(ColMajor); // [i, Q, m]
            ybar
        };
        let ybar_ri: Tsr<f64> = {
            let solved: Tsr<f64> = g_vix.view().into_shape([nvir * nocc, naux]) % s_op.view(); // [(a i), Q]
            drop(g_vix);
            let ybar = solved.view().into_shape([nvir, nocc, naux]).transpose([1, 2, 0]).into_contig(ColMajor); // [i, Q, a]
            ybar
        };

        // ── derivative-integral passes: shell batches over all atoms, atom scatter in-batch ──
        let mol_loc = mol.ao_loc();
        let aux_loc = aux.ao_loc();
        let mol_ao_slice = mol_obj.aoslice_by_atom();
        let aux_slice = mol_obj.make_auxmol_fake().aoslice_by_atom();
        let nbas = mol.nbas();
        let nauxbas = aux.nbas();
        let ncomp = 3usize;
        // transient per AO function: raw ip1 + transposed copy + fold scratch + partner fold
        let ip1_unit = 6 * nao * naux * ncomp + 2 * naux * nmo;
        let shell_batch = calc_batch_size::<f64>(ip1_unit, mem_avail, None, None).max(1);
        // transient per aux function: raw ip2 + unpacked + folded (g_o, f_jk, f_ri)
        let ip2_unit = 3 * nao_tp + 3 * nao * nao + 3 * nocc * (nao + nmo + nvir);
        let aux_batch = calc_batch_size::<f64>(ip2_unit, mem_avail, None, None).max(1);
        handle_memory_exceed(ip1_unit as f64 * 8.0 / 1048576.0, mem_avail, abort);
        handle_memory_exceed(ip2_unit as f64 * 8.0 / 1048576.0, mem_avail, abort);

        let mut de_jk: Tsr<f64> = rt::zeros(([3, natm], &device));
        let mut de_rint: Tsr<f64> = rt::zeros(([3, natm], &device));
        let mut de_jk_acc = vec![0.0f64; 3 * natm];
        let mut de_rint_acc = vec![0.0f64; 3 * natm];
        let mut ydd_sol_atoms: Vec<Tsr<f64>> = Vec::with_capacity(natm);
        let mut ydr_sol_atoms: Vec<Tsr<f64>> = Vec::with_capacity(natm);

        // ip1 side: u-slot / v-slot of the derivative integrals, per atom
        for atm in 0..natm {
            let [shl_a0, shl_a1, _, _] = mol_ao_slice[atm];
                let mut ydd_raw: Tsr<f64> = rt::zeros(([naux, ncomp], &device));
            let mut ydr_raw: Tsr<f64> = rt::zeros(([naux, ncomp], &device));
            let mut t4_3c = [0.0f64; 3];
            let mut ri_3c = [0.0f64; 3];
            let shl_batches = blocksize_partition(&mol_loc[shl_a0..=shl_a1], shell_batch)
                .into_iter()
                .map(|[i0, i1]| [shl_a0 + i0, shl_a0 + i1]);
            for [s0, s1] in shl_batches {
                let ip1 = {
                    let shl_slices = [[s0, s1], [0, nbas], [0, nauxbas]];
                    let (out, shape) = CInt::integrate_cross("int3c2e_ip1", [&mol, &mol, &aux], "s1", shl_slices).into();
                    rt::asarray((out, shape, &device))
                }; // [nub, naov, naux, ncomp]
                let nub = ip1.shape()[0];
                let naov = ip1.shape()[1];
                // forge t1/t2 density vectors of this batch (`u` rows of the densities)
                let ip1_mat = ip1.view().into_shape([nub * naov, naux * ncomp]);
                {
                    let d_rows = d_hf.i((mol_loc[s0]..mol_loc[s1], ..)).into_contig(ColMajor);
                    let ydd_seg = ip1_mat.view().t() % d_rows.view().reshape([nub * naov, 1]);
                    *&mut ydd_raw += &ydd_seg.into_shape([naux, ncomp]);
                }
                {
                    let d_rows = d_r_ao.i((mol_loc[s0]..mol_loc[s1], ..)).into_contig(ColMajor);
                    let ydr_seg = ip1_mat.view().t() % d_rows.view().reshape([nub * naov, 1]);
                    *&mut ydr_raw += &ydr_seg.into_shape([naux, ncomp]);
                }
                // transposed copy `[naov, (nub naux ncomp)]`: the full-AO axis becomes a GEMM index
                let ip1_t = ip1.view().transpose([1, 0, 2, 3]).into_contig(ColMajor);
                let ip1_t2 = ip1_t.view().into_shape([naov, nub * naux * ncomp]);
                let r = mol_loc[s0]..mol_loc[s1];
                // u-slot (C0 on the atom rows) and v-slot (C1_dr / C0 swapped), t4 then RI term
                t4_3c = add_fold_ip1_batched(ip1_t2.view(), nub, c0.i((r.clone(), ..)), c1_dr.view(), ybar_jk.view(), 2.0, t4_3c);
                t4_3c = add_fold_ip1_batched_v(ip1_t2.view(), nub, c1_dr.i((r.clone(), ..)), c0.view(), ybar_jk.view(), 2.0, t4_3c);
                ri_3c = add_fold_ip1_batched(ip1_t2.view(), nub, c0.i((r.clone(), ..)), cv.view(), ybar_ri.view(), -4.0, ri_3c);
                ri_3c = add_fold_ip1_batched_v(ip1_t2.view(), nub, cv.i((r, ..)), c0.view(), ybar_ri.view(), -4.0, ri_3c);
            }
            ydd_sol_atoms.push(s_op.view() % ydd_raw.view());
            ydr_sol_atoms.push(s_op.view() % ydr_raw.view());
            for t in 0..ncomp {
                de_jk_acc[t * natm + atm] += t4_3c[t];
                de_rint_acc[t * natm + atm] += ri_3c[t];
            }
        }

        // ip2 side: Q-slot batches spanning all aux shells, scattered by atom afterwards
        let mut wdd_all: Tsr<f64> = rt::zeros(([naux, ncomp], &device));
        let mut wdr_all: Tsr<f64> = rt::zeros(([naux, ncomp], &device));
        let q_batches = blocksize_partition(&aux_loc[0..=nauxbas], aux_batch);
        for [q0, q1] in q_batches {
            let ip2 = {
                let shl_slices = [[0, nbas], [0, nbas], [q0, q1]];
                let (out, shape) = CInt::integrate_cross("int3c2e_ip2", [&mol, &mol, &aux], "s2ij", shl_slices).into();
                rt::asarray((out, shape, &device))
            }; // [nao_tp, nb_q, ncomp]
            let nb_q = ip2.shape()[1];
            let ip2_2d = ip2.view().into_shape([nao_tp, nb_q * ncomp]);
            // density vectors of this aux batch, scattered into the atom-local rows
            let wdd_batch = (ip2_2d.view().t() % d_hf_tp.view().reshape([nao_tp, 1])).into_shape([nb_q, ncomp]);
            let wdr_batch = (ip2_2d.view().t() % d_r_ao_tp.view().reshape([nao_tp, 1])).into_shape([nb_q, ncomp]);
            let (qa, qb) = (aux_loc[q0], aux_loc[q1]);
            wdd_all.i_mut((qa..qb, ..)).assign(&wdd_batch);
            wdr_all.i_mut((qa..qb, ..)).assign(&wdr_batch);
            // MO folds with C_o first, then the nmo/nvir side (never forming [naux, 3, x, y])
            // two batched half-transforms (the per-(q,t) `c0^T % col % w_right`, batched over the
            // aux batch), never forming the `[naux, 3, x, y]` intermediate of the metric solve
            let ip2_unp = ip2_2d.unpack_tri(Upper, FlagSymm::Sy); // [mu, nu, (q t)]
            let w1 = c0.view().t() % &ip2_unp; // [i, nu, (q t)]
            // the batched-matmul output stores the batch axis first; materialize standard
            // col-major so the `(i, m)` / `(i, a)` axes pair with the partner's `[i, q, x]` slots
            let f_jk = (w1.view() % c1_dr.view()).into_contig(ColMajor); // [i, m, (q t)]
            let f_ri = (w1.view() % cv.view()).into_contig(ColMajor); // [i, a, (q t)]
            for t in 0..ncomp {
                // `r_jk_q[q] = sum_{i m} F_jk[i, m, (q t)] Ȳ[i, q, m]` — the partner is stored once,
                // so the two contraction axes are paired individually (`(i m)` is not merged)
                let r_jk_q = rt::vecdot(
                    &ybar_jk.i((.., qa..qb, ..)),
                    &f_jk.i((.., .., t * nb_q..(t + 1) * nb_q)).transpose([0, 2, 1]),
                    ([0, 2], [0, 2]),
                );
                let r_ri_q = rt::vecdot(
                    &ybar_ri.i((.., qa..qb, ..)),
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
                    de_rint_acc[t * natm + atm] -= 4.0 * r_ri_q.i(f0..f1).sum_all();
                }
            }
        }

        // ── per-atom metric couplings, L_1 terms and assembly ──
        for atm in 0..natm {
            let [_, _, auq0, auq1] = aux_slice[atm];
            let s_op_cols = s_op.i((.., auq0..auq1));
            let ydd2 = &s_op_cols % wdd_all.i((auq0..auq1, ..)); // [naux, 3]
            let ydr2 = &s_op_cols % wdr_all.i((auq0..auq1, ..));
            let mut ydd_l1: Tsr<f64> = rt::zeros(([naux, ncomp], &device));
            let mut ydr_l1: Tsr<f64> = rt::zeros(([naux, ncomp], &device));
            let mut t4_l1 = [0.0f64; 3];
            let mut ri_l1 = [0.0f64; 3];
            // `Cd`: forge's L_1 generator of this atom, contracted with the solve operator;
            // `Eig`: the raw metric derivative contracts against the folded weights directly
            for t in 0..ncomp {
                match &l1_cd {
                    Some((l_lower, l_mask)) => {
                        let m0 = &s_op_cols % tsr_int2c2e_ip1.i((auq0..auq1, .., t));
                        let mut m: Tsr<f64> = &m0 % s_op.view().t();
                        let m_t = m.view().t().into_owned();
                        *&mut m += &m_t;
                        let l1_t = (l_lower.view() % &(l_mask * &m)).mapv(|x: f64| -x);
                        let l1di = s_op.view() % l1_t.view();
                        ydd_l1.i_mut((.., t)).assign(&(l1di.view() % y_dot_d.view()));
                        ydr_l1.i_mut((.., t)).assign(&(l1di.view() % y_dot_dr.view()));
                        t4_l1[t] = 2.0 * rt::vecdot(&l1di, r_jk.view(), -1).sum_all();
                        ri_l1[t] = -4.0 * rt::vecdot(&l1di, r_ri.view(), -1).sum_all();
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

            // assemble forge t1 + t2 + t4 (D_r-carrying) and the RI term
            let ydd_sol = &ydd_sol_atoms[atm];
            let ydr_sol = &ydr_sol_atoms[atm];
            let mut grad = [0.0f64; 3];
            let mut grad_ri = [0.0f64; 3];
            for t in 0..ncomp {
                let v_ydd = ydd_sol.i((.., t));
                let v_ydr = ydr_sol.i((.., t));
                let c_ydd: Tsr<f64> = ydd2.i((.., t)) + ydd_l1.i((.., t));
                let c_ydr: Tsr<f64> = 2.0 * &v_ydr + &ydr2.i((.., t)) + &ydr_l1.i((.., t));
                let s1: f64 = rt::vecdot(&y_dot_d, c_ydr.view(), -1).sum_all();
                let s2: f64 = rt::vecdot(&y_dot_dr, (&c_ydd + &(2.0 * &v_ydd)).view(), -1).sum_all();
                grad[t] = -s1 - s2 + t4_l1[t] + de_jk_acc[t * natm + atm];
                grad_ri[t] = ri_l1[t] + de_rint_acc[t * natm + atm];
            }
            de_jk.i_mut((.., atm)).assign(&rt::asarray((grad.as_slice(), [3], &device)));
            de_rint.i_mut((.., atm)).assign(&rt::asarray((grad_ri.as_slice(), [3], &device)));
        }

        // ── doubly-hybrid (final-functional) terms, all absent for the pure-MP2 family ──
        // `de_k_dh`: exchange-derivative difference upon the SCF density (`factor_k =
        //     dfa_hybrid_pos - dfa_hybrid_scf`; the J part is unchanged and covered by the "SCF"
        //     entry), i.e. the D_scf part of forge `get_gradient_jk`'s `0.5 cx_n C D_mo` piece;
        // `de_xc_n`: XC skeleton difference `xc_n - xc` upon D_scf (forge `_get_gradient_gga`
        //     explicit part; the SCF-functional one is the "SCF" entry's `de_xc`);
        // `de_resp`: `<D_r_ao, F_1[xc]>`, the explicit AO-derivative of the SCF functional's XC
        //     potential (forge `_get_gradient_gga` response part; its fxc-on-D_r half enters the
        //     CP-SCF kernel of the Z-vector instead);
        // `de_ovlp_dh`: occupied-Pulay difference `-2 Tr(S1, W_n) - (-Tr(S1, dme0))` with
        //     `W_n = C_oo F_n(D_scf)_oo C_oo^T` (forge `_get_gradient_enfunc`), against the
        //     "SCF" entry's `dme0` term.
        let mut de_k_dh: Tsr<f64> = rt::zeros(([3, natm], &device));
        let mut de_xc_n: Tsr<f64> = rt::zeros(([3, natm], &device));
        let mut de_resp: Tsr<f64> = rt::zeros(([3, natm], &device));
        let mut de_ovlp_dh: Tsr<f64> = rt::zeros(([3, natm], &device));
        if has_final_functional || add_resp {
            if add_delta_k {
                let mut grad_helper = dh_grad_helper(scf_data, self.mpi_operator);
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
                // add their difference
                let dao_xc_n = xc_skeleton_dao(scf_data, self.mpi_operator, false, scf_data.grids.as_ref().unwrap());
                let dao_xc_scf = xc_skeleton_dao(scf_data, self.mpi_operator, true, scf_data.grids.as_ref().unwrap());
                let dao_xc: Tsr<f64> = get_grad_dao_ovlp((&dao_xc_n - &dao_xc_scf).view(), d_hf.view());
                for atm in 0..natm {
                    let [_, _, p0, p1] = mol_ao_slice[atm];
                    *&mut de_xc_n.i_mut((.., atm)) += &dao_xc.i(p0..p1).sum_axes(0);
                }
            }
            if add_resp {
                let (coords, weights, atm_idx_grids, quadrature_weights) = grids_regrouped.as_ref().unwrap();
                let dm0 = util::get_dm0_restricted(mo_coeff.view(), mo_occ.view());
                let mut ni = NIMatmul::new(&mol, coords, weights, atm_idx_grids, quadrature_weights);
                // the explicit AO-derivative of the SCF functional's XC potential (forge
                // `hessian.rks._get_vxc_deriv1`), contracted with the relaxed density on the fly;
                // the grid-shift terms are absent, following the forge reference
                de_resp = de_xc_response_term(&mol, &scf_xc_func_list(scf_data), &mut ni, dm0.view(), d_r_ao.view());
            }
            if has_final_functional {
                let fock_n = fock_n.as_ref().unwrap();
                let f_n_oo = (c0.view().t() % fock_n.view() % c0.view()).into_contig(ColMajor);
                let w_n = (c0.view() % &f_n_oo % c0.view().t()).into_contig(ColMajor);
                // the "SCF" entry's `de_ovlp` carries the `dme0 = C (n eps) C^T` term, while the
                // final-functional Pulay reads `-2 Tr(S1, W_n)` (no occupancy factor); add the
                // difference `f(2 W_n - dme0)` only (`f` = the two-half contraction pattern)
                let dme0 = crate::grad::rhf::get_dme0(mo_coeff.view(), mo_occ.view(), mo_energy.view());
                let dao_ovlp: Tsr<f64> =
                    get_grad_dao_ovlp(tsr_ipovlp.view(), (&(&w_n * 2.0) - &dme0).view());
                for atm in 0..natm {
                    let [_, _, p0, p1] = mol_ao_slice[atm];
                    *&mut de_ovlp_dh.i_mut((.., atm)) += &dao_ovlp.i(p0..p1).sum_axes(0);
                }
            }
        }

        // ── assemble the parts ──
        let de = de_h.view() + de_s1.view() + de_jk.view() + de_rint.view() + de_k_dh.view() + de_xc_n.view()
            + de_resp.view() + de_ovlp_dh.view();
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

/// Response-density XC term `<D_r_ao, F_1[A, t]>` of the DH gradient (forge `_get_gradient_gga`
/// response part), with `F_1[A, t, mu, nu]` the explicit AO-derivative of the SCF functional's XC
/// potential (the gradient-level analogue of forge `hessian.rks._get_vxc_deriv1`, including its
/// fxc-upon-skeleton-density part; the grid-shift terms are absent, following the forge
/// reference).
///
/// Chunk-parallel over the (atom-grouped) grid on first-and-second-order AO channels, with the
/// relaxed density contracted immediately, so only the Hessian-level kernels of
/// [`crate::dft::numint_matmul::hess_rks`] are reused and no full-grid
/// `[nao, nao, 3, natm]` accumulator is held.
fn de_xc_response_term(
    mol: &CInt,
    xc_func_list: &[(f64, libxc::functional::LibXCFunctional)],
    ni: &mut NIMatmul,
    dm0: TsrView,
    d_r_ao: TsrView,
) -> Tsr<f64> {
    use crate::dft::numint_matmul::hess_rks::{get_drho, get_rho_exc_vxc_fxc, get_vmat_fxc, get_vmat_ip, get_vmat_vxc};
    use crate::dft::xceff::prelude::XCDenType::*;

    let natm = mol.natm();
    let nao = mol.nao();
    let ngrids = ni.weights.len();
    let device = dm0.device().clone();
    let aoslices = mol.aoslice_by_atom();
    let func_refs: Vec<&libxc::functional::LibXCFunctional> = xc_func_list.iter().map(|(_, f)| f).collect();
    let xc_type = determine_den_type_from_list(&func_refs);
    // gradient-level AO derivative order (pyscf `grad.rks` convention; the Hessian needs one more)
    let ao_deriv = match xc_type {
        RHO => 1,
        _ => 2,
    };
    let ncomp_ao_dm0 = xc_type.num_ao_comp();

    let de_resp: Tsr<f64> = rt::zeros(([3, natm].f(), &device));
    let nchunk_target = (rayon::current_num_threads() * 8).max(1);
    let nsplit = ((ngrids + nchunk_target - 1) / nchunk_target).max(1);
    let chunks = (0..ngrids).step_by(nsplit).map(|s| (s, (s + nsplit).min(ngrids))).collect::<Vec<_>>();
    let guard = std::sync::Mutex::new(());
    use rayon::prelude::*;
    let weights_full = ni.weights.clone();
    let d_r_ao_ref = &d_r_ao;
    let dm0_owned: Tsr<f64> = dm0.into_contig(ColMajor).into_owned();
    let ni_ref = &*ni;
    let device_ref = &device;
    let aoslices_ref = &aoslices;
    chunks.into_par_iter().for_each(|(start, end)| {
        let mut ni_chunk = ni_ref.split_batch(start, end);
        let weights = rt::asarray((&weights_full[start..end], device_ref));
        let ao = ni_chunk.get_cached_ao(ao_deriv);
        let ao_dm0 = ao.i((Ellipsis, ..ncomp_ao_dm0)) % &dm0_owned;
        let (_rho, _exc, vxc, fxc) = get_rho_exc_vxc_fxc(xc_func_list, ao.view(), ao_dm0.view());
        let wv = &weights * &vxc;
        let wf = &weights * &fxc;
        let drho = get_drho(xc_type, ao.view(), ao_dm0.view(), aoslices_ref);
        let vmat_ip = get_vmat_ip(xc_type, ao.view(), wv.view());
        let vmat_fxc = get_vmat_fxc(xc_type, ao.view(), drho.view(), wf.view());
        let vmat_vxc = get_vmat_vxc(vmat_ip.view(), aoslices_ref);
        let vmat = &vmat_fxc + &vmat_vxc;
        let mut de_chunk: Tsr<f64> = rt::zeros(([3, natm].f(), &device));
        for atm in 0..natm {
            for t in 0..3 {
                let v: f64 = rt::vecdot(
                    &d_r_ao_ref.view().into_shape([nao * nao]),
                    &vmat.i((.., .., t, atm)).into_shape([nao * nao]),
                    -1,
                )
                .sum_all();
                de_chunk[[t, atm]] = v;
            }
        }
        let _lock = guard.lock().unwrap();
        unsafe { *&mut de_resp.force_mut() += &de_chunk };
    });
    de_resp
}

/// Device tensor (e.g. a `[3, natm]` gradient part or a `[nao, nao]` density) into a
/// `MatrixFull` of the same shape.
fn to_matrix_full(tsr: Tsr<f64>) -> MatrixFull<f64> {
    let shape: [usize; 2] = tsr.shape().to_vec().try_into().unwrap();
    MatrixFull::from_vec(shape, tsr.into_shape(-1).into_raw()).unwrap()
}

/// Fold a metric-derivative weight matrix for the eigen convention: the `L_1` term
/// `sum_weights . raw d(J^-1/2)` equals `<dJ, F(M S^-1)>` with `F` the (self-adjoint) Fréchet
/// derivative of `J^-1/2`, i.e. `V (D ∘ (V^T M^T V)) V^T` with
/// `D_ij = -1 / (sqrt(e_i) (sqrt(e_i) + sqrt(e_j)))`; the returned `G + G^T` pairs with one
/// atom block of the raw `int2c2e_ip1` (its transposed half included). Eigenpairs below
/// `threshold` are discarded, exactly as in the decomposition itself.
fn l1_fold_eig(weight: TsrView<f64>, j2c_e: TsrView<f64>, j2c_v: TsrView<f64>, threshold: f64) -> Tsr<f64> {
    let n = j2c_e.view().less(threshold).sum();
    let sr = j2c_e.i(n..).pow(0.5);
    let v = j2c_v.i((.., n..)).into_contig(ColMajor);
    let ssum = &sr.i((.., None)) + &sr.i((None, ..));
    let d: Tsr<f64> = (&sr.i((.., None)) * &ssum).mapv(|x: f64| -1.0 / x);
    let p = v.view().t() % (weight.t() % v.view()); // V^T M^T V
    let g: Tsr<f64> = v.view() % &(&d * &p) % v.view().t();
    (&g + &g.t()).into_owned()
}

/// One ip1 derivative-integral pass on an AO-shell batch, reduced against the pre-contracted
/// partner `Ȳ` (the metric solve is hoisted out of the atom loop). This is the `u` slot, where
/// the atom-side weight contracts the partner's leading axis (`v` slot:
/// [`add_fold_ip1_batched_v`]).
///
/// Computes `acc[t] += factor sum_{u, Q, y} G[(u Q), y](t) K[(u Q), y]`, where the derivative
/// fold `G[(u Q), y](t) = sum_v ip1_t2[v, (u Q t)] c_col[v, y]` and the partner fold
/// `K[(u Q), y] = sum_x c_row[u, x] ybar[x, Q, y]` share the same contiguous `(u Q)` row order,
/// so the reduction is a last-axis [`rt::vecdot`].
fn add_fold_ip1_batched(
    ip1_t2: TsrView<f64>, // [naov, nub naux ncomp]
    nub: usize,
    c_row: TsrView<f64>,  // [nub, x], the atom-side MO weight
    c_col: TsrView<f64>,  // [naov, y], the full-AO-side MO weight
    ybar: TsrView<f64>,   // [x, naux, y], pre-contracted partner
    factor: f64,
    mut acc: [f64; 3],
) -> [f64; 3] {
    let device = ip1_t2.device().clone();
    let naov = ip1_t2.shape()[0];
    let naux = ip1_t2.shape()[1] / (nub * 3);
    let x = c_row.shape()[1];
    let y = c_col.shape()[1];
    let mut k_buf: Tsr<f64> = rt::zeros(([nub, naux * y], &device));
    k_buf.view_mut().matmul_from(&c_row, &ybar.reshape([x, naux * y]), 1.0, 0.0);
    let k_buf_view = k_buf.view();
    let k2 = k_buf_view.reshape([nub * naux, y]);
    let mut g_buf: Tsr<f64> = rt::zeros(([nub * naux, y], &device));
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
/// stored.
fn add_fold_ip1_batched_v(
    ip1_t2: TsrView<f64>, // [naov, nub naux ncomp]
    nub: usize,
    c_row: TsrView<f64>,  // [nub, x], the atom-side MO weight (contracted partner axis)
    c_col: TsrView<f64>,  // [naov, y], the full-AO-side MO weight (partner's leading axis)
    ybar: TsrView<f64>,   // [y, naux, x], pre-contracted partner in the single kept layout
    factor: f64,
    mut acc: [f64; 3],
) -> [f64; 3] {
    let device = ip1_t2.device().clone();
    let naov = ip1_t2.shape()[0];
    let naux = ip1_t2.shape()[1] / (nub * 3);
    let x = c_row.shape()[1];
    let y = c_col.shape()[1];
    let mut k_buf: Tsr<f64> = rt::zeros(([y * naux, nub], &device));
    k_buf.view_mut().matmul_from(&ybar.reshape([y * naux, x]), &c_row.t(), 1.0, 0.0);
    let mut g_buf: Tsr<f64> = rt::zeros(([y, naux, nub], &device));
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

impl GradAPI for RDHGradient<'_> {
    fn get_gradient(&self) -> MatrixFull<f64> {
        self.result["de"].clone()
    }
    fn get_energy(&self) -> f64 {
        self.energy
    }
}

/// Whether the SCF-iteration functional carries a non-HF XC part (the gate of the
/// response-density XC term; the kernels handle LDA/GGA/MGGA alike, unlike forge's GGA-only gate).
fn has_scf_xc(scf_data: &SCF) -> bool {
    !scf_data.mol.xc_data.dfa_compnt_scf.is_empty()
}

/// A bare gradient helper carrying the SCF control flags, for reuse of the gradient kernels
/// (`calc_de_jk` with custom J/K factors, `get_vxc_rayon_new`).
fn dh_grad_helper<'a>(scf_data: &'a SCF, mpi_operator: &'a Option<MPIOperator>) -> RIRHFGradient<'a> {
    let mol_obj = &scf_data.mol;
    let flags = RIHFGradientFlagsBuilder::default()
        .print_level(mol_obj.ctrl.print_level)
        .max_memory(mol_obj.ctrl.max_memory)
        .auxbasis_response(mol_obj.ctrl.auxbasis_response)
        .factor_j(None)
        .factor_k(None)
        .build()
        .unwrap();
    RIRHFGradient { scf_data, flags, mpi_operator, result: HashMap::new() }
}

/// AO-derivative XC potential contraction `dao_vxc[nao, 3]` of one functional (the row-scatter,
/// contracted with the SCF density and the two-half factor, is its skeleton-gradient
/// contribution); `scf` selects the SCF-iteration functional, else the final (`pos`) one.
fn xc_skeleton_dao(scf_data: &SCF, mpi_operator: &Option<MPIOperator>, scf: bool, grids: &crate::dft::Grids) -> Tsr<f64> {
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
    get_vxc_rayon_new(&dh_grad_helper(scf_data, mpi_operator), &xc_data_sel, &mut grids_clone, mol_obj, 16)
}
