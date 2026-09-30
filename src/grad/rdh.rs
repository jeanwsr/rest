//! Restricted doubly-hybrid (`xc = "mp2"`) analytic gradient: the PT2-correlation part.
//!
//! The SCF (HF) part of the gradient is provided by the existing `"SCF"` entry of
//! [`crate::main_driver::eval_force`]; this entry covers the D_r-carrying (correlation)
//! remainder, following the D_r-dependent blocks of pyscf-forge `dh/grad/dfdh.py`:
//!
//! - `de_h`: hcore-derivative term `<H_1(A), D_r>` (forge `_get_gradient_pt2`, first term);
//! - `de_s1`: overlap-Pulay term `<W, S_1(A)>` with `W = W_I + W_II + W_III` assembled from the
//!   full four-block PT2 generalized Fock, the orbital-energy terms and the SCF response upon
//!   the relaxed correlation density (forge `_get_gradient_pt2`, second term);
//! - `de_jk`: `t1 + t2 + t4` of forge `get_gradient_jk`, restricted to the D_r-carrying weights
//!   (`C1 = C D_r^S`; the `0.5 C D_mo` exchange part belongs to the `"SCF"` entry);
//! - `de_rint`: `4 <G_ia, Y_1_ia(A)>` (forge `_get_gradient_pt2`, third term).
//!
//! The rimatr must be CD-decomposed (`[ctrl.j2c_decomp] policy = "cd"`), matching the
//! pyscf-forge (and pyscf `df`) convention `cderi[P, uv] = (L^-1 raw3c)[P, uv]`; every
//! derivative contraction left-solves against the same lower-Cholesky factor.

use crate::analdrv::config::AnalDrvConfig;
use crate::analdrv::response::rresp_interface::{rscf_resp_interface, RRespSCF};
use crate::analdrv::response::trait_rgfock::{GFockFlags, RGFockAPI};
use crate::analdrv::response::trait_rresp::RRespAPI;
use crate::grad::traits::GradAPI;
use crate::mpi_io::MPIOperator;
use crate::ri_jk::pure_decompose::solve_by_j2c_mut;
use crate::ri_jk::util;
use crate::ri_jk::{get_j2c_decomp, J2CDecompOption, J2CDecompPolicy, J2CDecompose, J2C_THRESH};
use crate::ri_pt2::rgfock_pt2::RGFockPT2;
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
}

impl<'a> RDHGradient<'a> {
    pub fn new(scf_data: &'a SCF, mpi_operator: &'a Option<MPIOperator>) -> Self {
        Self { scf_data, mpi_operator, result: HashMap::new(), energy: 0.0 }
    }

    pub fn calc(&mut self) -> &mut Self {
        let scf_data = self.scf_data;
        let mol_obj = &scf_data.mol;
        let device = DeviceBLAS::default();
        assert!(!mol_obj.ctrl.spin_polarization, "RDHGradient supports only restricted (spin-polarization = false) references.");
        assert!(mol_obj.start_mo == 0, "RDHGradient does not support frozen core (start_mo).");
        assert!(self.mpi_operator.is_none(), "RDHGradient is not MPI-parallelized yet.");
        if mol_obj.ctrl.j2c_decomp.policy != J2CDecompPolicy::Cd {
            panic!(
                "Doubly-hybrid analytic gradient requires the CD-decomposed rimatr. Please set\n[ctrl.j2c_decomp]\npolicy = \"cd\"\nin the ctrl input."
            );
        }
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
        let config = AnalDrvConfig::default();
        let mut resp_objs: RRespSCF = rscf_resp_interface(scf_data, &config);
        let [c_os, c_ss]: [f64; 2] = mol_obj
            .xc_data
            .dfa_paramr_adv
            .clone()
            .expect("Doubly-hybrid gradient requires the PT2 spin factors (dfa_paramr_adv).")
            .try_into()
            .expect("dfa_paramr_adv must have exactly two entries.");
        let j3c = rimatr.0.to_rstsr_view(&device);
        let mut pt2 = RGFockPT2::<f64>::new(
            mo_coeff.clone(),
            mo_occ,
            mo_energy.clone(),
            j3c.into_cow(),
            None,
            vec![0, nocc],
            c_os,
            c_ss,
        );
        pt2.dump_g_vix = true;
        resp_objs.make_cpscf_preparation(mo_coeff.view(), pt2.mo_occ.view(), pt2.mo_energy.view());
        // full four-block request (also caches `g_vix` and the SCF response upon rdm1_corr),
        // then the relaxed density through the DH-level Z-vector
        let _ = pt2.make_gfock(Some(&mut resp_objs), GFockFlags::OO | GFockFlags::VV);
        let _ = pt2.make_rdm1_resp(&mut resp_objs);
        let rdm1_resp = pt2.result["rdm1_resp"].to_owned();
        let gfock_part = pt2.result["gfock_part_full"].to_owned();
        let rdm1 = pt2.result["rdm1"].to_owned();
        let g_vix = pt2.result["g_vix"].to_owned(); // [nvir, nocc, naux]
        self.energy = pt2.e_corr.unwrap_or(0.0);

        let d_r_sym = (&rdm1_resp + &rdm1_resp.t()).into_contig(ColMajor);
        let d_r_s: Tsr<f64> = d_r_sym.mapv(|x| 0.5 * x);
        let c1_dr = (mo_coeff.view() % d_r_s.view()).into_contig(ColMajor);
        let d_r_ao = (mo_coeff.view() % d_r_s.view() % mo_coeff.view().t()).into_contig(ColMajor);

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
        {
            let mut hcore_gen = crate::grad::rhf::generator_deriv_hcore(scf_data);
            let tsr_ipovlp = {
                let (out, shape) = mol.integrate("int1e_ipovlp", "s1", None).into();
                rt::asarray((out, shape, &device))
            };
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

        // ── CD metric objects ──
        let j2c_decomp = get_j2c_decomp(
            &aux,
            &device,
            J2CDecompOption { policy: J2CDecompPolicy::Cd, threshold: Some(J2C_THRESH), ..Default::default() },
        );
        let j2c_l = match &j2c_decomp {
            J2CDecompose::Cd { j2c_l, uplo: Upper, .. } => j2c_l.clone(),
            _ => unreachable!("CD policy checked above"),
        };
        // U^-1 solves U X = I (j2c = U^T U); forge's L_inv = inv(L_lower) = (U^-1)^T
        let l_inv = {
            let eye = rt::eye((naux, &device));
            let u_inv = rt::linalg::solve_triangular((j2c_l.view(), eye.view(), Upper));
            u_inv.t().into_contig(ColMajor)
        };
        let l_lower = j2c_l.t().into_contig(ColMajor);
        let tsr_int2c2e_ip1 = {
            let (out, shape) = aux.integrate("int2c2e_ip1", "s1", None).into();
            rt::asarray((out, shape, &device))
        };
        // strictly-lower mask with 1/2 diagonal, of forge `generator_L_1`
        let mut l_mask: Tsr<f64> = rt::zeros(([naux, naux], &device));
        for i in 0..naux {
            for j in 0..i {
                l_mask[[i, j]] = 1.0;
            }
            l_mask[[i, i]] = 0.5;
        }

        // ── persistent MO intermediates (Y_ip, metric couplings R/R2); peak 2 * nocc nao naux ──
        let mem_avail = mol_obj.ctrl.max_memory.map(|m| m - detect_used_memory_mb("proc"));
        let abort = mol_obj.ctrl.abort_on_mem_exceed;
        handle_memory_exceed((nocc * nmo * naux + nvir * nocc * naux) as f64 * 8.0 / 1048576.0, mem_avail, abort);
        let mut y_ip: Tsr<f64> = rt::zeros(([nocc, nmo, naux], &device));
        let mut y_dot_d: Tsr<f64> = rt::zeros(([naux], &device));
        let mut y_dot_dr: Tsr<f64> = rt::zeros(([naux], &device));
        let mut r_jk: Tsr<f64> = rt::zeros(([naux, naux], &device));
        let mut r_ri: Tsr<f64> = rt::zeros(([naux, naux], &device));
        let g2d = g_vix.view().into_shape([nvir * nocc, naux]); // (a i)-packed rows, P columns
        for p in 0..naux {
            let col = ederi_utp.i((.., p)).unpack_tri(Upper, FlagSymm::Sy);
            y_ip.i_mut((.., .., p)).assign(&(c0.view().t() % &col % mo_coeff.view()));
            let s_d: f64 = (&col * d_hf.view()).sum_all();
            y_dot_d[[p]] = s_d;
            let s_dr: f64 = (&col * d_r_ao.view()).sum_all();
            y_dot_dr[[p]] = s_dr;
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

        // ── per-atom RI-derivative terms ──
        let mol_loc = mol.ao_loc();
        let aux_loc = aux.ao_loc();
        let mol_ao_slice = mol_obj.aoslice_by_atom();
        let aux_slice = mol_obj.make_auxmol_fake().aoslice_by_atom();
        let nbas = mol.nbas();
        let nauxbas = aux.nbas();
        let shell_batch = calc_batch_size::<f64>(3 * nao * naux, mem_avail, None, None).max(1);
        let aux_batch = calc_batch_size::<f64>(3 * nao_tp, mem_avail, None, None).max(1);
        // MO-fold batches: each solved fold `[naux, 3, x_b, y]` stays below the batching budget
        let xbatch_jk = calc_batch_size::<f64>(6 * naux * nocc, mem_avail, None, None).max(1);
        let xbatch_ri = calc_batch_size::<f64>(6 * naux * nocc, mem_avail, None, None).max(1);
        handle_memory_exceed(3.0 * (nao * naux) as f64 * 8.0 / 1048576.0, mem_avail, abort);

        let mut de_jk: Tsr<f64> = rt::zeros(([3, natm], &device));
        let mut de_rint: Tsr<f64> = rt::zeros(([3, natm], &device));
        for atm in 0..natm {
            let [shl_a0, shl_a1, au0, au1] = mol_ao_slice[atm];
            let [shl_q0, shl_q1, auq0, auq1] = aux_slice[atm];
            let c0_atom = c0.i((au0..au1, ..)).into_contig(ColMajor);
            let cv_atom = cv.i((au0..au1, ..)).into_contig(ColMajor);
            let c1_atom = c1_dr.i((au0..au1, ..)).into_contig(ColMajor);
            let l_inv_cols = l_inv.i((.., auq0..auq1)).into_contig(ColMajor); // [naux, nb_q]

            // forge L_1 generator of this atom, contracted with L_inv
            let mut l1di = Vec::with_capacity(3);
            for t in 0..3 {
                let m0 = l_inv.i((.., auq0..auq1)) % tsr_int2c2e_ip1.i((auq0..auq1, .., t));
                let mut m: Tsr<f64> = m0 % l_inv.view().t();
                let m_t = m.view().t().into_owned();
                *&mut m += &m_t;
                let l1_t = (l_lower.view() % (&l_mask * &m)).mapv(|x: f64| -x);
                l1di.push(l_inv.view() % l1_t.view());
            }

            // AO-shell batches of atom A: density vectors and MO folds of int3c2e_ip1
            let mut ydd_raw: Tsr<f64> = rt::zeros(([naux, 3], &device));
            let mut ydr_raw: Tsr<f64> = rt::zeros(([naux, 3], &device));
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
                }; // [nub, nao, naux, 3]
                let nub = ip1.shape()[0];
                for u in 0..nub {
                    let ip1_u = ip1.i((u, .., .., ..)); // [nao, naux, 3]
                    *&mut ydd_raw += &(&ip1_u * d_hf.i((au0 + u, ..)).i((.., None, None))).sum_axes(0);
                    *&mut ydr_raw += &(&ip1_u * d_r_ao.i((au0 + u, ..)).i((.., None, None))).sum_axes(0);
                }
                // u-slot (C0 on the atom rows) and v-slot (C1_dr / C0 swapped), t4 then RI term
                t4_3c = add_fold_ip1(ip1.view(), c0_atom.view(), c1_dr.view(), xbatch_jk, &j2c_decomp, y_ip.view(), None, 2.0, t4_3c);
                t4_3c = add_fold_ip1(ip1.view(), c1_atom.view(), c0.view(), xbatch_jk, &j2c_decomp, y_ip.view(), Some([1, 0]), 2.0, t4_3c);
                ri_3c = add_fold_ip1(ip1.view(), c0_atom.view(), cv.view(), xbatch_ri, &j2c_decomp, g_vix.view(), Some([1, 0]), -4.0, ri_3c);
                ri_3c = add_fold_ip1(ip1.view(), cv_atom.view(), c0.view(), xbatch_ri, &j2c_decomp, g_vix.view(), None, -4.0, ri_3c);
            }

            // aux-shell batches of atom A: density vectors and MO folds of int3c2e_ip2
            let mut wdd: Tsr<f64> = rt::zeros(([auq1 - auq0, 3], &device));
            let mut wdr: Tsr<f64> = rt::zeros(([auq1 - auq0, 3], &device));
            let d_hf_tp = crate::grad::rhf::pack_triu_tilde(d_hf.view());
            let d_r_ao_tp = crate::grad::rhf::pack_triu_tilde(d_r_ao.view());
            let q_batches = blocksize_partition(&aux_loc[shl_q0..=shl_q1], aux_batch)
                .into_iter()
                .map(|[i0, i1]| [shl_q0 + i0, shl_q0 + i1]);
            for [q0, q1] in q_batches {
                let ip2 = {
                    let shl_slices = [[0, nbas], [0, nbas], [q0, q1]];
                    let (out, shape) = CInt::integrate_cross("int3c2e_ip2", [&mol, &mol, &aux], "s2ij", shl_slices).into();
                    rt::asarray((out, shape, &device))
                }; // [nao_tp, nb_q, 3]
                *&mut wdd += &(&d_hf_tp.i((.., None, None)) * ip2.view()).sum_axes(0);
                *&mut wdr += &(&d_r_ao_tp.i((.., None, None)) * ip2.view()).sum_axes(0);
                let folded_jk = fold_ip2(ip2.view(), c0.view(), c1_dr.view(), 0..nmo); // [3, nb_q, nocc, nmo]
                let solved_jk = metric_aux_atom(l_inv_cols.view(), folded_jk.view());
                t4_3c = add_contract(solved_jk.view(), y_ip.view(), None, 2.0, t4_3c);
                let folded_ri = fold_ip2(ip2.view(), c0.view(), cv.view(), 0..nvir); // [3, nb_q, nocc, nvir]
                let solved_ri = metric_aux_atom(l_inv_cols.view(), folded_ri.view());
                ri_3c = add_contract(solved_ri.view(), g_vix.view(), Some([1, 0]), -4.0, ri_3c);
            }
            let ydd2: Tsr<f64> = l_inv_cols.view() % wdd.view(); // [naux, 3]
            let ydr2: Tsr<f64> = l_inv_cols.view() % wdr.view();
            let ydd_sol: Tsr<f64> = l_inv.view() % ydd_raw.view();
            let ydr_sol: Tsr<f64> = l_inv.view() % ydr_raw.view();

            // forge L_1 contributions to the Y_1 vectors and to t4 / the RI term
            let mut ydd_l1: Tsr<f64> = rt::zeros(([naux, 3], &device));
            let mut ydr_l1: Tsr<f64> = rt::zeros(([naux, 3], &device));
            let mut t4_l1 = [0.0f64; 3];
            let mut ri_l1 = [0.0f64; 3];
            for t in 0..3 {
                ydd_l1.i_mut((.., t)).assign(&(l1di[t].view() % y_dot_d.view()));
                ydr_l1.i_mut((.., t)).assign(&(l1di[t].view() % y_dot_dr.view()));
                t4_l1[t] = 2.0 * (&l1di[t] * r_jk.view()).sum_all() as f64;
                ri_l1[t] = -4.0 * (&l1di[t] * r_ri.view()).sum_all() as f64;
            }

            // assemble forge t1 + t2 + t4 (D_r-carrying) and the RI term
            let mut grad = [0.0f64; 3];
            let mut grad_ri = [0.0f64; 3];
            for t in 0..3 {
                let v_ydd = ydd_sol.i((.., t));
                let v_ydr = ydr_sol.i((.., t));
                let c_ydd = ydd2.i((.., t)) + ydd_l1.i((.., t));
                let c_ydr = 2.0 * &v_ydr + &ydr2.i((.., t)) + &ydr_l1.i((.., t));
                let s1: f64 = (&y_dot_d * &c_ydr).sum_all();
                let s2: f64 = (&y_dot_dr * &(2.0 * &v_ydd + &c_ydd)).sum_all();
                grad[t] = -s1 - s2 + t4_3c[t] + t4_l1[t];
                grad_ri[t] = ri_3c[t] + ri_l1[t];
            }
            de_jk.i_mut((.., atm)).assign(&rt::asarray((grad.as_slice(), [3], &device)));
            de_rint.i_mut((.., atm)).assign(&rt::asarray((grad_ri.as_slice(), [3], &device)));
        }

        // ── assemble the parts ──
        let de = de_h.view() + de_s1.view() + de_jk.view() + de_rint.view();
        for (key, val) in [("de_h", de_h), ("de_s1", de_s1), ("de_jk", de_jk), ("de_rint", de_rint), ("de", de.into_owned())] {
            self.result.insert(key.to_string(), to_matrix_full(val));
        }
        self
    }
}

/// `[3, natm]` device tensor into a `MatrixFull` of the same shape.
fn to_matrix_full(tsr: Tsr<f64>) -> MatrixFull<f64> {
    let shape: [usize; 2] = tsr.shape().to_vec().try_into().unwrap();
    MatrixFull::from_vec(shape, tsr.into_shape(-1).into_raw()).unwrap()
}

/// Contract the atom-row axis `u` of a raw `ip1` batch `[nub, nao, naux, 3]` with `w_row`
/// (atom-A rows of an MO matrix, giving index `x`) and the full AO axis with `w_full` (giving
/// index `y`), left-solve against the CD metric, and accumulate
/// `acc[t] += factor <partner[x, y, P], folded[P, t, x, y]>`.
/// `swap` transposes the (x, y) axes of the partner when it is stored as `[y, x, naux]`.
#[allow(clippy::too_many_arguments)]
fn add_fold_ip1(
    ip1: TsrView<f64>,
    w_row: TsrView<f64>,
    w_full: TsrView<f64>,
    xbatch: usize,
    j2c_decomp: &J2CDecompose,
    partner: TsrView<f64>,
    swap: Option<[usize; 2]>,
    factor: f64,
    mut acc: [f64; 3],
) -> [f64; 3] {
    let device = ip1.device().clone();
    let nub = ip1.shape()[0];
    let naov = ip1.shape()[1];
    let naux = ip1.shape()[2];
    let ncomp = ip1.shape()[3];
    let y = w_full.shape()[1];
    let x = w_row.shape()[1];
    let mut stacked: Tsr<f64> = rt::zeros(([nub, y, naux * ncomp], &device));
    for u in 0..nub {
        let ip1_u = ip1.i((u, .., .., ..)).into_contig(ColMajor);
        let ip1_u = ip1_u.reshape([naov, naux * ncomp]); // (v, P t)
        stacked.i_mut((u, .., ..)).assign(&(w_full.t() % ip1_u.view())); // [y, naux 3]
    }
    let xbatch = xbatch.min(x).max(1);
    for x0 in (0..x).step_by(xbatch) {
        let x1 = (x0 + xbatch).min(x);
        let folded = w_row.i((.., x0..x1)).t() % stacked.view().reshape([nub, y * naux * ncomp]); // [x, (y P t)]
        // [x, y, naux, 3] -> [naux, 3, x, y], then left-solve by L^-1 against the CD metric
        let mut solved = folded.into_shape([x1 - x0, y, naux, ncomp]).transpose([2, 3, 0, 1]).into_contig(ColMajor);
        {
            let mut blk = rt::asarray((solved.raw_mut(), [naux, ncomp * (x1 - x0) * y].f(), &device));
            solve_by_j2c_mut(blk.view_mut(), j2c_decomp, FlagSide::L, true);
        }
        acc = add_contract(solved.view(), partner.view(), swap, factor, acc);
    }
    acc
}

/// `acc[t] += factor sum_P <partner[:, :, P], folded[P, t, :, :]>`.
fn add_contract(folded: TsrView<f64>, partner: TsrView<f64>, swap: Option<[usize; 2]>, factor: f64, mut acc: [f64; 3]) -> [f64; 3] {
    let naux = folded.shape()[0];
    let ncomp = folded.shape()[1];
    let partner = match swap {
        None => partner,
        Some([a, b]) => partner.transpose([a, b, 2]),
    };
    for t in 0..ncomp {
        let mut s = 0.0;
        for p in 0..naux {
            s += (&folded.i((p, t, .., ..)) * partner.i((.., .., p))).sum_all();
        }
        acc[t] += factor * s;
    }
    acc
}

/// Fold a raw `ip2` batch `[nao_tp, nb_q, 3]` column-by-column with the MO pair transform
/// `w_left^T col w_right[:, yr]`; returns `[3, nb_q, x, y_len]`.
fn fold_ip2(ip2: TsrView<f64>, w_left: TsrView<f64>, w_right: TsrView<f64>, yr: std::ops::Range<usize>) -> Tsr<f64> {
    let nb_q = ip2.shape()[1];
    let ncomp = ip2.shape()[2];
    let x = w_left.shape()[1];
    let device = ip2.device().clone();
    let mut out = rt::zeros(([ncomp, nb_q, x, yr.len()], &device));
    for q in 0..nb_q {
        for t in 0..ncomp {
            let col = ip2.i((.., q, t)).unpack_tri(Upper, FlagSymm::Sy);
            out.i_mut((t, q, .., ..)).assign(&(w_left.t() % &col % w_right.i((.., yr.clone()))));
        }
    }
    out
}

/// `sum_{Q in atom} L_inv[P, Q] folded[Q, t, x, y]`; `[3, nb_q, x, y] -> [naux, 3, x, y]`.
fn metric_aux_atom(l_inv_cols: TsrView<f64>, folded: TsrView<f64>) -> Tsr<f64> {
    let shape = folded.shape();
    let [ncomp, nb_q, x, y] = [shape[0], shape[1], shape[2], shape[3]];
    let naux = l_inv_cols.shape()[0];
    let folded_t = folded.transpose([1, 0, 2, 3]).into_contig(ColMajor); // [nb_q, 3, x, y]
    (l_inv_cols % folded_t.view().reshape([nb_q, ncomp * x * y])).into_shape([naux, ncomp, x, y])
}

impl GradAPI for RDHGradient<'_> {
    fn get_gradient(&self) -> MatrixFull<f64> {
        self.result["de"].clone()
    }
    fn get_energy(&self) -> f64 {
        self.energy
    }
}
