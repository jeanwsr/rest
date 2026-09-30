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
//! Every in-loop metric solve of the old per-atom scheme is replaced by an exact reordering:
//! the persistent MO partners are pre-contracted once as `Ȳ = partner % S` (`S` = the solve
//! operator that generated `rimatr`, obtained from `get_j2c_decomp(ctrl.j2c_decomp)`), and each
//! shell batch reduces directly (last-axis `rt::vecdot`) against `Ȳ`, never materializing
//! `[naux, 3, x, y]` intermediates. The metric-derivative (`L_1`) term, however, follows forge's
//! Cholesky-generator identity and therefore still requires the `Cd` convention (same as the
//! rimatr itself), so a matching `policy` is asserted below.

use crate::analdrv::config::AnalDrvConfig;
use crate::analdrv::response::rresp_interface::{rscf_resp_interface, RRespSCF};
use crate::analdrv::response::trait_rgfock::{GFockFlags, RGFockAPI};
use crate::analdrv::response::trait_rresp::RRespAPI;
use crate::grad::traits::GradAPI;
use crate::mpi_io::MPIOperator;
use crate::ri_jk::util;
use crate::ri_jk::{get_j2c_decomp, J2CDecompose, J2C_THRESH};
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
        drop(pt2);

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
        // strictly-lower mask with 1/2 diagonal, of forge `generator_L_1` (Cd only)
        let (l_lower, l_mask) = match &j2c_decomp {
            J2CDecompose::Cd { j2c_l, .. } => {
                let l_lower = j2c_l.t().into_contig(ColMajor);
                let mut l_mask: Tsr<f64> = rt::zeros(([naux, naux], &device));
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

        // ── persistent MO intermediates; density dots of the rimatr by two GEMVs ──
        let mem_avail = mol_obj.ctrl.max_memory.map(|m| m - detect_used_memory_mb("proc"));
        let abort = mol_obj.ctrl.abort_on_mem_exceed;
        handle_memory_exceed((nocc * nmo * naux + nvir * nocc * naux) as f64 * 8.0 / 1048576.0, mem_avail, abort);
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

        // ── pre-contractions `Ȳ = partner % S` (replace all in-loop metric solves) ──
        let ybar_y2: Tsr<f64> = y_ip.view().into_shape([nocc * nmo, naux]) % s_op.view(); // [(i m), Q]
        let ybar_g2i: Tsr<f64> = g_vix.view().transpose([1, 0, 2]).into_contig(ColMajor).view().into_shape([nocc * nvir, naux]) % s_op.view(); // [(i a), Q]
        // layout copies `[x, Q, y]` for the ip1 shell-batch reductions
        let ybar_jk_u = ybar_y2.view().into_shape([nocc, nmo, naux]).transpose([0, 2, 1]).into_contig(ColMajor); // [i, Q, m]
        let ybar_jk_v = y_ip.view().transpose([1, 0, 2]).into_contig(ColMajor).view().into_shape([nmo * nocc, naux]) % s_op.view(); // [(m i), Q]
        let ybar_jk_v = ybar_jk_v.view().into_shape([nmo, nocc, naux]).transpose([0, 2, 1]).into_contig(ColMajor); // [m, Q, i]
        let ybar_ri_u = ybar_g2i.view().into_shape([nocc, nvir, naux]).transpose([0, 2, 1]).into_contig(ColMajor); // [i, Q, a]
        let ybar_ri_v = g_vix.view().into_shape([nvir * nocc, naux]) % s_op.view(); // [(a i), Q]
        let ybar_ri_v = ybar_ri_v.view().into_shape([nvir, nocc, naux]).transpose([0, 2, 1]).into_contig(ColMajor); // [a, Q, i]
        drop(y_ip);
        drop(g_vix);

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
                t4_3c = add_fold_ip1_batched(ip1_t2.view(), nub, c0.i((r.clone(), ..)), c1_dr.view(), ybar_jk_u.view(), 2.0, t4_3c);
                t4_3c = add_fold_ip1_batched(ip1_t2.view(), nub, c1_dr.i((r.clone(), ..)), c0.view(), ybar_jk_v.view(), 2.0, t4_3c);
                ri_3c = add_fold_ip1_batched(ip1_t2.view(), nub, c0.i((r.clone(), ..)), cv.view(), ybar_ri_u.view(), -4.0, ri_3c);
                ri_3c = add_fold_ip1_batched(ip1_t2.view(), nub, cv.i((r, ..)), c0.view(), ybar_ri_v.view(), -4.0, ri_3c);
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
            // col-major so the leading-(i m)/(i a) reshape pairs with the `Ȳ` rows
            let f_jk = (w1.view() % c1_dr.view()).into_contig(ColMajor); // [i, m, (q t)]
            let f_ri = (w1.view() % cv.view()).into_contig(ColMajor); // [i, a, (q t)]
            let f_jk2 = f_jk.view().into_shape([nocc * nmo, nb_q * ncomp]);
            let f_ri2 = f_ri.view().into_shape([nocc * nvir, nb_q * ncomp]);
            for t in 0..ncomp {
                // `r_jk_q[q] = sum_{i m} F_jk[(i m), (q t)] Ȳ[(i m), q]` — contracted over (i m)
                let r_jk_q = rt::vecdot(&f_jk2.i((.., t * nb_q..(t + 1) * nb_q)), ybar_y2.i((.., qa..qb)), 0);
                let r_ri_q = rt::vecdot(&f_ri2.i((.., t * nb_q..(t + 1) * nb_q)), ybar_g2i.i((.., qa..qb)), 0);
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
            // forge L_1 generator of this atom, contracted with the solve operator
            for t in 0..ncomp {
                let m0 = &s_op_cols % tsr_int2c2e_ip1.i((auq0..auq1, .., t));
                let mut m: Tsr<f64> = &m0 % s_op.view().t();
                let m_t = m.view().t().into_owned();
                *&mut m += &m_t;
                let l1_t = (l_lower.view() % &(&l_mask * &m)).mapv(|x: f64| -x);
                let l1di = s_op.view() % l1_t.view();
                ydd_l1.i_mut((.., t)).assign(&(l1di.view() % y_dot_d.view()));
                ydr_l1.i_mut((.., t)).assign(&(l1di.view() % y_dot_dr.view()));
                t4_l1[t] = 2.0 * rt::vecdot(&l1di, r_jk.view(), -1).sum_all();
                ri_l1[t] = -4.0 * rt::vecdot(&l1di, r_ri.view(), -1).sum_all();
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

/// One ip1 derivative-integral pass on an AO-shell batch, reduced against the pre-contracted
/// partner `Ȳ` (the metric solve is hoisted out of the atom loop).
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

impl GradAPI for RDHGradient<'_> {
    fn get_gradient(&self) -> MatrixFull<f64> {
        self.result["de"].clone()
    }
    fn get_energy(&self) -> f64 {
        self.energy
    }
}
