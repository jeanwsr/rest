//! Pure functions for the unrestricted RI-PT2 electronic derivative (generalized Fock and
//! related quantities), the unrestricted twin of
//! [`crate::ri_pt2::pure_pt2_r_elecderiv`].
//!
//! Spin blocks follow pyscf-forge `dh/resp.py::_prepare_pt2_u` (per spin $\sigma$, with the
//! opposite spin labelled $\varsigma$):
//!
//! - amplitudes $t^{\sigma\varsigma}_{ijab} = (i a_\sigma | j b_\varsigma) / \Delta$, with the
//!   biorthogonal combinations $T^{\sigma\sigma} = \frac{1}{2} c_{ss} (t - t^{\top})$ (the
//!   transpose over the virtual pair only; the compound $(i,a) \leftrightarrow (j,b)$
//!   antisymmetry then holds through the amplitude pair symmetry, cf. forge
//!   `hermi_sum_last2dim`) and $T^{\alpha\beta} = c_{os} t$ (single-counted);
//! - correlation rdm1 per spin, $D^{\sigma}_{oo} = -2 \Sigma T^{\sigma\sigma} t^{\sigma\sigma}
//!   - \Sigma T^{\alpha\beta} t^{\alpha\beta}$ (and $+ / \varsigma$-transposed contracts for the
//!   virtual block);
//! - the amplitude intermediate $G^\sigma_{ia}[\mathtt{P}] = 4 \Sigma T^{\sigma\sigma}
//!   (j b_\sigma | \mathtt{P}) + 2 \Sigma T^{\alpha\beta} (j b_\varsigma | \mathtt{P})$;
//! - the (full four-block) generalized Fock contribution with coefficient $1$ on the
//!   $G$-contractions (the restricted kernel scales the same blocks by $4$; with the
//!   $G$ normalization above the two conventions give identical $W$ blocks), plus the
//!   $2 \varepsilon \cdot \mathrm{rdm1}$ orbital-energy terms of the diagonal blocks;
//! - the correlation energy $\Sigma T g$ over all three spin blocks, i.e.
//!   $c_{os} E_{os} + c_{ss} E_{ss}$ already in the coefficient-weighted form.
//!
//! Every block streams its amplitudes over windows ([`UPT2ElecDerivIncoreInp::index_occ_outer_vec`]
//! / [`UPT2ElecDerivIncoreInp::index_vir_outer_vec`], cf.
//! [`occ_batch_index`](crate::ri_pt2::occ_batch_index)): only window-sized buffers exist
//! transiently and are discarded after their rdm1/$G$ accumulation, so no `nocc^2 nvir^2`
//! amplitude tensor is ever materialized. The same-spin blocks use the pair-symmetry structure of
//! the restricted kernel (triangular pair loop with the transpose fill) and window their outer
//! occupied index. The $\alpha\beta$ block stores only the plain amplitudes ($T = c_{os} t$
//! enters every contraction as a factor); each of its four rdm1 Grams must keep its own index
//! pair complete in the working set, and the four pair up as α-side / β-side, so the amplitudes
//! are streamed in two passes (α- then β-virtual windowed) — each a per-window partial-trace
//! GEMM pair, with no `[.]^2`-sized intermediate.

use crate::ri_jk::pure_ao2mo::get_ao2mo_s2ij_to_s1_notrans;
use crate::utilities::rstsr_util::{Tsr, TsrView};
use rayon::prelude::*;
use rstsr::prelude::*;
use std::sync::{Arc, Mutex};

pub struct UPT2ElecDerivIncoreInp<'a> {
    /// Cholesky-decomposed 3c2e ERI, shape `[nao_tp, naux]`.
    pub cderi: TsrView<'a>,
    /// Optionally pre-transformed (vir, occ, aux) three-center integrals per spin.
    pub cderi_vox: [Option<Tsr>; 2],
    /// Occupied MO coefficients, shape `[nao, nocc_s]` per spin.
    pub occ_coeff: [TsrView<'a>; 2],
    /// Virtual MO coefficients, shape `[nao, nvir_s]` per spin.
    pub vir_coeff: [TsrView<'a>; 2],
    /// Occupied orbital energies, shape `[nocc_s]` per spin.
    pub occ_energy: [TsrView<'a>; 2],
    /// Virtual orbital energies, shape `[nvir_s]` per spin.
    pub vir_energy: [TsrView<'a>; 2],
    /// Occupancy-window boundaries of the streamed same-spin blocks, per spin (cf.
    /// [`occ_batch_index`](crate::ri_pt2::occ_batch_index)): the σσ block of spin `s` streams its
    /// outer occupied index through `index_occ_outer_vec[s]`. Each vector starts at 0 and ends at
    /// the spin's `nocc`.
    pub index_occ_outer_vec: [Vec<usize>; 2],
    /// Virtual-window boundaries of the streamed opposite-spin block, per spin: entry `0` windows
    /// the α virtual index of the first (energy/$G$/β-side rdm1) pass, entry `1` the β virtual
    /// index of the second (α-side rdm1) pass. Each vector starts at 0 and ends at the spin's
    /// `nvir`.
    pub index_vir_outer_vec: [Vec<usize>; 2],
}

pub struct UPT2ElecDerivIncoreArg {
    /// Opposite-spin correlation factor $c_{os}$.
    pub c_os: f64,
    /// Same-spin correlation factor $c_{ss}$.
    pub c_ss: f64,
}

pub struct UPT2ElecDerivIncoreOut {
    /// PT2 correlation energy $c_{os} E_{os} + c_{ss} E_{ss}$.
    pub e_corr: f64,
    /// Unrelaxed correlation rdm1, shape `[nmo_s, nmo_s]` per spin.
    pub rdm1_corr: [Tsr; 2],
    /// Full four-block generalized Fock contribution, shape `[nmo_s, nmo_s]` per spin, with the
    /// orbital-energy terms of the diagonal blocks included.
    pub gfock_part: [Tsr; 2],
    /// Amplitude intermediate $G^\sigma$, shape `[nvir_s, nocc_s, naux]` per spin.
    pub g_vix: [Tsr; 2],
}

pub fn get_rupt2_elec_deriv_incore(
    input: &UPT2ElecDerivIncoreInp,
    arg: &UPT2ElecDerivIncoreArg,
) -> UPT2ElecDerivIncoreOut {
    let UPT2ElecDerivIncoreInp { cderi, cderi_vox, occ_coeff, vir_coeff, occ_energy, vir_energy, index_occ_outer_vec, index_vir_outer_vec } = input;
    let UPT2ElecDerivIncoreArg { c_os, c_ss } = arg;
    let (c_os, c_ss) = (*c_os, *c_ss);
    let device = cderi.device().clone();
    let naux = cderi.shape()[1];
    let nocc = [occ_coeff[0].shape()[1], occ_coeff[1].shape()[1]];
    let nvir = [vir_coeff[0].shape()[1], vir_coeff[1].shape()[1]];
    let nmo = [nocc[0] + nvir[0], nocc[1] + nvir[1]];
    for s in 0..2 {
        assert_eq!(index_occ_outer_vec[s].first(), Some(&0), "index_occ_outer_vec[{s}] must start with 0");
        assert_eq!(index_occ_outer_vec[s].last(), Some(&nocc[s]), "index_occ_outer_vec[{s}] must end with nocc");
        assert!(index_occ_outer_vec[s].is_sorted(), "index_occ_outer_vec[{s}] must be sorted");
    }
    for s in 0..2 {
        assert_eq!(index_vir_outer_vec[s].first(), Some(&0), "index_vir_outer_vec[{s}] must start with 0");
        assert_eq!(index_vir_outer_vec[s].last(), Some(&nvir[s]), "index_vir_outer_vec[{s}] must end with nvir");
        assert!(index_vir_outer_vec[s].is_sorted(), "index_vir_outer_vec[{s}] must be sorted");
    }

    let cderi_vox: [Tsr; 2] =
        [0, 1].map(|s| match &cderi_vox[s] {
            Some(x) => x.to_owned(),
            None => get_ao2mo_s2ij_to_s1_notrans(
                cderi.view(),
                Upper,
                &[vir_coeff[s].view()],
                &[occ_coeff[s].view()],
                |x| x,
            )
            .remove(0),
        });

    let e_corr = Arc::new(Mutex::new(0.0));
    let mut rdm1 = [
        rt::zeros(([nmo[0], nmo[0]].f(), &device)),
        rt::zeros(([nmo[1], nmo[1]].f(), &device)),
    ];
    let mut g_vix = [
        rt::zeros(([nvir[0], nocc[0], naux].f(), &device)),
        rt::zeros(([nvir[1], nocc[1], naux].f(), &device)),
    ];
    let d_vv_outer = [
        (-vir_energy[0].i((.., None)) - vir_energy[0].i((None, ..))).into_dim::<Ix2>(),
        (-vir_energy[1].i((.., None)) - vir_energy[1].i((None, ..))).into_dim::<Ix2>(),
    ];

    // ── same-spin blocks: pair-symmetrized amplitudes streamed over occupied windows ── //
    for s in 0..2 {
        if nocc[s] == 0 || nvir[s] == 0 {
            continue;
        }
        let (nv, no) = (nvir[s], nocc[s]);
        let so = 0..no;
        let sv = no..nmo[s];
        // biorthogonal coefficients: T = bi1 t - bi2 t^T with T = 1/2 c_ss (t - t^T)
        let (bi1, bi2) = (0.5 * c_ss, 0.5 * c_ss);
        for io_slice in index_occ_outer_vec[s].windows(2) {
            let nstep = io_slice[1] - io_slice[0];
            let mut t_vivo: Tsr = rt::zeros(([nv, nstep, nv, no].f(), &device));
            let mut T_vivo: Tsr = rt::zeros(([nv, nstep, nv, no].f(), &device));
            let mut g_vix_b: Tsr = rt::zeros(([nv, nstep, naux].f(), &device));
            // pairs (i, j): j <= i inside the window, all j outside it (each pair once)
            let mut pair_ij: Vec<[usize; 2]> = Vec::new();
            for i in io_slice[0]..io_slice[1] {
                let range_1 = io_slice[0]..=i;
                let range_2 = 0..io_slice[0];
                let range_3 = io_slice[1]..no;
                pair_ij.extend(range_1.chain(range_2).chain(range_3).map(|j| [i, j]));
            }
            let e_corr_s = e_corr.clone();
            pair_ij.into_par_iter().for_each(|[i, j]| {
                let d_ij = occ_energy[s][[i]] + occ_energy[s][[j]];
                let g_vv = cderi_vox[s].i((.., i, ..)) % cderi_vox[s].i((.., j, ..)).t();
                let d_vv = &d_vv_outer[s] + d_ij;
                let t_vv = &g_vv / &d_vv;
                let t_vv_ = T_vv_full(&t_vv, bi1, bi2);
                if j <= i {
                    let diag_scale = if i == j { 1.0 } else { 2.0 };
                    *e_corr_s.lock().unwrap() += diag_scale * (&t_vv_ * &g_vv).sum();
                }
                let mut t_vivo = unsafe { t_vivo.force_mut() };
                let mut t_vivo_T = unsafe { T_vivo.force_mut() };
                t_vivo.i_mut((.., i - io_slice[0], .., j)).assign(&t_vv);
                t_vivo_T.i_mut((.., i - io_slice[0], .., j)).assign(&t_vv_);
                if (io_slice[0] <= j) && (j < i) {
                    t_vivo.i_mut((.., j - io_slice[0], .., i)).assign(t_vv.t());
                    t_vivo_T.i_mut((.., j - io_slice[0], .., i)).assign(t_vv_.t());
                }
            });
            // D[oo] -= 2 sum T t; D[vv] += 2 sum T t (per-window partial sums over the i rows)
            let scr = t_vivo.view().reshape((-1, no)).t() % T_vivo.view().reshape((-1, no));
            *&mut rdm1[s].i_mut((so.clone(), so.clone())) -= &(scr * 2.0);
            let scr = t_vivo.view().reshape((nv, -1)) % T_vivo.view().reshape((nv, -1)).t();
            *&mut rdm1[s].i_mut((sv.clone(), sv.clone())) += &(scr * 2.0);
            // G^s[same-spin part] = 4 T[viv, o] % cderi_vox[vo, x]
            let mut g_vix_b_2d = rt::asarray((g_vix_b.raw_mut(), [nv * nstep, naux].f(), &device));
            g_vix_b_2d.matmul_from(
                &T_vivo.view().into_shape([nv * nstep, nv * no]),
                &cderi_vox[s].view().into_shape([nv * no, naux]),
                4.0,
                0.0,
            );
            *&mut g_vix[s].i_mut((.., io_slice[0]..io_slice[1], ..)) += &g_vix_b;
        }
    }

    // ── opposite-spin block, two streaming passes ── //
    // Each rdm1 piece is a Gram (outer product summed over the remaining three indices) whose
    // own index pair must be complete in the working set; the four Grams pair up as α-side
    // (α-occ / α-virt pairs) and β-side (β-occ / β-virt pairs), so one pass per side — each
    // windowing its own α respectively β virtual index — covers all four without a `nocc^2
    // nvir^2` tensor. The energy and both G intermediates ride the first pass.
    if nocc[0] > 0 && nocc[1] > 0 && nvir[0] > 0 && nvir[1] > 0 {
        let (nva, nvb, noa, nob) = (nvir[0], nvir[1], nocc[0], nocc[1]);
        let d_vv_ab =
            (-vir_energy[0].i((.., None)) - vir_energy[1].i((None, ..))).into_dim::<Ix2>();

        // --- pass 1: α-virtual-windowed, β transparent; energy, G^α/G^β, β-side Grams ---
        // permuted copy [i, a, P] keeps the `(i a)` row order mergeable for the `G^β` contraction
        let cderi_vox_0_iao =
            cderi_vox[0].view().into_shape([nva, noa, naux]).transpose([1, 0, 2]).into_contig(ColMajor);
        for io_slice in index_vir_outer_vec[0].windows(2) {
            let [a0, a1] = [io_slice[0], io_slice[1]];
            let nva_b = a1 - a0;
            let mut t_x: Tsr = rt::zeros(([nva_b, noa, nvb, nob].f(), &device));
            let pair_ik: Vec<[usize; 2]> =
                (0..noa).flat_map(|i| (0..nob).map(move |k| [i, k])).collect();
            let e_corr_x = e_corr.clone();
            pair_ik.into_par_iter().for_each(|[i, k]| {
                let d_ik = occ_energy[0][[i]] + occ_energy[1][[k]];
                let g_ab = cderi_vox[0].i((a0..a1, i, ..)) % cderi_vox[1].i((.., k, ..)).t();
                let d_ab = &d_vv_ab.i((a0..a1, ..)) + d_ik;
                let t_ab = &g_ab / &d_ab;
                *e_corr_x.lock().unwrap() += c_os * (&t_ab * &g_ab).sum();
                let mut t_x = unsafe { t_x.force_mut() };
                t_x.i_mut((.., i, .., k)).assign(&t_ab);
            });
            // G^α[(a i), P] += 2 c_os t[(a i), (b k)] (b k|P)
            let t2_a = t_x.view().into_shape([nva_b * noa, nvb * nob]);
            let g_a = (t2_a.view() % cderi_vox[1].view().into_shape([nvb * nob, naux])) * (2.0 * c_os);
            *&mut g_vix[0].i_mut((a0..a1, .., ..)) += &g_a.into_shape([nva_b, noa, naux]);
            // G^β[(b k), P] += 2 c_os Σ_{a, i} t[(i a), (b k)] (i a|P) over the window's α
            let t_ia = t_x.view().transpose([1, 0, 2, 3]).into_contig(ColMajor);
            let t2_ia = t_ia.view().into_shape([noa * nva_b, nvb * nob]);
            let c_ia_2d = cderi_vox_0_iao.i((.., a0..a1, ..)).view().into_shape([noa * nva_b, naux]);
            let g_b = (t2_ia.view().t() % c_ia_2d.view()) * (2.0 * c_os);
            *&mut g_vix[1] += &g_b.into_shape([nvb, nob, naux]);
            // β-side Grams: complete β-occupied / β-virtual pairs, partial sums over the α window
            let t3 = t_x.view().into_shape([nva_b * noa * nvb, nob]); // [(a i b), k]
            let m_b_oo = t3.view().t() % t3.view();
            *&mut rdm1[1].i_mut((0..nob, 0..nob)) -= &(m_b_oo * c_os);
            let w = t_x.view().transpose([2, 0, 1, 3]).into_contig(ColMajor);
            let w = w.view().into_shape([nvb, nva_b * noa * nob]); // [b, (a i k)]
            let m_b_vv = w.view() % w.view().t();
            *&mut rdm1[1].i_mut((nob..nmo[1], nob..nmo[1])) += &(m_b_vv * c_os);
        }

        // --- pass 2: β-virtual-windowed, α transparent; α-side Grams (no energy/G) ---
        let cderi_vox_0_2d = cderi_vox[0].view().into_shape([nva * noa, naux]); // [(a i), P]
        let d_alpha_full = (-vir_energy[0].i((.., None)) + occ_energy[0].i((None, ..))).into_dim::<Ix2>();
        for jo_slice in index_vir_outer_vec[1].windows(2) {
            let [b0, b1] = [jo_slice[0], jo_slice[1]];
            let nvb_b = b1 - b0;
            let d_beta_b = (-vir_energy[1].i((b0..b1, None)) + occ_energy[1].i((None, ..))).into_dim::<Ix2>();
            let yb = cderi_vox[1]
                .i((b0..b1, .., ..))
                .into_contig(ColMajor)
                .view()
                .into_shape([nvb_b * nob, naux]);
            let g_all = cderi_vox_0_2d.view() % yb.view().t(); // [(a i), (b j)]
            let d_all = &d_alpha_full.view().into_shape([nva * noa, 1]) + &d_beta_b.view().into_shape([1, nvb_b * nob]);
            let t_x = (g_all / d_all).into_contig(ColMajor);
            let t_x = t_x.view().into_shape([nva, noa, nvb_b, nob]);
            // α-occupied Gram: [(a b j), i] partial over the β window
            let a_oo = t_x.view().transpose([0, 2, 3, 1]).into_contig(ColMajor);
            let a_oo = a_oo.view().into_shape([nva * nvb_b * nob, noa]);
            let m_a_oo = a_oo.view().t() % a_oo.view();
            *&mut rdm1[0].i_mut((0..noa, 0..noa)) -= &(m_a_oo * c_os);
            // α-virtual Gram: [a, (i b j)] partial over the β window
            let v = t_x.view().into_shape([nva, noa * nvb_b * nob]);
            let m_a_vv = v.view() % v.view().t();
            *&mut rdm1[0].i_mut((noa..nmo[0], noa..nmo[0])) += &(m_a_vv * c_os);
        }
    }

    // ── generalized Fock blocks: per-aux-column contractions upon the complete G^s ── //
    let mut gfock = [
        rt::zeros(([nmo[0], nmo[0]].f(), &device)),
        rt::zeros(([nmo[1], nmo[1]].f(), &device)),
    ];
    for s in 0..2 {
        if nocc[s] == 0 || nvir[s] == 0 {
            continue;
        }
        let (nv, no) = (nvir[s], nocc[s]);
        let (mocc, mvir) = (&occ_coeff[s], &vir_coeff[s]);
        let g_oo: Arc<Mutex<Tsr>> = Arc::new(Mutex::new(rt::zeros(([no, no].f(), &device))));
        let g_ov: Arc<Mutex<Tsr>> = Arc::new(Mutex::new(rt::zeros(([no, nv].f(), &device))));
        let g_vo: Arc<Mutex<Tsr>> = Arc::new(Mutex::new(rt::zeros(([nv, no].f(), &device))));
        let g_vv: Arc<Mutex<Tsr>> = Arc::new(Mutex::new(rt::zeros(([nv, nv].f(), &device))));
        (0..naux).into_par_iter().for_each(|p| {
            let cderi_sy = cderi.i((.., p)).unpack_tri(Upper, FlagSymm::Sy);
            let cderi_oi = mocc.view().t() % (&cderi_sy % mocc.view());
            let g_p = g_vix[s].i((.., .., p));
            let vox_p = cderi_vox[s].i((.., .., p));
            let ov_part = &cderi_oi % g_p.t();
            let y_vv = mvir.view().t() % (&cderi_sy % mvir.view());
            let vo_part = &y_vv % &g_p;
            let oo_part = g_p.t() % &vox_p;
            let vv_part = &g_p % vox_p.t();
            *g_ov.lock().unwrap() += &ov_part;
            *g_vo.lock().unwrap() += &vo_part;
            *g_oo.lock().unwrap() += &oo_part;
            *g_vv.lock().unwrap() += &vv_part;
        });
        let g_oo = g_oo.lock().unwrap();
        let g_ov = g_ov.lock().unwrap();
        let g_vo = g_vo.lock().unwrap();
        let g_vv = g_vv.lock().unwrap();
        // orbital-energy terms of the diagonal blocks: F[i, j] += 2 eps_j rdm1[i, j]
        let scale_occ = occ_energy[s].mapv(|x| 2.0 * x);
        let scale_vir = vir_energy[s].mapv(|x| 2.0 * x);
        let so = 0..no;
        let sv = no..nmo[s];
        *&mut gfock[s].i_mut((so.clone(), so.clone())) +=
            &(&*g_oo + &scale_occ.i((None, ..)) * rdm1[s].i((so.clone(), so.clone())));
        gfock[s].i_mut((so.clone(), sv.clone())).assign(&*g_ov);
        gfock[s].i_mut((sv.clone(), so.clone())).assign(&*g_vo);
        *&mut gfock[s].i_mut((sv.clone(), sv.clone())) +=
            &(&*g_vv + &scale_vir.i((None, ..)) * rdm1[s].i((sv.clone(), sv.clone())));
    }

    let e_corr = *e_corr.lock().unwrap();
    UPT2ElecDerivIncoreOut {
        e_corr,
        rdm1_corr: rdm1,
        gfock_part: gfock,
        g_vix,
    }
}

/// Biorthogonal same-spin amplitude `T = bi1 t - bi2 t^T`.
fn T_vv_full(t_vv: &Tsr, bi1: f64, bi2: f64) -> Tsr {
    t_vv.mapv(|x| bi1 * x) - t_vv.t().mapv(|x| bi2 * x)
}
