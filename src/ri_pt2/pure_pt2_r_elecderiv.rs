//! Pure functions for generalized Fock (restricted) and related methods for RI-PT2.
#![warn(unused)]
use num::traits::NumAssignOps;
use num::{FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use rstsr::prelude::*;
use rt::blas::BlasFloat;
use std::sync::{Arc, Mutex};

type Tsr<T, D = IxD> = Tensor<T, DeviceBLAS, D>;
type TsrView<'a, T, D = IxD> = TensorView<'a, T, DeviceBLAS, D>;
type TsrCow<'a, T, D = IxD> = TensorCow<'a, T, DeviceBLAS, D>;

pub struct RPT2ElecDerivIncoreInp<'a, T, O>
where
    T: BlasFloat + 'static,
    O: BlasFloat + 'static,
{
    pub cderi: TsrView<'a, T>,
    pub cderi_vox: Option<TsrView<'a, O>>,
    pub occ_coeff: TsrView<'a, T>,
    pub vir_coeff: TsrView<'a, T>,
    pub occ_energy: TsrView<'a, T>,
    pub vir_energy: TsrView<'a, T>,
    pub index_occ_outer_vec: &'a [usize],
}

pub struct RPT2ElecDerivIncoreArg {
    pub c_os: f64,
    pub c_ss: f64,
}

pub struct RPT2ElecDerivIncoreOut {
    /// MP2 correlation energy. This is side product from property evaluation.
    pub e_corr: f64,
    /// **Partial** generalized Fock matrix contribution. Shape `(nmo, nmo)`.
    ///
    /// Note this lacks SCF response upon rdm1_corr contribution (the
    /// $A_{ai, pq} D_{pq}^{\mathrm{RDM}}$ term of the Lagrangian, which requires a response
    /// object of the underlying SCF).
    pub gfock_part: Tsr<f64>,
    /// 1-RDM in MO basis (correlation contribution, unrelaxed). Shape `(nmo, nmo)`.
    pub rdm1_corr: Tsr<f64>,
}

pub fn get_rpt2_elec_deriv_incore<T, O>(
    input: &RPT2ElecDerivIncoreInp<T, O>,
    arg: &RPT2ElecDerivIncoreArg,
    cast: impl Fn(T) -> O + Send + Sync,
) -> RPT2ElecDerivIncoreOut
where
    T: BlasFloat + ToPrimitive + FromPrimitive + NumAssignOps + 'static,
    O: BlasFloat + ToPrimitive + FromPrimitive + NumAssignOps + 'static,
{
    let VAL_0 = O::from_f64(0.0).unwrap();
    let VAL_1 = O::from_f64(1.0).unwrap();

    // --- input --- //

    let RPT2ElecDerivIncoreInp { cderi, cderi_vox, occ_coeff, vir_coeff, occ_energy, vir_energy, index_occ_outer_vec } =
        input;
    let RPT2ElecDerivIncoreArg { c_os, c_ss } = arg;

    let bi1_scale = O::from_f64(2.0 * c_os).unwrap();
    let bi2_scale = O::from_f64(*c_ss).unwrap();
    let device = cderi.device().clone();

    // --- dimension check --- //

    let &[nao_tp, naux] = cderi.shape().as_array().unwrap();
    let &[nao, nocc] = occ_coeff.shape().as_array().unwrap();
    let &[_, nvir] = vir_coeff.shape().as_array().unwrap();

    assert_eq!(nao_tp, nao * (nao + 1) / 2, "cderi shape mismatch (nao)");
    assert_eq!(nocc, occ_energy.shape()[0], "occ_energy shape mismatch (nocc)");
    assert_eq!(nvir, vir_energy.shape()[0], "vir_energy shape mismatch (nvir)");
    assert_eq!(nao, vir_coeff.shape()[0], "vir_coeff shape mismatch (nao)");
    assert_eq!(index_occ_outer_vec.first(), Some(&0), "index_occ_outer_vec must start with 0");
    assert_eq!(index_occ_outer_vec.last(), Some(&nocc), "index_occ_outer_vec must end with nocc");
    assert!(index_occ_outer_vec.is_sorted(), "index_occ_outer_vec must be sorted");

    let nstep_occ_max = index_occ_outer_vec.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
    let nmo = nocc + nvir;
    let so = rt::slice!(0, nocc);
    let sv = rt::slice!(nocc, nmo);

    // --- output --- //

    let eng_corr_double: Arc<Mutex<f64>> = Arc::new(Mutex::new(0.0));
    let mut gfock_part = rt::zeros(([nmo, nmo].f(), &device));
    let mut rdm1_corr = rt::zeros(([nmo, nmo].f(), &device));
    let w3: Arc<Mutex<Tsr<O>>> = Arc::new(Mutex::new(rt::zeros(([nocc, nvir].f(), &device))));
    let w4: Arc<Mutex<Tsr<O>>> = Arc::new(Mutex::new(rt::zeros(([nvir, nocc].f(), &device))));

    // --- buffer allocation --- //

    // 0th tensors
    let mut t_vivo: Tsr<O> = rt::zeros(([nvir, nstep_occ_max, nvir, nocc].f(), &device));
    let mut T_vivo: Tsr<O> = rt::zeros(([nvir, nstep_occ_max, nvir, nocc].f(), &device));
    let mut G_vix: Tsr<O> = rt::zeros(([nvir, nstep_occ_max, naux].f(), &device));

    // --- initiate necessary tensors --- //

    let d_vv_outer = -vir_energy.i((.., None)) - vir_energy.i((None, ..));
    let d_vv_outer = d_vv_outer.into_dim::<Ix2>();

    // --- block-1 --- //

    use crate::ri_jk::pure_ao2mo::get_ao2mo_s2ij_to_s1_notrans;
    let cderi_vox: TsrCow<'_, O> = match cderi_vox {
        Some(cderi_vox) => cderi_vox.view().into_cow(),
        None => {
            let mut cderi_vox_lst =
                get_ao2mo_s2ij_to_s1_notrans(cderi.view(), Upper, &[vir_coeff.view()], &[occ_coeff.view()], &cast);
            cderi_vox_lst.remove(0).into_cow()
        },
    };

    // coefficient matrices in `O`, for the block-4 contractions upon `O`-typed intermediates
    let occ_coeff: Tsr<O> = occ_coeff.mapv(|x| cast(x));
    let vir_coeff: Tsr<O> = vir_coeff.mapv(|x| cast(x));

    for io_slice in index_occ_outer_vec.windows(2) {
        let nstep_occ = io_slice[1] - io_slice[0];
        let t_vivo = rt::asarray((t_vivo.raw_mut(), [nvir, nstep_occ, nvir, nocc].f(), &device));
        let T_vivo = rt::asarray((T_vivo.raw_mut(), [nvir, nstep_occ, nvir, nocc].f(), &device));
        let mut G_vix = rt::asarray((G_vix.raw_mut(), [nvir, nstep_occ, naux].f(), &device));

        // generate (i, j) pairs
        let mut pair_ij = Vec::new();

        for i in io_slice[0]..io_slice[1] {
            let range_1 = io_slice[0]..=i;
            let range_2 = 0..io_slice[0];
            let range_3 = io_slice[1]..nocc;
            pair_ij.extend(range_1.chain(range_2).chain(range_3).map(|j| [i, j]));
        }
        let npair_ij = pair_ij.len();

        // --- block-2 --- //

        (0..npair_ij).into_par_iter().for_each(|idx| {
            let mut eng_corr_ij = 0.0;
            // index
            let [i, j] = pair_ij[idx];
            // t_vv, T_vv
            let d_ij = occ_energy[[i]] + occ_energy[[j]];
            let g_vv = cderi_vox.i((.., i, ..)) % cderi_vox.i((.., j, ..)).t();
            let d_vv = (&d_vv_outer + d_ij).mapv(&cast);
            let t_vv = &g_vv / &d_vv;
            let T_vv = &t_vv * bi1_scale - t_vv.t() * bi2_scale;
            if j <= i {
                eng_corr_ij += (&T_vv * &g_vv).sum().to_f64().unwrap();
                let diag_scale = if i == j { 1.0 } else { 2.0 };
                *eng_corr_double.lock().unwrap() += diag_scale * eng_corr_ij;
            }

            // assign to t_vivo, T_vivo
            let mut t_vivo = unsafe { t_vivo.force_mut() };
            let mut T_vivo = unsafe { T_vivo.force_mut() };
            let i_ = i - io_slice[0];
            t_vivo.i_mut((.., i_, .., j)).assign(&t_vv);
            T_vivo.i_mut((.., i_, .., j)).assign(&T_vv);
            if (io_slice[0] <= j) && (j < i) {
                let j_ = j - io_slice[0];
                t_vivo.i_mut((.., j_, .., i)).assign(t_vv.t());
                T_vivo.i_mut((.., j_, .., i)).assign(T_vv.t());
            }
        });

        // --- block-3 --- //

        // D[oo] = t[viv,o]' % T[viv,o]
        let scr = t_vivo.reshape((-1, nocc)).t() % T_vivo.reshape((-1, nocc));
        *&mut rdm1_corr.i_mut((so, so)) -= 2.0 * scr.mapv(|x| x.to_f64().unwrap());
        // D[vv] = t[v,ivo] % T[v,ivo]'
        let scr = t_vivo.reshape((nvir, -1)) % T_vivo.reshape((nvir, -1)).t();
        *&mut rdm1_corr.i_mut((sv, sv)) += 2.0 * scr.mapv(|x| x.to_f64().unwrap());
        // G[vi,x] = T[vi,vo] % cderi[vo,x]
        let mut scr_G_vix = rt::asarray((G_vix.raw_mut(), [nvir * nstep_occ, naux].f(), &device));
        scr_G_vix.matmul_from(
            T_vivo.reshape((nvir * nstep_occ, nvir * nocc)),
            cderi_vox.reshape((nvir * nocc, naux)),
            VAL_1,
            VAL_0,
        );

        // --- block-4 --- //

        let mocc_batch = occ_coeff.i((.., io_slice[0]..io_slice[1]));
        let w4_bi: Arc<Mutex<Tsr<O>>> = Arc::new(Mutex::new(rt::zeros(([nao, nstep_occ].f(), &device))));
        (0..naux).into_par_iter().for_each(|p| {
            // cderi[oix]: ao2mo
            let cderi_sy = cderi.i((.., p)).unpack_tri(Upper, FlagSymm::Sy).mapv(&cast);
            let cderi_bi = &cderi_sy % &mocc_batch;
            let cderi_oi = occ_coeff.t() % &cderi_bi;
            // w3[ov] = cderi[oi, x] % G[vi, x]'
            let w3_part = cderi_oi % G_vix.i((.., .., p)).t();
            {
                let mut w3 = w3.lock().unwrap();
                *w3 += &w3_part;
            }

            // w4[bi] = cderi[bb, x] % G[bi, x]
            let G_bi = &vir_coeff % G_vix.i((.., .., p));
            let w4_bi_part = &cderi_sy % &G_bi;
            {
                let mut w4_bi = w4_bi.lock().unwrap();
                *w4_bi += &w4_bi_part;
            }
        });
        let w4_bi = w4_bi.lock().unwrap();
        w4.lock().unwrap().i_mut((.., io_slice[0]..io_slice[1])).assign(vir_coeff.t() % w4_bi.view());
    }

    let e_corr = *eng_corr_double.lock().unwrap();
    let w3 = w3.lock().unwrap();
    let w4 = w4.lock().unwrap();
    gfock_part.i_mut((so, sv)).assign(w3.mapv(|x| 4.0 * x.to_f64().unwrap()));
    gfock_part.i_mut((sv, so)).assign(w4.mapv(|x| 4.0 * x.to_f64().unwrap()));

    RPT2ElecDerivIncoreOut { e_corr, gfock_part, rdm1_corr }
}
