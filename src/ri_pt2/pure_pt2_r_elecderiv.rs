//! Pure functions for generalized Fock (restricted) and related methods for RI-PT2.
use enumflags2::BitFlags;
// #![warn(unused)]
use itertools::{izip, Itertools};
use num::{FromPrimitive, ToPrimitive};
use num_complex::ComplexFloat;
use rayon::prelude::*;
use rstsr::prelude::*;
use rt::blas::BlasFloat;
use std::sync::{Arc, Mutex};

use crate::analdrv::response::trait_rgfock::{GFockParts, RGFockAPI};
use crate::utilities::buffer_pool::BufferPool;

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
    pub e_corr: f64,
    pub gfock: Tsr<f64>,
    pub rdm1_corr: Tsr<f64>,
}

pub fn get_rpt2_elec_deriv_incore<T, O>(
    input: &RPT2ElecDerivIncoreInp<T, O>,
    arg: &RPT2ElecDerivIncoreArg,
    cast: impl Fn(T) -> O + Send + Sync,
) -> RPT2ElecDerivIncoreOut
where
    T: BlasFloat + ToPrimitive + FromPrimitive + 'static,
    O: BlasFloat + ToPrimitive + FromPrimitive + 'static,
{
    let VAL_0 = O::from_f64(0.0).unwrap();
    let VAL_1 = O::from_f64(1.0).unwrap();
    let VAL_2 = O::from_f64(2.0).unwrap();

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
    let mut gfock = rt::zeros(([nmo, nmo].f(), &device));
    let mut rdm1_corr = rt::zeros(([nmo, nmo].f(), &device));

    // --- buffer allocation --- //

    let init_buf_t = || -> Tsr<O, Ix2> { rt::zeros(([nao, nao].f(), &device)).into_dim() };
    let buf_t1_pool = BufferPool::new(init_buf_t);
    let buf_t2_pool = BufferPool::new(init_buf_t);
    let buf_t3_pool = BufferPool::new(init_buf_t);

    // 0th tensors
    let mut t_vivo: Tsr<O, Ix4> = rt::zeros(([nvir, nstep_occ_max, nvir, nocc].f(), &device)).into_dim();
    let mut T_vivo: Tsr<O, Ix4> = rt::zeros(([nvir, nstep_occ_max, nvir, nocc].f(), &device)).into_dim();

    // --- initiate necessary tensors --- //
    let d_vv_outer = -vir_energy.i((.., None)) - vir_energy.i((None, ..));
    let d_vv_outer = d_vv_outer.into_dim::<Ix2>();

    // --- block-1 --- //

    let cderi_vox: TsrCow<'_, O> = match cderi_vox {
        Some(cderi_vox) => cderi_vox.view().into_cow(),
        None => {
            use crate::ri_jk::pure_ao2mo::get_ao2mo_s2ij_to_s1_notrans;
            let mut cderi_vox_lst =
                get_ao2mo_s2ij_to_s1_notrans(cderi.view(), Upper, &[vir_coeff.view()], &[occ_coeff.view()], &cast);
            cderi_vox_lst.remove(0).into_cow()
        },
    };

    for io_slice in index_occ_outer_vec.windows(2) {
        let nstep_occ = io_slice[1] - io_slice[0];
        let mut t_vivo = rt::asarray((t_vivo.raw_mut(), [nvir, nstep_occ, nvir, nocc].f(), &device));
        let mut T_vivo = rt::asarray((T_vivo.raw_mut(), [nvir, nstep_occ, nvir, nocc].f(), &device));

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
            // temporary buffer
            let mut buf_t1 = buf_t1_pool.get();
            let mut buf_t2 = buf_t2_pool.get();
            let mut t_vv = rt::asarray((buf_t1.raw_mut(), [nvir, nvir].f(), &device));
            let mut T_vv = rt::asarray((buf_t2.raw_mut(), [nvir, nvir].f(), &device));
            let mut eng_corr_ij = 0.0;
            // index
            let [i, j] = pair_ij[idx];
            // t_vv, T_vv
            // t_vv /= d_vv
            // T_vv = (2 * c_os) * t_vv - c_ss * t_vv.t()
            let d_ij = occ_energy[[i]] + occ_energy[[j]];
            t_vv.matmul_from(cderi_vox.i((.., i, ..)), cderi_vox.i((.., j, ..)).t(), VAL_1, VAL_0);
            for a in 0..nvir {
                // diagonal part
                let d_aa_inv = cast(d_ij + d_vv_outer[[a, a]]).recip();
                let g_aa = t_vv[[a, a]];
                let t_aa = g_aa * d_aa_inv;
                let T_aa = (bi1_scale - bi2_scale) * t_aa;
                t_vv[[a, a]] = t_aa;
                T_vv[[a, a]] = T_aa;
                if j <= i {
                    eng_corr_ij += (T_aa * g_aa).to_f64().unwrap();
                }
                // off-diagonal part
                for b in 0..a {
                    let d_ab_inv = cast(d_ij + d_vv_outer[[a, b]]).recip();
                    let g_ab = t_vv[[a, b]];
                    let g_ba = t_vv[[b, a]];
                    let t_ab = g_ab * d_ab_inv;
                    let t_ba = g_ba * d_ab_inv;
                    let T_ab = bi1_scale * t_ab - bi2_scale * t_ba;
                    let T_ba = bi1_scale * t_ba - bi2_scale * t_ab;
                    t_vv[[a, b]] = t_ab;
                    t_vv[[b, a]] = t_ba;
                    T_vv[[a, b]] = T_ab;
                    T_vv[[b, a]] = T_ba;
                    if j <= i {
                        eng_corr_ij += (T_ab * g_ab + T_ba * g_ba).to_f64().unwrap();
                    }
                }
            }
            let diag_scale = if i == j { 1.0 } else { 2.0 };
            *eng_corr_double.lock().unwrap() += diag_scale * eng_corr_ij;

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

            buf_t1_pool.put(buf_t1);
            buf_t2_pool.put(buf_t2);
        });

        // --- block-3 --- //

        let scr = t_vivo.reshape((-1, nocc)).t() % T_vivo.reshape((-1, nocc));
        *&mut rdm1_corr.i_mut((so, so)) -= 2.0 * scr.mapv(|x| x.to_f64().unwrap());
        let scr = t_vivo.reshape((nvir, -1)) % T_vivo.reshape((nvir, -1)).t();
        *&mut rdm1_corr.i_mut((sv, sv)) += 2.0 * scr.mapv(|x| x.to_f64().unwrap());
    }

    let e_corr = *eng_corr_double.lock().unwrap();
    RPT2ElecDerivIncoreOut { e_corr, gfock, rdm1_corr }
}
