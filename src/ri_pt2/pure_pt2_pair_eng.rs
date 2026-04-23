#![warn(unused)]
use itertools::{izip, Itertools};
use num::ToPrimitive;
use rayon::prelude::*;
use rstsr::prelude::*;

use crate::utilities::buffer_pool::BufferPool;
use rt::blas::BlasFloat;

type Tsr<T, D = IxD> = Tensor<T, DeviceBLAS, D>;
type TsrView<'a, T, D = IxD> = TensorView<'a, T, DeviceBLAS, D>;

pub struct PairEng {
    pub os: Option<Tsr<f64>>,
    pub ss: Option<Tsr<f64>>,
}

pub fn get_ript2_energy_pair_intra<T>(
    cderi_xvo: TsrView<T>,
    occ_energy: TsrView<f64>,
    vir_energy: TsrView<f64>,
    occ_occupation: Option<TsrView<f64>>,
    vir_occupation: Option<TsrView<f64>>,
    ss_only: bool,
) -> PairEng
where
    T: BlasFloat + ToPrimitive + 'static,
{
    // note this function is required to be real
    // otherwise, conj is required
    assert_eq!(occ_energy.ndim(), 1, "occ_energy must be 1D");
    assert_eq!(vir_energy.ndim(), 1, "vir_energy must be 1D");
    assert_eq!(cderi_xvo.ndim(), 3, "cderi_xvo must be 3D");
    let device = cderi_xvo.device().clone();

    // shape sanity check
    let nocc = occ_energy.size();
    let nvir = vir_energy.size();
    let naux = cderi_xvo.shape()[0];
    assert_eq!(cderi_xvo.shape(), &[naux, nvir, nocc], "cderi_xvo shape must be (naux, nvir, nocc)");

    // shape sanity check and unwrap for occupation numbers
    let occ_occupation = match occ_occupation {
        Some(occ_occ) => {
            assert_eq!(occ_occ.shape(), &[nocc], "occ_occupation shape must match occ_energy");
            occ_occ.to_owned()
        },
        None => rt::ones(([nocc].f(), &device)),
    };
    let vir_occupation = match vir_occupation {
        Some(vir_occ) => {
            assert_eq!(vir_occ.shape(), &[nvir], "vir_occupation shape must match vir_energy");
            vir_occ.to_owned()
        },
        None => rt::zeros(([nvir].f(), &device)),
    };

    // useful intermediates and buffers
    let d_vv: Tsr<f64, Ix2> = (-vir_energy.i((.., None)) - vir_energy.i((None, ..))).into_dim::<Ix2>();
    let n_vv: Tsr<f64, Ix2> =
        ((1.0_f64 - &vir_occupation.i((.., None))) * (1.0_f64 - &vir_occupation.i((None, ..)))).into_dim::<Ix2>();
    let scr_g_init = || rt::zeros(([nvir, nvir].f(), &device)).into_dim::<Ix2>();
    let scr_g_buffer = BufferPool::new(scr_g_init);

    if !ss_only {
        let eng_pair_bi1 = rt::zeros(([nocc, nocc].f(), &device));
        let eng_pair_bi2 = rt::zeros(([nocc, nocc].f(), &device));
        // create upper triangular indices pair iterator, parallel iterate it
        let idx_pair_iter = (0..nocc).flat_map(|i| (i..nocc).map(move |j| (i, j))).collect_vec();
        idx_pair_iter.into_par_iter().for_each(|(i, j)| {
            let mut scr_g = scr_g_buffer.get();
            let d_ij = occ_energy[[i]] + occ_energy[[j]];
            let n_ij = occ_occupation[[i]] * occ_occupation[[j]];

            // The following code is intutive, well, it can be even optimized by lowering memory
            // footprints of multiple (nvir, nvir) intermediates allocation.
            /*
            let g_ab = cderi_xvo.i((.., .., i)).t() % cderi_xvo.i((.., .., j));
            let g_ab = g_ab.mapv(|x| x.to_f64().unwrap()); // transform to high-precision f64 for energy calculation
            let d_ab = &d_vv + d_ij;
            let t_ab = &g_ab / &d_ab;
            let e_bi1 = (&t_ab * &g_ab).sum();
            let e_bi2 = (&t_ab.t() * &g_ab).sum();
            */
            // let scr_g = &cderi_xvo.i((.., .., i)).t() % &cderi_xvo.i((.., .., j));
            // let mut scr_g = rt::zeros(([nvir, nvir].f(), &device));
            scr_g.matmul_from(&cderi_xvo.i((.., .., i)).t(), &cderi_xvo.i((.., .., j)), T::one(), T::zero());
            let (e_bi1, e_bi2) = izip!(scr_g.iter(), scr_g.t().iter(), d_vv.iter(), n_vv.iter()).fold(
                (0.0, 0.0),
                |(e_bi1, e_bi2), (g_ab, g_ba, d_ab, n_ab)| {
                    let g_ab = g_ab.to_f64().unwrap();
                    let g_ba = g_ba.to_f64().unwrap();
                    let t_ab = g_ab / (d_ab + d_ij) * n_ab * n_ij;
                    let delta_e_bi1 = t_ab * g_ab;
                    let delta_e_bi2 = t_ab * g_ba;
                    (e_bi1 + delta_e_bi1, e_bi2 + delta_e_bi2)
                },
            );

            let mut eng_pair_bi1 = unsafe { eng_pair_bi1.force_mut() };
            let mut eng_pair_bi2 = unsafe { eng_pair_bi2.force_mut() };
            eng_pair_bi1[[i, j]] = e_bi1;
            eng_pair_bi1[[j, i]] = e_bi1;
            eng_pair_bi2[[i, j]] = e_bi2;
            eng_pair_bi2[[j, i]] = e_bi2;

            // put back the buffer for reuse
            scr_g_buffer.put(scr_g);
        });
        let eng_pair_os = eng_pair_bi1;
        let eng_pair_ss = &eng_pair_os - eng_pair_bi2;
        PairEng { os: Some(eng_pair_os), ss: Some(eng_pair_ss) }
    } else {
        let eng_pair_ss = rt::zeros(([nocc, nocc].f(), &device));
        // create upper triangular (non-diagonal) indices pair iterator, parallel iterate it
        let idx_pair_iter = (0..nocc).flat_map(|i| (i + 1..nocc).map(move |j| (i, j))).collect_vec();
        idx_pair_iter.into_par_iter().for_each(|(i, j)| {
            let mut scr_g = scr_g_buffer.get();
            let d_ij = occ_energy[[i]] + occ_energy[[j]];
            let n_ij = occ_occupation[[i]] * occ_occupation[[j]];

            // the following code is intutive but can be further optimized.
            /*
            let g_ab = cderi_xvo.i((.., .., i)).t() % cderi_xvo.i((.., .., j));
            let g_ab = &g_ab - &g_ab.t(); // antisymmetrize the integrals
            let g_ab = g_ab.mapv(|x| x.to_f64().unwrap());
            let d_ab = &d_vv + d_ij;
            let t_ab = &g_ab / &d_ab;
            let e_ss = (&t_ab * &g_ab).sum();
            */
            scr_g.matmul_from(&cderi_xvo.i((.., .., i)).t(), &cderi_xvo.i((.., .., j)), T::one(), T::zero());
            let e_ss = izip!(scr_g.iter(), scr_g.t().iter(), d_vv.iter(), n_vv.iter()).fold(
                0.0,
                |e_ss, (g_ab, g_ba, d_ab, n_ab)| {
                    let g_ab_asymm = g_ab.to_f64().unwrap() - g_ba.to_f64().unwrap();
                    let delta_e_ss = g_ab_asymm * g_ab_asymm / (d_ab + d_ij) * n_ab * n_ij;
                    e_ss + delta_e_ss
                },
            );

            let mut eng_pair_ss = unsafe { eng_pair_ss.force_mut() };
            eng_pair_ss[[i, j]] = e_ss;
            eng_pair_ss[[j, i]] = e_ss;

            // put back the buffer for reuse
            scr_g_buffer.put(scr_g);
        });
        PairEng { os: None, ss: Some(eng_pair_ss) }
    }
}

pub fn get_riuospt2_energy_pair_intra<T>(
    cderi_xvo: [TsrView<T>; 2],
    occ_energy: [TsrView<f64>; 2],
    vir_energy: [TsrView<f64>; 2],
    occ_occupation: Option<[TsrView<f64>; 2]>,
    vir_occupation: Option<[TsrView<f64>; 2]>,
) -> PairEng
where
    T: BlasFloat + ToPrimitive + 'static,
{
    // note this function is required to be real
    // otherwise, conj is required

    const SPINS: [usize; 2] = [0, 1]; // 0 for alpha, 1 for beta
    const A: usize = 0;
    const B: usize = 1;

    let mut nocc = [0, 0];
    let mut nvir = [0, 0];

    for spin in SPINS {
        // shape sanity check
        assert_eq!(occ_energy[spin].ndim(), 1, "occ_energy[{spin}] must be 1D");
        assert_eq!(vir_energy[spin].ndim(), 1, "vir_energy[{spin}] must be 1D");
        assert_eq!(cderi_xvo[spin].ndim(), 3, "cderi_xvo[{spin}] must be 3D");
        nocc[spin] = occ_energy[spin].size();
        nvir[spin] = vir_energy[spin].size();
    }
    let naux = cderi_xvo[A].shape()[0];
    assert_eq!(cderi_xvo[A].shape(), &[naux, nvir[A], nocc[A]], "cderi_xvo shape not match");
    assert_eq!(cderi_xvo[B].shape(), &[naux, nvir[B], nocc[B]], "cderi_xvo shape not match");
    let device = cderi_xvo[A].device().clone();

    // shape sanity check and unwrap for occupation numbers
    let occ_occupation = match occ_occupation {
        Some(occ_occ) => {
            for spin in SPINS {
                assert_eq!(
                    occ_occ[spin].shape(),
                    &[nocc[spin]],
                    "occ_occupation[{spin}] shape must match occ_energy[{spin}]"
                );
            }
            [occ_occ[A].to_owned(), occ_occ[B].to_owned()]
        },
        None => [rt::ones(([nocc[A]].f(), &device)), rt::ones(([nocc[B]].f(), &device))],
    };
    let vir_occupation = match vir_occupation {
        Some(vir_occ) => {
            for spin in SPINS {
                assert_eq!(
                    vir_occ[spin].shape(),
                    &[nvir[spin]],
                    "vir_occupation[{spin}] shape must match vir_energy[{spin}]"
                );
            }
            [vir_occ[A].to_owned(), vir_occ[B].to_owned()]
        },
        None => [rt::zeros(([nvir[A]].f(), &device)), rt::zeros(([nvir[B]].f(), &device))],
    };

    // useful intermediates and buffers
    let d_vv = (-vir_energy[A].i((.., None)) - vir_energy[B].i((None, ..))).into_dim::<Ix2>();
    let n_vv =
        ((1.0_f64 - &vir_occupation[A].i((.., None))) * (1.0_f64 - &vir_occupation[B].i((None, ..)))).into_dim::<Ix2>();

    let scr_g_init = || rt::zeros(([nvir[A], nvir[B]].f(), &device)).into_dim::<Ix2>();
    let scr_g_buffer = BufferPool::new(scr_g_init);

    let eng_pair_os = rt::zeros(([nocc[A], nocc[B]].f(), &device));
    // create indices pair iterator (not triangular at this time), parallel iterate it
    let idx_pair_iter = (0..nocc[A]).cartesian_product(0..nocc[B]).collect_vec();
    idx_pair_iter.into_par_iter().for_each(|(i, j)| {
        let mut scr_g = scr_g_buffer.get();
        let d_ij = occ_energy[A][[i]] + occ_energy[B][[j]];
        let n_ij = occ_occupation[A][[i]] * occ_occupation[B][[j]];

        // the following code is intutive but can be further optimized.
        /*
        let g_ab = cderi_xvo[A].i((.., .., i)).t() % cderi_xvo[B].i((.., .., j));
        let g_ab = g_ab.mapv(|x| x.to_f64().unwrap());
        let d_ab = &d_vv + d_ij;
        let t_ab = &g_ab / &d_ab;
        let e_os = (&t_ab * &g_ab).sum();
        */
        scr_g.matmul_from(&cderi_xvo[A].i((.., .., i)).t(), &cderi_xvo[B].i((.., .., j)), T::one(), T::zero());
        let e_os = izip!(scr_g.iter(), d_vv.iter(), n_vv.iter()).fold(0.0, |e_os, (g_ab, d_ab, n_ab)| {
            let g_ab = g_ab.to_f64().unwrap();
            let delta_e_os = g_ab * g_ab / (d_ab + d_ij) * n_ij * n_ab;
            e_os + delta_e_os
        });

        let mut eng_pair_os = unsafe { eng_pair_os.force_mut() };
        eng_pair_os[[i, j]] = e_os;

        // put back the buffer for reuse
        scr_g_buffer.put(scr_g);
    });

    PairEng { os: Some(eng_pair_os), ss: None }
}
