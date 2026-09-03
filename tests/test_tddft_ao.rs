//! Integration tests for the AO-mode TDDFT kernels (`ri_tddft::matvec_ao`).
//!
//! Moved from the in-crate `#[cfg(test)]` module: these tests exercise the
//! public kernel primitives against naive four-index references built from a
//! synthetic packed 3c2e integral table.

use pyrest::ri_jk::pure_incore::{get_vj_ri_incore_nonsym, get_vk_ri_incore_coeff_pair, get_vk_ri_incore_dm};
use pyrest::ri_tddft::matvec_ao::{contract_back, transition_density, RimatrTuple};
use pyrest::ri_tddft::tddft::{FxcDriver, TDDFTData};
use pyrest::ri_tddft::TDDFTMode;
use pyrest::dft::xceff::prelude::XCDenType;
use pyrest::scf_io::SCFType;
use pyrest::utilities::rstsr_util::{RestTensorToRstsrTsrAPI, RestTensorToRstsrViewAPI};
use rest_tensors::matrixupper::map_upper_to_full;
use rest_tensors::MatrixFull;
use rstsr::prelude::*;

/// Deterministic pseudo-random fill (mirrors matvec.rs tests).
fn pseudo(n: usize, seed: f64) -> Vec<f64> {
    (0..n).map(|i| ((i as f64 + seed) * 0.37 + seed).sin() * 0.5).collect()
}

/// Build a synthetic rimatr: nao=6 → npair=21, naux=4, random folded ri3fn.
fn synthetic_rimatr(nao: usize, naux: usize) -> RimatrTuple {
    let npair = nao * (nao + 1) / 2;
    let ri3fn = MatrixFull::from_vec([npair, naux], pseudo(npair * naux, 1.7)).unwrap();
    let basbas2baspar = MatrixFull::from_vec([nao, 1], vec![0usize; nao]).unwrap();
    Some((ri3fn, basbas2baspar, vec![]))
}

/// Unfold a folded upper-triangle column into a full symmetric matrix.
fn unfold(m: &[f64], nao: usize) -> MatrixFull<f64> {
    let npair = m.len();
    let index_map = map_upper_to_full(npair).unwrap();
    let mut mq = MatrixFull::new([nao, nao], 0.0);
    for (k, ij) in index_map.data.iter().enumerate() {
        let v = m[k];
        mq[[ij[0], ij[1]]] = v;
        mq[[ij[1], ij[0]]] = v;
    }
    mq
}

/// Reconstruct the full four-index integral (μν|λσ) = Σ_Q B_μνQ B_λσQ.
fn four_index_integrals(rimatr: &RimatrTuple) -> Vec<Vec<Vec<Vec<f64>>>> {
    let (ri3fn, basbas2baspar, _) = rimatr.as_ref().unwrap();
    let nao = basbas2baspar.size[0];
    let naux = ri3fn.size[1];
    let ms: Vec<MatrixFull<f64>> = (0..naux)
        .map(|q| unfold(&ri3fn.iter_columns_full().nth(q).unwrap().to_vec(), nao))
        .collect();
    let mut eri = vec![vec![vec![vec![0.0; nao]; nao]; nao]; nao];
    for mu in 0..nao { for nu in 0..nao { for lam in 0..nao { for sig in 0..nao {
        let mut s = 0.0;
        for q in 0..naux {
            s += ms[q][[mu, nu]] * ms[q][[lam, sig]];
        }
        eri[mu][nu][lam][sig] = s;
    }}}}
    eri
}

#[test]
fn test_transition_density() {
    let nao = 6; let occ = 3; let vir = 4;
    let c_occ = MatrixFull::from_vec([nao, occ], pseudo(nao * occ, 2.1)).unwrap();
    let c_vir = MatrixFull::from_vec([nao, vir], pseudo(nao * vir, 3.3)).unwrap();
    let z: Vec<f64> = pseudo(occ * vir, 4.4);
    let p = transition_density(&c_occ, &c_vir, &z);
    // naive reference
    for mu in 0..nao { for nu in 0..nao {
        let mut s = 0.0;
        for i in 0..occ { for a in 0..vir {
            s += c_occ[[mu, i]] * c_vir[[nu, a]] * z[i + a * occ];
        }}
        assert!((p[[mu, nu]] - s).abs() < 1e-12, "P[{},{}] = {} vs {}", mu, nu, p[[mu, nu]], s);
    }}
}

#[test]
fn test_ri_coulomb_ao_vs_naive() {
    let nao = 6; let naux = 4;
    let rimatr = synthetic_rimatr(nao, naux);
    let eri = four_index_integrals(&rimatr);
    // Non-symmetric transition density
    let p = MatrixFull::from_vec([nao, nao], pseudo(nao * nao, 5.5)).unwrap();
    let device = DeviceBLAS::default();
    let (ri3fn, _, _) = rimatr.as_ref().unwrap();
    let cderi = ri3fn.to_rstsr_view(&device);
    let dms = vec![p.clone()].as_slice().to_rstsr(&device);
    let js = pyrest::ri_jk::pure_incore::get_vj_ri_incore_nonsym(cderi, dms.view());
    let f = MatrixFull::from_vec([nao, nao], js.i((.., .., 0)).iter().copied().collect()).unwrap();
    for mu in 0..nao { for nu in 0..nao {
        let mut s = 0.0;
        for lam in 0..nao { for sig in 0..nao {
            s += p[[lam, sig]] * eri[mu][nu][lam][sig];
        }}
        assert!((f[[mu, nu]] - s).abs() < 1e-10,
            "J[{},{}] = {} vs {}", mu, nu, f[[mu, nu]], s);
    }}
}

#[test]
fn test_ri_exchange_ao_vs_naive() {
    let nao = 6; let naux = 4;
    let rimatr = synthetic_rimatr(nao, naux);
    let eri = four_index_integrals(&rimatr);
    let p = MatrixFull::from_vec([nao, nao], pseudo(nao * nao, 6.6)).unwrap();
    let device = DeviceBLAS::default();
    let (ri3fn, _, _) = rimatr.as_ref().unwrap();
    let cderi = ri3fn.to_rstsr_view(&device);
    let dms = vec![p.clone()].as_slice().to_rstsr(&device);
    let ks = pyrest::ri_jk::pure_incore::get_vk_ri_incore_dm(cderi, dms.view(), 2);
    let k = MatrixFull::from_vec([nao, nao], ks.i((.., .., 0)).iter().copied().collect()).unwrap();
    // K[μν] = Σ_λσ P_λσ (μλ|σν)
    for mu in 0..nao { for nu in 0..nao {
        let mut s = 0.0;
        for lam in 0..nao { for sig in 0..nao {
            s += p[[lam, sig]] * eri[mu][lam][sig][nu];
        }}
        assert!((k[[mu, nu]] - s).abs() < 1e-10,
            "K[{},{}] = {} vs {}", mu, nu, k[[mu, nu]], s);
    }}
}

fn build_ao_data(nvar: usize) -> TDDFTData {
    let nao = 6; let occ = 3; let vir = 4; let ng = 17;

    let c_occ = MatrixFull::from_vec([nao, occ], pseudo(nao * occ, 11.2)).unwrap();
    let c_vir = MatrixFull::from_vec([nao, vir], pseudo(nao * vir, 12.3)).unwrap();
    let den_type = if nvar == 4 { XCDenType::SIGMA } else { XCDenType::RHO };
    let data = TDDFTData {
        mode: TDDFTMode::AO,
        alpha_hybrid: 0.0,
        fxc: None,
        c_occ: vec![c_occ],
        c_vir: vec![c_vir],
        ni: None,
        fxc_eff: None,
        den_type: Some(den_type),
        grid_batch: false,
        fxc_driver: Some(FxcDriver::MO),
        psi_occ: None,
        psi_occ_grad: None,
        ri_ov: None,
        ri_oo_exch: None,
        ri_vv_exch: None,
        ri_ov_exch: None,
        reftype: SCFType::RHF,
    };
    data
}

#[test]
fn test_b_exchange_uses_transposed_density() {
    // Verify b_matvec_ao's exchange sign/construction by checking that
    // the full-matrix route reproduces K_B[ia] = Σ_jb (ib|aj) z_jb.
    let nao = 6; let naux = 4;
    let rimatr = synthetic_rimatr(nao, naux);
    let eri = four_index_integrals(&rimatr);
    let data = build_ao_data(1); // nvar irrelevant for exchange
    let c_occ = &data.c_occ[0];
    let c_vir = &data.c_vir[0];
    let occ = c_occ.size[1];
    let vir = c_vir.size[1];
    let z: Vec<f64> = pseudo(occ * vir, 16.7);
    let p = transition_density(c_occ, c_vir, &z);
    let p_t = p.transpose_and_drop();
    let device = DeviceBLAS::default();
    let (ri3fn, _, _) = rimatr.as_ref().unwrap();
    let cderi = ri3fn.to_rstsr_view(&device);
    let dms = vec![p_t.clone()].as_slice().to_rstsr(&device);
    let ks = pyrest::ri_jk::pure_incore::get_vk_ri_incore_dm(cderi, dms.view(), 2);
    let k = MatrixFull::from_vec([nao, nao], ks.i((.., .., 0)).iter().copied().collect()).unwrap();
    let result = contract_back(&k, c_occ, c_vir);
    // naive MO reference: Σ_jb (ib|aj) z_jb with MO integrals via 4-center
    for i in 0..occ { for a in 0..vir {
        let mut s = 0.0;
        for j in 0..occ { for b in 0..vir {
            // (ib|aj) in AO terms over all MO coefficients
            let mut eri_mo = 0.0;
            for mu in 0..nao { for nu in 0..nao { for lam in 0..nao { for sig in 0..nao {
                eri_mo += c_occ[[mu, i]] * c_vir[[nu, b]] * c_vir[[lam, a]] * c_occ[[sig, j]]
                    * eri[mu][nu][lam][sig];
            }}}}
            s += z[j + b * occ] * eri_mo;
        }}
        assert!((result[i + a * occ] - s).abs() < 1e-10,
            "K_B[{},{}] = {} vs {}", i, a, result[i + a * occ], s);
    }}
}

#[test]
fn test_exchange_coeff_route_matches_dm() {
    // The "semitrans" driver folds the amplitudes first: CX = C_vir·Xᵀ (nao×occ),
    // then K[CX·C_occᵀ] = K[Pᵀ] via ri_jk::get_vk_ri_incore_coeff_pair.
    // Checks: (a) K[Pᵀ] from the coeff route == exact dm route on Pᵀ (B block);
    //         (b) its transpose == exact dm route on P (A block, k = occ side).
    let nao = 6; let naux = 4;
    let rimatr = synthetic_rimatr(nao, naux);
    let data = build_ao_data(1);
    let c_occ = &data.c_occ[0];
    let c_vir = &data.c_vir[0];
    let occ = c_occ.size[1];
    let vir = c_vir.size[1];
    let z: Vec<f64> = pseudo(occ * vir, 18.9);
    let device = DeviceBLAS::default();
    let (ri3fn, _, _) = rimatr.as_ref().unwrap();
    let cderi = ri3fn.to_rstsr_view(&device);

    let p = transition_density(c_occ, c_vir, &z);
    let p_t = p.clone().transpose_and_drop();
    let dms = vec![p, p_t.clone()].as_slice().to_rstsr(&device);
    let ks_exact = pyrest::ri_jk::pure_incore::get_vk_ri_incore_dm(cderi.view(), dms.view(), naux);

    // fold: x_t [vir, occ, 1] = Xᵀ; CX = C_vir·Xᵀ [nao, occ, 1]
    let mut x_t = vec![0.0_f64; vir * occ];
    for a in 0..vir { for i in 0..occ {
        x_t[a + i * vir] = z[i + a * occ];
    }}
    let x_tsr = rt::asarray((x_t, [vir, occ, 1].f(), &device));
    let c_vir_v = c_vir.to_rstsr_view(&device);
    let c_occ_v = c_occ.to_rstsr_view(&device);
    let mut cx = rt::zeros(([nao, occ, 1].f(), &device));
    cx.i_mut((.., .., 0)).matmul_from(&c_vir_v, &x_tsr.i((.., .., 0)), 1.0, 0.0);
    let k_coeff = pyrest::ri_jk::pure_incore::get_vk_ri_incore_coeff_pair(
        cderi, cx.view(), c_occ_v.view(), naux,
    );

    // (a) B block: coeff output == K[Pᵀ]
    for (s, v) in ks_exact.i((.., .., 1)).iter().enumerate() {
        let w = k_coeff.i((.., .., 0)).iter().nth(s).unwrap();
        assert!((w - v).abs() < 1e-10,
            "coeff K[Pᵀ][{}] = {} vs exact {}", s, w, v);
    }
    // (b) A block: transpose of coeff output == K[P]
    let k_exact0 = ks_exact.i((.., .., 0));
    let nao2 = nao * nao;
    for r in 0..nao { for c in 0..nao {
        assert!((k_coeff[[r, c, 0]] - k_exact0[[c, r]]).abs() < 1e-10,
            "K[P][{},{}] via transpose = {} vs exact {}",
            c, r, k_coeff[[r, c, 0]], k_exact0[[c, r]]);
    }}
    let _ = nao2;
}

/// B-block exchange derivation (ao_kernel_block `is_b` branch):
///   C_occᵀ·K[Pᵀ]·C_vir  ==  (C_virᵀ·K[P]·C_occ)ᵀ
/// i.e. the transpose-identity formula vs the literal transposed-density route.
/// Holds because each M_Q is symmetric (K[Pᵀ] = K[P]ᵀ).
#[test]
fn test_b_exchange_transpose_identity() {
    let nao = 6; let occ = 3; let vir = 4; let naux = 4;
    let rimatr = synthetic_rimatr(nao, naux);
    let (ri3fn, _, _) = rimatr.as_ref().unwrap();
    let device = DeviceBLAS::default();
    let cderi = ri3fn.to_rstsr_view(&device);
    let c_occ = MatrixFull::from_vec([nao, occ], pseudo(nao * occ, 21.0)).unwrap();
    let c_vir = MatrixFull::from_vec([nao, vir], pseudo(nao * vir, 22.0)).unwrap();
    let p = MatrixFull::from_vec([nao, nao], pseudo(nao * nao, 23.0)).unwrap();
    let p_t = p.clone().transpose_and_drop();

    // literal route: K[Pᵀ] contracted as C_occᵀ K C_vir
    let k_pt = get_vk_ri_incore_dm(
        cderi.view(),
        vec![p_t].as_slice().to_rstsr(&device).view(),
        naux,
    );
    let k_pt_m = MatrixFull::from_vec([nao, nao], k_pt.raw()[..nao * nao].to_vec()).unwrap();
    let ref_b = contract_back(&k_pt_m, &c_occ, &c_vir);

    // new derivation: K[P], then (C_virᵀ K C_occ)ᵀ with the swapped-role contraction
    let k_p = get_vk_ri_incore_dm(
        cderi.view(),
        vec![p].as_slice().to_rstsr(&device).view(),
        naux,
    );
    let k_p_m = MatrixFull::from_vec([nao, nao], k_p.raw()[..nao * nao].to_vec()).unwrap();
    let cv = contract_back(&k_p_m, &c_vir, &c_occ); // flat a + i*vir

    let mut max_d = 0.0_f64;
    for i in 0..occ {
        for a in 0..vir {
            let d = (cv[a + i * vir] - ref_b[i + a * occ]).abs();
            if d > max_d {
                max_d = d;
            }
        }
    }
    assert!(max_d < 1e-10, "B-block exchange identity violated: max|D| = {max_d:e}");
}

