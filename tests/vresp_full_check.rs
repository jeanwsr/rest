//! Full-fxc validation: the density-space response operator
//! (`response_potential_batched`) must reproduce the amplitude-space kernel
//! (`a_matvec_ao_batched`) on real decks — transition densities built from
//! random amplitudes, contracted back, diagonal added.
//!
//! COMMENTED OUT: response API disabled (see matvec_ao.rs).
//! Kept as reference; re-enable by removing the outer cfg guard.
#![cfg(any())]
//! `#[ignore]`d by default: the decks live under /tmp/opencode/utddft (not
//! shipped with the repo). Run explicitly with
//! `cargo test --test vresp_full_check -- --ignored`.

use pyrest::molecule_io::Molecule;
use pyrest::ri_tddft::matvec_ao::{
    a_matvec_ao_batched, contract_back, response_potential_batched, transition_density,
};
use pyrest::ri_tddft::tddft::prepare_ao_data;
use pyrest::ri_tddft::utils::tddft_sector_params;
use pyrest::scf_io::{self, SCFType};
use rest_tensors::MatrixFull;

fn pseudo(n: usize, seed: f64) -> Vec<f64> {
    let mut x = seed;
    (0..n).map(|_| {
        x = (x * 91.7 + 3.1).sin() * 100.0;
        x.fract() - 0.5
    }).collect()
}

fn slice_rows(z_block: &MatrixFull<f64>, r0: usize, r1: usize) -> MatrixFull<f64> {
    let nrow = z_block.size[0];
    let m = z_block.size[1];
    let nr = r1 - r0;
    let mut out = MatrixFull::new([nr, m], 0.0);
    for s in 0..m {
        out.data[s * nr..(s + 1) * nr]
            .copy_from_slice(&z_block.data[s * nrow + r0..s * nrow + r1]);
    }
    out
}

fn check_deck(deck: &str, expect_uhf: bool) {
    println!("=== {deck} ===");
    let mol = Molecule::build(deck.to_string(), None).unwrap();
    let scf = scf_io::scf(mol, &None).unwrap();
    assert_eq!((scf.scftype == SCFType::UHF), expect_uhf);

    let xlet = 'R';
    let sectors = tddft_sector_params(&scf);
    let n_sec = sectors.len();
    let dim_total: usize = sectors.iter().map(|s| s.dim()).sum();
    let m = 4usize;

    // random amplitude block
    let mut z_block = MatrixFull::new([dim_total, m], 0.0);
    z_block.data.copy_from_slice(&pseudo(dim_total * m, 0.77));

    // prepared AO data (kernel tables, per-sector MO coefficients)
    let mut data = prepare_ao_data(&scf);
    assert_eq!(data.n_sectors(), n_sec);

    // ── density-space vresp on the transition densities ──
    let mut p_sectors: Vec<Vec<MatrixFull<f64>>> = Vec::with_capacity(n_sec);
    {
        let mut base = 0usize;
        for (i_sec, sec) in sectors.iter().enumerate() {
            let z_sec = slice_rows(&z_block, base, base + sec.dim());
            let ps: Vec<MatrixFull<f64>> = (0..m)
                .map(|s| {
                    let z: Vec<f64> = (0..sec.dim()).map(|r| z_sec[[r, s]]).collect();
                    transition_density(&data.c_occ[i_sec], &data.c_vir[i_sec], &z)
                })
                .collect();
            p_sectors.push(ps);
            base += sec.dim();
        }
    }
    let v1 = response_potential_batched(&mut data, &scf.rimatr, &p_sectors, xlet);
    assert_eq!(v1.len(), n_sec * m);

    // ── amplitude-space reference: kernel + per-sector diagonal ──
    let ref_amp = a_matvec_ao_batched(&scf, &mut data, &z_block, xlet);

    // compare per sector/column: contract_back(v1) + diag·z == ref rows
    let mut base = 0usize;
    let mut max_d = 0.0f64;
    let mut argmax = (0usize, 0usize, 0usize, 0usize);
    for (i_sec, sec) in sectors.iter().enumerate() {
        let ks = &scf.eigenvalues[i_sec];
        for s in 0..m {
            let amp = contract_back(&v1[i_sec * m + s], &data.c_occ[i_sec], &data.c_vir[i_sec]);
            for a in 0..sec.vir_size {
                for i in 0..sec.occ_size {
                    let idx = i + a * sec.occ_size;
                    let want = ref_amp[[base + idx, s]];
                    let got = amp[idx]
                        + (ks[sec.lumo + a] - ks[sec.start_mo + i]) * z_block[[base + idx, s]];
                    let d = (want - got).abs();
                    if d > max_d {
                        max_d = d;
                        argmax = (i_sec, s, i, a);
                    }
                }
            }
        }
        base += sec.dim();
    }
    println!("max |Δ| = {max_d:.3e} at (sector={}, s={}, i={}, a={})", argmax.0, argmax.1, argmax.2, argmax.3);

    // ── decomposition: locate the diverging contribution ──
    // (both paths honor the pub fields alpha_hybrid / fxc_driver)
    let alpha0 = data.alpha_hybrid;
    let driver0 = data.fxc_driver;

    // (a) K isolated: full − no-K(alpha=0)
    data.alpha_hybrid = 0.0;
    let v1_nok = response_potential_batched(&mut data, &scf.rimatr, &p_sectors, xlet);
    let ref_nok = a_matvec_ao_batched(&scf, &mut data, &z_block, xlet);
    data.alpha_hybrid = alpha0;
    let mut kmax = 0.0f64;
    for i_sec in 0..n_sec {
        for s in 0..m {
            let a_full = contract_back(&v1[i_sec * m + s], &data.c_occ[i_sec], &data.c_vir[i_sec]);
            let a_nok = contract_back(&v1_nok[i_sec * m + s], &data.c_occ[i_sec], &data.c_vir[i_sec]);
            let r_full = (0..sectors[i_sec].dim()).map(|r| ref_amp[[{
                let mut b = 0usize;
                for s2 in sectors.iter().take(i_sec) { b += s2.dim(); }
                b + r
            }, s]]).collect::<Vec<_>>();
            let r_nok = (0..sectors[i_sec].dim()).map(|r| ref_nok[[{
                let mut b = 0usize;
                for s2 in sectors.iter().take(i_sec) { b += s2.dim(); }
                b + r
            }, s]]).collect::<Vec<_>>();
            for idx in 0..sectors[i_sec].dim() {
                let dk_v = a_full[idx] - a_nok[idx];
                let dk_r = r_full[idx] - r_nok[idx];
                kmax = kmax.max((dk_v - dk_r).abs());
            }
        }
    }
    println!("K-isolated max |Δ| = {kmax:.3e}");

    // (b) fxc isolated: (no-K, fxc on) − (no-K, fxc off) — alpha stays 0
    data.fxc_driver = None;
    let v1_nofxc = response_potential_batched(&mut data, &scf.rimatr, &p_sectors, xlet);
    let ref_nofxc = a_matvec_ao_batched(&scf, &mut data, &z_block, xlet);
    data.fxc_driver = driver0;
    let mut jmax = 0.0f64;
    for i_sec in 0..n_sec {
        let b = sectors.iter().take(i_sec).map(|s2| s2.dim()).sum::<usize>();
        for s in 0..m {
            let a_nok = contract_back(&v1_nok[i_sec * m + s], &data.c_occ[i_sec], &data.c_vir[i_sec]);
            let a_nofxc = contract_back(&v1_nofxc[i_sec * m + s], &data.c_occ[i_sec], &data.c_vir[i_sec]);
            for idx in 0..sectors[i_sec].dim() {
                let d_fxc_v = a_nok[idx] - a_nofxc[idx];
                let d_fxc_r = ref_nok[[b + idx, s]] - ref_nofxc[[b + idx, s]];
                jmax = jmax.max((d_fxc_v - d_fxc_r).abs());
            }
        }
    }
    println!("fxc-isolated max |Δ| = {jmax:.3e}");
    assert!(max_d < 1e-10, "vresp != amplitude kernel: {max_d:e}");
}

#[test]
#[ignore]
fn vresp_matches_kernel_rhf_h2() {
    check_deck("/tmp/opencode/utddft/h2_rks_tda_ao/ctrl.in", false);
}

#[test]
#[ignore]
fn vresp_matches_kernel_uks_h2o() {
    check_deck("/tmp/opencode/utddft/h2o_uks_tda_ao/ctrl.in", true);
}

#[test]
#[ignore]
fn vresp_matches_kernel_uks_h2o_dm_driver() {
    check_deck("/tmp/opencode/utddft/h2o_uks_tda_ao_dm/ctrl.in", true);
}

#[test]
#[ignore]
fn vresp_matches_kernel_uks_h2o_lda_sem() {
    check_deck("/tmp/opencode/utddft/h2o_uks_lda_sem/ctrl.in", true);
}

#[test]
#[ignore]
fn vresp_matches_kernel_uks_h2o_nobatch_sem() {
    check_deck("/tmp/opencode/utddft/h2o_uks_nobatch_sem/ctrl.in", true);
}

#[test]
#[ignore]
fn vresp_matches_kernel_uks_nh2() {
    check_deck("/tmp/opencode/utddft/nh2_uks_tda_ao/ctrl.in", true);
}
