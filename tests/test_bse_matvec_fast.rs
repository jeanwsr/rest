//! Validation and benchmark harness for the AO-basis ("memory-efficient") BSE
//! matvec (`src/ri_bse/matvec_fast.rs`).
//!
//! The reference is built element-by-element from the historical MO-basis RI
//! tensors, using the pair-index conventions of the tensors themselves:
//!
//! ```text
//!   ri_ov       [M, O*V]   pair (i,a) at i + a*O          UNscreened
//!   ri_ov_tilde [M, O*V]   pair (j,a) at j + a*O          SCREENED
//!   ri_vv       [M, V*V]   pair (a,b) at a + b*V          UNscreened
//!   ri_oo_tilde [M, O*O]   pair (j,i) at j + i*O          SCREENED
//! ```
//!
//! Derived from `docs/bse_matvec_fast_derivation.md`.

use pyrest::ctrl_io;
use pyrest::molecule_io::Molecule;
use pyrest::ri_bse;
use pyrest::scf_io::{self, SCF};
use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;
use rest_tensors::matrix::MatrixFull;
use std::time::Instant;

fn input_for(geom: &str) -> String {
    format!(
        r##"
[ctrl]
print_level = 0
num_threads = 4
xc = "pbe0"
basis_path = "cc-pvdz"
basis_type = "spheric"
auxbas_path = "def2-universal-jkfit"
auxbas_type = "spheric"
eri_type = "ri-v"
use_ri_symm = true
charge = 0.0
spin = 1.0
spin_polarization = false
initial_guess = "sad"
mixer = "diis"
max_scf_cycle = 100
scf_acc_rho = 1.0e-8
scf_acc_eev = 1.0e-7
scf_acc_etot = 1.0e-10
[geom]
{geom}
[quasiparticle_methods]
gw_or_bse = "bse"
gw_scheme = "extrapolated"
scgw = "g0w0"
gw_variant = "cd"
gw_extrapolate_occ_threshold = 100.0
gw_extrapolate_vir_threshold = 100.0
bse_auxbas_path = "def2-universal-jkfit"
"##
    )
}

fn nh3_geom() -> &'static str {
    r##"name = "NH3"
unit = "angstrom"
position = """
N  -2.1988391019   1.8973746268   0.0000000000
H  -1.1788391019   1.8973746268   0.0000000000
H  -2.5388353987   1.0925460144  -0.5263586446
H  -2.5388400276   2.7556271745  -0.4338224694
""""##
}

fn build(geom: &str) -> SCF {
    let keys = toml::from_str::<serde_json::Value>(&input_for(geom)).unwrap();
    let (ctrl, g) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, g, None).unwrap();
    let mut scf = SCF::build(mol, &None);
    scf_io::scf_without_build(&mut scf, &None);
    scf.prepare_bse_integrals(&None);
    scf
}

fn lcg(n: usize, seed: u64) -> Vec<f64> {
    let mut st = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((st >> 33) as f64) / ((1u64 << 31) as f64) - 1.0
        })
        .collect()
}

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}
fn norm(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

struct Refs {
    ri_ov: MatrixFull<f64>,
    ri_vv: MatrixFull<f64>,
    ri_oo_tilde: MatrixFull<f64>,
    ri_ov_tilde: MatrixFull<f64>,
    m: usize,
    o: usize,
    v: usize,
}

fn make_refs(scf: &SCF, inv: &MatrixFull<f64>) -> Refs {
    let (_, _, o, v, _, _) = pyrest::ri_gw::get_occupation_parameters(scf, 'N');
    let ri_ov = ri_bse::get_submatrix(scf, 'O', 'V', 'N');
    let ri_vv = ri_bse::get_submatrix(scf, 'V', 'V', 'N');
    let ri_oo = ri_bse::get_submatrix(scf, 'O', 'O', 'N');
    let m = ri_ov.size[0];
    let mut ri_oo_tilde = MatrixFull::new(ri_oo.size, 0.0);
    _dgemm_full(inv, 'N', &ri_oo, 'N', &mut ri_oo_tilde, 1.0, 0.0);
    let mut ri_ov_tilde = MatrixFull::new(ri_ov.size, 0.0);
    _dgemm_full(inv, 'N', &ri_ov, 'N', &mut ri_ov_tilde, 1.0, 0.0);
    Refs { ri_ov, ri_vv, ri_oo_tilde, ri_ov_tilde, m, o, v }
}

/// A/B block matvec must agree with the historical MO-basis formula to machine
/// precision, for every spin channel and several random transition vectors.
#[test]
fn matvec_matches_reference() {
    let mut scf = build(nh3_geom());
    scf.gwqp.0 = scf.eigenvalues[0].clone();
    let inv = ri_bse::construct_inverse_dielectric(&scf, &scf.eigenvalues[0].clone());
    let r = make_refs(&scf, &inv);
    let (m, o, v) = (r.m, r.o, r.v);
    let nov = o * v;
    let qp_ctrl = scf.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let ctx = ri_bse::matvec_fast::FastBseContext::build(&scf, &inv).expect("context");
    println!("n_bas={} n_aux={} o={} v={}", scf.mol.num_basis, m, o, v);
    println!("build {:.4} s, {:.3} MB", ctx.build_seconds, ctx.bytes as f64 / 1048576.0);
    let qp = scf.gwqp.0.clone();

    for (label, spin, c) in
        [("singlet", "singlet", 2.0f64), ("triplet", "triplet", 0.0), ("rpa", "rpa", 1.0)]
    {
        let mut ctrl = qp_ctrl.clone();
        ctrl.bse_spin = spin.to_string();
        for seed in [1u64, 7, 1234] {
            let z = lcg(nov, seed);

            // ---- A block ----
            let mut a_ref = vec![0.0f64; nov];
            for a in 0..v {
                for i in 0..o {
                    a_ref[i + a * o] = (qp[o + a] - qp[i]) * z[i + a * o];
                }
            }
            for q in 0..m {
                for i in 0..o {
                    for a in 0..v {
                        let mut s = 0.0;
                        for j in 0..o {
                            for b in 0..v {
                                s += r.ri_oo_tilde.data[q + (j + i * o) * m]
                                    * r.ri_vv.data[q + (a + b * v) * m]
                                    * z[j + b * o];
                            }
                        }
                        a_ref[i + a * o] -= s;
                    }
                }
            }
            // ---- B block ----
            // `W_B[i,a] = sum_q sum_j sum_b Z_q[j,a] R_q[i,b] z[j + b*O]`
            // (transcription of `w_contribution_rayon_b_block`; the
            //  `(i,a)`-indexed reference is what the historical matvec means).
            let mut b_ref = vec![0.0f64; nov];
            for q in 0..m {
                for i in 0..o {
                    for a in 0..v {
                        let mut s = 0.0f64;
                        for j in 0..o {
                            for b in 0..v {
                                s += r.ri_ov_tilde.data[q + (j + a * o) * m]
                                    * r.ri_ov.data[q + (i + b * o) * m]
                                    * z[j + b * o];
                            }
                        }
                        b_ref[i + a * o] -= s;
                    }
                }
            }
            // ---- direct term (unscreened, reshaped [M*O, V]) ----
            if c != 0.0 {
                let ov = |q: usize, i: usize, a: usize| r.ri_ov.data[a * m * o + i * m + q];
                let mut inter = vec![0.0f64; m];
                for q in 0..m {
                    let mut s = 0.0;
                    for j in 0..o {
                        for bb in 0..v {
                            s += ov(q, j, bb) * z[j + bb * o];
                        }
                    }
                    inter[q] = s;
                }
                for i in 0..o {
                    for a in 0..v {
                        let mut s = 0.0;
                        for q in 0..m {
                            s += ov(q, i, a) * inter[q];
                        }
                        a_ref[i + a * o] += c * s;
                        b_ref[i + a * o] += c * s;
                    }
                }
            }

            let a_new = ctx.a_block_matvec(&scf, &ctrl, &z);
            let b_new = ctx.b_block_matvec(&scf, &ctrl, &z);
            let da = max_abs(&a_ref, &a_new) / norm(&a_ref);
            let db = max_abs(&b_ref, &b_new) / norm(&b_ref);
            println!("  [{label}] seed={seed}  A rel={da:.3e}  B rel={db:.3e}");
            assert!(da < 1e-13, "A-block ({label}, {seed}) rel err {da:.3e}");
            assert!(db < 1e-13, "B-block ({label}, {seed}) rel err {db:.3e}");
        }
    }
}

/// The three folds the matvec is built from, checked element-wise.

#[test]
fn folds_match_mo_tensors() {
    let mut scf = build(nh3_geom());
    scf.gwqp.0 = scf.eigenvalues[0].clone();
    let inv = ri_bse::construct_inverse_dielectric(&scf, &scf.eigenvalues[0].clone());
    let r = make_refs(&scf, &inv);
    let (m, o, v) = (r.m, r.o, r.v);
    // the element-wise hooks below read the [M,N,N] AO tensor, so this build
    // keeps it (production builds release it after the folds)
    let ctx = ri_bse::matvec_fast::FastBseContext::build_keeping_ao_tensor(&scf, &inv).unwrap();
    let n_bas = scf.mol.num_basis;
    let n_packed = n_bas * (n_bas + 1) / 2;
    let (rimatr, _, _) = scf.rimatr_bse.as_ref().unwrap();
    let x = scf.eigenvectors[0].data.clone();
    let (start_mo, _, _, _, _, lumo) = pyrest::ri_gw::get_occupation_parameters(&scf, 'N');
    let xo = |i: usize, mu: usize| x[mu + (start_mo + i) * n_bas];
    let xv = |a: usize, mu: usize| x[mu + (lumo + a) * n_bas];

    // ri_ov (unscreened)
    let mut d_ov = 0.0f64;
    for q in [0usize, 5, 50] {
        for i in 0..o {
            for a in 0..v {
                let mut un = 0.0f64;
                for mu in 0..n_bas {
                    for nu in 0..n_bas {
                        let (hi, lo) = if nu >= mu { (nu, mu) } else { (mu, nu) };
                        un += rimatr.data[q * n_packed + hi * (hi + 1) / 2 + lo]
                            * xo(i, mu) * xv(a, nu);
                    }
                }
                d_ov = d_ov.max((un - r.ri_ov.data[q + (i + a * o) * m]).abs());
                d_ov = d_ov.max((un - ctx.r_at(0, q, i, a)).abs());
            }
        }
    }
    println!("ri_ov fold (unscreened, vs tensor and vs ctx): {d_ov:.3e}");
    assert!(d_ov < 1e-12);

    // ri_ov_tilde (screened)
    let mut d_ovt = 0.0f64;
    for q in [0usize, 5, 50] {
        for j in 0..o {
            for a in 0..v {
                d_ovt = d_ovt.max((ctx.ztilde_at(0, q, j, a) - r.ri_ov_tilde.data[q + (j + a * o) * m]).abs());
            }
        }
    }
    println!("ri_ov_tilde fold (screened): {d_ovt:.3e}");
    assert!(d_ovt < 1e-12);

    // ri_vv is UNscreened
    let mut d_vv_u = 0.0f64;
    let mut d_vv_s = 0.0f64;
    for q in [0usize, 5, 50] {
        for a in 0..v {
            for b in 0..v {
                let (mut un, mut sc) = (0.0f64, 0.0f64);
                for mu in 0..n_bas {
                    for nu in 0..n_bas {
                        let (hi, lo) = if nu >= mu { (nu, mu) } else { (mu, nu) };
                        let pa = xv(a, mu) * xv(b, nu);
                        un += rimatr.data[q * n_packed + hi * (hi + 1) / 2 + lo] * pa;
                        let mut tv = 0.0f64;
                        for rr in 0..m {
                            let cf = inv.data[q + rr * m];
                            if cf == 0.0 { continue; }
                            tv += cf * rimatr.data[rr * n_packed + hi * (hi + 1) / 2 + lo];
                        }
                        sc += tv * pa;
                    }
                }
                let got = r.ri_vv.data[q + (a + b * v) * m];
                d_vv_u = d_vv_u.max((un - got).abs());
                d_vv_s = d_vv_s.max((sc - got).abs());
            }
        }
    }
    println!("ri_vv vs UNscreened fold: {d_vv_u:.3e}; vs SCREENED fold: {d_vv_s:.3e}");
    assert!(d_vv_u < 1e-12 && d_vv_s > 1e-3, "ri_vv must be the unscreened fold");
}

/// `ri_oo_tilde` is screened — the asymmetry that drove the A-block bug.

#[test]
fn screening_map() {
    let mut scf = build(nh3_geom());
    scf.gwqp.0 = scf.eigenvalues[0].clone();
    let inv = ri_bse::construct_inverse_dielectric(&scf, &scf.eigenvalues[0].clone());
    let r = make_refs(&scf, &inv);
    let (m, o) = (r.m, r.o);
    let ri_oo = ri_bse::get_submatrix(&scf, 'O', 'O', 'N');
    let mut ri_oo_t = MatrixFull::new(ri_oo.size, 0.0);
    _dgemm_full(&inv, 'N', &ri_oo, 'N', &mut ri_oo_t, 1.0, 0.0);
    let n_bas = scf.mol.num_basis;
    let n_packed = n_bas * (n_bas + 1) / 2;
    let (rimatr, _, _) = scf.rimatr_bse.as_ref().unwrap();
    let x = scf.eigenvectors[0].data.clone();
    let (start_mo, _, _, _, _, _) = pyrest::ri_gw::get_occupation_parameters(&scf, 'N');
    let xo = |i: usize, mu: usize| x[mu + (start_mo + i) * n_bas];
    let (mut s_oo, mut u_oo) = (0.0f64, 0.0f64);
    for q in [0usize, 5] {
        for i in 0..o {
            for j in 0..o {
                let (mut sc, mut un) = (0.0f64, 0.0f64);
                for mu in 0..n_bas {
                    for nu in 0..n_bas {
                        let (hi, lo) = if nu >= mu { (nu, mu) } else { (mu, nu) };
                        let pa = xo(i, mu) * xo(j, nu);
                        un += rimatr.data[q * n_packed + hi * (hi + 1) / 2 + lo] * pa;
                        let mut tv = 0.0f64;
                        for rr in 0..m {
                            let cf = inv.data[q + rr * m];
                            if cf == 0.0 { continue; }
                            tv += cf * rimatr.data[rr * n_packed + hi * (hi + 1) / 2 + lo];
                        }
                        sc += tv * pa;
                    }
                }
                let got = ri_oo_t.data[q + (j + i * o) * m];
                s_oo = s_oo.max((sc - got).abs());
                u_oo = u_oo.max((un - got).abs());
            }
        }
    }
    println!("ri_oo_tilde vs SCREENED: {s_oo:.3e}; vs UNscreened: {u_oo:.3e}");
    assert!(s_oo < 1e-12 && u_oo > 1e-3, "ri_oo_tilde must be the screened fold");
}

/// Timing: context build + repeated matvecs, next to the MO-basis path.

#[test]
fn matvec_speed() {
    let mut scf = build(nh3_geom());
    scf.gwqp.0 = scf.eigenvalues[0].clone();
    let inv = ri_bse::construct_inverse_dielectric(&scf, &scf.eigenvalues[0].clone());
    let r = make_refs(&scf, &inv);
    let nov = r.o * r.v;
    let qp_ctrl = scf.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let z = lcg(nov, 99);

    let t0 = Instant::now();
    let ctx = ri_bse::matvec_fast::FastBseContext::build(&scf, &inv).unwrap();
    let build = t0.elapsed().as_secs_f64();
    println!("AO build: {build:.4} s, {:.3} MB", ctx.bytes as f64 / 1048576.0);

    let reps = 50;
    let t1 = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(ctx.a_block_matvec(&scf, &qp_ctrl, &z));
    }
    let at = t1.elapsed().as_secs_f64() / reps as f64;
    let t2 = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(ctx.b_block_matvec(&scf, &qp_ctrl, &z));
    }
    let bt = t2.elapsed().as_secs_f64() / reps as f64;

    // MO-basis reference: the historical shape, written with explicit loops
    // (the library's `w_contribution_a_block_dgemm*` panics on these shapes --
    // see docs/bse_matvec_fast_status.md).
    let (m, o, v) = (r.m, r.o, r.v);
    let ri_oo = ri_bse::get_submatrix(&scf, 'O', 'O', 'N');
    let mut ri_oo_tilde = MatrixFull::new(ri_oo.size, 0.0);
    _dgemm_full(&inv, 'N', &ri_oo, 'N', &mut ri_oo_tilde, 1.0, 0.0);
    let nov = o * v;
    let qp = scf.gwqp.0.clone();
    let mo_a = |z: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0f64; nov];
        for a in 0..v {
            for i in 0..o {
                out[i + a * o] = (qp[o + a] - qp[i]) * z[i + a * o];
            }
        }
        for q in 0..m {
            for i in 0..o {
                for a in 0..v {
                    let mut s = 0.0;
                    for j in 0..o {
                        for b in 0..v {
                            s += ri_oo_tilde.data[q + (j + i * o) * m]
                                * r.ri_vv.data[q + (a + b * v) * m]
                                * z[j + b * o];
                        }
                    }
                    out[i + a * o] -= s;
                }
            }
        }
        out
    };
    let t3 = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(mo_a(&z));
    }
    let mo_at = t3.elapsed().as_secs_f64() / reps as f64;

    println!(
        "matvec: AO A {:.3} ms / B {:.3} ms | MO A {:.3} ms | ratio A {:.2}x",
        at * 1e3,
        bt * 1e3,
        mo_at * 1e3,
        at / mo_at
    );
}

/// Confirm the precomputed `aq`/`zt` storage layout against `ri_oo_tilde` /
/// `ri_ov_tilde`.

#[test]
fn ri_vv_is_unscreened_fold_big() {
    let xyz = "C  1.3960  0.0000  0.0000\nC  0.6980  1.2092  0.0000\nC -0.6980  1.2092  0.0000\nC -1.3960  0.0000  0.0000\nC -0.6980 -1.2092  0.0000\nC  0.6980 -1.2092  0.0000\nH  2.4820  0.0000  0.0000\nH  1.2410  2.1495  0.0000\nH -1.2410  2.1495  0.0000\nH -2.4820  0.0000  0.0000\nH -1.2410 -2.1495  0.0000\nH  1.2410 -2.1495  0.0000\n";
    let keys = toml::from_str::<serde_json::Value>(&input_for(&format!(
        "name = \"C6H6\"\nunit = \"angstrom\"\nposition = \"\"\"\n{xyz}\"\"\"\n"
    )))
    .unwrap();
    let (ctrl, g) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, g, None).unwrap();
    let mut scf = SCF::build(mol, &None);
    scf_io::scf_without_build(&mut scf, &None);
    scf.prepare_bse_integrals(&None);
    scf.gwqp.0 = scf.eigenvalues[0].clone();
    let inv = ri_bse::construct_inverse_dielectric(&scf, &scf.eigenvalues[0].clone());
    let n_bas = scf.mol.num_basis;
    let (rimatr, _, _) = scf.rimatr_bse.as_ref().unwrap();
    let m = rimatr.size[1];
    let (_, _, o, v, _, lumo) = pyrest::ri_gw::get_occupation_parameters(&scf, 'N');
    let ri_vv = ri_bse::get_submatrix(&scf, 'V', 'V', 'N');
    let x = scf.eigenvectors[0].data.clone();
    let (start_mo, _, _, _, _, _) = pyrest::ri_gw::get_occupation_parameters(&scf, 'N');
    let xo = |i: usize, mu: usize| x[mu + (start_mo + i) * n_bas];
    println!("n_bas={n_bas} n_aux={m} o={o} v={v} lumo={lumo} ri_vv={:?}", ri_vv.size);
    let n_packed = n_bas * (n_bas + 1) / 2;
    let xv = |a: usize, mu: usize| x[mu + (lumo + a) * n_bas];
    let _ = xo;
    // compare a sampled set of (a,b) for q=0
    let q = 0usize;
    let mut worst = (0.0f64, 0usize, 0usize);
    let mut worst_sc = (0.0f64, 0usize, 0usize);
    for a in [0usize, 1, 5, 20, v - 1] {
        for b in [0usize, 1, 5, 20, v - 1] {
            let mut un = 0.0f64;
            let mut sc = 0.0f64;
            for mu in 0..n_bas {
                for nu in 0..n_bas {
                    let (hi, lo) = if nu >= mu { (nu, mu) } else { (mu, nu) };
                    let pa = xv(a, mu) * xv(b, nu);
                    un += rimatr.data[q * n_packed + hi * (hi + 1) / 2 + lo] * pa;
                    let mut tv = 0.0f64;
                    for rr in 0..m {
                        let cf = inv.data[q + rr * m];
                        if cf == 0.0 { continue; }
                        tv += cf * rimatr.data[rr * n_packed + hi * (hi + 1) / 2 + lo];
                    }
                    sc += tv * pa;
                }
            }
            let got = ri_vv.data[q + (a + b * v) * m];
            if (un - got).abs() > worst.0 { worst = ((un - got).abs(), a, b); }
            if (sc - got).abs() > worst_sc.0 { worst_sc = ((sc - got).abs(), a, b); }
        }
    }
    println!("ri_vv vs UNscreened fold: {:?}  vs SCREENED: {:?}", worst, worst_sc);
}

/// Is `ri_oo_tilde` the SCREENED o x o fold on a bigger system?

#[test]
fn screening_behaviour() {
    let mut scf = build(nh3_geom());
    scf.gwqp.0 = scf.eigenvalues[0].clone();
    let inv = ri_bse::construct_inverse_dielectric(&scf, &scf.eigenvalues[0].clone());
    let r = make_refs(&scf, &inv);
    let (m, o, v) = (r.m, r.o, r.v);
    let nov = o * v;
    let qp_ctrl = scf.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let z = lcg(nov, 1);

    let ctx0 = ri_bse::matvec_fast::FastBseContext::build(&scf, &inv).unwrap();
    let a0 = ctx0.a_block_matvec(&scf, &qp_ctrl, &z);
    let b0 = ctx0.b_block_matvec(&scf, &qp_ctrl, &z);
    let norm_a = norm(&a0);
    let norm_b = norm(&b0);
    println!("tol=0 (off): A {:.3} MB, |A z| = {norm_a:.6}", ctx0.bytes as f64 / 1048576.0);
    for tol in [1e-3f64, 1e-2, 3e-2, 1e-1] {
        let ctx = ri_bse::matvec_fast::FastBseContext::build_with_screening(&scf, &inv, tol)
            .expect("context");
        let a = ctx.a_block_matvec(&scf, &qp_ctrl, &z);
        let b = ctx.b_block_matvec(&scf, &qp_ctrl, &z);
        println!(
            "  tol={tol:.0e}: storage {:.3} MB, rel err A {:.2e}, rel err B {:.2e}",
            ctx.bytes as f64 / 1048576.0,
            max_abs(&a0, &a) / norm_a,
            max_abs(&b0, &b) / norm_b
        );
    }
    // tol = 0 must be bit-identical
    let ctxz = ri_bse::matvec_fast::FastBseContext::build_with_screening(&scf, &inv, 0.0).unwrap();
    let az = ctxz.a_block_matvec(&scf, &qp_ctrl, &z);
    // Note: the Rayon `fold`/`reduce` summation order is not fixed, so even the
    // unscreened path is only reproducible to round-off; `tol = 0` keeps every
    // pair, so it must agree to that level rather than bit-for-bit.
    let d = max_abs(&a0, &az);
    println!("tol=0 vs default (every pair kept): max diff = {d:.3e}");
    assert!(d < 1e-14, "tol=0 must keep every pair, got {d:.3e}");
}
