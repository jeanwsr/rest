//! Phase-resolved diagnostic for the full-CD engine's non-frontier-orbital
//! gradient defect.
//!
//! The static-subtracted CD self-energy is a linear functional of three
//! response objects,
//!
//!     Sigma_sub = sum_m [ sum_wi a_{wi,m} W_m(iu_wi) + staticc_m W_m(0)
//!                         + sum_r s_r W_m(zeta_r) ],
//!
//! so its derivative splits into an imaginary-axis phase, a static phase and a
//! residue phase.  (The residue phase used to index its `proj = y_res^T qia`
//! product with a row-major stride although `gemm_nt` returns a column-major
//! buffer; that only showed up with two or more active residues.)  This test rebuilds each phase from the public cache fields,
//! finite-differences it at a fixed external frequency, and compares with the
//! correspondingly masked analytic pullback (`GwGradConfig::phase_mask`).

use pyrest::ri_gw::gw_grad::{CdQpCache, GwCdGradEngine, GwGradConfig};
use pyrest::scf_io::SCF;

const H2O: &str = "
        O     0.000000000000     0.000000000000     0.117300000000
        H     0.000000000000     0.757200000000    -0.469200000000
        H     0.000000000000    -0.757200000000    -0.469200000000
    ";

fn build_mol(name: &str, geom: &str) -> SCF {
    use pyrest::ctrl_io;
    use pyrest::molecule_io::Molecule;
    use pyrest::scf_io;

    let input_token = format!(
        r##"
[ctrl]
     print_level =          0
     xc =                   "hf"
     basis_path =           "sto-3g"
     charge =               0.0
     spin =                 1.0
     spin_polarization =    false
     initial_guess =        "hcore"
     mixer =                "diis"
     num_threads =          1
     scf_acc_rho =          1.0e-12
     scf_acc_eev =          1.0e-10
     scf_acc_etot =         1.0e-14

[geom]
    name = "{}"
    unit = "Angstrom"
    position = """{}"""
"##,
        name, geom
    );
    let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = SCF::build(mol, &None);
    scf_io::scf_without_build(&mut scf_data, &None);
    scf_data
}

fn set_gw_ctrl(scf: &mut SCF) {
    scf.mol.ctrl.quasiparticle_methods =
        Some(pyrest::ctrl_io::quasiparticle_methods::QuasiParticle {
            gw_scheme: "extrapolated".to_string(),
            use_low_rank_contour: false,
            low_rank_tolerance: 1.0e-10,
            nomega_chi_real: 6,
            ..Default::default()
        });
}

fn cfg_of(scf: &SCF) -> GwGradConfig {
    let mut cfg = GwGradConfig::from_scf(scf);
    cfg.qpe_tol = 1.0e-13;
    cfg
}

fn displaced(scf_data: &SCF, atm: usize, comp: usize, h: f64) -> SCF {
    let mut new_scf = scf_data.clone();
    let mut vec_xyz = vec![0.0f64; 3];
    vec_xyz[comp] = h;
    new_scf.mol.geom.geom_shift(atm, vec_xyz);
    new_scf.mol.ctrl.print_level = 0;
    new_scf.mol.ctrl.initial_guess = String::from("inherit");
    pyrest::scf_io::initialize_scf(&mut new_scf, &None);
    pyrest::scf_io::scf_without_build(&mut new_scf, &None);
    new_scf
}

/// The five pieces of `Sigma_sub(omega)` rebuilt from the public cache
/// fields, mirroring the coefficient tables of `cd_qp_pullback`:
/// `(E_imag, E_W0_a, E_static_bare, E_res, E_W0_s)` with
/// `Sigma = E_imag - E_W0_a + E_static_bare + E_res - E_W0_s`.
fn phases(eng: &GwCdGradEngine, cache: &CdQpCache) -> [f64; 5] {
    let base = &eng.base;
    let e = &base.e;
    let nocc = base.nocc;
    let nmo = base.nmo;
    let naux = base.naux;
    let eta = base.config.eta;
    let ef = 0.5 * (e[nocc - 1] + e[nocc]);
    let omega = cache.omega;
    let n = cache.target;
    let pi = std::f64::consts::PI;

    let mut active = vec![0.0f64; nmo];
    for r in &cache.residues {
        active[r.m] += r.s;
    }
    let mut e_imag = 0.0f64;
    let mut e_w0a = 0.0f64;
    let mut e_static = 0.0f64;
    let mut e_res = 0.0f64;
    let mut e_w0s = 0.0f64;
    for m in 0..nmo {
        let f_m = if m < nocc { 1.0 } else { 0.0 };
        let tr = omega - e[m];
        let ti = -eta * (ef - e[m]).signum();
        let (t2r, t2i) = (tr * tr - ti * ti, 2.0 * tr * ti);
        let mut sum_a = (0.0f64, 0.0f64);
        for (wi, &(u, wt)) in base.quad.iter().enumerate() {
            let (dr, di) = (t2r + u * u, t2i);
            let inv = 1.0 / (dr * dr + di * di);
            // kk = t / (t^2 + u^2)
            let kkr = (tr * dr + ti * di) * inv;
            let kki = (ti * dr - tr * di) * inv;
            let a = (-wt / pi * kkr, -wt / pi * kki);
            sum_a.0 += a.0;
            sum_a.1 += a.1;
            e_imag += a.0 * cache.w_imag[wi][m];
        }
        e_w0a += sum_a.0 * cache.w0[m].re - sum_a.1 * cache.w0[m].im;
        e_static += (0.5 - f_m) * cache.w0[m].re;
        e_w0s += active[m] * cache.w0[m].re;
    }
    for r in &cache.residues {
        let col = (n + r.m * nmo) * naux;
        let qcol = &base.q[col..col + naux];
        let mut wr = 0.0f64;
        for p in 0..naux {
            wr += qcol[p] * (r.y[p].re - cache.y0_metric[r.m * naux + p]);
        }
        e_res += r.s * wr;
    }
    [e_imag, e_w0a, e_static, e_res, e_w0s]
}

/// Masked functional `Phi = d_i (E_imag - E_W0a) + d_s E_static + d_r (E_res - E_W0s)`.
fn phi_mask(pieces: &[f64; 5], mask: u32) -> f64 {
    let mut v = 0.0;
    if mask & 1 != 0 {
        v += pieces[0] - pieces[1];
    }
    if mask & 2 != 0 {
        v += pieces[2];
    }
    if mask & 4 != 0 {
        v += pieces[3] - pieces[4];
    }
    v
}

#[test]
fn h2o_cd_phase_decomposition() {
    let mut scf_data = build_mol("H2O", H2O);
    set_gw_ctrl(&mut scf_data);
    let engine = GwCdGradEngine::new(&scf_data, cfg_of(&scf_data));
    let natm = scf_data.mol.geom.elem.len();
    let nmo = scf_data.eigenvalues[0].len();
    let h = 1.0e-4;
    let targets: Vec<usize> = match std::env::var("REST_CD_TARGET") {
        Ok(s) => vec![s.parse().unwrap()],
        Err(_) => (0..nmo).collect(),
    };

    for t in targets {
        let omega_ref = engine.qp_energy_of(t);
        let cache = engine.cache_at(omega_ref, t);
        let p0 = phases(&engine, &cache);
        let sigma_full = engine.sigma_sub_at(omega_ref, t).0;
        println!(
            "=== target {} omega {:.10}  Sigma {:.10}  pieces imag {:.10} w0a {:.10} static {:.10} res {:.10} w0s {:.10}",
            t, omega_ref, sigma_full, p0[0], p0[1], p0[2], p0[3], p0[4]
        );
        assert!(
            (phi_mask(&p0, 0b111) - sigma_full).abs() < 1.0e-9,
            "phase reconstruction mismatch for target {}: {:.3e}",
            t,
            phi_mask(&p0, 0b111) - sigma_full
        );

        let masks = [0u32, 1, 2, 4, 7];
        let mut masked: Vec<Vec<f64>> = Vec::new();
        for &mask in masks.iter() {
            let mut cfg = cfg_of(&scf_data);
            cfg.phase_mask = mask;
            let eng = GwCdGradEngine::new(&scf_data, cfg);
            let c = eng.cache_at(omega_ref, t);
            masked.push(eng.fixed_omega_gradient(&c));
        }

        let mut fd: Vec<Vec<f64>> = masks.iter().map(|_| vec![0.0f64; natm * 3]).collect();
        for atm in 0..natm {
            for comp in 0..3 {
                let sp = displaced(&scf_data, atm, comp, h);
                let sm = displaced(&scf_data, atm, comp, -h);
                let ep = GwCdGradEngine::new(&sp, cfg_of(&sp));
                let em = GwCdGradEngine::new(&sm, cfg_of(&sm));
                let pp = phases(&ep, &ep.cache_at(omega_ref, t));
                let pm = phases(&em, &em.cache_at(omega_ref, t));
                for (idx, &mask) in masks.iter().enumerate() {
                    let dphi = (ep.base.e[t] + phi_mask(&pp, mask))
                        - (em.base.e[t] + phi_mask(&pm, mask));
                    fd[idx][atm * 3 + comp] = dphi / (2.0 * h);
                }
            }
        }

        let names = ["none", "imag", "static", "res ", "all "];
        for (idx, ga) in masked.iter().enumerate() {
            let gf = &fd[idx];
            let mut worst = (0.0f64, 0usize);
            for i in 0..ga.len() {
                let d = (ga[i] - gf[i]).abs();
                let d = if d.is_nan() { f64::INFINITY } else { d };
                if d > worst.0 {
                    worst = (d, i);
                }
                println!(
                    "    [{} mask {:03b}] comp {:2}: ana {:+.8e} fd {:+.8e} d {:.2e}",
                    names[idx], masks[idx], i, ga[i], gf[i], d
                );
            }
            println!(
                "    [{} mask {:03b}] WORST {:.3e} at comp {}",
                names[idx], masks[idx], worst.0, worst.1
            );
            assert!(
                worst.0 < 1.0e-6,
                "target {} phase {} (mask {:03b}): analytic pullback does not match the \
                 finite difference of its own energy contribution: {:.3e} > 1e-6 at comp {}",
                t,
                names[idx],
                masks[idx],
                worst.0,
                worst.1
            );
        }
    }
}
