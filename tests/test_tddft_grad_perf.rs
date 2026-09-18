//! TDDFT/TDA analytic-gradient performance twin of
//! `bench/tddft_grad_perf_pyscf.py` (requirement 3: speed and memory must not be
//! significantly worse than PySCF).
//!
//! Same molecules / basis / auxbasis, BLYP, singlet TDA, nstates = 3, one thread.
//! Reports SCF, TDDFT and TDDFT-gradient wall times plus peak RSS.
//!
//! Opt-in (slow): `REST_TDDFT_GRAD_PERF=1 cargo test -p rest --test
//! test_tddft_grad_perf -- --nocapture --test-threads=1`

use pyrest::ctrl_io;
use pyrest::grad::rhf::RIRHFGradient;
use pyrest::molecule_io::Molecule;
use pyrest::ri_tddft::{tddft_main, TddftGradEngine};
use pyrest::scf_io::{self, SCF};
use std::time::Instant;

// Ordered smallest -> largest: `peak_rss_mb()` reports a process-wide
// high-water mark, so a large case early in the list would inflate every
// later one.
const CASES: [(&str, &str); 5] = [
    (
        "h2o_defs",
        "O 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692",
    ),
    (
        "h2co_defs",
        "C 0.0 0.0 0.0\nO 0.0 0.0 1.2074\nH 0.0 0.9418 -0.5853\nH 0.0 -0.9418 -0.5853",
    ),
    (
        "c2h4_defs",
        "C 0.6695 0.0 0.0\nC -0.6695 0.0 0.0\nH 1.2301 0.9289 0.0\n\
         H 1.2301 -0.9289 0.0\nH -1.2301 0.9289 0.0\nH -1.2301 -0.9289 0.0",
    ),
    (
        "c4h6_defs",
        "C -1.462772 1.122980 0.0\nC -0.733500 0.0 0.0\nC 0.733500 0.0 0.0\n\
         C 1.462772 -1.122980 0.0\nH -0.969284 2.091504 0.0\nH -2.548282 1.066091 0.0\n\
         H -1.252172 -0.955274 0.0\nH 1.252172 0.955274 0.0\nH 0.969284 -2.091504 0.0\n\
         H 2.548282 -1.066091 0.0",
    ),
    (
        "c10h8_defs",
        "C -1.2124 1.4000 0.0\nC -2.4248 0.7000 0.0\nC -2.4248 -0.7000 0.0\n\
         C -1.2124 -1.4000 0.0\nC 0.0000 -0.7000 0.0\nC 0.0000 0.7000 0.0\n\
         C 1.2124 1.4000 0.0\nC 2.4248 0.7000 0.0\nC 2.4248 -0.7000 0.0\n\
         C 1.2124 -1.4000 0.0\nH -1.2124 2.4800 0.0\nH -3.3601 1.2400 0.0\n\
         H -3.3601 -1.2400 0.0\nH -1.2124 -2.4800 0.0\nH 1.2124 2.4800 0.0\n\
         H 3.3601 1.2400 0.0\nH 3.3601 -1.2400 0.0\nH 1.2124 -2.4800 0.0",
    ),
];

/// Current resident set (live footprint), from `/proc/self/statm`.
/// `VmHWM` is a high-water mark and cannot show whether a peak was transient.
fn current_rss_mb() -> f64 {
    let s = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
    s.split_whitespace()
        .nth(1)
        .and_then(|v| v.parse::<f64>().ok())
        .map(|pages| pages * 4096.0 / 1024.0 / 1024.0)
        .unwrap_or(f64::NAN)
}

/// Process-wide high-water RSS.  Because `VmHWM` is monotonic it cannot be
/// attributed to a single case, so `CASES` must stay ordered smallest -> largest
/// for the per-case numbers to be meaningful (see the note above it).
fn peak_rss_mb() -> f64 {
    // Linux: VmHWM in /proc/self/status is the peak resident set size.
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            if let Some(kb) = rest.split_whitespace().next() {
                if let Ok(v) = kb.parse::<f64>() {
                    return v / 1024.0;
                }
            }
        }
    }
    f64::NAN
}

fn build(name: &str, geom: &str) -> SCF {
    let xc = std::env::var("REST_TDDFT_GRAD_PERF_XC").unwrap_or_else(|_| "blyp".to_string());
    // `REST_TDDFT_GRAD_PERF_MAXMEM=none` omits `max_memory` entirely.  REST
    // consumes it as `max_memory - currently_used_mb`, so an explicit value and
    // "unset" are *not* the same budget; the harness must support both.
    let maxmem = std::env::var("REST_TDDFT_GRAD_PERF_MAXMEM").unwrap_or_else(|_| "none".to_string());
    let maxmem_line = if maxmem.is_empty() || maxmem == "none" {
        String::new()
    } else {
        format!("    max_memory = {}\n", maxmem)
    };
    // `tddft_mode`: "mo" (default) or "ao".  Only the TDDFT excitation-energy
    // step is affected; `TddftGradEngine` always uses the AO `FxcHessianCache`.
    let tdmode = std::env::var("REST_TDDFT_GRAD_PERF_TDMODE").unwrap_or_default();
    let tdmode_line = if tdmode.is_empty() {
        String::new()
    } else {
        format!("    tddft_mode = \"{}\"\n", tdmode)
    };
    let token = format!(
        r##"
[ctrl]
    print_level = 0
    num_threads = 1
{maxmem_line}    xc = "{xc}"
    basis_path = "def2-svp"
    auxbas_path = "def2-svp-rifit"
    eri_type = "ri-v"
    charge = 0.0
    spin = 1.0
    spin_polarization = false
    auxbasis_response = true
    initial_guess = "hcore"
    max_scf_cycle = 100
    scf_acc_rho = 1.0e-10
    scf_acc_eev = 1.0e-9
    scf_acc_etot = 1.0e-12

[tddft]
    tddft_method = "tda"
    {tdmode_line}tddft_spin = "singlet"
    nroots = 3
    davidson_tol = 1.0e-8
    davidson_max_iter = 80

[geom]
    name = "{name}"
    unit = "Angstrom"
    position = """
{geom}
    """
"##
    );
    let keys = toml::from_str::<serde_json::Value>(&token[..]).unwrap();
    let (ctrl, geom_parsed) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom_parsed, None).unwrap();
    SCF::build(mol, &None)
}

#[test]
fn tddft_tda_gradient_perf() {
    if std::env::var("REST_TDDFT_GRAD_PERF").is_err() {
        return;
    }
    println!("[rest]  case          nao nmo nocc |    scf    tddft     grad     mem(MB)");
    println!("[rest]  xc = {}", std::env::var("REST_TDDFT_GRAD_PERF_XC").unwrap_or_else(|_| "blyp".into()));
    let only = std::env::var("REST_TDDFT_GRAD_PERF_ONLY").ok();
    for (name, geom) in CASES {
        if let Some(o) = &only {
            if o != name {
                continue;
            }
        }
        let mut scf = build(name, geom);
        let nao = scf.mol.num_basis;
        let nmo = scf.eigenvalues[0].len();

        let t0 = Instant::now();
        scf_io::scf_without_build(&mut scf, &None);
        let t_scf = t0.elapsed().as_secs_f64();
        let m_scf = peak_rss_mb();
        let c_scf = current_rss_mb();

        let t0 = Instant::now();
        let out = tddft_main(&mut scf).expect("TDDFT failed");
        let t_td = t0.elapsed().as_secs_f64();
        scf.tddft_excitations = Some(out.excitations.clone());
        let m_td = peak_rss_mb();
        let c_td = current_rss_mb();

        let t0 = Instant::now();
        {
            let mut gs = RIRHFGradient::new(&scf, &None);
            gs.calc_rks();
        }
        let t_gs = t0.elapsed().as_secs_f64();
        let m_gs = peak_rss_mb();
        let c_gs = current_rss_mb();
        let tda = scf
            .mol
            .ctrl
            .tddft
            .as_ref()
            .unwrap()
            .tddft_method
            .eq_ignore_ascii_case("tda");
        let exc = scf.tddft_excitations.as_ref().unwrap();
        let x = pyrest::ri_bse::dipoles::normalize(&exc[0].1, tda);
        let y = vec![0.0; x.len()];
        let t1 = Instant::now();
        let engine = TddftGradEngine::new(&scf, 1, true, tda, x, y);
        let t_new = t1.elapsed().as_secs_f64();
        let _ = engine.response_gradient();
        let t_resp = t1.elapsed().as_secs_f64() - t_new;
        let t_grad = t0.elapsed().as_secs_f64();
        let m_end = peak_rss_mb();
        let c_end = current_rss_mb();
        println!("[rest]        gs={:.2}s new={:.2}s response={:.2}s", t_gs, t_new, t_resp);
        println!(
            "[rest-mem]  peak: scf={:.0} tddft={:.0} gs_grad={:.0} response={:.0} MB",
            m_scf, m_td, m_gs, m_end
        );
        println!(
            "[rest-live] rss:  scf={:.0} tddft={:.0} gs_grad={:.0} response={:.0} MB",
            c_scf, c_td, c_gs, c_end
        );

        let nocc = scf.occupation[0].iter().filter(|&&o| o > 0.5).count();
        println!(
            "[rest]  {:12}  {:3} {:3} {:4} | {:7.2}s {:7.2}s {:7.2}s  {:8.1}",
            name,
            nao,
            nmo,
            nocc,
            t_scf,
            t_td,
            t_grad,
            peak_rss_mb()
        );
    }
}
