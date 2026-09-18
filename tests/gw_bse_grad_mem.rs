//! Memory-scaling scan of the CD-GW and BSE analytic gradients.
//!
//! Purpose: measure how the *peak* and the *resident* memory of the two
//! gradient engines depend on the basis-set size, so the leading scaling
//! exponent can be fitted (this is the GW/BSE counterpart of
//! `tests/test_tddft_grad_perf.rs`).
//!
//! Protocol
//! --------
//! * One case per process (`REST_GBMEM_CASE=<name>`): `VmHWM` is a
//!   process-wide high-water mark, so a second case in the same process would
//!   inherit the first one's peak.
//! * A 1 ms sampler thread records the current resident set
//!   (`/proc/self/statm`); the driver marks phase boundaries.  For every phase
//!   the report gives
//!     - `peak` = max RSS sample inside the phase window (transient peaks),
//!     - `live` = RSS at the phase boundary (what stays resident),
//!     - `hwm`  = `VmHWM` at the boundary (process high-water mark so far).
//! * `[gbmem-obj]` lines give the exact byte size of the dominant data
//!   structures (computed from the public engine fields), so the RSS curve can
//!   be attributed to specific objects rather than guessed.
//!
//! Run (one case per process):
//! ```text
//! LD_LIBRARY_PATH=~/rest_workspace/lib CARGO_TARGET_DIR=target_u \
//!   REST_GBMEM_CASE=h2o_svp REST_GBMEM_METHOD=bse \
//!   cargo test --release --test gw_bse_grad_mem -- --nocapture
//! ```
//! `REST_GBMEM_METHOD` is `gw`, `bse` or `both` (default `both`);
//! `REST_GBMEM_SPLIT=1` additionally marks the *stages* inside the gradient
//! (CPHF / pullback / assembly), i.e. it replicates `analytic_gradient_*`
//! with marks between the stages (identical operations, identical peak);
//! split records are stored under `<case>_<method>split`;
//! `REST_GBMEM_OUT` overrides the JSON output path.

use pyrest::ri_bse::bse_grad::{BseGradEngine, BseSolver, ScreeningEnergy};
use pyrest::ri_gw::gw_grad::{GwCdGradEngine, GwGradConfig, GwGradEngine, QpCache};
use pyrest::scf_io::SCF;
use pyrest::solvers::davidson::DavidsonConfig;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ---------------------------------------------------------------------------
// cases
// ---------------------------------------------------------------------------

struct Case {
    name: &'static str,
    geom: &'static str,
    basis: &'static str,
    auxbasis: &'static str,
}

/// Two orthogonal scans of the basis-set axis:
/// * a fixed molecule (h2o) with four basis sets, 24 -> ~92 AO functions;
/// * a molecule series at fixed def2-svp, 24 -> 180 AO functions.
/// All molecules have only 1-D irreps (the canonical-response Roothaan
/// reconstruction of the gradient is singular for degenerate blocks).
const CASES: &[Case] = &[
    Case {
        name: "h2o_svp",
        geom: "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692",
        basis: "def2-svp",
        auxbasis: "def2-svp-rifit",
    },
    Case {
        name: "h2o_tzvp",
        geom: "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692",
        basis: "def2-tzvp",
        auxbasis: "def2-tzvp-rifit",
    },
    Case {
        name: "h2o_qzvpp",
        geom: "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692",
        basis: "def2-qzvpp",
        auxbasis: "def2-qzvpp-rifit",
    },
    Case {
        name: "h2o_atz",
        geom: "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692",
        basis: "aug-cc-pvtz",
        auxbasis: "aug-cc-pvtz-rifit",
    },
    Case {
        name: "h2o_avdz",
        geom: "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692",
        basis: "aug-cc-pvdz",
        auxbasis: "cc-pvdz-rifit",
    },
    Case {
        name: "h2co_svp",
        geom: "C 0.0 0.0 0.0; O 0.0 0.0 1.2074; H 0.0 0.9418 -0.5853; H 0.0 -0.9418 -0.5853",
        basis: "def2-svp",
        auxbasis: "def2-svp-rifit",
    },
    Case {
        name: "c2h4_svp",
        geom: "C 0.6695 0.0 0.0; C -0.6695 0.0 0.0; H 1.2301 0.9289 0.0; H 1.2301 -0.9289 0.0; H -1.2301 0.9289 0.0; H -1.2301 -0.9289 0.0",
        basis: "def2-svp",
        auxbasis: "def2-svp-rifit",
    },
    Case {
        name: "c2h4_tzvp",
        geom: "C 0.6695 0.0 0.0; C -0.6695 0.0 0.0; H 1.2301 0.9289 0.0; H 1.2301 -0.9289 0.0; H -1.2301 0.9289 0.0; H -1.2301 -0.9289 0.0",
        basis: "def2-tzvp",
        auxbasis: "def2-tzvp-rifit",
    },
    Case {
        name: "c2h4_qzvpp",
        geom: "C 0.6695 0.0 0.0; C -0.6695 0.0 0.0; H 1.2301 0.9289 0.0; H 1.2301 -0.9289 0.0; H -1.2301 0.9289 0.0; H -1.2301 -0.9289 0.0",
        basis: "def2-qzvpp",
        auxbasis: "def2-qzvpp-rifit",
    },
    Case {
        name: "c4h6_svp",
        geom: "C -1.462772 1.122980 0.000000; C -0.733500 0.000000 0.000000; C 0.733500 0.000000 0.000000; C 1.462772 -1.122980 -0.000000; H -0.969284 2.091504 0.000000; H -2.548282 1.066091 0.000000; H -1.252172 -0.955274 0.000000; H 1.252172 0.955274 -0.000000; H 0.969284 -2.091504 -0.000000; H 2.548282 -1.066091 -0.000000",
        basis: "def2-svp",
        auxbasis: "def2-svp-rifit",
    },
    Case {
        name: "c4h6_tzvp",
        geom: "C -1.462772 1.122980 0.000000; C -0.733500 0.000000 0.000000; C 0.733500 0.000000 0.000000; C 1.462772 -1.122980 -0.000000; H -0.969284 2.091504 0.000000; H -2.548282 1.066091 0.000000; H -1.252172 -0.955274 0.000000; H 1.252172 0.955274 -0.000000; H 0.969284 -2.091504 -0.000000; H 2.548282 -1.066091 -0.000000",
        basis: "def2-tzvp",
        auxbasis: "def2-tzvp-rifit",
    },
    Case {
        name: "c10h8_svp",
        geom: "C 0.000000 0.700000 0.000000; C -1.212436 1.400000 0.000000; C -2.424871 0.700000 0.000000; C -2.424871 -0.700000 0.000000; C -1.212436 -1.400000 0.000000; C -0.000000 -0.700000 0.000000; C 2.424871 0.700000 0.000000; C 1.212436 1.400000 0.000000; C 1.212436 -1.400000 0.000000; C 2.424871 -0.700000 0.000000; H -1.913008 2.230963 0.011000; H -3.472109 1.002312 0.000000; H -3.472109 -1.002312 0.000000; H -1.926008 -2.223963 0.000000; H 3.472109 1.002312 0.000000; H 1.926008 2.223963 0.000000; H 1.926008 -2.223963 0.000000; H 3.472109 -1.002312 0.000000",
        basis: "def2-svp",
        auxbasis: "def2-svp-rifit",
    },
];

// ---------------------------------------------------------------------------
// memory instrumentation
// ---------------------------------------------------------------------------

/// Current resident set size (live footprint), from `/proc/self/statm`.
fn live_rss_mb() -> f64 {
    let s = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
    s.split_whitespace()
        .nth(1)
        .and_then(|v| v.parse::<f64>().ok())
        .map(|pages| pages * 4096.0 / 1024.0 / 1024.0)
        .unwrap_or(f64::NAN)
}

/// Process-wide resident high-water mark, from `/proc/self/status`.
fn peak_rss_mb() -> f64 {
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

struct Mark {
    label: String,
    t: f64,
    live: f64,
    hwm: f64,
}

struct Phase {
    label: String,
    secs: f64,
    peak_mb: f64,
    live_mb: f64,
    hwm_mb: f64,
}

/// Background RSS sampler + phase marks.
///
/// The sampler keeps **one running maximum per phase** (bounded memory: the
/// instrument must not grow while measuring a multi-GB workload).  A raw
/// `(t, rss)` trace is only retained when explicitly requested
/// (`REST_GBMEM_TRACE=1`), and the caller downsamples it.
struct MemTrace {
    t0: Instant,
    stop: Arc<AtomicBool>,
    phase: Arc<AtomicUsize>,
    maxima: Arc<Mutex<Vec<f64>>>,
    samples: Arc<Mutex<Vec<(f64, f64)>>>,
    trace: bool,
    handle: Option<std::thread::JoinHandle<()>>,
    marks: Vec<Mark>,
}

impl MemTrace {
    fn start(trace: bool) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let phase = Arc::new(AtomicUsize::new(0));
        let maxima = Arc::new(Mutex::new(vec![0.0f64]));
        let samples = Arc::new(Mutex::new(Vec::<(f64, f64)>::new()));
        let t0 = Instant::now();
        let (s2, st2, p2, m2) =
            (samples.clone(), stop.clone(), phase.clone(), maxima.clone());
        let handle = std::thread::spawn(move || {
            while !st2.load(Ordering::Relaxed) {
                let t = t0.elapsed().as_secs_f64();
                let m = live_rss_mb();
                if trace {
                    s2.lock().unwrap().push((t, m));
                }
                let p = p2.load(Ordering::Relaxed);
                let mut mx = m2.lock().unwrap();
                if p < mx.len() && m > mx[p] {
                    mx[p] = m;
                }
                drop(mx);
                std::thread::sleep(std::time::Duration::from_micros(1000));
            }
        });
        Self {
            t0,
            stop,
            phase,
            maxima,
            samples,
            trace,
            handle: Some(handle),
            marks: Vec::new(),
        }
    }

    /// Resident size recorded at the most recent mark.
    fn last_live(&self) -> f64 {
        self.marks.last().map(|m| m.live).unwrap_or(f64::NAN)
    }

    /// Close the current phase and open the next one.
    fn mark(&mut self, label: &str) {
        let t = self.t0.elapsed().as_secs_f64();
        let live = live_rss_mb();
        let hwm = peak_rss_mb();
        println!("[gbmem-mark] {label:24} t={t:9.3}s live={live:9.1}MB hwm={hwm:9.1}MB");
        self.marks.push(Mark { label: label.to_string(), t, live, hwm });
        // the phase that just ended keeps its maximum; open the next one
        let mut mx = self.maxima.lock().unwrap();
        if let Some(last) = mx.last_mut() {
            *last = last.max(live);
        }
        mx.push(0.0);
        self.phase.store(mx.len() - 1, Ordering::Relaxed);
    }

    /// Stop sampling and reduce to per-phase peaks.
    ///
    /// A mark closes the phase that precedes it, so the maximum of window
    /// `[mark[i], mark[i+1])` carries the label of `mark[i+1]` (its end);
    /// `mark[0]` is the process baseline.
    fn finish(mut self) -> (Vec<Phase>, Vec<(f64, f64)>, f64) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        let maxima = self.maxima.lock().unwrap().clone();
        let samples = self.samples.lock().unwrap().clone();
        let baseline = self.marks.first().map(|m| m.live).unwrap_or(f64::NAN);
        let mut phases = Vec::new();
        for i in 0..self.marks.len().saturating_sub(1) {
            let (a, b) = (&self.marks[i], &self.marks[i + 1]);
            // maxima[i] belongs to the window that *starts* at mark[i]; when a
            // trace was requested, use it (same windows, higher resolution)
            let peak = if self.trace {
                samples
                    .iter()
                    .filter(|(t, _)| *t >= a.t && *t < b.t)
                    .fold(a.live, |acc, (_, v)| acc.max(*v))
            } else {
                maxima.get(i + 1).copied().unwrap_or(a.live).max(a.live)
            };
            phases.push(Phase {
                label: b.label.clone(),
                secs: b.t - a.t,
                peak_mb: peak,
                live_mb: b.live,
                hwm_mb: b.hwm,
            });
        }
        (phases, samples, baseline)
    }
}

/// Format a byte count as MB.
fn mb(bytes: usize) -> f64 {
    bytes as f64 / 1024.0 / 1024.0
}

// ---------------------------------------------------------------------------
// SCF setup (same protocol as tests/bse_grad_perf.rs)
// ---------------------------------------------------------------------------

fn build_scf(case: &Case) -> SCF {
    use pyrest::ctrl_io;
    use pyrest::molecule_io::Molecule;
    use pyrest::scf_io;

    let geom_lines = case
        .geom
        .split(';')
        .map(|s| s.trim())
        .collect::<Vec<_>>()
        .join("\n");
    let input_token = format!(
        r##"
[ctrl]
     print_level =          0
     xc =                   "hf"
     basis_path =           "{basis}"
     auxbas_path =          "{aux}"
     charge =               0.0
     spin =                 1.0
     spin_polarization =    false
     initial_guess =        "hcore"
     mixer =                "diis"
     num_threads =          1
     scf_acc_rho =          1.0e-12
     scf_acc_eev =          1.0e-10
     scf_acc_etot =         1.0e-14

[quasiparticle_methods]
     gw_scheme =            "extrapolated"
     use_low_rank_contour = true
     low_rank_tolerance =   1.0e-10
     low_rank_grid_type =   "linear"
     nomega_chi_real =      6
     cdgw_eta =             0.001
     cdgw_res_tol =         0.001
     nomega_sigma =         10
     step_sigma =           0.05
     bse_cutoff_energy =    1.0

[geom]
    unit = "Angstrom"
    position = """
{geom}
"""
"##,
        basis = case.basis,
        aux = case.auxbasis,
        geom = geom_lines
    );
    let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = SCF::build(mol, &None);
    scf_io::scf_without_build(&mut scf_data, &None);
    scf_data
}

// ---------------------------------------------------------------------------
// runners
// ---------------------------------------------------------------------------

/// Dominant geometry-fixed objects of `GwGradEngine` (bytes).
fn gw_engine_objects(e: &GwGradEngine) -> serde_json::Value {
    let naux = e.naux;
    let nao = e.nao;
    let nmo = e.nmo;
    let nquad = e.quad.len();
    let raw_i3 = e.raw.i3.len() * 8;
    let dj1 = e.raw.dj1.len() * 8;
    let j2 = e.raw.j2.len() * 8;
    let q = e.q.len() * 8;
    let q_pairmajor = e.q_pairmajor.len() * 8;
    let qt = e.qt.len() * 8;
    let qia = e.qia.len() * 8;
    let qtia = e.qtia.len() * 8;
    let metric = (e.j_mat.data.len() + e.j_chol.data.len() + e.s_metric.len()) * 8;
    // CdScreening: nquad real Cholesky factors + one complex LU factor
    let screening = nquad * naux * naux * 8 + 2 * naux * naux * 8;
    // LR screening factors (only the LR/BSE engine builds them)
    let lr_factors: usize = e
        .lr_imag
        .iter()
        .chain(e.lr_grid.iter())
        .map(|f| (f.eigvec.len() + f.eigval.len()) * 8)
        .sum();
    serde_json::json!({
        "raw_i3": mb(raw_i3),
        "raw_dj1": mb(dj1),
        "raw_j2": mb(j2),
        "q_nao2naux_like": mb(q),
        "q_pairmajor": mb(q_pairmajor),
        "qt": mb(qt),
        "qia": mb(qia),
        "qtia": mb(qtia),
        "metric_j_jchol_smetric": mb(metric),
        "screening_est": mb(screening),
        "lr_factors": mb(lr_factors),
        "total_est": mb(raw_i3 + dj1 + j2 + q + q_pairmajor + qt + qia + qtia + metric
            + screening + lr_factors),
        "dims": {"nao": nao, "nmo": nmo, "naux": naux, "nquad": nquad},
    })
}

/// Byte sizes of the BSE-side objects that stay resident.
fn bse_objects(gw: &GwGradEngine, bse: &BseGradEngine) -> serde_json::Value {
    let s = &bse.screen;
    let naux = s.naux;
    let nmo = gw.nmo;
    let mut qp = 0usize;
    for c in &bse.qp_caches {
        qp += c.w_imag.len() * c.w_imag.first().map_or(0, |v| v.len()) * 8;
        for (_u, y, y0) in &c.y_imag {
            qp += (y.len() + y0.len()) * 8;
        }
        for (_m, _s, _z, y, y0) in &c.residues {
            qp += (y.len() + y0.len()) * 8;
        }
    }
    let screen_bytes = (s.gap.len()
        + s.gap_full.len()
        + s.qov.len()
        + s.qov_full.len()
        + s.qoo.len()
        + s.qvv.len()
        + s.v_ia.len()
        + s.u_ij.len()
        + s.u_ab.len()
        + s.u_ia.len())
        * 8;
    let dim = s.nocc * s.nvir;
    // the BSE gradient also carries the LR `GwGradEngine` (raw.i3, q, qt,
    // q_pairmajor, metric, LR factors) for the whole run
    let g = gw_engine_objects(gw);
    serde_json::json!({
        "gw_engine_total_est": g["total_est"].clone(),
        "gw_raw_i3": g["raw_i3"].clone(),
        "gw_q": g["q_nao2naux_like"].clone(),
        "gw_qt": g["qt"].clone(),
        "gw_q_pairmajor": g["q_pairmajor"].clone(),
        "gw_metric": g["metric_j_jchol_smetric"].clone(),
        "gw_lr_factors": g["lr_factors"].clone(),
        "qp_caches_resident": mb(qp),
        "screen_blocks": mb(screen_bytes),
        "screen_qov_full": mb(s.qov_full.len() * 8),
        "bse_matrices_2xd2": mb(2 * dim * dim * 8),
        "bq_covector_1x": mb(naux * nmo * nmo * 8),
        "dims": {
            "nmo": nmo, "nocc": s.nocc, "nvir_act": s.nvir,
            "nvir_full": s.nvir_full, "naux": naux, "dim": dim,
        },
    })
}

fn run_gw(
    case: &Case,
    out: &mut serde_json::Map<String, serde_json::Value>,
    trace: bool,
    split: bool,
) {
    let mut tr = MemTrace::start(trace);
    tr.mark("start");

    let mut scf_data = build_scf(case);
    scf_data.mol.ctrl.print_level = 0;
    scf_data.mol.ctrl.initial_guess = String::from("hcore");
    pyrest::scf_io::initialize_scf(&mut scf_data, &None);
    pyrest::scf_io::scf_without_build(&mut scf_data, &None);
    tr.mark("scf");

    let mut cfg = GwGradConfig::from_scf(&scf_data);
    cfg.qpe_tol = 1.0e-13;
    let mut engine = GwCdGradEngine::new(&scf_data, cfg);
    tr.mark("gw_engine_new");

    let nocc = scf_data.homo[0] + 1;
    let cache = engine.build_target_cache(nocc - 1);
    tr.mark("qp_cache_homo");

    engine.release_screening();
    tr.mark("release_screening");

    // `split` replicates `analytic_gradient_with_cache` with phase marks
    // between its three stages (identical operations, so identical peak).
    let grad = if split {
        let responses = engine.base.canonical_response_batch();
        tr.mark("grad_cphf");
        let caches = [&cache];
        let (bq, bj, be, bb) = engine.cd_qp_pullback(&caches, &[vec![1.0]]);
        tr.mark("grad_pullback");
        let mut grad = vec![0.0f64; engine.base.natm * 3];
        for atm in 0..engine.base.natm {
            let blocks = engine.base.raw.d_atom_blocks(atm);
            for comp in 0..3 {
                let dj = engine.base.raw.d_j_atom(atm, comp);
                let (u, eps1, b_x) = responses[atm * 3 + comp].clone();
                let qx = engine.base.qx_from_u_blocks(&blocks, comp, &u);
                let g = engine.base.contract_perturbation(
                    &bq, &bj, &be, &bb, &qx, &dj, &eps1, &b_x,
                );
                grad[atm * 3 + comp] = g[0];
            }
        }
        tr.mark("grad_assembly");
        grad
    } else {
        let g = engine.analytic_gradient_with_cache(&cache);
        tr.mark("gw_grad");
        g
    };

    // glibc keeps freed heap pages in its arena rather than munmap-ing them,
    // which inflates the apparent resident footprint of the *next* stage; the
    // Hessian driver calls this between stages, the GW/BSE gradient path never
    // does.  Measured here so "resident at end" can be split into structural
    // data and allocator ballast.
    let live_pre_trim = tr.last_live();
    pyrest::hessian::memory_monitor::trim_to_os(0);
    tr.mark("trim");

    let (phases, samples, baseline) = tr.finish();
    let objs = gw_engine_objects(&engine.base);
    let peak = peak_rss_mb();
    let live = live_pre_trim;
    let live_post_trim = live_rss_mb();
    println!(
        "[gbmem-sum] method=gw case={} nao={} nmo={} nocc={} naux={} natm={} peak={:.1}MB live_end={:.1}MB",
        case.name,
        engine.base.nao,
        engine.base.nmo,
        engine.base.nocc,
        engine.base.naux,
        engine.base.natm,
        peak,
        live
    );
    for (k, v) in objs.as_object().unwrap() {
        if k != "dims" {
            println!("[gbmem-obj] method=gw case={} {}={:.1}MB", case.name, k, v.as_f64().unwrap());
        }
    }
    println!("[gbmem-obj] method=gw case={} dims={}", case.name, objs["dims"]);
    println!("[gbmem-val] case={} omega={:.10} z={:.8} grad0={:+.10e}", case.name, cache.omega, cache.z_factor, grad[0]);

    let mut phases_json = Vec::new();
    for p in &phases {
        println!(
            "[gbmem-phase] method=gw case={} phase={:18} secs={:8.2} peak={:9.1}MB live={:9.1}MB hwm={:9.1}MB",
            case.name, p.label, p.secs, p.peak_mb, p.live_mb, p.hwm_mb
        );
        phases_json.push(serde_json::json!({
            "label": p.label, "secs": p.secs,
            "peak_mb": p.peak_mb, "live_mb": p.live_mb, "hwm_mb": p.hwm_mb,
        }));
    }
    let mut rec = serde_json::json!({
        "case": case.name,
        "method": "gw",
        "basis": case.basis,
        "auxbasis": case.auxbasis,
        "nao": engine.base.nao,
        "nmo": engine.base.nmo,
        "nocc": engine.base.nocc,
        "naux": engine.base.naux,
        "natm": engine.base.natm,
        "nquad": engine.base.quad.len(),
        "peak_mb": peak,
        "live_end_mb": live,
        "live_post_trim_mb": live_post_trim,
        "baseline_mb": baseline,
        "phases": phases_json,
        "objects": objs,
        "omega": cache.omega,
        "z_factor": cache.z_factor,
    });
    if trace {
        let ds: Vec<(f64, f64)> = samples.iter().step_by(50).map(|&(t, m)| (t, m)).collect();
        rec["trace"] = serde_json::json!(ds);
    }
    out.insert(format!("{}_gw{}", case.name, if split { "split" } else { "" }), rec);
}

fn run_bse(
    case: &Case,
    out: &mut serde_json::Map<String, serde_json::Value>,
    trace: bool,
    split: bool,
) {
    let mut tr = MemTrace::start(trace);
    tr.mark("start");

    let mut scf_data = build_scf(case);
    scf_data.mol.ctrl.print_level = 0;
    scf_data.mol.ctrl.initial_guess = String::from("hcore");
    pyrest::scf_io::initialize_scf(&mut scf_data, &None);
    pyrest::scf_io::scf_without_build(&mut scf_data, &None);
    tr.mark("scf");

    let cfg = {
        let mut c = GwGradConfig::from_scf(&scf_data);
        c.qpe_tol = 1.0e-13;
        c
    };
    let gw = GwGradEngine::new(&scf_data, cfg);
    tr.mark("gw_engine_new");

    let vir_cutoff = scf_data
        .mol
        .ctrl
        .quasiparticle_methods
        .as_ref()
        .unwrap()
        .bse_cutoff_energy;
    let mut bse = BseGradEngine::new_with_cutoff(
        &scf_data, &gw, 's', ScreeningEnergy::Reference, vir_cutoff,
    );
    tr.mark("bse_engine_new_qp");

    let dim = bse.screen.nocc * bse.screen.nvir;
    if dim <= 128 {
        bse.solve(2, BseSolver::Dense, &DavidsonConfig::default());
    } else {
        let mut dcfg = DavidsonConfig::default();
        dcfg.max_subspace = 150;
        dcfg.max_iter = 300;
        let davidson_ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            bse.solve(2, BseSolver::Davidson, &dcfg)
        }))
        .is_ok();
        if !davidson_ok || bse.roots.len() < 2 {
            println!("[gbmem] Davidson failed or incomplete; falling back to dense");
            bse.solve(2, BseSolver::Dense, &DavidsonConfig::default());
        }
    }
    tr.mark("bse_solve");

    // `split` replicates `analytic_gradients` with phase marks between its
    // four stages (identical operations, so identical peak).
    let (grads, omegas) = if split {
        let nk = bse.roots.len();
        let naux = gw.naux;
        let nmo = gw.nmo;
        let nocc = gw.nocc;
        let nvir = bse.screen.nvir;
        let pair = |p: usize, q: usize| p + q * nmo;
        let mut weights: Vec<Vec<f64>> = Vec::with_capacity(nk);
        let mut ebar_direct: Vec<Vec<f64>> = Vec::with_capacity(nk);
        let mut bq_k: Vec<Vec<f64>> = Vec::with_capacity(nk);
        let mut bj_k: Vec<Vec<f64>> = Vec::with_capacity(nk);
        for root in &bse.roots {
            let w_act =
                pyrest::ri_bse::bse_grad::qp_gap_weights(&root.x, &root.y, nocc, nvir);
            let mut w = vec![0.0f64; nmo];
            w[..nocc + nvir].copy_from_slice(&w_act);
            let (qbar, jbar, ebar) = pyrest::ri_bse::bse_grad::kernel_vjp(
                &bse.screen, &root.x, &root.y, bse.kappa, pair, nmo,
            );
            match bse.screening_energy {
                ScreeningEnergy::Qp => {
                    for p in 0..nmo {
                        w[p] += ebar[p];
                    }
                }
                ScreeningEnergy::Reference => ebar_direct.push(ebar),
            }
            weights.push(w);
            bq_k.push(qbar);
            bj_k.push(jbar);
        }
        tr.mark("grad_kernel_vjp");

        let caches: Vec<&QpCache> = bse.qp_caches.iter().collect();
        let (bq, bj, be, bb) = gw.qp_pullback(&caches, &weights);
        tr.mark("grad_qp_pullback");

        let responses = gw.canonical_response_batch();
        tr.mark("grad_cphf");

        let zero_q = vec![0.0f64; naux * nmo * nmo];
        let zero_j = vec![0.0f64; naux * naux];
        let mut grads = vec![vec![0.0f64; gw.natm * 3]; nk];
        for atm in 0..gw.natm {
            let blocks = gw.raw.d_atom_blocks(atm);
            for comp in 0..3 {
                let dj = gw.raw.d_j_atom(atm, comp);
                let (u, eps1, b_x) = responses[atm * 3 + comp].clone();
                let qx = gw.qx_from_u_blocks(&blocks, comp, &u);
                for k in 0..nk {
                    let qp = gw.contract_perturbation(
                        &bq[k..k + 1],
                        &bj[k..k + 1],
                        &be[k..k + 1],
                        &bb[k..k + 1],
                        &qx,
                        &dj,
                        &eps1,
                        &b_x,
                    )[0];
                    let kernel = bq_k[k].iter().zip(qx.iter()).map(|(a, b)| a * b).sum::<f64>()
                        + bj_k[k].iter().zip(dj.iter()).map(|(a, b)| a * b).sum::<f64>()
                        + ebar_direct[k].iter().zip(eps1.iter()).map(|(a, b)| a * b).sum::<f64>();
                    grads[k][atm * 3 + comp] = qp + kernel;
                }
            }
        }
        // `zero_q`/`zero_j` are dead allocations in `analytic_gradients`;
        // keep them alive so the split path measures the same source-level
        // footprint (the optimizer removes them in both paths alike).
        std::hint::black_box((&zero_q, &zero_j));
        tr.mark("grad_assembly");
        let omegas = bse.roots.iter().map(|r| r.omega).collect();
        (grads, omegas)
    } else {
        let g = bse.analytic_gradients();
        tr.mark("bse_grad");
        g
    };

    // glibc keeps freed heap pages in its arena rather than munmap-ing them,
    // which inflates the apparent resident footprint of the *next* stage; the
    // Hessian driver calls this between stages, the GW/BSE gradient path never
    // does.  Measured here so "resident at end" can be split into structural
    // data and allocator ballast.
    let live_pre_trim = tr.last_live();
    pyrest::hessian::memory_monitor::trim_to_os(0);
    tr.mark("trim");

    let (phases, samples, baseline) = tr.finish();
    let objs = bse_objects(&gw, &bse);
    let peak = peak_rss_mb();
    let live = live_pre_trim;
    let live_post_trim = live_rss_mb();
    println!(
        "[gbmem-sum] method=bse case={} nao={} nmo={} nocc={} nvir={} naux={} natm={} dim={} peak={:.1}MB live_end={:.1}MB",
        case.name, gw.nao, gw.nmo, gw.nocc, bse.screen.nvir, gw.naux, gw.natm, dim, peak, live
    );
    for (k, v) in objs.as_object().unwrap() {
        if k != "dims" {
            println!("[gbmem-obj] method=bse case={} {}={:.1}MB", case.name, k, v.as_f64().unwrap());
        }
    }
    println!("[gbmem-obj] method=bse case={} dims={}", case.name, objs["dims"]);
    println!(
        "[gbmem-val] case={} omegas={:?} grad00={:+.10e}",
        case.name,
        omegas.iter().map(|w| (w * 1e6).round() / 1e6).collect::<Vec<_>>(),
        grads[0][0]
    );

    let mut phases_json = Vec::new();
    for p in &phases {
        println!(
            "[gbmem-phase] method=bse case={} phase={:18} secs={:8.2} peak={:9.1}MB live={:9.1}MB hwm={:9.1}MB",
            case.name, p.label, p.secs, p.peak_mb, p.live_mb, p.hwm_mb
        );
        phases_json.push(serde_json::json!({
            "label": p.label, "secs": p.secs,
            "peak_mb": p.peak_mb, "live_mb": p.live_mb, "hwm_mb": p.hwm_mb,
        }));
    }
    let mut rec = serde_json::json!({
        "case": case.name,
        "method": "bse",
        "basis": case.basis,
        "auxbasis": case.auxbasis,
        "nao": gw.nao,
        "nmo": gw.nmo,
        "nocc": gw.nocc,
        "nvir_act": bse.screen.nvir,
        "naux": gw.naux,
        "natm": gw.natm,
        "dim": dim,
        "nquad": gw.quad.len(),
        "peak_mb": peak,
        "live_end_mb": live,
        "live_post_trim_mb": live_post_trim,
        "baseline_mb": baseline,
        "phases": phases_json,
        "objects": objs,
        "omegas": omegas,
    });
    if trace {
        let ds: Vec<(f64, f64)> = samples.iter().step_by(50).map(|&(t, m)| (t, m)).collect();
        rec["trace"] = serde_json::json!(ds);
    }
    out.insert(format!("{}_bse{}", case.name, if split { "split" } else { "" }), rec);
}

#[test]
fn gw_bse_grad_memory_scan() {
    let case_name = match std::env::var("REST_GBMEM_CASE") {
        Ok(v) => v,
        Err(_) => {
            println!(
                "[gbmem] set REST_GBMEM_CASE=<name> to run one case; cases: {}",
                CASES.iter().map(|c| c.name).collect::<Vec<_>>().join(", ")
            );
            return;
        }
    };
    let case = CASES
        .iter()
        .find(|c| c.name == case_name)
        .unwrap_or_else(|| panic!("unknown case {case_name}"));
    let method = std::env::var("REST_GBMEM_METHOD").unwrap_or_else(|_| "both".to_string());
    let trace = std::env::var("REST_GBMEM_TRACE").is_ok();
    let split = std::env::var("REST_GBMEM_SPLIT").is_ok();
    let out_path =
        std::env::var("REST_GBMEM_OUT").unwrap_or_else(|_| "/tmp/gw_bse_grad_mem.json".to_string());

    let mut report = serde_json::Map::new();
    if method == "gw" || method == "both" {
        run_gw(case, &mut report, trace, split);
    }
    if method == "bse" || method == "both" {
        run_bse(case, &mut report, trace, split);
    }

    // merge into the existing report so one process per case can be used
    let mut merged: serde_json::Map<String, serde_json::Value> = std::fs::read_to_string(&out_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    for (k, v) in report {
        merged.insert(k, v);
    }
    std::fs::write(&out_path, serde_json::to_string_pretty(&merged).unwrap()).unwrap();
    println!("[gbmem] written {out_path}");
}
