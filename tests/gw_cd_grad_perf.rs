//! Full (non-low-rank) CD-G0W0 analytic-gradient performance benchmark.
//!
//! Same molecules / basis / auxbasis / numerical protocol as
//! bench/gw_cd_grad_pyscf.py (pyscf.gw.gw_cd_grad_optimized), so timings,
//! quasiparticle energies, gradients AND peak memory are directly
//! comparable.  Single-threaded.  Writes /tmp/gw_cd_grad_rest.json.
//!
//! Run: LD_LIBRARY_PATH=~/rest_workspace/lib cargo test --release \
//!      --test gw_cd_grad_perf -- --nocapture
//! (REST_CD_BENCH_ONLY=<case> runs one case per process for per-case
//! peak-memory measurement, matching PySCF's ru_maxrss protocol.)

use pyrest::ri_gw::gw_grad::{GwCdGradEngine, GwGradConfig};
use pyrest::scf_io::SCF;
use std::time::Instant;

struct Case {
    name: &'static str,
    geom: &'static str,
    basis: &'static str,
    auxbasis: &'static str,
}

const CASES: &[Case] = &[
    Case {
        name: "h2o_defs",
        geom: "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692",
        basis: "def2-svp",
        auxbasis: "def2-svp-rifit",
    },
    Case {
        name: "h2co_defs",
        geom: "C 0.0 0.0 0.0; O 0.0 0.0 1.2074; H 0.0 0.9418 -0.5853; H 0.0 -0.9418 -0.5853",
        basis: "def2-svp",
        auxbasis: "def2-svp-rifit",
    },
    Case {
        name: "c2h4_defs",
        geom: "C 0.6695 0.0 0.0; C -0.6695 0.0 0.0; H 1.2301 0.9289 0.0; H 1.2301 -0.9289 0.0; H -1.2301 0.9289 0.0; H -1.2301 -0.9289 0.0",
        basis: "def2-svp",
        auxbasis: "def2-svp-rifit",
    },
    Case {
        name: "h2o2_defs",
        geom: "O 0.0 0.7340 0.0577; O 0.0 -0.7340 0.0577; H 0.9083 0.8657 -0.2223; H -0.9083 -0.8657 -0.2223",
        basis: "def2-svp",
        auxbasis: "def2-svp-rifit",
    },
    Case {
        name: "ch3oh_defs",
        geom: "C 0.0 0.0 0.0; O 1.36 0.0 0.0; H 1.7 0.0 0.94; H -0.3564 1.027 0.0; H -0.3564 -1.027 0.0; H -0.3564 0.0 1.027",
        basis: "def2-svp",
        auxbasis: "def2-svp-rifit",
    },
    Case {
        name: "h2o_aug",
        geom: "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692",
        basis: "aug-cc-pvdz",
        auxbasis: "cc-pvdz-rifit",
    },
    Case {
        name: "c4h6_defs",
        geom: "C -1.462772 1.122980 0.000000; C -0.733500 0.000000 0.000000; C 0.733500 0.000000 0.000000; C 1.462772 -1.122980 -0.000000; H -0.969284 2.091504 0.000000; H -2.548282 1.066091 0.000000; H -1.252172 -0.955274 0.000000; H 1.252172 0.955274 -0.000000; H 0.969284 -2.091504 -0.000000; H 2.548282 -1.066091 -0.000000",
        basis: "def2-svp",
        auxbasis: "def2-svp-rifit",
    },
    Case {
        name: "c4h6_aug",
        geom: "C -1.462772 1.122980 0.000000; C -0.733500 0.000000 0.000000; C 0.733500 0.000000 0.000000; C 1.462772 -1.122980 -0.000000; H -0.969284 2.091504 0.000000; H -2.548282 1.066091 0.000000; H -1.252172 -0.955274 0.000000; H 1.252172 0.955274 -0.000000; H 0.969284 -2.091504 -0.000000; H 2.548282 -1.066091 -0.000000",
        basis: "aug-cc-pvdz",
        auxbasis: "cc-pvdz-rifit",
    },
    Case {
        name: "c10h8_defs",
        geom: "C 0.000000 0.700000 0.000000; C -1.212436 1.400000 0.000000; C -2.424871 0.700000 0.000000; C -2.424871 -0.700000 0.000000; C -1.212436 -1.400000 0.000000; C -0.000000 -0.700000 0.000000; C 2.424871 0.700000 0.000000; C 1.212436 1.400000 0.000000; C 1.212436 -1.400000 0.000000; C 2.424871 -0.700000 0.000000; H -1.913008 2.230963 0.011000; H -3.472109 1.002312 0.000000; H -3.472109 -1.002312 0.000000; H -1.926008 -2.223963 0.000000; H 3.472109 1.002312 0.000000; H 1.926008 2.223963 0.000000; H 1.926008 -2.223963 0.000000; H 3.472109 -1.002312 0.000000",
        basis: "def2-svp",
        auxbasis: "def2-svp-rifit",
    },
];

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
     use_low_rank_contour = false
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

fn peak_rss_mb() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: f64 = rest
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .unwrap_or(0.0);
            return kb / 1024.0;
        }
    }
    0.0
}

#[test]
fn bench_cd_grad_small_molecules() {
    let only = std::env::var("REST_CD_BENCH_ONLY").ok();
    let mut report = serde_json::Map::new();
    for case in CASES {
        if let Some(name) = &only {
            if case.name != name.as_str() {
                continue;
            }
        }
        let mut scf_data = build_scf(case);
        scf_data.mol.ctrl.print_level = 0;
        scf_data.mol.ctrl.initial_guess = String::from("hcore");
        pyrest::scf_io::initialize_scf(&mut scf_data, &None);
        pyrest::scf_io::scf_without_build(&mut scf_data, &None);

        let mut cfg = GwGradConfig::from_scf(&scf_data);
        cfg.qpe_tol = 1.0e-13;
        let t0 = Instant::now();
        let mut engine = GwCdGradEngine::new(&scf_data, cfg);
        let t_engine = t0.elapsed().as_secs_f64();

        let nocc = scf_data.homo[0] + 1;
        let lumo = scf_data.lumo[0];
        // QP caches first (this is what the screening factors are needed
        // for), then release them before the gradient phase so the gradient
        // peak memory stays low
        let t0 = Instant::now();
        let cache_homo = engine.build_target_cache(nocc - 1);
        let t_cache_h = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let cache_lumo = engine.build_target_cache(lumo);
        let t_cache_l = t0.elapsed().as_secs_f64();
        engine.release_screening();

        let t0 = Instant::now();
        let grad_homo = engine.analytic_gradient_with_cache(&cache_homo);
        let t_grad_h = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let grad_lumo = engine.analytic_gradient_with_cache(&cache_lumo);
        let t_grad_l = t0.elapsed().as_secs_f64();

        let (omega_homo, z_homo) = (cache_homo.omega, cache_homo.z_factor);
        let (omega_lumo, z_lumo) = (cache_lumo.omega, cache_lumo.z_factor);

        let mem_mb = peak_rss_mb();
        println!(
            "[rest ] {:14} nao={:3} nmo={:3} naux={:3} | engine {:6.2}s  cache {:6.2}/{:6.2}s  grad {:7.2}/{:7.2}s  mem {:7.1}MB",
            case.name,
            engine.base.nao,
            engine.base.nmo,
            engine.base.naux,
            t_engine,
            t_cache_h,
            t_cache_l,
            t_grad_h,
            t_grad_l,
            mem_mb
        );
        println!(
            "        homo omega={:.10} Z={:.8} grad[2]={:+.10e}   lumo omega={:.10} Z={:.8} grad[2]={:+.10e}",
            omega_homo, z_homo, grad_homo[2], omega_lumo, z_lumo, grad_lumo[2]
        );
        report.insert(
            case.name.to_string(),
            serde_json::json!({
                "basis": case.basis,
                "auxbasis": case.auxbasis,
                "nao": engine.base.nao,
                "nmo": engine.base.nmo,
                "naux": engine.base.naux,
                "natm": engine.base.natm,
                "peak_mem_mb": mem_mb,
                "secs": {
                    "engine": t_engine,
                    "cache_homo": t_cache_h,
                    "cache_lumo": t_cache_l,
                    "grad_homo": t_grad_h,
                    "grad_lumo": t_grad_l,
                },
                "homo": {"target": nocc - 1, "omega": omega_homo, "z": z_homo, "grads": grad_homo},
                "lumo": {"target": lumo, "omega": omega_lumo, "z": z_lumo, "grads": grad_lumo},
            }),
        );
    }
    let out = "/tmp/gw_cd_grad_rest.json";
    let mut merged: serde_json::Map<String, serde_json::Value> = std::fs::read_to_string(out)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    for (k, v) in report {
        merged.insert(k, v);
    }
    std::fs::write(out, serde_json::to_string_pretty(&merged).unwrap()).unwrap();
    println!("written {}", out);
}
