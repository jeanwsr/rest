//! BSE analytic-gradient performance benchmark on small molecules.
//!
//! Same molecules / basis / auxbasis / numerical protocol as
//! bench/bench_bse_grad_pyscf.py, so timings and gradients are directly
//! comparable.  Single-threaded.  Writes /tmp/bse_grad_rest.json.
//!
//! Bases are taken from REST's basis-set-pool; only basis sets whose pool
//! files contain no combined (SP) shells are used, cf. the notes below.
//! All benchmark molecules have only 1-D irreps (no exactly degenerate
//! orbitals): the canonical-response Roothaan reconstruction of the
//! gradient (in REST and PySCF alike) is singular for degenerate blocks.
//!
//! Run: LD_LIBRARY_PATH=~/rest_workspace/lib cargo test --release \
//!      --test bse_grad_perf -- --nocapture

use pyrest::ri_bse::bse_grad::{BseGradEngine, BseSolver, ScreeningEnergy};
use pyrest::ri_gw::gw_grad::{GwGradConfig, GwGradEngine};
use pyrest::scf_io::SCF;
use pyrest::solvers::davidson::DavidsonConfig;
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

fn peak_rss_mb() -> f64 {
    // VmHWM = peak resident set size of this process (kB on Linux)
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
fn bench_bse_grad_small_molecules() {
    let only = std::env::var("REST_BENCH_ONLY").ok();
    let mut report = serde_json::Map::new();
    for case in CASES {
        if let Some(name) = &only {
            if case.name != name.as_str() {
                continue;
            }
        }
        let mut scf_data = build_scf(case);

        // SCF (RI-JK reference, same convention as PySCF's density_fit)
        scf_data.mol.ctrl.print_level = 0;
        scf_data.mol.ctrl.initial_guess = String::from("hcore");
        let t0 = Instant::now();
        pyrest::scf_io::initialize_scf(&mut scf_data, &None);
        pyrest::scf_io::scf_without_build(&mut scf_data, &None);
        let t_scf = t0.elapsed().as_secs_f64();

        // GW engine + QP caches for all orbitals + BSE solve.
        // The quantised nearest-grid residue protocol is REST's production
        // default and matches PySCF's LR-CD gradient module exactly; the
        // smooth exact-residue mode (exact_residue_z = true) is only for
        // finite-difference validation.
        let cfg = {
            let mut c = GwGradConfig::from_scf(&scf_data);
            c.qpe_tol = 1.0e-13;
            c
        };
        let t0 = Instant::now();
        let gw = GwGradEngine::new(&scf_data, cfg);
        let t_engine = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
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
        let nmo = gw.nmo;
        let dim = bse.screen.nocc * bse.screen.nvir;
        // same automatic dense/iterative switch as PySCF's BSEGradients;
        // the Davidson uses a wider subspace and falls back to the dense
        // reference if it fails to deliver the requested roots
        if dim <= 128 {
            bse.solve(2, BseSolver::Dense, &DavidsonConfig::default());
        } else {
            let mut cfg = DavidsonConfig::default();
            cfg.max_subspace = 150;
            cfg.max_iter = 300;
            let davidson_ok = std::panic::catch_unwind(
                std::panic::AssertUnwindSafe(|| bse.solve(2, BseSolver::Davidson, &cfg)),
            )
            .is_ok();
            if !davidson_ok || bse.roots.len() < 2 {
                println!("[rest ] Davidson failed or incomplete; falling back to dense");
                bse.solve(2, BseSolver::Dense, &DavidsonConfig::default());
            }
        }
        let t_bse = t0.elapsed().as_secs_f64();

        // analytic gradients of both roots
        let t0 = Instant::now();
        let (grads, omegas) = bse.analytic_gradients();
        let t_grad = t0.elapsed().as_secs_f64();

        let total = t_scf + t_engine + t_bse + t_grad;
        let mem_mb = peak_rss_mb();
        println!(
            "[rest ] {:14} nao={:3} nmo={:3} naux={:3} dim={:4} | scf {:7.2}s  gw_engine {:6.2}s  qp+bse {:6.2}s  grad {:6.2}s  total {:7.2}s  mem {:7.1}MB  exci={:?}",
            case.name,
            gw.nao,
            nmo,
            gw.naux,
            dim,
            t_scf,
            t_engine,
            t_bse,
            t_grad,
            total,
            mem_mb,
            omegas
                .iter()
                .map(|w| (w * 1.0e4).round() / 1.0e4)
                .collect::<Vec<_>>()
        );
        report.insert(
            case.name.to_string(),
            serde_json::json!({
                "basis": case.basis,
                "auxbasis": case.auxbasis,
                "nao": gw.nao,
                "nmo": nmo,
                "naux": gw.naux,
                "natm": gw.natm,
                "dim": dim,
                "omegas": omegas,
                "grads": grads,
                "peak_mem_mb": mem_mb,
                "secs": {
                    "scf": t_scf,
                    "gw_engine": t_engine,
                    "bse": t_bse,
                    "grad": t_grad,
                    "total": total,
                },
            }),
        );
    }
    let out = "/tmp/bse_grad_rest.json";
    // merge into an existing report so cases can be run one process each
    // (per-case peak-RSS measurement via REST_BENCH_ONLY)
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
