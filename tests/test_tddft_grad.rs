//! Validation of the TDDFT/TDA analytic gradient (`src/ri_tddft/tddft_grad.rs`).
//!
//!  * analytic vs central finite differences of the excitation energy
//!  * analytic vs PySCF `grad.tdrks` reference values (RI-level agreement)

use pyrest::ctrl_io;
use pyrest::grad::rhf::RIRHFGradient;
use pyrest::molecule_io::Molecule;
use pyrest::ri_tddft::{tddft_main, TddftGradEngine};
use pyrest::scf_io::{self, SCF};
use rest_tensors::MatrixFull;

const GEOM_H2O: &str = r#"
    O     0.00000000     0.00000000     0.11730000
    H     0.00000000     0.75720000    -0.46920000
    H     0.00000000    -0.75720000    -0.46920000
"#;

fn build(xc: &str, method: &str) -> SCF {
    let input_token = format!(
        r##"
[ctrl]
    print_level = 0
    num_threads = 1
    xc = "{xc}"
    basis_path = "def2-svp"
    auxbas_path = "def2-universal-jkfit"
    eri_type = "ri-v"
    charge = 0.0
    spin = 1.0
    spin_polarization = false
    auxbasis_response = true
    initial_guess = "hcore"
    max_scf_cycle = 100
    scf_acc_rho = 1.0e-12
    scf_acc_eev = 1.0e-10
    scf_acc_etot = 1.0e-13

[tddft]
    tddft_method = "{method}"
    tddft_spin = "singlet"
    nroots = 3
    davidson_tol = 1.0e-11
    davidson_max_iter = 80

[geom]
    name = "H2O"
    unit = "Angstrom"
    position = """
{GEOM_H2O}
    """
"##,
        xc = xc,
        method = method,
        GEOM_H2O = GEOM_H2O
    );
    let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = SCF::build(mol, &None);
    scf_io::scf_without_build(&mut scf_data, &None);
    let out = tddft_main(&mut scf_data).expect("TDDFT failed");
    assert!(!out.excitations.is_empty(), "no TDDFT states found");
    scf_data.tddft_excitations = Some(out.excitations.clone());
    scf_data
}

/// Analytic TDDFT gradient = REST ground-state RKS gradient + response.
fn analytic_gradient_parts(scf_data: &SCF, state: usize) -> (Vec<f64>, Vec<f64>) {
    let mut gs = RIRHFGradient::new(scf_data, &None);
    gs.calc_rks();
    let de_gs = gs.result.get("de").unwrap().clone();

    let tda = scf_data
        .mol
        .ctrl
        .tddft
        .as_ref()
        .unwrap()
        .tddft_method
        .eq_ignore_ascii_case("tda");
    let exc = scf_data.tddft_excitations.as_ref().expect("amplitudes");
    let raw = &exc[state - 1].1;
    let norm = pyrest::ri_bse::dipoles::normalize(raw, tda);
    let (x, y) = if tda {
        (norm, vec![0.0; raw.len()])
    } else {
        let d = raw.len() / 2;
        (norm[..d].to_vec(), norm[d..].to_vec())
    };
    let engine = TddftGradEngine::new(scf_data, state, true, tda, x, y);
    let resp = engine.response_gradient();
    (de_gs.data.clone(), resp.data.clone())
}

fn analytic_gradient(scf_data: &SCF, state: usize) -> Vec<f64> {
    let (gs, resp) = analytic_gradient_parts(scf_data, state);
    let mut de = gs;
    for (a, b) in de.iter_mut().zip(resp.iter()) {
        *a += b;
    }
    de
}

fn excitation_energy_displaced(scf: &SCF, atm: usize, comp: usize, h: f64, state: usize) -> f64 {
    let mut s = scf.clone();
    let mut v = vec![0.0f64; 3];
    v[comp] = h;
    s.mol.geom.geom_shift(atm, v);
    s.mol.ctrl.print_level = 0;
    s.mol.ctrl.initial_guess = String::from("inherit");
    scf_io::initialize_scf(&mut s, &None);
    scf_io::scf_without_build(&mut s, &None);
    // The TDDFT analytic gradient is the derivative of the *total* excited-state
    // energy `E_GS + omega` (PySCF `grad_elec` contains the ground-state terms).
    let e_gs = s.scf_energy;
    let out = tddft_main(&mut s).expect("TDDFT failed at displaced geometry");
    e_gs + out.energies[state - 1]
}

fn fd_gradient(scf: &SCF, state: usize, h: f64) -> Vec<f64> {
    let natm = scf.mol.geom.nfree;
    let mut de = vec![0.0f64; natm * 3];
    for atm in 0..natm {
        for comp in 0..3 {
            let ep = excitation_energy_displaced(scf, atm, comp, h, state);
            let em = excitation_energy_displaced(scf, atm, comp, -h, state);
            de[atm * 3 + comp] = (ep - em) / (2.0 * h);
        }
    }
    de
}

fn check_vs_fd(xc: &str, method: &str, tol: f64) {
    let scf_data = build(xc, method);
    let state = 1usize;
    let ana = analytic_gradient(&scf_data, state);
    let h = 2.0e-4;
    let fd = fd_gradient(&scf_data, state, h);
    println!("=== {xc}/{method} state {state}: analytic vs FD ===");
    let mut maxdiff = 0.0f64;
    for (i, (a, f)) in ana.iter().zip(fd.iter()).enumerate() {
        let d = (a - f).abs();
        if d > maxdiff {
            maxdiff = d;
        }
        println!("  {:2}: ana {:+.10}  fd {:+.10}  |d| {:.2e}", i, a, f, d);
    }
    assert!(
        maxdiff < tol,
        "{xc}/{method}: analytic-vs-FD max deviation {:.3e} exceeds {:.1e}",
        maxdiff,
        tol
    );
}

/// PySCF `grad.tdrks.Gradients` total gradient for H2O/def2-svp/PBE/TDA state 1
/// (from `bench/tddft_grad_pyscf.py`, exact integrals).
const PYSCF_H2O_PBE_TDA_S1: [[f64; 3]; 3] = [
    [6.09694050e-17, -9.38999571e-15, -1.21371521e-01],
    [-3.95180790e-17, -7.51727301e-02, 6.06894536e-02],
    [-6.95165000e-17, 7.51727301e-02, 6.06894536e-02],
];

#[test]
fn tddft_tda_gradient_h2o_pbe_matches_fd() {
    check_vs_fd("pbe", "tda", 1.0e-4);
}

#[test]
fn tddft_tda_gradient_h2o_b3lyp_matches_fd() {
    check_vs_fd("b3lyp", "tda", 1.0e-4);
}

#[test]
fn tddft_tda_gradient_h2o_pbe_matches_pyscf() {
    let scf_data = build("pbe", "tda");
    let (gs, resp) = analytic_gradient_parts(&scf_data, 1);
    let _ = &gs;
    let ana = analytic_gradient(&scf_data, 1);
    let mut maxdiff = 0.0f64;
    for atm in 0..3 {
        for comp in 0..3 {
            let d = (ana[atm * 3 + comp] - PYSCF_H2O_PBE_TDA_S1[atm][comp]).abs();
            println!(
                "  atom {} comp {}: REST {:+.10} (gs {:+.8} resp {:+.10})  PySCF {:+.10}  |d| {:.2e}",
                atm, comp, ana[atm * 3 + comp], gs[atm * 3 + comp], resp[atm * 3 + comp],
                PYSCF_H2O_PBE_TDA_S1[atm][comp], d
            );
            if d > maxdiff {
                maxdiff = d;
            }
        }
    }
    println!("REST vs PySCF (PBE/TDA) max |d| = {:.3e}", maxdiff);
    assert!(
        maxdiff < 1.0e-4,
        "REST-vs-PySCF max deviation {:.3e} exceeds 1e-4 (RI-level target)",
        maxdiff
    );
}

#[test]
fn debug_zero_assembly() {
    if std::env::var("REST_TDDFT_GRAD_DEBUG").is_err() {
        return;
    }
    let scf_data = build("pbe", "tda");
    let mut gs = RIRHFGradient::new(&scf_data, &None);
    gs.calc_rks();
    let de_gs = gs.result.get("de").unwrap().clone();
    let de_nuc = gs.result.get("de_nuc").unwrap().clone();
    let exc = scf_data.tddft_excitations.as_ref().unwrap();
    let x = pyrest::ri_bse::dipoles::normalize(&exc[0].1, true);
    let engine = TddftGradEngine::new(&scf_data, 1, true, true, x, vec![0.0; exc[0].1.len()]);
    let zero = engine.zero_amplitude_gradient();
    // Ground-state auxiliary-basis response magnitude (to calibrate how much of
    // the TDDFT response-vs-FD gap is the missing aux term).
    {
        let mut gs_noaux = RIRHFGradient::new(&scf_data, &None);
        gs_noaux.flags.auxbasis_response = false;
        gs_noaux.calc_rks();
        let de_noaux = gs_noaux.result.get("de").unwrap().clone();
        let mut maxd = 0.0f64;
        for i in 0..de_gs.data.len() {
            maxd = maxd.max((de_gs.data[i] - de_noaux.data[i]).abs());
        }
        println!("ground-state aux response max |d| = {:.3e}", maxd);
        for key in ["de_jaux", "de_kaux"] {
            if let Some(v) = gs.result.get(key) {
                for a in 0..v.size[1] {
                    println!("{}[{}] = [{:.10}, {:.10}, {:.10}]", key, a,
                        v[[0,a]], v[[1,a]], v[[2,a]]);
                }
            }
        }
    }
    println!("=== zero-amplitude assembly vs ground-state gradient ===");
    for i in 0..9 {
        println!(
            "  {}: zero+nuc {:+.10}  gs {:+.10}  diff {:.2e}",
            i,
            zero.data[i] + de_nuc.data[i],
            de_gs.data[i],
            (zero.data[i] + de_nuc.data[i] - de_gs.data[i]).abs()
        );
    }
}

#[test]
fn dump_reference_data() {
    if std::env::var("REST_DUMP_TDDFT_GRAD").is_err() {
        return;
    }
    let scf_data = build("pbe", "tda");
    let ana = analytic_gradient(&scf_data, 1);
    let exc = scf_data.tddft_excitations.as_ref().unwrap();
    let out = serde_json::json!({
        "h2o_pbe_tda_s1": {
            "grad": (0..3).map(|a| (0..3).map(|c| ana[a*3+c]).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "energies": exc.iter().map(|(e,_)| *e).collect::<Vec<_>>(),
        }
    });
    std::fs::write("/tmp/tddft_grad_rest.json", serde_json::to_string_pretty(&out).unwrap()).unwrap();

    // `rest_xy.json` for bench/tda_grad_pyscf_groups.py: `x` must reshape to
    // `[nocc,nvir]` with `x[i,a]`, i.e. the transpose of REST's `[nocc,nvir]`
    // column-major layout `x[i + a*nocc]`.
    let (_, _, nocc, nvir, _, _) =
        pyrest::ri_tddft::utils::tddft_occupation_parameters(&scf_data);
    let raw = &exc[0].1;
    let norm = pyrest::ri_bse::dipoles::normalize(raw, true);
    let mut x_rowmajor = vec![0.0; nocc * nvir];
    for a in 0..nvir {
        for i in 0..nocc {
            x_rowmajor[i * nvir + a] = norm[i + a * nocc];
        }
    }
    let xy = serde_json::json!({
        "x": x_rowmajor,
        "omega": exc[0].0,
        "nocc": nocc,
        "nvir": nvir,
    });
    std::fs::write("/tmp/rest_xy.json", serde_json::to_string(&xy).unwrap()).unwrap();
    println!("wrote /tmp/tddft_grad_rest.json and /tmp/rest_xy.json");
}

#[allow(dead_code)]
fn to_mat(v: &[f64], natm: usize) -> MatrixFull<f64> {
    MatrixFull::from_vec([3, natm], v.to_vec()).unwrap()
}


/// PySCF exact-integral B3LYP ground-state gradient (reference for diagnosis).
const PYSCF_H2O_B3LYP_GS: [[f64; 3]; 3] = [
    [4.17404160e-16, -1.26362385e-16, -1.27053391e-02],
    [-5.25434815e-16, -5.65288367e-03, 6.35687171e-03],
    [3.73403698e-16, 5.65288367e-03, 6.35687171e-03],
];

const PYSCF_H2O_B3LYP_TDA_S1: [[f64; 3]; 3] = [
    [5.42511885e-16, -1.15560993e-14, -1.12246796e-01],
    [-3.68528455e-17, -7.13302502e-02, 5.61272822e-02],
    [-1.75428910e-16, 7.13302502e-02, 5.61272822e-02],
];

#[test]
fn tddft_tda_gradient_h2o_b3lyp_split_vs_pyscf() {
    let scf_data = build("b3lyp", "tda");
    let (gs, resp) = analytic_gradient_parts(&scf_data, 1);
    println!("=== B3LYP ground state ===");
    let mut mgs = 0.0f64;
    for atm in 0..3 {
        for c in 0..3 {
            let d = (gs[atm * 3 + c] - PYSCF_H2O_B3LYP_GS[atm][c]).abs();
            mgs = mgs.max(d);
            println!("  gs {} {}: REST {:+.10} PySCF {:+.10} |d| {:.2e}",
                atm, c, gs[atm * 3 + c], PYSCF_H2O_B3LYP_GS[atm][c], d);
        }
    }
    println!("B3LYP gs max |d| = {:.3e}", mgs);
    println!("=== B3LYP response ===");
    let mut mr = 0.0f64;
    for atm in 0..3 {
        for c in 0..3 {
            let pr = PYSCF_H2O_B3LYP_TDA_S1[atm][c] - PYSCF_H2O_B3LYP_GS[atm][c];
            let d = (resp[atm * 3 + c] - pr).abs();
            mr = mr.max(d);
            println!("  resp {} {}: REST {:+.10} PySCF {:+.10} |d| {:.2e}",
                atm, c, resp[atm * 3 + c], pr, d);
        }
    }
    println!("B3LYP response max |d| = {:.3e}", mr);
    assert!(mgs < 1.0e-5, "B3LYP ground-state gradient off by {:.3e}", mgs);
    assert!(mr < 1.0e-4, "B3LYP response gradient off by {:.3e}", mr);
}

/// PySCF `grad.tdrks.Gradients` for H2O/def2-svp/SVWN (pure LDA)/TDA state 1.
/// This exercises the LDA branch of `xc_eval_mat`, which the GGA tests do not.
const PYSCF_H2O_SVWN_TDA_S1: [[f64; 3]; 3] = [
    [0.0, -0.0, -1.27746387e-01],
    [0.0, -7.83512473e-02, 6.38799376e-02],
    [0.0, 7.83512473e-02, 6.38799376e-02],
];

#[test]
fn tddft_tda_gradient_h2o_svwn_matches_pyscf() {
    let scf_data = build("svwn", "tda");
    let ana = analytic_gradient(&scf_data, 1);
    let mut maxdiff = 0.0f64;
    for atm in 0..3 {
        for comp in 0..3 {
            let d = (ana[atm * 3 + comp] - PYSCF_H2O_SVWN_TDA_S1[atm][comp]).abs();
            println!(
                "  svwn atom {} comp {}: REST {:+.10}  PySCF {:+.10}  |d| {:.2e}",
                atm, comp, ana[atm * 3 + comp], PYSCF_H2O_SVWN_TDA_S1[atm][comp], d
            );
            maxdiff = maxdiff.max(d);
        }
    }
    println!("REST vs PySCF (SVWN/TDA) max |d| = {:.3e}", maxdiff);
    assert!(
        maxdiff < 1.0e-4,
        "REST-vs-PySCF (SVWN/TDA) max deviation {:.3e} exceeds 1e-4",
        maxdiff
    );
}

/// The response-only assembly must reproduce the legacy
/// `assemble(x,y) - assemble(0,0)` oracle, for both a pure GGA and a hybrid
/// (which exercises the K derivative tables and the Z-vector).
///
/// The comparison tolerance is bounded by the CPHF Z-vector solve's **run-to-run
/// reproducibility**, measured here: the hybrid Krylov solve stalls ("subspace
/// exhausted") at a residual of ~1e-8, and the low-rank K matvec uses a rayon
/// `fold`/`reduce` whose summation order is not fixed, so two independent solves
/// of the same RHS differ by ~1e-8.  A pure functional (K absent from the
/// matvec, solve converges) reproduces to ~1e-14.
#[test]
fn response_only_matches_legacy_assembly() {
    for (xc, tol) in [("pbe", 1.0e-12), ("b3lyp", 1.0e-11)] {
        let scf_data = build(xc, "tda");
        let tda = true;
        let exc = scf_data.tddft_excitations.as_ref().unwrap();
        let x = pyrest::ri_bse::dipoles::normalize(&exc[0].1, tda);
        let y = vec![0.0; x.len()];
        let engine = TddftGradEngine::new(&scf_data, 1, true, tda, x, y);
        let diff = |a: &MatrixFull<f64>, b: &MatrixFull<f64>| -> f64 {
            (0..a.data.len()).fold(0.0f64, |m, k| m.max((a.data[k] - b.data[k]).abs()))
        };
        let direct = engine.response_gradient();
        let legacy = engine.response_gradient_legacy();
        // Self-reproducibility of each path (each runs its own CPHF solve).
        let direct2 = engine.response_gradient();
        let legacy2 = engine.response_gradient_legacy();
        println!(
            "[{xc}] direct-vs-legacy = {:.3e} | direct self = {:.3e} | legacy self = {:.3e}",
            diff(&direct, &legacy),
            diff(&direct, &direct2),
            diff(&legacy, &legacy2),
        );
        let mx = diff(&direct, &legacy);
        assert!(
            mx < tol,
            "{xc}: response-only vs legacy max |d| = {:.3e} > {:.1e}",
            mx,
            tol
        );
    }
}


/// Batch B (low-rank factor representation) regression: the transposed
/// densities are now `(r, l)` of the original factor pair and the virtual-space
/// block is `U+ U+^T + U- U-^T`, instead of re-factorising with `nvir` columns.
/// Both representations describe exactly the same four generating densities, so
/// the gradient must agree to roundoff.  `REST_TDDFT_GRAD_FACTOR_LEGACY=1`
/// restores the old construction.
///
/// Only meaningful for hybrids (a pure functional never builds K factors).
#[test]
fn batch_b_factor_representation_is_equivalent() {
    let scf_data = build("b3lyp", "tda");
    let exc = scf_data.tddft_excitations.as_ref().unwrap();
    let x = pyrest::ri_bse::dipoles::normalize(&exc[0].1, true);
    let y = vec![0.0; x.len()];

    let new = {
        std::env::remove_var("REST_TDDFT_GRAD_FACTOR_LEGACY");
        let engine = TddftGradEngine::new(&scf_data, 1, true, true, x.clone(), y.clone());
        engine.response_gradient()
    };
    let old = {
        std::env::set_var("REST_TDDFT_GRAD_FACTOR_LEGACY", "1");
        let engine = TddftGradEngine::new(&scf_data, 1, true, true, x.clone(), y.clone());
        let g = engine.response_gradient();
        std::env::remove_var("REST_TDDFT_GRAD_FACTOR_LEGACY");
        g
    };
    let mx = (0..new.data.len()).fold(0.0f64, |m, k| m.max((new.data[k] - old.data[k]).abs()));
    println!("[batch-B] new vs legacy factor representation max |d| = {:.3e}", mx);
    assert!(mx < 1.0e-10, "Batch B factor change altered the gradient by {:.3e}", mx);
}

/// Larger-system (naphthalene, nao=180) cross-check of the analytic gradient
/// against PySCF `grad.tdrks` at the RI level.
///
/// Opt-in (needs the reference dump first):
///   ~/pyscf_env/bin/python bench/tddft_grad_c10h8_reference.py
///   REST_TDDFT_GRAD_BIG=1 cargo test -p rest --test test_tddft_grad -- --nocapture
#[test]
fn c10h8_matches_pyscf() {
    if std::env::var("REST_TDDFT_GRAD_BIG").is_err() {
        return;
    }
    let raw = std::fs::read_to_string("/tmp/tddft_grad_c10h8_pyscf.json")
        .expect("run bench/tddft_grad_c10h8_reference.py first");
    let r: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let natm = r["natm"].as_u64().unwrap() as usize;
    let ref_grad: Vec<f64> = (0..natm)
        .flat_map(|a| (0..3).map(move |c| (a, c)))
        .map(|(a, c)| r["grad"][a][c].as_f64().unwrap())
        .collect();

    let token = r##"
[ctrl]
    print_level = 0
    num_threads = 1
    xc = "blyp"
    basis_path = "def2-svp"
    auxbas_path = "AUXBAS"
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
    tddft_spin = "singlet"
    nroots = 3
    davidson_tol = 1.0e-8
    davidson_max_iter = 80

[geom]
    name = "c10h8"
    unit = "Angstrom"
    position = """
C -1.2124  1.4000  0.0
C -2.4248  0.7000  0.0
C -2.4248 -0.7000  0.0
C -1.2124 -1.4000  0.0
C  0.0000 -0.7000  0.0
C  0.0000  0.7000  0.0
C  1.2124  1.4000  0.0
C  2.4248  0.7000  0.0
C  2.4248 -0.7000  0.0
C  1.2124 -1.4000  0.0
H -1.2124  2.4800  0.0
H -3.3601  1.2400  0.0
H -3.3601 -1.2400  0.0
H -1.2124 -2.4800  0.0
H  1.2124  2.4800  0.0
H  3.3601  1.2400  0.0
H  3.3601 -1.2400  0.0
H  1.2124 -2.4800  0.0
    """
"##;
    let aux = std::env::var("REST_TDDFT_GRAD_BIG_AUX").unwrap_or_else(|_| "def2-svp-rifit".into());
    let token = token.replace("AUXBAS", &aux);
    let keys = toml::from_str::<serde_json::Value>(&token[..]).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = SCF::build(mol, &None);
    scf_io::scf_without_build(&mut scf_data, &None);
    let out = tddft_main(&mut scf_data).expect("TDDFT failed");
    assert_eq!(out.excitations.len(), 3);
    println!("c10h8 aux = {}", aux);
    scf_data.tddft_excitations = Some(out.excitations.clone());
    let omega_rest = out.energies[0];
    let omega_ref = r["omega"].as_f64().unwrap();
    println!("c10h8 omega: REST {:.10}  PySCF {:.10}  |d| {:.2e}",
             omega_rest, omega_ref, (omega_rest - omega_ref).abs());

    let mut gs = RIRHFGradient::new(&scf_data, &None);
    gs.calc_rks();
    let de_gs = gs.result.get("de").unwrap().clone();
    let exc = scf_data.tddft_excitations.as_ref().unwrap();
    let x = pyrest::ri_bse::dipoles::normalize(&exc[0].1, true);
    let y = vec![0.0; x.len()];
    let engine = TddftGradEngine::new(&scf_data, 1, true, true, x, y);
    let resp = engine.response_gradient();
    let mut maxdiff = 0.0f64;
    let mut maxval = 0.0f64;
    for k in 0..natm * 3 {
        let ana = de_gs.data[k] + resp.data[k];
        let d = (ana - ref_grad[k]).abs();
        maxval = maxval.max(ref_grad[k].abs());
        maxdiff = maxdiff.max(d);
    }
    println!("c10h8 gradient vs PySCF: max |d| = {:.3e} (max |ref| = {:.3e})", maxdiff, maxval);
    // Tolerance is the RI-vs-exact gap, not an implementation error: REST runs
    // RI (def2-svp-rifit) while the PySCF reference uses exact 4-center
    // integrals (PySCF has no DF-TDDFT gradient -- `grad.tdrhf` raises
    // NotImplementedError).  Tightening the aux basis shrinks both numbers.
    assert!(omega_ref > 0.0);
    assert!(maxdiff < 1.0e-3, "c10h8 gradient vs PySCF max |d| = {:.3e} exceeds 1e-3", maxdiff);
}
