//! PCM 四种模型（cpcm/cosmo/iefpcm/ssvpe）的 H₂O 力测试。
//!
//! 几何与设置同 `test/solvent/grad_err`（B3LYP/def2-TZVP，water，bondi 腔体半径）。

use super::*;

static INPUT_TEMPLATE: &str = r#"
[ctrl]
    job_type = "force"
    print_level = 1
    xc = "b3lyp"
    frozen_core_postscf = 0
    basis_path = "def2-TZVP"
    auxbas_path = "def2-SV(P)-JKFIT"
    initial_guess = "sad"
    fchk = false
    pruning = "nwchem"
    radial_grid_method = "treutler"
    scf_acc_rho = 1.0e-8
    scf_acc_eev = 1.0e-8
    max_scf_cycle = 250
    solvent_model = "MODEL"
    solvent = "water"
    solv_chunk = 64
    solvent_ri = false
    pcm_cavity_radii = "bondi"
    num_threads = 1
    charge = 0
    spin = 1
    spin_polarization = false

[geom]
    name = "NAME"
    unit = "angstrom"
    position = '''
        O       -0.00000000     -0.00000001      0.11932257
        H        0.00000000      0.74368977     -0.49691077
        H        0.00000000     -0.74368976     -0.49691076
    '''
"#;

fn input_for(model: &str) -> String {
    INPUT_TEMPLATE.replace("MODEL", model).replace("NAME", &format!("{model}_h2o"))
}

// ── PySCF 参考 de_pcm（Hartree/Bohr），来自 test/solvent/grad_err/ref/<model>_h2o.json ──
//    REST 的 de_solvent = nuc+qv+solver 对应 PySCF 的 de_pcm；
//    注意 ref 里的 de_total 还含 solute 电子梯度，不能直接比。

/// PySCF `de_pcm`，cpcm_h2o.json
static CPCM_REF: [[f64; 3]; 3] = [
    [-1.0413867311502278e-17, 3.312352115477287e-10, -0.017845915209527323],
    [3.0764501342072205e-19, -0.005331694612799411, 0.008922957668862667],
    [2.5965143179757885e-17, 0.005331694281564195, 0.008922957540664602],
];

/// PySCF `de_pcm`，cosmo_h2o.json
static COSMO_REF: [[f64; 3]; 3] = [
    [-2.0681432423947608e-17, 3.282642953348349e-10, -0.017708719204199072],
    [5.628705988072237e-18, -0.005294208031479967, 0.00885435966551891],
    [2.465535231061692e-17, 0.00529420770321571, 0.008854359538680103],
];

/// PySCF `de_pcm`，iefpcm_h2o.json
static IEFPCM_REF: [[f64; 3]; 3] = [
    [3.4020520906309754e-17, 3.297823469789249e-10, -0.017740770947880673],
    [-7.821214078339716e-17, -0.0053055435584388995, 0.00887038553873115],
    [4.316311404561613e-17, 0.005305543228656598, 0.008870385409149585],
];

/// PySCF `de_pcm`，ssvpe_h2o.json
static SSVPE_REF: [[f64; 3]; 3] = [
    [9.579255743717328e-18, 3.1147515490755886e-10, -0.017596690396909333],
    [-2.7109908732159987e-17, -0.0051164153898972725, 0.008798345174518137],
    [1.7415562199931143e-17, 0.005116415078422135, 0.008798345222391146],
];

// ── REST golden（Hartree）：test/solvent/grad_err/logs/<model>_h2o.log（2026-09-10 重基准，release） ──
//    (SCF 总能量, 溶剂能量)。能量容差 1e-7：SCF 收敛判据本身是 1e-8，1e-8 容差过紧。

static CPCM_GOLD: (f64, f64) = (-76.4734158006, -0.0123020152);
static COSMO_GOLD: (f64, f64) = (-76.4733378610, -0.0122038167);
static IEFPCM_GOLD: (f64, f64) = (-76.4733456614, -0.0122137671);
static SSVPE_GOLD: (f64, f64) = (-76.4733010986, -0.0121543519);

fn check_pcm(model: &str, ref_de: &[[f64; 3]; 3], gold: (f64, f64), grad_tol: f64) {
    let (scf, de_solvent) = run_solvent_force(&input_for(model));

    // 紧容差：REST golden（能量 1e-7，见上方说明）
    assert!(
        (scf.scf_energy - gold.0).abs() < 1e-7,
        "{model}: scf_energy {:.10} != golden {:.10}",
        scf.scf_energy,
        gold.0
    );
    let e_solv = scf.energies.get("solvent_energy").expect("solvent_energy missing")[0];
    assert!(
        (e_solv - gold.1).abs() < 1e-7,
        "{model}: solvent_energy {:.10} != golden {:.10}",
        e_solv,
        gold.1
    );

    // 松容差：PySCF de_pcm（2026-09-10 实测 max|diff|：cpcm/cosmo ~6.9e-6，iefpcm 2.7e-6，ssvpe 1.25e-4）
    let diff = max_abs_diff(&de_solvent, ref_de);
    assert!(
        diff < grad_tol,
        "{model}: de_solvent vs PySCF de_pcm max|diff| = {diff:.3e} >= {grad_tol:.1e}"
    );
}

#[test]
fn test_cpcm_h2o_force() {
    check_pcm("cpcm", &CPCM_REF, CPCM_GOLD, 1e-4);
}

#[test]
fn test_cosmo_h2o_force() {
    check_pcm("cosmo", &COSMO_REF, COSMO_GOLD, 1e-4);
}

#[test]
fn test_iefpcm_h2o_force() {
    check_pcm("iefpcm", &IEFPCM_REF, IEFPCM_GOLD, 1e-4);
}

#[test]
fn test_ssvpe_h2o_force() {
    check_pcm("ssvpe", &SSVPE_REF, SSVPE_GOLD, 5e-4);
}
