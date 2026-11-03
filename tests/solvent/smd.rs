//! SMD CDS 测试：能量（gcds）、SASA（areacds）、CDS 梯度（dcds）。
//!
//! 几何与设置同 `test/solvent/smd_benchmark`（B3LYP/def2-TZVP，water，bondi 半径）。
//! PySCF 参考来自 `smd_benchmark/ref/*.json`（能量/面积）与 `smd_benchmark/grad_ref/*.json`
//! （`cds_only.dcds`）。

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
    solvent_model = "SMD"
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
COORDS
    '''
"#;

fn input_for(name: &str, coords: &str) -> String {
    INPUT_TEMPLATE.replace("NAME", name).replace("COORDS", coords)
}

// ── PySCF 参考：gcds (Hartree)、SASA (Å²)、dcds (Hartree/Bohr, [natm][3]) ──

/// 01_h2o_water
static H2O_GCDS: f64 = 0.0022985796052604466;
static H2O_AREA: f64 = 55.66506679528824;
static H2O_DCDS: [[f64; 3]; 3] = [
    [0.0, 0.0, 0.0002929251525097983],
    [-2.4410463205460343e-21, 0.00045009708710884265, -0.00014646257625489922],
    [2.4410463205460343e-21, -0.00045009708710884265, -0.00014646257625489922],
];

/// 08_hcooh_water
static HCOOH_GCDS: f64 = 0.007208719659629998;
static HCOOH_AREA: f64 = 89.91756821381607;
static HCOOH_DCDS: [[f64; 3]; 5] = [
    [7.494615289128537e-20, 0.0005390877892567961, -0.0033063842056490294],
    [-8.585832827577183e-21, 0.0006493768448232207, 0.00030321578274631326],
    [-6.285369850989688e-20, -0.003349033369448763, 0.0020448131109197317],
    [6.79556548913856e-20, 0.0023165050071646907, 0.0013533052136476124],
    [-1.2972525732356916e-20, -0.00015593627179594433, -0.0003949499016646276],
];

/// 09_ch3nh2_water
static CH3NH2_GCDS: f64 = 0.001907780429566257;
static CH3NH2_AREA: f64 = 88.9298471766125;
static CH3NH2_DCDS: [[f64; 3]; 7] = [
    [-4.6812423723483737e-20, 1.402818994969764e-07, -0.0007615726007927487],
    [1.9215686176641048e-19, -7.02186355852256e-19, -0.0009330392609494452],
    [0.0, 0.0004681921627781685, -0.0002447933518457587],
    [0.0004056137220279411, -0.00023416622233883238, -0.000244866697961117],
    [-0.0004056137220279411, -0.00023416622233883238, -0.000244866697961117],
    [-5.2192538871050124e-20, 0.0001338131627480014, 0.0012145693047550938],
    [-4.388589201215511e-20, -0.00013381316274800132, 0.0012145693047550942],
];

// ── REST golden（Hartree）：test/solvent/smd_benchmark/logs/rest/*.log（2026-09-10 重基准，release） ──
//    (SCF 总能量, 溶剂能量=静电+CDS)。能量容差 1e-7：SCF 收敛判据本身是 1e-8，1e-8 容差过紧。

static H2O_GOLD: (f64, f64) = (-76.4758570755, -0.0154537819);
static HCOOH_GOLD: (f64, f64) = (-189.7393301666, -0.0192680094);
static CH3NH2_GOLD: (f64, f64) = (-95.8821885838, -0.0037635188);

fn check_smd(
    input: &str,
    label: &str,
    ref_gcds: f64,
    ref_area: f64,
    ref_dcds: &[[f64; 3]],
    gold: (f64, f64),
) {
    let (scf, de_solvent) = run_solvent_force(input);

    // 紧容差：REST golden（能量 1e-7，见上方说明）
    assert!(
        (scf.scf_energy - gold.0).abs() < 1e-7,
        "{label}: scf_energy {:.10} != golden {:.10}",
        scf.scf_energy,
        gold.0
    );
    let e_solv = scf.energies.get("solvent_energy").expect("solvent_energy missing")[0];
    assert!(
        (e_solv - gold.1).abs() < 1e-7,
        "{label}: solvent_energy {:.10} != golden {:.10}",
        e_solv,
        gold.1
    );

    // 松容差：PySCF CDS 能量/面积（benchmark 判据 rel 1e-5）
    let pstatic = &scf.solvent_static_obj.as_ref().expect("solvent_static_obj missing").pstatic;
    let e_cds = pstatic.e_cds.expect("e_cds missing");
    let tarea = pstatic.tarea.expect("tarea missing");
    let rel_e = rel_err(e_cds, ref_gcds);
    let rel_a = rel_err(tarea, ref_area);
    assert!(rel_e <= 1e-5, "{label}: gcds {e_cds:.10} vs PySCF {ref_gcds:.10} (rel {rel_e:.2e})");
    assert!(rel_a <= 1e-5, "{label}: areacds {tarea:.6} vs PySCF {ref_area:.6} (rel {rel_a:.2e})");

    // CDS 梯度：判据同 run_grad_test.py（abs 1e-8 或 rel 1e-4，近零放宽）
    let d_cds = pstatic.d_cds.as_ref().expect("d_cds missing");
    assert_eq!(d_cds.len(), 3);
    for a in 0..ref_dcds.len() {
        for xyz in 0..3 {
            let got = d_cds[xyz][(a, 0)];
            let want = ref_dcds[a][xyz];
            let diff = (got - want).abs();
            let m = got.abs().max(want.abs());
            let rel = if m > 1e-15 { diff / m } else { 0.0 };
            assert!(
                diff < 1e-8 || rel <= 1e-4 || m < 1e-8,
                "{label}: dcds atom {a} xyz {xyz}: got {got:.6e}, want {want:.6e}, \
                 diff {diff:.2e}, rel {rel:.2e}"
            );
        }
    }

    // 全链 sanity：force 任务的 de_solvent 必须有限
    assert!(
        de_solvent.data.iter().all(|v| v.is_finite()),
        "{label}: de_solvent 含非有限值"
    );
}

#[test]
fn test_smd_h2o_water() {
    let coords = "\
     O        0.000000       0.000000       0.117349
     H        0.000000       0.757160      -0.469396
     H        0.000000      -0.757160      -0.469396";
    check_smd(&input_for("h2o_water", coords), "01_h2o_water", H2O_GCDS, H2O_AREA, &H2O_DCDS, H2O_GOLD);
}

#[test]
fn test_smd_hcooh_water() {
    let coords = "\
     C        0.000000       0.000000       0.000000
     H        0.000000       0.812000       1.102000
     O        0.000000       1.142000      -0.637000
     O        0.000000      -1.142000      -0.637000
     H        0.000000      -1.460000      -1.534000";
    check_smd(
        &input_for("hcooh_water", coords),
        "08_hcooh_water",
        HCOOH_GCDS,
        HCOOH_AREA,
        &HCOOH_DCDS,
        HCOOH_GOLD,
    );
}

#[test]
fn test_smd_ch3nh2_water() {
    let coords = "\
     C        0.000000       0.000000      -0.721000
     N        0.000000       0.000000       0.749000
     H        0.000000       0.941000      -1.213000
     H        0.814982      -0.470500      -1.213000
     H       -0.814982      -0.470500      -1.213000
     H        0.000000       0.832000       1.196000
     H        0.000000      -0.832000       1.196000";
    check_smd(
        &input_for("ch3nh2_water", coords),
        "09_ch3nh2_water",
        CH3NH2_GCDS,
        CH3NH2_AREA,
        &CH3NH2_DCDS,
        CH3NH2_GOLD,
    );
}
