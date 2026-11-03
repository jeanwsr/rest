#![allow(non_snake_case)]

//! Solvent (PCM / SMD) 集成测试 —— 测试标准化设计的 T1 档。
//!
//! 双层断言策略：
//! - **PySCF 参考值**（从 `test/solvent/{grad_err,smd_benchmark}` 的 JSON 提取为常量）：
//!   验证物理正确性，松容差（按对应套件实测 RMS 设定）；
//! - **REST golden**（从 `test/solvent/grad_err/logs/*.log`、
//!   `test/solvent/smd_benchmark/logs/rest/*.log` 提取）：验证"有没有改坏"，紧容差。
//!
//! 运行：`cargo test -p rest --test test_solvent`（建议配合根 Cargo.toml 的
//! `[profile.test] opt-level = 3`）。
//! T3 重基准后若 golden 发生变化，需同步更新常量及其出处注释。

mod pcm;
mod smd;

use pyrest::ctrl_io;
use pyrest::grad::rhf::RIRHFGradient;
use pyrest::molecule_io::Molecule;
use pyrest::scf_io::{scf_without_build, SCF};
use rest_tensors::MatrixFull;

/// 跑一个带溶剂的 force 任务（SCF + 解析梯度），返回 SCF 数据与 `de_solvent` [3, natm]。
pub fn run_solvent_force(input_token: &str) -> (SCF, MatrixFull<f64>) {
    let keys = toml::from_str::<serde_json::Value>(input_token).unwrap();
    let (ctrl, geom) = ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = SCF::build(mol, &None);
    scf_without_build(&mut scf_data, &None);
    let mut grad = RIRHFGradient::new(&scf_data, &None);
    grad.calc();
    let de_solvent = grad.result.get("de_solvent").expect("de_solvent missing").clone();
    (scf_data, de_solvent)
}

/// `max |REST[xyz, a] − ref[a][xyz]|`（ref 为 [natm][3] 布局）。
pub fn max_abs_diff(de: &MatrixFull<f64>, r: &[[f64; 3]]) -> f64 {
    let natm = r.len();
    let mut mx = 0.0f64;
    for a in 0..natm {
        for xyz in 0..3 {
            mx = mx.max((de[(xyz, a)] - r[a][xyz]).abs());
        }
    }
    mx
}

/// 相对误差（分母取 max(|a|,|b|)；两者近零时返回 0）。
pub fn rel_err(a: f64, b: f64) -> f64 {
    let m = a.abs().max(b.abs());
    if m < 1e-15 { 0.0 } else { (a - b).abs() / m }
}
