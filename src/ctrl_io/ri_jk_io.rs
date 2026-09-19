use serde::{Deserialize, Serialize};
use serde_inline_default::serde_inline_default;

/// Normalize input string by removing spaces, hyphens, underscores, and converting to lowercase.
pub fn normalize_input_string(input: &str) -> String {
    // note that e-[num], E-[num] should not be normalized to e[num], E[num]
    // in these cases, we first substitute to a temporary token, then remove the unwanted
    // characters, and finally restore the original format.
    const TEMP_TOKEN: &str = "**temp*token**";
    // may use regex for finding number after hyphen
    let reg = regex::Regex::new(r"(?P<letter>[eE])-(?P<number>\d+)").unwrap();
    let temp_str = reg.replace_all(input, format!("$letter{TEMP_TOKEN}$number").as_str());
    temp_str.trim().replace([' ', '-', '_'], "").to_lowercase().replace(TEMP_TOKEN, "-")
}

/// Deserialize a serde_json::Value into a specified type T.
///
/// This will panic if deserialization fails, instead of giving a default value.
pub fn serde_from_value<T: serde::de::DeserializeOwned>(v: &serde_json::Value) -> T {
    serde_json::from_value(v.clone()).unwrap()
}

/// Flag for algorithms of Coulomb contribution (J) to Fock operator.
///
/// - `default`: use default algorithm (currently same to `ri`).
/// - `ri`: use RI algorithm with incore fitting integrals; depending to the memory available, it
///   will decide whether to use `ri-incore` or `ri-direct`.
/// - `ri-incore`: use RI algorithm with decomposed 3c-2e ERI stored in DRAM.
/// - `ri-direct`: use RI algorithm with on-the-fly computation of 3c-2e ERI, no need to store them
///   in DRAM.
/// - `ri-schwartz`: use RI algorithm with on-the-fly computation of 3c-2e ERI and per-shell-pair
///   Schwarz screening; a screened variant of `ri-direct` that skips shell-pair/aux-shell blocks
///   whose Coulomb contribution is provably negligible. Settings of this algorithm are controlled
///   by `[ctrl.ri_jk]` (see [`RiJKOption`]). This option is only available for the J part, i.e., it
///   can only be specified through `algorithm_j`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AlgorithmJ {
    #[default]
    Default,
    Ri,
    RiIncore,
    RiDirect,
    RiSchwartz,
}

/// Options of the RI-J/RI-K algorithms, read from the `[ctrl.ri_jk]` table.
///
/// The fields currently control the Schwartz screening of the `ri-schwartz` RI-J algorithm (see
/// [`AlgorithmJ::RiSchwartz`]); they have no effect on other algorithms.
#[serde_inline_default]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RiJKOption {
    /// Integral neglect threshold (Hartree) of the Schwartz screening. A (shell pair, aux shell)
    /// block enters the evaluation only when its bound Coulomb contribution reaches this
    /// threshold. Default is `1e-12`.
    #[serde_inline_default(1e-12)]
    pub schwartz_threshold: f64,
    /// Threshold of the static overlap pre-mask applied before the exact Schwarz bound sweep:
    /// shell pairs whose most-diffuse primitives overlap by less than this value are excluded
    /// from the setup. Default is `1e-24`.
    #[serde_inline_default(1e-24)]
    pub schwartz_overlap_tol2: f64,
}

impl Default for RiJKOption {
    fn default() -> Self {
        RiJKOption { schwartz_threshold: 1e-12, schwartz_overlap_tol2: 1e-24 }
    }
}

/// Flag for algorithms of Exchange contribution (K) to Fock operator.
///
/// - `default`: use default algorithm (currently same to `ri`).
/// - `ri`: use RI algorithm with incore fitting integrals; depending to the memory available, it
///   will decide whether to use `ri-incore` or `ri-direct`.
/// - `ri-incore`: use RI algorithm with decomposed 3c-2e ERI stored in DRAM.
/// - `ri-direct`: use RI algorithm with on-the-fly computation of 3c-2e ERI, no need to store them
///   in DRAM.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AlgorithmK {
    #[default]
    Default,
    Ri,
    RiIncore,
    RiDirect,
}

/// Flag for combined algorithms of Coulomb and Exchange contributions (J and K) to Fock operator.
///
/// - `default`: use default algorithm (currently same to `ri`).
/// - `ri`: use RI algorithm with incore fitting integrals; depending to the memory available, it
///   will decide whether to use `ri-incore` or `ri-direct`.
/// - `ri-incore`: use RI algorithm with decomposed 3c-2e ERI stored in DRAM.
/// - `ri-direct`: use RI algorithm with on-the-fly computation of 3c-2e ERI, no need to store them
///   in DRAM.
/// - `separated(algorithm_j, algorithm_k)`: use separate algorithms for J and K, specified by
///   `algorithm_j` (of type [`AlgorithmJ`]) and `algorithm_k` (of type [`AlgorithmK`]).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AlgorithmJK {
    #[default]
    Default,
    Ri,
    RiIncore,
    RiDirect,
    Separated(AlgorithmJ, AlgorithmK),
}
