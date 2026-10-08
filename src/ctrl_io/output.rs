//! Keywords of the `[output]` block of the control input.
//!
//! The `[output]` block configures how the requested output files (the
//! `outputs` list in the `[ctrl]` block) are generated.

use serde::{Deserialize, Serialize};

/// Options of the `[output]` block of the control input.
#[derive(Debug,Clone,Serialize, Deserialize)]
#[serde(default)]
pub struct OutputKeywords {
    /// Dump the density matrices into the fchk file: "auto" (default) dumps
    /// every density that has been calculated (`Total/Spin SCF Density`, and
    /// for double-hybrid runs additionally `Total/Spin MP2 Density`), "scf"
    /// dumps only the SCF density, "false" dumps none.
    pub fchk_dm: String,
    /// Which writer generates the Gaussian-ordered content of the fchk file
    /// (MO coefficients, and the density layout they must agree with):
    /// "default" (the MOKIT-derived librest2fch library when compiled in, the
    /// native Rust writer otherwise), "librest2fch" or "rust". The density
    /// itself is always computed by REST; the librest2fch path preserves it
    /// (its own gen_density branch is not used, since it would rebuild the
    /// density from aufbau occupations and drop the UHF spin density). Note
    /// that omitting the density (`fchk_dm = "false"`) requires a librest2fch
    /// library that honors `gen_density = 0`; very old libraries ignore it
    /// and always dump the density.
    pub fchk_writer: String,
}

impl Default for OutputKeywords {
    fn default() -> Self {
        OutputKeywords { fchk_dm: String::from("auto"), fchk_writer: String::from("default") }
    }
}

pub fn parse_output_keywords(tmp_keys: &serde_json::Value) -> OutputKeywords {
    match tmp_keys.get("output") {
        Some(tmp_output) => serde_json::from_value(tmp_output.clone()).unwrap(),
        None => OutputKeywords::default(),
    }
}
