//! Baseline storage and verification for calc_ej_ek() G-term outputs.
//!
//! Used by the `Blas` path to diff against the stored `Inline` baseline.
//! If a G-term's Blas output differs from baseline by more than `verify_tol`,
//! `verify_term` panics with diff details (position, ref value, blas value).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// One baseline = one (system, config) tuple. Stores all 10 G-term outputs
/// plus h_partial max_abs for sanity check.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EjEkBaseline {
    pub system: String,
    pub nao: usize,
    pub naux: usize,
    pub nocc: usize,
    pub natm: usize,
    /// Max allowed diff between Blas output and stored baseline.
    pub verify_tol: f64,
    /// Each entry is length 9 * natm * natm, indexed as
    /// i0 * natm * 9 + j0 * 9 + x * 3 + y.
    pub terms: HashMap<String, Vec<f64>>,
    pub h_partial_max_abs: f64,
    pub git_sha: String,
}

/// The 10 G-term keys stored in baseline.terms.
pub const BASELINE_TERM_KEYS: &[&str] = &[
    "ej_basic",
    "ej_vjd",
    "ek_vkd",
    "ej_vj1",
    "ek_vk1",
    "ek_ri1",
    "ek_ri2d",
    "ek_ri2o",
    "ej_ri1",
    "ej_ri2d",
    "ej_ri2o",
];

impl EjEkBaseline {
    /// Save baseline to a JSON file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load baseline from a JSON file.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    /// Verify that a Blas output matches the stored baseline for one term.
    /// Panics with diff details if max_abs_diff > tol.
    pub fn verify_term(&self, term_key: &str, blas_out: &[f64], tol: f64) {
        let baseline = self.terms.get(term_key).unwrap_or_else(|| {
            panic!(
                "EjEkBaseline: term '{}' not in baseline. Available: {:?}",
                term_key,
                self.terms.keys().collect::<Vec<_>>()
            )
        });
        assert_eq!(
            blas_out.len(),
            baseline.len(),
            "EjEkBaseline: term '{}' length mismatch (blas={}, baseline={})",
            term_key,
            blas_out.len(),
            baseline.len()
        );
        let mut max_diff = 0.0f64;
        let mut max_idx = 0usize;
        for (i, (a, b)) in baseline.iter().zip(blas_out.iter()).enumerate() {
            let d = (a - b).abs();
            if d > max_diff {
                max_diff = d;
                max_idx = i;
            }
        }
        if max_diff > tol {
            // Decode index: i0 * natm * 9 + j0 * 9 + x * 3 + y
            let natm = self.natm;
            let i0 = max_idx / (natm * 9);
            let rem = max_idx - i0 * natm * 9;
            let j0 = rem / 9;
            let xy = rem - j0 * 9;
            let x = xy / 3;
            let y = xy - x * 3;
            panic!(
                "EjEkBaseline verify FAILED for term '{}':\n  \
                 max_diff = {:.3e} > tol = {:.3e}\n  \
                 at flat index {} (i0={}, j0={}, x={}, y={})\n  \
                 baseline value = {:.17e}\n  \
                 blas value     = {:.17e}",
                term_key,
                max_diff,
                tol,
                max_idx,
                i0,
                j0,
                x,
                y,
                baseline[max_idx],
                blas_out[max_idx],
            );
        }
    }
}
