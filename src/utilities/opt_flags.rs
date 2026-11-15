//! Runtime switches used while validating staged optimisations.
//!
//! Each switch falls back to the original code path so that the optimised and
//! the reference behaviour can be compared in one binary.

use std::sync::OnceLock;

fn flag(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

/// `REST_GTO_LEGACY=1`: use the allocating per-primitive GTO kernels.
pub fn gto_legacy() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| flag("REST_GTO_LEGACY"))
}

/// `REST_EIG_LEGACY=1`: use the dense `dspevx` overlap probe.
pub fn eig_legacy() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| flag("REST_EIG_LEGACY"))
}

/// Safety factor between the a-priori (Schwarz) pre-screen cut and the storage-level pruning cut.
///
/// The Schwarz bound bounds the *untransformed* three-centre block while the pruning acts on the
/// row maximum of the *metric-transformed* tensor, so the a-priori cut has to sit below the
/// pruning one. The historic value is 10; `REST_PRESCREEN_SAFETY` overrides it for calibration.
pub fn prescreen_safety() -> f64 {
    static FACTOR: OnceLock<f64> = OnceLock::new();
    *FACTOR.get_or_init(|| {
        std::env::var("REST_PRESCREEN_SAFETY")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| *v >= 1.0)
            .unwrap_or(10.0)
    })
}

/// Resident set size of this process in MiB-free KiB, for the compaction log.
pub fn rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1).and_then(|v| v.parse::<u64>().ok()))
        .map(|pages| pages * 4)
        .unwrap_or(0)
}

/// Keep the historical "copy into a fresh buffer" compaction (`REST_COMPACT_COPY=1`).
///
/// The default is the in-place column left shift, which keeps the peak at the steady state.
pub fn compact_copy() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| flag("REST_COMPACT_COPY"))
}
