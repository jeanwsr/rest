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
