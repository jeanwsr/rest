use crate::ri_pt2::PT2FPMode;
use serde::{Deserialize, Serialize};
use serde_inline_default::serde_inline_default;

#[serde_inline_default]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RiPt2Option {
    /// Opposite-spin factor for PT2. Default is None (xc-functional dependent).
    pub os_factor: Option<f64>,
    /// Same-spin factor for PT2. Default is None (xc-functional dependent).
    pub ss_factor: Option<f64>,
    /// MPI communication mode for PT2 calculations. Default is 0 (no MPI).
    #[serde_inline_default(0)]
    pub mpi_mode: usize,
    /// Floating-point precision mode for PT2 calculations. Default is FP32.
    /// Only effective when `new_driver = true`.
    #[serde_inline_default(PT2FPMode::FP32)]
    pub fp_mode: PT2FPMode,
    /// Switch to select the PT2 implementation.
    ///
    /// - `false` (default): use the legacy driver built on `MatrixFull`/`RIFull`.
    ///   This driver supports RHF/UHF/ROHF, rayon parallelism, and MPI (mpich)
    ///   through `*_pt2_rayon_mpi`. It uses `ao2mo_rayon_v02` for AO→MO
    ///   transformation, which has a small per-thread scratch and frees
    ///   `rimatr` immediately after the transform (see `scf_io/mod.rs:3284`),
    ///   giving a much lower memory peak.
    /// - `true`: use the new driver built on rstsr `Tensor` (pair-engine based).
    ///   This driver only supports non-MPI RHF/UHF, but offers FP32 mode and
    ///   multi-threaded BLAS per pair contraction. Note that without additional
    ///   fixes (see `obtain_cderi_xvo_*` and `pure_ao2mo.rs:228`), the new
    ///   driver has a substantially larger memory footprint due to a large
    ///   `scratch_buf` and the fact that `rimatr` is never released.
    #[serde_inline_default(false)]
    pub new_driver: bool,

    /// Switch to enable streaming AO→MO + PT2 main loop on the legacy driver.
    ///
    /// When `true` (default), the PT2 main loop processes occupied orbitals in
    /// blocks of `stream_block_size`. For each (block_i, block_j) pair, only
    /// the corresponding columns of `ri3mo` are materialized via an M1-optimized
    /// ao2mo (`ao2mo_rayon_m1`); the full `[naux, nvir, nocc]` tensor is never
    /// stored. `rimatr` remains in memory throughout. The next block pair's
    /// ao2mo is pre-fetched in a scoped thread during the current PT2 contraction
    /// (Fix B pipelining), hiding most of the ao2mo wall time behind PT2 compute.
    ///
    /// Set to `false` to use the original legacy driver (`*_pt2_rayon_mpi`),
    /// which materializes the full `ri3mo` tensor via `generate_ri3mo_rayon`.
    /// The legacy driver supports MPI parallelism but requires significantly
    /// more memory.
    ///
    /// Conditions for streaming to activate (otherwise falls back to legacy):
    ///   - MPI not active (single-node only for now)
    ///   - `use_ri_symm = true` (rimatr materialized)
    ///   - DFA family is PT2 (not SBGE2/SCSRPA)
    ///   - `new_driver = false` (new_driver takes precedence if set)
    #[serde_inline_default(true)]
    pub streaming: bool,

    /// Block size for streaming PT2 (only effective when `streaming = true`).
    ///
    /// - `None` (default): auto-select based on available memory and thread
    ///   count. Typical auto values: 32 (96-core UKS), 64 (48-core RHF).
    /// - `Some(B)`: force block size to `B`. Must satisfy `B^2 >= num_threads`
    ///   to keep enough (i, j) pairs per block-pair for rayon parallelism.
    #[serde_inline_default(None)]
    pub stream_block_size: Option<usize>,
}

/// Manual `Default` implementation so that `streaming = true` by default.
/// The derived `Default` would set all bools to `false`, but we want streaming
/// to be the default algorithm. When the `[ctrl.ri_pt2]` section is entirely
/// absent from the input, `unwrap_or_default()` uses this implementation.
impl Default for RiPt2Option {
    fn default() -> Self {
        RiPt2Option {
            os_factor: None,
            ss_factor: None,
            mpi_mode: 0,
            fp_mode: PT2FPMode::FP32,
            new_driver: false,
            streaming: true,
            stream_block_size: None,
        }
    }
}
