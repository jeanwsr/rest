use crate::ri_pt2::{PT2Engine, PT2FPMode, PT2TorchFoldMode};
use serde::{Deserialize, Serialize};
use serde_inline_default::serde_inline_default;

#[serde_inline_default]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RiPt2Option {
    /// Opposite-spin factor for PT2. Default is None (xc-functional dependent).
    pub os_factor: Option<f64>,
    /// Same-spin factor for PT2. Default is None (xc-functional dependent).
    pub ss_factor: Option<f64>,
    /// MPI communication mode for PT2 calculations. Default is 0 (no MPI).
    #[serde_inline_default(0)]
    pub mpi_mode: usize,
    /// Floating-point precision mode for PT2 calculations. Default is FP32.
    /// Effective for the PT2 energy only when `new_driver = true` (the legacy energy driver is
    /// always f64); the analdrv generalized-Fock/property path
    /// ([`rgfock_dh_interface`](crate::analdrv::response::rgfock_interface::rgfock_dh_interface))
    /// follows this keyword regardless of `new_driver`.
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

    /// Selection of the PT2 pair-energy contraction engine.
    ///
    /// - `Cpu` (default): the built-in CPU drivers (see `new_driver` / `streaming`).
    /// - `Torch`: delegate the pair-energy contraction to the PyTorch kernel
    ///   (`dfmp2_addons`, vendored at `rest/src/ri_pt2/py/`) running on a CUDA
    ///   device through an embedded CPython interpreter (pyo3).
    #[serde_inline_default(PT2Engine::Cpu)]
    pub engine: PT2Engine,

    /// CUDA device list for the torch engine, e.g. `torch_devices = [0]` (default) or
    /// `torch_devices = [0, 1]`. The list defines *work slots* and the values only pin
    /// slots to physical GPUs (occ clusters are assigned by position in the list). With
    /// a single entry, the single-device intra-pair kernel runs on that device; with
    /// two or more entries, the multi-device driver (intra + inter contraction with
    /// half-splitting assembly) distributes the occupied orbitals over the listed
    /// slots. Indices are *logical* CUDA device ids (after `CUDA_VISIBLE_DEVICES`
    /// filtering). Entries must be *pairwise distinct*; a duplicated id is rejected
    /// (to split the work over one GPU, use `torch_force_batch_inter` instead).
    /// Ignored when `engine = Cpu`.
    #[serde_inline_default(vec![0])]
    pub torch_devices: Vec<usize>,

    /// Force the batched intra+inter evaluation even when `torch_devices` lists a
    /// single device. The occupied space is then split into `nbatch` balanced
    /// clusters (see `torch_nbatch`) processed by the multi-device driver on that
    /// one GPU: each cluster only uploads its own cderi slice and pair blocks, so
    /// the peak GPU memory scales down with the cluster size. This is the remedy
    /// for GPU OOM when the full intra evaluation (one whole-`cderi` upload plus
    /// macro-batched pair GEMMs) exceeds a single device's memory. No effect when
    /// `torch_devices` already lists 2+ devices (the intra+inter driver runs
    /// anyway) or when `engine = Cpu`.
    #[serde_inline_default(false)]
    pub torch_force_batch_inter: bool,

    /// Occupied-cluster batch count for the intra+inter torch evaluation
    /// (`torch_devices` with 2+ entries, or `torch_force_batch_inter = true`).
    /// The occupied space is split into `len(torch_devices) * nbatch` balanced
    /// clusters. `None` (default) auto-detects from per-device free GPU memory
    /// (each cluster's cderi slice budgeted to 40% of it). Ignored by the
    /// single-device path and when `engine = Cpu`.
    #[serde_inline_default(None)]
    pub torch_nbatch: Option<usize>,

    /// Accumulation precision of the element-wise energy fold following the f32
    /// matmul in the torch engine (kernel env `MP2_FOLD`):
    /// - `F32Acc` (default): fold in f32 with f64 accumulation of pair energies.
    /// - `F64`: upcast the f32 matmul result to f64 and fold in f64 (slowest,
    ///   most accurate for FP32 runs).
    /// - `F32`: fold entirely in f32 (fastest, least accurate).
    /// No effect for `fp_mode = FP64` or when `engine = Cpu`.
    #[serde_inline_default(PT2TorchFoldMode::F32Acc)]
    pub torch_fold: PT2TorchFoldMode,

    /// Macro-batch size of occupied pairs in the torch engine's j<=i contraction
    /// loop (kernel env `MP2_BATCH`). Larger values trade GPU memory for fewer,
    /// larger GEMMs. Default 16; `0` means one full batch (all pairs at once).
    /// No effect when `engine = Cpu`.
    #[serde_inline_default(16)]
    pub torch_batch: usize,
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
            engine: PT2Engine::Cpu,
            torch_devices: vec![0],
            torch_force_batch_inter: false,
            torch_nbatch: None,
            torch_fold: PT2TorchFoldMode::F32Acc,
            torch_batch: 16,
        }
    }
}
