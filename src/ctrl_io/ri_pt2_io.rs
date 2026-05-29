use crate::ri_pt2::PT2FPMode;
use serde::{Deserialize, Serialize};
use serde_inline_default::serde_inline_default;

#[serde_inline_default]
#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RiPt2Option {
    /// Opposite-spin factor for PT2. Default is None (xc-functional dependent).
    pub os_factor: Option<f64>,
    /// Same-spin factor for PT2. Default is None (xc-functional dependent).
    pub ss_factor: Option<f64>,
    /// MPI communication mode for PT2 calculations. Default is 0 (no MPI).
    #[serde_inline_default(0)]
    pub mpi_mode: usize,
    /// Floating-point precision mode for PT2 calculations. Default is FP32.
    #[serde_inline_default(PT2FPMode::FP32)]
    pub fp_mode: PT2FPMode,
}
