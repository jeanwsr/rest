use serde::{Deserialize, Serialize};
use serde_inline_default::serde_inline_default;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnalDrvTask {
    /// Analytical Hessian matrix.
    #[serde(
        alias = "hessian",
        alias = "hess",
        alias = "frequency",
        alias = "freq",
        alias = "vibration",
        alias = "vib",
        alias = "thermo"
    )]
    Hessian,
}

#[serde_inline_default]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalDrvConfig {
    #[serde_inline_default(0.0)]
    pub cphf_level_shift: f64,
    #[serde_inline_default(1e-8)]
    pub cphf_tol: f64,
    #[serde_inline_default(42)]
    pub cphf_max_cycle: usize,
    #[serde_inline_default(14)]
    pub cphf_max_space: usize,
    #[serde_inline_default(1e-14)]
    pub cphf_lindep: f64,
    #[serde_inline_default(None)]
    pub verbose: Option<usize>,
    #[serde_inline_default(None)]
    pub atm_list: Option<Vec<usize>>,
    /// Grid level for the CP-KS response calculation.
    ///
    /// By default, the CP-KS grid level is set to `grid_gen_level.max(3) - 2` (much coarser than the SCF grid).
    #[serde_inline_default(None)]
    pub grid_level_cphf: Option<usize>,
    /// Grid level for the skeleton grid used to evaluate the XC potential and kernel.
    ///
    /// By default, the skeleton grid level is set to
    /// - `grid_gen_level` for LDA/GGA functionals
    /// - `grid_gen_level + 2` for MGGA (TAU) functionals.
    #[serde_inline_default(None)]
    pub grid_level_skeleton: Option<usize>,
    /// Tolerance for point group detection in vibrational analysis. Default to 1e-5 Bohr.
    ///
    /// Note that this tolerance will be divided by sqrt(1 + natm).
    #[serde_inline_default(1.0e-5)]
    pub tol_point_group: f64,
    
    /// Option to print gaussian-like thermo analysis (c.f. Psi4 vibration code). Default to false.
    /// 
    /// The canonical way of current REST of thermo analysis, is adding `[thermo]` section in control input,
    /// which will perform shermo-like thermo analysis.
    /// This gaussian-like thermo analysis is only for comparison purpose.
    #[serde_inline_default(false)]
    pub gau_thermo: bool,
}

impl Default for AnalDrvConfig {
    fn default() -> Self {
        Self {
            cphf_level_shift: 0.0,
            cphf_tol: 1e-8,
            cphf_max_cycle: 42,
            cphf_max_space: 14,
            cphf_lindep: 1e-14,
            verbose: None,
            atm_list: None,
            grid_level_cphf: None,
            grid_level_skeleton: None,
            tol_point_group: 1.0e-5,
            gau_thermo: false,
        }
    }
}
