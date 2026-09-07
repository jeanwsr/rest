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

/* #region AnalDrvGeneralCfg */

/// General settings of the analytical derivative driver.
#[serde_inline_default]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalDrvGeneralCfg {
    #[serde_inline_default(None)]
    pub verbose: Option<usize>,
}

impl Default for AnalDrvGeneralCfg {
    fn default() -> Self {
        Self { verbose: None }
    }
}

/* #endregion AnalDrvGeneralCfg */

/* #region AnalDrvCpscfCfg */

/// Settings of the CP-SCF solver.
///
/// These keywords control how the CP-SCF equations are solved, but not what is passed into
/// the solver (which is controlled by [`AnalDrvNucgradCfg`], e.g. `atm_list`).
///
/// The legacy `cphf_*` / `*_cphf` key names are still accepted as aliases.
#[serde_inline_default]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalDrvCpscfCfg {
    #[serde(rename = "cpscf_level_shift", alias = "cphf_level_shift")]
    #[serde_inline_default(0.0)]
    pub level_shift: f64,
    #[serde(rename = "cpscf_tol", alias = "cphf_tol")]
    #[serde_inline_default(1e-9)]
    pub tol: f64,
    #[serde(rename = "cpscf_max_cycle", alias = "cphf_max_cycle")]
    #[serde_inline_default(42)]
    pub max_cycle: usize,
    #[serde(rename = "cpscf_max_space", alias = "cphf_max_space")]
    #[serde_inline_default(14)]
    pub max_space: usize,
    #[serde(rename = "cpscf_lindep", alias = "cphf_lindep")]
    #[serde_inline_default(1e-15)]
    pub lindep: f64,
    #[serde(rename = "cpscf_tol_inflation", alias = "cphf_tol_inflation")]
    #[serde_inline_default(1000.0)]
    pub tol_inflation: f64,
    /// Grid level for the CP-SCF response calculation.
    ///
    /// By default, the CP-SCF grid level is set to `grid_gen_level.max(3) - 2` (much coarser than the SCF grid).
    #[serde(rename = "grid_level_cpscf", alias = "grid_level_cphf")]
    #[serde_inline_default(None)]
    pub grid_level: Option<usize>,
}

impl Default for AnalDrvCpscfCfg {
    fn default() -> Self {
        Self {
            level_shift: 0.0,
            tol: 1e-9,
            max_cycle: 42,
            max_space: 14,
            lindep: 1e-15,
            tol_inflation: 1000.0,
            grid_level: None,
        }
    }
}

/* #endregion AnalDrvCpscfCfg */

/* #region AnalDrvNucgradCfg */

/// Settings for the nuclear-coordinate derivative properties (gradient, Hessian, dipole/polar
/// derivatives, and the derived vibrational/thermochemical analysis).
///
/// These keywords control what is differentiated (`atm_list`) and how the derivative property is
/// evaluated, but not how the CP-SCF equations are solved (see [`AnalDrvCpscfCfg`]).
#[serde_inline_default]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalDrvNucgradCfg {
    #[serde_inline_default(None)]
    pub atm_list: Option<Vec<usize>>,
    /// Grid level for the skeleton grid used to evaluate the XC potential and kernel.
    ///
    /// By default, the skeleton grid level is set to `grid_gen_level` (the SCF grid), except for
    /// MGGA (TAU) functionals with `grid_shift_deriv = false`, which add 2 levels.
    #[serde_inline_default(None)]
    pub grid_level_skeleton: Option<usize>,
    /// Include the Becke grid-shift derivatives (the nuclear-coordinate derivatives of the
    /// grid weights) in the DFT skeleton Hessian and the f1ao skeleton Fock derivatives.
    ///
    /// With it on (default), `de_xc_skeleton` and `vmat_deriv1_grid` are translationally
    /// invariant, and the skeleton grid defaults to the SCF grid for every functional family
    /// (including MGGA); with it off, the results equal the grid-fixed formulation, for which
    /// MGGA defaults the skeleton grid to `grid_gen_level + 2`.  Requires the standard
    /// atom-generated grids (external grids carry no atom attribution; disable it there).
    #[serde_inline_default(true)]
    pub grid_shift_deriv: bool,
    /// Tolerance for point group detection in vibrational analysis. Default to 1e-5 Bohr.
    ///
    /// Note that this tolerance will be divided by sqrt(1 + natm).
    #[serde_inline_default(1.0e-5)]
    pub tol_point_group: f64,
    /// Step size (Bohr) for the finite-difference Hessian of the empirical dispersion
    /// (DFTD3/DFTD4) contribution, for which no analytical Hessian is available. The
    /// dispersion Hessian is obtained by central differences of the analytic dispersion
    /// gradient. Default to 3e-4 Bohr.
    #[serde_inline_default(3.0e-4)]
    pub dftd_hess_step: f64,

    /// Option to print gaussian-like thermo analysis (c.f. Psi4 vibration code). Default to false.
    ///
    /// The canonical way of current REST of thermo analysis, is adding `[thermo]` section in control input,
    /// which will perform shermo-like thermo analysis.
    /// This gaussian-like thermo analysis is only for comparison purpose.
    #[serde_inline_default(false)]
    pub gau_thermo: bool,
}

impl Default for AnalDrvNucgradCfg {
    fn default() -> Self {
        Self {
            atm_list: None,
            grid_level_skeleton: None,
            grid_shift_deriv: true,
            tol_point_group: 1.0e-5,
            dftd_hess_step: 3.0e-4,
            gau_thermo: false,
        }
    }
}

/* #endregion AnalDrvNucgradCfg */

/* #region AnalDrvConfig */

/// Configuration of the analytical derivative driver.
///
/// This is a summary struct of the three sub-category configurations, which are flattened into the
/// same `[analdrv]` section of the control input:
/// - [`AnalDrvGeneralCfg`]: general settings;
/// - [`AnalDrvCpscfCfg`]: CP-SCF solver settings;
/// - [`AnalDrvNucgradCfg`]: nuclear-coordinate derivative property settings.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct AnalDrvConfig {
    #[serde(flatten)]
    pub general: AnalDrvGeneralCfg,
    #[serde(flatten)]
    pub cpscf: AnalDrvCpscfCfg,
    #[serde(flatten)]
    pub nucgrad: AnalDrvNucgradCfg,
}

/* #endregion AnalDrvConfig */

#[cfg(test)]
mod tests {
    use super::*;

    /// The three sub-configs flatten into the single flat `[analdrv]` key space of the control
    /// input; the current `cpscf_*` key names must work, and the legacy `cphf_*` / `*_cphf` key
    /// names must keep working as aliases.
    #[test]
    fn test_flatten_deserialize_and_serialize() {
        let v = serde_json::json!({
            "cpscf_tol": 1.0e-8,
            "cpscf_max_cycle": 99,
            "grid_level_cpscf": 2,
            "grid_level_skeleton": 4,
            "grid_shift_deriv": false,
            "gau_thermo": true,
            "atm_list": [0, 1],
            "verbose": 3,
        });
        let config: AnalDrvConfig = serde_json::from_value(v).unwrap();
        assert_eq!(config.cpscf.tol, 1.0e-8);
        assert_eq!(config.cpscf.max_cycle, 99);
        assert_eq!(config.cpscf.level_shift, 0.0);
        assert_eq!(config.cpscf.grid_level, Some(2));
        assert_eq!(config.nucgrad.grid_level_skeleton, Some(4));
        assert!(!config.nucgrad.grid_shift_deriv);
        assert!(config.nucgrad.gau_thermo);
        assert_eq!(config.nucgrad.atm_list, Some(vec![0, 1]));
        assert_eq!(config.general.verbose, Some(3));

        // the legacy key names are accepted as aliases
        let v = serde_json::json!({
            "cphf_tol": 1.0e-7,
            "cphf_max_cycle": 98,
            "grid_level_cphf": 1,
        });
        let config: AnalDrvConfig = serde_json::from_value(v).unwrap();
        assert_eq!(config.cpscf.tol, 1.0e-7);
        assert_eq!(config.cpscf.max_cycle, 98);
        assert_eq!(config.cpscf.grid_level, Some(1));

        // an empty section must give exactly the default configuration
        let config: AnalDrvConfig = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(config, AnalDrvConfig::default());

        // serialization stays flat, under the new key names
        let mut config = AnalDrvConfig::default();
        config.cpscf.tol = 1.0e-8;
        let v = serde_json::to_value(&config).unwrap();
        assert_eq!(v["cpscf_tol"], serde_json::json!(1.0e-8));
    }
}
