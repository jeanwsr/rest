use serde::{Deserialize, Serialize};
use serde_inline_default::serde_inline_default;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnalDrvTask {
    /// Analytical Hessian matrix (and the derived vibrational/thermochemical analysis).
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
    /// Electric multipole moments (dipole to hexadecapole; see [`AnalDrvMultipoleCfg`]).
    #[serde(alias = "multipole", alias = "pole", alias = "dipole")]
    Multipole,
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

/* #region AnalDrvRespCfg */

/// Settings of the response (CP-SCF) solver.
///
/// These keywords control how the response (CP-SCF-type) equations are solved, but not what is
/// passed into the solver (which is controlled by [`AnalDrvNucgradCfg`], e.g. `atm_list`).
///
/// All these keywords are specific to the CP-SCF-type solve, hence the `cpscf_*` key names.
/// `grid_level` is not a solver setting but the DFT grid of the CP-SCF response evaluation (the
/// response path; the fock path and the DH generalized Fock stay on the SCF grid), so it is named
/// in the `grid_level_*` family of [`AnalDrvNucgradCfg::grid_level_skeleton`] with `cpscf` as the
/// scope.
///
/// The legacy `cphf_*` prefixed key names are still accepted as aliases.
#[serde_inline_default]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalDrvRespCfg {
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
    /// Grid level for the CP-SCF response path (the DFT evaluation of the response/A-tensor
    /// contractions; the fock path stays on the SCF grid).
    ///
    /// By default, the response grid level is set to `grid_gen_level.max(3) - 2` (much coarser than the SCF grid).
    #[serde(rename = "grid_level_cpscf", alias = "grid_level_cphf")]
    #[serde_inline_default(None)]
    pub grid_level: Option<usize>,
}

impl Default for AnalDrvRespCfg {
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

/* #endregion AnalDrvRespCfg */

/* #region AnalDrvNucgradCfg */

/// Settings for the nuclear-coordinate derivative properties (gradient, Hessian, dipole/polar
/// derivatives, and the derived vibrational/thermochemical analysis).
///
/// These keywords control what is differentiated (`atm_list`) and how the derivative property is
/// evaluated, but not how the response (CP-SCF-type) equations are solved (see
/// [`AnalDrvRespCfg`]).
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

/* #region AnalDrvMultipoleCfg */

/// Relaxation treatment of the double-hybrid (DH) density increments for the multipole moments.
///
/// Only meaningful for PT2-family post-SCF (fifth-DFA) methods; silently ignored otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MultipoleRdm1Relax {
    /// Relaxed density: solve the Z-vector (CP-SCF) and include the response increment.
    #[serde(rename = "relaxed")]
    Relaxed,
    /// Unrelaxed density: the correlation rdm1 increment only, no CP-SCF solve.
    #[serde(rename = "unrelaxed")]
    Unrelaxed,
}

impl Default for MultipoleRdm1Relax {
    fn default() -> Self {
        Self::Relaxed
    }
}

/// Settings of the electric multipole moment evaluation.
///
/// These keywords control what is evaluated in the [`Multipole`](AnalDrvTask::Multipole) task.
/// All moments are evaluated in atomic units; the default origin is the coordinate origin
/// `[0, 0, 0]` (Bohr), the same print convention as Gaussian and pyscf.
#[serde_inline_default]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalDrvMultipoleCfg {
    /// Orders of the multipole moments to evaluate: 1 = dipole, 2 = quadrupole, 3 = octupole,
    /// 4 = hexadecapole. Default to all `[1, 2, 3, 4]`.
    #[serde(rename = "multipole_orders")]
    #[serde_inline_default(vec![1, 2, 3, 4])]
    pub orders: Vec<usize>,
    /// Explicit origin (Bohr) of the multipole evaluation. By default `None`, meaning the
    /// coordinate origin `[0, 0, 0]` — the same print convention as Gaussian and pyscf. Note
    /// that raw moments of order >= 2 depend on where the molecule sits in the input
    /// coordinate frame; set this keyword explicitly (e.g. to the center of nuclear mass) for
    /// origin-independent reporting or literature comparison.
    #[serde(rename = "multipole_origin")]
    #[serde_inline_default(None)]
    pub origin: Option<[f64; 3]>,
    /// Relaxation of the DH density increments, see [`MultipoleRdm1Relax`]. Default to relaxed.
    #[serde(rename = "multipole_rdm1_relax")]
    #[serde_inline_default(MultipoleRdm1Relax::Relaxed)]
    pub rdm1_relax: MultipoleRdm1Relax,
    /// Dump the total density matrix (SCF density plus the correlation increments, following
    /// `multipole_rdm1_relax`) into the Gaussian formatted-checkpoint file `{molecule}.fchk`,
    /// appended as the `Total MP2 Density` section (lower-triangular, Gaussian AO order — the
    /// same convention as the MO coefficients of the fchk output; the name follows Gaussian's
    /// post-SCF density convention, and for double hybrids it is simply the storage name of
    /// the DH total density, for which Gaussian has no analog). The dumped density is the one
    /// contracted for the multipole moments, i.e. its contraction with any property integral
    /// reproduces the electronic moment (`dip_tot = dip_nuc - Tr(D int1e_r)`).
    /// Post-SCF (PT2-family) methods only; silently ignored for SCF-level methods.
    /// Default to false.
    #[serde(rename = "multipole_rdm1_dump")]
    #[serde_inline_default(false)]
    pub rdm1_dump: bool,
}

impl Default for AnalDrvMultipoleCfg {
    fn default() -> Self {
        Self {
            orders: vec![1, 2, 3, 4],
            origin: None,
            rdm1_relax: MultipoleRdm1Relax::Relaxed,
            rdm1_dump: false,
        }
    }
}

/* #endregion AnalDrvMultipoleCfg */

/* #region AnalDrvConfig */

/// Configuration of the analytical derivative driver.
///
/// This is a summary struct of the four sub-category configurations, which are flattened into the
/// same `[analdrv]` section of the control input:
/// - [`AnalDrvGeneralCfg`]: general settings;
/// - [`AnalDrvRespCfg`]: response (CP-SCF) solver settings;
/// - [`AnalDrvNucgradCfg`]: nuclear-coordinate derivative property settings;
/// - [`AnalDrvMultipoleCfg`]: electric multipole moment settings.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct AnalDrvConfig {
    #[serde(flatten)]
    pub general: AnalDrvGeneralCfg,
    #[serde(flatten)]
    pub resp: AnalDrvRespCfg,
    #[serde(flatten)]
    pub nucgrad: AnalDrvNucgradCfg,
    #[serde(flatten)]
    pub multipole: AnalDrvMultipoleCfg,
}

/* #endregion AnalDrvConfig */

#[cfg(test)]
mod tests {
    use super::*;

    /// The four sub-configs flatten into the single flat `[analdrv]` key space of the control
    /// input; the current `cpscf_*` key names must work, and the legacy `cphf_*` key names must
    /// keep working as aliases. The response DFT grid is named in the `grid_level_*` family
    /// (cf. `grid_level_skeleton`), with `cpscf` as its scope.
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
        assert_eq!(config.resp.tol, 1.0e-8);
        assert_eq!(config.resp.max_cycle, 99);
        assert_eq!(config.resp.level_shift, 0.0);
        assert_eq!(config.resp.grid_level, Some(2));
        assert_eq!(config.nucgrad.grid_level_skeleton, Some(4));
        assert!(!config.nucgrad.grid_shift_deriv);
        assert!(config.nucgrad.gau_thermo);
        assert_eq!(config.nucgrad.atm_list, Some(vec![0, 1]));
        assert_eq!(config.general.verbose, Some(3));
        // multipole sub-config defaults
        assert_eq!(config.multipole.orders, vec![1, 2, 3, 4]);
        assert_eq!(config.multipole.origin, None);
        assert_eq!(config.multipole.rdm1_relax, MultipoleRdm1Relax::Relaxed);
        assert!(!config.multipole.rdm1_dump);

        // multipole sub-config keys
        let v = serde_json::json!({
            "multipole_orders": [1, 2],
            "multipole_origin": [0.5, -0.5, 1.0],
            "multipole_rdm1_relax": "unrelaxed",
            "multipole_rdm1_dump": true,
        });
        let config: AnalDrvConfig = serde_json::from_value(v).unwrap();
        assert_eq!(config.multipole.orders, vec![1, 2]);
        assert_eq!(config.multipole.origin, Some([0.5, -0.5, 1.0]));
        assert_eq!(config.multipole.rdm1_relax, MultipoleRdm1Relax::Unrelaxed);
        assert!(config.multipole.rdm1_dump);

        // the legacy cphf_* key names are accepted as aliases
        let v = serde_json::json!({
            "cphf_tol": 1.0e-7,
            "cphf_max_cycle": 98,
            "cphf_level_shift": 0.1,
            "cphf_lindep": 1.0e-14,
            "cphf_tol_inflation": 999.0,
            "grid_level_cphf": 1,
        });
        let config: AnalDrvConfig = serde_json::from_value(v).unwrap();
        assert_eq!(config.resp.tol, 1.0e-7);
        assert_eq!(config.resp.max_cycle, 98);
        assert_eq!(config.resp.level_shift, 0.1);
        assert_eq!(config.resp.lindep, 1.0e-14);
        assert_eq!(config.resp.tol_inflation, 999.0);
        assert_eq!(config.resp.grid_level, Some(1));

        // an empty section must give exactly the default configuration
        let config: AnalDrvConfig = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(config, AnalDrvConfig::default());

        // serialization stays flat, under the cpscf_* / grid_level_* key names
        let mut config = AnalDrvConfig::default();
        config.resp.tol = 1.0e-8;
        config.resp.max_space = 20;
        config.resp.grid_level = Some(2);
        config.multipole.rdm1_relax = MultipoleRdm1Relax::Unrelaxed;
        let v = serde_json::to_value(&config).unwrap();
        assert_eq!(v["cpscf_tol"], serde_json::json!(1.0e-8));
        assert_eq!(v["cpscf_max_space"], serde_json::json!(20));
        assert_eq!(v["grid_level_cpscf"], serde_json::json!(2));
        assert_eq!(v["multipole_rdm1_relax"], serde_json::json!("unrelaxed"));
    }
}
