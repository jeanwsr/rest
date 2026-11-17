//! Public detection API.

use super::molecule::SymmMolecule;

/// A detected point group.
#[derive(Debug, Clone)]
pub struct PointGroup {
    /// Schoenflies symbol, e.g. `"C2v"`, `"D3d"`, `"Td"`, `"D_inf_h"`, `"S4"`
    /// (single atoms report `"ATOM"`).
    pub full_name: String,
    /// Rotational symmetry number σ (Sn yields n/2, so this may be fractional).
    pub sigma: f64,
    /// Whether the molecule is linear (`D_inf_h`/`C_inf_v`).
    pub is_linear: bool,
    /// Whether the geometry is invariant under inversion through its center
    /// of mass (vacuously `false` for single atoms).
    pub has_inversion: bool,
    /// Order of the highest proper rotation axis (0 for linear, 1 if only
    /// the identity; see [`super::molecule::Tmpl::principal_order`]).
    pub principal_order: u32,
}

impl SymmMolecule {
    /// Detect the point group and rotational symmetry number.
    pub fn detect(&self) -> PointGroup {
        let (template, n, is_linear, has_inv) = self.detect_inner();
        PointGroup {
            full_name: template.full_name(n),
            sigma: template.sigma(n),
            is_linear,
            has_inversion: has_inv,
            principal_order: template.principal_order(n),
        }
    }
}
