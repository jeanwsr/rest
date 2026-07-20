//! Public detection API.

use super::molecule::SymmMolecule;

/// A detected point group.
#[derive(Debug, Clone)]
pub struct PointGroup {
    /// Schoenflies symbol, e.g. `"C2v"`, `"D3d"`, `"Td"`, `"D_inf_h"`, `"S4"`.
    pub full_name: String,
    /// Rotational symmetry number σ (Sn yields n/2, so this may be fractional).
    pub sigma: f64,
}

impl SymmMolecule {
    /// Detect the point group and rotational symmetry number.
    pub fn detect(&self) -> PointGroup {
        let (template, n) = self.detect_inner();
        PointGroup {
            full_name: template.full_name(n),
            sigma: template.sigma(n),
        }
    }
}
