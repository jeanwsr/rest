//! Molecular symmetry: point-group detection and rotational symmetry number.
//!
//! One unified output ([`PointGroupInfo`]) behind two selectable engines:
//!
//! - [`point_group_detect`] (**default**, [`Engine::Psi4`]): port of Psi4's
//!   libmints algorithm (`psi4/driver/qcdb/libmintsmolecule.py`). Mass-aware
//!   (isotopologues break symmetry correctly), detects Sn groups with the
//!   fractional rotational symmetry number σ = n/2, and exposes linearity /
//!   inversion / principal-axis information.
//! - [`classical`] ([`Engine::Classical`]): self-contained classical
//!   symmetry-element search (Beruski & Vidal style). Mass-free; the
//!   Schoenflies label is best-effort (coarse for rare groups such as pure
//!   Sn), but σ is always correct (number of proper rotations).
//!
//! Both engines return the same [`PointGroupInfo`] with labels in the Psi4
//! convention (`"D_inf_h"`, `"C_inf_v"`; single atoms report `"ATOM"`), which
//! keeps printed output stable across engines. The `engine` field records
//! provenance, which also makes cross-engine agreement tests trivial.
//!
//! # Example
//! ```ignore
//! use rest::symmetry::detect_point_group;
//!
//! let pg = detect_point_group(&position_bohr, &elems);
//! println!("point group {} (sigma = {})", pg.label, pg.sigma);
//! ```

pub mod classical;
pub mod point_group_detect;

pub use point_group_detect::{PointGroup, SymmMolecule};

use tensors::MatrixFull;

/// Default position-matching tolerance (Bohr) of the facade
/// ([`detect_point_group`]). Optimized symmetric geometries typically deviate
/// from ideal symmetry by << 1e-2 Bohr.
pub const POS_TOL: f64 = 1.0e-2;

/// Detection engine selection. The default is the Psi4 port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Engine {
    /// Psi4 libmints port ([`point_group_detect`]); mass-aware, Sn-capable.
    #[default]
    Psi4,
    /// Hand-written classical element search ([`classical`]); mass-free.
    Classical,
}

/// Unified point-group detection result, identical in shape for every engine.
#[derive(Debug, Clone)]
pub struct PointGroupInfo {
    /// Schoenflies label in the Psi4 convention, guaranteed pure **ASCII**
    /// (print/log safe): `"C1"`, `"C2v"`, `"Td"`, `"S4"`, `"D_inf_h"`,
    /// `"C_inf_v"`; single atoms report `"ATOM"`.
    pub label: String,
    /// Rotational symmetry number σ (fractional `n/2` for Sn groups).
    pub sigma: f64,
    /// Whether the geometry is linear (`D_inf_h`/`C_inf_v`).
    pub is_linear: bool,
    /// Whether the geometry is invariant under inversion through its center
    /// (vacuously `false` for single atoms).
    pub has_inversion: bool,
    /// Order of the highest proper rotation axis (0 for linear, 1 if only
    /// the identity).
    pub principal_order: u32,
    /// Which engine produced this result.
    pub engine: Engine,
}

/// Detect the point group with the default engine ([`Engine::Psi4`]) and the
/// default tolerance ([`POS_TOL`]).
///
/// Positions are expected in **Bohr** as a `[3, natm]` matrix
/// (`position[[coord, atom]]`); `elems` gives element symbols.
pub fn detect_point_group(position: &MatrixFull<f64>, elems: &[String]) -> PointGroupInfo {
    detect_point_group_tol(position, elems, POS_TOL)
}

/// Detect with the default engine and an explicit tolerance (Bohr).
pub fn detect_point_group_tol(position: &MatrixFull<f64>, elems: &[String], tol: f64) -> PointGroupInfo {
    detect_with_engine(position, elems, Engine::default(), tol)
}

/// Detect with the default engine, overriding per-atom masses (amu), e.g. for
/// isotopologues. Coordinates are `[[x, y, z]; natm]` in **Bohr**.
pub fn detect_point_group_with_masses(
    symbols: &[String],
    coords: &[[f64; 3]],
    masses: &[f64],
    tol: f64,
) -> PointGroupInfo {
    let mol = SymmMolecule::new(symbols.to_vec(), coords.to_vec())
        .with_masses(masses.to_vec())
        .with_tol(tol);
    point_group_info_psi4(&mol.detect())
}

/// Detect with an explicit engine and tolerance.
pub fn detect_with_engine(
    position: &MatrixFull<f64>,
    elems: &[String],
    engine: Engine,
    tol: f64,
) -> PointGroupInfo {
    match engine {
        Engine::Psi4 => {
            let coords: Vec<[f64; 3]> = (0..elems.len())
                .map(|a| [position[[0, a]], position[[1, a]], position[[2, a]]])
                .collect();
            let mol = SymmMolecule::new(elems.to_vec(), coords).with_tol(tol);
            point_group_info_psi4(&mol.detect())
        },
        Engine::Classical => classical::detect_classical(position, elems, tol),
    }
}

fn point_group_info_psi4(pg: &PointGroup) -> PointGroupInfo {
    PointGroupInfo {
        label: pg.full_name.clone(),
        sigma: pg.sigma,
        is_linear: pg.is_linear,
        has_inversion: pg.has_inversion,
        principal_order: pg.principal_order,
        engine: Engine::Psi4,
    }
}

