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
    /// Schoenflies label in the Psi4 convention: `"C1"`, `"C2v"`, `"Td"`,
    /// `"S4"`, `"D_inf_h"`, `"C_inf_v"`; single atoms report `"ATOM"`.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_pos(flat: &[f64]) -> MatrixFull<f64> {
        let natm = flat.len() / 3;
        let mut m = MatrixFull::new([3, natm], 0.0);
        for a in 0..natm {
            for c in 0..3 { m[[c, a]] = flat[a * 3 + c]; }
        }
        m
    }

    fn elems_of(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    /// Both engines must produce the identical label and sigma on these
    /// exact-symmetry geometries.
    fn assert_engines_agree(flat: &[f64], names: &[&str], want_label: &str, want_sigma: f64) {
        let pos = mk_pos(flat);
        let elems = elems_of(names);
        for engine in [Engine::Psi4, Engine::Classical] {
            let pg = detect_with_engine(&pos, &elems, engine, POS_TOL);
            assert_eq!(pg.engine, engine);
            assert_eq!(pg.label, want_label, "label mismatch for engine {engine:?}");
            assert!(
                (pg.sigma - want_sigma).abs() < 1.0e-12,
                "sigma mismatch for engine {engine:?}: {} (want {want_sigma})",
                pg.sigma
            );
        }
    }

    #[test]
    fn ch4_is_td_sigma12() {
        assert_engines_agree(
            &[
                0.0, 0.0, 0.0,
                1.0897, 1.0897, 1.0897,
                -1.0897, -1.0897, 1.0897,
                1.0897, -1.0897, -1.0897,
                -1.0897, 1.0897, -1.0897,
            ],
            &["C", "H", "H", "H", "H"],
            "Td", 12.0,
        );
    }

    #[test]
    fn h2o_is_c2v_sigma2() {
        assert_engines_agree(
            &[0.0, 0.0, 0.0, 0.0, 0.757, 0.587, 0.0, -0.757, 0.587],
            &["O", "H", "H"],
            "C2v", 2.0,
        );
    }

    #[test]
    fn co2_is_dinfh_sigma2() {
        let pos = mk_pos(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.16, 0.0, 0.0, -1.16]);
        let elems = elems_of(&["C", "O", "O"]);
        for engine in [Engine::Psi4, Engine::Classical] {
            let pg = detect_with_engine(&pos, &elems, engine, POS_TOL);
            assert_eq!(pg.label, "D_inf_h");
            assert_eq!(pg.sigma, 2.0);
            assert!(pg.is_linear);
            assert!(pg.has_inversion);
            assert_eq!(pg.principal_order, 0);
        }
    }

    #[test]
    fn nh3_is_c3v_sigma3() {
        // built from exact trigonometry so the C3 symmetry holds at machine
        // precision (the Psi4 engine's degeneracy gates are relative)
        let (r, zh, zn) = (0.9392_f64, -0.2_f64, 0.25_f64);
        let (c120, s120) = (-0.5_f64, 3.0_f64.sqrt() / 2.0);
        assert_engines_agree(
            &[
                0.0, 0.0, zn,
                r, 0.0, zh,
                r * c120, r * s120, zh,
                r * c120, -r * s120, zh,
            ],
            &["N", "H", "H", "H"],
            "C3v", 3.0,
        );
    }

    #[test]
    fn c2h6_staggered_is_d3d_sigma6() {
        let (r, zh, zc) = (1.028, 1.133, 0.77);
        let (c120, s120) = (-0.5, 3.0_f64.sqrt() / 2.0);
        assert_engines_agree(
            &[
                0.0, 0.0, zc,
                0.0, 0.0, -zc,
                r, 0.0, zh,
                r * c120, r * s120, zh,
                r * c120, -r * s120, zh,
                r * 0.5, r * s120, -zh,
                -r, 0.0, -zh,
                r * 0.5, -r * s120, -zh,
            ],
            &["C", "C", "H", "H", "H", "H", "H", "H"],
            "D3d", 6.0,
        );
    }

    #[test]
    fn hcn_is_cinfv_sigma1() {
        assert_engines_agree(
            &[0.0, 0.0, 0.0, 0.0, 0.0, -1.06, 0.0, 0.0, 1.16],
            &["C", "H", "N"],
            "C_inf_v", 1.0,
        );
    }

    #[test]
    fn hfco_is_cs_sigma1() {
        // planar formyl fluoride: single mirror plane (the molecular plane)
        assert_engines_agree(
            &[
                0.0, 0.0, 0.0,
                0.0, 0.0, 1.2,
                0.9, 0.0, -0.3,
                -0.85, 0.0, -0.35,
            ],
            &["C", "O", "F", "H"],
            "Cs", 1.0,
        );
    }

    #[test]
    fn single_atom_is_atom_sigma1() {
        assert_engines_agree(&[0.0, 0.0, 0.0], &["Ne"], "ATOM", 1.0);
        let pos = mk_pos(&[0.0, 0.0, 0.0]);
        let elems = elems_of(&["Ne"]);
        for engine in [Engine::Psi4, Engine::Classical] {
            let pg = detect_with_engine(&pos, &elems, engine, POS_TOL);
            assert!(!pg.is_linear);
            assert!(!pg.has_inversion); // vacuous, unified to false
            assert_eq!(pg.principal_order, 1);
        }
    }

    #[test]
    fn ch3d_is_c3v_sigma3() {
        // D breaks the tetrahedral symmetry element-wise (both engines)
        assert_engines_agree(
            &[
                0.0, 0.0, 0.0,
                1.0897, 1.0897, 1.0897,
                -1.0897, -1.0897, 1.0897,
                1.0897, -1.0897, -1.0897,
                -1.0897, 1.0897, -1.0897,
            ],
            &["C", "H", "H", "H", "D"],
            "C3v", 3.0,
        );
    }

    #[test]
    fn isotopologue_mass_breaks_symmetry() {
        // 13CH4: same element symbols, one heavier H. The Psi4 engine is
        // mass-aware and must reduce Td -> C3v (sigma 3); the mass-free
        // classical engine (by design) still reports Td / 12.
        let coords = vec![
            [0.0, 0.0, 0.0],
            [1.0897, 1.0897, 1.0897],
            [-1.0897, -1.0897, 1.0897],
            [1.0897, -1.0897, -1.0897],
            [-1.0897, 1.0897, -1.0897],
        ];
        let symbols = elems_of(&["C", "H", "H", "H", "H"]);
        let masses = [12.0, 1.00783, 1.00783, 1.00783, 13.003355];
        let pg = detect_point_group_with_masses(&symbols, &coords, &masses, POS_TOL);
        assert_eq!(pg.engine, Engine::Psi4);
        assert_eq!(pg.label, "C3v");
        assert_eq!(pg.sigma, 3.0);

        // sanity: identical masses restore the full Td symmetry
        let masses = [12.0, 1.00783, 1.00783, 1.00783, 1.00783];
        let pg = detect_point_group_with_masses(&symbols, &coords, &masses, POS_TOL);
        assert_eq!(pg.label, "Td");
        assert_eq!(pg.sigma, 12.0);
    }

    #[test]
    fn disphenoid_is_d2d_sigma4() {
        // 4-atom disphenoid (equal elements): S4(z) axis, two perpendicular
        // C2' axes, two sigma-d planes, no inversion => D2d. Exercises the
        // symmetric-top Sn-vs-Dnd discrimination on both engines.
        let pos = mk_pos(&[
            1.0, 2.0, 1.0,
            -1.0, -2.0, 1.0,
            -2.0, 1.0, -1.0,
            2.0, -1.0, -1.0,
        ]);
        let elems = elems_of(&["C", "C", "C", "C"]);
        for engine in [Engine::Psi4, Engine::Classical] {
            let pg = detect_with_engine(&pos, &elems, engine, POS_TOL);
            assert_eq!(pg.label, "D2d", "label mismatch for engine {engine:?}");
            assert_eq!(pg.sigma, 4.0, "sigma mismatch for engine {engine:?}");
            assert!(!pg.has_inversion);
            assert_eq!(pg.principal_order, 2);
        }
    }

    #[test]
    fn sn_template_has_fractional_sigma() {
        // The Sn branch of the Psi4 engine yields sigma = n/2 (fractional).
        // Pure-Sn geometries are chemically pathological (a generic S4 orbit
        // always forces the sigma_d/C2' operations, upgrading to D2d), so the
        // fractional machinery is tested at the template level.
        use point_group_detect::molecule::Tmpl;
        assert_eq!(Tmpl::Sn.sigma(4), 2.0);
        assert_eq!(Tmpl::Sn.sigma(8), 4.0);
        assert_eq!(Tmpl::Sn.full_name(4), "S4");
        assert_eq!(Tmpl::Sn.principal_order(4), 2);
    }

    #[test]
    fn perturbed_geometries_keep_symmetry_at_default_tol() {
        // real-world (optimized, slightly unsymmetrized) geometries: ~1e-4
        // Bohr deviations from ideal symmetry must not degrade detection at
        // the default tolerance. This exercises the tolerance-relative gates
        // (rotor classification, sigma_v candidates, perpendicular C2 scan)
        // that replace Psi4's absolute thresholds (Psi4 symmetrizes
        // coordinates first; REST detects on raw input).
        let d = 3.0e-4_f64;

        // NH3 perturbed: stays C3v (sigma 3), not Cs/C1
        let (r, zh, zn) = (0.9392_f64, -0.2_f64, 0.25_f64);
        let (c120, s120) = (-0.5_f64, 3.0_f64.sqrt() / 2.0);
        let pos = mk_pos(&[
            0.0, 0.0, zn + d,
            r, d, zh,
            r * c120, r * s120 + d, zh,
            r * c120, -r * s120, zh - d,
        ]);
        let elems = elems_of(&["N", "H", "H", "H"]);
        let pg = detect_with_engine(&pos, &elems, Engine::Psi4, POS_TOL);
        assert_eq!(pg.label, "C3v");
        assert_eq!(pg.sigma, 3.0);

        // staggered C2H6 perturbed: stays D3d (sigma 6), not S6/C2
        let (r, zh, zc) = (1.028_f64, 1.133_f64, 0.77_f64);
        let pos = mk_pos(&[
            0.0, 0.0, zc,
            0.0, 0.0, -zc,
            r, 0.0, zh,
            r * c120, r * s120, zh,
            r * c120, -r * s120, zh,
            r * 0.5, r * s120 + d, -zh,
            -r, 0.0, -zh,
            r * 0.5, -r * s120, -zh + d,
        ]);
        let elems = elems_of(&["C", "C", "H", "H", "H", "H", "H", "H"]);
        let pg = detect_with_engine(&pos, &elems, Engine::Psi4, POS_TOL);
        assert_eq!(pg.label, "D3d");
        assert_eq!(pg.sigma, 6.0);
        assert!(pg.has_inversion);
    }

    #[test]
    fn unknown_elements_are_interned() {
        // Two elements absent from the mass table must not be conflated.
        // Au-Pt diatomic: correct answer is C_inf_v (no inversion since
        // Au != Pt). If both collapsed to the same pseudo-Z (the old
        // `unwrap_or(0)` behavior), the molecule would falsely acquire an
        // inversion center and report D_inf_h / sigma 2.
        let pos = mk_pos(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let elems = elems_of(&["Au", "Pt"]);
        let pg = detect_with_engine(&pos, &elems, Engine::Psi4, POS_TOL);
        assert_eq!(pg.label, "C_inf_v");
        assert_eq!(pg.sigma, 1.0);
        assert!(pg.is_linear);
        assert!(!pg.has_inversion);
    }
}
