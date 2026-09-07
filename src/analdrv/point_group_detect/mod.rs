//! Molecular point-group detection — a pure-std port of Psi4's libmints
//! algorithm (`psi4/driver/qcdb/libmintsmolecule.py`).
//!
//! # Note on AI usage
//!
//! This module is purely written by AI (GLM-5.2 with Claude Code). Human review is virtually nonexistent.
//!
//! I (ajz34) notices some redundant implementations (for examples elements).
//! These files can be refactored in the future, or kept as independent implementations for clarity.
//!
//! As long as functionality retains, I'm not aware if anyone will modify this module freely.
//!
//! Unittests are not included in this module. For more information, see
//! <https://gitee.com/ajz34/rust-showcase-point-group>.
//!
//! # Example
//! ```ignore
//! use rest::analdrv::point_group_detect::SymmMolecule;
//!
//! // H2O (Bohr) -> C2v
//! let mol = SymmMolecule::new(
//!     vec!["O".into(), "H".into(), "H".into()],
//!     vec![[0.0, 0.0, 0.12], [0.0, 1.54, -0.97], [0.0, -1.54, -0.97]],
//! );
//! let pg = mol.detect();
//! assert_eq!(pg.full_name, "C2v");
//! ```
//!
//! See `NOTES.md` and `psi4_ref/` for the reference algorithm and test suite.

pub mod bits;
pub mod detect;
pub mod elements;
pub mod geom;
pub mod linalg;
pub mod matrix;
pub mod molecule;
pub mod vec3;

pub mod interface_to_rest;

pub use detect::PointGroup;
pub use molecule::SymmMolecule;
