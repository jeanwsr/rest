//! Bitwise encoding of the D2h subgroups and symmetry operations. Verbatim
//! port of `psi4/driver/qcdb/libmintspointgrp.py` (`SymmOps`, `PointGroups`,
//! `similar`, `bits_to_basic_name`).
//!
//! Each of the 8 D2h operations is a bit:
//! `E=0, C2_z=1, C2_y=2, C2_x=4, i=8, σ_xy=16, σ_xz=32, σ_yz=64` (ID=128 sentinel).
//! A point group is the OR of its members' bits.

#![allow(dead_code)]

pub mod symm_ops {
    pub const E: u8 = 0;
    pub const C2_Z: u8 = 1;
    pub const C2_Y: u8 = 2;
    pub const C2_X: u8 = 4;
    pub const I: u8 = 8;
    pub const SIGMA_XY: u8 = 16;
    pub const SIGMA_XZ: u8 = 32;
    pub const SIGMA_YZ: u8 = 64;
    pub const ID: u8 = 128;
}

pub mod point_groups {
    use super::symm_ops::*;
    pub const C1: u8 = E;
    pub const CI: u8 = E | I;
    pub const C2X: u8 = E | C2_X;
    pub const C2Y: u8 = E | C2_Y;
    pub const C2Z: u8 = E | C2_Z;
    pub const CSZ: u8 = E | SIGMA_XY;
    pub const CSY: u8 = E | SIGMA_XZ;
    pub const CSX: u8 = E | SIGMA_YZ;
    pub const D2: u8 = E | C2_X | C2_Y | C2_Z;
    pub const C2VX: u8 = E | C2_X | SIGMA_XY | SIGMA_XZ;
    pub const C2VY: u8 = E | C2_Y | SIGMA_XY | SIGMA_YZ;
    pub const C2VZ: u8 = E | C2_Z | SIGMA_XZ | SIGMA_YZ;
    pub const C2HX: u8 = E | C2_X | SIGMA_YZ | I;
    pub const C2HY: u8 = E | C2_Y | SIGMA_XZ | I;
    pub const C2HZ: u8 = E | C2_Z | SIGMA_XY | I;
    pub const D2H: u8 = E | C2_X | C2_Y | C2_Z | I | SIGMA_XY | SIGMA_XZ | SIGMA_YZ;
}

/// The 7 non-identity operations tested by `find_highest_point_group`, paired
/// with the diagonal of their 3×3 matrix `[d00, d11, d22]` (used for the
/// element-wise `naivemult` application). Order matches the reference.
pub fn tested_ops() -> [(u8, [f64; 3]); 7] {
    use symm_ops::*;
    [
        (C2_Z, [-1.0, -1.0, 1.0]),
        (C2_Y, [-1.0, 1.0, -1.0]),
        (C2_X, [1.0, -1.0, -1.0]),
        (I, [-1.0, -1.0, -1.0]),
        (SIGMA_XY, [1.0, 1.0, -1.0]),
        (SIGMA_XZ, [1.0, -1.0, 1.0]),
        (SIGMA_YZ, [-1.0, 1.0, 1.0]),
    ]
}

/// Simple (non-directional) Schoenflies symbol from bits, lowercase.
/// Reference: `bits_to_basic_name`. Returns one of
/// `c1, ci, c2, cs, d2, c2v, c2h, d2h`.
pub fn bits_to_basic_name(bits: u8) -> &'static str {
    use point_groups::*;
    match bits {
        C1 => "c1",
        CI => "ci",
        C2X | C2Y | C2Z => "c2",
        CSX | CSY | CSZ => "cs",
        D2 => "d2",
        C2VX | C2VY | C2VZ => "c2v",
        C2HX | C2HY | C2HZ => "c2h",
        D2H => "d2h",
        _ => "c1", // reference raises; we degrade to c1 for unexpected combos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_names() {
        use point_groups::*;
        assert_eq!(bits_to_basic_name(C1), "c1");
        assert_eq!(bits_to_basic_name(C2VZ), "c2v");
        assert_eq!(bits_to_basic_name(D2H), "d2h");
        assert_eq!(bits_to_basic_name(CSX), "cs");
    }
}
