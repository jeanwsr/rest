//! Geometry-level helpers operating on `N×3` point sets. Port of the free
//! functions in `psi4/driver/qcdb/libmintsmolecule.py`:
//! `matrix_3d_rotation`, `matrix_3d_rotation_Cn`, `equal_but_for_row_order`,
//! `atom_present_in_geom`, and the `atom_at_position` lookup.

use super::matrix::{householder, points_apply, rodrigues};

/// Rotate a set of row-vectors `mat` about `axis` by `phi` radians. If `sn` is
/// true, additionally reflect through the plane perpendicular to `axis`
/// (improper rotation). Reference: `matrix_3d_rotation` (libmintsmolecule.py:3167).
pub fn matrix_3d_rotation(mat: &[[f64; 3]], axis: &[f64; 3], phi: f64, sn: bool) -> Vec<[f64; 3]> {
    let r = rodrigues(axis, phi);
    let mut rotated = points_apply(mat, &r);
    if sn {
        let h = householder(axis);
        rotated = points_apply(&rotated, &h);
    }
    rotated
}

/// Find the highest `n` such that a `Cn` (proper rotation if `reflect=false`,
/// improper `Sn` if `reflect=true`) about `axis` maps `coord` onto itself.
/// `max_cn_to_check` of 0 means "probe up to `coord.len()`". Reference:
/// `matrix_3d_rotation_Cn` (libmintsmolecule.py:3148).
pub fn matrix_3d_rotation_cn(
    coord: &[[f64; 3]],
    axis: &[f64; 3],
    reflect: bool,
    tol: f64,
    max_cn_to_check: usize,
) -> usize {
    let max_possible = if max_cn_to_check == 0 { coord.len() } else { max_cn_to_check };
    let mut cn = 1; // C1 always present
    let two_pi = 2.0 * std::f64::consts::PI;
    for n in 2..=max_possible {
        let rotated = matrix_3d_rotation(coord, axis, two_pi / n as f64, reflect);
        if equal_but_for_row_order(coord, &rotated, tol) {
            cn = n;
        }
    }
    cn
}

/// Set equality of two `N×3` geometries ignoring row order: every row of `mat`
/// must match *some* row of `rhs` element-wise within `tol`. Reference:
/// `equal_but_for_row_order` (libmintsmolecule.py:3228).
pub fn equal_but_for_row_order(mat: &[[f64; 3]], rhs: &[[f64; 3]], tol: f64) -> bool {
    'outer: for m in 0..mat.len() {
        for m_rhs in 0..rhs.len() {
            let mut matched = true;
            for n in 0..mat[m].len() {
                if (mat[m][n] - rhs[m_rhs][n]).abs() > tol {
                    matched = false;
                    break;
                }
            }
            if matched {
                continue 'outer; // found a matching row for row m
            }
        }
        return false; // no matching row for row m
    }
    true
}

/// Is there an atom (row) in `geom` within `tol` of point `b` (Euclidean)?
/// Reference: `atom_present_in_geom` (libmintsmolecule.py:3136).
pub fn atom_present_in_geom(geom: &[[f64; 3]], b: &[f64; 3], tol: f64) -> bool {
    for a in geom {
        let d0 = a[0] - b[0];
        let d1 = a[1] - b[1];
        let d2 = a[2] - b[2];
        if (d0 * d0 + d1 * d1 + d2 * d2).sqrt() < tol {
            return true;
        }
    }
    false
}

/// Index of the nearest atom to `b`, if within `tol`. Reference:
/// `Molecule.atom_at_position` (libmintsmolecule.py:1153) — nearest-neighbor
/// with squared-distance cutoff `tol*tol`. Returns `None` if none within tol.
pub fn atom_at_position(geom: &[[f64; 3]], b: &[f64; 3], tol: f64) -> Option<usize> {
    if geom.is_empty() {
        return None;
    }
    let mut best = 0usize;
    let mut best_d2 = f64::INFINITY;
    for (i, a) in geom.iter().enumerate() {
        let d0 = a[0] - b[0];
        let d1 = a[1] - b[1];
        let d2 = a[2] - b[2];
        let d2 = d0 * d0 + d1 * d1 + d2 * d2;
        if d2 < best_d2 {
            best_d2 = d2;
            best = i;
        }
    }
    if best_d2 < tol * tol {
        Some(best)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_order_equality() {
        let a = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]];
        let b = [[2.0, 2.0, 2.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        assert!(equal_but_for_row_order(&a, &b, 1e-9));
        let c = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [9.0, 9.0, 9.0]];
        assert!(!equal_but_for_row_order(&a, &c, 1e-9));
    }

    #[test]
    fn cn_about_z_square() {
        // four points on the unit circle in xy: C4 about z
        let pts = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]];
        let cn = matrix_3d_rotation_cn(&pts, &[0.0, 0.0, 1.0], false, 1e-9, 0);
        assert_eq!(cn, 4);
    }

    #[test]
    fn atom_at_position_nearest() {
        let g = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        assert_eq!(atom_at_position(&g, &[1.05, 0.0, 0.0], 0.1), Some(1));
        assert_eq!(atom_at_position(&g, &[5.0, 5.0, 5.0], 0.1), None);
    }
}
