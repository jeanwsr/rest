//! 3×3 matrix helpers. Port of the matrix utilities used by Psi4's libmints
//! (`vecutil.py`: `zero`, `identity`, `mult`, `transpose`, `matadd`) plus the
//! Rodrigues rotation matrix and Householder reflection matrix used by
//! `matrix_3d_rotation`. Pure std.

pub type Matrix3 = [[f64; 3]; 3];

#[inline]
pub fn zero() -> Matrix3 {
    [[0.0; 3]; 3]
}

#[inline]
pub fn identity() -> Matrix3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

#[inline]
pub fn transpose(m: &Matrix3) -> Matrix3 {
    [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]]
}

/// 3×3 · 3×3 matmul. Reference: `vecutil.mult`.
pub fn matmul(a: &Matrix3, b: &Matrix3) -> Matrix3 {
    let mut r = zero();
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += a[i][k] * b[k][j];
            }
            r[i][j] = s;
        }
    }
    r
}

/// N×3 (rows are points) · 3×3 -> N×3. Reference: `vecutil.mult` with the
/// `TypeError` vector fallback not needed since we always pass 3 columns.
pub fn points_matmul(points: &[[f64; 3]], m: &Matrix3) -> Vec<[f64; 3]> {
    let mut r = vec![[0.0; 3]; points.len()];
    for i in 0..points.len() {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += points[i][k] * m[k][j];
            }
            r[i][j] = s;
        }
    }
    r
}

/// N×3 (rows) · 3×3ᵀ -> N×3, i.e. each row `r` becomes `m · r` (treating rows
/// as column vectors transformed by `m`). This is `coord · Rᵀ` in the reference
/// (`matrix_3d_rotation`), where `R` is the row-vector rotation matrix.
pub fn points_apply(points: &[[f64; 3]], m: &Matrix3) -> Vec<[f64; 3]> {
    let mut r = vec![[0.0; 3]; points.len()];
    for i in 0..points.len() {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += m[j][k] * points[i][k];
            }
            r[i][j] = s;
        }
    }
    r
}

/// Rodrigues rotation matrix for rotation by `phi` about `axis` (need not be
/// normalized). Returns the row-vector form `R` from `matrix_3d_rotation`
/// (`libmintsmolecule.py:3167`); the caller applies it as `coord · Rᵀ` via
/// [`points_apply`].
///
/// **Careful:** the off-diagonal signs follow the reference's asymmetric
/// pattern, not the symmetric textbook form.
pub fn rodrigues(axis: &[f64; 3], phi: f64) -> Matrix3 {
    let w = super::vec3::normalize(axis);
    let (wx, wy, wz) = (w[0], w[1], w[2]);
    let cp = 1.0 - phi.cos();
    let s = phi.sin();
    let c = phi.cos();
    [
        [wx * wx * cp + c, wx * wy * cp - s * wz, wx * wz * cp + s * wy],
        [wy * wx * cp + s * wz, wy * wy * cp + c, wy * wz * cp - s * wx],
        [wz * wx * cp - s * wy, wz * wy * cp + s * wx, wz * wz * cp + c],
    ]
}

/// Householder reflection matrix `H = I - 2 w wᵀ` for reflection through the
/// plane perpendicular to `axis` (normalized internally). Used for the improper
/// (Sn) branch of `matrix_3d_rotation`. Reference: `libmintsmolecule.py:3201`.
pub fn householder(axis: &[f64; 3]) -> Matrix3 {
    let w = super::vec3::normalize(axis);
    let (wx, wy, wz) = (w[0], w[1], w[2]);
    let mut h = identity();
    h[0][0] -= 2.0 * wx * wx;
    h[1][1] -= 2.0 * wy * wy;
    h[2][2] -= 2.0 * wz * wz;
    h[1][0] += 2.0 * wx * wy;
    h[2][0] += 2.0 * wx * wz;
    h[2][1] += 2.0 * wy * wz;
    h[0][1] += 2.0 * wx * wy;
    h[0][2] += 2.0 * wx * wz;
    h[1][2] += 2.0 * wy * wz;
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_identity() {
        let m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let id = identity();
        let p = matmul(&m, &id);
        for i in 0..3 {
            for j in 0..3 {
                assert!((p[i][j] - m[i][j]).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn rodrigues_about_z_90() {
        // rotating the x-axis (as a row) about z by 90° should give +y
        let r = rodrigues(&[0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2);
        let out = points_apply(&[[1.0, 0.0, 0.0]], &r);
        assert!((out[0][0]).abs() < 1e-15);
        assert!((out[0][1] - 1.0).abs() < 1e-15);
        assert!((out[0][2]).abs() < 1e-15);
    }

    #[test]
    fn householder_z_flips_xy() {
        // reflection through plane perp to z (the xy-plane) flips z, keeps x,y
        let h = householder(&[0.0, 0.0, 1.0]);
        let out = points_apply(&[[1.0, 2.0, 3.0]], &h);
        assert!((out[0][0] - 1.0).abs() < 1e-15);
        assert!((out[0][1] - 2.0).abs() < 1e-15);
        assert!((out[0][2] + 3.0).abs() < 1e-15);
    }
}
