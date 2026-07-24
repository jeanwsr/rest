//! 3D vector helpers. Direct port of `psi4/driver/qcdb/vecutil.py` vector
//! routines. Vectors are `[f64; 3]` (length-3 throughout, matching the
//! reference). Pure std.

pub type Vec3 = [f64; 3];

pub const ZERO: f64 = 1.0e-14;

#[inline]
pub fn norm(v: &Vec3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[inline]
pub fn add(v: &Vec3, u: &Vec3) -> Vec3 {
    [v[0] + u[0], v[1] + u[1], v[2] + u[2]]
}

#[inline]
pub fn sub(v: &Vec3, u: &Vec3) -> Vec3 {
    [v[0] - u[0], v[1] - u[1], v[2] - u[2]]
}

#[inline]
pub fn dot(v: &Vec3, u: &Vec3) -> f64 {
    v[0] * u[0] + v[1] * u[1] + v[2] * u[2]
}

#[inline]
pub fn scale(v: &Vec3, d: f64) -> Vec3 {
    [v[0] * d, v[1] * d, v[2] * d]
}

/// Element-wise (component-wise) product. Reference: `naivemult`.
#[inline]
pub fn naivemult(v: &Vec3, u: &Vec3) -> Vec3 {
    [v[0] * u[0], v[1] * u[1], v[2] * u[2]]
}

#[inline]
pub fn normalize(v: &Vec3) -> Vec3 {
    let m = norm(v);
    [v[0] / m, v[1] / m, v[2] / m]
}

#[inline]
pub fn cross(v: &Vec3, u: &Vec3) -> Vec3 {
    [
        v[1] * u[2] - v[2] * u[1],
        v[2] * u[0] - v[0] * u[2],
        v[0] * u[1] - v[1] * u[0],
    ]
}

/// Unit vector perpendicular to length-3 vectors `u` and `v`.
/// Reference: `vecutil.py:perp_unit`. Handles the degenerate cross-product
/// case by choosing a vector perpendicular to the larger of `u`, `v` in the
/// plane of their two largest components.
pub fn perp_unit(u: &Vec3, v: &Vec3) -> Vec3 {
    // try cross product
    let result = cross(u, v);
    let rdotr = dot(&result, &result);

    if rdotr < 1.0e-16 {
        // cross product too small to normalize: pick the larger of u, v
        let (d, dotprodd) = if dot(u, u) < dot(v, v) {
            (*v, dot(v, v))
        } else {
            (*u, dot(u, u))
        };

        if dotprodd < 1.0e-16 {
            // both tiny -> arbitrary
            return [1.0, 0.0, 0.0];
        }

        // choose a vector perpendicular to d, in the plane of d's two largest
        // components (90° rotation within that plane).
        let absd = [d[0].abs(), d[1].abs(), d[2].abs()];
        let (axis0, axis1) = if (absd[1] - absd[0]) > 1.0e-12 {
            if (absd[2] - absd[0]) > 1.0e-12 {
                (1, 2)
            } else {
                (1, 0)
            }
        } else if (absd[2] - absd[1]) > 1.0e-12 {
            (0, 2)
        } else {
            (0, 1)
        };
        let mut r = [0.0, 0.0, 0.0];
        r[axis0] = d[axis1];
        r[axis1] = -d[axis0];
        normalize(&r)
    } else {
        scale(&result, 1.0 / rdotr.sqrt())
    }
}

/// Rotate vector `v` about `axis` by `theta` radians.
/// Reference: `vecutil.py:rotate`. Used by `is_axis`; not on the main
/// `set_full_point_group` path (which uses the matrix form in `geom.rs`).
pub fn rotate(v: &Vec3, theta: f64, axis: &Vec3) -> Vec3 {
    // (reference computes `unitaxis = normalize(axis)` but never uses it; omitted)
    // parallel component along axis
    let parallel = scale(axis, dot(v, axis) / dot(axis, axis));
    let perpendicular = sub(v, &parallel);
    // third orthonormal axis
    let mut third = perp_unit(axis, &perpendicular);
    third = scale(&third, norm(&perpendicular));

    let mut result = add(&parallel, &add(&scale(&perpendicular, theta.cos()), &scale(&third, theta.sin())));
    for item in result.iter_mut() {
        if item.abs() < ZERO {
            *item = 0.0;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_basics() {
        assert!((norm(&[3.0, 4.0, 0.0]) - 5.0).abs() < 1e-15);
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, -5.0, 6.0]), 4.0 - 10.0 + 18.0);
        assert_eq!(cross(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]);
        let n = normalize(&[0.0, 0.0, 5.0]);
        assert!((n[2] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn perp_unit_basic() {
        let p = perp_unit(&[0.0, 0.0, 1.0], &[1.0, 0.0, 0.0]);
        // perpendicular to both -> ±y
        assert!((dot(&p, &[0.0, 0.0, 1.0]).abs()) < 1e-15);
        assert!((dot(&p, &[1.0, 0.0, 0.0]).abs()) < 1e-15);
        assert!((norm(&p) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn rotate_about_z() {
        let r = rotate(&[1.0, 0.0, 0.0], std::f64::consts::FRAC_PI_2, &[0.0, 0.0, 1.0]);
        assert!((r[0] - 0.0).abs() < 1e-15);
        assert!((r[1] - 1.0).abs() < 1e-15);
        assert!((r[2] - 0.0).abs() < 1e-15);
    }
}
