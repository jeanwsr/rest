//! Generate Lebedev angular grids.
//!
//! The grids are read from the original static coordinate and weight tables.
//!
//! # References
//!
//! V.I. Lebedev, and D.N. Laikov
//! "A quadrature formula for the sphere of the 131st
//! algebraic order of accuracy"
//! Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.

use super::parameters::LEBEDEV_NGRID;
use super::tables;

/// Generate Lebedev angular grid for given number of points.
///
/// Valid numbers of points are the entries of [`LEBEDEV_NGRID`] except the
/// first one (1). Weights are normalized such that they sum up to 1.
pub(super) fn angular_grid(num_points: usize) -> (Vec<(f64, f64, f64)>, Vec<f64>) {
    let offsets = tables::offsets::offsets();

    let offset: usize = match offsets.get(&num_points) {
        Some(v) => *v,
        None => panic!(
            "angular_grid called with unsupported num_points, allowed are: {:?}",
            &LEBEDEV_NGRID[1..]
        ),
    };

    (
        tables::coordinates::COORDINATES[offset..(offset + num_points)].to_vec(),
        tables::weights::WEIGHTS[offset..(offset + num_points)].to_vec(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Weights of every Lebedev grid sum to 1, and all points on unit sphere.
    #[test]
    fn static_lebedev_grid_sanity() {
        for &n in &LEBEDEV_NGRID[1..] {
            let (grid, weights) = angular_grid(n);
            assert_eq!(grid.len(), n);
            let sum: f64 = weights.iter().sum();
            // Restored legacy table rounded decimal weights accumulate ~4e-13;
            // retain exact published table values.
            assert!((sum - 1.0).abs() < 1.0e-12, "weights sum to {sum} for n = {n}");
            for &(x, y, z) in &grid {
                assert!((x * x + y * y + z * z - 1.0).abs() < 1.0e-13);
            }
        }
    }
}
