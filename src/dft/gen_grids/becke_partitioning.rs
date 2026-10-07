//! Becke partitioning according to _A. D. Becke. The Journal of Chemical Physics 88, 2547-2553 (1988)_.
//! Reference can be found [here](https://doi.org/10.1063/1.454033).
//! 
use rayon::prelude::*;

use super::bragg;
use super::parameters;

/// Which radii enter Becke's atomic-size adjustment.
///
/// * `Becke`: Becke's original expression, J. Chem. Phys. 88, 2547 (1988), using the Bragg
///   radii themselves. This has been REST's behaviour so far.
/// * `Treutler`: the Treutler-Ahlrichs variant, J. Chem. Phys. 102, 346 (1995), which
///   replaces the radii by their square roots. PySCF's `Grids` defaults to this variant
///   (`radi.treutler_atomic_radii_adjust`) and PyFock's `size_adjustment_table` with
///   `scheme='treutler'` implements the same one, so selecting it makes REST reproduce
///   their partition weights and thus their grid weights.
///
/// Both variants satisfy `sum_A pbecke_A = 1` at every grid point. They differ by a
/// quadrature error that decays as the grid is refined, so they agree in the dense-grid limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadiiAdjust {
    Becke,
    Treutler,
}

impl RadiiAdjust {
    /// Parse the `radii_adjust` ctrl keyword. Anything but `treutler` means `Becke`.
    pub fn from_str(name: &str) -> Self {
        match name.trim().to_lowercase().as_str() {
            "treutler" => RadiiAdjust::Treutler,
            _ => RadiiAdjust::Becke,
        }
    }

    /// Radius that enters the adjustment factor.
    #[inline]
    pub fn radius(self, r: f64) -> f64 {
        match self {
            RadiiAdjust::Becke => r,
            RadiiAdjust::Treutler => r.sqrt(),
        }
    }

    /// Becke's `a_AB = u_AB / (u_AB^2 - 1)`, clamped to `[-0.5, 0.5]`.
    ///
    /// Only the ratio of the two radii enters, so their unit is irrelevant.
    #[inline]
    pub fn factor(self, r_a: f64, r_b: f64) -> f64 {
        let ra = self.radius(r_a);
        let rb = self.radius(r_b);
        if (ra - rb).abs() <= parameters::SMALL {
            return 0.0;
        }
        let u_ab = (ra + rb) / (rb - ra);
        (u_ab / (u_ab * u_ab - 1.0)).min(0.5).max(-0.5)
    }
}

// JCP 88, 2547 (1988), eq. 20
#[inline]
fn f3(x: f64, hardness: usize) -> f64 {
    let mut f = x;

    for _ in 0..hardness {
        f *= 1.5 - 0.5 * f * f;
    }

    f
}

fn distance(p1: &(f64, f64, f64), p2: &(f64, f64, f64)) -> f64 {
    let dx = p1.0 - p2.0;
    let dy = p1.1 - p2.1;
    let dz = p1.2 - p2.2;

    (dx * dx + dy * dy + dz * dz).sqrt()
}

// JCP 88, 2547 (1988)
pub fn partitioning_weight(
    center_index: usize,
    center_coordinates_bohr: &[(f64, f64, f64)],
    proton_charges: &[i32],
    grid_coordinates_bohr: (f64, f64, f64),
    hardness: usize,
    radii_adjust: RadiiAdjust,
) -> f64 {
    let num_centers = proton_charges.len();

    let mut pa = vec![1.0; num_centers];

    for ia in 0..num_centers {
        let dist_a = distance(&grid_coordinates_bohr, &center_coordinates_bohr[ia]);

        let r_a = bragg::get_bragg_angstrom(proton_charges[ia]);

        for ib in 0..ia {
            let dist_b = distance(&grid_coordinates_bohr, &center_coordinates_bohr[ib]);

            let r_b = bragg::get_bragg_angstrom(proton_charges[ib]);

            let dist_ab = distance(&center_coordinates_bohr[ia], &center_coordinates_bohr[ib]);

            // JCP 88, 2547 (1988), eq. 11
            let mu_ab = (dist_a - dist_b) / dist_ab;

            let mut nu_ab = mu_ab;
            if (r_a - r_b).abs() > parameters::SMALL {
                // the two variants differ only here: raw radii (Becke) or their square
                // roots (Treutler-Ahlrichs, the PySCF/PyFock default)
                let a_ab = radii_adjust.factor(r_a, r_b);

                nu_ab += a_ab * (1.0 - mu_ab * mu_ab);
            }

            let f = f3(nu_ab, hardness);

            if (1.0 - f).abs() > parameters::SMALL {
                pa[ia] *= 0.5 * (1.0 - f);
                pa[ib] *= 0.5 * (1.0 + f);
            } else {
                // avoid numerical issues
                pa[ia] = 0.0;
            }
        }
    }

    let w: f64 = pa.iter().sum();

    if w.abs() > parameters::SMALL {
        pa[center_index] / w
    } else {
        1.0
    }
}

// ---------------------------------------------------------------------------
// Blocked / parallel evaluation of the same weights
//
// `partitioning_weight` recomputes, at every grid point, quantities that depend on the
// molecule only (interatomic distances, Bragg radii, radii-adjustment factors) and
// allocates one `Vec` per point. The blocked path below hoists all of that into
// `BeckePairTable`, walks one block of grid points at a time with the atom-pair loop on
// the outside, and lets every worker thread reuse its scratch buffers. The arithmetic of
// a single grid point is unchanged, so every weight is bit-identical to the value
// `partitioning_weight` returns for that point.
// ---------------------------------------------------------------------------

/// Target size (in `f64` elements) of one scratch buffer per worker thread; the block
/// length follows from it, so the per-thread scratch is bounded by
/// `2 * SCRATCH_ELEMS * 8 B = 512 KB` regardless of the molecule.
const SCRATCH_ELEMS: usize = 1 << 15;
/// Lower bound of the block length. Keeps the innermost loop long enough to vectorise.
const BLOCK_MIN: usize = 64;
/// Upper bound of the block length. Keeps the scratch within the target size.
const BLOCK_MAX: usize = 2048;

/// Length of the grid block processed in one pass of the blocked kernel.
pub fn block_length(num_centers: usize) -> usize {
    (SCRATCH_ELEMS / num_centers.max(1)).clamp(BLOCK_MIN, BLOCK_MAX)
}

// `f3` with the iteration count known at compile time, so the loop is fully unrolled and
// gives exactly the same sequence of roundings as the runtime loop.
#[inline(always)]
fn f3_const<const H: usize>(x: f64) -> f64 {
    let mut f = x;
    for _ in 0..H {
        f *= 1.5 - 0.5 * f * f;
    }
    f
}

/// Everything in Becke's partitioning that depends on the molecule only.
///
/// Build once per atomic grid and reuse it for every grid point of that atom. The storage
/// is `O(num_centers^2)`, about 17 bytes per atom pair, the same order as the molecular
/// tables of `becke_partitioning_deriv`.
#[derive(Debug, Clone)]
pub struct BeckePairTable {
    num_centers: usize,
    /// `a_ab` of JCP 88, 2547 (1988), eq. (A5). Entry `(i, j)` at `i * n + j`.
    a_ab: Vec<f64>,
    /// Whether the two Bragg radii differ enough for the adjustment to apply.
    use_adjust: Vec<bool>,
    /// Interatomic distances in bohr. Entry `(i, j)` at `i * n + j`; the diagonal is unused.
    dist_ab: Vec<f64>,
}

impl BeckePairTable {
    /// Precompute the pair constants of `center_coordinates_bohr` and `proton_charges`.
    pub fn new(
        center_coordinates_bohr: &[(f64, f64, f64)],
        proton_charges: &[i32],
        radii_adjust: RadiiAdjust,
    ) -> Self {
        assert_eq!(
            center_coordinates_bohr.len(),
            proton_charges.len(),
            "BeckePairTable: coordinates and proton charges must have the same length"
        );
        let n = proton_charges.len();
        let radii: Vec<f64> = proton_charges
            .iter()
            .map(|&charge| bragg::get_bragg_angstrom(charge))
            .collect();
        let mut table = BeckePairTable {
            num_centers: n,
            a_ab: vec![0.0; n * n],
            use_adjust: vec![false; n * n],
            dist_ab: vec![0.0; n * n],
        };
        for i in 0..n {
            for j in 0..i {
                let dist = distance(&center_coordinates_bohr[i], &center_coordinates_bohr[j]);
                table.dist_ab[i * n + j] = dist;
                table.dist_ab[j * n + i] = dist;
                // the same predicate and the same expression as partitioning_weight
                if (radii[i] - radii[j]).abs() > parameters::SMALL {
                    table.use_adjust[i * n + j] = true;
                    table.use_adjust[j * n + i] = true;
                    table.a_ab[i * n + j] = radii_adjust.factor(radii[i], radii[j]);
                    table.a_ab[j * n + i] = radii_adjust.factor(radii[j], radii[i]);
                }
            }
        }
        table
    }

    /// Number of centers the table was built for.
    pub fn num_centers(&self) -> usize {
        self.num_centers
    }
}

/// One block of grid points. `distances` and `products` are laid out as
/// `[center * block + point]` and must hold at least `num_centers * block` elements.
#[allow(clippy::too_many_arguments)]
fn block_kernel<F: Fn(f64) -> f64>(
    table: &BeckePairTable,
    center_index: usize,
    centers: &[(f64, f64, f64)],
    grid: &[(f64, f64, f64)],
    block: usize,
    out: &mut [f64],
    distances: &mut [f64],
    products: &mut [f64],
    switch: F,
) {
    let n = table.num_centers;
    let nb = out.len();
    debug_assert!(nb <= block);
    debug_assert!(distances.len() >= n * block && products.len() >= n * block);
    debug_assert_eq!(centers.len(), n);
    debug_assert_eq!(grid.len(), nb);

    // one distance pass per grid point, shared by every pair that involves that point
    for k in 0..n {
        let c = centers[k];
        let row = &mut distances[k * block..k * block + nb];
        for (p, g) in grid.iter().enumerate() {
            let dx = g.0 - c.0;
            let dy = g.1 - c.1;
            let dz = g.2 - c.2;
            row[p] = (dx * dx + dy * dy + dz * dz).sqrt();
        }
        products[k * block..k * block + nb].fill(1.0);
    }

    for i in 0..n {
        for j in 0..i {
            let idx = i * n + j;
            let a_ab = table.a_ab[idx];
            let use_adjust = table.use_adjust[idx];
            let dist_ab = table.dist_ab[idx];

            // disjoint scratch rows, so the inner loop runs without bounds checks
            let (dist_lo, dist_hi) = distances.split_at(i * block);
            let dist_j = &dist_lo[j * block..j * block + nb];
            let dist_i = &dist_hi[..nb];
            let (prod_lo, prod_hi) = products.split_at_mut(i * block);
            let prod_j = &mut prod_lo[j * block..j * block + nb];
            let prod_i = &mut prod_hi[..nb];

            for p in 0..nb {
                // JCP 88, 2547 (1988), eq. 11
                let mu_ab = (dist_i[p] - dist_j[p]) / dist_ab;

                let mut nu_ab = mu_ab;
                if use_adjust {
                    nu_ab += a_ab * (1.0 - mu_ab * mu_ab);
                }

                let f = switch(nu_ab);

                // The saturation handling of partitioning_weight, written branchless:
                // keep == 0 reproduces "pa[ia] = 0.0" together with an untouched pa[ib],
                // keep == 1 is the plain update. x * 1.0 and x + 0.0 are exact for the
                // non-negative products accumulated here, so both branches stay bit-identical.
                let keep = ((1.0 - f).abs() > parameters::SMALL) as u8 as f64;
                prod_i[p] *= 0.5 * (1.0 - f) * keep;
                prod_j[p] *= 0.5 * (1.0 + f) * keep + (1.0 - keep);
            }
        }
    }

    for p in 0..nb {
        let mut w = 0.0f64;
        for k in 0..n {
            w += products[k * block + p];
        }
        out[p] = if w.abs() > parameters::SMALL {
            products[center_index * block + p] / w
        } else {
            1.0
        };
    }
}

/// Dispatch on the screening hardness, so the common iteration counts are unrolled while
/// any other value keeps the generic loop.
#[allow(clippy::too_many_arguments)]
fn kernel_with_hardness(
    table: &BeckePairTable,
    center_index: usize,
    centers: &[(f64, f64, f64)],
    grid: &[(f64, f64, f64)],
    block: usize,
    out: &mut [f64],
    distances: &mut [f64],
    products: &mut [f64],
    hardness: usize,
) {
    match hardness {
        1 => block_kernel(table, center_index, centers, grid, block, out, distances, products, f3_const::<1>),
        2 => block_kernel(table, center_index, centers, grid, block, out, distances, products, f3_const::<2>),
        3 => block_kernel(table, center_index, centers, grid, block, out, distances, products, f3_const::<3>),
        other => block_kernel(table, center_index, centers, grid, block, out, distances, products, move |x: f64| f3(x, other)),
    }
}

/// Partition weights of every point of `grid`, evaluated in one serial pass.
///
/// Bit-identical to calling `partitioning_weight` once per grid point. Useful for testing
/// and for callers that are already inside a parallel region.
pub fn partitioning_weights_block(
    table: &BeckePairTable,
    center_index: usize,
    centers: &[(f64, f64, f64)],
    grid: &[(f64, f64, f64)],
    hardness: usize,
    out: &mut [f64],
) {
    assert_eq!(grid.len(), out.len(), "partitioning_weights_block: grid and output length differ");
    if grid.is_empty() {
        return;
    }
    let n = table.num_centers;
    let block = grid.len();
    let mut distances = vec![0.0f64; n * block];
    let mut products = vec![1.0f64; n * block];
    kernel_with_hardness(table, center_index, centers, grid, block, out, &mut distances, &mut products, hardness);
}

/// Partition weights of every point of `grid`, split into blocks and evaluated in parallel
/// by rayon. Bit-identical to `partitioning_weights_block` and therefore to
/// `partitioning_weight`, independently of the thread count and of the block length.
///
/// Do not call this from inside another rayon parallel region; the nested region would
/// oversubscribe the pool. The current call site, the serial per-atom loop of `atom_grid`,
/// runs on the main thread.
pub fn partitioning_weights_par(
    table: &BeckePairTable,
    center_index: usize,
    centers: &[(f64, f64, f64)],
    grid: &[(f64, f64, f64)],
    hardness: usize,
    out: &mut [f64],
) {
    assert_eq!(grid.len(), out.len(), "partitioning_weights_par: grid and output length differ");
    if grid.is_empty() {
        return;
    }
    let n = table.num_centers;
    let block = block_length(n);
    out.par_chunks_mut(block)
        .enumerate()
        .for_each_init(
            // one scratch pair per worker, reused for every block that worker takes
            || (vec![0.0f64; n * block], vec![1.0f64; n * block]),
            |(distances, products), (block_index, chunk)| {
                let start = block_index * block;
                let grid_block = &grid[start..start + chunk.len()];
                kernel_with_hardness(
                    table, center_index, centers, grid_block, block, chunk, distances, products, hardness,
                );
            },
        );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random molecule with a few elements repeated, so both the
    /// "radii differ" and the "radii equal" branches of the adjustment are exercised.
    fn toy_system(n: usize, seed: u64) -> (Vec<(f64, f64, f64)>, Vec<i32>) {
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let pool = [1i32, 6, 8, 16];
        let mut centers = Vec::with_capacity(n);
        let mut charges = Vec::with_capacity(n);
        for _ in 0..n {
            centers.push((next() * 6.0 - 3.0, next() * 6.0 - 3.0, next() * 6.0 - 3.0));
            charges.push(pool[(next() * 4.0) as usize % pool.len()]);
        }
        (centers, charges)
    }

    /// Grid points that include near-collinear placements, which is where the saturation
    /// branch of the reference implementation fires.
    fn toy_grid(npoints: usize, seed: u64) -> Vec<(f64, f64, f64)> {
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut points = Vec::with_capacity(npoints);
        for k in 0..npoints {
            let theta = 1e-3 * (k as f64 % 7.0);
            let length = 1.0 + 8.0 * next();
            points.push((-length * theta.cos(), length * theta.sin(), 0.0));
        }
        points
    }

    #[test]
    fn blocked_and_parallel_match_scalar_reference_bitwise() {
        for &n in &[2usize, 5, 17] {
            let (centers, charges) = toy_system(n, 0x9E37_79B9_7F4A_7C15 ^ n as u64);
            let points = toy_grid(600, 0x2545_F491_4F6C_DD1D ^ n as u64);
            for &scheme in &[RadiiAdjust::Becke, RadiiAdjust::Treutler] {
                for &hardness in &[1usize, 2, 3, 5] {
                    let table = BeckePairTable::new(&centers, &charges, scheme);
                    assert_eq!(table.num_centers(), n);
                    for center_index in 0..n {
                        let mut blocked = vec![0.0f64; points.len()];
                        partitioning_weights_block(&table, center_index, &centers, &points, hardness, &mut blocked);
                        let mut parallel = vec![0.0f64; points.len()];
                        partitioning_weights_par(&table, center_index, &centers, &points, hardness, &mut parallel);
                        for (p, point) in points.iter().enumerate() {
                            let reference =
                                partitioning_weight(center_index, &centers, &charges, *point, hardness, scheme);
                            assert_eq!(
                                blocked[p].to_bits(), reference.to_bits(),
                                "blocked path differs (n={n}, center={center_index}, point={p}, hardness={hardness})"
                            );
                            assert_eq!(
                                parallel[p].to_bits(), reference.to_bits(),
                                "parallel path differs (n={n}, center={center_index}, point={p}, hardness={hardness})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn kernel_is_invariant_to_block_length() {
        let n = 7usize;
        let (centers, charges) = toy_system(n, 0xDEAD_BEEF_1234_5678);
        let points = toy_grid(777, 0x0123_4567_89AB_CDEF);
        let table = BeckePairTable::new(&centers, &charges, RadiiAdjust::Treutler);
        let mut reference = vec![0.0f64; points.len()];
        partitioning_weights_block(&table, 0, &centers, &points, 3, &mut reference);
        for &block in &[1usize, 3, 64, 128, 777] {
            let mut out = vec![0.0f64; points.len()];
            let mut distances = vec![0.0f64; n * block];
            let mut products = vec![1.0f64; n * block];
            for (index, chunk) in out.chunks_mut(block).enumerate() {
                let start = index * block;
                kernel_with_hardness(
                    &table, 0, &centers, &points[start..start + chunk.len()], block, chunk,
                    &mut distances, &mut products, 3,
                );
            }
            for (p, (a, b)) in out.iter().zip(reference.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "block={block} differs at point {p}");
            }
        }
    }


    /// The saturation branch of `partitioning_weight` is the only place where the blocked
    /// kernel could diverge from it. Force it: two atoms on the x axis, points beyond the
    /// second one and almost exactly collinear, then check both the bit-identity and that
    /// the branch really fired.
    #[test]
    fn saturation_branch_is_exercised_and_matches() {
        let centers = vec![(2.8f64, 0.0, 0.0), (0.0, 0.0, 0.0)];
        let charges = vec![8i32, 1];
        let scheme = RadiiAdjust::Treutler;
        let table = BeckePairTable::new(&centers, &charges, scheme);
        let length = 5.0f64;
        let points: Vec<(f64, f64, f64)> = (0..64)
            .map(|k| {
                let theta = 1e-4 * k as f64;
                (-length * theta.cos(), length * theta.sin(), 0.0)
            })
            .collect();
        let mut blocked = vec![0.0f64; points.len()];
        partitioning_weights_block(&table, 0, &centers, &points, 3, &mut blocked);
        let a_ab = scheme.factor(bragg::get_bragg_angstrom(charges[0]), bragg::get_bragg_angstrom(charges[1]));
        let dist_ab = distance(&centers[0], &centers[1]);
        let mut fired = 0usize;
        for (p, point) in points.iter().enumerate() {
            let reference = partitioning_weight(0, &centers, &charges, *point, 3, scheme);
            assert_eq!(blocked[p].to_bits(), reference.to_bits(), "saturated point {p} differs");
            let mu_ab = (distance(point, &centers[0]) - distance(point, &centers[1])) / dist_ab;
            let nu_ab = mu_ab + a_ab * (1.0 - mu_ab * mu_ab);
            if (1.0 - f3(nu_ab, 3)).abs() <= parameters::SMALL {
                fired += 1;
            }
        }
        assert!(fired > 0, "the saturation branch was never exercised");
    }

    #[test]
    fn empty_grid_is_accepted() {
        let (centers, charges) = toy_system(3, 42);
        let table = BeckePairTable::new(&centers, &charges, RadiiAdjust::Becke);
        let points: Vec<(f64, f64, f64)> = Vec::new();
        let mut out: Vec<f64> = Vec::new();
        partitioning_weights_block(&table, 0, &centers, &points, 3, &mut out);
        partitioning_weights_par(&table, 0, &centers, &points, 3, &mut out);
        assert!(out.is_empty());
    }
}


