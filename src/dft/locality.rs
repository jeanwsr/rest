//! Shared locality-screening utilities for grid-based kernels.
//!
//! Both the SCF AO tabulation (`Grids::prepare_tabulated_ao_sparse`) and the
//! analytical-gradient block pipeline (`dft::num_int::GridBlockProducer`) need
//! the same two decisions:
//!
//! 1. **Geometric pre-screen** — which atoms can contribute basis values above
//!    a cutoff anywhere inside a block of grid points
//!    ([`atom_influence_radii`] + [`block_bbox`] + [`atom_reaches_block`]).
//! 2. **Row screen** — which basis rows of an evaluated atom table survive a
//!    value cutoff ([`screen_rows_by_value`]).
//!
//! Keeping them in one place pins the two paths to identical screening
//! semantics. The radius envelope below is a rigorous upper bound of the
//! tabulated GTO magnitude in both basis conventions (spheric and cartesian;
//! derivation in AO_AOP_NORM_AUDIT.md), so a geometric skip never drops a row
//! that the value screen would have kept (the SCF tabulation therefore stays
//! bit-identical when the pre-screen is added).

use rest_tensors::MatrixFull;

use crate::molecule_io::Molecule;

/// Conservative per-atom influence radii (bohr) for AO screening: beyond
/// `radius(a)` no basis function on atom `a` can exceed `cutoff` in magnitude.
///
/// The envelope of one shell is `c_dir * C_contr * r^l * exp(-alpha_min r^2)`:
///
/// - `C_contr = sum_p |d_p|`, where `d_p` are the shell coefficients as stored
///   after `basis_normalization` (they already carry libcint's `gto_norm` and
///   the contraction renormalization, and multiply the tabulated polynomial
///   directly). No extra `Norm0(alpha_p, l)` factor may be applied: for
///   diffuse exponents `Norm0 < 1` and the bound would no longer dominate the
///   tabulated values (that was the pre-fix bug, strongest for cartesian
///   shells with l >= 2 whose tabulated rows are raw polynomials).
/// - `c_dir` bounds the direction amplitude of one tabulated row and depends
///   on the tabulation convention (derivation: AO_AOP_NORM_AUDIT.md §4):
///   * spheric — the tabulated value is `d_p * r^l * C_lm(r_hat) * exp(..)`
///     (the `sqrt(I_L) * norm0 * f * c2s` chain collapses elementwise to the
///     normalized real solid harmonic), so
///     `c_dir = sup |C_lm| <= sqrt((2l+1)/(4 pi))`;
///   * cartesian — the tabulated value is
///     `d_p * [fac_sp(l) if l<2 else 1] * x^lx y^ly z^lz * exp(..)`, and
///     `|x^lx y^ly z^lz| <= r^l`, so `c_dir = fac_sp(l)` for l < 2 and 1 for
///     l >= 2 (`fac_sp` matches libcint's `CINTcommon_fac_sp`:
///     `fac_sp(0) = 1/(2 sqrt(pi))`, `fac_sp(1) = sqrt(3/(4 pi))`).
///
/// Both factors are validated by the unit tests below (direction sup and
/// shell envelope properties) and the bound holds for every exponent, tight
/// or diffuse.
///
/// A 0.5 bohr margin is added on top. Derivatives amplify the value by a
/// modest polynomial factor at these distances; consumers that screen
/// derivative-driven contractions should pass a cutoff tightened accordingly
/// (the gradient pipeline uses `ao_cutoff * 1e-3`).
pub fn atom_influence_radii(mol: &Molecule, cutoff: f64) -> Vec<f64> {
    let cart_basis = mol.ctrl.basis_type == "cartesian";
    mol.basis4elem
        .iter()
        .map(|elem| {
            let mut r_max = 0.0_f64;
            for shell in &elem.electron_shells {
                let l = shell.angular_momentum.iter().map(|&x| x.max(0) as usize).max().unwrap_or(0);
                let alpha_min = shell.exponents.iter().cloned().fold(f64::INFINITY, f64::min);
                if !alpha_min.is_finite() || alpha_min <= 0.0 {
                    continue;
                }
                // libcint CINTcommon_fac_sp: applied to s and p shells only
                let fac_sp = match l {
                    0 => 0.5 / std::f64::consts::PI.sqrt(),
                    1 => (3.0 / (4.0 * std::f64::consts::PI)).sqrt(),
                    _ => 1.0,
                };
                let c_dir = if cart_basis {
                    fac_sp
                } else {
                    ((2.0 * l as f64 + 1.0) / (4.0 * std::f64::consts::PI)).sqrt()
                };
                for coeffs in &shell.coefficients {
                    let c_contr: f64 = coeffs
                        .iter()
                        .zip(shell.exponents.iter())
                        .map(|(c, _a)| c.abs())
                        .sum();
                    r_max = f64::max(r_max, envelope_radius(c_dir * c_contr, l, alpha_min, cutoff));
                }
            }
            r_max + 0.5
        })
        .collect()
}

/// Largest r (bohr) at which `C * r^l * exp(-alpha r^2) >= cutoff`, solved by
/// bracketing the tail root and bisecting; 0 when the envelope never reaches
/// the cutoff (a function that is everywhere negligible).
pub fn envelope_radius(c: f64, l: usize, alpha: f64, cutoff: f64) -> f64 {
    if c <= 0.0 {
        return 0.0;
    }
    let f = |r: f64| c * r.powi(l as i32) * (-alpha * r * r).exp();
    let r_peak = (l as f64 / (2.0 * alpha)).sqrt();
    if f(r_peak) < cutoff {
        return 0.0;
    }
    let mut lo = r_peak.max(1.0e-8);
    let mut hi = (r_peak * 2.0).max(1.0);
    while f(hi) >= cutoff {
        hi *= 2.0;
        if hi > 1.0e4 {
            return hi;
        }
    }
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if f(mid) >= cutoff {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Axis-aligned bounding box of a block of grid points, `[lo, hi]` per axis.
pub fn block_bbox(coords: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for c in coords {
        for k in 0..3 {
            lo[k] = lo[k].min(c[k]);
            hi[k] = hi[k].max(c[k]);
        }
    }
    (lo, hi)
}

/// Whether an atom at `center` with influence `radius` can reach any point of
/// the block bounding box (center-to-box distance not exceeding the radius).
pub fn atom_reaches_block(center: &[f64; 3], radius: f64, lo: &[f64; 3], hi: &[f64; 3]) -> bool {
    let mut d2 = 0.0_f64;
    for k in 0..3 {
        let d = (lo[k] - center[k]).max(0.0) + (center[k] - hi[k]).max(0.0);
        d2 += d * d;
    }
    d2 <= radius * radius
}

/// Row screen of a per-atom AO table `[nbas, num_grids]` (column-major): keep
/// the indices of rows whose absolute value exceeds `cutoff` at some grid
/// point. This is the same predicate the SCF sparse AO tabulation applies.
pub fn screen_rows_by_value(vals: &MatrixFull<f64>, num_grids: usize, cutoff: f64) -> Vec<usize> {
    let nbas = vals.size[0];
    (0..nbas)
        .filter(|&r| (0..num_grids).any(|g| vals[[r, g]].abs() > cutoff))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis_io::{gto_value_serial, BasCell, Basis4Elem};
    use rest_libcint::CintType;

    /// `c_dir` used by `atom_influence_radii`, replicated here so the tests
    /// pin the same constants.
    fn c_dir_for(l: usize, cart_basis: bool) -> f64 {
        let fac_sp = match l {
            0 => 0.5 / std::f64::consts::PI.sqrt(),
            1 => (3.0 / (4.0 * std::f64::consts::PI)).sqrt(),
            _ => 1.0,
        };
        if cart_basis {
            fac_sp
        } else {
            ((2.0 * l as f64 + 1.0) / (4.0 * std::f64::consts::PI)).sqrt()
        }
    }

    /// Deterministic quasi-uniform directions (Fibonacci sphere).
    fn fibonacci_directions(n: usize) -> Vec<[f64; 3]> {
        let ga = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        (0..n)
            .map(|i| {
                let y = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
                let r = (1.0 - y * y).max(0.0).sqrt();
                let theta = ga * i as f64;
                [r * theta.cos(), y, r * theta.sin()]
            })
            .collect()
    }

    fn build_shell(l: i32, cint_type: &CintType) -> Basis4Elem {
        // deliberately span tight to very diffuse exponents: the envelope
        // must hold for every alpha, including the regime where the old
        // norm0-based bound failed (norm0 < 1)
        let mut cell = BasCell {
            function_type: None,
            region: None,
            angular_momentum: vec![l],
            exponents: vec![50.0, 5.0, 0.5, 0.05],
            coefficients: vec![
                vec![0.15, 0.45, 0.40, 0.20],
                vec![0.05, 0.30, 0.50, 0.35],
            ],
            native_coefficients: vec![],
        };
        cell.basis_normalization(cint_type);
        Basis4Elem {
            electron_shells: vec![cell],
            references: None,
            ecp_potentials: None,
            ecp_electrons: None,
            global_index: (0, 0),
        }
    }

    /// Property: for every tabulated row (both conventions), the value at any
    /// point is bounded by `c_dir * (sum_p |d_p|) * r^l * exp(-alpha_min r^2)`,
    /// which is exactly the envelope `atom_influence_radii` solves for.
    #[test]
    fn envelope_bounds_tabulated_values_both_conventions() {
        let center = [0.0_f64; 3];
        let radii = [0.3, 1.0, 2.0, 4.0, 7.0, 10.0];
        let directions = fibonacci_directions(64);
        for l in 0..=4_i32 {
            for (cart, type_name, cint_type) in [
                (false, "spheric", CintType::Spheric),
                (true, "cartesian", CintType::Cartesian),
            ] {
                let bas = build_shell(l, &cint_type);
                let alpha_min = bas.electron_shells[0]
                    .exponents
                    .iter()
                    .cloned()
                    .fold(f64::INFINITY, f64::min);
                let lf = l as f64;
                let c_dir = c_dir_for(l as usize, cart);
                let mut points: Vec<[f64; 3]> = Vec::new();
                for &r in radii.iter() {
                    for d in directions.iter() {
                        points.push([r * d[0], r * d[1], r * d[2]]);
                    }
                }
                let basis_type = type_name.to_string();
                let tab = gto_value_serial(&points, &center, &bas, &basis_type);
                // envelope per contraction; rows of contraction x use coeffs[x]
                let contrs: Vec<f64> = bas.electron_shells[0]
                    .coefficients
                    .iter()
                    .map(|co| co.iter().map(|c| c.abs()).sum())
                    .collect();
                let row_per_contr = tab.size[0] / contrs.len();
                for g in 0..points.len() {
                    let r = radii[g / directions.len()];
                    let env_r = |c: f64| c_dir * c * r.powf(lf) * (-alpha_min * r * r).exp();
                    for i in 0..tab.size[0] {
                        let env = env_r(contrs[i / row_per_contr]);
                        assert!(
                            tab[[i, g]].abs() <= env * (1.0 + 1.0e-12),
                            "{} l={} row={} g={}: |{}| > env {}",
                            type_name, l, i, g, tab[[i, g]], env
                        );
                    }
                }
            }
        }
    }

    /// The spheric direction amplitude is the normalized real solid harmonic:
    /// its sup must not exceed sqrt((2l+1)/(4 pi)) (and nearly attains it at
    /// the pole of the m=0 row, so the bound is not vacuous). A single
    /// primitive is used so the radial/contraction factor factors out exactly.
    #[test]
    fn spheric_direction_sup_matches_normalized_solid_harmonic() {
        let center = [0.0_f64; 3];
        for l in 0..=4_i32 {
            let mut cell = BasCell {
                function_type: None,
                region: None,
                angular_momentum: vec![l],
                exponents: vec![0.7],
                coefficients: vec![vec![1.0]],
                native_coefficients: vec![],
            };
            cell.basis_normalization(&CintType::Spheric);
            let bas = Basis4Elem {
                electron_shells: vec![cell],
                references: None,
                ecp_potentials: None,
                ecp_electrons: None,
                global_index: (0, 0),
            };
            let alpha = 0.7_f64;
            let d: f64 = bas.electron_shells[0].coefficients[0][0].abs();
            let r = 2.0_f64;
            let radial = d * r.powf(l as f64) * (-alpha * r * r).exp();
            let directions = fibonacci_directions(256);
            let points: Vec<[f64; 3]> = directions
                .iter()
                .map(|dd| [r * dd[0], r * dd[1], r * dd[2]])
                .collect();
            let basis_type = "spheric".to_string();
            let tab = gto_value_serial(&points, &center, &bas, &basis_type);
            let mut sup = 0.0_f64;
            for g in 0..points.len() {
                for i in 0..tab.size[0] {
                    sup = sup.max(tab[[i, g]].abs() / radial);
                }
            }
            let bound = ((2.0 * l as f64 + 1.0) / (4.0 * std::f64::consts::PI)).sqrt();
            assert!(sup <= bound * (1.0 + 1.0e-9), "l={}: sup {} > bound {}", l, sup, bound);
            // the m=0 pole nearly attains the bound
            assert!(sup > 0.9 * bound, "l={}: bound too loose: sup {} vs {}", l, sup, bound);
        }
    }
}
