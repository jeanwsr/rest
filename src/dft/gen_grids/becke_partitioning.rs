//! Becke partitioning according to _A. D. Becke. The Journal of Chemical Physics 88, 2547-2553 (1988)_.
//! Reference can be found [here](https://doi.org/10.1063/1.454033).
//! 
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
