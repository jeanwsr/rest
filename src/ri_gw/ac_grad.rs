// ============================================================================
// Analytic derivatives of the Padé (analytic-continuation) self-energy.
//
// The energy path (`ri_gw::scgw::single_orbital_gw_ac`) builds the
// analytic-continuation self-energy from `n` samples of `Sigma_c` on the
// imaginary axis,
//
//     y_s = Sigma_c(i lambda_s)          (s = 0..n-1),
//
// at the shifted nodes `z_s = ef + i lambda_s`, turns them into a continued
// fraction (Thiele's reciprocal-difference scheme, exactly as
// `ri_gw::ac::PadeApproximant::from_imaginary_axis`) and evaluates the
// retarded value at `omega + i eta`.
//
// The nuclear gradient needs
//
//     d Sigma / dR = Re[ sum_s (dP/dy_s) dy_s/dR + (dP/def) def/dR ],
//
// i.e. the derivative of the approximant with respect to every sample `y_s`
// and with respect to the Fermi shift `ef` (which moves **all** nodes,
// `d z_s / d ef = 1`).  It also needs
//
//     d Sigma / d omega  = Re[dP/dz],
//     d^2 Sigma / d omega^2 = Re[d^2P/dz^2],
//     d^2 Sigma / d omega dR  (through d^2P/dz dy_s, d^2P/dz def),
//
// the last one for the linearized (one-step Newton) QP branch.
//
// This module provides all of those in one forward sweep.  Every arithmetic
// quantity carries
//
//     ( value, d/dz, d^2/dz^2, d/dp, d^2/dz dp )
//
// where `p` runs over the `n` sample values followed by `ef`.  The continued
// fraction is evaluated with that algebra, so the derivatives are exact
// (machine precision) and the same branches as the production energy code are
// followed.
// ============================================================================

use num_complex::Complex64;

/// Breakdown threshold of the continued-fraction construction, identical to
/// [`crate::ri_gw::ac::PadeApproximant`]'s internal tolerance.
const BREAKDOWN_TOLERANCE: f64 = 1.0e-14;

/// Forward-mode dual number carrying the derivatives the Padé gradient needs.
#[derive(Clone, Debug)]
struct Dual {
    /// value
    v: Complex64,
    /// `d/dz`
    d1: Complex64,
    /// `d^2/dz^2`
    d2: Complex64,
    /// `d/dp` for every parameter `p`
    g: Vec<Complex64>,
    /// `d^2/dz dp`
    g1: Vec<Complex64>,
}

impl Dual {
    fn constant(v: Complex64, np: usize) -> Self {
        Self {
            v,
            d1: Complex64::new(0.0, 0.0),
            d2: Complex64::new(0.0, 0.0),
            g: vec![Complex64::new(0.0, 0.0); np],
            g1: vec![Complex64::new(0.0, 0.0); np],
        }
    }

    fn parameter(v: Complex64, index: usize, np: usize) -> Self {
        let mut out = Self::constant(v, np);
        out.g[index] = Complex64::new(1.0, 0.0);
        out
    }

    fn variable_z(v: Complex64, np: usize) -> Self {
        let mut out = Self::constant(v, np);
        out.d1 = Complex64::new(1.0, 0.0);
        out
    }

    fn add(&self, other: &Self) -> Self {
        let np = self.g.len();
        let mut g = Vec::with_capacity(np);
        let mut g1 = Vec::with_capacity(np);
        for i in 0..np {
            g.push(self.g[i] + other.g[i]);
            g1.push(self.g1[i] + other.g1[i]);
        }
        Self {
            v: self.v + other.v,
            d1: self.d1 + other.d1,
            d2: self.d2 + other.d2,
            g,
            g1,
        }
    }

    fn sub(&self, other: &Self) -> Self {
        let np = self.g.len();
        let mut g = Vec::with_capacity(np);
        let mut g1 = Vec::with_capacity(np);
        for i in 0..np {
            g.push(self.g[i] - other.g[i]);
            g1.push(self.g1[i] - other.g1[i]);
        }
        Self {
            v: self.v - other.v,
            d1: self.d1 - other.d1,
            d2: self.d2 - other.d2,
            g,
            g1,
        }
    }

    fn mul(&self, other: &Self) -> Self {
        let np = self.g.len();
        let (a, b) = (self, other);
        // (a0 + a1 e + a2 e^2/2)(b0 + b1 e + b2 e^2/2), e = dz
        let v = a.v * b.v;
        let d1 = a.d1 * b.v + a.v * b.d1;
        let d2 = a.d2 * b.v + 2.0 * a.d1 * b.d1 + a.v * b.d2;
        let mut g = Vec::with_capacity(np);
        let mut g1 = Vec::with_capacity(np);
        for i in 0..np {
            g.push(a.g[i] * b.v + a.v * b.g[i]);
            g1.push(a.g1[i] * b.v + a.g[i] * b.d1 + a.d1 * b.g[i] + a.v * b.g1[i]);
        }
        Self { v, d1, d2, g, g1 }
    }

    fn inv(&self) -> Option<Self> {
        let av = self.v;
        if av.norm() <= BREAKDOWN_TOLERANCE {
            return None;
        }
        let av2 = av * av;
        let inv = Complex64::new(1.0, 0.0) / av;
        let inv2 = Complex64::new(1.0, 0.0) / av2;
        let inv3 = inv2 / av;
        let np = self.g.len();
        let v = inv;
        let d1 = -self.d1 * inv2;
        let d2 = 2.0 * self.d1 * self.d1 * inv3 - self.d2 * inv2;
        let mut g = Vec::with_capacity(np);
        let mut g1 = Vec::with_capacity(np);
        for i in 0..np {
            g.push(-self.g[i] * inv2);
            g1.push(-self.g1[i] * inv2 + 2.0 * self.g[i] * self.d1 * inv3);
        }
        Some(Self { v, d1, d2, g, g1 })
    }

    fn div(&self, other: &Self) -> Option<Self> {
        Some(self.mul(&other.inv()?))
    }
}

/// Derivatives of the Padé approximant at one evaluation point.
#[derive(Clone, Debug)]
pub struct PadeDerivatives {
    /// `P(z)`
    pub value: Complex64,
    /// `dP/dz`
    pub dvalue_dz: Complex64,
    /// `d^2P/dz^2`
    pub d2value_dz2: Complex64,
    /// `dP/dp` with parameters `[y_0 .. y_{n-1}, ef]`
    pub dvalue_dparam: Vec<Complex64>,
    /// `d^2P/dz dp` with the same parameter layout
    pub d2value_dz_dparam: Vec<Complex64>,
}

/// Build the Thiele continued fraction of `values` on the shifted nodes
/// `ef + i lambda_s` and evaluate it, together with all derivatives needed by
/// the analytic gradient.
///
/// This mirrors [`crate::ri_gw::ac::PadeApproximant::from_imaginary_axis`] +
/// [`crate::ri_gw::ac::PadeApproximant::evaluate`] branch for branch, so the
/// value agrees with the energy path to machine precision.
///
/// Returns `None` when the continued fraction breaks down, exactly where the
/// energy path would return `AcError::ContinuedFractionBreakdown`.
pub fn pade_derivatives(
    lambdas: &[f64],
    ef: f64,
    values: &[Complex64],
    z: Complex64,
) -> Option<PadeDerivatives> {
    let n = values.len();
    if n == 0 || lambdas.len() != n || !z.re.is_finite() || !z.im.is_finite() {
        return None;
    }
    let np = n + 1;
    let zero = Complex64::new(0.0, 0.0);
    let ef_index = n;

    // The production code returns a constant approximant when every sample
    // coincides (this also covers the single-sample case).
    if n == 1
        || values
            .iter()
            .all(|v| (*v - values[0]).norm() <= BREAKDOWN_TOLERANCE)
    {
        let mut g = vec![zero; np];
        // `coefficients[0] = values[0]`: the approximant is independent of `ef`.
        g[0] = Complex64::new(1.0, 0.0);
        return Some(PadeDerivatives {
            value: values[0],
            dvalue_dz: zero,
            d2value_dz2: zero,
            dvalue_dparam: g,
            d2value_dz_dparam: vec![zero; np],
        });
    }

    let val: Vec<Dual> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| Dual::parameter(v, i, np))
        .collect();
    let ef_dual = Dual::parameter(Complex64::new(ef, 0.0), ef_index, np);
    let node: Vec<Dual> = lambdas
        .iter()
        .map(|&lambda| {
            let mut nd = ef_dual.clone();
            nd.v += Complex64::new(0.0, lambda);
            nd
        })
        .collect();

    let mut coefficients: Vec<Dual> = Vec::with_capacity(n);
    coefficients.push(val[0].clone());
    for i in 1..n {
        let mut reciprocal = val[i].sub(&coefficients[0]);
        for j in 1..i {
            if reciprocal.v.norm() <= BREAKDOWN_TOLERANCE {
                return None;
            }
            let num = node[i].sub(&node[j - 1]);
            reciprocal = num.div(&reciprocal)?.sub(&coefficients[j]);
        }
        if reciprocal.v.norm() <= BREAKDOWN_TOLERANCE {
            return None;
        }
        let num = node[i].sub(&node[i - 1]);
        coefficients.push(num.div(&reciprocal)?);
    }

    let z_dual = Dual::variable_z(z, np);
    let last = coefficients.len() - 1;
    let mut result = coefficients[last].clone();
    for i in (0..last).rev() {
        if result.v.norm() <= BREAKDOWN_TOLERANCE {
            return None;
        }
        let num = z_dual.sub(&node[i]);
        result = coefficients[i].add(&num.div(&result)?);
    }

    Some(PadeDerivatives {
        value: result.v,
        dvalue_dz: result.d1,
        d2value_dz2: result.d2,
        dvalue_dparam: result.g,
        d2value_dz_dparam: result.g1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rational(z: Complex64) -> Complex64 {
        // (3 + z) / (2 + 3z + z^2): a rational function with two poles.
        (Complex64::new(3.0, 0.0) + z)
            / (Complex64::new(2.0, 0.0)
                + z * Complex64::new(3.0, 0.0)
                + z * z)
    }

    fn samples(lambdas: &[f64], ef: f64) -> Vec<Complex64> {
        lambdas
            .iter()
            .map(|&l| rational(Complex64::new(ef, l)))
            .collect()
    }

    #[test]
    fn value_matches_production_approximant() {
        let lambdas = [0.1, 0.3, 0.7, 1.2, 2.0];
        let ef = 0.0;
        let values = samples(&lambdas, ef);
        let samples_ac: Vec<crate::ri_gw::ac::ImaginaryAxisSample> = lambdas
            .iter()
            .zip(values.iter())
            .map(|(&l, &v)| crate::ri_gw::ac::ImaginaryAxisSample::new_shifted(l, ef, v))
            .collect();
        let pade = crate::ri_gw::ac::PadeApproximant::from_imaginary_axis(&samples_ac).unwrap();

        for &(re, im) in &[(0.5, 0.002), (-0.7, 0.01), (2.0, 0.05)] {
            let z = Complex64::new(re, im);
            let ours = pade_derivatives(&lambdas, ef, &values, z).unwrap();
            let prod = pade.evaluate(z).unwrap();
            assert!(
                (ours.value - prod).norm() < 1.0e-13,
                "value mismatch at {z}: ours {} prod {}",
                ours.value,
                prod
            );
        }
    }

    #[test]
    fn derivatives_match_finite_differences() {
        let lambdas = [0.2, 0.5, 1.1, 2.3, 4.0];
        let ef = 0.05;
        let values = samples(&lambdas, ef);
        let z = Complex64::new(0.4, 0.01);
        let d = pade_derivatives(&lambdas, ef, &values, z).unwrap();

        // d/dz: central difference of the same forward value.
        let h = 1.0e-6;
        let fp = pade_derivatives(&lambdas, ef, &values, z + h).unwrap().value;
        let fm = pade_derivatives(&lambdas, ef, &values, z - h).unwrap().value;
        let dz_fd = (fp - fm) / (2.0 * h);
        assert!(
            (d.dvalue_dz - dz_fd).norm() < 1.0e-7,
            "d/dz {} vs {}",
            d.dvalue_dz,
            dz_fd
        );

        let fpp = pade_derivatives(&lambdas, ef, &values, z + h).unwrap().dvalue_dz;
        let fmm = pade_derivatives(&lambdas, ef, &values, z - h).unwrap().dvalue_dz;
        let d2z_fd = (fpp - fmm) / (2.0 * h);
        assert!(
            (d.d2value_dz2 - d2z_fd).norm() < 1.0e-6,
            "d2/dz2 {} vs {}",
            d.d2value_dz2,
            d2z_fd
        );

        // d/dy_s: perturb one sample.
        for s in 0..values.len() {
            let mut vp = values.clone();
            let mut vm = values.clone();
            vp[s] += Complex64::new(h, 0.0);
            vm[s] -= Complex64::new(h, 0.0);
            let fd = (pade_derivatives(&lambdas, ef, &vp, z).unwrap().value
                - pade_derivatives(&lambdas, ef, &vm, z).unwrap().value)
                / (2.0 * h);
            assert!(
                (d.dvalue_dparam[s] - fd).norm() < 1.0e-6,
                "d/dy_{s} {} vs {}",
                d.dvalue_dparam[s],
                fd
            );
            // mixed d2/dz dy_s
            let fdp = pade_derivatives(&lambdas, ef, &vp, z).unwrap().dvalue_dz;
            let fdm = pade_derivatives(&lambdas, ef, &vm, z).unwrap().dvalue_dz;
            let mix = (fdp - fdm) / (2.0 * h);
            assert!(
                (d.d2value_dz_dparam[s] - mix).norm() < 1.0e-6,
                "d2/dz dy_{s} {} vs {}",
                d.d2value_dz_dparam[s],
                mix
            );
        }

        // d/def: moves every node.
        let fd_ef = (pade_derivatives(&lambdas, ef + h, &values, z).unwrap().value
            - pade_derivatives(&lambdas, ef - h, &values, z).unwrap().value)
            / (2.0 * h);
        assert!(
            (d.dvalue_dparam[values.len()] - fd_ef).norm() < 1.0e-6,
            "d/def {} vs {}",
            d.dvalue_dparam[values.len()],
            fd_ef
        );
    }
}
