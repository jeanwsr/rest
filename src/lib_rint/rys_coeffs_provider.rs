use crate::lib_rint::rys_recursion::{CoeffProvider, RecurrenceCoeffs};

/// B10, B01', B00, C00, C00'
pub struct RysCoeffProvider;

impl CoeffProvider for RysCoeffProvider {
    #[allow(clippy::too_many_arguments)]
    fn coeffs_for(
        &self,
        t: &f64,
        p_center: &f64,
        q_center: &f64,
        ax: &f64,
        _bx: &f64,
        cx: &f64,
        _dx: &f64,
        alpha: &f64,
        beta: &f64,
        gamma: &f64,
        delta: &f64,
        _rho: &f64,
    ) -> RecurrenceCoeffs {
        // A, B
        let a = alpha + beta;
        let b = gamma + delta;

        // Gaussian product centers
        let xa = p_center;
        let xb = q_center;

        let t2 = t;

        // (41) C00
        let c00 = (xa - ax) + (b / (a + b)) * (xb - xa) * t2;

        // (45) C00'
        let c00p = (xb - cx) + (a / (a + b)) * (xa - xb) * t2;

        // (42) B00
        let b00 = t2 / (2.0 * (a + b));

        // (43) B10
        let b10 = 1.0 / (2.0 * a) - (b / (2.0 * a * (a + b))) * t2;

        // (46) B01'
        let b01p = 1.0 / (2.0 * b) - (a / (2.0 * b * (a + b))) * t2;

        RecurrenceCoeffs {
            b10,
            b01p,
            b00,
            c00,
            c00p,
        }
    }
}
