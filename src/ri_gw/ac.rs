use num_complex::Complex64;
use std::error::Error;
use std::fmt;

const BREAKDOWN_TOLERANCE: f64 = 1.0e-14;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImaginaryAxisSample {
    pub omega: f64,
    pub value: Complex64,
}

impl ImaginaryAxisSample {
    pub fn new(omega: f64, value: Complex64) -> Self {
        Self { omega, value }
    }

    fn node(self) -> Complex64 {
        Complex64::new(0.0, self.omega)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcError {
    EmptySamples,
    NegativeFrequency { index: usize },
    NonFiniteSample { index: usize },
    DuplicateFrequency { first: usize, second: usize },
    ContinuedFractionBreakdown { index: usize },
    InvalidEvaluationPoint,
}

impl fmt::Display for AcError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySamples => {
                write!(formatter, "at least one imaginary-axis sample is required")
            }
            Self::NegativeFrequency { index } => {
                write!(
                    formatter,
                    "sample {index} has a negative imaginary frequency"
                )
            }
            Self::NonFiniteSample { index } => {
                write!(formatter, "sample {index} contains a non-finite number")
            }
            Self::DuplicateFrequency { first, second } => write!(
                formatter,
                "samples {first} and {second} have duplicate frequencies"
            ),
            Self::ContinuedFractionBreakdown { index } => write!(
                formatter,
                "continued-fraction construction broke down at sample {index}"
            ),
            Self::InvalidEvaluationPoint => write!(formatter, "evaluation point is not finite"),
        }
    }
}

impl Error for AcError {}

#[derive(Debug, Clone)]
pub struct PadeApproximant {
    nodes: Vec<Complex64>,
    values: Vec<Complex64>,
    coefficients: Vec<Complex64>,
}

impl PadeApproximant {
    pub fn from_imaginary_axis(samples: &[ImaginaryAxisSample]) -> Result<Self, AcError> {
        validate_samples(samples)?;

        let nodes: Vec<Complex64> = samples.iter().map(|sample| sample.node()).collect();
        let values: Vec<Complex64> = samples.iter().map(|sample| sample.value).collect();

        if values
            .iter()
            .all(|value| (*value - values[0]).norm() <= BREAKDOWN_TOLERANCE)
        {
            return Ok(Self {
                nodes: vec![nodes[0]],
                values: vec![values[0]],
                coefficients: vec![values[0]],
            });
        }

        let mut coefficients = Vec::with_capacity(samples.len());
        coefficients.push(values[0]);

        for i in 1..samples.len() {
            let mut reciprocal_difference = values[i] - coefficients[0];

            for j in 1..i {
                if reciprocal_difference.norm() <= BREAKDOWN_TOLERANCE {
                    return Err(AcError::ContinuedFractionBreakdown { index: i });
                }

                reciprocal_difference =
                    (nodes[i] - nodes[j - 1]) / reciprocal_difference - coefficients[j];
            }

            if reciprocal_difference.norm() <= BREAKDOWN_TOLERANCE {
                return Err(AcError::ContinuedFractionBreakdown { index: i });
            }

            coefficients.push((nodes[i] - nodes[i - 1]) / reciprocal_difference);
        }

        Ok(Self {
            nodes,
            values,
            coefficients,
        })
    }

    pub fn evaluate(&self, z: Complex64) -> Result<Complex64, AcError> {
        if !is_finite_complex(z) {
            return Err(AcError::InvalidEvaluationPoint);
        }

        if let Some(index) = self.nodes.iter().position(|node| *node == z) {
            return Ok(self.values[index]);
        }

        if self.coefficients.len() == 1 {
            return Ok(self.coefficients[0]);
        }

        let last = self.coefficients.len() - 1;
        let mut result = self.coefficients[last];

        for i in (0..last).rev() {
            if result.norm() <= BREAKDOWN_TOLERANCE {
                return Err(AcError::ContinuedFractionBreakdown { index: i });
            }

            result = self.coefficients[i] + (z - self.nodes[i]) / result;
        }

        Ok(result)
    }

    pub fn evaluate_retarded(&self, energy: f64, eta: f64) -> Result<Complex64, AcError> {
        if !energy.is_finite() || !eta.is_finite() || eta <= 0.0 {
            return Err(AcError::InvalidEvaluationPoint);
        }

        self.evaluate(Complex64::new(energy, eta))
    }
}

fn validate_samples(samples: &[ImaginaryAxisSample]) -> Result<(), AcError> {
    if samples.is_empty() {
        return Err(AcError::EmptySamples);
    }

    for (index, sample) in samples.iter().enumerate() {
        if sample.omega < 0.0 {
            return Err(AcError::NegativeFrequency { index });
        }

        if !sample.omega.is_finite() || !is_finite_complex(sample.value) {
            return Err(AcError::NonFiniteSample { index });
        }

        if let Some(first) = samples[..index]
            .iter()
            .position(|previous| previous.omega == sample.omega)
        {
            return Err(AcError::DuplicateFrequency {
                first,
                second: index,
            });
        }
    }

    Ok(())
}

fn is_finite_complex(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn known_rational_function(z: Complex64) -> Complex64 {
        (Complex64::new(1.0, 0.0) + 0.5 * z) / (Complex64::new(1.0, 0.0) + 0.25 * z)
    }

    #[test]
    fn reproduces_known_rational_function() {
        let samples: Vec<_> = [0.0, 1.0, 2.0]
            .iter()
            .map(|omega| {
                let z = Complex64::new(0.0, *omega);
                ImaginaryAxisSample::new(*omega, known_rational_function(z))
            })
            .collect();

        let approximant = PadeApproximant::from_imaginary_axis(&samples).unwrap();
        let z = Complex64::new(0.7, 0.1);
        let difference = approximant.evaluate(z).unwrap() - known_rational_function(z);

        assert!(difference.norm() < 1.0e-12);
    }

    #[test]
    fn exactly_reproduces_input_nodes() {
        let samples: Vec<_> = [0.0, 1.0, 2.0]
            .iter()
            .map(|omega| {
                let z = Complex64::new(0.0, *omega);
                ImaginaryAxisSample::new(*omega, known_rational_function(z))
            })
            .collect();

        let approximant = PadeApproximant::from_imaginary_axis(&samples).unwrap();

        for sample in samples {
            let value = approximant.evaluate(sample.node()).unwrap();
            assert_eq!(value, sample.value);
        }
    }

    #[test]
    fn accepts_constant_samples() {
        let value = Complex64::new(2.0, -0.5);
        let samples = [
            ImaginaryAxisSample::new(0.0, value),
            ImaginaryAxisSample::new(1.0, value),
            ImaginaryAxisSample::new(2.0, value),
        ];

        let approximant = PadeApproximant::from_imaginary_axis(&samples).unwrap();
        let continued = approximant.evaluate_retarded(1.5, 0.01).unwrap();

        assert!((continued - value).norm() < 1.0e-14);
    }

    #[test]
    fn rejects_duplicate_frequencies() {
        let samples = [
            ImaginaryAxisSample::new(1.0, Complex64::new(1.0, 0.0)),
            ImaginaryAxisSample::new(1.0, Complex64::new(2.0, 0.0)),
        ];

        assert!(matches!(
            PadeApproximant::from_imaginary_axis(&samples),
            Err(AcError::DuplicateFrequency {
                first: 0,
                second: 1
            })
        ));
    }

    #[test]
    fn rejects_invalid_retarded_broadening() {
        let samples = [ImaginaryAxisSample::new(0.0, Complex64::new(1.0, 0.0))];

        let approximant = PadeApproximant::from_imaginary_axis(&samples).unwrap();

        assert_eq!(
            approximant.evaluate_retarded(1.0, 0.0),
            Err(AcError::InvalidEvaluationPoint)
        );
    }

    #[test]
    fn reproduces_rational_function_over_larger_grid() {
        // f(z) = (3 + z) / (2 + 3z + z^2), rational (order-2 non-degenerate).
        // Sample on 7 imaginary-axis points, then evaluate at a complex test point.
        fn target(z: Complex64) -> Complex64 {
            (Complex64::new(3.0, 0.0) + z) / (Complex64::new(2.0, 0.0) + z * Complex64::new(3.0, 0.0) + z * z)
        }

        let omegas = [0.1, 0.3, 0.7, 1.2, 2.0, 3.5, 6.0];
        let samples: Vec<_> = omegas
            .iter()
            .map(|&omega| {
                let z = Complex64::new(0.0, omega);
                ImaginaryAxisSample::new(omega, target(z))
            })
            .collect();

        let approx = PadeApproximant::from_imaginary_axis(&samples).unwrap();

        // Evaluate near the real axis with a small broadening.
        let z = Complex64::new(0.5, 0.002);
        let err = (approx.evaluate(z).unwrap() - target(z)).norm();
        assert!(err < 5.0e-10, "large error {:.2e} at z={}", err, z);

        // Also test evaluate_retarded wrapper.
        let sigma = approx.evaluate_retarded(0.5, 0.002).unwrap();
        let err2 = (sigma - target(z)).norm();
        assert!(err2 < 5.0e-10, "retarded error {:.2e}", err2);
    }

    #[test]
    fn pade_interpolates_complex_test_points() {
        // Sample the known rational function at a few points.
        fn target(z: Complex64) -> Complex64 {
            (Complex64::new(1.0, 0.0) + 0.5 * z) / (Complex64::new(1.0, 0.0) + 0.25 * z)
        }

        // Use fewer samples (<= order of rational function + 1)
        // to avoid continued-fraction breakdown from over-sampling.
        let omegas = [0.5, 2.0, 5.0];
        let samples: Vec<_> = omegas
            .iter()
            .map(|&omega| {
                let z = Complex64::new(0.0, omega);
                ImaginaryAxisSample::new(omega, target(z))
            })
            .collect();

        let approx = PadeApproximant::from_imaginary_axis(&samples).unwrap();

        // Evaluate at a point near the real axis.
        let z = Complex64::new(0.8, 0.01);
        let err = (approx.evaluate(z).unwrap() - target(z)).norm();
        assert!(err < 1.0e-10, "interpolation error {:.2e}", err);

        // Retarded evaluation with positive broadening.
        let sigma = approx.evaluate_retarded(0.8, 0.01).unwrap();
        let exact_ret = target(Complex64::new(0.8, 0.01));
        let err2 = (sigma - exact_ret).norm();
        assert!(err2 < 1.0e-10, "retarded error {:.2e}", err2);
    }

    #[test]
    fn pade_single_sample_returns_constant() {
        let value = Complex64::new(1.7, -0.3);
        let samples = [ImaginaryAxisSample::new(2.0, value)];
        let approx = PadeApproximant::from_imaginary_axis(&samples).unwrap();

        let result = approx.evaluate_retarded(0.5, 0.01).unwrap();
        assert!((result - value).norm() < 1.0e-14);
    }

    #[test]
    fn rejects_zero_imaginary_samples() {
        assert!(matches!(
            PadeApproximant::from_imaginary_axis(&[]),
            Err(AcError::EmptySamples)
        ));
    }

    #[test]
    fn rejects_negative_frequency() {
        let samples = [ImaginaryAxisSample::new(-0.5, Complex64::new(1.0, 0.0))];
        assert!(matches!(
            PadeApproximant::from_imaginary_axis(&samples),
            Err(AcError::NegativeFrequency { index: 0 })
        ));
    }

    #[test]
    fn rejects_nan_imaginary_value() {
        let samples = [ImaginaryAxisSample::new(1.0, Complex64::new(f64::NAN, 0.0))];
        assert!(matches!(
            PadeApproximant::from_imaginary_axis(&samples),
            Err(AcError::NonFiniteSample { index: 0 })
        ));
    }

    #[test]
    fn rejects_inf_imaginary_value() {
        let samples = [ImaginaryAxisSample::new(1.0, Complex64::new(f64::INFINITY, 0.0))];
        assert!(matches!(
            PadeApproximant::from_imaginary_axis(&samples),
            Err(AcError::NonFiniteSample { index: 0 })
        ));
    }
}
