use crate::ctrl_io::quasiparticle_methods::QuasiParticle;
use crate::ri_gw;
use crate::constants::PI;

pub fn fourier_series(
    sin_coeff: &[f64],
    cos_coeff: &[f64],
    powers: usize,
    t: f64,
    x: f64
) -> f64 {
    let mut sum = 0.0;
    for k in 1..=powers {
        let kf = k as f64;
        let arg = PI * x * kf / t;
        sum += sin_coeff[k-1] * arg.sin() + cos_coeff[k-1] * arg.cos();
    }
    sum
}
pub fn define_fourier_series(qp_ctrl:&QuasiParticle)->(usize,f64,Vec<f64>,Vec<f64>){
    let sin_coeff_path=qp_ctrl.fse_sin_coeff_path.clone();
    let cos_coeff_path=qp_ctrl.fse_cos_coeff_path.clone();
    let sin_coeff=ri_gw::read_floats(&sin_coeff_path).expect("Failure when reading from sin coefficients file!");
    let cos_coeff=ri_gw::read_floats(&cos_coeff_path).expect("Failure when reading from sin coefficients file!");
    let mut powers:usize=0;
    let t=qp_ctrl.gw_span_energy;
    if sin_coeff.len()==cos_coeff.len(){
        powers=sin_coeff.len();
    }else{
        panic!("Length of sin_coeff NOT EQUAL TO cos_coeff! Exited, make sure they are the same");
    }
    (powers,t,sin_coeff,cos_coeff)
}
pub fn sigma_fourier(origin: f64, omega: f64, powers: usize, t: f64, sin_coeff: &[f64], cos_coeff: &[f64]) -> f64 {
    fourier_series(sin_coeff, cos_coeff, powers, t, omega - origin)
}

/// Compute Hermite polynomial H_n(x) using recurrence relation
pub fn hermite_polynomial(n: usize, x: f64) -> f64 {
    match n {
        0 => 1.0,
        1 => 2.0 * x,
        _ => {
            let mut h_n_minus_2 = 1.0; // H_0
            let mut h_n_minus_1 = 2.0 * x; // H_1
            for k in 2..=n {
                let h_n = 2.0 * x * h_n_minus_1 - 2.0 * (k - 1) as f64 * h_n_minus_2;
                h_n_minus_2 = h_n_minus_1;
                h_n_minus_1 = h_n;
            }
            h_n_minus_1
        }
    }
}

/// Compute Hermite function series: sum_{n=0}^{N-1} coeff[n] * H_n(x) * exp(-x^2/2)
pub fn hermite_series(coeff: &[f64], x: f64) -> f64 {
    let mut sum = 0.0;
    for (n, &c) in coeff.iter().enumerate() {
        sum += c * hermite_polynomial(n, x);
    }
    sum * (-x * x / 2.0).exp()
}

/// Read Hermite coefficients from file and return them as Vec<f64>
pub fn define_hermite_series_from_file(coeff_path: &str) -> Vec<f64> {
    ri_gw::read_floats(coeff_path).expect("Failure when reading from Hermite coefficients file!")
}

/// Define Hermite series from QuasiParticle control parameters
pub fn define_hermite_series(qp_ctrl: &QuasiParticle) -> Vec<f64> {
    let coeff_path = qp_ctrl.hermite_coeff_path.clone();
    ri_gw::read_floats(&coeff_path).expect("Failure when reading from Hermite coefficients file!")
}

/// Compute self-energy using Hermite series expansion
pub fn sigma_hermite(origin: f64, omega: f64, coeff: &[f64]) -> f64 {
    hermite_series(coeff, omega - origin)
}