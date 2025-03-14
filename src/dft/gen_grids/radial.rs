//! This mod is used to generate radial grids for DFT calculation.<br>
//! Currently supported methods: 
//! * Krack-Koster radial grid[^1]
//! * Lindh-Malmqvist-Gagliardi radial grid[^2]
//! * Treutler-Ahlrichs radial grid[^3]
//! 
//! [^1]: [M. Krack, A. M. Köster. The Journal of Chemical Physics 108, 3226-3234 (1998)](https://doi.org/10.1063/1.475719).
//! 
//! [^2]: [R. Lindh, P.-Å. Malmqvist, L. Gagliardi. Theoretical Chemistry Accounts 106, 178-187 (2001)](https://dx.doi.org/10.1007/s002140100263).
//! 
//! [^3]: [O. Treutler, R. Ahlrichs. The Journal of Chemical Physics 102, 346-354 (1995)](https://doi.org/10.1063/1.469408).
//! 

use std::collections::HashMap;
use std::f64::consts::PI;

use super::bragg;
use super::bse;
use super::parameters;
use statrs::function::gamma;

/// Krack-Koster radial grid according to _M. Krack, A. M. Köster. The Journal of Chemical Physics 108, 3226-3234 (1998)_, eqs. 9-13.<br>
/// Reference can be found [here](https://doi.org/10.1063/1.475719).  
/// 
/// Arguments:<br>
/// num_points: Number of radial grids to be generated.<br>
/// 
/// Returns:<br>
/// A tuple of two vectors, radial grid coordinates and weights respectively.<br>
pub fn radial_grid_kk(num_points: usize) -> (Vec<f64>, Vec<f64>) {
    let n = num_points as i32;
    let mut rws: Vec<_> = (1..=n).map(|i| kk_r_w(i, n)).collect();
    rws.reverse();
    rws.iter().cloned().unzip()
}

/// Generate Krack-Koster radial grid $r_i$ and weight $w_i$<br>
/// _M. Krack, A. M. Köster. The Journal of Chemical Physics 108, 3226-3234 (1998)_, eqs. 9-10.<br>
/// # Math:
/// $$
/// \begin{aligned}
/// &r_i = \frac{n+1-2i}{n+1} + \frac{2}{\pi}\left[1+\frac{2}{3}\sin^2\left(\frac{i\pi}{n+1}\right)\right]
/// \cos\left(\frac{i\pi}{n+1}\right)\sin\left(\frac{i\pi}{n+1}\right)
/// \newline
/// &w_i = \frac{16}{3(n+1)}\sin^4\left(\frac{i\pi}{n+1}\right)
/// \end{aligned}
/// $$
/// 
/// 
fn kk_r_w(i: i32, n: i32) -> (f64, f64) {
    let pi = std::f64::consts::PI;

    let angle = ((i as f64) * pi) / ((n + 1) as f64);
    let s = angle.sin();
    let c = angle.cos();
    let t1 = ((n + 1 - 2 * i) as f64) / ((n + 1) as f64);
    let t2 = 2.0 / pi;
    let x = t1 + t2 * c * s * (1.0 + (2.0 / 3.0) * s * s);
    let f = 1.0 / 2.0_f64.ln();
    let r = f * (2.0 / (1.0 - x)).ln();
    let w = r * r * (f / (1.0 - x)) * (16.0 * s * s * s * s) / (3.0 * (n as f64) + 3.0);

    (r, w)
}

  
/// Download basis set information from www.basissetexchange.org to determine alpha_max and alpha_min, 
/// returning radial grids coordinates and weights.
//let bse.rs to determine max and min angular grid number(alpha max & min)
//
pub fn radial_grid_lmg_bse(
    basis_set: &str,
    radial_precision: f64,
    proton_charge: i32,
) -> (Vec<f64>, Vec<f64>) {
    let (alpha_min, alpha_max) = bse::ang_min_and_max(basis_set, proton_charge as usize);

    radial_grid_lmg(alpha_min, alpha_max, radial_precision, proton_charge)
}

/// Lindh, Malmqvist, and Gagliardi radial grid according to _R. Lindh, P.-Å. Malmqvist, L. Gagliardi.
/// Theoretical Chemistry Accounts 106, 178-187 (2001)_.<br>
/// Reference can be found [here](https://dx.doi.org/10.1007/s002140100263).
pub fn radial_grid_lmg(
    alpha_min: HashMap<usize, f64>,
    alpha_max: f64,
    radial_precision: f64,
    proton_charge: i32,
) -> (Vec<f64>, Vec<f64>) {
    // factor 2.0 to match DIRAC code
    let r_inner = get_r_inner(radial_precision, alpha_max * 2.0);

    let mut h = std::f64::MAX;
    let mut r_outer: f64 = 0.0;

    // we need alpha_min sorted by l
    // at this point not sure why ... need to check again the literature
    let mut v: Vec<_> = alpha_min.into_iter().collect();
    v.sort_by(|x, y| x.0.cmp(&y.0));

    for (l, a) in v {
        if a > 0.0 {
            r_outer = r_outer.max(get_r_outer(
                radial_precision,
                a,
                l,
                4.0 * bragg::get_bragg_angstrom(proton_charge),
            ));
            assert!(r_outer > r_inner);
            h = h.min(get_h(radial_precision, l, 0.1 * (r_outer - r_inner)));
        }
    }
    assert!(r_outer > h);

    let c = r_inner / (h.exp() - 1.0);
    let num_points = ((1.0 + (r_outer / c)).ln() / h) as usize;

    let mut rs = Vec::new();
    let mut ws = Vec::new();

    for i in 1..=num_points {
        let r = c * (((i as f64) * h).exp() - 1.0);
        rs.push(r);
        ws.push((r + c) * r * r * h);
    }

    (rs, ws)
}

/// Same as [`radial_grid_kk`].
/// 
pub fn radial_grid_gc2nd (n: usize) -> (Vec<f64>, Vec<f64>)  {
    let ln2 = 1.0 / f64::ln(2.0);
    let fac = (16.0 / 3.0) / ((n+1) as f64);
    let mut x1 = vec![];
    let mut xi = vec![];

    for i in 1..n+1 {
        x1.push((i as f64)*PI / ((n+1) as f64));
    };

    for i in 1..n+1 {
        let a = ((n as isize)-1-2*((i-1) as isize)) as f64;
        let b = (1.0+2.0/3.0*f64::sin(x1[i-1]).powf(2.0))*f64::sin(2.0*x1[i-1]) / PI;
        xi.push(
            a / ((n+1) as f64) + b
        );
    };  // is there a mistake in xi? Should be (n+1-2i) not (n-1-2i)

    let xi_rev: Vec<&f64> = xi.iter().rev().collect();
    let xi_new: Vec<f64> = xi.iter().zip(xi_rev.iter()).map(|(xi, xi_rev)| (xi - *xi_rev) / 2.0 ).collect();
    let r: Vec<f64> = xi_new.iter().map(|xi_new| 1.0 - f64::ln(1.0 + xi_new) * ln2).collect();
    let w: Vec<f64> = xi_new.iter().zip(x1.iter()).map(|(xi_new, x1)| fac * f64::sin(*x1).powf(4.0) * ln2 / (1.0 + xi_new)).collect();
    return (r,w);

}


/// Treutler-Ahlrichs radial grid according to _O. Treutler, R. Ahlrichs. The Journal of Chemical Physics 102, 346-354 (1995)_.<br>
/// Quadrature T2 with the mapping M4. <br>
/// Reference can be found [here](https://doi.org/10.1063/1.469408).  
/// 
/// Arguments:<br>
/// n: Number of radial grids to be generated.<br>
/// 
/// Returns:<br>
/// A tuple of two vectors, radial grid coordinates and weights respectively.<br>
/// # Math:
/// $$
/// \begin{aligned}
/// &r_i = \frac{\xi}{\ln2}(a+x_i)^\alpha\ln{\frac{a+1}{1-x_i}}
/// \newline
/// &w_i = \frac{\pi}{n+1}\sqrt{1-x_i^2}\frac{(a+x_i)^{\alpha}}{\ln2}
/// \left(\frac{\alpha}{a+x_i}\ln{\frac{a+1}{1-x_i}}+\frac{1}{1-x_i}\right)
/// \newline
/// &x_i = \cos{\frac{i\pi}{n+1}}
/// \newline\newline
/// &\text{where } \xi=1.0\text{, }\alpha=0.6\text{, }a=1.
/// \end{aligned}

/// $$
pub fn radial_grid_treutler(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut r: Vec<f64> = vec![];
    let mut w: Vec<f64> = vec![];
    let step = PI / ((n+1) as f64);
    let ln2 = 1.0 / f64::ln(2.0);
    for i in 1..n+1 {
        let x = f64::cos((i as f64)*step);
        r.push(-ln2*(1.0+x).powf(0.6)*f64::ln((1.0-x)/2.0));
        w.push(step*f64::sin((i as f64)*step)*ln2*(1.0+x).powf(0.6)*(-0.6/(1.0+x)*f64::ln((1.0-x)/2.0)+1.0/(1.0-x)))
    }
    let r_rev: Vec<f64> = r.into_iter().rev().collect();
    let w_rev: Vec<f64> = w.into_iter().rev().collect();
    return (r_rev, w_rev)

}



// TCA 106, 178 (2001), eq. 25
// we evaluate r_inner for s functions
// truncation error
/// Lindh, Malmqvist, and Gagliardi radial grid according to _R. Lindh, P.-Å. Malmqvist, L. Gagliardi.
/// Theoretical Chemistry Accounts 106, 178-187 (2001)_, eq.25.<br>
/// # Math:
/// $$
/// \ln\left(\frac{1}{R_\text{H}}\right)+\frac{m+3}{2}\ln\alpha_\text{H}r_l^2=D_\text{m}
/// $$
/// 
fn get_r_inner(max_error: f64, alpha_inner: f64) -> f64 {
    let d = 1.9;

    let mut r = d - (1.0 / max_error).ln();
    r *= 2.0 / 3.0;
    r = r.exp() / alpha_inner;
    r = r.sqrt();

    r
}


// TCA 106, 178 (2001), eq. 19
// relative error
/// Lindh, Malmqvist, and Gagliardi radial grid according to _R. Lindh, P.-Å. Malmqvist, L. Gagliardi.
/// Theoretical Chemistry Accounts 106, 178-187 (2001)_, eq.19.<br>
/// # Math:
/// $$
/// R_\text{L}\approx\Gamma\left(\frac{m+3}{2}\right)(\alpha{r_{k_\text{H}}}^2)
/// ^{\frac{m+1}{2}}e^{-\alpha{r_{k_\text{H}}}^2}
/// $$
/// 
///
fn get_r_outer(max_error: f64, alpha_outer: f64, l: usize, guess: f64) -> f64 {
    let m = (2 * l) as f64;
    let mut r_old = std::f64::MAX;
    let mut step = 0.5;
    let mut sign = 1.0;
    let mut r = guess;

    while (r_old - r).abs() > parameters::SMALL {
        let c = gamma::gamma((m + 3.0) / 2.0);
        let a = (alpha_outer * r * r).powf((m + 1.0) / 2.0);
        let e = (-alpha_outer * r * r).exp();
        let f = c * a * e;

        let sign_old = sign;
        sign = if f > max_error { 1.0 } else { -1.0 };
        if r < 0.0 {
            sign = 1.0
        }
        if sign * sign_old < 0.0 {
            step *= 0.1;
        }

        r_old = r;
        r += sign * step;
    }

    r
}


// TCA 106, 178 (2001), eqs. 17 and 18
// absolute value of gamma function
/// Lindh, Malmqvist, and Gagliardi radial grid according to _R. Lindh, P.-Å. Malmqvist, L. Gagliardi.
/// Theoretical Chemistry Accounts 106, 178-187 (2001)_, eqs.17-18.<br>
/// # Math:
/// $$
/// \begin{aligned}
/// &R_\text{D}^{(0)}\approx\frac{4\sqrt{2}\pi}{h}e^{\frac{-\pi^2}{2h}}
/// \newline
/// &R_\text{D}^{(m)}\approx\frac{\Gamma(\frac{3}{2})}{\Gamma\left(\frac{m+3}{2}\right)}
/// \left(\frac{\pi}{h}\right)^{\frac{m}{2}}R_\text{D}^{(0)}
/// \end{aligned}
/// $$
/// 
///
fn get_h(max_error: f64, l: usize, guess: f64) -> f64 {
    let m = (2 * l) as f64;
    let mut h_old = std::f64::MAX;
    let mut h = guess;
    let mut step = 0.1 * guess;
    let mut sign = -1.0;

    let pi = std::f64::consts::PI;

    while (h_old - h).abs() > parameters::SMALL {
        let c0 = 4.0 * (2.0 as f64).sqrt() * pi;
        let cm = gamma::gamma(3.0 / 2.0) / gamma::gamma((m + 3.0) / 2.0);
        let p0 = 1.0 / h;
        let e0 = (-pi * pi / (2.0 * h)).exp();
        let pm = (pi / h).powf(m / 2.0);
        let rd0 = c0 * p0 * e0;
        let f = cm * pm * rd0;

        let sign_old = sign;
        sign = if f > max_error { -1.0 } else { 1.0 };
        if h < 0.0 {
            sign = 1.0
        }
        if sign * sign_old < 0.0 {
            step *= 0.1;
        }

        h_old = h;
        h += sign * step;
    }

    h
}


pub fn radial_grid_murray(){

}

pub fn radial_grid_becke(){

}

pub fn radial_grid_delley(){

}

pub fn radial_grid_mura_knowles(){
    
}

