//! Scalar math helpers shared across modules: overflow-safe exact (double-)
//! factorials (u128 / log-space), the exact low-L double-factorial lookup table,
//! and hot-path power micro-optimizations.

/// Factorial n! as u128 (sufficient for small n used in normalization).
/// For larger n you should switch to log-factorial to avoid overflow.
pub fn factorial_u128(n: usize) -> u128 {
    if n <= 1 {
        return 1;
    }
    let mut result: u128 = 1;
    for i in 2..=n {
        result = result.saturating_mul(i as u128);
    }
    result
}

/// Legacy-name factorial returning f64, u128-backed: exact through 33!
/// (the old i32 version overflowed above 12!). For n < 0 returns 1.0.
/// New code should prefer `factorial_u128`.
#[inline]
pub fn factorial(n: i32) -> f64 {
    factorial_u128(n.max(0) as usize) as f64
}

/// Legacy-name double factorial n!! returning f64, u128-backed: exact far
/// beyond the old i32 version's 21!! overflow limit. Convention difference to
/// the old i32 version: returns 1.0 for ALL n <= 0 (the old odd-negative
/// values like (-3)!! = -1 were never exercised by any caller; callers only
/// rely on (-1)!! = 1). New code should prefer `double_factorial_u128`.
#[inline]
pub fn double_factorial(n: i32) -> f64 {
    double_factorial_u128(n) as f64
}

/// Double factorial n!! as u128.
/// For n <= 0, return 1.
pub fn double_factorial_u128(n: i32) -> u128 {
    if n <= 0 {
        return 1;
    }
    let mut result: u128 = 1;
    let mut k = n;
    while k > 0 {
        result = result.saturating_mul(k as u128);
        k -= 2;
    }
    result
}

/// log((2k-1)!!) accumulated form, used for log-safe normalization.
/// Here input is n where we mean (2n-1)!! for odd double factorial.
pub fn log_double_factorial_odd(n: u32) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let mut s = 0.0_f64;
    for k in 1..=n {
        let odd = 2.0 * (k as f64) - 1.0;
        s += odd.ln();
    }
    s
}

/// Exact double-factorial values for the odd inputs used by cartesian GTO
/// normalization (l <= 7). Bit-identical by construction; do NOT replace with
/// a lgamma round-trip (introduces ~1e-13 relative drift into normalization).
#[inline]
pub fn specific_double_factorial(end:i32) -> f64 {
    match end {
       -1 => 1.0_f64,
        0 => 1.0_f64,
        1 => 1.0_f64,
        3 => 3.0_f64,
        5 => 15.0_f64,
        7 => 105.0_f64,
        9 => 945.0_f64,
       11 => 10395.0_f64,
       13 => 135135.0_f64,
        _ => panic!("Error: at present only provide the double factorial for the odd numbers from -2 to 10"),
    }
}

/// x^n with fast paths for the small/half-integer exponents dominant in GTO
/// evaluation; falls back to `powf` otherwise.
#[inline]
pub fn opowf(x: f64, n: f64) -> f64 {
    match n {
        0.0 => 1.0,
        0.5 => x.sqrt(),
        1.0 => x,
        2.0 => x*x,
        3.0 => x*x*x,
        4.0 => x*x*x*x,
        5.0 => x*x*x*x*x,
        6.0 => x*x*x*x*x*x,
        _ => x.powf(n),
    }
}
