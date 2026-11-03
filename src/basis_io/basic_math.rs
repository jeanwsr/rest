//! Basic math utilities (factorials, inverse trig, misc) shared by GTO evaluation.
pub fn pi() -> f64 {
    return std::f64::consts::PI;
}

pub fn rad2deg(x:f64) -> f64 {
    return x*180.00/pi();
}

pub fn arctan(x: f64) -> f64 {
    let y = x;
    let sqrt_three = (3.0_f64).sqrt();
    if y < 0.0 {
        return -1.0 * arctan(-1.0 * y);
    } else
    /* if positive */
    {
        if y < 1.0 {
            if y <= 0.267949 {
                return y - (y*y*y / 3.0) + (y*y*y*y*y / 5.0);
            } else {
                return rad2deg(
                    (pi() / 6.0)
                        + arctan(( sqrt_three * y) - 1.0)
                            / arctan(sqrt_three + y),
                        //+ arctan(((3.0_f64).sqrt() * y) - 1.0)
                        //    / arctan((3.0_f64).sqrt() + y),
                );
            }
            
        } else {
            return (pi() / 2.0) - arctan(1.0 / y);
        }
    }
}
pub fn arcsin(x: f64) -> f64 {
    return arctan(x / (1.0 - x*x).sqrt());
}
pub fn arccos(x: f64) -> f64 {
    return arctan((1.0 - x*x).sqrt() / x);
}

pub fn factorial(end:i32) -> i32 {
    if end == 0 {return 1
    } else {
        return (1..=end).fold(1,|acc,x| {acc * x})
    }
}
#[inline]
pub fn double_factorial(end:i32) -> i32 {
    // for positive number, it yields the corresponding double factorial number
    // for negtive number, it yields the reciprocal ef the double factorial number
    if end == 0 || end == 1 || end == -1 {return 1};
    if end>0 {
        let (tmp_start, tmp_end) = (1,end/2);
        if end%2 == 1 {   //odd
            return (tmp_start..=tmp_end).fold(1,|acc,x| {acc*(2*x+1)});
        } else {  // even
            return (tmp_start..=tmp_end).fold(1,|acc,x| {acc * 2* x});
        }
    } else {
        let (tmp_start, tmp_end) = (end/2,-1);
        if end%2 == -1 {   //odd
            return (tmp_start..=tmp_end).fold(1,|acc,x| {acc*(2*x+1)});
        } else {  // even
            return 0;
        }
    }
}
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
#[inline]
pub fn delta_r(r1: &[f64;3], r2: &[f64;3]) -> ([f64;3],f64) {
    let mut dr = [0.0;3];
    let mut res = 0.0;
    for i in 0..3 {
        dr[i] = r1[i] - r2[i];
        res += dr[i]*dr[i];
    }
    return (dr, res);
}

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
