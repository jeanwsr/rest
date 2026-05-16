use libxc::prelude::*;
use std::collections::HashMap;

pub fn xc_code_fdqc(name: &str) -> [usize; 3] {
    let lower_name = name.to_lowercase();
    if lower_name == "hf" {
        [0, 0, 0]
    } else if lower_name == "svwn" {
        [0, 1, 7]
    } else if lower_name == "svwn-rpa" {
        [0, 1, 8]
    } else if lower_name == "pz-lda" {
        [0, 1, 9]
    } else if lower_name == "pw-lda" {
        [0, 1, 12]
    } else if lower_name == "blyp" {
        [0, 106, 131]
    } else if lower_name == "xlyp" {
        [166, 0, 0]
    } else if lower_name == "pbe" {
        [0, 101, 130]
    } else if lower_name == "xpbe" {
        [0, 123, 136]
    } else if lower_name == "scan" {
        [0, 263, 267]
    } else if lower_name == "revscan" {
        [0, 581, 582]
    } else if lower_name == "m06-l" {
        [0, 203, 233]
    } else if lower_name == "mn15-l" {
        [0, 260, 261]
    } else if lower_name == "r2scan" {
        [0, 497, 498]
    } else if lower_name == "tpss" {
        [0, 202, 231]
    } else if lower_name == "b3lyp" {
        [402, 0, 0]
    } else if lower_name == "x3lyp" {
        [411, 0, 0]
    } else if lower_name == "pbe0" {
        [406, 0, 0]
    } else if lower_name == "scan0" {
        [0, 264, 267]
    } else if lower_name == "tpssh" {
        [457, 0, 0]
    } else if lower_name == "m05-2x" || lower_name == "m052x" {
        [0, 439, 238]
    } else if lower_name == "m05" {
        [0, 438, 237]
    } else if lower_name == "m06" {
        [0, 449, 235]
    } else if lower_name == "m06-2x" || lower_name == "m062x" {
        [0, 450, 236]
    } else if lower_name == "mn15" {
        [0, 268, 269]
    } else if lower_name == "wb97x" {
        [464, 0, 0]
    } else if lower_name == "cam-b3lyp" || lower_name == "camb3lyp" {
        [433, 0, 0]
    } else if lower_name == "lc-blyp" || lower_name == "lcblyp" {
        [400, 0, 0]
    } else if lower_name == "lc-wpbe" || lower_name == "lcwpbe" {
        [478, 0, 0]
    } else if lower_name == "hse06" || lower_name == "hse" {
        [428, 0, 0]
    } else if lower_name == "hse03" {
        [427, 0, 0]
    } else if lower_name == "lda_x_slater" {
        [0, 1, 0]
    } else {
        for (name, value) in LIBXC_FUNC_MAP.iter() {
            let prefixed = format!("XC_{}", name);
            if prefixed.starts_with("XC_")
                && format!("xc_{}", lower_name) == prefixed.to_lowercase()
            {
                if name.contains("_XC_") {
                    return [*value as usize, 0, 0];
                } else if name.contains("_C_") {
                    return [0, 0, *value as usize];
                } else if name.contains("_X_") {
                    return [0, *value as usize, 0];
                }
            }
        }
        panic!(
            "Unknown XC method is specified: {}. You can try using `xc_parser = \"parse_xc\"` and see if works in the ctrl.in input configuration.",
            &name
        );
    }
}

pub fn xc_code_to_name(code: usize) -> String {
    libxc::util::libxc_functional_get_name(code as i32).unwrap_or_else(|| "Unknown_XC".to_string())
}

pub fn xc_func_init(func_id: usize, spin_channel: usize) -> LibXCFunctional {
    let spin = if spin_channel == 1 {
        LibXCSpin::Unpolarized
    } else {
        LibXCSpin::Polarized
    };
    LibXCFunctional::from_number(func_id as i32, spin)
}

pub fn lda_exc_vxc(func: &LibXCFunctional, rho: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut input = HashMap::new();
    input.insert("rho".to_string(), rho);
    let (buf, layout) = func.compute_lda(&input, 1).unwrap();
    let exc = buf[layout.get("zk").unwrap()].to_vec();
    let vrho = buf[layout.get("vrho").unwrap()].to_vec();
    (exc, vrho)
}

pub fn gga_exc_vxc(func: &LibXCFunctional, rho: &[f64], sigma: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut input = HashMap::new();
    input.insert("rho".to_string(), rho);
    input.insert("sigma".to_string(), sigma);
    let (buf, layout) = func.compute_gga(&input, 1).unwrap();
    let exc = buf[layout.get("zk").unwrap()].to_vec();
    let vrho = buf[layout.get("vrho").unwrap()].to_vec();
    let vsigma = buf[layout.get("vsigma").unwrap()].to_vec();
    (exc, vrho, vsigma)
}

pub fn mgga_exc_vxc(
    func: &LibXCFunctional,
    rho: &[f64],
    sigma: &[f64],
    lapl: &[f64],
    tau: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut input = HashMap::new();
    input.insert("rho".to_string(), rho);
    input.insert("sigma".to_string(), sigma);
    if func.needs_laplacian() {
        input.insert("lapl".to_string(), lapl);
    }
    input.insert("tau".to_string(), tau);
    let (buf, layout) = func.compute_mgga(&input, 1).unwrap();
    let exc = buf[layout.get("zk").unwrap()].to_vec();
    let vrho = buf[layout.get("vrho").unwrap()].to_vec();
    let vsigma = buf[layout.get("vsigma").unwrap()].to_vec();
    let vlapl = layout
        .get("vlapl")
        .map(|r| buf[r].to_vec())
        .unwrap_or_default();
    let vtau = buf[layout.get("vtau").unwrap()].to_vec();
    (exc, vrho, vsigma, vlapl, vtau)
}

pub fn lda_exc(func: &LibXCFunctional, rho: &[f64]) -> Vec<f64> {
    let mut input = HashMap::new();
    input.insert("rho".to_string(), rho);
    let (buf, layout) = func.compute_lda(&input, 0).unwrap();
    buf[layout.get("zk").unwrap()].to_vec()
}

pub fn gga_exc(func: &LibXCFunctional, rho: &[f64], sigma: &[f64]) -> Vec<f64> {
    let mut input = HashMap::new();
    input.insert("rho".to_string(), rho);
    input.insert("sigma".to_string(), sigma);
    let (buf, layout) = func.compute_gga(&input, 0).unwrap();
    buf[layout.get("zk").unwrap()].to_vec()
}

pub fn mgga_exc(func: &LibXCFunctional, rho: &[f64], sigma: &[f64], lapl: &[f64], tau: &[f64]) -> Vec<f64> {
    let mut input = HashMap::new();
    input.insert("rho".to_string(), rho);
    input.insert("sigma".to_string(), sigma);
    if func.needs_laplacian() {
        input.insert("lapl".to_string(), lapl);
    }
    input.insert("tau".to_string(), tau);
    let (buf, layout) = func.compute_mgga(&input, 0).unwrap();
    buf[layout.get("zk").unwrap()].to_vec()
}

pub fn eval_libxc_func_new(
    xc_func: &LibXCFunctional,
    spin: usize,
    deriv: usize,
    np: usize,
    rho: &[f64],
    sigma: Option<&[f64]>,
    lapl: Option<&[f64]>,
    tau: Option<&[f64]>,
    outbuf: &mut [f64],
) {
    assert!(deriv <= 3, "Derivative order must be 0, 1, 2 or 3");
    let _ = (spin, np);

    let mut input: HashMap<String, &[f64]> = HashMap::new();
    input.insert("rho".to_string(), rho);
    if let Some(s) = sigma {
        input.insert("sigma".to_string(), s);
    }
    if let Some(l) = lapl {
        if xc_func.needs_laplacian() {
            input.insert("lapl".to_string(), l);
        }
    }
    if let Some(t) = tau {
        if xc_func.needs_tau() {
            input.insert("tau".to_string(), t);
        }
    }

    xc_func
        .compute_xc_with_unsliced_output(&input, outbuf, deriv)
        .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xc_code_fdqc_pbe() {
        let code = xc_code_fdqc("pbe");
        assert_eq!(code, [0, 101, 130]);
    }

    #[test]
    fn test_xc_code_fdqc_b3lyp() {
        let code = xc_code_fdqc("b3lyp");
        assert_eq!(code, [402, 0, 0]);
    }

    #[test]
    fn test_is_hyb_cam() {
        let cam = xc_func_init(433, 1);
        assert!(cam.is_hyb_cam());
        let hyb = xc_func_init(402, 1);
        assert!(!hyb.is_hyb_cam());
    }

    #[test]
    fn test_needs_laplacian() {
        let func = xc_func_init(263, 1);
        assert!(!func.needs_laplacian());
    }

    #[test]
    fn test_flags_vv10() {
        let func = xc_func_init(263, 1);
        assert!(!func.flags().contains(LibXCFlags::VV10));
    }

    #[test]
    fn test_xc_code_fdqc_libxc_lookup() {
        let code = xc_code_fdqc("gga_x_pbe");
        assert_eq!(code[1], 101);
    }
}
