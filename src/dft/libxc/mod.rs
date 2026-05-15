use std::collections::HashMap;

use ::libxc::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LibXCFamily {
    LDA,
    GGA,
    MGGA,
    HybridGGA,
    HybridMGGA,
    Unknown,
}

impl LibXCFamily {
    fn from_libxc(family: ::libxc::enums::LibXCFamily) -> Self {
        match family {
            ::libxc::enums::LibXCFamily::LDA | ::libxc::enums::LibXCFamily::HybLDA => LibXCFamily::LDA,
            ::libxc::enums::LibXCFamily::GGA => LibXCFamily::GGA,
            ::libxc::enums::LibXCFamily::MGGA => LibXCFamily::MGGA,
            ::libxc::enums::LibXCFamily::HybGGA => LibXCFamily::HybridGGA,
            ::libxc::enums::LibXCFamily::HybMGGA => LibXCFamily::HybridMGGA,
            _ => LibXCFamily::Unknown,
        }
    }
}

#[derive(Debug)]
pub struct XcFuncType {
    inner: LibXCFunctional,
    pub xc_func_family: LibXCFamily,
}

impl XcFuncType {
    pub fn xc_func_init(func_id: usize, spin_channel: usize) -> XcFuncType {
        let spin = if spin_channel == 1 {
            LibXCSpin::Unpolarized
        } else {
            LibXCSpin::Polarized
        };
        let inner = LibXCFunctional::from_number(func_id as i32, spin);
        let xc_func_family = LibXCFamily::from_libxc(inner.family());
        XcFuncType {
            inner,
            xc_func_family,
        }
    }

    pub fn xc_func_end(&mut self) {
        // No-op: Drop handles cleanup automatically
    }

    pub fn xc_version(&self) {
        let (major, minor, micro) = ::libxc::util::libxc_version();
        println!("Libxc version: {}.{}.{}", major, minor, micro);
    }

    pub fn is_rsh(&self) -> bool {
        self.inner.is_hyb_cam()
    }

    pub fn xc_hyb_cam_coef(&self) -> (f64, f64, f64) {
        self.inner.cam_coef().unwrap_or((0.0, 0.0, 0.0))
    }

    pub fn xc_hyb_exx_coeff(&self) -> f64 {
        self.inner.hyb_exx_coef().unwrap_or(0.0)
    }

    pub fn is_nlc(&self) -> bool {
        self.inner.flags().contains(LibXCFlags::VV10)
    }

    pub fn use_laplacian(&self) -> bool {
        self.inner.needs_laplacian()
    }

    pub fn use_density_gradient(&self) -> bool {
        !matches!(self.xc_func_family, LibXCFamily::LDA)
    }

    pub fn use_kinetic_density(&self) -> bool {
        matches!(
            self.xc_func_family,
            LibXCFamily::MGGA | LibXCFamily::HybridMGGA
        )
    }

    pub fn use_exact_exchange(&self) -> bool {
        matches!(
            self.xc_func_family,
            LibXCFamily::HybridGGA | LibXCFamily::HybridMGGA
        )
    }

    pub fn is_lda(&self) -> bool {
        matches!(self.xc_func_family, LibXCFamily::LDA)
    }

    pub fn is_gga(&self) -> bool {
        matches!(self.xc_func_family, LibXCFamily::GGA)
    }

    pub fn is_mgga(&self) -> bool {
        matches!(self.xc_func_family, LibXCFamily::MGGA)
    }

    pub fn is_hybrid_gga(&self) -> bool {
        matches!(self.xc_func_family, LibXCFamily::HybridGGA)
    }

    pub fn is_hybrid_mgga(&self) -> bool {
        matches!(self.xc_func_family, LibXCFamily::HybridMGGA)
    }

    pub fn get_family_name(&self) -> String {
        match self.xc_func_family {
            LibXCFamily::LDA => "LDA".to_string(),
            LibXCFamily::GGA => "GGA".to_string(),
            LibXCFamily::MGGA => "MGGA".to_string(),
            LibXCFamily::HybridGGA => "HybridGGA".to_string(),
            LibXCFamily::HybridMGGA => "HybridMGGA".to_string(),
            LibXCFamily::Unknown => "Unknown DFA".to_string(),
        }
    }

    pub fn get_libxc_family(&self) -> LibXCFamily {
        self.xc_func_family.clone() // LibXCFamily derives Clone
    }

    pub fn get_libxc_references(&self) -> Vec<String> {
        self.inner
            .references()
            .iter()
            .map(|r| r.ref_text.clone())
            .collect()
    }

    pub fn xc_func_info_printout(&self) {
        println!("{}", self.inner.describe());
    }

    pub fn xc_code_fdqc(name: &str) -> (usize, usize, usize) {
        let lower_name = name.to_lowercase();
        if lower_name.eq("hf") {
            (0, 0, 0)
        } else if lower_name.eq("svwn") {
            (0, 1, 7)
        } else if lower_name.eq("svwn-rpa") {
            (0, 1, 8)
        } else if lower_name.eq("pz-lda") {
            (0, 1, 9)
        } else if lower_name.eq("pw-lda") {
            (0, 1, 12)
        } else if lower_name.eq("blyp") {
            (0, 106, 131)
        } else if lower_name.eq("xlyp") {
            (166, 0, 0)
        } else if lower_name.eq("pbe") {
            (0, 101, 130)
        } else if lower_name.eq("xpbe") {
            (0, 123, 136)
        } else if lower_name.eq("scan") {
            (0, 263, 267)
        } else if lower_name.eq("revscan") {
            (0, 581, 582)
        } else if lower_name.eq("r2scan") {
            (0, 497, 498)
        } else if lower_name.eq("tpss") {
            (0, 202, 231)
        } else if lower_name.eq("b3lyp") {
            (402, 0, 0)
        } else if lower_name.eq("x3lyp") {
            (411, 0, 0)
        } else if lower_name.eq("pbe0") {
            (406, 0, 0)
        } else if lower_name.eq("scan0") {
            (0, 264, 267)
        } else if lower_name.eq("tpssh") {
            (457, 0, 0)
        } else if lower_name.eq("lda_x_slater") {
            (0, 1, 0)
        } else {
            for (name, value) in LIBXC_FUNC_MAP.iter() {
                if name.starts_with("XC_") && format!("xc_{}", lower_name) == name.to_lowercase() {
                    if name.contains("_XC_") {
                        return (*value as usize, 0, 0);
                    } else if name.contains("_C_") {
                        return (0, 0, *value as usize);
                    } else if name.contains("_X_") {
                        return (0, *value as usize, 0);
                    }
                }
            }
            (0, 0, 0)
        }
    }

    pub fn code_to_name(code: usize) -> String {
        ::libxc::util::libxc_functional_get_name(code as i32).unwrap_or_else(|| "Unknown_XC".to_string())
    }

    // Simple LDA/GGA/MGGA compute wrappers
    pub fn lda_exc(&self, rho: &[f64]) -> Vec<f64> {
        let np = rho.len() / (self.inner.spin() as usize);
        let mut input = HashMap::new();
        input.insert("rho".to_string(), rho);
        let (buf, layout) = self.inner.compute_lda(&input, 0).unwrap();
        buf[layout.get("zk").unwrap()].to_vec()
    }

    pub fn gga_exc(&self, rho: &[f64], sigma: &[f64]) -> Vec<f64> {
        let np = rho.len() / (self.inner.spin() as usize);
        let mut input = HashMap::new();
        input.insert("rho".to_string(), rho);
        input.insert("sigma".to_string(), sigma);
        let (buf, layout) = self.inner.compute_gga(&input, 0).unwrap();
        buf[layout.get("zk").unwrap()].to_vec()
    }

    pub fn mgga_exc(&self, rho: &[f64], sigma: &[f64], _lapl: &[f64], tau: &[f64]) -> Vec<f64> {
        let mut input = HashMap::new();
        input.insert("rho".to_string(), rho);
        input.insert("sigma".to_string(), sigma);
        if self.inner.needs_laplacian() {
            input.insert("lapl".to_string(), _lapl);
        }
        input.insert("tau".to_string(), tau);
        let (buf, layout) = self.inner.compute_mgga(&input, 0).unwrap();
        buf[layout.get("zk").unwrap()].to_vec()
    }

    pub fn lda_exc_vxc(&self, rho: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut input = HashMap::new();
        input.insert("rho".to_string(), rho);
        let (buf, layout) = self.inner.compute_lda(&input, 1).unwrap();
        let exc = buf[layout.get("zk").unwrap()].to_vec();
        let vrho = buf[layout.get("vrho").unwrap()].to_vec();
        (exc, vrho)
    }

    pub fn gga_exc_vxc(&self, rho: &[f64], sigma: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut input = HashMap::new();
        input.insert("rho".to_string(), rho);
        input.insert("sigma".to_string(), sigma);
        let (buf, layout) = self.inner.compute_gga(&input, 1).unwrap();
        let exc = buf[layout.get("zk").unwrap()].to_vec();
        let vrho = buf[layout.get("vrho").unwrap()].to_vec();
        let vsigma = buf[layout.get("vsigma").unwrap()].to_vec();
        (exc, vrho, vsigma)
    }

    pub fn mgga_exc_vxc(
        &self,
        rho: &[f64],
        sigma: &[f64],
        _lapl: &[f64],
        tau: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut input = HashMap::new();
        input.insert("rho".to_string(), rho);
        input.insert("sigma".to_string(), sigma);
        if self.inner.needs_laplacian() {
            input.insert("lapl".to_string(), _lapl);
        }
        input.insert("tau".to_string(), tau);
        let (buf, layout) = self.inner.compute_mgga(&input, 1).unwrap();
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

    pub fn get_family_name_std(family_id: u32) -> String {
        match family_id {
            1 => "LDA".to_string(),
            2 => "GGA".to_string(),
            4 => "MGGA".to_string(),
            32 => "Hybrid GGA".to_string(),
            64 => "Hybrid MGGA".to_string(),
            _ => "Unknown".to_string(),
        }
    }

    pub fn get_libxc_spin(&self) -> usize {
        self.inner.spin() as usize
    }
}

pub fn eval_libxc_func_new(
    xc_func: &XcFuncType,
    spin: usize,
    deriv: usize,
    np: usize,
    rho: &[f64],
    sigma: Option<&[f64]>,
    lapl: Option<&[f64]>,
    tau: Option<&[f64]>,
    exc: &mut [f64],
) {
    assert!(deriv <= 3, "Derivative order must be 0, 1, 2 or 3");

    let spin_enum = if spin == 0 {
        LibXCSpin::Unpolarized
    } else {
        LibXCSpin::Polarized
    };

    // Build input map
    let mut input: HashMap<String, &[f64]> = HashMap::new();
    input.insert("rho".to_string(), rho);
    if let Some(s) = sigma {
        input.insert("sigma".to_string(), s);
    }
    if let Some(l) = lapl {
        if xc_func.inner.needs_laplacian() {
            input.insert("lapl".to_string(), l);
        }
    }
    if let Some(t) = tau {
        input.insert("tau".to_string(), t);
    }

    // Use compute_xc_with_unsliced_output to write directly into `exc`
    xc_func
        .inner
        .compute_xc_with_unsliced_output(&input, exc, deriv)
        .unwrap();
}

pub fn eval_libxc_func(
    xc_func: &XcFuncType,
    spin: usize,
    deriv: usize,
    np: usize,
    rho: &[f64],
    sigma: Option<&[f64]>,
    lapl: Option<&[f64]>,
    tau: Option<&[f64]>,
    exc: &mut [f64],
) {
    eval_libxc_func_new(xc_func, spin, deriv, np, rho, sigma, lapl, tau, exc)
}

/// Name-to-value map for libxc functional constants, compatible with old `names_and_values::MAP`.
pub mod names_and_values {
    use super::*;

    lazy_static::lazy_static! {
        pub static ref MAP: Vec<(String, usize)> = {
            LIBXC_FUNC_MAP
                .iter()
                .map(|(name, value)| (format!("XC_{}", name), *value as usize))
                .collect()
        };
    }
}
