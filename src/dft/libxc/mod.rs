#![allow(unused)]
pub mod ffi_xc;
pub mod names_and_values;

use std::convert::TryInto;
use std::os::raw::c_int;
use std::os::raw::c_double;
use std::ffi::CStr;
use std::mem::ManuallyDrop;
use std::slice;

use tensors::MatrixFull;

//use self::ffi_xc::xc_func_info_type;
use crate::dft::libxc::ffi_xc::xc_func_info_type;

use super::Grids;

//use self::libxc::xc_func_type;
//use self::libxc::func_params_type;

//pub(crate) mod ffi_xc;

#[derive(Clone,Debug)]
pub enum LibXCFamily {
    LDA,
    GGA,
    MGGA,
    HybridGGA,
    HybridMGGA,
    Unknown
}

#[derive(Clone,Debug)]
pub struct XcFuncType {
    xc_func_type: *mut ffi_xc::xc_func_type,
    pub xc_func_info_type: *const ffi_xc::xc_func_info_type,
    pub xc_func_family: LibXCFamily,
    spin_channel: usize,
}
//pub struct XcFuncInfoType {
//    xc_func_info_type: Option<*const ffi_xc::xc_func_info_type>,
//    x_func_info_type: Option<*const ffi_xc::xc_func_info_type>,
//    c_func_info_type: Option<*const ffi_xc::xc_func_info_type>,
//}

impl XcFuncType {

    pub fn xc_version(&self) {
        let mut vmajor:c_int = 0;
        let mut vminor:c_int = 0;
        let mut vmicro:c_int = 0;
        unsafe{ffi_xc::xc_version(&mut  vmajor, &mut vminor, &mut vmicro)};
        println!("Libxc version: {}.{}.{}", vmajor, vminor, vmicro);
    }

    pub fn xc_func_init(func_id: usize, spin_channel: usize) -> XcFuncType {
        let mut xc_func_type = unsafe{ffi_xc::xc_func_alloc()};
        let init = unsafe{ffi_xc::xc_func_init(
            xc_func_type,
            func_id as c_int, 
            spin_channel as c_int)};

        let xc_func_info_type = unsafe{ffi_xc::xc_func_get_info(xc_func_type)};

        //let xc_func_info_type = match xc_func_type[0] {
        //    Some(xc_func_type) => {Some(unsafe{ffi_xc::xc_func_get_info(xc_func_type)})},
        //    None => None
        //};
        //let x_func_info_type = match xc_func_type[1] {
        //    Some(xc_func_type) => {Some(unsafe{ffi_xc::xc_func_get_info(xc_func_type)})},
        //    None => None
        //};
        //let c_func_info_type = match xc_func_type[2] {
        //    Some(xc_func_type) => {Some(unsafe{ffi_xc::xc_func_get_info(xc_func_type)})},
        //    None => None
        //};
        //let xc_func_info_type = [xc_func_info_type,x_func_info_type,c_func_info_type];


        let xc_func_family = XcFuncType::get_family_enum(xc_func_info_type);
        //let xc_func_family = match xc_func_info_type[0] {
        //    Some(xc_func_info_type) => {Some(XcFuncType::get_family_enum(xc_func_info_type))},
        //    None => None
        //};
        //let x_func_family = match xc_func_info_type[1] {
        //    Some(xc_func_info_type) => {Some(XcFuncType::get_family_enum(xc_func_info_type))},
        //    None => None
        //};
        //let c_func_family = match xc_func_info_type[2] {
        //    Some(xc_func_info_type) => {Some(XcFuncType::get_family_enum(xc_func_info_type))},
        //    None => None
        //};
        //let xc_func_family = [xc_func_family, x_func_family,c_func_family];

        XcFuncType {
            xc_func_type,
            xc_func_info_type,
            xc_func_family,
            spin_channel 
        }
    }

    //pub fn xc_func_init_fdqc(name: &str, spin_channel: usize) -> XcFuncType {
    //    let lower_name = name.to_lowercase();
    //    let xc_code: (usize, usize,usize) = XcFuncType::xc_code_fdqc(name);
    //    XcFuncType::xc_func_init(xc_code, spin_channel)
    //}


    pub fn xc_code_fdqc(name: &str) -> (usize,usize,usize) {
        let lower_name = name.to_lowercase();
        // for a list of exchange-correlation functionals
        if lower_name.eq(&"hf".to_string()) {
            (0,0,0)
        } else if lower_name.eq(&"svwn".to_string()) {
            (0,1,7)
        } else if lower_name.eq(&"svwn-rpa".to_string()) {
            (0,1,8)
        } else if lower_name.eq(&"pz-lda".to_string()) {
            (0,1,9)
        } else if lower_name.eq(&"pw-lda".to_string()) {
            (0,1,12)
        } else if lower_name.eq(&"blyp".to_string()) {
            (0,106,131)
        } else if lower_name.eq(&"xlyp".to_string()) {
            (166,0,0)
        } else if lower_name.eq(&"pbe".to_string()) {
            (0,101,130)
        } else if lower_name.eq(&"xpbe".to_string()) {
            (0,123,136)
        } else if lower_name.eq(&"scan".to_string()) {
            (0,263,267)
        } else if lower_name.eq(&"revscan".to_string()) {
            (0,581,582)
        } else if lower_name.eq(&"r2scan".to_string()) {
            (0,497,498)
        } else if lower_name.eq(&"tpss".to_string()) {
            (0,202,231)
        } else if lower_name.eq(&"b3lyp".to_string()) {
            (402,0,0)
        } else if lower_name.eq(&"x3lyp".to_string()) {
            (411,0,0)
        } else if lower_name.eq(&"pbe0".to_string()) {
            (406,0,0)
        } else if lower_name.eq(&"scan0".to_string()) {
            (0,264,267)
        } else if lower_name.eq(&"tpssh".to_string()) {
            (457,0,0)
        }
        // for a list of exchange functionals
        else if lower_name.eq(&"lda_x_slater".to_string()) {
            (0,1,0)
        } else {
            for (name, value) in names_and_values::MAP.iter() {
                if name.starts_with("XC_") && format!("xc_{}", lower_name) == name.to_lowercase() {
                    if name.contains("_XC_") {
                        return (*value, 0, 0);
                    } else if name.contains("_C_") {
                        return (0, 0, *value);
                    } else if name.contains("_X_") {
                        return (0, *value, 0);
                    }
                }
            }
            (0,0,0)
        }
    }

    pub fn xc_func_end(&mut self) {
        unsafe{ffi_xc::xc_func_end(self.xc_func_type)}
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

    //pub fn is_dfa(&self) -> bool {
    //    self.xc_func_type.iter().fold(false, |acc,xc_func_type| {
    //        match xc_func_type {
    //            Some(_) => {acc || true},
    //            None => {acc || false},
    //        }
    //    })
    //}
    pub fn use_density_gradient(&self) -> bool {
        match self.xc_func_family {
            LibXCFamily::LDA => {false},
            _ => {true},
        }
    }
    
    pub fn use_kinetic_density(&self) -> bool {
        match self.xc_func_family {
            LibXCFamily::MGGA => {true},
            LibXCFamily::HybridMGGA => {true},
            _ => {false},
        }
    }

    pub fn use_exact_exchange(&self) -> bool {
        match self.xc_func_family {
            LibXCFamily::HybridGGA => {true},
            LibXCFamily::HybridMGGA => {true},
            _ => {false},
        }
    }

    pub fn is_lda(&self) -> bool {
        match self.xc_func_family {
            LibXCFamily::LDA => true,
            _ => false
        }
    }

    pub fn is_gga(&self) -> bool {
        match self.xc_func_family {
            LibXCFamily::GGA => true,
            _ => false
        }
    }

    pub fn is_mgga(&self) -> bool {
        match self.xc_func_family {
            LibXCFamily::MGGA => true,
            _ => false
        }
    }

    pub fn is_hybrid_gga(&self) -> bool {
        match self.xc_func_family {
            LibXCFamily::HybridGGA => true,
            _ => false
        }
    }

    pub fn is_hybrid_mgga(&self) -> bool {
        match self.xc_func_family {
            LibXCFamily::HybridMGGA => true,
            _ => false
        }
    }

    pub fn xc_hyb_exx_coeff(&self) -> f64 {
        unsafe{ffi_xc::xc_hyb_exx_coef(self.xc_func_type)}
    }

    pub fn lda_exc(&self, rho: &[f64]) -> Vec<f64> {
        let length = rho.len()/&self.spin_channel;
        //println!("debug rho length: {}",length);
        let mut exc = vec![0.0; length];
        unsafe{
            ffi_xc::xc_lda_exc(
                self.xc_func_type,
                length as u64,
                rho.as_ptr(),
                exc.as_mut_ptr());
        }
        exc
    }

    pub fn gga_exc(&self, rho: &[f64], sigma: &[f64]) -> Vec<f64> {
        let length = rho.len()/&self.spin_channel;
        let mut exc = vec![0.0; length];
        unsafe{
            ffi_xc::xc_gga_exc(
                self.xc_func_type,
                length as u64,
                rho.as_ptr(),
                sigma.as_ptr(),
                exc.as_mut_ptr(),
            );
        }
        exc
    }

    pub fn mgga_exc(&self, rho: &[f64], sigma: &[f64], lapl: &[f64], tau: &[f64]) -> Vec<f64> {
        let length = rho.len()/&self.spin_channel;
        let mut exc = vec![0.0; length];
        unsafe{
            ffi_xc::xc_mgga_exc(
                self.xc_func_type,
                length as u64,
                rho.as_ptr(),
                sigma.as_ptr(),
                lapl.as_ptr(),
                tau.as_ptr(),
                exc.as_mut_ptr(),
            );
        }
        exc
    }

    pub fn lda_exc_vxc(&self, rho: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let length = rho.len()/&self.spin_channel;
        //println!("debug rho length: {}",length);
        let mut exc = vec![0.0; length];
        let mut vrho = vec![0.0; length*&self.spin_channel];
        unsafe{
            ffi_xc::xc_lda_exc_vxc(
                self.xc_func_type,
                length as u64,
                rho.as_ptr(),
                exc.as_mut_ptr(),
                vrho.as_mut_ptr());
        }
        // println!("Debug: In lda_exc_vxc exc: {:?}", &exc);
        (exc,vrho)
    }
    
    pub fn gga_exc_vxc(&self, rho: &[f64], sigma: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let length = rho.len()/&self.spin_channel;
        let mut exc = vec![0.0; length];
        let mut vrho = vec![0.0; length*&self.spin_channel];
        let mut vsigma = if self.spin_channel == 1 {
            vec![0.0; length]
        } else {
            vec![0.0; length*3]
        };
        unsafe{
            ffi_xc::xc_gga_exc_vxc(
                self.xc_func_type,
                length as u64,
                rho.as_ptr(),
                sigma.as_ptr(),
                exc.as_mut_ptr(),
                vrho.as_mut_ptr(),
                vsigma.as_mut_ptr()
            );
        }
        // println!("Debug: In gga_exc_vxc exc: {:?}", &exc);
        (exc,vrho,vsigma)
    }

    pub fn mgga_exc_vxc(&self, rho: &[f64], sigma: &[f64], lapl: &[f64], tau: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let length = rho.len() / &self.spin_channel;
        let mut exc = vec![0.0; length];
        let mut vrho = vec![0.0; length * &self.spin_channel];
        let mut vsigma = if self.spin_channel == 1 {
            vec![0.0; length]
        } else {
            vec![0.0; length*3]
        };
        let mut vtau = vec![0.0; length * &self.spin_channel];
        let mut vlapl = vec![0.0; length * &self.spin_channel];
        unsafe{
            ffi_xc::xc_mgga_exc_vxc(
                self.xc_func_type,
                length as u64,
                rho.as_ptr(),
                sigma.as_ptr(),
                lapl.as_ptr(),
                tau.as_ptr(),
                exc.as_mut_ptr(),
                vrho.as_mut_ptr(),
                vsigma.as_mut_ptr(),
                vlapl.as_mut_ptr(),
                vtau.as_mut_ptr()
            );
        }
        (exc,vrho,vsigma,vlapl,vtau)
    }

    // xc_func_info relevant functions:
    pub fn get_family_name_std(family_id: u32) -> String {
        if family_id == ffi_xc::XC_FAMILY_LDA {
            "LDA".to_string()
        } else if family_id == ffi_xc::XC_FAMILY_GGA {
            "GGA".to_string()
        } else if family_id == ffi_xc::XC_FAMILY_MGGA {
            "MGGA".to_string()
        } else if family_id == ffi_xc::XC_FAMILY_HYB_GGA {
            "Hybrid GGA".to_string()
        } else if family_id == ffi_xc::XC_FAMILY_HYB_MGGA {
            "Hybrid MGGA".to_string()
        } else {
            "Unknown".to_string()
        }
    }

    pub fn get_family_id(xc_info: *const xc_func_info_type) -> u32 {
        unsafe {ffi_xc::xc_func_info_get_family(xc_info) as u32}
    }
    pub fn get_family_enum(xc_info: *const xc_func_info_type) -> LibXCFamily {
        let family_id = XcFuncType::get_family_id(xc_info);
        if family_id == ffi_xc::XC_FAMILY_LDA {
            LibXCFamily::LDA
        } else if family_id == ffi_xc::XC_FAMILY_GGA {
            LibXCFamily::GGA
        } else if family_id == ffi_xc::XC_FAMILY_MGGA {
            LibXCFamily::MGGA
        } else if family_id == ffi_xc::XC_FAMILY_HYB_GGA {
            LibXCFamily::HybridGGA
        } else if family_id == ffi_xc::XC_FAMILY_HYB_MGGA {
            LibXCFamily::HybridMGGA
        } else {
            LibXCFamily::Unknown
        }
    }

    pub fn get_libxc_family(&self) -> LibXCFamily {
        self.xc_func_family.clone()
    }

    pub fn printout_family_name_ref(xc_info: *const xc_func_info_type, x_or_c: &str) {
        let xc_family = {
            let tmp_i: u32 = XcFuncType::get_family_id(xc_info);
            XcFuncType::get_family_name_std(tmp_i)
        };
        let xc_name = unsafe {
            let c_buf = ffi_xc::xc_func_info_get_name(xc_info);
            let c_str = CStr::from_ptr(c_buf);
            let str_slice = c_str.to_str().unwrap();
            str_slice.to_owned()
        };
        println!("The {} functional '{}' belongs to the '{}' family and is defined in the reference(s):",x_or_c,xc_name,xc_family);
        (0..5).for_each(|i| unsafe{
            let c_ref = ffi_xc::xc_func_info_get_references(xc_info, i);
            if c_ref != std::ptr::null() {
                let x_ref = {
                    let c_buf = ffi_xc::xc_func_reference_get_ref(c_ref);
                    let c_str = CStr::from_ptr(c_buf);
                    let str_slice = c_str.to_str().unwrap_or_default();
                    str_slice.to_owned()
                };
                println!("({}): {}",i,x_ref);
            };
        });
    }
    pub fn xc_func_info_printout(&self) {
        //let x_or_c = ["exchange-correlation","exchange","correlation"];
        //self.xc_func_info_type.iter().zip(x_or_c.iter()).for_each(|(xc_func_info_type,x_or_c)| {
        //    if let Some(xc_info) = xc_func_info_type {
        //        XcFuncType::printout_family_name_ref(*xc_info, *x_or_c);
        //    }
        //});
        let functype = "density";
        XcFuncType::printout_family_name_ref(self.xc_func_info_type, functype);
    }
}


struct SliceSplitter<'a> {
    data: &'a mut [f64],
    offset: usize,
}

impl<'a> SliceSplitter<'a> {
    fn new(data: &'a mut [f64]) -> Self {
        Self { data, offset: 0 }
    }

    fn take(&mut self, len:usize) -> &'a mut [f64] {
        if len == 0 {
            return &mut [];
        }
        if self.offset + len > self.data.len() {
            panic!("SliceSplitter: Attempt to take more elements than available in the slice, buffer overflow!");
        }
        let ptr = self.data.as_mut_ptr();
        let slice = unsafe {
            slice::from_raw_parts_mut(ptr.add(self.offset), len)
        };
        self.offset += len;
        slice
    }

    fn current_offset(&self) -> usize {
        self.offset
    }
}

fn as_mut_ptr_or_null<T>(slice: &mut [T]) -> *mut T {
    if slice.is_empty() {
        std::ptr::null_mut()
    } else {
        slice.as_mut_ptr()
    }
}

pub fn eval_libxc_func_new(xc_func: &XcFuncType, spin: usize, deriv: usize, np: usize, rho: &[f64], sigma: Option<&[f64]>, lapl: Option<&[f64]>, tau: Option<&[f64]>, exc: &mut [f64]) {
    assert!(deriv <= 3, "Derivative order must be 0, 1, 2 or 3");
    let mut splitter = SliceSplitter::new(exc);
    let zk = splitter.take(np); // exc

    // unsafe {println!("Debug: In eval_libxc_func_new xc_func_type: {:?}", *xc_func.xc_func_type);}

    match xc_func.xc_func_family {
        // leave fourth order terms as null pointers, currently not supported 
        LibXCFamily::LDA => {
            let (vrho, v2rho2, v3rho3) = if spin == 1 {
                (
                    if deriv > 0 { splitter.take(np*2) } else {&mut []},
                    if deriv > 1 { splitter.take(np*3) } else {&mut []},
                    if deriv > 2 { splitter.take(np*4) } else {&mut []},
                )
            } else {
                (
                    if deriv > 0 { splitter.take(np) } else {&mut []},
                    if deriv > 1 { splitter.take(np) } else {&mut []},
                    if deriv > 2 { splitter.take(np) } else {&mut []},
                )
            };
            // println!("debug vrho length: {}", vrho.len());
            unsafe {
                ffi_xc::xc_lda(
                    xc_func.xc_func_type,
                    np as u64,
                    rho.as_ptr(),
                    zk.as_mut_ptr(),
                    as_mut_ptr_or_null(vrho),
                    as_mut_ptr_or_null(v2rho2),
                    as_mut_ptr_or_null(v3rho3),
                    std::ptr::null_mut(), // v4rho4
                );
            }
        },
        
        LibXCFamily::GGA | LibXCFamily::HybridGGA => {
            let (
                vrho, vsigma, v2rho2, v2rhosigma, v2sigma2,
                v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3
            ) = if spin == 1 {
                (
                    if deriv > 0 { splitter.take(np*2) } else {&mut []},
                    if deriv > 0 { splitter.take(np*3) } else {&mut []},
                    if deriv > 1 { splitter.take(np*3) } else {&mut []}, // v2rho2
                    if deriv > 1 { splitter.take(np*6) } else {&mut []}, // v2rhosigma
                    if deriv > 1 { splitter.take(np*6) } else {&mut []}, // v2sigma2
                    if deriv > 2 { splitter.take(np*4) } else {&mut []}, // v3rho3
                    if deriv > 2 { splitter.take(np*9) } else {&mut []}, // v3rho2sigma
                    if deriv > 2 { splitter.take(np*12) } else {&mut []}, // v3rhosigma2
                    if deriv > 2 { splitter.take(np*10) } else {&mut []}, // v3sigma3
                )
            } else {
                (
                    if deriv > 0 { splitter.take(np) } else {&mut []},
                    if deriv > 0 { splitter.take(np) } else {&mut []},
                    if deriv > 1 { splitter.take(np) } else {&mut []}, // v2rho2
                    if deriv > 1 { splitter.take(np) } else {&mut []}, // v2rhosigma
                    if deriv > 1 { splitter.take(np) } else {&mut []}, // v2sigma2
                    if deriv > 2 { splitter.take(np) } else {&mut []}, // v3rho3
                    if deriv > 2 { splitter.take(np) } else {&mut []}, // v3rho2sigma
                    if deriv > 2 { splitter.take(np) } else {&mut []}, // v3rhosigma2
                    if deriv > 2 { splitter.take(np) } else {&mut []}, // v3sigma3 
                )
            };
            unsafe {
                ffi_xc::xc_gga(
                    xc_func.xc_func_type,
                    np as u64,
                    rho.as_ptr(),
                    sigma.unwrap().as_ptr(),
                    zk.as_mut_ptr(),
                    as_mut_ptr_or_null(vrho),
                    as_mut_ptr_or_null(vsigma),
                    as_mut_ptr_or_null(v2rho2),
                    as_mut_ptr_or_null(v2rhosigma),
                    as_mut_ptr_or_null(v2sigma2),
                    as_mut_ptr_or_null(v3rho3),
                    as_mut_ptr_or_null(v3rho2sigma),
                    as_mut_ptr_or_null(v3rhosigma2),
                    as_mut_ptr_or_null(v3sigma3),
                    std::ptr::null_mut(), // v4rho4
                    std::ptr::null_mut(), // v4rho3sigma
                    std::ptr::null_mut(), // v4rho2sigma2
                    std::ptr::null_mut(), // v4rhosigma3
                    std::ptr::null_mut(), // v4sigma4
                );
            }
            // if deriv > 0 {
            //     println!("Debug: In libxc_eval_xc exc: {:?}", zk);
            //     println!("Debug: In libxc_eval_xc vrho: {:?}", vrho);
            //     println!("Debug: In libxc_eval_xc vsigma: {:?}", vsigma);
            // }
        },
        // Currently, laplacian is not provided, so we set those quantities to empty
        LibXCFamily::MGGA | LibXCFamily::HybridMGGA => {
            let (vrho, vsigma, vtau) = if spin == 1 {
                (
                    if deriv > 0 { splitter.take(np*2) } else {&mut []}, 
                    if deriv > 0 { splitter.take(np*3) } else {&mut []},
                    if deriv > 0 { splitter.take(np*2) } else {&mut []},

                )
            } else {
                (
                    if deriv > 0 { splitter.take(np) } else {&mut []},
                    if deriv > 0 { splitter.take(np) } else {&mut []},
                    if deriv > 0 { splitter.take(np) } else {&mut []},
                )
            };

            
            let (v2rho2, v2rhosigma, v2sigma2, v2rhotau, v2sigmatau, v2tau2) = if spin == 1 {
                (
                    if deriv > 1 { splitter.take(np*3) } else {&mut []}, // v2rho2
                    if deriv > 1 { splitter.take(np*6) } else {&mut []}, // v2rhosigma
                    if deriv > 1 { splitter.take(np*6) } else {&mut []}, // v2sigma2
                    if deriv > 1 { splitter.take(np*4) } else {&mut []}, // v2rhotau
                    if deriv > 1 { splitter.take(np*6) } else {&mut []}, // v2sigmatau
                    if deriv > 1 { splitter.take(np*3) } else {&mut []}, // v2tau2
                )
            } else {
                (
                    if deriv > 1 { splitter.take(np) } else {&mut []}, // v2rho2
                    if deriv > 1 { splitter.take(np) } else {&mut []}, // v2rhosigma
                    if deriv > 1 { splitter.take(np) } else {&mut []}, // v2sigma2
                    if deriv > 1 { splitter.take(np) } else {&mut []}, // v2rhotau
                    if deriv > 1 { splitter.take(np) } else {&mut []}, // v2sigmatau
                    if deriv > 1 { splitter.take(np) } else {&mut []}, // v2tau2
                )
            };
            let (
                v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3,
                v3rho2tau, v3rhosigmatau, v3rhotau2, 
                v3sigma2tau, v3sigmatau2, v3tau3
            ) = if spin == 1 {
                (
                    if deriv > 2 { splitter.take(np*4) } else {&mut []}, // v3rho3
                    if deriv > 2 { splitter.take(np*9) } else {&mut []}, // v3rho2sigma
                    if deriv > 2 { splitter.take(np*12) } else {&mut []}, // v3rhosigma2
                    if deriv > 2 { splitter.take(np*10) } else {&mut []}, // v3sigma3
                    if deriv > 2 { splitter.take(np*6) } else {&mut []}, // v3rho2tau
                    if deriv > 2 { splitter.take(np*12) } else {&mut []}, // v3rhosigmatau
                    if deriv > 2 { splitter.take(np*6) } else {&mut []}, // v3rhotau2
                    if deriv > 2 { splitter.take(np*12) } else {&mut []}, // v3sigma2tau
                    if deriv > 2 { splitter.take(np*9) } else {&mut []}, // v3sigmatau2
                    if deriv > 2 { splitter.take(np*4) } else {&mut []}, // v3tau3
                )
            } else {
                (
                    if deriv > 2 { splitter.take(np) } else {&mut []}, // v3rho3
                    if deriv > 2 { splitter.take(np) } else {&mut []}, // v3rho2sigma
                    if deriv > 2 { splitter.take(np) } else {&mut []}, // v3rhosigma2
                    if deriv > 2 { splitter.take(np) } else {&mut []}, // v3sigma3
                    if deriv > 2 { splitter.take(np) } else {&mut []}, // v3rho2tau
                    if deriv > 2 { splitter.take(np) } else {&mut []}, // v3rhosigmatau
                    if deriv > 2 { splitter.take(np) } else {&mut []}, // v3rhotau2
                    if deriv > 2 { splitter.take(np) } else {&mut []}, // v3sigma2tau
                    if deriv > 2 { splitter.take(np) } else {&mut []}, // v3sigmatau2
                    if deriv > 2 { splitter.take(np) } else {&mut []}, // v3tau3
                )
            };
            unsafe {
                ffi_xc::xc_mgga(
                    xc_func.xc_func_type,
                    np as u64,
                    rho.as_ptr(),
                    sigma.unwrap().as_ptr(),
                    std::ptr::null(), // laplacian is not provided
                    tau.unwrap().as_ptr(),
                    zk.as_mut_ptr(),
                    as_mut_ptr_or_null(vrho),
                    as_mut_ptr_or_null(vsigma),
                    std::ptr::null_mut(), // vlapl 
                    as_mut_ptr_or_null(vtau),
                    as_mut_ptr_or_null(v2rho2),
                    as_mut_ptr_or_null(v2rhosigma),
                    std::ptr::null_mut(), // v2rholapl
                    as_mut_ptr_or_null(v2rhotau),
                    as_mut_ptr_or_null(v2sigma2),
                    std::ptr::null_mut(), // v2sigmalapl
                    as_mut_ptr_or_null(v2sigmatau),
                    std::ptr::null_mut(), // v2lapl2
                    std::ptr::null_mut(), // v2lapltau 
                    as_mut_ptr_or_null(v2tau2),
                    as_mut_ptr_or_null(v3rho3),
                    as_mut_ptr_or_null(v3rho2sigma),
                    std::ptr::null_mut(), // v3rho2lapl
                    as_mut_ptr_or_null(v3rho2tau),
                    as_mut_ptr_or_null(v3rhosigma2),
                    std::ptr::null_mut(), // v3rhosigmalapl
                    as_mut_ptr_or_null(v3rhosigmatau),
                    std::ptr::null_mut(), // v3rholapl2
                    std::ptr::null_mut(), // v3rholapltau
                    as_mut_ptr_or_null(v3rhotau2),
                    as_mut_ptr_or_null(v3sigma3),
                    std::ptr::null_mut(), // v3sigma2lapl
                    as_mut_ptr_or_null(v3sigma2tau),
                    std::ptr::null_mut(), // v3sigmalapl2
                    std::ptr::null_mut(), // v3sigmalapltau
                    as_mut_ptr_or_null(v3sigmatau2),
                    std::ptr::null_mut(), // v3lapl3
                    std::ptr::null_mut(), // v3lapl2tau
                    std::ptr::null_mut(), // v3lapltau2
                    as_mut_ptr_or_null(v3tau3),
                    std::ptr::null_mut(), // v4rho4
                    std::ptr::null_mut(), // v4rho3sigma
                    std::ptr::null_mut(), // v4rho3lapl
                    std::ptr::null_mut(), // v4rho3tau
                    std::ptr::null_mut(), // v4rho2sigma2
                    std::ptr::null_mut(), // v4rho2sigmalapl
                    std::ptr::null_mut(), // v4rho2sigmatau
                    std::ptr::null_mut(), // v4rho2lapl2
                    std::ptr::null_mut(), // v4rho2lapltau
                    std::ptr::null_mut(), // v4rho2tau2
                    std::ptr::null_mut(), // v4rhosigma3
                    std::ptr::null_mut(), // v4rhosigma2lapl
                    std::ptr::null_mut(), // v4rhosigma2tau
                    std::ptr::null_mut(), // v4rhosigmalapl2
                    std::ptr::null_mut(), // v4rhosigmalapltau
                    std::ptr::null_mut(), // v4rhosigmatau2
                    std::ptr::null_mut(), // v4rholapl3
                    std::ptr::null_mut(), // v4rholapl2tau
                    std::ptr::null_mut(), // v4rholapltau2
                    std::ptr::null_mut(), // v4rhotau3
                    std::ptr::null_mut(), // v4sigma4
                    std::ptr::null_mut(), // v4sigma3lapl
                    std::ptr::null_mut(), // v4sigma3tau
                    std::ptr::null_mut(), // v4sigma2lapl2
                    std::ptr::null_mut(), // v4sigma2lapltau
                    std::ptr::null_mut(), // v4sigma2tau2
                    std::ptr::null_mut(), // v4sigmalapl3
                    std::ptr::null_mut(), // v4sigmalapl2tau
                    std::ptr::null_mut(), // v4sigmalapltau2
                    std::ptr::null_mut(), // v4sigmatau3
                    std::ptr::null_mut(), // v4lapl4
                    std::ptr::null_mut(), // v4lapl3tau
                    std::ptr::null_mut(), // v4lapl2tau2
                    std::ptr::null_mut(), // v4lapltau3
                    std::ptr::null_mut(), // v4tau4
                );
            }
        },

        _ => {
            panic!("Unsupported functional family: {:?}", xc_func.xc_func_family);
        },

    }

}


pub fn eval_libxc_func(xc_func: &XcFuncType, spin: usize, deriv: usize, np: usize, rho: &[f64], sigma: Option<&[f64]>, lapl: Option<&[f64]>, tau: Option<&[f64]>, exc: &mut [f64]) {
    /*
        Help function to evaluate xc tensor (currently up to 3rd derivative order) for a given libxc functional type.
        Args:
            xc_func: &XcFuncType - libxc functional type
            spin: usize - number of spin (0 for unpolarized, 1 for polarized)
            deriv: usize - derivative order (0 for 0th, 1 for 1st, 2 for 2nd, 3 for 3rd)
            np: usize - number of grid points
            rho: &[f64] - density array, size of (spin+1)*np
            sigma: Option<&[f64]> - density gradient array, size of np for spin=0, 3*np for spin=1 (optional, for GGA and higher)
            lapl: Option<&[f64]> - laplacian array, size of (spin+1)*np (optional, for MGGA, currently not used)
            tau: Option<&[f64]> - kinetic energy density array, size of (spin+1)*np (optional, for MGGA)
            exc: &mut [f64] - output vector for exchange-correlation tensor, size of (nvar + 1)*np, 1 for xc energy density (deriv = 0), nvar refers to number of density derivartives variables depends on functional type and derviative order
    */
    assert!(deriv <= 3, "Derivative order must be 0, 1, 2 or 3");

    // xc derivative variables 
    // 1st order 
    let mut vrho:   Option<&[f64]> = None;
    let mut vsigma: Option<&[f64]> = None;
    let mut vlapl:  Option<&[f64]> = None;
    let mut vtau:   Option<&[f64]> = None;
    // 2nd order
    let mut v2rho2:      Option<&[f64]> = None;
    let mut v2rhosigma:  Option<&[f64]> = None;
    let mut v2sigma2:    Option<&[f64]> = None;
    let mut v2lapl2:     Option<&[f64]> = None;
    let mut v2tau2:      Option<&[f64]> = None;
    let mut v2rholapl:   Option<&[f64]> = None;
    let mut v2rhotau:    Option<&[f64]> = None;
    let mut v2sigmalapl: Option<&[f64]> = None;
    let mut v2sigmatau:  Option<&[f64]> = None;
    let mut v2lapltau:   Option<&[f64]> = None;
    // 3rd order
    let mut v3rho3:         Option<&[f64]> = None;
    let mut v3rho2sigma:    Option<&[f64]> = None;
    let mut v3rhosigma2:    Option<&[f64]> = None;
    let mut v3sigma3:       Option<&[f64]> = None;
    let mut v3rho2lapl:     Option<&[f64]> = None;
    let mut v3rho2tau:      Option<&[f64]> = None;
    let mut v3rhosigmalapl: Option<&[f64]> = None;
    let mut v3rhosigmatau:  Option<&[f64]> = None;
    let mut v3rholapl2:     Option<&[f64]> = None;
    let mut v3rholapltau:   Option<&[f64]> = None;
    let mut v3rhotau2:      Option<&[f64]> = None;
    let mut v3sigma2lapl:   Option<&[f64]> = None;
    let mut v3sigma2tau:    Option<&[f64]> = None;
    let mut v3sigmalapl2:   Option<&[f64]> = None;
    let mut v3sigmalapltau: Option<&[f64]> = None;
    let mut v3sigmatau2:    Option<&[f64]> = None;
    let mut v3lapl3:        Option<&[f64]> = None;
    let mut v3lapl2tau:     Option<&[f64]> = None;
    let mut v3lapltau2:     Option<&[f64]> = None;
    let mut v3tau3:         Option<&[f64]> = None;
    // 4th order 
    let mut v4rho4:            Option<&[f64]> = None;
    let mut v4rho3sigma:       Option<&[f64]> = None;
    let mut v4rho3lapl:        Option<&[f64]> = None;
    let mut v4rho3tau:         Option<&[f64]> = None;
    let mut v4rho2sigma2:      Option<&[f64]> = None;
    let mut v4rho2sigmalapl:   Option<&[f64]> = None;
    let mut v4rho2sigmatau:    Option<&[f64]> = None;
    let mut v4rho2lapl2:       Option<&[f64]> = None;
    let mut v4rho2lapltau:     Option<&[f64]> = None;
    let mut v4rho2tau2:        Option<&[f64]> = None;
    let mut v4rhosigma3:       Option<&[f64]> = None;
    let mut v4rhosigma2lapl:   Option<&[f64]> = None;
    let mut v4rhosigma2tau:    Option<&[f64]> = None;
    let mut v4rhosigmalapl2:   Option<&[f64]> = None;
    let mut v4rhosigmalapltau: Option<&[f64]> = None;
    let mut v4rhosigmatau2:    Option<&[f64]> = None;
    let mut v4rholapl3:        Option<&[f64]> = None;
    let mut v4rholapl2tau:     Option<&[f64]> = None;
    let mut v4rholapltau2:     Option<&[f64]> = None;
    let mut v4rhotau3:         Option<&[f64]> = None;
    let mut v4sigma4:          Option<&[f64]> = None;
    let mut v4sigma3lapl:      Option<&[f64]> = None;
    let mut v4sigma3tau:       Option<&[f64]> = None;
    let mut v4sigma2lapl2:     Option<&[f64]> = None;
    let mut v4sigma2lapltau:   Option<&[f64]> = None;
    let mut v4sigma2tau2:      Option<&[f64]> = None;
    let mut v4sigmalapl3:      Option<&[f64]> = None;
    let mut v4sigmalapl2tau:   Option<&[f64]> = None;
    let mut v4sigmalapltau2:   Option<&[f64]> = None;
    let mut v4sigmatau3:       Option<&[f64]> = None;
    let mut v4lapl4:           Option<&[f64]> = None;
    let mut v4lapl3tau:        Option<&[f64]> = None;
    let mut v4lapl2tau2:       Option<&[f64]> = None;
    let mut v4lapltau3:        Option<&[f64]> = None;
    let mut v4tau4:            Option<&[f64]> = None;


    // println!("Debug: In eval_libxc_func, rho: {:?}", rho);
    // if let Some(sigma) = sigma {
    //     println!("Debug: In eval_libxc_func, sigma: {:?}", sigma);
    // }

    match xc_func.xc_func_family {
        LibXCFamily::LDA => {
            if spin == 1 {
                if deriv > 0 {
                    vrho = Some(&exc[np..np*3]);
                }
                if deriv > 1 {
                    v2rho2 = Some(&exc[np*3..np*6]);
                }
                if deriv > 2 {
                    v3rho3 = Some(&exc[np*6..np*9]);
                }
            } else {
                if deriv > 0 {
                    vrho = Some(&exc[np..np*2]);
                }
                if deriv > 1 {
                    v2rho2 = Some(&exc[np*2..np*3]);
                }
                if deriv > 2 {
                    v3rho3 = Some(&exc[np*3..np*4]);
                }
            }
            unsafe {
                ffi_xc::xc_lda(
                    xc_func.xc_func_type,
                    np as u64,
                    rho.as_ptr(),
                    exc.as_ptr() as *mut _,
                    vrho.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v2rho2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3rho3.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rho4.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                );
            }
            // if deriv > 0 {
            //     println!("Debug: In libxc_eval_xc exc: {:?}", exc);
            //     println!("Debug: In libxc_eval_xc vrho: {:?}", vrho);;
            // }
            // if let Some(vrho) = vrho {
            //     println!("In eval_xc_func: vrho: {:?}", vrho);
            // }
        },
        
        LibXCFamily::GGA | LibXCFamily::HybridGGA => {
            if spin == 1 {
                if deriv > 0 {
                    vrho = Some(&exc[np..np*3]);
                    vsigma = Some(&exc[np*3..np*6]);
                }
                if deriv > 1 {
                    v2rho2 = Some(&exc[np*6..np*9]);
                    v2rhosigma = Some(&exc[np*9..np*15]);
                    v2sigma2 = Some(&exc[np*15..np*21]);
                }
                if deriv > 2 {
                    v3rho3 = Some(&exc[np*21..np*25]);
                    v3rho2sigma = Some(&exc[np*25..np*34]);
                    v3rhosigma2 = Some(&exc[np*34..np*46]);
                    v3sigma3 = Some(&exc[np*34..np*44]);
                }
            } else {
                if deriv > 0 {
                    vrho = Some(&exc[np..np*2]);
                    vsigma = Some(&exc[np*2..np*3]);
                }
                if deriv > 1 {
                    v2rho2 = Some(&exc[np*3..np*4]);
                    v2rhosigma = Some(&exc[np*4..np*5]);
                    v2sigma2 = Some(&exc[np*5..np*6]);
                }
                if deriv > 2 {
                    v3rho3 = Some(&exc[np*6..np*7]);
                    v3rho2sigma = Some(&exc[np*7..np*8]);
                    v3rhosigma2 = Some(&exc[np*8..np*9]);
                    v3sigma3 = Some(&exc[np*9..np*10]);
                }
            }
            unsafe {
                ffi_xc::xc_gga(
                    xc_func.xc_func_type,
                    np as u64,
                    rho.as_ptr(),
                    sigma.unwrap().as_ptr(),
                    exc.as_ptr() as *mut _,
                    vrho.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    vsigma.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v2rho2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v2rhosigma.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v2sigma2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3rho3.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3rho2sigma.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3rhosigma2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3sigma3.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rho4.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rho3sigma.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rho2sigma2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rhosigma3.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4sigma4.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                );
            }
            // if deriv > 0 {
            //     println!("Debug: In libxc_eval_xc exc: {:?}", exc);
            //     println!("Debug: In libxc_eval_xc vrho: {:?}", vrho);
            //     println!("Debug: In libxc_eval_xc vsigma: {:?}", vsigma);
            // }
        },
        
        LibXCFamily::MGGA | LibXCFamily::HybridMGGA => {
            if spin == 1 {
                if deriv > 0 {
                    vrho = Some(&exc[np..np*3]);
                    vsigma = Some(&exc[np*3..np*6]);
                    vtau = Some(&exc[np*9..np*11]);
                }
                if deriv > 1 {
                    v2rho2 = Some(&exc[np*11..np*14]);
                    v2rhosigma = Some(&exc[np*14..np*20]);
                    v2sigma2 = Some(&exc[np*20..np*26]);
                    v2rhotau = Some(&exc[np*26..np*30]);
                    v2sigmatau = Some(&exc[np*30..np*36]);
                    v2tau2 = Some(&exc[np*36..np*39]);
                }
                if deriv > 2 {
                    v3rho3 = Some(&exc[np*39..np*43]);
                    v3rho2sigma = Some(&exc[np*43..np*52]);
                    v3rhosigma2 = Some(&exc[np*52..np*64]);
                    v3sigma3 = Some(&exc[np*64..np*74]);
                    v3rho2tau = Some(&exc[np*74..np*80]);
                    v3rhosigmatau = Some(&exc[np*80..np*92]);
                    v3rhotau2 = Some(&exc[np*92..np*98]);
                    v3sigma2tau = Some(&exc[np*98..np*110]);
                    v3sigmatau2 = Some(&exc[np*110..np*119]);
                    v3tau3 = Some(&exc[np*119..np*123]);
                }
            } else {
                if deriv > 0 {
                    vrho = Some(&exc[np..np*2]);
                    vsigma = Some(&exc[np*2..np*3]);
                    vtau = Some(&exc[np*3..np*4]);
                }
                if deriv > 1 {
                    v2rho2 = Some(&exc[np*4..np*5]);
                    v2rhosigma = Some(&exc[np*5..np*6]);
                    v2sigma2 = Some(&exc[np*6..np*7]);
                    v2rhotau = Some(&exc[np*7..np*8]);
                    v2sigmatau = Some(&exc[np*8..np*9]);
                    v2tau2 = Some(&exc[np*9..np*10]);
                }
                if deriv > 2 {
                    v3rho3 = Some(&exc[np*10..np*11]);
                    v3rho2sigma = Some(&exc[np*11..np*12]);
                    v3rhosigma2 = Some(&exc[np*12..np*13]);
                    v3sigma3 = Some(&exc[np*13..np*14]);
                    v3rho2tau = Some(&exc[np*14..np*15]);
                    v3rhosigmatau = Some(&exc[np*15..np*16]);
                    v3rhotau2 = Some(&exc[np*16..np*17]);
                    v3sigma2tau = Some(&exc[np*17..np*18]);
                    v3sigmatau2 = Some(&exc[np*18..np*19]);
                    v3tau3 = Some(&exc[np*19..np*20]);
                }
            }
            unsafe {
                ffi_xc::xc_mgga(
                    xc_func.xc_func_type,
                    np as u64,
                    rho.as_ptr(),
                    sigma.unwrap().as_ptr(),
                    std::ptr::null(), // laplacian is not used currently
                    tau.unwrap().as_ptr(),
                    exc.as_ptr() as *mut _,
                    vrho.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    vsigma.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    vlapl.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    vtau.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v2rho2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v2rhosigma.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v2rholapl.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v2rhotau.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v2sigma2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v2sigmalapl.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v2sigmatau.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v2lapl2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v2lapltau.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v2tau2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3rho3.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3rho2sigma.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3rho2lapl.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3rho2tau.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3rhosigma2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3rhosigmalapl.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3rhosigmatau.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3rholapl2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3rholapltau.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3rhotau2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3sigma3.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3sigma2lapl.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3sigma2tau.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3sigmalapl2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3sigmalapltau.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3sigmatau2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3lapl3.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3lapl2tau.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3lapltau2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v3tau3.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rho4.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rho3sigma.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rho3lapl.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rho3tau.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rho2sigma2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rho2sigmalapl.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rho2sigmatau.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rho2lapl2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rho2lapltau.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rho2tau2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rhosigma3.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rhosigma2lapl.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rhosigma2tau.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rhosigmalapl2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rhosigmalapltau.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rhosigmatau2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rholapl3.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rholapl2tau.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rholapltau2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4rhotau3.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4sigma4.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4sigma3lapl.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4sigma3tau.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4sigma2lapl2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4sigma2lapltau.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4sigma2tau2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4sigmalapl3.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4sigmalapl2tau.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4sigmalapltau2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4sigmatau3.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4lapl4.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4lapl3tau.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4lapl2tau2.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4lapltau3.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _,
                    v4tau4.map_or(std::ptr::null(), |v| v.as_ptr()) as *mut _
                );
            }
        },
        
        _ => {
            panic!("Unsupported functional family: {:?}", xc_func.xc_func_family);
        },

    }
}


#[test]
fn test_libxc() {
    let rho:Vec<f64> = vec![0.1,0.2,0.3,0.4,0.5,0.6,0.8];
    let sigma:Vec<f64> = vec![0.2,0.3,0.4,0.5,0.6,0.7];
    //let mut exc:Vec<c_double> = vec![0.0,0.0,0.0,0.0,0.0];
    //let mut vrho:Vec<c_double> = vec![0.0,0.0,0.0,0.0,0.0];
    let func_id: usize = ffi_xc::XC_GGA_X_XPBE as usize;
    let spin_channel: usize = 1;

    let mut my_xc = XcFuncType::xc_func_init(1,spin_channel); 
    //let mut my_xc = XcFuncType::xc_func_init_fdqc(&"pw-lda",spin_channel); 

    my_xc.xc_version();

    my_xc.xc_func_info_printout();

    let (exc, vrho) = my_xc.lda_exc_vxc(&rho);

    println!("{:?}", exc);
    println!("{:?}", vrho);

    //let xc_info = my_xc.xc_func_get_info();

    //let xc_name = unsafe {
    //    let c_buf = ffi_xc::xc_func_info_get_name(xc_info);
    //    let c_str = CStr::from_ptr(c_buf);
    //    let str_slice = c_str.to_str().unwrap();
    //    str_slice.to_owned()
    //};
    //println!("{}",xc_name);


    //let xc_name = unsafe{String::from_raw_parts(xc_name, 10, 10)};

    

    //println!("{:?}",my_xc.);

    //let (exc,vrho) = my_xc.xc_exc_vxc(&rho, &sigma).unwrap();

    //println!("{:?}", exc.data);
    //println!("{:?}", vrho.data);


    //let mut p_xc_func_type = unsafe{ffi_xc::xc_func_alloc()};

    //let init = unsafe{ffi_xc::xc_func_init(p_xc_func_type,func_id, ffi_xc::XC_UNPOLARIZED as c_int)};

    //unsafe{
    //    let c_exc = (exc.as_mut_ptr(),exc.len(),exc.capacity());
    //    let c_vrho = (vrho.as_mut_ptr(),vrho.len(),vrho.capacity());
    //    ffi_xc::xc_lda_exc_vxc(p_xc_func_type,5,rho.as_ptr(),c_exc.0,c_vrho.0);
    //    exc = Vec::from_raw_parts(c_exc.0,c_exc.1,c_exc.2);
    //    vrho = Vec::from_raw_parts(c_vrho.0,c_vrho.1,c_vrho.2);
    //};
    //println!("{:?}", exc);
    //println!("{:?}", vrho);

    //unsafe{ffi_xc::xc_func_end(p_xc_func_type)};

    my_xc.xc_func_end()
}

