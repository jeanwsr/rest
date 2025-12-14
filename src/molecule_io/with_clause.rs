#![allow(non_snake_case)]

use crate::Molecule;

const PTR_RINV_ORIG: usize = rest_libcint::ffi::cint_ffi::PTR_RINV_ORIG as usize;
const PTR_RINV_ZETA: usize = rest_libcint::ffi::cint_ffi::PTR_RINV_ZETA as usize;
const PTR_ZETA: usize = rest_libcint::ffi::cint_ffi::PTR_ZETA as usize;
const AS_RINV_ORIG_ATOM: usize = rest_libcint::ffi::cecp_ffi::AS_RINV_ORIG_ATOM as usize;

impl Molecule {
    pub fn get_rinv_origin(&self) -> [f64; 3] {
        self.cint_env[PTR_RINV_ORIG..(PTR_RINV_ORIG + 3)].try_into().unwrap()
    }

    pub fn set_rinv_origin(&mut self, rinv_origin: [f64; 3]) {
        self.cint_env[PTR_RINV_ORIG..(PTR_RINV_ORIG + 3)].copy_from_slice(&rinv_origin);
    }

    pub fn with_rinv_origin<F, R>(&mut self, rinv_origin: [f64; 3], f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let original = self.get_rinv_origin();
        self.set_rinv_origin(rinv_origin);
        let result = f(self);
        self.set_rinv_origin(original);
        result
    }

    pub fn get_rinv_zeta(&self) -> f64 {
        self.cint_env[PTR_RINV_ZETA]
    }

    pub fn set_rinv_zeta(&mut self, rinv_zeta: f64) {
        self.cint_env[PTR_RINV_ZETA] = rinv_zeta;
    }

    pub fn with_rinv_zeta<F, R>(&mut self, rinv_zeta: f64, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let original = self.get_rinv_zeta();
        self.set_rinv_zeta(rinv_zeta);
        let result = f(self);
        self.set_rinv_zeta(original);
        result
    }

    pub fn get_rinv_orig_atom(&self) -> usize {
        self.cint_env[AS_RINV_ORIG_ATOM] as usize
    }

    pub fn set_rinv_orig_atom(&mut self, rinv_orig_atom: usize) {
        self.cint_env[AS_RINV_ORIG_ATOM] = rinv_orig_atom as f64;
    }

    pub fn with_rinv_orig_atom<F, R>(&mut self, rinv_orig_atom: usize, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let original = self.get_rinv_orig_atom();
        self.set_rinv_orig_atom(rinv_orig_atom);
        let result = f(self);
        self.set_rinv_orig_atom(original);
        result
    }

    pub fn with_rinv_at_nucleus<F, R>(&mut self, atm_id: usize, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let zeta = self.cint_env[self.cint_atm[atm_id][PTR_ZETA] as usize];
        let rinv = self.geom.get_coord(atm_id).try_into().unwrap();

        if zeta == 0.0 {
            self.with_rinv_orig_atom(atm_id, |mol| mol.with_rinv_origin(rinv, f))
        } else {
            self.with_rinv_orig_atom(atm_id, |mol| mol.with_rinv_origin(rinv, |mol| mol.with_rinv_zeta(zeta, f)))
        }
    }
}
