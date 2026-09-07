//! Optimized RI-JK Hessian computation for unrestricted SCF (UHF).
//!
//! This module reuses the J/K-separated skeleton driver
//! [`crate::ri_jk::hess_r::get_rijk_skeleton_decomposed_separated`] (which natively handles an
//! arbitrary number of spin sets) to produce J (from the total density) and both K spins
//! (K^alpha, K^beta) in a **single** pass that shares every 3c-integral batch.
//!
//! - J (Coulomb) is spin-independent: built from the total density ``D^alpha + D^beta``.
//! - K (exchange) is ``K^alpha + K^beta``; each spin channel is built from its own occupied
//!   orbitals (UHF occ = 1, so ``mocc_2 = mocc``).
//!
//! Scaling difference to the restricted ([`crate::ri_jk::hess_r::RHessRIJK`]) counterpart:
//! - Skeleton: ``factor_j * de_J - factor_k * de_K`` (K coefficient ``-1``, not ``-0.5`` as in RHF),
//!   because UHF ``de_K = K^alpha + K^beta`` already absorbs the spin sum (matches
//!   [`crate::ri_jk::hess_u_naive::UHessRIJKNaive`]).
//! - First derivative (bra form): ``factor_j * (j1ao @ mocc_s) - factor_k * k1bra_s`` per spin (no
//!   ``0.5`` factor), again matching the naive UHF convention.
//!
//! The response (`get_response_bra`) reuses the separated J/K response core
//! [`crate::ri_jk::hess_r::get_rijk_response_bra_separated`] shared with RHF: J is produced once
//! in AO form from the total density response and right half-transformed per spin; K is produced
//! per spin in bra form (same-spin only).

use super::prelude_dev::*;
use crate::analdrv::prelude::*;
use crate::ri_jk::util::*;

use crate::ri_jk::decompose::*;
use crate::ri_jk::hess_r::{
    generate_cderi_with_decomp, get_rijk_response_bra_separated, get_rijk_skeleton_decomposed_separated, KEYS_J02,
    KEYS_J11, KEYS_J1AO, KEYS_J20, KEYS_K02, KEYS_K11, KEYS_K1BRA, KEYS_K20,
};

/* #region impl */

pub struct UHessRIJK<'a> {
    pub mol: CInt,
    pub aux: CInt,
    pub factor_j: f64,
    pub factor_k: f64,
    pub cderi: TsrCow<'a>,
    pub j2c_decomp: J2CDecompose,
    pub intmd: HashMap<String, Tsr>, // intermediates
    pub result: HashMap<&'static str, Tsr>,
    pub timing: Vec<(String, f64)>,
    pub is_skeleton_ready: bool,
}

impl<'a> UHessRIJK<'a> {
    pub fn new_without_cderi(mol: &CInt, aux: &CInt, factor_j: f64, factor_k: f64) -> Self {
        let j2c_decomp_option = J2CDecompOption { policy: J2CDecompPolicy::Cd, threshold: Some(1e-14), uplo: Upper, distributed: J2CDistributedMode::Off };
        let device = DeviceBLAS::default();
        let (cderi, j2c_decomp) = generate_cderi_with_decomp(mol, aux, j2c_decomp_option, &device);
        Self {
            mol: mol.clone(),
            aux: aux.clone(),
            factor_j,
            factor_k,
            cderi: cderi.into_cow(),
            j2c_decomp,
            intmd: HashMap::new(),
            result: HashMap::new(),
            timing: Vec::new(),
            is_skeleton_ready: false,
        }
    }

    pub fn new_with_cderi(
        mol: &CInt,
        aux: &CInt,
        factor_j: f64,
        factor_k: f64,
        cderi: TsrCow<'a>,
        j2c_decomp: J2CDecompose,
    ) -> Self {
        Self {
            mol: mol.clone(),
            aux: aux.clone(),
            factor_j,
            factor_k,
            cderi: cderi.into_cow(),
            j2c_decomp,
            intmd: HashMap::new(),
            result: HashMap::new(),
            timing: Vec::new(),
            is_skeleton_ready: false,
        }
    }

    /// Build the total density ``D^alpha + D^beta`` from the per-spin mo_coeff / mo_occ.
    ///
    /// This is only used for an explicit sanity check; the skeleton driver builds the same total
    /// density internally when `dm0` is passed as `None`.
    #[allow(dead_code)]
    fn get_total_dm0(&self, mo_coeff: &[TsrView; 2], mo_occ: &[TsrView; 2]) -> Tsr {
        let nao = mo_coeff[0].shape()[0];
        let device = self.cderi.device();
        let mut dm0 = rt::zeros(([nao, nao], device));
        for s in 0..2 {
            dm0 += get_dm0_restricted(mo_coeff[s].view(), mo_occ[s].view());
        }
        dm0
    }

    pub fn ensure_skeleton(&mut self, mo_coeff: &[TsrView; 2], mo_occ: &[TsrView; 2], atm_list: Option<&[usize]>) {
        if self.is_skeleton_ready {
            return;
        }
        // nset = 2 (alpha, beta); the driver builds the total density for J internally and produces
        // one K output per spin set, sharing every 3c-integral batch.
        let mo_coeff_slice: &[TsrView] = &[mo_coeff[0].view(), mo_coeff[1].view()];
        let mo_occ_slice: &[TsrView] = &[mo_occ[0].view(), mo_occ[1].view()];
        let (j_out, k_outs, timing) = get_rijk_skeleton_decomposed_separated(
            &self.mol,
            &self.aux,
            mo_coeff_slice,
            mo_occ_slice,
            self.cderi.view(),
            &self.j2c_decomp,
            self.factor_j != 0.0,
            self.factor_k != 0.0,
            72, // TODO: batch size `72` should be tunable by max-memory.
            atm_list,
            None,
        );
        self.timing.extend(timing);

        if let Some(j_out) = j_out {
            for (key, value) in j_out.into_iter() {
                self.intmd.insert(key.to_string(), value);
            }
        };

        for (iset, k_out) in k_outs.into_iter().enumerate() {
            // note the keys can clash for output of k across spin sets.
            // for storage of intermediates, we append `<spin_{iset}>` to the key name.
            for (key, value) in k_out.into_iter() {
                self.intmd.insert(format!("{key}<spin_{iset}>"), value);
            }
        }

        self.is_skeleton_ready = true;
    }
}

impl<'a> AnalDrvBaseAPI for UHessRIJK<'a> {}

impl<'a> UHessElecInteractAPI for UHessRIJK<'a> {
    fn make_skeleton_hess(
        &mut self,
        mo_coeff: &[TsrView; 2],
        mo_occ: &[TsrView; 2],
        atm_list: Option<&[usize]>,
    ) -> Tsr {
        self.ensure_skeleton(mo_coeff, mo_occ, atm_list);
        let intmd = &self.intmd;

        let device = self.cderi.device();
        let natm = atm_list.map_or_else(|| self.mol.natm(), |list| list.len());
        let hess_init = || -> Tsr { rt::zeros(([3, 3, natm, natm], device)) };

        // helper: sum a set of K keys over both spin channels
        let sum_k_keys = |keys: &[&'static str]| -> Tsr {
            let mut acc = hess_init();
            for s in 0..2 {
                for &key in keys {
                    acc += &intmd[&format!("{key}<spin_{s}>")];
                }
            }
            acc
        };

        let mut de = hess_init();
        if self.factor_j != 0.0 {
            let de_J20 = KEYS_J20.iter().map(|&key| &intmd[key]).fold(hess_init(), |acc, x| acc + x);
            let de_J11 = KEYS_J11.iter().map(|&key| &intmd[key]).fold(hess_init(), |acc, x| acc + x);
            let de_J02 = KEYS_J02.iter().map(|&key| &intmd[key]).fold(hess_init(), |acc, x| acc + x);
            let de_J = &de_J20 + &de_J11 + &de_J02;
            de += self.factor_j * &de_J;
            self.result.insert("de_J20", de_J20);
            self.result.insert("de_J11", de_J11);
            self.result.insert("de_J02", de_J02);
            self.result.insert("de_J", de_J);
        }
        if self.factor_k != 0.0 {
            let de_K20 = sum_k_keys(&KEYS_K20);
            let de_K11 = sum_k_keys(&KEYS_K11);
            let de_K02 = sum_k_keys(&KEYS_K02);
            let de_K = &de_K20 + &de_K11 + &de_K02;
            // UHF: K coefficient is -1 (not -0.5 as in RHF) because de_K already includes the spin sum.
            de -= self.factor_k * &de_K;
            self.result.insert("de_K20", de_K20);
            self.result.insert("de_K11", de_K11);
            self.result.insert("de_K02", de_K02);
            self.result.insert("de_K", de_K);
        }
        self.result.insert("de_skeleton", de.clone());
        de
    }

    fn get_deriv1_ao(
        &mut self,
        _mo_coeff: &[TsrView; 2],
        _mo_occ: &[TsrView; 2],
        _atm_list: Option<&[usize]>,
    ) -> [Tsr; 2] {
        unimplemented!("This function is not implemented for optimized RI-JK hessian. Use `get_deriv1_bra` instead.")
    }

    fn get_deriv1_bra(
        &mut self,
        mo_coeff: &[TsrView; 2],
        mo_occ: &[TsrView; 2],
        atm_list: Option<&[usize]>,
    ) -> [Tsr; 2] {
        self.ensure_skeleton(mo_coeff, mo_occ, atm_list);
        let intmd = &self.intmd;

        let device = self.cderi.device();
        let natm = atm_list.map_or_else(|| self.mol.natm(), |list| list.len());
        let nao = mo_coeff[0].shape()[0];
        let occidx = [mo_occ[0].view().greater(0.0).into_vec(), mo_occ[1].view().greater(0.0).into_vec()];
        let nocc = [occidx[0].iter().filter(|&&x| x).count(), occidx[1].iter().filter(|&&x| x).count()];
        let mocc = [
            mo_coeff[0].view().bool_select(-1, &occidx[0]).into_contig(ColMajor),
            mo_coeff[1].view().bool_select(-1, &occidx[1]).into_contig(ColMajor),
        ];

        let deriv1_ao_init = || -> Tsr { rt::zeros(([nao, nao, 3, natm], device)) };
        let deriv1_bra_init = |s: usize| -> Tsr { rt::zeros(([nao, nocc[s], 3, natm], device)) };

        let mut deriv1_bra = [deriv1_bra_init(0), deriv1_bra_init(1)];

        // J is spin-independent (held in AO form, shared across spins); right half-transform per spin.
        if self.factor_j != 0.0 {
            let j1ao = KEYS_J1AO.iter().map(|&key| &intmd[key]).fold(deriv1_ao_init(), |acc, x| acc + x);
            for s in 0..2 {
                deriv1_bra[s] += self.factor_j * (&j1ao % &mocc[s]);
            }
            self.result.insert("j1ao", j1ao);
        }
        // K is spin-resolved; k1bra^s is stored as the right half-transform ``k1ao^s @ mocc_s``
        // (shape ``[nao, nocc_s, 3, natm]``), matching the restricted optimized convention.
        // Note: per-spin k1bra have different `nocc_s` and cannot be summed across spins.
        if self.factor_k != 0.0 {
            for s in 0..2 {
                let ks = KEYS_K1BRA
                    .iter()
                    .map(|&key| &intmd[&format!("{key}<spin_{s}>")])
                    .fold(deriv1_bra_init(s), |acc, x| acc + x);
                // UHF: no 0.5 factor (occ = 1), unlike RHF.
                deriv1_bra[s] -= self.factor_k * &ks;
                self.result.insert(if s == 0 { "k1bra_0" } else { "k1bra_1" }, ks);
            }
        }
        self.result.insert("deriv1_bra_0", deriv1_bra[0].clone());
        self.result.insert("deriv1_bra_1", deriv1_bra[1].clone());
        deriv1_bra
    }

    fn make_response_preparation(&mut self, mo_coeff: &[TsrView; 2], mo_occ: &[TsrView; 2]) {
        self.intmd.insert("mo_coeff_0".to_string(), mo_coeff[0].view().into_contig(RowMajor));
        self.intmd.insert("mo_coeff_1".to_string(), mo_coeff[1].view().into_contig(RowMajor));
        self.intmd.insert("mo_occ_0".to_string(), mo_occ[0].to_owned());
        self.intmd.insert("mo_occ_1".to_string(), mo_occ[1].to_owned());
    }

    fn get_response_bra(&mut self, bra: &[TsrView; 2]) -> [Tsr; 2] {
        let mo_coeff = [self.intmd["mo_coeff_0"].view(), self.intmd["mo_coeff_1"].view()];
        let mo_occ = [self.intmd["mo_occ_0"].view(), self.intmd["mo_occ_1"].view()];
        let cderi = self.cderi.view();
        let device = mo_coeff[0].device();
        // Shared separated J/K response core: J (AO form, from total density) + per-spin K (bra form).
        let (j_ao, k_bras) = get_rijk_response_bra_separated(
            cderi,
            &mo_coeff,
            &mo_occ,
            bra,
            self.factor_j != 0.0,
            self.factor_k != 0.0,
            72, // TODO: batch size `72` should be tunable by max-memory.
        );

        let nao = mo_coeff[0].shape()[0];
        let occidx = [mo_occ[0].view().greater(0.0).into_vec(), mo_occ[1].view().greater(0.0).into_vec()];
        let mocc = [
            mo_coeff[0].view().bool_select(-1, &occidx[0]).into_contig(ColMajor),
            mo_coeff[1].view().bool_select(-1, &occidx[1]).into_contig(ColMajor),
        ];
        let nocc = [mocc[0].shape()[1], mocc[1].shape()[1]];

        let mut resp = [None, None];
        for s in 0..2 {
            let shape = bra[s].shape().to_vec();
            let nprop: usize = shape[2..].iter().product();
            let mut r = rt::zeros(([nao, nocc[s], nprop], device));
            // J: spin-independent AO operator, right half-transformed by this spin's mocc.
            // The shared `j_ao` carries the RHF symmetrization factor (effective `4 * J1`); UHF
            // naive J uses `2 * J1`, so an extra `0.5` prefactor is applied here (occ = 1 vs 2).
            if let Some(j_ao) = j_ao.as_ref() {
                r += 0.5 * self.factor_j * (j_ao.view() % &mocc[s]);
            }
            // K: same-spin bra form (UHF occ = 1, so no 0.5 factor — unlike RHF). The core already
            // bakes in the exchange sign, so this is an additive contribution.
            if let Some(k_bra) = k_bras.get(s) {
                r += self.factor_k * k_bra.view().reshape((nao, nocc[s], nprop));
            }
            resp[s] = Some(r.into_shape(shape));
        }
        [resp[0].take().unwrap(), resp[1].take().unwrap()]
    }
}

/* #endregion */
