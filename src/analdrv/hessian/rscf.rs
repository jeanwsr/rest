//! Hessian implementations for restricted SCF.

use crate::analdrv::prelude::*;

/// Working solver and maintainer of all hessian components for restricted SCF method.
///
/// The lifetime parameters decouple the response object's borrow from its data, and the response
/// object's data region from the hessian component objects: `'a` is the region of the SCF data
/// the response object borrows, `'b` the (typically shorter) borrow of `resp` itself, and `'c`
/// the region of the hessian component objects (`ovlp_obj` and the lists). Bundling `'a` with
/// `'b` (`&'a mut RRespSCF<'a>`) is rejected by dropck: the borrow would have to end before the
/// referent's destructor, while that destructor pins the pointee's lifetime until the referent's
/// drop. Keeping `'a` separate from `'c` allows the response object to be shared from outside
/// (built once by the caller, e.g. the analdrv task loop) while the component objects remain
/// function-local.
pub struct RHessSCF<'a, 'b, 'c> {
    pub mo_coeff: Tsr,
    pub mo_occ: Tsr,
    pub mo_energy: Tsr,
    pub ovlp_obj: &'c mut RHessOvlp,
    pub nuc_list: Vec<&'c mut dyn HessNucAPI>,
    pub core_list: Vec<&'c mut dyn RHessCoreAPI>,
    pub el_list: Vec<&'c mut dyn RHessElecInteractAPI>,
    /// Response (fock/response) object for the electron-interaction contributions (the composite
    /// `RRespSCF` of the SCF); the CP-SCF machinery (`make_response_preparation`/`response_mo`)
    /// calls it, separate from the skeleton-hessian objects in `el_list`.
    pub resp: &'b mut RRespSCF<'a>,
    /// The `AnalDrvNucgradCfg` sub-config (`atm_list` etc.); the CP-SCF solver settings
    /// (`level_shift` etc.) come from the captured `resp_cfg` on `self.resp`.
    pub cfg: AnalDrvNucgradCfg,
    pub result: HashMap<String, Tsr>,
    /// Timing information. Represented by wall time in second.
    pub timing: Vec<(String, f64)>,
}

impl<'a, 'b, 'c> RHessSCF<'a, 'b, 'c> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mo_coeff: Tsr,
        mo_occ: Tsr,
        mo_energy: Tsr,
        ovlp_obj: &'c mut RHessOvlp,
        nuc_list: Vec<&'c mut dyn HessNucAPI>,
        core_list: Vec<&'c mut dyn RHessCoreAPI>,
        el_list: Vec<&'c mut dyn RHessElecInteractAPI>,
        resp: &'b mut RRespSCF<'a>,
        cfg: &AnalDrvNucgradCfg,
    ) -> Self {
        Self {
            mo_coeff,
            mo_occ,
            mo_energy,
            ovlp_obj,
            nuc_list,
            core_list,
            el_list,
            resp,
            cfg: cfg.clone(),
            result: HashMap::new(),
            timing: Vec::new(),
        }
    }

    /// Number of atoms over which the Hessian is computed. This is `atm_list.len()` if
    /// `atm_list` is `Some`, otherwise the total number of atoms in the molecule.
    pub fn natm(&self) -> usize {
        match &self.cfg.atm_list {
            Some(list) => list.len(),
            None => self.ovlp_obj.natm(),
        }
    }

    /// Return the list of (global) atom indices the Hessian is computed for, ordered the same
    /// way as the local indexing used in the returned Hessian.
    pub fn atm_indices(&self) -> Vec<usize> {
        match &self.cfg.atm_list {
            Some(list) => list.clone(),
            None => (0..self.ovlp_obj.natm()).collect(),
        }
    }

    /// Compute the dimensionless CP-SCF right-hand side, along with necessary intermediates for
    /// later steps.
    ///
    /// Note there are some differences compared to usual CP-SCF:
    /// - Usual CP-SCF is `(ea - ei) U - AU = B`, where now we handle something like `U + (A / (ea -
    ///   ei)) U = - B / (ea - ei)`
    /// - We now handle the U in all-occ block, instead of standard vir-occ block; this will omit
    ///   the response evaluation during rhs (B), making the rhs evaluation cheap, but we also need
    ///   to carefully handle the CP-SCF equation. this behavior should be similar to PySCF's
    ///   `solve_withs1`.
    ///
    /// Note **dimless** here means the CP-SCF equation is of no quantity dimension (量纲), but not
    /// the tensor structure is dimensionless.
    ///
    /// # Returns
    ///     
    /// A dictionary containing:
    /// - `rhs : shape `[nmo, nocc, 3, natm]`. The dimensionless CP-SCF right-hand side.
    /// - `f1mo` : shape `[nmo, nocc, 3, natm]`. The first-order derivative of the Fock matrix in MO
    ///   basis.
    /// - `s1mo` : shape `[nmo, nocc, 3, natm]`. The first-order derivative of the overlap matrix in
    ///   MO basis.
    pub fn compute_dimless_cpscf_rhs(&mut self) -> HashMap<&'static str, Tsr> {
        // setups
        let t0 = std::time::Instant::now();
        let mo_coeff = &self.mo_coeff;
        let mo_occ = &self.mo_occ;
        let mo_energy = &self.mo_energy;
        let level_shift = self.resp.resp_cfg.level_shift;
        let device = mo_coeff.device().clone();

        let [nao, nmo] = mo_coeff.shape().to_vec().try_into().unwrap();
        let occidx = mo_occ.view().greater(0).into_vec();
        let viridx = occidx.iter().map(|&x| !x).collect_vec();
        let mocc = mo_coeff.bool_select(-1, &occidx);
        let eocc = mo_energy.bool_select(-1, &occidx);
        let evir = mo_energy.bool_select(-1, &viridx);
        let nocc = occidx.iter().filter(|&&x| x).count();
        let natm = self.natm();
        let atm_indices = self.atm_indices();
        let atm_list = self.cfg.atm_list.as_deref();

        let e_ai = evir.i((.., None)) - eocc.i((None, ..));
        let e_ai_shift = &e_ai + level_shift;

        // --- f1mo --- //

        // fock skeleton derivative (core contribution)
        let mut f1ao_core: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
        for core_obj in self.core_list.iter() {
            let t1 = std::time::Instant::now();
            let mut gen_core_deriv1 = core_obj.generator_deriv1();
            for (A_loc, &A_glob) in atm_indices.iter().enumerate() {
                *&mut f1ao_core.i_mut((Ellipsis, A_loc)) += gen_core_deriv1(A_glob);
            }
            self.timing.push((
                format!("in compute_dimless_cpscf_rhs, f1ao_core_{}", core_obj.get_type_name()),
                t1.elapsed().as_secs_f64(),
            ));
        }

        // fock skeleton derivative (electron interaction contribution, half-transformed to bra)
        let mut f1bra_el: Tsr = rt::zeros(([nao, nocc, 3, natm], &device));
        for el_obj in self.el_list.iter_mut() {
            let t1 = std::time::Instant::now();
            f1bra_el += el_obj.get_deriv1_bra(mo_coeff.view(), mo_occ.view(), atm_list);
            self.timing.push((
                format!("in compute_dimless_cpscf_rhs, f1bra_el_{}", el_obj.get_type_name()),
                t1.elapsed().as_secs_f64(),
            ));
        }

        // construct whole f1mo
        let f1bra = f1bra_el + f1ao_core % &mocc;
        let f1mo = mo_coeff.t() % f1bra;

        // --- s1mo --- //

        let t1 = std::time::Instant::now();

        let mut gen_ovlp_deriv1 = self.ovlp_obj.generator_deriv1();
        let mut s1ao: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
        for (A_loc, &A_glob) in atm_indices.iter().enumerate() {
            *&mut s1ao.i_mut((Ellipsis, A_loc)) += gen_ovlp_deriv1(A_glob);
        }
        let s1mo = mo_coeff.t() % (&s1ao % &mocc);

        self.timing.push(("in compute_dimless_cpscf_rhs, s1mo".to_string(), t1.elapsed().as_secs_f64()));

        // --- dimensionless cpscf rhs --- //

        let so = rt::slice!(0, nocc);
        let sv = rt::slice!(nocc, nmo);
        let b1mo = &f1mo - &s1mo * eocc.i((None, ..));
        let mut rhs = rt::zeros(([nmo, nocc, 3, natm], &device));
        *&mut rhs.i_mut(sv) += -b1mo.i(sv) / &e_ai_shift;
        *&mut rhs.i_mut(so) += -0.5 * s1mo.i(so);

        self.timing.push(("compute_dimless_cpscf_rhs".to_string(), t0.elapsed().as_secs_f64()));
        HashMap::from([("f1mo", f1mo), ("s1mo", s1mo), ("rhs", rhs)])
    }

    /// Prepare the response for CP-SCF calculation.
    ///
    /// This involves all response objects, and stores the orbital state in the response object
    /// for its inherent CP-SCF machinery.
    pub fn make_response_preparation(&mut self) {
        let t0 = std::time::Instant::now();
        self.resp.make_cpscf_preparation(self.mo_coeff.view(), self.mo_occ.view(), self.mo_energy.view());
        self.timing.push(("make_response_preparation".to_string(), t0.elapsed().as_secs_f64()));
    }

    /// Finalize the CP-SCF calculation by computing necessary intermediates for Hessian assembly.
    ///
    /// This includes:
    /// - Re-computing the mo1 (as post-iteration computation), as well as removing the level shift.
    /// - Computing the derivative of occupied orbital energy with respect to perturbation (mo_e1).
    ///   Note occupied orbital energy (shape [nocc]) is diagonal of Fock, and Fock matrix is
    ///   diagonal. However, with the definition that `U[occ, occ] = -0.5 S1[occ, occ]`, the
    ///   off-diagonal part of derivative of Fock in occupied-occupied block is not zero. That's why
    ///   this term is actually matrix.
    ///
    /// # Parameters
    ///
    /// - `f1mo` : shape `[nmo, nocc, 3, natm]`. The first-order derivative of the Fock matrix in MO
    ///   basis, obtained from [`Self::compute_dimless_cpscf_rhs`].
    /// - `s1mo` : shape `[nmo, nocc, 3, natm]`. The first-order derivative of the overlap matrix in
    ///   MO basis, obtained from [`Self::compute_dimless_cpscf_rhs`].
    /// - `mo1` : shape `[nmo, nocc, 3, natm]`. The perturbation in MO space obtained from Krylov
    ///   solver.
    ///
    /// # Returns
    ///
    /// `HashMap<&str, Tsr>`
    ///
    /// - `mo1` : shape `[nmo, nocc, 3, natm]`. The finalized perturbation in MO space.
    /// - `mo_e1` : shape `[nocc, nocc, 3, natm]`. The derivative of occupied orbital energies (Fock
    ///   matrix) with respect to perturbation.
    pub fn finalize_cpscf(&mut self, f1mo: TsrView, s1mo: TsrView, mo1: TsrView) -> HashMap<&'static str, Tsr> {
        let t0 = std::time::Instant::now();
        let mo_occ = self.mo_occ.view();
        let mo_energy = self.mo_energy.view();
        let occidx = mo_occ.view().greater(0).into_vec();
        let viridx = occidx.iter().map(|&x| !x).collect_vec();
        let nocc = occidx.iter().filter(|&&x| x).count();
        let nmo = mo_occ.shape()[0];
        let eocc = mo_energy.bool_select(-1, &occidx);
        let evir = mo_energy.bool_select(-1, &viridx);
        let so = rt::slice!(0, nocc);
        let sv = rt::slice!(nocc, nmo);
        let e_ai = evir.i((.., None)) - eocc.i((None, ..));
        let e_ij = eocc.i((.., None)) - eocc.i((None, ..));

        // last-iter the cp-hf equation, and remove the level-shift
        let b1mo = f1mo - s1mo * eocc.i((None, ..)) + self.resp.response_mo(mo1.view());
        let mut mo1 = mo1.to_owned();
        mo1.i_mut(sv).assign(-b1mo.i(sv) / e_ai);

        // get the derivative of fock matrix in occ-occ block (derivative of orbital energy with
        // rotation)
        let mo_e1 = b1mo.i(so) + mo1.i(so) * e_ij;

        self.timing.push(("finalize_cpscf".to_string(), t0.elapsed().as_secs_f64()));
        HashMap::from([("mo1", mo1), ("mo_e1", mo_e1)])
    }

    /// Compute the CP-SCF contribution to the Hessian using the finalized CP-SCF results.
    ///
    /// # Parameters
    ///
    /// - `f1mo` : shape `[nmo, nocc, 3, natm]`. The first-order derivative of the Fock matrix in MO
    ///   basis, obtained from [`Self::compute_dimless_cpscf_rhs`].
    /// - `s1mo` : shape `[nmo, nocc, natm, 3]`. The first-order skeleton derivative of the overlap
    ///   matrix in MO basis, obtained from [`Self::compute_dimless_cpscf_rhs`].
    /// - `mo1` : shape `[nmo, nocc, natm, 3]`. The finalized perturbation in MO space obtained from
    ///   [`Self::finalize_cpscf`].
    /// - `mo_e1` : shape `[nocc, nocc, 3, natm]`. The derivative of occupied orbital energies (Fock
    ///   matrix) with respect to perturbation, obtained from [`Self::finalize_cpscf`].
    ///
    /// # Returns
    ///
    /// - `de_cpscf` : shape `[3, 3, natm, natm]`. The CP-SCF contribution to the Hessian.
    pub fn get_cpscf_hess(&self, f1mo: TsrView, s1mo: TsrView, mo1: TsrView, mo_e1: TsrView) -> Tsr {
        let natm = self.natm();
        let mo_occ = self.mo_occ.view();
        let mo_energy = self.mo_energy.view();
        let occidx = mo_occ.view().greater(0).into_vec();
        let nocc = occidx.iter().filter(|&&x| x).count();
        let eocc = mo_energy.bool_select(-1, &occidx);
        let so = rt::slice!(0, nocc);
        let device = mo1.device().clone();

        let s1oo = s1mo.i(so);
        let mut de_cpscf = rt::zeros(([3, 3, natm, natm], &device));
        // well, code style is ruined by rustfmt ...
        for A in 0..natm {
            for B in 0..=A {
                let mut de_BA = de_cpscf.i_mut((.., .., B, A));
                de_BA += 4 * (f1mo.i((.., .., None, .., A)) * mo1.i((.., .., .., None, B))).sum_axes([0, 1]);
                de_BA -= 4
                    * (s1mo.i((.., .., None, .., A)) * mo1.i((.., .., .., None, B)) * eocc.i((None, ..)))
                        .sum_axes([0, 1]);
                de_BA -= 2 * (s1oo.i((.., .., None, .., A)) * mo_e1.i((.., .., .., None, B))).sum_axes([0, 1]);
            }
            for B in 0..A {
                let de_to_copy = de_cpscf.i((.., .., B, A)).t().to_owned();
                *&mut de_cpscf.i_mut((.., .., A, B)) += de_to_copy;
            }
        }
        de_cpscf
    }

    /// Compute the CP-SCF contribution to the Hessian by running through the entire CP-SCF
    /// workflow.
    ///
    /// - Compute the dimensionless CP-SCF right-hand side and necessary intermediates.
    /// - Prepare the response for CP-SCF calculation.
    /// - Solve the dimensionless CP-SCF equation using a Krylov solver.
    /// - Finalize the CP-SCF results by computing necessary intermediates for Hessian assembly.
    /// - Compute the CP-SCF contribution to the Hessian using the finalized CP-SCF results.
    ///
    /// # Returns
    ///
    /// - `de_cpscf` : shape `[3, 3, natm, natm]`. The CP-SCF contribution to the Hessian.
    ///
    /// Cached on first call (stored in `result` as `de_cpscf`); repeated calls directly return
    /// the stored result.
    pub fn make_cpscf_hess(&mut self) -> Tsr {
        if let Some(de_cpscf) = self.result.get("de_cpscf") {
            return de_cpscf.to_owned();
        }
        let pre_cpscf_dict = self.compute_dimless_cpscf_rhs();
        let f1mo = pre_cpscf_dict["f1mo"].view();
        let s1mo = pre_cpscf_dict["s1mo"].view();
        let rhs = pre_cpscf_dict["rhs"].view();

        self.make_response_preparation();
        let mo1 = self.resp.solve_dimless_cpscf(rhs.view());
        let finalize_dict = self.finalize_cpscf(f1mo.view(), s1mo.view(), mo1.view());
        let mo1 = finalize_dict["mo1"].view();
        let mo_e1 = finalize_dict["mo_e1"].view();

        let de_cpscf = self.get_cpscf_hess(f1mo.view(), s1mo.view(), mo1.view(), mo_e1.view());
        self.result.insert("de_cpscf".to_string(), de_cpscf.to_owned());
        de_cpscf
    }

    /// Compute the total skeleton contribution to the Hessian.
    ///
    /// **Total** means that we sum over all skeleton contributions from both core and
    /// electron-interaction objects.
    ///
    /// # Parameters
    ///
    /// - `mo_coeff` : shape `[nao, nmo]`. The molecular orbital coefficients.
    /// - `mo_occ` : shape `[nmo]`. The orbital occupations.
    ///
    /// # Returns
    ///
    /// - `de_skeleton` : shape `[3, 3, natm, natm]`. The total skeleton contribution to the
    ///   Hessian.
    ///
    /// Cached on first call (stored in `result` as `de_skeleton`); repeated calls directly return
    /// the stored result.
    pub fn make_skeleton_hess(&mut self) -> Tsr {
        if let Some(de_skeleton) = self.result.get("de_skeleton") {
            return de_skeleton.to_owned();
        }
        let natm = self.natm();
        let mo_coeff = self.mo_coeff.view();
        let mo_occ = self.mo_occ.view();
        let atm_list = self.cfg.atm_list.as_deref();

        let device = self.mo_coeff.device().clone();
        let mut de_skeleton = rt::zeros(([3, 3, natm, natm], &device));
        for nuc_obj in self.nuc_list.iter_mut() {
            let t0 = std::time::Instant::now();
            let de_nuc = nuc_obj.make_skeleton_hess(atm_list);
            let nuc_obj_name = nuc_obj.get_type_name();
            self.result.insert(format!("de_skeleton_{}", nuc_obj_name), de_nuc.to_owned());
            self.timing.push((format!("de_skeleton_{}", nuc_obj_name,), t0.elapsed().as_secs_f64()));
            de_skeleton += de_nuc;
        }
        for core_obj in self.core_list.iter_mut() {
            let t0 = std::time::Instant::now();
            let de_core = core_obj.make_skeleton_hess(mo_coeff.view(), mo_occ.view(), atm_list);
            let core_obj_name = core_obj.get_type_name();
            self.result.insert(format!("de_skeleton_{}", core_obj_name), de_core.to_owned());
            self.timing.push((format!("de_skeleton_{}", core_obj_name,), t0.elapsed().as_secs_f64()));
            de_skeleton += de_core;
        }
        for el_obj in self.el_list.iter_mut() {
            let t0 = std::time::Instant::now();
            let de_el = el_obj.make_skeleton_hess(mo_coeff.view(), mo_occ.view(), atm_list);
            let el_obj_name = el_obj.get_type_name();
            self.result.insert(format!("de_skeleton_{}", el_obj_name), de_el.to_owned());
            self.timing.push((format!("de_skeleton_{}", el_obj_name,), t0.elapsed().as_secs_f64()));
            de_skeleton += de_el;
        }
        self.result.insert("de_skeleton".to_string(), de_skeleton.clone());
        de_skeleton
    }

    /// Compute the total Hessian by summing over skeleton, overlap, and CP-SCF contributions.
    ///
    /// # Returns
    ///
    /// - `de_hess` : shape `[3, 3, natm, natm]`. The total Hessian.
    ///
    /// Cached on first call (stored in `result` as `de_tot`); repeated calls directly return the
    /// stored result.
    pub fn make_hess(&mut self) -> Tsr {
        if let Some(de_tot) = self.result.get("de_tot") {
            return de_tot.to_owned();
        }
        let t0 = std::time::Instant::now();
        let mo_coeff = self.mo_coeff.view();
        let mo_occ = self.mo_occ.view();
        let mo_energy = self.mo_energy.view();
        let dme0 = get_dme0_restricted(mo_coeff, mo_occ, mo_energy);
        let atm_list = self.cfg.atm_list.clone();

        let de_skeleton = self.make_skeleton_hess();

        let t1 = std::time::Instant::now();
        let de_ovlp = self.ovlp_obj.make_hess(dme0.view(), atm_list.as_deref());
        self.result.insert("de_ovlp".to_string(), de_ovlp.to_owned());
        self.timing.push(("de_ovlp".to_string(), t1.elapsed().as_secs_f64()));

        let t1 = std::time::Instant::now();
        let de_cpscf = self.make_cpscf_hess();
        self.result.insert("de_cpscf".to_string(), de_cpscf.to_owned());
        self.timing.push(("de_cpscf".to_string(), t1.elapsed().as_secs_f64()));

        let de_tot = de_skeleton + de_ovlp + de_cpscf;
        self.result.insert("de_tot".to_string(), de_tot.to_owned());
        self.timing.push(("de_tot".to_string(), t0.elapsed().as_secs_f64()));
        de_tot
    }
}
