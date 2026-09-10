//! Hessian implementations for unrestricted SCF.

use crate::analdrv::prelude::*;
/// Working solver and maintainer of all hessian components for unrestricted SCF method.
pub struct UHessSCF<'a> {
    pub mo_coeff: [Tsr; 2],
    pub mo_occ: [Tsr; 2],
    pub mo_energy: [Tsr; 2],
    pub ovlp_obj: &'a mut UHessOvlp,
    pub nuc_list: Vec<&'a mut dyn HessNucAPI>,
    pub core_list: Vec<&'a mut dyn UHessCoreAPI>,
    pub el_list: Vec<&'a mut dyn UHessElecInteractAPI>,
    pub config: AnalDrvConfig,
    pub result: HashMap<String, Tsr>,
    /// Timing information. Represented by wall time in second.
    pub timing: Vec<(String, f64)>,
}

impl<'a> UHessSCF<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mo_coeff: [Tsr; 2],
        mo_occ: [Tsr; 2],
        mo_energy: [Tsr; 2],
        ovlp_obj: &'a mut UHessOvlp,
        nuc_list: Vec<&'a mut dyn HessNucAPI>,
        core_list: Vec<&'a mut dyn UHessCoreAPI>,
        el_list: Vec<&'a mut dyn UHessElecInteractAPI>,
        config: &AnalDrvConfig,
    ) -> Self {
        Self {
            mo_coeff,
            mo_occ,
            mo_energy,
            ovlp_obj,
            nuc_list,
            core_list,
            el_list,
            config: config.clone(),
            result: HashMap::new(),
            timing: Vec::new(),
        }
    }

    /// Number of atoms over which the Hessian is computed. This is `atm_list.len()` if
    /// `atm_list` is `Some`, otherwise the total number of atoms in the molecule.
    pub fn natm(&self) -> usize {
        match &self.config.nucgrad.atm_list {
            Some(list) => list.len(),
            None => self.ovlp_obj.natm(),
        }
    }

    /// Return the list of (global) atom indices the Hessian is computed for, ordered the same
    /// way as the local indexing used in the returned Hessian.
    pub fn atm_indices(&self) -> Vec<usize> {
        match &self.config.nucgrad.atm_list {
            Some(list) => list.clone(),
            None => (0..self.ovlp_obj.natm()).collect(),
        }
    }

    /// Compute the dimensionless CP-SCF right-hand side, along with necessary intermediates for later
    /// steps.
    ///
    /// # Returns
    ///     
    /// A dictionary containing:
    /// - `rhs : shape `[nmo, nocc_α, 3, natm]` and `[nmo, nocc_β, 3, natm]`. The dimensionless CP-SCF
    ///   right-hand side.
    /// - `f1mo` : shape `[nmo, nocc_α, 3, natm]` and `[nmo, nocc_β, 3, natm]`. The first-order
    ///   derivative of the Fock matrix in MO basis.
    /// - `s1mo` : shape `[nmo, nocc_α, 3, natm]` and `[nmo, nocc_β, 3, natm]`. The first-order
    ///   derivative of the overlap matrix in MO basis.
    pub fn compute_dimless_cpscf_rhs(&mut self) -> HashMap<&'static str, Tsr> {
        // setups
        let t0 = std::time::Instant::now();
        let [α, β] = [0, 1];
        let mo_coeff = [self.mo_coeff[α].view(), self.mo_coeff[β].view()];
        let mo_occ = [self.mo_occ[α].view(), self.mo_occ[β].view()];
        let mo_energy = [self.mo_energy[α].view(), self.mo_energy[β].view()];
        let level_shift = self.config.resp.level_shift;
        let device = mo_coeff[α].device().clone();

        let nao = mo_coeff[α].shape()[0];
        let nmo = [mo_coeff[α].shape()[1], mo_coeff[β].shape()[1]];
        let occidx = [mo_occ[α].view().greater(0).into_vec(), mo_occ[β].view().greater(0).into_vec()];
        let viridx = [occidx[α].iter().map(|&x| !x).collect_vec(), occidx[β].iter().map(|&x| !x).collect_vec()];
        let mocc = [mo_coeff[α].bool_select(-1, &occidx[α]), mo_coeff[β].bool_select(-1, &occidx[β])];
        let eocc = [mo_energy[α].bool_select(-1, &occidx[α]), mo_energy[β].bool_select(-1, &occidx[β])];
        let evir = [mo_energy[α].bool_select(-1, &viridx[α]), mo_energy[β].bool_select(-1, &viridx[β])];
        let nocc = [mocc[α].shape()[1], mocc[β].shape()[1]];
        let natm = self.natm();
        let atm_indices = self.atm_indices();
        let atm_list = self.config.nucgrad.atm_list.as_deref();

        let e_ai = [evir[α].i((.., None)) - eocc[α].i((None, ..)), evir[β].i((.., None)) - eocc[β].i((None, ..))];
        let e_ai_shift = [&e_ai[α] + level_shift, &e_ai[β] + level_shift];

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
        let mut f1bra_el: [Tsr; 2] =
            [rt::zeros(([nao, nocc[α], 3, natm], &device)), rt::zeros(([nao, nocc[β], 3, natm], &device))];
        for el_obj in self.el_list.iter_mut() {
            let t1 = std::time::Instant::now();
            let bra = el_obj.get_deriv1_bra(&mo_coeff, &mo_occ, atm_list);
            f1bra_el[α] += &bra[α];
            f1bra_el[β] += &bra[β];
            self.timing.push((
                format!("in compute_dimless_cpscf_rhs, f1bra_el_{}", el_obj.get_type_name()),
                t1.elapsed().as_secs_f64(),
            ));
        }

        // construct whole f1mo
        let f1mo_α = mo_coeff[α].t() % (&f1ao_core % &mocc[α] + &f1bra_el[α]);
        let f1mo_β = mo_coeff[β].t() % (&f1ao_core % &mocc[β] + &f1bra_el[β]);

        // --- s1mo --- //

        let t1 = std::time::Instant::now();

        let mut gen_ovlp_deriv1 = self.ovlp_obj.generator_deriv1();
        let mut s1ao: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
        for (A_loc, &A_glob) in atm_indices.iter().enumerate() {
            *&mut s1ao.i_mut((Ellipsis, A_loc)) += gen_ovlp_deriv1(A_glob);
        }
        let s1mo_α = mo_coeff[α].t() % (&s1ao % &mocc[α]);
        let s1mo_β = mo_coeff[β].t() % (&s1ao % &mocc[β]);

        self.timing.push(("in compute_dimless_cpscf_rhs, s1mo".to_string(), t1.elapsed().as_secs_f64()));

        // --- dimensionless rhs --- //

        let so = [rt::slice!(0, nocc[α]), rt::slice!(0, nocc[β])];
        let sv = [rt::slice!(nocc[α], nmo[α]), rt::slice!(nocc[β], nmo[β])];
        let b1mo_α = &f1mo_α - &s1mo_α * eocc[α].i((None, ..));
        let b1mo_β = &f1mo_β - &s1mo_β * eocc[β].i((None, ..));
        let mut rhs_α = rt::zeros(([nmo[α], nocc[α], 3, natm], &device));
        let mut rhs_β = rt::zeros(([nmo[β], nocc[β], 3, natm], &device));
        *&mut rhs_α.i_mut(sv[α]) += -b1mo_α.i(sv[α]) / &e_ai_shift[α];
        *&mut rhs_β.i_mut(sv[β]) += -b1mo_β.i(sv[β]) / &e_ai_shift[β];
        *&mut rhs_α.i_mut(so[α]) += -0.5 * s1mo_α.i(so[α]);
        *&mut rhs_β.i_mut(so[β]) += -0.5 * s1mo_β.i(so[β]);

        self.timing.push(("compute_dimless_cpscf_rhs".to_string(), t0.elapsed().as_secs_f64()));
        HashMap::from([
            ("f1mo_0", f1mo_α),
            ("f1mo_1", f1mo_β),
            ("s1mo_0", s1mo_α),
            ("s1mo_1", s1mo_β),
            ("rhs_0", rhs_α),
            ("rhs_1", rhs_β),
        ])
    }

    /// Prepare the response for CP-SCF calculation.
    ///
    /// This involves all electron-interaction objects.
    pub fn make_response_preparation(&mut self) {
        let t0 = std::time::Instant::now();
        let mo_coeff = [self.mo_coeff[0].view(), self.mo_coeff[1].view()];
        let mo_occ = [self.mo_occ[0].view(), self.mo_occ[1].view()];
        for el_obj in self.el_list.iter_mut() {
            let t1 = std::time::Instant::now();
            el_obj.make_response_preparation(&mo_coeff, &mo_occ);
            self.timing.push((
                format!("in make_response_preparation, {}", el_obj.get_type_name()),
                t1.elapsed().as_secs_f64(),
            ));
        }
        self.timing.push(("make_response_preparation".to_string(), t0.elapsed().as_secs_f64()));
    }

    /// Compute the response of the system to a given perturbation in MO space (mo1), which is
    /// needed for CP-SCF.
    ///
    /// # Parameters
    ///
    /// - `mo1` : shape `[nmo, nocc_α, ...]` and `[nmo, nocc_β, ...]`. The perturbation in MO space.
    ///
    /// # Returns
    ///
    /// - `resp` : shape `[nmo, nocc_α, ...]` and `[nmo, nocc_β, ...]`. The response in MO space.
    pub fn response_mo(&mut self, mo1: &[TsrView; 2]) -> [Tsr; 2] {
        let [α, β] = [0, 1];
        let ubra_α = &self.mo_coeff[α] % &mo1[α];
        let ubra_β = &self.mo_coeff[β] % &mo1[β];
        let mut resp_α = rt::zeros_like(&ubra_α);
        let mut resp_β = rt::zeros_like(&ubra_β);

        for el_obj in self.el_list.iter_mut() {
            let t1 = std::time::Instant::now();
            let el_resp = el_obj.get_response_bra(&[ubra_α.view(), ubra_β.view()]);
            resp_α += self.mo_coeff[α].t() % &el_resp[α];
            resp_β += self.mo_coeff[β].t() % &el_resp[β];
            self.timing.push((format!("in response_mo, {}", el_obj.get_type_name()), t1.elapsed().as_secs_f64()));
        }
        [resp_α, resp_β]
    }

    /// Compute the dimensionless response for CP-SCF calculation.
    ///
    /// # Parameters
    ///
    /// - `mo1` : shape `[nmo, nocc_α, ...]` and `[nmo, nocc_β, ...]`. The perturbation in MO space.
    ///
    /// # Returns
    ///
    /// - `resp` : shape `[nmo, nocc_α, ...]` and `[nmo, nocc_β, ...]`. The dimensionless response
    ///   in MO space.
    pub fn response_dimless_cpscf(&mut self, mo1: &[TsrView; 2]) -> [Tsr; 2] {
        let t0 = std::time::Instant::now();
        let [α, β] = [0, 1];
        let mo_occ = [self.mo_occ[α].view(), self.mo_occ[β].view()];
        let occidx = [mo_occ[α].view().greater(0).into_vec(), mo_occ[β].view().greater(0).into_vec()];
        let viridx = [occidx[α].iter().map(|&x| !x).collect_vec(), occidx[β].iter().map(|&x| !x).collect_vec()];
        let nocc = [occidx[α].iter().filter(|&&x| x).count(), occidx[β].iter().filter(|&&x| x).count()];
        let nmo = [mo_occ[α].shape()[0], mo_occ[β].shape()[0]];
        let eocc = [
            self.mo_energy[α].view().bool_select(-1, &occidx[α]),
            self.mo_energy[β].view().bool_select(-1, &occidx[β]),
        ];
        let evir = [
            self.mo_energy[α].view().bool_select(-1, &viridx[α]),
            self.mo_energy[β].view().bool_select(-1, &viridx[β]),
        ];
        let e_ai = [evir[α].i((.., None)) - eocc[α].i((None, ..)), evir[β].i((.., None)) - eocc[β].i((None, ..))];
        let level_shift = self.config.resp.level_shift;
        let e_ai_shift = [&e_ai[0] + level_shift, &e_ai[1] + level_shift];
        let so = [rt::slice!(0, nocc[α]), rt::slice!(0, nocc[β])];
        let sv = [rt::slice!(nocc[α], nmo[α]), rt::slice!(nocc[β], nmo[β])];

        let mut resp = self.response_mo(mo1);

        // handle dimension less denominator and occupied response part
        if level_shift != 0.0 {
            *&mut resp[α] -= level_shift * &mo1[α];
            *&mut resp[β] -= level_shift * &mo1[β];
        }
        *&mut resp[α].i_mut(sv[α]) /= &e_ai_shift[α];
        *&mut resp[β].i_mut(sv[β]) /= &e_ai_shift[β];
        resp[α].i_mut(so[α]).fill(0.0);
        resp[β].i_mut(so[β]).fill(0.0);
        self.timing.push(("response_dimless_cpscf".to_string(), t0.elapsed().as_secs_f64()));
        resp
    }

    /// Solve the dimensionless CP-SCF equation using a Krylov solver.
    ///
    /// # Parameters
    ///
    /// - `rhs` : shape `[nmo, nocc_α, ...]` and `[nmo, nocc_β, ...]`. Dimensionless right-hand
    ///   side.
    ///
    /// # Returns
    ///
    /// - `mo1` : shape `[nmo, nocc_α, ...]` and `[nmo, nocc_β, ...]`. Perturbation in MO space that
    ///   solves the dimensionless CP-SCF equation.
    pub fn solve_dimless_cpscf(&mut self, rhs: &[TsrView; 2]) -> [Tsr; 2] {
        let t0 = std::time::Instant::now();
        let [α, β] = [0, 1];
        let rhs_shape = [rhs[α].shape().to_vec(), rhs[β].shape().to_vec()];
        let nmo = [rhs[α].shape()[0], rhs[β].shape()[0]];
        let nocc = [rhs[α].shape()[1], rhs[β].shape()[1]];
        let rhs = [rhs[α].reshape((nmo[α], nocc[α], -1)), rhs[β].reshape((nmo[β], nocc[β], -1))];
        let device = rhs[α].device().clone();

        let tol = self.config.resp.tol;
        let max_cycle = self.config.resp.max_cycle;
        let max_space = self.config.resp.max_space;
        let lindep = self.config.resp.lindep;
        let tol_inflation = self.config.resp.tol_inflation;

        let pack_flattened = |x: &[TsrView; 2]| -> Tsr {
            // original: [nmo_α, nocc_α, nprop] and [nmo_β, nocc_β, nprop]
            // target: [nmo_α * nocc_α + nmo_β * nocc_β, nprop]
            assert_eq!(x[α].ndim(), 3, "Expected x[α] to have shape [nmo_α, nocc_α, nprop]");
            assert_eq!(x[β].ndim(), 3, "Expected x[β] to have shape [nmo_β, nocc_β, nprop]");
            let nprop = x[α].shape()[2];
            let mut x_flattened = rt::zeros(([nmo[α] * nocc[α] + nmo[β] * nocc[β], nprop], &device));
            for A in 0..nprop {
                x_flattened.i_mut((..nmo[α] * nocc[α], A)).assign(x[α].i((.., .., A)).reshape(-1));
                x_flattened.i_mut((nmo[α] * nocc[α].., A)).assign(x[β].i((.., .., A)).reshape(-1));
            }
            x_flattened
        };

        let unpack_flattened = |x: TsrView| -> [Tsr; 2] {
            // original: [nmo_α * nocc_α + nmo_β * nocc_β, nprop]
            // target: [nmo_α, nocc_α, nprop] and [nmo_β, nocc_β, nprop]
            assert_eq!(x.ndim(), 2, "Expected x to have shape [nmo_α * nocc_α + nmo_β * nocc_β, nprop]");
            let nprop = x.shape()[1];
            let idx_split = nmo[α] * nocc[α];
            let mut x_α = rt::zeros(([nmo[α], nocc[α], nprop], &device));
            let mut x_β = rt::zeros(([nmo[β], nocc[β], nprop], &device));
            for A in 0..nprop {
                x_α.i_mut((.., .., A)).assign(x.i((..idx_split, A)).reshape((nmo[α], nocc[α])));
                x_β.i_mut((.., .., A)).assign(x.i((idx_split.., A)).reshape((nmo[β], nocc[β])));
            }
            [x_α, x_β]
        };

        let response_cpscf_flattened = |x: TsrView| -> Tsr {
            // split x by spin and reshape to original shape
            let [x_α, x_β] = unpack_flattened(x);
            // compute response by usual means
            let resp = self.response_dimless_cpscf(&[x_α.view(), x_β.view()]);
            // flatten resp to shape (nmo*nocc, nprop)
            let resp_view = resp.iter().map(|r| r.view()).collect_array().unwrap();
            pack_flattened(&resp_view)
        };

        let rhs_view = rhs.iter().map(|r| r.view()).collect_array().unwrap();
        let rhs_packed = pack_flattened(&rhs_view);
        let mo1_flattened = krylov_block(
            response_cpscf_flattened,
            rhs_packed.view(),
            None,
            tol,
            max_cycle,
            max_space,
            lindep,
            tol_inflation,
        );
        let [mo1_α, mo1_β] = unpack_flattened(mo1_flattened.view());
        let mo1_α = mo1_α.into_shape(rhs_shape[α].to_vec());
        let mo1_β = mo1_β.into_shape(rhs_shape[β].to_vec());

        self.timing.push(("solve_dimless_cpscf".to_string(), t0.elapsed().as_secs_f64()));
        [mo1_α, mo1_β]
    }

    /// Finalize the CP-SCF calculation by computing necessary intermediates for Hessian assembly.
    ///
    ///
    /// # Parameters
    ///
    /// - `f1mo` : shape `[nmo_α, nocc_α, 3, natm]` and `[nmo_β, nocc_β, 3, natm]`. The first-order
    ///   derivative of the Fock matrix in MO basis, obtained from
    ///   [`Self::compute_dimless_cpscf_rhs`].
    /// - `s1mo` : shape `[nmo_α, nocc_α, 3, natm]` and `[nmo_β, nocc_β, 3, natm]`. The first-order
    ///   derivative of the overlap matrix in MO basis, obtained from
    ///   [`Self::compute_dimless_cpscf_rhs`].
    /// - `mo1` : shape `[nmo_α, nocc_α, 3, natm]` and `[nmo_β, nocc_β, 3, natm]`. The perturbation
    ///   in MO space obtained from Krylov solver.
    ///
    /// # Returns
    ///
    /// `HashMap<&str, Tsr>`
    ///
    /// - `mo1_0`, `mo1_1` : shape `[nmo_α, nocc_α, 3, natm]` and `[nmo_β, nocc_β, 3, natm]`. The
    ///   finalized perturbation in MO space.
    /// - `mo_e1_0`, `mo_e1_1` : shape `[nocc_α, nocc_α, 3, natm]` and `[nocc_β, nocc_β, 3, natm]`.
    ///   The derivative of occupied orbital energies (Fock matrix) with respect to perturbation.
    pub fn finalize_cpscf(
        &mut self,
        f1mo: &[TsrView; 2],
        s1mo: &[TsrView; 2],
        mo1: &[TsrView; 2],
    ) -> HashMap<&'static str, Tsr> {
        let t0 = std::time::Instant::now();
        let [α, β] = [0, 1];
        let mo_occ = [self.mo_occ[α].view(), self.mo_occ[β].view()];
        let occidx = [mo_occ[α].view().greater(0).into_vec(), mo_occ[β].view().greater(0).into_vec()];
        let viridx = [occidx[α].iter().map(|&x| !x).collect_vec(), occidx[β].iter().map(|&x| !x).collect_vec()];
        let nocc = [occidx[α].iter().filter(|&&x| x).count(), occidx[β].iter().filter(|&&x| x).count()];
        let nmo = [mo_occ[α].shape()[0], mo_occ[β].shape()[0]];
        let eocc = [
            self.mo_energy[α].view().bool_select(-1, &occidx[α]),
            self.mo_energy[β].view().bool_select(-1, &occidx[β]),
        ];
        let evir = [
            self.mo_energy[α].view().bool_select(-1, &viridx[α]),
            self.mo_energy[β].view().bool_select(-1, &viridx[β]),
        ];
        let so = [rt::slice!(0, nocc[α]), rt::slice!(0, nocc[β])];
        let sv = [rt::slice!(nocc[α], nmo[α]), rt::slice!(nocc[β], nmo[β])];
        let e_ai = [evir[α].i((.., None)) - eocc[α].i((None, ..)), evir[β].i((.., None)) - eocc[β].i((None, ..))];
        let e_ij = [eocc[α].i((.., None)) - eocc[α].i((None, ..)), eocc[β].i((.., None)) - eocc[β].i((None, ..))];

        // last-iter the cp-hf equation, and remove the level-shift
        let last_resp = self.response_mo(mo1);
        let b1mo_α = &f1mo[α] - &s1mo[α] * eocc[α].i((None, ..)) + &last_resp[α];
        let b1mo_β = &f1mo[β] - &s1mo[β] * eocc[β].i((None, ..)) + &last_resp[β];
        let mut mo1_α = mo1[α].to_owned();
        let mut mo1_β = mo1[β].to_owned();
        mo1_α.i_mut(sv[α]).assign(-b1mo_α.i(sv[α]) / &e_ai[α]);
        mo1_β.i_mut(sv[β]).assign(-b1mo_β.i(sv[β]) / &e_ai[β]);

        // get the derivative of fock matrix in occ-occ block (derivative of orbital energy with rotation)
        let mo_e1_α = b1mo_α.i(so[α]) + &mo1_α.i(so[α]) * &e_ij[α];
        let mo_e1_β = b1mo_β.i(so[β]) + &mo1_β.i(so[β]) * &e_ij[β];

        self.timing.push(("finalize_cpscf".to_string(), t0.elapsed().as_secs_f64()));
        HashMap::from([("mo1_0", mo1_α), ("mo1_1", mo1_β), ("mo_e1_0", mo_e1_α), ("mo_e1_1", mo_e1_β)])
    }

    pub fn get_cpscf_hess(
        &self,
        f1mo: &[TsrView; 2],
        s1mo: &[TsrView; 2],
        mo1: &[TsrView; 2],
        mo_e1: &[TsrView; 2],
    ) -> Tsr {
        let [α, β] = [0, 1];
        let natm = self.natm();
        let mo_occ = [self.mo_occ[α].view(), self.mo_occ[β].view()];
        let occidx = [mo_occ[α].view().greater(0).into_vec(), mo_occ[β].view().greater(0).into_vec()];
        let nocc = [occidx[α].iter().filter(|&&x| x).count(), occidx[β].iter().filter(|&&x| x).count()];
        let eocc = [
            self.mo_energy[α].view().bool_select(-1, &occidx[α]),
            self.mo_energy[β].view().bool_select(-1, &occidx[β]),
        ];
        let so = [rt::slice!(0, nocc[α]), rt::slice!(0, nocc[β])];
        let s1oo = [s1mo[α].i(so[α]), s1mo[β].i(so[β])];
        let device = f1mo[α].device().clone();

        let mut de_cpscf = rt::zeros(([3, 3, natm, natm], &device));
        for A in 0..natm {
            for B in 0..=A {
                let mut de_BA = de_cpscf.i_mut((.., .., B, A));
                for σ in [α, β] {
                    de_BA += 2 * (f1mo[σ].i((.., .., None, .., A)) * mo1[σ].i((.., .., .., None, B))).sum_axes([0, 1]);
                    de_BA -= 2
                        * (s1mo[σ].i((.., .., None, .., A)) * mo1[σ].i((.., .., .., None, B)) * eocc[σ].i((None, ..)))
                            .sum_axes([0, 1]);
                    de_BA -= (s1oo[σ].i((.., .., None, .., A)) * mo_e1[σ].i((.., .., .., None, B))).sum_axes([0, 1]);
                }
            }
            for B in 0..A {
                let de_to_copy = de_cpscf.i((.., .., B, A)).t().to_owned();
                *&mut de_cpscf.i_mut((.., .., A, B)) += de_to_copy;
            }
        }
        de_cpscf
    }

    pub fn make_cpscf_hess(&mut self) -> Tsr {
        let pre_cpscf_dict = self.compute_dimless_cpscf_rhs();
        let f1mo = [pre_cpscf_dict.get("f1mo_0").unwrap().view(), pre_cpscf_dict.get("f1mo_1").unwrap().view()];
        let s1mo = [pre_cpscf_dict.get("s1mo_0").unwrap().view(), pre_cpscf_dict.get("s1mo_1").unwrap().view()];
        let rhs = [pre_cpscf_dict.get("rhs_0").unwrap().view(), pre_cpscf_dict.get("rhs_1").unwrap().view()];

        self.make_response_preparation();
        let mo1 = self.solve_dimless_cpscf(&rhs);
        let mo1_view = [mo1[0].view(), mo1[1].view()];
        let finalize_dict = self.finalize_cpscf(&f1mo, &s1mo, &mo1_view);
        let mo1 = [finalize_dict.get("mo1_0").unwrap().view(), finalize_dict.get("mo1_1").unwrap().view()];
        let mo_e1 = [finalize_dict.get("mo_e1_0").unwrap().view(), finalize_dict.get("mo_e1_1").unwrap().view()];

        self.get_cpscf_hess(&f1mo, &s1mo, &mo1, &mo_e1)
    }

    /// Compute the total skeleton contribution to the Hessian.
    ///
    /// **Total** means that we sum over all skeleton contributions from both core and
    /// electron-interaction objects.
    ///
    /// # Returns
    ///
    /// - `de_skeleton` : shape `[3, 3, natm, natm]`. The total skeleton contribution to the
    ///   Hessian.
    pub fn make_skeleton_hess(&mut self) -> Tsr {
        let [α, β] = [0, 1];
        let natm = self.natm();
        let mo_coeff = [self.mo_coeff[α].view(), self.mo_coeff[β].view()];
        let mo_occ = [self.mo_occ[α].view(), self.mo_occ[β].view()];
        let atm_list = self.config.nucgrad.atm_list.as_deref();

        let device = self.mo_coeff[α].device().clone();
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
            let de_core = core_obj.make_skeleton_hess(&mo_coeff, &mo_occ, atm_list);
            let core_obj_name = core_obj.get_type_name();
            self.result.insert(format!("de_skeleton_{}", core_obj_name), de_core.to_owned());
            self.timing.push((format!("de_skeleton_{}", core_obj_name,), t0.elapsed().as_secs_f64()));
            de_skeleton += de_core;
        }
        for el_obj in self.el_list.iter_mut() {
            let t0 = std::time::Instant::now();
            let de_el = el_obj.make_skeleton_hess(&mo_coeff, &mo_occ, atm_list);
            let el_obj_name = el_obj.get_type_name();
            self.result.insert(format!("de_skeleton_{}", el_obj_name), de_el.to_owned());
            self.timing.push((format!("de_skeleton_{}", el_obj_name,), t0.elapsed().as_secs_f64()));
            de_skeleton += de_el;
        }
        de_skeleton
    }

    /// Compute the total Hessian by summing over skeleton, overlap, and CP-SCF contributions.
    ///
    /// # Returns
    ///
    /// - `de_hess` : shape `[3, 3, natm, natm]`. The total Hessian.
    pub fn make_hess(&mut self) -> Tsr {
        let t0 = std::time::Instant::now();
        let [α, β] = [0, 1];
        let dme0 = [
            get_dme0_restricted(self.mo_coeff[α].view(), self.mo_occ[α].view(), self.mo_energy[α].view()),
            get_dme0_restricted(self.mo_coeff[β].view(), self.mo_occ[β].view(), self.mo_energy[β].view()),
        ];
        let atm_list = self.config.nucgrad.atm_list.clone();

        let de_skeleton = self.make_skeleton_hess();

        let t1 = std::time::Instant::now();
        let de_ovlp = self.ovlp_obj.make_hess([dme0[α].view(), dme0[β].view()], atm_list.as_deref());
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
