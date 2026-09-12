//! Multipole moment driver for restricted SCF and double-hybrid (DH) methods.

use crate::analdrv::prelude::*;
use crate::analdrv::response::rgfock_interface::RGFockDH;
use crate::analdrv::response::trait_rgfock::RGFockAPI;

/// A wrapper around [`CInt::integrate`] for multipole integrals, evaluated at the given common
/// origin (with save/restore of the previous origin setting).
///
/// # Notes
///
/// The output is column-major. The flat multipole component axis (the last axis, of size
/// `3^l` for the order-`l` integrator `int1e_r`, `int1e_rr`, `int1e_rrr`, `int1e_rrrr`) is
/// reshaped to `l` trailing axes of 3, with the first Cartesian index the slowest:
/// `comp = sum_k t_k 3^(l - 1 - k)`.
pub fn multipole_intor(mol: &mut CInt, intor_name: &str, origin: [f64; 3], device: &DeviceBLAS) -> Tsr {
    let (out, shape) = mol.with_common_origin(origin, |mol| mol.integrate(intor_name, "s1", None).into());

    let tsr: Tsr = rt::asarray((out, shape.clone(), device));

    // resolve the multipole order from the flat component axis
    let nao = mol.nao();
    match intor_name {
        "int1e_r" => tsr.into_shape([nao, nao, 3]),
        "int1e_rr" => tsr.into_shape([nao, nao, 3, 3]),
        "int1e_rrr" => tsr.into_shape([nao, nao, 3, 3, 3]),
        "int1e_rrrr" => tsr.into_shape([nao, nao, 3, 3, 3, 3]),
        _ => panic!("unsupported integral type {intor_name}"),
    }
}

/// Convert a raw (second-moment) quadrupole tensor to its traceless form:
/// `Theta_traceless = 3/2 (Theta - Tr(Theta) I / 3)`.
///
/// # Parameters
///
/// - `quad` : shape `[3, 3]`. The raw quadrupole tensor.
///
/// # Returns
///
/// - `quad_traceless` : shape `[3, 3]`. The traceless quadrupole tensor.
pub fn quadrupole_to_traceless(quad: &Tsr) -> Tsr {
    assert_eq!(quad.shape(), &[3, 3], "quadrupole tensor should be of shape [3, 3].");
    let device = quad.device().clone();
    let tr = quad[[0, 0]] + quad[[1, 1]] + quad[[2, 2]];
    let mut eye: Tsr = rt::zeros(([3, 3].f(), &device));
    eye[[0, 0]] = 1.0;
    eye[[1, 1]] = 1.0;
    eye[[2, 2]] = 1.0;
    1.5 * (quad - &(eye * (tr / 3.0)))
}

/// Intor name and `result`/`timing` key prefix of a multipole order.
fn multipole_order_spec(order: usize) -> (&'static str, &'static str) {
    match order {
        1 => ("int1e_r", "dip"),
        2 => ("int1e_rr", "quad"),
        3 => ("int1e_rrr", "oct"),
        4 => ("int1e_rrrr", "hex"),
        _ => panic!("unsupported multipole order {order}; supported orders are 1-4."),
    }
}

/// Sorted (non-decreasing) Cartesian multi-indices enumerating the unique components of a
/// permutation-symmetric order-`order` moment tensor (6 for order 2, 10 for order 3, 15 for
/// order 4).
fn sorted_multi_indices(order: usize) -> Vec<Vec<usize>> {
    let mut out = vec![vec![]];
    for _ in 0..order {
        let mut next = Vec::new();
        for idx in &out {
            let start = idx.last().copied().unwrap_or(0);
            for t in start..3 {
                let mut i = idx.clone();
                i.push(t);
                next.push(i);
            }
        }
        out = next;
    }
    out
}

/// Working solver and maintainer of multipole moment evaluation for restricted SCF (and
/// double-hybrid, DH) methods.
///
/// The evaluation is fully incremental: every contribution is a contraction of some density
/// matrix with the multipole integral tensors, so the SCF and the DH levels share this single
/// driver; the DH objects (the [`RGFockDH`] composite, and the [`RRespSCF`] response object it
/// borrows for the Z-vector solve) are simply optional fields. The orders 1-4 (dipole,
/// quadrupole, octupole, hexadecapole) also share a single generic evaluation core.
///
/// The two lifetime parameters follow the `RHessSCF` convention: `'a` is the region of the SCF
/// data the objects borrow, `'b` the (typically shorter) borrows of the DH objects themselves.
///
/// # Structure
///
/// - `mol` : the molecule object, for the electronic multipole integrals.
/// - `mo_coeff` : shape `[nao, nmo]`; `mo_occ` : shape `[nmo]`. Converged orbitals of the
///   SCF-iteration functional. Orbital energies are not needed for property evaluation, and are not
///   stored.
/// - `origin` : the origin of the multipole evaluation, fixed at construction (default `[0.0; 3]`).
/// - `nuc_obj` : the non-electronic (nuclear charge) contribution object.
/// - `gfock` : optional DH composite. When present, the unrelaxed correlation rdm1 contribution is
///   evaluated; when `resp` is also present, the relaxed (Z-vector) contribution is evaluated as
///   well. `gfock` without `resp` gives the unrelaxed moments only.
/// - `result` : evaluated contributions, keyed `<prefix>_<part>` with prefixes `dip`/`quad`/
///   `oct`/`hex` and parts `nuc`/`scf`/`corr`/`resp`/`tot` (plus `quad_tot_traceless`).
/// - `timing` : wall-time information, keyed similarly to `result`.
pub struct RMultipoleDH<'a, 'b> {
    pub mol: CInt,
    pub mo_coeff: Tsr,
    pub mo_occ: Tsr,
    pub origin: [f64; 3],
    pub nuc_obj: MultipoleNucCharge,
    pub gfock: Option<&'b mut RGFockDH<'a>>,
    pub resp: Option<&'b mut RRespSCF<'a>>,
    pub result: HashMap<String, Tsr>,
    /// Timing information. Represented by wall time in second.
    pub timing: Vec<(String, f64)>,
}

impl<'a, 'b> RMultipoleDH<'a, 'b> {
    /// Create the multipole driver.
    ///
    /// # Parameters
    ///
    /// - `mol` : [`CInt`]. The molecule object.
    /// - `mo_coeff` : shape `[nao, nmo]`. Converged SCF molecular orbital coefficients.
    /// - `mo_occ` : shape `[nmo]`. Converged SCF occupation numbers.
    /// - `origin` : the origin of the multipole evaluation.
    /// - `gfock` : optional DH composite for the response density increments.
    /// - `resp` : optional response object, required for the relaxed (Z-vector) increments; only
    ///   used when `gfock` is also given.
    pub fn new(
        mol: &CInt,
        mo_coeff: Tsr,
        mo_occ: Tsr,
        origin: [f64; 3],
        gfock: Option<&'b mut RGFockDH<'a>>,
        resp: Option<&'b mut RRespSCF<'a>>,
    ) -> Self {
        let device = mo_coeff.device().clone();
        Self {
            mol: mol.clone(),
            mo_coeff,
            mo_occ,
            origin,
            nuc_obj: MultipoleNucCharge::new(mol, &device),
            gfock,
            resp,
            result: HashMap::new(),
            timing: Vec::new(),
        }
    }

    /// Generic evaluation core of the raw Cartesian moment of the given order (1 = dipole, 2 =
    /// quadrupole, 3 = octupole, 4 = hexadecapole): all contributions, stored in `result`.
    ///
    /// Cached on first call per order (the origin and the orbitals are fixed at construction);
    /// repeated evaluation of the same order directly returns the stored total.
    ///
    /// The per-order public methods are thin wrappers of this function; the only order-specific
    /// extras beyond the shared structure are the traceless quadrupole (stored under
    /// `quad_tot_traceless` for order 2).
    fn make_multipole_core(&mut self, order: usize) -> Tsr {
        let (intor, prefix) = multipole_order_spec(order);
        if let Some(tot) = self.result.get(&format!("{prefix}_tot")) {
            return tot.to_owned();
        }
        let device = self.mo_coeff.device().clone();

        // multipole integrals at the fixed origin: shape [nao, nao, 3, ..., 3] (`order` axes of 3)
        let t0 = std::time::Instant::now();
        let int = multipole_intor(&mut self.mol, intor, self.origin, &device);
        self.timing.push((format!("{prefix}_intor"), t0.elapsed().as_secs_f64()));

        // nuclear contribution
        let t0 = std::time::Instant::now();
        let m_nuc = match order {
            1 => self.nuc_obj.make_dipole_nuc(self.origin),
            2 => self.nuc_obj.make_quadrupole_nuc(self.origin),
            3 => self.nuc_obj.make_octupole_nuc(self.origin),
            4 => self.nuc_obj.make_hexadecapole_nuc(self.origin),
            _ => unreachable!("order range checked in multipole_order_spec"),
        };
        self.timing.push((format!("{prefix}_nuc"), t0.elapsed().as_secs_f64()));

        // SCF density contribution
        let t0 = std::time::Instant::now();
        let dm0 = get_dm0_restricted(self.mo_coeff.view(), self.mo_occ.view());
        let m_scf = -(&int * &dm0).sum_axes([0, 1]);
        self.timing.push((format!("{prefix}_scf"), t0.elapsed().as_secs_f64()));

        let mut m_tot = &m_nuc + &m_scf;
        self.result.insert(format!("{prefix}_nuc"), m_nuc);
        self.result.insert(format!("{prefix}_scf"), m_scf);

        // DH density increments
        if let Some(gfock) = self.gfock.as_deref_mut() {
            // unrelaxed correlation rdm1 contribution
            let t0 = std::time::Instant::now();
            let rdm1_corr = gfock.make_rdm1();
            let rdm1_corr_ao = self.mo_coeff.view() % rdm1_corr.view() % self.mo_coeff.view().t();
            let m_corr = -(&int * &rdm1_corr_ao).sum_axes([0, 1]);
            self.timing.push((format!("{prefix}_corr"), t0.elapsed().as_secs_f64()));

            m_tot = &m_tot + &m_corr;
            self.result.insert(format!("{prefix}_corr"), m_corr);

            // relaxed (Z-vector) contribution; requires the response object
            if let Some(resp) = self.resp.as_deref_mut() {
                let t0 = std::time::Instant::now();
                gfock.make_response_preparation(resp);
                let rdm1_resp = gfock.make_rdm1_resp(resp);
                // `rdm1_resp - rdm1_corr` is essentially the Z-vector on the vir-occ block, and
                // is deliberately not symmetrized; the (possible) antisymmetric part does not
                // contribute to traces against the symmetric multipole integrals.
                let dz_ao = self.mo_coeff.view() % (&rdm1_resp - &rdm1_corr).view() % self.mo_coeff.view().t();
                let m_resp = -(&int * &dz_ao).sum_axes([0, 1]);
                self.timing.push((format!("{prefix}_resp"), t0.elapsed().as_secs_f64()));

                m_tot = &m_tot + &m_resp;
                self.result.insert(format!("{prefix}_resp"), m_resp);
            }
        }

        if order == 2 {
            let m_tot_traceless = quadrupole_to_traceless(&m_tot);
            self.result.insert("quad_tot_traceless".to_string(), m_tot_traceless);
        }
        self.result.insert(format!("{prefix}_tot"), m_tot.clone());
        m_tot
    }

    /// Evaluate the dipole moment: all contributions, stored in `result`.
    ///
    /// # Returns
    ///
    /// - `dip_tot` : shape `[3]`. The total dipole moment, the sum of all contributions.
    pub fn make_dipole(&mut self) -> Tsr {
        self.make_multipole_core(1)
    }

    /// Evaluate the raw quadrupole moment: all contributions, stored in `result` (together with
    /// the traceless total under `quad_tot_traceless`).
    ///
    /// # Returns
    ///
    /// - `quad_tot` : shape `[3, 3]`. The total raw (second-moment) quadrupole moment.
    pub fn make_quadrupole(&mut self) -> Tsr {
        self.make_multipole_core(2)
    }

    /// Evaluate the raw octupole moment: all contributions, stored in `result`.
    ///
    /// # Returns
    ///
    /// - `oct_tot` : shape `[3, 3, 3]`. The total raw octupole moment.
    pub fn make_octupole(&mut self) -> Tsr {
        self.make_multipole_core(3)
    }

    /// Evaluate the raw hexadecapole moment: all contributions, stored in `result`.
    ///
    /// # Returns
    ///
    /// - `hex_tot` : shape `[3, 3, 3, 3]`. The total raw hexadecapole moment.
    pub fn make_hexadecapole(&mut self) -> Tsr {
        self.make_multipole_core(4)
    }

    /// Total density matrix in AO basis, shape `[nao, nao]`: the SCF density plus the
    /// correlation rdm1, and, for `relaxed = true`, the symmetrized Z-vector increment
    /// (`0.5 * (Z + Z^T)` on the vir-occ and occ-vir blocks). This is the density contracted
    /// for the multipole moments — its trace against any symmetric one-electron property
    /// integral reproduces the electronic moment — and the quantity dumped to the fchk file
    /// by `multipole_rdm1_dump`.
    ///
    /// Requires the DH composite (a post-SCF method); for the relaxed increment the response
    /// object must be present, and the moment evaluation must have run before (the Z-vector
    /// and the correlation rdm1 are then already cached). The (possible) antisymmetric part
    /// of the raw vir-occ increment does not contribute to traces against symmetric
    /// integrals, so the symmetrized and the raw forms are equivalent for the moments; the
    /// symmetric form is the canonical relaxed density and is used here.
    ///
    /// # Returns
    ///
    /// - `dm_total_ao` : shape `[nao, nao]`. The total density matrix in AO basis.
    pub fn get_total_density_ao(&mut self, relaxed: bool) -> Tsr {
        let dm0 = get_dm0_restricted(self.mo_coeff.view(), self.mo_occ.view());
        let Some(gfock) = self.gfock.as_deref_mut() else {
            panic!("The total density increment requires a post-SCF (PT2-family) method; for SCF-level methods the total density is the SCF density alone.")
        };
        let rdm1_corr = gfock.make_rdm1();
        let mut rdm1_total = rdm1_corr.to_owned();
        if relaxed {
            let resp = self.resp.as_deref_mut().expect(
                "Relaxed total density requires the response object (RRespSCF), which the relaxed multipole mode provides.",
            );
            let rdm1_resp = gfock.make_rdm1_resp(resp);
            // `rdm1_resp - rdm1_corr` is the Z-vector on the vir-occ block (see
            // `make_multipole_core`); symmetrize it to 0.5 * (Z + Z^T).
            let dz = &rdm1_resp - &rdm1_corr;
            rdm1_total = rdm1_total + 0.5 * (&dz + &dz.t());
        }
        dm0 + self.mo_coeff.view() % rdm1_total.view() % self.mo_coeff.view().t()
    }

    /// Print the evaluated multipole moments in the analdrv output style.
    ///
    /// Only sections whose entries exist in `result` are printed (so a dipole-only evaluation
    /// does not print an empty quadrupole section). All quantities are in atomic units. The
    /// print is gated by `print_level >= 1` (the timing output stays at `>= 2`); the output
    /// style of this section is expected to be refactored later.
    pub fn print_multipole(&self, print_level: usize) {
        if print_level < 1 {
            return;
        }
        println!("=============== Multipole Moments (in analdrv) ===============");
        println!("Origin (a.u.): [{:14.8}, {:14.8}, {:14.8}]", self.origin[0], self.origin[1], self.origin[2]);

        if self.result.contains_key("dip_tot") {
            println!("Electric dipole moment (a.u.):");
            for (key, label) in [
                ("dip_nuc", "nuclear charges"),
                ("dip_scf", "SCF density"),
                ("dip_corr", "unrelaxed corr. density"),
                ("dip_resp", "response (Z-vector)"),
                ("dip_tot", "total"),
            ] {
                if let Some(v) = self.result.get(key) {
                    println!("    {:<24}: {:16.12}", label, v);
                }
            }
            // total dipole additionally in Debye (1 a.u. = 2.54174623 Debye)
            if let Some(v) = self.result.get("dip_tot") {
                let au2debye = crate::constants::AU2DEBYE;
                println!(
                    "    {:<24}: {:16.12} {:16.12} {:16.12}",
                    "total, Debye (X Y Z)",
                    v[[0]] * au2debye,
                    v[[1]] * au2debye,
                    v[[2]] * au2debye
                );
            }
        }

        self.print_moment_section("Electric quadrupole moment, raw second moments (a.u.)", "quad_tot", 2);
        self.print_moment_section("Electric quadrupole moment, traceless (a.u.)", "quad_tot_traceless", 2);
        self.print_moment_section("Electric octupole moment (a.u.)", "oct_tot", 3);
        self.print_moment_section("Electric hexadecapole moment (a.u.)", "hex_tot", 4);
    }

    /// Print one moment section from the `result` entry `key`, listing the unique components of
    /// the permutation-symmetric order-`order` tensor. Does nothing if the key is absent.
    fn print_moment_section(&self, title: &str, key: &str, order: usize) {
        let Some(tot) = self.result.get(key) else { return };
        let flat = tot.clone().into_shape(-1);
        println!("{title}:");
        for idx in sorted_multi_indices(order) {
            let label: String = idx.iter().map(|&t| ['X', 'Y', 'Z'][t]).collect();
            let c = idx.iter().enumerate().fold(0usize, |acc, (k, &t)| acc + t * 3usize.pow((order - 1 - k) as u32));
            println!("    {label}: {:16.12}", flat[[c]]);
        }
    }
}
