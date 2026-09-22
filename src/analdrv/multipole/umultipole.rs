//! Multipole moment driver for unrestricted SCF methods.

use super::rmultipole::{multipole_intor, multipole_order_spec, quadrupole_to_traceless, sorted_multi_indices};
use crate::analdrv::prelude::*;

/// Working solver and maintainer of multipole moment evaluation for unrestricted SCF methods.
///
/// The unrestricted sibling of [`RMultipoleDH`](super::rmultipole::RMultipoleDH), at the SCF
/// level only: the moments are the nuclear plus SCF-density (α+β) contractions. The post-SCF
/// (fifth-DFA) density increments are not implemented for unrestricted methods and are rejected
/// at the interface, so this driver holds no gfock/response objects; the `DH` of the name only
/// mirrors the restricted driver, where the corresponding fields will slot in when the
/// unrestricted double-hybrid machinery exists. The orders 1-4 (dipole, quadrupole, octupole,
/// hexadecapole) share a single generic evaluation core, and the traceless quadrupole is stored
/// alongside the raw second moments.
///
/// # Structure
///
/// - `mol` : the molecule object, for the electronic multipole integrals.
/// - `mo_coeff` : shape `[nao, nmo_s]` per spin; `mo_occ` : shape `[nmo_s]` per spin. Converged
///   orbitals of the SCF-iteration functional (`nmo_α` and `nmo_β` may differ). Orbital
///   energies are not needed for property evaluation, and are not stored.
/// - `origin` : the origin of the multipole evaluation, fixed at construction (default `[0.0; 3]`).
/// - `nuc_obj` : the non-electronic (nuclear charge) contribution object.
/// - `result` : evaluated contributions, keyed `<prefix>_<part>` with prefixes `dip`/`quad`/
///   `oct`/`hex` and parts `nuc`/`scf`/`tot` (plus `quad_tot_traceless`).
/// - `timing` : wall-time information, keyed similarly to `result`.
pub struct UMultipoleDH {
    pub mol: CInt,
    pub mo_coeff: [Tsr; 2],
    pub mo_occ: [Tsr; 2],
    pub origin: [f64; 3],
    pub nuc_obj: MultipoleNucCharge,
    pub result: HashMap<String, Tsr>,
    /// Timing information. Represented by wall time in second.
    pub timing: Vec<(String, f64)>,
}

impl UMultipoleDH {
    /// Create the multipole driver.
    ///
    /// # Parameters
    ///
    /// - `mol` : [`CInt`]. The molecule object.
    /// - `mo_coeff` : shape `[nao, nmo_s]` per spin. Converged SCF molecular orbital coefficients.
    /// - `mo_occ` : shape `[nmo_s]` per spin. Converged SCF occupation numbers.
    /// - `origin` : the origin of the multipole evaluation.
    pub fn new(mol: &CInt, mo_coeff: [Tsr; 2], mo_occ: [Tsr; 2], origin: [f64; 3]) -> Self {
        let device = mo_coeff[0].device().clone();
        Self {
            mol: mol.clone(),
            mo_coeff,
            mo_occ,
            origin,
            nuc_obj: MultipoleNucCharge::new(mol, &device),
            result: HashMap::new(),
            timing: Vec::new(),
        }
    }

    /// Generic evaluation core of the raw Cartesian moment of the given order (1 = dipole, 2 =
    /// quadrupole, 3 = octupole, 4 = hexadecapole): the nuclear and the SCF-density (α+β)
    /// contributions, stored in `result`.
    ///
    /// Cached on first call per order (the origin and the orbitals are fixed at construction);
    /// repeated evaluation of the same order directly returns the stored total.
    fn make_multipole_core(&mut self, order: usize) -> Tsr {
        let [α, β] = [0, 1];
        let (intor, prefix) = multipole_order_spec(order);
        if let Some(tot) = self.result.get(&format!("{prefix}_tot")) {
            return tot.to_owned();
        }
        let device = self.mo_coeff[α].device().clone();

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

        // SCF density contribution (the total α+β density)
        let t0 = std::time::Instant::now();
        let dm0 = get_dm0_restricted(self.mo_coeff[α].view(), self.mo_occ[α].view())
            + get_dm0_restricted(self.mo_coeff[β].view(), self.mo_occ[β].view());
        let m_scf = -(&int * &dm0).sum_axes([0, 1]);
        self.timing.push((format!("{prefix}_scf"), t0.elapsed().as_secs_f64()));

        let m_tot = &m_nuc + &m_scf;
        self.result.insert(format!("{prefix}_nuc"), m_nuc);
        self.result.insert(format!("{prefix}_scf"), m_scf);

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

    /// Print the evaluated multipole moments in the analdrv output style.
    ///
    /// Only sections whose entries exist in `result` are printed (so a dipole-only evaluation
    /// does not print an empty quadrupole section). All quantities are in atomic units. The
    /// print is gated by `print_level >= 1` (the timing output stays at `>= 2`).
    pub fn print_multipole(&self, print_level: usize) {
        if print_level < 1 {
            return;
        }
        println!("=============== Multipole Moments (in analdrv) ===============");
        println!("Origin (a.u.): [{:14.8}, {:14.8}, {:14.8}]", self.origin[0], self.origin[1], self.origin[2]);

        if self.result.contains_key("dip_tot") {
            println!("Electric dipole moment (a.u.):");
            for (key, label) in [("dip_nuc", "nuclear charges"), ("dip_scf", "SCF density"), ("dip_tot", "total")] {
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
