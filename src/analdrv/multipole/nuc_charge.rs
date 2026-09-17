use crate::analdrv::prelude::*;

/// Raw Cartesian multipole moment of the nuclear charges.
///
/// # Parameters
///
/// - `mol` : [`CInt`]. The molecule object.
/// - `device` : [`DeviceBLAS`]. The device on which the returned tensor is allocated.
/// - `origin` : the origin with respect to which the moment is evaluated.
/// - `order` : the Cartesian order `l` (1 = dipole, 2 = quadrupole, 3 = octupole,
///   4 = hexadecapole).
///
/// # Returns
///
/// - `mom_nuc` : shape `[3] * order`. The raw nuclear charge contribution, in the column-major
///   component convention `comp = sum_k t_k 3^(order - 1 - k)` (first Cartesian index the
///   slowest), matching the reshape convention of [`multipole_intor`
///   ](crate::analdrv::multipole::rmultipole::multipole_intor).
///
/// # Notes
///
/// The charges are [`CInt::atom_charges`], which returns the ECP-effective nuclear charges
/// (`Z - n_ecp`) when ECP is enabled. This is deliberate, and consistent with an ECP valence-only
/// electron density; note that some other dipole helpers in REST (like the geometry-level
/// `evaluate_dipole_moment`) use the full element charges instead.
pub fn get_nuc_charge_multipole(mol: &CInt, device: &DeviceBLAS, origin: [f64; 3], order: usize) -> Tsr {
    assert!(order >= 1, "multipole order should be at least 1 (dipole).");
    let ncomp = 3usize.pow(order as u32);

    // qs: [natm]. The (ECP-effective) nuclear charges. rs: [natm] of [f64; 3] coordinates.
    let qs = mol.atom_charges();
    let rs = mol.atom_coords();

    let mut comps = vec![0.0f64; ncomp];
    for (&q, r) in qs.iter().zip(rs.iter()) {
        let dr = [r[0] - origin[0], r[1] - origin[1], r[2] - origin[2]];
        for c in 0..ncomp {
            // the moment is symmetric to the permutation of the Cartesian indices, so the
            // extraction order of the indices from the flat component is irrelevant
            let mut val = q;
            let mut rem = c;
            for _ in 0..order {
                val *= dr[rem % 3];
                rem /= 3;
            }
            comps[c] += val;
        }
    }
    rt::asarray((comps, vec![3usize; order], device))
}

/// Multipole contribution from nuclear charges.
pub struct MultipoleNucCharge {
    pub mol: CInt,
    pub device: DeviceBLAS,
}

impl MultipoleNucCharge {
    pub fn new(mol: &CInt, device: &DeviceBLAS) -> Self {
        Self { mol: mol.clone(), device: device.clone() }
    }
}

impl AnalDrvBaseAPI for MultipoleNucCharge {}

impl MultipoleNucAPI for MultipoleNucCharge {
    fn make_dipole_nuc(&mut self, origin: [f64; 3]) -> Tsr {
        get_nuc_charge_multipole(&self.mol, &self.device, origin, 1)
    }

    fn make_quadrupole_nuc(&mut self, origin: [f64; 3]) -> Tsr {
        get_nuc_charge_multipole(&self.mol, &self.device, origin, 2)
    }

    fn make_octupole_nuc(&mut self, origin: [f64; 3]) -> Tsr {
        get_nuc_charge_multipole(&self.mol, &self.device, origin, 3)
    }

    fn make_hexadecapole_nuc(&mut self, origin: [f64; 3]) -> Tsr {
        get_nuc_charge_multipole(&self.mol, &self.device, origin, 4)
    }
}
