use crate::analdrv::prelude::*;

/// Abstract class for multipole-related API for non-electronic contributions.
///
/// # Term Explanation
///
/// **Nuc component** here actually means the term is of zero-order with right of (electron)
/// density matrix. For multipole moments this is the contribution of the classical nuclear
/// charges; in the future it may also cover other classical charges (like point charges of an
/// external field source).
///
/// This contribution is not related to density, so no RHF/UHF/GHF distinguishment required.
///
/// Higher multipole orders beyond the hexadecapole can be added later as further methods of this
/// trait; the integral binding already exposes the required integrators.
pub trait MultipoleNucAPI: AnalDrvBaseAPI {
    /// Generate the nuclear (non-electronic) contribution to the dipole moment.
    ///
    /// # Parameters
    ///
    /// - `origin` : the origin with respect to which the moment is evaluated. The dipole moment
    ///   of a charge-neutral molecule is origin-independent; the explicit origin allows this to
    ///   be verified numerically.
    ///
    /// # Returns
    ///
    /// - `dip_nuc` : shape `[3]`. The nuclear contribution to the dipole moment.
    fn make_dipole_nuc(&mut self, origin: [f64; 3]) -> Tsr;

    /// Generate the nuclear (non-electronic) contribution to the raw quadrupole moment.
    ///
    /// # Parameters
    ///
    /// - `origin` : the origin with respect to which the moment is evaluated. Unlike the dipole
    ///   moment, the quadrupole moment is origin-dependent in general; see the module
    ///   documentation for the shift formula.
    ///
    /// # Returns
    ///
    /// - `quad_nuc` : shape `[3, 3]`. The raw (second-moment) nuclear contribution; convert to
    ///   the traceless form by
    ///   [`quadrupole_to_traceless`](crate::analdrv::multipole::rmultipole::quadrupole_to_traceless).
    fn make_quadrupole_nuc(&mut self, origin: [f64; 3]) -> Tsr;

    /// Generate the nuclear (non-electronic) contribution to the raw octupole moment.
    ///
    /// # Returns
    ///
    /// - `oct_nuc` : shape `[3, 3, 3]`. The raw octupole moment of the nuclear charges, in the
    ///   column-major component convention `comp = 9 t1 + 3 t2 + t3` (first Cartesian index the
    ///   slowest).
    fn make_octupole_nuc(&mut self, origin: [f64; 3]) -> Tsr;

    /// Generate the nuclear (non-electronic) contribution to the raw hexadecapole moment.
    ///
    /// # Returns
    ///
    /// - `hex_nuc` : shape `[3, 3, 3, 3]`. The raw hexadecapole moment of the nuclear charges, in
    ///   the column-major component convention `comp = 27 t1 + 9 t2 + 3 t3 + t4` (first Cartesian
    ///   index the slowest).
    fn make_hexadecapole_nuc(&mut self, origin: [f64; 3]) -> Tsr;
}
