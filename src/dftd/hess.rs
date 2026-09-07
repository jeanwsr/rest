use crate::analdrv::hessian::trait_rhess::HessNucAPI;
use crate::analdrv::hessian::trait_util::HessUtilAPI;
use crate::dftd::energy::DispSpec;
use crate::molecule_io::Molecule;
use crate::utilities::rstsr_util::Tsr;
use itertools::Itertools;
use rstsr::prelude::*;

/// Hessian contribution from empirical dispersion correction (DFTD3/DFTD4).
///
/// The dispersion energy does not depend on the (electron) density matrix, so it is a
/// nuclear-like term (see [`HessNucAPI`]); no SCF or CP-HF response is involved. The dftd
/// libraries provide no analytical Hessian, so the Hessian is evaluated numerically by
/// central finite differences of the analytic dispersion gradient. Only this contribution
/// is numerical; the other Hessian contributions (JK, DFT, ...) remain analytical.
///
/// The object holds its own copy of the molecule; the finite-difference displacements are
/// applied to (and restored on) this copy, leaving the input molecule untouched.
pub struct HessDFTD {
    pub mol: Molecule,
    /// Finite-difference step in Bohr.
    pub step: f64,
    spec: DispSpec,
}

impl HessDFTD {
    /// Create the object. Returns `None` if no empirical dispersion is specified.
    pub fn new(mol: &Molecule, step: f64) -> Option<Self> {
        let spec = DispSpec::resolve(mol)?;
        Some(Self { mol: mol.clone(), step, spec })
    }

    /// The dispersion gradient at the geometry currently held in `self.mol`.
    fn disp_gradient(&self) -> Vec<f64> {
        let (_, grad, _) = self.spec.evaluate(&self.mol);
        // the evaluation always requests the gradient from the dftd libraries
        grad.expect("The dftd library does not return the dispersion gradient.")
    }
}

impl HessUtilAPI for HessDFTD {}

impl HessNucAPI for HessDFTD {
    fn make_skeleton_hess(&mut self, atm_list: Option<&[usize]>) -> Tsr {
        // Note `natm_orig` is the number of atoms of the original molecule; the returned
        // Hessian is of the selected atoms only (`natm = atm_list.len()`).
        let natm_orig = self.mol.geom.elem.len();
        let atm_list = atm_list.map(|v| v.to_vec()).unwrap_or_else(|| (0..natm_orig).collect_vec());
        let natm = atm_list.len();

        let device = DeviceBLAS::default();
        let mut hess: Tsr = rt::zeros(([3, 3, natm, natm], &device));
        let step = self.step;

        // Central finite differences of the dispersion gradient: displace atom A along
        // direction t by ±step, and difference the gradient at all selected atoms.
        for (iA, &A) in atm_list.iter().enumerate() {
            for t in 0..3 {
                let mut disp = vec![0.0; 3];
                disp[t] = step;
                self.mol.geom.geom_shift(A, disp.clone());
                let grad_plus = self.disp_gradient();
                disp[t] = -2.0 * step;
                self.mol.geom.geom_shift(A, disp.clone());
                let grad_minus = self.disp_gradient();
                // restore the geometry
                disp[t] = step;
                self.mol.geom.geom_shift(A, disp);

                for (iB, &B) in atm_list.iter().enumerate() {
                    for s in 0..3 {
                        hess[[s, t, iB, iA]] = (grad_plus[3 * B + s] - grad_minus[3 * B + s]) / (2.0 * step);
                    }
                }
            }
        }

        // symmetrize: the finite-difference Hessian is symmetric only up to round-off
        0.5 * (&hess + &hess.transpose([1, 0, 3, 2]))
    }
}
