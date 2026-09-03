use crate::dftd::energy::dftd;
use crate::scf_io::SCF;

use crate::grad::traits::GradAPI;

/// A struct for DFTD gradient computation, which implements the GradAPI trait.
pub struct DFTDGrad<'a> {
    pub(crate) scf_data: &'a SCF,
    pub(crate) evaluated: bool,
    pub(crate) result: Option<(f64, Option<Vec<f64>>, Option<Vec<f64>>)>,
}

impl<'a> DFTDGrad<'a> {
    pub fn new(scf_data: &'a SCF) -> Self {
        Self { scf_data, evaluated: false, result: None }
    }

    pub fn make_grad(&mut self) {
        if self.evaluated {
            println!("[WARN] DFTD gradient is already evaluated. Will use the last result.");
        } else {
            self.result = dftd(&self.scf_data.mol);
            self.evaluated = true;
        }
    }
}

impl<'a> GradAPI for DFTDGrad<'a> {
    fn get_energy(&self) -> f64 {
        if let Some((energy, _, _)) = self.result {
            energy
        } else {
            panic!("Energy is not computed yet. Please call make_grad() first.");
        }
    }

    fn get_gradient(&self) -> tensors::MatrixFull<f64> {
        if let Some((_, Some(grad), _)) = &self.result {
            let vec = grad.clone();
            let natm = self.scf_data.mol.geom.elem.len();
            tensors::MatrixFull::from_vec([3, natm], vec).unwrap()
        } else {
            panic!("Gradient is not computed yet. Please call make_grad() first.");
        }
    }
}
