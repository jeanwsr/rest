pub mod ej_ek_baseline;
pub mod memory_monitor;
pub mod rhf;
pub mod traits;
pub mod xc_hessian;
pub use ej_ek_baseline::{EjEkBaseline, BASELINE_TERM_KEYS};
pub use rhf::{
    test_cphf_hessian, test_ck7_h1ao, test_ck8_aux_response,
    compute_hessian, compute_frequencies,
    rhf_hessian_main,
};

use std::io::{self, Write};

use crate::scf_io::SCF;
use crate::utilities;
use tensors::MatrixFull;

pub fn numerical_hessian(scf_data: &SCF, displace: f64) -> MatrixFull<f64> {
    let num_atoms = scf_data.mol.geom.nfree;
    let mut num_hess = MatrixFull::new([num_atoms * 3, num_atoms * 3], 0.0);

    if scf_data.mol.ctrl.print_level > 0 {
        print!("Numerical Hessian calculation ...");
        io::stdout().flush().unwrap();
    }

    // Placeholder: numerical Hessian by finite difference of forces
    // To be implemented in future
    let _dump = displace;

    num_hess
}
