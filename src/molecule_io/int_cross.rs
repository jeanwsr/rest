use rest_libcint::prelude::*;
use rest_tensors::MatrixFull;
use crate::molecule_io::Molecule;

impl Molecule {
    /// Compute cross integrals between basis functions of two molecules:
    /// ⟨ μ | op | ν ⟩, where μ ∈ self, ν ∈ other
    ///
    /// # Arguments
    /// - `other`: the second molecule
    /// - `op_name`: operator name, supports "ovlp", "kinetic", "nuclear"
    ///
    /// # Returns
    /// - `MatrixFull<f64>` with shape [self.num_basis, other.num_basis]
    pub fn int_cross(&self, other: &Molecule, op_name: String) -> MatrixFull<f64> {
        let intor = match op_name.as_str() {
            "ovlp" => "int1e_ovlp",
            "kinetic" => "int1e_kin",
            "nuclear" => "int1e_nuc",
            _ => panic!("Error:: unsupported op_name for int_cross: {}. Supported: ovlp, kinetic, nuclear", op_name),
        };

        let cint_data1 = self.initialize_cint(false);
        let cint_data2 = other.initialize_cint(false);

        let (out, shape) = CInt::integrate_cross(
            intor,
            [&cint_data1, &cint_data2],
            "s1",
            None,
        ).into();

        MatrixFull::from_vec([shape[0], shape[1]], out).unwrap()
    }
}
