use rest_libcint::CintType;
use crate::basis_io::BasCell;

pub fn count_basis_functions(shells: &[BasCell], cint_type: &CintType) -> usize {
    shells.iter().map(|shell| {
        let ang = shell.angular_momentum[0] as usize;
        let bas_num = match cint_type {
            CintType::Cartesian => (ang + 1) * (ang + 2) / 2,
            CintType::Spheric => ang * 2 + 1,
            CintType::Spinor => panic!("Spinor is not yet implemented"),
        };
        bas_num * shell.coefficients.len()
    }).sum()
}
