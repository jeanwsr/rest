//! External field module
//! 
//! Current implemented external field is only the dipole field.

use rest_libcint::prelude::*;
use rstsr::prelude::*;
use tensors::{MatrixFull, RIFull, BasicMatrixOpt};
use crate::grad::traits::GradAPI;
use crate::molecule_io::Molecule;

type Tsr<T> = Tensor<T, DeviceBLAS, IxD>;

/// External field description
/// 
/// Currently only dipole field is implemented.
/// - dipole: Dipole field vector [x, y, z]
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ExtField<T> {
    pub dipole: Option<[T; 3]>,
    // pub quadrupole: Option<[T; 9]>,
    // pub octupole: Option<[T; 27]>,
}

impl ExtField<f64> {
    pub fn empty() -> Self {
        Self { dipole: None }
    }

    /// Contribution to 2c integral
    pub fn contribution_2c(self, mol: &Molecule) -> MatrixFull<f64> {
        use rest_libcint_wrapper::*;
        // handle dipole
        let nao = mol.num_basis;
        let mut cint = mol.initialize_cint(false);
        let mut result = MatrixFull::new([nao, nao], 0.0);

        if let Some(dipole) = self.dipole {
            // handle dipole field
            // println!("Dipole field: {:?}", dipole);
            let dipole_norm = dipole.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
            // println!("Dipole norm: {:?}", dipole_norm);
            if dipole_norm < 10.0 * f64::EPSILON {
                println!("Dipole field is close to zero, ignore it");
            } else {
                let tsr_int1e_r = {
                    let (out, shape) = cint.integral_s1::<int1e_r>(None);
                    RIFull::from_vec(shape.try_into().unwrap(), out).unwrap()
                };
                for t in (0..3) {
                    let tsr_int1e_t = tsr_int1e_r.get_reducing_matrix(t).unwrap();
                    result += tsr_int1e_t.to_matrixfull().unwrap() * (-dipole[t]);
                }
            }
        }

        return result;
    }

    /// Derivative contribution to 2c integral (AO-derivative on bra side only)
    ///
    /// Output dimension: [3][nao, nao], corresponding to x/y/z derivative channels.
    pub fn contribution_2c_grad(self, mol: &Molecule) -> Tsr<f64> {
        use rest_libcint_wrapper::*;

        let nao = mol.num_basis;
        let device = DeviceBLAS::default();
        let mut cint = mol.initialize_cint(false);
        let mut result: Tsr<f64> = rt::zeros(([nao, nao, 3], &device));

        if let Some(dipole) = self.dipole {
            let dipole_norm = dipole.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
            if dipole_norm < 10.0 * f64::EPSILON {
                return result;
            }

            // let tsr_int1e_irp = {
            //     let (out, shape) = cint.integral_s1::<int1e_irp>(None);
            //     // print!("int1e_irp shape: {:?}\n", shape);
            //     rt::asarray((out, shape, &device)).into_shape((nao, nao, 3, 3)).swapaxes(0, 1).to_owned()
            // };

            let tsr_int1e_irp = {
                let (out, shape) = cint.integrate("int1e_irp", "s1", None).into();
                rt::asarray((out, shape, &device)).into_shape((nao, nao, 3, 3)).swapaxes(0, 1).to_owned()
            };
            

            // int1e_irp has 9 components: (r_t, p_a) for t,a in [x,y,z] 
            // packed as comp = t * 3 + a
            // contribution from dipole field is sum_{t} -dipole[t] * -int1e_irp(r_t, p_a) (\nabla r = - \nabla R)
            // for a in 0..3 {
            //     for t in 0..3 {
            //         let comp = a* 3 + t;
            //         *&mut result.i_mut((a, .., ..)) += tsr_int1e_irp.i((.., .., comp)) * (dipole[t]);
            //     }
            // }
            for t in 0..3 {
                result += tsr_int1e_irp.i((.., .., .., t)) * (-dipole[t]);
            }
            // result.swapaxes(0, 1);
            // result = &result + result.swapaxes(0, 1);
        }

        result
    }
}

/// Gradient evaluator for external-field contribution only.
pub struct ExtFieldGrad<'a> {
    pub mol: &'a Molecule,
    pub dm: &'a MatrixFull<f64>,
    pub ext_field: ExtField<f64>,
    pub result: Option<MatrixFull<f64>>,
    pub energy: f64,
}

impl<'a> ExtFieldGrad<'a> {
    pub fn new(mol: &'a Molecule, dm: &'a MatrixFull<f64>) -> Self {
        Self {
            mol,
            dm,
            ext_field: mol.geom.ext_field.clone(),
            result: None,
            energy: 0.0,
        }
    }

    pub fn calc(&mut self) -> MatrixFull<f64> {
        let device = DeviceBLAS::default();
        let natm = self.mol.geom.elem.len();
        let nao = self.mol.num_basis;

        let ext_field_dipole = self.ext_field.dipole.unwrap();
        let dipole_norm = ext_field_dipole.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        if dipole_norm < 10.0 * f64::EPSILON {
            return MatrixFull::new([3, natm], 0.0);
        }
        let mut de_ext_tsr: Tsr<f64> = rt::zeros(([3, natm], &device));

        // Nuclear grad contribution from external field:
        // E_nuc^ext = sum_A Z_A^eff (R_A \dot F)
        // dE_nuc^ext / dR_{A,t} = Z_A^eff F_t
        let atom_charge = crate::geom_io::get_charge(&self.mol.geom.elem);
        let necp_by_atom = self
            .mol
            .basis4elem
            .iter()
            .take(natm)
            .map(|x| x.ecp_electrons.unwrap_or(0) as f64)
            .collect::<Vec<f64>>();
        let eff_charge = rt::asarray((atom_charge, &device)) - rt::asarray((necp_by_atom, &device));
        let dipole = rt::asarray((ext_field_dipole.to_vec(), &device));

        de_ext_tsr += dipole.i((.., None)) * eff_charge.i((None, ..));

        // Electron grad contribution from external field:
        // E_ele^ext = -1 * sum_{pq} P_{pq} H_r_{pq} \dot F
        // dE_ele^ext / dR_{A,t} = -1 * sum_{pq} P_{pq} (dH_r_{pq} / dR_{A,t}) \dot F
        let h1_ext = self.ext_field.clone().contribution_2c_grad(self.mol);
        let dm = rt::asarray((&self.dm.data, self.dm.size, &device));

        // self.energy = h_ext.iter().zip(self.dm.iter()).map(|(h, d)| h * d).sum::<f64>();

        let ao_slice = self.mol.aoslice_by_atom();

        for atm in 0..natm {
            let mut tmp_holder: Tsr<f64> = rt::zeros(([nao, nao, 3], &device));
            let [_, _, p0, p1] = ao_slice[atm];
            *&mut tmp_holder.i_mut(p0..p1) += &h1_ext.i(p0..p1);
            let tmp_holder = (&tmp_holder + tmp_holder.swapaxes(0, 1)).into_contig(FlagOrder::F);
            *&mut de_ext_tsr.i_mut((.., atm)) -= (tmp_holder * dm.i((.., .., None))).sum_axes([0, 1]);
        }


        let de_ext = {
            let raw = de_ext_tsr.into_shape(-1).into_raw();
            MatrixFull::from_vec([3, natm], raw).unwrap()
        };
        
        self.result = Some(de_ext.clone());
        de_ext
    }
}

impl GradAPI for ExtFieldGrad<'_> {
    fn get_gradient(&self) -> MatrixFull<f64> {
        self.result.clone().unwrap_or_else(|| MatrixFull::new([3, self.mol.geom.elem.len()], 0.0))
    }

    fn get_energy(&self) -> f64 {
        self.energy
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ctrl_io::{parse_ctl_from_json, InputKeywords};
    use crate::scf_io::scf_without_build;
    use rstsr::prelude::*;
    use tensors::BasicMatrix;

    #[test]
    fn test_nh3() {
        let mut mol = initialize_nh3();
        let mut ext_field = ExtField::empty();
        ext_field.dipole = Some([0.0, 0.0, 1.0]);
        let tsr = {
            let tsr = ext_field.contribution_2c(&mol);
            let shape = tsr.size();
            let vec = tsr.data;
            rt::asarray((vec, shape))
        };
        println!("2c integral: {:16.10?}", tsr);
    }

    fn initialize_nh3() -> Molecule {
        let input_token = r##"
[ctrl]
     xc =                   "hf"
     basis_path =           "def2-TZVP"
     auxbas_path =          "def2-universal-JKFIT"
     num_threads =          4
     spin =                 1

[geom]
    name = "NH3"
    unit = "Angstrom"
    position = """
        N  0.0  0.0  0.0
        H  0.0  1.5  1.0
        H  1.4  1.1  0.0
        H  1.2  0.0  1.3
    """
"##;
        let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
        let (mut ctrl, mut geom) = parse_ctl_from_json(&keys).unwrap();
        Molecule::build_native(ctrl, geom, None).unwrap()
    }
}
