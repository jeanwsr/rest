use rayon::prelude::*;
use rstsr::prelude::*;
use std::collections::HashMap;
use std::sync::mpsc::channel;
use tensors::{matrix_blas_lapack::_power_rayon_for_symmetric_matrix, MatrixFull};

type Tsr<T> = Tensor<T, DeviceBLAS, IxD>;
type TsrView<'a, T> = TensorView<'a, T, DeviceBLAS, IxD>;
type TsrMut<'a, T> = TensorMut<'a, T, DeviceBLAS, IxD>;

use crate::constants::AUXBAS_THRESHOLD;
use crate::grad::traits::GradAPI;
use crate::scf_io;
use crate::scf_io::SCF;
use crate::Molecule;
use super::rhf::{RIHFGradientFlags};
use super::uhf::{RIUHFGradient};
use crate::dft::{Grids, DFA4REST};
use crate::dft::xc_deriv::XCType;
use crate::dft::num_int::{NumInt, XCData, BlockSettings};
use crate::dft::num_int::{eval_rho5_batch, eval_rho5_dm_only_batch, eval_ao_batch};
use crate::dft::libxc_itrf::{eval_xc_eff};
use super::rks::{gga_grad_sum, tau_grad_dot};
use crate::utilities::{self, balancing};

use tensors::matrix_blas_lapack::{omp_get_num_threads_wrapper, omp_set_num_threads_wrapper};


impl<'a> NumInt<'a> for RIUHFGradient<'a> {
    fn gen_xc_data(&'a self, scf_data: &'a SCF, spin: usize) -> XCData<'a> {
        let xc_method = &self.scf_data.mol.xc_data;
        let xc_type = if xc_method.use_kinetic_density() {
            XCType::MGGA
        } else if xc_method.use_density_gradient() {
            XCType::GGA
        } else {
            XCType::LDA
        };

        let xc_code = &xc_method.dfa_compnt_scf;
        let xc_params = &xc_method.dfa_paramr_scf;

        let density_matrix = &scf_data.density_matrix;
        let mo_coeffs = scf_data.eigenvectors.clone().to_vec();
        let mo_occ = scf_data.occupation.clone().to_vec();

        let device = DeviceOpenBLAS::default();

        XCData {
            xc_type,
            xc_code,
            xc_params,
            device: device,
            dm: density_matrix,
            mo_coeffs: Some(mo_coeffs),
            occ: Some(mo_occ),
        }
    }


    fn get_vxc(
        &'a self,
        xc_data: &'a XCData,
        grids: &mut Grids,
        mol: &Molecule,
        spin: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<MatrixFull<f64>>) {
        unimplemented!("Gradient calculation does not goes here.");
    }


    fn get_fxc(
        &'a self,
        xc_data0: &'a XCData,
        xc_data1: &'a XCData,
        grids: &mut Grids,
        mol: &Molecule,
        spin: usize,
    ) -> Vec<MatrixFull<f64>> {
        unreachable!("Gradient calculation does not support get_fxc");
    }
}


impl RIUHFGradient<'_> {
    pub fn calc_de_xc(&mut self) -> &mut Self {
        // get dx vxc [nao, nao, 3, 2]
        // println!("Calculating dx vxc");
        let scf_data = self.scf_data; 
        let xc_data = self.gen_xc_data(scf_data, 1);
        let mut mol = scf_data.mol.clone();
        let grids = &mut Grids::build(&mut mol);
        // let dao_vxc = get_vxc(&self, &xc_data, grids, &scf_data.mol, self.flags.max_memory.unwrap_or(2000.0) as usize);
        let dao_vxc = get_vxc_rayon(&self, &xc_data, grids, &scf_data.mol);
        // println!("Finished calculating dx vxc");
        
        // contract dao_vxc and dm (tuv, uv -> tu) and sum over spin case
        let nao = mol.num_basis;
        let mut dao_xc:Tsr<f64> = rt::zeros(([nao, 3], &xc_data.device));        
        let dm = get_dm(scf_data, &xc_data.device);
        for ispin in 0..2usize {
            let dm_s = dm[ispin].view();
            dao_xc += (&dao_vxc[ispin] * &dm_s.i((.., .., None))).sum_axes(1) * 2.0;
        }
        // println!("Finished calculating dao_xc");
        
        // scatter result to natm  
        let natm = mol.geom.elem.len();
        let mut de_xc: Tsr<f64> = rt::zeros(([3, natm], &xc_data.device));
        let ao_slice = mol.aoslice_by_atom();
        for atm in 0..natm {
            let [_, _, p0, p1] = ao_slice[atm];
            *&mut de_xc.i_mut((.., atm)) += &dao_xc.i((p0..p1)).sum_axes(0);
        }

        // return to REST, note f-contiguous transpose
        let de_xc_raw = de_xc.into_raw_parts().0.into_cpu_vec().unwrap();
        let de_xc = MatrixFull::from_vec([3, natm], de_xc_raw).unwrap();
        self.result.insert("de_xc".into(), de_xc);
        
        // println!("Finished calculating de_xc");

        self
    }


    pub fn calc_uks(&mut self) -> &MatrixFull<f64> {
        let mut time_records = crate::utilities::TimeRecords::new();
        time_records.new_item("uks grad", "uks grad");
        time_records.new_item("uks grad calc_de_nuc", "uks grad calc_de_nuc");
        time_records.new_item("uks grad calc_de_ovlp", "uks grad calc_de_ovlp");
        time_records.new_item("uks grad calc_de_hcore", "uks grad calc_de_hcore");
        time_records.new_item("uks grad calc_de_jk", "uks grad calc_de_jk");
        time_records.new_item("uks grad calc_de_xc", "uks grad calc_de_xc");

        time_records.count_start("uks grad");

        time_records.count_start("uks grad calc_de_nuc");
        self.calc_de_nuc();
        time_records.count("uks grad calc_de_nuc");

        time_records.count_start("uks grad calc_de_ovlp");
        self.calc_de_ovlp();
        time_records.count("uks grad calc_de_ovlp");

        time_records.count_start("uks grad calc_de_hcore");
        self.calc_de_hcore();
        time_records.count("uks grad calc_de_hcore");

        if self.flags.factor_j.is_some() || self.flags.factor_k.is_some() {
            time_records.count_start("uks grad calc_de_jk");
            self.calc_de_jk();
            time_records.count("uks grad calc_de_jk");
        }

        time_records.count_start("uks grad calc_de_xc");
        self.calc_de_xc();
        time_records.count("uks grad calc_de_xc");

        time_records.count("uks grad");

        let mut de = self.result.get("de_nuc").unwrap().clone();
        de += self.result.get("de_ovlp").unwrap().clone();
        de += self.result.get("de_hcore").unwrap().clone();
        self.result.get("de_j").map(|x| de += x.clone());
        self.result.get("de_k").map(|x| de += x.clone());
        self.result.get("de_jaux").map(|x| de += x.clone());
        self.result.get("de_kaux").map(|x| de += x.clone());
        self.result.get("de_xc").map(|x| de += x.clone());
        self.result.insert("de".into(), de);

        if self.flags.print_level >= 2 {
            time_records.report_all();
        }

        return self.result.get("de").unwrap();
    }


}


fn get_dm(scf_data: &SCF, device: &DeviceBLAS) -> Vec<Tsr<f64>> {
    // This can be reconsidered if 3-D (spin, ao, ao) is better.
    // Currently, vector of 2-D (ao, ao) is used.
    // from uhf.rs 
    let mut result = vec![];
    for spin in [0, 1] {
        let dm = &scf_data.density_matrix[spin];
        result.push(rt::asarray((&dm.data, dm.size, device)).to_owned());
    }
    return result;
}


fn get_vxc(gradient_method: &RIUHFGradient, xc_data: &XCData, grids: &mut Grids, mol: &Molecule, max_memory: usize) -> Vec<Tsr<f64>> {
    let num_grids = grids.weights.len();
    let num_basis = mol.num_basis;
    // determine block settings 
    let ao_deriv = match xc_data.xc_type {
        XCType::HF => 1usize,
        XCType::LDA => 1usize,
        XCType::GGA => 2usize,
        XCType::MGGA => 2usize,
    };
    
    let ao_comp = (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) / 6; // number of components for ao derivatives 
    let blksize = max_memory * 1_000_000 / 8 / ((ao_comp + 1) * num_basis);
    let blksize = blksize.min(num_grids).max(4); 
    let block_settings = BlockSettings {
        ao_deriv,
        max_memory,
        blksize,
    };
   
    let device = &xc_data.device;

    // UKS case 
    let spin = 1usize;
    let nspin = spin + 1;
    let mut vmat_a = rt::zeros(([num_basis, num_basis, 3], device));
    let mut vmat_b = rt::zeros(([num_basis, num_basis, 3], device));
    let mut vmat = vec![vmat_a, vmat_b];

    let grids_iterator = gradient_method.block_loop(mol, grids, block_settings);
    // calculation starts here 
    let deriv = 1 as usize;          
    grids_iterator.into_iter().for_each(
        |block| 
        {
            let num_grids = block.weights.len();
            let loc_rho = if let Some(mo_coeffs) = &xc_data.mo_coeffs {
                eval_rho5_batch(&block.ao, xc_data.xc_type, mo_coeffs, xc_data.occ.as_ref().unwrap(), nspin, num_grids)
            } else {
                eval_rho5_dm_only_batch(&block.ao, xc_data.xc_type, xc_data.dm, nspin, num_grids)
            };
            // println!("Shape of loc_rho: {:?}", loc_rho.shape());

            let loc_xc_tensor = eval_xc_eff(xc_data.xc_code, xc_data.xc_params, xc_data.xc_type, spin, &loc_rho.raw(), num_grids, deriv);
            let loc_vxc = if let Some(vxc) = loc_xc_tensor[1].as_ref() {
                rt::asarray(vxc)
            } else {
                panic!("Exchange-correlation potential is not provided for the given xc_type: {:?}", xc_data.xc_type);
            };
            // println!("Shape of loc_vxc: {:?}", loc_vxc.shape()); // assume to be [num_grids, num_components, nspin]

            let loc_weights = rt::asarray((block.weights, device));
            let ao_shape = vec![num_basis, num_grids, ao_comp];
            let loc_ao = rt::asarray((&block.ao.data, ao_shape, device));

            let mut loc_wv = loc_vxc * &loc_weights.i((.., None, None));

            match xc_data.xc_type {
                XCType::LDA => {
                    for ispin in 0..nspin {
                        let aow_s = &loc_ao.i((.., .., 0)) * &loc_wv.i((None, .., 0, ispin)); 
                        let aow_s = aow_s.t();
                        for ic in 0..3 {
                            vmat[ispin].i_mut((.., .., ic)).matmul_from(
                                &loc_ao.i((.., .., ic+1)), &aow_s, 1.0, 1.0
                            );
                        }
                    }
                }, 
                XCType::GGA => {
                    for ispin in 0..nspin {
                        // let mut vmat_s = vmat.i((.., .., .., ispin)).to_owned();
                        loc_wv.i_mut((.., 0, ispin)).mul_assign(0.5);
                        gga_grad_sum(&mut vmat[ispin], loc_ao.view(), loc_wv.i((.., .., ispin)));
                    }
                },
                XCType::MGGA => {
                    for ispin in 0..nspin {
                        // let mut vmat_s = vmat.i((.., .., .., ispin)).to_owned();
                        loc_wv.i_mut((.., 0, ispin)).mul_assign(0.5);
                        loc_wv.i_mut((.., 4, ispin)).mul_assign(0.5);
                        gga_grad_sum(&mut vmat[ispin], loc_ao.view(), loc_wv.i((.., .., ispin)));
                        tau_grad_dot(&mut vmat[ispin], loc_ao.view(), loc_wv.i((.., .., ispin)));
                    }
                },
                XCType::HF => {
                    unreachable!("HF gradient calculation does not support here in get_vxc");
                },
            }
        }
    );
    // nabla R = - nabla r 
    for ispin in 0..nspin {
        vmat[ispin].mul_assign(-1.0);
    }

    vmat
}


fn get_vxc_rayon(gradient_method: &RIUHFGradient, xc_data: &XCData, grids: &mut Grids, mol: &Molecule) -> Vec<Tsr<f64>> {
    let default_omp_num_threads = omp_get_num_threads_wrapper();

    let num_basis = mol.num_basis;
    // determine block settings 
    let ao_deriv = match xc_data.xc_type {
        XCType::HF => 1usize,
        XCType::LDA => 1usize,
        XCType::GGA => 2usize,
        XCType::MGGA => 2usize,
    };
    let ao_comp = (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) / 6;
    let device = &xc_data.device;
    let deriv = 1 as usize;
    // UKS case 
    let spin = 1usize;
    let nspin = spin + 1;
    let mut vmat_a = rt::zeros(([num_basis, num_basis, 3], device));
    let mut vmat_b = rt::zeros(([num_basis, num_basis, 3], device));
    let mut vmat = vec![vmat_a, vmat_b];

    let (sender, receiver) = channel();
    grids.parallel_balancing.par_iter()
    .for_each_with(
        sender,|s, range_grids| 
        {
            omp_set_num_threads_wrapper(1);
            // eval batch ao 
            let num_grids = range_grids.len();
            let loc_coordinates = &grids.coordinates[range_grids.clone()];
            let loc_ao = eval_ao_batch(mol, loc_coordinates, ao_deriv, num_grids);
            // eval batch rho 
            let loc_rho = if let Some(mo_coeffs) = &xc_data.mo_coeffs {
                eval_rho5_batch(&loc_ao, xc_data.xc_type, mo_coeffs, xc_data.occ.as_ref().unwrap(), nspin, num_grids)
            } else {
                eval_rho5_dm_only_batch(&loc_ao, xc_data.xc_type, xc_data.dm, nspin, num_grids)
            };
            // eval vxc on grids 
            let loc_xc_tensor = eval_xc_eff(xc_data.xc_code, xc_data.xc_params, xc_data.xc_type, spin, &loc_rho.raw(), num_grids, deriv);
            // let loc_rho = rt::asarray((rho_array, shape, device));
            let loc_vxc = if let Some(vxc) = loc_xc_tensor[1].as_ref() {
                rt::asarray(vxc)
            } else {
                panic!("Exchange-correlation potential is not provided for the given xc_type: {:?}", xc_data.xc_type);
            };
            // eval batch vxc_ao 
            let mut loc_vmat_a = rt::zeros(([num_basis, num_basis, 3], device));
            let mut loc_vmat_b = rt::zeros(([num_basis, num_basis, 3], device));
            let mut loc_vmat = vec![loc_vmat_a, loc_vmat_b];
            let loc_weights = rt::asarray((&grids.weights[range_grids.clone()], device));
            let ao_shape = vec![num_basis, num_grids, ao_comp];
            let loc_ao = rt::asarray((&loc_ao.data, ao_shape, device));
            
            let mut loc_wv = loc_vxc * &loc_weights.i((.., None, None));

            match xc_data.xc_type {
                XCType::LDA => {
                    for ispin in 0..nspin {
                        let aow_s = &loc_ao.i((.., .., 0)) * &loc_wv.i((None, .., 0, ispin)); 
                        let aow_s = aow_s.t();
                        for ic in 0..3 {
                            loc_vmat[ispin].i_mut((.., .., ic, ispin)).matmul_from(
                                &loc_ao.i((.., .., ic+1)), &aow_s, 1.0, 1.0
                            );
                        }
                    }
                }, 
                XCType::GGA => {
                    for ispin in 0..nspin {
                        loc_wv.i_mut((.., 0, ispin)).mul_assign(0.5);
                        gga_grad_sum(&mut loc_vmat[ispin], loc_ao.view(), loc_wv.i((.., .., ispin)));
                    }
                },
                XCType::MGGA => {
                    for ispin in 0..nspin {
                        loc_wv.i_mut((.., 0, ispin)).mul_assign(0.5);
                        loc_wv.i_mut((.., 4, ispin)).mul_assign(0.5);
                        gga_grad_sum(&mut loc_vmat[ispin], loc_ao.view(), loc_wv.i((.., .., ispin)));
                        tau_grad_dot(&mut loc_vmat[ispin], loc_ao.view(), loc_wv.i((.., .., ispin)));
                    }
                },
                XCType::HF => {
                    unreachable!("HF gradient calculation does not support here in get_vxc");
                },
            }
            s.send((loc_vmat)).unwrap();
        } 
    );
    receiver.into_iter().for_each(
        |(loc_vmat)| 
        {
            vmat[0] += loc_vmat[0].view();
            vmat[1] += loc_vmat[1].view();
        }
    );
    omp_set_num_threads_wrapper(default_omp_num_threads);
    // nabla R = - nabla r
    for ispin in 0..nspin {
        vmat[ispin].mul_assign(-1.0);
    }
    vmat
}