// use num_traits::ToPrimitive;
use rayon::prelude::*;
// use rest_libcint::prelude::*;
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
use crate::grad::rhf::{RIRHFGradient, RIHFGradientFlags};
use crate::dft::{Grids, DFA4REST};
use crate::dft::xc_deriv::XCType;
use crate::dft::num_int::{NumInt, XCData, BlockSettings};
use crate::dft::num_int::{eval_rho5_batch, eval_rho5_dm_only_batch, eval_ao_batch};
use crate::dft::libxc_itrf::{eval_xc_eff};
use crate::utilities::{self, balancing};

use tensors::matrix_blas_lapack::{omp_get_num_threads_wrapper, omp_set_num_threads_wrapper};


impl<'a> NumInt<'a> for RIRHFGradient<'a> {
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
        let num_grids = grids.weights.len();
        let num_basis = mol.num_basis;
        // determine block settings 
        let ao_deriv = match xc_data.xc_type {
            XCType::HF => 1usize,
            XCType::LDA => 1usize,
            XCType::GGA => 2usize,
            XCType::MGGA => 2usize,
        };
        let max_memory = if let Some(max_memory) = self.flags.max_memory {
            max_memory as usize
        } else {
            2000 as usize // default to 2000 MB
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
        
        let mut nelec = vec![0.0; 1];
        let mut exc_sum = vec![0.0; 1];
        let mut vxc = rt::zeros(([num_basis, num_basis, 3], device));

        let grids_iterator = self.block_loop(mol, grids, block_settings);
        // calculation starts here 
        let deriv = 1usize;          
        grids_iterator.into_iter().for_each(
            |block| 
            {
                let num_grids = block.weights.len();
                let loc_rho = if let Some(mo_coeffs) = &xc_data.mo_coeffs {
                    eval_rho5_batch(&block.ao, xc_data.xc_type, mo_coeffs, xc_data.occ.as_ref().unwrap(), spin + 1, num_grids)
                } else {
                    eval_rho5_dm_only_batch(&block.ao, xc_data.xc_type, xc_data.dm, spin + 1, num_grids)
                };

                let loc_xc_tensor = eval_xc_eff(xc_data.xc_code, xc_data.xc_params, xc_data.xc_type, spin, loc_rho.raw(), num_grids, deriv);
                
                // let loc_rho = rt::asarray((rho_array, shape, device));
                let loc_vxc = if let Some(vxc) = loc_xc_tensor[1].as_ref() {
                    rt::asarray(vxc)
                } else {
                    panic!("Exchange-correlation potential is not provided for the given xc_type: {:?}", xc_data.xc_type);
                };
                let loc_weights = rt::asarray((block.weights, device));
                let ao_shape = vec![num_basis, num_grids, ao_comp];
                let loc_ao = rt::asarray((&block.ao.data, ao_shape, device));
                
                let mut loc_wv = loc_vxc * &loc_weights.i((.., None));

                match xc_data.xc_type {
                    XCType::LDA => {
                        let aow = &loc_ao.i((.., .., 0)) * &loc_wv.i((None, ..)); 
                        let aow = aow.t();
                        for ic in 0..3 {
                            vxc.i_mut(ic).matmul_from(
                                &loc_ao.i((.., .., ic+1)), &aow, 1.0, 1.0
                            );
                        }
                    }, 
                    XCType::GGA => {
                        loc_wv.i_mut((.., 0)).mul_assign(0.5);
                        gga_grad_sum(&mut vxc, loc_ao.view(), loc_wv.view());
                    },
                    XCType::MGGA => {
                        loc_wv.i_mut((.., 0)).mul_assign(0.5);
                        loc_wv.i_mut((.., 4)).mul_assign(0.5);
                        gga_grad_sum(&mut vxc, loc_ao.view(), loc_wv.view());
                        tau_grad_dot(&mut vxc, loc_ao.view(), loc_wv.view());
                    },
                    XCType::HF => {
                        unreachable!("HF gradient calculation does not support here in get_vxc");
                    },
                }
            }
        );
        vxc.mul_assign(-1.0);

        let mut vmat = vec!();
        for ic in 0..3 {
            vmat.push(MatrixFull::from_vec([num_basis, num_basis], vxc.i((.., .., ic)).to_vec()).unwrap());
        }


        (nelec, exc_sum, vmat)

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

impl RIRHFGradient<'_> {
    pub fn calc_de_xc(&mut self) -> &mut Self {
        // get dx vxc [nao, nao, 3]
        let scf_data = self.scf_data; 
        let xc_data = self.gen_xc_data(scf_data, 0);
        let mut mol = scf_data.mol.clone();
        let grids = &mut Grids::build(&mut mol);
        // let dao_vxc = get_vxc(&self, &xc_data, grids, &scf_data.mol, self.flags.max_memory.unwrap_or(2000.0) as usize);
        // let dao_vxc = get_vxc_rayon(&self, &xc_data, grids, &scf_data.mol);
        let dao_vxc = get_vxc_rayon_new(&self, &xc_data, grids, &mol, 16usize);
        // println!("Print dao_vxc: {:?}", dao_vxc);
        
        // contract dao_vxc and dm (tuv, uv -> tu)
        let nao = mol.num_basis;
        let mut dao_xc:Tsr<f64> = rt::zeros(([nao, 3], &xc_data.device));        
        let dm = get_dm(scf_data, &xc_data.device);
        dao_xc = 2.0 * (&dao_vxc * &dm.i((.., .., None))).sum_axes(1);
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
        let mut de_xc_raw = de_xc.into_raw_parts().0.into_cpu_vec().unwrap();
        // MPI: the numerical grids are distributed across ranks (`Grids::build`), so each rank
        // only integrates its local grid portion; sum the XC gradient contribution over ranks.
        #[cfg(feature = "mpi")]
        if let Some(mpi_op) = self.mpi_operator {
            let mut reduced = vec![0.0_f64; de_xc_raw.len()];
            crate::mpi_io::mpi_allreduce(
                &mpi_op.world,
                &de_xc_raw,
                &mut reduced,
                &mpi::collective::SystemOperation::sum(),
            );
            de_xc_raw.copy_from_slice(&reduced);
        }
        let de_xc = MatrixFull::from_vec([3, natm], de_xc_raw).unwrap();
        self.result.insert("de_xc".into(), de_xc);
        
        // println!("Finished calculating de_xc");

        self
    }

    pub fn calc_rks(&mut self) -> &MatrixFull<f64> {
        let mut time_records = crate::utilities::TimeRecords::new();
        time_records.new_item("rks grad", "rks grad");
        time_records.new_item("rks grad calc_de_nuc", "rks grad calc_de_nuc");
        time_records.new_item("rks grad calc_de_ovlp", "rks grad calc_de_ovlp");
        time_records.new_item("rks grad calc_de_hcore", "rks grad calc_de_hcore");
        time_records.new_item("rks grad calc_de_ext_field", "rks grad calc_de_ext_field");
        time_records.new_item("rks grad calc_de_jk", "rks grad calc_de_jk");
        time_records.new_item("rks grad calc_de_xc", "rks grad calc_de_xc");
        time_records.new_item("rks grad calc_de_qmmm", "rks grad calc_de_qmmm");
        time_records.new_item("rks grad calc_de_solvent", "rks grad calc_de_solvent");
        time_records.new_item("rks grad calc_de_ext_field", "rks grad calc_de_ext_field");

        time_records.count_start("rks grad");

        time_records.count_start("rks grad calc_de_nuc");
        self.calc_de_nuc();
        time_records.count("rks grad calc_de_nuc");

        time_records.count_start("rks grad calc_de_ovlp");
        self.calc_de_ovlp();
        time_records.count("rks grad calc_de_ovlp");

        time_records.count_start("rks grad calc_de_hcore");
        self.calc_de_hcore();
        time_records.count("rks grad calc_de_hcore");

        if self.flags.ext_field_dipole.is_some() {
            time_records.count_start("rhf grad calc_de_ext_field");
            self.calc_de_ext_field();
            time_records.count("rhf grad calc_de_ext_field");
        }

        time_records.count_start("rks grad calc_de_qmmm");
        self.calc_de_qmmm();
        time_records.count("rks grad calc_de_qmmm");

        if self.flags.factor_j.is_some() || self.flags.factor_k.is_some() {
            time_records.count_start("rks grad calc_de_jk");
            self.calc_de_jk();
            time_records.count("rks grad calc_de_jk");
        }

        time_records.count_start("rks grad calc_de_xc");
        self.calc_de_xc();
        time_records.count("rks grad calc_de_xc");

        time_records.count_start("rks grad calc_de_solvent");
        self.calc_de_solvent();
        time_records.count("rks grad calc_de_solvent");

        time_records.count("rks grad");

        let mut de = self.result.get("de_nuc").unwrap().clone();
        de += self.result.get("de_ovlp").unwrap().clone();
        de += self.result.get("de_hcore").unwrap().clone();
        self.result.get("de_j").map(|x| de += x.clone());
        self.result.get("de_k").map(|x| de += x.clone());
        self.result.get("de_sr").map(|x| de += x.clone());
        self.result.get("de_jaux").map(|x| de += x.clone());
        self.result.get("de_kaux").map(|x| de += x.clone());
        self.result.get("de_sraux").map(|x| de += x.clone());
        self.result.get("de_xc").map(|x| de += x.clone());
        self.result.get("de_qmmm").map(|x| de += x.clone());
        self.result.get("de_solvent").map(|x| de += x.clone());
        self.result.get("de_ext_field").map(|x| de += x.clone());
        self.result.insert("de".into(), de);

        if self.flags.print_level >= 2 {
            time_records.report_all();
        }

        return self.result.get("de").unwrap();
    }
}


fn get_dm(scf_data: &SCF, device: &DeviceBLAS) -> Tsr<f64> {
    let dm = &scf_data.density_matrix[0];
    return rt::asarray((&dm.data, dm.size, device)).to_owned();
}



pub fn gga_grad_sum(vmat: &mut Tsr<f64>, ao:TsrView<f64>, wv:TsrView<f64>) {
    let aow = &ao.i((.., .., 0..4)) * &wv.i((None, .., 0..4));
    let aow = aow.t();
    // ao_x (nbas, ngrid) vrho (ngrid) ao 
    // ao_x (nbas, ngrid) (vsigma[P]) nabla_ao 
    for ic in 0..4 {
        vmat.i_mut((.., .., 0)).matmul_from(
            &ao.i((.., .., 1)), &aow.i(ic), 1.0, 1.0 
        );
        vmat.i_mut((.., .., 1)).matmul_from(
            &ao.i((.., .., 2)), &aow.i(ic), 1.0, 1.0 
        );
        vmat.i_mut((.., .., 2)).matmul_from(
            &ao.i((.., .., 3)), &aow.i(ic), 1.0, 1.0 
        );
    }
    let aow = make_dR_dao_w(ao.view(), wv);
    for ic in 0..3 {
        vmat.i_mut((.., .., ic)).matmul_from(
            &aow.i((.., .., ic)), &ao.i((.., .., 0)).t(), 1.0, 1.0
        );
    }
}


pub fn tau_grad_dot(vmat: &mut Tsr<f64>, ao:TsrView<f64>, wv:TsrView<f64>) {
    // vtau ao_x
    let aow = &ao.i((.., .., 1)) * &wv.i((None, .., 4));
    let aow = aow.t();
    // contract with ao_xx, ao_xy, ao_xz 
    vmat.i_mut((.., .., 0)).matmul_from(
        &ao.i((.., .., 4)), &aow, 1.0, 1.0 
    );
    vmat.i_mut((.., .., 1)).matmul_from(
        &ao.i((.., .., 5)), &aow, 1.0, 1.0 
    );
    vmat.i_mut((.., .., 2)).matmul_from(
        &ao.i((.., .., 6)), &aow, 1.0, 1.0 
    );
    // vtau ao_y
    let aow = &ao.i((.., .., 2)) * &wv.i((None, .., 4));
    let aow = aow.t();
    // contract with ao_xy, ao_yy, ao_yz
    vmat.i_mut((.., .., 0)).matmul_from(
        &ao.i((.., .., 5)), &aow, 1.0, 1.0 
    );
    vmat.i_mut((.., .., 1)).matmul_from(
        &ao.i((.., .., 7)), &aow, 1.0, 1.0 
    );
    vmat.i_mut((.., .., 2)).matmul_from(
        &ao.i((.., .., 8)), &aow, 1.0, 1.0 
    );
    // vtau ao_z
    let aow = &ao.i((.., .., 3)) * &wv.i((None, .., 4));
    let aow = aow.t();
    // contract with ao_xz, ao_yz, ao_zz
    vmat.i_mut((.., .., 0)).matmul_from(
        &ao.i((.., .., 6)), &aow, 1.0, 1.0 
    );
    vmat.i_mut((.., .., 1)).matmul_from(
        &ao.i((.., .., 8)), &aow, 1.0, 1.0 
    );
    vmat.i_mut((.., .., 2)).matmul_from(
        &ao.i((.., .., 9)), &aow, 1.0, 1.0 
    );
}    


#[allow(non_snake_case)]
fn make_dR_dao_w(ao:TsrView<f64>, wv:TsrView<f64>) -> Tsr<f64> {
    let mut aow = &ao.i((.., .., 1..4)) * &wv.i((None, .., 0));
    aow.i_mut((.., .., 0)).add_assign(&ao.i((.., .., 4)) * &wv.i((None, .., 1))); // dX nabla_x
    aow.i_mut((.., .., 0)).add_assign(&ao.i((.., .., 5)) * &wv.i((None, .., 2))); // dX nabla_y
    aow.i_mut((.., .., 0)).add_assign(&ao.i((.., .., 6)) * &wv.i((None, .., 3))); // dX nabla_z
    aow.i_mut((.., .., 1)).add_assign(&ao.i((.., .., 5)) * &wv.i((None, .., 1))); // dY nabla_x
    aow.i_mut((.., .., 1)).add_assign(&ao.i((.., .., 7)) * &wv.i((None, .., 2))); // dY nabla_y
    aow.i_mut((.., .., 1)).add_assign(&ao.i((.., .., 8)) * &wv.i((None, .., 3))); // dY nabla_z
    aow.i_mut((.., .., 2)).add_assign(&ao.i((.., .., 6)) * &wv.i((None, .., 1))); // dZ nabla_x
    aow.i_mut((.., .., 2)).add_assign(&ao.i((.., .., 8)) * &wv.i((None, .., 2))); // dZ nabla_y
    aow.i_mut((.., .., 2)).add_assign(&ao.i((.., .., 9)) * &wv.i((None, .., 3))); // dZ nabla_z
    
    aow 
}


fn get_vxc(gradient_method: &RIRHFGradient, xc_data: &XCData, grids: &mut Grids, mol: &Molecule, max_memory:usize) -> Tsr<f64> {
    let num_grids = grids.weights.len();
    let num_basis = mol.num_basis;
    // determine block settings 
    let ao_deriv = match xc_data.xc_type {
        XCType::HF => 1usize,
        XCType::LDA => 1usize,
        XCType::GGA => 2usize,
        XCType::MGGA => 2usize,
    };
    // let max_memory = if let Some(max_memory) = gradient_method.flags.max_memory {
    //     max_memory as usize
    // } else {
    //     2000 as usize // default to 2000 MB
    // };
    let ao_comp = (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) / 6; // number of components for ao derivatives 
    let blksize = max_memory * 1_000_000 / 8 / ((ao_comp + 1) * num_basis);
    let blksize = blksize.min(num_grids).max(4); 
    // let blksize:usize = 1024;
    let block_settings = BlockSettings {
        ao_deriv,
        max_memory,
        blksize,
    };

    let device = &xc_data.device;
    
    // let mut nelec = vec![0.0; 1];
    // let mut exc_sum = vec![0.0; 1];
    let mut vxc = rt::zeros(([num_basis, num_basis, 3], device));

    let grids_iterator = gradient_method.block_loop(mol, grids, block_settings);
    // calculation starts here 
    let deriv = 1 as usize;          
    grids_iterator.into_iter().for_each(
        |block| 
        {
            let num_grids = block.weights.len();
            let loc_rho = if let Some(mo_coeffs) = &xc_data.mo_coeffs {
                eval_rho5_batch(&block.ao, xc_data.xc_type, mo_coeffs, xc_data.occ.as_ref().unwrap(), 1, num_grids)
            } else {
                eval_rho5_dm_only_batch(&block.ao, xc_data.xc_type, xc_data.dm, 1, num_grids)
            };

            let loc_xc_tensor = eval_xc_eff(xc_data.xc_code, xc_data.xc_params, xc_data.xc_type, 0, &loc_rho.raw(), num_grids, deriv);
            
            // let loc_rho = rt::asarray((rho_array, shape, device));
            let loc_vxc = if let Some(vxc) = loc_xc_tensor[1].as_ref() {
                rt::asarray(vxc)
            } else {
                panic!("Exchange-correlation potential is not provided for the given xc_type: {:?}", xc_data.xc_type);
            };
            // println!("[RIRHFGradient] In get_vxc, shape of loc_vxc: {:?}", loc_vxc);
            let loc_weights = rt::asarray((block.weights, device));
            let ao_shape = vec![num_basis, num_grids, ao_comp];
            let loc_ao = rt::asarray((&block.ao.data, ao_shape, device));

            let mut loc_wv = loc_vxc * &loc_weights.i((.., None));

            // println!("In get_vxc shape of loc_wv: {:?}", loc_wv.shape());

            match xc_data.xc_type {
                XCType::LDA => {
                    let aow = &loc_ao.i((.., .., 0)) * &loc_wv.i((None, .., 0)); 
                    // let aow = aow.t();
                    for ic in 0..3 {
                        vxc.i_mut((.., .., ic)).matmul_from(
                            &loc_ao.i((.., .., ic+1)), &aow.t(), 1.0, 1.0
                        );
                    }
                }, 
                XCType::GGA => {
                    loc_wv.i_mut((.., 0)).mul_assign(0.5);
                    gga_grad_sum(&mut vxc, loc_ao.view(), loc_wv.view());
                },
                XCType::MGGA => {
                    loc_wv.i_mut((.., 0)).mul_assign(0.5);
                    loc_wv.i_mut((.., 4)).mul_assign(0.5);
                    gga_grad_sum(&mut vxc, loc_ao.view(), loc_wv.view());
                    tau_grad_dot(&mut vxc, loc_ao.view(), loc_wv.view());
                },
                XCType::HF => {
                    unreachable!("HF gradient calculation does not support here in get_vxc");
                },
            }
        }
    );
    // nabla R = - nabla r 
    vxc *= -1.0;

    vxc

}


fn get_vxc_rayon(gradient_method: &RIRHFGradient, xc_data: &XCData, grids: &mut Grids, mol: &Molecule) -> Tsr<f64> {
    //In order to ensure the efficiency, we disable the openmp ability and re-open it in the end of subroutien
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
    let mut vmat = rt::zeros(([num_basis, num_basis, 3], device));

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
                eval_rho5_batch(&loc_ao, xc_data.xc_type, mo_coeffs, xc_data.occ.as_ref().unwrap(), 1, num_grids)
            } else {
                eval_rho5_dm_only_batch(&loc_ao, xc_data.xc_type, xc_data.dm, 1, num_grids)
            };
            // eval vxc on grids 
            let loc_xc_tensor = eval_xc_eff(xc_data.xc_code, xc_data.xc_params, xc_data.xc_type, 0, &loc_rho.raw(), num_grids, deriv);
            // let loc_rho = rt::asarray((rho_array, shape, device));
            let loc_vxc = if let Some(vxc) = loc_xc_tensor[1].as_ref() {
                rt::asarray(vxc)
            } else {
                panic!("Exchange-correlation potential is not provided for the given xc_type: {:?}", xc_data.xc_type);
            };
            // eval batch vxc_ao 
            let mut loc_vmat = rt::zeros(([num_basis, num_basis, 3], device));
            let loc_weights = rt::asarray((&grids.weights[range_grids.clone()], device));
            let ao_shape = vec![num_basis, num_grids, ao_comp];
            let loc_ao = rt::asarray((&loc_ao.data, ao_shape, device));
            
            let mut loc_wv = loc_vxc * &loc_weights.i((.., None));

            match xc_data.xc_type {
                XCType::LDA => {
                    let aow = &loc_ao.i((.., .., 0)) * &loc_wv.i((None, .., 0)); 
                    let aow = aow.t();
                    for ic in 0..3 {
                        loc_vmat.i_mut((.., .., ic)).matmul_from(
                            &loc_ao.i((.., .., ic+1)), &aow, 1.0, 1.0
                        );
                    }
                }, 
                XCType::GGA => {
                    loc_wv.i_mut((.., 0)).mul_assign(0.5);
                    gga_grad_sum(&mut loc_vmat, loc_ao.view(), loc_wv.view());
                },
                XCType::MGGA => {
                    loc_wv.i_mut((.., 0)).mul_assign(0.5);
                    loc_wv.i_mut((.., 4)).mul_assign(0.5);
                    gga_grad_sum(&mut loc_vmat, loc_ao.view(), loc_wv.view());
                    tau_grad_dot(&mut loc_vmat, loc_ao.view(), loc_wv.view());
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
            vmat += loc_vmat;
        }
    );
    omp_set_num_threads_wrapper(default_omp_num_threads);
    // nabla R = - nabla r
    vmat *= -1.0;
    vmat
}


fn get_vxc_rayon_new(gradient_method: &RIRHFGradient, xc_data: &XCData, grids: &mut Grids, mol: &Molecule, max_memory: usize) -> Tsr<f64> {
    let default_omp_num_threads = omp_get_num_threads_wrapper();

    let num_grids = grids.weights.len();
    let num_basis = mol.num_basis;

    let ao_deriv = match xc_data.xc_type {
        XCType::HF    => 1,
        XCType::LDA   => 1,
        XCType::GGA   => 2,
        XCType::MGGA  => 2,
    };
    let ao_comp = (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) / 6;
    let batch_size = 64;
    let blksize = (max_memory * 1_000_000 / 8 / ((ao_comp + 1) * num_basis * batch_size))
        .min(num_grids/batch_size+1).max(4) * batch_size;
    if gradient_method.flags.print_level >= 2 {
        println!("In get_vxc_rayon_new: BLKSIZE={:?}.", blksize);
    };
    
    let block_settings = BlockSettings { ao_deriv, max_memory, blksize };
    let device = &xc_data.device;
    let deriv = 1usize;

    let mut vxc = rt::zeros(([num_basis, num_basis, 3], device));
    let (sender, receiver) = channel();

    gradient_method.par_block_loop(mol, grids, block_settings)
        .for_each_with(sender, |s, block| {
            omp_set_num_threads_wrapper(1);

            let ng = block.weights.len();
            let loc_rho = if let Some(mo) = &xc_data.mo_coeffs 
            {
                eval_rho5_batch(&block.ao, xc_data.xc_type, mo, xc_data.occ.as_ref().unwrap(), 1, ng)
            } else {
                eval_rho5_dm_only_batch(&block.ao, xc_data.xc_type, xc_data.dm, 1, ng)
            };

            let xc_ten = eval_xc_eff(xc_data.xc_code, xc_data.xc_params, xc_data.xc_type, 0, &loc_rho.raw(), ng, deriv);
            let loc_vxc = rt::asarray(xc_ten[1].as_ref().expect("vxc not provided for this xc_type"));

            let loc_w = rt::asarray((block.weights, device));
            let ao_shape = vec![num_basis, ng, ao_comp];
            let loc_ao = rt::asarray((&block.ao.data, ao_shape, device));

            let mut wv = loc_vxc * &loc_w.i((.., None));
            let mut loc_vmat = rt::zeros(([num_basis, num_basis, 3], device));

            match xc_data.xc_type {
                XCType::LDA => {
                    let aow = &loc_ao.i((.., .., 0)) * &wv.i((None, .., 0)); 
                    let aow = aow.t();
                    for ic in 0..3 {
                        loc_vmat.i_mut(ic).matmul_from(
                            &loc_ao.i((.., .., ic + 1)), &aow, 1.0, 1.0);
                    }
                }
                XCType::GGA => {
                    wv.i_mut((.., 0)).mul_assign(0.5);
                    gga_grad_sum(&mut loc_vmat, loc_ao.view(), wv.view());
                }
                XCType::MGGA => {
                    wv.i_mut((.., 0)).mul_assign(0.5);
                    wv.i_mut((.., 4)).mul_assign(0.5);
                    gga_grad_sum(&mut loc_vmat, loc_ao.view(), wv.view());
                    tau_grad_dot(&mut loc_vmat, loc_ao.view(), wv.view());
                }
                XCType::HF => unreachable!("HF gradient not supported in get_vxc_rayon_new"),
            }
            s.send(loc_vmat).unwrap();
        });

    receiver.into_iter().for_each(|m| vxc += m);

    omp_set_num_threads_wrapper(default_omp_num_threads);
    vxc *= -1.0;
    vxc
}


// #[cfg(test)]
// #[allow(non_snake_case)]
// mod debug {
//     use super::*;
//     use crate::ctrl_io::InputKeywords;
//     use crate::scf_io::scf_without_build;

//     #[test]
//     fn test_nh3() {
//         let scf_data = initialize_nh3();
//         let time = std::time::Instant::now();
//         let scf_grad = test_with_scf(&scf_data);
//         println!("Time elapsed: {:?}", time.elapsed());

//         let de = scf_grad.result.get("de").unwrap().clone();
//         let de = rt::asarray((de.data, de.size));
//         #[rustfmt::skip]
//         let de_ref = vec![
//             -0.1137786866, -0.1161365056, -0.1125150713,
//              0.0004215289,  0.0659156136,  0.0553809300,
//              0.0630962392,  0.0488708661, -0.0140011525,
//              0.0502609185,  0.0013500258,  0.0711352938,
//         ];
//         let de_ref = rt::asarray((&de_ref, [3, 4]));
//         println!("Maximum Error {:?}", (&de_ref - &de).abs().max_all());
//         assert!((de_ref - de).abs().max_all() < 1.0e-5);
//     }


//     fn test_with_scf(scf_data: &SCF) -> RIRHFGradient {
//         let mut scf_grad = RIRHFGradient::new(scf_data);
//         scf_grad.calc_rks();

//         println!("=== de ===");
//         let de = scf_grad.result.get("de").unwrap().clone();
//         let de = rt::asarray((de.data, de.size));
//         println!("{:12.6}", de.t());

//         println!("=== de_nuc ===");
//         scf_grad.result.get("de_nuc").map(|de| {
//             let de = rt::asarray((&de.data, de.size));
//             println!("{:12.6}", de.t());
//         });

//         println!("=== de_ovlp ===");
//         scf_grad.result.get("de_ovlp").map(|de| {
//             let de = rt::asarray((&de.data, de.size));
//             println!("{:12.6}", de.t());
//         });

//         println!("=== de_hcore ===");
//         scf_grad.result.get("de_hcore").map(|de| {
//             let de = rt::asarray((&de.data, de.size));
//             println!("{:12.6}", de.t());
//         });

//         println!("=== de_j ===");
//         scf_grad.result.get("de_j").map(|de| {
//             let de = rt::asarray((&de.data, de.size));
//             println!("{:12.6}", de.t());
//         });

//         println!("=== de_k ===");
//         scf_grad.result.get("de_k").map(|de| {
//             let de = rt::asarray((&de.data, de.size));
//             println!("{:12.6}", de.t());
//         });

//         println!("=== de_jaux ===");
//         scf_grad.result.get("de_jaux").map(|de| {
//             let de = rt::asarray((&de.data, de.size));
//             println!("{:12.6}", de.t());
//         });

//         println!("=== de_kaux ===");
//         scf_grad.result.get("de_kaux").map(|de| {
//             let de = rt::asarray((&de.data, de.size));
//             println!("{:12.6}", de.t());
//         });

//         return scf_grad;
//     }

//     fn initialize_nh3() -> SCF {
//         let input_token = r##"
// [ctrl]
//      print_level =          2
//      xc =                   "pbe"
//      basis_path =           "basis-set-pool/def2-SVP"
//      auxbas_path =          "basis-set-pool/def2-SVP-JKFIT"
//      basis_type =           "spheric"
//      eri_type =             "ri-v"
//      auxbas_type =          "spheric"
//      guessfile =            "none"
//      chkfile =              "none"
//      charge =               0.0
//      spin =                 1.0
//      spin_polarization =    false
//      auxbasis_response =    true
//      external_grids =       "none"
//      initial_guess=         "sad"
//      mixer =                "diis"
//      num_max_diis =         8
//      start_diis_cycle =     3
//      mix_param =            0.8
//      max_scf_cycle =        100
//      scf_acc_rho =          1.0e-10
//      scf_acc_eev =          1.0e-10
//      scf_acc_etot =         1.0e-11
//      num_threads =          16

// [geom]
//     name = "NH3"
//     unit = "Angstrom"
//     position = """
//         N  0.0  0.0  0.0
//         H  0.0  1.5  1.0
//         H  1.4  1.1  0.0
//         H  1.2  0.0  1.3
//     """
// "##;
//         let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
//         let (mut ctrl, mut geom) = InputKeywords::parse_ctl_from_json(&keys).unwrap();
//         let mol = Molecule::build_native(ctrl, geom, None).unwrap();
//         let mut scf_data = scf_io::SCF::build(mol, &None);
//         scf_without_build(&mut scf_data, &None);
//         return scf_data;
//     }

// }


