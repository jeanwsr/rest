use std::sync::{Arc, Mutex};
use rstsr::prelude::*;
use rest_tensors::{MatrixFull, RIFull};
use rest_tensors::matrix_blas_lapack::{ _einsum_01_serial, _einsum_02_serial};
use tensors::matrix_blas_lapack::{_dgemm};
use crate::scf_io::SCF;
use crate::molecule_io::Molecule;
use crate::basis_io::{spheric_gto_deriv_batch_serial};
use crate::dft::{Grids, DFA4REST};
use crate::dft::xc_deriv::XCType;
use crate::scf_io::util::occupied_orbital_count_with_threshold;
use crate::dft::libxc_itrf::eval_xc_eff;


pub fn eval_ao_batch(mol:&Molecule, coords:&[[f64; 3]], ao_deriv:usize, num_grids:usize) -> RIFull<f64> {
    // let default_omp_num_threads = unsafe {utilities::openblas_get_num_threads()};
    // let default_omp_num_threads = utilities::omp_get_num_threads_wrapper();
    // utilities::omp_set_num_threads_wrapper(1);

    let n_components_deriv = (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) / 6;
    
    let mut loc_ao = RIFull::new([mol.num_basis, num_grids, n_components_deriv], 0.0);

    mol.basis4elem.iter()
    .zip(mol.geom.position.iter_columns_full())
    .for_each(
        |(elem, geom)|
        {
            let ind_glb_bas = elem.global_index.0;
            let loc_num_bas = elem.global_index.1;
            let start = ind_glb_bas;
            let end = start + loc_num_bas;
            let mut tmp_geom = [0.0; 3];
            tmp_geom.iter_mut().zip(geom.iter()).for_each(|value| {*value.0 = *value.1});
            let tab_ao_elem = spheric_gto_deriv_batch_serial(&coords, &tmp_geom, elem, ao_deriv);

            for ic in 0..n_components_deriv {
                let gto_ic = tab_ao_elem.get(ic).unwrap();
                loc_ao.copy_from_matr(start..end, 0..num_grids, ic, 0, 
                   gto_ic, 0..loc_num_bas, 0..num_grids);
            }
            // loc_ao.copy_from_ri(start..end, 0..num_grids, 0..n_components_cart, &tab_ao_elem, 0..loc_num_bas, 0..num_grids, 0..n_components_cart);
        }
    );

    loc_ao
}


pub fn eval_rho5_batch(ao:&RIFull<f64>, xc_type:XCType, mo:&Vec<MatrixFull<f64>>, occ:&Vec<Vec<f64>>, spin_channel:usize, num_grids:usize) -> Tensor<f64> {
    /*
        Evaluate density on a batch of ao grids value for given mo coefficients and occupation.
        Args:
            ao: gto value on grids [num_grids, num_basis, nderiv],
                nderiv = 1 for ao0, 4 for ao0, ao_x, ao_y, ao_z, 10 for 2nd order derivatives (symmetric)
            xc_type: exchange-correlation dfa type
            mo: orbital coeffients [num_basis, num_state]
            occ: occupation number [num_state]
            spin_channel: number of spins, 1 for Restricted, 2 for Unrestricted
            range_grids: range of grids
        Return:
            rho [num_grids, nvar, spin_channel]
            where nvar = 1 for LDA, 4 for GGA, 5 for MGGA 
            tabulated as rho, rho_x, rho_y, rho_z, tau
        Note:
            laplacian is not implemented
    */
    let num_basis = mo[0].size.get(0).unwrap();
    let num_state = mo[0].size.get(1).unwrap();

    let nvar = match xc_type {
        XCType::HF => 1, 
        XCType::LDA => 1,
        XCType::GGA => 4,
        XCType::MGGA => 5,
    };
    let mut cur_rho = RIFull::new([num_grids, nvar, spin_channel], 0.0);

    let ao0 = &ao.get_reducing_matrix(0).unwrap();

    for i_spin in 0..spin_channel {
        let mo_s = mo.get(i_spin).unwrap();
        let homo_s = occ[i_spin].iter()
            .enumerate()
            .filter(|(i,occ)| **occ >=1.0e-6)
            .map(|(i,occ)| i).max();
        let mut occ_s = if let Some(homo_s) = homo_s {
            occ.get(i_spin).unwrap()[0..homo_s+1].iter().map(|occ| occ.sqrt()).collect::<Vec<f64>>()
        } else {
            // In this case, no electrons in the i_spin channel, for which homo_s = None
            vec![]
        };

        let num_occ = occ_s.len();
        let mut wmo = _einsum_01_serial(&mo_s.to_matrixfullslice(), &occ_s);
        let mut tmo = MatrixFull::new([num_occ, num_grids], 0.0);
        // tmo = C.T matmul ao (half transform)
        _dgemm(
            &wmo, (0..wmo.size[0], 0..wmo.size[1]), 'T', 
            ao0, (0..ao0.size[0], 0..num_grids), 'N',
            &mut tmo, (0..wmo.size[1], 0..num_grids),
            1.0, 0.0
        );
        // spin case: rho_s = tmo * tmo (C_ug * C_vg * phi_ug * phi_vg => rho_g) 
        let rho_s = _einsum_02_serial(&tmo.to_matrixfullslice(), &tmo.to_matrixfullslice());
        // cur_rho[ispin][0] = rho_s
        cur_rho.get_reducing_matrix_mut(i_spin).unwrap()
        .iter_mut_j(0)
        .zip(rho_s.iter())
        .for_each(
            |(to, from)| {*to = *from}
        );
        if nvar > 1 { // gga and upper case 
            for ic in 0..3 { // x, y, z
                let mut tmop = MatrixFull::new([num_occ, num_grids], 0.0);
                let aop_ic = &ao.get_reducing_matrix(ic+1).unwrap();
                _dgemm(
                    &wmo, (0..wmo.size[0], 0..wmo.size[1]), 'T',
                    aop_ic, (0..aop_ic.size[0], 0..num_grids), 'N',
                    &mut tmop, (0..wmo.size[1], 0..num_grids),
                    1.0, 0.0
                );
                let rhop_ic_s = _einsum_02_serial(&tmop.to_matrixfullslice(), &tmo.to_matrixfullslice());
                // cur_rho[ispin][ic+1] = rhop_ic_s
                cur_rho.get_reducing_matrix_mut(i_spin).unwrap()
                .iter_mut_j(ic + 1)
                .zip(rhop_ic_s.iter())
                .for_each(
                    |(to, from)| {*to = *from * 2.0}
                );
                if nvar == 5 { // mgga
                    let tau_s = _einsum_02_serial(&tmop.to_matrixfullslice(), &tmop.to_matrixfullslice());
                    // cur_rho[ispin][5] = tau_s
                    // tau = 1/2 * |grad phi|^2 sum over x,y,z
                    cur_rho.get_reducing_matrix_mut(i_spin).unwrap()
                    .iter_mut_j(4)
                    .zip(tau_s.iter())
                    .for_each(
                        |(to, from)| {*to += *from*0.5}
                    );
                }
            }
        }
    }
    let rho_ten = rt::asarray((&cur_rho.data, [num_grids, nvar, spin_channel])).to_owned();
    rho_ten

}


pub fn eval_rho5_dm_only_batch(ao:&RIFull<f64>, xc_type:XCType, dms:&Vec<MatrixFull<f64>>, spin_channel:usize, num_grids:usize) -> Tensor<f64> {
    /*
        Evaluate density on grids for a given set of density matrices.
        Args:
            ao: gto value on grids [num_grids, num_basis, nderiv] (column major),
                nderiv = 1 for ao0, 4 for ao0, ao_x, ao_y, ao_z, 10 for 2nd order derivatives (symmetric)
            xc_method: exchange-correlation dfa type
            dms: density matrices of shape [num_basis, num_bais, spin_channel, nsets] 
            spin_channel: number of spins, 1 for Restricted, 2 for Unrestricted
            range_grids: range of grids
        Return:
            rho [num_grids, nvar, spin_channel]
            where nvar = 1 for LDA, 4 for GGA, 5 for MGGA 
            tabulated as rho, rho_x, rho_y, rho_z, tau
        Note:
            laplacian is not implemented
    */

    let num_basis = dms[0].size[0];

    let nvar = match xc_type {
        XCType::HF => 1, 
        XCType::LDA => 1,
        XCType::GGA => 4,
        XCType::MGGA => 5,
    };

    let mut cur_rho = RIFull::new([num_grids, nvar, spin_channel], 0.0);
    
    let ao0 = &ao.get_reducing_matrix(0).unwrap();

    for i_spin in 0..spin_channel {
        // rho 
        let dm_s = &dms[i_spin];
        let mut wao = MatrixFull::new([num_basis, num_grids],0.0);
        _dgemm(
            dm_s, (0..num_basis, 0..num_basis), 'N',
            ao0, (0..num_basis, 0..num_grids), 'N',
            &mut wao,  (0..num_basis, 0..num_grids), 
            1.0, 0.0
        );
        ao0.iter_columns(0..num_grids).unwrap()
        .zip(wao.iter_columns_full())
        .map(|(ao_r,wao_r)| (ao_r, wao_r))
        .zip(cur_rho.get_reducing_matrix_mut(i_spin).unwrap().iter_column_mut(0))
        .for_each(
            |((ao_r, wao_r), cur_rho_s)| 
            {
                *cur_rho_s = wao_r.iter().zip(ao_r.iter()).fold(0.0, |acc, (a,b)| {acc + a*b})
            }
        );
        if nvar > 1 {
            for ic in 0..3usize {
                let mut aop_ic = &ao.get_reducing_matrix(ic + 1).unwrap();
                let mut wao = MatrixFull::new([num_basis, num_grids],0.0);
                _dgemm(
                    dm_s, (0..num_basis, 0..num_basis), 'N',
                    aop_ic, (0..num_basis, 0..num_grids), 'N',
                    &mut wao, (0..num_basis, 0..num_grids), 1.0, 0.0
                );
            
                ao0.iter_columns(0..num_grids).unwrap()
                .zip(wao.iter_columns_full())
                .map(|(ao_r, wao_r)| (ao_r, wao_r))
                .zip(cur_rho.get_reducing_matrix_mut(i_spin).unwrap().iter_column_mut(ic + 1))
                .for_each(
                    |((ao_r, wao_r), cur_rhop_r)| 
                    {
                        *cur_rhop_r = 2.0 * wao_r.iter().zip(ao_r.iter()).fold(0.0, |acc,(wao,ao)| {acc + wao*ao})
                    }
                );
                if nvar == 5 {
                    aop_ic.iter_columns(0..num_grids).unwrap()
                    .zip(wao.iter_columns_full())
                    .map(|(aop_r, wao_r)|(aop_r, wao_r))
                    .zip(cur_rho.get_reducing_matrix_mut(i_spin).unwrap().iter_column_mut(4))
                    .for_each(
                        |((aop_r, wao_r), cur_tau_s)|
                        {
                            *cur_tau_s += 0.5 * wao_r.iter().zip(aop_r.iter()).fold(0.0, |acc, (a,b)| {acc + a*b})
                        }
                    );
                }
            }
        }
    } // end spin case 
    let rho_ten = rt::asarray((&cur_rho.data, [num_grids, nvar, spin_channel])).to_owned();
    rho_ten
}


pub fn eval_rho5_spin_batch(ao:&RIFull<f64>, xc_type:XCType, mo:&MatrixFull<f64>, occ:&Vec<f64>, num_grids:usize) -> Tensor<f64> {
    /*
        Evaluate density on a batch of ao grids value for given mo coefficients and occupation in one spin channel.
        Args:
            ao: gto value on grids [num_grids, num_basis, nderiv],
                nderiv = 1 for ao0, 4 for ao0, ao_x, ao_y, ao_z, 10 for 2nd order derivatives (symmetric)
            xc_type: exchange-correlation dfa type
            mo: orbital coeffients [num_basis, num_state]
            occ: occupation number [num_state]
            range_grids: range of grids
        Return:
            rho [num_grids, nvar]
            where nvar = 1 for LDA, 4 for GGA, 5 for MGGA 
            tabulated as rho, rho_x, rho_y, rho_z, tau
        Note:
            laplacian is not implemented
    */

    let num_basis = mo.size.get(0).unwrap();
    let num_state = mo.size.get(1).unwrap();

    let nvar = match xc_type {
        XCType::HF => 1, 
        XCType::LDA => 1,
        XCType::GGA => 4,
        XCType::MGGA => 5,
    };
    let mut cur_rho = MatrixFull::new([num_grids, nvar], 0.0);

    let ao0 = &ao.get_reducing_matrix(0).unwrap();

    
    
    let num_occ = occupied_orbital_count_with_threshold(occ, 1.0e-6);
    let mut occ_tmp = occ[0..num_occ].iter().map(|occ| occ.sqrt()).collect::<Vec<f64>>();

    let num_occ = occ_tmp.len();
    let mut wmo = _einsum_01_serial(&mo.to_matrixfullslice(), &occ_tmp);
    let mut tmo = MatrixFull::new([num_occ, num_grids], 0.0);
    // tmo = C.T matmul ao (half transform)
    _dgemm(
        &wmo, (0..wmo.size[0], 0..wmo.size[1]), 'T', 
        ao0, (0..ao0.size[0], 0..num_grids), 'N',
        &mut tmo, (0..wmo.size[1], 0..num_grids),
        1.0, 0.0
    );
    // rho_0 = tmo * tmo (C_ug * C_vg * phi_ug * phi_vg => rho_g) 
    let rho_0 = _einsum_02_serial(&tmo.to_matrixfullslice(), &tmo.to_matrixfullslice());
    // cur_rho[0] = rho_0
    cur_rho.iter_column_mut(0).zip(rho_0.iter())
    .for_each(
        |(to, from)| {*to = *from}
    );
        
    if nvar > 1 { // gga and upper case 
        for ic in 0..3 { // x, y, z
            let mut tmop = MatrixFull::new([num_occ, num_grids], 0.0);
            let aop_ic = &ao.get_reducing_matrix(ic+1).unwrap();
            _dgemm(
                &wmo, (0..wmo.size[0], 0..wmo.size[1]), 'T',
                aop_ic, (0..aop_ic.size[0], 0..num_grids), 'N',
                &mut tmop, (0..wmo.size[1], 0..num_grids),
                1.0, 0.0
            );
            let rhop_ic = _einsum_02_serial(&tmop.to_matrixfullslice(), &tmo.to_matrixfullslice());
            // cur_rho[ic+1] = rhop_ic
            cur_rho.iter_column_mut(ic + 1).zip(rhop_ic.iter())
            .for_each(
                |(to, from)| {*to = *from * 2.0}
            );
            if nvar == 5 { // mgga
                let tau = _einsum_02_serial(&tmop.to_matrixfullslice(), &tmop.to_matrixfullslice());
                // cur_rho[5] = tau
                // tau = 1/2 * |grad phi|^2 sum over x,y,z
                cur_rho.iter_column_mut(4).zip(tau.iter())
                .for_each(
                    |(to, from)| {*to += *from*0.5}
                );
            }
        }
    }

    let rho_ten = rt::asarray((cur_rho.data, [num_grids, nvar]));
    rho_ten 
}


pub fn eval_rho5_spin_dm_only_batch(ao:&RIFull<f64>, xc_type:XCType, dm:&MatrixFull<f64>, num_grids:usize) -> Tensor<f64> {
    let num_basis = dm.size[0];

    let nvar = match xc_type {
        XCType::HF => 1, 
        XCType::LDA => 1,
        XCType::GGA => 4,
        XCType::MGGA => 5,
    };

    let mut cur_rho = MatrixFull::new([num_grids, nvar], 0.0);
    let ao0 = &ao.get_reducing_matrix(0).unwrap();
    // rho0 
    let mut wao = MatrixFull::new([num_basis, num_grids],0.0);
    _dgemm(
        dm, (0..num_basis, 0..num_basis), 'N',
        ao0, (0..num_basis, 0..num_grids), 'N',
        &mut wao,  (0..num_basis, 0..num_grids), 
        1.0, 0.0
    );
    ao0.iter_columns(0..num_grids).unwrap()
    .zip(wao.iter_columns_full())
    .map(|(ao_r,wao_r)| (ao_r, wao_r))
    .zip(cur_rho.iter_column_mut(0))
    .for_each(
        |((ao_r, wao_r), cur_rho_0)| 
        {
            *cur_rho_0 = wao_r.iter().zip(ao_r.iter()).fold(0.0, |acc, (a,b)| {acc + a*b})
        }
    );
    // gga and upper case
    if nvar > 1 {
        // nabla rho 
        for ic in 0..3usize {
            let mut aop_ic = &ao.get_reducing_matrix(ic + 1).unwrap();
            let mut wao = MatrixFull::new([num_basis, num_grids],0.0);
            _dgemm(
                dm, (0..num_basis, 0..num_basis), 'N',
                aop_ic, (0..num_basis, 0..num_grids), 'N',
                &mut wao, (0..num_basis, 0..num_grids), 1.0, 0.0
            );
            ao0.iter_columns(0..num_grids).unwrap()
            .zip(wao.iter_columns_full())
            .map(|(ao_r, wao_r)| (ao_r, wao_r))
            .zip(cur_rho.iter_column_mut(ic + 1))
            .for_each(
                |((ao_r, wao_r), cur_rhop_r)| 
                {
                    *cur_rhop_r = 2.0 * wao_r.iter().zip(ao_r.iter()).fold(0.0, |acc,(wao,ao)| {acc + wao*ao})
                }
            );
            // tau 
            if nvar == 5 {
                aop_ic.iter_columns(0..num_grids).unwrap()
                .zip(wao.iter_columns_full())
                .map(|(aop_r, wao_r)|(aop_r, wao_r))
                .zip(cur_rho.iter_column_mut(4))
                .for_each(
                    |((aop_r, wao_r), cur_tau)|
                    {
                        *cur_tau += 0.5 * wao_r.iter().zip(aop_r.iter()).fold(0.0, |acc, (a,b)| {acc + a*b})
                    }
                );
            }
        }
    }

    let rho_ten = rt::asarray((cur_rho.data, [num_grids, nvar]));
    rho_ten
}

#[derive(Debug, Clone)]
pub struct BlockSettings {
    pub ao_deriv: usize,
    pub max_memory: usize, // in MB
    pub blksize: usize, // block size of batched grids
}

pub struct BlockData<'a> {
    pub coords: &'a [[f64; 3]],
    pub weights: &'a [f64],
    pub ao: RIFull<f64>, // atomic orbitals evaluated on grids
}

pub struct XCData<'a> {
    pub xc_type: XCType,
    pub xc_code: &'a Vec<usize>,
    pub xc_params: &'a Vec<f64>,
    pub device: DeviceOpenBLAS,
    pub dm: &'a Vec<MatrixFull<f64>>, 
    pub mo_coeffs: Option<Vec<MatrixFull<f64>>>,
    pub occ: Option<Vec<Vec<f64>>>,
}


pub struct GridIterator<'a> {
    mol: &'a Molecule,
    grids: &'a Grids,
    settings: BlockSettings,
    current: usize,
    total_grids: usize,
}

impl<'a> GridIterator<'a> {
    pub fn new(
        mol: &'a Molecule,
        grids: &'a Grids,
        settings: BlockSettings,
    ) -> Self {
        let total_grids = grids.weights.len();
        Self {
            mol,
            grids,
            settings,
            current: 0,
            total_grids,
        }
    }
}

impl<'a> Iterator for GridIterator<'a> {
    type Item = BlockData<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.total_grids {
            return None;
        }
        let start = self.current;
        let end = (start + self.settings.blksize).min(self.total_grids);
        self.current = end;

        let coords = &self.grids.coordinates[start..end];
        let weights = &self.grids.weights[start..end];
        let num_grids = weights.len();
        let ao = eval_ao_batch(self.mol, coords, self.settings.ao_deriv, num_grids);

        Some(BlockData{
            coords: &self.grids.coordinates[start..end],
            weights: &self.grids.weights[start..end],
            ao: ao,
        })
    }
}

pub struct GridParallelIterator<'a> {
    mol: &'a Molecule,
    grids: &'a Grids,
    settings: BlockSettings,
    current: Arc<Mutex<usize>>, 
    total_grids: usize,
}

impl<'a> GridParallelIterator<'a> {
    pub fn new(
        mol: &'a Molecule,
        grids: &'a Grids,
        settings: BlockSettings,
    ) -> Self {
        let total_grids = grids.coordinates.len() / 3;
        Self {
            mol,
            grids,
            settings,
            current: Arc::new(Mutex::new(0)),
            total_grids,
        }
    }
}

// impl<'a> ParallelIterator for GridParallelIterator<'a> {
//     type Item = BlockData<'a>;

//     fn drive_unindexed<C>(self, consumer: C) -> C::Result
//     where
//         C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
//     {
//         rayon::iter::plumbing::bridge(self, consumer)
//     }

//     fn opt_len(&self) -> Option<usize> {
//         Some((self.total_grids + self.settings.blksize - 1) / self.settings.blksize)
//     }
// }

// impl<'a> rayon::iter::plumbing::ParallelIteratorBridge for GridParallelIterator<'a> {
//     fn fold_with<F>(self, folder: F) -> F
//     where
//         F: rayon::iter::plumbing::Folder<Self::Item>,
//     {
//         let blksize = self.settings.blksize;
//         let mut current = self.current.lock().unwrap();
            
//         while *current < self.total_grids {
//             let start = *current;
//             let end = (start + blksize).min(self.total_grids);
//             *current = end; 

//             drop(current);
                
//             let block = BlockData {
//                 ao: eval_ao_batch(
//                     self.mol,
//                     &self.grids.coordinates.slice(s![start..end]),
//                     self.settings.ao_deriv,
//                 ),
//                 weights: self.grids.weights.slice(s![start..end]),
//                 coords: self.grids.coordinates.slice(s![start..end]),
//             };

//             let folder = folder.consume(block);
//             current = self.current.lock().unwrap();
//         }
//         folder 
//     }
// }

pub trait NumInt<'a> {

    fn gen_xc_data(&'a self, scf_data:&'a SCF, spin:usize) -> XCData<'a>;

    fn get_vxc(&'a self, xc_data: &'a XCData, grids: &mut Grids, mol: &Molecule, spin: usize) -> (Vec<f64>, Vec<f64>, Vec<MatrixFull<f64>>);

    fn get_fxc(&'a self, xc_data0: &'a XCData, xc_data1: &'a XCData, grids: &mut Grids, mol: &Molecule, spin: usize) -> Vec<MatrixFull<f64>>;

    fn block_loop(
        &'a self,
        mol: &'a Molecule,
        grids: &'a mut Grids,
        settings: BlockSettings,
    ) -> GridIterator<'a> {
        GridIterator::new(mol, grids, settings)
    }

    fn par_blook_loop(
        &'a self,
        mol: &'a Molecule,
        grids: &'a mut Grids,
        settings: BlockSettings,
    ) -> GridParallelIterator<'a> {
        GridParallelIterator::new(mol, grids, settings)
    }
}


impl<'a> NumInt<'a> for DFA4REST {
    fn gen_xc_data(&'a self, scf_data: &'a SCF, spin: usize) -> XCData<'a> {
        let xc_type = if self.use_kinetic_density() {
            XCType::MGGA
        } else if self.use_density_gradient() {
            XCType::GGA
        } else {
            XCType::LDA
        };

        let xc_code = &self.dfa_compnt_scf;
        let xc_params = &self.dfa_paramr_scf;

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

    fn get_fxc(
        &'a self,
        xc_data0: &'a XCData,
        xc_data1: &'a XCData,
        grids: &mut Grids,
        mol: &Molecule,
        spin: usize,
    ) -> Vec<MatrixFull<f64>> {
        todo!("get_fxc has not yet implemented for DFA4REST");
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
            XCType::HF => 0usize,
            XCType::LDA => 0usize,
            XCType::GGA => 1usize,
            XCType::MGGA => 1usize,
        };
        let max_memory = if let Some(max_memory) = mol.ctrl.max_memory {
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
        
        let mut nelec = vec![0.0; spin+1];
        let mut exc_sum = vec![0.0; spin+1];
        let mut vxc = rt::zeros(([num_basis, num_basis, spin + 1], device));
        let mut vxc_mgga_temp = match xc_data.xc_type {
            XCType::MGGA => Some(rt::zeros(([num_basis, num_basis, spin + 1], device))),
            _ => None,
        };
        let grids_iterator = self.block_loop(mol, grids, block_settings);
        // calculation starts here 
        let deriv = 1usize;          
        grids_iterator.into_iter().for_each(
            |block| 
            {
                let num_grids = block.weights.len();
                let loc_rho = if let Some(mo_coeffs) = &xc_data.mo_coeffs {
                    eval_rho5_batch(&block.ao, xc_data.xc_type, mo_coeffs, xc_data.occ.as_ref().unwrap(), spin, num_grids)
                } else {
                    eval_rho5_dm_only_batch(&block.ao, xc_data.xc_type, xc_data.dm, spin, num_grids)
                };
                let shape = loc_rho.shape().clone();

                let rho_array = loc_rho.into_vec();
                let loc_xc_tensor = eval_xc_eff(xc_data.xc_code, xc_data.xc_params, xc_data.xc_type, spin, &rho_array, num_grids, deriv);

                let loc_rho = rt::asarray((rho_array, shape, device));
                // let loc_exc = rt::asarray((&loc_xc_tensor[0].unwrap()));
                // let loc_vxc = rt::asarray((&loc_xc_tensor[1].unwrap()));

                let loc_exc = if let Some(exc) = loc_xc_tensor[0].as_ref() {
                    rt::asarray(exc)
                } else {
                    panic!("Exchange-correlation energy is not provided for the given xc_type: {:?}", xc_data.xc_type);
                };

                let loc_vxc = if let Some(vxc) = loc_xc_tensor[1].as_ref() {
                    rt::asarray(vxc)
                } else {
                    panic!("Exchange-correlation potential is not provided for the given xc_type: {:?}", xc_data.xc_type);
                };

                let loc_weights = rt::asarray((block.weights, device));

                let ao_shape = vec![num_basis, num_grids, ao_comp];
                let loc_ao = rt::asarray((&block.ao.data, ao_shape, device));
                

                // let loc_exc = loc_xc_tensor[0].as_ref().unwrap();
                // let loc_vxc = loc_xc_tensor[1].as_ref().unwrap();

                match spin {
                    0 => {
                        let loc_rho0 = loc_rho.i((0..num_grids, 0, 0));
                        let loc_den = loc_rho0 * &loc_weights;
                        let loc_nelec = rt::sum(&loc_den);
                        nelec[0] += loc_nelec;
                        let loc_exc_sum = (loc_exc * &loc_den).sum();
                        exc_sum[0] += loc_exc_sum;

                        let mut loc_wv = loc_vxc * &loc_weights.i((.., None)); 

                        match xc_data.xc_type {
                            XCType::LDA => {
                                let aow = &loc_ao.i((.., .., 0)) * &loc_wv; 
                                vxc.i_mut((.., .., 0)).matmul_from(
                                    &aow, &loc_ao.i((.., .., 0)).t(), 1.0, 1.0
                                );

                            },
                            XCType::GGA => {
                                loc_wv.i_mut((.., 0)).mul_assign(0.5);
                                
                                let aow = &loc_ao * &loc_wv.i((.., None, ..));
                                for ic in 0..=3 {
                                    vxc.i_mut(0).matmul_from(
                                        &aow.i((.., .., ic)), &loc_ao.i((.., .., ic)).t(), 1.0, 1.0
                                    );
                                }
                            },
                            XCType::MGGA => {
                                loc_wv.i_mut((.., 0)).mul_assign(0.5);
                                loc_wv.i_mut((.., 4)).mul_assign(0.5);

                                for ic in 0..=3 {
                                    let aow = &loc_ao.i((.., .., ic)) * &loc_wv.i((.., None, ic));
                                    vxc.i_mut(0).matmul_from(
                                        &loc_ao.i((.., .., 0)), &aow.t(), 1.0, 1.0
                                    );
                                }
                                let Some(v1) = vxc_mgga_temp.as_mut() else {
                                    panic!("vxc_mgga_temp is not initialized for MGGA type");
                                };
                                for ic in 0..=3 {
                                    let vtau_ket = &loc_ao.i((.., .., ic+1)) * &loc_wv.i((.., None, 4));
                                    v1.i_mut(0).matmul_from(
                                        &loc_ao.i((.., .., ic+1)), &vtau_ket.t(), 1.0, 1.0
                                    );
                                }
                            },
                            XCType::HF => {},
                        }
                    },
                    1 => {
                        let loc_rho_a = loc_rho.i((0..num_grids, 0, 0));
                        let loc_rho_b = loc_rho.i((0..num_grids, 0, 1));
                        let loc_den_a = loc_rho_a * &loc_weights;
                        let loc_den_b = loc_rho_b * &loc_weights;
                        let loc_nelec_a = rt::sum(&loc_den_a);
                        let loc_nelec_b = rt::sum(&loc_den_b);
                        nelec[0] += loc_nelec_a;
                        nelec[1] += loc_nelec_b;
                        let loc_exc_sum_a = (&loc_exc * &loc_den_a).sum();
                        let loc_exc_sum_b = (&loc_exc * &loc_den_b).sum();
                        exc_sum[0] += loc_exc_sum_a;
                        exc_sum[1] += loc_exc_sum_b;

                        //loc_vxc shape [num_grids, nderiv ,spin] nderiv = 1, 4, 5 for LDA (vrho), GGA (vrho, vnabla_rho), MGGA (.., vtau)
                        let mut loc_wv = &loc_vxc * &loc_weights.i((.., None, None)); 

                        match xc_data.xc_type {
                            XCType::LDA => {
                                let loc_wv_a = loc_wv.i((.., 0, 0));
                                let loc_wv_b = loc_wv.i((.., 0, 1));
                                let aow_a = &loc_ao.i((.., .., 0)) * &loc_wv_a.i((.., None)); 
                                let aow_b = &loc_ao.i((.., .., 0)) * &loc_wv_b.i((.., None));
                                vxc.i_mut((.., .., 0)).matmul_from(
                                    &aow_a, &loc_ao.i((.., .., 0)).t(), 1.0, 1.0
                                );
                                vxc.i_mut((.., .., 1)).matmul_from(
                                    &aow_b, &loc_ao.i((.., .., 0)).t(), 1.0, 1.0
                                );

                            },
                            XCType::GGA => {
                                loc_wv.i_mut((.., 0, ..)).mul_assign(0.5);
                                let loc_wv_a = loc_wv.i((.., .., 0));
                                let aow_a = &loc_ao.i((.., .., 0..4)) * &loc_wv_a.i((.., None, ..));
                                let loc_wv_b = loc_wv.i((.., .., 1));
                                let aow_b = &loc_ao .i((.., .., 0..4)) * &loc_wv_b.i((.., None, ..));
                                for ic in 0..4 {
                                    vxc.i_mut((.., .., 0)).matmul_from(
                                        &loc_ao.i((.., .., 0)), &aow_a.i((.., .., ic)).t(), 1.0, 1.0
                                    );
                                    vxc.i_mut((.., .., 1)).matmul_from(
                                        &loc_ao.i((.., .., 0)), &aow_b.i((.., .., ic)).t(), 1.0, 1.0
                                    );
                                }
                            },
                            XCType::MGGA => {
                                loc_wv.i_mut((.., 0, ..)).mul_assign(0.5);
                                loc_wv.i_mut((.., 4, ..)).mul_assign(0.5);
                                let Some(v1) = vxc_mgga_temp.as_mut() else {
                                    panic!("vxc_mgga_temp is not initialized for MGGA type");
                                };

                                let loc_wv_a = loc_wv.i((.., .., 0));
                                let aow_a = &loc_ao.i((.., .., 0..4)) * &loc_wv_a.i((.., None, 0..4));
                                let loc_wv_b = loc_wv.i((.., .., 1));
                                let aow_b = &loc_ao .i((.., .., 0..4)) * &loc_wv_b.i((.., None, 0..4));

                                for ic in 0..4 {
                                    vxc.i_mut((.., .., 0)).matmul_from(
                                        &loc_ao.i((.., .., 0)), &aow_a.i((.., .., ic)).t(), 1.0, 1.0
                                    );
                                    vxc.i_mut((.., .., 1)).matmul_from(
                                        &loc_ao.i((.., .., 0)), &aow_b.i((.., .., ic)).t(), 1.0, 1.0
                                    );
                                    let vtau_ket_a = &loc_ao.i((.., .., ic+1)) * &loc_wv_a.i((.., None, 4));
                                    let vtau_ket_b = &loc_ao.i((.., .., ic+1)) * &loc_wv_b.i((.., None, 4));
                                    v1.i_mut((.., .., 0)).matmul_from(
                                        &loc_ao.i((.., .., ic+1)), &vtau_ket_a.t(), 1.0, 1.0
                                    );
                                    v1.i_mut((.., .., 1)).matmul_from(
                                        &loc_ao.i((.., .., ic+1)), &vtau_ket_b.t(), 1.0, 1.0
                                    );
                                }
                            },
                            XCType::HF => {},
                        }

                    },
                    _ => unreachable!(),
                }
            }
        ); // end block loop

        let mut vmat = vec!();

        match xc_data.xc_type {
            XCType::LDA => {},
            XCType::GGA => {
                let vxc_trans = vxc.transpose([0, 2, 1]);
                vxc = &vxc + vxc_trans;
            },
            XCType::MGGA => {
                let vxc_trans = vxc.transpose([0, 2, 1]);
                vxc = &vxc + vxc_trans;
                let Some(v1) = vxc_mgga_temp.as_ref() else {
                    panic!("vxc_mgga_temp is not initialized for MGGA type");
                };
                vxc.add_assign(v1);
            },
            XCType::HF => {
                unreachable!()
            },
        }

        vmat.push(MatrixFull::from_vec([num_basis, num_basis], vxc.i(0).to_vec()).unwrap());
        if spin == 1 {
            vmat.push(MatrixFull::from_vec([num_basis, num_basis], vxc.i(1).to_vec()).unwrap());
        }

        (nelec, exc_sum, vmat)
    }


}



