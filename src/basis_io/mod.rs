//! Basis I/O: BSE-format basis/ECP parsing (`BasCell`/`Basis4Elem` + Raw mirrors),
//! libcint-consistent normalization, and GTO evaluation (see `gto_cartesian`/`gto_value`).
use rest_tensors::matrix_blas_lapack::_dgemm_nn;
use rest_tensors::{MatrixFull, RIFull};
use itertools::{Itertools, izip};
use rayon::prelude::*;
use serde::{Deserialize,Serialize};
use serde_json::{Result,Value};
use anyhow;
use tensors::{MathMatrix,ParMathMatrix};
use tensors::matrix_blas_lapack::_dgemm_nn_serial;
use std::collections::HashMap;
use std::convert::TryInto;
use std::fs;
use std::env;
use std::path;
use std::io::{Write,BufRead, BufReader};
use rest_libcint::{CINTR2CDATA, CintType};
use std::f64::consts::{PI, E};
use libm;
use crate::constants::{self, C2S_L0, C2S_L1, C2S_L2, C2S_L3, C2S_L4, CarBasInfo, c2s_matrix_const, cartesian_gto_const, cint_cartesian_gto_const};
use crate::utilities;
pub mod bse_downloader;
pub mod basis_list;
pub mod etb;
pub mod ecp;
mod gto_cartesian;
pub use gto_cartesian::*;
mod gto_value;
pub use gto_value::*;
#[cfg(test)]
mod tests;

#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct BasCellRaw {
    pub function_type: Option<String>,
    pub region: Option<String>,
    pub angular_momentum: Vec<i32>,
    pub exponents: Vec<String>,
    pub coefficients: Vec<Vec<String>>,
}
#[derive(Serialize,Deserialize,Clone, Debug)]
pub struct RefCell {
    pub reference_description: Option<String>,
    pub reference_keys: Option<Vec<String>>,
}

#[derive(Clone, Debug,Serialize,Deserialize)]
pub struct ECPCellRaw {
    pub angular_momentum: Vec<i32>,
    pub coefficients: Vec<Vec<String>>,
    pub ecp_type: Option<String>,
    pub r_exponents: Vec<i32>,
    pub gaussian_exponents: Vec<String>,
}
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct Basis4ElemRaw {
    pub electron_shells: Vec<BasCellRaw>,
    pub references: Option<Vec<RefCell>>,
    pub ecp_potentials: Option<Vec<ECPCellRaw>>,
    pub ecp_electrons: Option<usize>,
}

/// #BasCell
/// `BasCell` has the same organization structure of the basis set files from the basis set exchange (BSE) 
///  - BasCell.function_type:     "gto_spherical" or "gto"  
///  - BasCell.region:  
///  - BasCell.angular_momentum:  The angular momentums of the GTOs in this cell  
///  - BasCell.exponents:         The exponents  
///  - BasCell.coefficients:      The coefficients 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasCell {
    pub function_type: Option<String>,
    pub region: Option<String>,
    pub angular_momentum: Vec<i32>,
    pub exponents: Vec<f64>,
    pub coefficients: Vec<Vec<f64>>,
    pub native_coefficients: Vec<Vec<f64>>,
    //pub c2r_scaling: Vec<f64>,
}

#[derive(Clone, Debug,Serialize,Deserialize)]
pub struct ECPCell {
    pub angular_momentum: Vec<i32>,
    pub coefficients: Vec<Vec<f64>>,
    pub ecp_type: Option<String>,
    pub r_exponents: Vec<i32>,
    pub gaussian_exponents: Vec<f64>,
}

///  # BasInfo
/// `BasInfo` contains the elemental information of each group of GTOs, including
///   self.bas_name: The GTOs in the group has the same angular momentum. 
///                  It is the name for the angularm momentum, for example, S, P, D, F, and so forth.
///   self.bas_type: The GTOs are `Primitive` or `Contractive`
///   self.elem_index0: The index of atom where the GTOs are located
///   self.cint_index0: The global start index of this basis function group 
///   self.cint_index1: The number of basis functions in this group
#[derive(Clone, Debug)]
pub struct BasInfo {
    pub bas_name: String,
    pub bas_type: String,
    pub elem_index0: usize,
    pub cint_index0: usize,
    pub cint_index1: usize,
}

impl  BasInfo {
    pub fn new() -> BasInfo {
        BasInfo {
            bas_name : String::from("S"),
            bas_type : String::from("Primitive"),
            elem_index0 : 0,
            cint_index0 : 0,
            cint_index1 : 0,
        }
    }
    pub fn formated_name(&self) -> String {
        format!("{}-e{}-cint{}-{}",self.bas_name,self.elem_index0, self.cint_index0,self.cint_index1)
    }
}

/// #Basis4Elem
/// `Basis4Elem` contains all informations of the GTO basis functions located in a given element
///     self.electron_shells: the GTO basis functions organized cell by cell [`BasCell`](BasCell)
///     self.references: The reference of each basis cell.
///     self.global_index: The global index for the given atom
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Basis4Elem {
    pub electron_shells: Vec<BasCell>,
    pub references: Option<Vec<RefCell>>,
    pub ecp_potentials: Option<Vec<ECPCell>>,
    pub ecp_electrons: Option<usize>,
    pub global_index: (usize,usize)
}

impl Basis4Elem {
    pub fn parse_json(json: &String)-> anyhow::Result<Basis4Elem> {
        let tmp_basis: Basis4ElemRaw = serde_json::from_str(&json)?;
        let mut tmp_vec:Vec<BasCell> = vec![];
        &tmp_basis.electron_shells.iter().for_each(|x: &BasCellRaw| {
            tmp_vec.extend(x.parse());
        });
        let (ecp_electrons, ecp_potentials) = if let (Some(ecp_electrons), Some(ecp_potentials))= (&tmp_basis.ecp_electrons, &tmp_basis.ecp_potentials)  {
            (
                Some(ecp_electrons.clone()), 
                Some(ecp_potentials.iter().map(|x| x.parse()).collect::<Vec<ECPCell>>())
            )
        } else {
            (None, None)
        };
        //println!("debug 0");
        if let Some(ecp_electrons_unwrap)=&ecp_electrons {
            println!("ecp_electrons: {}", &ecp_electrons_unwrap)
        };
        if let Some(ecp_potentials_unwrap)=&ecp_potentials {
            for tmp_ecp in ecp_potentials_unwrap {
                println!("ecp_potentials: {:?}", tmp_ecp);
            }
        };
        Ok(Basis4Elem{
            electron_shells: tmp_vec,
            references: tmp_basis.references,
            ecp_electrons,
            ecp_potentials,
            global_index: (0,0)
        })
    }
    pub fn parse_json_from_file(file_name: String, cint_type: &CintType)-> anyhow::Result<Basis4Elem> {
        let tmp_cont = fs::read_to_string(&file_name[..])?;
        //let tmp_basis:Basis4ElemRaw = serde_json::from_str(&tmp_cont[..])?;
        let raw:Value = serde_json::from_str(&tmp_cont[..])?;
        let tmp_basis:Basis4ElemRaw = if !raw["electron_shells"].is_null() {
            serde_json::from_value(raw)?
        } else if !raw["elements"].is_null() {
            // Treat json from `bse convert-basis`, which looks like
            //     { "elements": { "1": { "electron_shells": [ ... ] } }
            // assume there's only one element
            serde_json::from_value(raw["elements"].as_object().unwrap().values().next().unwrap().clone())?
        } else {
            panic!("The basis set file is not in the right format: {:?}", &file_name);
        };
        let mut tmp_vec:Vec<BasCell> = vec![];
        &tmp_basis.electron_shells.iter().for_each(|x: &BasCellRaw| {
            let tmp_bas_cell = x.parse();
            //println!("{:?}",tmp_bas_cell);
            for mut x_bascell in tmp_bas_cell {
                x_bascell.basis_normalization(cint_type);
                tmp_vec.push(x_bascell);
            }
        });
        let (ecp_electrons, ecp_potentials) = if let (Some(ecp_electrons), Some(ecp_potentials))= (&tmp_basis.ecp_electrons, &tmp_basis.ecp_potentials)  {
            (
                Some(ecp_electrons.clone()), 
                Some(ecp_potentials.iter().map(|x| x.parse()).collect::<Vec<ECPCell>>())
            )
        } else {
            (None, None)
        };
        //println!("debug 1");
        //if let Some(ecp_electrons_unwrap)=&ecp_electrons {
        //    println!("ecp_electrons: {}", &ecp_electrons_unwrap)
        //};
        //if let Some(ecp_potentials_unwrap)=&ecp_potentials {
        //    for tmp_ecp in ecp_potentials_unwrap {
        //        println!("ecp_potentials: {:?}", tmp_ecp);
        //    }
        //};
        // Re-ordering the basis function shells according to the angular momentum
        tmp_vec.sort_by(|a,b| a.angular_momentum[0].cmp(&b.angular_momentum[0]));
        Ok(Basis4Elem{
            electron_shells: tmp_vec,
            references: tmp_basis.references,
            ecp_electrons,
            ecp_potentials,
            global_index: (0,0)
        })
    }
    pub fn to_numgrid_io(&self) -> (HashMap<usize,f64>,f64) {
        let mut alpha_max = -f64::MAX;
        let mut alpha_min = HashMap::new();
        self.electron_shells.iter().for_each(|value| {
            let angular_momentum = value.angular_momentum[0] as usize;
            value.exponents.iter().for_each(|exponent| {
                alpha_max = alpha_max.max(*exponent);
                let s = alpha_min.entry(angular_momentum).or_insert(f64::MAX);
                if exponent < s {
                    *s = *exponent;
                }
            });
        });
        (alpha_min,alpha_max)
    }


}


impl ECPCellRaw {
    pub fn parse(&self) -> ECPCell {
        let to_ecpcell = ECPCell {
            angular_momentum: self.angular_momentum.clone(),    
            ecp_type: self.ecp_type.clone(),
            r_exponents: self.r_exponents.clone(),
            coefficients: {
                self.coefficients.iter().map(|i| i.iter().map(|j| j.parse().unwrap()).collect::<Vec<f64>>()).collect::<Vec<Vec<f64>>>()
            },
            gaussian_exponents: {
                self.gaussian_exponents.iter().map(|i| i.parse().unwrap()).collect::<Vec<f64>>()
            },
        };
        if to_ecpcell.angular_momentum.len()!=to_ecpcell.coefficients.len() {
            panic!("The number of angular momentum is not equal to the number of coefficients\n")
        };
        if to_ecpcell.r_exponents.len()!=to_ecpcell.gaussian_exponents.len() {
            panic!("The number of r_exponents is not equal to the number of gaussian_exponents\n")
        };
        to_ecpcell
    }
}


impl BasCellRaw {
    pub fn parse(&self) -> Vec<BasCell> {
        let mut bascell_vec: Vec<BasCell> = vec![];
        let mut tmp_vec_0: Vec<i32> = self.angular_momentum.clone();

        // parse the vector of exponents
        let mut tmp_vec_1 = self.exponents.iter()
            .map(|i| i.parse().unwrap())
            .collect::<Vec<f64>>();

        // parse the vector of coefficients
        let mut tmp_vec_2 = self.coefficients.iter().map(|i| i.iter()
            .map(|j| j.parse().unwrap()).collect::<Vec<f64>>())
            .collect::<Vec<Vec<f64>>>();

        if self.angular_momentum.len()>1 && self.angular_momentum.len()==self.coefficients.len() {
            // for Pople basis sets (for example, 6-31G)
            for i_bas in (0..self.angular_momentum.len()) {
                bascell_vec.push(
                    BasCell { 
                        function_type: self.function_type.clone(), 
                        region: self.region.clone(), 
                        angular_momentum: vec![tmp_vec_0[i_bas]], 
                        exponents: tmp_vec_1.clone(), 
                        coefficients: vec![tmp_vec_2[i_bas].clone()],
                        native_coefficients: vec![tmp_vec_2[i_bas].clone()]
                    }
                );
            }
        } else if self.angular_momentum.len()>1 && self.angular_momentum.len()!=self.coefficients.len(){
            // improper setting for the Pople basis sets
            panic!("The basis set with a wrong format: The number of angular momentums is more than one, but is unequal to the number of the coefficient set\n
                    {:?}", &self);
        } else if self.angular_momentum.len()==1 {
            // for any other standard basis set
            //let mut tmp_vec_2_0:Vec<Vec<f64>> = vec![];
            //let mut tmp_vec_2_1:Vec<Vec<f64>> = vec![];
            //for i_coeff in (0..tmp_vec_2.len()) {
            //    //let full_power = tmp_vec_2[i_coeff].iter().fold(true,|check,i| check && *i!=0.0f64);
            //    let full_power = true;
            //    if full_power {
            //        tmp_vec_2_0.push(tmp_vec_2[i_coeff].clone())
            //    } else {
            //        tmp_vec_2_1.push(tmp_vec_2[i_coeff].clone())
            //    };
            //}
            // consider the contracted GTOs that use all primitive GTOs in the given shell
            bascell_vec.push(
                BasCell { 
                    function_type: self.function_type.clone(), 
                    region: self.region.clone(), 
                    angular_momentum: tmp_vec_0.clone(), 
                    exponents: tmp_vec_1.clone(), 
                    coefficients: tmp_vec_2.clone(), 
                    native_coefficients: tmp_vec_2.clone()
                }
            );
            //// now consider the contracted (or primitive) GTOs that use part of primitive GTOs in the given shell
            //for i_coeff in (0..tmp_vec_2_1.len()) {
            //    let mut tmp_vec_1_1:Vec<f64> = vec![];
            //    let mut tmp_vec_2_2:Vec<f64> = vec![];
            //    for (i,j) in tmp_vec_2_1[i_coeff].iter().zip(tmp_vec_1.iter()) {
            //        if *i!=0.0 {
            //            tmp_vec_1_1.push(*j);
            //            tmp_vec_2_2.push(*i);
            //        }
            //    }
            //    bascell_vec.push(
            //        BasCell { 
            //            function_type: self.function_type.clone(), 
            //            region: self.region.clone(), 
            //            angular_momentum: tmp_vec_0.clone(), 
            //            exponents: tmp_vec_1_1.clone(), 
            //            coefficients: vec![tmp_vec_2_2.clone()] 
            //        }
            //    );
            //}
        } else {
            panic!("The angular momentum is missing for the basis set : {:?}", &self);
        }
        bascell_vec
    }
}

impl BasCell {
   pub fn basis_normalization(&mut self, cint_type: &CintType) {
       /// To use libcint properly, we should rescale the coefficients of the contractive GTOs,
       /// A detailed interpreation is necessary, which, however, is missing at present.
       let mut atm: Vec<Vec<i32>> = vec![vec![1, 20, 1, 23, 0, 0]];
       let mut env: Vec<f64> = vec![0.0_f64; 24]; 
       let bas_start: i32 = 24;
       let bas: Vec<Vec<i32>> = vec![vec![0,
                                self.angular_momentum[0],
                                self.exponents.len() as i32, 
                                self.coefficients.len() as i32,
                                0,
                                bas_start,
                                bas_start + self.exponents.len() as i32,
                                0
                                ]];
        env.extend(self.exponents.iter());
        //self.exponents.iter().for_each(|x| {
        //    env.push(*x);
        //});
        //(0..self.exponents.len()).into_iter().for_each(|x| {
        //    println!("gto_norm for {}: {}",self.exponents[x],
        //        CINTR2CDATA::gto_norm(self.angular_momentum[0] as std::os::raw::c_int,self.exponents[x]));
        //});
        //println!(" before basis-normalization: self.coefficients {:?}",self.coefficients);
        let mut tmp_exponents = self.exponents.to_owned();
        if self.angular_momentum.len()>1 && self.angular_momentum.len()==self.coefficients.len() {
            // for Pople basis sets (for example, 6-31G)
            for i_bas in (0..self.angular_momentum.len()) {
                let mut tmp_ang = self.angular_momentum[i_bas];
                let mut tmp_coefficients: Vec<Vec<f64>> = vec![];
                for coe_vec in self.coefficients.iter() {
                    let mut tmp_coefficients_column: Vec<f64> = vec![];
                    coe_vec.iter().enumerate().for_each(|(ix,x)| {
                        let tmp_value = CINTR2CDATA::gto_norm(tmp_ang as std::os::raw::c_int,
                                                          self.exponents[ix]);
                        //println!("gto_norm for {}: {}",tmp_exponents[ix],tmp_value);
                        env.push(*x*tmp_value);
                        tmp_coefficients_column.push(*x*tmp_value);
                        //env.push(*x);
                        //tmp_coefficients_column.push(*x);
                    });
                    tmp_coefficients.push(tmp_coefficients_column);
                };
            }
        } else if self.angular_momentum.len()>1 && self.angular_momentum.len()!=self.coefficients.len(){
            // improper setting for the Pople basis sets
            panic!("Error: the Pople basis set is specified with a wrong format: {:?}", &self);
        } else if self.angular_momentum.len()==1 {
            // for any other standard basis set
            let mut tmp_ang = self.angular_momentum[0];
            let mut tmp_coefficients: Vec<Vec<f64>> = vec![];
            //println!("before basis-normalization: self.coefficients {:?}",self.coefficients);
            //for (index, coe_vec) in self.coefficients.iter().enumerate() {
            for coe_vec in self.coefficients.iter() {
                let mut tmp_coefficients_column: Vec<f64> = vec![];
                coe_vec.iter().enumerate().for_each(|(ix,x)| {
                    let tmp_value = CINTR2CDATA::gto_norm(tmp_ang as std::os::raw::c_int,
                                                      self.exponents[ix]);
                    //println!("gto_norm for {}: {}",tmp_exponents[ix],tmp_value);
                    //let tmp_value =  1.0;
                    env.push(*x*tmp_value);
                    tmp_coefficients_column.push(*x*tmp_value);
                    //env.push(*x);
                    //tmp_coefficients_column.push(*x);
                });
                tmp_coefficients.push(tmp_coefficients_column);
            };
            self.native_coefficients = self.coefficients.clone();
            self.coefficients = tmp_coefficients.clone();

            // now for normalization
            let mut cint_data = CINTR2CDATA::new();
            let natm = atm.len() as i32;
            let nbas = bas.len() as i32;
            let (ang,n_len) = match cint_type {
                CintType::Cartesian => {let ang = tmp_ang as usize; (ang,(ang+1)*(ang+2)/2)},
                CintType::Spheric => {let ang = tmp_ang as usize; (ang, ang*2+1)},
                _ => panic!("The angular momentum is missing for the basis set : {:?}", &self)
            };
            //let n_len = (tmp_ang*2+1) as usize;
            cint_data.initial_r2c(&atm, natm, &bas, nbas, &env);
            //cint_data.set_cint_type(CintType::Spheric);
            cint_data.set_cint_type(*cint_type);
            cint_data.cint1e_ovlp_optimizer_rust();
            let num_bas = self.coefficients.len();
            let num_pri = self.exponents.len();
            let buf = MatrixFull::from_vec([n_len*num_bas, n_len*num_bas],cint_data.cint_ij(0, 0,&String::from("ovlp"))).unwrap();
            //buf.formated_output(6,"full");

            (0..num_bas).into_iter().for_each(|x| {
                //let mut norm_fac_array = vec![1.0_f64;n_len];
                let org_coeff = self.coefficients[x].clone();
                //let xx = (n_len*num_bas)*(x*n_len)+n_len*x;
                let norm_fac = 1.0_f64/buf[(x*n_len, x*n_len)].sqrt();
                self.coefficients[x].iter_mut().enumerate().for_each(|(ix,y)| {
                    *y *= norm_fac;
                    //norm_fac_array[ix] = 1.0/&norm_fac;
                    //println!("debug: norm: {:?}", 1.0/&norm_fac);
                });
                //println!("debug: norm: {:?}", norm_fac_array);
                //println!("debug: norm: {:?} \n  nat_coeff: {:?} \n org_coeff: {:?}, \n coef: {:?}", norm_fac, &self.native_coefficients[x], org_coeff, &self.coefficients[x]);
            });
            //println!("after coef {:?}",&self.coefficients);
            cint_data.final_c2r();
            //println!(" after basis-normalization: self.coefficients {:?}",self.coefficients);
        } else {
            panic!("The angular momentum is missing for the basis set : {:?}", &self);
        }
   }
}



