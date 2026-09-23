pub mod uhf;
pub mod rks;
pub mod uks;
pub mod rhf;
pub mod traits;

use std::io::{self, Write};

use crate::main_driver::{collect_total_energy, performance_essential_calculations};
use crate::{constants::{BOHR, EV}, scf_io::{initialize_scf, scf_without_build, SCF}, utilities};
use crate::mpi_io::{MPIData, MPIOperator};
use tensors::MatrixFull;

//pub fn numerical_hessian(scf_data: &SCF, mpi_operator: &Option<MPIOperator>) -> (f64,MatrixFull<f64>) { 
//    let num_atoms =  scf_data.mol.geom.nfree;
//    let mut num_hessian = MatrixFull::new([num_atoms*3,num_atoms*3],0.0);
//    (0..num_atoms).into_iter().for_each(|atm_idx| { 
//        (0..num_atoms).into_iter().for_each(|atm_idx2| { 
//            (0..num_atoms).into_iter().for_each(|atm_idx3| { 
//                (0..num_atoms).into_iter().for_each(|atm_idx4| { 
//
//                });
//            });
//        });
//    });
//}
pub fn numerical_force(scf_data: &mut SCF, displace: f64, mpi_operator: &Option<MPIOperator>) -> (f64,MatrixFull<f64>) {
    let num_atoms =  scf_data.mol.geom.nfree;
    let mut num_force = MatrixFull::new([3,num_atoms],0.0);
    // Every displaced job rebuilds the integrals and grids in `initialize_scf`, so the resident
    // tensors would be copied 6N times only to be overwritten, and a second copy of them would be
    // held for the whole loop: GBs per displacement on a large system. Park them for the duration
    // of the loop (the caller gets them back before this function returns).
    let parked = (
        scf_data.ijkl.take(),
        scf_data.ri3fn.take(),
        scf_data.ri3fn_sr.take(),
        scf_data.ri3fn_isdf.take(),
        scf_data.ri3fn_bse.take(),
        scf_data.rimatr.take(),
        scf_data.rimatr_sr.take(),
        scf_data.rimatr_bse.take(),
        scf_data.ri3mo.take(),
        scf_data.ri3mo_full.take(),
        scf_data.tab_ao.take(),
        scf_data.m.take(),
        scf_data.grids.take(),
    );
    if scf_data.mol.ctrl.print_level > 0 {
        if let Some(mp_op) = mpi_operator {
            if mp_op.rank == 0 {
                print!("Numerical force calculation ...");
                io::stdout().flush().unwrap();
            }
        } else {
            print!("Numerical force calculation ...");
            io::stdout().flush().unwrap();
        }
    }
    (0..num_atoms).into_iter().for_each(|atm_idx| {
        if scf_data.mol.ctrl.print_level > 0 {
            if let Some(mp_op) = mpi_operator {
                if mp_op.rank == 0 {
                    print!("| {:3}", &atm_idx);
                    io::stdout().flush().unwrap();
                }
            } else {
                print!("| {:3}", &atm_idx);
                io::stdout().flush().unwrap();
            }
        }
        let mut num_force_atm = &mut num_force[(..,atm_idx)];
        num_force_atm.iter_mut().enumerate().for_each(|(xyz,per_force)| {
            let mut time_mark = utilities::TimeRecords::new();
            
            // move the atom along + direction
            let mut vec_xyz = vec![0.0;3];
            vec_xyz[xyz] = displace;
            // `(*scf_data).clone()`: `scf_data` is a `&mut SCF`, and `scf_data.clone()` would
            // clone the reference instead of the structure
            let mut new_scf = (*scf_data).clone();
            // update the geometry
            new_scf.mol.geom.geom_shift(atm_idx, vec_xyz);
            // update the control file
            // ==== DEBUG IGOR ====
            if let Some(mp_op) = &mpi_operator {
                if mp_op.rank == 0 {
                    new_scf.mol.ctrl.print_level = 0;
                } else {
                    new_scf.mol.ctrl.print_level = 0;
                }
            } else {
                new_scf.mol.ctrl.print_level = 0;
            }
            // ==== DEBUG IGOR ====
            new_scf.mol.ctrl.initial_guess = String::from("inherit");
            initialize_scf(&mut new_scf, mpi_operator);
            let de0 = performance_essential_calculations(&mut new_scf, &mut time_mark, mpi_operator);

            // move the atom along - direction
            let mut vec_xyz = vec![0.0;3];
            vec_xyz[xyz] = -displace;
            new_scf = (*scf_data).clone();
            // update the geometry
            new_scf.mol.geom.geom_shift(atm_idx, vec_xyz);
            // update the control file
            // ==== DEBUG IGOR ====
            if let Some(mp_op) = &mpi_operator {
                if mp_op.rank == 0 {
                    new_scf.mol.ctrl.print_level = 0;
                } else {
                    new_scf.mol.ctrl.print_level = 0;
                }
            } else {
                new_scf.mol.ctrl.print_level = 0;
            }
            // ==== DEBUG IGOR ====
            new_scf.mol.ctrl.initial_guess = String::from("inherit");
            initialize_scf(&mut new_scf, mpi_operator);
            let de1 = performance_essential_calculations(&mut new_scf, &mut time_mark, mpi_operator);

            *per_force = 0.5*(de0-de1)/displace;

        })
    });

    // hand the parked tensors back, so the caller sees the SCF it passed in
    (
        scf_data.ijkl,
        scf_data.ri3fn,
        scf_data.ri3fn_sr,
        scf_data.ri3fn_isdf,
        scf_data.ri3fn_bse,
        scf_data.rimatr,
        scf_data.rimatr_sr,
        scf_data.rimatr_bse,
        scf_data.ri3mo,
        scf_data.ri3mo_full,
        scf_data.tab_ao,
        scf_data.m,
        scf_data.grids,
    ) = parked;

    if scf_data.mol.ctrl.print_level > 0 {
        if let Some(mp_op) = mpi_operator {
            if mp_op.rank == 0 {
                print!("|\n");
                io::stdout().flush().unwrap();
            }
        } else {
            print!("|\n");
            io::stdout().flush().unwrap();
        }
    }

    (collect_total_energy(scf_data),num_force)
}

pub fn formated_force(force: &MatrixFull<f64>, elem: &Vec<String>) -> String {
    let mut output = String::new();
    force.iter_columns_full().zip(elem.iter()).for_each(|(force, elem)| {
        output  = format!("{}{:3}{:16.8}{:16.8}{:16.8}\n", output, elem, force[0],force[1],force[2]);
    });

    output

}

pub fn formated_force_ev(force: &MatrixFull<f64>, elem: &Vec<String>) -> String {
    let mut output = String::new();
    force.iter_columns_full().zip(elem.iter()).for_each(|(force, elem)| {
        output  = format!("{}{:3}{:16.8}{:16.8}{:16.8}\n", output, elem, force[0]*EV/BOHR,force[1]*EV/BOHR,force[2]*EV/BOHR);
    });

    output

}


//pub fn evaluate(x: &[f64], gx: &mut [f64]) -> f64 {
//
//    
//    fn to_matrixfull(x: &[f64]) -> MatrixFull<f64> {
//        let num_atoms = x.len()/3;
//        MatrixFull::from_vec([3,num_atoms],x.to_vec()).unwrap()
//    }
//
//}