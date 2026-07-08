use super::{SCF, SCFType};
use super::util::integer_homo_lumo;

impl SCF {
    pub fn print_homo_lumo_gap(&self) {
        let has_smear = self.mol.ctrl.smear.is_some();
        let (homo_idx, lumo_idx) = if has_smear {
            integer_homo_lumo(&self.mol.num_elec, self.mol.spin_channel)
        } else {
            (self.homo, self.lumo)
        };
        let is_rohf = match self.scftype {
            SCFType::ROHF => true,
            _ => false,
        };
        if self.mol.ctrl.print_level>0 {
            if self.mol.spin_channel==1 {
                let i_homo = homo_idx[0];
                let i_lumo = lumo_idx[0];
                let homo = self.eigenvalues[0][i_homo];
                if i_lumo < self.eigenvalues[0].len()  {
                    let lumo = self.eigenvalues[0][i_lumo];
                    if has_smear {
                        print!("(integer-occupation pure-state gap) ");
                    }
                    println!("HOMO ({:3}): {:16.8}, LUMO ({:3}): {:14.6}, H-L Gap: {:16.8}", i_homo, homo, i_lumo, lumo, lumo-homo);
                } else {
                    println!("{:?}", &self.eigenvalues[0]);
                    println!("HOMO: {:16.8} (No virtual orbtials available)", homo);
                }
            } else {
                for i_spin in (0..self.mol.spin_channel) {
                    if (self.mol.num_elec[i_spin+1] > 1.0E-5) {
                        let i_homo = homo_idx[i_spin];
                        let i_lumo = lumo_idx[i_spin];
                        if ! is_rohf {
                            let homo = self.eigenvalues[i_spin][i_homo];
                            let lumo = self.eigenvalues[i_spin][i_lumo];
                            if has_smear {
                                print!("(integer-occupation pure-state gap) ");
                            }
                            println!("Spin {:2}: HOMO ({:3}): {:14.6}, LUMO ({:3}): {:14.6}, H-L Gap: {:16.8}", i_spin, i_homo, homo, i_lumo, lumo, lumo-homo);
                        } else {
                            let homo = self.eigenvalues[0][i_homo];
                            let lumo = self.eigenvalues[0][i_lumo];
                            if has_smear {
                                print!("(integer-occupation pure-state gap) ");
                            }
                            println!("Spin {:2}: HOMO ({:3}): {:14.6}, LUMO ({:3}): {:14.6}, H-L Gap: {:16.8}", i_spin, i_homo, homo, i_lumo, lumo, lumo-homo);
                        }
                    } else {
                        println!("No electron with spin {:2}", i_spin);
                    }
                }
            }
        }
    }
}

impl SCF {
    pub fn formated_eigenvalues(&self,num_state_to_print:usize) {
        let mut cur_num_state_to_print = 0;
        let spin_channel = self.mol.spin_channel;
        match self.scftype {
            SCFType::RHF => {
                println!("Eigenvalues in Restricted HF (or KS) calculation:");
                println!("{:>8}{:>14}{:>18}",String::from("State"),
                                        String::from("Occupation"),
                                        String::from("Eigenvalue"));
                if self.occupation[0].len() < num_state_to_print {
                    cur_num_state_to_print = self.eigenvalues[0].len();
                } else {
                    cur_num_state_to_print = num_state_to_print;
                }
                for i_state in (0..cur_num_state_to_print) {
                    println!("{:>8}{:>14.5}{:>18.6}",i_state,self.occupation[0][i_state],self.eigenvalues[0][i_state]);
                }
            },
            SCFType::UHF => {
                println!("Eigenvalues in Unrestricted HF (or KS) calculation:");
                for i_spin in (0..spin_channel) {
                    if i_spin == 0 {
                        println!("Spin-up eigenvalues");
                        println!(" ");
                    } else {
                        println!(" ");
                        println!("Spin-down eigenvalues");
                        println!(" ");
                    }
                    println!("{:>8}{:>14}{:>18}",String::from("State"),
                                                String::from("Occupation"),
                                                String::from("Eigenvalue"));
                    if self.occupation[i_spin].len() < num_state_to_print {
                        cur_num_state_to_print = self.eigenvalues[i_spin].len();
                        println!("the number of eigenvalues in {} spin channel is {}", i_spin, self.occupation[i_spin].len());
                    } else {
                        cur_num_state_to_print = num_state_to_print;
                        for i_state in (0..cur_num_state_to_print) {
                            println!("{:>8}{:>14.5}{:>18.6}",i_state,self.occupation[i_spin][i_state],self.eigenvalues[i_spin][i_state]);
                        }
                    }
                }
            },
            SCFType::ROHF => {
                println!("Eigenvalues in Rrestricted Openshell HF (or KS) calculation:");
                println!("{:>8}{:>14}{:>18}",String::from("State"),
                                        String::from("Occupation"),
                                        String::from("Eigenvalue"));
                // combine the occ numbers of alpha and beta channel
                let combined_occupation: Vec<f64> =  self.occupation[0]
                                                        .iter()
                                                        .zip(self.occupation[1].iter())
                                                        .map(|(a, b)| a + b)
                                                        .collect();
                if combined_occupation.len() < num_state_to_print {
                    cur_num_state_to_print = combined_occupation.len();
                } else {
                    cur_num_state_to_print = num_state_to_print;
                }
                for i_state in (0..cur_num_state_to_print) {
                    println!("{:>8}{:>14.5}{:>18.6}",i_state,combined_occupation[i_state],self.eigenvalues[0][i_state]);
                }
            }
        }
    }
}

impl SCF {
    pub fn formated_eigenvectors(&self) {
        let spin_channel = self.mol.spin_channel;
        if spin_channel==1 || matches!(&self.scftype, SCFType::ROHF){
            self.eigenvectors[0].formated_output(5, "full");
        } else {
            (0..spin_channel).into_iter().for_each(|i_spin|{
                if i_spin == 0 {
                    println!("Spin-up eigenvalues");
                    println!(" ");
                } else {
                    println!(" ");
                    println!("Spin-down eigenvalues");
                    println!(" ");
                }
                self.eigenvectors[i_spin].formated_output(5, "full");
            });
        }
    }
}

pub fn print_force_for_ghost_point_charges(scf_data: &SCF) {
    if let Some(ghost_forces) = scf_data.compute_ghost_charge_forces() {
        if scf_data.mol.ctrl.print_level > 0 {
            println!("Calculating forces on ghost point charges");
        }
    
        
        println!("------ Forces on point charges [a.u.] ------");
        let geom = &scf_data.mol.geom;

        ghost_forces.iter_columns_full().enumerate()
            .zip(geom.ghost_pc_pos.iter_columns_full())
            .zip(geom.ghost_pc_chrg.iter())
            .for_each(|(((i,force),position),charge)| {
            let elem_str = format!("Q{:04}", i+1);  
            let force_str = format!("{:15.8}{:15.8}{:15.8}", 
                                force[0], force[1], force[2]);
            println!("    {:<8} {}", elem_str, force_str);
            if scf_data.mol.ctrl.print_level > 1 {
                println!("      Charge: {:.6}, Position: [{:.6}, {:.6}, {:.6}]", 
                        charge, position[0], position[1], position[2]);
            }
        });
        println!("--------------------------------------------");
    } else {
        if scf_data.mol.ctrl.print_level > 0 {
            println!("No ghost point charges found");
        }
    }
}
