use crate::molecule_io::Molecule;
use crate::constants::{ELEM1ST, ELEM2ND, ELEM3RD, ELEM4TH, ELEM5TH, ELEM6TH, ELEMTMS};

impl Molecule {
    pub fn generate_start_mo(&self, ecp_electrons: usize) -> usize {
        let mut start_mo = count_frozen_core_states(self.ctrl.frozen_core_postscf, &self.geom.elem);
        let ecp_orbs = ecp_electrons/2;
        if ecp_orbs == 0 {
            if self.ctrl.print_level > 0 {
                println!("For SCF calculation, no core orbital is frozen by effective core potential (ECP) approximation");
            }
        } else if start_mo < ecp_orbs {
            if self.ctrl.print_level > 0 {
                println!("<start_mo> for the frozen-core post-SCF methods {} is smaller than the number of ecp orbitals {}. Set <start_mo> = 0",
                    start_mo, ecp_orbs);
            }
            start_mo = 0;
        } else {
            if self.ctrl.print_level > 0 {
                println!("<start_mo> for the frozen-core post-SCF methods {} is larger than the number of ecp orbitals {}. Take the <start_mo> -= ecp_orbitals",
                        start_mo, ecp_orbs);
            }
            start_mo = start_mo - ecp_orbs
        }
        start_mo
    }
}


pub fn count_frozen_core_states(n_frozen_shell: i32, elem: &Vec<String>) -> usize {
    let mut n_low_state = 0_usize;
    let mut n_tm = 0_usize;

    //let n_frozen_shell = self.ctrl.frozen_core_postscf;
    let (n_frozen_shell_1, n_frozen_shell_2) = if n_frozen_shell > 10 {
        let n_frozen_shell_1 = n_frozen_shell%10;
        let n_frozen_shell_2 = n_frozen_shell/10;
        (n_frozen_shell_1, n_frozen_shell_2)
    } else {
        (n_frozen_shell, n_frozen_shell)
    };

    elem.iter().for_each(|sn| {
        let formated_elem = crate::geom_io::formated_element_name(&sn);
        let flag_first_row  = ELEM1ST.iter().fold(false, |acc, elem| acc || elem.eq(&formated_elem));
        let flag_second_row = if flag_first_row {
            false
        } else {
            ELEM2ND.iter().fold(false, |acc, elem| acc || elem.eq(&formated_elem))
        };
        let flag_third_row  = if flag_first_row || flag_second_row {
            false
        } else {
            ELEM3RD.iter().fold(false, |acc, elem| acc || elem.eq(&formated_elem))
        };
        let flag_fourth_row = if flag_first_row || flag_second_row || flag_third_row {
            false
        } else {
            ELEM4TH.iter().fold(false, |acc, elem| acc || elem.eq(&formated_elem))
        };
        let flag_fifth_row  = if flag_first_row || flag_second_row || flag_third_row || flag_fourth_row {
            false
        } else {
            ELEM5TH.iter().fold(false, |acc, elem| acc || elem.eq(&formated_elem))
        };
        let flag_sixth_row  = if flag_first_row || flag_second_row || flag_third_row || flag_fourth_row || flag_fifth_row {
            false
        } else {
            ELEM6TH.iter().fold(false, |acc, elem| acc || elem.eq(&formated_elem))
        };
        let flag_tm = ELEMTMS.iter().fold(false, |acc, elem| acc || elem.eq(&formated_elem));

        let n_frozen_shell_curr = if flag_tm {
            n_tm += 1;
            n_frozen_shell_2
        } else {
            n_frozen_shell_1
        };

        if flag_first_row {
            n_low_state += 0
        } else if flag_second_row {
            if n_frozen_shell_curr==0 {
                n_low_state += 0
            } else if (n_frozen_shell_curr==1) {
                n_low_state += 1
            } else {
                n_low_state += 0
            }
        } else if flag_third_row {
            if n_frozen_shell_curr==0 {
                n_low_state += 0
            } else if n_frozen_shell_curr==1 {
                n_low_state += 5
            } else if n_frozen_shell_curr==2 {
                n_low_state += 1
            } else {
                n_low_state += 0
            }
        } else if flag_fourth_row {
            if n_frozen_shell_curr==0 {
                n_low_state += 0
            } else if n_frozen_shell_curr==1 {
                n_low_state += 9
            } else if n_frozen_shell_curr==2 {
                n_low_state += 5
            } else if n_frozen_shell_curr==3 {
                n_low_state += 1
            } else {
                n_low_state += 0
            }
        } else if flag_fifth_row {
            if n_frozen_shell_curr==0 {
                n_low_state += 0
            } else if n_frozen_shell_curr==1 {
                n_low_state += 18
            } else if n_frozen_shell_curr==2 {
                // NOTE: for 4d-block elements, 3d orbitals are frozen as well for n_frozen_shell = 2
                if flag_tm {n_low_state += 14} else {n_low_state += 9}
            } else if n_frozen_shell_curr==3 {
                n_low_state += 5
            } else if n_frozen_shell_curr==4 {
                n_low_state += 1
            } else {
                n_low_state += 0
            }
        } else if flag_sixth_row {
            if n_frozen_shell_curr==0 {
                n_low_state += 0
            } else if n_frozen_shell_curr==1 {
                n_low_state += 34
            } else if n_frozen_shell_curr==2 {
                // NOTE: for 5d-block elements, 4d and 4f orbitals are frozen as well for n_frozen_shell = 2
                //       It leads to a core shell with 60 electrons
                if flag_tm {n_low_state += 30} else {n_low_state += 18}
            } else if n_frozen_shell_curr==3 {
                n_low_state += 9
            } else if n_frozen_shell_curr==4 {
                n_low_state += 5
            } else if n_frozen_shell_curr==5 {
                n_low_state += 1
            } else {
                n_low_state += 0
            }
        };
    });

    n_low_state

}