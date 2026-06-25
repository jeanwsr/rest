use std::fs::File;
use std::io::Write;
use crate::scf_io::SCF;
use num_complex::Complex;
use std::ffi::c_char;
use std::f64::consts::PI;
use rayon::prelude::*;
use approx::abs_diff_eq;
use std::collections::HashMap;
use crate::constants::ANG;
use crate::geom_io::{GeomUnit,get_charge};
use crate::molecule_io::Molecule;

pub fn rrs_pbc_output(scf_data: &SCF, unit_cell_elem: Vec<String>, index_map: HashMap<(i32,i32,i32),Vec<usize>>) {
    println!("RRS-PBC method invoked");
    let (real_k,pbc_result) = rrs_pbc_new(&scf_data,index_map);
    let charge_vec = get_charge(&unit_cell_elem);
    let mut cell_charge = 0.0;
    charge_vec.iter().for_each(|c| { cell_charge += c});
    println!("RRS-PBC results:");
    match scf_data.mol.ctrl.pbc_eigenval.clone() {
        None => {
            println!("RRS-PBC dimension: {:?}",scf_data.mol.geom.pbc_dim);
            println!("Unit Cell: {:?}",unit_cell_elem);
            println!("Occupied orbitals: {:?}",(cell_charge/2.0) as usize);
            real_k.par_iter().zip(pbc_result.par_iter()).for_each(|(k,res)| {
                println!("k = {:?}",k);
                println!("Orbital energy = {:?}",res);
            });
        },
        other => {
            let file_name = other.unwrap();
            let mut file = File::create(file_name).unwrap();
            writeln!(&mut file,"RRS-PBC dimension: {:?}",scf_data.mol.geom.pbc_dim);
            writeln!(&mut file,"Unit Cell: {:?}",unit_cell_elem);
            writeln!(&mut file,"Occupied orbitals: {:?}",(cell_charge/2.0) as usize);
            let result: Vec<String> = real_k.par_iter()
                .zip(pbc_result.par_iter())
                .map(|(a_val, b_val)| format!("k = {:?}\nOrbital energy = {:?}\n",a_val,b_val))
                .collect();
            let result_string = result.join("");
            writeln!(&mut file,"{}",result_string);
        }
    };
    let mut tot_k_points = 0 as usize;
    if scf_data.mol.geom.pbc_dim == 1 {
        tot_k_points = scf_data.mol.geom.k_points[0];
    } else if scf_data.mol.geom.pbc_dim == 2 {
        tot_k_points = scf_data.mol.geom.k_points[0] * scf_data.mol.geom.k_points[1];
    } else if scf_data.mol.geom.pbc_dim == 3 {
        tot_k_points = scf_data.mol.geom.k_points[0] * scf_data.mol.geom.k_points[1] * scf_data.mol.geom.k_points[2];
    }
    let mut homo_lumo_gap = vec![0.0;tot_k_points];
    homo_lumo_gap.iter_mut().zip(pbc_result.iter()).for_each(|(h,p)| {
        *h = (p[(cell_charge/2.0) as usize] - p[(cell_charge/2.0) as usize - 1])*27.2113863;
    });
    let min_hlg = homo_lumo_gap.iter().filter(|&&x| !x.is_nan()).min_by(|a, b| a.partial_cmp(b).unwrap());
    println!("HOCO-LUCO gap: {:?} eV",min_hlg.unwrap());
    let elec = scf_data.mol.num_elec[0];
    let hlg = (scf_data.eigenvalues[0][(elec/2.0) as usize] - scf_data.eigenvalues[0][(elec/2.0) as usize - 1])*27.2113863;
    println!("HOMO-LUMO gap: {:?} eV",hlg);
}

pub fn rrs_pbc_check(mol: &Molecule) {
    let spin = mol.ctrl.spin;
    if !(abs_diff_eq!(spin,1.0,epsilon=1e-6)) {
        panic!("Error: RRS-PBC is only supported for closed-shell systems currently.");
    }
    let rrs_pbc_dim = mol.geom.pbc_dim;
    let pbc_vec_dim = mol.geom.rrs_pbc_vec.data.len();
    if pbc_vec_dim < (rrs_pbc_dim * (3 as usize)) {
        panic!("RRS-PBC vector does not match PBC dimension! Find vector length = {:?} and dimension = {:?}.",pbc_vec_dim,rrs_pbc_dim);
    }
    if pbc_vec_dim > (rrs_pbc_dim * (3 as usize)) {
        println!("Warning: RRS-PBC vector is too long.");
    }
    let step_len = mol.geom.max_step.len();
    let k_points_len = mol.geom.k_points.len();
    if step_len < rrs_pbc_dim {
        panic!("Max step vector does not match PBC dimension! Find vector length = {:?} and dimension = {:?}.",step_len,rrs_pbc_dim);
    }
    if step_len > rrs_pbc_dim {
        println!("Warning: max step vector is too long.");
    }
    if k_points_len < rrs_pbc_dim {
        panic!("K_points vector does not match PBC dimension! Find vector length = {:?} and dimension = {:?}.",k_points_len,rrs_pbc_dim);
    }
    if k_points_len > rrs_pbc_dim {
        println!("Warning: k_points vector is too long.");
    }
}

pub fn rrs_pbc_match(mol: &Molecule) -> (HashMap<(i32,i32,i32),Vec<usize>>, Vec<String>) {
    //unit: GeomUnit
    //rrs_pbc_dim: usize
    //position: MatrixFull<f64>
    //unit_cell_index: Vec<usize>
    //rrs_pbc_vec: MatrixFull<f64>
    //max_step: Vec<usize>
    let unit = mol.geom.unit.clone();
    let rrs_pbc_dim = mol.geom.pbc_dim;
    let position = mol.geom.position.clone();
    let unit_cell_index = mol.geom.unit_cell_index.clone();
    let rrs_pbc_vec = mol.geom.rrs_pbc_vec.clone();
    let max_step = mol.geom.max_step.clone();
    let mut full_elem: Vec<String> = vec![];
    mol.geom.elem.iter().for_each(|elem| {
        full_elem.push(String::from(elem.clone()));
    });

    if rrs_pbc_dim > 3 || rrs_pbc_dim <= 0 {
        panic!("Illegal RRS-PBC dim: {:?}",rrs_pbc_dim);
    };
    //transfer position to Vec<Vec>>
    let vec_pos = position.data;
    let num_elem = vec_pos.len() / 3;
    let mut tot_pos = vec![vec![0.0;3];num_elem]; //[N_elem,3]
    for i in 0..num_elem {
        let tmp = vec_pos[3*i..(3*i+3)].to_vec();
        let mut tmp2 = &mut tot_pos[i];
        tmp2.iter_mut().zip(tmp.iter()).for_each(|(dst,src)| {
            *dst = *src;
        });
    };
    //transfer rrs_pbc_vec to Vec<Vec<>>
    let vec_rrs_pbc_vec = rrs_pbc_vec.data;
    let mut tot_pbc_vec = vec![vec![0.0;3];3]; //[3,3]
    let factor = match unit {
        GeomUnit::Angstrom => ANG,
        GeomUnit::Bohr => 1.0,
    };
    for i in 0..rrs_pbc_dim {
        let tmp = vec_rrs_pbc_vec[3*i..(3*i+3)].to_vec();
        let mut tmp2 = &mut tot_pbc_vec[i];
        tmp2.iter_mut().zip(tmp.iter()).for_each(|(dst,src)| {
            *dst = *src / factor;
        });
    };
    //get core position
    let num_unit_elem = unit_cell_index.len();
    let mut unit_elem_vec = vec![];
    unit_cell_index.iter().for_each(|idx| {
        unit_elem_vec.push(full_elem[*idx].clone());
    });
    let mut core_unit_positon = vec![vec![0.0;3];num_unit_elem]; //N_units,3
    unit_cell_index.iter().zip(core_unit_positon.iter_mut()).for_each(|(c,cp)| {
        *cp = vec_pos[3*c..(3*c+3)].to_vec();
    });
    let mut index_map: HashMap<(i32,i32,i32), Vec<usize>> = HashMap::new(); //(p_value) -> elem_idx
    let mut steps = (0 as i32,0 as i32, 0 as i32);
    if rrs_pbc_dim == 1 {
        steps = (max_step[0] as i32,0 as i32,0 as i32);
    } else if rrs_pbc_dim == 2 {
        steps = (max_step[0] as i32,max_step[1] as i32,0 as i32);
    } else if rrs_pbc_dim == 3 {
        steps = (max_step[0] as i32,max_step[1] as i32,max_step[2] as i32);
    } else {
        panic!("Illegal RRS-PBC dim!");
    };

    //both edge will be considered
    for step_x in (-steps.0)..(steps.0 + 1 as i32) {
        for step_y in (-steps.1)..(steps.1 + 1 as i32) {
            for step_z in (-steps.2)..(steps.2 + 1 as i32) {
                let tmp_p_value = (step_x,step_y,step_z);
                let mut tmp_elem_idx = vec![];
                
                let mut tmp_unit = vec![vec![0.0;3];num_unit_elem];
                tmp_unit.iter_mut().enumerate().zip(core_unit_positon.iter()).for_each(|((dst_idx,dst),src)| {
                    dst.iter_mut().zip(src.iter()).zip(tot_pbc_vec[0].iter()).zip(tot_pbc_vec[1].iter()).zip(tot_pbc_vec[2].iter()).for_each(|((((dd,ss),x),y),z)| {
                        *dd = *ss + (step_x as f64) * *x + (step_y as f64) * *y + (step_z as f64) * *z;
                    });
                    //try to match tmp_unit
                    tot_pos.iter().enumerate().for_each(|(pidx,p)| {
                        let bool_match = abs_diff_eq!(p[0],dst[0],epsilon=1e-6) && abs_diff_eq!(p[1],dst[1],epsilon=1e-6) && abs_diff_eq!(p[2],dst[2],epsilon=1e-6) && (full_elem[pidx] == unit_elem_vec[dst_idx]);
                        if bool_match {
                            tmp_elem_idx.push(pidx);
                        };
                    });
                });
                if tmp_elem_idx.len() == num_unit_elem { //match successfully
                    index_map.entry(tmp_p_value).or_insert(tmp_elem_idx);
                } else {
                    println!("Warning: RRS-PBC step path {:?} is not found.",(step_x,step_y,step_z));
                }
            }
        }
    }
    (index_map,unit_elem_vec)
}
pub fn rrs_pbc_new(scf: &SCF,index_map: HashMap<(i32,i32,i32),Vec<usize>>) -> (Vec<Vec<f64>>,Vec<Vec<f64>>) {
    //return (k,w)
    let rrs_pbc_vec = scf.mol.geom.rrs_pbc_vec.clone();
    let rrs_pbc_dim = scf.mol.geom.pbc_dim;
    let unit = scf.mol.geom.unit.clone();
    let factor = match unit {
        GeomUnit::Angstrom => ANG,
        GeomUnit::Bohr => 1.0,
    };
    let mut vec_rrs_pbc_vec = rrs_pbc_vec.data.clone();
    vec_rrs_pbc_vec.iter_mut().for_each(|v| {
        *v /= factor;
    });
    let k_points_vec = scf.mol.geom.k_points.clone();
    //match elem to index in hamiltonian
    let mut hamiltonian_map: HashMap<(i32,i32,i32),Vec<(usize,usize)>> = HashMap::new();
    let mut num_units = 0 as usize;
    for (key,elem_index) in index_map {
        num_units += 1;
        let mut hamiltonian_index = vec![(0 as usize,0 as usize);elem_index.len()];
        elem_index.iter().zip(hamiltonian_index.iter_mut()).for_each(|(elem, ham)| {
            *ham = scf.mol.basis4elem[*elem].global_index.clone();
        });
        hamiltonian_map.insert(key,hamiltonian_index); //same as basis4elem, (start_idx, length)
    }
    //get unit hamiltonian size and core_index
    let mut unit_hamiltonian_size = 0 as usize;
    let mut core_index: Vec<usize> = vec![];
    hamiltonian_map.get(&(0,0,0)).unwrap().iter().for_each(|tup| {
        unit_hamiltonian_size += tup.1;
        for i in (tup.0)..(tup.0+tup.1) {
            core_index.push(i);
        };
    });
    //fold hamiltonian and overlap
    //save as hashmap
    let fock = scf.hamiltonian[0].clone().to_matrixfull().unwrap();
    let ovlp = scf.ovlp.clone().to_matrixfull().unwrap();
    let hamiltonian_size = fock.size[0];
    let mut fold_fock: HashMap<(i32,i32,i32),Vec<f64>> = HashMap::new();
    let mut fold_ovlp: HashMap<(i32,i32,i32),Vec<f64>> = HashMap::new();
    for (key,row) in hamiltonian_map {
        let mut tmp_fock = vec![0.0;unit_hamiltonian_size*unit_hamiltonian_size];
        let mut tmp_ovlp = vec![0.0;unit_hamiltonian_size*unit_hamiltonian_size];
        let mut tmp_index = vec![];
        row.iter().for_each(|tup| {
            for i in (tup.0)..(tup.0+tup.1) {
                tmp_index.push(i);
            };
        });
        tmp_index.iter().enumerate().for_each(|(c_idx,c)| {
            core_index.iter().enumerate().for_each(|(r_idx,r)| {
                tmp_fock[r_idx+c_idx*unit_hamiltonian_size] = fock.data[*r + *c * hamiltonian_size].clone();
                tmp_ovlp[r_idx+c_idx*unit_hamiltonian_size] = ovlp.data[*r + *c * hamiltonian_size].clone();
            });
        });
        fold_fock.insert(key,tmp_fock);
        fold_ovlp.insert(key,tmp_ovlp);
    };
    let mut tot_k_points = 0 as usize;
    if rrs_pbc_dim == 1 {
        tot_k_points = k_points_vec[0].clone() + 1;
    } else if rrs_pbc_dim == 2 {
        tot_k_points = (k_points_vec[0].clone()+1) * (k_points_vec[1].clone()+1);
    } else if rrs_pbc_dim == 3 {
        tot_k_points = (k_points_vec[0].clone()+1) * (k_points_vec[1].clone()+1) * (k_points_vec[2].clone()+1);
    } else {
        panic!("Find illegal PBC dimension: {:?}",rrs_pbc_dim);
    }
    //real_k save all k values, and it has the same order as real_w, k_fock and k_ovlp
    let mut k_fock = vec![vec![Complex::new(0.0,0.0);unit_hamiltonian_size*unit_hamiltonian_size];tot_k_points];
    let mut k_ovlp = vec![vec![Complex::new(0.0,0.0);unit_hamiltonian_size*unit_hamiltonian_size];tot_k_points];
    let mut real_k = vec![vec![0.0;3];tot_k_points];
    if rrs_pbc_dim == 1 {
        let k_points = k_points_vec[0];
        k_fock.par_iter_mut()
            .zip(k_ovlp.par_iter_mut())
            .zip(real_k.par_iter_mut())
            .enumerate()
            .for_each(|(idx,((fock_item,ovlp_item),k_item))| {
                let k = idx;
                let mut tmp_fock = vec![Complex::new(0.0,0.0);unit_hamiltonian_size*unit_hamiltonian_size];
                let mut tmp_ovlp = vec![Complex::new(0.0,0.0);unit_hamiltonian_size*unit_hamiltonian_size];
                let fac = (k as f64)/(k_points as f64) - 0.5;
                let pre_factor = 2.0*PI*fac;
                fold_fock.iter().for_each(|(p,f)| {
                    let factor = Complex::new(0.0,pre_factor*(p.0 as f64)).exp();
                    tmp_fock.iter_mut().zip(f.iter()).for_each(|(tf,ff)| {
                        *tf += factor * Complex::new(*ff,0.0);
                    });
                });
                fold_ovlp.iter().for_each(|(p,f)| {
                    let factor = Complex::new(0.0,pre_factor*(p.0 as f64)).exp();
                    tmp_ovlp.iter_mut().zip(f.iter()).for_each(|(tf,ff)| {
                        *tf += factor * Complex::new(*ff,0.0);
                    });
                });
                let tmp_k = vec![fac,0.0,0.0];
                *fock_item = tmp_fock;
                *ovlp_item = tmp_ovlp;
                *k_item = tmp_k;
        });
    } else if rrs_pbc_dim == 2 {
        //get p_vec
        let vec1 = vec_rrs_pbc_vec[0..3].to_vec().clone();
        let vec2 = vec_rrs_pbc_vec[3..6].to_vec().clone();
        let vec3 = cross_product(&vec1,&vec2);
        //create k_vec
        let mut k_vec1 = cross_product(&vec2,&vec3);
        let mut k_vec2 = cross_product(&vec3,&vec1);
        //normalization, vecx \cdot k_vecx = 2PI
        let mut tmp = 0.0;
        k_vec1.iter().zip(vec1.iter()).for_each(|(x,y)| {
            tmp += *x * *y;
        });
        k_vec1.iter_mut().for_each(|x| {
            *x *= 2.0 * PI / tmp;
        });

        let mut tmp = 0.0;
        k_vec2.iter().zip(vec2.iter()).for_each(|(x,y)| {
            tmp += *x * *y;
        });
        k_vec2.iter_mut().for_each(|x| {
            *x *= 2.0 * PI / tmp;
        });

        let k_points_1 = k_points_vec[0];
        let k_points_2 = k_points_vec[1];

        k_fock.par_iter_mut()
            .zip(k_ovlp.par_iter_mut())
            .zip(real_k.par_iter_mut())
            .enumerate()
            .for_each(|(idx,((fock_item,ovlp_item),k_item))| {
                let kx = idx % (k_points_1+1);
                let ky = idx / (k_points_1+1);
                let mut tmp_fock = vec![Complex::new(0.0,0.0);unit_hamiltonian_size*unit_hamiltonian_size];
                let mut tmp_ovlp = vec![Complex::new(0.0,0.0);unit_hamiltonian_size*unit_hamiltonian_size];
                let mut pre_factor1 = 2.0 * (kx as f64)/(k_points_1 as f64) - 1.0;
                let mut pre_factor2 = 2.0 * (ky as f64)/(k_points_2 as f64) - 1.0;
                let mut k_vec = vec![0.0;3];
                k_vec1.iter().zip(k_vec2.iter()).zip(k_vec.iter_mut()).for_each(|((v1,v2),v)| {
                    *v = pre_factor1 * *v1 + pre_factor2 * *v2;
                });
                fold_fock.iter().for_each(|(p,f)| {
                    let mut tmp_p = vec![0.0;3];
                    tmp_p.iter_mut().zip(vec1.iter()).zip(vec2.iter()).for_each(|((pp,v1),v2)| {
                        *pp = (p.0 as f64) * *v1 + (p.1 as f64) * *v2;
                    });
                    let mut tmp_factor = 0.0;
                    tmp_p.iter().zip(k_vec.iter()).for_each(|(pp,kk)| {
                        tmp_factor += *pp * *kk;
                    });
                    let factor = Complex::new(0.0,tmp_factor).exp();
                    tmp_fock.iter_mut().zip(f.iter()).for_each(|(tf,ff)| {
                        *tf += factor * Complex::new(*ff,0.0);
                    });
                });
                fold_ovlp.iter().for_each(|(p,f)| {
                    let mut tmp_p = vec![0.0;3];
                    tmp_p.iter_mut().zip(vec1.iter()).zip(vec2.iter()).for_each(|((pp,v1),v2)| {
                        *pp = (p.0 as f64) * *v1 + (p.1 as f64) * *v2;
                    });
                    let mut tmp_factor = 0.0;
                    tmp_p.iter().zip(k_vec.iter()).for_each(|(pp,kk)| {
                        tmp_factor += *pp * *kk;
                    });
                    let factor = Complex::new(0.0,tmp_factor).exp();
                    tmp_ovlp.iter_mut().zip(f.iter()).for_each(|(tf,ff)| {
                        *tf += factor * Complex::new(*ff,0.0);
                    });
                });
                let tmp_k = vec![pre_factor1,pre_factor2,0.0];
                *fock_item = tmp_fock;
                *ovlp_item = tmp_ovlp;
                *k_item = tmp_k;
            });
    } else if rrs_pbc_dim == 3 {
        let vec1 = vec_rrs_pbc_vec[0..3].to_vec().clone();
        let vec2 = vec_rrs_pbc_vec[3..6].to_vec().clone();
        let vec3 = vec_rrs_pbc_vec[6..9].to_vec().clone();
        let mut k_vec1 = cross_product(&vec2,&vec3);
        let mut k_vec2 = cross_product(&vec3,&vec1);
        let mut k_vec3 = cross_product(&vec1,&vec2);
        //normalization, vecx \cdot k_vecx = 2PI
        let mut tmp = 0.0;
        k_vec1.iter().zip(vec1.iter()).for_each(|(x,y)| {
            tmp += *x * *y;
        });
        k_vec1.iter_mut().for_each(|x| {
            *x *= 2.0 * PI / tmp;
        });

        let mut tmp = 0.0;
        k_vec2.iter().zip(vec2.iter()).for_each(|(x,y)| {
            tmp += *x * *y;
        });
        k_vec2.iter_mut().for_each(|x| {
            *x *= 2.0 * PI / tmp;
        });

        let mut tmp = 0.0;
        k_vec3.iter().zip(vec3.iter()).for_each(|(x,y)| {
            tmp += *x * *y;
        });
        k_vec3.iter_mut().for_each(|x| {
            *x *= 2.0 * PI / tmp;
        });

        let k_points_1 = k_points_vec[0];
        let k_points_2 = k_points_vec[1];
        let k_points_3 = k_points_vec[2];

        k_fock.par_iter_mut()
            .zip(k_ovlp.par_iter_mut())
            .zip(real_k.par_iter_mut())
            .enumerate()
            .for_each(|(idx,((fock_item,ovlp_item),k_item))| {
                let kx = idx / (k_points_2 * k_points_3);
                let remainder = idx % (k_points_2 * k_points_3);
                let ky = remainder / k_points_3;
                let kz = remainder % k_points_3;
                let mut tmp_fock = vec![Complex::new(0.0,0.0);unit_hamiltonian_size*unit_hamiltonian_size];
                let mut tmp_ovlp = vec![Complex::new(0.0,0.0);unit_hamiltonian_size*unit_hamiltonian_size];
                let pre_factor1 = 2.0 * (kx as f64)/(k_points_1 as f64) - 1.0;
                let pre_factor2 = 2.0 * (ky as f64)/(k_points_2 as f64) - 1.0;
                let pre_factor3 = 2.0 * (kz as f64)/(k_points_3 as f64) - 1.0;
                let mut k_vec = vec![0.0;3];
                k_vec1.iter().zip(k_vec2.iter()).zip(k_vec3.iter()).zip(k_vec.iter_mut()).for_each(|(((v1,v2),v3),v)| {
                    *v = pre_factor1 * *v1 + pre_factor2 * *v2 + pre_factor3 * *v3; 
                });
                fold_fock.iter().for_each(|(p,f)| {
                    let mut tmp_p = vec![0.0;3];
                    tmp_p.iter_mut().zip(vec1.iter()).zip(vec2.iter()).zip(vec3.iter()).for_each(|(((pp,v1),v2),v3)| {
                        *pp = (p.0 as f64) * *v1 + (p.1 as f64) * *v2 +(p.2 as f64) * *v3;
                    });
                    let mut tmp_factor = 0.0;
                    tmp_p.iter().zip(k_vec.iter()).for_each(|(pp,kk)| {
                        tmp_factor += *pp * *kk;
                    });
                    let factor = Complex::new(0.0,tmp_factor).exp();
                    tmp_fock.iter_mut().zip(f.iter()).for_each(|(tf,ff)| {
                        *tf += factor * Complex::new(*ff,0.0);
                    });
                });
                fold_ovlp.iter().for_each(|(p,f)| {
                    let mut tmp_p = vec![0.0;3];
                    tmp_p.iter_mut().zip(vec1.iter()).zip(vec2.iter()).zip(vec3.iter()).for_each(|(((pp,v1),v2),v3)| {
                        *pp = (p.0 as f64) * *v1 + (p.1 as f64) * *v2 +(p.2 as f64) * *v3;
                    });
                    let mut tmp_factor = 0.0;
                    tmp_p.iter().zip(k_vec.iter()).for_each(|(pp,kk)| {
                        tmp_factor += *pp * *kk;
                    });
                    let factor = Complex::new(0.0,tmp_factor).exp();
                    tmp_ovlp.iter_mut().zip(f.iter()).for_each(|(tf,ff)| {
                        *tf += factor * Complex::new(*ff,0.0);
                    });
                });
                let tmp_k = vec![pre_factor1,pre_factor2,pre_factor3];
                *fock_item = tmp_fock;
                *ovlp_item = tmp_ovlp;
                *k_item = tmp_k;
        });
    } else {
        panic!("Illegal PBC dimension");
    };
    //make all k_xxx hermitian
    let mut herm_k_fock = vec![vec![Complex::new(0.0,0.0);unit_hamiltonian_size*unit_hamiltonian_size];tot_k_points];
    herm_k_fock.par_iter_mut().zip(k_fock.par_iter()).for_each(|(pf,f)| {
        for i in 0..unit_hamiltonian_size {
            for j in 0..unit_hamiltonian_size {
                pf[i+j*unit_hamiltonian_size] = (f[i+j*unit_hamiltonian_size] + f[j+i*unit_hamiltonian_size].conj())/2.0;
            };
        };
    });
    let mut herm_k_ovlp = vec![vec![Complex::new(0.0,0.0);unit_hamiltonian_size*unit_hamiltonian_size];tot_k_points];
    herm_k_ovlp.par_iter_mut().zip(k_ovlp.par_iter()).for_each(|(pf,f)| {
        for i in 0..unit_hamiltonian_size {
            for j in 0..unit_hamiltonian_size {
                pf[i+j*unit_hamiltonian_size] = (f[i+j*unit_hamiltonian_size] + f[j+i*unit_hamiltonian_size].conj())/2.0;
            };
        };
    });

    //solve eigvalues
    let mut w = vec![vec![Complex::new(0.0,0.0);unit_hamiltonian_size];tot_k_points];
    herm_k_fock.par_iter_mut().zip(herm_k_ovlp.par_iter_mut()).zip(w.par_iter_mut()).for_each(|((hf,ho),ww)| {
        let jobvl = b'N' as c_char;
        let jobvr = b'V' as c_char;
        let n = unit_hamiltonian_size.clone() as i32;
        let mut alpha = vec![Complex::new(0.0,0.0);unit_hamiltonian_size];
        let mut beta = vec![Complex::new(0.0,0.0);unit_hamiltonian_size];
        let mut vr = vec![Complex::new(0.0,0.0);unit_hamiltonian_size*unit_hamiltonian_size];
        let lwork = 2 * n;
        let mut work = vec![Complex::new(0.0,0.0);lwork as usize];
        let mut rwork = vec![0.0;(8 * n) as usize];
        let mut info = 0 as i32;
        unsafe {
            zggev_(
                &jobvl as *const c_char,
                &jobvr as *const c_char,
                &n as *const i32,
                hf.as_mut_ptr(),
                &n as *const i32,
                ho.as_mut_ptr(),
                &n as *const i32,
                alpha.as_mut_ptr(),
                beta.as_mut_ptr(),
                std::ptr::null_mut(),
                &n as *const i32,
                vr.as_mut_ptr(),
                &n as *const i32,
                work.as_mut_ptr(),
                &lwork as *const i32,
                rwork.as_mut_ptr(),
                &mut info as *mut i32)
        }
        if info != 0 {
            println!("ZGGEV returns INFO: {:?}",info);
        }
        ww.iter_mut().zip(alpha.iter()).zip(beta.iter()).for_each(|((www,aa),bb)| {
            *www = *aa / *bb;
        });
    });

    let mut real_w = vec![vec![0.0;unit_hamiltonian_size];tot_k_points];
    real_w.iter_mut().zip(w.iter()).for_each(|(rw,cw)| {
        rw.iter_mut().zip(cw.iter()).for_each(|(rwi,cwi)| {
            *rwi = cwi.re;
        });
    });

    real_w.par_iter_mut().for_each(|ww| {
        ww.sort_by(|a,b| a.partial_cmp(b).unwrap());
    });

    (real_k,real_w)
}

pub fn cross_product(vec1: &Vec<f64>, vec2: &Vec<f64>) -> Vec<f64> {
    let l1 = vec1.len();
    let l2 = vec2.len();
    let mut v3 = vec![];
    if (l1 == 3) && (l2 == 3) {
        v3 = vec![vec1[1] * vec2[2] - vec1[2] * vec2[1],vec1[2] * vec2[0] - vec1[0] * vec2[2],vec1[0] * vec2[1] - vec1[1] * vec2[0]];
    } else {
        panic!("Cross product if only implemented for 3-Vec. Find dim1 = {:?} and dim2 = {:?}",l1,l2);
    }
    v3
}

#[link(name="lapack")]
extern "C" {
    fn zgemm_(
        transa: *const c_char,
        transb: *const c_char,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const Complex<f64>,
        a: *const Complex<f64>,
        lda: *const i32,
        b: *const Complex<f64>,
        ldb: *const i32,
        beta: *const Complex<f64>,
        c: *mut Complex<f64>,
        ldc: *const i32,
    );
}

#[link(name="lapack")]
extern "C" {
    fn zgeev_(
        jobvl: *const c_char,
        jobvr: *const c_char,
        n: *const i32,
        a: *const Complex<f64>,
        lda: *const i32,
        w: *mut Complex<f64>,
        vl: *mut Complex<f64>,
        ldvl: *const i32,
        vr: *mut Complex<f64>,
        ldvr: *const i32,
        work: *mut Complex<f64>,
        lwork: *const i32,
        rwork: *mut f64,
        info: *mut i32,
        );
}

#[link(name="lapack")]
extern "C" {
    fn zpotrf_(
        uplo: *const c_char,
        n: *const i32,
        a: *mut Complex<f64>,
        lda: *const i32,
        info: *mut i32);
}

#[link(name="lapack")]
extern "C" {
    fn ztrtrs_(
        uplo: *const c_char,
        trans: *const c_char,
        diag: *const c_char,
        n: *const i32,
        nrhs: *const i32,
        a: *mut Complex<f64>,
        lda: *const i32,
        b: *mut Complex<f64>,
        ldb: *const i32,
        info: *mut i32);
}

#[link(name="lapack")]
extern "C" {
    fn zggev_(
        jobvl: *const c_char,
        jobvr: *const c_char,
        n: *const i32,
        a: *mut Complex<f64>,
        lda: *const i32,
        b: *mut Complex<f64>,
        ldb: *const i32,
        alpha: *mut Complex<f64>,
        beta: *mut Complex<f64>,
        vl: *mut Complex<f64>,
        ldvl: *const i32,
        vr: *mut Complex<f64>,
        ldvr: *const i32,
        work: *mut Complex<f64>,
        lwork: *const i32,
        rwork: *mut f64,
        info: *mut i32,
        );
}
