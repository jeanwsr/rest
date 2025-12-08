use serde_json;
use serde::{Deserialize, Serialize};
use crate::ri_gw::get_occupation_parameters;
use crate::scf_io::SCF;
use std::{f64, fs::File, io::Write};


#[derive(Serialize, Deserialize, Debug)]
struct PySOCData{
    natoms:usize,
    elem:Vec<String>,
    position:Vec<f64>,
    num_mo:usize,
    num_ao:usize,
    num_occupied_orbitals:usize,
    num_virtual_orbitals:usize,
    basis_path:String,
    singlet_excitations:Vec<(f64,Vec<f64>)>,
    triplet_excitations:Vec<(f64,Vec<f64>)>,
    mo_energies:Vec<f64>,
    mo_coeff:Vec<f64>,
    ovlp:Vec<f64>
}
pub fn write_pysoc_file(scf_data:&SCF,singlets:Vec<(f64,Vec<f64>)>,triplets:Vec<(f64,Vec<f64>)>)-> Result<(), Box<dyn std::error::Error>>{
    let (start_mo,n_ao,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'Y');
    let (start_mo,n_mo,occ_size,vir_size,homo,lumo)=get_occupation_parameters(scf_data,'N');
    let data=PySOCData{
        natoms:scf_data.mol.geom.elem.len(),
        elem:scf_data.mol.geom.elem.clone(),
        position:scf_data.mol.geom.position.data.clone(),
        num_mo:n_mo,
        num_ao:n_ao,
        num_occupied_orbitals:occ_size,
        num_virtual_orbitals:vir_size,
        basis_path:scf_data.mol.ctrl.basis_path.clone(),
        singlet_excitations:singlets,
        triplet_excitations:triplets,
        mo_energies:scf_data.eigenvalues[0].clone(),
        mo_coeff:scf_data.eigenvectors[0].data.clone(),
        ovlp:scf_data.ovlp.clone().data
    };
    let file = File::create("REST-PySOC Input.json")?;
    serde_json::to_writer_pretty(file, &data)?;
    println!("JSON FILE HAS BEEN GENERATED!");
    Ok(())
}