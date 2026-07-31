use std::collections::HashMap;
use std::path::Path;
use hdf5;
use hdf5::types::VarLenUnicode;
// use hdf5::types::TypeDescriptor;
use crate::scf_io::{SCF, SCFType};
use tensors::matrix::MatrixFull;
use crate::geom_io::{GeomCell, GeomUnit, MOrC, get_mass_charge};
use crate::external_field::extfield::ExtField;
use crate::basis_io::Basis4Elem;
use rest_libcint::CintType;
use crate::constants::BOHR;

pub fn write_scf_attribute<T>(group: &hdf5::Group, dataset_name: &str, value: &[T]) 
where 
    T: Clone + hdf5::H5Type
{
    if let Ok(dataset) = group.dataset(dataset_name) {
        match dataset.write_raw(value) {
            Ok(_) => (),
            Err(e) => println!("Error writing dataset {}: {:?}", dataset_name, e),
        }
    } else {
        let builder = group.new_dataset_builder();
        match builder.with_data(value).create(dataset_name) {
            Ok(_) => (),
            Err(e) => println!("Error creating dataset {}: {:?}", dataset_name, e),
        }
    }
}

pub fn write_string_scalar(file: &hdf5::File, dataset_name: &str, value: &str) {
    use std::str::FromStr;
    let unicode_string = VarLenUnicode::from_str(value).unwrap();
    
    if let Ok(dataset) = file.dataset(dataset_name) {
        // Dataset exists - overwrite it
        dataset.write_scalar(&unicode_string).unwrap();
    } else {
        // Dataset doesn't exist - create it with string type
        let ds = file.new_dataset::<VarLenUnicode>().shape(())
            .create(dataset_name).unwrap();
        ds.write_scalar(&unicode_string).unwrap();
    }
}

pub fn save_chkfile(scf_data: &SCF) {
    let chkfile= &scf_data.mol.ctrl.chkfile;
    let path = Path::new(chkfile);
    //if path.exists() {std::fs::remove_file(chkfile).unwrap()};
    //let file = hdf5::File::create(chkfile).unwrap();
    //let scf = file.create_group("scf").unwrap();
    let file = if path.exists() {
        hdf5::File::open_rw(chkfile).unwrap()
    } else {
        hdf5::File::create(chkfile).unwrap()
    };
    println!("write chkfile: {}", chkfile);
    let is_exist = file.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("scf")});
    let scf = if is_exist {
        file.group("scf").unwrap()
    } else {
        file.create_group("scf").unwrap()
    };

    write_scf_attribute(&scf, "e_tot", &[scf_data.scf_energy]);
    write_scf_attribute(&scf, "num_basis", &[scf_data.mol.num_basis]);
    write_scf_attribute(&scf, "spin_channel", &[scf_data.mol.spin_channel]);
    write_scf_attribute(&scf, "num_states", &[scf_data.mol.num_state]);
    write_scf_attribute(&scf, "spin", &[scf_data.mol.ctrl.spin]);
    write_scf_attribute(&scf, "charge", &[scf_data.mol.ctrl.charge]);

    // let is_exist = scf.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("mo_coeff")});
    let mut eigenvectors: Vec<f64> = vec![];
    for i_spin in 0..scf_data.mol.spin_channel {
        let tmp_eigenvectors = scf_data.eigenvectors[i_spin].transpose();
        eigenvectors.extend(tmp_eigenvectors.data.iter());
        if let SCFType::ROHF = scf_data.scftype { // ROHF: only process i_spin = 0, since alpha/beta eigenvectors are the same
            break
        }
    }
    // println!("eigenvectors {:?}", scf_data.eigenvectors);
    write_scf_attribute(&scf, "mo_coeff", &eigenvectors);

    let mut eigenvalues: Vec<f64> = vec![];
    for i_spin in 0..scf_data.mol.spin_channel {
        eigenvalues.extend(scf_data.eigenvalues[i_spin].iter());
        if let SCFType::ROHF = scf_data.scftype { // ROHF: only process i_spin = 0, since alpha/beta eigenvalues are the same
            break 
        }
    }
    write_scf_attribute(&scf, "mo_energy", &eigenvalues);

    let mut occ: Vec<f64> = vec![];
    for i_spin in 0..scf_data.mol.spin_channel {
        occ.extend(scf_data.occupation[i_spin].iter());
    }
    // println!("occupation {:?}", scf_data.occupation);
    // for compatibility with old rest, may be removed in the future
    write_scf_attribute(&scf, "mo_occupation", &occ);
    // for compatibility with pyscf
    write_scf_attribute(&scf, "mo_occ", &occ);

    let mol = &scf_data.mol;
    // let (atm, bas, env) = (mol.cint_atm.clone(), mol.cint_bas.clone(), mol.cint_env.clone());
    // convert hashmap of atm, bas, env to one json string
    let mol_info = serde_json::to_string(&serde_json::json!({
        "_atm": mol.cint_atm.clone(),
        "_bas": mol.cint_bas.clone(),
        "_ecpbas": mol.cint_ecpbas.clone(),
        "_env": mol.cint_env.clone(),
    })).unwrap();
    write_string_scalar(&file, "mol", &mol_info);
    let basis4elem = serde_json::to_string(&mol.basis4elem).unwrap();
    write_string_scalar(&file, "molecule/basis4elem", &basis4elem);
    let cinttype = cint_type_as_str(&mol.cint_type);
    write_string_scalar(&file, "molecule/cinttype", &cinttype);

    let geom = &scf_data.mol.geom;
    let geom_json = serde_json::to_string(&serde_json::json!({
        "name":     geom.name.clone(),
        "elem":     geom.elem.clone(),
        "unit":     geom.unit,
        "position": geom.position.iter().copied().collect::<Vec<f64>>(),
    })).unwrap();
    write_string_scalar(&file, "molecule/geom", &geom_json);

    file.close();
}

pub fn cint_type_as_str(cint_type: &CintType) -> &'static str {
    match cint_type {
        CintType::Spheric => "spheric",
        CintType::Cartesian => "cartesian",
        CintType::Spinor => "spinor",
    }
}

pub fn save_hamiltonian(scf_data: &SCF) {
    let chkfile= &scf_data.mol.ctrl.chkfile;
    let path = Path::new(chkfile);
    let file = if path.exists() {
        hdf5::File::open_rw(chkfile).unwrap()
    } else {
        hdf5::File::create(chkfile).unwrap()
    };
    let is_exist = file.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("scf")});
    let scf = if is_exist {
        file.group("scf").unwrap()
    } else {
        file.create_group("scf").unwrap()
    };
    let mut hamiltonians: Vec<f64> = vec![];
    for i_spin in 0..scf_data.mol.spin_channel {
        let tmp_eigenvectors = scf_data.hamiltonian[i_spin].to_matrixfull().unwrap();
        hamiltonians.extend(tmp_eigenvectors.data);
        // hamiltonians.extend(scf_data.eigenvalues[i_spin].iter());
    }
    let is_hamiltonian = scf.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("hamiltonian")});
    if is_hamiltonian {
        let dataset = scf.dataset("hamiltonian").unwrap();
        dataset.write(&hamiltonians);
    } else {
        let builder = scf.new_dataset_builder();
        builder.with_data(&hamiltonians).create("hamiltonian");
    };
    file.close();
}

pub fn save_overlap(scf_data: &SCF) {
    let chkfile= &scf_data.mol.ctrl.chkfile;
    let path = Path::new(chkfile);
    let file = if path.exists() {
        hdf5::File::open_rw(chkfile).unwrap()
    } else {
        hdf5::File::create(chkfile).unwrap()
    };
    let is_exist = file.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("scf")});
    let scf = if is_exist {
        file.group("scf").unwrap()
    } else {
        file.create_group("scf").unwrap()
    };
    // let mut overlap: Vec<f64> = vec![];
    let overlap = scf_data.ovlp.to_matrixfull().unwrap();
    let is_exist = scf.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("overlap")});
    if is_exist {
        let dataset = scf.dataset("overlap").unwrap();
        dataset.write(&overlap.data);
    } else {
        let builder = scf.new_dataset_builder();
        builder.with_data(&overlap.data).create("overlap");
    };
    file.close();
}

pub fn save_geometry(scf_data: &SCF) {
    let ang = BOHR;
    let chkfile= &scf_data.mol.ctrl.chkfile;
    let path = Path::new(chkfile);
    let file = if path.exists() {
        hdf5::File::open_rw(chkfile).unwrap()
    } else {
        hdf5::File::create(chkfile).unwrap()
    };
    let is_geom = file.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("geom")});
    let geom = if is_geom {
        file.group("geom").unwrap()
    } else {
        file.create_group("geom").unwrap()
    };
    let mass_charge = get_mass_charge(&scf_data.mol.geom.elem);
    //let mut geometry: Vec<(f64,f64,f64,f64)> = vec![];
    let mut geometry: Vec<[f64;4]> = vec![];
    mass_charge.iter().zip(scf_data.mol.geom.position.iter_columns_full()).for_each(|(mass_charge, position)| {
        geometry.push([mass_charge.1,position[0]*ang,position[1]*ang,position[2]*ang]);
        //geometry.push((mass_charge.1,position[0]*ang,position[1]*ang,position[2]*ang));
    });

    let is_geom = geom.member_names().unwrap().iter().fold(false,|is_exist,x| {is_exist || x.eq("position")});
    if is_geom {
        let dataset = geom.dataset("position").unwrap();
        dataset.write(&geometry);
    } else {
        let builder = geom.new_dataset_builder();
        builder.with_data(&geometry).create("position");
    }
    file.close();
}

pub fn has_dm(file: &hdf5::File) -> bool {
    let mut has_dm = false;
    if let Ok(_init_guess) = file.dataset("init_guess") {
        has_dm = true;
    }

    has_dm
}

pub fn has_mo_coeff(file: &hdf5::File) -> bool {
    let mut has_mo = false;
    if let Ok(_mo_coeff) = file.dataset("scf/mo_coeff") {
        has_mo = true;
    }

    has_mo
}

pub fn load_cint_data(chkfile: &String) -> (Option<(Vec<Vec<i32>>, Vec<Vec<i32>>, Vec<f64>)>, Option<Vec<Vec<i32>>>, Option<Vec<Basis4Elem>>, Option<CintType>) {
    let file = hdf5::File::open(chkfile).unwrap();
    let mol_info = file.dataset("mol").unwrap().read_scalar::<VarLenUnicode>().unwrap();
    let json_string = mol_info.as_str();
    let mol: HashMap<String, serde_json::Value> = serde_json::from_str(json_string).unwrap();
    let atm = serde_json::from_value(mol.get("_atm").unwrap().clone()).unwrap();
    let bas = serde_json::from_value(mol.get("_bas").unwrap().clone()).unwrap();
    let env = serde_json::from_value(mol.get("_env").unwrap().clone()).unwrap();
    let ecpbas = if let Some(ecpbas_value) = mol.get("_ecpbas") {
        // Some(serde_json::from_value(ecpbas_value.clone()).unwrap())
        if let Some(ecpbas_vec) = serde_json::from_value(ecpbas_value.clone()).ok() {
            Some(ecpbas_vec)
        } else {
            None
        }
    } else {
        None
    };
    // let bs = file.dataset("molecule/basis4elem").ok();
    let mut basis4elem: Option<Vec<Basis4Elem>> = None;
    if let Ok(ds) = file.dataset("molecule/basis4elem") {
        if let Some(basis4elem_value) = ds.read_scalar::<VarLenUnicode>().ok() {
            basis4elem = Some(serde_json::from_str(basis4elem_value.as_str()).unwrap());
        }
    }

    let mut cinttype: Option<CintType> = None;
    if let Ok(ds) = file.dataset("molecule/cinttype") {
        if let Some(cinttype_value) = ds.read_scalar::<VarLenUnicode>().ok() {
            cinttype = Some(cinttype_value.as_str().into());
        }
    }

    (Some((atm, bas, env)), ecpbas, basis4elem, cinttype)
}

pub fn load_geom(chkfile: &String) -> Option<GeomCell> {
    let file = hdf5::File::open(chkfile).unwrap();
    let ds = file.dataset("molecule/geom").ok()?;
    let json_str = ds.read_scalar::<VarLenUnicode>().ok()?;
    let json: HashMap<String, serde_json::Value> = serde_json::from_str(json_str.as_str()).ok()?;

    let name = json.get("name").and_then(|v| v.as_str()).unwrap_or("a molecule").to_string();
    let elem: Vec<String> = serde_json::from_value(json.get("elem")?.clone()).ok()?;
    let unit: GeomUnit = serde_json::from_value(json.get("unit")?.clone()).ok()?;
    let natoms = elem.len();
    let position = if natoms > 0 {
        let position_data: Vec<f64> = serde_json::from_value(json.get("position")?.clone()).ok()?;
        MatrixFull::from_vec([3, natoms], position_data).unwrap()
    } else {
        MatrixFull::empty()
    };

    Some(GeomCell {
        name,
        elem,
        fix: vec![],
        unit,
        position,
        nfree: 0,
        lattice: MatrixFull::empty(),
        pbc: MOrC::Molecule,
        ghost_bs_elem: vec![],
        ghost_bs_pos: MatrixFull::empty(),
        ghost_pc_chrg: vec![],
        ghost_pc_pos: MatrixFull::empty(),
        ghost_ep_path: vec![],
        ghost_ep_pos: MatrixFull::empty(),
        rest: vec![],
        ext_field: ExtField::empty(),
        rg_position: MatrixFull::empty(),
        rg_elem: vec![],
        rrs_pbc: false,
        unit_cell_index: vec![],
        pbc_dim: 1,
        rrs_pbc_vec: MatrixFull::empty(),
        max_step: vec![],
        k_points: vec![],
    })
}

pub fn load_basic(chkfile: &String) -> Option<(usize, usize, usize, Option<f64>, Option<f64>)> {
    let file = hdf5::File::open(chkfile).unwrap();
    let scf = file.group("scf").unwrap();
    let mut num_basis = None;
    let mut num_states = None;
    let mut spin_channel = None;
    let mut spin = None;
    let mut charge = None;
    if let Ok(nb) = scf.dataset("num_basis") {
        num_basis = Some(nb.read_raw::<usize>().unwrap()[0]);
    }
    if let Ok(nmo) = scf.dataset("num_states") {
        num_states = Some(nmo.read_raw::<usize>().unwrap()[0]);
    }
    if let Ok(s) = scf.dataset("spin_channel") {
        spin_channel = Some(s.read_raw::<usize>().unwrap()[0]);
    }
    if num_basis.is_none() || num_states.is_none() || spin_channel.is_none() {
        let mo_coeff_data = scf.dataset("mo_coeff").unwrap();
        match mo_coeff_data.ndim() {
            1 => {},
            2 => {
                num_basis = Some(mo_coeff_data.shape()[0]);
                num_states = Some(mo_coeff_data.shape()[1]);
                spin_channel = Some(1);
            },
            3 => {
                spin_channel = Some(mo_coeff_data.shape()[0]);
                num_basis = Some(mo_coeff_data.shape()[1]);
                num_states = Some(mo_coeff_data.shape()[2]);
            },
            _ => {
                panic!("Unsupported shape of mo_coeff dataset: {:?}", mo_coeff_data.shape());
            }
        }
    }

    // let spin = scf.dataset("spin").ok()
    //     .and_then(|ds| ds.read_raw::<f64>().ok())
    //     .map(|v| v[0]);
    // let charge = scf.dataset("charge").ok()
    //     .and_then(|ds| ds.read_raw::<f64>().ok())
    //     .map(|v| v[0]);
    if let Ok(s) = scf.dataset("spin") {
        spin = Some(s.read_raw::<f64>().unwrap()[0]);
    }
    if let Ok(c) = scf.dataset("charge") {
        charge = Some(c.read_raw::<f64>().unwrap()[0]);
    }

    if num_basis.is_some() && num_states.is_some() && spin_channel.is_some() {
        Some((num_basis.unwrap(), num_states.unwrap(), spin_channel.unwrap(), spin, charge))
    } else {
        None
    }
}
