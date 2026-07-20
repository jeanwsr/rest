use std::{env, path};
use std::collections::HashMap;
use lazy_static::lazy_static;
use log::{warn,debug};

const DEFAULT_BASIS_PATH: &str = "/opt/rest_workspace/rest/basis-set-pool/";

// macro_rules! debug {
//     ($print_level:expr, $($arg:tt)*) => {
//         if $print_level >= 2 {
//             println!($($arg)*);
//         }
//     };
// }

pub fn get_rest_basis_dir(print_level:usize) -> Vec<String> {
    let mut rest_basis_dir:Vec<String> = vec![];
    if let Some(val) = env::var("REST_BASIS_DIR").ok() {
        if !val.is_empty() {
            if path::Path::new(&val).is_dir() {
                // println!("Detected the REST_BASIS_DIR environment variable: {}", val);
                debug!("Add the REST_BASIS_DIR path ({}) to rest_basis_dir.", &val);
                // rest_basis_dir = val + "/";
                rest_basis_dir.push(val+"/");
                // return rest_basis_dir;
            } else {
                debug!("The specified REST_BASIS_DIR path {} is not a valid directory.", val);
            }
        } else {
            debug!("The REST_BASIS_DIR environment variable is empty.");
        }
    } else {
        debug!("The REST_BASIS_DIR environment variable is not set.");
    }

    let current_exe =  env::current_exe().unwrap();
    let mut inside_conda = false;
    let mut conda_prefix = String::new();
    if let Some(p) = env::var("CONDA_PREFIX").ok() {
        conda_prefix = p.clone();
        inside_conda = current_exe.starts_with(&conda_prefix);
        debug!("Detected CONDA_PREFIX: {}, inside_conda: {}", conda_prefix, inside_conda);
    }
    if inside_conda {
        let conda_basis_path = conda_prefix + "/share/rest/basis-set-pool/";
        if path::Path::new(&conda_basis_path).is_dir() {
            println!("Add the conda environment path ({}) to rest_basis_dir.", &conda_basis_path);
            rest_basis_dir.push(conda_basis_path.clone());
            // return rest_basis_dir;
        } else {
            debug!("The conda basis set path {} is not a valid directory.", conda_basis_path);
        }
    } 

    let default_basis_path = String::from(DEFAULT_BASIS_PATH);
    if path::Path::new(&default_basis_path).is_dir() {
        debug!("Add the default path ({}) to rest_basis_dir.", &default_basis_path);
        rest_basis_dir.push(default_basis_path.clone());
    } else if let Some(rest_home) = env::var("REST_HOME").ok() {
        let alt_basis_path = rest_home + "/rest/basis-set-pool/";
        if path::Path::new(&alt_basis_path).is_dir() {
            debug!("Add the alternative default path ({}) to rest_basis_dir.", &alt_basis_path);
            rest_basis_dir.push(alt_basis_path.clone());
        } 
    } 
    if rest_basis_dir.is_empty() {
        warn!("Cannot find the default basis set path.");
        warn!("Please check if one of the following paths exist:");
        warn!("  1. The path set by the REST_BASIS_DIR environment variable");
        warn!("  2. The conda environment path: $CONDA_PREFIX/share/rest/basis-set-pool/ (if using conda)");
        warn!("  3. The default installation path: {}", DEFAULT_BASIS_PATH);
        warn!("  4. The alternative default path: $REST_HOME/rest/basis-set-pool/");
    }
    rest_basis_dir
}

lazy_static!{
    static ref BASIS_ALIAS: HashMap<&'static str, &'static str> = HashMap::from([
        ("def2-sv(p)-jkfit", "def2-universal-jkfit"),
        ("6-31g**-rifit", "6-31gss-rifit"),
        ("6-311g**-rifit", "6-311gss-rifit")
    ]);

    static ref BASIS_MISSING_WARN: HashMap<&'static str, &'static str> = HashMap::from([
        ("def2-svp-jkfit", "not supported, please use def2-universal-jkfit instead")
    ]);
} 

pub fn filter_by_alias(basis_name: &String) -> String {
    if BASIS_ALIAS.contains_key(basis_name.as_str()) {
        let alias_name = BASIS_ALIAS.get(basis_name.as_str()).unwrap();
        warn!("The basis set name {} is an alias to {}", basis_name, alias_name);
        alias_name.to_string()
    } else if BASIS_MISSING_WARN.contains_key(basis_name.as_str()) {
        let warn_msg = BASIS_MISSING_WARN.get(basis_name.as_str()).unwrap();
        panic!("The basis set {} is {}", basis_name, warn_msg);
    } else {
        basis_name.clone()
    }
}

pub fn get_valid_basis_path(tmp_bas: &String, rest_basis_dir: &Vec<String>, bastype: &str) -> String {
    if path::Path::new(tmp_bas).is_dir() {
        debug!("The specified path for the basis sets: {}", tmp_bas);
        tmp_bas.clone()
    } else if !rest_basis_dir.is_empty() {
        match get_valid_basis_path_from_dirlist(tmp_bas, rest_basis_dir, bastype) {
            Some(valid_path) => valid_path,
            None => {
                warn!("Cannot find the specified folder for the {} sets: ({})", bastype, tmp_bas);
                warn!("REST will try to fetch them from the basis-set-exchange pool (https://www.basissetexchange.org/)");
                tmp_bas.clone() 
            }
        }
    } else {
        warn!("Cannot find the specified folder for the {} sets: ({})", bastype, tmp_bas);
        warn!("REST will try to fetch them from the basis-set-exchange pool (https://www.basissetexchange.org/)");
        tmp_bas.clone()
    }
}

pub fn get_valid_basis_path_from_dirlist(tmp_bas: &String, rest_basis_dir: &Vec<String>, bastype: &str) -> Option<String> {
    let mut found = false;
    let tmp_bas_lower = filter_by_alias(&tmp_bas.to_lowercase());
    for dir in rest_basis_dir.iter() {
        let try_path = dir.clone() + &tmp_bas_lower;
        if path::Path::new(&try_path).is_dir() {
            debug!("The specified path for the {} sets is: {} (lowercase), try to find it from the rest_basis_dir", bastype, tmp_bas_lower);
            debug!("Found: {}", try_path);
            return Some(try_path);
        }
    }
    None
}
