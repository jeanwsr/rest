use std::{env, path};

const DEFAULT_BASIS_PATH: &str = "/opt/rest_workspace/rest/basis-set-pool/";

macro_rules! debug {
    ($print_level:expr, $($arg:tt)*) => {
        if $print_level >= 2 {
            println!($($arg)*);
        }
    };
}

pub fn get_rest_basis_dir(print_level:usize) -> String {
    let mut rest_basis_dir: String = String::new();
    if let Some(val) = env::var("REST_BASIS_DIR").ok() {
        if !val.is_empty() {
            if path::Path::new(&val).is_dir() {
                // println!("Detected the REST_BASIS_DIR environment variable: {}", val);
                println!("Set rest_basis_dir from REST_BASIS_DIR environment variable {}.", val);
                rest_basis_dir = val + "/";
                return rest_basis_dir;
            } else {
                debug!(print_level, "The specified REST_BASIS_DIR path {} is not a valid directory.", val);
            }
        } else {
            debug!(print_level, "The REST_BASIS_DIR environment variable is empty.");
        }
    } else {
        debug!(print_level, "The REST_BASIS_DIR environment variable is not set.");
    }
    let default_basis_path = String::from(DEFAULT_BASIS_PATH);
    let mut found_basis_path = true;
    if path::Path::new(&default_basis_path).is_dir() {
        println!("Use the default path ({}) as rest_basis_dir.", &default_basis_path);
        rest_basis_dir = default_basis_path.clone();
    } else if let Some(rest_home) = env::var("REST_HOME").ok() {
        let alt_basis_path = rest_home + "/rest/basis-set-pool/";
        if path::Path::new(&alt_basis_path).is_dir() {
            println!("Use the alternative default path ({}) as rest_basis_dir.", &alt_basis_path);
            rest_basis_dir = alt_basis_path.clone();
        } else {
            found_basis_path = false;
        }
    } else {
        found_basis_path = false;
    }
    if !found_basis_path {
        println!("Cannot find the default basis set path from default path ({}) or $REST_HOME/rest/basis-set-pool/.", &default_basis_path);
    }
    rest_basis_dir
}

pub fn get_valid_basis_path(tmp_bas: &String, rest_basis_dir: &String, bastype: &str) -> String {
    if path::Path::new(tmp_bas).is_dir() {
        println!("The specified path for the basis sets: {}", tmp_bas);
        tmp_bas.clone()
    } else if !rest_basis_dir.is_empty() && path::Path::new(&(rest_basis_dir.clone()+tmp_bas)).is_dir() {
        println!("The specified path for the {} sets is: {}, try to find it from the rest_basis_dir", bastype, tmp_bas);
        let tmp_bas_full = rest_basis_dir.clone()+&tmp_bas;
        println!("Found: {}", tmp_bas_full);
        tmp_bas_full
    } else {
        println!("Cannot find the specified folder for the {} sets: ({})", bastype, tmp_bas);
        println!("REST will try to fetch them from the basis-set-exchange pool (https://www.basissetexchange.org/)");
        tmp_bas.clone()
    }
}