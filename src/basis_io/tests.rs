//! Tests for basis_io: basis-file (BSE JSON) parsing, GTO normalization and derivatives.
use super::*;
use std::env;
use std::path;
use std::fs;
use std::f64::consts::PI;
use crate::utilities::basic_math::double_factorial;
use rayon::prelude::*;

#[test]
fn import_ecp()-> anyhow::Result<()> {
    let rest_home = env::var("REST_HOME").expect("The environment variable REST_HOME is not set");
    let mut file_path = path::PathBuf::from(rest_home).join("rest/basis-set-pool/def2-svp/Au.json");
    let tmp_string = fs::read_to_string(file_path)?;
    let tmp_basis:Basis4ElemRaw = serde_json::from_str(&tmp_string[..])?;
    if let (Some(ecp_electrons), Some(ecp_potentials))= (&tmp_basis.ecp_electrons, &tmp_basis.ecp_potentials)  {
        println!("debug ecp electrons: {}", &ecp_electrons);
        ecp_potentials.iter().for_each(|ecp| {
            println!("debug ecp: {:?}", ecp.parse());
        });
    };
    Ok(())
}

#[test]
fn test_double_factorial() {
    for x in 0..=19 {
        println!("{},{}",x,double_factorial(x));
    }
    for x in -1..=19 {
        println!("{},{}",x,double_factorial(x));
    }
}
#[test]
fn debug_cartesian_gto() {
    let a = 3.0;
    let l = 2usize;
    let c = [0.0, 0.0,0.0];
    let r = [1.0, 2.0,3.0];
    //cartesian_gto(a, l, c, r);
}
#[test]
fn debug_par() {
    let vec1:Vec<i32> = (0..1000).map(|i| i).collect();
    let vec2:Vec<i32> = (1000..2000).map(|i| i).collect();
    let vec3:Vec<i32> = (2000..3000).map(|i| i).collect();
    let vec4:Vec<i32> = (3000..4000).map(|i| i).collect();
    vec1.par_iter().zip(vec2.par_iter()).map(|(a,b)| (a,b))
    .zip(vec3.par_iter().zip(vec4.par_iter()).map(|(c,d)| (c,d)))
    .for_each(|(a,b)| {println!("{:4}, {:4}, {:4}, {:4}, {:4}, {:4}",a.0,a.1,b.0,b.1, b.0-a.0,b.1-a.1)});
}


#[test]
fn norm_factor_gto_radial() {
    let l = 2_i32;
    let a = 2.0_f64;
    let value1 = cint_norm_factor(l,a);
    let value2 = _gaussian_int(l*2+2,a*2.0).sqrt();
    let norm0 = ((2.0*a/PI).powf(0.75)*(4.0*a).powf((l as f64)/2.0));
    println!("{},{},{},{}",value1,value2,norm0, value1*norm0);
}


#[test]
fn test_cart_gto_deriv() {
    let a = 2.0;
    let r = [[1.0, 2.0, 3.0]];
    let gto_center = [0.0, 0.0, 0.0];
    let cur_l = 2;
    // let bas = Basis4Elem::new();
    // let ao_deriv = 2;
    let ground_truth_0 = cartesian_gto_batch_v04(a, cur_l, &gto_center, &r);
    let ground_truth_1 = cartesian_gto_1st_batch_v03(a, cur_l, &gto_center, &r);
    let test_res = cartesian_gto_1st_batch_v04(a, cur_l, &gto_center, &r);
    println!("ao0: {:?}", ground_truth_0);
    println!("ao1: {:?}", ground_truth_1);
    println!("test_res: {:?}", test_res);
}
