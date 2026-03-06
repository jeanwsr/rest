use crate::ctrl_io::quasiparticle_methods::QuasiParticle;
use crate::ri_gw;
use crate::constants::PI;

pub fn fourier_series(
    sin_coeff: &Vec<f64>,
    cos_coeff: &Vec<f64>,
    powers: usize,
    t:f64,
    x: f64
) -> f64 {


    let y=(1..=powers).fold(0.0, |acc, k| {
        let kf = k as f64;
        acc + sin_coeff[k-1]*(PI*x*kf/t).sin()
            + cos_coeff[k-1]*(PI*x*kf/t).cos()
    });
    //println!("Fourier Function:x={},y={}",x,y);
    y
}
pub fn define_fourier_series(qp_ctrl:&QuasiParticle)->(usize,f64,Vec<f64>,Vec<f64>){
    let sin_coeff_path=qp_ctrl.fse_sin_coeff_path.clone();
    let cos_coeff_path=qp_ctrl.fse_cos_coeff_path.clone();
    let sin_coeff=ri_gw::read_floats(&sin_coeff_path).expect("Failure when reading from sin coefficients file!");
    let cos_coeff=ri_gw::read_floats(&cos_coeff_path).expect("Failure when reading from sin coefficients file!");
    let mut powers:usize=0;
    let t=qp_ctrl.gw_span_energy;
    if sin_coeff.len()==cos_coeff.len(){
        powers=sin_coeff.len();
    }else{
        panic!("Length of sin_coeff NOT EQUAL TO cos_coeff! Exited, make sure they are the same");
    }
    (powers,t,sin_coeff,cos_coeff)
}
pub fn sigma_fourier(origin:f64,omega:f64,powers:usize,t:f64,sin_coeff: &Vec<f64>,cos_coeff: &Vec<f64>)->f64{
    fourier_series(sin_coeff,cos_coeff,powers,t,omega-origin)
}