/* 
xc functional interface to Libxc for REST 
*/

use core::panic;
use rayon::prelude::*;
use rstsr::prelude::*;
use crate::dft::libxc::{XcFuncType, LibXCFamily, eval_libxc_func_new};
use crate::dft::xc_deriv::{XCType, xc_indices_transform, transform_xc_inner, count_combinations};


// fn comp(n:usize, m:usize) -> usize {
//     //Compute composition C(n, m) = n!/(m!(n-m)!)
//     if m > n {
//         return 0;
//     }
//     if m == 0 || m == n {
//         return 1;
//     }
//     let mut res: usize = 1;
//     for i in 0..m {
//         res *= n - i;
//         res /= i + 1;
//     }
//     res
// }

/* 
_XC_NVAR = {
    ('HF', 0): (1, 1),
    ('HF', 1): (1, 2),
    ('LDA', 0): (1, 1),
    ('LDA', 1): (1, 2),
    ('GGA', 0): (4, 2),
    ('GGA', 1): (4, 5),
    ('MGGA', 0): (5, 3),
    ('MGGA', 1): (5, 7),
}
*/
fn get_nvar(xc_type: XCType, spin: usize, deriv: usize) -> (usize, usize) {
    // Get the number of density variables for the given xc_family and spin
    let n_rho_var:usize = match xc_type {
        XCType::HF => if spin == 0 { 1 } else { 2 },
        XCType::LDA => if spin == 0 { 1 } else { 2 },
        XCType::GGA => if spin == 0 { 2 } else { 5 },
        XCType::MGGA => if spin == 0 { 3 } else { 7 },
    };
    // compute total number of derivative components 
    let n_deriv = count_combinations(n_rho_var + deriv, deriv);
    (n_rho_var, n_deriv)
}

static XC_NVAR1_OFFSETS:[usize; 6] = [0, 1, 2,  3,   4,   5];   
static XC_NVAR2_OFFSETS:[usize; 6] = [0, 1, 3,  6,  10,  15];
static XC_NVAR3_OFFSETS:[usize; 6] = [0, 1, 4, 10,  20,  35];
static XC_NVAR5_OFFSETS:[usize; 6] = [0, 1, 6, 21,  56, 126];
static XC_NVAR7_OFFSETS:[usize; 6] = [0, 1, 8, 36, 120, 330];

const VSEG1: [usize; 3] = [2, 3, 2];
const FSEG1: [usize; 6] = [3, 6, 6, 4, 6, 3];
const KSEG1: [usize; 10] = [4, 9, 12, 10, 6, 12, 6, 12, 9, 4];
const LSEG1: [usize; 15] = [5, 12, 18, 20, 15, 8, 18, 9, 24, 18, 8, 20, 18, 12, 5];
const SEG1: [Option<&[usize]>; 5] = [
    None,
    Some(&VSEG1),
    Some(&FSEG1),
    Some(&KSEG1),
    Some(&LSEG1),
];

fn axpy(dst: &mut [f64], src: &[f64], fac: f64, np: usize, nsrc: usize) {
    for j in 0..nsrc {
        for i in 0..np {
            let dst_idx = j * np + i;
            let src_idx = i * nsrc + j;
            dst[dst_idx] += fac * src[src_idx];
        }
    }
}

// fn axpy_rayon(dst: &mut [f64], src: &[f64], fac: f64, np: usize, nsrc: usize) {
//     (0..nsrc).into_par_iter().for_each(|j| {
//         for i in 0..np {
//             let dst_idx = j * np + i;
//             let src_idx = i * nsrc + j;
//             dst[dst_idx] += fac * src[src_idx];
//         }
//     });
// }


fn merge_xc(output: &mut Vec<f64>, outbuf: &Vec<f64>, factor: f64, xc_type:XCType, spin:usize, deriv:usize, nvar:usize, np:usize) {
    // exc
    for i in 0..np {
        output[i] += factor * outbuf[i];
    }

    // println!("[merge_xc] spin: {}, xc_type: {:?}, deriv: {}, nvar: {}", spin, xc_type, deriv, nvar);

    let offsets0 = match xc_type {
        XCType::HF => XC_NVAR1_OFFSETS,
        XCType::LDA => XC_NVAR1_OFFSETS,
        XCType::GGA => XC_NVAR2_OFFSETS,
        XCType::MGGA => XC_NVAR3_OFFSETS,
    };

    if spin == 0 {
        let offsets1 = match nvar {
            1 => XC_NVAR1_OFFSETS, // LDA
            4 => XC_NVAR2_OFFSETS, // GGA
            5 => XC_NVAR3_OFFSETS, // MGGA
            _ => XC_NVAR3_OFFSETS, // default to MGGA
        };
        for order in 1..=deriv {
            let pin_start = offsets0[order] * np;
            let pin_end = offsets0[order + 1] * np;
            let pin = &outbuf[pin_start..pin_end];
            let pout_start = offsets1[order] * np;
            let pout_end = pout_start + (offsets0[order + 1] - offsets0[order]) * np;
            let pout = &mut output[pout_start..pout_end];
            // let nsrc = offsets0[order + 1] - offsets0[order];
            pout.par_iter_mut().zip(pin.par_iter())
            .for_each(
                |(to, &from)| 
                {
                    *to += factor * from;
                }
            );
        }
        return;
    }

    // spin unpolarized case
    let offsets1 = match nvar {
        1 => XC_NVAR2_OFFSETS, // LDA
        4 => XC_NVAR5_OFFSETS, // GGA
        5 => XC_NVAR7_OFFSETS, // MGGA
        _ => XC_NVAR7_OFFSETS, // default to MGGA
    };

    let mut pin_offset = np;
    for order in 1..=deriv {
        let terms = offsets0[order + 1] - offsets0[order];
        let pseg1 = SEG1[order].expect("Invalid segment for order");

        let mut pout_start = offsets1[order] * np;

        for i in 0..terms {
            let nsrc = pseg1[i];
            let pout = &mut output[pout_start..pout_start + nsrc * np];
            let pin = &outbuf[pin_offset..pin_offset + nsrc * np];

            axpy(pout, pin, factor, np, nsrc);

            pin_offset += nsrc * np;
            pout_start += nsrc * np;
        }
    }

}


fn eval_xc1(
    func_ids:&Vec<usize>, 
    func_factors:&Vec<f64>,
    xc_type: XCType, 
    spin: usize,
    rho: &[f64],
    sigma: Option<&[f64]>,
    lapl: Option<&[f64]>,
    tau: Option<&[f64]>,
    np: usize,
    deriv: usize,
) -> Vec<f64> {

    let (nvar, n_components) = get_nvar(xc_type, spin, deriv);
    // Compute output nvar (for merge_xc layout) from xc_type
    let out_nvar: usize = match xc_type {
        XCType::LDA | XCType::HF => 1,
        XCType::GGA => 4,
        XCType::MGGA => 5,
    };
    let mut output = vec![0.0; np * n_components];

    func_ids.iter()
    .zip(func_factors.iter())
    .for_each(
        |(func_id, xc_param)|
        {
            let xc_func = XcFuncType::xc_func_init(*func_id, spin+1);
            let cur_xc_type = match xc_func.get_libxc_family() {
                LibXCFamily::LDA => XCType::LDA,
                LibXCFamily::GGA => XCType::GGA,
                LibXCFamily::MGGA => XCType::MGGA,
                LibXCFamily::HybridGGA => XCType::GGA, // Hybrid GGA is treated as GGA
                LibXCFamily::HybridMGGA => XCType::MGGA, // Hybrid MGGA is treated as MGGA
                _ => panic!("Unresolved xc family: {:?}", xc_func.get_libxc_family()),
            };
            let mut outbuf = vec![0.0; np * n_components];
            eval_libxc_func_new(&xc_func, spin, deriv, np, rho, sigma, lapl, tau,  &mut outbuf);
            merge_xc(&mut output, &outbuf, *xc_param, cur_xc_type, spin, deriv, out_nvar, np);
        }
    );

    output
}


fn generate_rho_vec(rho_u:&[f64], spin:usize, nvar: usize, np: usize) -> Vec<f64> {
    let rho_len = match nvar {
        1 => if spin == 0 { np } else { np * 2 },
        4 => if spin == 0 { np * 2 } else { np * 5 },
        5 => if spin == 0 { np * 3 } else { np * 7 },
        _ => unreachable!(),
    };
    let mut rho_vec = vec![0.0; rho_len];

    let rho_d = if spin == 1 {
        &rho_u[np * nvar..]
    } else {
        &[]
    };

    match nvar {
        1 => { // lda
            if spin == 1 {
                for i in 0..np {
                    rho_vec[i * 2] = rho_u[i];
                    rho_vec[i * 2 + 1] = rho_d[i];
                }
            }
            else {
                rho_vec[..np].copy_from_slice(&rho_u[..np]);
            }
        },
        4 => { // gga
            let sigma = if spin == 1 {
                let (rho_part, sigma_part) = rho_vec.split_at_mut(np * 2);
                for i in 0..np {
                    rho_part[i * 2] = rho_u[i];
                    rho_part[i * 2 + 1] = rho_d[i];
                }
                sigma_part
            } else {
                let (rho_part, sigma_part) = rho_vec.split_at_mut(np);
                rho_part.copy_from_slice(&rho_u[..np]);
                sigma_part
            };
            let gxu = &rho_u[np..np*2];
            let gyu = &rho_u[np*2..np*3];
            let gzu = &rho_u[np*3..np*4];
            if spin == 1 {
                let gxd = &rho_d[np..np*2];
                let gyd = &rho_d[np*2..np*3];
                let gzd = &rho_d[np*3..np*4];
                for i in 0..np {
                    sigma[i * 3 + 0] = gxu[i] * gxu[i] + gyu[i] * gyu[i] + gzu[i] * gzu[i];
                    sigma[i * 3 + 1] = gxu[i] * gxd[i] + gyu[i] * gyd[i] + gzu[i] * gzd[i];
                    sigma[i * 3 + 2] = gxd[i] * gxd[i] + gyd[i] * gyd[i] + gzd[i] * gzd[i];
                }
            } else {
                for i in 0..np {
                    sigma[i] = gxu[i].powi(2) + gyu[i].powi(2) + gzu[i].powi(2);
                }
            }
        },
        5 => { // mgga
            let sigma = if spin == 1 {
                let (rho_part, rest) = rho_vec.split_at_mut(np * 2);
                let (sigma_part, tau_part) = rest.split_at_mut(np * 3);
                for i in 0..np {
                    rho_part[i * 2] = rho_u[i];
                    rho_part[i * 2 + 1] = rho_d[i];
                    tau_part[i * 2] = rho_u[np * 4 + i];
                    tau_part[i * 2 + 1] = rho_d[np * 4 + i];
                }
                sigma_part
            } else {
                let (rho_part, rest) = rho_vec.split_at_mut(np);
                let (sigma_part, tau_part) = rest.split_at_mut(np);
                rho_part.copy_from_slice(&rho_u[..np]);
                tau_part.copy_from_slice(&rho_u[np * 4..np * 5]);
                sigma_part
            };
            let gxu = &rho_u[np..np*2];
            let gyu = &rho_u[np*2..np*3];
            let gzu = &rho_u[np*3..np*4];
            if spin == 1 {
                let gxd = &rho_d[np..np*2];
                let gyd = &rho_d[np*2..np*3];
                let gzd = &rho_d[np*3..np*4];
                for i in 0..np {
                    sigma[i * 3 + 0] = gxu[i] * gxu[i] + gyu[i] * gyu[i] + gzu[i] * gzu[i];
                    sigma[i * 3 + 1] = gxu[i] * gxd[i] + gyu[i] * gyd[i] + gzu[i] * gzd[i];
                    sigma[i * 3 + 2] = gxd[i] * gxd[i] + gyd[i] * gyd[i] + gzd[i] * gzd[i];
                }
            } else {
                for i in 0..np {
                    sigma[i] = gxu[i].powi(2) + gyu[i].powi(2) + gzu[i].powi(2);
                }
            }
        },
        _ => unreachable!(),
    }

    rho_vec
}


pub fn eval_xc_eff(func_ids:&Vec<usize>, func_factors:&Vec<f64>, xc_type:XCType, spin:usize, rho_array:&[f64], np:usize, deriv:usize) -> Vec<Option<Tensor<f64, DeviceBLAS>>> {

    // let (xc_type, nvar) = match xc_family {
    //     "LDA" => (XCType::LDA, 1 as usize),
    //     "GGA" => (XCType::GGA, 4 as usize),
    //     "Meta-GGA" => (XCType::MGGA, 5 as usize),
    //     "Hybrid-GGA" => (XCType::GGA, 4 as usize),
    //     "Hybrid-meta-GGA" => (XCType::MGGA, 5 as usize),
    //     _ => (XCType::GGA, 4 as usize), // default to GGA
    // };

    let nvar = match xc_type {
        XCType::HF => 1,
        XCType::LDA => 1,
        XCType::GGA => 4,
        XCType::MGGA => 5,
    };
    let rho_vec = generate_rho_vec(&rho_array, spin, nvar, np);
    let rho = &rho_vec[..np * (spin + 1)];
    let sigma = match xc_type { 
        XCType::GGA | XCType::MGGA => if spin == 1 {
            Some(&rho_vec[np*2..np*5])
        } else {
            Some(&rho_vec[np..np*2])
        },
        _ => None, 
    };
    let lapl: Option<&[f64]> = None; // Currently not supported
    let tau = match xc_type {
        XCType::MGGA => if spin == 1 {
            Some(&rho_vec[np*5..np*7])
        } else {
            Some(&rho_vec[np*2..np*3])
        },
        _ => None,
    };

    let libxc_out = eval_xc1(func_ids, func_factors, xc_type, spin, rho, sigma, lapl, tau, np, deriv);

    let (_, xc_n_components) = get_nvar(xc_type, spin, deriv);
    let device = DeviceBLAS::default();
    let rho_ten_shape: Vec<usize> = match spin {
        0 => vec![np, nvar], // unpolarized case
        1 => vec![np, nvar, 2], // polarized case
        _ => panic!("Invalid spin value: {}", spin),
    };
    let rho_ten = rt::asarray((rho_array, rho_ten_shape, &device));
    let xc_val_libxc = rt::asarray((&libxc_out, [np, xc_n_components], &device));
    // println!("Debug: in eval_xc_eff libxc_out: {:?}", &xc_val_libxc);
    let xc_val_transform = xc_indices_transform(xc_val_libxc, xc_type, spin, deriv);

    // exc, vxc, fxc, kxc
    let mut xc_tensor_vec = vec![Some(xc_val_transform.i((Ellipsis, 0)).to_owned())];

    for order in 1..=deriv {
        let cur_xc_tensor = transform_xc_inner(rho_ten.view(), xc_val_transform.view(), xc_type, spin, order);
        xc_tensor_vec.push(Some(cur_xc_tensor));
    }

    if deriv < 3 {
    // Always return [e, v, f, k] tensors
        for _ in deriv + 1..=3 {
            xc_tensor_vec.push(None);
        }
    }

    xc_tensor_vec
} 
