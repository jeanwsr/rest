use super::prelude::*;
use super::xc_deriv::{libxc_transform_xcfun_indices, transform_xc_inner};
use libxc::compute_cpu::LibXCCpuInput;
use libxc::prelude::*;

use LibXCSpin::*;
use XCDenType::*;

/// Determine the density type required by the given XC functional.
pub fn determine_den_type(xc_func: &LibXCFunctional) -> XCDenType {
    match xc_func.family() {
        LibXCFamily::LDA | LibXCFamily::HybLDA => RHO,
        LibXCFamily::GGA | LibXCFamily::HybGGA => SIGMA,
        LibXCFamily::MGGA | LibXCFamily::HybMGGA => {
            if xc_func.needs_laplacian() {
                LAPL
            } else {
                TAU
            }
        },
        _ => panic!("Unsupported functional family: {:?}", xc_func.family()),
    }
}

pub fn determine_den_type_from_list(xc_func_list: &[&LibXCFunctional]) -> XCDenType {
    xc_func_list.iter().map(|f| determine_den_type(f)).max_by_key(|&den_type| den_type.num_nvar()).unwrap()
}

/// Evaluate the XC energy/potential, to LibXC raw output.
pub fn libxc_eval_inner(xc_func: &LibXCFunctional, rho: TsrView, deriv: usize) -> (Vec<f64>, LibXCOutputLayout) {
    // sanity check
    // rho must be either [ngrids, 1/4/5/6] or [ngrids, 1/4/5/6, 2]
    match xc_func.spin() {
        Unpolarized => ni_check_shape!(rho.ndim(), 2, "rho for unpolarized functionals must be a 2-dim tensor"),
        Polarized => {
            ni_check_shape!(rho.ndim(), 3, "rho for polarized functionals must be a 3-dim tensor");
            ni_check_shape!(rho.shape()[2], 2, "rho for polarized functionals must have last dimension of size 2");
        },
    }
    // we do not support laplacian currently
    if xc_func.needs_laplacian() {
        panic!("Laplacian-dependent functionals are not supported yet");
    }
    let den_type = determine_den_type(xc_func);
    ni_check_shape!(rho.shape()[1] >= den_type.num_nvar(), "Input density does not have enough components");
    let do_rho = matches!(den_type, RHO | SIGMA | TAU | LAPL);
    let do_sigma = matches!(den_type, SIGMA | TAU | LAPL);
    let do_tau = matches!(den_type, TAU | LAPL);

    if xc_func.spin() == Unpolarized {
        // build up owned components
        let xc_rho = do_rho.then(|| rho.i((.., 0)).to_vec());
        let xc_sigma = do_sigma.then(|| rt::vecdot(rho.i((.., 1..4)), rho.i((.., 1..4)), 1).into_vec());
        let xc_tau = do_tau.then(|| rho.i((.., 4)).to_vec());
        // construct xc_input
        let mut xc_input = LibXCCpuInput::new();
        for (key, value) in [("rho", xc_rho.as_ref()), ("sigma", xc_sigma.as_ref()), ("tau", xc_tau.as_ref())] {
            value.map(|v| xc_input.insert(key.to_string(), v.as_slice()));
        }
        xc_func.compute_xc(&xc_input, deriv).unwrap_or_else(|e| panic!("LibXC compute_xc failed: {e}"))
    } else {
        // build up owned components
        // note libxc's convention is [2, ngrids] for rho, not [ngrids, 2].
        // we need to transpose the input rho before feeding into libxc.
        let xc_rho = do_rho.then(|| rho.i((.., 0, ..)).t().into_shape(-1).to_vec());
        let xc_tau = do_tau.then(|| rho.i((.., 4, ..)).t().into_shape(-1).to_vec());
        // sigma is more complicated, as it requires (uu ud dd) components.
        let xc_sigma = do_sigma.then(|| {
            let ngrids = rho.shape()[0];
            let mut sigma = rt::zeros(([ngrids, 3], &rho.device().clone()));
            sigma.i_mut((.., 0)).vecdot_from(rho.i((.., 1..4, 0)), rho.i((.., 1..4, 0)), 1);
            sigma.i_mut((.., 1)).vecdot_from(rho.i((.., 1..4, 0)), rho.i((.., 1..4, 1)), 1);
            sigma.i_mut((.., 2)).vecdot_from(rho.i((.., 1..4, 1)), rho.i((.., 1..4, 1)), 1);
            sigma.t().into_shape(-1).to_vec()
        });
        let mut xc_input = LibXCCpuInput::new();
        for (key, value) in [("rho", xc_rho.as_ref()), ("sigma", xc_sigma.as_ref()), ("tau", xc_tau.as_ref())] {
            value.map(|v| xc_input.insert(key.to_string(), v.as_slice()));
        }
        xc_func.compute_xc(&xc_input, deriv).unwrap_or_else(|e| panic!("LibXC compute_xc failed: {e}"))
    }
}

/// Transpose matrix with buffer.
///
/// This will perform inplace, but algorithm is naive. Should be able to optimize but that's too
/// hard for me.
fn transpose_with_buffer(slc: &mut [f64], m: usize, n: usize, buf: &mut [f64]) {
    // currently it is a naive implementation
    assert!(slc.len() >= n * m);
    assert!(buf.len() >= n * m);
    for j in 0..m {
        for i in 0..n {
            buf[j * n + i] = slc[i * m + j];
        }
    }
    slc[..n * m].copy_from_slice(&buf[..n * m]);
}

/// Evaluate effective XC potential from LibXC functional and density, in serial.
pub fn libxc_eval_eff_serial(xc_func: &LibXCFunctional, rho: TsrView, deriv: usize) -> Vec<Tsr> {
    let den_type = determine_den_type(xc_func);
    let device = rho.device().clone();
    let (mut xc_val, xc_layout) = libxc_eval_inner(xc_func, rho.view(), deriv);
    // transpose the spin-related components
    // first find the largest intermediate size
    let ngrids = rho.shape()[0];
    let buf_size = xc_layout.iter_to_range().map(|(_, r)| r.end - r.start).max().unwrap_or(ngrids);
    let mut buf = vec![0.0; buf_size];
    if xc_func.spin() == Polarized {
        for (_, r) in xc_layout.iter_to_range() {
            if r.end - r.start != ngrids {
                let ncomp = (r.end - r.start) / ngrids;
                transpose_with_buffer(&mut xc_val[r], ncomp, ngrids, &mut buf);
            }
        }
    }
    let ngrids = rho.shape()[0];
    let xlen = xc_val.len() / ngrids;
    let xc_val = rt::asarray((xc_val, [ngrids, xlen].f(), &device));
    let xc_val = libxc_transform_xcfun_indices(xc_val.view(), den_type, xc_func.spin(), deriv);
    (0..=deriv).map(|order| transform_xc_inner(rho.view(), xc_val.view(), den_type, xc_func.spin(), order)).collect()
}

/// Evaluate effective XC potential from LibXC functional and density, in parallel.
pub fn libxc_eval_eff_parallel(
    xc_func: &LibXCFunctional,
    rho: TsrView,
    deriv: usize,
    par_chunk_size: Option<usize>,
) -> Vec<Tsr> {
    // if in threadpool (thread-index is Some), we use 1 thread to avoid nested parallelism.
    let nthreads = rayon::current_thread_index().map_or(rayon::current_num_threads(), |_| 1);
    if nthreads == 1 {
        return libxc_eval_eff_serial(xc_func, rho, deriv);
    }

    // determine chunk size
    // this setting is probably good? anyway, user can set by argument.
    let den_type = determine_den_type(xc_func);
    let spin = xc_func.spin();
    let par_chunk_size = par_chunk_size.unwrap_or(match (den_type, spin) {
        (RHO, Unpolarized) => 16384,
        (RHO, Polarized) => 6144,
        (SIGMA, _) => 384,
        (TAU | LAPL, _) => 256,
    });
    let ngrids = rho.shape()[0];
    let par_chunk_size = ngrids.div_ceil(nthreads).max(par_chunk_size);
    if par_chunk_size >= ngrids {
        // if chunk size is larger than total grids, just do serial computation.
        return libxc_eval_eff_serial(xc_func, rho, deriv);
    }

    // determine output shape at this stage
    let nvar = den_type.num_nvar();
    // generate shapes
    let out_shapes = (0..=deriv)
        .map(|order| {
            // unpolarized: [ngrids, [nvar] * deriv]
            // polarized: [ngrids, [nvar, 2] * deriv]
            let mut shape = vec![ngrids];
            for _ in 0..order {
                match spin {
                    Unpolarized => shape.push(nvar),
                    Polarized => shape.extend_from_slice(&[nvar, 2]),
                }
            }
            shape
        })
        .collect_vec();
    // generate tensors
    let xc_eff = out_shapes.iter().map(|shape| rt::zeros((shape.to_vec(), &rho.device().clone()))).collect_vec();

    // parallel computation
    (0..ngrids).into_par_iter().step_by(par_chunk_size).for_each(|start| {
        let stop = (start + par_chunk_size).min(ngrids);
        let rho_chunk = rho.i(start..stop);
        let xc_eff_chunk = libxc_eval_eff_serial(xc_func, rho_chunk, deriv);
        for (order, xc_eff_order) in xc_eff_chunk.into_iter().enumerate() {
            let xc_eff_orig = xc_eff[order].i(start..stop);
            let mut xc_eff_orig = unsafe { xc_eff_orig.force_mut() };
            xc_eff_orig.assign(&xc_eff_order);
        }
    });

    xc_eff
}

/// Evaluate effective XC potential from LibXC functional and density, with parallel option.
///
/// # Parameter
///
/// - `xc_func`: The LibXC functional to evaluate.
///
///   The user should make sure to initialize the LibXC functional with **proper
///   spin-polarization**, and possibly **proper omega/parameter settings**.
///
/// - `rho`: The density tensor. The shape must be
///
///   - Spin Unpolarized: `[ngrids, nvar]`
///   - Spin Polarized: `[ngrids, nvar, 2]`
///
///   Where `nvar` is the number of variables:
///
///   - RHO: 1 ($\rho$)
///   - SIGMA: 4 ($\rho$, $\rho_x$, $\rho_y$, $\rho_z$)
///   - TAU: 5 ($\rho$, $\rho_x$, $\rho_y$, $\rho_z$, $\tau$)
///
///   Note we slightly differ to the PySCF's convention. RHO's dimension `nvar=1` cannot be
///   squeezed out. This is to make the code more consistent and easier to implement.
///
/// - `deriv`: The maximum derivative order to evaluate. Note the smaller derivates will also be
///   evaluated and returned.
///
/// - `par`: How to parallelize the evaluation.
///
///   - true/false: Whether to parallelize or not. If true, it will use the default parallelization
///     strategy.
///   - usize: The chunk size to be parallelized. If None, it will use the default chunk size.
///   
///   The default chunksize depends on density type and spin:
///
///   - RHO, Unpolarized: 16384
///   - RHO, Polarized: 6144
///   - SIGMA: 384
///   - TAU: 256
///
/// # Input/Output Shapes
///
/// Please note the output will contain all derivatives up to `deriv` user specified. For example,
/// if `deriv = 1` for GGA (SIGMA) and unpolarized, the output will contain two tensors, first of
/// shape `[ngrids]`, second of shape `[ngrids, 3]`.
///
/// | deriv | Output `xc_eff`<br>Unpolarized | Output `xc_eff`<br>Polarized          |
/// |-------|--------------------------------|---------------------------------------|
/// | 0     | `[ngrids]`                     | `[ngrids]`                            |
/// | 1     | `[ngrids, nvar]`               | `[ngrids, nvar, 2]`                   |
/// | 2     | `[ngrids, nvar, nvar]`         | `[ngrids, nvar, 2, nvar, 2]`          |
/// | 3     | `[ngrids, nvar, nvar, nvar]`   | `[ngrids, nvar, 2, nvar, 2, nvar, 2]` |
pub fn libxc_eval_eff(xc_func: &LibXCFunctional, rho: TsrView, deriv: usize, par: impl Into<XCPar>) -> Vec<Tsr> {
    let par = par.into();
    match par {
        XCPar::Par { chunk_size } => libxc_eval_eff_parallel(xc_func, rho, deriv, chunk_size),
        XCPar::Serial => libxc_eval_eff_serial(xc_func, rho, deriv),
    }
}
