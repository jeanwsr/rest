//! XC Potential generation (parallel enhanced)

use super::prelude::*;
use XCDenType::*;

/// Contract AO values with a weight vector to produce a symmetric matrix.
///
/// # Parameters
///
/// - `den_type`: the type of density to compute. Can be `RHO`, `SIGMA`, `TAU`.
/// - `wv` : weight vector, shape `[ngrids, nvar]`
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `out` : output buffer, shape `[nao, nao]`
/// - `buf` : scratch buffer of length at least `ngrids * nao`
fn contract_ao_wv_without_symmetrize(den_type: XCDenType, wv: TsrView, ao: TsrView, mut out: TsrMut, buf: &mut [f64]) {
    ni_check_shape!(wv.ndim(), 2, "Weight vector must be 2-dim");
    let nvar = wv.shape()[1];
    let ngrids = wv.shape()[0];
    ni_check_shape!(den_type.num_nvar(), nvar, "Dimension mismatch for input density type");
    ni_check_shape!(ao.ndim(), 3, "AO tensor must be 3-dim");
    let nao = ao.shape()[1];
    ni_check_shape!(ao.shape()[0], ngrids, "AO grids dimension mismatch");
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient");

    ni_check_shape!(out.shape(), [nao, nao], "Output shape mismatch");
    ni_check_shape!(buf.len() >= ngrids * nao, "Buffer length insufficient");

    if den_type == LAPL {
        panic!("Contracting AO with LAPL density type is not supported");
    }

    // clean notation of slice, just for readability

    /// Macro for slicing AO tensor with the last index.
    /// Returns [ngrids, nao] view for the specified component index.
    macro_rules! ao_ {
        [$idx:expr] => {
            ao.i((.., .., $idx))
        };
    }
    /// Macro for slicing weight vector with the last index.
    /// Returns [ngrids] view for the specified component index.
    macro_rules! wv_ {
        [$idx:expr] => {
            wv.i((.., $idx))
        };
    }

    let device = out.device().clone();
    let mut scr = rt::asarray((buf, [ngrids, nao], &device));

    // RHO contribution
    // out += 0.5 * ao_![0].t() % (wv_![0] * ao_![0]);
    rt::mul_with_output(ao_![0], wv_![0], scr.view_mut());
    out.matmul_from(ao_![0].t(), scr.view(), 0.5, 0.0);
    // SIGMA contribution
    if matches!(den_type, SIGMA | TAU) {
        // out += ao_![1].t() % (wv_![1] * ao_![0]);
        rt::mul_with_output(ao_![0], wv_![1], scr.view_mut());
        out.matmul_from(ao_![1].t(), scr.view(), 1.0, 1.0);
        // out += ao_![2].t() % (wv_![2] * ao_![0]);
        rt::mul_with_output(ao_![0], wv_![2], scr.view_mut());
        out.matmul_from(ao_![2].t(), scr.view(), 1.0, 1.0);
        // out += ao_![3].t() % (wv_![3] * ao_![0]);
        rt::mul_with_output(ao_![0], wv_![3], scr.view_mut());
        out.matmul_from(ao_![3].t(), scr.view(), 1.0, 1.0);
    }
    // TAU contribution
    if matches!(den_type, TAU) {
        // out += 0.25 * ao_![1].t() % (wv_![4] * ao_![1]);
        rt::mul_with_output(ao_![1], wv_![4], scr.view_mut());
        out.matmul_from(ao_![1].t(), scr.view(), 0.25, 1.0);
        // out += 0.25 * ao_![2].t() % (wv_![4] * ao_![2]);
        rt::mul_with_output(ao_![2], wv_![4], scr.view_mut());
        out.matmul_from(ao_![2].t(), scr.view(), 0.25, 1.0);
        // out += 0.25 * ao_![3].t() % (wv_![4] * ao_![3]);
        rt::mul_with_output(ao_![3], wv_![4], scr.view_mut());
        out.matmul_from(ao_![3].t(), scr.view(), 0.25, 1.0);
    }
}

/// Evaluate XC potential (1st order) with vxc_eff.
///
/// # Parameters
///
/// - `den_type`: the type of density to compute. Can be `RHO`, `SIGMA`, `TAU`.
/// - `vxc_eff` : effective potential for XC, shape `[ngrids, nvar]`
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `weights` : grid weights, shape `[ngrids]`
/// - `vxc` : output vxc, shape `[nao, nao]`
/// - `nchunk` : number of grid points to process in one chunk
pub fn rks_vxc_pot_with_eff_with_output(
    den_type: XCDenType,
    vxc_eff: TsrView,
    ao: TsrView,
    weights: TsrView,
    mut vxc: TsrMut,
    nchunk: usize,
) {
    ni_check_shape!(vxc_eff.ndim(), 2, "Effective potential must be 2-dim");
    let nvar = vxc_eff.shape()[1];
    let ngrids = vxc_eff.shape()[0];
    ni_check_shape!(weights.shape(), [ngrids], "Weights shape mismatch");
    ni_check_shape!(den_type.num_nvar(), nvar, "Dimension mismatch for input density type");
    ni_check_shape!(ao.ndim(), 3, "AO tensor must be 3-dim");
    ni_check_shape!(ao.shape()[0], ngrids, "AO grids dimension mismatch");
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient");
    let nao = ao.shape()[1];
    ni_check_shape!(vxc.shape(), [nao, nao], "Output shape mismatch");

    if den_type == LAPL {
        panic!("Contracting AO with LAPL density type is not supported");
    }

    // vxc_eff contraction
    let vxc_contracted = weights * vxc_eff;

    // buffer pool initialization
    // Each BufferPool lazily creates per-thread buffers; peak usage = nthreads * sum of init sizes f64
    let buffer_init = || vec![0.0; nchunk * nao];
    let buffer_pool = BufferPool::new(buffer_init);
    let vxc_init = || vec![0.0; nao * nao];
    let vxc_pool = BufferPool::new(vxc_init);

    // task numbers
    let ntask_grid = ngrids.div_ceil(nchunk);
    let ntask = ntask_grid;

    // atomic guard to avoid racing write
    let guard = Mutex::new(());

    (0..ntask).into_par_iter().for_each(|itask| {
        // determine task configuration
        let igrid = itask;

        // determine the grid chunk for this task
        let start = igrid * nchunk;
        let end = ((igrid + 1) * nchunk).min(ngrids);

        // get buffer from pool
        let mut buf = buffer_pool.get();
        let mut vxc_buf = vxc_pool.get();
        let mut vxc_local = rt::asarray((&mut vxc_buf, [nao, nao], ao.device()));

        // perform actual evaulation
        let vxc_contracted_chunk = vxc_contracted.i(start..end);
        let ao_chunk = ao.i(start..end);
        contract_ao_wv_without_symmetrize(
            den_type,
            vxc_contracted_chunk.view(),
            ao_chunk.view(),
            vxc_local.view_mut(),
            &mut buf,
        );

        // write back with lock
        let lock = guard.lock().unwrap();
        let mut vxc = unsafe { vxc.force_mut() };
        *&mut vxc += &vxc_local;
        drop(lock);

        // return buffer to pool
        buffer_pool.put(buf);
        vxc_pool.put(vxc_buf);
    });

    // finally symmetrize the output
    let vxc_buf = vxc.swapaxes(0, 1).to_owned();
    *&mut vxc += vxc_buf;
}

/// Evaluate XC potential (2nd order, RKS) with fxc_eff (parallel enhanced).
///
/// # Parameters
///
/// - `den_type`: the type of density to compute. Can be `RHO`, `SIGMA`, `TAU`.
/// - `fxc_eff` : effective XC kernel, shape `[ngrids, nvar, nvar]`
/// - `rho1` : first-order density response, shape `[ngrids, nvar, nset]`
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `weights` : grid weights, shape `[ngrids]`
/// - `fxc` : output fxc, shape `[nao, nao, nset]`
/// - `nchunk` : number of grid points to process in one chunk
pub fn rks_fxc_pot_with_eff_with_output(
    den_type: XCDenType,
    fxc_eff: TsrView,
    rho1: TsrView,
    ao: TsrView,
    weights: TsrView,
    mut fxc: TsrMut,
    nchunk: usize,
) {
    ni_check_shape!(rho1.ndim(), 3, "rho1 tensor must be 3-dim");
    let nset = rho1.shape()[2];
    let nvar = rho1.shape()[1];
    let ngrids = rho1.shape()[0];
    ni_check_shape!(fxc_eff.shape(), [ngrids, nvar, nvar], "fxc_eff shape mismatch");
    ni_check_shape!(weights.shape(), [ngrids], "Weights shape mismatch");
    ni_check_shape!(den_type.num_nvar(), nvar, "Dimension mismatch for input density type");
    ni_check_shape!(ao.ndim(), 3, "AO tensor must be 3-dim");
    ni_check_shape!(ao.shape()[0], ngrids, "AO grids dimension mismatch");
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient");
    let nao = ao.shape()[1];
    ni_check_shape!(fxc.shape(), [nao, nao, nset], "Output shape mismatch");

    if den_type == LAPL {
        panic!("Contracting AO with LAPL density type is not supported");
    }

    // fxc_eff contraction
    let fxc_eff_weighted = &weights * &fxc_eff;

    // buffer pool initialization
    // Each BufferPool lazily creates per-thread buffers; peak usage = nthreads * sum of init sizes f64
    let buffer_init = || vec![0.0; nchunk * nao];
    let buffer_pool = BufferPool::new(buffer_init);
    let fxc_init = || vec![0.0; nao * nao];
    let fxc_pool = BufferPool::new(fxc_init);

    // task numbers
    let ntask_grid = ngrids.div_ceil(nchunk);
    let ntask_i = nset;
    let ntask = ntask_grid * ntask_i;

    // atomic guard to avoid racing write
    let guard = (0..ntask_i).map(|_| Mutex::new(())).collect_vec();

    (0..ntask).into_par_iter().for_each(|itask| {
        // determine task configuration
        let i = itask % ntask_i;
        let igrid = itask / ntask_i;

        // determine the grid chunk for this task
        let start = igrid * nchunk;
        let end = ((igrid + 1) * nchunk).min(ngrids);

        // get buffer from pool
        let mut buf = buffer_pool.get();
        let mut fxc_buf = fxc_pool.get();
        let mut fxc_local = rt::asarray((&mut fxc_buf, [nao, nao], ao.device()));

        // perform actual evaluation
        let rho1_chunk = rho1.i((start..end, .., None, i));
        let fxc_eff_weighted_chunk = fxc_eff_weighted.i(start..end);
        let fxc_contracted_chunk = (&fxc_eff_weighted_chunk * rho1_chunk).sum_axes(1);
        let ao_chunk = ao.i(start..end);
        contract_ao_wv_without_symmetrize(
            den_type,
            fxc_contracted_chunk.view(),
            ao_chunk.view(),
            fxc_local.view_mut(),
            &mut buf,
        );

        // write back with lock
        let lock = guard[i].lock().unwrap();
        let mut fxc = unsafe { fxc.force_mut() };
        *&mut fxc.i_mut((.., .., i)) += &fxc_local;
        drop(lock);

        // return buffer to pool
        buffer_pool.put(buf);
        fxc_pool.put(fxc_buf);
    });

    // finally symmetrize the output
    let mut fxc_buf: Tsr = rt::zeros(([nao, nao], fxc.device()));
    for i in 0..nset {
        fxc_buf.assign(&fxc.i((.., .., i)).t());
        *&mut fxc.i_mut((.., .., i)) += &fxc_buf;
    }
}

/// Contract AO with wv for RHO/SIGMA/TAU, bra-transformed variant (parallel enhanced).
///
/// This produces an asymmetric `[nao, nocc]` output (no symmetrization needed).
///
/// # Parameters
///
/// - `den_type`: the type of density to compute. Can be `RHO`, `SIGMA`, `TAU`.
/// - `wv` : weight vector, shape `[ngrids, nvar]`
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `ao_bra` : bra-transformed AO values, shape `[ngrids, nocc, ncomp]`
/// - `out` : output buffer, shape `[nao, nocc]`
/// - `buf` : scratch buffer of length at least `ngrids * nocc`
fn contract_ao_wv_bra(
    den_type: XCDenType,
    wv: TsrView,
    ao: TsrView,
    ao_bra: TsrView,
    mut out: TsrMut,
    buf: &mut [f64],
) {
    ni_check_shape!(wv.ndim(), 2, "Weight vector must be 2-dim");
    let nvar = wv.shape()[1];
    let ngrids = wv.shape()[0];
    ni_check_shape!(den_type.num_nvar(), nvar, "Dimension mismatch for input density type");
    ni_check_shape!(ao.ndim(), 3, "AO tensor must be 3-dim");
    let nao = ao.shape()[1];
    ni_check_shape!(ao.shape()[0], ngrids, "AO grids dimension mismatch");
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient");
    ni_check_shape!(ao_bra.ndim(), 3, "ao_bra tensor must be 3-dim");
    ni_check_shape!(ao_bra.shape()[0], ngrids, "ao_bra grids dimension mismatch");
    let nocc = ao_bra.shape()[1];
    ni_check_shape!(ao_bra.shape()[2] >= den_type.num_ao_comp(), "ao_bra component dimension insufficient");
    ni_check_shape!(out.shape(), [nao, nocc], "Output shape mismatch");
    ni_check_shape!(buf.len() >= ngrids * nocc, "Buffer length insufficient");

    if den_type == LAPL {
        panic!("Contracting AO with LAPL density type is not supported");
    }

    macro_rules! ao_ {
        [$idx:expr] => {
            ao.i((.., .., $idx))
        };
    }
    macro_rules! ao_bra_ {
        [$idx:expr] => {
            ao_bra.i((.., .., $idx))
        };
    }
    macro_rules! wv_ {
        [$idx:expr] => {
            wv.i((.., $idx))
        };
    }

    let device = out.device().clone();
    let mut scr = rt::asarray((buf, [ngrids, nocc], &device));

    // RHO contribution (coefficient 1.0, not 0.5 — no symmetrization)
    rt::mul_with_output(ao_bra_![0], wv_![0], scr.view_mut());
    out.matmul_from(ao_![0].t(), scr.view(), 1.0, 0.0);

    // SIGMA contribution (6 terms: ao_bra[t]*wv[t]@ao[0].T + ao_bra[0]*wv[t]@ao[t].T)
    if matches!(den_type, SIGMA | TAU) {
        for t in 1..4 {
            rt::mul_with_output(ao_bra_![t], wv_![t], scr.view_mut());
            out.matmul_from(ao_![0].t(), scr.view(), 1.0, 1.0);
            rt::mul_with_output(ao_bra_![0], wv_![t], scr.view_mut());
            out.matmul_from(ao_![t].t(), scr.view(), 1.0, 1.0);
        }
    }

    // TAU contribution (coefficient 0.5, not 0.25 — no symmetrization)
    if matches!(den_type, TAU) {
        for t in 1..4 {
            rt::mul_with_output(ao_bra_![t], wv_![4], scr.view_mut());
            out.matmul_from(ao_![t].t(), scr.view(), 0.5, 1.0);
        }
    }
}

/// Evaluate XC potential (2nd order, RKS) with fxc_eff, bra transformed (parallel enhanced).
///
/// Bra is usually the occupied orbital coefficient (row-major applied to $\mu$, col-major applied
/// to $\nu$), which can lower the computational cost.
///
/// # Parameters
///
/// - `den_type`: the type of density to compute. Can be `RHO`, `SIGMA`, `TAU`.
/// - `fxc_eff` : effective XC kernel, shape `[ngrids, nvar, nvar]`
/// - `rho1` : first-order density response, shape `[ngrids, nvar, nset]`
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `weights` : grid weights, shape `[ngrids]`
/// - `bra` : bra orbital coefficients, shape `[nao, nocc]`
/// - `fxc` : output fxc (bra transformed), shape `[nao, nocc, nset]`
/// - `nchunk` : number of grid points to process in one chunk
#[allow(clippy::too_many_arguments)]
pub fn rks_fxc_pot_with_eff_bra_trans_with_output(
    den_type: XCDenType,
    fxc_eff: TsrView,
    rho1: TsrView,
    ao: TsrView,
    weights: TsrView,
    bra: TsrView,
    fxc: TsrMut,
    nchunk: usize,
) {
    ni_check_shape!(rho1.ndim(), 3, "rho1 tensor must be 3-dim");
    let nset = rho1.shape()[2];
    let nvar = rho1.shape()[1];
    let ngrids = rho1.shape()[0];
    ni_check_shape!(fxc_eff.shape(), [ngrids, nvar, nvar], "fxc_eff shape mismatch");
    ni_check_shape!(weights.shape(), [ngrids], "Weights shape mismatch");
    ni_check_shape!(den_type.num_nvar(), nvar, "Dimension mismatch for input density type");
    ni_check_shape!(ao.ndim(), 3, "AO tensor must be 3-dim");
    ni_check_shape!(ao.shape()[0], ngrids, "AO grids dimension mismatch");
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient");
    let nao = ao.shape()[1];
    ni_check_shape!(bra.ndim(), 2, "bra must be 2-dim");
    ni_check_shape!(bra.shape()[0], nao, "bra first dimension must match nao");
    let nocc = bra.shape()[1];
    ni_check_shape!(fxc.shape(), [nao, nocc, nset], "Output shape mismatch");

    if den_type == LAPL {
        panic!("Contracting AO with LAPL density type is not supported");
    }

    // Pre-compute ao_bra: [ngrids, nocc, ncomp]
    let device = ao.device().clone();
    let ncomp = den_type.num_ao_comp();
    let mut ao_bra = rt::zeros(([ngrids, nocc, ncomp], &device));
    for c in 0..ncomp {
        ao_bra.i_mut((.., .., c)).matmul_from(ao.i((.., .., c)), &bra, 1.0, 0.0);
    }

    // fxc_eff contraction
    let fxc_eff_weighted = &weights * &fxc_eff;

    // buffer pool initialization
    // Each BufferPool lazily creates per-thread buffers; peak usage = nthreads * sum of init sizes f64
    let buffer_init = || vec![0.0; nchunk * nocc];
    let buffer_pool = BufferPool::new(buffer_init);
    let fxc_init = || vec![0.0; nao * nocc];
    let fxc_pool = BufferPool::new(fxc_init);

    // task numbers
    let ntask_grid = ngrids.div_ceil(nchunk);
    let ntask_i = nset;
    let ntask = ntask_grid * ntask_i;

    // atomic guard to avoid racing write
    let guard = (0..ntask_i).map(|_| Mutex::new(())).collect_vec();

    (0..ntask).into_par_iter().for_each(|itask| {
        // determine task configuration
        let i = itask % ntask_i;
        let igrid = itask / ntask_i;

        // determine the grid chunk for this task
        let start = igrid * nchunk;
        let end = ((igrid + 1) * nchunk).min(ngrids);

        // get buffer from pool
        let mut buf = buffer_pool.get();
        let mut fxc_buf = fxc_pool.get();
        let mut fxc_local = rt::asarray((&mut fxc_buf, [nao, nocc], ao.device()));

        // perform actual evaluation
        let rho1_chunk = rho1.i((start..end, .., None, i));
        let fxc_eff_weighted_chunk = fxc_eff_weighted.i(start..end);
        let fxc_contracted_chunk = (&fxc_eff_weighted_chunk * rho1_chunk).sum_axes(1);
        let ao_chunk = ao.i(start..end);
        let ao_bra_chunk = ao_bra.i(start..end);
        contract_ao_wv_bra(
            den_type,
            fxc_contracted_chunk.view(),
            ao_chunk.view(),
            ao_bra_chunk.view(),
            fxc_local.view_mut(),
            &mut buf,
        );

        // write back with lock
        let lock = guard[i].lock().unwrap();
        let mut fxc = unsafe { fxc.force_mut() };
        *&mut fxc.i_mut((.., .., i)) += &fxc_local;
        drop(lock);

        // return buffer to pool
        buffer_pool.put(buf);
        fxc_pool.put(fxc_buf);
    });
}

/// Evaluate XC potential (3rd order, RKS) with kxc_eff (parallel enhanced).
///
/// # Parameters
///
/// - `den_type`: the type of density to compute. Can be `RHO`, `SIGMA`, `TAU`.
/// - `kxc_eff` : effective XC kernel, shape `[ngrids, nvar, nvar, nvar]`
/// - `rho1` : first-order density response, shape `[ngrids, nvar, nset1]`
/// - `rho2` : second-order density response, shape `[ngrids, nvar, nset2]`
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `weights` : grid weights, shape `[ngrids]`
/// - `kxc` : output kxc, shape `[nao, nao, nset1, nset2]`
/// - `nchunk` : number of grid points to process in one chunk
#[allow(clippy::too_many_arguments)]
pub fn rks_kxc_pot_with_eff_with_output(
    den_type: XCDenType,
    kxc_eff: TsrView,
    rho1: TsrView,
    rho2: TsrView,
    ao: TsrView,
    weights: TsrView,
    mut kxc: TsrMut,
    nchunk: usize,
) {
    ni_check_shape!(rho1.ndim(), 3, "rho1 tensor must be 3-dim");
    ni_check_shape!(rho2.ndim(), 3, "rho2 tensor must be 3-dim");
    let nset1 = rho1.shape()[2];
    let nset2 = rho2.shape()[2];
    let nvar = rho1.shape()[1];
    let ngrids = rho1.shape()[0];
    ni_check_shape!(kxc_eff.shape(), [ngrids, nvar, nvar, nvar], "kxc_eff shape mismatch");
    ni_check_shape!(rho2.shape()[0..2], [ngrids, nvar], "rho2 shape mismatch");
    ni_check_shape!(weights.shape(), [ngrids], "Weights shape mismatch");
    ni_check_shape!(den_type.num_nvar(), nvar, "Dimension mismatch for input density type");
    ni_check_shape!(ao.ndim(), 3, "AO tensor must be 3-dim");
    ni_check_shape!(ao.shape()[0], ngrids, "AO grids dimension mismatch");
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient");
    let nao = ao.shape()[1];
    ni_check_shape!(kxc.shape(), [nao, nao, nset1, nset2], "Output shape mismatch");

    if den_type == LAPL {
        panic!("Contracting AO with LAPL density type is not supported");
    }

    // kxc_eff contraction
    let kxc_eff_weighted = &weights * &kxc_eff;

    // buffer pool initialization
    // Each BufferPool lazily creates per-thread buffers; peak usage = nthreads * sum of init sizes f64
    let buffer_init = || vec![0.0; nchunk * nao];
    let buffer_pool = BufferPool::new(buffer_init);
    let kxc_init = || vec![0.0; nao * nao];
    let kxc_pool = BufferPool::new(kxc_init);

    // task numbers
    let ntask_grid = ngrids.div_ceil(nchunk);
    let ntask_i = nset1 * nset2;
    let ntask = ntask_grid * ntask_i;

    // atomic guard to avoid racing write
    let guard = (0..ntask_i).map(|_| Mutex::new(())).collect_vec();

    (0..ntask).into_par_iter().for_each(|itask| {
        // determine task configuration
        let j = itask % ntask_i;
        let i1 = j % nset1;
        let i2 = j / nset1;
        let igrid = itask / ntask_i;

        // determine the grid chunk for this task
        let start = igrid * nchunk;
        let end = ((igrid + 1) * nchunk).min(ngrids);

        // get buffer from pool
        let mut buf = buffer_pool.get();
        let mut kxc_buf = kxc_pool.get();
        let mut kxc_local = rt::asarray((&mut kxc_buf, [nao, nao], ao.device()));

        // perform actual evaluation
        let rho1_chunk = rho1.i((start..end, .., None, i1));
        let rho2_chunk = rho2.i((start..end, .., None, i2));
        let kxc_eff_weighted_chunk = kxc_eff_weighted.i(start..end);
        // Two-step contraction: first with rho1, then with rho2
        let temp = (&kxc_eff_weighted_chunk * rho1_chunk).sum_axes(1);
        let kxc_contracted_chunk = (&temp * rho2_chunk).sum_axes(1);
        let ao_chunk = ao.i(start..end);
        contract_ao_wv_without_symmetrize(
            den_type,
            kxc_contracted_chunk.view(),
            ao_chunk.view(),
            kxc_local.view_mut(),
            &mut buf,
        );

        // write back with lock
        let lock = guard[j].lock().unwrap();
        let mut kxc = unsafe { kxc.force_mut() };
        *&mut kxc.i_mut((.., .., i1, i2)) += &kxc_local;
        drop(lock);

        // return buffer to pool
        buffer_pool.put(buf);
        kxc_pool.put(kxc_buf);
    });

    // finally symmetrize the output
    let mut kxc_buf: Tsr = rt::zeros(([nao, nao], kxc.device()));
    for i2 in 0..nset2 {
        for i1 in 0..nset1 {
            kxc_buf.assign(&kxc.i((.., .., i1, i2)).t());
            *&mut kxc.i_mut((.., .., i1, i2)) += &kxc_buf;
        }
    }
}

/// Evaluate XC potential (1st order, UKS) with vxc_eff (parallel enhanced).
///
/// # Parameters
///
/// - `den_type`: the type of density to compute. Can be `RHO`, `SIGMA`, `TAU`.
/// - `vxc_eff` : effective XC potential, shape `[ngrids, nvar, 2]`
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `weights` : grid weights, shape `[ngrids]`
/// - `vxc` : output vxc, shape `[nao, nao, 2]`
/// - `nchunk` : number of grid points to process in one chunk
pub fn uks_vxc_pot_with_eff_with_output(
    den_type: XCDenType,
    vxc_eff: TsrView,
    ao: TsrView,
    weights: TsrView,
    mut vxc: TsrMut,
    nchunk: usize,
) {
    ni_check_shape!(vxc_eff.ndim(), 3, "Effective potential must be 3-dim");
    let nvar = vxc_eff.shape()[1];
    let ngrids = vxc_eff.shape()[0];
    ni_check_shape!(vxc_eff.shape()[2], 2, "vxc_eff must have 2 spin channels");
    ni_check_shape!(weights.shape(), [ngrids], "Weights shape mismatch");
    ni_check_shape!(den_type.num_nvar(), nvar, "Dimension mismatch for input density type");
    ni_check_shape!(ao.ndim(), 3, "AO tensor must be 3-dim");
    ni_check_shape!(ao.shape()[0], ngrids, "AO grids dimension mismatch");
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient");
    let nao = ao.shape()[1];
    ni_check_shape!(vxc.shape(), [nao, nao, 2], "Output shape mismatch");

    if den_type == LAPL {
        panic!("Contracting AO with LAPL density type is not supported");
    }

    // vxc_eff contraction
    let vxc_eff_weighted = &weights * &vxc_eff;

    // buffer pool initialization
    // Each BufferPool lazily creates per-thread buffers; peak usage = nthreads * sum of init sizes f64
    let buffer_init = || vec![0.0; nchunk * nao];
    let buffer_pool = BufferPool::new(buffer_init);
    let vxc_init = || vec![0.0; nao * nao];
    let vxc_pool = BufferPool::new(vxc_init);

    // task numbers
    let ntask_grid = ngrids.div_ceil(nchunk);
    let ntask_i = 2;
    let ntask = ntask_grid * ntask_i;

    // atomic guard to avoid racing write
    let guard = (0..ntask_i).map(|_| Mutex::new(())).collect_vec();

    (0..ntask).into_par_iter().for_each(|itask| {
        // determine task configuration
        let s = itask % ntask_i;
        let igrid = itask / ntask_i;

        // determine the grid chunk for this task
        let start = igrid * nchunk;
        let end = ((igrid + 1) * nchunk).min(ngrids);

        // get buffer from pool
        let mut buf = buffer_pool.get();
        let mut vxc_buf = vxc_pool.get();
        let mut vxc_local = rt::asarray((&mut vxc_buf, [nao, nao], ao.device()));

        // perform actual evaluation
        let vxc_contracted_chunk = vxc_eff_weighted.i((start..end, .., s));
        let ao_chunk = ao.i(start..end);
        contract_ao_wv_without_symmetrize(
            den_type,
            vxc_contracted_chunk.view(),
            ao_chunk.view(),
            vxc_local.view_mut(),
            &mut buf,
        );

        // write back with lock
        let lock = guard[s].lock().unwrap();
        let mut vxc = unsafe { vxc.force_mut() };
        *&mut vxc.i_mut((.., .., s)) += &vxc_local;
        drop(lock);

        // return buffer to pool
        buffer_pool.put(buf);
        vxc_pool.put(vxc_buf);
    });

    // finally symmetrize the output
    let mut vxc_buf: Tsr = rt::zeros(([nao, nao], vxc.device()));
    for s in 0..2 {
        vxc_buf.assign(&vxc.i((.., .., s)).t());
        *&mut vxc.i_mut((.., .., s)) += &vxc_buf;
    }
}

/// Evaluate XC potential (2nd order, UKS) with fxc_eff (parallel enhanced).
///
/// # Parameters
///
/// - `den_type`: the type of density to compute. Can be `RHO`, `SIGMA`, `TAU`.
/// - `fxc_eff` : effective XC kernel, shape `[ngrids, nvar, 2, nvar, 2]`
/// - `rho1` : first-order density response, shape `[ngrids, nvar, 2, nset]`
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `weights` : grid weights, shape `[ngrids]`
/// - `fxc` : output fxc, shape `[nao, nao, 2, nset]`
/// - `nchunk` : number of grid points to process in one chunk
pub fn uks_fxc_pot_with_eff_with_output(
    den_type: XCDenType,
    fxc_eff: TsrView,
    rho1: TsrView,
    ao: TsrView,
    weights: TsrView,
    mut fxc: TsrMut,
    nchunk: usize,
) {
    ni_check_shape!(rho1.ndim(), 4, "rho1 tensor must be 4-dim");
    let nset = rho1.shape()[3];
    let nvar = rho1.shape()[1];
    let ngrids = rho1.shape()[0];
    ni_check_shape!(rho1.shape()[2], 2, "rho1 must have 2 spin channels");
    ni_check_shape!(fxc_eff.shape(), [ngrids, nvar, 2, nvar, 2], "fxc_eff shape mismatch");
    ni_check_shape!(weights.shape(), [ngrids], "Weights shape mismatch");
    ni_check_shape!(den_type.num_nvar(), nvar, "Dimension mismatch for input density type");
    ni_check_shape!(ao.ndim(), 3, "AO tensor must be 3-dim");
    ni_check_shape!(ao.shape()[0], ngrids, "AO grids dimension mismatch");
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient");
    let nao = ao.shape()[1];
    ni_check_shape!(fxc.shape(), [nao, nao, 2, nset], "Output shape mismatch");

    if den_type == LAPL {
        panic!("Contracting AO with LAPL density type is not supported");
    }

    // fxc_eff contraction
    let fxc_eff_weighted = &weights * &fxc_eff;

    // buffer pool initialization
    // Each BufferPool lazily creates per-thread buffers; peak usage = nthreads * sum of init sizes f64
    let buffer_init = || vec![0.0; nchunk * nao];
    let buffer_pool = BufferPool::new(buffer_init);
    let fxc_init = || vec![0.0; nao * nao];
    let fxc_pool = BufferPool::new(fxc_init);

    // task numbers
    let ntask_grid = ngrids.div_ceil(nchunk);
    let ntask_i = 2 * nset;
    let ntask = ntask_grid * ntask_i;

    // atomic guard to avoid racing write
    let guard = (0..ntask_i).map(|_| Mutex::new(())).collect_vec();

    (0..ntask).into_par_iter().for_each(|itask| {
        // determine task configuration
        let j = itask % ntask_i;
        let s = j % 2;
        let i = j / 2;
        let igrid = itask / ntask_i;

        // determine the grid chunk for this task
        let start = igrid * nchunk;
        let end = ((igrid + 1) * nchunk).min(ngrids);

        // get buffer from pool
        let mut buf = buffer_pool.get();
        let mut fxc_buf = fxc_pool.get();
        let mut fxc_local = rt::asarray((&mut fxc_buf, [nao, nao], ao.device()));

        // perform actual evaluation
        let rho1_chunk = rho1.i((start..end, .., .., None, i));
        let fxc_eff_weighted_chunk = fxc_eff_weighted.i(start..end);
        // Contract over the inner spin+var pair (axes 1 and 2)
        let fxc_contracted_chunk = (&fxc_eff_weighted_chunk.i((.., .., .., .., s)) * rho1_chunk).sum_axes([1, 2]);
        let ao_chunk = ao.i(start..end);
        contract_ao_wv_without_symmetrize(
            den_type,
            fxc_contracted_chunk.view(),
            ao_chunk.view(),
            fxc_local.view_mut(),
            &mut buf,
        );

        // write back with lock
        let lock = guard[j].lock().unwrap();
        let mut fxc = unsafe { fxc.force_mut() };
        *&mut fxc.i_mut((.., .., s, i)) += &fxc_local;
        drop(lock);

        // return buffer to pool
        buffer_pool.put(buf);
        fxc_pool.put(fxc_buf);
    });

    // finally symmetrize the output
    let mut fxc_buf: Tsr = rt::zeros(([nao, nao], fxc.device()));
    for i in 0..nset {
        for s in 0..2 {
            fxc_buf.assign(&fxc.i((.., .., s, i)).t());
            *&mut fxc.i_mut((.., .., s, i)) += &fxc_buf;
        }
    }
}

/// Evaluate XC potential (3rd order, UKS) with kxc_eff (parallel enhanced).
///
/// # Parameters
///
/// - `den_type`: the type of density to compute. Can be `RHO`, `SIGMA`, `TAU`.
/// - `kxc_eff` : effective XC kernel, shape `[ngrids, nvar, 2, nvar, 2, nvar, 2]`
/// - `rho1` : first-order density response, shape `[ngrids, nvar, 2, nset1]`
/// - `rho2` : second-order density response, shape `[ngrids, nvar, 2, nset2]`
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `weights` : grid weights, shape `[ngrids]`
/// - `kxc` : output kxc, shape `[nao, nao, 2, nset1, nset2]`
/// - `nchunk` : number of grid points to process in one chunk
#[allow(clippy::too_many_arguments)]
pub fn uks_kxc_pot_with_eff_with_output(
    den_type: XCDenType,
    kxc_eff: TsrView,
    rho1: TsrView,
    rho2: TsrView,
    ao: TsrView,
    weights: TsrView,
    mut kxc: TsrMut,
    nchunk: usize,
) {
    ni_check_shape!(rho1.ndim(), 4, "rho1 tensor must be 4-dim");
    ni_check_shape!(rho2.ndim(), 4, "rho2 tensor must be 4-dim");
    let nset1 = rho1.shape()[3];
    let nset2 = rho2.shape()[3];
    let nvar = rho1.shape()[1];
    let ngrids = rho1.shape()[0];
    ni_check_shape!(rho1.shape()[2], 2, "rho1 must have 2 spin channels");
    ni_check_shape!(rho2.shape()[0..3], [ngrids, nvar, 2], "rho2 shape mismatch");
    ni_check_shape!(kxc_eff.shape(), [ngrids, nvar, 2, nvar, 2, nvar, 2], "kxc_eff shape mismatch");
    ni_check_shape!(weights.shape(), [ngrids], "Weights shape mismatch");
    ni_check_shape!(den_type.num_nvar(), nvar, "Dimension mismatch for input density type");
    ni_check_shape!(ao.ndim(), 3, "AO tensor must be 3-dim");
    ni_check_shape!(ao.shape()[0], ngrids, "AO grids dimension mismatch");
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient");
    let nao = ao.shape()[1];
    ni_check_shape!(kxc.shape(), [nao, nao, 2, nset1, nset2], "Output shape mismatch");

    if den_type == LAPL {
        panic!("Contracting AO with LAPL density type is not supported");
    }

    // kxc_eff contraction
    let kxc_eff_weighted = &weights * &kxc_eff;

    // buffer pool initialization
    // Each BufferPool lazily creates per-thread buffers; peak usage = nthreads * sum of init sizes f64
    let buffer_init = || vec![0.0; nchunk * nao];
    let buffer_pool = BufferPool::new(buffer_init);
    let kxc_init = || vec![0.0; nao * nao];
    let kxc_pool = BufferPool::new(kxc_init);

    // task numbers
    let ntask_grid = ngrids.div_ceil(nchunk);
    let ntask_i = 2 * nset1 * nset2;
    let ntask = ntask_grid * ntask_i;

    // atomic guard to avoid racing write
    let guard = (0..ntask_i).map(|_| Mutex::new(())).collect_vec();

    (0..ntask).into_par_iter().for_each(|itask| {
        // determine task configuration
        let j = itask % ntask_i;
        let s = j % 2;
        let i1 = (j / 2) % nset1;
        let i2 = (j / 2) / nset1;
        let igrid = itask / ntask_i;

        // determine the grid chunk for this task
        let start = igrid * nchunk;
        let end = ((igrid + 1) * nchunk).min(ngrids);

        // get buffer from pool
        let mut buf = buffer_pool.get();
        let mut kxc_buf = kxc_pool.get();
        let mut kxc_local = rt::asarray((&mut kxc_buf, [nao, nao], ao.device()));

        // perform actual evaluation
        let rho1_chunk = rho1.i((start..end, .., .., None, None, None, i1));
        let rho2_chunk = rho2.i((start..end, .., .., None, i2));
        let kxc_eff_weighted_chunk = kxc_eff_weighted.i(start..end);
        // Two-step contraction for UKS kxc
        let kxc_slice = kxc_eff_weighted_chunk.i((.., .., .., .., .., .., s));
        let temp = (&kxc_slice * rho1_chunk).sum_axes([1, 2]);
        let kxc_contracted_chunk = (&temp * rho2_chunk).sum_axes([1, 2]);
        let ao_chunk = ao.i(start..end);
        contract_ao_wv_without_symmetrize(
            den_type,
            kxc_contracted_chunk.view(),
            ao_chunk.view(),
            kxc_local.view_mut(),
            &mut buf,
        );

        // write back with lock
        let lock = guard[j].lock().unwrap();
        let mut kxc = unsafe { kxc.force_mut() };
        *&mut kxc.i_mut((.., .., s, i1, i2)) += &kxc_local;
        drop(lock);

        // return buffer to pool
        buffer_pool.put(buf);
        kxc_pool.put(kxc_buf);
    });

    // finally symmetrize the output
    let mut kxc_buf: Tsr = rt::zeros(([nao, nao], kxc.device()));
    for i2 in 0..nset2 {
        for i1 in 0..nset1 {
            for s in 0..2 {
                kxc_buf.assign(&kxc.i((.., .., s, i1, i2)).t());
                *&mut kxc.i_mut((.., .., s, i1, i2)) += &kxc_buf;
            }
        }
    }
}

/// Evaluate XC potential (2nd order, UKS) with fxc_eff, bra transformed (parallel enhanced).
///
/// Bra is usually the occupied orbital coefficient (row-major applied to $\mu$, col-major applied
/// to $\nu$), which can lower the computational cost.
///
/// # Parameters
///
/// - `den_type`: the type of density to compute. Can be `RHO`, `SIGMA`, `TAU`.
/// - `fxc_eff` : effective XC kernel, shape `[ngrids, nvar, 2, nvar, 2]`
/// - `rho1` : first-order density response, shape `[ngrids, nvar, 2, nset]`
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `weights` : grid weights, shape `[ngrids]`
/// - `bra` : bra orbital coefficients, first shape `[nao, nocc_alpha]`, second shape `[nao,
///   nocc_beta]`
/// - `fxc` : output fxc (bra transformed), first shape `[nao, nocc_alpha, nset]`, second shape
///   `[nao, nocc_beta, nset]`
/// - `nchunk` : number of grid points to process in one chunk
#[allow(clippy::too_many_arguments)]
pub fn uks_fxc_pot_with_eff_bra_trans_with_output(
    den_type: XCDenType,
    fxc_eff: TsrView,
    rho1: TsrView,
    ao: TsrView,
    weights: TsrView,
    bra: &[TsrView; 2],
    fxc: &mut [TsrMut; 2],
    nchunk: usize,
) {
    ni_check_shape!(rho1.ndim(), 4, "rho1 tensor must be 4-dim");
    let nset = rho1.shape()[3];
    let nvar = rho1.shape()[1];
    let ngrids = rho1.shape()[0];
    ni_check_shape!(rho1.shape()[2], 2, "rho1 must have 2 spin channels");
    ni_check_shape!(fxc_eff.shape(), [ngrids, nvar, 2, nvar, 2], "fxc_eff shape mismatch");
    ni_check_shape!(weights.shape(), [ngrids], "Weights shape mismatch");
    ni_check_shape!(den_type.num_nvar(), nvar, "Dimension mismatch for input density type");
    ni_check_shape!(ao.ndim(), 3, "AO tensor must be 3-dim");
    ni_check_shape!(ao.shape()[0], ngrids, "AO grids dimension mismatch");
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient");
    let nao = ao.shape()[1];
    for bra_spin in bra.iter() {
        ni_check_shape!(bra_spin.ndim(), 2, "bra tensors must be 2-dim");
        ni_check_shape!(bra_spin.shape()[0], nao, "bra first dimension must match nao");
    }
    let nocc_alpha = bra[0].shape()[1];
    let nocc_beta = bra[1].shape()[1];
    let nocc_max = nocc_alpha.max(nocc_beta);
    ni_check_shape!(fxc[0].shape(), [nao, nocc_alpha, nset], "Output shape mismatch");
    ni_check_shape!(fxc[1].shape(), [nao, nocc_beta, nset], "Output shape mismatch");

    if den_type == LAPL {
        panic!("Contracting AO with LAPL density type is not supported");
    }

    // Pre-compute ao_bra: [ngrids, nocc, 2, ncomp]
    let device = ao.device().clone();
    let ncomp = den_type.num_ao_comp();
    let mut ao_bra_alpha = rt::zeros(([ngrids, nocc_alpha, ncomp], &device));
    let mut ao_bra_beta = rt::zeros(([ngrids, nocc_beta, ncomp], &device));
    for c in 0..ncomp {
        ao_bra_alpha.i_mut((.., .., c)).matmul_from(ao.i((.., .., c)), &bra[0], 1.0, 0.0);
        ao_bra_beta.i_mut((.., .., c)).matmul_from(ao.i((.., .., c)), &bra[1], 1.0, 0.0);
    }

    // fxc_eff contraction
    let fxc_eff_weighted = &weights * &fxc_eff;

    // buffer pool initialization
    // Each BufferPool lazily creates per-thread buffers; peak usage = nthreads * sum of init sizes f64
    let buffer_init = || vec![0.0; nchunk * nao];
    let buffer_pool = BufferPool::new(buffer_init);
    let fxc_init = || vec![0.0; nao * nocc_max];
    let fxc_pool = BufferPool::new(fxc_init);

    // task numbers
    let ntask_grid = ngrids.div_ceil(nchunk);
    let ntask_i = 2 * nset;
    let ntask = ntask_grid * ntask_i;

    // atomic guard to avoid racing write
    let guard = (0..ntask_i).map(|_| Mutex::new(())).collect_vec();

    (0..ntask).into_par_iter().for_each(|itask| {
        // determine task configuration
        let j = itask % ntask_i;
        let s = j % 2;
        let i = j / 2;
        let igrid = itask / ntask_i;

        // determine the grid chunk for this task
        let start = igrid * nchunk;
        let end = ((igrid + 1) * nchunk).min(ngrids);

        // get buffer from pool
        let nocc = if s == 0 { nocc_alpha } else { nocc_beta };
        let mut buf = buffer_pool.get();
        let mut fxc_buf = fxc_pool.get();
        let mut fxc_local = rt::asarray((&mut fxc_buf, [nao, nocc], ao.device()));

        // perform actual evaluation
        let rho1_chunk = rho1.i((start..end, .., .., None, i));
        let fxc_eff_weighted_chunk = fxc_eff_weighted.i(start..end);
        // Contract over the inner spin+var pair (axes 1 and 2)
        let fxc_contracted_chunk = (&fxc_eff_weighted_chunk.i((.., .., .., .., s)) * rho1_chunk).sum_axes([1, 2]);
        let ao_chunk = ao.i(start..end);
        let ao_bra_chunk = if s == 0 { ao_bra_alpha.i(start..end) } else { ao_bra_beta.i(start..end) };
        contract_ao_wv_bra(
            den_type,
            fxc_contracted_chunk.view(),
            ao_chunk.view(),
            ao_bra_chunk.view(),
            fxc_local.view_mut(),
            &mut buf,
        );

        // write back with lock
        let lock = guard[j].lock().unwrap();
        let mut fxc_spin = unsafe { fxc[s].force_mut() };
        *&mut fxc_spin.i_mut((.., .., i)) += &fxc_local;
        drop(lock);

        // return buffer to pool
        buffer_pool.put(buf);
        fxc_pool.put(fxc_buf);
    });
}
