//! Density evaluation (parallel enhanced)

use super::prelude::*;
use XCDenType::*;

/// Evaluate density from density matrices (parallel enhanced).
///
/// # Parameters
///
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `dm_list` : density matrices, each of shape `[nao, nao]`; one per set
/// - `den_type` : which density components to compute
/// - `out` : output buffer, shape `[ngrids, num_rho_comp, nset]`
/// - `ngrids_chunk` : number of grid points to process in one chunk
pub fn get_rho_from_dm_with_output(ao: TsrView, dm_list: &[TsrView], den_type: XCDenType, out: TsrMut, nchunk: usize) {
    ni_check_shape!(ao.ndim(), 3, "AO values must be 3-dim");
    let nao = ao.shape()[1];

    for dm in dm_list {
        ni_check_shape!(dm.ndim(), 2, "Each density matrix must be 2-dim");
        ni_check_shape!(dm.shape()[0..2], [nao, nao], "Density matrix must match AO dimension");
    }
    let nset = dm_list.len();
    let ngrids = ao.shape()[0];
    let nvar = den_type.num_nvar();
    let device = ao.device().clone();

    ni_check_shape!(out.shape().clone(), [ngrids, nvar, nset], "Output shape mismatch");
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient");

    // buffer pool initialization
    // Each BufferPool lazily creates per-thread buffers; peak usage = nthreads * nchunk * (nao +
    // nvar) f64
    let scr_pool = BufferPool::new(|| vec![0.0; nchunk * nao]);
    let out_pool = BufferPool::new(|| vec![0.0; nchunk * nvar]);

    // task numbers
    let ntask_grid = ngrids.div_ceil(nchunk);
    let ntask_i = nset;
    let ntask = ntask_grid * ntask_i;

    (0..ntask).into_par_iter().for_each(|itask| {
        // determine task configuration
        let iset = itask % ntask_i;
        let igrid = itask / ntask_i;

        // determine the grid chunk for this task
        let start = igrid * nchunk;
        let end = ((igrid + 1) * nchunk).min(ngrids);
        let chunk_size = end - start;

        let dm = &dm_list[iset];
        let ao_chunk = ao.i(start..end);

        // get buffers from pool
        let mut scr_buf = scr_pool.get();
        let mut out_buf = out_pool.get();
        out_buf.fill(0.0);

        let mut scr = rt::asarray((&mut scr_buf, [chunk_size, nao].f(), &device));
        let mut out_local = rt::asarray((&mut out_buf, [chunk_size, nvar].f(), &device));

        // rho part
        scr.matmul_from(ao_chunk.i((.., .., 0)), dm, 1.0, 0.0);
        out_local.i_mut((.., 0)).vecdot_from(&scr, ao_chunk.i((.., .., 0)), 1);
        // sigma part
        if matches!(den_type, SIGMA | TAU | LAPL) {
            out_local.i_mut((.., 1..4)).vecdot_from(&scr.i((.., .., None)), &ao_chunk.i((.., .., 1..4)), 1);
            *&mut out_local.i_mut((.., 1..4)) *= 2.0;
        }
        // lapl part (second derivative of AO)
        if matches!(den_type, LAPL) {
            for t in [4, 7, 9] {
                *&mut out_local.i_mut((.., 5)) += 2.0 * rt::vecdot(&scr, ao_chunk.i((.., .., t)), 1);
            }
        }
        // tau part
        if matches!(den_type, TAU | LAPL) {
            for t in 1..4 {
                scr.matmul_from(ao_chunk.i((.., .., t)), dm, 0.5, 0.0);
                *&mut out_local.i_mut((.., 4)) += rt::vecdot(&scr, ao_chunk.i((.., .., t)), 1);
            }
        }
        // lapl part (tau contribution)
        if matches!(den_type, LAPL) {
            let tau_contrib = 4.0 * out_local.i((.., 4)).to_owned();
            *&mut out_local.i_mut((.., 5)) += tau_contrib;
        }

        // write back (should not race by design)
        let mut out = unsafe { out.force_mut() };
        out.i_mut((start..end, .., iset)).assign(&out_local);

        // return buffers to pool
        scr_pool.put(scr_buf);
        out_pool.put(out_buf);
    });
}

/// Evaluate density from homogeneous bra-ket (parallel enhanced).
///
/// # Parameters
///
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `bra_list` : orbital coefficient matrices, each of shape `[nao, nocc_i]`
/// - `den_type` : which density components to compute
/// - `out` : output buffer, shape `[ngrids, num_rho_comp, nset]`
/// - `nchunk` : number of grid points to process in one chunk
pub fn get_rho_from_homogeneous_braket_with_output(
    ao: TsrView,
    bra_list: &[TsrView],
    den_type: XCDenType,
    out: TsrMut,
    nchunk: usize,
) {
    ni_check_shape!(ao.ndim(), 3, "AO values must be 3-dim");
    let nao = ao.shape()[1];

    for bra in bra_list {
        ni_check_shape!(bra.ndim(), 2, "Each bra must be 2-dim");
        ni_check_shape!(nao, bra.shape()[0], "AO dimension must match braket dimension");
    }
    let nocc_max = bra_list.iter().map(|bra| bra.shape()[1]).max().unwrap_or(0);

    let nset = bra_list.len();
    let ngrids = ao.shape()[0];
    let nvar = den_type.num_nvar();
    let device = ao.device().clone();

    ni_check_shape!(out.shape().clone(), [ngrids, nvar, nset], "Output shape mismatch");
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient");

    // buffer pool initialization
    // Each BufferPool lazily creates per-thread buffers; peak usage = nthreads * nchunk * (2 *
    // nocc_max + nvar) f64
    let scr1_pool = BufferPool::new(|| vec![0.0; nchunk * nocc_max]);
    let scr2_pool = BufferPool::new(|| vec![0.0; nchunk * nocc_max]);
    let out_pool = BufferPool::new(|| vec![0.0; nchunk * nvar]);

    // task numbers
    let ntask_grid = ngrids.div_ceil(nchunk);
    let ntask_i = nset;
    let ntask = ntask_grid * ntask_i;

    (0..ntask).into_par_iter().for_each(|itask| {
        // determine task configuration
        let iset = itask % ntask_i;
        let igrid = itask / ntask_i;

        // determine the grid chunk for this task
        let start = igrid * nchunk;
        let end = ((igrid + 1) * nchunk).min(ngrids);
        let chunk_size = end - start;

        let bra = &bra_list[iset];
        let nocc = bra.shape()[1];
        let ao_chunk = ao.i(start..end);

        // get buffers from pool
        let mut scr1_buf = scr1_pool.get();
        let mut scr2_buf = scr2_pool.get();
        let mut out_buf = out_pool.get();
        out_buf.fill(0.0);

        let mut scr1 = rt::asarray((&mut scr1_buf, [chunk_size, nocc].f(), &device));
        let mut scr2 = rt::asarray((&mut scr2_buf, [chunk_size, nocc].f(), &device));
        let mut out_local = rt::asarray((&mut out_buf, [chunk_size, nvar].f(), &device));

        // rho part
        scr1.matmul_from(ao_chunk.i((.., .., 0)), bra, 1.0, 0.0);
        out_local.i_mut((.., 0)).vecdot_from(&scr1, &scr1, 1);
        if matches!(den_type, SIGMA | TAU | LAPL) {
            for t in 1..4 {
                scr2.matmul_from(ao_chunk.i((.., .., t)), bra, 1.0, 0.0);
                // sigma part
                out_local.i_mut((.., t)).vecdot_from(&scr1, &scr2, 1);
                *&mut out_local.i_mut((.., t)) *= 2.;
                // tau part
                if matches!(den_type, TAU | LAPL) {
                    *&mut out_local.i_mut((.., 4)) += 0.5 * rt::vecdot(&scr2, &scr2, 1);
                }
            }
        }
        if matches!(den_type, LAPL) {
            // lapl part (second derivative of AO)
            for t in [4, 7, 9] {
                scr2.matmul_from(ao_chunk.i((.., .., t)), bra, 1.0, 0.0);
                *&mut out_local.i_mut((.., 5)) += 2.0 * rt::vecdot(&scr1, &scr2, 1);
            }
            // lapl part (tau contribution)
            let tau_contrib = 4.0 * out_local.i((.., 4)).to_owned();
            *&mut out_local.i_mut((.., 5)) += tau_contrib;
        }

        // write back (should not race by design)
        let mut out = unsafe { out.force_mut() };
        out.i_mut((start..end, .., iset)).assign(&out_local);

        // return buffers to pool
        scr1_pool.put(scr1_buf);
        scr2_pool.put(scr2_buf);
        out_pool.put(out_buf);
    });
}

/// Evaluate density from one bra with multiple kets (parallel enhanced).
///
/// # Parameters
///
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `bra` : shared orbital coefficient matrix, shape `[nao, nocc]`
/// - `ket_list` : orbital coefficient matrices, each of shape `[nao, nocc]`
/// - `den_type` : which density components to compute
/// - `out` : output buffer, shape `[ngrids, num_rho_comp, nset]`
/// - `nchunk` : number of grid points to process in one chunk
pub fn get_rho_from_one_bra_mult_ket_with_output(
    ao: TsrView,
    bra: TsrView,
    ket_list: &[TsrView],
    den_type: XCDenType,
    out: TsrMut,
    nchunk: usize,
) {
    ni_check_shape!(ao.ndim(), 3, "AO values must be 3-dim");
    let nao = ao.shape()[1];

    ni_check_shape!(bra.ndim(), 2, "Bra must be 2-dim");
    ni_check_shape!(nao, bra.shape()[0], "Bra first dimension must match AO dimension");
    let nocc = bra.shape()[1];

    for ket in ket_list {
        ni_check_shape!(ket.ndim(), 2, "Each ket must be 2-dim");
        ni_check_shape!(nao, ket.shape()[0], "Ket first dimension must match AO dimension");
        ni_check_shape!(ket.shape()[1], nocc, "Ket second dimension must match bra");
    }
    let nset = ket_list.len();
    let ngrids = ao.shape()[0];
    let nvar = den_type.num_nvar();
    let device = ao.device().clone();

    ni_check_shape!(out.shape().clone(), [ngrids, nvar, nset], "Output shape mismatch");
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient");

    // buffer pool initialization
    // Each BufferPool lazily creates per-thread buffers; peak usage = nthreads * nchunk * (3 *
    // nocc + nvar) f64
    let scr1_pool = BufferPool::new(|| vec![0.0; nchunk * nocc]);
    let scr2_pool = BufferPool::new(|| vec![0.0; nchunk * nocc]);
    let scr3_pool = BufferPool::new(|| vec![0.0; nchunk * nocc]);
    let out_pool = BufferPool::new(|| vec![0.0; nchunk * nvar]);

    // task numbers
    let ntask_grid = ngrids.div_ceil(nchunk);
    let ntask_i = nset;
    let ntask = ntask_grid * ntask_i;

    (0..ntask).into_par_iter().for_each(|itask| {
        // determine task configuration
        let iset = itask % ntask_i;
        let igrid = itask / ntask_i;

        // determine the grid chunk for this task
        let start = igrid * nchunk;
        let end = ((igrid + 1) * nchunk).min(ngrids);
        let chunk_size = end - start;

        let ket = &ket_list[iset];
        let ao_chunk = ao.i(start..end);

        // get buffers from pool
        let mut scr1_buf = scr1_pool.get();
        let mut scr2_buf = scr2_pool.get();
        let mut scr3_buf = scr3_pool.get();
        let mut out_buf = out_pool.get();
        out_buf.fill(0.0);

        let mut scr1 = rt::asarray((&mut scr1_buf, [chunk_size, nocc].f(), &device));
        let mut scr2 = rt::asarray((&mut scr2_buf, [chunk_size, nocc].f(), &device));
        let mut scr3 = rt::asarray((&mut scr3_buf, [chunk_size, nocc].f(), &device));
        let mut out_local = rt::asarray((&mut out_buf, [chunk_size, nvar].f(), &device));

        // Pre-compute scr1 = ao_0_chunk @ bra
        scr1.matmul_from(ao_chunk.i((.., .., 0)), &bra, 1.0, 0.0);

        // rho part
        scr2.matmul_from(ao_chunk.i((.., .., 0)), ket, 1.0, 0.0);
        out_local.i_mut((.., 0)).vecdot_from(&scr1, &scr2, 1);

        // sigma part
        if matches!(den_type, SIGMA | TAU | LAPL) {
            for t in 1..4 {
                scr3.matmul_from(ao_chunk.i((.., .., t)), ket, 1.0, 0.0);
                out_local.i_mut((.., t)).vecdot_from(&scr1, &scr3, 1);
                scr3.matmul_from(ao_chunk.i((.., .., t)), &bra, 1.0, 0.0);
                *&mut out_local.i_mut((.., t)) += rt::vecdot(&scr3, &scr2, 1);
            }
        }

        // lapl part (second derivative of AO), must come before tau which overwrites scr2
        if matches!(den_type, LAPL) {
            for t in [4, 7, 9] {
                scr3.matmul_from(ao_chunk.i((.., .., t)), ket, 1.0, 0.0);
                *&mut out_local.i_mut((.., 5)) += rt::vecdot(&scr1, &scr3, 1);
                scr3.matmul_from(ao_chunk.i((.., .., t)), &bra, 1.0, 0.0);
                *&mut out_local.i_mut((.., 5)) += rt::vecdot(&scr3, &scr2, 1);
            }
        }

        // tau part (overwrites scr2, which is no longer needed for sigma/lapl)
        if matches!(den_type, TAU | LAPL) {
            for t in 1..4 {
                scr2.matmul_from(ao_chunk.i((.., .., t)), ket, 1.0, 0.0);
                scr3.matmul_from(ao_chunk.i((.., .., t)), &bra, 1.0, 0.0);
                *&mut out_local.i_mut((.., 4)) += 0.5 * rt::vecdot(&scr3, &scr2, 1);
            }
        }

        // lapl part (tau contribution)
        if matches!(den_type, LAPL) {
            let tau_contrib = 4.0 * out_local.i((.., 4)).to_owned();
            *&mut out_local.i_mut((.., 5)) += tau_contrib;
        }

        // write back (should not race by design)
        let mut out = unsafe { out.force_mut() };
        out.i_mut((start..end, .., iset)).assign(&out_local);

        // return buffers to pool
        scr1_pool.put(scr1_buf);
        scr2_pool.put(scr2_buf);
        scr3_pool.put(scr3_buf);
        out_pool.put(out_buf);
    });
}

/// Evaluate density from multiple bra-ket pairs (parallel enhanced).
///
/// # Parameters
///
/// - `ao` : AO values and derivatives, shape `[ngrids, nao, ncomp]`
/// - `bra_list` : orbital coefficient matrices for bra
/// - `ket_list` : orbital coefficient matrices for ket
/// - `den_type` : which density components to compute
/// - `out` : output buffer, shape `[ngrids, num_rho_comp, nset]`
/// - `nchunk` : number of grid points to process in one chunk
pub fn get_rho_from_mult_bra_mult_ket_with_output(
    ao: TsrView,
    bra_list: &[TsrView],
    ket_list: &[TsrView],
    den_type: XCDenType,
    out: TsrMut,
    nchunk: usize,
) {
    ni_check_shape!(ao.ndim(), 3, "AO values must be 3-dim");
    let nao = ao.shape()[1];

    ni_check_shape!(bra_list.len(), ket_list.len(), "bra_list and ket_list must have same length");
    let nocc_max = bra_list.iter().map(|bra| bra.shape()[1]).max().unwrap_or(0);

    for (bra, ket) in bra_list.iter().zip(ket_list.iter()) {
        ni_check_shape!(bra.ndim(), 2, "Each bra must be 2-dim");
        ni_check_shape!(ket.ndim(), 2, "Each ket must be 2-dim");
        ni_check_shape!(nao, bra.shape()[0], "Bra first dimension must match AO dimension");
        ni_check_shape!(nao, ket.shape()[0], "Ket first dimension must match AO dimension");
        ni_check_shape!(bra.shape()[1], ket.shape()[1], "Bra and ket occupation must match");
    }
    let nset = bra_list.len();
    let ngrids = ao.shape()[0];
    let nvar = den_type.num_nvar();
    let device = ao.device().clone();

    ni_check_shape!(out.shape().clone(), [ngrids, nvar, nset], "Output shape mismatch");
    ni_check_shape!(ao.shape()[2] >= den_type.num_ao_comp(), "AO component dimension insufficient");

    // buffer pool initialization
    // Each BufferPool lazily creates per-thread buffers; peak usage = nthreads * nchunk * (3 *
    // nocc_max + nvar) f64
    let scr1_pool = BufferPool::new(|| vec![0.0; nchunk * nocc_max]);
    let scr2_pool = BufferPool::new(|| vec![0.0; nchunk * nocc_max]);
    let scr3_pool = BufferPool::new(|| vec![0.0; nchunk * nocc_max]);
    let out_pool = BufferPool::new(|| vec![0.0; nchunk * nvar]);

    // task numbers
    let ntask_grid = ngrids.div_ceil(nchunk);
    let ntask_i = nset;
    let ntask = ntask_grid * ntask_i;

    (0..ntask).into_par_iter().for_each(|itask| {
        // determine task configuration
        let iset = itask % ntask_i;
        let igrid = itask / ntask_i;

        // determine the grid chunk for this task
        let start = igrid * nchunk;
        let end = ((igrid + 1) * nchunk).min(ngrids);
        let chunk_size = end - start;

        let bra = &bra_list[iset];
        let ket = &ket_list[iset];
        let nocc = bra.shape()[1];
        let ao_chunk = ao.i(start..end);

        // get buffers from pool
        let mut scr1_buf = scr1_pool.get();
        let mut scr2_buf = scr2_pool.get();
        let mut scr3_buf = scr3_pool.get();
        let mut out_buf = out_pool.get();
        out_buf.fill(0.0);

        let mut scr1 = rt::asarray((&mut scr1_buf, [chunk_size, nocc].f(), &device));
        let mut scr2 = rt::asarray((&mut scr2_buf, [chunk_size, nocc].f(), &device));
        let mut scr3 = rt::asarray((&mut scr3_buf, [chunk_size, nocc].f(), &device));
        let mut out_local = rt::asarray((&mut out_buf, [chunk_size, nvar].f(), &device));

        // rho part
        scr1.matmul_from(ao_chunk.i((.., .., 0)), bra, 1.0, 0.0);
        scr2.matmul_from(ao_chunk.i((.., .., 0)), ket, 1.0, 0.0);
        out_local.i_mut((.., 0)).vecdot_from(&scr1, &scr2, 1);

        // sigma part
        if matches!(den_type, SIGMA | TAU | LAPL) {
            for t in 1..4 {
                scr3.matmul_from(ao_chunk.i((.., .., t)), ket, 1.0, 0.0);
                out_local.i_mut((.., t)).vecdot_from(&scr1, &scr3, 1);
                scr3.matmul_from(ao_chunk.i((.., .., t)), bra, 1.0, 0.0);
                *&mut out_local.i_mut((.., t)) += rt::vecdot(&scr3, &scr2, 1);
            }
        }

        // lapl part (second derivative of AO), must come before tau which overwrites scr1/scr2
        if matches!(den_type, LAPL) {
            for t in [4, 7, 9] {
                scr3.matmul_from(ao_chunk.i((.., .., t)), ket, 1.0, 0.0);
                *&mut out_local.i_mut((.., 5)) += rt::vecdot(&scr1, &scr3, 1);
                scr3.matmul_from(ao_chunk.i((.., .., t)), bra, 1.0, 0.0);
                *&mut out_local.i_mut((.., 5)) += rt::vecdot(&scr3, &scr2, 1);
            }
        }

        // tau part (overwrites scr1/scr2, which are no longer needed for sigma/lapl)
        if matches!(den_type, TAU | LAPL) {
            for t in 1..4 {
                scr1.matmul_from(ao_chunk.i((.., .., t)), bra, 1.0, 0.0);
                scr2.matmul_from(ao_chunk.i((.., .., t)), ket, 1.0, 0.0);
                *&mut out_local.i_mut((.., 4)) += 0.5 * rt::vecdot(&scr1, &scr2, 1);
            }
        }

        // lapl part (tau contribution)
        if matches!(den_type, LAPL) {
            let tau_contrib = 4.0 * out_local.i((.., 4)).to_owned();
            *&mut out_local.i_mut((.., 5)) += tau_contrib;
        }

        // write back (should not race by design)
        let mut out = unsafe { out.force_mut() };
        out.i_mut((start..end, .., iset)).assign(&out_local);

        // return buffers to pool
        scr1_pool.put(scr1_buf);
        scr2_pool.put(scr2_buf);
        scr3_pool.put(scr3_buf);
        out_pool.put(out_buf);
    });
}
