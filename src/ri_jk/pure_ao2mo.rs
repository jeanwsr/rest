use crate::utilities::buffer_pool::BufferPool;

use super::prelude_dev::*;
use core::any::TypeId;

/* #region ao2mo s2ij_to_s1 */

/// Basis transformation ao2mo `tp(uv)P, uiA, vaA -> iaPA`.
///
/// Inputs will be
/// - `j3c`: shape (nao_tp, ...np), at least 2-dim.
/// - `bra`: vector of shape (nao, ni); `ni` does not need to be the same for all `bra` in the set.
/// - `ket`: vector of shape (nao, na); `na` does not need to be the same for all `ket` in the set.
/// - Number of set in `bra` and `ket` must be at the same (`nset`).
///
/// Output will be
/// - `output`: vector of shape (ni, na, ...np)
///
/// This function may try to first contract `bra` if `ni` is smaller, otherwise first contract `ket`
/// if `na` is smaller.
///
/// For implementation, though `?symm` may be better, `?symm` does not lower the FLOPs, but only
/// merely lowers the memory footprint. So I believe using directly `dgemm` is also fine.
pub fn get_ao2mo_s2ij_to_s1_notrans_with_output<T, O>(
    j3c: TsrView<T>,
    uplo: FlagUpLo,
    bra: &[TsrView<T>],
    ket: &[TsrView<T>],
    output: &[TsrMut<O>],
    cast: impl Fn(T) -> O + Send + Sync,
) where
    T: BlasFloat + 'static,
    O: BlasFloat + 'static,
{
    // shape sanity check and changes
    assert!(j3c.ndim() >= 1, "j3c must have at least 1 dimension");
    let nao_tp = j3c.shape()[0];
    let nao = util::retrive_tp_dim(nao_tp).unwrap();
    let shape_j3c_remain = j3c.shape()[1..].to_vec();

    assert!(bra.len() == ket.len(), "Number of sets in bra and ket must be the same");
    assert!(output.len() == bra.len(), "Output length must match number of sets in bra and ket");
    let nset = bra.len();

    // reshape tensors to standard shapes
    let j3c = j3c.reshape((nao_tp, -1));
    let np = j3c.shape()[1];

    // - output_flat: output mutables reshaped to (ni, na, np)
    // - ns: minimum of ni and na among all sets, used for scratch size
    let mut output_flat = vec![];
    let mut ns = usize::MAX;
    for iset in 0..nset {
        let ni = bra[iset].shape()[1];
        let na = ket[iset].shape()[1];
        assert!(bra[iset].ndim() == 2, "bra[{iset}] must have 2 dimensions");
        assert!(ket[iset].ndim() == 2, "ket[{iset}] must have 2 dimensions");
        assert_eq!(bra[iset].shape()[0], nao, "bra[{iset}] must have the same number of AO as j3c");
        assert_eq!(ket[iset].shape()[0], nao, "ket[{iset}] must have the same number of AO as j3c");
        assert_eq!(output[iset].shape()[0], ni, "output[{iset}] must have the same ni as bra[{iset}]");
        assert_eq!(output[iset].shape()[1], na, "output[{iset}] must have the same na as ket[{iset}]");
        assert_eq!(
            output[iset].shape()[2..],
            shape_j3c_remain,
            "output[{iset}] must have the same remaining shape as j3c"
        );
        let out_iset = unsafe { output[iset].force_mut().into_compatible_shape((ni, na, np), ColMajor) };
        output_flat.push(out_iset);
        ns = ns.max(ni.min(na));
    }

    // scratch buffer for half-transform
    let scratch_pool = BufferPool::new(|| unsafe { uninitialized_vec(nao * ns).unwrap() });

    // check if type T is the same to O; if so, create a scratch buffer
    let scratch_out_pool = BufferPool::new(|| unsafe { uninitialized_vec(ns * ns).unwrap() });

    (0..np).into_par_iter().for_each(|p| {
        let j3c_p = j3c.i((.., p)).unpack_tri(uplo, FlagSymm::He);
        for iset in 0..nset {
            let ni = bra[iset].shape()[1];
            let na = ket[iset].shape()[1];
            let out_iset = output_flat[iset].i((.., .., p));
            let mut out_iset = unsafe { out_iset.force_mut() };
            let mut scratch_buf = scratch_pool.get();
            if ni < na {
                // contract bra first
                // - `bra` ui, `j3c` uv -> `scratch` iv
                let mut scratch = rt::asarray((&mut scratch_buf, [nao, ni].f(), j3c.device()));
                scratch.matmul_from(&bra[iset].t(), &j3c_p, T::one(), T::zero());
                // - `scratch` iv, `ket` va -> `out` ia
                // same type: out_iset.matmul_from(&scratch, &ket[iset], T::one(), T::zero());
                if TypeId::of::<T>() == TypeId::of::<O>() {
                    // unsafe cast
                    let mut out_iset = unsafe { core::mem::transmute::<TsrMut<O>, TsrMut<T>>(out_iset) };
                    out_iset.matmul_from(&scratch, &ket[iset], T::one(), T::zero());
                } else {
                    let mut scratch_out_buf = scratch_out_pool.get();
                    let mut scratch_out = rt::asarray((&mut scratch_out_buf, [ni, na].f(), j3c.device()));
                    scratch_out.matmul_from(&scratch, &ket[iset], T::one(), T::zero());
                    out_iset.assign(&scratch_out.mapv(&cast));
                    scratch_out_pool.put(scratch_out_buf);
                };
            } else {
                // contract ket first
                // - `j3c` uv, `ket` va -> `scratch` ua
                let mut scratch = rt::asarray((&mut scratch_buf, [nao, na].f(), j3c.device()));
                scratch.matmul_from(&j3c_p, &ket[iset], T::one(), T::zero());
                // - `bra` ui, `scratch` ua -> `out` ia
                // same type: out_iset.matmul_from(&bra[iset].t(), &scratch, T::one(), T::zero());
                if TypeId::of::<T>() == TypeId::of::<O>() {
                    // unsafe cast
                    let mut out_iset = unsafe { core::mem::transmute::<TsrMut<O>, TsrMut<T>>(out_iset) };
                    out_iset.matmul_from(&bra[iset].t(), &scratch, T::one(), T::zero());
                } else {
                    let mut scratch_out_buf = scratch_out_pool.get();
                    let mut scratch_out = rt::asarray((&mut scratch_out_buf, [ni, na].f(), j3c.device()));
                    scratch_out.matmul_from(&bra[iset].t(), &scratch, T::one(), T::zero());
                    out_iset.assign(&scratch_out.mapv(&cast));
                    scratch_out_pool.put(scratch_out_buf);
                };
            }
            scratch_pool.put(scratch_buf);
        }
    });
}

pub fn get_ao2mo_s2ij_to_s1_notrans<T, O>(
    j3c: TsrView<T>,
    uplo: FlagUpLo,
    bra: &[TsrView<T>],
    ket: &[TsrView<T>],
    cast: impl Fn(T) -> O + Send + Sync,
) -> Vec<Tsr<O>>
where
    T: BlasFloat + 'static,
    O: BlasFloat + 'static,
{
    let mut output = vec![];
    for iset in 0..bra.len() {
        let ni = bra[iset].shape()[1];
        let na = ket[iset].shape()[1];
        let shape_j3c_remain = j3c.shape()[1..].to_vec();
        let shape_iset = [ni, na].into_iter().chain(shape_j3c_remain.into_iter()).collect_vec();
        let out_iset = rt::zeros((shape_iset.f(), j3c.device()));
        output.push(out_iset);
    }
    let output_mut = output.iter_mut().map(|t| t.view_mut()).collect_vec();
    get_ao2mo_s2ij_to_s1_notrans_with_output(j3c, uplo, bra, ket, &output_mut, &cast);
    output
}

/* #endregion */

/* #region ao2mo s2ij_to_s1 trans */

/// Basis transformation ao2mo `tp(uv)P, uiA, vaA -> PiaA`.
///
/// Similar inputs compared to [``get_ao2mo_s2ij_to_s1_notrans_with_output``].
///
/// Output will be
/// - `output`: vector of shape (...np, ni, na)
///
/// Note this algorithm requires large scratch area; nbatch will be set to 20% number of auxiliary
/// basis (with minimum of 16) by default, to control the memory usage. For finer control, the outer
/// routine will need to calcuate the required scratch size.
///
/// Memory estimation:
/// - batched:
///   - `j3c_batch`: (nao, nao, nbatch)
///   - `scratch`: (nao, ns, nbatch)
///   - `scratch_out`: (ns, nbatch, nthread), optional
pub fn get_ao2mo_s2ij_to_s1_trans_with_output<T, O>(
    j3c: TsrView<T>,
    uplo: FlagUpLo,
    bra: &[TsrView<T>],
    ket: &[TsrView<T>],
    nbatch: Option<usize>,
    output: &[TsrMut<O>],
    cast: impl Fn(T) -> O + Send + Sync,
) where
    T: BlasFloat + 'static,
    O: BlasFloat + 'static,
{
    // shape sanity check and changes
    assert!(j3c.ndim() >= 1, "j3c must have at least 1 dimension");
    let nao_tp = j3c.shape()[0];
    let nao = util::retrive_tp_dim(nao_tp).unwrap();
    let shape_j3c_remain = j3c.shape()[1..].to_vec();

    assert!(bra.len() == ket.len(), "Number of sets in bra and ket must be the same");
    assert!(output.len() == bra.len(), "Output length must match number of sets in bra and ket");
    let nset = bra.len();

    // reshape tensors to standard shapes
    let j3c = j3c.reshape((nao_tp, -1));
    let np = j3c.shape()[1];

    const MIN_NBATCH: usize = 16;
    let nbatch = nbatch.unwrap_or((np as f64 * 0.2).ceil() as usize).max(MIN_NBATCH).min(np);

    // - output_flat: output mutables reshaped to (ni, na, np)
    // - ns: maximum of ni and na among all sets, used for scratch size
    let mut output_flat = vec![];
    let mut ns = 0;
    for iset in 0..nset {
        let ni = bra[iset].shape()[1];
        let na = ket[iset].shape()[1];
        assert!(bra[iset].ndim() == 2, "bra[{iset}] must have 2 dimensions");
        assert!(ket[iset].ndim() == 2, "ket[{iset}] must have 2 dimensions");
        assert_eq!(bra[iset].shape()[0], nao, "bra[{iset}] must have the same number of AO as j3c");
        assert_eq!(ket[iset].shape()[0], nao, "ket[{iset}] must have the same number of AO as j3c");
        let out_shape = output[iset].shape().to_vec();
        assert!(out_shape.len() >= 2, "output shape must be larger than two dimensions");
        assert_eq!(out_shape[out_shape.len() - 2], ni, "output[{iset}] must have the same ni as bra[{iset}]");
        assert_eq!(out_shape[out_shape.len() - 1], na, "output[{iset}] must have the same na as ket[{iset}]");
        assert_eq!(
            output[iset].shape()[..out_shape.len() - 2],
            shape_j3c_remain,
            "output[{iset}] must have the same remaining shape as j3c"
        );
        let out_iset = unsafe { output[iset].force_mut().into_compatible_shape((np, ni, na), ColMajor) };
        output_flat.push(out_iset);
        ns = ns.max(ni.max(na));
    }

    // scratch buffer for half-transform
    let mut scratch_buf: Vec<T> = unsafe { uninitialized_vec(nao * ns * nbatch).unwrap() };

    // check if type T is the same to O; if so, create a scratch buffer
    let scratch_out_pool = BufferPool::new(|| unsafe { uninitialized_vec(ns * nbatch).unwrap() });

    for p0 in (0..np).step_by(nbatch) {
        let p1 = (p0 + nbatch).min(np);
        let nb = p1 - p0;
        let j3c_batch = j3c.i((.., p0..p1)).unpack_tri(uplo, FlagSymm::He);
        for iset in 0..nset {
            let ni = bra[iset].shape()[1];
            let na = ket[iset].shape()[1];
            let out_iset = output_flat[iset].i_mut((p0..p1, .., ..));
            if ni < na {
                // contract bra first
                // - `j3c` uvP, `bra` ui -> `scratch` vPi, usual dgemm (combine indices vP, contract u)
                let mut scratch = rt::asarray((&mut scratch_buf, [nao * nb, ni].f(), j3c.device()));
                let j3c_batch = j3c_batch.reshape((nao, nao * nb));
                scratch.matmul_from(&j3c_batch.t(), &bra[iset], T::one(), T::zero());
                // - `scratch` vPi, `ket` va -> `out` Pia, batch by i (usual dgemm requires P contiguous)
                let scratch = rt::asarray((&mut scratch_buf, [nao, nb, ni].f(), j3c.device()));
                (0..ni).into_par_iter().for_each(|i| {
                    let scratch_i = scratch.i((.., .., i));
                    let out_i = out_iset.i((.., i, ..));
                    let mut out_i = unsafe { out_i.force_mut() };
                    // if same type
                    // out_i.matmul_from(&scratch_i.t(), &ket[iset], T::one(), T::zero());
                    if TypeId::of::<T>() == TypeId::of::<O>() {
                        // unsafe cast
                        let mut out_i = unsafe { core::mem::transmute::<TsrMut<O>, TsrMut<T>>(out_i) };
                        out_i.matmul_from(&scratch_i.t(), &ket[iset], T::one(), T::zero());
                    } else {
                        let mut scratch_out_buf = scratch_out_pool.get();
                        let mut scratch_out = rt::asarray((&mut scratch_out_buf, [nb, na].f(), j3c.device()));
                        scratch_out.matmul_from(&scratch_i.t(), &ket[iset], T::one(), T::zero());
                        out_i.assign(&scratch_out.mapv(&cast));
                        scratch_out_pool.put(scratch_out_buf);
                    };
                });
            } else {
                // contract ket first
                // - `j3c` uvP, `ket` va -> `scratch` uPa, iter by P
                let scratch = rt::asarray((&mut scratch_buf, [nao, nb, na].f(), j3c.device()));
                (0..nb).into_par_iter().for_each(|b| {
                    let scratch_b = scratch.i((.., b, ..));
                    let mut scratch_b = unsafe { scratch_b.force_mut() };
                    scratch_b.matmul_from(&j3c_batch.i((.., .., b)), &ket[iset], T::one(), T::zero());
                });
                // - `scratch` uPa, `bra` ui -> `out` Pia, iter by a
                (0..na).into_par_iter().for_each(|a| {
                    let scratch_a = scratch.i((.., .., a));
                    let out_a = out_iset.i((.., .., a));
                    let mut out_a = unsafe { out_a.force_mut() };
                    // if same type
                    // out_a.matmul_from(&scratch_a.t(), &bra[iset], T::one(), T::zero());
                    if TypeId::of::<T>() == TypeId::of::<O>() {
                        // unsafe cast
                        let mut out_a = unsafe { core::mem::transmute::<TsrMut<O>, TsrMut<T>>(out_a) };
                        out_a.matmul_from(&scratch_a.t(), &bra[iset], T::one(), T::zero());
                    } else {
                        let mut scratch_out_buf = scratch_out_pool.get();
                        let mut scratch_out = rt::asarray((&mut scratch_out_buf, [nb, ni].f(), j3c.device()));
                        scratch_out.matmul_from(&scratch_a.t(), &bra[iset], T::one(), T::zero());
                        out_a.assign(&scratch_out.mapv(&cast));
                        scratch_out_pool.put(scratch_out_buf);
                    };
                });
            }
        }
    }
}

pub fn get_ao2mo_s2ij_to_s1_trans<T, O>(
    j3c: TsrView<T>,
    uplo: FlagUpLo,
    bra: &[TsrView<T>],
    ket: &[TsrView<T>],
    nbatch: Option<usize>,
    cast: impl Fn(T) -> O + Send + Sync,
) -> Vec<Tsr<O>>
where
    T: BlasFloat + 'static,
    O: BlasFloat + 'static,
{
    let mut output = vec![];
    for iset in 0..bra.len() {
        let ni = bra[iset].shape()[1];
        let na = ket[iset].shape()[1];
        let shape_j3c_remain = j3c.shape()[1..].to_vec();
        let shape_iset = shape_j3c_remain.into_iter().chain([ni, na].into_iter()).collect_vec();
        let out_iset = rt::zeros((shape_iset.f(), j3c.device()));
        output.push(out_iset);
    }
    let output_mut = output.iter_mut().map(|t| t.view_mut()).collect_vec();
    get_ao2mo_s2ij_to_s1_trans_with_output(j3c, uplo, bra, ket, nbatch, &output_mut, cast);
    output
}

/* #endregion */
