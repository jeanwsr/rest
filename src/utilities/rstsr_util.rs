//! rest_tensor to RSTSR interchange.

use rstsr::prelude::*;
use rayon::prelude::*;
use rstsr_core::{prelude_dev::OpAssignAPI, storage::creation::DeviceCreationAnyAPI};
use tensors::{BasicMatrix, MatrixFull};

pub type Tsr<T = f64> = Tensor<T, DeviceBLAS, IxD>;
pub type TsrView<'a, T = f64> = TensorView<'a, T, DeviceBLAS, IxD>;
pub type TsrMut<'a, T = f64> = TensorMut<'a, T, DeviceBLAS, IxD>;
pub type TsrCow<'a, T = f64> = TensorCow<'a, T, DeviceBLAS, IxD>;

/// The process-wide RSTSR device.  Use this instead of `DeviceBLAS::default()`.
///
/// `DeviceBLAS::default()` is **not** free.  rstsr's `DeviceCpuRayon::generate_pool` builds a
/// brand-new rayon `ThreadPool` and spawns `rayon::current_num_threads()` workers on *every*
/// call, so it costs `O(pool)` thread creations and keeps their stacks and TLS alive for as
/// long as the device is.  REST called it per grid sub-block (`eval_xc_eff`, 1400 times per
/// v_xc sweep on polyene n=7), per AO evaluation (`NIMatmul::prepare_ao`), per Krylov matvec
/// conversion (`to_rstsr`) and about forty other places -- thousands of times per gradient
/// card.  With 40 rayon workers each holding a 40-thread device pool alive at once that is
/// ~1600 threads in the process.  Measured on the n=7 gradient card, the `[vxc timing] xc`
/// figure -- which is exactly `eval_xc_eff` -- grew from **19.6 s of CPU at 16 threads to
/// 371.1 s at 40**, and 10.2 s after this became a single shared instance.
///
/// The device is only an `Arc<ThreadPool>` plus a layout flag, and rstsr's `same_device`
/// compares the pool's *current* thread count and the default order, not the identity, so one
/// lazily created instance is equivalent and interoperates with anything built from
/// `DeviceBLAS::default()`.
///
/// The pool size stays `rayon::current_num_threads()`: rstsr's *broadcasted* contractions run
/// on this pool, and `with_num_threads` also asks OpenBLAS for that many threads around every
/// `%` issued from a serial stretch.  Bounding the size here was tried and does **not** work
/// -- a device whose thread count differs from the default one trips rstsr's `DeviceMismatch`
/// check in `op_binary_arithmetic` wherever a tensor meets one built elsewhere.  The
/// serial-BLAS oversubscription it would have fixed is handled at the BLAS symbol instead;
/// see `rest_tensors::matrix::matrix_blas_lapack`.
pub fn device() -> DeviceBLAS {
    static DEV: std::sync::OnceLock<DeviceBLAS> = std::sync::OnceLock::new();
    DEV.get_or_init(DeviceBLAS::default).clone()
}

/* #region interchange between rstsr and rest_tensor */

// In REST, we always use DeviceBLAS as backend in most cases.

pub trait RestTensorToRstsrTsrAPI<T> {
    fn to_rstsr(&self, device: &DeviceBLAS) -> Tsr<T>;
}

pub trait RestTensorToRstsrViewAPI<T> {
    fn to_rstsr_view(&self, device: &DeviceBLAS) -> TsrView<'_, T>;
}

pub trait RestTensorToRstsrMutAPI<T> {
    fn to_rstsr_mut(&mut self, device: &DeviceBLAS) -> TsrMut<'_, T>;
}

pub trait RestTensorIntoRstsrTsrAPI<T> {
    fn into_rstsr(self, device: &DeviceBLAS) -> Tsr<T>;
}

pub fn layout_from_rest_tensor_matrix<B, T>(matr: &B) -> Layout<IxD>
where
    T: Clone,
    for<'a> B: BasicMatrix<'a, T>,
{
    let shape = matr.size().to_vec();
    let stride = matr.indicing().map(|x| x as isize).to_vec();
    let offset = 0;
    Layout::new(shape, stride, offset).unwrap()
}

impl<T> RestTensorToRstsrTsrAPI<T> for MatrixFull<T>
where
    T: Clone,
{
    fn to_rstsr(&self, device: &DeviceBLAS) -> Tsr<T> {
        let layout = layout_from_rest_tensor_matrix(self);
        let upper_bound = layout.bounds_index().unwrap().1;

        let data_ref = self.data_ref().unwrap();
        let vec = data_ref[..upper_bound].to_vec();
        let device = crate::utilities::rstsr_util::device();
        rt::asarray((vec, layout, &device))
    }
}

impl<T> RestTensorToRstsrViewAPI<T> for MatrixFull<T>
where
    T: Clone,
{
    fn to_rstsr_view(&self, device: &DeviceBLAS) -> TsrView<'_, T> {
        let layout = layout_from_rest_tensor_matrix(self);
        let data_ref = self.data_ref().unwrap();
        rt::asarray((data_ref, layout, device))
    }
}

impl<T> RestTensorToRstsrMutAPI<T> for MatrixFull<T>
where
    T: Clone,
{
    fn to_rstsr_mut(&mut self, device: &DeviceBLAS) -> TsrMut<'_, T> {
        let layout = layout_from_rest_tensor_matrix(self);
        let data_ref_mut = self.data_ref_mut().unwrap();
        rt::asarray((data_ref_mut, layout, device))
    }
}

impl<T> RestTensorToRstsrTsrAPI<T> for &[&MatrixFull<T>]
where
    T: Clone,
    DeviceBLAS: DeviceAPI<T> + OpAssignAPI<T, IxD> + DeviceCreationAnyAPI<T>,
{
    fn to_rstsr(&self, device: &DeviceBLAS) -> Tsr<T> {
        if self.len() == 0 {
            panic!("Empty slice cannot be converted to RSTSR Tensor.");
        }
        let layout_single = layout_from_rest_tensor_matrix(self[0]);
        let nset = self.len();
        let mut layout_shape = layout_single.shape().clone();
        let mut layout_stride = layout_single.stride().clone();
        layout_shape.push(nset);
        layout_stride.push(layout_single.size() as isize);
        let layout = Layout::new(layout_shape, layout_stride, 0).unwrap();

        let mut tsr = unsafe { rt::empty((layout, device)) };
        for (iset, matr) in self.iter().enumerate() {
            let mut slc = tsr.i_mut((.., .., iset));
            slc.assign(&matr.to_rstsr_view(device));
        }
        tsr
    }
}

impl<T> RestTensorToRstsrTsrAPI<T> for &[MatrixFull<T>]
where
    T: Clone,
    DeviceBLAS: DeviceAPI<T> + OpAssignAPI<T, IxD> + DeviceCreationAnyAPI<T>,
{
    fn to_rstsr(&self, device: &DeviceBLAS) -> Tsr<T> {
        let slice_of_refs: Vec<&MatrixFull<T>> = self.iter().collect();
        slice_of_refs.as_slice().to_rstsr(device)
    }
}

impl<T> RestTensorToRstsrTsrAPI<T> for &[Vec<T>]
where
    T: Clone,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>> + OpAssignAPI<T, IxD> + DeviceCreationAnyAPI<T>,
{
    fn to_rstsr(&self, device: &DeviceBLAS) -> Tsr<T> {
        if self.len() == 0 {
            panic!("Empty slice cannot be converted to RSTSR Tensor.");
        }
        let n = self[0].len();
        let nset = self.len();
        let layout = vec![n, nset].f();

        let mut tsr = unsafe { rt::empty((layout, device)) };
        for (iset, vec) in self.iter().enumerate() {
            let mut slc = tsr.i_mut((.., iset));
            let vec_tsr: TsrView<T> = rt::asarray((vec, device));
            slc.assign(&vec_tsr);
        }
        tsr
    }
}

impl<T> RestTensorIntoRstsrTsrAPI<T> for Vec<T>
where
    T: Clone,
{
    fn into_rstsr(self, device: &DeviceBLAS) -> Tsr<T> {
        let n = self.len();
        let layout = Layout::new(vec![n], vec![1], 0).unwrap();
        rt::asarray((self, layout, device))
    }
}

impl<T> RestTensorToRstsrTsrAPI<T> for &Vec<T>
where
    T: Clone,
{
    fn to_rstsr(&self, device: &DeviceBLAS) -> Tsr<T> {
        let n = self.len();
        let layout = Layout::new(vec![n], vec![1], 0).unwrap();
        rt::asarray((self.to_vec(), layout, device))
    }
}

/* #endregion */

/* #region general utilities */

/// Fingerprint for f64 tensor.
///
/// This function corresponds to `pyscf.lib.fingerprint`, but will be column-major.
pub fn fingerprint_f64(tsr: TsrView<f64>) -> f64 {
    let tsr = tsr.reshape(-1);
    tsr.iter()
        .into_par_iter()
        .enumerate()
        .fold(|| 0.0_f64, |acc, (i, &x)| acc + x * (i as f64).cos())
        .reduce(|| 0.0_f64, |a, b| a + b)
}

/// Check if two tensors are exactly identical (shape and value).
///
/// This function is used for the cache-invalidation checks of `make_`-style cached evaluations:
/// the cached result is reused only when the new inputs match the ones the cache was evaluated
/// with.
pub fn is_same_tensor(a: TsrView, b: TsrView) -> bool {
    a.shape() == b.shape() && rt::allclose(&a, &b, (0.0, 0.0))
}

/* #endregion */
