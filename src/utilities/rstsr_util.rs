//! rest_tensor to RSTSR interchange.

use rstsr::prelude::*;
use rayon::prelude::*;
use rstsr_core::{prelude_dev::OpAssignAPI, storage::creation::DeviceCreationAnyAPI};
use tensors::{BasicMatrix, MatrixFull};

pub type Tsr<T = f64> = Tensor<T, DeviceBLAS, IxD>;
pub type TsrView<'a, T = f64> = TensorView<'a, T, DeviceBLAS, IxD>;
pub type TsrMut<'a, T = f64> = TensorMut<'a, T, DeviceBLAS, IxD>;
pub type TsrCow<'a, T = f64> = TensorCow<'a, T, DeviceBLAS, IxD>;

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
        let device = DeviceBLAS::default();
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

/* #endregion */
