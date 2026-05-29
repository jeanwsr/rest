use super::prelude_dev::*;
use super::pure_decompose::*;
use serde::{Deserialize, Serialize};
use serde_inline_default::serde_inline_default;

pub const J2C_THRESH: f64 = 1e-13;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum J2CDecompPolicy {
    /// Cholesky decomposition (give upper/lower triangular decomposition of matrix).
    #[serde(alias = "cholesky", alias = "cd")]
    Cd,
    /// Eigen decomposition (give symmetric decomposition of matrix).
    #[serde(alias = "eigen", alias = "eig", alias = "eigenvalue")]
    Eig,
}

/// Policy for 2c-2e ERI (j2c) decomposition.
///
/// - `Cd`: Cholesky decomposition
///   - None: no threshold, fails if j2c is not positive-definite
///   - threshold: if the diagonal elements of the Cholesky factor are smaller than the threshold,
///     will make matrix to be sufficiently positive-definite with the given threshold, then perform
///     Cholesky decomposition again.
/// - `Eig`: Eigen decomposition, make matrix power -1/2 by strict way with eigenvalues that larger
///   than given threshold, a more orthogonal but costly way than Cholesky decomposition.
#[serde_inline_default]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct J2CDecompOption {
    /// The policy for 2c-2e ERI decomposition. Default to `Eig`.
    #[serde_inline_default(J2CDecompPolicy::Eig)]
    pub policy: J2CDecompPolicy,
    /// The threshold for 2c-2e ERI decomposition. Default to `1e-13`.
    #[serde_inline_default(Some(J2C_THRESH))]
    pub threshold: Option<f64>,
    /// The flag indicating whether the Cholesky factor is upper or lower triangular. Default to `Upper`.
    ///
    /// This is developer option. In most cases, col-major uses upper triangular.
    /// Lower triangular is only for debug and testing purposes.
    ///
    /// This field is only used for Cholesky decomposition, and will be ignored for eigen decomposition.
    #[serde_inline_default(Upper)]
    pub uplo: FlagUpLo,
}

impl Default for J2CDecompOption {
    fn default() -> Self {
        J2CDecompOption { policy: J2CDecompPolicy::Eig, threshold: Some(J2C_THRESH), uplo: Upper }
    }
}

/// Output of decomposed intermediates for 2c-2e ERI matrix (j2c).
pub enum J2CDecompose {
    /// Cholesky decomposition
    ///
    /// Required fields:
    /// - `j2c_l`: the Cholesky factor of 2c-2e ERI
    /// - `uplo`: the flag indicating whether the Cholesky factor is upper or lower triangular
    ///
    /// Optional field:
    /// - `j2c_l_inv`: the inverse of the Cholesky factor, only for debugging and testing purposes;
    ///   use TRSM with `j2c_l` is recommended
    Cd { j2c_l: Tsr<f64>, uplo: FlagUpLo, j2c_l_inv: Option<Tsr<f64>> },

    /// Eigen decomposition
    ///
    /// Required fields:
    /// - `j2c_l_inv`: the -1/2 power of the 2c-2e ERI matrix
    /// - `threshold`: the threshold for eigenvalue decomposition
    ///
    /// Optional field:
    /// - `j2c_e`: the eigenvalues of the 2c-2e ERI matrix
    /// - `j2c_v`: the eigenvectors of the 2c-2e ERI matrix
    Eig { j2c_l_inv: Tsr<f64>, j2c_e: Option<Tsr<f64>>, j2c_v: Option<Tsr<f64>> },
}

/// Generate decomposed 3c-2e ERI (cderi/rimatr).
///
/// The output matrix will be of shape (nao_tp, naux).
/// - `nao_tp` is the number of unique basis pairs, which is `nao * (nao + 1) / 2` where `nao` is
///   the number of atomic orbitals (basis functions).
///
/// Note the output will depends on different j2c_decomp policy.
///
/// # Notes on memory usage
///
/// This function returns a very large matrix `cderi`/`rimatr` (nao_tp, naux), or of approx size
///
/// > 0.5 * nao^2 * naux * 8 bytes
///
/// Except for this large matrix, the extra memory should not be large:
///
/// - j2c decomp requires about `10 * naux^2` f64 elements, depending on whether divide-and-conquer
///   eigh is called.
/// - by Cholesky way, solve j3c to cderi does not require extra memory
/// - by eigen way, the matmul will cost at most `2 * naux^2` or 4% of the final cderi matrix.
pub fn generate_rimatr_bare(mol_obj: &Molecule, omega: Option<f64>) -> MatrixFull<f64> {
    let mut mol = util::get_cint_mol(mol_obj);
    let mut aux = util::get_cint_aux(mol_obj);

    if let Some(omega) = omega {
        mol.set_omega(omega);
        aux.set_omega(omega);
    }

    let j2c_decomp_option = mol_obj.ctrl.j2c_decomp;

    let device = DeviceBLAS::default();

    let j2c_decomp = get_j2c_decomp(&aux, &device, j2c_decomp_option);
    let j3c = {
        let (out, shape) = CInt::integrate_cross("int3c2e", [&mol, &mol, &aux], "s2ij", None).into();
        rt::asarray((out, shape.f(), &device))
    };
    let cderi = get_solved_j3c(j3c, &j2c_decomp, false);

    let shape = cderi.shape().to_vec().try_into().unwrap();
    MatrixFull::from_vec(shape, cderi.into_shape(-1).into_vec()).unwrap()
}

/// Generate `basbas2baspar``, `baspar2basbas` of REST for upper triangular indexing of basis pairs.
///
/// These code are directly extracted from previous code of IYZ.
pub fn generate_baspar(nao: usize) -> (MatrixFull<usize>, Vec<[usize; 2]>) {
    let nao_tp = nao * (nao + 1) / 2;
    let mut basbas2baspar = MatrixFull::new([nao, nao], 0_usize);
    let mut baspar2basbas = vec![[0_usize; 2]; nao_tp];
    basbas2baspar.iter_columns_full_mut().enumerate().for_each(|(j, slice_x)| {
        slice_x.iter_mut().enumerate().for_each(|(i, map_v)| {
            let baspar_ind = if i < j { (j + 1) * j / 2 + i } else { (i + 1) * i / 2 + j };
            *map_v = baspar_ind;
            baspar2basbas[baspar_ind] = [i, j];
        });
    });
    (basbas2baspar, baspar2basbas)
}
