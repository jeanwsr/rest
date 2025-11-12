use crate::molecule_io::Molecule;
use crate::utilities::memory_batch::*;
use crate::utilities::rstsr_util::*;
use rayon::prelude::*;
use rest_libcint::prelude::*;
use rstsr::prelude::*;
use rstsr_core::prelude_dev::uninitialized_vec;
use tensors::{MatrixFull, MatrixUpper};

type Tsr<T> = Tensor<T, DeviceBLAS, IxD>;
type TsrView<'a, T> = TensorView<'a, T, DeviceBLAS, IxD>;
type TsrMut<'a, T> = TensorMut<'a, T, DeviceBLAS, IxD>;

/* #region ri-vj incore */

/// Generate Coulomb (J) matrix using RI incore method (cholesky decomposed ERI in-memory).
///
/// This function is low-level implementation, using RSTSR tensors as input and output.
///
/// # Parameters
///
/// - `cderi`: [`TsrView<f64>`]
///
///   - Cholesky decomposed 3c-2e ERI in shape (nao_tp, naux), stored in f-contiguous order.
///   - `nao_tp = nao * (nao + 1) / 2`, where `nao` is number of atomic orbitals.
///   - This tensor is assumed to be in AO basis.
///
/// - `dms`: [`TsrView<f64>`]
///
///   - Density matrices in shape (nao, nao, nset), stored in f-contiguous order.
///   - We will check `ndim == 3`. Please expand dimension if necessary, especially for RHF case
///     where `nset = 1`.
///   - This matrix is assumed to be in AO basis.
///   - This matrix is assumed to be symmetric. We will not perform symmetry check or symmetrize
///     operation.
///
/// # Returns
///
/// - [`Tsr<f64>`]
///
///   - Coulomb (J) matrices in shape (nao, nao, nset), stored in f-contiguous order.
///   - J matrices are symmetric by definition in real arithmetic.
///
/// # Formula and Algorithm
///
/// $$
/// J_{\mu \nu}[\mathbf{D}^\mathbb{A}] = \sum_{P} \sum_{\kappa \lambda} Y_{\mu \nu, P}
/// Y_{\kappa \lambda, P} D_{\kappa \lambda}^\mathbb{A}
/// $$
///
/// - AO indices ($\mu \nu$, $\kappa, \lambda$) are in packed upper-triangular format.
/// - This function implements the above formula directly using matrix multiplications.
///
/// There is virtually no memory requirement other than input and output tensors.
pub fn generate_vj_ri_incore_with_rstsr(cderi: TsrView<f64>, dms: TsrView<f64>) -> Tsr<f64> {
    assert_eq!(dms.ndim(), 3, "DM must have 3 dimensions");
    assert_eq!(cderi.ndim(), 2, "Cholesky ERI must have 2 dimensions");

    // get shapes
    let nset = dms.shape()[2];
    let nao = dms.shape()[0];
    let nao_tp = (nao + 1) * nao / 2;

    // shape check
    assert_eq!(cderi.shape()[0], nao_tp, "Cholesky ERI must have shape (nao_tp, naux)");

    // pack density matrix with upper-triangular, diagonal doubled
    let mut dms_tp = rt::zeros(([nao_tp, nset].f(), dms.device()));
    for iset in 0..nset {
        let dm = dms.i((.., .., iset));
        let dm_diag = dm.diagonal(None);
        let mut dm = 2.0_f64 * &dm;
        dm.diagonal_mut(None).assign(&dm_diag);
        let dm_tp = dm.pack_triu();
        dms_tp.i_mut((.., iset)).assign(dm_tp);
    }

    // generate j contribution
    let scr_j = cderi.t() % &dms_tp;
    let js_tp = &cderi % &scr_j;

    // returns symmetrized part
    js_tp.unpack_tri(Upper, FlagSymm::Sy)
}

/* #endregion ri-vj incore */

/* #region ri-vj direct */

/// Generate Coulomb (J) matrix using RI direct method (on-the-fly).
///
/// This function is low-level implementation, using RSTSR tensors as input and output. For
/// high-level interface (using rest_tensors as input and output), please refer to
/// [`generate_vj_ri_direct`].
///
/// # Parameters
///
/// - `dms`: [`TsrView<f64>`]
///
///   - Density matrices in shape (nao, nao, nset), stored in f-contiguous order.
///   - We will check `ndim == 3`. Please expand dimension if necessary, especially for RHF case
///     where `nset = 1`.
///   - This matrix is assumed to be in AO basis.
///   - This matrix is assumed to be symmetric. We will not perform symmetry check or symmetrize
///     operation.
///
/// - `mol_obj`: [`Molecule`]
///
///   - Please make sure auxiliary basis is assigned to this molecule object.
///
/// - `batch_size`: `usize`
///
///   - Block size for auxiliary basis partitioning. This value controls memory usage.
///   - To give a proper value, please refer to memory estimation function
///     [`mem_estimate_vj_ri_direct`].
///
/// # Returns
///
/// - [`Tsr<f64>`]
///
///   - Coulomb (J) matrices in shape (nao, nao, nset), stored in f-contiguous order.
///   - J matrices are symmetric by definition in real arithmetic.
///
/// # Formula and Algorithm
///
/// $$
/// \begin{align*}
/// \mathscr{T}_P^\text{1} [\mathbf{D}^\mathbb{A}] &= \sum_{\kappa \lambda} g_{\kappa \lambda, P}
/// D_{\kappa \lambda}^\mathbb{A} \tag{eq.1} \\
/// \mathscr{T}_P^\text{2} [\mathbf{D}^\mathbb{A}] &= \sum_{Q} (\mathbf{J}^{-1})_{PQ}
/// \mathscr{T}_Q^\text{1} [\mathbf{D}^\mathbb{A}] \tag{eq.2} \\
/// J_{\mu \nu} [\mathbf{D}^\mathbb{A}] &= \sum_{P} g_{\mu \nu, P} \mathscr{T}_P^\text{2}
/// [\mathbf{D}^\mathbb{A}] \tag{eq.3}
/// \end{align*}
/// $$
///
/// - AO indices ($\mu \nu$, $\kappa, \lambda$) are in packed upper-triangular format.
/// - Auxiliary basis are batched, sparately in (eq.1) and (eq.3).
/// - (eq.2) is solved using general linear solver.
///
/// Fixed memory requirement:
///
/// - Storage of $\mathscr{T}_P^\text{1} [\mathbf{D}^\mathbb{A}]$ and $\mathscr{T}_P^\text{2}
///   [\mathbf{D}^\mathbb{A}]$, which costs (naux * nset * 2).
/// - Storage of decomposed or inversed 2c-2e ERI $J_{PQ}$, which costs approximately (naux * naux).
///
/// Batched memory requirement (controlled by `batch_size`):
///
/// - Storage of 3c-2e ERI $g_{\kappa \lambda, P}$ for a batch of auxiliary basis, which costs
///   (nao_tp * batch_size).
pub fn generate_vj_ri_direct_with_rstsr(dms: TsrView<f64>, mol_obj: &Molecule, batch_size: usize) -> Tsr<f64> {
    // dm shape: (nao, nao, nset) in f-contig
    assert_eq!(dms.ndim(), 3, "DM must have 3 dimensions");

    let mut mol = mol_obj.initialize_cint(true);
    let mut aux = mol_obj.make_auxmol_fake().initialize_cint(false);

    // get shapes
    let nset = dms.shape()[2];
    let nao = dms.shape()[0];
    let nao_tp = (nao + 1) * nao / 2;
    let naux = aux.cgto_loc().last().unwrap().clone();
    let device = dms.device().clone();
    let nbas = mol_obj.cint_bas.len() as i32;
    let nbas_aux = mol_obj.cint_aux_bas.len() as i32;

    // get partition
    let aux_loc = &mol.cgto_loc()[(nbas as usize)..];
    let partition = blocksize_partition(&aux_loc, batch_size);

    // int2c2e (may be stored in SCF iteration, generate on-the-fly costs some but not that much)
    let tsr_int2c2e = {
        let shls_slice = [[nbas, nbas + nbas_aux], [nbas, nbas + nbas_aux]];
        let (out, shape) = mol.integral_s1::<int2c2e>(Some(&shls_slice));
        rt::asarray((out, shape.f(), &device))
    };

    // pack density matrix with upper-triangular, diagonal doubled
    let mut dms_tp = rt::zeros(([nao_tp, nset].f(), &device));
    for iset in 0..nset {
        let dm = dms.i((.., .., iset));
        let dm_diag = dm.diagonal(None);
        let mut dm = 2.0_f64 * &dm;
        dm.diagonal_mut(None).assign(&dm_diag);
        let dm_tp = dm.pack_triu();
        dms_tp.i_mut((.., iset)).assign(dm_tp);
    }

    // generate scr_j
    let mut scr_j = rt::zeros(([naux, nset].f(), &device));
    let mut idx_ao = 0;
    for &[shl0, shl1] in &partition {
        let nbatch_ao = aux_loc[shl1] - aux_loc[shl0];
        let shls_slice = [[0, nbas], [0, nbas], [nbas + shl0 as i32, nbas + shl1 as i32]];
        let int3c2e_batch = {
            let (out, shape) = mol.integral_s2ij::<int3c2e>(Some(&shls_slice));
            rt::asarray((out, shape.f(), &device))
        };
        let slc = slice!(idx_ao, idx_ao + nbatch_ao);
        scr_j.i_mut(slc).matmul_from(&int3c2e_batch.t(), &dms_tp, 1.0, 0.0);
        idx_ao += nbatch_ao;
    }

    // solve scr_j
    let scr_js = rt::linalg::solve_general((tsr_int2c2e, scr_j));

    // generate j contribution
    let mut js_tp = rt::zeros(([nao_tp, nset].f(), &device));
    let mut idx_ao = 0;
    for &[shl0, shl1] in &partition {
        let nbatch_ao = aux_loc[shl1] - aux_loc[shl0];
        let shls_slice = [[0, nbas], [0, nbas], [nbas + shl0 as i32, nbas + shl1 as i32]];
        let int3c2e_batch = {
            let (out, shape) = mol.integral_s2ij::<int3c2e>(Some(&shls_slice));
            rt::asarray((out, shape.f(), &device))
        };
        let slc = slice!(idx_ao, idx_ao + nbatch_ao);
        js_tp.matmul_from(&int3c2e_batch, &scr_js.i(slc), 1.0, 1.0);
        idx_ao += nbatch_ao;
    }

    // returns symmetrized part
    js_tp.unpack_tri(Upper, FlagSymm::Sy)
}

/// Estimate memory requirement for VJ RI direct method.
///
/// This function corresponds to [`generate_vj_ri_direct_with_rstsr`].
///
/// # Parameters
///
/// - `nao`: Number of atomic orbitals.
/// - `naux`: Number of auxiliary basis functions.
/// - `nset`: Number of density matrix sets.
pub const fn mem_estimate_vj_ri_direct(nao: usize, naux: usize, nset: usize) -> MemEstimate {
    let nao_tp = (nao + 1) * nao / 2;
    // int3c2e batch
    let batched = nao_tp;
    // scr_j, scr_js, int2c2e, js_tp / dms_tp, js_tp
    let fixed = naux * nset * 2 + naux * naux + nao * nao * nset + nao_tp * nset;
    let thread = 0;
    MemEstimate { batched, fixed, thread }
}

pub fn generate_vj_ri_direct(dms: &[MatrixFull<f64>], mol_obj: &Molecule, batch_size: usize) -> Vec<MatrixUpper<f64>> {
    // dm shape: (nao, nao, nset) in f-contig
    let device = DeviceBLAS::default();
    let dms_rstsr = dms.to_rstsr(&device);

    let nao = dms_rstsr.shape()[0];
    let nset = dms_rstsr.shape()[2];
    let nao_tp = (nao + 1) * nao / 2;

    let js_rstsr = generate_vj_ri_direct_with_rstsr(dms_rstsr.view(), mol_obj, batch_size);

    // Tsr -> Vec<MatrixUpper>
    let mut js = vec![];
    for iset in 0..nset {
        let j = js_rstsr.i((.., .., iset)).pack_tri(Upper).into_vec();
        js.push(unsafe { MatrixUpper::from_vec_unchecked(nao_tp, j) });
    }
    js
}

/* #endregion ri-vj direct */

/* #region ri-vk incore */

/// Generate Exchange (K) matrix using RI incore method with MO coefficients and occupations.
///
/// This function is low-level implementation, using RSTSR tensors as input and output.
///
/// # Parameters
///
/// - `cderi`: [`TsrView<f64>`]
///
///   - Cholesky decomposed 3c-2e ERI in shape (nao_tp, naux), stored in f-contiguous order.
///   - `nao_tp = nao * (nao + 1) / 2`, where `nao` is number of atomic orbitals.
///   - This tensor is assumed to be in AO basis.
///
/// - `mo_coeff`: [`TsrView<f64>`]
///
///   - Molecular orbital coefficients in shape (nao, nmo, nset), stored in f-contiguous order.
///   - We will check `ndim == 3`. Please expand dimension if necessary, especially for RHF case
///     where `nset = 1`.
///
/// - `mo_occ`: [`TsrView<f64>`]
///
///   - Molecular orbital occupations in shape (nmo, nset), stored in f-contiguous order.
///   - We will check `ndim == 2`. Please expand dimension if necessary, especially for RHF case
///     where `nset = 1`.
///
/// - `batch_size`: `usize`
///
///   - Batch size for auxiliary basis partitioning. This value controls memory usage.
///
/// # Returns
///
/// - [`Tsr<f64>`]
///
///   - Exchange (K) matrices in shape (nao, nao, nset), stored in f-contiguous order.
///   - K matrices are symmetric by definition in real arithmetic.
///
/// # Formula and Algorithm
///
/// $$
/// \begin{align*}
/// Y_{\mu i, P}^\mathbb{A} &= \sum_{\nu} \sqrt{n_i} C_{\nu i}^\mathbb{A} Y_{\mu \nu, P} \\
/// K_{\mu \nu} [\mathbf{D}^\mathbb{A}] &= \sum_{P i} Y_{\mu i, P}^\mathbb{A} Y_{\nu i,
/// P}^\mathbb{A} \end{align*}
/// $$
///
/// Fixed memory requirement:
///
/// - Storage of output K matrix, which costs (nao * nao * nset).
///
/// Batched memory requirement (controlled by `batch_size`):
///
/// - Storage of half-transformed integrals $Y_{\mu i, P}^\mathbb{A}$, which costs (nao * nocc *
///   batch_size). Note that `nocc` can be different for different sets (spin-channels for example),
///   and this can be determined by the largest `nocc` among all sets.
///
/// Thread memory requirement:
///
/// - Storage of cderi per auxiliary, which costs (nao * nao * nthread).
pub fn generate_vk_ri_incore_coeff_with_rstsr(
    cderi: TsrView<f64>,
    mo_coeff: TsrView<f64>,
    mo_occ: TsrView<f64>,
    batch_size: usize,
) -> Tsr<f64> {
    assert_eq!(mo_coeff.ndim(), 3, "Molecular coefficients must have 3 dimensions");
    assert_eq!(mo_occ.ndim(), 2, "Molecular occupations must have 2 dimensions");
    assert_eq!(cderi.ndim(), 2, "Cholesky ERI must have 2 dimensions");

    // get shapes
    let nao = mo_coeff.shape()[0];
    let nmo = mo_coeff.shape()[1];
    let nset = mo_coeff.shape()[2];
    let naux = cderi.shape()[1];
    let nao_tp = (nao + 1) * nao / 2;
    let device = cderi.device().clone();

    // shape check
    assert_eq!(cderi.shape(), &[nao_tp, naux], "Cholesky ERI must have shape (nao_tp, naux)");
    assert_eq!(mo_occ.shape(), &[nmo, nset], "Molecular occupations must have shape (nmo, nset)");

    // check occupation not less than zero
    let occ_neg = rt::lt(&mo_occ, 0.0).sum();
    if occ_neg > 0 {
        println!("[WARN] in generate_vk_ri_incore_coeff_with_rstsr, negative occupation found: {occ_neg} elements < 0");
    }

    // compress mo_coeff with occupation
    let mut occ_coeff_list = vec![];
    for iset in 0..nset {
        // generate occ_coeff by sqrt(occupation) * coefficient
        let occ_mask = rt::gt(mo_occ.i((.., iset)), f64::EPSILON).into_vec();
        let occ_coeff = mo_coeff.i((.., .., iset)).bool_select(-1, &occ_mask);
        let occ = mo_occ.i((.., iset)).bool_select(-1, &occ_mask).sqrt();
        occ_coeff_list.push(occ_coeff * occ.i((None, ..)));
    }

    // initialize vk as result
    let mut ks = rt::zeros(([nao, nao, nset].f(), &device));

    // process each auxiliary function
    (0..naux).step_by(batch_size).for_each(|iaux| {
        // get and unpack cderi for this auxiliary function
        // cderi_iaux: (nao, nao, nbatch)
        let nbatch = if iaux + batch_size <= naux { batch_size } else { naux - iaux };

        for iset in 0..nset {
            // half-transformed integrals: (nao, nocc, nbatch)
            let occ_coeff = &occ_coeff_list[iset];
            let nocc = occ_coeff.shape()[1];
            let cderi_half = unsafe { rt::empty(([nao, nocc, nbatch].f(), &device)) };
            (0..nbatch).into_par_iter().for_each(|p| {
                let cderi_iaux = cderi.i((.., iaux + p)).unpack_tri(Upper, FlagSymm::Sy);
                let cderi_half_iaux = cderi_half.i((.., .., p));
                let mut cderi_half_iaux = unsafe { cderi_half_iaux.force_mut() };
                cderi_half_iaux.matmul_from(&cderi_iaux, &occ_coeff, 1.0, 0.0);
            });
            // build vk contribution
            let cderi_half = cderi_half.into_shape([nao, nocc * nbatch]);
            ks.i_mut((.., .., iset)).matmul_from(&cderi_half, &cderi_half.t(), 1.0, 1.0);
        }
    });
    ks
}

/// Estimate memory requirement for VK RI incore method with MO coefficients and occupations.
///
/// This function corresponds to [`generate_vk_ri_incore_coeff_with_rstsr`].
///
/// # Parameters
///
/// - `nao`: Number of atomic orbitals.
/// - `naux`: Number of auxiliary basis functions.
/// - `nocc_max`: Maximum number of occupied molecular orbitals among all sets.
/// - `nset`: Number of density matrix sets.
pub fn mem_estimate_vk_ri_incore_coeff(nao: usize, naux: usize, nocc_max: usize, nset: usize) -> MemEstimate {
    // int3c2e batch
    let batched = nao * nocc_max;
    // ks
    let fixed = nao * nao * nset;
    // cderi per auxiliary
    let thread = nao * nao;
    MemEstimate { batched, fixed, thread }
}

/* #endregion ri-vk incore */

/* #region ri-vk semi-incore */

/// Generate Exchange (K) matrix using RI semi-incore method with MO coefficients and occupations.
///
/// Semi-incore here means that 3c-2e ERI are computed on-the-fly in batches, while the
/// half-transformed integrals are stored in memory. In this way, we still need to re-evaluate all
/// 3c-2e ERI every time we compute each component of K matrix, and still require a large amount of
/// DRAM consumption, but less computation effort than fully direct method.
///
/// This function is low-level implementation, using RSTSR tensors as input and output.
///
/// # Parameters
///
/// - `mo_coeff`: [`TsrView<f64>`]
///
///   - Molecular orbital coefficients in shape (nao, nmo, nset), stored in f-contiguous order.
///   - We will check `ndim == 3`. Please expand dimension if necessary, especially for RHF case
///     where `nset = 1`.
///
/// - `mo_occ`: [`TsrView<f64>`]
///
///   - Molecular orbital occupations in shape (nmo, nset), stored in f-contiguous order.
///   - We will check `ndim == 2`. Please expand dimension if necessary, especially for RHF case
///     where `nset = 1`.
///
/// - `mol_obj`: [`Molecule`]
///
///   - Please make sure auxiliary basis is assigned to this molecule object.
///
/// - `batch_size`: `usize`
///
///   - Batch size for auxiliary basis partitioning. This value controls memory usage.
///
/// # Returns
///
/// - [`Tsr<f64>`]
///
///   - Exchange (K) matrices in shape (nao, nao, nset), stored in f-contiguous order.
///   - K matrices are symmetric by definition in real arithmetic.
///
/// # Formula and Algorithm
///
/// $$
/// \begin{align*}
/// g_{\mu i, P}^\mathbb{A} &= \sum_{\nu} \sqrt{n_i} C_{\nu i}^\mathbb{A} g_{\mu \nu, P} \\
/// Y_{\mu i, P}^\mathbb{A} &= \sum_{Q} (\mathbf{L}^{-1})_{PQ} g_{\mu i, Q}^\mathbb{A} \\
/// K_{\mu \nu} [\mathbf{D}^\mathbb{A}] &= \sum_{P i} Y_{\mu i, P}^\mathbb{A} Y_{\nu i,
/// P}^\mathbb{A}
/// \end{align*}
/// $$
///
/// Only the first equation is evaluated by batch, while the second and third equations are not
/// batched.
///
/// Fixed memory requirement:
///
/// - Storage of output K matrix, which costs (nao * nao * nset).
/// - Storage of half-transformed integrals $Y_{\mu i, P}^\mathbb{A}$, which costs (nao * nocc_max *
///   naux).
/// - Storage of decomposed 2c-2e ERI $L_{PQ}$, which costs approximately (naux * naux).
///
/// Batched memory requirement (controlled by `batch_size`):
///
/// - Storage of half-transformed integrals $g_{\mu \nu, P}^\mathbb{A}$ for a batch of auxiliary
///   basis, which costs (nao * nao * batch_size).
///
/// Thread memory requirement:
///
/// - Storage of cderi per auxiliary, which costs (nao * nao * nthread).
pub fn generate_vk_ri_semi_incore_coeff_with_rstsr(
    mo_coeff: TsrView<f64>,
    mo_occ: TsrView<f64>,
    mol_obj: &Molecule,
    batch_size: usize,
) -> Tsr<f64> {
    assert_eq!(mo_coeff.ndim(), 3, "Molecular coefficients must have 3 dimensions");
    assert_eq!(mo_occ.ndim(), 2, "Molecular occupations must have 2 dimensions");

    // initialize mol and aux
    let mut mol = mol_obj.initialize_cint(true);
    let mut aux = mol_obj.make_auxmol_fake().initialize_cint(false);

    // get shapes
    let nao = mo_coeff.shape()[0];
    let nmo = mo_coeff.shape()[1];
    let nset = mo_coeff.shape()[2];
    let naux = aux.cgto_loc().last().unwrap().clone();
    let device = mo_coeff.device().clone();
    let nbas = mol_obj.cint_bas.len() as i32;
    let nbas_aux = mol_obj.cint_aux_bas.len() as i32;

    // shape check
    assert_eq!(mo_occ.shape(), &[nmo, nset], "Molecular occupations must have shape (nmo, nset)");

    // check occupation not less than zero
    let occ_neg = rt::lt(&mo_occ, 0.0).sum();
    if occ_neg > 0 {
        println!("[WARN] in generate_vk_ri_incore_coeff_with_rstsr, negative occupation found: {occ_neg} elements < 0");
    }

    // compress mo_coeff with occupation
    let mut occ_coeff_list = vec![];
    for iset in 0..nset {
        // generate occ_coeff by sqrt(occupation) * coefficient
        let occ_mask = rt::gt(mo_occ.i((.., iset)), f64::EPSILON).into_vec();
        let occ_coeff = mo_coeff.i((.., .., iset)).bool_select(-1, &occ_mask);
        let occ = mo_occ.i((.., iset)).bool_select(-1, &occ_mask).sqrt();
        occ_coeff_list.push(occ_coeff * occ.i((None, ..)));
    }
    let nocc_max = occ_coeff_list.iter().map(|x| x.shape()[1]).max().unwrap();

    // get partition
    let aux_loc = &mol.cgto_loc()[(nbas as usize)..];
    let partition = blocksize_partition(&aux_loc, batch_size);

    // initialize vk as result
    let mut ks = rt::zeros(([nao, nao, nset].f(), &device));

    // initialize half-transformed cderi storage
    let mut eri_half_vec = unsafe { uninitialized_vec::<f64>(nao * nocc_max * naux).unwrap() };

    // initialize int2c2e and perform cholesky decomposition
    let tsr_int2c2e = {
        let shls_slice = [[nbas, nbas + nbas_aux], [nbas, nbas + nbas_aux]];
        let (out, shape) = mol.integral_s1::<int2c2e>(Some(&shls_slice));
        rt::asarray((out, shape.f(), &device))
    };
    let tsr_int2c2e_l = rt::linalg::cholesky((tsr_int2c2e, Upper));

    for iset in 0..nset {
        // get half-transformed cderi for this density matrix set
        let occ_coeff = &occ_coeff_list[iset];
        let nocc = occ_coeff.shape()[1];
        let eri_half = rt::asarray((&mut eri_half_vec, [nao, nocc, naux].f(), &device));

        for &[shl0, shl1] in &partition {
            let aux0 = aux_loc[shl0] - nao;
            let aux1 = aux_loc[shl1] - nao;
            let nbatch_aux = aux1 - aux0;
            let shls_slice = [[0, nbas], [0, nbas], [nbas + shl0 as i32, nbas + shl1 as i32]];
            let int3c2e_batch = {
                let (out, shape) = mol.integral_s2ij::<int3c2e>(Some(&shls_slice));
                rt::asarray((out, shape.f(), &device))
            };
            // half-transform
            (0..nbatch_aux).into_par_iter().for_each(|p| {
                let cderi_iaux = int3c2e_batch.i((.., p)).unpack_tri(Upper, FlagSymm::Sy);
                let cderi_half_iaux = eri_half.i((.., .., aux0 + p));
                let mut cderi_half_iaux = unsafe { cderi_half_iaux.force_mut() };
                cderi_half_iaux.matmul_from(&cderi_iaux, &occ_coeff, 1.0, 0.0);
            });
        }

        // solve eri_half
        let eri_half = rt::asarray((&mut eri_half_vec, [nao * nocc, naux].f(), &device)).into_reverse_axes();
        let cderi_half = rt::linalg::solve_triangular((tsr_int2c2e_l.t(), eri_half, Lower));

        // build vk contribution
        let cderi_half = rt::asarray((&mut eri_half_vec, [nao, nocc * naux].f(), &device));
        ks.i_mut((.., .., iset)).matmul_from(&cderi_half, &cderi_half.t(), 1.0, 1.0);
    }
    ks
}

/// Estimate memory requirement for VK RI semi-incore method with MO coefficients and occupations.
///
/// This function corresponds to [`generate_vk_ri_semi_incore_coeff_with_rstsr`].
///
/// # Parameters
///
/// - `nao`: Number of atomic orbitals.
/// - `naux`: Number of auxiliary basis functions.
/// - `nocc_max`: Maximum number of occupied molecular orbitals among all sets.
/// - `nset`: Number of density matrix sets.
pub fn mem_estimate_vk_ri_semi_incore_coeff(nao: usize, naux: usize, nocc_max: usize, nset: usize) -> MemEstimate {
    // int3c2e batch
    let batched = nao * nao;
    // ks, eri_half, int2c2e
    let fixed = nao * nao * nset + nao * nocc_max * naux + nao * nao;
    // cderi per auxiliary
    let thread = nao * nao;
    MemEstimate { batched, fixed, thread }
}

/* #endregion ri-vk semi-incore */

/* #region ri-vk direct coeff */

/// Generate Exchange (K) matrix using RI direct method with MO coefficients and occupations.
///
/// Direct here means that both 3c-2e ERI and half-transformed ERI are computed on-the-fly in
/// batches. This will cost less memory, but more computation effort.
///   
/// This function is low-level implementation, using RSTSR tensors as input and output.
///
/// # Parameters
///
/// - `mo_coeff`: [`TsrView<f64>`]
///
///   - Molecular orbital coefficients in shape (nao, nmo, nset), stored in f-contiguous order.
///   - We will check `ndim == 3`. Please expand dimension if necessary, especially for RHF case
///     where `nset = 1`.
///
/// - `mo_occ`: [`TsrView<f64>`]
///
///   - Molecular orbital occupations in shape (nmo, nset), stored in f-contiguous order.
///   - We will check `ndim == 2`. Please expand dimension if necessary, especially for RHF case
///     where `nset = 1`.
///
/// - `mol_obj`: [`Molecule`]
///
///   - Please make sure auxiliary basis is assigned to this molecule object.
///
/// - `batch_size`: `usize`
///
///   - Batch size for auxiliary basis partitioning. This value controls memory usage.
///
/// # Returns
///
/// - [`Tsr<f64>`]
///
///   - Exchange (K) matrices in shape (nao, nao, nset), stored in f-contiguous order.
///   - K matrices are symmetric by definition in real arithmetic.
///
/// # Formula and Algorithm
///
/// This is similar to the semi-incore method, but batch occurs on the atomic orbital index $\mu$
/// and $\nu$, instead of auxiliary index $P$. This function will generate k contribution to fock
/// batch-by-batch.
///
/// Fixed memory requirement:
///
/// - Storage of output K matrix, which costs (nao * nao * nset).
/// - Storage of decomposed 2c-2e ERI $L_{PQ}$, which costs approximately (naux * naux).
///
/// Batched memory requirement (controlled by `batch_size`):
///
/// - Storage of 3c-2e ERI $g_{\mu \kappa, P}$ for a batch of atomic orbitals, which costs
///   (nbatch_ao * nao * naux).
/// - Storage of half-transformed integrals $Y_{\mu i, P}^\mathbb{A}$ for a batch of atomic
///   orbitals, which costs (nbatch_ao * nocc_max * naux * nset).
/// - Storage of another half-transformed integrals $Y_{\nu j, P}^\mathbb{A}$ for a batch of atomic
///   orbitals, but only for one set at a time, which costs (nbatch_ao * nocc_max * naux).
///
/// Thread memory requirement:
///
/// - Storage of cderi per auxiliary, which costs (nao * nao * nthread).
pub fn generate_vk_ri_direct_coeff_with_rstsr(
    mo_coeff: TsrView<f64>,
    mo_occ: TsrView<f64>,
    mol_obj: &Molecule,
    batch_size: usize,
) -> Tsr<f64> {
    assert_eq!(mo_coeff.ndim(), 3, "Molecular coefficients must have 3 dimensions");
    assert_eq!(mo_occ.ndim(), 2, "Molecular occupations must have 2 dimensions");

    // initialize mol and aux
    let mut mol = mol_obj.initialize_cint(true);
    let mut aux = mol_obj.make_auxmol_fake().initialize_cint(false);

    // get shapes
    let nao = mo_coeff.shape()[0];
    let nmo = mo_coeff.shape()[1];
    let nset = mo_coeff.shape()[2];
    let naux = aux.cgto_loc().last().unwrap().clone();
    let device = mo_coeff.device().clone();
    let nbas = mol_obj.cint_bas.len() as i32;
    let nbas_aux = mol_obj.cint_aux_bas.len() as i32;

    // shape check
    assert_eq!(mo_occ.shape(), &[nmo, nset], "Molecular occupations must have shape (nmo, nset)");

    // check occupation not less than zero
    let occ_neg = rt::lt(&mo_occ, 0.0).sum();
    if occ_neg > 0 {
        println!("[WARN] in generate_vk_ri_incore_coeff_with_rstsr, negative occupation found: {occ_neg} elements < 0");
    }

    // compress mo_coeff with occupation
    let mut occ_coeff_list = vec![];
    for iset in 0..nset {
        // generate occ_coeff by sqrt(occupation) * coefficient
        let occ_mask = rt::gt(mo_occ.i((.., iset)), f64::EPSILON).into_vec();
        let occ_coeff = mo_coeff.i((.., .., iset)).bool_select(-1, &occ_mask);
        let occ = mo_occ.i((.., iset)).bool_select(-1, &occ_mask).sqrt();
        occ_coeff_list.push(occ_coeff * occ.i((None, ..)));
    }
    let nocc_max = occ_coeff_list.iter().map(|x| x.shape()[1]).max().unwrap();

    // get partition
    let ao_loc = &mol.cgto_loc()[..=(nbas as usize)];
    let partition = blocksize_partition(&ao_loc, batch_size);

    // initialize vk as result
    let mut ks = rt::zeros(([nao, nao, nset].f(), &device));

    // initialize int2c2e and perform cholesky decomposition
    let tsr_int2c2e = {
        let shls_slice = [[nbas, nbas + nbas_aux], [nbas, nbas + nbas_aux]];
        let (out, shape) = mol.integral_s1::<int2c2e>(Some(&shls_slice));
        rt::asarray((out, shape.f(), &device))
    };
    let tsr_int2c2e_l = rt::linalg::cholesky((tsr_int2c2e, Upper));

    for (batch_i, &[shl0_i, shl1_i]) in partition.iter().enumerate() {
        let ao0_i = ao_loc[shl0_i];
        let ao1_i = ao_loc[shl1_i];
        let nbatch_ao_i = ao1_i - ao0_i;
        let shls_slice_i = [[shl0_i as i32, shl1_i as i32], [0, nbas], [nbas, nbas + nbas_aux]];
        let int3c2e_batch_i = {
            let (out, shape) = mol.integral_s1::<int3c2e>(Some(&shls_slice_i));
            rt::asarray((out, shape.f(), &device))
        };

        // half-transform for batch i
        let mut cderi_half_i = vec![];
        for iset in 0..nset {
            let nocc = occ_coeff_list[iset].shape()[1];
            let mut eri_half_iset_vec = unsafe { uninitialized_vec::<f64>(nbatch_ao_i * nocc * naux).unwrap() };
            let eri_half_iset = rt::asarray((&mut eri_half_iset_vec, [nbatch_ao_i, nocc, naux].f(), &device));
            (0..naux).into_par_iter().for_each(|p| {
                let cderi_half_iaux = eri_half_iset.i((.., .., p));
                let mut cderi_half_iaux = unsafe { cderi_half_iaux.force_mut() };
                cderi_half_iaux.matmul_from(&int3c2e_batch_i.i((.., .., p)), &occ_coeff_list[iset], 1.0, 0.0);
            });
            let eri_half_iset =
                rt::asarray((&mut eri_half_iset_vec, [nbatch_ao_i * nocc, naux].f(), &device)).into_reverse_axes();
            rt::linalg::solve_triangular((tsr_int2c2e_l.t(), eri_half_iset, Lower));
            let cderi_half_iset = rt::asarray((eri_half_iset_vec, [nbatch_ao_i, nocc, naux].f(), &device));
            cderi_half_i.push(cderi_half_iset);
        }

        // perform contribution to vk for intra-batch i
        for iset in 0..nset {
            let nocc = occ_coeff_list[iset].shape()[1];
            let cderi_half_iset = cderi_half_i[iset].reshape([nbatch_ao_i, nocc * naux]);
            ks.i_mut((ao0_i..ao1_i, ao0_i..ao1_i, iset)).matmul_from(&cderi_half_iset, &cderi_half_iset.t(), 1.0, 1.0);
        }

        for batch_j in 0..batch_i {
            let &[shl0_j, shl1_j] = &partition[batch_j];
            let ao0_j = ao_loc[shl0_j];
            let ao1_j = ao_loc[shl1_j];
            let nbatch_ao_j = ao1_j - ao0_j;
            let shls_slice_j = [[shl0_j as i32, shl1_j as i32], [0, nbas], [nbas, nbas + nbas_aux]];
            let int3c2e_batch_j = {
                let (out, shape) = mol.integral_s1::<int3c2e>(Some(&shls_slice_j));
                rt::asarray((out, shape.f(), &device))
            };

            let mut eri_half_iset_vec = unsafe { uninitialized_vec::<f64>(nbatch_ao_j * nocc_max * naux).unwrap() };
            for iset in 0..nset {
                // half-transform for batch j
                let nocc = occ_coeff_list[iset].shape()[1];
                let eri_half_iset = rt::asarray((&mut eri_half_iset_vec, [nbatch_ao_j, nocc, naux].f(), &device));
                (0..naux).into_par_iter().for_each(|p| {
                    let cderi_half_iaux = eri_half_iset.i((.., .., p));
                    let mut cderi_half_iaux = unsafe { cderi_half_iaux.force_mut() };
                    cderi_half_iaux.matmul_from(&int3c2e_batch_j.i((.., .., p)), &occ_coeff_list[iset], 1.0, 0.0);
                });
                let eri_half_iset =
                    rt::asarray((&mut eri_half_iset_vec, [nbatch_ao_j * nocc, naux].f(), &device)).into_reverse_axes();
                rt::linalg::solve_triangular((tsr_int2c2e_l.t(), eri_half_iset, Lower));

                // contribution to vk between batch i and j
                let cderi_half_iset_j = rt::asarray((&eri_half_iset_vec, [nbatch_ao_j, nocc * naux].f(), &device));
                let cderi_half_iset_i = cderi_half_i[iset].reshape([nbatch_ao_i, nocc * naux]);
                let mut scr_k = unsafe { rt::empty(([nbatch_ao_i, nbatch_ao_j].f(), &device)) };
                scr_k.matmul_from(&cderi_half_iset_i, &cderi_half_iset_j.t(), 1.0, 0.0);
                *&mut ks.i_mut((ao0_i..ao1_i, ao0_j..ao1_j, iset)) += &scr_k;
                *&mut ks.i_mut((ao0_j..ao1_j, ao0_i..ao1_i, iset)) += &scr_k.t();
            }
        }
    }
    ks
}

/// Estimate memory requirement for VK RI direct method with MO coefficients and occupations.
///
/// This function corresponds to [`generate_vk_ri_direct_coeff_with_rstsr`].
///
/// # Parameters
///
/// - `nao`: Number of atomic orbitals.
/// - `naux`: Number of auxiliary basis functions.
/// - `nocc_max`: Maximum number of occupied molecular orbitals among all sets.
/// - `nset`: Number of density matrix sets.
pub fn mem_estimate_vk_ri_direct_coeff(nao: usize, naux: usize, nocc_max: usize, nset: usize) -> MemEstimate {
    // int3c2e, cderi_half_i, eri_half_iset_vec
    let batched = nao * naux + nocc_max * naux * (nset + 1);
    // ks, int2c2e
    let fixed = nao * nao * nset + naux * naux;
    // cderi per auxiliary
    let thread = nao * nao;
    MemEstimate { batched, fixed, thread }
}

pub fn generate_vk_ri_direct_coeff(
    scaling_factor: f64,
    mo_coeff: &[MatrixFull<f64>],
    mo_occ: &[Vec<f64>],
    mol_obj: &Molecule,
    batch_size: usize,
) -> Vec<MatrixUpper<f64>> {
    let device = DeviceBLAS::default();
    let mo_coeff = mo_coeff.to_rstsr(&device);
    let mo_occ = mo_occ.to_rstsr(&device);

    let nao = mo_coeff.shape()[0];
    let nocc = mo_coeff.shape()[1];
    let nset = mo_coeff.shape()[2];
    let nao_tp = (nao + 1) * nao / 2;

    // shape sanity check
    assert_eq!(mo_occ.shape(), &[nocc, nset], "Molecular occupations must have shape (nmo, nset)");

    let mut ks_rstsr = generate_vk_ri_direct_coeff_with_rstsr(mo_coeff.view(), mo_occ.view(), mol_obj, batch_size);
    ks_rstsr *= scaling_factor;

    // Tsr -> Vec<MatrixUpper>
    let mut ks = vec![];
    for iset in 0..nset {
        let k = ks_rstsr.i((.., .., iset)).pack_tri(Upper).into_vec();
        ks.push(unsafe { MatrixUpper::from_vec_unchecked(nao_tp, k) })
    }
    ks
}

/* #endregion ri-vk direct coeff */

/* #region ri-vk direct dm */

pub fn generate_vk_ri_direct_dm_with_rstsr(dms: TsrView<f64>, mol_obj: &Molecule, batch_size: usize) -> Tsr<f64> {
    assert_eq!(dms.ndim(), 3, "Density matrices must have 3 dimensions");

    // initialize mol and aux
    let mut mol = mol_obj.initialize_cint(true);
    let mut aux = mol_obj.make_auxmol_fake().initialize_cint(false);

    // get shapes
    let nao = dms.shape()[0];
    let nset = dms.shape()[2];
    let naux = aux.cgto_loc().last().unwrap().clone();
    let device = dms.device().clone();
    let nbas = mol_obj.cint_bas.len() as i32;
    let nbas_aux = mol_obj.cint_aux_bas.len() as i32;

    // shape check
    assert_eq!(dms.shape(), &[nao, nao, nset], "Density matrices must have shape (nao, nao, nset)");

    // get partition
    let ao_loc = &mol.cgto_loc()[..=(nbas as usize)];
    let partition = blocksize_partition(&ao_loc, batch_size);

    // initialize vk as result
    let mut ks = rt::zeros(([nao, nao, nset].f(), &device));

    // initialize int2c2e and perform cholesky decomposition
    let tsr_int2c2e = {
        let shls_slice = [[nbas, nbas + nbas_aux], [nbas, nbas + nbas_aux]];
        let (out, shape) = mol.integral_s1::<int2c2e>(Some(&shls_slice));
        rt::asarray((out, shape.f(), &device))
    };
    let tsr_int2c2e_l = rt::linalg::cholesky((tsr_int2c2e, Upper));

    for (batch_i, &[shl0_i, shl1_i]) in partition.iter().enumerate() {
        let ao0_i = ao_loc[shl0_i];
        let ao1_i = ao_loc[shl1_i];
        let nbatch_ao_i = ao1_i - ao0_i;
        let shls_slice_i = [[shl0_i as i32, shl1_i as i32], [0, nbas], [nbas, nbas + nbas_aux]];
        let int3c2e_batch_i = {
            let (out, shape) = mol.integral_s1::<int3c2e>(Some(&shls_slice_i));
            rt::asarray((out, shape.f(), &device))
        };

        // half-transform for batch i
        let mut inveri_half_i = vec![];
        for iset in 0..nset {
            let mut eri_half_iset_vec = unsafe { uninitialized_vec::<f64>(nbatch_ao_i * nao * naux).unwrap() };
            let eri_half_iset = rt::asarray((&mut eri_half_iset_vec, [nbatch_ao_i, nao, naux].f(), &device));
            (0..naux).into_par_iter().for_each(|p| {
                let cderi_half_iaux = eri_half_iset.i((.., .., p));
                let mut cderi_half_iaux = unsafe { cderi_half_iaux.force_mut() };
                cderi_half_iaux.matmul_from(&int3c2e_batch_i.i((.., .., p)), &dms.i((.., .., iset)), 1.0, 0.0);
            });
            let eri_half_iset =
                rt::asarray((&mut eri_half_iset_vec, [nbatch_ao_i * nao, naux].f(), &device)).into_reverse_axes();
            let eri_half_iset = rt::linalg::solve_triangular((tsr_int2c2e_l.t(), eri_half_iset, Lower));
            rt::linalg::solve_triangular((tsr_int2c2e_l.view(), eri_half_iset, Upper));
            let inveri_half_iset = rt::asarray((eri_half_iset_vec, [nbatch_ao_i, nao, naux].f(), &device));
            inveri_half_i.push(inveri_half_iset);
        }

        // perform contribution to vk for intra-batch i
        for iset in 0..nset {
            let int3c2e_batch_i = int3c2e_batch_i.reshape([nbatch_ao_i, nao * naux]);
            let inveri_half_iset = inveri_half_i[iset].reshape([nbatch_ao_i, nao * naux]);
            ks.i_mut((ao0_i..ao1_i, ao0_i..ao1_i, iset)).matmul_from(&inveri_half_iset, &int3c2e_batch_i.t(), 1.0, 1.0);
        }

        for batch_j in 0..batch_i {
            let &[shl0_j, shl1_j] = &partition[batch_j];
            let ao0_j = ao_loc[shl0_j];
            let ao1_j = ao_loc[shl1_j];
            let nbatch_ao_j = ao1_j - ao0_j;
            let shls_slice_j = [[shl0_j as i32, shl1_j as i32], [0, nbas], [nbas, nbas + nbas_aux]];
            let int3c2e_batch_j = {
                let (out, shape) = mol.integral_s1::<int3c2e>(Some(&shls_slice_j));
                rt::asarray((out, shape.f(), &device))
            };

            for iset in 0..nset {
                // half-transform for batch j
                let mut scr_k = unsafe { rt::empty(([nbatch_ao_i, nbatch_ao_j].f(), &device)) };
                let inveri_half_iset_i = inveri_half_i[iset].reshape([nbatch_ao_i, nao * naux]);
                let int3c2e_batch_j = int3c2e_batch_j.reshape([nbatch_ao_j, nao * naux]);
                scr_k.matmul_from(&inveri_half_iset_i, &int3c2e_batch_j.t(), 1.0, 0.0);
                *&mut ks.i_mut((ao0_i..ao1_i, ao0_j..ao1_j, iset)) += &scr_k;
                *&mut ks.i_mut((ao0_j..ao1_j, ao0_i..ao1_i, iset)) += &scr_k.t();
            }
        }
    }
    ks
}

/// Estimate memory requirement for VK RI direct method with density matrices.
pub fn mem_estimate_vk_ri_direct_dm(nao: usize, naux: usize, nset: usize) -> MemEstimate {
    // int3c2e_batch, inveri_half_iset
    let batched = nao * naux + nao * naux * nset;
    // ks, int2c2e
    let fixed = nao * nao * nset + naux * naux;
    // cderi per auxiliary
    let thread = nao * nao;
    MemEstimate { batched, fixed, thread }
}

pub fn generate_vk_ri_direct_dm(
    scaling_factor: f64,
    dms: &[MatrixFull<f64>],
    mol_obj: &Molecule,
    batch_size: usize,
) -> Vec<MatrixUpper<f64>> {
    let device = DeviceBLAS::default();
    let dms = dms.to_rstsr(&device);

    let nao = dms.shape()[0];
    let nset = dms.shape()[2];
    let nao_tp = (nao + 1) * nao / 2;

    // shape sanity check
    assert_eq!(dms.shape(), &[nao, nao, nset], "Density matrices must have shape (nao, nao, nset)");

    let mut ks_rstsr = generate_vk_ri_direct_dm_with_rstsr(dms.view(), mol_obj, batch_size);
    ks_rstsr *= scaling_factor;

    // Tsr -> Vec<MatrixUpper>
    let mut ks = vec![];
    for iset in 0..nset {
        let k = ks_rstsr.i((.., .., iset)).pack_tri(Upper).into_vec();
        ks.push(unsafe { MatrixUpper::from_vec_unchecked(nao_tp, k) })
    }
    ks
}

/* #endregion ri-vk direct dm */

#[cfg(test)]
mod debug {
    use super::*;
    use crate::scf_io::SCF;

    #[test]
    fn test_nh3_j() {
        let scf_data = initialize_nh3();
        let device = DeviceBLAS::default();

        let dm = [&scf_data.density_matrix[0]].as_ref().to_rstsr(&device);
        let mol = &scf_data.mol;

        // incore
        let rimatr = scf_data.rimatr.as_ref().unwrap().0.to_rstsr_view(&device);
        let j_rstsr = generate_vj_ri_incore_with_rstsr(rimatr, dm.view());
        let fp = fingerprint_f64(j_rstsr.i((.., .., 0)));
        let ref_fp = 37.83424292927407;
        assert!((fp / ref_fp - 1.0).abs() < 1e-5);

        // direct, full batch
        let j_rstsr = generate_vj_ri_direct_with_rstsr(dm.view(), mol, 10000);
        let fp = fingerprint_f64(j_rstsr.i((.., .., 0)));
        let ref_fp = 37.83424292927407;
        assert!((fp / ref_fp - 1.0).abs() < 1e-5);

        // direct, small batch
        let j_rstsr = generate_vj_ri_direct_with_rstsr(dm.view(), mol, 16);
        let fp = fingerprint_f64(j_rstsr.i((.., .., 0)));
        let ref_fp = 37.83424292927407;
        assert!((fp / ref_fp - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_nh3_k() {
        let scf_data = initialize_nh3();
        let device = DeviceBLAS::default();

        let mo_coeff = [&scf_data.eigenvectors[0]].as_ref().to_rstsr(&device);
        let mo_occ = (&scf_data.occupation[0]).to_rstsr(&device).into_slice((.., None));
        let dms = [&scf_data.density_matrix[0]].as_ref().to_rstsr(&device);
        let mol = &scf_data.mol;

        // incore, full batch
        let rimatr = scf_data.rimatr.as_ref().unwrap().0.to_rstsr_view(&device);
        let k_rstsr = generate_vk_ri_incore_coeff_with_rstsr(rimatr.view(), mo_coeff.view(), mo_occ.view(), 10000);
        let fp = fingerprint_f64(k_rstsr.i((.., .., 0)));
        let ref_fp = 12.950224351107128;
        assert!((fp / ref_fp - 1.0).abs() < 1e-5);

        // incore, small batch
        let k_rstsr = generate_vk_ri_incore_coeff_with_rstsr(rimatr.view(), mo_coeff.view(), mo_occ.view(), 16);
        let fp = fingerprint_f64(k_rstsr.i((.., .., 0)));
        let ref_fp = 12.950224351107128;
        assert!((fp / ref_fp - 1.0).abs() < 1e-5);

        // semi-incore, full batch
        let k_rstsr = generate_vk_ri_semi_incore_coeff_with_rstsr(mo_coeff.view(), mo_occ.view(), mol, 10000);
        let fp = fingerprint_f64(k_rstsr.i((.., .., 0)));
        let ref_fp = 12.950224351107128;
        assert!((fp / ref_fp - 1.0).abs() < 1e-5);

        // semi-incore, small batch
        let k_rstsr = generate_vk_ri_semi_incore_coeff_with_rstsr(mo_coeff.view(), mo_occ.view(), mol, 16);
        let fp = fingerprint_f64(k_rstsr.i((.., .., 0)));
        let ref_fp = 12.950224351107128;
        assert!((fp / ref_fp - 1.0).abs() < 1e-5);

        // direct, full batch
        let k_rstsr = generate_vk_ri_direct_coeff_with_rstsr(mo_coeff.view(), mo_occ.view(), mol, 10000);
        let fp = fingerprint_f64(k_rstsr.i((.., .., 0)));
        let ref_fp = 12.950224351107128;
        assert!((fp / ref_fp - 1.0).abs() < 1e-5);

        // direct, small batch
        let k_rstsr = generate_vk_ri_direct_coeff_with_rstsr(mo_coeff.view(), mo_occ.view(), mol, 16);
        let fp = fingerprint_f64(k_rstsr.i((.., .., 0)));
        let ref_fp = 12.950224351107128;
        assert!((fp / ref_fp - 1.0).abs() < 1e-5);

        // direct, full batch, dm version
        let k_rstsr = generate_vk_ri_direct_dm_with_rstsr(dms.view(), mol, 10000);
        let fp = fingerprint_f64(k_rstsr.i((.., .., 0)));
        let ref_fp = 12.950224351107128;
        assert!((fp / ref_fp - 1.0).abs() < 1e-5);

        // direct, small batch, dm version
        let k_rstsr = generate_vk_ri_direct_dm_with_rstsr(dms.view(), mol, 16);
        let fp = fingerprint_f64(k_rstsr.i((.., .., 0)));
        let ref_fp = 12.950224351107128;
        assert!((fp / ref_fp - 1.0).abs() < 1e-5);
    }

    fn initialize_nh3() -> SCF {
        let input_token = r##"
[ctrl]
     print_level =          2
     method =               "hf"
     basis_path =           "basis-set-pool/def2-TZVP"
     auxbas_path =          "basis-set-pool/def2-SVP-JKFIT"
     eri_type =             "ri-v"
     charge =               0.0
     spin =                 1.0
     spin_polarization =    false
     initial_guess=         "sad"
     mixer =                "diis"
     num_threads =          16

[geom]
    name = "NH3"
    unit = "Angstrom"
    position = """
        N  0.0  0.0  0.0
        H  0.0  1.5  1.0
        H  1.4  1.1  0.0
        H  1.2  0.0  1.3
    """
"##;
        let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
        let (mut ctrl, mut geom) = crate::ctrl_io::parse_ctl_from_json(&keys).unwrap();
        let mol = Molecule::build_native(ctrl, geom, None).unwrap();
        let mut scf_data = crate::scf_io::SCF::build(mol, &None);
        crate::scf_io::scf_without_build(&mut scf_data, &None);
        return scf_data;
    }
}
