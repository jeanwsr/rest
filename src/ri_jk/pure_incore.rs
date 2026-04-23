use super::prelude_dev::*;

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
pub fn get_vj_ri_incore(cderi: TsrView<f64>, dms: TsrView<f64>) -> Tsr<f64> {
    assert_eq!(dms.ndim(), 3, "DM must have 3 dimensions");
    assert_eq!(cderi.ndim(), 2, "Cholesky ERI must have 2 dimensions");

    // get shapes
    let nset = dms.shape()[2];
    let nao = dms.shape()[0];
    let nao_tp = (nao + 1) * nao / 2;

    // shape check
    assert_eq!(cderi.shape()[0], nao_tp, "Cholesky ERI must have shape (nao_tp, naux)");

    // pack density matrix with upper-triangular, diagonal doubled
    // -- (eq.1) -- //
    let mut dms_tp: Tsr<f64> = rt::zeros(([nao_tp, nset].f(), dms.device()));
    for iset in 0..nset {
        let dm = dms.i((.., .., iset));
        let dm_diag = dm.diagonal(None);
        let mut dm = 2.0_f64 * &dm;
        dm.diagonal_mut(None).assign(&dm_diag);
        let dm_tp = dm.pack_triu();
        dms_tp.i_mut((.., iset)).assign(dm_tp);
    }

    // generate j contribution
    // -- (eq.2) -- //
    let scr_j = cderi.t() % &dms_tp;
    // -- (eq.3) -- //
    let js_tp = &cderi % &scr_j;

    // returns symmetrized part
    // -- (eq.4) -- //
    js_tp.unpack_tri(Upper, FlagSymm::Sy)
}

/* #endregion ri-vj incore */

/* #region ri-vk incore coeff */

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
pub fn get_vk_ri_incore_coeff(
    cderi: TsrView<f64>,
    mo_coeff: TsrView<f64>,
    mo_occ: TsrView<f64>,
    batch_size: usize,
) -> Tsr<f64> {
    assert_eq!(mo_coeff.ndim(), 3, "Molecular orbital coefficients must have 3 dimensions");
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
    // -- (eq.1) -- //
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
                // -- (eq.2) -- //
                let cderi_iaux = cderi.i((.., iaux + p)).unpack_tri(Upper, FlagSymm::Sy);
                let cderi_half_iaux = cderi_half.i((.., .., p));
                let mut cderi_half_iaux = unsafe { cderi_half_iaux.force_mut() };
                // -- (eq.3) -- //
                cderi_half_iaux.matmul_from(&cderi_iaux, occ_coeff, 1.0, 0.0);
            });
            // build vk contribution
            // -- (eq.4) -- //
            let cderi_half = cderi_half.into_shape([nao, nocc * nbatch]);
            ks.i_mut((.., .., iset)).matmul_from(&cderi_half, &cderi_half.t(), 1.0, 1.0);
        }
    });
    ks
}

/* #endregion ri-vk incore coeff */

/* #region ri-vk incore dm */

/// Generate Exchange (K) matrix using RI incore method with density matrices.
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
pub fn get_vk_ri_incore_dm(cderi: TsrView<f64>, dms: TsrView<f64>, batch_size: usize) -> Tsr<f64> {
    assert_eq!(dms.ndim(), 3, "DM must have 3 dimensions");
    assert_eq!(cderi.ndim(), 2, "Cholesky ERI must have 2 dimensions");

    // get shapes
    let nao = dms.shape()[0];
    let nset = dms.shape()[2];
    let naux = cderi.shape()[1];
    let nao_tp = (nao + 1) * nao / 2;
    let device = cderi.device().clone();

    // shape check
    assert_eq!(cderi.shape(), &[nao_tp, naux], "Cholesky ERI must have shape (nao_tp, naux)");

    // initialize vk as result
    let mut ks = rt::zeros(([nao, nao, nset].f(), &device));

    // process each auxiliary function
    (0..naux).step_by(batch_size).for_each(|iaux| {
        // get and unpack cderi for this auxiliary function
        // cderi_iaux: (nao, nao, nbatch)
        let nbatch = if iaux + batch_size <= naux { batch_size } else { naux - iaux };

        // unpack cderi for this batch
        // -- (eq.1) -- //
        let cderi_batch: Tsr<f64> = unsafe { rt::empty(([nao, nao, nbatch].f(), &device)) };
        (0..nbatch).into_par_iter().for_each(|p| {
            let cderi_iaux = cderi.i((.., iaux + p)).unpack_tri(Upper, FlagSymm::Sy);
            let cderi_iaux_mut = cderi_batch.i((.., .., p));
            let mut cderi_iaux_mut = unsafe { cderi_iaux_mut.force_mut() };
            cderi_iaux_mut.assign(&cderi_iaux);
        });

        for iset in 0..nset {
            // half-transformed integrals: (nao, nao, nbatch)
            // -- (eq.2) -- //
            let dm = dms.i((.., .., iset));
            let cderi_half = unsafe { rt::empty(([nao, nao, nbatch].f(), &device)) };
            (0..nbatch).into_par_iter().for_each(|p| {
                let cderi_iaux = cderi_batch.i((.., .., p));
                let cderi_half_iaux = cderi_half.i((.., .., p));
                let mut cderi_half_iaux = unsafe { cderi_half_iaux.force_mut() };
                cderi_half_iaux.matmul_from(&cderi_iaux, &dm, 1.0, 0.0);
            });
            // build vk contribution
            // -- (eq.3) -- //
            let cderi_half = cderi_half.into_shape([nao, nao * nbatch]);
            let cderi_batch = cderi_batch.reshape([nao, nao * nbatch]);
            ks.i_mut((.., .., iset)).matmul_from(&cderi_half, &cderi_batch.t(), 1.0, 1.0);
        }
    });
    ks
}

/* #endregion ri-vk incore dm */
