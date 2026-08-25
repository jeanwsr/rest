use super::prelude_dev::*;
use log::warn;

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
/// - `mol`: [`CInt`]; `aux`: [`CInt`]
///
///   - Molecule and auxiliary basis objects.
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
pub fn get_vj_ri_direct(dms: TsrView<f64>, mol: &CInt, aux: &CInt, batch_size: usize) -> Tsr<f64> {
    // dm shape: (nao, nao, nset) in f-contig
    assert_eq!(dms.ndim(), 3, "DM must have 3 dimensions");

    // get shapes
    let nao = mol.nao();
    let naux = aux.nao();
    let nbas = mol.nbas();
    let nset = dms.shape()[2];
    let nao_tp = (nao + 1) * nao / 2;
    let device = dms.device().clone();

    // get aux partition
    let aux_loc = &aux.ao_loc();
    let partition = blocksize_partition(aux_loc, batch_size);

    // int2c2e (may be stored in SCF iteration, generate on-the-fly costs some but not that much)
    let tsr_int2c2e = {
        let (out, shape) = aux.integrate("int2c2e", "s1", None).into();
        rt::asarray((out, shape.f(), &device))
    };

    // pack density matrix with upper-triangular, diagonal doubled
    // -- (eq.1) -- //
    let mut dms_tp: Tsr<f64> = rt::zeros(([nao_tp, nset].f(), &device));
    for iset in 0..nset {
        let dm = dms.i((.., .., iset));
        let dm_diag = dm.diagonal(None);
        let mut dm = 2.0_f64 * &dm;
        dm.diagonal_mut(None).assign(&dm_diag);
        let dm_tp = dm.pack_triu();
        dms_tp.i_mut((.., iset)).assign(dm_tp);
    }

    // generate scr_j
    let mut scr_j: Tsr<f64> = rt::zeros(([naux, nset].f(), &device));
    let mut idx_ao = 0;
    for &[shl0, shl1] in &partition {
        // -- (eq.2) -- //
        let nbatch_ao = aux_loc[shl1] - aux_loc[shl0];
        let shls_slice = [[0, nbas], [0, nbas], [shl0, shl1]];
        let int3c2e_batch = {
            let (out, shape) = CInt::integrate_cross("int3c2e", [mol, mol, aux], "s2ij", shls_slice).into();
            rt::asarray((out, shape.f(), &device))
        };
        // -- (eq.3) -- //
        let slc = slice!(idx_ao, idx_ao + nbatch_ao);
        scr_j.i_mut(slc).matmul_from(&int3c2e_batch.t(), &dms_tp, 1.0, 0.0);
        idx_ao += nbatch_ao;
    }

    // solve scr_j
    // -- (eq.4) -- //
    let scr_js = rt::linalg::solve_general((tsr_int2c2e, scr_j));

    // generate j contribution
    let mut js_tp: Tsr<f64> = rt::zeros(([nao_tp, nset].f(), &device));
    let mut idx_ao = 0;
    for &[shl0, shl1] in &partition {
        // -- (eq.5) -- //
        let nbatch_ao = aux_loc[shl1] - aux_loc[shl0];
        let shls_slice = [[0, nbas], [0, nbas], [shl0, shl1]];
        let int3c2e_batch = {
            let (out, shape) = CInt::integrate_cross("int3c2e", [mol, mol, aux], "s2ij", shls_slice).into();
            rt::asarray((out, shape.f(), &device))
        };
        // -- (eq.6) -- //
        let slc = slice!(idx_ao, idx_ao + nbatch_ao);
        js_tp.matmul_from(&int3c2e_batch, &scr_js.i(slc), 1.0, 1.0);
        idx_ao += nbatch_ao;
    }

    // returns symmetrized part
    // -- (eq.7) -- //
    js_tp.unpack_tri(Upper, FlagSymm::Sy)
}

/* #endregion ri-vj direct */

/* #region ri-vk semi-direct */

/// Generate Exchange (K) matrix using RI semi-direct method with MO coefficients and occupations.
///
/// semi-direct here means that 3c-2e ERI are computed on-the-fly in batches, while the
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
/// - `mol`: [`CInt`]; `aux`: [`CInt`]
///
///   - Molecule and auxiliary basis objects.
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
pub fn get_vk_ri_semi_direct_coeff(
    mo_coeff: TsrView<f64>,
    mo_occ: TsrView<f64>,
    mol: &CInt,
    aux: &CInt,
    batch_size: usize,
) -> Tsr<f64> {
    assert_eq!(mo_coeff.ndim(), 3, "Molecular orbital coefficients must have 3 dimensions");
    assert_eq!(mo_occ.ndim(), 2, "Molecular occupations must have 2 dimensions");

    // get shapes
    let nao = mol.nao();
    let naux = aux.nao();
    let nbas = mol.nbas();
    let nmo = mo_coeff.shape()[1];
    let nset = mo_coeff.shape()[2];
    let device = mo_coeff.device().clone();

    // shape check
    assert_eq!(mo_occ.shape(), &[nmo, nset], "Molecular occupations must have shape (nmo, nset)");

    // check occupation not less than zero
    let occ_neg = rt::lt(&mo_occ, 0.0).sum();
    if occ_neg > 0 {
        warn!("in generate_vk_ri_incore_coeff_with_rstsr, negative occupation found: {occ_neg} elements < 0");
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
    let nocc_max = occ_coeff_list.iter().map(|x| x.shape()[1]).max().unwrap();

    // get partition
    let aux_loc = &aux.ao_loc();
    let partition = blocksize_partition(aux_loc, batch_size);

    // initialize vk as result
    let mut ks = rt::zeros(([nao, nao, nset].f(), &device));

    // initialize half-transformed cderi storage
    let mut eri_half_vec = unsafe { uninitialized_vec::<f64>(nao * nocc_max * naux).unwrap() };

    // initialize int2c2e and perform cholesky decomposition
    // -- (eq.2) -- //
    let tsr_int2c2e = {
        let (out, shape) = aux.integrate("int2c2e", "s1", None).into();
        rt::asarray((out, shape.f(), &device))
    };
    let tsr_int2c2e_l = rt::linalg::cholesky((tsr_int2c2e, Upper));

    for iset in 0..nset {
        // get half-transformed cderi for this density matrix set
        let occ_coeff = &occ_coeff_list[iset];
        let nocc = occ_coeff.shape()[1];
        let eri_half = rt::asarray((&mut eri_half_vec[..nao * nocc * naux], [nao, nocc, naux].f(), &device));

        for &[shl0, shl1] in &partition {
            // -- (eq.3) -- //
            let aux0 = aux_loc[shl0];
            let aux1 = aux_loc[shl1];
            let nbatch_aux = aux1 - aux0;
            let shls_slice = [[0, nbas], [0, nbas], [shl0, shl1]];
            let int3c2e_batch = {
                let (out, shape) = CInt::integrate_cross("int3c2e", [mol, mol, aux], "s2ij", shls_slice).into();
                rt::asarray((out, shape.f(), &device))
            };
            // half-transform
            (0..nbatch_aux).into_par_iter().for_each(|p| {
                // -- (eq.4) -- //
                let cderi_iaux = int3c2e_batch.i((.., p)).unpack_tri(Upper, FlagSymm::Sy);
                let cderi_half_iaux = eri_half.i((.., .., aux0 + p));
                let mut cderi_half_iaux = unsafe { cderi_half_iaux.force_mut() };
                // -- (eq.5) -- //
                cderi_half_iaux.matmul_from(&cderi_iaux, occ_coeff, 1.0, 0.0);
            });
        }

        // solve eri_half
        // -- (eq.6) -- //
        let eri_half =
            rt::asarray((&mut eri_half_vec[..nao * nocc * naux], [nao * nocc, naux].f(), &device)).into_reverse_axes();
        rt::linalg::solve_triangular((tsr_int2c2e_l.t(), eri_half, Lower));

        // build vk contribution
        // -- (eq.7) -- //
        let cderi_half = rt::asarray((&mut eri_half_vec[..nao * nocc * naux], [nao, nocc * naux].f(), &device));
        ks.i_mut((.., .., iset)).matmul_from(&cderi_half, &cderi_half.t(), 1.0, 1.0);
    }
    ks
}

/// Estimate memory requirement for VK RI semi-direct method with MO coefficients and occupations.
///
/// This function corresponds to [`generate_vk_ri_semi_direct_coeff_with_rstsr`].
///
/// # Parameters
///
/// - `nao`: Number of atomic orbitals.
/// - `naux`: Number of auxiliary basis functions.
/// - `nocc_max`: Maximum number of occupied molecular orbitals among all sets.
/// - `nset`: Number of density matrix sets.
pub fn mem_estimate_vk_ri_semi_direct_coeff(nao: usize, naux: usize, nocc_max: usize, nset: usize) -> MemEstimate {
    let nao_tp = (nao + 1) * nao / 2;
    // int3c2e batch
    let batched = nao_tp;
    // occ_coeff_list, int2c2e, eri_half, ks
    // note: eri_half is the major memory consumer here
    let fixed = nao * nocc_max * nset + nao * nao + nao * nocc_max * naux + nao * nao * nset;
    // cderi per auxiliary
    let thread = nao * nao;
    MemEstimate { batched, fixed, thread }
}

/* #endregion ri-vk semi-direct */

/* #region ri-vk direct dm */

/// Generate Exchange (K) matrix using RI direct method with density matrices.
///
/// This function is low-level implementation, using RSTSR tensors as input and output. For
/// high-level interface, please refer to [`generate_vk_ri_direct_dm`].
///
/// # Parameters
///
/// - `dms`: [`TsrView<f64>`]
///
///   - Density matrices in shape (nao, nao, nset), stored in f-contiguous order.
///   - We will check `ndim == 3`. Please expand dimension if necessary, especially for RHF case
///     where `nset = 1`.
///   - Each density matrix corresponds to one set of K matrix to be computed.
///   - Please ensure the density matrices are symmetric.
///
/// - `mol`: [`CInt`]; `aux`: [`CInt`]
///
///   - Molecule and auxiliary basis objects.
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
pub fn get_vk_ri_direct_dm(dms: TsrView<f64>, mol: &CInt, aux: &CInt, batch_size: usize) -> Tsr<f64> {
    assert_eq!(dms.ndim(), 3, "Density matrices must have 3 dimensions");

    // get shapes
    let nao = mol.nao();
    let naux = aux.nao();
    let nbas = mol.nbas();
    let nset = dms.shape()[2];
    let device = dms.device().clone();

    // shape check
    assert_eq!(dms.shape(), &[nao, nao, nset], "Density matrices must have shape (nao, nao, nset)");

    // get partition
    let ao_loc = &mol.ao_loc();
    let partition = blocksize_partition(ao_loc, batch_size);

    // initialize vk as result
    let mut ks = rt::zeros(([nao, nao, nset].f(), &device));

    // initialize int2c2e and perform cholesky decomposition
    // -- (eq.1) -- //
    let tsr_int2c2e = {
        let (out, shape) = aux.integrate("int2c2e", "s1", None).into();
        rt::asarray((out, shape.f(), &device))
    };
    let tsr_int2c2e_l = rt::linalg::cholesky((tsr_int2c2e, Upper));

    for (batch_i, &[shl0_i, shl1_i]) in partition.iter().enumerate() {
        // -- (eq.2) -- //
        let ao0_i = ao_loc[shl0_i];
        let ao1_i = ao_loc[shl1_i];
        let nbatch_ao_i = ao1_i - ao0_i;
        let shls_slice_i = [[shl0_i, shl1_i], [0, nbas], [0, aux.nbas()]];
        let int3c2e_batch_i = {
            let (out, shape) = CInt::integrate_cross("int3c2e", [mol, mol, aux], "s1", shls_slice_i).into();
            rt::asarray((out, shape.f(), &device))
        };

        // half-transform for batch i
        let mut inveri_half_i = vec![];
        for iset in 0..nset {
            let mut eri_half_iset_vec = unsafe { uninitialized_vec::<f64>(nbatch_ao_i * nao * naux).unwrap() };
            let eri_half_iset = rt::asarray((&mut eri_half_iset_vec, [nbatch_ao_i, nao, naux].f(), &device));
            // -- (eq.3) -- //
            (0..naux).into_par_iter().for_each(|p| {
                let cderi_half_iaux = eri_half_iset.i((.., .., p));
                let mut cderi_half_iaux = unsafe { cderi_half_iaux.force_mut() };
                cderi_half_iaux.matmul_from(&int3c2e_batch_i.i((.., .., p)), &dms.i((.., .., iset)), 1.0, 0.0);
            });
            // -- (eq.4), (eq.5) -- //
            let eri_half_iset =
                rt::asarray((&mut eri_half_iset_vec, [nbatch_ao_i * nao, naux].f(), &device)).into_reverse_axes();
            let eri_half_iset = rt::linalg::solve_triangular((tsr_int2c2e_l.t(), eri_half_iset, Lower));
            rt::linalg::solve_triangular((tsr_int2c2e_l.view(), eri_half_iset, Upper));
            let inveri_half_iset = rt::asarray((eri_half_iset_vec, [nbatch_ao_i, nao, naux].f(), &device));
            inveri_half_i.push(inveri_half_iset);
        }

        // perform contribution to vk for intra-batch i
        // -- (eq.7) case 1 -- //
        for iset in 0..nset {
            let int3c2e_batch_i = int3c2e_batch_i.reshape([nbatch_ao_i, nao * naux]);
            let inveri_half_iset = inveri_half_i[iset].reshape([nbatch_ao_i, nao * naux]);
            ks.i_mut((ao0_i..ao1_i, ao0_i..ao1_i, iset)).matmul_from(&inveri_half_iset, &int3c2e_batch_i.t(), 1.0, 0.0);
        }

        for batch_j in 0..batch_i {
            // -- (eq.6) -- //
            let &[shl0_j, shl1_j] = &partition[batch_j];
            let ao0_j = ao_loc[shl0_j];
            let ao1_j = ao_loc[shl1_j];
            let nbatch_ao_j = ao1_j - ao0_j;
            let shls_slice_j = [[shl0_j, shl1_j], [0, nbas], [0, aux.nbas()]];
            let int3c2e_batch_j = {
                let (out, shape) = CInt::integrate_cross("int3c2e", [mol, mol, aux], "s1", shls_slice_j).into();
                rt::asarray((out, shape.f(), &device))
            };

            // -- (eq.7) case 2 -- //
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
    let thread = 0;
    MemEstimate { batched, fixed, thread }
}

/* #endregion ri-vk direct dm */
