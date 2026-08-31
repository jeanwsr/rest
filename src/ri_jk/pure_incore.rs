use super::prelude_dev::*;
use log::{debug, warn};

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

    // pack density matrix into the triangle (off-diagonals doubled via 2D, diagonal kept)
    // -- pack -- //
    let mut dms_tp: Tsr<f64> = rt::zeros(([nao_tp, nset].f(), dms.device()));
    for iset in 0..nset {
        let dm = dms.i((.., .., iset));
        let dm_diag = dm.diagonal(None);
        let mut dm = 2.0_f64 * &dm;
        dm.diagonal_mut(None).assign(&dm_diag);
        let dm_tp = dm.pack_triu();
        dms_tp.i_mut((.., iset)).assign(dm_tp);
    }

    // first contraction: scr_P = sum_tp dms_tp[tp] * cderi[tp, P]
    // -- contract-dm -- //
    let scr_j = cderi.t() % &dms_tp;
    // second contraction: J_tp = sum_P scr_P * cderi[tp, P]
    // -- build-j -- //
    let js_tp = &cderi % &scr_j;

    // unpack the packed triangle to the full symmetric J
    // -- unpack -- //
    js_tp.unpack_tri(Upper, FlagSymm::Sy)
}

/// Generate **Coulomb (J) matrix** using the RI incore method for a
/// **non-symmetric** density matrix (e.g. a TDDFT transition density
/// `P = C_occ·z·C_virᵀ`).
///
/// The Coulomb kernel `(μν|λσ)` is symmetric in λ↔σ, so `J[P]` depends only on
/// the symmetric part of `P`. This function symmetrizes internally by folding
/// `P + Pᵀ` with the diagonal restored to `P_ii` before the packed contraction,
/// which is exact for non-symmetric densities (no caller-side symmetrization
/// needed).
///
/// # Parameters
///
/// - `cderi`: [`TsrView<f64>`] — Cholesky decomposed 3c-2e ERI `(nao_tp, naux)`.
/// - `dms`: [`TsrView<f64>`] — density matrices `(nao, nao, nset)` (may be
///   non-symmetric).
///
/// # Returns
///
/// - [`Tsr<f64>`] — Coulomb (J) matrices `(nao, nao, nset)`, symmetric.
pub fn get_vj_ri_incore_nonsym(cderi: TsrView<f64>, dms: TsrView<f64>) -> Tsr<f64> {
    assert_eq!(dms.ndim(), 3, "DM must have 3 dimensions");
    assert_eq!(cderi.ndim(), 2, "Cholesky ERI must have 2 dimensions");

    // get shapes
    let nset = dms.shape()[2];
    let nao = dms.shape()[0];
    let nao_tp = (nao + 1) * nao / 2;

    // shape check
    assert_eq!(cderi.shape()[0], nao_tp, "Cholesky ERI must have shape (nao_tp, naux)");

    // fold the non-symmetric density: pack (P + Pᵀ) with the diagonal restored
    // to P_ii, so off-diagonal pairs contribute (P_ij + P_ji) — exactly
    // sum_lsigma P_lsigma B_lsigma for the lambda<->sigma-symmetric Coulomb kernel.
    // -- fold-and-pack -- //
    let mut dms_tp: Tsr<f64> = rt::zeros(([nao_tp, nset].f(), dms.device()));
    for iset in 0..nset {
        let dm = dms.i((.., .., iset));
        let dm_diag = dm.diagonal(None);
        let mut dm_sym = &dm + dm.t();
        dm_sym.diagonal_mut(None).assign(&dm_diag);
        let dm_tp = dm_sym.pack_triu();
        dms_tp.i_mut((.., iset)).assign(dm_tp);
    }

    // first contraction: scr_P = sum_tp dms_tp[tp] * cderi[tp, P]
    // -- contract-dm -- //
    let scr_j = cderi.t() % &dms_tp;
    // second contraction: J_tp = sum_P scr_P * cderi[tp, P]
    // -- build-j -- //
    let js_tp = &cderi % &scr_j;

    // unpack the packed triangle to the full symmetric J
    // -- unpack -- //
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
        warn!("in generate_vk_ri_incore_coeff_with_rstsr, negative occupation found: {occ_neg} elements < 0");
    }

    // compress mo_coeff with occupation: keep occupied columns, scale by sqrt(n_i)
    // -- occ-scaled-coeff -- //
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
                // unpack one packed cderi column to the full (nao, nao) M_P
                let cderi_iaux = cderi.i((.., iaux + p)).unpack_tri(Upper, FlagSymm::Sy);
                let cderi_half_iaux = cderi_half.i((.., .., p));
                let mut cderi_half_iaux = unsafe { cderi_half_iaux.force_mut() };
                // left half-transform: (mu i, P) = sum_nu M_P[mu, nu] * C[nu, i]
                // -- half-transform -- //
                cderi_half_iaux.matmul_from(&cderi_iaux, occ_coeff, 1.0, 0.0);
            });
            // accumulate K_s += (M_P C) * (M_P C)^T over this batch
            // -- accumulate-k -- //
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

        // unpack one batch of packed cderi columns to full (nao, nao) M_P
        // -- unpack-batch -- //
        let cderi_batch: Tsr<f64> = unsafe { rt::empty(([nao, nao, nbatch].f(), &device)) };
        (0..nbatch).into_par_iter().for_each(|p| {
            let cderi_iaux = cderi.i((.., iaux + p)).unpack_tri(Upper, FlagSymm::Sy);
            let cderi_iaux_mut = cderi_batch.i((.., .., p));
            let mut cderi_iaux_mut = unsafe { cderi_iaux_mut.force_mut() };
            cderi_iaux_mut.assign(&cderi_iaux);
        });

        for iset in 0..nset {
            // half-transformed integrals: (M_P dm)[mu, nu] for this batch
            // -- half-transform -- //
            let dm = dms.i((.., .., iset));
            let cderi_half = unsafe { rt::empty(([nao, nao, nbatch].f(), &device)) };
            (0..nbatch).into_par_iter().for_each(|p| {
                let cderi_iaux = cderi_batch.i((.., .., p));
                let cderi_half_iaux = cderi_half.i((.., .., p));
                let mut cderi_half_iaux = unsafe { cderi_half_iaux.force_mut() };
                cderi_half_iaux.matmul_from(&cderi_iaux, &dm, 1.0, 0.0);
            });
            // accumulate K_s += (M_P dm) * M_P^T over this batch
            // -- accumulate-k -- //
            let cderi_half = cderi_half.into_shape([nao, nao * nbatch]);
            let cderi_batch = cderi_batch.reshape([nao, nao * nbatch]);
            ks.i_mut((.., .., iset)).matmul_from(&cderi_half, &cderi_batch.t(), 1.0, 1.0);
        }
    });
    ks
}

/* #endregion ri-vk incore dm */

/* #region ri-vk incore coeff pair */

/// Generate Exchange (K) matrix using RI incore method from a pair of
/// "occupied-side" MO coefficient blocks.
///
/// Computes (per set $\mathbb{A}$)
///
/// $$ K^{\mathbb{A}}_{\mu\nu} = \sum_{P,\,i\le k} (M_P\, c_{\mathrm{left},\mathbb{A}})_{\mu i}\,(M_P\, c_{\mathrm{right}})_{\nu i} $$
///
/// i.e. the exchange of the factorized density $P^{\mathbb{A}} = c_{\mathrm{left},\mathbb{A}}\, c_{\mathrm{right}}^{\mathrm{T}}$
/// without materializing it. This generalizes [`get_vk_ri_incore_coeff`]
/// (the $c_{\mathrm{left}} = c_{\mathrm{right}}$ case) to two distinct coefficient
/// matrices. The primary use case is AO-mode TDDFT, where the transition
/// density factorizes exactly as $P^{\mathbb{A}} = (C_{vir} X_{\mathbb{A}}^{\mathrm{T}})\, C_{occ}^{\mathrm{T}}$
/// (rank $\le n_\mathrm{occ}$), so the exchange costs $O(n_\mathrm{aux} n_\mathrm{ao}^2 n_\mathrm{occ})$
/// instead of $O(n_\mathrm{aux} n_\mathrm{ao}^3)$.
///
/// # Parameters
///
/// - `cderi`: [`TsrView<f64>`]
///
///   - Cholesky decomposed 3c-2e ERI in shape (nao_tp, naux), f-contiguous, AO basis.
///
/// - `c_left`: [`TsrView<f64>`]
///
///   - Batch of left coefficient blocks, shape (nao, k, nset), f-contiguous, AO basis.
///
/// - `c_right`: [`TsrView<f64>`]
///
///   - Single right coefficient block, shape (nao, k), f-contiguous, AO basis.
///   - Held fixed across the set index, so its half-transform is computed once
///     per auxiliary batch and reused for all sets.
///
/// - `batch_size`: `usize`
///
///   - Batch size for auxiliary basis partitioning. This value controls memory usage.
///
/// # Returns
///
/// - [`Tsr<f64>`]
///
///   - Exchange (K) matrices in shape (nao, nao, nset), f-contiguous.
///   - Not symmetric in general ($c_{\mathrm{left}} \ne c_{\mathrm{right}}$).
pub fn get_vk_ri_incore_coeff_pair(
    cderi: TsrView<f64>,
    c_left: TsrView<f64>,
    c_right: TsrView<f64>,
    batch_size: usize,
) -> Tsr<f64> {
    assert_eq!(cderi.ndim(), 2, "Cholesky ERI must have 2 dimensions");
    assert_eq!(c_left.ndim(), 3, "c_left must have 3 dimensions (nao, k, nset)");
    assert_eq!(c_right.ndim(), 2, "c_right must have 2 dimensions (nao, k)");

    // get shapes
    let nao = c_right.shape()[0];
    let k = c_right.shape()[1];
    let nset = c_left.shape()[2];
    let naux = cderi.shape()[1];
    let nao_tp = (nao + 1) * nao / 2;
    let device = cderi.device().clone();

    // shape check
    assert_eq!(cderi.shape(), &[nao_tp, naux], "Cholesky ERI must have shape (nao_tp, naux)");
    assert_eq!(c_left.shape()[0], nao, "c_left rows must match nao");
    assert_eq!(c_left.shape()[1], k, "c_left column count must match c_right");
    assert_eq!(c_right.shape(), &[nao, k]);

    // initialize vk as result
    let mut ks = rt::zeros(([nao, nao, nset].f(), &device));

    // process each auxiliary batch
    (0..naux).step_by(batch_size).for_each(|iaux| {
        let nbatch = if iaux + batch_size <= naux { batch_size } else { naux - iaux };

        // unpack cderi once for this batch: cderi_batch (nao, nao, nbatch)
        let cderi_batch: Tsr<f64> = unsafe { rt::empty(([nao, nao, nbatch].f(), &device)) };
        (0..nbatch).into_par_iter().for_each(|p| {
            let cderi_iaux = cderi.i((.., iaux + p)).unpack_tri(Upper, FlagSymm::Sy);
            let dst = cderi_batch.i((.., .., p));
            let mut dst = unsafe { dst.force_mut() };
            dst.assign(&cderi_iaux);
        });

        // right half-transform, once per batch: yl = M_P * c_right (amortized over sets)
        // -- right-half-transform -- //
        let yl = unsafe { rt::empty(([nao, k, nbatch].f(), &device)) };
        (0..nbatch).into_par_iter().for_each(|p| {
            let m_p = cderi_batch.i((.., .., p));
            let yl_p = yl.i((.., .., p));
            let mut yl_p = unsafe { yl_p.force_mut() };
            yl_p.matmul_from(&m_p, &c_right, 1.0, 0.0);
        });
        let yl = yl.into_shape([nao, k * nbatch]);

        // left half-transform per set, then accumulate the outer product
        for iset in 0..nset {
            // left half-transform: yx = M_P * c_left[iset]
            // -- left-half-transform -- //
            let yx = unsafe { rt::empty(([nao, k, nbatch].f(), &device)) };
            (0..nbatch).into_par_iter().for_each(|p| {
                let m_p = cderi_batch.i((.., .., p));
                let yx_p = yx.i((.., .., p));
                let mut yx_p = unsafe { yx_p.force_mut() };
                yx_p.matmul_from(&m_p, &c_left.i((.., .., iset)), 1.0, 0.0);
            });
            // accumulate K_s += yx * yl^T over this batch
            // -- accumulate-k -- //
            let yx = yx.into_shape([nao, k * nbatch]);
            ks.i_mut((.., .., iset)).matmul_from(&yx, &yl.t(), 1.0, 1.0);
        }
    });
    ks
}

/* #endregion ri-vk incore coeff pair */

/* #region ri-vk incore low-rank (SVD) dm */

/// Default relative singular-value threshold for the low-rank exchange
/// [`get_vk_ri_incore_dm_lowrank`]. Singular values with `σ_i < tol·σ_max` are
/// dropped.

/// Generate a **low-rank (SVD-truncated) Exchange (K) matrix** using the RI
/// incore method, contracting the transition density `P = C_occ · z · C_virᵀ`
/// without ever forming the full `[nao, nao]` density matrix.
///
/// The amplitude matrix `z` (`[occ, vir]`) is decomposed by SVD:
/// `z = U · diag(σ) · Vᵀ`. Keeping only the `k` singular values above a
/// relative tolerance `σ_i ≥ svd_tol · σ_max` gives the rank-k factors
/// `C_left = C_occ·U_k` and `C_right = C_vir·V_k`, and the exchange becomes
///
///   K = Σ_Q M_Q·P·M_Q = Σ_Q (M_Q·C_left)·diag(σ_k)·(M_Q·C_right)ᵀ
///
/// with cost `O(naux·nao²·k)` instead of the exact `O(naux·nao³)`. Handles
/// non-symmetric transition densities exactly; the B-block exchange is obtained
/// by transposing `z` (or swapping the `C_occ`/`C_vir` roles).
///
/// # Parameters
///
/// - `cderi`: Cholesky decomposed 3c-2e ERI, shape `(nao_tp, naux)`, f-contiguous.
/// - `c_occ`: Occupied MO coefficients, shape `(nao, occ)`.
/// - `c_vir`: Virtual MO coefficients, shape `(nao, vir)`.
/// - `z`: Amplitude (excitation) matrix, shape `(occ, vir)`.
/// - `svd_tol`: Relative singular-value threshold; `σ_i ≥ svd_tol·σ_max` are
///   kept. `svd_tol <= 0` keeps all singular values (exact, up to SVD roundoff).
/// - `batch_size`: Auxiliary-basis batch size for memory control.
///
/// # Returns
///
/// - [`Tsr<f64>`]: Exchange matrix, shape `(nao, nao)`.
pub fn get_vk_ri_incore_dm_lowrank(
    cderi: TsrView<f64>,
    c_occ: TsrView<f64>,
    c_vir: TsrView<f64>,
    z: TsrView<f64>,
    svd_tol: f64,
    batch_size: usize,
) -> Tsr<f64> {
    let nao = c_occ.shape()[0];
    let occ = c_occ.shape()[1];
    let vir = c_vir.shape()[1];
    let naux = cderi.shape()[1];
    let nao_tp = (nao + 1) * nao / 2;
    let device = cderi.device().clone();

    assert_eq!(cderi.shape(), &[nao_tp, naux], "Cholesky ERI must have shape (nao_tp, naux)");
    assert_eq!(c_occ.shape(), &[nao, occ], "c_occ must have shape (nao, occ)");
    assert_eq!(c_vir.shape(), &[nao, vir], "c_vir must have shape (nao, vir)");
    assert_eq!(z.shape(), &[occ, vir], "z must have shape (occ, vir)");

    // ── Step 1: SVD of the amplitude matrix: z = U·Σ·Vᵀ ──
    let (u, s, vt) = rt::linalg::svd(z).into();
    let nsv = s.shape()[0];

    // ── Step 2: truncate to the singular values above svd_tol·σ_max ──
    let s_data: Vec<f64> = s.iter().copied().collect();
    let s_max = s_data.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let k = if svd_tol <= 0.0 || s_max <= 0.0 {
        nsv
    } else {
        s_data.iter().take_while(|&&s_i| s_i >= svd_tol * s_max).count()
    }.max(1);
    debug!("lowrank exchange: z [{occ}x{vir}] nsv={nsv} k={k} svd_tol={svd_tol} s1={:.3e} smax={:.3e}", s_data.first().copied().unwrap_or(0.0), s_max);

    // ── Step 3: rank-k factors C_left = C_occ·U_k, C_right = C_vir·V_kᵀ ──
    let u_k = u.i((.., ..k));
    let vt_k = vt.i((..k, ..));
    let s_k = rt::diag(&s.i(..k));
    let c_left = rt::matmul(c_occ, &u_k);          // [nao, k]
    let c_right = rt::matmul(c_vir, &vt_k.t());    // [nao, k]

    // ── Step 4: per-aux-batch exchange accumulation ──
    // K = Σ_Q M_Q·P·M_Q = Σ_Q (M_Q·C_left)·Σ_k·(M_Q·C_right)ᵀ
    let mut ks = rt::zeros(([nao, nao].f(), &device));
    (0..naux).step_by(batch_size).for_each(|iaux| {
        let nbatch = if iaux + batch_size <= naux { batch_size } else { naux - iaux };

        // unpack cderi for this batch: (nao, nao, nbatch)
        let cderi_batch: Tsr<f64> = unsafe { rt::empty(([nao, nao, nbatch].f(), &device)) };
        (0..nbatch).into_par_iter().for_each(|p| {
            let cderi_iaux = cderi.i((.., iaux + p)).unpack_tri(Upper, FlagSymm::Sy);
            let cderi_iaux_mut = cderi_batch.i((.., .., p));
            let mut cderi_iaux_mut = unsafe { cderi_iaux_mut.force_mut() };
            cderi_iaux_mut.assign(&cderi_iaux);
        });

        // per aux (parallel): K_p = (M_Q·C_left)·Σ_k·(M_Q·C_right)ᵀ, [nao, nao]
        let k_batch: Tsr<f64> = unsafe { rt::empty(([nao, nao, nbatch].f(), &device)) };
        (0..nbatch).into_par_iter().for_each(|p| {
            let m_q = cderi_batch.i((.., .., p));
            let x_q = rt::matmul(&m_q, &c_left);       // [nao, k]
            let y_q = rt::matmul(&m_q, &c_right);      // [nao, k]
            let k_q = rt::matmul(&rt::matmul(&x_q, &s_k), &y_q.t()); // [nao, nao]
            let k_q_view = k_batch.i((.., .., p));
            let mut k_q_mut = unsafe { k_q_view.force_mut() };
            k_q_mut.assign(&k_q);
        });

        // accumulate batch into K (serial, cheap)
        for p in 0..nbatch {
            let k_q = k_batch.i((.., .., p));
            *&mut ks.i_mut((.., ..)) += &k_q;
        }
    });
    ks
}

/* #endregion ri-vk incore low-rank (SVD) dm */

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random fill.
    fn pseudo(n: usize, seed: f64) -> Vec<f64> {
        (0..n).map(|i| ((i as f64 + seed) * 0.37).sin() * 0.5 + 0.25).collect()
    }

    /// Build a synthetic (nao_tp, naux) folded cderi + MO coeffs.
    fn synthetic_set(nao: usize, occ: usize, vir: usize, naux: usize) -> (Tsr<f64>, Tsr<f64>, Tsr<f64>, Tsr<f64>, usize) {
        let device = DeviceBLAS::default();
        let nao_tp = nao * (nao + 1) / 2;
        let cderi = rt::asarray((pseudo(nao_tp * naux, 1.7), [nao_tp, naux].f(), &device));
        let c_occ = rt::asarray((pseudo(nao * occ, 2.1), [nao, occ].f(), &device));
        let c_vir = rt::asarray((pseudo(nao * vir, 3.3), [nao, vir].f(), &device));
        // low-rank z: z = A·Bᵀ with A [occ,r], B [vir,r]
        let r = 2;
        let a = rt::asarray((pseudo(occ * r, 4.1), [occ, r].f(), &device));
        let b = rt::asarray((pseudo(vir * r, 5.2), [vir, r].f(), &device));
        let z = rt::matmul(&a, &b.t());
        (cderi, c_occ, c_vir, z, nao_tp)
    }

    #[test]
    fn test_vk_ri_incore_dm_lowrank_matches_exact() {
        let nao = 6; let occ = 3; let vir = 4; let naux = 5;
        let (cderi, c_occ, c_vir, z, nao_tp) = synthetic_set(nao, occ, vir, naux);
        let device = cderi.device().clone();

        // exact reference: P = C_occ·z·C_virᵀ, K = get_vk_ri_incore_dm(cderi, [P])
        let p = rt::matmul(&rt::matmul(&c_occ, &z), &c_vir.t()); // [nao, nao]
        let p3 = rt::asarray((p.iter().copied().collect::<Vec<f64>>(), [nao, nao, 1].f(), &device));
        let k_exact = get_vk_ri_incore_dm(cderi.view(), p3.view(), 64);
        let k_exact = k_exact.i((.., .., 0));

        // low-rank with svd_tol=0 must reproduce the exact K (full SVD rank)
        let k_lr = get_vk_ri_incore_dm_lowrank(cderi.view(), c_occ.view(), c_vir.view(), z.view(), 0.0, 64);
        assert!(rt::allclose(k_lr.view(), k_exact, None), "low-rank (tol=0) != exact");

        // sanity: cderi shape check used internally
        assert_eq!(cderi.shape(), &[nao_tp, naux]);
    }

    #[test]
    fn test_vk_ri_incore_dm_lowrank_truncation() {
        // A truncated (small svd_tol > 0) result must not be far from exact for a
        // genuinely low-rank z (rank 2), since only negligible σ are dropped.
        let nao = 6; let occ = 3; let vir = 4; let naux = 5;
        let (cderi, c_occ, c_vir, z, _) = synthetic_set(nao, occ, vir, naux);
        let device = cderi.device().clone();
        let p = rt::matmul(&rt::matmul(&c_occ, &z), &c_vir.t());
        let p3 = rt::asarray((p.iter().copied().collect::<Vec<f64>>(), [nao, nao, 1].f(), &device));
        let k_exact_full = get_vk_ri_incore_dm(cderi.view(), p3.view(), 64);
        let k_exact = k_exact_full.i((.., .., 0));

        // Drop σ < 1e-4·σ_max. For rank-2 z, σ3/σ1 is tiny, so this keeps rank 2.
        let k_lr = get_vk_ri_incore_dm_lowrank(cderi.view(), c_occ.view(), c_vir.view(), z.view(), 1.0e-4, 64);
        assert!(rt::allclose(k_lr.view(), k_exact, None), "truncated low-rank != exact");
    }

    #[test]
    fn test_vk_ri_incore_coeff_pair_matches_exact() {
        let nao = 6; let k_dim = 3; let naux = 5; let nset = 2;
        let device = DeviceBLAS::default();
        let nao_tp = nao * (nao + 1) / 2;
        let cderi = rt::asarray((pseudo(nao_tp * naux, 1.7), [nao_tp, naux].f(), &device));
        // c_left differs per set; c_right shared (the multi-root TDDFT pattern)
        let c_left = rt::asarray((pseudo(nao * k_dim * nset, 6.3), [nao, k_dim, nset].f(), &device));
        let c_right = rt::asarray((pseudo(nao * k_dim, 7.9), [nao, k_dim].f(), &device));

        let k_pair = get_vk_ri_incore_coeff_pair(cderi.view(), c_left.view(), c_right.view(), 64);

        // exact reference: P_s = c_left_s · c_rightᵀ, K_s = get_vk_ri_incore_dm([P_s])
        for s in 0..nset {
            let cl = c_left.i((.., .., s));
            let p = rt::matmul(&cl, &c_right.t()); // [nao, nao]
            let p3 = rt::asarray((p.iter().copied().collect::<Vec<f64>>(), [nao, nao, 1].f(), &device));
            let k_exact_full = get_vk_ri_incore_dm(cderi.view(), p3.view(), 64);
            let k_exact = k_exact_full.i((.., .., 0));
            assert!(
                rt::allclose(k_pair.i((.., .., s)), k_exact, None),
                "coeff_pair set {s} != exact density-driven K"
            );
        }
    }
}
