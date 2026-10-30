//! Response (Fock and response matrix) implementation for RI-JK, restricted.
//!
//! The response object [`RRespRIJK`] is a standalone object holding the minimal state for the
//! fock/response functionality of the RI-JK electronic-interaction contribution; it is fully
//! independent of the skeleton-hessian machinery of `RHessRIJK` (which owns its own copy of the
//! factors and `cderi`).
//!
//! Fock generation routes through the pure in-core functions (`get_vj_ri_incore`,
//! `get_vk_ri_incore_dm`/`get_vk_ri_incore_coeff`) on the stored `cderi`, instead of any SCF-level
//! hamiltonian driver. The J/K factors are absorbed into the returned fock tensor, consistent with
//! how the hessian-side `get_deriv1_bra` applies the factors.
//!
//! The response-bra core [`get_rijk_response_bra_separated`] (shared by the RHF response object
//! and the UHF hessian's response path) also lives in this module.

use super::prelude_dev::*;
use crate::analdrv::prelude::*;
use crate::grad::rhf::pack_triu_tilde;
use crate::ri_jk::pure_incore::{get_vj_ri_incore, get_vk_ri_incore_coeff, get_vk_ri_incore_dm};
use crate::ri_jk::util::get_dm0_restricted;

/* #region response kernel intermediates */

/// Bra-independent intermediates of the RI-JK response kernel (the K part of
/// [`get_rijk_response_bra_separated`]), precomputed once per orbital state.
///
/// Every aux batch of the kernel loads the packed `cderi` slice, unpacks it to `[nao, nao, nbatch]`,
/// and half-transforms it with the occupied coefficients into `cderi_bxo [nao, nbatch, nocc]` and
/// `cderi_oxo [nocc, nbatch, nocc]`. None of these three objects depends on the `bra`, yet they are
/// rebuilt on every kernel call. The kernel is applied to `3 * natm` right-hand sides by an
/// analytic Hessian (where the rebuild is amortized), but to a single right-hand side by a Z-vector
/// solve (DH/PT2 relaxed multipoles, the CD-GW gradient), which iterates the kernel tens of times
/// for one column — there the rebuild dominates the call.
///
/// This struct holds the three objects per batch, so that the up-front pass is paid once and every
/// later call only performs the bra-dependent contractions. The unpacked batches are the memory-
/// dominant part (`nao^2 * naux` doubles); see [`build_rijk_response_intmd`] for the budget policy.
pub struct RijkRespIntmd {
    /// Aux-batch size the batches were built with; the kernel must use the same value.
    pub nbatch_aux: usize,
    /// Unpacked `cderi` batches, each `[nao, nao, nbatch]`, in aux order; empty when not cached.
    pub cderi_batches: Vec<Tsr>,
    /// Per-spin half-transforms (in the spin order of `mo_coeff`); empty when not cached.
    pub spins: Vec<RijkRespSpinIntmd>,
}

/// Per-spin part of [`RijkRespIntmd`]; one entry per aux batch, in aux order.
pub struct RijkRespSpinIntmd {
    /// Number of occupied orbitals of this spin (must match the kernel's `nocc`).
    pub nocc: usize,
    /// `cderi @ mocc`: `[nao, nbatch, nocc]` per batch.
    pub bxo: Vec<Tsr>,
    /// `mocc^T @ cderi_bxo`: `[nocc, nbatch, nocc]` per batch.
    pub oxo: Vec<Tsr>,
}

impl RijkRespIntmd {
    /// An empty cache (nothing cached; the kernel falls back to its own computation).
    pub fn none(nbatch_aux: usize) -> Self {
        Self { nbatch_aux, cderi_batches: Vec::new(), spins: Vec::new() }
    }

    /// Memory held by the cache, in bytes.
    pub fn memory_bytes(&self) -> usize {
        let batch_bytes =
            |v: &Vec<Tsr>| v.iter().map(|x| x.shape().iter().product::<usize>() * std::mem::size_of::<f64>()).sum::<usize>();
        batch_bytes(&self.cderi_batches) + self.spins.iter().map(|s| batch_bytes(&s.bxo) + batch_bytes(&s.oxo)).sum::<usize>()
    }

    /// Whether any part of the cache is populated.
    pub fn is_active(&self) -> bool {
        !self.cderi_batches.is_empty() || self.spins.iter().any(|s| !s.bxo.is_empty())
    }
}

/// Build [`RijkRespIntmd`] for one orbital state (one or two spins).
///
/// `budget_bytes` caps the memory held by the cache; the two tiers are budgeted independently, so a
/// system whose unpacked `cderi` exceeds the budget still gets the (much smaller) half-transforms
/// cached:
/// - the unpacked `cderi` batches cost `nao^2 * naux` doubles,
/// - the half-transforms cost `(nao + nocc) * naux * nocc` doubles per spin.
///
/// A non-positive `budget_bytes` disables the cache entirely (the returned object is inactive).
#[allow(clippy::too_many_arguments)]
pub fn build_rijk_response_intmd(
    cderi: TsrView,
    mo_coeff: &[TsrView],
    mo_occ: &[TsrView],
    do_k: bool,
    nbatch_aux: usize,
    budget_bytes: f64,
) -> RijkRespIntmd {
    if !do_k || budget_bytes <= 0.0 {
        // The Coulomb response reads packed cderi directly; none of the
        // unpacked/half-transformed exchange cache is used for a pure DFA.
        return RijkRespIntmd::none(nbatch_aux);
    }
    let nset = mo_coeff.len();
    assert_eq!(mo_occ.len(), nset);
    let nao = mo_coeff[0].shape()[0];
    let naux = cderi.shape()[1];

    let mocc: Vec<Tsr> =
        (0..nset).map(|s| mo_coeff[s].view().bool_select(-1, &mo_occ[s].view().greater(0).into_vec())).collect();
    let nocc: Vec<usize> = mocc.iter().map(|m| m.shape()[1]).collect();

    let nbatch = naux.div_ceil(nbatch_aux.max(1));
    let bytes_of = |n: usize| n as f64 * std::mem::size_of::<f64>() as f64;
    let bytes_cderi = bytes_of(nao * nao * naux);
    let bytes_aux = if do_k { nocc.iter().map(|&n| bytes_of((nao + n) * naux * n)).sum() } else { 0.0 };
    let cache_cderi = bytes_cderi <= budget_bytes;
    let cache_aux = do_k && bytes_aux <= budget_bytes;
    if !cache_cderi && !cache_aux {
        return RijkRespIntmd::none(nbatch_aux);
    }

    let mut cderi_batches: Vec<Tsr> = Vec::with_capacity(if cache_cderi { nbatch } else { 0 });
    let mut spins: Vec<RijkRespSpinIntmd> = (0..nset)
        .map(|s| RijkRespSpinIntmd {
            nocc: nocc[s],
            bxo: Vec::with_capacity(if cache_aux { nbatch } else { 0 }),
            oxo: Vec::with_capacity(if cache_aux { nbatch } else { 0 }),
        })
        .collect();

    for iaux_start in (0..naux).step_by(nbatch_aux) {
        let iaux_end = (iaux_start + nbatch_aux).min(naux);
        let nbatch = iaux_end - iaux_start;
        let slc = rt::slice!(iaux_start, iaux_end);
        let batch = cderi.i((.., slc)).unpack_tri(Upper, FlagSymm::Sy);
        if cache_aux {
            for s in 0..nset {
                let bxo = (batch.reshape([nao, nao * nbatch]).t() % &mocc[s]).into_shape([nao, nbatch, nocc[s]]);
                let oxo = (&mocc[s].t() % bxo.reshape([nao, nbatch * nocc[s]])).into_shape([nocc[s], nbatch, nocc[s]]);
                spins[s].bxo.push(bxo.into_contig(ColMajor));
                spins[s].oxo.push(oxo.into_contig(ColMajor));
            }
        }
        if cache_cderi {
            cderi_batches.push(batch.into_contig(ColMajor));
        }
    }
    RijkRespIntmd { nbatch_aux, cderi_batches, spins }
}

/// Single-RHS exchange response (Z vectors). Parallelize independent auxiliary
/// batches, with direct BLAS inside each task. The general tensor path launches
/// several threaded products per batch, although a Z solve has only one RHS.
/// Keep the same batch partition and merge order for all worker counts.
fn exchange_response_single(
    cderi: TsrView,
    mocc: TsrView,
    bra: TsrView,
    nbatch_aux: usize,
    cache: Option<&RijkRespIntmd>,
    spin: usize,
) -> Tsr {
    use rayon::prelude::*;
    use rest_tensors::{MatrixFull, MatrixFullSlice};
    use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;
    let nao = mocc.shape()[0];
    let no = mocc.shape()[1];
    let naux = cderi.shape()[1];
    let nt = nao * (nao + 1) / 2;
    let device = cderi.device().clone();
    let cderi = cderi.to_contig(ColMajor);
    let mocc = mocc.to_contig(ColMajor);
    let bra = bra.to_contig(ColMajor);
    let packed = &cderi.raw()[cderi.offset()..cderi.offset() + nt * naux];
    let co = &mocc.raw()[mocc.offset()..mocc.offset() + nao * no];
    let u = &bra.raw()[bra.offset()..bra.offset() + nao * no];
    // Only slices cross the Rayon boundary; the RSTSR device stays on the caller.
    let buffers: Vec<_> = (0..naux.div_ceil(nbatch_aux)).map(|b| {
        fn slice(t: &Tsr) -> &[f64] { &t.raw()[t.offset()..t.offset() + t.shape().iter().product::<usize>()] }
        let full = cache.and_then(|c| c.cderi_batches.get(b)).map(slice);
        let sp = cache.map(|c| &c.spins[spin]);
        let bxo = sp.and_then(|c| c.bxo.get(b)).map(slice);
        let oxo = sp.and_then(|c| c.oxo.get(b)).map(slice);
        (full, bxo, oxo)
    }).collect();
    // Bound live unpacking/intermediate scratch independently of pool size.
    // One batch is unavoidable; additional concurrent batches share 256 MiB.
    let scratch = 8 * nbatch_aux * (nao * nao + 2 * nao * no + 2 * no * no);
    let wave = ((256 * 1024 * 1024) / scratch.max(1)).max(1);
    let parts: Vec<Vec<f64>> = buffers.chunks(wave).enumerate().flat_map(|(w, buffers)| {
        buffers.par_iter().enumerate().map(|(b, &(full, bxo, oxo))| {
        let batch = w * wave + b;
        let p0 = batch * nbatch_aux;
        let nb = nbatch_aux.min(naux - p0);
        let gemm = |a: &[f64], ar, ac, ta, b: &[f64], br, bc, tb, out: &mut MatrixFull<f64>, alpha, beta| {
            let a = MatrixFullSlice { size: &[ar, ac], indicing: &[1, ar], data: a };
            let b = MatrixFullSlice { size: &[br, bc], indicing: &[1, br], data: b };
            _dgemm_full(&a, ta, &b, tb, out, alpha, beta);
        };
        let mut unpacked = Vec::new();
        let full = full.unwrap_or_else(|| {
            unpacked.resize(nao * nao * nb, 0.0);
            for p in 0..nb {
                let mut k = (p0 + p) * nt;
                for j in 0..nao { for i in 0..=j {
                    let v = packed[k]; k += 1;
                    unpacked[p * nao * nao + i + j * nao] = v;
                    unpacked[p * nao * nao + j + i * nao] = v;
                }}
            }
            &unpacked
        });
        let mut b_owned = MatrixFull::new([nao * nb, if bxo.is_none() { no } else { 0 }], 0.0);
        if bxo.is_none() {
            gemm(full, nao, nao * nb, 'T', co, nao, no, 'N', &mut b_owned, 1.0, 0.0);
        }
        let bxo = bxo.unwrap_or(&b_owned.data);
        let mut o_owned = MatrixFull::new([no, if oxo.is_none() { nb * no } else { 0 }], 0.0);
        if oxo.is_none() {
            gemm(co, nao, no, 'T', bxo, nao, nb * no, 'N', &mut o_owned, 1.0, 0.0);
        }
        let oxo = oxo.unwrap_or(&o_owned.data);
        let mut bx = MatrixFull::new([nao * nb, no], 0.0);
        gemm(full, nao, nao * nb, 'T', u, nao, no, 'N', &mut bx, 1.0, 0.0);
        let mut out = MatrixFull::new([nao, no], 0.0);
        gemm(&bx.data, nao, nb * no, 'N', oxo, no, nb * no, 'T', &mut out, -1.0, 0.0);
        let mut ox = MatrixFull::new([no, nb * no], 0.0);
        gemm(co, nao, no, 'T', &bx.data, nao, nb * no, 'N', &mut ox, 1.0, 0.0);
        gemm(bxo, nao, nb * no, 'N', &ox.data, no, nb * no, 'T', &mut out, -1.0, 1.0);
        out.data
        }).collect::<Vec<_>>()
    }).collect();
    let mut out = vec![0.0; nao * no];
    for part in parts { for (v, p) in out.iter_mut().zip(part) { *v += p; } }
    rt::asarray((out, [nao, no, 1].f(), &device))
}

/* #endregion response kernel intermediates */

/* #region response */

/// Separated J/K response-bra core, shared by RHF and UHF.
///
/// # Shapes
///
/// - `cderi`: `[nao_tp, naux]`
/// - `mo_coeff[s]`: `[nao, nmo_s]`, `mo_occ[s]`: `[nmo_s]`, `bra[s]`: `[nao, nocc_s, ...]` (the
///   trailing dimensions, collectively `nprop`, must agree across spins)
///
/// # Returns
///
/// A tuple `(j_ao, k_bras)`:
/// - `j_ao`: `Option<Tsr>` of shape `[nao, nao, nprop]` — the **spin-independent** Coulomb response
///   operator in AO basis, built from the total density response `sum_s bra_s @ mocc_s.T` (already
///   carrying the internal factor `2.0` from the symmetric cderi contraction; the consumer applies
///   `factor_j` and the per-spin right half-transform `... @ mocc_s`). `None` if `do_j` is false.
/// - `k_bras`: `Vec<Tsr>` (one entry per spin) of shape `[nao, nocc_s, nprop]` — the same-spin
///   exchange response in bra form (already carrying its internal sign/scale; the consumer applies
///   `factor_k`). Empty if `do_k` is false.
///
/// # Convention notes
///
/// - J sees the **total** density response, so a single AO operator is produced and shared across
///   spins; this is why UHF can reuse the RHF J path verbatim.
/// - K is strictly same-spin; each spin's bra form is produced independently.
/// - The internal factors (`2.0` on J, the two-term symmetrized sum on K) match the existing RHF
///   optimized response; the per-method `factor_j` / `factor_k` and the RHF `0.5` vs UHF `1.0`
///   exchange prefactor are applied by the consumer, not here.
/// - `intmd` optionally supplies the bra-independent K intermediates precomputed by
///   [`build_rijk_response_intmd`]; it is used only when it matches this call (same aux batching,
///   same spin count and occupations), otherwise the intermediates are recomputed in place. The
///   result is identical either way.
#[allow(clippy::too_many_arguments)]
pub fn get_rijk_response_bra_separated(
    cderi: TsrView,
    mo_coeff: &[TsrView],
    mo_occ: &[TsrView],
    bra: &[TsrView],
    do_j: bool,
    do_k: bool,
    nbatch_aux: usize,
    intmd: Option<&RijkRespIntmd>,
) -> (Option<Tsr>, Vec<Tsr>) {
    // notes on shape
    // - cderi: [nao_tp, naux]
    // - mo_coeff[s]: [nao, nmo_s]
    // - mo_occ[s]: [nmo_s]
    // - bra[s]: [nao, nocc_s, ...]  (trailing dims collectively `nprop`, same across spins)

    let nset = mo_coeff.len();
    assert_eq!(mo_occ.len(), nset);
    assert_eq!(bra.len(), nset);
    assert!(nset >= 1);

    let nao = mo_coeff[0].shape()[0];
    let naux = cderi.shape()[1];
    let nao_tp = nao * (nao + 1) / 2;
    assert_eq!(cderi.shape()[0], nao_tp);
    let device = cderi.device().clone();

    // per-spin occupied coefficients and reshaped bras
    let mocc: Vec<Tsr> = (0..nset)
        .map(|s| {
            let occidx = mo_occ[s].view().greater(0).into_vec();
            mo_coeff[s].view().bool_select(-1, &occidx)
        })
        .collect();
    let nocc: Vec<usize> = mocc.iter().map(|m| m.shape()[1]).collect();
    let bra_shape_orig: Vec<Vec<usize>> = bra.iter().map(|b| b.shape().to_vec()).collect();
    let bra: Vec<Tsr> = (0..nset).map(|s| bra[s].view().reshape((nao, nocc[s], -1)).into_contig(ColMajor)).collect();
    let nprop = bra[0].shape()[2];
    for s in 0..nset {
        assert_eq!(bra[s].shape()[2], nprop, "bra trailing dim (nprop) must agree across spins");
    }

    let mut j_ao: Option<Tsr> = None;
    let mut k_bras: Vec<Tsr> = Vec::new();

    // --- J contribution (spin-independent, AO form, from total density response) --- //

    if do_j {
        // dm1_total = sum_s (bra_s @ mocc_s.T), then symmetrize; pack with tilde; the symmetric
        // cderi contraction carries the internal factor 2.0 (matches the RHF optimized response).
        let mut dm1: Tsr = rt::zeros(([nao, nao, nprop], &device));
        for s in 0..nset {
            dm1 += &bra[s] % &mocc[s].t();
        }
        let dm1 = &dm1 + &dm1.swapaxes(0, 1);
        let dm1_tp = pack_triu_tilde(dm1.view());
        let itm_j_aux = cderi.t() % &dm1_tp;
        // The internal 2.0 factor of the symmetric contraction is applied to the small
        // `[naux, nprop]` auxiliary vector, not to the full `[nao_tp, naux]` `cderi`: scaling the
        // latter materialises a copy of the whole integral matrix (tens of MB to GB) on **every**
        // kernel call, which dominates a one-column application (`nao^2 * naux` writes against
        // `naux * nprop` writes). Exact in floating point: doubling the left operand of every
        // product is the same as doubling each product and each partial sum.
        let resp_tp_j: Tsr = &cderi % (2.0 * &itm_j_aux);
        j_ao = Some(resp_tp_j.unpack_tri(Upper, FlagSymm::Sy));
    }

    // --- K contribution (same-spin, bra form, two symmetrized terms) --- //

    if do_k {
        // Accept the precomputed intermediates only when they describe exactly this call.
        let cache = intmd.filter(|c| {
            c.nbatch_aux == nbatch_aux
                && c.spins.len() == nset
                && c.spins.iter().zip(nocc.iter()).all(|(sp, &n)| sp.nocc == n)
        });
        for s in 0..nset {
            if nprop == 1 && nocc[s] > 0 {
                let k = exchange_response_single(cderi.view(), mocc[s].view(), bra[s].view(), nbatch_aux, cache, s);
                k_bras.push(k.into_shape(bra_shape_orig[s].clone()));
                continue;
            }
            let mocc_s = &mocc[s];
            let bra_s = &bra[s];
            let spin_cache = cache.map(|c| &c.spins[s]);
            let mut resp_bra_k: Tsr = rt::zeros_like(bra_s);
            for (ibatch, iaux_start) in (0..naux).step_by(nbatch_aux).enumerate() {
                let iaux_end = (iaux_start + nbatch_aux).min(naux);
                let slc = rt::slice!(iaux_start, iaux_end);
                // note: the following `naux` is the batch size, shadowing the outer one for brevity
                let naux = iaux_end - iaux_start;

                // - cderi: [nao, nao, naux]
                // - cderi_bxo: [nao, naux, nocc]
                // - cderi_oxo: [nocc, naux, nocc]
                // - cderi_box: [nao, nocc, naux]
                //
                // All three are bra-independent; take them from the cache when available, else
                // build them here (the `_owned` holders keep the freshly built ones alive).
                let cached_cderi = cache.and_then(|c| c.cderi_batches.get(ibatch));
                let cderi_owned =
                    if cached_cderi.is_none() { Some(cderi.i((.., slc)).unpack_tri(Upper, FlagSymm::Sy)) } else { None };
                let cderi = match cached_cderi {
                    Some(x) => x.view(),
                    None => cderi_owned.as_ref().unwrap().view(),
                };

                let cached_bxo = spin_cache.and_then(|sp| sp.bxo.get(ibatch));
                let bxo_owned = if cached_bxo.is_none() {
                    Some((cderi.reshape([nao, nao * naux]).t() % mocc_s).into_shape([nao, naux, nocc[s]]))
                } else {
                    None
                };
                let cderi_bxo = match cached_bxo {
                    Some(x) => x.view(),
                    None => bxo_owned.as_ref().unwrap().view(),
                };

                let cached_oxo = spin_cache.and_then(|sp| sp.oxo.get(ibatch));
                let oxo_owned = if cached_oxo.is_none() {
                    Some((mocc_s.t() % cderi_bxo.reshape([nao, naux * nocc[s]])).into_shape([nocc[s], naux, nocc[s]]))
                } else {
                    None
                };
                let cderi_oxo = match cached_oxo {
                    Some(x) => x.view(),
                    None => oxo_owned.as_ref().unwrap().view(),
                };

                for a in 0..nprop {
                    let bra_sa = bra_s.i((.., .., a));
                    let mut respka = resp_bra_k.i_mut((.., .., a));
                    // k contribution part 0: uPj, iPj -> ui
                    let cderi_bxo_1 =
                        (cderi.reshape([nao, nao * naux]).t() % &bra_sa).into_shape([nao, naux, nocc[s]]);
                    respka -=
                        cderi_bxo_1.reshape([nao, naux * nocc[s]]) % cderi_oxo.reshape([nocc[s], naux * nocc[s]]).t();
                    // k contribution part 1: uPj, iPj -> ui (i from mocc, j from bra)
                    let cderi_oxo_1 =
                        (mocc_s.t() % cderi_bxo_1.reshape([nao, naux * nocc[s]])).into_shape([nocc[s], naux, nocc[s]]);
                    respka -=
                        cderi_bxo.reshape([nao, naux * nocc[s]]) % cderi_oxo_1.reshape([nocc[s], naux * nocc[s]]).t();
                }
            }
            // restore original trailing shape for this spin's bra
            let mut shape = bra_shape_orig[s].clone();
            shape[0] = nao;
            k_bras.push(resp_bra_k.into_shape(shape));
        }
    }

    (j_ao, k_bras)
}

/* #endregion */

/// Response (fock/response matrix) object for RI-JK, restricted.
///
/// This object deliberately holds only what the response contractions need: the Cholesky
/// decomposed ERI, the J/K factors, and the response intermediates. It does **not** need
/// `mol`/`aux`/`j2c_decomp` — those belong to the skeleton-hessian machinery of the hessian
/// object.
///
/// Main-thread only (not `Send`/`Sync`).
pub struct RRespRIJK<'a> {
    /// Coulomb factor, absorbed into the returned fock/response tensors.
    pub factor_j: f64,
    /// Exchange factor, absorbed into the returned fock/response tensors.
    pub factor_k: f64,
    /// Cholesky decomposed 3c-2e ERI, shape `[nao_tp, naux]`.
    pub cderi: TsrCow<'a>,
    /// Response intermediates: `mo_coeff [nao, nmo]` (RowMajor) and `mo_occ [nmo]`, stored by
    /// [`RRespAPI::make_response_preparation`].
    pub intmd: HashMap<String, Tsr>,
    /// Aux-batch size of the response kernel, used both to build and to apply
    /// [`Self::resp_intmd`].
    pub nbatch_aux: usize,
    /// Whether [`RRespAPI::make_response_preparation`] should build [`Self::resp_intmd`].
    pub cache_kernel: bool,
    /// Memory budget (in bytes) of [`Self::resp_intmd`]; see [`build_rijk_response_intmd`].
    pub cache_budget: f64,
    /// Precomputed bra-independent kernel intermediates, built by
    /// [`RRespAPI::make_response_preparation`] and consumed by `get_response_bra`.
    pub resp_intmd: RijkRespIntmd,
    /// Wall time (seconds) spent building [`Self::resp_intmd`].
    pub cache_timing: f64,
}

impl<'a> RRespRIJK<'a> {
    /// Create from a Cholesky-decomposed ERI (`cderi`).
    ///
    /// The response-kernel cache is off by default (see [`Self::with_cache_kernel`]); it is an
    /// opt-in trade of memory for speed, enabled by the analysis driver through the CP-SCF config.
    pub fn new_with_cderi(factor_j: f64, factor_k: f64, cderi: TsrCow<'a>) -> Self {
        Self {
            factor_j,
            factor_k,
            cderi,
            intmd: HashMap::new(),
            nbatch_aux: 72,
            cache_kernel: false,
            cache_budget: 0.0,
            resp_intmd: RijkRespIntmd::none(72),
            cache_timing: 0.0,
        }
    }

    /// Enable (or disable) caching of the bra-independent response-kernel intermediates, and set
    /// its memory budget in bytes.
    pub fn with_cache_kernel(mut self, enabled: bool, budget_bytes: f64) -> Self {
        self.cache_kernel = enabled;
        self.cache_budget = budget_bytes;
        self
    }
}

impl<'a> AnalDrvBaseAPI for RRespRIJK<'a> {}

impl<'a> RRespAPI for RRespRIJK<'a> {
    /// Fock from density matrix: `factor_j * J(rdm) - 0.5 * factor_k * K(rdm)`.
    ///
    /// The `rdm` is assumed to be symmetric, as required by the pure in-core functions.
    fn get_fock_rdm(&mut self, rdm: TsrView) -> Tsr {
        assert_eq!(rdm.ndim(), 2, "rdm must have 2 dimensions");
        let nao = rdm.shape()[0];
        let device = self.cderi.device();

        // pure functions require (nao, nao, nset) f-contiguous inputs
        let dms = rdm.into_contig(ColMajor).into_shape((nao, nao, 1));

        // TODO: batch size `72` should be tunable by max-memory.
        let vj = get_vj_ri_incore(self.cderi.view(), dms.view()).i((.., .., 0)).into_contig(ColMajor);
        let vk = get_vk_ri_incore_dm(self.cderi.view(), dms.view(), 72).i((.., .., 0)).into_contig(ColMajor);

        let mut fock = rt::zeros(([nao, nao], device));
        if self.factor_j != 0.0 {
            fock += self.factor_j * &vj;
        }
        if self.factor_k != 0.0 {
            fock -= 0.5 * self.factor_k * &vk;
        }
        fock
    }

    /// Fock from molecular coefficients: J by the dm route (no coeff-native vj exists), K by the
    /// coeff-native pure function.
    fn get_fock_coeff(&mut self, mo_coeff: TsrView, mo_occ: TsrView) -> Tsr {
        let [nao, nmo] = mo_coeff.shape().to_vec().try_into().unwrap();
        let device = self.cderi.device();

        let dm = get_dm0_restricted(mo_coeff.view(), mo_occ.view());
        let dms = dm.into_contig(ColMajor).into_shape((nao, nao, 1));
        let vj = get_vj_ri_incore(self.cderi.view(), dms.view()).i((.., .., 0)).into_contig(ColMajor);

        let coeff = mo_coeff.into_contig(ColMajor).into_shape((nao, nmo, 1));
        let occ = mo_occ.into_contig(ColMajor).into_shape((nmo, 1));

        // TODO: batch size `72` should be tunable by max-memory.
        let vk =
            get_vk_ri_incore_coeff(self.cderi.view(), coeff.view(), occ.view(), 72).i((.., .., 0)).into_contig(ColMajor);

        let mut fock = rt::zeros(([nao, nao], device));
        if self.factor_j != 0.0 {
            fock += self.factor_j * &vj;
        }
        if self.factor_k != 0.0 {
            fock -= 0.5 * self.factor_k * &vk;
        }
        fock
    }

    fn get_response_rdm(&mut self, rdm: TsrView) -> Tsr {
        // assume rdm can be symmetrized, 4 * J - 2 * K
        assert_eq!(rdm.ndim(), 2, "rdm must have 2 dimensions");
        let [nao, nao2] = rdm.shape().to_vec().try_into().unwrap();
        assert_eq!(nao, nao2, "rdm must be square");
        let device = self.cderi.device();

        let rdm_sym = ((&rdm + &rdm.t()) * 0.5).into_contig(ColMajor).into_shape((nao, nao, 1));

        let mut resp = rt::zeros(([nao, nao], device));
        if self.factor_j != 0.0 {
            let vj = get_vj_ri_incore(self.cderi.view(), rdm_sym.view()).i((.., .., 0)).into_contig(ColMajor);
            resp += 4.0 * self.factor_j * &vj;
        }
        // TODO: batch size `72` should be tunable by max-memory.
        if self.factor_k != 0.0 {
            let vk = get_vk_ri_incore_dm(self.cderi.view(), rdm_sym.view(), 72).i((.., .., 0)).into_contig(ColMajor);
            resp -= 2.0 * self.factor_k * &vk;
        }
        resp
    }

    fn make_response_preparation(&mut self, mo_coeff: TsrView, mo_occ: TsrView) {
        // Build the bra-independent response-kernel intermediates. This is the only place where the
        // orbitals are known, and the kernel is re-applied on every CP-SCF iteration afterwards.
        if self.cache_kernel {
            let t0 = std::time::Instant::now();
            // `mo_coeff` / `mo_occ` are still owned here (they are stored below), so hand the
            // builder fresh views of them.
            let mo_coeff_v = mo_coeff.view();
            let mo_occ_v = mo_occ.view();
            self.resp_intmd = build_rijk_response_intmd(
                self.cderi.view(),
                &[mo_coeff_v],
                &[mo_occ_v],
                self.factor_k != 0.0,
                self.nbatch_aux,
                self.cache_budget,
            );
            self.cache_timing = t0.elapsed().as_secs_f64();
            if std::env::var("REST_RESP_KERNEL_TIMING").is_ok() {
                let mb = self.resp_intmd.memory_bytes() as f64 / 1024.0 / 1024.0;
                println!(
                    "[resp-kernel] cache built in {:.3} s (active: {}, memory: {:.1} MiB)",
                    self.cache_timing,
                    self.resp_intmd.is_active(),
                    mb
                );
            }
        } else {
            self.resp_intmd = RijkRespIntmd::none(self.nbatch_aux);
            self.cache_timing = 0.0;
        }

        self.intmd.insert("mo_coeff".to_string(), mo_coeff.into_contig(RowMajor));
        self.intmd.insert("mo_occ".to_string(), mo_occ.to_owned());
    }

    fn get_response_bra(&mut self, bra: TsrView) -> Tsr {
        let mo_coeff = self.intmd["mo_coeff"].view();
        let mo_occ = self.intmd["mo_occ"].view();
        let cderi = self.cderi.view();

        // RHF (single spin) assembly of the separated J/K response core.
        // - J (AO form, from total density) contracted with `mocc` and scaled by `factor_j`.
        // - K (same-spin bra form) scaled by `factor_k`; the core already bakes in the exchange sign.
        // - RHF exchange prefactor (occ = 2) is folded into `factor_k`, matching the naive convention.
        let shape_bra = bra.shape().to_vec();
        let nao = mo_coeff.shape()[0];
        let device = mo_coeff.device();
        let occidx = mo_occ.view().greater(0).into_vec();
        let mocc = mo_coeff.bool_select(-1, &occidx);
        let nocc = mocc.shape()[1];
        let nprop: usize = shape_bra[2..].iter().product();

        let (j_ao, k_bras) = get_rijk_response_bra_separated(
            cderi,
            &[mo_coeff.view()],
            &[mo_occ.view()],
            &[bra.view()],
            self.factor_j != 0.0,
            self.factor_k != 0.0,
            self.nbatch_aux,
            Some(&self.resp_intmd),
        );

        let mut resp: Tsr = rt::zeros(([nao, nocc, nprop], device));
        if let Some(resp_ao_j) = j_ao {
            resp += self.factor_j * (resp_ao_j % &mocc);
        }
        if let Some(k_bra) = k_bras.first() {
            // K bra is returned in the original trailing shape; flatten trailing dims to (nao, nocc, nprop).
            resp += self.factor_k * k_bra.view().reshape((nao, nocc, nprop));
        }
        resp.into_shape(shape_bra)
    }
}
