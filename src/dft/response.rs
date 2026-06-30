/// PySCF-style response function generator for RKS/RHF.
///
/// Matches PySCF scf/_response_functions.py:
///   - `_gen_rhf_response(singlet=None)` → AO-space Fock response `vind(dm1)`
///   - `gen_vind` via hessian/rhf.py → AO↔MO bridging
///
/// For HF: vind(dm1) = J[dm1] - 0.5*K[dm1]
/// For DFT: vind(dm1) = fxc[dm1] + J[dm1] - hyb*K[dm1]

use rest_tensors::{MatrixFull, MatrixUpper};
use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;
use crate::scf_io::{SCF, SCFType};
use crate::dft::num_int::{prepare_fxc_data, fxc_matvec_old,
    eval_ao_batch, eval_rho5_batch};
use crate::dft::libxc_itrf::eval_xc_eff;
use crate::dft::xc_deriv::XCType;
use std::sync::atomic::{AtomicU64, Ordering as AOrdering};

/// Global cumulative timing of compute_fxc_response_ao calls, in ns.
static FXC_TIMING_NS: AtomicU64 = AtomicU64::new(0);
/// Reset cumulative fxc timing counter (call before a block of fxc work).
pub fn reset_fxc_timing() {
    FXC_TIMING_NS.store(0, AOrdering::Relaxed);
    FXC_DGEMM1_NS.store(0, AOrdering::Relaxed);
    FXC_RHO1_NS.store(0, AOrdering::Relaxed);
    FXC_WV_NS.store(0, AOrdering::Relaxed);
    FXC_AOW_NS.store(0, AOrdering::Relaxed);
    FXC_DGEMM2_NS.store(0, AOrdering::Relaxed);
    FXC_HERMISUM_NS.store(0, AOrdering::Relaxed);
}
/// Read cumulative fxc timing in seconds, then reset.
pub fn read_fxc_timing_s() -> f64 {
    let ns = FXC_TIMING_NS.swap(0, AOrdering::Relaxed);
    ns as f64 * 1e-9
}

// Detailed sub-component timing (in ns). Accumulated across all blocks and RHS.
static FXC_DGEMM1_NS: AtomicU64 = AtomicU64::new(0);   // dm1 · ao_d[0]
static FXC_RHO1_NS: AtomicU64 = AtomicU64::new(0);     // ρ₁ contractions (element-wise)
static FXC_WV_NS: AtomicU64 = AtomicU64::new(0);       // wv = ρ₁·fxc·w kernel application
static FXC_AOW_NS: AtomicU64 = AtomicU64::new(0);      // aow = Σ ao_d[x]·wv[x]
static FXC_DGEMM2_NS: AtomicU64 = AtomicU64::new(0);   // ao_d[0] · aow^T
static FXC_HERMISUM_NS: AtomicU64 = AtomicU64::new(0); // vmat hermi-sum

/// Read all fxc sub-component timings as (name, seconds) pairs, then reset.
pub fn read_fxc_subtimings_s() -> Vec<(&'static str, f64)> {
    let grab = |a: &AtomicU64| a.swap(0, AOrdering::Relaxed) as f64 * 1e-9;
    vec![
        ("fxc: dgemm1 (dm·ao)",      grab(&FXC_DGEMM1_NS)),
        ("fxc: rho1 contractions",   grab(&FXC_RHO1_NS)),
        ("fxc: wv kernel apply",     grab(&FXC_WV_NS)),
        ("fxc: aow build",           grab(&FXC_AOW_NS)),
        ("fxc: dgemm2 (ao·aow)",     grab(&FXC_DGEMM2_NS)),
        ("fxc: hermi_sum",           grab(&FXC_HERMISUM_NS)),
    ]
}

/// Precomputed workspace for gen_vind: caches C_occ, C_vir slices.
pub struct VindWorkspace {
    pub c_occ: MatrixFull<f64>,
    pub c_vir: MatrixFull<f64>,
    pub c_frozen: MatrixFull<f64>,  // frozen occupied MOs (shape [nao, nfrozen])
    pub nao: usize,
    pub nocc: usize,
    pub nvir: usize,
    pub nfrozen: usize,
    pub dim: usize,
}

impl VindWorkspace {
    pub fn new(scf: &SCF, nocc: usize, nvir: usize, start_mo: usize, lumo: usize) -> Self {
        let nao = scf.mol.num_basis;
        let mo_coeff = &scf.eigenvectors[0];
        let dim = nocc * nvir;
        let nfrozen = start_mo;  // MOs 0..start_mo are frozen

        let mut c_occ = MatrixFull::new([nao, nocc], 0.0);
        let mut c_vir = MatrixFull::new([nao, nvir], 0.0);
        let mut c_frozen = MatrixFull::new([nao, nfrozen], 0.0);
        for j in 0..nocc { for i in 0..nao { c_occ[[i, j]] = mo_coeff[[i, start_mo + j]]; } }
        for j in 0..nvir { for i in 0..nao { c_vir[[i, j]] = mo_coeff[[i, lumo + j]]; } }
        for j in 0..nfrozen { for i in 0..nao { c_frozen[[i, j]] = mo_coeff[[i, j]]; } }

        VindWorkspace { c_occ, c_vir, c_frozen, nao, nocc, nvir, nfrozen, dim }
    }
}

/// Compute J in AO basis (upper triangular format).
pub(crate) fn compute_j_upper(
    scf: &SCF,
    dm_vec: &Vec<MatrixFull<f64>>,
) -> MatrixUpper<f64> {
    if let Some(ref _rimatr) = scf.rimatr {
        let vj = crate::scf_io::vj_upper_with_rimatr_sync(&scf.rimatr, dm_vec, 1, 1.0);
        if vj[0].size > 1 { return vj[0].clone(); }
    }
    if let Some(ref _ri3fn) = scf.ri3fn {
        let vj = crate::scf_io::vj_upper_with_ri_v(&scf.ri3fn, dm_vec, 1, 1.0);
        if vj[0].size > 1 { return vj[0].clone(); }
    }
    panic!("No RI tensor available for J computation");
}

/// Compute K in AO basis (upper triangular format).
pub(crate) fn compute_k_upper(
    scf: &SCF,
    dm_vec: &Vec<MatrixFull<f64>>,
) -> MatrixUpper<f64> {
    if let Some(ref _rimatr) = scf.rimatr {
        let vk = crate::scf_io::vk_upper_with_rimatr_use_dm_only_sync(&scf.rimatr, dm_vec, 1, 1.0);
        if vk[0].size > 1 { return vk[0].clone(); }
    }
    if let Some(ref _ri3fn) = scf.ri3fn {
        let vk = crate::scf_io::vk_upper_with_ri_v_use_dm_only_sync(&scf.ri3fn, dm_vec, 1, 1.0);
        if vk[0].size > 1 { return vk[0].clone(); }
    }
    panic!("No RI tensor available for K computation");
}

// ============================================================================
// PySCF-style fxc kernel caching for Hessian CP-HF
//
// Mirrors `cache_xc_kernel` + `nr_rks_fxc(fxc=...)` in PySCF: all
// ground-state-dependent quantities (AO values on grid, ρ₀, fxc kernel
// evaluated at ρ₀) are computed ONCE and reused across every CP-HF matvec.
// Only ρ₁ and the final contraction depend on dm1 and are computed per call.
// ============================================================================

/// Cached ground-state data for one grid block.
pub struct FxcBlock {
    pub nb: usize,
    pub weights: Vec<f64>,
    /// AO values + derivatives on this block; one [nao, nb] matrix per
    /// derivative component (0=ao, 1=ao_x, 2=ao_y, 3=ao_z for GGA).
    pub ao_d: Vec<MatrixFull<f64>>,
    /// fxc kernel for this block, layout `fxc_raw[g + x*nb + y*nvar*nb]`
    /// (= fxc[x,y,g] in column-major), already × weight baked out.
    /// Length: nb * nvar * nvar.
    pub fxc_raw: Vec<f64>,
}

/// PySCF `cache_xc_kernel` analog for the RKS Hessian fxc response.
///
/// Built once per Hessian calculation by `prepare_fxc_hessian_cache`, then
/// passed (by reference) to every `compute_fxc_response_ao_cached` call
/// inside the CP-HF Krylov loop. This avoids re-evaluating AO basis, ρ₀,
/// and the libxc fxc kernel on each matvec.
pub struct FxcHessianCache {
    pub nao: usize,
    pub nvar: usize,       // 1 (LDA) or 4 (GGA)
    pub blocks: Vec<FxcBlock>,
}

/// Build the Hessian fxc cache: iterates grid blocks once, evaluates AO +
/// derivatives, ρ₀, and the fxc kernel via `eval_xc_eff(deriv=2)`.
///
/// Matches the per-block work that the old `compute_fxc_response_ao` did on
/// every call. After this function returns, no further AO/libxc evaluation
/// is needed for the entire CP-HF phase.
pub fn prepare_fxc_hessian_cache(scf: &SCF) -> FxcHessianCache {
    let _t = std::time::Instant::now();
    let mol = &scf.mol;
    let grids = scf.grids.as_ref().expect("prepare_fxc_hessian_cache requires scf.grids");
    let nao = mol.num_basis;
    let func_ids = &mol.xc_data.dfa_compnt_scf;
    let func_factors = &mol.xc_data.dfa_paramr_scf;

    let xc_type = if mol.xc_data.use_density_gradient() {
        XCType::GGA
    } else {
        XCType::LDA
    };
    let nvar = match xc_type {
        XCType::LDA => 1,
        XCType::GGA => 4,
        _ => panic!("prepare_fxc_hessian_cache: only LDA and GGA supported"),
    };
    let ao_deriv = if nvar == 4 { 1 } else { 0 };
    let nderiv = (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) / 6;

    // Ground-state density (ρ₀) is invariant across matvecs — evaluate once.
    let mo_vec = vec![scf.eigenvectors[0].clone()];
    let occ_vec = vec![scf.occupation[0].clone()];

    let mut blocks: Vec<FxcBlock> = Vec::new();
    for block_range in &grids.parallel_balancing {
        let start = block_range.start;
        let end = block_range.end;
        let nb = end - start;
        if nb == 0 { continue; }
        let coords_block = &grids.coordinates[start..end];
        let weights_block = &grids.weights[start..end];

        // AO + derivatives for this block
        let ao = eval_ao_batch(mol, coords_block, ao_deriv, nb);
        let ao_d: Vec<MatrixFull<f64>> = (0..nderiv)
            .map(|d| {
                let view = ao.get_reducing_matrix(d).unwrap();
                MatrixFull::from_vec([nao, nb], view.iter().copied().collect()).unwrap()
            })
            .collect();

        // Ground-state ρ₀ → fxc kernel
        let rho_tensor = eval_rho5_batch(&ao, xc_type, &mo_vec, &occ_vec, 1, nb);
        let rho_array: Vec<f64> = {
            let raw = rho_tensor.raw();
            let off = rho_tensor.offset();
            raw[off..off + nb * nvar].to_vec()
        };
        let xc_tensors = eval_xc_eff(func_ids, func_factors, xc_type, 0, &rho_array, nb, 2);
        let fxc_t = xc_tensors[2].as_ref().expect("prepare_fxc_hessian_cache: fxc required");
        let fxc_raw_src = fxc_t.raw();
        let fxc_off = fxc_t.offset();
        let nv2 = nvar * nvar;
        let mut fxc_raw = vec![0.0; nb * nv2];
        // Layout: fxc_t stores fxc[g + k*nb] for k in 0..nv2 (matches old usage).
        // We copy this layout verbatim so compute_fxc_response_ao_cached can
        // index identically to the old code (`g + x*nb + y*nvar*nb`).
        for k in 0..nv2 {
            for g in 0..nb {
                fxc_raw[g + k * nb] = fxc_raw_src[fxc_off + g + k * nb];
            }
        }

        blocks.push(FxcBlock {
            nb,
            weights: weights_block.to_vec(),
            ao_d,
            fxc_raw,
        });
    }

    println!("  FxcHessianCache prepared: nao={}, nvar={}, blocks={} ({} grids) in {:.3}s",
             nao, nvar, blocks.len(),
             blocks.iter().map(|b| b.nb).sum::<usize>(),
             _t.elapsed().as_secs_f64());

    FxcHessianCache { nao, nvar, blocks }
}

/// AO-basis fxc response using a precomputed `FxcHessianCache`.
///
/// Equivalent to the old `compute_fxc_response_ao(scf, dm1)` but only does
/// the dm1-dependent work: ρ₁ = ao·dm1·ao contraction, apply fxc kernel,
/// then back-projection. The AO evaluation, ρ₀, and libxc fxc kernel eval
/// are all read from the cache. Grid blocks are processed in parallel with
/// rayon (the per-block partial `vmat` is reduced by summation).
pub fn compute_fxc_response_ao_cached(
    cache: &FxcHessianCache,
    dm1: &MatrixFull<f64>,
) -> MatrixFull<f64> {
    use rayon::prelude::*;
    let _t_fxc = std::time::Instant::now();
    let nao = cache.nao;
    let nvar = cache.nvar;

    let vmat = cache.blocks.par_iter()
        .map(|block| compute_fxc_response_block(block, dm1, nao, nvar))
        .reduce(|| MatrixFull::new([nao, nao], 0.0), |mut a, b| {
            a += b.clone();
            a
        });

    FXC_TIMING_NS.fetch_add(_t_fxc.elapsed().as_nanos() as u64, AOrdering::Relaxed);
    vmat
}

/// Batched AO-basis fxc response for multiple density matrices.
///
/// Phase A optimization: processes all RHS in a single par_iter dispatch
/// over grid blocks. Each block visits the cached AO data once and the
/// cached fxc kernel once, then loops over RHS to compute ρ₁/wv/aow per
/// RHS. This avoids N_RHS separate rayon fork/join overhead and improves
/// cache locality on the AO arrays.
///
/// Returns one [nao, nao] response matrix per input dm.
pub fn compute_fxc_response_ao_cached_batched(
    cache: &FxcHessianCache,
    dms: &[MatrixFull<f64>],
) -> Vec<MatrixFull<f64>> {
    use rayon::prelude::*;
    let _t_fxc = std::time::Instant::now();
    let nao = cache.nao;
    let nvar = cache.nvar;
    let n_rhs = dms.len();
    if n_rhs == 0 {
        return Vec::new();
    }

    // Each parallel task processes one block, returning a flat Vec<f64> of
    // length n_rhs*nao*nao (one partial response per RHS, concatenated).
    // Reduce sums across blocks element-wise.
    let summed: Vec<f64> = cache.blocks
        .par_iter()
        .map(|block| {
            let mut flat = vec![0.0; n_rhs * nao * nao];
            for i in 0..n_rhs {
                let v_partial = compute_fxc_response_block(block, &dms[i], nao, nvar);
                let off = i * nao * nao;
                for k in 0..nao * nao { flat[off + k] = v_partial.data[k]; }
            }
            flat
        })
        .reduce(
            || vec![0.0; n_rhs * nao * nao],
            |mut acc, block_flat| {
                for k in 0..acc.len() { acc[k] += block_flat[k]; }
                acc
            },
        );

    // Unflatten into Vec<MatrixFull>.
    let mut results = Vec::with_capacity(n_rhs);
    for i in 0..n_rhs {
        let off = i * nao * nao;
        results.push(MatrixFull::from_vec([nao, nao], summed[off..off + nao * nao].to_vec()).unwrap());
    }

    FXC_TIMING_NS.fetch_add(_t_fxc.elapsed().as_nanos() as u64, AOrdering::Relaxed);
    results
}

/// Per-grid-block fxc response: returns the partial [nao, nao] contribution
/// from this block. Pure (no shared mutable state) so safe to call in parallel.
fn compute_fxc_response_block(
    block: &FxcBlock,
    dm1: &MatrixFull<f64>,
    nao: usize,
    nvar: usize,
) -> MatrixFull<f64> {
    let nb = block.nb;
    let ao_d = &block.ao_d;
    let fxc_raw = &block.fxc_raw;
    let mut vmat = MatrixFull::new([nao, nao], 0.0);
    // Subtimings are opt-in (env var) to avoid atomic-cache-line contention
    // that otherwise costs ~3x wall time on small systems.
    let do_timing = std::env::var("REST_FXC_PROFILE").is_ok();
    let now = || if do_timing { Some(std::time::Instant::now()) } else { None };
    let acc = |t0: Option<std::time::Instant>, counter: &AtomicU64| {
        if let Some(t) = t0 {
            counter.fetch_add(t.elapsed().as_nanos() as u64, AOrdering::Relaxed);
        }
    };

    // ρ₁[μ,g] = Σ_ν dm1[μ,ν] · ao[0][ν,g]
    let mut c0 = MatrixFull::new([nao, nb], 0.0);
    let t = now();
    _dgemm_full(dm1, 'N', &ao_d[0], 'N', &mut c0, 1.0, 0.0);
    acc(t, &FXC_DGEMM1_NS);

    if nvar == 1 {
        // LDA
        let mut aow = MatrixFull::new([nao, nb], 0.0);
        let t = now();
        for g in 0..nb {
            let mut rho1 = 0.0;
            for mu in 0..nao { rho1 += ao_d[0][[mu, g]] * c0[[mu, g]]; }
            let wf_rho = block.weights[g] * fxc_raw[g] * rho1;
            for mu in 0..nao { aow[[mu, g]] = ao_d[0][[mu, g]] * wf_rho; }
        }
        acc(t, &FXC_RHO1_NS);
        let t = now();
        _dgemm_full(&aow, 'N', &ao_d[0], 'T', &mut vmat, 1.0, 1.0);
        acc(t, &FXC_DGEMM2_NS);
    } else {
        // GGA — nvar == 4
        let t = now();
        let mut rho1 = vec![0.0; 4 * nb];
        for g in 0..nb {
            let mut r0 = 0.0;
            for mu in 0..nao { r0 += ao_d[0][[mu, g]] * c0[[mu, g]]; }
            rho1[0 + g * 4] = r0;
        }
        for x in 1..4 {
            for g in 0..nb {
                let mut rx = 0.0;
                for mu in 0..nao { rx += ao_d[x][[mu, g]] * c0[[mu, g]]; }
                rho1[x + g * 4] = 2.0 * rx;
            }
        }
        acc(t, &FXC_RHO1_NS);

        let t = now();
        let mut wv = vec![0.0; 4 * nb];
        for g in 0..nb {
            let w = block.weights[g];
            for x in 0..4 {
                let mut acc_w = 0.0;
                for y in 0..4 {
                    let fxc_xy = fxc_raw[g + x * nb + y * 4 * nb];
                    acc_w += rho1[y + g * 4] * fxc_xy;
                }
                let mut val = w * acc_w;
                if x == 0 { val *= 0.5; }
                wv[x + g * 4] = val;
            }
        }
        acc(t, &FXC_WV_NS);

        let t = now();
        let mut aow = MatrixFull::new([nao, nb], 0.0);
        for g in 0..nb {
            for mu in 0..nao {
                let mut v = 0.0;
                for x in 0..4 { v += ao_d[x][[mu, g]] * wv[x + g * 4]; }
                aow[[mu, g]] = v;
            }
        }
        acc(t, &FXC_AOW_NS);

        let t = now();
        let mut m_block = MatrixFull::new([nao, nao], 0.0);
        _dgemm_full(&ao_d[0], 'N', &aow, 'T', &mut m_block, 1.0, 0.0);
        acc(t, &FXC_DGEMM2_NS);

        let t = now();
        for mu in 0..nao {
            for nu in 0..nao {
                vmat[[mu, nu]] += m_block[[mu, nu]] + m_block[[nu, mu]];
            }
        }
        acc(t, &FXC_HERMISUM_NS);
    }

    vmat
}


/// Optimized fvind: z (MO VO-block) → G(z) = C_vir^T@(J-0.5*K)@C_occ (+ fxc).
///
/// Uses precomputed VindWorkspace to avoid redundant slices and DGEMM for projection.
///
/// `fxc_cache`: when `Some`, adds the RKS fxc kernel response using the
/// precomputed `FxcHessianCache` (PySCF `cache_xc_kernel` analog). Pass
/// `None` for HF or when no XC contribution is desired.
pub fn gen_vind_opt(
    scf: &SCF,
    ws: &VindWorkspace,
    z_vo: &[f64],  // active VO block, flat [i + a*nocc] (nvir * nocc)
    fxc_cache: Option<&FxcHessianCache>,
    z_oo: Option<&[f64]>,  // occ-occ block, flat [i + j*nocc]
    z_fo: Option<&[f64]>,  // frozen-occ block, flat [i + k*nocc] (nfrozen * nocc)
) -> Vec<f64> {
    // Returns: [frozen_response (nfrozen*nocc), VO_response (nvir*nocc)] as single flat vec
    let nao = ws.nao;
    let nocc = ws.nocc;
    let nvir = ws.nvir;
    let nfrozen = ws.nfrozen;
    let dim = ws.dim;

    // ── Step 1: Build AO density matrix: VO contribution ──
    let mut z_scaled = MatrixFull::new([nvir, nocc], 0.0);
    for a in 0..nvir { for i in 0..nocc { z_scaled[[a, i]] = 2.0 * z_vo[i + a * nocc]; } }

    // dp1 = C_vir @ (2*z) @ C_occ^T
    let mut t1 = MatrixFull::new([nao, nocc], 0.0);
    _dgemm_full(&ws.c_vir, 'N', &z_scaled, 'N', &mut t1, 1.0, 0.0);
    let mut dp1 = MatrixFull::new([nao, nao], 0.0);
    _dgemm_full(&t1, 'N', &ws.c_occ, 'T', &mut dp1, 1.0, 0.0);

    // dm1 = dp1 + dp1^T  (dp2 ≡ dp1^T mathematically: the VO + OV contributions)
    // Skip the 2 extra GEMMs for t2/dp2 and the temporary allocation.
    let mut dm1 = MatrixFull::new([nao, nao], 0.0);
    for i in 0..nao { for j in 0..nao {
        dm1[[i, j]] = dp1[[i, j]] + dp1[[j, i]];
    }}

    // Occ-occ block contribution: dm1 += 2*C_occ @ z_oo @ C_occ^T + h.c.
    if let Some(z_oo) = z_oo {
        // z_oo_scaled[i, j] = 2.0 * z_oo[i + j * nocc]  (occ, occ) in column-major
        let mut z_oo_scaled = MatrixFull::new([nocc, nocc], 0.0);
        for i in 0..nocc { for j in 0..nocc {
            z_oo_scaled[[i, j]] = 2.0 * z_oo[i + j * nocc];
        }}
        // dp3 = C_occ @ (2*z_oo) @ C_occ^T
        let mut t3 = MatrixFull::new([nao, nocc], 0.0);
        _dgemm_full(&ws.c_occ, 'N', &z_oo_scaled, 'N', &mut t3, 1.0, 0.0);
        let mut dp3 = MatrixFull::new([nao, nao], 0.0);
        _dgemm_full(&t3, 'N', &ws.c_occ, 'T', &mut dp3, 1.0, 0.0);
        // dp3 is symmetric (z_oo_scaled may not be, so add dp3 + dp3^T)
        for i in 0..nao { for j in 0..nao {
            dm1[[i, j]] += dp3[[i, j]] + dp3[[j, i]];
        }}
    }

    // Frozen-occ contribution: dm1 += 2*C_frozen @ z_fo @ C_occ^T + h.c.
    if let Some(z_fo) = z_fo {
        let mut z_fo_scaled = MatrixFull::new([nfrozen, nocc], 0.0);
        for k in 0..nfrozen { for i in 0..nocc { z_fo_scaled[[k, i]] = 2.0 * z_fo[i + k * nocc]; }}
        let mut t4 = MatrixFull::new([nao, nocc], 0.0);
        _dgemm_full(&ws.c_frozen, 'N', &z_fo_scaled, 'N', &mut t4, 1.0, 0.0);
        let mut dp4 = MatrixFull::new([nao, nao], 0.0);
        _dgemm_full(&t4, 'N', &ws.c_occ, 'T', &mut dp4, 1.0, 0.0);
        let mut t5 = MatrixFull::new([nao, nfrozen], 0.0);
        _dgemm_full(&ws.c_occ, 'N', &z_fo_scaled, 'T', &mut t5, 1.0, 0.0);
        let mut dp5 = MatrixFull::new([nao, nao], 0.0);
        _dgemm_full(&t5, 'N', &ws.c_frozen, 'T', &mut dp5, 1.0, 0.0);
        for i in 0..nao { for j in 0..nao { dm1[[i, j]] += dp4[[i, j]] + dp5[[i, j]]; }}
    }

    let dm_vec = vec![dm1.clone()];

    // ── Step 2: Compute J, K via REST JK ──
    let j_full = compute_j_upper(scf, &dm_vec).to_matrixfull()
        .unwrap_or_else(|| panic!("J to_matrixfull failed"));
    let k_full = compute_k_upper(scf, &dm_vec).to_matrixfull()
        .unwrap_or_else(|| panic!("K to_matrixfull failed"));

    // ── Step 3: v_ao = J - 0.5*hyb*K (RHF) or J - hyb*K (UHF/ROHF) ──
    // hyb from DFA: for HF (dfa_compnt_scf empty), dfa_hybrid_scf=0, but
    // the SCF HF uses hardcoded scaling=-0.5 (generate_hf_hamiltonian_ri_v),
    // so we must set hyb=1.0 for HF to get the correct exchange response.
    let is_hf = scf.mol.xc_data.dfa_compnt_scf.is_empty();
    let hyb = if is_hf { 1.0 } else { scf.mol.xc_data.dfa_hybrid_scf };
    // dm1 = 2*dmbare in gen_vind_opt → J and K are linear, so J[2*dm] = 2*J[dmbare]
    // and K[2*dm] = 2*K[dmbare]. The SCF response needs 2*J - K, so:
    //   v_ao = 2*J[dmbare] - hyb*K[dmbare] = J[2*dmbare] - 0.5*hyb*K[2*dmbare]
    // Thus: k_scaling = 0.5*hyb
    let k_scaling = match scf.scftype {
        SCFType::RHF => 0.5 * hyb,
        _ => hyb,
    };
    let mut v_ao = MatrixFull::new([nao, nao], 0.0);
    for i in 0..nao { for j in 0..nao {
        v_ao[[i, j]] = j_full[[i, j]] - k_scaling * k_full[[i, j]];
    }}

    // ── Step 4: fxc contribution (AO basis, added before MO projection) ──
    // For RKS, the response function adds the XC kernel response fxc[dm1].
    // We compute it from the full symmetric `dm1` (already scaled by the
    // caller via the factor of 2 in `dp1`/`dp2`) so the result is
    // automatically correctly scaled — no extra factor of 2 or 1/4 needed.
    // This matches PySCF's `gen_rks_response(singlet=None)` which calls
    // `nr_rks_fxc(dm1)` (spin=0 unpolarized fxc) on the full perturbed DM.
    //
    // The ground-state-dependent parts (AO, ρ₀, fxc kernel) are read from
    // `FxcHessianCache`, which is built once for the entire CP-HF phase —
    // only the dm1-dependent contraction is performed here.
    if let Some(cache) = fxc_cache {
        let fxc_ao = compute_fxc_response_ao_cached(cache, &dm1);
        for i in 0..nao { for j in 0..nao {
            v_ao[[i, j]] += fxc_ao[[i, j]];
        }}
    }

    // ── Step 5: Project v_ao to frozen and active virtual rows ──
    // Frozen row projection: C_occ^T @ v_ao @ C_frozen → [nocc, nfrozen]
    // VO row projection: C_occ^T @ v_ao @ C_vir → [nocc, nvir]

    // Frozen projection
    let mut tmp_fr = MatrixFull::new([nao, nfrozen], 0.0);
    _dgemm_full(&v_ao, 'N', &ws.c_frozen, 'N', &mut tmp_fr, 1.0, 0.0);
    let mut fro_result = MatrixFull::new([nocc, nfrozen], 0.0);
    _dgemm_full(&ws.c_occ, 'T', &tmp_fr, 'N', &mut fro_result, 1.0, 0.0);

    // VO projection
    let mut tmp_vo = MatrixFull::new([nao, nvir], 0.0);
    _dgemm_full(&v_ao, 'N', &ws.c_vir, 'N', &mut tmp_vo, 1.0, 0.0);
    let mut vo_result = MatrixFull::new([nocc, nvir], 0.0);
    _dgemm_full(&ws.c_occ, 'T', &tmp_vo, 'N', &mut vo_result, 1.0, 0.0);

    // ── Step 6: Flatten to [frozen_response (nfrozen*nocc), VO_response (nvir*nocc)] ──
    let fo_size = nfrozen * nocc;
    let total = fo_size + dim;
    let mut res = vec![0.0; total];
    for k in 0..nfrozen { for i in 0..nocc { res[i + k * nocc] = fro_result[[i, k]]; }}
    for a in 0..nvir { for i in 0..nocc { res[fo_size + i + a * nocc] = vo_result[[i, a]]; }}
    res
}

/// Batched version of `gen_vind_opt` for processing multiple RHS vectors
/// in one call. Phase A optimization: amortizes rayon dispatch and AO cache
/// lookups across all RHS.
///
/// `z_vo_batch`: slice of n_rhs vectors, each `nvir * nocc` (active VO block,
/// flat `[i + a*nocc]`).
/// `z_oo_batch`, `z_fo_batch`: matching occ-occ and frozen-occ blocks; for
/// Krylov matvec these are all zero (pass slices of zeros).
///
/// Returns n_rhs vectors, each `[frozen_response (nfrozen*nocc), VO_response (nvir*nocc)]`.
pub fn gen_vind_opt_batched(
    scf: &SCF,
    ws: &VindWorkspace,
    z_vo_batch: &[&[f64]],            // n_rhs × nvir*nocc
    fxc_cache: Option<&FxcHessianCache>,
    z_oo_batch: Option<&[&[f64]]>,    // n_rhs × nocc²  (None or all zeros for Krylov)
    z_fo_batch: Option<&[&[f64]]>,    // n_rhs × nfrozen*nocc
) -> Vec<Vec<f64>> {
    let nao = ws.nao;
    let nocc = ws.nocc;
    let nvir = ws.nvir;
    let nfrozen = ws.nfrozen;
    let dim = ws.dim;
    let n_rhs = z_vo_batch.len();
    if n_rhs == 0 { return Vec::new(); }

    // ── Step 1: Build AO density matrix per RHS ──
    // dm1[i] = dp1[i] + dp1[i]^T (+ OO and FO contributions if provided)
    let mut dms: Vec<MatrixFull<f64>> = Vec::with_capacity(n_rhs);
    for i in 0..n_rhs {
        let z_vo = z_vo_batch[i];
        let z_oo = z_oo_batch.map(|s| s[i]);
        let z_fo = z_fo_batch.map(|s| s[i]);

        // dp1 = C_vir @ (2*z) @ C_occ^T
        let mut z_scaled = MatrixFull::new([nvir, nocc], 0.0);
        for a in 0..nvir { for i in 0..nocc { z_scaled[[a, i]] = 2.0 * z_vo[i + a * nocc]; } }
        let mut t1 = MatrixFull::new([nao, nocc], 0.0);
        _dgemm_full(&ws.c_vir, 'N', &z_scaled, 'N', &mut t1, 1.0, 0.0);
        let mut dp1 = MatrixFull::new([nao, nao], 0.0);
        _dgemm_full(&t1, 'N', &ws.c_occ, 'T', &mut dp1, 1.0, 0.0);
        // dm1 = dp1 + dp1^T
        let mut dm1 = MatrixFull::new([nao, nao], 0.0);
        for r in 0..nao { for c in 0..nao {
            dm1[[r, c]] = dp1[[r, c]] + dp1[[c, r]];
        }}

        // OO contribution
        if let Some(z_oo) = z_oo {
            let mut z_oo_scaled = MatrixFull::new([nocc, nocc], 0.0);
            for r in 0..nocc { for c in 0..nocc {
                z_oo_scaled[[r, c]] = 2.0 * z_oo[r + c * nocc];
            }}
            let mut t3 = MatrixFull::new([nao, nocc], 0.0);
            _dgemm_full(&ws.c_occ, 'N', &z_oo_scaled, 'N', &mut t3, 1.0, 0.0);
            let mut dp3 = MatrixFull::new([nao, nao], 0.0);
            _dgemm_full(&t3, 'N', &ws.c_occ, 'T', &mut dp3, 1.0, 0.0);
            for r in 0..nao { for c in 0..nao {
                dm1[[r, c]] += dp3[[r, c]] + dp3[[c, r]];
            }}
        }

        // Frozen-occ contribution
        if let Some(z_fo) = z_fo {
            let mut z_fo_scaled = MatrixFull::new([nfrozen, nocc], 0.0);
            for k in 0..nfrozen { for r in 0..nocc {
                z_fo_scaled[[k, r]] = 2.0 * z_fo[r + k * nocc];
            }}
            let mut t4 = MatrixFull::new([nao, nocc], 0.0);
            _dgemm_full(&ws.c_frozen, 'N', &z_fo_scaled, 'N', &mut t4, 1.0, 0.0);
            let mut dp4 = MatrixFull::new([nao, nao], 0.0);
            _dgemm_full(&t4, 'N', &ws.c_occ, 'T', &mut dp4, 1.0, 0.0);
            let mut t5 = MatrixFull::new([nao, nfrozen], 0.0);
            _dgemm_full(&ws.c_occ, 'N', &z_fo_scaled, 'T', &mut t5, 1.0, 0.0);
            let mut dp5 = MatrixFull::new([nao, nao], 0.0);
            _dgemm_full(&t5, 'N', &ws.c_frozen, 'T', &mut dp5, 1.0, 0.0);
            for r in 0..nao { for c in 0..nao {
                dm1[[r, c]] += dp4[[r, c]] + dp5[[r, c]];
            }}
        }
        dms.push(dm1);
    }

    // ── Step 2: Compute J, K per RHS (sequential — Phase C can batch) ──
    let is_hf = scf.mol.xc_data.dfa_compnt_scf.is_empty();
    let hyb = if is_hf { 1.0 } else { scf.mol.xc_data.dfa_hybrid_scf };
    let k_scaling = match scf.scftype {
        SCFType::RHF => 0.5 * hyb,
        _ => hyb,
    };

    let mut v_ao_batch: Vec<MatrixFull<f64>> = Vec::with_capacity(n_rhs);
    for i in 0..n_rhs {
        let dm_vec = vec![dms[i].clone()];
        let j_full = compute_j_upper(scf, &dm_vec).to_matrixfull()
            .unwrap_or_else(|| panic!("J to_matrixfull failed"));
        let k_full = compute_k_upper(scf, &dm_vec).to_matrixfull()
            .unwrap_or_else(|| panic!("K to_matrixfull failed"));
        let mut v_ao = MatrixFull::new([nao, nao], 0.0);
        for r in 0..nao { for c in 0..nao {
            v_ao[[r, c]] = j_full[[r, c]] - k_scaling * k_full[[r, c]];
        }}
        v_ao_batch.push(v_ao);
    }

    // ── Step 3: Add batched fxc response (KEY Phase A win) ──
    if let Some(cache) = fxc_cache {
        let fxc_batch = compute_fxc_response_ao_cached_batched(cache, &dms);
        for i in 0..n_rhs {
            for k in 0..nao * nao {
                v_ao_batch[i].data[k] += fxc_batch[i].data[k];
            }
        }
    }

    // ── Step 4: Project each v_ao to frozen and VO rows, flatten ──
    let fo_size = nfrozen * nocc;
    let total = fo_size + dim;
    let mut results = Vec::with_capacity(n_rhs);
    for i in 0..n_rhs {
        let v_ao = &v_ao_batch[i];
        let mut tmp_fr = MatrixFull::new([nao, nfrozen], 0.0);
        _dgemm_full(v_ao, 'N', &ws.c_frozen, 'N', &mut tmp_fr, 1.0, 0.0);
        let mut fro_result = MatrixFull::new([nocc, nfrozen], 0.0);
        _dgemm_full(&ws.c_occ, 'T', &tmp_fr, 'N', &mut fro_result, 1.0, 0.0);

        let mut tmp_vo = MatrixFull::new([nao, nvir], 0.0);
        _dgemm_full(v_ao, 'N', &ws.c_vir, 'N', &mut tmp_vo, 1.0, 0.0);
        let mut vo_result = MatrixFull::new([nocc, nvir], 0.0);
        _dgemm_full(&ws.c_occ, 'T', &tmp_vo, 'N', &mut vo_result, 1.0, 0.0);

        let mut res = vec![0.0; total];
        for k in 0..nfrozen { for r in 0..nocc { res[r + k * nocc] = fro_result[[r, k]]; }}
        for a in 0..nvir { for r in 0..nocc { res[fo_size + r + a * nocc] = vo_result[[r, a]]; }}
        results.push(res);
    }
    results
}

/// Verify that gen_vind's fxc part matches existing fxc_matvec.
pub fn verify_fxc_matvec(scf: &SCF) -> Result<(), String> {
    println!("\n=== Verifying fxc_matvec consistency ===");
    let (start_mo, _num_state, occ_size, vir_size, _homo, lumo) =
        crate::ri_tddft::utils::tddft_occupation_parameters(scf);
    let dim = occ_size * vir_size;
    let is_dft = !scf.mol.xc_data.dfa_compnt_scf.is_empty();

    let fxc_data = if is_dft {
        println!("  DFT mode: preparing fxc data...");
        Some(prepare_fxc_data(scf))
    } else {
        println!("  HF mode: no fxc kernel.");
        None
    };

    // Random test vector
    let mut z_test = vec![0.0; dim];
    for i in 0..dim { z_test[i] = ((i * 7 + 13) as f64).sin() * 0.1; }

    // fxc_matvec only (use old version for deterministic verification)
    let fxc_only = fxc_data.as_ref().map(|f| fxc_matvec_old(f, &z_test))
        .unwrap_or_else(|| vec![0.0; dim]);

    let ws = VindWorkspace::new(scf, occ_size, vir_size, start_mo, lumo);

    // gen_vind total (JK+fxc)
    let cache = if is_dft { Some(prepare_fxc_hessian_cache(scf)) } else { None };
    let gen_v = gen_vind_opt(scf, &ws, &z_test, cache.as_ref(), None, None);

    // gen_vind JK only (no fxc)
    let jk_only = gen_vind_opt(scf, &ws, &z_test, None, None, None);

    // fxc part from gen_vind = total - JK (skip frozen rows in response)
    let fo_size = ws.nfrozen * occ_size;
    let mut fxc_from_gen = vec![0.0; dim];
    for i in 0..dim { fxc_from_gen[i] = gen_v[fo_size + i] - jk_only[fo_size + i]; }

    // Compare
    let mut diff_norm = 0.0;
    for i in 0..dim { let d = fxc_from_gen[i] - fxc_only[i]; diff_norm += d * d; }
    diff_norm = diff_norm.sqrt();

    println!("  dim = {}", dim);
    println!("  |fxc_matvec|        = {:.10e}", fxc_only.iter().map(|x|x*x).sum::<f64>().sqrt());
    println!("  |gen_vind fxc part| = {:.10e}", fxc_from_gen.iter().map(|x|x*x).sum::<f64>().sqrt());
    println!("  |fxc diff|          = {:.2e}", diff_norm);
    println!("  |gen_vind JK|       = {:.10e}", jk_only.iter().map(|x|x*x).sum::<f64>().sqrt());

    if !is_dft || diff_norm < 1e-12 {
        println!("  ✅ fxc_matvec in gen_vind matches existing fxc_matvec!");
    } else {
        println!("  ⚠️  fxc part differs by {:.2e}", diff_norm);
    }
    println!("  === fxc_matvec verification complete ===\n");
    Ok(())
}

