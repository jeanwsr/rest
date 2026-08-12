/// PySCF-style response function generator for RKS/RHF.
///
/// Matches PySCF scf/_response_functions.py:
///   - `_gen_rhf_response(singlet=None)` → AO-space Fock response `vind(dm1)`
///   - `gen_vind` via hessian/rhf.py → AO↔MO bridging
///
/// For HF: vind(dm1) = J[dm1] - 0.5*K[dm1]
/// For DFT: vind(dm1) = fxc[dm1] + J[dm1] - hyb*K[dm1]

use rest_tensors::{MatrixFull, MatrixUpper, RIFull};
use rest_tensors::matrix::matrix_blas_lapack::{_dgemm_full, _dsymm, omp_set_num_threads_wrapper};
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
    // Force single-threaded BLAS inside this (possibly parallel) grid task:
    // the process-wide OpenBLAS pool would otherwise oversubscribe the CPU
    // (n_blocks rayon tasks × OpenBLAS threads) and stall the contractions.
    omp_set_num_threads_wrapper(1);
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
// ============================================================================
// Low-rank CPHF exchange-response (K) precomputation
//
// In the CPHF Krylov matvec, dm1 = dp1 + dp1ᵀ with dp1 = C_vir·Z'·C_occᵀ
// (rank ≤ 2·nocc). The K term
//     K = Σ_p B_p·dm1·B_pᵀ            (B_p = RI 3-center column, symmetric)
// then needs only its VO projection:
//     C_occᵀ·K·C_vir = Σ_p K_p·Z'·N_pᵀ  +  (Σ_p M_p·Z'·L_pᵀ)ᵀ
// with the four ground-state (z-independent) factors
//     K_p = C_occᵀ·B_p·C_vir   [nocc, nvir]
//     N_p = C_virᵀ·B_p·C_occ   [nvir, nocc]
//     M_p = C_virᵀ·B_p·C_vir   [nvir, nvir]   (symmetric)
//     L_p = C_occᵀ·B_p·C_occ   [nocc, nocc]   (symmetric)
// Precomputed once per Hessian; each matvec does 2 batched GEMMs
// (K_batch·Z', M_batch·Z') + 2 small accumulations. FLOPs per matvec drop
// from ~2·P·N³ (dsymm pair) to ~2·P·nvir²·nocc (~10× for N=358, nocc=34).
// ============================================================================
pub struct KLowRankPrecompute {
    pub nocc: usize,
    pub nvir: usize,
    pub naux: usize,
    /// K_p [P·nocc, nvir] col-major
    pub k_batch: MatrixFull<f64>,
    /// N_p [P·nvir, nocc] col-major
    pub n_batch: MatrixFull<f64>,
    /// M_p [P·nvir, nvir] col-major
    pub m_batch: MatrixFull<f64>,
    /// L_p [P·nocc, nocc] col-major
    pub l_batch: MatrixFull<f64>,
}

impl KLowRankPrecompute {
    /// Build from the RI 3-center tensor (rimatr preferred, ri3fn fallback)
    /// and the MO coefficients. Returns None if no RI tensor is available.
    pub fn new(scf: &SCF, ws: &VindWorkspace) -> Option<Self> {
        use rayon::prelude::*;
        let nao = ws.nao;
        let nocc = ws.nocc;
        let nvir = ws.nvir;
        let c_occ = &ws.c_occ;
        let c_vir = &ws.c_vir;
        enum Src<'a> {
            Rim(&'a MatrixFull<f64>, usize), // packed upper [N(N+1)/2, P]
            Ri3(&'a RIFull<f64>),            // full symmetric [N, N, P]
        }
        let src: Src<'_>;
        let naux;
        if let Some((ri, _, _)) = &scf.rimatr {
            naux = ri.size[1];
            src = Src::Rim(ri, ri.size[0]);
        } else if let Some(ri3) = &scf.ri3fn {
            naux = ri3.size[2];
            src = Src::Ri3(ri3);
        } else {
            return None;
        }
        if std::env::var("REST_VERIFY_LOWRANK").is_ok() {
            match &src {
                Src::Rim(ri, nbp) => eprintln!("DBG lowrank src: rimatr size={:?} naux={}", ri.size, naux),
                Src::Ri3(ri3) => eprintln!("DBG lowrank src: ri3fn size={:?} naux={}", ri3.size, naux),
            }
        }
        let size_kn = naux * nocc * nvir;
        let size_nn = naux * nvir * nocc;
        let size_mm = naux * nvir * nvir;
        let size_ll = naux * nocc * nocc;
        // Batched layout buffers (col-major: K_batch[(p,i) + a·(P·nocc)] etc.).
        let mut k_p = vec![0.0; size_kn];
        let mut n_p = vec![0.0; size_nn];
        let mut m_p = vec![0.0; size_mm];
        let mut l_p = vec![0.0; size_ll];
        // Per-p blocks are built in chunks (128 p each) so the intermediate
        // (kp/np/mp/lp tuples, ~0.9 GB for all p) never coexists with the
        // final batched buffers — peak build RSS ≈ k_p+n_p+m_p+l_p (0.9 GB)
        // + one chunk (0.13 GB) instead of 1.8 GB.
        const CHUNK: usize = 128;
        for p_lo in (0..naux).step_by(CHUNK) {
            let p_hi = (p_lo + CHUNK).min(naux);
            let cols_chunk: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> = (p_lo..p_hi)
                .into_par_iter()
                .map(|p| {
                    let mut b = vec![0.0; nao * nao];
                    match &src {
                        Src::Rim(ri, num_baspair) => {
                            let col = &ri.data[p * num_baspair..(p + 1) * num_baspair];
                            let mut it = col.iter();
                            for nu in 0..nao {
                                for mu in 0..=nu {
                                    let v = *it.next().unwrap();
                                    b[mu + nu * nao] = v;
                                    b[nu + mu * nao] = v;
                                }
                            }
                        }
                        Src::Ri3(ri3) => {
                            let col = &ri3.data[p * nao * nao..(p + 1) * nao * nao];
                            b.copy_from_slice(col);
                        }
                    }
                    let b_mat = MatrixFull::from_vec([nao, nao], b).unwrap();
                    let mut bcv = MatrixFull::new([nao, nvir], 0.0);
                    _dsymm(&b_mat, c_vir, &mut bcv, 'L', 'U', 1.0, 0.0); // B_p·C_vir
                    let mut bco = MatrixFull::new([nao, nocc], 0.0);
                    _dsymm(&b_mat, c_occ, &mut bco, 'L', 'U', 1.0, 0.0); // B_p·C_occ
                    let mut kp = MatrixFull::new([nocc, nvir], 0.0);
                    _dgemm_full(c_occ, 'T', &bcv, 'N', &mut kp, 1.0, 0.0);
                    let mut np = MatrixFull::new([nvir, nocc], 0.0);
                    _dgemm_full(c_vir, 'T', &bco, 'N', &mut np, 1.0, 0.0);
                    let mut mp = MatrixFull::new([nvir, nvir], 0.0);
                    _dgemm_full(c_vir, 'T', &bcv, 'N', &mut mp, 1.0, 0.0);
                    let mut lp = MatrixFull::new([nocc, nocc], 0.0);
                    _dgemm_full(c_occ, 'T', &bco, 'N', &mut lp, 1.0, 0.0);
                    (kp.data, np.data, mp.data, lp.data)
                })
                .collect();
            for (p_local, (kp, np, mp, lp)) in cols_chunk.iter().enumerate() {
                let p = p_lo + p_local;
                for a in 0..nvir {
                    for i in 0..nocc {
                        k_p[p * nocc + i + a * nocc * naux] = kp[i + a * nocc];
                    }
                }
                for c in 0..nocc {
                    for a in 0..nvir {
                        n_p[p * nvir + a + c * nvir * naux] = np[a + c * nvir];
                    }
                }
                for v in 0..nvir {
                    for u in 0..nvir {
                        m_p[p * nvir + u + v * nvir * naux] = mp[u + v * nvir];
                    }
                }
                for c in 0..nocc {
                    for i in 0..nocc {
                        l_p[p * nocc + i + c * nocc * naux] = lp[i + c * nocc];
                    }
                }
            }
            drop(cols_chunk);
        }
        Some(KLowRankPrecompute {
            nocc, nvir, naux,
            k_batch: MatrixFull::from_vec([nocc * naux, nvir], k_p).unwrap(),
            n_batch: MatrixFull::from_vec([nvir * naux, nocc], n_p).unwrap(),
            m_batch: MatrixFull::from_vec([nvir * naux, nvir], m_p).unwrap(),
            l_batch: MatrixFull::from_vec([nocc * naux, nocc], l_p).unwrap(),
        })
    }
}

/// K contribution to the VO-projected CPHF response for one RHS z (flat
/// [nocc·nvir], index i + a·nocc — same layout as the z input).
///
///   C_occᵀ·K·C_vir = Σ_p K_p·Z'·N_pᵀ  +  (Σ_p M_p·Z'·L_pᵀ)ᵀ
///
/// Two batched GEMMs (K_batch·Z', M_batch·Z') + two parallel accumulations.
/// FLOPs: ~2·P·(nvir²·nocc + nocc²·nvir) vs ~2·P·N³ for the dsymm pair.
pub fn k_vo_lowrank(pre: &KLowRankPrecompute, z: &[f64]) -> Vec<f64> {
    use rayon::prelude::*;
    let _tk = std::time::Instant::now();
    let nocc = pre.nocc;
    let nvir = pre.nvir;
    let naux = pre.naux;
    // Z' [nvir, nocc] col-major: Z'[u + c·nvir] = 2·z[c + u·nocc]
    let mut zp = vec![0.0; nvir * nocc];
    for u in 0..nvir {
        for c in 0..nocc {
            zp[u + c * nvir] = 2.0 * z[c + u * nocc];
        }
    }
    let z_mat = MatrixFull::from_vec([nvir, nocc], zp).unwrap();
    // Tk = K_batch [P·nocc, nvir] @ Z' → [P·nocc, nocc]
    let mut tk = MatrixFull::new([nocc * naux, nocc], 0.0);
    _dgemm_full(&pre.k_batch, 'N', &z_mat, 'N', &mut tk, 1.0, 0.0);
    // T = M_batch [P·nvir, nvir] @ Z' → [P·nvir, nocc]
    let mut t = MatrixFull::new([nvir * naux, nocc], 0.0);
    _dgemm_full(&pre.m_batch, 'N', &z_mat, 'N', &mut t, 1.0, 0.0);
    let tk_d = &tk.data;
    let t_d = &t.data;
    let n_p = &pre.n_batch.data;
    let l_p = &pre.l_batch.data;
    // vo_pos[i,a] += Σ_c Tk_p[i,c]·N_p[a,c] ; vo_neg[u,c] += Σ_d T_p[u,d]·L_p[c,d]
    let (pos, neg) = (0..naux)
        .into_par_iter()
        .fold(
            || (vec![0.0; nocc * nvir], vec![0.0; nvir * nocc]),
            |(mut pos, mut neg), p| {
                for i in 0..nocc {
                    for a in 0..nvir {
                        let mut s = 0.0;
                        for c in 0..nocc {
                            s += tk_d[p * nocc + i + c * nocc * naux]
                                * n_p[p * nvir + a + c * nvir * naux];
                        }
                        pos[i + a * nocc] += s;
                    }
                }
                for u in 0..nvir {
                    for c in 0..nocc {
                        let mut s = 0.0;
                        for d in 0..nocc {
                            s += t_d[p * nvir + u + d * nvir * naux]
                                * l_p[p * nocc + c + d * nocc * naux];
                        }
                        neg[u + c * nvir] += s;
                    }
                }
                (pos, neg)
            },
        )
        .reduce(
            || (vec![0.0; nocc * nvir], vec![0.0; nvir * nocc]),
            |(mut a1, mut b1), (a2, b2)| {
                for k in 0..a1.len() {
                    a1[k] += a2[k];
                }
                for k in 0..b1.len() {
                    b1[k] += b2[k];
                }
                (a1, b1)
            },
        );
    // result[i + a·nocc] = vo_pos[i,a] + vo_neg[a,i]
    let mut res = vec![0.0; nocc * nvir];
    for i in 0..nocc {
        for a in 0..nvir {
            res[i + a * nocc] = pos[i + a * nocc] + neg[a + i * nvir];
        }
    }
    if std::env::var("REST_CPHF_PROFILE").is_ok() {
        eprintln!("CPHF-PROF kvo {:.4}s", _tk.elapsed().as_secs_f64());
    }
    res
}

/// Batched version of [`k_vo_lowrank`]: processes n_rhs z-vectors in one call
/// so the big M_batch/K_batch GEMMs amortize the 740 MB read across all RHS
/// (N = nocc·n_rhs instead of nocc per call). T/Tk are streamed in p-blocks
/// (PB aux p each) and contracted immediately, keeping peak RSS ≈ PB-block
/// outputs (133 MB at PB=32) instead of the full [P·nvir, nocc·n_rhs] T.
pub fn k_vo_lowrank_batched(pre: &KLowRankPrecompute, z_batch: &[&[f64]]) -> Vec<Vec<f64>> {
    use rayon::prelude::*;
    let _tkb = std::time::Instant::now();
    let mut _t_copy = 0.0f64;
    let mut _t_gemm = 0.0f64;
    let mut _t_contr = 0.0f64;
    let nocc = pre.nocc;
    let nvir = pre.nvir;
    let naux = pre.naux;
    let n_rhs = z_batch.len();
    let dim = nocc * nvir;
    // Z' [nvir, nocc·n_rhs] col-major: Z'[u + (c + z·nocc)·nvir] = 2·z[c + u·nocc]
    let mut zp = vec![0.0; nvir * nocc * n_rhs];
    for z in 0..n_rhs {
        for u in 0..nvir {
            for c in 0..nocc {
                zp[u + (c + z * nocc) * nvir] = 2.0 * z_batch[z][c + u * nocc];
            }
        }
    }
    let z_mat = MatrixFull::from_vec([nvir, nocc * n_rhs], zp).unwrap();
    let n_p = &pre.n_batch.data;
    let l_p = &pre.l_batch.data;
    const PB: usize = 32;
    let mut res_all = vec![vec![0.0; dim]; n_rhs];
    for p0 in (0..naux).step_by(PB) {
        let pe = (p0 + PB).min(naux);
        let npb = pe - p0;
        // Tk_blk = K_batch[p-block]·Z' → [npb·nocc, nocc·n_rhs].
        // K_batch/M_batch are col-major [P·rows, nvir]: the p-block rows are
        // contiguous *within each column*, so the block matrix must be copied
        // column by column (not sliced).
        let _tc = std::time::Instant::now();
        let mut tk_blk = MatrixFull::new([npb * nocc, nocc * n_rhs], 0.0);
        {
            // p-block rows are contiguous within each column of the col-major
            // K_batch; copy columns in parallel, assemble col-major afterwards.
            let k_cols: Vec<Vec<f64>> = (0..nvir)
                .into_par_iter()
                .map(|a| {
                    let s = p0 * nocc + a * (nocc * naux);
                    pre.k_batch.data[s..s + npb * nocc].to_vec()
                })
                .collect();
            let mut k_blk_m = MatrixFull::new([npb * nocc, nvir], 0.0);
            for (a, col) in k_cols.iter().enumerate() {
                k_blk_m.data[a * (npb * nocc)..(a + 1) * (npb * nocc)].copy_from_slice(col);
            }
            let _tg1 = std::time::Instant::now();
            _dgemm_full(&k_blk_m, 'N', &z_mat, 'N', &mut tk_blk, 1.0, 0.0);
            _t_gemm += _tg1.elapsed().as_secs_f64();
        }
        // T_blk = M_batch[p-block]·Z' → [npb·nvir, nocc·n_rhs]
        let mut t_blk = MatrixFull::new([npb * nvir, nocc * n_rhs], 0.0);
        {
            let m_cols: Vec<Vec<f64>> = (0..nvir)
                .into_par_iter()
                .map(|a| {
                    let s = p0 * nvir + a * (nvir * naux);
                    pre.m_batch.data[s..s + npb * nvir].to_vec()
                })
                .collect();
            let mut m_blk_m = MatrixFull::new([npb * nvir, nvir], 0.0);
            for (a, col) in m_cols.iter().enumerate() {
                m_blk_m.data[a * (npb * nvir)..(a + 1) * (npb * nvir)].copy_from_slice(col);
            }
            let _tg2 = std::time::Instant::now();
            _dgemm_full(&m_blk_m, 'N', &z_mat, 'N', &mut t_blk, 1.0, 0.0);
            _t_gemm += _tg2.elapsed().as_secs_f64();
            _t_copy += _tc.elapsed().as_secs_f64();
        }
        let tk_d = &tk_blk.data;
        let t_d = &t_blk.data;
        // Contract Tk/T into res_all via per-worker private accumulators
        // (par_chunks, one merge per worker — no fold/reduce tree).
        let _tc2 = std::time::Instant::now();
        {
            let nw = rayon::current_num_threads().max(1);
            let chunk = npb.div_ceil(nw);
            let partials: Vec<((Vec<f64>, Vec<f64>))> = (0..nw)
                .into_par_iter()
                .map(|w| {
                    let lo = (w * chunk).min(npb);
                    let hi = ((w + 1) * chunk).min(npb);
                    let mut pos = vec![0.0; n_rhs * dim];
                    let mut neg = vec![0.0; n_rhs * nvir * nocc];
                    for pl in lo..hi {
                        let p = p0 + pl;
                        let tkb = &tk_d[pl * nocc..];
                        let tb = &t_d[pl * nvir..];
                        // Cache-friendly: c/d outer; contiguous inner writes
                        // (pos[i·nvir+a], neg[u·nocc+c]).
                        for z in 0..n_rhs {
                            let pb = &mut pos[z * dim..];
                            for c in 0..nocc {
                                let ncol = &n_p[p * nvir + c * nvir * naux..p * nvir + c * nvir * naux + nvir];
                                let tcol_base = (c + z * nocc) * (npb * nocc);
                                for i in 0..nocc {
                                    let tki = tkb[i + tcol_base];
                                    for a in 0..nvir {
                                        pb[i * nvir + a] += tki * ncol[a];
                                    }
                                }
                            }
                            let nb = &mut neg[z * nvir * nocc..];
                            for d in 0..nocc {
                                let tcol_base = (d + z * nocc) * (npb * nvir);
                                for u in 0..nvir {
                                    let tu = tb[u + tcol_base];
                                    for c in 0..nocc {
                                        nb[u * nocc + c] += tu
                                            * l_p[p * nocc + c + d * nocc * naux];
                                    }
                                }
                            }
                        }
                    }
                    (pos, neg)
                })
                .collect();
            for (pos, neg) in partials {
                for z in 0..n_rhs {
                    let r = &mut res_all[z];
                    let pz = &pos[z * dim..];
                    for i in 0..nocc {
                        for a in 0..nvir {
                            r[i + a * nocc] += pz[i * nvir + a];
                        }
                    }
                    let nz = &neg[z * nvir * nocc..];
                    for u in 0..nvir {
                        for c in 0..nocc {
                            r[c + u * nocc] += nz[u * nocc + c];
                        }
                    }
                }
            }
        }
        _t_contr += _tc2.elapsed().as_secs_f64();
        drop(tk_blk);
        drop(t_blk);
    }
    if std::env::var("REST_CPHF_PROFILE").is_ok() {
        eprintln!("CPHF-PROF kvob n={} total {:.3}s copy {:.3}s gemm {:.3}s contr {:.3}s",
            n_rhs, _tkb.elapsed().as_secs_f64(), _t_copy, _t_gemm, _t_contr);
    }
    res_all
}

/// Batched RI-J: computes J = V⁻¹·(Cᵀ·dm) for all n_rhs densities in ONE
/// pair of GEMMs (ri3fnᵀ·dm_upper_batch, ri3fn·tmp_mu), amortizing the
/// 176 MB ri3fn read across the batch (was 1 dgemv pair per RHS).
pub fn vj_upper_rimatr_batched(
    scf: &SCF,
    dms: &[MatrixFull<f64>],
) -> Vec<MatrixFull<f64>> {
    use itertools::Itertools;
    let n_rhs = dms.len();
    let nao = dms[0].size[0];
    let (ri3fn, _, baspar2basbas) = scf.rimatr.as_ref().unwrap();
    let npair = ri3fn.size[0];
    let naux = ri3fn.size[1];
    // Upper-triangle compressed density per RHS (diagonal halved), using
    // MatrixUpper's compression order (identical to vj_upper_with_rimatr_sync
    // so the ri3fn rows match).
    let mut dm_upper = vec![0.0; npair * n_rhs];
    for k in 0..n_rhs {
        let mut upper =
            MatrixUpper::from_vec(npair, dms[k].iter_matrixupper().unwrap().map(|x| *x).collect())
                .unwrap();
        upper.iter_diagonal_mut().for_each(|x| *x *= 0.5);
        dm_upper[k * npair..(k + 1) * npair].copy_from_slice(&upper.data);
    }
    let dm_m = MatrixFull::from_vec([npair, n_rhs], dm_upper).unwrap();
    // tmp_mu[aux, k] = Σ_pair ri3fn[pair,aux]·dm_upper[pair,k] (×2)
    let mut tmp_mu = MatrixFull::new([naux, n_rhs], 0.0);
    _dgemm_full(ri3fn, 'T', &dm_m, 'N', &mut tmp_mu, 2.0, 0.0);
    // vj[pair', k] = Σ_aux ri3fn[pair',aux]·tmp_mu[aux,k]
    let mut vj = MatrixFull::new([npair, n_rhs], 0.0);
    _dgemm_full(ri3fn, 'N', &tmp_mu, 'N', &mut vj, 1.0, 0.0);
    // Expand to symmetric [nao, nao].
    let mut out: Vec<MatrixFull<f64>> = Vec::with_capacity(n_rhs);
    for k in 0..n_rhs {
        let mut m = vec![0.0; nao * nao];
        for (ipair, &[mu, nu]) in baspar2basbas.iter().enumerate() {
            let v = vj.data[ipair + k * npair];
            m[mu + nu * nao] = v;
            if mu != nu {
                m[nu + mu * nao] = v;
            }
        }
        out.push(MatrixFull::from_vec([nao, nao], m).unwrap());
    }
    if std::env::var("REST_VERIFY_LOWRANK").is_ok() {
        let j_ref = compute_j_upper(scf, &vec![dms[0].clone()]).to_matrixfull().unwrap();
        let mut mx = 0.0f64;
        for r in 0..nao {
            for c in 0..nao {
                let d = (out[0][[r, c]] - j_ref[[r, c]]).abs();
                if d > mx { mx = d; }
            }
        }
        eprintln!("DBG vj_batched[0] max_diff={:.3e}", mx);
    }
    out
}

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
    let j_full = compute_j_upper(scf, &dm_vec).to_matrixfull()
        .unwrap_or_else(|| panic!("J to_matrixfull failed"));
    // Skip the exchange response entirely for pure (LDA/GGA) DFAs: the K term
    // is scaled by k_scaling = 0, so compute_k_upper would be pure waste. K is
    // O(naux·nao³) per call and dominates the JK phase of every matvec.
    let k_full = if k_scaling != 0.0 {
        compute_k_upper(scf, &dm_vec).to_matrixfull()
            .unwrap_or_else(|| panic!("K to_matrixfull failed"))
    } else {
        MatrixFull::new([nao, nao], 0.0)
    };

    // ── Step 3: v_ao = J - k_scaling*K ──
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
    k_lowrank: Option<&KLowRankPrecompute>, // low-rank K path (Krylov matvec)
) -> Vec<Vec<f64>> {
    let nao = ws.nao;
    let nocc = ws.nocc;
    let nvir = ws.nvir;
    let nfrozen = ws.nfrozen;
    let dim = ws.dim;
    let n_rhs = z_vo_batch.len();
    if n_rhs == 0 { return Vec::new(); }
    // Low-rank K applies only when dm1 is the pure VO low-rank form
    // (no OO/FO blocks) and there are no frozen orbitals to project onto.
    let use_lowrank_k = k_lowrank.is_some()
        && z_oo_batch.is_none()
        && z_fo_batch.is_none()
        && nfrozen == 0;

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
    // Low-rank K: precompute the VO projection once per RHS (bypasses the
    // full [N,N] K assembly + Step-4 projection entirely).
    let mut k_vo_batch: Option<Vec<Vec<f64>>> = if use_lowrank_k {
        Some(k_vo_lowrank_batched(k_lowrank.unwrap(), z_vo_batch))
    } else {
        None
    };
    if std::env::var("REST_VERIFY_LOWRANK").is_ok() && use_lowrank_k {
        for i in 0..n_rhs {
            let dm_vec = vec![dms[i].clone()];
            let k_full = compute_k_upper(scf, &dm_vec).to_matrixfull()
                .unwrap_or_else(|| panic!("K to_matrixfull failed"));
            let mut tmp_vo = MatrixFull::new([nao, nvir], 0.0);
            _dgemm_full(&k_full, 'N', &ws.c_vir, 'N', &mut tmp_vo, 1.0, 0.0);
            let mut vo_orig = MatrixFull::new([nocc, nvir], 0.0);
            _dgemm_full(&ws.c_occ, 'T', &tmp_vo, 'N', &mut vo_orig, 1.0, 0.0);
            let kv = &k_vo_batch.as_ref().unwrap()[i];
            let mut mx = 0.0f64;
            for a in 0..nvir { for r in 0..nocc {
                let d = (vo_orig[[r, a]] - kv[r + a * nocc]).abs();
                if d > mx { mx = d; }
            }}
            let norm = (0..nvir*nocc).map(|k| vo_orig.data[k]*vo_orig.data[k]).sum::<f64>().sqrt();
            eprintln!("DBG lowrank[{}]: max_diff={:.3e} orig_norm={:.3e}", i, mx, norm);
        }
    }
    let j_batch: Vec<MatrixFull<f64>> = if let Some(_rimatr) = &scf.rimatr {
        vj_upper_rimatr_batched(scf, &dms)
    } else {
        (0..n_rhs)
            .map(|i| {
                let dm_vec = vec![dms[i].clone()];
                compute_j_upper(scf, &dm_vec)
                    .to_matrixfull()
                    .unwrap_or_else(|| panic!("J to_matrixfull failed"))
            })
            .collect()
    };
    for i in 0..n_rhs {
        let j_full = &j_batch[i];
        // Skip the exchange response for pure DFAs (k_scaling == 0); K is
        // O(naux·nao³) and would dominate the per-RHS matvec cost as waste.
        let k_full = if k_scaling != 0.0 && !use_lowrank_k {
            let dm_vec = vec![dms[i].clone()];
            compute_k_upper(scf, &dm_vec)
                .to_matrixfull()
                .unwrap_or_else(|| panic!("K to_matrixfull failed"))
        } else {
            MatrixFull::new([nao, nao], 0.0)
        };
        let mut v_ao = MatrixFull::new([nao, nao], 0.0);
        for r in 0..nao {
            for c in 0..nao {
                v_ao[[r, c]] = j_full[[r, c]] - k_scaling * k_full[[r, c]];
            }
        }
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
        // Low-rank K: the K contribution to the VO projection was computed
        // directly in MO space — subtract k_scaling·K_vo.
        if let Some(kvb) = &k_vo_batch {
            for k in 0..dim {
                res[fo_size + k] -= k_scaling * kvb[i][k];
            }
        }
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

