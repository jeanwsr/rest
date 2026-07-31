//! RKS-unique analytic Hessian contributions.
//!
//! Free functions that scatter directly into existing buffers passed by &mut
//! slice — zero new allocation. The XC computational kernels themselves live
//! in xc_hessian.rs and are NOT duplicated here.

use crate::scf_io::SCF;
use crate::hessian::memory_monitor;

/// Guard for optional CPU-monitor instrumentation.
macro_rules! cpu_section {
    ($mon:expr, $label:expr) => {
        let _guard = if let Some(ref m) = $mon {
            Some(m.section($label))
        } else {
            None::<crate::hessian::cpu_monitor::CpuSection<'_>>
        };
    };
}

/// Add the RKS XC contribution (vxc_diag + vxc_deriv2) to the electronic
/// Hessian partial `h_partial` (column-major, n3*n3, modified in place).
///
/// `h_partial` is the existing buffer owned by the caller (the live
/// `result["h_partial"].data` slice); this function writes into it directly
/// using the same column-major indexing as the inline block it replaced:
///   `h_partial[(ia*3+a)*n3 + (ja*3+b)]`.
///
/// `timings` receives the per-stage `(label, elapsed)` pairs the caller
/// records for its timing profile.
pub fn add_vxc_h_partial(
    scf: &SCF,
    h_partial: &mut [f64],
    timings: &mut Vec<(&'static str, std::time::Duration)>,
) {
    let _t_rks_xc = std::time::Instant::now();

    // ── Optional CPU-monitor instrumentation ──
    let _cpu_mon = if std::env::var("REST_EJ_EK_CPU_TRACE").as_deref() == Ok("1") {
        let mon = crate::hessian::cpu_monitor::CpuMonitor::default_period();
        let tr = crate::hessian::cpu_monitor::ThreadReport::collect();
        tr.print("add_vxc_h_partial");

        // Show grid parallelism plan
        if let Some(ref grids) = scf.grids {
            let block_ranges: &[std::ops::Range<usize>] = &grids.parallel_balancing;
            let (sub_blocks, concurrency, sub_nb) =
                crate::hessian::xc_hessian::plan_grid_split(block_ranges);
            let n_sub = if sub_nb > 0 { sub_blocks.len() } else { block_ranges.len() };
            let sub_info = if sub_nb > 0 {
                format!("| {} sub-blocks (max {} pts each)", n_sub, sub_nb)
            } else {
                format!("| {} blocks", block_ranges.len())
            };
            println!(
                "  [cpu] grid plan: adaptive {} | concurrency = {} | {} ngrid | rayon {} threads",
                sub_info, concurrency, grids.coordinates.len(),
                rayon::current_num_threads()
            );
        }
        println!("  [cpu] RKS XC h_partial instrumentation active\n");
        Some(mon)
    } else {
        None
    };
    let mol_rks = &scf.mol;
    let nao_xc = mol_rks.num_basis;
    let natm_xc = mol_rks.geom.nfree;
    let n3_xc = natm_xc * 3;
    let xc_type = if mol_rks.xc_data.use_density_gradient() {
        crate::dft::xc_deriv::XCType::GGA
    } else {
        crate::dft::xc_deriv::XCType::LDA
    };
    let aoslices_xc = crate::hessian::xc_hessian::build_aoslices(mol_rks);
    let dm0_xc = &scf.density_matrix[0];

    // ── Optional grid block diagnostics ──
    if _cpu_mon.is_some() {
        if let Some(ref grids) = scf.grids {
            let nblocks = grids.parallel_balancing.len();
            let nactive = grids.parallel_balancing.iter()
                .filter(|r| r.end > r.start).count();
            println!("  [cpu] grid blocks: {} total, {} non-empty, {} ngrid",
                nblocks, nactive, grids.coordinates.len());
        }
    }

    // Return freed Phase 1-4 memory to OS before the DFT grid sweep.
    memory_monitor::trim_to_os(scf.mol.ctrl.print_level);

    // vxc_diag (computed once, scattered to diagonal atom blocks)
    let _t_diag = std::time::Instant::now();
    // Phase 2: streaming mode — process grid blocks in small concurrent
    // batches instead of collecting all AO data in a cache. Peak AO
    // memory stays bounded by the adaptive grid split plan.
    {
        cpu_section!(_cpu_mon, "rks:vxc_diag:grid");
        let vxc_diag_mat = crate::hessian::xc_hessian::vxc_diag_streaming(scf, xc_type);
        cpu_section!(_cpu_mon, "rks:vxc_diag:scatter");
        for ia in 0..natm_xc {
            let (p0, p1) = aoslices_xc[ia];
            for a in 0..3 { for b in 0..3 {
                let row0 = (a * 3 + b) * nao_xc;
                let mut s = 0.0;
                for mu in p0..p1 { for nu in 0..nao_xc {
                    s += vxc_diag_mat[[row0 + mu, nu]] * dm0_xc[[mu, nu]];
                }}
                h_partial[(ia * 3 + a) * n3_xc + (ia * 3 + b)] += s * 2.0;
            }}
        }
        timings.push(("  rks: vxc_diag", _t_diag.elapsed()));
        drop(vxc_diag_mat);
    }
    memory_monitor::trim_to_os(scf.mol.ctrl.print_level);

    // vxc_deriv2 (per-atom, symmetrized) — streaming with deriv=2.
    let _t_d2 = std::time::Instant::now();
    {
        cpu_section!(_cpu_mon, "rks:vxc_d2:grid");
        let vxc_d2 = crate::hessian::xc_hessian::vxc_deriv2_streaming(scf, xc_type);
        cpu_section!(_cpu_mon, "rks:vxc_d2:scatter");
        for ia in 0..natm_xc {
            for ja in 0..=ia {
                let (q0, q1) = aoslices_xc[ja];
                for a in 0..3 { for b in 0..3 {
                    let row0 = (a * 3 + b) * nao_xc;
                    let mut s = 0.0;
                    for mu in q0..q1 { for nu in 0..nao_xc {
                        s += vxc_d2[ia][[row0 + mu, nu]] * dm0_xc[[mu, nu]];
                    }}
                    h_partial[(ia * 3 + a) * n3_xc + (ja * 3 + b)] += s * 2.0;
                }}
            }
        }
        for ia in 0..natm_xc {
            for ja in 0..ia {
                for a in 0..3 { for b in 0..3 {
                    h_partial[(ja * 3 + b) * n3_xc + (ia * 3 + a)] =
                        h_partial[(ia * 3 + a) * n3_xc + (ja * 3 + b)];
                }}
            }
        }
        timings.push(("  rks: vxc_deriv2", _t_d2.elapsed()));
        drop(vxc_d2);
    }

    // ── CPU monitor final report ──
    if let Some(ref mon) = _cpu_mon {
        mon.report();
    }

    timings.push(("  rks: xc_add", _t_rks_xc.elapsed()));
}

/// Compute the RKS vxc_deriv1 contribution to h1ao, per atom.
///
/// Returns `Vec<Vec<f64>>` indexed `[atom][nao*nao]` — the same shape
/// `calc_h1ao` previously built inline for `vxc_d1`. Memory-neutral: this is
/// the small per-atom vec the caller already allocates; no n3*n3 buffer is
/// involved.
pub fn compute_vxc_h1ao(scf: &SCF) -> Vec<Vec<f64>> {
    let mol = &scf.mol;
    let xct = if mol.xc_data.use_density_gradient() {
        crate::dft::xc_deriv::XCType::GGA
    } else {
        crate::dft::xc_deriv::XCType::LDA
    };
    let v = crate::hessian::xc_hessian::vxc_deriv1_streaming(scf, xct);
    v.iter().map(|m| m.iter().copied().collect()).collect()
}

// ═══════════════════════════════════════════════════════════════════
// RKS fxc CP-HF response glue
//
// These three free functions are the cleanly-separable pieces of the RKS
// fxc CP-HF contribution. They are invoked from `calc_cphf_contrib` in
// rhf.rs at three points:
//   1. once before the solve phase   → `prepare_fxc_cache`
//   2. inside the mo_e1 loop body     → `add_fxc_to_v_ao`  (in-place)
//   3. twice after fxc-heavy phases   → `record_fxc_*_timings`
//
// The genuinely entangled bit — threading `fxc_cache_ref` (None for HF,
// Some(cache) for RKS) into the shared batched Krylov solve — stays
// in rhf.rs because extracting it would require restructuring the entire
// solve phase, which violates the "leave RHF-shared solve logic untouched"
// constraint. HF passing `None` is benign and does not branch on RKS-ness.
//
// Memory: zero new allocation. `add_fxc_to_v_ao` writes into the existing
// `v_ao` MatrixFull the mo_e1 loop already owns; no n3*n3 buffer is
// allocated anywhere in this module. The XC kernels themselves
// (`FxcHessianCache`, `compute_fxc_response_ao_cached`) live in
// `dft::response` and are invoked, not duplicated.
// ═══════════════════════════════════════════════════════════════════

use crate::dft::response::{
    reset_fxc_timing, read_fxc_timing_s, read_fxc_subtimings_s,
    prepare_fxc_hessian_cache, compute_fxc_response_ao_cached, FxcHessianCache,
};
use tensors::MatrixFull;

/// Build the fxc kernel cache for the CP-HF phase (PySCF `cache_xc_kernel`
/// analog). Returns `None` for HF (no DFA components) so the caller can pass
/// the resulting `Option<&FxcHessianCache>` uniformly into the shared solve
/// calls without branching.
///
/// Resets the global fxc timing counter first so the subsequent solve-phase
/// fxc work is attributed correctly.
pub fn prepare_fxc_cache(scf: &SCF) -> Option<FxcHessianCache> {
    reset_fxc_timing();
    let is_hf = scf.mol.xc_data.dfa_compnt_scf.is_empty();
    if is_hf {
        None
    } else {
        Some(prepare_fxc_hessian_cache(scf))
    }
}

/// Add the RKS fxc AO response to the in-loop `v_ao` buffer, in place.
///
/// This is the per-(atom,direction) fxc contribution to the mo_e1 vind
/// response: `v_ao += compute_fxc_response_ao_cached(cache, dm1)`. It is the
/// RKS analog of the `J - 0.5*hyb*K` terms that the caller has already added
/// to `v_ao` for both HF and RKS.
///
/// `dm1` and `v_ao` are the loop-local `[nao, nao]` MatrixFull buffers the
/// mo_e1 loop already owns; this function does not allocate. Iteration order
/// over `(p, q)` matches the inline block it replaced exactly.
pub fn add_fxc_to_v_ao(
    cache: &FxcHessianCache,
    dm1: &MatrixFull<f64>,
    v_ao: &mut MatrixFull<f64>,
) {
    let nao = cache.nao;
    let fxc_ao = compute_fxc_response_ao_cached(cache, dm1);
    for p in 0..nao {
        for q in 0..nao {
            v_ao[[p, q]] += fxc_ao[[p, q]];
        }
    }
}

/// Record the post-solve-phase fxc timings into the caller's timing profile.
///
/// Mirrors the inline block that previously lived in `calc_cphf_contrib`
/// after the Krylov solve: the `cphf: fxc_solve` total plus the
/// detailed per-subcomponent breakdown (matches PySCF `nr_rks_fxc` internals).
/// `cache_elapsed` is the wall-clock time spent building the cache (returned
/// by the caller from around its `prepare_fxc_cache` call).
pub fn record_fxc_solve_timings(
    timings: &mut Vec<(&'static str, std::time::Duration)>,
    cache_elapsed: std::time::Duration,
) {
    timings.push(("  cphf: fxc_cache", cache_elapsed));
    let fxc_solve_ns = (read_fxc_timing_s() * 1e9) as u64;
    timings.push(("  cphf: fxc_solve", std::time::Duration::from_nanos(fxc_solve_ns)));
    for (name, secs) in read_fxc_subtimings_s() {
        if secs > 0.0 {
            let label: &'static str = Box::leak(
                format!("    {}", name).into_boxed_str());
            timings.push((label, std::time::Duration::from_secs_f64(secs)));
        }
    }
}

/// Record the post-mo_e1-phase fxc timings into the caller's timing profile.
///
/// Mirrors the inline block that previously lived in `calc_cphf_contrib`
/// after the mo_e1 loop: the `cphf: fxc_moe1` total. The sub-component
/// counters were reset by the caller before the mo_e1 phase via
/// `reset_fxc_timing`.
pub fn record_fxc_moe1_timings(
    timings: &mut Vec<(&'static str, std::time::Duration)>,
) {
    let fxc_moe1_ns = (read_fxc_timing_s() * 1e9) as u64;
    timings.push(("  cphf: fxc_moe1", std::time::Duration::from_nanos(fxc_moe1_ns)));
}
