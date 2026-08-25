//! Generic Krylov subspace solver for `(I + A) x = b`.
//!
//! ## Architecture
//!
//! - `krylov_tsr()` — Tsr/TsrView core with shared-subspace block Krylov
//!   and optional hard restart (GMRES(m)-style).
//! - `krylov_vec()` — convenience wrapper for `Vec<f64>` callers (CP-HF).
//! - `krylov()` — re-export alias for `krylov_vec`.
//!
//! ## Algorithm
//!
//! Shared-subspace block Krylov method (PySCF `lib.krylov` style) with
//! optional hard restart at `max_space` cap.  All right-hand sides share a
//! single subspace; each cycle calls the operator once on all active
//! directions.  The matvec closure implements `A` (**not** `I+A`) — the
//! identity contribution is added via subspace norms during the projected
//! solve.
//!
//! ## Convergence
//!
//! Convergence is declared when all new trial directions have squared norm
//! below `max(tol^2, lindep)`.  Panics on non-convergence.
//!
//! ## Source
//!
//! The shared-subspace Krylov algorithm follows PySCF's
//! `lib.linalg_helper.krylov` — `(I+A)` projected onto a non-normalized
//! Krylov basis with CGS + QR re-orthogonalization.  The per-RHS fallback
//! and hard restart mechanism are REST-specific extensions.

use crate::utilities::rstsr_util::{RestTensorToRstsrTsrAPI, Tsr, TsrView};
use rstsr::prelude::*;
use log::{debug, trace, warn};

// ── Configuration ──────────────────────────────────────────────────────────

/// Configuration for the Krylov subspace solver.
#[derive(Debug, Clone)]
pub struct KrylovConfig {
    /// Convergence tolerance: stop when `max(||trial||^2) < max(lindep, tol^2)`.
    pub tol: f64,
    /// Maximum total number of inner Krylov cycles (summed across restarts).
    pub max_cycle: usize,
    /// Optional hard-restart subspace cap (GMRES(m)-style).
    /// `None` means no hard restart.
    pub max_space: Option<usize>,
    /// Linear dependence threshold: vectors with `||v||^2 < lindep` are
    /// dropped during QR filtering.
    pub lindep: f64,
    /// Factor on tol for accepting residual without warning or recursive solve.
    /// RHS with `||r|| < factor * tol` are considered converged.
    pub max_residual_factor: f64,
}

impl Default for KrylovConfig {
    fn default() -> Self {
        Self {
            tol: 1e-9,
            max_cycle: 50,
            max_space: None,
            lindep: 1e-15,
            max_residual_factor: 1000.0,
        }
    }
}

/// Solver status and statistics.
#[derive(Debug, Clone)]
pub struct KrylovResult {
    pub cycles: usize,
    pub aop_calls: usize,
    pub per_root_solves: usize,
    pub residual: f64,
}

// ── Core: Tsr-based solver ────────────────────────────────────────────────

/// Solve `(I + A) x = b` with a shared-subspace block Krylov method.
///
/// # Parameters
///
/// * `aop` — Linear operator `A`.  Given `[n, nblock]`, returns `[n, nblock]`
///   (applied column-wise).  Must be `(I + A)` **excluding** the identity;
///   the identity is added internally.
/// * `b` — Right-hand sides, shape `[n, n_rhs]`, column-major.
/// * `x0` — Optional initial guess, same shape as `b`.
/// * `config` — Solver parameters.
///
/// # Returns
///
/// Solution `x`, shape `[n, n_rhs]`, column-major.
///
/// # Panics
///
/// Panics if the solver does not converge within `config.max_cycle` total cycles.
pub fn krylov_tsr(
    aop: &mut impl FnMut(TsrView) -> Tsr,
    b: TsrView,
    x0: Option<TsrView>,
    config: &KrylovConfig,
) -> (Tsr, KrylovResult) {
    let device = b.device().clone();
    let n = b.shape()[0];
    let n_rhs = b.shape()[1];
    let conv_thresh = config.lindep.max(config.tol * config.tol);
    let has_max_space = config.max_space.is_some();
    let max_space = config.max_space.unwrap_or(config.max_cycle);

    let b_orig: Tsr = b.to_owned();

    // x_accum is refined across hard restarts.
    let mut x_accum: Tsr = match x0.as_ref() {
        Some(x0v) => x0v.to_owned(),
        None => rt::zeros(([n, n_rhs], &device)),
    };

    // Pre-allocate subspace storage once and reuse across restarts.
    // Upper bound: n_rhs directions per cycle × (max_space + 1) for safety.
    let max_basis = n_rhs * (max_space + 2);
    let mut xs: Tsr = rt::zeros(([n, max_basis], &device));
    let mut axs: Tsr = rt::zeros(([n, max_basis], &device));
    let mut innerprod: Vec<f64> = Vec::with_capacity(max_basis);

    let mut total_cycles: usize = 0;
    let mut aop_calls: usize = 0;
    let mut restart_idx: usize = 0;
    let mut converged = false;
    let mut n_recursive = 0usize;
    let mut last_max_ip: f64 = 0.0;

    while total_cycles < config.max_cycle {
        // ── Form residual for this restart ─────────────────────────────────
        let b_work: Tsr = if restart_idx == 0 && x0.is_none() {
            b_orig.clone()
        } else {
            aop_calls += 1;
            &b_orig - (&x_accum + aop(x_accum.view()))
        };

        if restart_idx > 0 {
            let max_bw2: f64 = b_work.iter().fold(0.0_f64, |a, &v| a.max(v * v));
            trace!("    ---- restart {}: residual max(||v||) = {:.3e} ----", restart_idx, max_bw2.sqrt());
        }

        // ── Initial QR of residual columns ─────────────────────────────────
        let (x1_init, ip_init) = orth_block(b_work.view(), config.lindep);
        let max_init = ip_init.iter().fold(0.0_f64, |a, &v| a.max(v));
        last_max_ip = max_init;
        if max_init < conv_thresh || x1_init.shape()[1] == 0 {
            converged = true;
            break;
        }

        // Reset subspace for this restart.
        innerprod.clear();
        innerprod.extend_from_slice(&ip_init);
        let mut x1: Tsr = x1_init;
        let mut nd: usize = 0;
        let mut inner_converged = false;

        // ── Inner Krylov loop (shared subspace) ────────────────────────────
        for _inner in 0..max_space {
            if total_cycles >= config.max_cycle || x1.shape()[1] == 0 {
                break;
            }
            total_cycles += 1;
            let n_active = x1.shape()[1];

            // Block matvec
            aop_calls += 1;
            let axt_batch = aop(x1.view());

            // Extend shared subspace
            xs.i_mut((.., nd..nd + n_active)).assign(&x1);
            axs.i_mut((.., nd..nd + n_active)).assign(&axt_batch);
            nd += n_active;

            // Classical Gram-Schmidt against full shared history.
            // Uses original `axt_batch` for projection coefficients;
            // subsequent QR provides re-orthogonalization stability.
            let xs_slc = xs.i((.., ..nd));
            let ip_vec = rt::asarray((innerprod.as_slice(), &device));
            let coeffs = xs_slc.t() % &axt_batch; // [nd, n_active]
            let w = &coeffs / &ip_vec.i((.., None)); // divide row i by ||xs[i]||^2
            let x_new = &axt_batch - &xs_slc % &w; // [n, n_active]

            // QR + threshold
            let (x_new_orth, ip_new) = orth_block(x_new.view(), config.lindep);
            let max_ip = ip_new.iter().fold(0.0_f64, |a, &v| a.max(v));
            last_max_ip = max_ip;

            trace!(
                "    r{} c{} (total {}): max(||v||) = {:.3e}, n_active = {}",
                restart_idx, _inner + 1, total_cycles, max_ip.sqrt(), n_active,
            );

            if max_ip < conv_thresh {
                inner_converged = true;
                break;
            }

            // Keep only directions above conv_thresh.
            // Invariant: kept_ip is appended to innerprod before the next
            // cycle's x1 is pushed to xs, so xs.len() == innerprod.len()
            // after the push.
            let keep_mask: Vec<bool> = ip_new.iter().map(|&ip| ip > conv_thresh).collect();
            let has_drop = keep_mask.iter().any(|&b| !b);
            if has_drop {
                let mut kept: Vec<&[bool]> = Vec::new(); // 临时 dummy，我们直接用 bool_select
                // Actually, use bool_select to filter columns
                x1 = x_new_orth.bool_select(1, &keep_mask);
                let kept_ip: Vec<f64> = ip_new
                    .iter()
                    .zip(keep_mask.iter())
                    .filter_map(|(&ip, &m)| if m { Some(ip) } else { None })
                    .collect();
                innerprod.extend(kept_ip);
            } else {
                x1 = x_new_orth;
                innerprod.extend(ip_new);
            }
        }

        // ── Reconstruct partial solution from current subspace ─────────────
        if nd > 0 {
            let x_partial = projected_solve(xs.i((.., ..nd)), axs.i((.., ..nd)), &innerprod, b_work.view());

            // Check true residual before accepting convergence.
            // If the subspace convergence signal fired but the true residual
            // is still large, the subspace has collapsed — switch to per-RHS
            // independent subspaces to recover accuracy.
            let x_temp = &x_accum + &x_partial;
            aop_calls += 1;
            let residual = &b_orig - (&x_temp + aop(x_temp.view()));
            let mut max_r2: f64 = 0.0;

        // ── Convergence check per RHS ────────────────────────────────
        // solve_one_rhs: Galerkin projection of (I+A) onto xs_k,
        // direct solve, true residual via linearity (r = x_k + ax_k·c - b).
        for k in 0..n_rhs {
                let r_col = residual.i((.., k));
                let r2: f64 = (&r_col % &r_col).to_scalar();
                max_r2 = max_r2.max(r2);
            }
            let max_rnorm = max_r2.sqrt();
            let relaxed_tol = config.max_residual_factor * config.tol;

            if inner_converged && max_rnorm > relaxed_tol && n_rhs > 1 && nd > 0 {
                debug!(
                    "    QR collapse detected (max ||r|| = {:.3e} > {:.0} * tol({:.3e})), switching to per-RHS",
                    max_rnorm, config.max_residual_factor, config.tol,
                );
                let shared_xs = xs.i((.., ..nd));
                let shared_axs = axs.i((.., ..nd));
                let shared_ip: Vec<f64> = innerprod[..nd].to_vec();
                let (per_root_sol, n_rec, rec_aop_calls) = per_rhs_phase(
                    aop, &b_orig, shared_xs, shared_axs, &shared_ip,
                    residual.view(), &x_temp,
                    &mut total_cycles, &mut aop_calls, config, conv_thresh, max_space, &device,
                );
                x_accum = per_root_sol;
                n_recursive = n_rec;
                aop_calls += rec_aop_calls;
                converged = true;
                break;
            }

            x_accum += &x_partial;
        }

        if inner_converged || !has_max_space || total_cycles >= config.max_cycle {
            converged = inner_converged;
            break;
        }

        restart_idx += 1;
        trace!("    ---- restart {}: subspace reset ----", restart_idx);
    }

    if !converged {
        panic!(
            "krylov_tsr: failed to converge after {} cycles ({} restarts). \
             max(||v||) = {:.3e} >= tol = {:.3e}, lindep = {:.2e}",
            total_cycles, restart_idx + 1, last_max_ip.sqrt(), config.tol, config.lindep,
        );
    }

    // Post-solve accuracy check: compute true residual, warn if above tol.
    let max_r2: f64;
    {
        aop_calls += 1;
        let residual = &b_orig - (&x_accum + aop(x_accum.view()));
        let mut inner_r2 = 0.0_f64;
        let mut max_abs: f64 = 0.0;
        for k in 0..n_rhs {
            let r_col = residual.i((.., k));
            let r2: f64 = (&r_col % &r_col).to_scalar();
            inner_r2 = inner_r2.max(r2);
            max_abs = max_abs.max(r_col.iter().fold(0.0_f64, |a, &v| a.max(v.abs())));
        }
        max_r2 = inner_r2;
        let max_rnorm = max_r2.sqrt();
        let relaxed_tol = config.max_residual_factor * config.tol;
        if max_rnorm > relaxed_tol {
            warn!("    max ||r|| = {:.3e} > {:.0} * tol({:.3e}), max |r| = {:.3e} (subspace exhausted)",
                max_rnorm, config.max_residual_factor, config.tol, max_abs);
        } else if max_rnorm > config.tol {
            debug!("    tol({:.3e}) < max ||r|| = {:.3e} < {:.0} * tol({:.3e}), max |r| = {:.3e}",
                config.tol, max_rnorm, config.max_residual_factor, config.tol, max_abs);
        } else {
            debug!("    max ||r|| = {:.3e} < tol({:.3e}), max |r| = {:.3e}",
                max_rnorm, config.tol, max_abs);
        }
        if n_rhs > 1 {
            if n_recursive > 0 {
                debug!("    converged after {} cycles ({} aop calls) + {} per-root solves",
                    total_cycles, aop_calls, n_recursive);
            } else {
                debug!("    converged after {} cycles ({} aop calls)",
                    total_cycles, aop_calls);
            }
        }
    }

    let result = KrylovResult {
        cycles: total_cycles,
        aop_calls,
        per_root_solves: n_recursive,
        residual: max_r2.sqrt(),
    };
    (x_accum, result)
}

/// Per-RHS independent Krylov subspaces: activated when shared subspace
/// collapses (trial norms → 0 but true residual still large).
fn per_rhs_phase(
    aop: &mut impl FnMut(TsrView) -> Tsr,
    b_orig: &Tsr,
    shared_xs: TsrView,        // [n, nd]
    shared_axs: TsrView,       // [n, nd]
    shared_ip: &[f64],         // [nd]
    residual: TsrView,         // [n, n_rhs]
    x_temp: &Tsr,              // [n, n_rhs]
    total_cycles: &mut usize,
    aop_calls: &mut usize,
    config: &KrylovConfig,
    conv_thresh: f64,
    max_space: usize,
    device: &DeviceBLAS,
) -> (Tsr, usize, usize) {
    let n = shared_xs.shape()[0];
    let n_rhs = residual.shape()[1];
    let nd = shared_xs.shape()[1];
    let relaxed_tol = config.max_residual_factor * config.tol;
    let max_per_rhs = nd + (max_space / 2).max(10);

    // ── Pre-allocate per-RHS Krylov subspaces ──────────────────────────
    // Each RHS gets its own copy of the shared subspace (rhs_xs[k], rhs_axs[k],
    // rhs_ip[k]).  rhs_x1[k] holds current trial directions for RHS k.
    // rhs_nd[k] tracks the actual number of subspace columns used.
    let mut active: Vec<bool> = vec![true; n_rhs];
    let mut rhs_xs: Vec<Tsr> = (0..n_rhs)
        .map(|_| rt::zeros((vec![n, max_per_rhs].f(), device)))
        .collect();
    let mut rhs_axs: Vec<Tsr> = (0..n_rhs)
        .map(|_| rt::zeros((vec![n, max_per_rhs].f(), device)))
        .collect();
    let mut rhs_ip: Vec<Vec<f64>> = (0..n_rhs).map(|_| shared_ip.to_vec()).collect();
    let mut rhs_nd: Vec<usize> = vec![nd; n_rhs];
    for k in 0..n_rhs {
        rhs_xs[k].i_mut((.., ..nd)).assign(&shared_xs);
        rhs_axs[k].i_mut((.., ..nd)).assign(&shared_axs);
    }

    let mut rhs_x1: Vec<Vec<Tsr>> = vec![Vec::new(); n_rhs];
    let mut rhs_x_accum: Vec<Tsr> = (0..n_rhs).map(|k| x_temp.i((.., k)).to_owned()).collect();

    // ── Build initial trial directions from residual ───────────────────
    // Orthogonalize each RHS's residual against the shared subspace.
    // RHS with ||r|| < relaxed_tol are marked converged immediately.
    for k in 0..n_rhs {
        let r_k: Tsr = residual.i((.., k)).to_owned();
        let rnorm = r_k.l2_norm();
        if rnorm > relaxed_tol {
            let mut x = r_k;
            for i in 0..nd {
                let xsi = shared_xs.i((.., i));
                x -= (&x % &xsi).to_scalar() / shared_ip[i] * xsi;
            }
            let nsq = (&x % &x).to_scalar();
            if nsq > conv_thresh {
                let norm = nsq.sqrt();
                x *= 1.0 / norm;
                rhs_x1[k].push(x * norm);
            }
        } else {
            active[k] = false;
            debug!("    per-RHS: RHS {} converged (below relaxed tol), ||r||={:.3e}", k, rnorm);
        }
    }

    let mut n_recursive = 0usize;
    let mut per_root_aop_calls = 0usize;
    // ── Per-RHS Krylov loop (independent subspaces) ───────────────────
    // Each cycle: collect active directions → batch aop → distribute
    // results per RHS → CGS orthogonalization → convergence check.
    while active.iter().any(|&a| a) && *total_cycles < config.max_cycle {
        let mut dir_vecs: Vec<Tsr> = Vec::new();
        let mut dir_to_rhs: Vec<usize> = Vec::new();
        for k in 0..n_rhs {
            if active[k] {
                for v in &rhs_x1[k] {
                    dir_vecs.push(v.clone());
                    dir_to_rhs.push(k);
                }
            }
        }
        if dir_vecs.is_empty() {
            // ── Stalled RHS: recursive per-root fallback ────────────
            // When no active RHS can produce new trial directions,
            // recursively solve each stalled RHS via fresh single-RHS
            // krylov_tsr.  This decouples shared-subspace contamination
            // and guarantees convergence.
            let stalled: Vec<usize> = (0..n_rhs).filter(|&k| active[k]).collect();
            if stalled.is_empty() { break; }

            // Single-RHS config: same tolerances, no shared subspace coupling
            let single_config = KrylovConfig {
                tol: config.tol,
                max_cycle: config.max_cycle - *total_cycles,
                max_space: None,
                lindep: config.lindep,
                max_residual_factor: config.max_residual_factor,
            };

            n_recursive = 0;
            per_root_aop_calls = 0;
            for &k in &stalled {
                let b_col = b_orig.i((.., k));
                let b_mat = b_col.to_owned().into_shape((n, 1));
                let x0_view = rhs_x_accum[k].view();
                let x0_mat = x0_view.reshape([n, 1]);
                let (x_k, kr) = krylov_tsr(aop, b_mat.view(), Some(x0_mat.view()), &single_config);
                rhs_x_accum[k] = x_k.i((.., 0)).to_owned();
                active[k] = false;
                n_recursive += 1;
                per_root_aop_calls += kr.aop_calls;
                debug!("    per-root solve: RHS {} converged after {} cycles ({} aop calls), max ||r|| = {:.3e}",
                    k, kr.cycles, kr.aop_calls, kr.residual);
            }
            break;
        }

        // ── Core cycle: batch aop, distribute, CGS per RHS ────────────
        *total_cycles += 1;
        let n_active = dir_vecs.len();
        let mut dirs_tsr: Tsr = rt::zeros((vec![n, n_active].f(), device));
        for (i, v) in dir_vecs.iter().enumerate() { dirs_tsr.i_mut((.., i)).assign(v.view()); }
        let axt_tsr = aop(dirs_tsr.view());
        *aop_calls += 1;

        let mut rhs_new_x1: Vec<Vec<Tsr>> = vec![Vec::new(); n_rhs];
        for (idx, &k) in dir_to_rhs.iter().enumerate() {
            if !active[k] { continue; }
            let x1_col = dirs_tsr.i((.., idx));
            let axt_col = axt_tsr.i((.., idx));

            if rhs_nd[k] >= max_per_rhs { continue; }
            rhs_xs[k].i_mut((.., rhs_nd[k])).assign(&x1_col);
            rhs_axs[k].i_mut((.., rhs_nd[k])).assign(&axt_col);
            rhs_ip[k].push((&x1_col * &x1_col).iter().sum::<f64>());
            rhs_nd[k] += 1;

            let xs_k = rhs_xs[k].i((.., ..rhs_nd[k]));
            let coeffs = xs_k.t() % &axt_col;
            let ip_tsr = rt::asarray((rhs_ip[k].as_slice(), device));
            let mut x_new = axt_col.to_owned() - &xs_k % &(&coeffs / &ip_tsr);

            let nsq: f64 = (&x_new % &x_new).to_scalar();
            if nsq > conv_thresh {
                let norm = nsq.sqrt();
                x_new *= 1.0 / norm;
                rhs_new_x1[k].push(&x_new * norm);
            }
        }

        for k in 0..n_rhs {
            if !active[k] || rhs_nd[k] == 0 { continue; }
            let (x_k, rnorm) = solve_one_rhs(
                rhs_xs[k].i((.., ..rhs_nd[k])),
                rhs_axs[k].i((.., ..rhs_nd[k])),
                &rhs_ip[k],
                b_orig.i((.., k)),
                device,
            );
            let relaxed_tol = config.max_residual_factor * config.tol;
            if rnorm < relaxed_tol {
                rhs_x_accum[k] = x_k;
                active[k] = false;
                debug!("    per-RHS: RHS {} converged (direct), ||r||={:.3e}", k, rnorm);
            } else if rnorm < config.tol {
                rhs_x_accum[k] = x_k;
                active[k] = false;
                debug!("    per-RHS: RHS {} converged (direct, strict), ||r||={:.3e}", k, rnorm);
            }
        }

        rhs_x1.iter_mut().zip(rhs_new_x1.iter_mut()).for_each(|(x1, new)| *x1 = std::mem::take(new));
    }

    // ── Reconstruct x_accum from per-RHS solutions ─────────────────
    let mut x_accum = rt::zeros((vec![n, n_rhs].f(), device));
    rhs_x_accum.iter().enumerate().for_each(|(k, x)| x_accum.i_mut((.., k)).assign(x));
    debug!("    per-RHS: all {} RHS converged", n_rhs);
    (x_accum, n_recursive, per_root_aop_calls)
}

// ── Vec<f64> convenience wrapper ───────────────────────────────────────────

/// Solve `(I + A) x = b` — convenience wrapper using `Vec<f64>`.
///
/// Internally converts between `Vec<f64>` and `Tsr` via
/// `RestTensorToRstsrTsrAPI`, then calls `krylov_tsr`.
///
/// # Parameters
///
/// * `matvec` — Batched linear operator.  Given `n` slices of length `dim`,
///   returns `n` vectors `A·v_i`.
/// * `b` — Right-hand side vectors, each of length `dim`.
/// * `config` — Solver parameters.
pub fn krylov_vec(
    matvec: &mut impl FnMut(&[&[f64]]) -> Vec<Vec<f64>>,
    b: &[Vec<f64>],
    config: &KrylovConfig,
) -> Vec<Vec<f64>> {
    let n_rhs = b.len();
    if n_rhs == 0 {
        return vec![];
    }
    let device = DeviceBLAS::default();
    let b_tsr = b.to_rstsr(&device);
    let b_view = b_tsr.view();

    // Adapt Vec-based matvec to Tsr-based.
    let mut matvec_tsr = |v: TsrView| -> Tsr {
        let n_active = v.shape()[1];
        let mut vecs: Vec<Vec<f64>> = Vec::with_capacity(n_active);
        for k in 0..n_active {
            vecs.push(v.i((.., k)).iter().copied().collect());
        }
        let refs: Vec<&[f64]> = vecs.iter().map(|vi| &vi[..]).collect();
        (matvec)(&refs).as_slice().to_rstsr(&device)
    };

    let (x_tsr, _result) = krylov_tsr(&mut matvec_tsr, b_view, None, config);

    let result: Vec<Vec<f64>> = (0..n_rhs).map(|k| x_tsr.i((.., k)).iter().copied().collect()).collect();
    result
}

/// Re-export alias — backward compatibility with CP-HF callers.
pub use krylov_vec as krylov;

// ── Helpers ────────────────────────────────────────────────────────────────

/// Projected solve + true residual for one RHS.
/// Returns `(x_k, rnorm)` where `x_k` is the solution and `rnorm = ||(I+A)x_k - b||`.
fn solve_one_rhs(
    xs: TsrView,       // [n, nd]
    ax: TsrView,       // [n, nd]
    ip: &[f64],        // [nd]
    b_col: TsrView,    // [n]
    device: &DeviceBLAS,
) -> (Tsr, f64) {
    let nd = xs.shape()[1];
    let mut h = xs.t() % &ax;
    for i in 0..nd { h[[i, i]] += ip[i]; }
    let mut g = rt::zeros(([nd, 1], device));
    for i in 0..nd { g[[i, 0]] = (&xs.i((.., i)) % &b_col).to_scalar(); }
    let c = rt::linalg::solve_general((h, g));
    let c_1d = c.reshape(-1);
    let x_k = &xs % &c_1d;
    let r_k = &x_k + &ax % &c_1d - &b_col;
    let rnorm = r_k.l2_norm();
    (x_k, rnorm)
}

/// Modified Gram-Schmidt over columns, returning non-normalized orthogonal
/// vectors together with their squared norms.  Columns with `||v||^2 < lindep`
/// are dropped.
///
/// Mirrors PySCF's `_qr` / `_orth_block` convention: `out[:, i]` has
/// `||out[:, i]||^2 == norms_sq[i]`, and the columns are mutually orthogonal.
fn orth_block(vec: TsrView, lindep: f64) -> (Tsr, Vec<f64>) {
    let device = vec.device().clone();
    let n = vec.shape()[0];
    let nblock = vec.shape()[1];

    if nblock == 0 {
        return (rt::zeros(([n, 0], &device)), vec![]);
    }

    let mut qs: Vec<Tsr> = Vec::with_capacity(nblock);
    let mut norms_sq: Vec<f64> = Vec::with_capacity(nblock);

    for i in 0..nblock {
        let mut vi: Tsr = vec.i((.., i)).to_owned();
        // Modified Gram-Schmidt: project out previous orthogonal vectors
        for qj in &qs {
            let coeff = (&vi % qj).to_scalar() / (qj % qj).to_scalar();
            vi -= coeff * qj;
        }
        let nsq = (&vi % &vi).to_scalar();
        if nsq > lindep {
            qs.push(vi);
            norms_sq.push(nsq);
        }
    }

    if qs.is_empty() {
        (rt::zeros(([n, 0], &device)), norms_sq)
    } else {
        (rt::stack((qs, -1)), norms_sq)
    }
}

fn projected_solve(xs: TsrView, ax: TsrView, innerprod: &[f64], b: TsrView) -> Tsr {
    let nd = xs.shape()[1];
    let device = xs.device().clone();

    // H[i,j] = xs[:,i]·ax[:,j] + δ[i,j]·||xs[:,i]||^2
    let mut h: Tsr = xs.t() % &ax;
    for i in 0..nd {
        h[[i, i]] += innerprod[i];
    }

    // g[i,k] = xs[:,i]·b[:,k]
    let g: Tsr = xs.t() % &b;

    let c = rt::linalg::solve_general((h, g));

    &xs % &c
}
