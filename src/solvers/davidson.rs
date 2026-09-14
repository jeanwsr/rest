use itertools::Itertools;
use log::{debug, info, trace, warn};
use rest_tensors::matrix::matrix_blas_lapack::{_dgeev, _dgemm_full, _dinverse, _dpotrf, _dsyev};
use rest_tensors::matrix::{MathMatrix, MatrixFull};
use std::cmp;
use std::time::Instant;

/// Davidson solver configuration, decoupled from QuasiParticle/TDDFTParameters.
#[derive(Debug, Clone)]
pub struct DavidsonConfig {
    pub max_subspace: usize,
    pub add_dim: usize,
    pub restart_dim: usize,
    pub max_iter: usize,
    pub tol: f64,
    /// Linear dependence threshold: vectors with `||v||^2 < lindep` are dropped.
    pub lindep: f64,
    /// Use Modified Gram-Schmidt (true) vs Classical GS (false) for
    /// orthogonalization of new trial vectors.
    pub use_mgs: bool,
    /// Detect divergence (`|r| > 1.0 && |r|/|r_last| > 3.0`) and restart
    /// the iteration from a clean subspace.
    pub divergence_restart: bool,
    /// Track eigenstate ordering across iterations via overlap matrices
    /// to prevent state flipping (useful for near-degenerate roots).
    pub track_states: bool,
}

impl Default for DavidsonConfig {
    fn default() -> Self {
        Self {
            max_subspace: 60,
            add_dim: 2,
            restart_dim: 6,
            max_iter: 100,
            tol: 1e-10,
            lindep: 1e-14,
            use_mgs: true,
            divergence_restart: true,
            track_states: false,
        }
    }
}

pub fn zip_and_sort(eigenvalues: &Vec<f64>, eigenvectors: &MatrixFull<f64>) -> Vec<(f64, Vec<f64>)> {
    let mut eigens: Vec<_> = eigenvalues
        .iter()
        .cloned()
        .zip(eigenvectors.iter_columns_full())
        .collect();
    eigens.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    eigens.into_iter().map(|(e, v)| (e, v.to_vec())).collect()
}
pub fn generate_initial_guess(diag: &Vec<f64>, nroots_ctrl: usize) -> MatrixFull<f64> {
    let mut indicies: Vec<_> = diag.iter().enumerate().collect();
    indicies.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let nroots = cmp::max(nroots_ctrl, 2);
    let indicies: Vec<_> = indicies.iter().take(nroots).map(|(i, _)| *i).collect();
    let mut initial_guess = MatrixFull::new([diag.len(), nroots], 0.0);
    indicies.iter().enumerate().for_each(|(k, i)| {
        initial_guess[[*i, k]] = 1.0;
    });
    initial_guess
}

/// Clamped preconditioner: `v / (omega - diag)` with floor on denominator.
fn precond(v: f64, omega: f64, diag: f64) -> f64 {
    let denom = omega - diag;
    if denom.abs() < 1e-8 {
        v / (if denom >= 0.0 { 1e-8 } else { -1e-8 })
    } else {
        v / denom
    }
}

fn reorder_by_overlap(
    v_prev: &MatrixFull<f64>,
    x_proj_nroots: &mut MatrixFull<f64>,
    omega_nroots: &mut Vec<f64>,
) {
    let n_prev = v_prev.size[1];
    let n_curr = omega_nroots.len().min(x_proj_nroots.size[1]);
    let n_comp = n_prev.min(n_curr);
    if n_comp == 0 {
        return;
    }
    let m = v_prev.size[0];
    let mut ovlp = MatrixFull::new([n_prev, n_curr], 0.0);
    let mut v_prev_sub = MatrixFull::new([m, n_prev], 0.0);
    for j in 0..n_prev {
        for i in 0..m {
            v_prev_sub[[i, j]] = v_prev[[i, j]];
        }
    }
    _dgemm_full(&v_prev_sub, 'T', &*x_proj_nroots, 'N', &mut ovlp, 1.0, 0.0);
    let mut ordering: Vec<usize> = Vec::new();
    for k in 0..n_comp {
        let mut max_idx = k;
        let mut max_abs = 0.0;
        for j in 0..n_prev {
            let ov = ovlp[[j, k]].abs();
            if ov > max_abs {
                max_abs = ov;
                max_idx = j;
            }
        }
        ordering.push(max_idx);
    }
    if ordering.iter().enumerate().any(|(i, &j)| i != j) && ordering.len() == n_comp {
        let mut reordered_omega = vec![0.0; n_curr];
        let mut reordered_x = MatrixFull::new([m, n_curr], 0.0);
        let mut reverse: Vec<Option<usize>> = vec![None; n_prev];
        for (k, &prev_idx) in ordering.iter().enumerate() {
            if prev_idx < n_prev {
                reverse[prev_idx] = Some(k);
            }
        }
        for j in 0..n_prev {
            if let Some(k) = reverse[j] {
                if k < n_curr {
                    reordered_omega[j] = omega_nroots[k];
                    for i in 0..m {
                        reordered_x[[i, j]] = x_proj_nroots[[i, k]];
                    }
                }
            }
        }
        *omega_nroots = reordered_omega;
        *x_proj_nroots = reordered_x;
    }
}

fn fmt_vec(vals: &[f64]) -> String {
    let items: Vec<String> = vals
        .iter()
        .map(|&v| {
            if v == 0.0 {
                "0".to_string()
            } else if v == f64::INFINITY {
                "inf".to_string()
            } else if v.abs() < 1e-3 {
                format!("{:.2e}", v)
            } else {
                format!("{:.3}", v)
            }
        })
        .collect();
    format!("[{}]", items.join(", "))
}

/// Orthogonalize `x` against subspace basis `ss`. CGS or MGS depending on flag.
fn orth_ss(x: &mut Vec<f64>, ss: &MatrixFull<f64>, use_mgs: bool) {
    if use_mgs {
        ss.iter_columns_full().for_each(|v| {
            let sv = v.to_vec();
            let prod = dot_product(&sv, x);
            *x = vector_scaled_add(x, 1.0, &sv, -prod);
        });
    } else {
        let orig = x.clone();
        let svs: Vec<Vec<f64>> = ss.iter_columns_full().map(|v| v.to_vec()).collect();
        let mut result = orig.clone();
        for sv in &svs {
            let prod = dot_product(sv, &orig);
            result = vector_scaled_add(&result, 1.0, sv, -prod);
        }
        *x = result;
    }
}

/// TDA Davidson eigensolver with a **batched** matvec interface.
///
/// Applies the operator to a whole block of trial vectors at once
/// (`FnMut(&MatrixFull) -> MatrixFull`), so callers can amortize work across
/// the subspace (e.g. batch the fxc/RI-J/K evaluation in AO-mode TDDFT).
/// Also available as the `tda_davidson_solver_batched` alias.
pub fn davidson_solver_batched<F1>(
    mut a_matvec_block: F1,
    nroots: usize,
    diag: &Vec<f64>,
    initial_guess: MatrixFull<f64>,
    config: &DavidsonConfig,
) -> Vec<(f64, Vec<f64>)>
where
    F1: FnMut(&MatrixFull<f64>) -> MatrixFull<f64>,
{
    let mut ss = initial_guess;
    let xlen = diag.len();
    let mut x_solutions = MatrixFull::new([xlen, 0], 0.0);
    let mut eigenvalues: Vec<f64> = Vec::new();
    let mut iter_num = 0;
    // For divergence restart: track previous step's max residual norm
    let mut max_dx_last: f64 = 1e9;
    let mut x_solutions_prev = MatrixFull::new([xlen, 0], 0.0);
    let mut eigenvalues_prev: Vec<f64> = Vec::new();
    // For eigenstate tracking
    let mut v_prev = MatrixFull::new([0, 0], 0.0);
    let mut e_prev: Vec<f64> = Vec::new();

    info!("    Davidson solver: tol ||r|| < {:.2e}, tol |de| < {:.2e}, nroots={}, dim={}",
        config.tol.sqrt(), config.tol, nroots, xlen);

    loop {
        let start = Instant::now();
        let restart = if ss.size[1] > config.max_subspace - config.add_dim {
            true
        } else {
            false
        };
        iter_num += 1;
        let m = ss.size[1];
        // Batched application of A to the whole subspace block.
        let a_ss = a_matvec_block(&ss);
        // debug!("m={}", m);
        let mut ss_t_a_ss = MatrixFull::new([m, m], 0.0);
        _dgemm_full(&ss, 'T', &a_ss, 'N', &mut ss_t_a_ss, 1.0, 0.0);
        // debug!("A projection:");
        // ss_t_a_ss.formated_output(1000,"full");
        drop(a_ss);
        let (Some(x_proj), omega, _) = _dsyev(&ss_t_a_ss, 'V') else {
            panic!("dsyev failure!")
        };
        let mut x_proj_nroots = MatrixFull::new([m, 0], 0.0);
        let mut omega_nroots: Vec<f64> = Vec::new();
        debug!("    omega: {}", fmt_vec(&omega));
        let collect_sol_num = if restart {
            config.restart_dim
        } else {
            cmp::min(ss.size[1], config.add_dim)
        };
        omega
            .iter()
            .zip(x_proj.iter_columns_full())
            .enumerate()
            .for_each(|(n, (omega_i, x_i))| {
                if n < collect_sol_num {
                    omega_nroots.push(*omega_i);
                    x_proj_nroots.push_column(x_i);
                }
            });
        let m = ss_t_a_ss.size[0];
        // Eigenstate tracking: reorder current roots to match previous ordering
        if config.track_states {
            if !e_prev.is_empty() {
                reorder_by_overlap(&v_prev, &mut x_proj_nroots, &mut omega_nroots);
            }
            v_prev = x_proj_nroots.clone();
            e_prev = omega_nroots.clone();
        }

        let mut x_full = MatrixFull::new([xlen, collect_sol_num], 0.0);
        _dgemm_full(&ss, 'N', &x_proj_nroots, 'N', &mut x_full, 1.0, 0.0);
        // Batched application of A to the collected solutions.
        let ax_full = a_matvec_block(&x_full);
        let mut residues = ax_full;
        let mut eigenvalue_matrix = MatrixFull::new([collect_sol_num, collect_sol_num], 0.0);
        (0..collect_sol_num).for_each(|i| eigenvalue_matrix[[i, i]] = omega[i]);
        let mut omega_x_full = MatrixFull::new([xlen, collect_sol_num], 0.0);
        _dgemm_full(&x_full, 'N', &eigenvalue_matrix, 'N', &mut omega_x_full, 1.0, 0.0);
        let mut omega_x = MatrixFull::new([xlen, collect_sol_num], 0.0);
        _dgemm_full(&x_full, 'N', &eigenvalue_matrix, 'N', &mut omega_x_full, 1.0, 0.0);
        residues = residues.scaled_add(&omega_x_full, -1.0).unwrap();
        let r_norms: Vec<f64> = residues.iter_columns_full().take(nroots)
            .map(|vec| vec.iter().map(|x| x * x).sum::<f64>().sqrt())
            .collect();
        eigenvalues = omega.iter().take(nroots).copied().collect();
        let de_vals: Vec<f64> = if eigenvalues_prev.is_empty() {
            Vec::new()
        } else {
            r_norms.iter().enumerate()
                .map(|(n, _)| {
                    if n < eigenvalues_prev.len() {
                        (omega[n] - eigenvalues_prev[n]).abs()
                    } else {
                        f64::INFINITY
                    }
                })
                .collect()
        };
        if de_vals.is_empty() {
            debug!("    residues: {}", fmt_vec(&r_norms));
        } else {
            debug!("    residues: {}, |de|: {}", fmt_vec(&r_norms), fmt_vec(&de_vals));
        }
        let mut converged = vec![false; nroots];
        let n_residues = residues.size[1];
        let mut max_residue_norm: f64 = 0.0;
        for (n, &norm) in r_norms.iter().enumerate() {
            if n < n_residues.min(nroots) {
                max_residue_norm = max_residue_norm.max(norm);
                let de_ok = de_vals.is_empty() || de_vals[n] <= config.tol;
                converged[n] = norm <= config.tol.sqrt() && de_ok;
            }
        }
        if n_residues < nroots {
            warn!("  only {} trial vectors for {} roots", n_residues, nroots);
        }
        let all_converged = converged.iter().all(|&c| c);
        let n_converged = converged.iter().filter(|&&c| c).count();
        // Divergence detection and restart
        if config.divergence_restart
            && !all_converged
            && max_residue_norm > 1.0
            && max_residue_norm / max_dx_last.max(1e-30) > 3.0
            && ss.size[1] > nroots + 2
        {
            warn!("  Divergence detected: |r|={:.3e}, |r_last|={:.3e}, restarting", max_residue_norm, max_dx_last);
            // Restore previous state
            x_solutions = x_solutions_prev.clone();
            eigenvalues = eigenvalues_prev.clone();
            ss = MatrixFull::new([xlen, 0], 0.0);
            x_solutions
                .iter_columns_full()
                .take(config.restart_dim)
                .for_each(|xi| ss.push_column(xi));
            residues.iter_columns_full().enumerate().for_each(|(i, residue_i)| {
                let mut preconditioned: Vec<f64> = residue_i
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| precond(v, omega[i], diag[j]))
                    .collect();
                trace!("Un-orthogonalized to add:{:#?}", preconditioned);
                orth_ss(&mut preconditioned, &ss, config.use_mgs);
                let orthogonalized_norm_sq = dot_product(&preconditioned, &preconditioned);
                if orthogonalized_norm_sq > config.lindep {
                    let orthogonalized_norm = orthogonalized_norm_sq.powf(0.5);
                    preconditioned = num_product(&preconditioned, 1.0 / orthogonalized_norm);
                    ss.push_column(&preconditioned);
                }
            });
            max_dx_last = 1e9;
            continue;
        }
        max_dx_last = max_residue_norm;

        if all_converged {
            info!("    iter {}: space={}, converged={}/{}", iter_num, ss.size[1], n_converged, nroots);
            x_full.iter_columns_full().enumerate().for_each(|(n, vec)| {
                if n < nroots {
                    x_solutions.push_column(vec)
                }
            });
            info!("Davidson Solver has converged.");
            trace!("    final round took {:?}", start.elapsed());
            break;
        }
        if iter_num > config.max_iter {
            break;
        }
        // Save current state for potential divergence recovery
        x_solutions_prev = x_solutions.clone();
        eigenvalues_prev = eigenvalues.clone();

        if !restart {
            residues.iter_columns_full().enumerate().for_each(|(i, residue_i)| {
                let mut preconditioned: Vec<f64> = residue_i
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| precond(v, omega[i], diag[j]))
                    .collect();
                trace!("Un-orthogonalized to add:{:#?}", preconditioned);
                orth_ss(&mut preconditioned, &ss, config.use_mgs);
                let orthogonalized_norm_sq = dot_product(&preconditioned, &preconditioned);
                if orthogonalized_norm_sq > config.lindep {
                    let orthogonalized_norm = orthogonalized_norm_sq.powf(0.5);
                    preconditioned = num_product(&preconditioned, 1.0 / orthogonalized_norm);
                    ss.push_column(&preconditioned);
                }
            });
        } else {
            debug!("Explicit Restart");
            ss = MatrixFull::new([xlen, 0], 0.0);
            x_full
                .iter_columns_full()
                .take(config.restart_dim)
                .for_each(|x_i| ss.push_column(x_i));
            residues.iter_columns_full().enumerate().for_each(|(i, residue_i)| {
                let mut preconditioned: Vec<f64> = residue_i
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| precond(v, omega[i], diag[j]))
                    .collect();
                trace!("Un-orthogonalized to add:{:#?}", preconditioned);
                orth_ss(&mut preconditioned, &ss, config.use_mgs);
                let orthogonalized_norm_sq = dot_product(&preconditioned, &preconditioned);
                if orthogonalized_norm_sq > config.lindep {
                    let orthogonalized_norm = orthogonalized_norm_sq.powf(0.5);
                    preconditioned = num_product(&preconditioned, 1.0 / orthogonalized_norm);
                    ss.push_column(&preconditioned);
                }
            });
        }
        info!("    iter {}: space={}, converged={}/{}", iter_num, ss.size[1], n_converged, nroots);
        // trace!("Search Space:");
        // ss.formated_output(1000,"full");
        trace!("This iteration took {:?}", start.elapsed());
    }
    let eigenvectors: Vec<_> = x_solutions
        .iter_columns_full()
        .map(|v1| {
            let mut vec = v1.to_vec();
            vec
        })
        .collect();
    let mut eigenpairs: Vec<_> = eigenvalues
        .iter()
        .zip(eigenvectors.iter())
        .map(|(value, vector)| (*value, vector.clone()))
        .collect();
    eigenpairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    eigenpairs
}

/// Backward-compatible per-vector wrapper around [`davidson_solver_batched`].
///
/// Adapters the batched closure interface to the legacy single-vector API by
/// applying the matvec column-by-column.
pub fn davidson_solver<F1>(
    mut a_matvec: F1,
    nroots: usize,
    diag: &Vec<f64>,
    initial_guess: MatrixFull<f64>,
    config: &DavidsonConfig,
) -> Vec<(f64, Vec<f64>)>
where
    F1: FnMut(&Vec<f64>) -> Vec<f64>,
{
    davidson_solver_batched(
        |block: &MatrixFull<f64>| -> MatrixFull<f64> {
            let mut out = MatrixFull::new([block.size[0], 0], 0.0);
            block.iter_columns_full().for_each(|col| {
                let z = col.to_vec();
                let r = a_matvec(&z);
                out.push_column(&r);
            });
            out
        },
        nroots,
        diag,
        initial_guess,
        config,
    )
}

/// Backward-compatible alias.
pub use davidson_solver as tda_davidson_solver;

/// Batched-interface alias: A applied to a whole block of trial vectors at once.
pub use davidson_solver_batched as tda_davidson_solver_batched;

/// Linear-response Davidson eigensolver with a **batched** matvec interface.
///
/// Applies A and B to a whole block of trial vectors at once
/// (`FnMut(&MatrixFull) -> MatrixFull` each), solving the symmetricized Casida
/// equation `(A−B)^(1/2)(A+B)(A−B)^(1/2) z = ω² z`.
pub fn lr_davidson_solver_batched<F1, F2>(
    mut a_matvec_block: F1,
    mut b_matvec_block: F2,
    nroots: usize,
    diag: &Vec<f64>,
    initial_guess: MatrixFull<f64>,
    config: &DavidsonConfig,
) -> Vec<(f64, Vec<f64>)>
where
    F1: FnMut(&MatrixFull<f64>) -> MatrixFull<f64>,
    F2: FnMut(&MatrixFull<f64>) -> MatrixFull<f64>,
{
    let mut ss = initial_guess;
    let xlen = diag.len();
    let mut x_solutions = MatrixFull::new([xlen, 0], 0.0);
    let mut y_solutions = MatrixFull::new([xlen, 0], 0.0);
    let mut eigenvalues: Vec<f64> = Vec::new();
    let mut iter_num = 0;
    let mut max_dx_last: f64 = 1e9;
    let mut x_solutions_prev = MatrixFull::new([xlen, 0], 0.0);
    let mut y_solutions_prev = MatrixFull::new([xlen, 0], 0.0);
    let mut eigenvalues_prev: Vec<f64> = Vec::new();

    info!("    LR Davidson solver: tol ||r|| < {:.2e}, tol |de| < {:.2e}, nroots={}, dim={}",
        config.tol.sqrt(), config.tol, nroots, xlen);

    loop {
        let start = Instant::now();
        iter_num += 1;
        let m = ss.size[1];
        // Batched application of A to the whole subspace block.
        let a_ss = a_matvec_block(&ss);
        debug!("m={}", m);
        let mut ss_t_a_ss = MatrixFull::new([m, m], 0.0);
        _dgemm_full(&ss, 'T', &a_ss, 'N', &mut ss_t_a_ss, 1.0, 0.0);
        // debug!("A projection:");
        // ss_t_a_ss.formated_output(1000,"full");
        drop(a_ss);
        // Batched application of B to the whole subspace block.
        let b_ss = b_matvec_block(&ss);
        let mut ss_t_b_ss = MatrixFull::new([m, m], 0.0);
        _dgemm_full(&ss, 'T', &b_ss, 'N', &mut ss_t_b_ss, 1.0, 0.0);
        // debug!("B projection:");
        // ss_t_b_ss.formated_output(1000,"full");
        let ss_t_amb_ss = ss_t_a_ss.scaled_add(&ss_t_b_ss, -1.0).unwrap();
        let ss_t_apb_ss = ss_t_a_ss.scaled_add(&ss_t_b_ss, 1.0).unwrap();
        // debug!("AmB projection:");
        // ss_t_amb_ss.formated_output(1000,"full");
        let mut g = ss_t_amb_ss.clone();
        let cholesky_ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut g_chol = ss_t_amb_ss.clone();
            _dpotrf(&mut g_chol, 'L');
            (0..m).cartesian_product(0..m).for_each(|(i, j)| {
                if i < j {
                    g_chol[[i, j]] = 0.0;
                }
            });
            g = g_chol;
        }));
        if cholesky_ok.is_err() {
            warn!("  Cholesky failed (A-B not positive-definite): this can happen with hybrid functionals.");
            warn!("  Falling back to TDA approximation for this iteration.");
        }
        let cholesky_result: Option<(MatrixFull<f64>, MatrixFull<f64>, Vec<f64>)> = if cholesky_ok.is_ok() {
            // Cholesky succeeded: use standard symmetrized Casida approach.
            // (A-B) = GG^T, solve G^T(A+B)G Z = omega^2 Z
            // where G = chol(S^T(A-B)S) is the lower-triangular Cholesky factor.
            // debug!("ApB projection:");
            // ss_t_apb_ss.formated_output(1000,"full");
            let mut apb_g = MatrixFull::new([m, m], 0.0);
            _dgemm_full(&ss_t_apb_ss, 'N', &g, 'N', &mut apb_g, 1.0, 0.0);
            let mut gt_apb_g = MatrixFull::new([m, m], 0.0);
            _dgemm_full(&g, 'T', &apb_g, 'N', &mut gt_apb_g, 1.0, 0.0);
            drop(apb_g);
            // ginv_xpy: G^{-1}(X+Y) — eigenvectors of G^T(A+B)G
            // omega2: squares of desired excitation energies
            let (Some(ginv_xpy), omega2, _) = _dsyev(&gt_apb_g, 'V') else {
                panic!("dsyev failure!")
            };

            let mut ginv_xpy_pos = MatrixFull::new([m, 0], 0.0);
            let mut omega2_pos: Vec<f64> = Vec::new();
            for (n, (w2, col)) in omega2.iter().zip(ginv_xpy.iter_columns_full()).enumerate() {
                if *w2 > 0.0 && omega2_pos.len() < nroots {
                    omega2_pos.push(*w2);
                    ginv_xpy_pos.push_column(col);
                }
            }
            if omega2_pos.is_empty() {
                None
            } else {
                let m_sub = g.size[0];
                let mut xpy_sub = MatrixFull::new([m_sub, nroots], 0.0);
                _dgemm_full(&g, 'N', &ginv_xpy_pos, 'N', &mut xpy_sub, 1.0, 0.0);
                let ginv = _dinverse(&g).expect("_dinverse");
                let mut xmy_sub = MatrixFull::new([m_sub, nroots], 0.0);
                _dgemm_full(&ginv, 'T', &ginv_xpy_pos, 'N', &mut xmy_sub, 1.0, 0.0);
                let omega_vec: Vec<f64> = omega2_pos.iter().map(|w| w.sqrt()).collect();
                Some((xpy_sub, xmy_sub, omega_vec))
            }
        } else {
            warn!("  Cholesky failed (A-B not positive-definite), switching to direct subspace solver");
            None
        };

        let (mut xpy, mut xmy, mut omega) = match cholesky_result {
            Some((xp, xm, ow)) => (xp, xm, ow),
            None => {
                // Cholesky fallback: solve [A B; -B -A] on the projected subspace
                // Build 2m x 2m matrix H = [A_proj,  B_proj; -B_proj, -A_proj]
                // Diagonalize via dgeev, keep real positive eigenvalues
                let mut h_full = MatrixFull::new([2 * m, 2 * m], 0.0);
                for i in 0..m {
                    for j in 0..m {
                        h_full[[i, j]] = ss_t_a_ss[[i, j]];
                    }
                }
                for i in 0..m {
                    for j in 0..m {
                        h_full[[i, m + j]] = ss_t_b_ss[[i, j]];
                    }
                }
                for i in 0..m {
                    for j in 0..m {
                        h_full[[m + i, j]] = -ss_t_b_ss[[i, j]];
                    }
                }
                for i in 0..m {
                    for j in 0..m {
                        h_full[[m + i, m + j]] = -ss_t_a_ss[[i, j]];
                    }
                }

                let (_, wr, wi, _vl, vr, _info) = _dgeev(&h_full, 'N', 'V');

                let mut pairs: Vec<(f64, Vec<f64>)> = wr
                    .iter()
                    .zip(wi.iter().zip(vr.iter_columns_full()))
                    .filter(|(wr_i, (wi_i, _))| **wr_i > 1e-4 && wi_i.abs() < 1e-6)
                    .map(|(wr_i, (_, v))| (*wr_i, v.to_vec()))
                    .collect();
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                pairs.truncate(nroots);

                if pairs.is_empty() {
                    warn!("  No positive eigenvalues found in subspace.");
                    return vec![];
                }

                let n_found = pairs.len();
                let mut xp = MatrixFull::new([m, n_found], 0.0);
                let mut xm = MatrixFull::new([m, n_found], 0.0);
                let mut ow: Vec<f64> = Vec::new();

                for (k, (val, vec)) in pairs.iter().enumerate() {
                    ow.push(*val);
                    for i in 0..m {
                        let vx = vec[i];
                        let vy = vec[m + i];
                        xp[[i, k]] = vx + vy;
                        xm[[i, k]] = vx - vy;
                    }
                    let omega_sqrt = val.sqrt();
                    for i in 0..m {
                        xp[[i, k]] /= omega_sqrt;
                        xm[[i, k]] *= omega_sqrt;
                    }
                }
                (xp, xm, ow)
            }
        };
        // debug!("Subspace xmy:");
        // xmy.formated_output(1000,"full");
        let n_omega = omega.len();
        for j in 0..n_omega {
            let eig = omega[j];
            for i in 0..m {
                xmy[[i, j]] *= eig.sqrt();
                xpy[[i, j]] /= eig.sqrt();
            }
        }
        let n_omega = omega.len();
        if n_omega == 0 {
            warn!("  No TDDFT roots found in subspace.");
            break;
        }
        debug!("    omega: {}", fmt_vec(&omega));
        let mut xmy_full = MatrixFull::new([xlen, n_omega], 0.0);
        let mut xpy_full = MatrixFull::new([xlen, n_omega], 0.0);
        _dgemm_full(&ss, 'N', &xmy, 'N', &mut xmy_full, 1.0, 0.0);
        _dgemm_full(&ss, 'N', &xpy, 'N', &mut xpy_full, 1.0, 0.0);
        let axmy = a_matvec_block(&xmy_full);
        let bxmy = b_matvec_block(&xmy_full);
        let ambxmy = axmy.scaled_add(&bxmy, -1.0).unwrap();
        let axpy = a_matvec_block(&xpy_full);
        let bxpy = b_matvec_block(&xpy_full);
        let apbxpy = axpy.scaled_add(&bxpy, 1.0).unwrap();
        // R_left  = (A-B)(X-Y) - Omega(X+Y)
        // R_right = (A+B)(X+Y) - Omega(X-Y)
        let mut left_residues = ambxmy;
        let mut right_residues = apbxpy;
        let mut eigenvalue_matrix = MatrixFull::new([n_omega, n_omega], 0.0);
        (0..n_omega).for_each(|i| eigenvalue_matrix[[i, i]] = omega[i]);
        let mut omega_xpy = MatrixFull::new([xlen, n_omega], 0.0);
        _dgemm_full(&xpy_full, 'N', &eigenvalue_matrix, 'N', &mut omega_xpy, 1.0, 0.0);
        let mut omega_xmy = MatrixFull::new([xlen, n_omega], 0.0);
        _dgemm_full(&xmy_full, 'N', &eigenvalue_matrix, 'N', &mut omega_xmy, 1.0, 0.0);
        left_residues = left_residues.scaled_add(&omega_xpy, -1.0).unwrap();
        right_residues = right_residues.scaled_add(&omega_xmy, -1.0).unwrap();
        let left_norms: Vec<f64> = left_residues.iter_columns_full()
            .map(|vec| vec.iter().map(|x| x * x).sum::<f64>().sqrt())
            .collect();
        let right_norms: Vec<f64> = right_residues.iter_columns_full()
            .map(|vec| vec.iter().map(|x| x * x).sum::<f64>().sqrt())
            .collect();
        eigenvalues = omega.iter().take(nroots).copied().collect();
        let de_vals: Vec<f64> = if eigenvalues_prev.is_empty() {
            Vec::new()
        } else {
            left_norms.iter().enumerate()
                .map(|(n, _)| {
                    if n < eigenvalues_prev.len() {
                        (omega[n] - eigenvalues_prev[n]).abs()
                    } else {
                        f64::INFINITY
                    }
                })
                .collect()
        };
        if de_vals.is_empty() {
            debug!("    residues: {}", fmt_vec(&left_norms));
        } else {
            debug!("    residues: {}, |de|: {}", fmt_vec(&left_norms), fmt_vec(&de_vals));
        }
        let mut left_converged = vec![false; nroots];
        let mut max_residue_norm: f64 = 0.0;
        for (n, &norm) in left_norms.iter().enumerate() {
            if n < nroots {
                max_residue_norm = max_residue_norm.max(norm);
                let de_ok = de_vals.is_empty() || de_vals[n] <= config.tol;
                left_converged[n] = norm <= config.tol.sqrt() && de_ok;
            }
        }
        let n_left_converged = left_converged.iter().filter(|&&c| c).count();
        let mut all_converged = left_converged.iter().all(|&c| c);
        if all_converged {
            for &norm in &right_norms {
                max_residue_norm = max_residue_norm.max(norm);
                if norm > config.tol.sqrt() {
                    all_converged = false;
                    break;
                }
            }
        }
        // Divergence detection and restart
        if config.divergence_restart
            && !all_converged
            && max_residue_norm > 1.0
            && max_residue_norm / max_dx_last.max(1e-30) > 3.0
            && ss.size[1] > nroots + 2
        {
            warn!("  Divergence detected: |r|={:.3e}, |r_last|={:.3e}, restarting", max_residue_norm, max_dx_last);
            x_solutions = x_solutions_prev.clone();
            y_solutions = y_solutions_prev.clone();
            eigenvalues = eigenvalues_prev.clone();
            ss = MatrixFull::new([xlen, 0], 0.0);
            x_solutions
                .iter_columns_full()
                .take(config.restart_dim)
                .for_each(|xi| ss.push_column(xi));
            max_dx_last = 1e9;
            continue;
        }
        max_dx_last = max_residue_norm;
        x_solutions_prev = x_solutions.clone();
        y_solutions_prev = y_solutions.clone();
        eigenvalues_prev = eigenvalues.clone();

        if all_converged {
            info!("    iter {}: space={}, converged=left:{}/{}", iter_num, ss.size[1], n_left_converged, nroots);
            let mut x = xmy_full.scaled_add(&xpy_full, 1.0).unwrap();
            let mut y = xpy_full.scaled_add(&xmy_full, -1.0).unwrap();
            x.self_multiple(0.5);
            y.self_multiple(0.5);
            x_solutions.append_column(&x);
            y_solutions.append_column(&y);
            info!("Davidson Solver has converged.");
            trace!("    final iteration took {:?}", start.elapsed());
            break;
        }
        if iter_num > config.max_iter {
            break;
        }
        left_residues.iter_columns_full().enumerate().for_each(|(i, residue)| {
            let mut preconditioned: Vec<f64> = residue
                .iter()
                .enumerate()
                .map(|(k, &v)| precond(v, omega[i], diag[k]))
                .collect();
            trace!("Un-orthogonalized to add:{:#?}", preconditioned);
            orth_ss(&mut preconditioned, &ss, config.use_mgs);
            let orthogonalizrd_norm_sq = dot_product(&preconditioned, &preconditioned);
            if orthogonalizrd_norm_sq > config.lindep {
                let orthogonalizrd_norm = orthogonalizrd_norm_sq.powf(0.5);
                debug!("Before normalization:{:#?},norm={}", preconditioned, orthogonalizrd_norm);
                preconditioned = num_product(&preconditioned, 1.0 / orthogonalizrd_norm);
                ss.push_column(&preconditioned);
            }
        });
        right_residues.iter_columns_full().enumerate().for_each(|(i, residue)| {
            let mut preconditioned: Vec<f64> = residue
                .iter()
                .enumerate()
                .map(|(k, &v)| precond(v, omega[i], diag[k]))
                .collect();
            trace!("Un-orthogonalized to add:{:#?}", preconditioned);
            orth_ss(&mut preconditioned, &ss, config.use_mgs);
            let orthogonalizrd_norm_sq = dot_product(&preconditioned, &preconditioned);
            if orthogonalizrd_norm_sq > config.lindep {
                let orthogonalizrd_norm = orthogonalizrd_norm_sq.powf(0.5);
                debug!("Before normalization:{:#?},norm={}", preconditioned, orthogonalizrd_norm);
                preconditioned = num_product(&preconditioned, 1.0 / orthogonalizrd_norm);
                preconditioned = num_product(&preconditioned, 1.0 / orthogonalizrd_norm);
                ss.push_column(&preconditioned);
            }
        });
        // trace!("Search Space:");
        // ss.formated_output(1000, "full");
        info!("    iter {}: space={}, converged=left:{}/{}", iter_num, ss.size[1], n_left_converged, nroots);
        trace!("This iteration took {:?}", start.elapsed());
    }
    let eigenvectors: Vec<_> = x_solutions
        .iter_columns_full()
        .zip(y_solutions.iter_columns_full())
        .map(|(v1, v2)| {
            let mut vec = v1.to_vec();
            vec.extend(v2);
            vec
        })
        .collect();
    let mut eigenpairs: Vec<_> = eigenvalues
        .iter()
        .zip(eigenvectors.iter())
        .map(|(value, vector)| (*value, vector.clone()))
        .collect();
    eigenpairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    eigenpairs
}

/// Backward-compatible per-vector wrapper around [`lr_davidson_solver_batched`].
///
/// Adapters the batched closure interface to the legacy single-vector API by
/// applying the matvecs column-by-column.
pub fn lr_davidson_solver<F1, F2>(
    mut a_matvec: F1,
    mut b_matvec: F2,
    nroots: usize,
    diag: &Vec<f64>,
    initial_guess: MatrixFull<f64>,
    config: &DavidsonConfig,
) -> Vec<(f64, Vec<f64>)>
where
    F1: FnMut(&Vec<f64>) -> Vec<f64>,
    F2: FnMut(&Vec<f64>) -> Vec<f64>,
{
    let a_apply = |block: &MatrixFull<f64>| -> MatrixFull<f64> {
        let mut out = MatrixFull::new([block.size[0], 0], 0.0);
        block.iter_columns_full().for_each(|col| {
            let z = col.to_vec();
            let r = a_matvec(&z);
            out.push_column(&r);
        });
        out
    };
    let b_apply = |block: &MatrixFull<f64>| -> MatrixFull<f64> {
        let mut out = MatrixFull::new([block.size[0], 0], 0.0);
        block.iter_columns_full().for_each(|col| {
            let z = col.to_vec();
            let r = b_matvec(&z);
            out.push_column(&r);
        });
        out
    };
    lr_davidson_solver_batched(a_apply, b_apply, nroots, diag, initial_guess, config)
}

pub fn dot_product(vec1: &Vec<f64>, vec2: &Vec<f64>) -> f64 {
    vec1.iter().zip(vec2.iter()).fold(0.0, |acc, (x1, x2)| acc + x1 * x2)
}
pub fn num_product(vec1: &Vec<f64>, num: f64) -> Vec<f64> {
    vec1.iter().map(|x| x * num).collect()
}
pub fn vector_scaled_add(vec1: &Vec<f64>, scale1: f64, vec2: &Vec<f64>, scale2: f64) -> Vec<f64> {
    vec1.iter()
        .zip(vec2.iter())
        .map(|(x1, x2)| x1 * scale1 + x2 * scale2)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a symmetric, diagonally-dominated matrix `[n, n]` from a
    /// deterministic pseudo-random pattern plus a strong diagonal, so that the
    /// Davidson subspace iteration converges cleanly for a few roots.
    fn synthetic_symmetric(n: usize, seed: f64) -> MatrixFull<f64> {
        let mut a = MatrixFull::new([n, n], 0.0);
        for i in 0..n {
            a[[i, i]] = 1.0 + (i as f64 + seed) * 0.5;
        }
        for i in 0..n {
            for j in 0..i {
                let v = ((i as f64 + seed) * (j as f64 + 0.3)).sin() * 0.05;
                a[[i, j]] = v;
                a[[j, i]] = v;
            }
        }
        a
    }

    fn per_vector_a(a: &MatrixFull<f64>) -> impl FnMut(&Vec<f64>) -> Vec<f64> + '_ {
        move |z: &Vec<f64>| -> Vec<f64> {
            let mut out = vec![0.0; a.size[0]];
            for i in 0..a.size[0] {
                for j in 0..a.size[1] {
                    out[i] += a[[i, j]] * z[j];
                }
            }
            out
        }
    }

    fn batched_a(a: &MatrixFull<f64>) -> impl FnMut(&MatrixFull<f64>) -> MatrixFull<f64> + '_ {
        move |block: &MatrixFull<f64>| -> MatrixFull<f64> {
            let mut out = MatrixFull::new([block.size[0], block.size[1]], 0.0);
            for s in 0..block.size[1] {
                for i in 0..a.size[0] {
                    let mut acc = 0.0;
                    for j in 0..a.size[1] {
                        acc += a[[i, j]] * block[[j, s]];
                    }
                    out[[i, s]] = acc;
                }
            }
            out
        }
    }

    #[test]
    fn test_batched_tda_matches_per_vector() {
        let n = 40;
        let a = synthetic_symmetric(n, 2.0);
        let hdiag: Vec<f64> = (0..n).map(|i| a[[i, i]]).collect();
        let nroots = 3;
        let config = DavidsonConfig {
            max_subspace: 12,
            add_dim: 3,
            restart_dim: 3,
            max_iter: 30,
            tol: 1e-10,
            ..Default::default()
        };
        let guess = generate_initial_guess(&hdiag, nroots);

        let per_vec = davidson_solver(per_vector_a(&a), nroots, &hdiag, guess.clone(), &config);
        let batched = davidson_solver_batched(batched_a(&a), nroots, &hdiag, guess.clone(), &config);

        assert_eq!(per_vec.len(), batched.len(), "same number of converged roots");
        for (e1, e2) in per_vec.iter().zip(batched.iter()) {
            assert!((e1.0 - e2.0).abs() < 1e-8,
                "eigenvalue mismatch: per-vector {} vs batched {}", e1.0, e2.0);
        }
    }

    #[test]
    fn test_batched_lr_matches_per_vector() {
        let n = 40;
        // LR path: keep both A and B strictly diagonal so A-B and A+B are
        // trivially positive definite (guarantees the Cholesky path finds
        // positive roots). B proportional to A gives clean ±sqrt(a²-b²) roots.
        let mut a = MatrixFull::new([n, n], 0.0);
        for i in 0..n {
            a[[i, i]] = 1.0 + i as f64 * 0.5;
        }
        let mut b = MatrixFull::new([n, n], 0.0);
        for i in 0..n {
            b[[i, i]] = 0.3 * a[[i, i]];
        }
        let hdiag: Vec<f64> = (0..n).map(|i| a[[i, i]]).collect();
        let nroots = 3;
        let config = DavidsonConfig {
            max_subspace: 12,
            add_dim: 3,
            restart_dim: 3,
            max_iter: 30,
            tol: 1e-10,
            ..Default::default()
        };
        let guess = generate_initial_guess(&hdiag, nroots);

        let per_vec = lr_davidson_solver(
            per_vector_a(&a), per_vector_a(&b),
            nroots, &hdiag, guess.clone(), &config);
        let batched = lr_davidson_solver_batched(
            batched_a(&a), batched_a(&b),
            nroots, &hdiag, guess.clone(), &config);

        assert_eq!(per_vec.len(), batched.len(), "same number of converged roots");
        for (e1, e2) in per_vec.iter().zip(batched.iter()) {
            assert!((e1.0 - e2.0).abs() < 1e-8,
                "eigenvalue mismatch: per-vector {} vs batched {}", e1.0, e2.0);
        }
    }
}
