// ============================================================================
// BSE-specific FEAST driver: pre-processing and post-processing only.
//
// The context-independent FEAST algorithm (and its CG/GMRES sub-solvers) now
// lives in `crate::solvers::feast`.  This module builds the BSE operator
// closures from the SCF/RI data, calls the generic FEAST solver, and converts
// the returned vectors back to the Davidson-style `[X; Y]` layout.
//
// The renormalized-doubles workflow (a first FEAST solve in a small auxiliary
// basis followed by a Rayleigh–Ritz projection in the full auxiliary basis)
// keeps both of those stages here; only the intermediate FEAST eigenvalue
// solve is delegated to `solvers::feast`.
// ============================================================================
use crate::ctrl_io::quasiparticle_methods::QuasiParticle;
use crate::ri_bse::matvec;
use crate::ri_bse::{construct_energy_diag_for_a, construct_inverse_dielectric, get_submatrix};
use crate::ri_gw::get_occupation_parameters;
use crate::scf_io::SCF;
use crate::solvers::feast::{cg, feast};
use crate::tensors::MathMatrix;
use rest_tensors::matrix::matrix_blas_lapack::{
    _dgeev, _dgemm_full, _dgemm_scaled, _dinverse, _dpotrf, _dsyevd,
};
use rest_tensors::{BasicMatrix, MatrixFull};
use std::sync::Arc;
use std::time::Instant;

// ============================================================================
// Section 4: BSE-specific FEAST solver entry points
// ============================================================================

/// Reconstruct a Davidson-style `[X; Y]` eigenvector (length 2n) from a FEAST
/// non-TDA X-block eigenvector.
///
/// FEAST's non-TDA path solves the squared Hermitian problem
///   (A−B)·(A+B)⁻¹·u = ω²·u        (GEP: operator = (A−B), metric = (A+B)⁻¹)
/// whose eigenvector is `u = X+Y` (length n = occ·vir). From the BSE identity
///   (A−B)(X+Y) = ω·(X−Y)
/// we recover
///   X = (u + (A−B)u / ω) / 2
///   Y = (u − (A−B)u / ω) / 2
/// and return them stacked as `[X ; Y]` (length 2n), matching the layout of
/// `lr_davidson_solver` so that `export_pysoc_json`, `dipoles::normalize`, etc.
/// work unchanged.
///
/// # Arguments
/// * `xpy_block` - the FEAST eigenvector `u = X+Y` (length n)
/// * `amb_xpy`   - `(A−B)·xpy_block`, i.e. `ω·(X−Y)` (length n); caller computes
///                 this with whatever RI integrals are currently in scope
/// * `omega`     - the excitation energy ω (> 0)
fn reconstruct_xy_from_xpy(xpy_block: &[f64], amb_xpy: &[f64], omega: f64) -> Vec<f64> {
    debug_assert_eq!(xpy_block.len(), amb_xpy.len());
    let n = xpy_block.len();
    let mut full = vec![0.0; 2 * n];
    for i in 0..n {
        let x = (xpy_block[i] + amb_xpy[i] / omega) * 0.5;
        let y = (xpy_block[i] - amb_xpy[i] / omega) * 0.5;
        full[i] = x;
        full[n + i] = y;
    }
    full
}

/// Reconstruct FEAST non-TDA eigenpairs into Davidson-style `[X;Y]` layout.
///
/// FEAST non-TDA returns eigenvectors that are length `n = occ·vir` and live in
/// the "X+Y space" (the squared-problem eigenvector `u = X+Y`, recovered as
/// `(A−B)u/ω + u = 2X` by `feast_solve_bse_nontda`). Downstream consumers
/// (`export_pysoc_json`, `dipoles::normalize(_, false)`, `leading_components`)
/// expect the Davidson `[X;Y]` layout (length 2n). This helper applies the BSE
/// identity `(A−B)(X+Y) = ω(X−Y)` to rebuild `[X;Y]` for every eigenpair.
///
/// Builds the (A−B) matvec from the regular RI integrals (mirrors
/// `feast_solve_bse_nontda`'s construction). `qp_ctrl.bse_spin` selects the
/// spin channel.
fn reconstruct_nontda_pairs_to_xy(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    eigenpairs: Vec<(f64, Vec<f64>)>,
    occ_size: usize,
    vir_size: usize,
) -> Vec<(f64, Vec<f64>)> {
    if eigenpairs.is_empty() {
        return eigenpairs;
    }
    let ks_energies: Vec<f64> = scf_data.eigenvalues[0].clone();
    let epsilon: Vec<f64> = if qp_ctrl.bse_qp_polarization {
        scf_data.gwqp.0.clone()
    } else {
        ks_energies
    };
    let inverse_dielectric = construct_inverse_dielectric(scf_data, &epsilon);
    // AO ("ao") or MO ("mo") backend.  In the renormalized-doubles workflow this
    // runs after `rimatr_bse` has been cleared, so an AO context built here folds
    // the *regular* `rimatr` (see `BseMatvec`).
    let mv = super::BseMatvec::new(scf_data, &inverse_dielectric, true);

    // (A−B) matvec closure
    let amb_matvec = |z: &Vec<f64>| -> Vec<f64> { mv.amb(scf_data, qp_ctrl, z) };

    eigenpairs
        .into_iter()
        .map(|(omega, vec)| {
            let amb = amb_matvec(&vec);                  // (A−B)(X+Y) = ω(X−Y)
            let xy = reconstruct_xy_from_xpy(&vec, &amb, omega);  // [X;Y], length 2n
            (omega, xy)
        })
        .collect()
}

/// Build s-only matvec closures and diag for use as inner GMRES preconditioner.
/// Must be called while `ri3fn_bse`/`rimatr_bse` are still populated (before clearing).
/// Returns (a_mul, b_mul, diag) where a_mul/b_mul are `Box<dyn Fn>` closures
/// and diag is the diagonal of A (energy gaps) used for inner GMRES.
fn build_s_only_precond_data(
    scf_data: &mut SCF,
    qp_ctrl: &QuasiParticle,
    quasiparticle_energies: &Vec<f64>,
    occ_size: usize,
    vir_size: usize,
    bse_tda: bool,
) -> (
    Option<Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync>>,
    Option<Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync>>,
    Option<Vec<f64>>,
) {
    let ks_energies: Vec<f64> = scf_data.eigenvalues[0].clone();
    let epsilon: Vec<f64> = if qp_ctrl.bse_qp_polarization {
        quasiparticle_energies.clone()
    } else {
        ks_energies.clone()
    };
    let inverse_dielectric = construct_inverse_dielectric(scf_data, &epsilon);

    // Build QP energy gaps diagonal
    let energies: Vec<f64> = if qp_ctrl.bse_qp_polarization {
        scf_data.gwqp.0.clone()
    } else {
        scf_data.eigenvalues[0].clone()
    };
    let diag_a: Vec<f64> = construct_energy_diag_for_a(&energies, occ_size, vir_size);

    let qp_ctrl_c = qp_ctrl.clone();

    if bse_tda {
        // The s-only preconditioner is assembled from MO-basis RI tensors, which
        // the AO matvec deliberately never materialises.  Rather than build them
        // just for a preconditioner (and hold two fold sets at once), fall back to
        // the diagonal preconditioner that the FEAST solver already supports.
        if crate::ctrl_io::quasiparticle_methods::style_is_ao(&qp_ctrl.bse_matvec_style) {
            println!(
                "bse_matvec_style = \"ao\": skipping the s-only inner-GMRES preconditioner \
                 (it is built from MO-basis tensors); using the diagonal preconditioner. \
                 The FEAST matvec itself runs on the AO path."
            );
            return (None, None, Some(diag_a));
        }
        // ── TDA: a_mul = A_matvec(s-only), b_mul = identity ──
        let num_auxbas = inverse_dielectric.size[0];
        let ri_ov = get_submatrix(scf_data, 'O', 'V', 'N');
        let mut ri_vv = get_submatrix(scf_data, 'V', 'V', 'N');
        let ri_oo = get_submatrix(scf_data, 'O', 'O', 'N');

        let mut ri_oo_tilde = MatrixFull::new(ri_oo.size, 0.0);
        _dgemm_full(&inverse_dielectric, 'N', &ri_oo, 'N', &mut ri_oo_tilde, 1.0, 0.0);
        ri_oo_tilde.reshape([num_auxbas * occ_size, occ_size]);
        ri_oo_tilde = ri_oo_tilde.transpose_and_drop();
        ri_oo_tilde.reshape([occ_size * num_auxbas, occ_size]);
        ri_vv.reshape([num_auxbas * vir_size, vir_size]);

        // Capture all needed data by value for Send + Sync safety
        let occ_v = occ_size;
        let vir_v = vir_size;
        let qpc = qp_ctrl_c;
        let energies_s = energies.clone();

        let a_mul: Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync> = Arc::new(
            move |z_vec: &Vec<f64>| -> Vec<f64> {
                let xlet = if qpc.bse_spin == "triplet" { 'T' }
                           else if qpc.bse_spin == "singlet" { 'S' } else { 'R' };
                // Diagonal contribution
                let mut result = matvec::diagonal_contribution_standalone(
                    z_vec, &energies_s, occ_v, vir_v);
                // W contribution (s-only RI)
                let w = matvec::w_contribution_a_block_dgemm_standalone(
                    &ri_vv, z_vec, &ri_oo_tilde, &qpc, occ_v, vir_v);
                result = w.iter().zip(result.iter()).map(|(w_i, z_i)| -w_i + z_i).collect();
                // Coulomb contribution (full RI for accuracy)
                if xlet == 'S' {
                    let v = matvec::coulomb_contribution(&ri_ov, z_vec);
                    result = v.iter().zip(result.iter()).map(|(v_i, z_i)| 2.0 * v_i + z_i).collect();
                } else if xlet == 'R' {
                    let v = matvec::coulomb_contribution(&ri_ov, z_vec);
                    result = v.iter().zip(result.iter()).map(|(v_i, z_i)| v_i + z_i).collect();
                }
                result
            },
        );
        let b_mul: Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync> =
            Arc::new(|z_vec: &Vec<f64>| z_vec.clone());

        (Some(a_mul), Some(b_mul), Some(diag_a))
    } else {
        // non-TDA: for now fall back to diagonal (inner GMRES precond not yet implemented for non-TDA)
        println!("Warning: inner_gmres preconditioner not yet implemented for non-TDA; using diagonal instead.");
        (None, None, None)
    }
}

/// Filter eigenvalues and eigenvectors to only those within [emin, emax].
fn filter_eigenpairs(
    eigenvals: Vec<f64>,
    eigenvecs: Vec<Vec<f64>>,
    emin: f64,
    emax: f64,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut fil_vals = Vec::new();
    let mut fil_vecs = Vec::new();
    for (e, v) in eigenvals.into_iter().zip(eigenvecs.into_iter()) {
        if e >= emin - 1e-12 && e <= emax + 1e-12 {
            fil_vals.push(e);
            fil_vecs.push(v);
        }
    }
    (fil_vals, fil_vecs)
}

/// Print Rayleigh-Ritz eigenpairs using the same format as the BSE excitation output.
fn print_ritz_eigenpairs(
    scf_data: &SCF,
    ritz_eigenvalues: &Vec<f64>,
    ritz_eigenvectors: &Vec<Vec<f64>>,
    label: &str,
    occ_size: usize,
    vir_size: usize,
) {
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let dipole_matrix = super::dipoles::compute_dipole_matrix(scf_data);
    let k = ritz_eigenvalues.len();
    println!("Rayleigh-Ritz {}: {} excitations within the window:", label, k);
    for (n, (e, vec)) in ritz_eigenvalues.iter().zip(ritz_eigenvectors.iter()).enumerate() {
        let v = super::dipoles::normalize(vec, true);
        let dipole_square = super::dipoles::transition_dipole_square(&dipole_matrix, &v, true);
        println!("#{} Excitation energy={}, norm={:.6}", n, e,
                 vec.iter().map(|x| x*x).sum::<f64>().sqrt());
        println!("Transition Dipole Square:{}; Oscillator Strength:{}",
                 dipole_square, dipole_square * e * 2.0 / 3.0);
        super::leading_components(&v, occ_size, vir_size,qp_ctrl.print_nto);
    }
}

/// Rayleigh-Ritz refinement: project Round 1 eigenvectors onto the full A matrix
/// (all angular momenta) and diagonalize the projected subspace to get optimal
/// linear combinations for warm-starting Round 2 FEAST.
/// Returns (Ritz_eigenvalues, Ritz_eigenvectors).
fn rayleigh_ritz_refine(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    quasiparticle_energies: &Vec<f64>,
    occ_size: usize,
    vir_size: usize,
    eigvecs: &Vec<Vec<f64>>,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let k = eigvecs.len();
    if k == 0 {
        return (Vec::new(), Vec::new());
    }

    let ks_energies: Vec<f64> = scf_data.eigenvalues[0].clone();
    let epsilon: Vec<f64> = if qp_ctrl.bse_qp_polarization {
        quasiparticle_energies.clone()
    } else {
        ks_energies.clone()
    };
    let inverse_dielectric = construct_inverse_dielectric(scf_data, &epsilon);

    // Build full A-matrix matvec closure (same pattern as feast_solve_bse_tda).
    //
    // For non-TDA the squared-form operator is (A-B) with metric (A+B)^{-1},
    // mirroring feast_solve_bse_nontda; the Round-1 FEAST eigenvectors are
    // X+Y vectors that are orthonormal in the (A+B)^{-1} metric (NOT in L2).
    // The old code used the A-block operator with an L2 metric, which made the
    // projected Gram matrix indefinite → Cholesky failed → empty result → panic.
    // AO ("ao") or MO ("mo") backend.  This stage runs *after* the round-1
    // caller has cleared `rimatr_bse`, so the AO context folds the regular
    // `rimatr` -- which is what the refinement must use.
    let mv = super::BseMatvec::new(scf_data, &inverse_dielectric, !qp_ctrl.bse_tda);

    // TDA operator = A; non-TDA operator = (A-B).
    let a_matvec = |z: &Vec<f64>| -> Vec<f64> { mv.a(scf_data, qp_ctrl, z) };
    let amb_matvec = |z: &Vec<f64>| -> Vec<f64> { mv.amb(scf_data, qp_ctrl, z) };
    let op_matvec: &dyn Fn(&Vec<f64>) -> Vec<f64> = if qp_ctrl.bse_tda { &a_matvec } else { &amb_matvec };

    // Normalize input eigenvectors to unit L2 norm
    let mut eigvecs_norm: Vec<Vec<f64>> = Vec::with_capacity(k);
    for v in eigvecs.iter() {
        let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            eigvecs_norm.push(v.iter().map(|&x| x / norm).collect());
        } else {
            eigvecs_norm.push(v.clone());
        }
    }

    // Compute op * v_i for each normalized eigenvector
    let av: Vec<Vec<f64>> = eigvecs_norm.iter().map(|v| op_matvec(v)).collect();

    // Check for NaN/Inf in A*v results
    let mut av_bad = false;
    let mut av_bad_col = 0;
    let mut av_bad_kind = "";
    for (j, v) in av.iter().enumerate() {
        for &x in v.iter() {
            if x.is_nan() { av_bad = true; av_bad_kind = "NaN"; av_bad_col = j; break; }
            if x.is_infinite() { av_bad = true; av_bad_kind = "Inf"; av_bad_col = j; break; }
        }
        if av_bad { break; }
    }
    if av_bad {
        eprintln!("Warning: Rayleigh-Ritz A*vector contains {} in column {} — \
                   using original eigenvectors.", av_bad_kind, av_bad_col);
        return (Vec::new(), eigvecs.clone());
    }

    // Pack eigenvectors and A*v into MatrixFull for BLAS-based projection.
    // Use _dgemm_scaled (same path that FEAST subspace diagonalization uses
    // successfully) instead of manual dot products.
    let n = occ_size * vir_size;
    let mut v_mat = MatrixFull::new([n, k], 0.0);
    let mut av_mat = MatrixFull::new([n, k], 0.0);
    for j in 0..k {
        for i in 0..n {
            v_mat[[i, j]] = eigvecs_norm[j][i];
            av_mat[[i, j]] = av[j][i];
        }
    }

    // H_proj = V^T · (A·V)   via BLAS
    let h_proj = _dgemm_scaled(&v_mat, 'T', &av_mat, 'N', 1.0);

    // Diagnostic
    let mut h_min = f64::INFINITY; let mut h_max = f64::NEG_INFINITY;
    for i in 0..k {
        for j in 0..k {
            let hv = h_proj[[i, j]];
            if hv.is_finite() { h_min = h_min.min(hv); h_max = h_max.max(hv); }
        }
    }
    println!("Rayleigh-Ritz: H_proj range=[{:.3e}, {:.3e}] (k={})", h_min, h_max, k);

    let (ritz_eigenvalues, psi_opt) = if qp_ctrl.bse_tda {
        // TDA: eigenvectors from FEAST are orthonormal (B=I) → standard EVP.
        // Use _dgeev (QR algorithm) instead of _dsyevd (divide-and-conquer)
        // because dsyevd can fail to converge for certain eigenvalue distributions.
        let (_, wr, wi, _, vr, info) = _dgeev(&h_proj, 'N', 'V');
        if info != 0 {
            eprintln!("Warning: Rayleigh-Ritz dgeev failed with info={}, using original eigenvectors", info);
            return (Vec::new(), eigvecs.clone());
        }
        // Select real eigenvalues and sort ascending.
        // h_proj is symmetric so all eigenvalues should be real; the filter
        // is a safety net against numerical noise in dgeev.
        let mut eigen_pairs: Vec<(usize, f64)> = (0..k)
            .filter(|&j| wi[j].abs() < 1e-10)
            .map(|j| (j, wr[j]))
            .collect();
        if eigen_pairs.is_empty() {
            eprintln!("Warning: Rayleigh-Ritz dgeev returned no real eigenvalues, using original eigenvectors");
            return (Vec::new(), eigvecs.clone());
        }
        eigen_pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let evals: Vec<f64> = eigen_pairs.iter().map(|&(_, v)| v).collect();
        let m_sel = eigen_pairs.len();
        let mut psi_mat = MatrixFull::new([k, m_sel], 0.0);
        for (col, &(orig_j, _)) in eigen_pairs.iter().enumerate() {
            for row in 0..k {
                psi_mat[[row, col]] = vr[[row, orig_j]];
            }
        }
        (evals, Some(psi_mat))
    } else {
        // Non-TDA: Round-1 FEAST solved (A-B)u = ω²(A+B)^{-1}u, equivalently
        //   (A+B)(A-B) u = ω² u ,            u = X+Y .
        // The converged X+Y vectors are orthonormal in the (A+B)^{-1} metric,
        // NOT in L2, so any metric-based Cholesky/GEP projection is fragile
        // (indefinite Gram matrix / CG asymmetry).  Instead we project the
        // (generally non-symmetric) operator T = (A+B)(A-B) onto an L2-
        // orthonormal basis Q and solve the resulting standard (non-symmetric)
        // eigenproblem with dgeev — no metric, no Cholesky, no CG.

        // (1) Build an L2-orthonormal basis Q from the input vectors via
        //     modified Gram-Schmidt.  (eigvecs_norm are unit L2-norm but not
        //     mutually orthogonal.)
        let mut q_orth: Vec<Vec<f64>> = Vec::with_capacity(k);
        for v in eigvecs_norm.iter() {
            let mut w = v.clone();
            for q in q_orth.iter() {
                let proj: f64 = w.iter().zip(q.iter()).map(|(wi, qi)| wi * qi).sum();
                for i in 0..n { w[i] -= proj * q[i]; }
            }
            let nrm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if nrm > 1e-12 {
                for x in w.iter_mut() { *x /= nrm; }
                q_orth.push(w);
            }
        }
        let k_eff = q_orth.len();
        if k_eff == 0 {
            eprintln!("Warning: Rayleigh-Ritz non-TDA: subspace collapsed to rank 0, using original eigenvectors");
            return (Vec::new(), eigvecs.clone());
        }
        // Pack Q (n × k_eff)
        let mut q_mat = MatrixFull::new([n, k_eff], 0.0);
        for j in 0..k_eff { for i in 0..n { q_mat[[i, j]] = q_orth[j][i]; } }

        // (2) T·Q where T = (A+B)(A-B): apply (A-B) then (A+B) to each column.
        //     op_matvec is (A-B); apb_matvec is (A+B).
        let apb_matvec = |p: &Vec<f64>| -> Vec<f64> { mv.apb(scf_data, qp_ctrl, p) };
        let mut tq = MatrixFull::new([n, k_eff], 0.0);
        for j in 0..k_eff {
            let qj: Vec<f64> = (0..n).map(|i| q_mat[[i, j]]).collect();
            let amb_qj = op_matvec(&qj);      // (A-B) q_j
            let t_qj = apb_matvec(&amb_qj);   // (A+B)(A-B) q_j
            for i in 0..n { tq[[i, j]] = t_qj[i]; }
        }

        // (3) T_proj = Q^T · (T·Q)  (k_eff × k_eff, generally non-symmetric)
        let t_proj = _dgemm_scaled(&q_mat, 'T', &tq, 'N', 1.0);
        let mut t_min = f64::INFINITY; let mut t_max = f64::NEG_INFINITY;
        for i in 0..k_eff { for j in 0..k_eff {
            let tv = t_proj[[i, j]];
            if tv.is_finite() { t_min = t_min.min(tv); t_max = t_max.max(tv); }
        }}
        println!("Rayleigh-Ritz: T_proj range=[{:.3e}, {:.3e}] (k_eff={})", t_min, t_max, k_eff);

        // (4) Standard (non-symmetric) EVP on T_proj → eigenvalues are ω².
        let (_, wr, wi, _, vr, info) = _dgeev(&t_proj, 'N', 'V');
        if info != 0 {
            eprintln!("Warning: Rayleigh-Ritz non-TDA dgeev failed with info={}, using original eigenvectors", info);
            return (Vec::new(), eigvecs.clone());
        }
        // Keep real, positive ω² eigenvalues, sort ascending by ω.
        let mut eigen_pairs: Vec<(usize, f64)> = (0..k_eff)
            .filter(|&j| wi[j].abs() < 1e-8 && wr[j] > 0.0)
            .map(|j| (j, wr[j]))
            .collect();
        if eigen_pairs.is_empty() {
            eprintln!("Warning: Rayleigh-Ritz non-TDA: no real positive ω² eigenvalues, using original eigenvectors");
            return (Vec::new(), eigvecs.clone());
        }
        eigen_pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let m_sel = eigen_pairs.len();
        let omega: Vec<f64> = eigen_pairs.iter().map(|&(_, w2)| w2.sqrt()).collect();
        // Eigenvector coefficients in the Q basis: vr[:,orig_j]
        let mut psi_mat = MatrixFull::new([k_eff, m_sel], 0.0);
        for (col, &(orig_j, _)) in eigen_pairs.iter().enumerate() {
            for row in 0..k_eff { psi_mat[[row, col]] = vr[[row, orig_j]]; }
        }
        // Stash Q so the Ritz-transform below uses the orthonormal basis:
        // ritz_vec[i] = Σ_j psi[j,i] * q_orth[j].  We return evals=ω here; the
        // caller filters by [emin, emax] (the ω window).  Replace eigvecs_norm
        // with Q (length k_eff) — the transform loop below uses eigvecs_norm.len().
        eigvecs_norm = q_orth.clone();
        (omega, Some(psi_mat))
    };
    let psi = match psi_opt {
        Some(p) => p,
        None => {
            eprintln!("Warning: Rayleigh-Ritz generalized EVP failed, using original eigenvectors");
            return (Vec::new(), eigvecs.clone());
        }
    };

    // Transform: ritz_vec[i] = Σ_j psi[j,i] * eigvecs_norm[j], then L2-normalize.
    // _dgeev eigenvectors are not unit-norm, so explicit normalization is needed.
    // Use eigvecs_norm.len() as the row dimension: it equals k for TDA, and
    // k_eff (≤ k) for non-TDA where eigvecs_norm was replaced by the orthonormal
    // basis q_orth above.
    let k_rows = eigvecs_norm.len();
    let mut ritz_vecs: Vec<Vec<f64>> = Vec::with_capacity(ritz_eigenvalues.len());
    for i in 0..ritz_eigenvalues.len() {
        let mut new_v = vec![0.0; n];
        for j in 0..k_rows {
            let coeff = psi[[j, i]];
            for idx in 0..n {
                new_v[idx] += coeff * eigvecs_norm[j][idx];
            }
        }
        let norm: f64 = new_v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in new_v.iter_mut() { *x /= norm; }
        }
        ritz_vecs.push(new_v);
    }

    (ritz_eigenvalues, ritz_vecs)
}

/// Solve generalized eigenvalue problem H·C = S·C·E via Cholesky of S.
/// Only called for non-TDA where basis vectors are not orthonormal.
/// Returns (eigenvalues, Some(eigenvectors)) on success, or (empty, None) on failure.
fn solve_generalized_eigenproblem(
    h: &MatrixFull<f64>,
    s: &mut MatrixFull<f64>,
    k: usize,
) -> (Vec<f64>, Option<MatrixFull<f64>>) {
    // Cholesky S = L·L^T
    let cholesky_ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        _dpotrf(s, 'L');
    }));

    if cholesky_ok.is_err() {
        // S not SPD — fall back to dsyevd on H
        let (psi_opt, evals, info) = _dsyevd(h, 'V');
        if info != 0 {
            eprintln!("Warning: Rayleigh-Ritz dsyevd fallback failed with info={}", info);
            return (Vec::new(), None);
        }
        return (evals, Some(psi_opt.unwrap()));
    }

    // Build lower-triangular L from factorized S
    let mut l_mat = MatrixFull::new([k, k], 0.0);
    for col in 0..k {
        for row in col..k {
            l_mat[[row, col]] = s[[row, col]];
        }
    }
    let l_inv = match _dinverse(&l_mat) {
        Some(m) => m,
        None => {
            let (psi_opt, evals, info) = _dsyevd(h, 'V');
            if info != 0 {
                eprintln!("Warning: Rayleigh-Ritz dsyevd fallback failed with info={}", info);
                return (Vec::new(), None);
            }
            return (evals, Some(psi_opt.unwrap()));
        }
    };
    let l_inv_t = l_inv.transpose();

    // H_trans = L⁻¹ · H · L⁻ᵀ, then standard EVP
    let tmp = _dgemm_scaled(&l_inv, 'N', h, 'N', 1.0);
    let h_trans = _dgemm_scaled(&tmp, 'N', &l_inv_t, 'N', 1.0);

    let (psi_trans_opt, evals, info) = _dsyevd(&h_trans, 'V');
    if info != 0 {
        eprintln!("Warning: Rayleigh-Ritz dsyevd on transformed H failed with info={}", info);
        // Last resort: try dsyevd on original H
        let (psi_opt, evals2, info2) = _dsyevd(h, 'V');
        if info2 != 0 {
            eprintln!("Warning: Rayleigh-Ritz dsyevd last-resort failed with info={}", info2);
            return (Vec::new(), None);
        }
        return (evals2, Some(psi_opt.unwrap()));
    }
    let psi_trans = psi_trans_opt.unwrap();

    // Transform back: C = L⁻ᵀ · Ψ_trans
    let psi = _dgemm_scaled(&l_inv_t, 'N', &psi_trans, 'N', 1.0);
    (evals, Some(psi))
}

/// FEAST solver for singlet BSE excitations (handles both TDA and non-TDA).
/// When `bse_feast_renormalized_doubles` is true, uses s-only FEAST plus
/// Rayleigh-Ritz refinement with full integrals — no second FEAST round.
pub fn feast_solve_bse_singlet(scf_data:&mut SCF)->Vec<(f64,Vec<f64>)>{
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    // For TDA, FEAST solves A*x = omega*x (eigenvalue = omega directly).
    // For non-TDA, FEAST solves for omega^2, so the search range must be squared.
    let (eigenrange_min, eigenrange_max) = if qp_ctrl.bse_tda {
        (qp_ctrl.bse_eigenrange_min, qp_ctrl.bse_eigenrange_max)
    } else {
        (qp_ctrl.bse_eigenrange_min * qp_ctrl.bse_eigenrange_min,
         qp_ctrl.bse_eigenrange_max * qp_ctrl.bse_eigenrange_max)
    };
    let m_expected=qp_ctrl.bse_m_expected;
    let max_feast_iter=qp_ctrl.bse_max_feast_iter;
    let tol_feast=qp_ctrl.bse_tol_feast;
    let quasiparticle_energies=scf_data.gwqp.0.clone();
    let mut qp_ctrl_singlet=qp_ctrl.clone();
    qp_ctrl_singlet.bse_spin=String::from("singlet");

    println!("[DEBUG] feast_solve_bse_singlet: bse_feast_renormalized_doubles={}", qp_ctrl.bse_feast_renormalized_doubles);
    if qp_ctrl.bse_feast_renormalized_doubles {
        let ew = qp_ctrl.bse_renormalized_doubles_extra_width;
        let (r1_min, r1_max) = if qp_ctrl.bse_tda {
            (qp_ctrl.bse_eigenrange_min - ew, qp_ctrl.bse_eigenrange_max + ew)
        } else {
            let orig_min = (qp_ctrl.bse_eigenrange_min - ew).max(0.0);
            let orig_max = qp_ctrl.bse_eigenrange_max + ew;
            (orig_min * orig_min, orig_max * orig_max)
        };
        println!("--- Round 1: BSE-specific RI integrals (s-only) for singlet, range=[{:.6},{:.6}] ---",
                 r1_min, r1_max);
        let round1 = feast_solve_bse_spin(scf_data,&qp_ctrl_singlet,&quasiparticle_energies,
                             r1_min, r1_max, m_expected, max_feast_iter, tol_feast, None,
                             None, None, "diagonal", None, 0.0001, 0, 0);
        let n_found = round1.len();
        println!("Round 1 found {} eigenpairs within the window", n_found);
        let eigvecs: Vec<Vec<f64>> = round1.into_iter().map(|(_, v)| v).collect();

        let (_, _, occ_size, vir_size, _, _) = get_occupation_parameters(scf_data, 'N');

        // Clear BSE-specific integrals → fallback to regular RI for Rayleigh-Ritz
        scf_data.ri3fn_bse = None;
        scf_data.rimatr_bse = None;

        // Rayleigh-Ritz refinement with full integrals
        let (ritz_vals, ritz_vecs) = rayleigh_ritz_refine(scf_data, &qp_ctrl_singlet, &quasiparticle_energies,
                                           occ_size, vir_size, &eigvecs);
        let emin = qp_ctrl.bse_eigenrange_min;
        let emax = qp_ctrl.bse_eigenrange_max;
        let (fil_vals, fil_vecs) = filter_eigenpairs(ritz_vals, ritz_vecs, emin, emax);
        print_ritz_eigenpairs(scf_data, &fil_vals, &fil_vecs, "singlet", occ_size, vir_size);
        let raw: Vec<(f64, Vec<f64>)> = fil_vals.into_iter().zip(fil_vecs.into_iter()).collect();
        // Rebuild [X;Y] for non-TDA so downstream export/print code is consistent.
        if qp_ctrl.bse_tda { raw } else {
            reconstruct_nontda_pairs_to_xy(scf_data, &qp_ctrl_singlet, raw, occ_size, vir_size)
        }
    } else {
        let raw = feast_solve_bse_spin(scf_data,&qp_ctrl_singlet,&quasiparticle_energies,
                             eigenrange_min,eigenrange_max,m_expected,max_feast_iter,tol_feast,None,
                             None, None, "diagonal", None, 0.0001, 0, 0);
        let (_, _, occ_size, vir_size, _, _) = get_occupation_parameters(scf_data, 'N');
        if qp_ctrl.bse_tda { raw } else {
            reconstruct_nontda_pairs_to_xy(scf_data, &qp_ctrl_singlet, raw, occ_size, vir_size)
        }
    }
}

/// FEAST solver for triplet BSE excitations (handles both TDA and non-TDA).
/// When `bse_feast_renormalized_doubles` is true, uses s-only FEAST plus
/// Rayleigh-Ritz refinement with full integrals — no second FEAST round.
pub fn feast_solve_bse_triplet(scf_data:&mut SCF)->Vec<(f64,Vec<f64>)>{
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    // For TDA, FEAST solves A*x = omega*x (eigenvalue = omega directly).
    // For non-TDA, FEAST solves for omega^2, so the search range must be squared.
    let (eigenrange_min, eigenrange_max) = if qp_ctrl.bse_tda {
        (qp_ctrl.bse_eigenrange_min, qp_ctrl.bse_eigenrange_max)
    } else {
        (qp_ctrl.bse_eigenrange_min * qp_ctrl.bse_eigenrange_min,
         qp_ctrl.bse_eigenrange_max * qp_ctrl.bse_eigenrange_max)
    };
    let m_expected=qp_ctrl.bse_m_expected;
    let max_feast_iter=qp_ctrl.bse_max_feast_iter;
    let tol_feast=qp_ctrl.bse_tol_feast;
    let quasiparticle_energies=scf_data.gwqp.0.clone();
    let mut qp_ctrl_triplet=qp_ctrl.clone();
    qp_ctrl_triplet.bse_spin=String::from("triplet");

    if qp_ctrl.bse_feast_renormalized_doubles {
        let ew = qp_ctrl.bse_renormalized_doubles_extra_width;
        let (r1_min, r1_max) = if qp_ctrl.bse_tda {
            (qp_ctrl.bse_eigenrange_min - ew, qp_ctrl.bse_eigenrange_max + ew)
        } else {
            let orig_min = (qp_ctrl.bse_eigenrange_min - ew).max(0.0);
            let orig_max = qp_ctrl.bse_eigenrange_max + ew;
            (orig_min * orig_min, orig_max * orig_max)
        };
        println!("--- Round 1: BSE-specific RI integrals (s-only) for triplet, range=[{:.6},{:.6}] ---",
                 r1_min, r1_max);
        let round1 = feast_solve_bse_spin(scf_data,&qp_ctrl_triplet,&quasiparticle_energies,
                             r1_min, r1_max, m_expected, max_feast_iter, tol_feast, None,
                             None, None, "diagonal", None, 0.0001, 0, 0);
        let n_found = round1.len();
        println!("Round 1 found {} eigenpairs within the window", n_found);
        let eigvecs: Vec<Vec<f64>> = round1.into_iter().map(|(_, v)| v).collect();

        let (_, _, occ_size, vir_size, _, _) = get_occupation_parameters(scf_data, 'N');

        // Clear BSE-specific integrals → fallback to regular RI for Rayleigh-Ritz
        scf_data.ri3fn_bse = None;
        scf_data.rimatr_bse = None;

        // Rayleigh-Ritz refinement with full integrals
        let (ritz_vals, ritz_vecs) = rayleigh_ritz_refine(scf_data, &qp_ctrl_triplet, &quasiparticle_energies,
                                           occ_size, vir_size, &eigvecs);
        let (ritz_vals, ritz_vecs) = rayleigh_ritz_refine(scf_data, &qp_ctrl_triplet, &quasiparticle_energies,
                                           occ_size, vir_size, &eigvecs);
        let emin = qp_ctrl.bse_eigenrange_min;
        let emax = qp_ctrl.bse_eigenrange_max;
        let (fil_vals, fil_vecs) = filter_eigenpairs(ritz_vals, ritz_vecs, emin, emax);
        print_ritz_eigenpairs(scf_data, &fil_vals, &fil_vecs, "triplet", occ_size, vir_size);
        let raw: Vec<(f64, Vec<f64>)> = fil_vals.into_iter().zip(fil_vecs.into_iter()).collect();
        // Rebuild [X;Y] for non-TDA so downstream export/print code is consistent.
        if qp_ctrl.bse_tda { raw } else {
            reconstruct_nontda_pairs_to_xy(scf_data, &qp_ctrl_triplet, raw, occ_size, vir_size)
        }
    } else {
        let raw = feast_solve_bse_spin(scf_data,&qp_ctrl_triplet,&quasiparticle_energies,
                             eigenrange_min,eigenrange_max,m_expected,max_feast_iter,tol_feast,None,
                             None, None, "diagonal", None, 0.0001, 0, 0);
        let (_, _, occ_size, vir_size, _, _) = get_occupation_parameters(scf_data, 'N');
        if qp_ctrl.bse_tda { raw } else {
            reconstruct_nontda_pairs_to_xy(scf_data, &qp_ctrl_triplet, raw, occ_size, vir_size)
        }
    }
}

/// Internal helper: solve for a specific spin (set in qp_ctrl.bse_spin).
fn feast_solve_bse_spin(
    scf_data:&SCF,
    qp_ctrl:&QuasiParticle,
    quasiparticle_energies:&Vec<f64>,
    eigenrange_min:f64,
    eigenrange_max:f64,
    m_expected:usize,
    max_feast_iter:usize,
    tol_feast:f64,
    custom_init_vectors: Option<&Vec<Vec<f64>>>,
    precond_a_mul: Option<Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync>>,
    precond_b_mul: Option<Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync>>,
    precond_type: &str,
    precond_diag: Option<&Vec<f64>>,
    inner_gmres_tol: f64,
    inner_gmres_restart: usize,
    inner_gmres_max_iter: usize,
)->Vec<(f64,Vec<f64>)>{
    let (_,_,occ_size,vir_size,_,_)=get_occupation_parameters(scf_data,'N');
    let ks_energies:Vec<f64>=scf_data.eigenvalues[0].clone();
    let mut epsilon=ks_energies.clone();
    if qp_ctrl.bse_qp_polarization==true{
        epsilon=quasiparticle_energies.clone();
    }
    let inverse_dielectric=construct_inverse_dielectric(scf_data,&epsilon);

    if qp_ctrl.bse_tda==true{
        feast_solve_bse_tda(scf_data,qp_ctrl,&inverse_dielectric,occ_size,vir_size,
                            eigenrange_min,eigenrange_max,m_expected,max_feast_iter,tol_feast,
                            custom_init_vectors,
                            precond_a_mul, precond_b_mul, precond_type, precond_diag, inner_gmres_tol, inner_gmres_restart, inner_gmres_max_iter)
    }else{
        feast_solve_bse_nontda(scf_data,qp_ctrl,&inverse_dielectric,occ_size,vir_size,
                               eigenrange_min,eigenrange_max,m_expected,max_feast_iter,tol_feast,
                               custom_init_vectors,
                               precond_a_mul, precond_b_mul, precond_type, precond_diag, inner_gmres_tol, inner_gmres_restart, inner_gmres_max_iter)
    }
}

/// TDA branch for a single spin.
fn feast_solve_bse_tda(
    scf_data:&SCF,
    qp_ctrl:&QuasiParticle,
    inverse_dielectric:&MatrixFull<f64>,
    occ_size:usize,
    vir_size:usize,
    eigenrange_min:f64,
    eigenrange_max:f64,
    m_expected:usize,
    max_feast_iter:usize,
    tol_feast:f64,
    custom_init_vectors: Option<&Vec<Vec<f64>>>,
    precond_a_mul: Option<Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync>>,
    precond_b_mul: Option<Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync>>,
    precond_type: &str,
    precond_diag: Option<&Vec<f64>>,
    inner_gmres_tol: f64,
    inner_gmres_restart: usize,
    inner_gmres_max_iter: usize,
)->Vec<(f64,Vec<f64>)>{
    // AO ("ao") or MO ("mo") backend; in AO mode no MO-basis RI tensor is built.
    let mv = super::BseMatvec::new(scf_data, inverse_dielectric, false);

    let feast_a_matvec=|z:&Vec<f64>|{
        mv.a(scf_data,qp_ctrl,z)
    };
    let feast_b_matvec=|z:&Vec<f64>|{
        z.clone()
    };

    // ── GMRES diagonal preconditioner for TDA ──
    // The transformed GMRES matrix is (z·I − A), diagonal ≈ z − D_j
    // where D_j = ε_a − ε_i are the quasi-particle energy gaps.
    let diag_a: Vec<f64> = {
        let energies: Vec<f64> = if qp_ctrl.bse_qp_polarization {
            scf_data.gwqp.0.clone()
        } else {
            scf_data.eigenvalues[0].clone()
        };
        construct_energy_diag_for_a(&energies, occ_size, vir_size)
    };

    let gmres_restart = qp_ctrl.bse_feast_gmres_restart;
    let gmres_max_iter = qp_ctrl.bse_feast_gmres_max_iter;
    let gmres_tol = qp_ctrl.bse_feast_cg_tol;
    //let n_quad = qp_ctrl.bse_feast_n_quad;
    feast(occ_size*vir_size,&feast_a_matvec,&feast_b_matvec,None,None,
          eigenrange_min,eigenrange_max,m_expected,max_feast_iter,tol_feast,
          gmres_restart,gmres_max_iter,gmres_tol,
          Some(&diag_a),
          &qp_ctrl.bse_feast_init_guess_type, Some(&diag_a),
          qp_ctrl.bse_feast_gaussian_width_factor,
          qp_ctrl.bse_feast_contour_rayon,
          custom_init_vectors,
          precond_a_mul, precond_b_mul, precond_type, precond_diag, inner_gmres_tol, inner_gmres_restart, inner_gmres_max_iter)
}

/// Non-TDA branch for a single spin.
fn feast_solve_bse_nontda(
    scf_data:&SCF,
    qp_ctrl:&QuasiParticle,
    inverse_dielectric:&MatrixFull<f64>,
    occ_size:usize,
    vir_size:usize,
    eigenrange_min:f64,
    eigenrange_max:f64,
    m_expected:usize,
    max_feast_iter:usize,
    tol_feast:f64,
    custom_init_vectors: Option<&Vec<Vec<f64>>>,
    precond_a_mul: Option<Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync>>,
    precond_b_mul: Option<Arc<dyn Fn(&Vec<f64>) -> Vec<f64> + Send + Sync>>,
    precond_type: &str,
    precond_diag: Option<&Vec<f64>>,
    inner_gmres_tol: f64,
    inner_gmres_restart: usize,
    inner_gmres_max_iter: usize,
)->Vec<(f64,Vec<f64>)>{
    // AO ("ao") or MO ("mo") backend; in AO mode no MO-basis RI tensor is built.
    let mv = super::BseMatvec::new(scf_data, inverse_dielectric, true);

    let feast_a_matvec=|z:&Vec<f64>|->Vec<f64>{
        mv.amb(scf_data,qp_ctrl,z)
    };
    // ── Diagonal preconditioner data for non-TDA ──
    // D_j = ε_a − ε_i are the diagonal QP energy gaps.  They are used:
    //   1. as the diagonal preconditioner for CG solves of (A+B);
    //   2. squared, as the diagonal preconditioner for the contour GMRES
    //      system z·I − (A+B)(A−B).
    let energies: Vec<f64> = if qp_ctrl.bse_qp_polarization {
        scf_data.gwqp.0.clone()
    } else {
        scf_data.eigenvalues[0].clone()
    };
    let diag = construct_energy_diag_for_a(&energies, occ_size, vir_size);
    let diag_sq: Vec<f64> = diag.iter().map(|&d| d * d).collect();

    let feast_b_matvec=|z:&Vec<f64>|->Vec<f64>{
        let apb_matvec=|p:&Vec<f64>|->Vec<f64>{ mv.apb(scf_data,qp_ctrl,p) };
        cg(&apb_matvec,z,qp_ctrl.bse_feast_cg_max_iter,qp_ctrl.bse_feast_cg_tol,Some(&diag))
    };

    let gmres_restart = qp_ctrl.bse_feast_gmres_restart;
    let gmres_max_iter = qp_ctrl.bse_feast_gmres_max_iter;
    let gmres_tol = qp_ctrl.bse_feast_cg_tol;
    let gmres_a_mul=|z:&Vec<f64>|->Vec<f64>{
        let apb_q=feast_a_matvec(z);
        mv.apb(scf_data,qp_ctrl,&apb_q)
    };
    let gmres_b_mul=|z:&Vec<f64>|{
        z.clone()
    };

    let eigenpairs_xpy=feast(occ_size*vir_size,&feast_a_matvec,&feast_b_matvec,Some(&gmres_a_mul),Some(&gmres_b_mul),
                             eigenrange_min,eigenrange_max,m_expected,max_feast_iter,tol_feast,
                             gmres_restart,gmres_max_iter,gmres_tol,
                             Some(&diag_sq),
                             &qp_ctrl.bse_feast_init_guess_type, Some(&diag),
                             qp_ctrl.bse_feast_gaussian_width_factor,
                             qp_ctrl.bse_feast_contour_rayon,
                             custom_init_vectors,
                             precond_a_mul, precond_b_mul, precond_type, precond_diag, inner_gmres_tol, inner_gmres_restart, inner_gmres_max_iter);
    eigenpairs_xpy.iter().map(|(omega2,xpy)|{
        let xmy=feast_a_matvec(xpy);
        (omega2.sqrt(),xmy.iter().zip(xpy.iter()).map(|(xmy_k,xpy_k)|(xmy_k/omega2.sqrt())+xpy_k).collect::<Vec<_>>())
    }).collect()
}

/// Solve both singlet and triplet BSE excitations using FEAST.
/// When `bse_feast_renormalized_doubles` is true, runs two rounds:
/// Round 1 with BSE-specific RI integrals (both spins), Round 2 with
/// regular integrals warm-started by Round 1 eigenvectors.
pub fn feast_solve_bse(scf_data:&mut SCF)->(Vec<(f64,Vec<f64>)>,Vec<(f64,Vec<f64>)>){
    let start=Instant::now();
    let qp_ctrl=scf_data.mol.ctrl.quasiparticle_methods.clone().unwrap();
    let eigenrange_min_orig = qp_ctrl.bse_eigenrange_min;
    let eigenrange_max_orig = qp_ctrl.bse_eigenrange_max;
    let eigenrange_min = eigenrange_min_orig * eigenrange_min_orig;
    let eigenrange_max = eigenrange_max_orig * eigenrange_max_orig;
    let m_expected=qp_ctrl.bse_m_expected;
    let max_feast_iter=qp_ctrl.bse_max_feast_iter;
    let tol_feast=qp_ctrl.bse_tol_feast;
    let quasiparticle_energies=scf_data.gwqp.0.clone();

    let (_,_,occ_size,vir_size,_,_)=get_occupation_parameters(scf_data,'N');
    let ks_energies:Vec<f64>=scf_data.eigenvalues[0].clone();
    let mut epsilon=ks_energies.clone();
    if qp_ctrl.bse_qp_polarization==true{
        epsilon=quasiparticle_energies.clone();
    }
    let inverse_dielectric=construct_inverse_dielectric(scf_data,&epsilon);

    if !qp_ctrl.bse_feast_renormalized_doubles {
        // ---- Original single-round flow ----
        let mut eigenpairs_singlet:Vec<(f64,Vec<f64>)>=Vec::new();
        let mut eigenpairs_triplet:Vec<(f64,Vec<f64>)>=Vec::new();

        if qp_ctrl.bse_tda==true{
            let mut qp_ctrl_s=qp_ctrl.clone();
            qp_ctrl_s.bse_spin=String::from("singlet");
            eigenpairs_singlet=feast_solve_bse_tda(
                scf_data,&qp_ctrl_s,&inverse_dielectric,occ_size,vir_size,
                eigenrange_min_orig,eigenrange_max_orig,m_expected,max_feast_iter,tol_feast,None,
                None, None, "diagonal", None, 0.0001, 0, 0);
            let one_feast_time=start.elapsed();
            println!("Singlets (FEAST) calculation took {:?}",one_feast_time);

            let mut qp_ctrl_t=qp_ctrl.clone();
            qp_ctrl_t.bse_spin=String::from("triplet");
            eigenpairs_triplet=feast_solve_bse_tda(
                scf_data,&qp_ctrl_t,&inverse_dielectric,occ_size,vir_size,
                eigenrange_min_orig,eigenrange_max_orig,m_expected,max_feast_iter,tol_feast,None,
                None, None, "diagonal", None, 0.0001, 0, 0);
            println!("Triplets (FEAST) calculation took {:?}",start.elapsed()-one_feast_time);
        }else{
            let mut qp_ctrl_s=qp_ctrl.clone();
            qp_ctrl_s.bse_spin=String::from("singlet");
            let singlet_raw=feast_solve_bse_nontda(
                scf_data,&qp_ctrl_s,&inverse_dielectric,occ_size,vir_size,
                eigenrange_min,eigenrange_max,m_expected,max_feast_iter,tol_feast,None,
                None, None, "diagonal", None, 0.0001, 0, 0);
            let one_feast_time=start.elapsed();
            println!("Singlets (FEAST) calculation took {:?}",one_feast_time);
            // Rebuild [X;Y] (length 2n) from FEAST's X+Y-space vectors so that
            // downstream export/dipole code (written for Davidson's [X;Y]) works.
            eigenpairs_singlet=reconstruct_nontda_pairs_to_xy(
                scf_data,&qp_ctrl_s,singlet_raw,occ_size,vir_size);

            let mut qp_ctrl_t=qp_ctrl.clone();
            qp_ctrl_t.bse_spin=String::from("triplet");
            let triplet_raw=feast_solve_bse_nontda(
                scf_data,&qp_ctrl_t,&inverse_dielectric,occ_size,vir_size,
                eigenrange_min,eigenrange_max,m_expected,max_feast_iter,tol_feast,None,
                None, None, "diagonal", None, 0.0001, 0, 0);
            println!("Triplets (FEAST) calculation took {:?}",start.elapsed()-one_feast_time);
            eigenpairs_triplet=reconstruct_nontda_pairs_to_xy(
                scf_data,&qp_ctrl_t,triplet_raw,occ_size,vir_size);
        }

        return (eigenpairs_singlet,eigenpairs_triplet);
    }

    // ---- Renormalized doubles flow ----
    // Round 1: s-only FEAST → Rayleigh-Ritz with full integrals → output result
    let ew = qp_ctrl.bse_renormalized_doubles_extra_width;
    let (r1_min_tda, r1_max_tda) = (eigenrange_min_orig - ew, eigenrange_max_orig + ew);
    let r1_min_nontda = (eigenrange_min_orig - ew).max(0.0);
    let r1_min_nontda_sq = r1_min_nontda * r1_min_nontda;
    let r1_max_nontda_sq = (eigenrange_max_orig + ew) * (eigenrange_max_orig + ew);
    println!("--- Round 1: BSE-specific RI integrals (s-only) for both spins, original range widened by {} ---", ew);

    let round1_singlet: Vec<(f64,Vec<f64>)>;
    let round1_triplet: Vec<(f64,Vec<f64>)>;

    if qp_ctrl.bse_tda==true{
        let mut qp_ctrl_s=qp_ctrl.clone();
        qp_ctrl_s.bse_spin=String::from("singlet");
        round1_singlet=feast_solve_bse_tda(
            scf_data,&qp_ctrl_s,&inverse_dielectric,occ_size,vir_size,
            r1_min_tda,r1_max_tda,m_expected,max_feast_iter,tol_feast,None,
            None, None, "diagonal", None, 0.0001, 0, 0);
        let one_feast_time=start.elapsed();
        println!("Round 1 Singlets took {:?}",one_feast_time);

        let mut qp_ctrl_t=qp_ctrl.clone();
        qp_ctrl_t.bse_spin=String::from("triplet");
        round1_triplet=feast_solve_bse_tda(
            scf_data,&qp_ctrl_t,&inverse_dielectric,occ_size,vir_size,
            r1_min_tda,r1_max_tda,m_expected,max_feast_iter,tol_feast,None,
            None, None, "diagonal", None, 0.0001, 0, 0);
        println!("Round 1 Triplets took {:?}",start.elapsed()-one_feast_time);
    }else{
        let mut qp_ctrl_s=qp_ctrl.clone();
        qp_ctrl_s.bse_spin=String::from("singlet");
        round1_singlet=feast_solve_bse_nontda(
            scf_data,&qp_ctrl_s,&inverse_dielectric,occ_size,vir_size,
            r1_min_nontda_sq,r1_max_nontda_sq,m_expected,max_feast_iter,tol_feast,None,
            None, None, "diagonal", None, 0.0001, 0, 0);
        let one_feast_time=start.elapsed();
        println!("Round 1 Singlets took {:?}",one_feast_time);

        let mut qp_ctrl_t=qp_ctrl.clone();
        qp_ctrl_t.bse_spin=String::from("triplet");
        round1_triplet=feast_solve_bse_nontda(
            scf_data,&qp_ctrl_t,&inverse_dielectric,occ_size,vir_size,
            r1_min_nontda_sq,r1_max_nontda_sq,m_expected,max_feast_iter,tol_feast,None,
            None, None, "diagonal", None, 0.0001, 0, 0);
        println!("Round 1 Triplets took {:?}",start.elapsed()-one_feast_time);
    }

    let n_s = round1_singlet.len();
    let n_t = round1_triplet.len();
    println!("Round 1 found {} singlet + {} triplet eigenpairs", n_s, n_t);

    let eigvecs_s: Vec<Vec<f64>> = round1_singlet.into_iter().map(|(_, v)| v).collect();
    let eigvecs_t: Vec<Vec<f64>> = round1_triplet.into_iter().map(|(_, v)| v).collect();

    // Clear BSE integrals → fallback to regular RI for Rayleigh-Ritz
    scf_data.ri3fn_bse = None;
    scf_data.rimatr_bse = None;

    // Rayleigh-Ritz refinement with full integrals
    let mut qp_ctrl_rr_s = qp_ctrl.clone();
    qp_ctrl_rr_s.bse_spin = String::from("singlet");
    let (ritz_vals_s, ritz_vecs_s) = rayleigh_ritz_refine(scf_data, &qp_ctrl_rr_s, &quasiparticle_energies,
                                         occ_size, vir_size, &eigvecs_s);
    let (fil_vals_s, fil_vecs_s) = filter_eigenpairs(ritz_vals_s, ritz_vecs_s,
                                                      eigenrange_min_orig, eigenrange_max_orig);
    print_ritz_eigenpairs(scf_data, &fil_vals_s, &fil_vecs_s, "singlet", occ_size, vir_size);
    let mut qp_ctrl_rr_t = qp_ctrl.clone();
    qp_ctrl_rr_t.bse_spin = String::from("triplet");
    let (ritz_vals_t, ritz_vecs_t) = rayleigh_ritz_refine(scf_data, &qp_ctrl_rr_t, &quasiparticle_energies,
                                         occ_size, vir_size, &eigvecs_t);
    let (fil_vals_t, fil_vecs_t) = filter_eigenpairs(ritz_vals_t, ritz_vecs_t,
                                                      eigenrange_min_orig, eigenrange_max_orig);
    print_ritz_eigenpairs(scf_data, &fil_vals_t, &fil_vecs_t, "triplet", occ_size, vir_size);

    let eigenpairs_singlet_raw: Vec<(f64, Vec<f64>)> =
        fil_vals_s.into_iter().zip(fil_vecs_s.into_iter()).collect();
    let eigenpairs_triplet_raw: Vec<(f64, Vec<f64>)> =
        fil_vals_t.into_iter().zip(fil_vecs_t.into_iter()).collect();

    // For non-TDA the Ritz vectors are length-n X+Y-space vectors; rebuild them
    // into Davidson-style [X;Y] (length 2n) so export/dipole code works. TDA
    // vectors are already in the correct layout (X block, Y=0) — pass through.
    let eigenpairs_singlet = if qp_ctrl.bse_tda {
        eigenpairs_singlet_raw
    } else {
        reconstruct_nontda_pairs_to_xy(scf_data, &qp_ctrl_rr_s, eigenpairs_singlet_raw, occ_size, vir_size)
    };
    let eigenpairs_triplet = if qp_ctrl.bse_tda {
        eigenpairs_triplet_raw
    } else {
        reconstruct_nontda_pairs_to_xy(scf_data, &qp_ctrl_rr_t, eigenpairs_triplet_raw, occ_size, vir_size)
    };
    (eigenpairs_singlet,eigenpairs_triplet)
}