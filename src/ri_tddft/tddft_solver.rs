/// TDDFT solver orchestrator
///
/// Coordinates the full TDDFT calculation:
/// 1. Read control parameters
/// 2. Prepare fxc kernel data
/// 3. Obtain and reshape RI integrals
/// 4. Build diagonal preconditioner
/// 5. Generate initial guess
/// 6. Call Davidson solver (TDA or full LR)
/// 7. Print excitation energies and properties

use rest_tensors::MatrixFull;
use crate::scf_io::SCF;
use crate::ri_bse::dipoles;
use crate::solvers::davidson as davidson_solver;
use crate::solvers::davidson::DavidsonConfig;
use crate::dft::num_int::{FXCMatvecData, prepare_fxc_data, set_fxc_use_optimized};
use crate::ri_tddft::matvec::{self, a_matvec, b_matvec};
use crate::ri_tddft::utils::{tddft_occupation_parameters, tddft_get_submatrix, compute_tddft_dipole_matrix};
use crate::ri_tddft::feast_solver;

/// Main TDDFT entry point
///
/// Called from main_driver after SCF convergence.
/// Expects scf.mol.ctrl.tddft to be Some(...) with valid TDDFT parameters.

pub struct TddftOutput {
    pub energies: Vec<f64>,
    pub osc: Vec<f64>,
}

pub fn tddft_main(scf: &mut SCF) -> Result<TddftOutput, String> {
    // ═══ Step 1: Extract control parameters ═══
    let tddft_ctrl = scf.mol.ctrl.tddft.clone()
        .ok_or_else(|| "TDDFT control parameters not set".to_string())?;
    let nroots = tddft_ctrl.nroots.max(1);
    let tddft_method = tddft_ctrl.tddft_method.clone();
    let tddft_spin = tddft_ctrl.tddft_spin.clone();
    let xlet = if tddft_spin == "singlet" { 'S' } else if tddft_spin == "triplet" { 'T' } else { 'R' };
    let is_tda = tddft_method == "tda" || tddft_method == "TDA";

    // Enable optimised (rayon-parallel) fxc kernel if requested
    set_fxc_use_optimized(tddft_ctrl.tddft_use_optimized_fxc);

    println!("\n=== TDDFT Calculation ===");
    println!("Method: {}", if is_tda { "TDA" } else { "Full LR" });
    println!("Spin: {}", if xlet == 'S' { "Singlet" } else { "Triplet" });
    println!("Number of roots: {}", nroots);

    // ═══ Step 2: Get orbital dimensions ═══
    let (start_mo, num_state, occ_size, vir_size, homo, lumo) =
        tddft_occupation_parameters(scf);
        if scf.mol.ctrl.print_level > 1 {
            let cutoff = scf.mol.ctrl.tddft.as_ref().map(|c| c.tddft_cutoff_energy).unwrap_or(1.0e6);
            if cutoff < 1.0e5 {
                println!("  TDDFT virtual cutoff: {:.4} Ha, {} states retained", cutoff, num_state);
            }
            if start_mo > scf.mol.start_mo {
                println!("  TDDFT frozen core: -2.00 Ha threshold, {} orbitals frozen (MO 0..{})",
                    start_mo - scf.mol.start_mo, start_mo);
            }
        }
        let dim = occ_size * vir_size;
    if dim == 0 {
        return Err("No occupied-virtual excitation space (all orbitals frozen)".to_string());
    }
    println!("occ_size={}, vir_size={}, dim={}", occ_size, vir_size, dim);

    // ═══ Step 3: Prepare fxc data ═══
    let fxc_data = prepare_fxc_data(scf);
    let alpha_hybrid = fxc_data.alpha_hybrid;

    // ═══ Step 4: Obtain and reshape RI integrals ═══
    println!("Obtaining RI integrals...");
    let ri_ov = tddft_get_submatrix(scf, 'O', 'V', start_mo, occ_size, vir_size, homo, lumo, num_state);
    let ri_oo = tddft_get_submatrix(scf, 'O', 'O', start_mo, occ_size, vir_size, homo, lumo, num_state);
    let ri_vv = tddft_get_submatrix(scf, 'V', 'V', start_mo, occ_size, vir_size, homo, lumo, num_state);
    let num_auxbas = ri_ov.size[0];
    println!("num_auxbas = {}", num_auxbas);

    // Reshape RI_OO for A-block exchange: [naux, occ*occ] → [occ*naux, occ]
    let mut ri_oo_exch = ri_oo.clone();
    ri_oo_exch.reshape([num_auxbas * occ_size, occ_size]);
    ri_oo_exch = ri_oo_exch.transpose_and_drop();
    ri_oo_exch.reshape([occ_size * num_auxbas, occ_size]);

    // Reshape RI_VV for A-block exchange: [naux, vir*vir] → [naux*vir, vir]
    let mut ri_vv_exch = ri_vv.clone();
    ri_vv_exch.reshape([num_auxbas * vir_size, vir_size]);

    // Reshape RI_OV for B-block exchange: [naux, occ*vir] → [naux*occ, vir]
    let mut ri_ov_exch = ri_ov.clone();
    ri_ov_exch.reshape([num_auxbas * occ_size, vir_size]);

    // ═══ Step 5: Build diagonal preconditioner ═══
    let hdiag = matvec::build_hdiag(scf);
    println!("Diagonal preconditioner built, min gap = {:.6}",
        hdiag.iter().fold(f64::INFINITY, |a, &b| a.min(b)));

    // ═══ Step 6: Generate initial guess ═══
    let init_nroots = nroots.min(dim);
    let initial_guess = davidson_solver::generate_initial_guess(&hdiag, init_nroots);
    println!("Initial guess generated: {} vectors", initial_guess.size[1]);

    // ═══ Step 7: Prepare control parameters for Davidson solver ═══
    let converged_tol = tddft_ctrl.davidson_tol.max(1e-12);
    let max_iter = tddft_ctrl.davidson_max_iter.max(10);
    let max_subspace = (nroots * 4).max(tddft_ctrl.davidson_max_subspace).min(dim);
    let add_dim = nroots.min(tddft_ctrl.davidson_max_subspace).max(2)
        .min((dim / 2).max(nroots));
    // Ensure max_subspace > add_dim to prevent subtraction underflow
    let max_subspace = max_subspace.max(add_dim + 1).min(dim);

    // Davidson solver configuration (decoupled from QuasiParticle)
    let davidson_cfg = DavidsonConfig {
        max_subspace,
        add_dim,
        restart_dim: nroots.max(2),
        max_iter,
        tol: converged_tol,
        ..Default::default()
    };

    // ═══ Step 8: Diagnostic: check A matrix symmetry for first few columns ═══
    {
        let ndiag = dim.min(6);
        // Test A[i,j] vs A[j,i] for first ndiag columns
        let mut a_probe = vec![0.0; ndiag * ndiag * 2]; // *2 for two sets
        let mut e_col = vec![0.0; dim];
        for col in 0..ndiag {
            e_col[col] = 1.0;
            let a_col = a_matvec(scf, &fxc_data, &ri_ov, &ri_oo_exch, &ri_vv_exch, &e_col, xlet, alpha_hybrid);
            e_col[col] = 0.0;
            for row in 0..ndiag {
                a_probe[row + col * ndiag] = a_col[row];
            }
        }
        let mut max_asym = 0.0;
        let mut max_offdiag = 0.0;
        for j in 0..ndiag {
            for i in 0..ndiag {
                let diff = (a_probe[i + j * ndiag] - a_probe[j + i * ndiag]).abs();
                if diff > max_asym { max_asym = diff; }
                if i != j {
                    let off = a_probe[i + j * ndiag].abs();
                    if off > max_offdiag { max_offdiag = off; }
                }
            }
        }
        // Also check fxc tensor symmetry for GGA
        if fxc_data.nvar == 4 {
            let nv2 = 16;
            let mut max_fxc_asym = 0.0;
            for g in (0..fxc_data.ngrids).step_by(fxc_data.ngrids.max(1) / 10) {
                for alpha in 0..4 {
                    for beta in 0..4 {
                        let idx_ab = g + alpha * fxc_data.ngrids + beta * 4 * fxc_data.ngrids;
                        let idx_ba = g + beta * fxc_data.ngrids + alpha * 4 * fxc_data.ngrids;
                        let diff = (fxc_data.wfxc[idx_ab] - fxc_data.wfxc[idx_ba]).abs();
                        if diff > max_fxc_asym { max_fxc_asym = diff; }
                    }
                }
            }
            println!("    GGA fxc tensor max asymmetry = {:.2e}", max_fxc_asym);
        }
        println!("  A matrix diagnostic: first {}×{} submatrix", ndiag, ndiag);
        println!("    Max asymmetry |A[i,j]-A[j,i]| = {:.2e}", max_asym);
        println!("    Max off-diagonal element = {:.6}", max_offdiag);
        for i in 0..ndiag.min(3) {
            println!("    A[{},{}] = {:.10} (gap={:.10}, kernel={:.10})",
                i, i, a_probe[i + i * ndiag], hdiag[i], a_probe[i + i * ndiag] - hdiag[i]);
        }
    }

    // ═══ Step 8: Call solver (full diag, Davidson, or FEAST) ═══
    let eigenpairs = if tddft_ctrl.tddft_feast_solver {
        // ── FEAST solver path ──
        println!("Using FEAST eigensolver (experimental)");
        let eigenrange_min = tddft_ctrl.tddft_feast_eigenrange_min;
        let eigenrange_max = tddft_ctrl.tddft_feast_eigenrange_max;
        let m_expected = tddft_ctrl.tddft_feast_m_expected;
        let max_feast_iter = tddft_ctrl.tddft_feast_max_iter;
        let tol_feast = tddft_ctrl.tddft_feast_tol;
        let gmres_restart = tddft_ctrl.tddft_feast_gmres_restart;
        let gmres_max_iter = tddft_ctrl.tddft_feast_gmres_max_iter;
        let gmres_tol = tddft_ctrl.tddft_feast_cg_tol;
        let cg_max_iter = tddft_ctrl.tddft_feast_cg_max_iter;
        let cg_tol = tddft_ctrl.tddft_feast_cg_tol;
        let init_guess_type = tddft_ctrl.tddft_feast_init_guess_type.as_str();
        let gaussian_width_factor = tddft_ctrl.tddft_feast_gaussian_width_factor;

        if is_tda {
            feast_solver::feast_solve_tddft_tda(
                scf, &fxc_data,
                &ri_ov, &ri_oo_exch, &ri_vv_exch,
                &hdiag, xlet, alpha_hybrid,
                eigenrange_min, eigenrange_max,
                m_expected, max_feast_iter, tol_feast,
                gmres_restart, gmres_max_iter, gmres_tol,
                init_guess_type, gaussian_width_factor,
            )
        } else {
            feast_solver::feast_solve_tddft_lr(
                scf, &fxc_data,
                &ri_ov, &ri_oo_exch, &ri_vv_exch, &ri_ov_exch,
                &hdiag, xlet, alpha_hybrid,
                eigenrange_min, eigenrange_max,
                m_expected, max_feast_iter, tol_feast,
                gmres_restart, gmres_max_iter, gmres_tol,
                cg_max_iter, cg_tol,
                init_guess_type, gaussian_width_factor,
            )
        }
    } else if dim <= 15 {
        println!("Small system (dim={}), building full A matrix for diagnosis...", dim);
        let mut a_mat = vec![0.0; dim * dim];
        for col in 0..dim {
            let mut e_col = vec![0.0; dim];
            e_col[col] = 1.0;
            let a_col = a_matvec(scf, &fxc_data, &ri_ov, &ri_oo_exch, &ri_vv_exch, &e_col, xlet, alpha_hybrid);
            for row in 0..dim {
                a_mat[row + col * dim] = a_col[row];
            }
            if col == 0 {
                println!("  A·e0 [0] = {:.10} (gap={:.10}, kernel={:.10})",
                    a_col[0], hdiag[0], a_col[0] - hdiag[0]);
            }
        }
        // Diagonalize
        let mut a = MatrixFull::from_vec([dim, dim], a_mat).unwrap();
        let (eigvecs_opt, eigvals, _info) = rest_tensors::matrix::matrix_blas_lapack::_dsyev(&a, 'V');
        let eigvecs = eigvecs_opt.expect("dsyev failed");
        let mut pairs: Vec<(f64, Vec<f64>)> = eigvals.iter()
            .zip(eigvecs.iter_columns_full())
            .map(|(e, v)| (*e, v.to_vec()))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        pairs.truncate(nroots.min(dim));
        pairs
    } else if is_tda {
        println!("Solving TDA eigenvalue problem...");
        davidson_solver::tda_davidson_solver(
            |z: &Vec<f64>| a_matvec(scf, &fxc_data, &ri_ov, &ri_oo_exch, &ri_vv_exch, z, xlet, alpha_hybrid),
            nroots,
            &hdiag,
            initial_guess,
            &davidson_cfg,
        )
    } else {
        println!("Solving full linear response eigenvalue problem...");
        davidson_solver::lr_davidson_solver(|z: &Vec<f64>| a_matvec(scf, &fxc_data, &ri_ov, &ri_oo_exch, &ri_vv_exch, z, xlet, alpha_hybrid),
            |z: &Vec<f64>| b_matvec(scf, &fxc_data, &ri_ov, &ri_ov_exch, z, xlet, alpha_hybrid),
            nroots,
            &hdiag,
            initial_guess,
            &davidson_cfg,
        )
    };

    // ═══ Step 9: Compute and print results (BSE-compatible format) ═══
    let n_found = eigenpairs.len();
    let tda_flag = is_tda;
    let n_print = n_found.min(30);
    let dipole_matrix = compute_tddft_dipole_matrix(scf, start_mo, occ_size, vir_size, homo, lumo);
    let singlet_triplet = if xlet == 'S' { "Singlet" } else if xlet == 'T' { "Triplet" } else { "" };
    println!("\nFirst {} {} Excitations:", n_found.min(n_print), singlet_triplet);

    let mut td_energies: Vec<f64> = Vec::new();
    let mut td_osc: Vec<f64> = Vec::new();
    for (n, (energy, vector)) in eigenpairs[..n_print].iter().enumerate() {
        let vec_norm: f64 = vector.iter().map(|x| x * x).sum::<f64>().sqrt();

        println!("#{} Excitation energy={}, norm={:.6}", n, energy, vec_norm);

        // Normalize and compute transition dipole (BSE-compatible order:
        // transition_dipole_square prints "Dipole Moment Components" as a side effect)
        let norm_vec = dipoles::normalize(vector, tda_flag);
        let dipole_sq = dipoles::transition_dipole_square(&dipole_matrix, &norm_vec, tda_flag);
        let osc_strength = dipole_sq * energy * 2.0 / 3.0;
        td_energies.push(*energy);
        td_osc.push(osc_strength);
        println!("\tTransition Dipole Square:{}; Oscillator Strength:{}",
            dipole_sq, osc_strength);

        // Print leading components
        let mut components: Vec<(usize, usize, f64)> = norm_vec.iter()
            .enumerate()
            .map(|(idx, &val)| {
                let i = idx % occ_size;
                let a = idx / occ_size;
                (i, a, val)
            })
            .collect();
        components.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());
        for (k, (i, a, val)) in components.iter().enumerate() {
            if k < 5 {
                println!("      #{}->#{},amplitude={}", start_mo + i, lumo + a, val);
            }
        }
    }

    println!("The first excitation obtained by TDDFT is {}", eigenpairs[0].0);
    println!("TDDFT calculation completed successfully.");
    Ok(TddftOutput { energies: td_energies, osc: td_osc })
}

/// Full TDA diagonalization for small systems (dim <= 3)
/// Avoids Davidson solver issues with tiny problem sizes.
fn full_diag_tda(
    scf: &SCF,
    fxc_data: &FXCMatvecData,
    ri_ov: &MatrixFull<f64>,
    ri_oo_exch: &MatrixFull<f64>,
    ri_vv_exch: &MatrixFull<f64>,
    dim: usize,
    xlet: char,
    alpha_hybrid: f64,
    nroots: usize,
) -> Vec<(f64, Vec<f64>)> {
    // Build the full A matrix column by column
    let mut a_mat = vec![0.0; dim * dim];
    for col in 0..dim {
        let mut e_col = vec![0.0; dim];
        e_col[col] = 1.0;
        let a_col = a_matvec(scf, fxc_data, ri_ov, ri_oo_exch, ri_vv_exch, &e_col, xlet, alpha_hybrid);
        for row in 0..dim {
            a_mat[row + col * dim] = a_col[row];
        }
    }

    // Diagonalize
    let n = dim;
    let mut a = MatrixFull::from_vec([n, n], a_mat).unwrap();
    let (eigvecs_opt, eigvals, _info) = rest_tensors::matrix::matrix_blas_lapack::_dsyev(&a, 'V');
    let eigvecs = eigvecs_opt.expect("dsyev failed for TDA full diagonalization");

    // Sort and return
    let mut pairs: Vec<(f64, Vec<f64>)> = eigvals.iter()
        .zip(eigvecs.iter_columns_full())
        .map(|(e, v)| (*e, v.to_vec()))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    pairs.truncate(nroots);
    pairs
}
