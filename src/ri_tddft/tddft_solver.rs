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
use crate::scf_io::{SCF, SCFType};
use crate::ri_bse::dipoles;
use crate::solvers::davidson as davidson_solver;
use crate::solvers::davidson::DavidsonConfig;
use crate::dft::num_int::{FXCMatvecData, prepare_fxc_data, set_fxc_use_optimized};
use crate::ri_tddft::matvec::{self, a_matvec, b_matvec};
use crate::ri_tddft::matvec_ao;
use crate::ri_tddft::utils::{tddft_occupation_parameters, tddft_occupation_parameters_u,
    compute_tddft_dipole_matrix, compute_tddft_dipole_matrix_u, normalize_u,
    transition_dipole_square_u, tddft_get_submatrix};
use crate::ri_tddft::feast_solver;
use crate::ri_tddft::tddft::{build_a, build_b, prepare_ao_data, prepare_mo_data};
use crate::ri_tddft::{TDDFTData, TDDFTMode};
use log::{warn, debug};

/// Main TDDFT entry point
///
/// Called from main_driver after SCF convergence.
/// Expects scf.mol.ctrl.tddft to be Some(...) with valid TDDFT parameters.

pub struct TddftOutput {
    pub energies: Vec<f64>,
    pub osc: Vec<f64>,
}

/// Dense full-LR eigenpairs via the symmetrized Casida reduction (mirrors the
/// LR Davidson's preferred Cholesky route, solvers/davidson.rs):
///
///   A±B;  A−B = G Gᵀ (Cholesky);  Gᵀ(A+B)G Z' = ω² Z';
///   Z = X+Y = G Z';  W = X−Y = ω G⁻ᵀ Z';  X = (Z+W)/2, Y = (Z−W)/2.
///
/// Each returned pair is `(ω, [X; Y])` — the same format as the batched LR
/// Davidson (length 2·dim), so the Step-9 post-processing is identical.
/// Returns `None` when A−B is not positive-definite (the Davidson's fallback
/// convention: warn and degrade to the TDA approximation).
fn dense_lr_eigenpairs(
    a_full: &MatrixFull<f64>,
    b_full: &MatrixFull<f64>,
    nroots: usize,
) -> Option<Vec<(f64, Vec<f64>)>> {
    use rest_tensors::matrix::matrix_blas_lapack::{_dgemm_full, _dinverse, _dpotrf, _dsyev};
    let dim = a_full.size[0];
    let mut apb = MatrixFull::new([dim, dim], 0.0);
    let mut amb = MatrixFull::new([dim, dim], 0.0);
    for i in 0..dim {
        for j in 0..dim {
            apb[[i, j]] = a_full[[i, j]] + b_full[[i, j]];
            amb[[i, j]] = a_full[[i, j]] - b_full[[i, j]];
        }
    }
    // Cholesky A−B = G Gᵀ; `_dpotrf` panics when A−B is not positive-definite.
    let mut g = amb;
    let chol_ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        _dpotrf(&mut g, 'L');
    }));
    if chol_ok.is_err() {
        warn!(
            "  Dense LR: Cholesky of (A-B) failed (not positive-definite); \
             falling back to TDA approximation."
        );
        return None;
    }
    for i in 0..dim {
        for j in 0..dim {
            if j > i {
                g[[i, j]] = 0.0;
            }
        }
    }
    // Symmetric problem: Gᵀ(A+B)G Z' = ω² Z'
    let mut apb_g = MatrixFull::new([dim, dim], 0.0);
    _dgemm_full(&apb, 'N', &g, 'N', &mut apb_g, 1.0, 0.0);
    let mut gt_apb_g = MatrixFull::new([dim, dim], 0.0);
    _dgemm_full(&g, 'T', &apb_g, 'N', &mut gt_apb_g, 1.0, 0.0);
    let (eigvecs_opt, omega2, _info) = _dsyev(&gt_apb_g, 'V');
    let eigvecs = eigvecs_opt?;
    let ginv = _dinverse(&g)?;
    let mut pairs: Vec<(f64, Vec<f64>)> = Vec::new();
    for (w2, v) in omega2.iter().zip(eigvecs.iter_columns_full()) {
        if *w2 <= 1e-12 {
            continue;
        }
        let omega = w2.sqrt();
        // Z = X+Y = G z'; W = X−Y = ω G⁻ᵀ z'
        let z_col = MatrixFull::from_vec([dim, 1], v.to_vec()).unwrap();
        let mut xpy = MatrixFull::new([dim, 1], 0.0);
        _dgemm_full(&g, 'N', &z_col, 'N', &mut xpy, 1.0, 0.0);
        let mut gtz = MatrixFull::new([dim, 1], 0.0);
        _dgemm_full(&ginv, 'T', &z_col, 'N', &mut gtz, 1.0, 0.0);
        let mut vec = Vec::with_capacity(2 * dim);
        for i in 0..dim {
            let z_i = xpy[[i, 0]];
            let w_i = omega * gtz[[i, 0]];
            vec.push(0.5 * (z_i + w_i)); // X
        }
        for i in 0..dim {
            let z_i = xpy[[i, 0]];
            let w_i = omega * gtz[[i, 0]];
            vec.push(0.5 * (z_i - w_i)); // Y
        }
        pairs.push((omega, vec));
        if pairs.len() >= nroots {
            break;
        }
    }
    if pairs.is_empty() {
        None
    } else {
        Some(pairs)
    }
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
    let is_ao = tddft_ctrl.tddft_mode == "ao";

    // ═══ Unrestricted (UKS/UHF) reference: collinear uTDDFT ═══
    // The amplitude space is the concatenation [z_alpha; z_beta] (PySCF
    // tdscf/uhf.py layout); the response is not spin-resolved (no
    // singlet/triplet). Phase 1 supports AO mode only.
    // Gate on the SCF REFERENCE TYPE, not spin_channel: ROHF also carries
    // spin_channel == 2 (scf_io sets it for the two-spin Roothaan density),
    // but its eigenvectors are not the semicanonical orbitals the RI/UKS
    // machinery needs — ROHF-TDDFT is rejected explicitly until implemented
    // (it would require semi_eigenvectors + a polarized kernel).
    let is_u = scf.scftype == SCFType::UHF;
    if scf.scftype == SCFType::ROHF {
        return Err("ROHF-TDDFT is not yet supported. Use spin_polarization=true \
                    (UKS) for open-shell excited states."
            .to_string());
    }
    if is_u {
        if !is_ao {
            return Err("Unrestricted (UKS) TDDFT currently requires tddft_mode=\"ao\" \
                        (the MO-mode kernels are still restricted-only).".to_string());
        }
        if tddft_ctrl.tddft_feast_solver {
            return Err("FEAST solver is not supported for unrestricted (UKS) TDDFT.".to_string());
        }
        if tddft_ctrl.tddft_fxc_driver == "mo" {
            return Err("tddft_fxc_driver=\"mo\" is not supported for unrestricted (UKS) TDDFT; \
                        use \"semitrans\" or \"dm\".".to_string());
        }
        if tddft_spin != "singlet" {
            // The default is "singlet", so an explicit triplet/nonscf setting
            // cannot be distinguished — unrestricted response ignores the
            // keyword either way (debug-level note only).
            log::debug!("tddft_spin=\"{}\" is ignored for an unrestricted reference", tddft_spin);
        }
    }

    // `grid_batch` is an AO-mode-only memory-bounded fxc option (default true).
    // MO mode does not use the grid-batched kernel, so the flag is silently
    // ignored there (visible at debug level).
    if tddft_ctrl.grid_batch && !is_ao {
        warn!("grid_batch is only applicable in AO mode; ignoring it in MO mode.");
    }

    // Triplet TDDFT is only supported in AO mode: the MO-mode fxc kernel
    // (prepare_fxc_data) still hardcodes the singlet factor, which would
    // silently produce wrong triplet roots (verified H2/PBE0: MO triplet
    // deviates from PySCF by ~1e-2 Ha). AO mode implements the spin-polarized
    // kernel combination (CPL, 256, 454).
    if !is_ao && tddft_spin == "triplet" {
        return Err("Triplet TDDFT is not yet supported in MO mode \
                    (the fxc kernel is hardcoded singlet); use tddft_mode=\"ao\".".to_string());
    }

    // Enable optimised (rayon-parallel) fxc kernel if requested
    set_fxc_use_optimized(tddft_ctrl.tddft_use_optimized_fxc);

    println!("\n=== TDDFT Calculation ===");
    println!("Method: {}", if is_tda { "TDA" } else { "Full LR" });
    println!("Mode: {}", tddft_ctrl.tddft_mode);
    if is_ao {
        println!("  RI-K driver: {}", tddft_ctrl.tddft_ao_rik_driver);
        println!("  fxc driver: {}", tddft_ctrl.tddft_fxc_driver);
    }
    if is_u {
        println!("Spin: Unrestricted (alpha + beta excitation sectors)");
    } else {
        println!("Spin: {}", if xlet == 'S' { "Singlet" } else { "Triplet" });
    }
    println!("Number of roots: {}", nroots);

    // ═══ Step 2: Get orbital dimensions ═══
    let (start_mo, num_state, occ_size, vir_size, homo, lumo) =
        tddft_occupation_parameters(scf);
    // Unrestricted: per-spin occupied/virtual windows (frozen core + virtual
    // cutoff resolved independently on each spin channel).
    let sectors_u = if is_u { Some(tddft_occupation_parameters_u(scf)) } else { None };
        if scf.mol.ctrl.print_level > 1 {
        if let Some(sec) = &sectors_u {
            for (i_spin, s) in sec.iter().enumerate() {
                let tag = if i_spin == 0 { "alpha" } else { "beta" };
                println!("  TDDFT {} sector: occ={} [MO {}..{}], vir={} [MO {}..{}], dim={}",
                    tag, s.occ_size, s.start_mo, s.homo, s.vir_size, s.lumo, s.num_state - 1, s.dim());
            }
        } else {
            let cutoff = scf.mol.ctrl.tddft.as_ref().map(|c| c.tddft_cutoff_energy).unwrap_or(1.0e6);
            if cutoff < 1.0e5 {
                println!("  TDDFT virtual cutoff: {:.4} Ha, {} states retained", cutoff, num_state);
            }
            if start_mo > scf.mol.start_mo {
                println!("  TDDFT frozen core: -2.00 Ha threshold, {} orbitals frozen (MO 0..{})",
                    start_mo - scf.mol.start_mo, start_mo);
            }
        }
        }
    let dim = match &sectors_u {
        Some(sec) => sec[0].dim() + sec[1].dim(),
        None => occ_size * vir_size,
    };
    if dim == 0 {
        return Err("No occupied-virtual excitation space (all orbitals frozen)".to_string());
    }
    if let Some(sec) = &sectors_u {
        println!("occ_a={}, vir_a={}, dim_a={}; occ_b={}, vir_b={}, dim_b={}; dim={}",
            sec[0].occ_size, sec[0].vir_size, sec[0].dim(),
            sec[1].occ_size, sec[1].vir_size, sec[1].dim(), dim);
    } else {
        println!("occ_size={}, vir_size={}, dim={}", occ_size, vir_size, dim);
    }

    // ═══ Step 3+4: Prepare the shared TDDFT data (fxc kernel + mode-specific tensors) ═══
    // AO mode is prepared by `prepare_ao_data` (fxc kernel via numint_matmul,
    // NIMatmul integrator, AO transition-density path); MO mode by
    // `prepare_mo_data` (MO-basis RI tensors). Both return the same `TDDFTData`.
    // RefCell: shared mutable state needed by the batched Davidson closures,
    // each of which requires `&mut TDDFTData` (NIMatmul cache).
    //
    // Free the SCF-tabulated dense AO tables (grids.ao / grids.aop — the
    // largest scratch tables of an AO-mode run) before preparing AO-mode
    // data: the AO paths never read them, and no other consumer reads them
    // afterwards (Hirshfeld decompresses on demand).
    if is_ao {
        if let Some(g) = scf.grids.as_mut() {
            g.ao = None;
            g.aop = None;
        }
    }
    let data: std::cell::RefCell<TDDFTData> = std::cell::RefCell::new(
        if is_ao {
            if is_u {
                println!("Unrestricted (UKS) AO mode: spin-polarized kernels over the concatenated \
                          [alpha; beta] amplitude space");
            } else {
                println!("AO mode: using AO transition-density kernels (no MO-basis RI tensors)");
            }
            // FEAST is not implemented for the AO path.
            // Note: `response_tddft` bypasses this function entirely (dispatched
            // separately in main_driver) and always uses MO-basis machinery
            // regardless of `tddft_mode`.
            if tddft_ctrl.tddft_feast_solver {
                return Err("FEAST solver is not supported with tddft_mode=\"ao\"".to_string());
            }
            // `prepare_ao_data` handles both reference types (RHF one-sector /
            // UHF two-sector kernels and coefficients).
            prepare_ao_data(scf)
        } else {
            prepare_mo_data(scf)
        },
    );

    // Hybrid coefficient: identical for both modes (from the kernel data).
    let alpha_hybrid = data.borrow().alpha_hybrid;

    // ═══ Step 5: Build diagonal preconditioner ═══
    let hdiag = if is_u { matvec::build_hdiag_u(scf) } else { matvec::build_hdiag(scf) };
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
    // MO-mode matvec closures (used by diagnostic + MO solver dispatch; the
    // dense path and AO mode use the dedicated builders / batched matvecs).
    let scf_ref: &SCF = scf;
    let a_apply = |z: &Vec<f64>| -> Vec<f64> {
        let d = data.borrow();
        a_matvec(scf_ref, &d, z, xlet)
    };
    let b_apply = |z: &Vec<f64>| -> Vec<f64> {
        let d = data.borrow();
        b_matvec(scf_ref, &d, z, xlet)
    };

    // fxc kernel table (mode-agnostic view for diagnostics).
    // Gated behind print_level > 1: the A-matrix probe costs min(6, dim) extra
    // single-column matvec applications (no set amortization), and both
    // diagnostics are only watched in verbose runs. The scope also ensures the
    // immutable RefCell guard is dropped before the solver dispatch (which
    // borrows `data` mutably).
    if scf.mol.ctrl.print_level > 1 {
    // fxc tensor symmetry check for GGA (uses the MO-mode kernel table `wfxc`,
    // which carries the `[g,α,β]` weighted layout; AO mode stores the raw
    // kernel in `fxc_eff`/NIMatmul instead, so the check is MO-only).
    let kernel_guard = data.borrow();
    if let Some(kernel) = &kernel_guard.fxc {
        if kernel.nvar == 4 {
            let nv2 = 16;
            let mut max_fxc_asym = 0.0;
            for g in (0..kernel.ngrids).step_by(kernel.ngrids.max(1) / 10) {
                for alpha in 0..4 {
                    for beta in 0..4 {
                        let idx_ab = g + alpha * kernel.ngrids + beta * 4 * kernel.ngrids;
                        let idx_ba = g + beta * kernel.ngrids + alpha * 4 * kernel.ngrids;
                        let diff = (kernel.wfxc[idx_ab] - kernel.wfxc[idx_ba]).abs();
                        if diff > max_fxc_asym { max_fxc_asym = diff; }
                    }
                }
            }
            println!("    GGA fxc tensor max asymmetry = {:.2e}", max_fxc_asym);
        }
    }
    drop(kernel_guard);

    let ndiag = dim.min(6);
        // Test A[i,j] vs A[j,i] for first ndiag columns.
        // MO mode: per-vector a_apply; AO mode: batched matvec on single-column blocks
        // (AO-on-grids for the per-vector path is no longer populated).
        let mut a_probe = vec![0.0; ndiag * ndiag * 2]; // *2 for two sets
        let mut e_col = vec![0.0; dim];
        for col in 0..ndiag {
            e_col[col] = 1.0;
            let a_col = if matches!(data.borrow().mode, TDDFTMode::AO) {
                let mut block = MatrixFull::from_vec([dim, 1], e_col.clone()).unwrap();
                let res = matvec_ao::a_matvec_ao_batched(
                    scf_ref, &mut *data.borrow_mut(), &mut block, xlet);
                res.data.clone()
            } else {
                a_apply(&e_col)
            };
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
                scf, &*data.borrow(), &hdiag, xlet,
                eigenrange_min, eigenrange_max,
                m_expected, max_feast_iter, tol_feast,
                gmres_restart, gmres_max_iter, gmres_tol,
                init_guess_type, gaussian_width_factor,
            )
        } else {
            feast_solver::feast_solve_tddft_lr(
                scf, &*data.borrow(), &hdiag, xlet,
                eigenrange_min, eigenrange_max,
                m_expected, max_feast_iter, tol_feast,
                gmres_restart, gmres_max_iter, gmres_tol,
                cg_max_iter, cg_tol,
                init_guess_type, gaussian_width_factor,
            )
        }
    } else if dim <= 15 {
        println!("Small system (dim={}), building full A matrix for diagnosis...", dim);
        let a_full = build_a(scf_ref, &mut *data.borrow_mut(), xlet);
        if log::log_enabled!(log::Level::Debug) {
            println!("  A·e0 [0] = {:.10} (gap={:.10}, kernel={:.10})",
                a_full[[0, 0]], hdiag[0], a_full[[0, 0]] - hdiag[0]);
        }
        // Full LR: build B too and solve the symmetrized Casida reduction
        // (fall back to the TDA approximation if A−B is not positive-definite).
        let lr_pairs = if is_tda {
            None
        } else {
            let b_full = build_b(scf_ref, &mut *data.borrow_mut(), xlet);
            dense_lr_eigenpairs(&a_full, &b_full, nroots)
        };
        if let Some(pairs) = lr_pairs {
            pairs
        } else {
            // TDA (or LR with A−B not positive-definite): diagonalize A only.
            let mut a = a_full;
            let (eigvecs_opt, eigvals, _info) = rest_tensors::matrix::matrix_blas_lapack::_dsyev(&a, 'V');
            let eigvecs = eigvecs_opt.expect("dsyev failed");
            let mut pairs: Vec<(f64, Vec<f64>)> = eigvals.iter()
                .zip(eigvecs.iter_columns_full())
                .map(|(e, v)| (*e, v.to_vec()))
                .collect();
            pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            pairs.truncate(nroots.min(dim));
            pairs
        }
    } else {
        // Layered by mode, then by method: AO/MO owns the matvec family,
        // TDA/LR is the inner branch. (FEAST and dim<=15 are outer special cases.)
        let mode = data.borrow().mode;
        match mode {
            TDDFTMode::AO => {
                if is_tda {
                    println!("Solving TDA eigenvalue problem (AO-mode batched matvec)...");
                    davidson_solver::tda_davidson_solver_batched(
                        |z_block: &MatrixFull<f64>| {
                            matvec_ao::a_matvec_ao_batched(scf_ref, &mut *data.borrow_mut(), z_block, xlet)
                        },
                        nroots,
                        &hdiag,
                        initial_guess,
                        &davidson_cfg,
                    )
                } else {
                    println!("Solving full linear response eigenvalue problem (AO-mode batched matvec)...");
                    davidson_solver::lr_davidson_solver_batched(
                        |z_block: &MatrixFull<f64>| {
                            matvec_ao::a_matvec_ao_batched(scf_ref, &mut *data.borrow_mut(), z_block, xlet)
                        },
                        |z_block: &MatrixFull<f64>| {
                            matvec_ao::b_matvec_ao_batched(scf_ref, &mut *data.borrow_mut(), z_block, xlet)
                        },
                        nroots,
                        &hdiag,
                        initial_guess,
                        &davidson_cfg,
                    )
                }
            }
            TDDFTMode::MO => {
                if is_tda {
                    println!("Solving TDA eigenvalue problem...");
                    davidson_solver::tda_davidson_solver(
                        |z: &Vec<f64>| a_apply(z),
                        nroots,
                        &hdiag,
                        initial_guess,
                        &davidson_cfg,
                    )
                } else {
                    println!("Solving full linear response eigenvalue problem...");
                    davidson_solver::lr_davidson_solver(|z: &Vec<f64>| a_apply(z),
                        |z: &Vec<f64>| b_apply(z),
                        nroots,
                        &hdiag,
                        initial_guess,
                        &davidson_cfg,
                    )
                }
            }
        }
    };

    // ═══ Step 9: Compute and print results (BSE-compatible format) ═══
    // Report the kernel-step timing attribution (debug level), per mode.
    match data.borrow().mode {
        TDDFTMode::AO => crate::ri_tddft::matvec_ao::ao_timing_report(),
        TDDFTMode::MO => crate::ri_tddft::matvec::mo_timing_report(),
    }
    let n_found = eigenpairs.len();
    let tda_flag = is_tda;
    let n_print = n_found.min(30);
    let dipole_matrix = if let Some(sec) = &sectors_u {
        compute_tddft_dipole_matrix_u(scf, sec)
    } else {
        compute_tddft_dipole_matrix(scf, start_mo, occ_size, vir_size, homo, lumo)
    };
    let singlet_triplet = if is_u {
        "Unrestricted"
    } else if xlet == 'S' {
        "Singlet"
    } else {
        "Triplet"
    };
    println!("\nFirst {} {} Excitations:", n_found.min(n_print), singlet_triplet);

    let mut td_energies: Vec<f64> = Vec::new();
    let mut td_osc: Vec<f64> = Vec::new();
    for (n, (energy, vector)) in eigenpairs[..n_print].iter().enumerate() {
        let vec_norm: f64 = vector.iter().map(|x| x * x).sum::<f64>().sqrt();

        println!("#{} Excitation energy={}, norm={:.6}", n, energy, vec_norm);

        // Normalize and compute transition dipole (BSE-compatible order:
        // transition_dipole_square prints "Dipole Moment Components" as a side effect)
        // Unrestricted: PySCF uhf.py convention (no closed-shell 2/sqrt(2) factors).
        let norm_vec = if is_u {
            normalize_u(vector, tda_flag)
        } else {
            dipoles::normalize(vector, tda_flag)
        };
        let dipole_sq = if is_u {
            transition_dipole_square_u(&dipole_matrix, &norm_vec, tda_flag)
        } else {
            dipoles::transition_dipole_square(&dipole_matrix, &norm_vec, tda_flag)
        };
        let osc_strength = dipole_sq * energy * 2.0 / 3.0;
        td_energies.push(*energy);
        td_osc.push(osc_strength);
        println!("\tTransition Dipole Square:{}; Oscillator Strength:{}",
            dipole_sq, osc_strength);

        // Print leading components
        if let Some(sec) = &sectors_u {
            // Unrestricted: decode (spin sector, occ, vir) from the concatenated index.
            let dim_a = sec[0].dim();
            let mut components: Vec<(usize, usize, usize, f64)> = norm_vec.iter()
                .enumerate()
                .map(|(idx, &val)| {
                    let (s_i, local) = if idx < dim_a { (0usize, idx) } else { (1usize, idx - dim_a) };
                    (s_i, local % sec[s_i].occ_size, local / sec[s_i].occ_size, val)
                })
                .collect();
            components.sort_by(|a, b| b.3.abs().partial_cmp(&a.3.abs()).unwrap());
            for (k, (s_i, i, a, val)) in components.iter().enumerate() {
                if k < 5 {
                    let spin_char = if *s_i == 0 { 'a' } else { 'b' };
                    println!("      #{}{}->{}{},amplitude={}",
                        sec[*s_i].start_mo + i, spin_char, sec[*s_i].lumo + a, spin_char, val);
                }
            }
        } else {
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
    }

    println!("The first excitation obtained by TDDFT is {}", eigenpairs[0].0);
    println!("TDDFT calculation completed successfully.");
    Ok(TddftOutput { energies: td_energies, osc: td_osc })
}

