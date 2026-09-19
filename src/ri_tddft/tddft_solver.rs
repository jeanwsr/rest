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
use crate::ri_bse::{dipoles, pysoc_export};
use crate::solvers::davidson as davidson_solver;
use crate::solvers::davidson::DavidsonConfig;
use crate::dft::num_int::set_fxc_use_optimized;
use crate::ri_tddft::matvec::{self, a_matvec, b_matvec};
use crate::ri_tddft::matvec_ao;
use crate::ri_tddft::utils::{
    tddft_occupation_parameters, tddft_occupation_parameters_u,
    compute_tddft_dipole_matrix, compute_tddft_dipole_matrix_u, normalize_u, transition_dipole_square_u,
};
use crate::ri_tddft::feast_solver;
use crate::ri_tddft::tddft::{build_a, build_b, prepare_ao_data, prepare_mo_data};
use crate::ri_tddft::{TDDFTData, TDDFTMode};
use log::warn;

/// Main TDDFT entry point
///
/// Called from main_driver after SCF convergence.
/// Expects scf.mol.ctrl.tddft to be Some(...) with valid TDDFT parameters.

pub struct TddftOutput {
    pub energies: Vec<f64>,
    pub osc: Vec<f64>,
    /// `(excitation energy, eigenvector)` pairs, in the solver's own ordering.
    /// The eigenvector is the raw Davidson/dense vector (length `dim` for TDA,
    /// `2*dim` for full LR); callers normalise it with `dipoles::normalize`.
    pub excitations: Vec<(f64, Vec<f64>)>,
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
    let is_tda = tddft_method == "tda" || tddft_method == "TDA";

    // Enable optimised (rayon-parallel) fxc kernel if requested
    set_fxc_use_optimized(tddft_ctrl.tddft_use_optimized_fxc);

    let is_ao = tddft_ctrl.tddft_mode == "ao";

    // ═══ Unrestricted (UKS/UHF) reference: collinear uTDDFT ═══
    // Gate on the SCF REFERENCE TYPE, not spin_channel: ROHF also carries
    // spin_channel == 2 (scf_io sets it for the two-spin Roothaan density),
    // but its eigenvectors are not the semicanonical orbitals the RI/UKS
    // machinery needs — ROHF-TDDFT is rejected explicitly until implemented
    // (it would require semi_eigenvectors + a polarized kernel).
    //
    // UHF references run through the SAME unified flow below for both modes;
    // `tddft_mode` only selects the kernel machinery: "mo" (the default)
    // prepares the per-sector MO-basis RI bundles (`ri_terms`) + the
    // spin-resolved fxc table (`fxc_u`) and the sector-generic per-vector
    // matvecs serve both references; "ao" uses the sector-generic
    // AO machinery. The unrestricted response has one single, spin-coupled
    // channel (the Coulomb kernel couples the alpha and beta blocks) —
    // `tddft_spin` has no singlet/triplet meaning here and an explicit
    // non-default value is rejected below.
    if scf.scftype == SCFType::ROHF {
        return Err("ROHF-TDDFT is not yet supported. Use spin_polarization=true \
                    (UKS) for open-shell excited states."
            .to_string());
    }
    let is_u = scf.scftype == SCFType::UHF;
    if is_u {
        // `tddft_spin` selects a spin-adapted channel, which only exists for a
        // restricted reference. Reject an explicit non-default value instead
        // of silently reinterpreting it (previously "triplet" silently switched
        // the Coulomb coupling off, and "both" ran the same operator twice).
        if let Some(v) = tddft_ctrl.tddft_spin.as_deref() {
            if v != "singlet" {
                return Err(format!(
                    "tddft_spin = \"{}\" is not applicable to unrestricted TDDFT \
                     (spin_polarization = true).  Unrestricted TDDFT has a single, spin-coupled \
                     response channel (the Coulomb kernel couples the alpha and beta blocks); \
                     there is no spin-adapted singlet/triplet channel to select. \
                     Remove tddft_spin from the input.",
                    v
                ));
            }
        }
        // PySOC export needs both singlet and triplet transition amplitudes,
        // which only the restricted path can supply.
        if tddft_ctrl.pysoc {
            return Err("pysoc = true requires the restricted path (tddft_spin = \"both\") \
                        and is not available for an unrestricted reference \
                        (spin_polarization = true)."
                .to_string());
        }
        if is_ao && tddft_ctrl.tddft_fxc_driver == "mo" {
            return Err("tddft_fxc_driver=\"mo\" is not supported for unrestricted (UKS) TDDFT \
                        in AO mode; use \"semitrans\" or \"dm\".".to_string());
        }
    }

    // ── Restricted (spin-adapted) path ──
    // Here `tddft_spin` is a genuine physical label: the closed-shell reference
    // can be rotated into the singlet/triplet subspaces, which turns the
    // spin-independent Coulomb kernel into the familiar factors 2 (singlet) and
    // 0 (triplet).
    let tddft_spin = tddft_ctrl.restricted_spin().to_string();
    let xlet = if tddft_spin == "singlet" { 'S' } else if tddft_spin == "triplet" { 'T' } else { 'R' };

    if is_u {
        // (Reached only with tddft_mode = "ao".)
        if tddft_ctrl.tddft_feast_solver {
            return Err("FEAST solver is not supported for unrestricted (UKS) TDDFT.".to_string());
        }
        if tddft_ctrl.tddft_fxc_driver == "mo" {
            return Err("tddft_fxc_driver=\"mo\" is not supported for unrestricted (UKS) TDDFT; \
                        use \"semitrans\" or \"dm\".".to_string());
        }
        if tddft_spin != "singlet" {
            // The default is "singlet", so an explicit triplet/both setting
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
    // kernel combination (CPL, 256, 454).  `tddft_spin = "both"` includes a
    // triplet half, so it is restricted identically.
    if !is_ao && (tddft_spin == "triplet" || tddft_spin == "both") {
        return Err("Triplet TDDFT is not yet supported in MO mode \
                    (the fxc kernel is hardcoded singlet); use tddft_mode=\"ao\".".to_string());
    }

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
        println!("Spin: {}", if tddft_spin == "both" {
            "Singlet + Triplet"
        } else if xlet == 'S' {
            "Singlet"
        } else if xlet == 'T' {
            "Triplet"
        } else {
            ""
        });
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
    // Data preparation covers all four (mode x reference) cells: AO is
    // sector-generic (both references); MO mode builds the per-sector RI
    // bundles (`ri_terms`) and the restricted/spin-resolved fxc tables in one
    // `prepare_mo_data`.
    let data: std::cell::RefCell<TDDFTData> = std::cell::RefCell::new(
        if is_ao {
            println!("Reftype: {}", if is_u { "UKS" } else { "RKS" });
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

    // ═══ Step 8+9: Solve and print one spin channel ═══
    // Extracted so that `tddft_spin = "both"` can solve singlet and triplet
    // in a single pass over the same prepared TDDFTData (the fxc kernel and
    // RI tensors are spin-independent; the spin enters per matvec via `xlet`).
    let scf_ref: &SCF = scf;
    let run_spin = |xlet: char, initial_guess: &MatrixFull<f64>|
        -> (Vec<(f64, Vec<f64>)>, Vec<f64>, Vec<f64>) {
        // ── Step 8: Diagnostic: check A matrix symmetry for first few columns ──
        // MO-mode matvec closures (used by diagnostic + MO solver dispatch; the
        // dense path and AO mode use the dedicated builders / batched matvecs).
        // Sector-generic: one call serves restricted and unrestricted data.
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
        if scf_ref.mol.ctrl.print_level > 1 {
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

        // ── Step 8: Call solver (full diag, Davidson, or FEAST) ──
        let eigenpairs = if tddft_ctrl.tddft_feast_solver && !is_u {
            // ── FEAST solver path (restricted MO mode only) ──
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
                    scf_ref, &*data.borrow(), &hdiag, xlet,
                    eigenrange_min, eigenrange_max,
                    m_expected, max_feast_iter, tol_feast,
                    gmres_restart, gmres_max_iter, gmres_tol,
                    init_guess_type, gaussian_width_factor,
                )
            } else {
                feast_solver::feast_solve_tddft_lr(
                    scf_ref, &*data.borrow(), &hdiag, xlet,
                    eigenrange_min, eigenrange_max,
                    m_expected, max_feast_iter, tol_feast,
                    gmres_restart, gmres_max_iter, gmres_tol,
                    cg_max_iter, cg_tol,
                    init_guess_type, gaussian_width_factor,
                )
            }
        } else {
            if tddft_ctrl.tddft_feast_solver {
                // FEAST for unrestricted TDDFT is not implemented; fall back
                // to Davidson with a warning instead of silently producing
                // wrong results.
                eprintln!("Warning: FEAST solver for unrestricted TDDFT is not implemented; using Davidson.");
            }
            let is_mo_u = is_u && !is_ao;
            if is_mo_u && dim <= 15 && is_tda {
                // MO-U small TDA: exact dense diagonalisation is robust.
                let mut a_mat = vec![0.0; dim * dim];
                for col in 0..dim {
                    let mut e_col = vec![0.0; dim];
                    e_col[col] = 1.0;
                    let a_col = a_apply(&e_col);
                    for row in 0..dim {
                        a_mat[row + col * dim] = a_col[row];
                    }
                }
                let a = MatrixFull::from_vec([dim, dim], a_mat).unwrap();
                let (eigvecs_opt, eigvals, _info) = rest_tensors::matrix::matrix_blas_lapack::_dsyev(&a, 'V');
                let eigvecs = eigvecs_opt.expect("dsyev failed");
                let mut pairs: Vec<(f64, Vec<f64>)> = eigvals.iter()
                    .zip(eigvecs.iter_columns_full())
                    .map(|(e, v)| (*e, v.to_vec()))
                    .collect();
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                pairs.truncate(nroots.min(dim));
                pairs
            } else if is_mo_u && !is_tda && dim <= 80 {
                // MO-U small/medium full LR: build the explicit non-Hermitian
                // matrix [A  B; -B -A] and diagonalise it. This avoids slow or
                // oscillating Davidson iterations for small unrestricted systems.
                println!("Building unrestricted full LR matrix ({} x {})...", 2 * dim, 2 * dim);
                let mut a_mat = vec![0.0; dim * dim];
                let mut b_mat = vec![0.0; dim * dim];
                for col in 0..dim {
                    let mut e_col = vec![0.0; dim];
                    e_col[col] = 1.0;
                    let a_col = a_apply(&e_col);
                    let b_col = b_apply(&e_col);
                    for row in 0..dim {
                        a_mat[row + col * dim] = a_col[row];
                        b_mat[row + col * dim] = b_col[row];
                    }
                }
                let n2 = 2 * dim;
                let mut h = MatrixFull::new([n2, n2], 0.0);
                for j in 0..dim {
                    for i in 0..dim {
                        let av = a_mat[i + j * dim];
                        let bv = b_mat[i + j * dim];
                        h[[i, j]] = av;
                        h[[dim + i, j]] = -bv;
                        h[[i, dim + j]] = bv;
                        h[[dim + i, dim + j]] = -av;
                    }
                }
                let (_, wr, wi, _vl, vr, _info) =
                    rest_tensors::matrix::matrix_blas_lapack::_dgeev(&h, 'N', 'V');
                let mut pairs: Vec<(f64, Vec<f64>)> = wr.iter()
                    .zip(wi.iter().zip(vr.iter_columns_full()))
                    .filter(|(wr_i, (wi_i, _))| **wr_i > 1e-8 && wi_i.abs() < 1e-6)
                    .map(|(wr_i, (_, v))| (*wr_i, v.to_vec()))
                    .collect();
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                pairs.truncate(nroots.min(dim));
                pairs
            } else if is_mo_u {
                // MO-U iterative: per-vector Davidson over the concatenated α;β vector.
                if is_tda {
                    println!("Solving unrestricted TDA eigenvalue problem...");
                    davidson_solver::tda_davidson_solver(
                        a_apply, nroots, &hdiag, initial_guess.clone(), &davidson_cfg,
                    )
                } else {
                    println!("Solving unrestricted full linear response eigenvalue problem...");
                    davidson_solver::lr_davidson_solver(
                        a_apply, b_apply, nroots, &hdiag, initial_guess.clone(), &davidson_cfg,
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
                            initial_guess.clone(),
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
                            initial_guess.clone(),
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
                            initial_guess.clone(),
                            &davidson_cfg,
                        )
                    } else {
                        println!("Solving full linear response eigenvalue problem...");
                        davidson_solver::lr_davidson_solver(|z: &Vec<f64>| a_apply(z),
                            |z: &Vec<f64>| b_apply(z),
                            nroots,
                            &hdiag,
                            initial_guess.clone(),
                            &davidson_cfg,
                        )
                    }
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
        let n_print = n_found.min(30);
        // Unrestricted (AO-mode fall-through): concatenated [alpha; beta]
        // dipole matrix; PySCF uhf.py post-processing convention. Restricted:
        // single-sector matrix and the closed-shell BSE conventions.
        let dipole_matrix = if let Some(sec) = &sectors_u {
            compute_tddft_dipole_matrix_u(scf_ref, sec)
        } else {
            compute_tddft_dipole_matrix(scf_ref, start_mo, occ_size, vir_size, homo, lumo)
        };
        let singlet_triplet = if is_u {
            "Unrestricted"
        } else if xlet == 'S' {
            "Singlet"
        } else if xlet == 'T' {
            "Triplet"
        } else {
            ""
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
                normalize_u(vector, is_tda)
            } else {
                dipoles::normalize(vector, is_tda)
            };
            let dipole_sq = if is_u {
                transition_dipole_square_u(&dipole_matrix, &norm_vec, is_tda)
            } else {
                dipoles::transition_dipole_square(&dipole_matrix, &norm_vec, is_tda)
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

        (eigenpairs, td_energies, td_osc)
    };

    // ═══ Step 10: Dispatch on the requested spin channel(s) ═══
    if tddft_spin == "both" {
        // ── Both singlet and triplet: singlet first, then triplet ──
        println!("\n--- Singlet ---");
        let (eigenpairs_singlet, mut energies, mut osc) = run_spin('S', &initial_guess);

        println!("\n--- Triplet ---");
        let (eigenpairs_triplet, energies_t, osc_t) = run_spin('T', &initial_guess);

        // JSON order: singlet roots first, then triplet roots.
        energies.extend(energies_t);
        osc.extend(osc_t);

        // PySOC export
        if tddft_ctrl.pysoc {
            pysoc_export::export_pysoc_json(
                scf_ref,
                &eigenpairs_singlet,
                &eigenpairs_triplet,
                is_tda,
                "rest_pysoc_export.json",
                "TDDFT",
                start_mo, num_state, occ_size, vir_size,
            );
        }

        println!("TDDFT (both spins) calculation completed successfully.");
        println!("TDDFT calculation completed successfully.");
        Ok(TddftOutput { energies, osc, excitations: eigenpairs_singlet })
    } else {
        // ── Single spin ──
        let (eigenpairs, energies, osc) = run_spin(xlet, &initial_guess);
        println!("The first excitation obtained by TDDFT is {}", energies[0]);
        println!("TDDFT calculation completed successfully.");
        Ok(TddftOutput { energies, osc, excitations: eigenpairs })
    }
}

