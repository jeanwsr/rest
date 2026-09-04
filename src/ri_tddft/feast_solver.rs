// ============================================================================
// TDDFT-specific FEAST solver wrappers
//
// Provides TDA and full-LR entry points that build matvec closures using
// TDDFT's a_matvec / b_matvec functions and pass them to the generic FEAST
// algorithm in ri_bse::feast_solver.
// ============================================================================

use rest_tensors::MatrixFull;
use crate::scf_io::SCF;
use crate::dft::num_int::FXCMatvecData;
use crate::ri_tddft::matvec;

/// TDA branch: solve A*x = omega*x using FEAST.
///
/// All TDDFT data (fxc_data, RI matrices, hdiag, etc.) is pre-prepared by
/// the caller (tddft_main) and passed by reference.
///
/// Returns eigenpairs: (eigenvalue, eigenvector) sorted by eigenvalue.
pub fn feast_solve_tddft_tda(
    scf: &SCF,
    data: &crate::ri_tddft::TDDFTData,
    hdiag: &Vec<f64>,
    xlet: char,
    // FEAST parameters
    eigenrange_min: f64,
    eigenrange_max: f64,
    m_expected: usize,
    max_feast_iter: usize,
    tol_feast: f64,
    gmres_restart: usize,
    gmres_max_iter: usize,
    gmres_tol: f64,
    init_guess_type: &str,
    gaussian_width_factor: f64,
) -> Vec<(f64, Vec<f64>)> {
    let dim = hdiag.len();
    if dim == 0 {
        return vec![];
    }

    // TDA eigenvalue problem is A*x = omega*x (eigenvalue = omega directly).
    // The FEAST search window is in units of omega (Ha), not omega^2.
    let eig_min = eigenrange_min;
    let eig_max = eigenrange_max;

    // A-block matvec: A(z) = Coulomb + exchange + fxc
    let feast_a_matvec = |z: &Vec<f64>| -> Vec<f64> {
        matvec::a_matvec(scf, data, z, xlet)
    };

    // B-matrix = I (standard eigenvalue problem)
    let feast_b_matvec = |z: &Vec<f64>| -> Vec<f64> { z.clone() };

    let mut result: Vec<(f64, Vec<f64>)> = crate::ri_bse::feast_solver::feast(
        dim,
        &feast_a_matvec,
        &feast_b_matvec,
        None,                              // gmres_a_mul: use a_matvec
        None,                              // gmres_b_mul: use identity
        eig_min,
        eig_max,
        m_expected,
        max_feast_iter,
        tol_feast,
        gmres_restart,
        gmres_max_iter,
        gmres_tol,
        Some(hdiag),                       // GMRES diagonal preconditioner
        init_guess_type,
        Some(hdiag),                       // init diagonal for Gaussian guess
        gaussian_width_factor,
        true, // use_contour_rayon (not configurable from TDDFT)
        None, // custom_init_vectors
        None, None, "diagonal", None, 0.0001, 0, 0, // precond params
    );

    // Filter out spurious eigenvalues near zero (produced by the generic
    // feast() function when padding the subspace to maintain dimensions).
    result.retain(|(val, _)| *val > tol_feast);
    result
}

/// Full LR (non-TDA) branch: solve (A-B)(A+B)*(X+Y) = omega^2*(X+Y) using FEAST.
///
/// All TDDFT data is pre-prepared by the caller and passed by reference.
///
/// Returns eigenpairs: (excitation_energy, eigenvector_X) sorted by energy.
pub fn feast_solve_tddft_lr(
    scf: &SCF,
    data: &crate::ri_tddft::TDDFTData,
    hdiag: &Vec<f64>,
    xlet: char,
    // FEAST parameters
    eigenrange_min: f64,
    eigenrange_max: f64,
    m_expected: usize,
    max_feast_iter: usize,
    tol_feast: f64,
    gmres_restart: usize,
    gmres_max_iter: usize,
    gmres_tol: f64,
    cg_max_iter: usize,
    cg_tol: f64,
    init_guess_type: &str,
    gaussian_width_factor: f64,
) -> Vec<(f64, Vec<f64>)> {
    let dim = hdiag.len();
    if dim == 0 {
        return vec![];
    }

    let eig_min = eigenrange_min * eigenrange_min;
    let eig_max = eigenrange_max * eigenrange_max;

    // Squared KS gaps for the GMRES preconditioner
    let hdiag_sq: Vec<f64> = hdiag.iter().map(|&d| d * d).collect();

    // --- Closures for the FEAST algorithm ---

    // (A-B) matvec: used as the "A" operator in the transformed EVP
    let feast_a_matvec = |z: &Vec<f64>| -> Vec<f64> {
        let a = matvec::a_matvec(scf, data, z, xlet);
        let b = matvec::b_matvec(scf, data, z, xlet);
        a.into_iter().zip(b.into_iter()).map(|(a, b)| a - b).collect()
    };

    // (A+B)^{-1} matvec: used as the "B" operator in the transformed EVP.
    // In the transformed problem, B is replaced by (A+B)^{-1}, so we solve
    // (A+B) * y = z via CG.
    let feast_b_matvec = |z: &Vec<f64>| -> Vec<f64> {
        let apb_matvec = |p: &Vec<f64>| -> Vec<f64> {
            let a = matvec::a_matvec(scf, data, p, xlet);
            let b = matvec::b_matvec(scf, data, p, xlet);
            a.into_iter().zip(b.into_iter()).map(|(a, b)| a + b).collect()
        };
        crate::ri_bse::feast_solver::cg(
            &apb_matvec,
            z,
            cg_max_iter,
            cg_tol,
            Some(hdiag),
        )
    };

    // GMRES operator: (A+B)*(A-B) — this is the full operator that the
    // linear system in the FEAST contour integration acts on.
    let gmres_a_mul = |z: &Vec<f64>| -> Vec<f64> {
        // (A-B)*z
        let amb = {
            let a = matvec::a_matvec(scf, data, z, xlet);
            let b = matvec::b_matvec(scf, data, z, xlet);
            a.into_iter().zip(b.into_iter()).map(|(a, b)| a - b).collect::<Vec<f64>>()
        };
        // (A+B)*(A-B)*z = (A+B)*amb
        let a_amb = matvec::a_matvec(scf, data, &amb, xlet);
        let b_amb = matvec::b_matvec(scf, data, &amb, xlet);
        a_amb.into_iter().zip(b_amb.into_iter()).map(|(a, b)| a + b).collect()
    };

    // GMRES B-operator = I
    let gmres_b_mul = |z: &Vec<f64>| -> Vec<f64> { z.clone() };

    // Run FEAST — returns (omega^2, X+Y) pairs
    let eigenpairs_xpy = crate::ri_bse::feast_solver::feast(
        dim,
        &feast_a_matvec,
        &feast_b_matvec,
        Some(&gmres_a_mul as &(dyn Fn(&Vec<f64>) -> Vec<f64> + Sync)),
        Some(&gmres_b_mul as &(dyn Fn(&Vec<f64>) -> Vec<f64> + Sync)),
        eig_min,
        eig_max,
        m_expected,
        max_feast_iter,
        tol_feast,
        gmres_restart,
        gmres_max_iter,
        gmres_tol,
        Some(&hdiag_sq),
        init_guess_type,
        Some(hdiag),
        gaussian_width_factor,
        true, // use_contour_rayon (not configurable from TDDFT)
        None, // custom_init_vectors
        None, None, "diagonal", None, 0.0001, 0, 0, // precond params
    );

    // Post-process: convert (omega^2, X+Y) -> (omega, X)
    // Filter out spurious eigenvalues (from subspace padding) and NaN-prone entries.
    eigenpairs_xpy
        .into_iter()
        .filter(|(omega2, _)| *omega2 > tol_feast)
        .map(|(omega2, xpy)| {
            let omega = omega2.sqrt();
            let xmy = feast_a_matvec(&xpy); // = omega * (X-Y)
            // X = (X+Y) + (X-Y)  where X-Y = xmy/omega
            let x: Vec<f64> = xmy
                .into_iter()
                .zip(xpy.into_iter())
                .map(|(xmy_k, xpy_k)| (xmy_k / omega) + xpy_k)
                .collect();
            (omega, x)
        })
        .collect()
}
