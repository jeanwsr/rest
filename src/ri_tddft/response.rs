/// Damped (frequency-domain) TDDFT linear response solver
///
/// Implements the damped TDDFT response calculation, solving the 4-component
/// non-Hermitian linear system:
///
///   [A - ω,   B,    -γ,    0 ] [rp]   [μ_z]
///   [B,     A+ω,     0,    γ] [rm]   [μ_z]
///   [γ,      0,    A-ω,   B] [ip] = [ 0 ]
///   [0,     -γ,     B,   A+ω] [im]   [ 0 ]
///
/// where A and B are the TDDFT A and B matrix blocks (including fxc kernel
/// and exchange contributions, using KS orbital energy differences), ω is
/// the external field frequency, and γ is the lifetime broadening.
///
/// Reference: ri_bse::damped implementation adapted for TDDFT operators.

use itertools::Itertools;
use std::time::Instant;
use std::fs::OpenOptions;
use std::io::Write;

use rest_tensors::MatrixFull;
use rest_tensors::matrix::matrix_blas_lapack::{_dsolve, _dgemm_full};

use crate::ri_bse::damped::{
    fourvec_dot_product, fourvec_scaled_add,
    klopper_subspace_solver, fourvec_gmres, poples_numerical_trick,
    obtain_mu_ia_z, eval_ao_on_grids,
};
use crate::ri_bse::davidson_solver::vector_scaled_add;
use crate::ri_bse::dipoles;
use crate::dft::num_int::{FXCMatvecData, prepare_fxc_data, set_fxc_use_optimized};
use crate::ri_tddft::matvec::{self, a_matvec, b_matvec};
use crate::ri_tddft::utils::{tddft_occupation_parameters, tddft_get_submatrix};
use crate::scf_io::SCF;

// ========================================================================
// Helper functions (TDDFT-specific, using KS energies instead of QP energies)
// ========================================================================

/// Build energy diagonal from KS orbital energies for TDDFT
///
/// Returns vector of length occ_size * vir_size containing (ε_a - ε_i)
/// for each occupied-virtual pair, using KS eigenvalues (not QP energies).
pub fn build_ks_energy_diag(scf: &SCF, occ_size: usize, vir_size: usize) -> Vec<f64> {
    let (start_mo, _num_state, _occ_size, _vir_size, _homo, lumo) =
        tddft_occupation_parameters(scf);
    let ks = &scf.eigenvalues[0];
    let mut diag = Vec::with_capacity(occ_size * vir_size);
    for a in 0..vir_size {
        for i in 0..occ_size {
            diag.push(ks[lumo + a] - ks[start_mo + i]);
        }
    }
    diag
}

/// Compute z-component transition dipole vector in MO basis (TDDFT version)
///
/// Accounts for frozen core orbitals via start_mo and lumo offsets.
/// Returns vector of length occ_size * vir_size with μ_z^{ia} for each (i,a).
pub fn compute_mu_z_vec_tddft(scf: &SCF) -> Vec<f64> {
    let (start_mo, _num_state, occ_size, vir_size, _homo, lumo) =
        tddft_occupation_parameters(scf);
    let ao_dip = dipoles::obtain_ao_dips(scf, None);
    let eigenvectors = scf.eigenvectors[0].clone();
    let dim = occ_size * vir_size;
    let mut mu_z_vec = vec![0.0; dim];
    for i in 0..occ_size {
        for a in 0..vir_size {
            let idx = i + a * occ_size;
            mu_z_vec[idx] = obtain_mu_ia_z(&eigenvectors, &ao_dip, start_mo + i, lumo + a);
        }
    }
    mu_z_vec
}

/// Non-interacting response vector (initial guess) using KS energy gaps
///
/// Constructs the 4-component non-interacting response vector:
///   p0_rp[i,a] = (Δ - ω) / ((ω - Δ)² + γ²) * μ_z
///   p0_rm[i,a] = (ω + Δ) / ((ω + Δ)² + γ²) * μ_z
///   p0_ip[i,a] = -γ / ((ω - Δ)² + γ²) * μ_z
///   p0_im[i,a] =  γ / ((ω + Δ)² + γ²) * μ_z
/// where Δ = ε_a - ε_i (KS gap).
pub fn prepare_p0_r_i_tddft(
    mu_z_vec: &[f64],
    ks_energies: &[f64],
    start_mo: usize,
    lumo: usize,
    omega: f64,
    gamma: f64,
    occ_size: usize,
    vir_size: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let dim = occ_size * vir_size;
    let mut p0rp = vec![0.0; dim];
    let mut p0rm = vec![0.0; dim];
    let mut p0ip = vec![0.0; dim];
    let mut p0im = vec![0.0; dim];
    for i in 0..occ_size {
        for a in 0..vir_size {
            let idx = i + a * occ_size;
            let mu_ia_z = mu_z_vec[idx];
            let gap = ks_energies[lumo + a] - ks_energies[start_mo + i];
            let denom_p = (omega - gap).powi(2) + gamma.powi(2);
            let denom_m = (omega + gap).powi(2) + gamma.powi(2);
            p0rp[idx] = (gap - omega) / denom_p * mu_ia_z;
            p0rm[idx] = (omega + gap) / denom_m * mu_ia_z;
            p0ip[idx] = -gamma / denom_p * mu_ia_z;
            p0im[idx] = gamma / denom_m * mu_ia_z;
        }
    }
    (p0rp, p0rm, p0ip, p0im)
}

/// TDDFT pair (P, M) matrix-vector product
///
/// Returns (K_P, K_M) where:
///   K_P = A * P + B * M
///   K_M = B * P + A * M
/// using TDDFT A and B blocks (with fxc kernel and exchange).
fn pairvec_matvec_tddft(
    scf: &SCF,
    fxc_data: &FXCMatvecData,
    ri_ov: &MatrixFull<f64>,
    ri_oo_exch: &MatrixFull<f64>,
    ri_vv_exch: &MatrixFull<f64>,
    ri_ov_exch: &MatrixFull<f64>,
    p_vec: &[f64],
    m_vec: &[f64],
    xlet: char,
    alpha_hybrid: f64,
) -> (Vec<f64>, Vec<f64>) {
    let p_owned = p_vec.to_vec();
    let m_owned = m_vec.to_vec();
    let ap = a_matvec(scf, fxc_data, ri_ov, ri_oo_exch, ri_vv_exch, &p_owned, xlet, alpha_hybrid);
    let am = a_matvec(scf, fxc_data, ri_ov, ri_oo_exch, ri_vv_exch, &m_owned, xlet, alpha_hybrid);
    let bp = b_matvec(scf, fxc_data, ri_ov, ri_ov_exch, &p_owned, xlet, alpha_hybrid);
    let bm = b_matvec(scf, fxc_data, ri_ov, ri_ov_exch, &m_owned, xlet, alpha_hybrid);
    (
        ap.iter().zip(bm.iter()).map(|(a, b)| a + b).collect(),
        bp.iter().zip(am.iter()).map(|(a, b)| a + b).collect(),
    )
}

/// Update W vectors with diagonal correction using KS energy gaps
///
/// TDDFT version of update_w_vecs from ri_bse::damped, using KS energy gaps
/// instead of QP energies. Applies the T(z) = A_4c_diag^{-1} * Interaction(z)
/// operator needed by the Pople subspace solver.
fn update_w_vecs_tddft<F1>(
    rp: &[f64],
    rm: &[f64],
    ip: &[f64],
    im: &[f64],
    pair_matvec: F1,
    ks_energies: &[f64],
    start_mo: usize,
    lumo: usize,
    omega: f64,
    gamma: f64,
    occ_size: usize,
    vir_size: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)
where
    F1: Fn(&(Vec<f64>, Vec<f64>)) -> (Vec<f64>, Vec<f64>),
{
    let preal = (rp.to_vec(), rm.to_vec());
    let pimag = (ip.to_vec(), im.to_vec());
    let (krp, krm) = pair_matvec(&preal);
    let (kip, kim) = pair_matvec(&pimag);

    let dim = occ_size * vir_size;
    let mut wrp = vec![0.0; dim];
    let mut wrm = vec![0.0; dim];
    let mut wip = vec![0.0; dim];
    let mut wim = vec![0.0; dim];

    for i in 0..occ_size {
        for a in 0..vir_size {
            let idx = i + a * occ_size;
            let gap = ks_energies[lumo + a] - ks_energies[start_mo + i];
            let prefactor_p = 1.0 / ((omega - gap).powi(2) + gamma.powi(2));
            let prefactor_m = 1.0 / ((omega + gap).powi(2) + gamma.powi(2));

            // T(z) = A_4c_diag^{-1} * Interaction
            wrp[idx] = -prefactor_p * ((omega - gap) * krp[idx] - gamma * kip[idx]);
            wip[idx] = -prefactor_p * ((omega - gap) * kip[idx] + gamma * krp[idx]);
            wrm[idx] = prefactor_m * ((omega + gap) * krm[idx] - gamma * kim[idx]);
            wim[idx] = prefactor_m * ((omega + gap) * kim[idx] + gamma * krm[idx]);

            // Diagonal correction (cancels D contribution, forces T(z_ni) = 0)
            wrp[idx] += gap * prefactor_p * ((omega - gap) * rp[idx] - gamma * ip[idx]);
            wrm[idx] -= gap * prefactor_m * ((omega + gap) * rm[idx] - gamma * im[idx]);
            wip[idx] += gap * prefactor_p * ((omega - gap) * ip[idx] + gamma * rp[idx]);
            wim[idx] -= gap * prefactor_m * ((omega + gap) * im[idx] + gamma * rm[idx]);
        }
    }
    (wrp, wrm, wip, wim)
}

// ========================================================================
// Result printing
// ========================================================================

/// Print damped TDDFT results: polarizability and induced density norm
fn print_damped_tddft_results(
    solution: &(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>),
    mu_z_vec: &[f64],
    occ_size: usize,
    vir_size: usize,
    omega: f64,
    gamma: f64,
) {
    let dim = occ_size * vir_size;
    let density_real: Vec<f64> = (0..dim).map(|ia| solution.0[ia] + solution.1[ia]).collect();
    let density_imag: Vec<f64> = (0..dim).map(|ia| solution.2[ia] + solution.3[ia]).collect();

    // Frequency-dependent polarizability α_zz(ω) = Σ_{ia} μ_z^{ia} * ΔP_{ia}
    let pol_re: f64 = density_real.iter().zip(mu_z_vec.iter()).map(|(d, m)| d * m).sum();
    let pol_im: f64 = density_imag.iter().zip(mu_z_vec.iter()).map(|(d, m)| d * m).sum();

    println!("  --- Damped TDDFT Results ---");
    println!("    Frequency ω = {:.8} Ha ({:.4} eV)", omega, omega * 27.2114);
    println!("    Lifetime  γ = {:.8} Ha", gamma);
    println!("    Re[α_zz(ω)] = {:.12e} a.u.", pol_re);
    println!("    Im[α_zz(ω)] = {:.12e} a.u.", pol_im);
    println!("    |Re[ΔP]|    = {:.12e}",
             density_real.iter().map(|x| x * x).sum::<f64>().sqrt());
    println!("    |Im[ΔP]|    = {:.12e}",
             density_imag.iter().map(|x| x * x).sum::<f64>().sqrt());
}

/// Export polarized density to AO_Polarized_Density_Matrix.txt and
/// Polarized_Density_Grids.txt (mirrors ri_bse::damped::export_density).
fn export_density_tddft(
    scf: &SCF,
    solution: &(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>),
    start_mo: usize,
    occ_size: usize,
    vir_size: usize,
    lumo: usize,
    num_state: usize,
    grids: &[[f64; 3]],
    grid_x_start: f64, grid_x_end: f64, grid_x_points: usize,
    grid_y_start: f64, grid_y_end: f64, grid_y_points: usize,
    grid_z_start: f64, grid_z_end: f64, grid_z_points: usize,
) {
    let density_real: Vec<f64> = (0..occ_size * vir_size)
        .map(|ia| solution.0[ia] + solution.1[ia]).collect();
    let polarized_density_matrix =
        MatrixFull::from_vec([occ_size, vir_size], density_real).unwrap();

    // Build AO polarized density matrix: P_AO = C_occ * P_MO * C_vir^T
    let occ_slice_start = start_mo * num_state;
    let vir_slice_start = lumo * num_state;
    let occ_eigenvecs = MatrixFull::from_vec([num_state, occ_size],
        scf.eigenvectors[0].data[occ_slice_start..occ_slice_start + occ_size * num_state].to_vec()).unwrap();
    let vir_eigenvecs = MatrixFull::from_vec([num_state, vir_size],
        scf.eigenvectors[0].data[vir_slice_start..vir_slice_start + vir_size * num_state].to_vec()).unwrap();
    let mut first_prod = MatrixFull::new([num_state, vir_size], 0.0);
    let mut ao_polarized_density_matrix = MatrixFull::new([num_state, num_state], 0.0);
    _dgemm_full(&occ_eigenvecs, 'N', &polarized_density_matrix, 'N', &mut first_prod, 1.0, 0.0);
    _dgemm_full(&first_prod, 'N', &vir_eigenvecs, 'T', &mut ao_polarized_density_matrix, 1.0, 0.0);

    // Write AO matrix
    let mut file = OpenOptions::new().write(true).create(true).truncate(true)
        .open("AO_Polarized_Density_Matrix.txt").expect("open failure");
    writeln!(file, "AO Basis Polarized Density Matrix (damped TDDFT):\nSize:{:?}\nData(Column Major){:#?}",
             ao_polarized_density_matrix.size, ao_polarized_density_matrix.data).expect("write failure");

    // Evaluate AO on grids and compute grid density
    let grid_val = eval_ao_on_grids(&scf.mol, grids);
    let mut first_prod = MatrixFull::new([num_state, grids.len()], 0.0);
    _dgemm_full(&ao_polarized_density_matrix, 'N', &grid_val, 'N', &mut first_prod, 1.0, 0.0);
    let grid_data: Vec<f64> = first_prod.iter_columns_full().zip(grid_val.iter_columns_full()).map(|(fp, gval)| {
        let fp_vec = fp.to_vec();
        let gval_vec = gval.to_vec();
        fp_vec.iter().zip(gval_vec.iter()).fold(0.0, |acc, (x, y)| acc + x * y)
    }).collect();

    // Write grid file
    let mut file = OpenOptions::new().write(true).create(true).truncate(true)
        .open("Polarized_Density_Grids.txt").expect("open failure");

    writeln!(file, "{}", "=".repeat(70)).expect("write failure");
    writeln!(file, "POLARIZED DENSITY ON SPATIAL GRIDS (damped TDDFT result)").expect("write failure");
    writeln!(file, "{}", "=".repeat(70)).expect("write failure");

    // Geometry
    writeln!(file, "\n{}", "-".repeat(70)).expect("write failure");
    writeln!(file, "MOLECULAR GEOMETRY").expect("write failure");
    writeln!(file, "{}", "-".repeat(70)).expect("write failure");
    writeln!(file, "{}", "=".repeat(70)).expect("write failure");
    writeln!(file, "{}", "=".repeat(70)).expect("write failure");
    let elem = &scf.mol.geom.elem;
    let position = scf.mol.geom.position.clone();
    for (i, e) in elem.iter().enumerate() {
        let pos = position.iter_columns_full().nth(i).unwrap();
        writeln!(file, "{:>2}  {:.8e}  {:.8e}  {:.8e}", e, pos[0], pos[1], pos[2]).expect("write failure");
    }

    // Grid info
    writeln!(file, "\n{}", "-".repeat(70)).expect("write failure");
    writeln!(file, "GRID SAMPLING INFORMATION").expect("write failure");
    writeln!(file, "{}", "-".repeat(70)).expect("write failure");
    writeln!(file, "{}", "=".repeat(70)).expect("write failure");
    writeln!(file, "X start: {:.8e}  X end: {:.8e}  X points: {:>6}", grid_x_start, grid_x_end, grid_x_points).expect("write failure");
    writeln!(file, "Y start: {:.8e}  Y end: {:.8e}  Y points: {:>6}", grid_y_start, grid_y_end, grid_y_points).expect("write failure");
    writeln!(file, "Z start: {:.8e}  Z end: {:.8e}  Z points: {:>6}", grid_z_start, grid_z_end, grid_z_points).expect("write failure");
    let x_step = if grid_x_points > 1 { (grid_x_end - grid_x_start) / (grid_x_points - 1) as f64 } else { 0.0 };
    let y_step = if grid_y_points > 1 { (grid_y_end - grid_y_start) / (grid_y_points - 1) as f64 } else { 0.0 };
    let z_step = if grid_z_points > 1 { (grid_z_end - grid_z_start) / (grid_z_points - 1) as f64 } else { 0.0 };
    writeln!(file, "X step: {:.8e}  Y step: {:.8e}  Z step: {:.8e}", x_step, y_step, z_step).expect("write failure");
    writeln!(file, "Total grid points: {}", grids.len()).expect("write failure");

    // Grid data
    writeln!(file, "\n{}", "-".repeat(70)).expect("write failure");
    writeln!(file, "GRID DENSITY DATA").expect("write failure");
    writeln!(file, "{}", "-".repeat(70)).expect("write failure");
    writeln!(file, "(OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z)").expect("write failure");
    writeln!(file, "{}", "=".repeat(70)).expect("write failure");
    writeln!(file, "{:>10}  {:>20}  {:>20}  {:>20}  {:>20}", "Index", "X", "Y", "Z", "Density").expect("write failure");
    writeln!(file, "{}", "-".repeat(70)).expect("write failure");
    for (i, g) in grids.iter().enumerate() {
        writeln!(file, "{:>10}  {:.8e}  {:.8e}  {:.8e}  {:.12e}", i, g[0], g[1], g[2], grid_data[i]).expect("write failure");
    }

    writeln!(file, "\n{}", "=".repeat(70)).expect("write failure");
    writeln!(file, "END OF FILE").expect("write failure");
    writeln!(file, "{}", "=".repeat(70)).expect("write failure");

    println!("  Density exported to AO_Polarized_Density_Matrix.txt and Polarized_Density_Grids.txt");
}

// ========================================================================
// Solver implementations
// ========================================================================

/// Pople-Krylov subspace solver for damped TDDFT
///
/// Uses poples_numerical_trick with TDDFT-specific pair matvec and
/// KS-energy-based diagonal correction.
fn damped_tddft_pople(
    scf: &SCF,
    fxc_data: &FXCMatvecData,
    ri_ov: &MatrixFull<f64>,
    ri_oo_exch: &MatrixFull<f64>,
    ri_vv_exch: &MatrixFull<f64>,
    ri_ov_exch: &MatrixFull<f64>,
    mu_z_vec: &[f64],
    ks_energies: &[f64],
    start_mo: usize,
    lumo: usize,
    occ_size: usize,
    vir_size: usize,
    omega: f64,
    gamma: f64,
    xlet: char,
    alpha_hybrid: f64,
    tol: f64,
    max_iter: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    println!("  Solver: Pople-Krylov subspace");

    let ks_diag = build_ks_energy_diag(scf, occ_size, vir_size);

    let p0 = prepare_p0_r_i_tddft(mu_z_vec, ks_energies, start_mo, lumo, omega, gamma, occ_size, vir_size);

    let wrapped_pair_matvec = |pairvec: &(Vec<f64>, Vec<f64>)| -> (Vec<f64>, Vec<f64>) {
        pairvec_matvec_tddft(scf, fxc_data, ri_ov, ri_oo_exch, ri_vv_exch, ri_ov_exch,
                             &pairvec.0, &pairvec.1, xlet, alpha_hybrid)
    };

    let wrapped_update_w =
        |u: &(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)| -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
            update_w_vecs_tddft(&u.0, &u.1, &u.2, &u.3,
                                |x| wrapped_pair_matvec(x),
                                ks_energies, start_mo, lumo, omega, gamma, occ_size, vir_size)
        };

    let diag_13: Vec<f64> = ks_diag.iter().map(|d| d - omega).collect();
    let diag_24: Vec<f64> = ks_diag.iter().map(|d| d + omega).collect();
    let precond = |z: &(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)| -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let apply_inv = |v: &[f64], d: &[f64]| -> Vec<f64> {
            v.iter().zip(d.iter()).map(|(vi, di)| {
                if di.abs() > 1e-8 { vi / di } else { 0.0 }
            }).collect()
        };
        (apply_inv(&z.0, &diag_13), apply_inv(&z.1, &diag_24),
         apply_inv(&z.2, &diag_13), apply_inv(&z.3, &diag_24))
    };

    let start = Instant::now();
    let solution = poples_numerical_trick(&p0, |z| wrapped_update_w(z), |z| precond(z), tol, max_iter);
    let elapsed = start.elapsed();
    println!("  Pople solver finished in {:?}", elapsed);

    print_damped_tddft_results(&solution, mu_z_vec, occ_size, vir_size, omega, gamma);
    solution
}

/// Build the TDDFT 4-component matrix-vector product closure (shared by GMRES and Klopper)
///
/// Returns a closure that computes:
///   A_4c * (rp, rm, ip, im) = (vec1, vec2, vec3, vec4)
/// with the full damped TDDFT linear response operator.
fn build_fourvec_matvec_tddft<'a>(
    scf: &'a SCF,
    fxc_data: &'a FXCMatvecData,
    ri_ov: &'a MatrixFull<f64>,
    ri_oo_exch: &'a MatrixFull<f64>,
    ri_vv_exch: &'a MatrixFull<f64>,
    ri_ov_exch: &'a MatrixFull<f64>,
    xlet: char,
    alpha_hybrid: f64,
    omega: f64,
    gamma: f64,
) -> impl Fn(&(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) + 'a {
    move |z: &(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)| {
        let a_z0 = a_matvec(scf, fxc_data, ri_ov, ri_oo_exch, ri_vv_exch, &z.0, xlet, alpha_hybrid);
        let a_z1 = a_matvec(scf, fxc_data, ri_ov, ri_oo_exch, ri_vv_exch, &z.1, xlet, alpha_hybrid);
        let a_z2 = a_matvec(scf, fxc_data, ri_ov, ri_oo_exch, ri_vv_exch, &z.2, xlet, alpha_hybrid);
        let a_z3 = a_matvec(scf, fxc_data, ri_ov, ri_oo_exch, ri_vv_exch, &z.3, xlet, alpha_hybrid);
        let b_z0 = b_matvec(scf, fxc_data, ri_ov, ri_ov_exch, &z.0, xlet, alpha_hybrid);
        let b_z1 = b_matvec(scf, fxc_data, ri_ov, ri_ov_exch, &z.1, xlet, alpha_hybrid);
        let b_z2 = b_matvec(scf, fxc_data, ri_ov, ri_ov_exch, &z.2, xlet, alpha_hybrid);
        let b_z3 = b_matvec(scf, fxc_data, ri_ov, ri_ov_exch, &z.3, xlet, alpha_hybrid);

        let vec1 = vector_scaled_add(
            &vector_scaled_add(&a_z0, 1.0, &b_z1, 1.0), 1.0,
            &vector_scaled_add(&z.0, -omega, &z.2, -gamma), 1.0);
        let vec2 = vector_scaled_add(
            &vector_scaled_add(&b_z0, 1.0, &a_z1, 1.0), 1.0,
            &vector_scaled_add(&z.1, omega, &z.3, gamma), 1.0);
        let vec3 = vector_scaled_add(
            &vector_scaled_add(&a_z2, 1.0, &b_z3, 1.0), 1.0,
            &vector_scaled_add(&z.0, gamma, &z.2, -omega), 1.0);
        let vec4 = vector_scaled_add(
            &vector_scaled_add(&b_z2, 1.0, &a_z3, 1.0), 1.0,
            &vector_scaled_add(&z.1, -gamma, &z.3, omega), 1.0);

        (vec1, vec2, vec3, vec4)
    }
}

/// Build diagonal preconditioner closure for damped TDDFT
fn build_precond_tddft(
    ks_diag: &[f64],
    omega: f64,
) -> impl Fn(&(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) + '_ {
    let diag_13: Vec<f64> = ks_diag.iter().map(|d| d - omega).collect();
    let diag_24: Vec<f64> = ks_diag.iter().map(|d| d + omega).collect();
    move |z: &(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)| {
        let apply_inv = |v: &[f64], d: &[f64]| -> Vec<f64> {
            v.iter().zip(d.iter()).map(|(vi, di)| {
                if di.abs() > 1e-8 { vi / di } else { 0.0 }
            }).collect()
        };
        (apply_inv(&z.0, &diag_13), apply_inv(&z.1, &diag_24),
         apply_inv(&z.2, &diag_13), apply_inv(&z.3, &diag_24))
    }
}

/// GMRES solver for damped TDDFT
fn damped_tddft_gmres(
    scf: &SCF,
    fxc_data: &FXCMatvecData,
    ri_ov: &MatrixFull<f64>,
    ri_oo_exch: &MatrixFull<f64>,
    ri_vv_exch: &MatrixFull<f64>,
    ri_ov_exch: &MatrixFull<f64>,
    mu_z_vec: &[f64],
    occ_size: usize,
    vir_size: usize,
    omega: f64,
    gamma: f64,
    xlet: char,
    alpha_hybrid: f64,
    tol: f64,
    max_iter: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    println!("  Solver: Restarted GMRES(30)");

    let ks_diag = build_ks_energy_diag(scf, occ_size, vir_size);
    let fourvec_matvec = build_fourvec_matvec_tddft(
        scf, fxc_data, ri_ov, ri_oo_exch, ri_vv_exch, ri_ov_exch,
        xlet, alpha_hybrid, omega, gamma,
    );
    let precond = build_precond_tddft(&ks_diag, omega);

    let dim = occ_size * vir_size;
    let mu_z = mu_z_vec.to_vec();
    let rhs = (mu_z.clone(), mu_z, vec![0.0; dim], vec![0.0; dim]);

    let start = Instant::now();
    let solution = fourvec_gmres(&fourvec_matvec, &precond, &rhs, tol, max_iter);
    let elapsed = start.elapsed();
    println!("  GMRES solver finished in {:?}", elapsed);

    print_damped_tddft_results(&solution, mu_z_vec, occ_size, vir_size, omega, gamma);
    solution
}

/// Klopper subspace solver for damped TDDFT
fn damped_tddft_klopper(
    scf: &SCF,
    fxc_data: &FXCMatvecData,
    ri_ov: &MatrixFull<f64>,
    ri_oo_exch: &MatrixFull<f64>,
    ri_vv_exch: &MatrixFull<f64>,
    ri_ov_exch: &MatrixFull<f64>,
    mu_z_vec: &[f64],
    ks_energies: &[f64],
    start_mo: usize,
    lumo: usize,
    occ_size: usize,
    vir_size: usize,
    omega: f64,
    gamma: f64,
    xlet: char,
    alpha_hybrid: f64,
    tol: f64,
    max_iter: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    println!("  Solver: Klopper subspace");

    let ks_diag = build_ks_energy_diag(scf, occ_size, vir_size);
    let fourvec_matvec = build_fourvec_matvec_tddft(
        scf, fxc_data, ri_ov, ri_oo_exch, ri_vv_exch, ri_ov_exch,
        xlet, alpha_hybrid, omega, gamma,
    );
    let precond = build_precond_tddft(&ks_diag, omega);

    let p0 = prepare_p0_r_i_tddft(mu_z_vec, ks_energies, start_mo, lumo, omega, gamma, occ_size, vir_size);
    let dim = occ_size * vir_size;
    let mu_z = mu_z_vec.to_vec();
    let rhs = (mu_z.clone(), mu_z, vec![0.0; dim], vec![0.0; dim]);

    let start = Instant::now();
    let solution = klopper_subspace_solver(&fourvec_matvec, &precond, &rhs, &p0, tol, max_iter);
    let elapsed = start.elapsed();
    println!("  Klopper solver finished in {:?}", elapsed);

    print_damped_tddft_results(&solution, mu_z_vec, occ_size, vir_size, omega, gamma);
    solution
}

/// Dense solver for damped TDDFT (small systems only)
fn damped_tddft_dense(
    scf: &SCF,
    fxc_data: &FXCMatvecData,
    ri_ov: &MatrixFull<f64>,
    ri_oo_exch: &MatrixFull<f64>,
    ri_vv_exch: &MatrixFull<f64>,
    ri_ov_exch: &MatrixFull<f64>,
    mu_z_vec: &[f64],
    occ_size: usize,
    vir_size: usize,
    omega: f64,
    gamma: f64,
    xlet: char,
    alpha_hybrid: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    println!("  Solver: Dense (LAPACK LU)");
    let dim = occ_size * vir_size;
    let n4 = 4 * dim;

    // Build A and B matrices column by column
    println!("  Building full A and B matrices ({} x {})...", dim, dim);
    let build_start = Instant::now();
    let mut a_mat = vec![0.0; dim * dim];
    let mut b_mat = vec![0.0; dim * dim];
    for col in 0..dim {
        let mut e_col = vec![0.0; dim];
        e_col[col] = 1.0;
        let a_col = a_matvec(scf, fxc_data, ri_ov, ri_oo_exch, ri_vv_exch, &e_col, xlet, alpha_hybrid);
        let b_col = b_matvec(scf, fxc_data, ri_ov, ri_ov_exch, &e_col, xlet, alpha_hybrid);
        for row in 0..dim {
            a_mat[row + col * dim] = a_col[row];
            b_mat[row + col * dim] = b_col[row];
        }
    }
    println!("  Matrix construction took {:?}", build_start.elapsed());

    // Construct 4-component linear system
    let prep_start = Instant::now();
    let mut a_4c = MatrixFull::new([n4, n4], 0.0);
    for i in 0..dim {
        for j in 0..dim {
            let av = a_mat[i + j * dim];
            let bv = b_mat[i + j * dim];
            // Real blocks (1,2)
            a_4c[[i, j]] = av;
            a_4c[[dim + i, j]] = bv;
            a_4c[[i, dim + j]] = bv;
            a_4c[[dim + i, dim + j]] = av;
            // Imaginary blocks (3,4)
            a_4c[[i + 2 * dim, j + 2 * dim]] = av;
            a_4c[[3 * dim + i, j + 2 * dim]] = bv;
            a_4c[[i + 2 * dim, 3 * dim + j]] = bv;
            a_4c[[3 * dim + i, 3 * dim + j]] = av;
        }
    }

    // Add frequency and lifetime shifts
    for idx in 0..dim {
        a_4c[[idx, idx]] -= omega;                          // rp block
        a_4c[[dim + idx, dim + idx]] += omega;              // rm block
        a_4c[[idx + 2 * dim, idx + 2 * dim]] -= omega;     // ip block
        a_4c[[3 * dim + idx, 3 * dim + idx]] += omega;     // im block
        // Lifetime coupling between real and imaginary blocks
        a_4c[[idx + 2 * dim, idx]] += gamma;
        a_4c[[3 * dim + idx, dim + idx]] -= gamma;
        a_4c[[idx, idx + 2 * dim]] -= gamma;
        a_4c[[dim + idx, 3 * dim + idx]] += gamma;
    }

    // RHS: (μ_z, μ_z, 0, 0)
    let mut rhs = vec![0.0; n4];
    for idx in 0..dim {
        rhs[idx] = mu_z_vec[idx];
        rhs[dim + idx] = mu_z_vec[idx];
    }
    println!("  Linear system preparation took {:?}", prep_start.elapsed());

    // Solve via LAPACK LU
    let solve_start = Instant::now();
    let result = _dsolve(&a_4c, &rhs).expect("Dense LU solve failed for damped TDDFT");
    println!("  Dense LU solve took {:?}", solve_start.elapsed());

    let solution = (
        result[0..dim].to_vec(),
        result[dim..2 * dim].to_vec(),
        result[2 * dim..3 * dim].to_vec(),
        result[3 * dim..4 * dim].to_vec(),
    );

    print_damped_tddft_results(&solution, mu_z_vec, occ_size, vir_size, omega, gamma);
    solution
}

// ========================================================================
// Public entry point
// ========================================================================

/// Main damped TDDFT entry point
///
/// Called from main_driver when `damped_tddft = true` in the `[tddft]` section.
/// Dispatches to the selected solver variant (pople, gmres, klopper, dense).
///
/// The damped TDDFT calculation solves the frequency-domain linear response:
///   (H_4c - ω - iγ) · Z(ω) = μ
/// and computes the frequency-dependent polarizability α_zz(ω).
pub fn damped_tddft(scf: &mut SCF) -> Result<(), String> {
    let tddft_ctrl = scf.mol.ctrl.tddft.clone()
        .ok_or_else(|| "TDDFT control parameters not set for damped TDDFT".to_string())?;

    if !tddft_ctrl.damped_tddft {
        return Ok(());
    }

    let omega = tddft_ctrl.external_field_freq;
    let gamma = tddft_ctrl.lifetime_gamma;
    let solver = tddft_ctrl.damped_tddft_solver.clone();
    let tol = tddft_ctrl.damped_tddft_tol;
    let max_iter = tddft_ctrl.damped_tddft_max_iter;
    let xlet = if tddft_ctrl.tddft_spin == "singlet" { 'S' }
               else if tddft_ctrl.tddft_spin == "triplet" { 'T' }
               else { 'R' };

    // Enable optimised (rayon-parallel) fxc kernel if requested
    set_fxc_use_optimized(tddft_ctrl.tddft_use_optimized_fxc);

    let (start_mo, num_state, occ_size, vir_size, homo, lumo) = tddft_occupation_parameters(scf);
    let dim = occ_size * vir_size;
    if dim == 0 {
        return Err("No occupied-virtual excitation space for damped TDDFT".to_string());
    }

    let ks_energies = &scf.eigenvalues[0].clone();

    println!("\n=== Damped TDDFT Calculation ===");
    println!("  Method: Damped linear response (frequency-domain)");
    println!("  Spin: {}", if xlet == 'S' { "Singlet" } else if xlet == 'T' { "Triplet" } else { "Generic" });
    println!("  occ_size={}, vir_size={}, dim={}", occ_size, vir_size, dim);
    println!("  ω = {:.8} Ha ({:.4} eV)", omega, omega * 27.2114);
    println!("  γ = {:.8} Ha", gamma);

    // Prepare fxc data
    let fxc_data = prepare_fxc_data(scf);
    let alpha_hybrid = fxc_data.alpha_hybrid;

    // Obtain RI integrals
    println!("  Obtaining RI integrals...");
    let ri_ov = tddft_get_submatrix(scf, 'O', 'V', start_mo, occ_size, vir_size, homo, lumo, num_state);
    let ri_oo = tddft_get_submatrix(scf, 'O', 'O', start_mo, occ_size, vir_size, homo, lumo, num_state);
    let ri_vv = tddft_get_submatrix(scf, 'V', 'V', start_mo, occ_size, vir_size, homo, lumo, num_state);
    let num_auxbas = ri_ov.size[0];
    println!("  num_auxbas = {}", num_auxbas);

    // Reshape for exchange
    let mut ri_oo_exch = ri_oo.clone();
    ri_oo_exch.reshape([num_auxbas * occ_size, occ_size]);
    ri_oo_exch = ri_oo_exch.transpose_and_drop();
    ri_oo_exch.reshape([occ_size * num_auxbas, occ_size]);

    let mut ri_vv_exch = ri_vv.clone();
    ri_vv_exch.reshape([num_auxbas * vir_size, vir_size]);

    let mut ri_ov_exch = ri_ov.clone();
    ri_ov_exch.reshape([num_auxbas * occ_size, vir_size]);

    // Compute dipole vector
    let mu_z_vec = compute_mu_z_vec_tddft(scf);

    // Dispatch to solver
    let solution = match solver.as_str() {
        "pople" => damped_tddft_pople(
            scf, &fxc_data, &ri_ov, &ri_oo_exch, &ri_vv_exch, &ri_ov_exch,
            &mu_z_vec, ks_energies, start_mo, lumo, occ_size, vir_size,
            omega, gamma, xlet, alpha_hybrid, tol, max_iter,
        ),
        "gmres" => damped_tddft_gmres(
            scf, &fxc_data, &ri_ov, &ri_oo_exch, &ri_vv_exch, &ri_ov_exch,
            &mu_z_vec, occ_size, vir_size,
            omega, gamma, xlet, alpha_hybrid, tol, max_iter,
        ),
        "klopper" => damped_tddft_klopper(
            scf, &fxc_data, &ri_ov, &ri_oo_exch, &ri_vv_exch, &ri_ov_exch,
            &mu_z_vec, ks_energies, start_mo, lumo, occ_size, vir_size,
            omega, gamma, xlet, alpha_hybrid, tol, max_iter,
        ),
        "dense" => damped_tddft_dense(
            scf, &fxc_data, &ri_ov, &ri_oo_exch, &ri_vv_exch, &ri_ov_exch,
            &mu_z_vec, occ_size, vir_size,
            omega, gamma, xlet, alpha_hybrid,
        ),
        other => return Err(format!(
            "Invalid damped_tddft_solver: \"{}\". Expected \"pople\", \"gmres\", \"klopper\", or \"dense\".",
            other
        )),
    };

    // Export polarized density to grid files
    if !tddft_ctrl.damped_tddft_grids.is_empty() {
        println!("  Exporting polarized density to grid files...");
        export_density_tddft(
            scf, &solution,
            start_mo, occ_size, vir_size, lumo, num_state,
            &tddft_ctrl.damped_tddft_grids,
            tddft_ctrl.damped_tddft_x_start, tddft_ctrl.damped_tddft_x_end, tddft_ctrl.damped_tddft_x_points,
            tddft_ctrl.damped_tddft_y_start, tddft_ctrl.damped_tddft_y_end, tddft_ctrl.damped_tddft_y_points,
            tddft_ctrl.damped_tddft_z_start, tddft_ctrl.damped_tddft_z_end, tddft_ctrl.damped_tddft_z_points,
        );
    }

    println!("  Damped TDDFT calculation completed.\n");
    Ok(())
}
