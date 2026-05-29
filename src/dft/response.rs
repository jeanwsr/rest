/// PySCF-style response function generator for RKS/RHF.
///
/// Matches PySCF scf/_response_functions.py:
///   - `_gen_rhf_response(singlet=None)` → AO-space Fock response `vind(dm1)`
///   - `gen_vind` via hessian/rhf.py → AO↔MO bridging
///
/// For HF: vind(dm1) = J[dm1] - 0.5*K[dm1]
/// For DFT: vind(dm1) = fxc[dm1] + J[dm1] - hyb*K[dm1]

use rest_tensors::{MatrixFull, MatrixUpper};
use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;
use crate::scf_io::SCF;
use crate::dft::num_int::{FXCMatvecData, prepare_fxc_data, fxc_matvec};

/// Precomputed workspace for gen_vind: caches C_occ, C_vir slices.
pub struct VindWorkspace {
    pub c_occ: MatrixFull<f64>,
    pub c_vir: MatrixFull<f64>,
    pub nao: usize,
    pub nocc: usize,
    pub nvir: usize,
    pub dim: usize,
}

impl VindWorkspace {
    pub fn new(scf: &SCF, nocc: usize, nvir: usize, start_mo: usize, lumo: usize) -> Self {
        let nao = scf.mol.num_basis;
        let mo_coeff = &scf.eigenvectors[0];
        let dim = nocc * nvir;

        let mut c_occ = MatrixFull::new([nao, nocc], 0.0);
        let mut c_vir = MatrixFull::new([nao, nvir], 0.0);
        for j in 0..nocc { for i in 0..nao { c_occ[[i, j]] = mo_coeff[[i, start_mo + j]]; } }
        for j in 0..nvir { for i in 0..nao { c_vir[[i, j]] = mo_coeff[[i, lumo + j]]; } }

        VindWorkspace { c_occ, c_vir, nao, nocc, nvir, dim }
    }
}

/// Compute J in AO basis (upper triangular format).
fn compute_j_upper(
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
fn compute_k_upper(
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

/// Optimized fvind: z (MO VO-block) → G(z) = C_vir^T@(J-0.5*K)@C_occ (+ fxc).
///
/// Uses precomputed VindWorkspace to avoid redundant slices and DGEMM for projection.
pub fn gen_vind_opt(
    scf: &SCF,
    ws: &VindWorkspace,
    z: &[f64],
    fxc_data: Option<&FXCMatvecData>,
) -> Vec<f64> {
    let nao = ws.nao;
    let nocc = ws.nocc;
    let nvir = ws.nvir;
    let dim = ws.dim;

    // ── Step 1: Build AO density matrix dm1 = 2*C_vir @ z @ C_occ^T + h.c. ──
    // z_scaled = 2*z, as [nvir, nocc]
    let mut z_scaled = MatrixFull::new([nvir, nocc], 0.0);
    for a in 0..nvir { for i in 0..nocc { z_scaled[[a, i]] = 2.0 * z[i + a * nocc]; } }

    // dp1 = C_vir @ (2*z) @ C_occ^T
    let mut t1 = MatrixFull::new([nao, nocc], 0.0);
    _dgemm_full(&ws.c_vir, 'N', &z_scaled, 'N', &mut t1, 1.0, 0.0);
    let mut dp1 = MatrixFull::new([nao, nao], 0.0);
    _dgemm_full(&t1, 'N', &ws.c_occ, 'T', &mut dp1, 1.0, 0.0);

    // dp2 = C_occ @ (2*z^T) @ C_vir^T  (= dp1^T)
    let mut t2 = MatrixFull::new([nao, nvir], 0.0);
    _dgemm_full(&ws.c_occ, 'N', &z_scaled, 'T', &mut t2, 1.0, 0.0);
    let mut dp2 = MatrixFull::new([nao, nao], 0.0);
    _dgemm_full(&t2, 'N', &ws.c_vir, 'T', &mut dp2, 1.0, 0.0);

    let mut dm1 = MatrixFull::new([nao, nao], 0.0);
    for i in 0..nao { for j in 0..nao { dm1[[i, j]] = dp1[[i, j]] + dp2[[i, j]]; } }
    let dm_vec = vec![dm1];

    // ── Step 2: Compute J, K via REST JK, convert from upper to full ──
    let j_full = compute_j_upper(scf, &dm_vec).to_matrixfull()
        .unwrap_or_else(|| panic!("J to_matrixfull failed"));
    let k_full = compute_k_upper(scf, &dm_vec).to_matrixfull()
        .unwrap_or_else(|| panic!("K to_matrixfull failed"));

    // ── Step 3: v_ao = J - 0.5*K ──
    let mut v_ao = MatrixFull::new([nao, nao], 0.0);
    for i in 0..nao { for j in 0..nao {
        v_ao[[i, j]] = j_full[[i, j]] - 0.5 * k_full[[i, j]];
    }}

    // ── Step 4: fxc contribution ──
    let fxc_mo = if let Some(fxc) = fxc_data { fxc_matvec(fxc, z) } else { vec![0.0; dim] };

    // ── Step 5: Project v_ao to MO VO-block via DGEMM ──
    // tmp[nocc, nvir] = C_occ^T @ v_ao @ C_vir
    // Step 5a: tmp2[nao, nvir] = v_ao @ C_vir
    let mut tmp2 = MatrixFull::new([nao, nvir], 0.0);
    _dgemm_full(&v_ao, 'N', &ws.c_vir, 'N', &mut tmp2, 1.0, 0.0);
    // Step 5b: result[nocc, nvir] = C_occ^T @ tmp2
    let mut result = MatrixFull::new([nocc, nvir], 0.0);
    _dgemm_full(&ws.c_occ, 'T', &tmp2, 'N', &mut result, 1.0, 0.0);

    // Flatten to [i + a*nocc] convention and add fxc
    let mut res = vec![0.0; dim];
    for a in 0..nvir { for i in 0..nocc {
        res[i + a * nocc] = result[[i, a]] + fxc_mo[i + a * nocc];
    }}
    res
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

    // fxc_matvec only
    let fxc_only = fxc_data.as_ref().map(|f| fxc_matvec(f, &z_test))
        .unwrap_or_else(|| vec![0.0; dim]);

    let ws = VindWorkspace::new(scf, occ_size, vir_size, start_mo, lumo);

    // gen_vind total (JK+fxc)
    let gen_v = gen_vind_opt(scf, &ws, &z_test, fxc_data.as_ref());

    // gen_vind JK only (no fxc)
    let jk_only = gen_vind_opt(scf, &ws, &z_test, None);

    // fxc part from gen_vind = total - JK
    let mut fxc_from_gen = vec![0.0; dim];
    for i in 0..dim { fxc_from_gen[i] = gen_v[i] - jk_only[i]; }

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
