// ============================================================================
// dynamicbse_matvec.rs — New RI projection matvec scheme for frequency-dependent BSE
//
// Implements the direct IA-pair summation method for the screened-exchange (W)
// contribution, replacing the epsilon^{-1} dressing approach.
//
// Core formula (A-block, from new-matvec-scheme.tex Eq. 27):
//   Y = Σ_p { [D^p ⊙ (U^p·X)]·(V^p)^T  −  U^p·[D^p ⊙ (X·(V^p)^T)] }
//
// B-block extension:
//   Y = Σ_p { [(U^p ⊙ D^p) @ (X^T @ V^p)]  −  [(U^p @ X^T) @ (D^p ⊙ V^p)] }
//
// No epsilon^{-1} pre-computation needed.
// ============================================================================

use num::Complex;
use tensors::MatrixFull;
use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;

use crate::ctrl_io::quasiparticle_methods::QuasiParticle;
use crate::ri_gw::get_occupation_parameters;
use crate::scf_io::SCF;

use super::matvec::{coulomb_contribution, diagonal_elements_contribution};

// ============================================================================
// Pre-processing helpers
// ============================================================================

/// Pre-compute Delta_IA[p] = eps_occ[I] - eps_vir[A] for all IA pairs
/// (corresponds to Δ_p = ε_I − ε_A in the .tex notation).
/// The pair index p = I + A * occ_size.
pub fn compute_delta_ia(
    eps_occ: &[f64],
    eps_vir: &[f64],
    occ_size: usize,
    vir_size: usize,
) -> Vec<f64> {
    let nIA = occ_size * vir_size;
    let mut delta = Vec::with_capacity(nIA);
    for a in 0..vir_size {
        for i in 0..occ_size {
            delta.push(eps_occ[i] - eps_vir[a]);
        }
    }
    delta
}

/// Auto-select IA block size based on target memory budget (~512 MB for intermediates).
pub fn auto_block_size(no: usize, nv: usize, n_IA: usize) -> usize {
    // Per-p memory (bytes): U[no²] + V[nv²] + D[no·nv] + A[no·nv] + H[no·nv] + B[no·nv] + C[no·nv]
    // ≈ (no² + nv² + 5·no·nv) * 8
    let per_p = (no * no + nv * nv + 5 * no * nv) * 8;
    if per_p == 0 {
        return 1;
    }
    let target_bytes: f64 = 512.0 * 1024.0 * 1024.0; // 512 MB
    let max_b = (target_bytes / per_p as f64) as usize;
    max_b.max(1).min(n_IA)
}

/// Build denominator matrix D[p, i, a] = 1/(ω + Δ_p + ε_i - ε_a)
/// Storage: D data is flat with layout [B, no, nv] where [no, nv] is column-major.
pub fn build_denominator_real(
    omega: f64,
    delta: &[f64],
    eps_occ: &[f64],
    eps_vir: &[f64],
    block_start: usize,
    block_size: usize,
    occ_size: usize,
    vir_size: usize,
) -> Vec<f64> {
    let no = occ_size;
    let nv = vir_size;
    let mut d = vec![0.0; block_size * no * nv];
    for p in 0..block_size {
        let dp = delta[block_start + p];
        let base = p * no * nv;
        for i in 0..no {
            let eps_i = eps_occ[i];
            for a in 0..nv {
                let val = omega + dp + eps_i - eps_vir[a];
                d[base + i + a * no] = 1.0 / val;
            }
        }
    }
    d
}

/// Build denominator D_re[p,i,a] + i·D_im[p,i,a] for complex ω.
pub fn build_denominator_complex(
    omega: Complex<f64>,
    delta: &[f64],
    eps_occ: &[f64],
    eps_vir: &[f64],
    block_start: usize,
    block_size: usize,
    occ_size: usize,
    vir_size: usize,
) -> (Vec<f64>, Vec<f64>) {
    let no = occ_size;
    let nv = vir_size;
    let mut d_re = vec![0.0; block_size * no * nv];
    let mut d_im = vec![0.0; block_size * no * nv];
    for p in 0..block_size {
        let dp = delta[block_start + p];
        let base = p * no * nv;
        for i in 0..no {
            let eps_i = eps_occ[i];
            for a in 0..nv {
                let z = Complex::new(omega.re + dp + eps_i - eps_vir[a], omega.im);
                let inv = z.inv();
                d_re[base + i + a * no] = inv.re;
                d_im[base + i + a * no] = inv.im;
            }
        }
    }
    (d_re, d_im)
}

// ============================================================================
// Helper: extract a column from a column-major MatrixFull as a slice
// ============================================================================
fn get_column<'a>(mat: &'a MatrixFull<f64>, col: usize) -> &'a [f64] {
    let rows = mat.size[0];
    let start = col * rows;
    &mat.data[start..start + rows]
}

// ============================================================================
// A-block W (screened exchange) - real version
// ============================================================================

/// A-block W·X via IA-pair summation (real ω).
///
/// ri_oo_T: [no², naux]  — RI occupied-occupied block, transposed from original [naux, no²]
/// ri_vv_T: [nv², naux]  — RI virtual-virtual block, transposed from original [naux, nv²]
/// ri_IA_T: [nIA, naux]  — RI occupied-virtual (IA pairs), transposed from original [naux, nIA]
///   where nIA = occ_size * vir_size
/// delta_IA: [nIA]       — ε_vir[A] - ε_occ[I] for each IA pair
/// eps_occ: [no], eps_vir: [nv] — orbital energies
/// omega: real frequency
/// x: [no * nv] input vector
/// block_size: IA batch size
/// exchange_rescaling: scaling factor (e.g., bse_exchange_rescaling)
///
/// Returns y: [no * nv] output vector
pub fn w_a_block_new(
    occ_size: usize,
    vir_size: usize,
    ri_oo_T: &MatrixFull<f64>,
    ri_vv_T: &MatrixFull<f64>,
    ri_IA_T: &MatrixFull<f64>,
    delta_IA: &[f64],
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: f64,
    x: &[f64],
    block_size: usize,
    exchange_rescaling: f64,
) -> Vec<f64> {
    let no = occ_size;
    let nv = vir_size;
    let nIA = no * nv;
    let naux = ri_oo_T.size[1];

    let x_mat = MatrixFull::from_vec([no, nv], x.to_vec()).unwrap();
    let mut y_mat = MatrixFull::new([no, nv], 0.0);

    let n_blocks = (nIA + block_size - 1) / block_size;

    for block_idx in 0..n_blocks {
        let block_start = block_idx * block_size;
        let B = block_size.min(nIA - block_start);

        // ── Construct R_IA_block^T = [naux, B] from ri_IA_T rows ──
        let mut RIA_T = MatrixFull::new([naux, B], 0.0);
        for p in 0..B {
            let src_row = block_start + p;
            for k in 0..naux {
                // ri_IA_T is [nIA, naux], column-major
                // RIA_T is [naux, B], column-major
                RIA_T.data[k + p * naux] = ri_IA_T.data[src_row + k * nIA];
            }
        }

        // ── U_flat = ri_oo_T @ RIA_T, shape [no², B] ──
        let mut U_flat = MatrixFull::new([no * no, B], 0.0);
        _dgemm_full(ri_oo_T, 'N', &RIA_T, 'N', &mut U_flat, 1.0, 0.0);

        // ── V_flat = ri_vv_T @ RIA_T, shape [nv², B] ──
        let mut V_flat = MatrixFull::new([nv * nv, B], 0.0);
        _dgemm_full(ri_vv_T, 'N', &RIA_T, 'N', &mut V_flat, 1.0, 0.0);

        // ── Build denominator D [B, no, nv] ──
        let d_data = build_denominator_real(omega, delta_IA, eps_occ, eps_vir,
                                            block_start, B, no, nv);

        // ── Process each p in the block ──
        for p in 0..B {
            // Extract U_p [no, no] from U_flat
            let u_slice = get_column(&U_flat, p);
            let u_mat = MatrixFull::from_vec([no, no], u_slice.to_vec()).unwrap();

            // Extract V_p [nv, nv] from V_flat
            let v_slice = get_column(&V_flat, p);
            let v_mat = MatrixFull::from_vec([nv, nv], v_slice.to_vec()).unwrap();

            // Denominator slice D_p [no, nv]
            let d_base = p * no * nv;

            // ── First term: H = D ⊙ (U·X), Y += H·V^T ──
            // A_p = U_p · X  [no,nn]@[no,nv] = [no,nv]
            let mut a_mat = MatrixFull::new([no, nv], 0.0);
            _dgemm_full(&u_mat, 'N', &x_mat, 'N', &mut a_mat, 1.0, 0.0);

            // H_p = D_p * A_p (elementwise)
            let mut h_mat = MatrixFull::new([no, nv], 0.0);
            for idx in 0..(no * nv) {
                h_mat.data[idx] = d_data[d_base + idx] * a_mat.data[idx];
            }

            // Y += H_p · V_p^T  [no,nv]@[nv,nv] = [no,nv]
            _dgemm_full(&h_mat, 'N', &v_mat, 'T', &mut y_mat, exchange_rescaling, 1.0);

            // ── Second term: C = D ⊙ (X·V^T), Y -= U·C ──
            // B_p = X · V_p^T  [no,nv]@[nv,nv] = [no,nv]
            let mut b_mat = MatrixFull::new([no, nv], 0.0);
            _dgemm_full(&x_mat, 'N', &v_mat, 'T', &mut b_mat, 1.0, 0.0);

            // C_p = D_p * B_p (elementwise)
            let mut c_mat = MatrixFull::new([no, nv], 0.0);
            for idx in 0..(no * nv) {
                c_mat.data[idx] = d_data[d_base + idx] * b_mat.data[idx];
            }

            // Y -= U_p · C_p  [no,no]@[no,nv] = [no,nv]
            _dgemm_full(&u_mat, 'N', &c_mat, 'N', &mut y_mat, -exchange_rescaling, 1.0);
        }
    }

    y_mat.data
}

// ============================================================================
// A-block W (screened exchange) - complex (real-embedded) version
// ============================================================================

/// A-block W·(x_re + i·x_im) via IA-pair summation (complex ω, real-embedded).
///
/// Returns (y_re, y_im).
pub fn w_a_block_new_complex(
    occ_size: usize,
    vir_size: usize,
    ri_oo_T: &MatrixFull<f64>,
    ri_vv_T: &MatrixFull<f64>,
    ri_IA_T: &MatrixFull<f64>,
    delta_IA: &[f64],
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: Complex<f64>,
    x_re: &[f64],
    x_im: &[f64],
    block_size: usize,
    exchange_rescaling: f64,
) -> (Vec<f64>, Vec<f64>) {
    let no = occ_size;
    let nv = vir_size;
    let nIA = no * nv;
    let naux = ri_oo_T.size[1];

    let x_re_mat = MatrixFull::from_vec([no, nv], x_re.to_vec()).unwrap();
    let x_im_mat = MatrixFull::from_vec([no, nv], x_im.to_vec()).unwrap();
    let mut y_re_mat = MatrixFull::new([no, nv], 0.0);
    let mut y_im_mat = MatrixFull::new([no, nv], 0.0);

    let n_blocks = (nIA + block_size - 1) / block_size;

    for block_idx in 0..n_blocks {
        let block_start = block_idx * block_size;
        let B = block_size.min(nIA - block_start);

        // R_IA_block^T [naux, B]
        let mut RIA_T = MatrixFull::new([naux, B], 0.0);
        for p in 0..B {
            let src_row = block_start + p;
            for k in 0..naux {
                RIA_T.data[k + p * naux] = ri_IA_T.data[src_row + k * nIA];
            }
        }

        // U_flat = ri_oo_T @ RIA_T → [no², B]
        let mut U_flat = MatrixFull::new([no * no, B], 0.0);
        _dgemm_full(ri_oo_T, 'N', &RIA_T, 'N', &mut U_flat, 1.0, 0.0);

        // V_flat = ri_vv_T @ RIA_T → [nv², B]
        let mut V_flat = MatrixFull::new([nv * nv, B], 0.0);
        _dgemm_full(ri_vv_T, 'N', &RIA_T, 'N', &mut V_flat, 1.0, 0.0);

        // D_re + i·D_im
        let (d_re, d_im) = build_denominator_complex(
            omega, delta_IA, eps_occ, eps_vir, block_start, B, no, nv);

        for p in 0..B {
            let u_slice = get_column(&U_flat, p);
            let u_mat = MatrixFull::from_vec([no, no], u_slice.to_vec()).unwrap();

            let v_slice = get_column(&V_flat, p);
            let v_mat = MatrixFull::from_vec([nv, nv], v_slice.to_vec()).unwrap();

            let d_base = p * no * nv;

            // ── First term ──
            // A_re = U · X_re, A_im = U · X_im
            let mut a_re = MatrixFull::new([no, nv], 0.0);
            let mut a_im = MatrixFull::new([no, nv], 0.0);
            _dgemm_full(&u_mat, 'N', &x_re_mat, 'N', &mut a_re, 1.0, 0.0);
            _dgemm_full(&u_mat, 'N', &x_im_mat, 'N', &mut a_im, 1.0, 0.0);

            // H_re = D_re ⊙ A_re - D_im ⊙ A_im
            // H_im = D_re ⊙ A_im + D_im ⊙ A_re
            let mut h_re = MatrixFull::new([no, nv], 0.0);
            let mut h_im = MatrixFull::new([no, nv], 0.0);
            for idx in 0..(no * nv) {
                let dr = d_re[d_base + idx];
                let di = d_im[d_base + idx];
                let ar = a_re.data[idx];
                let ai = a_im.data[idx];
                h_re.data[idx] = dr * ar - di * ai;
                h_im.data[idx] = dr * ai + di * ar;
            }

            // Y_re += H_re · V^T, Y_im += H_im · V^T
            _dgemm_full(&h_re, 'N', &v_mat, 'T', &mut y_re_mat, exchange_rescaling, 1.0);
            _dgemm_full(&h_im, 'N', &v_mat, 'T', &mut y_im_mat, exchange_rescaling, 1.0);

            // ── Second term ──
            // B_re = X_re · V^T, B_im = X_im · V^T
            let mut b_re = MatrixFull::new([no, nv], 0.0);
            let mut b_im = MatrixFull::new([no, nv], 0.0);
            _dgemm_full(&x_re_mat, 'N', &v_mat, 'T', &mut b_re, 1.0, 0.0);
            _dgemm_full(&x_im_mat, 'N', &v_mat, 'T', &mut b_im, 1.0, 0.0);

            // C_re = D_re ⊙ B_re - D_im ⊙ B_im
            // C_im = D_re ⊙ B_im + D_im ⊙ B_re
            let mut c_re = MatrixFull::new([no, nv], 0.0);
            let mut c_im = MatrixFull::new([no, nv], 0.0);
            for idx in 0..(no * nv) {
                let dr = d_re[d_base + idx];
                let di = d_im[d_base + idx];
                let br = b_re.data[idx];
                let bi = b_im.data[idx];
                c_re.data[idx] = dr * br - di * bi;
                c_im.data[idx] = dr * bi + di * br;
            }

            // Y_re -= U · C_re - U · C_im  (real embedding)
            // Y_im -= U · C_im + U · C_re
            // Wait — the formula is Y -= U · C, and C = C_re + i·C_im
            // So U·C = U·C_re + i·U·C_im
            // Real embedding: Y_re -= U·C_re, Y_im -= U·C_im
            _dgemm_full(&u_mat, 'N', &c_re, 'N', &mut y_re_mat, -exchange_rescaling, 1.0);
            _dgemm_full(&u_mat, 'N', &c_im, 'N', &mut y_im_mat, -exchange_rescaling, 1.0);
        }
    }

    (y_re_mat.data, y_im_mat.data)
}

// ============================================================================
// B-block W (screened exchange) - real version
// ============================================================================

/// B-block W·X via IA-pair summation (real ω).
///
/// For the B-block, U^p = V^p are the same projection of ri_IA onto IA pairs,
/// yielding matrices of shape [no, nv].
///
/// Formula:
///   Y = Σ_p { [(U^p ⊙ D^p) @ (X^T @ V^p)]  −  [(U^p @ X^T) @ (D^p ⊙ V^p)] }
///
/// ri_IA_T: [nIA, naux] — RI occupied-virtual (IA pairs)
///   where nIA = occ_size * vir_size
pub fn w_b_block_new(
    occ_size: usize,
    vir_size: usize,
    ri_IA_T: &MatrixFull<f64>,
    delta_IA: &[f64],
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: f64,
    x: &[f64],
    block_size: usize,
) -> Vec<f64> {
    let no = occ_size;
    let nv = vir_size;
    let nIA = no * nv;
    let naux = ri_IA_T.size[1];

    let x_mat = MatrixFull::from_vec([no, nv], x.to_vec()).unwrap();
    let mut y_mat = MatrixFull::new([no, nv], 0.0);

    let n_blocks = (nIA + block_size - 1) / block_size;

    for block_idx in 0..n_blocks {
        let block_start = block_idx * block_size;
        let B = block_size.min(nIA - block_start);

        // R_IA_block^T [naux, B]
        let mut RIA_T = MatrixFull::new([naux, B], 0.0);
        for p in 0..B {
            let src_row = block_start + p;
            for k in 0..naux {
                RIA_T.data[k + p * naux] = ri_IA_T.data[src_row + k * nIA];
            }
        }

        // U_flat = ri_IA_T_block @ RIA_T  → [no*nv, B]
        // But ri_IA_T_block is just the same ri_IA_T rows projected onto itself...
        // Actually, we need the projection of RI_IA onto each IA pair.
        // U^p_{i,b} = Σ_μ ri_IA[(i,b), μ] * ri_IA[(IA_p), μ]
        //
        // In the transposed layout:
        // U_flat[(i+b*no), p] = Σ_k ri_IA_T[(i+b*no), k] * RIA_T[k, p]
        //
        // This is: U_flat = ri_IA_T @ RIA_T  → [no*nv, naux] @ [naux, B] = [no*nv, B]

        // Extract the relevant block of ri_IA_T for the GEMM
        // But ri_IA_T already has shape [nIA, naux], and we want
        // U_flat = (sub-block of ri_IA_T) @ RIA_T
        // which is the same as the full matrix product since we only extract block columns later.
        // We need: U_flat = ri_IA_T(block_start:block_start+B, :) @ RIA_T ?
        // No, the formula is: U[p,i,b] = Σ_μ R_ov[(i,b), μ] * R_ov[(IA_p), μ]
        // For U_flat[(i+b*no), p] = Σ_μ ri_IA_T[(i+b*no), μ] * (RIA block) [μ, p]
        // = (ri_IA_T @ RIA_T)[(i+b*no), p]
        // So U_flat = ri_IA_T @ RIA_T → [nIA, B]

        let mut U_flat = MatrixFull::new([nIA, B], 0.0);
        _dgemm_full(ri_IA_T, 'N', &RIA_T, 'N', &mut U_flat, 1.0, 0.0);

        // Denominator D [B, no, nv]
        let d_data = build_denominator_real(omega, delta_IA, eps_occ, eps_vir,
                                            block_start, B, no, nv);

        // Process each p in the block
        for p in 0..B {
            // U_p [no, nv] — same as V_p (U and V are the same matrix)
            let u_slice = get_column(&U_flat, p);
            let u_mat = MatrixFull::from_vec([no, nv], u_slice.to_vec()).unwrap();
            // For B-block: V_p = U_p (same data)
            let v_mat = &u_mat;

            let d_base = p * no * nv;

            // ── First term: Y += (U ⊙ D) @ (X^T @ V) ──
            // H_p = U_p ⊙ D_p [no,nv] (elementwise)
            let mut h_mat = MatrixFull::new([no, nv], 0.0);
            for idx in 0..(no * nv) {
                h_mat.data[idx] = u_mat.data[idx] * d_data[d_base + idx];
            }

            // M = X^T @ V  [nv,no]@[no,nv] = [nv,nv]
            let mut m_mat = MatrixFull::new([nv, nv], 0.0);
            _dgemm_full(&x_mat, 'T', v_mat, 'N', &mut m_mat, 1.0, 0.0);

            // Y += H @ M  [no,nv]@[nv,nv] = [no,nv]
            _dgemm_full(&h_mat, 'N', &m_mat, 'N', &mut y_mat, 1.0, 1.0);

            // ── Second term: Y -= (U @ X^T) @ (D ⊙ V) ──
            // N = U @ X^T  [no,nv]@[nv,no] = [no,no]
            let mut n_mat = MatrixFull::new([no, no], 0.0);
            _dgemm_full(&u_mat, 'N', &x_mat, 'T', &mut n_mat, 1.0, 0.0);

            // C_p = D_p ⊙ V_p [no,nv]
            let mut c_mat = MatrixFull::new([no, nv], 0.0);
            for idx in 0..(no * nv) {
                c_mat.data[idx] = d_data[d_base + idx] * v_mat.data[idx];
            }

            // Y -= N @ C  [no,no]@[no,nv] = [no,nv]
            _dgemm_full(&n_mat, 'N', &c_mat, 'N', &mut y_mat, -1.0, 1.0);
        }
    }

    y_mat.data
}

// ============================================================================
// B-block W (screened exchange) - complex (real-embedded) version
// ============================================================================

/// B-block W·(x_re + i·x_im) via IA-pair summation (complex ω, real-embedded).
pub fn w_b_block_new_complex(
    occ_size: usize,
    vir_size: usize,
    ri_IA_T: &MatrixFull<f64>,
    delta_IA: &[f64],
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: Complex<f64>,
    x_re: &[f64],
    x_im: &[f64],
    block_size: usize,
) -> (Vec<f64>, Vec<f64>) {
    let no = occ_size;
    let nv = vir_size;
    let nIA = no * nv;
    let naux = ri_IA_T.size[1];

    let x_re_mat = MatrixFull::from_vec([no, nv], x_re.to_vec()).unwrap();
    let x_im_mat = MatrixFull::from_vec([no, nv], x_im.to_vec()).unwrap();
    let mut y_re_mat = MatrixFull::new([no, nv], 0.0);
    let mut y_im_mat = MatrixFull::new([no, nv], 0.0);

    let n_blocks = (nIA + block_size - 1) / block_size;

    for block_idx in 0..n_blocks {
        let block_start = block_idx * block_size;
        let B = block_size.min(nIA - block_start);

        // R_IA_block^T [naux, B]
        let mut RIA_T = MatrixFull::new([naux, B], 0.0);
        for p in 0..B {
            let src_row = block_start + p;
            for k in 0..naux {
                RIA_T.data[k + p * naux] = ri_IA_T.data[src_row + k * nIA];
            }
        }

        // U_flat = ri_IA_T @ RIA_T → [nIA, B]
        let mut U_flat = MatrixFull::new([nIA, B], 0.0);
        _dgemm_full(ri_IA_T, 'N', &RIA_T, 'N', &mut U_flat, 1.0, 0.0);

        // D_re + i·D_im
        let (d_re, d_im) = build_denominator_complex(
            omega, delta_IA, eps_occ, eps_vir, block_start, B, no, nv);

        for p in 0..B {
            let u_slice = get_column(&U_flat, p);
            let u_mat = MatrixFull::from_vec([no, nv], u_slice.to_vec()).unwrap();
            let v_mat = &u_mat;

            let d_base = p * no * nv;

            // ── First term: Y += (U ⊙ D) @ (X^T @ V) ──
            // H_re = U ⊙ D_re, H_im = U ⊙ D_im
            let mut h_re = MatrixFull::new([no, nv], 0.0);
            let mut h_im = MatrixFull::new([no, nv], 0.0);
            for idx in 0..(no * nv) {
                h_re.data[idx] = u_mat.data[idx] * d_re[d_base + idx];
                h_im.data[idx] = u_mat.data[idx] * d_im[d_base + idx];
            }

            // M_re = X_re^T @ V, M_im = X_im^T @ V
            // (X_re + i·X_im)^T @ V = X_re^T @ V + i·X_im^T @ V
            // since V is real
            let mut m_re = MatrixFull::new([nv, nv], 0.0);
            let mut m_im = MatrixFull::new([nv, nv], 0.0);
            _dgemm_full(&x_re_mat, 'T', v_mat, 'N', &mut m_re, 1.0, 0.0);
            _dgemm_full(&x_im_mat, 'T', v_mat, 'N', &mut m_im, 1.0, 0.0);

            // Y += (H_re + i·H_im) @ (M_re + i·M_im)
            // = (H_re@M_re - H_im@M_im) + i·(H_re@M_im + H_im@M_re)
            let mut y1_re = MatrixFull::new([no, nv], 0.0);
            let mut y1_im = MatrixFull::new([no, nv], 0.0);
            let mut y2_re = MatrixFull::new([no, nv], 0.0);
            let mut y2_im = MatrixFull::new([no, nv], 0.0);
            let mut y3_re = MatrixFull::new([no, nv], 0.0);
            let mut y3_im = MatrixFull::new([no, nv], 0.0);
            _dgemm_full(&h_re, 'N', &m_re, 'N', &mut y1_re, 1.0, 0.0);
            _dgemm_full(&h_im, 'N', &m_im, 'N', &mut y1_im, 1.0, 0.0); // actually for re
            _dgemm_full(&h_re, 'N', &m_im, 'N', &mut y2_re, 1.0, 0.0);
            _dgemm_full(&h_im, 'N', &m_re, 'N', &mut y2_im, 1.0, 0.0);
            for idx in 0..(no * nv) {
                y_re_mat.data[idx] += y1_re.data[idx] - y1_im.data[idx];
                y_im_mat.data[idx] += y2_re.data[idx] + y2_im.data[idx];
            }

            // ── Second term: Y -= (U @ X^T) @ (D ⊙ V) ──
            // N_re = U @ X_re^T, N_im = U @ X_im^T
            let mut n_re = MatrixFull::new([no, no], 0.0);
            let mut n_im = MatrixFull::new([no, no], 0.0);
            _dgemm_full(&u_mat, 'N', &x_re_mat, 'T', &mut n_re, 1.0, 0.0);
            _dgemm_full(&u_mat, 'N', &x_im_mat, 'T', &mut n_im, 1.0, 0.0);

            // C_re = D_re ⊙ V, C_im = D_im ⊙ V
            let mut c_re = MatrixFull::new([no, nv], 0.0);
            let mut c_im = MatrixFull::new([no, nv], 0.0);
            for idx in 0..(no * nv) {
                c_re.data[idx] = d_re[d_base + idx] * v_mat.data[idx];
                c_im.data[idx] = d_im[d_base + idx] * v_mat.data[idx];
            }

            // Y -= (N_re + i·N_im) @ (C_re + i·C_im)
            // = N_re@C_re - N_im@C_im + i·(N_re@C_im + N_im@C_re)
            let mut z1_re = MatrixFull::new([no, nv], 0.0);
            let mut z1_im = MatrixFull::new([no, nv], 0.0);
            let mut z2_re = MatrixFull::new([no, nv], 0.0);
            let mut z2_im = MatrixFull::new([no, nv], 0.0);
            _dgemm_full(&n_re, 'N', &c_re, 'N', &mut z1_re, 1.0, 0.0);
            _dgemm_full(&n_im, 'N', &c_im, 'N', &mut z1_im, 1.0, 0.0);
            _dgemm_full(&n_re, 'N', &c_im, 'N', &mut z2_re, 1.0, 0.0);
            _dgemm_full(&n_im, 'N', &c_re, 'N', &mut z2_im, 1.0, 0.0);
            for idx in 0..(no * nv) {
                y_re_mat.data[idx] -= z1_re.data[idx] - z1_im.data[idx];
                y_im_mat.data[idx] -= z2_re.data[idx] + z2_im.data[idx];
            }
        }
    }

    (y_re_mat.data, y_im_mat.data)
}

// ============================================================================
// Complete A-block matvec: diag·x − W·x + V·x
// ============================================================================

/// A-block matvec: result = D·x − W·x + V·x
pub fn a_block_matvec_new(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    ri_oo_T: &MatrixFull<f64>,
    ri_vv_T: &MatrixFull<f64>,
    ri_IA_T: &MatrixFull<f64>,
    delta_IA: &[f64],
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: f64,
    x: &[f64],
    block_size: usize,
) -> Vec<f64> {
    let (_, _, occ_size, vir_size, _, _) = get_occupation_parameters(scf_data, 'N');

    let xlet = if qp_ctrl.bse_spin == "triplet" { 'T' }
               else if qp_ctrl.bse_spin == "singlet" { 'S' }
               else { 'R' };

    // 1. Diagonal contribution: D · x
    let mut result = diagonal_elements_contribution(scf_data, &x.to_vec());

    // 2. W contribution (screened exchange, negative sign)
    let w = w_a_block_new(
        occ_size, vir_size,
        ri_oo_T, ri_vv_T, ri_IA_T,
        delta_IA, eps_occ, eps_vir,
        omega, x, block_size,
        qp_ctrl.bse_exchange_rescaling,
    );

    for i in 0..result.len() {
        result[i] -= w[i];
    }

    // 3. Coulomb contribution V (singlet or 'R' only)
    let ri_ov = crate::ri_bse::get_submatrix(scf_data, 'O', 'V', 'N');
    if xlet == 'S' || xlet == 'R' {
        let scale = if xlet == 'S' { 2.0 } else { 1.0 };
        let v = coulomb_contribution(&ri_ov, &x.to_vec());
        for i in 0..result.len() {
            result[i] += scale * v[i];
        }
    }

    result
}

// ============================================================================
// Complete A-block matvec (complex, real-embedded)
// ============================================================================

pub fn a_block_matvec_new_complex(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    ri_oo_T: &MatrixFull<f64>,
    ri_vv_T: &MatrixFull<f64>,
    ri_IA_T: &MatrixFull<f64>,
    delta_IA: &[f64],
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: Complex<f64>,
    x_re: &[f64],
    x_im: &[f64],
    block_size: usize,
) -> (Vec<f64>, Vec<f64>) {
    let (_, _, occ_size, vir_size, _, _) = get_occupation_parameters(scf_data, 'N');

    let xlet = if qp_ctrl.bse_spin == "triplet" { 'T' }
               else if qp_ctrl.bse_spin == "singlet" { 'S' }
               else { 'R' };

    let n = occ_size * vir_size;

    // 1. Diagonal contribution (real diagonal, applied to re/im)
    let mut result_re = diagonal_elements_contribution(scf_data, &x_re.to_vec());
    let mut result_im = diagonal_elements_contribution(scf_data, &x_im.to_vec());

    // 2. W contribution (complex)
    let (w_re, w_im) = w_a_block_new_complex(
        occ_size, vir_size,
        ri_oo_T, ri_vv_T, ri_IA_T,
        delta_IA, eps_occ, eps_vir,
        omega, x_re, x_im, block_size,
        qp_ctrl.bse_exchange_rescaling,
    );

    for i in 0..n {
        result_re[i] -= w_re[i];
        result_im[i] -= w_im[i];
    }

    // 3. Coulomb contribution (real matrix, applied to re/im)
    let ri_ov = crate::ri_bse::get_submatrix(scf_data, 'O', 'V', 'N');
    if xlet == 'S' || xlet == 'R' {
        let scale = if xlet == 'S' { 2.0 } else { 1.0 };
        let v_re = coulomb_contribution(&ri_ov, &x_re.to_vec());
        let v_im = coulomb_contribution(&ri_ov, &x_im.to_vec());
        for i in 0..n {
            result_re[i] += scale * v_re[i];
            result_im[i] += scale * v_im[i];
        }
    }

    (result_re, result_im)
}

// ============================================================================
// Complete B-block matvec: −W·x + V·x
// ============================================================================

pub fn b_block_matvec_new(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    ri_IA_T: &MatrixFull<f64>,
    delta_IA: &[f64],
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: f64,
    x: &[f64],
    block_size: usize,
) -> Vec<f64> {
    let (_, _, occ_size, vir_size, _, _) = get_occupation_parameters(scf_data, 'N');

    let xlet = if qp_ctrl.bse_spin == "triplet" { 'T' }
               else if qp_ctrl.bse_spin == "singlet" { 'S' }
               else { 'R' };

    let n = occ_size * vir_size;

    // 1. W contribution (negative sign)
    let w = w_b_block_new(
        occ_size, vir_size, ri_IA_T, delta_IA, eps_occ, eps_vir,
        omega, x, block_size,
    );

    let mut result: Vec<f64> = w.iter().map(|wi| -wi).collect();

    // 2. Coulomb contribution
    let ri_ov = crate::ri_bse::get_submatrix(scf_data, 'O', 'V', 'N');
    if xlet == 'S' || xlet == 'R' {
        let scale = if xlet == 'S' { 2.0 } else { 1.0 };
        let v = coulomb_contribution(&ri_ov, &x.to_vec());
        for i in 0..n {
            result[i] += scale * v[i];
        }
    }

    result
}

// ============================================================================
// Complete B-block matvec (complex, real-embedded)
// ============================================================================

pub fn b_block_matvec_new_complex(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    ri_IA_T: &MatrixFull<f64>,
    delta_IA: &[f64],
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: Complex<f64>,
    x_re: &[f64],
    x_im: &[f64],
    block_size: usize,
) -> (Vec<f64>, Vec<f64>) {
    let (_, _, occ_size, vir_size, _, _) = get_occupation_parameters(scf_data, 'N');

    let xlet = if qp_ctrl.bse_spin == "triplet" { 'T' }
               else if qp_ctrl.bse_spin == "singlet" { 'S' }
               else { 'R' };

    let n = occ_size * vir_size;

    // 1. W contribution (negative sign)
    let (w_re, w_im) = w_b_block_new_complex(
        occ_size, vir_size, ri_IA_T, delta_IA, eps_occ, eps_vir,
        omega, x_re, x_im, block_size,
    );

    let mut result_re: Vec<f64> = w_re.iter().map(|wi| -wi).collect();
    let mut result_im: Vec<f64> = w_im.iter().map(|wi| -wi).collect();

    // 2. Coulomb contribution (real matrix)
    let ri_ov = crate::ri_bse::get_submatrix(scf_data, 'O', 'V', 'N');
    if xlet == 'S' || xlet == 'R' {
        let scale = if xlet == 'S' { 2.0 } else { 1.0 };
        let v_re = coulomb_contribution(&ri_ov, &x_re.to_vec());
        let v_im = coulomb_contribution(&ri_ov, &x_im.to_vec());
        for i in 0..n {
            result_re[i] += scale * v_re[i];
            result_im[i] += scale * v_im[i];
        }
    }

    (result_re, result_im)
}

// ============================================================================
// Composite matvec: y = ((A+B)(A−B) − ω²I)·x  (real)
// ============================================================================

pub fn composite_matvec_new_real(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    ri_oo_T: &MatrixFull<f64>,
    ri_vv_T: &MatrixFull<f64>,
    ri_IA_T: &MatrixFull<f64>,
    delta_IA: &[f64],
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: f64,
    x: &[f64],
    block_size: usize,
) -> Vec<f64> {
    let n = x.len();

    // t = (A−B)·x
    let a_x = a_block_matvec_new(
        scf_data, qp_ctrl, ri_oo_T, ri_vv_T, ri_IA_T,
        delta_IA, eps_occ, eps_vir, omega, x, block_size,
    );
    let b_x = b_block_matvec_new(
        scf_data, qp_ctrl, ri_IA_T, delta_IA, eps_occ, eps_vir,
        omega, x, block_size,
    );
    let t: Vec<f64> = a_x.iter().zip(b_x.iter()).map(|(a, b)| a - b).collect();

    // s = (A+B)·t
    let a_t = a_block_matvec_new(
        scf_data, qp_ctrl, ri_oo_T, ri_vv_T, ri_IA_T,
        delta_IA, eps_occ, eps_vir, omega, &t, block_size,
    );
    let b_t = b_block_matvec_new(
        scf_data, qp_ctrl, ri_IA_T, delta_IA, eps_occ, eps_vir,
        omega, &t, block_size,
    );
    let s: Vec<f64> = a_t.iter().zip(b_t.iter()).map(|(a, b)| a + b).collect();

    // y = s − ω²·x
    let omega2 = omega * omega;
    s.iter().zip(x.iter()).map(|(s_i, x_i)| s_i - omega2 * x_i).collect()
}

// ============================================================================
// Composite matvec: y = ((A+B)(A−B) − z²I)·(x_re + i·x_im)  (complex, real-embedded)
// ============================================================================

pub fn composite_matvec_new_complex(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    ri_oo_T: &MatrixFull<f64>,
    ri_vv_T: &MatrixFull<f64>,
    ri_IA_T: &MatrixFull<f64>,
    delta_IA: &[f64],
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: Complex<f64>,
    x_re: &[f64],
    x_im: &[f64],
    block_size: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = x_re.len();

    // ── t = (A−B)·(x_re + i·x_im) ──
    let (a_re, a_im) = a_block_matvec_new_complex(
        scf_data, qp_ctrl, ri_oo_T, ri_vv_T, ri_IA_T,
        delta_IA, eps_occ, eps_vir, omega, x_re, x_im, block_size,
    );
    let (b_re, b_im) = b_block_matvec_new_complex(
        scf_data, qp_ctrl, ri_IA_T, delta_IA, eps_occ, eps_vir,
        omega, x_re, x_im, block_size,
    );
    let mut t_re = Vec::with_capacity(n);
    let mut t_im = Vec::with_capacity(n);
    for i in 0..n {
        t_re.push(a_re[i] - b_re[i]);
        t_im.push(a_im[i] - b_im[i]);
    }

    // ── s = (A+B)·t ──
    let (a_s_re, a_s_im) = a_block_matvec_new_complex(
        scf_data, qp_ctrl, ri_oo_T, ri_vv_T, ri_IA_T,
        delta_IA, eps_occ, eps_vir, omega, &t_re, &t_im, block_size,
    );
    let (b_s_re, b_s_im) = b_block_matvec_new_complex(
        scf_data, qp_ctrl, ri_IA_T, delta_IA, eps_occ, eps_vir,
        omega, &t_re, &t_im, block_size,
    );
    let mut s_re = Vec::with_capacity(n);
    let mut s_im = Vec::with_capacity(n);
    for i in 0..n {
        s_re.push(a_s_re[i] + b_s_re[i]);
        s_im.push(a_s_im[i] + b_s_im[i]);
    }

    // ── y = s − z²·x ──
    // z² = (ω_r² - ω_i²) + i·(2·ω_r·ω_i)
    let omega_re = omega.re;
    let omega_im = omega.im;
    let a2mb2 = omega_re * omega_re - omega_im * omega_im;
    let two_ab = 2.0 * omega_re * omega_im;

    let mut y_re = Vec::with_capacity(n);
    let mut y_im = Vec::with_capacity(n);
    for i in 0..n {
        let z2_x_re = a2mb2 * x_re[i] - two_ab * x_im[i];
        let z2_x_im = two_ab * x_re[i] + a2mb2 * x_im[i];
        y_re.push(s_re[i] - z2_x_re);
        y_im.push(s_im[i] - z2_x_im);
    }

    (y_re, y_im)
}
