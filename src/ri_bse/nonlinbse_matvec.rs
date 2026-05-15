use num::Complex;
use rayon::iter::ParallelBridge;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;
use tensors::{ri, MathMatrix, MatrixFull};
use crate::ctrl_io::quasiparticle_methods::QuasiParticle;
use crate::ri_gw;
use crate::ri_gw::get_occupation_parameters;
use crate::molecule_io::Molecule;
use crate::scf_io::SCF;
use super::matvec_trace;
use rest_tensors::matrix::matrix_blas_lapack::{_dgemm_full, _dinverse};

// ============================================================================
// 复频率响应矩阵和逆介电矩阵
// ============================================================================
pub fn response_matrix_complex(quasiparticle_energies_w:&Vec<f64>,occ_size:usize,vir_size:usize,ri_ov:&MatrixFull<f64>,omega:Complex<f64>)->MatrixFull<Complex<f64>>{
    let num_auxbas=ri_ov.size[0];
    let mut ri_to_be_processed=MatrixFull::<Complex<f64>>::new([num_auxbas,0],Complex::new(0.0,0.0));
    let mut ri_as_vecs:Vec<(usize,Vec<Complex<f64>>)>=ri_ov.iter_columns_full().enumerate().par_bridge().map(|(n,ri_n)|{
        let i=n%occ_size;
        let a=occ_size+n/occ_size;
        let energy_gap=quasiparticle_energies_w[a]-quasiparticle_energies_w[i];
        let denominator = Complex::new(energy_gap.powf(2.0), 0.0) - omega.powf(2.0);
        let multiply_number = Complex::new(-2.0 * energy_gap, 0.0) / denominator;
        let ri_n_vec: Vec<Complex<f64>> = ri_n.iter().map(|&x| multiply_number * Complex::new(x, 0.0)).collect();
        (n,ri_n_vec)
    }).collect();
    ri_as_vecs.sort_by_key(|(i, _)| *i);
    for ri_vec in ri_as_vecs {
        ri_to_be_processed.push_column(&ri_vec.1);
    }
    // response = ri_ov * ri_to_be_processed^T  (via LAPACK zgemm)
    // ri_ov is real, so convert to complex first (pure real part)
    let mut response:MatrixFull<Complex<f64>>=MatrixFull::new([num_auxbas,num_auxbas],Complex::new(0.0,0.0));
    let ri_ov_c64 = MatrixFull::<Complex<f64>>::from_matrixfull_re(ri_ov).unwrap();
    response.lapack_zgemm(&ri_ov_c64, &ri_to_be_processed, 'N', 'T', Complex::new(1.0, 0.0), Complex::new(0.0, 0.0));
    response.self_multiple(Complex::new(2.0, 0.0));
    response
}
pub fn inverse_dielectric_matrix_complex(response:&MatrixFull<Complex<f64>>)->(MatrixFull<f64>,MatrixFull<f64>){
    let num_auxbas = response.size[0];
    // dielectric = 1 - response  (complex)
    let mut dielectric = response.clone();
    dielectric.self_multiple(Complex::new(-1.0, 0.0));
    for i in 0..num_auxbas {
        dielectric[[i, i]] += Complex::new(1.0, 0.0);
    }
    // Complex matrix inversion via LAPACK: zgetrf + zgetri
    let ipiv = dielectric.lapack_zgetrf();
    dielectric.lapack_zgetri(&ipiv);
    // Split into real and imaginary parts
    let re = MatrixFull::from_vec(dielectric.size, dielectric.data.iter().map(|c| c.re).collect()).unwrap();
    let im = MatrixFull::from_vec(dielectric.size, dielectric.data.iter().map(|c| c.im).collect()).unwrap();
    (re, im)
}

// ============================================================================
// 实频率响应矩阵和逆介电矩阵
// ============================================================================
pub fn response_matrix_real(quasiparticle_energies_w:&Vec<f64>,occ_size:usize,vir_size:usize,ri_ov:&MatrixFull<f64>,omega:f64)->MatrixFull<f64>{
    let num_auxbas=ri_ov.size[0];
    let mut ri_to_be_processed=MatrixFull::new([num_auxbas,0],0.0_f64);
    let mut ri_as_vecs:Vec<(usize,Vec<f64>)>=ri_ov.iter_columns_full().enumerate().par_bridge().map(|(n,ri_n)|{
        let i=n%occ_size;
        let a=occ_size+n/occ_size;
        let energy_gap=quasiparticle_energies_w[a]-quasiparticle_energies_w[i];
        let multiply_number=-2.0*energy_gap/(energy_gap.powi(2)-omega.powi(2));
        let ri_n_vec=ri_n.to_vec();
        (n,ri_n_vec.iter().map(|&x|x*multiply_number).collect::<Vec<f64>>())
    }).collect();
    ri_as_vecs.sort_by_key(|(i, _)| *i);
    for ri_vec in ri_as_vecs {
        ri_to_be_processed.push_column(&ri_vec.1);
    }
    let mut response:MatrixFull<f64>=MatrixFull::new([num_auxbas,num_auxbas],0.0);
    _dgemm_full(ri_ov,'N',&ri_to_be_processed,'T',&mut response,1.0,0.0);
    response.self_multiple(2.0);
    response
}
pub fn inverse_dielectric_matrix_real(response:&MatrixFull<f64>)->MatrixFull<f64>{
    let num_auxbas = response.size[0];
    let mut dielectric = response.clone();
    dielectric.self_multiple(-1.0);
    for i in 0..num_auxbas {
        dielectric[[i, i]] += 1.0;
    }
    _dinverse(&dielectric).expect("unsuccessful _dinverse in inverse_dielectric_matrix_real")
}

// ============================================================================
// 复数频率 W matvec：实嵌入，纯实数运算
// ============================================================================

/// A block W · z 的复数版本（实嵌入 2N 向量）
///
/// 给定 z = z_re + i·z_im，计算：
///   (W_re + i·W_im) · (z_re + i·z_im)
///
/// 实嵌入结果：
///   result_re = W_re·z_re − W_im·z_im
///   result_im = W_re·z_im + W_im·z_re
///
/// 所有运算均为实数 _dgemm_full，不涉及复数线性代数。
/// 矩阵 ri_vv, ri_oo_tilde_re, ri_oo_tilde_im 的 shape 与原实数版本完全一致。
pub fn w_contribution_a_block_complex(
    occ_size: usize,
    vir_size: usize,
    ri_vv: &MatrixFull<f64>,
    ri_oo_tilde_re: &MatrixFull<f64>,
    ri_oo_tilde_im: &MatrixFull<f64>,
    z_re: &[f64],
    z_im: &[f64],
    exchange_rescaling: f64,
) -> (Vec<f64>, Vec<f64>) {
    let num_auxbas = ri_vv.size[0] / ri_vv.size[1];

    // Step 1: t = ri_vv * z_mat^T   (投影到 aux 基)
    let z_mat_re = MatrixFull::from_vec([occ_size, vir_size], z_re.to_vec()).unwrap();
    let z_mat_im = MatrixFull::from_vec([occ_size, vir_size], z_im.to_vec()).unwrap();

    let mut t_re = MatrixFull::new([num_auxbas * vir_size, occ_size], 0.0);
    let mut t_im = MatrixFull::new([num_auxbas * vir_size, occ_size], 0.0);

    _dgemm_full(ri_vv, 'N', &z_mat_re, 'T', &mut t_re, 1.0, 0.0);
    _dgemm_full(ri_vv, 'N', &z_mat_im, 'T', &mut t_im, 1.0, 0.0);

    // Step 2: 重排（transpose + reshape）
    t_re = t_re.transpose_and_drop();
    t_re.reshape([num_auxbas * occ_size, vir_size]);
    t_im = t_im.transpose_and_drop();
    t_im.reshape([num_auxbas * occ_size, vir_size]);

    // Step 3: result = ri_oo_tilde^T · t
    //   result_re = ri_oo_tilde_re^T · t_re  −  ri_oo_tilde_im^T · t_im
    //   result_im = ri_oo_tilde_re^T · t_im  +  ri_oo_tilde_im^T · t_re
    let mut tmp1 = MatrixFull::new([occ_size, vir_size], 0.0);
    let mut tmp2 = MatrixFull::new([occ_size, vir_size], 0.0);
    let mut tmp3 = MatrixFull::new([occ_size, vir_size], 0.0);
    let mut tmp4 = MatrixFull::new([occ_size, vir_size], 0.0);

    _dgemm_full(ri_oo_tilde_re, 'T', &t_re, 'N', &mut tmp1, 1.0, 0.0);
    _dgemm_full(ri_oo_tilde_im, 'T', &t_im, 'N', &mut tmp2, 1.0, 0.0);
    _dgemm_full(ri_oo_tilde_re, 'T', &t_im, 'N', &mut tmp3, 1.0, 0.0);
    _dgemm_full(ri_oo_tilde_im, 'T', &t_re, 'N', &mut tmp4, 1.0, 0.0);

    let result_re: Vec<f64> = tmp1.data.iter().zip(tmp2.data.iter())
        .map(|(a, b)| (a - b) * exchange_rescaling).collect();
    let result_im: Vec<f64> = tmp3.data.iter().zip(tmp4.data.iter())
        .map(|(a, b)| (a + b) * exchange_rescaling).collect();

    (result_re, result_im)
}

/// Helper: B block 数据重排（并行块交换）
/// Matches the original w_contribution_b_block_dgemm reorganisation.
/// Input data has occ_size × occ_size blocks, each of num_auxbas elements,
/// laid out column-major as [num_auxbas*occ_size, occ_size].
fn reorganize_b_block(data: &[f64], num_auxbas: usize, occ_size: usize) -> Vec<f64> {
    let mut result = vec![0.0; data.len()];
    result.par_chunks_exact_mut(num_auxbas).enumerate().for_each(|(new_idx, target_chunk)| {
        let n2 = new_idx / occ_size;
        let n1 = new_idx % occ_size;
        let orig_idx = n1 * occ_size + n2;
        let source_start = orig_idx * num_auxbas;
        let source_end = source_start + num_auxbas;
        target_chunk.copy_from_slice(&data[source_start..source_end]);
    });
    result
}

/// B block W · z 的复数版本（实嵌入 2N 向量）
pub fn w_contribution_b_block_complex(
    occ_size: usize,
    vir_size: usize,
    ri_ov: &MatrixFull<f64>,
    ri_ov_tilde_re: &MatrixFull<f64>,
    ri_ov_tilde_im: &MatrixFull<f64>,
    z_re: &[f64],
    z_im: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let num_auxbas = ri_ov.size[0] / occ_size;

    // Step 1: t = ri_ov * z_mat^T   (投影到 aux 基)
    let z_mat_re = MatrixFull::from_vec([occ_size, vir_size], z_re.to_vec()).unwrap();
    let z_mat_im = MatrixFull::from_vec([occ_size, vir_size], z_im.to_vec()).unwrap();

    let mut t_re = MatrixFull::new([num_auxbas * occ_size, occ_size], 0.0);
    let mut t_im = MatrixFull::new([num_auxbas * occ_size, occ_size], 0.0);

    _dgemm_full(ri_ov, 'N', &z_mat_re, 'T', &mut t_re, 1.0, 0.0);
    _dgemm_full(ri_ov, 'N', &z_mat_im, 'T', &mut t_im, 1.0, 0.0);

    // Step 2: B block 特有的数据重排
    let reorg_re = reorganize_b_block(&t_re.data, num_auxbas, occ_size);
    let reorg_im = reorganize_b_block(&t_im.data, num_auxbas, occ_size);

    let t_reorg_re = MatrixFull::from_vec([num_auxbas * occ_size, occ_size], reorg_re).unwrap();
    let t_reorg_im = MatrixFull::from_vec([num_auxbas * occ_size, occ_size], reorg_im).unwrap();

    // Step 3: result = t^T · ri_ov_tilde
    //   result_re = t_re^T · ri_ov_tilde_re  −  t_im^T · ri_ov_tilde_im
    //   result_im = t_re^T · ri_ov_tilde_im  +  t_im^T · ri_ov_tilde_re
    let mut tmp1 = MatrixFull::new([occ_size, vir_size], 0.0);
    let mut tmp2 = MatrixFull::new([occ_size, vir_size], 0.0);
    let mut tmp3 = MatrixFull::new([occ_size, vir_size], 0.0);
    let mut tmp4 = MatrixFull::new([occ_size, vir_size], 0.0);

    _dgemm_full(&t_reorg_re, 'T', ri_ov_tilde_re, 'N', &mut tmp1, 1.0, 0.0);
    _dgemm_full(&t_reorg_im, 'T', ri_ov_tilde_im, 'N', &mut tmp2, 1.0, 0.0);
    _dgemm_full(&t_reorg_re, 'T', ri_ov_tilde_im, 'N', &mut tmp3, 1.0, 0.0);
    _dgemm_full(&t_reorg_im, 'T', ri_ov_tilde_re, 'N', &mut tmp4, 1.0, 0.0);

    let result_re: Vec<f64> = tmp1.data.iter().zip(tmp2.data.iter())
        .map(|(a, b)| a - b).collect();
    let result_im: Vec<f64> = tmp3.data.iter().zip(tmp4.data.iter())
        .map(|(a, b)| a + b).collect();

    (result_re, result_im)
}

// ============================================================================
// 完整的 A / B block matvec（复数版）
// ============================================================================

/// A block matvec 复数版：result = D·z − W·z + (2)V·z
///
/// 输入 z_re, z_im 各为长度 occ_size × vir_size 的实向量，
/// 输出 (result_re, result_im) 各为同样长度的实向量。
/// 所有内部运算均为实数运算（实嵌入 2N 模式）。
pub fn a_block_matvec_complex(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    ri_vv: &MatrixFull<f64>,
    ri_ov: &MatrixFull<f64>,
    ri_oo_tilde_re: &MatrixFull<f64>,
    ri_oo_tilde_im: &MatrixFull<f64>,
    z_re: &[f64],
    z_im: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    matvec_trace::trace("a_block_matvec_complex");
    let (start_mo, num_state, occ_size, vir_size, homo, lumo) = get_occupation_parameters(scf_data, 'N');
    let xlet = if qp_ctrl.bse_spin == "triplet" { 'T' }
               else if qp_ctrl.bse_spin == "singlet" { 'S' }
               else { 'R' };

    // 1. 对角元贡献（实对角矩阵，re/im 各自独立）
    let mut result_re = super::matvec::diagonal_elements_contribution(scf_data, &z_re.to_vec());
    let mut result_im = super::matvec::diagonal_elements_contribution(scf_data, &z_im.to_vec());

    // 2. W 贡献：result = diag − W·z
    let (w_re, w_im) = w_contribution_a_block_complex(
        occ_size, vir_size,
        ri_vv, ri_oo_tilde_re, ri_oo_tilde_im,
        z_re, z_im,
        qp_ctrl.bse_exchange_rescaling,
    );
    for i in 0..result_re.len() {
        result_re[i] = result_re[i] - w_re[i];
        result_im[i] = result_im[i] - w_im[i];
    }

    // 3. 库仑贡献 V（实矩阵，对角作用于实嵌入）
    if xlet == 'S' || xlet == 'R' {
        let scale = if xlet == 'S' { 2.0 } else { 1.0 };
        let v_re = super::matvec::coulomb_contribution(ri_ov, &z_re.to_vec());
        let v_im = super::matvec::coulomb_contribution(ri_ov, &z_im.to_vec());
        for i in 0..result_re.len() {
            result_re[i] += scale * v_re[i];
            result_im[i] += scale * v_im[i];
        }
    }

    (result_re, result_im)
}

/// B block matvec 复数版：result = −W·z + (2)V·z
pub fn b_block_matvec_complex(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    ri_ov_a: &MatrixFull<f64>,
    ri_ov_b: &MatrixFull<f64>,
    ri_ov_tilde_re: &MatrixFull<f64>,
    ri_ov_tilde_im: &MatrixFull<f64>,
    z_re: &[f64],
    z_im: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    matvec_trace::trace("b_block_matvec_complex");
    let (start_mo, num_state, occ_size, vir_size, homo, lumo) = get_occupation_parameters(scf_data, 'N');
    let xlet = if qp_ctrl.bse_spin == "triplet" { 'T' }
               else if qp_ctrl.bse_spin == "singlet" { 'S' }
               else { 'R' };

    // 1. W 贡献（负号）
    let (w_re, w_im) = w_contribution_b_block_complex(
        occ_size, vir_size,
        ri_ov_b, ri_ov_tilde_re, ri_ov_tilde_im,
        z_re, z_im,
    );
    let mut result_re: Vec<f64> = w_re.iter().map(|w| -w).collect();
    let mut result_im: Vec<f64> = w_im.iter().map(|w| -w).collect();

    // 2. 库仑贡献 V（实矩阵，对角作用于实嵌入）
    if xlet == 'S' || xlet == 'R' {
        let scale = if xlet == 'S' { 2.0 } else { 1.0 };
        let v_re = super::matvec::coulomb_contribution(ri_ov_a, &z_re.to_vec());
        let v_im = super::matvec::coulomb_contribution(ri_ov_a, &z_im.to_vec());
        for i in 0..result_re.len() {
            result_re[i] += scale * v_re[i];
            result_im[i] += scale * v_im[i];
        }
    }

    (result_re, result_im)
}

// ============================================================================
// 复合频率 matvec: y = ((A+B)(A-B) - z²I) · x
// ============================================================================

/// 复合 matvec: y = ((A+B)(A-B) - z²I) · x
///
/// 数学推导：
///   设 z = ω + iη, 则 z² = (ω²-η²) + i·(2ωη)
///   z²I 的实嵌入形式 (x = x_re + i·x_im):
///
///     z² · x = [(ω²-η²)·x_re − 2ωη·x_im]  +  i·[2ωη·x_re + (ω²-η²)·x_im]
///
///   完整算子实嵌入为 2N×2N 实矩阵:
///
///     [ (A+B)(A-B) 的实部    −(A+B)(A-B)的虚部 ]   [ (ω²-η²)I    −2ωη·I ]
///     [ (A+B)(A-B) 的虚部     (A+B)(A-B)的实部 ] − [  2ωη·I    (ω²-η²)I ]
///
///   计算分为三步 (纯实数运算):
///     1. t  = (A-B) · x           (2 次复 matvec)
///     2. s  = (A+B) · t           (2 次复 matvec)
///     3. y  = s − z² · x          (元素级减法)
pub fn composite_matvec_complex(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    ri_vv: &MatrixFull<f64>,
    ri_ov_a: &MatrixFull<f64>,
    ri_ov_b: &MatrixFull<f64>,
    ri_oo_tilde_re: &MatrixFull<f64>,
    ri_oo_tilde_im: &MatrixFull<f64>,
    ri_ov_tilde_re: &MatrixFull<f64>,
    ri_ov_tilde_im: &MatrixFull<f64>,
    x_re: &[f64],
    x_im: &[f64],
    omega: Complex<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let n = x_re.len();

    // ── Step 1: t = (A-B)·x ──
    let (a_re, a_im) = a_block_matvec_complex(
        scf_data, qp_ctrl, ri_vv, ri_ov_a,
        ri_oo_tilde_re, ri_oo_tilde_im,
        x_re, x_im,
    );
    let (b_re, b_im) = b_block_matvec_complex(
        scf_data, qp_ctrl, ri_ov_a, ri_ov_b,
        ri_ov_tilde_re, ri_ov_tilde_im,
        x_re, x_im,
    );
    let mut t_re = Vec::with_capacity(n);
    let mut t_im = Vec::with_capacity(n);
    for i in 0..n {
        t_re.push(a_re[i] - b_re[i]);
        t_im.push(a_im[i] - b_im[i]);
    }

    // ── Step 2: s = (A+B)·t ──
    let (a_s_re, a_s_im) = a_block_matvec_complex(
        scf_data, qp_ctrl, ri_vv, ri_ov_a,
        ri_oo_tilde_re, ri_oo_tilde_im,
        &t_re, &t_im,
    );
    let (b_s_re, b_s_im) = b_block_matvec_complex(
        scf_data, qp_ctrl, ri_ov_a, ri_ov_b,
        ri_ov_tilde_re, ri_ov_tilde_im,
        &t_re, &t_im,
    );
    let mut s_re = Vec::with_capacity(n);
    let mut s_im = Vec::with_capacity(n);
    for i in 0..n {
        s_re.push(a_s_re[i] + b_s_re[i]);
        s_im.push(a_s_im[i] + b_s_im[i]);
    }

    // ── Step 3: y = s − z²·x ──
    //   z²·x = (ω²-η²)·x_re − 2ωη·x_im  +  i·(2ωη·x_re + (ω²-η²)·x_im)
    let omega_re = omega.re;
    let omega_im = omega.im;
    let a2mb2 = omega_re * omega_re - omega_im * omega_im;
    let two_ab = 2.0 * omega_re * omega_im;

    let mut y_re = Vec::with_capacity(n);
    let mut y_im = Vec::with_capacity(n);
    for i in 0..n {
        y_re.push(s_re[i] - a2mb2 * x_re[i] + two_ab * x_im[i]);
        y_im.push(s_im[i] - two_ab * x_re[i] - a2mb2 * x_im[i]);
    }

    (y_re, y_im)
}

// ============================================================================
// 实频率复合 matvec: y = ((A+B)(A-B) - ω²I) · x
// ============================================================================

/// 实频率复合 matvec: y = ((A+B)(A-B) - ω²I) · x
///
/// 使用 matvec.rs 中的实数 a_block_matvec / b_block_matvec。
/// 计算步骤:
///   1. t  = (A-B)·x  = A·x − B·x
///   2. s  = (A+B)·t  = A·t + B·t
///   3. y  = s − ω²·x
pub fn composite_matvec_real(
    scf_data: &SCF,
    qp_ctrl: &QuasiParticle,
    ri_vv: &MatrixFull<f64>,
    ri_ov_a: &MatrixFull<f64>,
    ri_ov_b: &MatrixFull<f64>,
    ri_oo_tilde: &MatrixFull<f64>,
    ri_ov_tilde: &MatrixFull<f64>,
    x: &[f64],
    omega: f64,
) -> Vec<f64> {
    let n = x.len();

    // Step 1: t = (A-B)·x
    let a_x = super::matvec::a_block_matvec(scf_data, qp_ctrl, ri_vv, ri_ov_a, ri_oo_tilde, &x.to_vec());
    let b_x = super::matvec::b_block_matvec(scf_data, qp_ctrl, ri_ov_a, ri_ov_b, ri_ov_tilde, &x.to_vec());
    let t: Vec<f64> = a_x.iter().zip(b_x.iter()).map(|(a, b)| a - b).collect();

    // Step 2: s = (A+B)·t
    let a_t = super::matvec::a_block_matvec(scf_data, qp_ctrl, ri_vv, ri_ov_a, ri_oo_tilde, &t);
    let b_t = super::matvec::b_block_matvec(scf_data, qp_ctrl, ri_ov_a, ri_ov_b, ri_ov_tilde, &t);
    let s: Vec<f64> = a_t.iter().zip(b_t.iter()).map(|(a, b)| a + b).collect();

    // Step 3: y = s − ω²·x
    let omega2 = omega * omega;
    s.iter().zip(x.iter()).map(|(s_i, x_i)| s_i - omega2 * x_i).collect()
}