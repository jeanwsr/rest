// ============================================================================
// 2.5D MPI 下的 dRPA 关联能驱动
// ============================================================================
//
// 目的
// ----
// 为 DFAFamily::RPA（rpa@pbe / rpa@b3lyp 等，入口 ri_rpa::rpa_calculations）
// 提供第一条 MPI 路径（线性归属重分布，持久内存 ≈ M_total/p）。此前
// rpa_calculations 在 MPI 下直接 panic（Phase 0 护栏）：串行响应内核
// evaluate_response_serial 按全量张量索引编写，与 MPI 下 aux-分布 ri3mo 不兼容。
//
// 数学结构
// --------
// 与 SCSRPA 同为单占据指标求和：χ₀(ω) = Σ_σ Σ_{j∈occ^σ} B_j(ω)B_j(ω)^T。
// 差异在于：① 两自旋响应**累加进同一矩阵**（evaluate_response_serial 的行为）；
// ② 闭壳层（spin_channel==1）响应 ×2；③ integrand 只含 dRPA 项（log|det| + Tr），
// 无谱半径/λ-积分/级数分支。
//
// 数值约定（逐字镜像串行 evaluate_response_serial，ri_rpa/mod.rs:373-454）
// ----------------------------------------------------------------
// - 能隙夹逼：gap ∈ (0,1e-6) → +1e-6；gap ∈ (-1e-6,0) → -1e-6（**严格不等号**，
//   与 scsrpa 的 >=/<= 略有不同，gap==0 不夹逼 → ζ=0）；
// - ζ = 2·gap/(gap²+ω²)·(j_occ·num_spin/2)·(1-k_occ·num_spin/2)（Yang 系综公式）；
// - **无** de-excitation level shift、**无**显式占据过滤（占据信息全在 ζ 因子里，
//   occ=0 → ζ=0 → 列为 0，与串行逐位一致）；
// - num_occ = homo+1（num_elec≤1e-6 时为 0）；ROHF 用 semi_eigenvalues。
//
// 改动纪律：串行路径零改动；本模块仅在 cfg(feature="mpi") 下编译。
// ============================================================================

#[cfg(feature = "mpi")]
use crate::mpi_io::MPIOperator;
#[cfg(feature = "mpi")]
use crate::scf_io::{SCF, SCFType};
#[cfg(feature = "mpi")]
use tensors::{MatrixFull, BasicMatrix};

#[cfg(feature = "mpi")]
use super::scsrpa_25d::build_freq_grids;
#[cfg(feature = "mpi")]
use super::{evaluate_rpa_integrand};

#[cfg(feature = "mpi")]
use crate::constants::PI;

/// 本地 partial 响应（dRPA 版）：对 ownership 块内每个占据 j，
/// polar_partial += tmp(ω,j)·rimo_j^T。逐字镜像 evaluate_response_serial 的
/// j 循环与 ζ 公式；差别仅在 j 的枚举范围由 ownership 块划分。
#[cfg(feature = "mpi")]
fn local_response_25d(
    final_tensor: &rest_tensors::RIFull<f64>,
    ctx: &crate::ri_pt2::pt2_25d::Ctx25dBlock,
    grid_rank: usize,
    n2_global: usize,
    freq: f64,
    eigenvalues: &Vec<f64>,
    occupation: &Vec<f64>,
    occ_range_start: usize,  // occ_range.start（= start_mo）
    virt_range_start: usize, // vir_range.start（= lumo）
    num_occ: usize,          // 该自旋 homo+1（无电子自旋为 0）
    num_state: usize,
    frac_spin_occ: f64,
    polar_partial: &mut MatrixFull<f64>,
) {
    use tensors::matrix_blas_lapack::_dgemm_full;
    let block_start_idx = &ctx.block_start_idx;
    let num_block = block_start_idx.len();
    let n0_global = final_tensor.size[0];
    let n1_global = final_tensor.size[1];

    let mut tmp_matrix = MatrixFull::new([n0_global, n1_global], 0.0);

    for &blk in &ctx.ownership[grid_rank] {
        let blk_start = block_start_idx[blk];
        let blk_end = if blk == num_block - 1 { n2_global - 1 } else { block_start_idx[blk + 1] - 1 };
        for j_loc in blk_start..=blk_end {
            let j_state = occ_range_start + j_loc;
            if j_state >= num_occ {
                continue;
            }
            let j_state_eigen = eigenvalues[j_state];
            let j_state_occ = occupation[j_state];
            let rimo_j = match final_tensor.get_reducing_matrix_global_n2(j_loc) {
                Some(m) => m,
                None => continue,
            };
            // 串行代码对每个 j 重新开零矩阵：未被写入的 k 列须为 0
            tmp_matrix.data.iter_mut().for_each(|x| *x = 0.0);
            for k_state in virt_range_start..num_state {
                let k_state_eigen = eigenvalues[k_state];
                let k_state_occ = occupation[k_state];
                let mut energy_gap = j_state_eigen - k_state_eigen;
                if energy_gap < 1.0e-6 && energy_gap > 0.0 {
                    energy_gap += 1.0e-6;
                } else if energy_gap > -1.0e-6 && energy_gap < 0.0 {
                    energy_gap += -1.0e-6;
                };
                let zeta = 2.0f64 * energy_gap
                    / (energy_gap.powf(2.0) + freq * freq)
                    * j_state_occ
                    * frac_spin_occ
                    * (1.0f64 - k_state_occ * frac_spin_occ);
                let k_loc_state = k_state - virt_range_start;
                let col_start = k_loc_state * n0_global;
                let from_col = &rimo_j.data[col_start..col_start + n0_global];
                let to_iter = tmp_matrix.iter_submatrix_mut(0..n0_global, k_loc_state..k_loc_state + 1);
                to_iter.zip(from_col.iter()).for_each(|(to, from)| {
                    *to = *from * zeta
                });
            }
            _dgemm_full(&tmp_matrix, 'N', &rimo_j, 'T', polar_partial, 1.0, 1.0);
        }
    }
}

/// dRPA 关联能 2.5D MPI 驱动。返回 Ec[RPA]（未含任何杂化泛函组合——
/// 与串行 evaluate_rpa_correlation_rayon 同约定）。
#[cfg(feature = "mpi")]
pub fn rpa_correlation_rayon_mpi_25d(
    scf_data: &SCF,
    mpi_operator: &Option<MPIOperator>,
) -> anyhow::Result<f64> {
    use mpi::collective::SystemOperation;
    use mpi::traits::*;
    use crate::ri_pt2::pt2_25d::{initialize_metadata, swap_ownership};

    let (mpi_op, mpi_ix) = match (mpi_operator, &scf_data.mol.mpi_data) {
        (Some(op), Some(ix)) => (op, ix),
        _ => panic!("MPI not initialized for the 2.5d RPA evaluation"),
    };
    let my_rank = mpi_ix.rank;
    let local_n0_range = if let Some(loc_auxbas) = &mpi_ix.auxbas {
        loc_auxbas[my_rank].clone()
    } else {
        panic!("Memory distribution should be initialized for the auxiliary basis sets before post-SCF calculations")
    };

    let spin_channel = scf_data.mol.spin_channel;
    let num_state = scf_data.mol.num_state;
    let frac_spin_occ = spin_channel as f64 / 2.0f64;

    // ROHF 使用 semi-canonical 本征值（与串行一致）
    let semi = if let SCFType::ROHF = scf_data.scftype {
        Some(scf_data.semi_eigenvalues.as_ref().unwrap())
    } else {
        None
    };
    let eigenvalues: Vec<&Vec<f64>> = (0..spin_channel)
        .map(|i_spin| match &semi {
            Some(s) => &s[i_spin],
            None => &scf_data.eigenvalues[i_spin],
        })
        .collect();

    let grid = mpi_op.initialize_grid();
    let ri3mo_vec = match &scf_data.ri3mo {
        Some(v) => v,
        None => panic!("RI3MO should be initialized before the 2.5d RPA evaluation"),
    };
    let (alpha_rimo, vir_range, occ_range) = &ri3mo_vec[0];
    let beta_rimo = if spin_channel == 2 { Some(&ri3mo_vec[1].0) } else { None };
    let n0_local = alpha_rimo.size[0];
    let n1_global = alpha_rimo.size[1];
    let n2_global = alpha_rimo.size[2];

    let ctx = initialize_metadata(&grid, n2_global);
    let mut n0_global_tmp: u64 = 0;
    grid.cart_comm
        .all_reduce_into(&(n0_local as u64), &mut n0_global_tmp, &SystemOperation::sum());
    let n0_global = n0_global_tmp as usize;

    let redistributed_alpha =
        swap_ownership(&grid, &ctx, alpha_rimo, n0_global, n1_global, n2_global, &local_n0_range);
    let redistributed_beta = match beta_rimo {
        Some(b) => Some(swap_ownership(&grid, &ctx, b, n0_global, n1_global, n2_global, &local_n0_range)),
        None => None,
    };

    let num_occ: Vec<usize> = (0..spin_channel)
        .map(|i_spin| {
            if scf_data.mol.num_elec[i_spin + 1] <= 1.0e-6 {
                0
            } else {
                scf_data.homo[i_spin] + 1
            }
        })
        .collect();

    let occ_start = occ_range.start;
    let vir_start = vir_range.start;
    let grids = build_freq_grids(scf_data);

    let use_distributed = rpa_use_distributed(scf_data, mpi_op, n0_global);
    #[cfg(feature = "scalapack")]
    let nb_distributed = {
        let blac = &mpi_op.cblacsgrid;
        rpa_distributed_block_size(n0_global, blac.nprow, blac.npcol)
    };

    let mut rpa_c_energy = 0.0_f64;
    for (omega, weight) in grids.omega.iter().zip(grids.weight.iter()) {
        let integrand = if use_distributed {
            #[cfg(feature = "scalapack")]
            {
                rpa_integrand_distributed_25d(
                    scf_data, mpi_op,
                    &redistributed_alpha, redistributed_beta.as_ref(),
                    &ctx, grid.rank as usize, n2_global, *omega,
                    &eigenvalues, &scf_data.occupation,
                    occ_start, vir_start, &num_occ, num_state,
                    frac_spin_occ, n0_global, nb_distributed,
                )
            }
            #[cfg(not(feature = "scalapack"))]
            {
                unreachable!("use_distributed requires the scalapack feature")
            }
        } else {
            // 两自旋响应累加进同一 partial（镜像串行单矩阵累加行为）
            let mut partial = MatrixFull::new([n0_global, n0_global], 0.0);
            for i_spin in 0..spin_channel {
                let tensor = if i_spin == 0 { &redistributed_alpha } else { redistributed_beta.as_ref().unwrap() };
                local_response_25d(
                    tensor, &ctx, grid.rank as usize, n2_global, *omega,
                    eigenvalues[i_spin], &scf_data.occupation[i_spin],
                    occ_start, vir_start, num_occ[i_spin], num_state,
                    frac_spin_occ, &mut partial,
                );
            }
            let mut reduced = vec![0.0f64; partial.data.len()];
            grid.cart_comm
                .any_process()
                .all_reduce_into(&partial.data[..], &mut reduced[..], &SystemOperation::sum());
            let mut polar = MatrixFull::new([n0_global, n0_global], 0.0);
            polar.data.copy_from_slice(&reduced);

            // 闭壳层 ×2（镜像串行 evaluate_rpa_correlation_rayon 的 response_freq *= 2.0）
            if spin_channel == 1 {
                polar *= 2.0;
            }
            evaluate_rpa_integrand(&mut polar)
        };

        if scf_data.mol.ctrl.print_level > 1 {
            println!(
                " (freq, weight, rpa_c): {:16.8},{:16.8},{:16.8}",
                omega, weight, integrand
            );
        }
        rpa_c_energy += integrand * weight;
    }

    rpa_c_energy *= 0.5 / PI;
    Ok(rpa_c_energy)
}

// ============================================================================
// ScaLAPACK 分布式响应（dRPA，cfg(feature="scalapack")）
// ============================================================================
// 目标：消除复制式 [naux²] 响应矩阵与冗余的每频率 dgetrf（naux ≥ 10⁴ 时的内存地板）。
//
// 构造：线性归属下每 rank 持其占据块的完整 [naux×nvir] 切片；χ̃₀ 按 BLACS 行带
// （块循环行，行数 ≈ naux/nprow）本地累加 partial → 全网格 allreduce → 属主
// (myrow==band) 抽出本列块循环切片，组装成 DistributedMatrixFull（局部 naux²/p）。
// 每频率通信总量 ≈ naux²/rank（与全矩阵 allreduce 同阶），但常驻内存从 ~4-8·naux²
// 降到 naux²/nprow（band）+ naux²/p（循环块）。
//
// integrand：trace（预处理前局部对角和，全网格归约）→ 本地 (diag−1)·(−1) 预处理 →
// pdgetrf → 局部 ln|det U| 归约。det 符号取 |·|，与串行 abs(log|det|) 语义一致
// （pdgetrf 含部分主元，|det| = Π|U_ii| 不受主元符号影响）。
// ============================================================================

#[cfg(feature = "scalapack")]
use tensors::matrix::distributedmatrixfull::{
    DistributedMatrixFull, pdgetrf, pdgetrf_ipiv_len, pdgetrf_local_lnabsdet,
    pd_numroc_iproc, pd_neg_shift_local, pd_trace_local,
};

/// 是否走分布式响应路径（cfg scalapack 时才可能为 true）。
#[cfg(feature = "scalapack")]
fn rpa_use_distributed(scf_data: &SCF, mpi_op: &MPIOperator, naux_global: usize) -> bool {
    use crate::ctrl_io::HamiltonianDistributedMode;
    match scf_data.mol.ctrl.rpa_distributed {
        HamiltonianDistributedMode::On => true,
        HamiltonianDistributedMode::Off => false,
        HamiltonianDistributedMode::Auto => naux_global >= 8192 && mpi_op.size >= 32,
    }
}

#[cfg(not(feature = "scalapack"))]
fn rpa_use_distributed(_scf_data: &SCF, _mpi_op: &MPIOperator, _naux_global: usize) -> bool {
    false
}

/// 行带受限本地响应：与 local_response_25d 同一 ζ 数学，但只算 BLACS 行带
/// `i_band`（块循环行 r：r = (b·nprow+i_band)·nb+off）对应的 partial 行。
#[cfg(feature = "scalapack")]
fn local_response_band_25d(
    final_tensor: &rest_tensors::RIFull<f64>,
    ctx: &crate::ri_pt2::pt2_25d::Ctx25dBlock,
    grid_rank: usize,
    n2_global: usize,
    freq: f64,
    eigenvalues: &Vec<f64>,
    occupation: &Vec<f64>,
    occ_range_start: usize,
    virt_range_start: usize,
    num_occ: usize,
    num_state: usize,
    frac_spin_occ: f64,
    i_band: i32,
    nb: i32,
    naux_global: usize,
    nprow: i32,
    band: &mut MatrixFull<f64>,
) {
    use tensors::matrix_blas_lapack::_dgemm_full;
    let block_start_idx = &ctx.block_start_idx;
    let num_block = block_start_idx.len();
    let n1_global = final_tensor.size[1];
    let n_rows_band = band.size[0];
    let mut tmp_band = MatrixFull::new([n_rows_band, n1_global], 0.0);
    // 预计算本行带的全局行列表（aux 行，升序）
    let mut band_rows: Vec<usize> = Vec::with_capacity(n_rows_band);
    let n_blocks = (naux_global as i32 + nb - 1) / nb;
    for b in 0..n_blocks {
        if b % nprow == i_band {
            let r0 = (b * nb) as usize;
            let r1 = ((b + 1) * nb).min(naux_global as i32) as usize;
            for r in r0..r1 {
                band_rows.push(r);
            }
        }
    }
    assert_eq!(band_rows.len(), n_rows_band);

    for &blk in &ctx.ownership[grid_rank] {
        let blk_start = block_start_idx[blk];
        let blk_end = if blk == num_block - 1 { n2_global - 1 } else { block_start_idx[blk + 1] - 1 };
        for j_loc in blk_start..=blk_end {
            let j_state = occ_range_start + j_loc;
            if j_state >= num_occ {
                continue;
            }
            let j_state_eigen = eigenvalues[j_state];
            let j_state_occ = occupation[j_state];
            let rimo_j = match final_tensor.get_reducing_matrix_global_n2(j_loc) {
                Some(m) => m,
                None => continue,
            };
            tmp_band.data.iter_mut().for_each(|x| *x = 0.0);
            // 逐虚轨道 k：zeta 缩放行带局部列（rimo_j 为 [naux×nvir] 列主序）
            for k_state in virt_range_start..num_state {
                let k_state_eigen = eigenvalues[k_state];
                let k_state_occ = occupation[k_state];
                let mut energy_gap = j_state_eigen - k_state_eigen;
                if energy_gap < 1.0e-6 && energy_gap > 0.0 {
                    energy_gap += 1.0e-6;
                } else if energy_gap > -1.0e-6 && energy_gap < 0.0 {
                    energy_gap += -1.0e-6;
                };
                let zeta = 2.0f64 * energy_gap
                    / (energy_gap.powf(2.0) + freq * freq)
                    * j_state_occ
                    * frac_spin_occ
                    * (1.0f64 - k_state_occ * frac_spin_occ);
                let k_loc = k_state - virt_range_start;
                let src_col = &rimo_j.data[k_loc * naux_global..k_loc * naux_global + naux_global];
                let to_iter = tmp_band.iter_submatrix_mut(0..n_rows_band, k_loc..k_loc + 1);
                to_iter.zip(band_rows.iter()).for_each(|(to, &r)| {
                    *to = src_col[r] * zeta
                });
            }
            _dgemm_full(&tmp_band, 'N', &rimo_j, 'T', band, 1.0, 1.0);
        }
    }
}

/// 单频率分布式 integrand：χ̃₀ 行带构造 + 循环组装 + pdgetrf 的 ln|det| + trace。
/// 返回该频率 integrand（所有 rank 一致）。
#[cfg(feature = "scalapack")]
fn rpa_integrand_distributed_25d(
    scf_data: &SCF,
    mpi_op: &crate::mpi_io::MPIOperator,
    redistributed_alpha: &rest_tensors::RIFull<f64>,
    redistributed_beta: Option<&rest_tensors::RIFull<f64>>,
    ctx: &crate::ri_pt2::pt2_25d::Ctx25dBlock,
    grid_rank: usize,
    n2_global: usize,
    omega: f64,
    eigenvalues: &Vec<&Vec<f64>>,
    occupation: &[Vec<f64>],
    occ_start: usize,
    vir_start: usize,
    num_occ: &Vec<usize>,
    num_state: usize,
    frac_spin_occ: f64,
    naux_global: usize,
    nb: i32,
) -> f64 {
    use mpi::collective::SystemOperation;
    use mpi::traits::*;
    let spin_channel = scf_data.mol.spin_channel;
    let blac = &mpi_op.cblacsgrid;
    let nprow = blac.nprow;
    let naux = naux_global as i32;
    let mut chi = DistributedMatrixFull::new(blac, naux, naux, nb, nb, 0, 0, 0.0_f64);
    let local_rows_chi = chi.size()[0];

    for i_band in 0..nprow {
        let n_rows = pd_numroc_iproc(naux, nb, i_band, nprow);
        let mut band = MatrixFull::new([n_rows, naux_global], 0.0);
        for i_spin in 0..spin_channel {
            let tensor = if i_spin == 0 {
                redistributed_alpha
            } else {
                redistributed_beta.expect("beta tensor missing")
            };
            local_response_band_25d(
                tensor, ctx, grid_rank, n2_global, omega,
                eigenvalues[i_spin], &occupation[i_spin],
                occ_start, vir_start, num_occ[i_spin], num_state,
                frac_spin_occ, i_band, nb, naux_global, nprow,
                &mut band,
            );
        }
        if spin_channel == 1 {
            band *= 2.0;
        }
        // 行带跨全网格归约（每个 occ 属主都贡献了本带）
        let mut reduced = vec![0.0f64; band.data.len()];
        mpi_op
            .world
            .any_process()
            .all_reduce_into(&band.data[..], &mut reduced[..], &SystemOperation::sum());
        band.data.copy_from_slice(&reduced);

        if blac.myrow == i_band {
            // 抽出本 mycol 的列块循环切片 → chi 局部块（行数一致：myrow==i_band）
            let local_cols_chi = chi.size()[1];
            let npcol = blac.npcol;
            let n_col_blocks = (naux + nb - 1) / nb;
            for ic in 0..local_cols_chi {
                // 逆映射：局部列 ic → 全局列 g_col
                let bj = (ic as i32) / nb;
                let c_off = (ic as i32) % nb;
                let g_col = (bj * npcol + blac.mycol) * nb + c_off;
                let col = &band.data[(g_col as usize) * n_rows..(g_col as usize) * n_rows + n_rows];
                let dst = &mut chi.data[ic * local_rows_chi..(ic + 1) * local_rows_chi];
                dst.copy_from_slice(col);
            }
            let _ = n_col_blocks;
        }
    }

    let tr_loc = pd_trace_local(blac, &chi, nb, nb);
    let mut trace = 0.0_f64;
    mpi_op.world.any_process().all_reduce_into(&tr_loc, &mut trace, &SystemOperation::sum());

    pd_neg_shift_local(blac, &mut chi, nb, nb);
    let mut ipiv = vec![0i32; pdgetrf_ipiv_len(blac, naux, nb)];
    let info = pdgetrf(blac, &mut chi, 1, 1, &mut ipiv);
    if info > 0 && scf_data.mol.ctrl.print_level > 1 {
        eprintln!("WARNING: pdgetrf returned info = {} (distributed dRPA)", info);
    }
    let ln_loc = pdgetrf_local_lnabsdet(blac, &chi, nb, nb);
    let mut ln_det = 0.0_f64;
    mpi_op.world.any_process().all_reduce_into(&ln_loc, &mut ln_det, &SystemOperation::sum());

    ln_det + trace
}

/// 与驱动内循环的辅助函数（scalapack 编译门内使用）。
#[cfg(feature = "scalapack")]
fn rpa_distributed_block_size(naux: usize, nprow: i32, npcol: i32) -> i32 {
    let denom = 2 * nprow.max(npcol);
    (256i32).min((naux as i32) / denom).max(1)
}
