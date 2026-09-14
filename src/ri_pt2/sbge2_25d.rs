// ============================================================================
// 2.5D MPI 下的 SBGE2（ZRPS 双杂化相关能）本地批内核
// ============================================================================
//
// 目的
// ----
// 让 SBGE2 复用 ri_pt2/pt2_25d.rs 的 2.5D 重分布框架（swap_ownership：
// ri3mo 从 aux 索引 1D 分布重排为占据索引对角归属，每进程持完整 n_aux×n_vir
// 切片，内存 ≈ 总张量/√p，p=进程数），从而在大体系 MPI 下获得与 PT2-25d
// 相同的内存路线（close_shell_pt2_rayon_mpi_25d 目前对 SBGE2 是 unreachable）。
//
// 数值约定
// --------
// 本模块逐字镜像串行 close_shell_sbge2_detailed_rayon（ri_pt2/sbge2.rs）的
// 每 (i,j) 电子对能量公式：本地 dgemm 得完整 eri_virt → 构建 denominator
// [double_gap, occ_state] → BGE2 电子对自洽迭代（iterator_close_shell_eij_serial，
// enhanced=1/screening=1/shifted=0，阈值 1e-8/100 步）→ i≠j 加倍。
// (i,j) 无序对由 2D 网格按 (row_blk mod r, col_blk mod c) 唯一划分，各 rank
// 只算自己的对、全程零通信，末端一次标量 allreduce——因此逐对结果与串行
// 逐位一致（求和顺序在 rank 内的对枚举顺序上与串行不同，只影响末位）。
//
// 改动纪律：串行路径零改动；仅新增 2.5D 路径（cfg(feature="mpi") 内部）。
// ============================================================================

use crate::mpi_io::MPIOperator;
use crate::scf_io::SCF;

#[cfg(feature = "mpi")]
fn local_computation_close_shell_bge2(
    final_tensor: &rest_tensors::RIFull<f64>,
    ctx: &crate::ri_pt2::pt2_25d::Ctx25dBlock,
    n2_global: usize,
    eigenvalues: &Vec<f64>,
    occupation: &Vec<f64>,
    occ_offset: usize,
    virt_offset: usize,
) -> (f64, f64) {
    let row_block_idx = &ctx.row_block_idx;
    let col_block_idx = &ctx.col_block_idx;
    let block_start_idx = &ctx.block_start_idx;
    let num_block = block_start_idx.len();
    let n0_global = final_tensor.size[0];
    let n1_global = final_tensor.size[1];

    let enhanced_factor = 1.0_f64;
    let screening_factor = 1.0_f64;
    let shifted_factor = 0.0_f64;
    let lumo = virt_offset; // 全局第一个虚轨道 == vir_range.start
    let lumo_min = virt_offset;

    let mut local_bge2_os = 0.0_f64;
    let mut local_bge2_ss = 0.0_f64;

    let mut eri_virt = rest_tensors::MatrixFull::new([n1_global, n1_global], 0.0_f64);
    let mut denominator = rest_tensors::MatrixFull::new([n1_global, n1_global], [0.0_f64; 2]);

    for &row_blk in row_block_idx {
        let row_start = block_start_idx[row_blk];
        let row_end = if row_blk == num_block - 1 { n2_global - 1 } else { block_start_idx[row_blk + 1] - 1 };
        for &col_blk in col_block_idx {
            let col_start = block_start_idx[col_blk];
            let col_end = if col_blk == num_block - 1 { n2_global - 1 } else { block_start_idx[col_blk + 1] - 1 };
            if row_blk > col_blk {
                continue;
            } else if row_blk < col_blk {
                // 跨块：必有 i < j
                assert!(row_end < col_start);
                for i in row_start..=row_end {
                    let ri_i = final_tensor.get_reducing_matrix_global_n2(i).unwrap();
                    let i_state_eigen = eigenvalues[i + occ_offset];
                    let i_state_occ = occupation[i + occ_offset] / 2.0;
                    if i_state_occ.abs() <= 1.0e-6 {
                        continue;
                    }
                    for j in col_start..=col_end {
                        let ri_j = final_tensor.get_reducing_matrix_global_n2(j).unwrap();
                        let j_state_eigen = eigenvalues[j + occ_offset];
                        let j_state_occ = occupation[j + occ_offset] / 2.0;
                        if j_state_occ.abs() <= 1.0e-6 {
                            continue;
                        }
                        // eri_virt = ri_i^T · ri_j（全 n0_global × n1_global，本地无通信）
                        rest_tensors::matrix_blas_lapack::_dgemm(
                            &ri_i, (0..n0_global, 0..n1_global), 'T',
                            &ri_j, (0..n0_global, 0..n1_global), 'N',
                            &mut eri_virt, (0..n1_global, 0..n1_global),
                            1.0, 0.0,
                        );
                        let (e_ss, e_os, _mp2_ss, _mp2_os) = sbge2_pair_close_shell(
                            &eri_virt, &mut denominator, eigenvalues, occupation,
                            i_state_eigen, j_state_eigen, i_state_occ, j_state_occ,
                            lumo, lumo_min, n1_global,
                            enhanced_factor, screening_factor, shifted_factor,
                        );
                        local_bge2_ss -= 2.0 * e_ss; // i<j：串行在 pair 外 ×2
                        local_bge2_os -= 2.0 * e_os;
                    }
                }
            } else {
                // 同块：i <= j
                for i in row_start..=row_end {
                    let ri_i = final_tensor.get_reducing_matrix_global_n2(i).unwrap();
                    let i_state_eigen = eigenvalues[i + occ_offset];
                    let i_state_occ = occupation[i + occ_offset] / 2.0;
                    if i_state_occ.abs() <= 1.0e-6 {
                        continue;
                    }
                    for j in col_start..=row_end.min(col_end) {
                        if i > j {
                            continue;
                        }
                        let ri_j = final_tensor.get_reducing_matrix_global_n2(j).unwrap();
                        let j_state_eigen = eigenvalues[j + occ_offset];
                        let j_state_occ = occupation[j + occ_offset] / 2.0;
                        if j_state_occ.abs() <= 1.0e-6 {
                            continue;
                        }
                        rest_tensors::matrix_blas_lapack::_dgemm(
                            &ri_i, (0..n0_global, 0..n1_global), 'T',
                            &ri_j, (0..n0_global, 0..n1_global), 'N',
                            &mut eri_virt, (0..n1_global, 0..n1_global),
                            1.0, 0.0,
                        );
                        let off_diag = i != j;
                        let (e_ss, e_os, _mp2_ss, _mp2_os) = sbge2_pair_close_shell(
                            &eri_virt, &mut denominator, eigenvalues, occupation,
                            i_state_eigen, j_state_eigen, i_state_occ, j_state_occ,
                            lumo, lumo_min, n1_global,
                            enhanced_factor, screening_factor, shifted_factor,
                        );
                        local_bge2_ss -= if off_diag { 2.0 * e_ss } else { e_ss };
                        local_bge2_os -= if off_diag { 2.0 * e_os } else { e_os };
                    }
                }
            }
        }
    }
    (local_bge2_os, local_bge2_ss)
}

/// 单个 (i,j) 电子对的 sBGE2 能量（正数；调用方负责符号与 off-diag ×2）。
/// 镜像 ri_pt2/sbge2.rs close_shell_sbge2_detailed_rayon 中
/// 「denominator 构建 + e_mp2 初始化 + BGE2 自洽迭代」三段（L99-151）。
#[cfg(feature = "mpi")]
fn sbge2_pair_close_shell(
    eri_virt: &rest_tensors::MatrixFull<f64>,
    denominator: &mut rest_tensors::MatrixFull<[f64; 2]>,
    eigenvalues: &Vec<f64>,
    occupation: &Vec<f64>,
    i_state_eigen: f64,
    j_state_eigen: f64,
    i_state_occ: f64,
    j_state_occ: f64,
    lumo: usize,
    lumo_min: usize,
    n1_global: usize,
    enhanced_factor: f64,
    screening_factor: f64,
    shifted_factor: f64,
) -> (f64, f64, f64, f64) {
    let ij_state_eigen = i_state_eigen + j_state_eigen;
    let mut e_mp2_ss = 0.0_f64;
    let mut e_mp2_os = 0.0_f64;

    // denominator 与 MP2 初始化（全虚方阵，含占据过滤；与串行一致）
    for i_loc_virt in 0..n1_global {
        let i_virt_eigen = eigenvalues[i_loc_virt + lumo];
        let i_virt_occ = occupation[i_loc_virt + lumo] / 2.0;
        if (1.0 - i_virt_occ).abs() <= 1.0e-6 {
            continue;
        }
        for j_loc_virt in 0..n1_global {
            let j_virt_eigen = eigenvalues[j_loc_virt + lumo];
            let j_virt_occ = occupation[j_loc_virt + lumo] / 2.0;
            if (1.0 - j_virt_occ).abs() <= 1.0e-6 {
                denominator[[i_loc_virt, j_loc_virt]] = [0.0, 0.0];
                continue;
            }
            let mut double_gap = i_virt_eigen + j_virt_eigen - ij_state_eigen;
            if double_gap.abs() <= 1.0e-6 {
                double_gap = 1.0e-6;
            }
            let occ_state = i_state_occ * j_state_occ * (1.0 - i_virt_occ) * (1.0 - j_virt_occ);
            let e_ab = eri_virt[[i_loc_virt, j_loc_virt]];
            let e_ba = eri_virt[[j_loc_virt, i_loc_virt]];
            e_mp2_ss += (e_ab - e_ba) * e_ab / double_gap * occ_state;
            e_mp2_os += e_ab * e_ab / double_gap * occ_state;
            denominator[[i_loc_virt, j_loc_virt]] = [double_gap, occ_state];
        }
    }

    // BGE2 电子对自洽迭代（阈值 1e-8 / 100 步，与串行完全同一实现）
    let (e_eij_ss, e_eij_os, _num_iter) = crate::ri_pt2::sbge2::iterator_close_shell_eij_serial(
        eri_virt, denominator, e_mp2_ss, e_mp2_os,
        enhanced_factor, screening_factor, shifted_factor,
        lumo, lumo_min, lumo + n1_global,
    );

    // 清理：denominator 只写命中项，未命中项在上方已显式清零
    for i_loc_virt in 0..n1_global {
        for j_loc_virt in 0..n1_global {
            denominator[[i_loc_virt, j_loc_virt]] = [0.0, 0.0];
        }
    }
    (e_eij_ss, e_eij_os, e_mp2_ss, e_mp2_os)
}

/// 2.5D close-shell sBGE2（ZRPS）MPI 驱动。
/// 结构镜像 ri_pt2/mod.rs close_shell_pt2_rayon_mpi_25d；仅内核换成 BGE2。
pub fn close_shell_sbge2_rayon_mpi_25d(
    scf_data: &SCF,
    mpi_operator: &Option<MPIOperator>,
) -> anyhow::Result<[f64; 3]> {
    #[cfg(feature = "mpi")]
    {
        if let (Some(mpi_op), Some(mpi_ix)) = (&mpi_operator, &scf_data.mol.mpi_data) {
            use mpi::collective::SystemOperation;
            use mpi::traits::*;
            use crate::ri_pt2::pt2_25d::{initialize_metadata, swap_ownership};

            let my_rank = mpi_ix.rank;
            let local_n0_range = if let Some(loc_auxbas) = &mpi_ix.auxbas {
                loc_auxbas[my_rank].clone()
            } else {
                panic!("Memory distribution should be initialized for the auxiliary basis sets before post-SCF calculations")
            };

            let grid = mpi_op.initialize_grid();
            let mut global_bge2_os = 0.0_f64;
            let mut global_bge2_ss = 0.0_f64;
            if let Some(ri3mo_vec) = &scf_data.ri3mo {
                let eigenvalues = scf_data.eigenvalues.get(0).unwrap();
                let occupation = scf_data.occupation.get(0).unwrap();

                let (rimo, vir_range, occ_range) = &ri3mo_vec[0];
                let n0_local = rimo.size[0];
                let n1_global = rimo.size[1];
                let n2_global = rimo.size[2];

                let ctx = initialize_metadata(&grid, n2_global);
                let mut n0_global_tmp: u64 = 0;
                grid.cart_comm.all_reduce_into(&(n0_local as u64), &mut n0_global_tmp, &SystemOperation::sum());
                let n0_global = n0_global_tmp as usize;

                let redistributed_rimo = swap_ownership(
                    &grid, &ctx, &rimo, n0_global, n1_global, n2_global, &local_n0_range,
                );
                let (local_os, local_ss) = local_computation_close_shell_bge2(
                    &redistributed_rimo, &ctx, n2_global,
                    eigenvalues, occupation, occ_range.start, vir_range.start,
                );
                grid.cart_comm.all_reduce_into(&local_os, &mut global_bge2_os, &SystemOperation::sum());
                grid.cart_comm.all_reduce_into(&local_ss, &mut global_bge2_ss, &SystemOperation::sum());
            }
            Ok([global_bge2_os + global_bge2_ss, global_bge2_os, global_bge2_ss])
        } else {
            panic!("MPI not initialized for 2.5d");
        }
    }
    #[cfg(not(feature = "mpi"))]
    {
        panic!("2.5d SBGE2 requires the mpi feature");
    }
}

// =========================== open-shell (UHF/ROHF) 2.5D SBGE2 =====================
// 镜像串行 open_shell_sbge2_detailed_rayon（sbge2.rs L358-670）：
//   - SS 部分（αα、ββ 各自）：占据对严格 i<j，虚对三角 a<b，项 (e_ab-e_ba)^2；
//   - OS 部分（αβ）：占据对全 Cartesian（i_α × j_β），虚对全 Cartesian（a_α × b_β），项 e_ab^2；
//   - 每对走 open-shell BGE2 自洽迭代 iterator_open_shell_eij_serial；
//   - ROHF 时轨道能取 semi_eigenvalues（与串行一致）。
// 对分区与 PT2 open-shell 25d（local_computation_ss/os_batch）一致，仅 SS 剔除对角对。

#[cfg(feature = "mpi")]
fn sbge2_pair_open_ss(
    eri_virt: &rest_tensors::MatrixFull<f64>,
    denominator: &mut rest_tensors::MatrixFull<[f64; 2]>,
    eigenvalues: &Vec<f64>,
    occupation: &Vec<f64>,
    i_state_eigen: f64,
    j_state_eigen: f64,
    i_state_occ: f64,
    j_state_occ: f64,
    lumo: usize,
    lumo_min: usize,
    num_state: usize,
    enhanced_factor: f64,
    screening_factor: f64,
    shifted_factor: f64,
) -> (f64, f64) {
    let ij_state_eigen = i_state_eigen + j_state_eigen;
    let mut e_mp2_ss = 0.0_f64;
    let mut n_den = 0_usize;
    for i_virt in lumo..num_state {
        let i_virt_occ = occupation[i_virt];
        if (1.0 - i_virt_occ).abs() <= 1.0e-6 {
            continue;
        }
        for j_virt in (i_virt + 1)..num_state {
            let j_virt_occ = occupation[j_virt];
            if (1.0 - j_virt_occ).abs() <= 1.0e-6 {
                continue;
            }
            let mut double_gap = eigenvalues[i_virt] + eigenvalues[j_virt] - ij_state_eigen;
            if double_gap.abs() <= 10.0e-6 {
                double_gap = 1.0e-6;
            }
            let occ_state = i_state_occ * j_state_occ * (1.0 - i_virt_occ) * (1.0 - j_virt_occ);
            let i_loc = i_virt - lumo_min;
            let j_loc = j_virt - lumo_min;
            let e_ab = eri_virt[[i_loc, j_loc]];
            let e_ba = eri_virt[[j_loc, i_loc]];
            e_mp2_ss += (e_ab - e_ba).powf(2.0) / double_gap * occ_state;
            denominator[[i_loc, j_loc]] = [double_gap, occ_state];
            n_den += 1;
        }
    }
    let _ = n_den;
    let (e_eij_ss, _, _) = crate::ri_pt2::sbge2::iterator_open_shell_eij_serial(
        eri_virt, denominator, e_mp2_ss, 0.0,
        enhanced_factor, screening_factor, shifted_factor,
        lumo, lumo, true, lumo_min, num_state,
    );
    // 清理三角占用的 denominator（只清 a<b 上三角已写位置）
    for i_loc in 0..eri_virt.size[0] {
        for j_loc in (i_loc + 1)..eri_virt.size[1] {
            denominator[[i_loc, j_loc]] = [0.0, 0.0];
        }
    }
    (e_eij_ss, e_mp2_ss)
}

#[cfg(feature = "mpi")]
fn sbge2_pair_open_os(
    eri_virt: &rest_tensors::MatrixFull<f64>,
    denominator: &mut rest_tensors::MatrixFull<[f64; 2]>,
    alpha_eigenvalues: &Vec<f64>,
    beta_eigenvalues: &Vec<f64>,
    alpha_occupation: &Vec<f64>,
    beta_occupation: &Vec<f64>,
    i_state_eigen: f64,
    j_state_eigen: f64,
    i_state_occ: f64,
    j_state_occ: f64,
    lumo_i: usize,
    lumo_j: usize,
    lumo_min: usize,
    num_state: usize,
    enhanced_factor: f64,
    screening_factor: f64,
    shifted_factor: f64,
) -> (f64, f64) {
    let ij_state_eigen = i_state_eigen + j_state_eigen;
    let mut e_mp2_os = 0.0_f64;
    for i_virt in lumo_i..num_state {
        let i_virt_occ = alpha_occupation[i_virt];
        if (1.0 - i_virt_occ).abs() <= 1.0e-6 {
            continue;
        }
        for j_virt in lumo_j..num_state {
            let j_virt_occ = beta_occupation[j_virt];
            if (1.0 - j_virt_occ).abs() <= 1.0e-6 {
                continue;
            }
            let mut double_gap = alpha_eigenvalues[i_virt] + beta_eigenvalues[j_virt] - ij_state_eigen;
            if double_gap.abs() <= 10.0e-6 {
                double_gap = 1.0e-6;
            }
            let occ_state = i_state_occ * j_state_occ * (1.0 - i_virt_occ) * (1.0 - j_virt_occ);
            let i_loc = i_virt - lumo_min;
            let j_loc = j_virt - lumo_min;
            let e_ab = eri_virt[[i_loc, j_loc]];
            e_mp2_os += e_ab * e_ab / double_gap * occ_state;
            denominator[[i_loc, j_loc]] = [double_gap, occ_state];
        }
    }
    let (_, e_eij_os, _) = crate::ri_pt2::sbge2::iterator_open_shell_eij_serial(
        eri_virt, denominator, 0.0, e_mp2_os,
        enhanced_factor, screening_factor, shifted_factor,
        lumo_i, lumo_j, false, lumo_min, num_state,
    );
    for i_loc in 0..eri_virt.size[0] {
        for j_loc in 0..eri_virt.size[1] {
            denominator[[i_loc, j_loc]] = [0.0, 0.0];
        }
    }
    (e_eij_os, e_mp2_os)
}

/// same-spin BGE2 本地核：对 (i,j) 严格 i<j，由 num_occ 截断占据范围。
/// 返回已带负号的局部 ss 能量（累加进 local_term_ss）。
#[cfg(feature = "mpi")]
fn local_computation_ss_bge2(
    final_tensor: &rest_tensors::RIFull<f64>,
    ctx: &crate::ri_pt2::pt2_25d::Ctx25dBlock,
    n2_global: usize,
    eigenvalues: &Vec<f64>,
    occupation: &Vec<f64>,
    occ_offset: usize,
    virt_offset: usize,
    num_occ: usize,
    num_state: usize,
) -> f64 {
    let row_block_idx = &ctx.row_block_idx;
    let col_block_idx = &ctx.col_block_idx;
    let block_start_idx = &ctx.block_start_idx;
    let num_block = block_start_idx.len();
    let n0_global = final_tensor.size[0];
    let n1_global = final_tensor.size[1];
    let enhanced_factor = 1.0_f64;
    let screening_factor = 1.0_f64;
    let shifted_factor = 0.0_f64;
    let lumo = virt_offset;
    let lumo_min = virt_offset;

    let mut local_term_ss = 0.0_f64;
    let mut eri_virt = rest_tensors::MatrixFull::new([n1_global, n1_global], 0.0_f64);
    let mut denominator = rest_tensors::MatrixFull::new([n1_global, n1_global], [0.0_f64; 2]);

    for &col_blk in col_block_idx {
        let col_start = block_start_idx[col_blk];
        let mut col_end = if col_blk == num_block - 1 { n2_global - 1 } else { block_start_idx[col_blk + 1] - 1 };
        if col_start + occ_offset < num_occ && col_end + occ_offset >= num_occ {
            col_end = num_occ - occ_offset - 1;
        } else if col_start + occ_offset >= num_occ {
            continue;
        }
        for &row_blk in row_block_idx {
            let row_start = block_start_idx[row_blk];
            let mut row_end = if row_blk == num_block - 1 { n2_global - 1 } else { block_start_idx[row_blk + 1] - 1 };
            if row_start + occ_offset < num_occ && row_end + occ_offset >= num_occ {
                row_end = num_occ - occ_offset - 1;
            } else if row_start + occ_offset >= num_occ {
                continue;
            }
            if row_blk > col_blk {
                continue;
            }
            for i in row_start..=row_end {
                let ri_i = final_tensor.get_reducing_matrix_global_n2(i).unwrap();
                let i_state_eigen = eigenvalues[i + occ_offset];
                let i_state_occ = occupation[i + occ_offset];
                if i_state_occ.abs() <= 1.0e-6 {
                    continue;
                }
                for j in col_start..=col_end {
                    if i >= j {
                        continue; // SS 严格 i<j
                    }
                    let ri_j = final_tensor.get_reducing_matrix_global_n2(j).unwrap();
                    let j_state_eigen = eigenvalues[j + occ_offset];
                    let j_state_occ = occupation[j + occ_offset];
                    if j_state_occ.abs() <= 1.0e-6 {
                        continue;
                    }
                    rest_tensors::matrix_blas_lapack::_dgemm(
                        &ri_i, (0..n0_global, 0..n1_global), 'T',
                        &ri_j, (0..n0_global, 0..n1_global), 'N',
                        &mut eri_virt, (0..n1_global, 0..n1_global),
                        1.0, 0.0,
                    );
                    let (e_ss, _mp2_ss) = sbge2_pair_open_ss(
                        &eri_virt, &mut denominator, eigenvalues, occupation,
                        i_state_eigen, j_state_eigen, i_state_occ, j_state_occ,
                        lumo, lumo_min, num_state,
                        enhanced_factor, screening_factor, shifted_factor,
                    );
                    local_term_ss -= e_ss;
                }
            }
        }
    }
    local_term_ss
}

/// opposite-spin BGE2 本地核：α×β 全 Cartesian。
#[cfg(feature = "mpi")]
fn local_computation_os_bge2(
    alpha_tensor: &rest_tensors::RIFull<f64>,
    beta_tensor: &rest_tensors::RIFull<f64>,
    ctx: &crate::ri_pt2::pt2_25d::Ctx25dBlock,
    n2_global: usize,
    alpha_eigenvalues: &Vec<f64>,
    beta_eigenvalues: &Vec<f64>,
    alpha_occupation: &Vec<f64>,
    beta_occupation: &Vec<f64>,
    occ_offset: usize,
    virt_offset: usize,
    alpha_num_occ: usize,
    beta_num_occ: usize,
    num_state: usize,
) -> f64 {
    let row_block_idx = &ctx.row_block_idx;
    let col_block_idx = &ctx.col_block_idx;
    let block_start_idx = &ctx.block_start_idx;
    let num_block = block_start_idx.len();
    let n0_global = alpha_tensor.size[0];
    let n1_global = alpha_tensor.size[1];
    let enhanced_factor = 1.0_f64;
    let screening_factor = 1.0_f64;
    let shifted_factor = 0.0_f64;
    let lumo_min = virt_offset;

    let mut local_term_os = 0.0_f64;
    let mut eri_virt = rest_tensors::MatrixFull::new([n1_global, n1_global], 0.0_f64);
    let mut denominator = rest_tensors::MatrixFull::new([n1_global, n1_global], [0.0_f64; 2]);

    for &col_blk in col_block_idx {
        let col_start = block_start_idx[col_blk];
        let mut col_end = if col_blk == num_block - 1 { n2_global - 1 } else { block_start_idx[col_blk + 1] - 1 };
        if col_start + occ_offset < beta_num_occ && col_end + occ_offset >= beta_num_occ {
            col_end = beta_num_occ - occ_offset - 1;
        } else if col_start + occ_offset >= beta_num_occ {
            continue;
        }
        let lumo_j = virt_offset;
        for &row_blk in row_block_idx {
            let row_start = block_start_idx[row_blk];
            let mut row_end = if row_blk == num_block - 1 { n2_global - 1 } else { block_start_idx[row_blk + 1] - 1 };
            if row_start + occ_offset < alpha_num_occ && row_end + occ_offset >= alpha_num_occ {
                row_end = alpha_num_occ - occ_offset - 1;
            } else if row_start + occ_offset >= alpha_num_occ {
                continue;
            }
            for i in row_start..=row_end {
                let ri_i = alpha_tensor.get_reducing_matrix_global_n2(i).unwrap();
                let i_state_eigen = alpha_eigenvalues[i + occ_offset];
                let i_state_occ = alpha_occupation[i + occ_offset];
                if i_state_occ.abs() <= 1.0e-6 {
                    continue;
                }
                for j in col_start..=col_end {
                    let ri_j = beta_tensor.get_reducing_matrix_global_n2(j).unwrap();
                    let j_state_eigen = beta_eigenvalues[j + occ_offset];
                    let j_state_occ = beta_occupation[j + occ_offset];
                    if j_state_occ.abs() <= 1.0e-6 {
                        continue;
                    }
                    rest_tensors::matrix_blas_lapack::_dgemm(
                        &ri_i, (0..n0_global, 0..n1_global), 'T',
                        &ri_j, (0..n0_global, 0..n1_global), 'N',
                        &mut eri_virt, (0..n1_global, 0..n1_global),
                        1.0, 0.0,
                    );
                    let (e_os, _mp2_os) = sbge2_pair_open_os(
                        &eri_virt, &mut denominator,
                        alpha_eigenvalues, beta_eigenvalues,
                        alpha_occupation, beta_occupation,
                        i_state_eigen, j_state_eigen, i_state_occ, j_state_occ,
                        lumo_j, lumo_j, lumo_min, num_state,
                        enhanced_factor, screening_factor, shifted_factor,
                    );
                    local_term_os -= e_os;
                }
            }
        }
    }
    local_term_os
}

/// open-shell（UHF/ROHF）SBGE2 2.5D MPI 驱动（ROHF 时轨道能取 semi_eigenvalues）。
pub fn open_shell_sbge2_rayon_mpi_25d(
    scf_data: &SCF,
    mpi_operator: &Option<MPIOperator>,
) -> anyhow::Result<[f64; 3]> {
    #[cfg(feature = "mpi")]
    {
        if let (Some(mpi_op), Some(mpi_ix)) = (&mpi_operator, &scf_data.mol.mpi_data) {
            use mpi::collective::SystemOperation;
            use mpi::traits::*;
            use crate::ri_pt2::pt2_25d::{initialize_metadata, swap_ownership};

            let my_rank = mpi_ix.rank;
            let local_n0_range = if let Some(loc_auxbas) = &mpi_ix.auxbas {
                loc_auxbas[my_rank].clone()
            } else {
                panic!("Memory distribution should be initialized for the auxiliary basis sets before post-SCF calculations")
            };

            let is_rohf = matches!(scf_data.scftype, crate::scf_io::SCFType::ROHF);
            let alpha_eigenvalues = if is_rohf {
                &scf_data.semi_eigenvalues.as_ref().unwrap()[0]
            } else {
                &scf_data.eigenvalues[0]
            };
            let beta_eigenvalues = if is_rohf {
                &scf_data.semi_eigenvalues.as_ref().unwrap()[1]
            } else {
                &scf_data.eigenvalues[1]
            };
            let alpha_occupation = &scf_data.occupation[0];
            let beta_occupation = &scf_data.occupation[1];
            let alpha_num_occ = if scf_data.mol.num_elec[1] <= 1.0e-6 { 0 } else { scf_data.homo[0] + 1 };
            let beta_num_occ = if scf_data.mol.num_elec[2] <= 1.0e-6 { 0 } else { scf_data.homo[1] + 1 };
            let num_state = scf_data.mol.num_state;

            let grid = mpi_op.initialize_grid();
            let mut global_os = 0.0_f64;
            let mut global_ss = 0.0_f64;
            if let Some(ri3mo_vec) = &scf_data.ri3mo {
                let (alpha_rimo, vir_range, occ_range) = &ri3mo_vec[0];
                let (beta_rimo, _, _) = &ri3mo_vec[1];
                let n0_local = alpha_rimo.size[0];
                let n1_global = alpha_rimo.size[1];
                let n2_global = alpha_rimo.size[2];

                let ctx = initialize_metadata(&grid, n2_global);
                let mut n0_global_tmp: u64 = 0;
                grid.cart_comm.all_reduce_into(&(n0_local as u64), &mut n0_global_tmp, &SystemOperation::sum());
                let n0_global = n0_global_tmp as usize;

                let redistributed_alpha = swap_ownership(&grid, &ctx, &alpha_rimo, n0_global, n1_global, n2_global, &local_n0_range);
                let redistributed_beta = swap_ownership(&grid, &ctx, &beta_rimo, n0_global, n1_global, n2_global, &local_n0_range);

                let term_aa = local_computation_ss_bge2(
                    &redistributed_alpha, &ctx, n2_global,
                    alpha_eigenvalues, alpha_occupation, occ_range.start, vir_range.start,
                    alpha_num_occ, num_state,
                );
                let term_bb = local_computation_ss_bge2(
                    &redistributed_beta, &ctx, n2_global,
                    beta_eigenvalues, beta_occupation, occ_range.start, vir_range.start,
                    beta_num_occ, num_state,
                );
                let local_ss = term_aa + term_bb;
                let local_os = local_computation_os_bge2(
                    &redistributed_alpha, &redistributed_beta, &ctx, n2_global,
                    alpha_eigenvalues, beta_eigenvalues, alpha_occupation, beta_occupation,
                    occ_range.start, vir_range.start, alpha_num_occ, beta_num_occ, num_state,
                );
                grid.cart_comm.all_reduce_into(&local_ss, &mut global_ss, &SystemOperation::sum());
                grid.cart_comm.all_reduce_into(&local_os, &mut global_os, &SystemOperation::sum());
            }
            Ok([global_os + global_ss, global_os, global_ss])
        } else {
            panic!("MPI not initialized for 2.5d");
        }
    }
    #[cfg(not(feature = "mpi"))]
    {
        panic!("2.5d SBGE2 requires the mpi feature");
    }
}

// =========================== detailed（SCC15 用）2.5D SBGE2 =====================
// 镜像串行 close/open_shell_sbge2_detailed_rayon 的返回约定：
//   ([total, os, ss], [eij_00, eij_01, eij_11])，eij_{xy}[(i,j)] = (PT2 对能, sBGE2 对能)
// 逐对 (PT2, sBGE2) 值由 2.5D 对唯一划分本地计算，eij 矩阵跨 rank allreduce 求和
// （每对恰属一个 rank，求和即还原）。数值约定与串行 detailed 逐字一致：
//   - close：eij_00/11[(i,j)]_{i<j} = (-raw_ss, -eij_ss)；eij_01[(i,j)]_{全矩形} = (-raw_os, -eij_os)
//     （串行：×2 后存 -term，再 /2 → 净 -raw）
//   - open SS：i<j 严格，(-raw_ss, -eij_ss)；open OS：全矩形，(-raw_os, -eij_os)
// ==========================================================================================

/// 闭壳层 detailed 本地核：返回 (local_os, local_ss, 6 个本地 eij f64 矩阵数据)。
#[cfg(feature = "mpi")]
fn local_computation_close_shell_bge2_detailed(
    final_tensor: &rest_tensors::RIFull<f64>,
    ctx: &crate::ri_pt2::pt2_25d::Ctx25dBlock,
    n2_global: usize,
    eigenvalues: &Vec<f64>,
    occupation: &Vec<f64>,
    occ_offset: usize,
    virt_offset: usize,
    num_occ: usize,
) -> (f64, f64, Vec<Vec<f64>>) {
    let row_block_idx = &ctx.row_block_idx;
    let col_block_idx = &ctx.col_block_idx;
    let block_start_idx = &ctx.block_start_idx;
    let num_block = block_start_idx.len();
    let n0_global = final_tensor.size[0];
    let n1_global = final_tensor.size[1];

    let enhanced_factor = 1.0_f64;
    let screening_factor = 1.0_f64;
    let shifted_factor = 0.0_f64;
    let lumo = virt_offset;
    let lumo_min = virt_offset;

    let mut local_bge2_os = 0.0_f64;
    let mut local_bge2_ss = 0.0_f64;

    // eij 局部矩阵（全零初始化；每对恰属一个 rank，allreduce 后即完整矩阵）
    let n2 = num_occ;
    let mut eij00_pt2 = vec![0.0f64; n2 * n2];
    let mut eij00_bge = vec![0.0f64; n2 * n2];
    let mut eij01_pt2 = vec![0.0f64; n2 * n2];
    let mut eij01_bge = vec![0.0f64; n2 * n2];
    let mut eij11_pt2 = vec![0.0f64; n2 * n2];
    let mut eij11_bge = vec![0.0f64; n2 * n2];

    let mut eri_virt = rest_tensors::MatrixFull::new([n1_global, n1_global], 0.0_f64);
    let mut denominator = rest_tensors::MatrixFull::new([n1_global, n1_global], [0.0_f64; 2]);

    for &row_blk in row_block_idx {
        let row_start = block_start_idx[row_blk];
        let row_end = if row_blk == num_block - 1 { n2_global - 1 } else { block_start_idx[row_blk + 1] - 1 };
        for &col_blk in col_block_idx {
            let col_start = block_start_idx[col_blk];
            let col_end = if col_blk == num_block - 1 { n2_global - 1 } else { block_start_idx[col_blk + 1] - 1 };
            if row_blk > col_blk {
                continue;
            } else if row_blk < col_blk {
                assert!(row_end < col_start);
                for i in row_start..=row_end {
                    let ri_i = final_tensor.get_reducing_matrix_global_n2(i).unwrap();
                    let i_state_eigen = eigenvalues[i + occ_offset];
                    let i_state_occ = occupation[i + occ_offset] / 2.0;
                    if i_state_occ.abs() <= 1.0e-6 {
                        continue;
                    }
                    for j in col_start..=col_end {
                        let ri_j = final_tensor.get_reducing_matrix_global_n2(j).unwrap();
                        let j_state_eigen = eigenvalues[j + occ_offset];
                        let j_state_occ = occupation[j + occ_offset] / 2.0;
                        if j_state_occ.abs() <= 1.0e-6 {
                            continue;
                        }
                        rest_tensors::matrix_blas_lapack::_dgemm(
                            &ri_i, (0..n0_global, 0..n1_global), 'T',
                            &ri_j, (0..n0_global, 0..n1_global), 'N',
                            &mut eri_virt, (0..n1_global, 0..n1_global),
                            1.0, 0.0,
                        );
                        let (e_ss, e_os, mp2_ss, mp2_os) = sbge2_pair_close_shell(
                            &eri_virt, &mut denominator, eigenvalues, occupation,
                            i_state_eigen, j_state_eigen, i_state_occ, j_state_occ,
                            lumo, lumo_min, n1_global,
                            enhanced_factor, screening_factor, shifted_factor,
                        );
                        local_bge2_ss -= 2.0 * e_ss;
                        local_bge2_os -= 2.0 * e_os;
                        // i<j：eij_00/11 = (-raw_ss, -eij_ss)；eij_01 = (-raw_os, -eij_os)
                        // eij_01 为全矩形语义：下三角 (j,i) 镜像同值（镜像串行 i>j 分支）
                        eij00_pt2[i + j * n2] = -mp2_ss;
                        eij00_bge[i + j * n2] = -e_ss;
                        eij11_pt2[i + j * n2] = -mp2_ss;
                        eij11_bge[i + j * n2] = -e_ss;
                        eij01_pt2[i + j * n2] = -mp2_os;
                        eij01_bge[i + j * n2] = -e_os;
                        eij01_pt2[j + i * n2] = -mp2_os;
                        eij01_bge[j + i * n2] = -e_os;
                    }
                }
            } else {
                for i in row_start..=row_end {
                    let ri_i = final_tensor.get_reducing_matrix_global_n2(i).unwrap();
                    let i_state_eigen = eigenvalues[i + occ_offset];
                    let i_state_occ = occupation[i + occ_offset] / 2.0;
                    if i_state_occ.abs() <= 1.0e-6 {
                        continue;
                    }
                    for j in col_start..=row_end.min(col_end) {
                        if i > j {
                            continue;
                        }
                        let ri_j = final_tensor.get_reducing_matrix_global_n2(j).unwrap();
                        let j_state_eigen = eigenvalues[j + occ_offset];
                        let j_state_occ = occupation[j + occ_offset] / 2.0;
                        if j_state_occ.abs() <= 1.0e-6 {
                            continue;
                        }
                        rest_tensors::matrix_blas_lapack::_dgemm(
                            &ri_i, (0..n0_global, 0..n1_global), 'T',
                            &ri_j, (0..n0_global, 0..n1_global), 'N',
                            &mut eri_virt, (0..n1_global, 0..n1_global),
                            1.0, 0.0,
                        );
                        let off_diag = i != j;
                        let (e_ss, e_os, mp2_ss, mp2_os) = sbge2_pair_close_shell(
                            &eri_virt, &mut denominator, eigenvalues, occupation,
                            i_state_eigen, j_state_eigen, i_state_occ, j_state_occ,
                            lumo, lumo_min, n1_global,
                            enhanced_factor, screening_factor, shifted_factor,
                        );
                        if off_diag {
                            local_bge2_ss -= 2.0 * e_ss;
                            local_bge2_os -= 2.0 * e_os;
                        } else {
                            local_bge2_ss -= e_ss;
                            local_bge2_os -= e_os;
                        }
                        // 对角 (i,i)：eij_01 = (-raw_os, -eij_os)；非对角 i<j 补下三角镜像
                        eij00_pt2[i + j * n2] = -mp2_ss;
                        eij00_bge[i + j * n2] = -e_ss;
                        eij11_pt2[i + j * n2] = -mp2_ss;
                        eij11_bge[i + j * n2] = -e_ss;
                        eij01_pt2[i + j * n2] = -mp2_os;
                        eij01_bge[i + j * n2] = -e_os;
                        if off_diag {
                            eij01_pt2[j + i * n2] = -mp2_os;
                            eij01_bge[j + i * n2] = -e_os;
                        }
                    }
                }
            }
        }
    }
    let eij_data = vec![eij00_pt2, eij00_bge, eij01_pt2, eij01_bge, eij11_pt2, eij11_bge];
    (local_bge2_os, local_bge2_ss, eij_data)
}

/// 闭壳层 detailed 驱动：返回串行 detailed 同构签名。
#[cfg(feature = "mpi")]
pub fn close_shell_sbge2_detailed_rayon_mpi_25d(
    scf_data: &SCF,
    mpi_operator: &Option<MPIOperator>,
) -> anyhow::Result<([f64; 3], [rest_tensors::MatrixFull<(f64, f64)>; 3])> {
    #[cfg(feature = "mpi")]
    {
        use mpi::collective::SystemOperation;
        use mpi::traits::*;
        use crate::ri_pt2::pt2_25d::{initialize_metadata, swap_ownership};

        let (mpi_op, mpi_ix) = match (mpi_operator, &scf_data.mol.mpi_data) {
            (Some(op), Some(ix)) => (op, ix),
            _ => panic!("MPI not initialized for 2.5d"),
        };
        let my_rank = mpi_ix.rank;
        let local_n0_range = if let Some(loc_auxbas) = &mpi_ix.auxbas {
            loc_auxbas[my_rank].clone()
        } else {
            panic!("Memory distribution should be initialized for the auxiliary basis sets before post-SCF calculations")
        };

        let grid = mpi_op.initialize_grid();
        if let Some(ri3mo_vec) = &scf_data.ri3mo {
            let eigenvalues = scf_data.eigenvalues.get(0).unwrap();
            let occupation = scf_data.occupation.get(0).unwrap();
            let num_occ = if scf_data.mol.num_elec[0] <= 1.0e-6 { 0 } else { scf_data.homo[0] + 1 };

            let (rimo, vir_range, occ_range) = &ri3mo_vec[0];
            let n0_local = rimo.size[0];
            let n1_global = rimo.size[1];
            let n2_global = rimo.size[2];

            let ctx = initialize_metadata(&grid, n2_global);
            let mut n0_global_tmp: u64 = 0;
            grid.cart_comm.all_reduce_into(&(n0_local as u64), &mut n0_global_tmp, &SystemOperation::sum());
            let n0_global = n0_global_tmp as usize;

            let redistributed_rimo = swap_ownership(
                &grid, &ctx, &rimo, n0_global, n1_global, n2_global, &local_n0_range,
            );
            let (local_os, local_ss, mut eij_data) = local_computation_close_shell_bge2_detailed(
                &redistributed_rimo, &ctx, n2_global,
                eigenvalues, occupation, occ_range.start, vir_range.start, num_occ,
            );
            let mut global_os = 0.0_f64;
            let mut global_ss = 0.0_f64;
            grid.cart_comm.all_reduce_into(&local_os, &mut global_os, &SystemOperation::sum());
            grid.cart_comm.all_reduce_into(&local_ss, &mut global_ss, &SystemOperation::sum());
            for mat in eij_data.iter_mut() {
                let mut reduced = vec![0.0f64; mat.len()];
                grid.cart_comm.any_process().all_reduce_into(&mat[..], &mut reduced[..], &SystemOperation::sum());
                std::mem::swap(mat, &mut reduced);
            }
            let combine = |pt2: &Vec<f64>, bge: &Vec<f64>| -> rest_tensors::MatrixFull<(f64, f64)> {
                let mut m = rest_tensors::MatrixFull::new([num_occ, num_occ], (0.0_f64, 0.0_f64));
                for idx in 0..(num_occ * num_occ) {
                    m.data[idx] = (pt2[idx], bge[idx]);
                }
                m
            };
            let eij_00 = combine(&eij_data[0], &eij_data[1]);
            let eij_01 = combine(&eij_data[2], &eij_data[3]);
            let eij_11 = combine(&eij_data[4], &eij_data[5]);
            Ok(([global_os + global_ss, global_os, global_ss], [eij_00, eij_01, eij_11]))
        } else {
            panic!("RI3MO should be initialized before the 2.5d SBGE2 detailed calculation")
        }
    }
}

/// 开壳层（UHF/ROHF）detailed 本地核。
#[cfg(feature = "mpi")]
fn local_computation_open_shell_bge2_detailed(
    alpha_tensor: &rest_tensors::RIFull<f64>,
    beta_tensor: &rest_tensors::RIFull<f64>,
    ctx: &crate::ri_pt2::pt2_25d::Ctx25dBlock,
    n2_global: usize,
    alpha_eigenvalues: &Vec<f64>,
    beta_eigenvalues: &Vec<f64>,
    alpha_occupation: &Vec<f64>,
    beta_occupation: &Vec<f64>,
    occ_offset: usize,
    virt_offset: usize,
    alpha_num_occ: usize,
    beta_num_occ: usize,
    num_state: usize,
    num_occu_max: usize,
) -> (f64, f64, Vec<Vec<f64>>) {
    let row_block_idx = &ctx.row_block_idx;
    let col_block_idx = &ctx.col_block_idx;
    let block_start_idx = &ctx.block_start_idx;
    let num_block = block_start_idx.len();
    let n0_global = alpha_tensor.size[0];
    let n1_global = alpha_tensor.size[1];

    let enhanced_factor = 1.0_f64;
    let screening_factor = 1.0_f64;
    let shifted_factor = 0.0_f64;
    let lumo_min = virt_offset;

    let mut local_ss = 0.0_f64;
    let mut local_os = 0.0_f64;

    let n2 = num_occu_max;
    let mut eij00_pt2 = vec![0.0f64; n2 * n2];
    let mut eij00_bge = vec![0.0f64; n2 * n2];
    let mut eij01_pt2 = vec![0.0f64; n2 * n2];
    let mut eij01_bge = vec![0.0f64; n2 * n2];
    let mut eij11_pt2 = vec![0.0f64; n2 * n2];
    let mut eij11_bge = vec![0.0f64; n2 * n2];

    let mut eri_virt = rest_tensors::MatrixFull::new([n1_global, n1_global], 0.0_f64);
    let mut denominator = rest_tensors::MatrixFull::new([n1_global, n1_global], [0.0_f64; 2]);

    // ---------- SS 部分：αα、ββ 各自，i<j 严格 ----------
    for (spin, (tensor, eigenvalues, occupation, num_occ)) in [
        (0usize, (alpha_tensor, alpha_eigenvalues, alpha_occupation, alpha_num_occ)),
        (1usize, (beta_tensor, beta_eigenvalues, beta_occupation, beta_num_occ)),
    ] {
        for &row_blk in row_block_idx {
            let row_start = block_start_idx[row_blk];
            let mut row_end = if row_blk == num_block - 1 { n2_global - 1 } else { block_start_idx[row_blk + 1] - 1 };
            if row_start + occ_offset < num_occ && row_end + occ_offset >= num_occ {
                row_end = num_occ - occ_offset - 1;
            } else if row_start + occ_offset >= num_occ {
                continue;
            }
            for &col_blk in col_block_idx {
                let col_start = block_start_idx[col_blk];
                let mut col_end = if col_blk == num_block - 1 { n2_global - 1 } else { block_start_idx[col_blk + 1] - 1 };
                if col_start + occ_offset < num_occ && col_end + occ_offset >= num_occ {
                    col_end = num_occ - occ_offset - 1;
                } else if col_start + occ_offset >= num_occ {
                    continue;
                }
                if row_blk > col_blk {
                    continue;
                }
                for i in row_start..=row_end {
                    let ri_i = tensor.get_reducing_matrix_global_n2(i).unwrap();
                    let i_state_eigen = eigenvalues[i + occ_offset];
                    let i_state_occ = occupation[i + occ_offset];
                    if i_state_occ.abs() <= 1.0e-6 {
                        continue;
                    }
                    for j in col_start..=col_end {
                        if i >= j {
                            continue; // SS 严格 i<j
                        }
                        let ri_j = tensor.get_reducing_matrix_global_n2(j).unwrap();
                        let j_state_eigen = eigenvalues[j + occ_offset];
                        let j_state_occ = occupation[j + occ_offset];
                        if j_state_occ.abs() <= 1.0e-6 {
                            continue;
                        }
                        rest_tensors::matrix_blas_lapack::_dgemm(
                            &ri_i, (0..n0_global, 0..n1_global), 'T',
                            &ri_j, (0..n0_global, 0..n1_global), 'N',
                            &mut eri_virt, (0..n1_global, 0..n1_global),
                            1.0, 0.0,
                        );
                        let (e_ss, mp2_ss) = sbge2_pair_open_ss(
                            &eri_virt, &mut denominator, eigenvalues, occupation,
                            i_state_eigen, j_state_eigen, i_state_occ, j_state_occ,
                            virt_offset, lumo_min, num_state,
                            enhanced_factor, screening_factor, shifted_factor,
                        );
                        local_ss -= e_ss;
                        let (pt2_mat, bge_mat) = if spin == 0 {
                            (&mut eij00_pt2, &mut eij00_bge)
                        } else {
                            (&mut eij11_pt2, &mut eij11_bge)
                        };
                        pt2_mat[i + j * n2] = -mp2_ss;
                        bge_mat[i + j * n2] = -e_ss;
                    }
                }
            }
        }
    }

    // ---------- OS 部分：α×β 全 Cartesian ----------
    for &col_blk in col_block_idx {
        let col_start = block_start_idx[col_blk];
        let mut col_end = if col_blk == num_block - 1 { n2_global - 1 } else { block_start_idx[col_blk + 1] - 1 };
        if col_start + occ_offset < beta_num_occ && col_end + occ_offset >= beta_num_occ {
            col_end = beta_num_occ - occ_offset - 1;
        } else if col_start + occ_offset >= beta_num_occ {
            continue;
        }
        for &row_blk in row_block_idx {
            let row_start = block_start_idx[row_blk];
            let mut row_end = if row_blk == num_block - 1 { n2_global - 1 } else { block_start_idx[row_blk + 1] - 1 };
            if row_start + occ_offset < alpha_num_occ && row_end + occ_offset >= alpha_num_occ {
                row_end = alpha_num_occ - occ_offset - 1;
            } else if row_start + occ_offset >= alpha_num_occ {
                continue;
            }
            for i in row_start..=row_end {
                let ri_i = alpha_tensor.get_reducing_matrix_global_n2(i).unwrap();
                let i_state_eigen = alpha_eigenvalues[i + occ_offset];
                let i_state_occ = alpha_occupation[i + occ_offset];
                if i_state_occ.abs() <= 1.0e-6 {
                    continue;
                }
                for j in col_start..=col_end {
                    let ri_j = beta_tensor.get_reducing_matrix_global_n2(j).unwrap();
                    let j_state_eigen = beta_eigenvalues[j + occ_offset];
                    let j_state_occ = beta_occupation[j + occ_offset];
                    if j_state_occ.abs() <= 1.0e-6 {
                        continue;
                    }
                    rest_tensors::matrix_blas_lapack::_dgemm(
                        &ri_i, (0..n0_global, 0..n1_global), 'T',
                        &ri_j, (0..n0_global, 0..n1_global), 'N',
                        &mut eri_virt, (0..n1_global, 0..n1_global),
                        1.0, 0.0,
                    );
                    let (e_os, mp2_os) = sbge2_pair_open_os(
                        &eri_virt, &mut denominator,
                        alpha_eigenvalues, beta_eigenvalues,
                        alpha_occupation, beta_occupation,
                        i_state_eigen, j_state_eigen, i_state_occ, j_state_occ,
                        virt_offset, virt_offset, lumo_min, num_state,
                        enhanced_factor, screening_factor, shifted_factor,
                    );
                    local_os -= e_os;
                    eij01_pt2[i + j * n2] = -mp2_os;
                    eij01_bge[i + j * n2] = -e_os;
                }
            }
        }
    }

    let eij_data = vec![eij00_pt2, eij00_bge, eij01_pt2, eij01_bge, eij11_pt2, eij11_bge];
    (local_os, local_ss, eij_data)
}

/// 开壳层（UHF/ROHF）detailed 驱动：返回串行 detailed 同构签名。
#[cfg(feature = "mpi")]
pub fn open_shell_sbge2_detailed_rayon_mpi_25d(
    scf_data: &SCF,
    mpi_operator: &Option<MPIOperator>,
) -> anyhow::Result<([f64; 3], [rest_tensors::MatrixFull<(f64, f64)>; 3])> {
    #[cfg(feature = "mpi")]
    {
        use mpi::collective::SystemOperation;
        use mpi::traits::*;
        use crate::ri_pt2::pt2_25d::{initialize_metadata, swap_ownership};

        let (mpi_op, mpi_ix) = match (mpi_operator, &scf_data.mol.mpi_data) {
            (Some(op), Some(ix)) => (op, ix),
            _ => panic!("MPI not initialized for 2.5d"),
        };
        let my_rank = mpi_ix.rank;
        let local_n0_range = if let Some(loc_auxbas) = &mpi_ix.auxbas {
            loc_auxbas[my_rank].clone()
        } else {
            panic!("Memory distribution should be initialized for the auxiliary basis sets before post-SCF calculations")
        };

        let is_rohf = matches!(scf_data.scftype, crate::scf_io::SCFType::ROHF);
        let alpha_eigenvalues = if is_rohf { &scf_data.semi_eigenvalues.as_ref().unwrap()[0] } else { &scf_data.eigenvalues[0] };
        let beta_eigenvalues = if is_rohf { &scf_data.semi_eigenvalues.as_ref().unwrap()[1] } else { &scf_data.eigenvalues[1] };
        let alpha_occupation = &scf_data.occupation[0];
        let beta_occupation = &scf_data.occupation[1];
        let alpha_num_occ = if scf_data.mol.num_elec[1] <= 1.0e-6 { 0 } else { scf_data.homo[0] + 1 };
        let beta_num_occ = if scf_data.mol.num_elec[2] <= 1.0e-6 { 0 } else { scf_data.homo[1] + 1 };
        let num_state = scf_data.mol.num_state;
        let num_occu_max = alpha_num_occ.max(beta_num_occ);

        let grid = mpi_op.initialize_grid();
        if let Some(ri3mo_vec) = &scf_data.ri3mo {
            let (alpha_rimo, vir_range, occ_range) = &ri3mo_vec[0];
            let (beta_rimo, _, _) = &ri3mo_vec[1];
            let n0_local = alpha_rimo.size[0];
            let n1_global = alpha_rimo.size[1];
            let n2_global = alpha_rimo.size[2];

            let ctx = initialize_metadata(&grid, n2_global);
            let mut n0_global_tmp: u64 = 0;
            grid.cart_comm.all_reduce_into(&(n0_local as u64), &mut n0_global_tmp, &SystemOperation::sum());
            let n0_global = n0_global_tmp as usize;

            let redistributed_alpha = swap_ownership(&grid, &ctx, alpha_rimo, n0_global, n1_global, n2_global, &local_n0_range);
            let redistributed_beta = swap_ownership(&grid, &ctx, beta_rimo, n0_global, n1_global, n2_global, &local_n0_range);

            let (local_os, local_ss, mut eij_data) = local_computation_open_shell_bge2_detailed(
                &redistributed_alpha, &redistributed_beta, &ctx, n2_global,
                alpha_eigenvalues, beta_eigenvalues, alpha_occupation, beta_occupation,
                occ_range.start, vir_range.start, alpha_num_occ, beta_num_occ, num_state, num_occu_max,
            );
            let mut global_os = 0.0_f64;
            let mut global_ss = 0.0_f64;
            grid.cart_comm.all_reduce_into(&local_os, &mut global_os, &SystemOperation::sum());
            grid.cart_comm.all_reduce_into(&local_ss, &mut global_ss, &SystemOperation::sum());
            for mat in eij_data.iter_mut() {
                let mut reduced = vec![0.0f64; mat.len()];
                grid.cart_comm.any_process().all_reduce_into(&mat[..], &mut reduced[..], &SystemOperation::sum());
                std::mem::swap(mat, &mut reduced);
            }
            let combine = |pt2: &Vec<f64>, bge: &Vec<f64>| -> rest_tensors::MatrixFull<(f64, f64)> {
                let mut m = rest_tensors::MatrixFull::new([num_occu_max, num_occu_max], (0.0_f64, 0.0_f64));
                for idx in 0..(num_occu_max * num_occu_max) {
                    m.data[idx] = (pt2[idx], bge[idx]);
                }
                m
            };
            let eij_00 = combine(&eij_data[0], &eij_data[1]);
            let eij_01 = combine(&eij_data[2], &eij_data[3]);
            let eij_11 = combine(&eij_data[4], &eij_data[5]);
            Ok(([global_os + global_ss, global_os, global_ss], [eij_00, eij_01, eij_11]))
        } else {
            panic!("RI3MO should be initialized before the 2.5d SBGE2 detailed calculation")
        }
    }
}
