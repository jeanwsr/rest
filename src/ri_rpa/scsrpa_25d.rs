// ============================================================================
// 2.5D MPI 下的 SCS-RPA（R-xDH7）关联能驱动
// ============================================================================
//
// 目的
// ----
// 让 DFAFamily::SCSRPA（R-xDH7、SCS-RPA）复用 ri_pt2/pt2_25d.rs 的 2.5D 重分布框架
// （线性归属 initialize_metadata_linear + redistribute_to_diag：ri3mo 从 aux 索引 1D
// 分布重排为 blk%P 单归属，每进程持完整 n_aux×n_vir 切片 ≈ n_occ/P 个，持久内存
// ≈ M_total/p；p=进程数），从而获得第一条可用的 MPI 路径（此前 2.5D 分派为
// unreachable，1D 回退为 panic——因为串行自旋响应内核按全量张量索引编写，
// 与 MPI 下 aux-分布 ri3mo 不兼容）。
//
// 数学结构
// --------
// 与 PT2/SBGE2 的占据对 (i,j) 二重求和不同，自旋响应是**单占据指标**求和：
//     χ₀^σ(ω)[P,Q] = Σ_{j∈occ^σ} B_j(ω)·B_j(ω)^T
//     B_j(ω)[P,a] = ζ_ja(ω) · rimo^σ[P,a,j]
// 因此本地核只对 ctx.ownership[rank] 的占据块（对角唯一划分）累加 partial 响应，
// 每频率每自旋一次 [n_aux²] allreduce；归约后所有 rank 持一致满阵，
// evaluate_osrpa_integrand（dgetrf / Neumann 级数 / λ-积分）原样冗余复用，
// 加权标量各 rank 相同，无需末端归约。
//
// 数值约定（逐字镜像串行 evaluate_spin_response_serial / evaluate_osrpa_integrand）
// ----------------------------------------------------------------
// - 能隙夹逼 ±1e-6；Yang 系综 Green 函数分数占据公式（frac_spin_occ=spin_channel/2）；
// - rpa_de_excitation_parameters 的 de-excitation level shift（默认 scale=0 → 0）；
// - 占据过滤 j_occ≥1e-6、(1-occ·frac)≥1e-6；num_occ = homo+1（num_elec>1e-6）；
// - ROHF 用 semi_eigenvalues；spin_channel==1 时 integrand 内部自带 ×2 因子；
// - ω=0 谱半径（sc_check）经同一归约路径计算，所有 rank 分支一致（防死锁）。
//
// 内存剖面（每 rank）
// -------------------
// 持久张量 ≈ M_total/p（线性归属，Phase 3 内存路线；单指标求和核只需对角归属，
// 无需 PT2/SBGE2 对核的 row/col 覆盖）。每频率 scratch：n_spin×[n_aux²] partial
// （复制式，不随 p 缩）。naux ≥ 10⁴ 的分布化响应见可行性报告 §3.6（Phase 3 后续）。
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
use super::scsrpa::{evaluate_osrpa_integrand, evaluate_special_radius, screening_de_excitation};
#[cfg(feature = "mpi")]
use super::{trans_gauss_legendre_grids, gauss_legendre_grids, logarithmic_grid};

#[cfg(feature = "mpi")]
use crate::constants::PI;

/// ζ 缩放系数：逐字镜像 evaluate_spin_response_serial 的内层公式。
/// 返回 None 表示该 (j,k) 对被占据过滤剔除。
#[cfg(feature = "mpi")]
#[inline]
fn zeta_factor(
    j_state_eigen: f64,
    k_state_eigen: f64,
    j_state_occ: f64,
    k_state_occ: f64,
    freq: f64,
    frac_spin_occ: f64,
    de_excitation: [f64; 4],
) -> Option<f64> {
    if j_state_occ.abs() < 1.0e-6 {
        return None;
    }
    if (1.0 - k_state_occ * frac_spin_occ).abs() < 1.0e-6 {
        return None;
    }
    let mut energy_gap = j_state_eigen - k_state_eigen;
    if energy_gap < 1.0e-6 && energy_gap >= 0.0 {
        energy_gap += 1.0e-6;
    } else if energy_gap > -1.0e-6 && energy_gap < 0.0 {
        energy_gap += -1.0e-6;
    };
    let [a, b, sigma, scaling_factor] = de_excitation;
    let level_shift = screening_de_excitation(energy_gap, freq, a, b, sigma, scaling_factor);
    let zeta = 2.0f64 * (energy_gap + level_shift)
        / ((energy_gap + level_shift).powf(2.0) + freq * freq)
        * (j_state_occ * frac_spin_occ)
        * (1.0f64 - k_state_occ * frac_spin_occ);
    Some(zeta)
}

/// 本地 partial 自旋响应：对 ownership 块内的每个占据 j，
/// polar_partial += B_j(ω)·B_j(ω)^T（B_j = ζ 缩放后的 rimo_j 切片）。
/// j 循环镜像 evaluate_spin_response_serial；唯一差别是 j 的枚举范围由
/// ownership 块划分（对角唯一划分，Σ_rank 恰好覆盖每个 j 一次）。
#[cfg(feature = "mpi")]
fn local_spin_response_25d(
    final_tensor: &rest_tensors::RIFull<f64>,
    ctx: &crate::ri_pt2::pt2_25d::Ctx25dBlock,
    grid_rank: usize,
    n2_global: usize,
    freq: f64,
    eigenvalues: &Vec<f64>,
    occupation: &Vec<f64>,
    occ_range_start: usize, // occ_range.start（= start_mo）
    virt_range_start: usize, // vir_range.start（= lumo）
    num_occ: usize,          // 该自旋的 homo+1（无电子自旋为 0）
    num_state: usize,
    frac_spin_occ: f64,
    de_excitation: [f64; 4],
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
            if j_state_occ.abs() <= 1.0e-6 {
                continue;
            }
            let rimo_j = match final_tensor.get_reducing_matrix_global_n2(j_loc) {
                Some(m) => m,
                None => continue, // 该 j 不在本 rank 的 ownership 块内（理论不可达）
            };
            // 串行代码对每个 j 重新开零矩阵：被占据过滤剔除的 k 列须为 0
            tmp_matrix.data.iter_mut().for_each(|x| *x = 0.0);
            // 逐虚轨道 k：全局 k ∈ [lumo, num_state)，张量列索引 k_loc = k - lumo。
            // rimo_j 为 [n0_global × n1_global] 列优先切片，直接按列取数（dist 张量
            // 的存储位序非单调，必须经 get_reducing_matrix_global_n2 的全局查找）。
            for k_state in virt_range_start..num_state {
                let k_state_eigen = eigenvalues[k_state];
                let k_state_occ = occupation[k_state];
                if let Some(zeta) = zeta_factor(
                    j_state_eigen,
                    k_state_eigen,
                    j_state_occ,
                    k_state_occ,
                    freq,
                    frac_spin_occ,
                    de_excitation,
                ) {
                    let k_loc_state = k_state - virt_range_start;
                    let col_start = k_loc_state * n0_global;
                    let from_col = &rimo_j.data[col_start..col_start + n0_global];
                    let to_iter = tmp_matrix.iter_submatrix_mut(0..n0_global, k_loc_state..k_loc_state + 1);
                    to_iter.zip(from_col.iter()).for_each(|(to, from)| {
                        *to = *from * zeta
                    });
                }
            }
            _dgemm_full(&tmp_matrix, 'N', &rimo_j, 'T', polar_partial, 1.0, 1.0);
        }
    }
}

/// 归约后的满阵 + integrand：每 rank 一致地重复 evaluate_osrpa_integrand。
/// 返回该频率的 (total, os, ss) 被积函数值。
#[cfg(feature = "mpi")]
fn integrand_after_reduce(
    spin_polar_freq: &mut Vec<MatrixFull<f64>>,
    spin_channel: usize,
    lambda_omega: &Vec<f64>,
    lambda_weight: &Vec<f64>,
    sc_check: &[bool; 2],
) -> [f64; 3] {
    evaluate_osrpa_integrand(spin_polar_freq, spin_channel, lambda_omega, lambda_weight, sc_check)
}

#[cfg(feature = "mpi")]
pub(crate) struct FreqGrids {
    pub(crate) omega: Vec<f64>,
    pub(crate) weight: Vec<f64>,
    pub(crate) lambda_omega: Vec<f64>,
    pub(crate) lambda_weight: Vec<f64>,
}

#[cfg(feature = "mpi")]
pub(crate) fn build_freq_grids(scf_data: &SCF) -> FreqGrids {
    let num_freq = scf_data.mol.ctrl.frequency_points;
    let freq_grid_type = scf_data.mol.ctrl.freq_grid_type;
    let max_freq = scf_data.mol.ctrl.freq_cut_off;
    let (omega, weight) = if freq_grid_type == 0 {
        trans_gauss_legendre_grids(1.0, num_freq)
    } else if freq_grid_type == 1 {
        gauss_legendre_grids([0.0, max_freq], num_freq)
    } else if freq_grid_type == 2 {
        logarithmic_grid([0.0, max_freq], num_freq)
    } else {
        trans_gauss_legendre_grids(1.0, num_freq)
    };
    let num_lambda = scf_data.mol.ctrl.lambda_points;
    let (lambda_omega, lambda_weight) = gauss_legendre_grids([0.0, 1.0], num_lambda);
    FreqGrids { omega, weight, lambda_omega, lambda_weight }
}

/// 通用驱动：RHF（单张量）与 UHF/ROHF（双张量）共用。
/// 返回 [total, os, ss]（与串行 evaluate_osrpa_correlation_detailed_rayon 同约定，
/// 未含 0.5/π 因子之外的任何缩放——调用方语义与串行一致）。
#[cfg(feature = "mpi")]
fn osrpa_rayon_mpi_25d_impl(
    scf_data: &mut SCF,
    mpi_operator: &Option<MPIOperator>,
) -> anyhow::Result<[f64; 3]> {
    use mpi::collective::SystemOperation;
    use mpi::traits::*;
    use crate::ri_pt2::pt2_25d::{initialize_metadata_linear, redistribute_to_diag};

    let (mpi_op, mpi_ix) = match (mpi_operator, &scf_data.mol.mpi_data) {
        (Some(op), Some(ix)) => (op, ix),
        _ => panic!("MPI not initialized for the 2.5d SCSRPA evaluation"),
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
    let de_excitation = if let Some(value) = scf_data.mol.ctrl.rpa_de_excitation_parameters {
        value
    } else {
        [1.0, 1.0, 1.0, 0.0]
    };

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
        None => panic!("RI3MO should be initialized before the 2.5d SCSRPA evaluation"),
    };

    let (alpha_rimo, vir_range, occ_range) = &ri3mo_vec[0];
    let beta_rimo = if spin_channel == 2 { Some(&ri3mo_vec[1].0) } else { None };
    let n0_local = alpha_rimo.size[0];
    let n1_global = alpha_rimo.size[1];
    let n2_global = alpha_rimo.size[2];

    let ctx = initialize_metadata_linear(&grid, n2_global);
    let mut n0_global_tmp: u64 = 0;
    grid.cart_comm
        .all_reduce_into(&(n0_local as u64), &mut n0_global_tmp, &SystemOperation::sum());
    let n0_global = n0_global_tmp as usize;

    // 2.5D 重分布（α，β）：每 rank 持 row∪col 块的完整 [n_aux×n_vir] 切片；
    // 本地核只用 ownership 块（对角唯一划分）。
    let redistributed_alpha =
        redistribute_to_diag(&grid, &ctx, alpha_rimo, n0_global, &local_n0_range);
    let redistributed_beta = match beta_rimo {
        Some(b) => Some(redistribute_to_diag(&grid, &ctx, b, n0_global, &local_n0_range)),
        None => None,
    };

    // 每自旋占据数（镜像串行：num_elec≤1e-6 时为 0）
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

    // ---------------------------------------------------------------
    // ω=0：归约后的 χ₀^σ(0) → special radius → sc_check（所有 rank 一致）
    // ---------------------------------------------------------------
    let mut sc_check = [false; 2];
    {
        let mut spin_polar_freq: Vec<MatrixFull<f64>> = vec![MatrixFull::empty(); 2];
        for i_spin in 0..spin_channel {
            let mut partial = MatrixFull::new([n0_global, n0_global], 0.0);
            let tensor = if i_spin == 0 { &redistributed_alpha } else { redistributed_beta.as_ref().unwrap() };
            local_spin_response_25d(
                tensor, &ctx, grid.rank as usize, n2_global, 0.0,
                eigenvalues[i_spin], &scf_data.occupation[i_spin],
                occ_start, vir_start, num_occ[i_spin], num_state,
                frac_spin_occ, de_excitation,
                &mut partial,
            );
            let mut reduced = vec![0.0f64; partial.data.len()];
            grid.cart_comm.any_process().all_reduce_into(&partial.data[..], &mut reduced[..], &SystemOperation::sum());
            let mut polar = MatrixFull::new([n0_global, n0_global], 0.0);
            polar.data.copy_from_slice(&reduced);
            spin_polar_freq[i_spin] = polar;
        }
        let mut special_radius = [0.0f64; 2];
        for i_spin in 0..spin_channel {
            let polar_freq = spin_polar_freq.get(i_spin).unwrap();
            special_radius[i_spin] = evaluate_special_radius(polar_freq);
            sc_check[i_spin] = special_radius[i_spin] > 0.8f64;
        }
        if spin_channel == 1 {
            sc_check = [false, false];
            special_radius[1] = special_radius[0];
        }
        if scf_data.mol.ctrl.print_level > 0 {
            println!(
                "Special radius of non-interacting response matrix: ({:16.8}, {:16.8})",
                special_radius[0], special_radius[1]
            );
        }
        // 持久化 ω=0 谱半径：SCC15（scc15_for_rxdh7 MPI 分支）直接复用，
        // 免去第二次完整重分布 + ω=0 响应重算（原每次 SCC15 MPI 运行重复一次）。
        scf_data.energies.insert(String::from("special_radius"), vec![special_radius[0], special_radius[1]]);
    }

    // ---------------------------------------------------------------
    // 频率积分：每频率本地 partial → allreduce → integrand（冗余、各 rank 一致）
    // ---------------------------------------------------------------
    let mut rpa_c_energy = 0.0_f64;
    let mut rpa_c_energy_os = 0.0_f64;
    let mut rpa_c_energy_ss = 0.0_f64;

    for (omega, weight) in grids.omega.iter().zip(grids.weight.iter()) {
        let mut spin_polar_freq: Vec<MatrixFull<f64>> = vec![MatrixFull::empty(); 2];
        for i_spin in 0..spin_channel {
            let mut partial = MatrixFull::new([n0_global, n0_global], 0.0);
            let tensor = if i_spin == 0 { &redistributed_alpha } else { redistributed_beta.as_ref().unwrap() };
            local_spin_response_25d(
                tensor, &ctx, grid.rank as usize, n2_global, *omega,
                eigenvalues[i_spin], &scf_data.occupation[i_spin],
                occ_start, vir_start, num_occ[i_spin], num_state,
                frac_spin_occ, de_excitation,
                &mut partial,
            );
            let mut reduced = vec![0.0f64; partial.data.len()];
            grid.cart_comm.any_process().all_reduce_into(&partial.data[..], &mut reduced[..], &SystemOperation::sum());
            let mut polar = MatrixFull::new([n0_global, n0_global], 0.0);
            polar.data.copy_from_slice(&reduced);
            spin_polar_freq[i_spin] = polar;
        }

        let [integr, integr_os, integr_ss] = integrand_after_reduce(
            &mut spin_polar_freq,
            spin_channel,
            &grids.lambda_omega,
            &grids.lambda_weight,
            &sc_check,
        );

        if scf_data.mol.ctrl.print_level > 1 {
            println!(
                " (freq, weight, rpa_c): {:16.8},{:16.8},{:16.8}, {:16.8}, {:16.8}",
                omega, weight, integr, integr_os, integr_ss
            );
        }

        rpa_c_energy += integr * weight;
        rpa_c_energy_os += integr_os * weight;
        rpa_c_energy_ss += integr_ss * weight;
    }

    rpa_c_energy *= 0.5 / PI;
    rpa_c_energy_os *= 0.5 / PI;
    rpa_c_energy_ss *= 0.5 / PI;

    Ok([rpa_c_energy, rpa_c_energy_os, rpa_c_energy_ss])
}

/// 闭壳层（RHF）入口：分派用。
#[cfg(feature = "mpi")]
pub fn close_shell_osrpa_rayon_mpi_25d(
    scf_data: &mut SCF,
    mpi_operator: &Option<MPIOperator>,
) -> anyhow::Result<[f64; 3]> {
    osrpa_rayon_mpi_25d_impl(scf_data, mpi_operator)
}

/// 开壳层（UHF/ROHF）入口：分派用（ROHF 的 semi_eigenvalues 在 impl 内处理）。
#[cfg(feature = "mpi")]
pub fn open_shell_osrpa_rayon_mpi_25d(
    scf_data: &mut SCF,
    mpi_operator: &Option<MPIOperator>,
) -> anyhow::Result<[f64; 3]> {
    osrpa_rayon_mpi_25d_impl(scf_data, mpi_operator)
}

/// 仅计算 ω=0 谱半径（SCC15 用）：镜像串行 evaluate_special_radius_only，
/// 经 2.5D 重分布 + 归约路径，保证 MPI 下所有 rank 结果一致。
#[cfg(feature = "mpi")]
pub fn evaluate_special_radius_only_25d(
    scf_data: &SCF,
    mpi_operator: &Option<MPIOperator>,
) -> [f64; 2] {
    use mpi::collective::SystemOperation;
    use mpi::traits::*;
    use crate::ri_pt2::pt2_25d::{initialize_metadata_linear, redistribute_to_diag};

    let (mpi_op, mpi_ix) = match (mpi_operator, &scf_data.mol.mpi_data) {
        (Some(op), Some(ix)) => (op, ix),
        _ => panic!("MPI not initialized for the 2.5d SCSRPA evaluation"),
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
    let de_excitation = if let Some(value) = scf_data.mol.ctrl.rpa_de_excitation_parameters {
        value
    } else {
        [1.0, 1.0, 1.0, 0.0]
    };
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
        None => panic!("RI3MO should be initialized before the 2.5d SCSRPA evaluation"),
    };
    let (alpha_rimo, vir_range, occ_range) = &ri3mo_vec[0];
    let beta_rimo = if spin_channel == 2 { Some(&ri3mo_vec[1].0) } else { None };
    let n0_local = alpha_rimo.size[0];
    let n1_global = alpha_rimo.size[1];
    let n2_global = alpha_rimo.size[2];

    let ctx = initialize_metadata_linear(&grid, n2_global);
    let mut n0_global_tmp: u64 = 0;
    grid.cart_comm
        .all_reduce_into(&(n0_local as u64), &mut n0_global_tmp, &SystemOperation::sum());
    let n0_global = n0_global_tmp as usize;

    let redistributed_alpha =
        redistribute_to_diag(&grid, &ctx, alpha_rimo, n0_global, &local_n0_range);
    let redistributed_beta = match beta_rimo {
        Some(b) => Some(redistribute_to_diag(&grid, &ctx, b, n0_global, &local_n0_range)),
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

    let mut special_radius = [0.0f64; 2];
    for i_spin in 0..spin_channel {
        let mut partial = MatrixFull::new([n0_global, n0_global], 0.0);
        let tensor = if i_spin == 0 { &redistributed_alpha } else { redistributed_beta.as_ref().unwrap() };
        local_spin_response_25d(
            tensor, &ctx, grid.rank as usize, n2_global, 0.0,
            eigenvalues[i_spin], &scf_data.occupation[i_spin],
            occ_range.start, vir_range.start, num_occ[i_spin], num_state,
            frac_spin_occ, de_excitation,
            &mut partial,
        );
        let mut reduced = vec![0.0f64; partial.data.len()];
        grid.cart_comm
            .any_process()
            .all_reduce_into(&partial.data[..], &mut reduced[..], &SystemOperation::sum());
        let mut polar = MatrixFull::new([n0_global, n0_global], 0.0);
        polar.data.copy_from_slice(&reduced);
        special_radius[i_spin] = evaluate_special_radius(&polar);
    }
    if spin_channel == 1 {
        special_radius[1] = special_radius[0];
    }
    special_radius
}
