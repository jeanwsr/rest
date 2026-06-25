# RI-r2 Semi-direct 优化总结

日期：2026-05-08

本文记录 `src/lib_rint/mod.rs` 中当前 RI-r2 semi-direct 路径的优化工作。目标是让 RI-r2 的数据流更接近成熟的 SCF RI-J/K 路径，同时继续保留 R2 metric 的正规处理方式，即 `metric_eigh/pinv`。

## 主要修改

### 1. 改为 kept-batch-first 的 semi-direct 数据流

RI-r2 的 semi-direct 路径已经围绕 kept/metric batch 重新组织。

现在主循环的目标结构是：

```text
for kept batch Q:
    分配 rho_Q
    分配 H_Q

    for AO shell block:
        生成 B_block_Q
        rho_Q += B_block_Q^T D
        H_Q += B_block_Q C_occ

    K += H_Q H_Q^T
    J += B_Q rho_Q

    丢弃 rho_Q, H_Q, B_Q
```

这替代了之前不够理想的模式：每个 shell block 可能生成或暂存更大的 `B[pair, all_kept]` 对象。

### 2. 避免长期保存完整 fitted B

优化后的路径避免把完整 fitted 张量

```text
B[pair, all_kept]
```

作为核心的长期中间量保存。现在主要处理当前 batch 的 block：

```text
B_block_Q[pair_len, kept_len]
```

这样可以降低内存压力，也避免把完整 fitted 存储作为默认数据流。

### 3. K 路径使用 batch-local H

K-like contraction 是二次型，不能像 J 那样压缩成只含 kept 维度的 rho 向量。因此实现中会构造当前 batch 的半变换张量：

```text
H_Q[ao, occ, kept_len]
```

只有当当前 kept batch 的所有 AO/shell block 贡献都累积完成之后，才形成 K 贡献：

```text
K += H_Q H_Q^T
```

这保留了必须存在的跨 shell 项：

```text
(sum_s h_s) (sum_s h_s)^T
```

而不是错误或不完整的形式：

```text
sum_s h_s h_s^T
```

### 4. 明确区分 J 和 K 的职责

优化后的路径明确区分两个 contraction：

```text
J path:
    rho_Q += B_Q^T D
    J += B_Q rho_Q

K path:
    H_Q += B_Q C_occ
    K += H_Q H_Q^T
```

这样可以把 J-like 的线性压缩和 K-like 的二次半变换分开处理。

### 5. semi-direct 内存规划与诊断日志

semi-direct 路径现在会打印更清晰的内存/cache 规划，包括：

```text
incore_peak
H_batch
K_scratch_peak
kept_batch
batch_fitted_cache
kept_batches
fitted_shell_block_passes
```

这些日志可以帮助判断运行时间主要耗在 metric 对角化、raw three-center 积分生成、H 构建、J/K contraction、cache I/O，还是重复 kept-batch pass。

### 6. 移除 REST_R2_DIRECT_KEPT_BLOCK override

主 kept-batch planner 中移除了手动环境变量 override：

```text
REST_R2_DIRECT_KEPT_BLOCK
```

它可能把 `kept_batch` 强制压到过小的值，例如 256，从而引入额外的 fitted-shell-block pass，并造成明显变慢。

现在 kept batch 由自动内存 planner 选择。

### 7. 保留正规的 metric 处理

R2 metric 路径仍然使用正规的特征值/伪逆处理：

```text
metric_eigh / pinv
```

没有引入 Cholesky 分解，因为 R2 metric 不保证具有标准 SCF RI-J/K Coulomb metric 那样的正定结构。

### 8. 增加更大 benchmark 覆盖

新增了更大的水簇 benchmark，用于观察规模化行为：

```text
benchmark_water8_ccpvdz_scf_vs_r2_semidirect_shell_blocks
```

它用于补充已有的 `(H2O)4/cc-pVDZ` benchmark。

### 9. 后续补充：放宽后重新约束 kept-batch planner

在移除手动 `REST_R2_DIRECT_KEPT_BLOCK` 后，曾经尝试让自动 planner 更积极地选择较大的 `kept_batch`。这对 `(H2O)8/cc-pVDZ` 这类仍能一次放下的体系可以减少重复 pass，但对真正大体系会有内存超限风险。

因此后续又补充了更稳健的内存规划：

```text
kept_batch 预算 =
    总内存预算
  - metric_factor 常驻内存
  - J/K 输出矩阵
  - 并行 worker 的 raw_3c 临时块
```

然后只把剩余预算的一部分用于 kept batch，默认由：

```text
REST_R2_DIRECT_KEPT_MEMORY_FRACTION = 0.60
```

控制。同时 worker 输出队列从无界 channel 改为有界队列，避免大体系中 `local_fitted_Q` 在队列中无限堆积。

## 观察到的 Benchmark 行为

代表性的 release benchmark 观察如下：

```text
(H2O)4/cc-pVDZ 默认：
    R2/SCF 大约 5-6x
    kept_batches = 1
    fitted_shell_block_passes = 1

(H2O)8/cc-pVDZ 早期默认策略：
    R2/SCF 大约 8.3x
    kept_batch 大约 519
    kept_batches = 4
    fitted_shell_block_passes = 4

(H2O)8/cc-pVDZ 在移除 override 前强制 kept_batch=256：
    R2/SCF 大约 13.6x
    kept_batches = 8
    fitted_shell_block_passes = 8

(H2O)8/cc-pVDZ 在放宽并重新约束 planner 后：
    kept_batch = 2025
    kept_batches = 1
    fitted_shell_block_passes = 1
    R2/SCF 大约 4x
```

强制过小的 kept batch 会因为增加重复 raw-3c/H-build 工作而明显变慢。移除手动 override 并使用自动内存 planner，可以避免这类配置导致的性能退化。

## 已经改善的部分

这轮优化的主要价值不只是当前 benchmark wall time 的下降，更重要的是：

- 数据流更清晰、更安全；
- 降低长期保存完整 `B[pair, all_kept]` 的风险；
- K 路径使用 batch-local H 生命周期；
- 保留正确的 K 累积不变量；
- 内存诊断日志更清楚；
- 移除了危险的手动 kept-batch override；
- 增强了大体系 benchmark 可见性；
- 自动 planner 现在会考虑固定常驻内存和并行临时块，避免对大体系过度乐观。

简而言之，当前实现已经更适合作为后续性能优化的基础。

## 仍然存在的性能瓶颈

benchmark 中观察到的主要耗时仍然是：

```text
metric_eigh
raw_3c / fit_h
```

当前修改改善了数据流和内存行为，但没有消除这两个主要运行时成本。后续优化应重点关注：

- 在数学上允许时减少或复用 metric diagonalization；
- 加速 raw three-center R2 积分生成；
- 改进 shell/block 调度与筛选；
- 在内存允许时减少重复 pass；
- 改善 H-build 的 packing/scatter 局部性；
- 为大体系校准 `REST_R2_DIRECT_KEPT_MEMORY_FRACTION`、worker 数和 batch fitted cache 策略。

## 实际理解

这一轮优化把 RI-r2 semi-direct 从依赖大中间量的 semi-direct 实现，推进到更接近 batch-local RI-J/K 的数据流。

它主要是一次正确性、内存分布和 profiling 改进。进一步提速需要针对真正占主导的 kernel，而不只是继续改变高层数据流。
