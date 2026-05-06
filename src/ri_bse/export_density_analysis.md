# Export Density 数值不稳定性分析

## 问题概要

在完全相同的输入条件下，重复计算 damped BSE 的 export_density 得到不一致的结果：

| 现象 | 描述 |
|------|------|
| **密度矩阵元符号反转** | ~10% 的矩阵元在两次计算间符号相反，绝对值基本吻合，具有聚集特征（整行/整列反转） |
| **网格密度偏差大** | 仅 ~15% 的网格点密度偏差 < 1e-4，其余偏差在 1e-4 ~ 1e-1 范围 |

---

## 误差机制分析

### 机制一：导出密度中 par_bridge 的乱序问题（高置信度）

**位置**: `damped.rs:273-277`

```rust
let grid_data: Vec<f64> = first_prod.iter_columns_full()
    .zip(grid_val.iter_columns_full())
    .par_bridge()                              // <-- 问题
    .map(|(fp, gval)| { ... dot product ... })
    .collect();
```

`par_bridge()` 将串行迭代器转为并行。Rayon 的 `ParallelBridge` **不是** `IndexedParallelIterator`，因此 `.collect()` 的**输出顺序不保证与输入顺序一致**。这意味着：

- `grid_data[i]` 在不同运行中可能对应**不同的网格点**
- 即使密度矩阵完全一致，网格密度值也会被分配到错误的坐标上
- 这是 **85% 网格点出现 1e-4~1e-1 偏差的最可能原因**

**严重程度**: 高。直接导致网格密度计算结果不可重复。

### 机制二：线性系统的病态性（高置信度）

damped BSE 的 4-分量线性系统为：

```
[A-ω   B    -γI   0 ] [X ]   [μ]
[ B   A+ω    0    γI] [Y ] = [μ]
[γI    0   A-ω   B ] [X']   [0]
[0   -γI    B   A+ω] [Y']   [0]
```

其中 A ≈ diag(Δε) + W + V，B ≈ W' + V'。

当外场频率 ω 接近某个 BSE 激发能（即 ω ≈ Δε 的某个本征值），且展宽 γ 很小时，系统矩阵接近奇异。条件数 κ ≈ 1/γ（在共振附近）。这导致：

- 微小的矩阵构造误差被放大 1/γ 倍
- 近简并的激发通道对数值扰动极敏感
- 密度矩阵中符号反转的**聚集特征**（整行/整列反转）恰好对应某个近简并的激发模式

**严重程度**: 中高。解释了符号反转的聚集特征，也是网格密度偏差的次要贡献因素。

### 机制三：RI 积分构造的并行非确定性（中等置信度）

**位置**: `scf_io/mod.rs` 中的 `ao2mo_rayon_v02` 函数

```
rimat_chunk.data_ref().unwrap()
    .par_chunks_exact(num_bpair).enumerate()
    .for_each_with(sender, |sender, (q, chunk)| { ... sender.send(...) })
```

该函数使用 `crossbeam_channel` 收集并行计算结果。在多线程同时发送时，**信道接收顺序由操作系统调度决定**，不同运行中消息到达顺序不同。如果接收端代码不按索引严格排序（而只是按接收顺序处理），则 RI 积分矩阵会引入非确定性的浮点差异。

这些微小差异进入 A、B 矩阵后，对于病态系统（机制二）会被显著放大。

另外，在 `response_matrix` (`ri_gw/mod.rs:543-565`) 内部，`par_bridge().map()` 中嵌套了 `par_iter_mut()`，形成双层并行。虽然元素乘法本身是确定性的，但嵌套并行可能导致线程超额订阅，间接影响 BLAS 行为。

**严重程度**: 中。对常规体系影响很小，但对病态系统会通过机制二放大。

### 机制四：Davidson 等迭代求解器的收敛路径差异（低置信度）

对于 Pople、GMRES、Klopper 等迭代求解器，迭代过程在近简并的情况下可能沿不同路径收敛，导致解向量的符号组合不同。但用户报告 **dense 求解器也有此问题**，说明迭代求解器不是根本原因。

**严重程度**: 低（对 dense 求解器不适用）。

---

## 解决方案

### 方案一：修复 export_density 的并行乱序（必须修复）

**方法 A — 使用有序并行**：

将 `par_bridge()` 替换为带索引的并行方式，并在 collect 后按索引排序：

```rust
let mut grid_data_pairs: Vec<(usize, f64)> = first_prod.iter_columns_full()
    .zip(grid_val.iter_columns_full())
    .enumerate()
    .par_bridge()
    .map(|(idx, (fp, gval))| {
        let dot = fp.iter().zip(gval.iter()).fold(0.0, |acc, (x, y)| acc + x * y);
        (idx, dot)
    })
    .collect();
grid_data_pairs.sort_by_key(|(i, _)| *i);
let grid_data: Vec<f64> = grid_data_pairs.into_iter().map(|(_, v)| v).collect();
```

**方法 B — 完全串行**（简单但较慢）：

去掉 `par_bridge()`，直接使用串行 `map().collect()`。对于密度网格计算，这个操作通常不是性能瓶颈：

```rust
let grid_data: Vec<f64> = first_prod.iter_columns_full()
    .zip(grid_val.iter_columns_full())
    .map(|(fp, gval)| {
        fp.iter().zip(gval.iter()).fold(0.0, |acc, (x, y)| acc + x * y)
    })
    .collect();
```

### 方案二：改善线性系统的条件数（推荐）

**方法 A — 添加 Tikhonov 正则化**：

在线性系统对角线添加一个小量 λ：

```rust
for i in 0..4*occ_size*vir_size {
    linear_equation_lhs[[i, i]] += regularization;
}
```

λ 应远小于 γ 但足够稳定求解，建议 λ = γ × 1e-3。

**方法 B — 检查 γ 和 ω 参数**：

- 如果 γ 过小（如 < 1e-4），系统会接近奇异
- 如果 ω 过于接近某个激发能，建议稍微偏移 ω 或增大 γ

### 方案三：确保 RI 积分的确定性构造（推荐）

在 `ao2mo_rayon_v02` 的信道接收端，确保按索引排序后再组装矩阵：

```rust
let mut results_by_index: Vec<(usize, Vec<f64>)> = receiver.into_iter().collect();
results_by_index.sort_by_key(|(i, _)| *i);
for (_, data) in results_by_index {
    // 按序组装
}
```

### 方案四：并行区域中的 BLAS 线程控制（好的实践）

在调用 BLAS/LAPACK 的并行区域中设置 `omp_set_num_threads(1)`，避免嵌套并行导致的线程超额订阅和浮点非确定性：

```rust
use std::ffi::CString;
extern "C" { fn omp_set_num_threads(n: i32); }

// 在进入 Rayon 并行区域前
unsafe { omp_set_num_threads(1); }
```

---

## 诊断建议

### 1. 验证 par_bridge 乱序假说

在 `export_density` 函数开头添加以下调试代码，将网格点索引与密度值一起输出：

```rust
// 临时调试：禁用并行，按序计算
let grid_data: Vec<f64> = first_prod.iter_columns_full()
    .zip(grid_val.iter_columns_full())
    .enumerate()
    .map(|(i, (fp, gval))| {
        let val = fp.iter().zip(gval.iter()).fold(0.0, |acc, (x, y)| acc + x * y);
        if i < 10 {
            println!("Grid[{}] density = {:.12e}", i, val);
        }
        val
    })
    .collect();
```

如果串行版本下两次计算得到一致的网格密度（偏差 < 1e-12），则确认问题来自 par_brideg 乱序。

### 2. 验证线性系统病态性

计算并输出矩阵的条件数估计：

```rust
// 在 _dsolve 之前添加
use lapack::dgecon;
let mut anorm = 0.0;
// ... 计算矩阵的 1-范数 ...
let mut rcond = 0.0;
let mut work = vec![0.0; 4 * n];
let mut iwork = vec![0; n];
unsafe {
    dgecon('1', n, &linear_equation_lhs.data, n, &anorm, &mut rcond, &mut work, &mut iwork, &mut info);
}
println!("Estimated condition number: {:.2e}", 1.0 / rcond);
```

如果条件数 > 1e8（双精度机器精度的倒数），则病态性确认。

### 3. 对比不同求解器

固定修复 export_density 后，比较 dense 与 GMRES 求解器的密度矩阵结果。如果在 dense 求解器下结果仍然不一致，则问题出在矩阵构造阶段（机制三）；如果只有迭代求解器不一致，则问题主要在迭代收敛。

---

## 总结

| 问题 | 根因 | 修复优先级 |
|------|------|-----------|
| 网格密度值大量偏差 | `par_bridge()` 输出乱序（机制一） | **立即修复** |
| 密度矩阵元符号反转 | 线性系统病态 + 矩阵构造非确定性（机制二 + 三） | **高优先级** |
| 迭代求解器额外差异 | 收敛路径对初始条件敏感（机制四） | 补充修复 |

建议的修复路径：
1. 立即修复 `export_density` 中的 `par_bridge()` 乱序（方案一方法 A/B）
2. 在 RI 积分构造中按索引排序确保确定性（方案三）
3. 评估线性系统的条件数，必要时添加正则化（方案二）
4. 修复后验证 dense 求解器的可重复性，再检查迭代求解器
