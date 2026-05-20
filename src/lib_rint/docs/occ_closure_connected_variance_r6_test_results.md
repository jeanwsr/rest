# Occupied-only connected variance R^-6 test results

本文档记录 `rest/src/lib_rint/mod.rs` 中测试级 occupied-only RI-Coulomb connected variance 原型的长程标度验证结果。

当前计算量不是能量，而是片段间 connected variance 描述符：

```text
X_AB = <0_A 0_B | V_AB Q_A Q_B V_AB | 0_A 0_B>
```

在当前 RI 实现中使用：

```text
X_AB^RI = sum_{q,r} C_A(q,r) C_B(q,r)
```

其中 `C_A` 和 `C_B` 是通过 occupied-only closure 构造的片段 mode covariance。若要转成 closure-PT2-like 能量，需要再定义有效分母：

```text
E_disp^{occ-cl} ~= - X_AB / Delta_eff
```

## Test commands

```bash
cargo test -p rest --release --lib occ_closure_ri_coulomb_connected_noble_dimers_loglog_slope -- --ignored --nocapture
```

```bash
cargo test -p rest --release --lib occ_closure_ri_coulomb_connected_he2_loglog_slope -- --ignored --nocapture
```

```bash
cargo test -p rest --release --lib occ_closure_ri_coulomb_connected_methane_dimer_loglog_slope -- --ignored --nocapture
```

```bash
cargo test -p rest --release --lib occ_closure_ablation_feature_scaling_loglog -- --ignored --nocapture
```

## Ne2 / cc-pVDZ

主基组：

```text
/home/cfh/rest_workspace/rest/basis-set-pool/cc-pVDZ
```

辅助基组：

```text
/home/cfh/rest_workspace/rest/basis-set-pool/def2-SV(P)-JKFIT
```

测试点：

| R / Ang | R / bohr | X_AB | log R | log \|X_AB\| |
|---:|---:|---:|---:|---:|
| 4.000 | 7.558905 | 3.8774330352014067e-5 | 2.02272628 | -10.15775212 |
| 5.000 | 9.448631 | 9.8784002059595223e-6 | 2.24586983 | -11.52515998 |
| 7.000 | 13.228083 | 1.2798267751985090e-6 | 2.58234206 | -13.56878582 |
| 10.000 | 18.897261 | 1.4860416691945653e-7 | 2.93901701 | -15.72197966 |

线性拟合：

```text
log |X_AB| vs log R slope = -6.070948
```

结论：`Ne2/cc-pVDZ` 的 RI-Coulomb connected variance 长程斜率接近 `-6`，符合色散型 leading behavior 的预期。

## He2 / cc-pVDZ

主基组：

```text
/home/cfh/rest_workspace/rest/basis-set-pool/cc-pVDZ
```

辅助基组：

```text
/home/cfh/rest_workspace/rest/basis-set-pool/def2-SV(P)-JKFIT
```

测试点：

| R / Ang | R / bohr | X_AB | log R | log \|X_AB\| |
|---:|---:|---:|---:|---:|
| 4.000 | 7.558905 | 7.0206930370925913e-6 | 2.02272628 | -11.86664862 |
| 5.000 | 9.448631 | 1.8404253810534989e-6 | 2.24586983 | -13.20551383 |
| 6.000 | 11.338357 | 6.1635467484932634e-7 | 2.42819139 | -14.29944327 |
| 7.000 | 13.228083 | 2.4442740842120073e-7 | 2.58234206 | -15.22434747 |
| 8.000 | 15.117809 | 1.0969786381330828e-7 | 2.71587346 | -16.02553594 |
| 10.000 | 18.897261 | 2.8756631202322385e-8 | 2.93901701 | -17.36439745 |

线性拟合：

```text
log |X_AB| vs log R slope = -6.000004
```

结论：`He2/cc-pVDZ` 的 RI-Coulomb connected variance 长程斜率几乎严格为 `-6`。

## CH4--CH4 / STO-3G

主基组：

```text
/home/cfh/rest_workspace/rest/basis-set-pool/sto-3g
```

辅助基组：

```text
/home/cfh/rest_workspace/rest/basis-set-pool/def2-SV(P)-JKFIT
```

几何：

```text
两个四面体 CH4 片段沿 z 轴平移，C--C 距离作为 R。
```

测试点：

| R / Ang | R / bohr | X_AB | log R | log \|X_AB\| |
|---:|---:|---:|---:|---:|
| 5.000 | 9.448631 | 5.2412866958203217e-5 | 2.24586983 | -9.85635844 |
| 6.000 | 11.338357 | 1.8677183888735735e-5 | 2.42819139 | -10.88820789 |
| 7.000 | 13.228083 | 7.7165052138947538e-6 | 2.58234206 | -11.77214899 |
| 8.000 | 15.117809 | 3.5605255096250029e-6 | 2.71587346 | -12.54560241 |
| 10.000 | 18.897261 | 9.6531142417762285e-7 | 2.93901701 | -13.85081507 |
| 12.000 | 22.676714 | 3.2938497069183468e-7 | 3.12133857 | -14.92603865 |

线性拟合：

```text
log |X_AB| vs log R slope = -5.794946
```

结论：`CH4--CH4/STO-3G` 的 RI-Coulomb connected variance 长程斜率接近 `-6`。它比 `He2` 和 `Ne2` 偏离稍大，主要因为甲烷是有限大小的多原子片段，`5--12 Ang` 区间仍包含多极展开的有限距离修正和取向效应；但整体趋势已经符合色散型 leading behavior。

## Ablation feature scaling

为了对应下面的 ablation 设计：

```text
Model A: baseline
Model B: baseline + E_J^{r^-2} + E_K^{r^-2}
Model C: baseline + X_AB^RI
Model D: baseline + X_AB^RI + E_J^{r^-2} + E_K^{r^-2}
```

这里额外测试了片段间 cross 版本的 `E_J^{r^-2}` 和 `E_K^{r^-2}`。注意：这里不是全体系 raw `E_J/E_K`，而是 dimer interaction / cross-fragment component；如果直接用全体系 raw 值，单体内部常数会进一步掩盖长程趋势。

### Slope summary

| system | X_AB^RI slope | E_J^{r^-2}_cross slope | E_K^{r^-2}_cross slope |
|---|---:|---:|---:|
| He2 / cc-pVDZ | -6.000004 | -2.012918 | -99.433940 |
| Ne2 / cc-pVDZ | -6.070948 | -2.009925 | -97.389268 |
| CH4--CH4 / STO-3G | -5.794946 | -2.023485 | -75.422856 |

### Interpretation

`X_AB^RI` 在三个测试上都接近 `R^-6`，因此它符合 closure / Unsold 型色散 numerator 的长程行为。对应 ablation 里的 Model C，这个特征是当前最物理的 long-range dispersion gene。

`E_J^{r^-2}_cross` 在三个测试上都接近 `R^-2`，这正是普通 direct density-density 的长程行为。它的数值通常也比 `X_AB^RI` 大很多，因此如果 Model B 在训练集上变好、但远距离曲线外推变差，很可能就是模型把它当成 shortcut 使用了。

`E_K^{r^-2}_cross` 远距离快速衰减到接近零，斜率拟合给出非常大的负数，本质上是指数衰减的交换/重叠型信息，不是 leading dispersion。它可能对平衡区或短程 Pauli/重叠修正有用，但不应被解释为长程色散。

因此，对这组测试而言，预期 ablation 结论是：

```text
Model C 应该负责正确的 long-range R^-6 行为。
Model B 可能在训练集上提供 shortcut，但长程物理错误，因为 E_J^{r^-2} 是 R^-2。
Model D 如果优于 C，收益更可能来自短程/平衡区修正，而不是长程色散本身。
```

## Notes

- 当前结果来自 `--release` 测试。
- 这些测试是原型验证，不是生产 API。
- 输出的 `X_AB` 是 descriptor / variance numerator，不是最终色散能。
