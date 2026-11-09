# REST × SHARC 非绝热动力学接口

本文档说明如何在 REST 的 GW-BSE 模块上运行 SHARC 非绝热动力学（surface hopping），
包括安装位置、新增的输入选项、后端实现、物理定义、使用流程与验证结果。

* SHARC 安装目录：`/home/qiruigao/SHARC`（可执行文件在 `/home/qiruigao/SHARC/bin`）
* REST 源码：`/home/qiruigao/rest_workspace/rest`
* 接口脚本：`/home/qiruigao/SHARC/bin/SHARC_REST.py`
* 示例：`/home/qiruigao/SHARC/examples/SHARC_REST/`

---

## 1. SHARC 安装

SHARC 3.0 源码取自 <https://github.com/sharc-md/sharc>，用 gfortran 编译：

```bash
cd ~/SHARC/source
make            # 串行执行（模块间有依赖，-j 并行会失败）
make install    # 复制 sharc.x / laser.x / data_extractor.x / diagonalizer.x 到 bin/
```

链接需要 `libfftw3`。系统只装了运行库 `libfftw3.so.3`，因此建立了
`~/local/lib/libfftw3.so -> /usr/lib/x86_64-linux-gnu/libfftw3.so.3`，
并在 `source/Makefile` 的 `LD` 中加入 `-L/home/qiruigao/local/lib`。
`wfoverlap_ascii.x` 直接使用发行包自带的预编译版本。

环境变量（已写入 `~/.bashrc`）：

```bash
export SHARC=/home/qiruigao/SHARC/bin
export PATH=$SHARC:$PATH
```

验证：运行发行包自带测试轨迹

```bash
cd ~/SHARC/tests/INPUT/Analytical_overlap
mkdir -p $SCRATCH/TRAJ && export SCRADIR=$SCRATCH && bash run.sh
```

`output.lis` 给出完整轨迹（能量守恒误差 < 1e-4 eV / 2 fs），说明 `sharc.x`、
`wfoverlap`、Python 工具均可正常工作。

---

## 2. 新增的 REST 输入选项

全部位于 `[quasiparticle_methods]` 块，且只在 `gw_or_bse = "bse_sharc"` 时生效，
不影响原有任何计算路径。

| 关键字 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `gw_or_bse` | string | — | 设为 `"bse_sharc"` 触发一次 SHARC 步 |
| `sharc_nstates` | int | 0 | 总电子态数（含基态）。`0` 时退回 `gradient_nroots + 1` |
| `sharc_do_grad` | bool | true | 计算所有态的解析核梯度 |
| `sharc_do_nac` | bool | true | 计算非绝热耦合矢量 |
| `sharc_do_overlap` | bool | true | 与上一步计算态间重叠矩阵（需要 `sharc_previous_state`） |
| `sharc_do_dipole` | bool | false | 输出态/跃迁偶极矩 |
| `sharc_previous_state` | string | `""` | 上一步的态文件路径（JSON），空则跳过重叠 |
| `sharc_state_output` | string | `sharc_state.json` | 本步写出的态文件（供下一步读取） |
| `sharc_output` | string | `sharc_step.json` | 本步结果文件（接口脚本解析该文件） |
| `sharc_step_index` | int | -1 | 轨迹步编号（仅记录） |
| `sharc_nac_mode` | string | `"overlap"` | `"off"` 时不计算 NAC |
| `sharc_nac_fd_step` | float | 1e-3 | NAC 中对 AO 交叉重叠做中心差分时的位移（bohr） |

一次 SHARC 步做的事情：

1. 按 `gw_scheme` 跑 GW（或从 checkpoint 恢复），得到 QP 能量；
2. 用与 `bse_gradient` 完全相同的引擎（`GwGradEngine` + `BseGradEngine`）求解
   `sharc_nstates - 1` 个 BSE 根；
3. 逐态组装解析梯度：`dE_I/dR = dE_GS/dR + dΩ_I/dR`；
4. 计算与非绝热耦合；
5. 写出 `sharc_step.json`（接口读取）与 `sharc_state.json`（下一步读取）。

---

## 3. 后端实现

### 3.1 新模块 `src/ri_bse/sharc.rs`

* **交叉几何 AO 重叠**：把当前分子克隆一份、只改写 libcint 环境中的核坐标，
  用 `Molecule::int_cross` 得到 `⟨μ(R₀)|ν(R₁)⟩`，再变换到 MO 基：
  `S^MO = C₀ᵀ S^AO C₁`。
* **行列式重叠**（严格解，`tests` 中与暴力行列式逐元素比对到 1e-16）：

  ```
  ⟨Φ₀|Φ₀⟩ = det(S_oo)
  ⟨Φ₀|Φ_i^a⟩ = det(S_oo) · (S_oo⁻¹ S_ov)_{ia}
  ⟨Φ_i^a|Φ_j^b⟩ = det(S_oo) · [ (1+s1)(1+s4) − s2 s3 ]
      s1 = (S_vo S_oo⁻¹)_{ai} − 1
      s3 = (S_oo⁻¹)_{ji}
      dl = (S_vv − S_vo − S_ov + S_oo)_{abij}
      s2 = (R − S_vo − S_ov + S_oo)_{abij} + dl·s1 ,  R = S_vo S_oo⁻¹ S_ov
      s4 = (S_oo⁻¹ S_ov)_{jb} − 1 + dl·s3
  ```

  其中 `i,j` 取自**完整**占据空间、`a,b` 取自完整虚空间；只有 BSE 振幅限制在
  活动窗口内。非 TDA（RPA）态按 `X_I X_J − Y_I Y_J` 度规处理。
* **重叠矩阵** `S_IJ = ⟨Ψ_I(R₀)|Ψ_J(R₁)⟩`，`I,J = 0..nstates`，0 为基态。
* **NAC** `d_IJ = ∂/∂R′ ⟨Ψ_I(R)|Ψ_J(R′)⟩|_{R′=R}`：对同一重叠表达式关于
  末态几何求导。MO 系数随核坐标变化（轨道响应完整保留），BSE 振幅冻结在
  当前几何。该量正是 overlap/local-diabatization 传播所用的耦合，
  写出前做了反对称化并把对角元置零。
* **相位约定**：`fix_mo_phases`（每个 MO 最大 AO 系数取正）、
  `fix_amplitude_phases`（每个根最大振幅取正）、
  `align_phases_to_previous` / `mo_overlap_signs` + `apply_mo_signs` +
  `apply_amplitude_signs`（沿轨迹与上一步对齐）。没有这些约定，
  LAPACK 的任意本征矢符号会让 `det(S_oo)` 或某个虚轨道在步间随机变号，
  重叠矩阵出现 ±1 跳变。

### 3.2 驱动 `src/post_scf_analysis/mod.rs`

* 新增 `gw_or_bse = "bse_sharc"` 分支与 `sharc_step()`；
* 把 BSE 活动空间窗口逻辑抽成 `bse_gradient_cutoffs()`，`bse_gradient` 与
  `bse_sharc` 共用（行为完全不变）。

### 3.3 接口 `SHARC/SHARC_REST.py`

完整的 SHARC 3 接口（`QM.in` → REST → `QM.out`）：

* 解析 `QM.in`（几何、单位、态数、请求 `H/DM/GRAD/NACDR/OVERLAP`、
  `init/cleanup/savedir/step`）；
* 读取 `REST.template` / `REST.resources`（键值文本格式）；
* 生成 `rest_ctrl.in` 并调用 `<programdir>/rest -i rest_ctrl.in`；
* 解析 `sharc_step.json`，写出 `QM.out` 的 `! 1 / ! 2 / ! 3 / ! 5 / ! 6` 块；
* 管理 `savedir`：把上一步的 `sharc_state.json` 作为 `sharc_previous_state`
  传回 REST，运行成功后更新。

已在 `SHARC/bin/setup_traj.py` 的 `Interfaces` 字典中注册 REST（编号 12），
并提供 `get_REST` / `prepare_REST`。

---

## 3.4 Σ_c 的频率处理：AC-GW（默认）

SHARC 管线默认使用 **AC-GW**（`gw_variant = "ac"`）：Σ_c 只在虚轴采样，用 Thiele 连分式
（Padé）延拓到实轴。解析梯度走 `ri_gw::gw_grad` 的 AC 分支（`GwGradConfig::gw_variant`），
与能量使用**同一个** `GwGradEngine` 缓存（`AcQpData`），因此能量与梯度严格同泛函。

可调参数（写入 `[quasiparticle_methods]` 或 `REST.template`）：

| 关键字 | 默认 | 含义 |
|---|---|---|
| `gw_variant` | `ac` | `ac` / `cd` / `fscd` / `wac` |
| `ac_num_samples` | 16 | 连分式用的虚轴采样点数 |
| `ac_omega_max` | 5.0 | 采样频率上限（Ha） |
| `ac_eta` | 1e-3 | AC 求值展宽 |

**为什么默认改成 AC**：CD 变体在拉伸/解离几何上出现非物理的梯度尖峰（8–535 eV/Å），
导致轨迹能量不守恒；换 AC 后同样三个体系的尖峰全部消失，能量漂移降到 1e-3 eV 以下
（见 `sharc_rest_test/RESULTS.md` 第 3 节）。

## 4. 使用流程

### 4.1 直接跑一个 REST 单点

```toml
[quasiparticle_methods]
     gw_or_bse = "bse_sharc"
     gw_scheme = "extrapolated"
     bse_spin = "singlet"
     bse_tda = true
     gradient_bse_window = 0.0     # 0 = 保留全部虚轨道，活动空间不随几何漂移
     sharc_nstates = 4
     sharc_do_grad = true
     sharc_do_nac = true
     sharc_do_overlap = false
```

```bash
rest -i ctrl.in
# → sharc_step.json, sharc_state.json
```

### 4.2 通过 SHARC 运行

`QM/REST.template`（电子结构参数）与 `QM/REST.resources`（路径/并行）见
`SHARC/examples/SHARC_REST/`。`QM/runQM.sh`：

```bash
#!/bin/bash
$SHARC/SHARC_REST.py QM.in
```

`QM.in` 由 SHARC 写；如需手工单点测试，可用
`SHARC/examples/SHARC_REST/QM.in`。

### 4.3 轨迹

SHARC 的 `coupling` 推荐设为 `overlap`（local diabatization）：
重叠矩阵是严格量，且不需要小时间步。

```
nstates      4
actstates    4
coupling     overlap
```

若使用 `coupling nac`，则使用本文档中的 NAC（注意第 6 节的近似）。

---

## 5. 验证

### 5.1 单元测试（严格数学）

```bash
cargo test --lib ri_bse::sharc
```

* `determinant_overlap_formula_matches_bruteforce`：随机非正交 MO 重叠矩阵，
  所有 `(i,a,j,b)` 与暴力行列式一致（误差 < 1e-12）；
* `ground_excited_overlap_matches_bruteforce`；
* `derivative_of_overlap_matches_finite_difference`：解析导数 vs 有限差分。

### 5.2 端到端（H₂O / def2-SVP / PBE / G₀W₀ + TDA-BSE）

| 检查 | 结果 |
|---|---|
| 同几何重叠矩阵 | 单位矩阵（1e-15） |
| **bra/ket 双向一致性** | `S_IJ(R0,R1) = S_JI(R1,R0)`，最大偏差 **4.9e-15** |
| **有限差分求和规则** | `d_01 + d_10 = 8.5e-8`（应为 0） |
| 4 步轨迹重叠对角元 | 0.99996 ×3，逐步一致、恒正；非对角 1e-12 |
| 振幅响应 A（H₂O, H1-x） | `d01` 全数值 0.0708 vs 冻结振幅 0.0198，A ≈ 0.051 a.u. |
| 态间重叠的非对角元（+h 与 −h） | 严格反对称：S(+h) = −S(−h)（1e-16） |
| 基态 NAC（d₀₀）vs 重叠有限差分 | 一致到 1e-9（如 −0.094948 a.u.） |
| 各态对角 NAC（d_II）vs 有限差分 | 一致到 1e-6（0.064139 / 0.074791） |
| NAC 反对称性 / 对角 | `d_IJ = −d_JI`，对角严格为 0 |
| QM.out 块 1/2/3/5/6 | 3×3、3×3×3、3×3×3、3×3×3×3、3×3 |
| SHARC 自带轨迹测试 | `tests/INPUT/Analytical_overlap` 通过 |
| 与 BSE 能量路径的一致性 | `bse_sharc` 与 `bse` 卡片给出完全相同的 Ω（0.265816893058 Ha） |

> 注意：本接口在 SHARC 步内先显式跑 GW、再把 QP 能量钉给梯度引擎
> （`gradient_qp_from_checkpoint` 的语义），因此激发能与 `bse` 卡片逐位一致。
> 已有的 `bse_gradient` 卡片不跑显式 GW，其 Ω 与能量路径相差约 0.02 Ha，
> 属于该路径既有行为，未在本次工作中改动。

**关于 NAC 的定量准确性**：对角元（I=J，含基态）与全数值微分一致到 1e-9/1e-6；
非对角元与“包含振幅响应”的全数值微分仍有偏差（H₂O/def2-SVP 上 d₀₁ 解析 0.0198
vs 全数值 0.0688，d₁₀ −0.0198 vs −0.2279 a.u.）。

分解：`d_full = d_frozen + A`，其中 `A_IJ = Σ_μ c_Iμ ċ_Jμ` 是 BSE 本征矢（CI 系数）
随几何的响应 —— 即第 6.2 条所说缺失项。但实测偏差大于 `A` 的单独估计：把同一对
几何反过来算（bra/ket 交换）得到的同一矩阵元并不相等（例如 ⟨Ψ₀(R₀)|Ψ₂(R₁)⟩ 与
⟨Ψ₂(R₁)|Ψ₀(R₀)⟩ 相差 4.3 倍，仅差整体符号），说明**用有限差分作“真值”本身混入了
两套态表示**，不能干净地分离出 `A`。要严格定量，需要 (i) 加一个“冻结末态振幅”的
诊断开关，或 (ii) 直接实现 `∂H_BSE/∂R` 在态间的矩阵元（即 BSE 本征矢解析导数）。
因此：**动力学请用 overlap 传播；NACDR 仅作诊断/对照**。

---

## 5.3 已知问题与验证状态（截至本次工作）

修复历史（均由上述一致性检验发现）：

1. 重叠的 bra 侧振幅误用当前几何的 `c`（应为上一步的 `c`）；
2. 相位对齐跑在 MO 重定相之前，导致对齐被撤销、`S_II` 出现 −1 跳变；
3. **交叉几何 AO 重叠用错了坐标**：bra 侧的 libcint 分子被建成当前几何，
   于是 `S_IJ` 实际是 `C_prevᵀ·S^AO(R_cur,R_cur)·C_cur`，只有在几何接近时
   近似正确（0.001 Å 位移下误差 ~1e-4，正好与真实非对角元同量级，因而被掩盖）。

第 3 条修复后：双向一致性 4.9e-15、求和规则 8.5e-8、轨迹对角元 0.99996 恒定。

**尚未完成的验证**（因此目前只能称“内部一致性通过”，不能说“充分验证”）：

* 与 SHARC 自带 `wfoverlap` 的独立交叉比对（把 MO 系数、CI 系数、行列式、
  交叉 AO 重叠按 wfoverlap 格式写出，直接对比 S）——这是文档标准的“金标准”；
* 真实多步轨迹的物理检验（能量守恒、布居、hop 统计）；
* NAC 的冻结振幅近似与全数值微分仍有 A ≈ 0.05 a.u. 的差（仅影响 NACDR，
  不影响 overlap 路线）。

## 6. 限制与注意事项

1. **仅自旋限制闭壳层、仅单重态**。REST 的 GW-BSE 目前没有 SOC，
   `soc` 请求会被接口明确拒绝。
2. **NAC 是“冻结振幅”近似**：`d_IJ = ∂<Ψ_I(R)|Ψ_J(R′)>/∂R′` 只包含轨道响应
   （MO 系数随核坐标的变化，通过交叉几何 AO 重叠的导数完整保留），
   **不含 BSE 振幅随几何的变化**（CI 系数响应）。对角元与数值微分一致；
   不同态之间的非对角元在振幅混合处有 O(0.1) 量级偏差（见 5.2 的实测数字）。
   若需要严格 NAC，需要 BSE 本征矢的解析导数（`∂H_BSE/∂R` 在态间的矩阵元），
   属于后续工作。**动力学推荐使用 overlap 传播**（严格重叠矩阵 + 局部透热化），
   此时不依赖 NAC。
3. **活动空间必须逐步一致**：重叠与 NAC 都要求 `occ_start / nocc / vir_start /
   nvir` 在各步相同。使用 `gradient_bse_window = 0`（全虚空间）或固定的
   `gradient_bse_nvir`；用能量窗口会让轨道漂进漂出，REST 会以明确报错终止。
4. **基组与原子顺序必须一致**（同一轨迹内天然满足）。
5. 激发态永久偶极与激发态间跃迁偶极没有静态 BSE 响应，当前只输出
   `μ₀₀`（基态）与 `μ₀I`（跃迁偶极）；`sharc_do_dipole` 默认关闭。
6. 计算量：每步 = 1 次 GW + 1 次 BSE + 每个根一次梯度（`sharc_do_grad`）+
   少量重叠积分（NAC 需要 6N 次交叉重叠积分构造，通常远小于 GW 本身）。
