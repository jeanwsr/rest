# 内置分子动力学（MD / AIMD）

`job_type = "md"` 时启用内置分子动力学：纯 QM 即 AIMD（每步先收敛 SCF 再移动原子核，MD 循环与伞形采样在 REST 内部，QM/MM 的 MM 侧用内嵌 OpenMM 做单点）。所有 MD 参数放在 `[md]` 区块，也可写在 `[ctrl]` 中（`[md]` 优先）。

## 常规 MD 关键词
- `ensemble`：取值String类型。`"nvt"`(缺省，Langevin) | `"nve"`(VelocityVerlet) | `"opt"`(几何优化) | `"sp"`(单点能量/受力)
- `dt`：取值f64类型。时间步长，单位fs。缺省0.5
- `steps`：取值usize类型。MD/优化步数。缺省100
- `temperature`：取值f64类型。温度，单位K。缺省300.0
- `friction`：取值f64类型。Langevin 摩擦系数。缺省0.02。单位由 `friction_units` 决定：
    取 `"ase"` 时以 ASE 时间单位计（1 ASE 时间单位 ≈ 10.18 fs），与旧 `ase.md.langevin.Langevin` 的
    定义一致，旧 ASE 脚本里的 friction 数值可直接照抄（0.02 约对应 τ = 509 fs）；取 `"fs_inv"` 时以 1/fs 计。
- `friction_units`：取值String类型。`"ase"`(缺省) | `"fs_inv"`(单位 1/fs)
- `seed`：取值u64类型。随机数种子，0表示随机。缺省0
- `init_velocities`：取值String类型。初速度来源：`"maxwell"`(缺省) | `"file"` | `"zero"`
- `velocities_file`：取值String类型。`init_velocities = "file"` 时的速度文件（每行 `Sym vx vy vz`，Å/fs）
- `restart_input`：取值String类型。从该 restart 文件同时读取**位置与速度**续跑（覆盖 `[geom]` 与 `init_velocities`，无需改写 `[geom]`）
- `traj_interval`：取值usize类型。轨迹与 restart 写出间隔。缺省10
- `equil_steps`：取值usize类型。预平衡步数：此前用同一恒温器积分但不写任何输出，production 阶段从其后重新计步。缺省0
- `out_prefix`：取值String类型。所有输出文件的前缀。缺省空
- `restart_output`：取值String类型。restart 文件名。缺省 `"md_restart"`
- `outputs`：取值String列表。输出开关：`md_log` | `traj` | `dipole` | `energy_force` | `umbrella_csv` | `restart`
- `opt_algorithm`：取值String类型。`ensemble = "opt"` 的优化器：`"fire"`(缺省) | `"lbfgs"`
- `opt_fmax`：取值f64类型。优化的力收敛判据，单位eV/Å。缺省0.02
- `opt_steps`：取值usize类型。优化的最大步数。缺省200

## QM/MM 关键词
- `qmmm_mm_file`：取值String类型。MM 的 GRO 文件（SOL 残基自动按 TIP3P 处理）；不提供 `qmmm_system_xml` 时用内嵌 OpenMM 自动建 MM 体系
- `qmmm_system_xml`（别名 `mm_system_xml`）：取值String类型。完整的 OpenMM System XML（保留 CMAP/CustomTorsion 等所有力）
- `qmmm_gro_skip`：取值usize类型。GRO 文件最前面与 `[geom]` QM 块重复的原子数（QM 原子须在前）
- `qmmm_links`：取值`[(usize,usize)]`类型。共价切断键的 `[qm_bdry, mm_host]` 对（全局 GRO 索引），link H 追加在 `[geom]` 最后
- `qmmm_link_r_eq`：取值f64类型。link H 键长，单位Å。缺省1.09
- `qmmm_mm_cutoff`：取值f64类型。嵌入点电荷的选取半径，单位Å。缺省10.0
- `qmmm_pbc`：取值bool类型。MM 是否使用周期性边界。缺省true
- `qmmm_lj`：取值元素→`[sigma(Å), epsilon(kJ/mol)]`。QM–MM 的 LJ 参数
- `qmmm_build_box`：取值bool类型。围绕 `[geom]` 溶质自动搭 TIP3P 水盒并写出 `auto_box.gro`。缺省false
    - `qmmm_box_margin`：水盒外边距，单位Å。缺省11.0
    - `qmmm_box_spacing`：水分子网格间距，单位Å。缺省3.1
- `qmmm_build_system`：取值bool类型。由经典力场**内部生成** `system_qmmm.xml`（QM/MM 分区规则自动套用）。缺省false
    - `qmmm_ff`：取值String列表。OpenMM 力场文件（如 `["amber14-all.xml", "amber14/tip3p.xml"]`，名字由 OpenMM 自带数据目录解析；自定义残基模板写文件路径）
    - `qmmm_mm_pdb`：取值String类型。标准残基的 PDB（残基/原子名须与力场匹配）
    - `qmmm_qm_atoms`：取值usize列表。QM 原子下标（0-based，PDB 顺序，可散落）；给出后 REST 自动重排
    - `qmmm_frontier`：取值usize列表。需电荷清零的 frontier MM 原子
    - `qmmm_frontier_rescale`：取值String类型。被清零的 frontier 电荷是否均摊回同残基其余 MM 原子：`"residue"`(缺省) | `"none"`
- `qmmm_top`：取值String类型。力场拓扑文件（`.top`，递归解析 `#include`），需配合 `qmmm_ff_dir` 使用
    - `qmmm_ff_dir`：取值String类型。力场参数目录（含 `ffnonbonded.itp`/`ffbonded.itp`/`cmap.itp`/`nbfix.itp`）
    - `qmmm_nb_cutoff`：取值f64类型。内部生成的 MM System 的 PME 截断，单位nm。缺省1.0（与 `qmmm_mm_cutoff` 的嵌入截断相互独立）

## 伞形采样关键词
- `umbrella_cv`：取值String类型。`"dihedral"`(4原子) | `"distance"`(2) | `"angle"`(3) | `"distance_diff"`(4，d(i,j)−d(k,l))
- `umbrella_atoms`：取值usize列表。CV 原子在组合数组（QM 在前，MM 在后）中的下标
- `umbrella_center`：取值f64类型。CV 中心；角度类用度，距离类用Å
- `umbrella_kappa`：取值f64类型。偏置强度，单位kcal/mol
- `umbrella_potential`：取值String类型。`"cosine"`(缺省) | `"harmonic"`
- `umbrella_sum_kappa`：取值f64类型。`distance_diff` 追加的第二项偏置强度 ½·ks·(s−s0)²（s = d1 + d2），单位kcal/mol
- `umbrella_sum_center`：取值f64类型。上式 s0，单位Å

## 纯 MM 关键词（`pure_mm = true`）
- `pure_mm`：取值布尔类型（写在 `[ctrl]`）。设为 `true` 时整个体系为纯 MM，不做电子结构计算；此时不需要 `xc`、`basis_path` 与 `[geom]`
- `mm_file`：取值String类型。全 MM 的 GRO 文件
- `mm_system_xml`：取值String类型。完整的 OpenMM System XML
- `mm_cutoff`：取值f64类型。MM 截断，单位Å

## 配置示例

- 例子一：纯 QM 的 NVT AIMD（水分子，B3LYP/def2-SVP）
    ```toml
    [ctrl]
    job_type   = "md"
    xc         = "b3lyp"
    basis_path = "def2-svp"

    [geom]
    unit = "angstrom"
    position = """
    O    0.00000000    0.00000000    0.00000000
    H    0.75700000    0.58600000    0.00000000
    H   -0.75700000    0.58600000    0.00000000
    """

    [md]
    ensemble      = "nvt"
    dt            = 0.5
    steps         = 2000
    temperature   = 300.0
    friction      = 0.02
    seed          = 42
    traj_interval = 1
    outputs       = ["md_log", "traj", "restart"]
    ```
- 例子二：纯 QM 的几何优化
    ```toml
    [ctrl]
    job_type   = "md"
    xc         = "b3lyp"
    basis_path = "def2-svp"

    [geom]
    unit = "angstrom"
    position = """
    O    0.00000000    0.00000000    0.00000000
    H    0.75700000    0.58600000    0.00000000
    H   -0.75700000    0.58600000    0.00000000
    """

    [md]
    ensemble      = "opt"
    opt_algorithm = "fire"
    opt_fmax      = 0.02
    opt_steps     = 200
    outputs       = ["restart"]
    ```
- 例子三：QM/MM，乙烷 + 自动 TIP3P 水盒
    ```toml
    [ctrl]
    job_type   = "md"
    xc         = "b3lyp"
    basis_path = "6-31gs"
    eri_type   = "ri-v"
    spin       = 1
    charge     = 0.0

    [geom]
    unit = "angstrom"
    position = """
    C       12.50000000     12.50000000     13.26750000
    C       12.50000000     12.50000000     11.73250000
    H       13.52571000     12.50000000     13.64213000
    H       11.98714000     13.38846000     13.64213000
    H       11.98714000     11.61154000     13.64213000
    H       13.01286000     13.38846000     11.35787000
    H       11.47429000     12.50000000     11.35787000
    H       13.01286000     11.61154000     11.35787000
    """

    [md]
    ensemble         = "nvt"
    dt               = 0.5
    steps            = 2000
    temperature      = 300.0
    seed             = 42
    qmmm_build_box   = true
    qmmm_box_margin  = 8.0
    qmmm_box_spacing = 3.1
    qmmm_mm_cutoff   = 999.0
    qmmm_lj          = { C = [3.39967, 0.457730], H = [2.64953, 0.065689] }
    outputs          = ["md_log", "traj", "energy_force", "restart"]
    ```
- 例子四：QM/MM 共价切断
    ```toml
    [ctrl]
    job_type   = "md"
    xc         = "b3lyp"
    basis_path = "6-31gs"

    [geom]
    unit = "angstrom"
    position = """
    C     14.40000000    12.50000000    12.92000000
    H     14.70000000    13.53000000    13.13000000
    H     15.20000000    11.99000000    12.39000000
    H     14.21000000    11.99000000    13.87000000
    C     13.13000000    12.50000000    12.08000000
    H     13.13000000    13.37000000    11.42000000
    H     13.13000000    11.63000000    11.42000000
    H     12.29794971    12.50000000    12.63470020
    """

    [md]
    ensemble        = "nvt"
    dt              = 0.5
    steps           = 2000
    temperature     = 300.0
    qmmm_mm_file    = "init.gro"
    qmmm_system_xml = "system_qmmm.xml"
    qmmm_gro_skip   = 7
    qmmm_links      = [[4, 7]]
    qmmm_link_r_eq  = 1.00
    qmmm_mm_cutoff  = 999.0
    outputs         = ["md_log", "traj", "energy_force", "restart"]
    ```
- 例子五：REST 内部生成 `system_qmmm.xml`——标准残基（`[ctrl]`/`[geom]` 见例子四）
    ```toml
    [md]
    ensemble              = "opt"
    opt_fmax              = 0.02
    qmmm_build_system     = true
    qmmm_ff               = ["amber14-all.xml", "amber14/tip3p.xml"]
    qmmm_mm_pdb           = "system.pdb"
    qmmm_qm_atoms         = [0, 1, 2, 3, 4, 5, 6]
    qmmm_links            = [[4, 7]]
    qmmm_frontier         = [7]
    qmmm_frontier_rescale = "residue"
    ```
- 例子六：REST 内部生成 `system_qmmm.xml`——含非标准残基（`[ctrl]`/`[geom]` 见例子四）
    ```toml
    [md]
    ensemble          = "nvt"
    dt                = 0.5
    steps             = 2000
    temperature       = 300.0
    qmmm_build_system = true
    qmmm_ff           = ["ligand.xml", "amber14-all.xml", "amber14/tip3p.xml"]
    qmmm_mm_pdb       = "system.pdb"
    qmmm_qm_atoms     = [0, 1, 2, 3, 4, 5, 6]
    qmmm_links        = [[4, 7]]
    qmmm_frontier     = [7, 8, 9]
    ```
- 例子七：REST 内部生成 `system_qmmm.xml`——力场以拓扑/参数文件形式给出（`[ctrl]`/`[geom]` 见例子四）
    ```toml
    [md]
    ensemble      = "nvt"
    dt            = 0.5
    steps         = 2000
    temperature   = 300.0
    qmmm_top      = "topol.top"
    qmmm_ff_dir   = "forcefield"
    qmmm_mm_file  = "system.gro"
    qmmm_qm_atoms = [0, 1, 2, 3, 4, 5, 6]
    qmmm_links    = [[4, 7]]
    qmmm_frontier = [7, 8, 9]
    ```
- 例子八：伞形采样（`[ctrl]`/`[geom]` 见例子三/四）
    ```toml
    [md]
    ensemble           = "nvt"
    dt                 = 0.5
    steps              = 50000
    equil_steps        = 2000
    temperature        = 300.0
    friction           = 0.02
    qmmm_build_box     = true
    umbrella_cv        = "distance_diff"
    umbrella_atoms     = [6022, 6000, 6003, 6000]
    umbrella_center    = 0.0
    umbrella_kappa     = 200.0
    umbrella_potential = "harmonic"
    umbrella_sum_kappa = 50.0
    umbrella_sum_center = 1.9
    outputs            = ["md_log", "umbrella_csv", "restart"]
    ```
- 例子九：续跑
    ```toml
    [md]
    restart_input = "md_restart"
    steps         = 20000
    outputs       = ["md_log", "traj", "restart"]
    ```
- 例子十：纯 MM 的 MD
    ```toml
    [ctrl]
    pure_mm  = true
    job_type = "md"

    [md]
    ensemble      = "nvt"
    dt            = 0.5
    steps         = 20000
    temperature   = 300.0
    friction      = 0.02
    mm_file       = "system.gro"
    outputs       = ["md_log", "traj", "restart"]
    ```
