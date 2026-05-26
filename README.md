# REST项目介绍和程序安装
  请参见[REST开发组页面](https://gitee.com/restgroup)。**以下为REST程序的具体使用说明**
# For English Users:
  - This manual can be used as a prompt file for state-of-the-art Large Language Models (LLMs), such as DeepSeek and Tongyi. By providing this content to an LLM, you can effectively utilize it as an online support assistant and to generate the input file for computational tasks you need. (Note: ChatGPT has not been tested due to restrictions by the US government.)
# 用于生成REST输入卡的系统提示词
- 基于Rust语言的新一代电子结构计算软件REST（Rust-based Electronic Structure Toolkit）由复旦大学化学理论研究中心开发，在徐昕教授的领导下，由张颖教授担任首席开发者完成。
- 根据用户需求，结合知识库和上下文，帮助用户生成REST程序的输入卡。 
- REST输入卡使用TOML格式，包含[ctrl]、[geom]和[geometric_pyo3]三个控制区
    - [ctrl]申明具体计算方法、（辅助）基组、数值方法参数等
    - [geom]提供研究体系的名字、结构以及结构相关的ghost原子、点电荷以及赝势等
	- [geometric_pyo3]设置的参数仅用于`opt_engine=geometric_pyo3`结构优化引擎的控制
- 生成输入卡之后，与程序手册上的关键词进行对比，做如下确认：
  - 输出格式是TOML
  - 输入卡必须包含[ctrl]和[geom]两个区块
  - [ctrl]中的大部分关键词有缺省设置。若用户无具体要求，不必出现在输入卡中
  - 如果用户没有明确要求，设置num_threads为10
  - [ctr]区块中必须声明的关键词：
     1. 计算方法和配置相关：`xc`，`basis_path`，`print_level`，以及`num_threads`等
     1. 体系相关: `spin`, `charge`, `spin_polarization`等
  - 仅当使用了`opt_engine=geometric_pyo3`时，才要申明[geometric_pyo3]区
  - 当调用geometric_pyo3引擎做缺省的最稳结构优化时，不需要申请[geometric_pyo3]
  - 调用的方法的关键词是否使用"xc"，不能无中生有地用其它的关键词，比如“method"等
  - D3BJ、D3以及D4是经验色散校正，需要用`empirical_dispersion`申明。比如"X3LYP-D3BJ"方法需要拆分成"xc=x3lyp"和"empirical_dispersion=d3bj"
  - 关键词`spin`和`charge`是在`[ctrl]`区，而不是在`[geom]`区
  - 分子结构的关键词是`position`，不能无中生有地用其它的关键词——比如“coord"和"molecule"等
  - 分子结构`position`的申明使用String，比如
`
"""
  H 0.0 0.0 0.0
  H 0.75 0.0 0.0
"""
`
  - 输入卡中不采用'''符号
  - 反复迭代比较，直至输入卡一次性全部满足上述要求
  - 输出REST程序的输入卡，使用String的格式，包含换行符号'\n'，并且对'"'符号进行'\"'转译

# Detailed descrption of `[ctrl]` block in the control file

## 系统设置相关关键词（Keyword）
- `num_threads`: 取值i32类型。任务最大可调用线程数目，缺省为1
- `print_level`: 取值i32类型。程序输出信息量，数字越大，输出信息量越多。缺省为1。0表示完全无输出
- `max_memory`: 取值f64类型。程序可使用的最大内存，单位为MB。缺省不设置，使用当前节点所有可用内存。
- `abort_on_mem_exceed`: 取值布尔类型。是否在内存超出限制时终止计算。缺省为true。**设置false存在风险，不建议一般用户设置该选项**。

## 具体计算任务相关关键词（Keyword）
- `job_type`: 取值String类型。设置计算任务类型。目前可以进行的计算任务为:
    1. `energy`: 单点能量计算（缺省）。等价设置有：`single point`，`single_point`等
    1. `opt`: 基于数值力的构型优化。等价设置有：`geometry optimization`, `relax`, `geom_opt`等
	1. `force`: 计算当前结构下的受力。等价设置有：`gradient`
	1. `numerical dipole`: 计算数值偶极。等价设置有：`numdipole`
- `auxbasis_response`：开启辅助基导数。缺省为true
- `opt_engine`: 取值String类型。构型优化引擎。可选项有：`LBFGS`、`geometric-pyo3`（缺省）
- `numerical_force`: 取值布尔类型。是否计算数值力。缺省为false
- `nforce_displacement`:　取值f64类型。数值力计算中的结构位移值，缺省是0.0013 Bohr
- `ndipole_displacement`:　取值f64类型。数值Dipole计算中的外电场位移值，缺省是3.0E-4 Bohr

## 计算体系相关关键词（Keyword）
- `charge`：取值f64类型。体系的总电荷数
- `spin`: 取值i32类型。体系的自旋多重度。假设体系总自旋量子数为s，则取值2s+1
- `spin_polarization`: 取值布尔类型。是否开放自旋极化。
    - 当spin取值1时，缺省为false（RHF计算）
    - 当spin取值大于1时，缺省为true（UHF计算）
    - 若要进行ROHF计算，需要在spin取值大于1时将spin_polarization设为false。目前仅支持高自旋（即所有未成对电子自旋平行）的情况。
- `outputs`: 取值Vec\<String\>。用于计算结束后输出结果。可输出的信息包括：
    - `dipole`    偶极
    - `fchk`　    Gaussian程序的fchk文件
    - `cube_orb`  格点化的轨道文件信息 
    - `molden`　　结果输出为molden程序的格式
    - `geometry`  输出分子结构文件
    - `force`     输出分子受力信息
    - `force_for_ghost_point_charges`  在[geom]部分输入ghost point charges时输出原子区域对这些ghost点电荷的作用力
- `cube_orb_setting`: 取值[f64;2]。`cube_orb`格点参数设置。前一个值(margin)是边界信息，第二个值(num_grids)是生成格点的数目。缺省值为[3.0, 80.0]
- `cube_orb_indices`: 取值Vec\<[usize;3]\>。
	- 指定需要生成cube文件的一组轨道。缺省为空，即`[]`
    - 每一个矢量元素`[usize;3]`代表一组轨道信息：
    - 第一个值(start_orb)为起始轨道的index
    - 第二个值(end_orb)为截止轨道的index，与start_orb组成闭区间(closed interval)
    - 第三个值(i_spin)是这组轨道所在的自旋通道。0为alpha自旋;1为beta自旋
    - **注意**：REST的轨道排序从0开始，因此第一个轨道是0。如果HOMO是第X个轨道，在REST的排序是X-1
	- 举例来说：闭壳层基态苯分子体系的HOMO-1、HOMO和LUMO是第20、21和22个轨道，在REST中的排序是19、20和21。打印alpha自旋通道上这三个轨道的设置是[[19,21,0]]
- `cube_orb_type`: 取值String类型。指定生成的cube文件类型：
  - `wavefunction`: 生成轨道波函数的cube文件（缺省）
  - `density`: 生成轨道概率密度的cube文件（|ψ|²）

## 计算方法相关关键词（Keyword）
- `xc`：取值String类型。调用的电子结构计算方法。目前REST支持
    0. 波函数方法：HF、MP2
    1. 局域密度泛函近似：LDA
    2. 广义梯度泛函近似：BLYP、PBE、xPBE、XLYP
    3. 动能密度泛函近似：SCAN、M06-L、MN15-L、TPSS
    4. 杂化泛函近似：B3LYP、X3LYP、PBE0、M05、M05-2X、M06、M06-2X、SCAN0、MN15
    5. 第五阶泛函近似：XYG3、XYGJOS、XYG7、xDH-PBE0、sBGE2、ZRPS、scsRPA、R-xDH7、RPA@PBE、RPA@B3LYP
    - HF、LDA、BLYP、PBE、B3LYP、PBE0是自洽场计算方法，若用户未申明具体基组，则使用def2-TZVPP基组 (`basis_path = {basis_set_pool}/def2-TZVPP`)
    - MP2、XYG3、XYGJOS、XYG7、sBGE2、ZRPS、scsRPA、R-xDH7、RPA@PBE、RPA@B3LYP为后自洽场计算方法。若用户未申明具体基组，则使用def2-QZVPP基组 (`basis_path = {basis_set_pool}/def2-QZVPP`)
    - RPA@PBE、RPA@B3LYP表示后自洽场RPA计算使用PBE、B3LYP方法的轨道
- `empirical_dispersion`:　取值为String。针对低级别密度泛涵方法（包括LDA、BLYP、PBE、B3LYP、PBE0等）的经验色散校正方法。目前支持D3, D3BJ和D4。对于XYG3型双杂化泛涵比如XYG3、XYG7、XYGJOS、scsRPA、R-xDH7、RPA等不需要经验色散校正
- `post_ai_correction`：取值String。AI辅助的校正方法。目前仅支持SCC15，并只能和R-xDH7重整化双杂化泛涵方法相匹配。相关文章见：Wang, Y.; Lin, Z.; Ouyang, R.; Jiang, B.; Zhang, I. Y.; Xu, X. Toward Efficient and Unified Treatment of Static and Dynamic Correlations in Generalized Kohn–Sham Density Functional Theory. JACS Au 2024, 4 (8), 3205–3216. https://doi.org/10.1021/jacsau.4c00488
- `post_xc`：取值Vec\<String\>。采用自洽收敛的轨道和密度，进行不同的交换－关联泛函(xc)的计算。允许的方法包括REST支持的"xc"方法
- `post_correlation`：取值Vec\<String\>。采用自洽收敛的轨道和密度，进行后自洽场高等级相关能方法计算。允许的方法包括PT2、sBGE2、RPA、scsRPA等
- `pt2_ss_factor`: 取值f64。采用 Spin-Component-Scaled 方式计算 MP2 型相关能（SCS-MP2）时, 用于控制自旋平行（same spin）分量贡献的缩放系数，适用于 `xc` 关键词为 MP2、SCS-MP2或双杂化泛函，以及 `post_correlation` 设置为 PT2 的情况
- `pt2_os_factor`: 取值f64。适用场景与 `pt2_ss_factor` 一致，采用 SCS-MP2 方法计算相关能贡献时，用于缩放自旋反平行（opposite spin）分量贡献的系数

## DFT积分格点相关关键词（Keyword）
- `grid_gen_level`: 取值usize。格点精度等级，数值越大越精确。缺省为3
- `pruning`: 取值String。DFT方法或sap初猜所选用格点筛选。目前，REST支持nwchem，sg1以及none。其中none为不筛选。缺省为nwchem
- `radial_grid_method`: 取值String。径向格点的生成方法。目前REST支持truetler，gc2nd， delley, becke, mura_knowles及lmg。缺省为truetler

## 基组相关关键词（Keyword）
- `eri_type`: 取值String类型。设置四中心积分计算方法。选项：
    - `analytic`: 解析计算（仅限开发测试，不推荐用于实际计算）
	- `ri-v`: 使用密度拟合方法近似计算。(默认值)
	> 注意：REST中的analytic算法并未被充分优化，仅供程序开发测评使用，不建议在实际计算中使用
- `basis_type`: 取值String类型。设置基函数类型。选项：
    - `Spheric`: 球谐型基函数（默认值）
	- `Cartesian`：笛卡尔型基函数
- `basis_path`: 取值String类型，必需参数，无默认值。指定基组文件路径。该文件夹内包含各元素的基组JSON文件（如`C.json`、`O.json`）
    - 格式示例：  
      ```bash
      # 完整路径格式（明确指定）
      basis_path = /opt/rest_workspace/rest/basis-set-pool/cc-pVTZ
      
      # 简写格式（程序自动搜索）
      basis_path = cc-pVTZ
      ```
	- **路径优先级**： 
	  当使用简写格式时，程序按以下顺序搜索 `cc-pVTZ` 文件夹：
	  1. **当前工作目录**（`$PWD/cc-pVTZ`）
      2. 环境变量 `REST_BASIS_DIR`指定目录（`$REST_BASIS_DIR/cc-pVTZ`）
      3. REST docker 默认路径：`/opt/rest_workspace/rest/basis-set-pool/cc-pVTZ`
	  4. REST conda 默认路径： `$CONDA_PREFIX/share/rest/basis-set-pool/cc-pVTZ`
      5. `$REST_HOME/rest/basis-set-pool/cc-pVTZ`
      6. 自动从BSE库在线获取（若本地未找到），并保存在当前工作目录
    - 以上 2-5 情形不区分基组名称的大小写。 
- `auxbas_path`: 取值String类型，指定辅助基组路径，仅当`eri_type=ri-v`时需要设置  
    缺省值：def2-universal-jkfit。（该基组较大，建议根据体系换用匹配的辅助基组）
	设置方式与`basis_path`相同，程序搜索路径一致。

### 自定义基组使用方法
REST 支持用户自定义或混合基组：
1. 创建基组文件夹  
    在任意位置（如当前工作目录或基组库目录）创建文件夹：
	```bash
	mkdir my_custom_basis
	```
2. 准备基组文件  
    在文件夹内放入各元素的基组 JSON 文件：
	```bash
	# 文件夹结构示例
    my_custom_basis/
    ├── C.json
    ├── O.json
    ├── H.json
    └── N.json
	```
3. 在输入文件中引用  
    ```bash
	# 方式1：使用相对路径（从当前目录查找）
    basis_path = ./my_custom_basis
    
    # 方式2：使用文件夹名（按搜索优先级查找）
    basis_path = my_custom_basis
    
    # 方式3：使用绝对路径（直接定位）
    basis_path = /home/user/my_custom_basis
	```
### 配置示例
```plaintext
# 示例1：使用标准基组和缺省辅助基组（简写格式）
eri_type = ri-v
basis_type = Spheric
basis_path = cc-pvtz                    # 程序自动搜索 cc-pvtz 文件夹

# 示例2：使用标准基组，同时申明辅助基组（简写格式）
eri_type = ri-v
basis_type = Spheric
basis_path = cc-pvtz                    # 程序自动搜索 cc-pvtz 文件夹
auxbas_path = def2-universal-jkfit      # 程序自动搜索同名辅助基组文件夹

# 示例3：使用自定义基组（相对路径）
eri_type = ri-v
basis_type = Cartesian
basis_path = ./my_project_basis         # 使用当前目录下的自定义基组
auxbas_path = ./my_aux_basis            # 使用自定义辅助基组

# 示例4：使用完整路径
eri_type = ri-v
basis_type = Spheric
basis_path = /shared/basis/def2-TZVP    # 明确指定完整路径
auxbas_path = def2-universal-jkfit      # 简写格式，自动搜索
```

## 自洽场计算相关关键词（Keyword）
- `initial_guess`: 取值String。分子体系进行自洽场运算所用的初始猜测方法。目前REST支持:
	1. `sad` : 对体系各原子进行自洽场计算得到自洽的密度矩阵后，将多个密度矩阵按顺序置于对角位置后得到初始的密度矩阵进行自洽场运算。缺省为sad
    1. `vsap`: Superposition of Atomic Potentials的初始猜测方法。采用半经验方法对体系势能项进行估计，与libcint生成的动能项进行加和后得到初始的Fock矩阵
	1. `hcore`: Hcore则对应单电子近似初猜，直接将由libcint生成的hcore矩阵作为初始猜测的fock矩阵进行计算
- `guess_mix`: 取值布尔类型，是否采用混合HOMO和LUMO的方法获得对称性破缺初猜。由此可以破坏体系的空间对称性和自旋对称性，有助于得到单重态UHF波函数。缺省为false。
- `guess_mix_theta_deg`: 取值`[f64;2]`或f64，分别设置两个自旋通道的混合角度（单位：度）。
    - 设为0.0，则表示完全不混合
    - 在0.0-90.0范围内，角度越大，表示破坏原始初猜效果越显著。一般建议取值0.0-45.0。缺省为[15.0, 15.0]
- `chkfile`: 取值String。给定初始猜测所在位置/路径。缺省为none
- `mixer`：取值String。辅助自洽场收敛的方法。目前REST支持direct，diis，linear及ddiis。Direct对应不使用辅助收敛方法，linear对应于线性辅助收敛方法，diis对应于direct inversion in the iterative subspace。Diis是有效的加速收敛方法。缺省为diis
- `mix_param`: 取值f64。Diis方法或linear方法的混合系数。缺省为0.2
- `start_diis_cycle`: 取值i32。开始使用diis加速收敛方法的循环数。缺省为2
- `num_max_diis`: 取值i32。diis空间大小。缺省为8
- `max_scf_cycle`: 取值i32。自洽场运算的最大迭代循环数。缺省为100
- `noiter`: 取值布尔类型。是否跳过自洽场运算。缺省为false
- `scf_acc_rho`: 取值f64。自洽场运算密度矩阵的收敛标准。缺省为1.0e-8
- `scf_acc_eev`: 取值f64。自洽场运算能量差平方和的收敛标准。缺省为1.0e-6
- `scf_acc_etot`: 取值f64。自洽场运算总能量的收敛标准。缺省为1.0e-8
- `level_shift`: 取值f64。对于发生近简并振荡不收敛的情况，可以采用level_shift的方式人为破坏简并，加速收敛。单位为hartree，缺省值为0.0
- `start_check_oscillation`: 取值i32。开始检查并自洽场计算不收敛发生振荡的循环数。当监控到自洽场发生振荡，SCF能量上升的情况，开启一次线性混合方案（linear）。缺省为20
- `force_state_occupation`: 取值是Vector。 Constrained DFT (C-DFT) 计算方法。具体设置如下：
     - `[
  [reference, prev_state, prev_spin, target_spin, force_occ, force_check_min, force_check_max],
  [reference, prev_state, prev_spin, target_spin, force_occ, force_check_min, force_check_max],
  ...
]`
     - Vector中的每一项对应于一个轨道的约束。
     - `reference`: （可选）取值String。C-DFT的计算需要有一个常规的DFT计算结果，并以hdf5的格式存在`reference`中。若省略，则默认与chkfile相同。
     - `prev_state`和`prev_spin`：取值i32。定位需要约束的轨道在reference中的轨道序号和自旋通道
     - `target_spin`：（可选）取值i32。在C-DFT计算中，约束轨道的目标自旋通道
        - 若省略该值，则默认与`prev_spin`相同
        - 若给定，则会在指定自旋通道中寻找与prev_state/prev_spin最相似的轨道。
     - `force_occ`：取值f64。设置上述定位的轨道在约束DFT（C-DFT）计算中的取值
     - `force_check_min`和`force_check_max`：取值i32。在C-DFT的自洽计算中设置搜索窗口，仅从这个窗口中寻找和prev_state/prev_spin最相似的轨道
- `algorithm_jk`: 设置 Fock 矩阵计算中 J (Coulomb) 和 K (Exchange) 两部分的算法：
    - `ri-direct`: 强制使用 direct RI 算法。对于 RI-K 部分，取决于内存大小，可能会使用 semi-direct 算法 (储存相对较小的 $O(N^3)$ 的 $g_{\mu i, P}$)。
    - `ri-incore`: 强制使用 incore RI 算法 (储存完整的 Cholesky decomposed 3c-2e ERI $Y_{\mu \nu, P}$)。该算法对内存需求较大，但计算速度更快。
    - `ri`: 自动选择 incore 或 direct RI 算法，取决于自洽场计算前内存大小；在内存空间较大时选择更快的 incore 方法，内存空间较小时选择 ri-direct 方法。
    - `default`: 目前同 `ri`。
- `algorithm_j`: 设置 Fock 矩阵计算中 J (Coulomb) 部分的算法；该关键词是高级选项，一般用户建议使用`algorithm_jk`关键词进行整体设置。
    - 下述选项同 `algorithm_jk`：`ri-direct`, `ri-incore`, `ri`, `default`。
- `algorithm_k`: 设置 Fock 矩阵计算中 K (Exchange) 部分的算法；该关键词是高级选项，一般用户建议使用`algorithm_jk`关键词进行整体设置。
    - 下述选项同 `algorithm_jk`：`ri-direct`, `ri-incore`, `ri`, `default`。

## 后自洽场计算相关关键词（Keyword）
- `frozen_core_postscf`: 取值i32，且小于100的两位正整数或者一位正整数。对于后自洽场方法，包括MP2和第五阶密度泛函近似，需要考虑激发组态的贡献。由于原子的内层电子（core electrons）通常不参与化学成键，仅有最外几个价层参与（按主量子数划分）。因此我们可以采用冻芯近似（frozen core approximation）
    - 缺省值为`0`，代表考虑所有电子，不使用冻心近似
	- 当设置为一位数`n`的时候，不区分原子是主族元素还是过渡金属，冻心近似下只考虑涉及`n`个最外价层的电子激发组态。
	- 当设置为两位数`mn`的时候，则区分原子类型，对于主族元素考虑`n`个价层上的电子激发（个位上的数），而对过渡金属则考虑`m`个价层（十位上的数）
	- 几个示例和几点说明：
	    - `sp`电子所属电子层由`主量子数`来区分。以第三周期元素Si、P、S为例，`3s3p`属于最高第三价层，而`2s2p`是次高第二价层
	    - `d`电子所属电子层由`主量子数-1`来区分。以3d过渡金属Fe、Cu、Zn为例，`3d4s`属于最高第四价层，而`3s3p`是次高第二价层
	    - `f`电子所属电子层由`主量子数-2`来区分。以5d过渡金属Ir、Pt、Au为例，`4f5d6s`属于最高第六价层，而`4d5s5p`是次高第五价层
	    - `fronzen_core_postscf=1`（即`n=1`），表示第三周期元素仅考虑`3s3p`价层电子的贡献，而不考虑`1s2s2p`轨道的电子激发
		- 若`n`等于或大于主族元素占据轨道的电子层数，代表对于这个元素不采用冻心近似，等价于`n=0`.
	    - `fronzen_core_postscf=2`（即`n=2`），表示第二周期元素同时考虑`2s2p`最高价层和`1s`次高价层（即最低核层）的贡献，等价于`n=0`不开冻心近似。
	    - **注意**：对于传统密度泛函方法，本参数设置不起作用
- `frequency_points`：取值i32。对于RPA型的相关能计算方法，比如RPA、SCSRPA和R-xDH7等，需要对频率空间进行数值积分。这里设置频率积分的格点数目。缺省为20
- `freq_grid_type`：取值i32。对于RPA型相关能计算方法做格点化准备:
     - `0`: 代表使用modified Gauss-Legendre格点。缺省为0
	 - `1`: 代表standard Gausss-Legendre格点
	 - `2`: 代表Logarithmic格点。
- `lambda_points`：取值i32。对于SCSRPA和R-xDH7等方法，对于开窍层的强关联体系，需要对绝热涨落途径（lambda）数值积分。这里设置lambda积分的格点数目。缺省为20
## GW-BSE计算相关设置
- `quasiparticle_methods`: 取值String，设置为gw即开启GW计算准粒子能量，设置为bse即在计算或读取准粒子能量后进一步开启BSE计算垂直激发能。缺省为空，即不触发任何GW-BSE计算
- `gw_scheme`: 取值String，决定使用何种方式计算GW准粒子，无论进行GW还是BSE都需要设置此项。GW计算建议设置为extrapolated，即计算费米面附近一定范围内的准例子能量，其余轨道的准例子能量根据费米面附近的准粒子能量来外推。BSE计算还可以设置为parse from file，通过再设置`parse qp path`（取值String，读取纯数据文本文件的路径）即可读取预先已计算好的GW准例子能量用于BSE计算
- `threshold`: 取值f64，单位为Hatree，在extrapolated方案中决定计算费米面附近计算准粒子能量的SCF轨道范围，费米面加减threshold范围以外的轨道的准粒子能量将由已计算的准粒子能量外推，缺省为0.1
- `parse qp path`: 取值String，若设置gw_scheme=“parse from file”则必须设置此项，读取纯数据文本文件的路径,从此路径读取预先已计算好的GW准粒子能量用于BSE计算
- `scgw`: 取值String，若设置为evgw即开启循环迭代的GW计算，缺省为g0w0，即只进行一轮GW计算
- `evgw_rounds`: 取值usize，若设置scgw=“evgw”则必须设置此项，循环迭代evgw的次数
- `renormalized_singles`: 取值bool，设置为true即在进行GW计算之前先使用密度泛函的密度矩阵投影计算HF哈密顿量并将其对角化，得到的本征值是RS粒子能量，使用RS粒子能量初始化GW中的G部分。参考文献：J. Phys. Chem. Let. 2019, 10 (3), 447-452.
- `w_rs`: 取值bool，设置`renormalized_singles`=true时进一步设置`w_rs`=true可以进一步使用RS粒子能量初始化GW中的W部分
- `bse_spin`: 取值String，需要进行BSE计算时必须设置此项，指定计算何种自旋的激发，可以设置为”singlet”或”triplet”
- `bse_cutoff_energy`: 取值f64，单位为Hatree，进行BSE计算时DFT能级高于此能量的轨道的准粒子能量将不参与BSE kernel的构建，用于削减构建的BSE kernel的维数，减少对角化计算时间，缺省为1.5
- `bse_tda`: 取值bool，设置为true则使用TDA近似，即BSE kernel只保留左上部分的子矩阵。缺省为false
## RRS-PBC计算相关设置
- `pbc_eigenval`: 取值String，用于指定存储k点和能级信息的文件路径。如果设置为"none"或"None"则直接打印到标准输出。缺省为"none"。相关文章见Zhang, I.Y., Jiang, J., Gao, B. *et al.* RRS-PBC: a molecular approach for periodic systems. *Sci. China Chem.* **57**, 1399–1404 (2014). https://doi.org/10.1007/s11426-014-5183-y
## 溶剂化计算相关设置
- `solvent_model`: 取值String, 用于指定用于计算的溶剂模型。目前支持CPCM, COSMO, IEFPCM, SS(V)PE。缺省为CPCM。溶剂化梯度计算目前只支持CPCM, COSMO。
- `solvent_enabled`: 取值bool，设置为true则启用溶剂化计算。如果没有`solvent_enabled`字段但有`solvent_model`的设置且内容非空的时候，同样启用溶剂化计算。其他情况缺省为false。
- `solv_epsilon`: 取值f64, 为溶质的介电常数。缺省为1.0 (真空)。介电常数表可以参考 http://sobereva.com/g09/k_scrf.htm 的最后。
- `solvent_ri`: 取值bool, 设置为true为溶剂化能计算开启辅助基，设置为false溶剂化计算不开启辅助基。缺省为true。目前溶剂化梯度(job_type = "opt"/"force")计算没有用辅助基。
# Detailed descrption of [geometric_pyo3] block in the control file
- `maxiter`：取值i32。结构优化的最大步数上限。缺省值：300
- `converge_energy`：取值f64。构型优化中上下两步能量变化的收敛阈值。缺省值：1.0e-6
- `converge_grms`：取值f64。梯度的收敛阈值。缺省值：3.0e-4
- `converge_gmax`：取值f64。最大梯度的收敛阈值。缺省值：4.5e-4
- `converge_drms`：取值f64。构型优化中上下两步构型变化的收敛阈值。缺省值：1.2e-3
- `converge_dmax`：取值f64。最大构型变化的收敛阈值。缺省值：1.8e-3
- `coordsys`：取值String。坐标系统设置。缺省值："tric"。如果有其他需求见geomeTRIC的官方说明：https://geometric.readthedocs.io/en/latest/
- `transition`：取值bool，设置为true则开启过渡态搜索。缺省值为：false（对应于稳态搜索）
- `hessian`：取值String。决定是否以及何时进行Hessian矩阵计算（目前只支持数值Hessian计算）。
    - "never"：不做Hessian矩阵计算（缺省：稳态搜索）
	- "first"：只对初始结构计算Hessian矩阵（缺省：过渡态搜索）
	- "last"：计算优化好的结构的Hessian矩阵计算
	- "first+last"：计算初始和优化好的两个结构的Hessian矩阵
	- "stop"：不做构型优化，只计算初始结构的Hessian矩阵
	- "each"：计算构型优化中每一步的Hessian矩阵
- `frequency`：取值bool，当得到Hessian矩阵后，是否开展频率计算和热化学分析。缺省值：true
- `thermo`：取值[f64;2]，提供热力学分析的状态：[温度 (K),压强 (bar)]。缺省值：[300.0, 1.0]
- 例子一：开启GGA、meta-GGA或者杂化泛函的稳态构型优化（以x3lyp为例），则不需要使用[geometric_pyo3]区的设置
    ```
	[ctrl]
         xc = x3lyp
         job_type =                  "opt"
         opt_engine =                "geometric-pyo3"
         xc =                        "x3lyp"
         empirical_dispersion =      "d3bj"
         basis_path =                "cc-pVDZ"
         auxbas_path =               "def2-universal-jkfit"
         charge =                    0.0
         spin =                      1.0
         spin_polarization =         false
    
    [geom]
         name = "CO"
         unit = "angstrom"
         position = """
            C  0.00000000000      0.0000000000      0.0000000000
            O  1.20000000000      0.0000000000      0.0000000000
        """ 
    ```
- 例子二：如果使用双杂化泛函等没有解析力的方法，则需要在[ctrl]区开启数值力的计算功能（仅展示与上个例子不同的设置）
    ```
	[ctrl]
	    xc = xyg3
		numerical_force = true
	```
- 例子三：开启过渡态优化和频率计算，并且设置非常规状态（100华氏度、1.5个大气压）
    ```
	[geometric_pyo3]
	    transition = true
		hessian = "first+last"
		thermo = [398.0, 1.5]
	```
# Detailed descrption of [geom] block in the control file
- `name`：取值String类型。分子体系的名称
- `unit`：取值String类型。坐标单位。目前支持：angstrom和bohr
- `position`：取值String类型。分子体系的坐标，目前支持xyz格式
    - 一个例子：
`position = '''
        N  -2.1988391019      1.8973746268      0.0000000000
        H  -1.1788391019      1.8973746268      0.0000000000
        H  -2.5388353987      1.0925460144     -0.5263586446
        H  -2.5388400276      2.7556271745     -0.4338224694
'''
`
- `ghost`:　取值String类型。每一行对应一个ghost原子、点电荷或者ghost赝势的设置
   - 对于ghost原子，其格式为：`basis set  <elem>  <position>`
   - 对于点电荷，其格式为：`point charge  <charge>  <position>`
   - 对于ghost赝势，其格式为：`potential  <file_name>  <position>`
       - `<elem>`：取值i32类型。设置这一ghost原子的元素类型，以便程序找到正确的基组
       - `<charge>`:　取值f64类型。设置点电荷的电荷量
       - `<file_name>`: 取值String类型。设置赝势的文件名称
       - `<position>`:　取值[f64;3]类型，设置ghost原子，点电荷和赝势对应的xyz坐标
    - 一个例子：
`ghost = '''
        basis set             Cl           0.000000   0.00000    1.6000000
        point charge        -0.3           0.500000   0.60000    0.8000000
        point charge         1.5           0.200000   0.30000    1.1000000
        potential       O_ghost.json       0.500000   0.60000    0.8000000
        potential       Mg_ghost.json      0.200000   0.30000    1.1000000
'''
`
- `ext_field_dipole`：取值 `[f64;3]` 类型。外加偶极电场 (以 `[0., 0., 0.]` 为规范原点) 在 [x, y, z] 三个分量上的强度。单位 a.u.。
    - 一个例子：
        ```
        [geom]
        name = "NH3"
        unit = "Angstrom"
        position = """
            N  0.0  0.0  0.0
            H  0.0  1.5  1.0
            H  1.4  1.1  0.0
            H  1.2  0.0  1.3
        """
        ext_field_dipole = [0.0, 0.1, 0.0]
        ```
- `rrs_pbc`: 取值为bool。如果设置为true则启动RRS-PBC计算。缺省为false
  
- `unit_cell_index`: 取值为Vec<usize>。核心晶胞包含的原子在`position`中的序号。缺省为空
  
  - **注意**：原子的序号从0开始
    
  - 一个例子：
    
    ```
    [geom]
    name = "NH3"
    unit = "Angstrom"
    position = """
        H  0.0  0.0  0.0
        C  1.1  0.0  0.0
        C  2.4  0.0  0.0
        C  3.7  0.0  0.0  # 这个原子属于核心晶胞
        C  5.0  0.0  0.0  # 这个原子也属于核心晶胞
        C  6.3  0.0  0.0
        C  8.0  0.0  0.0
        H  9.1  0.0  0.0
    """
    rrs_pbc = true
    unit_cell_index = [3,4]
    ```
    
- `pbc_dim`: 取值为usize。周期性的维度。缺省为1
  
- `rrs_pbc_vec`：取值为Vec<f64>。周期性体系的晶格矢量。缺省为空
  
  - **注意**：每个晶格矢量均为三维，如果周期性的维度不为1，则将所有晶格矢量拼接为一个作为该参数。如果晶格矢量的总长度超过周期性维度对应的长度，将抛出一个警告并忽略多余部分
    
  - 一个例子：
    
    ```
    pbc_dim = 2  # 视为二维周期性体系
    rrs_pbc_vec = [1.0,0.0,0.0,0.0,1.0,0.0,5.14]  # 两个晶格矢量分别为[1.0,0.0,0.0]和[0.0,1.0,0.0]，多余的5.14被忽略
    ```
    
- `max_step`：取值为Vec<String>。从核心晶胞开始作用晶格矢量的最大次数。缺省为空
  
  - **注意**：RRS-PBC算法会从核心晶胞开始按照平移矢量向正负方向各平移至多`max_step`次，匹配元素和位置均符合的晶胞，并跳过不符合的。如果长度超过周期性维度，将忽略多余部分。匹配结果会在`print_level`至少为1时输出，例如：
    
    ```
    For step path = (-1, 0, 0), found target index = [0, 1, 2, 3]
    For step path = (0, 0, 0), found target index = [4，5，6，7]
    For step path = (1, 0, 0), found target index = [8，9，10，11] 
    ```
    
- `k_points`: 取值为Vec<usize>。每个周期性维度下的k点数量。缺省为空
  
  - **注意**：k点的选取方法为在倒格矢和倒格矢的反向之间均匀分布，并包含两侧边界。为了确保均匀分布的k点能够覆盖高对称点，推荐将数量设置的大一些
