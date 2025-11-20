# RI-JK On-The-Fly


## RI-JK 原理


### RI-JK 背景介绍


RI-JK (这里仅讨论与实现 RI-V) 的策略是将计算量较大的 4c-2e ERI，拆解成 3c-2e ERI 与 2c-2e ERI 的缩并计算。我们需要利用到
- 3c-2e ERI $g_{\mu \nu, P} = (\mu \nu | P)$
- 2c-2e ERI $J_{PQ} = (P | Q)$

电子积分的计算目的总是实现 Fock 矩阵计算 (在这里是与密度矩阵 $D_{\mu \nu}$ 或轨道系数 $C_{\mu i}$ 的缩并)，因此仅仅讨论拆分电子积分是不行的，还要具体地考虑张量缩并。

RI-JK 得到 Fock 矩阵的整体过程非常简单：

- J 积分：(我们这里约定，辅助基指标的 $J_{PQ}$ 是电子积分，原子轨道指标的 $J_{\mu \nu} [\mathbf{D}]$ 是 Fock 矩阵的库伦贡献)

    $$
    J_{\mu \nu}[\mathbf{D}] = \sum_{PQ} \sum_{\kappa \lambda} g_{\mu \nu, P} (\mathbf{J}^{-1})_{PQ} g_{\kappa \lambda, Q} D_{\kappa \lambda}
    $$

- K 积分：

    $$
    K_{\mu \nu}[\mathbf{D}] = \sum_{PQ} \sum_{\kappa \lambda} g_{\mu \kappa, P} (\mathbf{J}^{-1})_{PQ} g_{\nu \lambda, Q} D_{\kappa \lambda}
    $$


### 内存充足时的考量因素


上述积分的表达非常简单，但需要考量的因素也有不少。


我们作分类讨论。对于内存足够的情况，我们需要考虑到

- **电子密度并非是唯一的**。上述表达式使用了单个密度矩阵的表述 ($D_{\mu \nu}$)，但实际情况是我们可能处理多密度矩阵问题：对于梯度计算问题是性质数量 $n_\mathrm{prop}$，对于自旋极化问题是待处理的自旋数量 $n_\mathrm{spin}$ (闭壳层 $n_\mathrm{spin} = 1$，非限制开壳层 $n_\mathrm{spin} = 2$，多参考与限制性开壳层可能有其他情形)。

    对于这种情况，我们需要做到的事情是，尽量避免多次电子积分的计算；如果要作分批算法，要尽量保证电子积分的生成与内存读写次数较少。

- **复数情形**。这在大多数情况下是碰不到的；但如果基组是复轨道，那么情形会有所不同。我们这里不讨论这种情况。复轨道下，上述表达式会存在指标顺序问题与共轭问题，是不正确的。

- **对称性**。电子积分、张量缩并都可以通过利用对称性得到计算量、内存占用量上的简化。这里 3c-2e ERI $g_{\mu \nu, P}$ 与密度矩阵 $D_{\mu \nu}$ 具有 $(\mu, \nu)$ 的双重对称性。

- **2c-2e ERI 的正定性**。2c-2e ERI $J_{PQ}$ 的正定性可以用于快速的电子积分分解与组合。记 Cholesky 分解后的三角矩阵 $L_{PR}$：

    $$
    J_{PQ} = \sum_R L_{PR} L_{QR}
    $$

    我们可以得到 Cholesky 分解的 3c-2e ERI $Y_{\mu \nu, P}$：

    $$
    Y_{\mu \nu, P} = \sum_{Q} g_{\mu \nu, Q} (\mathbf{L}^{-1})_{PQ}
    $$

    同时需要留意到，这里的 $(\mathbf{L}^{-1})_{PQ}$ 未必需要通过求逆计算，而是通过三角矩阵求解实现。

    但也需要留意，$J_{PQ}$ 在实际应用情景中，可能因为各种原因而表现出半正定现象。这需要用其他方式解决。

- **轨道系数替代密度矩阵**。我们可以注意到，密度矩阵可以用占据轨道表示

    $$
    D_{\mu \nu} = \sum_{i} C_{\mu i} C_{\nu i} n_i
    $$

    密度矩阵大多数时候是不满秩且半正定的 (少数例外情况包括 MP2 等微扰方法的 1-RDM)。利用这一特性，轨道系数的使用经常可以节省计算量。

    但在一些特定情况中 (特别是梯度计算时)，会遇到密度矩阵尽管不满秩，但也并非半正定的情况。例如密度矩阵的导数矩阵，是可以通过下述方式得到的：

    $$
    \mathbf{D}_{\mu \nu}^\mathbb{A} = \sum_{i} C_{\mu i} C_{\nu i}^\mathbb{A} + \text{swap} (\mu, \nu)
    $$

    其中，轨道系数导数 $C_{\nu i}^\mathbb{A}$ 是轨道系数与 U 矩阵的乘积 $\sum_{p} C_{\nu p} U_{pi}^\mathbb{A}$。对于这类型密度矩阵，在本征值的表现是正负本征值相等且各一半；在实际计算上，使用左右系数的计算方式会更自然 (参考 Psi4) 且计算量也是最省的。


### 具体算法与函数接口


- 后文中，我们记 $n_{\mathrm{tp(basis)}} = \frac{1}{2} n_\mathrm{basis} (\mathrm{basis} + 1)$，即三角存储的原子轨道矩阵大小。 
- 下划线的角标，意味着其在计算时需要被分批。注意一些算法还会对其他指标 (特别是 $\mathbb{A}$) 进行明确的循环。


**函数 `generate_vj_ri_incore_with_rstsr`**


$$
\begin{align*}
D_{\mathrm{tp} (\mu \nu)}^\mathbb{A} &\mathop{\tilde{\bowtie}} D_\mathrm{\mu \nu}^\mathbb{A}
\tag{eq.1} \\
\mathscr{T}_{P}^\mathbb{A} &= \sum_{\mathrm{tp} (\mu \nu)} D_{\mathrm{tp} (\mu \nu)}^\mathbb{A} Y_{\mathrm{tp} (\mu \nu), P}
\tag{eq.2} \\
J_{\mathrm{tp} (\mu \nu)}^\mathbb{A} &= \sum_{P} \mathscr{T}_{P}^\mathbb{A} Y_{\mathrm{tp} (\mu \nu), P}
\tag{eq.3} \\
J_{\mu \nu}^\mathbb{A} &\bowtie J_{\mathrm{tp} (\mu \nu)}^\mathbb{A}
\tag{eq.4}
\end{align*}
$$


| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `cderi` | $Y_{\mathrm{tp} (\mu \nu), P}$ | $(\mathrm{tp} (\mu \nu), P)$ | $(n_\mathrm{tp(basis)}, n_\mathrm{aux})$ | $\bowtie$ |
| `dms` | $D_\mathrm{\mu \nu}^\mathbb{A}$ | $(\mu, \nu, \mathbb{A})$ | $(n_\mathrm{basis}, n_\mathrm{basis}, n_\mathrm{set})$ | sym |


**函数 `generate_vk_ri_incore_coeff_with_rstsr`**


$$
\begin{align*}
\tilde{C}_{\mu i}^\mathbb{A} &= C_{\mu i} \sqrt{n_i}
\tag{eq.1} \\
Y_{\mu \nu, \underline{P}} &\bowtie Y_{\mathrm{tp}(\mu \nu), \underline{P}}
\tag{eq.2} \\
Y_{\mu i, \underline{P}}^\mathbb{A} &= \sum_\nu Y_{\mu \nu, \underline{P}} \tilde{C}_{\nu i}
\tag{eq.3} \\
K_{\mu \nu}^\mathbb{A} &= \sum_{i \underline{P}} Y_{\mu i, \underline{P}}^\mathbb{A} Y_{\nu i, \underline{P}}^\mathbb{A}
\tag{eq.4}
\end{align*}
$$


| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `cderi` | $Y_{\mathrm{tp} (\mu \nu), P}$ | $(\mathrm{tp} (\mu \nu), P)$ | $(n_\mathrm{tp(basis)}, n_\mathrm{aux})$ | $\bowtie$ |
| `mo_coeff` | $C_{\mu p}^\mathbb{A}$ | $(\mu, p, \mathbb{A})$ | $(n_\mathrm{basis}, n_\mathrm{orb}, n_\mathrm{aux})$ |
| `mo_occ` | $n_{p}^\mathbb{A}$ | $(p, \mathbb{A})$ | $(n_\mathrm{orb}, n_\mathrm{aux})$ |
| `batch_size` | | | | for index $P$ |


| 内存需求类型 | 公式序号 | 表达式 | 指标顺序 | 内存需求量 | 其他说明 |
|--|--|--|--|--|--|
| batched | (eq.3) | $Y_{\mu i, \underline{P}}^\mathbb{A}$ | $(\mu, i, \underline{P})$ | $(n_\mathrm{basis}, n_\mathrm{occ}, n_\mathrm{batch})$ |
| fixed | (eq.1) | $\tilde{C}_{\mu i}^\mathbb{A}$ | $(\mu, i, \mathbb{A})$ | $(n_\mathrm{basis}, n_\mathrm{occ}, n_\mathrm{set})$ |
| fixed | (eq.4) | $K_{\mu \nu}^\mathbb{A}$ | $(\mu, \nu, \mathbb{A})$ | $(n_\mathrm{basis}, n_\mathrm{basis}, n_\mathrm{set})$ |
| thread | (eq.2) | $Y_{\mu \nu, \underline{P}}$ | $(\mu, \nu)$ | $(n_\mathrm{basis}, n_\mathrm{basis})$ |


**函数 `generate_vk_ri_incore_dm_with_rstsr`**


$$
\begin{align*}
Y_{\mu \nu, \underline{P}} &\bowtie Y_{\mathrm{tp}(\mu \nu), \underline{P}}
\tag{eq.1} \\
\mathscr{T}_{\mu \nu, \underline{P}}^\mathbb{A} &= \sum_\lambda Y_{\mu \lambda, \underline{P}} D_{\nu \lambda}^\mathbb{A}
\tag{eq.2} \\
K_{\mu \nu}^\mathbb{A} &= \sum_{\kappa \underline{P}} \mathscr{T}_{\mu \kappa, \underline{P}}^\mathbb{A} Y_{\mu \kappa, \underline{P}}
\tag{eq.3}
\end{align*}
$$


| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `cderi` | $Y_{\mathrm{tp} (\mu \nu), P}$ | $(\mathrm{tp} (\mu \nu), P)$ | $(n_\mathrm{tp(basis)}, n_\mathrm{aux})$ | $\bowtie$ |
| `dms` | $D_\mathrm{\mu \nu}^\mathbb{A}$ | $(\mu, \nu, \mathbb{A})$ | $(n_\mathrm{basis}, n_\mathrm{basis}, n_\mathrm{set})$ | sym |
| `batch_size` | | | | for index $P$ |


| 内存需求类型 | 公式序号 | 表达式 | 指标顺序 | 内存需求量 | 其他说明 |
|--|--|--|--|--|--|
| batched | (eq.1) | $Y_{mu \nu, \underline{P}}$ | $(\mu, \nu, \underline{P})$ | $(n_\mathrm{basis}, n_\mathrm{basis}, n_\mathrm{batch})$ |
| batched | (eq.2) | $\mathscr{T}_{\mu \nu, \underline{P}}^\mathbb{A}$ | $(\mu, \nu, \underline{P})$ | $(n_\mathrm{basis}, n_\mathrm{basis}, n_\mathrm{batch})$ |
| fixed | (eq.3) | $K_{\mu \nu}^\mathbb{A}$ | $(\mu, \nu, \mathbb{A})$ | $(n_\mathrm{basis}, n_\mathrm{basis}, n_\mathrm{set})$ |
| thread | (eq.1) | $Y_{mu \nu, \underline{P}}$ | $(\mu, \nu)$ | $(n_\mathrm{basis}, n_\mathrm{basis})$ | 独立于其他并行区间 |
| thread | (eq.2) | $\mathscr{T}_{\mu \nu, \underline{P}}^\mathbb{A}$ | $(\mu, \nu)$ | $(n_\mathrm{basis}, n_\mathrm{basis})$ | 独立于其他并行区间 |


目前在 REST 中，已有 `SCF.generate_vj_with_ri_v_sync` 与 `SCF.generate_vk_with_ri_v` 执行相同功能，而且性能上应该不会有明显差别 (内存用量上有差别)，因此我们就不将这两个程序链入自洽场计算中了。


## 分批的 RI-JK 算法


这份文档主要目的是实现内存不充足时的 RI-JK 算法。上面我们简要回顾了内存充足时需要讨论的情形，是因为一些技巧在内存不充足时仍然可以使用。

为有效利用 RI-JK 算法，我们一般需要存储完整的 Cholesky 分解 $Y_{\mu \nu, P}$ 张量。这是非常大的三维张量，即使可以利用对称性也只能降低一半存储。对于 100 原子 TZ 基组，这种存储量就会开始变得不可接受。

但既然如此，我们就必须要对电子积分进行分批、且 Cholesky 分解也会变得困难。这里将遇到的核心困难是，是否存在完美的分批方式，以及如何具体地实现分批算法。


### 分批 RI-J 算法


**函数 `generate_vj_ri_direct_with_rstsr`**


$$
\begin{align*}D_{\mathrm{tp} (\mu \nu)}^\mathbb{A} &\mathop{\tilde{\bowtie}} D_\mathrm{\mu \nu}^\mathbb{A}
\tag{eq.1} \\
g_{\mathrm{tp}(\mu \nu), \underline{P}} &\quad \text{generate}
\tag{eq.2} \\
\mathscr{T}_{\underline{P}}^{\mathbb{A}, 1} &= \sum_{\mathrm{tp}(\mu \nu)} g_{\mathrm{tp}(\mu \nu), \underline{P}} D_{\mathrm{tp}(\mu \nu)}^\mathbb{A}
\tag{eq.3} \\
\mathscr{T}_P^{\mathbb{A}, 2} &= \sum_{Q} (\mathbf{J}^{-1})_{PQ} \mathscr{T}_P^{\mathbb{A}, 1}
\tag{eq.4} \\
g_{\mathrm{tp}(\mu \nu), \underline{P}} &\quad \text{generate}
\tag{eq.5} \\
J_{\mathrm{tp}(\mu \nu)}^\mathbb{A} &= \sum_{\underline{P}} g_{\mathrm{tp}(\mu \nu), \underline{P}} \mathscr{T}_P^{\mathbb{A}, 2}
\tag{eq.6} \\
J_{\mu \nu}^\mathbb{A} &\bowtie J_{\mathrm{tp}(\mu \nu)}^\mathbb{A}
\tag{eq.7}
\end{align*}
$$


| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `dms` | $D_\mathrm{\mu \nu}^\mathbb{A}$ | $(\mu, \nu, \mathbb{A})$ | $(n_\mathrm{basis}, n_\mathrm{basis}, n_\mathrm{set})$ | sym |
| `batch_size` | | | | for index $P$ |


| 内存需求类型 | 公式序号 | 表达式 | 指标顺序 | 内存需求量 | 其他说明 |
|--|--|--|--|--|--|
| batched | (eq.2), (eq.5) | $g_{\mathrm{tp}(\mu \nu), \underline{P}}$ | $(\mathrm{tp} (\mu \nu), \underline{P})$ | $(n_\mathrm{tp(basis)}, n_\mathrm{batch})$ | (eq.2) 与 (eq.5) 独立 |
| fixed | (eq.1) | $D_{\mathrm{tp} (\mu \nu)}^\mathbb{A}$ | $(\mathrm{tp} (\mu \nu), \mathbb{A})$ | $(n_\mathrm{tp(basis)}, n_\mathrm{set})$ |
| fixed | (eq.3) | $\mathscr{T}_{P}^{\mathbb{A}, 1}$ | $(P, \mathbb{A})$ | $(n_\mathrm{aux}, n_\mathrm{set})$ |
| fixed | (eq.4) | $\mathscr{T}_{P}^{\mathbb{A}, 2}$ | $(P, \mathbb{A})$ | $(n_\mathrm{aux}, n_\mathrm{set})$ | 复用 (eq.3) $\mathscr{T}_{P}^{\mathbb{A}, 1}$ |
| fixed | (eq.4) | $(\mathbf{J}^{-1})_{PQ}$ | $(P, Q)$ | $(n_\mathrm{basis}, n_\mathrm{basis})$ | LAPACK DSYSV |
| fixed | (eq.6) | $J_{\mathrm{tp}(\mu \nu)}^\mathbb{A}$ | $(\mathrm{tp} (\mu \nu), \mathbb{A})$ | $(n_\mathrm{tp(basis)}, n_\mathrm{set})$ | 复用 (eq.1) $D_{\mathrm{tp} (\mu \nu)}^\mathbb{A}$ |
| fixed | (eq.7) | $J_{\mu \nu}^\mathbb{A}$ | $(\mu, \nu, \mathbb{A})$ | $(n_\mathrm{basis}, n_\mathrm{basis}, n_\mathrm{set})$ |


整个分批 RI-J 算法，比起内存充足时的 RI-J 算法，代价是

- 每次 SCF 迭代都需要重新计算一次电子积分；这是无法避免的代价。
- (eq.2) 与 (eq.5) 需要计算两次电子积分，即电子积分的计算代价是两倍。关于这一点，我总怀疑应该有其他办法，但暂时没有想出来。 
- (eq.4) 的计算涉及到求逆；在我们实际程序中，使用了基于 LU 分解的 `rt::linalg::solve` 函数。但需要指出，这里实际上也可以用 Cholesky 分解进行计算。但考虑到这一步的计算消耗一般不是大的 (相比于电子积分的巨大 $O(N^3)$)，我认为这一步将效率优化到底的必要性不大。

### 分批 RI-K 算法：可完整储存 $g_{\mu i, P}$ 的情景 (semi-direct)


首先，RI-K 计算一般都假定使用轨道系数，通常会更高效，实现难度也相对较小。

不同于 RI-J，RI-K 的分批算法非常容易导致计算数量级的上升：或者是张量缩并部分升到低 $O(N^5)$、或者是电子积分升到低 $O(N^4)$。这通常是比较难以接受的。

因此，我们仍然需要考虑，是否可以存储 $g_{\mu i, P}$ 的中间张量。这对避免进一步提升计算复杂度而言，是至关重要的。


**函数 `generate_vk_ri_semi_direct_coeff_with_rstsr`**


$$
\begin{align*}
\tilde{C}_{\mu i}^\mathbb{A} &= C_{\mu i} \sqrt{n_i}
\tag{eq.1} \\
L_{PQ} &= \mathrm{Cholesky} (J_{PQ})
\tag{eq.2} \\
g_{\mathrm{tp}(\mu \nu), \underline{P}} &\quad \text{generate}
\tag{eq.3} \\
g_{\mu \nu, \underline{P}} &\bowtie g_{\mathrm{tp}(\mu \nu), \underline{P}}
\tag{eq.4} \\
g_{\mu i, \underline{P}}^\mathbb{A} &= \sum_\nu g_{\mu \nu, \underline{P}} \tilde{C}_{\nu i}^\mathbb{A}
\tag{eq.5} \\
Y_{\mu i, P}^\mathbb{A} &= \sum_Q (\mathbf{L}^{-1})_{QP} g_{\mu i, Q}^\mathbb{A}
\tag{eq.6} \\
K_{\mu \nu}^\mathbb{A} &= \sum_{i P} Y_{\mu i, P}^\mathbb{A} Y_{\nu i, P}^\mathbb{A}
\tag{eq.7}
\end{align*}
$$


| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `mo_coeff` | $C_{\mu p}^\mathbb{A}$ | $(\mu, p, \mathbb{A})$ | $(n_\mathrm{basis}, n_\mathrm{orb}, n_\mathrm{aux})$ |
| `mo_occ` | $n_{p}^\mathbb{A}$ | $(p, \mathbb{A})$ | $(n_\mathrm{orb}, n_\mathrm{aux})$ |
| `batch_size` | | | | for index $P$ |


| 内存需求类型 | 公式序号 | 表达式 | 指标顺序 | 内存需求量 | 其他说明 |
|--|--|--|--|--|--|
| batched | (eq.3) | $g_{\mathrm{tp}(\mu \nu), \underline{P}}$ | $\mathrm{tp}(\mu \nu), \underline{P}$ | $(n_\mathrm{tp(basis)}, n_\mathrm{batch})$ |
| fixed | (eq.1) | $\tilde{C}_{\mu i}^\mathbb{A}$ | $(\mu, i, \mathbb{A})$ | $(n_\mathrm{basis}, n_\mathrm{occ}, n_\mathrm{set})$ |
| fixed | (eq.2) | $L_{PQ}$ | $(P, Q)$ | $(n_\mathrm{aux}, n_\mathrm{aux})$ |
| fixed | (eq.5) | $g_{\mu i, P}^\mathbb{A}$ | $(\mu, i, P)$ | $(n_\mathrm{basis}, n_\mathrm{occ}, n_\mathrm{aux})$ | 主要内存消耗 |
| fixed | (eq.7) | $K_{\mu \nu}^\mathbb{A}$ | $(\mu, \nu, \mathbb{A})$ | $(n_\mathrm{basis}, n_\mathrm{basis}, n_\mathrm{set})$ |
| thread | (eq.4) | $g_{\mu \nu, \underline{P}}$ | $(\mu, \nu)$ | $(n_\mathrm{basis}, n_\mathrm{basis})$ |


这里计算，代价是
- 每次 SCF 迭代计算时，需要重新计算一遍 3c-2e ERI $g_{\mu \nu, P}$；这是不可避免的。
- 对每个性质角标 $\mathbb{A}$ 作循环时，需要重新计算一遍 3c-2e ERI；这或许有其他办法，但我暂时想不出。意味着 RHF 并没有额外代价，但对于 UHF 或梯度性质计算时，就会有比较严重的额外电子积分计算代价。

需要留意，上述计算中，每次都对 $\mathbb{A}$ 作循环，而不是直接存储 $g_{\mu i, P}^\mathbb{A}$ 的四维张量；存储一个四维张量还是太恐怖了。

同时留意 (eq.6) 没有额外的内存消耗。这里使用的是 BLAS 的 `DTRSV` 函数 (三角矩阵求解)，因此不需要额外内存。


### 分批 RI-K 算法：不能完整储存 $g_{\mu i, P}$ 的情景 (direct)


在无法存储所有 $g_{\mu i, P}$ 的情况下，某种程度上，最好的解决方法并非使用 RI-K，而是使用 4c-2e ERI 作 integral screening。对于 4c-2e ERI，其分批算法是比较直观的。

但对于 3c-2e ERI，中间又涉及到一步 2c-2e ERI 的求逆，所有指标都至少耦合了一次、难以拆散。因此，我们必须要接受因分批算法而导致的计算复杂度上升的代价。由于电子积分是 $O(N^3)$ (尽管是代价很大的三次方算法)，我们能接受其代价上升到 $O(N^4)$。

同时，这种情形下，使用分子轨道代替密度矩阵的劣势会出现：当分批数量较大时，电子积分与轨道系数的缩并计算代价将会是接近 $O(N^5)$ 的。因此，我们这里将要采用密度矩阵的 RI-K 分批算法。

为了程序设计上的方便，这里我们将不会利用 $g_{\mu \nu, P}$ 关于 $(\mu, \nu)$ 的二重对称性，且分批将会针对 $\mu$ 指标进行。如果分批较小，计算代价会稍微多一些但不会超过两倍；分批较多时，该算法应在 RI 前提下，很难再更快了。


**函数 `generate_vk_ri_direct_dm_with_rstsr`**


$$
\begin{align*}
L_{PQ} &= \mathrm{Cholesky} (J_{PQ})
\tag{eq.1} \\
g_{\underline{\mu} \nu, P} &\quad \text{generate}
\tag{eq.2} \\
\mathscr{G}_{\underline{\mu} \nu, P}^\mathbb{A} &= \sum_\lambda g_{\underline{\mu} \nu, P} D_{\nu \lambda}^\mathbb{A}
\tag{eq.3} \\
\mathscr{T}_{\underline{\mu} \nu, P}^\mathbb{A} &= \sum_Q (\mathbf{L}^{-1})_{QP} \mathscr{G}_{\underline{\mu} \nu, Q}^\mathbb{A}
\tag{eq.4} \\
\mathscr{Y}_{\underline{\mu} \nu, P}^\mathbb{A} &= \sum_Q (\mathbf{L}^{-1})_{PQ} \mathscr{T}_{\underline{\mu} \nu, Q}^\mathbb{A}
\tag{eq.5} \\
g_{\underline{\nu} \kappa, P} &\quad \text{generate}
\tag{eq.6} \\
K_{\underline{\mu \nu}}^\mathbb{A} = K_{\underline{\nu \mu}}^\mathbb{A} &= \sum_{\kappa P} \mathscr{Y}_{\underline{\mu} \kappa, P}^\mathbb{A} g_{\underline{\nu} \kappa, P}
\tag{eq.7}
\end{align*}
$$


| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `dms` | $D_\mathrm{\mu \nu}^\mathbb{A}$ | $(\mu, \nu, \mathbb{A})$ | $(n_\mathrm{basis}, n_\mathrm{basis}, n_\mathrm{set})$ | sym |
| `batch_size` | | | | for index $\mu$ of $g_{\mu \nu, P}$ |


| 内存需求类型 | 公式序号 | 表达式 | 指标顺序 | 内存需求量 | 其他说明 |
|--|--|--|--|--|--|
| batched | (eq.2) | $g_{\underline{\mu} \nu, P}$ | $(\underline{\mu}, \nu, P)$ | $(n_\mathrm{batch}, n_\mathrm{basis}, n_\mathrm{aux})$ |
| batched | (eq.5) | $\mathscr{Y}_{\underline{\mu} \nu, P}^\mathbb{A}$ | $(\underline{\mu}, \nu, P, \mathbb{A})$ | $(n_\mathrm{batch}, n_\mathrm{basis}, n_\mathrm{aux}, n_\mathrm{set})$ | (eq.3)-(eq.5) 使用同一内存 |
| batched | (eq.6) | $g_{\underline{\nu} \kappa, P}$ | $(\underline{\nu}, \kappa, P)$ | $(n_\mathrm{batch}, n_\mathrm{basis}, n_\mathrm{aux})$ | 覆盖 (eq.2) |
| fixed | (eq.1) | $L_{PQ}$ | $(P, Q)$ | $(n_\mathrm{aux}, n_\mathrm{aux})$ |
| fixed | (eq.7) | $K_{\underline{\mu \nu}}^\mathbb{A}$ | $(\underline{\mu}, \nu, \mathbb{A})$ | $(n_\mathrm{batch}, n_\mathrm{basis}, n_\mathrm{set})$ |
