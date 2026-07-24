# Notes on LibXC/XCFun convention transformation

## Explanation by specific example

LibXC and XCFun have different conventions on how to handle the spin-polarized cases.

We will use GGA spin-polarized second-order derivative (fxc) as example to explain that.

As the first step, we state that for the energy part (zk) and first-order derivative part (vxc), LibXC and XCFun have the same conventions. The order is

| Index | Notation |
|--|--|
| 0 | zk |
| 1 | r_u |
| 2 | r_d |
| 3 | s_uu |
| 4 | s_ud |
| 5 | s_dd |

For the notation in above table,
- Rho is spin-polarization two-component, `r_u` means $\rho^\uparrow$, `r_d` means $\rho^\downarrow$.
- Sigma is spin-polarization three-component, `s_uu` means $\sigma^{\uparrow \uparrow}$, `s_ud` means $\sigma^{\uparrow \downarrow}$, `s_dd` means $\sigma^{\downarrow \downarrow}$.

For the fxc part, there will involve two densities as variables. The LibXC and XCFun have fundamental difference at

- LibXC respect which type of density first (sorted by rho, sigma), then its spin;
- XCFun respect the spin-component first (sorted by r_u, r_d, s_uu, s_ud, s_dd).

It is more favorable to use XCFun-style for future DFT evaluation. However, LibXC supports more functionals, with better API design and popularity. So an index transform mapping is required. For this specific task,

| Index | Notation<br>LibXC | Notation<br>XCFun | Map |
|-:|--|--|-:|
|  6 | r_u  / r_u  | r_u  / r_u  |  6 |
|  7 | r_u  / r_d  | r_u  / r_d  |  7 |
|  8 | r_d  / r_d  | r_u  / s_uu |  9 |
|  9 | r_u  / s_uu | r_u  / s_ud | 10 |
| 10 | r_u  / s_ud | r_u  / s_dd | 11 |
| 11 | r_u  / s_dd | r_d  / r_d  |  8 |
| 12 | r_d  / s_uu | r_d  / s_uu | 12 |
| 13 | r_d  / s_ud | r_d  / s_ud | 13 |
| 14 | r_d  / s_dd | r_d  / s_dd | 14 |
| 15 | s_uu / s_uu | s_uu / s_uu | 15 |
| 16 | s_uu / s_ud | s_uu / s_ud | 16 |
| 17 | s_uu / s_dd | s_uu / s_dd | 17 |
| 18 | s_ud / s_ud | s_ud / s_ud | 18 |
| 19 | s_ud / s_dd | s_ud / s_dd | 19 |
| 20 | s_dd / s_dd | s_dd / s_dd | 20 |

We can see that

- LibXC will first category `r/r` (6--8), then `r/s` (9--14), then `s/s` (16--20). For each category, sort by spin.
- XCFun will first category `r_u` (6--10), then `r_d` (11--14), then `s_uu` (15--17), then `s_ud` (18--19), then `s_dd` (20). For each category, sort by the same way in first category.

## Extension to the example

- **Higher derivative**: We may encounter more higher derivatives (usually up to 4th derivative, but can be more).
- **More kinds of density**: We may use more density types. The priority is RHO > SIGMA > TAU > LAPL.
  - TAU: `t_u`, `t_d`
  - LAPL: `l_u`, `l_d`
- We assume the RHO (LDA) only inputs RHO; SIGMA (GGA) inputs both RHO and SIGMA; TAU (some meta-GGA) inputs RHO SIGMA TAU, and LAPL (some meta-GGA) inputs all RHO SIGMA TAU LAPL (though some LAPL meta-GGAs does not actually input tau, but we still require the TAU to be available for simplicity).
