use crate::scf_io;
use crate::scf_io::SCF;
use crate::Molecule;
use crate::hessian::ej_ek_baseline::{EjEkBaseline, BASELINE_TERM_KEYS};
use crate::hessian::memory_monitor::{self, MemMonitor};
use rest_libcint::prelude::*;
use rstsr::prelude::*;
use std::collections::HashMap;
use tensors::matrix_blas_lapack::_power_rayon_for_symmetric_matrix;
use tensors::MatrixFull;

use crate::utilities::rstsr_util::*;

/// Routing for each optimizable G-term in calc_ej_ek().
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum TermPath {
    /// Original for-loop implementation (verified baseline).
    Inline,
    /// Optimized rstsr BLAS implementation.
    Blas,
}

impl Default for TermPath {
    /// Default to BLAS — this is now the trusted, optimized path.
    /// Use `Inline` only for verification against the BLAS baseline.
    fn default() -> Self { TermPath::Blas }
}

/// Per-G-term optimization flags for calc_ej_ek() Phase 4.
#[derive(Clone, Debug)]
pub struct EjEkOptFlags {
    pub g4_ek_vk1: TermPath,
    pub g5_ek_ri1: TermPath,
    pub g6_ek_ri2d: TermPath,
    pub g7_ek_ri2o: TermPath,
    pub g8_ej_ri1: TermPath,
    pub g9_ej_ri2d: TermPath,
    pub g10_ej_ri2o: TermPath,
    /// Max allowed diff between Blas output and stored baseline.
    pub verify_tol: f64,
}

impl Default for EjEkOptFlags {
    fn default() -> Self {
        EjEkOptFlags {
            g4_ek_vk1: TermPath::Blas,
            g5_ek_ri1: TermPath::Blas,
            g6_ek_ri2d: TermPath::Blas,
            g7_ek_ri2o: TermPath::Blas,
            g8_ej_ri1: TermPath::Blas,
            g9_ej_ri2d: TermPath::Blas,
            g10_ej_ri2o: TermPath::Blas,
            verify_tol: 1e-9,
        }
    }
}

#[non_exhaustive]
#[derive(derive_builder::Builder)]
pub struct RIRHFHessianFlags {
    #[builder(default = 0)] pub print_level: usize,
    #[builder(default = "None")] pub max_memory: Option<f64>,
    #[builder(default = true)] pub auxbasis_response: bool,
    /// Whether to compute exchange (K) terms at all. Default true (RHF).
    /// Pure DFT sets this to false to skip all K work.
    #[builder(default = true)] pub with_k: bool,
    /// Coulomb scaling factor. Default 1.0 (RHF and RKS both use 1.0).
    #[builder(default = "Some(1.0)")] pub factor_j: Option<f64>,
    /// Exchange scaling factor applied to vk1/ek in h_partial and h1ao.
    /// For RHF: 1.0. For RKS hybrid: hyb (e.g. 0.2 for B3LYP). For pure DFT: 0.0.
    #[builder(default = "Some(1.0)")] pub factor_k: Option<f64>,
    #[builder(default = false)] pub with_cphf: bool,
    #[builder(default)] pub ej_ek_opt: EjEkOptFlags,
}

/// Read-only bundle of all Phase 1-3 intermediates needed by `_blas` functions.
///
/// Constructed inside the `Blas` match arm of each G-term (via the `build_ej_ek_ctx!`
/// macro) to avoid borrow conflicts with the `Inline` arm, which uses the same local
/// variables directly. Phase 1-3 intermediates are read-only after Phase 3 ends.
///
/// Also carries libcint handles (`int3c_ctx`) for per-atom integral evaluation.
pub struct EjEkContext<'a> {
    pub nao: usize,
    pub naux: usize,
    pub nocc: usize,
    pub natm: usize,
    pub nao3: usize,
    pub blk: &'a [(usize, usize, usize)],
    pub aux_blk: &'a [(usize, usize, usize)],
    pub dm0: &'a [f64],
    pub mc2: &'a [f64],
    pub i2inv: &'a [f64],
    pub i21: &'a [f64],
    pub i211: &'a [f64],
    pub i2ip: &'a [f64],
    pub r0: &'a [f64],
    pub rk: &'a [f64],
    pub rj1: &'a [f64],
    pub wj1: &'a [f64],
    pub vjd: &'a [f64],
    pub vkd: &'a [f64],
    pub rhok_IkP_per_atom: &'a [Vec<f64>],
    pub vk2buf: &'a [f64],
    pub wj2: &'a [f64],
    pub wki: &'a [f64],
    pub wk2: &'a [f64],
    pub rkoo: &'a [f64],
    pub r2c0: &'a [f64],
    pub wj001: &'a [f64],
    pub tmpf: &'a [f64],
    pub device: DeviceBLAS,
    // Libcint handles for per-atom 3c integral recomputation.
    // Avoids holding full ip1/ipv/ipip2/ip12 tensors in memory.
    pub cint_all: &'a CINTR2CDATA,
    pub nreg: usize,
    pub aux_nbas: usize,
    pub aoslices: &'a Vec<[usize; 4]>,
    pub auxslices: &'a Vec<[usize; 4]>,
}

/// Compute a per-atom block of a 3c-2e derivative integral from libcint,
/// restricting the first AO index to one atom's shells. This avoids the
/// memory cost of evaluating and storing the full [deriv, nao, nao, naux]
/// tensor — each G-term computes only the atom blocks it needs, on demand.
fn int3c_atom_block(cint_all: &CINTR2CDATA, name: &str,
                    shl0: usize, shl1: usize,
                    nreg: usize, aux_nbas: usize) -> Vec<f64> {
    let slc: &[[usize; 2]] = &[[shl0, shl1], [0, nreg], [nreg, nreg + aux_nbas]];
    let (v, _): (Vec<f64>, Vec<usize>) =
        cint_all.integrate_row_major(name, "s1", Some(slc)).into();
    v
}

// ── BLAS-optimized G-term stubs (replaced per-task in Tasks 5-11) ──
// Each takes a read-only EjEkContext, writes to `out`, and verifies against
// the loaded baseline. Currently all `todo!()` — will be implemented per-task.

fn g4_ek_vk1_blas(ctx: &EjEkContext, ip1: &[f64], ipv: &[f64], out: &mut [f64]) {
    let nao = ctx.nao;
    let naux = ctx.naux;
    let nocc = ctx.nocc;
    let natm = ctx.natm;
    let nao3 = ctx.nao3;
    let blk = ctx.blk;
    let dm0 = ctx.dm0;
    let mc2 = ctx.mc2;
    let rk = ctx.rk;
    let rhok_IkP_per_atom = ctx.rhok_IkP_per_atom;
    let vk2buf = ctx.vk2buf;
    let device = ctx.device.clone();

    let i_t = |i0: usize, j0: usize, x: usize, y: usize| -> usize {
        i0 * natm * 9 + j0 * 9 + x * 3 + y
    };

    for i0 in 0..natm {
        let (_, p0, ni) = blk[i0];

        // ── Compute tmp[p, k, jj] = Σ_i_occ rk[p, k, i_occ] · mc2[(p0+jj), i_occ] ──
        // (unchanged: single GEMM per i0 atom)
        let rk_2d = rt::asarray((rk, [naux * nao, nocc].f(), &device));
        let mut mc2_block_stage = vec![0.0; nocc * ni];
        for i_occ in 0..nocc { for jj in 0..ni {
            mc2_block_stage[i_occ + jj * nocc] = mc2[(p0 + jj) * nocc + i_occ];
        }}
        let mc2_block_t = rt::asarray((&mc2_block_stage, [nocc, ni].f(), &device));
        let tmp_result = &rk_2d % &mc2_block_t; // [naux*nao, ni].f()
        let mut tmp = vec![0.0; naux * nao * ni];
        for p in 0..naux { for k in 0..nao { for jj in 0..ni {
            tmp[p * nao * ni + k * ni + jj] = tmp_result[[p + k * naux, jj]];
        }}}

        for j0 in 0..=i0 {
            let (_, q0, qj) = blk[j0];
            if qj == 0 { continue; }
            let rk_atom = &rhok_IkP_per_atom[j0];

            // Pre-stage dm0_block2 (for Part 2): [qj, nao].f()
            let mut dm0_block2_stage = vec![0.0; qj * nao];
            for j_ao in 0..qj { for k_ao in 0..nao {
                dm0_block2_stage[j_ao + k_ao * qj] = dm0[(q0 + j_ao) * nao + k_ao];
            }}

            // Accumulators for all 9 (x, y) pairs
            let mut part1_acc = vec![0.0; 9];
            let mut part2_acc = vec![0.0; 9];

            for i_bra in 0..ni {
                // ── Part 1 (batched over x, y): 1 GEMM + 9 for-loop dots ──
                // M1_all F-order [3*nao, naux]: element (x*nao+j_all, p) = ip1[x, p0+i_bra, j_all, p]
                //   fill: idx = (x*nao+j_all) + p*(3*nao)   [F-order: i + j*M, M = 3*nao]
                //   source: ip1[x*nao3*naux + (p0+i_bra)*nao*naux + j_all*naux + p]
                // R_all F-order [3*qj, naux]: element (y*qj+k_ao, p) = rhok_atom[(p0+i_bra)*qj*naux*3 + k_ao*naux*3 + p*3 + y]
                //   fill: idx = (y*qj+k_ao) + p*(3*qj)
                // GEMM: V1_all = M1_all @ R_all^T → [3*nao, 3*qj] F-order
                //   V1_all[x*nao+j, y*qj+k] = Σ_p ip1[x, p0+i_bra, j, p] · rhok[j0, i_bra, k, p, y]
                // Part 1[x, y] += Σ_{j,k} V1_all[x*nao+j, y*qj+k] · dm0[(q0+k)*nao + j]  (for-loop dot)
                {
                    let mut m1_stage = vec![0.0; 3 * nao * naux];
                    for x in 0..3 { for j_all in 0..nao { for p in 0..naux {
                        m1_stage[(x * nao + j_all) + p * (3 * nao)] =
                            ip1[x * nao3 * naux + (p0 + i_bra) * nao * naux + j_all * naux + p];
                    }}}
                    let m1_t = rt::asarray((&m1_stage, [3 * nao, naux].f(), &device));
                    let mut r_stage = vec![0.0; 3 * qj * naux];
                    for y in 0..3 { for k_ao in 0..qj { for p in 0..naux {
                        r_stage[(y * qj + k_ao) + p * (3 * qj)] =
                            rk_atom[(p0 + i_bra) * qj * naux * 3 + k_ao * naux * 3 + p * 3 + y];
                    }}}
                    let r_t = rt::asarray((&r_stage, [3 * qj, naux].f(), &device));
                    let v1_all = (&m1_t % &r_t.t()); // [3*nao, 3*qj]
                    let v1_raw = v1_all.into_shape(-1).into_raw();
                    // Part 1[x, y] = Σ_{j,k} V1_all[x*nao+j, y*qj+k] · dm0[(q0+k)*nao + j]
                    // V1_all F-order [3*nao, 3*qj]: element (x*nao+j, y*qj+k) at (x*nao+j) + (y*qj+k)*(3*nao)
                    for x in 0..3 { for y in 0..3 {
                        let mut s = 0.0;
                        for j in 0..nao { for k in 0..qj {
                            s += v1_raw[(x * nao + j) + (y * qj + k) * (3 * nao)]
                                * dm0[(q0 + k) * nao + j];
                        }}
                        part1_acc[x * 3 + y] += s;
                    }}
                }

                // ── Part 2 (batched over c=9): 1 GEMM + 9 for-loop dots ──
                // ipv_all F-order [9*qj, naux]: element (c*qj+j_ao, p) = ipv[c, p0+i_bra, q0+j_ao, p]
                //   fill: idx = (c*qj+j_ao) + p*(9*qj)
                //   source: ipv[c*nao3*naux + (p0+i_bra)*nao*naux + (q0+j_ao)*naux + p]
                // tmp_slice F-order [nao, naux]: element (k_ao, p) = tmp[p*nao*ni + k_ao*ni + i_bra]
                //   fill: idx = k_ao + p*nao
                // GEMM: V2_all = ipv_all @ tmp_slice^T → [9*qj, nao] F-order
                //   V2_all[c*qj+j, k] = Σ_p ipv[c, p0+i_bra, q0+j, p] · tmp[p, k, i_bra]
                // Part 2[c] += Σ_{j,k} V2_all[c*qj+j, k] · dm0[(q0+j)*nao + k]  (for-loop dot)
                {
                    let mut ipv_stage = vec![0.0; 9 * qj * naux];
                    for c in 0..9 { for j_ao in 0..qj { for p in 0..naux {
                        ipv_stage[(c * qj + j_ao) + p * (9 * qj)] =
                            ipv[c * nao3 * naux + (p0 + i_bra) * nao * naux + (q0 + j_ao) * naux + p];
                    }}}
                    let ipv_t = rt::asarray((&ipv_stage, [9 * qj, naux].f(), &device));
                    let mut tmp_slice_stage = vec![0.0; nao * naux];
                    for k_ao in 0..nao { for p in 0..naux {
                        tmp_slice_stage[k_ao + p * nao] = tmp[p * nao * ni + k_ao * ni + i_bra];
                    }}
                    let tmp_slice_t = rt::asarray((&tmp_slice_stage, [nao, naux].f(), &device));
                    let v2_all = (&ipv_t % &tmp_slice_t.t()); // [9*qj, nao]
                    let v2_raw = v2_all.into_shape(-1).into_raw();
                    // V2_all F-order [9*qj, nao]: element (c*qj+j, k) at (c*qj+j) + k*(9*qj)
                    // Part 2[c] = Σ_{j,k} V2_all[c*qj+j, k] · dm0[(q0+j)*nao + k]
                    for c in 0..9 {
                        let mut s = 0.0;
                        for j in 0..qj { for k in 0..nao {
                            s += v2_raw[(c * qj + j) + k * (9 * qj)]
                                * dm0[(q0 + j) * nao + k];
                        }}
                        part2_acc[c] += s;
                    }
                }
            }

            // ── Part 3: for-loop dot (small) ──
            let mut part3 = vec![0.0; 9];
            for c in 0..9 { for j_ao in 0..qj { for k_ao in 0..ni {
                part3[c] += vk2buf[c * nao3 + (q0 + j_ao) * nao + (p0 + k_ao)]
                    * dm0[(q0 + j_ao) * nao + (p0 + k_ao)];
            }}}

            for x in 0..3 { for y in 0..3 {
                let c = x * 3 + y;
                out[i_t(i0, j0, x, y)] = part1_acc[c] + part2_acc[c] + part3[c];
            }}
        }
    }
}
fn g5_ek_ri1_blas(ctx: &EjEkContext, ip12: &[f64], out: &mut [f64]) {
    let nao = ctx.nao; let naux = ctx.naux; let nocc = ctx.nocc;
    let natm = ctx.natm; let nao3 = ctx.nao3;
    let blk = ctx.blk; let aux_blk = ctx.aux_blk;
    let dm0 = ctx.dm0; let mc2 = ctx.mc2;
    let i21 = ctx.i21; let rk = ctx.rk; let wki = ctx.wki; let tmpf = ctx.tmpf;
    let device = ctx.device.clone();
    let i_t = |i0: usize, j0: usize, x: usize, y: usize| -> usize {
        i0 * natm * 9 + j0 * 9 + x * 3 + y
    };

    // wk1_IpJ_full via GEMM: wki @ dm0 (unchanged)
    let n3a = nao * naux * 3;
    let mut wki_stage = vec![0.0; n3a * nao];
    for i in 0..nao { for p in 0..naux { for y in 0..3 {
        let ipy = i * naux * 3 + p * 3 + y;
        for j in 0..nao { wki_stage[ipy + j * n3a] = wki[i * naux * 3 * nao + p * 3 * nao + y * nao + j]; }
    }}}
    let dm0_t = rt::asarray((dm0, [nao, nao].f(), &device));
    let wki_t = rt::asarray((&wki_stage, [n3a, nao].f(), &device));
    let wk1_full = &wki_t % &dm0_t;
    let mut wk1_IpJ_full = vec![0.0; nao * naux * 3 * nao];
    for i in 0..nao { for p in 0..naux { for y in 0..3 { for k in 0..nao {
        wk1_IpJ_full[i * naux * 3 * nao + p * 3 * nao + y * nao + k] = wk1_full[[i * naux * 3 + p * 3 + y, k]];
    }}}}

    // mc2_t: F-order [nocc, nao], element (i_occ, J) = mc2[J*nocc + i_occ]
    let mc2_t = rt::asarray((mc2, [nocc, nao].f(), &device));

    // i21 staged for batched wk1_pJI: [3*naux, naux] F-order
    let mut i21_batch = vec![0.0; 3 * naux * naux];
    for y in 0..3 { for p in 0..naux { for q in 0..naux {
        i21_batch[(y * naux + p) + q * (3 * naux)] = i21[y * naux * naux + p * naux + q];
    }}}
    let i21_batch_t = rt::asarray((&i21_batch, [3 * naux, naux].f(), &device));

    for i0 in 0..natm { let (_, p0, ni) = blk[i0];
        // wkp from tmpf (no rho_ip1 intermediate)
        // wkp[p, x, ii, j] = tmpf[p + (x*nao² + (p0+ii)*nao + j)*naux]
        let ni_nao = ni * nao;
        let mut wkp = vec![0.0; naux * 3 * ni_nao];
        for p in 0..naux { for x in 0..3 { for ii in 0..ni { for j in 0..nao {
            wkp[p * 3 * ni_nao + x * ni_nao + ii * nao + j] =
                tmpf[p + (x * nao3 + (p0 + ii) * nao + j) * naux];
        }}}}

        // rk_P_I (1 GEMM) — unchanged
        let mut rk_P_I = vec![0.0; naux * nocc * ni];
        {
            let mut rk_stage = vec![0.0; naux * nocc * nao];
            for p in 0..naux { for j_occ in 0..nocc { for l in 0..nao {
                rk_stage[(p * nocc + j_occ) + l * (naux * nocc)] =
                    rk[p + l * naux + j_occ * naux * nao];
            }}}
            let rk_2d_g5 = rt::asarray((&rk_stage, [naux * nocc, nao].f(), &device));
            let dm0_block_t = dm0_t.i((.., p0..p0 + ni));
            let rk_P_I_2d = (&rk_2d_g5 % &dm0_block_t);
            let rk_P_I_raw = rk_P_I_2d.into_shape(-1).into_raw();
            for p in 0..naux { for j_occ in 0..nocc { for ii in 0..ni {
                rk_P_I[p * nocc * ni + j_occ * ni + ii] =
                    rk_P_I_raw[(p * nocc + j_occ) + ii * (naux * nocc)];
            }}}
        }

        // rk_PJI (1 GEMM) — unchanged
        let mut rk_PJI = vec![0.0; naux * nao * ni];
        {
            let mut rk_P_I_stage = vec![0.0; naux * ni * nocc];
            for p in 0..naux { for ii in 0..ni { for j_occ in 0..nocc {
                rk_P_I_stage[(p * ni + ii) + j_occ * (naux * ni)] =
                    rk_P_I[p * nocc * ni + j_occ * ni + ii];
            }}}
            let rk_P_I_t = rt::asarray((&rk_P_I_stage, [naux * ni, nocc].f(), &device));
            let rk_PJI_2d = (&rk_P_I_t % &mc2_t);
            let rk_PJI_raw = rk_PJI_2d.into_shape(-1).into_raw();
            for p in 0..naux { for ii in 0..ni { for j in 0..nao {
                rk_PJI[p * nao * ni + j * ni + ii] =
                    rk_PJI_raw[(p * ni + ii) + j * (naux * ni)];
            }}}
        }

        // wk1_pJI via 1 batched GEMM [3*naux, naux] @ [naux, nao*ni] → [3*naux, nao*ni]
        //   result[y*naux+p, J*ni+ii] = Σ_q i21[y, p, q] · rk_PJI[q, J, ii] = wk1_pJI[y, p, J, ii] ✓
        // scatter: wk1_pJI[y*naux*nao*ni + p*nao*ni + J*ni + ii] = result[y*naux+p, J*ni+ii]
        let rk_pji_t_g5 = rt::asarray((&rk_PJI, [nao * ni, naux].f(), &device));
        let wk1_pJI_res = &i21_batch_t % &rk_pji_t_g5.t(); // [3*naux, nao*ni]
        let mut wk1_pJI = vec![0.0; 3 * naux * nao * ni];
        let wk1_pJI_raw = wk1_pJI_res.into_shape(-1).into_raw();
        for y in 0..3 { for p in 0..naux { for J in 0..nao { for ii in 0..ni {
            wk1_pJI[y * naux * nao * ni + p * nao * ni + J * ni + ii] =
                wk1_pJI_raw[(y * naux + p) + (J * ni + ii) * (3 * naux)];
        }}}}

        // wk1_IpJ slice from wk1_IpJ_full — unchanged
        let mut wk1_IpJ = vec![0.0; ni * naux * 3 * nao];
        for ii in 0..ni { for p in 0..naux { for y in 0..3 { for k in 0..nao {
            wk1_IpJ[ii * naux * 3 * nao + p * 3 * nao + y * nao + k] =
                wk1_IpJ_full[(p0 + ii) * naux * 3 * nao + p * 3 * nao + y * nao + k];
        }}}}

        // rho2c_PQ via GEMM: wkp_2d @ rk_PJI_2d (read from tmpf directly, no wkp intermediate)
        let naux3 = naux * 3;
        let mut wkp_stage_g5 = vec![0.0; naux3 * ni_nao];
        for x in 0..3 { for p in 0..naux { for ii in 0..ni { for j in 0..nao {
            wkp_stage_g5[(x * naux + p) + (ii * nao + j) * naux3] =
                tmpf[p + (x * nao3 + (p0 + ii) * nao + j) * naux];
        }}}}
        let wkp_t_g5 = rt::asarray((&wkp_stage_g5, [naux3, ni_nao].f(), &device));
        let mut rk_pji_rho_stage = vec![0.0; ni_nao * naux];
        for q in 0..naux { for J in 0..nao { for ii in 0..ni {
            rk_pji_rho_stage[(ii * nao + J) + q * ni_nao] = rk_PJI[q * nao * ni + J * ni + ii];
        }}}
        let rk_pji_rho2c_t = rt::asarray((&rk_pji_rho_stage, [ni_nao, naux].f(), &device));
        let rho2c_res = &wkp_t_g5 % &rk_pji_rho2c_t;
        let mut rho2c_PQ = vec![0.0; 3 * naux * naux];
        for x in 0..3 { for q in 0..naux { for p in 0..naux {
            rho2c_PQ[x * naux * naux + q * naux + p] = rho2c_res[[x * naux + p, q]];
        }}}

        // T1-T4 for-loops
        for j0 in 0..natm { let (_, aq0, ql) = aux_blk[j0]; if ql == 0 { continue; }
            // T1-T4: for-loops (staging overhead exceeds GEMM benefit for these contractions)
            let mut t1 = vec![0.0; 9];
            for x in 0..9 { let mut s = 0.0; for ii in 0..ni { for j in 0..nao { for qp in 0..ql {
                let pg = aq0 + qp;
                s += ip12[x * nao3 * naux + (p0+ii) * nao * naux + j * naux + pg]
                    * rk_PJI[pg * nao * ni + j * ni + ii];
            }}} t1[x] = s; }
            let mut t2 = vec![0.0; 9];
            for x in 0..3 { for y in 0..3 { let mut s = 0.0; for qp in 0..ql { let pg = aq0 + qp;
                for ii in 0..ni { for j in 0..nao {
                    s += wkp[pg * 3 * ni * nao + x * ni * nao + ii * nao + j]
                        * wk1_pJI[y * naux * nao * ni + pg * nao * ni + j * ni + ii];
            }} } t2[x*3+y] = s; }}
            let mut t3 = vec![0.0; 9];
            for x in 0..3 { for y in 0..3 { let mut s = 0.0; for qp in 0..ql { let qg = aq0 + qp;
                for paux in 0..naux {
                    s += rho2c_PQ[x * naux * naux + qg * naux + paux] * i21[y * naux * naux + qg * naux + paux];
            }} t3[x*3+y] = s; }}
            let mut t4 = vec![0.0; 9];
            for x in 0..3 { for y in 0..3 { let mut s = 0.0; for qp in 0..ql { let pg = aq0 + qp;
                for ii in 0..ni { for j in 0..nao {
                    s += wkp[pg * 3 * ni * nao + x * ni * nao + ii * nao + j]
                        * wk1_IpJ[ii * naux * 3 * nao + pg * 3 * nao + y * nao + j];
            }} } t4[x*3+y] = s; }}
            for x in 0..3 { for y in 0..3 {
                let v = t1[x*3+y] - t2[x*3+y] - t3[x*3+y] + t4[x*3+y];
                out[i_t(i0,j0,x,y)] += v;
                out[i_t(j0,i0,x,y)] += t1[y*3+x] - t2[y*3+x] - t3[y*3+x] + t4[y*3+x];
            }}
        }
    }
}
#[allow(dead_code)]
fn g6_ek_ri2d_blas(ctx: &EjEkContext, ipip2: &[f64], out: &mut [f64]) {
    let nao = ctx.nao; let naux = ctx.naux; let nocc = ctx.nocc; let natm = ctx.natm;
    let nao3 = ctx.nao3;
    let aux_blk = ctx.aux_blk; let mc2 = ctx.mc2; let rkoo = ctx.rkoo;
    let r2c0 = ctx.r2c0; let i211 = ctx.i211;
    let device = ctx.device.clone();
    let i_t = |i0: usize, j0: usize, x: usize, y: usize| -> usize {
        i0 * natm * 9 + j0 * 9 + x * 3 + y
    };

    // mc2_t: F-order [nocc, nao], element (i, p) = mc2[p*nocc + i] = mc2[p, i] (math)
    //   storage: mc2[p*nocc + i] → F-order [nocc, nao] flat: i + p*nocc = p*nocc + i ✓
    let mc2_t = rt::asarray((mc2, [nocc, nao].f(), &device));

    for i0 in 0..natm { let (_, ap0, ni_aux) = aux_blk[i0];
        if ni_aux == 0 { continue; }

        // ── rkj[p, J, I] = Σ_{i,j} rkoo[ap0+p, i, j] · mc2[J, j] · mc2[I, i]  (2 GEMMs) ──
        // Step 1: tmp1[p, i, J] = Σ_j rkoo[p, i, j] · mc2[J, j]
        //   staging rkoo_all F-order [ni_aux*nocc, nocc]: element (p*nocc+i, j) = rkoo[(ap0+p)*nocc² + i*nocc + j]
        //     fill: idx = (p*nocc+i) + j*(ni_aux*nocc)   [F-order: i + j*M, M = ni_aux*nocc]
        //   GEMM: tmp1 = rkoo_all @ mc2_t  → [ni_aux*nocc, nao]
        //   tmp1[p*nocc+i, J] = Σ_j rkoo_all[p*nocc+i, j] · mc2_t[j, J]
        //                     = Σ_j rkoo[p, i, j] · mc2[J, j] ✓
        let mut rkoo_stage = vec![0.0; ni_aux * nocc * nocc];
        for p in 0..ni_aux { for i in 0..nocc { for j in 0..nocc {
            rkoo_stage[(p * nocc + i) + j * (ni_aux * nocc)] =
                rkoo[(ap0 + p) * nocc * nocc + i * nocc + j];
        }}}
        let rkoo_t = rt::asarray((&rkoo_stage, [ni_aux * nocc, nocc].f(), &device));
        let tmp1 = (&rkoo_t % &mc2_t); // [ni_aux*nocc, nao]
        let tmp1_raw = tmp1.into_shape(-1).into_raw();
        // Step 2: rkj[p, J, I] = Σ_i tmp1[p, i, J] · mc2[I, i] = Σ_i tmp1[p*nocc+i, J] · mc2_t[i, I]
        //   staging tmp1_t_2d F-order [ni_aux*nao, nocc]: element (p*nao+J, i) = tmp1[p*nocc+i, J]
        //     fill: idx = (p*nao+J) + i*(ni_aux*nao)   [F-order: i + j*M, M = ni_aux*nao]
        //     source: tmp1_raw[(p*nocc+i) + J*(ni_aux*nocc)]  (F-order flat of [ni_aux*nocc, nao])
        //   GEMM: rkj_2d = tmp1_t_2d @ mc2_t  → [ni_aux*nao, nao]
        //   rkj_2d[p*nao+J, I] = Σ_i tmp1_t_2d[p*nao+J, i] · mc2_t[i, I]
        //                     = Σ_i tmp1[p, i, J] · mc2[I, i] = rkj[p, J, I] ✓
        //   scatter: rkj[p*nao² + J*nao + I] = rkj_2d_raw[(p*nao+J) + I*(ni_aux*nao)]
        let mut rkj = vec![0.0; ni_aux * nao * nao];
        {
            let mut tmp1_t_stage = vec![0.0; ni_aux * nao * nocc];
            for p in 0..ni_aux { for j_idx in 0..nao { for i in 0..nocc {
                tmp1_t_stage[(p * nao + j_idx) + i * (ni_aux * nao)] =
                    tmp1_raw[(p * nocc + i) + j_idx * (ni_aux * nocc)];
            }}}
            let tmp1_t_t = rt::asarray((&tmp1_t_stage, [ni_aux * nao, nocc].f(), &device));
            let rkj_2d = (&tmp1_t_t % &mc2_t); // [ni_aux*nao, nao]
            let rkj_2d_raw = rkj_2d.into_shape(-1).into_raw();
            for p in 0..ni_aux { for j_idx in 0..nao { for i_idx in 0..nao {
                rkj[p * nao * nao + j_idx * nao + i_idx] =
                    rkj_2d_raw[(p * nao + j_idx) + i_idx * (ni_aux * nao)];
            }}}
        }

        // ── ta[x] = 0.5 * Σ_{I,J,p} ipip2[x, I, J, ap0+p] · rkj[p, I, J]  (1 GEMM) ──
        // rkj is symmetric in (I, J) (proven via t3c[l,j]=t3c[j,l] symmetry), so
        // rkj[p, I, J] = rkj[p, J, I]. Original loop reads rkj[p*nao² + I*nao + J] = rkj[p, J'=I, I'=J]
        // = rkj[p, I, J] (by symmetry). We use rkj in stored (p, J, I) ordering for zero-copy view.
        //
        // staging ipip2_block_2d F-order [9, ni_aux*nao²]: element (x, p*nao²+J*nao+I) = ipip2[x, I, J, ap0+p]
        //   fill: idx = x + (p*nao² + J*nao + I)*9   [F-order: i + j*M, M = 9]
        //   source: ipip2[x*nao²*naux + I*nao*naux + J*naux + (ap0+p)]
        // rkj_col: zero-copy view F-order [ni_aux*nao², 1]: element (p*nao²+J*nao+I, 0) = rkj[p, J, I]
        //   (rkj storage IS p*nao²+J*nao+I, matching F-order flat of [ni_aux*nao², 1])
        // GEMM: ta_col[9, 1] = ipip2_block_2d @ rkj_col
        //   ta_col[x, 0] = Σ_{(p,J,I)} ipip2[x, I, J, ap0+p] · rkj[p, J, I]
        //                = Σ_{I,J,p} ipip2[x, I, J, ap0+p] · rkj[p, I, J]  (by symmetry)
        //                = 2 * ta[x] ✓
        let mut ta = vec![0.0; 9];
        {
            let npij = ni_aux * nao3;
            let mut ipip2_stage = vec![0.0; 9 * npij];
            for x in 0..9 { for p in 0..ni_aux { for j in 0..nao { for i in 0..nao {
                ipip2_stage[x + (p * nao3 + j * nao + i) * 9] =
                    ipip2[x * nao3 * naux + i * nao * naux + j * naux + (ap0 + p)];
            }}}}
            let ipip2_t_2d = rt::asarray((&ipip2_stage, [9, npij].f(), &device));
            let rkj_col_t = rt::asarray((&rkj, [npij, 1].f(), &device));
            let ta_col = (&ipip2_t_2d % &rkj_col_t); // [9, 1]
            let ta_col_raw = ta_col.into_shape(-1).into_raw();
            for x in 0..9 { ta[x] = 0.5 * ta_col_raw[x]; }
        }

        // ── tb[x] = -0.5 * Σ_{p,q} r2c0[ap0+p, q] · i211[x, ap0+p, q]  (1 GEMM) ──
        // staging i211_block_2d F-order [9, ni_aux*naux]: element (x, p*naux+q) = i211[x, ap0+p, q]
        //   fill: idx = x + (p*naux+q)*9
        //   source: i211[x*naux² + (ap0+p)*naux + q]
        // r2c0_col: zero-copy view F-order [ni_aux*naux, 1]: element (p*naux+q, 0) = r2c0[ap0+p, q]
        //   r2c0 stored as r2c0[(ap0+p)*naux + q]; slice &r2c0[ap0*naux..] has element at p*naux+q ✓
        // GEMM: tb_col[9, 1] = i211_block_2d @ r2c0_col
        //   tb_col[x, 0] = Σ_{p,q} i211[x, ap0+p, q] · r2c0[ap0+p, q] = -2*tb[x] ✓
        let mut tb = vec![0.0; 9];
        {
            let npq = ni_aux * naux;
            let mut i211_stage = vec![0.0; 9 * npq];
            for x in 0..9 { for p in 0..ni_aux { for q in 0..naux {
                i211_stage[x + (p * naux + q) * 9] =
                    i211[x * naux * naux + (ap0 + p) * naux + q];
            }}}
            let i211_t_2d = rt::asarray((&i211_stage, [9, npq].f(), &device));
            let r2c0_block = &r2c0[ap0 * naux..];
            let r2c0_col_t = rt::asarray((r2c0_block, [npq, 1].f(), &device));
            let tb_col = (&i211_t_2d % &r2c0_col_t); // [9, 1]
            let tb_col_raw = tb_col.into_shape(-1).into_raw();
            for x in 0..9 { tb[x] = -0.5 * tb_col_raw[x]; }
        }

        for x in 0..3 { for y in 0..3 { out[i_t(i0, i0, x, y)] += ta[x * 3 + y] + tb[x * 3 + y]; }}
    }
}

fn g7_ek_ri2o_blas(ctx: &EjEkContext, out: &mut [f64]) {
    let nao = ctx.nao;
    let naux = ctx.naux;
    let nocc = ctx.nocc;
    let natm = ctx.natm;
    let nao3 = ctx.nao3;
    let _ = (nao, nao3);
    let aux_blk = ctx.aux_blk;
    let i2inv = ctx.i2inv;
    let i21 = ctx.i21;
    let i2ip = ctx.i2ip;
    let wk2 = ctx.wk2;
    let rkoo = ctx.rkoo;
    let r2c0 = ctx.r2c0;
    let device = ctx.device.clone();
    let i_t = |i0: usize, j0: usize, x: usize, y: usize| -> usize {
        i0 * natm * 9 + j0 * 9 + x * 3 + y
    };
    let nocc2 = nocc * nocc;

    // ══════════════════════════════════════════════════════════
    // Part A: per-atom rho2c1 (replaces Inline lines 858-907)
    //
    // Source layouts (verified from Inline code, all row-major):
    //   i21   : [3, naux, naux], element (x, p, q) at x*naux² + p*naux + q
    //   i2inv : treated as row-major (ap0+r, p) at (ap0+r)*naux + p
    //           (i2inv is symmetric, so row/col-major give same values)
    //   r2c0  : row-major (r, q) at r*naux + q  (also symmetric)
    //   wk2   : [naux, 3, nocc, nocc], element (r, x, i, j) at r*3*nocc² + x*nocc² + i*nocc + j
    //   rkoo  : [naux, nocc, nocc], element (q, i, j) at q*nocc² + i*nocc + j
    //
    // Output: rho2c1_per_atom[x, p, q] per atom, row-major [3, naux, naux]
    // ══════════════════════════════════════════════════════════
    let mut rho2c1_per_atom = vec![0.0; natm * 3 * naux * naux];

    // Pre-stage i2inv and r2c0 as full col-major [naux, naux] views (zero-copy,
    // since the storage [naux, naux].f() reads element (r, p) at r + p*naux,
    // and the Inline reads i2inv[(ap0+r)*naux + p] = i2inv[(ap0+r) + p*naux] when
    // interpreted as col-major (p, ap0+r). Because i2inv is symmetric, both
    // interpretations yield the same value. We use the col-major view directly.
    let i2inv_cm: TsrView<f64> = rt::asarray((i2inv, [naux, naux].f(), &device));
    let r2c0_cm: TsrView<f64> = rt::asarray((r2c0, [naux, naux].f(), &device));

    for i0 in 0..natm {
        let (_, ap0, ni_aux) = aux_blk[i0];
        if ni_aux == 0 { continue; }

        // ── A1: ip1_2c = i21[:, ap0:ap0+ni_aux, :] @ i2inv  (3 GEMMs as one big GEMM) ──
        // staging: shape [3*ni_aux, naux].f(), element (x*ni+p, q) at (x*ni+p) + q*(3*ni)
        //   source: i21[x, ap0+p, q] = i21[x*naux² + (ap0+p)*naux + q]
        let mut i21_stage = vec![0.0; 3 * ni_aux * naux];
        for x in 0..3 { for p in 0..ni_aux { for q in 0..naux {
            i21_stage[(x * ni_aux + p) + q * (3 * ni_aux)] =
                i21[x * naux * naux + (ap0 + p) * naux + q];
        }}}
        let i21_mm = rt::asarray((&i21_stage, [3 * ni_aux, naux].f(), &device));
        // ip1_2c [3*ni_aux, naux] = i21_mm @ i2inv_cm
        let ip1_2c_t = &i21_mm % &i2inv_cm;
        // ip1_r2c [3*ni_aux, naux] = 0.5 * i21_mm @ r2c0_cm
        let ip1_r2c_t = (&i21_mm % &r2c0_cm) * 0.5;

        // Copy ip1_2c, ip1_r2c into row-major [3, ni_aux, naux] for consistent indexing
        // (matches Inline layout: [x*ni_aux*naux + r*naux + c])
        let mut ip1_2c = vec![0.0; 3 * ni_aux * naux];
        let mut ip1_r2c = vec![0.0; 3 * ni_aux * naux];
        for x in 0..3 { for r in 0..ni_aux { for c in 0..naux {
            // ip1_2c_t is col-major [3*ni_aux, naux], element (x*ni+r, c) at (x*ni+r) + c*(3*ni)
            let v2 = ip1_2c_t[[x * ni_aux + r, c]];
            let vr = ip1_r2c_t[[x * ni_aux + r, c]];
            ip1_2c[x * ni_aux * naux + r * naux + c] = v2;
            ip1_r2c[x * ni_aux * naux + r * naux + c] = vr;
        }}}

        // ── A2: s1, s2 via GEMM ──
        // s1[x, p, q] = Σ_r ip1_r2c[x, r, q] · i2inv[ap0+r, p]
        //   = (i2inv_block^T @ ip1_r2c_x)[p, q]  where i2inv_block[r, p] = i2inv[ap0+r, p]
        //   ip1_r2c_x is [ni_aux, naux] (rows r, cols q); we need ip1_r2c_x^T @ ... no,
        //   let's recompute: s1[p, q] = Σ_r i2inv[ap0+r, p] * ip1_r2c[r, q]
        //     = (i2inv_block^T)[p, r] @ ip1_r2c_x[r, q]   → [naux, naux]
        //   i2inv_block: extract rows [ap0..ap0+ni_aux], all cols → col-major [ni_aux, naux].f()
        //     element (r, p) = i2inv[ap0+r, p] = i2inv[(ap0+r) + p*naux] (col-major symmetric)
        //   i2inv_block^T: [naux, ni_aux]
        //   ip1_r2c_x: [ni_aux, naux], element (r, q) = ip1_r2c[x, r, q]
        //
        // s2[x, p, q] = Σ_r ip1_2c[x, r, p] · r2c0[ap0+r, q]
        //   = (ip1_2c_x^T)[p, r] @ r2c0_block[r, q]  → [naux, naux]
        //   r2c0_block: rows [ap0..ap0+ni_aux], all cols → col-major [ni_aux, naux].f()
        //     element (r, q) = r2c0[ap0+r, q] = r2c0[(ap0+r) + q*naux] (col-major symmetric)

        // Build i2inv_block and r2c0_block as col-major [ni_aux, naux].f()
        // staging: element (r, p) at r + p*ni_aux
        let mut i2inv_block_stage = vec![0.0; ni_aux * naux];
        let mut r2c0_block_stage = vec![0.0; ni_aux * naux];
        for r in 0..ni_aux { for p in 0..naux {
            // i2inv[ap0+r, p] — read via the same expression Inline uses (symmetric safe)
            i2inv_block_stage[r + p * ni_aux] = i2inv[(ap0 + r) * naux + p];
            r2c0_block_stage[r + p * ni_aux] = r2c0[(ap0 + r) * naux + p];
        }}
        let i2inv_block_t = rt::asarray((&i2inv_block_stage, [ni_aux, naux].f(), &device));
        let r2c0_block_t = rt::asarray((&r2c0_block_stage, [ni_aux, naux].f(), &device));

        // s1[x, p, q] = Σ_r i2inv_block[r, p] · ip1_r2c[x, r, q]
        //   = (i2inv_block^T)[p, r] @ ip1_r2c_x[r, q]  → [naux, naux]
        //   result is col-major [naux, naux].f()
        let mut s1 = vec![0.0; 3 * naux * naux];
        let mut s2 = vec![0.0; 3 * naux * naux];
        for x in 0..3 {
            // ip1_r2c_x: stage as col-major [ni_aux, naux].f(), element (r, q) at r + q*ni_aux
            //   source: ip1_r2c[x, r, q] = ip1_r2c[x*ni_aux*naux + r*naux + q]
            let mut ipr_x = vec![0.0; ni_aux * naux];
            for r in 0..ni_aux { for q in 0..naux {
                ipr_x[r + q * ni_aux] = ip1_r2c[x * ni_aux * naux + r * naux + q];
            }}
            let ipr_x_t = rt::asarray((&ipr_x, [ni_aux, naux].f(), &device));
            // s1_x = i2inv_block^T @ ipr_x_t  → [naux, naux].f()
            let s1_x = &i2inv_block_t.t() % &ipr_x_t;
            // Copy to row-major [naux, naux]: element (p, q) at p*naux + q
            // s1_x is col-major [naux, naux], element (p, q) at p + q*naux
            for p in 0..naux { for q in 0..naux {
                s1[x * naux * naux + p * naux + q] = s1_x[[p, q]];
            }}

            // ip1_2c_x: stage as col-major [ni_aux, naux].f(), element (r, p) at r + p*ni_aux
            //   source: ip1_2c[x, r, p] = ip1_2c[x*ni_aux*naux + r*naux + p]
            let mut ip2_x = vec![0.0; ni_aux * naux];
            for r in 0..ni_aux { for p in 0..naux {
                ip2_x[r + p * ni_aux] = ip1_2c[x * ni_aux * naux + r * naux + p];
            }}
            let ip2_x_t = rt::asarray((&ip2_x, [ni_aux, naux].f(), &device));
            // s2_x = ip1_2c_x^T @ r2c0_block_t  → [naux, naux].f()
            //   s2[p, q] = Σ_r ip1_2c_x[r, p] · r2c0_block[r, q]
            //   = (ip1_2c_x^T)[p, r] @ r2c0_block[r, q]
            let s2_x = &ip2_x_t.t() % &r2c0_block_t;
            for p in 0..naux { for q in 0..naux {
                s2[x * naux * naux + p * naux + q] = s2_x[[p, q]];
            }}
        }

        // ── A3: tmp_so via two GEMM stages ──
        // inner[x, r, q] = Σ_{i,j} wk2[ap0+r, x, i, j] · rkoo[q, i, j]
        //   = (wk2_x_2d)[r, ij] @ (rkoo_t)[ij, q]  → [ni_aux, naux]
        //   wk2_x_2d: stage [ni_aux, nocc²].f(), element (r, ij) at r + ij*ni_aux
        //     source: wk2[ap0+r, x, i, j] = wk2[(ap0+r)*3*nocc² + x*nocc² + i*nocc + j]
        //     ij = i*nocc + j
        //   rkoo_t: stage [nocc², naux].f(), element (ij, q) at ij + q*nocc²
        //     source: rkoo[q, i, j] = rkoo[q*nocc² + i*nocc + j]
        //
        // tmp_so[x, p, q] = Σ_r inner[x, r, q] · i2inv[ap0+r, p]
        //   = (i2inv_block^T)[p, r] @ inner_x[r, q]  → [naux, naux]
        //
        // The Inline computes TWO terms (with p↔q swap). We observe:
        //   val -= wk2[ap0+r,x,i,j] * rkoo[q,i,j] * i2inv[ap0+r, p]   ← tmp_so[x, p, q]
        //   val -= wk2[ap0+r,x,i,j] * rkoo[p,i,j] * i2inv[ap0+r, q]   ← tmp_so[x, q, p]
        // So val = s1 + s2 - tmp_so[x, p, q] - tmp_so[x, q, p]
        //        = s1 + s2 - tmp_so[x, p, q] - tmp_so[x, p, q]^T
        // (The Inline subtracts both, matching r2c1 -= tmp_so + tmp_so^T.)

        // rkoo_t (shared across all x): [nocc², naux].f()
        let mut rkoo_stage = vec![0.0; nocc2 * naux];
        for ij in 0..nocc2 { for q in 0..naux {
            // rkoo[q, i, j] with ij = i*nocc + j → rkoo[q*nocc² + ij]
            rkoo_stage[ij + q * nocc2] = rkoo[q * nocc2 + ij];
        }}
        let rkoo_t = rt::asarray((&rkoo_stage, [nocc2, naux].f(), &device));

        let mut tmp_so = vec![0.0; 3 * naux * naux];
        for x in 0..3 {
            // wk2_x_2d: [ni_aux, nocc²].f()
            let mut wk2_x_stage = vec![0.0; ni_aux * nocc2];
            for r in 0..ni_aux { for ij in 0..nocc2 {
                // wk2[ap0+r, x, ij] = wk2[(ap0+r)*3*nocc² + x*nocc² + ij]
                wk2_x_stage[r + ij * ni_aux] = wk2[(ap0 + r) * 3 * nocc2 + x * nocc2 + ij];
            }}
            let wk2_x_t = rt::asarray((&wk2_x_stage, [ni_aux, nocc2].f(), &device));
            // inner_x = wk2_x_t @ rkoo_t → [ni_aux, naux].f()
            let inner_x = &wk2_x_t % &rkoo_t;
            // tmp_so_x = i2inv_block^T @ inner_x → [naux, naux].f()
            let tmp_so_x = &i2inv_block_t.t() % &inner_x;
            for p in 0..naux { for q in 0..naux {
                tmp_so[x * naux * naux + p * naux + q] = tmp_so_x[[p, q]];
            }}
        }

        // ── Assemble r2c1[x, p, q] = s1 + s2 - tmp_so[x, p, q] - tmp_so[x, q, p] ──
        for x in 0..3 { for p in 0..naux { for q in 0..naux {
            let val = s1[x * naux * naux + p * naux + q]
                + s2[x * naux * naux + p * naux + q]
                - tmp_so[x * naux * naux + p * naux + q]
                - tmp_so[x * naux * naux + q * naux + p];
            rho2c1_per_atom[i0 * 3 * naux * naux + x * naux * naux + p * naux + q] = val;
        }}}
    }

    // ══════════════════════════════════════════════════════════
    // Part B: pair loop (replaces Inline lines 908-970)
    // T1, T2: small for-loops kept (per spec, not worth GEMM overhead).
    // T3: keep existing BLAS chain (already optimized in Inline).
    // ══════════════════════════════════════════════════════════
    for i0 in 0..natm { let (_, ap0, ni_aux) = aux_blk[i0];
        let r2c1 = &rho2c1_per_atom[i0 * 3 * naux * naux..];
        for j0 in 0..natm { let (_, aq0, nq) = aux_blk[j0];
            // T1: 0.5 * Σ_{p,q} r2c0[ap0+p, aq0+q] · i2ip[xy, ap0+p, aq0+q]
            let mut t1 = vec![0.0; 9];
            for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                for p in 0..ni_aux { for q in 0..nq {
                    s += r2c0[(ap0 + p) * naux + (aq0 + q)]
                        * i2ip[(x * 3 + y) * naux * naux + (ap0 + p) * naux + (aq0 + q)];
                }} t1[x * 3 + y] = s * 0.5;
            }}
            // T2: Σ_{p,q} r2c1[x, aq0+p, q] · i21[y, aq0+p, q]
            let mut t2 = vec![0.0; 9];
            for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                for p in 0..nq { for q in 0..naux {
                    s += r2c1[x * naux * naux + (aq0 + p) * naux + q]
                        * i21[y * naux * naux + (aq0 + p) * naux + q];
                }} t2[x * 3 + y] = s;
            }}
            // T3: 0.5 * Σ_{p,q,i,j} wk2[ap0+p, x, i, j] · i2inv[ap0+p, aq0+q] · wk2[aq0+q, y, i, j]
            // Existing BLAS matmul chain (kept verbatim from Inline).
            let ni_t3 = ni_aux; let nj_t3 = nq;
            let mut t3 = vec![0.0; 9];
            {
                let mut wi_block = vec![0.0; ni_t3 * 3 * nocc2];
                let mut wj_block = vec![0.0; nj_t3 * 3 * nocc2];
                for p in 0..ni_t3 { for x in 0..3 { for i in 0..nocc { for j in 0..nocc {
                    let src = wk2[(ap0 + p) * 3 * nocc2 + x * nocc2 + i * nocc + j];
                    wi_block[p + (x * nocc2 + i * nocc + j) * ni_t3] = src;
                }}}}
                for p in 0..nj_t3 { for x in 0..3 { for i in 0..nocc { for j in 0..nocc {
                    let src = wk2[(aq0 + p) * 3 * nocc2 + x * nocc2 + i * nocc + j];
                    wj_block[p + (x * nocc2 + i * nocc + j) * nj_t3] = src;
                }}}}
                let wi_t = rt::asarray((&wi_block, [ni_t3, 3 * nocc2].f(), &device));
                let wj_t = rt::asarray((&wj_block, [nj_t3, 3 * nocc2].f(), &device));
                let mut i2inv_sub = vec![0.0; ni_t3 * nj_t3];
                for p in 0..ni_t3 { for q in 0..nj_t3 {
                    i2inv_sub[p + q * ni_t3] = i2inv[(ap0 + p) * naux + (aq0 + q)];
                }}
                let i2_sub = rt::asarray((&i2inv_sub, [ni_t3, nj_t3].f(), &device));
                for x in 0..3 {
                    let wi_x = wi_t.i((.., x * nocc2..(x + 1) * nocc2));
                    for y in 0..3 {
                        let wj_y = wj_t.i((.., y * nocc2..(y + 1) * nocc2));
                        let outer = &wi_x % &wj_y.t();
                        let prod = &outer * i2_sub.i((.., .., None));
                        let val = prod.sum_axes(&[0, 1]).into_shape(-1).into_raw()[0];
                        t3[x * 3 + y] = val * 0.5;
                    }
                }
            }
            for x in 0..3 { for y in 0..3 {
                let ek_val = (t1[x * 3 + y] + t2[x * 3 + y] + t3[x * 3 + y]) * 0.5;
                out[i_t(i0, j0, x, y)] += ek_val;
                out[i_t(j0, i0, x, y)] += (t1[y * 3 + x] + t2[y * 3 + x] + t3[y * 3 + x]) * 0.5;
            }}
        }
    }

}
#[allow(dead_code)]
fn g8_ej_ri1_blas(ctx: &EjEkContext, ip12: &[f64], out: &mut [f64]) {
    let nao = ctx.nao; let naux = ctx.naux; let natm = ctx.natm; let nao3 = ctx.nao3;
    let blk = ctx.blk; let aux_blk = ctx.aux_blk;
    let dm0 = ctx.dm0; let i21 = ctx.i21;
    let r0 = ctx.r0; let rj1 = ctx.rj1; let wj001 = ctx.wj001; let wj2 = ctx.wj2;
    let device = ctx.device.clone();
    let i_t = |i0: usize, j0: usize, x: usize, y: usize| -> usize {
        i0 * natm * 9 + j0 * 9 + x * 3 + y
    };

    for i0 in 0..natm { let (_, p0, ni) = blk[i0];
        // w11: for-loop (staging overhead exceeds GEMM benefit for this contraction)
        let mut w11 = vec![0.0; 9 * naux];
        for x in 0..9 { for p in 0..naux { let mut ss = 0.0;
            for ii in 0..ni { for j in 0..nao {
                ss += ip12[x * nao3 * naux + (p0+ii) * nao * naux + j * naux + p]
                    * dm0[(p0 + ii) * nao + j];
            }} w11[x * naux + p] = ss;
        }}
        // ── t1, t2, t3, t4: keep as for-loops (small: ≤ 9*ql*naux per pair) ──
        for j0 in 0..natm { let (_, aq0, ql) = aux_blk[j0];
            let q0 = aq0;
            let mut t1 = vec![0.0; 9];
            for x in 0..9 { let mut s = 0.0;
                for qp in 0..ql { s += w11[x * naux + q0 + qp] * r0[q0 + qp]; } t1[x] = s;
            }
            let mut t2 = vec![0.0; 9];
            for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                for qp in 0..ql { let qg = q0 + qp;
                    for paux in 0..naux {
                        s += i21[y * naux * naux + qg * naux + paux] * r0[qg] * rj1[i0 * naux * 3 + paux * 3 + x];
                    }
                } t2[x * 3 + y] = s;
            }}
            let mut t3 = vec![0.0; 9];
            for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                for qp in 0..ql { let qg = q0 + qp;
                    s += rj1[i0 * naux * 3 + qg * 3 + x] * wj001[y * naux + qg];
                } t3[x * 3 + y] = s;
            }}
            let mut t4 = vec![0.0; 9];
            for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                for qp in 0..ql { let qg = q0 + qp;
                    s += rj1[i0 * naux * 3 + qg * 3 + x] * wj2[qg * 3 + y];
                } t4[x * 3 + y] = s;
            }}
            for x in 0..3 { for y in 0..3 {
                let v = (t1[x * 3 + y] - t2[x * 3 + y] - t3[x * 3 + y] + t4[x * 3 + y]) * 2.0;
                out[i_t(i0, j0, x, y)] += v;
                out[i_t(j0, i0, x, y)] += (t1[y * 3 + x] - t2[y * 3 + x] - t3[y * 3 + x] + t4[y * 3 + x]) * 2.0;
            }}
        }
    }
}
#[allow(dead_code)]
fn g9_ej_ri2d_blas(ctx: &EjEkContext, ipip2: &[f64], out: &mut [f64]) {
    let nao = ctx.nao; let naux = ctx.naux; let natm = ctx.natm; let nao3 = ctx.nao3;
    let aux_blk = ctx.aux_blk;
    let dm0 = ctx.dm0; let i211 = ctx.i211; let r0 = ctx.r0;
    let device = ctx.device.clone();
    let i_t = |i0: usize, j0: usize, x: usize, y: usize| -> usize {
        i0 * natm * 9 + j0 * 9 + x * 3 + y
    };

    for i0 in 0..natm { let (_, ap0, ni_aux) = aux_blk[i0];
        if ni_aux == 0 { continue; }

        // td: for-loop (staging overhead exceeds GEMM benefit for this contraction)
        let mut td = vec![0.0; 9];
        for x in 0..9 { let mut s = 0.0;
            for i in 0..nao { for j in 0..nao { for p in 0..ni_aux {
                s += ipip2[x * nao * nao * naux + i * nao * naux + j * naux + (ap0 + p)]
                    * dm0[j * nao + i] * r0[ap0 + p];
            }}} td[x] = s;
        }

        // ── te[x] = Σ_{p,q} r0[ap0+p] · i211[x, ap0+p, q] · r0[q]  (keep as for-loop, small) ──
        let mut te = vec![0.0; 9];
        for x in 0..9 { let mut s = 0.0;
            for p in 0..ni_aux { for q in 0..naux {
                s += r0[ap0 + p] * i211[x * naux * naux + (ap0 + p) * naux + q] * r0[q];
            }} te[x] = s;
        }
        for x in 0..3 { for y in 0..3 { out[i_t(i0, i0, x, y)] += td[x * 3 + y] - te[x * 3 + y]; }}
    }
}
#[allow(dead_code)]
fn g10_ej_ri2o_blas(ctx: &EjEkContext, out: &mut [f64]) {
    let naux = ctx.naux; let natm = ctx.natm;
    let aux_blk = ctx.aux_blk;
    let i2inv = ctx.i2inv; let i21 = ctx.i21; let i2ip = ctx.i2ip;
    let r0 = ctx.r0; let wj2 = ctx.wj2; let wj001 = ctx.wj001;
    let device = ctx.device.clone();
    let i_t = |i0: usize, j0: usize, x: usize, y: usize| -> usize {
        i0 * natm * 9 + j0 * 9 + x * 3 + y
    };

    // i2inv_t: zero-copy F-order [naux, naux]
    let i2inv_t = rt::asarray((i2inv, [naux, naux].f(), &device));

    let mut rs1_per = vec![0.0; natm * 3 * naux];
    let mut r01_per = vec![0.0; natm * 3 * naux];
    let mut r10_per = vec![0.0; natm * 3 * naux];
    for i0 in 0..natm { let (_, ap0, ni_aux) = aux_blk[i0];
        if ni_aux == 0 { continue; }

        // ── rs1_per[i0, x, q] = Σ_p wj2[(ap0+p)*3 + x] · i2inv[(ap0+p)*naux + q]  (1 GEMM) ──
        // staging wj2_block_2d F-order [3, ni_aux]: element (x, p) = wj2[(ap0+p)*3 + x]
        //   fill: idx = x + p*3
        //   source: wj2[(ap0+p)*3 + x]
        // i2inv_block: slice i2inv_t rows [ap0, ap0+ni_aux) → [ni_aux, naux] (non-contiguous view, but GEMM handles it)
        //   Actually, i2inv stored as i2inv[(ap0+p) + q*naux] (F-order [naux, naux]).
        //   Staging i2inv_block_2d F-order [ni_aux, naux]: element (p, q) = i2inv[(ap0+p)*naux + q]
        //     fill: idx = p + q*ni_aux  (F-order)
        //     source: i2inv[(ap0+p)*naux + q]  — NOTE: this is row-major [ni_aux, naux] with (p,q) at p*naux+q
        //   F-order [ni_aux, naux] flat: p + q*ni_aux ≠ p*naux + q (unless ni_aux == naux)
        //   So we need staging for i2inv_block.
        // GEMM: rs1 = wj2_block_2d @ i2inv_block_2d → [3, naux]
        //   rs1[x, q] = Σ_p wj2[(ap0+p)*3+x] · i2inv[(ap0+p)*naux+q] ✓
        //   scatter: rs1_per[i0*3*naux + x*naux + q] = rs1_raw[x + q*3]  (F-order flat of [3, naux])
        {
            let mut wj2_stage = vec![0.0; 3 * ni_aux];
            for x in 0..3 { for p in 0..ni_aux {
                wj2_stage[x + p * 3] = wj2[(ap0 + p) * 3 + x];
            }}
            let wj2_t_2d = rt::asarray((&wj2_stage, [3, ni_aux].f(), &device));
            let mut i2inv_block_stage = vec![0.0; ni_aux * naux];
            for p in 0..ni_aux { for q in 0..naux {
                i2inv_block_stage[p + q * ni_aux] = i2inv[(ap0 + p) * naux + q];
            }}
            let i2inv_block_t = rt::asarray((&i2inv_block_stage, [ni_aux, naux].f(), &device));
            let rs1 = (&wj2_t_2d % &i2inv_block_t); // [3, naux]
            let rs1_raw = rs1.into_shape(-1).into_raw();
            for x in 0..3 { for q in 0..naux {
                rs1_per[i0 * 3 * naux + x * naux + q] = rs1_raw[x + q * 3];
            }}
        }

        // ── r01_per[i0, x, q] = Σ_p wj001[x*naux + (ap0+p)] · i2inv[(ap0+p)*naux + q]  (1 GEMM) ──
        // staging wj001_block_2d F-order [3, ni_aux]: element (x, p) = wj001[x*naux + (ap0+p)]
        //   fill: idx = x + p*3
        // i2inv_block_2d: same as rs1 (reuse staging)
        // GEMM: r01 = wj001_block_2d @ i2inv_block_2d → [3, naux]
        {
            let mut wj001_stage = vec![0.0; 3 * ni_aux];
            for x in 0..3 { for p in 0..ni_aux {
                wj001_stage[x + p * 3] = wj001[x * naux + (ap0 + p)];
            }}
            let wj001_t_2d = rt::asarray((&wj001_stage, [3, ni_aux].f(), &device));
            let mut i2inv_block_stage = vec![0.0; ni_aux * naux];
            for p in 0..ni_aux { for q in 0..naux {
                i2inv_block_stage[p + q * ni_aux] = i2inv[(ap0 + p) * naux + q];
            }}
            let i2inv_block_t = rt::asarray((&i2inv_block_stage, [ni_aux, naux].f(), &device));
            let r01 = (&wj001_t_2d % &i2inv_block_t); // [3, naux]
            let r01_raw = r01.into_shape(-1).into_raw();
            for x in 0..3 { for q in 0..naux {
                r01_per[i0 * 3 * naux + x * naux + q] = r01_raw[x + q * 3];
            }}
        }

        // ── ip1_2c[x, p, r] = Σ_q i21[x, ap0+p, q] · i2inv[q, r]  (1 GEMM, main bottleneck) ──
        // staging i21_block_2d F-order [3*ni_aux, naux]: element (x*ni_aux+p, q) = i21[x, ap0+p, q]
        //   fill: idx = (x*ni_aux+p) + q*(3*ni_aux)   [F-order: i + j*M, M = 3*ni_aux]
        //   source: i21[x*naux² + (ap0+p)*naux + q]
        // i2inv_t: zero-copy F-order [naux, naux], element (q, r) = i2inv[q + r*naux] = i2inv[q*naux + r]...
        //   Wait: i2inv stored as i2inv[p + q*naux] (col-major [naux, naux]). F-order [naux, naux] flat: p + q*naux.
        //   So i2inv_t[q, r] = i2inv[q + r*naux] (F-order element (q, r) at q + r*naux). ✓
        // GEMM: ip1_2c = i21_block_2d @ i2inv_t → [3*ni_aux, naux]
        //   ip1_2c[x*ni_aux+p, r] = Σ_q i21[x, ap0+p, q] · i2inv_t[q, r] = Σ_q i21[x, ap0+p, q] · i2inv[q, r] ✓
        //   scatter: ip1_2c[x*ni_aux*naux + p*naux + r] = ip1_2c_raw[(x*ni_aux+p) + r*(3*ni_aux)]
        let mut ip1_2c = vec![0.0; 3 * ni_aux * naux];
        {
            let mut i21_stage = vec![0.0; 3 * ni_aux * naux];
            for x in 0..3 { for p in 0..ni_aux { for q in 0..naux {
                i21_stage[(x * ni_aux + p) + q * (3 * ni_aux)] =
                    i21[x * naux * naux + (ap0 + p) * naux + q];
            }}}
            let i21_t_2d = rt::asarray((&i21_stage, [3 * ni_aux, naux].f(), &device));
            let ip1_2c_2d = (&i21_t_2d % &i2inv_t); // [3*ni_aux, naux]
            let ip1_2c_raw = ip1_2c_2d.into_shape(-1).into_raw();
            for x in 0..3 { for p in 0..ni_aux { for r in 0..naux {
                ip1_2c[x * ni_aux * naux + p * naux + r] =
                    ip1_2c_raw[(x * ni_aux + p) + r * (3 * ni_aux)];
            }}}
        }

        // ── r10_per[i0, x, r] = Σ_p r0[ap0+p] · ip1_2c[x, p, r]  (1 GEMM) ──
        // staging r0_block_row F-order [1, ni_aux]: element (0, p) = r0[ap0+p] (zero-copy from &r0[ap0..])
        // ip1_2c_reshaped F-order [ni_aux, 3*naux]: element (p, x*naux+r) = ip1_2c[x*ni_aux*naux + p*naux + r]
        //   Need staging: source is x*ni_aux*naux + p*naux + r, target (F-order) is p + (x*naux+r)*ni_aux
        // GEMM: r10 = r0_block_row @ ip1_2c_reshaped → [1, 3*naux]
        //   r10[0, x*naux+r] = Σ_p r0[ap0+p] · ip1_2c[x, p, r] ✓
        //   scatter: r10_per[i0*3*naux + x*naux + r] = r10_raw[x*naux + r]
        {
            let r0_block_t = rt::asarray((&r0[ap0..], [1, ni_aux].f(), &device));
            let mut ip1_2c_res_stage = vec![0.0; ni_aux * 3 * naux];
            for x in 0..3 { for p in 0..ni_aux { for r in 0..naux {
                ip1_2c_res_stage[p + (x * naux + r) * ni_aux] =
                    ip1_2c[x * ni_aux * naux + p * naux + r];
            }}}
            let ip1_2c_res_t = rt::asarray((&ip1_2c_res_stage, [ni_aux, 3 * naux].f(), &device));
            let r10 = (&r0_block_t % &ip1_2c_res_t); // [1, 3*naux]
            let r10_raw = r10.into_shape(-1).into_raw();
            for x in 0..3 { for r in 0..naux {
                r10_per[i0 * 3 * naux + x * naux + r] = r10_raw[x * naux + r];
            }}
        }
    }
    // ── O1–O6: keep as for-loops (small: ≤ 9*ni_aux*nq per pair) ──
    for i0 in 0..natm { let (_, ap0, ni_aux) = aux_blk[i0];
        for j0 in 0..natm { let (_, aq0, nq) = aux_blk[j0];
            let mut o1 = vec![0.0; 9];
            for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                for p in 0..ni_aux { for q in 0..nq {
                    s += r0[ap0 + p] * i2ip[(x * 3 + y) * naux * naux + (ap0 + p) * naux + (aq0 + q)] * r0[aq0 + q];
                }} o1[x * 3 + y] = s * 0.5;
            }}
            let mut o2 = vec![0.0; 9];
            for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                for qp in 0..nq { let qg = aq0 + qp;
                    s += rs1_per[i0 * 3 * naux + x * naux + qg] * wj001[y * naux + qg];
                } o2[x * 3 + y] = s;
            }}
            let mut o3 = vec![0.0; 9];
            for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                for qp in 0..nq { let qg = aq0 + qp;
                    s += rs1_per[i0 * 3 * naux + x * naux + qg] * wj2[qg * 3 + y];
                } o3[x * 3 + y] = s * 0.5;
            }}
            let mut o4 = vec![0.0; 9];
            for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                for qp in 0..nq { let qg = aq0 + qp;
                    s += r01_per[i0 * 3 * naux + x * naux + qg] * wj001[y * naux + qg];
                } o4[x * 3 + y] = s * 0.5;
            }}
            let mut o5 = vec![0.0; 9];
            for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                for qp in 0..nq { let qg = aq0 + qp;
                    for paux in 0..naux {
                        s += i21[y * naux * naux + qg * naux + paux] * r0[qg] * rs1_per[i0 * 3 * naux + x * naux + paux];
                    }
                } o5[x * 3 + y] = s;
            }}
            let mut o6 = vec![0.0; 9];
            for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                for qp in 0..nq { let qg = aq0 + qp;
                    s += r10_per[i0 * 3 * naux + x * naux + qg] * wj001[y * naux + qg];
                } o6[x * 3 + y] = s;
            }}
            for x in 0..3 { for y in 0..3 {
                let v = o1[x * 3 + y] - o2[x * 3 + y] + o3[x * 3 + y] + o4[x * 3 + y] - o5[x * 3 + y] + o6[x * 3 + y];
                out[i_t(i0, j0, x, y)] += v;
                out[i_t(j0, i0, x, y)] += o1[y * 3 + x] - o2[y * 3 + x] + o3[y * 3 + x] + o4[y * 3 + x] - o5[y * 3 + x] + o6[y * 3 + x];
            }}
        }
    }
}

pub struct RIRHFHessian<'a> {
    pub scf_data: &'a SCF,
    pub flags: RIRHFHessianFlags,
    pub result: HashMap<String, MatrixFull<f64>>,
    /// h1ao[ia]: first-order Fock response for atom ia, shape [3*nao, nao].
    pub h1ao: Vec<MatrixFull<f64>>,
    pub timings: Vec<(&'static str, std::time::Duration)>,
    /// H2 optimization: integrals shared between `calc_ej_ek` and
    /// `calc_h1ao` (int2c2e_ip1, V^{-1}, int3c2e, int3c2e_ip1). Built once
    /// in `calc_ej_ek` and reused in `calc_h1ao` to avoid recomputing
    /// expensive libcint calls.
    pub shared_integrals: Option<SharedHessianIntegrals>,
}

/// Shared RI integrals between `calc_ej_ek` and `calc_h1ao`.
/// Layouts match what each consumer expects:
/// - `int2c2e_ip1`: raw `integrate_row_major` output for "int2c2e_ip1" with aux slice
/// - `vinv`: V^{-1} in column-major `[naux, naux]` (from `compute_vinv`)
/// - `int3c2e` / `int3c2e_ip1`: raw `integrate_row_major` output with 3c slice
pub struct SharedHessianIntegrals {
    pub int2c2e_ip1: Vec<f64>,
    pub vinv: Vec<f64>,
    pub int3c2e: Vec<f64>,
}

impl RIRHFHessian<'_> {
    pub fn new(scf_data: &SCF) -> RIRHFHessian<'_> {
        match scf_data.scftype { scf_io::SCFType::RHF => {},
            _ => panic!("SCF type is not suitable for RHF Hessian."),
        }
        // Auto-detect RKS: skip exchange for pure DFT, set factor_k=0
        let is_dft = !scf_data.mol.xc_data.dfa_compnt_scf.is_empty();
        let flags = if is_dft {
            RIRHFHessianFlagsBuilder::default()
                .with_k(false)
                .factor_k(Some(0.0))
                .build().unwrap()
        } else {
            RIRHFHessianFlagsBuilder::default().build().unwrap()
        };
        RIRHFHessian { scf_data, flags,
            result: HashMap::new(),
            h1ao: Vec::new(),
            timings: Vec::new(),
            shared_integrals: None,
        }
    }

    /// Check if this is an RKS (DFT) calculation: non-empty XC functional list.
    fn is_rks(&self) -> bool {
        !self.scf_data.mol.xc_data.dfa_compnt_scf.is_empty()
    }
    /// Check if the XC functional contains a hybrid (exact-exchange) component.
    fn _is_hybrid(&self) -> bool {
        // libxc_is_hybrid equivalent: check if any func_id is a hybrid
        // For now, rely on mol.xc_data.dfa_paramr_scf factor_k values
        false
    }

    pub fn print_timings(&self) {
        println!("\n  --- Hessian timing profile ---");
        // TOTAL should reflect wall-clock time of the top-level pipeline,
        // without double-counting nested (indented) sub-timings.
        //
        // Top-level independent stages (not indented, not children of others):
        //   calc_e1, calc_ej_ek, calc_h1ao, compute_hessian (sum)
        // compute_hessian internally calls calc_cphf_contrib + calc_hess_nuc,
        // so those two are its children (already counted in compute_hessian).
        let children: &[&str] = &[
            "calc_cphf_contrib", "calc_hess_nuc",
        ];
        let mut total = std::time::Duration::ZERO;
        for (name, dur) in &self.timings {
            let secs = dur.as_secs_f64();
            println!("  {:<30} {:8.3} s", name, secs);
            // Skip indented sub-timings (they start with ' ')
            // and skip known children of compute_hessian.
            if !name.starts_with(' ') && !children.contains(&&name[..]) {
                total += *dur;
            }
        }
        println!("  {:<30} {:8.3} s", "TOTAL", total.as_secs_f64());
    }

    pub fn calc_e1(&mut self) -> &mut Self {
        let _t = std::time::Instant::now();
        let scf = self.scf_data; let mol = &scf.mol;
        let dm0_mat = &scf.density_matrix[0]; let nao = mol.num_basis;
        let dm0: Vec<f64> = dm0_mat.iter().copied().collect();
        let c = &scf.eigenvectors[0]; let eps = &scf.eigenvalues[0];
        let nocc = (scf.homo[0] + 1) as usize;
        let mut dme0 = vec![0.0; nao * nao];
        for p in 0..nao { for q in 0..nao {
            let mut s = 0.0;
            for i in 0..nocc { s += c[[p,i]] * eps[i] * c[[q,i]]; }
            dme0[p * nao + q] = s * 2.0;
        }}
        self.result.insert("e1".to_string(), compute_e1(mol, &dm0, &dme0));
        self.timings.push(("calc_e1", _t.elapsed()));
        self
    }

    /// Compute ej = basic + vjd + vj1 + ri1 + ri2d + ri2o
    /// Ported from standalone prototype main.rs, using rest_libcint integrals.
    pub fn calc_ej_ek(&mut self) -> &mut Self {
        let _t_global = std::time::Instant::now();
        let scf = self.scf_data; let mol = &scf.mol;
        let nao = mol.num_basis; let natm = mol.geom.nfree;
        let aoslices = mol.aoslice_by_atom();
        // SCF data
        let dm0_mat = &scf.density_matrix[0];
        let dm0: Vec<f64> = dm0_mat.iter().copied().collect(); // col-major
        let c = &scf.eigenvectors[0]; let eps = &scf.eigenvalues[0];
        let mo_occ = &scf.occupation[0];
        let nocc = (scf.homo[0] + 1) as usize;
        // mocc_2 = C_occ · sqrt(occ)  (only occupied cols, weighted)
        let mut mc2 = vec![0.0; nao * nocc];
        for p in 0..nao { for i in 0..nocc {
            mc2[p * nocc + i] = c[[p, i]] * (mo_occ[i] as f64).sqrt();
        }}
        // dme0 not needed for ej/ek

        // ── Set up auxiliary basis ──
        // Use combined CInt (regular + aux) for both 2c and 3c integrals
        let cint_reg = mol.initialize_cint(false);
        let nreg = cint_reg.nbas(); // regular shells
        let cint_all = mol.initialize_cint(true); // reg + aux combined
        let naux_shell = cint_all.nbas() - nreg; // aux shells
        let auxmol = mol.make_auxmol_fake();
        let naux = auxmol.num_basis;
        let auxslices = auxmol.aoslice_by_atom();

        // Compute V = int2c2e (aux-only, via shell slice)
        let aux_slc = &[[nreg, nreg + naux_shell], [nreg, nreg + naux_shell]];
        let (int2c_v,_) = cint_all.integrate_row_major("int2c2e","s1",Some(&aux_slc[..])).into();
        let mut int2c = vec![0.0; naux * naux];
        if int2c_v.len() == naux * naux { int2c = int2c_v; }
        else { // triangular expansion
            let mut idx = 0;
            for j in 0..naux { for i in 0..=j {
                int2c[i + j * naux] = int2c_v[idx];
                int2c[j + i * naux] = int2c_v[idx]; idx += 1;
            }}
        }
        // Column-major V^{-1}
        let mut int2c_cm = vec![0.0; naux * naux];
        for p in 0..naux { for q in 0..naux { int2c_cm[p + q * naux] = int2c[p * naux + q]; }}
        let vinv = compute_vinv(&int2c_cm, naux);
        let i2inv = vinv.clone();

        // ── 2c-2e auxiliary basis integrals (via combined CInt with aux-only slice) ──
        let aux_slc_ref: &[[usize; 2]] = &[[nreg, nreg + naux_shell], [nreg, nreg + naux_shell]];
        // Use cint_all with shell slice for aux-only derivative integrals
        let (i21_v,_) = <(Vec<f64>,Vec<usize>)>::from(cint_all.integrate_row_major("int2c2e_ip1","s1",Some(aux_slc_ref)));
        let (i211_v,_) = <(Vec<f64>,Vec<usize>)>::from(cint_all.integrate_row_major("int2c2e_ipip1","s1",Some(aux_slc_ref)));
        let (i212_v,_) = cint_all.integrate_row_major("int2c2e_ip1ip2","s1",Some(aux_slc_ref)).into();
        let i21 = i21_v; let i211 = i211_v; let i212 = i212_v;

        // ── Per-atom block ranges ──
        let mut blk = Vec::with_capacity(natm);
        for ia in 0..natm {
            let p0 = aoslices[ia][2] as usize; let p1 = aoslices[ia][3] as usize;
            blk.push((ia, p0, p1 - p0));
        }
        let mut aux_blk = Vec::with_capacity(natm);
        for ia in 0..natm {
            let p0 = auxslices[ia][2] as usize; let p1 = auxslices[ia][3] as usize;
            aux_blk.push((ia, p0, p1 - p0));
        }

        let nao3 = nao * nao;
        let i9  = |c: usize, p: usize, q: usize| c * nao3 + p * nao + q;
        let i9p = |c: usize, p: usize, pp: usize| c * naux * naux + p * naux + pp;

        // ══════════════════════════════════════════════════════════
        // Compute all 3c-2e integrals via rest_libcint
        // ══════════════════════════════════════════════════════════
        // 3c-2e integrals via combined CInt (shell slice: reg, reg, aux)
        let slc_3c: &[[usize; 2]] = &[[0, nreg], [0, nreg], [nreg, nreg + naux_shell]];
        let int3c = |name: &str| -> Vec<f64> {
            let (v,_) = cint_all.integrate_row_major(name, "s1", Some(slc_3c)).into();
            v
        };
        // Phase 1-3 + G3 need: t3c, ip1, ip2, ipip1, ipv.
        // ip12 (g5/g8) and ipip2 (g6/g9) are only needed later — delay their
        // computation until just before their respective G-terms to avoid
        // holding 2×324 MiB of dead integrals through Phases 1-3.
        let t3c  = int3c("int3c2e");        // (N,N,P)
        let ip1  = int3c("int3c2e_ip1");    // (3,N,N,P)
        let ip2  = int3c("int3c2e_ip2");    // (3,N,N,P)
        let ipv  = int3c("int3c2e_ipvip1"); // (9,N,N,P)  — also used by G3
        let ipip1= int3c("int3c2e_ipip1");  // (9,N,N,P)
        // H2 optimization: cache the four integrals that `calc_h1ao` would
        // otherwise recompute. Clones are O(naux²) or O(naux·nao²), cheap
        // compared to the libcint evaluation it saves.
        self.shared_integrals = Some(SharedHessianIntegrals {
            int2c2e_ip1: i21.clone(),
            vinv: vinv.clone(),
            int3c2e: t3c.clone(),
        });
        self.timings.push(("  ej_ek: integrals", _t_global.elapsed()));
        let _tej = std::time::Instant::now();

        // ── rstsr device and tensor views (shared across all phases) ──
        let device = DeviceBLAS::default();
        // Column-major tensors (native F-order)
        let dm0_t: TsrView<f64> = rt::asarray((&dm0, [nao, nao].f(), &device));
        let mc2_t: TsrView<f64> = rt::asarray((&mc2, [nocc, nao].f(), &device));
        let vinv_t: TsrView<f64> = rt::asarray((&vinv, [naux, naux].f(), &device));
        // Row-major integrals → F-order by reversing dims (p fastest → first axis)
        let t3c_t: TsrView<f64> = rt::asarray((&t3c, [naux, nao, nao].f(), &device));
        let ip1_t: TsrView<f64>  = rt::asarray((&ip1, [naux, nao, nao, 3].f(), &device));
        let ip2_t: TsrView<f64>  = rt::asarray((&ip2, [naux, nao, nao, 3].f(), &device));
        let ipv_t: TsrView<f64>  = rt::asarray((&ipv, [naux, nao, nao, 9].f(), &device));
        let ipip1_t: TsrView<f64> = rt::asarray((&ipip1, [naux, nao, nao, 9].f(), &device));
        let i21_t: TsrView<f64>  = rt::asarray((&i21, [naux, naux, 3].f(), &device));
        let i2inv_t: TsrView<f64> = rt::asarray((&i2inv, [naux, naux].f(), &device));

        // ══ Phase 1a: rhoj0_P, rhok0_Pl_ (prototype L57-66) ══
        let _tp1a = std::time::Instant::now();
        // ── rhoj0: r0r[p] = Σ_{i,j} t3c[i, j, p] · dm0[i, j]  (single GEMM) ──
        // Per-atom loop in original code just accumulates into the same r0r across all AO blocks,
        // which (because AO blocks are disjoint and cover all AOs) equals the full sum.
        // staging t3c_pij_2d[naux, nao²] F-order: element (p, i*nao + j) = t3c[i, j, p]
        //   source: t3c[i*nao*naux + j*naux + p]
        //   fill: idx = p + (i*nao + j)*naux   [F-order: i + j*M, M = naux]
        // dm0_col[nao², 1] F-order: element (i*nao+j, 0) = dm0[i*nao+j]  (symmetric: = dm0_t[i, j])
        // GEMM: r0r_col[naux, 1] = t3c_pij_2d @ dm0_col
        //   result element (p, 0) = Σ_{i,j} t3c[i, j, p] · dm0[i, j] = r0r[p]  ✓
        let mut r0r = vec![0.0; naux];
        {
            let mut t3c_pij_stage = vec![0.0; naux * nao3];
            for p in 0..naux { for i in 0..nao { for j in 0..nao {
                t3c_pij_stage[p + (i * nao + j) * naux] = t3c[i * nao * naux + j * naux + p];
            }}}
            let t3c_pij_t = rt::asarray((&t3c_pij_stage, [naux, nao3].f(), &device));
            let dm0_col_t = rt::asarray((&dm0, [nao3, 1].f(), &device));
            let r0r_col = (&t3c_pij_t % &dm0_col_t); // [naux, 1]
            let r0r_raw = r0r_col.into_shape(-1).into_raw();
            r0r.copy_from_slice(&r0r_raw);
        }
        // ── rhok0: rkr[p, i, oc] = Σ_j t3c[i, j, p] · mc2[j, oc]  (single GEMM) ──
        // Per-atom loop writes disjoint i_ao ranges (p0+ii), covering all AOs after all atoms.
        // staging t3c_pi_j_2d[naux*nao, nao] F-order: element (p + i*naux, j) = t3c[i, j, p]
        //   source: t3c[i*nao*naux + j*naux + p]
        //   fill: idx = (p + i*naux) + j*(naux*nao)   [F-order: i + j*M, M = naux*nao]
        // mc2_t[oc, j] = mc2[j*nocc + oc]. GEMM: rkr_2d = t3c_pi_j_2d @ mc2_t^T
        //   result element (p + i*naux, oc) = Σ_j t3c[i, j, p] · mc2[j, oc] = rkr[p, i, oc]  ✓
        // rkr layout: rkr[p + i*naux + oc*naux*nao] (row-major [naux, nao, nocc])
        //   = rkr_2d_raw[(p + i*naux) + oc*(naux*nao)] (F-order flat of [naux*nao, nocc])  ✓
        let mut rkr = vec![0.0; naux * nao * nocc];
        {
            let mut t3c_pi_j_stage = vec![0.0; naux * nao * nao];
            for p in 0..naux { for i in 0..nao { for j in 0..nao {
                t3c_pi_j_stage[(p + i * naux) + j * (naux * nao)] = t3c[i * nao * naux + j * naux + p];
            }}}
            let t3c_pi_j_t = rt::asarray((&t3c_pi_j_stage, [naux * nao, nao].f(), &device));
            let rkr_2d = (&t3c_pi_j_t % &mc2_t.t()); // [naux*nao, nocc]
            let rkr_raw = rkr_2d.into_shape(-1).into_raw();
            rkr.copy_from_slice(&rkr_raw);
        }
        // ── Apply V⁻¹: r0 = V⁻¹ · r0r  (GEMM), rk = V⁻¹ · rkr  (GEMM) ──
        // r0_col[naux, 1] = vinv_t[naux, naux] @ r0r_col[naux, 1]
        // rk_2d[naux, nao*nocc] = vinv_t[naux, naux] @ rkr_t_2d[naux, nao*nocc]
        //   rkr viewed as F-order [naux, nao*nocc]: element (p, i + oc*nao) = rkr[p, i, oc]
        //   F-order flat: p + (i + oc*nao)*naux = p + i*naux + oc*nao*naux (matches rkr source)
        //   result element (p, i + oc*nao) = Σ_q vinv[p, q] · rkr[q, i, oc] = rk[p, i, oc]  ✓
        //   rk layout matches rkr (downstream code indexes rk[p + i*naux + oc*naux*nao])
        let mut r0 = vec![0.0; naux]; let mut rk = vec![0.0; naux * nao * nocc];
        {
            let r0r_col_t = rt::asarray((&r0r, [naux, 1].f(), &device));
            let r0_col = (&vinv_t % &r0r_col_t); // [naux, 1]
            let r0_raw = r0_col.into_shape(-1).into_raw();
            r0.copy_from_slice(&r0_raw);
        }
        {
            let rkr_t_2d = rt::asarray((&rkr, [naux, nao * nocc].f(), &device));
            let rk_2d = (&vinv_t % &rkr_t_2d); // [naux, nao*nocc]
            let rk_raw = rk_2d.into_shape(-1).into_raw();
            rk.copy_from_slice(&rk_raw);
        }

        // t3c (and its unused view t3c_t) is no longer needed after Phase 1a —
        // the cached clone in shared_integrals (for calc_h1ao) was taken above.
        // Free ~naux*nao² doubles before the rest of calc_ej_ek runs.
        drop(t3c_t);
        drop(t3c);

        self.timings.push(("  p1a_rhoj0_rhok0", _tp1a.elapsed()));
        // ══ Phase 1b: vj1_diag, vk1_diag (prototype L68-76) ══
        let _tp1b = std::time::Instant::now();
        let mut vjd = vec![0.0; 9 * nao * nao];
        let mut vkd = vec![0.0; 9 * nao * nao];
        // ── vjd[x, i, j] = Σ_p ipip1[x, i, j, p] · r0[p]  (single GEMM) ──
        // staging ipip1_2d[9*nao², naux] F-order: element (x*nao²+i*nao+j, p) = ipip1[x, i, j, p]
        //   fill: idx = (x*nao²+i*nao+j) + p*(9*nao²)   [F-order: i + j*M, M = 9*nao²]
        // GEMM: vjd_col[9*nao², 1] = ipip1_2d @ r0_col[naux, 1]
        //   result element (x*nao²+i*nao+j, 0) = Σ_p ipip1[x, i, j, p] · r0[p] = vjd[x, i, j]  ✓
        // scatter: vjd[x*nao² + i*nao + j] = vjd_col_raw[x*nao²+i*nao+j]
        {
            let mut ipip1_stage = vec![0.0; 9 * nao3 * naux];
            for x in 0..9 { for i in 0..nao { for j in 0..nao { for p in 0..naux {
                ipip1_stage[(x * nao3 + i * nao + j) + p * (9 * nao3)] =
                    ipip1[x * nao3 * naux + i * nao * naux + j * naux + p];
            }}}}
            let ipip1_t_2d = rt::asarray((&ipip1_stage, [9 * nao3, naux].f(), &device));
            let r0_col_t = rt::asarray((&r0, [naux, 1].f(), &device));
            let vjd_col = (&ipip1_t_2d % &r0_col_t); // [9*nao², 1]
            let vjd_col_raw = vjd_col.into_shape(-1).into_raw();
            vjd.copy_from_slice(&vjd_col_raw);
        }
        let mut rkm = vec![0.0; naux * nao * nao];
        // rkm[p,l,J] = Σ_j rk[p,l,j] * mc2[J,j]  (already BLAS, kept as-is)
        // rk: [naux, nao, nocc] = (p, l, j). Reshape to [naux*nao, nocc]
        // mc2: [nocc, nao] = (j, J). mc2.T: [nao, nocc]
        // rkm_2d = rk_2d @ mc2.T: [naux*nao, nocc] @ [nocc, nao] → [naux*nao, nao]
        {
            let rk_2d: TsrView<f64> = rt::asarray((&rk, [naux * nao, nocc].f(), &device));
            let rkm_2d = &rk_2d % &mc2_t; // [naux*nao, nocc] @ [nocc, nao] → [naux*nao, nao]
            let flat = rkm_2d.into_shape(-1).into_raw();
            // Map from flat[p*l+J*naux*nao] to rkm[p + l*naux + J*naux*nao]
            // flat is col-major [naux*nao, nao] = (combined_idx, J)
            // flat[p*l + J*(naux*nao)] → rkm[p + l*naux + J*naux*nao]
            // where combined_idx = p + l*naux = p*1 + l*naux (F-order within [naux, nao])
            // so flat[p + l*naux + J*naux*nao] = rkm[p + l*naux + J*naux*nao]
            // They match! ✓
            rkm.copy_from_slice(&flat);
        }
        // ── vkd[x, i, l] = Σ_{p,j} ipip1[x, i, j, p] · rkm[p, l, j]  (9 GEMMs, one per x) ──
        // staging ipip1_x_2d[nao, nao*naux] F-order: element (i, j*naux+p) = ipip1[x, i, j, p]
        //   fill: idx = i + (j*naux+p)*nao   [F-order: i + j*M, M = nao]
        // staging rkm_jpl_2d[nao*naux, nao] F-order: element (j*naux+p, l) = rkm[p, l, j]
        //   fill: idx = (j*naux+p) + l*(nao*naux)   [F-order: i + j*M, M = nao*naux]
        //   source: rkm[p + l*naux + j*naux*nao]
        // GEMM: vkd_x_2d[nao, nao] = ipip1_x_2d @ rkm_jpl_2d
        //   result element (i, l) = Σ_{j,p} ipip1[x, i, j, p] · rkm[p, l, j] = vkd[x, i, l]  ✓
        // scatter: vkd[x*nao² + i*nao + l] = vkd_x_raw[i + l*nao]  (F-order flat of [nao, nao])
        // rkm_jpl_2d is the SAME for all x, stage once outside x loop.
        let mut rkm_jpl_stage = vec![0.0; nao * naux * nao];
        for j in 0..nao { for p in 0..naux { for l in 0..nao {
            rkm_jpl_stage[(j * naux + p) + l * (nao * naux)] =
                rkm[p + l * naux + j * naux * nao];
        }}}
        let rkm_jpl_t = rt::asarray((&rkm_jpl_stage, [nao * naux, nao].f(), &device));
        for x in 0..9 {
            let mut ipip1_x_stage = vec![0.0; nao * nao * naux];
            for i in 0..nao { for j in 0..nao { for p in 0..naux {
                ipip1_x_stage[i + (j * naux + p) * nao] =
                    ipip1[x * nao3 * naux + i * nao * naux + j * naux + p];
            }}}
            let ipip1_x_t = rt::asarray((&ipip1_x_stage, [nao, nao * naux].f(), &device));
            let vkd_x = (&ipip1_x_t % &rkm_jpl_t); // [nao, nao]
            let vkd_x_raw = vkd_x.into_shape(-1).into_raw();
            for i in 0..nao { for l in 0..nao {
                vkd[x * nao3 + i * nao + l] = vkd_x_raw[i + l * nao];
            }}
        }

        // ipip1 (9*naux*nao²), rkm and rkm_jpl_stage (naux*nao² each), and the
        // unused ipip1_t view are only used in Phase 1b. Free before the
        // heavier Phase 2/3 allocations.
        drop(ipip1_t);
        drop(rkm_jpl_t);
        drop(ipip1);
        drop(rkm);
        drop(rkm_jpl_stage);

        self.timings.push(("  p1b_vjd_vkd", _tp1b.elapsed()));
        // ══ Phase 2: rhoj1, wj1, rho_ip1 (prototype L79-88) ══
        let _tp2 = std::time::Instant::now();
        let m3 = 3 * nao * nao;
        let mut ip1c = vec![0.0; m3 * naux];
        for x in 0..3 { for i in 0..nao { for j in 0..nao { for p in 0..naux {
            ip1c[(x * nao * nao + i * nao + j) + p * m3] = ip1[x * nao3 * naux + i * nao * naux + j * naux + p];
        }}}}
        let mut tmpf = vec![0.0; naux * m3];
        // tmpf = V⁻¹ @ ip1c^T:  tmpf[p, c] = Σ_q V⁻¹[p,q] * ip1c[c,q]
        // ip1c is col-major [m3, naux]: element (c, q) = data[c + q*m3]
        // Need ip1c^T as [naux, m3]: element (q, c) = data[c + q*m3]
        // ip1c_t.t() gives view [naux, m3]. Element (q, c) = data[c + q*m3] ✓
        {
            let ip1c_t: TsrView<f64> = rt::asarray((&ip1c, [m3, naux].f(), &device));
            let tmpf_t = &vinv_t % &ip1c_t.t(); // [naux, naux] % [naux, m3] → [naux, m3]
            let flat = tmpf_t.into_shape(-1).into_raw();
            tmpf.copy_from_slice(&flat);
        }
        // ip1c (3*naux*nao²) was only a staging buffer for the tmpf GEMM above.
        drop(ip1c);
        let mut rj1 = vec![0.0; natm * naux * 3]; let mut wj1 = vec![0.0; natm * naux * 3];
        // ── rj1/wj1: for each atom, contract tmpf (rj1) or ip1 (wj1) with dm0 block over (i, j) ──
        // For each atom ib, for each x:
        //   rj1[ib, p, x] = Σ_{ii, j} tmpf[p, x*nao² + (p0+ii)*nao + j] · dm0[(p0+ii)*nao + j]
        //   wj1[ib, p, x] = Σ_{ii, j} ip1[x, p0+ii, j, p] · dm0[(p0+ii)*nao + j]
        // staging dm0_atom_col[ni*nao, 1] F-order: element (ii*nao+j, 0) = dm0[(p0+ii)*nao+j]
        // staging tmpf_x_atom_2d[naux, ni*nao] F-order per (ib, x): element (p, ii*nao+j) = tmpf[p + (x*nao²+(p0+ii)*nao+j)*naux]
        //   fill: idx = p + (ii*nao+j)*naux   [F-order: i + j*M, M = naux]
        // staging ip1_x_atom_2d[naux, ni*nao] F-order per (ib, x): element (p, ii*nao+j) = ip1[x*nao²*naux+(p0+ii)*nao*naux+j*naux+p]
        //   fill: idx = p + (ii*nao+j)*naux
        // GEMM: rj1_x_col[naux, 1] = tmpf_x_atom_2d @ dm0_atom_col
        // GEMM: wj1_x_col[naux, 1] = ip1_x_atom_2d @ dm0_atom_col
        // scatter: rj1[ib*naux*3 + p*3 + x] = rj1_x_col_raw[p];  wj1 similar
        for (ib, &(_, p0, ni)) in blk.iter().enumerate() {
            let ninao = ni * nao;
            let mut dm0_atom_col = vec![0.0; ninao];
            for ii in 0..ni { for j in 0..nao {
                dm0_atom_col[ii * nao + j] = dm0[(p0 + ii) * nao + j];
            }}
            let dm0_atom_col_t = rt::asarray((&dm0_atom_col, [ninao, 1].f(), &device));
            for x in 0..3 {
                // rj1
                let mut tmpf_x_stage = vec![0.0; naux * ninao];
                for p in 0..naux { for ii in 0..ni { for j in 0..nao {
                    tmpf_x_stage[p + (ii * nao + j) * naux] =
                        tmpf[p + (x * nao3 + (p0 + ii) * nao + j) * naux];
                }}}
                let tmpf_x_t = rt::asarray((&tmpf_x_stage, [naux, ninao].f(), &device));
                let rj1_x_col = (&tmpf_x_t % &dm0_atom_col_t); // [naux, 1]
                let rj1_x_raw = rj1_x_col.into_shape(-1).into_raw();
                // wj1
                let mut ip1_x_stage = vec![0.0; naux * ninao];
                for p in 0..naux { for ii in 0..ni { for j in 0..nao {
                    ip1_x_stage[p + (ii * nao + j) * naux] =
                        ip1[x * nao3 * naux + (p0 + ii) * nao * naux + j * naux + p];
                }}}
                let ip1_x_t = rt::asarray((&ip1_x_stage, [naux, ninao].f(), &device));
                let wj1_x_col = (&ip1_x_t % &dm0_atom_col_t); // [naux, 1]
                let wj1_x_raw = wj1_x_col.into_shape(-1).into_raw();
                for p in 0..naux {
                    rj1[ib * naux * 3 + p * 3 + x] = rj1_x_raw[p];
                    wj1[ib * naux * 3 + p * 3 + x] = wj1_x_raw[p];
                }
            }
        }

        self.timings.push(("  p2_rhoj1_wj1", _tp2.elapsed()));
        // ══ Phase 3a: vk2buf + rhok_IkP (prototype L168-183, from Python) ══
        let _tp3a = std::time::Instant::now();
        // rhok_ip1_IkP = einsum('pykl,li->ikpy', tmp_ip1, dm0)
        // → result[i(ALL_AO), k(BLOCK), p(AUX), y(DERIV)] of shape (N, ni, P, 3)
        // Store per-atom as (N * ni * P * 3) flat
        let mut rhok_IkP_per_atom: Vec<Vec<f64>> = (0..natm).map(|_| vec![0.0; nao * nao * naux * 3]).collect();
        let mut rhok_PkI_full = vec![0.0; naux * nao * nao * 3];
        // ── Per-atom: rhok[i, k, p, y] = Σ_l tmpf[p, y*nao² + (p0+k)*nao + l] · dm0[l, i]  (1 GEMM per atom) ──
        // staging tmp_ikp_2d[nao, ni*naux*3] F-order: element (l, k + p*ni + y*ni*naux) = tmpf[p, c]
        //   where c = y*nao² + (p0+k)*nao + l
        //   source flat: tmpf[p + c*naux]
        //   fill: idx = l + (k + p*ni + y*ni*naux)*nao   [F-order: i + j*M, M = nao]
        // GEMM: rhok_2d[ni*naux*3, nao] = tmp_ikp_2d^T @ dm0_t   (tmp_ikp_2d^T[l, (k,p,y)] = tmp_ikp_2d[(k,p,y), l]... wait)
        //   Actually tmp_ikp_t^T has shape [ni*naux*3, nao], element ((k,p,y), l) = tmp_ikp_t[l, (k,p,y)]
        //   GEMM: rhok_2d[(k,p,y), i] = Σ_l tmp_ikp_t^T[(k,p,y), l] · dm0_t[l, i]
        //                             = Σ_l tmp_ikp_t[l, (k,p,y)] · dm0_t[l, i] = rhok[i, k, p, y]  ✓
        // scatter to rhok_IkP_per_atom (row-major [nao, ni, naux, 3]) and rhok_PkI_full (row-major [naux, nao, nao, 3])
        for (ib, &(_, p0, ni)) in blk.iter().enumerate() {
            let mut tmp_ikp_stage = vec![0.0; nao * ni * naux * 3];
            for l in 0..nao { for k in 0..ni { for p in 0..naux { for y in 0..3 {
                let c = y * nao3 + (p0 + k) * nao + l;
                tmp_ikp_stage[l + (k + p * ni + y * ni * naux) * nao] = tmpf[p + c * naux];
            }}}}
            let tmp_ikp_t = rt::asarray((&tmp_ikp_stage, [nao, ni * naux * 3].f(), &device));
            let rhok_2d = (&tmp_ikp_t.t() % &dm0_t); // [ni*naux*3, nao]
            let rhok_2d_raw = rhok_2d.into_shape(-1).into_raw();
            let rhok = &mut rhok_IkP_per_atom[ib];
            for i in 0..nao { for k in 0..ni { for p in 0..naux { for y in 0..3 {
                let val = rhok_2d_raw[(k + p * ni + y * ni * naux) + i * (ni * naux * 3)];
                rhok[i * ni * naux * 3 + k * naux * 3 + p * 3 + y] = val;
                // rhok_PkI[p, k_abs, i, y] = rhok[i, k, p, y] where k_abs = p0 + k
                rhok_PkI_full[p * nao * nao * 3 + (p0 + k) * nao * 3 + i * 3 + y] = val;
            }}}}
        }
        // ── vk2buf[x, y, k, i] = Σ_{p,j} ip1[x, i, j, p] · rhok_PkI[p, k, j, y]  (9 GEMMs) ──
        // staging ip1_x_2d[nao, nao*naux] F-order per x: element (i, j*naux+p) = ip1[x, i, j, p]
        //   fill: idx = i + (j*naux+p)*nao   [F-order: i + j*M, M = nao]
        // staging rhok_PkI_y_2d[nao*naux, nao] F-order per y: element (j*naux+p, k) = rhok_PkI[p, k, j, y]
        //   fill: idx = (j*naux+p) + k*(nao*naux)
        //   source: rhok_PkI_full[p*nao*nao*3 + k*nao*3 + j*3 + y]
        // GEMM per (x,y): vk2buf_xy[nao, nao] = ip1_x_2d @ rhok_PkI_y_2d
        //   result element (i, k) = Σ_{j,p} ip1[x, i, j, p] · rhok_PkI[p, k, j, y] = vk2buf[x, y, k, i]  ✓
        // scatter: vk2buf[(x*3+y)*nao² + k*nao + i] = vk2buf_xy_raw[i + k*nao]  (F-order flat of [nao, nao])
        let mut vk2buf = vec![0.0; 9 * nao * nao];
        // Pre-stage ip1_x_2d for all 3 x (reused across y)
        let mut ip1_x_stages: Vec<Vec<f64>> = Vec::with_capacity(3);
        for x in 0..3 {
            let mut s = vec![0.0; nao * nao * naux];
            for i in 0..nao { for j in 0..nao { for p in 0..naux {
                s[i + (j * naux + p) * nao] = ip1[x * nao3 * naux + i * nao * naux + j * naux + p];
            }}}
            ip1_x_stages.push(s);
        }
        for y in 0..3 {
            let mut rhok_y_stage = vec![0.0; nao * naux * nao];
            for j in 0..nao { for p in 0..naux { for k in 0..nao {
                rhok_y_stage[(j * naux + p) + k * (nao * naux)] =
                    rhok_PkI_full[p * nao * nao * 3 + k * nao * 3 + j * 3 + y];
            }}}
            let rhok_y_t = rt::asarray((&rhok_y_stage, [nao * naux, nao].f(), &device));
            for x in 0..3 {
                let ip1_x_t = rt::asarray((&ip1_x_stages[x], [nao, nao * naux].f(), &device));
                let vk2buf_xy = (&ip1_x_t % &rhok_y_t); // [nao, nao]
                let vk2buf_xy_raw = vk2buf_xy.into_shape(-1).into_raw();
                for k in 0..nao { for i in 0..nao {
                    vk2buf[(x * 3 + y) * nao3 + k * nao + i] = vk2buf_xy_raw[i + k * nao];
                }}
            }
        }

        // rhok_PkI_full (3*naux*nao²) and ip1_x_stages (3*naux*nao²) are only
        // used to build vk2buf in Phase 3a. Free before Phase 3b allocations.
        drop(rhok_PkI_full);
        drop(ip1_x_stages);

        self.timings.push(("  p3a_vk2buf_rhok", _tp3a.elapsed()));
        // ══ Phase 3b: wj_ip2, wk_ip2_Ipk, wk_ip2_P__ (prototype L92-97) ══
        let _tp3b = std::time::Instant::now();
        let mut wj2 = vec![0.0; naux * 3];
        let mut wki = vec![0.0; nao * naux * 3 * nao];
        let mut wk2 = vec![0.0; naux * 3 * nocc * nocc];

        // ── wj2[p, y] = Σ_{k,l} ip2[y, k, l, p] · dm0[k, l]  (3 GEMMs) ──
        // dm0 is col-major; since dm0 is symmetric, dm0[k*nao+l] (row-major access in original
        // for-loop) = dm0[k + l*nao] (col-major flat) = dm0_t[k, l]. So we can use dm0[0..nao²]
        // as a flat vector with k*nao+l indexing and get the same result.
        // staging ip2_y_2d[naux, nao²] F-order: element (p, k*nao+l) = ip2[y, k, l, p]
        //   fill: idx = p + (k*nao+l)*naux   [F-order: i + j*M, M = naux]
        // GEMM: wj2_y[naux, 1] = ip2_y_2d[naux, nao²] @ dm0_col[nao², 1]
        {
            let dm0_col_t = rt::asarray((&dm0, [nao3, 1].f(), &device));
            for y in 0..3 {
                let mut ip2_y_stage = vec![0.0; naux * nao3];
                for p in 0..naux { for k in 0..nao { for l in 0..nao {
                    ip2_y_stage[p + (k * nao + l) * naux] =
                        ip2[y * nao3 * naux + k * nao * naux + l * naux + p];
                }}}
                let ip2_y_t = rt::asarray((&ip2_y_stage, [naux, nao3].f(), &device));
                let wj2_y = (&ip2_y_t % &dm0_col_t); // [naux, 1]
                let wj2_y_raw = wj2_y.into_shape(-1).into_raw();
                for p in 0..naux { wj2[p * 3 + y] = wj2_y_raw[p]; }
            }
        }

        // ── wki[i, p, y, k] = Σ_l ip2[y, k, l, p] · dm0[l, i]  (3 GEMMs, one per y) ──
        // staging ip2_y_klp_2d[nao*naux, nao] F-order: element (k*naux+p, l) = ip2[y, k, l, p]
        //   fill: idx = (k*naux+p) + l*(nao*naux)   [F-order: i + j*M, M = nao*naux]
        // dm0_t[i, l] = dm0[i + l*nao] (col-major) = dm0[l*nao + i] (symmetric)
        // GEMM: wki_y_2d = ip2_y_klp_2d @ dm0_t^T   (dm0_t^T[l, i] = dm0_t[i, l])
        //   result element (k*naux+p, i) = Σ_l ip2[y, k, l, p] · dm0_t[i, l] = wki[i, p, y, k]  ✓
        // scatter: wki[i*naux*3*nao + p*3*nao + y*nao + k] = wki_y_raw[(k*naux+p) + i*(nao*naux)]
        for y in 0..3 {
            let mut ip2_y_klp_stage = vec![0.0; nao * naux * nao];
            for k in 0..nao { for p in 0..naux { for l in 0..nao {
                ip2_y_klp_stage[(k * naux + p) + l * (nao * naux)] =
                    ip2[y * nao3 * naux + k * nao * naux + l * naux + p];
            }}}
            let ip2_y_klp_t = rt::asarray((&ip2_y_klp_stage, [nao * naux, nao].f(), &device));
            let wki_y = (&ip2_y_klp_t % &dm0_t.t()); // [nao*naux, nao]
            let wki_y_raw = wki_y.into_shape(-1).into_raw();
            for i in 0..nao { for p in 0..naux { for k in 0..nao {
                wki[i * naux * 3 * nao + p * 3 * nao + y * nao + k] =
                    wki_y_raw[(k * naux + p) + i * (nao * naux)];
            }}}
        }

        // ── wk2[p, x, i, j] = Σ_{u,v} ip2[x, u, v, p] · mc2[u, i] · mc2[v, j]  (6 GEMMs: 2 per x) ──
        // Step 1 (per x): tmp1[v, p, i] = Σ_u ip2[x, u, v, p] · mc2[u, i]
        //   staging ip2_x_uvp_2d[naux*nao, nao] F-order: element (v*naux+p, u) = ip2[x, u, v, p]
        //     fill: idx = (v*naux+p) + u*(naux*nao)   [F-order: i + j*M, M = naux*nao]
        //   mc2_t[i, u] = mc2[u*nocc + i]. GEMM: tmp1 = ip2_x_uvp_2d @ mc2_t^T
        //     result element (v*naux+p, i) = Σ_u ip2[x, u, v, p] · mc2_t[i, u] = tmp1[v, p, i]  ✓
        // Step 2 (per x): wk2[p, x, i, j] = Σ_v tmp1[v, p, i] · mc2[v, j]
        //   staging tmp1_v_pi_2d[nao, naux*nocc] F-order: element (v, p*nocc+i) = tmp1[v, p, i]
        //     fill: idx = v + (p*nocc+i)*nao   [F-order: i + j*M, M = nao]
        //     source: tmp1_raw[(v*naux+p) + i*(naux*nao)]  (Step 1 flat, F-order [naux*nao, nocc])
        //   mc2_t[j, v] = mc2[v*nocc + j]. GEMM: wk2_x = mc2_t @ tmp1_v_pi_2d
        //     result element (j, p*nocc+i) = Σ_v mc2_t[j, v] · tmp1_v_pi[v, p*nocc+i]
        //                                    = Σ_v mc2[v, j] · tmp1[v, p, i] = wk2[p, x, i, j]  ✓
        //   scatter: wk2[p*3*nocc² + x*nocc² + i*nocc + j] = wk2_x_raw[j + (p*nocc+i)*nocc]
        for x in 0..3 {
            let mut ip2_x_uvp_stage = vec![0.0; naux * nao * nao];
            for v in 0..nao { for p in 0..naux { for u in 0..nao {
                ip2_x_uvp_stage[(v * naux + p) + u * (naux * nao)] =
                    ip2[x * nao3 * naux + u * nao * naux + v * naux + p];
            }}}
            let ip2_x_uvp_t = rt::asarray((&ip2_x_uvp_stage, [naux * nao, nao].f(), &device));
            let tmp1 = (&ip2_x_uvp_t % &mc2_t.t()); // [naux*nao, nocc]
            let tmp1_raw = tmp1.into_shape(-1).into_raw();
            let mut tmp1_v_pi_stage = vec![0.0; nao * naux * nocc];
            for v in 0..nao { for p in 0..naux { for i in 0..nocc {
                tmp1_v_pi_stage[v + (p * nocc + i) * nao] =
                    tmp1_raw[(v * naux + p) + i * (naux * nao)];
            }}}
            let tmp1_v_pi_t = rt::asarray((&tmp1_v_pi_stage, [nao, naux * nocc].f(), &device));
            let wk2_x = (&mc2_t % &tmp1_v_pi_t); // [nocc, naux*nocc]
            let wk2_x_raw = wk2_x.into_shape(-1).into_raw();
            for p in 0..naux { for i in 0..nocc { for j in 0..nocc {
                wk2[p * 3 * nocc * nocc + x * nocc * nocc + i * nocc + j] =
                    wk2_x_raw[j + (p * nocc + i) * nocc];
            }}}
        }

        // ip2 (3*naux*nao²) and its unused view ip2_t are only used in Phase 3b.
        drop(ip2_t);
        drop(ip2);

        self.timings.push(("  p3b_wj2_wk2", _tp3b.elapsed()));
        // ══ Phase 3c: rhok0_P__, rho2c_0, int2c_ip_ip (prototype L98-106) ══
        let _tp3c = std::time::Instant::now();
        let mut rkoo = vec![0.0; naux * nocc * nocc];
        for p in 0..naux { for i in 0..nocc { for jj in 0..nocc {
            let mut s = 0.0; for l in 0..nao { s += rk[p + l * naux + i * naux * nao] * mc2[l * nocc + jj]; }
            rkoo[p * nocc * nocc + i * nocc + jj] = s;
        }}}
        let mut r2c0 = vec![0.0; naux * naux];
        for p in 0..naux { for q in 0..naux {
            let mut s = 0.0; for i in 0..nocc { for j in 0..nocc { s += rkoo[p * nocc * nocc + i * nocc + j] * rkoo[q * nocc * nocc + j * nocc + i]; }}
            r2c0[p + q * naux] = s;
        }}
        // Phase 3c: i2ip via rstsr matmul (P1 priority)
        // i2ip[xy,p,s] = (i21[x] @ i2inv @ i21[y]^T)[p,s] - i212[xy,p,s]
        // Step 1: reshape i21 to (3*naux, naux), matmul with i2inv -> (3*naux, naux)
        let mut i2ip = vec![0.0; 9 * naux * naux];
        {
            let mut i21_mm = vec![0.0; 3 * naux * naux];
            for x in 0..3 { for p in 0..naux { for q in 0..naux {
                i21_mm[(x * naux + p) + q * (3 * naux)] = i21[x * naux * naux + p * naux + q];
            }}}
            let i21_blas = rt::asarray((&i21_mm, [3 * naux, naux].f(), &device));
            let tmp = &i21_blas % &i2inv_t; // tmp[x*naux+p, r]
            // Step 2: result = tmp @ i21^T as (3*naux, 3*naux)
            let mut i21_mm_t = vec![0.0; naux * 3 * naux];
            for x in 0..3 { for q in 0..naux { for p in 0..naux {
                i21_mm_t[q + (x * naux + p) * naux] = i21[x * naux * naux + p * naux + q];
            }}}
            let i21_blas_t = rt::asarray((&i21_mm_t, [naux, 3 * naux].f(), &device));
            let i2ip_full = &tmp % &i21_blas_t;
            let i2ip_flat = i2ip_full.into_shape(-1).into_raw();
            for x in 0..3 { for y in 0..3 {
                let xy = x * 3 + y;
                for p in 0..naux { for s in 0..naux {
                    let val = i2ip_flat[(x * naux + p) + (y * naux + s) * (3 * naux)];
                    i2ip[xy * naux * naux + p * naux + s] = val - i212[xy * naux * naux + p * naux + s];
                }}
            }}
        }
        // i212 (9*naux²) is only consumed inside the i2ip block above.
        drop(i212);
        let mut wj001 = vec![0.0; 3 * naux];
        for y in 0..3 { for p in 0..naux {
            let mut s = 0.0; for q in 0..naux { s += i21[y * naux * naux + p * naux + q] * r0[q]; }
            wj001[y * naux + p] = s;
        }}

        // ══════════════════════════════════════════════════════════
        self.timings.push(("  p3c_rkoo_r2c0_i2ip", _tp3c.elapsed()));
        self.timings.push(("  phases_1-3", _tej.elapsed()));

        // ── Baseline save/load for Blas verification ──────────────
        // Determine system tag from (nao, naux, natm) for baseline filename.
        // Used to load the saved Inline output so each Blas arm can diff against it.
        // Generate system tag from dimensions so baseline is unique per (molecule, basis)
        // without hardcoding any specific molecule.
        let sys_tag = format!("nao{}_naux{}_natm{}", nao, naux, natm);
        let baseline_dir = std::path::Path::new("target");
        let baseline_path = baseline_dir.join(format!("ej_ek_baseline_{}.json", sys_tag));
        let opt = &self.flags.ej_ek_opt;
        let any_inline = opt.g4_ek_vk1 == TermPath::Inline
            || opt.g5_ek_ri1 == TermPath::Inline
            || opt.g6_ek_ri2d == TermPath::Inline
            || opt.g7_ek_ri2o == TermPath::Inline
            || opt.g8_ej_ri1 == TermPath::Inline
            || opt.g9_ej_ri2d == TermPath::Inline
            || opt.g10_ej_ri2o == TermPath::Inline;
        // Load baseline when any G-term is Inline (verification mode).
        // The Inline arm compares its output against the stored BLAS baseline.
        let baseline: Option<EjEkBaseline> = if any_inline {
            match EjEkBaseline::load(&baseline_path) {
                Ok(b) => {
                    assert_eq!(b.nao, nao, "Baseline nao mismatch for {}", sys_tag);
                    assert_eq!(b.naux, naux, "Baseline nao mismatch for {}", sys_tag);
                    assert_eq!(b.nocc, nocc, "Baseline nocc mismatch for {}", sys_tag);
                    assert_eq!(b.natm, natm, "Baseline natm mismatch for {}", sys_tag);
                    if b.system != sys_tag {
                        eprintln!(
                            "Warning: baseline system tag '{}' differs from runtime '{}'",
                            b.system, sys_tag
                        );
                    }
                    Some(b)
                }
                Err(e) => {
                    eprintln!(
                        "Warning: BLAS baseline not found at {:?}: {}.\n\
                         Running Inline (verify) paths without verification.\n\
                         Run once with all Blas G-terms enabled to generate baseline.",
                        baseline_path, e
                    );
                    None
                }
            }
        } else {
            None
        };

        // Total AO+aux shells for per-atom libcint shell-slice calls.
        let naux_nbas = nreg + naux_shell;
        // Macro to build an EjEkContext borrowing all Phase 1-3 locals in scope.
        // Defined inside calc_ej_ek so the borrowed identifiers resolve to the
        // function's locals (Rust macro hygiene requires this for field shorthand).
        macro_rules! build_ej_ek_ctx {
            () => { EjEkContext {
                nao, naux, nocc, natm, nao3,
                blk: &blk, aux_blk: &aux_blk,
                dm0: &dm0, mc2: &mc2,
                i2inv: &i2inv, i21: &i21, i211: &i211, i2ip: &i2ip,
                r0: &r0, rk: &rk, rj1: &rj1, wj1: &wj1,
                vjd: &vjd, vkd: &vkd,
                rhok_IkP_per_atom: &rhok_IkP_per_atom,
                vk2buf: &vk2buf,
                wj2: &wj2, wki: &wki, wk2: &wk2,
                rkoo: &rkoo, r2c0: &r2c0, wj001: &wj001, tmpf: &tmpf,
                device: DeviceBLAS::default(),
                cint_all: &cint_all, nreg, aux_nbas: naux_nbas,
                aoslices: &aoslices, auxslices: &auxslices,
            } };
        }

        // Phase 4: Contribution arrays
        // ══════════════════════════════════════════════════════════
        let _tp4 = std::time::Instant::now();
        let aa9 = natm * natm * 9;
        let mut ej_basic = vec![0.0; aa9]; let mut ej_vjd = vec![0.0; aa9];
        let mut ej_vj1 = vec![0.0; aa9]; let mut ej_ri1 = vec![0.0; aa9];
        let mut ej_ri2d = vec![0.0; aa9]; let mut ej_ri2o = vec![0.0; aa9];
        let mut ek_vkd = vec![0.0; aa9]; let mut ek_vk1 = vec![0.0; aa9];
        let mut ek_ri1 = vec![0.0; aa9]; let mut ek_ri2d = vec![0.0; aa9];
        let mut ek_ri2o = vec![0.0; aa9];
        let i_t = |i0, j0, x, y| i0 * natm * 9 + j0 * 9 + x * 3 + y;

        // ══ rstsr accelerated G1-G3 (standard path, replaces for loops) ══
        let device = DeviceBLAS::default();
        let dm0_t: TsrView<f64> = rt::asarray((&dm0, [nao, nao].f(), &device));
        let vjd_t: TsrView<f64> = rt::asarray((&vjd, [nao, nao, 9].f(), &device));
        let vkd_t: TsrView<f64> = rt::asarray((&vkd, [nao, nao, 9].f(), &device));
        let r0_t: TsrView<f64> = rt::asarray((&r0, [naux].f(), &device));
        let ipv_t: TsrView<f64> = rt::asarray((&ipv, [naux, nao, nao, 9].f(), &device));
        // --- G1: ej_basic via matmul ---
        let n3 = natm * 3;
        let mut rj1_mm = vec![0.0; n3 * naux];
        let mut wj1_mm = vec![0.0; n3 * naux];
        for a in 0..natm { for x in 0..3 { let ax = a * 3 + x;
            for p in 0..naux {
                rj1_mm[ax + p * n3] = rj1[a * naux * 3 + p * 3 + x];
                wj1_mm[ax + p * n3] = wj1[a * naux * 3 + p * 3 + x];
            }
        }}
        let rj1_t = rt::asarray((&rj1_mm, [n3, naux].f(), &device));
        let wj1_t = rt::asarray((&wj1_mm, [n3, naux].f(), &device));
        let ej_rstsr = (&rj1_t % &wj1_t.t()) * 4.0;
        for a in 0..natm { for b in 0..natm { for x in 0..3 { for y in 0..3 {
            ej_basic[i_t(a, b, x, y)] = ej_rstsr[[a * 3 + x, b * 3 + y]];
        }}}}
        // --- G2: ej_vjd, ek_vkd via broadcast+sum ---
        for (i0, &(_, p0, ni)) in blk.iter().enumerate() {
            let vjd_block = vjd_t.i((.., p0..p0+ni, ..));
            let vkd_block = vkd_t.i((.., p0..p0+ni, ..));
            let dm0_block = dm0_t.i((.., p0..p0+ni));
            let prod_j = &vjd_block * dm0_block.i((.., .., None));
            let prod_k = &vkd_block * dm0_block.i((.., .., None));
            let sum_j_v = prod_j.sum_axes(&[0, 1]).into_shape(-1).into_raw();
            let sum_k_v = prod_k.sum_axes(&[0, 1]).into_shape(-1).into_raw();
            for c in 0..9 { let (x, y) = (c / 3, c % 3);
                ej_vjd[i_t(i0, i0, x, y)] = sum_j_v[c] * 2.0;
                ek_vkd[i_t(i0, i0, x, y)] = sum_k_v[c];
            }
        }
        // --- G3: ej_vj1 via broadcast+sum ---
        for (i0, &(ib, p0, ni)) in blk.iter().enumerate() {
            let ipv_block = ipv_t.i((.., .., p0..p0+ni, ..));
            let vj1_mat = (&ipv_block * r0_t.i((.., None, None, None))).sum_axes(&[0]);
            for (j0, &(jb, q0, qj)) in blk.iter().enumerate().take(i0 + 1) {
                let vj1_sub = vj1_mat.i((q0..q0+qj, .., ..));
                let dm0_sub = dm0_t.i((q0..q0+qj, p0..p0+ni));
                let prod = &vj1_sub * dm0_sub.i((.., .., None));
                let svec = prod.sum_axes(&[0, 1]).into_shape(-1).into_raw();
                for c in 0..9 { let (x1, x2) = (c / 3, c % 3);
                    ej_vj1[i_t(i0, j0, x1, x2)] = svec[c] * 2.0;
                }
            }
        }
        // Free rstsr views before for-loop sections 4-10
        drop((dm0_t, vjd_t, vkd_t, r0_t, ipv_t, rj1_t, wj1_t, ej_rstsr));

        // ── rstsr views for RI terms (G4-G10), reuses device, i21_t, i2inv_t from Phase 1-3 ──
        let wk2_t: TsrView<f64> = rt::asarray((&wk2, [nocc, nocc, 3, naux].f(), &device));
        let rkoo_t: TsrView<f64> = rt::asarray((&rkoo, [nocc, nocc, naux].f(), &device));
        let r2c0_t: TsrView<f64> = rt::asarray((&r2c0, [naux, naux].f(), &device));
        let i2ip_t: TsrView<f64> = rt::asarray((&i2ip, [naux, naux, 9].f(), &device));
        let i211_t: TsrView<f64> = rt::asarray((&i211, [naux, naux, 9].f(), &device));
        // NOTE: ipip2_t and ip12_t views were removed — the underlying Vecs
        // are now computed lazily (after g4) and dropped after their last
        // G-term use (g8 for ip12, g9 for ipip2).
        let wj2_t: TsrView<f64> = rt::asarray((&wj2, [3, naux].f(), &device));
        let wj001_t: TsrView<f64> = rt::asarray((&wj001, [naux, 3].f(), &device));
        let r0_t_ri: TsrView<f64> = rt::asarray((&r0, [naux].f(), &device));
        let rj1_t_ri: TsrView<f64> = rt::asarray((&rj1, [3, naux, natm].f(), &device));

        match self.flags.ej_ek_opt.g4_ek_vk1 {
            TermPath::Inline => {
                let _t_g4_ek_vk1 = std::time::Instant::now();
        // 4. ek_vk1 from vk2buf + ipvip1 (prototype L253-268)
        // vk1 from ipvip1: einsum('pki,ji->pkj', rk_block, mc2) then einsum('xijp,pki->xjk')
        for i0 in 0..natm { let (ib, p0, ni) = blk[i0];
            // tmp = einsum('pki,ji->pkj', rhok0_Pl_, mocc_2[p0:p1])
            // → tmp[p, k(ALL_AO), j(block)] = Σ_i rk[p, k, i] * mc2[(p0+j), i]
            let mut tmp = vec![0.0; naux * nao * ni];
            for p in 0..naux { for k in 0..nao { for jj in 0..ni {
                let mut s = 0.0; for i_occ in 0..nocc {
                    s += rk[p + k * naux + i_occ * naux * nao] * mc2[(p0 + jj) * nocc + i_occ];
                } tmp[p * nao * ni + k * ni + jj] = s;
            }}}
            for j0 in 0..=i0 { let (_, q0, qj) = blk[j0];
                // ek_vk1 = (ip1·rhok + ipv·tmp + vk2buf) · dm0
                for x in 0..3 { for y in 0..3 {
                    let c = x*3+y;
                    let mut part1 = 0.0; let mut part2 = 0.0; let mut part3 = 0.0;
                    // Part 1: ip1·rhok_ikp·dm0
                    // rhok_ikp[i_bra, k(ket_block), p, y] = rhok_IkP_per_atom[atom_of_k][(p0+i_bra)][k_local][p][y]
                    for i_bra in 0..ni { for j_all in 0..nao { for k_ao in 0..qj { for p in 0..naux {
                        let k_abs = q0 + k_ao;
                        let (ka, kp0, kni) = blk.iter().find(|&&(_, s, sz)| k_abs >= s && k_abs < s+sz).map(|&(ia,s,sz)| (ia,s,sz)).unwrap();
                        let k_local = k_abs - kp0;
                        let rk_atom = &rhok_IkP_per_atom[ka];
                        let rval = rk_atom[(p0 + i_bra) * kni * naux * 3 + k_local * naux * 3 + p * 3 + y];
                        part1 += ip1[x * nao3 * naux + (p0+i_bra) * nao * naux + j_all * naux + p]
                            * rval * dm0[(q0+k_ao) * nao + j_all];
                    }}}}
                    // Part 2: ipv·tmp·dm0
                    // Python: vk1 += einsum('xijp,pki->xjk', ipv, tmp).reshape(3,3,nao,nao)
                    // → result[x1,x2,j,k] added to vk1[x1,x2,k_slot,j_slot]
                    // vk1[:,:,q0:q1] slices k_slot = result's j = int3c's 2nd AO
                    // So j_ao(ket_block), k_ao(all_AO)
                    for i_bra in 0..ni { for j_ao in 0..qj { for k_ao in 0..nao { for p in 0..naux {
                        part2 += ipv[c * nao3 * naux + (p0+i_bra) * nao * naux + (q0+j_ao) * naux + p]
                            * tmp[p * nao * ni + k_ao * ni + i_bra]
                            * dm0[(q0+j_ao) * nao + k_ao];
                    }}}}
                    // Part 3: vk2buf·dm0
                    // Python: ek += Σ_{k∈ket} Σ_{j∈bra} vk2buf[x,y,k_ket,j_bra] * dm0[k_ket,j_bra]
                    // Rust store: vk2buf[c, k, i] = ip1[x,i,j,p] * rhok[p,k,j,y]
                    // So index: vk2buf[c, k_ket, j_bra] = vk2buf[c, (q0+j_ao), k_ao]
                    for j_ao in 0..qj { for k_ao in p0..p0+ni {
                        part3 += vk2buf[c * nao3 + (q0 + j_ao) * nao + k_ao]
                            * dm0[(q0 + j_ao) * nao + k_ao];
                    }}
                    ek_vk1[i_t(i0, j0, x, y)] = part1 + part2 + part3;
                }}
            }
        }
                self.timings.push(("  g4_ek_vk1", _t_g4_ek_vk1.elapsed()));
                if let Some(ref bl) = baseline {
                    bl.verify_term("ek_vk1", &ek_vk1, self.flags.ej_ek_opt.verify_tol);
                }
            }
            TermPath::Blas => {
                let _t_g4 = std::time::Instant::now();
                let ctx = build_ej_ek_ctx!();
                g4_ek_vk1_blas(&ctx, &ip1, &ipv, &mut ek_vk1);
                self.timings.push(("  g4_ek_vk1", _t_g4.elapsed()));
            }
        }
        // g4 is the last consumer of ip1 and ipv — free them before computing
        // the delayed ip12/ipip2 integrals. Saves ~432 MiB (108+324).
        drop(ip1);
        drop(ipv);

        // Delayed integrals (only used in Phase 4 G-terms):
        //   ip12  ~324 MiB — used by g5 and g8
        //   ipip2 ~324 MiB — used by g6 and g9
        let ip12  = int3c("int3c2e_ip1ip2"); // (9,N,N,P)
        let ipip2 = int3c("int3c2e_ipip2");  // (9,N,N,P)

        match self.flags.ej_ek_opt.g5_ek_ri1 {
            TermPath::Inline => {
        // 5. ek_ri1: RI first-order K response (prototype L179-226)
        let _t_e5 = std::time::Instant::now();
        // Need wk1_Pij = rho_ip1-like data from tmpf
        // Build rho_ip1: (nao, nao, naux, 3) from tmpf
        let mut rho_ip1 = vec![0.0; nao * nao * naux * 3];
        for p in 0..naux { for x in 0..3 { for i in 0..nao { for j in 0..nao {
            rho_ip1[i * nao * naux * 3 + j * naux * 3 + p * 3 + x] = tmpf[p + (x * nao3 + i * nao + j) * naux];
        }}}}
        // Pre-compute wk1_IpJ from wki (the wk_ip2_Ipk intermediate)
        let mut wk1_IpJ_full = vec![0.0; nao * naux * 3 * nao]; // same layout as wki
        for i in 0..nao { for p in 0..naux { for y in 0..3 { for k in 0..nao {
            let mut s = 0.0; for j in 0..nao {
                s += wki[i * naux * 3 * nao + p * 3 * nao + y * nao + j] * dm0[j * nao + k];
            }
            wk1_IpJ_full[i * naux * 3 * nao + p * 3 * nao + y * nao + k] = s;
        }}}}

        for i0 in 0..natm { let (_, p0, ni) = blk[i0];
            // Build wk1_Pij = rho_ip1[p0:p0+ni].transpose(2,3,0,1) → (naux, 3, ni, nao)
            let mut wkp = vec![0.0; naux * 3 * ni * nao];
            for p in 0..naux { for x in 0..3 { for ii in 0..ni { for j in 0..nao {
                wkp[p * 3 * ni * nao + x * ni * nao + ii * nao + j] =
                    rho_ip1[(p0 + ii) * nao * naux * 3 + j * naux * 3 + p * 3 + x];
            }}}}
            // rhok0_P_I = einsum('plj,il->pji', rk_block, dm0_block)
            // rk[p, l_ao, j_occ] * dm0[i_block, l_ao] → rk_P_I[p, j_occ, i_block]
            let mut rk_P_I = vec![0.0; naux * nocc * ni];
            for p in 0..naux { for j_occ in 0..nocc { for ii in 0..ni {
                let mut s = 0.0; for l in 0..nao {
                    s += rk[p + l * naux + j_occ * naux * nao] * dm0[l * nao + (p0 + ii)];
                } rk_P_I[p * nocc * ni + j_occ * ni + ii] = s;
            }}}
            // rhok0_PJI = einsum('pji,Jj->pJi', rhok0_P_I, mocc_2)
            let mut rk_PJI = vec![0.0; naux * nao * ni];
            for p in 0..naux { for J in 0..nao { for ii in 0..ni {
                let mut s = 0.0; for j_occ in 0..nocc {
                    s += rk_P_I[p * nocc * ni + j_occ * ni + ii] * mc2[J * nocc + j_occ];
                } rk_PJI[p * nao * ni + J * ni + ii] = s;
            }}}
            // wk1_pJI = einsum('ypq,qji->ypji', i21, rk_PJI)
            let mut wk1_pJI = vec![0.0; 3 * naux * nao * ni];
            for y in 0..3 { for p in 0..naux { for J in 0..nao { for ii in 0..ni {
                let mut s = 0.0; for q in 0..naux {
                    s += i21[y * naux * naux + p * naux + q] * rk_PJI[q * nao * ni + J * ni + ii];
                } wk1_pJI[y * naux * nao * ni + p * nao * ni + J * ni + ii] = s;
            }}}}
            // wk1_IpJ = einsum('ipyk,kj->ipyj', wk_ip2_Ipk[p0:p1], dm0)
            let mut wk1_IpJ = vec![0.0; ni * naux * 3 * nao];
            for ii in 0..ni { for p in 0..naux { for y in 0..3 { for k in 0..nao {
                wk1_IpJ[ii * naux * 3 * nao + p * 3 * nao + y * nao + k] =
                    wk1_IpJ_full[(p0 + ii) * naux * 3 * nao + p * 3 * nao + y * nao + k];
            }}}}
            // rho2c_PQ = einsum('pxij,qji->xqp', wk1_Pij, rk_PJI)
            let mut rho2c_PQ = vec![0.0; 3 * naux * naux];
            for x in 0..3 { for q in 0..naux { for p_aux in 0..naux {
                let mut s = 0.0; for ii in 0..ni { for j in 0..nao {
                    s += wkp[p_aux * 3 * ni * nao + x * ni * nao + ii * nao + j]
                        * rk_PJI[q * nao * ni + j * ni + ii];
                }} rho2c_PQ[x * naux * naux + q * naux + p_aux] = s;
            }}}

            for j0 in 0..natm { let (_, aq0, aq_sz) = aux_blk[j0]; let ql = aq_sz;
                // Use full ip12 integral with AO offset p0
                let q0 = aq0;
                // Term1: 'xijp,pji->x' from ip1ip2 × rk_PJI
                let mut t1 = vec![0.0; 9];
                for x in 0..9 { let mut s = 0.0;
                    for ii in 0..ni { for j in 0..nao { for qp in 0..ql {
                        let pg = q0 + qp;
                        s += ip12[x * nao3 * naux + (p0+ii) * nao * naux + j * naux + pg]
                            * rk_PJI[pg * nao * ni + j * ni + ii];
                    }}} t1[x] = s;
                }
                // Term2: 'pxij,ypji->xy' wkp × wk1_pJI
                let mut t2 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for qp in 0..ql { let pg = q0 + qp;
                        for ii in 0..ni { for j in 0..nao {
                            s += wkp[pg * 3 * ni * nao + x * ni * nao + ii * nao + j]
                                * wk1_pJI[y * naux * nao * ni + pg * nao * ni + j * ni + ii];
                        }}
                    } t2[x * 3 + y] = s;
                }}
                // Term3: 'xqp,yqp->xy' rho2c_PQ × i21
                let mut t3 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for qp in 0..ql { let qg = q0 + qp;
                        for paux in 0..naux {
                            s += rho2c_PQ[x * naux * naux + qg * naux + paux]
                                * i21[y * naux * naux + qg * naux + paux];
                        }
                    } t3[x * 3 + y] = s;
                }}
                // Term4: 'pxij,ipyj->xy' wkp × wk1_IpJ
                let mut t4 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for qp in 0..ql { let pg = q0 + qp;
                        for ii in 0..ni { for j in 0..nao {
                            s += wkp[pg * 3 * ni * nao + x * ni * nao + ii * nao + j]
                                * wk1_IpJ[ii * naux * 3 * nao + pg * 3 * nao + y * nao + j];
                        }}
                    } t4[x * 3 + y] = s;
                }}
                // _ek = t1 - t2 - t3 + t4
                for x in 0..3 { for y in 0..3 {
                    let v = t1[x * 3 + y] - t2[x * 3 + y] - t3[x * 3 + y] + t4[x * 3 + y];
                    ek_ri1[i_t(i0, j0, x, y)] += v;
                    ek_ri1[i_t(j0, i0, x, y)] += t1[y * 3 + x] - t2[y * 3 + x] - t3[y * 3 + x] + t4[y * 3 + x];
                }}
            }
        }
        self.timings.push(("  ek_ri1", _t_e5.elapsed()));
                if let Some(ref bl) = baseline {
                    bl.verify_term("ek_ri1", &ek_ri1, self.flags.ej_ek_opt.verify_tol);
                }
            }
            TermPath::Blas => {
                let _t_e5 = std::time::Instant::now();
                let ctx = build_ej_ek_ctx!();
                g5_ek_ri1_blas(&ctx, &ip12, &mut ek_ri1);
                self.timings.push(("  ek_ri1", _t_e5.elapsed()));
                if let Some(ref bl) = baseline {
                    bl.verify_term("ek_ri1", &ek_ri1, self.flags.ej_ek_opt.verify_tol);
                }
            }
        }
        match self.flags.ej_ek_opt.g6_ek_ri2d {
            TermPath::Inline => {
        // 6. ek_ri2d: RI second-order K diagonal (prototype L230-250)
        let _t_e6 = std::time::Instant::now();
        for i0 in 0..natm { let (_, ap0, ni_aux) = aux_blk[i0];
            let mut rkj = vec![0.0; ni_aux * nao * nao];
            for p in 0..ni_aux { for J in 0..nao { for I in 0..nao {
                let mut s = 0.0; for j in 0..nocc { for i in 0..nocc {
                    s += rkoo[(ap0 + p) * nocc * nocc + i * nocc + j] * mc2[J * nocc + j] * mc2[I * nocc + i];
                }} rkj[p * nao * nao + J * nao + I] = s;
            }}}
            // Use full ipip2 integral with aux offset ap0
            let mut ta = vec![0.0; 9];
            for x in 0..9 { let mut s = 0.0;
                for I in 0..nao { for J in 0..nao { for p in 0..ni_aux {
                    s += ipip2[x * nao * nao * naux + I * nao * naux + J * naux + (ap0 + p)]
                        * rkj[p * nao * nao + I * nao + J];
                }}} ta[x] = s * 0.5;
            }
            let mut tb = vec![0.0; 9];
            for x in 0..9 { let mut s = 0.0;
                for p in 0..ni_aux { for q in 0..naux {
                    s += r2c0[(ap0 + p) * naux + q] * i211[x * naux * naux + (ap0 + p) * naux + q];
                }} tb[x] = s * (-0.5);
            }
            for x in 0..3 { for y in 0..3 {
                ek_ri2d[i_t(i0, i0, x, y)] += ta[x * 3 + y] + tb[x * 3 + y];
            }}
        }

        self.timings.push(("  ek_ri2d", _t_e6.elapsed()));
                if let Some(ref bl) = baseline {
                    bl.verify_term("ek_ri2d", &ek_ri2d, self.flags.ej_ek_opt.verify_tol);
                }
            }
            TermPath::Blas => {
                let _t_e6 = std::time::Instant::now();
                let ctx = build_ej_ek_ctx!();
                g6_ek_ri2d_blas(&ctx, &ipip2, &mut ek_ri2d);
                self.timings.push(("  ek_ri2d", _t_e6.elapsed()));
                if let Some(ref bl) = baseline {
                    bl.verify_term("ek_ri2d", &ek_ri2d, self.flags.ej_ek_opt.verify_tol);
                }
            }
        }
        match self.flags.ej_ek_opt.g7_ek_ri2o {
            TermPath::Inline => {
        // 7. ek_ri2o: RI second-order K off-diagonal (prototype L253-278)
        let _t_e7 = std::time::Instant::now();
        // Need rho2c_1 intermediate, which we compute here
        let mut rho2c1_per_atom = vec![0.0; natm * 3 * naux * naux];
        for i0 in 0..natm { let (_, ap0, ni_aux) = aux_blk[i0];
            // ip1_2c_2c = ip1[:,ap0:ap1,:] · i2inv  ⇒  (3, ni, P) — via BLAS matmul
            let mut ip1_2c = vec![0.0; 3 * ni_aux * naux];
            let mut ip1_r2c = vec![0.0; 3 * ni_aux * naux];
            {
                // Stage i21[3, ni, naux] → col-major [3*ni, naux]
                let mut i21_stage = vec![0.0; 3 * ni_aux * naux];
                for x in 0..3 { for p in 0..ni_aux { for q in 0..naux {
                    i21_stage[(x * ni_aux + p) + q * (3 * ni_aux)] =
                        i21[x * naux * naux + (ap0 + p) * naux + q];
                }}}
                let i21_mm = rt::asarray((&i21_stage, [3 * ni_aux, naux].f(), &device));
                // ip1_2c = i21 @ i2inv
                let ip1_2c_tmp = &i21_mm % &i2inv_t; // [3*ni, naux]
                // ip1_r2c = 0.5 * i21 @ r2c0
                let ip1_r2c_tmp = (&i21_mm % &r2c0_t) * 0.5; // [3*ni, naux]
                // Copy back — col-major [3*ni, naux] layout: element [x*ni+p, r]
                for x in 0..3 { for p in 0..ni_aux { for r in 0..naux {
                    ip1_2c[x * ni_aux * naux + p * naux + r] = ip1_2c_tmp[[x * ni_aux + p, r]];
                    ip1_r2c[x * ni_aux * naux + p * naux + r] = ip1_r2c_tmp[[x * ni_aux + p, r]];
                }}}
            }
            // rho2c_1 = ip1_rho2c · i2inv[ap0:ap1] + ip1_2c_2c · r2c0[ap0:ap1] - tmp_so - tmp_so.T
            // where ip1_rho2c[x, r, q] * i2inv[ap0+r, p] (note: p from i2inv, q from r2c0)
            let mut r2c1 = vec![0.0; 3 * naux * naux];
            for x in 0..3 { for p in 0..naux { for q in 0..naux {
                let mut s1 = 0.0; let mut s2 = 0.0;
                for r in 0..ni_aux {
                    // s1 = ip1_rho2c[x, r, q] * i2inv[ap0+r, p]
                    s1 += ip1_r2c[x * ni_aux * naux + r * naux + q] * i2inv[(ap0 + r) * naux + p];
                    s2 += ip1_2c[x * ni_aux * naux + r * naux + p] * r2c0[(ap0 + r) * naux + q];
                }
                let mut val = s1 + s2;
                // tmp_so[x, p, q] = Σ_{r,i,j} wk2[ap0+r, x, i, j] * rkoo[q, i, j] * i2inv[ap0+r, p]
                // rho2c_1 -= tmp_so + tmp_so^T(swap last two axes)
                for r in 0..ni_aux { for i in 0..nocc { for j in 0..nocc {
                    val -= wk2[(ap0 + r) * 3 * nocc * nocc + x * nocc * nocc + i * nocc + j]
                        * rkoo[q * nocc * nocc + i * nocc + j]
                        * i2inv[(ap0 + r) * naux + p];
                    val -= wk2[(ap0 + r) * 3 * nocc * nocc + x * nocc * nocc + i * nocc + j]
                        * rkoo[p * nocc * nocc + i * nocc + j]
                        * i2inv[(ap0 + r) * naux + q];
                }}}
                r2c1[x * naux * naux + p * naux + q] = val;
            }}}
            for x in 0..3 { for p in 0..naux { for q in 0..naux {
                rho2c1_per_atom[i0 * 3 * naux * naux + x * naux * naux + p * naux + q] = r2c1[x * naux * naux + p * naux + q];
            }}}
        }
        for i0 in 0..natm { let (_, ap0, ni_aux) = aux_blk[i0];
            let r2c1 = &rho2c1_per_atom[i0 * 3 * naux * naux..];
            for j0 in 0..natm { let (_, aq0, nq) = aux_blk[j0];
                // T1: 0.5 * einsum('pq,xypq->xy', r2c0_sub, i2ip_sub)
                let mut t1 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for p in 0..ni_aux { for q in 0..nq {
                        s += r2c0[(ap0 + p) * naux + (aq0 + q)]
                            * i2ip[(x * 3 + y) * naux * naux + (ap0 + p) * naux + (aq0 + q)];
                    }} t1[x * 3 + y] = s * 0.5;
                }}
                // T2: einsum('xpq,ypq->xy', rho2c_1, i21_sub)
                let mut t2 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for p in 0..nq { for q in 0..naux {
                        s += r2c1[x * naux * naux + (aq0 + p) * naux + q]
                            * i21[y * naux * naux + (aq0 + p) * naux + q];
                    }} t2[x * 3 + y] = s;
                }}
                // T3: 0.5 * einsum('pxij,pq,qyij->xy', wk2_sub, i2cinv_sub, wk2_sub_2)
                // Replaced 7-level loop with BLAS matmul chain
                let ni_t3 = ni_aux; let nj_t3 = nq;
                let nocc2 = nocc * nocc;
                let mut t3 = vec![0.0; 9];
                {
                    // Stage wk2 blocks as col-major [ni, 3*nocc²] and [nj, 3*nocc²]
                    let mut wi_block = vec![0.0; ni_t3 * 3 * nocc2];
                    let mut wj_block = vec![0.0; nj_t3 * 3 * nocc2];
                    for p in 0..ni_t3 { for x in 0..3 { for i in 0..nocc { for j in 0..nocc {
                        let src = wk2[(ap0 + p) * 3 * nocc2 + x * nocc2 + i * nocc + j];
                        wi_block[p + (x * nocc2 + i * nocc + j) * ni_t3] = src;
                    }}}}
                    for p in 0..nj_t3 { for x in 0..3 { for i in 0..nocc { for j in 0..nocc {
                        let src = wk2[(aq0 + p) * 3 * nocc2 + x * nocc2 + i * nocc + j];
                        wj_block[p + (x * nocc2 + i * nocc + j) * nj_t3] = src;
                    }}}}
                    let wi_t = rt::asarray((&wi_block, [ni_t3, 3 * nocc2].f(), &device));
                    let wj_t = rt::asarray((&wj_block, [nj_t3, 3 * nocc2].f(), &device));
                    // i2inv_sub: [ni, nj]
                    let mut i2inv_sub = vec![0.0; ni_t3 * nj_t3];
                    for p in 0..ni_t3 { for q in 0..nj_t3 {
                        i2inv_sub[p + q * ni_t3] = i2inv[(ap0 + p) * naux + (aq0 + q)];
                    }}
                    let i2_sub = rt::asarray((&i2inv_sub, [ni_t3, nj_t3].f(), &device));
                    // T3[x,y] = Σ_{p,q,i,j} wk2[p,x,i,j] * i2inv[p,q] * wk2[q,y,i,j]
                    for x in 0..3 {
                        let wi_x = wi_t.i((.., x * nocc2..(x + 1) * nocc2));
                        for y in 0..3 {
                            let wj_y = wj_t.i((.., y * nocc2..(y + 1) * nocc2));
                            let outer = &wi_x % &wj_y.t(); // [ni, nj]
                            let prod = &outer * i2_sub.i((.., .., None));
                            let val = prod.sum_axes(&[0, 1]).into_shape(-1).into_raw()[0];
                            t3[x * 3 + y] = val * 0.5;
                        }
                    }
                }
                for x in 0..3 { for y in 0..3 {
                    let ek_val = (t1[x * 3 + y] + t2[x * 3 + y] + t3[x * 3 + y]) * 0.5;
                    ek_ri2o[i_t(i0, j0, x, y)] += ek_val;
                    ek_ri2o[i_t(j0, i0, x, y)] += (t1[y * 3 + x] + t2[y * 3 + x] + t3[y * 3 + x]) * 0.5;
                }}
        }
        }
        self.timings.push(("  ek_ri2o", _t_e7.elapsed()));
                if let Some(ref bl) = baseline {
                    bl.verify_term("ek_ri2o", &ek_ri2o, self.flags.ej_ek_opt.verify_tol);
                }
            }
            TermPath::Blas => {
                let _t_e7 = std::time::Instant::now();
                let ctx = build_ej_ek_ctx!();
                g7_ek_ri2o_blas(&ctx, &mut ek_ri2o);
                self.timings.push(("  ek_ri2o", _t_e7.elapsed()));
                if let Some(ref bl) = baseline {
                    bl.verify_term("ek_ri2o", &ek_ri2o, self.flags.ej_ek_opt.verify_tol);
                }
            }
        }
        match self.flags.ej_ek_opt.g8_ej_ri1 {
            TermPath::Inline => {
        // 8. ej_ri1: RI first-order J response (prototype L281-313)
        let _t_e8 = std::time::Instant::now();
        for i0 in 0..natm { let (ib, p0, ni) = blk[i0];
            let mut w11 = vec![0.0; 9 * naux];
            for x in 0..9 { for p in 0..naux { let mut ss = 0.0;
                for ii in 0..ni { for j in 0..nao {
                    ss += ip12[x * nao3 * naux + (p0+ii) * nao * naux + j * naux + p]
                        * dm0[(p0 + ii) * nao + j];
                }} w11[x * naux + p] = ss;
            }}
            for j0 in 0..natm { let (_, aq0, ql) = aux_blk[j0];
                let q0 = aq0;
                // T1: 'xp,p->x' w11[:,q0:q1], r0[q0:q1]
                let mut t1 = vec![0.0; 9];
                for x in 0..9 { let mut s = 0.0;
                    for qp in 0..ql { s += w11[x * naux + q0 + qp] * r0[q0 + qp]; } t1[x] = s;
                }
                // T2: 'yqp,q,px->xy' i21, r0, rj1
                // T2: 'yqp,q,px->xy' → i21[y, q(g), p(x)]  where q∈aux(j0), p∈all_aux
                let mut t2 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for qp in 0..ql { let qg = q0 + qp;
                        for paux in 0..naux {
                            s += i21[y * naux * naux + qg * naux + paux] * r0[qg] * rj1[i0 * naux * 3 + paux * 3 + x];
                        }
                    } t2[x * 3 + y] = s;
                }}
                // T3: 'px,yp->xy' rj1, wj001
                let mut t3 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for qp in 0..ql { let qg = q0 + qp;
                        s += rj1[i0 * naux * 3 + qg * 3 + x] * wj001[y * naux + qg];
                    } t3[x * 3 + y] = s;
                }}
                // T4: 'px,py->xy' rj1, wj2
                let mut t4 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for qp in 0..ql { let qg = q0 + qp;
                        s += rj1[i0 * naux * 3 + qg * 3 + x] * wj2[qg * 3 + y];
                    } t4[x * 3 + y] = s;
                }}
                for x in 0..3 { for y in 0..3 {
                    let v = (t1[x * 3 + y] - t2[x * 3 + y] - t3[x * 3 + y] + t4[x * 3 + y]) * 2.0;
                    ej_ri1[i_t(i0, j0, x, y)] += v;
                    ej_ri1[i_t(j0, i0, x, y)] += (t1[y * 3 + x] - t2[y * 3 + x] - t3[y * 3 + x] + t4[y * 3 + x]) * 2.0;
                }}
            }
        }
        self.timings.push(("  ej_ri1", _t_e8.elapsed()));
                if let Some(ref bl) = baseline {
                    bl.verify_term("ej_ri1", &ej_ri1, self.flags.ej_ek_opt.verify_tol);
                }
            }
            TermPath::Blas => {
                let _t_e8 = std::time::Instant::now();
                let ctx = build_ej_ek_ctx!();
                g8_ej_ri1_blas(&ctx, &ip12, &mut ej_ri1);
                self.timings.push(("  ej_ri1", _t_e8.elapsed()));
                if let Some(ref bl) = baseline {
                    bl.verify_term("ej_ri1", &ej_ri1, self.flags.ej_ek_opt.verify_tol);
                }
            }
        }
        // g8 is the last consumer of ip12 — free it. Saves ~324 MiB.
        drop(ip12);

        match self.flags.ej_ek_opt.g9_ej_ri2d {
            TermPath::Inline => {
        // 9. ej_ri2d: RI second-order J diagonal
        let _t_e9 = std::time::Instant::now();
        for i0 in 0..natm { let (_, ap0, ni_aux) = aux_blk[i0];
            let mut td = vec![0.0; 9];
            for x in 0..9 { let mut s = 0.0;
                for i in 0..nao { for j in 0..nao { for p in 0..ni_aux {
                    s += ipip2[x * nao * nao * naux + i * nao * naux + j * naux + (ap0 + p)]
                        * dm0[j * nao + i] * r0[ap0 + p];
                }}} td[x] = s;
            }
            let mut te = vec![0.0; 9];
            for x in 0..9 { let mut s = 0.0;
                for p in 0..ni_aux { for q in 0..naux {
                    s += r0[ap0 + p] * i211[x * naux * naux + (ap0 + p) * naux + q] * r0[q];
                }} te[x] = s;
            }
            for x in 0..3 { for y in 0..3 {
                ej_ri2d[i_t(i0, i0, x, y)] += td[x * 3 + y] - te[x * 3 + y];
            }}
        }
        self.timings.push(("  ej_ri2d", _t_e9.elapsed()));
                if let Some(ref bl) = baseline {
                    bl.verify_term("ej_ri2d", &ej_ri2d, self.flags.ej_ek_opt.verify_tol);
                }
            }
            TermPath::Blas => {
                let _t_e9 = std::time::Instant::now();
                let ctx = build_ej_ek_ctx!();
                g9_ej_ri2d_blas(&ctx, &ipip2, &mut ej_ri2d);
                self.timings.push(("  ej_ri2d", _t_e9.elapsed()));
                if let Some(ref bl) = baseline {
                    bl.verify_term("ej_ri2d", &ej_ri2d, self.flags.ej_ek_opt.verify_tol);
                }
            }
        }
        // g9 is the last consumer of ipip2 — free it. Saves ~324 MiB.
        drop(ipip2);

        match self.flags.ej_ek_opt.g10_ej_ri2o {
            TermPath::Inline => {
        // 10. ej_ri2o: RI second-order J off-diagonal
        let _t_e10 = std::time::Instant::now();
        // Need rhoj1_so, rhoj0_01, rhoj0_10 per atom
        let mut rs1_per = vec![0.0; natm * 3 * naux];
        let mut r01_per = vec![0.0; natm * 3 * naux];
        let mut r10_per = vec![0.0; natm * 3 * naux];
        for i0 in 0..natm { let (_, ap0, ni_aux) = aux_blk[i0];
            // rhoj1_so = wj2[ap0:ap1] · i2inv   → (3, P)
            for x in 0..3 { for q in 0..naux {
                let mut s = 0.0; for p in 0..ni_aux {
                    s += wj2[(ap0 + p) * 3 + x] * i2inv[(ap0 + p) * naux + q];
                } rs1_per[i0 * 3 * naux + x * naux + q] = s;
            }}
            // rhoj0_01 = wj001[:,ap0:ap1] · i2inv  → (3, P)
            for x in 0..3 { for q in 0..naux {
                let mut s = 0.0; for p in 0..ni_aux {
                    s += wj001[x * naux + (ap0 + p)] * i2inv[(ap0 + p) * naux + q];
                } r01_per[i0 * 3 * naux + x * naux + q] = s;
            }}
            // ip1_2c_2c = ip1[:,ap0:ap1,:] · i2inv
            let mut ip1_2c = vec![0.0; 3 * ni_aux * naux];
            for x in 0..3 { for p in 0..ni_aux { for r in 0..naux {
                let mut s = 0.0; for q in 0..naux {
                    s += i21[x * naux * naux + (ap0 + p) * naux + q] * i2inv[q + r * naux];
                } ip1_2c[x * ni_aux * naux + p * naux + r] = s;
            }}}
            // rhoj0_10 = r0[ap0:ap1] · ip1_2c_2c  → (3, P)
            for x in 0..3 { for r in 0..naux {
                let mut s = 0.0; for p in 0..ni_aux {
                    s += r0[ap0 + p] * ip1_2c[x * ni_aux * naux + p * naux + r];
                } r10_per[i0 * 3 * naux + x * naux + r] = s;
            }}
        }
        for i0 in 0..natm { let (_, ap0, ni_aux) = aux_blk[i0];
            for j0 in 0..natm { let (_, aq0, nq) = aux_blk[j0];
                // O1: 0.5*'p,xypq,q->xy'  (p in i0's aux block, q in j0's aux block)
                let mut o1 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for p in 0..ni_aux { for q in 0..nq {
                        s += r0[ap0 + p] * i2ip[(x * 3 + y) * naux * naux + (ap0 + p) * naux + (aq0 + q)] * r0[aq0 + q];
                    }} o1[x * 3 + y] = s * 0.5;
                }}
                // O2: 'xp,yp->xy' rs1 × wj001
                let mut o2 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for qp in 0..nq { let qg = aq0 + qp;
                        s += rs1_per[i0 * 3 * naux + x * naux + qg] * wj001[y * naux + qg];
                    } o2[x * 3 + y] = s;
                }}
                // O3: 0.5*'xp,py->xy' rs1 × wj2
                let mut o3 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for qp in 0..nq { let qg = aq0 + qp;
                        s += rs1_per[i0 * 3 * naux + x * naux + qg] * wj2[qg * 3 + y];
                    } o3[x * 3 + y] = s * 0.5;
                }}
                // O4: 0.5*'xp,yp->xy' r01 × wj001
                let mut o4 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for qp in 0..nq { let qg = aq0 + qp;
                        s += r01_per[i0 * 3 * naux + x * naux + qg] * wj001[y * naux + qg];
                    } o4[x * 3 + y] = s * 0.5;
                }}
                // O5: 'yqp,q,xp->xy' i21 × r0 × rs1
                let mut o5 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for qp in 0..nq { let qg = aq0 + qp;
                        for paux in 0..naux {
                            s += i21[y * naux * naux + qg * naux + paux] * r0[qg] * rs1_per[i0 * 3 * naux + x * naux + paux];
                        }
                    } o5[x * 3 + y] = s;
                }}
                // O6: 'xp,yp->xy' r10 × wj001
                let mut o6 = vec![0.0; 9];
                for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                    for qp in 0..nq { let qg = aq0 + qp;
                        s += r10_per[i0 * 3 * naux + x * naux + qg] * wj001[y * naux + qg];
                    } o6[x * 3 + y] = s;
                }}
                for x in 0..3 { for y in 0..3 {
                    let v = o1[x * 3 + y] - o2[x * 3 + y] + o3[x * 3 + y] + o4[x * 3 + y] - o5[x * 3 + y] + o6[x * 3 + y];
                    ej_ri2o[i_t(i0, j0, x, y)] += v;
                    ej_ri2o[i_t(j0, i0, x, y)] += o1[y * 3 + x] - o2[y * 3 + x] + o3[y * 3 + x] + o4[y * 3 + x] - o5[y * 3 + x] + o6[y * 3 + x];
                }}
            }
        }

        self.timings.push(("  ej_ri2o", _t_e10.elapsed()));
                if let Some(ref bl) = baseline {
                    bl.verify_term("ej_ri2o", &ej_ri2o, self.flags.ej_ek_opt.verify_tol);
                }
            }
            TermPath::Blas => {
                let _t_e10 = std::time::Instant::now();
                let ctx = build_ej_ek_ctx!();
                g10_ej_ri2o_blas(&ctx, &mut ej_ri2o);
                self.timings.push(("  ej_ri2o", _t_e10.elapsed()));
                if let Some(ref bl) = baseline {
                    bl.verify_term("ej_ri2o", &ej_ri2o, self.flags.ej_ek_opt.verify_tol);
                }
            }
        }
        // All G-term contractions are done. The raw RI integrals (ip1, ipv,
        // ipip2, ip12, i21, i211, i2ip, tmpf, ip1c) and the large Phase 1-3
        // intermediates they were contracted into (rhok_IkP_per_atom, vk2buf,
        // wj2, wki, wk2, rkoo, r2c0, wj001, r0, rk, rj1, wj1, vjd, vkd) plus
        // their rstsr views are no longer referenced. Drop them now so the
        // subsequent h_partial assembly, RKS XC contributions, and baseline
        // save run without carrying O(naux·nao²) + O(natm·nao²·naux) memory.
        drop((i2ip_t, i211_t, wk2_t, rkoo_t, r2c0_t,
              wj2_t, wj001_t, r0_t_ri, rj1_t_ri));
        // ip1, ipv, ip12, ipip2 already dropped after their last G-term use.
        drop((i21, i211, i2ip, tmpf));
        drop((rhok_IkP_per_atom, vk2buf, wj2, wki, wk2, rkoo, r2c0,
              wj001, r0, rk, rj1, wj1, vjd, vkd));
        // ── Sum contributions ──
        let mut ej = vec![0.0; aa9];
        let mut ek = vec![0.0; aa9];
        for i in 0..aa9 {
            ej[i] = ej_basic[i] + ej_vjd[i] + ej_vj1[i] + ej_ri1[i] + ej_ri2d[i] + ej_ri2o[i];
            ek[i] = ek_vkd[i] + ek_vk1[i] + ek_ri1[i] + ek_ri2d[i] + ek_ri2o[i];
        }
        self.timings.push(("  phases_4-sum", _tp4.elapsed()));
        self.timings.push(("  ej_ek: contributions", _tej.elapsed()));
        // Symmetrize: (i0,j0) → (j0,i0) by copying
        for i0 in 0..natm { for j0 in 0..i0 { for x in 0..3 { for y in 0..3 {
            let a = i_t(i0, j0, x, y); let b = i_t(j0, i0, y, x);
            ej[b] = ej[a]; ek[b] = ek[a];
        }}}}

        // ── Store results ──
        // Flatten to col-major [n3, n3] MatrixFull
        let to_mat = |arr: &[f64]| -> MatrixFull<f64> {
            let n3 = natm * 3; let mut m = vec![0.0; n3 * n3];
            for i0 in 0..natm { for j0 in 0..natm { for x in 0..3 { for y in 0..3 {
                m[(i0 * 3 + x) + (j0 * 3 + y) * n3] = arr[i_t(i0, j0, x, y)];
            }}}}
            // Symmetrize j0 < i0
            for i0 in 0..natm { for j0 in 0..i0 { for x in 0..3 { for y in 0..3 {
                m[(j0 * 3 + y) + (i0 * 3 + x) * n3] = m[(i0 * 3 + x) + (j0 * 3 + y) * n3];
            }}}}
            MatrixFull::from_vec([n3, n3], m).unwrap()
        };
        self.result.insert("ej".to_string(), to_mat(&ej));
        self.result.insert("ek".to_string(), to_mat(&ek));
        // Store individual contributions for debugging
        self.result.insert("ej_basic".to_string(), to_mat(&ej_basic));
        self.result.insert("ej_vjd".to_string(), to_mat(&ej_vjd));
        self.result.insert("ej_vj1".to_string(), to_mat(&ej_vj1));
        self.result.insert("ej_ri1".to_string(), to_mat(&ej_ri1));
        self.result.insert("ej_ri2d".to_string(), to_mat(&ej_ri2d));
        self.result.insert("ej_ri2o".to_string(), to_mat(&ej_ri2o));
        self.result.insert("ek_vkd".to_string(), to_mat(&ek_vkd));
        self.result.insert("ek_vk1".to_string(), to_mat(&ek_vk1));
        self.result.insert("ek_ri1".to_string(), to_mat(&ek_ri1));
        self.result.insert("ek_ri2d".to_string(), to_mat(&ek_ri2d));
        self.result.insert("ek_ri2o".to_string(), to_mat(&ek_ri2o));


        // ── Finish h_partial as before ──
        // h_partial = e1 + factor_j*ej - factor_k*ek
        // (factor_k defaults to 1.0 for RHF; RKS wrapper sets it to hyb.)
        let factor_j = self.flags.factor_j.unwrap_or(1.0);
        let factor_k = self.flags.factor_k.unwrap_or(1.0);
        let e1_arr = if let Some(e1_mat) = self.result.get("e1") {
            // Convert MatrixFull back to flat (natm, natm, 3, 3) for summation
            let n3 = natm * 3; let mut e1_flat = vec![0.0; aa9];
            for i0 in 0..natm { for j0 in 0..natm { for x in 0..3 { for y in 0..3 {
                e1_flat[i_t(i0, j0, x, y)] = e1_mat[[(i0*3+x) as _, (j0*3+y) as _]];
            }}}}
            e1_flat
        } else { vec![0.0; aa9] };
        let mut hp = vec![0.0; aa9];
        for i in 0..aa9 { hp[i] = e1_arr[i] + factor_j * ej[i] - factor_k * ek[i]; }
        self.result.insert("h_partial".to_string(), to_mat(&hp));

        // ── RKS: add XC contributions (vxc_diag + vxc_deriv2) ──
        if self.is_rks() {
            let _t_rks_xc = std::time::Instant::now();
            let scf_rks: &SCF = self.scf_data;
            let mol_rks = &scf_rks.mol;
            let nao_xc = mol_rks.num_basis;
            let natm_xc = mol_rks.geom.nfree;
            let n3_xc = natm_xc * 3;
            let xc_type = if mol_rks.xc_data.use_density_gradient() {
                crate::dft::xc_deriv::XCType::GGA
            } else {
                crate::dft::xc_deriv::XCType::LDA
            };
            let aoslices_xc = crate::hessian::xc_hessian::build_aoslices(mol_rks);
            let dm0_xc = &scf_rks.density_matrix[0];

            let hp_entry = self.result.remove("h_partial")
                .expect("h_partial not set");
            let mut hp_flat: Vec<f64> = hp_entry.iter().copied().collect();

            // Return freed Phase 1-4 memory to OS before the DFT grid sweep.
            memory_monitor::trim_to_os();

            // vxc_diag (computed once, scattered to diagonal atom blocks)
            let _t_diag = std::time::Instant::now();
            // Phase 2: streaming mode — process grid blocks in small concurrent
            // batches instead of collecting all AO data in a cache. Peak AO
            // memory ≈ grid_concurrency() × per_block_size instead of
            // num_blocks × per_block_size.
            let vxc_diag_mat = crate::hessian::xc_hessian::vxc_diag_streaming(scf_rks, xc_type);
            for ia in 0..natm_xc {
                let (p0, p1) = aoslices_xc[ia];
                for a in 0..3 { for b in 0..3 {
                    let row0 = (a * 3 + b) * nao_xc;
                    let mut s = 0.0;
                    for mu in p0..p1 { for nu in 0..nao_xc {
                        s += vxc_diag_mat[[row0 + mu, nu]] * dm0_xc[[mu, nu]];
                    }}
                    hp_flat[(ia * 3 + a) * n3_xc + (ia * 3 + b)] += s * 2.0;
                }}
            }
            self.timings.push(("  rks: vxc_diag", _t_diag.elapsed()));
            // Free vxc_diag intermediates before the heavier vxc_deriv2 sweep.
            drop(vxc_diag_mat);
            memory_monitor::trim_to_os();

            // vxc_deriv2 (per-atom, symmetrized) — streaming with deriv=2.
            let _t_d2 = std::time::Instant::now();
            let vxc_d2 = crate::hessian::xc_hessian::vxc_deriv2_streaming(scf_rks, xc_type);
            for ia in 0..natm_xc {
                for ja in 0..=ia {
                    let (q0, q1) = aoslices_xc[ja];
                    for a in 0..3 { for b in 0..3 {
                        let row0 = (a * 3 + b) * nao_xc;
                        let mut s = 0.0;
                        for mu in q0..q1 { for nu in 0..nao_xc {
                            s += vxc_d2[ia][[row0 + mu, nu]] * dm0_xc[[mu, nu]];
                        }}
                        hp_flat[(ia * 3 + a) * n3_xc + (ja * 3 + b)] += s * 2.0;
                    }}
                }
            }
            for ia in 0..natm_xc {
                for ja in 0..ia {
                    for a in 0..3 { for b in 0..3 {
                        hp_flat[(ja * 3 + b) * n3_xc + (ia * 3 + a)] =
                            hp_flat[(ia * 3 + a) * n3_xc + (ja * 3 + b)];
                    }}
                }
            }

            self.result.insert("h_partial".to_string(),
                MatrixFull::from_vec([n3_xc, n3_xc], hp_flat).unwrap());
            self.timings.push(("  rks: vxc_deriv2", _t_d2.elapsed()));
            self.timings.push(("  rks: xc_add", _t_rks_xc.elapsed()));
        }

        // ── Save baseline (when no Inline verification happened and no baseline yet) ──
        // Saves BLAS (trusted) output as reference for future Inline verification runs.
        // The save happens once per (system, config) — delete the file to regenerate.
        if baseline.is_none() && !baseline_path.exists() {
            let mut terms = HashMap::new();
            for &key in BASELINE_TERM_KEYS {
                let arr: &[f64] = match key {
                    "ej_basic" => &ej_basic,
                    "ej_vjd" => &ej_vjd,
                    "ek_vkd" => &ek_vkd,
                    "ej_vj1" => &ej_vj1,
                    "ek_vk1" => &ek_vk1,
                    "ek_ri1" => &ek_ri1,
                    "ek_ri2d" => &ek_ri2d,
                    "ek_ri2o" => &ek_ri2o,
                    "ej_ri1" => &ej_ri1,
                    "ej_ri2d" => &ej_ri2d,
                    "ej_ri2o" => &ej_ri2o,
                    _ => continue,
                };
                terms.insert(key.to_string(), arr.to_vec());
            }
            let hp_max = hp.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            let sha = std::process::Command::new("git")
                .args(["rev-parse", "--short", "HEAD"])
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| "unknown".to_string());
            let bl = EjEkBaseline {
                system: sys_tag.to_string(),
                nao,
                naux,
                nocc,
                natm,
                verify_tol: opt.verify_tol,
                terms,
                h_partial_max_abs: hp_max,
                git_sha: sha,
            };
            match bl.save(&baseline_path) {
                Ok(()) => println!("  Saved ej_ek baseline to {:?}", baseline_path),
                Err(e) => {
                    // Try creating parent dir and retry once
                    if let Some(parent) = baseline_path.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    match bl.save(&baseline_path) {
                        Ok(()) => println!("  Saved ej_ek baseline to {:?}", baseline_path),
                        Err(e2) => eprintln!(
                            "Warning: failed to save baseline to {:?}: {} / {}",
                            baseline_path, e, e2
                        ),
                    }
                }
            }
        }

        self.timings.push(("calc_ej_ek", _t_global.elapsed()));
        self
    }

    /// Compute h1ao[ia] = hcore^{(1)}(ia) + vj1[ia] - 0.5*vk1[ia].
    /// Self-contained: recomputes all intermediates from SCF data.
    pub fn calc_h1ao(&mut self) -> &mut Self {
        let _t = std::time::Instant::now();
        let scf = self.scf_data; let mol = &scf.mol;
        let nao = mol.num_basis; let natm = mol.geom.nfree;
        let nocc = (scf.homo[0] + 1) as usize;
        let aoslices = mol.aoslice_by_atom();
        let cint_reg = mol.initialize_cint(false); let nreg = cint_reg.nbas();
        let cint_all = mol.initialize_cint(true);
        let naux_shell = cint_all.nbas() - nreg;
        let auxmol = mol.make_auxmol_fake(); let naux = auxmol.num_basis;
        let auxslices = auxmol.aoslice_by_atom();

        // V, V^{-1}
        let aux_slc_arr = [[nreg, nreg + naux_shell], [nreg, nreg + naux_shell]];
        let aux_slc: &[[usize; 2]] = &aux_slc_arr[..];
        // H2 optimization: prefer cached integrals from `calc_ej_ek`. Falls
        // back to recomputing when the cache is absent (e.g. when calc_h1ao
        // is called standalone without a preceding calc_ej_ek).
        let (i21, int2c_v, int2c, int2c_cm, vinv, t3c): (
            Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>,
        ) = if let Some(shared) = self.shared_integrals.as_ref() {
            // Recompute int2c_v/int2c/int2c_cm locally only — they are cheap
            // (O(naux²)) and not in the cache (the cached `vinv` is what
            // matters; calc_ej_ek doesn't preserve int2c_v as a named output).
            let int2c_v: Vec<f64> = {
                let (v, _): (Vec<f64>, Vec<usize>) =
                    cint_all.integrate_row_major("int2c2e", "s1", Some(aux_slc)).into();
                v
            };
            let int2c = if int2c_v.len() == naux * naux { int2c_v.clone() } else {
                let mut v = vec![0.0; naux * naux]; let mut idx = 0;
                for j in 0..naux { for i in 0..=j { v[i + j * naux] = int2c_v[idx]; v[j + i * naux] = int2c_v[idx]; idx += 1; }} v
            };
            let mut int2c_cm = vec![0.0; naux * naux];
            for p in 0..naux { for q in 0..naux { int2c_cm[p + q * naux] = int2c[p * naux + q]; }}
            (
                shared.int2c2e_ip1.clone(),
                int2c_v,
                int2c,
                int2c_cm,
                shared.vinv.clone(),
                shared.int3c2e.clone(),
            )
        } else {
            // int2c2e_ip1 on aux-only: derivative of the 2c metric (for auxbasis_response)
            let (i21, _): (Vec<f64>, Vec<usize>) =
                cint_all.integrate_row_major("int2c2e_ip1", "s1", Some(aux_slc)).into();
            let (int2c_v, _): (Vec<f64>, Vec<usize>) =
                cint_all.integrate_row_major("int2c2e", "s1", Some(aux_slc)).into();
            let int2c = if int2c_v.len() == naux * naux { int2c_v.clone() } else {
                let mut v = vec![0.0; naux * naux]; let mut idx = 0;
                for j in 0..naux { for i in 0..=j { v[i + j * naux] = int2c_v[idx]; v[j + i * naux] = int2c_v[idx]; idx += 1; }} v
            };
            let mut int2c_cm = vec![0.0; naux * naux];
            for p in 0..naux { for q in 0..naux { int2c_cm[p + q * naux] = int2c[p * naux + q]; }}
            let vinv = compute_vinv(&int2c_cm, naux);

            // 3c integrals
            let slc_3c: &[[usize; 2]] = &[[0, nreg], [0, nreg], [nreg, nreg + naux_shell]];
            let (t3c, _): (Vec<f64>, Vec<usize>) =
                cint_all.integrate_row_major("int3c2e", "s1", Some(slc_3c)).into();
            (i21, int2c_v, int2c, int2c_cm, vinv, t3c)
        };
        // ip1 (int3c2e_ip1) is no longer cached — the BLAS path computes it
        // per-atom from libcint directly (see H2b, H7, and the per-atom main
        // loop below). Only the Inline fallback needs the full [3,nao,nao,naux]
        // tensor (~116 MiB for C6H6), so compute it lazily only when that path
        // is active. In the default BLAS mode we leave it empty to save memory.
        let method_wj = std::env::var("REST_H1AO_METHOD").unwrap_or_default();
        let use_blas_wj = !method_wj.eq_ignore_ascii_case("inline");
        let slc_3c: &[[usize; 2]] = &[[0, nreg], [0, nreg], [nreg, nreg + naux_shell]];
        let ip1: Vec<f64> = if use_blas_wj {
            Vec::new()
        } else {
            let (v, _): (Vec<f64>, Vec<usize>) =
                cint_all.integrate_row_major("int3c2e_ip1", "s1", Some(slc_3c)).into();
            v
        };
        let _ = int2c_v; // (retained for downstream compat; no longer recomputed here)

        // SCF data
        let dm0_mat = &scf.density_matrix[0];
        let mut dm0 = vec![0.0; nao * nao];
        for r in 0..nao { for c in 0..nao { dm0[r * nao + c] = dm0_mat[[r, c]]; }}
        let c = &scf.eigenvectors[0];
        let mo_occ_data = &scf.occupation[0];
        let mut mc2 = vec![0.0; nao * nocc];
        for p in 0..nao { for i in 0..nocc { mc2[p * nocc + i] = c[[p, i]] * (mo_occ_data[i] as f64).sqrt(); }}

        // BLAS switch for calc_h1ao: REST_H1AO_METHOD=inline uses old for-loops,
        // default (or =blas) uses rstsr GEMM. Verified to machine precision vs inline.
        // (`use_blas_wj` is defined earlier, before the ip1 guard.)
        // Single DeviceBLAS instance reused by all BLAS blocks.
        // Creating DeviceBLAS::default() spawns a Rayon thread pool, so doing it
        // per-block (42× previously) was a major hidden cost.
        // Always create it (cheap if unused in inline mode) so all BLAS blocks
        // can share a single &DeviceBLAS reference.
        let device = DeviceBLAS::default();

        // rho0_Pij, rhoj0_P, rhok0_Pl_
        // H4: rho0_all[ia] = vinv · t3c_block  (GEMM [naux, naux] @ [naux, ni*nao])
        //   t3c row-major [(p0..p1), nao, naux]: element (ii, j, P) at (p0+ii)*nao*naux + j*naux + P
        //   block row-major [naux, ni, nao]: element (P, ii, j) at P*ni*nao + ii*nao + j  (reindex from t3c)
        //   coef = vinv · block: row-major [naux, ni*nao], coef[P, idx] = Σ_Q vinv[P,Q] · block[Q, idx]
        //   GEMM: vinv_t [naux, naux] @ block_t [naux, ni*nao] = coef_t [naux, ni*nao]
        //   vinv is col-major (from compute_vinv): vinv[P + Q*naux] = V^{-1}[P,Q]
        //     F-order [naux, naux]: element (P, Q) at P + Q*naux ✓ (vinv[P + Q*naux])
        //   block staged as F-order [naux, ni*nao]: element (P, ii*nao+j) at P + (ii*nao+j)*naux
        //   coef_t F-order [naux, ni*nao]: element (P, idx) at P + idx*naux
        //     = same layout as block_t, so coef can be read directly as row-major [naux, ni*nao]
        //     (row-major flat = P*ni*nao + idx; F-order flat = P + idx*naux; these differ!)
        //     Need to scatter coef_t to row-major for downstream code that reads rho0_all as row-major.
        let nao3 = nao * nao;
        let use_blas_h4 = use_blas_wj;
        let mut rho0_all: Vec<Vec<f64>> = Vec::with_capacity(natm);
        for ia in 0..natm {
            let p0 = aoslices[ia][2] as usize; let p1 = aoslices[ia][3] as usize; let ni = p1 - p0;
            // Reindex t3c to block row-major [naux, ni, nao]
            let mut block = vec![0.0; naux * ni * nao];
            for P in 0..naux { for ii in 0..ni { for j in 0..nao {
                block[P * ni * nao + ii * nao + j] = t3c[(p0+ii)*nao*naux + j*naux + P];
            }}}
            let mut coef = vec![0.0; naux * ni * nao];
            if use_blas_h4 {
                use rstsr::prelude::*;
                // vinv is already F-order [naux, naux] (element (P,Q) at P + Q*naux)
                let vinv_t = rt::asarray((&vinv, [naux, naux].f(), &device));
                // Stage block as F-order [naux, ni*nao]: element (P, ii*nao+j) at P + (ii*nao+j)*naux
                let mut block_stage = vec![0.0; naux * ni * nao];
                for p in 0..naux { for idx in 0..ni * nao {
                    block_stage[p + idx * naux] = block[p * ni * nao + idx];
                }}
                let block_t = rt::asarray((&block_stage, [naux, ni * nao].f(), &device));
                let coef_t = (&vinv_t % &block_t); // [naux, ni*nao] F-order
                let coef_raw = coef_t.into_shape(-1).into_raw();
                // Scatter to row-major [naux, ni, nao]
                for p in 0..naux { for idx in 0..ni * nao {
                    coef[p * ni * nao + idx] = coef_raw[p + idx * naux];
                }}
            } else {
                for P in 0..naux { for Q in 0..naux {
                    let vpq = vinv[P + Q * naux]; if vpq.abs() < 1e-15 { continue; }
                    for idx in 0..ni * nao { coef[P * ni * nao + idx] += vpq * block[Q * ni * nao + idx]; }
                }}
            }
            rho0_all.push(coef);
        }
        let mut rho0_full = vec![0.0; naux * nao3];
        let mut rhoj0_P = vec![0.0; naux];
        let mut rhok0_Pl_ = vec![0.0; naux * nao * nocc];
        // H8: rhoj0_P[P] += Σ_{ii,j} co[P,ii,j] * dm0[(p0+ii),j]  (gemv or GEMM [naux, ni*nao] @ [ni*nao, 1])
        // H6: rhok0_Pl_[P, (p0+ii), occ] += Σ_j co[P,ii,j] * mc2[j,occ]
        //   GEMM co [naux*ni, nao] @ mc2 [nao, nocc] = tmp [naux*ni, nocc]
        //   scatter-accumulate to rhok0_Pl_ at rows (p0+ii)
        let use_blas_h6h8 = use_blas_wj; // same env var REST_H1AO_METHOD
        for ia in 0..natm {
            let p0 = aoslices[ia][2] as usize; let p1 = aoslices[ia][3] as usize; let ni = p1 - p0;
            let co = &rho0_all[ia];
            // rho0_full scatter (data movement, no contraction)
            for P in 0..naux { for ii in 0..ni { for j in 0..nao {
                rho0_full[P * nao3 + (p0+ii)*nao + j] = co[P * ni * nao + ii * nao + j];
            }}}
            if use_blas_h6h8 {
                use rstsr::prelude::*;
                // ── H8: rhoj0_P += co · dm0_slice ──
                // co row-major [naux, ni, nao]: (P, ii, j) at P*ni*nao + ii*nao + j
                //   Stage as F-order [naux, ni*nao]: (P, ii*nao+j) at P + (ii*nao+j)*naux
                // dm0_slice: dm0 row-major [nao, nao], rows p0..p0+ni
                //   Stage as F-order [ni*nao, 1]: (ii*nao+j, 0) = dm0[(p0+ii)*nao + j]
                let mut co_stage = vec![0.0; naux * ni * nao];
                for p in 0..naux { for idx in 0..ni * nao {
                    co_stage[p + idx * naux] = co[p * ni * nao + idx];
                }}
                let co_t = rt::asarray((&co_stage, [naux, ni * nao].f(), &device));
                let mut dm0_slice_stage = vec![0.0; ni * nao];
                for ii in 0..ni { for j in 0..nao {
                    dm0_slice_stage[ii * nao + j] = dm0[(p0 + ii) * nao + j];
                }}
                let dm0_slice_t = rt::asarray((&dm0_slice_stage, [ni * nao, 1].f(), &device));
                let rhoj0_t = (&co_t % &dm0_slice_t); // [naux, 1] F-order
                let rhoj0_raw = rhoj0_t.into_shape(-1).into_raw();
                for p in 0..naux { rhoj0_P[p] += rhoj0_raw[p]; }

                // ── H6: rhok0_Pl_ += scatter(co @ mc2) ──
                // co staged as F-order [naux*ni, nao]: (P*ni+ii, j) at (P*ni+ii) + j*(naux*ni)
                //   source: co[P*ni*nao + ii*nao + j] = co row-major [naux, ni, nao]
                // mc2 row-major [nao, nocc]: (j, occ) at j*nocc + occ
                //   F-order [nao, nocc]: (j, occ) at j + occ*nao = NOT same as j*nocc+occ
                //   Stage mc2 as F-order [nao, nocc]: element (j, occ) at j + occ*nao
                let mut co_h6_stage = vec![0.0; naux * ni * nao];
                for p in 0..naux { for ii in 0..ni { for j in 0..nao {
                    co_h6_stage[(p * ni + ii) + j * (naux * ni)] = co[p * ni * nao + ii * nao + j];
                }}}
                let co_h6_t = rt::asarray((&co_h6_stage, [naux * ni, nao].f(), &device));
                let mut mc2_stage = vec![0.0; nao * nocc];
                for j in 0..nao { for occ in 0..nocc {
                    mc2_stage[j + occ * nao] = mc2[j * nocc + occ];
                }}
                let mc2_h6_t = rt::asarray((&mc2_stage, [nao, nocc].f(), &device));
                let rk_pl_t = (&co_h6_t % &mc2_h6_t); // [naux*ni, nocc] F-order
                let rk_pl_raw = rk_pl_t.into_shape(-1).into_raw();
                // Scatter-accumulate to rhok0_Pl_ row-major [naux, nao, nocc]:
                //   rhok0_Pl_[P, (p0+ii), occ] = rhok0_Pl_[P*nao*nocc + (p0+ii)*nocc + occ]
                //   from rk_pl_t[P*ni+ii, occ] = rk_pl_raw[(P*ni+ii) + occ*(naux*ni)]
                for p in 0..naux { for ii in 0..ni { for occ in 0..nocc {
                    rhok0_Pl_[p * nao * nocc + (p0 + ii) * nocc + occ] +=
                        rk_pl_raw[(p * ni + ii) + occ * (naux * ni)];
                }}}
            } else {
                for P in 0..naux { for ii in 0..ni { for j in 0..nao {
                    rhoj0_P[P] += co[P * ni * nao + ii * nao + j] * dm0[(p0+ii)*nao + j];
                }}}
                for P in 0..naux { for ii in 0..ni { for j in 0..nao { for occ in 0..nocc {
                    rhok0_Pl_[P * nao * nocc + (p0+ii)*nocc + occ] +=
                        co[P * ni * nao + ii * nao + j] * mc2[j * nocc + occ];
                }}}}
            }
        }

        // wj_ip1_pij per AO atom block (for auxbasis_response): Σ_p i21[x,q,p] * coef3c[p,i,j]
        //
        // BLAS implementation: per-atom GEMM.
        //   i21 staged once as F-order [3*naux, naux] (element (x*naux+q, p) = i21[x,q,p])
        //   rho0_all[ia] viewed as F-order [naux, ni*nao] (element (p, ii*nao+j) = co[p*ni*nao + ii*nao + j])
        //     — this is a zero-copy view since rho0_all is row-major [naux, ni, nao] = F-order [naux, ni*nao] transposed...
        //     actually row-major [naux, ni, nao] means element [p,ii,j] at p*ni*nao+ii*nao+j,
        //     which IS F-order [ni*nao, naux] transposed = F-order [naux, ni*nao] if we read p as the slow index.
        //     Wait: F-order [naux, ni*nao] flat index = p + (ii*nao+j)*naux. That's NOT the same as p*ni*nao+ii*nao+j.
        //     So rho0_all is row-major [naux, ni*nao], = C-order. To use as F-order [ni*nao, naux] we'd need .t().
        //     Simpler: stage rho0 as F-order [naux, ni*nao] explicitly.
        //   GEMM: i21_t [3*naux, naux] @ rho0_t [naux, ni*nao] = out_t [3*naux, ni*nao]
        //   out_t F-order element (x*naux+q, ii*nao+j) = Σ_p i21_t(x*naux+q, p) * rho0_t(p, ii*nao+j)
        //                                       = Σ_p i21[x,q,p] * rho0[p,ii,j] ✓
        //   scatter to row-major [naux, ni, 3, nao]: block[q*ni*3*nao + ii*3*nao + x*nao + j]
        //     = out_t_raw[(x*naux+q) + (ii*nao+j)*(3*naux)]
        let mut wj_ip1_pij: Vec<Vec<f64>> = Vec::with_capacity(natm);
        if self.flags.auxbasis_response {
            if use_blas_wj {
                use rstsr::prelude::*;
                // Stage i21 once as F-order [3*naux, naux]: element (x*naux+q, p) = i21[x,q,p]
                let mut i21_stage = vec![0.0; 3 * naux * naux];
                for x in 0..3 { for q in 0..naux { for p in 0..naux {
                    i21_stage[(x * naux + q) + p * (3 * naux)] =
                        i21[x * naux * naux + q * naux + p];
                }}}
                let i21_t = rt::asarray((&i21_stage, [3 * naux, naux].f(), &device));
                for ia in 0..natm {
                    let p0_a = aoslices[ia][2] as usize; let p1_a = aoslices[ia][3] as usize;
                    let ni = p1_a - p0_a;
                    let co = &rho0_all[ia]; // [naux * ni * nao], row-major [naux, ni, nao]
                    // Stage rho0 as F-order [naux, ni*nao]: element (p, ii*nao+j) = co[p*ni*nao + ii*nao + j]
                    let mut rho0_stage = vec![0.0; naux * ni * nao];
                    for p in 0..naux { for idx in 0..ni * nao {
                        rho0_stage[p + idx * naux] = co[p * ni * nao + idx];
                    }}
                    let rho0_t = rt::asarray((&rho0_stage, [naux, ni * nao].f(), &device));
                    let out_t = (&i21_t % &rho0_t); // [3*naux, ni*nao] F-order
                    let out_raw = out_t.into_shape(-1).into_raw();
                    // Scatter to row-major [naux, ni, 3, nao]
                    let mut block = vec![0.0; naux * ni * 3 * nao];
                    for x in 0..3 { for q in 0..naux { for ii in 0..ni { for j in 0..nao {
                        block[q * ni * 3 * nao + ii * 3 * nao + x * nao + j] =
                            out_raw[(x * naux + q) + (ii * nao + j) * (3 * naux)];
                    }}}}
                    wj_ip1_pij.push(block);
                }
            } else {
                for ia in 0..natm {
                    let p0 = aoslices[ia][2] as usize; let p1 = aoslices[ia][3] as usize; let ni = p1 - p0;
                    let co = &rho0_all[ia]; // [naux * ni * nao]
                    let mut block = vec![0.0; naux * ni * 3 * nao];
                    for q in 0..naux { for ii in 0..ni { for x in 0..3 { for j in 0..nao {
                        let mut s = 0.0;
                        for p in 0..naux {
                            s += i21[x * naux * naux + q * naux + p]
                                * co[p * ni * nao + ii * nao + j];
                        }
                        block[q * ni * 3 * nao + ii * 3 * nao + x * nao + j] = s;
                    }}}}
                    wj_ip1_pij.push(block);
                }
            }
        }

        // vj1_buf (per-atom) + vk1_buf (single, accumulated once)
        let mut vj1_buf = vec![0.0; natm * 3 * nao3];
        // H2a: rhok0_PlJ[P,l,J] = Σ_j rhok0_Pl_[P,l,j] * mc2[J,j]
        //   GEMM: rhok0_Pl_ staged F-order [naux*nao, nocc] @ mc2^T F-order [nocc, nao]
        //         = rhok0_PlJ F-order [naux*nao, nao]
        //   rhok0_Pl_ row-major [naux, nao, nocc]: (P,l,j) at P*nao*nocc + l*nocc + j
        //     F-order [naux*nao, nocc]: (P*nao+l, j) at (P*nao+l) + j*(naux*nao)
        //     = same as row-major flat! (because row-major [naux,nao,nocc] flat = P*nao*nocc+l*nocc+j,
        //       and F-order [naux*nao, nocc] flat = (P*nao+l) + j*(naux*nao) = P*nao+l + j*naux*nao)
        //     NOT the same. Need explicit staging.
        //   mc2 row-major [nao, nocc]: (J,j) at J*nocc + j
        //     Need mc2^T as F-order [nocc, nao]: (j, J) at j + J*nocc = J*nocc + j = row-major flat ✓
        //     So mc2 row-major [nao, nocc] = F-order [nocc, nao] transposed... actually
        //     F-order [nocc, nao] flat = j + J*nocc. row-major [nao,nocc] flat = J*nocc + j. Same! ✓
        //   rhok0_PlJ F-order [naux*nao, nao]: (P*nao+l, J) at (P*nao+l) + J*(naux*nao)
        //     scatter to row-major [naux, nao, nao]: (P,l,J) at P*nao² + l*nao + J
        //
        // H2b: vk1_buf[x,i,l] = Σ_{P,j} ip1[x,i,j,P] * rhok0_PlJ[P,l,j]
        //   Per x: GEMM ip1[x] [nao, nao*naux] @ rhok0_PlJ_reorder [nao*naux, nao] = vk1[x] [nao, nao]
        //   ip1[x] row-major [nao, nao, naux]: (i,j,P) at i*nao*naux + j*naux + P
        //     F-order [nao, nao*naux]: (i, j*naux+P) at i + (j*naux+P)*nao
        //     NOT same as row-major flat. Need staging.
        //   rhok0_PlJ row-major [naux, nao, nao]: (P,l,j) at P*nao² + l*nao + j
        //     Need F-order [nao*naux, nao] with rows (j,P), cols l:
        //       element (j*naux+P, l) at (j*naux+P) + l*(nao*naux)
        //   vk1[x] F-order [nao, nao]: (i, l) at i + l*nao
        //     scatter to vk1_buf row-major [3, nao, nao]: (x,i,l) at x*nao² + i*nao + l
        let use_blas_h2 = use_blas_wj; // same env var REST_H1AO_METHOD controls all H1-H8
        let mut rhok0_PlJ = vec![0.0; naux * nao * nao];
        let mut vk1_buf = vec![0.0; 3 * nao3];
        // Pre-stage rhok0_Pl_ once as F-order [naux*nao, nocc] — shared by H2a and H3a.
        // Previously H3a re-staged this (2.5M elements × natm = 30M useless copies for C6H6).
        let rk_pl_stage: Vec<f64> = if use_blas_h2 {
            let mut stage = vec![0.0; naux * nao * nocc];
            for p in 0..naux { for l in 0..nao { for j in 0..nocc {
                stage[(p * nao + l) + j * (naux * nao)] =
                    rhok0_Pl_[p * nao * nocc + l * nocc + j];
            }}}
            stage
        } else { Vec::new() };
        if use_blas_h2 {
            use rstsr::prelude::*;
            // ── H2a: rhok0_PlJ = rhok0_Pl_ @ mc2^T ── (uses pre-staged rk_pl_stage)
            let rk_pl_t = rt::asarray((&rk_pl_stage, [naux * nao, nocc].f(), &device));
            let mc2_t_h2 = rt::asarray((mc2.as_slice(), [nocc, nao].f(), &device));
            let plj_t = (&rk_pl_t % &mc2_t_h2); // [naux*nao, nao] F-order
            let plj_raw = plj_t.into_shape(-1).into_raw();
            for p in 0..naux { for l in 0..nao { for j_idx in 0..nao {
                rhok0_PlJ[p * nao * nao + l * nao + j_idx] =
                    plj_raw[(p * nao + l) + j_idx * (naux * nao)];
            }}}

            // ── H2b: vk1_buf[x,i,l] = Σ_{P,j} ip1[x,i,j,P] * rhok0_PlJ[P,l,j] ──
            // PySCF-style aux-blocked: compute int3c2e_ip1 per aux-atom-block,
            // contract immediately. Avoids loading 640MB global ip1 tensor.
            // For each aux atom block [ap0:ap1]:
            //   int3c_ip1_block[3, nao, nao, nq] computed fresh
            //   rhok0_PlJ_block = rhok0_PlJ[ap0:ap1, :, :]  (already computed)
            //   vk1_buf[x] += ip1_block[x] @ rhok0_PlJ_block_reord
            //
            // ip1_block[x, i, j, p_local] row-major: x*nao*nao*nq + i*nao*nq + j*nq + p_local
            // rhok0_PlJ_block[P_global, l, j] row-major: P_global*nao² + l*nao + j
            //   = rhok0_PlJ[(ap0+p_local)*nao² + l*nao + j]
            // vk1_buf[x,i,l] += Σ_{j, p_local} ip1_block[x,i,j,p_local] * rhok0_PlJ_block[(ap0+p_local),l,j]
            //
            // GEMM per x: ip1_block[x] [nao, nao*nq] @ plj_block_reord [nao*nq, nao] = [nao, nao]
            //   ip1_block[x] F-order [nao, nao*nq]: (i, j*nq+p_local) at i + (j*nq+p_local)*nao
            //   plj_block_reord F-order [nao*nq, nao]: (j*nq+p_local, l) at (j*nq+p_local) + l*(nao*nq)
            //     source: rhok0_PlJ[(ap0+p_local)*nao² + l*nao + j]
            for ia_aux in 0..natm {
                let aux_shl0 = auxslices[ia_aux][0] as usize;
                let aux_shl1 = auxslices[ia_aux][1] as usize;
                let ap0 = auxslices[ia_aux][2] as usize;
                let ap1 = auxslices[ia_aux][3] as usize;
                let nq = ap1 - ap0;
                if nq == 0 { continue; }
                // Compute int3c2e_ip1 for this aux block: [3, nao, nao, nq]
                let aux_block_slc: &[[usize; 2]] = &[
                    [0, nreg], [0, nreg],
                    [nreg + aux_shl0, nreg + aux_shl1],
                ];
                let (ip1_block, _): (Vec<f64>, Vec<usize>) =
                    cint_all.integrate_row_major("int3c2e_ip1", "s1", Some(aux_block_slc)).into();
                // ip1_block layout: [3, nao, nao, nq] row-major
                //   element (x, i, j, p_local) at x*nao*nao*nq + i*nao*nq + j*nq + p_local

                // Stage plj_block_reord as F-order [nao*nq, nao]: rows (j, p_local), cols l
                let mut plj_block_stage = vec![0.0; nao * nq * nao];
                for j in 0..nao { for p_loc in 0..nq { for l in 0..nao {
                    plj_block_stage[(j * nq + p_loc) + l * (nao * nq)] =
                        rhok0_PlJ[(ap0 + p_loc) * nao * nao + l * nao + j];
                }}}
                let plj_block_t = rt::asarray((&plj_block_stage, [nao * nq, nao].f(), &device));

                for x in 0..3 {
                    // Stage ip1_block[x] as F-order [nao, nao*nq]: cols (j, p_local)
                    let mut ip1_block_stage = vec![0.0; nao * nao * nq];
                    for i in 0..nao { for j in 0..nao { for p_loc in 0..nq {
                        ip1_block_stage[i + (j * nq + p_loc) * nao] =
                            ip1_block[x * nao * nao * nq + i * nao * nq + j * nq + p_loc];
                    }}}
                    let ip1_block_t = rt::asarray((&ip1_block_stage, [nao, nao * nq].f(), &device));
                    let vk1_block_t = (&ip1_block_t % &plj_block_t); // [nao, nao] F-order
                    let vk1_block_raw = vk1_block_t.into_shape(-1).into_raw();
                    // Accumulate to vk1_buf row-major [3, nao, nao]
                    for i in 0..nao { for l in 0..nao {
                        vk1_buf[x * nao3 + i * nao + l] += vk1_block_raw[i + l * nao];
                    }}
                }
            }
        } else {
            for P in 0..naux { for l in 0..nao { for J in 0..nao { let mut s = 0.0;
                for j in 0..nocc { s += rhok0_Pl_[P * nao * nocc + l * nocc + j] * mc2[J * nocc + j]; }
                rhok0_PlJ[P * nao * nao + l * nao + J] = s;
            }}}
            // vk1_buf: single aux partition (use full ip1), accumulate ONCE
            for x in 0..3 { for i in 0..nao { for l in 0..nao { let mut s = 0.0;
                for P in 0..naux { for j in 0..nao {
                    s += ip1[x * nao3 * naux + i * nao * naux + j * naux + P]
                        * rhok0_PlJ[P * nao * nao + l * nao + j];
                }} vk1_buf[x * nao3 + i * nao + l] += s;
            }}}
        }
        // Stage rho0_full once as F-order [naux, nao²] — it is constant across atoms.
        // Previously this was done per-atom (22.8M useless element copies for C4H6).
        // We store the staged data in a Vec that lives until end of function, and
        // create the TensorView inside each iteration (zero-copy from the Vec).
        let rho0f_stage: Vec<f64> = if use_blas_wj {
            let mut stage = vec![0.0; naux * nao3];
            for p in 0..naux { for idx in 0..nao3 {
                stage[p + idx * naux] = rho0_full[p * nao3 + idx];
            }}
            stage
        } else { Vec::new() };
        // ── H7: vj1_buf per-atom ──
        // PySCF-style: compute int3c2e_ip1 per atom (not global), contract immediately.
        // This avoids loading a 640MB global ip1 tensor and its cache-unfriendly staging.
        // ip1_atom[3, ni, nao, naux] = ∂(ii,jj|P)/∂x for ii in atom ia's shells
        for ia in 0..natm {
            let shl0 = aoslices[ia][0] as usize; let shl1 = aoslices[ia][1] as usize;
            let q0 = aoslices[ia][2] as usize; let q1 = aoslices[ia][3] as usize;
            let ni = q1 - q0;
            // Compute per-atom int3c2e_ip1 (same as ip1_a computed later in per-atom loop)
            let atom_slc_h7: &[[usize; 2]] = &[[shl0, shl1], [0, nreg], [nreg, nreg + naux_shell]];
            let (ip1_atom, _): (Vec<f64>, Vec<usize>) =
                cint_all.integrate_row_major("int3c2e_ip1", "s1", Some(atom_slc_h7)).into();
            // ip1_atom layout: [3, ni, nao, naux] row-major, element (x, ii, j, P) at
            //   x*ni*nao*naux + ii*nao*naux + j*naux + P
            // H7a: wj1[x,P] = Σ_{ii,j} ip1_atom[x,ii,j,P] * dm0[j, q0+ii]
            //   = Σ_{ii,j} ip1_atom[x,ii,j,P] * dm0[j*nao + (q0+ii)]
            // H7b: vj1_buf[ia][x,i,j] += Σ_P wj1[x,P] * rho0_full[P,i,j]
            let use_blas_h7 = use_blas_wj;
            let mut wj1 = vec![0.0; 3 * naux];
            if use_blas_h7 {
                use rstsr::prelude::*;
                // ── H7a: wj1 = ip1_atom^T · dm0_vec ──
                // ip1_atom staged as F-order [3*naux, ni*nao]:
                //   row (x*naux+P), col (ii*nao+j): element = ip1_atom[x*ni*nao*naux + ii*nao*naux + j*naux + P]
                let mut ip1_stage = vec![0.0; 3 * naux * ni * nao];
                for x in 0..3 { for p in 0..naux { for ii in 0..ni { for j in 0..nao {
                    ip1_stage[(x * naux + p) + (ii * nao + j) * (3 * naux)] =
                        ip1_atom[x * ni * nao * naux + ii * nao * naux + j * naux + p];
                }}}}
                let ip1_t = rt::asarray((&ip1_stage, [3 * naux, ni * nao].f(), &device));
                // dm0_vec: dm0[j, q0+ii] → F-order [ni*nao, 1]
                let mut dm0_vec = vec![0.0; ni * nao];
                for ii in 0..ni { for j in 0..nao {
                    dm0_vec[ii * nao + j] = dm0[j * nao + (q0 + ii)];
                }}
                let dm0_t = rt::asarray((&dm0_vec, [ni * nao, 1].f(), &device));
                let wj1_t = (&ip1_t % &dm0_t); // [3*naux, 1] F-order
                let wj1_raw = wj1_t.into_shape(-1).into_raw();
                for x in 0..3 { for p in 0..naux {
                    wj1[x * naux + p] = wj1_raw[x * naux + p];
                }}

                // ── H7b: vj1_buf[ia] += wj1 · rho0_full ──
                // Reuse pre-staged rho0f_stage (constant across atoms)
                let mut wj1_f_stage = vec![0.0; 3 * naux];
                for x in 0..3 { for p in 0..naux {
                    wj1_f_stage[x + p * 3] = wj1[x * naux + p];
                }}
                let wj1_f_t = rt::asarray((&wj1_f_stage, [3, naux].f(), &device));
                let rho0f_t = rt::asarray((&rho0f_stage, [naux, nao3].f(), &device));
                let vj1_t = (&wj1_f_t % &rho0f_t); // [3, nao²] F-order
                let vj1_raw = vj1_t.into_shape(-1).into_raw();
                let off = ia * 3 * nao3;
                for x in 0..3 { for idx in 0..nao3 {
                    vj1_buf[off + x * nao3 + idx] += vj1_raw[x + idx * 3];
                }}
            } else {
                for x in 0..3 { for P in 0..naux { let mut s = 0.0;
                    for ii in 0..ni { for j in 0..nao {
                        s += ip1_atom[x * ni * nao * naux + ii * nao * naux + j * naux + P] * dm0[j * nao + (q0 + ii)];
                    }} wj1[x * naux + P] = s;
                }}
                for x in 0..3 { for P in 0..naux { let wp = wj1[x * naux + P]; if wp.abs() < 1e-15 { continue; }
                    for i in 0..nao { for j in 0..nao {
                        vj1_buf[ia * 3 * nao3 + x * nao3 + i * nao + j] += wp * rho0_full[P * nao3 + i * nao + j];
                    }}
                }}
            }
        }

        // Save intermediates for debug comparison (reindex to MatrixFull column-major)
        // rhoj0_P: store as [naux, 1]
        self.result.insert("rhoj0_P".to_string(),
            MatrixFull::from_vec([naux, 1], rhoj0_P.clone()).unwrap());
        // vk1_buf: store as [3*nao, nao]
        {
            let mut vk1b_mf = vec![0.0; 3 * nao3];
            for x in 0..3 { for i in 0..nao { for j in 0..nao {
                let src = x * nao3 + i * nao + j;
                let dst = (x * nao + i) + j * 3 * nao;
                vk1b_mf[dst] = vk1_buf[src];
            }}}
            self.result.insert("vk1_buf".to_string(),
                MatrixFull::from_vec([3*nao, nao], vk1b_mf).unwrap());
        }
        // vj1_buf[ia]: per-atom raw buffer before any correction
        for ia in 0..natm {
            let off = ia * 3 * nao3;
            let mut vj1b_mf = vec![0.0; 3 * nao3];
            for x in 0..3 { for i in 0..nao { for j in 0..nao {
                let src = x * nao3 + i * nao + j;
                let dst = (x * nao + i) + j * 3 * nao;
                vj1b_mf[dst] = vj1_buf[off + src];
            }}}
            self.result.insert(format!("vj1_buf_{}", ia),
                MatrixFull::from_vec([3*nao, nao], vj1b_mf).unwrap());
        }

        // Per-atom: h1ao = hcore_deriv + vj1 - 0.5*vk1 + vxc_deriv1
        let _t_vxc_d1 = std::time::Instant::now();
        let vxc_d1: Option<Vec<Vec<f64>>> = if self.is_rks() {
            let xct = if mol.xc_data.use_density_gradient() {
                crate::dft::xc_deriv::XCType::GGA
            } else {
                crate::dft::xc_deriv::XCType::LDA
            };
            let v = crate::hessian::xc_hessian::vxc_deriv1_streaming(scf, xct);
            Some(v.iter().map(|m| m.iter().copied().collect()).collect())
        } else { None };
        self.h1ao.clear();
        for ia in 0..natm {
            let shl0 = aoslices[ia][0] as usize; let shl1 = aoslices[ia][1] as usize;
            let p0 = aoslices[ia][2] as usize; let p1 = aoslices[ia][3] as usize; let ni = p1 - p0;
            let off = ia * 3 * nao3;

            // vj1
            let atom_slc: &[[usize; 2]] = &[[shl0, shl1], [0, nreg], [nreg, nreg + naux_shell]];
            let (ip1_a, _): (Vec<f64>, Vec<usize>) =
                cint_all.integrate_row_major("int3c2e_ip1", "s1", Some(atom_slc)).into();
            let mut vj1 = vec![0.0; 3 * nao3];
            for i in 0..3*nao3 { vj1[i] = -vj1_buf[off + i]; }
            // Save neg_buf (before any rhoj0 correction, for debug comparison)
            let vj1_neg_buf = vj1.clone();
            // vj1[:,p0:p1] correction (rows, matching PySCF: only the differentiated-AO side)
            for x in 0..3 { for ii in 0..ni { for j in 0..nao { let mut s = 0.0;
                for P in 0..naux { s += ip1_a[x * ni * nao * naux + ii * nao * naux + j * naux + P] * rhoj0_P[P]; }
                vj1[x * nao3 + (p0+ii)*nao + j] -= s;
            }}}
            // vj1_presym: after row rhoj0 correction, before symmetrization (= PySCF's _gen_jk vj1)
            let vj1_presym = vj1.clone();
            // Save sub-components for debug comparison
            {
                let mut nbuf = vec![0.0; 3*nao3]; let mut pres = vec![0.0; 3*nao3];
                for x in 0..3 { for i in 0..nao { for j in 0..nao {
                    let src = x * nao3 + i * nao + j;
                    let dst = (x * nao + i) + j * 3 * nao;
                    nbuf[dst] = vj1_neg_buf[src];
                    pres[dst] = vj1_presym[src];
                }}}
                self.result.insert(format!("vj1_neg_buf_{}", ia), MatrixFull::from_vec([3*nao,nao],nbuf).unwrap());
                self.result.insert(format!("vj1_presym_{}", ia), MatrixFull::from_vec([3*nao,nao],pres).unwrap());
            }
            // ── auxbasis_response=1: per-atom corrections (before symmetrization) ──
            let mut ip2_a: Vec<f64> = Vec::new();
            let mut qi_aux: usize = 0;
            let mut q0_aux: usize = 0;
            let mut pij_all: Vec<f64> = Vec::new();
            if self.flags.auxbasis_response {
                let aux_shl0 = auxslices[ia][0] as usize;
                let aux_shl1 = auxslices[ia][1] as usize;
                q0_aux = auxslices[ia][2] as usize;
                let q1 = auxslices[ia][3] as usize;
                qi_aux = q1 - q0_aux;
                // int3c2e_ip2 for this aux atom: (ij|∂P/∂x)
                let ip2_slc: &[[usize; 2]] = &[[0, nreg], [0, nreg], [nreg + aux_shl0, nreg + aux_shl1]];
                let ip2_result: (Vec<f64>, Vec<usize>) =
                    cint_all.integrate_row_major("int3c2e_ip2", "s1", Some(ip2_slc)).into();
                ip2_a = ip2_result.0;
                // Build pij_all from wj_ip1_pij (equivalent to _load_dim0 for q0_aux:q1)
                pij_all = vec![0.0; qi_aux * nao * 3 * nao];
                for ia2 in 0..natm {
                    let i0 = aoslices[ia2][2] as usize;
                    let i1 = aoslices[ia2][3] as usize;
                    let ni2 = i1 - i0;
                    let block = &wj_ip1_pij[ia2];
                    for p_off in 0..qi_aux {
                        let p_global = q0_aux + p_off;
                        for i_off in 0..ni2 {
                            let i_global = i0 + i_off;
                            for x in 0..3 { for j in 0..nao {
                                pij_all[p_off * nao * 3 * nao + i_global * 3 * nao + x * nao + j] =
                                    block[p_global * ni2 * 3 * nao + i_off * 3 * nao + x * nao + j];
                            }}
                        }
                    }
                }
                // rhoj1[x,P] = Σ_{i,j} ip2[x,i,j,P] * dm0[j,i]
                let mut rhoj1 = vec![0.0; 3 * qi_aux];
                for x in 0..3 { for P in 0..qi_aux {
                    let mut s = 0.0;
                    for i in 0..nao { for j in 0..nao {
                        s += ip2_a[x * nao * nao * qi_aux + i * nao * qi_aux + j * qi_aux + P]
                            * dm0[j * nao + i];
                    }}
                    rhoj1[x * qi_aux + P] = s;
                }}
                // vj1 correction term 1: += -0.5 * Σ_P rho0_full[q0_aux+P,i,j] * rhoj1[x,P]
                for x in 0..3 { for i in 0..nao { for j in 0..nao {
                    let mut s = 0.0;
                    for P in 0..qi_aux {
                        s += rho0_full[(q0_aux + P) * nao3 + i * nao + j] * rhoj1[x * qi_aux + P];
                    }
                    vj1[x * nao3 + i * nao + j] -= 0.5 * s;
                }}}
                // vj1 correction term 2: += -0.5 * Σ_P ip2[x,i,j,P] * rhoj0_P[q0_aux+P]
                for x in 0..3 { for i in 0..nao { for j in 0..nao {
                    let mut s = 0.0;
                    for P in 0..qi_aux {
                        s += ip2_a[x * nao * nao * qi_aux + i * nao * qi_aux + j * qi_aux + P]
                            * rhoj0_P[q0_aux + P];
                    }
                    vj1[x * nao3 + i * nao + j] -= 0.5 * s;
                }}}
                // vj1 correction term 3: += +0.5 * Σ_{p,q} i21[x,q0_aux+p,q] * rhoj0_P[q] * rho0_full[q0_aux+p,i,j]
                // Factorized: the inner Σ_q is independent of (i,j).
                //   Step 1: temp[x, p_off] = Σ_q i21[x, q0_aux+p_off, q] * rhoj0_P[q]  (3×qi_aux×naux FLOPs)
                //   Step 2: vj1[x,i,j] += 0.5 * Σ_{p_off} temp[x,p_off] * rho0_full[q0_aux+p_off, i, j]  (3×nao²×qi_aux FLOPs)
                // This reduces ~212M FLOPs/atom to ~715K for C4H6 (~300x reduction).
                {
                    // Step 1: compute temp[x, p_off]
                    let mut temp = vec![0.0; 3 * qi_aux];
                    for x in 0..3 { for p_off in 0..qi_aux {
                        let mut s = 0.0;
                        for q in 0..naux {
                            s += i21[x * naux * naux + (q0_aux + p_off) * naux + q] * rhoj0_P[q];
                        }
                        temp[x * qi_aux + p_off] = s;
                    }}
                    // Step 2: vj1[x,i,j] += 0.5 * Σ_{p_off} temp[x,p_off] * rho0_full[q0_aux+p_off, i, j]
                    for x in 0..3 { for i in 0..nao { for j in 0..nao {
                        let mut s = 0.0;
                        for p_off in 0..qi_aux {
                            s += temp[x * qi_aux + p_off]
                                * rho0_full[(q0_aux + p_off) * nao3 + i * nao + j];
                        }
                        vj1[x * nao3 + i * nao + j] += 0.5 * s;
                    }}}
                }
                // vj1 correction term 4: += +0.5 * Σ_p pij_all[p,i,x,j] * rhoj0_P[q0_aux+p]
                for x in 0..3 { for i in 0..nao { for j in 0..nao {
                    let mut s = 0.0;
                    for p_off in 0..qi_aux {
                        s += pij_all[p_off * nao * 3 * nao + i * 3 * nao + x * nao + j]
                            * rhoj0_P[q0_aux + p_off];
                    }
                    vj1[x * nao3 + i * nao + j] += 0.5 * s;
                }}}
                // Save vj1_aux for debug comparison
                {
                    let mut aux_mat = vec![0.0; 3 * nao3];
                    for x in 0..3 { for i in 0..nao { for j in 0..nao {
                        let src = x * nao3 + i * nao + j;
                        let dst = (x * nao + i) + j * 3 * nao;
                        aux_mat[dst] = vj1[src];
                    }}}
                    self.result.insert(format!("vj1_aux_{}", ia),
                        MatrixFull::from_vec([3*nao, nao], aux_mat).unwrap());
                }
            }
            for x in 0..3 { for i in 0..nao { for j in (i+1)..nao {
                let a = x * nao3 + i * nao + j; let b = x * nao3 + j * nao + i;
                let sym = vj1[a] + vj1[b]; vj1[a] = sym; vj1[b] = sym;
            }} for i in 0..nao { vj1[x * nao3 + i * nao + i] *= 2.0; }}


            // vk1
            // H3a: rhok0_PlJ_a[P,l,J] = Σ_j rhok0_Pl_[P,l,j] * mc2[(p0+J),j]
            //   GEMM: rhok0_Pl_ [naux*nao, nocc] @ mc2_slice^T [nocc, ni] = rhok0_PlJ_a [naux*nao, ni]
            //   (rhok0_Pl_ staging same as H2a, mc2_slice needs staging for p0..p0+ni rows)
            // H3b: vk1[x,k,jj] = -Σ_{P,ii} ip1_a[x,ii,jj,P] * rhok0_PlJ_a[P,k,ii]
            //   Per x: GEMM ip1_a[x] [nao, ni*naux] @ rhok0_PlJ_a_reord [ni*naux, nao] = vk1[x] [nao, nao]
            let mut rhok0_PlJ_a = vec![0.0; naux * nao * ni];
            let mut vk1 = vec![0.0; 3 * nao3];
            if use_blas_h2 {
                use rstsr::prelude::*;
                // ── H3a ── (reuses pre-staged rk_pl_stage)
                let rk_pl_t = rt::asarray((&rk_pl_stage, [naux * nao, nocc].f(), &device));
                // Stage mc2_slice (rows p0..p0+ni) as F-order [nocc, ni]: element (j, J) = mc2[(p0+J)*nocc + j]
                let mut mc2_slice_stage = vec![0.0; nocc * ni];
                for j in 0..nocc { for jj in 0..ni {
                    mc2_slice_stage[j + jj * nocc] = mc2[(p0 + jj) * nocc + j];
                }}
                let mc2_slice_t = rt::asarray((&mc2_slice_stage, [nocc, ni].f(), &device));
                let plj_a_t = (&rk_pl_t % &mc2_slice_t); // [naux*nao, ni] F-order
                let plj_a_raw = plj_a_t.into_shape(-1).into_raw();
                // Scatter to row-major [naux, nao, ni]
                for p in 0..naux { for l in 0..nao { for jj in 0..ni {
                    rhok0_PlJ_a[p * nao * ni + l * ni + jj] =
                        plj_a_raw[(p * nao + l) + jj * (naux * nao)];
                }}}

                // ── H3b ──
                // Stage rhok0_PlJ_a as F-order [ni*naux, nao] with rows (ii,P), cols k:
                //   element (ii*naux+P, k) = rhok0_PlJ_a[P, k, ii] = rhok0_PlJ_a[P*nao*ni + k*ni + ii]
                let mut plj_a_reord = vec![0.0; ni * naux * nao];
                for ii in 0..ni { for p in 0..naux { for k in 0..nao {
                    plj_a_reord[(ii * naux + p) + k * (ni * naux)] =
                        rhok0_PlJ_a[p * nao * ni + k * ni + ii];
                }}}
                let plj_a_reord_t = rt::asarray((&plj_a_reord, [ni * naux, nao].f(), &device));
                    for x in 0..3 {
                    // Stage ip1_a[x] as F-order [nao, ni*naux] with cols (ii,P):
                    //   element (jj, ii*naux+P) = ip1_a[x,ii,jj,P] = ip1_a[x*ni*nao*naux + ii*nao*naux + jj*naux + P]
                    let mut ip1a_stage = vec![0.0; nao * ni * naux];
                    for jj in 0..nao { for ii in 0..ni { for p in 0..naux {
                        ip1a_stage[jj + (ii * naux + p) * nao] =
                            ip1_a[x * ni * nao * naux + ii * nao * naux + jj * naux + p];
                    }}}
                    let ip1a_t = rt::asarray((&ip1a_stage, [nao, ni * naux].f(), &device));
                    let vk1_t = (&ip1a_t % &plj_a_reord_t); // [nao, nao] F-order
                    let vk1_raw = vk1_t.into_shape(-1).into_raw();
                            // Scatter (with negation) to vk1 row-major [3, nao, nao]: (x,k,jj) at x*nao² + k*nao + jj
                    for k in 0..nao { for jj in 0..nao {
                        vk1[x * nao3 + k * nao + jj] = -vk1_raw[k + jj * nao];
                    }}
                }
            } else {
                for P in 0..naux { for l in 0..nao { for J in 0..ni { let mut s = 0.0;
                    for j in 0..nocc { s += rhok0_Pl_[P * nao * nocc + l * nocc + j] * mc2[(p0+J)*nocc + j]; }
                    rhok0_PlJ_a[P * nao * ni + l * ni + J] = s;
                }}}
                for x in 0..3 { for k in 0..nao { for jj in 0..nao { let mut s = 0.0;
                    for P in 0..naux { for ii in 0..ni {
                        s += ip1_a[x * ni * nao * naux + ii * nao * naux + jj * naux + P]
                            * rhok0_PlJ_a[P * nao * ni + k * ni + ii];
                    }} vk1[x * nao3 + k * nao + jj] = -s;
                }}}
            }
            let vk1_step1 = vk1.clone(); // before vk1_buf correction
            // vk1_buf correction (on rows, matching vk1_buf's layout)
            for x in 0..3 { for i in p0..p1 { for j in 0..nao {
                vk1[x * nao3 + i * nao + j] -= vk1_buf[x * nao3 + i * nao + j];
            }}}
            // Save pre-sym vk1 for debug comparison with PySCF _gen_jk
            let vk1_presym = vk1.clone();
            // ── auxbasis_response=1: vk1 corrections (before vk1 symmetrization) ──
            if self.flags.auxbasis_response && qi_aux > 0 {
                let q0 = q0_aux;
                // rhok0_PlJ[P,l,J] = Σ_j rhok0_Pl_[q0+P,l,j] * mc2[J,j]
                // GEMM: rhok0_Pl_slice [qi_aux*nao, nocc] @ mc2^T [nocc, nao] = rhok0_PlJ [qi_aux*nao, nao]
                //   rhok0_Pl_slice: rhok0_Pl_ rows [q0..q0+qi_aux], all l, all j
                //     F-order [qi_aux*nao, nocc]: (P*nao+l, j) at (P*nao+l) + j*(qi_aux*nao)
                //     source: rhok0_Pl_[(q0+P)*nao*nocc + l*nocc + j]
                let mut rhok0_PlJ = vec![0.0; qi_aux * nao * nao];
                {
                    use rstsr::prelude::*;
                    let mut rkp_stage = vec![0.0; qi_aux * nao * nocc];
                    for p in 0..qi_aux { for l in 0..nao { for j in 0..nocc {
                        rkp_stage[(p * nao + l) + j * (qi_aux * nao)] =
                            rhok0_Pl_[(q0 + p) * nao * nocc + l * nocc + j];
                    }}}
                    let rkp_t = rt::asarray((&rkp_stage, [qi_aux * nao, nocc].f(), &device));
                    let mc2_t_aux = rt::asarray((mc2.as_slice(), [nocc, nao].f(), &device));
                    let plj_t = (&rkp_t % &mc2_t_aux); // [qi_aux*nao, nao] F-order
                    let plj_raw = plj_t.into_shape(-1).into_raw();
                    for p in 0..qi_aux { for l in 0..nao { for j_idx in 0..nao {
                        rhok0_PlJ[p * nao * nao + l * nao + j_idx] =
                            plj_raw[(p * nao + l) + j_idx * (qi_aux * nao)];
                    }}}
                }
                // vk1 correction term 1: -= Σ_{P,j} rhok0_PlJ[P,l,j] * ip2[x,i,j,P]
                // GEMM per x: ip2_a[x] [nao, nao*qi_aux] @ rhok0_PlJ_reord [nao*qi_aux, nao] = [nao, nao]
                //   rhok0_PlJ_reord F-order [nao*qi_aux, nao]: (j*qi_aux+P, l) at (j*qi_aux+P) + l*(nao*qi_aux)
                //     source: rhok0_PlJ[P*nao² + l*nao + j]
                //   This staging is reused by term 2 (same rhok0_PlJ_reord).
                let mut plj_reord_aux = vec![0.0; nao * qi_aux * nao];
                for j in 0..nao { for p in 0..qi_aux { for l in 0..nao {
                    plj_reord_aux[(j * qi_aux + p) + l * (nao * qi_aux)] =
                        rhok0_PlJ[p * nao * nao + l * nao + j];
                }}}
                {
                    use rstsr::prelude::*;
                    let plj_reord_t = rt::asarray((&plj_reord_aux, [nao * qi_aux, nao].f(), &device));
                    // Term 1: vk1 -= ip2 @ plj_reord
                    for x in 0..3 {
                        let mut ip2_stage = vec![0.0; nao * nao * qi_aux];
                        for i in 0..nao { for j in 0..nao { for p in 0..qi_aux {
                            ip2_stage[i + (j * qi_aux + p) * nao] =
                                ip2_a[x * nao * nao * qi_aux + i * nao * qi_aux + j * qi_aux + p];
                        }}}
                        let ip2_t = rt::asarray((&ip2_stage, [nao, nao * qi_aux].f(), &device));
                        let vk1_corr_t = (&ip2_t % &plj_reord_t);
                        let vk1_corr_raw = vk1_corr_t.into_shape(-1).into_raw();
                        for i in 0..nao { for l in 0..nao {
                            vk1[x * nao3 + i * nao + l] -= vk1_corr_raw[i + l * nao];
                        }}
                    }
                    // Term 2: vk1 += pij @ plj_reord (reuse plj_reord_t)
                    for x in 0..3 {
                        let mut pij_stage = vec![0.0; nao * nao * qi_aux];
                        for i in 0..nao { for j in 0..nao { for p in 0..qi_aux {
                            pij_stage[i + (j * qi_aux + p) * nao] =
                            pij_all[p * nao * 3 * nao + j * 3 * nao + x * nao + i];
                        }}}
                        let pij_t = rt::asarray((&pij_stage, [nao, nao * qi_aux].f(), &device));
                        let vk1_corr_t = (&pij_t % &plj_reord_t);
                        let vk1_corr_raw = vk1_corr_t.into_shape(-1).into_raw();
                        for i in 0..nao { for l in 0..nao {
                            vk1[x * nao3 + i * nao + l] += vk1_corr_raw[i + l * nao];
                        }}
                    }
                }
                // Save vk1_aux for debug comparison
                {
                    let mut aux_mat = vec![0.0; 3 * nao3];
                    for x in 0..3 { for i in 0..nao { for j in 0..nao {
                        let src = x * nao3 + i * nao + j;
                        let dst = (x * nao + i) + j * 3 * nao;
                        aux_mat[dst] = vk1[src];
                    }}}
                    self.result.insert(format!("vk1_aux_{}", ia),
                        MatrixFull::from_vec([3*nao, nao], aux_mat).unwrap());
                }
            }
            for x in 0..3 { for i in 0..nao { for j in (i+1)..nao {
                let a = x * nao3 + i * nao + j; let b = x * nao3 + j * nao + i;
                let sym = vk1[a] + vk1[b]; vk1[a] = sym; vk1[b] = sym;
            }} for i in 0..nao { vk1[x * nao3 + i * nao + i] *= 2.0; }}


            // hcore_deriv
            let h1 = build_hcore_first_deriv(mol, ia);

            // h1ao = h1 + factor_j*vj1 - 0.5*factor_k*vk1 plus debug intermediates
            // (factor_k defaults to 1.0 for RHF; RKS wrapper sets it to hyb.)
            // Reindex from [x][i][j] layout to MatrixFull column-major
            let factor_j = self.flags.factor_j.unwrap_or(1.0);
            let factor_k = self.flags.factor_k.unwrap_or(1.0);
            let mut h1ao_mat = vec![0.0; 3 * nao3];
            let mut vj1_mat = vec![0.0; 3 * nao3];
            let mut vk1_mat = vec![0.0; 3 * nao3];
            let mut vj1p_mat = vec![0.0; 3 * nao3];
            let mut vk1p_mat = vec![0.0; 3 * nao3];
            let mut vj1s1_mat = vec![0.0; 3 * nao3];
            let mut vk1s1_mat = vec![0.0; 3 * nao3];
            for x in 0..3 { for i in 0..nao { for j in 0..nao {
                let src = x * nao3 + i * nao + j;
                let dst = (x * nao + i) + j * 3 * nao;
                let vxc1_val = vxc_d1.as_ref().map_or(0.0, |v| v[ia][dst]);
                h1ao_mat[dst] = h1[src] + factor_j * vj1[src] - 0.5 * factor_k * vk1[src] + vxc1_val;
                vj1_mat[dst] = vj1[src];         // post-sym
                vk1_mat[dst] = vk1[src];         // post-sym
                vj1p_mat[dst] = vj1_presym[src];  // pre-sym (= after rhoj0 corr)
                vk1p_mat[dst] = vk1_presym[src];  // pre-sym (= after vk1_buf corr)
                vj1s1_mat[dst] = vj1_neg_buf[src];  // after -vj1_buf only (no rhoj0 corr)
                vk1s1_mat[dst] = vk1_step1[src];  // after -ip1@rhok0 only
            }}}
            self.h1ao.push(MatrixFull::from_vec([3 * nao, nao], h1ao_mat).unwrap());
            self.result.insert(format!("vj1_{}", ia), MatrixFull::from_vec([3 * nao, nao], vj1_mat).unwrap());
            self.result.insert(format!("vk1_{}", ia), MatrixFull::from_vec([3 * nao, nao], vk1_mat).unwrap());
            self.result.insert(format!("vj1_presym_{}", ia), MatrixFull::from_vec([3 * nao, nao], vj1p_mat).unwrap());
            self.result.insert(format!("vk1_presym_{}", ia), MatrixFull::from_vec([3 * nao, nao], vk1p_mat).unwrap());
            self.result.insert(format!("vj1_neg_buf_{}", ia), MatrixFull::from_vec([3 * nao, nao], vj1s1_mat).unwrap());
            self.result.insert(format!("vk1_step1_{}", ia), MatrixFull::from_vec([3 * nao, nao], vk1s1_mat).unwrap());
        }
        // Save int2c_ip1 for debug comparison
        {
            let mut i21_mf = vec![0.0; 3 * naux * naux];
            for x in 0..3 { for p in 0..naux { for q in 0..naux {
                let src = x * naux * naux + p * naux + q;
                let dst = (x * naux + p) + q * 3 * naux;
                i21_mf[dst] = i21[src];
            }}}
            self.result.insert("int2c_ip1".to_string(),
                MatrixFull::from_vec([3 * naux, naux], i21_mf).unwrap());
        }
        if self.is_rks() {
            self.timings.push(("  rks: vxc_deriv1", _t_vxc_d1.elapsed()));
        }
        // shared_integrals (int2c2e_ip1, vinv, int3c2e, int3c2e_ip1) were
        // shared from calc_ej_ek to avoid re-running libcint here. They are
        // not needed by any later stage (calc_cphf_contrib, calc_hess_nuc),
        // so free them now instead of carrying them until RIRHFHessian drops.
        self.shared_integrals = None;
        self.timings.push(("calc_h1ao", _t.elapsed()));
        self
    }


    /// Solve CP-HF for each perturbed atom and compute the CP-HF contribution
    /// to the electronic Hessian: de2 += 4·h1ao·dm1 (then - s1·dm1_e, - s1oo·mo_e1).
    ///
    /// Requires `calc_h1ao()` to have been called first.
    /// Stores result in `self.result["cphf_contrib"]` as MatrixFull [n3, n3].
    pub fn calc_cphf_contrib(&mut self) -> &mut Self {
        let _t = std::time::Instant::now();
        use crate::ri_cphf::{
            CPHFSolverPySCF, transform_h1ao_ao2mo,
            build_s1ao_deriv, transform_s1ao_ao2mo,
        };
        use tensors::matrix_blas_lapack::_dgemm_full;

        let scf = self.scf_data;
        let mol = &scf.mol;
        let nao = mol.num_basis;
        let natm = mol.geom.nfree;
        let n3 = natm * 3;
        let aa9 = natm * natm * 9;

        if self.h1ao.len() != natm {
            panic!("calc_cphf_contrib: call calc_h1ao() first (h1ao.len={})", self.h1ao.len());
        }

        // Use full occupation (no frozen core) to match PySCF's kernel() behavior.
        // calc_h1ao and calc_ej_ek both use nocc = homo+1 (full occupation), so
        // the CP-HF contribution must also use full occupation for consistency.
        let solver = CPHFSolverPySCF::new_full(scf);
        let nmo = solver.nmo;
        let nocc = solver.nocc;
        let start_mo = solver.start_mo;
        let c_mo = &solver.c_mo;

        // Extract C_occ [nao, nocc] from full MO coefficients
        let mut c_occ = MatrixFull::new([nao, nocc], 0.0);
        for p in 0..nao { for i in 0..nocc {
            c_occ[[p, i]] = c_mo[[p, start_mo + i]];
        }}

        // Occupied energies
        let mut eps_occ = vec![0.0; nocc];
        for i in 0..nocc {
            eps_occ[i] = solver.mo_energy[start_mo + i];
        }

        let s1_zero = vec![0.0; nmo * nocc];

        // ── Step 1: Solve CP-HF per atom, per direction ──
        use crate::dft::response::{reset_fxc_timing, read_fxc_timing_s, read_fxc_subtimings_s,
            prepare_fxc_hessian_cache, compute_fxc_response_ao_cached, FxcHessianCache};
        reset_fxc_timing();
        // Build the fxc kernel cache ONCE for the entire CP-HF phase (PySCF
        // `cache_xc_kernel` analog). This amortises AO/ρ₀/libxc-fxc evaluation
        // across all Krylov matvecs (~100s for H₂O, ~1000s for larger systems).
        // `is_dft` decides whether to populate the cache; HF passes None.
        let is_hf_for_cache = scf.mol.xc_data.dfa_compnt_scf.is_empty();
        let _t_cache = std::time::Instant::now();
        let fxc_cache: Option<FxcHessianCache> = if is_hf_for_cache {
            None
        } else {
            Some(prepare_fxc_hessian_cache(scf))
        };
        let fxc_cache_ref = fxc_cache.as_ref();
        let _t_cache_elapsed = _t_cache.elapsed();
        let _t_solve = std::time::Instant::now();
        let mut mo1_all: Vec<Vec<Vec<f64>>> = vec![vec![vec![]; 3]; natm];
        let mut s1ao_all: Vec<Vec<Vec<f64>>> = vec![vec![vec![]; 3]; natm];
        let mut s1_mo_all: Vec<Vec<Vec<f64>>> = vec![vec![vec![]; 3]; natm];
        // Cache h1_mo per atom (3 directions each). Avoids recomputing
        // transform_h1ao_ao2mo in the mo_e1 loop below.
        let mut h1_mo_all: Vec<Vec<Vec<f64>>> = vec![vec![vec![]; 3]; natm];

        // Solver selection: override via REST_CPHF_METHOD env var (default "krylov").
        // REST_CPHF_COMPARE=1 → for atom 0, dir 0, also run dense and print maxdiff.
        let cphf_method = std::env::var("REST_CPHF_METHOD").unwrap_or_default();
        let use_krylov = cphf_method.eq_ignore_ascii_case("krylov") || cphf_method.is_empty();
        let do_compare = std::env::var("REST_CPHF_COMPARE").is_ok();
        let krylov_max_cycle: usize = std::env::var("REST_CPHF_KRYLOV_MAXCYCLE")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(50);
        let krylov_tol: f64 = std::env::var("REST_CPHF_KRYLOV_TOL")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(1e-8);
        if use_krylov {
            println!("  CP-HF: using Krylov solver (max_cycle={}, tol={:.1e})",
                     krylov_max_cycle, krylov_tol);
        } else {
            println!("  CP-HF: using dense solver");
        }

        // ════════════════════════════════════════════════════════════════
        // Phase A: Batched CP-HF Krylov solve
        //
        // Build all 3*natom RHS upfront (including OO correction), then solve
        // them simultaneously via `solve_krylov_batched`. Each Krylov cycle
        // dispatches ONE batched matvec across all active RHS (instead of
        // 3*natom separate matvec calls per cycle), amortizing rayon fork/
        // join and AO-cache loads.
        // ════════════════════════════════════════════════════════════════
        let n_pert = 3 * natm;
        let mut rhs_all: Vec<Vec<f64>> = Vec::with_capacity(n_pert);
        let mut rhs_meta: Vec<(usize, usize)> = Vec::with_capacity(n_pert);

        // ── First pass: build RHS, cache h1_mo / s1ao / s1_mo per (ia, dir) ──
        for ia in 0..natm {
            let h1_mo = transform_h1ao_ao2mo(&solver, &self.h1ao[ia]);
            let s1ao_ia = build_s1ao_deriv(mol, ia);
            let s1_mo_ia = transform_s1ao_ao2mo(&solver, &s1ao_ia);
            for dir in 0..3 { h1_mo_all[ia][dir] = h1_mo[dir].clone(); }

            for dir in 0..3 {
                s1ao_all[ia][dir] = s1ao_ia[dir].clone();
                s1_mo_all[ia][dir] = s1_mo_ia[dir].clone();
                let rhs = if use_krylov {
                    // Includes OO correction (one fvind call per RHS).
                    solver.build_rhs_with_oo_correction(
                        scf, fxc_cache_ref, &h1_mo[dir], &s1_mo_ia[dir])
                } else {
                    // Dense path builds its own RHS internally; store a sentinel.
                    Vec::new()
                };
                rhs_all.push(rhs);
                rhs_meta.push((ia, dir));
            }
        }

        // ── Second pass: solve (batched Krylov or per-RHS dense fallback) ──
        let mo1_full_all: Vec<Vec<f64>> = if use_krylov {
            // Single batched solve for all 3*natom RHS.
            let u_vo_all = solver.solve_krylov_batched(
                scf, fxc_cache_ref, &rhs_all, krylov_max_cycle, krylov_tol);
            // Assemble full (nmo*nocc) solution per RHS from VO + OO blocks.
            rhs_meta.iter().zip(u_vo_all.iter()).map(|(&(ia, dir), u_vo)| {
                let u_oo = solver.solve_occ_occ_from_s1(&s1_mo_all[ia][dir]);
                solver.assemble_full_solution(u_vo, &u_oo)
            }).collect()
        } else {
            rhs_meta.iter().map(|&(ia, dir)| {
                solver.solve_dense(scf, fxc_cache_ref,
                    &h1_mo_all[ia][dir], &s1_mo_all[ia][dir])
                    .expect("CP-HF dense solve failed")
            }).collect()
        };

        // ── Third pass: distribute to mo1_all + preserve debug output ──
        for (k, &(ia, dir)) in rhs_meta.iter().enumerate() {
            let mo1_full = mo1_full_all[k].clone();
            mo1_all[ia][dir] = mo1_full.clone();

            // Debug: dense-vs-krylov comparison for atom 0, dir 0.
            if use_krylov && do_compare && ia == 0 && dir == 0 {
                let mo1_dense = solver.solve_dense(scf, fxc_cache_ref,
                    &h1_mo_all[ia][dir], &s1_mo_all[ia][dir])
                    .expect("CP-HF dense solve failed (comparison)");
                let mut md = 0.0f64;
                let mut md_vo = 0.0f64;
                let nmo = solver.nmo; let nocc = solver.nocc;
                let lumo = solver.lumo; let start_mo = solver.start_mo;
                for i in 0..mo1_full.len() {
                    let d = (mo1_full[i] - mo1_dense[i]).abs();
                    if d > md { md = d; }
                    let row = i % nmo;
                    let col = i / nmo;
                    if row >= lumo && row < nmo && col >= start_mo && col < start_mo + nocc {
                        if d > md_vo { md_vo = d; }
                    }
                }
                println!("  DEBUG dense-vs-krylov mo1[0][0]: maxdiff={:.4e}, VO_maxdiff={:.4e}",
                         md, md_vo);
                self.result.insert("mo1_dense_00".to_string(),
                    MatrixFull::from_vec([mo1_dense.len(), 1], mo1_dense).unwrap());
            }

            // Save h1_mo, s1_mo, mo1 for atom 0, direction 0 for debug comparison.
            if ia == 0 && dir == 0 {
                let nmo = solver.nmo; let nocc = solver.nocc;
                self.result.insert("h1_mo_00".to_string(),
                    MatrixFull::from_vec([nmo * nocc, 1], h1_mo_all[ia][dir].clone()).unwrap());
                self.result.insert("s1_mo_00".to_string(),
                    MatrixFull::from_vec([nmo * nocc, 1], s1_mo_all[ia][dir].clone()).unwrap());
                {
                    let s1ao_flat: Vec<f64> = (0..nao*nao).map(|i| s1ao_all[ia][dir][i]).collect();
                    let mut mx = 0.0;
                    for &v in &s1ao_flat { let a = v.abs(); if a > mx { mx = a; }}
                    println!("  DEBUG s1ao[0] max_abs={:.4e}", mx);
                    self.result.insert("s1ao_00".to_string(),
                        MatrixFull::from_vec([nao, nao], s1ao_flat).unwrap());
                }
                self.result.insert("mo1_full_00".to_string(),
                    MatrixFull::from_vec([nmo * nocc, 1], mo1_full.clone()).unwrap());
                let mut rhs_full = vec![0.0; nmo * nocc];
                for col in 0..nocc {
                    let e_occ = solver.mo_energy[solver.start_mo + col];
                    for a in 0..solver.nvir {
                        let row = solver.lumo + a;
                        let nmc_idx = row + col * nmo;
                        let h1v = h1_mo_all[ia][dir][nmc_idx];
                        let s1v = s1_mo_all[ia][dir][nmc_idx];
                        let ai_idx = col + a * nocc;
                        rhs_full[nmc_idx] = -(h1v - s1v * e_occ) * solver.e_ai[ai_idx];
                    }
                }
                self.result.insert("rhs_full_00".to_string(),
                    MatrixFull::from_vec([nmo * nocc, 1], rhs_full).unwrap());
            }
        }

        self.timings.push(("  cphf: fxc_cache", _t_cache_elapsed));
        self.timings.push(("  cphf: solve", _t_solve.elapsed()));
        let fxc_solve_ns = (read_fxc_timing_s() * 1e9) as u64;
        self.timings.push(("  cphf: fxc_solve", std::time::Duration::from_nanos(fxc_solve_ns)));
        // Detailed fxc sub-component breakdown (matches PySCF's nr_rks_fxc internals).
        for (name, secs) in read_fxc_subtimings_s() {
            if secs > 0.0 {
                let label: &'static str = Box::leak(
                    format!("    {}", name).into_boxed_str());
                self.timings.push((label, std::time::Duration::from_secs_f64(secs)));
            }
        }

        // Also compute mo_e1 (first-order orbital energy correction) for each atom/direction
        // mo_e1 = (h1 - s1*e_i + fvind(mo1))_oo + mo1_oo*(e_i - e_i)
        reset_fxc_timing();
        let _t_mo_e1 = std::time::Instant::now();
        let mut mo_e1_all: Vec<Vec<Vec<f64>>> = vec![vec![vec![]; 3]; natm];
        {
            use crate::dft::response::{VindWorkspace, compute_j_upper, compute_k_upper};
            use tensors::matrix_blas_lapack::_dgemm_full;

            let ws2 = VindWorkspace::new(scf, nocc, solver.nvir, start_mo, solver.lumo);
            // RKS: hybrid exchange scaling (0.5*hyb for closed-shell). For HF
            // (no DFA components), hyb=1.0 so k_scaling=0.5, recovering the
            // historical J - 0.5*K response.
            let is_hf = scf.mol.xc_data.dfa_compnt_scf.is_empty();
            let hyb_e1 = if is_hf { 1.0 } else { scf.mol.xc_data.dfa_hybrid_scf };
            let k_scaling_e1 = 0.5 * hyb_e1;
            for ia in 0..natm {
                for dir in 0..3 {
                    let mo1_full = &mo1_all[ia][dir];
                    // Reuse cached h1_mo and s1_mo from the CP-HF solve phase
                    // (avoids 3*natm redundant calls to transform_h1ao_ao2mo
                    //  and 3*natm redundant calls to transform_s1ao_ao2mo).
                    let h1_mo = &h1_mo_all[ia];
                    let s1_mo_ia = &s1_mo_all[ia];

                    // Build dm1 from the FULL mo1 solution
                    // dm1 = 2*C_full @ mo1 @ C_occ^T + 2*C_occ @ mo1^T @ C_full^T
                    let mut x = MatrixFull::new([nmo, nocc], 0.0);
                    for col in 0..nocc { for row in 0..nmo {
                        x[[row, col]] = 2.0 * mo1_full[row + col * nmo];  // *2 for RHF
                    }}
                    // dm = C_full @ (2*x) @ C_occ^T
                    let mut t1 = MatrixFull::new([nao, nocc], 0.0);
                    _dgemm_full(&solver.c_mo, 'N', &x, 'N', &mut t1, 1.0, 0.0);
                    let mut dm = MatrixFull::new([nao, nao], 0.0);
                    _dgemm_full(&t1, 'N', &c_occ, 'T', &mut dm, 1.0, 0.0);
                    // dm1 = dm + dm^T (proper symmetric)
                    let mut dm1 = MatrixFull::new([nao, nao], 0.0);
                    for i in 0..nao { for j in 0..nao { dm1[[i, j]] = dm[[i, j]] + dm[[j, i]]; }}

                    // Compute J and K
                    let dm_vec = vec![dm1.clone()];
                    let j_full = compute_j_upper(scf, &dm_vec).to_matrixfull().unwrap();
                    let k_full = compute_k_upper(scf, &dm_vec).to_matrixfull().unwrap();
                    let mut v_ao = MatrixFull::new([nao, nao], 0.0);
                    for p in 0..nao { for q in 0..nao {
                        v_ao[[p, q]] = j_full[[p, q]] - k_scaling_e1 * k_full[[p, q]];
                    }}
                    // Add fxc response for RKS (matches PySCF `gen_rks_response`
                    // with `singlet=None`: vind(dm1) = J - 0.5*hyb*K + nr_rks_fxc(dm1)).
                    // Uses the precomputed `FxcHessianCache` — no AO/ρ₀/libxc
                    // re-evaluation here.
                    if let Some(cache) = fxc_cache_ref {
                        let fxc_ao = compute_fxc_response_ao_cached(cache, &dm1);
                        for p in 0..nao { for q in 0..nao {
                            v_ao[[p, q]] += fxc_ao[[p, q]];
                        }}
                    }

                    // Project to OO: C_occ^T @ v_ao @ C_occ
                    let mut tmp_oo = MatrixFull::new([nao, nocc], 0.0);
                    _dgemm_full(&v_ao, 'N', &c_occ, 'N', &mut tmp_oo, 1.0, 0.0);
                    let mut fvind_oo = MatrixFull::new([nocc, nocc], 0.0);
                    _dgemm_full(&c_occ, 'T', &tmp_oo, 'N', &mut fvind_oo, 1.0, 0.0);

                    // mo_e1[i,j] = (h1 - s1*e_i)[occ,occ] + fvind_oo + mo1_oo * (e_i[:,None] - e_i)
                    let mut mo_e1 = vec![0.0; nocc * nocc];
                    for j in 0..nocc { for i in 0..nocc {
                        let e_j = eps_occ[j];
                        let e_i_i = eps_occ[i];
                        let h1s1 = h1_mo[dir][(start_mo + i) + j * nmo]
                                  - s1_mo_ia[dir][(start_mo + i) + j * nmo] * e_j;
                        let f_oo = fvind_oo[[i, j]];
                        let mo1_oo_ij = mo1_full[(start_mo + i) + j * nmo];
                        mo_e1[i + j * nocc] = h1s1 + f_oo + mo1_oo_ij * (e_i_i - e_j);
                    }}
                    mo_e1_all[ia][dir] = mo_e1;
                }
            }
        }

        self.timings.push(("  cphf: mo_e1", _t_mo_e1.elapsed()));
        let fxc_moe1_ns = (read_fxc_timing_s() * 1e9) as u64;
        self.timings.push(("  cphf: fxc_moe1", std::time::Duration::from_nanos(fxc_moe1_ns)));

        // ── Step 2: Contract mo1[ja] with h1ao[ia] → CP-HF contribution ──
        let _t_contract = std::time::Instant::now();
        // de2 is in flat (natm, natm, 3, 3) format matching h_partial
        let mut cphf = vec![0.0; aa9];
        let i_t = |i0: usize, j0: usize, x: usize, y: usize| -> usize {
            i0 * natm * 9 + j0 * 9 + x * 3 + y
        };

        // Pre-compute s1oo[ia][dir] = C_occ^T @ s1ao[ia][dir] @ C_occ (nocc, nocc)
        let mut s1oo_all: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; nocc * nocc]; 3]; natm];
        for ia in 0..natm {
            for dx in 0..3 {
                let s1_flat = &s1ao_all[ia][dx];
                let mut s1_mat = MatrixFull::new([nao, nao], 0.0);
                for i in 0..nao { for j in 0..nao { s1_mat[[i, j]] = s1_flat[i * nao + j]; }}
                let mut tmp = MatrixFull::new([nao, nocc], 0.0);
                _dgemm_full(&s1_mat, 'N', &c_occ, 'N', &mut tmp, 1.0, 0.0);
                let mut s1oo = MatrixFull::new([nocc, nocc], 0.0);
                _dgemm_full(&c_occ, 'T', &tmp, 'N', &mut s1oo, 1.0, 0.0);
                for i in 0..nocc { for j in 0..nocc { s1oo_all[ia][dx][i + j * nocc] = s1oo[[i, j]]; }}
            }
        }

        // Pre-allocate workspace matrices (reused across all i0, j0, dir_y iterations).
        // Replaces the per-iteration naive triple loops with BLAS dgemm.
        let mut mo1_ao_mat = MatrixFull::new([nao, nocc], 0.0);
        let mut dm1_mat = MatrixFull::new([nao, nao], 0.0);
        let mut dm1_e_mat = MatrixFull::new([nao, nao], 0.0);
        // Precompute energy-weighted C_occ: c_occ_e[p,i] = c_occ[p,i] * eps_occ[i].
        // Then dm1_e = mo1_ao @ c_occ_e^T (single GEMM, no per-element scaling).
        let mut c_occ_e = MatrixFull::new([nao, nocc], 0.0);
        for i in 0..nocc {
            let e = eps_occ[i];
            for p in 0..nao { c_occ_e[[p, i]] = c_occ[[p, i]] * e; }
        }
        // Slice metadata for borrowing mo1_full as a MatrixFullSlice without cloning.
        let mo1_sz = [nmo, nocc];
        let mo1_ind = [1usize, nmo];

        // Follow PySCF: compute de2[i0,j0] for j0 <= i0 only,
        // then fill de2[j0,i0] = de2[i0,j0].T by symmetry.
        for i0 in 0..natm {
            let ia = i0;
            for j0 in 0..=i0 {
                let ja = j0;
                let mo1_full_ja = &mo1_all[ja];
                let s1ao_ia = &s1ao_all[ia];

                for dir_y in 0..3 {
                    // Back-transform mo1[ja][dir_y]: MO → AO via BLAS dgemm.
                    // mo1_ao[nao, nocc] = c_mo[nao, nmo] @ mo1_full[nmo, nocc]
                    let mo1_full = &mo1_full_ja[dir_y];
                    let mo1_full_slice = tensors::matrix::matrixfullslice::MatrixFullSlice {
                        size: &mo1_sz, indicing: &mo1_ind, data: &mo1_full[..],
                    };
                    _dgemm_full(c_mo, 'N', &mo1_full_slice, 'N', &mut mo1_ao_mat, 1.0, 0.0);

                    // Save mo1_ao for atom 0, direction 0 for debug
                    if ia == 0 && ja == 0 && dir_y == 0 {
                        self.result.insert("mo1_ao_00".to_string(), mo1_ao_mat.clone());
                    }

                    // dm1[p,q] = Σ_i mo1_ao[p,i] * c_occ[q,i] = mo1_ao @ c_occ^T
                    _dgemm_full(&mo1_ao_mat, 'N', &c_occ, 'T', &mut dm1_mat, 1.0, 0.0);
                    // dm1_e[p,q] = Σ_i mo1_ao[p,i] * c_occ[q,i] * eps[i] = mo1_ao @ c_occ_e^T
                    _dgemm_full(&mo1_ao_mat, 'N', &c_occ_e, 'T', &mut dm1_e_mat, 1.0, 0.0);

                    // dm1_mat/dm1_e_mat are column-major [nao, nao]: data[p + q*nao].
                    let dm1 = &dm1_mat.data;
                    let dm1_e = &dm1_e_mat.data;
                    let h1_data_ia = &self.h1ao[ia].data;

                    for dir_x in 0..3 {
                        // de2[x,y] += 4 * Σ_pq h1ao[ia][x,p,q] * dm1[p,q]
                        // h1ao is [3*nao, nao] column-major: data[(x*nao+p) + q*(3*nao)]
                        // dm1 is [nao, nao] column-major: data[p + q*nao]
                        let mut contrib = 0.0;
                        for p in 0..nao {
                            for q in 0..nao {
                                let h1_idx = dir_x * nao + p + q * 3 * nao;
                                contrib += h1_data_ia[h1_idx] * dm1[p + q * nao];
                            }
                        }
                        let idx = i_t(ia, ja, dir_x, dir_y);
                        cphf[idx] += 4.0 * contrib;

                        // s1 correction (if s1ao available)
                        let s1_data = &s1ao_ia[dir_x];
                        if s1_data.len() == nao * nao {
                            let mut s1c = 0.0;
                            for p in 0..nao {
                                for q in 0..nao {
                                    // s1_data is row-major: s1_data[p*nao + q]
                                    // dm1_e is column-major: dm1_e[p + q*nao]
                                    s1c += s1_data[p * nao + q] * dm1_e[p + q * nao];
                                }
                            }
                            cphf[idx] -= 4.0 * s1c;

                            // s1oo·mo_e1 term: de2 -= 2 * Σ_ij s1oo[x,i,j] * mo_e1[y,i,j]
                            let s1oo_data = &s1oo_all[ia][dir_x];
                            let mo_e1_data = &mo_e1_all[ja][dir_y];
                            let mut s1e1 = 0.0;
                            for j in 0..nocc { for i in 0..nocc {
                                s1e1 += s1oo_data[i + j * nocc] * mo_e1_data[i + j * nocc];
                            }}
                            cphf[idx] -= 2.0 * s1e1;
                        }
                    }
                }
            }
        }

        // Fill lower triangle by symmetry: de2[ja,ia] = de2[ia,ja].T
        for ia in 0..natm {
            for ja in 0..ia {
                for x in 0..3 {
                    for y in 0..3 {
                        let i_ij = i_t(ia, ja, x, y);
                        let i_ji = i_t(ja, ia, y, x);
                        cphf[i_ji] = cphf[i_ij];
                    }
                }
            }
        }

        // Store in self.result as MatrixFull [n3, n3]
        let to_mat = |arr: &[f64]| -> MatrixFull<f64> {
            let mut m = vec![0.0; n3 * n3];
            for i0 in 0..natm { for j0 in 0..natm { for x in 0..3 { for y in 0..3 {
                m[(i0 * 3 + x) + (j0 * 3 + y) * n3] = arr[i_t(i0, j0, x, y)];
            }}}}
            for i0 in 0..natm { for j0 in 0..i0 {
                for x in 0..3 { for y in 0..3 {
                    m[(j0 * 3 + y) + (i0 * 3 + x) * n3] = m[(i0 * 3 + x) + (j0 * 3 + y) * n3];
                }}
            }}
            MatrixFull::from_vec([n3, n3], m).unwrap()
        };
        self.timings.push(("  cphf: contract", _t_contract.elapsed()));
        self.timings.push(("calc_cphf_contrib", _t.elapsed()));
        self.result.insert("cphf_contrib".to_string(), to_mat(&cphf));
        self
    }

    /// Compute nuclear repulsion Hessian.
    /// Formula: ∂²/∂R_A∂R_B Σ_{I≠J} Z_I Z_J / |R_I - R_J|
    /// Result is stored as MatrixFull [n3, n3] (n3 = natm*3).
    pub fn calc_hess_nuc(&mut self) -> &mut Self {
        let _t = std::time::Instant::now();
        let mol = &self.scf_data.mol;
        let natm = mol.geom.nfree;
        let n3 = natm * 3;
        let mut h = vec![0.0; n3 * n3];  // column-major
        let atom_charges: Vec<f64> = crate::geom_io::get_charge(&mol.geom.elem);

        for ia in 0..natm {
            let r_ia = mol.geom.get_coord(ia);
            let q_ia = atom_charges[ia];

            for ja in 0..natm {
                if ia == ja { continue; }

                let r_ja = mol.geom.get_coord(ja);
                let q_ja = atom_charges[ja];

                let mut r12 = [0.0f64; 3];
                let mut r2 = 0.0;
                for x in 0..3 {
                    r12[x] = r_ia[x] - r_ja[x];
                    r2 += r12[x] * r12[x];
                }
                let r = r2.sqrt();
                let r3 = r * r * r;
                let r5 = r3 * r * r;

                let qq = q_ia * q_ja;

                // Diagonal block (ia,ia): -δ_{μν}*qq/r³ + 3*qq*r12[μ]*r12[ν]/r⁵
                for x in 0..3 {
                    for y in 0..3 {
                        let kronecker = if x == y { 1.0 } else { 0.0 };
                        let val = -kronecker * qq / r3 + 3.0 * qq * r12[x] * r12[y] / r5;
                        h[(ia*3 + x) + (ia*3 + y) * n3] += val;
                    }
                }

                // Off-diagonal block (ia,ja): δ_{μν}*qq/r³ - 3*qq*r12[μ]*r12[ν]/r⁵
                for x in 0..3 {
                    for y in 0..3 {
                        let kronecker = if x == y { 1.0 } else { 0.0 };
                        let val = kronecker * qq / r3 - 3.0 * qq * r12[x] * r12[y] / r5;
                        h[(ia*3 + x) + (ja*3 + y) * n3] = val;
                    }
                }
            }
        }

        self.result.insert("hess_nuc".to_string(),
            MatrixFull::from_vec([n3, n3], h).unwrap());
        self.timings.push(("calc_hess_nuc", _t.elapsed()));
        self
    }

    /// Compute full analytical Hessian: h_partial + CP-HF contribution + hess_nuc.
    ///
    /// Requires prior `calc_h1ao()` call. Returns the total Hessian as
    /// MatrixFull [n3, n3] stored in `self.result["hess_total"]`.
    pub fn compute_hessian(&mut self) -> &mut Self {
        let _t = std::time::Instant::now();
        self.calc_cphf_contrib().calc_hess_nuc();

        let h_partial = self.result.get("h_partial")
            .expect("calc_ej_ek must be called before compute_hessian");
        let cphf_contrib = self.result.get("cphf_contrib")
            .expect("calc_cphf_contrib must be called before compute_hessian");
        let hess_nuc = self.result.get("hess_nuc")
            .expect("calc_hess_nuc must be called before compute_hessian");

        let n3 = h_partial.size[0];  // n3 = natm * 3
        let mut total = vec![0.0; n3 * n3];
        for i in 0..n3 { for j in 0..n3 {
            total[i + j * n3] = h_partial[[i, j]] + cphf_contrib[[i, j]] + hess_nuc[[i, j]];
        }}
        self.result.insert("hess_total".to_string(),
            MatrixFull::from_vec([n3, n3], total).unwrap());
        self.timings.push(("compute_hessian (sum)", _t.elapsed()));
        self
    }
}

// ── Helper: V^{-1} via matrix power. int2c must be column-major ──
fn compute_vinv(int2c_col_major: &[f64], n: usize) -> Vec<f64> {
    let m = MatrixFull::from_vec([n, n], int2c_col_major.to_vec()).unwrap();
    let inv = _power_rayon_for_symmetric_matrix(&m, -1.0, 1e-12).unwrap();
    inv.data
}

// ══════════════════════════════════════════════════════════
// hcore assembly and e1 computation (below)
// ══════════════════════════════════════════════════════════

/// Build hcore matrix for atom pair (ia,ja) matching PySCF's hcore_generator.
fn build_hcore(mol: &Molecule, ia: usize, ja: usize) -> Vec<f64> {
    let nao = mol.num_basis; let n3 = nao * nao;
    let aoslices = mol.aoslice_by_atom();
    let p0 = aoslices[ia][2] as usize; let ni = aoslices[ia][3] - p0;
    let q0 = aoslices[ja][2] as usize; let qj = aoslices[ja][3] - q0;
    let atom_charges: Vec<f64> = crate::geom_io::get_charge(&mol.geom.elem);
    let i9 = |c: usize, p: usize, q: usize| c * n3 + p * nao + q;
    let cint = mol.initialize_cint(false);
    let (k_aa,_): (Vec<f64>,_) = cint.integrate_row_major("int1e_ipipkin","s1",None).into();
    let (n_aa,_): (Vec<f64>,_) = cint.integrate_row_major("int1e_ipipnuc","s1",None).into();
    let (k_ab,_): (Vec<f64>,_) = cint.integrate_row_major("int1e_ipkinip","s1",None).into();
    let (n_ab,_): (Vec<f64>,_) = cint.integrate_row_major("int1e_ipnucip","s1",None).into();
    let h1aa: Vec<f64> = k_aa.iter().zip(n_aa.iter()).map(|(k,n)| k+n).collect();
    let h1ab: Vec<f64> = k_ab.iter().zip(n_ab.iter()).map(|(k,n)| k+n).collect();
    let mut h = vec![0.0; 9 * n3];
    if ia == ja {
        let zi = atom_charges[ia];
        let mut rmol = mol.initialize_cint(false);
        rmol.with_rinv_at_nucleus(ia, |mr| {
            let (r2aa,_): (Vec<f64>,_) = mr.integrate_row_major("int1e_ipiprinv","s1",None).into();
            let (r2ab,_): (Vec<f64>,_) = mr.integrate_row_major("int1e_iprinvip","s1",None).into();
            for c in 0..9 { for p in 0..nao { for q in 0..nao {
                let i = i9(c, p, q); h[i] = -zi * (r2aa[i] + r2ab[i]);
                if p >= p0 && p < p0+ni { h[i] += h1aa[i] + zi * (r2aa[i] + r2ab[i]); }
                if q >= p0 && q < p0+ni { h[i] += zi * r2aa[i9(c, q, p)] + zi * r2ab[i]; }
                if p >= p0 && p < p0+ni && q >= p0 && q < p0+ni { h[i] += h1ab[i]; }
            }}}
            for c in 0..9 { for p in 0..nao { for q in 0..p {
                let i1=i9(c,p,q);let i2=i9(c,q,p);let v=h[i1]+h[i2];h[i1]=v;h[i2]=v;
            }}}
            for c in 0..9 { for p in 0..nao { h[i9(c, p, p)] *= 2.0; }}
        });
    } else {
        let zi = atom_charges[ia]; let zj = atom_charges[ja];
        for c in 0..9 { for p in 0..ni { for q in 0..qj {
            h[i9(c, p0+p, q0+q)] += h1ab[i9(c, p0+p, q0+q)];
        }}}
        let mut rmol = mol.initialize_cint(false);
        rmol.with_rinv_at_nucleus(ia, |mr| {
            let (r2aa,_): (Vec<f64>,_)=mr.integrate_row_major("int1e_ipiprinv","s1",None).into();
            let (r2ab,_): (Vec<f64>,_)=mr.integrate_row_major("int1e_iprinvip","s1",None).into();
            for c in 0..9 { for p in 0..qj { for q in 0..nao { h[i9(c, q0+p, q)] += zi * r2aa[i9(c, q0+p, q)]; }}}
            for x in 0..3 { for y in 0..3 { let cs=x*3+y;let cd=y*3+x;
                for p in 0..qj { for q in 0..nao { h[i9(cd, q0+p, q)] += zi * r2ab[i9(cs, q0+p, q)]; }}
            }}
        });
        let mut rmol = mol.initialize_cint(false);
        rmol.with_rinv_at_nucleus(ja, |mr| {
            let (r2aa,_): (Vec<f64>,_)=mr.integrate_row_major("int1e_ipiprinv","s1",None).into();
            let (r2ab,_): (Vec<f64>,_)=mr.integrate_row_major("int1e_iprinvip","s1",None).into();
            for c in 0..9 { for p in 0..ni { for q in 0..nao { h[i9(c, p0+p, q)] += zj * r2aa[i9(c, p0+p, q)]; }}}
            for c in 0..9 { for p in 0..ni { for q in 0..nao { h[i9(c, p0+p, q)] += zj * r2ab[i9(c, p0+p, q)]; }}}
        });
        for c in 0..9 { for p in 0..nao { for q in 0..p {
            let i1=i9(c,p,q);let i2=i9(c,q,p);let v=h[i1]+h[i2];h[i1]=v;h[i2]=v;
        }}}
        for c in 0..9 { for p in 0..nao { h[i9(c, p, p)] *= 2.0; }}
    }
    h
}

/// Compute hcore^{(1)} for atom ia: first derivative of T + V_nuc (3, nao, nao).
/// Follows the existing REST grad/rhf.rs generator_deriv_hcore convention:
///   h1 = -(int1e_ipkin + int1e_ipnuc)
///   vrinv = -Z * int1e_iprinv
///   vrinv[:,p0:p1] += h1[:,p0:p1]
///   return vrinv + vrinv^T
pub fn build_hcore_first_deriv(mol: &Molecule, ia: usize) -> Vec<f64> {
    let nao = mol.num_basis; let nao2 = nao * nao;
    let aoslices = mol.aoslice_by_atom();
    let p0 = aoslices[ia][2] as usize; let p1 = aoslices[ia][3] as usize;
    let cint = mol.initialize_cint(false);
    // h1 = -(int1e_ipkin + int1e_ipnuc)  following REST grad convention
    let (ipkin, _): (Vec<f64>, _) = cint.integrate_row_major("int1e_ipkin", "s1", None).into();
    let (ipnuc, _): (Vec<f64>, _) = cint.integrate_row_major("int1e_ipnuc", "s1", None).into();
    let atom_charges: Vec<f64> = crate::geom_io::get_charge(&mol.geom.elem);
    let zi = atom_charges[ia];
    // vrinv = -Z * int1e_iprinv  (3, nao, nao)
    let mut vrinv = vec![0.0; 3 * nao2];
    let mut rmol = mol.initialize_cint(false);
    rmol.with_rinv_at_nucleus(ia, |mr| {
        let (inv, _): (Vec<f64>, _) = mr.integrate_row_major("int1e_iprinv", "s1", None).into();
        for x in 0..3 { for i in 0..nao { for j in 0..nao {
            vrinv[x * nao2 + i * nao + j] = -zi * inv[x * nao2 + i * nao + j];
        }}}
    });
    // vrinv[:,p0:p1] += h1[:,p0:p1]  — h1 = -(ipkin + ipnuc), only atom block
    for x in 0..3 { for i in p0..p1 { for j in 0..nao {
        vrinv[x * nao2 + i * nao + j] += -(ipkin[x * nao2 + i * nao + j] + ipnuc[x * nao2 + i * nao + j]);
    }}}
    // symmetrize vrinv + vrinv^T
    let mut h1 = vec![0.0; 3 * nao2];
    for x in 0..3 { for i in 0..nao { for j in (i+1)..nao {
        let a = x * nao2 + i * nao + j;
        let b = x * nao2 + j * nao + i;
        let sym = vrinv[a] + vrinv[b];
        h1[a] = sym; h1[b] = sym;
    }}
        for i in 0..nao {
            let d = x * nao2 + i * nao + i;
            h1[d] = 2.0 * vrinv[d];
        }
    }
    h1
}

/// Compute e1 = -2·s1aa·dme0 - 2·s1ab·dme0 + hcore·dm0.
pub fn compute_e1(mol: &Molecule, dm0: &[f64], dme0: &[f64]) -> MatrixFull<f64> {
    let natm = mol.geom.nfree; let nao = mol.num_basis; let n3 = natm * 3;
    let cint_mol = mol.initialize_cint(false);
    let aoslices = mol.aoslice_by_atom();
    let i9 = |c: usize, p: usize, q: usize| c * nao * nao + p * nao + q;
    let (s1aa,_) = cint_mol.integrate_row_major("int1e_ipipovlp","s1",None).into();
    let (s1ab,_) = cint_mol.integrate_row_major("int1e_ipovlpip","s1",None).into();
    let mut e1_ten = vec![0.0; natm * natm * 9];
    let i_t = |i0, j0, x, y| i0 * natm * 9 + j0 * 9 + x * 3 + y;
    for i0 in 0..natm {
        let p0 = aoslices[i0][2] as usize; let ni = aoslices[i0][3] - p0;
        for x in 0..3 { for y in 0..3 {
            let c = x*3+y; let mut s=0.0;
            for p in 0..ni { for q in 0..nao { s += s1aa[i9(c, p0+p, q)] * dme0[(p0+p)*nao+q]; }}
            e1_ten[i_t(i0, i0, x, y)] -= s * 2.0;
        }}
        for j0 in 0..=i0 {
            let q0 = aoslices[j0][2] as usize; let qj = aoslices[j0][3] - q0;
            for x in 0..3 { for y in 0..3 {
                let c = x*3+y; let mut s=0.0;
                for p in 0..ni { for q in 0..qj { s += s1ab[i9(c, p0+p, q0+q)] * dme0[(p0+p)*nao+q0+q]; }}
                e1_ten[i_t(i0, j0, x, y)] -= s * 2.0;
            }}
            let hc = build_hcore(mol, i0, j0);
            for x in 0..3 { for y in 0..3 {
                let c = x*3+y; let mut s=0.0;
                for p in 0..nao { for q in 0..nao { s += hc[i9(c, p, q)] * dm0[p + q*nao]; }}
                e1_ten[i_t(i0, j0, x, y)] += s;
            }}
        }
    }
    let mut e1_mat = vec![0.0; n3 * n3];
    for i0 in 0..natm { for j0 in 0..natm { for x in 0..3 { for y in 0..3 {
        e1_mat[(i0*3+x)+(j0*3+y)*n3] = e1_ten[i_t(i0, j0, x, y)];
    }}}}
    for i0 in 0..natm { for j0 in 0..i0 { for x in 0..3 { for y in 0..3 {
        e1_mat[(j0*3+y)+(i0*3+x)*n3] = e1_mat[(i0*3+x)+(j0*3+y)*n3];
    }}}}
    MatrixFull::from_vec([n3, n3], e1_mat).unwrap()
}

/// Compute full analytical Hessian from SCF data.
///
/// Returns the total Hessian matrix (natm*3, natm*3) in column-major format.
/// Steps: calc_e1 → calc_ej_ek → calc_h1ao → compute_hessian.
pub fn compute_hessian(scf: &SCF) -> Result<MatrixFull<f64>, String> {
    let mut hess = RIRHFHessian::new(scf);
    hess.calc_e1().calc_ej_ek().calc_h1ao().compute_hessian();
    hess.result.remove("hess_total")
        .ok_or_else(|| "hess_total not found after compute_hessian".to_string())
}

/// Compute vibrational frequencies and normal modes from the Hessian matrix.
///
/// Returns (frequencies_cm1, normal_modes) where:
///   - frequencies_cm1: [natm*3] array in cm⁻¹, sorted ascending.
///     First 6 modes (3 trans + 3 rot) should be near zero.
///     Imaginary frequencies reported as negative.
///   - normal_modes: [natm*3, natm*3] column-major, each column is a
///     mass-weighted Cartesian displacement vector (3*atom + xyz).
pub fn compute_frequencies(scf: &SCF) -> Result<(Vec<f64>, MatrixFull<f64>), String> {
    let hess_total = compute_hessian(scf)?;
    let mol = &scf.mol;
    let natm = mol.geom.nfree;
    let n3 = natm * 3;
    let mass_charge = crate::geom_io::get_mass_charge(&mol.geom.elem);

    // Mass-weight Hessian: H_mw[i,j] = H[i,j] / sqrt(m_i * m_j)
    let mut hmw = vec![0.0; n3 * n3];
    for ia in 0..natm {
        let ma = mass_charge[ia].0;
        for ja in 0..natm {
            let mb = mass_charge[ja].0;
            let factor = 1.0 / (ma * mb).sqrt();
            for x in 0..3 {
                for y in 0..3 {
                    let i = ia * 3 + x;
                    let j = ja * 3 + y;
                    hmw[i + j * n3] = hess_total[[i, j]] * factor;
                }
            }
        }
    }

    // Diagonalize mass-weighted Hessian via LAPACK dsyev
    let hmw_mat = MatrixFull::from_vec([n3, n3], hmw).unwrap();
    use rest_tensors::matrix::matrix_blas_lapack::_dsyev;
    let (eigvecs_opt, eigvals, _) = _dsyev(&hmw_mat, 'V');
    let eigvecs = eigvecs_opt.ok_or("Hessian diagonalization failed")?;

    // Convert eigenvalues to frequencies in cm⁻¹
    // ω² = λ (in a.u.), ω = √λ (a.u.), ν = ω/(2π) (a.u.)
    // 1 Hartree = 219474.63 cm⁻¹
    let conv = 219474.63 / 1822.8885f64.sqrt();  // = 5140.49
    let mut freqs = vec![0.0; n3];
    for i in 0..n3 {
        let lambda = eigvals[i];
        if lambda > 0.0 {
            freqs[i] = lambda.sqrt() * conv;
        } else {
            // Imaginary frequency → report as negative
            freqs[i] = -(-lambda).sqrt() * conv;
        }
    }

    Ok((freqs, eigvecs))
}

/// Main entry point for CP-HF / Hessian / frequency calculations.
/// Dispatches based on CPHFParameters::calculation field.
pub fn rhf_hessian_main(
    scf: &SCF,
    cphf_ctrl: &crate::ctrl_io::cphf_parameters::CPHFParameters,
    time_mark: &mut crate::utilities::TimeRecords,
) {
    match cphf_ctrl.calculation.as_str() {
        "h_partial" => {
            println!("\n=== h_partial Only Calculation ===");

            let mut hess = RIRHFHessian::new(scf);
            // Env var override for development: REST_EJ_EK_GX=inline|verify
            {
                let opt = &mut hess.flags.ej_ek_opt;
                for (name, field) in [
                    ("REST_EJ_EK_G4", &mut opt.g4_ek_vk1),
                    ("REST_EJ_EK_G5", &mut opt.g5_ek_ri1),
                    ("REST_EJ_EK_G6", &mut opt.g6_ek_ri2d),
                    ("REST_EJ_EK_G7", &mut opt.g7_ek_ri2o),
                    ("REST_EJ_EK_G8", &mut opt.g8_ej_ri1),
                    ("REST_EJ_EK_G9", &mut opt.g9_ej_ri2d),
                    ("REST_EJ_EK_G10", &mut opt.g10_ej_ri2o),
                ] {
                    if let Ok(v) = std::env::var(name) {
                        if v == "inline" || v == "verify" {
                            *field = TermPath::Inline;
                        } else if v == "blas" {
                            *field = TermPath::Blas;
                        }
                    }
                }
            }
            // Only compute e1 and ej/ek — skip h1ao, cphf_contrib, hess_nuc
            hess.calc_e1().calc_ej_ek();

            // Print h_partial max_abs
            if let Some(hp) = hess.result.get("h_partial") {
                let n3 = hp.size[0];
                let mut hp_max = 0.0;
                for i in 0..n3 * n3 { let v = hp.data[i].abs(); if v > hp_max { hp_max = v; }}
                println!("  h_partial max_abs={:.4e}", hp_max);
            }
            // Print full timing profile
            hess.print_timings();
        }
        "hessian" => {
            println!("\n=== Analytical Hessian Calculation (solver={}) ===", cphf_ctrl.solver);
            time_mark.new_item("Hessian", "analytical Hessian");
            time_mark.count_start("Hessian");

            // ── System-size report & memory monitor setup ───────────────
            let nao = scf.mol.num_basis;
            let nocc = (scf.homo[0] + 1) as usize;
            let natm = scf.mol.geom.nfree;
            let naux = scf.mol.make_auxmol_fake().num_basis;
            let ngrids = scf.grids.as_ref().map(|g| g.coordinates.len()).unwrap_or(0);
            memory_monitor::print_system_size(
                "before Hessian pipeline", natm, nao, nocc, naux, ngrids,
            );
            let limit_gb = scf.mol.ctrl.max_memory_gb;
            let monitor = MemMonitor::start(limit_gb, std::time::Duration::from_millis(20));
            println!(
                "  Memory monitor: limit = {}",
                limit_gb.map(|g| format!("{:.3} GiB (abort on exceed)", g))
                        .unwrap_or_else(|| "NONE (peak tracking only)".to_string())
            );

            // Build full object to access components
            let mut hess = RIRHFHessian::new(scf);
            // Env var override for development: REST_EJ_EK_GX=inline|verify
            // BLAS is the default (no env var needed). Set to 'inline' or 'verify'
            // to run the old for-loop path and diff against the BLAS baseline.
            {
                let opt = &mut hess.flags.ej_ek_opt;
                for (name, field) in [
                    ("REST_EJ_EK_G4", &mut opt.g4_ek_vk1),
                    ("REST_EJ_EK_G5", &mut opt.g5_ek_ri1),
                    ("REST_EJ_EK_G6", &mut opt.g6_ek_ri2d),
                    ("REST_EJ_EK_G7", &mut opt.g7_ek_ri2o),
                    ("REST_EJ_EK_G8", &mut opt.g8_ej_ri1),
                    ("REST_EJ_EK_G9", &mut opt.g9_ej_ri2d),
                    ("REST_EJ_EK_G10", &mut opt.g10_ej_ri2o),
                ] {
                    if let Ok(v) = std::env::var(name) {
                        if v == "inline" || v == "verify" {
                            *field = TermPath::Inline;
                        } else if v == "blas" {
                            *field = TermPath::Blas;
                        }
                    }
                }
            }
            let mut overall_peak_mb: f64 = 0.0_f64;
            let stage_report = |label: &str, monitor: &MemMonitor, overall: &mut f64| {
                let stage_peak = monitor.stage_peak_mb();
                if stage_peak > *overall { *overall = stage_peak; }
                println!(
                    "  [mem] after {:<16}: stage peak RSS = {:8.3} MiB ({:.3} GiB) | overall peak = {:.3} MiB ({:.3} GiB)",
                    label, stage_peak, stage_peak / 1024.0, *overall, *overall / 1024.0
                );
            };
            hess.calc_e1();
            stage_report("calc_e1", &monitor, &mut overall_peak_mb);
            memory_monitor::trim_to_os();
            hess.calc_ej_ek();
            stage_report("calc_ej_ek", &monitor, &mut overall_peak_mb);
            memory_monitor::trim_to_os();
            hess.calc_h1ao();
            stage_report("calc_h1ao", &monitor, &mut overall_peak_mb);
            memory_monitor::trim_to_os();
            hess.compute_hessian();
            stage_report("compute_hessian", &monitor, &mut overall_peak_mb);
            let hess_total = hess.result.get("hess_total")
                .cloned()
                .ok_or_else(|| "hess_total not found".to_string());
            match hess_total {
                Ok(hess_total) => {
                    let n3 = hess_total.size[0];
                    let natm = n3 / 3;
                    println!("  Hessian matrix [{}x{}]:", n3, n3);
                    if cphf_ctrl.verbose > 0 {
                        for i in 0..n3.min(9) {
                            print!("    row[{:2}]:", i);
                            for j in 0..n3.min(9) { print!(" {:10.4e}", hess_total[[i, j]]); }
                            println!();
                        }
                    }
                    // Also print individual components
                    if let Some(hp) = hess.result.get("h_partial") {
                        let mut hp_max = 0.0;
                        for i in 0..n3*n3 { let v = hp.data[i].abs(); if v > hp_max { hp_max = v; }}
                        println!("  h_partial max_abs={:.4e}", hp_max);
                    }
                    if let Some(cc) = hess.result.get("cphf_contrib") {
                        let mut cc_max = 0.0;
                        for i in 0..n3*n3 { let v = cc.data[i].abs(); if v > cc_max { cc_max = v; }}
                        println!("  cphf_contrib max_abs={:.4e}", cc_max);
                    }
                    if let Some(hn) = hess.result.get("hess_nuc") {
                        let mut hn_max = 0.0;
                        for i in 0..n3*n3 { let v = hn.data[i].abs(); if v > hn_max { hn_max = v; }}
                        println!("  hess_nuc max_abs={:.4e}", hn_max);
                    }
                    let mut max_abs = 0.0;
                    for i in 0..n3*n3 { let v = hess_total.data[i].abs(); if v > max_abs { max_abs = v; }}
                    println!("  Hessian total max_abs={:.4e}", max_abs);

                    // Save components as .npy for external comparison
                    if cphf_ctrl.verbose > 0 {
                        let tmpdir = std::env::temp_dir().join(format!("rest_hess_{}", std::process::id()));
                        let _ = std::fs::create_dir_all(&tmpdir);
                        let save_mat = |name: &str, key: &str, res: &HashMap<String, MatrixFull<f64>>| {
                            if let Some(m) = res.get(key) {
                                write_npy_f64(tmpdir.join(name).to_str().unwrap(), &m.data);
                            }
                        };
                        save_mat("rest_hess_tot.npy", "hess_total", &hess.result);
                        save_mat("rest_h_partial.npy", "h_partial", &hess.result);
                        save_mat("rest_cphf.npy", "cphf_contrib", &hess.result);
                        save_mat("rest_hess_nuc.npy", "hess_nuc", &hess.result);
                        let scf_data = hess.scf_data;
                        let dm0_mat = &scf_data.density_matrix[0];
                        let nao = scf_data.mol.num_basis;
                        let mut dm0_c = vec![0.0; nao * nao];
                        for r in 0..nao { for c in 0..nao { dm0_c[r * nao + c] = dm0_mat[[r, c]]; }}
                        write_npy_f64(tmpdir.join("dm0.npy").to_str().unwrap(), &dm0_c);
                        let nmo = scf_data.eigenvalues[0].len();
                        let c = &scf_data.eigenvectors[0];
                        let mut mc_c = vec![0.0; nao * nmo];
                        for r in 0..nao { for c2 in 0..nmo { mc_c[r * nao + c2] = c[[r, c2]]; }}
                        write_npy_f64(tmpdir.join("mo_coeff.npy").to_str().unwrap(), &mc_c);
                        write_npy_f64(tmpdir.join("mo_energy.npy").to_str().unwrap(), &scf_data.eigenvalues[0]);
                        let mut mo_occ_flat = vec![0.0; nmo];
                        for i in 0..nmo { mo_occ_flat[i] = scf_data.occupation[0][i] as f64; }
                        write_npy_f64(tmpdir.join("mo_occ.npy").to_str().unwrap(), &mo_occ_flat);
                        println!("  Saved components to {:?}", tmpdir);
                    }
                },
                Err(e) => eprintln!("Error in Hessian calculation: {}", e),
            }
            hess.print_timings();
            // Final memory report
            let final_rss = memory_monitor::current_rss_mb();
            println!(
                "\n  [mem] Hessian pipeline finished: final RSS = {:.3} MiB ({:.3} GiB), overall peak observed = {:.3} MiB ({:.3} GiB)",
                final_rss, final_rss / 1024.0,
                overall_peak_mb, overall_peak_mb / 1024.0
            );
            monitor.stop();
            time_mark.count("Hessian");
        },
        "frequencies" => {
            println!("\n=== Vibrational Frequency Calculation (solver={}) ===", cphf_ctrl.solver);
            time_mark.new_item("Frequencies", "vibrational frequencies");
            time_mark.count_start("Frequencies");
            match compute_frequencies(scf) {
                Ok((freqs, modes)) => {
                    let n3 = freqs.len();
                    let natm = n3 / 3;
                    println!("\n  Vibrational Frequencies (cm):");
                    println!("  {:>4}  {:>10}  {:>10}", "Mode", "Freq/cm", "Symmetry");
                    println!("  {}  {}  {}", "----", "----------", "----------");
                    for i in 0..n3 {
                        let sym = if freqs[i].abs() < 10.0 { "---" } else { "A" };
                        println!("  {:>4}  {:>10.2}  {:>10}", i + 1, freqs[i], sym);
                    }
                    if cphf_ctrl.verbose > 0 {
                        println!("\n  Normal Modes (mass-weighted Cartesian, column-major):");
                        for i in 0..n3 {
                            println!("  Mode {} ({:10.2} cm):", i + 1, freqs[i]);
                            for ia in 0..natm {
                                print!("    atom {:>2}:", ia + 1);
                                for x in 0..3 { print!(" {:10.4e}", modes[[ia * 3 + x, i]]); }
                                println!();
                            }
                        }
                    }
                },
                Err(e) => eprintln!("Error in frequency calculation: {}", e),
            }
            time_mark.count("Frequencies");
        },
        "rks_vxc_deriv1" => {
            println!("\n=== RKS vxc_deriv1 (XC first derivative for h1ao) ===");
            let xc_type = if scf.mol.xc_data.use_density_gradient() {
                crate::dft::xc_deriv::XCType::GGA
            } else {
                crate::dft::xc_deriv::XCType::LDA
            };
            let vmat = crate::hessian::xc_hessian::vxc_deriv1(scf, xc_type);
            let natm = vmat.len();
            let nao = scf.mol.num_basis;
            println!("  vxc_deriv1: {} atoms, nao = {}", natm, nao);
            let mut all_max = 0.0f64;
            for (ia, m) in vmat.iter().enumerate() {
                let max_abs = m.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
                if max_abs > all_max { all_max = max_abs; }
                println!("    vxc_deriv1[{}]: max_abs={:.6e}", ia, max_abs);
                if cphf_ctrl.verbose > 1 {
                    for x in 0..3 {
                        print!("      x={} first 3 rows, 1 col:", x);
                        for i in 0..3 { print!(" {:11.4e}", m[[x * nao + i, 0]]); }
                        println!();
                    }
                }
            }
            println!("  overall max_abs = {:.6e}", all_max);
            // Save flat (natm*3*nao, nao) row-major-friendly data to .npy for comparison.
            // vmat[ia] is [3*nao, nao] column-major (REST layout). Stack ia along dim 0.
            // Output shape: (natm, 3, nao, nao) in C-order to match PySCF.
            let mut flat = vec![0.0; natm * 3 * nao * nao];
            for ia in 0..natm {
                for x in 0..3 {
                    for i in 0..nao {
                        for j in 0..nao {
                            // REST vmat[ia] is column-major [3*nao, nao]: element (row=x*nao+i, col=j)
                            let v = vmat[ia][[x * nao + i, j]];
                            // C-order [ia, x, i, j]
                            flat[((ia * 3 + x) * nao + i) * nao + j] = v;
                        }
                    }
                }
            }
            write_npy_f64("/tmp/rks_vxc_deriv1_h2o_rest.npy", &flat);
            println!("  Saved /tmp/rks_vxc_deriv1_h2o_rest.npy as 1D ({} elements)", flat.len());
            println!("  Reshape in Python to ({}, 3, {}, {})", natm, nao, nao);
        },
        "rks_vxc_diag" => {
            println!("\n=== RKS vxc_diag (XC diagonal 2nd-deriv for h_partial) ===");
            let xc_type = if scf.mol.xc_data.use_density_gradient() {
                crate::dft::xc_deriv::XCType::GGA
            } else {
                crate::dft::xc_deriv::XCType::LDA
            };
            let veff = crate::hessian::xc_hessian::vxc_diag(scf, xc_type);
            let nao = scf.mol.num_basis;
            // veff is [9*nao, nao] column-major; (alpha, beta) block at rows [(alpha*3+beta)*nao .. +nao].
            // Per-block max_abs:
            let mut all_max = 0.0f64;
            for alpha in 0..3 { for beta in 0..3 {
                let row0 = (alpha * 3 + beta) * nao;
                let mut m_max = 0.0f64;
                for i in 0..nao { for j in 0..nao {
                    let v = veff[[row0 + i, j]].abs();
                    if v > m_max { m_max = v; }
                }}
                println!("  veff_diag[alpha={}, beta={}]: max_abs={:.6e}", alpha, beta, m_max);
                if m_max > all_max { all_max = m_max; }
            }}
            println!("  overall max_abs = {:.6e}", all_max);
            // Save flat C-order (3, 3, nao, nao) for Python comparison.
            let mut flat = vec![0.0; 9 * nao * nao];
            for alpha in 0..3 { for beta in 0..3 {
                let row0 = (alpha * 3 + beta) * nao;
                for i in 0..nao { for j in 0..nao {
                    flat[((alpha * 3 + beta) * nao + i) * nao + j] = veff[[row0 + i, j]];
                }}
            }}
            write_npy_f64("/tmp/rks_vxc_diag_h2o_rest.npy", &flat);
            println!("  Saved /tmp/rks_vxc_diag_h2o_rest.npy as 1D ({} elements)", flat.len());
            println!("  Reshape in Python to (3, 3, {}, {})", nao, nao);
        },
        "rks_vxc_deriv2" => {
            println!("\n=== RKS vxc_deriv2 (XC off-diagonal 2nd-deriv for h_partial) ===");
            let xc_type = if scf.mol.xc_data.use_density_gradient() {
                crate::dft::xc_deriv::XCType::GGA
            } else {
                crate::dft::xc_deriv::XCType::LDA
            };
            let vmat = crate::hessian::xc_hessian::vxc_deriv2(scf, xc_type);
            let natm = vmat.len();
            let nao = scf.mol.num_basis;
            println!("  vxc_deriv2: {} atoms, nao = {}", natm, nao);
            let mut all_max = 0.0f64;
            for (ia, m) in vmat.iter().enumerate() {
                let max_abs = m.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
                if max_abs > all_max { all_max = max_abs; }
                if cphf_ctrl.verbose > 0 {
                    // Per (α, β) block max
                    let mut per_block = String::new();
                    for alpha in 0..3 { for beta in 0..3 {
                        let row0 = (alpha * 3 + beta) * nao;
                        let mut m_max = 0.0f64;
                        for i in 0..nao { for j in 0..nao {
                            let v = m[[row0 + i, j]].abs();
                            if v > m_max { m_max = v; }
                        }}
                        per_block.push_str(&format!(" ({},{})={:.3e}", alpha, beta, m_max));
                    }}
                    println!("    vmat[{}]: max_abs={:.6e} blocks:{}", ia, max_abs, per_block);
                } else {
                    println!("    vmat[{}]: max_abs={:.6e}", ia, max_abs);
                }
            }
            println!("  overall max_abs = {:.6e}", all_max);
            // Save flat C-order (natm, 3, 3, nao, nao) for Python comparison.
            let mut flat = vec![0.0; natm * 9 * nao * nao];
            for ia in 0..natm {
                for alpha in 0..3 { for beta in 0..3 {
                    let row0 = (alpha * 3 + beta) * nao;
                    for i in 0..nao { for j in 0..nao {
                        flat[((ia * 9 + alpha * 3 + beta) * nao + i) * nao + j] =
                            vmat[ia][[row0 + i, j]];
                    }}
                }}
            }
            write_npy_f64("/tmp/rks_vxc_deriv2_h2o_rest.npy", &flat);
            println!("  Saved /tmp/rks_vxc_deriv2_h2o_rest.npy as 1D ({} elements)", flat.len());
            println!("  Reshape in Python to ({}, 3, 3, {}, {})", natm, nao, nao);
        },
        other => {
            eprintln!("Unknown CPHF calculation type: '{}'. Supported: hessian, h_partial, frequencies, rks_vxc_deriv1, rks_vxc_diag, rks_vxc_deriv2", other);
        },
    }
}

/// Compute the CP-HF Hessian contribution and compare with PySCF reference.
/// Called from main_driver (solver = "cphf_hess").
pub fn test_cphf_hessian(scf: &SCF) -> Result<(), String> {
    use crate::ri_cphf::CPHFSolverPySCF;
    let natm = scf.mol.geom.nfree;
    let n3 = natm * 3;
    let nao = scf.mol.num_basis;
    let mo_occ = &scf.occupation[0];
    let mo_energy = &scf.eigenvalues[0];
    // Use full occupation (no frozen core) to match production calc_cphf_contrib
    let nocc = mo_occ.iter().filter(|&&o| o > 0.0).count();
    let start_mo = 0;
    let lumo = start_mo + nocc;

    println!("\n  === CP-HF Hessian integration test ===");
    println!("  SCF energy: {:.14} Ha", scf.scf_energy);
    println!("  natm={}, n3={}", natm, n3);

    // Compute electronic partial Hessian + h1ao + CP-HF contribution
    let mut hess = RIRHFHessian::new(scf);
    hess.calc_e1().calc_ej_ek().calc_h1ao().calc_cphf_contrib().calc_hess_nuc();

    let h_partial = hess.result.get("h_partial")
        .ok_or("h_partial not found — call calc_ej_ek first")?;
    let cphf_contrib = hess.result.get("cphf_contrib")
        .ok_or("cphf_contrib not found — call calc_cphf_contrib first")?;



    // Print h_partial
    println!("\n  h_partial matrix [{}x{}]:", n3, n3);
    for i in 0..n3.min(9) {
        print!("    row[{}]:", i);
        for j in 0..n3.min(9) {
            print!(" {:9.4e}", h_partial[[i, j]]);
        }
        println!();
    }
    let mut hp_max = 0.0;
    for i in 0..n3 { for j in 0..n3 {
        let v = h_partial[[i, j]].abs(); if v > hp_max { hp_max = v; }
    }}
    println!("  h_partial max_abs={:.4e}", hp_max);

    // Print CP-HF contribution
    println!("\n  CP-HF contribution matrix [{}x{}]:", n3, n3);
    let mut cc_max = 0.0;
    for i in 0..n3 { for j in 0..n3 {
        let v = cphf_contrib[[i, j]].abs();
        if v > cc_max { cc_max = v; }
    }}
    println!("  CP-HF max_abs={:.4e}", cc_max);

    // Print full electronic Hessian = h_partial + cphf_contrib
    let hess_elec: MatrixFull<f64> = {
        let n3 = n3; let mut m = vec![0.0; n3 * n3];
        for i in 0..n3 { for j in 0..n3 {
            m[i + j * n3] = h_partial[[i, j]] + cphf_contrib[[i, j]];
        }}
        MatrixFull::from_vec([n3, n3], m).unwrap()
    };
    println!("\n  Total electronic Hessian (h_partial + CP-HF) [{}x{}]:", n3, n3);
    let mut he_max = 0.0;
    for i in 0..n3 { for j in 0..n3 {
        let v = hess_elec[[i, j]].abs(); if v > he_max { he_max = v; }
    }}
    println!("  Hessian max_abs={:.4e}", he_max);

    // ── Write SCF data and call PySCF reference script ──
    let tmpdir = std::env::temp_dir().join(format!("hess_cphf_{}", std::process::id()));
    std::fs::create_dir_all(&tmpdir).map_err(|e| format!("mkdir: {}", e))?;
    let nmo = scf.eigenvalues[0].len();
    let dm0_mat = &scf.density_matrix[0];
    let mut dm0_c = vec![0.0; nao * nao];
    for r in 0..nao { for c in 0..nao { dm0_c[r * nao + c] = dm0_mat[[r, c]]; }}
    write_npy_f64(tmpdir.join("dm0.npy").to_str().unwrap(), &dm0_c);
    let mut mo_occ_flat = vec![0.0; nmo];
    for i in 0..nmo { mo_occ_flat[i] = scf.occupation[0][i] as f64; }
    write_npy_f64(tmpdir.join("mo_occ.npy").to_str().unwrap(), &mo_occ_flat);
    let c = &scf.eigenvectors[0];
    let mut mc_c = vec![0.0; nao * nmo];
    for r in 0..nao { for c2 in 0..nmo { mc_c[r * nao + c2] = c[[r, c2]]; }}
    write_npy_f64(tmpdir.join("mo_coeff.npy").to_str().unwrap(), &mc_c);
    write_npy_f64(tmpdir.join("mo_energy.npy").to_str().unwrap(), &scf.eigenvalues[0]);

    let atom_str = "O 0.00000000 0.00000000 0.12982363; H 0.75933475 0.00000000 -0.46621158; H -0.75933475 0.00000000 -0.46621158";
    let meta = serde_json::json!({
        "atom": atom_str, "basis": "def2-svp", "auxbasis": "def2-svp-rifit",
        "nao": nao, "nmo": nmo,
        "dm0": tmpdir.join("dm0.npy").to_str().unwrap(),
        "mo_occ": tmpdir.join("mo_occ.npy").to_str().unwrap(),
        "mo_coeff": tmpdir.join("mo_coeff.npy").to_str().unwrap(),
        "mo_energy": tmpdir.join("mo_energy.npy").to_str().unwrap(),
    });
    std::fs::write(tmpdir.join("meta.json"), serde_json::to_string(&meta).unwrap())
        .map_err(|e| format!("write meta.json: {}", e))?;

    let script_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("hessian")
        .join("h2o_cphf_hess_ref.py");

    println!("\n  Calling PySCF reference: h2o_cphf_hess_ref.py");
    let output = std::process::Command::new("python3")
        .arg(script_path.to_str().unwrap())
        .arg(tmpdir.join("meta.json").to_str().unwrap())
        .output()
        .map_err(|e| format!("Failed to run PySCF reference: {}", e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("PySCF error:\n{}", stderr));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse sections marked with ###
    let lines: Vec<&str> = stdout.lines().collect();
    let parse_section = |start_mark: &str, end_mark: &str| -> Vec<f64> {
        let mut in_section = false; let mut vals = Vec::new();
        for &line in &lines {
            let t = line.trim();
            if t == start_mark { in_section = true; continue; }
            if t == end_mark { in_section = false; continue; }
            if in_section { if let Ok(v) = t.parse::<f64>() { vals.push(v); } }
        }
        vals
    };

    let py_hp = parse_section("### H_PARTIAL_MAT ###", "### END ###");
    let py_he = parse_section("### HESS_ELEC_MAT ###", "### END ###");
    let py_cc = parse_section("### CPHF_CONTRIB_MAT ###", "### END ###");

    // PySCF outputs col-major flat of [n3, n3] matrix
    let from_py_flat = |py: &[f64]| -> MatrixFull<f64> {
        let mut m = vec![0.0; n3 * n3];
        for i in 0..n3 { for j in 0..n3 {
            m[i + j * n3] = py[i + j * n3]; // already col-major
        }}
        MatrixFull::from_vec([n3, n3], m).unwrap()
    };

    // ── Compare mo1, h1_mo, s1_mo for atom 0, direction 0 ──
    let compare_mo_vec = |name: &str, py: &[f64], key: &str| {
        if let Some(rust) = hess.result.get(key) {
            if py.len() == rust.data.len() {
                let mut md = 0.0;
                for i in 0..py.len() { let d = (rust.data[i] - py[i]).abs(); if d > md { md = d; }}
                println!("  {:<20} maxdiff={:.4e}", name, md);
            } else { println!("  {:<20} len mismatch: py={}, rust={}", name, py.len(), rust.data.len()); }
        } else { println!("  {:<20} key '{}' not found", name, key); }
    };
    compare_mo_vec("h1_mo[0][0]", &parse_section("### H1_MO_00 ###", "### END ###"), "h1_mo_00");
    compare_mo_vec("s1_mo[0][0]", &parse_section("### S1_MO_00 ###", "### END ###"), "s1_mo_00");
    let py_mo1 = parse_section("### MO1_AO_00 ###", "### END ###");
    compare_mo_vec("mo1_ao[0][0]", &py_mo1, "mo1_ao_00");
    let py_rhs = parse_section("### RHS_FULL_00 ###", "### END ###");
    if py_rhs.len() == nao * nocc {
        let nmo = py_rhs.len() / nocc;
        if let Some(rust) = hess.result.get("rhs_full_00") {
            let mut md_vo = 0.0; let mut md_oo = 0.0;
            for col in 0..nocc { for row in 0..nmo {
                let idx = row + col * nmo;
                let d = (rust.data[idx] - py_rhs[idx]).abs();
                if row >= lumo { if d > md_vo { md_vo = d; }}
                else if row >= start_mo { if d > md_oo { md_oo = d; }}
            }}
            println!("  rhs_full[0][0]   VO_maxdiff={:.4e} OO_maxdiff={:.4e}",
                     md_vo, md_oo);
        }
    }

    if py_hp.len() == n3 * n3 && py_he.len() == n3 * n3 && py_cc.len() == n3 * n3 {
        let ref_hp = from_py_flat(&py_hp);
        let ref_he = from_py_flat(&py_he);
        let ref_cc = from_py_flat(&py_cc);

        // Compare h_partial
        let mut md_hp = 0.0;
        for i in 0..n3 { for j in 0..n3 {
            let d = (h_partial[[i, j]] - ref_hp[[i, j]]).abs(); if d > md_hp { md_hp = d; }
        }}
        println!("\n  h_partial vs PySCF: maxdiff={:.4e}", md_hp);

        // Compare CP-HF contribution
        let mut md_cc = 0.0;
        for i in 0..n3 { for j in 0..n3 {
            let d = (cphf_contrib[[i, j]] - ref_cc[[i, j]]).abs(); if d > md_cc { md_cc = d; }
        }}
        println!("  CP-HF contrib vs PySCF: maxdiff={:.4e}", md_cc);

        // Compare full Hessian
        let mut md_he = 0.0;
        for i in 0..n3 { for j in 0..n3 {
            let d = (hess_elec[[i, j]] - ref_he[[i, j]]).abs(); if d > md_he { md_he = d; }
        }}
        println!("  Hessian elec vs PySCF: maxdiff={:.4e}", md_he);

        // Compare hess_nuc and total Hessian
        let py_hn = parse_section("### HESS_NUC_MAT ###", "### END ###");
        let py_ht = parse_section("### HESS_TOTAL_MAT ###", "### END ###");
        if let Some(rust_hn) = hess.result.get("hess_nuc") {
            if py_hn.len() == n3 * n3 {
                let mut md_hn = 0.0;
                for i in 0..n3*n3 { let d = (rust_hn.data[i] - py_hn[i]).abs(); if d > md_hn { md_hn = d; }}
                println!("  hess_nuc vs PySCF: maxdiff={:.4e}", md_hn);

                // Total Hessian = hess_elec + hess_nuc
                if py_ht.len() == n3 * n3 {
                    let mut md_ht = 0.0;
                    for i in 0..n3 { for j in 0..n3 {
                        let d = (hess_elec[[i, j]] + rust_hn[[i, j]] - py_ht[i + j * n3]).abs();
                        if d > md_ht { md_ht = d; }
                    }}
                    println!("  Hessian total vs PySCF: maxdiff={:.4e}", md_ht);
                }

                // Compare frequencies
                let py_freqs = parse_section("### FREQUENCIES ###", "### END ###");
                if py_freqs.len() == n3 {
                    let hess_total_vec: Vec<f64> = (0..n3*n3).map(|idx| {
                        let j = idx / n3; let i = idx % n3;
                        hess_elec[[i, j]] + rust_hn[[i, j]]
                    }).collect();
                    let hess_tot = MatrixFull::from_vec([n3, n3], hess_total_vec).unwrap();

                    let mass_charge = crate::geom_io::get_mass_charge(&scf.mol.geom.elem);
                    let mut hmw = vec![0.0; n3 * n3];
                    for ia in 0..natm {
                        for ja in 0..natm {
                            let factor = 1.0 / (mass_charge[ia].0 * mass_charge[ja].0).sqrt();
                            for x in 0..3 { for y in 0..3 {
                                hmw[(ia*3+x) + (ja*3+y)*n3] = hess_tot[[ia*3+x, ja*3+y]] * factor;
                            }}
                        }
                    }
                    use rest_tensors::matrix::matrix_blas_lapack::_dsyev;
                    let hmw_mat = MatrixFull::from_vec([n3, n3], hmw).unwrap();
                    let (_, eigvals, _) = _dsyev(&hmw_mat, 'N');
                    let conv = 219474.63 / 1822.8885f64.sqrt();  // = 5140.49
                    let mut md_freq = 0.0;
                    for i in 0..n3 {
                        let lambda = eigvals[i];
                        let f = if lambda > 0.0 {
                            lambda.sqrt() * conv
                        } else {
                            -(-lambda).sqrt() * conv
                        };
                        let d = (f - py_freqs[i]).abs();
                        if d > md_freq { md_freq = d; }
                    }
                    println!("  frequencies vs PySCF: maxdiff={:.4e} cm⁻¹", md_freq);
                }
            } else { println!("  hess_nuc comparison: size mismatch"); }
        } else { println!("  hess_nuc: not found in REST result"); }

        println!("\n  Benchmark results:");
        println!("    h_partial    vs PySCF: md={:.4e}  {:6}",
                 md_hp, if md_hp < 1e-4 { "PASS" } else { "FAIL" });
        println!("    cphf_contrib vs PySCF: md={:.4e}  {:6}",
                 md_cc, if md_cc < 1e-4 { "PASS" } else { "FAIL" });
        if md_cc > 0.1 {
            println!("    NOTE: cphf_contrib large discrepancy → calc_h1ao() in rhf.rs needs debugging");
        }
        println!("    hess_elec    vs PySCF: md={:.4e}  {:6}",
                 md_he, if md_he < 1e-4 { "PASS" } else { "FAIL" });
    } else {
        println!("\n  WARNING: PySCF returned mismatched data sizes");
        println!("    h_partial: {} (expected {})", py_hp.len(), n3 * n3);
        println!("    hess_elec: {} (expected {})", py_he.len(), n3 * n3);
        println!("    cphf_contrib: {} (expected {})", py_cc.len(), n3 * n3);
        println!("  (printing raw PySCF output for debugging)");
        for line in &lines { println!("    {}", line); }
    }

    let _ = std::fs::remove_dir_all(&tmpdir);
    Ok(())
}

/// CK7: verify calc_h1ao against PySCF make_h1.
/// Called from main_driver (solver = "ck7").
///
/// Exports REST SCF data as .npy, invokes PySCF to compute reference h1ao,
/// then compares h1ao[ia] element-by-element.
pub fn test_ck7_h1ao(scf: &SCF) -> Result<(), String> {
    let nao = scf.mol.num_basis;
    let natm = scf.mol.geom.nfree;
    let nmo = scf.eigenvalues[0].len();
    let nao3 = nao * nao;

    println!("\n  === CK7: h1ao verification ===");
    println!("  SCF energy: {:.14} Ha", scf.scf_energy);
    println!("  nao={}, natm={}, nmo={}", nao, natm, nmo);

    // Run calc_h1ao (disable auxbasis_response for fair comparison with ck7_h1ao_flat.py)
    let mut hess = RIRHFHessian::new(scf);
    hess.flags.auxbasis_response = false;
    hess.calc_h1ao();
    if hess.h1ao.len() != natm {
        return Err(format!("h1ao length mismatch: {} vs {}", hess.h1ao.len(), natm));
    }

    // Write SCF data to temp dir
    let tmpdir = std::env::temp_dir().join(format!("hess_ck7_{}", std::process::id()));
    std::fs::create_dir_all(&tmpdir).map_err(|e| format!("mkdir: {}", e))?;

    let atom_str = "O 0.00000000 0.00000000 0.11709209; H 0.75677522 0.00000000 -0.46836837; H -0.75677522 0.00000000 -0.46836837";
    let meta = serde_json::json!({
        "atom": atom_str, "basis": "def2-svp", "auxbasis": "def2-svp-rifit",
        "nao": nao, "nmo": nmo,
        "dm0": tmpdir.join("dm0.npy").to_str().unwrap(),
        "mo_occ": tmpdir.join("mo_occ.npy").to_str().unwrap(),
        "mo_coeff": tmpdir.join("mo_coeff.npy").to_str().unwrap(),
        "mo_energy": tmpdir.join("mo_energy.npy").to_str().unwrap(),
    });
    std::fs::write(tmpdir.join("meta.json"), serde_json::to_string(&meta).unwrap())
        .map_err(|e| format!("write meta.json: {}", e))?;

    let dm0_mat = &scf.density_matrix[0];
    let mut dm0_c = vec![0.0; nao * nao];
    for r in 0..nao { for c in 0..nao { dm0_c[r * nao + c] = dm0_mat[[r, c]]; }}
    write_npy_f64(tmpdir.join("dm0.npy").to_str().unwrap(), &dm0_c);
    // Read back and verify round-trip
    let dm0_back = read_npy_f64(tmpdir.join("dm0.npy").to_str().unwrap());
    let mut dm0_md = 0.0;
    for i in 0..nao*nao { let d = (dm0_c[i] - dm0_back[i]).abs(); if d > dm0_md { dm0_md = d; }}
    if dm0_md > 1e-15 { println!("  WARNING: dm0 round-trip error = {:.4e}", dm0_md); }
    println!("  dm0 max={:.4e} (round-trip ok, err={:.4e})",
             dm0_c.iter().map(|v| v.abs()).fold(0.0f64, |a,b| a.max(b)), dm0_md);
    let mut mo_occ_flat = vec![0.0; scf.occupation[0].len()];
    for i in 0..scf.occupation[0].len() { mo_occ_flat[i] = scf.occupation[0][i] as f64; }
    write_npy_f64(tmpdir.join("mo_occ.npy").to_str().unwrap(), &mo_occ_flat);
    let c = &scf.eigenvectors[0];
    let mut mc_c = vec![0.0; nao * nmo];
    for r in 0..nao { for c2 in 0..nmo { mc_c[r * nao + c2] = c[[r, c2]]; }}
    write_npy_f64(tmpdir.join("mo_coeff.npy").to_str().unwrap(), &mc_c);
    write_npy_f64(tmpdir.join("mo_energy.npy").to_str().unwrap(), &scf.eigenvalues[0]);

    // Call Python reference script
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/hessian/pyscf_ref/ck7_h1ao_flat.py");
    let py_out = tmpdir.join("py_out");
    std::fs::create_dir_all(&py_out).map_err(|e| format!("mkdir py_out: {}", e))?;

    let output = std::process::Command::new("python3")
        .arg(script.to_str().unwrap())
        .arg(tmpdir.join("meta.json").to_str().unwrap())
        .output()
        .map_err(|e| format!("Python script failed: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Python error:\n{}", stderr));
    }

    // Parse output sections
    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    let parse_section = |start_mark: &str, end_mark: &str| -> Vec<f64> {
        let mut in_section = false; let mut vals = Vec::new();
        for &line in &lines {
            let t = line.trim();
            if t == start_mark { in_section = true; continue; }
            if t == end_mark { in_section = false; continue; }
            if in_section { if let Ok(v) = t.parse::<f64>() { vals.push(v); } }
        }
        vals
    };

    // Compare h1ao, and also vj1/vk1/hcore for atom 0
    let mut h1ao_md = 0.0; let mut hcore_md = 0.0;
    let mut vj1pres_md = 0.0; let mut vj1post_md = 0.0;
    let mut vk1pres_md = 0.0; let mut vk1post_md = 0.0;

    for ia in 0..natm {
        let h1_start = format!("### H1AO_{} ###", ia);
        let vj1_start = format!("### VJ1_{} ###", ia);
        let vk1_start = format!("### VK1_{} ###", ia);
        let hc_start = format!("### HCORE_{} ###", ia);

        let py_ha = parse_section(&h1_start, "### END ###");
        let py_vj = parse_section(&vj1_start, "### END ###");
        let py_vk = parse_section(&vk1_start, "### END ###");
        let py_hc = parse_section(&hc_start, "### END ###");

        if py_ha.len() != 3 * nao3 { return Err(format!("h1ao[{}]: exp {}, got {}", ia, 3*nao3, py_ha.len())); }
        if py_vj.len() != 3 * nao3 { return Err(format!("vj1[{}]: exp {}, got {}", ia, 3*nao3, py_vj.len())); }
        if py_vk.len() != 3 * nao3 { return Err(format!("vk1[{}]: exp {}, got {}", ia, 3*nao3, py_vk.len())); }
        if py_hc.len() != 3 * nao3 { return Err(format!("hcore[{}]: exp {}, got {}", ia, 3*nao3, py_hc.len())); }

        // h1ao comparison
        {
            let r = &hess.h1ao[ia];
            let mut md = 0.0;
            for i in 0..3*nao3 { let d = (r.data[i] - py_ha[i]).abs(); if d > md { md = d; }}
            if md > h1ao_md { h1ao_md = md; }
        }
        // vj1 post-sym (what _gen_jk yields matches pre-sym for PySCF)
        {
            let r = hess.result.get(&format!("vj1_{}", ia)).unwrap();
            let mut md = 0.0;
            for i in 0..3*nao3 { let d = (r.data[i] - py_vj[i]).abs(); if d > md { md = d; }}
            if md > vj1post_md { vj1post_md = md; }
        }
        // vk1 post-sym
        {
            let r = hess.result.get(&format!("vk1_{}", ia)).unwrap();
            let mut md = 0.0;
            for i in 0..3*nao3 { let d = (r.data[i] - py_vk[i]).abs(); if d > md { md = d; }}
            if md > vk1post_md { vk1post_md = md; }
        }

        // vj1 pre-sym vs PySCF _gen_jk's VJ1 (which is pre-sym)
        {
            let r = hess.result.get(&format!("vj1_presym_{}", ia)).unwrap();
            let mut md = 0.0;
            for i in 0..3*nao3 { let d = (r.data[i] - py_vj[i]).abs(); if d > md { md = d; }}
            if md > vj1pres_md { vj1pres_md = md; }
        }
        // vk1 pre-sym vs PySCF _gen_jk's VK1
        {
            let r = hess.result.get(&format!("vk1_presym_{}", ia)).unwrap();
            let mut md = 0.0;
            for i in 0..3*nao3 { let d = (r.data[i] - py_vk[i]).abs(); if d > md { md = d; }}
            if md > vk1pres_md { vk1pres_md = md; }
        }
        // Debug: compare step1 max-abs against presym (to see correction magnitude)
        if ia == 0 {
            let step1 = hess.result.get(&format!("vj1_neg_buf_{}", ia)).unwrap();
            let presym = hess.result.get(&format!("vj1_presym_{}", ia)).unwrap();
            let mut s1_abs = 0.0; let mut pr_abs = 0.0; let mut py_abs = 0.0;
            for i in 0..3*nao3 { let a=step1.data[i].abs();if a>s1_abs{s1_abs=a;} let b=presym.data[i].abs();if b>pr_abs{pr_abs=b;} let c=py_vj[i].abs();if c>py_abs{py_abs=c;}}
            println!("  DEBUG vj1[0]: step1_max={:.4e}  presym_max={:.4e}  py_max={:.4e}",
                     s1_abs, pr_abs, py_abs);
        }

        // hcore derivative: convert from `build_hcore_first_deriv` layout
        let rust_hc_raw = build_hcore_first_deriv(&scf.mol, ia);
        let mut rust_hc_flat = vec![0.0; 3 * nao3];
        for x in 0..3 { for i in 0..nao { for j in 0..nao {
            let src = x * nao3 + i * nao + j;
            let dst = (x * nao + i) + j * 3 * nao;
            rust_hc_flat[dst] = rust_hc_raw[src];
        }}}
        let mut hc_md = 0.0;
        for i in 0..3*nao3 { let d = (rust_hc_flat[i] - py_hc[i]).abs(); if d > hc_md { hc_md = d; }}
        if hc_md > hcore_md { hcore_md = hc_md; }

        println!("  ia={}: hcore={:.2e}  vj1_pre={:.2e} vj1_post={:.2e} vk1_pre={:.2e} vk1_post={:.2e} h1ao={:.2e}",
                 ia, hc_md, vj1pres_md, vj1post_md, vk1pres_md, vk1post_md, h1ao_md);
    }

    // ── Detailed intermediate comparison via new PySCF script ──
    println!("\n  === Detailed intermediate comparison ===");
    let script2 = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/hessian/pyscf_ref/ck7_intermediates.py");
    let output2 = std::process::Command::new("python3")
        .arg(script2.to_str().unwrap())
        .arg(tmpdir.join("meta.json").to_str().unwrap())
        .output()
        .map_err(|e| format!("Python intermediate script failed: {}", e))?;
    if !output2.status.success() {
        let stderr = String::from_utf8_lossy(&output2.stderr);
        return Err(format!("Python intermediate error:\n{}", stderr));
    }
    let stdout2 = String::from_utf8_lossy(&output2.stdout);
    let lines2: Vec<&str> = stdout2.lines().collect();
    let parse_section2 = |start_mark: &str| -> Vec<f64> {
        let mut in_section = false; let mut vals = Vec::new();
        for &line in &lines2 {
            let t = line.trim();
            if t == start_mark { in_section = true; continue; }
            if t == "### END ###" { if in_section { break; } continue; }
            if in_section { if let Ok(v) = t.parse::<f64>() { vals.push(v); } }
        }
        vals
    };
    // Compare intermediates
    let compare_flat = |name: &str, py_vals: &[f64], rust_mat: &MatrixFull<f64>| {
        let rows = rust_mat.size[0];
        let mut md = 0.0; let mut flat_idx = 0usize;
        for i in 0..py_vals.len() {
            let d = (rust_mat.data[i] - py_vals[i]).abs();
            if d > md { md = d; flat_idx = i; }
        }
        println!("  {:<20}  maxdiff={:.4e}  (at flat idx={})", name, md, flat_idx);
        md
    };
    // rhoj0_P
    let py_rhoj0 = parse_section2("### RHOJ0_P ###");
    {
        let r = &hess.result["rhoj0_P"];
        if py_rhoj0.len() == r.data.len() {
            let mut md = 0.0;
            for i in 0..r.data.len() { let d = (r.data[i] - py_rhoj0[i]).abs(); if d > md { md = d; }}
            println!("  {:<20}  maxdiff={:.4e}", "rhoj0_P", md);
        } else { println!("  {:<20}  SKIP (len mismatch)", "rhoj0_P"); }
    }
    // vk1_buf
    let py_vk1buf = parse_section2("### VK1_BUF ###");
    if py_vk1buf.len() == 3 * nao3 {
        compare_flat("vk1_buf", &py_vk1buf, &hess.result["vk1_buf"]);
    } else { println!("  {:<20}  SKIP", "vk1_buf"); }
    // Per-atom intermediates
    for ia in 0..natm {
        // vj1_buf[ia]
        let py_vj1b = parse_section2(&format!("### VJ1_BUF_{} ###", ia));
        if py_vj1b.len() == 3 * nao3 {
            compare_flat(&format!("vj1_buf[{}]", ia), &py_vj1b,
                         &hess.result[&format!("vj1_buf_{}", ia)]);
        }
        // vj1_neg_buf[ia]
        let py_vj1nb = parse_section2(&format!("### VJ1_NEGBUF_{} ###", ia));
        if py_vj1nb.len() == 3 * nao3 {
            compare_flat(&format!("vj1_neg_buf[{}]", ia), &py_vj1nb,
                         &hess.result[&format!("vj1_neg_buf_{}", ia)]);
        }
        // vj1_presym[ia]
        let py_vj1pre = parse_section2(&format!("### VJ1_PRESYM_{} ###", ia));
        if py_vj1pre.len() == 3 * nao3 {
            compare_flat(&format!("vj1_presym[{}]", ia), &py_vj1pre,
                         &hess.result[&format!("vj1_presym_{}", ia)]);
        }
        // vk1_presym[ia]
        let py_vk1pre = parse_section2(&format!("### VK1_PRESYM_{} ###", ia));
        if py_vk1pre.len() == 3 * nao3 {
            compare_flat(&format!("vk1_presym[{}]", ia), &py_vk1pre,
                         &hess.result[&format!("vk1_presym_{}", ia)]);
        }
        // vj1[ia] post-sym
        let py_vj1 = parse_section2(&format!("### VJ1_{} ###", ia));
        if py_vj1.len() == 3 * nao3 {
            compare_flat(&format!("vj1[{}]", ia), &py_vj1,
                         &hess.result[&format!("vj1_{}", ia)]);
        }
        // vk1[ia] post-sym
        let py_vk1 = parse_section2(&format!("### VK1_{} ###", ia));
        if py_vk1.len() == 3 * nao3 {
            compare_flat(&format!("vk1[{}]", ia), &py_vk1,
                         &hess.result[&format!("vk1_{}", ia)]);
        }
    }

    println!("  === Summary ===");
    println!("  hcore_deriv maxdiff={:.2e}", hcore_md);
    println!("  vj1 pre-sym maxdiff={:.2e}", vj1pres_md);
    println!("  vj1 post-sym maxdiff={:.2e}", vj1post_md);
    println!("  vk1 pre-sym maxdiff={:.2e}", vk1pres_md);
    println!("  vk1 post-sym maxdiff={:.2e}", vk1post_md);
    println!("  h1ao        maxdiff={:.2e}", h1ao_md);

    let _ = std::fs::remove_dir_all(&tmpdir);
    Ok(())
}

pub fn test_ck8_aux_response(scf: &SCF) -> Result<(), String> {
    let nao = scf.mol.num_basis;
    let natm = scf.mol.geom.nfree;
    let nao3 = nao * nao;
    let naux = {
        let mol = &scf.mol;
        let cint_all = mol.initialize_cint(true);
        let nreg = mol.initialize_cint(false).nbas();
        let naux_shell = cint_all.nbas() - nreg;
        let auxmol = mol.make_auxmol_fake();
        auxmol.num_basis
    };

    println!("\n  === CK8: auxbasis_response verification ===");
    println!("  SCF energy: {:.14} Ha", scf.scf_energy);
    println!("  nao={}, natm={}, naux={}", nao, natm, naux);

    // Run calc_h1ao with default flags (auxbasis_response=true)
    let mut hess = RIRHFHessian::new(scf);
    hess.calc_h1ao();
    if hess.h1ao.len() != natm {
        return Err(format!("h1ao length mismatch: {} vs {}", hess.h1ao.len(), natm));
    }

    // Write SCF data to temp dir
    let tmpdir = std::env::temp_dir().join(format!("hess_ck8_{}", std::process::id()));
    std::fs::create_dir_all(&tmpdir).map_err(|e| format!("mkdir: {}", e))?;

    let atom_str = "O 0.00000000 0.00000000 0.11709209; H 0.75677522 0.00000000 -0.46836837; H -0.75677522 0.00000000 -0.46836837";
    let meta = serde_json::json!({
        "atom": atom_str, "basis": "def2-svp", "auxbasis": "def2-svp-rifit",
        "nao": nao, "nmo": scf.eigenvalues[0].len(),
        "dm0": tmpdir.join("dm0.npy").to_str().unwrap(),
        "mo_occ": tmpdir.join("mo_occ.npy").to_str().unwrap(),
        "mo_coeff": tmpdir.join("mo_coeff.npy").to_str().unwrap(),
        "mo_energy": tmpdir.join("mo_energy.npy").to_str().unwrap(),
    });
    std::fs::write(tmpdir.join("meta.json"), serde_json::to_string(&meta).unwrap())
        .map_err(|e| format!("write meta.json: {}", e))?;

    let dm0_mat = &scf.density_matrix[0];
    let mut dm0_c = vec![0.0; nao * nao];
    for r in 0..nao { for c in 0..nao { dm0_c[r * nao + c] = dm0_mat[[r, c]]; }}
    write_npy_f64(tmpdir.join("dm0.npy").to_str().unwrap(), &dm0_c);
    let mut mo_occ_flat = vec![0.0; scf.occupation[0].len()];
    for i in 0..scf.occupation[0].len() { mo_occ_flat[i] = scf.occupation[0][i] as f64; }
    write_npy_f64(tmpdir.join("mo_occ.npy").to_str().unwrap(), &mo_occ_flat);
    let c = &scf.eigenvectors[0];
    let nmo = scf.eigenvalues[0].len();
    let mut mc_c = vec![0.0; nao * nmo];
    for r in 0..nao { for c2 in 0..nmo { mc_c[r * nao + c2] = c[[r, c2]]; }}
    write_npy_f64(tmpdir.join("mo_coeff.npy").to_str().unwrap(), &mc_c);
    write_npy_f64(tmpdir.join("mo_energy.npy").to_str().unwrap(), &scf.eigenvalues[0]);

    // Call Python reference script
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/hessian/pyscf_ref/ck8_aux_response.py");
    let output = std::process::Command::new("python3")
        .arg(script.to_str().unwrap())
        .arg(tmpdir.join("meta.json").to_str().unwrap())
        .output()
        .map_err(|e| format!("Python script failed: {}", e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Python error:\n{}", stderr));
    }

    // Parse output sections
    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    let parse_section = |start_mark: &str| -> Vec<f64> {
        let mut in_section = false; let mut vals = Vec::new();
        for &line in &lines {
            let t = line.trim();
            if t == start_mark { in_section = true; continue; }
            if t == "### END ###" { if in_section { break; } continue; }
            if in_section { if let Ok(v) = t.parse::<f64>() { vals.push(v); } }
        }
        vals
    };

    // Compare intermediates
    let compare_key = |name: &str, py_vals: &[f64], key: &str| {
        if py_vals.len() != 3 * nao3 {
            println!("  {:<20}  SKIP (py len={})", name, py_vals.len());
            return;
        }
        let r = match hess.result.get(key) {
            Some(m) => m,
            None => { println!("  {:<20}  SKIP (no REST key '{}')", name, key); return; }
        };
        let mut md = 0.0;
        for i in 0..py_vals.len() { let d = (r.data[i] - py_vals[i]).abs(); if d > md { md = d; }}
        println!("  {:<20}  maxdiff={:.4e}", name, md);
    };

    // Compare h1ao directly
    for ia in 0..natm {
        let py_vj1a = parse_section(&format!("### VJ1_AUX_{} ###", ia));
        let py_vk1a = parse_section(&format!("### VK1_AUX_{} ###", ia));
        let py_vj1  = parse_section(&format!("### VJ1_{} ###", ia));
        let py_vk1  = parse_section(&format!("### VK1_{} ###", ia));
        let py_h1ao = parse_section(&format!("### H1AO_{} ###", ia));

        compare_key(&format!("vj1_aux[{}]", ia), &py_vj1a, &format!("vj1_aux_{}", ia));
        compare_key(&format!("vk1_aux[{}]", ia), &py_vk1a, &format!("vk1_aux_{}", ia));
        compare_key(&format!("vj1[{}]", ia), &py_vj1, &format!("vj1_{}", ia));
        compare_key(&format!("vk1[{}]", ia), &py_vk1, &format!("vk1_{}", ia));

        // h1ao comparison
        if py_h1ao.len() == 3 * nao3 {
            let r = &hess.h1ao[ia];
            let mut md = 0.0;
            for i in 0..3*nao3 { let d = (r.data[i] - py_h1ao[i]).abs(); if d > md { md = d; }}
            println!("  {:<20}  maxdiff={:.4e}", format!("h1ao[{}]", ia), md);
        }
    }

    println!("  === CK8 Summary: auxbasis_response=1 corrections ===");
    let _ = std::fs::remove_dir_all(&tmpdir);
    Ok(())
}

// ── Minimal .npy reader/writer ──
/// Write flat f64 slice to .npy v1.0 (C-order).
fn write_npy_f64(path: &str, data: &[f64]) {
    use std::fs::File;
    use std::io::Write;
    let mut f = File::create(path).unwrap();
    f.write_all(b"\x93NUMPY\x01\x00").unwrap();
    let hdr = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': ({},), }}", data.len());
    let pad = (64 - (11 + hdr.len()) % 64) % 64;
    let hdr_pad = hdr + &" ".repeat(pad) + "\n";
    f.write_all(&(hdr_pad.len() as u16).to_le_bytes()).unwrap();
    f.write_all(hdr_pad.as_bytes()).unwrap();
    for &v in data { f.write_all(&v.to_le_bytes()).unwrap(); }
}

/// Read flat f64 array from .npy v1.0/v2.0/v3.0 (C-order).
pub fn read_npy_f64(path: &str) -> Vec<f64> {
    use std::fs::File; use std::io::Read;
    let mut f = File::open(path).unwrap(); let mut magic = [0u8; 6];
    f.read_exact(&mut magic).unwrap(); assert_eq!(&magic, b"\x93NUMPY");
    let mut ver = [0u8; 2]; f.read_exact(&mut ver).unwrap();
    let hlen = match (ver[0], ver[1]) {
        (1,0)=>{let mut b=[0u8;2];f.read_exact(&mut b).unwrap();u16::from_le_bytes(b)as usize}
        (2,0)|(3,0)=>{let mut b=[0u8;4];f.read_exact(&mut b).unwrap();u32::from_le_bytes(b)as usize}
        _=>panic!("unsupported npy version"),
    };
    let mut hdr = vec![0u8; hlen]; f.read_exact(&mut hdr).unwrap();
    let hdr_str = std::str::from_utf8(&hdr).unwrap();
    if hdr_str.contains("'fortran_order': True")||hdr_str.contains("\"fortran_order\": True") {
        panic!("F-order npy not supported");
    }
    let mut raw = Vec::new(); f.read_to_end(&mut raw).unwrap();
    raw.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

/// Setup helper: build H2O SCF with RI-V (def2-svp-rifit), return common data.
#[cfg(test)]
fn setup_h2o_ri() -> (SCF, usize, usize, usize, usize) {
    let token = r###"
[ctrl]
xc = "hf"
basis_path = "basis-set-pool/def2-svp"
auxbas_path = "basis-set-pool/def2-svp-rifit"
basis_type = "spheric"
auxbas_type = "spheric"
eri_type = "ri-v"
charge = 0.0
spin = 1.0
num_threads = 1
print_level = 0
[geom]
name = "H2O"
unit = "Angstrom"
position = """
    O   0.00000000   0.00000000   0.11709209
    H   0.75677522   0.00000000  -0.46836837
    H  -0.75677522   0.00000000  -0.46836837
"""
"###;
    let keys = toml::from_str::<serde_json::Value>(token).unwrap();
    let (ctrl, geom) = crate::ctrl_io::parse_ctl_from_json(&keys).unwrap();
    let mol = crate::Molecule::build_native(ctrl, geom, None).unwrap();
    let mut scf_data = scf_io::SCF::build(mol, &None);
    crate::scf_io::scf_without_build(&mut scf_data, &None);
    let nao = scf_data.mol.num_basis;
    let natm = scf_data.mol.geom.nfree;
    let nocc = (scf_data.homo[0] + 1) as usize;
    let nmo = scf_data.eigenvalues[0].len();
    (scf_data, nao, natm, nocc, nmo)
}

#[cfg(test)]
#[allow(non_snake_case)]
mod debug { use super::*;

    #[test]
    fn test_e1_vs_pyscf() {
        let input_token = r###"
[ctrl]
xc = "hf"
basis_path = "basis-set-pool/def2-svp"
basis_type = "spheric"
charge = 0.0
spin = 1.0
num_threads = 1
print_level = 0
[geom]
name = "H2O"
unit = "Angstrom"
position = """
    O   0.00000000   0.00000000   0.11709209
    H   0.75677522   0.00000000  -0.46836837
    H  -0.75677522   0.00000000  -0.46836837
"""
"###;
        let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
        let (ctrl,geom)=crate::ctrl_io::parse_ctl_from_json(&keys).unwrap();
        let mol=crate::Molecule::build_native(ctrl,geom,None).unwrap();
        let nao=mol.num_basis;let natm=mol.geom.nfree;let n3=natm*3;
        let pydm0=read_npy_f64("src/hessian/h2o_dm0.npy");
        let pydme0=read_npy_f64("src/hessian/h2o_dme0.npy");
        let mut dc=vec![0.0;nao*nao];let mut ec=vec![0.0;nao*nao];
        for r in 0..nao{for c in 0..nao{dc[r+c*nao]=pydm0[r*nao+c];ec[r+c*nao]=pydme0[r*nao+c];}}
        let em=compute_e1(&mol,&dc,&ec);
        let ref_data=read_npy_f64("src/hessian/h2o_e1_full_ref.npy");
        let mut ref_vec=vec![0.0;n3*n3];
        for ia in 0..natm{for ja in 0..natm{for x in 0..3{for y in 0..3{
            ref_vec[(ja*3+y)*n3+(ia*3+x)]=ref_data[(ia*3+x)*n3+(ja*3+y)];
        }}}}
        let rm=MatrixFull::from_vec([n3,n3],ref_vec).unwrap();
        let mut md=0.0;for i in 0..n3{for j in 0..n3{let d=(em[[i,j]]-rm[[i,j]]).abs();if d>md{md=d;}}}
        println!("e1 maxdiff: {:.4e}",md);assert!(md<1e-4,"e1 mismatch: maxdiff={:.4e}",md);
        println!("PASS e1 (diff={:.4e})",md);
    }

    #[test]
    fn test_calc_e1_from_rest_scf() {
        let input_token = r###"
[ctrl]
xc = "hf"
basis_path = "basis-set-pool/def2-svp"
basis_type = "spheric"
eri_type = "analytic"
charge = 0.0
spin = 1.0
num_threads = 1
print_level = 0
scf_acc_etot = 1e-11
scf_acc_rho = 1e-8
scf_acc_eev = 1e-6
max_scf_cycle = 100
[geom]
name = "H2O"
unit = "Angstrom"
position = """
    O   0.00000000   0.00000000   0.11709209
    H   0.75677522   0.00000000  -0.46836837
    H  -0.75677522   0.00000000  -0.46836837
"""
"###;
        let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
        let (ctrl,geom)=crate::ctrl_io::parse_ctl_from_json(&keys).unwrap();
        let mol=crate::Molecule::build_native(ctrl,geom,None).unwrap();
        let mut scf_data=scf_io::SCF::build(mol,&None);
        crate::scf_io::scf_without_build(&mut scf_data,&None);
        let nao=scf_data.mol.num_basis;let natm=scf_data.mol.geom.nfree;let n3=natm*3;
        println!("REST SCF energy={:.14}",scf_data.scf_energy);
        let mut hess=RIRHFHessian::new(&scf_data); hess.calc_e1();
        let em=hess.result.get("e1").unwrap();
        let ref_data=read_npy_f64("src/hessian/h2o_e1_full_ref.npy");
        let mut ref_vec=vec![0.0;n3*n3];
        for ia in 0..natm{for ja in 0..natm{for x in 0..3{for y in 0..3{
            ref_vec[(ja*3+y)*n3+(ia*3+x)]=ref_data[(ia*3+x)*n3+(ja*3+y)];
        }}}}
        let rm=MatrixFull::from_vec([n3,n3],ref_vec).unwrap();
        let mut md=0.0;for i in 0..n3{for j in 0..n3{let d=(em[[i,j]]-rm[[i,j]]).abs();if d>md{md=d;}}}
        println!("e1 maxdiff (REST SCF): {:.4e}",md);
        assert!(md<1e-3,"e1 mismatch: maxdiff={:.4e}",md);
        println!("PASS calc_e1 (diff={:.4e})",md);
    }

    #[test]
    fn test_ri_integrals() {
        let input_token = r###"
[ctrl]
xc = "hf"
basis_path = "basis-set-pool/def2-svp"
auxbas_path = "basis-set-pool/def2-svp-rifit"
basis_type = "spheric"
auxbas_type = "spheric"
eri_type = "ri-v"
charge = 0.0
spin = 1.0
num_threads = 1
print_level = 0
[geom]
name = "H2O"
unit = "Angstrom"
position = """
    O   0.00000000   0.00000000   0.11709209
    H   0.75677522   0.00000000  -0.46836837
    H  -0.75677522   0.00000000  -0.46836837
"""
"###;
        let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
        let (ctrl,geom)=crate::ctrl_io::parse_ctl_from_json(&keys).unwrap();
        let mol=crate::Molecule::build_native(ctrl,geom,None).unwrap();
        // Build aux CInt directly
        let auxmol = mol.make_auxmol_fake();
        let nao=mol.num_basis; let naux=auxmol.num_basis;
        let natm=mol.geom.nfree;
        println!("H2O def2-svp-rifit: nao={}, naux={}, natm={}", nao, naux, natm);
        assert_eq!(nao, 24, "nao should be 24");
        assert_eq!(naux, 76, "naux should be 76");
        // Check 2c integral
        let cint_aux = auxmol.initialize_cint(false);
        let (v,_) = cint_aux.integrate_row_major("int2c2e","s1",None).into();
        let full = naux * naux;
        assert_eq!(v.len(), full, "int2c2e row-major s1 shape");
        println!("int2c2e ok: {} elements ({}x{})", v.len(), naux, naux);
        // Check 3c integral via integrate_cross
        let cint_mol = mol.initialize_cint(false);
        let (v3,_) = CInt::integrate_cross_row_major("int3c2e", [&cint_mol, &cint_mol, &cint_aux], "s1", None).into();
        println!("int3c2e cross: {} elements (expected {})", v3.len(), nao*nao*naux);
        assert_eq!(v3.len(), nao*nao*naux, "int3c2e cross shape");
        println!("PASS: RI integrals infrastructure works");
    }

    #[test]
    fn test_h_partial_vs_pyscf() {
        let input_token = r###"
[ctrl]
xc = "hf"
basis_path = "basis-set-pool/def2-svp"
auxbas_path = "basis-set-pool/def2-svp-rifit"
basis_type = "spheric"
auxbas_type = "spheric"
eri_type = "ri-v"
charge = 0.0
spin = 1.0
num_threads = 1
print_level = 0
scf_acc_etot = 1e-11
scf_acc_rho = 1e-8
scf_acc_eev = 1e-6
max_scf_cycle = 100
[geom]
name = "H2O"
unit = "Angstrom"
position = """
    O   0.00000000   0.00000000   0.11709209
    H   0.75677522   0.00000000  -0.46836837
    H  -0.75677522   0.00000000  -0.46836837
"""
"###;
        let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
        let (ctrl,geom)=crate::ctrl_io::parse_ctl_from_json(&keys).unwrap();
        let mol=crate::Molecule::build_native(ctrl,geom,None).unwrap();
        let mut scf_data=scf_io::SCF::build(mol,&None);
        crate::scf_io::scf_without_build(&mut scf_data,&None);
        let nao=scf_data.mol.num_basis;let natm=scf_data.mol.geom.nfree;let n3=natm*3;
        println!("REST SCF (ri-v) energy={:.14}", scf_data.scf_energy);

        // Compare REST dm0 vs PySCF dm0
        {
            let nao = scf_data.mol.num_basis;
            let pydm0_flat = read_npy_f64(
                "/home/admin/.local/lib/python3.10/site-packages/pyscf/debug-hessian-data/dm0.npy");
            let rest_dm0 = &scf_data.density_matrix[0]; // MatrixFull col-major
            let pydm0_mat = MatrixFull::from_vec([nao, nao], {
                let mut cm = vec![0.0; nao*nao];
                for r in 0..nao { for c in 0..nao { cm[r + c*nao] = pydm0_flat[r*nao + c]; }}
                cm
            }).unwrap();
            let mut dm0_frob = 0.0; let mut dm0_max = 0.0;
            for i in 0..nao { for j in 0..nao {
                let d = rest_dm0[[i,j]] - pydm0_mat[[i,j]]; dm0_frob += d*d; if d.abs()>dm0_max {dm0_max=d.abs();}
            }}
            println!("dm0 Frob={:.4e} maxdiff={:.4e}", dm0_frob.sqrt(), dm0_max);
            println!("dm0 O-H1 block (rows 0..2, cols 14..16):");
            for i in 0..2 { print!("  REST row{}:",i);
                for j in 14..17 { print!(" {:12.8e}", rest_dm0[[i,j]]); } println!(); }
            for i in 0..2 { print!("  PySCF row{}:",i);
                for j in 14..17 { print!(" {:12.8e}", pydm0_mat[[i,j]]); } println!(); }
        }

        // Compare REST SCF data (mc2) vs PySCF
        {
            let c = &scf_data.eigenvectors[0];
            let mo_occ = &scf_data.occupation[0];
            let nocc = (scf_data.homo[0] + 1) as usize;
            let mut mc2_rest = vec![0.0; nao * nocc];
            for p in 0..nao { for i in 0..nocc { mc2_rest[p * nocc + i] = c[[p, i]] * (mo_occ[i] as f64).sqrt(); }}
            // mocc_2.npy is Fortran-order, save C-order first
            // Using /tmp/mc2_c_order.npy (converted from mocc_2.npy)
            let py_mc2 = read_npy_f64("/tmp/mc2_c_order.npy");
            let mut mc2_frob = 0.0; let mut mc2_max = 0.0;
            for i in 0..nao*nocc { let d = mc2_rest[i] - py_mc2[i]; mc2_frob += d*d; let ad=d.abs(); if ad>mc2_max{mc2_max=ad;} }
            println!("mc2 Frob={:.4e} maxdiff={:.4e}", mc2_frob.sqrt(), mc2_max);
        }

        // Compute full Hessian
        let mut hess=RIRHFHessian::new(&scf_data);
        hess.calc_e1();
        hess.calc_ej_ek();
        // Debug: check ej and ek values
        let e1 = hess.result.get("e1").unwrap();
        let ej = hess.result.get("ej").unwrap();
        let ek = hess.result.get("ek").unwrap();
        let hp = hess.result.get("h_partial").unwrap();
        let mut max_abs = |name: &str, m: &MatrixFull<f64>| {
            let mut mx = 0.0;
            for i in 0..n3 { for j in 0..n3 { let v = m[[i,j]].abs(); if v > mx { mx = v; } }}
            println!("  {} max_abs={:.4e}", name, mx);
        };
        max_abs("e1", e1); max_abs("ej", ej); max_abs("ek", ek); max_abs("h_partial", hp);
        let e1 = hess.result.get("e1").unwrap();
        let ej = hess.result.get("ej").unwrap();
        let ek = hess.result.get("ek").unwrap();
        let hp = hess.result.get("h_partial").unwrap();

        // Load PySCF reference
        let to_ref = |name| -> MatrixFull<f64> {
            let d = read_npy_f64(&format!("src/hessian/pyscf_{}_ref.npy", name));
            let mut r = vec![0.0; n3 * n3];
            for ia in 0..natm { for ja in 0..natm { for x in 0..3 { for y in 0..3 {
                r[(ja*3+y)*n3+(ia*3+x)] = d[(ia*3+x)*n3+(ja*3+y)];
            }}}}
            MatrixFull::from_vec([n3, n3], r).unwrap()
        };
        let ref_e1 = to_ref("e1"); let ref_ej = to_ref("ej");
        let ref_ek = to_ref("ek"); let ref_hp = to_ref("h_partial");

        let mut md_e1=0.0;let mut md_ej=0.0;let mut md_ek=0.0;let mut md_hp=0.0;
        let mut frob_e1=0.0;let mut frob_ej=0.0;let mut frob_ek=0.0;let mut frob_hp=0.0;
        for i in 0..n3 { for j in 0..n3 {
            let d1=e1[[i,j]]-ref_e1[[i,j]];let ad1=d1.abs();if ad1>md_e1{md_e1=ad1;} frob_e1+=d1*d1;
            let dj=ej[[i,j]]-ref_ej[[i,j]];let adj=dj.abs();if adj>md_ej{md_ej=adj;} frob_ej+=dj*dj;
            let dk=ek[[i,j]]-ref_ek[[i,j]];let adk=dk.abs();if adk>md_ek{md_ek=adk;} frob_ek+=dk*dk;
            let dh=hp[[i,j]]-ref_hp[[i,j]];let adh=dh.abs();if adh>md_hp{md_hp=adh;} frob_hp+=dh*dh;
        }}
        // Print individual component diagnostics
        println!("REST vs PySCF (maxabs): e1={:.4e}  ej={:.4e}  ek={:.4e}  h_partial={:.4e}",
                 md_e1, md_ej, md_ek, md_hp);
        println!("REST vs PySCF (Frob)  : e1={:.4e}  ej={:.4e}  ek={:.4e}  h_partial={:.4e}",
                 frob_e1.sqrt(), frob_ej.sqrt(), frob_ek.sqrt(), frob_hp.sqrt());

        // Frobenius norm for each component against PySCF reference
        let frob = |a: &MatrixFull<f64>, b: &MatrixFull<f64>| -> f64 {
            let mut s = 0.0;
            for i in 0..n3 { for j in 0..n3 { let d = a[[i,j]] - b[[i,j]]; s += d * d; }}
            s.sqrt()
        };
        // Load component refs from debug-hessian-data (flat (natm,natm,9) = 81 elements)
        let load_ref = |name: &str| -> MatrixFull<f64> {
            let ref_path = format!("/home/admin/.local/lib/python3.10/site-packages/pyscf/debug-hessian-data/{}.npy", name);
            let d = read_npy_f64(&ref_path);
            if d.len() != 81 { panic!("{}: expected 81 elements, got {}", name, d.len()); }
            let mut m = vec![0.0; n3 * n3];
            for i0 in 0..natm { for j0 in 0..natm { for x in 0..3 { for y in 0..3 {
                m[(i0*3+x)+(j0*3+y)*n3] = d[i0*natm*9 + j0*9 + x*3 + y];
            }}}}
            MatrixFull::from_vec([n3, n3], m).unwrap()
        };
        let comp_map = vec![
            ("ej_basic","basic_ej_ref"), ("ej_vjd",""), ("ej_vj1",""),
            ("ej_ri1","ej_ri1_ref"), ("ej_ri2d","ej_ri2d_ref"), ("ej_ri2o","ej_ri2o_ref"),
            ("ek_vkd","ek_vkd_ref"), ("ek_vk1","ek_vk1_ref"),
            ("ek_ri1","ek_ri1_ref"), ("ek_ri2d","ek_ri2d_ref"), ("ek_ri2o","ek_ri2o_ref"),
        ];
        println!("\nFrobenius norms (per-component REST vs PySCF):");
        for (name, ref_name) in &comp_map {
            let rest_mat = hess.result.get(*name);
            if rest_mat.is_none() { println!("  {:12}: NO DATA", name); continue; }
            if ref_name.is_empty() { println!("  {:12}: NO REF FILE", name); continue; }
            let ref_mat = load_ref(ref_name);
            let fnorm = frob(rest_mat.unwrap(), &ref_mat);
            println!("  {:12}: {:12.4e}", name, fnorm);
        }
        // Check the important result: h_partial
        if md_hp < 1e-4 {
            println!("PASS: h_partial matches PySCF reference (diff={:.4e})", md_hp);
        } else {
            println!("FAIL: h_partial mismatch (diff={:.4e}), expected < 1e-4", md_hp);
            // Print matrices for debugging
            println!("\nREST e1:"); for i in 0..n3.min(6){print!("  row{}:",i);for j in 0..n3{print!(" {:8.4e}",e1[[i,j]]);}println!();}
            println!("\nREF e1:"); for i in 0..n3.min(6){print!("  row{}:",i);for j in 0..n3{print!(" {:8.4e}",ref_e1[[i,j]]);}println!();}
        }
        assert!(md_hp < 1e-3, "h_partial mismatch: maxdiff={:.4e}", md_hp);
    }

    #[test]
    fn test_c2h4_h_partial_vs_pyscf() {
        let input_token = r###"
[ctrl]
xc = "hf"
basis_path = "basis-set-pool/def2-svp"
auxbas_path = "basis-set-pool/def2-svp-rifit"
basis_type = "spheric"
auxbas_type = "spheric"
eri_type = "ri-v"
charge = 0.0
spin = 1.0
num_threads = 1
print_level = 0
scf_acc_etot = 1e-11
scf_acc_rho = 1e-8
scf_acc_eev = 1e-6
max_scf_cycle = 100
[geom]
name = "C2H4"
unit = "Angstrom"
position = """
    C    0.00000000   0.00000000   0.66950000
    C    0.00000000   0.00000000  -0.66950000
    H    0.00000000   0.92890000   1.23270000
    H    0.00000000  -0.92890000   1.23270000
    H    0.00000000   0.92890000  -1.23270000
    H    0.00000000  -0.92890000  -1.23270000
"""
"###;
        let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
        let (ctrl,geom)=crate::ctrl_io::parse_ctl_from_json(&keys).unwrap();
        let mol=crate::Molecule::build_native(ctrl,geom,None).unwrap();
        let mut scf_data=scf_io::SCF::build(mol,&None);
        crate::scf_io::scf_without_build(&mut scf_data,&None);
        let nao=scf_data.mol.num_basis;let natm=scf_data.mol.geom.nfree;let n3=natm*3;
        println!("C2H4 def2-svp: nao={}, natm={}", nao, natm);
        println!("REST SCF energy: {:.14}", scf_data.scf_energy);

        // Compare SCF energy against PySCF reference
        let pyscf_energy: f64 = std::fs::read_to_string("src/hessian/c2h4_pyscf_energy.txt")
            .expect("Missing c2h4_pyscf_energy.txt").trim().parse().unwrap();
        let energy_diff = (scf_data.scf_energy - pyscf_energy).abs();
        println!("SCF energy diff: {:.4e} (REST={:.14}, PySCF={:.14})",
                 energy_diff, scf_data.scf_energy, pyscf_energy);
        assert!(energy_diff < 1e-8, "SCF energy mismatch: {:.4e}", energy_diff);

        // Compare dm0 against PySCF (col-major)
        {
            let pydm0_flat = read_npy_f64("src/hessian/c2h4_dm0.npy");
            let rest_dm0 = &scf_data.density_matrix[0];
            let pydm0_mat = MatrixFull::from_vec([nao, nao], {
                let mut cm = vec![0.0; nao*nao];
                for r in 0..nao { for c in 0..nao { cm[r + c*nao] = pydm0_flat[r*nao + c]; }}
                cm
            }).unwrap();
            let mut dm0_frob = 0.0; let mut dm0_max = 0.0;
            for i in 0..nao { for j in 0..nao {
                let d = rest_dm0[[i,j]] - pydm0_mat[[i,j]];
                dm0_frob += d*d; if d.abs()>dm0_max {dm0_max=d.abs();}
            }}
            println!("dm0 Frob={:.4e} maxdiff={:.4e}", dm0_frob.sqrt(), dm0_max);
            // dm0 should match closely
            assert!(dm0_max < 1e-4, "dm0 maxdiff too large: {:.4e}", dm0_max);
        }

        // Compute Hessian
        let mut hess=RIRHFHessian::new(&scf_data);
        hess.calc_e1();
        hess.calc_ej_ek();
        let hp = hess.result.get("h_partial").unwrap();

        // Compare h_partial against PySCF reference
        let ref_data = read_npy_f64("src/hessian/c2h4_h_partial_ref.npy");
        assert_eq!(ref_data.len(), n3 * n3, "ref data length mismatch");
        let ref_mat = MatrixFull::from_vec([n3, n3], ref_data).unwrap();

        let mut md_hp = 0.0; let mut frob_hp = 0.0;
        for i in 0..n3 { for j in 0..n3 {
            let d = hp[[i,j]] - ref_mat[[i,j]];
            let ad = d.abs();
            if ad > md_hp { md_hp = ad; }
            frob_hp += d*d;
        }}
        println!("h_partial vs PySCF: maxdiff={:.4e}  Frob={:.4e}", md_hp, frob_hp.sqrt());

        if md_hp < 1e-4 {
            println!("PASS: C2H4 h_partial matches PySCF reference (diff={:.4e})", md_hp);
        } else {
            println!("FAIL: C2H4 h_partial mismatch (diff={:.4e}), expected < 1e-4", md_hp);
        }
        assert!(md_hp < 1e-3, "C2H4 h_partial mismatch: maxdiff={:.4e}", md_hp);
    }

    /// CK5: vk1[ia] per atom, verified against PySCF _gen_jk.
    #[test]
    fn test_ck5_vk1() {
        let (mut scf_data, nao, natm, nocc, nmo) = setup_h2o_ri();
        let aoslices = scf_data.mol.aoslice_by_atom();
        let cint_reg = scf_data.mol.initialize_cint(false); let nreg = cint_reg.nbas();
        let cint_all = scf_data.mol.initialize_cint(true);
        let naux_shell = cint_all.nbas() - nreg;
        let auxmol = scf_data.mol.make_auxmol_fake(); let naux = auxmol.num_basis;

        // V, V^{-1}, 3c integrals
        let aux_slc_arr = [[nreg, nreg + naux_shell], [nreg, nreg + naux_shell]];
        let aux_slc: &[[usize; 2]] = &aux_slc_arr[..];
        let (int2c_v, _): (Vec<f64>, Vec<usize>) =
            cint_all.integrate_row_major("int2c2e", "s1", Some(aux_slc)).into();
        let int2c = if int2c_v.len() == naux * naux { int2c_v } else {
            let mut v = vec![0.0; naux * naux]; let mut idx = 0;
            for j in 0..naux { for i in 0..=j { v[i + j * naux] = int2c_v[idx]; v[j + i * naux] = int2c_v[idx]; idx += 1; }} v
        };
        let mut int2c_cm = vec![0.0; naux * naux];
        for p in 0..naux { for q in 0..naux { int2c_cm[p + q * naux] = int2c[p * naux + q]; }}
        let vinv = compute_vinv(&int2c_cm, naux);

        let slc_3c: &[[usize; 2]] = &[[0, nreg], [0, nreg], [nreg, nreg + naux_shell]];
        let (t3c, _): (Vec<f64>, Vec<usize>) =
            cint_all.integrate_row_major("int3c2e", "s1", Some(slc_3c)).into();
        let (ip1, _): (Vec<f64>, Vec<usize>) =
            cint_all.integrate_row_major("int3c2e_ip1", "s1", Some(slc_3c)).into();

        // SCF data
        let dm0_mat = &scf_data.density_matrix[0];
        let mut dm0 = vec![0.0; nao * nao];
        for r in 0..nao { for c in 0..nao { dm0[r * nao + c] = dm0_mat[[r, c]]; }}
        let c = &scf_data.eigenvectors[0];
        let mo_occ_data = &scf_data.occupation[0];
        let eps = &scf_data.eigenvalues[0];
        let mut mc2 = vec![0.0; nao * nocc];
        for p in 0..nao { for i in 0..nocc { mc2[p * nocc + i] = c[[p, i]] * (mo_occ_data[i] as f64).sqrt(); }}

        // CK3: rhok0_Pl_
        let nao3 = nao * nao;
        let mut rhok0_Pl_ = vec![0.0; naux * nao * nocc];
        for ia in 0..natm {
            let p0 = aoslices[ia][2] as usize; let p1 = aoslices[ia][3] as usize; let ni = p1 - p0;
            let mut block = vec![0.0; naux * ni * nao];
            for P in 0..naux { for ii in 0..ni { for j in 0..nao {
                block[P * ni * nao + ii * nao + j] = t3c[(p0+ii)*nao*naux + j*naux + P];
            }}}
            let mut coef3c = vec![0.0; naux * ni * nao];
            for P in 0..naux { for Q in 0..naux {
                let vpq = vinv[P + Q * naux]; if vpq.abs() < 1e-15 { continue; }
                for idx in 0..ni * nao { coef3c[P * ni * nao + idx] += vpq * block[Q * ni * nao + idx]; }
            }}
            for P in 0..naux { for ii in 0..ni { for j in 0..nao { for occ in 0..nocc {
                rhok0_Pl_[P * nao * nocc + (p0+ii)*nocc + occ] +=
                    coef3c[P * ni * nao + ii * nao + j] * mc2[j * nocc + occ];
            }}}}
        }

        // vk1_buf: rhok0_PlJ = einsum('plj,Jj->plJ', rhok0_Pl_, mocc_2)
        // then: vk1_buf += einsum('xijp,plj->xil', ip1, rhok0_PlJ)
        let mut rhok0_PlJ = vec![0.0; naux * nao * nao];
        for P in 0..naux { for l in 0..nao { for J in 0..nao { let mut s = 0.0;
            for j in 0..nocc { s += rhok0_Pl_[P * nao * nocc + l * nocc + j] * mc2[J * nocc + j]; }
            rhok0_PlJ[P * nao * nao + l * nao + J] = s;
        }}}
        // Single aux partition: full int3c2e_ip1
        let mut vk1_buf = vec![0.0; 3 * nao * nao];
        for x in 0..3 { for i in 0..nao { for l in 0..nao { let mut s = 0.0;
            for P in 0..naux { for j in 0..nao {
                s += ip1[x * nao3 * naux + i * nao * naux + j * naux + P]
                    * rhok0_PlJ[P * nao * nao + l * nao + j];
            }} vk1_buf[x * nao3 + i * nao + l] += s;
        }}}

        // Write inputs + call Python
        let tmpdir = std::env::temp_dir().join(format!("hess_ck5_{}", std::process::id()));
        std::fs::create_dir_all(&tmpdir).unwrap();
        let atom_str = "O 0.00000000 0.00000000 0.11709209; H 0.75677522 0.00000000 -0.46836837; H -0.75677522 0.00000000 -0.46836837";
        let meta = serde_json::json!({
            "atom": atom_str, "basis": "def2-svp", "auxbasis": "def2-svp-rifit",
            "nao": nao, "nmo": nmo,
            "dm0": tmpdir.join("dm0.npy").to_str().unwrap(),
            "mo_occ": tmpdir.join("mo_occ.npy").to_str().unwrap(),
            "mo_coeff": tmpdir.join("mo_coeff.npy").to_str().unwrap(),
            "mo_energy": tmpdir.join("mo_energy.npy").to_str().unwrap(),
        });
        std::fs::write(tmpdir.join("meta.json"), serde_json::to_string(&meta).unwrap()).unwrap();
        let mut dm0_c = vec![0.0; nao * nao];
        for r in 0..nao { for c in 0..nao { dm0_c[r * nao + c] = dm0_mat[[r, c]]; }}
        write_npy_f64(tmpdir.join("dm0.npy").to_str().unwrap(), &dm0_c);
        let mut mo_occ_flat = vec![0.0; scf_data.occupation[0].len()];
        for i in 0..scf_data.occupation[0].len() { mo_occ_flat[i] = scf_data.occupation[0][i] as f64; }
        write_npy_f64(tmpdir.join("mo_occ.npy").to_str().unwrap(), &mo_occ_flat);
        let mut mc_c = vec![0.0; nao * nmo];
        for r in 0..nao { for c2 in 0..nmo { mc_c[r * nao + c2] = c[[r, c2]]; }}
        write_npy_f64(tmpdir.join("mo_coeff.npy").to_str().unwrap(), &mc_c);
        write_npy_f64(tmpdir.join("mo_energy.npy").to_str().unwrap(), eps);

        // Call ck4_vj1.py (saves both vj1 and vk1)
        let vj1_script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/hessian/pyscf_ref/ck4_vj1.py");
        let py_out = tmpdir.join("py_out");
        std::fs::create_dir_all(&py_out).unwrap();
        let status = std::process::Command::new("python3")
            .arg(vj1_script.to_str().unwrap())
            .arg(tmpdir.join("meta.json").to_str().unwrap())
            .arg(py_out.to_str().unwrap())
            .status().expect("Failed to run ck4_vj1.py");
        assert!(status.success(), "ck4_vj1.py failed");

        // Also call ck5_vk1.py for additional intermediate comparison
        let vk1_script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/hessian/pyscf_ref/ck5_vk1.py");
        let vk1_out = tmpdir.join("vk1_out");
        std::fs::create_dir_all(&vk1_out).unwrap();
        let _ = std::process::Command::new("python3")
            .arg(vk1_script.to_str().unwrap())
            .arg(tmpdir.join("meta.json").to_str().unwrap())
            .arg(vk1_out.to_str().unwrap())
            .status();

        // ── Compare rhok0_Pl_ (CK3 check) ──
        let py_rk = read_npy_f64(vk1_out.join("ck5_rhok0_Pl_.npy").to_str().unwrap());
        let mut rk_md = 0.0;
        for i in 0..naux*nao*nocc { let d = (rhok0_Pl_[i] - py_rk[i]).abs(); if d > rk_md { rk_md = d; }}
        println!("  rhok0_Pl_ vs Py: maxdiff={:.4e}", rk_md);
        assert!(rk_md < 1e-8, "CK5 rhok0_Pl_ mismatch: {:.4e}", rk_md);

        // ── Compare vk1_buf ──
        let py_vk1buf = read_npy_f64(vk1_out.join("ck5_vk1buf.npy").to_str().unwrap());
        let mut buf_md = 0.0;
        for i in 0..3*nao3 { let d = (vk1_buf[i] - py_vk1buf[i]).abs(); if d > buf_md { buf_md = d; }}
        println!("  vk1_buf vs Py: maxdiff={:.4e}", buf_md);
        assert!(buf_md < 1e-7, "CK5 vk1_buf mismatch: {:.4e}", buf_md);

        // ── Compare per-atom vk1 ──
        let mut maxdiff = 0.0;
        for ia in 0..natm {
            let shl0 = aoslices[ia][0] as usize; let shl1 = aoslices[ia][1] as usize;
            let p0 = aoslices[ia][2] as usize; let p1 = aoslices[ia][3] as usize; let ni = p1 - p0;

            // rhok0_PlJ = einsum('plj,Jj->plJ', rhok0_Pl_, mc2[p0:p1])  → (naux, nao, ni)
            let mut rhok0_PlJ_a = vec![0.0; naux * nao * ni];
            for P in 0..naux { for l in 0..nao { for J in 0..ni { let mut s = 0.0;
                for j in 0..nocc {
                    s += rhok0_Pl_[P * nao * nocc + l * nocc + j] * mc2[(p0+J)*nocc + j];
                } rhok0_PlJ_a[P * nao * ni + l * ni + J] = s;
            }}}
            // vk1[x,k,j] = -Σ_{P,i} ip1_atom[x,i,j,P] * rhok0_PlJ[P,k,i]
            // ip1_atom: (3, ni, nao, naux), rhok0_PlJ: (naux, nao, ni)
            let atom_slc: &[[usize; 2]] = &[[shl0, shl1], [0, nreg], [nreg, nreg + naux_shell]];
            let (ip1_a, _): (Vec<f64>, Vec<usize>) =
                cint_all.integrate_row_major("int3c2e_ip1", "s1", Some(atom_slc)).into();
            let mut vk1_raw = vec![0.0; 3 * nao3];
            for x in 0..3 { for k in 0..nao { for jj in 0..nao { let mut s = 0.0;
                for P in 0..naux { for ii in 0..ni {
                    s += ip1_a[x * ni * nao * naux + ii * nao * naux + jj * naux + P]
                        * rhok0_PlJ_a[P * nao * ni + k * ni + ii];
                }} vk1_raw[x * nao3 + k * nao + jj] = -s;
            }}}
            // vk1_raw[:,p0:p1] -= vk1_buf[:,p0:p1]
            for x in 0..3 { for i in p0..p1 { for j in 0..nao {
                vk1_raw[x * nao3 + i * nao + j] -= vk1_buf[x * nao3 + i * nao + j];
            }}}
            // symmetrize
            for x in 0..3 { for i in 0..nao { for j in (i+1)..nao {
                let a = x * nao3 + i * nao + j;
                let b = x * nao3 + j * nao + i;
                let sym = vk1_raw[a] + vk1_raw[b]; vk1_raw[a] = sym; vk1_raw[b] = sym;
            }}
                for i in 0..nao { vk1_raw[x * nao3 + i * nao + i] *= 2.0; }
            }
            // Compare against _gen_jk yield
            let py_path = py_out.join(format!("ck5_vk1_{}.npy", ia));
            let py_vk1 = read_npy_f64(py_path.to_str().unwrap());
            assert_eq!(py_vk1.len(), 3 * nao3);
            let mut ia_md = 0.0;
            for i in 0..3*nao3 { let d = (vk1_raw[i] - py_vk1[i]).abs(); if d > ia_md { ia_md = d; }}
            println!("  ia={} vk1: maxdiff={:.4e}", ia, ia_md);
            if ia_md > maxdiff { maxdiff = ia_md; }
        }
        println!("CK5 vk1 maxdiff={:.4e} (tol=1e-5)", maxdiff);
        assert!(maxdiff < 1e-5, "CK5 vk1 mismatch: maxdiff={:.4e}", maxdiff);
        println!("PASS: CK5 vk1");
        let _ = std::fs::remove_dir_all(&tmpdir);
    }

    /// CK1: rho0_Pij, CK2: rhoj0_P, CK3: rhok0_Pl_
    /// Validated against on-the-fly PySCF reference via ck123_rho.py.
    #[test]
    fn test_ck123_rho() {
        let (mut scf_data, nao, natm, nocc, nmo) = setup_h2o_ri();
        let aoslices = scf_data.mol.aoslice_by_atom();
        let cint_reg = scf_data.mol.initialize_cint(false); let nreg = cint_reg.nbas();
        let cint_all = scf_data.mol.initialize_cint(true);
        let naux_shell = cint_all.nbas() - nreg;
        let auxmol = scf_data.mol.make_auxmol_fake(); let naux = auxmol.num_basis;

        // V and V^{-1}
        let aux_slc_arr = [[nreg, nreg + naux_shell], [nreg, nreg + naux_shell]];
        let aux_slc: &[[usize; 2]] = &aux_slc_arr[..];
        let (int2c_v, _): (Vec<f64>, Vec<usize>) =
            cint_all.integrate_row_major("int2c2e", "s1", Some(aux_slc)).into();
        let int2c = if int2c_v.len() == naux * naux { int2c_v.clone() } else {
            let mut v = vec![0.0; naux * naux]; let mut idx = 0;
            for j in 0..naux { for i in 0..=j { v[i + j * naux] = int2c_v[idx]; v[j + i * naux] = int2c_v[idx]; idx += 1; }} v
        };
        let mut int2c_cm = vec![0.0; naux * naux];
        for p in 0..naux { for q in 0..naux { int2c_cm[p + q * naux] = int2c[p * naux + q]; }}
        let vinv = compute_vinv(&int2c_cm, naux);

        // int3c2e
        let slc_3c: &[[usize; 2]] = &[[0, nreg], [0, nreg], [nreg, nreg + naux_shell]];
        let (t3c, _): (Vec<f64>, Vec<usize>) =
            cint_all.integrate_row_major("int3c2e", "s1", Some(slc_3c)).into();

        // SCF data
        let dm0_mat = &scf_data.density_matrix[0];
        let mut dm0 = vec![0.0; nao * nao];
        for r in 0..nao { for c in 0..nao { dm0[r * nao + c] = dm0_mat[[r, c]]; }}
        let c = &scf_data.eigenvectors[0];
        let mo_occ_data = &scf_data.occupation[0];
        let mut mc2 = vec![0.0; nao * nocc];
        let eps = &scf_data.eigenvalues[0];
        for p in 0..nao { for i in 0..nocc { mc2[p * nocc + i] = c[[p, i]] * (mo_occ_data[i] as f64).sqrt(); }}

        // ── CK1: rho0_Pij per atom ──
        let mut rho0_Pij_all: Vec<Vec<f64>> = Vec::with_capacity(natm);
        for ia in 0..natm {
            let p0 = aoslices[ia][2] as usize; let p1 = aoslices[ia][3] as usize; let ni = p1 - p0;
            let mut block = vec![0.0; naux * ni * nao];
            for P in 0..naux { for ii in 0..ni { for j in 0..nao {
                block[P * ni * nao + ii * nao + j] = t3c[(p0+ii)*nao*naux + j*naux + P];
            }}}
            let mut coef3c = vec![0.0; naux * ni * nao];
            for P in 0..naux { for Q in 0..naux {
                let vpq = vinv[P + Q * naux]; if vpq.abs() < 1e-15 { continue; }
                for idx in 0..ni * nao { coef3c[P * ni * nao + idx] += vpq * block[Q * ni * nao + idx]; }
            }}
            rho0_Pij_all.push(coef3c);
        }

        // ── CK2: rhoj0_P ──
        let mut rhoj0_P = vec![0.0; naux];
        for ia in 0..natm {
            let p0 = aoslices[ia][2] as usize; let p1 = aoslices[ia][3] as usize; let ni = p1 - p0;
            let coef3c = &rho0_Pij_all[ia];
            for P in 0..naux { for ii in 0..ni { for j in 0..nao {
                rhoj0_P[P] += coef3c[P * ni * nao + ii * nao + j] * dm0[(p0+ii)*nao + j];
            }}}
        }

        // ── CK3: rhok0_Pl_ ──
        let mut rhok0_Pl_ = vec![0.0; naux * nao * nocc];
        for ia in 0..natm {
            let p0 = aoslices[ia][2] as usize; let p1 = aoslices[ia][3] as usize; let ni = p1 - p0;
            let coef3c = &rho0_Pij_all[ia];
            for P in 0..naux { for ii in 0..ni { for j in 0..nao { for occ in 0..nocc {
                rhok0_Pl_[P * nao * nocc + (p0+ii)*nocc + occ] +=
                    coef3c[P * ni * nao + ii * nao + j] * mc2[j * nocc + occ];
            }}}}
        }

        // ── Write inputs + call Python ──
        let tmpdir = std::env::temp_dir().join(format!("hess_ck123_{}", std::process::id()));
        std::fs::create_dir_all(&tmpdir).unwrap();
        let atom_str = "O 0.00000000 0.00000000 0.11709209; H 0.75677522 0.00000000 -0.46836837; H -0.75677522 0.00000000 -0.46836837";
        let meta = serde_json::json!({
            "atom": atom_str, "basis": "def2-svp", "auxbasis": "def2-svp-rifit",
            "nao": nao, "nmo": nmo,
            "dm0": tmpdir.join("dm0.npy").to_str().unwrap(),
            "mo_occ": tmpdir.join("mo_occ.npy").to_str().unwrap(),
            "mo_coeff": tmpdir.join("mo_coeff.npy").to_str().unwrap(),
            "mo_energy": tmpdir.join("mo_energy.npy").to_str().unwrap(),
        });
        std::fs::write(tmpdir.join("meta.json"), serde_json::to_string(&meta).unwrap()).unwrap();

        let mut dm0_c = vec![0.0; nao * nao];
        for r in 0..nao { for c in 0..nao { dm0_c[r * nao + c] = dm0_mat[[r, c]]; }}
        write_npy_f64(tmpdir.join("dm0.npy").to_str().unwrap(), &dm0_c);
        let mut mo_occ_flat = vec![0.0; scf_data.occupation[0].len()];
        for i in 0..scf_data.occupation[0].len() { mo_occ_flat[i] = scf_data.occupation[0][i] as f64; }
        write_npy_f64(tmpdir.join("mo_occ.npy").to_str().unwrap(), &mo_occ_flat);
        let mut mc_c = vec![0.0; nao * nmo];
        for r in 0..nao { for c2 in 0..nmo { mc_c[r * nao + c2] = c[[r, c2]]; }}
        write_npy_f64(tmpdir.join("mo_coeff.npy").to_str().unwrap(), &mc_c);
        write_npy_f64(tmpdir.join("mo_energy.npy").to_str().unwrap(), eps);

        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/hessian/pyscf_ref/ck123_rho.py");
        let py_out = tmpdir.join("py_out");
        std::fs::create_dir_all(&py_out).unwrap();
        let status = std::process::Command::new("python3")
            .arg(script.to_str().unwrap())
            .arg(tmpdir.join("meta.json").to_str().unwrap())
            .arg(py_out.to_str().unwrap())
            .status().expect("Failed to run ck123_rho.py");
        assert!(status.success(), "ck123_rho.py failed");

        // ── Compare ──
        let mut tol = 1e-8;
        // CK1
        for ia in 0..natm {
            let p0 = aoslices[ia][2] as usize; let p1 = aoslices[ia][3] as usize; let ni = p1 - p0;
            let py_data = read_npy_f64(py_out.join(format!("ck1_Pij_{}.npy", ia)).to_str().unwrap());
            assert_eq!(py_data.len(), naux * ni * nao);
            let mut md = 0.0; for i in 0..py_data.len() { let d = (rho0_Pij_all[ia][i] - py_data[i]).abs(); if d > md { md = d; }}
            assert!(md < tol, "CK1 ia={} maxdiff={:.4e}", ia, md);
        }
        // CK2
        { let py_rj = read_npy_f64(py_out.join("ck2_rhoj0_P.npy").to_str().unwrap());
          let mut md = 0.0; for i in 0..naux { let d = (rhoj0_P[i] - py_rj[i]).abs(); if d > md { md = d; }}
          assert!(md < tol, "CK2 maxdiff={:.4e}", md); }
        // CK3
        { let py_rk = read_npy_f64(py_out.join("ck3_rhok0_Pl_.npy").to_str().unwrap());
          let mut md = 0.0; for i in 0..naux*nao*nocc { let d = (rhok0_Pl_[i] - py_rk[i]).abs(); if d > md { md = d; }}
          assert!(md < tol, "CK3 maxdiff={:.4e}", md); }

        println!("PASS: CK123 (tol={:.0e})", tol);
        let _ = std::fs::remove_dir_all(&tmpdir);
    }

    /// CK4: vj1[ia] for each atom, verified against on-the-fly PySCF _gen_jk.
    #[test]
    fn test_ck4_vj1() {
        let (mut scf_data, nao, natm, nocc, nmo) = setup_h2o_ri();
        let aoslices = scf_data.mol.aoslice_by_atom();
        let cint_reg = scf_data.mol.initialize_cint(false); let nreg = cint_reg.nbas();
        let cint_all = scf_data.mol.initialize_cint(true);
        let naux_shell = cint_all.nbas() - nreg;
        let auxmol = scf_data.mol.make_auxmol_fake(); let naux = auxmol.num_basis;

        // V, V^{-1}
        let aux_slc_arr = [[nreg, nreg + naux_shell], [nreg, nreg + naux_shell]];
        let aux_slc: &[[usize; 2]] = &aux_slc_arr[..];
        let (int2c_v, _): (Vec<f64>, Vec<usize>) =
            cint_all.integrate_row_major("int2c2e", "s1", Some(aux_slc)).into();
        let int2c = if int2c_v.len() == naux * naux { int2c_v.clone() } else {
            let mut v = vec![0.0; naux * naux]; let mut idx = 0;
            for j in 0..naux { for i in 0..=j { v[i + j * naux] = int2c_v[idx]; v[j + i * naux] = int2c_v[idx]; idx += 1; }} v
        };
        let mut int2c_cm = vec![0.0; naux * naux];
        for p in 0..naux { for q in 0..naux { int2c_cm[p + q * naux] = int2c[p * naux + q]; }}
        let vinv = compute_vinv(&int2c_cm, naux);

        // 3c integrals
        let slc_3c: &[[usize; 2]] = &[[0, nreg], [0, nreg], [nreg, nreg + naux_shell]];
        let (t3c, _): (Vec<f64>, Vec<usize>) =
            cint_all.integrate_row_major("int3c2e", "s1", Some(slc_3c)).into();
        let (ip1, _): (Vec<f64>, Vec<usize>) =
            cint_all.integrate_row_major("int3c2e_ip1", "s1", Some(slc_3c)).into();

        // SCF data
        let dm0_mat = &scf_data.density_matrix[0];
        let mut dm0 = vec![0.0; nao * nao];
        for r in 0..nao { for c in 0..nao { dm0[r * nao + c] = dm0_mat[[r, c]]; }}
        let c = &scf_data.eigenvectors[0];
        let mo_occ_data = &scf_data.occupation[0];
        let eps = &scf_data.eigenvalues[0];
        let mut mc2 = vec![0.0; nao * nocc];
        for p in 0..nao { for i in 0..nocc { mc2[p * nocc + i] = c[[p, i]] * (mo_occ_data[i] as f64).sqrt(); }}

        // ── rho0_Pij per atom + rho0_full + rhoj0_P ──
        let mut rho0_Pij_all: Vec<Vec<f64>> = Vec::with_capacity(natm);
        for ia in 0..natm {
            let p0 = aoslices[ia][2] as usize; let p1 = aoslices[ia][3] as usize; let ni = p1 - p0;
            let mut block = vec![0.0; naux * ni * nao];
            for P in 0..naux { for ii in 0..ni { for j in 0..nao {
                block[P * ni * nao + ii * nao + j] = t3c[(p0+ii)*nao*naux + j*naux + P];
            }}}
            let mut coef3c = vec![0.0; naux * ni * nao];
            for P in 0..naux { for Q in 0..naux {
                let vpq = vinv[P + Q * naux]; if vpq.abs() < 1e-15 { continue; }
                for idx in 0..ni * nao { coef3c[P * ni * nao + idx] += vpq * block[Q * ni * nao + idx]; }
            }}
            rho0_Pij_all.push(coef3c);
        }
        let nao3 = nao * nao;
        let mut rho0_full = vec![0.0; naux * nao * nao];
        for ia in 0..natm {
            let p0 = aoslices[ia][2] as usize; let p1 = aoslices[ia][3] as usize; let ni = p1 - p0;
            let co = &rho0_Pij_all[ia];
            for P in 0..naux { for ii in 0..ni { for j in 0..nao {
                rho0_full[P * nao3 + (p0+ii)*nao + j] = co[P * ni * nao + ii * nao + j];
            }}}
        }
        let mut rhoj0_P = vec![0.0; naux];
        for ia in 0..natm {
            let p0 = aoslices[ia][2] as usize; let p1 = aoslices[ia][3] as usize; let ni = p1 - p0;
            let co = &rho0_Pij_all[ia];
            for P in 0..naux { for ii in 0..ni { for j in 0..nao {
                rhoj0_P[P] += co[P * ni * nao + ii * nao + j] * dm0[(p0+ii)*nao + j];
            }}}
        }

        // ── vj1_buf[ia] via per-atom-block wj1 (matching PySCF L433-435) ──
        let mut vj1_buf = vec![0.0; natm * 3 * nao3];
        for ia in 0..natm {
            let q0 = aoslices[ia][2] as usize; let q1 = aoslices[ia][3] as usize;
            let mut wj1 = vec![0.0; 3 * naux];
            for x in 0..3 { for P in 0..naux { let mut s = 0.0;
                for k in q0..q1 { for l in 0..nao {
                    s += ip1[x * nao3 * naux + k * nao * naux + l * naux + P] * dm0[l * nao + k];
                }} wj1[x * naux + P] = s;
            }}
            for x in 0..3 { for P in 0..naux { let wp = wj1[x * naux + P]; if wp.abs() < 1e-15 { continue; }
                for i in 0..nao { for j in 0..nao {
                    vj1_buf[ia * 3 * nao3 + x * nao3 + i * nao + j] += wp * rho0_full[P * nao3 + i * nao + j];
                }}
            }}
        }

        // ── Per-atom vj1 assembly (matching PySCF L443-471) ──
        let tmpdir = std::env::temp_dir().join(format!("hess_ck4vj1_{}", std::process::id()));
        std::fs::create_dir_all(&tmpdir).unwrap();
        let atom_str = "O 0.00000000 0.00000000 0.11709209; H 0.75677522 0.00000000 -0.46836837; H -0.75677522 0.00000000 -0.46836837";
        let meta = serde_json::json!({
            "atom": atom_str, "basis": "def2-svp", "auxbasis": "def2-svp-rifit",
            "nao": nao, "nmo": nmo,
            "dm0": tmpdir.join("dm0.npy").to_str().unwrap(),
            "mo_occ": tmpdir.join("mo_occ.npy").to_str().unwrap(),
            "mo_coeff": tmpdir.join("mo_coeff.npy").to_str().unwrap(),
            "mo_energy": tmpdir.join("mo_energy.npy").to_str().unwrap(),
        });
        std::fs::write(tmpdir.join("meta.json"), serde_json::to_string(&meta).unwrap()).unwrap();
        let mut dm0_c = vec![0.0; nao * nao];
        for r in 0..nao { for c in 0..nao { dm0_c[r * nao + c] = dm0_mat[[r, c]]; }}
        write_npy_f64(tmpdir.join("dm0.npy").to_str().unwrap(), &dm0_c);
        let mut mo_occ_flat = vec![0.0; scf_data.occupation[0].len()];
        for i in 0..scf_data.occupation[0].len() { mo_occ_flat[i] = scf_data.occupation[0][i] as f64; }
        write_npy_f64(tmpdir.join("mo_occ.npy").to_str().unwrap(), &mo_occ_flat);
        let mut mc_c = vec![0.0; nao * nmo];
        for r in 0..nao { for c2 in 0..nmo { mc_c[r * nao + c2] = c[[r, c2]]; }}
        write_npy_f64(tmpdir.join("mo_coeff.npy").to_str().unwrap(), &mc_c);
        write_npy_f64(tmpdir.join("mo_energy.npy").to_str().unwrap(), eps);

        // Call Python to get PySCF reference vj1
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/hessian/pyscf_ref/ck4_vj1.py");
        let py_out = tmpdir.join("py_out");
        std::fs::create_dir_all(&py_out).unwrap();
        let status = std::process::Command::new("python3")
            .arg(script.to_str().unwrap())
            .arg(tmpdir.join("meta.json").to_str().unwrap())
            .arg(py_out.to_str().unwrap())
            .status().expect("Failed to run ck4_vj1.py");
        assert!(status.success(), "ck4_vj1.py failed");

        // ── Compute REST vj1, compare with PySCF ──
        let mut maxdiff = 0.0;
        for ia in 0..natm {
            let shl0 = aoslices[ia][0] as usize; let shl1 = aoslices[ia][1] as usize;
            let p0 = aoslices[ia][2] as usize; let p1 = aoslices[ia][3] as usize;
            let ni = p1 - p0;
            let off = ia * 3 * nao3;

            // Step 1: vj1 = -vj1_buf[ia]
            let mut vj1 = vec![0.0; 3 * nao3];
            for i in 0..3*nao3 { vj1[i] = -vj1_buf[off + i]; }

            // Step 2: vj1[:,p0:p1] -= int3c_ip1[:,p0:p1,:] · rhoj0_P  (shell-sliced correction)
            let atom_slc: &[[usize; 2]] = &[[shl0, shl1], [0, nreg], [nreg, nreg + naux_shell]];
            let (ip1_atom, _): (Vec<f64>, Vec<usize>) =
                cint_all.integrate_row_major("int3c2e_ip1", "s1", Some(atom_slc)).into();
            // ip1_atom layout: x * ni * nao * naux + ii * nao * naux + j * naux + P
            for x in 0..3 { for ii in 0..ni { for j in 0..nao { let mut s = 0.0;
                for P in 0..naux { s += ip1_atom[x * ni * nao * naux + ii * nao * naux + j * naux + P] * rhoj0_P[P]; }
                vj1[x * nao3 + (p0+ii)*nao + j] -= s;
            }}}

            // Save pre-sym copy for diagnostic comparison
            let mut vj1_presym = vj1.clone();
            // Step 3: symmetrize vj1 = vj1 + vj1^T (only i < j to avoid double counting)
            for x in 0..3 { for i in 0..nao { for j in (i+1)..nao {
                let a = x * nao3 + i * nao + j;
                let b = x * nao3 + j * nao + i;
                let sym = vj1[a] + vj1[b];
                vj1[a] = sym; vj1[b] = sym;
            }}
                // diagonal: vj1[x,i,i] += vj1[x,i,i] = 2 * vj1[x,i,i]
                for i in 0..nao {
                    let d = x * nao3 + i * nao + i;
                    vj1[d] *= 2.0;
                }
            }

            // Compare with PySCF
            let py_path = py_out.join(format!("ck4_vj1_{}.npy", ia));
            let py_vj1 = read_npy_f64(py_path.to_str().unwrap());
            assert_eq!(py_vj1.len(), 3 * nao3);
            let mut ia_md = 0.0;
            for i in 0..3*nao3 {
                let d = (vj1[i] - py_vj1[i]).abs();
                if d > ia_md { ia_md = d; }
            }
            println!("  ia={}: maxdiff={:.4e}", ia, ia_md);
            if ia_md > maxdiff { maxdiff = ia_md; }
                        // Debug: verify against PySCF
            if ia == 0 {
                let bi = 1 * nao3 + 0 * nao + 4;
                println!("    vj1[0] maxdiff={:.4e}  buf[1,0,4]={:.6e}  raw={:.6e}",
                    ia_md, vj1_buf[off + bi], vj1_presym[bi]);
            }
        }
        println!("CK4 vj1 maxdiff={:.4e} (tol=1e-5)", maxdiff);
        assert!(maxdiff < 1e-5, "CK4 vj1 mismatch: maxdiff={:.4e}", maxdiff);
        println!("PASS: CK4 vj1");
        let _ = std::fs::remove_dir_all(&tmpdir);
    }

    /// CK4 with NH3 molecule, confirming against PySCF _gen_jk.
    #[test]
    fn test_ck4_nh3() {
        let input_token = r###"
[ctrl]
xc = "hf"
basis_path = "basis-set-pool/def2-svp"
auxbas_path = "basis-set-pool/def2-svp-rifit"
basis_type = "spheric"
auxbas_type = "spheric"
eri_type = "ri-v"
charge = 0.0
spin = 1.0
num_threads = 1
print_level = 0
[geom]
name = "NH3"
unit = "Angstrom"
position = """
    N    0.00000000   0.00000000   0.00000000
    H    0.94300000   0.00000000  -0.27200000
    H   -0.47100000   0.81600000  -0.27200000
    H   -0.47100000  -0.81600000  -0.27200000
"""
"###;
        let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
        let (ctrl, geom) = crate::ctrl_io::parse_ctl_from_json(&keys).unwrap();
        let mol = crate::Molecule::build_native(ctrl, geom, None).unwrap();
        let mut scf_data = scf_io::SCF::build(mol, &None);
        crate::scf_io::scf_without_build(&mut scf_data, &None);
        let nao = scf_data.mol.num_basis; let natm = scf_data.mol.geom.nfree;
        let nocc = (scf_data.homo[0] + 1) as usize; let nmo = scf_data.eigenvalues[0].len();
        let aoslices = scf_data.mol.aoslice_by_atom();
        let cint_reg = scf_data.mol.initialize_cint(false); let nreg = cint_reg.nbas();
        let cint_all = scf_data.mol.initialize_cint(true);
        let naux_shell = cint_all.nbas() - nreg;
        let auxmol = scf_data.mol.make_auxmol_fake(); let naux = auxmol.num_basis;

        let aux_slc_arr = [[nreg, nreg + naux_shell], [nreg, nreg + naux_shell]];
        let aux_slc: &[[usize; 2]] = &aux_slc_arr[..];
        let (int2c_v, _): (Vec<f64>, Vec<usize>) =
            cint_all.integrate_row_major("int2c2e", "s1", Some(aux_slc)).into();
        let int2c = if int2c_v.len() == naux * naux { int2c_v.clone() } else {
            let mut v = vec![0.0; naux * naux]; let mut idx = 0;
            for j in 0..naux { for i in 0..=j { v[i + j * naux] = int2c_v[idx]; v[j + i * naux] = int2c_v[idx]; idx += 1; }} v
        };
        let mut int2c_cm = vec![0.0; naux * naux];
        for p in 0..naux { for q in 0..naux { int2c_cm[p + q * naux] = int2c[p * naux + q]; }}
        let vinv = compute_vinv(&int2c_cm, naux);

        let slc_3c: &[[usize; 2]] = &[[0, nreg], [0, nreg], [nreg, nreg + naux_shell]];
        let (t3c, _): (Vec<f64>, Vec<usize>) =
            cint_all.integrate_row_major("int3c2e", "s1", Some(slc_3c)).into();
        let (ip1, _): (Vec<f64>, Vec<usize>) =
            cint_all.integrate_row_major("int3c2e_ip1", "s1", Some(slc_3c)).into();

        let dm0_mat = &scf_data.density_matrix[0];
        let mut dm0 = vec![0.0; nao * nao];
        for r in 0..nao { for c in 0..nao { dm0[r * nao + c] = dm0_mat[[r, c]]; }}
        let c = &scf_data.eigenvectors[0];
        let mo_occ_data = &scf_data.occupation[0];
        let eps = &scf_data.eigenvalues[0];
        let mut mc2 = vec![0.0; nao * nocc];
        for p in 0..nao { for i in 0..nocc { mc2[p * nocc + i] = c[[p, i]] * (mo_occ_data[i] as f64).sqrt(); }}

        // rho0_Pij + rhoj0_P + rho0_full
        let mut rho0_all: Vec<Vec<f64>> = Vec::with_capacity(natm);
        for ia in 0..natm {
            let p0 = aoslices[ia][2] as usize; let p1 = aoslices[ia][3] as usize; let ni = p1 - p0;
            let mut block = vec![0.0; naux * ni * nao];
            for P in 0..naux { for ii in 0..ni { for j in 0..nao {
                block[P * ni * nao + ii * nao + j] = t3c[(p0+ii)*nao*naux + j*naux + P];
            }}}
            let mut coef = vec![0.0; naux * ni * nao];
            for P in 0..naux { for Q in 0..naux {
                let vpq = vinv[P + Q * naux]; if vpq.abs() < 1e-15 { continue; }
                for idx in 0..ni * nao { coef[P * ni * nao + idx] += vpq * block[Q * ni * nao + idx]; }
            }}
            rho0_all.push(coef);
        }
        let nao3 = nao * nao;
        let mut rho0_full = vec![0.0; naux * nao3];
        for ia in 0..natm {
            let p0 = aoslices[ia][2] as usize; let p1 = aoslices[ia][3] as usize; let ni = p1 - p0;
            let co = &rho0_all[ia];
            for P in 0..naux { for ii in 0..ni { for j in 0..nao {
                rho0_full[P * nao3 + (p0+ii)*nao + j] = co[P * ni * nao + ii * nao + j];
            }}}
        }
        let mut rhoj0_P = vec![0.0; naux];
        for ia in 0..natm {
            let p0 = aoslices[ia][2] as usize; let p1 = aoslices[ia][3] as usize; let ni = p1 - p0;
            let co = &rho0_all[ia];
            for P in 0..naux { for ii in 0..ni { for j in 0..nao {
                rhoj0_P[P] += co[P * ni * nao + ii * nao + j] * dm0[(p0+ii)*nao + j];
            }}}
        }

        // vj1_buf
        let mut vj1_buf = vec![0.0; natm * 3 * nao3];
        for ia in 0..natm {
            let q0 = aoslices[ia][2] as usize; let q1 = aoslices[ia][3] as usize;
            let mut wj1 = vec![0.0; 3 * naux];
            for x in 0..3 { for P in 0..naux { let mut s = 0.0;
                for k in q0..q1 { for l in 0..nao {
                    s += ip1[x * nao3 * naux + k * nao * naux + l * naux + P] * dm0[l * nao + k];
                }} wj1[x * naux + P] = s;
            }}
            for x in 0..3 { for P in 0..naux { let wp = wj1[x * naux + P]; if wp.abs() < 1e-15 { continue; }
                for i in 0..nao { for j in 0..nao {
                    vj1_buf[ia * 3 * nao3 + x * nao3 + i * nao + j] += wp * rho0_full[P * nao3 + i * nao + j];
                }}
            }}
        }

        // Write inputs
        let tmpdir = std::env::temp_dir().join(format!("hess_ck4nh3_{}", std::process::id()));
        std::fs::create_dir_all(&tmpdir).unwrap();
        let atom_str = "N 0.00000000 0.00000000 0.00000000; H 0.94300000 0.00000000 -0.27200000; H -0.47100000 0.81600000 -0.27200000; H -0.47100000 -0.81600000 -0.27200000";
        let meta = serde_json::json!({
            "atom": atom_str, "basis": "def2-svp", "auxbasis": "def2-svp-rifit",
            "nao": nao, "nmo": nmo,
            "dm0": tmpdir.join("dm0.npy").to_str().unwrap(),
            "mo_occ": tmpdir.join("mo_occ.npy").to_str().unwrap(),
            "mo_coeff": tmpdir.join("mo_coeff.npy").to_str().unwrap(),
            "mo_energy": tmpdir.join("mo_energy.npy").to_str().unwrap(),
        });
        std::fs::write(tmpdir.join("meta.json"), serde_json::to_string(&meta).unwrap()).unwrap();
        let mut dm0_c = vec![0.0; nao * nao];
        for r in 0..nao { for c in 0..nao { dm0_c[r * nao + c] = dm0_mat[[r, c]]; }}
        write_npy_f64(tmpdir.join("dm0.npy").to_str().unwrap(), &dm0_c);
        let mut mo_occ_flat = vec![0.0; scf_data.occupation[0].len()];
        for i in 0..scf_data.occupation[0].len() { mo_occ_flat[i] = scf_data.occupation[0][i] as f64; }
        write_npy_f64(tmpdir.join("mo_occ.npy").to_str().unwrap(), &mo_occ_flat);
        let mut mc_c = vec![0.0; nao * nmo];
        for r in 0..nao { for c2 in 0..nmo { mc_c[r * nao + c2] = c[[r, c2]]; }}
        write_npy_f64(tmpdir.join("mo_coeff.npy").to_str().unwrap(), &mc_c);
        write_npy_f64(tmpdir.join("mo_energy.npy").to_str().unwrap(), eps);

        // Call Python
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/hessian/pyscf_ref/ck4_vj1.py");
        let py_out = tmpdir.join("py_out");
        std::fs::create_dir_all(&py_out).unwrap();
        let status = std::process::Command::new("python3")
            .arg(script.to_str().unwrap())
            .arg(tmpdir.join("meta.json").to_str().unwrap())
            .arg(py_out.to_str().unwrap())
            .status().expect("Failed to run ck4_vj1.py");
        assert!(status.success(), "ck4_vj1.py failed");

        // Compute REST vj1 and compare
        let mut maxdiff = 0.0;
        for ia in 0..natm {
            let shl0 = aoslices[ia][0] as usize; let shl1 = aoslices[ia][1] as usize;
            let p0 = aoslices[ia][2] as usize; let p1 = aoslices[ia][3] as usize; let ni = p1 - p0;
            let off = ia * 3 * nao3;

            let mut vj1 = vec![0.0; 3 * nao3];
            for i in 0..3*nao3 { vj1[i] = -vj1_buf[off + i]; }

            let atom_slc: &[[usize; 2]] = &[[shl0, shl1], [0, nreg], [nreg, nreg + naux_shell]];
            let (ip1_atom, _): (Vec<f64>, Vec<usize>) =
                cint_all.integrate_row_major("int3c2e_ip1", "s1", Some(atom_slc)).into();
            for x in 0..3 { for ii in 0..ni { for j in 0..nao { let mut s = 0.0;
                for P in 0..naux {
                    s += ip1_atom[x * ni * nao * naux + ii * nao * naux + j * naux + P] * rhoj0_P[P];
                }
                vj1[x * nao3 + (p0+ii)*nao + j] -= s;
            }}}

            for x in 0..3 { for i in 0..nao { for j in (i+1)..nao {
                let a = x * nao3 + i * nao + j;
                let b = x * nao3 + j * nao + i;
                let sym = vj1[a] + vj1[b]; vj1[a] = sym; vj1[b] = sym;
            }}
                for i in 0..nao { vj1[x * nao3 + i * nao + i] *= 2.0; }
            }

            let py_path = py_out.join(format!("ck4_vj1_{}.npy", ia));
            let py_vj1 = read_npy_f64(py_path.to_str().unwrap());
            assert_eq!(py_vj1.len(), 3 * nao3);
            let mut ia_md = 0.0;
            for i in 0..3*nao3 { let d = (vj1[i] - py_vj1[i]).abs(); if d > ia_md { ia_md = d; }}
            println!("  ia={}: maxdiff={:.4e}", ia, ia_md);
            if ia_md > maxdiff { maxdiff = ia_md; }
        }
        println!("CK4 NH3 vj1 maxdiff={:.4e} (tol=1e-5)", maxdiff);
        assert!(maxdiff < 1e-5, "CK4 NH3 vj1 mismatch: maxdiff={:.4e}", maxdiff);
        println!("PASS: CK4 NH3");
        let _ = std::fs::remove_dir_all(&tmpdir);
    }

    /// CK6: hcore^{(1)} first derivative, verified against PySCF gradient hcore_generator.
    #[test]
    fn test_ck6_hcore() {
        let input_token = r###"
[ctrl]
xc = "hf"
basis_path = "basis-set-pool/def2-svp"
basis_type = "spheric"
charge = 0.0
spin = 1.0
num_threads = 1
print_level = 0
[geom]
name = "H2O"
unit = "Angstrom"
position = """
    O   0.00000000   0.00000000   0.11709209
    H   0.75677522   0.00000000  -0.46836837
    H  -0.75677522   0.00000000  -0.46836837
"""
"###;
        let keys = toml::from_str::<serde_json::Value>(&input_token[..]).unwrap();
        let (ctrl, geom) = crate::ctrl_io::parse_ctl_from_json(&keys).unwrap();
        let mol = crate::Molecule::build_native(ctrl, geom, None).unwrap();
        let nao = mol.num_basis; let natm = mol.geom.nfree; let nao3 = nao * nao;

        // Compute in Rust via build_hcore_first_deriv
        let mut hcore_rust: Vec<Vec<f64>> = Vec::with_capacity(natm);
        for ia in 0..natm {
            hcore_rust.push(build_hcore_first_deriv(&mol, ia));
        }

        // Write minimal inputs for Python (hcore only needs mol geometry + basis)
        let tmpdir = std::env::temp_dir().join(format!("hess_ck6_{}", std::process::id()));
        std::fs::create_dir_all(&tmpdir).unwrap();
        let atom_str = "O 0.00000000 0.00000000 0.11709209; H 0.75677522 0.00000000 -0.46836837; H -0.75677522 0.00000000 -0.46836837";
        let meta = serde_json::json!({
            "atom": atom_str, "basis": "def2-svp",
        });
        std::fs::write(tmpdir.join("meta.json"), serde_json::to_string(&meta).unwrap()).unwrap();

        // Call Python
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/hessian/pyscf_ref/ck6_hcore.py");
        let py_out = tmpdir.join("py_out");
        std::fs::create_dir_all(&py_out).unwrap();
        let status = std::process::Command::new("python3")
            .arg(script.to_str().unwrap())
            .arg(tmpdir.join("meta.json").to_str().unwrap())
            .arg(py_out.to_str().unwrap())
            .status().expect("Failed to run ck6_hcore.py");
        assert!(status.success(), "ck6_hcore.py failed");

        // Compare per atom
        let mut maxdiff = 0.0;
        for ia in 0..natm {
            let py_h = read_npy_f64(py_out.join(format!("ck6_hcore_{}.npy", ia)).to_str().unwrap());
            assert_eq!(py_h.len(), 3 * nao3);
            let rust_h = &hcore_rust[ia];
            let mut ia_md = 0.0;
            for i in 0..3*nao3 { let d = (rust_h[i] - py_h[i]).abs(); if d > ia_md { ia_md = d; }}
            println!("  ia={}: maxdiff={:.4e}", ia, ia_md);
            if ia_md > maxdiff { maxdiff = ia_md; }
        }
        println!("CK6 hcore maxdiff={:.4e} (tol=1e-5)", maxdiff);
        assert!(maxdiff < 1e-5, "CK6 hcore mismatch: maxdiff={:.4e}", maxdiff);
        println!("PASS: CK6 hcore");
        let _ = std::fs::remove_dir_all(&tmpdir);
    }

    /// CK7: full calc_h1ao() method producing h1ao[ia] = hcore + vj1 - 0.5*vk1.
    #[test]
    fn test_ck7_h1ao() {
        let (mut scf_data, nao, natm, _nocc, nmo) = setup_h2o_ri();
        let nmo = scf_data.eigenvalues[0].len(); let nao3 = nao * nao;

        // Run calc_h1ao
        let mut hess = RIRHFHessian::new(&scf_data);
        hess.calc_h1ao();
        assert_eq!(hess.h1ao.len(), natm);

        // Write inputs + call Python
        let tmpdir = std::env::temp_dir().join(format!("hess_ck7_{}", std::process::id()));
        std::fs::create_dir_all(&tmpdir).unwrap();
        let atom_str = "O 0.00000000 0.00000000 0.11709209; H 0.75677522 0.00000000 -0.46836837; H -0.75677522 0.00000000 -0.46836837";
        let meta = serde_json::json!({
            "atom": atom_str, "basis": "def2-svp", "auxbasis": "def2-svp-rifit",
            "nao": nao, "nmo": nmo,
            "dm0": tmpdir.join("dm0.npy").to_str().unwrap(),
            "mo_occ": tmpdir.join("mo_occ.npy").to_str().unwrap(),
            "mo_coeff": tmpdir.join("mo_coeff.npy").to_str().unwrap(),
            "mo_energy": tmpdir.join("mo_energy.npy").to_str().unwrap(),
        });
        std::fs::write(tmpdir.join("meta.json"), serde_json::to_string(&meta).unwrap()).unwrap();
        let dm0_mat = &scf_data.density_matrix[0];
        let mut dm0_c = vec![0.0; nao * nao];
        for r in 0..nao { for c in 0..nao { dm0_c[r * nao + c] = dm0_mat[[r, c]]; }}
        write_npy_f64(tmpdir.join("dm0.npy").to_str().unwrap(), &dm0_c);
        let mut mo_occ_flat = vec![0.0; scf_data.occupation[0].len()];
        for i in 0..scf_data.occupation[0].len() { mo_occ_flat[i] = scf_data.occupation[0][i] as f64; }
        write_npy_f64(tmpdir.join("mo_occ.npy").to_str().unwrap(), &mo_occ_flat);
        let c = &scf_data.eigenvectors[0];
        let mut mc_c = vec![0.0; nao * nmo];
        for r in 0..nao { for c2 in 0..nmo { mc_c[r * nao + c2] = c[[r, c2]]; }}
        write_npy_f64(tmpdir.join("mo_coeff.npy").to_str().unwrap(), &mc_c);
        write_npy_f64(tmpdir.join("mo_energy.npy").to_str().unwrap(), &scf_data.eigenvalues[0]);

        // Call ck4_vj1.py (saves vj1+vk1) + ck7_h1ao.py (saves h1ao)
        let vj1_script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/hessian/pyscf_ref/ck4_vj1.py");
        let h1ao_script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/hessian/pyscf_ref/ck7_h1ao.py");
        let py_out = tmpdir.join("py_out");
        std::fs::create_dir_all(&py_out).unwrap();
        for script in &[vj1_script, h1ao_script] {
            let status = std::process::Command::new("python3")
                .arg(script.to_str().unwrap())
                .arg(tmpdir.join("meta.json").to_str().unwrap())
                .arg(py_out.to_str().unwrap())
                .status().expect("Failed to call Python script");
            assert!(status.success(), "Python script failed: {:?}", script);
        }

        // Verify internal consistency: PySCF make_h1's h1ao = h1 + vj1 - 0.5*vk1
        if false { // disable verbose check; already verified by CK4-CK6
            let py_h1ao = read_npy_f64(py_out.join("ck7_h1ao_0.npy").to_str().unwrap());
            let py_vj1 = read_npy_f64(py_out.join("ck4_vj1_0.npy").to_str().unwrap());
            let py_vk1 = read_npy_f64(py_out.join("ck5_vk1_0.npy").to_str().unwrap());
            let mut py_h1 = vec![0.0; 3*nao3];
            for i in 0..3*nao3 { py_h1[i] = py_h1ao[i] - py_vj1[i] + 0.5*py_vk1[i]; }
            let rust_h1 = build_hcore_first_deriv(&scf_data.mol, 0);
            let md = (0..3*nao3).map(|i| (rust_h1[i] - py_h1[i]).abs()).fold(0.0f64, |a,b| a.max(b));
            println!("  hcore consistency: maxdiff={:.4e}", md);
        }

        // Compare
        let mut maxdiff = 0.0;
        for ia in 0..natm {
            let py_h = read_npy_f64(py_out.join(format!("ck7_h1ao_{}.npy", ia)).to_str().unwrap());
            assert_eq!(py_h.len(), 3 * nao3);
            let rust_h = &hess.h1ao[ia];
            let mut ia_md = 0.0;
            for i in 0..3*nao3 { let d = (rust_h.data[i] - py_h[i]).abs(); if d > ia_md { ia_md = d; }}
            println!("  ia={}: maxdiff={:.4e}", ia, ia_md);
            if ia_md > maxdiff { maxdiff = ia_md; }
        }
        println!("CK7 h1ao maxdiff={:.4e} (tol=1e-5)", maxdiff);
        assert!(maxdiff < 1e-5, "CK7 h1ao mismatch: maxdiff={:.4e}", maxdiff);
        println!("PASS: CK7 h1ao");
        let _ = std::fs::remove_dir_all(&tmpdir);
    }
}
