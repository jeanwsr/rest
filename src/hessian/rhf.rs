use crate::hessian::memory_monitor::{self, MemMonitor};
use crate::scf_io;
use crate::scf_io::SCF;
use crate::Molecule;
use rest_libcint::prelude::*;
use rstsr::prelude::*;
use std::collections::HashMap;
use tensors::matrix_blas_lapack::_power_rayon_for_symmetric_matrix;
use tensors::MatrixFull;

use crate::constants::{FQ, HARTREE2WAVENUMBER};
use crate::utilities::rstsr_util::*;

const CPHF_KRYLOV_MAX_CYCLE: usize = 50;
const CPHF_KRYLOV_TOL: f64 = 1.0e-12;

/// Read-only bundle of all Phase 1-3 intermediates needed by `_blas` functions.
///
/// Constructed for each G-term through `build_ej_ek_ctx!`. Phase 1-3
/// intermediates are read-only after Phase 3 ends.
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
    pub wj2: &'a [f64],
    pub wk2: &'a [f64],
    pub rkoo: &'a [f64],
    pub r2c0: &'a [f64],
    pub wj001: &'a [f64],
    pub ip1: &'a [f64],
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
fn int3c_atom_block(
    cint_all: &CINTR2CDATA,
    name: &str,
    shl0: usize,
    shl1: usize,
    nreg: usize,
    aux_nbas: usize,
) -> Vec<f64> {
    let slc: &[[usize; 2]] = &[[shl0, shl1], [0, nreg], [nreg, nreg + aux_nbas]];
    let (v, _): (Vec<f64>, Vec<usize>) = cint_all.integrate_row_major(name, "s1", Some(slc)).into();
    v
}

// ── BLAS-optimized G-term implementations ──

/// G3 (ej_vj1) + g4 (ek_vk1) combined, per-AO-atom streaming implementation.
///
/// The 9·N²·P derivative integral `int3c2e_ipvip1` is never materialized in
/// full: it is evaluated per-AO-atom block (9·ni·N·P) via a libcint shell
/// slice inside the i0 loop and consumed immediately by both G3 and g4, so
/// the full tensor never coexists with ip1/tmpf/rhok. `do_k` gates the
/// exchange part (g4): pure LDA/GGA DFAs (factor_k == 0) skip it.
fn g3_g4_vj1_vk1_blas(ctx: &EjEkContext, out_vj1: &mut [f64], out_vk1: &mut [f64], do_k: bool) {
    let nao = ctx.nao;
    let naux = ctx.naux;
    let nocc = ctx.nocc;
    let natm = ctx.natm;
    let nao3 = ctx.nao3;
    let blk = ctx.blk;
    let dm0 = ctx.dm0;
    let mc2 = ctx.mc2;
    let rk = ctx.rk;
    let r0 = ctx.r0;
    let i2inv = ctx.i2inv;
    let device = ctx.device.clone();

    let i_t = |i0: usize, j0: usize, x: usize, y: usize| -> usize {
        i0 * natm * 9 + j0 * 9 + x * 3 + y
    };

    // ══════════════════════════════════════════════════════════════
    // G4 (ek_vk1) — C_occ-factorized (plan B): no rhok/vk2buf staging.
    //
    // Full g4 identity (verified equivalent to the step6 part1+part2+part3):
    //   ek_vk1[i0,j0,x,y] = A + B + P2
    //   A = Σ_{p,a,b} W[x,p,a,b]·Z[p,y,b,a]
    //       W[x,p,a,b] = Σ_{i∈i0} W2[x,i,p,b]·mc2[i,a]
    //       Z[p,y,b,a] = Σ_{k∈j0} Z2[p,y,k,a]·mc2[k,b]
    //   B = Σ_{i∈i0,k∈j0} H[x,i,y,k]·dm0[q0+k, p0+i]
    //       H[x,i,y,k] = Σ_{p,a} W2[x,i,p,a]·Z2[p,y,k,a]
    //   P2 = Σ_{i∈i0,j∈j0,p} ipv[c,i,j,p]·V[p,i,j]
    //       V[p,i,j] = Σ_a Ua[p,i,a]·mc2[j,a]
    //       Ua[p,i,a] = Σ_o mc2[i,o]·Q[p,o,a],  Q[p,o,a] = Σ_k rk[p,k,o]·mc2[k,a]
    // where W2[x,i,p,a] = Σ_j ip1[x,i,j,p]·mc2[j,a],
    //       Z2[p,y,k,a] = Σ_l tmpf[p,y,k,l]·mc2[l,a]  (tmpf = V⁻¹·ip1; the
    //       V⁻¹ transform is deferred: Z2u = Σ_l ip1·mc2[l,a], Z2 = V⁻¹·Z2u).
    // Only per-atom blocks of ip1/ipv are touched; the 3·N²·P rhok, 9·N²
    // vk2buf and 3·N²·P tmpf intermediates are eliminated.
    // ══════════════════════════════════════════════════════════════
    let mc2_t_full = rt::asarray((mc2, [nocc, nao].f(), &device)); // row a, col j
    let i2inv_t = rt::asarray((i2inv, [naux, naux].f(), &device));
    // Q_T [O, P·O] F-order: row a, col (p,o);  Q_T[a + (p·O+o)·O] = Q[p,o,a]
    //   Q[p,o,a] = Σ_k rk[p,k,o]·mc2[k,a]
    // Built directly in the transposed layout (instead of staging rk into
    // [P·O, N] + GEMM + per-atom Q^T restaging) — the old rk→[m,nao] staging
    // wrote with a 239 KB column stride (measured ~2.4 s of g3_g4's 18.5 s).
    // The direct build reads rk with p contiguous (stride 1) and writes
    // q_t in contiguous 34-element rows; no full Q intermediate, no Q^T stage.
    let q_t_raw: Vec<f64>;
    {
        assert!(nocc <= 64, "Q_T direct build assumes nocc <= 64");
        let mut qt = vec![0.0; nocc * naux * nocc];
        for p in 0..naux {
            for o in 0..nocc {
                let mut row = [0.0f64; 64];
                let rbase = o * naux * nao;
                for k in 0..nao {
                    let rv = rk[p + k * naux + rbase];
                    for a in 0..nocc { row[a] += rv * mc2[k * nocc + a]; }
                }
                let qbase = (p * nocc + o) * nocc;
                for a in 0..nocc { qt[qbase + a] = row[a]; }
            }
        }
        q_t_raw = qt;
    }
    let n_batch = if do_k { 2 } else { 1 };
    let mut z2_per_atom: Vec<Vec<f64>> = vec![Vec::new(); natm];
    // Z2 atoms built in two batches (half the aux atoms each) so the
        // ── Z2_per_atom[j0] = [P·3·qj, O] F-order: row (p,y,k), col a
        //    Z2[p,y,k,a] = Σ_l tmpf[p,y,q0+k,l]·mc2[l,a]
        //    Deferred V⁻¹: Z2u[p,y,k,a] = Σ_l ip1[y,q0+k,l,p]·mc2[l,a],
        //    then Z2 = V⁻¹·Z2u  (avoids the 3·N²·P tmpf intermediate) ──
        // Z2 atoms built in two batches (half the aux atoms each) so the
        // 0.8 GB z2_per_atom never coexists with the ipv block. The whole
        // i0 loop runs per batch (G3 re-assigns the same values — fine);
        // G4's j0 loop skips z2 atoms not in the current batch.
        for half in 0..n_batch {
            let j0_lo = half * natm / 2;
            let j0_hi = (half + 1) * natm / 2;
            if do_k {
            for j0 in j0_lo..j0_hi {
            let (_, q0, qj) = blk[j0];
            if qj == 0 { continue; }
            let m = naux * 3 * qj;
            // Z2u [P·3·qj, O] F-order: row (p,y,k), col a = Σ_l ip1[y,q0+k,l,p]·mc2[l,a]
            // ip1[j0-row-block] integrated once per atom (234 MB) and dropped;
            // the full 2.7 GB int3c2e_ip1 is never built/read here.
            let shl0z = ctx.aoslices[j0][0] as usize;
            let shl1z = ctx.aoslices[j0][1] as usize;
            let ip1_slc: &[[usize; 2]] = &[[shl0z, shl1z], [0, ctx.nreg],
                [ctx.nreg, ctx.aux_nbas]];
            let (ip1_b, _): (Vec<f64>, Vec<usize>) = ctx.cint_all
                .integrate_row_major("int3c2e_ip1", "s1", Some(ip1_slc))
                .into();
            // ip1_b layout: [3, qj, N, P] row-major (y, k, l, p)
            let mut b = vec![0.0; m * nao];
            for p in 0..naux {
                for y in 0..3 {
                    for k in 0..qj {
                        let row = (p * 3 + y) * qj + k;
                        for l in 0..nao {
                            b[row + l * m] =
                                ip1_b[y * qj * nao * naux + k * nao * naux + l * naux + p];
                        }
                    }
                }
            }
            drop(ip1_b);
            let b_t = rt::asarray((&b, [m, nao].f(), &device));
            let z2u = &b_t % &mc2_t_full.t(); // [m, O]
            // stage Z2u as [P, 3·qj·O] F-order: row p, col (y,k,a)
            let z2u_raw = z2u.into_shape(-1).into_raw();
            let mut z2u_p = vec![0.0; naux * 3 * qj * nocc];
            for p in 0..naux {
                for y in 0..3 {
                    for k in 0..qj {
                        for a in 0..nocc {
                            z2u_p[p + ((y * qj + k) * nocc + a) * naux] =
                                z2u_raw[(p * 3 + y) * qj + k + a * m];
                        }
                    }
                }
            }
            let z2u_p_t = rt::asarray((&z2u_p, [naux, 3 * qj * nocc].f(), &device));
            let z2 = &i2inv_t % &z2u_p_t; // [P, 3·qj·O] row p, col (y,k,a)
            // scatter back to [P·3·qj, O] F-order layout (as consumed by g4)
            let z2_raw = z2.into_shape(-1).into_raw();
            let mut z2_out = vec![0.0; m * nocc];
            for p in 0..naux {
                for y in 0..3 {
                    for k in 0..qj {
                        for a in 0..nocc {
                            z2_out[(p * 3 + y) * qj + k + a * m] =
                                z2_raw[p + ((y * qj + k) * nocc + a) * naux];
                        }
                    }
                }
            }
            z2_per_atom[j0] = z2_out;
            }
            }
            // ── main i0 loop (G3 + G4; G4's j0 filtered to current z2 batch) ──
        for i0 in 0..natm {
            let (_, p0, ni) = blk[i0];
            if ni == 0 { continue; }
            if std::env::var("REST_MEM_TRACE").is_ok() && i0 % 3 == 0 {
                eprintln!("MEMTRACE g34-iter-{:02}        RSS = {:.1} MiB", i0, memory_monitor::current_rss_mb());
            }

            // ── Evaluate ipv block for atom i0: [9, ni, nao, naux] row-major ──
            // element (c, ii, j, p) at c*ni*nao*naux + ii*nao*naux + j*naux + p
            // ── W2 [3·ni·P, O] F-order: row (x,i,p), col a
            //    W2[x,i,p,a] = Σ_j ip1[x,p0+i,j,p]·mc2[j,a]  (1 GEMM per i0) ──
            // ip1[i0-row-block] integrated once per atom (234 MB), dropped after.
            let shl0w = ctx.aoslices[i0][0] as usize;
            let shl1w = ctx.aoslices[i0][1] as usize;
            let ip1_slc: &[[usize; 2]] = &[[shl0w, shl1w], [0, ctx.nreg],
                [ctx.nreg, ctx.aux_nbas]];
            let (ip1_b, _): (Vec<f64>, Vec<usize>) = ctx.cint_all
                .integrate_row_major("int3c2e_ip1", "s1", Some(ip1_slc))
                .into();
            let m_w2 = 3 * ni * naux;
            let w2: Vec<f64>;
            {
                // Cache-friendly staging: j outer, p inner → both `a` (row-major
                // [m_w2, N] cols contiguous) and ip1 (p fastest) write/read with
                // stride 1.
                let mut a = vec![0.0; m_w2 * nao];
                for x in 0..3 { for i in 0..ni { for j in 0..nao {
                    let ba = (x * ni + i) * naux + j * m_w2;
                    let bi = x * ni * nao * naux + i * nao * naux + j * naux;
                    for p in 0..naux { a[ba + p] = ip1_b[bi + p]; }
                }}}
                let a_t = rt::asarray((&a, [m_w2, nao].f(), &device));
                let w2_t = &a_t % &mc2_t_full.t(); // [m_w2, O]
                w2 = w2_t.into_shape(-1).into_raw();
            }
            drop(ip1_b);
            // ── Ua [P·ni, O] F-order: row (p,i), col a
            //    Ua[p,i,a] = Σ_o mc2[p0+i,o]·Q[p,o,a]  (via Ua^T = mc2_blk @ Q_T) ──
            let mut ua = vec![0.0; naux * ni * nocc];
            {
                let q_t = rt::asarray((&q_t_raw, [nocc, naux * nocc].f(), &device));
                // mc2_blk [ni, O] F-order: row i, col o
                let mut mc2_blk = vec![0.0; ni * nocc];
                for i in 0..ni { for o in 0..nocc {
                    mc2_blk[i + o * ni] = mc2[(p0 + i) * nocc + o];
                }}
                let mc2_blk_t = rt::asarray((&mc2_blk, [ni, nocc].f(), &device));
                let ua_t = &mc2_blk_t % &q_t; // [ni, P·O] row i, col (p,a)
                let ua_t_raw = ua_t.into_shape(-1).into_raw();
                for p in 0..naux { for i in 0..ni { for a in 0..nocc {
                    ua[p * ni + i + a * (naux * ni)] = ua_t_raw[i + (p * nocc + a) * ni];
                }}}
            }
            // ── Per-i0 GEMM inputs (reused across j0) ──
            // W2_A [m_w, ni] F-order: row (x,p,b), col i ; W2_A[(x,p,b), i] = W2[x,i,p,b]
            // W [m_w, O] = W2_A @ mc2_blk : W[(x,p,b), a] = Σ_i W2[x,i,p,b]·mc2[i,a]
            // W_A [3, m_ao] F-order: row x, col m=(p,a,b) ; W_A[x, m] = W[x,p,b,a]
            let m_w = 3 * naux * nocc;
            let m_ao = naux * nocc * nocc;
            let w_raw: Vec<f64>;
            {
                let mut w2_a = vec![0.0; m_w * ni];
                for x in 0..3 { for p in 0..naux { for b in 0..nocc {
                    let row = (x * naux + p) * nocc + b;
                    for i in 0..ni {
                        w2_a[row + i * m_w] = w2[(x * ni + i) * naux + p + b * m_w2];
                    }
                }}}
                let w2_a_t = rt::asarray((&w2_a, [m_w, ni].f(), &device));
                let mut mc2_blk = vec![0.0; ni * nocc];
                for i in 0..ni { for o in 0..nocc {
                    mc2_blk[i + o * ni] = mc2[(p0 + i) * nocc + o];
                }}
                let mc2_blk_t = rt::asarray((&mc2_blk, [ni, nocc].f(), &device));
                let w = &w2_a_t % &mc2_blk_t; // [m_w, O]
                w_raw = w.into_shape(-1).into_raw();
            }
            let mut w_a = vec![0.0; 3 * m_ao];
            for x in 0..3 { for p in 0..naux { for a in 0..nocc { for b in 0..nocc {
                w_a[x + ((p * nocc + a) * nocc + b) * 3] =
                    w_raw[(x * naux + p) * nocc + b + a * m_w];
            }}}}
            // W2_H [3·ni, P·O] F-order: row (x,i), col (p,a)
            let m_h = naux * nocc;
            let w2_h: Vec<f64>;
            {
                let mut w2_h_buf = vec![0.0; 3 * ni * m_h];
                for x in 0..3 { for i in 0..ni {
                    let row = x * ni + i;
                    for p in 0..naux { for a in 0..nocc {
                        w2_h_buf[row + (p * nocc + a) * (3 * ni)] =
                            w2[(x * ni + i) * naux + p + a * m_w2];
                    }}
                }}
                w2_h = w2_h_buf;
            }

            if i0 == 0 {
            }

            // ── v2_all[j0] = mc2_blk2[j0]·Uaᵀ ([qj, P·ni] per j0≤i0, 190 MB)
            //    precomputed so ipv can stream per aux block. ──
            let mut v2_all: Vec<Vec<f64>> = Vec::with_capacity(i0 + 1);
            for j0 in 0..=i0 {
                let (_, q0, qj) = blk[j0];
                if qj == 0 { v2_all.push(Vec::new()); continue; }
                let mut mc2_blk2 = vec![0.0; qj * nocc];
                for k in 0..qj { for o in 0..nocc {
                    mc2_blk2[k + o * qj] = mc2[(q0 + k) * nocc + o];
                }}
                let mc2_blk2_t = rt::asarray((&mc2_blk2, [qj, nocc].f(), &device));
                let ua_t = rt::asarray((&ua, [naux * ni, nocc].f(), &device));
                let v2_t = &mc2_blk2_t % &ua_t.t(); // [qj, P·ni]
                v2_all.push(v2_t.into_shape(-1).into_raw());
            }

            // ── ipv streamed per aux block ([9, ni, N, qi], 39 MB each):
            //    vj1_mat += Σ_p ipv·r0 and out_p2_all[j0] += Σ_p ipv·v2 —
            //    the 700 MB full ipv block never exists. ──
            let shl0 = ctx.aoslices[i0][0] as usize;
            let shl1 = ctx.aoslices[i0][1] as usize;
            let mut vj1_mat = vec![0.0; 9 * ni * nao];
            let mut out_p2_all = vec![0.0; (i0 + 1) * 9];
            for aux_qb in 0..natm {
                let aux_shl0 = ctx.auxslices[aux_qb][0] as usize;
                let aux_shl1 = ctx.auxslices[aux_qb][1] as usize;
                let ipv_slc: &[[usize; 2]] = &[[shl0, shl1], [0, ctx.nreg],
                    [ctx.nreg + aux_shl0, ctx.nreg + aux_shl1]];
                let (ipv_qb, _): (Vec<f64>, Vec<usize>) = ctx.cint_all
                    .integrate_row_major("int3c2e_ipvip1", "s1", Some(ipv_slc))
                    .into();
                let qi = ctx.auxslices[aux_qb][3] as usize - ctx.auxslices[aux_qb][2] as usize;
                let qbp = ctx.auxslices[aux_qb][2] as usize;
                // vj1_mat[c, ii, j] += Σ_{p∈qb} ipv_qb[c,ii,j,p]·r0[p]
                for c in 0..9 {
                    for ii in 0..ni {
                        for j in 0..nao {
                            let base = c * ni * nao * qi + ii * nao * qi + j * qi;
                            let row = &ipv_qb[base..base + qi];
                            let mut s0 = 0.0f64; let mut s1 = 0.0f64;
                            let mut s2 = 0.0f64; let mut s3 = 0.0f64;
                            let mut p = 0usize;
                            while p + 4 <= qi {
                                s0 += row[p] * r0[qbp + p];
                                s1 += row[p + 1] * r0[qbp + p + 1];
                                s2 += row[p + 2] * r0[qbp + p + 2];
                                s3 += row[p + 3] * r0[qbp + p + 3];
                                p += 4;
                            }
                            while p < qi { s0 += row[p] * r0[qbp + p]; p += 1; }
                            vj1_mat[c * ni * nao + ii * nao + j] += s0 + s1 + s2 + s3;
                        }
                    }
                }
                // out_p2_all[j0][c] += Σ_{p∈qb, i, j} ipv_qb[c,i,q0+j,p]·v2[j,(p,i)]
                for j0 in 0..=i0 {
                    let v2 = &v2_all[j0];
                    if v2.is_empty() { continue; }
                    let (_, q0, qj) = blk[j0];
                    let op_base = j0 * 9;
                    for c in 0..9 {
                        let mut s = 0.0;
                        for j in 0..qj {
                            let vj = j;
                            let ibase = c * ni * nao * qi + (q0 + j) * qi;
                            for pb in (0..qi).step_by(64) {
                                let pe = (pb + 64).min(qi);
                                for ib in (0..ni).step_by(8) {
                                    let ie = (ib + 8).min(ni);
                                    for p in pb..pe { for i in ib..ie {
                                        s += ipv_qb[ibase + i * nao * qi + p]
                                            * v2[vj + ((qbp + p) * ni + i) * qj];
                                    }}
                                }
                            }
                        }
                        out_p2_all[op_base + c] += s;
                    }
                }
                drop(ipv_qb);
            }

            // ── G3: ej_vj1[i0, j0, c] = Σ_{ii∈i0, j∈j0} vj1_mat[c,ii,j]·dm0[j,ii]·2 ──
            for j0 in 0..=i0 {
                let (_, q0, qj) = blk[j0];
                if qj == 0 { continue; }
                for c in 0..9 {
                    let mut s = 0.0;
                    for ii in 0..ni { for jj in 0..qj {
                        s += vj1_mat[c * ni * nao + ii * nao + (q0 + jj)]
                            * dm0[(q0 + jj) + (p0 + ii) * nao];
                    }}
                    let (x1, x2) = (c / 3, c % 3);
                    out_vj1[i_t(i0, j0, x1, x2)] = s * 2.0;
                }
            }

            if !do_k { continue; }

            // ══════════ G4 plan B ══════════
            for j0 in 0..=i0 {
                if z2_per_atom[j0].is_empty() { continue; }
                let (_, q0, qj) = blk[j0];
                if qj == 0 { continue; }
                let z2 = &z2_per_atom[j0]; // [P·3·qj, O] F-order: row (p,y,k), col a
                let m_z2 = naux * 3 * qj;

                // ── mc2_blk2 [qj, O] F-order: row k, col o ──
                let mut mc2_blk2 = vec![0.0; qj * nocc];
                for k in 0..qj { for o in 0..nocc {
                    mc2_blk2[k + o * qj] = mc2[(q0 + k) * nocc + o];
                }}
                let mc2_blk2_t = rt::asarray((&mc2_blk2, [qj, nocc].f(), &device));

                // ── A 项 ──
                let out_a: Vec<f64>;
                {
                    // Z2_A [m_z, qj] F-order: row (p,y,b), col k ; Z2_A[(p,y,b), k] = Z2[p,y,k,b]
                    // Z [m_z, O] = Z2_A @ mc2_blk2 : Z[(p,y,b), a] = Σ_k Z2[p,y,k,a]·mc2[k,b]
                    // Z_A [m_ao, 3] F-order: row m=(p,b,a), col y ; Z_A[m, y] = Z[p,y,b,a]
                    let m_z = m_w;
                    let mut z2_a = vec![0.0; m_z * qj];
                    for p in 0..naux { for y in 0..3 { for b in 0..nocc {
                        let row = (p * 3 + y) * nocc + b;
                        for k in 0..qj {
                            z2_a[row + k * m_z] = z2[(p * 3 + y) * qj + k + b * m_z2];
                        }
                    }}}
                    let z2_a_t = rt::asarray((&z2_a, [m_z, qj].f(), &device));
                    let z = &z2_a_t % &mc2_blk2_t; // [m_z, O]
                    let z_raw = z.into_shape(-1).into_raw();
                    let mut z_a = vec![0.0; m_ao * 3];
                    // Blocked transpose: y outer, (a,b) 8×8 blocks — z_raw read
                    // b-contiguous (full cache line) and z_a write a-contiguous
                    // (full cache line). The old y-inner order wrote z_a at a
                    // 2.5 MB column stride (8× write amplification, ~1.5 s).
                    for y in 0..3 {
                        for p in 0..naux {
                            for ab in (0..nocc).step_by(8) {
                                let ae = (ab + 8).min(nocc);
                                for bb in (0..nocc).step_by(8) {
                                    let be = (bb + 8).min(nocc);
                                    for a in ab..ae {
                                        let rbase = (p * 3 + y) * nocc + a * m_z;
                                        for b in bb..be {
                                            z_a[((p * nocc + b) * nocc + a) + y * m_ao] =
                                                z_raw[rbase + b];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let w_a_t = rt::asarray((&w_a, [3, m_ao].f(), &device));
                    let z_a_t = rt::asarray((&z_a, [m_ao, 3].f(), &device));
                    let oa = &w_a_t % &z_a_t; // [3, 3]
                    out_a = oa.into_shape(-1).into_raw();
                }

                // ── B 项 ──
                // Z2_H [P·O, 3·qj] F-order: row (p,a), col (y,k)
                let out_b: [f64; 9];
                {
                    let mut z2_h = vec![0.0; m_h * 3 * qj];
                    // Blocked staging: (a,k) 8×16 blocks — z2 read k-contiguous
                    // and z2_h write a-contiguous (both full cache lines), vs the
                    // old k-inner order writing z2_h at a 2.5 MB column stride.
                    for p in 0..naux { for y in 0..3 {
                        for ab in (0..nocc).step_by(8) {
                            let ae = (ab + 8).min(nocc);
                            for kb in (0..qj).step_by(16) {
                                let ke = (kb + 16).min(qj);
                                for a in ab..ae {
                                    let rbase = (p * 3 + y) * qj + a * m_z2;
                                    for k in kb..ke {
                                        z2_h[(p * nocc + a) + (y * qj + k) * m_h] =
                                            z2[rbase + k];
                                    }
                                }
                            }
                        }
                    }}
                    let w2_h_t = rt::asarray((&w2_h, [3 * ni, m_h].f(), &device));
                    let z2_h_t = rt::asarray((&z2_h, [m_h, 3 * qj].f(), &device));
                    let h = &w2_h_t % &z2_h_t; // [3·ni, 3·qj] row (x,i), col (y,k)
                    let h_raw = h.into_shape(-1).into_raw();
                    let mut ob = [0.0f64; 9];
                    for x in 0..3 { for y in 0..3 {
                        let mut s = 0.0;
                        for i in 0..ni { for k in 0..qj {
                            s += h_raw[x * ni + i + (y * qj + k) * (3 * ni)]
                                * dm0[(q0 + k) * nao + (p0 + i)];
                        }}
                        ob[x * 3 + y] = s;
                    }}
                    out_b = ob;
                }

                // ── part2 (ipv streamed per aux block above): out_p2_all[j0] ──
                for x in 0..3 { for y in 0..3 {
                    let c = x * 3 + y;
                    out_vk1[i_t(i0, j0, x, y)] = out_a[x + y * 3] + out_b[c] + out_p2_all[j0 * 9 + c];
                }}
            }
        }

            for j0 in j0_lo..j0_hi {
                z2_per_atom[j0] = Vec::new();
            }
        }
}

/// g5 (ek_ri1) + g8 (ej_ri1) combined, per-AO-atom streaming implementation.
///
/// The 9·N²·P derivative integral `int3c2e_ip1ip2` is never materialized in
/// full: it is evaluated per-AO-atom block (9·ni·N·P) via a libcint shell
/// slice inside the i0 loop and consumed immediately by both g5 and g8, so
/// the full tensor never coexists with tmpf/wki/rhok. `do_k` gates the
/// exchange part (g5): pure LDA/GGA DFAs (factor_k == 0) skip it.
///
/// Also computes wk1_IpJ (the wki·dm0 contraction) per-atom block instead of
/// materializing the full 3·N²·P intermediate.
fn g5_g8_ri1_blas(ctx: &EjEkContext, out_ek: &mut [f64], out_ej: &mut [f64], do_k: bool) {
    let nao = ctx.nao; let naux = ctx.naux; let nocc = ctx.nocc;
    let natm = ctx.natm; let nao3 = ctx.nao3;
    let blk = ctx.blk; let aux_blk = ctx.aux_blk;
    let dm0 = ctx.dm0; let mc2 = ctx.mc2;
    let i21 = ctx.i21; let rk = ctx.rk;
    let i2inv = ctx.i2inv;
    let r0 = ctx.r0; let rj1 = ctx.rj1; let wj001 = ctx.wj001; let wj2 = ctx.wj2;
    let device = ctx.device.clone();
    let i2inv_t = rt::asarray((i2inv, [naux, naux].f(), &device));
    let i_t = |i0: usize, j0: usize, x: usize, y: usize| -> usize {
        i0 * natm * 9 + j0 * 9 + x * 3 + y
    };


    // mc2_t: F-order [nocc, nao], element (i_occ, J) = mc2[J*nocc + i_occ]
    let mc2_t = rt::asarray((mc2, [nocc, nao].f(), &device));
    let dm0_t = rt::asarray((dm0, [nao, nao].f(), &device));

    // i21 staged for batched wk1_pJI: [3*naux, naux] F-order
    let mut i21_batch = vec![0.0; 3 * naux * naux];
    for y in 0..3 { for p in 0..naux { for q in 0..naux {
        i21_batch[(y * naux + p) + q * (3 * naux)] = i21[y * naux * naux + p * naux + q];
    }}}
    let i21_batch_t = rt::asarray((&i21_batch, [3 * naux, naux].f(), &device));

    // L[y,q,q'] = Σ_pg V⁻¹[pg,q]·i21[y, q', pg]  ([3,P,P] row-major) — the
    // V⁻¹ weight absorbed on the i21 side for t3 (whose aux sum runs over
    // the rk_PJI row block j0, unlike t2's tmpf row block j0). NOTE: i21 is
    // NOT symmetric in (p,q) — int2c2e_ip1 is antisymmetric (i21[y,p,q] =
    // -i21[y,q,p]) — so the index order here (q' first, pg second) must match
    // t3's contraction i21[y, qg, paux].
    let mut L_vinv = vec![0.0; 3 * naux * naux];
    for y in 0..3 { for q in 0..naux { for qp in 0..naux {
        let mut s = 0.0;
        for pg in 0..naux {
            s += i2inv[pg + q * naux] * i21[y * naux * naux + qp * naux + pg];
        }
        L_vinv[(y * naux + q) * naux + qp] = s;
    }}}

    for i0 in 0..natm { let (_, p0, ni) = blk[i0];
        if ni == 0 { continue; }

        // ── ip12 block [9, ni, nao, naux] row-major, element (x, ii, j, p)
        //    at x*ni*nao*naux + ii*nao*naux + j*naux + p ──
        let shl0 = ctx.aoslices[i0][0] as usize;
        let shl1 = ctx.aoslices[i0][1] as usize;
        let ip12_slc: &[[usize; 2]] = &[[shl0, shl1], [0, ctx.nreg], [ctx.nreg, ctx.aux_nbas]];
        let (ip12_i0, _): (Vec<f64>, Vec<usize>) = ctx.cint_all
            .integrate_row_major("int3c2e_ip1ip2", "s1", Some(ip12_slc))
            .into();

        // (do_k Loop A/B moved below the main i0 loop — they re-traverse
        //  all atoms once; keeping them inside this per-i0 loop would
        //  multiply their out_ek contributions by natm)

        // ── g8 (ej_ri1): w11 from the ip12 block, then t1-t4 for-loops ──
        // w11[x,p] = Σ_{ii,j} ip12[x,ii,j,p]·dm0[(p0+ii),j] — explicit SIMD
        // loop (p inner): ip12 read p-contiguous, w11 accumulate contiguous.
        // The old p-outer/j-inner order read ip12 at 7 KB jumps (8×
        // amplification over 234 MB/atom).
        let mut w11 = vec![0.0; 9 * naux];
        for x in 0..9 { for ii in 0..ni { for j in 0..nao {
            let dmv = dm0[(p0 + ii) * nao + j];
            let base = x * ni * nao * naux + ii * nao * naux + j * naux;
            let row = &ip12_i0[base..base + naux];
            let wbase = x * naux;
            let mut p = 0usize;
            while p + 4 <= naux {
                w11[wbase + p] += row[p] * dmv;
                w11[wbase + p + 1] += row[p + 1] * dmv;
                w11[wbase + p + 2] += row[p + 2] * dmv;
                w11[wbase + p + 3] += row[p + 3] * dmv;
                p += 4;
            }
            while p < naux { w11[wbase + p] += row[p] * dmv; p += 1; }
        }}}
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
                out_ej[i_t(i0, j0, x, y)] += v;
                out_ej[i_t(j0, i0, x, y)] += (t1[y * 3 + x] - t2[y * 3 + x] - t3[y * 3 + x] + t4[y * 3 + x]) * 2.0;
            }}
        }
    }

        if do_k {
            use std::time::Instant;
            let _t_g5a = Instant::now();
            let mut _t_g5a_rho2c = 0.0f64;
            let mut _t_g5a_t1t3 = 0.0f64;
            let mut _t_g5b_tmpf = 0.0f64;
            let mut _t_g5b_rkpji = 0.0f64;
            let mut _t_g5b_wk = 0.0f64;
            let mut _t_g5b_t2t4 = 0.0f64;
            // ══════════════════════════════════════════════════════════
            // Loop A (i0 outer): t1 + t3.
            //   ip12[i0-block] integrated once; rk_PJI[i0-block] built once.
            //   t3's rho2c_PQ is rebuilt from ip1 (V⁻¹ absorbed on the i21
            //   side: R = ip1_xᵀ·rk_PJI, rho2c = V⁻¹·R) — no tmpf, no
            //   per-(i0,j0) re-expansion.
            // Loop B (j0 outer): t2 + t4.
            //   tmpf[j0-block] = V⁻¹[j0-block,:]·ip1ᵀ built once per aux atom
            if std::env::var("REST_MEM_TRACE").is_ok() {
                eprintln!("MEMTRACE g5-LB-iter         RSS = {:.1} MiB", memory_monitor::current_rss_mb());
            }
            //   (full-K GEMM); rk_PJI[i0-block] rebuilt per (j0,i0) (~2 s);
            //   wk1_pJI[j0-block] = i21[pg∈j0]·rk_PJI and
            //   wk1_IpJ[j0-block] = wki[·,pg∈j0,·,·]·dm0 both full-K GEMMs.
            // ══════════════════════════════════════════════════════════
            // ── Loop A: t1 + t3 ──
            if std::env::var("REST_MEM_TRACE").is_ok() {
                eprintln!("MEMTRACE g5-loopA-start      RSS = {:.1} MiB", memory_monitor::current_rss_mb());
            }
            for i0 in 0..natm {
                let (_, p0, ni) = blk[i0];
                if ni == 0 { continue; }
                let shl0 = ctx.aoslices[i0][0] as usize;
                let shl1 = ctx.aoslices[i0][1] as usize;
                let ip12_slc: &[[usize; 2]] =
                    &[[shl0, shl1], [0, ctx.nreg], [ctx.nreg, ctx.aux_nbas]];
                if std::env::var("REST_MEM_TRACE").is_ok() && i0 % 2 == 0 {
                    crate::hessian::memory_monitor::trim_to_os(0);
                    eprintln!("MEMTRACE g5A-ip12-{:02}      RSS = {:.1} MiB", i0, memory_monitor::current_rss_mb());
                }
                let (ip12_i0, _): (Vec<f64>, Vec<usize>) = ctx.cint_all
                    .integrate_row_major("int3c2e_ip1ip2", "s1", Some(ip12_slc))
                    .into();
                // ── rk_P_I (1 GEMM) ──
                let mut rk_P_I = vec![0.0; naux * nocc * ni];
                {
                    let m_rk = naux * nocc;
                    let mut rk_stage = vec![0.0; m_rk * nao];
                    for l in 0..nao { for j_occ in 0..nocc {
                        let rbase = l * naux + j_occ * naux * nao;
                        for p in 0..naux {
                            rk_stage[(p * nocc + j_occ) + l * m_rk] = rk[p + rbase];
                        }
                    }}
                    let rk_2d_g5 = rt::asarray((&rk_stage, [m_rk, nao].f(), &device));
                    let dm0_block_t = dm0_t.i((.., p0..p0 + ni));
                    let rk_P_I_2d = (&rk_2d_g5 % &dm0_block_t);
                    let rk_P_I_raw = rk_P_I_2d.into_shape(-1).into_raw();
                    for p in 0..naux { for j_occ in 0..nocc { for ii in 0..ni {
                        rk_P_I[p * nocc * ni + j_occ * ni + ii] =
                            rk_P_I_raw[(p * nocc + j_occ) + ii * (naux * nocc)];
                    }}}
                }
                // ── rk_PJI (1 GEMM), p-inner [P, N·ni] F-order ──
                let mut rk_PJI = vec![0.0; naux * nao * ni];
                {
                    let mut rk_P_I_stage = vec![0.0; naux * ni * nocc];
                    for j_occ in 0..nocc { for p in 0..naux {
                        let rbase = p * nocc * ni + j_occ * ni;
                        for ii in 0..ni {
                            rk_P_I_stage[(p * ni + ii) + j_occ * (naux * ni)] =
                                rk_P_I[rbase + ii];
                        }
                    }}
                    let rk_P_I_t = rt::asarray((&rk_P_I_stage, [naux * ni, nocc].f(), &device));
                    let rk_PJI_2d = (&rk_P_I_t % &mc2_t);
                    let rk_PJI_raw = rk_PJI_2d.into_shape(-1).into_raw();
                    for j in 0..nao { for ii in 0..ni {
                        let rbase = j * (naux * ni) + ii;
                        for p in 0..naux {
                            rk_PJI[p + (j * ni + ii) * naux] = rk_PJI_raw[rbase + p * ni];
                        }
                    }}
                }
                // ── rho2c_PQ[x,q,p] rebuilt from ip1 (no tmpf, no ip1 full):
                //    R[x,p,q] = Σ_{ii,j} ip1[x,p0+ii,j,p]·rk_PJI[q,j,ii]  (GEMM
                //    [P, N·ni]@[N·ni, P] per x), rho2c = V⁻¹·R. ip1[i0-row-block]
                //    integrated once per atom (234 MB), dropped after use. ──
                let shl0a = ctx.aoslices[i0][0] as usize;
                let shl1a = ctx.aoslices[i0][1] as usize;
                let ip1_slc: &[[usize; 2]] = &[[shl0a, shl1a], [0, ctx.nreg],
                    [ctx.nreg, ctx.aux_nbas]];
                let (ip1_b, _): (Vec<f64>, Vec<usize>) = ctx.cint_all
                    .integrate_row_major("int3c2e_ip1", "s1", Some(ip1_slc))
                    .into();
                let _t_r = Instant::now();
                let mut rho2c = vec![0.0; 3 * naux * naux];
                {
                    let rk_pji_t_r = rt::asarray((&rk_PJI, [naux, nao * ni].f(), &device));
                    let mut ip1_x_blk = vec![0.0; naux * nao * ni];
                    for x in 0..3 {
                        for ii in 0..ni { for j in 0..nao {
                            let ibase = x * ni * nao * naux + ii * nao * naux + j * naux;
                            // column c = j·ni + ii must match rk_PJI's
                            // [P, N·ni] column order (j outer, ii inner)
                            let cbase = (j * ni + ii) * naux;
                            for p in 0..naux { ip1_x_blk[p + cbase] = ip1_b[ibase + p]; }
                        }}
                        let ip1_x_t = rt::asarray((&ip1_x_blk, [naux, nao * ni].f(), &device));
                        let r_x = &ip1_x_t % &rk_pji_t_r.t(); // [P, P] row p, col q
                        let r_x_raw = r_x.into_shape(-1).into_raw();
                        let r_x_t = rt::asarray((&r_x_raw, [naux, naux].f(), &device));
                        let rho_x = &i2inv_t % &r_x_t; // [P, P]
                        let rho_x_raw = rho_x.into_shape(-1).into_raw();
                        for q in 0..naux { for p in 0..naux {
                            rho2c[x * naux * naux + q * naux + p] = rho_x_raw[p + q * naux];
                        }}
                    }
                }
                drop(ip1_b);
                _t_g5a_rho2c += _t_r.elapsed().as_secs_f64();
                if std::env::var("REST_MEM_TRACE").is_ok() && i0 % 2 == 0 {
                    eprintln!("MEMTRACE g5A-rho2c-{:02}    RSS = {:.1} MiB", i0, memory_monitor::current_rss_mb());
                }
                let _t_a2 = Instant::now();
                for j0 in 0..natm { let (_, aq0, ql) = aux_blk[j0]; if ql == 0 { continue; }
                    // ── t1 ──
                    let mut t1 = [0.0f64; 9];
                    for x in 0..9 { let mut s = 0.0; for ii in 0..ni { for j in 0..nao { for qp in 0..ql {
                        let pg = aq0 + qp;
                        s += ip12_i0[x * ni * nao * naux + ii * nao * naux + j * naux + pg]
                            * rk_PJI[pg + (j * ni + ii) * naux];
                    }}} t1[x] = s; }
                    // ── t3 (rho2c·i21, i21 NOT symmetric in (p,q)) ──
                    let mut t3 = [0.0f64; 9];
                    for x in 0..3 { for y in 0..3 { let mut s = 0.0;
                        for qg in 0..ql { for paux in 0..naux {
                            s += rho2c[x * naux * naux + (aq0 + qg) * naux + paux]
                               * i21[y * naux * naux + (aq0 + qg) * naux + paux];
                        }}
                        t3[x * 3 + y] = s;
                    }}


                    for x in 0..3 { for y in 0..3 {
                        out_ek[i_t(i0,j0,x,y)] += t1[x*3+y] - t3[x*3+y];
                        out_ek[i_t(j0,i0,x,y)] += t1[y*3+x] - t3[y*3+x];
                    }}
                }
                if std::env::var("REST_MEM_TRACE").is_ok() && i0 % 2 == 0 {
                    eprintln!("MEMTRACE g5A-t1-{:02}        RSS = {:.1} MiB", i0, memory_monitor::current_rss_mb());
                }
                _t_g5a_t1t3 += _t_a2.elapsed().as_secs_f64();
                drop(ip12_i0);
            }
            eprintln!("G5T LoopA total {:.2}s  rho2c {:.2}s  t1t3 {:.2}s",
                _t_g5a.elapsed().as_secs_f64(), _t_g5a_rho2c, _t_g5a_t1t3);
            // ── Loop B: t2 + t4 ──
            if std::env::var("REST_MEM_TRACE").is_ok() {
                eprintln!("MEMTRACE g5-loopB-start      RSS = {:.1} MiB", memory_monitor::current_rss_mb());
            }
            // rk_P_I cache [P·ni, O] F-order (row (p,ii), col o) built once per
            // i0 (i0-dependent only — was rebuilt 324× inside the (j0,i0)
            // loops with an 86 MB rk_stage each): 133 MB resident.
            let mut rk_p_i_cache: Vec<Vec<f64>> = Vec::with_capacity(natm);
            for i0c in 0..natm {
                let (_, p0c, nic) = blk[i0c];
                if nic == 0 { rk_p_i_cache.push(Vec::new()); continue; }
                let m_rk = naux * nocc;
                let mut rk_stage = vec![0.0; m_rk * nao];
                for l in 0..nao { for j_occ in 0..nocc {
                    let rbase = l * naux + j_occ * naux * nao;
                    for p in 0..naux {
                        rk_stage[(p * nocc + j_occ) + l * m_rk] = rk[p + rbase];
                    }
                }}
                let rk_2d_g5 = rt::asarray((&rk_stage, [m_rk, nao].f(), &device));
                let dm0_block_t = dm0_t.i((.., p0c..p0c + nic));
                let rk_P_I_2d = (&rk_2d_g5 % &dm0_block_t); // [P·O, ni]
                let rk_P_I_raw = rk_P_I_2d.into_shape(-1).into_raw();
                // stage to [P, ni·O] F-order: element (p, (ii,o)) at p + (ii·O+o)·P
                let mut rk_p_i_c = vec![0.0; naux * nic * nocc];
                for o in 0..nocc { for p in 0..naux { for ii in 0..nic {
                    rk_p_i_c[p + (ii * nocc + o) * naux] =
                        rk_P_I_raw[(p * nocc + o) + ii * (naux * nocc)];
                }}}
                rk_p_i_cache.push(rk_p_i_c);
            }
            for j0 in 0..natm { let (_, aq0, ql) = aux_blk[j0]; if ql == 0 { continue; }
                let _t_b = Instant::now();
                // tmpf[j0-block] = V⁻¹[j0-block,:]·ip1ᵀ : [ql, 3N²] F-order
                // The full 2.7 GB int3c2e_ip1 tensor is NOT built: ip1 is
                // re-integrated per aux-atom block (150 MB) inside a qb loop
                // and accumulated into tmpf_j0. V⁻¹[j0-block, qb-block] GEMMs
                // (K = qbi) per (j0, qb); ~18× the integrate calls but each
                // block is only 150 MB, keeping the peak RSS ≈ ip1[qb] +
                // tmpf_j0 + per-block working set (~0.5 GB).
                let mut _ttmpf_int = 0.0f64;
                let mut _ttmpf_gemm = 0.0f64;
                let mut tmpf_j0 = vec![0.0; ql * 3 * nao3];
                {
                    let mut vinv_j0 = vec![0.0; ql * naux]; // [ql, P] F-order (pl, q)
                    for pl in 0..ql { for q in 0..naux {
                        vinv_j0[pl + q * ql] = i2inv[(aq0 + pl) + q * naux];
                    }}
                    for qb in 0..natm {
                        let (_, qb0, qbi) = aux_blk[qb];
                        if qbi == 0 { continue; }
                        if std::env::var("REST_MEM_TRACE").is_ok() && j0 == 0 && qb % 3 == 0 {
                            eprintln!("MEMTRACE tmpf-j0:{}-qb:{}   RSS = {:.1} MiB", j0, qb, memory_monitor::current_rss_mb());
                        }
                        let aux_shl0 = ctx.auxslices[qb][0] as usize;
                        let aux_shl1 = ctx.auxslices[qb][1] as usize;
                        let ip1_qb_slc: &[[usize; 2]] = &[[0, ctx.nreg], [0, ctx.nreg],
                            [ctx.nreg + aux_shl0, ctx.nreg + aux_shl1]];
                        let _ti = Instant::now();
                        let (ip1_qb, _): (Vec<f64>, Vec<usize>) = ctx.cint_all
                            .integrate_row_major("int3c2e_ip1", "s1", Some(ip1_qb_slc))
                            .into();
                        _ttmpf_int += _ti.elapsed().as_secs_f64();
                        let _tg = Instant::now();
                        // ip1_qb row-major [3, N, N, qbi] IS F-order [qbi, 3N²]
                        // (row q, col (x,i,j)) — zero-copy view.
                        let ip1_qb_f = rt::asarray((&ip1_qb, [qbi, 3 * nao3].f(), &device));
                        let mut vinv_qb = vec![0.0; ql * qbi]; // [ql, qbi] F-order
                        for pl in 0..ql { for ql2 in 0..qbi {
                            vinv_qb[pl + ql2 * ql] = vinv_j0[pl + (qb0 + ql2) * ql];
                        }}
                        let vinv_qb_t = rt::asarray((&vinv_qb, [ql, qbi].f(), &device));
                        let tf_part = &vinv_qb_t % &ip1_qb_f; // [ql, 3N²]
                        let tf_raw = tf_part.into_shape(-1).into_raw();
                        for c in 0..(3 * nao3) {
                            for pl in 0..ql {
                                tmpf_j0[pl + c * ql] += tf_raw[pl + c * ql];
                            }
                        }
                        _ttmpf_gemm += _tg.elapsed().as_secs_f64();
                        drop(ip1_qb);
                    }
                }
                eprintln!("G5T tmpf-j{} int {:.2}s gemm+copy {:.2}s", j0, _ttmpf_int, _ttmpf_gemm);
                if std::env::var("REST_MEM_TRACE").is_ok() && j0 % 3 == 0 {
                    eprintln!("MEMTRACE g5B-tmpf-{:02}     RSS = {:.1} MiB", j0, memory_monitor::current_rss_mb());
                }
                _t_g5b_tmpf += _t_b.elapsed().as_secs_f64();
                let _t_wk = Instant::now();
                // i21[j0-block] : [3ql, P] F-order (y,pl) row, q col
                let mut i21_j0 = vec![0.0; 3 * ql * naux];
                for y in 0..3 { for pl in 0..ql { for q in 0..naux {
                    i21_j0[(y * ql + pl) + q * (3 * ql)] =
                        i21[y * naux * naux + (aq0 + pl) * naux + q];
                }}}
                let i21_j0_t = rt::asarray((&i21_j0, [3 * ql, naux].f(), &device));
                // ip2[pg∈j0] block: [3, N, N, ql] row-major (y, j', u, pl).
                // wk1_IpJ[ii,pl,y,j] = (dm0·ip2_y_pl·dm0)[p0+ii, j] — built on
                // the fly (two batched GEMMs per y), no wki full tensor.
                let aux_shl0 = ctx.auxslices[j0][0] as usize;
                let aux_shl1 = ctx.auxslices[j0][1] as usize;
                let ip2_slc: &[[usize; 2]] =
                    &[[0, ctx.nreg], [0, ctx.nreg], [ctx.nreg + aux_shl0, ctx.nreg + aux_shl1]];
                let (ip2_j0, _): (Vec<f64>, Vec<usize>) = ctx.cint_all
                    .integrate_row_major("int3c2e_ip2", "s1", Some(ip2_slc))
                    .into();
                // A_all[(pl,j'),j] = Σ_u ip2_y[j',u,pl]·dm0[u,j] per y — j0-level
                // (independent of i0); stored transposed as A_r [N, N·ql]
                // (row j', col (pl,j)) so the per-i0 B = dm0·A is one GEMM.
                let mut a_r_all = vec![0.0; 3 * nao * nao * ql];
                for y in 0..3 {
                    let mut ip2_y_2d = vec![0.0; nao * ql * nao];
                    for jp in 0..nao { for u in 0..nao { for pl in 0..ql {
                        ip2_y_2d[(pl * nao + jp) + u * (nao * ql)] =
                            ip2_j0[y * nao * nao * ql + jp * nao * ql + u * ql + pl];
                    }}}
                    let ip2_y_t = rt::asarray((&ip2_y_2d, [nao * ql, nao].f(), &device));
                    let a_t = &ip2_y_t % &dm0_t; // [N·ql, N] row (pl,j'), col j
                    let a_raw = a_t.into_shape(-1).into_raw();
                    let mut a_r = vec![0.0; nao * nao * ql];
                    for pl in 0..ql { for j in 0..nao { for jp in 0..nao {
                        a_r[jp + (pl * nao + j) * nao] =
                            a_raw[(pl * nao + jp) + j * (nao * ql)];
                    }}}
                    a_r_all[y * nao * nao * ql..(y + 1) * nao * nao * ql]
                        .copy_from_slice(&a_r);
                }
                let mut _twk_rk = 0.0f64;
                let mut _twk_gemm = 0.0f64;
                for i0 in 0..natm {
                    let (_, p0, ni) = blk[i0];
                    if ni == 0 { continue; }
                    let _tr = Instant::now();
                    // wk1_pJI[j0-block, i0] = i21[j0-block]·rk_PJI, rk_PJI = rk_P_I·mc2.
                    // Associative reorder: W1 = i21_j0·rk_P_I_cache[i0] ([3ql, P]@[P, ni·O]),
                    // then wk1_pji = W1·mc2 per-ii GEMM — avoids materializing
                    // rk_PJI [P, N·ni] (78 MB staging × 324) entirely.
                    let rk_p_i_c_t = rt::asarray((&rk_p_i_cache[i0], [naux, ni * nocc].f(), &device));
                    let w1_t = &i21_j0_t % &rk_p_i_c_t; // [3ql, ni·O] row (y,pl), col (ii,o)
                    let w1_raw = w1_t.into_shape(-1).into_raw();
                    // wk1_pji layout [3ql, ni·N] F-order: row (y,pl), col (ii·N+j)
                    let mut wk1_pji_j0 = vec![0.0; 3 * ql * ni * nao];
                    for ii in 0..ni {
                        let mut w1_ii = vec![0.0; 3 * ql * nocc];
                        for y in 0..3 { for pl in 0..ql { for o in 0..nocc {
                            w1_ii[(y * ql + pl) + o * (3 * ql)] =
                                w1_raw[(y * ql + pl) + (ii * nocc + o) * (3 * ql)];
                        }}}
                        let w1_ii_t = rt::asarray((&w1_ii, [3 * ql, nocc].f(), &device));
                        let out_ii_t = &w1_ii_t % &mc2_t; // [3ql, N]
                        let out_ii_raw = out_ii_t.into_shape(-1).into_raw();
                        for y in 0..3 { for pl in 0..ql { for j in 0..nao {
                            wk1_pji_j0[(y * ql + pl) + (ii * nao + j) * (3 * ql)] =
                                out_ii_raw[(y * ql + pl) + j * (3 * ql)];
                        }}}
                    }
                    _twk_rk += _tr.elapsed().as_secs_f64();
                    let _tg2 = Instant::now();
                    // wk1_IpJ[j0-block] = dm0·A per i0: B[ii,(pl,j)] =
                    //   Σ_j' dm0[p0+ii,j']·A_r[j',(pl,j)] — GEMM [ni,N]@[N,N·ql]
                    //   per y (A_r precomputed at j0 level). wk1_IpJ[pl,ii,y,j]=B[ii,(pl,j)].
                    let mut dm0_p0 = vec![0.0; ni * nao]; // [ni, N] F-order (row ii, col j')
                    for ii in 0..ni { for jp in 0..nao {
                        dm0_p0[ii + jp * ni] = dm0[(p0 + ii) * nao + jp];
                    }}
                    let dm0_p0_t = rt::asarray((&dm0_p0, [ni, nao].f(), &device));
                    let mut wk1_ipj_alt = vec![0.0; ql * ni * 3 * nao];
                    for y in 0..3 {
                        let a_r_t = rt::asarray((&a_r_all[y * nao * nao * ql..], [nao, nao * ql].f(), &device));
                        let b_t = &dm0_p0_t % &a_r_t; // [ni, N·ql] row ii, col (pl,j)
                        let b_raw = b_t.into_shape(-1).into_raw();
                        for pl in 0..ql { for ii in 0..ni { for j in 0..nao {
                            wk1_ipj_alt[pl + (ii * 3 * nao + y * nao + j) * ql] =
                                b_raw[ii + (pl * nao + j) * ni];
                        }}}
                    }
                let _t_t24 = Instant::now();
                // t2 / t4 (single pass over tmpf_j0)
                    let mut t2 = [0.0f64; 9];
                    let mut t4 = [0.0f64; 9];
                    for x in 0..3 { for y in 0..3 {
                        let mut s2 = 0.0; let mut s4 = 0.0;
                        for ii in 0..ni { for j in 0..nao {
                            let c = x * nao3 + (p0 + ii) * nao + j;
                            let c_pji = (ii * nao + j) * (3 * ql);
                            let c_ipj = (ii * 3 * nao + y * nao + j) * ql;
                            for pl in 0..ql {
                                let v = tmpf_j0[pl + c * ql];
                                s2 += v * wk1_pji_j0[(y * ql + pl) + c_pji];
                                s4 += v * wk1_ipj_alt[pl + c_ipj];
                            }
                        }}
                        t2[x * 3 + y] = s2;
                        t4[x * 3 + y] = s4;
                    }}
                    for x in 0..3 { for y in 0..3 {
                        out_ek[i_t(i0,j0,x,y)] += -t2[x*3+y] + t4[x*3+y];
                        out_ek[i_t(j0,i0,x,y)] += -t2[y*3+x] + t4[y*3+x];
                    }}
                    let _tt = _t_t24.elapsed().as_secs_f64();
                    _t_g5b_t2t4 += _tt;
                    _twk_gemm += _tg2.elapsed().as_secs_f64();
                    if i0 == 0 {
                        eprintln!("G5T wk-j{} rk {:.2}s gemm {:.2}s", j0, _twk_rk, _twk_gemm);
                    }

                }
                _t_g5b_wk += _t_wk.elapsed().as_secs_f64();
            }
            eprintln!("G5T LoopA total {:.2}s  rho2c {:.2}s  t1t3 {:.2}s | LoopB tmpf {:.2}s  wk {:.2}s  t2t4 {:.2}s",
                _t_g5a.elapsed().as_secs_f64(), _t_g5a_rho2c, _t_g5a_t1t3,
                _t_g5b_tmpf, _t_g5b_wk, _t_g5b_t2t4);

        }
}

/// g6 (ek_ri2d) + g9 (ej_ri2d) combined, per-aux-atom streaming implementation.
///
/// The 9·N²·P derivative integral `int3c2e_ipip2` is never materialized in
/// full: it is evaluated per-aux-atom block (9·N²·qi) via a libcint shell
/// slice inside the i0 (aux-atom) loop and consumed immediately by both g6
/// and g9, so the full tensor never coexists with rkoo/r2c0/tmpf. `do_k`
/// gates the exchange part (g6): pure LDA/GGA DFAs (factor_k == 0) skip it.
fn g6_g9_ri2d_blas(ctx: &EjEkContext, out_ek: &mut [f64], out_ej: &mut [f64], do_k: bool) {
    let nao = ctx.nao; let naux = ctx.naux; let nocc = ctx.nocc; let natm = ctx.natm;
    let nao3 = ctx.nao3;
    let aux_blk = ctx.aux_blk; let mc2 = ctx.mc2; let rkoo = ctx.rkoo;
    let r2c0 = ctx.r2c0; let i211 = ctx.i211;
    let dm0 = ctx.dm0; let r0 = ctx.r0;
    let device = ctx.device.clone();
    let i_t = |i0: usize, j0: usize, x: usize, y: usize| -> usize {
        i0 * natm * 9 + j0 * 9 + x * 3 + y
    };

    // mc2_t: F-order [nocc, nao], element (i, p) = mc2[p*nocc + i]
    let mc2_t = rt::asarray((mc2, [nocc, nao].f(), &device));

    for i0 in 0..natm { let (_, ap0, ni_aux) = aux_blk[i0];
        if ni_aux == 0 { continue; }

        // ── ipip2 block [9, nao, nao, qi] row-major, element (x, I, J, p)
        //    at x*nao*nao*qi + I*nao*qi + J*qi + p ──
        let aux_shl0 = ctx.auxslices[i0][0] as usize;
        let aux_shl1 = ctx.auxslices[i0][1] as usize;
        let ipip2_slc: &[[usize; 2]] =
            &[[0, ctx.nreg], [0, ctx.nreg], [ctx.nreg + aux_shl0, ctx.nreg + aux_shl1]];
        let (ipip2_i0, _): (Vec<f64>, Vec<usize>) = ctx.cint_all
            .integrate_row_major("int3c2e_ipip2", "s1", Some(ipip2_slc))
            .into();

        let mut td = vec![0.0; 9];  // g9: filled in fused loop (do_k) or standalone (!do_k)
        if do_k {
            // ── rkj[p, J, I] = Σ_{i,j} rkoo[ap0+p, i, j] · mc2[J, j] · mc2[I, i]  (2 GEMMs) ──
            let mut rkj = vec![0.0; ni_aux * nao * nao];
            {
                let mut rkoo_stage = vec![0.0; ni_aux * nocc * nocc];
                for p in 0..ni_aux { for i in 0..nocc { for j in 0..nocc {
                    rkoo_stage[(p * nocc + i) + j * (ni_aux * nocc)] =
                        rkoo[(ap0 + p) * nocc * nocc + i * nocc + j];
                }}}
                let rkoo_t = rt::asarray((&rkoo_stage, [ni_aux * nocc, nocc].f(), &device));
                let tmp1 = (&rkoo_t % &mc2_t); // [ni_aux*nocc, nao]
                let tmp1_raw = tmp1.into_shape(-1).into_raw();
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

            // ── ta[x] = 0.5·Σ_{I,J,p} ipip2[x,I,J,ap0+p]·rkj[p,I,J]  and
            //    td[x] = Σ_{I,J,p} ipip2[x,I,J,ap0+p]·dm0[J,I]·r0[p]  (g9) ──
            // Fused single pass over ipip2_i0 (p contiguous): the old ta GEMM
            // staged ipip2 into [9, qi·N²] with 408 B jumps (8× read
            // amplification over 452 MB/atom), and td re-read the block.
            let mut ta = vec![0.0; 9];
            {
                let npij = ni_aux * nao3;
                // rkj_r: reorder rkj to p-inner (matches ipip2's p-fastest
                // column): rkj_r[p + j·qi + i·N·qi] = rkj[p·N² + j·N + i]
                let mut rkj_r = vec![0.0; npij];
                for i in 0..nao { for j in 0..nao { for p in 0..ni_aux {
                    rkj_r[p + j * ni_aux + i * nao * ni_aux] =
                        rkj[p * nao3 + j * nao + i];
                }}}
                for x in 0..9 { for i in 0..nao { for j in 0..nao {
                    let dmv = dm0[j * nao + i];
                    let base = x * nao * nao * ni_aux + i * nao * ni_aux + j * ni_aux;
                    let rbase = j * ni_aux + i * nao * ni_aux;
                    let mut s0 = 0.0f64; let mut s1 = 0.0f64;
                    for p in 0..ni_aux {
                        let v = ipip2_i0[base + p];
                        s0 += v * rkj_r[p + rbase];
                        s1 += v * r0[ap0 + p];
                    }
                    ta[x] += s0; td[x] += s1 * dmv;
                }}}
                for x in 0..9 { ta[x] *= 0.5; }
            }

            // ── tb[x] = -0.5 * Σ_{p,q} r2c0[ap0+p, q] · i211[x, ap0+p, q]  (1 GEMM) ──
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

            for x in 0..3 { for y in 0..3 { out_ek[i_t(i0, i0, x, y)] += ta[x * 3 + y] + tb[x * 3 + y]; }}
        }

        // ── g9 (ej_ri2d): te from i211; td was fused with ta above (do_k) ──
        if !do_k {
            for x in 0..9 { let mut s = 0.0;
                for i in 0..nao { for j in 0..nao { for p in 0..ni_aux {
                    s += ipip2_i0[x * nao * nao * ni_aux + i * nao * ni_aux + j * ni_aux + p]
                        * dm0[j * nao + i] * r0[ap0 + p];
                }}} td[x] = s;
            }
        }
        let mut te = vec![0.0; 9];
        for x in 0..9 { let mut s = 0.0;
            for p in 0..ni_aux { for q in 0..naux {
                s += r0[ap0 + p] * i211[x * naux * naux + (ap0 + p) * naux + q] * r0[q];
            }} te[x] = s;
        }
        for x in 0..3 { for y in 0..3 { out_ej[i_t(i0, i0, x, y)] += td[x * 3 + y] - te[x * 3 + y]; }}
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
    /// Exact-exchange scale: 1 for RHF, hybrid coefficient for RKS.
    pub factor_k: f64,
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
    /// r0 = V⁻¹·(t3c·dm0), computed in calc_ej_ek Phase 1a. calc_h1ao's
    /// rhoj0_P is mathematically identical (Σ_block V⁻¹·t3c_block·dm0_block),
    /// so it is handed over instead of recomputed.
    pub r0: Vec<f64>,
    /// rk = V⁻¹·(t3c·mc2), likewise reused as calc_h1ao's rhok0_Pl_.
    pub rk: Vec<f64>,
}

impl RIRHFHessian<'_> {
    pub fn new(scf_data: &SCF) -> RIRHFHessian<'_> {
        match scf_data.scftype {
            scf_io::SCFType::RHF => {}
            _ => panic!("SCF type is not suitable for RHF Hessian."),
        }
        // Auto-detect RKS: pure DFA keeps factor_k=0 (no exact exchange);
        // hybrid DFA (e.g. B3LYP, hyb=0.2) scales the K term by hyb so that
        //   h_partial = e1 + ej - hyb*ek        (PySCF df/hessian/rks.py:60)
        //   h1ao      = h1 + vj1 - 0.5*hyb*vk1  (PySCF df/hessian/rks.py:109)
        let is_dft = !scf_data.mol.xc_data.dfa_compnt_scf.is_empty();
        let factor_k = if is_dft {
            scf_data.mol.xc_data.dfa_hybrid_scf
        } else {
            1.0
        };
        RIRHFHessian {
            scf_data,
            factor_k,
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
        self.scf_data.mol.xc_data.is_hybrid()
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
        let scf = self.scf_data;
        if scf.mol.ctrl.print_level > 0 {
            println!("  >> Entering h_partial (ej_ek) stage ...");
        }
        let mol = &scf.mol;
        let nao = mol.num_basis;
        let natm = mol.geom.nfree;
        let aoslices = mol.aoslice_by_atom();
        // SCF data
        let dm0_mat = &scf.density_matrix[0];
        let dm0: Vec<f64> = dm0_mat.iter().copied().collect(); // col-major
        let c = &scf.eigenvectors[0];
        let eps = &scf.eigenvalues[0];
        let mo_occ = &scf.occupation[0];
        let nocc = (scf.homo[0] + 1) as usize;
        // mocc_2 = C_occ · sqrt(occ)  (only occupied cols, weighted)
        let mut mc2 = vec![0.0; nao * nocc];
        for p in 0..nao {
            for i in 0..nocc {
            mc2[p * nocc + i] = c[[p, i]] * (mo_occ[i] as f64).sqrt();
            }
        }
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
        let (int2c_v, _) = cint_all
            .integrate_row_major("int2c2e", "s1", Some(&aux_slc[..]))
            .into();
        let mut int2c = vec![0.0; naux * naux];
        if int2c_v.len() == naux * naux {
            int2c = int2c_v;
        } else {
            // triangular expansion
            let mut idx = 0;
            for j in 0..naux {
                for i in 0..=j {
                int2c[i + j * naux] = int2c_v[idx];
                    int2c[j + i * naux] = int2c_v[idx];
                    idx += 1;
                }
            }
        }
        // Column-major V^{-1}
        let mut int2c_cm = vec![0.0; naux * naux];
        for p in 0..naux {
            for q in 0..naux {
                int2c_cm[p + q * naux] = int2c[p * naux + q];
            }
        }
        let vinv = compute_vinv(&int2c_cm, naux);
        let i2inv = vinv.clone();

        // ── 2c-2e auxiliary basis integrals (via combined CInt with aux-only slice) ──
        let aux_slc_ref: &[[usize; 2]] = &[[nreg, nreg + naux_shell], [nreg, nreg + naux_shell]];
        // Use cint_all with shell slice for aux-only derivative integrals
        let (i21_v, _) = <(Vec<f64>, Vec<usize>)>::from(cint_all.integrate_row_major(
            "int2c2e_ip1",
            "s1",
            Some(aux_slc_ref),
        ));
        let (i211_v, _) = <(Vec<f64>, Vec<usize>)>::from(cint_all.integrate_row_major(
            "int2c2e_ipip1",
            "s1",
            Some(aux_slc_ref),
        ));
        let (i212_v, _) = cint_all
            .integrate_row_major("int2c2e_ip1ip2", "s1", Some(aux_slc_ref))
            .into();
        let i21 = i21_v;
        let i211 = i211_v;
        let i212 = i212_v;

        // ── Per-atom block ranges ──
        let mut blk = Vec::with_capacity(natm);
        for ia in 0..natm {
            let p0 = aoslices[ia][2] as usize;
            let p1 = aoslices[ia][3] as usize;
            blk.push((ia, p0, p1 - p0));
        }
        let mut aux_blk = Vec::with_capacity(natm);
        for ia in 0..natm {
            let p0 = auxslices[ia][2] as usize;
            let p1 = auxslices[ia][3] as usize;
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
            let (v, _) = cint_all
                .integrate_row_major(name, "s1", Some(slc_3c))
                .into();
            v
        };
        // Phase 1-3 + G3 need: t3c, ip1, ip2, ipip1, ipv.
        // Each derivative integral is computed lazily right before its first
        // consumer and dropped after its last, so the O(9·N²·P) tensors never
        // coexist: ipip1 → Phase 1b, ip1 → Phase 2 (kept through g4), ip2 →
        // Phase 3b, ipv → Phase 4 G3/G4. This keeps the calc_ej_ek peak RSS
        // down to ~the largest single 3c derivative tensor + staging buffers.
        let mem_trace = std::env::var("REST_MEM_TRACE").is_ok();
        let mt = |tag: &str| {
            if mem_trace {
                eprintln!("MEMTRACE {:<22} RSS = {:.1} MiB",
                    tag, memory_monitor::current_rss_mb());
            }
        };
        let t3c  = int3c("int3c2e");        // (N,N,P)
        mt("t3c alloc");
        // int3c2e_ip1 (2.7 GB full) is NEVER materialized: every consumer
        // (Phase 2 wj1, g3_g4 W2/Z2u, g5 rho2c/tmpf_j0) integrates the
        // per-atom block it needs and drops it. Peak RSS contribution of the
        // ip1 integrals drops from 2.7 GB to one 150-234 MB block.
        // H2 optimization: cache the two 2c/3c integrals that `calc_h1ao`
        // would otherwise recompute (int3c2e is moved in after Phase 1a).
        self.shared_integrals = Some(SharedHessianIntegrals {
            int2c2e_ip1: i21.clone(),
            vinv: vinv.clone(),
            int3c2e: Vec::new(),
            r0: Vec::new(),
            rk: Vec::new(),
        });
        self.timings
            .push(("  ej_ek: integrals", _t_global.elapsed()));
        let _tej = std::time::Instant::now();

        // ── rstsr device and tensor views (shared across all phases) ──
        let device = DeviceBLAS::default();
        // Column-major tensors (native F-order)
        let dm0_t: TsrView<f64> = rt::asarray((&dm0, [nao, nao].f(), &device));
        let mc2_t: TsrView<f64> = rt::asarray((&mc2, [nocc, nao].f(), &device));
        let vinv_t: TsrView<f64> = rt::asarray((&vinv, [naux, naux].f(), &device));
        // Row-major integrals → F-order by reversing dims (p fastest → first axis)
        let t3c_t: TsrView<f64> = rt::asarray((&t3c, [naux, nao, nao].f(), &device));
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
            // Explicit SIMD loop: r0r[p] = Σ_{i,j} t3c[i,j,p]·dm0[i,j], p inner
            // (t3c read p-contiguous, r0r accumulate contiguous). The old
            // staging wrote t3c_pij with 7 KB jumps on both sides over 902 MB.
            for i in 0..nao { for j in 0..nao {
                let dmv = dm0[i * nao + j];
                let base = i * nao * naux + j * naux;
                let row = &t3c[base..base + naux];
                let mut p = 0usize;
                while p + 4 <= naux {
                    r0r[p] += row[p] * dmv;
                    r0r[p + 1] += row[p + 1] * dmv;
                    r0r[p + 2] += row[p + 2] * dmv;
                    r0r[p + 3] += row[p + 3] * dmv;
                    p += 4;
                }
                while p < naux { r0r[p] += row[p] * dmv; p += 1; }
            }}
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
            // SIMD: rkr[p + i*naux + occ*naux*nao] += Σ_j t3c[i,j,p]·mc2[j,occ]
            // (p inner, t3c read p-contiguous; no 0.9 GB t3c_pi_j_stage —
            //  the old staging + GEMM held t3c AND the staging copy together,
            //  inflating the Phase 1a peak to ~2.5 GB).
            for i in 0..nao {
                for j in 0..nao {
                    let rbase = i * nao * naux + j * naux;
                    let row = &t3c[rbase..rbase + naux];
                    for occ in 0..nocc {
                        let mc = mc2[j * nocc + occ];
                        let wbase = i * naux + occ * naux * nao;
                        let mut p = 0usize;
                        while p + 4 <= naux {
                            rkr[p + wbase] += row[p] * mc;
                            rkr[p + 1 + wbase] += row[p + 1] * mc;
                            rkr[p + 2 + wbase] += row[p + 2] * mc;
                            rkr[p + 3 + wbase] += row[p + 3] * mc;
                            p += 4;
                        }
                        while p < naux { rkr[p + wbase] += row[p] * mc; p += 1; }
                    }
                }
            }
        }
        // ── Apply V⁻¹: r0 = V⁻¹ · r0r  (GEMM), rk = V⁻¹ · rkr  (GEMM) ──
        // r0_col[naux, 1] = vinv_t[naux, naux] @ r0r_col[naux, 1]
        // rk_2d[naux, nao*nocc] = vinv_t[naux, naux] @ rkr_t_2d[naux, nao*nocc]
        //   rkr viewed as F-order [naux, nao*nocc]: element (p, i + oc*nao) = rkr[p, i, oc]
        //   F-order flat: p + (i + oc*nao)*naux = p + i*naux + oc*nao*naux (matches rkr source)
        //   result element (p, i + oc*nao) = Σ_q vinv[p, q] · rkr[q, i, oc] = rk[p, i, oc]  ✓
        //   rk layout matches rkr (downstream code indexes rk[p + i*naux + oc*naux*nao])
        let mut r0 = vec![0.0; naux];
        let mut rk = vec![0.0; naux * nao * nocc];
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

        // t3c (and its unused view t3c_t) is no longer needed after Phase 1a.
        // Dropped here instead of moving into shared_integrals: calc_h1ao now
        // re-integrates int3c2e per-AO-atom block (shell slice), so the
        // 0.9 GB (N,N,P) full tensor never coexists with ip1/tmpf/wki.
        drop(t3c_t);
        drop(t3c);
        // Hand off r0/rk (computed in Phase 1a above) to calc_h1ao, which
        // would otherwise rebuild rhoj0_P / rhok0_Pl_ from scratch.
        self.shared_integrals.as_mut().unwrap().r0 = r0.clone();
        self.shared_integrals.as_mut().unwrap().rk = rk.clone();

        self.timings.push(("  p1a_rhoj0_rhok0", _tp1a.elapsed()));
        mt("after Phase1a");
        // ══ Phase 1b: vj1_diag, vk1_diag (prototype L68-76) ══
        //
        // The 9·N²·P derivative integral `int3c2e_ipip1` is never materialized
        // in full: it is evaluated per-AO-atom block (9·ni·N·P) via libcint
        // shell slices, contracted immediately into vjd/vkd, and dropped. Peak
        // integral memory drops from 9·N²·P to 9·ni·N·P (÷natm).
        let _tp1b = std::time::Instant::now();
        let mut vjd = vec![0.0; 9 * nao * nao];
        let mut vkd = vec![0.0; 9 * nao * nao];
        // vkd[x, i, l] = Σ_{p,j} ipip1[x, i, j, p] · rkm[p, l, j], with
        // rkm[p,l,j] = Σ_occ rk[p,l,occ]·mc2[j,occ]. The 0.9 GB rkm is built
        // ONCE by GEMM (fast); the old 0.9 GB rkm_jpl_stage transposed copy
        // is gone — per (atom, l-block) only the [N·P, lcnt] slice rkm_l is
        // re-staged (memcpy-class, 0.11 GB) then vkd_l = rkm_lᵀ·ipip1 GEMMs.
        let r0_col_t = rt::asarray((&r0, [naux, 1].f(), &device));
        // rkm[p,l,j] = Σ_occ rk[p,l,occ]·mc2[j,occ] is NOT built in full
        // (0.9 GB): per (atom, l-block) the [P·lc, N] slice is assembled by
        // GEMM rk[l-block]·mc2ᵀ and re-staged to [N·P, lc], so the peak
        // Phase 1b RSS is ipip1 block (0.45 GB) + rkm_l (0.11 GB) + rk.
        // l-block count: ~45 l per block → 8 blocks for nao = 358.
        let lcnt = 45usize;

        // Stream over AO atom blocks: one libcint shell-slice call per atom,
        // contract into vjd/vkd, drop the block.
        for (ia, &(_, p0, ni)) in blk.iter().enumerate() {
            if ni == 0 { continue; }
            if mem_trace && ia % 3 == 0 {
                eprintln!("MEMTRACE p1b-iter-{:02}         RSS = {:.1} MiB", ia, memory_monitor::current_rss_mb());
            }
            let shl0 = aoslices[ia][0] as usize;
            let shl1 = aoslices[ia][1] as usize;
            let ipip1_slc: &[[usize; 2]] =
                &[[shl0, shl1], [0, nreg], [nreg, nreg + naux_shell]];
            let (ipip1_b, _): (Vec<f64>, Vec<usize>) = cint_all
                .integrate_row_major("int3c2e_ipip1", "s1", Some(ipip1_slc))
                .into();
            if mem_trace {
                eprintln!("MEMTRACE p1b-ipip1-{:02}     RSS = {:.1} MiB", ia, memory_monitor::current_rss_mb());
            }
            // ipip1_b layout: [9, ni, nao, naux] row-major, element (x, ii, j, p)
            //   at x*ni*nao*naux + ii*nao*naux + j*naux + p

            // ── vjd 块: vjd[x, p0+ii, j] = Σ_p ipip1_b[x, ii, j, p] · r0[p]  ──
            // Explicit SIMD loop (same pattern as g3's vj1_mat): the staging
            // + [9·ni·N, P]×[P,1] GEMM wrote `st` with an 800 KB column stride
            // (cache-hostile). The plain loop reads ipip1_b (p contiguous) and
            // r0 (L2-resident) at full DRAM rate.
            for x in 0..9 { for ii in 0..ni { for j in 0..nao {
                let base = x * ni * nao * naux + ii * nao * naux + j * naux;
                let row = &ipip1_b[base..base + naux];
                let mut s0 = 0.0f64; let mut s1 = 0.0f64; let mut s2 = 0.0f64; let mut s3 = 0.0f64;
                let mut p = 0usize;
                while p + 4 <= naux {
                    s0 += row[p] * r0[p];
                    s1 += row[p + 1] * r0[p + 1];
                    s2 += row[p + 2] * r0[p + 2];
                    s3 += row[p + 3] * r0[p + 3];
                    p += 4;
                }
                while p < naux { s0 += row[p] * r0[p]; p += 1; }
                vjd[x * nao3 + (p0 + ii) * nao + j] = s0 + s1 + s2 + s3;
            }}}

            // ── vkd 块: vkd[x, p0+ii, l] = Σ_{p,j} ipip1_b[x, ii, j, p] · rkm[p, l, j] ──
            // Zero-copy ipip1 view: row-major [9, ni, N, P] IS F-order
            // [P·N, m9] (row (p,j) with p fastest, col (ii,x)). Per l-block:
            // rkm_l [N·P, lcnt] built on the fly, vkd_lᵀ = rkm_lᵀ·ipip1.
            {
                let m9 = 9 * ni;
                let ipip1_view = rt::asarray((&ipip1_b, [naux * nao, ni * 9].f(), &device));
                for lb in (0..nao).step_by(lcnt) {
                    let le = (lb + lcnt).min(nao);
                    let lc = le - lb;
                    // rkm_l [(j*naux+p) + l_loc*(nao*naux)] = Σ_occ rk[p,lb+l_loc,occ]·mc2[j,occ]
                    // SIMD: per (l_loc,p), outer occ loop accumulates s[j] (contiguous).
                    // rkm_l [(j*naux+p) + l_loc*(nao*naux)] = Σ_occ rk[p,lb+l_loc,occ]
                    // ·mc2[j,occ] — GEMM rk[l-block]·mc2ᵀ then re-stage.
                    let mut rk_l_stage = vec![0.0; naux * lc * nocc];
                    for p in 0..naux {
                        for l_loc in 0..lc {
                            for occ in 0..nocc {
                                rk_l_stage[(p * lc + l_loc) + occ * (naux * lc)] =
                                    rk[p + (lb + l_loc) * naux + occ * naux * nao];
                            }
                        }
                    }
                    if mem_trace && ia == 0 && lb == 0 {
                        eprintln!("MEMTRACE p1b-rkml-gemm-a  RSS = {:.1} MiB", memory_monitor::current_rss_mb());
                    }
                    let rk_l_t = rt::asarray((&rk_l_stage, [naux * lc, nocc].f(), &device));
                    let c_t = &rk_l_t % &mc2_t; // [P·lc, N] row (p,l), col j
                    if mem_trace && ia == 0 && lb == 0 {
                        eprintln!("MEMTRACE p1b-rkml-gemm-b  RSS = {:.1} MiB", memory_monitor::current_rss_mb());
                    }
                    let c_raw = c_t.into_shape(-1).into_raw();
                    let mut rkm_l = vec![0.0; nao * naux * lc];
                    for p in 0..naux {
                        for l_loc in 0..lc {
                            for j in 0..nao {
                                rkm_l[j * naux + p + l_loc * (nao * naux)] =
                                    c_raw[(p * lc + l_loc) + j * (naux * lc)];
                            }
                        }
                    }
                    if mem_trace && ia == 0 && lb == 0 {
                        eprintln!("MEMTRACE p1b-vkdgemm      RSS = {:.1} MiB", memory_monitor::current_rss_mb());
                    }
                    let rkm_l_t = rt::asarray((&rkm_l, [naux * nao, lc].f(), &device));
                    if mem_trace && ia == 0 && lb == 0 {
                        crate::hessian::memory_monitor::trim_to_os(0);
                        eprintln!("MEMTRACE p1b-vkdgemm-b    RSS = {:.1} MiB", memory_monitor::current_rss_mb());
                    }
                    let vkd_l = &rkm_l_t.t() % &ipip1_view; // [lc, 9·ni]
                    if mem_trace && ia == 0 && lb == 0 {
                        eprintln!("MEMTRACE p1b-vkdgemm-a    RSS = {:.1} MiB", memory_monitor::current_rss_mb());
                    }
                    let raw = vkd_l.into_shape(-1).into_raw();
                    for x in 0..9 {
                        for ii in 0..ni {
                            for l_loc in 0..lc {
                                vkd[x * nao3 + (p0 + ii) * nao + (lb + l_loc)] +=
                                    raw[l_loc + (x * ni + ii) * lc];
                            }
                        }
                    }
                }
            }
            drop(ipip1_b);
        }
        self.timings.push(("  p1b_vjd_vkd", _tp1b.elapsed()));
        mt("after Phase1b");
        // ══ Phase 2: rhoj1, wj1 (prototype L79-88) ══
        let _tp2 = std::time::Instant::now();
        let m3 = 3 * nao * nao;
        let mut rj1 = vec![0.0; natm * naux * 3];
        let mut wj1 = vec![0.0; natm * naux * 3];
        // wj1 computed per atom; rj1 = V⁻¹·wj1 is deferred to a single GEMM
        // below (the V⁻¹ transform is no longer applied to a full 3·N²·P
        // tmpf = V⁻¹@ip1ᵀ intermediate — each consumer applies V⁻¹ to its own
        // small factor instead).
        // For each atom ib, for each x:
        //   wj1[ib, p, x] = Σ_{ii, j} ip1[x, p0+ii, j, p] · dm0[(p0+ii)*nao + j]
        // staging dm0_atom_col[ni*nao, 1] F-order: element (ii*nao+j, 0) = dm0[(p0+ii)*nao+j]
        // staging ip1_x_atom_2d[naux, ni*nao] F-order per (ib, x): element (p, ii*nao+j) = ip1[x*nao²*naux+(p0+ii)*nao*naux+j*naux+p]
        //   fill: idx = p + (ii*nao+j)*naux
        // GEMM: wj1_x_col[naux, 1] = ip1_x_atom_2d @ dm0_atom_col
        // scatter: wj1[ib*naux*3 + p*3 + x] = wj1_x_col_raw[p]
        for (ib, &(_, p0, ni)) in blk.iter().enumerate() {
            if ni == 0 { continue; }
            // wj1[ib,p,x] = Σ_{ii,j} ip1[x,p0+ii,j,p]·dm0[(p0+ii),j]
            // ip1[ib-row-block] integrated per atom (shell slice on the first
            // AO index), contracted immediately, dropped — the full 2.7 GB
            // int3c2e_ip1 tensor is never read here (and later never built).
            let shl0 = aoslices[ib][0] as usize;
            let shl1 = aoslices[ib][1] as usize;
            let ip1_slc: &[[usize; 2]] =
                &[[shl0, shl1], [0, nreg], [nreg, nreg + naux_shell]];
            let (ip1_b, _): (Vec<f64>, Vec<usize>) = cint_all
                .integrate_row_major("int3c2e_ip1", "s1", Some(ip1_slc))
                .into();
            // ip1_b layout: [3, ni, N, P] row-major, element (x, ii, j, p)
            //   at x*ni*nao*naux + ii*nao*naux + j*naux + p
            // Explicit SIMD loop (p inner): ip1_b read p-contiguous, wj1
            // accumulate at 24 B stride.
            for x in 0..3 {
                for ii in 0..ni { for j in 0..nao {
                    let dmv = dm0[(p0 + ii) * nao + j];
                    let base = x * ni * nao * naux + ii * nao * naux + j * naux;
                    let row = &ip1_b[base..base + naux];
                    let wbase = ib * naux * 3 + x;
                    let mut p = 0usize;
                    while p + 4 <= naux {
                        wj1[wbase + p * 3] += row[p] * dmv;
                        wj1[wbase + (p + 1) * 3] += row[p + 1] * dmv;
                        wj1[wbase + (p + 2) * 3] += row[p + 2] * dmv;
                        wj1[wbase + (p + 3) * 3] += row[p + 3] * dmv;
                        p += 4;
                    }
                    while p < naux { wj1[wbase + p * 3] += row[p] * dmv; p += 1; }
                }}
            }
            drop(ip1_b);
        }
        // ── rj1 = V⁻¹ @ wj1 :  rj1[ib,p,x] = Σ_q V⁻¹[p,q]·wj1[ib,q,x]  (1 GEMM) ──
        {
            let n3 = natm * 3;
            let mut wj1_mm = vec![0.0; naux * n3];
            for ib in 0..natm { for x in 0..3 {
                let col = ib * 3 + x;
                for p in 0..naux {
                    wj1_mm[p + col * naux] = wj1[ib * naux * 3 + p * 3 + x];
                }
            }}
            let wj1_mm_t = rt::asarray((&wj1_mm, [naux, n3].f(), &device));
            let rj1_mm = &vinv_t % &wj1_mm_t; // [naux, n3]
            let rj1_raw = rj1_mm.into_shape(-1).into_raw();
            for ib in 0..natm { for x in 0..3 {
                let col = ib * 3 + x;
                for p in 0..naux {
                    rj1[ib * naux * 3 + p * 3 + x] = rj1_raw[p + col * naux];
                }
            }}
        }

        // tmpf = V⁻¹·ip1ᵀ is NOT materialized anymore: g5 absorbs the V⁻¹
        // weight into wk1_pJI/wk1_IpJ per aux-atom block (see g5_g8_ri1_blas),
        // so the 2.7 GB (P, 3·N²) tensor is gone entirely.

        self.timings.push(("  p2_rhoj1_wj1", _tp2.elapsed()));


        mt("after Phase2");
        // ══ Phase 3b: wj_ip2, wk_ip2_Ipk, wk_ip2_P__ (prototype L92-97) ══
        //
        // The 3·N²·P derivative integral `int3c2e_ip2` is never materialized
        // in full: it is evaluated per-AO-atom block (3·ni·N·P) and contracted
        // immediately into wj2/wki/wk2 (accumulated across blocks). Because
        // the 3c-2e integral is symmetric in its two AO indices, the single
        // first-index block ip2_b[y, ii, j, p] = ip2[y, p0+ii, j, p] also
        // provides the second-index block via transposition (used by wk2).
        let _tp3b = std::time::Instant::now();
        let mut wj2 = vec![0.0; naux * 3];
        // wki (N·P·3·N, 2.7 GB) is NOT materialized anymore: its only
        // consumer (g5's wk1_IpJ[j0-block]) now integrates int3c2e_ip2 per
        // aux-atom block and builds wk1_IpJ = dm0·ip2·dm0 on the fly.
        let mut wk2 = vec![0.0; naux * 3 * nocc * nocc];
        let dm0_col_t = rt::asarray((&dm0, [nao3, 1].f(), &device));

        for (ib, &(_, p0, ni)) in blk.iter().enumerate() {
            if ni == 0 { continue; }
            let shl0 = aoslices[ib][0] as usize;
            let shl1 = aoslices[ib][1] as usize;
            let ip2_slc: &[[usize; 2]] = &[[shl0, shl1], [0, nreg], [nreg, nreg + naux_shell]];
            let (ip2_b, _): (Vec<f64>, Vec<usize>) = cint_all
                .integrate_row_major("int3c2e_ip2", "s1", Some(ip2_slc))
                .into();
            // ip2_b[y*ni*nao*naux + ii*nao*naux + j*naux + p] = ip2[y, p0+ii, j, p]

            // ── wj2[p, y] += Σ_{ii∈block, j} ip2_b[y, ii, j, p] · dm0[(p0+ii), j] ──
            // Explicit SIMD loop (p inner): the staging + [P, ni·N]×[ni·N,1]
            // GEMM wrote `st` with 7 KB jumps on both sides. The plain loop
            // reads ip2_b p-contiguous and accumulates into wj2 (24 B stride).
            for y in 0..3 {
                for ii in 0..ni { for j in 0..nao {
                    let dmv = dm0[(p0 + ii) * nao + j];
                    let base = y * ni * nao * naux + ii * nao * naux + j * naux;
                    let row = &ip2_b[base..base + naux];
                    let mut p = 0usize;
                    while p + 4 <= naux {
                        wj2[p * 3 + y] += row[p] * dmv;
                        wj2[(p + 1) * 3 + y] += row[p + 1] * dmv;
                        wj2[(p + 2) * 3 + y] += row[p + 2] * dmv;
                        wj2[(p + 3) * 3 + y] += row[p + 3] * dmv;
                        p += 4;
                    }
                    while p < naux { wj2[p * 3 + y] += row[p] * dmv; p += 1; }
                }}
            }

            // ── wk2[p, x, i, j] += Σ_{u, v∈block} ip2[x, u, v, p] · mc2[u, i] · mc2[v, j]
            //    Step 1: tmp1[ii, p, i] = Σ_u ip2_b[x, ii, u, p] · mc2[u, i]  (symmetry: v=p0+ii)
            //    Step 2: wk2[x, p, i, j] += Σ_ii mc2[p0+ii, j] · tmp1[ii, p, i] ──
            for x in 0..3 {
                // staging: ii,u outer + p inner → both sides stride-1.
                let mut st = vec![0.0; ni * naux * nao];
                for ii in 0..ni { for u in 0..nao {
                    let base_s = ii * naux + u * (ni * naux);
                    let base_i = x * ni * nao * naux + ii * nao * naux + u * naux;
                    for p in 0..naux { st[base_s + p] = ip2_b[base_i + p]; }
                }}
                let t = rt::asarray((&st, [ni * naux, nao].f(), &device));
                let tmp1 = &t % &mc2_t.t(); // [ni*naux, nocc]
                let tmp1_raw = tmp1.into_shape(-1).into_raw();
                // tv scatter: ii,i outer + p inner → tmp1 read stride-1,
                // tv write 272 B stride (was 214 KB read jumps).
                let mut tv = vec![0.0; ni * naux * nocc];
                for ii in 0..ni { for i in 0..nocc {
                    let rbase = ii * naux + i * (ni * naux);
                    for p in 0..naux {
                        tv[ii + (p * nocc + i) * ni] = tmp1_raw[rbase + p];
                    }
                }}
                let tv_t = rt::asarray((&tv, [ni, naux * nocc].f(), &device));
                // mc2 block rows [p0..p0+ni] as F-order [nocc, ni]: element (j, ii) = mc2[(p0+ii)*nocc + j]
                let mut mc2_blk = vec![0.0; nocc * ni];
                for j in 0..nocc { for ii in 0..ni {
                    mc2_blk[j + ii * nocc] = mc2[(p0 + ii) * nocc + j];
                }}
                let mc2_blk_t = rt::asarray((&mc2_blk, [nocc, ni].f(), &device));
                let wk2_x = &mc2_blk_t % &tv_t; // [nocc, naux*nocc]
                let raw = wk2_x.into_shape(-1).into_raw();
                for p in 0..naux { for i in 0..nocc { for j in 0..nocc {
                    wk2[p * 3 * nocc * nocc + x * nocc * nocc + i * nocc + j] +=
                        raw[j + (p * nocc + i) * nocc];
                }}}
            }
            drop(ip2_b);
        }

        self.timings.push(("  p3b_wj2_wk2", _tp3b.elapsed()));
        mt("after Phase3b");
        if mem_trace {
            crate::hessian::memory_monitor::trim_to_os(0);
            eprintln!("MEMTRACE after-trim-3b      RSS = {:.1} MiB", memory_monitor::current_rss_mb());
        }
        // ══ Phase 3c: rhok0_P__, rho2c_0, int2c_ip_ip (prototype L98-106) ══
        let _tp3c = std::time::Instant::now();
        let mut rkoo = vec![0.0; naux * nocc * nocc];
        for p in 0..naux {
            for i in 0..nocc {
                for jj in 0..nocc {
                    let mut s = 0.0;
                    for l in 0..nao {
                        s += rk[p + l * naux + i * naux * nao] * mc2[l * nocc + jj];
                    }
            rkoo[p * nocc * nocc + i * nocc + jj] = s;
                }
            }
        }
        let mut r2c0 = vec![0.0; naux * naux];
        for p in 0..naux {
            for q in 0..naux {
                let mut s = 0.0;
                for i in 0..nocc {
                    for j in 0..nocc {
                        s += rkoo[p * nocc * nocc + i * nocc + j]
                            * rkoo[q * nocc * nocc + j * nocc + i];
                    }
                }
            r2c0[p + q * naux] = s;
            }
        }
        // Phase 3c: i2ip via rstsr matmul (P1 priority)
        // i2ip[xy,p,s] = (i21[x] @ i2inv @ i21[y]^T)[p,s] - i212[xy,p,s]
        // Step 1: reshape i21 to (3*naux, naux), matmul with i2inv -> (3*naux, naux)
        let mut i2ip = vec![0.0; 9 * naux * naux];
        {
            let mut i21_mm = vec![0.0; 3 * naux * naux];
            for x in 0..3 {
                for p in 0..naux {
                    for q in 0..naux {
                        i21_mm[(x * naux + p) + q * (3 * naux)] =
                            i21[x * naux * naux + p * naux + q];
                    }
                }
            }
            let i21_blas = rt::asarray((&i21_mm, [3 * naux, naux].f(), &device));
            let tmp = &i21_blas % &i2inv_t; // tmp[x*naux+p, r]
            // Step 2: result = tmp @ i21^T as (3*naux, 3*naux)
            let mut i21_mm_t = vec![0.0; naux * 3 * naux];
            for x in 0..3 {
                for q in 0..naux {
                    for p in 0..naux {
                i21_mm_t[q + (x * naux + p) * naux] = i21[x * naux * naux + p * naux + q];
                    }
                }
            }
            let i21_blas_t = rt::asarray((&i21_mm_t, [naux, 3 * naux].f(), &device));
            let i2ip_full = &tmp % &i21_blas_t;
            let i2ip_flat = i2ip_full.into_shape(-1).into_raw();
            for x in 0..3 {
                for y in 0..3 {
                let xy = x * 3 + y;
                    for p in 0..naux {
                        for s in 0..naux {
                    let val = i2ip_flat[(x * naux + p) + (y * naux + s) * (3 * naux)];
                            i2ip[xy * naux * naux + p * naux + s] =
                                val - i212[xy * naux * naux + p * naux + s];
                        }
                    }
                }
            }
        }
        // i212 (9*naux²) is only consumed inside the i2ip block above.
        drop(i212);
        let mut wj001 = vec![0.0; 3 * naux];
        for y in 0..3 {
            for p in 0..naux {
                let mut s = 0.0;
                for q in 0..naux {
                    s += i21[y * naux * naux + p * naux + q] * r0[q];
                }
            wj001[y * naux + p] = s;
            }
        }

        // ══════════════════════════════════════════════════════════
        self.timings.push(("  p3c_rkoo_r2c0_i2ip", _tp3c.elapsed()));
        self.timings.push(("  phases_1-3", _tej.elapsed()));

        // Total AO+aux shells for per-atom libcint shell-slice calls.
        let naux_nbas = nreg + naux_shell;
        // Build a read-only view of the Phase 1-3 intermediates for each G-term.
        macro_rules! build_ej_ek_ctx {
            ($ip1:expr, $tmpf:expr) => {
                EjEkContext {
                    nao,
                    naux,
                    nocc,
                    natm,
                    nao3,
                    blk: &blk,
                    aux_blk: &aux_blk,
                    dm0: &dm0,
                    mc2: &mc2,
                    i2inv: &i2inv,
                    i21: &i21,
                    i211: &i211,
                    i2ip: &i2ip,
                    r0: &r0,
                    rk: &rk,
                    rj1: &rj1,
                    wj1: &wj1,
                    vjd: &vjd,
                    vkd: &vkd,
                    wj2: &wj2,
                    wk2: &wk2,
                    rkoo: &rkoo,
                    r2c0: &r2c0,
                    wj001: &wj001,
                    ip1: $ip1,
                    tmpf: $tmpf,
                device: DeviceBLAS::default(),
                    cint_all: &cint_all,
                    nreg,
                    aux_nbas: naux_nbas,
                    aoslices: &aoslices,
                    auxslices: &auxslices,
                }
            };
        }

        // Phase 4: Contribution arrays
        // ══════════════════════════════════════════════════════════
        mt("after Phase3c");
        let _tp4 = std::time::Instant::now();
        let aa9 = natm * natm * 9;
        let mut ej_basic = vec![0.0; aa9];
        let mut ej_vjd = vec![0.0; aa9];
        let mut ej_vj1 = vec![0.0; aa9];
        let mut ej_ri1 = vec![0.0; aa9];
        let mut ej_ri2d = vec![0.0; aa9];
        let mut ej_ri2o = vec![0.0; aa9];
        let mut ek_vkd = vec![0.0; aa9];
        let mut ek_vk1 = vec![0.0; aa9];
        let mut ek_ri1 = vec![0.0; aa9];
        let mut ek_ri2d = vec![0.0; aa9];
        let mut ek_ri2o = vec![0.0; aa9];
        let i_t = |i0, j0, x, y| i0 * natm * 9 + j0 * 9 + x * 3 + y;

        // ══ rstsr accelerated G1-G2 (standard path, replaces for loops) ══
        let device = DeviceBLAS::default();
        let dm0_t: TsrView<f64> = rt::asarray((&dm0, [nao, nao].f(), &device));
        let vjd_t: TsrView<f64> = rt::asarray((&vjd, [nao, nao, 9].f(), &device));
        let vkd_t: TsrView<f64> = rt::asarray((&vkd, [nao, nao, 9].f(), &device));
        let r0_t: TsrView<f64> = rt::asarray((&r0, [naux].f(), &device));
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
            for c in 0..9 {
                let (x, y) = (c / 3, c % 3);
                ej_vjd[i_t(i0, i0, x, y)] = sum_j_v[c] * 2.0;
                ek_vkd[i_t(i0, i0, x, y)] = sum_k_v[c];
            }
        }
        // Free rstsr views used by G1/G2 before the g3+g4 block stream.
        drop((dm0_t, vjd_t, vkd_t, r0_t, rj1_t, wj1_t, ej_rstsr));

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

        // g4/g5/g6/g7 build the exchange block `ek`. For pure LDA/GGA DFAs
        // factor_k == 0, so `ek` is scaled out of h_partial entirely — skip the
        // exchange G-terms (and their O(naux·nao³) contractions) altogether.
        // g4 is the last consumer of ip1.
        {
            let _t_g4 = std::time::Instant::now();
            let ctx = build_ej_ek_ctx!(&[], &[]);
            g3_g4_vj1_vk1_blas(&ctx, &mut ej_vj1, &mut ek_vk1, self.factor_k != 0.0);
            self.timings.push(("  g3_g4_vj1_vk1", _t_g4.elapsed()));
        }

        mt("after g3_g4");
        if mem_trace {
            eprintln!("MEMTRACE g3_g4-start        RSS = {:.1} MiB", memory_monitor::current_rss_mb());
        }
        if mem_trace {
            crate::hessian::memory_monitor::trim_to_os(0);
            eprintln!("MEMTRACE trim-g34         RSS = {:.1} MiB", memory_monitor::current_rss_mb());
        }
        // g4 consumed ip1 (W2/Z2u). g5 still reads ip1 (t2/t4 V⁻¹-absorbed
        // forms), so ip1 stays alive until after g5.

        // ip12 (9·N²·P) is evaluated per-AO-atom block inside g5_g8_ri1_blas,
        // so the full tensor never exists. ipip2 (below) is likewise streamed
        // per-aux-atom block by g6/g9.
        {
            let _t_e5 = std::time::Instant::now();
            let ctx = build_ej_ek_ctx!(&[], &[]);
            g5_g8_ri1_blas(&ctx, &mut ek_ri1, &mut ej_ri1, self.factor_k != 0.0);
            self.timings.push(("  g5g8_ri1", _t_e5.elapsed()));
        }

        // int3c2e_ip1 was never materialized; nothing to drop here.
        mt("after g5_g8");
        if mem_trace {
            crate::hessian::memory_monitor::trim_to_os(0);
            eprintln!("MEMTRACE after-trim-g5      RSS = {:.1} MiB", memory_monitor::current_rss_mb());
        }

        // ipip2 (9·N²·P) is evaluated per-aux-atom block inside g6_g9_ri2d_blas,
        // so the full tensor never exists.
        {
            let _t_e6 = std::time::Instant::now();
            let ctx = build_ej_ek_ctx!(&[], &[]);
            g6_g9_ri2d_blas(&ctx, &mut ek_ri2d, &mut ej_ri2d, self.factor_k != 0.0);
            self.timings.push(("  g6g9_ri2d", _t_e6.elapsed()));

            let _t_e7 = std::time::Instant::now();
            let ctx = build_ej_ek_ctx!(&[], &[]);
            g7_ek_ri2o_blas(&ctx, &mut ek_ri2o);
            self.timings.push(("  ek_ri2o", _t_e7.elapsed()));
        }

                let _t_e10 = std::time::Instant::now();
            let ctx = build_ej_ek_ctx!(&[], &[]);
                g10_ej_ri2o_blas(&ctx, &mut ej_ri2o);
                self.timings.push(("  ej_ri2o", _t_e10.elapsed()));

        // All G-term contractions are done. The raw RI integrals (ip1, ipv,
        // ip12, ipip2, i21, i211, i2ip, tmpf) and the large Phase 1-3
        // intermediates they were contracted into (wj2, wki, wk2, rkoo, r2c0,
        // wj001, r0, rk, rj1, wj1, vjd, vkd) plus their rstsr views are no
        // longer referenced. Drop them now so the subsequent h_partial
        // assembly and RKS XC contributions run without carrying
        // O(naux·nao²) memory.
        drop((
            i2ip_t, i211_t, wk2_t, rkoo_t, r2c0_t, wj2_t, wj001_t, r0_t_ri, rj1_t_ri,
        ));
        // ip1, ipv, ip12, ipip2 already dropped after their last G-term use.
        drop((i21, i211, i2ip));
        drop((
            wj2,
            wk2,
            rkoo,
            r2c0,
            wj001,
            r0,
            rk,
            rj1,
            wj1,
            vjd,
            vkd,
        ));
        // ── Sum contributions ──
        let mut ej = vec![0.0; aa9];
        let mut ek = vec![0.0; aa9];
        for i in 0..aa9 {
            ej[i] = ej_basic[i] + ej_vjd[i] + ej_vj1[i] + ej_ri1[i] + ej_ri2d[i] + ej_ri2o[i];
            ek[i] = ek_vkd[i] + ek_vk1[i] + ek_ri1[i] + ek_ri2d[i] + ek_ri2o[i];
        }
        if std::env::var("REST_VERIFY_WK1").is_ok() {
            let sum = |v: &[f64]| v.iter().fold(0.0f64, |a, &b| a + b);
            eprintln!("COMP ek_ri1={:.6e} ek_vk1={:.6e} ek_ri2d={:.6e} ek_ri2o={:.6e}",
                sum(&ek_ri1), sum(&ek_vk1), sum(&ek_ri2d), sum(&ek_ri2o));
            eprintln!("COMP ej_ri1={:.6e} ej_vj1={:.6e} ej_ri2d={:.6e}",
                sum(&ej_ri1), sum(&ej_vj1), sum(&ej_ri2d));
        }
        self.timings.push(("  phases_4-sum", _tp4.elapsed()));
        self.timings.push(("  ej_ek: contributions", _tej.elapsed()));
        // Symmetrize: (i0,j0) → (j0,i0) by copying
        for i0 in 0..natm {
            for j0 in 0..i0 {
                for x in 0..3 {
                    for y in 0..3 {
                        let a = i_t(i0, j0, x, y);
                        let b = i_t(j0, i0, y, x);
                        ej[b] = ej[a];
                        ek[b] = ek[a];
                    }
                }
            }
        }

        // ── Store results ──
        // Flatten to col-major [n3, n3] MatrixFull
        let to_mat = |arr: &[f64]| -> MatrixFull<f64> {
            let n3 = natm * 3;
            let mut m = vec![0.0; n3 * n3];
            for i0 in 0..natm {
                for j0 in 0..natm {
                    for x in 0..3 {
                        for y in 0..3 {
                            m[(i0 * 3 + x) + (j0 * 3 + y) * n3] = arr[i_t(i0, j0, x, y)];
                        }
                    }
                }
            }
            // Symmetrize j0 < i0
            for i0 in 0..natm {
                for j0 in 0..i0 {
                    for x in 0..3 {
                        for y in 0..3 {
                            m[(j0 * 3 + y) + (i0 * 3 + x) * n3] =
                                m[(i0 * 3 + x) + (j0 * 3 + y) * n3];
                        }
                    }
                }
            }
            MatrixFull::from_vec([n3, n3], m).unwrap()
        };
        self.result.insert("ej".to_string(), to_mat(&ej));
        self.result.insert("ek".to_string(), to_mat(&ek));
        // Store individual contributions for debugging
        self.result
            .insert("ej_basic".to_string(), to_mat(&ej_basic));
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

        // ── Finish h_partial ──
        // h_partial = e1 + ej - factor_k*ek
        let e1_arr = if let Some(e1_mat) = self.result.get("e1") {
            // Convert MatrixFull back to flat (natm, natm, 3, 3) for summation
            let n3 = natm * 3;
            let mut e1_flat = vec![0.0; aa9];
            for i0 in 0..natm {
                for j0 in 0..natm {
                    for x in 0..3 {
                        for y in 0..3 {
                            e1_flat[i_t(i0, j0, x, y)] =
                                e1_mat[[(i0 * 3 + x) as _, (j0 * 3 + y) as _]];
                        }
                    }
                }
            }
            e1_flat
        } else {
            vec![0.0; aa9]
        };
        let mut hp = vec![0.0; aa9];
        for i in 0..aa9 {
            hp[i] = e1_arr[i] + ej[i] - self.factor_k * ek[i];
        }
        self.result.insert("h_partial".to_string(), to_mat(&hp));

        // ── RKS: add XC contributions (vxc_diag + vxc_deriv2) ──
        // Scatters directly into the live h_partial buffer (zero new allocation).
        // The implementation lives in rks.rs; split-borrow of self.result.data
        // and self.timings from a shared &SCF ref compiles cleanly here.
        if self.is_rks() {
            let scf = self.scf_data;
            let hp_data = &mut self
                .result
                .get_mut("h_partial")
                .expect("h_partial not set")
                .data;
            crate::hessian::rks::add_vxc_h_partial(scf, hp_data, &mut self.timings);
        }

        self.timings.push(("calc_ej_ek", _t_global.elapsed()));
        self
    }

    /// Compute h1ao[ia] = hcore^{(1)}(ia) + vj1[ia] - 0.5*vk1[ia].
    /// Requires the RI intermediates produced by `calc_ej_ek`.
    pub fn calc_h1ao(&mut self) -> &mut Self {
        let _t = std::time::Instant::now();
        // Debug intermediates (vj1_*/vk1_* per-atom matrices, ~0.6 GB) are
        // gated behind REST_VERIFY_H1AO — they are not needed for the
        // Hessian itself and were the dominant RSS left behind in `result`.
        let keep_debug = std::env::var("REST_VERIFY_H1AO").is_ok();
        let scf = self.scf_data;
        let mol = &scf.mol;
        let nao = mol.num_basis;
        let natm = mol.geom.nfree;
        let nocc = (scf.homo[0] + 1) as usize;
        let aoslices = mol.aoslice_by_atom();
        let cint_reg = mol.initialize_cint(false);
        let nreg = cint_reg.nbas();
        let cint_all = mol.initialize_cint(true);
        let naux_shell = cint_all.nbas() - nreg;
        let auxmol = mol.make_auxmol_fake();
        let naux = auxmol.num_basis;
        let auxslices = auxmol.aoslice_by_atom();

        let shared = self
            .shared_integrals
            .as_ref()
            .expect("calc_h1ao: call calc_ej_ek() first");
        let i21 = shared.int2c2e_ip1.clone();
        let vinv = shared.vinv.clone();

        // SCF data
        let dm0_mat = &scf.density_matrix[0];
        let mut dm0 = vec![0.0; nao * nao];
        for r in 0..nao {
            for c in 0..nao {
                dm0[r * nao + c] = dm0_mat[[r, c]];
            }
        }
        let c = &scf.eigenvectors[0];
        let mo_occ_data = &scf.occupation[0];
        let mut mc2 = vec![0.0; nao * nocc];
        for p in 0..nao {
            for i in 0..nocc {
                mc2[p * nocc + i] = c[[p, i]] * (mo_occ_data[i] as f64).sqrt();
            }
        }

        // A single DeviceBLAS instance is reused by all H1AO contractions.
        // Creating DeviceBLAS::default() spawns a Rayon thread pool, so doing it
        // per-block (42× previously) was a major hidden cost.
        // All BLAS blocks share this instance.
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
        // rhoj0_P ≡ r0 and rhok0_Pl_ ≡ rk were already computed in calc_ej_ek
        // Phase 1a (V⁻¹·t3c·dm0 / V⁻¹·t3c·mc2); reuse instead of recomputing.
        let rhoj0_P = shared.r0.clone();
        // rhok0_Pl_ is row-major [naux, nao, nocc] (P*nao*nocc + l*nocc + occ),
        // while shared.rk is F-order [naux, nao, nocc] (p + l*naux + occ*naux*nao).
        // Mathematically identical values; re-layout once here (N·P·O elements).
        let rhok0_Pl_ = {
            let mut v = vec![0.0; naux * nao * nocc];
            for p in 0..naux { for l in 0..nao { for occ in 0..nocc {
                v[p * nao * nocc + l * nocc + occ] =
                    shared.rk[p + l * naux + occ * naux * nao];
            }}}
            v
        };

        // Stage i21 once as F-order [3*naux, naux]: element (x*naux+q, p) = i21[x,q,p]
        use rstsr::prelude::*;
        let mut i21_stage = vec![0.0; 3 * naux * naux];
        for x in 0..3 {
            for q in 0..naux {
                for p in 0..naux {
                    i21_stage[(x * naux + q) + p * (3 * naux)] =
                        i21[x * naux * naux + q * naux + p];
                }
            }
        }
        let i21_t = rt::asarray((&i21_stage, [3 * naux, naux].f(), &device));

        // Per-atom stream: co = V⁻¹·t3c_block is scattered straight into
        // rho0_full and immediately consumed by the wj_ip1_pij GEMM; the
        // per-atom co (rho0_all) is never stored, saving a full N²·P buffer.
        // ══ coef_cache built in TWO batches (half the atoms each): the 1.4 GB
        // cache never coexists with per-atom ip2/pij_all/rho0_qb blocks.
        // vj1_accum/vk1_accum (55 MB each) collect the coef-dependent pieces
        // across batches; batch-independent assembly runs once after. ══
        // H7a: wj1[ia][x,P] = Σ_{ii∈ia,j} ip1[x,p0+ii,j,P]·dm0[j,q0+ii], cached
        // once (47 KB/atom), reused by H7b in every batch.
        let mut wj1_cache: Vec<Vec<f64>> = Vec::with_capacity(natm);
        for ia in 0..natm {
            let shl0 = aoslices[ia][0] as usize;
            let shl1 = aoslices[ia][1] as usize;
            let q0 = aoslices[ia][2] as usize;
            let q1 = aoslices[ia][3] as usize;
            let ni = q1 - q0;
            if ni == 0 { wj1_cache.push(Vec::new()); continue; }
            let atom_slc_h7: &[[usize; 2]] = &[[shl0, shl1], [0, nreg], [nreg, nreg + naux_shell]];
            let (ip1_atom, _): (Vec<f64>, Vec<usize>) = cint_all
                .integrate_row_major("int3c2e_ip1", "s1", Some(atom_slc_h7))
                .into();
            let mut wj1 = vec![0.0; 3 * naux];
            for x in 0..3 {
                for ii in 0..ni {
                    for j in 0..nao {
                        let dmv = dm0[j * nao + (q0 + ii)];
                        let base = x * ni * nao * naux + ii * nao * naux + j * naux;
                        let row = &ip1_atom[base..base + naux];
                        let wbase = x * naux;
                        let mut p = 0usize;
                        while p + 4 <= naux {
                            wj1[wbase + p] += row[p] * dmv;
                            wj1[wbase + p + 1] += row[p + 1] * dmv;
                            wj1[wbase + p + 2] += row[p + 2] * dmv;
                            wj1[wbase + p + 3] += row[p + 3] * dmv;
                            p += 4;
                        }
                        while p < naux { wj1[wbase + p] += row[p] * dmv; p += 1; }
                    }
                }
            }
            wj1_cache.push(wj1);
            drop(ip1_atom);
        }
        let mut vj1_accum = vec![0.0; natm * 3 * nao3];
        let mut vk1_accum = vec![0.0; natm * 3 * nao3];
        let n_half = (natm + 1) / 2;
        for half in 0..2 {
            let alo = half * n_half;
            let ahi = ((half + 1) * n_half).min(natm);
            if alo >= ahi { continue; }
            let mut coef_cache: Vec<Vec<f64>> = vec![Vec::new(); natm];
            // ── coef batch build: coef_cache[ia] = V⁻¹·t3c[ia] (F-order [naux, ni·nao]) ──
            for ia in alo..ahi {
                let p0 = aoslices[ia][2] as usize;
                let p1 = aoslices[ia][3] as usize;
                let ni = p1 - p0;
                if ni == 0 { continue; }
                let shl0 = aoslices[ia][0] as usize;
                let shl1 = aoslices[ia][1] as usize;
                let t3c_slc: &[[usize; 2]] =
                    &[[shl0, shl1], [0, nreg], [nreg, nreg + naux_shell]];
                let (t3c_b, _): (Vec<f64>, Vec<usize>) = cint_all
                    .integrate_row_major("int3c2e", "s1", Some(t3c_slc))
                    .into();
                let mut block = vec![0.0; naux * ni * nao];
                for P in 0..naux {
                    for ii in 0..ni {
                        for j in 0..nao {
                            block[P * ni * nao + ii * nao + j] =
                                t3c_b[ii * nao * naux + j * naux + P];
                        }
                    }
                }
                drop(t3c_b);
                let vinv_t = rt::asarray((&vinv, [naux, naux].f(), &device));
                let mut block_stage = vec![0.0; naux * ni * nao];
                for p in 0..naux {
                    for idx in 0..ni * nao {
                        block_stage[p + idx * naux] = block[p * ni * nao + idx];
                    }
                }
                let block_t = rt::asarray((&block_stage, [naux, ni * nao].f(), &device));
                let coef_t = (&vinv_t % &block_t);
                let coef_raw = coef_t.into_shape(-1).into_raw();
                coef_cache[ia] = coef_raw;
            }
            // ── H7b: vj1_accum[ia] += Σ_{ia2∈batch} wj1_cache[ia]·coef[ia2] ──
            for ia in 0..natm {
                let wj1 = &wj1_cache[ia];
                if wj1.is_empty() { continue; }
                let mut wj1_f_stage = vec![0.0; 3 * naux];
                for x in 0..3 {
                    for p in 0..naux {
                        wj1_f_stage[x + p * 3] = wj1[x * naux + p];
                    }
                }
                let wj1_f_t = rt::asarray((&wj1_f_stage, [3, naux].f(), &device));
                let off = ia * 3 * nao3;
                for ia2 in alo..ahi {
                    let i0b = aoslices[ia2][2] as usize;
                    let i1b = aoslices[ia2][3] as usize;
                    let ni2 = i1b - i0b;
                    if ni2 == 0 { continue; }
                    let coef7_t = rt::asarray((&coef_cache[ia2], [naux, ni2 * nao].f(), &device));
                    let vj1_t = (&wj1_f_t % &coef7_t);
                    let vj1_raw = vj1_t.into_shape(-1).into_raw();
                    for x in 0..3 {
                        for ii in 0..ni2 {
                            for j in 0..nao {
                                vj1_accum[off + x * nao3 + (i0b + ii) * nao + j] +=
                                    vj1_raw[x + (ii * nao + j) * 3];
                            }
                        }
                    }
                }
            }
            // ── coef-dependent vj1/vk1 corrections (this batch) ──
            for ia in 0..natm {
                let aux_shl0 = auxslices[ia][0] as usize;
                let aux_shl1 = auxslices[ia][1] as usize;
                let q0_aux = auxslices[ia][2] as usize;
                let q1 = auxslices[ia][3] as usize;
                let qi_aux = q1 - q0_aux;
                if qi_aux == 0 { continue; }
                let ip2_slc: &[[usize; 2]] =
                    &[[0, nreg], [0, nreg], [nreg + aux_shl0, nreg + aux_shl1]];
                let (ip2_a, _): (Vec<f64>, Vec<usize>) = cint_all
                    .integrate_row_major("int3c2e_ip2", "s1", Some(ip2_slc))
                    .into();
                let mut rhoj1 = vec![0.0; 3 * qi_aux];
                for x in 0..3 {
                    for P in 0..qi_aux {
                        let mut s = 0.0;
                        for i in 0..nao {
                            for j in 0..nao {
                                s += ip2_a[x * nao * nao * qi_aux + i * nao * qi_aux + j * qi_aux + P]
                                    * dm0[j * nao + i];
                            }
                        }
                        rhoj1[x * qi_aux + P] = s;
                    }
                }
                let mut pij_all = vec![0.0; qi_aux * nao * 3 * nao];
                {
                    let mut i21_qb = vec![0.0; 3 * qi_aux * naux];
                    for x in 0..3 { for p_off in 0..qi_aux { for p in 0..naux {
                        i21_qb[(x * qi_aux + p_off) + p * (3 * qi_aux)] =
                            i21[x * naux * naux + (q0_aux + p_off) * naux + p];
                    }}}
                    let i21_qb_t = rt::asarray((&i21_qb, [3 * qi_aux, naux].f(), &device));
                    for ia2 in alo..ahi {
                        let i0 = aoslices[ia2][2] as usize;
                        let i1 = aoslices[ia2][3] as usize;
                        let ni2 = i1 - i0;
                        if ni2 == 0 { continue; }
                        let coef_t = rt::asarray((&coef_cache[ia2], [naux, ni2 * nao].f(), &device));
                        let out_t = &i21_qb_t % &coef_t;
                        let out_raw = out_t.into_shape(-1).into_raw();
                        for p_off in 0..qi_aux {
                            for i_off in 0..ni2 {
                                let ig = i0 + i_off;
                                for x in 0..3 { for j in 0..nao {
                                    pij_all[p_off * nao * 3 * nao + ig * 3 * nao + x * nao + j] =
                                        out_raw[(x * qi_aux + p_off) + (i_off * nao + j) * (3 * qi_aux)];
                                }}
                            }
                        }
                    }
                }
                let mut rho0_qb = vec![0.0; qi_aux * nao * nao];
                for ia2 in alo..ahi {
                    let i0 = aoslices[ia2][2] as usize;
                    let i1 = aoslices[ia2][3] as usize;
                    let ni2 = i1 - i0;
                    if ni2 == 0 { continue; }
                    let cb = &coef_cache[ia2];
                    for p_off in 0..qi_aux { for i_off in 0..ni2 {
                        let ig = i0 + i_off;
                        for j in 0..nao {
                            rho0_qb[p_off * nao * nao + ig * nao + j] =
                                cb[(q0_aux + p_off) + (i_off * nao + j) * naux];
                        }
                    }}
                }
                let aoff = ia * 3 * nao3;
                // term1: vj1 -= 0.5·Σ rho0_qb·rhoj1  →  vj1_accum += 0.5·Σ
                for x in 0..3 { for i in 0..nao { for j in 0..nao {
                    let mut s = 0.0;
                    for p_off in 0..qi_aux {
                        s += rho0_qb[p_off * nao * nao + i * nao + j] * rhoj1[x * qi_aux + p_off];
                    }
                    vj1_accum[aoff + x * nao3 + i * nao + j] += 0.5 * s;
                }}}
                // term3: vj1 += 0.5·Σ temp·rho0_qb  →  vj1_accum -= 0.5·Σ
                {
                    let mut temp = vec![0.0; 3 * qi_aux];
                    for x in 0..3 { for p_off in 0..qi_aux {
                        let mut s = 0.0;
                        for q in 0..naux {
                            s += i21[x * naux * naux + (q0_aux + p_off) * naux + q] * rhoj0_P[q];
                        }
                        temp[x * qi_aux + p_off] = s;
                    }}
                    for x in 0..3 { for i in 0..nao { for j in 0..nao {
                        let mut s = 0.0;
                        for p_off in 0..qi_aux {
                            s += temp[x * qi_aux + p_off] * rho0_qb[p_off * nao * nao + i * nao + j];
                        }
                        vj1_accum[aoff + x * nao3 + i * nao + j] -= 0.5 * s;
                    }}}
                }
                // term4: vj1 += 0.5·Σ pij_all·rhoj0_P  →  vj1_accum -= 0.5·Σ
                for x in 0..3 { for i in 0..nao { for j in 0..nao {
                    let mut s = 0.0;
                    for p_off in 0..qi_aux {
                        s += pij_all[p_off * nao * 3 * nao + i * 3 * nao + x * nao + j]
                            * rhoj0_P[q0_aux + p_off];
                    }
                    vj1_accum[aoff + x * nao3 + i * nao + j] -= 0.5 * s;
                }}}
                // vk1 term2: vk1 += pij@plj_reord  →  vk1_accum +=
                if self.factor_k != 0.0 {
                    let q0 = q0_aux;
                    let mut rhok0_PlJ = vec![0.0; qi_aux * nao * nao];
                    {
                        let mut rkp_stage = vec![0.0; qi_aux * nao * nocc];
                        for p in 0..qi_aux { for l in 0..nao { for j in 0..nocc {
                            rkp_stage[(p * nao + l) + j * (qi_aux * nao)] =
                                rhok0_Pl_[(q0 + p) * nao * nocc + l * nocc + j];
                        }}}
                        let rkp_t = rt::asarray((&rkp_stage, [qi_aux * nao, nocc].f(), &device));
                        let mc2_t_aux = rt::asarray((mc2.as_slice(), [nocc, nao].f(), &device));
                        let plj_t = (&rkp_t % &mc2_t_aux);
                        let plj_raw = plj_t.into_shape(-1).into_raw();
                        for p in 0..qi_aux { for l in 0..nao { for j_idx in 0..nao {
                            rhok0_PlJ[p * nao * nao + l * nao + j_idx] =
                                plj_raw[(p * nao + l) + j_idx * (qi_aux * nao)];
                        }}}
                    }
                    let mut plj_reord_aux = vec![0.0; nao * qi_aux * nao];
                    for j in 0..nao { for p in 0..qi_aux { for l in 0..nao {
                        plj_reord_aux[(j * qi_aux + p) + l * (nao * qi_aux)] =
                            rhok0_PlJ[p * nao * nao + l * nao + j];
                    }}}
                    let plj_reord_t = rt::asarray((&plj_reord_aux, [nao * qi_aux, nao].f(), &device));
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
                            vk1_accum[aoff + x * nao3 + i * nao + l] += vk1_corr_raw[i + l * nao];
                        }}
                    }
                }
                drop(ip2_a); drop(pij_all); drop(rho0_qb);
            }
        }

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
        // H2a/H2b build the exchange-side vk1_buf (K-only). For pure DFAs
        // factor_k == 0 so vk1 is scaled out of h1ao — skip all of it. The
        // consumed vars are hoisted so the (also gated) per-atom vk1 block
        // below can borrow them without recomputation.
        let mut vk1_buf = vec![0.0; 3 * nao3];
        let mut rk_pl_stage: Vec<f64> = Vec::new();
        if self.factor_k != 0.0 {
        let mut vk1_buf_new = vec![0.0; 3 * nao3];
        // Pre-stage rhok0_Pl_ once as F-order [naux*nao, nocc] — shared by H2a and H3a.
        // Previously H3a re-staged this (2.5M elements × natm = 30M useless copies for C6H6).
        let rk_pl_stage_new: Vec<f64> = {
            let mut stage = vec![0.0; naux * nao * nocc];
            for p in 0..naux {
                for l in 0..nao {
                    for j in 0..nocc {
                        stage[(p * nao + l) + j * (naux * nao)] =
                            rhok0_Pl_[p * nao * nocc + l * nocc + j];
                    }
                }
            }
            stage
        };

        // H2a/H2b: the 0.9 GB rhok0_PlJ (= rhok0_Pl_·mc2ᵀ) is NOT built in
        // full: each aux-atom block builds only its [nq, N, N] slice on the
        // fly from rhok0_Pl_ + mc2 (~50 MB) and contracts with ip1_block.

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
            if nq == 0 {
                continue;
            }
            if std::env::var("REST_MEM_TRACE").is_ok() && ia_aux == 0 {
                eprintln!("MEMTRACE h1ao-H2b-iter      RSS = {:.1} MiB", memory_monitor::current_rss_mb());
            }
            // Compute int3c2e_ip1 for this aux block: [3, nao, nao, nq]
            let aux_block_slc: &[[usize; 2]] =
                &[[0, nreg], [0, nreg], [nreg + aux_shl0, nreg + aux_shl1]];
            let (ip1_block, _): (Vec<f64>, Vec<usize>) = cint_all
                .integrate_row_major("int3c2e_ip1", "s1", Some(aux_block_slc))
                .into();
            // ip1_block layout: [3, nao, nao, nq] row-major
            //   element (x, i, j, p_local) at x*nao*nao*nq + i*nao*nq + j*nq + p_local

            // plj_block_stage [N·nq, N] F-order: rows (j, p_loc), cols l
            // = Σ_occ rhok0_Pl_[(ap0+p_loc), l, occ]·mc2[j, occ] — built on the
            // fly (no 0.9 GB rhok0_PlJ); SIMD over j (contiguous).
            let mut plj_block_stage = vec![0.0; nao * nq * nao];
            for p_loc in 0..nq {
                for l in 0..nao {
                    let rbase = (ap0 + p_loc) * nao * nocc + l * nocc;
                    let rrow = &rhok0_Pl_[rbase..rbase + nocc];
                    for j in 0..nao {
                        let mut s = 0.0;
                        let mbase = j * nocc;
                        for occ in 0..nocc { s += rrow[occ] * mc2[mbase + occ]; }
                        plj_block_stage[(j * nq + p_loc) + l * (nao * nq)] = s;
                    }
                }
            }
            let plj_block_t = rt::asarray((&plj_block_stage, [nao * nq, nao].f(), &device));

            for x in 0..3 {
                // Stage ip1_block[x] as F-order [nao, nao*nq]: cols (j, p_local)
                let mut ip1_block_stage = vec![0.0; nao * nao * nq];
                for i in 0..nao {
                    for j in 0..nao {
                        for p_loc in 0..nq {
                            ip1_block_stage[i + (j * nq + p_loc) * nao] =
                                ip1_block[x * nao * nao * nq + i * nao * nq + j * nq + p_loc];
                        }
                    }
                }
                let ip1_block_t = rt::asarray((&ip1_block_stage, [nao, nao * nq].f(), &device));
                let vk1_block_t = (&ip1_block_t % &plj_block_t); // [nao, nao] F-order
                let vk1_block_raw = vk1_block_t.into_shape(-1).into_raw();
                // Accumulate to vk1_buf row-major [3, nao, nao]
                for i in 0..nao {
                    for l in 0..nao {
                        vk1_buf_new[x * nao3 + i * nao + l] += vk1_block_raw[i + l * nao];
                    }
                }
            }
        }
        vk1_buf = vk1_buf_new;
        rk_pl_stage = rk_pl_stage_new;
        }

        // Save intermediates for debug comparison (reindex to MatrixFull column-major)
        // rhoj0_P: store as [naux, 1]
        if keep_debug {
            self.result.insert(
                "rhoj0_P".to_string(),
                MatrixFull::from_vec([naux, 1], rhoj0_P.clone()).unwrap(),
            );
        }
        // vk1_buf: store as [3*nao, nao]
        {
            let mut vk1b_mf = vec![0.0; 3 * nao3];
            for x in 0..3 {
                for i in 0..nao {
                    for j in 0..nao {
                        let src = x * nao3 + i * nao + j;
                        let dst = (x * nao + i) + j * 3 * nao;
                        vk1b_mf[dst] = vk1_buf[src];
                    }
                }
            }
            if keep_debug {
                self.result.insert(
                    "vk1_buf".to_string(),
                    MatrixFull::from_vec([3 * nao, nao], vk1b_mf).unwrap(),
                );
            }
        }
        // vj1_buf[ia]: per-atom raw buffer before any correction
        for ia in 0..natm {
            let off = ia * 3 * nao3;
            let mut vj1b_mf = vec![0.0; 3 * nao3];
            for x in 0..3 {
                for i in 0..nao {
                    for j in 0..nao {
                        let src = x * nao3 + i * nao + j;
                        let dst = (x * nao + i) + j * 3 * nao;
                        vj1b_mf[dst] = vj1_accum[off + src];
                    }
                }
            }
            if keep_debug {
                self.result.insert(
                    format!("vj1_buf_{}", ia),
                    MatrixFull::from_vec([3 * nao, nao], vj1b_mf).unwrap(),
                );
            }
        }

        // Per-atom: h1ao = hcore_deriv + vj1 - 0.5*vk1 + vxc_deriv1
        let _t_vxc_d1 = std::time::Instant::now();
        let vxc_d1: Option<Vec<Vec<f64>>> = if self.is_rks() {
            Some(crate::hessian::rks::compute_vxc_h1ao(scf))
        } else {
            None
        };
        self.h1ao.clear();
        for ia in 0..natm {
            let shl0 = aoslices[ia][0] as usize;
            let shl1 = aoslices[ia][1] as usize;
            let p0 = aoslices[ia][2] as usize;
            let p1 = aoslices[ia][3] as usize;
            let ni = p1 - p0;
            let off = ia * 3 * nao3;

            // vj1
            let atom_slc: &[[usize; 2]] = &[[shl0, shl1], [0, nreg], [nreg, nreg + naux_shell]];
            let (ip1_a, _): (Vec<f64>, Vec<usize>) = cint_all
                .integrate_row_major("int3c2e_ip1", "s1", Some(atom_slc))
                .into();
            let mut vj1 = vec![0.0; 3 * nao3];
            for i in 0..3 * nao3 {
                vj1[i] = -vj1_accum[off + i];
            }
            // Save neg_buf (before any rhoj0 correction, for debug comparison)
            let vj1_neg_buf = vj1.clone();
            // vj1[:,p0:p1] correction (rows, matching PySCF: only the differentiated-AO side)
            for x in 0..3 {
                for ii in 0..ni {
                    for j in 0..nao {
                        let mut s = 0.0;
                        for P in 0..naux {
                            s += ip1_a[x * ni * nao * naux + ii * nao * naux + j * naux + P]
                                * rhoj0_P[P];
                        }
                        vj1[x * nao3 + (p0 + ii) * nao + j] -= s;
                    }
                }
            }
            // vj1_presym: after row rhoj0 correction, before symmetrization (= PySCF's _gen_jk vj1)
            let vj1_presym = vj1.clone();
            // Save sub-components for debug comparison
            {
                let mut nbuf = vec![0.0; 3 * nao3];
                let mut pres = vec![0.0; 3 * nao3];
                for x in 0..3 {
                    for i in 0..nao {
                        for j in 0..nao {
                            let src = x * nao3 + i * nao + j;
                            let dst = (x * nao + i) + j * 3 * nao;
                            nbuf[dst] = vj1_neg_buf[src];
                            pres[dst] = vj1_presym[src];
                        }
                    }
                }
                if keep_debug {
                    self.result.insert(
                        format!("vj1_neg_buf_{}", ia),
                        MatrixFull::from_vec([3 * nao, nao], nbuf).unwrap(),
                    );
                    self.result.insert(
                        format!("vj1_presym_{}", ia),
                        MatrixFull::from_vec([3 * nao, nao], pres).unwrap(),
                    );
                }
            }
            // ── Auxiliary-basis response corrections (before symmetrization) ──
            let mut ip2_a: Vec<f64> = Vec::new();
            let mut qi_aux: usize = 0;
            let mut q0_aux: usize = 0;

            let aux_shl0 = auxslices[ia][0] as usize;
            let aux_shl1 = auxslices[ia][1] as usize;
            q0_aux = auxslices[ia][2] as usize;
            let q1 = auxslices[ia][3] as usize;
            qi_aux = q1 - q0_aux;
            // int3c2e_ip2 for this aux atom: (ij|∂P/∂x) — used by term2 and
            // vk1 term1 (both batch-independent); coef-dependent terms 1/3/4
            // were accumulated into vj1_accum/vk1_accum in the batch loop.
            if qi_aux > 0 {
                let ip2_slc: &[[usize; 2]] =
                    &[[0, nreg], [0, nreg], [nreg + aux_shl0, nreg + aux_shl1]];
                let ip2_result: (Vec<f64>, Vec<usize>) = cint_all
                    .integrate_row_major("int3c2e_ip2", "s1", Some(ip2_slc))
                    .into();
                ip2_a = ip2_result.0;
            }
            // vj1 correction term 2: += -0.5 * Σ_P ip2[x,i,j,P] * rhoj0_P[q0_aux+P]
            for x in 0..3 {
                for i in 0..nao {
                    for j in 0..nao {
                        let mut s = 0.0;
                        for P in 0..qi_aux {
                            s += ip2_a[x * nao * nao * qi_aux + i * nao * qi_aux + j * qi_aux + P]
                                * rhoj0_P[q0_aux + P];
                        }
                        vj1[x * nao3 + i * nao + j] -= 0.5 * s;
                    }
                }
            }
            // Save vj1_aux for debug comparison
            {
                let mut aux_mat = vec![0.0; 3 * nao3];
                for x in 0..3 {
                    for i in 0..nao {
                        for j in 0..nao {
                            let src = x * nao3 + i * nao + j;
                            let dst = (x * nao + i) + j * 3 * nao;
                            aux_mat[dst] = vj1[src];
                        }
                    }
                }
                if keep_debug {
                    self.result.insert(
                        format!("vj1_aux_{}", ia),
                        MatrixFull::from_vec([3 * nao, nao], aux_mat).unwrap(),
                    );
                }
            }

            for x in 0..3 {
                for i in 0..nao {
                    for j in (i + 1)..nao {
                        let a = x * nao3 + i * nao + j;
                        let b = x * nao3 + j * nao + i;
                        let sym = vj1[a] + vj1[b];
                        vj1[a] = sym;
                        vj1[b] = sym;
                    }
                }
                for i in 0..nao {
                    vj1[x * nao3 + i * nao + i] *= 2.0;
                }
            }

            // vk1 (K-only): skipped entirely for pure DFAs (factor_k == 0),
            // in which case vk1 stays zero and 0.5*factor_k*vk1 == 0 in h1ao.
            let mut vk1 = vec![0.0; 3 * nao3];
            let mut vk1_step1 = vk1.clone();
            let mut vk1_presym = vk1.clone();
            if self.factor_k != 0.0 {
            // H3a: rhok0_PlJ_a[P,l,J] = Σ_j rhok0_Pl_[P,l,j] * mc2[(p0+J),j]
            //   GEMM: rhok0_Pl_ [naux*nao, nocc] @ mc2_slice^T [nocc, ni] = rhok0_PlJ_a [naux*nao, ni]
            //   (rhok0_Pl_ staging same as H2a, mc2_slice needs staging for p0..p0+ni rows)
            // H3b: vk1[x,k,jj] = -Σ_{P,ii} ip1_a[x,ii,jj,P] * rhok0_PlJ_a[P,k,ii]
            //   Per x: GEMM ip1_a[x] [nao, ni*naux] @ rhok0_PlJ_a_reord [ni*naux, nao] = vk1[x] [nao, nao]
            let mut rhok0_PlJ_a = vec![0.0; naux * nao * ni];

            use rstsr::prelude::*;
            // ── H3a ── (reuses pre-staged rk_pl_stage)
            let rk_pl_t = rt::asarray((&rk_pl_stage, [naux * nao, nocc].f(), &device));
            // Stage mc2_slice (rows p0..p0+ni) as F-order [nocc, ni]: element (j, J) = mc2[(p0+J)*nocc + j]
            let mut mc2_slice_stage = vec![0.0; nocc * ni];
            for j in 0..nocc {
                for jj in 0..ni {
                    mc2_slice_stage[j + jj * nocc] = mc2[(p0 + jj) * nocc + j];
                }
            }
            let mc2_slice_t = rt::asarray((&mc2_slice_stage, [nocc, ni].f(), &device));
            let plj_a_t = (&rk_pl_t % &mc2_slice_t); // [naux*nao, ni] F-order
            let plj_a_raw = plj_a_t.into_shape(-1).into_raw();
            // Scatter to row-major [naux, nao, ni]
            for p in 0..naux {
                for l in 0..nao {
                    for jj in 0..ni {
                        rhok0_PlJ_a[p * nao * ni + l * ni + jj] =
                            plj_a_raw[(p * nao + l) + jj * (naux * nao)];
                    }
                }
            }

            // ── H3b ──
            // Stage rhok0_PlJ_a as F-order [ni*naux, nao] with rows (ii,P), cols k:
            //   element (ii*naux+P, k) = rhok0_PlJ_a[P, k, ii] = rhok0_PlJ_a[P*nao*ni + k*ni + ii]
            let mut plj_a_reord = vec![0.0; ni * naux * nao];
            for ii in 0..ni {
                for p in 0..naux {
                    for k in 0..nao {
                        plj_a_reord[(ii * naux + p) + k * (ni * naux)] =
                            rhok0_PlJ_a[p * nao * ni + k * ni + ii];
                    }
                }
            }
            let plj_a_reord_t = rt::asarray((&plj_a_reord, [ni * naux, nao].f(), &device));
            for x in 0..3 {
                // Stage ip1_a[x] as F-order [nao, ni*naux] with cols (ii,P):
                //   element (jj, ii*naux+P) = ip1_a[x,ii,jj,P] = ip1_a[x*ni*nao*naux + ii*nao*naux + jj*naux + P]
                let mut ip1a_stage = vec![0.0; nao * ni * naux];
                for jj in 0..nao {
                    for ii in 0..ni {
                        for p in 0..naux {
                            ip1a_stage[jj + (ii * naux + p) * nao] =
                                ip1_a[x * ni * nao * naux + ii * nao * naux + jj * naux + p];
                        }
                    }
                }
                let ip1a_t = rt::asarray((&ip1a_stage, [nao, ni * naux].f(), &device));
                let vk1_t = (&ip1a_t % &plj_a_reord_t); // [nao, nao] F-order
                let vk1_raw = vk1_t.into_shape(-1).into_raw();
                // Scatter (with negation) to vk1 row-major [3, nao, nao]: (x,k,jj) at x*nao² + k*nao + jj
                for k in 0..nao {
                    for jj in 0..nao {
                        vk1[x * nao3 + k * nao + jj] = -vk1_raw[k + jj * nao];
                    }
                }
            }

            vk1_step1 = vk1.clone(); // before vk1_buf correction
                                         // vk1_buf correction (on rows, matching vk1_buf's layout)
            for x in 0..3 {
                for i in p0..p1 {
                    for j in 0..nao {
                        vk1[x * nao3 + i * nao + j] -= vk1_buf[x * nao3 + i * nao + j];
                    }
                }
            }
            // Save pre-sym vk1 for debug comparison with PySCF _gen_jk
            vk1_presym = vk1.clone();
            // ── Auxiliary-basis response corrections (before vk1 symmetrization) ──
            if qi_aux > 0 {
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
                    for p in 0..qi_aux {
                        for l in 0..nao {
                            for j in 0..nocc {
                                rkp_stage[(p * nao + l) + j * (qi_aux * nao)] =
                                    rhok0_Pl_[(q0 + p) * nao * nocc + l * nocc + j];
                            }
                        }
                    }
                    let rkp_t = rt::asarray((&rkp_stage, [qi_aux * nao, nocc].f(), &device));
                    let mc2_t_aux = rt::asarray((mc2.as_slice(), [nocc, nao].f(), &device));
                    let plj_t = (&rkp_t % &mc2_t_aux); // [qi_aux*nao, nao] F-order
                    let plj_raw = plj_t.into_shape(-1).into_raw();
                    for p in 0..qi_aux {
                        for l in 0..nao {
                            for j_idx in 0..nao {
                                rhok0_PlJ[p * nao * nao + l * nao + j_idx] =
                                    plj_raw[(p * nao + l) + j_idx * (qi_aux * nao)];
                            }
                        }
                    }
                }
                // vk1 correction term 1: -= Σ_{P,j} rhok0_PlJ[P,l,j] * ip2[x,i,j,P]
                // GEMM per x: ip2_a[x] [nao, nao*qi_aux] @ rhok0_PlJ_reord [nao*qi_aux, nao] = [nao, nao]
                //   rhok0_PlJ_reord F-order [nao*qi_aux, nao]: (j*qi_aux+P, l) at (j*qi_aux+P) + l*(nao*qi_aux)
                //     source: rhok0_PlJ[P*nao² + l*nao + j]
                //   This staging is reused by term 2 (same rhok0_PlJ_reord).
                let mut plj_reord_aux = vec![0.0; nao * qi_aux * nao];
                for j in 0..nao {
                    for p in 0..qi_aux {
                        for l in 0..nao {
                            plj_reord_aux[(j * qi_aux + p) + l * (nao * qi_aux)] =
                                rhok0_PlJ[p * nao * nao + l * nao + j];
                        }
                    }
                }
                {
                    use rstsr::prelude::*;
                    let plj_reord_t =
                        rt::asarray((&plj_reord_aux, [nao * qi_aux, nao].f(), &device));
                    // Term 1: vk1 -= ip2 @ plj_reord
                    for x in 0..3 {
                        let mut ip2_stage = vec![0.0; nao * nao * qi_aux];
                        for i in 0..nao {
                            for j in 0..nao {
                                for p in 0..qi_aux {
                                    ip2_stage[i + (j * qi_aux + p) * nao] =
                                        ip2_a[x * nao * nao * qi_aux
                                            + i * nao * qi_aux
                                            + j * qi_aux
                                            + p];
                                }
                            }
                        }
                        let ip2_t = rt::asarray((&ip2_stage, [nao, nao * qi_aux].f(), &device));
                        let vk1_corr_t = (&ip2_t % &plj_reord_t);
                        let vk1_corr_raw = vk1_corr_t.into_shape(-1).into_raw();
                        for i in 0..nao {
                            for l in 0..nao {
                                vk1[x * nao3 + i * nao + l] -= vk1_corr_raw[i + l * nao];
                            }
                        }
                    }
                    // Term 2 (coef-dependent): vk1 += vk1_accum[ia]
                    // (accumulated across coef batches in the batch loop)
                    for x in 0..3 {
                        for i in 0..nao {
                            for l in 0..nao {
                                vk1[x * nao3 + i * nao + l] +=
                                    vk1_accum[ia * 3 * nao3 + x * nao3 + i * nao + l];
                            }
                        }
                    }
                }
                // Save vk1_aux for debug comparison
                {
                    let mut aux_mat = vec![0.0; 3 * nao3];
                    for x in 0..3 {
                        for i in 0..nao {
                            for j in 0..nao {
                                let src = x * nao3 + i * nao + j;
                                let dst = (x * nao + i) + j * 3 * nao;
                                aux_mat[dst] = vk1[src];
                            }
                        }
                    }
                    if keep_debug {
                        self.result.insert(
                            format!("vk1_aux_{}", ia),
                            MatrixFull::from_vec([3 * nao, nao], aux_mat).unwrap(),
                        );
                    }
                }
            }
            for x in 0..3 {
                for i in 0..nao {
                    for j in (i + 1)..nao {
                        let a = x * nao3 + i * nao + j;
                        let b = x * nao3 + j * nao + i;
                        let sym = vk1[a] + vk1[b];
                        vk1[a] = sym;
                        vk1[b] = sym;
                    }
                }
                for i in 0..nao {
                    vk1[x * nao3 + i * nao + i] *= 2.0;
                }
            }
            }

            // hcore_deriv
            let h1 = build_hcore_first_deriv(mol, ia);

            // h1ao = h1 + vj1 - 0.5*factor_k*vk1 plus debug intermediates
            // (factor_k defaults to 1.0 for RHF; RKS wrapper sets it to hyb.)
            // Reindex from [x][i][j] layout to MatrixFull column-major
            let mut h1ao_mat = vec![0.0; 3 * nao3];
            let mut vj1_mat = vec![0.0; 3 * nao3];
            let mut vk1_mat = vec![0.0; 3 * nao3];
            let mut vj1p_mat = vec![0.0; 3 * nao3];
            let mut vk1p_mat = vec![0.0; 3 * nao3];
            let mut vj1s1_mat = vec![0.0; 3 * nao3];
            let mut vk1s1_mat = vec![0.0; 3 * nao3];
            for x in 0..3 {
                for i in 0..nao {
                    for j in 0..nao {
                        let src = x * nao3 + i * nao + j;
                        let dst = (x * nao + i) + j * 3 * nao;
                        let vxc1_val = vxc_d1.as_ref().map_or(0.0, |v| v[ia][dst]);
                        h1ao_mat[dst] =
                            h1[src] + vj1[src] - 0.5 * self.factor_k * vk1[src] + vxc1_val;
                        vj1_mat[dst] = vj1[src]; // post-sym
                        vk1_mat[dst] = vk1[src]; // post-sym
                        vj1p_mat[dst] = vj1_presym[src]; // pre-sym (= after rhoj0 corr)
                        vk1p_mat[dst] = vk1_presym[src]; // pre-sym (= after vk1_buf corr)
                        vj1s1_mat[dst] = vj1_neg_buf[src]; // after -vj1_buf only (no rhoj0 corr)
                        vk1s1_mat[dst] = vk1_step1[src]; // after -ip1@rhok0 only
                    }
                }
            }
            self.h1ao
                .push(MatrixFull::from_vec([3 * nao, nao], h1ao_mat).unwrap());
            if keep_debug {
                self.result.insert(
                    format!("vj1_{}", ia),
                    MatrixFull::from_vec([3 * nao, nao], vj1_mat).unwrap(),
                );
                self.result.insert(
                    format!("vk1_{}", ia),
                    MatrixFull::from_vec([3 * nao, nao], vk1_mat).unwrap(),
                );
                self.result.insert(
                    format!("vj1_presym_{}", ia),
                    MatrixFull::from_vec([3 * nao, nao], vj1p_mat).unwrap(),
                );
                self.result.insert(
                    format!("vk1_presym_{}", ia),
                    MatrixFull::from_vec([3 * nao, nao], vk1p_mat).unwrap(),
                );
                self.result.insert(
                    format!("vj1_neg_buf_{}", ia),
                    MatrixFull::from_vec([3 * nao, nao], vj1s1_mat).unwrap(),
                );
                self.result.insert(
                    format!("vk1_step1_{}", ia),
                    MatrixFull::from_vec([3 * nao, nao], vk1s1_mat).unwrap(),
                );
            }
        }
        // Save int2c_ip1 for debug comparison
        {
            let mut i21_mf = vec![0.0; 3 * naux * naux];
            for x in 0..3 {
                for p in 0..naux {
                    for q in 0..naux {
                        let src = x * naux * naux + p * naux + q;
                        let dst = (x * naux + p) + q * 3 * naux;
                        i21_mf[dst] = i21[src];
                    }
                }
            }
            if keep_debug {
                self.result.insert(
                    "int2c_ip1".to_string(),
                    MatrixFull::from_vec([3 * naux, naux], i21_mf).unwrap(),
                );
            }
        }
        if self.is_rks() {
            self.timings
                .push(("  rks: vxc_deriv1", _t_vxc_d1.elapsed()));
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
        crate::hessian::memory_monitor::trim_to_os(0);
        if self.scf_data.mol.ctrl.print_level > 0 {
            println!("  >> Entering CP-HF contribution stage ...");
        }
        use crate::ri_cphf::{
            build_s1ao_deriv, transform_h1ao_ao2mo, transform_s1ao_ao2mo, CPHFSolverPySCF,
        };
        use crate::dft::response::gen_vind_opt_batched;
        use tensors::matrix_blas_lapack::_dgemm_full;

        let scf = self.scf_data;
        let mol = &scf.mol;
        let nao = mol.num_basis;
        let natm = mol.geom.nfree;
        let n3 = natm * 3;
        let aa9 = natm * natm * 9;

        if self.h1ao.len() != natm {
            panic!(
                "calc_cphf_contrib: call calc_h1ao() first (h1ao.len={})",
                self.h1ao.len()
            );
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
        for p in 0..nao {
            for i in 0..nocc {
            c_occ[[p, i]] = c_mo[[p, start_mo + i]];
            }
        }

        // Occupied energies
        let mut eps_occ = vec![0.0; nocc];
        for i in 0..nocc {
            eps_occ[i] = solver.mo_energy[start_mo + i];
        }

        let s1_zero = vec![0.0; nmo * nocc];

        // ── Step 1: Solve CP-HF per atom, per direction ──
        // RKS fxc cache preparation is extracted to rks.rs. HF passes None.
        let _t_cache = std::time::Instant::now();
        let fxc_cache: Option<crate::dft::response::FxcHessianCache> =
            crate::hessian::rks::prepare_fxc_cache(scf);
        let fxc_cache_ref = fxc_cache.as_ref();
        let _t_cache_elapsed = _t_cache.elapsed();
        let _t_solve = std::time::Instant::now();
        let _t_rhsbuild = std::time::Instant::now();
        let mut mo1_all: Vec<Vec<Vec<f64>>> = vec![vec![vec![]; 3]; natm];
        let mut s1ao_all: Vec<Vec<Vec<f64>>> = vec![vec![vec![]; 3]; natm];
        let mut s1_mo_all: Vec<Vec<Vec<f64>>> = vec![vec![vec![]; 3]; natm];
        // Cache h1_mo per atom (3 directions each). Avoids recomputing
        // transform_h1ao_ao2mo in the mo_e1 loop below.
        let mut h1_mo_all: Vec<Vec<Vec<f64>>> = vec![vec![vec![]; 3]; natm];

        println!(
            "  CP-HF: using batched Krylov solver (max_cycle={}, tol={:.1e})",
            CPHF_KRYLOV_MAX_CYCLE, CPHF_KRYLOV_TOL
        );

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
        // The occupied-occupied correction is z_oo = -0.5·s1_oo. Instead of one
        // gen_vind_opt call (with a full fxc grid sweep) per RHS, collect all
        // 3*natm OO blocks and evaluate them in ONE batched call below.
        let fo_size_oo = solver.ws.nfrozen * nocc;
        let mut rhs_base_all: Vec<Vec<f64>> = Vec::with_capacity(n_pert);
        let mut z_oo_batch: Vec<Vec<f64>> = Vec::with_capacity(n_pert);
        for ia in 0..natm {
            let h1_mo = transform_h1ao_ao2mo(&solver, &self.h1ao[ia]);
            let s1ao_ia = build_s1ao_deriv(mol, ia);
            let s1_mo_ia = transform_s1ao_ao2mo(&solver, &s1ao_ia);
            for dir in 0..3 {
                h1_mo_all[ia][dir] = h1_mo[dir].clone();
            }

            for dir in 0..3 {
                s1ao_all[ia][dir] = s1ao_ia[dir].clone();
                s1_mo_all[ia][dir] = s1_mo_ia[dir].clone();
                // VO RHS without the OO correction.
                let b_base = solver.build_rhs_with_s1(&h1_mo[dir], &s1_mo_ia[dir]);
                // OO correction block: z_oo[i,j] = -0.5·s1_oo[i,j].
                let mut z_oo = vec![0.0; nocc * nocc];
                for j in 0..nocc {
                    for i in 0..nocc {
                        z_oo[i + j * nocc] =
                            -0.5 * s1_mo_ia[dir][(start_mo + i) + j * nmo];
                    }
                }
                rhs_base_all.push(b_base);
                z_oo_batch.push(z_oo);
                rhs_meta.push((ia, dir));
            }
        }
        // One batched fvind call for all 3*natm OO corrections (shares a single
        // fxc grid sweep across RHS instead of 3*natm separate ones).
        let zero_vo = vec![0.0; solver.dim];
        let z_vo_refs: Vec<&[f64]> = (0..n_pert).map(|_| zero_vo.as_slice()).collect();
        let z_oo_refs: Vec<&[f64]> = z_oo_batch.iter().map(|v| v.as_slice()).collect();
        if std::env::var("REST_CPHF_PROFILE").is_ok() {
            eprintln!("CPHF-PROF rhs-build-p1 {:.3}s", _t_rhsbuild.elapsed().as_secs_f64());
        }
        let _t_oob = std::time::Instant::now();
        let oo_resp_batch = gen_vind_opt_batched(
            scf,
            &solver.ws,
            &z_vo_refs,
            fxc_cache_ref,
            Some(&z_oo_refs),
            None,
            None, // OO path keeps the dense K (z_oo is present)
        );
        if std::env::var("REST_CPHF_PROFILE").is_ok() {
            eprintln!("CPHF-PROF rhs-build-oo {:.3}s", _t_oob.elapsed().as_secs_f64());
        }
        for (k, _) in rhs_meta.iter().enumerate() {
            // g_oo[ia] = VO response of the OO perturbation at (ia_row, ia_col).
            let resp = &oo_resp_batch[k];
            let mut rhs = rhs_base_all[k].clone();
            for ia in 0..solver.dim {
                rhs[ia] -= resp[fo_size_oo + ia] * solver.e_ai[ia];
            }
            rhs_all.push(rhs);
        }

        // ── Second pass: one batched Krylov solve for all 3*natom RHS ──
            let u_vo_all = solver.solve_krylov_batched(
            scf,
            fxc_cache_ref,
            &rhs_all,
            CPHF_KRYLOV_MAX_CYCLE,
            CPHF_KRYLOV_TOL,
        );

        // ── Third pass: assemble each full solution directly into mo1_all
        // (no intermediate mo1_full_all copy), plus preserve debug output ──
        for (k, &(ia, dir)) in rhs_meta.iter().enumerate() {
            let u_oo = solver.solve_occ_occ_from_s1(&s1_mo_all[ia][dir]);
            let mo1_full = solver.assemble_full_solution(&u_vo_all[k], &u_oo);
            mo1_all[ia][dir] = mo1_full.clone();

            // Save h1_mo, s1_mo, mo1 for atom 0, direction 0 for debug comparison.
            if ia == 0 && dir == 0 {
                let nmo = solver.nmo;
                let nocc = solver.nocc;
                self.result.insert(
                    "h1_mo_00".to_string(),
                    MatrixFull::from_vec([nmo * nocc, 1], h1_mo_all[ia][dir].clone()).unwrap(),
                );
                self.result.insert(
                    "s1_mo_00".to_string(),
                    MatrixFull::from_vec([nmo * nocc, 1], s1_mo_all[ia][dir].clone()).unwrap(),
                );
                {
                    let s1ao_flat: Vec<f64> =
                        (0..nao * nao).map(|i| s1ao_all[ia][dir][i]).collect();
                    let mut mx = 0.0;
                    for &v in &s1ao_flat {
                        let a = v.abs();
                        if a > mx {
                            mx = a;
                        }
                    }
                    if self.scf_data.mol.ctrl.print_level > 1 {
                        println!("  DEBUG s1ao[0] max_abs={:.4e}", mx);
                    }
                    self.result.insert(
                        "s1ao_00".to_string(),
                        MatrixFull::from_vec([nao, nao], s1ao_flat).unwrap(),
                    );
                }
                self.result.insert(
                    "mo1_full_00".to_string(),
                    MatrixFull::from_vec([nmo * nocc, 1], mo1_full.clone()).unwrap(),
                );
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
                self.result.insert(
                    "rhs_full_00".to_string(),
                    MatrixFull::from_vec([nmo * nocc, 1], rhs_full).unwrap(),
                );
            }
        }

        // RKS fxc solve-phase timings extracted to rks.rs. Note: the fxc timing
        // counter must be read BEFORE the mo_e1 phase resets it, so this call
        // happens immediately after the solve phase ends.
        crate::hessian::rks::record_fxc_solve_timings(&mut self.timings, _t_cache_elapsed);
        self.timings.push(("  cphf: solve", _t_solve.elapsed()));

        // Also compute mo_e1 (first-order orbital energy correction) for each atom/direction
        // mo_e1 = (h1 - s1*e_i + fvind(mo1))_oo + mo1_oo*(e_i - e_i)
        // Reset the fxc timing counter before the mo_e1 phase so the fxc work
        // done inside this loop (via add_fxc_to_v_ao) is attributed to fxc_moe1.
        crate::dft::response::reset_fxc_timing();
        let _t_mo_e1 = std::time::Instant::now();
        let mut mo_e1_all: Vec<Vec<Vec<f64>>> = vec![vec![vec![]; 3]; natm];
        {
            use crate::dft::response::{compute_j_upper, compute_k_upper, VindWorkspace};
            use tensors::matrix_blas_lapack::_dgemm_full;

            let ws2 = VindWorkspace::new(scf, nocc, solver.nvir, start_mo, solver.lumo);
            // RKS: hybrid exchange scaling (0.5*hyb for closed-shell). For HF
            // (no DFA components), hyb=1.0 so k_scaling=0.5, recovering the
            // historical J - 0.5*K response.
            let is_hf = scf.mol.xc_data.dfa_compnt_scf.is_empty();
            let hyb_e1 = if is_hf {
                1.0
            } else {
                scf.mol.xc_data.dfa_hybrid_scf
            };
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
                    for col in 0..nocc {
                        for row in 0..nmo {
                        x[[row, col]] = 2.0 * mo1_full[row + col * nmo];  // *2 for RHF
                        }
                    }
                    // dm = C_full @ (2*x) @ C_occ^T
                    let mut t1 = MatrixFull::new([nao, nocc], 0.0);
                    _dgemm_full(&solver.c_mo, 'N', &x, 'N', &mut t1, 1.0, 0.0);
                    let mut dm = MatrixFull::new([nao, nao], 0.0);
                    _dgemm_full(&t1, 'N', &c_occ, 'T', &mut dm, 1.0, 0.0);
                    // dm1 = dm + dm^T (proper symmetric)
                    let mut dm1 = MatrixFull::new([nao, nao], 0.0);
                    for i in 0..nao {
                        for j in 0..nao {
                            dm1[[i, j]] = dm[[i, j]] + dm[[j, i]];
                        }
                    }

                    // Compute J and K
                    let dm_vec = vec![dm1.clone()];
                    let j_full = compute_j_upper(scf, &dm_vec).to_matrixfull().unwrap();
                    // Skip the exchange response for pure (LDA/GGA) DFAs where
                    // k_scaling_e1 == 0 — K is O(naux·nao³) pure waste here.
                    let k_full = if k_scaling_e1 != 0.0 {
                        compute_k_upper(scf, &dm_vec).to_matrixfull().unwrap()
                    } else {
                        MatrixFull::new([nao, nao], 0.0)
                    };
                    let mut v_ao = MatrixFull::new([nao, nao], 0.0);
                    for p in 0..nao {
                        for q in 0..nao {
                        v_ao[[p, q]] = j_full[[p, q]] - k_scaling_e1 * k_full[[p, q]];
                        }
                    }
                    // Add fxc response for RKS (matches PySCF `gen_rks_response`
                    // with `singlet=None`: vind(dm1) = J - 0.5*hyb*K + nr_rks_fxc(dm1)).
                    // Uses the precomputed `FxcHessianCache` — no AO/ρ₀/libxc
                    // re-evaluation here. The in-place addition is extracted to
                    // rks.rs (writes into the loop-local v_ao buffer, zero alloc).
                    if let Some(cache) = fxc_cache_ref {
                        crate::hessian::rks::add_fxc_to_v_ao(cache, &dm1, &mut v_ao);
                    }

                    // Project to OO: C_occ^T @ v_ao @ C_occ
                    let mut tmp_oo = MatrixFull::new([nao, nocc], 0.0);
                    _dgemm_full(&v_ao, 'N', &c_occ, 'N', &mut tmp_oo, 1.0, 0.0);
                    let mut fvind_oo = MatrixFull::new([nocc, nocc], 0.0);
                    _dgemm_full(&c_occ, 'T', &tmp_oo, 'N', &mut fvind_oo, 1.0, 0.0);

                    // mo_e1[i,j] = (h1 - s1*e_i)[occ,occ] + fvind_oo + mo1_oo * (e_i[:,None] - e_i)
                    let mut mo_e1 = vec![0.0; nocc * nocc];
                    for j in 0..nocc {
                        for i in 0..nocc {
                        let e_j = eps_occ[j];
                        let e_i_i = eps_occ[i];
                        let h1s1 = h1_mo[dir][(start_mo + i) + j * nmo]
                                  - s1_mo_ia[dir][(start_mo + i) + j * nmo] * e_j;
                        let f_oo = fvind_oo[[i, j]];
                        let mo1_oo_ij = mo1_full[(start_mo + i) + j * nmo];
                        mo_e1[i + j * nocc] = h1s1 + f_oo + mo1_oo_ij * (e_i_i - e_j);
                        }
                    }
                    mo_e1_all[ia][dir] = mo_e1;
                }
            }
        }

        self.timings.push(("  cphf: mo_e1", _t_mo_e1.elapsed()));
        // RKS fxc mo_e1-phase timing extracted to rks.rs.
        crate::hessian::rks::record_fxc_moe1_timings(&mut self.timings);

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
                for i in 0..nao {
                    for j in 0..nao {
                        s1_mat[[i, j]] = s1_flat[i * nao + j];
                    }
                }
                let mut tmp = MatrixFull::new([nao, nocc], 0.0);
                _dgemm_full(&s1_mat, 'N', &c_occ, 'N', &mut tmp, 1.0, 0.0);
                let mut s1oo = MatrixFull::new([nocc, nocc], 0.0);
                _dgemm_full(&c_occ, 'T', &tmp, 'N', &mut s1oo, 1.0, 0.0);
                for i in 0..nocc {
                    for j in 0..nocc {
                        s1oo_all[ia][dx][i + j * nocc] = s1oo[[i, j]];
                    }
                }
            }
        }

        // Precompute energy-weighted C_occ: c_occ_e[p,i] = c_occ[p,i] * eps_occ[i].
        // Then dm1_e = mo1_ao @ c_occ_e^T (single GEMM, no per-element scaling).
        let mut c_occ_e = MatrixFull::new([nao, nocc], 0.0);
        for i in 0..nocc {
            let e = eps_occ[i];
            for p in 0..nao {
                c_occ_e[[p, i]] = c_occ[[p, i]] * e;
            }
        }
        // Slice metadata for borrowing mo1_full as a MatrixFullSlice without cloning.
        let mo1_sz = [nmo, nocc];
        let mo1_ind = [1usize, nmo];

        // ── Hoist the (ja, dir_y)-dependent back-transform ──
        // mo1_ao/dm1/dm1_e depend only on ja and dir_y, but the old code
        // recomputed them inside the (i0, j0) loop for every pair (each ja
        // served natm-ja i0 values). Precompute once per (ja, dir_y) and
        // reuse: 3·natm·(natm+1)/2 GEMMs → 3·natm GEMMs per matrix.
        let mut dm1_all: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); 3]; natm];
        let mut dm1_e_all: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); 3]; natm];
        {
            let mut mo1_ao_mat = MatrixFull::new([nao, nocc], 0.0);
            let mut dm1_mat = MatrixFull::new([nao, nao], 0.0);
            let mut dm1_e_mat = MatrixFull::new([nao, nao], 0.0);
            for ja in 0..natm {
                for dir_y in 0..3 {
                    // Back-transform mo1[ja][dir_y]: MO → AO via BLAS dgemm.
                    // mo1_ao[nao, nocc] = c_mo[nao, nmo] @ mo1_full[nmo, nocc]
                    let mo1_full = &mo1_all[ja][dir_y];
                    let mo1_full_slice = tensors::matrix::matrixfullslice::MatrixFullSlice {
                        size: &mo1_sz,
                        indicing: &mo1_ind,
                        data: &mo1_full[..],
                    };
                    _dgemm_full(c_mo, 'N', &mo1_full_slice, 'N', &mut mo1_ao_mat, 1.0, 0.0);

                    // Save mo1_ao for atom 0, direction 0 for debug
                    if ja == 0 && dir_y == 0 {
                        self.result
                            .insert("mo1_ao_00".to_string(), mo1_ao_mat.clone());
                    }

                    // dm1[p,q] = Σ_i mo1_ao[p,i] * c_occ[q,i] = mo1_ao @ c_occ^T
                    _dgemm_full(&mo1_ao_mat, 'N', &c_occ, 'T', &mut dm1_mat, 1.0, 0.0);
                    // dm1_e[p,q] = Σ_i mo1_ao[p,i] * c_occ[q,i] * eps[i] = mo1_ao @ c_occ_e^T
                    _dgemm_full(&mo1_ao_mat, 'N', &c_occ_e, 'T', &mut dm1_e_mat, 1.0, 0.0);
                    dm1_all[ja][dir_y] = dm1_mat.data.clone();
                    dm1_e_all[ja][dir_y] = dm1_e_mat.data.clone();
                }
            }
        }

        // Follow PySCF: compute de2[i0,j0] for j0 <= i0 only,
        // then fill de2[j0,i0] = de2[i0,j0].T by symmetry.
        for i0 in 0..natm {
            let ia = i0;
            for j0 in 0..=i0 {
                let ja = j0;
                let s1ao_ia = &s1ao_all[ia];

                for dir_y in 0..3 {
                    // dm1/dm1_e are column-major [nao, nao]: data[p + q*nao].
                    let dm1 = &dm1_all[ja][dir_y];
                    let dm1_e = &dm1_e_all[ja][dir_y];
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
                            for j in 0..nocc {
                                for i in 0..nocc {
                                s1e1 += s1oo_data[i + j * nocc] * mo_e1_data[i + j * nocc];
                                }
                            }
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
            for i0 in 0..natm {
                for j0 in 0..natm {
                    for x in 0..3 {
                        for y in 0..3 {
                m[(i0 * 3 + x) + (j0 * 3 + y) * n3] = arr[i_t(i0, j0, x, y)];
                        }
                    }
                }
            }
            for i0 in 0..natm {
                for j0 in 0..i0 {
                    for x in 0..3 {
                        for y in 0..3 {
                            m[(j0 * 3 + y) + (i0 * 3 + x) * n3] =
                                m[(i0 * 3 + x) + (j0 * 3 + y) * n3];
                        }
                    }
                }
            }
            MatrixFull::from_vec([n3, n3], m).unwrap()
        };
        self.timings
            .push(("  cphf: contract", _t_contract.elapsed()));
        self.timings.push(("calc_cphf_contrib", _t.elapsed()));
        self.result
            .insert("cphf_contrib".to_string(), to_mat(&cphf));
        self
    }

    /// Compute nuclear repulsion Hessian.
    /// Formula: ∂²/∂R_A∂R_B Σ_{I≠J} Z_I Z_J / |R_I - R_J|
    /// Result is stored as MatrixFull [n3, n3] (n3 = natm*3).
    pub fn calc_hess_nuc(&mut self) -> &mut Self {
        let _t = std::time::Instant::now();
        if self.scf_data.mol.ctrl.print_level > 0 {
            println!("  >> Entering hess_nuc stage ...");
        }
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
pub fn compute_frequencies_from_hessian(
    hess_total: &MatrixFull<f64>,
    scf: &SCF,
) -> Result<(Vec<f64>, MatrixFull<f64>), String> {
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
    let conv = HARTREE2WAVENUMBER / FQ.sqrt();  // = 5140.49
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

pub fn compute_frequencies(scf: &SCF) -> Result<(Vec<f64>, MatrixFull<f64>), String> {
    let h = compute_hessian(scf)?;
    compute_frequencies_from_hessian(&h, scf)
}

/// Main entry point for analytical Hessian / frequency calculations.
///
/// Always computes and saves the analytical Hessian. If
/// `HessianParameters::frequencies` is set, also computes and saves the
/// vibrational frequencies and eigenmodes (reusing the already-computed
/// Hessian matrix).
pub fn rhf_hessian_main(
    scf: &SCF,
    hess_ctrl: &crate::ctrl_io::hessian_parameters::HessianParameters,
    time_mark: &mut crate::utilities::TimeRecords,
) {
    let hess_total: MatrixFull<f64> = match run_hessian_pipeline(scf, hess_ctrl, time_mark) {
        Ok(m) => m,
        Err(e) => { eprintln!("Error in Hessian calculation: {}", e); return; }
    };
    // Compute frequencies when requested directly, or implicitly when
    // thermochemistry is requested (it needs the harmonic frequencies).
    let thermo_ctrl = scf.mol.ctrl.thermo.clone();
    let need_freq = hess_ctrl.frequencies || thermo_ctrl.is_some();
    let freqs_opt: Option<Vec<f64>> = if need_freq {
        if hess_ctrl.frequencies {
            run_frequencies_from(scf, hess_ctrl, &hess_total, time_mark)
        } else {
            // thermo requested without the frequencies flag: still need freqs
            match compute_frequencies_from_hessian(&hess_total, scf) {
                Ok((freqs, _)) => Some(freqs),
                Err(e) => { eprintln!("Error computing frequencies for thermochemistry: {}", e); None }
            }
        }
    } else { None };
    if let (Some(params), Some(freqs)) = (thermo_ctrl.as_ref(), freqs_opt.as_ref()) {
        crate::thermo::run_thermochemistry(scf, freqs, params, time_mark);
    }
}

/// Run the full analytical Hessian pipeline: calc_e1 → calc_ej_ek →
/// calc_h1ao → compute_hessian. Prints diagnostics, saves HessianMatrix.txt
/// and (when verbose) component .npy files. Returns the total Hessian matrix
/// (natm*3, natm*3) in column-major layout.
fn run_hessian_pipeline(
    scf: &SCF,
    hess_ctrl: &crate::ctrl_io::hessian_parameters::HessianParameters,
    time_mark: &mut crate::utilities::TimeRecords,
) -> Result<MatrixFull<f64>, String> {
    let pl = scf.mol.ctrl.print_level;
    let hess_start = std::time::Instant::now();
    if pl > 0 {
        println!("\n=== Analytical Hessian Calculation ===");
    }
    time_mark.new_item("Hessian", "analytical Hessian");
    time_mark.count_start("Hessian");

    // ── System-size report & memory monitor setup ───────────────
    let nao = scf.mol.num_basis;
    let nocc = (scf.homo[0] + 1) as usize;
    let natm = scf.mol.geom.nfree;
    let naux = scf.mol.make_auxmol_fake().num_basis;
    let ngrids = scf.grids.as_ref().map(|g| g.coordinates.len()).unwrap_or(0);
    if pl > 1 {
        memory_monitor::print_system_size("before Hessian pipeline", natm, nao, nocc, naux, ngrids);
    }
    let limit_gb = scf.mol.ctrl.max_memory;
    let monitor = MemMonitor::start(limit_gb, std::time::Duration::from_millis(20));
    if pl > 1 {
        println!(
            "  Memory monitor: limit = {}",
            limit_gb
                .map(|g| format!("{:.3} GiB (abort on exceed)", g))
                    .unwrap_or_else(|| "NONE (peak tracking only)".to_string())
        );
    }

    // Build full object to access components.
    let mut hess = RIRHFHessian::new(scf);
    let mut overall_peak_mb: f64 = 0.0_f64;
    let stage_report = |label: &str, monitor: &MemMonitor, overall: &mut f64, pl: usize| {
        let stage_peak = monitor.stage_peak_mb();
        if stage_peak > *overall {
            *overall = stage_peak;
        }
        if pl > 1 {
            println!(
                "  [mem] after {:<16}: stage peak RSS = {:8.3} MiB ({:.3} GiB) | overall peak = {:.3} MiB ({:.3} GiB)",
                label, stage_peak, stage_peak / 1024.0, *overall, *overall / 1024.0
            );
        }
    };
    hess.calc_e1();
    stage_report("calc_e1", &monitor, &mut overall_peak_mb, pl);
    memory_monitor::trim_to_os(pl);
            hess.calc_ej_ek();
            stage_report("calc_ej_ek", &monitor, &mut overall_peak_mb, pl);
    memory_monitor::trim_to_os(pl);
    hess.calc_h1ao();
    stage_report("calc_h1ao", &monitor, &mut overall_peak_mb, pl);
    memory_monitor::trim_to_os(pl);
    hess.compute_hessian();
    stage_report("compute_hessian", &monitor, &mut overall_peak_mb, pl);
    let hess_total = hess
        .result
        .get("hess_total")
        .cloned()
        .ok_or_else(|| "hess_total not found".to_string())?;

    let n3 = hess_total.size[0];
    let natm = n3 / 3;
    if pl > 1 {
        println!("  Hessian matrix [{}x{}]:", n3, n3);
        if hess_ctrl.verbose > 0 {
            for i in 0..n3.min(9) {
                print!("    row[{:2}]:", i);
                for j in 0..n3.min(9) {
                    print!(" {:10.4e}", hess_total[[i, j]]);
                }
                println!();
            }
        }
        // Also print individual components
        if let Some(hp) = hess.result.get("h_partial") {
            let mut hp_max = 0.0;
            for i in 0..n3 * n3 {
                let v = hp.data[i].abs();
                if v > hp_max {
                    hp_max = v;
                }
            }
            println!("  h_partial max_abs={:.4e}", hp_max);
        }
        if let Some(cc) = hess.result.get("cphf_contrib") {
            let mut cc_max = 0.0;
            for i in 0..n3 * n3 {
                let v = cc.data[i].abs();
                if v > cc_max {
                    cc_max = v;
                }
            }
            println!("  cphf_contrib max_abs={:.4e}", cc_max);
        }
        if let Some(hn) = hess.result.get("hess_nuc") {
            let mut hn_max = 0.0;
            for i in 0..n3 * n3 {
                let v = hn.data[i].abs();
                if v > hn_max {
                    hn_max = v;
                }
            }
            println!("  hess_nuc max_abs={:.4e}", hn_max);
        }
    }
    let mut max_abs = 0.0;
    for i in 0..n3 * n3 {
        let v = hess_total.data[i].abs();
        if v > max_abs {
            max_abs = v;
        }
    }
    if pl > 0 {
        println!("  Hessian total max_abs={:.4e}", max_abs);
    }

    // ── Save Hessian matrix to txt (always, path from ctrl) ──
    let mol_name = &scf.mol.geom.name;
    let mut out = String::new();
    out.push_str(&format!(
        "# {} Hessian: natm={} n3={} (Hartree/Bohr^2)\n",
        mol_name, natm, n3
    ));
    for i in 0..n3 {
        let mut row = String::new();
        for j in 0..n3 {
            row.push_str(&format!(" {:17.10e}", hess_total[[i, j]]));
        }
        out.push_str(row.trim_start());
        out.push('\n');
    }
    match std::fs::write(&hess_ctrl.hessian_matrix_path, out) {
        Ok(_) => {
            if pl > 0 {
                println!(
                    "  Hessian matrix saved to {}",
                    hess_ctrl.hessian_matrix_path
                );
            }
        }
        Err(e) => eprintln!(
            "  WARNING: failed to write Hessian matrix to {}: {}",
            hess_ctrl.hessian_matrix_path, e
        ),
    }

    // Save components as .npy for external comparison
    if hess_ctrl.verbose > 0 {
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
        for r in 0..nao {
            for c in 0..nao {
                dm0_c[r * nao + c] = dm0_mat[[r, c]];
            }
        }
        write_npy_f64(tmpdir.join("dm0.npy").to_str().unwrap(), &dm0_c);
        let nmo = scf_data.eigenvalues[0].len();
        let c = &scf_data.eigenvectors[0];
        let mut mc_c = vec![0.0; nao * nmo];
        for r in 0..nao {
            for c2 in 0..nmo {
                mc_c[r * nao + c2] = c[[r, c2]];
            }
        }
        write_npy_f64(tmpdir.join("mo_coeff.npy").to_str().unwrap(), &mc_c);
        write_npy_f64(
            tmpdir.join("mo_energy.npy").to_str().unwrap(),
            &scf_data.eigenvalues[0],
        );
        let mut mo_occ_flat = vec![0.0; nmo];
        for i in 0..nmo {
            mo_occ_flat[i] = scf_data.occupation[0][i] as f64;
        }
        write_npy_f64(tmpdir.join("mo_occ.npy").to_str().unwrap(), &mo_occ_flat);
        if pl > 1 {
            println!("  Saved components to {:?}", tmpdir);
        }
    }

    if pl > 1 {
        hess.print_timings();
    }
    // Final memory report
    let final_rss = memory_monitor::current_rss_mb();
    if pl > 1 {
        println!(
            "\n  [mem] Hessian pipeline finished: final RSS = {:.3} MiB ({:.3} GiB), overall peak observed = {:.3} MiB ({:.3} GiB)",
            final_rss, final_rss / 1024.0,
            overall_peak_mb, overall_peak_mb / 1024.0
        );
    } else if pl > 0 {
        println!(
            "  Hessian memory peak: {:.3} MiB ({:.3} GiB)",
            overall_peak_mb,
            overall_peak_mb / 1024.0
        );
    }
    monitor.stop();
    time_mark.count("Hessian");
    if pl > 0 {
        println!(
            "  Hessian elapsed: {:.3} s",
            hess_start.elapsed().as_secs_f64()
        );
    }

    Ok(hess_total)
}

/// Compute and print vibrational frequencies + normal modes from an
/// already-computed total Hessian matrix, and save EigenModes.txt.
/// Reuses `compute_frequencies_from_hessian` so the Hessian is NOT recomputed.
fn run_frequencies_from(
    scf: &SCF,
    hess_ctrl: &crate::ctrl_io::hessian_parameters::HessianParameters,
    hess_total: &MatrixFull<f64>,
    time_mark: &mut crate::utilities::TimeRecords,
) -> Option<Vec<f64>> {
    let pl = scf.mol.ctrl.print_level;
    if pl > 0 {
        println!("\n=== Vibrational Frequency Calculation ===");
    }
    time_mark.new_item("Frequencies", "vibrational frequencies");
    time_mark.count_start("Frequencies");
    let freqs_opt = match compute_frequencies_from_hessian(hess_total, scf) {
        Ok((freqs, modes)) => {
            let n3 = freqs.len();
            let natm = n3 / 3;
            let mol_name = &scf.mol.geom.name;
            let elems: Vec<String> = scf.mol.geom.elem.iter()
                .map(|e| crate::geom_io::formated_element_name(e)).collect();
            if pl > 0 {
                println!("\n  Vibrational Frequencies (cm):");
                println!("  {:>4}  {:>10}  {:>10}", "Mode", "Freq/cm", "Symmetry");
                println!("  {}  {}  {}", "----", "----------", "----------");
                for i in 0..n3 {
                    let sym = if freqs[i].abs() < 10.0 { "---" } else { "A" };
                    println!("  {:>4}  {:>10.2}  {:>10}", i + 1, freqs[i], sym);
                }
            }
            if pl > 1 {
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

            // ── Save eigenmodes to txt (always) ──
            let mut out = String::new();
            out.push_str(&format!("# EigenModes: {}, natm={}\n", mol_name, natm));
            out.push_str("# Frequencies (cm^-1):\n");
            for i in 0..n3 {
                out.push_str(&format!("  Mode {:>3}: {:12.4}\n", i + 1, freqs[i]));
            }
            for i in 0..n3 {
                out.push_str(&format!("\n# Mode {} ({:.4} cm^-1) displacements:\n", i + 1, freqs[i]));
                for ia in 0..natm {
                    let dx = modes[[ia * 3, i]];
                    let dy = modes[[ia * 3 + 1, i]];
                    let dz = modes[[ia * 3 + 2, i]];
                    out.push_str(&format!("  atom {:>3} ({}): {:14.6e} {:14.6e} {:14.6e}\n",
                        ia + 1, elems.get(ia).map(|s| s.as_str()).unwrap_or("?"), dx, dy, dz));
                }
            }
            match std::fs::write(&hess_ctrl.eigenmodes_path, out) {
                Ok(_) => { if pl > 0 {
                    println!("  Eigenmodes saved to {}", hess_ctrl.eigenmodes_path);
                }}
                Err(e) => eprintln!("  WARNING: failed to write eigenmodes to {}: {}", hess_ctrl.eigenmodes_path, e),
            }
            Some(freqs)
        },
        Err(e) => { eprintln!("Error in frequency calculation: {}", e); None }
    };
    time_mark.count("Frequencies");
    freqs_opt
}


// ── Minimal .npy writer ──
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
