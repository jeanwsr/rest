//! Grid-based XC Hessian routines for RKS: make_dR_rho1, vxc_diag,
//! vxc_deriv1, vxc_deriv2 (LDA + GGA). Direct port of PySCF's
//! hessian/rks.py (_get_vxc_diag, _get_vxc_deriv1, _get_vxc_deriv2).
//!
//! Conventions:
//!   REST's `eval_ao_batch` returns RIFull of shape [nbasis, ngrids, nderiv].
//!   We extract each derivative `d` via `ao.get_reducing_matrix(d)` which
//!   gives a MatrixFull<f64> of shape [nbasis, ngrids].
//!   `ao_dm0[d]` (also [nbasis, ngrids]) is computed as `dm0 · ao_d[d]`,
//!   which for symmetric dm0 equals the transpose of PySCF's ao_dm0.
//!   All per-grid dot-products below are over the basis axis.

use crate::dft::num_int::{eval_ao_batch, eval_rho5_batch};
use crate::dft::libxc_itrf::eval_xc_eff;
use crate::dft::xc_deriv::XCType;
use crate::molecule_io::Molecule;
use crate::scf_io::SCF;
use rest_tensors::MatrixFull;
use rest_tensors::matrix::matrix_blas_lapack::{_dgemm_full, omp_set_num_threads_wrapper};
use rayon::prelude::*;

// ── AO second-derivative index convention (matches PySCF) ──
// ao_deriv=2 yields 10 components:
//   0=ϕ, 1=∂ₓϕ, 2=∂ᵧϕ, 3=∂_zϕ,
//   4=∂ₓₓ (XX), 5=∂ₓᵧ (XY=YX), 6=∂ₓ_z (XZ=ZX),
//   7=∂ᵧᵧ (YY), 8=∂ᵧ_z (YZ=ZY), 9=∂_zz (ZZ)
const XX: usize = 4;
const XY: usize = 5;
const XZ: usize = 6;
const YY: usize = 7;
const YZ: usize = 8;
const ZZ: usize = 9;

/// Build per-atom AO slices using `Molecule::aoslice_by_atom()`.
/// Returns `Vec<(p0, p1)>` (only the AO range; shell range discarded).
pub fn build_aoslices(mol: &Molecule) -> Vec<(usize, usize)> {
    mol.aoslice_by_atom()
        .into_iter()
        .map(|s| (s[2], s[3]))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────
// VxcHessianCache: shared AO + ground-state ρ cache for vxc_diag /
// vxc_deriv1 / vxc_deriv2. Eliminates the redundant `eval_ao_batch` and
// `eval_rho5_batch` calls when more than one vxc_* function is invoked
// (H1 optimization).
// ─────────────────────────────────────────────────────────────────────────

/// Per-block cached data: AO derivative tensors (up to `nderiv_max`) and
/// ground-state density values on the grid block.
pub struct VxcHessianBlock {
    pub ao_d: Vec<MatrixFull<f64>>,  // [nderiv_max] each [nao, nb]
    pub rho_array: Vec<f64>,         // [nb * nvar]
    pub weights: Vec<f64>,           // [nb]
    pub nb: usize,
}

/// Shared cache for all three vxc_* Hessian routines. Built once over the
/// full grid. `nderiv_max` is the maximum derivative order required across
/// the consumers: GGA needs nderiv_max=20 (ao_deriv=3, for vxc_diag);
/// LDA needs nderiv_max=4 (ao_deriv=1, for vxc_diag).
pub struct VxcHessianCache {
    pub blocks: Vec<VxcHessianBlock>,
    pub xc_type: XCType,
    pub nderiv_max: usize,
    pub nvar: usize,
    pub nao: usize,
}

/// Build the shared VxcHessianCache. Computes AO derivatives up to the
/// maximum order needed (ao_deriv=3 for GGA, ao_deriv=2 for LDA) and the
/// ground-state ρ on each grid block. Use this when sharing across
/// `vxc_diag` + `vxc_deriv2` (both consume the same cache).
pub fn build_vxc_hessian_cache(scf: &SCF, xc_type: XCType) -> VxcHessianCache {
    let ao_deriv = match xc_type {
        XCType::LDA => 2,   // vxc_diag uses 2nd derivatives even for LDA
        XCType::GGA => 3,   // vxc_diag needs 3rd derivatives for GGA
        _ => panic!("build_vxc_hessian_cache: only LDA and GGA supported"),
    };
    build_vxc_hessian_cache_with_deriv(scf, xc_type, ao_deriv)
}

/// Build the shared VxcHessianCache with a specific AO derivative order.
/// Use a smaller `ao_deriv` (e.g. 2 for GGA) when only `vxc_deriv1` /
/// `vxc_deriv2` will consume the cache — avoids computing unused
/// higher-order AO derivatives.
pub fn build_vxc_hessian_cache_with_deriv(scf: &SCF, xc_type: XCType, ao_deriv: usize) -> VxcHessianCache {
    let mol = &scf.mol;
    let grids = scf.grids.as_ref().expect("build_vxc_hessian_cache requires scf.grids");
    let nao = mol.num_basis;
    let func_ids = &mol.xc_data.dfa_compnt_scf;
    let func_factors = &mol.xc_data.dfa_paramr_scf;
    let mo_vec = vec![scf.eigenvectors[0].clone()];
    let occ_vec = vec![scf.occupation[0].clone()];

    let nvar = match xc_type {
        XCType::LDA => 1,
        XCType::GGA => 4,
        _ => panic!("build_vxc_hessian_cache: only LDA and GGA supported"),
    };
    let nderiv_max = (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) / 6;

    let blocks: Vec<VxcHessianBlock> = grids.parallel_balancing
        .par_iter()
        .filter_map(|block_range| {
            // Pin OpenBLAS to 1 thread per rayon worker — eval_rho5_batch
            // below uses DGEMM, which would otherwise spawn a nested OpenMP
            // team that contends with rayon (see dft/mod.rs:3747).
            omp_set_num_threads_wrapper(1);
            let start = block_range.start;
            let end = block_range.end;
            let nb = end - start;
            if nb == 0 { return None; }
            let coords_block = &grids.coordinates[start..end];
            let weights_block = &grids.weights[start..end];

            let ao = eval_ao_batch(mol, coords_block, ao_deriv, nb);
            let ao_d: Vec<MatrixFull<f64>> = (0..nderiv_max)
                .map(|d| {
                    let view = ao.get_reducing_matrix(d).unwrap();
                    MatrixFull::from_vec([nao, nb], view.iter().copied().collect()).unwrap()
                })
                .collect();

            let rho_tensor = eval_rho5_batch(&ao, xc_type, &mo_vec, &occ_vec, 1, nb);
            let rho_array: Vec<f64> = {
                let raw = rho_tensor.raw();
                let off = rho_tensor.offset();
                raw[off..off + nb * nvar].to_vec()
            };

            // Silence unused-warning while keeping the kernel available for
            // callers that want to re-evaluate vxc/fxc on the cached ρ.
            let _ = (func_ids, func_factors);

            Some(VxcHessianBlock {
                ao_d,
                rho_array,
                weights: weights_block.to_vec(),
                nb,
            })
        })
        .collect();

    VxcHessianCache { blocks, xc_type, nderiv_max, nvar, nao }
}

/// Compute `∂ρ_ν / ∂R_{A,α}` via chain rule on AO derivatives.
///
/// Direct port of PySCF hessian/rks.py:_make_dR_rho1.
///
/// Output shape: `[3 (α), nvar (ν), ngrids]` stored as flat Vec in
/// column-major order: `rho1[α + ν*3 + g*3*nvar]`.
///
/// Inputs:
///   `ao_d[d]` for d in 0..nderiv: each is `&MatrixFull<f64>` of shape
///     `[nbasis, ngrids]` extracted via `ao.get_reducing_matrix(d)`.
///   `ao_dm0[d]` for d in 0..nvar: each is `MatrixFull<f64>` of shape
///     `[nbasis, ngrids]`, computed as `dm0 · ao_d[d]`.
///   `ia`: atom index.
///   `(p0, p1)`: AO slice for atom `ia`.
///   `xc_type`: LDA or GGA.
pub fn make_dR_rho1(
    ao_d: &[&MatrixFull<f64>],     // [nderiv] references
    ao_dm0: &[MatrixFull<f64>],    // [nvar] owned (nvar = 1 for LDA, 4 for GGA)
    ia_p0: usize,
    ia_p1: usize,
    xc_type: XCType,
    ngrids: usize,
) -> Vec<f64> {
    let nvar = match xc_type {
        XCType::LDA => 1,
        XCType::GGA => 4,
        _ => panic!("make_dR_rho1: only LDA and GGA supported"),
    };
    let ni = ia_p1 - ia_p0;
    let mut rho1 = vec![0.0; 3 * nvar * ngrids];

    // Helper indexing: rho1[α + ν*3 + g*3*nvar]
    // let r = |alpha: usize, nu: usize, g: usize| -> usize {
    //     alpha + nu * 3 + g * 3 * nvar
    // };

    // Common LDA+GGA term: rho1[:, 0] += einsum('xpi,pi->xp', ao[1:4], ao_dm0[0])
    // For each direction x (0,1,2): rho1[x, 0, g] = Σ_{μ∈atom} ao[x+1][μ,g] · ao_dm0[0][μ,g]
    for x in 0..3 {
        let ao_x = &ao_d[x + 1];          // [nbasis, ngrids]
        let dm0_x = &ao_dm0[0];            // [nbasis, ngrids]
        for g in 0..ngrids {
            let mut s = 0.0;
            for mu in ia_p0..ia_p1 {
                s += ao_x[[mu, g]] * dm0_x[[mu, g]];
            }
            rho1[x + 0 * 3 + g * 3 * nvar] += s;
        }
    }

    if nvar == 4 {
        // GGA-only terms: ∂(∇ρ)/∂R contributions.
        // rho1[0,1] += einsum('pi,pi->p', ao[XX], ao_dm0[0])  etc.
        // (xx contribution to rho1[α=0, ν=1])
        // Mapping (PySCF lines 307-315):
        //   rho1[0,1] += ao[XX]·ao_dm0[0];   rho1[0,2] += ao[XY]·ao_dm0[0];   rho1[0,3] += ao[XZ]·ao_dm0[0];
        //   rho1[1,1] += ao[YX]·ao_dm0[0];   rho1[1,2] += ao[YY]·ao_dm0[0];   rho1[1,3] += ao[YZ]·ao_dm0[0];
        //   rho1[2,1] += ao[ZX]·ao_dm0[0];   rho1[2,2] += ao[ZY]·ao_dm0[0];   rho1[2,3] += ao[ZZ]·ao_dm0[0];
        // With YX=XY=5, ZX=XZ=6, ZY=YZ=8:
        let second_derivs = [[XX, 0, 1], [XY, 0, 2], [XZ, 0, 3],
                              [XY, 1, 1], [YY, 1, 2], [YZ, 1, 3],
                              [XZ, 2, 1], [YZ, 2, 2], [ZZ, 2, 3]];
        for &[didx, alpha, nu] in &second_derivs {
            let ao_2d = &ao_d[didx];
            let dm0_0 = &ao_dm0[0];
            for g in 0..ngrids {
                let mut s = 0.0;
                for mu in ia_p0..ia_p1 {
                    s += ao_2d[[mu, g]] * dm0_0[[mu, g]];
                }
                rho1[alpha + nu * 3 + g * 3 * nvar] += s;
            }
        }

        // rho1[:,1] += einsum('xpi,pi->xp', ao[1:4], ao_dm0[1])
        // rho1[:,2] += ... ao_dm0[2]
        // rho1[:,3] += ... ao_dm0[3]
        for nu in 1..4 {
            let dm0_nu = &ao_dm0[nu];
            for x in 0..3 {
                let ao_x = &ao_d[x + 1];
                for g in 0..ngrids {
                    let mut s = 0.0;
                    for mu in ia_p0..ia_p1 {
                        s += ao_x[[mu, g]] * dm0_nu[[mu, g]];
                    }
                    rho1[x + nu * 3 + g * 3 * nvar] += s;
                }
            }
        }
    }

    // PySCF returns rho1 * 2 (for |μ> DM <∂_R ν| conjugate).
    for v in rho1.iter_mut() { *v *= 2.0; }

    rho1
}

/// Compute `ao_dm0[d] = dm0 · ao_d[d]` for d in 0..nvar.
/// dm0: [nbasis, nbasis] column-major symmetric. ao_d[d]: [nbasis, ngrids].
/// Result: [nbasis, ngrids] (matches REST layout; equals PySCF ao_dm0^T for symmetric dm0).
fn compute_ao_dm0(
    dm0: &MatrixFull<f64>,
    ao_d: &[&MatrixFull<f64>],
    nvar: usize,
    nbasis: usize,
    ngrids: usize,
) -> Vec<MatrixFull<f64>> {
    let mut out = Vec::with_capacity(nvar);
    for d in 0..nvar {
        let mut m = MatrixFull::new([nbasis, ngrids], 0.0);
        // dm0 · ao_d[d]:  [nbasis, nbasis] · [nbasis, ngrids] -> [nbasis, ngrids]
        _dgemm_full(dm0, 'N', ao_d[d], 'N', &mut m, 1.0, 0.0);
        out.push(m);
    }
    out
}

/// XC first-derivative term that feeds h1ao (PySCF: _get_vxc_deriv1).
///
/// Output: Vec of `natm` matrices, each `[3*nao, nao]` column-major matching
/// REST's h1ao layout (rows [α*nao..(α+1)*nao] for direction α).
///
/// Pseudocode (LDA):
///   Term 1 (atom-independent): v_ip[α] = Σ_g w·vxc · ∂ϕ_α · ϕ
///   Term 2 (per atom ia):       vmat[ia, α] = Σ_g w·fxc · (∂ρ_α · dm0) · ϕ
///   Final: vmat[ia, rows=p0:p1] += v_ip[rows=p0:p1]; vmat[ia] = -vmat - vmat^T
pub fn vxc_deriv1(scf: &SCF, xc_type: XCType) -> Vec<MatrixFull<f64>> {
    // vxc_deriv1 only needs up to 2nd derivatives for GGA (or 1st for LDA).
    let ao_deriv = match xc_type { XCType::LDA => 1, XCType::GGA => 2, _ => panic!() };
    let cache = build_vxc_hessian_cache_with_deriv(scf, xc_type, ao_deriv);
    vxc_deriv1_cached(scf, xc_type, &cache)
}

/// Cached variant of `vxc_deriv1`: reuses AO derivatives and ground-state ρ
/// from `cache`. The cache must supply at least the first 10 derivatives
/// for GGA (or first 4 for LDA); `build_vxc_hessian_cache` guarantees this.
pub fn vxc_deriv1_cached(scf: &SCF, xc_type: XCType, cache: &VxcHessianCache) -> Vec<MatrixFull<f64>> {
    let mol = &scf.mol;
    let nao = mol.num_basis;
    let natm = mol.geom.nfree;
    let aoslices = build_aoslices(mol);
    let dm0 = &scf.density_matrix[0];
    let func_ids = &mol.xc_data.dfa_compnt_scf;
    let func_factors = &mol.xc_data.dfa_paramr_scf;

    let nvar = match xc_type {
        XCType::LDA => 1,
        XCType::GGA => 4,
        _ => panic!("vxc_deriv1: only LDA and GGA supported"),
    };

    // Parallel: each block produces (vmat_partial, v_ip_partial), then reduce.
    let (mut vmat, v_ip): (Vec<MatrixFull<f64>>, Vec<MatrixFull<f64>>) = cache.blocks
        .par_iter()
        .filter_map(|blk| {
            // Pin OpenBLAS to 1 thread per rayon worker (see vxc_diag_cached).
            omp_set_num_threads_wrapper(1);
            let nb = blk.nb;
            if nb == 0 { return None; }
            let weights_block = &blk.weights;
            let ao_d: &[MatrixFull<f64>] = &blk.ao_d;
            let rho_array = &blk.rho_array;
            let ao_d_refs: Vec<&MatrixFull<f64>> = ao_d.iter().collect();

            let xc_tensors = eval_xc_eff(func_ids, func_factors, xc_type, 0, rho_array, nb, 2);
            let vxc_t = xc_tensors[1].as_ref().expect("vxc required");
            let fxc_t = xc_tensors[2].as_ref().expect("fxc required");
            let vxc_raw = vxc_t.raw();
            let vxc_off = vxc_t.offset();
            let fxc_raw = fxc_t.raw();
            let fxc_off = fxc_t.offset();

            let mut vmat: Vec<MatrixFull<f64>> =
                (0..natm).map(|_| MatrixFull::new([3 * nao, nao], 0.0)).collect();
            let mut v_ip: Vec<MatrixFull<f64>> =
                (0..3).map(|_| MatrixFull::new([nao, nao], 0.0)).collect();

            match xc_type {
                XCType::LDA => {
                    let mut aow = MatrixFull::new([nao, nb], 0.0);
                    for g in 0..nb {
                        let wv = weights_block[g] * vxc_raw[vxc_off + g];
                        for mu in 0..nao { aow[[mu, g]] = ao_d[0][[mu, g]] * wv; }
                    }
                    for alpha in 0..3 {
                        _dgemm_full(&ao_d[alpha + 1], 'N', &aow, 'T', &mut v_ip[alpha], 1.0, 1.0);
                    }

                    let ao_dm0 = compute_ao_dm0(dm0, &ao_d_refs[..1], 1, nao, nb);
                    let wf_factor: Vec<f64> = (0..nb)
                        .map(|g| weights_block[g] * fxc_raw[fxc_off + g])
                        .collect();

                    for ia in 0..natm {
                        let (p0, p1) = aoslices[ia];
                        let mut wv = vec![0.0; 3 * nb];
                        for alpha in 0..3 {
                            for g in 0..nb {
                                let mut rho1 = 0.0;
                                for mu in p0..p1 {
                                    rho1 += ao_d[alpha + 1][[mu, g]] * ao_dm0[0][[mu, g]];
                                }
                                wv[alpha + 3 * g] = wf_factor[g] * rho1;
                            }
                        }
                        for alpha in 0..3 {
                            let mut aow_alpha = MatrixFull::new([nao, nb], 0.0);
                            for g in 0..nb {
                                let w = wv[alpha + 3 * g];
                                for mu in 0..nao { aow_alpha[[mu, g]] = ao_d[0][[mu, g]] * w; }
                            }
                            let row0 = alpha * nao;
                            let mut vmat_alpha = MatrixFull::new([nao, nao], 0.0);
                            for j in 0..nao { for i in 0..nao {
                                vmat_alpha[[i, j]] = vmat[ia][[row0 + i, j]];
                            }}
                            _dgemm_full(&aow_alpha, 'N', &ao_d[0], 'T', &mut vmat_alpha, 1.0, 1.0);
                            for j in 0..nao { for i in 0..nao {
                                vmat[ia][[row0 + i, j]] = vmat_alpha[[i, j]];
                            }}
                        }
                    }
                }
                XCType::GGA => {
                    let mut wv0 = vec![0.0; nb];
                    let mut wv1 = vec![0.0; nb];
                    let mut wv2 = vec![0.0; nb];
                    let mut wv3 = vec![0.0; nb];
                    for g in 0..nb {
                        let w = weights_block[g];
                        wv0[g] = 0.5 * w * vxc_raw[vxc_off + 0 * nb + g];
                        wv1[g] = w * vxc_raw[vxc_off + 1 * nb + g];
                        wv2[g] = w * vxc_raw[vxc_off + 2 * nb + g];
                        wv3[g] = w * vxc_raw[vxc_off + 3 * nb + g];
                    }
                    let mut aow_1a = MatrixFull::new([nao, nb], 0.0);
                    for mu in 0..nao { for g in 0..nb {
                        aow_1a[[mu, g]] = ao_d[0][[mu, g]] * wv0[g]
                            + ao_d[1][[mu, g]] * wv1[g]
                            + ao_d[2][[mu, g]] * wv2[g]
                            + ao_d[3][[mu, g]] * wv3[g];
                    }}
                    for alpha in 0..3 {
                        _dgemm_full(&ao_d[alpha + 1], 'N', &aow_1a, 'T', &mut v_ip[alpha], 1.0, 1.0);
                    }
                    let second_idx = [[XX, XY, XZ], [XY, YY, YZ], [XZ, YZ, ZZ]];
                    for alpha in 0..3 {
                        let mut aow_1b = MatrixFull::new([nao, nb], 0.0);
                        for mu in 0..nao { for g in 0..nb {
                            aow_1b[[mu, g]] = ao_d[alpha + 1][[mu, g]] * wv0[g]
                                + ao_d[second_idx[alpha][0]][[mu, g]] * wv1[g]
                                + ao_d[second_idx[alpha][1]][[mu, g]] * wv2[g]
                                + ao_d[second_idx[alpha][2]][[mu, g]] * wv3[g];
                        }}
                        _dgemm_full(&aow_1b, 'N', &ao_d[0], 'T', &mut v_ip[alpha], 1.0, 1.0);
                    }

                    let ao_dm0 = compute_ao_dm0(dm0, &ao_d_refs[..nvar], nvar, nao, nb);
                    for ia in 0..natm {
                        let (p0, p1) = aoslices[ia];
                        let dR_rho1 = make_dR_rho1(&ao_d_refs, &ao_dm0, p0, p1, xc_type, nb);
                        let mut wv = vec![0.0; 3 * nvar * nb];
                        for g in 0..nb {
                            let wf_g = weights_block[g];
                            for y in 0..nvar {
                                for s in 0..3 {
                                    let mut acc = 0.0;
                                    for x in 0..nvar {
                                        let fxc_xy = fxc_raw[fxc_off + g + x * nb + y * nvar * nb];
                                        let drho_sx = dR_rho1[s + x * 3 + g * 3 * nvar];
                                        acc += wf_g * fxc_xy * drho_sx;
                                    }
                                    let mut val = acc;
                                    if y == 0 { val *= 0.5; }
                                    wv[s + y * 3 + g * 3 * nvar] = val;
                                }
                            }
                        }
                        for s in 0..3 {
                            let mut aow_s = MatrixFull::new([nao, nb], 0.0);
                            for mu in 0..nao { for g in 0..nb {
                                let mut v = 0.0;
                                for y in 0..nvar {
                                    v += ao_d[y][[mu, g]] * wv[s + y * 3 + g * 3 * nvar];
                                }
                                aow_s[[mu, g]] = v;
                            }}
                            let row0 = s * nao;
                            let mut vmat_s = MatrixFull::new([nao, nao], 0.0);
                            for j in 0..nao { for i in 0..nao {
                                vmat_s[[i, j]] = vmat[ia][[row0 + i, j]];
                            }}
                            _dgemm_full(&aow_s, 'N', &ao_d[0], 'T', &mut vmat_s, 1.0, 1.0);
                            for j in 0..nao { for i in 0..nao {
                                vmat[ia][[row0 + i, j]] = vmat_s[[i, j]];
                            }}
                        }
                    }
                }
                _ => unreachable!(),
            }
            Some((vmat, v_ip))
        })
        .reduce(
            || (
                (0..natm).map(|_| MatrixFull::new([3 * nao, nao], 0.0)).collect::<Vec<_>>(),
                (0..3).map(|_| MatrixFull::new([nao, nao], 0.0)).collect::<Vec<_>>(),
            ),
            |(mut a_vmat, mut a_vip), (b_vmat, b_vip)| {
                for k in 0..natm { a_vmat[k] += b_vmat[k].clone(); }
                for k in 0..3 { a_vip[k] += b_vip[k].clone(); }
                (a_vmat, a_vip)
            },
        );

    // Final: vmat[ia, :, p0:p1, :] += v_ip[:, p0:p1, :]
    //        vmat[ia] = -vmat[ia] - vmat[ia]^T (last two axes transpose)
    for ia in 0..natm {
        let (p0, p1) = aoslices[ia];
        for alpha in 0..3 {
            let row0 = alpha * nao;
            for j in 0..nao {
                for i in p0..p1 {
                    vmat[ia][[row0 + i, j]] += v_ip[alpha][[i, j]];
                }
            }
        }
        // Symmetrize: vmat_new[α, μ, ν] = -vmat[α, μ, ν] - vmat[α, ν, μ]
        for alpha in 0..3 {
            let row0 = alpha * nao;
            // Build symmetrized copy
            let mut sym = MatrixFull::new([nao, nao], 0.0);
            for mu in 0..nao { for nu in 0..nao {
                let v_mu_nu = vmat[ia][[row0 + mu, nu]];
                let v_nu_mu = vmat[ia][[row0 + nu, mu]];
                sym[[mu, nu]] = -v_mu_nu - v_nu_mu;
            }}
            for mu in 0..nao { for nu in 0..nao {
                vmat[ia][[row0 + mu, nu]] = sym[[mu, nu]];
            }}
        }
    }

    vmat
}

// Third-order AO derivative indices (ao_deriv=3, components 10..20).
const XXX: usize = 10;
const XXY: usize = 11;
const XXZ: usize = 12;
const XYY: usize = 13;
const XYZ: usize = 14;
const XZZ: usize = 15;
const YYY: usize = 16;
const YYZ: usize = 17;
const YZZ: usize = 18;
const ZZZ: usize = 19;

/// XC diagonal second-derivative term that feeds h_partial (PySCF: _get_vxc_diag).
///
/// Output: `MatrixFull<f64>` shape `[9*nao, nao]` column-major holding the
/// `[3, 3, nao, nao]` tensor with α,β ∈ {0,1,2}. The reshape uses the same
/// ordering as PySCF: vmat[[0,1,2,1,3,4,2,4,5]] from internal (XX,XY,XZ,YY,YZ,ZZ).
/// Concretely, output row = (alpha*3 + beta)*nao + i, col = j  (alpha <= beta).
///
/// Uses deriv=1 (vxc only — no fxc) because this is the pure AO-second-deriv
/// piece, not a density-response piece.
pub fn vxc_diag(scf: &SCF, xc_type: XCType) -> MatrixFull<f64> {
    // Backward-compatible wrapper: build a one-shot cache and consume it.
    let cache = build_vxc_hessian_cache(scf, xc_type);
    vxc_diag_cached(scf, xc_type, &cache)
}

/// Cached variant of `vxc_diag`: reuses AO derivatives and ground-state ρ
/// from `cache`, only re-evaluating the XC kernel (`eval_xc_eff` deriv=1)
/// per block. Pairs with `build_vxc_hessian_cache` to share AO/ρ work across
/// `vxc_diag`, `vxc_deriv1`, and `vxc_deriv2`.
pub fn vxc_diag_cached(scf: &SCF, xc_type: XCType, cache: &VxcHessianCache) -> MatrixFull<f64> {
    let mol = &scf.mol;
    let nao = mol.num_basis;
    let func_ids = &mol.xc_data.dfa_compnt_scf;
    let func_factors = &mol.xc_data.dfa_paramr_scf;

    let nvar = match xc_type {
        XCType::LDA => 1,
        XCType::GGA => 4,
        _ => panic!("vxc_diag: only LDA and GGA supported"),
    };

    // Parallel: each block produces its own 6 partial v6 matrices, then we sum.
    let v6: Vec<MatrixFull<f64>> = cache.blocks
        .par_iter()
        .filter_map(|blk| {
            // Pin OpenBLAS to 1 thread per rayon worker — otherwise the
            // nested OpenMP team (default 8 threads) competes with rayon
            // for the same cores, causing ~10× slowdown on the many small
            // DGEMMs in this loop. Same pattern as dft/mod.rs:3747.
            omp_set_num_threads_wrapper(1);
            let nb = blk.nb;
            if nb == 0 { return None; }
            let weights_block = &blk.weights;
            let ao_d: &[MatrixFull<f64>] = &blk.ao_d;
            let rho_array = &blk.rho_array;

            let xc_tensors = eval_xc_eff(func_ids, func_factors, xc_type, 0, rho_array, nb, 1);
            let vxc_t = xc_tensors[1].as_ref().expect("vxc_diag: vxc required");
            let vxc_raw = vxc_t.raw();
            let vxc_off = vxc_t.offset();

            let mut v6: Vec<MatrixFull<f64>> =
                (0..6).map(|_| MatrixFull::new([nao, nao], 0.0)).collect();

            match xc_type {
                XCType::LDA => {
                    let mut aow = MatrixFull::new([nao, nb], 0.0);
                    for g in 0..nb {
                        let wv = weights_block[g] * vxc_raw[vxc_off + g];
                        for mu in 0..nao { aow[[mu, g]] = ao_d[0][[mu, g]] * wv; }
                    }
                    for i in 0..6 {
                        _dgemm_full(&ao_d[i + 4], 'N', &aow, 'T', &mut v6[i], 1.0, 1.0);
                    }
                }
                XCType::GGA => {
                    let mut wv0 = vec![0.0; nb];
                    let mut wv1 = vec![0.0; nb];
                    let mut wv2 = vec![0.0; nb];
                    let mut wv3 = vec![0.0; nb];
                    for g in 0..nb {
                        let w = weights_block[g];
                        wv0[g] = w * vxc_raw[vxc_off + 0 * nb + g];
                        wv1[g] = w * vxc_raw[vxc_off + 1 * nb + g];
                        wv2[g] = w * vxc_raw[vxc_off + 2 * nb + g];
                        wv3[g] = w * vxc_raw[vxc_off + 3 * nb + g];
                    }
                    let mut aow_1a = MatrixFull::new([nao, nb], 0.0);
                    for mu in 0..nao { for g in 0..nb {
                        aow_1a[[mu, g]] = ao_d[0][[mu, g]] * wv0[g]
                            + ao_d[1][[mu, g]] * wv1[g]
                            + ao_d[2][[mu, g]] * wv2[g]
                            + ao_d[3][[mu, g]] * wv3[g];
                    }}
                    for i in 0..6 {
                        _dgemm_full(&ao_d[i + 4], 'N', &aow_1a, 'T', &mut v6[i], 1.0, 1.0);
                    }

                    let wv_s = [wv1.as_slice(), wv2.as_slice(), wv3.as_slice()];
                    let contract_indices: [[usize; 3]; 6] = [
                        [XXX, XXY, XXZ],
                        [XXY, XYY, XYZ],
                        [XXZ, XYZ, XZZ],
                        [XYY, YYY, YYZ],
                        [XYZ, YYZ, YZZ],
                        [XZZ, YZZ, ZZZ],
                    ];
                    for (i, aoidx) in contract_indices.iter().enumerate() {
                        let mut aow_c = MatrixFull::new([nao, nb], 0.0);
                        for mu in 0..nao { for g in 0..nb {
                            aow_c[[mu, g]] = ao_d[aoidx[0]][[mu, g]] * wv_s[0][g]
                                + ao_d[aoidx[1]][[mu, g]] * wv_s[1][g]
                                + ao_d[aoidx[2]][[mu, g]] * wv_s[2][g];
                        }}
                        _dgemm_full(&aow_c, 'N', &ao_d[0], 'T', &mut v6[i], 1.0, 1.0);
                    }
                }
                _ => unreachable!(),
            }
            Some(v6)
        })
        .reduce(|| vec![MatrixFull::new([nao, nao], 0.0); 6], |mut a, b| {
            for k in 0..6 { a[k] += b[k].clone(); }
            a
        });

    // Reshape: 6 → 3x3 with same indexing as PySCF.
    let mut out = MatrixFull::new([9 * nao, nao], 0.0);
    let v3_map = [[0, 1, 2], [1, 3, 4], [2, 4, 5]];
    for alpha in 0..3 {
        for beta in 0..3 {
            let v6_idx = v3_map[alpha][beta];
            let row0 = (alpha * 3 + beta) * nao;
            for i in 0..nao { for j in 0..nao {
                out[[row0 + i, j]] = v6[v6_idx][[i, j]];
            }}
        }
    }
    out
}

/// Maximum number of grid blocks processed concurrently in streaming mode.
/// Reads `REST_HESS_GRID_CONCURRENCY` env var (default 2). Lower values
/// reduce peak RSS at the cost of less grid-level parallelism.
fn grid_concurrency() -> usize {
    std::env::var("REST_HESS_GRID_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&v| v >= 1)
        .unwrap_or(2)
}

/// Streaming variant of `vxc_diag`: evaluates AO + ρ₀ + XC kernel per block
/// in small concurrent batches, never collecting all blocks at once.
///
/// `ao_deriv` is forced to3 for GGA / 2 for LDA (same as `build_vxc_hessian_cache`).
/// Peak AO memory ≈ grid_concurrency() × per_block_ao_size.
pub fn vxc_diag_streaming(scf: &SCF, xc_type: XCType) -> MatrixFull<f64> {
    let mol = &scf.mol;
    let nao = mol.num_basis;
    let grids = scf.grids.as_ref().expect("vxc_diag_streaming requires scf.grids");
    let func_ids = &mol.xc_data.dfa_compnt_scf;
    let func_factors = &mol.xc_data.dfa_paramr_scf;
    let mo_vec = vec![scf.eigenvectors[0].clone()];
    let occ_vec = vec![scf.occupation[0].clone()];

    let ao_deriv = match xc_type { XCType::LDA => 2, XCType::GGA => 3, _ => panic!() };
    let nderiv_max = (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) / 6;

    let concurrency = grid_concurrency();
    let block_ranges: &[std::ops::Range<usize>] = &grids.parallel_balancing;

    // Accumulator: 6 partial [nao,nao] matrices
    let mut acc_v6: Vec<MatrixFull<f64>> =
        (0..6).map(|_| MatrixFull::new([nao, nao], 0.0)).collect();

    for chunk in block_ranges.chunks(concurrency) {
        let partials: Vec<Vec<MatrixFull<f64>>> = chunk
            .par_iter()
            .filter_map(|range| {
                omp_set_num_threads_wrapper(1);
                let nb = range.end - range.start;
                if nb == 0 { return None; }
                let coords_block = &grids.coordinates[range.start..range.end];
                let weights_block = &grids.weights[range.start..range.end];

                let ao = eval_ao_batch(mol, coords_block, ao_deriv, nb);
                let ao_d: Vec<MatrixFull<f64>> = (0..nderiv_max)
                    .map(|d| {
                        let view = ao.get_reducing_matrix(d).unwrap();
                        MatrixFull::from_vec([nao, nb], view.iter().copied().collect()).unwrap()
                    })
                    .collect();
                let nvar = match xc_type { XCType::LDA => 1, XCType::GGA => 4, _ => panic!() };
                let rho_tensor = eval_rho5_batch(&ao, xc_type, &mo_vec, &occ_vec, 1, nb);
                let rho_array: Vec<f64> = {
                    let raw = rho_tensor.raw();
                    let off = rho_tensor.offset();
                    raw[off..off + nb * nvar].to_vec()
                };

                let xc_tensors = eval_xc_eff(func_ids, func_factors, xc_type, 0, &rho_array, nb, 1);
                let vxc_t = xc_tensors[1].as_ref().expect("vxc_diag: vxc required");
                let vxc_raw = vxc_t.raw();
                let vxc_off = vxc_t.offset();

                let mut v6: Vec<MatrixFull<f64>> =
                    (0..6).map(|_| MatrixFull::new([nao, nao], 0.0)).collect();

                match xc_type {
                    XCType::LDA => {
                        let mut aow = MatrixFull::new([nao, nb], 0.0);
                        for g in 0..nb {
                            let wv = weights_block[g] * vxc_raw[vxc_off + g];
                            for mu in 0..nao { aow[[mu, g]] = ao_d[0][[mu, g]] * wv; }
                        }
                        for i in 0..6 {
                            _dgemm_full(&ao_d[i + 4], 'N', &aow, 'T', &mut v6[i], 1.0, 1.0);
                        }
                    }
                    XCType::GGA => {
                        let mut wv0 = vec![0.0; nb];
                        let mut wv1 = vec![0.0; nb];
                        let mut wv2 = vec![0.0; nb];
                        let mut wv3 = vec![0.0; nb];
                        for g in 0..nb {
                            let w = weights_block[g];
                            wv0[g] = w * vxc_raw[vxc_off + 0 * nb + g];
                            wv1[g] = w * vxc_raw[vxc_off + 1 * nb + g];
                            wv2[g] = w * vxc_raw[vxc_off + 2 * nb + g];
                            wv3[g] = w * vxc_raw[vxc_off + 3 * nb + g];
                        }
                        let mut aow_1a = MatrixFull::new([nao, nb], 0.0);
                        for mu in 0..nao { for g in 0..nb {
                            aow_1a[[mu, g]] = ao_d[0][[mu, g]] * wv0[g]
                                + ao_d[1][[mu, g]] * wv1[g]
                                + ao_d[2][[mu, g]] * wv2[g]
                                + ao_d[3][[mu, g]] * wv3[g];
                        }}
                        for i in 0..6 {
                            _dgemm_full(&ao_d[i + 4], 'N', &aow_1a, 'T', &mut v6[i], 1.0, 1.0);
                        }
                        let wv_s = [wv1.as_slice(), wv2.as_slice(), wv3.as_slice()];
                        let contract_indices: [[usize; 3]; 6] = [
                            [XXX, XXY, XXZ],
                            [XXY, XYY, XYZ],
                            [XXZ, XYZ, XZZ],
                            [XYY, YYY, YYZ],
                            [XYZ, YYZ, YZZ],
                            [XZZ, YZZ, ZZZ],
                        ];
                        for (i, aoidx) in contract_indices.iter().enumerate() {
                            let mut aow_c = MatrixFull::new([nao, nb], 0.0);
                            for mu in 0..nao { for g in 0..nb {
                                aow_c[[mu, g]] = ao_d[aoidx[0]][[mu, g]] * wv_s[0][g]
                                    + ao_d[aoidx[1]][[mu, g]] * wv_s[1][g]
                                    + ao_d[aoidx[2]][[mu, g]] * wv_s[2][g];
                            }}
                            _dgemm_full(&aow_c, 'N', &ao_d[0], 'T', &mut v6[i], 1.0, 1.0);
                        }
                    }
                    _ => unreachable!(),
                }
                Some(v6)
            })
            .collect();

        for v6 in partials {
            for k in 0..6 { acc_v6[k] += v6[k].clone(); }
        }
    }

    // Reshape: 6 → 3x3
    let mut out = MatrixFull::new([9 * nao, nao], 0.0);
    let v3_map = [[0, 1, 2], [1, 3, 4], [2, 4, 5]];
    for alpha in 0..3 {
        for beta in 0..3 {
            let v6_idx = v3_map[alpha][beta];
            let row0 = (alpha * 3 + beta) * nao;
            for i in 0..nao { for j in 0..nao {
                out[[row0 + i, j]] = acc_v6[v6_idx][[i, j]];
            }}
        }
    }
    out
}

/// Streaming variant of `vxc_deriv2`: evaluates AO (deriv=2) + ρ₀ + XC kernel
/// per block in small concurrent batches. Uses deriv=2 (not deriv=3) since
/// vxc_deriv2 only needs up to 2nd AO derivatives, halving per-block AO memory
/// compared to the shared cache (which forced deriv=3 for vxc_diag).
pub fn vxc_deriv2_streaming(scf: &SCF, xc_type: XCType) -> Vec<MatrixFull<f64>> {
    let mol = &scf.mol;
    let nao = mol.num_basis;
    let natm = mol.geom.nfree;
    let grids = scf.grids.as_ref().expect("vxc_deriv2_streaming requires scf.grids");
    let aoslices = build_aoslices(mol);
    let dm0 = &scf.density_matrix[0];
    let func_ids = &mol.xc_data.dfa_compnt_scf;
    let func_factors = &mol.xc_data.dfa_paramr_scf;
    let mo_vec = vec![scf.eigenvectors[0].clone()];
    let occ_vec = vec![scf.occupation[0].clone()];

    let nvar = match xc_type {
        XCType::LDA => 1, XCType::GGA => 4, _ => panic!(),
    };
    let ao_deriv = match xc_type { XCType::LDA => 1, XCType::GGA => 2, _ => panic!() };
    let nderiv_max = (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) / 6;

    let concurrency = grid_concurrency();
    let block_ranges: &[std::ops::Range<usize>] = &grids.parallel_balancing;

    // Accumulators
    let mut acc_vmat: Vec<MatrixFull<f64>> =
        (0..natm).map(|_| MatrixFull::new([9 * nao, nao], 0.0)).collect();
    let mut acc_ipip = MatrixFull::new([9 * nao, nao], 0.0);

    for chunk in block_ranges.chunks(concurrency) {
        let partials: Vec<(Vec<MatrixFull<f64>>, MatrixFull<f64>)> = chunk
            .par_iter()
            .filter_map(|range| {
                omp_set_num_threads_wrapper(1);
                let nb = range.end - range.start;
                if nb == 0 { return None; }
                let weights_block = &grids.weights[range.start..range.end];
                let coords_block = &grids.coordinates[range.start..range.end];

                let ao = eval_ao_batch(mol, coords_block, ao_deriv, nb);
                let ao_d: Vec<MatrixFull<f64>> = (0..nderiv_max)
                    .map(|d| {
                        let view = ao.get_reducing_matrix(d).unwrap();
                        MatrixFull::from_vec([nao, nb], view.iter().copied().collect()).unwrap()
                    })
                    .collect();
                let ao_d_refs: Vec<&MatrixFull<f64>> = ao_d.iter().collect();

                let rho_tensor = eval_rho5_batch(&ao, xc_type, &mo_vec, &occ_vec, 1, nb);
                let rho_array: Vec<f64> = {
                    let raw = rho_tensor.raw();
                    let off = rho_tensor.offset();
                    raw[off..off + nb * nvar].to_vec()
                };

                let xc_tensors = eval_xc_eff(func_ids, func_factors, xc_type, 0, &rho_array, nb, 2);
                let vxc_t = xc_tensors[1].as_ref().expect("vxc_deriv2: vxc required");
                let fxc_t = xc_tensors[2].as_ref().expect("vxc_deriv2: fxc required");
                let vxc_raw = vxc_t.raw();
                let vxc_off = vxc_t.offset();
                let fxc_raw = fxc_t.raw();
                let fxc_off = fxc_t.offset();

                let mut vmat: Vec<MatrixFull<f64>> =
                    (0..natm).map(|_| MatrixFull::new([9 * nao, nao], 0.0)).collect();
                let mut ipip = MatrixFull::new([9 * nao, nao], 0.0);

                let add_block = |dst: &mut MatrixFull<f64>, d1: usize, d2: usize,
                                 a: &MatrixFull<f64>, b: &MatrixFull<f64>| {
                    let row0 = (d1 * 3 + d2) * nao;
                    let mut blk = MatrixFull::new([nao, nao], 0.0);
                    for j in 0..nao { for i in 0..nao {
                        blk[[i, j]] = dst[[row0 + i, j]];
                    }}
                    _dgemm_full(a, 'N', b, 'T', &mut blk, 1.0, 1.0);
                    for j in 0..nao { for i in 0..nao {
                        dst[[row0 + i, j]] = blk[[i, j]];
                    }}
                };

                match xc_type {
                    XCType::LDA => {
                        let mut wv = vec![0.0; nb];
                        for g in 0..nb { wv[g] = weights_block[g] * vxc_raw[vxc_off + g]; }
                        let aow_scale: Vec<MatrixFull<f64>> = (0..3)
                            .map(|d| {
                                let mut m = MatrixFull::new([nao, nb], 0.0);
                                for mu in 0..nao { for g in 0..nb {
                                    m[[mu, g]] = ao_d[1 + d][[mu, g]] * wv[g];
                                }}
                                m
                            })
                            .collect();
                        for d1 in 0..3 {
                            for d2 in 0..3 {
                                add_block(&mut ipip, d1, d2, &aow_scale[d2], &ao_d[1 + d1]);
                            }
                        }
                        let ao_dm0 = compute_ao_dm0(dm0, &ao_d_refs[..1], 1, nao, nb);
                        let mut wf = vec![0.0; nb];
                        for g in 0..nb { wf[g] = weights_block[g] * fxc_raw[fxc_off + g]; }
                        for ia in 0..natm {
                            let (p0, p1) = aoslices[ia];
                            for d1 in 0..3 {
                                let mut wv_d1 = vec![0.0; nb];
                                for g in 0..nb {
                                    let mut rho1 = 0.0;
                                    for mu in p0..p1 {
                                        rho1 += ao_d[1 + d1][[mu, g]] * ao_dm0[0][[mu, g]];
                                    }
                                    wv_d1[g] = wf[g] * 2.0 * rho1;
                                }
                                let mut aow_d1 = MatrixFull::new([nao, nb], 0.0);
                                for mu in 0..nao { for g in 0..nb {
                                    aow_d1[[mu, g]] = ao_d[0][[mu, g]] * wv_d1[g];
                                }}
                                for d2 in 0..3 {
                                    add_block(&mut vmat[ia], d1, d2, &aow_d1, &ao_d[1 + d2]);
                                }
                            }
                        }
                    }
                    XCType::GGA => {
                        let mut wv0 = vec![0.0; nb];
                        let mut wv1 = vec![0.0; nb];
                        let mut wv2 = vec![0.0; nb];
                        let mut wv3 = vec![0.0; nb];
                        for g in 0..nb {
                            let w = weights_block[g];
                            wv0[g] = 0.5 * w * vxc_raw[vxc_off + 0 * nb + g];
                            wv1[g] = w * vxc_raw[vxc_off + 1 * nb + g];
                            wv2[g] = w * vxc_raw[vxc_off + 2 * nb + g];
                            wv3[g] = w * vxc_raw[vxc_off + 3 * nb + g];
                        }
                        let aow_diag = make_dR_dao_w(&ao_d, &wv0, &wv1, &wv2, &wv3, nao, nb);
                        for d1 in 0..3 {
                            for d2 in 0..3 {
                                add_block(&mut ipip, d1, d2, &aow_diag[d2], &ao_d[1 + d1]);
                            }
                        }
                        let ao_dm0 = compute_ao_dm0(dm0, &ao_d_refs[..nvar], nvar, nao, nb);
                        for ia in 0..natm {
                            let (p0, p1) = aoslices[ia];
                            let dR_rho1 = make_dR_rho1(&ao_d_refs, &ao_dm0, p0, p1, xc_type, nb);
                            let mut wv_block = vec![0.0; 3 * nvar * nb];
                            for g in 0..nb {
                                let wf_g = weights_block[g];
                                for y in 0..nvar {
                                    for s in 0..3 {
                                        let mut acc = 0.0;
                                        for x in 0..nvar {
                                            let fxc_xy = fxc_raw[fxc_off + g + x * nb + y * nvar * nb];
                                            let drho_sx = dR_rho1[s + x * 3 + g * 3 * nvar];
                                            acc += wf_g * fxc_xy * drho_sx;
                                        }
                                        let mut val = acc;
                                        if y == 0 { val *= 0.5; }
                                        wv_block[s + y * 3 + g * 3 * nvar] = val;
                                    }
                                }
                            }
                            for i in 0..3 {
                                let wv_i0: Vec<f64> = (0..nb).map(|g| wv_block[i + 0 * 3 + g * 3 * nvar]).collect();
                                let wv_i1: Vec<f64> = (0..nb).map(|g| wv_block[i + 1 * 3 + g * 3 * nvar]).collect();
                                let wv_i2: Vec<f64> = (0..nb).map(|g| wv_block[i + 2 * 3 + g * 3 * nvar]).collect();
                                let wv_i3: Vec<f64> = (0..nb).map(|g| wv_block[i + 3 * 3 + g * 3 * nvar]).collect();
                                let aow_i = make_dR_dao_w(&ao_d, &wv_i0, &wv_i1, &wv_i2, &wv_i3, nao, nb);
                                for d2 in 0..3 {
                                    add_block(&mut vmat[ia], i, d2, &aow_i[d2], &ao_d[0]);
                                }
                            }
                            for d1 in 0..3 {
                                let mut aow_d1 = MatrixFull::new([nao, nb], 0.0);
                                for mu in 0..nao { for g in 0..nb {
                                    let mut v = 0.0;
                                    for y in 0..nvar {
                                        v += ao_d[y][[mu, g]] * wv_block[d1 + y * 3 + g * 3 * nvar];
                                    }
                                    aow_d1[[mu, g]] = v;
                                }}
                                for d2 in 0..3 {
                                    add_block(&mut vmat[ia], d1, d2, &ao_d[1 + d2], &aow_d1);
                                }
                            }
                        }
                    }
                    _ => unreachable!(),
                }
                Some((vmat, ipip))
            })
            .collect();

        for (vmat_p, ipip_p) in partials {
            for k in 0..natm { acc_vmat[k] += vmat_p[k].clone(); }
            acc_ipip += ipip_p.clone();
        }
    }

    // ── Scatter ipip into each atom's AO column slice ──
    let needs_transpose = matches!(xc_type, XCType::GGA);
    for ia in 0..natm {
        let (p0, p1) = aoslices[ia];
        for d1 in 0..3 {
            for d2 in 0..3 {
                let row0 = (d1 * 3 + d2) * nao;
                let row0_T = (d2 * 3 + d1) * nao;
                for mu in 0..nao {
                    for nu in p0..p1 {
                        let mut inc = acc_ipip[[row0 + mu, nu]];
                        if needs_transpose {
                            inc += acc_ipip[[row0_T + nu, mu]];
                        }
                        acc_vmat[ia][[row0 + mu, nu]] += inc;
                    }
                }
            }
        }
    }

    acc_vmat
}

/// Streaming variant of `vxc_deriv1`: evaluates AO (deriv=2) + ρ₀ + XC kernel
/// per block in small concurrent batches. Used by `calc_h1ao` to avoid building
/// the full VxcHessianCache (~1.3 GiB for C6H6 GGA).
pub fn vxc_deriv1_streaming(scf: &SCF, xc_type: XCType) -> Vec<MatrixFull<f64>> {
    let mol = &scf.mol;
    let nao = mol.num_basis;
    let natm = mol.geom.nfree;
    let grids = scf.grids.as_ref().expect("vxc_deriv1_streaming requires scf.grids");
    let aoslices = build_aoslices(mol);
    let dm0 = &scf.density_matrix[0];
    let func_ids = &mol.xc_data.dfa_compnt_scf;
    let func_factors = &mol.xc_data.dfa_paramr_scf;
    let mo_vec = vec![scf.eigenvectors[0].clone()];
    let occ_vec = vec![scf.occupation[0].clone()];

    let nvar = match xc_type { XCType::LDA => 1, XCType::GGA => 4, _ => panic!() };
    let ao_deriv = match xc_type { XCType::LDA => 1, XCType::GGA => 2, _ => panic!() };
    let nderiv_max = (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) / 6;

    let concurrency = grid_concurrency();
    let block_ranges: &[std::ops::Range<usize>] = &grids.parallel_balancing;

    // Accumulators
    let mut acc_vmat: Vec<MatrixFull<f64>> =
        (0..natm).map(|_| MatrixFull::new([3 * nao, nao], 0.0)).collect();
    let mut acc_vip: Vec<MatrixFull<f64>> =
        (0..3).map(|_| MatrixFull::new([nao, nao], 0.0)).collect();

    for chunk in block_ranges.chunks(concurrency) {
        let partials: Vec<(Vec<MatrixFull<f64>>, Vec<MatrixFull<f64>>)> = chunk
            .par_iter()
            .filter_map(|range| {
                omp_set_num_threads_wrapper(1);
                let nb = range.end - range.start;
                if nb == 0 { return None; }
                let weights_block = &grids.weights[range.start..range.end];
                let coords_block = &grids.coordinates[range.start..range.end];

                let ao = eval_ao_batch(mol, coords_block, ao_deriv, nb);
                let ao_d: Vec<MatrixFull<f64>> = (0..nderiv_max)
                    .map(|d| {
                        let view = ao.get_reducing_matrix(d).unwrap();
                        MatrixFull::from_vec([nao, nb], view.iter().copied().collect()).unwrap()
                    })
                    .collect();
                let ao_d_refs: Vec<&MatrixFull<f64>> = ao_d.iter().collect();

                let rho_tensor = eval_rho5_batch(&ao, xc_type, &mo_vec, &occ_vec, 1, nb);
                let rho_array: Vec<f64> = {
                    let raw = rho_tensor.raw();
                    let off = rho_tensor.offset();
                    raw[off..off + nb * nvar].to_vec()
                };

                let xc_tensors = eval_xc_eff(func_ids, func_factors, xc_type, 0, &rho_array, nb, 2);
                let vxc_t = xc_tensors[1].as_ref().expect("vxc required");
                let fxc_t = xc_tensors[2].as_ref().expect("fxc required");
                let vxc_raw = vxc_t.raw();
                let vxc_off = vxc_t.offset();
                let fxc_raw = fxc_t.raw();
                let fxc_off = fxc_t.offset();

                let mut vmat: Vec<MatrixFull<f64>> =
                    (0..natm).map(|_| MatrixFull::new([3 * nao, nao], 0.0)).collect();
                let mut v_ip: Vec<MatrixFull<f64>> =
                    (0..3).map(|_| MatrixFull::new([nao, nao], 0.0)).collect();

                match xc_type {
                    XCType::LDA => {
                        let mut aow = MatrixFull::new([nao, nb], 0.0);
                        for g in 0..nb {
                            let wv = weights_block[g] * vxc_raw[vxc_off + g];
                            for mu in 0..nao { aow[[mu, g]] = ao_d[0][[mu, g]] * wv; }
                        }
                        for alpha in 0..3 {
                            _dgemm_full(&ao_d[alpha + 1], 'N', &aow, 'T', &mut v_ip[alpha], 1.0, 1.0);
                        }
                        let ao_dm0 = compute_ao_dm0(dm0, &ao_d_refs[..1], 1, nao, nb);
                        let wf_factor: Vec<f64> = (0..nb)
                            .map(|g| weights_block[g] * fxc_raw[fxc_off + g])
                            .collect();
                        for ia in 0..natm {
                            let (p0, p1) = aoslices[ia];
                            let mut wv = vec![0.0; 3 * nb];
                            for alpha in 0..3 {
                                for g in 0..nb {
                                    let mut rho1 = 0.0;
                                    for mu in p0..p1 {
                                        rho1 += ao_d[alpha + 1][[mu, g]] * ao_dm0[0][[mu, g]];
                                    }
                                    wv[alpha + 3 * g] = wf_factor[g] * rho1;
                                }
                            }
                            for alpha in 0..3 {
                                let mut aow_alpha = MatrixFull::new([nao, nb], 0.0);
                                for g in 0..nb {
                                    let w = wv[alpha + 3 * g];
                                    for mu in 0..nao { aow_alpha[[mu, g]] = ao_d[0][[mu, g]] * w; }
                                }
                                let row0 = alpha * nao;
                                let mut vmat_alpha = MatrixFull::new([nao, nao], 0.0);
                                for j in 0..nao { for i in 0..nao {
                                    vmat_alpha[[i, j]] = vmat[ia][[row0 + i, j]];
                                }}
                                _dgemm_full(&aow_alpha, 'N', &ao_d[0], 'T', &mut vmat_alpha, 1.0, 1.0);
                                for j in 0..nao { for i in 0..nao {
                                    vmat[ia][[row0 + i, j]] = vmat_alpha[[i, j]];
                                }}
                            }
                        }
                    }
                    XCType::GGA => {
                        let mut wv0 = vec![0.0; nb];
                        let mut wv1 = vec![0.0; nb];
                        let mut wv2 = vec![0.0; nb];
                        let mut wv3 = vec![0.0; nb];
                        for g in 0..nb {
                            let w = weights_block[g];
                            wv0[g] = 0.5 * w * vxc_raw[vxc_off + 0 * nb + g];
                            wv1[g] = w * vxc_raw[vxc_off + 1 * nb + g];
                            wv2[g] = w * vxc_raw[vxc_off + 2 * nb + g];
                            wv3[g] = w * vxc_raw[vxc_off + 3 * nb + g];
                        }
                        let mut aow_1a = MatrixFull::new([nao, nb], 0.0);
                        for mu in 0..nao { for g in 0..nb {
                            aow_1a[[mu, g]] = ao_d[0][[mu, g]] * wv0[g]
                                + ao_d[1][[mu, g]] * wv1[g]
                                + ao_d[2][[mu, g]] * wv2[g]
                                + ao_d[3][[mu, g]] * wv3[g];
                        }}
                        for alpha in 0..3 {
                            _dgemm_full(&ao_d[alpha + 1], 'N', &aow_1a, 'T', &mut v_ip[alpha], 1.0, 1.0);
                        }
                        let second_idx = [[XX, XY, XZ], [XY, YY, YZ], [XZ, YZ, ZZ]];
                        for alpha in 0..3 {
                            let mut aow_1b = MatrixFull::new([nao, nb], 0.0);
                            for mu in 0..nao { for g in 0..nb {
                                aow_1b[[mu, g]] = ao_d[alpha + 1][[mu, g]] * wv0[g]
                                    + ao_d[second_idx[alpha][0]][[mu, g]] * wv1[g]
                                    + ao_d[second_idx[alpha][1]][[mu, g]] * wv2[g]
                                    + ao_d[second_idx[alpha][2]][[mu, g]] * wv3[g];
                            }}
                            _dgemm_full(&aow_1b, 'N', &ao_d[0], 'T', &mut v_ip[alpha], 1.0, 1.0);
                        }
                        let ao_dm0 = compute_ao_dm0(dm0, &ao_d_refs[..nvar], nvar, nao, nb);
                        for ia in 0..natm {
                            let (p0, p1) = aoslices[ia];
                            let dR_rho1 = make_dR_rho1(&ao_d_refs, &ao_dm0, p0, p1, xc_type, nb);
                            let mut wv = vec![0.0; 3 * nvar * nb];
                            for g in 0..nb {
                                let wf_g = weights_block[g];
                                for y in 0..nvar {
                                    for s in 0..3 {
                                        let mut acc = 0.0;
                                        for x in 0..nvar {
                                            let fxc_xy = fxc_raw[fxc_off + g + x * nb + y * nvar * nb];
                                            let drho_sx = dR_rho1[s + x * 3 + g * 3 * nvar];
                                            acc += wf_g * fxc_xy * drho_sx;
                                        }
                                        let mut val = acc;
                                        if y == 0 { val *= 0.5; }
                                        wv[s + y * 3 + g * 3 * nvar] = val;
                                    }
                                }
                            }
                            for s in 0..3 {
                                let mut aow_s = MatrixFull::new([nao, nb], 0.0);
                                for mu in 0..nao { for g in 0..nb {
                                    let mut v = 0.0;
                                    for y in 0..nvar {
                                        v += ao_d[y][[mu, g]] * wv[s + y * 3 + g * 3 * nvar];
                                    }
                                    aow_s[[mu, g]] = v;
                                }}
                                let row0 = s * nao;
                                let mut vmat_s = MatrixFull::new([nao, nao], 0.0);
                                for j in 0..nao { for i in 0..nao {
                                    vmat_s[[i, j]] = vmat[ia][[row0 + i, j]];
                                }}
                                _dgemm_full(&aow_s, 'N', &ao_d[0], 'T', &mut vmat_s, 1.0, 1.0);
                                for j in 0..nao { for i in 0..nao {
                                    vmat[ia][[row0 + i, j]] = vmat_s[[i, j]];
                                }}
                            }
                        }
                    }
                    _ => unreachable!(),
                }
                Some((vmat, v_ip))
            })
            .collect();

        for (vmat_p, vip_p) in partials {
            for k in 0..natm { acc_vmat[k] += vmat_p[k].clone(); }
            for k in 0..3 { acc_vip[k] += vip_p[k].clone(); }
        }
    }

    // Final: vmat[ia, :, p0:p1, :] += v_ip[:, p0:p1, :]
    //        vmat[ia] = -vmat[ia] - vmat[ia]^T (last two axes transpose)
    for ia in 0..natm {
        let (p0, p1) = aoslices[ia];
        for alpha in 0..3 {
            let row0 = alpha * nao;
            for j in 0..nao {
                for i in p0..p1 {
                    acc_vmat[ia][[row0 + i, j]] += acc_vip[alpha][[i, j]];
                }
            }
        }
        for alpha in 0..3 {
            let row0 = alpha * nao;
            let mut sym = MatrixFull::new([nao, nao], 0.0);
            for mu in 0..nao { for nu in 0..nao {
                let v_mu_nu = acc_vmat[ia][[row0 + mu, nu]];
                let v_nu_mu = acc_vmat[ia][[row0 + nu, mu]];
                sym[[mu, nu]] = -v_mu_nu - v_nu_mu;
            }}
            for mu in 0..nao { for nu in 0..nao {
                acc_vmat[ia][[row0 + mu, nu]] = sym[[mu, nu]];
            }}
        }
    }

    acc_vmat
}


// Returns 3 matrices of shape [nao, nb], one per spatial direction α.
//   aow[α][μ, g] = ∂_α ϕ · wv0
//                + ∂²_{α,s(α,0)} ϕ · wv1
//                + ∂²_{α,s(α,1)} ϕ · wv2
//                + ∂²_{α,s(α,2)} ϕ · wv3
// where second_idx[α] = [XX_or_YX_or_ZX, XY_or_YY_or_ZY, XZ_or_YZ_or_ZZ].
// Uses ao_d indexed: 1=∂ₓ, 2=∂ᵧ, 3=∂_z, 4=XX, 5=XY, 6=XZ, 7=YY, 8=YZ, 9=ZZ.
fn make_dR_dao_w(
    ao_d: &[MatrixFull<f64>],
    wv0: &[f64],
    wv1: &[f64],
    wv2: &[f64],
    wv3: &[f64],
    nao: usize,
    nb: usize,
) -> [MatrixFull<f64>; 3] {
    let second_idx: [[usize; 3]; 3] = [[XX, XY, XZ], [XY, YY, YZ], [XZ, YZ, ZZ]];
    let wv_s = [wv1, wv2, wv3];
    let mut aow = [
        MatrixFull::new([nao, nb], 0.0),
        MatrixFull::new([nao, nb], 0.0),
        MatrixFull::new([nao, nb], 0.0),
    ];
    for alpha in 0..3 {
        for mu in 0..nao {
            for g in 0..nb {
                let mut v = ao_d[alpha + 1][[mu, g]] * wv0[g];
                for k in 0..3 {
                    v += ao_d[second_idx[alpha][k]][[mu, g]] * wv_s[k][g];
                }
                aow[alpha][[mu, g]] = v;
            }
        }
    }
    aow
}

/// XC off-diagonal second-derivative term that feeds h_partial (PySCF: _get_vxc_deriv2).
///
/// Output: Vec of `natm` matrices, each `[9*nao, nao]` column-major encoding the
/// `[3, 3, nao, nao]` tensor `vmat[ia, α, β, μ, ν]` where element is stored at
/// row `(α*3 + β)*nao + μ`, col `ν`. Lower-triangle `α, β` symmetric pairs share
/// data after the final symmetrization (vmat[ia, α, β] = vmat[ia, β, α] expected
/// after assembly).
///
/// Composition:
///   * `ipip[3, 3, nao, nao]`: atom-independent diagonal-in-atom term
///     (vxc · ∂²ϕ_αβ · ϕ). After all blocks, scattered into each atom's AO
///     column slice (and additionally its transpose for GGA).
///   * Per-atom `vmat[ia, α, β, μ, ν]`: fxc · ∂ρ_α^A · ∂ρ_β^B contractions
///     using `make_dR_rho1`.
pub fn vxc_deriv2(scf: &SCF, xc_type: XCType) -> Vec<MatrixFull<f64>> {
    // vxc_deriv2 needs up to 2nd derivatives for GGA (or 1st for LDA).
    let ao_deriv = match xc_type { XCType::LDA => 1, XCType::GGA => 2, _ => panic!() };
    let cache = build_vxc_hessian_cache_with_deriv(scf, xc_type, ao_deriv);
    vxc_deriv2_cached(scf, xc_type, &cache)
}

/// Cached variant of `vxc_deriv2`: reuses AO derivatives and ground-state ρ
/// from `cache`. The cache must supply at least the first 10 derivatives
/// for GGA (or first 4 for LDA); `build_vxc_hessian_cache` guarantees this.
pub fn vxc_deriv2_cached(scf: &SCF, xc_type: XCType, cache: &VxcHessianCache) -> Vec<MatrixFull<f64>> {
    let mol = &scf.mol;
    let nao = mol.num_basis;
    let natm = mol.geom.nfree;
    let aoslices = build_aoslices(mol);
    let dm0 = &scf.density_matrix[0];
    let func_ids = &mol.xc_data.dfa_compnt_scf;
    let func_factors = &mol.xc_data.dfa_paramr_scf;

    let nvar = match xc_type {
        XCType::LDA => 1,
        XCType::GGA => 4,
        _ => panic!("vxc_deriv2: only LDA and GGA supported"),
    };

    // Parallel: each block produces (vmat_partial, ipip_partial), then reduce.
    let (mut vmat, ipip): (Vec<MatrixFull<f64>>, MatrixFull<f64>) = cache.blocks
        .par_iter()
        .filter_map(|blk| {
            // Pin OpenBLAS to 1 thread per rayon worker — otherwise the
            // nested OpenMP team competes with rayon (see dft/mod.rs:3747).
            // vxc_deriv2's GGA path issues O(natm × 9 + 9) DGEMMs per block,
            // so the contention overhead here is even worse than in vxc_diag.
            omp_set_num_threads_wrapper(1);
            let nb = blk.nb;
            if nb == 0 { return None; }
            let weights_block = &blk.weights;
            let ao_d: &[MatrixFull<f64>] = &blk.ao_d;
            let rho_array = &blk.rho_array;
            let ao_d_refs: Vec<&MatrixFull<f64>> = ao_d.iter().collect();

            let xc_tensors = eval_xc_eff(func_ids, func_factors, xc_type, 0, rho_array, nb, 2);
            let vxc_t = xc_tensors[1].as_ref().expect("vxc_deriv2: vxc required");
            let fxc_t = xc_tensors[2].as_ref().expect("vxc_deriv2: fxc required");
            let vxc_raw = vxc_t.raw();
            let vxc_off = vxc_t.offset();
            let fxc_raw = fxc_t.raw();
            let fxc_off = fxc_t.offset();

            let mut vmat: Vec<MatrixFull<f64>> =
                (0..natm).map(|_| MatrixFull::new([9 * nao, nao], 0.0)).collect();
            let mut ipip = MatrixFull::new([9 * nao, nao], 0.0);

            // Helper closure to add a (d1, d2) block contribution.
            let add_block = |dst: &mut MatrixFull<f64>, d1: usize, d2: usize,
                             a: &MatrixFull<f64>, b: &MatrixFull<f64>| {
                let row0 = (d1 * 3 + d2) * nao;
                let mut blk = MatrixFull::new([nao, nao], 0.0);
                for j in 0..nao { for i in 0..nao {
                    blk[[i, j]] = dst[[row0 + i, j]];
                }}
                _dgemm_full(a, 'N', b, 'T', &mut blk, 1.0, 1.0);
                for j in 0..nao { for i in 0..nao {
                    dst[[row0 + i, j]] = blk[[i, j]];
                }}
            };

            match xc_type {
                XCType::LDA => {
                    let mut wv = vec![0.0; nb];
                    for g in 0..nb { wv[g] = weights_block[g] * vxc_raw[vxc_off + g]; }
                    let aow_scale: Vec<MatrixFull<f64>> = (0..3)
                        .map(|d| {
                            let mut m = MatrixFull::new([nao, nb], 0.0);
                            for mu in 0..nao { for g in 0..nb {
                                m[[mu, g]] = ao_d[1 + d][[mu, g]] * wv[g];
                            }}
                            m
                        })
                        .collect();
                    for d1 in 0..3 {
                        for d2 in 0..3 {
                            add_block(&mut ipip, d1, d2, &aow_scale[d2], &ao_d[1 + d1]);
                        }
                    }

                    let ao_dm0 = compute_ao_dm0(dm0, &ao_d_refs[..1], 1, nao, nb);
                    let mut wf = vec![0.0; nb];
                    for g in 0..nb { wf[g] = weights_block[g] * fxc_raw[fxc_off + g]; }
                    for ia in 0..natm {
                        let (p0, p1) = aoslices[ia];
                        for d1 in 0..3 {
                            let mut wv_d1 = vec![0.0; nb];
                            for g in 0..nb {
                                let mut rho1 = 0.0;
                                for mu in p0..p1 {
                                    rho1 += ao_d[1 + d1][[mu, g]] * ao_dm0[0][[mu, g]];
                                }
                                wv_d1[g] = wf[g] * 2.0 * rho1;
                            }
                            let mut aow_d1 = MatrixFull::new([nao, nb], 0.0);
                            for mu in 0..nao { for g in 0..nb {
                                aow_d1[[mu, g]] = ao_d[0][[mu, g]] * wv_d1[g];
                            }}
                            for d2 in 0..3 {
                                add_block(&mut vmat[ia], d1, d2, &aow_d1, &ao_d[1 + d2]);
                            }
                        }
                    }
                }
                XCType::GGA => {
                    let mut wv0 = vec![0.0; nb];
                    let mut wv1 = vec![0.0; nb];
                    let mut wv2 = vec![0.0; nb];
                    let mut wv3 = vec![0.0; nb];
                    for g in 0..nb {
                        let w = weights_block[g];
                        wv0[g] = 0.5 * w * vxc_raw[vxc_off + 0 * nb + g];
                        wv1[g] = w * vxc_raw[vxc_off + 1 * nb + g];
                        wv2[g] = w * vxc_raw[vxc_off + 2 * nb + g];
                        wv3[g] = w * vxc_raw[vxc_off + 3 * nb + g];
                    }
                    let aow_diag = make_dR_dao_w(ao_d, &wv0, &wv1, &wv2, &wv3, nao, nb);
                    for d1 in 0..3 {
                        for d2 in 0..3 {
                            add_block(&mut ipip, d1, d2, &aow_diag[d2], &ao_d[1 + d1]);
                        }
                    }

                    let ao_dm0 = compute_ao_dm0(dm0, &ao_d_refs[..nvar], nvar, nao, nb);
                    for ia in 0..natm {
                        let (p0, p1) = aoslices[ia];
                        let dR_rho1 = make_dR_rho1(&ao_d_refs, &ao_dm0, p0, p1, xc_type, nb);
                        let mut wv_block = vec![0.0; 3 * nvar * nb];
                        for g in 0..nb {
                            let wf_g = weights_block[g];
                            for y in 0..nvar {
                                for s in 0..3 {
                                    let mut acc = 0.0;
                                    for x in 0..nvar {
                                        let fxc_xy = fxc_raw[fxc_off + g + x * nb + y * nvar * nb];
                                        let drho_sx = dR_rho1[s + x * 3 + g * 3 * nvar];
                                        acc += wf_g * fxc_xy * drho_sx;
                                    }
                                    let mut val = acc;
                                    if y == 0 { val *= 0.5; }
                                    wv_block[s + y * 3 + g * 3 * nvar] = val;
                                }
                            }
                        }

                        for i in 0..3 {
                            let wv_i0: Vec<f64> = (0..nb).map(|g| wv_block[i + 0 * 3 + g * 3 * nvar]).collect();
                            let wv_i1: Vec<f64> = (0..nb).map(|g| wv_block[i + 1 * 3 + g * 3 * nvar]).collect();
                            let wv_i2: Vec<f64> = (0..nb).map(|g| wv_block[i + 2 * 3 + g * 3 * nvar]).collect();
                            let wv_i3: Vec<f64> = (0..nb).map(|g| wv_block[i + 3 * 3 + g * 3 * nvar]).collect();
                            let aow_i = make_dR_dao_w(ao_d, &wv_i0, &wv_i1, &wv_i2, &wv_i3, nao, nb);
                            for d2 in 0..3 {
                                add_block(&mut vmat[ia], i, d2, &aow_i[d2], &ao_d[0]);
                            }
                        }

                        for d1 in 0..3 {
                            let mut aow_d1 = MatrixFull::new([nao, nb], 0.0);
                            for mu in 0..nao { for g in 0..nb {
                                let mut v = 0.0;
                                for y in 0..nvar {
                                    v += ao_d[y][[mu, g]] * wv_block[d1 + y * 3 + g * 3 * nvar];
                                }
                                aow_d1[[mu, g]] = v;
                            }}
                            for d2 in 0..3 {
                                add_block(&mut vmat[ia], d1, d2, &ao_d[1 + d2], &aow_d1);
                            }
                        }
                    }
                }
                _ => unreachable!(),
            }
            Some((vmat, ipip))
        })
        .reduce(
            || (
                (0..natm).map(|_| MatrixFull::new([9 * nao, nao], 0.0)).collect::<Vec<_>>(),
                MatrixFull::new([9 * nao, nao], 0.0),
            ),
            |(mut a_vmat, mut a_ipip), (b_vmat, b_ipip)| {
                for k in 0..natm { a_vmat[k] += b_vmat[k].clone(); }
                a_ipip += b_ipip.clone();
                (a_vmat, a_ipip)
            },
        );

    // ── Scatter ipip into each atom's AO column slice ──
    let needs_transpose = matches!(xc_type, XCType::GGA);
    for ia in 0..natm {
        let (p0, p1) = aoslices[ia];
        for d1 in 0..3 {
            for d2 in 0..3 {
                let row0 = (d1 * 3 + d2) * nao;
                let row0_T = (d2 * 3 + d1) * nao;
                for mu in 0..nao {
                    for nu in p0..p1 {
                        let mut inc = ipip[[row0 + mu, nu]];
                        if needs_transpose {
                            inc += ipip[[row0_T + nu, mu]];
                        }
                        vmat[ia][[row0 + mu, nu]] += inc;
                    }
                }
            }
        }
    }

    vmat
}



