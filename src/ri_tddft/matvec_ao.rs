//! AO-basis matrix-vector products for TDDFT (`tddft_mode = "ao"`)
//!
//! The MO-based path (`matvec.rs`) pre-transforms three-center RI integrals
//! into the MO basis, costing O(naux·(nocc·nvir + nocc² + nvir²)) memory.
//! This AO path keeps the Davidson iteration in the MO amplitude space but
//! evaluates every kernel contribution from the **AO transition density**:
//!
//! P_z = C_occ · z_mat · C_vir^T
//!
//! - Coulomb:  F_J[μν] = Σ_Q B_{μν,Q} · (Σ_λσ B_{λσ,Q} P_λσ)
//!   evaluated with two DGEMVs on the folded-pair RI tensor; correct for
//!   non-symmetric transition densities (2×off-diagonal + 1×diagonal rule).
//! - Exchange: F_K[μν] = Σ_Q Σ_λσ P_λσ B_{μλ,Q} B_{νσ,Q} = Σ_Q M_Q·P·M_Q
//!   where M_Q is the full symmetric expansion of aux column Q.
//!   A-block uses P; B-block uses Pᵀ:
//!   C_occᵀ·K[Pᵀ]·C_vir gives K_B[ia] = Σ_jb (ib|aj) z_jb.
//! - fxc:      ρ_z(g) = Σ_μν P_μν φ_μ(g)φ_ν(g) → ×wfxc[g] → contract back
//!   with AO values on grids (LDA nvar=1 / GGA nvar=4).
//!
//! Final projection back to MO amplitudes: result = C_occᵀ · F · C_vir.

use rest_tensors::{MatrixFull, RIFull};
use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;
use rest_tensors::matrixupper::map_upper_to_full;
use rayon::prelude::*;

use crate::scf_io::SCF;
use crate::dft::num_int::FXCMatvecData;
use crate::dft::Grids;
use crate::dft::numint_matmul::nimatmul::NIMatmul;
use crate::dft::numint_matmul::hess_rks::eval_vxc_fxc_from_rho;
use crate::dft::xceff::prelude::{XCDenType, XCSpin};
use crate::ri_jk::util::get_cint_mol;
use crate::ri_tddft::utils::tddft_occupation_parameters;
use crate::ri_tddft::{TDDFTData, TDDFTMode};
use crate::utilities::rstsr_util::{RestTensorToRstsrTsrAPI, RestTensorToRstsrViewAPI, Tsr, TsrView};
use rstsr::prelude::*;
use libxc::prelude::*;

/// Type alias matching `scf.rimatr`:
/// (ri3fn folded [npair, naux], basbas2baspar map, baspar2basbas list)
pub type RimatrTuple = Option<(MatrixFull<f64>, MatrixFull<usize>, Vec<[usize; 2]>)>;


// ══════════════════════════════════════════════════════════════════
// Core contractions
// ══════════════════════════════════════════════════════════════════

/// Transition density P = C_occ · z_mat · C_virᵀ with z[i + a*occ].
pub fn transition_density(
    c_occ: &MatrixFull<f64>,
    c_vir: &MatrixFull<f64>,
    z: &[f64],
    nao: usize,
    occ_size: usize,
    vir_size: usize,
) -> MatrixFull<f64> {
    let z_mat = MatrixFull::from_vec([occ_size, vir_size], z.to_vec()).unwrap();
    let mut t1 = MatrixFull::new([nao, vir_size], 0.0);
    _dgemm_full(c_occ, 'N', &z_mat, 'N', &mut t1, 1.0, 0.0);
    let mut p = MatrixFull::new([nao, nao], 0.0);
    _dgemm_full(&t1, 'N', c_vir, 'T', &mut p, 1.0, 0.0);
    p
}

/// Batched Coulomb (J) over all trial vectors: $J[D^{\mathbb{A}}]$ for
/// $\mathbb{A} = 0..m$ in one ri_jk call, amortizing the cderi unpack +
/// `to_rstsr` conversions across nset.
///
/// Returns `[nao*nao, m]` with column $\mathbb{A}$ = flattened J (column-major
/// `[nao,nao]`), matching the `f_fxc_block` layout (`base = \mathbb{A}·nao·nao`).
fn get_j_ao_batched(scf: &SCF, p_block: &[MatrixFull<f64>]) -> MatrixFull<f64> {
    let device = DeviceBLAS::default();
    let (ri3fn, _, _) = scf.rimatr.as_ref()
        .expect("rimatr must be initialized for AO-mode TDDFT");
    let cderi = ri3fn.to_rstsr_view(&device);
    let dms_slice: &[MatrixFull<f64>] = p_block;
    let dms = dms_slice.to_rstsr(&device);               // [nao, nao, m]
    let nao = scf.mol.num_basis;
    let js = crate::ri_jk::pure_incore::get_vj_ri_incore_nonsym(cderi, dms.view()); // [nao,nao,m]
    let mut out = MatrixFull::new([nao * nao, p_block.len()], 0.0);
    for s in 0..p_block.len() {
        for (r, v) in js.i((.., .., s)).iter().enumerate() {
            out.data[s * nao * nao + r] = *v;
        }
    }
    out
}

/// Batched Exchange (K) over all trial vectors. Exact route: one
/// `get_vk_ri_incore_dm` over the whole `[nao,nao,m]` block. Low-rank route
/// (SVD is per-vector, so it cannot be batched): loops per vector and stacks.
///
/// Returns `[nao*nao, m]` (column $\mathbb{A}$ = flattened K, matching
/// `f_fxc_block`). `swap` selects the B-block form $K[D^{\mathbb{A}\top}]$: the
/// exact route uses the transposed `p_block` directly, and the low-rank route
/// swaps occ/vir coefficients and takes the transposed amplitude (since
/// $D^{\mathrm{T}} = C^{\mathrm{vir}} z^{\mathrm{T}} (C^{\mathrm{occ}})^{\mathrm{T}}$).
fn get_k_ao_batched(
    scf: &SCF,
    ao_data: &TDDFTData,
    z_block: &MatrixFull<f64>,
    p_block: &[MatrixFull<f64>],
    swap: bool,
) -> MatrixFull<f64> {
    let tddft_ctrl = scf.mol.ctrl.tddft.as_ref();
    let lowrank_k = tddft_ctrl.map_or(false, |t| t.tddft_lowrank_k);
    let svd_tol = tddft_ctrl.map_or(1.0e-6, |t| t.tddft_svd_tol);
    let nao = scf.mol.num_basis;
    let m = p_block.len();
    let device = DeviceBLAS::default();
    let (ri3fn, _, _) = scf.rimatr.as_ref()
        .expect("rimatr must be initialized for AO-mode TDDFT");
    let cderi = ri3fn.to_rstsr_view(&device);
    let naux = ri3fn.size[1];

    let mut out = MatrixFull::new([nao * nao, m], 0.0);
    if !lowrank_k {
        // Exact: one batched K over all trial vectors.
        let dms_slice: &[MatrixFull<f64>] = p_block;
        let dms = dms_slice.to_rstsr(&device);            // [nao,nao,m]
        let ks = crate::ri_jk::pure_incore::get_vk_ri_incore_dm(cderi, dms.view(), naux);
        for s in 0..m {
            for (r, v) in ks.i((.., .., s)).iter().enumerate() {
                out.data[s * nao * nao + r] = *v;
            }
        }
    } else {
        // Low-rank: per-vector SVD (cannot batch the SVD truncation).
        let occ_size = ao_data.c_occ.as_ref().unwrap().size[1];
        let vir_size = ao_data.c_vir.as_ref().unwrap().size[1];
        let c_occ_v = ao_data.c_occ.as_ref().unwrap().to_rstsr_view(&device);
        let c_vir_v = ao_data.c_vir.as_ref().unwrap().to_rstsr_view(&device);
        for s in 0..m {
            let z_col: Vec<f64> = (0..z_block.size[0]).map(|r| z_block[[r, s]]).collect();
            let (n1, n2) = if swap { (vir_size, occ_size) } else { (occ_size, vir_size) };
            let z_mat = MatrixFull::from_vec([n1, n2], z_col).unwrap();
            let (c_left, c_right) = if swap { (c_vir_v.view(), c_occ_v.view()) } else { (c_occ_v.view(), c_vir_v.view()) };
            let k2 = crate::ri_jk::pure_incore::get_vk_ri_incore_dm_lowrank(
                cderi.view(),
                c_left,
                c_right,
                z_mat.to_rstsr_view(&device),
                svd_tol,
                naux,
            );
            for (r, v) in k2.iter().enumerate() {
                out.data[s * nao * nao + r] = *v;
            }
        }
    }
    out
}

/// Project an AO-basis matrix back to MO amplitudes: r[i,a] = C_occᵀ·F·C_vir.
fn contract_back(
    f_full: &MatrixFull<f64>,
    c_occ: &MatrixFull<f64>,
    c_vir: &MatrixFull<f64>,
    occ_size: usize,
    vir_size: usize,
) -> Vec<f64> {
    let nao = f_full.size[0];
    let mut tmp = MatrixFull::new([occ_size, nao], 0.0);
    _dgemm_full(c_occ, 'T', f_full, 'N', &mut tmp, 1.0, 0.0);
    let mut r = MatrixFull::new([occ_size, vir_size], 0.0);
    _dgemm_full(&tmp, 'N', c_vir, 'N', &mut r, 1.0, 0.0);
    r.data
}


// ══════════════════════════════════════════════════════════════════
// Public A/B matvecs (drop-in replacements for matvec.rs versions)
// ══════════════════════════════════════════════════════════════════

/// Full A-block matvec in AO mode, applied to a block of trial vectors.
///
/// `z_block` is `[dim, m]`; returns `[dim, m]`. The fxc term is batched via
/// `NIMatmul` (all m vectors in one pass); RI-J/K stay per-vector.
/// Symmetrize a transition density: P_sym = (P + Pᵀ)/2.
fn symmetrize_density(p: &MatrixFull<f64>) -> MatrixFull<f64> {
    let [n, _] = p.size;
    let mut p_sym = MatrixFull::new([n, n], 0.0);
    for i in 0..n {
        for j in 0..n {
            p_sym[[i, j]] = 0.5 * (p[[i, j]] + p[[j, i]]);
        }
    }
    p_sym
}

/// Batched AO-mode fxc kernel over a block of transition densities via
/// NIMatmul (`make_rho_from_dm` + `make_fxc_pot_with_eff`, PySCF `vind(zs)`
/// style). When `grid_batch` is set, the evaluation runs over grid batches
/// (`split_batch`) so the full `[ngrids, nao, ncomp]` AO cache is never
/// materialized.
///
/// Returns `[nao, nao*m]` with column block $\mathbb{A}$ at
/// `data[\mathbb{A}·nao·nao ..]` (matching the assembly loops' `base`).
fn fxc_matvec_ao_batched(
    ao_data: &mut TDDFTData,
    p_block: &[MatrixFull<f64>],
    device: &DeviceBLAS,
) -> MatrixFull<f64> {
    use crate::dft::xceff::prelude::XCSpin;
    let nao = p_block[0].size[0];
    let m = p_block.len();
    let ngrids = ao_data.fxc.ngrids;
    let den_type = ao_data.den_type.expect("AO-mode fxc requires den_type");

    let mut out = MatrixFull::new([nao, nao * m], 0.0);
    // make_rho_from_dm assumes a symmetric density for its SIGMA response;
    // symmetrize the transition densities first (exact — the fxc kernel is
    // symmetric in μν so the fxc response only sees the symmetric part).
    let p_sym_block: Vec<MatrixFull<f64>> = p_block.iter().map(symmetrize_density).collect();
    let dms_slice: &[MatrixFull<f64>] = &p_sym_block;
    let dms = dms_slice.to_rstsr(device); // [nao, nao, m]
    let fxc_eff_view = ao_data.fxc_eff.as_ref().unwrap().view();

    let ni = ao_data.ni.as_mut().expect("AO-mode fxc requires NIMatmul");
    if !ao_data.grid_batch {
        // Full-grid path: one batched call over all sets.
        let dm_views: Vec<TsrView> = (0..m).map(|s| dms.i((.., .., s))).collect();
        let rho1 = ni.make_rho_from_dm(&dm_views, den_type); // [ngrids, nvar, m]
        let f_fxc = ni.make_fxc_pot_with_eff(fxc_eff_view, rho1.view(), den_type, XCSpin::Unpolarized);
        for s in 0..m {
            for (r, v) in f_fxc.i((.., .., s)).iter().enumerate() {
                out.data[s * nao * nao + r] = *v;
            }
        }
    } else {
        // Grid-batched path: evaluate per grid batch (all sets), accumulating.
        let nbatch = ni.nbatch;
        for start in (0..ngrids).step_by(nbatch) {
            let end = (start + nbatch).min(ngrids);
            let mut ni_batch = ni.split_batch(start, end);
            let dm_views: Vec<TsrView> = (0..m).map(|s| dms.i((.., .., s))).collect();
            let rho1_b = ni_batch.make_rho_from_dm(&dm_views, den_type); // [nb, nvar, m]
            let fxc_eff_b = fxc_eff_view.i((start..end));
            let f_b = ni_batch.make_fxc_pot_with_eff(fxc_eff_b, rho1_b.view(), den_type, XCSpin::Unpolarized);
            for s in 0..m {
                for (r, v) in f_b.i((.., .., s)).iter().enumerate() {
                    out.data[s * nao * nao + r] += *v;
                }
            }
        }
    }
    out
}

/// Kernel part of the A-block matvec (everything except the diagonal):
/// transition densities → batched RI-J/K + fxc → contract back to MO.
fn ao_a_kernel_block(
    scf: &SCF,
    ao_data: &mut TDDFTData,
    z_block: &MatrixFull<f64>,
    xlet: char,
) -> MatrixFull<f64> {
    let alpha_hybrid = ao_data.fxc.alpha_hybrid;
    let (_start_mo, _, occ_size, vir_size, _homo, _lumo) = tddft_occupation_parameters(scf);
    let dim = occ_size * vir_size;
    let m = z_block.size[1];
    let nao = scf.mol.num_basis;
    let coulomb_factor = if xlet == 'S' { 2.0 } else if xlet == 'R' { 1.0 } else { 0.0 };

    let mut result = MatrixFull::new([dim, m], 0.0);

    // Build P for each column, and batch the fxc over all columns.
    let device = DeviceBLAS::default();
    let mut p_block: Vec<MatrixFull<f64>> = Vec::with_capacity(m);
    for s in 0..m {
        let z: Vec<f64> = (0..dim).map(|r| z_block[[r, s]]).collect();
        p_block.push(transition_density(ao_data.c_occ.as_ref().unwrap(), ao_data.c_vir.as_ref().unwrap(), &z, nao, occ_size, vir_size));
    }
    let f_fxc_block = fxc_matvec_ao_batched(ao_data, &p_block, &device);

    // Batched RI-J/K over all trial vectors (single ri_jk call).
    let j_block = if coulomb_factor != 0.0 {
        Some(get_j_ao_batched(scf, &p_block))
    } else {
        None
    };
    let k_block = if alpha_hybrid.abs() > 1e-15 {
        Some(get_k_ao_batched(scf, ao_data, z_block, &p_block, false))
    } else {
        None
    };

    // Per-column assembly + contract back
    for s in 0..m {
        let base = s * nao * nao;
        let mut f_total = MatrixFull::new([nao, nao], 0.0);
        if let Some(jb) = &j_block {
            for idx in 0..nao * nao {
                f_total.data[idx] += coulomb_factor * jb.data[base + idx];
            }
        }
        if let Some(kb) = &k_block {
            for idx in 0..nao * nao {
                f_total.data[idx] -= alpha_hybrid * kb.data[base + idx];
            }
        }
        // fxc contribution for this vector (column s of the batched fxc block)
        for idx in 0..nao * nao {
            f_total.data[idx] += f_fxc_block.data[base + idx];
        }
        let kernel_mo = contract_back(&f_total, ao_data.c_occ.as_ref().unwrap(), ao_data.c_vir.as_ref().unwrap(), occ_size, vir_size);
        for r in 0..dim {
            result[[r, s]] += kernel_mo[r];
        }
    }
    result
}

pub fn a_matvec_ao_batched(
    scf: &SCF,
    ao_data: &mut TDDFTData,
    z_block: &MatrixFull<f64>,
    xlet: char,
) -> MatrixFull<f64> {
    let (start_mo, _, occ_size, vir_size, _homo, lumo) = tddft_occupation_parameters(scf);
    let dim = occ_size * vir_size;
    let m = z_block.size[1];
    let ks = &scf.eigenvalues[0];

    let mut result = ao_a_kernel_block(scf, ao_data, z_block, xlet);

    // Diagonal contribution (elementwise on the block)
    for s in 0..m {
        for a in 0..vir_size {
            for i in 0..occ_size {
                let idx = i + a * occ_size;
                result[[idx, s]] += (ks[lumo + a] - ks[start_mo + i]) * z_block[[idx, s]];
            }
        }
    }
    result
}

/// Build the full A matrix `[dim, dim]` directly (dense small-system path).
///
/// $A_{ia,jb} = (\varepsilon_a - \varepsilon_i)\delta_{ij}\delta_{ab} + \text{kernel}$,
/// constructed by applying the AO kernel block to the identity — one batched
/// J/K/fxc call across all `dim` columns instead of per-vector matvecs.
pub fn build_a_ao(scf: &SCF, ao_data: &mut TDDFTData, xlet: char) -> MatrixFull<f64> {
    let (start_mo, _, occ_size, vir_size, _homo, lumo) = tddft_occupation_parameters(scf);
    let dim = occ_size * vir_size;
    let ks = &scf.eigenvalues[0];

    // Identity block: each column is one unit amplitude vector e_(ia).
    let mut identity = MatrixFull::new([dim, dim], 0.0);
    for idx in 0..dim {
        identity[[idx, idx]] = 1.0;
    }

    let mut a_full = ao_a_kernel_block(scf, ao_data, &identity, xlet);

    // Diagonal matrix (ε_a − ε_i).
    for a in 0..vir_size {
        for i in 0..occ_size {
            a_full[[i + a * occ_size, i + a * occ_size]] += ks[lumo + a] - ks[start_mo + i];
        }
    }
    a_full
}

/// Full B-block matvec in AO mode, applied to a block of trial vectors.
///
/// `z_block` is `[dim, m]`; returns `[dim, m]`. The fxc term is batched via
/// `NIMatmul`; RI-J/K are batched (A-block uses D, B-block uses the transposed
/// density Dᵀ).
/// Kernel part of the B-block matvec (the B block has no diagonal).
fn ao_b_kernel_block(
    scf: &SCF,
    ao_data: &mut TDDFTData,
    z_block: &MatrixFull<f64>,
    xlet: char,
) -> MatrixFull<f64> {
    let alpha_hybrid = ao_data.fxc.alpha_hybrid;
    let (_start_mo, _, occ_size, vir_size, _homo, _lumo) = tddft_occupation_parameters(scf);
    let dim = occ_size * vir_size;
    let m = z_block.size[1];
    let nao = scf.mol.num_basis;
    let coulomb_factor = if xlet == 'S' { 2.0 } else if xlet == 'R' { 1.0 } else { 0.0 };

    let mut result = MatrixFull::new([dim, m], 0.0);

    let device = DeviceBLAS::default();
    let mut p_block: Vec<MatrixFull<f64>> = Vec::with_capacity(m);
    for s in 0..m {
        let z: Vec<f64> = (0..dim).map(|r| z_block[[r, s]]).collect();
        p_block.push(transition_density(ao_data.c_occ.as_ref().unwrap(), ao_data.c_vir.as_ref().unwrap(), &z, nao, occ_size, vir_size));
    }
    let f_fxc_block = fxc_matvec_ao_batched(ao_data, &p_block, &device);

    // Batched RI-J (J uses D; symmetric in μν after unfold so no transpose
    // needed) and exchange (B-block uses Pᵀ, so a transposed block is passed).
    let j_block = if coulomb_factor != 0.0 {
        Some(get_j_ao_batched(scf, &p_block))
    } else {
        None
    };
    let k_block = if alpha_hybrid.abs() > 1e-15 {
        // B-block: K[Pᵀ] with the transposed amplitude matrix zᵀ.
        let p_block_t: Vec<MatrixFull<f64>> =
            p_block.iter().map(|p| p.clone().transpose_and_drop()).collect();
        let mut z_block_t = MatrixFull::new([dim, m], 0.0);
        for s in 0..m {
            for i in 0..occ_size {
                for a in 0..vir_size {
                    z_block_t[[a + i * vir_size, s]] = z_block[[i + a * occ_size, s]];
                }
            }
        }
        Some(get_k_ao_batched(scf, ao_data, &z_block_t, &p_block_t, true))
    } else {
        None
    };

    for s in 0..m {
        let base = s * nao * nao;
        let mut f_total = MatrixFull::new([nao, nao], 0.0);
        if let Some(jb) = &j_block {
            for idx in 0..nao * nao {
                f_total.data[idx] += coulomb_factor * jb.data[base + idx];
            }
        }
        if let Some(kb) = &k_block {
            for idx in 0..nao * nao {
                f_total.data[idx] -= alpha_hybrid * kb.data[base + idx];
            }
        }
        for idx in 0..nao * nao {
            f_total.data[idx] += f_fxc_block.data[base + idx];
        }
        let kernel_mo = contract_back(&f_total, ao_data.c_occ.as_ref().unwrap(), ao_data.c_vir.as_ref().unwrap(), occ_size, vir_size);
        for r in 0..dim {
            result[[r, s]] += kernel_mo[r];
        }
    }
    result
}

pub fn b_matvec_ao_batched(
    scf: &SCF,
    ao_data: &mut TDDFTData,
    z_block: &MatrixFull<f64>,
    xlet: char,
) -> MatrixFull<f64> {
    ao_b_kernel_block(scf, ao_data, z_block, xlet)
}

/// Build the full B matrix `[dim, dim]` directly (dense small-system path),
/// by applying the B kernel block to the identity.
pub fn build_b_ao(scf: &SCF, ao_data: &mut TDDFTData, xlet: char) -> MatrixFull<f64> {
    let (_start_mo, _, occ_size, vir_size, _homo, _lumo) = tddft_occupation_parameters(scf);
    let dim = occ_size * vir_size;
    let mut identity = MatrixFull::new([dim, dim], 0.0);
    for idx in 0..dim {
        identity[[idx, idx]] = 1.0;
    }
    ao_b_kernel_block(scf, ao_data, &identity, xlet)
}

// ══════════════════════════════════════════════════════════════════
// Tests: validate AO contractions against naive four-index references
// ══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;


    /// Column-wise scaling of a [nao, ngrids] matrix: out[:, g] *= v[g].
    fn scale_columns(mat: &MatrixFull<f64>, v: &[f64]) -> MatrixFull<f64> {
        let [nr, nc] = mat.size;
        let mut out = mat.clone();
        for g in 0..nc {
            let s = v[g];
            for r in 0..nr {
                out[[r, g]] *= s;
            }
        }
        out
    }

    /// Column-wise dot products between two [nao, ngrids] matrices → [ngrids].
    fn column_dots(a: &MatrixFull<f64>, b: &MatrixFull<f64>) -> Vec<f64> {
        let [nr, nc] = a.size;
        (0..nc).map(|g| {
            (0..nr).map(|r| a[[r, g]] * b[[r, g]]).sum::<f64>()
        }).collect()
    }

    /// fxc contribution in AO form: returns the full AO matrix F_fxc[μν].
    ///
    /// ρ_z(g) built from the transition density P via AO values on grids,
    /// kernel applied through wfxc (weights included), then contracted back.
    pub fn fxc_matvec_ao(ao_data: &TDDFTData, p: &MatrixFull<f64>) -> MatrixFull<f64> {
        match ao_data.fxc.nvar {
            1 => fxc_ao_lda(ao_data, p),
            4 => fxc_ao_gga(ao_data, p),
            n => panic!("fxc_matvec_ao only supports LDA (nvar=1) and GGA (nvar=4); got {}", n),
        }
    }

    /// LDA (nvar=1): ρ_z[g] = Σ_μν P_μν φ_μ(g)φ_ν(g); F = Σ_g v[g] φ⊗φ.
    fn fxc_ao_lda(ao_data: &TDDFTData, p: &MatrixFull<f64>) -> MatrixFull<f64> {
        let ng = ao_data.fxc.ngrids;
        let ao = ao_data.ao.as_ref().expect("AO on grids required for the per-vector fxc path");

        // X = P · AO  →  ρ_z[g] = column_dot(AO[:,g], X[:,g])
        let mut x = MatrixFull::new([p.size[0], ng], 0.0);
        _dgemm_full(p, 'N', ao, 'N', &mut x, 1.0, 0.0);
        let rho_z = column_dots(ao, &x);

        let v: Vec<f64> = rho_z.iter().zip(ao_data.fxc.wfxc.iter())
            .map(|(r, w)| r * w).collect();

        // F = (AO scaled by v) · AOᵀ
        let ao_s = scale_columns(ao, &v);
        let mut f = MatrixFull::new([ao.size[0], ao.size[0]], 0.0);
        _dgemm_full(&ao_s, 'N', ao, 'T', &mut f, 1.0, 0.0);
        f
    }

    /// GGA (nvar=4): perturbed density has components
    /// ρ⁰ = ΣP φφ,  ρ^{d+1} = ΣP (∂_dφ_μ·φ_ν + φ_μ·∂_dφ_ν);
    /// kernel applied through wfxc[g, α, β], then contracted back per component.
    fn fxc_ao_gga(ao_data: &TDDFTData, p: &MatrixFull<f64>) -> MatrixFull<f64> {
        let ng = ao_data.fxc.ngrids;
        let nao = p.size[0];
        let ao = ao_data.ao.as_ref().expect("AO on grids required for the per-vector fxc path");
        let grads = ao_data.ao_grad.as_ref().expect("GGA requires AO gradients on grids");

        // Perturbed quantities on grids: x0 = P·AO, xd = P·AOgrad[d]
        let mut x0 = MatrixFull::new([nao, ng], 0.0);
        _dgemm_full(p, 'N', ao, 'N', &mut x0, 1.0, 0.0);
        let mut xd = [
            MatrixFull::new([nao, ng], 0.0),
            MatrixFull::new([nao, ng], 0.0),
            MatrixFull::new([nao, ng], 0.0),
        ];
        for d in 0..3 {
            let mut m = MatrixFull::new([nao, ng], 0.0);
            _dgemm_full(p, 'N', &grads[d], 'N', &mut m, 1.0, 0.0);
            xd[d] = m;
        }

        // ρ components: β=0 and β=d+1
        let mut rho: [Vec<f64>; 4] = Default::default();
        rho[0] = column_dots(ao, &x0);
        for d in 0..3 {
            let part1 = column_dots(&grads[d], &x0);
            let part2 = column_dots(ao, &xd[d]);
            rho[d + 1] = part1.into_iter().zip(part2.into_iter()).map(|(a, b)| a + b).collect();
        }

        // Kernel application: feff[α][g] = Σ_β wfxc[g + α·ng + β·4ng] · ρ^β[g]
        let wfxc = &ao_data.fxc.wfxc;
        let mut feff: [Vec<f64>; 4] = Default::default();
        for alpha in 0..4 {
            let mut col = vec![0.0_f64; ng];
            for g in 0..ng {
                let mut s = 0.0;
                for beta in 0..4 {
                    s += wfxc[g + alpha * ng + beta * 4 * ng] * rho[beta][g];
                }
                col[g] = s;
            }
            feff[alpha] = col;
        }

        // Contract each kernel component back to an AO matrix:
        //   α=0:      F += S(feff₀, AO) · AOᵀ
        //   α=d+1:    F += S(feff_α, AOgrad[d]) · AOᵀ + S(feff_α, AO) · AOgrad[d]ᵀ
        let mut f = MatrixFull::new([nao, nao], 0.0);
        let ao_s0 = scale_columns(ao, &feff[0]);
        _dgemm_full(&ao_s0, 'N', ao, 'T', &mut f, 1.0, 1.0);
        for d in 0..3 {
            let alpha = d + 1;
            let gd_s = scale_columns(&grads[d], &feff[alpha]);
            _dgemm_full(&gd_s, 'N', ao, 'T', &mut f, 1.0, 1.0);
            let ao_sa = scale_columns(ao, &feff[alpha]);
            _dgemm_full(&ao_sa, 'N', &grads[d], 'T', &mut f, 1.0, 1.0);
        }
        f
    }
    /// Deterministic pseudo-random fill (mirrors matvec.rs tests).
    fn pseudo(n: usize, seed: f64) -> Vec<f64> {
        (0..n).map(|i| ((i as f64 + seed) * 0.37 + seed).sin() * 0.5).collect()
    }

    /// Build a synthetic rimatr: nao=6 → npair=21, naux=4, random folded ri3fn.
    fn synthetic_rimatr(nao: usize, naux: usize) -> RimatrTuple {
        let npair = nao * (nao + 1) / 2;
        let ri3fn = MatrixFull::from_vec([npair, naux], pseudo(npair * naux, 1.7)).unwrap();
        let basbas2baspar = MatrixFull::from_vec([nao, 1], vec![0usize; nao]).unwrap();
        Some((ri3fn, basbas2baspar, vec![]))
    }

    /// Unfold a folded upper-triangle column into a full symmetric matrix.
    fn unfold(m: &[f64], nao: usize) -> MatrixFull<f64> {
        let npair = m.len();
        let index_map = map_upper_to_full(npair).unwrap();
        let mut mq = MatrixFull::new([nao, nao], 0.0);
        for (k, ij) in index_map.data.iter().enumerate() {
            let v = m[k];
            mq[[ij[0], ij[1]]] = v;
            mq[[ij[1], ij[0]]] = v;
        }
        mq
    }

    /// Reconstruct the full four-index integral (μν|λσ) = Σ_Q B_μνQ B_λσQ.
    fn four_index_integrals(rimatr: &RimatrTuple) -> Vec<Vec<Vec<Vec<f64>>>> {
        let (ri3fn, basbas2baspar, _) = rimatr.as_ref().unwrap();
        let nao = basbas2baspar.size[0];
        let naux = ri3fn.size[1];
        let ms: Vec<MatrixFull<f64>> = (0..naux)
            .map(|q| unfold(&ri3fn.iter_columns_full().nth(q).unwrap().to_vec(), nao))
            .collect();
        let mut eri = vec![vec![vec![vec![0.0; nao]; nao]; nao]; nao];
        for mu in 0..nao { for nu in 0..nao { for lam in 0..nao { for sig in 0..nao {
            let mut s = 0.0;
            for q in 0..naux {
                s += ms[q][[mu, nu]] * ms[q][[lam, sig]];
            }
            eri[mu][nu][lam][sig] = s;
        }}}}
        eri
    }

    #[test]
    fn test_transition_density() {
        let nao = 6; let occ = 3; let vir = 4;
        let c_occ = MatrixFull::from_vec([nao, occ], pseudo(nao * occ, 2.1)).unwrap();
        let c_vir = MatrixFull::from_vec([nao, vir], pseudo(nao * vir, 3.3)).unwrap();
        let z: Vec<f64> = pseudo(occ * vir, 4.4);
        let p = transition_density(&c_occ, &c_vir, &z, nao, occ, vir);
        // naive reference
        for mu in 0..nao { for nu in 0..nao {
            let mut s = 0.0;
            for i in 0..occ { for a in 0..vir {
                s += c_occ[[mu, i]] * c_vir[[nu, a]] * z[i + a * occ];
            }}
            assert!((p[[mu, nu]] - s).abs() < 1e-12, "P[{},{}] = {} vs {}", mu, nu, p[[mu, nu]], s);
        }}
    }

    #[test]
    fn test_ri_coulomb_ao_vs_naive() {
        let nao = 6; let naux = 4;
        let rimatr = synthetic_rimatr(nao, naux);
        let eri = four_index_integrals(&rimatr);
        // Non-symmetric transition density
        let p = MatrixFull::from_vec([nao, nao], pseudo(nao * nao, 5.5)).unwrap();
        let device = DeviceBLAS::default();
        let (ri3fn, _, _) = rimatr.as_ref().unwrap();
        let cderi = ri3fn.to_rstsr_view(&device);
        let dms = vec![p.clone()].as_slice().to_rstsr(&device);
        let js = crate::ri_jk::pure_incore::get_vj_ri_incore_nonsym(cderi, dms.view());
        let f = MatrixFull::from_vec([nao, nao], js.i((.., .., 0)).iter().copied().collect()).unwrap();
        for mu in 0..nao { for nu in 0..nao {
            let mut s = 0.0;
            for lam in 0..nao { for sig in 0..nao {
                s += p[[lam, sig]] * eri[mu][nu][lam][sig];
            }}
            assert!((f[[mu, nu]] - s).abs() < 1e-10,
                "J[{},{}] = {} vs {}", mu, nu, f[[mu, nu]], s);
        }}
    }

    #[test]
    fn test_ri_exchange_ao_vs_naive() {
        let nao = 6; let naux = 4;
        let rimatr = synthetic_rimatr(nao, naux);
        let eri = four_index_integrals(&rimatr);
        let p = MatrixFull::from_vec([nao, nao], pseudo(nao * nao, 6.6)).unwrap();
        let device = DeviceBLAS::default();
        let (ri3fn, _, _) = rimatr.as_ref().unwrap();
        let cderi = ri3fn.to_rstsr_view(&device);
        let dms = vec![p.clone()].as_slice().to_rstsr(&device);
        let ks = crate::ri_jk::pure_incore::get_vk_ri_incore_dm(cderi, dms.view(), 2);
        let k = MatrixFull::from_vec([nao, nao], ks.i((.., .., 0)).iter().copied().collect()).unwrap();
        // K[μν] = Σ_λσ P_λσ (μλ|σν)
        for mu in 0..nao { for nu in 0..nao {
            let mut s = 0.0;
            for lam in 0..nao { for sig in 0..nao {
                s += p[[lam, sig]] * eri[mu][lam][sig][nu];
            }}
            assert!((k[[mu, nu]] - s).abs() < 1e-10,
                "K[{},{}] = {} vs {}", mu, nu, k[[mu, nu]], s);
        }}
    }

    /// Reference LDA fxc from the textbook MO formula:
    /// ρ_z[g] = Σ_ia z_ia φ_i φ_a; result[i,a] = Σ_g wfxc ρ_z φ_i φ_a.
    fn fxc_lda_reference(data: &TDDFTData, p: &MatrixFull<f64>, z: &[f64]) -> Vec<f64> {
        let ng = data.fxc.ngrids;
        let c_occ = data.c_occ.as_ref().unwrap();
        let c_vir = data.c_vir.as_ref().unwrap();
        let occ = c_occ.size[1];
        let vir = c_vir.size[1];
        let ao = data.ao.as_ref().unwrap();
        // φ_i[g] = Σ_μ C_occ[μ,i] ao[μ,g]; φ_a[g] = Σ_μ C_vir[μ,a] ao[μ,g]
        let mut mo_occ = MatrixFull::new([occ, ng], 0.0);
        _dgemm_full(c_occ, 'T', ao, 'N', &mut mo_occ, 1.0, 0.0);
        let mut mo_vir = MatrixFull::new([vir, ng], 0.0);
        _dgemm_full(c_vir, 'T', ao, 'N', &mut mo_vir, 1.0, 0.0);
        let mut rho = vec![0.0; ng];
        for g in 0..ng {
            let mut s = 0.0;
            for i in 0..occ { for a in 0..vir {
                s += z[i + a * occ] * mo_occ[[i, g]] * mo_vir[[a, g]];
            }}
            rho[g] = s;
        }
        let mut result = vec![0.0; occ * vir];
        for g in 0..ng {
            let v = data.fxc.wfxc[g] * rho[g];
            for i in 0..occ { for a in 0..vir {
                result[i + a * occ] += v * mo_occ[[i, g]] * mo_vir[[a, g]];
            }}
        }
        result
    }

    /// Reference GGA fxc from the textbook MO formula (perturbed gradients).
    fn fxc_gga_reference(data: &TDDFTData, p: &MatrixFull<f64>, z: &[f64]) -> Vec<f64> {
        let ng = data.fxc.ngrids;
        let c_occ = data.c_occ.as_ref().unwrap();
        let c_vir = data.c_vir.as_ref().unwrap();
        let occ = c_occ.size[1];
        let vir = c_vir.size[1];
        let ao = data.ao.as_ref().unwrap();
        let grads = data.ao_grad.as_ref().unwrap();
        // MO values + gradients on grids
        let mut mo_occ = MatrixFull::new([occ, ng], 0.0);
        _dgemm_full(c_occ, 'T', ao, 'N', &mut mo_occ, 1.0, 0.0);
        let mut mo_vir = MatrixFull::new([vir, ng], 0.0);
        _dgemm_full(c_vir, 'T', ao, 'N', &mut mo_vir, 1.0, 0.0);
        let mut mo_occ_g: [MatrixFull<f64>; 3] = [
            MatrixFull::new([occ, ng], 0.0),
            MatrixFull::new([occ, ng], 0.0),
            MatrixFull::new([occ, ng], 0.0),
        ];
        let mut mo_vir_g: [MatrixFull<f64>; 3] = [
            MatrixFull::new([vir, ng], 0.0),
            MatrixFull::new([vir, ng], 0.0),
            MatrixFull::new([vir, ng], 0.0),
        ];
        for d in 0..3 {
            let mut o = MatrixFull::new([occ, ng], 0.0);
            _dgemm_full(data.c_occ.as_ref().unwrap(), 'T', &grads[d], 'N', &mut o, 1.0, 0.0);
            mo_occ_g[d] = o;
            let mut v = MatrixFull::new([vir, ng], 0.0);
            _dgemm_full(data.c_vir.as_ref().unwrap(), 'T', &grads[d], 'N', &mut v, 1.0, 0.0);
            mo_vir_g[d] = v;
        }
        // perturbed density components
        let mut rho: [Vec<f64>; 4] = Default::default();
        rho[0] = vec![0.0; ng];
        for d in 0..3 { rho[d + 1] = vec![0.0; ng]; }
        for g in 0..ng {
            let mut s0 = 0.0;
            for i in 0..occ { for a in 0..vir {
                s0 += z[i + a * occ] * mo_occ[[i, g]] * mo_vir[[a, g]];
            }}
            rho[0][g] = s0;
            for d in 0..3 {
                let mut s = 0.0;
                for i in 0..occ { for a in 0..vir {
                    let zz = z[i + a * occ];
                    s += zz * (mo_occ_g[d][[i, g]] * mo_vir[[a, g]] + mo_occ[[i, g]] * mo_vir_g[d][[a, g]]);
                }}
                rho[d + 1][g] = s;
            }
        }
        // kernel
        let wfxc = &data.fxc.wfxc;
        let mut feff: [Vec<f64>; 4] = Default::default();
        for alpha in 0..4 {
            let mut col = vec![0.0; ng];
            for g in 0..ng {
                let mut s = 0.0;
                for beta in 0..4 {
                    s += wfxc[g + alpha * ng + beta * 4 * ng] * rho[beta][g];
                }
                col[g] = s;
            }
            feff[alpha] = col;
        }
        // contract back
        let mut result = vec![0.0; occ * vir];
        for g in 0..ng {
            for i in 0..occ { for a in 0..vir {
                let mut s = feff[0][g] * mo_occ[[i, g]] * mo_vir[[a, g]];
                for d in 0..3 {
                    s += feff[d + 1][g] * (mo_occ_g[d][[i, g]] * mo_vir[[a, g]] + mo_occ[[i, g]] * mo_vir_g[d][[a, g]]);
                }
                result[i + a * occ] += s;
            }}
        }
        result
    }

    fn build_ao_data(nvar: usize) -> TDDFTData {
        let nao = 6; let occ = 3; let vir = 4; let ng = 17;
        let ao = MatrixFull::from_vec([nao, ng], pseudo(nao * ng, 7.7)).unwrap();
        let ao_grad = if nvar == 4 {
            Some([
                MatrixFull::from_vec([nao, ng], pseudo(nao * ng, 8.8)).unwrap(),
                MatrixFull::from_vec([nao, ng], pseudo(nao * ng, 9.9)).unwrap(),
                MatrixFull::from_vec([nao, ng], pseudo(nao * ng, 10.1)).unwrap(),
            ])
        } else { None };
        let c_occ = MatrixFull::from_vec([nao, occ], pseudo(nao * occ, 11.2)).unwrap();
        let c_vir = MatrixFull::from_vec([nao, vir], pseudo(nao * vir, 12.3)).unwrap();
        let wfxc = if nvar == 1 {
            pseudo(ng, 13.4)
        } else {
            pseudo(ng * 16, 13.4)
        };
        let fxc = FXCMatvecData {
            nvar,
            ngrids: ng,
            nocc: occ,
            nvir: vir,
            start_mo: 0,
            alpha_hybrid: 0.0,
            mo_occ: c_occ.clone(),
            mo_vir: c_vir.clone(),
            mo_occ_grad: None,
            mo_vir_grad: None,
            wfxc,
            use_opt: false,
        };
        let den_type = if nvar == 4 { XCDenType::SIGMA } else { XCDenType::RHO };
        TDDFTData {
            fxc,
            mode: TDDFTMode::AO,
            c_occ: Some(c_occ),
            c_vir: Some(c_vir),
            ao: Some(ao),
            ao_grad,
            ni: None,
            fxc_eff: None,
            den_type: Some(den_type),
            grid_batch: false,
            ri_ov: None,
            ri_oo_exch: None,
            ri_vv_exch: None,
            ri_ov_exch: None,
        }
    }

    #[test]
    fn test_fxc_ao_lda_matches_mo_reference() {
        let data = build_ao_data(1);
        let z: Vec<f64> = pseudo(data.c_occ.as_ref().unwrap().size[1] * data.c_vir.as_ref().unwrap().size[1], 14.5);
        let p = transition_density(data.c_occ.as_ref().unwrap(), data.c_vir.as_ref().unwrap(), &z, data.ao.as_ref().unwrap().size[0], data.c_occ.as_ref().unwrap().size[1], data.c_vir.as_ref().unwrap().size[1]);
        let f_full = fxc_matvec_ao(&data, &p);
        let result = contract_back(&f_full, data.c_occ.as_ref().unwrap(), data.c_vir.as_ref().unwrap(), data.c_occ.as_ref().unwrap().size[1], data.c_vir.as_ref().unwrap().size[1]);
        let reference = fxc_lda_reference(&data, &p, &z);
        for idx in 0..result.len() {
            assert!((result[idx] - reference[idx]).abs() < 1e-10,
                "LDA fxc[{}] = {} vs {}", idx, result[idx], reference[idx]);
        }
    }

    #[test]
    fn test_fxc_ao_gga_matches_mo_reference() {
        let data = build_ao_data(4);
        let z: Vec<f64> = pseudo(data.c_occ.as_ref().unwrap().size[1] * data.c_vir.as_ref().unwrap().size[1], 15.6);
        let p = transition_density(data.c_occ.as_ref().unwrap(), data.c_vir.as_ref().unwrap(), &z, data.ao.as_ref().unwrap().size[0], data.c_occ.as_ref().unwrap().size[1], data.c_vir.as_ref().unwrap().size[1]);
        let f_full = fxc_matvec_ao(&data, &p);
        let result = contract_back(&f_full, data.c_occ.as_ref().unwrap(), data.c_vir.as_ref().unwrap(), data.c_occ.as_ref().unwrap().size[1], data.c_vir.as_ref().unwrap().size[1]);
        let reference = fxc_gga_reference(&data, &p, &z);
        for idx in 0..result.len() {
            assert!((result[idx] - reference[idx]).abs() < 1e-10,
                "GGA fxc[{}] = {} vs {}", idx, result[idx], reference[idx]);
        }
    }

    #[test]
    fn test_b_exchange_uses_transposed_density() {
        // Verify b_matvec_ao's exchange sign/construction by checking that
        // the full-matrix route reproduces K_B[ia] = Σ_jb (ib|aj) z_jb.
        let nao = 6; let naux = 4;
        let rimatr = synthetic_rimatr(nao, naux);
        let eri = four_index_integrals(&rimatr);
        let data = build_ao_data(1); // nvar irrelevant for exchange
        let c_occ = data.c_occ.as_ref().unwrap();
        let c_vir = data.c_vir.as_ref().unwrap();
        let occ = c_occ.size[1];
        let vir = c_vir.size[1];
        let z: Vec<f64> = pseudo(occ * vir, 16.7);
        let p = transition_density(c_occ, c_vir, &z, nao, occ, vir);
        let p_t = p.transpose_and_drop();
        let device = DeviceBLAS::default();
        let (ri3fn, _, _) = rimatr.as_ref().unwrap();
        let cderi = ri3fn.to_rstsr_view(&device);
        let dms = vec![p_t.clone()].as_slice().to_rstsr(&device);
        let ks = crate::ri_jk::pure_incore::get_vk_ri_incore_dm(cderi, dms.view(), 2);
        let k = MatrixFull::from_vec([nao, nao], ks.i((.., .., 0)).iter().copied().collect()).unwrap();
        let result = contract_back(&k, c_occ, c_vir, occ, vir);
        // naive MO reference: Σ_jb (ib|aj) z_jb with MO integrals via 4-center
        for i in 0..occ { for a in 0..vir {
            let mut s = 0.0;
            for j in 0..occ { for b in 0..vir {
                // (ib|aj) in AO terms over all MO coefficients
                let mut eri_mo = 0.0;
                for mu in 0..nao { for nu in 0..nao { for lam in 0..nao { for sig in 0..nao {
                    eri_mo += c_occ[[mu, i]] * c_vir[[nu, b]] * c_vir[[lam, a]] * c_occ[[sig, j]]
                        * eri[mu][nu][lam][sig];
                }}}}
                s += z[j + b * occ] * eri_mo;
            }}
            assert!((result[i + a * occ] - s).abs() < 1e-10,
                "K_B[{},{}] = {} vs {}", i, a, result[i + a * occ], s);
        }}
    }
}
