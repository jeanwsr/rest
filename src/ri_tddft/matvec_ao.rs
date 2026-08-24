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
use rest_tensors::matrix::matrix_blas_lapack::{_dgemm_full, _dgemv};
use rest_tensors::matrixupper::map_upper_to_full;
use crossbeam::channel::unbounded;
use rayon::prelude::*;

use crate::scf_io::SCF;
use crate::dft::num_int::{FXCMatvecData, prepare_fxc_data};
use crate::dft::Grids;
use crate::dft::numint_matmul::nimatmul::NIMatmul;
use crate::dft::xceff::prelude::{XCDenType, XCSpin};
use crate::dft::xceff::libxc_wrap::determine_den_type;
use crate::ri_jk::util::get_cint_mol;
use crate::ri_tddft::utils::tddft_occupation_parameters;
use crate::utilities::rstsr_util::{RestTensorToRstsrViewAPI, RestTensorToRstsrTsrAPI, Tsr};
use rstsr::prelude::*;
use libxc::prelude::*;

/// Type alias matching `scf.rimatr`:
/// (ri3fn folded [npair, naux], basbas2baspar map, baspar2basbas list)
pub type RimatrTuple = Option<(MatrixFull<f64>, MatrixFull<usize>, Vec<[usize; 2]>)>;

/// Precomputed data for the AO-basis TDDFT matvec.
///
/// Reuses `prepare_fxc_data` for the kernel table (nvar/ngrids/alpha_hybrid);
/// stores AO values on grids plus occupied/virtual MO coefficients instead
/// of the MO-on-grid projections used by the MO-based path.
///
/// The batched path additionally builds a [`NIMatmul`] numerical-integrator
/// (AO values evaluated and cached via libcint) and the **raw** fxc kernel
/// `[ngrids, nvar, nvar]` (grid weights applied inside `NIMatmul`), so that
/// `make_fxc_pot_with_eff` reproduces the MO-path fxc contraction exactly.
pub struct TddftAoData {
    /// fxc kernel data (nvar, ngrids, wfxc, alpha_hybrid are consumed)
    pub fxc_base: FXCMatvecData,
    /// Occupied MO coefficients [nao, occ_size]
    pub c_occ: MatrixFull<f64>,
    /// Virtual MO coefficients [nao, vir_size]
    pub c_vir: MatrixFull<f64>,
    /// AO values on grids [nao, ngrids]
    pub ao: MatrixFull<f64>,
    /// For GGA: AO gradients on grids, each direction [nao, ngrids]
    pub ao_grad: Option<[MatrixFull<f64>; 3]>,
    /// Numerical integrator for the batched fxc kernel (AO cached via libcint).
    pub ni: Option<NIMatmul<'static>>,
    /// Raw (unweighted) fxc kernel `[ngrids, nvar, nvar]` for NIMatmul.
    pub fxc_eff: Option<Tsr>,
    /// Density type derived from nvar (RHO for LDA, SIGMA for GGA).
    pub den_type: Option<XCDenType>,
}

/// Prepare `TddftAoData` from a converged SCF object.
pub fn prepare_ao_data(scf: &SCF) -> TddftAoData {
    let (start_mo, _num_state, occ_size, vir_size, _homo, lumo) =
        tddft_occupation_parameters(scf);

    // Kernel table (wfxc), functional family and hybrid coefficient are
    // identical for both modes; the MO-on-grid projections inside this data
    // are simply not consumed by the AO matvec below.
    let fxc_base = prepare_fxc_data(scf);

    let grids = scf.grids.as_ref().expect("DFT grids must be initialized for AO-mode fxc");
    let ngrids = grids.weights.len();
    let num_basis = scf.mol.num_basis;

    // ── Dense AO values on grids ──
    let ao: MatrixFull<f64> = match &grids.ao {
        Some(a) => a.clone(),
        None => match &grids.ao_compressed {
            Some(c) => Grids::decompress_ao(c),
            None => panic!("AO on grids must be tabulated (dense or compressed)"),
        },
    };

    // ── GGA: dense AO gradients on grids ──
    let ao_grad: Option<[MatrixFull<f64>; 3]> = if fxc_base.nvar == 4 {
        let aop_owned: Option<RIFull<f64>>;
        let aop: &RIFull<f64> = match &grids.aop {
            Some(a) => { aop_owned = None; a }
            None => match &grids.aop_compressed {
                Some(c) => { aop_owned = Some(Grids::decompress_aop(c)); aop_owned.as_ref().unwrap() }
                None => panic!("AO gradients needed for GGA fxc (dense or compressed)"),
            },
        };
        let mut grads: [Option<MatrixFull<f64>>; 3] = [None, None, None];
        for d in 0..3 {
            let slice = aop.get_reducing_matrix(d).unwrap();
            grads[d] = MatrixFull::from_vec(
                [num_basis, ngrids],
                slice.iter().cloned().collect(),
            );
        }
        Some([grads[0].take().unwrap(), grads[1].take().unwrap(), grads[2].take().unwrap()])
    } else {
        None
    };

    // ── Extract occupied/virtual MO coefficients ──
    let eigvec = &scf.eigenvectors[0];
    let mut c_occ = MatrixFull::new([num_basis, occ_size], 0.0);
    for j in 0..occ_size {
        for i in 0..num_basis {
            c_occ[[i, j]] = eigvec[[i, start_mo + j]];
        }
    }
    let mut c_vir = MatrixFull::new([num_basis, vir_size], 0.0);
    for j in 0..vir_size {
        for i in 0..num_basis {
            c_vir[[i, j]] = eigvec[[i, lumo + j]];
        }
    }

    let ni = build_ni(scf, ngrids);
    let fxc_eff = build_fxc_eff(&fxc_base, ngrids);
    let den_type = Some(if fxc_base.nvar == 4 { XCDenType::SIGMA } else { XCDenType::RHO });

    TddftAoData {
        fxc_base,
        c_occ,
        c_vir,
        ao,
        ao_grad,
        ni,
        fxc_eff,
        den_type,
    }
}

/// Build a [`NIMatmul`] numerical integrator for the batched fxc kernel.
///
/// Uses the SCF DFT grid coordinates (AO evaluated via libcint and cached
/// inside the integrator). Grid weights are set to **ones**: the kernel table
/// `wfxc` already carries the weights, so NIMatmul's internal
/// `weights * fxc_eff` multiplication leaves it unchanged — reproducing the
/// MO-path contraction bit-identically.
fn build_ni(scf: &SCF, ngrids: usize) -> Option<NIMatmul<'static>> {
    let grids = scf.grids.as_ref()?;
    let cint = get_cint_mol(&scf.mol);
    let ones: Vec<f64> = vec![1.0; ngrids];
    Some(NIMatmul::new(&cint, &grids.coordinates, &ones))
}

/// Build the raw-shape fxc kernel `[ngrids, nvar, nvar]` (f-contiguous,
/// `idx = g + α·ng + β·nvar·ng`) from the weighted `wfxc` table.
fn build_fxc_eff(fxc_base: &FXCMatvecData, ngrids: usize) -> Option<Tsr> {
    let nvar = fxc_base.nvar;
    let device = DeviceBLAS::default();
    Some(rt::asarray((fxc_base.wfxc.clone(), [ngrids, nvar, nvar].f(), &device)))
}

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

/// Folded-pair Coulomb contraction J[P][μν] = Σ_λσ P_λσ (μν|λσ)
/// from the metric-decomposed AO RI tensor (`rimatr`'s ri3fn, folded upper).
///
/// Correct for non-symmetric P: the full double sum collapses onto the
/// folded triangle as 2×(off-diagonal pairs) + 1×(diagonal pairs).
pub fn ri_coulomb_ao(rimatr: &RimatrTuple, p: &MatrixFull<f64>) -> MatrixFull<f64> {
    let (ri3fn, basbas2baspar, _) = rimatr.as_ref()
        .expect("rimatr must be initialized for AO-mode TDDFT");
    let nao = basbas2baspar.size[0];
    let npair = ri3fn.size[0];
    let naux = ri3fn.size[1];
    let index_map = map_upper_to_full(npair).expect("rimatr ri3fn size is not a valid folded triangle");

    // Fold P onto the pair triangle. For a non-symmetric P, the full double
    // sum Σ_λσ P_λσ B_λσ collapses to (P_ij + P_ji)·B_ij on off-diagonal
    // pairs (B is symmetric) and P_ii·B_ii on the diagonal.
    let mut p_folded = vec![0.0_f64; npair];
    for (k, ij) in index_map.data.iter().enumerate() {
        let (i, j) = (ij[0], ij[1]);
        p_folded[k] = if i == j { p[[i, i]] } else { p[[i, j]] + p[[j, i]] };
    }

    // t_Q = Σ_pair B[pair, Q] · P_folded[pair]
    let mut t_q = vec![0.0_f64; naux];
    _dgemv(ri3fn, &p_folded, &mut t_q, 'T', 1.0, 0.0, 1, 1);

    // F_folded[pair] = Σ_Q B[pair, Q] · t_Q
    let mut f_folded = vec![0.0_f64; npair];
    _dgemv(ri3fn, &t_q, &mut f_folded, 'N', 1.0, 0.0, 1, 1);

    // Unfold into the full symmetric matrix.
    let mut f = MatrixFull::new([nao, nao], 0.0);
    for (k, ij) in index_map.data.iter().enumerate() {
        f[[ij[0], ij[1]]] = f_folded[k];
        f[[ij[1], ij[0]]] = f_folded[k];
    }
    f
}

/// RI exchange contraction K[P][μν] = Σ_λσ P_λσ (μλ|σν) from `rimatr`.
///
/// Accumulated over auxiliary columns Q as M_Q · P · M_Q (M_Q symmetric),
/// which handles non-symmetric transition densities exactly.
pub fn ri_exchange_ao(rimatr: &RimatrTuple, p: &MatrixFull<f64>) -> MatrixFull<f64> {
    let (ri3fn, basbas2baspar, _) = rimatr.as_ref()
        .expect("rimatr must be initialized for AO-mode TDDFT");
    let nao = basbas2baspar.size[0];
    let npair = ri3fn.size[0];
    let index_map = map_upper_to_full(npair).expect("rimatr ri3fn size is not a valid folded triangle");

    let mut result = MatrixFull::new([nao, nao], 0.0);
    let (sender, receiver) = unbounded();
    ri3fn.par_iter_columns_full().for_each_with(sender, |s, m| {
        // Unfold aux column Q into the full symmetric M_Q.
        let mut mq = MatrixFull::new([nao, nao], 0.0);
        for (k, ij) in index_map.data.iter().enumerate() {
            let v = m[k];
            mq[[ij[0], ij[1]]] = v;
            mq[[ij[1], ij[0]]] = v;
        }
        let mut tmp = MatrixFull::new([nao, nao], 0.0);
        _dgemm_full(&mq, 'N', p, 'N', &mut tmp, 1.0, 0.0);
        let mut kq = MatrixFull::new([nao, nao], 0.0);
        _dgemm_full(&tmp, 'N', &mq, 'N', &mut kq, 1.0, 0.0);
        s.send(kq.data).unwrap();
    });
    receiver.into_iter().for_each(|kq_data| {
        result.data.iter_mut().zip(kq_data.iter())
            .for_each(|(dst, src)| *dst += *src);
    });
    result
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
pub fn fxc_matvec_ao(ao_data: &TddftAoData, p: &MatrixFull<f64>) -> MatrixFull<f64> {
    match ao_data.fxc_base.nvar {
        1 => fxc_ao_lda(ao_data, p),
        4 => fxc_ao_gga(ao_data, p),
        n => panic!("fxc_matvec_ao only supports LDA (nvar=1) and GGA (nvar=4); got {}", n),
    }
}

/// LDA (nvar=1): ρ_z[g] = Σ_μν P_μν φ_μ(g)φ_ν(g); F = Σ_g v[g] φ⊗φ.
fn fxc_ao_lda(ao_data: &TddftAoData, p: &MatrixFull<f64>) -> MatrixFull<f64> {
    let ng = ao_data.fxc_base.ngrids;

    // X = P · AO  →  ρ_z[g] = column_dot(AO[:,g], X[:,g])
    let mut x = MatrixFull::new([p.size[0], ng], 0.0);
    _dgemm_full(p, 'N', &ao_data.ao, 'N', &mut x, 1.0, 0.0);
    let rho_z = column_dots(&ao_data.ao, &x);

    let v: Vec<f64> = rho_z.iter().zip(ao_data.fxc_base.wfxc.iter())
        .map(|(r, w)| r * w).collect();

    // F = (AO scaled by v) · AOᵀ
    let ao_s = scale_columns(&ao_data.ao, &v);
    let mut f = MatrixFull::new([ao_data.ao.size[0], ao_data.ao.size[0]], 0.0);
    _dgemm_full(&ao_s, 'N', &ao_data.ao, 'T', &mut f, 1.0, 0.0);
    f
}

/// GGA (nvar=4): perturbed density has components
/// ρ⁰ = ΣP φφ,  ρ^{d+1} = ΣP (∂_dφ_μ·φ_ν + φ_μ·∂_dφ_ν);
/// kernel applied through wfxc[g, α, β], then contracted back per component.
fn fxc_ao_gga(ao_data: &TddftAoData, p: &MatrixFull<f64>) -> MatrixFull<f64> {
    let ng = ao_data.fxc_base.ngrids;
    let nao = p.size[0];
    let grads = ao_data.ao_grad.as_ref().expect("GGA requires AO gradients on grids");

    // Perturbed quantities on grids: x0 = P·AO, xd = P·AOgrad[d]
    let mut x0 = MatrixFull::new([nao, ng], 0.0);
    _dgemm_full(p, 'N', &ao_data.ao, 'N', &mut x0, 1.0, 0.0);
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
    rho[0] = column_dots(&ao_data.ao, &x0);
    for d in 0..3 {
        let part1 = column_dots(&grads[d], &x0);
        let part2 = column_dots(&ao_data.ao, &xd[d]);
        rho[d + 1] = part1.into_iter().zip(part2.into_iter()).map(|(a, b)| a + b).collect();
    }

    // Kernel application: feff[α][g] = Σ_β wfxc[g + α·ng + β·4ng] · ρ^β[g]
    let wfxc = &ao_data.fxc_base.wfxc;
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
    let ao_s0 = scale_columns(&ao_data.ao, &feff[0]);
    _dgemm_full(&ao_s0, 'N', &ao_data.ao, 'T', &mut f, 1.0, 1.0);
    for d in 0..3 {
        let alpha = d + 1;
        let gd_s = scale_columns(&grads[d], &feff[alpha]);
        _dgemm_full(&gd_s, 'N', &ao_data.ao, 'T', &mut f, 1.0, 1.0);
        let ao_sa = scale_columns(&ao_data.ao, &feff[alpha]);
        _dgemm_full(&ao_sa, 'N', &grads[d], 'T', &mut f, 1.0, 1.0);
    }
    f
}

// ══════════════════════════════════════════════════════════════════
// Public A/B matvecs (drop-in replacements for matvec.rs versions)
// ══════════════════════════════════════════════════════════════════

/// Full A-block matvec in AO mode:
/// A·z = (ε_a − ε_i)·z + f_c·J[P_z] − c_x·K[P_z] + fxc[P_z]
///
/// singlet (xlet='S'): f_c = 2; triplet ('T'): f_c = 0; ROHF-like 'R': 1.
pub fn a_matvec_ao(
    scf: &SCF,
    ao_data: &TddftAoData,
    z: &Vec<f64>,
    xlet: char,
    alpha_hybrid: f64,
) -> Vec<f64> {
    let (start_mo, _, occ_size, vir_size, _homo, lumo) = tddft_occupation_parameters(scf);
    let dim = occ_size * vir_size;
    let ks = &scf.eigenvalues[0];

    // Step 1: diagonal contribution
    let mut result = vec![0.0; dim];
    for a in 0..vir_size {
        for i in 0..occ_size {
            let idx = i + a * occ_size;
            result[idx] += (ks[lumo + a] - ks[start_mo + i]) * z[idx];
        }
    }

    // Steps 2-4: kernels from the AO transition density
    let p = transition_density(&ao_data.c_occ, &ao_data.c_vir, z,
                               scf.mol.num_basis, occ_size, vir_size);

    let coulomb_factor = if xlet == 'S' { 2.0 } else if xlet == 'R' { 1.0 } else { 0.0 };
    let mut f_total = MatrixFull::new([scf.mol.num_basis, scf.mol.num_basis], 0.0);
    if coulomb_factor != 0.0 {
        let j = ri_coulomb_ao(&scf.rimatr, &p);
        for idx in 0..f_total.data.len() {
            f_total.data[idx] += coulomb_factor * j.data[idx];
        }
    }
    if alpha_hybrid.abs() > 1e-15 {
        let k = ri_exchange_ao(&scf.rimatr, &p);
        for idx in 0..f_total.data.len() {
            f_total.data[idx] -= alpha_hybrid * k.data[idx];
        }
    }
    let f_fxc = fxc_matvec_ao(ao_data, &p);
    for idx in 0..f_total.data.len() {
        f_total.data[idx] += f_fxc.data[idx];
    }

    // Step 5: project back to MO amplitudes
    let kernel_mo = contract_back(&f_total, &ao_data.c_occ, &ao_data.c_vir, occ_size, vir_size);
    for idx in 0..dim {
        result[idx] += kernel_mo[idx];
    }
    result
}

/// Full B-block matvec in AO mode:
/// B·z = f_c·J[P_z] − c_x·K[P_zᵀ] + fxc[P_z]
///
/// The exchange term uses the transposed transition density:
/// C_occᵀ·K[Pᵀ]·C_vir gives Σ_jb (ib|aj) z_jb.
pub fn b_matvec_ao(
    scf: &SCF,
    ao_data: &TddftAoData,
    z: &Vec<f64>,
    xlet: char,
    alpha_hybrid: f64,
) -> Vec<f64> {
    let (_start_mo, _, occ_size, vir_size, _homo, _lumo) = tddft_occupation_parameters(scf);
    let dim = occ_size * vir_size;
    let nao = scf.mol.num_basis;

    let mut result = vec![0.0; dim];

    let p = transition_density(&ao_data.c_occ, &ao_data.c_vir, z, nao, occ_size, vir_size);

    let coulomb_factor = if xlet == 'S' { 2.0 } else if xlet == 'R' { 1.0 } else { 0.0 };
    let mut f_total = MatrixFull::new([nao, nao], 0.0);
    if coulomb_factor != 0.0 {
        let j = ri_coulomb_ao(&scf.rimatr, &p);
        for idx in 0..f_total.data.len() {
            f_total.data[idx] += coulomb_factor * j.data[idx];
        }
    }
    if alpha_hybrid.abs() > 1e-15 {
        let p_t = p.clone().transpose_and_drop();
        let k = ri_exchange_ao(&scf.rimatr, &p_t);
        for idx in 0..f_total.data.len() {
            f_total.data[idx] -= alpha_hybrid * k.data[idx];
        }
    }
    let f_fxc = fxc_matvec_ao(ao_data, &p);
    for idx in 0..f_total.data.len() {
        f_total.data[idx] += f_fxc.data[idx];
    }

    contract_back(&f_total, &ao_data.c_occ, &ao_data.c_vir, occ_size, vir_size)
        .into_iter().zip(result.iter_mut()).for_each(|(src, dst)| *dst += src);
    result
}

// ══════════════════════════════════════════════════════════════════
// Batched (Option B) matvecs
//
// The Davidson solver variants in `solvers/davidson.rs` apply A (and B) to a
// whole block of trial vectors [dim, m] at once. The fxc contribution is then
// evaluated for all m vectors in a single `NIMatmul` pass (rho1 for all sets,
// one `make_fxc_pot_with_eff`), matching PySCF's batched `vind(zs)` design.
// The RI Coulomb/exchange remain per-vector loops (flop-bound, correct, and
// already validated against four-index references).
// ══════════════════════════════════════════════════════════════════

/// Symmetrize a transition density: P_sym = (P + Pᵀ)/2.
///
/// NIMatmul's density evaluator assumes a symmetric density matrix (the SIGMA
/// response is `2·Σ P φ⁰_μ φ^t_ν`). Since both the RHO and SIGMA kernels are
/// symmetric in μ↔ν, the fxc response is invariant under symmetrization of P,
/// so this is exact for non-symmetric transition densities.
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

/// Batched fxc contribution: F_fxc block `[nao, nao, m]` for a block of
/// transition densities, via `NIMatmul` (one batched rho + one batched kernel
/// contraction for all m vectors).
fn fxc_matvec_ao_batched(
    ao_data: &mut TddftAoData,
    p_block: &[MatrixFull<f64>],
    device: &DeviceBLAS,
) -> MatrixFull<f64> {
    let m = p_block.len();
    let nao = ao_data.c_occ.size[0];
    let ni = ao_data.ni.as_mut().expect("NIMatmul not initialized");
    let den_type = ao_data.den_type.expect("den_type not initialized");

    // Symmetrize each transition density (NIMatmul density evaluator assumes
    // symmetric dm; the fxc response is invariant under P → (P+Pᵀ)/2).
    let p_sym_list: Vec<MatrixFull<f64>> = p_block.iter().map(symmetrize_density).collect();
    let p_sym_views: Vec<_> = p_sym_list.iter().map(|p| p.to_rstsr_view(device)).collect();

    // rho1: [ngrids, nvar, m] — one batched density evaluation for all m.
    let rho1 = ni.make_rho_from_dm(&p_sym_views, den_type);

    // f_fxc: [nao, nao, m] — one batched kernel contraction for all m.
    let fxc_eff = ao_data.fxc_eff.as_ref().expect("fxc_eff not initialized").view();
    let f_fxc = ni.make_fxc_pot_with_eff(fxc_eff, rho1.view(), den_type, XCSpin::Unpolarized);

    // Copy the [nao, nao, m] tensor into a MatrixFull block.
    // The output is explicitly symmetrized by NIMatmul (F + Fᵀ), so the
    // storage-order layout of the copied slice is irrelevant.
    let mut out = MatrixFull::new([nao, nao * m], 0.0);
    for s in 0..m {
        let slice = f_fxc.i((.., .., s));
        let mut r = 0;
        for v in slice.iter() {
            out.data[s * nao * nao + r] = *v;
            r += 1;
        }
    }
    out
}

/// Full A-block matvec in AO mode, applied to a block of trial vectors.
///
/// `z_block` is `[dim, m]`; returns `[dim, m]`. The fxc term is batched via
/// `NIMatmul` (all m vectors in one pass); RI Coulomb/exchange stay per-vector.
pub fn a_matvec_ao_batched(
    scf: &SCF,
    ao_data: &mut TddftAoData,
    z_block: &MatrixFull<f64>,
    xlet: char,
    alpha_hybrid: f64,
) -> MatrixFull<f64> {
    let (start_mo, _, occ_size, vir_size, _homo, lumo) = tddft_occupation_parameters(scf);
    let dim = occ_size * vir_size;
    let m = z_block.size[1];
    let nao = scf.mol.num_basis;
    let ks = &scf.eigenvalues[0];
    let coulomb_factor = if xlet == 'S' { 2.0 } else if xlet == 'R' { 1.0 } else { 0.0 };

    let mut result = MatrixFull::new([dim, m], 0.0);

    // Step 1: diagonal contribution (elementwise on the block)
    for s in 0..m {
        for a in 0..vir_size {
            for i in 0..occ_size {
                let idx = i + a * occ_size;
                result[[idx, s]] += (ks[lumo + a] - ks[start_mo + i]) * z_block[[idx, s]];
            }
        }
    }

    // Steps 2-4: kernels from the AO transition densities.
    // Build P for each column, and batch the fxc over all columns.
    let device = DeviceBLAS::default();
    let mut p_block: Vec<MatrixFull<f64>> = Vec::with_capacity(m);
    for s in 0..m {
        let z: Vec<f64> = (0..dim).map(|r| z_block[[r, s]]).collect();
        p_block.push(transition_density(&ao_data.c_occ, &ao_data.c_vir, &z, nao, occ_size, vir_size));
    }
    let f_fxc_block = fxc_matvec_ao_batched(ao_data, &p_block, &device);

    // Per-vector J/K + contract back
    for s in 0..m {
        let p = &p_block[s];
        let mut f_total = MatrixFull::new([nao, nao], 0.0);
        if coulomb_factor != 0.0 {
            let j = ri_coulomb_ao(&scf.rimatr, p);
            for idx in 0..f_total.data.len() {
                f_total.data[idx] += coulomb_factor * j.data[idx];
            }
        }
        if alpha_hybrid.abs() > 1e-15 {
            let k = ri_exchange_ao(&scf.rimatr, p);
            for idx in 0..f_total.data.len() {
                f_total.data[idx] -= alpha_hybrid * k.data[idx];
            }
        }
        // fxc contribution for this vector (column s of the batched fxc block)
        let base = s * nao * nao;
        for idx in 0..f_total.data.len() {
            f_total.data[idx] += f_fxc_block.data[base + idx];
        }
        let kernel_mo = contract_back(&f_total, &ao_data.c_occ, &ao_data.c_vir, occ_size, vir_size);
        for r in 0..dim {
            result[[r, s]] += kernel_mo[r];
        }
    }
    result
}

/// Full B-block matvec in AO mode, applied to a block of trial vectors.
///
/// `z_block` is `[dim, m]`; returns `[dim, m]`. The fxc term is batched via
/// `NIMatmul`; RI Coulomb/exchange stay per-vector (A-block uses P, B-block Pᵀ).
pub fn b_matvec_ao_batched(
    scf: &SCF,
    ao_data: &mut TddftAoData,
    z_block: &MatrixFull<f64>,
    xlet: char,
    alpha_hybrid: f64,
) -> MatrixFull<f64> {
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
        p_block.push(transition_density(&ao_data.c_occ, &ao_data.c_vir, &z, nao, occ_size, vir_size));
    }
    let f_fxc_block = fxc_matvec_ao_batched(ao_data, &p_block, &device);

    for s in 0..m {
        let p = &p_block[s];
        let mut f_total = MatrixFull::new([nao, nao], 0.0);
        if coulomb_factor != 0.0 {
            let j = ri_coulomb_ao(&scf.rimatr, p);
            for idx in 0..f_total.data.len() {
                f_total.data[idx] += coulomb_factor * j.data[idx];
            }
        }
        if alpha_hybrid.abs() > 1e-15 {
            let p_t = p.clone().transpose_and_drop();
            let k = ri_exchange_ao(&scf.rimatr, &p_t);
            for idx in 0..f_total.data.len() {
                f_total.data[idx] -= alpha_hybrid * k.data[idx];
            }
        }
        let base = s * nao * nao;
        for idx in 0..f_total.data.len() {
            f_total.data[idx] += f_fxc_block.data[base + idx];
        }
        let kernel_mo = contract_back(&f_total, &ao_data.c_occ, &ao_data.c_vir, occ_size, vir_size);
        for r in 0..dim {
            result[[r, s]] += kernel_mo[r];
        }
    }
    result
}

// ══════════════════════════════════════════════════════════════════
// Tests: validate AO contractions against naive four-index references
// ══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;

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
        let f = ri_coulomb_ao(&rimatr, &p);
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
        let k = ri_exchange_ao(&rimatr, &p);
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
    fn fxc_lda_reference(data: &TddftAoData, p: &MatrixFull<f64>, z: &[f64]) -> Vec<f64> {
        let ng = data.fxc_base.ngrids;
        let occ = data.c_occ.size[1];
        let vir = data.c_vir.size[1];
        // φ_i[g] = Σ_μ C_occ[μ,i] ao[μ,g]; φ_a[g] = Σ_μ C_vir[μ,a] ao[μ,g]
        let mut mo_occ = MatrixFull::new([occ, ng], 0.0);
        _dgemm_full(&data.c_occ, 'T', &data.ao, 'N', &mut mo_occ, 1.0, 0.0);
        let mut mo_vir = MatrixFull::new([vir, ng], 0.0);
        _dgemm_full(&data.c_vir, 'T', &data.ao, 'N', &mut mo_vir, 1.0, 0.0);
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
            let v = data.fxc_base.wfxc[g] * rho[g];
            for i in 0..occ { for a in 0..vir {
                result[i + a * occ] += v * mo_occ[[i, g]] * mo_vir[[a, g]];
            }}
        }
        result
    }

    /// Reference GGA fxc from the textbook MO formula (perturbed gradients).
    fn fxc_gga_reference(data: &TddftAoData, p: &MatrixFull<f64>, z: &[f64]) -> Vec<f64> {
        let ng = data.fxc_base.ngrids;
        let occ = data.c_occ.size[1];
        let vir = data.c_vir.size[1];
        let grads = data.ao_grad.as_ref().unwrap();
        // MO values + gradients on grids
        let mut mo_occ = MatrixFull::new([occ, ng], 0.0);
        _dgemm_full(&data.c_occ, 'T', &data.ao, 'N', &mut mo_occ, 1.0, 0.0);
        let mut mo_vir = MatrixFull::new([vir, ng], 0.0);
        _dgemm_full(&data.c_vir, 'T', &data.ao, 'N', &mut mo_vir, 1.0, 0.0);
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
            _dgemm_full(&data.c_occ, 'T', &grads[d], 'N', &mut o, 1.0, 0.0);
            mo_occ_g[d] = o;
            let mut v = MatrixFull::new([vir, ng], 0.0);
            _dgemm_full(&data.c_vir, 'T', &grads[d], 'N', &mut v, 1.0, 0.0);
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
        let wfxc = &data.fxc_base.wfxc;
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

    fn build_ao_data(nvar: usize) -> TddftAoData {
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
        let fxc_base = FXCMatvecData {
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
        TddftAoData { fxc_base, c_occ, c_vir, ao, ao_grad, ni: None, fxc_eff: None, den_type: None }
    }

    #[test]
    fn test_fxc_ao_lda_matches_mo_reference() {
        let data = build_ao_data(1);
        let z: Vec<f64> = pseudo(data.c_occ.size[1] * data.c_vir.size[1], 14.5);
        let p = transition_density(&data.c_occ, &data.c_vir, &z, data.ao.size[0], data.c_occ.size[1], data.c_vir.size[1]);
        let f_full = fxc_matvec_ao(&data, &p);
        let result = contract_back(&f_full, &data.c_occ, &data.c_vir, data.c_occ.size[1], data.c_vir.size[1]);
        let reference = fxc_lda_reference(&data, &p, &z);
        for idx in 0..result.len() {
            assert!((result[idx] - reference[idx]).abs() < 1e-10,
                "LDA fxc[{}] = {} vs {}", idx, result[idx], reference[idx]);
        }
    }

    #[test]
    fn test_fxc_ao_gga_matches_mo_reference() {
        let data = build_ao_data(4);
        let z: Vec<f64> = pseudo(data.c_occ.size[1] * data.c_vir.size[1], 15.6);
        let p = transition_density(&data.c_occ, &data.c_vir, &z, data.ao.size[0], data.c_occ.size[1], data.c_vir.size[1]);
        let f_full = fxc_matvec_ao(&data, &p);
        let result = contract_back(&f_full, &data.c_occ, &data.c_vir, data.c_occ.size[1], data.c_vir.size[1]);
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
        let occ = data.c_occ.size[1];
        let vir = data.c_vir.size[1];
        let z: Vec<f64> = pseudo(occ * vir, 16.7);
        let p = transition_density(&data.c_occ, &data.c_vir, &z, nao, occ, vir);
        let p_t = p.transpose_and_drop();
        let k = ri_exchange_ao(&rimatr, &p_t);
        let result = contract_back(&k, &data.c_occ, &data.c_vir, occ, vir);
        // naive MO reference: Σ_jb (ib|aj) z_jb with MO integrals via 4-center
        for i in 0..occ { for a in 0..vir {
            let mut s = 0.0;
            for j in 0..occ { for b in 0..vir {
                // (ib|aj) in AO terms over all MO coefficients
                let mut eri_mo = 0.0;
                for mu in 0..nao { for nu in 0..nao { for lam in 0..nao { for sig in 0..nao {
                    eri_mo += data.c_occ[[mu, i]] * data.c_vir[[nu, b]] * data.c_vir[[lam, a]] * data.c_occ[[sig, j]]
                        * eri[mu][nu][lam][sig];
                }}}}
                s += z[j + b * occ] * eri_mo;
            }}
            assert!((result[i + a * occ] - s).abs() < 1e-10,
                "K_B[{},{}] = {} vs {}", i, a, result[i + a * occ], s);
        }}
    }
}
