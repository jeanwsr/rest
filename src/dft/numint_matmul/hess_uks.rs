// UKS Hessian XC terms with the Becke grid-shift; see also pyhessref/nimatmul/uks.py
//
// Unrestricted sibling of `hess_rks`: the single-spin helpers, the becke
// partition machinery, and the chunk-level batching driver are shared with the
// RKS implementation; only the spin-coupled pieces differ.  The spin extension
// of every grid-shift term is the obvious one — terms linear in `vxc` become a
// spin sum (vxc[alpha] against the alpha quantity plus vxc[beta] against the
// beta quantity), and terms quadratic in the fxc kernel become the four
// spin-pair sum (the same aa/ab/ba/bb structure as `get_de_fxc_uks`).
//
// The grid-shift terms are optional (`grid_shift` argument); with it off, only
// the grid-fixed terms are evaluated.  See `hess_rks` for the index and shape
// conventions; on top of those, spin-polarized tensors carry a trailing spin
// axis: `rho [ngrids, nvar, 2]`, `vxc [ngrids, nvar, 2]`,
// `fxc [ngrids, nvar, 2, nvar, 2]`, and the skeleton-Fock tensors exist per
// spin (`vmat_deriv1_a` / `vmat_deriv1_b`).

use super::hess_rks::{
    by_atom_chunk, contract_pvxc, get_de_becke_atom_1, get_de_becke_atom_2, get_de_vxc_diag, get_de_vxc_off, get_drho,
    get_hess_ao_deriv, get_hess_ncomp_ao_dm0, get_vmat_ip, get_vmat_vxc, make_dao_vxc_diag, make_dao_vxc_off,
    quad_split_by_atom, xc_fock_stack, BECKE_HARDNESS, NO_ATM,
};
use super::prelude::*;
use crate::analdrv::prelude::*;
use crate::dft::gen_grids::becke_partitioning_deriv::{
    becke_partition_with_tables, gen_adjustment_factor, try_atm_quad_split, AtmIndices, BeckeMolTables,
    BeckePartitionArg,
};

use std::sync::atomic::{AtomicUsize, Ordering};

use XCDenType::*;

/* #region const dimensions/indices definition */

const O: usize = 0;
const X: usize = 1;
const Y: usize = 2;
const Z: usize = 3;

#[allow(non_upper_case_globals)]
const α: usize = 0;
#[allow(non_upper_case_globals)]
const β: usize = 1;

/* #endregion */

/* #region macro for indexing last dimension */

macro_rules! index {
    ($tsr: ident, $($idx:expr),*) => {
        $tsr.i((Ellipsis, $($idx),*))
    };
}

macro_rules! index_mut {
    ($tsr: ident, $($idx:expr),*) => {
        (*&mut $tsr.i_mut((Ellipsis, $($idx),*)))
    };
}

/* #endregion */

/* #region basic pure functions of skeleton hessian evaluation */

/// On-grid spin-polarized density with 1st/2nd functional derivatives and the
/// per-particle XC energy density.
///
/// The per-particle energy density `exc` (order 0, spin-summed) is needed by
/// the `cddw` contraction of `de_becke_full_2` (there contracted with the
/// spin-summed value channel `rhoa[0] + rhob[0]`); callers that do not need it
/// simply ignore the output.
///
/// # Parameters
///
/// - `xc_func_list` : list of `(scale, functional)` pairs.  The overall family is the strictest one
///   across the list; contributions of looser families are added into their leading `nvar_i` slice.
/// - `ao` : shape `[ngrids, nao, ncomp]` (g, u, component).  AO values and derivatives; only each
///   family's leading channels are read.
/// - `ao_dm0α`, `ao_dm0β` : shape `[ngrids, nao, ncomp_ao_dm0]`.  Leading AO channels contracted
///   with the per-spin density matrices.
///
/// # Returns
///
/// - `rho` : shape `[ngrids, nvar, 2]` (g, x, sigma).  On-grid density components per spin.
/// - `exc` : shape `[ngrids]`.  Per-particle XC energy density (spin-summed).
/// - `vxc` : shape `[ngrids, nvar, 2]`.  1st functional derivative.
/// - `fxc` : shape `[ngrids, nvar, 2, nvar, 2]`.  2nd functional derivative.
pub fn get_rho_exc_vxc_fxc_uks(
    xc_func_list: &[(f64, LibXCFunctional)],
    ao: TsrView,
    ao_dm0α: TsrView,
    ao_dm0β: TsrView,
) -> (Tsr, Tsr, Tsr, Tsr) {
    assert!(!xc_func_list.is_empty(), "xc_func_list must not be empty");
    let xc_type = xc_func_list
        .iter()
        .map(|(_, f)| determine_den_type(f))
        .max_by_key(|t| t.num_nvar())
        .expect("xc_func_list must not be empty");
    let nvar = xc_type.num_nvar();
    let ngrids = ao.shape()[0];
    let device = ao.device().clone();

    let mut rho = rt::zeros(([ngrids, nvar, 2], &device));
    for (σ, ao_dm0σ) in [(α, &ao_dm0α), (β, &ao_dm0β)] {
        index_mut!(rho, 0, σ) += rt::vecdot(index!(ao, 0), index!(ao_dm0σ, O), 1);
        if matches!(xc_type, SIGMA | TAU) {
            index_mut!(rho, X, σ) += 2 * rt::vecdot(index!(ao, X), index!(ao_dm0σ, O), 1);
            index_mut!(rho, Y, σ) += 2 * rt::vecdot(index!(ao, Y), index!(ao_dm0σ, O), 1);
            index_mut!(rho, Z, σ) += 2 * rt::vecdot(index!(ao, Z), index!(ao_dm0σ, O), 1);
        }
        if matches!(xc_type, TAU) {
            index_mut!(rho, 4, σ) += 0.5
                * (rt::vecdot(index!(ao, X), index!(ao_dm0σ, X), 1)
                    + rt::vecdot(index!(ao, Y), index!(ao_dm0σ, Y), 1)
                    + rt::vecdot(index!(ao, Z), index!(ao_dm0σ, Z), 1))
        }
    }

    let mut exc = rt::zeros(([ngrids], &device));
    let mut vxc = rt::zeros(([ngrids, nvar, 2], &device));
    let mut fxc = rt::zeros(([ngrids, nvar, 2, nvar, 2], &device));
    for (scale, xc_func) in xc_func_list {
        let xc_type_i = determine_den_type(xc_func);
        let nvar_i = xc_type_i.num_nvar();
        let rho_i = rho.i((.., ..nvar_i, ..));
        let xc_eff = libxc_eval_eff(xc_func, rho_i, 2, false);
        let [e_i, vxc_i, fxc_i] = xc_eff.into_iter().collect_array().unwrap();
        exc += *scale * e_i.into_shape([ngrids]);
        *&mut vxc.i_mut((.., ..nvar_i, ..)) += *scale * vxc_i;
        *&mut fxc.i_mut((.., ..nvar_i, .., ..nvar_i, ..)) += *scale * fxc_i;
    }

    (rho, exc, vxc, fxc)
}

/// Single spin-pair fxc contraction `einsum("g, Atxg, xyg, Bsyg -> ABts",
/// weights, drho1, fxc_block, drho2)`, the inner loop of [`get_de_fxc_uks`].
///
/// # Parameters
///
/// - `wf_block` : shape `[ngrids, nvar, nvar]` (g, x, y); a single spin block of `wf`.
/// - `drho1`, `drho2` : shape `[ngrids, nvar, 3, natm]` (output of [`get_drho`]).
///
/// # Returns
///
/// - `de_fxc` : shape `[3, 3, natm, natm]` (t, s, A, B).
fn get_de_fxc_uks_inner(wf_block: TsrView, drho1: TsrView, drho2: TsrView) -> Tsr {
    let [ngrids, nvar, _, natm] = drho1.shape().iter().cloned().collect_array().unwrap();

    let tmp1 = rt::vecdot(wf_block.i((.., .., .., None, None)), drho1.i((.., .., None, .., ..)), 1);
    let tmp1 = tmp1.reshape([ngrids * nvar, natm * 3]);
    let drho2 = drho2.reshape([ngrids * nvar, natm * 3]);
    let tmp2 = tmp1.t() % drho2;

    tmp2.reshape([3, natm, 3, natm]).transpose([0, 2, 1, 3]).into_contig(ColMajor)
}

/// fxc contribution to the UKS XC skeleton Hessian: the four spin-pair sum of
/// single-spin `get_de_fxc` blocks,
/// `einsum("gxy, gxtA, gysB -> tsAB", wf, drho, drho)` per (s1, s2) pair.
///
/// # Parameters
///
/// - `wf` : shape `[ngrids, nvar, 2, nvar, 2]` (g, x, s1, y, s2).  Grid-weighted spin-polarized fxc
///   kernel.
/// - `drhoα`, `drhoβ` : shape `[ngrids, nvar, 3, natm]`, per-spin outputs of [`get_drho`].
///
/// # Returns
///
/// - `de_fxc` : shape `[3, 3, natm, natm]` (t, s, A, B), spin-pair summed.
pub fn get_de_fxc_uks(wf: TsrView, drhoα: TsrView, drhoβ: TsrView) -> Tsr {
    let de_αα = get_de_fxc_uks_inner(wf.i((.., .., α, .., α)), drhoα.view(), drhoα.view());
    let de_αβ = get_de_fxc_uks_inner(wf.i((.., .., α, .., β)), drhoα.view(), drhoβ.view());
    let de_βα = get_de_fxc_uks_inner(wf.i((.., .., β, .., α)), drhoβ.view(), drhoα.view());
    let de_ββ = get_de_fxc_uks_inner(wf.i((.., .., β, .., β)), drhoβ.view(), drhoβ.view());

    &de_αα + &de_αβ + &de_βα + &de_ββ
}

/// fxc contribution to the per-atom skeleton derivative of the Vxc Fock
/// matrices for UKS — the spin-coupled piece.  Unlike the spin-diagonal
/// [`get_vmat_vxc`] (reused from RKS per spin), the fxc contraction here
/// couples the two spin channels:
/// `field_α = wf_αα @ drho_α + wf_αβ @ drho_β` and symmetrically for beta,
/// one [`xc_fock_stack`] call per spin per atom row.
///
/// # Parameters
///
/// - `xc_type` : density family.
/// - `ao` : shape `[ngrids, nao, ncomp]`; reads the leading `num_ao_comp` channels (1 for RHO, 4
///   for SIGMA/TAU), through [`xc_fock_stack`].
/// - `drhoα`, `drhoβ` : shape `[ngrids, nvar, 3, natm]`, per-spin outputs of [`get_drho`].
/// - `wf` : shape `[ngrids, nvar, 2, nvar, 2]`.  Grid-weighted spin-polarized fxc kernel.
///
/// # Returns
///
/// - `vmat_fxc_α`, `vmat_fxc_β` : shape `[nao, nao, 3, natm]` each, assembled across the AO axes
///   (bra + ket).
pub fn get_vmat_fxc_uks(xc_type: XCDenType, ao: TsrView, drhoα: TsrView, drhoβ: TsrView, wf: TsrView) -> (Tsr, Tsr) {
    let natm = drhoα.shape()[3];
    let nao = ao.shape()[1];
    let device = ao.device();

    let mut vmat_fxc_α: Tsr = rt::zeros(([nao, nao, 3, natm], device));
    let mut vmat_fxc_β: Tsr = rt::zeros(([nao, nao, 3, natm], device));

    for A in 0..natm {
        // fields [g, x, t] = sum_{σ', y} wf[g, x, σ, y, σ'] drho_σ'[g, y, t, A], passed
        // unscaled (the 0.5/0.25 factors are absorbed by `xc_fock_stack`; see
        // [`get_vmat_fxc`])
        let field_α = rt::vecdot(wf.i((.., .., α, .., α)), drhoα.i((.., None, .., .., A)), 2)
            + rt::vecdot(wf.i((.., .., α, .., β)), drhoβ.i((.., None, .., .., A)), 2);
        let field_β = rt::vecdot(wf.i((.., .., β, .., α)), drhoα.i((.., None, .., .., A)), 2)
            + rt::vecdot(wf.i((.., .., β, .., β)), drhoβ.i((.., None, .., .., A)), 2);
        *&mut vmat_fxc_α.i_mut((.., .., .., A)) += &xc_fock_stack(xc_type, ao.view(), field_α.view());
        *&mut vmat_fxc_β.i_mut((.., .., .., A)) += &xc_fock_stack(xc_type, ao.view(), field_β.view());
    }

    (vmat_fxc_α, vmat_fxc_β)
}

/// Per-atom skeleton derivative of the per-spin Vxc Fock matrices: the sum of
/// the spin-coupled fxc contribution ([`get_vmat_fxc_uks`]) and the per-spin
/// basis-derivative (ipip) contribution ([`get_vmat_vxc`] reused from RKS).
/// Each part is assembled independently (bra + ket) and summed per spin.
#[allow(clippy::too_many_arguments)]
pub fn get_vmat_deriv1_uks(
    xc_type: XCDenType,
    ao: TsrView,
    drhoα: TsrView,
    drhoβ: TsrView,
    wf: TsrView,
    vmat_ip_α: TsrView,
    vmat_ip_β: TsrView,
    aoslices: &[[usize; 4]],
) -> (Tsr, Tsr) {
    let (vmat_fxc_α, vmat_fxc_β) = get_vmat_fxc_uks(xc_type, ao, drhoα, drhoβ, wf);
    let vmat_vxc_α = get_vmat_vxc(vmat_ip_α, aoslices);
    let vmat_vxc_β = get_vmat_vxc(vmat_ip_β, aoslices);
    (&vmat_fxc_α + &vmat_vxc_α, &vmat_fxc_β + &vmat_vxc_β)
}

/* #endregion */

/* #region becke grid-shift parts: skeleton hessian */

/// `de_becke_atom_1`, UKS spin extension:
/// `einsum("g, txg, xyg, Bsyg -> Bts", w, prho_l, fxc[s1, :, s2, :], drho_r)`
/// summed over the four spin pairs (s1, s2) — the same coupling structure as
/// [`get_de_fxc_uks`].  Each pair reuses the single-spin
/// [`get_de_becke_atom_1`] on the `fxc[s1, :, s2, :]` spin block; the free
/// direction axes `s`/`t` interchange is inherited per pair and is equivalent
/// under the driver's symmetrisation.
///
/// # Parameters
///
/// - `w` : shape `[ngrids]`.  Grid weights of the chunk.
/// - `prhoα`, `prhoβ` : shape `[ngrids, nvar, 3]` (g, x, t).  Spatial density derivative `d rho_σ /
///   dr`, per spin.
/// - `fxc` : shape `[ngrids, nvar, 2, nvar, 2]`.
/// - `drhoα`, `drhoβ` : shape `[ngrids, nvar, 3, natm]`.
///
/// # Returns
///
/// - `de_becke_atom_1` : shape `[3, 3, natm]` (t, s, A), for the last (B) axis scatter of the `[3,
///   3, natm, natm]` accumulator (see [`get_de_becke_atom_1`]).
pub fn get_de_becke_atom_1_uks(
    w: TsrView,
    prhoα: TsrView,
    prhoβ: TsrView,
    fxc: TsrView,
    drhoα: TsrView,
    drhoβ: TsrView,
) -> Tsr {
    let [_, _, _, natm] = drhoα.shape().iter().cloned().collect_array().unwrap();
    let device = drhoα.device().clone();

    let mut de_becke_atom_1: Tsr = rt::zeros(([3, 3, natm], &device));
    for (prho_l, s1) in [(&prhoα, α), (&prhoβ, β)] {
        for (drho_r, s2) in [(&drhoα, α), (&drhoβ, β)] {
            let term = get_de_becke_atom_1(w.view(), prho_l.view(), fxc.i((.., .., s1, .., s2)), drho_r.view());
            de_becke_atom_1 = &de_becke_atom_1 + &term;
        }
    }
    de_becke_atom_1
}

/// `de_becke_atom_2`, UKS spin extension:
/// `einsum("Bsg, xg, txg -> Bts", dw, vxc[σ], prho_σ)` summed over the two
/// spins.  Each spin reuses the single-spin [`get_de_becke_atom_2`].
///
/// # Parameters
///
/// - `dw` : shape `[ngrids, 3, natm]` (g, t, A).  Grid-first Becke `dw` (see
///   [`make_hessian_setup_chunk_becke_uks`]).
/// - `vxc` : shape `[ngrids, nvar, 2]`.
/// - `prhoα`, `prhoβ` : shape `[ngrids, nvar, 3]` (g, x, t).  Spatial density derivative `d rho_σ /
///   dr`.
///
/// # Returns
///
/// - `de_becke_atom_2` : shape `[3, 3, natm]` (t, s, A), for the last (B) axis scatter.
pub fn get_de_becke_atom_2_uks(dw: TsrView, vxc: TsrView, prhoα: TsrView, prhoβ: TsrView) -> Tsr {
    let term_α = get_de_becke_atom_2(dw.view(), vxc.i((.., .., α)), prhoα);
    let term_β = get_de_becke_atom_2(dw.view(), vxc.i((.., .., β)), prhoβ);
    &term_α + &term_β
}

/// `de_becke_atom_3`, UKS spin extension:
/// `einsum("g, xyg, syg, txg -> ts", w, fxc[s1, :, s2, :], prho_r, prho_l)`
/// summed over the four spin pairs — same structure as the single-spin
/// [`get_de_becke_atom_3`], but with the two prho slots filled by different
/// spins, so the body is spelled out instead of reusing the single-spin
/// function.
///
/// # Parameters
///
/// - `w` : shape `[ngrids]`.  Grid weights of the chunk.
/// - `prhoα`, `prhoβ` : shape `[ngrids, nvar, 3]` (g, x, t).  Spatial density derivative `d rho_σ /
///   dr`.
/// - `fxc` : shape `[ngrids, nvar, 2, nvar, 2]`.
///
/// # Returns
///
/// - `de_becke_atom_3` : shape `[3, 3]` (t, s), for the `[atm_idx, atm_idx]` diagonal block
///   scatter.
pub fn get_de_becke_atom_3_uks(w: TsrView, prhoα: TsrView, prhoβ: TsrView, fxc: TsrView) -> Tsr {
    let device = prhoα.device().clone();

    let mut de_becke_atom_3: Tsr = rt::zeros(([3, 3], &device));
    for (prho_l, s1) in [(&prhoα, α), (&prhoβ, β)] {
        for (prho_r, s2) in [(&prhoα, α), (&prhoβ, β)] {
            // fp [g, x, t] = sum_y fxc[s1, x, s2, y] prho_r[g, y, t]
            let fp = rt::vecdot(fxc.i((.., .., s1, .., s2, None)), prho_r.i((.., None, .., ..)), 2);
            // wprho_l [g, x, s]
            let wprho_l = prho_l * w.i((.., None, None));
            // t6 [t, s] = sum_{g, x} fp[g, x, t] wprho_l[g, x, s]
            let term = rt::vecdot(fp.i((.., .., None, ..)), wprho_l.i((.., .., .., None)), ([0, 1], [0, 1]));
            de_becke_atom_3 = &de_becke_atom_3 + &term;
        }
    }
    de_becke_atom_3
}

/// `de_becke_vxc_diag` / `de_becke_vxc_off`, UKS spin extension: same
/// structure as the single-spin `get_de_becke_vxc_parts`, with the
/// per-spin kernels (each built from its own `wv` weighting and `ao_dm0`
/// contraction in [`make_hessian_setup_chunk_becke_uks`]) summed before the
/// [`contract_pvxc`] scatter — the contraction is linear, so spin-summing
/// first is equivalent to contracting each spin separately.
///
/// # Parameters
///
/// - `dao_vxc_diag_α`, `dao_vxc_diag_β` : shape `[nao, 6]`, per-spin outputs of
///   [`make_dao_vxc_diag`].
/// - `dao_vxc_off_α`, `dao_vxc_off_β` : shape `[nao, nao, 3, 3]` (u, v, t, s), per-spin outputs of
///   [`make_dao_vxc_off`].
/// - `dm0α`, `dm0β` : shape `[nao, nao]`.  Per-spin density matrices in AO basis.
/// - `atm_idx` : atom that generated the chunk's grids.
/// - `aoslices` : shape `[natm, 4]`.
///
/// # Returns
///
/// - `de_becke_vxc_diag` : shape `[3, 3, natm]`, from the spin-summed `dao_vxc_diag` expanded to
///   dense (3, 3) pairs; the 0.5 factor of the grid-shift terms lives in [`contract_pvxc`].
/// - `de_becke_vxc_off` : shape `[3, 3, natm]`, from the spin-summed `dao_vxc_off` contracted with
///   the matching per-spin `dm0` on the leading AO axis.
///
/// Both for the last (B) axis scatter (see [`contract_pvxc`]).
#[allow(clippy::too_many_arguments)]
pub fn get_de_becke_vxc_parts_uks(
    dao_vxc_diag_α: TsrView,
    dao_vxc_diag_β: TsrView,
    dao_vxc_off_α: TsrView,
    dao_vxc_off_β: TsrView,
    dm0α: TsrView,
    dm0β: TsrView,
    atm_idx: usize,
    aoslices: &[[usize; 4]],
) -> (Tsr, Tsr) {
    let nao = dao_vxc_diag_α.shape()[0];

    // pvxc_diag [nao, 3, 3] = (dao_vxc_diag_α + dao_vxc_diag_β)[IDX_PAIR_TS]; the symmetric
    // pairs make it invariant under (t, s), so it doubles as the interchanged
    // kernel
    const IDX_PAIR_TS: [usize; 9] = [0, 1, 2, 1, 3, 4, 2, 4, 5];
    let pvxc_diag: Tsr = (&dao_vxc_diag_α + &dao_vxc_diag_β).index_select(1, IDX_PAIR_TS).into_shape([nao, 3, 3]);

    // pvxc_off[u, t, s] = sum_v (dao_vxc_off_α[u, v, t, s] dm0α[u, v]
    //                         + dao_vxc_off_β[u, v, t, s] dm0β[u, v])
    let pvxc_off: Tsr = rt::vecdot(dao_vxc_off_α, dm0α, 1) + rt::vecdot(dao_vxc_off_β, dm0β, 1);
    (contract_pvxc(pvxc_diag.view(), atm_idx, aoslices), contract_pvxc(pvxc_off.view(), atm_idx, aoslices))
}

/* #endregion */

/* #region becke grid-shift parts: f1ao (CP-KS RHS) */

/// f1ao-level Becke grid-shift parts of the per-spin skeleton Vxc Fock
/// derivatives: the per-spin increments `vmat_becke_dw_σ + vmat_becke_vxc_σ +
/// vmat_becke_fxc_σ` that restore translational invariance of each spin's
/// `vmat_deriv1_σ` (the DFT part of the CP-KS right-hand side f1ao), i.e.
/// `sum_A vmat_deriv1_grid_σ[A] ~ 0`.
///
/// - `vmat_becke_dw` (weight part): per-spin [`xc_fock_stack`] built with the Becke `dw[g, t, A]`
///   slices as the weight field and the spin's own `vxc` field; every grid of the chunk contributes
///   to every atom's row.
/// - `vmat_becke_vxc`: the chunk's per-spin `vmat_ip` symmetrised in AO — the chunk holds one
///   atom's grids, so `vmat_ip` already is the per-grid-atom kernel.
/// - `vmat_becke_fxc`: the fxc kernel spin-coupled and folded with the total spatial density
///   derivative of BOTH spins (`fxc[σ, :, σ', :]` against `prho_σ'`), contracted as an
///   [`xc_fock_stack`] on the chunk weights.  This mirrors the [`get_vmat_fxc_uks`] spin coupling.
///
/// # Parameters
///
/// - `xc_type` : density family.
/// - `ao` : shape `[ngrids, nao, ncomp]`; reads the leading `num_ao_comp` channels (1 for RHO, 4
///   for SIGMA/TAU), through [`xc_fock_stack`].
/// - `vxc` : shape `[ngrids, nvar, 2]`.
/// - `fxc` : shape `[ngrids, nvar, 2, nvar, 2]`.
/// - `prhoα`, `prhoβ` : shape `[ngrids, nvar, 3]` (g, x, t).  Spatial density derivative `d rho_σ /
///   dr`, per spin.
/// - `w` : shape `[ngrids]`.  Grid weights of the chunk.
/// - `dw` : shape `[ngrids, 3, natm]` (g, t, A).  Grid-first Becke `dw` (see
///   [`get_de_becke_atom_2_uks`]).
/// - `vmat_ip_α`, `vmat_ip_β` : shape `[nao, nao, 3]`, per-spin outputs of [`get_vmat_ip`].
///
/// # Returns
///
/// - `vmat_becke_dw_α`, `vmat_becke_dw_β` : shape `[nao, nao, 3, natm]`, filled on all rows.
/// - `vmat_becke_vxc_α/β`, `vmat_becke_fxc_α/β` : shape `[nao, nao, 3]` — the chunk atom's row,
///   scattered into the `[nao, nao, 3, natm]` accumulators by the driver.
#[allow(clippy::too_many_arguments)]
pub fn get_vmat_becke_parts_uks(
    xc_type: XCDenType,
    ao: TsrView,
    vxc: TsrView,
    fxc: TsrView,
    prhoα: TsrView,
    prhoβ: TsrView,
    w: TsrView,
    dw: TsrView,
    vmat_ip_α: TsrView,
    vmat_ip_β: TsrView,
) -> (Tsr, Tsr, Tsr, Tsr, Tsr, Tsr) {
    let nao = ao.shape()[1];
    let natm = dw.shape()[2];
    let device = ao.device().clone();

    // dw part: per-spin XC-style Fock with the becke dw[·, ·, A] rows as weights (all rows);
    // wv_σ [g, x, t] = vxc[g, x, σ] dw[g, t, A]
    let mut vmat_becke_dw_α = rt::zeros(([nao, nao, 3, natm], &device));
    let mut vmat_becke_dw_β = rt::zeros(([nao, nao, 3, natm], &device));
    for A in 0..natm {
        let wv_α = vxc.i((.., .., α, None)) * dw.i((.., None, .., A));
        *&mut vmat_becke_dw_α.i_mut((.., .., .., A)) += &xc_fock_stack(xc_type, ao.view(), wv_α.view());
        let wv_β = vxc.i((.., .., β, None)) * dw.i((.., None, .., A));
        *&mut vmat_becke_dw_β.i_mut((.., .., .., A)) += &xc_fock_stack(xc_type, ao.view(), wv_β.view());
    }

    // vxc part: chunk's per-spin vmat_ip symmetrised in AO
    let vmat_becke_vxc_α = &vmat_ip_α + vmat_ip_α.swapaxes(0, 1);
    let vmat_becke_vxc_β = &vmat_ip_β + vmat_ip_β.swapaxes(0, 1);

    // fxc part: fxc spin-coupled and folded with prho of both spins, contracted on the chunk
    // weights; fxc_prho_σ [g, x, t] = sum_{σ', y} fxc[σ, x, σ', y] prho_σ'[g, y, t]
    let fxc_prho_α: Tsr = rt::vecdot(fxc.i((.., .., α, .., α)), prhoα.i((.., None, ..)), 2)
        + rt::vecdot(fxc.i((.., .., α, .., β)), prhoβ.i((.., None, ..)), 2);
    let fxc_prho_β: Tsr = rt::vecdot(fxc.i((.., .., β, .., α)), prhoα.i((.., None, ..)), 2)
        + rt::vecdot(fxc.i((.., .., β, .., β)), prhoβ.i((.., None, ..)), 2);
    let wv_α: Tsr = fxc_prho_α * w.i((.., None, None));
    let wv_β: Tsr = fxc_prho_β * w.i((.., None, None));
    let vmat_becke_fxc_α = xc_fock_stack(xc_type, ao.view(), wv_α.view());
    let vmat_becke_fxc_β = xc_fock_stack(xc_type, ao.view(), wv_β.view());

    (vmat_becke_dw_α, vmat_becke_dw_β, vmat_becke_vxc_α, vmat_becke_vxc_β, vmat_becke_fxc_α, vmat_becke_fxc_β)
}

/* #endregion */

/* #region per-chunk evaluation */

/// Per-chunk evaluation of all UKS skeleton ingredients with the grid-shift.
/// Unrestricted sibling of
/// [`make_hessian_setup_chunk_becke`](super::hess_rks::make_hessian_setup_chunk_becke):
/// the chunk must hold grids of the single atom `atm_idx` (ByAtom attribution)
/// and computes its own AO integrals through `ni.get_cached_ao`.
///
/// With `grid_shift = false` only the grid-fixed terms are evaluated; a chunk
/// attributed to [`NO_ATM`] is then legal, and contributes only through the
/// grid-fixed terms.
///
/// # Parameters
///
/// - `mol` : molecule (AO slices, dimensions).
/// - `xc_func_list` : list of `(scale, functional)` pairs.
/// - `ni` : numerical-integration driver restricted to the chunk's grids.
/// - `dm0α`, `dm0β` : shape `[nao, nao]`.  Per-spin reference density matrices in AO basis.
/// - `atm_idx` : atom that generated the chunk's grids.
/// - `tables` : precomputed molecular tables of the Becke partition (shared across chunks); must be
///   `Some` when `grid_shift` is set.
/// - `hardness` : Becke switch-function hardness.
/// - `grid_shift` : evaluate the Becke grid-shift terms.
///
/// # Returns
///
/// Map from key to the chunk's contribution.  Full-grid keys accumulate
/// across chunks by a plain sum; grid-atom keys carry only the chunk atom's
/// contribution and are scattered by [`make_hessian_setup_becke_uks`]:
///
/// - Sum: `de_fxc`, `de_vxc_diag_a/b`,
///   `de_vxc_off_a/b`, `de_becke_full_1/2` `[3, 3, natm, natm]`; `vmat_ip_a/b [nao, nao, 3]`;
///   `vmat_fxc_a/b`, `vmat_vxc_a/b`, `vmat_deriv1_a/b`, `vmat_becke_dw_a/b` `[nao, nao, 3, natm]`.
/// - Scatter into column `B = atm_idx` (direction axes interchanged): `de_becke_atom_1/2`,
///   `de_becke_vxc_diag/off` `[3, 3, natm]`; `vmat_becke_vxc_a/b`, `vmat_becke_fxc_a/b` `[nao, nao,
///   3]`.
/// - Scatter into the `[atm_idx, atm_idx]` diagonal block: `de_becke_atom_3` `[3, 3]`.
///
/// The `de_becke_*` / `vmat_becke_*` keys are present only with `grid_shift`.
#[allow(clippy::too_many_arguments)]
pub fn make_hessian_setup_chunk_becke_uks(
    mol: &CInt,
    xc_func_list: &[(f64, LibXCFunctional)],
    ni: &mut NIMatmul,
    dm0α: TsrView,
    dm0β: TsrView,
    atm_idx: usize,
    tables: Option<&BeckeMolTables>,
    hardness: usize,
    grid_shift: bool,
) -> HashMap<&'static str, Tsr> {
    assert!(atm_idx != NO_ATM || !grid_shift, "the grid-shift terms require atom-attributed grids (ByAtom)");
    let natm = mol.natm();
    let device = dm0α.device().clone();
    let aoslices = mol.aoslice_by_atom();
    let xc_type = determine_den_type_from_list(&xc_func_list.iter().map(|(_, f)| f).collect_vec());

    // owned copies of the chunk's grid data; `ni` stays borrowed by the AO cache
    let grid_coords = ni.coords.clone();
    let quadrature_weights = ni.quadrature_weights.clone();
    let weights_data = ni.weights.clone();
    let ngrids = weights_data.len();

    // --- ao, rho, exc, vxc, fxc --- //

    let ao = ni.get_cached_ao(get_hess_ao_deriv(xc_type));
    let ncomp_ao_dm0 = get_hess_ncomp_ao_dm0(xc_type);
    let ao_dm0α = index!(ao, ..ncomp_ao_dm0) % &dm0α;
    let ao_dm0β = index!(ao, ..ncomp_ao_dm0) % &dm0β;
    let (rho, exc, vxc, fxc) = get_rho_exc_vxc_fxc_uks(xc_func_list, ao.view(), ao_dm0α.view(), ao_dm0β.view());

    let weights = rt::asarray((weights_data, &device));
    let wvα = &weights * vxc.i((.., .., α));
    let wvβ = &weights * vxc.i((.., .., β));
    let wf = &weights * &fxc;

    // --- drho, prho (per spin; the skeleton derivative is spin-diagonal) --- //

    let drhoα = get_drho(xc_type, ao.view(), ao_dm0α.view(), &aoslices);
    let drhoβ = get_drho(xc_type, ao.view(), ao_dm0β.view(), &aoslices);
    // prho_σ [ngrids, nvar, 3] = d rho_σ / dr = -(drho_σ summed over atoms)
    let prhoα = -drhoα.sum_axes(3);
    let prhoβ = -drhoβ.sum_axes(3);

    // --- without-becke parts --- //

    let de_fxc = get_de_fxc_uks(wf.view(), drhoα.view(), drhoβ.view());

    let dao_vxc_diag_α = make_dao_vxc_diag(xc_type, ao.view(), ao_dm0α.view(), wvα.view());
    let dao_vxc_diag_β = make_dao_vxc_diag(xc_type, ao.view(), ao_dm0β.view(), wvβ.view());
    let de_vxc_diag_α = get_de_vxc_diag(dao_vxc_diag_α.view(), &aoslices);
    let de_vxc_diag_β = get_de_vxc_diag(dao_vxc_diag_β.view(), &aoslices);

    let dao_vxc_off_α = make_dao_vxc_off(xc_type, ao.view(), wvα.view());
    let dao_vxc_off_β = make_dao_vxc_off(xc_type, ao.view(), wvβ.view());
    let de_vxc_off_α = get_de_vxc_off(dao_vxc_off_α.view(), dm0α.view(), &aoslices);
    let de_vxc_off_β = get_de_vxc_off(dao_vxc_off_β.view(), dm0β.view(), &aoslices);

    let vmat_ip_α = get_vmat_ip(xc_type, ao.view(), wvα.view());
    let vmat_ip_β = get_vmat_ip(xc_type, ao.view(), wvβ.view());

    // per-atom skeleton Vxc Fock derivative per spin: spin-coupled fxc part plus per-spin
    // basis-derivative (ipip) part; both are already assembled across the AO axes
    let (vmat_fxc_α, vmat_fxc_β) = get_vmat_fxc_uks(xc_type, ao.view(), drhoα.view(), drhoβ.view(), wf.view());
    let vmat_vxc_α = get_vmat_vxc(vmat_ip_α.view(), &aoslices);
    let vmat_vxc_β = get_vmat_vxc(vmat_ip_β.view(), &aoslices);
    let vmat_deriv1_α = &vmat_fxc_α + &vmat_vxc_α;
    let vmat_deriv1_β = &vmat_fxc_β + &vmat_vxc_β;

    let mut becke_entries: Vec<(&'static str, Tsr)> = Vec::new();
    if grid_shift {
        // --- becke partition: dw in full, ddw only through the cddw contraction --- //

        // cddw (nset = 1): the only ddw consumer is de_becke_full_2 = ddw . (exc * (rhoa[0] +
        // rhob[0]))
        let cddw = (&exc * (rho.i((.., 0, α)) + rho.i((.., 0, β)))).into_vec();

        let tables = tables.expect("BeckeMolTables must be given when grid_shift is set");
        let boundaries = by_atom_chunk(natm, atm_idx, ngrids);
        let deriv_arg = BeckePartitionArg {
            output_w: false,
            output_dw: true,
            output_ddw: false,
            contract_w: None,
            contract_dw: None,
            contract_ddw: Some(&cddw),
        };
        let becke_result = becke_partition_with_tables(
            tables,
            &grid_coords,
            AtmIndices::ByAtom(&boundaries),
            &quadrature_weights,
            hardness,
            1024,
            2,
            Some(deriv_arg),
        );
        // dw flat is C-order [A, t, g] == Fortran-order [g, t, A]
        let dw = rt::asarray((becke_result.dw.unwrap(), [ngrids, 3, natm].f(), &device));

        // --- becke grid-shift parts --- //

        // de_becke_full_1: einsum("Atg, xg, Bsxg -> ABts", dw, vxc[σ], drho_σ), summed over σ;
        // [t, s, A, B] = sum_g dw[g, t, A] vxc_drho[g, s, B], where
        // vxc_drho [g, s, B] = sum_x vxc[σ][g, x] drho_σ[g, x, s, B]
        let de_becke_full_1 = {
            let vxc_drho_α = rt::vecdot(drhoα.view(), vxc.i((.., .., α)), 1);
            let vxc_drho_β = rt::vecdot(drhoβ.view(), vxc.i((.., .., β)), 1);
            let vxc_drho = &vxc_drho_α + &vxc_drho_β;
            rt::vecdot(dw.i((.., .., None, .., None)), vxc_drho.i((.., None, .., None, ..)), 0)
        };

        // de_becke_full_2: einsum("AtBsg, g, g -> ABts", ddw, exc, rhoa[0] + rhob[0]) via the
        // cddw contraction above (nset = 1), naturally symmetric;
        // ddc flat is C-order [A, t, B, s, iset] == Fortran-order [iset, s, B, t, A]
        let de_becke_full_2 = rt::asarray((becke_result.ddc.unwrap(), [3, natm, 3, natm].f(), &device))
            .transpose([2, 0, 3, 1])
            .into_contig(ColMajor);

        // grid-atom parts: compact tensors for the chunk atom's row (resp. diagonal
        // block); the scatter into `[3, 3, natm, natm]` is done by
        // `make_hessian_setup_becke_uks`
        let de_becke_atom_1 =
            get_de_becke_atom_1_uks(weights.view(), prhoα.view(), prhoβ.view(), fxc.view(), drhoα.view(), drhoβ.view());
        let de_becke_atom_2 = get_de_becke_atom_2_uks(dw.view(), vxc.view(), prhoα.view(), prhoβ.view());
        let de_becke_atom_3 = get_de_becke_atom_3_uks(weights.view(), prhoα.view(), prhoβ.view(), fxc.view());

        let (de_becke_vxc_diag, de_becke_vxc_off) = get_de_becke_vxc_parts_uks(
            dao_vxc_diag_α.view(),
            dao_vxc_diag_β.view(),
            dao_vxc_off_α.view(),
            dao_vxc_off_β.view(),
            dm0α.view(),
            dm0β.view(),
            atm_idx,
            &aoslices,
        );

        let (vmat_becke_dw_α, vmat_becke_dw_β, vmat_becke_vxc_α, vmat_becke_vxc_β, vmat_becke_fxc_α, vmat_becke_fxc_β) =
            get_vmat_becke_parts_uks(
                xc_type,
                ao.view(),
                vxc.view(),
                fxc.view(),
                prhoα.view(),
                prhoβ.view(),
                weights.view(),
                dw.view(),
                vmat_ip_α.view(),
                vmat_ip_β.view(),
            );

        becke_entries.extend([
            ("de_becke_full_1", de_becke_full_1),
            ("de_becke_full_2", de_becke_full_2),
            ("de_becke_atom_1", de_becke_atom_1),
            ("de_becke_atom_2", de_becke_atom_2),
            ("de_becke_atom_3", de_becke_atom_3),
            ("de_becke_vxc_diag", de_becke_vxc_diag),
            ("de_becke_vxc_off", de_becke_vxc_off),
            ("vmat_becke_dw_a", vmat_becke_dw_α),
            ("vmat_becke_dw_b", vmat_becke_dw_β),
            ("vmat_becke_vxc_a", vmat_becke_vxc_α),
            ("vmat_becke_vxc_b", vmat_becke_vxc_β),
            ("vmat_becke_fxc_a", vmat_becke_fxc_α),
            ("vmat_becke_fxc_b", vmat_becke_fxc_β),
        ]);
    }

    let mut result = HashMap::from([
        ("de_fxc", de_fxc),
        ("de_vxc_diag_a", de_vxc_diag_α),
        ("de_vxc_diag_b", de_vxc_diag_β),
        ("de_vxc_off_a", de_vxc_off_α),
        ("de_vxc_off_b", de_vxc_off_β),
        ("vmat_ip_a", vmat_ip_α),
        ("vmat_ip_b", vmat_ip_β),
        ("vmat_fxc_a", vmat_fxc_α),
        ("vmat_fxc_b", vmat_fxc_β),
        ("vmat_vxc_a", vmat_vxc_α),
        ("vmat_vxc_b", vmat_vxc_β),
        ("vmat_deriv1_a", vmat_deriv1_α),
        ("vmat_deriv1_b", vmat_deriv1_β),
    ]);
    result.extend(becke_entries);
    result
}

/* #endregion */

/* #region parallel driver */

/// `x + x.transpose(1, 0, 3, 2)` on a `[3, 3, natm, natm]` (tsAB) tensor: the
/// `(A, t) <-> (B, s)` symmetrisation.
fn symmetrize_ts_ab(x: Tsr) -> Tsr {
    &x + x.transpose([1, 0, 3, 2])
}

/// Parallel driver for all UKS DFT skeleton ingredients with the grid-shift.
/// Unrestricted sibling of
/// [`make_hessian_setup_becke`](super::hess_rks::make_hessian_setup_becke),
/// sharing its flat chunk-level parallelization: [`quad_split_by_atom`] at
/// `nchunk` granularity produces `(atm_idx, start, end)` work units that never
/// cross an atom boundary, and each unit evaluates
/// [`make_hessian_setup_chunk_becke_uks`] (its own AO integrals included)
/// inside one flat par_iter.
///
/// With `grid_shift = false` the Becke terms are skipped entirely:
/// `de_xc_skeleton` equals `de_xc_skeleton_no_becke` and
/// `vmat_deriv1_grid_a/b` equal `vmat_deriv1_a/b`.  The Becke parameters are
/// fixed by grid generation (see
/// [`make_hessian_setup_becke`](super::hess_rks::make_hessian_setup_becke)).
///
/// # Parameters
///
/// - `mol` : molecule.
/// - `xc_func_list` : list of `(scale, functional)` pairs.
/// - `ni` : numerical-integration driver over the full grid; its `atm_idx` must be atom-grouped
///   (non-decreasing) and, with `grid_shift`, attribute every grid to an atom.
/// - `dm0α`, `dm0β` : shape `[nao, nao]`.  Per-spin reference density matrices in AO basis.
/// - `grid_shift` : evaluate the Becke grid-shift terms; requires full atom attribution of the
///   grids.
/// - `atm_list` : must be `None` or the full atom list.
/// - `verbose` : print per-chunk progress.
///
/// # Returns
///
/// - `result` : all keys of [`make_hessian_setup_chunk_becke_uks`] accumulated over chunks
///   (full-grid keys summed, grid-atom keys scattered into the chunk atom's column of the last (B)
///   axis), plus the assemblies: `de_xc_skeleton_no_becke [3, 3, natm, natm]` = `de_vxc_diag_a +
///   de_vxc_off_a + de_vxc_diag_b + de_vxc_off_b + de_fxc`; `de_xc_skeleton [3, 3, natm, natm]`
///   with all `de_becke_*` grid-shift parts added (translationally invariant);
///   `vmat_deriv1_grid_a/b [nao, nao, 3, natm]` = `vmat_deriv1_a/b + vmat_becke_dw_a/b +
///   vmat_becke_vxc_a/b + vmat_becke_fxc_a/b` (translationally invariant per spin).  The keys
///   `de_becke_full_1/atom_1/atom_2/vxc_diag/vxc_off` are symmetrised under `(A, t) <-> (B, s)`;
///   `de_becke_full_2` is naturally symmetric.
/// - `timing` : wall-time progress entries.
pub fn make_hessian_setup_becke_uks(
    mol: &CInt,
    xc_func_list: &[(f64, LibXCFunctional)],
    ni: &mut NIMatmul,
    dm0α: TsrView,
    dm0β: TsrView,
    grid_shift: bool,
    atm_list: Option<&[usize]>,
    verbose: bool,
) -> (HashMap<&'static str, Tsr>, IndexMap<&'static str, f64>) {
    assert!(!xc_func_list.is_empty(), "xc_func_list must not be empty");
    assert!(
        atm_list.is_none() || atm_list.unwrap().len() == mol.natm(),
        "the DFT skeleton hessian currently requires the full atom list"
    );
    let natm = mol.natm();
    let nao = mol.nao();
    let ngrids = ni.weights.len();
    let nchunk = ni.nchunk;
    let device = dm0α.device().clone();

    // per-atom grid boundaries, deduced from the (atom-grouped) attribution
    let atm_quad_split = try_atm_quad_split(&ni.atm_idx, natm)
        .expect("the hessian driver requires atom-grouped grids (NIMatmul::atm_idx non-decreasing)");
    if grid_shift {
        assert!(
            atm_quad_split[natm] == ngrids,
            "the grid-shift terms require full atom attribution of the grids; \
             disable grid_shift_deriv or use standard generated grids"
        );
    }

    // molecular tables of the Becke partition, built once and shared by all
    // chunks; the radii table derives from the Bragg radii of the nuclear
    // charges (see make_hessian_setup_becke)
    let hardness = BECKE_HARDNESS;
    let tables = grid_shift.then(|| {
        let proton_charges: Vec<i32> = mol.atom_charges().iter().map(|&c| c as i32).collect_vec();
        let adjustment_factor = gen_adjustment_factor(&proton_charges, ni.radii_adjust);
        BeckeMolTables::new(&mol.atom_coords(), &adjustment_factor, 2)
    });

    let chunks = quad_split_by_atom(&atm_quad_split, ngrids, nchunk);
    let nchunks = chunks.len();

    let de_fxc: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_vxc_diag_α: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_vxc_diag_β: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_vxc_off_α: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_vxc_off_β: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let vmat_ip_α: Tsr = rt::zeros(([nao, nao, 3], &device));
    let vmat_ip_β: Tsr = rt::zeros(([nao, nao, 3], &device));
    let vmat_fxc_α: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_fxc_β: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_vxc_α: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_vxc_β: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_deriv1_α: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_deriv1_β: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let de_becke_full_1: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_becke_full_2: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_becke_atom_1: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_becke_atom_2: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_becke_atom_3: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_becke_vxc_diag: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_becke_vxc_off: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let vmat_becke_dw_α: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_becke_dw_β: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_becke_vxc_α: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_becke_vxc_β: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_becke_fxc_α: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_becke_fxc_β: Tsr = rt::zeros(([nao, nao, 3, natm], &device));

    let timing = Arc::new(Mutex::new(IndexMap::from([("total", 0.0)])));
    let time_total = std::time::Instant::now();

    // serializes the `+=` reductions below
    let guard = Mutex::new(());

    // single-level parallelization by chunk; each task owns its `ni_chunk`
    // and its AO evaluation
    let ni = &*ni;
    let progress = AtomicUsize::new(0);

    chunks.into_par_iter().for_each(|(atm_idx, start, end)| {
        let mut ni_chunk = ni.split_batch(start, end);
        let result_chunk = make_hessian_setup_chunk_becke_uks(
            mol,
            xc_func_list,
            &mut ni_chunk,
            dm0α.view(),
            dm0β.view(),
            atm_idx,
            tables.as_ref(),
            hardness,
            grid_shift,
        );
        // sum the full-grid keys, scatter the grid-atom keys into the chunk
        // atom's column of the last (B) axis
        unsafe {
            let _lock = guard.lock().unwrap();
            *&mut de_fxc.force_mut() += &result_chunk["de_fxc"];
            *&mut de_vxc_diag_α.force_mut() += &result_chunk["de_vxc_diag_a"];
            *&mut de_vxc_diag_β.force_mut() += &result_chunk["de_vxc_diag_b"];
            *&mut de_vxc_off_α.force_mut() += &result_chunk["de_vxc_off_a"];
            *&mut de_vxc_off_β.force_mut() += &result_chunk["de_vxc_off_b"];
            *&mut vmat_ip_α.force_mut() += &result_chunk["vmat_ip_a"];
            *&mut vmat_ip_β.force_mut() += &result_chunk["vmat_ip_b"];
            *&mut vmat_fxc_α.force_mut() += &result_chunk["vmat_fxc_a"];
            *&mut vmat_fxc_β.force_mut() += &result_chunk["vmat_fxc_b"];
            *&mut vmat_vxc_α.force_mut() += &result_chunk["vmat_vxc_a"];
            *&mut vmat_vxc_β.force_mut() += &result_chunk["vmat_vxc_b"];
            *&mut vmat_deriv1_α.force_mut() += &result_chunk["vmat_deriv1_a"];
            *&mut vmat_deriv1_β.force_mut() += &result_chunk["vmat_deriv1_b"];
            if grid_shift {
                *&mut de_becke_full_1.force_mut() += &result_chunk["de_becke_full_1"];
                *&mut de_becke_full_2.force_mut() += &result_chunk["de_becke_full_2"];
                *&mut vmat_becke_dw_α.force_mut() += &result_chunk["vmat_becke_dw_a"];
                *&mut vmat_becke_dw_β.force_mut() += &result_chunk["vmat_becke_dw_b"];
                *&mut de_becke_atom_1.i((Ellipsis, atm_idx)).force_mut() += &result_chunk["de_becke_atom_1"];
                *&mut de_becke_atom_2.i((Ellipsis, atm_idx)).force_mut() += &result_chunk["de_becke_atom_2"];
                *&mut de_becke_atom_3.i((Ellipsis, atm_idx, atm_idx)).force_mut() += &result_chunk["de_becke_atom_3"];
                *&mut de_becke_vxc_diag.i((Ellipsis, atm_idx)).force_mut() += &result_chunk["de_becke_vxc_diag"];
                *&mut de_becke_vxc_off.i((Ellipsis, atm_idx)).force_mut() += &result_chunk["de_becke_vxc_off"];
                *&mut vmat_becke_vxc_α.i((Ellipsis, atm_idx)).force_mut() += &result_chunk["vmat_becke_vxc_a"];
                *&mut vmat_becke_vxc_β.i((Ellipsis, atm_idx)).force_mut() += &result_chunk["vmat_becke_vxc_b"];
                *&mut vmat_becke_fxc_α.i((Ellipsis, atm_idx)).force_mut() += &result_chunk["vmat_becke_fxc_a"];
                *&mut vmat_becke_fxc_β.i((Ellipsis, atm_idx)).force_mut() += &result_chunk["vmat_becke_fxc_b"];
            }
        }
        let ichunk = progress.fetch_add(1, Ordering::Relaxed);
        timing.lock().unwrap().insert("total", time_total.elapsed().as_secs_f64());
        if verbose {
            println!(
                "In make_hessian_setup_becke_uks, chunk {}/{} (atom {atm_idx}): grids {start}..{end}",
                ichunk + 1,
                nchunks,
            );
            println!("  Elapsed time from start (Wall time): {:.4} sec", timing.lock().unwrap()["total"]);
        }
    });

    // final assemblies
    let de_xc_skeleton_no_becke = &de_vxc_diag_α + &de_vxc_off_α + &de_vxc_diag_β + &de_vxc_off_β + &de_fxc;

    let (de_xc_skeleton, vmat_deriv1_grid_α, vmat_deriv1_grid_β, becke_entries) = if grid_shift {
        // (A, t) <-> (B, s) symmetrisation for the becke parts; de_becke_full_2
        // is naturally symmetric
        let de_becke_full_1 = symmetrize_ts_ab(de_becke_full_1);
        let de_becke_atom_1 = symmetrize_ts_ab(de_becke_atom_1);
        let de_becke_atom_2 = symmetrize_ts_ab(de_becke_atom_2);
        let de_becke_vxc_diag = symmetrize_ts_ab(de_becke_vxc_diag);
        let de_becke_vxc_off = symmetrize_ts_ab(de_becke_vxc_off);

        let de_xc_skeleton = &de_xc_skeleton_no_becke
            + &de_becke_full_1
            + &de_becke_full_2
            + &de_becke_atom_1
            + &de_becke_atom_2
            + &de_becke_atom_3
            + &de_becke_vxc_diag
            + &de_becke_vxc_off;
        let vmat_deriv1_grid_α = &vmat_deriv1_α + &vmat_becke_dw_α + &vmat_becke_vxc_α + &vmat_becke_fxc_α;
        let vmat_deriv1_grid_β = &vmat_deriv1_β + &vmat_becke_dw_β + &vmat_becke_vxc_β + &vmat_becke_fxc_β;
        let becke_entries = vec![
            ("de_becke_full_1", de_becke_full_1),
            ("de_becke_full_2", de_becke_full_2),
            ("de_becke_atom_1", de_becke_atom_1),
            ("de_becke_atom_2", de_becke_atom_2),
            ("de_becke_atom_3", de_becke_atom_3),
            ("de_becke_vxc_diag", de_becke_vxc_diag),
            ("de_becke_vxc_off", de_becke_vxc_off),
            ("vmat_becke_dw_a", vmat_becke_dw_α),
            ("vmat_becke_dw_b", vmat_becke_dw_β),
            ("vmat_becke_vxc_a", vmat_becke_vxc_α),
            ("vmat_becke_vxc_b", vmat_becke_vxc_β),
            ("vmat_becke_fxc_a", vmat_becke_fxc_α),
            ("vmat_becke_fxc_b", vmat_becke_fxc_β),
        ];
        (de_xc_skeleton, vmat_deriv1_grid_α, vmat_deriv1_grid_β, becke_entries)
    } else {
        (de_xc_skeleton_no_becke.clone(), vmat_deriv1_α.clone(), vmat_deriv1_β.clone(), Vec::new())
    };

    let mut result = HashMap::from([
        ("de_fxc", de_fxc),
        ("de_vxc_diag_a", de_vxc_diag_α),
        ("de_vxc_diag_b", de_vxc_diag_β),
        ("de_vxc_off_a", de_vxc_off_α),
        ("de_vxc_off_b", de_vxc_off_β),
        ("vmat_ip_a", vmat_ip_α),
        ("vmat_ip_b", vmat_ip_β),
        ("vmat_fxc_a", vmat_fxc_α),
        ("vmat_fxc_b", vmat_fxc_β),
        ("vmat_vxc_a", vmat_vxc_α),
        ("vmat_vxc_b", vmat_vxc_β),
        ("vmat_deriv1_a", vmat_deriv1_α),
        ("vmat_deriv1_b", vmat_deriv1_β),
        ("de_xc_skeleton_no_becke", de_xc_skeleton_no_becke),
        ("de_xc_skeleton", de_xc_skeleton),
        ("vmat_deriv1_grid_a", vmat_deriv1_grid_α),
        ("vmat_deriv1_grid_b", vmat_deriv1_grid_β),
    ]);
    result.extend(becke_entries);

    let timing = timing.lock().unwrap().clone();
    (result, timing)
}

/* #endregion */

/* #region final implementation of UKS Hessian */

/// UKS Hessian XC component with the Becke grid-shift, the unrestricted
/// sibling of [`super::hess_rks::RHessKSNIMatmul`]:
/// [`UHessElecInteractAPI`] with `make_skeleton_hess` returning the
/// translationally invariant `de_xc_skeleton` and `get_deriv1_ao` the
/// translationally invariant `vmat_deriv1_grid_a/b` per spin (their grid-fixed
/// values when `grid_shift` is off).
///
/// The grid data of `ni` carries the atom attribution; grids must be
/// atom-grouped (the ByAtom attribution scheme of `becke_partitioning_deriv`).
pub struct UHessKSNIMatmul<'a> {
    /// Molecule.
    pub mol: CInt,
    /// List of `(scale, functional)` pairs of the XC functional.
    pub xc_func_list: Vec<(f64, LibXCFunctional)>,
    /// Numerical-integration driver over the (atom-grouped) Hessian grid, built from the same
    /// grid data as the response object [`URespKSNIMatmul`] but an independent instance.
    ///
    /// [`URespKSNIMatmul`]: super::resp_uks::URespKSNIMatmul
    pub ni: NIMatmul<'a>,
    /// Evaluate the Becke grid-shift terms; requires full atom attribution of
    /// the grids.
    pub grid_shift: bool,
    /// Print per-chunk progress of the Hessian setup.
    pub verbose: bool,
    /// Skeleton-Hessian intermediates: all keys of [`make_hessian_setup_becke_uks`].
    pub intmd: HashMap<String, Tsr>,
}

impl<'a> UHessKSNIMatmul<'a> {
    /// Create a new UKS Hessian object.
    ///
    /// # Parameters
    ///
    /// - `mol` : molecule.
    /// - `xc_func_list` : list of `(scale, functional)` pairs.
    /// - `ni` : numerical-integration driver over the atom-grouped grid.
    /// - `grid_shift` : evaluate the Becke grid-shift terms.
    /// - `verbose` : print progress.
    pub fn new(
        mol: &CInt,
        xc_func_list: Vec<(f64, LibXCFunctional)>,
        ni: NIMatmul<'a>,
        grid_shift: bool,
        verbose: bool,
    ) -> Self {
        Self { mol: mol.clone(), xc_func_list, ni, grid_shift, verbose, intmd: HashMap::new() }
    }

    /// Perform the Hessian setup for UKS calculations.
    ///
    /// `de_xc_skeleton` and `vmat_deriv1_grid_a/b` are the main results.
    pub fn make_hessian_setup(&mut self, mo_coeff: &[TsrView; 2], mo_occ: &[TsrView; 2], atm_list: Option<&[usize]>) {
        let occidx_α = mo_occ[α].view().greater(0).into_vec();
        let occidx_β = mo_occ[β].view().greater(0).into_vec();
        let mocc_α = mo_coeff[α].bool_select(-1, &occidx_α);
        let mocc_β = mo_coeff[β].bool_select(-1, &occidx_β);
        let dm0α = &mocc_α % mocc_α.t();
        let dm0β = &mocc_β % mocc_β.t();

        let (result, _timing) = make_hessian_setup_becke_uks(
            &self.mol,
            &self.xc_func_list,
            &mut self.ni,
            dm0α.view(),
            dm0β.view(),
            self.grid_shift,
            atm_list,
            self.verbose,
        );

        for (key, val) in result.into_iter() {
            self.intmd.insert(key.to_string(), val);
        }
    }

    /// Check if the Hessian setup is done by verifying the presence of the
    /// "de_xc_skeleton" key in the intermediate results.
    pub fn is_hessian_setup_done(&self) -> bool {
        self.intmd.contains_key("de_xc_skeleton")
    }
}

impl<'a> AnalDrvBaseAPI for UHessKSNIMatmul<'a> {}

impl<'a> UHessElecInteractAPI for UHessKSNIMatmul<'a> {
    fn make_skeleton_hess(
        &mut self,
        mo_coeff: &[TsrView; 2],
        mo_occ: &[TsrView; 2],
        atm_list: Option<&[usize]>,
    ) -> Tsr {
        if !self.is_hessian_setup_done() {
            self.make_hessian_setup(mo_coeff, mo_occ, atm_list);
        }
        self.intmd["de_xc_skeleton"].to_owned()
    }

    fn get_deriv1_ao(
        &mut self,
        mo_coeff: &[TsrView; 2],
        mo_occ: &[TsrView; 2],
        atm_list: Option<&[usize]>,
    ) -> [Tsr; 2] {
        if !self.is_hessian_setup_done() {
            self.make_hessian_setup(mo_coeff, mo_occ, atm_list);
        }
        [self.intmd["vmat_deriv1_grid_a"].to_owned(), self.intmd["vmat_deriv1_grid_b"].to_owned()]
    }
}

/* #endregion */
