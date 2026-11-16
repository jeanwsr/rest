// ===========================================================================
// Adjoint ("Z-vector") evaluation of the CP-HF channel
// ===========================================================================
//
// After the `qx` channel has been folded, the whole gradient still needing the
// per-perturbation CP-HF response is
//
//     G_d = 2 Tr[u_d^T W] + sum_p be[p] eps1_d[p] + <B, dI_d> + bJ:dJ_d + bB.b_x_d
//
// with `u_d`, `eps1_d` the Roothaan response of perturbation `d`.  Writing the
// Roothaan formulas out,
//
//   u[p,q] = (f1_mo[p,q] - e_q s1_mo[p,q]) / (e_q - e_p)   (p != q),
//   u[p,p] = -s1_mo[p,p]/2,   eps1[p] = f1_mo[p,p] - e_p s1_mo[p,p],
//   f1_mo  = C^T (dh1 + Jd - Kd/2 + vind(dm1)) C,
//   dm1    = 2 sum_i (c1_i c_i^T + c_i c1_i^T),   c1 = C mo1,
//
// the response part is a **linear functional of `mo1` with a
// perturbation-independent covector**:
//
//   G_d^resp = <Lambda, f1_mo_d> + <Lambda_s, s1_mo_d>
//            = <Lam_ao, dh1_d> + <Lam_ao, Jd_d - Kd_d/2> + Tr[Vtil dm1_d]
//              + <Lam_ao, vind(dm1_d)> + <Lam_s,ao, dS_d>            (1)
//
// with (W the folded rotation factor of the qx channel)
//
//   Lambda[p,q]   = 2 W[p,q]/(e_q - e_p)  (p != q),  Lambda[p,p] = be[p]
//   Lambda_s[p,q] = -2 W[p,q] e_q/(e_q - e_p) (p != q),
//   Lambda_s[p,p] = -W[p,p] - be[p] e_p
//   Lam_ao = C Lambda C^T,  Lam_s,ao = C Lambda_s C^T
//   Vtil   = vind(Lam_ao) = J[Lam_ao] - fock_hyb/2 K[Lam_ao]     (self-adjoint)
//
// and then, since `mo1_d = S rhs_d` with `S` the CP-HF solver map,
//
//   Tr[Vtil dm1_d] = <L, mo1_d> = <Z, rhs_d>,   L[p,i] = 4 (C^T Vtil C_occ)[p,i]
//   Z = S^T L                                                     (2)
//
// so **one** adjoint solve replaces the `3 natm`-column Krylov solve.
//
// The two skeleton contractions in (1) are also linear in the derivative
// integrals, so they are folded into precomputed `[naux,nao,nao]` / `[naux,naux]`
// tensors:
//
//   <Lam_ao, Jd_d>  = sum_P a_P c_P + sum_P lam_P dc_P ,
//        a_P = sum_munu Lam_ao dI[P], lam_P = sum_munu Lam_ao I[P],
//        c = J^-1 rho, dc = J^-1 (drho - dJ c), rho = I^T dm, drho = dI^T dm
//   <Lam_ao, Kd_d>  = 4 <Theta, dI_d> - 2 <Psi, dJ_d> ,
//        Theta_P = Lam_ao G_P Cocc^T ,  G_P = (J^-1 Y)_P ,  Y[mu,P] = (mu i|P) C_i
//        Psi = J^-1 Phi ,  Phi[P,P'] = sum_i Y_i^T Lam_ao G_i
//
// Cost: all precomputations are `O(naux nao^3)`-at-most once per geometry; each
// perturbation then costs one streamed pass over the `naux` `[nao,nao]` `dI`
// slices plus `O(naux^2 + nmo nocc)` scalars, i.e. `O(naux nao^2)`.  Total
// `O(naux nao^3 + 3 natm naux nao^2)` instead of `O(3 natm (naux nao^2 nocc +
// naux nao^3 + ...))`.
//
// ---------------------------------------------------------------------------
// CP-KS (DFT / hybrid) references
// ---------------------------------------------------------------------------
//
// A DFT reference changes `f1_mo` and the static term, so (1) gains three
// channels; all of them are still folded once per geometry:
//
//   f1_mo = C^T (dh1 + Jd - hyb Kd/2 + v_xc^expl            <- skeleton
//                 + J(dm1) - hyb K(dm1)/2 + f_xc(dm1)) C    <- relaxed (via Z)
//   b_x[p] = -v_xc^expl[p,p] - f_xc_diag(dm1)[p]
//            - 2 sum_q u[q,p] v_xc_mo[q,p]                   <- `bb` covector
//
//  (i) the XC kernel in the adjoint density: `f_xc` is self-adjoint, so
//      `<Lambda, f_xc(dm1_d)> = Tr[f_xc(Lam_ao) dm1_d]` and it is simply added
//      to `Vtil` -- one grid sweep, no per-perturbation cost:
//        Vtil = J[Lam_ao] - hyb K[Lam_ao]/2 + f_xc(Lam_ao) .
// (ii) the explicit frozen-density v_xc skeleton `v_xc^expl` (= the grid
//      derivative at fixed density and orbitals, `hessian::rks::compute_vxc_h1ao`)
//      enters `f1_mo` exactly like `dh1`, so it needs *two* images: the Roothaan
//      covector `<Lam_ao, v_xc^expl>` (per perturbation, like `<Lam_ao, dh1>`)
//      and the `Z` right-hand side `-v_xc^expl[vir,occ]/D_ai`, whose fold is
//      `<W, v_xc^expl>` with `W = Zbra C_occ^T`.
// (iii) the static term `B_n = e_n + (1-alpha) Sigma_x,nn - v_xc,nn` is
//      differentiated through the diagonal covector `bb` (`b_x`).  Its
//      `-f_xc_diag(dm1)` part is `-Tr[f_xc(D_bb) dm1_d]` with
//      `D_bb = C diag(bb) C^T`, i.e. `-f_xc(D_bb)` added to `Vtil` as well, and
//      its orbital-rotation part is a covector on the **full** rotation `u`
//      (every MO column, not only the occupied ones the `Z` solve sees), so it
//      is folded into `Lambda`/`Lambda_s` before `Lam_ao` is formed:
//        C[p,q] = -2 bb[q] v_xc_mo[p,q] ,
//        Lambda[p,q]   += (C[p,q]-C[q,p]) / (2 (e_q-e_p)) ,
//        Lambda_s[p,q] += (e_p C[q,p]-e_q C[p,q]) / (2 (e_q-e_p)) ,
//        Lambda_s[p,p] += -C[p,p]/2           (from `u[p,p] = -s1[p,p]/2`).
//      Dropping this fold (or truncating `C` to its occupied columns, as the
//      first cut of this route did) leaves an O(1) error in the gradient.
//
// `Theta`/`Psi` are built for **unit** exchange, so the skeleton
// `-hyb Kd/2` enters as `-2 hyb Theta` (`di_cov`) and `hyb <Psi, dJ>` -- an HF
// reference has `hyb = 1` and cannot see the factor, a hybrid is wrong by
// `1/hyb` without it (`CphfAdjoint::k_scale`).
//
// The XC-kernel terms need the CP-KS grid: `prepare_fxc_cache` (the SCF grid,
// the same one the canonical route's relaxed Fock uses), one sweep for
// `f_xc(Lam_ao)` and one for `f_xc(D_bb)`.
//
// Symmetry-degenerate shells are handled by the "skip the shell-internal
// off-diagonal pairs" rule below: for an equal-weight shell average the
// covectors are isotropic on the shell (Schur's lemma), which reduces the shell
// block to `be Tr[A] - w sum_i s1[p_i,p_i]`, and the `b_x` rotation cocycle to
// its diagonal (also isotropic: `v_xc` is totally symmetric, so it is
// proportional to the identity inside a degenerate shell).
//
// Validation: `tests/gw_bse/test_adjoint_dft.rs` (water/formaldehyde/benzene,
// PBEh(45%), canonical vs adjoint) and the production bench
// `tests/gw_bse/gw_grad_bench_dft_adjoint.rs`.

/// Per-geometry adjoint data of the CP-HF channel; see the section header.
pub struct CphfAdjoint {
    /// `[naux, nao, nao]`: **symmetrised** coefficient of `dI`,
    /// `(B - 2 fock_hyb Theta + c_P Lam_ao) + transpose`.  The Coulomb
    /// `sum_P a_P c_P` term (`a_P = <Lam_ao, dI_P>`) is folded in, and the
    /// symmetrisation lets both `int3c2e_ip1` bra centres be contracted in a
    /// single pass (see [`GwGradEngine::adjoint_perturbation_gradient_local`]).
    pub di_cov: Vec<f64>,
    /// `[nao, nao]`: `dm + dm^T` (the SCF density is symmetric, so this is
    /// `2 dm`); used together with the factor `1/2` of the symmetrised slicing.
    pub dm_sym: Vec<f64>,
    /// `[naux]`: `J^-1 lambda`; replaces the per-perturbation metric solve by
    /// `sum_P lambda_P dc_P = sum_P lam_tilde_P (drho - dJ c)_P`.
    pub lam_tilde: Vec<f64>,
    /// `[nao, nao]`: `Lam_ao` (coefficient of `dh1`).
    pub lam_ao: Vec<f64>,
    /// `[nao, nao]`: `Lam_s,ao` (coefficient of `dS`, col-major like `dS`).
    pub lams_ao: Vec<f64>,
    /// `[naux]`: `sum_munu Lam_ao I[mu,nu,P]`.
    pub lam_aux: Vec<f64>,
    /// `[naux, naux]`: `Psi` (coefficient of `dJ`).
    pub psi: Vec<f64>,
    /// `[naux]`: `J^-1 rho`, `rho_P = sum_munu I[mu,nu,P] dm[mu,nu]`.
    pub c_aux: Vec<f64>,
    /// `[nao, nao]` SCF density (col-major).
    pub dm: Vec<f64>,
    /// `[3 natm]`: `<Z, rhs_d>` (only filled when `fold_cphf_rhs` is off, for
    /// cross-validation against the folded route).
    pub rhs_dot: Vec<f64>,
    /// whether `<Z, rhs_d>` was folded into the covectors (`false` = the
    /// reference route, which keeps `rhs_dot` and must not add the `W` terms).
    pub fold_rhs: bool,
    /// `[nao, nao]`: `W = Zbra C_occ^T`, the AO covector replacing `<Z, rhs_d>`
    /// (coefficient of `dh1`).
    pub w_dh1: Vec<f64>,
    /// `[nao, nao]`: coefficient of `dS`.
    pub w_ds: Vec<f64>,
    /// flattened `[nmo, nocc, 3, natm]` analdrv skeleton Fock derivative (only
    /// filled when `fold_cphf_rhs` is off; used by the verification tests).
    pub f1mo_ref: Vec<f64>,
    /// exchange scale of the skeleton Fock derivative (always 0.5 here)
    pub half_k: f64,
    /// hybrid (exact-exchange) fraction of the reference, `fock_hyb`; the
    /// skeleton exchange is `-0.5 * k_scale * Kd`, so the `Theta`/`Psi` tensors
    /// built for unit exchange enter the gradient with this weight (`1` for HF).
    pub k_scale: f64,
    /// wall-clock of the sub-phases (seconds):
    /// [0] covectors Lambda + Vtil = vind(Lam_ao), [1] analdrv object setup,
    /// [2] make_response_preparation, [3] compute_dimless_cpscf_rhs,
    /// [4] adjoint (Z) solve + <Z,rhs>, [5] skeleton foldings (Theta, Phi/Psi)
    pub timings: [f64; 6],
    /// verification fields (MO-basis covectors, relaxed-response operator, L and Z)
    pub lam_mo: Vec<f64>,
    pub lams_mo: Vec<f64>,
    pub vtil: Vec<f64>,
    pub l_mat: Vec<f64>,
    pub z_vec: Vec<f64>,
}

impl GwGradEngine<'_> {
    /// Whether the adjoint (Z-vector) CP-HF route applies to this reference.
    ///
    /// The route is the linear-response CP-HF solution of a **plain HF**
    /// reference; for a DFT reference it declines and the gradient falls back to
    /// the per-perturbation route. Callers check this *before* building the
    /// folded covector, so that the `naux * nmo^2` covector buffer can be
    /// released as soon as the fold has consumed it (see
    /// [`Self::analytic_gradient`]).
    ///
    /// This is the one predicate behind [`Self::build_cphf_adjoint`]'s early
    /// return: if that function ever declines for another reason, callers that
    /// rely on it must be revisited too.
    pub fn supports_cphf_adjoint(&self) -> bool {
        if !self.scf.mol.dfadef.is_dfa_scf() {
            return true;                       // plain HF (the validated route)
        }
        // DFT / global-hybrid (CP-KS): requires the kernel terms of
        // `build_cphf_adjoint`, which are opt-in; range-separated hybrids are
        // rejected by the energy path already.
        self.config.adjoint_cphf_dft && self.scf.mol.dfadef.omega().is_none()
    }

    /// Per-atom explicit grid derivative of `v_xc` (`d v_xc/dR` at frozen
    /// density and orbitals), `[natm][3*nao*nao]`, or `None` for an HF
    /// reference.  One grid sweep; shared by the canonical route's CP-KS
    /// right-hand side and by the adjoint route's `b_x` covector.
    pub fn vxc_grid_derivative_atoms(&self) -> Option<Vec<Vec<f64>>> {
        if !self.scf.mol.dfadef.is_dfa_scf() {
            return None;
        }
        let t0 = Instant::now();
        let v = crate::hessian::rks::compute_vxc_h1ao(self.scf);
        if grad_timing() {
            eprintln!(
                "[gw_grad timing]   cphf: vxc_deriv1 grid sweep {:8.2}s",
                t0.elapsed().as_secs_f64()
            );
        }
        Some(v)
    }

    /// Explicit part of the `b_x` covector of perturbation `(atm, comp)` in the
    /// MO basis, `b_x[p] = -<p| d v_xc^expl/dR |p>`.  The two U-dependent parts
    /// (orbital rotation `-2 sum_q U[q,p] v_xc[q,p]` and the kernel response of
    /// the first-order density) are folded into the adjoint right-hand side by
    /// [`Self::build_cphf_adjoint`], because the adjoint route never forms `U`.
    pub fn vxc_diag_bx(&self, atoms: &[Vec<f64>], atm: usize, comp: usize) -> Vec<f64> {
        let nao = self.nao;
        let nmo = self.nmo;
        let vao = vxc_block_ao(&atoms[atm], comp, nao);
        // MO transform and take the diagonal
        let t = gemm_nt(&self.c_mo.data, nao, nmo, &vao, nao, nao);
        let vmo = gemm_nn(&t, nmo, nao, &self.c_mo.data, nao, nmo);
        (0..nmo).map(|p| -vmo[p + p * nmo]).collect()
    }

    /// Build the adjoint data (see the section header).  Returns `None` when the
    /// reference has symmetry-degenerate shells (the route is not linear there)
    /// or when the reference is not a plain HF one.
    pub fn build_cphf_adjoint(
        &self,
        fold: &mut FoldedCovector,
        be_k: &[f64],
        bb_k: Option<&[f64]>,
    ) -> Option<CphfAdjoint> {
        use crate::analdrv::prelude::{
            AnalDrvConfig, HessNucAPI, RHessCoreAPI, RHessElecInteractAPI, RHessHcore,
            RHessOvlp, RHessSCF,
        };
        use crate::analdrv::response::rresp_interface::rscf_resp_interface;
        use crate::ri_jk::hess_r::RHessRIJK;
        use crate::ri_jk::util::{get_cint_aux, get_cint_mol};
        use crate::utilities::rstsr_util::RestTensorToRstsrViewAPI;
        use rstsr::prelude::*;

        // Symmetry-degenerate shells are supported: for an equal-weight shell
        // average the covectors are isotropic on the shell (Schur's lemma), so
        // the shell block of the Roothaan response reduces to its trace,
        //     sum_i be[p_i] eps1[p_i] + 2 sum_ij W[p_i,p_j] u[p_i,p_j]
        //         = be * Tr[A] - w * sum_i s1[p_i,p_i] ,   A_ij = f1_ij - e0 s1_ij,
        // which is linear in the perturbation.  That is exactly what the
        // "skip the shell-internal off-diagonal pairs" rule below computes.
        if !self.supports_cphf_adjoint() {
            return None;
        }

        let nao = self.nao;
        let nmo = self.nmo;
        let nocc = self.nocc;
        let naux = self.naux;
        let n2 = nao * nao;
        let e = &self.e;
        // DFT / hybrid references carry the XC-kernel terms (bits of
        // `REST_ADJOINT_DFT_MASK`); an HF reference is the validated route and
        // never enters them.
        let is_hf = !self.scf.mol.dfadef.is_dfa_scf();
        let dft_mask = dft_debug_mask();

        // Timer for the whole covector assembly, sections (a)/(a2)/(b).  It must
        // start *here*: the reported `timings[0]` used to be measured from a
        // point after this work, so "[gw_grad timing] adjoint: covectors" always
        // printed 0.00 s and hid the largest single piece of the adjoint.
        let t_covectors = Instant::now();

        // ---- (a) MO covectors Lambda, Lambda_s ---------------------------
        let mut lam = vec![0.0f64; nmo * nmo];
        let mut lams = vec![0.0f64; nmo * nmo];
        // NB: `f1_mo`/`s1_mo` are symmetric, so the covector is defined through
        // the full sum `sum_pq Lam[p,q] f1[p,q]`; each ordered pair (p,q) and
        // (q,p) of the Roothaan formulas contributes to the same matrix element.
        for p in 0..nmo {
            lam[p + p * nmo] = be_k[p];
            lams[p + p * nmo] = -fold.w_rot[p + p * nmo] - be_k[p] * e[p];
            for q in 0..nmo {
                // `p == q` and shell-internal pairs have `e_q == e_p`; the latter
                // are handled by the shell trace (see the guard comment above).
                if p == q || self.same_degenerate_shell(p, q) {
                    continue;
                }
                let den = e[q] - e[p];
                let wpq = fold.w_rot[p + q * nmo];
                let wqp = fold.w_rot[q + p * nmo];
                lam[p + q * nmo] = (wpq - wqp) / den;
                lams[p + q * nmo] = (e[p] * wqp - e[q] * wpq) / den;
            }
        }

        // ---- (a2) DFT / hybrid references: the orbital-rotation part of b_x --
        //
        // The static term `B_n = e_n + (1-alpha) Sigma_x,nn - v_xc,nn` is
        // differentiated through the diagonal covector `bb` (`b_x`, see
        // `canonical_finish`):
        //     b_x[p] = -vd1[p,p] - fxc_diag(dm1)[p] - 2 sum_q u[q,p] vxc_mo[q,p].
        // The last term is linear in the **full** orbital rotation `u`, not only
        // in its occupied columns `mo1`, so it cannot ride the `Z` solve (which
        // is a covector on `mo1` alone).  With the u-covector
        //     C[p,q] = -2 bb[q] vxc_mo[p,q] ,   sum_{p,q} u[p,q] C[p,q]
        //            = -2 sum_{p,q} bb[p] u[q,p] vxc_mo[q,p] ,
        // the Roothaan identity of the section header turns *any* u-covector
        // into an equivalent `(Lambda, Lambda_s)` pair: the off-diagonal pairs
        // through the orbital denominators and the diagonal through
        // `u[p,p] = -s1[p,p]/2`,
        //     Lambda_C[p,q] = (C[p,q]-C[q,p]) / (2 (e_q - e_p))
        //                   = (bb[p]-bb[q]) vxc_mo[p,q] / (e_q - e_p) ,
        //     Lam_s,C[p,q]  = (e_p C[q,p] - e_q C[p,q]) / (2 (e_q - e_p))
        //                   = (e_q bb[q] - e_p bb[p]) vxc_mo[p,q] / (e_q - e_p) ,
        //     Lam_s,C[p,p]  = -C[p,p]/2 = bb[p] vxc_mo[p,p] ,
        // so folding it in here makes every downstream use of `Lambda_ao` (the
        // folded `dI`/`dJ` covectors, `Vtil` and hence `L`/`Z`, and the explicit
        // `dh1`/`dS` contractions) carry the `b_x` rotation channel exactly, at
        // no per-perturbation cost.  (The earlier route put only the occupied
        // columns of `C` into `l_mat`; the virtual columns -- which are the ones
        // `bb` has support on whenever a virtual orbital is being
        // differentiated -- were silently dropped.)
        if !is_hf && (dft_debug_mask() & 2 != 0) {
            if let Some(bb) = bb_k {
                for p in 0..nmo {
                    lams[p + p * nmo] += bb[p] * self.vxc_mo[p + p * nmo];
                    for q in 0..nmo {
                        if p == q || self.same_degenerate_shell(p, q) {
                            continue;
                        }
                        let den = e[q] - e[p];
                        let vpq = self.vxc_mo[p + q * nmo];
                        lam[p + q * nmo] += (bb[p] - bb[q]) * vpq / den;
                        lams[p + q * nmo] +=
                            (e[q] * bb[q] - e[p] * bb[p]) * vpq / den;
                    }
                }
            }
        }

        // Lam_ao = C Lam C^T, Lam_s,ao = C Lam_s C^T
        let to_ao = |m: &[f64]| -> Vec<f64> {
            let mm = to_mat(m, nmo, nmo);
            let mut tmp = MatrixFull::new([nao, nmo], 0.0);
            let mut out = MatrixFull::new([nao, nao], 0.0);
            _dgemm_full(&self.c_mo, 'N', &mm, 'N', &mut tmp, 1.0, 0.0);
            _dgemm_full(&tmp, 'N', &self.c_mo, 'T', &mut out, 1.0, 0.0);
            out.data
        };
        let lam_ao = to_ao(&lam);
        let lams_ao = to_ao(&lams);

        // ---- (b) Vtil = vind(Lam_ao) and L = 4 C^T Vtil C_occ ------------
        let lam_ao_m = to_mat(&lam_ao, nao, nao);
        let j_up = crate::dft::response::compute_j_upper(self.scf, &vec![lam_ao_m.clone()]);
        let k_up = crate::dft::response::compute_k_upper(self.scf, &vec![lam_ao_m]);
        let j_full = j_up.to_matrixfull().expect("adjoint: J upper").data;
        let k_full = k_up.to_matrixfull().expect("adjoint: K upper").data;
        let mut vtil = vec![0.0f64; n2];
        for k in 0..n2 {
            vtil[k] = j_full[k] - 0.5 * self.fock_hyb * k_full[k];
        }
        // ---- DFT / hybrid references: the XC-kernel part ------------------
        // `vtil` is the functional derivative of the energy with respect to the
        // density, so a CP-KS reference adds the kernel's action on the adjoint
        // density, `vtil += f_xc(Lambda_ao)` -- the exact analogue of the
        // `f_xc(D^U)` term that the canonical route carries inside its relaxed
        // first-order Fock (`canonical_finish`).  One grid kernel sweep.
        // One kernel cache for both DFT terms below (the canonical route builds
        // its own; here two sweeps share one cache).
        let fxc_cache = if is_hf {
            None
        } else {
            Some(crate::hessian::rks::prepare_fxc_cache(self.scf).expect(
                "gw_grad: the adjoint CP-KS route needs the XC-kernel cache                  (an scf without grids cannot provide it)",
            ))
        };
        if !is_hf && dft_mask & 1 != 0 {
            let cache = fxc_cache.as_ref().expect("cache built above");
            let fxc = crate::dft::response::compute_fxc_response_ao_cached(
                cache,
                &to_mat(&lam_ao, nao, nao),
            );
            for (v, f) in vtil.iter_mut().zip(fxc.data.iter()) {
                *v += *f;
            }
        }
        let cocc: Vec<f64> = (0..nocc)
            .flat_map(|i| (0..nao).map(move |mu| self.c_mo[[mu, i]]))
            .collect();
        let mut l_mat = {
            let vm = to_mat(&vtil, nao, nao);
            let cm = to_mat(&cocc, nao, nocc);
            let mut tmp = MatrixFull::new([nao, nocc], 0.0);
            let mut out = MatrixFull::new([nmo, nocc], 0.0);
            _dgemm_full(&vm, 'N', &cm, 'N', &mut tmp, 1.0, 0.0);
            _dgemm_full(&self.c_mo, 'T', &tmp, 'N', &mut out, 4.0, 0.0);
            out.data
        };
        // ---- DFT / hybrid references: the density-response part of `b_x` --
        // After the `(a2)` fold above, `Lambda` carries the whole orbital-rotation
        // channel of `b_x`; what is left of the `bb` covector is the kernel
        // response of the first-order density,
        //   -sum_p bb[p] [C^T f_xc(D^U) C]_pp = -Tr[D^U f_xc(D_bb)]
        //                                      = -4 sum_{p,i} (C^T X C)[p,i] U[p,i],
        // with `D_bb = C diag(bb) C^T` and `X = f_xc(D_bb)` (the kernel is
        // self-adjoint, so this is the same sweep as `f_xc` of a density), i.e.
        // an extra `Vtil`-like covector `-f_xc(D_bb)` on the `Z` right-hand side.
        // The explicit `vd1` part stays per perturbation; the caller passes it
        // as the `b_x` argument of the perturbation contractions.
        if !is_hf && dft_mask & 4 != 0 {
            if let Some(bb) = bb_k {
                let mut dbb = vec![0.0f64; n2];
                for mu in 0..nao {
                    for nu in 0..nao {
                        let mut acc = 0.0;
                        for p in 0..nmo {
                            acc += self.c_mo[[mu, p]] * bb[p] * self.c_mo[[nu, p]];
                        }
                        dbb[mu + nu * nao] = acc;
                    }
                }
                let cache = fxc_cache.as_ref().expect("cache built above");
                let x = crate::dft::response::compute_fxc_response_ao_cached(
                    cache,
                    &to_mat(&dbb, nao, nao),
                );
                let t = gemm_nt(&self.c_mo.data, nao, nmo, &x.data, nao, nao);
                let xmo = gemm_nn(&t, nmo, nao, &self.c_mo.data, nao, nmo);
                for i in 0..nocc {
                    for p in 0..nmo {
                        l_mat[p + i * nmo] -= 4.0 * xmo[p + i * nmo];
                    }
                }
            }
        }

        // ---- (c) one adjoint CP-HF solve -> Z, and <Z, rhs_d> -------------
        let t_all = Instant::now();
        let mut timings = [0.0f64; 6];
        timings[0] = 0.0; // (a)+(b) are timed by the caller-visible total below
        let t_setup = Instant::now();
        let _ = t_all;
        let device = crate::utilities::rstsr_util::device();
        let mol = get_cint_mol(&self.scf.mol);
        let aux = get_cint_aux(&self.scf.mol);
        let mo_coeff = rt::asarray((&self.c_mo.data, [nao, nmo], &device)).into_contig(ColMajor);
        let mo_occ =
            rt::asarray((&self.scf.occupation[0], [nmo], &device)).into_contig(ColMajor);
        let mo_energy = rt::asarray((&e[..], [nmo], &device)).into_contig(ColMajor);
        let mut ovlp_obj = RHessOvlp::new(&mol, &device);
        let mut hcore_obj = RHessHcore::new(&mol, &device);
        let rimatr = &self
            .scf
            .rimatr
            .as_ref()
            .expect("gw_grad: adjoint CP-HF requires rimatr (use_auxbas + use_ri_symm)")
            .0;
        let cderi = rimatr.to_rstsr_view(&device).into_cow();
        let j2c_decomp =
            crate::ri_jk::get_j2c_decomp(&aux, &device, self.scf.mol.ctrl.j2c_decomp.clone());
        let mut rijk_obj = RHessRIJK::new_with_cderi(&mol, &aux, 1.0, 1.0, cderi, j2c_decomp);
        let mut config = AnalDrvConfig::default();
        // The adjoint route's accuracy is set by the Z solve, so the Krylov
        // tolerances are part of the gradient configuration (and can be
        // tightened; see `GwGradConfig::cpscf_tol`).
        config.resp.tol = self.config.cpscf_tol;
        config.resp.tol_inflation = self.config.cpscf_tol_inflation;
        // Everything from here to the end of the Z-vector correction lives in its own scope: the
        // response objects (and with them the RI-JK response-kernel cache, `nao^2 * naux` doubles —
        // 1030 MB at naphthalene/def2-TZVP) are released before the skeleton contractions below,
        // which never touch them. `z_vec`/`rhs_dot`/the timings are the only outputs needed later.
        let mut z_vec: Vec<f64>;
        let rhs_dot: Vec<f64>;
        let nvir: usize;
        let e_ai_shift: Vec<f64>;
        let f1mo_ref: Option<Vec<f64>>;
        assert!(
            is_hf || self.config.fold_cphf_rhs,
            "gw_grad: the adjoint CP-KS route needs fold_cphf_rhs = true"
        );
        let fold_rhs = self.config.fold_cphf_rhs;
        {
        let mut resp = rscf_resp_interface(self.scf, &config);
        timings[1] = t_setup.elapsed().as_secs_f64();

        timings[0] = t_covectors.elapsed().as_secs_f64();
        let t_rhs = Instant::now();
        let rhs_full_opt = {
            let nuc_list: Vec<&mut dyn HessNucAPI> = Vec::new();
            let core_list: Vec<&mut dyn RHessCoreAPI> = vec![&mut hcore_obj];
            let el_list: Vec<&mut dyn RHessElecInteractAPI> = vec![&mut rijk_obj];
            let mut hess = RHessSCF::new(
                mo_coeff, mo_occ, mo_energy, &mut ovlp_obj, nuc_list, core_list, el_list,
                &mut resp, &config.nucgrad,
            );
            hess.make_response_preparation();
            timings[2] = t_rhs.elapsed().as_secs_f64();
            let t2 = Instant::now();
            if fold_rhs {
                // only `f1mo`/`s1mo` are unavailable here; the skeleton tensor
                // itself is never needed (the `Z` contraction is folded below)
                (None, None)
            } else {
                let rhs_map = hess.compute_dimless_cpscf_rhs();
                timings[3] = t2.elapsed().as_secs_f64();
                let f1 = rhs_map["f1mo"].to_owned();
                let fr = f1.raw()[f1.offset()..f1.offset() + nmo * nocc * 3 * self.natm].to_vec();
                (Some(rhs_map["rhs"].to_owned()), Some(fr))
            }
        };
        let (rhs_full, f1mo_ref_inner) = rhs_full_opt;
        f1mo_ref = f1mo_ref_inner;

        let t_solve = Instant::now();
        // dimensionless rhs for the adjoint: `rhs_vir = L / (e_a - e_i)`, occ = 0
        {
            let st = resp.cpscf_state();
            let nv = st.nmo - st.nocc;
            let cont = st.e_ai_shift.to_owned().into_contig(ColMajor);
            let s = cont.raw()[cont.offset()..cont.offset() + nv * st.nocc].to_vec();
            nvir = nv;
            e_ai_shift = s;
        }
        let mut rhs_z = rt::zeros(([nmo, nocc].f(), &device));
        {
            let mut buf = rhs_z.i_mut((rt::slice!(nocc, nmo), ..));
            let mut src = vec![0.0f64; nvir * nocc];
            for i in 0..nocc {
                for a in 0..nvir {
                    src[a + i * nvir] = l_mat[(nocc + a) + i * nmo] / e_ai_shift[a + i * nvir];
                }
            }
            let t = rt::asarray((&src, [nvir, nocc], &device)).into_contig(ColMajor);
            buf.assign(&t);
        }
        let t_asm = t_solve.elapsed();
        let t_sv = Instant::now();
        let z_raw = resp.solve_dimless_cpscf(rhs_z.view()); // [nmo, nocc]
        let t_sv = t_sv.elapsed();
        let z_cont = z_raw.into_contig(ColMajor);
        z_vec = vec![0.0f64; nmo * nocc];
        z_vec.copy_from_slice(&z_cont.raw()[z_cont.offset()..z_cont.offset() + nmo * nocc]);
        let t_occ0 = Instant::now();
        let z_occ_corr = {
            let mut x = rt::zeros(([nmo, nocc].f(), &device));
            let mut zv = vec![0.0f64; nvir * nocc];
            for i in 0..nocc {
                for a in 0..nvir {
                    zv[a + i * nvir] = z_vec[(nocc + a) + i * nmo];
                }
            }
            let t = rt::asarray((&zv, [nvir, nocc], &device)).into_contig(ColMajor);
            x.i_mut((rt::slice!(nocc, nmo), ..)).assign(&t);
            let r = resp.response_mo(x.view()).into_contig(ColMajor);
            let rr = r.raw();
            let ro = r.offset();
            // occ block of an [nmo, nocc] col-major tensor: rows 0..nocc
            let mut out = vec![0.0f64; nocc * nocc];
            for i in 0..nocc {
                for p in 0..nocc {
                    out[p + i * nocc] = rr[ro + p + i * nmo];
                }
            }
            out
        };
        for i in 0..nocc {
            for p in 0..nocc {
                z_vec[p + i * nmo] = l_mat[p + i * nmo] - z_occ_corr[p + i * nocc];
            }
            // Z_vir = D (D + A)^-1 L_vir = D * x_vir (D symmetric)
            for a in 0..nvir {
                z_vec[(nocc + a) + i * nmo] *= e_ai_shift[a + i * nvir];
            }
        }

        // the CP-KS response object is done with both its Krylov solves; release the whole-grid
        // AO tables it retained (a pure cache, recomputed on demand) so that they do not sit
        // under the remaining gradient phases
        if !is_hf {
        }
        mem_probe("  adjoint: after Z solve");
        if std::env::var("REST_GWGRAD_ZDBG").is_ok() {
            eprintln!(
                "[zdbg] rhs-assembly {:.3}s solve {:.3}s occ-corr {:.3}s",
                t_asm.as_secs_f64(),
                t_sv.as_secs_f64(),
                t_occ0.elapsed().as_secs_f64()
            );
        }
        // <Z, rhs_d> for every perturbation (skipped on the folded route)
        rhs_dot = match &rhs_full {
            Some(rhs_full) => {
                let zt = rt::asarray((&z_vec, [nmo, nocc], &device)).into_contig(ColMajor);
                let prod = rhs_full * &zt.i((.., .., None, None));
                let s = prod.sum_axes([0, 1]); // [3, natm]
                let sr = s.raw();
                let so = s.offset();
                sr[so..so + 3 * self.natm].to_vec()
            }
            None => vec![0.0f64; 3 * self.natm],
        };

        timings[4] = t_solve.elapsed().as_secs_f64();
        }
        let t_skel = Instant::now();
        // ---- (c2) fold `sum_d <Z, rhs_d>` into AO covectors -------------
        //
        // `rhs_d` is built from the skeleton Fock/overlap derivatives,
        //   rhs[a,i] = -(f1mo[a,i] - s1mo[a,i] e_i)/D_ai ,  rhs[j,i] = -s1mo[j,i]/2,
        // so with `Zf[p,i] = -Z[p,i]/D_pi` (virtual p) and
        // `Zs[a,i] = Z[a,i] e_i/D_ai`, `Zs[j,i] = -Z[j,i]/2` (occupied j),
        //   sum_d <Z, rhs_d> = sum_{mu i} Zbra[mu,i] f1bra[mu,i] + <Zsbra, s1mo>
        // with `Zbra = C Zf`, `Zsbra = C Zs`.  Writing `f1bra = dh1·C_occ +
        // (Jd - Kd/2)·C_occ` and `s1mo = C^T dS C_occ` gives
        //   sum_d <Z, rhs_d> = <W, dh1_d + Jd_d - Kd_d/2> + <W_s, dS_d> ,
        //   W = Zbra C_occ^T ,  W_s = Zsbra C_occ^T ,
        // i.e. the same `dh1`/`dJ`/`dK`/`dS` contractions already built for
        // `Lambda_ao`, so `W` is simply added into those folded covectors.
        let (w_dh1, w_ds) = {
            let mut zf = vec![0.0f64; nmo * nocc];
            let mut zs = vec![0.0f64; nmo * nocc];
            for i in 0..nocc {
                for a in 0..nvir {
                    let p = nocc + a;
                    let d = e_ai_shift[a + i * nvir];
                    let z = z_vec[p + i * nmo];
                    zf[p + i * nmo] = -z / d;
                    zs[p + i * nmo] = z * e[i] / d;
                }
                for j in 0..nocc {
                    zs[j + i * nmo] = -0.5 * z_vec[j + i * nmo];
                }
            }
            let cm = to_mat(&cocc, nao, nocc);
            let to_ao = |m: &[f64]| -> Vec<f64> {
                let mm = to_mat(m, nmo, nocc);
                let mut tmp = MatrixFull::new([nao, nocc], 0.0);
                let mut out = MatrixFull::new([nao, nao], 0.0);
                _dgemm_full(&self.c_mo, 'N', &mm, 'N', &mut tmp, 1.0, 0.0);
                _dgemm_full(&tmp, 'N', &cm, 'T', &mut out, 1.0, 0.0);
                out.data
            };
            (to_ao(&zf), to_ao(&zs))
        };

        // ---- (d) folded skeleton tensors --------------------------------
        // Each contraction over the raw three-centre integrals is accumulated **one auxiliary-atom
        // block at a time**, so the full `nao^2 * naux` array (874 MB at naphthalene/def2-TZVP) is
        // never resident — this phase used to be the gradient's peak. The integral work is the same
        // shell-triplet sum as one `int3c2e` call, only split into `natm` calls.
        //
        //   Lam_aux = I^T Lam_ao   [naux]      rho = I^T dm   [naux]
        //   Y^T [naux, nocc*nao]   (G^T = J^-1 Y^T)
        let dm = self.scf.density_matrix[0].data.clone();
        let mut lam_aux = vec![0.0f64; naux];
        let mut rho = vec![0.0f64; naux];
        // `Lam_aux,W = I^T W` (the `Z` covector's partner) rides the same pass: one sweep over the
        // integral blocks serves all four contractions, so the shell triplets are evaluated once.
        let mut lam_aux_w = vec![0.0f64; naux];
        let mut yt = vec![0.0f64; naux * nocc * nao];
        {
            let cm = to_mat(&cocc, nao, nocc);
            let mut ip = MatrixFull::new([nao, nao], 0.0);
            let mut block = MatrixFull::new([nao, nocc], 0.0);
            self.raw.for_each_i3_atom_block(|p0, nb, blk| {
                lam_aux[p0..p0 + nb].copy_from_slice(&gemm_nt(blk, n2, nb, &lam_ao, n2, 1));
                rho[p0..p0 + nb].copy_from_slice(&gemm_nt(blk, n2, nb, &dm, n2, 1));
                lam_aux_w[p0..p0 + nb].copy_from_slice(&gemm_nt(blk, n2, nb, &w_dh1, n2, 1));
                for pl in 0..nb {
                    ip.data.copy_from_slice(&blk[pl * n2..(pl + 1) * n2]);
                    _dgemm_full(&ip, 'N', &cm, 'N', &mut block, 1.0, 0.0);
                    for col in 0..nao * nocc {
                        yt[(p0 + pl) + col * naux] = block.data[col];
                    }
                }
            });
        }
        let c_aux = self.solve_metric(&rho);
        let mut gt = yt.clone();
        self.solve_metric_batch(&mut gt, nao * nocc);

        // Folded exchange adjoint of an arbitrary AO covector `x_ao`:
        //   <x_ao, Kd> = 4 <Theta_x, dI> - 2 <Psi_x, dJ> ,
        //   Theta_x[P] = x_ao G_P Cocc^T ,  Phi_x[P,P'] = sum_i Y_i^T x_ao G_i ,
        //   Psi_x = J^-1 Phi_x .
        let cm = to_mat(&cocc, nao, nocc);
        // `x_theta` and `x_phi` are separate because `<X,Kd> = 2<Theta_{X+X^T},dI>
        // - 2<Psi_X,dJ>`: the `dY G^T` term symmetrises `X` (the partner is its
        // transpose) while the `Y G2^T` term does not.
        // The `dI` covector is `B - 2*Theta + c_P Lam_ao - 2*Theta_W + c_P W`, i.e. a sum of
        // two `[naux, nao, nao]` tensors plus rank-one-in-`P` terms. `Theta` and `B` are both
        // `[naux, nao, nao]` and neither is read afterwards, so the exchange fold accumulates
        // **into `B`'s buffer**: `b_ao` is taken out of `fold` and becomes `di_cov`. That keeps
        // a second `naux * nao^2` tensor (874 MB at naphthalene/def2-TZVP) out of the resident
        // set of the whole adjoint build and of the per-direction loop that follows.
        let mut di_cov = std::mem::take(&mut fold.b_ao);
        let wfac = if fold_rhs { 1.0 } else { 0.0 };
        // The exchange skeleton of the reference is `-0.5 * fock_hyb * Kd`
        // (`fock_hyb` = 1 for HF), so the adjoint tensors `Theta`/`Psi`, which
        // are built for the *unit* exchange (`4<Theta,dI> - 2<Psi,dJ> =
        // <x_ao,Kd>`, see the identity check in `test_gw_folded_qx`), enter the
        // gradient with this factor.
        let kscale = self.fock_hyb;
        let mut fold_exchange = |x_theta: &[f64], x_phi: &[f64]| -> Vec<f64> {
            let xm = to_mat(x_theta, nao, nao);
            let mut gp = MatrixFull::new([nao, nocc], 0.0);
            let mut xb = MatrixFull::new([nao, nocc], 0.0);
            let mut th = MatrixFull::new([nao, nao], 0.0);
            for p in 0..naux {
                for i in 0..nocc {
                    for nu in 0..nao {
                        gp.data[nu + i * nao] = gt[p + (i * nao + nu) * naux];
                    }
                }
                _dgemm_full(&xm, 'N', &gp, 'N', &mut xb, 1.0, 0.0);
                _dgemm_full(&xb, 'N', &cm, 'T', &mut th, 1.0, 0.0);
                // in-place: B_P -= 2 Theta_P, then add c_P Lam_ao (+ c_P W); the accumulated
                // value is symmetric in (mu, nu) (`a + at` and `at + a` are the same
                // floating-point number), so each unordered index pair is written once.
                let cp = c_aux[p];
                let off = p * n2;
                for mu in 0..nao {
                    for nu in 0..=mu {
                        let k = mu + nu * nao;
                        let kt = nu + mu * nao;
                        let a = di_cov[off + k] - 2.0 * kscale * th.data[k] + cp * lam_ao[k]
                            + wfac * cp * w_dh1[k];
                        let at = di_cov[off + kt] - 2.0 * kscale * th.data[kt] + cp * lam_ao[kt]
                            + wfac * cp * w_dh1[kt];
                        let v = a + at;
                        di_cov[off + k] = v;
                        di_cov[off + kt] = v;
                    }
                }
            }
            let xm = to_mat(x_phi, nao, nao);
            let mut acc = MatrixFull::new([naux, naux], 0.0);
            let mut yi = MatrixFull::new([nao, naux], 0.0);
            let mut gi = MatrixFull::new([nao, naux], 0.0);
            let mut xi = MatrixFull::new([nao, naux], 0.0);
            for i in 0..nocc {
                for p in 0..naux {
                    for nu in 0..nao {
                        yi.data[nu + p * nao] = yt[p + (i * nao + nu) * naux];
                        gi.data[nu + p * nao] = gt[p + (i * nao + nu) * naux];
                    }
                }
                _dgemm_full(&xm, 'N', &gi, 'N', &mut xi, 1.0, 0.0);
                // accumulate over occupied orbitals (beta = 1)
                _dgemm_full(&yi, 'T', &xi, 'N', &mut acc, 1.0, 1.0);
            }
            let mut phi_flat = acc.data;
            self.solve_metric_batch(&mut phi_flat, naux);
            phi_flat
        };
        // The `Z` covector `W` rides the same contractions, and both builders
        // are linear in their covector, so a single call each suffices:
        //   -2 Theta_{Lambda + (W+W^T)/2} = -2 Theta_Lambda - Theta_{W+W^T}
        //     (the `dY G^T` term symmetrises its covector, hence `(W+W^T)/2`),
        //   Psi_{Lambda + W} = Psi_Lambda + Psi_W .
        let mut x_theta = lam_ao.clone();
        let mut x_phi = lam_ao.clone();
        if fold_rhs {
            for mu in 0..nao {
                for nu in 0..nao {
                    let k = mu + nu * nao;
                    x_theta[k] += 0.5 * (w_dh1[k] + w_dh1[nu + mu * nao]);
                    x_phi[k] += w_dh1[k];
                }
            }
        }
        let psi = fold_exchange(&x_theta, &x_phi);

        timings[5] = t_skel.elapsed().as_secs_f64();
        if grad_timing() {
            eprintln!(
                "[gw_grad timing]   adjoint: covectors {:.2}s  setup {:.2}s  prep {:.2}s  rhs {:.2}s  Z solve {:.2}s  foldings {:.2}s",
                timings[0], timings[1], timings[2], timings[3], timings[4], timings[5]
            );
        }
        // dI covector = B - 2*fock_hyb*Theta (the skeleton uses the plain 1/2)
        // The Coulomb skeleton is <Lam_ao, Jd> = sum_P a_P c_P + sum_P lam_P dc_P
        // with a_P = <Lam_ao, dI_P>, c = J^-1 rho and dc = J^-1 (drho - dJ c).
        // J^-1 is symmetric, so sum_P lam_P dc_P = sum_P lam_tilde_P (drho - dJ c)_P
        // with lam_tilde = J^-1 lam; this removes the per-perturbation metric
        // solve and lets `c_P Lam_ao` be folded into the dI covector.
        let half_k = 0.5f64;
        // `sum_d <Z,rhs_d> = <W, dh1 + Jd - Kd/2> + <W_s, dS>` rides the same
        // folded contractions as `<Lambda_ao, ...>`, so the Coulomb covector
        // `lambda_W = I^T W`, the exchange tensors `Theta_W`/`Psi_W` and the
        // per-direction `dh1`/`dS` partners are simply added in.
        let mut lam_aux_total = lam_aux.clone();
        if fold_rhs {
            for (a, b) in lam_aux_total.iter_mut().zip(lam_aux_w.iter()) {
                *a += b;
            }
        }
        let lam_tilde = self.solve_metric(&lam_aux_total);
        // `dI^(a)` (first bra centre) and `dI^(b)` (second bra centre) are
        // transposes of each other and `dI^(c)` (auxiliary centre) is symmetric,
        // so the full `dI` slice is symmetric and contracting with the
        // symmetrised covector counts both bra centres at once:
        //   sum_munu X dI = sum X_sym dI^(a) + (1/2) sum X_sym dI^(c) .
        let mut dm_sym = vec![0.0f64; n2];
        for mu in 0..nao {
            for nu in 0..nao {
                dm_sym[mu + nu * nao] = dm[mu + nu * nao] + dm[nu + mu * nao];
            }
        }

        mem_probe("  adjoint: after folding (di_cov/lam/psi)");
        Some(CphfAdjoint {
            di_cov,
            lam_ao,
            lams_ao,
            lam_aux,
            psi,
            c_aux,
            dm,
            rhs_dot,
            fold_rhs,
            w_dh1,
            w_ds,
            f1mo_ref: f1mo_ref.unwrap_or_default(),
            half_k,
            k_scale: kscale,
            timings,
            lam_tilde,
            dm_sym,
            // `Theta` was folded into `di_cov` in place; it exists only when a verification run
            // asked for the individual pieces (`keep_adjoint_pieces`), because it duplicates an
            // `naux * nao^2` tensor that the gradient itself never reads.
            lam_mo: lam,
            lams_mo: lams,
            vtil,
            l_mat,
            z_vec,
        })
    }

    /// One perturbation's gradient from the adjoint data; same value as
    /// `folded_perturbation_gradient` but with the CP-HF response contracted
    /// through `Z` (no per-perturbation response).
    ///
    /// This variant materialises a dense `[nao,nao]` `dI` slice for every
    /// auxiliary function; see [`GwGradEngine::adjoint_perturbation_gradient_local`]
    /// for the locality-aware (asymptotically cheaper) version.
    #[allow(clippy::too_many_arguments)]
    pub fn adjoint_perturbation_gradient(
        &self,
        adj: &CphfAdjoint,
        blocks: &AtomDerivBlocks,
        comp: usize,
        h1r: &[f64],
        s1_comp: &[f64],
        dj: &[f64],
        bj_k: &[f64],
        bb_k: &[f64],
        b_x: &[f64],
        vxc_atom: Option<&Vec<f64>>,
        dir: usize,
    ) -> f64 {
        let n2 = self.nao * self.nao;
        let naux = self.naux;
        let mut ip = vec![0.0f64; n2];
        let mut g = 0.0f64;
        let mut g_dm = 0.0f64;
        for p in 0..naux {
            self.raw.d_i_block(blocks, comp, p, &mut ip);
            let off = p * n2;
            // the dense `dI` slice is symmetric, so the symmetrised covectors
            // count it with a factor 1/2
            g += 0.5 * flat_dot(&adj.di_cov[off..off + n2], &ip);
            g_dm += 0.5 * adj.lam_tilde[p] * flat_dot(&adj.dm_sym, &ip);
        }
        let vd1 = vxc_atom.map(|v| vxc_block_ao(v, comp, self.nao));
        self.finish_adjoint_gradient(
            adj, comp, h1r, s1_comp, dj, bj_k, bb_k, b_x, vd1.as_deref(), dir, g + g_dm,
        )
    }

    /// Locality-aware variant of [`GwGradEngine::adjoint_perturbation_gradient`].
    ///
    /// `dI[mu,nu,P]` is nonzero only where a centre sits on the perturbed atom:
    /// `int3c2e_ip1` gives the atom's AO rows (`mu` on the atom, all `nu`) and
    /// columns (`nu` on the atom, all `mu`), while `int3c2e_ip2` is dense in
    /// `[mu,nu]` but only for the atom's own auxiliary functions `P`.
    /// Contracting the covectors directly against those sliced blocks costs
    /// `O(naux nrow_A nao + naux_A nao^2)` per perturbation instead of the
    /// `O(naux nao^2)` of the dense route; summed over the `3 natm`
    /// perturbations the whole gradient drops from `O(3 natm naux nao^2)` to
    /// `O(naux nao^2)`.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn adjoint_perturbation_gradient_local(
        &self,
        adj: &CphfAdjoint,
        atm: usize,
        h1r: &[f64],
        s1_all: &[Vec<f64>],
        djs: &[Vec<f64>],
        bj_k: &[f64],
        bb_k: &[f64],
        b_x3: &[Vec<f64>],   // one v_xc covector per Cartesian component
        vxc_atom: Option<&Vec<f64>>,
        dir0: usize,
    ) -> [f64; 3] {
        let nao = self.nao;
        let n2 = nao * nao;
        let row0 = self.raw.ao_range[atm][0];
        let nrow = self.raw.ao_range[atm][1] - row0;

        // all three Cartesian components are contracted from the *same* streamed integral blocks
        let mut g_di = [0.0f64; 3];
        let mut g_dm = [0.0f64; 3];
        //
        // Parallelism.  This contraction is a pure reduction over the auxiliary index, and the
        // per-atom derivative tensors are not kept resident, so it is embarrassingly parallel:
        // every auxiliary-function batch of `aux_perturbation_batches` produces its own partial
        // sums, which are merged in index order afterwards.  The batch layout is a function of
        // the auxiliary-shell index alone — never of `rayon::current_num_threads()` — so a
        // gradient stays bit-for-bit reproducible under `num_threads`, the same discipline as
        // `utilities::par_sum`.  Before this change the whole `3 natm` loop ran on one core,
        // re-evaluating the atom's full `ip1`/`ip2` tensors for every auxiliary-shell block;
        // each batch now evaluates its own blocks in parallel and keeps them only for its own
        // contraction.
        let dm = &adj.dm_sym;
        let raw = &self.raw;
        let (a0, a1) = (raw.aux_range[atm][0], raw.aux_range[atm][1]);
        let mol = &raw.mol;
        let auxmol = &raw.auxmol;
        let ao_shl = raw.ao_shl_range[atm];
        let aux_rng = raw.aux_range[atm];
        // One parallel pass over the deterministic batches; each batch evaluates its own
        // `int3c2e_ip1`/`ip2` blocks (both, once) and reduces them to `[g_di | g_dm]`.
        let batches = raw.aux_perturbation_batches();
        // Bounded fan-out: each task materialises one auxiliary batch of the raw
        // derivative integrals (`nrow*nrow_A*nao*nb*3` doubles), so an unbounded
        // fan-out would add `workers * block` to the resident set of the whole
        // gradient.  The partial sums are merged in index order either way.
        let mut bi = 0usize;
        while bi < batches.len() {
            let hi = (bi + par_cap()).min(batches.len());
            let parts: Vec<[f64; 6]> = batches[bi..hi]
            .par_iter()
            .map(|&[f0, f1]| {
                let (d1c, d2c) = crate::ri_gw::gw_grad::RawRiTensors::perturbation_batch_blocks_pure(
                    mol, auxmol, &ao_shl, &aux_rng, nrow, nao, f0, f1,
                );
                let nb = f1 - f0;
                let stride1 = nrow * nao * nb;
                let mut acc = [0.0f64; 6];
                for comp in 0..3 {
                    let d1cur = &d1c[comp * stride1..(comp + 1) * stride1];
                    // int3c2e_ip1: rows `mu` on the atom, all `nu`.  `dI^(b)` is the transpose
                    // of `dI^(a)` and is already counted by the symmetrised covectors, so a
                    // single contiguous pass covers both bra centres.
                    // nu outside / mu_l inside keeps every access contiguous.
                    for pl in 0..nb {
                        let p = f0 + pl;
                        let cov = &adj.di_cov[p * n2..(p + 1) * n2];
                        let lt = adj.lam_tilde[p];
                        let base1 = pl * nrow * nao;
                        // The two accumulators are per auxiliary function: `lam_tilde` weights
                        // the `dm_sym` covector and is *not* constant over the batch.
                        let mut adi = [0.0f64; 4];
                        let mut adm = [0.0f64; 4];
                        for nu in 0..nao {
                            let doff = base1 + nu * nrow;
                            let coff = row0 + nu * nao;
                            let mut i = 0usize;
                            while i + 4 <= nrow {
                                let d = &d1cur[doff + i..doff + i + 4];
                                let c = &cov[coff + i..coff + i + 4];
                                let m = &dm[coff + i..coff + i + 4];
                                for j in 0..4 {
                                    let v = -d[j];
                                    adi[j] += c[j] * v;
                                    adm[j] += lt * m[j] * v;
                                }
                                i += 4;
                            }
                            while i < nrow {
                                let v = -d1cur[doff + i];
                                let k = coff + i;
                                adi[0] += cov[k] * v;
                                adm[0] += lt * dm[k] * v;
                                i += 1;
                            }
                        }
                        acc[comp] += adi.iter().sum::<f64>();
                        acc[3 + comp] += adm.iter().sum::<f64>();
                    }
                    // int3c2e_ip2: dense `[nao,nao]` symmetric slices of the atom's own auxiliary
                    // functions, with the 1/2 of the symmetrised contraction.  A batch outside
                    // the atom's own auxiliary functions has no `ip2` part at all.
                    let b_lo = f0.max(a0);
                    let b_hi = f1.min(a1);
                    let nb2 = b_hi.saturating_sub(b_lo);
                    if nb2 == 0 {
                        continue;
                    }
                    let stride2 = n2 * nb2;
                    let d2cur = &d2c[comp * stride2..(comp + 1) * stride2];
                    for pl in 0..nb2 {
                        let p = b_lo + pl;
                        let cov = &adj.di_cov[p * n2..(p + 1) * n2];
                        let lt = adj.lam_tilde[p];
                        let d2 = &d2cur[pl * n2..(pl + 1) * n2];
                        let mut cdi = [0.0f64; 4];
                        let mut cdm = [0.0f64; 4];
                        let mut k = 0usize;
                        while k + 4 <= n2 {
                            let d = &d2[k..k + 4];
                            let c = &cov[k..k + 4];
                            let m = &dm[k..k + 4];
                            for j in 0..4 {
                                let v = -0.5 * d[j];
                                cdi[j] += c[j] * v;
                                cdm[j] += lt * m[j] * v;
                            }
                            k += 4;
                        }
                        while k < n2 {
                            let v = -0.5 * d2[k];
                            cdi[0] += cov[k] * v;
                            cdm[0] += lt * dm[k] * v;
                            k += 1;
                        }
                        acc[comp] += cdi.iter().sum::<f64>();
                        acc[3 + comp] += cdm.iter().sum::<f64>();
                    }
                }
                    acc
                })
                .collect();
            for acc in parts.iter() {
                for comp in 0..3 {
                    g_di[comp] += acc[comp];
                    g_dm[comp] += acc[3 + comp];
                }
            }
            bi = hi;
        }
        let mut out = [0.0f64; 3];
        for comp in 0..3 {
            let vd1 = vxc_atom.map(|v| vxc_block_ao(v, comp, self.nao));
            out[comp] = self.finish_adjoint_gradient(
                adj, comp, h1r, &s1_all[comp], &djs[comp], bj_k, bb_k, &b_x3[comp],
                vd1.as_deref(), dir0 + comp, g_di[comp] + g_dm[comp],
            );
        }
        out
    }

    /// Shared tail of the two adjoint perturbation routines: everything except
    /// the `dI` contraction, whose value is passed in as `acc`.
    #[allow(clippy::too_many_arguments)]
    fn finish_adjoint_gradient(
        &self,
        adj: &CphfAdjoint,
        comp: usize,
        h1r: &[f64],
        s1_comp: &[f64],
        dj: &[f64],
        bj_k: &[f64],
        bb_k: &[f64],
        b_x: &[f64],
        vxc_d1_ao: Option<&[f64]>,
        dir: usize,
        acc: f64,
    ) -> f64 {
        let n2 = self.nao * self.nao;
        let naux = self.naux;
        let mut g = acc;
        // - sum_P lam_tilde_P (dJ c)_P, in fixed index blocks merged in order so
        // that the perturbation loop parities stay bit-identical under
        // `num_threads` (see `PERT_BLK`).
        let mut djc = 0.0f64;
        {
            const DJC_BLK: usize = 64;
            let blocks: Vec<usize> = (0..(naux + DJC_BLK - 1) / DJC_BLK).collect();
            let parts: Vec<f64> = blocks
                .par_iter()
                .map(|&b| {
                    let p0 = b * DJC_BLK;
                    let p1 = (p0 + DJC_BLK).min(naux);
                    let mut acc = 0.0f64;
                    for p in p0..p1 {
                        let mut s = 0.0f64;
                        for q in 0..naux {
                            s += dj[p + q * naux] * adj.c_aux[q];
                        }
                        acc += adj.lam_tilde[p] * s;
                    }
                    acc
                })
                .collect();
            for v in parts.iter() {
                djc += *v;
            }
        }
        g -= djc;
        // exchange skeleton remainder, core Hamiltonian and overlap
        // `Theta`/`Psi` are the unit-exchange adjoint tensors, so the skeleton
        // `-0.5 fock_hyb Kd` enters through `2 * half_k * k_scale`.
        g += 2.0 * adj.half_k * adj.k_scale * flat_dot(&adj.psi, dj);
        g += flat_dot(&adj.lam_ao, &h1r[comp * n2..(comp + 1) * n2]);
        g += flat_dot(&adj.lams_ao, s1_comp);
        // the folded `<Z, rhs_d>` channel: <W, dh1> + <W_s, dS> (the rest of
        // `<W, Jd - Kd/2>` rides the shared covectors)
        if adj.fold_rhs {
            g += flat_dot(&adj.w_dh1, &h1r[comp * n2..(comp + 1) * n2]);
            g += flat_dot(&adj.w_ds, s1_comp);
        }
        // DFT reference: the frozen-density v_xc skeleton enters the CP-KS
        // first-order Fock exactly like the core part of `f1mo`,
        //   f1_mo  += C^T v_xc^expl C ,
        // so it has *two* images in the adjoint route:
        //   - the `Z` right-hand side, `rhs_vxc[vir,occ] = -vxc_d1_mo[vir,occ]/D_ai`,
        //     whose fold is `<W, vxc_d1_ao>` with `W = Zbra C_occ^T`, and
        //   - the Roothaan covector itself, `<Lambda, v_xc^expl> = <Lambda_ao, vxc_d1_ao>`
        //     (the same channel as the `dh1` skeleton, which `lam_ao` carries).
        // Both are needed: they are the explicit-density and the
        // density-response halves of the derivative.
        if let Some(vd1) = vxc_d1_ao {
            g += flat_dot(&adj.w_dh1, vd1);
            if dft_debug_mask() & 8 != 0 {
                g += flat_dot(&adj.lam_ao, vd1);
            }
        }
        // metric channel, static term and the adjoint CP-HF channel
        g += flat_dot(bj_k, dj);
        g += flat_dot(bb_k, b_x);
        g += adj.rhs_dot[dir];
        g
    }
}

