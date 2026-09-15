//! Shared TDDFT data container + per-mode builders.
//!
//! `TDDFTData` holds everything the MO-based (`matvec.rs`) and AO-based
//! (`matvec_ao.rs`) matvec paths need. Members are `Option` and populated only
//! for the active mode:
//! - **MO mode**: `fxc` (full, with MO-on-grid projections) + the four
//!   MO-basis RI tensors (`ri_ov`, `ri_oo_exch`, `ri_vv_exch`, `ri_ov_exch`).
//! - **AO mode**: `fxc` (kernel-only, no MO projections) + `c_occ`/`c_vir`,
//!   the NIMatmul integrator (`ni`), the raw kernel (`fxc_eff`), `den_type`,
//!   `grid_batch`.

use rest_tensors::MatrixFull;

use crate::scf_io::SCF;
use crate::dft::num_int::{FXCMatvecData, prepare_fxc_data};
use crate::dft::Grids;
use crate::dft::numint_matmul::nimatmul::NIMatmul;
use crate::dft::numint_matmul::resp_rks::eval_vxc_fxc_from_rho;
use crate::dft::xceff::prelude::{determine_den_type, libxc_eval_eff, XCDenType, XCSpin};
use crate::ri_jk::util::get_cint_mol;
use crate::ri_tddft::utils::{tddft_occupation_parameters, tddft_get_submatrix};
use crate::utilities::rstsr_util::{RestTensorToRstsrViewAPI, Tsr};
use rstsr::prelude::*;
use libxc::prelude::*;

/// Which TDDFT matvec mode the shared data was prepared for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TDDFTMode {
    /// MO mode: precomputed MO-basis RI tensors (`matvec.rs`).
    MO,
    /// AO mode: AO transition-density kernels (`matvec_ao.rs`).
    AO,
}

/// Shared TDDFT data. Mode-specific members are `Option`; see module docs.
pub struct TDDFTData {
    /// The mode this data was prepared for (MO or AO).
    pub mode: TDDFTMode,
    /// Hybrid exchange coefficient c_x from the functional (shared).
    pub alpha_hybrid: f64,
    /// fxc kernel data, MO mode only (`None` in AO mode). Contains the
    /// MO-on-grid projections + weighted `wfxc` table. AO mode carries the
    /// raw kernel in `fxc_eff` and uses NIMatmul instead.
    pub fxc: Option<FXCMatvecData>,
    // ── AO mode only ──
    /// Occupied MO coefficients [nao, occ_size] (AO mode).
    pub c_occ: Option<MatrixFull<f64>>,
    /// Virtual MO coefficients [nao, vir_size] (AO mode).
    pub c_vir: Option<MatrixFull<f64>>,
    /// Numerical integrator for the batched fxc kernel (AO mode).
    pub ni: Option<NIMatmul<'static>>,
    /// Raw (unweighted) fxc kernel `[ngrids, nvar, nvar]` (AO mode).
    pub fxc_eff: Option<Tsr>,
    /// Density type (RHO/SIGMA) for AO mode.
    pub den_type: Option<XCDenType>,
    /// Batch the fxc AO evaluation over grid batches (AO mode, memory-bounded).
    pub grid_batch: bool,
    /// Resolved `tddft_fxc_driver` (AO mode); `None` for MO data — the fxc
    /// driver only applies in AO mode.
    pub fxc_driver: Option<FxcDriver>,
    /// Cached occ-MO projections on the grid (MO-style fxc driver):
    /// ψ_i(g) = Σ_μ C_μi φ_μ(g), layout [ngrids, nocc].
    pub psi_occ: Option<Tsr>,
    /// Cached occ-MO gradient projections (GGA, MO-style fxc driver): ∂_d ψ_i(g),
    /// layout [3, ngrids, nocc] with d ∈ {x, y, z}.
    pub psi_occ_grad: Option<Tsr>,
    // ── MO mode only ──
    /// [naux, occ*vir] RI tensor, Coulomb.
    pub ri_ov: Option<MatrixFull<f64>>,
    /// [occ*naux, occ] RI tensor, A-block exchange.
    pub ri_oo_exch: Option<MatrixFull<f64>>,
    /// [naux*vir, vir] RI tensor, A-block exchange.
    pub ri_vv_exch: Option<MatrixFull<f64>>,
    /// [naux*occ, vir] RI tensor, B-block exchange.
    pub ri_ov_exch: Option<MatrixFull<f64>>,
}

/// Prepare the shared TDDFT data for **MO mode**: the fxc kernel table
/// (`prepare_fxc_data`) plus the four MO-basis RI tensors.
/// Resolved `tddft_fxc_driver` for AO mode. Variants are spelled in the
/// established all-caps abbreviation form (cf. `TDDFTMode::MO`).
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FxcDriver {
    /// `"mo"`: occ/vir-reduced fxc; the vir side is streamed as C_vir-projected
    /// grid tables (the MO-mode fxc algorithm).
    MO,
    /// `"semitrans"`: C_vir folded into the amplitudes; the vir side contracts
    /// against the raw AO on grid, so no psi_vir is ever formed.
    SEMITRANS,
    /// `"dm"`: assembled-density NIMatmul fallback.
    DM,
}

pub fn prepare_mo_data(scf: &SCF) -> TDDFTData {
    println!("Obtaining RI integrals...");
    let (start_mo, num_state, occ_size, vir_size, homo, lumo) =
        tddft_occupation_parameters(scf);

    let fxc = prepare_fxc_data(scf);

    let ri_ov = tddft_get_submatrix(scf, 'O', 'V', start_mo, occ_size, vir_size, homo, lumo, num_state);
    let ri_oo = tddft_get_submatrix(scf, 'O', 'O', start_mo, occ_size, vir_size, homo, lumo, num_state);
    let ri_vv = tddft_get_submatrix(scf, 'V', 'V', start_mo, occ_size, vir_size, homo, lumo, num_state);
    let num_auxbas = ri_ov.size[0];
    println!("num_auxbas = {}", num_auxbas);

    // Reshape RI_OO for A-block exchange: [naux, occ*occ] → [occ*naux, occ]
    let mut ri_oo_exch = ri_oo.clone();
    ri_oo_exch.reshape([num_auxbas * occ_size, occ_size]);
    ri_oo_exch = ri_oo_exch.transpose_and_drop();
    ri_oo_exch.reshape([occ_size * num_auxbas, occ_size]);

    // Reshape RI_VV for A-block exchange: [naux, vir*vir] → [naux*vir, vir]
    let mut ri_vv_exch = ri_vv.clone();
    ri_vv_exch.reshape([num_auxbas * vir_size, vir_size]);

    // Reshape RI_OV for B-block exchange: [naux, occ*vir] → [naux*occ, vir]
    let mut ri_ov_exch = ri_ov.clone();
    ri_ov_exch.reshape([num_auxbas * occ_size, vir_size]);

    TDDFTData {
        mode: TDDFTMode::MO,
        alpha_hybrid: fxc.alpha_hybrid,
        fxc: Some(fxc),
        c_occ: None,
        c_vir: None,
        ni: None,
        fxc_eff: None,
        den_type: None,
        grid_batch: false,
        fxc_driver: None,
        psi_occ: None,
        psi_occ_grad: None,
        ri_ov: Some(ri_ov),
        ri_oo_exch: Some(ri_oo_exch),
        ri_vv_exch: Some(ri_vv_exch),
        ri_ov_exch: Some(ri_ov_exch),
    }
}

/// Prepare the shared TDDFT data for **AO mode**.
///
/// The fxc kernel is evaluated with the modern `numint_matmul` stack
/// (`eval_vxc_fxc_from_rho`), the same libxc wrapper used by the RKS response,
/// so the values match the MO path bit-identically. The **raw** kernel
/// `[ngrids, nvar, nvar]` is stored as `fxc_eff` (×2 singlet factor) for the
/// batched `NIMatmul` path, which applies the real grid weights internally.
/// AO mode carries no `FXCMatvecData` (`fxc: None`): the MO-on-grid
/// projections and weighted `wfxc` table are MO-only.
pub fn prepare_ao_data(scf: &SCF) -> TDDFTData {
    let (start_mo, _num_state, occ_size, vir_size, _homo, lumo) =
        tddft_occupation_parameters(scf);

    let grids = scf.grids.as_ref().expect("DFT grids must be initialized for AO-mode fxc");
    let ngrids = grids.weights.len();
    let num_basis = scf.mol.num_basis;
    let weights = &grids.weights;
    let xc_data = &scf.mol.xc_data;
    let nvar = if xc_data.use_density_gradient() { 4 } else { 1 };
    let alpha_hybrid = xc_data.dfa_hybrid_scf;
    let den_type = if nvar == 4 { XCDenType::SIGMA } else { XCDenType::RHO };

    // ── Numerical integrator with the real grid weights (AO cached via libcint) ──
    let cint = get_cint_mol(&scf.mol);
    let mut ni = NIMatmul::new(
        &cint,
        &grids.coordinates,
        weights,
        &grids.atm_idx,
        &grids.quadrature_weights,
    );
    let grid_batch = scf.mol.ctrl.tddft.as_ref().map_or(false, |t| t.grid_batch);

    // ── Ground-state density on grids, then the raw fxc kernel ──
    // rho0 is built from the occ-weighted density matrix (matches the MO path's
    // `eval_rho5_batch` ground-state rho), then libxc deriv=2 gives the raw
    // kernel `[ngrids, nvar, nvar]` — bit-identical to `prepare_fxc_data`.
    // When `grid_batch` is set, rho0 is assembled batch-by-batch via
    // `split_batch` so the full `[ngrids, nao, ncomp]` AO tensor is never cached.
    let device = DeviceBLAS::default();
    let dm0 = &scf.density_matrix[0];
    let rho0 = if grid_batch {
        let mut rho0 = rt::zeros(([ngrids, nvar, 1], &device));
        for start in (0..ngrids).step_by(ni.nbatch) {
            let end = (start + ni.nbatch).min(ngrids);
            let mut ni_batch = ni.split_batch(start, end);
            let dm0_view = dm0.to_rstsr_view(&device);
            let rho0_batch = ni_batch.make_rho_from_dm(&[dm0_view], den_type);
            rho0.i_mut((start..end, .., ..)).assign(&rho0_batch);
        }
        rho0
    } else {
        let dm0_view = dm0.to_rstsr_view(&device);
        ni.make_rho_from_dm(&[dm0_view], den_type) // [ngrids, nvar, 1]
    };

    let xc_func_list: Vec<(f64, LibXCFunctional)> = xc_data.dfa_compnt_scf
        .iter()
        .zip(xc_data.dfa_paramr_scf.iter())
        .map(|(&code, &param)| (param, LibXCFunctional::from_number(code as _, LibXCSpin::Unpolarized)))
        .collect();

    // ── Spin-channel-aware fxc kernel (CPL, 256, 454; PySCF `nr_rks_fxc_st`) ──
    // Singlet:   f_s = f↑↑ + f↑↓ = 2 × (unpolarized kernel at the total density).
    // Unpolarized ('R'): the bare unpolarized kernel (factor 1).
    // Triplet:   f_t = f↑↑ − f↑↓ cannot be obtained from an unpolarized
    //            evaluation; it requires a spin-polarized evaluation at
    //            (ρ/2, ∇ρ/2) per spin, combined along the antisymmetric direction.
    let tddft_spin = scf.mol.ctrl.tddft.as_ref().map_or("singlet", |t| t.tddft_spin.as_str());
    let rho0_g = rho0.i((.., .., 0)); // [ngrids, nvar] ground density + gradients
    let fxc_eff = match tddft_spin {
        "triplet" => {
            let xc_func_list_pol: Vec<(f64, LibXCFunctional)> = xc_data.dfa_compnt_scf
                .iter()
                .zip(xc_data.dfa_paramr_scf.iter())
                .map(|(&code, &param)| (param, LibXCFunctional::from_number(code as _, LibXCSpin::Polarized)))
                .collect();
            // Polarized closed-shell ground density (ρ/2, ∇ρ/2) per spin,
            // layout [ngrids, ncomp_in, 2] (GGA input: ncomp_in = 5, LDA: 1).
            let ncomp_in = if nvar == 4 { 5 } else { 1 };
            let mut rho_pol = rt::zeros(([ngrids, ncomp_in, 2], &device));
            for s in 0..2 {
                *&mut rho_pol.i_mut((.., 0, s)) += &(rho0_g.i((.., 0)) * 0.5);
                if ncomp_in == 5 {
                    for d in 0..3 {
                        *&mut rho_pol.i_mut((.., 1 + d, s)) += &(rho0_g.i((.., 1 + d)) * 0.5);
                    }
                }
            }
            // Spin-resolved second derivatives K[g, c1, s1, c2, s2], chain-ruled by
            // `transform_xc_inner` to per-spin (ρ_s, ∇ρ_s,x/y/z) variables. The
            // total-density response variable y ∈ {ρ, ∇ρ_x, ∇ρ_y, ∇ρ_z} maps onto
            // the same component index c = y of each spin channel.
            let nmax = if ncomp_in == 5 { 5 } else { 1 };
            let mut fxc_pol: Tsr = rt::zeros(([ngrids, nmax, 2, nmax, 2].f(), &device));
            for (scale, func) in &xc_func_list_pol {
                let ni_i = determine_den_type(func).num_nvar();
                let rho_i = rho_pol.i((.., ..ni_i, ..));
                let xc_eff = libxc_eval_eff(func, rho_i.view(), 2, None);
                *&mut fxc_pol.i_mut((.., ..ni_i, .., ..ni_i)) += *scale * xc_eff[2].view();
            }
            // Combine along the antisymmetric spin direction:
            // f_t[y1,y2] = ½ Σ_{s1,s2} (+1/−1) K[(y1,s1),(y2,s2)] = f↑↑ − f↑↓.
            let dir = [1.0_f64, -1.0_f64];
            let mut table: Tsr = rt::zeros(([ngrids, nvar, nvar].f(), &device));
            for s1 in 0..2 {
                for s2 in 0..2 {
                    let w = 0.5 * dir[s1] * dir[s2];
                    *&mut table += &(w * fxc_pol.i((.., ..nvar, s1, ..nvar, s2)));
                }
            }
            table
        }
        _ => {
            let (_, fxc_raw) = eval_vxc_fxc_from_rho(&xc_func_list, rho0_g); // [ngrids, nvar, nvar]
            const SINGLET_FXC_FACTOR: f64 = 2.0;
            match tddft_spin {
                "singlet" => SINGLET_FXC_FACTOR * fxc_raw,
                _ => fxc_raw, // 'R'/unpolarized response: bare kernel (factor 1)
            }
        }
    };

    // AO mode carries the raw `fxc_eff` kernel + NIMatmul; the weighted
    // `wfxc`/MO-projection table (FXCMatvecData) is MO-only, so `fxc: None`.
    let fxc: Option<FXCMatvecData> = None;

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

    // ── MO-style fxc driver: cache occ/vir MO projections on the grid ──
    // Built grid-batch-wise (split_batch) so the full [ngrids, nao, ncomp] AO
    // cache is never materialized; the small occ/vir tables are then reused by
    // every fxc matvec, making the per-call fxc cost occ/vir-reduced.
    // Layouts (t-ready, contiguous for the matvec GEMMs):
    // psi_occ [ngrids, nocc]; psi_occ_grad [3, ngrids, nocc]
    // (leading d-axis so the (d, chunk, ·) slices are contiguous).
    // These are the ONLY cached tables (small). The vir side (ψ_vir + its
    // gradients, which scale as nvir·ngrids) is NOT cached: fxc_mo_matvec
    // streams it per grid batch (AO eval + C_vir projection per batch) to
    // keep the memory footprint down.
    let fxc_driver = scf.mol.ctrl.tddft.as_ref().map_or(FxcDriver::SEMITRANS, |t| {
        match t.tddft_fxc_driver.as_str() {
            "mo" => FxcDriver::MO,
            "semitrans" => FxcDriver::SEMITRANS,
            "dm" => FxcDriver::DM,
            other => {
                log::warn!("Unknown tddft_fxc_driver = \"{}\"; falling back to \"dm\"", other);
                FxcDriver::DM
            }
        }
    });
    let (psi_occ, psi_occ_grad) = if matches!(fxc_driver, FxcDriver::MO | FxcDriver::SEMITRANS) {
        let c_occ_view = c_occ.to_rstsr_view(&device);
        let deriv = if nvar == 4 { 1 } else { 0 };
        let mut po = rt::zeros(([ngrids, occ_size].f(), &device));
        let mut pog = if nvar == 4 {
            Some(rt::zeros(([3, ngrids, occ_size].f(), &device)))
        } else {
            None
        };
        for start in (0..ngrids).step_by(ni.nbatch) {
            let end = (start + ni.nbatch).min(ngrids);
            let mut ni_batch = ni.split_batch(start, end);
            let ao = ni_batch.get_cached_ao(deriv); // [nb, nao, ncomp]
            let ao0 = ao.i((.., .., 0)); // [nb, nao]
            // ψ[g, i] = Σ_μ ao[g, μ] C[μ, i]  (project onto occ coefficients)
            po.i_mut((start..end, ..)).matmul_from(&ao0, &c_occ_view, 1.0, 0.0);
            if let Some(pog) = (&mut pog) {
                for d in 0..3 {
                    let aod = ao.i((.., .., 1 + d)); // ∂_d φ
                    pog.i_mut((d, start..end, ..)).matmul_from(&aod, &c_occ_view, 1.0, 0.0);
                }
            }
        }
        (Some(po), pog)
    } else {
        (None, None)
    };

    TDDFTData {
        mode: TDDFTMode::AO,
        alpha_hybrid,
        fxc,
        c_occ: Some(c_occ),
        c_vir: Some(c_vir),
        ni: Some(ni),
        fxc_eff: Some(fxc_eff),
        den_type: Some(den_type),
        grid_batch,
        fxc_driver: Some(fxc_driver),
        psi_occ,
        psi_occ_grad,
        ri_ov: None,
        ri_oo_exch: None,
        ri_vv_exch: None,
        ri_ov_exch: None,
    }
}

/// Build the full A matrix `[dim, dim]` directly (dense small-system path).
///
/// Mode-dispatching: AO mode applies the kernel block to the identity (one
/// batched J/K/fxc call); MO mode loops unit vectors through `a_matvec`.
pub fn build_a(scf: &SCF, data: &mut TDDFTData, xlet: char) -> MatrixFull<f64> {
    let (_start_mo, _, occ_size, vir_size, _homo, _lumo) = tddft_occupation_parameters(scf);
    let dim = occ_size * vir_size;
    match data.mode {
        TDDFTMode::AO => crate::ri_tddft::matvec_ao::build_a_ao(scf, data, xlet),
        TDDFTMode::MO => {
            let mut a_full = MatrixFull::new([dim, dim], 0.0);
            for col in 0..dim {
                let mut e_col = vec![0.0; dim];
                e_col[col] = 1.0;
                let a_col = crate::ri_tddft::matvec::a_matvec(scf, data, &e_col, xlet);
                for row in 0..dim {
                    a_full[[row, col]] = a_col[row];
                }
            }
            a_full
        }
    }
}

/// Build the full B matrix `[dim, dim]` directly (dense small-system path).
pub fn build_b(scf: &SCF, data: &mut TDDFTData, xlet: char) -> MatrixFull<f64> {
    let (_start_mo, _, occ_size, vir_size, _homo, _lumo) = tddft_occupation_parameters(scf);
    let dim = occ_size * vir_size;
    match data.mode {
        TDDFTMode::AO => crate::ri_tddft::matvec_ao::build_b_ao(scf, data, xlet),
        TDDFTMode::MO => {
            let mut b_full = MatrixFull::new([dim, dim], 0.0);
            for col in 0..dim {
                let mut e_col = vec![0.0; dim];
                e_col[col] = 1.0;
                let b_col = crate::ri_tddft::matvec::b_matvec(scf, data, &e_col, xlet);
                for row in 0..dim {
                    b_full[[row, col]] = b_col[row];
                }
            }
            b_full
        }
    }
}