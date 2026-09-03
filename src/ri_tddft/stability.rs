//! SCF wave-function stability analysis for RKS/RHF and UKS/UHF references,
//! built on the AO-mode TDDFT orbital-Hessian (A/B-operator) machinery.
//!
//! The stability operator is the **(A+B) orbital Hessian** (Seeger & Pople,
//! JCP 66, 3045 (1977)) — the full second derivative of the energy w.r.t.
//! occ→vir orbital rotations, including the B-type (de-excitation /
//! symmetrized-density) response — expressed here through the TDDFT A and B
//! matvecs on the same reference. With the REST TDA/RPA conventions:
//!
//! - RHF/RKS internal:   H = 4·(A^singlet + B^singlet)
//! - RHF/RKS external:   H = (A^triplet + B^triplet)   [the RHF→UHF
//!   instability direction is the triplet channel — PySCF `hop_rhf2uhf`,
//!   "see also rhf.TDA triplet excitation"]
//! - UHF/UKS internal:   H = 2·(A + B) over the concatenated [α;β] occ→vir
//!   space
//!
//! (factors follow PySCF `rhf_internal`/`uhf_internal` = hop·2 with
//! `gen_g_hop` = 2(A+B), and `rhf_external` = (A+B)^triplet un-scaled; the
//! former incomplete scf_io stability used the same 4/2 factors.)
//!
//! The lowest eigenvalues are obtained with the batched Davidson solver; the
//! solution is (internally/externally) stable iff λ_min ≥ −1e-5 (the PySCF
//! threshold). A negative eigenvalue means the SCF converged to a saddle
//! point; the eigenvector spans the downhill orbital-rotation direction.
//!
//! Not implemented (by scope): the real→complex (¹(A′−B′)) and UHF→GHF
//! external checks, ROHF stability, and the orbital "follow" (rotation +
//! re-run). Requires a DFT reference (the AO kernel machinery needs the
//! integration grid).
//!
//! References: Seeger & Pople, JCP 66, 3045 (1977); Bauernschmitt & Ahlrichs,
//! JCP 104, 9047 (1996); PySCF `scf/stability.py`.

use std::cell::RefCell;

use rest_tensors::MatrixFull;

use crate::ri_tddft::matvec::build_hdiag;
use crate::ri_tddft::matvec::build_hdiag_u;
use crate::ri_tddft::matvec_ao::{a_matvec_ao_batched, b_matvec_ao_batched};
use crate::ri_tddft::tddft::{prepare_ao_data_with_spin, TDDFTData};
use crate::scf_io::{SCF, SCFType};
use crate::solvers::davidson::{davidson_solver_batched, generate_initial_guess, DavidsonConfig};

/// PySCF's stability threshold: λ_min below this marks an instability.
pub const STABILITY_THRESHOLD: f64 = -1.0e-5;

/// Result of a stability analysis (check-only; no orbital following).
#[derive(Debug, Clone)]
pub struct StabilityReport {
    /// internal stability (RHF/RKS singlet or UHF/UKS orbital Hessian)
    pub stable_internal: Option<bool>,
    pub roots_internal: Vec<f64>,
    /// external stability (RHF→UHF triplet check; `None` for UKS — the
    /// UHF→GHF check is not implemented)
    pub stable_external: Option<bool>,
    pub roots_external: Vec<f64>,
}

/// Run the stability analysis requested by `[tddft] stability`
/// ("internal" | "external" | "full"), on the converged SCF reference.
pub fn stability(scf: &SCF) -> Result<StabilityReport, String> {
    let (mode, nroots, tol) = {
        let t = scf
            .mol
            .ctrl
            .tddft
            .as_ref()
            .ok_or("stability analysis requires the [tddft] input section")?;
        (t.stability.clone(), t.stability_nroots, t.stability_tol)
    };
    let do_internal = mode == "internal" || mode == "full";
    let do_external = mode == "external" || mode == "full";
    if !do_internal && !do_external {
        return Err(format!(
            "[tddft] stability = \"{mode}\" is not one of internal/external/full"
        ));
    }
    if scf.scftype == SCFType::ROHF {
        return Err("stability analysis is not implemented for ROHF references".to_string());
    }
    if scf.grids.is_none() {
        return Err(
            "stability analysis requires the DFT grids (only DFT references are supported)"
                .to_string(),
        );
    }
    let is_uhf = scf.scftype == SCFType::UHF;
    let method = if is_uhf { "UHF/UKS" } else { "RHF/RKS" };

    let mut report = StabilityReport {
        stable_internal: None,
        roots_internal: Vec::new(),
        stable_external: None,
        roots_external: Vec::new(),
    };

    if do_internal {
        // (factor, xlet): RHF internal H = 4(A^S+B^S); UKS internal H = 2(A+B)
        let (factor, xlet) = if is_uhf { (2.0, 'R') } else { (4.0, 'S') };
        let spin_override = if is_uhf { None } else { Some("singlet") };
        let (roots, stable) = {
            let data = RefCell::new(prepare_ao_data_with_spin(scf, spin_override));
            let hdiag = if is_uhf { build_hdiag_u(scf) } else { build_hdiag(scf) };
            hessian_roots(scf, &data, xlet, factor, &hdiag, nroots, tol)
        };
        report.roots_internal = roots;
        report.stable_internal = Some(stable);
        println!(
            "{method} internal stability: lowest eigenvalues = {:?}",
            report.roots_internal
        );
        println!(
            "{method} wavefunction {}",
            stability_phrase(report.stable_internal.unwrap(), "internal")
        );
    }

    if do_external {
        if is_uhf {
            println!("UHF/UKS -> GHF/GKS external stability: not implemented (skipped)");
        } else {
            // RHF→UHF: (A^T + B^T), factor 1, on the restricted reference
            let (roots, stable) = {
                let data = RefCell::new(prepare_ao_data_with_spin(scf, Some("triplet")));
                let hdiag = build_hdiag(scf);
                hessian_roots(scf, &data, 'T', 1.0, &hdiag, nroots, tol)
            };
            report.roots_external = roots;
            report.stable_external = Some(stable);
            println!(
                "{method} -> UHF/UKS external stability: lowest eigenvalues = {:?}",
                report.roots_external
            );
            println!(
                "{method} wavefunction {}",
                stability_phrase(stable, "RHF/RKS -> UHF/UKS external")
            );
        }
    }

    Ok(report)
}

fn stability_phrase(stable: bool, kind: &str) -> String {
    if stable {
        format!("is stable in the {kind} stability analysis")
    } else {
        format!("has an {kind} instability")
    }
}

/// Lowest eigenvalues of `factor × ((A+B) restricted orbital Hessian)` — the
/// A and B actions via the AO-mode TDA/RPA matvecs with `xlet` — in the
/// occ→vir rotation space, via the batched Davidson solver.
fn hessian_roots(
    scf: &SCF,
    data: &RefCell<TDDFTData>,
    xlet: char,
    factor: f64,
    hdiag: &Vec<f64>,
    nroots: usize,
    tol: f64,
) -> (Vec<f64>, bool) {
    let dim: usize = hdiag.len();
    if dim == 0 {
        return (Vec::new(), true);
    }
    // The stability Hessian is factor × (A+B); its diagonal ≈ factor × gap
    let hdiag_scaled: Vec<f64> = hdiag.iter().map(|v| v * factor).collect();
    let config = DavidsonConfig {
        tol,
        max_subspace: 60,
        add_dim: nroots + 2,
        max_iter: 200,
        ..Default::default()
    };
    let initial_guess = generate_initial_guess(&hdiag_scaled, nroots);
    let mut eigenpairs = davidson_solver_batched(
        |z: &MatrixFull<f64>| {
            // stability Hessian = factor × (A+B); both matvecs share the
            // prepared data (sequential &mut borrows through the RefCell)
            let mut r = a_matvec_ao_batched(scf, &mut data.borrow_mut(), z, xlet);
            let rb = b_matvec_ao_batched(scf, &mut data.borrow_mut(), z, xlet);
            for (v, b) in r.data.iter_mut().zip(rb.data.iter()) {
                *v = factor * (*v + *b);
            }
            r
        },
        nroots,
        &hdiag_scaled,
        initial_guess,
        &config,
    );
    eigenpairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let roots: Vec<f64> = eigenpairs.iter().map(|(e, _)| *e).collect();
    let stable = roots.first().map_or(true, |e| *e > STABILITY_THRESHOLD);
    (roots, stable)
}
