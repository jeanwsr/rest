//! One SHARC non-adiabatic-dynamics step.
//!
//! This is the driver behind `gw_or_bse = "bse_sharc"`; it keeps the SHARC
//! protocol details out of `post_scf_analysis::mod`.

use super::*;

/// Run one SHARC quantum-chemistry step.
///
/// Solves `sharc_nstates - 1` BSE roots at the current geometry, assembles the
/// analytic gradient of every state (ground state + roots), the inter-state
/// overlap matrix against `sharc_previous_state`, and the non-adiabatic
/// coupling vectors.  Writes `sharc_output` (the step result read by
/// `SHARC_REST.py`) and `sharc_state_output` (the restart file for the next
/// step), and returns the excitation energies.
pub(super) fn run_step(
    scf_data: &mut SCF,
    qp_ctrl: &crate::ctrl_io::quasiparticle_methods::QuasiParticle,
) -> anyhow::Result<Vec<f64>> {
    if scf_data.mol.spin_channel == 2 {
        panic!(
            "The SHARC interface (`gw_or_bse = \"bse_sharc\"`) supports only \
             spin-restricted (closed-shell) references."
        );
    }
    use crate::ri_bse::bse_grad::{BseGradEngine, BseSolver, ScreeningEnergy};
    use crate::ri_bse::sharc as sharc_backend;

    let nmo = scf_data.mol.num_state;
    let nao = scf_data.mol.num_basis;
    let natom = scf_data.mol.geom.elem.len();
    let nstates_total = if qp_ctrl.sharc_nstates > 1 {
        qp_ctrl.sharc_nstates
    } else if qp_ctrl.gradient_nroots > 0 {
        qp_ctrl.gradient_nroots + 1
    } else {
        qp_ctrl.gradient_state + 2
    };
    let nroots = nstates_total - 1;
    let multi = if qp_ctrl.bse_spin.eq_ignore_ascii_case("triplet") {'t'} else {'s'};
    let screen = if qp_ctrl.gradient_screening.eq_ignore_ascii_case("qp") {
        ScreeningEnergy::Qp
    } else {
        ScreeningEnergy::Reference
    };
    let mut cfg = ri_gw::gw_grad::GwGradConfig::from_scf(scf_data);
    // Consistency of d(Omega)/dR with Omega(R) requires the QP energies and
    // their geometric response to come from the same solve; pinning them to an
    // externally computed GW vector breaks that (measured 20-30 % error).
    cfg.pin_qp_from_checkpoint = false;
    cfg.z_min = qp_ctrl.gradient_z_min;
    cfg.force_static_fallback = qp_ctrl
        .gradient_force_fallback
        .split(',')
        .filter_map(|t| t.trim().parse::<usize>().ok())
        .collect();
    cfg.smooth_pole = qp_ctrl.gradient_smooth_pole;
    cfg.exact_residue_z = qp_ctrl.gradient_exact_residue;
    cfg.graphical_rescue = qp_ctrl.gradient_graphical_rescue;
    cfg.force_linearized = qp_ctrl
        .gradient_force_linearized
        .split(',')
        .filter_map(|t| t.trim().parse::<usize>().ok())
        .collect();

    let (cutoff, occ_cutoff) = bse_gradient_cutoffs(scf_data, qp_ctrl);
    let vxc = precompute_vxc_and_release_grid_ao(scf_data);

    // Solve the BSE and (optionally) differentiate every root.
    let (omegas, grads, x, y, occ_start, nocc_active, vir_start, nvir_active) = {
        let pin_qp = cfg.pin_qp_from_checkpoint && !scf_data.gwqp.0.is_empty();
        let is_cd = !matches!(
            qp_ctrl.gw_variant,
            crate::ctrl_io::quasiparticle_methods::GwVariant::Ac | crate::ctrl_io::quasiparticle_methods::GwVariant::Wac
        );
        let mut vxc = Some(vxc);
        let mut cd_engine = if is_cd {
            Some(ri_gw::gw_grad::GwCdGradEngine::new_with_vxc(
                scf_data, cfg.clone(), vxc.take().unwrap(),
            ))
        } else {
            None
        };
        let mut gw_engine = if is_cd {
            None
        } else {
            Some(ri_gw::gw_grad::GwGradEngine::new_with_vxc(
                scf_data, cfg, vxc.take().unwrap(),
            ))
        };
        if pin_qp {
            if let Some(cd) = cd_engine.as_mut() {
                cd.base.pin_qp_energies(scf_data.gwqp.0.clone());
            }
            if let Some(gw) = gw_engine.as_mut() {
                gw.pin_qp_energies(scf_data.gwqp.0.clone());
            }
        }
        let gw_ref = cd_engine
            .as_ref()
            .map(|cd| &cd.base)
            .or(gw_engine.as_ref())
            .expect("BSE gradient: no GW engine");
        let mut bse = BseGradEngine::new_with_window(
            scf_data, gw_ref, cd_engine.as_ref(), multi, screen, cutoff, occ_cutoff);
        println!(
            "SHARC step {}: BSE transition space {} occupied x {} of {} virtuals, spin channel '{multi}'",
            qp_ctrl.sharc_step_index, bse.screen.nocc, bse.screen.nvir, bse.screen.nvir_full
        );
        let dim = bse.screen.nocc * bse.screen.nvir;
        let solver = if dim <= 128 { BseSolver::Dense } else { BseSolver::Davidson };
        let mut dcfg = crate::solvers::davidson::DavidsonConfig::default();
        dcfg.add_dim = dcfg.add_dim.max(nroots + 2);
        dcfg.restart_dim = dcfg.restart_dim.max(nroots + 2);
        dcfg.max_subspace = dcfg.max_subspace.max(12 * (nroots + 2)).max(200);
        dcfg.max_iter = 400;
        bse.solve(nroots, solver, &dcfg);
        for (k, om) in bse.excitation_energies().iter().enumerate() {
            println!(
                "  SHARC state {}: {:.12} Ha = {:.6} eV",
                k + 1, om, om * 27.211386245988
            );
        }
        report_root_composition(scf_data, &bse);
        let omegas = bse.excitation_energies();
        let grads = if qp_ctrl.sharc_do_grad {
            bse.analytic_gradients().0
        } else {
            Vec::new()
        };
        let x: Vec<Vec<f64>> = bse.roots.iter().map(|r| r.x.clone()).collect();
        let y: Vec<Vec<f64>> = bse.roots.iter().map(|r| r.y.clone()).collect();
        (
            omegas, grads, x, y,
            bse.screen.occ_start, bse.screen.nocc,
            bse.screen.vir_start, bse.screen.nvir,
        )
    };
    if qp_ctrl.sharc_do_grad {
        assert_eq!(
            grads.len(), nroots,
            "SHARC: {} gradients assembled for {} roots", grads.len(), nroots
        );
    }

    // ---- state record ---------------------------------------------------
    let mut mo = vec![0.0; nao * nmo];
    {
        let c = &scf_data.eigenvectors[0];
        for p in 0..nao {
            for q in 0..nmo {
                mo[p * nmo + q] = c[[p, q]];
            }
        }
    }
    // Deterministic MO / amplitude phases: without them the arbitrary LAPACK
    // eigenvector signs make `det(S_oo)` and the root signs jump between steps.
    sharc_backend::fix_mo_phases(&mut mo, nao, nmo);
    let mut x = x;
    let mut y = y;
    sharc_backend::fix_amplitude_phases(&mut x, &mut y);
    // Track the state phases along the trajectory: the arbitrary sign of a
    // root is fixed once (above), but it can still flip when the leading
    // configuration changes.  Aligning each root with the previous step keeps
    // the diagonal of the overlap matrix positive and smooth.
    let prev_state: Option<sharc_backend::SharcState> =
        if qp_ctrl.sharc_previous_state.is_empty() {
            None
        } else {
            Some(sharc_backend::SharcState::load(&qp_ctrl.sharc_previous_state)?)
        };
    let mut geometry_bohr = Vec::with_capacity(natom);
    for a in 0..natom {
        geometry_bohr.push([
            scf_data.mol.geom.position[[0, a]],
            scf_data.mol.geom.position[[1, a]],
            scf_data.mol.geom.position[[2, a]],
        ]);
    }

    // Align the MO phases with the previous step and re-express the amplitudes
    // in the re-signed basis, so that every diagonal overlap `S_II` stays
    // positive along the trajectory (a flipped virtual orbital would otherwise
    // turn the overlap of the states that contain it into `-1`).
    let mol_for_overlap = scf_data.mol.clone();
    let mut s_mo_prev: Option<Vec<f64>> = None;
    if let Some(p) = prev_state.as_ref() {
        if p.nmo == nmo && p.nao == nao {
            let s0 = sharc_backend::cross_mo_overlap_at(
                &mol_for_overlap, &p.mo, nmo, nao, &p.geometry_bohr, &mo, &geometry_bohr);
            let signs = sharc_backend::mo_overlap_signs(&s0, nmo);
            sharc_backend::apply_mo_signs(&mut mo, nao, nmo, &signs);
            sharc_backend::apply_amplitude_signs(
                &mut x, &mut y, &signs, occ_start, nocc_active, vir_start, nvir_active);
            s_mo_prev = Some(sharc_backend::cross_mo_overlap_at(
                &mol_for_overlap, &p.mo, nmo, nao, &p.geometry_bohr, &mo, &geometry_bohr));
            // The MO re-signing above changes the *representation* of the
            // amplitudes, so the global root phases have to be aligned only
            // afterwards -- otherwise a state whose orbital was re-signed
            // comes out with the opposite sign and the overlap matrix mixes
            // `+` and `-` representations of the same state.
            sharc_backend::align_phases_to_previous(&mut x, &mut y, &p.x, &p.y);
        }
    }

    let cur = sharc_backend::SharcState {
        step: qp_ctrl.sharc_step_index,
        natom,
        elements: scf_data.mol.geom.elem.clone(),
        geometry_bohr: geometry_bohr.clone(),
        nao,
        nmo,
        nocc: vir_start,
        occ_start,
        nocc_active,
        vir_start,
        nvir_active,
        mo: mo.clone(),
        omega: omegas.clone(),
        x: x.clone(),
        y: y.clone(),
        scf_energy: scf_data.scf_energy,
    };

    // Everything present in the step result, in SHARC's ordering:
    // state 0 is the ground state, state I >= 1 is BSE root I-1.
    let ns = nstates_total;
    let mut energies = vec![0.0; ns];
    energies[0] = scf_data.scf_energy;
    for k in 0..nroots {
        energies[k + 1] = scf_data.scf_energy + omegas[k];
    }

    let mut gradients_out: Option<Vec<Vec<f64>>> = None;
    if qp_ctrl.sharc_do_grad {
        let gs = reference_gradient(scf_data);
        let mut g = Vec::with_capacity(ns);
        g.push(gs.clone());
        for k in 0..nroots {
            let mut v = Vec::with_capacity(3 * natom);
            for c in 0..(3 * natom) {
                v.push(gs[c] + grads[k][c]);
            }
            g.push(v);
        }
        report_analytic_gradient(scf_data, "SHARC_S0", &g[0]);
        gradients_out = Some(g);
    }

    // ---- inter-state overlap matrix (previous geometry) -----------------
    let mol = scf_data.mol.clone();
    let mut overlap_out: Option<Vec<Vec<f64>>> = None;
    if qp_ctrl.sharc_do_overlap && prev_state.is_some() {
        let prev = prev_state.as_ref().unwrap();
        sharc_backend::check_compatible(prev, &cur)?;
        let s_mo = s_mo_prev.clone().unwrap_or_else(|| {
            sharc_backend::cross_mo_overlap_at(
                &mol, &prev.mo, nmo, nao, &prev.geometry_bohr, &cur.mo, &geometry_bohr)
        });
        let s = sharc_backend::overlap_matrix(prev, &cur, &s_mo);
        println!("SHARC inter-state overlap matrix (previous -> current geometry):");
        for i in 0..s.len() {
            let row: Vec<String> = (0..s.len()).map(|j| format!("{:+.6}", s[i][j])).collect();
            println!("  {:2}: {}", i, row.join(" "));
        }
        overlap_out = Some(s);
    }

    // ---- non-adiabatic coupling vectors ---------------------------------
    let mut nac_out: Option<Vec<Vec<Vec<f64>>>> = None;
    if qp_ctrl.sharc_do_nac && !qp_ctrl.sharc_nac_mode.eq_ignore_ascii_case("off") {
        let s_mo = sharc_backend::cross_mo_overlap_at(
            &mol, &cur.mo, nmo, nao, &geometry_bohr, &cur.mo, &geometry_bohr);
        let ds_mo = sharc_backend::cross_mo_overlap_deriv(
            &mol, &cur.mo, nmo, nao, &geometry_bohr, &cur.mo, &geometry_bohr,
            qp_ctrl.sharc_nac_fd_step);
        let nac = sharc_backend::nac_vectors(&cur, &s_mo, &ds_mo);
        println!("SHARC non-adiabatic couplings (frozen-amplitude overlap derivative):");
        for i in 0..nac.len() {
            for j in 0..nac.len() {
                if i == j {
                    continue;
                }
                let norm = nac[i][j].iter().map(|v| v * v).sum::<f64>().sqrt();
                println!("  d[{i},{j}] |d| = {norm:.6e} a.u.");
            }
        }
        nac_out = Some(nac);
    }

    let dipoles_out = if qp_ctrl.sharc_do_dipole {
        Some(sharc_backend::dipole_moments(scf_data, &cur))
    } else {
        None
    };

    // ---- write the files -------------------------------------------------
    let step_out = sharc_backend::SharcStep {
        natom,
        elements: cur.elements.clone(),
        geometry_bohr,
        nstates: ns,
        energies,
        gradients: gradients_out,
        nac: nac_out,
        overlap: overlap_out,
        dipoles: dipoles_out,
        omega: omegas.clone(),
        multiplicities: vec![1],
        units: String::from("atomic"),
        step: qp_ctrl.sharc_step_index,
    };
    if !qp_ctrl.sharc_output.is_empty() {
        step_out
            .save(&qp_ctrl.sharc_output)
            .map_err(|e| anyhow::anyhow!("cannot write {}: {e}", qp_ctrl.sharc_output))?;
        println!("SHARC step result written to {}", qp_ctrl.sharc_output);
    }
    if !qp_ctrl.sharc_state_output.is_empty() {
        cur.save(&qp_ctrl.sharc_state_output)
            .map_err(|e| anyhow::anyhow!("cannot write {}: {e}", qp_ctrl.sharc_state_output))?;
        println!("SHARC restart state written to {}", qp_ctrl.sharc_state_output);
    }

    Ok(omegas)
}

