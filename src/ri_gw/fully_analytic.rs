// ============================================================================
// MolGW-style fully analytic G0W0
//
// This is the *diagonal* G0W0 variant: the correlation self-energy is built
// from the spectral (RPA pole/residue) representation of W_c,
//
//     W_c(ω) = Σ_s w_s w_s^T [1/(ω-Ω_s) - 1/(ω+Ω_s)],
//
// and each quasiparticle energy is obtained by solving the scalar equation
//
//     E_n = ε_n + (1-α) Σ_x,nn - V_xc,nn + Re Σ_c,nn(E_n).
//
// It is the REST/RI/libcint transcription of MolGW's `gw_selfenergy` fully
// analytic path.  The nonlinear FEAST driver in `nonlingw` uses the same RPA
// spectral representation but keeps the complete off-diagonal self-energy
// matrix; setting `feast_gw_zero_offdiag = true` there must reduce to this
// diagonal answer.
// ============================================================================

use std::time::Instant;

use rest_tensors::matrix::matrix_blas_lapack::_dgemm_full;
use tensors::{MathMatrix, MatrixFull};

use crate::ctrl_io::quasiparticle_methods::QuasiParticle;
use crate::scf_io::SCF;

use super::{
    compute_ri3mo_row, get_occupation_parameters,
    linear_interpolation_solver, static_exchange, exchange_matrix_from_scf, vxc_ao2mo,
};
use super::nonlingw::build_rpa_setup;

/// Diagonal correlation self-energy from the RPA spectral representation.
struct DiagonalSelfEnergy {
    /// One `[n_poles, n_g]` matrix per target orbital.
    u: Vec<MatrixFull<f64>>,
    g_energies: Vec<f64>,
    g_occupied: Vec<bool>,
    omega: Vec<f64>,
    eta: f64,
}

impl DiagonalSelfEnergy {
    fn sigma_nn(&self, target: usize, lambda: f64) -> f64 {
        let n_g = self.g_energies.len();
        let u = &self.u[target];
        let mut acc = 0.0_f64;
        for (s, &om) in self.omega.iter().enumerate() {
            for m in 0..n_g {
                let shift = if self.g_occupied[m] { om } else { -om };
                let denom = lambda - self.g_energies[m] + shift;
                let d = if self.eta > 0.0 {
                    denom / (denom * denom + self.eta * self.eta)
                } else if denom.abs() < 1.0e-14 {
                    0.0
                } else {
                    1.0 / denom
                };
                let val = u[[s, m]];
                acc += val * val * d;
            }
        }
        acc
    }
}

/// Top-level driver for `gw_or_bse = "fully_analytic_gw"`.
pub fn fully_analytic_gw_main(
    scf_data: &mut SCF,
    qp_ctrl: &QuasiParticle,
    _mpi_operator: &Option<crate::mpi_io::MPIOperator>,
) {
    let start = Instant::now();
    if scf_data.mol.spin_channel != 1 {
        panic!(
            "[fully_analytic_gw] currently supports only closed-shell \
             (spin-restricted) references."
        );
    }
    super::initialize_qp_g_w(scf_data);

    let (start_mo, num_state, occ_size, _vir_size, homo, lumo) =
        get_occupation_parameters(scf_data, 'Y');
    let (_, num_state_cutoff, _, _, _, _) = get_occupation_parameters(scf_data, 'N');

    println!("========================================");
    println!("  Fully analytic G0W0 (diagonal Σ_c)");
    println!("========================================");
    println!(
        "  MO window: start={}, num_state={}, occ={}, homo={}, lumo={}, target states={}",
        start_mo, num_state, occ_size, homo, lumo, num_state_cutoff
    );

    let qp_g = scf_data.gwqp.0.clone();
    let qp_w = scf_data.gwqp.1.clone();

    // ── RPA spectral representation of W_c ──────────────────────────────
    let t_rpa = Instant::now();
    let rpa_setup = build_rpa_setup(scf_data, qp_ctrl, &qp_w);
    let rpa = rpa_setup.rpa;
    let n_aux = rpa_setup.n_aux;
    println!(
        "[fully_analytic_gw] RPA: {} transitions -> {} poles, n_aux={}, built in {:?}",
        rpa_setup.n_trans,
        rpa.omega.len(),
        n_aux,
        t_rpa.elapsed()
    );

    // ── Diagonal U_s[n,m] = Σ_P (P|n m) w_s[P] ─────────────────────────
    let t_u = Instant::now();
    let n_g = num_state - start_mo;
    let mut w_mat = MatrixFull::new([n_aux, rpa.omega.len()], 0.0);
    for (s, w) in rpa.w_s.iter().enumerate() {
        for p in 0..n_aux {
            w_mat[[p, s]] = w[p];
        }
    }
    let n_target = num_state_cutoff
        .saturating_sub(start_mo)
        .min(num_state - start_mo);
    let mut u = Vec::with_capacity(n_target);
    let mut u_all = MatrixFull::new([rpa.omega.len(), n_g], 0.0);
    for p in 0..n_target {
        let ri_row = compute_ri3mo_row(scf_data, p);
        // ri_row is [n_aux, n_g] with column k = (P | p, start_mo+k)
        _dgemm_full(&w_mat, 'T', &ri_row, 'N', &mut u_all, 1.0, 0.0);
        u.push(u_all.clone());
        if (p + 1) % 10 == 0 || p + 1 == n_target {
            println!(
                "  [fully_analytic_gw] diagonal self-energy rows: {}/{}",
                p + 1,
                n_target
            );
        }
    }
    let sigma = DiagonalSelfEnergy {
        u,
        g_energies: (start_mo..num_state).map(|g| qp_g[g]).collect(),
        g_occupied: (start_mo..num_state).map(|g| g <= homo).collect(),
        omega: rpa.omega,
        eta: qp_ctrl.feast_gw_eta.abs(),
    };
    println!(
        "[fully_analytic_gw] diagonal self-energy built in {:?}",
        t_u.elapsed()
    );

    // ── Static term and scalar QP equation ──────────────────────────────
    let vxc_nn = vxc_ao2mo(scf_data);
    let v_mat = exchange_matrix_from_scf(scf_data);
    let mut qp_energies = qp_g.clone();
    println!("\n  Solving scalar QP equations:");
    println!("  MO    E_KS (Ha)       E_QP (Ha)       E_QP (eV)");
    println!("  ────  ──────────────  ──────────────  ───────────");
    for p in 0..n_target {
        let global_n = start_mo + p;
        let e_ks = scf_data.eigenvalues[0][global_n];
        let consts = e_ks + static_exchange(scf_data, &v_mat, global_n, homo)
            - vxc_nn[global_n];
        let side = if global_n > homo { 1.0 } else { -1.0 };
        let qp_eq = |omega: f64| consts + sigma.sigma_nn(p, omega) - omega;
        let (_have_crossing, qp) = linear_interpolation_solver(
            qp_eq,
            e_ks,
            side,
            qp_ctrl.gw_search_grid,
            qp_ctrl.gw_span_energy,
        );
        qp_energies[global_n] = qp;
        println!(
            "  {:>4}  {:>14.9}  {:>14.9}  {:>11.6}",
            global_n,
            e_ks,
            qp,
            qp * crate::constants::EV
        );
    }

    scf_data.gwqp.0 = qp_energies.clone();
    println!(
        "\n[fully_analytic_gw] total time: {:?}",
        start.elapsed()
    );
    super::display::full_quasiparticles(&qp_energies, occ_size);
}
