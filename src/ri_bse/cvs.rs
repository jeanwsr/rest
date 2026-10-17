//! Core-valence separated (CVS) Bethe-Salpeter equation.
//!
//! The ordinary BSE in [`super`] builds its transition space from *all*
//! occupied orbitals.  Core excitations (e.g. a N 1s -> pi* transition) then
//! sit tens to hundreds of eV above the valence excitations and are extremely
//! hard to converge with an iterative solver, while the valence block of the
//! kernel is irrelevant for them.  The CVS approximation removes that
//! difficulty by restricting the excitation space to
//!
//! ```text
//!     (i, a) with i in {core orbitals},  a in {unoccupied orbitals}
//! ```
//!
//! and dropping the coupling between core and valence excitations.  The
//! screening `W` is still evaluated from the full (occupied) polarizability by
//! default, exactly as in a conventional CVS-BSE calculation; the alternative
//! `cvs_screening = "core"` builds it from the core orbitals alone and exists
//! as a diagnostic.
//!
//! This module is deliberately self-contained: it reuses the production
//! `v`/`W` builders ([`super::construct_coulomb`], [`super::construct_raw_w`],
//! [`super::reorganize_w`]) and the production Davidson solver, but it never
//! changes the dimension conventions of the ordinary BSE path.  The trigger is
//! `gw_or_bse = "cvs bse"`; the GW stage is prepared by
//! [`prepare_cvs_gw`] so that the core quasiparticle energies are computed
//! explicitly instead of being rigidly extrapolated from the valence window.

use crate::ctrl_io::quasiparticle_methods::QuasiParticle;
use crate::ri_gw;
use crate::scf_io::SCF;
use crate::solvers::davidson::{
    generate_initial_guess, lr_davidson_solver, tda_davidson_solver, DavidsonConfig,
};
use rest_tensors::matrix::matrix_blas_lapack::{_dgeev, _dgemm_full};
use rest_tensors::RIFull;
use tensors::{MathMatrix, MatrixFull};

use super::matvec::{
    coulomb_contribution, w_contribution_a_block_dgemm_standalone,
    w_contribution_b_block_dgemm_standalone,
};
use super::{
    construct_coulomb, construct_raw_w, reorganize_w, zip_and_sort, BseOutput,
};

/// Resolved CVS transition-space and solver parameters.
#[derive(Debug, Clone)]
pub struct CvsOptions {
    /// Core occupied orbital indices (absolute, `0`-based).
    pub core: Vec<usize>,
    /// First virtual orbital (the LUMO for a restricted reference).
    pub vir_start: usize,
    /// Number of virtual orbitals in the CVS space.
    pub vir_size: usize,
    /// `core.len()` (kept as a separate field for the matrix builders).
    pub occ_size: usize,
    /// Last orbital index included by `bse_cutoff_energy`.
    pub num_state: usize,
    /// `true` for TDA, `false` for the full (non-TDA) BSE.
    pub tda: bool,
    /// `'S'` (singlet) or `'T'` (triplet).
    pub spin: char,
    /// `"full"` or `"core"` screening.
    pub screening: String,
    /// Requested number of roots (`0` = solver default).
    pub nroots: usize,
    pub eigenvalue_min: f64,
    pub eigenvalue_max: f64,
}

impl CvsOptions {
    pub fn core_range(&self) -> std::ops::Range<usize> {
        // Core orbitals are the lowest occupied levels, hence contiguous.
        self.core[0]..(self.core[self.core.len() - 1] + 1)
    }
    pub fn vir_range(&self) -> std::ops::Range<usize> {
        self.vir_start..self.num_state
    }
    /// Dimension of the CVS transition space.
    pub fn dim(&self) -> usize {
        self.occ_size * self.vir_size
    }
}

fn qp_ctrl(scf_data: &SCF) -> QuasiParticle {
    scf_data
        .mol
        .ctrl
        .quasiparticle_methods
        .clone()
        .expect("CVS BSE requires [quasiparticle_methods]")
}

/// Resolve the core/virtual spaces and the remaining CVS controls from the
/// input card, validating the cases the implementation does not support.
pub fn resolve_options(scf_data: &SCF) -> CvsOptions {
    let qp = qp_ctrl(scf_data);
    if scf_data.mol.spin_channel != 1 {
        panic!(
            "CVS BSE currently supports only a restricted (closed-shell) reference; \
             spin_polarization = true is not implemented."
        );
    }
    if scf_data.mol.start_mo != 0 {
        panic!(
            "CVS BSE needs the core orbitals inside the GW/BSE orbital window, but \
             frozen_core_postscf = {} sets start_mo = {}.  Set frozen_core_postscf = 0 \
             for a CVS calculation.",
            scf_data.mol.ctrl.frozen_core_postscf, scf_data.mol.start_mo
        );
    }

    let (_start_mo, num_state, occ_size_all, _vir_size, _homo, lumo) =
        ri_gw::get_occupation_parameters(scf_data, 'N');
    if num_state <= lumo {
        panic!(
            "CVS BSE: bse_cutoff_energy = {} Ha leaves no virtual orbitals (lumo index {lumo}, \
             num_state {num_state}).  Raise bse_cutoff_energy.",
            qp.bse_cutoff_energy
        );
    }

    let energies = ri_gw::current_orbital_energies(scf_data);
    let core: Vec<usize> = if qp.cvs_ncore > 0 {
        if qp.cvs_ncore > occ_size_all {
            panic!(
                "CVS BSE: cvs_ncore = {} exceeds the number of occupied orbitals ({}).",
                qp.cvs_ncore, occ_size_all
            );
        }
        (0..qp.cvs_ncore).collect()
    } else {
        (0..occ_size_all)
            .filter(|&i| energies[i] < qp.cvs_core_energy_max)
            .collect()
    };
    if core.is_empty() {
        panic!(
            "CVS BSE: no core orbitals were found.  Set cvs_ncore explicitly, or lower \
             cvs_core_energy_max (currently {} Ha); the occupied energies are {:?}.",
            qp.cvs_core_energy_max,
            &energies[0..occ_size_all]
        );
    }
    // The RI range helpers expect a contiguous core window.  Auto-detection by
    // an energy threshold is contiguous for a physically ordered spectrum; make
    // the failure explicit rather than silently dropping a core level.
    for (expected, &actual) in core.iter().enumerate() {
        if expected != actual {
            panic!(
                "CVS BSE: the selected core orbitals are not a contiguous lowest-energy \
                 window (selected {:?}).  Set cvs_ncore = {} instead.",
                core,
                core.iter().filter(|&&i| i < actual).count() + 1
            );
        }
    }

    let spin = match qp.bse_spin.trim().to_lowercase().as_str() {
        "" | "none" => 'S',
        "singlet" | "s" => 'S',
        "triplet" | "t" => 'T',
        "both" => {
            panic!(
                "CVS BSE: bse_spin = \"both\" is not supported; run the singlet and triplet \
                 CVS calculations as two separate jobs."
            )
        }
        other => panic!("CVS BSE: invalid bse_spin = \"{other}\""),
    };
    let screening = qp.cvs_screening.trim().to_lowercase();
    if screening != "full" && screening != "core" {
        panic!("CVS BSE: cvs_screening must be \"full\" or \"core\", got \"{}\"", qp.cvs_screening);
    }

    let occ_size = core.len();
    CvsOptions {
        core,
        vir_start: lumo,
        vir_size: num_state - lumo,
        occ_size,
        num_state,
        tda: qp.bse_tda,
        spin,
        screening,
        nroots: qp.cvs_nroots,
        eigenvalue_min: qp.cvs_eigenvalue_min,
        eigenvalue_max: qp.cvs_eigenvalue_max,
    }
}

/// Make sure the GW stage computes the core quasiparticle energies.
///
/// The default `gw_scheme = "extrapolated"` only evaluates the self-energy in
/// an energy window around HOMO/LUMO; states below that window receive the
/// rigid valence shift.  For a CVS calculation the core levels are exactly the
/// states whose QP energies matter, so widen the occupied window to include
/// every selected core orbital.  This changes the *computed* set, never the
/// self-energy model, and is a no-op for schemes that already treat all states.
pub fn prepare_cvs_gw(scf_data: &mut SCF) {
    let opts = resolve_options(scf_data);
    let e = ri_gw::current_orbital_energies(scf_data);
    let (_, _, _, _, homo, _) = ri_gw::get_occupation_parameters(scf_data, 'N');
    let e_homo = e[homo];
    let e_core_min = opts
        .core
        .iter()
        .map(|&i| e[i])
        .fold(f64::INFINITY, f64::min);
    // A small margin keeps the boundary state strictly inside the filter.
    let needed_occ_window = (e_homo - e_core_min).abs() + 1.0e-3;

    let qp = qp_ctrl(scf_data);
    if qp.gw_scheme.trim().eq_ignore_ascii_case("extrapolated") {
        let mut new_qp = qp.clone();
        let mut changed = false;
        if new_qp.gw_extrapolate_occ_threshold < needed_occ_window {
            new_qp.gw_extrapolate_occ_threshold = needed_occ_window;
            changed = true;
        }
        // The virtual window is deliberately *not* widened: the historical
        // `vir_threshold` keeps the explicitly computed virtuals around the
        // gap, while the higher virtuals retain the rigid valence shift.  This
        // is the stable choice -- evaluating the self-energy of every high
        // virtual can hit a nearby pole and produce a spurious QP energy (the
        // existing static-fallback branch does exactly that for states with
        // |E_KS| > gw_switch_fallback_threshold).  Users who need a larger
        // explicitly computed virtual window can raise
        // `gw_extrapolate_vir_threshold` themselves.
        if changed {
            println!(
                "CVS BSE: widening the extrapolated-GW occupied window so the core levels are \
                 computed explicitly (occ_threshold = {:.6} Ha).",
                new_qp.gw_extrapolate_occ_threshold
            );
            scf_data.mol.ctrl.quasiparticle_methods = Some(new_qp);
        }
    }
    if qp.use_low_rank_contour
        && qp.selfenergy_state_range < (opts.num_state - opts.core[0])
    {
        println!(
            "CVS BSE: note that selfenergy_state_range = {} may be too small to cover the core \
             states in the low-rank real-axis grid scan; increase it if the core QP energies \
             look unreliable.",
            qp.selfenergy_state_range
        );
    }
}

/// RI three-centre tensor for an arbitrary occupied x virtual MO range.
///
/// This is the same construction as [`super::get_submatrix`], but the occupied
/// range is the CVS core window rather than `start_mo..homo+1`.  The BSE-specific
/// auxiliary basis is honoured when it has been prepared.
pub fn get_submatrix_range(
    scf_data: &SCF,
    row_range: std::ops::Range<usize>,
    col_range: std::ops::Range<usize>,
) -> MatrixFull<f64> {
    let use_bse_integrals = scf_data.ri3fn_bse.is_some() || scf_data.rimatr_bse.is_some();
    let vector: Vec<(RIFull<f64>, std::ops::Range<usize>, std::ops::Range<usize>)> =
        if use_bse_integrals {
            scf_data.generate_ri3mo_bse(row_range.clone(), col_range.clone())
        } else {
            scf_data.generate_ri3mo_rayon_for_multiple_times(row_range.clone(), col_range.clone())
        };
    if scf_data.mol.ctrl.print_level > 1 {
        println!(
            "CVS RI tensor: rows [{:?}], cols [{:?}], {} column(s), aux basis = {}",
            row_range,
            col_range,
            col_range.len() * row_range.len(),
            if use_bse_integrals { "BSE-specific" } else { "regular" }
        );
    }
    vector.into_iter().next().unwrap().0.into_matfull_i_jk()
}

/// Inverse dielectric for the CVS kernel.
///
/// `"full"` delegates to the production `construct_inverse_dielectric`, so the
/// screening is identical to the ordinary BSE.  `"core"` rebuilds the
/// polarizability using only the core occupied orbitals.
fn construct_cvs_inverse_dielectric(
    scf_data: &SCF,
    opts: &CvsOptions,
    epsilon: &Vec<f64>,
) -> MatrixFull<f64> {
    if opts.screening == "full" {
        return super::construct_inverse_dielectric(scf_data, epsilon);
    }
    let epsilon = super::bse_screening_energies(scf_data, epsilon);
    let ri_ov_core = get_submatrix_range(scf_data, opts.core_range(), opts.vir_range());
    let mut eps = Vec::with_capacity(opts.occ_size + opts.vir_size);
    for &i in opts.core.iter() {
        eps.push(epsilon[i]);
    }
    for a in opts.vir_start..opts.num_state {
        eps.push(epsilon[a]);
    }
    let response = ri_gw::response_matrix(
        &eps,
        opts.occ_size,
        opts.vir_size,
        &ri_ov_core,
        0.0,
        'R',
        0.0,
    );
    ri_gw::inverse_dielectric_matrix(response, 'R')
}

/// CVS energy denominators `E_a - E_i` in the `i + a*ncore` layout.
fn energy_diagonal(qp_energies: &[f64], opts: &CvsOptions) -> Vec<f64> {
    let mut diag = vec![0.0; opts.dim()];
    for a in 0..opts.vir_size {
        for (ii, &i) in opts.core.iter().enumerate() {
            diag[ii + a * opts.occ_size] = qp_energies[opts.vir_start + a] - qp_energies[i];
        }
    }
    diag
}

/// The dense CVS `A` and (for non-TDA) `B` blocks.
///
/// Layout: `(i, a) -> i + a*ncore`, identical to the ordinary BSE.  The kernel
/// is built from the core-core, core-virtual and virtual-virtual RI tensors, so
/// the direct screened term is `W_{(i,a),(j,b)} = (ij|W|ab)` restricted to core
/// `i,j`.
fn build_cvs_blocks(
    scf_data: &SCF,
    opts: &CvsOptions,
    inverse_dielectric: &MatrixFull<f64>,
    qp_energies: &[f64],
    exchange_rescaling: f64,
) -> (MatrixFull<f64>, Option<MatrixFull<f64>>) {
    let dim = opts.dim();
    let ri_ov = get_submatrix_range(scf_data, opts.core_range(), opts.vir_range());
    let ri_oo = get_submatrix_range(scf_data, opts.core_range(), opts.core_range());
    let ri_vv = get_submatrix_range(scf_data, opts.vir_range(), opts.vir_range());

    let mut v = construct_coulomb(&ri_ov, &ri_ov);
    let raw_w_a = construct_raw_w(&ri_oo, &ri_vv, inverse_dielectric);
    let mut w_a = reorganize_w(raw_w_a, 'A', opts.occ_size, opts.vir_size);
    // Match the production matvec (`w_contribution_a_block_dgemm_standalone`),
    // which applies `bse_exchange_rescaling`; the ordinary dense BSE ignores
    // it, but for CVS we keep dense and matrix-free Davidson consistent.
    w_a.self_multiple(-exchange_rescaling);

    let mut a = w_a;
    let diag = energy_diagonal(qp_energies, opts);
    for k in 0..dim {
        a[[k, k]] += diag[k];
    }
    if opts.spin == 'S' {
        v.self_multiple(2.0);
        a = MatrixFull::add(&v, &a).unwrap();
    } else if opts.spin == 'R' {
        a = MatrixFull::add(&v, &a).unwrap();
    }

    if opts.tda {
        return (a, None);
    }

    let raw_w_b = construct_raw_w(&ri_ov, &ri_ov, inverse_dielectric);
    let mut w_b = reorganize_w(raw_w_b, 'B', opts.occ_size, opts.vir_size);
    // The production B-block matvec does not apply `bse_exchange_rescaling`
    // (only the A block does), so neither does the dense CVS B block.
    w_b.self_multiple(-1.0);
    let mut b = w_b;
    // `v` already carries the singlet/triplet prefactor from above.
    if opts.spin == 'S' || opts.spin == 'R' {
        b = MatrixFull::add(&v, &b).unwrap();
    }
    (a, Some(b))
}

/// Matrix-free CVS `A`/`B` operators for the Davidson solver.
///
/// The tensors follow exactly the layout used by the historical MO-basis BSE
/// matvec (`ri_bse::matvec`), but the occupied dimension is the CVS core
/// count.  Keeping the operators implicit lets a CVS calculation use a large
/// virtual space without materialising the dense `dim x dim` kernel.
struct CvsMatvec {
    ri_ov: MatrixFull<f64>,
    ri_vv: MatrixFull<f64>,
    ri_oo_tilde: MatrixFull<f64>,
    ri_ov_b: MatrixFull<f64>,
    ri_ov_tilde: MatrixFull<f64>,
    diag: Vec<f64>,
    spin: char,
    ncore: usize,
    nvir: usize,
}

impl CvsMatvec {
    fn build(
        scf_data: &SCF,
        opts: &CvsOptions,
        inverse_dielectric: &MatrixFull<f64>,
        qp_energies: &[f64],
    ) -> Self {
        let num_auxbas = inverse_dielectric.size[0];
        let (ncore, nvir) = (opts.occ_size, opts.vir_size);

        let ri_ov = get_submatrix_range(scf_data, opts.core_range(), opts.vir_range());
        let mut ri_vv = get_submatrix_range(scf_data, opts.vir_range(), opts.vir_range());
        ri_vv.reshape([num_auxbas * nvir, nvir]);

        let ri_oo = get_submatrix_range(scf_data, opts.core_range(), opts.core_range());
        let mut ri_oo_tilde = MatrixFull::new(ri_oo.size, 0.0);
        _dgemm_full(inverse_dielectric, 'N', &ri_oo, 'N', &mut ri_oo_tilde, 1.0, 0.0);
        ri_oo_tilde.reshape([num_auxbas * ncore, ncore]);
        ri_oo_tilde = ri_oo_tilde.transpose_and_drop();
        ri_oo_tilde.reshape([ncore * num_auxbas, ncore]);

        let mut ri_ov_tilde = MatrixFull::new(ri_ov.size, 0.0);
        _dgemm_full(inverse_dielectric, 'N', &ri_ov, 'N', &mut ri_ov_tilde, 1.0, 0.0);
        ri_ov_tilde.reshape([num_auxbas * ncore, nvir]);

        let mut ri_ov_b = ri_ov.clone();
        ri_ov_b.reshape([num_auxbas * ncore, nvir]);

        Self {
            ri_ov,
            ri_vv,
            ri_oo_tilde,
            ri_ov_b,
            ri_ov_tilde,
            diag: energy_diagonal(qp_energies, opts),
            spin: opts.spin,
            ncore,
            nvir,
        }
    }

    fn prefactor(&self) -> f64 {
        if self.spin == 'S' {
            2.0
        } else if self.spin == 'R' {
            1.0
        } else {
            0.0
        }
    }

    fn a(&self, qp: &QuasiParticle, z: &Vec<f64>) -> Vec<f64> {
        let w = w_contribution_a_block_dgemm_standalone(
            &self.ri_vv, z, &self.ri_oo_tilde, qp, self.ncore, self.nvir);
        let mut out = vec![0.0; z.len()];
        for k in 0..z.len() {
            out[k] = self.diag[k] * z[k] - w[k];
        }
        if self.prefactor() > 0.0 {
            let v = coulomb_contribution(&self.ri_ov, z);
            for k in 0..z.len() {
                out[k] += self.prefactor() * v[k];
            }
        }
        out
    }

    fn b(&self, z: &Vec<f64>) -> Vec<f64> {
        let w = w_contribution_b_block_dgemm_standalone(
            &self.ri_ov_b, z, &self.ri_ov_tilde, self.ncore, self.nvir);
        let mut out = vec![0.0; z.len()];
        for k in 0..z.len() {
            out[k] = -w[k];
        }
        if self.prefactor() > 0.0 {
            let v = coulomb_contribution(&self.ri_ov, z);
            for k in 0..z.len() {
                out[k] += self.prefactor() * v[k];
            }
        }
        out
    }
}

fn solve_dense(
    a: &MatrixFull<f64>,
    b: Option<&MatrixFull<f64>>,
    opts: &CvsOptions,
) -> Vec<(f64, Vec<f64>)> {
    let nov = opts.dim();
    let mut eigenpairs = if opts.tda {
        let (_vr, wr, _wi, _vl, vr, _info) = _dgeev(a, 'N', 'V');
        zip_and_sort(&wr, &vr)
    } else {
        let b = b.expect("non-TDA CVS needs the B block");
        let mut h = MatrixFull::new([2 * nov, 2 * nov], 0.0);
        for i in 0..nov {
            for j in 0..nov {
                h[[i, j]] = a[[i, j]];
                h[[nov + i, j]] = -b[[i, j]];
                h[[i, nov + j]] = b[[j, i]];
                h[[nov + i, nov + j]] = -a[[j, i]];
            }
        }
        let (_vr, wr, _wi, _vl, vr, _info) = _dgeev(&h, 'N', 'V');
        zip_and_sort(&wr, &vr)
    };
    eigenpairs.retain(|(e, _)| {
        *e >= opts.eigenvalue_min && *e <= opts.eigenvalue_max && e.is_finite()
    });
    if opts.nroots > 0 && eigenpairs.len() > opts.nroots {
        eigenpairs.truncate(opts.nroots);
    }
    eigenpairs
}

fn solve_davidson(
    matvec: &CvsMatvec,
    opts: &CvsOptions,
    qp: &QuasiParticle,
    diag: &Vec<f64>,
) -> Vec<(f64, Vec<f64>)> {
    let nroots = if opts.nroots > 0 {
        opts.nroots
    } else {
        qp.davidson_target_excitations.max(1)
    };
    let initial = generate_initial_guess(diag, nroots);
    let cfg = DavidsonConfig {
        max_subspace: qp.davidson_maximum_subspace_size,
        add_dim: qp.davidson_add_dimensions,
        restart_dim: qp.davidson_restart_dimensions,
        max_iter: qp.davidson_max_iter,
        tol: qp.davidson_converge_threshold,
        ..Default::default()
    };
    let mut eigenpairs = if opts.tda {
        tda_davidson_solver(
            |z: &Vec<f64>| matvec.a(qp, z),
            nroots,
            diag,
            initial,
            &cfg,
        )
    } else {
        lr_davidson_solver(
            |z: &Vec<f64>| matvec.a(qp, z),
            |z: &Vec<f64>| matvec.b(z),
            nroots,
            diag,
            initial,
            &cfg,
        )
    };
    eigenpairs.retain(|(e, _)| {
        *e >= opts.eigenvalue_min && *e <= opts.eigenvalue_max && e.is_finite()
    });
    eigenpairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    eigenpairs
}

fn print_transition_space(opts: &CvsOptions, scf_data: &SCF, qp: &QuasiParticle) {
    let e = ri_gw::current_orbital_energies(scf_data);
    println!("=========================================================");
    println!("CVS-BSE: core-valence separated Bethe-Salpeter calculation");
    println!("=========================================================");
    println!(
        "  spin channel            : {}",
        if opts.spin == 'S' { "singlet" } else { "triplet" }
    );
    println!(
        "  TDA                     : {}",
        if opts.tda { "on" } else { "off (full BSE)" }
    );
    println!("  screening               : {} occupied space", opts.screening);
    print!("  core orbitals           : ");
    for (k, &i) in opts.core.iter().enumerate() {
        if k > 0 {
            print!(", ");
        }
        print!("#{} (E_KS = {:.6} Ha)", i, e[i]);
    }
    println!();
    println!(
        "  virtual orbitals        : #{}..#{} ({} orbitals, bse_cutoff_energy = {} Ha)",
        opts.vir_start,
        opts.num_state - 1,
        opts.vir_size,
        qp.bse_cutoff_energy
    );
    println!(
        "  transition space        : {} core x {} virtual = {} (i + a*ncore layout)",
        opts.occ_size, opts.vir_size, opts.dim()
    );
    println!("  quasiparticle energies used for the diagonal: GW");
    print!("    core: ");
    for (k, &i) in opts.core.iter().enumerate() {
        if k > 0 {
            print!(", ");
        }
        print!("E_QP(#{} ) = {:.6} Ha", i, scf_data.gwqp.0[i]);
    }
    println!();
}

/// Print the leading core -> virtual components of one CVS eigenvector.
fn print_cvs_composition(
    eigenvector: &[f64],
    opts: &CvsOptions,
    tda: bool,
    n_components: usize,
) {
    let nov = opts.dim();
    let mut weighted: Vec<(usize, usize, f64)> = Vec::with_capacity(nov);
    for a in 0..opts.vir_size {
        for (ii, &i) in opts.core.iter().enumerate() {
            let idx = ii + a * opts.occ_size;
            let x = eigenvector.get(idx).copied().unwrap_or(0.0);
            let y = if tda {
                0.0
            } else {
                eigenvector.get(nov + idx).copied().unwrap_or(0.0)
            };
            // X^2 - Y^2 is the (signed) weight of the particle-hole component.
            weighted.push((ii, a, x * x - y * y));
        }
    }
    weighted.sort_by(|x, y| y.2.abs().partial_cmp(&x.2.abs()).unwrap_or(std::cmp::Ordering::Equal));
    print!("      core -> virtual (weight x^2-y^2): ");
    for (rank, (ii, a, w)) in weighted.iter().take(n_components).enumerate() {
        if rank > 0 {
            print!(", ");
        }
        print!(
            "#{} -> #{} ({:+.4})",
            opts.core[*ii],
            opts.vir_start + *a,
            w
        );
    }
    println!();
}

/// Main entry point for `gw_or_bse = "cvs bse"`.
///
/// Requires the GW quasiparticle energies in `scf_data.gwqp.0` and the BSE
/// auxiliary integrals (prepared by `prepare_bse_integrals`) to be in place.
pub fn cvs_bse_main(scf_data: &mut SCF) -> BseOutput {
    let opts = resolve_options(scf_data);
    let qp = qp_ctrl(scf_data);
    print_transition_space(&opts, scf_data, &qp);

    if scf_data.gwqp.0.len() < opts.num_state {
        panic!(
            "CVS BSE: the GW quasiparticle spectrum has {} entries but the CVS space needs {} \
             (bse_cutoff_energy = {} Ha).  Run GW for the full orbital window or raise the \
             cutoff handling.",
            scf_data.gwqp.0.len(),
            opts.num_state,
            qp.bse_cutoff_energy
        );
    }

    let mut epsilon = scf_data.eigenvalues[0].clone();
    if qp.bse_qp_polarization {
        epsilon = scf_data.gwqp.0.clone();
    }
    let inverse_dielectric = construct_cvs_inverse_dielectric(scf_data, &opts, &epsilon);
    println!(
        "  inverse dielectric      : {} x {}",
        inverse_dielectric.size[0], inverse_dielectric.size[1]
    );

    let qp_energies = scf_data.gwqp.0.clone();
    let diag = energy_diagonal(&qp_energies, &opts);

    let use_davidson = qp.bse_davidson_solver;
    let excitations = if use_davidson {
        println!(
            "CVS BSE: solving the {} x {} problem with matrix-free Davidson",
            opts.dim(),
            opts.dim()
        );
        let matvec = CvsMatvec::build(scf_data, &opts, &inverse_dielectric, &qp_energies);
        solve_davidson(&matvec, &opts, &qp, &diag)
    } else {
        println!("CVS BSE: solving the {} x {} problem densely", opts.dim(), opts.dim());
        let (a, b) = build_cvs_blocks(
            scf_data,
            &opts,
            &inverse_dielectric,
            &qp_energies,
            qp.bse_exchange_rescaling,
        );
        solve_dense(&a, b.as_ref(), &opts)
    };

    if excitations.is_empty() {
        println!(
            "CVS BSE: no roots survived the eigenvalue window [{}, {}] Ha.  Widen \
             cvs_eigenvalue_min/max.",
            opts.eigenvalue_min, opts.eigenvalue_max
        );
        return BseOutput {
            first_excitation: None,
            excitation_energies: Vec::new(),
        };
    }

    let dipole_matrix = super::dipoles::compute_dipole_matrix_cvs(
        scf_data,
        &opts.core,
        opts.vir_start,
        opts.vir_size,
        opts.spin == 'T',
    );

    println!(
        "\nCVS BSE results: {} root(s), spin = {}, TDA = {}",
        excitations.len(),
        if opts.spin == 'S' { "singlet" } else { "triplet" },
        opts.tda
    );
    for (k, (energy, vector)) in excitations.iter().enumerate() {
        let vec_norm: f64 = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        // Keep the historical `#n Excitation energy=...` layout so that the
        // regression harness's log scraper recognises CVS roots as BSE roots.
        println!("#{} Excitation energy={:.12}, norm={:.6}  ({:.6} eV)",
                 k, energy, vec_norm, energy * crate::constants::EV);
        let v = super::dipoles::normalize(vector, opts.tda);
        let dipole_square =
            super::dipoles::transition_dipole_square(&dipole_matrix, &v, opts.tda);
        println!(
            "\tTransition Dipole Square: {}; Oscillator Strength: {}",
            dipole_square,
            dipole_square * energy * 2.0 / 3.0
        );
        print_cvs_composition(vector, &opts, opts.tda, 5);
    }
    println!(
        "\nThe first excitation obtained by BSE is {:.12}  (CVS core excitation, {:.6} eV)",
        excitations[0].0,
        excitations[0].0 * crate::constants::EV
    );

    let spin_name = if opts.spin == 'S' { "singlet" } else { "triplet" };
    let rs_orbitals = qp.renormalized_singles && qp.rs_use_rs_orbitals;
    if qp.save_bse_terms {
        super::save_bse_amplitudes(
            &excitations,
            opts.occ_size,
            opts.vir_size,
            opts.tda,
            spin_name,
            rs_orbitals,
            "bse_terms.dat",
        );
    }
    if qp.save_bse_excitations {
        use std::io::Write;
        let line = excitations
            .iter()
            .map(|(n, _)| n.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open("bse_excitations.txt");
        writeln!(file.expect("write failure"), "{}", line).ok();
    }
    if qp.save_first_excitation {
        use std::io::Write;
        let save_path = qp.save_first_excitation_path.clone();
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(save_path);
        writeln!(file.expect("write failure"), "{}", excitations[0].0).ok();
    }

    BseOutput {
        first_excitation: Some(excitations[0].0),
        excitation_energies: excitations.iter().map(|(e, _)| *e).collect(),
    }
}
