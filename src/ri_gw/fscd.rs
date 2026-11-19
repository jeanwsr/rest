//! Frequency-sampled contour deformation (FSCD) for molecular G0W0.
//!
//! In contour-deformation (CD) GW the correlation self-energy is split into an
//! imaginary-axis integral and a sum over the poles of `G` that enter the
//! deformed contour,
//!
//! ```text
//!   Sigma_c(w) = -(1/2pi) Int G(w+iw') W(iw') dw'
//!                + Sum_p  s_p * pole_factor_p * W_np(|w - eps_p|) .
//! ```
//!
//! The residue frequencies `|w - eps_p|` depend on the trial quasiparticle
//! energy `w`.  A graphical/Newton root search therefore rebuilds the response
//! matrix -- and inverts it -- once per pole and per trial `w`.  For a state
//! far from the gap (a core level, or any state in a wide spectrum) the pole
//! count grows with the system size, and the repeated response builds dominate
//! the calculation.
//!
//! FSCD samples the *exact* residue frequencies once for the states that will
//! be computed, removes frequencies that are closer than a tolerance to an
//! already kept grid point, evaluates `W_np` only on that sparse grid, and
//! interpolates `W_np` at all other residue frequencies.  This follows the
//! frequency-sampling idea of TURBOMOLE's `fscdgrd` option: the expensive
//! response is built `n_grid` times instead of once per pole and per
//! root-search iteration.  The remaining frequencies are obtained by
//! linear/cubic interpolation here (TURBOMOLE continues `W` with a Padé fit);
//! a one-point exact-CD guard in the driver catches the rare case where the
//! interpolation error moves a quasiparticle root.
//!
//! The imaginary-axis term is untouched: it is still evaluated from the
//! regular `W_c(iw)` grid produced by [`super::generate_w_c`].

use std::sync::atomic::{AtomicUsize, Ordering};

use tensors::{matrix_blas_lapack::_dgemm_full, MathMatrix, MatrixFull};

use super::{inverse_dielectric_matrix, response_matrix};

/// Number of response matrices built by [`build_fscd_table`].  Used by the
/// regression/performance tests; production runs only pay one relaxed atomic
/// increment per grid point.
pub static FSCD_RESPONSE_BUILDS: AtomicUsize = AtomicUsize::new(0);

/// Reset the FSCD response-build counter.
pub fn reset_fscd_counter() {
    FSCD_RESPONSE_BUILDS.store(0, Ordering::Relaxed);
}

/// Read the FSCD response-build counter.
pub fn fscd_response_builds() -> usize {
    FSCD_RESPONSE_BUILDS.load(Ordering::Relaxed)
}

/// How `W_np(|w-eps_p|)` is obtained between two sampled frequencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FscdInterp {
    /// Linear interpolation between the bracketing grid points (default).
    Linear,
    /// Four-point cubic Lagrange interpolation (useful for convergence tests).
    Cubic,
}

impl FscdInterp {
    pub fn from_str(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "cubic" | "lagrange" => Self::Cubic,
            _ => Self::Linear,
        }
    }
}

/// Numerical controls for the frequency sampling.
#[derive(Debug, Clone)]
pub struct FscdConfig {
    /// Absolute merge threshold in Hartree.  Two candidate frequencies closer
    /// than this are represented by one grid point.  The default `0.05 Ha`
    /// (1.361 eV) is TURBOMOLE's `thrs1` default.
    pub domega: f64,
    /// Relative merge threshold.  Default `1e-3` (0.1%, TURBOMOLE `thrs2`).
    pub rel_tol: f64,
    /// Lorentzian broadening used when the real-axis response is built.
    pub eta: f64,
    /// Numerical tolerance of the CD pole test (`de >= -res_tol`).
    pub res_tol: f64,
    /// Half-width of the quasiparticle root-search window (Hartree).  The
    /// candidate frequencies are expanded by `+/- span` so that the grid also
    /// covers the trial energies visited by the root finder.
    pub span: f64,
    /// Interpolation rule between grid points.
    pub interp: FscdInterp,
    /// Optional **geometry-independent** frequency grid.
    ///
    /// The default grid is built from the current orbital energies
    /// (`|eps_p - eps_n| + ladder`), i.e. it moves with the geometry.  That is
    /// harmless for a QP-energy run but it makes the analytic gradient
    /// inconsistent: `d/dR` of the interpolant then also carries the motion of
    /// the nodes, which the gradient pullback does not model.  When this is
    /// `Some`, the grid is used verbatim and no residue-based construction
    /// happens; the driver fills it with exactly the pinned lattice the
    /// gradient engine uses (`build_real_axis_grid` with `omega_chi_max > 0`,
    /// whose `node_sens` is zero), so energy and gradient share one grid.
    pub freqs: Option<Vec<f64>>,
}

impl Default for FscdConfig {
    fn default() -> Self {
        Self {
            domega: 0.05,
            rel_tol: 1.0e-3,
            eta: 1.0e-3,
            res_tol: 1.0e-3,
            span: 0.2,
            interp: FscdInterp::Linear,
            freqs: None,
        }
    }
}

/// Sparse frequency grid plus the sampled `W_np` values.
///
/// `w[i][p][k]` is `W_{n,p}(freqs[k])` for target orbital `targets[i]` and
/// pole `p`.  For an occupied target the pole index is the occupied orbital
/// index (`0..occ_size`); for a virtual target it is the local virtual index
/// (`0..vir_size`, global orbital `occ_size+p`), matching the column layout of
/// `ri_row_n` used by [`super::contour_rayon`].
pub struct FscdTable {
    pub freqs: Vec<f64>,
    pub w: Vec<Vec<Vec<f64>>>,
    pub targets: Vec<usize>,
    pub occ_size: usize,
    pub vir_size: usize,
    pub interp: FscdInterp,
    /// Largest residue frequency represented by the table (Hartree).
    pub max_freq: f64,
    /// Number of response matrices built while constructing the table.
    pub n_response_builds: usize,
}

impl FscdTable {
    pub fn n_grid(&self) -> usize {
        self.freqs.len()
    }

    /// Number of `(target, pole)` pairs whose `W` is tabulated.
    pub fn n_pairs(&self) -> usize {
        self.w.iter().map(|row| row.len()).sum()
    }
}

/// Is `n` an occupied orbital in the CD pole convention?
#[inline]
fn is_occupied_target(n: usize, occ_size: usize) -> bool {
    n < occ_size
}

/// Build the FSCD frequency grid from the exact residue frequencies.
///
/// The candidates are `|eps_p - eps_n|` for every target `n` and every pole
/// `p` on the same side of the gap, expanded by a uniform ladder over
/// `[-span, +span]` so that the grid covers the trial energies rather than only
/// the initial Kohn-Sham energies.  The ladder has at most 201 points; its
/// spacing is `2*span/n_steps`, i.e. `domega` when that is affordable and the
/// cap spacing otherwise.  Frequencies closer than `domega` (absolute) or
/// `rel_tol` (relative) to an already kept point are discarded.  Zero is
/// always kept.
pub fn build_frequency_grid(
    targets: &[usize],
    qp_g: &[f64],
    occ_size: usize,
    vir_size: usize,
    cfg: &FscdConfig,
) -> Vec<f64> {
    // Pinned grid (gradient-capable): use it verbatim.
    if let Some(f) = cfg.freqs.as_ref() {
        if f.len() >= 2 {
            return f.clone();
        }
    }

    let mut candidates: Vec<f64> = Vec::new();
    candidates.push(0.0);

    // Offsets around each exact residue frequency.  The uniform ladder spans
    // the whole root-search + relaxation window; when `domega` is smaller than
    // the 201-point cap spacing, the cap sets the effective resolution so that
    // the table stays bounded.
    let span = cfg.span.max(1.0e-8);
    let step = cfg.domega.max(1.0e-8);
    // Uniform ladder around each exact frequency.  The ladder is capped at
    // 201 points; for a threshold finer than `2*span/200` the effective
    // spacing is that cap (never coarser than the requested `domega` would
    // allow for a moderate window).
    let n_steps = ((2.0 * span / step).ceil() as usize).clamp(1, 200);
    let offsets: Vec<f64> = (0..=n_steps)
        .map(|k| -span + 2.0 * span * (k as f64) / (n_steps as f64))
        .collect();
    for &n in targets {
        let occupied = is_occupied_target(n, occ_size);
        let n_poles = if occupied { occ_size } else { vir_size };
        for p in 0..n_poles {
            let pole = if occupied { p } else { occ_size + p };
            let de0 = (qp_g[pole] - qp_g[n]).abs();
            for &offset in &offsets {
                let f = de0 + offset;
                if f > 0.0 && f.is_finite() {
                    candidates.push(f);
                }
            }
        }
    }

    candidates.sort_by(|a, b| a.total_cmp(b));
    // A small margin keeps the clamped endpoint from being hit by the final
    // QP energy when it sits just outside the initial candidate range.
    let max_candidate = *candidates.last().unwrap_or(&0.0);
    let margin = 1.1_f64.max(1.0 + cfg.span / max_candidate.max(1.0e-8));
    let max_freq = (max_candidate * margin).max(1.0e-8);

    let mut grid: Vec<f64> = Vec::with_capacity(candidates.len());
    for f in candidates {
        if f > max_freq {
            continue;
        }
        if let Some(&last) = grid.last() {
            let diff = f - last;
            let absolute_close = diff < cfg.domega;
            let relative_close = diff <= cfg.rel_tol * f.abs().max(1.0e-12);
            if absolute_close || relative_close {
                continue;
            }
        }
        grid.push(f);
    }

    // Interpolation needs at least one interval.
    if grid.len() < 2 {
        grid.clear();
        grid.push(0.0);
        grid.push(max_freq);
    }

    grid
}

/// Build the sampled `W_np` table.
///
/// * `ri_ov` is the occupied-virtual RI tensor used by the CD response.
/// * `ri_rows[i]` is `(Q|n m)` for target orbital `targets[i]`.
/// * `qp_g` / `qp_w` are the Green's-function and screened-interaction energy
///   vectors used by the current GW pass.
#[allow(clippy::too_many_arguments)]
pub fn build_fscd_table(
    ri_ov: &MatrixFull<f64>,
    ri_rows: &[MatrixFull<f64>],
    targets: &[usize],
    qp_g: &Vec<f64>,
    qp_w: &Vec<f64>,
    occ_size: usize,
    vir_size: usize,
    cfg: &FscdConfig,
) -> FscdTable {
    assert_eq!(
        ri_rows.len(),
        targets.len(),
        "FSCD: one RI row per target orbital is required"
    );

    let freqs = build_frequency_grid(targets, qp_g, occ_size, vir_size, cfg);
    let n_grid = freqs.len();
    let n_aux = ri_ov.size[0];

    // Allocate w[i][p][k].
    let mut w: Vec<Vec<Vec<f64>>> = Vec::with_capacity(targets.len());
    for &n in targets {
        let occupied = is_occupied_target(n, occ_size);
        let n_poles = if occupied { occ_size } else { vir_size };
        let mut row = Vec::with_capacity(n_poles);
        for _ in 0..n_poles {
            row.push(vec![0.0; n_grid]);
        }
        w.push(row);
    }

    for (k, &freq) in freqs.iter().enumerate() {
        FSCD_RESPONSE_BUILDS.fetch_add(1, Ordering::Relaxed);
        let response = response_matrix(qp_w, occ_size, vir_size, ri_ov, freq, 'C', cfg.eta);
        let inverse_dielectric = inverse_dielectric_matrix(response, 'C');

        for (i, row) in ri_rows.iter().enumerate() {
            let [n_row_aux, n_mo] = row.size;
            debug_assert_eq!(n_row_aux, n_aux);
            let mut u = MatrixFull::new([n_aux, n_mo], 0.0);
            _dgemm_full(&inverse_dielectric, 'N', row, 'N', &mut u, 1.0, 0.0);

            let n = targets[i];
            let occupied = is_occupied_target(n, occ_size);
            let n_poles = if occupied { occ_size } else { vir_size };
            for p in 0..n_poles {
                let col = if occupied { p } else { occ_size + p };
                let dot: f64 = row
                    .iter_column(col)
                    .zip(u.iter_column(col))
                    .map(|(a, b)| a * b)
                    .sum();
                w[i][p][k] = dot;
            }
        }
    }

    FscdTable {
        max_freq: *freqs.last().unwrap_or(&0.0),
        freqs,
        w,
        targets: targets.to_vec(),
        occ_size,
        vir_size,
        interp: cfg.interp,
        n_response_builds: n_grid,
    }
}

/// Interpolate `vals` (defined on the ascending `freqs` grid) at `x`.
///
/// Values outside the grid are clamped to the nearest endpoint.  This is the
/// only approximation in FSCD besides the frequency merge itself; the caller
/// can shrink `domega` to converge it.
pub fn interpolate(freqs: &[f64], vals: &[f64], x: f64, interp: FscdInterp) -> f64 {
    debug_assert_eq!(freqs.len(), vals.len());
    let n = freqs.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 || x <= freqs[0] {
        return vals[0];
    }
    if x >= freqs[n - 1] {
        return vals[n - 1];
    }

    let hi = freqs.partition_point(|&f| f <= x);
    let lo = hi.saturating_sub(1);
    let denom = freqs[hi] - freqs[lo];
    let t = if denom.abs() > 0.0 {
        (x - freqs[lo]) / denom
    } else {
        0.0
    };

    match interp {
        FscdInterp::Linear => vals[lo] + t * (vals[hi] - vals[lo]),
        FscdInterp::Cubic => {
            // Four-point Lagrange stencil.  Shift the stencil at the two
            // boundaries so that it always contains four *distinct* nodes.
            if n < 4 {
                return vals[lo] + t * (vals[hi] - vals[lo]);
            }
            let start = lo.saturating_sub(1).min(n - 4);
            let i0 = start;
            let i1 = start + 1;
            let i2 = start + 2;
            let i3 = start + 3;
            let nodes = [freqs[i0], freqs[i1], freqs[i2], freqs[i3]];
            let values = [vals[i0], vals[i1], vals[i2], vals[i3]];
            let mut result = 0.0;
            for a in 0..4 {
                let mut basis = 1.0;
                for b in 0..4 {
                    if a != b {
                        basis *= (x - nodes[b]) / (nodes[a] - nodes[b]);
                    }
                }
                result += values[a] * basis;
            }
            result
        }
    }
}

/// Residue (contour) part of the FSCD self-energy for one target orbital.
///
/// This is the direct analogue of [`super::contour_rayon`], with `W_np` read
/// from the sampled table instead of being rebuilt at every pole.
pub fn contour_fscd(
    table: &FscdTable,
    target_index: usize,
    omega: f64,
    qp_g: &[f64],
    res_tol: f64,
) -> f64 {
    let n = table.targets[target_index];
    let occ_size = table.occ_size;
    let vir_size = table.vir_size;
    let fermi_energy = 0.5 * (qp_g[occ_size - 1] + qp_g[occ_size]);
    let sign = if omega > fermi_energy { 1.0 } else { -1.0 };

    let occupied = is_occupied_target(n, occ_size);
    let n_poles = if occupied { occ_size } else { vir_size };
    let mut sum = 0.0;

    for p in 0..n_poles {
        let (de, pole) = if occupied {
            (qp_g[p] - omega, p)
        } else {
            (omega - qp_g[occ_size + p], occ_size + p)
        };
        if de >= -res_tol {
            let pole_factor = if de.abs() < res_tol { 0.5 } else { 1.0 };
            let value = interpolate(
                &table.freqs,
                &table.w[target_index][p],
                de.abs(),
                table.interp,
            );
            sum += pole_factor * value;
        }
    }

    sign * sum
}

/// FSCD quasiparticle equation: `consts + Sigma_c(omega) - omega`.
///
/// The imaginary-axis piece is identical to the CD path and still uses the
/// regular `W_c(iw)` grid; only the residue piece is frequency-sampled.
#[allow(clippy::too_many_arguments)]
pub fn quasiparticle_equation_fscd(
    table: &FscdTable,
    target_index: usize,
    omega: f64,
    n: usize,
    consts: f64,
    qp_g: &Vec<f64>,
    qp_w: &Vec<f64>,
    num_state: usize,
    w_c_at_freqs: &Vec<(f64, f64, MatrixFull<f64>)>,
    res_tol: f64,
) -> f64 {
    let contour = contour_fscd(table, target_index, omega, qp_g, res_tol);
    let imag = super::calculate_imag(w_c_at_freqs, num_state, n, omega, qp_g, qp_w);
    consts + contour - imag - omega
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frequency_grid_merges_close_points_and_keeps_zero() {
        let targets = [0usize];
        // Three occupied poles, giving exact residue frequencies 0, 0.5, 1.0
        // (and a 0.01 companion that must merge into 0).
        let qp_g = [-2.0, -1.99, -1.5, -1.0, 0.0, 1.0];
        let cfg = FscdConfig {
            domega: 0.05,
            rel_tol: 0.0,
            span: 1.0,
            ..Default::default()
        };
        let grid = build_frequency_grid(&targets, &qp_g, 4, 2, &cfg);
        assert_eq!(grid[0], 0.0);
        // 0.01 merges into 0; 0.5 and 1.0 survive.
        assert!(grid.windows(2).all(|w| w[1] - w[0] >= 0.05 - 1.0e-14));
        assert!(grid.iter().any(|f| (*f - 0.5).abs() < 1.0e-12));
        assert!(grid.iter().any(|f| (*f - 1.0).abs() < 1.0e-12));
    }

    #[test]
    fn linear_interpolation_is_exact_on_grid_and_between_nodes() {
        let freqs = [0.0, 1.0, 2.0];
        let vals = [0.0, 2.0, 4.0];
        for (x, expected) in [(0.0, 0.0), (0.5, 1.0), (1.0, 2.0), (1.5, 3.0)] {
            let got = interpolate(&freqs, &vals, x, FscdInterp::Linear);
            assert!((got - expected).abs() < 1.0e-14);
        }
        // Outside the grid is clamped.
        assert!((interpolate(&freqs, &vals, -1.0, FscdInterp::Linear) - 0.0).abs() < 1.0e-14);
        assert!((interpolate(&freqs, &vals, 3.0, FscdInterp::Linear) - 4.0).abs() < 1.0e-14);
    }

    #[test]
    fn cubic_interpolation_reproduces_a_cubic_polynomial() {
        let freqs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let vals: Vec<f64> = freqs.iter().map(|x| x * x * x - 2.0 * x + 1.0).collect();
        for x in [0.25, 1.5, 2.75, 3.5] {
            let expected = x * x * x - 2.0 * x + 1.0;
            let got = interpolate(&freqs, &vals, x, FscdInterp::Cubic);
            assert!((got - expected).abs() < 1.0e-10, "x={x} got={got} expected={expected}");
        }
    }
}
