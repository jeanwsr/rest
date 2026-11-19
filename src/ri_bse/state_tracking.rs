//! Rotation-invariant excited-state tracking in the AO basis.
//!
//! The BSE amplitudes `(X, Y)` live in the active occupied-virtual MO space and
//! are therefore tied to the MO gauge.  A separate rotation of the occupied and
//! virtual orbitals,
//!
//! ```text
//! C_o -> C_o U_o^T,   C_v -> C_v U_v^T,
//! X   -> U_o X U_v^T, Y -> U_o Y U_v^T,
//! ```
//!
//! leaves the physical transition density
//!
//! ```text
//! T_{mu nu} = sum_{ia} (X_{ia} + Y_{ia}) C_{mu i} C_{nu a}
//! ```
//!
//! unchanged.  Comparing `T` between two geometries (through the cross-geometry
//! AO overlap `S_{mu nu} = <mu(R0)|nu(R1)>`) therefore identifies an electronic
//! state without depending on the MO gauge, the root sign, or the arbitrary
//! rotation inside a degenerate BSE manifold.
//!
//! For a near-degenerate group the individual roots are only defined up to a
//! unitary rotation.  [`project_onto_subspace`] projects the previous state onto
//! the new group in the AO metric; the result is the state of maximal overlap
//! (the one-state case of the subspace SVD/polar matching used in surface
//! hopping).

use rest_tensors::matrix::matrix_blas_lapack::{_dgemm_full, _dsyevd};
use tensors::{BasicMatrix, MathMatrix, MatrixFull};

/// Build the AO-basis transition density of one BSE/TDA root.
///
/// `c` is the `[nao, nmo]` MO coefficient matrix (column-major, as stored in
/// `SCF::eigenvectors`).  The active occupied orbitals are
/// `occ_start..occ_start+nocc`, the active virtuals `vir_start..vir_start+nvir`.
/// The amplitude layout is REST's `i + a*nocc`.
pub fn transition_density_ao(
    c: &MatrixFull<f64>,
    nao: usize,
    x: &[f64],
    y: &[f64],
    nocc: usize,
    nvir: usize,
    occ_start: usize,
    vir_start: usize,
) -> Vec<f64> {
    assert!(
        x.len() >= nocc * nvir,
        "transition_density_ao: amplitude vector {} shorter than {}",
        x.len(),
        nocc * nvir
    );
    let mut amp = vec![0.0f64; nocc * nvir];
    for a in 0..nvir {
        for i in 0..nocc {
            let k = i + a * nocc;
            amp[i + a * nocc] = x[k] + y.get(k).copied().unwrap_or(0.0);
        }
    }
    let amp = MatrixFull::from_vec([nocc, nvir], amp).unwrap();
    let mut co = MatrixFull::new([nao, nocc], 0.0);
    let mut cv = MatrixFull::new([nao, nvir], 0.0);
    for i in 0..nocc {
        for mu in 0..nao {
            co.data[mu + nao * i] = c[[mu, occ_start + i]];
        }
    }
    for a in 0..nvir {
        for mu in 0..nao {
            cv.data[mu + nao * a] = c[[mu, vir_start + a]];
        }
    }
    // T = C_o (X+Y) C_v^T
    let mut tmp = MatrixFull::new([nao, nvir], 0.0);
    _dgemm_full(&co, 'N', &amp, 'N', &mut tmp, 1.0, 0.0);
    let mut t = MatrixFull::new([nao, nao], 0.0);
    _dgemm_full(&tmp, 'N', &cv, 'T', &mut t, 1.0, 0.0);
    t.data
}

/// Convert a row-major `[nao, nao]` overlap matrix into the column-major layout
/// used by [`MatrixFull`] and by the transition densities in this module.
pub fn row_major_to_col(s: &[f64], nao: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; nao * nao];
    for mu in 0..nao {
        for nu in 0..nao {
            out[mu + nao * nu] = s[mu * nao + nu];
        }
    }
    out
}

fn dense(data: &[f64], n: usize) -> MatrixFull<f64> {
    MatrixFull::from_vec([n, n], data.to_vec()).unwrap()
}

/// Bilinear AO overlap `Tr(T0^T S T1 S^T)` of two transition densities.
///
/// With `S = <mu(R0)|nu(R1)>` this is the overlap of the two transition
/// densities across geometries.  All matrices are column-major.
pub fn cross_overlap(t0: &[f64], t1: &[f64], s: &[f64], nao: usize) -> f64 {
    let sm = dense(s, nao);
    let t1m = dense(t1, nao);
    let mut st = MatrixFull::new([nao, nao], 0.0);
    _dgemm_full(&sm, 'N', &t1m, 'N', &mut st, 1.0, 0.0);
    let mut a = MatrixFull::new([nao, nao], 0.0);
    _dgemm_full(&st, 'N', &sm, 'T', &mut a, 1.0, 0.0);
    t0.iter().zip(a.data.iter()).map(|(p, q)| p * q).sum()
}

/// Normalise a transition density in the AO metric `S`: return
/// `sqrt(Tr(T^T S T S^T))` and divide `t` by it.  Returns `0.0` (and leaves the
/// vector untouched) when the density is numerically zero.
pub fn normalize_density(t: &mut [f64], s: &[f64], nao: usize) -> f64 {
    let n2 = cross_overlap(t, t, s, nao);
    if n2 <= 1.0e-28 {
        return 0.0;
    }
    let n = n2.sqrt();
    let inv = 1.0 / n;
    for v in t.iter_mut() {
        *v *= inv;
    }
    n
}

/// Project `old` (transition density at the previous geometry) onto the span of
/// `new_ts` (the roots of a degenerate manifold at the current geometry).
///
/// Solves the metric least-squares problem
/// `G c = o`, `G_jk = <T_j|T_k>_{S_new}`, `o_j = <T_old|T_j>_{S_cross}`,
/// diagonalising `G` (so a linearly dependent solver basis is handled by a
/// pseudo-inverse).  Returns the normalised projected density and its
/// coefficients.  For an orthonormal group with a single member this reduces to
/// the ordinary normalised overlap.
pub fn project_onto_subspace(
    old: &[f64],
    new_ts: &[Vec<f64>],
    s_cross: &[f64],
    s_new: &[f64],
    nao: usize,
) -> (Vec<f64>, Vec<f64>) {
    let g = new_ts.len();
    if g == 0 {
        return (old.to_vec(), Vec::new());
    }
    if g == 1 {
        let o = cross_overlap(old, &new_ts[0], s_cross, nao);
        let sign = if o < 0.0 { -1.0 } else { 1.0 };
        return (new_ts[0].iter().map(|v| sign * v).collect(), vec![sign]);
    }
    let mut gram = vec![0.0f64; g * g];
    let mut rhs = vec![0.0f64; g];
    for j in 0..g {
        rhs[j] = cross_overlap(old, &new_ts[j], s_cross, nao);
        for k in 0..g {
            gram[j + k * g] = cross_overlap(&new_ts[j], &new_ts[k], s_new, nao);
        }
    }
    let (vecs, vals, _) = _dsyevd(&dense(&gram, g), 'V');
    let vecs = vecs.expect("state_tracking: subspace metric eigenvectors");
    let vmax = vals.iter().cloned().fold(0.0f64, f64::max);
    let thr = (1.0e-10 * vmax).max(1.0e-14);
    let mut coeff = vec![0.0f64; g];
    for k in 0..g {
        if vals[k] <= thr {
            continue;
        }
        let mut vk = 0.0f64;
        for j in 0..g {
            vk += vecs.data[j + k * g] * rhs[j];
        }
        vk /= vals[k];
        for j in 0..g {
            coeff[j] += vecs.data[j + k * g] * vk;
        }
    }
    // If the previous state has no projection onto the group, fall back to the
    // largest single overlap rather than returning a zero density.
    let mut norm2 = 0.0f64;
    for j in 0..g {
        for k in 0..g {
            norm2 += coeff[j] * gram[j + k * g] * coeff[k];
        }
    }
    if norm2 <= 1.0e-20 {
        let mut best = 0usize;
        let mut best_o = f64::NEG_INFINITY;
        for j in 0..g {
            if rhs[j].abs() > best_o {
                best_o = rhs[j].abs();
                best = j;
            }
        }
        let sign = if rhs[best] < 0.0 { -1.0 } else { 1.0 };
        return (new_ts[best].iter().map(|v| sign * v).collect(), {
            let mut c = vec![0.0f64; g];
            c[best] = sign;
            c
        });
    }
    let inv = 1.0 / norm2.sqrt();
    for c in coeff.iter_mut() {
        *c *= inv;
    }
    let mut out = vec![0.0f64; nao * nao];
    for j in 0..g {
        for p in 0..out.len() {
            out[p] += coeff[j] * new_ts[j][p];
        }
    }
    // Keep the overall sign aligned with the previous step.
    if cross_overlap(old, &out, s_cross, nao) < 0.0 {
        for v in out.iter_mut() {
            *v = -*v;
        }
        for c in coeff.iter_mut() {
            *c = -*c;
        }
    }
    (out, coeff)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rot2(theta: f64) -> [[f64; 2]; 2] {
        let (s, c) = theta.sin_cos();
        [[c, -s], [s, c]]
    }

    /// Apply C_o -> C_o U_o^T, C_v -> C_v U_v^T and X -> U_o X U_v^T.
    fn rotate(
        c: &MatrixFull<f64>,
        nao: usize,
        x: &[f64],
        nocc: usize,
        nvir: usize,
        occ_start: usize,
        vir_start: usize,
        uo: &[[f64; 2]; 2],
        uv: &[[f64; 2]; 2],
    ) -> (MatrixFull<f64>, Vec<f64>) {
        let mut c2 = c.clone();
        for mu in 0..nao {
            let a = c[[mu, occ_start]];
            let b = c[[mu, occ_start + 1]];
            // C_o' = C_o U_o^T
            c2.data[mu + nao * occ_start] = a * uo[0][0] + b * uo[0][1];
            c2.data[mu + nao * (occ_start + 1)] = a * uo[1][0] + b * uo[1][1];
            let a = c[[mu, vir_start]];
            let b = c[[mu, vir_start + 1]];
            // C_v' = C_v U_v^T
            c2.data[mu + nao * vir_start] = a * uv[0][0] + b * uv[0][1];
            c2.data[mu + nao * (vir_start + 1)] = a * uv[1][0] + b * uv[1][1];
        }
        // X'_{ia} = sum_{j,b} U_o[i,j] X_{jb} U_v[a,b]
        let mut x2 = vec![0.0f64; nocc * nvir];
        for i in 0..nocc {
            for a in 0..nvir {
                let mut v = 0.0f64;
                for j in 0..nocc {
                    for b in 0..nvir {
                        v += uo[i][j] * x[j + b * nocc] * uv[a][b];
                    }
                }
                x2[i + a * nocc] = v;
            }
        }
        (c2, x2)
    }

    #[test]
    fn transition_density_is_occ_vir_rotation_invariant() {
        let nao = 6;
        let nocc = 2;
        let nvir = 2;
        let occ_start = 0;
        let vir_start = 2;
        let nmo = 6;
        let mut c = MatrixFull::new([nao, nmo], 0.0);
        for mu in 0..nao {
            for p in 0..nmo {
                c.data[mu + nao * p] = ((mu + 1) as f64 * 0.37 + (p + 1) as f64 * 0.11).sin();
            }
        }
        let x: Vec<f64> = (0..nocc * nvir).map(|k| 0.3 + 0.1 * k as f64).collect();
        let y = vec![0.0f64; nocc * nvir];
        let t0 = transition_density_ao(&c, nao, &x, &y, nocc, nvir, occ_start, vir_start);
        let (c2, x2) = rotate(
            &c, nao, &x, nocc, nvir, occ_start, vir_start, &rot2(0.7), &rot2(-0.4),
        );
        let t1 = transition_density_ao(&c2, nao, &x2, &y, nocc, nvir, occ_start, vir_start);
        let err = t0.iter().zip(t1.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(err < 1.0e-12, "rotation invariance violated: {err}");
    }

    #[test]
    fn normalized_overlap_is_one_for_identical_density() {
        let nao = 3;
        let s = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut t = vec![0.2, -0.3, 0.1, 0.4, 0.0, 0.5, -0.2, 0.1, 0.3];
        let n = normalize_density(&mut t, &s, nao);
        assert!((n - (n * n).sqrt()).abs() < 1.0e-12);
        let o = cross_overlap(&t, &t, &s, nao);
        assert!((o - 1.0).abs() < 1.0e-12, "self overlap {o}");
    }

    #[test]
    fn subspace_projection_recovers_a_linear_combination() {
        let nao = 3;
        let s = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        // Two orthonormal (under S=I) densities.
        let mut t0 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut t1 = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        normalize_density(&mut t0, &s, nao);
        normalize_density(&mut t1, &s, nao);
        let old: Vec<f64> = t0.iter().zip(t1.iter()).map(|(a, b)| 0.6 * a + 0.8 * b).collect();
        let (proj, coeff) = project_onto_subspace(&old, &[t0, t1], &s, &s, nao);
        let err = proj.iter().zip(old.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(err < 1.0e-12, "projection error {err}, coeff {coeff:?}");
    }
}
