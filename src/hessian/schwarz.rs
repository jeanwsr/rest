//! Schwarz screening for the direct (4-center) Hessian ej_ek backend.
//!
//! Mirrors NWChem's three-level screening used in `twodd_coul_ex.F`:
//!   1. Shell-pair pre-screen:  s_ij * sch_max * q4max < tol2e  → skip (ish,jsh)
//!   2. Quartet pre-screen:     s_ijkl * q4max < tol2e          → skip (ksh,lsh)
//!   3. Density-aware screen:   s_ijkl * psum * scale <= tol2e  → skip quartet
//!
//! The Schwarz bound for shell pair (ish,jsh) is
//!   s_ij = sqrt(sqrt( Σ_{a,b,c,d} (ab|cd)² ))
//! i.e. the square root of the Frobenius norm of the (ish,jsh|ish,jsh)
//! integral block — a rotationally invariant upper bound on
//! sqrt(max |(μν|μν)|).  This matches NWChem `schwarz_init.F`.

use rest_libcint::CINTR2CDATA;

/// Default integral screening threshold (NWChem default `tol2e`).
pub const SCHWARZ_TOL2E: f64 = 1.0e-7;

/// Conservative pre-screening factor:
///   8  — max permutation scale factor
///   1  — no molecular symmetry (sym_number_ops = 0)
///   10000 — conservative upper bound on |psum| (2PDM Frobenius norm)
pub const SCHWARZ_Q4MAX: f64 = 8.0 * 1.0 * 10000.0;

/// Build the Schwarz shell-pair screening matrix.
///
/// Returns a symmetric `[nsh*nsh]` matrix (row-major) and the global maximum.
pub fn build_schwarz_shell(cint: &CINTR2CDATA, nsh: usize) -> (Vec<f64>, f64) {
    let mut schwarz = vec![0.0f64; nsh * nsh];
    let mut sch_max = 0.0f64;

    for ish in 0..nsh {
        for jsh in 0..=ish {
            let slc: &[[usize; 2]] = &[
                [ish, ish + 1],
                [jsh, jsh + 1],
                [ish, ish + 1],
                [jsh, jsh + 1],
            ];
            let (buf, _): (Vec<f64>, Vec<usize>) =
                cint.integrate_row_major("int2e", "s1", Some(slc)).into();
            let mut sum_sq = 0.0f64;
            for &v in &buf {
                sum_sq += v * v;
            }
            let s = if sum_sq > 0.0 { sum_sq.sqrt().sqrt() } else { 0.0 };
            schwarz[ish * nsh + jsh] = s;
            schwarz[jsh * nsh + ish] = s;
            if s > sch_max {
                sch_max = s;
            }
        }
    }
    (schwarz, sch_max)
}

/// Compute the Frobenius norm of the 2PDM block for shell quartet
/// (μ∈ish, ν∈jsh, λ∈ksh, σ∈lsh), matching NWChem `grad_make_twopdm`.
///
/// 2PDM(μ,ν,λ,σ) = 0.5·factor_j·D(μ,ν)·D(λ,σ)
///                − 0.125·factor_k·(D(μ,λ)·D(ν,σ) + D(μ,σ)·D(ν,λ))
///
/// `ao_off` = [ao_loc[ish], ao_loc[jsh], ao_loc[ksh], ao_loc[lsh]]
/// `ds`     = [di, dj, dk, dl]
#[allow(clippy::too_many_arguments)]
pub fn compute_psum(
    dm0: &[f64],
    nao: usize,
    ao_off: [usize; 4],
    ds: [usize; 4],
    factor_j: f64,
    factor_k: f64,
    with_k: bool,
) -> f64 {
    let d0 = ds[0];
    let d1 = ds[1];
    let d2 = ds[2];
    let d3 = ds[3];
    let i0 = ao_off[0];
    let j0 = ao_off[1];
    let k0 = ao_off[2];
    let l0 = ao_off[3];

    let mut psum_sq = 0.0f64;
    for a in 0..d0 {
        let mu = i0 + a;
        for b in 0..d1 {
            let nu = j0 + b;
            let d_munu = dm0[mu + nu * nao];
            for c in 0..d2 {
                let la = k0 + c;
                for dd in 0..d3 {
                    let si = l0 + dd;
                    let d_lasi = dm0[la + si * nao];
                    let mut pdm = 0.5 * factor_j * d_munu * d_lasi;
                    if with_k {
                        let d_mula = dm0[mu + la * nao];
                        let d_nusi = dm0[nu + si * nao];
                        let d_musi = dm0[mu + si * nao];
                        let d_nula = dm0[nu + la * nao];
                        pdm -= 0.125 * factor_k * (d_mula * d_nusi + d_musi * d_nula);
                    }
                    psum_sq += pdm * pdm;
                }
            }
        }
    }
    psum_sq.sqrt()
}
