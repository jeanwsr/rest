//! SMD CDS (Cavitation-Dispersion-Solvent structure) energy and gradient.
//!
//! Translated from Fortran source: https://github.com/nwchemgit/nwchem/blob/master/src/solvation/mnsol.F
//!
//! # Physical Model
//!
//! The CDS model computes the non-electrostatic contribution to solvation free energy:
//!
//! ```text
//! E_CDS = Σ_k A_k · (σ_k + cssigm) · 0.001   [kcal/mol]
//! ```
//!
//! where:
//! - `A_k` = SASA (solvent-accessible surface area) of atom k, in Å²
//! - `σ_k` = effective atomic surface tension of atom k, in cal/mol/Å²
//! - `cssigm` = molecular-level surface tension correction, in cal/mol/Å²
//! - `0.001` converts cal/mol → kcal/mol
//!
//! ## Two-Stage Computation
//!
//! 1. **Surface tension assignment** ([`smx_cds`]): Each atom gets an effective surface tension
//!    `σ_k = σ⁰(Z_k) + Σ_bonds COT(r; R₀, δ) · Δσ`, where `σ⁰` is the zeroth-order element
//!    value and bond corrections depend on the local chemical environment (H-X, O-X, N-C,
//!    C-C, C-N bond types).
//!
//! 2. **SASA computation** ([`dareal`]): The accessible solid angle Ω_k of each atomic sphere
//!    is computed via the DAREAL algorithm (Liotard, 1992), then `A_k = Ω_k · R_k²`.
//!
//! ## Key Quantities
//!
//! | Quantity | Physical meaning | Units |
//! |----------|-----------------|-------|
//! | `BONDI[Z]` | Bondi vdW radius of element Z | Å |
//! | `rad[k]` | Effective radius = BONDI[Z_k] + 0.4 (probe) | Å |
//! | `rkkval(t1,t2)` | Sum of covalent radii for SMD type pair = expected bond length R₀ | Å |
//! | `sigma[0..150]` | Zeroth-order atomic surface tension σ⁰ | cal/mol/Å² |
//! | `hsigma[0..150]` | H-atom bond correction Δσ_HZ | cal/mol/Å² |
//! | `cssigm` | Molecular surface tension (solvent-wide) | cal/mol/Å² |
//! | `SIGMA_MOL` | Coefficients for cssigm: [c_γ, c_β², c_φ², c_ψ²] | — |
//! | `COT(r; R₀, δ)` | Smooth bond-detection weight ∈ [0,1], C¹-continuous | dimensionless |
//! | `sts[k]` | Effective surface tension σ_k^eff | cal/mol/Å² |
//! | `dsts[dir]` | ∂σ_k/∂X_{iat,dir} as [nat, nat] matrix per direction | cal/(mol·Å³) |
//! | `area0` (Ω_k) | Accessible solid angle of sphere K | steradians (sr) |
//! | `area_atom[k]` (A_k) | SASA = Ω_k · R_k² | Å² |
//! | `datar[dir]` | ∂A_k/∂X_{iat,dir} as [nat, nat] matrix per direction | Å |
//!
//! ## Sigma Index Convention
//!
//! Indices 0..102 hold element-specific zeroth-order values σ⁰(Z).
//! Indices 101–150 hold bond-type correction values Δσ:
//!
//! | Index | Bond type | Δσ in water (cal/mol/Å²) |
//! |-------|-----------|--------------------------|
//! | 101 | C–C single bond | −72.95 |
//! | 103 | O–C bond | 68.69 |
//! | 104 | O–O bond | 0.00 |
//! | 105 | N–C coordination | −48.22 |
//! | 106 | O–N bond | 121.98 |
//! | 110 | C–N bond | 0.00 |
//! | 114 | O–P bond | 68.85 |
//! | 116 | N–C(3) triple bond | 84.10 |
//!
//! ## Unit Conversions
//!
//! - `TO_ANGS = BOHR` Bohr → Å
//! - `TO_KCAL = HARTREE2KCAL` Hartree → kcal/mol
//! - Internal computation in Å and kcal/mol; public API input/output in Bohr and Hartree.
//!
//! ## References
//!
//! - Marenich, Cramer, Truhlar, *JPCB* 2009, 113, 6378–6396 (SMD model)
//! - Liotard, D. (1992) — DAREAL accessible solid angle algorithm
//! - Rinaldi, D. & Liotard, D. — analytical derivatives

use std::f64::consts::PI;
use tensors::MatrixFull;
use crate::constants::solvent as data;
use crate::constants::{BOHR, HARTREE2KCAL};
use super::surface_utils::SmdCavityRadii;

// ============================================================================
//  Debug printing helper (controlled by env var REST_CDS_DEBUG=1)
// ============================================================================

fn cds_debug_enabled() -> bool {
    std::env::var("REST_CDS_DEBUG").map_or(false, |v| v == "1")
}

/// Print a labeled f64 scalar.
fn cds_print_scalar(label: &str, val: f64) {
    println!("CDS_DEBUG| {} = {:.12e}", label, val);
}

/// Print a labeled Vec<f64> with one value per line.
fn cds_print_vec(label: &str, v: &[f64]) {
    println!("CDS_DEBUG| {} [len={}]", label, v.len());
    for (i, val) in v.iter().enumerate() {
        println!("CDS_DEBUG|   [{}] = {:.12e}", i, val);
    }
}

/// Print labeled [nat][3] gradient array.
fn cds_print_grad(label: &str, g: &[[f64; 3]]) {
    println!("CDS_DEBUG| {} [nat={}]", label, g.len());
    for (i, row) in g.iter().enumerate() {
        println!("CDS_DEBUG|   [{}] = {:.12e} {:.12e} {:.12e}", i, row[0], row[1], row[2]);
    }
}

/// Print non-zero entries of a [f64; 151] array.
fn cds_print_sigma151(label: &str, arr: &[f64; 151]) {
    println!("CDS_DEBUG| {} non-zero entries:", label);
    for (i, &v) in arr.iter().enumerate() {
        if v.abs() > 1e-10 {
            println!("CDS_DEBUG|   [{}] = {:.12e}", i, v);
        }
    }
}

// ============================================================================
//  1. Constants & Parameter Data
// ============================================================================

/// Bondi van der Waals radii for elements Z=0..102 (Å).
/// Used to construct solvent-accessible surface: `R_k = BONDI[Z] + r_probe` (r_probe = 0.4 Å).
/// Mantina et al., CRC Handbook 2010; H from Bondi 1964.
static BONDI: [f64; 103] = [
    0.00, 1.20, 1.40, 1.82, 1.53, 1.92, 1.70, 1.55, 1.52, 1.47,
    1.54, 2.27, 1.73, 1.84, 2.10, 1.80, 1.80, 1.75, 1.88, 2.75,
    2.31, 2.16, 1.87, 1.79, 1.89, 1.97, 1.94, 1.92, 1.84, 1.86,
    2.10, 1.87, 2.11, 1.85, 1.90, 1.85, 2.02, 3.03, 2.49, 2.19,
    1.86, 2.07, 2.09, 2.09, 2.07, 1.95, 2.02, 2.03, 2.30, 1.93,
    2.17, 2.06, 2.06, 1.98, 2.16,
    // 55..102: not used by SMD, set to 0
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
];

/// Maps atomic number Z → SMD atom type.
/// SMD groups 103 elements into 11 types sharing covalent radii and bond-correction parameters.
/// Types: 1=H, 2=C, 3=N, 4=O, 5=F, 6=S, 7=Cl, 8=Br, 9=P, 10=I, 11=Si. 0 = unparameterized.
const NATCNV_TABLE: [(usize, usize); 11] = [
    (1,1), (6,2), (7,3), (8,4), (9,5), (14,11), (15,9), (16,6), (17,7), (35,8), (53,10),
];
fn natcnv(z: usize) -> usize {
    for &(zz, t) in &NATCNV_TABLE {
        if zz == z { return t; }
    }
    0
}

/// Sum of covalent radii for SMD type pair (Å), a symmetric 11×11 matrix.
/// Used as the expected bond length `R₀` in the COT bond-detection function:
/// - `r ≈ rkkval` → COT ≈ 1 (bonded)
/// - `r ≥ rkkval + δ` → COT = 0 (not bonded)
///
/// Typical values: H–C=1.55, C–C=1.84, C–N=1.84, C–O=1.84, O–O=2.75.
fn rkkval(itpc: usize, jtpc: usize) -> f64 {
    if itpc == 0 || jtpc == 0 || itpc > 11 || jtpc > 11 { return 0.0; }
    let m = [
        [0.00, 1.55, 1.55, 1.55, 0.00, 2.14, 0.00, 0.00, 0.00, 0.00, 0.00],
        [1.55, 1.84, 1.84, 1.84, 1.84, 2.20, 2.10, 2.30, 2.20, 2.60, 0.00],
        [1.55, 1.84, 1.85, 1.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [1.55, 1.84, 1.50, 2.75, 0.00, 1.71, 0.00, 0.00, 2.10, 0.00, 2.10],
        [0.00, 1.84, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [2.14, 2.20, 0.00, 1.71, 0.00, 2.75, 0.00, 0.00, 2.50, 0.00, 0.00],
        [0.00, 2.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 2.30, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 2.20, 0.00, 2.10, 0.00, 2.50, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 2.60, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 2.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    ];
    m[itpc - 1][jtpc - 1]
}

// ---- Sigma arrays for water (ICDS=1) ----
// sigma[Z] = zeroth-order atomic surface tension (cal/mol/Å²).
// hsigma[Z] = H-bond correction: H atom surface tension change when bonded to heavy atom Z.
// Only specific elements (H,C,N,O,F,P,S,Br,I) and bond-type indices 101–150 have non-zero values.

/// Zeroth-order atomic surface tensions for water, σ⁰(Z), in cal/mol/Å².
const SIGMA_AQ: [f64; 151] = {
    let mut s = [0.0f64; 151];
    // Zeroth-order σ⁰(Z) for elements
    s[1]=48.69; s[6]=129.74; s[9]=38.18; s[16]=-9.10; s[17]=9.82;
    s[35]=-8.72;
    // Bond-type corrections Δσ (indices 101–150, matching Fortran sigma array)
    s[101]=-72.95;  // C–C single bond
    s[103]=68.69;   // O–C bond
    s[105]=-48.22;  // N–C coordination
    s[106]=121.98;  // O–N bond
    s[114]=68.85;   // O–P bond
    s[116]=84.10;   // N–C(3) triple bond
    s
};
/// H-atom bond corrections for water, Δσ_HZ, in cal/mol/Å².
/// Only HSIGMA_AQ[6] = −60.77 is non-zero (H–C bond).
const HSIGMA_AQ: [f64; 151] = {
    let mut s = [0.0f64; 151];
    s[6] = -60.77;
    s
};

// ---- Sigma arrays for non-aqueous (ICDS=2) ----
// Non-aqueous σ is a linear combination of three basis functions:
//   σ(Z) = σ_N(Z)·n + σ_A(Z)·α + σ_B(Z)·β
// where n=refractive index, α=H-bond acidity, β=H-bond basicity.
// HSIGMA_N is the refractive-index-dependent H-correction (HSIGMA_A/B are all zero).

trait SigmaData { fn fill(&self, arr: &mut [f64; 151]); }
impl SigmaData for [f64; 151] {
    fn fill(&self, arr: &mut [f64; 151]) { arr.copy_from_slice(self); }
}

const SIGMA_N_DATA: [f64; 151] = build_sigma_n();
const SIGMA_A_DATA: [f64; 151] = build_sigma_a();
const SIGMA_B_DATA: [f64; 151] = build_sigma_b();
const HSIGMA_N_DATA: [f64; 151] = build_hsigma_n();

const fn build_sigma_n() -> [f64; 151] {
    let mut s = [0.0f64; 151];
    s[6]=58.10; s[7]=32.62; s[8]=-17.56; s[14]=-18.04; s[16]=-33.17;
    s[17]=-24.31; s[35]=-35.42; s[101]=-62.05; s[103]=-15.70; s[110]=-99.76;
    s
}
const fn build_sigma_a() -> [f64; 151] {
    let mut s = [0.0f64; 151];
    s[6]=48.10; s[8]=193.06; s[103]=95.99; s[105]=-41.00; s[110]=152.20;
    s
}
const fn build_sigma_b() -> [f64; 151] {
    let mut s = [0.0f64; 151];
    s[6]=32.87; s[8]=-43.79; s[104]=-128.16; s[106]=79.13;
    s
}
const fn build_hsigma_n() -> [f64; 151] {
    let mut s = [0.0f64; 151];
    s[6]=-36.37; s[8]=-19.39;
    s
}

/// Molecular-level surface tension coefficients: [c_γ, c_β², c_φ², c_ψ²].
///
/// ```text
/// cssigm = c_γ·γ + c_β²·β² + c_φ²·φ² + c_ψ²·ψ²
///        = 0.35·γ − 4.19·φ² − 6.68·ψ²   (β² term is zero)
/// ```
///
/// where γ=macroscopic surface tension, φ=aromatic carbon fraction, ψ=halogen fraction.
/// This is a solvent-wide contribution, independent of individual atomic environments.
/// For water, cssigm = 0 (already absorbed into the sigma arrays).
const SIGMA_MOL: [f64; 4] = [0.35, 0.00, -4.19, -6.68];

// ============================================================================
//  2. Helper Functions
// ============================================================================

/// Upper-triangular index for compressed pairwise storage.
/// Compresses a symmetric matrix (e.g. r_ij = r_ji) into a 1D array of size nat·(nat+1)/2.
#[inline(always)]
fn ij0(i: usize, j: usize) -> usize {
    if i > j { i * (i + 1) / 2 + j } else { j * (j + 1) / 2 + i }
}

/// 3-vector dot product, used for computing cos(θ) = û₁·û₂ in DAREAL.
#[inline(always)]
fn dot3(x: &[f64], y: &[f64]) -> f64 {
    x[0] * y[0] + x[1] * y[1] + x[2] * y[2]
}

/// 3-vector cross product, used in DAREAL free-intersection computation
/// to construct the direction orthogonal to two SS great-circle planes.
#[inline(always)]
fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
}

/// Copy N doubles (Fortran DCOPY).
#[inline(always)]
fn dcopy_n(n: usize, src: &[f64], dst: &mut [f64]) { dst[..n].copy_from_slice(&src[..n]); }

/// Scale N doubles in-place (Fortran DSCAL).
#[inline(always)]
fn dscal_n(n: usize, a: f64, x: &mut [f64]) {
    for v in &mut x[..n] { *v *= a; }
}

// ============================================================================
//  3. Solvent-Specific Sigma Assignment
// ============================================================================

/// Assign water sigma parameters.
/// Copies `SIGMA_AQ` → `sigma`, `HSIGMA_AQ` → `hsigma`, returns cssigm = 0.
/// Water is the SMD reference solvent with its own parameterization.
fn smd_cds_aq(sigma: &mut [f64; 151], hsigma: &mut [f64; 151]) -> f64 {
    sigma.copy_from_slice(&SIGMA_AQ);
    hsigma.copy_from_slice(&HSIGMA_AQ);
    0.0 // CSSIGM = 0 for water
}

/// Assign non-aqueous sigma parameters as linear combinations of solvent descriptors.
///
/// ```text
/// σ(Z)   = σ_N(Z)·n + σ_A(Z)·α + σ_B(Z)·β
/// hσ(Z)  = hσ_N(Z)·n          (hσ_A, hσ_B are all zero)
/// cssigm = c_γ·γ + c_φ·φ² + c_ψ·ψ²   (see [`SIGMA_MOL`])
/// ```
///
/// The three basis sets σ_N, σ_A, σ_B capture the dependence on solvent
/// polarizability (n), H-bond acidity (α), and H-bond basicity (β) respectively.
fn smd_cds_naq(
    sigma: &mut [f64; 151], hsigma: &mut [f64; 151],
    sola: f64, solb: f64, solc: f64, solg: f64, solh: f64, soln: f64,
) -> f64 {
    for i in 0..151 {
        sigma[i] = SIGMA_N_DATA[i] * soln + SIGMA_A_DATA[i] * sola + SIGMA_B_DATA[i] * solb;
        hsigma[i] = HSIGMA_N_DATA[i] * soln;
    }
    SIGMA_MOL[0] * solg + SIGMA_MOL[1] * solb * solb
        + SIGMA_MOL[2] * solc * solc + SIGMA_MOL[3] * solh * solh
}

// ============================================================================
//  4. SMXCDS: Surface Tension Assignment
// ============================================================================

/// COT (Continuous Order of Terminal atoms) — smooth bond-detection weight function.
///
/// ```text
/// COT(r; R₀, δ) = exp(δ / (r − R₀ − δ))   for r < R₀ + δ
///                = 0                        for r ≥ R₀ + δ
/// ```
///
/// Output is a C∞-smooth weight in [0,1]:
/// - `r ≈ R₀` (covalent bond distance) → COT ≈ 1 (bonded)
/// - `r ≥ R₀ + δ` → COT = 0 (not bonded)
///
/// # Parameters
/// - `r`: actual interatomic distance (Å)
/// - `rhld` (R₀): expected bond length = sum of covalent radii from [`rkkval`] (Å)
/// - `deltar` (δ): transition width, typically 0.30 Å
///
/// Returns `(COT, dCOT/dr)`.
#[inline]
fn cot_val(r: f64, rhld: f64, deltar: f64) -> (f64, f64) {
    let cutoff = rhld + deltar;
    if r < cutoff {
        let expont = deltar / (r - cutoff);
        let c = f64::exp(expont);
        (c, -c * expont / (r - cutoff))
    } else {
        (0.0, 0.0)
    }
}

/// Compute effective atomic surface tensions with bond-environment corrections.
///
/// ```text
/// σ_k^eff = σ⁰(Z_k) + Σ_bonds COT(r_kj; R₀, δ) · Δσ_bond_type
/// ```
///
/// Five bond-correction branches are applied (SMD-only):
/// H–X, O–X, N–C (coordination-dependent), C–C, C–N. Bond detection via [`cot_val`].
///
/// # Returns
/// - `sts[nat]`: effective surface tensions σ_k^eff (cal/mol/Å²)
/// - `dsts[3]`: ∂σ_k/∂X_{iat,dir}, each a `[nat, nat]` matrix, `dsts[dir][[iat, k]]`
fn smx_cds(
    atomic_numbers: &[usize], sigma: &[f64; 151], hsigma: &[f64; 151],
    nat: usize, rlio: &[f64], urlio: &[MatrixFull<f64>; 3],
) -> (Vec<f64>, [MatrixFull<f64>; 3]) {
    let mut sts = vec![0.0f64; nat];
    let mut dsts = [
        MatrixFull::new([nat, nat], 0.0),
        MatrixFull::new([nat, nat], 0.0),
        MatrixFull::new([nat, nat], 0.0),
    ];

    // ---- zeroth-order: σ_k = σ⁰(Z_k) ----
    for i in 0..nat {
        let z = atomic_numbers[i];
        sts[i] = sigma[z.min(150)];
    }
    let sts_base = sts.clone(); // snapshot for debug

    /// Apply a bond correction to atom i with gradient propagation:
    ///   σ_i += COT(r_ij) · Δσ
    ///   ∂σ_i/∂X_i_dir += Δσ · dCOT/dr · û_{j→i}    (→ dsts[dir][[i, i]])
    ///   ∂σ_i/∂X_j_dir -= Δσ · dCOT/dr · û_{j→i}    (→ dsts[dir][[j, i]])
    fn add_bond_corr(
        sts: &mut [f64], dsts: &mut [MatrixFull<f64>; 3], nat: usize,
        i: usize, j: usize, sig_val: f64,
        rlio: &[f64], urlio: &[MatrixFull<f64>; 3], rhld: f64, deltar: f64,
    ) {
        let r = rlio[ij0(i, j)];
        let (c, dc) = cot_val(r, rhld, deltar);
        sts[i] += c * sig_val;
        let d = sig_val * dc;
        for dir in 0..3 {
            let u_ji = urlio[dir][[j, i]];   // û_{j→i} component
            dsts[dir][[i, i]] += d * u_ji;
            dsts[dir][[j, i]] -= d * u_ji;
        }
    }

    // ---- H–X correction: σ_H += Σ_J COT(r_HJ) · hsigma[Z_J] ----
    for i in 0..nat {
        if atomic_numbers[i] != 1 { continue; }
        let itpc = natcnv(1);
        for j in 0..nat {
            if i == j { continue; }
            let ntp = atomic_numbers[j];
            let jtpc = natcnv(ntp);
            if ntp >= 151 { continue; }
            let rhld = rkkval(itpc, jtpc);
            add_bond_corr(&mut sts, &mut dsts, nat, i, j, hsigma[ntp], rlio, urlio, rhld, 0.30);
        }
    }
    let sts_after_hx = sts.clone();

    // ---- O–X correction (C, N, P, O) ----
    // O–O uses hardcoded R₀=1.80 Å for peroxy bonds.
    for i in 0..nat {
        if atomic_numbers[i] != 8 { continue; }
        let itpc = natcnv(8);
        for j in 0..nat {
            if i == j { continue; }
            let ntp = atomic_numbers[j];
            let jtpc = natcnv(ntp);
            let (sig_val, rhld, deltar) = match ntp {
                // Fortran mnsol.F:834 — O–C pair overrides R₀=1.330, δ=0.10
                6 => (sigma[103], 1.330, 0.10),
                7 => (sigma[106], rkkval(itpc, jtpc), 0.30),
                15 => (sigma[114], rkkval(itpc, jtpc), 0.30),
                8 => (sigma[104], 1.80, 0.30),
                _ => continue,
            };
            add_bond_corr(&mut sts, &mut dsts, nat, i, j, sig_val, rlio, urlio, rhld, deltar);
        }
    }
    let sts_after_ox = sts.clone();

    // ---- N–C correction (coordination-dependent) ----
    // RTKKS = Σ_{C_J} COT(N,C_J) · (C_coord_J)²
    // σ_N += RTKKS^1.3 · sigma[105],  sigma[105] = −48.22 (water).
    //
    // Save per-N RTKKS for gradient block below (Fortran mnsol.F:1065–1142).
    let mut n_rtkks: Vec<(usize, f64)> = Vec::new(); // (N_atom_index, rtkk_s)
    for i in 0..nat {
        if atomic_numbers[i] != 7 { continue; }
        let mut rtkk_s = 0.0f64;
        for j in 0..nat {
            if i == j || atomic_numbers[j] != 6 { continue; }
            let (c_ij, _) = cot_val(rlio[ij0(i, j)], rkkval(natcnv(7), natcnv(6)), 0.30);
            if c_ij <= 0.0 { continue; }
            let mut rtkk3 = 0.0f64;
            for k in 0..nat {
                if k == i || k == j { continue; }
                let rhld2 = rkkval(natcnv(atomic_numbers[k]), natcnv(6));
                let (c_jk, _) = cot_val(rlio[ij0(j, k)], rhld2, 0.30);
                rtkk3 += c_jk;
            }
            rtkk_s += c_ij * rtkk3 * rtkk3;
        }
        sts[i] += rtkk_s.powf(1.3) * sigma[105];
        n_rtkks.push((i, rtkk_s));
    }
    let sts_after_nc = sts.clone();

    // ---- N–C coordination gradient (Fortran mnsol.F:1065–1142) ----
    // Chain rule through F = RTKKS^1.3 · sigma[105]:
    //   ∂F/∂X = sigma[105] · 1.3 · RTKKS^0.3 · ∂(RTKKS)/∂X
    //          = c0 · ∂(RTKKS)/∂X
    // where RTKKS = Σ_J COT_IJ · SCOTC²,  SCOTC = Σ_K COT_JK.
    //
    // Two contributions per (N=i, C=j, K=k) triple:
    //   A (N-C bond length):  dsts[i,i] += c0·SCOTC²·dCOT_ij·u_{C→N}
    //                         dsts[j,i] -= c0·SCOTC²·dCOT_ij·u_{C→N}
    //   B (C-K bond length):  dsts[j,i] += c0·2·SCOTC·COT_ij·dCOT_jk·u_{K→C}
    //                         dsts[k,i] -= c0·2·SCOTC·COT_ij·dCOT_jk·u_{K→C}
    //
    // N-C(2) sigma[111] term is skipped (not used by SMD).
    for &(i, rtkk_s) in &n_rtkks {
        if rtkk_s <= 0.0 { continue; }
        let c0 = sigma[105] * 1.3 * rtkk_s.powf(0.3); // [cal/(mol·Å²)]
        let rhld_nc = rkkval(natcnv(7), natcnv(6)); // R₀(N,C)

        // ---- T1 debug: checkpoint smx-cds.nc-grad.rtkks ----
        if cds_debug_enabled() {
            println!("CDS_DEBUG| NC-grad entry: N-atom i={} rtkk_s={:.12e} c0={:.12e} sigma[105]={:.12e}",
                i, rtkk_s, c0, sigma[105]);
        }

        for j in 0..nat {
            if i == j || atomic_numbers[j] != 6 { continue; }
            let (c_ij, dc_ij) = cot_val(rlio[ij0(i, j)], rhld_nc, 0.30);
            if c_ij <= 0.0 { continue; }

            // Recompute SCOTC = Σ_K COT(C_J-K) (same as RTKK3 in energy loop)
            let mut scotc: f64 = 0.0;
            for k in 0..nat {
                if k == i || k == j { continue; }
                let rhld2 = rkkval(natcnv(atomic_numbers[k]), natcnv(6));
                let (c_jk, _) = cot_val(rlio[ij0(j, k)], rhld2, 0.30);
                scotc += c_jk;
            }

            // ---- T1 debug: checkpoint smx-cds.nc-grad.scotc ----
            if cds_debug_enabled() {
                println!("CDS_DEBUG| NC-grad scotc: i={} j={} scotc={:.12e} c_ij={:.12e} dc_ij={:.12e} r_NC={:.6e}",
                    i, j, scotc, c_ij, dc_ij, rlio[ij0(i, j)]);
            }

            // --- Contribution A: N-C bond distance change (Fortran 1105-1112) ---
            let pref_a = c0 * scotc * scotc * dc_ij; // [cal/(mol·Å³)]
            // ---- T1 debug: checkpoint smx-cds.nc-grad.contrib-a ----
            if cds_debug_enabled() {
                let ux = urlio[0][[j, i]];
                let uy = urlio[1][[j, i]];
                let uz = urlio[2][[j, i]];
                println!("CDS_DEBUG| NC-grad A: i={} j={} pref_a={:.12e} u_C→N=({:.6e},{:.6e},{:.6e}) → dsts[N,N]+=({:.6e},{:.6e},{:.6e}) dsts[C,N]-=({:.6e},{:.6e},{:.6e})",
                    i, j, pref_a, ux, uy, uz,
                    pref_a*ux, pref_a*uy, pref_a*uz,
                    pref_a*ux, pref_a*uy, pref_a*uz);
            }
            for dir in 0..3 {
                let u_ji = urlio[dir][[j, i]]; // û_{C→N}
                dsts[dir][[i, i]] += pref_a * u_ji;
                dsts[dir][[j, i]] -= pref_a * u_ji;
            }

            // --- Contribution B: C-K bond distance change (Fortran 1114-1139) ---
            if scotc <= 0.0 { continue; }
            let pref_b_common = c0 * 2.0 * scotc * c_ij; // [cal/(mol·Å²)]
            for k in 0..nat {
                if k == i || k == j { continue; }
                let rhld2 = rkkval(natcnv(atomic_numbers[k]), natcnv(6));
                let (_c_jk, dc_jk) = cot_val(rlio[ij0(j, k)], rhld2, 0.30);
                if dc_jk == 0.0 { continue; } // beyond cutoff
                let pref_b = pref_b_common * dc_jk; // [cal/(mol·Å³)]
                // ---- T1 debug: checkpoint smx-cds.nc-grad.contrib-b ----
                if cds_debug_enabled() {
                    let ux = urlio[0][[k, j]];
                    let uy = urlio[1][[k, j]];
                    let uz = urlio[2][[k, j]];
                    println!("CDS_DEBUG| NC-grad B: i={} j={} k={} dc_jk={:.12e} pref_b={:.12e} u_K→C=({:.6e},{:.6e},{:.6e}) → dsts[C,N]+=({:.6e},{:.6e},{:.6e}) dsts[K,N]-=({:.6e},{:.6e},{:.6e})",
                        i, j, k, dc_jk, pref_b, ux, uy, uz,
                        pref_b*ux, pref_b*uy, pref_b*uz,
                        pref_b*ux, pref_b*uy, pref_b*uz);
                }
                for dir in 0..3 {
                    let u_kj = urlio[dir][[k, j]]; // û_{K→C}
                    dsts[dir][[j, i]] += pref_b * u_kj;
                    dsts[dir][[k, i]] -= pref_b * u_kj;
                }
            }
        }
    }

    // ---- C–C single-bond correction: σ_C += Σ_{C_J} COT(r_CJ) · sigma[101] ----
    for i in 0..nat {
        if atomic_numbers[i] != 6 { continue; }
        let itpc = natcnv(6);
        for j in 0..nat {
            if i == j || atomic_numbers[j] != 6 { continue; }
            let rhld = rkkval(itpc, natcnv(6));
            let r = rlio[ij0(i, j)];
            let (c, _) = cot_val(r, rhld, 0.30);
            if cds_debug_enabled() && c > 1e-10 {
                println!("CDS_DEBUG| C-C pair i={} j={} r={:.6e} rhld={:.4} cot={:.6e} contrib={:.6e}",
                    i, j, r, rhld, c, c * sigma[101]);
            }
            add_bond_corr(&mut sts, &mut dsts, nat, i, j, sigma[101], rlio, urlio, rhld, 0.30);
        }
    }
    let sts_after_cc = sts.clone();

    // ---- C–N bond correction: σ_C += (Σ_{N_J} COT(r_CJ))² · sigma[110] ----
    // Energy:  σ_C += RTKK_CN² · sigma[110],  RTKK_CN = Σ_{N_J} COT(C, N_J)
    // Gradient: ∂σ_C/∂X = sigma[110] · 2·RTKK_CN · Σ_{N_J} dCOT(C,N_J)/dr · ∂r_{C,N_J}/∂X
    //   dsts[C,C]   += sigma[110]·2·RTKK_CN·dCOT·û_{N→C}
    //   dsts[N_J,C] -= sigma[110]·2·RTKK_CN·dCOT·û_{N→C}
    for i in 0..nat {
        if atomic_numbers[i] != 6 { continue; }
        let mut rtkk_cn = 0.0;
        for j in 0..nat {
            if atomic_numbers[j] == 7 {
                let (c, _) = cot_val(rlio[ij0(i, j)], rkkval(natcnv(6), natcnv(7)), 0.30);
                rtkk_cn += c;
            }
        }
        sts[i] += rtkk_cn * rtkk_cn * sigma[110];

        // ---- C–N gradient ----
        if rtkk_cn > 0.0 && sigma[110] != 0.0 {
            let pref = sigma[110] * 2.0 * rtkk_cn;
            for j in 0..nat {
                if atomic_numbers[j] != 7 { continue; }
                let (_, dc) = cot_val(rlio[ij0(i, j)], rkkval(natcnv(6), natcnv(7)), 0.30);
                if dc == 0.0 { continue; }
                for dir in 0..3 {
                    let u_ji = urlio[dir][[j, i]]; // û_{N→C}
                    dsts[dir][[i, i]] += pref * dc * u_ji;
                    dsts[dir][[j, i]] -= pref * dc * u_ji;
                }
            }
        }
    }
    let sts_after_cn = sts.clone();

    // ---- N–C(3) triple-bond correction: σ_N += Σ_{C_J} COT(r_NJ; R₀=1.225,δ=0.065) · sigma[116] ----
    // Fortran mnsol.F lines 1132–1160.  SMD-only, not used by other MN solvation models.
    // Gradient: ∂σ_N/∂X = Σ_{C_J} sigma[116] · ∂COT/∂r · û_{C→N}  (Fortran lines 1171-1181)
    for i in 0..nat {
        if atomic_numbers[i] != 7 { continue; }
        let mut rtkk_nc3 = 0.0;
        for j in 0..nat {
            if atomic_numbers[j] == 6 {
                let (c, dc) = cot_val(rlio[ij0(i, j)], 1.225, 0.065);
                rtkk_nc3 += c;
                if dc != 0.0 {
                    for dir in 0..3 {
                        let u_ji = urlio[dir][[j, i]]; // û_{C→N}
                        dsts[dir][[i, i]] += sigma[116] * dc * u_ji;
                        dsts[dir][[j, i]] -= sigma[116] * dc * u_ji;
                    }
                }
            }
        }
        sts[i] += rtkk_nc3 * sigma[116];
    }

    // ---- debug: per-atom sigma breakdown ----
    if cds_debug_enabled() {
        let elem_name = |z: usize| -> &'static str {
            match z {
                1=>"H",2=>"He",3=>"Li",4=>"Be",5=>"B",6=>"C",7=>"N",8=>"O",9=>"F",
                14=>"Si",15=>"P",16=>"S",17=>"Cl",35=>"Br",53=>"I",
                _=>"??",
            }
        };
        println!("CDS_DEBUG| smx_cds per-atom σ (cal/mol/Å²):");
        println!("CDS_DEBUG| {:>3} {:>3} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
            "k","Z","base","+H-X","+O-X","+N-C","+C-C","+C-N","+N-C3","=final");
        for k in 0..nat {
            let d_hx  = sts_after_hx[k] - sts_base[k];
            let d_ox  = sts_after_ox[k] - sts_after_hx[k];
            let d_nc  = sts_after_nc[k] - sts_after_ox[k];
            let d_cc  = sts_after_cc[k] - sts_after_nc[k];
            let d_cn  = sts_after_cn[k] - sts_after_cc[k];
            let d_nc3 = sts[k] - sts_after_cn[k];
            println!("CDS_DEBUG| {:>3} {:>3} {:>12.6e} {:>12.6e} {:>12.6e} {:>12.6e} {:>12.6e} {:>12.6e} {:>12.6e} {:>12.6e}",
                k, elem_name(atomic_numbers[k]),
                sts_base[k], d_hx, d_ox, d_nc, d_cc, d_cn, d_nc3, sts[k]);
        }
    }

    (sts, dsts)
}

// ============================================================================
//  5. DAREAL: Accessible Solid Angle
// ============================================================================

/// Boolean matrix for SS connectivity.
struct MatrixBool { data: Vec<bool>, n: usize }
impl MatrixBool {
    fn new(n: usize) -> Self { MatrixBool { data: vec![false; n*n], n } }
    #[inline] fn at(&self, i: usize, j: usize) -> bool { self.data[i * self.n + j] }
    #[inline] fn set(&mut self, i: usize, j: usize, v: bool) { self.data[i * self.n + j] = v; }
}

/// Workspace for the DAREAL accessible-solid-angle computation.
///
/// For a central sphere K overlapped by N neighbor spheres, each overlap defines a
/// spherical segment (SS) on K's surface. The half-cone angle θ_i follows from
/// the law of cosines: `cos(θ_i) = (R_K² + d_KI² − R_I²) / (2·R_K·d_KI)`.
///
/// ## Storage conventions
///
/// - `[3, ncross]` fields (`cosn`, `dsteta`, `diwork`, ...): column-major, column i
///   corresponds to SS i, 3 rows are the 3 Cartesian directions.
/// - `[ncross, ncross]` fields (`ctheta`, `dcteta[dir]`, ...): symmetric matrices,
///   `[[i, j]]` is the value for SS pair (i, j).
/// - `dcosn[i]`: `[3, 3]` Jacobian `∂û_{K→I}/∂X_I`, stored only for diagonal i=j.
///   Derivative w.r.t. X_K has **opposite sign**: `∂û/∂X_K = −J_i`.
///
/// ## `dcteta` indexing
///
/// `dcteta[dir][[i, j]]` = `∂cosθ_ij / ∂X_I` (derivative w.r.t. atom for SS i).
/// `dcteta[dir][[j, i]]` = `∂cosθ_ij / ∂X_J`. These are generally **not equal**.
/// Derivative w.r.t. center K: `−dcteta[[i,j]] − dcteta[[j,i]]` (computed on the fly).
struct DarealWs {
    stheta: Vec<f64>,                  // [ncross] sin(θ_i)
    ctheta: MatrixFull<f64>,           // [ncross, ncross] cos(θ)
    conect: MatrixBool,                // [ncross, ncross] connectivity
    cosn: MatrixFull<f64>,             // [3, ncross] û_{K→I}, column i
    dcosn: Vec<MatrixFull<f64>>,       // [ncross] of [3, 3] ∂û/∂X_I
    dcteta: [MatrixFull<f64>; 3],      // [3] × [ncross, ncross] ∂cosθ/∂X_dir
    dsteta: MatrixFull<f64>,           // [3, ncross] ∂sinθ/∂X_dir (reserved)
    // ---- Phase E: clustered-SS polygon-area workspace (energy only) ----
    /// Free-intersection connectivity table.
    /// Shape `[2*(ncross-1)+1, ncross]`: enough rows for every SS to have
    /// up to 2*(ncross−1) free intersections (two per connected neighbour).
    /// The last row stores the per-SS count (`ncnct_count[i]`).
    ncnct: MatrixFull<isize>,
    /// Intersection-point unit vectors PIJ on sphere K.
    /// `cosn_ij[i]` is `[3, ncross]`, same layout as `dcosn[i]`.
    /// `cosn_ij[i][[dir, j]]` = dir-component of unit vector from K to the
    /// intersection point between SS i and SS j (on i's great-circle boundary).
    cosn_ij: Vec<MatrixFull<f64>>,
    /// `sit[i] = 1.0 / sin(theta_i)`, precomputed reciprocal for angle formulas.
    sit: Vec<f64>,
    /// Dihedral-angle work array (size `ncross` for sorting).
    work: Vec<f64>,
    /// Swap buffer for `work` during bubble sort.
    work_buf: Vec<f64>,
    // gradient fields
    /// ∂(dihedral)/∂X_I   per dihedral-angle index,  [3, 2*ncross]
    diwork: MatrixFull<f64>,
    /// ∂(dihedral)/∂X_J   per dihedral-angle index,  [3, 2*ncross]
    djwork: MatrixFull<f64>,
    /// ∂(dihedral)/∂X_K   per dihedral-angle index,  [3, 2*ncross]
    dkwork: MatrixFull<f64>,
    /// D0 = −DI − DJ − DK  (gradient closure),       [3, 2*ncross]
    d0work: MatrixFull<f64>,
    dw_swap: Vec<f64>,
    /// ∂(A_slice)/∂X,  col 0 = atom K, cols 1..ncross = neighbours,  [3, ncross+1]
    dca_slc: MatrixFull<f64>,
    /// ∂(A_poly)/∂X,  same layout,                                      [3, ncross+1]
    dca_ply: MatrixFull<f64>,
    /// ∂(A_odd )/∂X,  temporary accumulator per SS,                    [3, ncross+1]
    dca_odd: MatrixFull<f64>,
    /// ∂COSN(*, li, lj) / ∂X_LI  — 3×3 Jacobian per neighbour.
    /// `dicosn[li]` is `[3*ncross, 3]`: rows `3*lj .. 3*lj+2`, cols `0..2`
    /// store the Jacobian for neighbour `lj`.  Fortran: DICOSN(3,3,NAT,*).
    dicosn: Vec<MatrixFull<f64>>,
    /// ∂COSN(*, li, lj) / ∂X_LJ  — same layout.  Fortran: DJCOSN(3,3,NAT,*).
    djcosn: Vec<MatrixFull<f64>>,
    /// ∂(cosθ_i)/∂R_K  for radius gradient (unused)
    dctetr: Vec<f64>,
    /// ∂(sinθ_i)/∂R_K
    dstetr: Vec<f64>,
    /// ∂(sit_i)/∂R_K
    dsitr: Vec<f64>,
}

/// Compute the accessible solid angle Ω_k of sphere K via the DAREAL algorithm
/// (Liotard, 1992). The per-atom SASA follows as `A_k = Ω_k · R_k²`.
///
/// Returns (area0, ncross, nc, darea) where darea[3*i+dir] = ∂Ω_k/∂X_dir
/// for atom nc[i] (i=0 = center K, i≥1 = neighbour).
/// Gradient is partial: Phase E2 dihedral derivatives not yet implemented.
fn dareal(
    nat: usize, k: usize, rad: &[f64],
    rlio: &[f64], urlio: &[MatrixFull<f64>; 3],
) -> (f64, usize, Vec<usize>, Vec<f64>) {
    let twopi = 2.0 * PI;
    let fourpi = 4.0 * PI;
    let epsi: f64 = 1.0e-11;

    let mut rk = rad[k];
    if rk <= 0.0 { return (0.0, 0, vec![0], vec![0.0; 3]); }

    let mut nc = vec![0usize; nat + 1];
    let mut ncross: usize;
    let mut area0 = fourpi;
    let mut darea = vec![0.0f64; 3];

    // Fortran: outer loop target for degeneracy restart (GOTO 10, mnsol.F:1733-1735)
    // When four spheres share a point at threshold ε, RK is increased and
    // the entire dareal computation restarts from Phase A.
    let mut jp_cnt: usize = 0;
    'dareal_loop: loop {
        ncross = 0;
        nc[0] = k;
        let epsk = epsi * rk;

        // ---- Phase A: overlap detection ----
        for i in 0..nat {
            if i == k || rad[i] <= 0.0 { continue; }
            let idx_ki = ij0(i, k);
            let gap1 = rk + rad[i] - rlio[idx_ki];       // no-overlap gap
            let gap2 = rlio[idx_ki] - (rk - rad[i]).abs(); // embedding gap
            if gap1 < epsk { continue; }                  // no overlap
            if gap2 < epsk {
                if rk <= rad[i] {
                    return (0.0, 0, vec![0; nat+1], vec![0.0; 3]); // K embedded in I
                }
            } else {
                ncross += 1;
                nc[ncross] = i;                           // partial overlap
            }
        }

        if cds_debug_enabled() && ncross == 0 && nat > 1 {
            // Should never happen for molecules with bonded atoms — diagnostic
            for i in 0..nat {
                if i == k || rad[i] <= 0.0 { continue; }
                let idx_ki = ij0(i, k);
                println!("CDS_DEBUG| dareal k={} i={}: rk={:.6e} rad_i={:.6e} r_ij={:.6e} gap1={:.6e} gap2={:.6e} epsk={:.6e}",
                    k, i, rk, rad[i], rlio[idx_ki],
                    rk + rad[i] - rlio[idx_ki],
                    rlio[idx_ki] - (rk - rad[i]).abs(),
                    epsk);
            }
        }

        if ncross == 0 {
            return (area0, ncross, nc, vec![0.0; 3 * (ncross + 1)]);
        }

        if cds_debug_enabled() {
            print!("CDS_DEBUG| dareal k={} ncross={} neighbors:", k, ncross);
            for l in 1..=ncross { print!(" {}", nc[l]); }
            println!();
        }

        let mut ws = DarealWs::new(ncross);

        // ---- Phase B: SS data initialization ----
        let rk_inv = 0.5 / rk;    // 1/(2R_K)
        let rk2 = rk * rk;        // R_K²

        for i in 0..ncross {
            let li = nc[i + 1];
            let idx_ki = ij0(li, k);
            let rik_inv = 1.0 / rlio[idx_ki];                        // 1/d_KI

            let ci = rk_inv * (rlio[idx_ki] + (rk2 - rad[li] * rad[li]) * rik_inv);
            ws.ctheta[[i, i]] = ci;                                  // cosθ_i
            ws.stheta[i] = (1.0 - ci * ci).sqrt();                   // sinθ_i

            // unit vector û_{K→I} as column i of cosn
            for dir in 0..3 {
                ws.cosn[[dir, i]] = urlio[dir][[k, li]];
            }

            // Derivatives
            let x_val = -ci / ws.stheta[i];                          // −cosθ/sinθ
            let drctht = rk_inv * (1.0 - (rk2 - rad[li] * rad[li]) * rik_inv * rik_inv);
            for dir in 0..3 {
                let cosni = ws.cosn[[dir, i]];
                ws.dcteta[dir][[i, i]] = drctht * cosni;             // ∂cosθ_i/∂X_dir
                ws.dsteta[[dir, i]] = x_val * ws.dcteta[dir][[i, i]]; // ∂sinθ_i/∂X_dir

                // J_i = ∂û_{K→I}/∂X_I: symmetric 3×3, stored as dcosn[i]
                let jac = &mut ws.dcosn[i];
                let cosni = cosni * rik_inv;                         // û_dir / d_KI
                for jdir in 0..3 {
                    let cosnij = -cosni * ws.cosn[[jdir, i]];        // −û_p·û_q / d
                    jac[[jdir, dir]] = cosnij;
                    jac[[dir, jdir]] = cosnij;
                }
                jac[[dir, dir]] += rik_inv;                          // + 1/d on diagonal
            }
        }

        // ---- Phase C: connectivity ----
        for ii in 1..ncross {
            for jj in 0..ii {
                if ws.conect.at(jj, jj) { continue; }

                let cisj = ws.ctheta[[ii, ii]] * ws.stheta[jj];
                let sicj = ws.stheta[ii] * ws.ctheta[[jj, jj]];
                let sisj = ws.stheta[ii] * ws.stheta[jj];

                // cosθ_ij = û_{K→I} · û_{K→J}
                let cij = dot3(ws.cosn.slice_column(ii), ws.cosn.slice_column(jj));
                ws.ctheta[[jj, ii]] = cij;
                ws.ctheta[[ii, jj]] = cij;

                // ∂cosθ_ij/∂X: using column dir of each Jacobian (= row dir, symmetric)
                for dir in 0..3 {
                    let j_ii = &ws.dcosn[ii];
                    let j_jj = &ws.dcosn[jj];
                    let di = dot3(
                        &[j_ii[[0, dir]], j_ii[[1, dir]], j_ii[[2, dir]]],
                        ws.cosn.slice_column(jj),
                    );
                    ws.dcteta[dir][[ii, jj]] = di;
                    let dj = dot3(
                        ws.cosn.slice_column(ii),
                        &[j_jj[[0, dir]], j_jj[[1, dir]], j_jj[[2, dir]]],
                    );
                    ws.dcteta[dir][[jj, ii]] = dj;
                }

                let tij = cij - ws.ctheta[[ii, ii]] * ws.ctheta[[jj, jj]];

                if tij > sisj - epsi * (cisj - sicj).abs() {
                    if ws.ctheta[[jj, jj]] > ws.ctheta[[ii, ii]] {
                        ws.conect.set(jj, jj, true);
                    } else {
                        ws.conect.set(ii, ii, true);
                        break;
                    }
                } else {
                    let epsij = epsi * (sicj + cisj);
                    if sicj + cisj >= 0.0 {
                        ws.conect.set(jj, ii, tij > epsij - sisj);
                    } else if tij <= -sisj - epsij {
                        return (0.0, ncross, nc, vec![0.0; 3 * (ncross + 1)]);
                    } else {
                        ws.conect.set(jj, ii, true);
                    }
                    ws.conect.set(ii, jj, ws.conect.at(jj, ii));
                }
            }
        }

        // ---- Phase D: isolated SS contribution ----
        let mut a_slice = 0.0f64;
        let mut nclust = 0usize;
        let mut lab = vec![0usize; ncross];

        for i in 0..ncross {
            if ws.conect.at(i, i) { continue; }
            let mut connected_in_cluster = false;
            for j in 0..ncross {
                if ws.conect.at(j, j) { continue; }
                if ws.conect.at(j, i) {
                    lab[nclust] = i;
                    nclust += 1;
                    connected_in_cluster = true;
                    break;
                }
            }
            if connected_in_cluster { continue; }
            a_slice += 1.0 - ws.ctheta[[i, i]];
            // Fortran lines 1582-1584: DCASLC(:,I) += −DCTETA(:,I,I),  DCASLC(:,0) += +DCTETA(:,I,I)
            for dir in 0..3 {
                ws.dca_slc[[dir, i + 1]] -= ws.dcteta[dir][[i, i]];
                ws.dca_slc[[dir, 0]] += ws.dcteta[dir][[i, i]];
            }
        }
        a_slice *= twopi;
        // Fortran line 1589: DSCALMN(3*(NCROSS+1), TWOPI, DCASLC)
        for dir in 0..3 {
            for col in 0..=ncross {
                ws.dca_slc[[dir, col]] *= twopi;
            }
        }

        if nclust == 0 {
            area0 = fourpi - a_slice;
            // Fortran lines 1594-1597: DAREA = −DCASLC
            darea.resize(3 * (ncross + 1), 0.0);
            for i in 0..=ncross {
                for dir in 0..3 {
                    darea[3 * i + dir] = -ws.dca_slc[[dir, i]];
                }
            }
            return (area0, ncross, nc, darea);
        }

        // ---- Phase E: clustered SS — spherical polygon area ----
        // Refs: Liotard (1992); Fortran mnsol.F lines 1603–2159.
        //
        // Algorithm in four parts:
        //   E1  Find free intersection points between every pair of connected
        //       SS in the cluster.  An intersection is "free" when it is
        //       not buried inside any third SS.
        //   E2  For each SS, walk its boundary through the ordered free
        //       intersections, computing dihedral angles.  Accumulate
        //       contributions to the polygon area (APOLY) and the
        //       spherical-slice correction (ASLICE).
        //   E3  Count distinct spherical polygons and add the interior
        //       angle at each vertex to APOLY.
        //   E4  Assemble the final solid angle:
        //         AREA = 4π − ASLICE − (APOLY mod 4π).
        //
        // Only the energy is computed here; gradient derivatives are not
        // implemented yet (LGRX / LGRR flags from the Fortran are omitted).

        if cds_debug_enabled() {
            println!("CDS_DEBUG| dareal k={} Phase E: nclust={} ncross={}",
                k, nclust, ncross);
        }

        // ---- E1: free intersections between clustered SS ----
        // WORK(L) = cosθ_L + ε·sinθ_L  (upper bound: inside SS L)
        // WORK(L+ncross) = cosθ_L − ε·sinθ_L  (lower bound: degenerate boundary)
        let mut work_hi = vec![0.0f64; nclust + 1]; // 1-indexed for convenience
        let mut work_lo = vec![0.0f64; nclust + 1];
        for ii in 0..nclust {
            let l_ss = lab[ii]; // SS index within the full [0..ncross) range
            work_hi[ii + 1] = ws.ctheta[[l_ss, l_ss]] + epsi * ws.stheta[l_ss];
            work_lo[ii + 1] = ws.ctheta[[l_ss, l_ss]] - epsi * ws.stheta[l_ss];
        }

        let mut nfree: usize = 0;
        // Last row of ncnct stores the per-SS count
        let ncrow = if ncross > 1 { 2 * (ncross - 1) } else { 0 };
        for ii in 0..ncross {
            ws.ncnct[[ncrow, ii]] = 0;
        }

        for ii in 1..nclust {
            let li = lab[ii]; // SS index
            for jj in 0..ii {
                let lj = lab[jj];
                if !ws.conect.at(lj, li) {
                    continue;
                }

                // ---- Coefficients A, B, C for the two intersection points ----
                // PIJ = A·û_i + B·û_j + C·(û_i×û_j)    (on i's boundary)
                // PJI = A·û_i + B·û_j − C·(û_i×û_j)    (on j's boundary)
                let cij = ws.ctheta[[lj, li]]; // cosθ_ij
                let sin2ij = 1.0 / (1.0 - cij * cij); // 1/sin²θ_ij
                let a_ij = (ws.ctheta[[li, li]] - ws.ctheta[[lj, lj]] * cij) * sin2ij;
                let b_ij = (ws.ctheta[[lj, lj]] - ws.ctheta[[li, li]] * cij) * sin2ij;
                let c_ij = {
                    let tmp = (1.0 - a_ij * ws.ctheta[[li, li]]
                                   - b_ij * ws.ctheta[[lj, lj]]) * sin2ij;
                    if tmp > 0.0 { tmp.sqrt() } else { 0.0 }
                };

                // Cross product VN = û_i × û_j
                let vn = {
                    let ui = ws.cosn.slice_column(li);
                    let uj = ws.cosn.slice_column(lj);
                    cross3(
                        &[ui[0], ui[1], ui[2]],
                        &[uj[0], uj[1], uj[2]],
                    )
                };

                // PIJ and PJI (3-vectors on sphere K)
                let mut p_ij = [0.0f64; 3];
                let mut p_ji = [0.0f64; 3];
                for dir in 0..3 {
                    p_ij[dir] = a_ij * ws.cosn[[dir, li]]
                              + b_ij * ws.cosn[[dir, lj]]
                              + c_ij * vn[dir];
                    p_ji[dir] = a_ij * ws.cosn[[dir, li]]
                              + b_ij * ws.cosn[[dir, lj]]
                              - c_ij * vn[dir];
                }

                // Store intersection vectors
                for dir in 0..3 {
                    ws.cosn_ij[li][[dir, lj]] = p_ij[dir];
                    ws.cosn_ij[lj][[dir, li]] = p_ji[dir];
                }

                // ---- E1 gradient: intersection Jacobians DICOSN/DJCOSN ----
                // Fortran mnsol.F:1635–1715.
                // Compute the 3×3 Jacobian matrices ∂COSN/∂X for both PIJ and PJI.
                //
                // DICOSN(icor, jcor, li, lj) = ∂COSN(icor, li, lj) / ∂X_LI(jcor)
                // DJCOSN(icor, jcor, li, lj) = ∂COSN(icor, li, lj) / ∂X_LJ(jcor)
                //
                // Stored in ws.dicosn[li] and ws.djcosn[li] as [3*ncross, 3]:
                //   block at rows [3*lj .. 3*lj+2] = 3×3 Jacobian for neighbour lj.
                {
                    let dcij = 0.5 / c_ij; // 1/(2·CIJ), Fortran line 1636

                    // --- DIVN, DJVN: derivatives of VN = û_i × û_j (Fortran 1638-1650) ---
                    // DIVN[alpha][icor] = (∂û_i/∂X_icor × û_j)_alpha
                    // DJVN[alpha][icor] = (û_i × ∂û_j/∂X_icor)_alpha
                    let mut divn = [[0.0f64; 3]; 3];
                    let mut djvn = [[0.0f64; 3]; 3];
                    for icor in 0..3 {
                        let dcos_i = &ws.dcosn[li]; // 3×3 ∂û_i/∂X
                        let dcos_j = &ws.dcosn[lj]; // 3×3 ∂û_j/∂X
                        let ui = ws.cosn.slice_column(li);
                        let uj = ws.cosn.slice_column(lj);
                        // DIVN = ∂û_i/∂X × û_j:  ε_{αβγ} * DCOSN(β,icor,LI) * COSN(γ,LJ)
                        divn[0][icor] = dcos_i[[1, icor]] * uj[2] - dcos_i[[2, icor]] * uj[1];
                        divn[1][icor] = dcos_i[[2, icor]] * uj[0] - dcos_i[[0, icor]] * uj[2];
                        divn[2][icor] = dcos_i[[0, icor]] * uj[1] - dcos_i[[1, icor]] * uj[0];
                        // DJVN = û_i × ∂û_j/∂X:  ε_{αβγ} * COSN(β,LI) * DCOSN(γ,icor,LJ)
                        djvn[0][icor] = ui[1] * dcos_j[[2, icor]] - ui[2] * dcos_j[[1, icor]];
                        djvn[1][icor] = ui[2] * dcos_j[[0, icor]] - ui[0] * dcos_j[[2, icor]];
                        djvn[2][icor] = ui[0] * dcos_j[[1, icor]] - ui[1] * dcos_j[[0, icor]];
                    }

                    let dsn2ij = 2.0 * cij * sin2ij; // Fortran line 1651: 2·cosθ·sin²θ

                    // Per-direction quantities: DIAIJ, DJAIJ, DJBIJ, DIBIJ, DICIJ, DJCIJ
                    // Fortran lines 1672-1692 (DO 63 ICOR=1,3)
                    let mut diaij = [0.0f64; 3];
                    let mut dj_aij = [0.0f64; 3];
                    let mut dj_bij = [0.0f64; 3];
                    let mut di_bij = [0.0f64; 3];
                    let mut di_cij = [0.0f64; 3];
                    let mut dj_cij = [0.0f64; 3];

                    for icor in 0..3 {
                        let dc_li_lj = ws.dcteta[icor][[li, lj]];
                        let dc_lj_li = ws.dcteta[icor][[lj, li]];
                        let dc_li_li = ws.dcteta[icor][[li, li]];
                        let dc_lj_lj = ws.dcteta[icor][[lj, lj]];
                        let dis2ij = dsn2ij * dc_li_lj;
                        let djs2ij = dsn2ij * dc_lj_li;

                        // DIAIJ = (DCTETA(LI,LI)-CTHETA(LJ,LJ)*DCTETA(LI,LJ))*SIN2IJ + AIJ*DIS2IJ
                        diaij[icor] = (dc_li_li - ws.ctheta[[lj, lj]] * dc_li_lj) * sin2ij
                                    + a_ij * dis2ij;
                        // DJAIJ (Fortran 1678-1680)
                        dj_aij[icor] = (-dc_lj_lj * cij - ws.ctheta[[lj, lj]] * dc_lj_li) * sin2ij
                                     + a_ij * djs2ij;
                        // DJBIJ (Fortran 1681-1683)
                        dj_bij[icor] = (dc_lj_lj - ws.ctheta[[li, li]] * dc_lj_li) * sin2ij
                                     + b_ij * djs2ij;
                        // DIBIJ (Fortran 1684-1686)
                        di_bij[icor] = (-dc_li_li * cij - ws.ctheta[[li, li]] * dc_li_lj) * sin2ij
                                     + b_ij * dis2ij;
                        // DICIJ (Fortran 1687-1689)
                        di_cij[icor] = -dcij * ((diaij[icor] * ws.ctheta[[li, li]]
                                                + a_ij * dc_li_li
                                                + di_bij[icor] * ws.ctheta[[lj, lj]]) * sin2ij)
                                     + 0.5 * c_ij * dis2ij;
                        // DJCIJ (Fortran 1690-1692)
                        dj_cij[icor] = -dcij * ((dj_bij[icor] * ws.ctheta[[lj, lj]]
                                                + b_ij * dc_lj_lj
                                                + dj_aij[icor] * ws.ctheta[[li, li]]) * sin2ij)
                                     + 0.5 * c_ij * djs2ij;
                    }

                    // --- Assemble DICOSN/DJCOSN (Fortran 1702-1714, DO 64) ---
                    // For each (icor, jcor) pair, build the 3×3 Jacobian blocks
                    // for both PIJ=(LI,LJ) with +C·VN and PJI=(LJ,LI) with −C·VN.
                    for icor in 0..3 {
                        for jcor in 0..3 {
                            let dicos = diaij[jcor] * ws.cosn[[icor, li]]
                                      + a_ij * ws.dcosn[li][[icor, jcor]]
                                      + di_bij[jcor] * ws.cosn[[icor, lj]];
                            let djcos = dj_aij[jcor] * ws.cosn[[icor, li]]
                                      + dj_bij[jcor] * ws.cosn[[icor, lj]]
                                      + b_ij * ws.dcosn[lj][[icor, jcor]];
                            let diwij = di_cij[jcor] * vn[icor] + c_ij * divn[icor][jcor];
                            let djwij = dj_cij[jcor] * vn[icor] + c_ij * djvn[icor][jcor];

                            // PIJ (COSN(*,LI,LJ)): +C·VN
                            ws.dicosn[li][[3 * lj + icor, jcor]] = dicos + diwij;
                            ws.djcosn[li][[3 * lj + icor, jcor]] = djcos + djwij;
                            // PJI (COSN(*,LJ,LI)): −C·VN  — Fortran 1713-1714
                            ws.dicosn[lj][[3 * li + icor, jcor]] = dicos - diwij;
                            ws.djcosn[lj][[3 * li + icor, jcor]] = djcos - djwij;
                        }
                    }
                }

                // ---- Check whether PIJ and PJI are "free" ----
                let mut free_ij = true;
                let mut free_ji = true;

                for ll_idx in 1..=nclust {
                    let ll_s = lab[ll_idx - 1];
                    if ll_s == li || ll_s == lj {
                        continue;
                    }
                    if ws.conect.at(ll_s, li) && ws.conect.at(ll_s, lj) {
                        // Is PJI inside SS L?
                        if free_ji {
                            let chek = dot3(&p_ji, ws.cosn.slice_column(ll_s));
                            if chek > work_hi[ll_idx] {
                                free_ji = false;
                            } else if chek >= work_lo[ll_idx] {
                                // Fortran: four spheres K, LI, LJ, LL share a point
                                // at threshold ε. Increase RK and restart (GOTO 10).
                                rk *= 1.0 + 4.0 * epsi;
                                jp_cnt += 1;
                                if cds_debug_enabled() {
                                    println!("CDS_DEBUG| dareal k={} E1 degeneracy (PJI): rk*={:.6e} jp_cnt={}",
                                        k, rk, jp_cnt);
                                }
                                continue 'dareal_loop;
                            }
                        }
                        // Is PIJ inside SS L?
                        if free_ij {
                            let chek = dot3(&p_ij, ws.cosn.slice_column(ll_s));
                            if chek > work_hi[ll_idx] {
                                free_ij = false;
                            } else if chek >= work_lo[ll_idx] {
                                // Fortran: degeneracy on PIJ side
                                rk *= 1.0 + 4.0 * epsi;
                                jp_cnt += 1;
                                if cds_debug_enabled() {
                                    println!("CDS_DEBUG| dareal k={} E1 degeneracy (PIJ): rk*={:.6e} jp_cnt={}",
                                        k, rk, jp_cnt);
                                }
                                continue 'dareal_loop;
                            }
                        }
                        if !free_ij && !free_ji {
                            break;
                        }
                    }
                }

                // ---- Record free intersections in ncnct ----
                // Positive neighbour → use cosn_ij[neighbour][ss]
                // Negative neighbour → use cosn_ij[ss][neighbour] (sign fixed later)
                // NCNCT stores (SS‑index + 1) with a sign convention:
                //   +val  → neighbour = val−1,  use cosn_ij[neighbour][ss]
                //   −val  → neighbour = val−1,  use cosn_ij[ss][neighbour]
                // The +1 offset avoids −0 ≡ 0 ambiguity with 0‑based indices.
                // Fortran uses 1‑based indices naturally; we mimic it here.
                if free_ji {
                    nfree += 1;
                    let m = ws.ncnct[[ncrow, li]] as usize;
                    ws.ncnct[[m, li]] = (lj + 1) as isize;               // +val
                    ws.ncnct[[ncrow, li]] = (m + 1) as isize;
                    let m2 = ws.ncnct[[ncrow, lj]] as usize;
                    ws.ncnct[[m2, lj]] = -((li + 1) as isize);           // −val
                    ws.ncnct[[ncrow, lj]] = (m2 + 1) as isize;
                }
                if free_ij {
                    nfree += 1;
                    let m = ws.ncnct[[ncrow, li]] as usize;
                    ws.ncnct[[m, li]] = -((lj + 1) as isize);            // −val
                    ws.ncnct[[ncrow, li]] = (m + 1) as isize;
                    let m2 = ws.ncnct[[ncrow, lj]] as usize;
                    ws.ncnct[[m2, lj]] = (li + 1) as isize;              // +val
                    ws.ncnct[[ncrow, lj]] = (m2 + 1) as isize;
                }
                if cds_debug_enabled() {
                    println!("CDS_DEBUG| E1 pair li={} lj={}: free_ji={} free_ij={}  nfree_sofar={}",
                        li, lj, free_ji, free_ij, nfree);
                }
            } // jj
        } // ii

        if cds_debug_enabled() {
            println!("CDS_DEBUG| dareal k={} Phase E1: nfree={}", k, nfree);
        }

        // No free intersections → sphere K buried by the cluster
        if nfree == 0 {
            if cds_debug_enabled() {
                println!("CDS_DEBUG| dareal k={} buried by cluster -> area=0", k);
            }
            return (0.0, 0, vec![0; nat + 1], vec![0.0; 3]);
        }

        // ---- E2: oriented dihedral angles along each SS boundary ----
        // Precompute sit[i] = 1 / sinθ_i
        for i_ss in 0..ncross {
            ws.sit[i_ss] = 1.0 / ws.stheta[i_ss];
        }

        let mut apoly = 0.0f64; // accumulated spherical polygon area
        let mut aslice = a_slice; // start from isolated-SS slice area

        for ii in 1..=nclust {
            let li = lab[ii - 1];
            let nphi = ws.ncnct[[ncrow, li]] as usize;
            if nphi == 0 {
                continue;
            }

            // --- E2 gradient A: zero DCAODD for this SS (Fortran 1801) ---
            for dir in 0..3 {
                for col in 0..=ncross {
                    ws.dca_odd[[dir, col]] = 0.0;
                }
            }

            // ---- Sign fixup for the first neighbour ----
            // NCNCT stores (SS‑index + 1), signed.  Decode to 0‑based lj.
            let lj_enc = ws.ncnct[[0, li]];
            // Guard against stale/zero entries (should never happen, but be safe)
            if lj_enc == 0 {
                // No valid neighbour — skip this SS
                break;
            }
            let (lj0, lj_positive) = if lj_enc > 0 {
                ((lj_enc - 1) as usize, true)                 // +val → neighbour is lj0
            } else {
                let pos = -lj_enc;
                ws.ncnct[[0, li]] = pos;                       // fix sign in table
                ((pos - 1) as usize, false)                    // −val → reverse direction
            };

            // Build CNIJ(1:3), the unit vector from K to the intersection point
            let mut cnij = [0.0f64; 3];
            for dir in 0..3 {
                cnij[dir] = if lj_positive {
                    ws.cosn_ij[lj0][[dir, li]]                 // + → cosn_ij[neighbour][li]
                } else {
                    ws.cosn_ij[li][[dir, lj0]]                 // − → cosn_ij[li][neighbour]
                };
            }

            // Cross product VIJ = û_i × CNIJ
            let vij = {
                let ui = ws.cosn.slice_column(li);
                cross3(&[ui[0], ui[1], ui[2]], &cnij)
            };
            // LPOLY = (VIJ · û_j) > 0  — determines whether odd or even
            // angles belong to the polygon
            let lpoly = dot3(&vij, ws.cosn.slice_column(lj0)) > 0.0;

            let c2i = ws.ctheta[[li, li]] * ws.ctheta[[li, li]];

            // --- E2 gradient B: CNIJ derivatives (Fortran 1809-1866) ---
            // B1: Decode DICNIJ/DJCNIJ from E1's DICOSN/DJCOSN with LI≥LJ symmetry.
            // DICNIJ[alpha][icor] = ∂CNIJ(alpha)/∂X_LI(icor)
            // DJCNIJ[alpha][icor] = ∂CNIJ(alpha)/∂X_LJ(icor)
            let mut dicnij = [[0.0f64; 3]; 3];
            let mut djcnij = [[0.0f64; 3]; 3];
            {
                let (src_li, src_lj) = if lj_positive {
                    (lj0, li) // CNIJ = COSN(*, LJ, LI) → DICOSN(*,*, LJ, LI)
                } else {
                    (li, lj0) // CNIJ = COSN(*, LI, LJ) → DICOSN(*,*, LI, LJ)
                };
                let use_normal = li >= lj0;
                for icor in 0..3 {
                    for jcor in 0..3 {
                        if use_normal {
                            dicnij[icor][jcor] = ws.dicosn[src_li][[3 * src_lj + icor, jcor]];
                            djcnij[icor][jcor] = ws.djcosn[src_li][[3 * src_lj + icor, jcor]];
                        } else {
                            // LI < LJ: swap DICOSN ↔ DJCOSN roles
                            dicnij[icor][jcor] = ws.djcosn[src_li][[3 * src_lj + icor, jcor]];
                            djcnij[icor][jcor] = ws.dicosn[src_li][[3 * src_lj + icor, jcor]];
                        }
                    }
                }
            }

            // B2: DCSIT = -SIT² · DSTETA  —  ∂(1/sinθ)/∂X  (Fortran 1848-1849)
            let mut dcsit = [0.0f64; 3];
            for dir in 0..3 {
                dcsit[dir] = -ws.sit[li].powi(2) * ws.dsteta[[dir, li]];
            }
            // B3: DCC2I = 2·cosθ · DCTETA  —  ∂(cos²θ)/∂X  (Fortran 1850)
            let mut dcc2i = [0.0f64; 3];
            for dir in 0..3 {
                dcc2i[dir] = 2.0 * ws.ctheta[[li, li]] * ws.dcteta[dir][[li, li]];
            }
            // B4: DIVIJ = ∂(û_i × CNIJ)/∂X_LI,  DJVIJ = ∂(û_i × CNIJ)/∂X_LJ
            // Fortran 1851-1865.
            let mut divij = [[0.0f64; 3]; 3];
            let mut djvij = [[0.0f64; 3]; 3];
            {
                let ui_data = ws.cosn.slice_column(li);
                let ui = [ui_data[0], ui_data[1], ui_data[2]];
                for icor in 0..3 {
                    // DIVIJ[alpha][icor] = (∂û_i/∂X_{icor} × CNIJ)_alpha + (û_i × DICNIJ(:,[icor]))_alpha
                    let du = &ws.dcosn[li];
                    divij[0][icor] = du[[1, icor]] * cnij[2] - du[[2, icor]] * cnij[1]
                                   + ui[1] * dicnij[2][icor] - ui[2] * dicnij[1][icor];
                    divij[1][icor] = du[[2, icor]] * cnij[0] - du[[0, icor]] * cnij[2]
                                   + ui[2] * dicnij[0][icor] - ui[0] * dicnij[2][icor];
                    divij[2][icor] = du[[0, icor]] * cnij[1] - du[[1, icor]] * cnij[0]
                                   + ui[0] * dicnij[1][icor] - ui[1] * dicnij[0][icor];
                    // DJVIJ[alpha][icor] = û_i × DJCNIJ(:,[icor])  (only second term — û_i independent of X_LJ)
                    djvij[0][icor] = ui[1] * djcnij[2][icor] - ui[2] * djcnij[1][icor];
                    djvij[1][icor] = ui[2] * djcnij[0][icor] - ui[0] * djcnij[2][icor];
                    djvij[2][icor] = ui[0] * djcnij[1][icor] - ui[1] * djcnij[0][icor];
                }
            }

            // ---- Compute dihedral angles for each consecutive pair ----
            for j_idx in 1..nphi {
                let lk_enc = ws.ncnct[[j_idx, li]];
                // Guard against zero entries (unvisited or stale); skip if invalid
                if lk_enc == 0 {
                    continue;
                }
                let lk0;
                let mut cnik = [0.0f64; 3];
                if lk_enc > 0 {
                    lk0 = (lk_enc - 1) as usize;
                    if lk0 >= ncross { continue; }
                    for dir in 0..3 {
                        cnik[dir] = ws.cosn_ij[lk0][[dir, li]];
                    }
                } else {
                    let pos = -lk_enc;
                    lk0 = (pos - 1) as usize;
                    if lk0 >= ncross { continue; }
                    ws.ncnct[[j_idx, li]] = pos;              // fix sign in table
                    for dir in 0..3 {
                        cnik[dir] = ws.cosn_ij[li][[dir, lk0]];
                    }
                }

                // Dihedral angle between planes (û_i, CNIJ_prev) and (û_i, CNIK)
                let x = dot3(&vij, &cnik);
                let y = dot3(&cnik, &cnij) - c2i;
                ws.work[j_idx - 1] = f64::atan2(x, y);
                if ws.work[j_idx - 1] <= 0.0 {
                    ws.work[j_idx - 1] += twopi;
                }

                // --- E2 gradient C: DICNIK/DKCNIK decode + atan2 chain rule ----
                // Fortran 1883-1936.
                // C1: Decode DICNIK/DKCNIK from E1's DICOSN/DJCOSN (same symmetry rule as B1).
                let mut dicnik = [[0.0f64; 3]; 3];
                let mut dkcnk = [[0.0f64; 3]; 3];
                {
                    let lk0_decoded = if lk_enc > 0 { (lk_enc - 1) as usize } else { ((-lk_enc) - 1) as usize };
                    let (src_li2, src_lk) = if lk_enc > 0 {
                        (lk0_decoded, li) // CNIK = COSN(*, LK, LI)
                    } else {
                        (li, lk0_decoded) // CNIK = COSN(*, LI, LK)
                    };
                    let use_normal2 = li >= lk0_decoded;
                    for icor in 0..3 {
                        for jcor in 0..3 {
                            if use_normal2 {
                                dicnik[icor][jcor] = ws.dicosn[src_li2][[3 * src_lk + icor, jcor]];
                                dkcnk[icor][jcor] = ws.djcosn[src_li2][[3 * src_lk + icor, jcor]];
                            } else {
                                dicnik[icor][jcor] = ws.djcosn[src_li2][[3 * src_lk + icor, jcor]];
                                dkcnk[icor][jcor] = ws.dicosn[src_li2][[3 * src_lk + icor, jcor]];
                            }
                        }
                    }
                }

                // C2: atan2 chain rule (Fortran 1918-1935).
                //   φ = atan2(x, y),  dx = -x/(x²+y²),  dy = y/(x²+y²)
                //   dφ = dy·dx + dx·dy
                let r2 = x * x + y * y;
                if r2 > 1e-30 {
                    let dx = -x / r2;
                    let dy = y / r2;
                    for icor in 0..3 {
                        // DIX = DIVIJ(:,icor)·CNIK + VIJ·DICNIK(:,icor)
                        let dix = divij[0][icor] * cnik[0] + divij[1][icor] * cnik[1] + divij[2][icor] * cnik[2]
                                + vij[0] * dicnik[0][icor] + vij[1] * dicnik[1][icor] + vij[2] * dicnik[2][icor];
                        // DIY = DICNIK(:,icor)·CNIJ + CNIK·DICNIJ(:,icor) − DCC2I(icor)
                        let diy = dicnik[0][icor] * cnij[0] + dicnik[1][icor] * cnij[1] + dicnik[2][icor] * cnij[2]
                                + cnik[0] * dicnij[0][icor] + cnik[1] * dicnij[1][icor] + cnik[2] * dicnij[2][icor]
                                - dcc2i[icor];
                        // DJX = DJVIJ(:,icor)·CNIK
                        let djx = djvij[0][icor] * cnik[0] + djvij[1][icor] * cnik[1] + djvij[2][icor] * cnik[2];
                        // DJY = CNIK·DJCNIJ(:,icor)
                        let djy = cnik[0] * djcnij[0][icor] + cnik[1] * djcnij[1][icor] + cnik[2] * djcnij[2][icor];
                        // DKX = VIJ·DKCNIK(:,icor)
                        let dkx = vij[0] * dkcnk[0][icor] + vij[1] * dkcnk[1][icor] + vij[2] * dkcnk[2][icor];
                        // DKY = DKCNIK(:,icor)·CNIJ
                        let dky = dkcnk[0][icor] * cnij[0] + dkcnk[1][icor] * cnij[1] + dkcnk[2][icor] * cnij[2];

                        // DIWORK / DJWORK / DKWORK / D0WORK (Fortran 1932-1935)
                        ws.diwork[[icor, j_idx - 1]] = dy * dix + dx * diy;
                        ws.djwork[[icor, j_idx - 1]] = dy * djx + dx * djy;
                        ws.dkwork[[icor, j_idx - 1]] = dy * dkx + dx * dky;
                        ws.d0work[[icor, j_idx - 1]] = -(ws.diwork[[icor, j_idx - 1]]
                                                        + ws.djwork[[icor, j_idx - 1]]
                                                        + ws.dkwork[[icor, j_idx - 1]]);
                    }
                } else {
                    // Degenerate: angle is indeterminate, zero out gradient columns
                    for icor in 0..3 {
                        ws.diwork[[icor, j_idx - 1]] = 0.0;
                        ws.djwork[[icor, j_idx - 1]] = 0.0;
                        ws.dkwork[[icor, j_idx - 1]] = 0.0;
                        ws.d0work[[icor, j_idx - 1]] = 0.0;
                    }
                }
            }

            // ---- Sort dihedral angles (ascending), bubble sort ----
            if nphi > 2 {
                for j_a in 0..nphi - 2 {
                    for j_b in j_a + 1..nphi - 1 {
                        if ws.work[j_a] > ws.work[j_b] {
                            // swap work
                            let tmp_w = ws.work[j_b];
                            ws.work[j_b] = ws.work[j_a];
                            ws.work[j_a] = tmp_w;
                            // swap corresponding ncnct entries
                            let tmp_n = ws.ncnct[[j_b + 1, li]];
                            ws.ncnct[[j_b + 1, li]] = ws.ncnct[[j_a + 1, li]];
                            ws.ncnct[[j_a + 1, li]] = tmp_n;
                            // --- E2 gradient D: swap gradient columns (Fortran 1972-1977) ---
                            for dir in 0..3 {
                                ws.dw_swap[dir] = ws.diwork[[dir, j_b]];
                                ws.diwork[[dir, j_b]] = ws.diwork[[dir, j_a]];
                                ws.diwork[[dir, j_a]] = ws.dw_swap[dir];

                                ws.dw_swap[dir] = ws.djwork[[dir, j_b]];
                                ws.djwork[[dir, j_b]] = ws.djwork[[dir, j_a]];
                                ws.djwork[[dir, j_a]] = ws.dw_swap[dir];

                                ws.dw_swap[dir] = ws.dkwork[[dir, j_b]];
                                ws.dkwork[[dir, j_b]] = ws.dkwork[[dir, j_a]];
                                ws.dkwork[[dir, j_a]] = ws.dw_swap[dir];

                                ws.dw_swap[dir] = ws.d0work[[dir, j_b]];
                                ws.d0work[[dir, j_b]] = ws.d0work[[dir, j_a]];
                                ws.d0work[[dir, j_a]] = ws.dw_swap[dir];
                            }
                        }
                    }
                }
            }

            // ---- Compute odd/even area contributions ----
            let mut aodd = ws.work[0];
            // --- E2 gradient E: DCAODD accumulation (Fortran 1949-2016) ---
            // First term (work[0]) — contribution from the first dihedral angle.
            {
                let lj_d = lj0; // first neighbour (already decoded above)
                let lk_enc1 = ws.ncnct[[1, li]] as usize; // second neighbour (encoded)
                if lk_enc1 > 0 && lk_enc1 <= ncross + 1 {
                    let lk_d = lk_enc1 - 1; // 0-based SS index
                    for dir in 0..3 {
                        ws.dca_odd[[dir, li + 1]] += ws.diwork[[dir, 0]];
                        ws.dca_odd[[dir, lj_d + 1]] += ws.djwork[[dir, 0]];
                        ws.dca_odd[[dir, lk_d + 1]] += ws.dkwork[[dir, 0]];
                        ws.dca_odd[[dir, 0]] += ws.d0work[[dir, 0]];
                    }
                }
            }
            if nphi > 2 {
                for j_idx in (2..nphi - 1).step_by(2) {
                    aodd += ws.work[j_idx] - ws.work[j_idx - 1];
                    // DCAODD alternating sum (Fortran 2000-2016)
                    // j_idx = 2,4,...,nphi-2  (Rust 0-based; Fortran J=3,5,...,NPHI-1)
                    let lk_enc_even = ws.ncnct[[j_idx, li]] as usize;      // even-index neighbour
                    let ll_enc_odd  = ws.ncnct[[j_idx + 1, li]] as usize;  // odd-index neighbour
                    for dir in 0..3 {
                        ws.dca_odd[[dir, li + 1]] += ws.diwork[[dir, j_idx]]
                                                   - ws.diwork[[dir, j_idx - 1]];
                        ws.dca_odd[[dir, lj0 + 1]] += ws.djwork[[dir, j_idx]]
                                                    - ws.djwork[[dir, j_idx - 1]];
                    }
                    if lk_enc_even > 0 && lk_enc_even <= ncross + 1 {
                        let lk_ev = lk_enc_even - 1;
                        for dir in 0..3 {
                            ws.dca_odd[[dir, lk_ev + 1]] -= ws.dkwork[[dir, j_idx - 1]];
                        }
                    }
                    if ll_enc_odd > 0 && ll_enc_odd <= ncross + 1 {
                        let ll_od = ll_enc_odd - 1;
                        for dir in 0..3 {
                            ws.dca_odd[[dir, ll_od + 1]] += ws.dkwork[[dir, j_idx]];
                        }
                    }
                    for dir in 0..3 {
                        ws.dca_odd[[dir, 0]] += ws.d0work[[dir, j_idx]]
                                               - ws.d0work[[dir, j_idx - 1]];
                    }
                }
            }
            let aeven = twopi - aodd;
            let x_slice = 1.0 - ws.ctheta[[li, li]];

            if cds_debug_enabled() {
                print!("CDS_DEBUG| E2 li={}: lpoly={} nphi={} work[0]={:.6e} aodd={:.6e} aeven={:.6e} x_slice={:.6e}",
                    li, lpoly, nphi, ws.work[0], aodd, aeven, x_slice);
                print!("  ncnct_pre=[");
                for kk in 0..nphi { print!(" {}", ws.ncnct[[kk, li]]); }
                print!(" ]");
            }
            if lpoly {
                // odd dihedrals are polygon vertices
                apoly += aodd;
                aslice += aeven * x_slice;
                // Fortran 2020-2025: DCASLC(:,LI) -= AEVEN*DCTETA,  DCASLC(:,0) += AEVEN*DCTETA
                for dir in 0..3 {
                    ws.dca_slc[[dir, li + 1]] -= aeven * ws.dcteta[dir][[li, li]];
                    ws.dca_slc[[dir, 0]] += aeven * ws.dcteta[dir][[li, li]];
                }
                // --- E2 gradient F (lpoly=true): DCAODD → DCAPLY/DCASLC (Fortran 2024-2037) ---
                // DCAPLY += DCAODD,  DCASLC -= DCAODD * X
                for dir in 0..3 {
                    for col in 0..=ncross {
                        let d_odd = ws.dca_odd[[dir, col]];
                        ws.dca_ply[[dir, col]] += d_odd;
                        ws.dca_slc[[dir, col]] -= d_odd * x_slice;
                    }
                }
            } else {
                // even dihedrals are polygon vertices
                apoly += aeven;
                aslice += aodd * x_slice;
                // Fortran 2043-2048: DCASLC(:,LI) -= AODD*DCTETA,  DCASLC(:,0) += AODD*DCTETA
                for dir in 0..3 {
                    ws.dca_slc[[dir, li + 1]] -= aodd * ws.dcteta[dir][[li, li]];
                    ws.dca_slc[[dir, 0]] += aodd * ws.dcteta[dir][[li, li]];
                }
                // --- E2 gradient F (lpoly=false): DCAODD → DCAPLY/DCASLC (Fortran 2047-2060) ---
                // DCAPLY -= DCAODD,  DCASLC += DCAODD * X
                for dir in 0..3 {
                    for col in 0..=ncross {
                        let d_odd = ws.dca_odd[[dir, col]];
                        ws.dca_ply[[dir, col]] -= d_odd;
                        ws.dca_slc[[dir, col]] += d_odd * x_slice;
                    }
                }
            }

            // Reorder labels so polygon vertices are at odd ranks.
            // Fortran (mnsol.F:2054-2058): only when .NOT.LPOLY.
            //   LPOLY=true  → odd dihedrals are already polygon vertices → no rotate
            //   LPOLY=false → even dihedrals are polygon vertices → shift by 1
            if !lpoly && nphi > 0 {
                let first = ws.ncnct[[0, li]];
                for j_idx in 0..nphi - 1 {
                    ws.ncnct[[j_idx, li]] = ws.ncnct[[j_idx + 1, li]];
                }
                ws.ncnct[[nphi - 1, li]] = first;
                // Rotate gradient columns to match NCNCT rotation (Fortran implicit:
                // the gradient arrays are indexed by dihedral position, not by
                // NCNCT position, so no column rotation is needed for DIWORK etc.
                // The NCNCT rotation already reorders the labels for E3.)
            }
            if cds_debug_enabled() {
                let rotated = !lpoly && nphi > 0;
                print!("  ncnct_post=[");
                for kk in 0..nphi { print!(" {}", ws.ncnct[[kk, li]]); }
                println!(" ]  rotate_applied={} (Fortran: only if !lpoly)", rotated);
                // Per-column E2 gradient accumulators (non-zero only)
                for col in 0..=ncross {
                    let has_nonzero = (0..3).any(|dir|
                        ws.dca_odd[[dir, col]].abs() > 1e-20
                        || ws.dca_ply[[dir, col]].abs() > 1e-20
                        || ws.dca_slc[[dir, col]].abs() > 1e-20
                    );
                    if has_nonzero {
                        print!("    E2 grad col={}: odd=[", col);
                        for dir in 0..3 { print!(" {:.6e}", ws.dca_odd[[dir, col]]); }
                        print!(" ] ply=[");
                        for dir in 0..3 { print!(" {:.6e}", ws.dca_ply[[dir, col]]); }
                        print!(" ] slc=[");
                        for dir in 0..3 { print!(" {:.6e}", ws.dca_slc[[dir, col]]); }
                        println!(" ]");
                    }
                }
            }
        } // ii (E2)

        // ---- E3: count polygons and sum vertex interior angles ----
        // Fortran DAREAL lines 2065–2152.  For each unvisited edge in the
        // NCNCT table, walk the full spherical polygon, computing the interior
        // angle φ at each vertex:
        //   φ = acos( (cosθ_AB − cosθ_A·cosθ_B) / (sinθ_A · sinθ_B) )
        //
        // Helper: compute φ(ia, ib) and accumulate its gradient (Fortran 2078-2090).
        /// Compute interior angle φ(ia, ib) and its gradient.
        /// Returns (φ, dφ/dR_ia, dφ/dR_ib).  The caller must accumulate into dca_ply.
        fn phi_with_grad(
            ia: usize, ib: usize, ws: &DarealWs,
        ) -> (f64, [f64; 3], [f64; 3]) {
            let p1 = ws.ctheta[[ia, ib]] - ws.ctheta[[ia, ia]] * ws.ctheta[[ib, ib]];
            let p2 = ws.sit[ia] * ws.sit[ib];
            let phi = f64::acos((p1 * p2).clamp(-1.0, 1.0));
            let mut daphi = [0.0f64; 3];
            let mut dbphi = [0.0f64; 3];
            let sin_phi = phi.sin();
            if sin_phi.abs() > 1e-15 {
                let s1nphi = 1.0 / sin_phi;
                let p1m = s1nphi * p1;
                let p2m = s1nphi * p2;
                for dir in 0..3 {
                    let dcsit_a = -ws.sit[ia].powi(2) * ws.dsteta[[dir, ia]];
                    let dcsit_b = -ws.sit[ib].powi(2) * ws.dsteta[[dir, ib]];
                    daphi[dir] = (ws.dcteta[dir][[ia, ib]] - ws.ctheta[[ib, ib]] * ws.dcteta[dir][[ia, ia]]) * p2m
                               + p1m * dcsit_a * ws.sit[ib];
                    dbphi[dir] = (ws.dcteta[dir][[ib, ia]] - ws.ctheta[[ia, ia]] * ws.dcteta[dir][[ib, ib]]) * p2m
                               + p1m * ws.sit[ia] * dcsit_b;
                }
            }
            (phi, daphi, dbphi)
        }
        /// Accumulate phi gradient into dca_ply columns.
        fn accum_phi_grad(ia: usize, ib: usize, daphi: &[f64; 3], dbphi: &[f64; 3], ws: &mut DarealWs) {
            for dir in 0..3 {
                ws.dca_ply[[dir, ia + 1]] -= daphi[dir];
                ws.dca_ply[[dir, ib + 1]] -= dbphi[dir];
                ws.dca_ply[[dir, 0]] += daphi[dir] + dbphi[dir];
            }
        }
        let mut npoly: usize = 0;
        for ii in 1..=nclust {
            let li = lab[ii - 1];
            let ncnt = ws.ncnct[[ncrow, li]] as usize;
            if cds_debug_enabled() {
                print!("CDS_DEBUG| E3 entry: li={} ncnt={}  ncnct=[", li, ncnt);
                for k in 0..ncnt {
                    print!(" {}", ws.ncnct[[k, li]]);
                }
                println!(" ]");
            }
            // Fortran loop: DO 160 J=2,NCNCT(MXSS,LI),2  (1-indexed, even → 0-idx odd)
            for j_idx in (1..ncnt).step_by(2) {
                if ws.ncnct[[j_idx, li]] == 0 {
                    continue; // already visited
                }
                // NCNCT stores (0‑based + 1): decode to 0‑based SS indices
                let ia_enc = ws.ncnct[[j_idx - 1, li]] as usize; // 1‑based encoded
                let mut ia = ia_enc - 1;                          // 0‑based
                if cds_debug_enabled() {
                    println!("CDS_DEBUG| E3 raw: j_idx={} ncnct[[{}]]={} ncnct[[{}]]={} ia_enc={} ia={}",
                        j_idx, j_idx - 1, ws.ncnct[[j_idx - 1, li]],
                        j_idx, ws.ncnct[[j_idx, li]], ia_enc, ia);
                }
                let mut ib = li;
                let ia_start = ia;       // Fortran NCNCT(J‑1,LI) — first vertex
                ws.ncnct[[j_idx, li]] = 0; // mark edge visited

                // --- first vertex φ(ia, ib) ---
                let (phi_first, daphi_first, dbphi_first) = phi_with_grad(ia, ib, &ws);
                apoly += phi_first;
                accum_phi_grad(ia, ib, &daphi_first, &dbphi_first, &mut ws);
                let mut last_phi = phi_first;
                let mut last_daphi = daphi_first;
                let mut last_dbphi = dbphi_first;

                if cds_debug_enabled() {
                    println!("CDS_DEBUG| E3 start edge: li={} ia={} ib={} phi={:.6e}",
                        li, ia, ib, phi_first);
                }

                // --- polygon walk (Fortran DO 140, mnsol.F:2102-2147) ---
                let mut polygon_closed = false;
                loop {
                    let ncnt_a = ws.ncnct[[ncrow, ia]] as usize;
                    let mut found = false;
                    for m_idx in (1..ncnt_a).step_by(2) {
                        if (ws.ncnct[[m_idx, ia]] as usize) == (ib + 1)          // encoded cmp
                            && ws.ncnct[[m_idx - 1, ia]] > 0
                        {
                            let ibold = ib; // Fortran IBOLD — save BEFORE update
                            ws.ncnct[[m_idx, ia]] = 0; // mark visited
                            let ia_new = (ws.ncnct[[m_idx - 1, ia]] as usize) - 1; // decode
                            // IB = IA (new IB = old IA), IA = NCNCT(L-1, IB) (new IA)
                            let ib_new = ia;

                            if ia_new != ibold {
                                // Fortran: normal step — φ(new_IA, new_IB)
                                let (phi, daphi_new, dbphi_new) = phi_with_grad(ia_new, ib_new, &ws);
                                apoly += phi;
                                accum_phi_grad(ia_new, ib_new, &daphi_new, &dbphi_new, &mut ws);
                                last_phi = phi;
                                last_daphi = daphi_new;
                                last_dbphi = dbphi_new;
                                if cds_debug_enabled() {
                                    println!("CDS_DEBUG| E3 walk: ia_old={} ib_old={} → ia_new={} ib_new={}  ibold={} (ibold≠ia_new→normal)  φ={:.6e}",
                                        ia, ib, ia_new, ib_new, ibold, phi);
                                }
                                ib = ib_new;
                                ia = ia_new;
                                found = true;
                                break;
                            } else {
                                // Fortran: IA == IBOLD — polygon closed (ELSE branch)
                                //   IB = LI, IA = NCNCT(J-1, LI)  (reset to start)
                                //   APOLY += PHI  (reuses last computed φ)
                                //   DCAPLY(IA, IB) -= DAPHI/DBPHI  (reuses last gradient)
                                apoly += last_phi;
                                accum_phi_grad(ia_start, li, &last_daphi, &last_dbphi, &mut ws);
                                ib = li;          // Fortran: IB = LI
                                ia = ia_start;    // Fortran: IA = NCNCT(J-1, LI)
                                if cds_debug_enabled() {
                                    println!("CDS_DEBUG| E3 walk-close: ia_old={} ib_old={} ia_new={} ib_new={}  ibold={} (ibold=ia_new→close)  reusing_last_phi={:.6e}  reset→(ia={} ib={})",
                                        ia, ib, ia_new, ib_new, ibold, last_phi, ia, ib);
                                }
                                polygon_closed = true;
                                found = true;
                                break;
                            }
                        }
                    }
                    if !found || polygon_closed {
                        break;
                    }
                }

                // Fortran line 2149: walking loop exhausted → just NPOLY = NPOLY + 1.
                // No closing vertex is added (unlike the IA==IBOLD branch, which
                // reuses the last PHI and proceeds to GOTO 140).
                npoly += 1;
            }
        }

        if cds_debug_enabled() {
            println!("CDS_DEBUG| dareal k={}  pre-E4: apoly_raw={:.6e} aslice={:.6e} npoly={} nfree={}",
                k, apoly, aslice, npoly, nfree);
        }
        // ---- E4: final solid angle ----
        // Fortran: APOLY=APOLY+(NPOLY-NFREE)*TWOPI
        //          AREA=FOURPI-ASLICE-MOD(APOLY,FOURPI)
        apoly += (npoly as isize - nfree as isize) as f64 * twopi;
        // Fortran MOD truncates toward zero; Rust's % (rem) does the same.
        // rem_euclid would give the positive remainder, which differs by 4π for negative APOLY.
        area0 = fourpi - aslice - apoly % fourpi;

        if cds_debug_enabled() {
            println!("CDS_DEBUG| dareal k={} Phase E done: nfree={} npoly={} aslice={:.6e} apoly={:.6e} area0={:.6e}",
                k, nfree, npoly, aslice, apoly, area0);
            // Print per-column gradient accumulators (the components of darea)
            for i in 0..=ncross {
                let has_nonzero = (0..3).any(|dir|
                    ws.dca_slc[[dir, i]].abs() > 1e-20 || ws.dca_ply[[dir, i]].abs() > 1e-20
                );
                if has_nonzero {
                    print!("CDS_DEBUG|   darea comps i={}: dca_slc=[", i);
                    for dir in 0..3 { print!(" {:.6e}", ws.dca_slc[[dir, i]]); }
                    print!(" ] dca_ply=[");
                    for dir in 0..3 { print!(" {:.6e}", ws.dca_ply[[dir, i]]); }
                    print!(" ] dca_odd=[");
                    for dir in 0..3 { print!(" {:.6e}", ws.dca_odd[[dir, i]]); }
                    println!(" ]");
                }
            }
        }

        // ---- assemble darea (Fortran lines 2160-2163) ----
        // DAREA(:,I) = −DCASLC(:,I) − DCAPLY(:,I)  for I = 0..NCROSS
        darea.resize(3 * (ncross + 1), 0.0);
        for i in 0..=ncross {
            for dir in 0..3 {
                darea[3 * i + dir] = -ws.dca_slc[[dir, i]] - ws.dca_ply[[dir, i]];
            }
        }
        return (area0, ncross, nc, darea);
    }
}

impl DarealWs {
    fn new(ncross: usize) -> Self {
        DarealWs {
            stheta: vec![0.0; ncross],
            ctheta: MatrixFull::new([ncross, ncross], 0.0),
            conect: MatrixBool::new(ncross),
            cosn: MatrixFull::new([3, ncross], 0.0),
            dcosn: (0..ncross).map(|_| MatrixFull::new([3, 3], 0.0)).collect(),
            dcteta: [
                MatrixFull::new([ncross, ncross], 0.0),
                MatrixFull::new([ncross, ncross], 0.0),
                MatrixFull::new([ncross, ncross], 0.0),
            ],
            dsteta: MatrixFull::new([3, ncross], 0.0),
            // Phase E
            ncnct: MatrixFull::new([2 * (ncross - 1) + 1, ncross], 0),
            cosn_ij: (0..ncross).map(|_| MatrixFull::new([3, ncross], 0.0)).collect(),
            sit: vec![0.0; ncross],
            work: vec![0.0; 2 * ncross.max(1)],
            work_buf: vec![0.0; ncross],
            // gradient (unused)
            diwork: MatrixFull::new([3, 2 * ncross.max(1)], 0.0),
            djwork: MatrixFull::new([3, 2 * ncross.max(1)], 0.0),
            dkwork: MatrixFull::new([3, 2 * ncross.max(1)], 0.0),
            d0work: MatrixFull::new([3, 2 * ncross.max(1)], 0.0),
            dw_swap: vec![0.0f64; 3 * ncross.max(1)],
            dca_slc: MatrixFull::new([3, ncross + 1], 0.0),
            dca_ply: MatrixFull::new([3, ncross + 1], 0.0),
            dca_odd: MatrixFull::new([3, ncross + 1], 0.0),
            dicosn: (0..ncross).map(|_| MatrixFull::new([3 * ncross.max(1), 3], 0.0)).collect(),
            djcosn: (0..ncross).map(|_| MatrixFull::new([3 * ncross.max(1), 3], 0.0)).collect(),
            dctetr: vec![0.0; ncross],
            dstetr: vec![0.0; ncross],
            dsitr: vec![0.0; ncross],
        }
    }
}

// ============================================================================
//  6. CDS Energy + Gradient Driver
// ============================================================================

/// Compute the total CDS energy and gradient.
///
/// Workflow:
/// 1. Build interatomic distance matrix `rlio` (Å) and unit-vector matrices `urlio[3]`
/// 2. [`smx_cds`] → effective surface tensions σ_k^eff and gradients `dsts[3]`
/// 3. For each atom k: [`dareal`] → Ω_k → A_k = Ω_k·R_k²
/// 4. `E_CDS = Σ_k A_k·(σ_k+cssigm)·0.001` [kcal/mol]
/// 5. `∂E/∂X = Σ_j[(σ_j+cssigm)·∂A_j/∂X + A_j·∂σ_j/∂X]·0.001` → Hartree/Bohr
///
/// # Returns
/// - `cdst_kcal`: total CDS energy (kcal/mol)
/// - `tarea`: total SASA (Å²)
/// - `dcds`: CDS gradient `[nat][3]` (Hartree/Bohr)
fn cds_eg(
    cssigm: f64, nat: usize,
    coords: &[[f64; 3]], atomic_numbers: &[usize],
    sigma: &[f64; 151], hsigma: &[f64; 151], rad: &[f64],
) -> (f64, f64, Vec<[f64; 3]>) {
    const TO_ANGS: f64 = BOHR;    // Bohr → Å
    const TO_KCAL: f64 = HARTREE2KCAL;       // Hartree → kcal/mol

    // ---- Build rlio (pairwise distances, Å) and urlio (unit vectors) ----
    let ncot = nat * (nat + 1) / 2;
    let mut rlio = vec![0.0f64; ncot];
    let mut urlio = [
        MatrixFull::new([nat, nat], 0.0),   // û_{j→i} x-component
        MatrixFull::new([nat, nat], 0.0),   // y-component
        MatrixFull::new([nat, nat], 0.0),   // z-component
    ];

    for i in 0..nat {
        for j in 0..=i {
            let idx = ij0(i, j);
            if i == j { rlio[idx] = 0.0; continue; }
            let dx = (coords[i][0] - coords[j][0]) * TO_ANGS;
            let dy = (coords[i][1] - coords[j][1]) * TO_ANGS;
            let dz = (coords[i][2] - coords[j][2]) * TO_ANGS;
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            rlio[idx] = r;
            // û_{j→i} = (X_i−X_j)/r,  û_{i→j} = −û_{j→i}
            urlio[0][[i, j]] = -dx / r; urlio[1][[i, j]] = -dy / r; urlio[2][[i, j]] = -dz / r;
            urlio[0][[j, i]] =  dx / r; urlio[1][[j, i]] =  dy / r; urlio[2][[j, i]] =  dz / r;
        }
    }

    // ---- Effective surface tensions ----
    let (sts, dsts) = smx_cds(atomic_numbers, sigma, hsigma, nat, &rlio, &urlio);

    // ---- SASA per atom via DAREAL ----
    let mut cdst_kcal = 0.0f64;
    let mut tarea = 0.0f64;
    let mut area_atom = vec![0.0f64; nat];            // A_k (Å²)
    let mut cd_sa = vec![0.0f64; nat];                 // per-atom CDS (kcal/mol)
    let mut datar = [
        MatrixFull::new([nat, nat], 0.0),              // ∂A/∂X_x
        MatrixFull::new([nat, nat], 0.0),              // ∂A/∂X_y
        MatrixFull::new([nat, nat], 0.0),              // ∂A/∂X_z
    ];

    for k in 0..nat {
        let (area0, ncross, nc, darea) = dareal(nat, k, rad, &rlio, &urlio);
        let ras2 = rad[k] * rad[k];                    // R_k²
        area_atom[k] = area0 * ras2;                   // A_k = Ω_k · R_k²
        cdst_kcal += area_atom[k] * (sts[k] + cssigm) * 0.001; // 0.001: cal→kcal
        tarea += area_atom[k];
        cd_sa[k] = area_atom[k] * (sts[k] + cssigm) * 0.001;

        // ---- Debug: darea per-atom ----
        if cds_debug_enabled() && ncross > 0 {
            println!("CDS_DEBUG| darea k={} ncross={} (∂Ω/∂X, sr⁻¹/Bohr) [len={}]",
                k, ncross, darea.len());
            for l in 0..=ncross {
                let j = if l == 0 { k } else { nc[l] };
                println!("CDS_DEBUG|   darea[l={}→atom={}] = {:.12e} {:.12e} {:.12e}",
                    l, j,
                    darea[3 * l], darea[3 * l + 1], darea[3 * l + 2]);
            }
        }

        // ∂A/∂X = R_k² · ∂Ω/∂X — accumulated into datar matrix
        // darea[l] is ∂Ω/∂X_{center} for l=0, ∂Ω/∂X_{neighbor_l} for l>0
        let expected_len = 3 * (ncross + 1);
        if ncross > 0 && darea.len() >= expected_len {
            for l in 0..=ncross {
                let j = if l == 0 { k } else { nc[l] };
                // Guard against malformed nc entries
                if j >= nat {
                    if cds_debug_enabled() {
                        eprintln!("CDS_DEBUG| WARNING: dareal returned bad nc[{}]={} (nat={}), skipping",
                            l, j, nat);
                    }
                    continue;
                }
                for dir in 0..3 {
                    datar[dir][[j, k]] += darea[dir + 3 * l] * ras2;
                }
            }
        } else if ncross > 0 {
            // darea/ncross mismatch — indicates a bug in dareal return path
            if cds_debug_enabled() {
                eprintln!("CDS_DEBUG| WARNING: dareal k={} ncross={} but darea.len={} (expected {}), skipping gradient accumulation",
                    k, ncross, darea.len(), expected_len);
            }
        }
    }

    // ---- Debug: datar matrix (∂A/∂X per atom pair) ----
    if cds_debug_enabled() {
        println!("CDS_DEBUG| datar[dir][iat,j] — ∂A_j/∂X_{{iat,dir}} (Å²/Bohr):");
        for dir in 0..3 {
            let ax = ["x", "y", "z"][dir];
            for iat in 0..nat {
                for j in 0..nat {
                    let v = datar[dir][[iat, j]];
                    if v.abs() > 1e-16 {
                        println!("CDS_DEBUG|   datar[{iat},{j}].{ax} = {:.12e}", v);
                    }
                }
            }
        }
        // Debug: dsts matrix (∂σ/∂X from smx_cds)
        let has_dsts = (0..3usize).any(|dir|
            (0..nat).any(|iat| (0..nat).any(|j| dsts[dir][[iat, j]].abs() > 1e-20))
        );
        if has_dsts {
            println!("CDS_DEBUG| dsts[dir][iat,j] — ∂σ_j/∂X_{{iat,dir}} (cal/mol/Å²/Å):");
            for dir in 0..3 {
                let ax = ["x", "y", "z"][dir];
                for iat in 0..nat {
                    for j in 0..nat {
                        let v = dsts[dir][[iat, j]];
                        if v.abs() > 1e-20 {
                            println!("CDS_DEBUG|   dsts[{iat},{j}].{ax} = {:.12e}", v);
                        }
                    }
                }
            }
        } else {
            println!("CDS_DEBUG| dsts — all zero (no sigma gradient contribution)");
        }
        // Debug: per-atom contribution breakdown to dcds
        println!("CDS_DEBUG| Gradient contribution breakdown (kcal/mol/Å → Hartree/Bohr):");
        for iat in 0..nat {
            for dir in 0..3 {
                let ax = ["x", "y", "z"][dir];
                let mut sasa_term = 0.0f64;   // Σ_j σ_j · ∂A_j/∂X_iat
                let mut sig_term = 0.0f64;    // Σ_j A_j · ∂σ_j/∂X_iat
                for j in 0..nat {
                    sasa_term += (sts[j] + cssigm) * datar[dir][[iat, j]] * 0.001;
                    sig_term += dsts[dir][[iat, j]] * area_atom[j] * 0.001;
                }
                let total_kcal_a = sasa_term + sig_term;
                let total_h_b = total_kcal_a / TO_KCAL * TO_ANGS;
                if total_kcal_a.abs() > 1e-20 {
                    println!("CDS_DEBUG|   dcds[{iat}].{ax}: sasa={:.6e} sigma={:.6e} sum(kcal/mol/Å)={:.6e} → {:.6e} Hartree/Bohr",
                        sasa_term, sig_term, total_kcal_a, total_h_b);
                }
            }
        }
    }

    // ---- CDS gradient ----
    // ∂E/∂X_{iat} = Σ_j [(σ_j+cssigm)·∂A_j/∂X_{iat} + A_j·∂σ_j/∂X_{iat}] · 0.001
    // kcal/(mol·Å) → Hartree/Bohr via /TO_KCAL * TO_ANGS
    let mut dcds = vec![[0.0f64; 3]; nat];
    for iat in 0..nat {
        for dir in 0..3 {
            let mut dcds_dir = 0.0f64;
            for j in 0..nat {
                dcds_dir += (sts[j] + cssigm) * datar[dir][[iat, j]] * 0.001;
                dcds_dir += dsts[dir][[iat, j]] * area_atom[j] * 0.001;
            }
            dcds[iat][dir] = dcds_dir / TO_KCAL * TO_ANGS;
        }
    }

    // ---- Debug output ----
    if cds_debug_enabled() {
        println!("CDS_DEBUG| ====== cds_eg intermediates ======");
        cds_print_scalar("cssigm", cssigm);
        println!("CDS_DEBUG| nat = {}", nat);
        // Effective radii (Å)
        for k in 0..nat {
            println!("CDS_DEBUG| rad[{}] = {:.12e} (Z={})", k, rad[k], atomic_numbers[k]);
        }
        // Distance matrix rlio (Å) — upper triangular
        println!("CDS_DEBUG| rlio distances (Ang):");
        for i in 0..nat {
            for j in 0..=i {
                if i != j { println!("CDS_DEBUG|   r[{}][{}] = {:.12e}", i, j, rlio[ij0(i, j)]); }
            }
        }
        cds_print_vec("sts (effective sigma_k, cal/mol/A^2)", &sts);
        cds_print_vec("area_atom (SASA per atom, A^2)", &area_atom);
        cds_print_vec("cd_sa (per-atom CDS, kcal/mol)", &cd_sa);
        // Compact per-atom line: Z sigma area → cd_sa
        print!("CDS_DEBUG| per-atom: ");
        for k in 0..nat {
            print!(" [{}]Z={} σ={:.2e} A={:.4}→{:.4}",
                k, atomic_numbers[k], sts[k], area_atom[k], cd_sa[k]);
        }
        println!();
        cds_print_scalar("cdst_kcal (total CDS, kcal/mol)", cdst_kcal);
        cds_print_scalar("tarea (total SASA, A^2)", tarea);
        cds_print_grad("dcds (Hartree/Bohr)", &dcds);
    }

    (cdst_kcal, tarea, dcds)
}

// ============================================================================
//  7. Public API
// ============================================================================

/// Compute the SMD CDS (Cavitation-Dispersion-Solvent structure) energy and gradient.
///
/// # Arguments
/// - `atomic_numbers`: atomic numbers Z ∈ [1, 102]
/// - `coords`: Cartesian coordinates `[nat][3]` in **Bohr**
/// - `icds`: solvent type — 1 = water, 2 = non-aqueous
/// - `solvent_descriptors`: `[n, n25, α, β, γ, ε, φ, ψ]` (only used when icds=2)
/// - `smd_cavity_radii`: SMD radii scheme:
///   - `Bondi` (default): all elements use `BONDI[z] + 0.4` (mnsol.F `VDWRAD`, aligned with PySCF)
///   - `BondiUff`: the 11 eq.16 elements keep `BONDI[z]`; other elements use
///     `BONDI_UFF_RADII[z]*BOHR` (reference snapshot table, bohr→Å); zero entries are
///     defensive only (table covers Z=1..103) and fall back to `BONDI[z]`
///
/// # Returns
/// - `gcds`: CDS free energy (Hartree)
/// - `tarea`: total solvent-accessible surface area (Å²)
/// - `dcds`: CDS energy gradient `[nat][3]` (Hartree/Bohr)
pub fn compute_cds(
    atomic_numbers: &[usize],
    coords: &[[f64; 3]],
    icds: i32,
    solvent_descriptors: &[f64; 8],
    smd_cavity_radii: SmdCavityRadii,
) -> (f64, f64, Vec<[f64; 3]>) {
    let nat = atomic_numbers.len();
    let mut sigma = [0.0f64; 151];
    let mut hsigma = [0.0f64; 151];

    let cssigm = if icds == 1 {
        smd_cds_aq(&mut sigma, &mut hsigma)
    } else {
        smd_cds_naq(
            &mut sigma, &mut hsigma,
            solvent_descriptors[2], // α (sola)
            solvent_descriptors[3], // β (solb)
            solvent_descriptors[6], // φ (solc)
            solvent_descriptors[4], // γ (solg)
            solvent_descriptors[7], // ψ (solh)
            solvent_descriptors[0], // n (soln)
        )
    };

    // Effective SASA sphere radius per atom: `rad[k] = R_base(Z_k) + 0.4 Å` (solvent probe).
    // Bondi scheme: R_base = BONDI[Z] (mnsol.F VDWRAD, legacy).
    // BondiUff scheme: eq.16 elements keep BONDI[Z]; others use BONDI_UFF_RADII[Z]*BOHR
    // (bohr→Å via REST BOHR = 0.529177 Å/bohr, multiplication); zero entry is defensive only.
    let rad: Vec<f64> = atomic_numbers.iter()
        .map(|&z| {
            let base = match smd_cavity_radii {
                SmdCavityRadii::Bondi => BONDI[z.min(102)],
                SmdCavityRadii::BondiUff => {
                    if natcnv(z) != 0 {
                        BONDI[z.min(102)]
                    } else if z < data::BONDI_UFF_RADII.len() && data::BONDI_UFF_RADII[z] > 0.0 {
                        data::BONDI_UFF_RADII[z] * BOHR
                    } else {
                        BONDI[z.min(102)]
                    }
                }
            };
            base + 0.4
        })
        .collect();

    let (gcds_kcal, tarea, dcds) = cds_eg(
        cssigm, nat, coords, atomic_numbers, &sigma, &hsigma, &rad,
    );

    // ---- Debug output ----
    if cds_debug_enabled() {
        println!("CDS_DEBUG| ====== compute_cds ======");
        cds_print_scalar("icds", icds as f64);
        cds_print_sigma151("sigma", &sigma);
        cds_print_sigma151("hsigma", &hsigma);
        cds_print_scalar("cssigm", cssigm);
        cds_print_vec("effective_radii (Bondi+0.4, Ang)", &rad);
        // solvent descriptors
        println!("CDS_DEBUG| solvent_descriptors = {:?}", solvent_descriptors);
        // Final outputs
        cds_print_scalar("gcds_kcal (CDS energy, kcal/mol)", gcds_kcal);
        cds_print_scalar("tarea (total SASA, A^2)", tarea);
        cds_print_grad("dcds (CDS gradient, Hartree/Bohr)", &dcds);
        const TO_KCAL: f64 = HARTREE2KCAL;
        cds_print_scalar("gcds_hartree (CDS energy, Hartree)", gcds_kcal / TO_KCAL);
    }

    // kcal/mol → Hartree (gradient already in Hartree/Bohr from cds_eg)
    const TO_KCAL: f64 = HARTREE2KCAL;
    let gcds = gcds_kcal / TO_KCAL;

    (gcds, tarea, dcds)
}
