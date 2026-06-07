//! SMD CDS (Cavitation-Dispersion-Solvent structure) energy and gradient.
//!
//! Translated from PySCF's Fortran source:
//!   pyscf/lib/solvent/mnsol.F (originally from NWChem src/solvation/)
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
//! | `dsts[dir,iat,k]` | ∂σ_k/∂X_{iat,dir} | cal/(mol·Å³) |
//! | `area0` (Ω_k) | Accessible solid angle of sphere K | steradians (sr) |
//! | `area_atom[k]` (A_k) | SASA = Ω_k · R_k² | Å² |
//! | `datar[dir,iat,k]` | ∂A_k/∂X_{iat,dir} | Å |
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
//! | 114 | O–P bond | −9.10 |
//!
//! ## Unit Conversions
//!
//! - `TO_ANGS = 0.529177` Bohr → Å
//! - `TO_KCAL = 627.509` Hartree → kcal/mol
//! - Internal computation in Å and kcal/mol; public API input/output in Bohr and Hartree.
//!
//! ## References
//!
//! - Marenich, Cramer, Truhlar, *JPCB* 2009, 113, 6378–6396 (SMD model)
//! - Liotard, D. (1992) — DAREAL accessible solid angle algorithm
//! - Rinaldi, D. & Liotard, D. — analytical derivatives

use std::f64::consts::PI;

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
    s[1]=48.69; s[6]=129.74; s[8]=38.18; s[15]=-9.10; s[16]=9.82;
    s[34]=-8.72; s[101]=-72.95; s[103]=68.69; s[105]=-48.22;
    s[106]=121.98; s[108]=68.85; s[109]=84.10;
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
    s[6]=58.10; s[7]=32.62; s[8]=-17.56; s[13]=-18.04; s[15]=-33.17;
    s[16]=-24.31; s[34]=-35.42; s[101]=-62.05; s[103]=-15.70; s[109]=-99.76;
    s
}
const fn build_sigma_a() -> [f64; 151] {
    let mut s = [0.0f64; 151];
    s[6]=48.10; s[8]=193.06; s[103]=95.99; s[105]=-41.00; s[109]=152.20;
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
fn dot3(x: &[f64; 3], y: &[f64; 3]) -> f64 {
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
/// The zeroth-order value `σ⁰(Z_k)` comes from the `sigma` table (cal/mol/Å²).
/// Five bond-correction branches are applied (SMD-only):
/// - **H–X**: H surface tension corrected by the heavy atom it's bonded to
/// - **O–X**: O surface tension by bonded atom type (C, N, P, O)
/// - **N–C**: coordination-dependent correction distinguishing –N–C–, –N=C<, –N≡C–
/// - **C–C**: single-bond correction for each C–C pair
/// - **C–N**: correction quadratic in C–N bond count
///
/// Bond detection is automatic via [`cot_val`] — no bond topology input needed.
///
/// # Returns
/// - `sts[nat]`: effective surface tensions σ_k^eff (cal/mol/Å²)
/// - `dsts[3·nat·nat]`: ∂σ_k/∂X_{iat,dir}, layout `[dir + 3*(iat + nat*k)]`
fn smx_cds(
    atomic_numbers: &[usize], sigma: &[f64; 151], hsigma: &[f64; 151],
    nat: usize, rlio: &[f64], urlio: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut sts = vec![0.0f64; nat];
    let mut dsts = vec![0.0f64; 3 * nat * nat];
    let ncot = nat * (nat + 1) / 2;
    let mut cot = vec![0.0f64; ncot];
    let mut dcot_dr = vec![0.0f64; ncot];

    // ---- zeroth-order: σ_k = σ⁰(Z_k) ----
    for i in 0..nat {
        let z = atomic_numbers[i];
        sts[i] = sigma[z.min(150)];
    }

    /// Apply a bond correction to atom i and propagate gradients:
    ///   σ_i += COT(r_ij) · Δσ
    ///   ∂σ_i/∂X_i += Δσ · dCOT/dr · û_{j→i}
    ///   ∂σ_i/∂X_j -= Δσ · dCOT/dr · û_{j→i}
    fn add_bond_corr(
        sts: &mut [f64], dsts: &mut [f64], nat: usize,
        i: usize, j: usize, sig_val: f64, // sig_val = Δσ (cal/mol/Å²)
        rlio: &[f64], urlio: &[f64], rhld: f64, deltar: f64,
    ) {
        let r = rlio[ij0(i, j)];
        let (c, dc) = cot_val(r, rhld, deltar);
        sts[i] += c * sig_val;
        let u_ji0 = urlio[0 + 3 * (j + nat * i)];
        let u_ji1 = urlio[1 + 3 * (j + nat * i)];
        let u_ji2 = urlio[2 + 3 * (j + nat * i)];
        let d = sig_val * dc;
        dsts[0 + 3 * (i + nat * i)] += d * u_ji0;
        dsts[1 + 3 * (i + nat * i)] += d * u_ji1;
        dsts[2 + 3 * (i + nat * i)] += d * u_ji2;
        dsts[0 + 3 * (j + nat * i)] -= d * u_ji0;
        dsts[1 + 3 * (j + nat * i)] -= d * u_ji1;
        dsts[2 + 3 * (j + nat * i)] -= d * u_ji2;
    }

    // ---- H–X correction: σ_H += Σ_J COT(r_HJ) · hsigma[Z_J] ----
    // R₀ = rkkval(H_type, J_type), δ = 0.30 Å.
    // e.g. HSIGMA_AQ[6] = −60.77 → H bonded to C has reduced surface tension.
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

    // ---- O–X correction: σ_O += COT(r_OJ) · σ[bond_index] ----
    // O–O uses hardcoded R₀=1.80 Å (not from rkkval) for peroxy bonds.
    for i in 0..nat {
        if atomic_numbers[i] != 8 { continue; }
        let itpc = natcnv(8);
        for j in 0..nat {
            if i == j { continue; }
            let ntp = atomic_numbers[j];
            let jtpc = natcnv(ntp);
            let (sig_val, rhld, deltar): (f64, f64, f64) = match ntp {
                6 => (sigma[103], rkkval(itpc, jtpc), 0.30),  // σ(O–C)
                7 => (sigma[106], rkkval(itpc, jtpc), 0.30),  // σ(O–N)
                15 => (sigma[114], rkkval(itpc, jtpc), 0.30), // σ(O–P)
                8 => (sigma[104], 1.80, 0.30),                // σ(O–O), R₀=1.80
                _ => continue,
            };
            add_bond_corr(&mut sts, &mut dsts, nat, i, j, sig_val, rlio, urlio, rhld, deltar);
        }
    }

    // ---- N–C correction (coordination-dependent) ----
    // Distinguishes –N–C– (sp³), –N=C< (sp²), –N≡C– (sp) via C coordination number:
    //   C_coord_J = Σ_{K≠N,C_J} COT(r_JK; R_JK, 0.30)     (≈ 3, 2, 1)
    //   RTKKS = Σ_{C_J} COT(r_NJ; R_NC, 0.30) · (C_coord_J)²
    //   σ_N += RTKKS^1.3 · sigma[105]   (sigma[105] = −48.22 for water)
    // The power 1.3 nonlinearly enhances discrimination of multiple-bond environments.
    // N.B. DSTS (gradient) is TODO — see Fortran mnsol.F lines 1053–1128.
    for i in 0..nat {
        if atomic_numbers[i] != 7 { continue; }
        let mut rtkk_s = 0.0f64;
        for j in 0..nat {
            if i == j || atomic_numbers[j] != 6 { continue; }
            let (c_ij, _) = cot_val(rlio[ij0(i, j)], rkkval(natcnv(7), natcnv(6)), 0.30);
            if c_ij <= 0.0 { continue; }
            let mut rtkk3 = 0.0f64; // C_coord_J
            for k in 0..nat {
                if k == i || k == j { continue; }
                let rhld2 = rkkval(natcnv(atomic_numbers[k]), natcnv(6));
                let (c_jk, _) = cot_val(rlio[ij0(j, k)], rhld2, 0.30);
                rtkk3 += c_jk;
            }
            rtkk_s += c_ij * rtkk3 * rtkk3;
        }
        let dholder = rtkk_s.powf(1.3);
        sts[i] += dholder * sigma[105];
    }

    // ---- C–C single-bond correction: σ_C += Σ_{C_J} COT(r_CJ) · sigma[101] ----
    // sigma[101] = −72.95 for water (negative = shielding effect).
    for i in 0..nat {
        if atomic_numbers[i] != 6 { continue; }
        let itpc = natcnv(6);
        for j in 0..nat {
            if i == j || atomic_numbers[j] != 6 { continue; }
            let rhld = rkkval(itpc, natcnv(6));
            add_bond_corr(&mut sts, &mut dsts, nat, i, j, sigma[101], rlio, urlio, rhld, 0.30);
        }
    }

    // ---- C–N bond correction: σ_C += (RTKK_CN)² · sigma[110] ----
    // RTKK_CN = Σ_{N_J} COT(r_CJ; R_CN, 0.30). sigma[110] = 0.0 for water.
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
    }

    (sts, dsts)
}

// ============================================================================
//  5. DAREAL: Accessible Solid Angle
// ============================================================================

/// Workspace for the DAREAL accessible-solid-angle computation.
///
/// For a central sphere K overlapped by N neighbor spheres, each overlap defines a
/// spherical segment (SS) on K's surface. The half-cone angle θ_i of SS i follows
/// from the law of cosines on triangle (R_K, R_I, d_KI):
///
/// ```text
/// cos(θ_i) = (R_K² + d_KI² − R_I²) / (2·R_K·d_KI)
/// ```
///
/// SS connectivity determines how the sphere surface is partitioned.
struct DarealWs {
    stheta: Vec<f64>,        // sin(θ_i) for each SS
    ctheta: MatrixView,      // cos(θ): diagonal = cos(θ_i), off-diagonal = cos(θ_ij)
    conect: MatrixBool,      // connectivity: whether SS i and j share a free boundary
    cosn: [Vec<f64>; 3],     // û_{K→I}: unit vectors from K to each neighbor I
    dcteta: [Vec<f64>; 3],   // ∂cosθ/∂X_dir
    dsteta: [Vec<f64>; 3],   // ∂sinθ/∂X_dir
    dcosn: Vec<f64>,         // ∂û/∂X, layout [3×3×ncross²]
    // Workspace for dihedral sorting (clustered-SS path)
    work: Vec<f64>,
    diwork: [Vec<f64>; 3],
    djwork: [Vec<f64>; 3],
    dkwork: [Vec<f64>; 3],
    d0work: [Vec<f64>; 3],
    dw_swap: Vec<f64>,
    // Gradient accumulators
    dca_slc: [Vec<f64>; 3],  // ∂(slice contribution)/∂X
    dca_ply: [Vec<f64>; 3],  // ∂(polygon contribution)/∂X
    dca_odd: [Vec<f64>; 3],  // ∂(odd/even sorting contribution)/∂X
}

/// Row-major 2D view over a `Vec<f64>`.
struct MatrixView { data: Vec<f64>, n: usize }
impl MatrixView {
    fn new(n: usize) -> Self { MatrixView { data: vec![0.0; n*n], n } }
    #[inline] fn at(&self, i: usize, j: usize) -> f64 { self.data[i * self.n + j] }
    #[inline] fn set(&mut self, i: usize, j: usize, v: f64) { self.data[i * self.n + j] = v; }
}

/// Boolean matrix for SS connectivity.
struct MatrixBool { data: Vec<bool>, n: usize }
impl MatrixBool {
    fn new(n: usize) -> Self { MatrixBool { data: vec![false; n*n], n } }
    #[inline] fn at(&self, i: usize, j: usize) -> bool { self.data[i * self.n + j] }
    #[inline] fn set(&mut self, i: usize, j: usize, v: bool) { self.data[i * self.n + j] = v; }
}

/// Compute the accessible solid angle Ω_k of sphere K via the DAREAL algorithm
/// (Liotard, 1992). Ω_k is the solid angle of the unit sphere region not occluded
/// by any neighbor sphere's cap. The per-atom SASA follows as `A_k = Ω_k · R_k²`.
///
/// # Returns
/// - `area0`: accessible solid angle Ω_k (steradians)
/// - `ncross`: number of overlapping neighbor spheres
/// - `nc`: neighbor indices, nc[0] = k, nc[1..ncross] = neighbor atoms
/// - `darea`: analytic gradient ∂Ω_k/∂X_{nc[l]}, layout `[3, ncross+1]`
///
/// **Current limitation:** the clustered-SS path (free intersections, dihedral sorting,
/// polygon tracing) is not yet implemented. When nclust > 0, returns an approximation
/// Ω_k ≈ 4π − A_slice (omitting polygon corrections, ~10% error for multi-atom molecules).
fn dareal(
    nat: usize, k: usize, rad: &[f64],
    rlio: &[f64], urlio: &[f64],
) -> (f64, usize, Vec<usize>, Vec<f64>) {
    let twopi = 2.0 * PI;
    let fourpi = 4.0 * PI;
    let epsi: f64 = 1.0e-11;
    let mxss = 2 * nat + 1;

    let rk = rad[k];
    if rk <= 0.0 { return (0.0, 0, vec![0], vec![0.0; 3]); }

    let mut nc = vec![0usize; nat + 1];
    let mut ncross: usize;
    let mut area0 = fourpi;
    let mut darea = vec![0.0f64; 3];

    let mut restart;
    loop {
        restart = false;
        ncross = 0;
        nc[0] = k;
        let epsk = epsi * rk;

        // ---- Phase A: overlap detection ----
        for i in 0..nat {
            if i == k || rad[i] <= 0.0 { continue; }
            if rk + rad[i] - rlio[ij0(i, k)] < epsk { continue; }     // no overlap
            if rlio[ij0(i, k)] - (rk - rad[i]).abs() < epsk {
                if rk <= rad[i] {
                    return (0.0, ncross, vec![0; nat+1], vec![0.0; 3]); // K embedded in I
                }
            } else {
                ncross += 1;
                nc[ncross] = i;                                         // partial overlap
            }
        }

        if ncross == 0 {
            return (area0, ncross, nc, vec![0.0; 3 * (ncross + 1)]);
        }

        let mut ws = DarealWs::new(ncross);

        // ---- Phase B: SS data initialization ----
        // cos(θ_i) = (1/(2R_K)) · [d_KI + (R_K² − R_I²)/d_KI]
        let rk_inv = 0.5 / rk;
        let rk2 = rk * rk;

        for i in 0..ncross {
            let li = nc[i + 1];
            let idx_ki = ij0(li, k);
            let rik_inv = 1.0 / rlio[idx_ki];

            let ci = rk_inv * (rlio[idx_ki] + (rk2 - rad[li] * rad[li]) * rik_inv);
            ws.ctheta.set(i, i, ci);
            ws.stheta[i] = (1.0 - ci * ci).sqrt();

            for dir in 0..3 {
                ws.cosn[dir][i * ncross + i] = urlio[dir + 3 * (k + nat * li)];
            }

            // Derivatives: ∂cosθ/∂X, ∂sinθ/∂X, ∂cosn/∂X
            let x_val = -ci / ws.stheta[i];
            let drctht = rk_inv * (1.0 - (rk2 - rad[li] * rad[li]) * rik_inv * rik_inv);
            for dir in 0..3 {
                let cosni_val = ws.cosn[dir][i * ncross + i];
                ws.dcteta[dir][i * ncross + i] = drctht * cosni_val;
                ws.dsteta[dir][i * ncross + i] = x_val * ws.dcteta[dir][i * ncross + i];

                // ∂(û_{KI})_p / ∂X_q = (−û_p·û_q + δ_pq) / d_KI
                let cosni = cosni_val * rik_inv;
                for jdir in 0..3 {
                    let cosnij = -cosni * ws.cosn[jdir][i * ncross + i];
                    let didx = dir + 3 * (jdir + 3 * (i + ncross * i));
                    ws.dcosn[didx] = cosnij;
                    let djdx = jdir + 3 * (dir + 3 * (i + ncross * i));
                    ws.dcosn[djdx] = cosnij;
                }
                let ddiag = dir + 3 * (dir + 3 * (i + ncross * i));
                ws.dcosn[ddiag] += rik_inv;
            }
        }

        // ---- Phase C: connectivity ----
        // Two SS i,j are connected if their intersection line lies on the sphere
        // surface (not buried under a third SS). Connectivity test uses
        // t_ij = cosθ_ij − cosθ_i·cosθ_j.
        for ii in 1..ncross {
            for jj in 0..ii {
                if ws.conect.at(jj, jj) { continue; }

                let cisj = ws.ctheta.at(ii, ii) * ws.stheta[jj];
                let sicj = ws.stheta[ii] * ws.ctheta.at(jj, jj);
                let sisj = ws.stheta[ii] * ws.stheta[jj];

                let cij = dot3(
                    &[ws.cosn[0][ii*ncross+ii], ws.cosn[1][ii*ncross+ii], ws.cosn[2][ii*ncross+ii]],
                    &[ws.cosn[0][jj*ncross+jj], ws.cosn[1][jj*ncross+jj], ws.cosn[2][jj*ncross+jj]],
                );
                ws.ctheta.set(jj, ii, cij);
                ws.ctheta.set(ii, jj, cij);

                for dir in 0..3 {
                    let di = dot3(
                        &[ws.dcosn[dir + 3*(0 + 3*(ii + ncross*ii))],
                          ws.dcosn[dir + 3*(1 + 3*(ii + ncross*ii))],
                          ws.dcosn[dir + 3*(2 + 3*(ii + ncross*ii))]],
                        &[ws.cosn[0][jj*ncross+jj], ws.cosn[1][jj*ncross+jj], ws.cosn[2][jj*ncross+jj]],
                    );
                    ws.dcteta[dir][ii * ncross + jj] = di;
                    let dj = dot3(
                        &[ws.cosn[0][ii*ncross+ii], ws.cosn[1][ii*ncross+ii], ws.cosn[2][ii*ncross+ii]],
                        &[ws.dcosn[dir + 3*(0 + 3*(jj + ncross*jj))],
                          ws.dcosn[dir + 3*(1 + 3*(jj + ncross*jj))],
                          ws.dcosn[dir + 3*(2 + 3*(jj + ncross*jj))]],
                    );
                    ws.dcteta[dir][jj * ncross + ii] = dj;
                }

                let tij = cij - ws.ctheta.at(ii, ii) * ws.ctheta.at(jj, jj);

                if tij > sisj - epsi * (cisj - sicj).abs() {
                    if ws.ctheta.at(jj, jj) > ws.ctheta.at(ii, ii) {
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
        // Cap area on unit sphere: A_cap = 2π(1 − cosθ).
        // A_slice = 2π · Σ_isolated (1 − cosθ_i)
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
            a_slice += 1.0 - ws.ctheta.at(i, i);
        }
        a_slice *= twopi;

        if nclust == 0 {
            area0 = fourpi - a_slice;
            darea.resize(3 * (ncross + 1), 0.0);
            return (area0, ncross, nc, darea);
        }

        // ---- Phase E: clustered SS (NOT YET IMPLEMENTED) ----
        // The full algorithm requires:
        // 1. Free-intersection computation (PIJ/PJI)
        // 2. Intersection classification (free vs buried)
        // 3. Dihedral-angle sorting via atan2
        // 4. Polygon vertex tracing
        // 5. Final assembly: Ω = 4π − A_slice − (A_poly + (N_poly−N_free)·2π) mod 4π
        //
        // Currently returns the isolated-SS approximation:
        area0 = fourpi - a_slice;
        darea.resize(3 * (ncross + 1), 0.0);
        return (area0, ncross, nc, darea);
    }
}

impl DarealWs {
    fn new(ncross: usize) -> Self {
        let n2 = ncross * ncross;
        let n3 = 3 * ncross;
        DarealWs {
            stheta: vec![0.0; ncross],
            ctheta: MatrixView::new(ncross),
            conect: MatrixBool::new(ncross),
            cosn: [vec![0.0; n2], vec![0.0; n2], vec![0.0; n2]],
            dcteta: [vec![0.0; n2], vec![0.0; n2], vec![0.0; n2]],
            dsteta: [vec![0.0; n3], vec![0.0; n3], vec![0.0; n3]],
            dcosn: vec![0.0; 3 * 3 * n2],
            work: vec![],
            diwork: [vec![], vec![], vec![]],
            djwork: [vec![], vec![], vec![]],
            dkwork: [vec![], vec![], vec![]],
            d0work: [vec![], vec![], vec![]],
            dw_swap: vec![],
            dca_slc: [vec![], vec![], vec![]],
            dca_ply: [vec![], vec![], vec![]],
            dca_odd: [vec![], vec![], vec![]],
        }
    }
}

// ============================================================================
//  6. CDS Energy + Gradient Driver
// ============================================================================

/// Compute the total CDS energy and gradient.
///
/// Workflow:
/// 1. Build interatomic distance matrix `rlio` (Å) and unit-vector matrix `urlio`
/// 2. [`smx_cds`] → effective surface tensions σ_k^eff and their gradients
/// 3. For each atom k: [`dareal`] → accessible solid angle Ω_k → SASA A_k = Ω_k · R_k²
/// 4. `E_CDS = Σ_k A_k · (σ_k^eff + cssigm) · 0.001` [kcal/mol]
/// 5. `∂E/∂X = Σ_j [(σ_j+cssigm)·∂A_j/∂X + A_j·∂σ_j/∂X] · 0.001` [→ Hartree/Bohr]
///
/// Internal computation uses Å and kcal/mol. Unit conversion to Hartree/Bohr
/// is applied at output.
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
    const TO_ANGS: f64 = 0.52917724924;    // Bohr → Å
    const TO_KCAL: f64 = 627.509451;       // Hartree → kcal/mol

    // ---- Build rlio (pairwise distances, Å) and urlio (unit vectors) ----
    let ncot = nat * (nat + 1) / 2;
    let mut rlio = vec![0.0f64; ncot];            // |X_i − X_j|, upper-triangular
    let mut urlio = vec![0.0f64; 3 * nat * nat];  // û_{j→i}, layout [dir + 3*(i + nat*j)]

    for i in 0..nat {
        for j in 0..=i {
            let idx = ij0(i, j);
            if i == j {
                rlio[idx] = 0.0;
                continue;
            }
            let dx = (coords[i][0] - coords[j][0]) * TO_ANGS;
            let dy = (coords[i][1] - coords[j][1]) * TO_ANGS;
            let dz = (coords[i][2] - coords[j][2]) * TO_ANGS;
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            rlio[idx] = r;
            urlio[0 + 3 * (i + nat * j)] = -dx / r;
            urlio[1 + 3 * (i + nat * j)] = -dy / r;
            urlio[2 + 3 * (i + nat * j)] = -dz / r;
            urlio[0 + 3 * (j + nat * i)] = dx / r;
            urlio[1 + 3 * (j + nat * i)] = dy / r;
            urlio[2 + 3 * (j + nat * i)] = dz / r;
        }
    }

    // ---- Effective surface tensions ----
    let (sts, dsts) = smx_cds(atomic_numbers, sigma, hsigma, nat, &rlio, &urlio);

    // ---- SASA per atom via DAREAL ----
    let mut cdst_kcal = 0.0f64;
    let mut tarea = 0.0f64;
    let mut area_atom = vec![0.0f64; nat];            // A_k (Å²)
    let mut cd_sa = vec![0.0f64; nat];                 // per-atom CDS (kcal/mol)
    let mut datar = vec![0.0f64; 3 * nat * nat];       // ∂A_k/∂X_{iat,dir}

    for k in 0..nat {
        let (area0, ncross, nc, darea) = dareal(nat, k, rad, &rlio, &urlio);
        let ras2 = rad[k] * rad[k];                    // R_k²
        area_atom[k] = area0 * ras2;                   // A_k = Ω_k · R_k²
        cdst_kcal += area_atom[k] * (sts[k] + cssigm) * 0.001; // 0.001: cal→kcal
        tarea += area_atom[k];
        cd_sa[k] = area_atom[k] * (sts[k] + cssigm) * 0.001;

        // Propagate darea (∂Ω/∂X) → datar (∂A/∂X): ∂A/∂X = R_k² · ∂Ω/∂X
        for l in 0..=ncross {
            let j = if l == 0 { k } else { nc[l] };
            for dir in 0..3 {
                datar[dir + 3 * (j + nat * k)] += darea[dir + 3 * l] * ras2;
            }
        }
    }

    // ---- CDS gradient ----
    // ∂E/∂X_{iat} = Σ_j [(σ_j+cssigm)·∂A_j/∂X_{iat} + A_j·∂σ_j/∂X_{iat}] · 0.001
    // Term 1: surface tension × area derivative
    // Term 2: area × surface tension derivative
    // Final conversion: kcal/(mol·Å) → Hartree/Bohr via /TO_KCAL * TO_ANGS
    let mut dcds = vec![[0.0f64; 3]; nat];
    for iat in 0..nat {
        for dir in 0..3 {
            let mut dcds_dir = 0.0f64;
            for j in 0..nat {
                dcds_dir += (sts[j] + cssigm) * datar[dir + 3 * (iat + nat * j)] * 0.001;
                dcds_dir += dsts[dir + 3 * (iat + nat * j)] * area_atom[j] * 0.001;
            }
            dcds[iat][dir] = dcds_dir / TO_KCAL * TO_ANGS;
        }
    }

    (cdst_kcal, tarea, dcds)
}

// ============================================================================
//  7. Public API
// ============================================================================

/// Compute the SMD CDS (Cavitation-Dispersion-Solvent structure) energy and gradient.
///
/// This is the main entry point for CDS calculations.
///
/// # Arguments
/// - `atomic_numbers`: atomic numbers Z ∈ [1, 102]
/// - `coords`: Cartesian coordinates `[nat][3]` in **Bohr**
/// - `icds`: solvent type — 1 = water, 2 = non-aqueous
/// - `solvent_descriptors`: `[n, n25, α, β, γ, ε, φ, ψ]` (only used when icds=2)
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

    // Effective radius = Bondi vdW + 0.4 Å solvent probe
    let rad: Vec<f64> = atomic_numbers.iter()
        .map(|&z| if z < 103 { BONDI[z] } else { 0.0 } + 0.4)
        .collect();

    let (gcds_kcal, tarea, dcds) = cds_eg(
        cssigm, nat, coords, atomic_numbers, &sigma, &hsigma, &rad,
    );

    // kcal/mol → Hartree (gradient already in Hartree/Bohr from cds_eg)
    const TO_KCAL: f64 = 627.509451;
    let gcds = gcds_kcal / TO_KCAL;

    (gcds, tarea, dcds)
}
