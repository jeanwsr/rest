//! SHARC non-adiabatic dynamics backend for the GW-BSE module.
//!
//! This module implements the two quantities that a surface-hopping
//! (SHARC) trajectory needs beyond the BSE excitation energies and their
//! analytic nuclear gradients:
//!
//! * the **inter-state overlap matrix** `S_IJ = <Psi_I(R0)|Psi_J(R1)>`
//!   between two consecutive geometries, and
//! * the **non-adiabatic coupling vectors** `d_IJ = <Psi_I|d/dR|Psi_J>`.
//!
//! # Wavefunction model
//!
//! A BSE (TDDFT) state is expanded in singly excited determinants on top of
//! the (GW corrected) Kohn-Sham reference,
//!
//! ```text
//! |Psi_I> = sum_ia [ X_I,ia |Phi_i^a> + Y_I,ia |Phi^a_i> ],
//! ```
//!
//! so all overlaps reduce to overlaps of determinants built from two
//! different, non-orthogonal sets of molecular orbitals.  With
//!
//! ```text
//! S_oo = C0^T S^AO C1  (occupied-occupied block of the MO overlap),
//! ```
//!
//! the ground-state overlap is `det(S_oo)`, the ground-excited overlap is
//! `det(S_oo) * (S_oo^{-1} S_ov)_{ib}`, and the excited-excited overlap
//! (`i,j` occupied, `a,b` virtual, all indices in the *full* reference
//! manifolds) is
//!
//! ```text
//! <Phi_i^a|Phi_j^b> = det(S_oo) * [ (1+s1)(1+s4) - s2 s3 ]
//! s1 = (S_vo S_oo^{-1})_{a i} - 1
//! s3 = (S_oo^{-1})_{j i}
//! s2 = (R - S_vo - S_ov + S_oo)_{abij} + dl * s1
//! s4 = (S_oo^{-1} S_ov)_{j b} - 1 + dl * s3
//! dl = (S_vv - S_vo - S_ov + S_oo)_{abij}
//! R  = S_vo S_oo^{-1} S_ov
//! ```
//!
//! This is the exact determinant overlap (verified in the unit tests below);
//! only the amplitudes are restricted to the BSE active window
//! (`occ_start..occ_start+nocc` x `vir_start..vir_start+nvir`), the
//! determinants themselves use the complete reference space.
//!
//! For a non-TDA (RPA) state the overlap carries the usual
//! `X_I X_J - Y_I Y_J` metric structure.
//!
//! # Cross-geometry AO overlaps
//!
//! `S^AO(R0,R1)` is evaluated as a *cross* integral between two libcint
//! molecule objects that share the basis and differ only in the nuclear
//! coordinates (`Molecule::int_cross`).  The derivative with respect to the
//! ket geometry -- needed for the NAC -- comes from the cross
//! `int1e_ipovlp` integral, see [`Molecule::int_cross_ipovlp`].
//!
//! # Approximation of the NAC
//!
//! The NAC is obtained by differentiating the *same* overlap expression with
//! respect to the ket geometry, i.e. with the BSE amplitudes held fixed:
//!
//! ```text
//! d_IJ = d/dR' <Psi_I(R)|Psi_J(R')> |_{R'=R}
//! ```
//!
//! This is the standard "frozen-amplitude" (wavefunction-overlap) coupling:
//! it contains the complete orbital response (the MO coefficients follow the
//! nuclei) but not the geometric response of the CI/BSE amplitudes.  It is
//! exactly the coupling that SHARC's overlap-based / local-diabatization
//! propagator uses, and it is anti-Hermitised before it is written out.  The
//! matrix is *not* the Hellmann-Feynman `X_I^T (dH/dR) X_J / (Omega_J-Omega_I)`
//! coupling of the BSE Hamiltonian.

use crate::molecule_io::Molecule;
use crate::scf_io::SCF;
use tensors::MathMatrix;
use serde::{Deserialize, Serialize};
use std::fs;

// ---------------------------------------------------------------------------
// small dense linear algebra, row-major throughout
// ---------------------------------------------------------------------------

/// `a[n,k] * b[k,m] -> c[n,m]`, all row-major.
fn gemm(a: &[f64], n: usize, k: usize, b: &[f64], m: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * m];
    for i in 0..n {
        let ai = &a[i * k..i * k + k];
        let ci = &mut c[i * m..i * m + m];
        for l in 0..k {
            let v = ai[l];
            if v == 0.0 {
                continue;
            }
            let bl = &b[l * m..l * m + m];
            for j in 0..m {
                ci[j] += v * bl[j];
            }
        }
    }
    c
}

/// Inverse and determinant of a small square matrix (Gauss-Jordan with
/// partial pivoting, row-major).  `S_oo` is the overlap between the occupied
/// orbitals of two neighbouring geometries, i.e. close to the identity, so
/// plain elimination is perfectly conditioned here.
fn mat_inv_det(a: &[f64], n: usize) -> (Vec<f64>, f64) {
    let mut m = a.to_vec();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    let mut det = 1.0;
    for col in 0..n {
        let mut piv = col;
        let mut best = m[col * n + col].abs();
        for r in (col + 1)..n {
            let v = m[r * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1.0e-14 {
            // Degenerate occupied spaces (a core orbital rotated onto
            // another one): fall back to a pseudo-inverse-like identity,
            // which makes the corresponding row/column of the overlap
            // vanish instead of producing NaNs.
            return (inv, 0.0);
        }
        if piv != col {
            for c in 0..n {
                m.swap(col * n + c, piv * n + c);
                inv.swap(col * n + c, piv * n + c);
            }
            det = -det;
        }
        let d = m[col * n + col];
        det *= d;
        let invd = 1.0 / d;
        for c in 0..n {
            m[col * n + c] *= invd;
            inv[col * n + c] *= invd;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = m[r * n + col];
            if f == 0.0 {
                continue;
            }
            for c in 0..n {
                m[r * n + c] -= f * m[col * n + c];
                inv[r * n + c] -= f * inv[col * n + c];
            }
        }
    }
    (inv, det)
}

fn copy_block(src: &[f64], nrow: usize, ncol: usize, i0: usize, j0: usize, lead: usize) -> Vec<f64> {
    let mut out = vec![0.0; nrow * ncol];
    for i in 0..nrow {
        for j in 0..ncol {
            out[i * ncol + j] = src[(i0 + i) * lead + (j0 + j)];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// MO-overlap blocks
// ---------------------------------------------------------------------------

/// The four blocks of the MO overlap matrix plus the derived quantities the
/// determinant overlap formula needs.
struct Blocks {
    n_oc: usize,
    n_vir: usize,
    soo: Vec<f64>,
    sov: Vec<f64>,
    svo: Vec<f64>,
    svv: Vec<f64>,
    /// `S_oo^{-1}`
    a: Vec<f64>,
    /// `det(S_oo)`
    det: f64,
    /// `A S_ov`, `n_oc x n_vir`
    p: Vec<f64>,
    /// `S_vo A`, `n_vir x n_oc`
    q: Vec<f64>,
    /// `S_vo A S_ov`, `n_vir x n_vir`
    r: Vec<f64>,
}

impl Blocks {
    fn new(s_mo: &[f64], nmo: usize, n_oc: usize, vir_start: usize) -> Blocks {
        let n_vir = nmo - vir_start;
        let soo = copy_block(s_mo, n_oc, n_oc, 0, 0, nmo);
        let sov = copy_block(s_mo, n_oc, n_vir, 0, vir_start, nmo);
        let svo = copy_block(s_mo, n_vir, n_oc, vir_start, 0, nmo);
        let svv = copy_block(s_mo, n_vir, n_vir, vir_start, vir_start, nmo);
        let (a, det) = mat_inv_det(&soo, n_oc);
        let p = gemm(&a, n_oc, n_oc, &sov, n_vir);
        let q = gemm(&svo, n_vir, n_oc, &a, n_oc);
        let r = gemm(&q, n_vir, n_oc, &sov, n_vir);
        Blocks { n_oc, n_vir, soo, sov, svo, svv, a, det, p, q, r }
    }

    #[inline]
    fn soo_at(&self, i: usize, j: usize) -> f64 {
        self.soo[i * self.n_oc + j]
    }
    #[inline]
    fn sov_at(&self, i: usize, b: usize) -> f64 {
        self.sov[i * self.n_vir + b]
    }
    #[inline]
    fn svo_at(&self, a: usize, j: usize) -> f64 {
        self.svo[a * self.n_oc + j]
    }
    #[inline]
    fn svv_at(&self, a: usize, b: usize) -> f64 {
        self.svv[a * self.n_vir + b]
    }
    #[inline]
    fn a_at(&self, j: usize, i: usize) -> f64 {
        self.a[j * self.n_oc + i]
    }
    #[inline]
    fn p_at(&self, i: usize, b: usize) -> f64 {
        self.p[i * self.n_vir + b]
    }
    #[inline]
    fn q_at(&self, a: usize, i: usize) -> f64 {
        self.q[a * self.n_oc + i]
    }
    #[inline]
    fn r_at(&self, a: usize, b: usize) -> f64 {
        self.r[a * self.n_vir + b]
    }

    /// `<Phi_i^a|Phi_j^b>` for one pair `(i,j)` of occupied orbitals, all
    /// virtuals at once (`a` runs over rows, `b` over columns).
    fn det_overlap_block(&self, i: usize, j: usize) -> Vec<f64> {
        let nv = self.n_vir;
        let mut out = vec![0.0; nv * nv];
        let aji = self.a_at(j, i);
        let soo_ij = self.soo_at(i, j);
        for a in 0..nv {
            let s1 = self.q_at(a, i) - 1.0;
            let svo_aj = self.svo_at(a, j);
            for b in 0..nv {
                let sov_ib = self.sov_at(i, b);
                let svv_ab = self.svv_at(a, b);
                // dl = (S_vv - S_vo - S_ov + S_oo)_{abij}
                let dl = svv_ab - svo_aj - sov_ib + soo_ij;
                // s2 uses R in place of S_vv
                let s2 = (self.r_at(a, b) - svo_aj - sov_ib + soo_ij) + dl * s1;
                let s4 = self.p_at(j, b) - 1.0 + dl * aji;
                out[a * nv + b] = self.det * ((1.0 + s1) * (1.0 + s4) - s2 * aji);
            }
        }
        out
    }
}

/// Derivative of the [`Blocks`] quantities with respect to one Cartesian
/// displacement of the ket geometry.
struct BlocksDeriv {
    dsoo: Vec<f64>,
    dsov: Vec<f64>,
    dsvo: Vec<f64>,
    dsvv: Vec<f64>,
    da: Vec<f64>,
    ddet: f64,
    dp: Vec<f64>,
    dq: Vec<f64>,
    dr: Vec<f64>,
}

fn blocks_deriv(b: &Blocks, ds_mo: &[f64], nmo: usize, vir_start: usize) -> BlocksDeriv {
    let (n_oc, n_vir) = (b.n_oc, b.n_vir);
    let dsoo = copy_block(ds_mo, n_oc, n_oc, 0, 0, nmo);
    let dsov = copy_block(ds_mo, n_oc, n_vir, 0, vir_start, nmo);
    let dsvo = copy_block(ds_mo, n_vir, n_oc, vir_start, 0, nmo);
    let dsvv = copy_block(ds_mo, n_vir, n_vir, vir_start, vir_start, nmo);

    // dA = -A dS_oo A ; ddet = det * tr(A dS_oo)
    let adsoo = gemm(&b.a, n_oc, n_oc, &dsoo, n_oc);
    let da = {
        let mut m = gemm(&adsoo, n_oc, n_oc, &b.a, n_oc);
        for v in m.iter_mut() {
            *v = -*v;
        }
        m
    };
    let mut trace = 0.0;
    for i in 0..n_oc {
        trace += aget(&adsoo, n_oc, i, i);
    }
    let ddet = b.det * trace;

    // dP = dA S_ov + A dS_ov
    let dp = {
        let t1 = gemm(&da, n_oc, n_oc, &b.sov, n_vir);
        let t2 = gemm(&b.a, n_oc, n_oc, &dsov, n_vir);
        add(&t1, &t2)
    };
    // dQ = dS_vo A + S_vo dA
    let dq = {
        let t1 = gemm(&dsvo, n_vir, n_oc, &b.a, n_oc);
        let t2 = gemm(&b.svo, n_vir, n_oc, &da, n_oc);
        add(&t1, &t2)
    };
    // dR = dQ S_ov + Q dS_ov
    let dr = {
        let t1 = gemm(&dq, n_vir, n_oc, &b.sov, n_vir);
        let t2 = gemm(&b.q, n_vir, n_oc, &dsov, n_vir);
        add(&t1, &t2)
    };
    BlocksDeriv { dsoo, dsov, dsvo, dsvv, da, ddet, dp, dq, dr }
}

fn aget(m: &[f64], n: usize, i: usize, j: usize) -> f64 {
    m[i * n + j]
}

fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

impl BlocksDeriv {
    /// Derivative of one `(i,j)` block of `<Phi_i^a|Phi_j^b>`.
    fn det_overlap_block(&self, b: &Blocks, i: usize, j: usize) -> Vec<f64> {
        let nv = b.n_vir;
        let no = b.n_oc;
        let mut out = vec![0.0; nv * nv];
        let aji = b.a_at(j, i);
        let daji = aget(&self.da, no, j, i);
        let soo_ij = b.soo_at(i, j);
        let dsoo_ij = aget(&self.dsoo, no, i, j);
        for a in 0..nv {
            let s1 = b.q_at(a, i) - 1.0;
            let ds1 = aget(&self.dq, no, a, i);
            let svo_aj = b.svo_at(a, j);
            let dsvo_aj = aget(&self.dsvo, no, a, j);
            for bb in 0..nv {
                let sov_ib = b.sov_at(i, bb);
                let dsov_ib = aget(&self.dsov, nv, i, bb);
                let svv_ab = b.svv_at(a, bb);
                let dsvv_ab = aget(&self.dsvv, nv, a, bb);
                let dl = svv_ab - svo_aj - sov_ib + soo_ij;
                let ddl = dsvv_ab - dsvo_aj - dsov_ib + dsoo_ij;
                let s2 = (b.r_at(a, bb) - svo_aj - sov_ib + soo_ij) + dl * s1;
                let ds2 = (aget(&self.dr, nv, a, bb) - dsvo_aj - dsov_ib + dsoo_ij)
                    + ddl * s1
                    + dl * ds1;
                let s4 = b.p_at(j, bb) - 1.0 + dl * aji;
                let ds4 = aget(&self.dp, nv, j, bb) + ddl * aji + dl * daji;
                let f = (1.0 + s1) * (1.0 + s4) - s2 * aji;
                let df = ds1 * (1.0 + s4) + (1.0 + s1) * ds4 - ds2 * aji - s2 * daji;
                out[a * nv + bb] = self.ddet * f + b.det * df;
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// state record (JSON) -- the restart file for the next SHARC step
// ---------------------------------------------------------------------------

/// Everything the next geometry needs in order to compute the inter-state
/// overlap matrix with this one.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct SharcState {
    /// SHARC step index (informative).
    #[serde(default)]
    pub step: i64,
    pub natom: usize,
    pub elements: Vec<String>,
    /// Nuclear coordinates in bohr.
    pub geometry_bohr: Vec<[f64; 3]>,
    pub nao: usize,
    pub nmo: usize,
    /// Number of occupied MOs of the reference determinant (`= vir_start`).
    pub nocc: usize,
    /// First active occupied orbital of the BSE transition space.
    pub occ_start: usize,
    /// Active occupied orbitals.
    pub nocc_active: usize,
    /// First virtual orbital of the reference (`= nocc`).
    pub vir_start: usize,
    /// Active virtual orbitals of the BSE transition space.
    pub nvir_active: usize,
    /// MO coefficients, row-major `C[p*nmo + q]`.
    pub mo: Vec<f64>,
    /// BSE excitation energies (Ha), one per excited state.
    pub omega: Vec<f64>,
    /// BSE amplitudes, flat index `i + a*nocc_active`.
    pub x: Vec<Vec<f64>>,
    #[serde(default)]
    pub y: Vec<Vec<f64>>,
    /// Mean-field reference energy (Ha).
    pub scf_energy: f64,
}

impl SharcState {
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let s = serde_json::to_string(self)?;
        fs::write(path, s)
    }

    pub fn load(path: &str) -> anyhow::Result<SharcState> {
        let s = fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("cannot read previous SHARC state '{path}': {e}"))?;
        let st: SharcState = serde_json::from_str(&s)
            .map_err(|e| anyhow::anyhow!("cannot parse previous SHARC state '{path}': {e}"))?;
        Ok(st)
    }

    /// Amplitude vector used on the bra side of the overlap: `X - Y`
    /// (the metric of the RPA problem; `Y = 0` for TDA).
    fn bra_amplitudes(&self, k: usize) -> Vec<f64> {
        let x = &self.x[k];
        if self.y.len() == self.x.len() && self.y[k].len() == x.len() {
            x.iter().zip(self.y[k].iter()).map(|(a, b)| a - b).collect()
        } else {
            x.clone()
        }
    }
}

/// Fix the arbitrary sign of every MO column: the largest AO coefficient of
/// that orbital is made positive.
///
/// The eigenvectors of two neighbouring geometries come out of LAPACK with
/// arbitrary and *independent* phases, so without this convention `det(S_oo)`
/// would jump between `+1` and `-1` from one step to the next and every
/// overlap would change sign at random.  Applying the same convention when the
/// state file is written and when it is read back makes the overlap matrix
/// smooth.
pub fn fix_mo_phases(mo: &mut [f64], nao: usize, nmo: usize) {
    for q in 0..nmo {
        let mut best = 0usize;
        let mut bestv = 0.0f64;
        for p in 0..nao {
            let v = mo[p * nmo + q].abs();
            if v > bestv {
                bestv = v;
                best = p;
            }
        }
        if mo[best * nmo + q] < 0.0 {
            for p in 0..nao {
                mo[p * nmo + q] = -mo[p * nmo + q];
            }
        }
    }
}

/// Fix the arbitrary sign of every BSE root: the largest-magnitude amplitude of
/// that root is made positive (the de-excitation part is flipped with it).
pub fn fix_amplitude_phases(x: &mut [Vec<f64>], y: &mut [Vec<f64>]) {
    for k in 0..x.len() {
        let mut best = 0usize;
        let mut bv = 0.0f64;
        for (i, v) in x[k].iter().enumerate() {
            if v.abs() > bv {
                bv = v.abs();
                best = i;
            }
        }
        if best < x[k].len() && x[k][best] < 0.0 {
            for v in x[k].iter_mut() {
                *v = -*v;
            }
            if k < y.len() {
                for v in y[k].iter_mut() {
                    *v = -*v;
                }
            }
        }
    }
}

/// Flip the sign of a root when it is opposite to the same root of the
/// previous step (maximal amplitude overlap).  This is a phase convention, not
/// a physical change, and it keeps `S_II` positive along a trajectory.
pub fn align_phases_to_previous(
    x: &mut [Vec<f64>],
    y: &mut [Vec<f64>],
    x_prev: &[Vec<f64>],
    y_prev: &[Vec<f64>],
) {
    for k in 0..x.len().min(x_prev.len()) {
        if x[k].len() != x_prev[k].len() {
            continue;
        }
        let mut dot = 0.0;
        for (a, b) in x[k].iter().zip(x_prev[k].iter()) {
            dot += a * b;
        }
        if y.len() == x.len() && y_prev.len() == x_prev.len() && y[k].len() == y_prev[k].len() {
            for (a, b) in y[k].iter().zip(y_prev[k].iter()) {
                dot -= a * b;
            }
        }
        if dot < 0.0 {
            for v in x[k].iter_mut() {
                *v = -*v;
            }
            if k < y.len() {
                for v in y[k].iter_mut() {
                    *v = -*v;
                }
            }
        }
    }
}

/// Sign of each MO of the current geometry relative to the previous one, read
/// from the diagonal of the MO overlap matrix (falling back to the largest
/// element of the row when the diagonal is small because two orbitals are
/// close in energy and rotate into each other).
pub fn mo_overlap_signs(s_mo: &[f64], nmo: usize) -> Vec<f64> {
    let mut signs = vec![1.0; nmo];
    for q in 0..nmo {
        let d = s_mo[q * nmo + q];
        if d.abs() > 0.5 {
            if d < 0.0 {
                signs[q] = -1.0;
            }
            continue;
        }
        let mut best = 0usize;
        let mut bv = 0.0f64;
        for p in 0..nmo {
            let v = s_mo[p * nmo + q].abs();
            if v > bv {
                bv = v;
                best = p;
            }
        }
        if s_mo[best * nmo + q] < 0.0 {
            signs[q] = -1.0;
        }
    }
    signs
}

/// Multiply MO column `q` by `signs[q]`.
pub fn apply_mo_signs(mo: &mut [f64], nao: usize, nmo: usize, signs: &[f64]) {
    for q in 0..nmo {
        if signs[q] < 0.0 {
            for p in 0..nao {
                mo[p * nmo + q] = -mo[p * nmo + q];
            }
        }
    }
}

/// Re-express the BSE amplitudes in the re-signed MO basis.  Flipping orbital
/// `q` multiplies every determinant containing it by `-1`, so the amplitude of
/// configuration `(i,a)` has to be multiplied by `signs[i]*signs[a]`; the
/// physical state (and hence the overlap with the previous step) is unchanged.
pub fn apply_amplitude_signs(
    x: &mut [Vec<f64>],
    y: &mut [Vec<f64>],
    signs: &[f64],
    occ_start: usize,
    nocc_active: usize,
    vir_start: usize,
    nvir_active: usize,
) {
    for k in 0..x.len() {
        for a in 0..nvir_active {
            let sa = signs[vir_start + a];
            for i in 0..nocc_active {
                let s = signs[occ_start + i] * sa;
                if s < 0.0 {
                    x[k][i + a * nocc_active] = -x[k][i + a * nocc_active];
                    if k < y.len() && y[k].len() == x[k].len() {
                        y[k][i + a * nocc_active] = -y[k][i + a * nocc_active];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// cross-geometry MO overlaps
// ---------------------------------------------------------------------------

/// A `Molecule` that carries the same basis set as `mol` but sits at a
/// different geometry.  Only the libcint environment (nuclear coordinates)
/// is rewritten; every basis/ECP structure is shared.
pub fn molecule_at_geometry(mol: &Molecule, geom_bohr: &[[f64; 3]]) -> Molecule {
    let mut m = mol.clone();
    for (a, xyz) in geom_bohr.iter().enumerate() {
        let ptr = m.cint_atm[a][1] as usize;
        m.cint_env[ptr] = xyz[0];
        m.cint_env[ptr + 1] = xyz[1];
        m.cint_env[ptr + 2] = xyz[2];
    }
    m
}

/// Cross-geometry AO overlap in row-major form, `s[mu*nao + nu] = <mu(R0)|nu(R1)>`.
fn cross_ao_overlap(mol0: &Molecule, mol1: &Molecule, nao: usize) -> Vec<f64> {
    let s = mol0.int_cross(mol1, "ovlp".to_string());
    debug_assert_eq!(s.size, [nao, nao]);
    let mut out = vec![0.0; nao * nao];
    for mu in 0..nao {
        for nu in 0..nao {
            out[mu * nao + nu] = s[[mu, nu]];
        }
    }
    out
}

/// `S^MO_{pq} = sum_{mu nu} C0_{mu p} <mu(R0)|nu(R1)> C1_{nu q}` for one
/// fixed ket geometry, row-major `[nmo,nmo]`.
pub fn cross_mo_overlap_at(
    mol: &Molecule,
    c0: &[f64],
    nmo: usize,
    nao: usize,
    geom0: &[[f64; 3]],
    c1: &[f64],
    geom1: &[[f64; 3]],
) -> Vec<f64> {
    // `c0` belongs to `geom0` and `c1` to `geom1`; the AO overlap has to be
    // evaluated between *those* two geometries, not between the current
    // molecule and `geom1` (a previous-step MO set in the current AO frame is
    // meaningless and silently produces a wrong overlap).
    let mol0 = molecule_at_geometry(mol, geom0);
    let mol1 = molecule_at_geometry(mol, geom1);
    let s_ao = cross_ao_overlap(&mol0, &mol1, nao);
    // S^MO = C0^T S^AO C1
    let tmp = gemm(&s_ao, nao, nao, c1, nmo); // [nao, nmo]
    let mut s_mo = vec![0.0; nmo * nmo];
    for p in 0..nmo {
        for q in 0..nmo {
            let mut v = 0.0;
            for mu in 0..nao {
                v += c0[mu * nmo + p] * tmp[mu * nmo + q];
            }
            s_mo[p * nmo + q] = v;
        }
    }
    if std::env::var("REST_SHARC_DEBUG").is_ok() {
        let mut off = 0.0f64;
        for p in 0..nmo {
            for q in 0..nmo {
                let d = s_mo[p * nmo + q] - if p == q { 1.0 } else { 0.0 };
                off = off.max(d.abs());
            }
        }
        let mut cnorm = 0.0f64;
        for p in 0..nmo {
            let mut n = 0.0;
            for mu in 0..nao {
                n += c0[mu * nmo + p] * c0[mu * nmo + p];
            }
            cnorm = cnorm.max((n - 1.0).abs());
        }
        eprintln!(
            "[sharc debug] nao={nao} nmo={nmo} max|S_mo-I|={off:.3e} max|Ccol^2-1|={cnorm:.3e}"
        );
        for p in 0..nmo.min(8) {
            let row: Vec<String> =
                (0..nmo.min(8)).map(|q| format!("{:+.3}", s_mo[p * nmo + q])).collect();
            eprintln!("   S[{p}] {}", row.join(" "));
        }
    }
    s_mo
}

/// Derivative of [`cross_mo_overlap_at`] with respect to the nuclear
/// coordinates of the **ket** geometry.
///
/// The derivative of the analytic overlap expression is taken with a central
/// finite difference of the cross AO overlap integrals (step `h`, bohr).
/// Doing it here -- at the level of the AO integrals, *before* the MO
/// transformation and the determinant algebra -- keeps the coupling exactly
/// consistent with the overlap matrix (`d/dR <Psi_I(R)|Psi_J(R')>` of the
/// same expression), and costs `6*natom` overlap builds instead of a new
/// integral-derivative code path.  The BSE amplitudes are frozen (see the
/// module documentation).
pub fn cross_mo_overlap_deriv(
    mol: &Molecule,
    c0: &[f64],
    nmo: usize,
    nao: usize,
    geom0: &[[f64; 3]],
    c1: &[f64],
    geom1: &[[f64; 3]],
    h: f64,
) -> Vec<Vec<f64>> {
    let natom = geom1.len();
    let mut out = Vec::with_capacity(3 * natom);
    for a in 0..natom {
        for c in 0..3 {
            let mut gp = geom1.to_vec();
            gp[a][c] += h;
            let sp = cross_mo_overlap_at(mol, c0, nmo, nao, geom0, c1, &gp);
            let mut gm = geom1.to_vec();
            gm[a][c] -= h;
            let sm = cross_mo_overlap_at(mol, c0, nmo, nao, geom0, c1, &gm);
            let d: Vec<f64> = sp
                .iter()
                .zip(sm.iter())
                .map(|(p, m)| (p - m) / (2.0 * h))
                .collect();
            out.push(d);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// overlap matrix and NAC vectors
// ---------------------------------------------------------------------------

/// Number of states handled (ground state + excited states).
fn n_states(st: &SharcState) -> usize {
    1 + st.omega.len()
}

pub fn check_compatible(a: &SharcState, b: &SharcState) -> anyhow::Result<()> {
    if a.nao != b.nao || a.nmo != b.nmo {
        anyhow::bail!(
            "SHARC state mismatch: previous step has nao/nmo = {}/{}, current has {}/{}",
            a.nao, a.nmo, b.nao, b.nmo
        );
    }
    if a.natom != b.natom {
        anyhow::bail!("SHARC state mismatch: {} atoms in the previous step, {} now", a.natom, b.natom);
    }
    if a.occ_start != b.occ_start
        || a.nocc_active != b.nocc_active
        || a.vir_start != b.vir_start
        || a.nvir_active != b.nvir_active
    {
        anyhow::bail!(
            "SHARC state mismatch: the BSE active space changed between steps \
             (occupied {}..{}, virtual {}..{} vs {}..{} / {}..{}).  Pin it with \
             `bse_active_occ_start` / `bse_active_vir_end` (or keep the same \
             `bse_cutoff_energy` / `gradient_bse_nvir` window).",
            a.occ_start, a.occ_start + a.nocc_active,
            a.vir_start, a.vir_start + a.nvir_active,
            b.occ_start, b.occ_start + b.nocc_active,
            b.vir_start, b.vir_start + b.nvir_active
        );
    }
    Ok(())
}

/// Overlap matrix `S[I][J] = <I(prev)|J(cur)>` for `I,J = 0..nstates`, with
/// state 0 the ground state.
pub fn overlap_matrix(prev: &SharcState, cur: &SharcState, s_mo: &[f64]) -> Vec<Vec<f64>> {
    let nmo = cur.nmo;
    let n_oc = cur.vir_start;
    let vir_start = cur.vir_start;
    let blocks = Blocks::new(s_mo, nmo, n_oc, vir_start);
    let ns = n_states(cur);
    let mut s = vec![vec![0.0; ns]; ns];

    // ground-ground
    s[0][0] = blocks.det;
    if ns == 1 {
        return s;
    }

    let na = cur.nocc_active;
    let nv = cur.nvir_active;
    debug_assert_eq!(prev.omega.len(), cur.omega.len());
    // The bra amplitudes belong to the *previous* geometry and the ket
    // amplitudes to the current one: `S_IJ = <Psi_I(R0)|Psi_J(R1)>`.  Mixing
    // the two (using the current amplitudes on the bra side as well) would
    // break every off-diagonal element and, through it, the numerical
    // derivative used to validate the couplings.
    let amps_bra: Vec<Vec<f64>> = (0..prev.omega.len().min(cur.omega.len()))
        .map(|k| prev.bra_amplitudes(k))
        .collect();
    let amps_ket: Vec<Vec<f64>> = (0..cur.omega.len())
        .map(|k| cur.bra_amplitudes(k))
        .collect();

    // ground <-> excited
    for k in 0..cur.omega.len() {
        // <Psi_0(R0)|Psi_J(R1)>: Q/P already carry the geometry dependence,
        // only the ket amplitude is contracted with P.
        let ck = &amps_ket[k];
        let mut acc = 0.0;
        for i in 0..na {
            for a in 0..nv {
                acc += ck[i + a * na] * blocks.p_at(cur.occ_start + i, a);
            }
        }
        s[0][k + 1] = blocks.det * acc;
        // <Psi_I(R0)|Psi_0(R1)>: the bra (excited) amplitude is contracted
        // with Q.
        if k < amps_bra.len() {
            let cb = &amps_bra[k];
            let mut acc2 = 0.0;
            for i in 0..na {
                for a in 0..nv {
                    acc2 += cb[i + a * na] * blocks.q_at(a, cur.occ_start + i);
                }
            }
            s[k + 1][0] = blocks.det * acc2;
        }
    }

    // excited-excited
    for i in 0..na {
        let im = cur.occ_start + i;
        for j in 0..na {
            let jm = cur.occ_start + j;
            let o = blocks.det_overlap_block(im, jm);
            // `det_overlap_block` spans *all* virtuals; the amplitudes only
            // cover the active window.
            let ostride = blocks.n_vir;
            for a in 0..nv {
                for b in 0..nv {
                    let ov = o[a * ostride + b];
                    if ov == 0.0 {
                        continue;
                    }
                    for k1 in 0..amps_bra.len() {
                        let c1 = amps_bra[k1][i + a * na];
                        if c1 == 0.0 {
                            continue;
                        }
                        for k2 in 0..amps_ket.len() {
                            let c2 = amps_ket[k2][j + b * na];
                            if c2 == 0.0 {
                                continue;
                            }
                            s[k1 + 1][k2 + 1] += c1 * c2 * ov;
                        }
                    }
                }
            }
        }
    }
    s
}

/// Non-adiabatic coupling vectors `d[I][J][3*atom+comp] = <I|d/dR|J>` at the
/// current geometry, computed from the analytic overlap expression with the
/// amplitudes frozen (see the module documentation).
///
/// `s_mo` must be the MO overlap of the current geometry with itself (the
/// identity, up to the BSE active-space convention) and `ds_mo` its
/// derivative with respect to the ket geometry.
pub fn nac_vectors(cur: &SharcState, s_mo: &[f64], ds_mo: &[Vec<f64>]) -> Vec<Vec<Vec<f64>>> {
    let nmo = cur.nmo;
    let n_oc = cur.vir_start;
    let vir_start = cur.vir_start;
    let blocks = Blocks::new(s_mo, nmo, n_oc, vir_start);
    let ns = n_states(cur);
    let ncoord = ds_mo.len();
    let na = cur.nocc_active;
    let nv = cur.nvir_active;

    let mut nac = vec![vec![vec![0.0; ncoord]; ns]; ns];
    if ns == 1 {
        return nac;
    }
    let amps: Vec<Vec<f64>> = (0..cur.omega.len()).map(|k| cur.bra_amplitudes(k)).collect();

    for (c, d_s_mo) in ds_mo.iter().enumerate() {
        let db = blocks_deriv(&blocks, d_s_mo, nmo, vir_start);
        // ground-ground
        nac[0][0][c] = db.ddet;

        // ground <-> excited
        for k in 0..cur.omega.len() {
            let cb = &amps[k];
            let mut acc = 0.0;
            let mut acc2 = 0.0;
            for i in 0..na {
                for a in 0..nv {
                    let cf = cb[i + a * na];
                    if cf == 0.0 {
                        continue;
                    }
                    acc += cf * db.dp[(cur.occ_start + i) * nv + a];
                    acc2 += cf * db.dq[a * n_oc + (cur.occ_start + i)];
                }
            }
            nac[0][k + 1][c] = db.ddet * {
                let mut v = 0.0;
                for i in 0..na {
                    for a in 0..nv {
                        v += cb[i + a * na] * blocks.p_at(cur.occ_start + i, a);
                    }
                }
                v
            } + blocks.det * acc;
            nac[k + 1][0][c] = db.ddet * {
                let mut v = 0.0;
                for i in 0..na {
                    for a in 0..nv {
                        v += cb[i + a * na] * blocks.q_at(a, cur.occ_start + i);
                    }
                }
                v
            } + blocks.det * acc2;
        }

        // excited-excited
        for i in 0..na {
            let im = cur.occ_start + i;
            for j in 0..na {
                let jm = cur.occ_start + j;
                let d = db.det_overlap_block(&blocks, im, jm);
                let dstride = blocks.n_vir;
                for a in 0..nv {
                    for b in 0..nv {
                        let dv = d[a * dstride + b];
                        if dv == 0.0 {
                            continue;
                        }
                        for k1 in 0..amps.len() {
                            let c1 = amps[k1][i + a * na];
                            if c1 == 0.0 {
                                continue;
                            }
                            for k2 in 0..amps.len() {
                                let c2 = amps[k2][j + b * na];
                                if c2 == 0.0 {
                                    continue;
                                }
                                nac[k1 + 1][k2 + 1][c] += c1 * c2 * dv;
                            }
                        }
                    }
                }
            }
        }
    }

    // The frozen-amplitude expression is only antisymmetric up to the
    // difference between the amplitude sets of the two states; enforce the
    // exact anti-Hermiticity SHARC relies on.  The diagonal is the derivative
    // of the state normalisation and is identically zero.
    for i in 0..ns {
        for c in 0..ncoord {
            nac[i][i][c] = 0.0;
        }
    }
    for i in 0..ns {
        for j in (i + 1)..ns {
            for c in 0..ncoord {
                let v = 0.5 * (nac[i][j][c] - nac[j][i][c]);
                nac[i][j][c] = v;
                nac[j][i][c] = -v;
            }
        }
    }
    nac
}

// ---------------------------------------------------------------------------
// dipole moments
// ---------------------------------------------------------------------------

/// State and transition dipole moments in atomic units, `dip[I][J][3]`.
///
/// * `dip[0][0]` is the mean-field ground-state dipole (nuclear + electronic).
/// * `dip[0][I] = dip[I][0]` is the BSE transition dipole
///   `2 sum_ia (X_I+Y_I)_ia mu^d_ia`, the same contraction REST uses for the
///   oscillator strength.
/// * excited-state permanent dipoles and excited-excited transition dipoles
///   are not part of the static BSE response; they are reported as the
///   ground-state dipole / zero respectively.
pub fn dipole_moments(scf: &SCF, cur: &SharcState) -> Vec<Vec<[f64; 3]>> {
    use crate::ri_bse::dipoles::{obtain_ao_dips, obtain_mu_ia};
    let ns = n_states(cur);
    let mut dip = vec![vec![[0.0; 3]; ns]; ns];

    let ao_dip = obtain_ao_dips(scf, None);
    let mut dm = scf.density_matrix[0].clone();
    if scf.mol.spin_channel == 2 && scf.density_matrix.len() > 1 {
        dm.self_add(&scf.density_matrix[1]);
    }
    let (nuc, _) = scf.mol.geom.evaluate_dipole_moment(None);
    let mut gs = nuc;
    for d in 0..3 {
        let blk = ao_dip.get_reducing_matrix(d).unwrap();
        let nao = cur.nao;
        let mut e = 0.0;
        for mu in 0..nao {
            for nu in 0..nao {
                e += dm[[mu, nu]] * blk.data[mu + nu * nao];
            }
        }
        gs[d] -= e;
    }
    for i in 0..ns {
        dip[i][i] = gs;
    }

    let na = cur.nocc_active;
    let nv = cur.nvir_active;
    let c = &scf.eigenvectors[0];
    for k in 0..cur.omega.len() {
        let mut mu = [0.0; 3];
        let x = &cur.x[k];
        let y = &cur.y[k];
        let has_y = cur.y.len() == cur.x.len() && y.len() == x.len();
        for i in 0..na {
            for a in 0..nv {
                let idx = i + a * na;
                let coeff = if has_y { x[idx] + y[idx] } else { x[idx] };
                if coeff == 0.0 {
                    continue;
                }
                let mu_ia = obtain_mu_ia(c, &ao_dip, cur.occ_start + i, cur.vir_start + a);
                for d in 0..3 {
                    mu[d] += 2.0 * coeff * mu_ia[d];
                }
            }
        }
        dip[0][k + 1] = mu;
        dip[k + 1][0] = mu;
    }
    dip
}

// ---------------------------------------------------------------------------
// JSON output of one SHARC step
// ---------------------------------------------------------------------------

/// Everything the `SHARC_REST.py` interface reads from `sharc_step.json`.
#[derive(Serialize)]
pub struct SharcStep {
    pub natom: usize,
    pub elements: Vec<String>,
    pub geometry_bohr: Vec<[f64; 3]>,
    pub nstates: usize,
    pub energies: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gradients: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nac: Option<Vec<Vec<Vec<f64>>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overlap: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dipoles: Option<Vec<Vec<[f64; 3]>>>,
    pub omega: Vec<f64>,
    pub multiplicities: Vec<usize>,
    pub units: String,
    pub step: i64,
}

impl SharcStep {
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let s = serde_json::to_string_pretty(self)?;
        fs::write(path, s)
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Determinant overlap by brute force (products of MO overlaps).
    fn brute(s: &[f64], nmo: usize, n_oc: usize, occ: &[usize], vir: &[usize], i: usize, a: usize, j: usize, b: usize) -> f64 {
        let mut bra: Vec<usize> = occ.to_vec();
        bra[i] = vir[a];
        let mut ket: Vec<usize> = occ.to_vec();
        ket[j] = vir[b];
        // determinant of the n_oc x n_oc overlap matrix
        let n = bra.len();
        let mut m = vec![0.0; n * n];
        for r in 0..n {
            for c in 0..n {
                m[r * n + c] = s[bra[r] * nmo + ket[c]];
            }
        }
        mat_inv_det(&m, n).1
    }

    #[test]
    fn determinant_overlap_formula_matches_bruteforce() {
        // three pseudo-random, non-orthogonal MO sets
        let nmo = 7;
        let n_oc = 4;
        let vir: Vec<usize> = (4..7).collect();
        let occ: Vec<usize> = (0..4).collect();
        let mut s = vec![0.0; nmo * nmo];
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        // near-identity plus noise keeps S_oo well conditioned
        for p in 0..nmo {
            for q in 0..nmo {
                s[p * nmo + q] = if p == q { 1.0 } else { 0.0 } + 0.08 * rnd();
            }
        }
        let blocks = Blocks::new(&s, nmo, n_oc, n_oc);
        let mut maxerr = 0.0f64;
        for i in 0..n_oc {
            for j in 0..n_oc {
                let blk = blocks.det_overlap_block(i, j);
                for a in 0..vir.len() {
                    for b in 0..vir.len() {
                        let exact = brute(&s, nmo, n_oc, &occ, &vir, i, a, j, b);
                        maxerr = maxerr.max((exact - blk[a * vir.len() + b]).abs());
                    }
                }
            }
        }
        assert!(maxerr < 1e-12, "determinant overlap error {maxerr}");
    }

    #[test]
    fn ground_excited_overlap_matches_bruteforce() {
        let nmo = 6;
        let n_oc = 3;
        let vir: Vec<usize> = (3..6).collect();
        let occ: Vec<usize> = (0..3).collect();
        let mut s = vec![0.0; nmo * nmo];
        let mut seed = 987654321u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        for p in 0..nmo {
            for q in 0..nmo {
                s[p * nmo + q] = if p == q { 1.0 } else { 0.0 } + 0.05 * rnd();
            }
        }
        let blocks = Blocks::new(&s, nmo, n_oc, n_oc);
        let mut maxerr = 0.0f64;
        for i in 0..n_oc {
            for a in 0..vir.len() {
                let mut ket = occ.clone();
                ket[i] = vir[a];
                let n = n_oc;
                let mut m = vec![0.0; n * n];
                for r in 0..n {
                    for c in 0..n {
                        m[r * n + c] = s[occ[r] * nmo + ket[c]];
                    }
                }
                let exact = mat_inv_det(&m, n).1;
                maxerr = maxerr.max((exact - blocks.det * blocks.p_at(i, a)).abs());
            }
        }
        assert!(maxerr < 1e-12, "ground-excited overlap error {maxerr}");
    }

    #[test]
    fn derivative_of_overlap_matches_finite_difference() {
        // Build a random (small) non-orthogonal overlap matrix and check the
        // analytic derivative machinery against a finite difference of the
        // SAME formula.
        let nmo = 6;
        let n_oc = 3;
        let vir: Vec<usize> = (3..6).collect();
        let occ: Vec<usize> = (0..3).collect();
        let mut seed = 555u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        let build = |scale: f64, rnd: &mut dyn FnMut() -> f64| {
            let mut s = vec![0.0; nmo * nmo];
            for p in 0..nmo {
                for q in 0..nmo {
                    s[p * nmo + q] = if p == q { 1.0 } else { 0.0 } + scale * rnd();
                }
            }
            s
        };
        let s = build(0.1, &mut rnd);
        let ds: Vec<f64> = (0..nmo * nmo).map(|_| 0.07 * rnd()).collect();
        let h = 1e-6;
        let blocks = Blocks::new(&s, nmo, n_oc, n_oc);
        let db = blocks_deriv(&blocks, &ds, nmo, n_oc);
        let mut s2 = s.clone();
        for k in 0..s.len() {
            s2[k] += h * ds[k];
        }
        let blocks2 = Blocks::new(&s2, nmo, n_oc, n_oc);
        let mut maxerr = 0.0f64;
        for i in 0..n_oc {
            for j in 0..n_oc {
                let a0 = blocks.det_overlap_block(i, j);
                let a1 = blocks2.det_overlap_block(i, j);
                let d = db.det_overlap_block(&blocks, i, j);
                for k in 0..a0.len() {
                    let fd = (a1[k] - a0[k]) / h;
                    maxerr = maxerr.max((fd - d[k]).abs());
                }
            }
        }
        // derivative magnitudes are O(1e-1), finite differences good to ~1e-8
        assert!(maxerr < 1e-7, "NAC derivative error {maxerr}");
        let _ = &vir;
        let _ = &occ;
    }
}
