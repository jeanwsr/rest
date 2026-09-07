//! Vibrational (harmonic) analysis module.
//!
//! All tensors are column-major. Geometry is stored as `[3, natm]`, masses as
//! `[natm]`, Hessian as `[3*natm, 3*natm]`.
//!
//! No IR intensity / dipole-derivative terms at this stage.
//!
//! # Note
//!
//! This module is direct transformation from psi4 (psi4/psi4/driver/qcdb/vib.py).
//! Some modifications are from PySCF (pyscf/hessian/thermo.py).
//! This file contains AI assisted code, and not fully reviewed by human.

use crate::analdrv::prelude::*;

// ---------------------------------------------------------------------------
// Physical constants (CODATA2014, matching the Python implementation)
// ---------------------------------------------------------------------------

use crate::constants::{
    AMU2KG, AVOGADRO, BOHR2ANG, BOLTZMANN, HARTREE2J, HARTREE2KCALMOL, HARTREE2KJMOL, HARTREE2WAVENUMBERS, PLANCK,
    R_GAS, SPEED_OF_LIGHT,
};

/// cm⁻¹ conversion factor from force-constant eigenvalues (atomic units).
fn uconv_cm1() -> f64 {
    (AVOGADRO * HARTREE2J * 1.0e19).sqrt() / (2.0 * std::f64::consts::PI * SPEED_OF_LIGHT * BOHR2ANG)
}

/// Mass-centred geometry `[3, natm]`.
pub fn mass_centred_geom(geom: TsrView, mass: TsrView) -> Tsr {
    let mass_sum = mass.to_vec().iter().sum::<f64>();
    let mc = (&geom * mass.i((None, ..))).sum_axes(1) / mass_sum; // [3]
    let mc_vec = mc.to_vec();
    &geom - rt::asarray((mc_vec, [3, 1].c(), geom.device()))
}

// ---------------------------------------------------------------------------
// Translation / rotation space
// ---------------------------------------------------------------------------

/// Idealized translation + rotation basis vectors.
///
/// # Parameters
/// - `mass` : `[natm]` atomic masses [u].
/// - `geom` : `[3, natm]` Cartesian geometry [a₀] (column-major).
/// - `space` : `"T"`, `"R"`, or `"TR"`.
///
/// # Returns
/// `tr` : `[3*natm, nrt]` orthonormal basis (each column is a TR vector).
pub fn get_tr_space(mass: TsrView, geom: TsrView, space: &str) -> Tsr {
    _get_tr_space(mass, geom, space, None, None)
}

/// Internal helper.  When `nrt_user` is `Some(k)`, the top-`k` singular
/// vectors are selected (rank set by the caller from rotor type).  Otherwise
/// `tol_user` (`Some(t)` absolute cutoff, or `None` for the default
/// `ndof × max(s) × ε` tolerance) determines the rank.
fn _get_tr_space(mass: TsrView, geom: TsrView, space: &str, tol_user: Option<f64>, nrt_user: Option<usize>) -> Tsr {
    let device = geom.device().clone();
    let natm = geom.shape()[1];
    let ndof = 3 * natm;

    let mass_vec = mass.reshape(-1).to_vec();

    // sqrtmmm: [3*natm], interleaved per-atom (m_a repeated 3×): [m0,m0,m0, m1,m1,m1, ...]
    let mut sqrtmmm = Vec::with_capacity(ndof);
    let mut xxx = Vec::with_capacity(ndof);
    let mut yyy = Vec::with_capacity(ndof);
    let mut zzz = Vec::with_capacity(ndof);
    for a in 0..natm {
        let sm = mass_vec[a].sqrt();
        let gx = geom[[0, a]];
        let gy = geom[[1, a]];
        let gz = geom[[2, a]];
        for _ in 0..3 {
            sqrtmmm.push(sm);
            xxx.push(gx);
            yyy.push(gy);
            zzz.push(gz);
        }
    }
    let sqrtmmm = rt::asarray((sqrtmmm, &device));
    let xxx = rt::asarray((xxx, &device));
    let yyy = rt::asarray((yyy, &device));
    let zzz = rt::asarray((zzz, &device));

    // unit vectors ux/uy/uz each [3*natm]: per-atom [1,0,0]/[0,1,0]/[0,0,1]
    let mut ux = Vec::with_capacity(ndof);
    let mut uy = Vec::with_capacity(ndof);
    let mut uz = Vec::with_capacity(ndof);
    for _ in 0..natm {
        ux.extend_from_slice(&[1.0, 0.0, 0.0]);
        uy.extend_from_slice(&[0.0, 1.0, 0.0]);
        uz.extend_from_slice(&[0.0, 0.0, 1.0]);
    }
    let ux = rt::asarray((ux, &device));
    let uy = rt::asarray((uy, &device));
    let uz = rt::asarray((uz, &device));

    // translation [3*natm]
    let t1 = &sqrtmmm * &ux;
    let t2 = &sqrtmmm * &uy;
    let t3 = &sqrtmmm * &uz;
    // rotation [3*natm]
    let r4 = &sqrtmmm * (&yyy * &uz - &zzz * &uy);
    let r5 = &sqrtmmm * (&zzz * &ux - &xxx * &uz);
    let r6 = &sqrtmmm * (&xxx * &uy - &yyy * &ux);

    let mut cols: Vec<Tsr> = Vec::new();
    if space.contains('T') {
        cols.push(t1);
        cols.push(t2);
        cols.push(t3);
    }
    if space.contains('R') {
        cols.push(r4);
        cols.push(r5);
        cols.push(r6);
    }
    if cols.is_empty() {
        cols.push(rt::asarray((vec![0.0_f64; ndof], &device)));
    }

    let tr_raw: Tsr = rt::stack((cols, -1)); // [3*natm, n_raw]

    // orthonormal basis for the column space via SVD: tr_raw = U S Vh, Q = U[:, :num]
    let (u, s, _vh): (Tsr, Tsr, Tsr) = rt::linalg::svd(tr_raw.view()).into();
    let svec = s.reshape(-1).to_vec();
    let num = match nrt_user {
        Some(k) => k,
        None => {
            let tol = match tol_user {
                Some(t) => t,
                None => {
                    let smax = svec.iter().copied().fold(0.0_f64, f64::max);
                    (ndof as f64) * smax * f64::EPSILON
                },
            };
            svec.iter().filter(|&&x| x > tol).count()
        },
    };
    u.i((.., ..num)).to_owned()
}

// ---------------------------------------------------------------------------
// Rotation constants and rotor type
// ---------------------------------------------------------------------------

/// Rotational constants.
///
/// # Parameters
/// - `mass` : `[natm]` atomic masses [u].
/// - `atom_coords` : `[3, natm]` **mass-centred** geometry [a₀].
/// - `unit` : `"GHz"` or `"wavenumber"`.
///
/// # Returns
/// `e` : `[3]` rotational constants (sorted ascending).
pub fn rotation_const(mass: TsrView, atom_coords: TsrView, unit: &str) -> Tsr {
    let device = mass.device().clone();
    // im = Σ_a m_a r_{a,r} r_{a,s}  -> [3, 3]; atom_coords is [3, natm]
    let weighted = &atom_coords * mass.i((None, ..)); // [3, natm]
    let im_rr = &weighted % atom_coords.t(); // [3, 3]
    let trace = im_rr.diagonal(None).sum();
    // I = trace*I - Σ m r r^T
    let eye: Tsr = rt::eye((3, &device));
    let im = &eye * trace - &im_rr;

    let mut e = rt::linalg::eigvalsh(im.view());
    let mut evec = e.reshape(-1).to_vec();
    for v in evec.iter_mut() {
        if v.abs() < 1e-9 {
            *v = 0.0;
        }
    }
    e = rt::asarray((evec, &device));

    let unit_im = AMU2KG * 5.2917721092e-11_f64.powi(2);
    let unit_hz = 1.0545718001391127e-34 / (4.0 * std::f64::consts::PI * unit_im);

    let unit_lower = unit.to_lowercase();
    let conv = if unit_lower == "ghz" {
        1e-9
    } else if unit_lower == "wavenumber" {
        1.0 / SPEED_OF_LIGHT * 1e-2
    } else {
        panic!("Unsupported unit {}", unit);
    };
    let mut out = Vec::with_capacity(3);
    for &x in e.reshape(-1).to_vec().iter() {
        if x.abs() < 1e-30 {
            out.push(f64::INFINITY);
        } else {
            out.push(unit_hz / x * conv);
        }
    }
    rt::asarray((out, &device))
}

/// Classify rotor type from rotational constants [GHz].
///
/// Returns `"ATOM"`, `"LINEAR"`, or `"REGULAR"`.
pub fn get_rotor_type(rot_const_ghz: TsrView) -> &'static str {
    let v = rot_const_ghz.reshape(-1).to_vec();
    if v.iter().all(|&x| x > 1e8) {
        "ATOM"
    } else if v[0] > 1e8 && (v[1] - v[2]).abs() < 1e-3 {
        "LINEAR"
    } else {
        "REGULAR"
    }
}

// ---------------------------------------------------------------------------
// VibInfo result container
// ---------------------------------------------------------------------------

/// Output of [`harmonic_analysis`].
///
/// All tensors are column-major. Modes are stored as **columns** of `q`/`w`/`x`
/// (each `[ndof, ndof]`). Per-mode quantities (`omega`, `mu`, `k`, ...) are
/// `[ndof]`.
#[derive(Clone)]
pub struct VibInfo {
    /// Number of degrees of freedom (3 × natm).
    pub ndof: usize,
    /// Frequency `[ndof]`, stored as `(real, imag)` pairs (imag > 0 ⇒ imaginary mode).
    pub omega: Vec<f64>,
    /// Imaginary flag per mode (true if `imag > real`).
    pub imag: Vec<bool>,
    /// Mass-weighted normal modes, flattened `[ndof × ndof]` in col-major order
    /// (columns = modes): element `(row, col)` = `q[row + col * ndof]`.
    pub q: Vec<f64>,
    /// Un-mass-weighted normal modes, same flat layout `[ndof × ndof]`.
    pub w: Vec<f64>,
    /// Normalized un-mass-weighted normal modes, same flat layout `[ndof × ndof]`.
    pub x: Vec<f64>,
    /// Degeneracy count per mode `[ndof]` (as `i64`).
    pub degeneracy: Vec<i64>,
    /// TR/V classification per mode: `"TR"`, `"V"`, or `"-"`.
    pub trv: Vec<&'static str>,
    /// Reduced mass `[ndof]` [u].
    pub mu: Vec<f64>,
    /// Force constant `[ndof]` [mDyne/Å].
    pub k: Vec<f64>,
    /// RMS deviation v=0 `[ndof]` [a₀·u^½].
    pub dq0: Vec<f64>,
    /// Turning point v=0 (mass-weighted) `[ndof]` [a₀·u^½].
    pub qtp0: Vec<f64>,
    /// Turning point v=0 (Cartesian) `[ndof]` [a₀].
    pub xtp0: Vec<f64>,
    /// Characteristic vibrational temperature `[ndof]` [K].
    pub theta_vib: Vec<f64>,
}

impl VibInfo {
    /// Frequency for mode `i` as a signed real number: positive for real modes,
    /// negative for imaginary modes (the "imaginary freq as negative" convention).
    /// For a real mode `omega[i]` is the frequency; for an imaginary mode
    /// `omega[i]` holds the magnitude and this returns its negation.
    pub fn freq_signed(&self, i: usize) -> f64 {
        if self.imag[i] {
            -self.omega[i]
        } else {
            self.omega[i]
        }
    }

    /// Indices of vibrational modes (`TRV == "V"`).
    pub fn vib_indices(&self) -> Vec<usize> {
        self.trv.iter().enumerate().filter(|(_, &t)| t == "V").map(|(i, _)| i).collect()
    }

    /// Number of atoms.
    pub fn nat(&self) -> usize {
        self.ndof / 3
    }

    /// Returns element `(row, col)` of a flat col-major `[ndof × ndof]` normal-
    /// coordinate array (`q`, `w`, or `x`).
    #[inline]
    pub fn normco_index(data: &[f64], ndof: usize, row: usize, col: usize) -> f64 {
        data[row + col * ndof]
    }

    /// Reconstruct a col-major `[ndof, ndof]` tensor from a flat normal-mode
    /// slice (e.g. `&self.q`, `&self.w`, `&self.x`) — for testing / advanced use.
    pub fn normco_matrix(&self, data: &[f64], device: &DeviceBLAS) -> Tsr {
        rt::asarray((data.to_vec(), [self.ndof, self.ndof].f(), device))
    }
}

// ---------------------------------------------------------------------------
// Helper: standardize column phases so extreme element is positive
// ---------------------------------------------------------------------------

/// Return a copy of `q` (column-major `[n, m]`) where each column is scaled so
/// that its element of maximum absolute value is positive (tol 1e-2).
fn phase_cols_to_max_element(q: TsrView, tol: f64) -> Tsr {
    let (n, m) = (q.shape()[0], q.shape()[1]);
    let mut out = q.to_owned();
    for v in 0..m {
        let mut vextreme = 0.0_f64;
        for r in 0..n {
            vextreme = vextreme.max(out[[r, v]].abs());
        }
        // first index whose fabs equals vextreme within tol
        let mut iextreme = 0;
        for r in 0..n {
            if (vextreme - out[[r, v]].abs()) < tol {
                iextreme = r;
                break;
            }
        }
        if out[[iextreme, v]] < 0.0 {
            for r in 0..n {
                out[[r, v]] = -out[[r, v]];
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// harmonic_analysis
// ---------------------------------------------------------------------------

/// Extract frequencies, normal modes, and other properties from an electronic
/// Hessian. Rust port of `pyhessref.vib.harmonic_analysis`.
///
/// # Parameters
/// - `hess` : `[3*natm, 3*natm]` non-mass-weighted Cartesian Hessian [E_h/a₀²].
/// - `geom` : `[3, natm]` Cartesian geometry [a₀] (column-major).
/// - `mass` : `[natm]` atomic masses [u].
/// - `project_trans` : project out idealized translations.
/// - `project_rot` : project out idealized rotations.
///
/// # Returns
/// [`VibInfo`] with all 3×natm modes (TR + V). Geometry must be in **bohr**.
pub fn harmonic_analysis(
    hess: TsrView,
    geom: TsrView,
    mass: TsrView,
    project_trans: bool,
    project_rot: bool,
) -> VibInfo {
    let device = hess.device().clone();
    let natm = mass.shape()[0];
    let ndof = 3 * natm;
    assert_eq!(geom.shape().as_slice(), &[3, natm], "geom must be [3, natm]");
    assert_eq!(hess.shape().as_slice(), &[ndof, ndof], "hess must be [3*natm, 3*natm]");

    let nmwhess = hess.into_contig(ColMajor);

    // --------------- rotor type → number of TR dof ---------------
    // The TR count is determined physically from the rotor type (3 for an
    // atom, 5 for a linear molecule, 6 otherwise), rather than by an SVD
    // rank cutoff on the (numerically truncated) TR basis.  This mirrors
    // PySCF's thermo.harmonic_analysis and makes the number of projected-out
    // directions self-consistent with the TR/V classification below.
    let geom_cm = {
        let mass_sum: f64 = mass.reshape(-1).to_vec().iter().sum();
        // mc = Σ m_a r_a / Σ m_a  -> [3]; reshape to [3, 1] to broadcast over natm
        let mc = (geom.view() * mass.i((None, ..))).sum_axes(1) / mass_sum; // [3]
        geom.view() - mc.i((.., None)) // [3, natm]
    };
    let rc_ghz = rotation_const(mass.view(), geom_cm.view(), "GHz");
    let rotor_type = get_rotor_type(rc_ghz.view());
    let mut nrt = match rotor_type {
        "ATOM" => 0,
        "LINEAR" => 5,
        _ => 6,
    };
    // translation is always 3 dof; rotations are 0/2/3 for ATOM/LINEAR/REGULAR.
    if !project_trans {
        nrt = nrt.max(3) - 3;
    }
    if !project_rot {
        let nrot = match rotor_type {
            "ATOM" => 0,
            "LINEAR" => 2,
            _ => 3,
        };
        nrt = nrt.max(nrot) - nrot;
    }

    // --------------- translation / rotation projector ---------------
    let space = format!("{}{}", if project_trans { "T" } else { "" }, if project_rot { "R" } else { "" });
    let tr_space = _get_tr_space(mass.view(), geom.view(), &space, None, Some(nrt)); // [ndof, nrt]

    // projector  P = I - Σ |tr⟩⟨tr|
    let nrt = tr_space.shape()[1];
    let mut p: Tsr = rt::zeros(([ndof, ndof], &device));
    for i in 0..ndof {
        p[[i, i]] = 1.0;
    }
    for c in 0..nrt {
        // outer(tr_col, tr_col): [ndof, ndof]
        let trc = tr_space.i((.., c)).to_owned(); // [ndof]
        let outer = &trc.i((.., None)) * &trc.i((None, ..)); // [ndof, ndof]
        *&mut p -= &outer;
    }

    // --------------- mass-weight & solve ---------------
    // sqrtmmm / sqrtmmminv : [ndof], interleaved per-atom
    let mass_vec = mass.reshape(-1).to_vec();
    let mut sqrtmmm = Vec::with_capacity(ndof);
    let mut sqrtmmminv = Vec::with_capacity(ndof);
    for a in 0..natm {
        let sm = mass_vec[a].sqrt();
        for _ in 0..3 {
            sqrtmmm.push(sm);
            sqrtmmminv.push(1.0 / sm);
        }
    }
    let sqrtmmminv_t = rt::asarray((&sqrtmmminv, &device));

    // mwhess[i,j] = hess[i,j] / sqrt(m_i * m_j)
    // numpy: (sqrtmmminv[:,None] * nmwhess) * sqrtmmminv[None,:]
    // col-major: nmwhess * sqrtmmminv[:, None]  broadcasts axis-0 ; then * sqrtmmminv[None,:] axis-1
    let mwhess = (&nmwhess * sqrtmmminv_t.i((.., None))) * sqrtmmminv_t.i((None, ..));

    // project & diagonalise: mwhess_proj = P^T mwhess P  (P symmetric so P^T=P)
    let mwhess_proj = p.t() % (&mwhess % &p);

    let (fc_au, qL_raw): (Tsr, Tsr) = rt::linalg::eigh(mwhess_proj.view()).into();
    // eigh returns ascending eigenvalues already; eigenvectors are columns.

    // sort ascending (eigh already ascending, but be explicit/safe)
    let fc_vec = fc_au.reshape(-1).to_vec();
    let mut order: Vec<usize> = (0..ndof).collect();
    order.sort_by(|&a, &b| fc_vec[a].partial_cmp(&fc_vec[b]).unwrap());

    // reorder eigenvalues and eigenvector columns
    let fc_sorted: Vec<f64> = order.iter().map(|&i| fc_vec[i]).collect();
    let mut qL: Tsr = rt::zeros(([ndof, ndof], &device));
    for (new, &old) in order.iter().enumerate() {
        for r in 0..ndof {
            qL[[r, new]] = qL_raw[[r, old]];
        }
    }
    // phase convention
    let qL = phase_cols_to_max_element(qL.view(), 1.0e-2);

    // --------------- frequencies (complex sqrt) ---------------
    let uconv_cm = uconv_cm1();
    // omega = sqrt(fc) * uconv ; imaginary if fc < 0
    let mut omega = Vec::with_capacity(ndof);
    let mut imag = Vec::with_capacity(ndof);
    for &fc in fc_sorted.iter() {
        if fc < 0.0 {
            let mag = (-fc).sqrt() * uconv_cm;
            omega.push(mag); // store magnitude; imag flag set
            imag.push(true);
        } else {
            omega.push(fc.sqrt() * uconv_cm);
            imag.push(false);
        }
    }
    let omega_real: Vec<f64> = (0..ndof).map(|i| if imag[i] { 0.0 } else { omega[i] }).collect();

    // --------------- degeneracies (group by round(omega_real, 1)) ---------------
    // Note: Python uses round(frequency_cm_1, 1) on the complex array (rounds real part).
    let mut degeneracy = vec![0i64; ndof];
    {
        // group indices by rounded real frequency
        let mut keys: Vec<(i64, usize)> = (0..ndof).map(|i| ((omega_real[i] * 10.0).round() as i64, i)).collect();
        keys.sort_by_key(|&(k, _)| k);
        let mut start = 0;
        while start < keys.len() {
            let k = keys[start].0;
            let mut end = start + 1;
            while end < keys.len() && keys[end].0 == k {
                end += 1;
            }
            let count = (end - start) as i64;
            for j in start..end {
                degeneracy[keys[j].1] = count;
            }
            start = end;
        }
    }

    // --------------- TR / V classification ---------------
    // After projection P = I - Σ|tr⟩⟨tr|, the TR directions lie in null(P)
    // and have machine-zero force constants (~1e-13), separated from every
    // genuine vibrational eigenvalue (a 10 cm⁻¹ mode is already ~2e-9) by
    // many orders of magnitude.  So the nrt modes with smallest
    // |force_constant| are the TR modes; the rest are vibrations.  This is
    // more robust than the per-mode SVD membership test (vec_in_space) and
    // its arbitrary 1e-4 tolerance.
    let mut order_by_fc: Vec<usize> = (0..ndof).collect();
    order_by_fc.sort_by(|&a, &b| fc_sorted[a].abs().partial_cmp(&fc_sorted[b].abs()).unwrap());
    let mut is_tr = vec![false; ndof];
    for &i in order_by_fc.iter().take(nrt) {
        is_tr[i] = true;
    }
    let mut trv: Vec<&'static str> = Vec::with_capacity(ndof);
    for i in 0..ndof {
        if is_tr[i] {
            trv.push("TR");
        } else if fc_sorted[i].abs().sqrt() < 1.0e-3 {
            trv.push("-");
        } else {
            trv.push("V");
        }
    }

    // --------------- conversion factors ---------------
    let uconv_mdyne_a = 0.1 * (2.0 * std::f64::consts::PI * SPEED_OF_LIGHT).powi(2) / AVOGADRO;
    let uconv_S =
        ((SPEED_OF_LIGHT * (2.0 * std::f64::consts::PI * BOHR2ANG).powi(2)) / (PLANCK * AVOGADRO * 1.0e21)).sqrt();

    // --------------- normal modes & reduced mass ---------------
    // w = m^{-1/2} q  ;  w[a,i] = q[a,i] / sqrt(m_a)
    // numpy: sqrtmmminv[:,None] * q  (broadcast rows)
    // col-major: q * sqrtmmminv[:, None]
    let wL = &qL * sqrtmmminv_t.i((.., None));
    // column-wise L2 norm: l2_norm_axes(0) sums over axis 0 (rows) per column
    let w_norm = wL.l2_norm_axes(0); // [ndof]
    let w_norm_vec = w_norm.reshape(-1).to_vec();
    let mu: Vec<f64> = w_norm_vec.iter().map(|&n| 1.0 / (n * n)).collect();

    // x = sqrt(mu) * w  ;  numpy: sqrt(mu) * w  (mu is [ndof], broadcasts over columns/axis-1)
    // col-major: w * sqrt(mu)[None, :]  → broadcasts over axis 1
    let sqrt_mu: Vec<f64> = mu.iter().map(|&m| m.sqrt()).collect();
    let sqrt_mu_t = rt::asarray((&sqrt_mu, &device));
    let xL = &wL * sqrt_mu_t.i((None, ..));

    // --------------- force constants ---------------
    // k = mu * omega^2 * uconv_mdyne_a  (uses real part of omega)
    let k: Vec<f64> = (0..ndof).map(|i| mu[i] * omega_real[i] * omega_real[i] * uconv_mdyne_a).collect();

    // --------------- turning points (v=0) ---------------
    let tp_rnc = (2.0 * 0.0 + 1.0_f64).sqrt(); // = 1
    let mut qtp0 = vec![0.0_f64; ndof];
    let mut xtp0 = vec![0.0_f64; ndof];
    for i in 0..ndof {
        let denom_q = omega_real[i].sqrt() * uconv_S;
        qtp0[i] = if denom_q == 0.0 || !denom_q.is_finite() { 0.0 } else { tp_rnc / denom_q };
        let denom_x = (omega_real[i] * mu[i]).sqrt() * uconv_S;
        xtp0[i] = if denom_x == 0.0 || !denom_x.is_finite() { 0.0 } else { tp_rnc / denom_x };
    }
    let dq0: Vec<f64> = qtp0.iter().map(|&q| q / 2.0_f64.sqrt()).collect();

    // --------------- characteristic vibrational temperature ---------------
    let uconv_K = 100.0 * PLANCK * SPEED_OF_LIGHT / BOLTZMANN;
    let theta_vib: Vec<f64> = omega_real.iter().map(|&w| w * uconv_K).collect();

    let q_vec = qL.reshape(-1).to_vec();
    let w_vec = wL.reshape(-1).to_vec();
    let x_vec = xL.reshape(-1).to_vec();

    VibInfo { ndof, omega, imag, q: q_vec, w: w_vec, x: x_vec, degeneracy, trv, mu, k, dq0, qtp0, xtp0, theta_vib }
}

// ---------------------------------------------------------------------------
// Rotor type enum for thermo
// ---------------------------------------------------------------------------

/// Rotor classification for thermochemistry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotorType {
    Atom,
    Linear,
    Regular,
}

impl RotorType {
    /// Parse from a string (`"RT_ATOM"` / `"RT_LINEAR"` / `"ATOM"` / `"LINEAR"` /
    /// anything else ⇒ `Regular`).
    pub fn parse(s: &str) -> Self {
        match s {
            "RT_ATOM" | "ATOM" => RotorType::Atom,
            "RT_LINEAR" | "LINEAR" => RotorType::Linear,
            _ => RotorType::Regular,
        }
    }

    /// Infer from rotational constants [GHz] (uses [`get_rotor_type`]).
    pub fn from_rot_const_ghz(rot_const_ghz: TsrView) -> Self {
        match get_rotor_type(rot_const_ghz) {
            "ATOM" => RotorType::Atom,
            "LINEAR" => RotorType::Linear,
            _ => RotorType::Regular,
        }
    }
}

// ---------------------------------------------------------------------------
// ThermoInfo result container
// ---------------------------------------------------------------------------

/// Output of [`thermo`]. All energy/heat-capacity values in atomic units
/// (S/Cv/Cp in [mEh/K], ZPE/E/H/G in [Eh]).
#[derive(Clone, Debug)]
pub struct GauThermoInfo {
    // conditions
    pub e0: f64,
    pub b: [f64; 3],        // rotational constants [cm⁻¹]
    pub b_ghz: [f64; 3],    // rotational constants [GHz]
    pub rotor_type: String, // "ATOM" / "LINEAR" / "REGULAR"
    pub sigma: i64,
    pub t: f64,
    pub p: f64,
    // component contributions (each 4: elec, trans, rot, vib)
    pub s: [f64; 4], // [mEh/K]
    pub cv: [f64; 4],
    pub cp: [f64; 4],
    pub zpe: [f64; 4], // [Eh]; elec/trans/rot entries are 0 except vib
    pub e: [f64; 4],
    pub h: [f64; 4],
    pub g: [f64; 4],
    // totals
    pub s_tot: f64,
    pub cv_tot: f64,
    pub cp_tot: f64,
    pub zpe_corr: f64,
    pub e_corr: f64,
    pub h_corr: f64,
    pub g_corr: f64,
    pub zpe_tot: f64,
    pub e_tot: f64,
    pub h_tot: f64,
    pub g_tot: f64,
}

/// Indices for the four thermo components.
pub const ELEC: usize = 0;
pub const TRANS: usize = 1;
pub const ROT: usize = 2;
pub const VIB: usize = 3;

/// Thermochemical analysis from harmonic vibrational output.
///
/// # Parameters
/// - `vib` : [`VibInfo`] from [`harmonic_analysis`].
/// - `t` : temperature [K].
/// - `p` : pressure [Pa].
/// - `multiplicity` : spin multiplicity.
/// - `molecular_mass` : total molecular mass [u].
/// - `e0` : electronic energy at well bottom [Eh].
/// - `sigma` : rotational (external) symmetry number.
/// - `rot_const` : `[3]` rotational constants [cm⁻¹].
/// - `rotor_type` : [`RotorType`]; use `None`-equivalent by passing the result of
///   [`RotorType::from_rot_const_ghz`].
#[allow(clippy::too_many_arguments)]
pub fn thermo(
    vib: &VibInfo,
    t: f64,
    p: f64,
    multiplicity: i64,
    molecular_mass: f64,
    e0: f64,
    sigma: i64,
    rot_const: &[f64],
    rotor_type: RotorType,
) -> GauThermoInfo {
    // sm[(quantity, term)] before unit conversion: S/Cv/Cp unitless, ZPE/E/H/G in [K]
    let mut s = [0.0_f64; 4];
    let mut cv = [0.0_f64; 4];
    let mut cp = [0.0_f64; 4];
    let mut zpe = [0.0_f64; 4];
    let mut e = [0.0_f64; 4];
    let mut h = [0.0_f64; 4];
    let mut g = [0.0_f64; 4];

    // ---------- electronic ----------
    s[ELEC] = (multiplicity as f64).ln();

    // ---------- translational ----------
    let beta = 1.0 / (BOLTZMANN * t);
    let q_trans = (2.0 * std::f64::consts::PI * molecular_mass * AMU2KG / (beta * PLANCK * PLANCK)).powf(1.5)
        * AVOGADRO
        / (beta * p);
    s[TRANS] = 2.5 + (q_trans / AVOGADRO).ln();
    cv[TRANS] = 1.5;
    cp[TRANS] = 2.5;
    e[TRANS] = 1.5 * t;
    h[TRANS] = 2.5 * t;

    // ---------- rotational ----------
    match rotor_type {
        RotorType::Atom => {},
        RotorType::Linear => {
            let q_rot = 1.0 / (beta * (sigma as f64) * 100.0 * SPEED_OF_LIGHT * PLANCK * rot_const[1]);
            s[ROT] = 1.0 + q_rot.ln();
            cv[ROT] = 1.0;
            cp[ROT] = 1.0;
            e[ROT] = t;
        },
        RotorType::Regular => {
            let phi = [
                rot_const[0] * 100.0 * SPEED_OF_LIGHT * PLANCK / BOLTZMANN,
                rot_const[1] * 100.0 * SPEED_OF_LIGHT * PLANCK / BOLTZMANN,
                rot_const[2] * 100.0 * SPEED_OF_LIGHT * PLANCK / BOLTZMANN,
            ];
            let q_rot =
                std::f64::consts::PI.sqrt() * t.powf(1.5) / ((sigma as f64) * (phi[0] * phi[1] * phi[2]).sqrt());
            s[ROT] = 1.5 + q_rot.ln();
            cv[ROT] = 1.5;
            cp[ROT] = 1.5;
            e[ROT] = 1.5 * t;
        },
    }
    h[ROT] = e[ROT];

    // ---------- vibrational ----------
    // vib-only modes, exclude imaginary
    let vib_idx = vib.vib_indices();
    let mut filtered_theta: Vec<f64> = Vec::new();
    for &i in &vib_idx {
        if !vib.imag[i] {
            filtered_theta.push(vib.theta_vib[i]);
        }
    }
    let t_safe = t.max(1e-14);
    let rT: Vec<f64> = filtered_theta.iter().map(|&th| th / t_safe).collect();

    // S_vib = Σ [ rT/(e^rT - 1) - ln(1 - e^-rT) ]
    let s_vib: f64 = rT.iter().map(|&r| r / r.exp_m1() - (1.0 - (-r).exp()).ln()).sum();
    // Cv_vib = Σ [ e^rT * (rT/(e^rT - 1))^2 ]
    let cv_vib: f64 = rT
        .iter()
        .map(|&r| {
            let denom = r.exp_m1();
            r.exp() * (r / denom).powi(2)
        })
        .sum();
    let zpe_vib = rT.iter().sum::<f64>() * t / 2.0;
    let e_vib = zpe_vib + rT.iter().map(|&r| r * t / r.exp_m1()).sum::<f64>();

    s[VIB] = s_vib;
    cv[VIB] = cv_vib;
    cp[VIB] = cv_vib;
    zpe[VIB] = zpe_vib;
    e[VIB] = e_vib;
    h[VIB] = e_vib;

    // ---------- Gibbs: G = H - T*S ----------
    for i in 0..4 {
        g[i] = h[i] - t * s[i];
    }

    // ---------- convert to atomic units ----------
    let uconv_r_ehk = R_GAS / HARTREE2KJMOL; // R [Eh/K] (×1000 → mEh/K)
    for i in 0..4 {
        s[i] *= uconv_r_ehk; // [mEh/K]
        cv[i] *= uconv_r_ehk;
        cp[i] *= uconv_r_ehk;
        zpe[i] *= uconv_r_ehk * 0.001; // [Eh]
        e[i] *= uconv_r_ehk * 0.001;
        h[i] *= uconv_r_ehk * 0.001;
        g[i] *= uconv_r_ehk * 0.001;
    }

    // ---------- totals ----------
    let s_tot: f64 = s.iter().sum();
    let cv_tot: f64 = cv.iter().sum();
    let cp_tot: f64 = cp.iter().sum();
    let zpe_corr: f64 = zpe.iter().sum();
    let e_corr: f64 = e.iter().sum();
    let h_corr: f64 = h.iter().sum();
    let g_corr: f64 = g.iter().sum();
    let zpe_tot = e0 + zpe_corr;
    let e_tot = e0 + e_corr;
    let h_tot = e0 + h_corr;
    let g_tot = e0 + g_corr;

    let b_ghz = [rot_const[0] * 29.9792458, rot_const[1] * 29.9792458, rot_const[2] * 29.9792458];
    let rotor_str = match rotor_type {
        RotorType::Atom => "ATOM",
        RotorType::Linear => "LINEAR",
        RotorType::Regular => "REGULAR",
    };

    GauThermoInfo {
        e0,
        b: [rot_const[0], rot_const[1], rot_const[2]],
        b_ghz,
        rotor_type: rotor_str.to_string(),
        sigma,
        t,
        p,
        s,
        cv,
        cp,
        zpe,
        e,
        h,
        g,
        s_tot,
        cv_tot,
        cp_tot,
        zpe_corr,
        e_corr,
        h_corr,
        g_corr,
        zpe_tot,
        e_tot,
        h_tot,
        g_tot,
    }
}

// ---------------------------------------------------------------------------
// Pretty-printer (port of pyhessref.vib.print_vibs)
// ---------------------------------------------------------------------------

/// Which normal-coordinate definition to print.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormCo {
    /// mass-weighted (`q`)
    Q,
    /// un-mass-weighted (`w`)
    W,
    /// normalized un-mass-weighted (`x`, default)
    X,
}

impl NormCo {
    /// Returns a col-major `[ndof, ndof]` tensor view of the chosen normal
    /// coordinate, reconstructed from the flat `Vec<f64>` storage.
    fn select<'a>(&self, vib: &'a VibInfo) -> TsrView<'a> {
        let data: &[f64] = match self {
            NormCo::Q => &vib.q,
            NormCo::W => &vib.w,
            NormCo::X => &vib.x,
        };
        rt::asarray((data, [vib.ndof, vib.ndof], &DeviceBLAS::default()))
    }
}

/// Right-justify `s` to width `w`, padding left with spaces.
fn rjust(s: &str, w: usize) -> String {
    let len = s.chars().count();
    if len >= w {
        s.to_string()
    } else {
        format!("{}{}", " ".repeat(w - len), s)
    }
}

/// Left-justify `s` to width `w`, padding right with spaces.
fn ljust(s: &str, w: usize) -> String {
    let len = s.chars().count();
    if len >= w {
        s.to_string()
    } else {
        format!("{}{}", s, " ".repeat(w - len))
    }
}

/// Center `s` to width `w`.
fn center(s: &str, w: usize) -> String {
    let len = s.chars().count();
    if len >= w {
        s.to_string()
    } else {
        let total = w - len;
        let left = total / 2;
        let right = total - left;
        format!("{}{}{}", " ".repeat(left), s, " ".repeat(right))
    }
}

/// Pretty-print vibrational analysis results. Rust port of
/// `pyhessref.vib.print_vibs`.
///
/// Only vibrational modes (`TRV == "V"`) are printed.
///
/// # Parameters
/// - `vib` : [`VibInfo`] from [`harmonic_analysis`].
/// - `atom_lbl` : atomic symbols; if empty, integers are used.
/// - `normco` : which normal coordinate to print ([`NormCo`]).
/// - `shortlong` : `true` for `(nat, 3)` layout, `false` for `(3*nat, 1)`.
/// - `groupby` : modes per row (`Some(n)`); `None` ⇒ 3 (short) / 6 (long); `Some(usize::MAX)` is
///   treated as "all" (clamped to active count).
/// - `prec` : decimal places for scalar properties.
/// - `ncprec` : decimal places for normal coordinates (`None` ⇒ 2 short / 4 long).
pub fn print_vibs(
    vib: &VibInfo,
    atom_lbl: &[&str],
    normco: NormCo,
    shortlong: bool,
    groupby: Option<usize>,
    prec: usize,
    ncprec: Option<usize>,
) -> String {
    let nat = vib.ndof / 3;
    let active: Vec<usize> = (0..vib.ndof).filter(|&i| vib.trv[i] == "V").collect();

    // print inactive modes (TR) first, if any
    println!("Inactive modes:");
    let inactive: Vec<usize> = (0..vib.ndof).filter(|&i| vib.trv[i] != "V").collect();
    for &i in &inactive {
        let trv = vib.trv[i];
        let freq = if vib.imag[i] { format!("{:.4}i", vib.omega[i]) } else { format!("{:.4}", vib.omega[i]) };
        println!("{:<2}  Freq [cm^-1]: {}", trv, freq);
    }
    println!("");
    println!("Active modes:");

    let presp = 2;
    let prewidth = 24usize;
    let colsp = 2usize;
    let (groupby, ncprec, width) = if shortlong {
        let g = groupby.unwrap_or(3);
        let n = ncprec.unwrap_or(2);
        (g, n, (n + 4) * 3)
    } else {
        let g = groupby.unwrap_or(6);
        let n = ncprec.unwrap_or(4);
        (g, n, n + 8)
    };
    let groupby = if groupby == usize::MAX { active.len().max(1) } else { groupby };

    let normco_t = normco.select(vib);

    let omega_str: Vec<String> = (0..vib.ndof)
        .map(|i| if vib.imag[i] { format!("{:.*}i", prec, vib.omega[i]) } else { format!("{:.*}", prec, vib.omega[i]) })
        .collect();

    let mut lines: Vec<String> = Vec::new();

    // running 1-based vibrational mode number (renumbered starting from 1)
    let mut vib_num = 1usize;
    let mut iter = active.iter();
    loop {
        let mut row: Vec<usize> = Vec::new();
        for _ in 0..groupby {
            match iter.next() {
                Some(&v) => row.push(v),
                None => break,
            }
        }
        if row.is_empty() {
            break;
        }

        // Vibration header — centered integer, colsp trailing
        let mut line = format!("{}{}", " ".repeat(presp), ljust("Vibration", prewidth));
        for &_ in &row {
            line.push_str(&format!("{}{}", center(&vib_num.to_string(), width), " ".repeat(colsp)));
            vib_num += 1;
        }
        lines.push(line);

        // Freq — centered, 2 trailing spaces (= colsp)
        let mut line = format!("{}{}", " ".repeat(presp), ljust("Freq [cm^-1]", prewidth));
        for &vib in &row {
            line.push_str(&format!("{}  ", center(&omega_str[vib], width)));
        }
        lines.push(line);

        // Irrep — skip (no irrep/symmetry in our implementation)
        // scalar property rows — centered, colsp trailing
        let labels: [&str; 5] = [
            "Reduced mass [u]",
            "Force const [mDyne/A]",
            "Turning point v=0 [a0]",
            "RMS dev v=0 [a0 u^1/2]",
            "Char temp [K]",
        ];
        let val_slices: [&[f64]; 5] = [&vib.mu, &vib.k, &vib.xtp0, &vib.dq0, &vib.theta_vib];
        for (label, vals) in labels.iter().zip(val_slices.iter()) {
            let mut l = format!("{}{}", " ".repeat(presp), ljust(label, prewidth));
            for &vib in &row {
                l.push_str(&format!("{}{}", center(&format!("{:.*}", prec, vals[vib]), width), " ".repeat(colsp)));
            }
            lines.push(l);
        }

        // separator
        let sep_len = prewidth + groupby * (width + colsp) - colsp;
        lines.push(format!("{}{}", " ".repeat(presp), "-".repeat(sep_len)));

        // normal coordinate values
        if shortlong {
            let cell = width / 3;
            for at in 0..nat {
                let lbl = if at < atom_lbl.len() { atom_lbl[at] } else { "" };
                let mut l = format!("{}{:5}   {}", " ".repeat(presp), at + 1, ljust(lbl, prewidth - 8));
                for &vib in &row {
                    let vx = normco_t[[3 * at, vib]];
                    let vy = normco_t[[3 * at + 1, vib]];
                    let vz = normco_t[[3 * at + 2, vib]];
                    for v in [vx, vy, vz] {
                        l.push_str(&center(&format!("{:.*}", ncprec, v), cell));
                    }
                    l.push_str(&" ".repeat(colsp));
                }
                lines.push(l);
            }
        } else {
            for at in 0..nat {
                let lbl = if at < atom_lbl.len() { atom_lbl[at] } else { "" };
                for xyz in 0..3 {
                    let axis = ['X', 'Y', 'Z'][xyz];
                    let mut l =
                        format!("{}{:5}    {}    {}", " ".repeat(presp), at + 1, axis, ljust(lbl, prewidth - 14));
                    for &vib in &row {
                        let v = normco_t[[3 * at + xyz, vib]];
                        l.push_str(&format!("{}{}", center(&format!("{:.*}", ncprec, v), width), " ".repeat(colsp)));
                    }
                    lines.push(l);
                }
            }
        }
        lines.push("".to_string()); // blank line after each group
    }

    lines.join("\n")
}

// ---------------------------------------------------------------------------
// Thermochemistry pretty-printer (port of Psi4 thermo display)
// ---------------------------------------------------------------------------

/// Pretty-print thermochemistry results. Rust port of Psi4's `thermo` display
/// section. Uses the same three-column layout (cal/(mol K) / J/(mol K) / mEh/K
/// for S/Cv/Cp; kcal/mol / kJ/mol / Eh for E/H/G/ZPE).
///
/// Not required to be byte-identical to Psi4, but sufficiently similar.
pub fn print_gau_thermo(th: &GauThermoInfo, multiplicity: i64, molecular_mass: f64) -> String {
    let t = th.t;
    let p = th.p;
    let sigma = th.sigma;
    let e0 = th.e0;

    // helper: format S/Cv/Cp row
    let fmt_scv = |label: &str, val_mEhK: f64| -> String {
        format!(
            "  {:<36}{:11.3} [cal/(mol K)]  {:11.3} [J/(mol K)]  {:15.8} [mEh/K]",
            label,
            val_mEhK * HARTREE2KCALMOL,
            val_mEhK * HARTREE2KJMOL,
            val_mEhK,
        )
    };
    // helper: format E/H/G/ZPE row
    let fmt_ehg = |label: &str, val_Eh: f64| -> String {
        format!(
            "  {:<36}{:11.3} [kcal/mol]  {:11.3} [kJ/mol]  {:15.8} [Eh]",
            label,
            val_Eh * HARTREE2KCALMOL,
            val_Eh * HARTREE2KJMOL,
            val_Eh,
        )
    };

    let component_names = ["Electronic", "Translational", "Rotational", "Vibrational"];

    let mut lines: Vec<String> = Vec::new();

    // ---- header ----
    lines.push("\n  ==> Thermochemistry Components <==".to_string());

    // ---- rotational constants ----
    lines.push(format!("\n  Rotational constants  {:11.6}  {:11.6}  {:11.6} [cm^-1]", th.b[0], th.b[1], th.b[2],));
    lines.push(format!(
        "                        {:11.4}  {:11.4}  {:11.4} [GHz]",
        th.b_ghz[0], th.b_ghz[1], th.b_ghz[2],
    ));
    lines.push(format!("  Rotor type             {}", th.rotor_type));

    // ---- Entropy S ----
    lines.push("\n\n  Entropy, S".to_string());
    for i in 0..4 {
        let mut l = fmt_scv(&format!("  {} S", component_names[i]), th.s[i]);
        match i {
            ELEC => l.push_str(&format!(" (multiplicity = {})", multiplicity)),
            TRANS => l.push_str(&format!(" (mol. weight = {:.4} [u], P = {:.2} [Pa])", molecular_mass, p)),
            ROT => l.push_str(&format!(" (symmetry no. = {})", sigma)),
            _ => {},
        }
        lines.push(l);
    }
    lines.push(fmt_scv("Total S", th.s_tot));
    lines.push(fmt_scv("Correction S", th.s_tot));

    // ---- Cv ----
    lines.push("\n\n  Constant volume heat capacity, Cv".to_string());
    for i in 0..4 {
        lines.push(fmt_scv(&format!("  {} Cv", component_names[i]), th.cv[i]));
    }
    lines.push(fmt_scv("Total Cv", th.cv_tot));
    lines.push(fmt_scv("Correction Cv", th.cv_tot));

    // ---- Cp ----
    lines.push("\n\n  Constant pressure heat capacity, Cp".to_string());
    for i in 0..4 {
        lines.push(fmt_scv(&format!("  {} Cp", component_names[i]), th.cp[i]));
    }
    lines.push(fmt_scv("Total Cp", th.cp_tot));
    lines.push(fmt_scv("Correction Cp", th.cp_tot));

    // ---- Energy Analysis ----
    lines.push("\n\n  ==> Thermochemistry Energy Analysis <==".to_string());

    // raw E_e
    let raw_e_e_label = "  Total E_e, Electronic energy at well bottom";
    lines.push("\n\n  Raw electronic energy, E_e".to_string());
    lines.push(format!("{}  {:>15.8} [Eh]", rjust(raw_e_e_label, 85), e0));

    // ---- ZPVE ----
    lines.push("\n\n  Zero-point vibrational energy, ZPVE = Sum_i omega_i / 2,  E_0 = E_e + ZPVE".to_string());
    {
        let mut l = fmt_ehg("  Vibrational ZPVE", th.zpe[VIB]);
        l.push_str(&format!(" {:15.3} [cm^-1]", th.zpe[VIB] * HARTREE2WAVENUMBERS));
        lines.push(l);
    }
    {
        let mut l = fmt_ehg("  Correction ZPVE to E_e", th.zpe_corr);
        l.push_str(&format!(" {:15.3} [cm^-1]", th.zpe_corr * HARTREE2WAVENUMBERS));
        lines.push(l);
    }
    let zpe_tot_label = "  Total E_0, Enthalpy at 0 [K]";
    lines.push(format!("{}  {:>15.8} [Eh]", rjust(zpe_tot_label, 85), th.zpe_tot));
    lines.push("  *** Absolute enthalpy, not an enthalpy of formation ***".to_string());

    // ---- Thermal energy E ----
    lines.push("\n\n  Thermal (internal) energy, E (includes ZPVE and finite-temperature corrections)".to_string());
    for i in 0..4 {
        let label = if i == ELEC {
            format!("  {} contrib to E beyond E_e", component_names[i])
        } else {
            format!("  {} contrib to E", component_names[i])
        };
        lines.push(fmt_ehg(&label, th.e[i]));
    }
    lines.push(fmt_ehg("  Correction E", th.e_corr));
    lines.push(format!(
        "  Total E, Thermal (internal) energy at {:7.2} [K]{}  {:>15.8} [Eh]",
        t,
        " ".repeat(47),
        th.e_tot
    ));

    // ---- Enthalpy H ----
    lines.push("\n\n  Enthalpy, H_trans = E_trans + k_B * T = E_trans + P * V".to_string());
    for i in 0..4 {
        let label = if i == ELEC {
            format!("  {} contrib to H beyond E_e", component_names[i])
        } else {
            format!("  {} contrib to H", component_names[i])
        };
        lines.push(fmt_ehg(&label, th.h[i]));
    }
    lines.push(fmt_ehg("  Correction H", th.h_corr));
    lines.push(format!("  Total H, Enthalpy at {:7.2} [K]{}  {:>15.8} [Eh]", t, " ".repeat(53), th.h_tot));
    lines.push("  *** Absolute enthalpy, not an enthalpy of formation ***".to_string());

    // ---- Gibbs G ----
    lines.push("\n\n  Gibbs free energy, G = H - T * S".to_string());
    for i in 0..4 {
        let label = if i == ELEC {
            format!("  {} contrib to G beyond E_e", component_names[i])
        } else {
            format!("  {} contrib to G", component_names[i])
        };
        lines.push(fmt_ehg(&label, th.g[i]));
    }
    lines.push(fmt_ehg("  Correction G", th.g_corr));
    lines.push(format!("  Total G, Gibbs energy at {:7.2} [K]{}  {:>15.8} [Eh]\n", t, " ".repeat(53), th.g_tot));

    lines.join("\n")
}
